//! The red chroma step (`docs/02` C12 step 1c) on the GPU: the colour kernel against
//! `ym_color::display::ColorChain`.
//!
//! * C12 and C13 of chains with the step at several pairs of strengths (the look's `up` and `dark_relief`),
//!   for the Rec.709 and the P3 limiting gamut, on an image of reds (stops, hues and chromas across the
//!   window and its tails) with colours elsewhere, in every storage mode, against the CPU trace;
//! * the step's own part: the device's change (step on minus step off) against the CPU's;
//! * the yellow window (the up side only): ochres and yellows move, the dark side never touches them;
//! * the DNG stop: C11 decoded, lifted and through the step with the P3-D65 tables whatever the output,
//!   against `ColorChain::apply_dng_pixel`;
//! * both strengths 0 (a look such as Signature), or the step off, are the exact plain ACES 2 on the device,
//!   bit for bit, and so is the real `signature.json`;
//! * grey, skin hues and the hues outside both windows are untouched on the device, bit for bit;
//! * tiles of a render equal the whole image.
//!
//! **Stated errors** (lavapipe, f32 against the f64 reference; Metal gets its own column in the CI log):
//! C12 display-linear 5e-4 relative of the largest channel (the gate of `gpu_color.rs`), C13 one 10-bit
//! code; the step's own part 1e-3 of display white (absolute); the DNG stop 2e-4 relative (the gate of
//! `gpu_color_white_lift.rs`). The measured numbers are printed.

mod common;

use std::sync::Arc;

use common::chains::{self, camera_image, pixel};
use common::gpus;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::deltae::{lab_to_xyz, xyz_to_lab};
use ym_color::display::red_chroma::{GREY, LAB_WHITE, S_MAX, YD_FLOOR};
use ym_color::display::{
    ChainParams, ColorChain, LookSetting, OutputTarget, RedChroma, RedChromaTables, RedGamut,
};
use ym_color::gamut::matrices;
use ym_color::look::{LookTransform, ParametricLook};
use ym_gpu::cpu::RgbImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;
/// Relative to the largest channel of C12 (the gate of `gpu_color.rs`).
const C12_TOLERANCE: f64 = 5e-4;
/// The step's own part, absolute, display white = 1.
const STEP_TOLERANCE: f64 = 1e-3;
/// The DNG stop, relative to `max(|v|, 0.01)`.
const DNG_TOLERANCE: f64 = 2e-4;

const SIGNATURE_JSON: &str = include_str!("../../../data/looks/builtin/signature.json");

fn render(
    dev: &common::TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    outputs: &[ColorOutput],
) -> Vec<ColorImage> {
    let chain = chain.clone();
    let img = img.clone();
    let outputs = outputs.to_vec();
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let prepared = PreparedChain::new(ctx, &chain)?;
            outputs
                .iter()
                .map(|o| prepared.render(ctx, &img, (0, 0), *o))
                .collect()
        })
        .expect("GPU render")
}

fn planar(img: &ColorImage) -> &[f32] {
    match &img.data {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

fn identity_matrix() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// A linear AWG4 colour at `s` stops over grey whose Lab at the display luminance of the neutral axis has
/// hue `h` (degrees) and chroma `c`.
fn awg4_at(tables: &RedChromaTables, s: f64, h: f64, c: f64) -> Vec3 {
    let yd = tables.yd_at(s.min(S_MAX)).clamp(YD_FLOOR, 1.0);
    let l = xyz_to_lab(LAB_WHITE.map(|v| v * yd), LAB_WHITE)[0];
    let hr = h.to_radians();
    let rel = lab_to_xyz([l, c * hr.cos(), c * hr.sin()], LAB_WHITE);
    let y = GREY * s.exp2();
    let m = &matrices().xyz_to_awg4;
    let xyz = rel.map(|v| v / yd * y);
    [
        m[0][0] * xyz[0] + m[0][1] * xyz[1] + m[0][2] * xyz[2],
        m[1][0] * xyz[0] + m[1][1] * xyz[1] + m[1][2] * xyz[2],
        m[2][0] * xyz[0] + m[2][1] * xyz[1] + m[2][2] * xyz[2],
    ]
}

/// 48 pixels per row (12 stops, 4 chromas), one row per hue (the windows, their tails and elsewhere), as AWG4 for the
/// identity-matrix chain (f16-exact, so every storage mode reads the same numbers).
fn red_image(gamut: RedGamut, hues: &[f64]) -> RgbImage {
    let tables = RedChromaTables::shared(gamut);
    let stops = [
        -6.5, -5.0, -3.0, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 3.0, 4.5, 6.5,
    ];
    let chromas = [6.0, 18.0, 35.0, 60.0];
    let (w, h) = (stops.len() * chromas.len(), hues.len());
    let mut rgb = Vec::with_capacity(3 * w * h);
    for &hue in hues {
        for &s in &stops {
            for &c in &chromas {
                for v in awg4_at(tables, s, hue, c) {
                    rgb.push(common::f16_round(v as f32));
                }
            }
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

/// The hues of the image: the red window and its tails, the skin band between the windows (45 to 60), the
/// yellow window and its tails (62 to 110), and elsewhere.
const WINDOW_HUES: [f64; 22] = [
    -35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 60.0, 64.0, 66.0, 70.0, 80.0,
    90.0, 100.0, 106.0, 112.0, 120.0, 200.0, 280.0,
];

fn chain_with(output: OutputTarget, strength: Option<RedChroma>) -> ColorChain {
    let mut p = ChainParams::new(identity_matrix(), chains::identity_tone());
    p.output = output;
    match strength {
        Some(r) => {
            let look = LookTransform::neutral().with_red_chroma(r);
            p.look = Some(LookSetting::new(Arc::new(look), 100.0));
        }
        None => p.red_chroma = false,
    }
    ColorChain::new(&p).expect("chain")
}

#[test]
fn c12_and_c13_match_the_cpu_and_the_step_acts() {
    for (output, gamut) in [
        (OutputTarget::Viewport, RedGamut::P3D65),
        (OutputTarget::Srgb, RedGamut::Rec709),
    ] {
        let img = red_image(gamut, &WINDOW_HUES);
        let n = img.width * img.height;
        let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
        let off = chain_with(output, None);
        let cpu_off: Vec<Vec3> = cams
            .iter()
            .map(|&c| off.trace(c).c12_display_linear)
            .collect();
        for dev in gpus() {
            let gpu_off = render(dev, &off, &img, &[ColorOutput::Stage(Stop::C12)]);
            let gpu_off = planar(&gpu_off[0]);
            let mut worst = (0.0f64, 0.0f64, 0.0f64);
            let mut worst_dng = 0.0f64;
            for (up, dn) in [(1.0, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 1.0)] {
                let chain = chain_with(output, Some(RedChroma::new(up, dn).expect("strengths")));
                assert_eq!(
                    chain.red_chroma().map(|r| (r.up(), r.dark_relief())),
                    Some((up, dn))
                );
                let out = render(
                    dev,
                    &chain,
                    &img,
                    &[
                        ColorOutput::Stage(Stop::C12),
                        ColorOutput::Stage(Stop::C13),
                        ColorOutput::Stage(Stop::DngLinearAwg4),
                    ],
                );
                let (c12, c13, dng) = (planar(&out[0]), planar(&out[1]), planar(&out[2]));
                let mut acting = 0;
                for i in 0..n {
                    let t = chain.trace(cams[i]);
                    let top = t
                        .c12_display_linear
                        .iter()
                        .copied()
                        .fold(0.0f64, f64::max)
                        .max(0.02);
                    let mut moved = false;
                    let want_dng = chain.apply_dng_pixel(cams[i]);
                    for ch in 0..3 {
                        let e = (f64::from(dng[ch * n + i]) - want_dng[ch]).abs()
                            / want_dng[ch].abs().max(0.01);
                        worst_dng = worst_dng.max(e);
                        assert!(
                            e <= DNG_TOLERANCE,
                            "{output:?} {:?} up {up} dn {dn} DNG {i}/{ch}: {} vs {}",
                            dev.mode,
                            dng[ch * n + i],
                            want_dng[ch]
                        );
                        let got = f64::from(c12[ch * n + i]);
                        let e = (got - t.c12_display_linear[ch]).abs();
                        worst.0 = worst.0.max(e / top);
                        assert!(
                            e <= C12_TOLERANCE * top,
                            "{output:?} {:?} up {up} dn {dn} C12 {i}/{ch}: {got} vs {} (camera {:?})",
                            dev.mode,
                            t.c12_display_linear[ch],
                            cams[i]
                        );
                        let e13 = (f64::from(c13[ch * n + i]) - t.c13_encoded[ch]).abs();
                        worst.1 = worst.1.max(e13);
                        assert!(
                            e13 <= CODE10,
                            "{output:?} {:?} C13 {i}/{ch}: {e13:.2e}",
                            dev.mode
                        );
                        // The step's own part: its change on the device against its change on the CPU.
                        let want_delta = t.c12_display_linear[ch] - cpu_off[i][ch];
                        let got_delta = got - f64::from(gpu_off[ch * n + i]);
                        let ed = (got_delta - want_delta).abs();
                        worst.2 = worst.2.max(ed);
                        assert!(
                            ed <= STEP_TOLERANCE,
                            "{output:?} {:?} up {up} dn {dn} step {i}/{ch}: {got_delta} vs {want_delta}",
                            dev.mode
                        );
                        moved |= want_delta.abs() > 1e-3;
                    }
                    acting += usize::from(moved);
                }
                // Not vacuous: the step changes a good share of the red pixels.
                if up > 0.0 {
                    assert!(
                        acting * 10 > n,
                        "{output:?} up {up}: {acting} of {n} pixels move"
                    );
                }
            }
            eprintln!(
                "{output:?} {:?}: red chroma step, worst C12 {:.1e} (relative to the largest channel), C13 {:.1e} (code values), step's own part {:.1e} (display white = 1), DNG stop {worst_dng:.1e} (relative)",
                dev.mode, worst.0, worst.1, worst.2
            );
        }
    }
}

#[test]
fn both_strengths_zero_or_the_step_off_is_plain_aces2_on_the_device() {
    let img = red_image(RedGamut::P3D65, &WINDOW_HUES);
    let stages = [
        ColorOutput::Stage(Stop::C12),
        ColorOutput::Stage(Stop::C13),
        ColorOutput::Stage(Stop::DngLinearAwg4),
    ];
    let off = chain_with(OutputTarget::Viewport, None);
    let zero = chain_with(OutputTarget::Viewport, Some(RedChroma::OFF));
    assert_eq!(zero.red_chroma(), None);
    assert_eq!(off.red_chroma(), None);
    // The real Signature look: both strengths 0 in its file, so it renders as plain ACES 2.
    let signature = ParametricLook::from_json(SIGNATURE_JSON)
        .expect("signature.json")
        .to_transform()
        .expect("transform");
    assert!(signature.red_chroma().is_off());
    let mut p = ChainParams::new(identity_matrix(), chains::identity_tone());
    p.output = OutputTarget::Viewport;
    p.look = Some(LookSetting::new(Arc::new(signature.clone()), 100.0));
    let sig_on = ColorChain::new(&p).expect("chain");
    p.red_chroma = false;
    let sig_off = ColorChain::new(&p).expect("chain");
    assert_eq!(sig_on.red_chroma(), None);
    let photo = camera_image(61, 37, 0x5167);
    for dev in gpus() {
        for (name, a, b, image) in [
            ("OFF look", &zero, &off, &img),
            ("Signature", &sig_on, &sig_off, &img),
            ("Signature on a photo", &sig_on, &sig_off, &photo),
        ] {
            let (x, y) = (
                render(dev, a, image, &stages),
                render(dev, b, image, &stages),
            );
            for (p, q) in x.iter().zip(&y) {
                let (p, q) = (planar(p), planar(q));
                assert!(
                    p.iter().zip(q).all(|(u, v)| u.to_bits() == v.to_bits()),
                    "{:?}: {name} is not plain ACES 2",
                    dev.mode
                );
            }
        }
    }
    // And on the CPU, bit for bit, for the same pixels (the picture and the DNG).
    for i in 0..(photo.width * photo.height) {
        let c = pixel(&photo, i);
        assert_eq!(
            sig_on.apply_to_display_linear(c).map(f64::to_bits),
            sig_off.apply_to_display_linear(c).map(f64::to_bits)
        );
        assert_eq!(
            sig_on.apply_dng_pixel(c).map(f64::to_bits),
            sig_off.apply_dng_pixel(c).map(f64::to_bits)
        );
    }
}

#[test]
fn grey_and_the_hues_outside_the_window_are_untouched_on_the_device() {
    let on = chain_with(OutputTarget::Viewport, Some(RedChroma::DEFAULT));
    let off = chain_with(OutputTarget::Viewport, None);
    // Outside both windows and their tails (reds -27 to 43 degrees, yellows 62 to 110): the skin band
    // between them (44 to 61) and everything from 111 to 333.
    let outside: Vec<f64> = vec![
        44.0, 48.0, 52.0, 57.0, 61.0, 111.0, 125.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
    ];
    let mut img_pixels = red_image(RedGamut::P3D65, &outside);
    // A grey ramp in the first row's place: every stop from -9 to +8, every pixel r = g = b.
    let w = img_pixels.width;
    for x in 0..w {
        let v = common::f16_round((GREY * (-9.0 + 17.0 * x as f64 / (w - 1) as f64).exp2()) as f32);
        img_pixels.set_pixel(x, 0, [v, v, v]);
    }
    let stages = [ColorOutput::Stage(Stop::C12), ColorOutput::Stage(Stop::C13)];
    for dev in gpus() {
        let (a, b) = (
            render(dev, &on, &img_pixels, &stages),
            render(dev, &off, &img_pixels, &stages),
        );
        for (p, q) in a.iter().zip(&b) {
            let (p, q) = (planar(p), planar(q));
            let same = p
                .iter()
                .zip(q)
                .filter(|(u, v)| u.to_bits() == v.to_bits())
                .count();
            assert_eq!(
                same,
                p.len(),
                "{:?}: {} samples moved",
                dev.mode,
                p.len() - same
            );
        }
    }
}

/// The yellow window acts on the up side only, on the device as on the CPU: with the dark relief alone, or
/// below the cross at -0.5 stop, the yellow rows are the plain ACES 2 bit for bit; with the up side they move.
#[test]
fn yellows_move_on_the_up_side_only_on_the_device() {
    let gamut = RedGamut::P3D65;
    let hues = [68.0, 80.0, 95.0];
    let img = red_image(gamut, &hues);
    let (w, rows) = (img.width, img.height);
    // The stops of `red_image` are 4 pixels each (4 chromas): the first 5 stops are below -0.5.
    let stops_below_cross = 5 * 4;
    let off = chain_with(OutputTarget::Viewport, None);
    let up = chain_with(OutputTarget::Viewport, Some(RedChroma::new(1.0, 0.0).expect("up")));
    let dn = chain_with(OutputTarget::Viewport, Some(RedChroma::new(0.0, 1.0).expect("dn")));
    let stage = [ColorOutput::Stage(Stop::C12)];
    for dev in gpus() {
        let base = render(dev, &off, &img, &stage);
        let with_up = render(dev, &up, &img, &stage);
        let with_dn = render(dev, &dn, &img, &stage);
        let (base, with_up, with_dn) = (planar(&base[0]), planar(&with_up[0]), planar(&with_dn[0]));
        let n = w * rows;
        let mut moved_up = 0;
        for row in 0..rows {
            for x in 0..w {
                let i = row * w + x;
                for ch in 0..3 {
                    let j = ch * n + i;
                    assert_eq!(
                        with_dn[j].to_bits(),
                        base[j].to_bits(),
                        "{:?}: dark relief moved a yellow ({row}, {x})",
                        dev.mode
                    );
                    if x < stops_below_cross {
                        assert_eq!(
                            with_up[j].to_bits(),
                            base[j].to_bits(),
                            "{:?}: the up side moved a dark yellow ({row}, {x})",
                            dev.mode
                        );
                    } else if (with_up[j] - base[j]).abs() > 1e-3 {
                        moved_up += 1;
                    }
                }
            }
        }
        assert!(moved_up > 60, "{:?}: the up side moved {moved_up} samples", dev.mode);
    }
}

#[test]
fn tiles_equal_the_whole_image() {
    let gamut = RedGamut::P3D65;
    let img = red_image(gamut, &WINDOW_HUES);
    let (w, h) = (img.width, img.height);
    let chain = chain_with(
        OutputTarget::Viewport,
        Some(RedChroma::new(1.0, 0.5).expect("strengths")),
    );
    for dev in gpus() {
        let whole = render(dev, &chain, &img, &[ColorOutput::Stage(Stop::C12)]);
        let whole = planar(&whole[0]);
        for tile in [7usize, 16, 25] {
            let plan = TilePlan::with_halo(w, h, tile, 0);
            let (chain2, img2) = (chain.clone(), img.clone());
            let tiles = dev
                .gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let prepared = PreparedChain::new(ctx, &chain2)?;
                    let mut collected = Vec::new();
                    prepared.render_tiled(
                        ctx,
                        &img2,
                        &plan,
                        ColorOutput::Stage(Stop::C12),
                        2,
                        token,
                        &mut |t, im| {
                            collected.push((t, im));
                            Ok(())
                        },
                    )?;
                    Ok(collected)
                })
                .expect("tiled render");
            for (t, im) in &tiles {
                let d = planar(im);
                let c = t.core;
                for ch in 0..3 {
                    for y in 0..c.h {
                        for x in 0..c.w {
                            let a = d[ch * c.w * c.h + y * c.w + x];
                            let b = whole[ch * w * h + (c.y + y) * w + c.x + x];
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{:?} tile {tile}: ({}, {}) differs from the whole image",
                                dev.mode,
                                c.x + x,
                                c.y + y
                            );
                        }
                    }
                }
            }
        }
    }
}
