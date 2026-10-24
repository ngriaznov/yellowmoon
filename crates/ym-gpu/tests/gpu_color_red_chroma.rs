//! The red chroma step (`docs/02` C12 step 1c) on the GPU: the colour kernel against
//! `ym_color::display::ColorChain`.
//!
//! * C12 and C13 of chains with the step at several pairs of strengths (the look's `up` and `dark_relief`),
//!   for the Rec.709 and the P3 limiting gamut, on an image of reds (stops, hues and chromas across the
//!   window and its tails) with colours elsewhere, in every storage mode, against the CPU trace;
//! * the step's own part: the device's change (step on minus step off) against the CPU's;
//! * both strengths 0 (a look such as Signature), or the step off, are the exact plain ACES 2 on the device,
//!   bit for bit, and so is the real `signature.json`;
//! * grey and the hues outside the window are untouched on the device, bit for bit;
//! * tiles of a render equal the whole image.
//!
//! **Stated errors** (lavapipe, f32 against the f64 reference; Metal gets its own column in the CI log):
//! C12 display-linear 5e-4 relative of the largest channel (the gate of `gpu_color.rs`), C13 one 10-bit
//! code; the step's own part 1e-3 of display white (absolute). The measured numbers are printed.

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

/// 48 x 13 pixels: 12 stops, 13 hues (the window, its tails and elsewhere), 4 chromas, as AWG4 for the
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

const WINDOW_HUES: [f64; 13] = [
    -35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 120.0, 200.0, 280.0,
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
                    &[ColorOutput::Stage(Stop::C12), ColorOutput::Stage(Stop::C13)],
                );
                let (c12, c13) = (planar(&out[0]), planar(&out[1]));
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
                    for ch in 0..3 {
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
                "{output:?} {:?}: red chroma step, worst C12 {:.1e} (relative to the largest channel), C13 {:.1e} (code values), step's own part {:.1e} (display white = 1)",
                dev.mode, worst.0, worst.1, worst.2
            );
        }
    }
}

#[test]
fn both_strengths_zero_or_the_step_off_is_plain_aces2_on_the_device() {
    let img = red_image(RedGamut::P3D65, &WINDOW_HUES);
    let stages = [ColorOutput::Stage(Stop::C12), ColorOutput::Stage(Stop::C13)];
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
    // And on the CPU, bit for bit, for the same pixels.
    for i in 0..(photo.width * photo.height) {
        let c = pixel(&photo, i);
        assert_eq!(
            sig_on.apply_to_display_linear(c).map(f64::to_bits),
            sig_off.apply_to_display_linear(c).map(f64::to_bits)
        );
    }
}

#[test]
fn grey_and_the_hues_outside_the_window_are_untouched_on_the_device() {
    let on = chain_with(OutputTarget::Viewport, Some(RedChroma::DEFAULT));
    let off = chain_with(OutputTarget::Viewport, None);
    // Hues 60 to 320 are outside the window and its tails (-27 to 43 degrees) by 17 degrees or more.
    let outside: Vec<f64> = (0..13).map(|i| 60.0 + 20.0 * f64::from(i)).collect();
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
