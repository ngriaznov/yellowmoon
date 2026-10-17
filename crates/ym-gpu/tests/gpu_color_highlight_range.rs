//! C7's highlight range (`docs/02` C7) on the GPU: the colour kernel against
//! `ym_color::tone::HighlightRange` and `ym_color::display::ColorChain`.
//!
//! * the gain alone (stop C7 of a chain with an identity matrix and no toe): the ratio of output to
//!   input against `HighlightRange::gain` at the input's luma;
//! * C7, C8, C11, the DNG stop, C12 and C13 of the range chains (`chains::range_cases`: r 0, 0.5 and 1
//!   at the Auto clip, everything on, a clip inside the fade, a clip over the cap, a display-baked look,
//!   a display LUT) against the CPU trace, in every storage mode;
//! * no bit changes below the knee (r = 1 against r = 0 at the same clip and toe), and r = 0 changes no
//!   bit below the cap (against C7 off);
//! * a range that moves nothing (r = 0, a NaN amount, a clip where the fade is 0, a clip at or over the
//!   ALEXA's) leaves every output bit-identical to C7 off, as the colour warp's test does for pins;
//! * tiles of a render with the range equal the whole image, bit for bit.
//!
//! **Stated errors**, measured on Mesa lavapipe (f32 maths against the f64 reference), the same in all
//! three storage modes; Metal gets its own column in the CI log:
//! * the gain, relative: 2.5e-6 (tolerance [`GAIN_TOLERANCE`], 3e-5; the steepest set, the clip 2 stops
//!   over grey at r = 1, multiplies the error of the f32 `log2` by up to 12);
//! * C7 8.5e-6 relative to the output ([`C7_RELATIVE`], 5e-5), C8 2.4e-7 and C11 6.0e-7 in LogC4 (gate
//!   2e-4), the DNG stop 6.0e-6 relative (2e-4), C12 7.6e-6 relative to the largest channel (5e-4), C13
//!   8.7e-6 code values (one 10-bit code, 9.8e-4).

mod common;

use common::chains::{self, AUTO_CLIP_STOPS, camera_image, clip_luma, pixel};
use common::{Rng, TestGpu, f16_round, gpus};
use ym_color::consts::Vec3;
use ym_color::consts::linalg::IDENTITY;
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_color::tone::{HIGHLIGHT_CAP_STOPS, ToneParams, luma};
use ym_gpu::cpu::RgbImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

/// The gain's relative error, f32 `log2`/`exp2` against f64, amplified by the slope of the range (up to
/// 12 where it is steepest). Room for a `log2` a few ulp off, as Metal's fast-math one is.
const GAIN_TOLERANCE: f64 = 3e-5;
/// C7 against the CPU, relative to the output: the gain's error and the f32 C6 under it. 5e-5 of a
/// linear value is 7e-5 stops, 5e-6 in LogC4, far inside the 2e-4 gate of the stages after it.
const C7_RELATIVE: f64 = 5e-5;
/// C8 to C11, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;

fn render(
    dev: &TestGpu,
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

/// The bits of pixel `i` of an output, whatever its format.
fn pixel_bits(img: &ColorImage, i: usize) -> Vec<u32> {
    let n = img.width * img.height;
    match &img.data {
        PixelData::F32Planar(v) => (0..3).map(|c| v[c * n + i].to_bits()).collect(),
        PixelData::Rgba8(v) => v[4 * i..4 * i + 4].iter().map(|&b| u32::from(b)).collect(),
        PixelData::U16Planar(v) => (0..3).map(|c| u32::from(v[c * n + i])).collect(),
    }
}

/// Every output that could show a stray gain: the linear and log stages, the DNG stop, C13, and the
/// 8-bit (dithered) and 16-bit pictures.
const WITNESS: [ColorOutput; 9] = [
    ColorOutput::Stage(Stop::C6),
    ColorOutput::Stage(Stop::C7),
    ColorOutput::Stage(Stop::C8),
    ColorOutput::Stage(Stop::C9),
    ColorOutput::Stage(Stop::C11),
    ColorOutput::Stage(Stop::DngLinearAwg4),
    ColorOutput::Stage(Stop::C13),
    ColorOutput::Rgba8 { dither: true },
    ColorOutput::U16 { bits: 16 },
];

/// Stops over middle grey of the luma a chain's C6 gives a camera pixel (`-inf` for a luma at or below 0).
fn c6_stops(chain: &ColorChain, cam: Vec3) -> f64 {
    let y = luma(chain.trace(cam).c6_linear_awg4);
    if y > 0.0 {
        (y / 0.18).log2()
    } else {
        f64::NEG_INFINITY
    }
}

/// A neutral ramp over -4 to +12 stops of luma, with a colour cast on every other pixel, as an f16-exact
/// image the identity-matrix chain takes as AWG4.
fn ramp_image(w: usize, h: usize) -> RgbImage {
    let mut rgb = Vec::with_capacity(3 * w * h);
    for i in 0..w * h {
        let s = -4.0 + 16.0 * i as f64 / (w * h - 1) as f64;
        let y = 0.18 * s.exp2();
        let cast = if i % 2 == 0 {
            [1.0, 1.0, 1.0]
        } else {
            [1.3, 0.9, 0.6]
        };
        for c in cast {
            rgb.push(f16_round((y * c) as f32));
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

/// [`camera_image`] without its few very bright pixels: through [`chains::fused_matrix`] every pixel stays
/// under 6.6 stops over grey, a stop under the ALEXA cap.
fn below_cap_image(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut rgb = Vec::with_capacity(3 * w * h);
    for i in 0..w * h {
        let lum = 0.18 * rng.range(-12.0, 5.5).exp2();
        let mut px = [
            lum * rng.range(0.35, 1.65),
            lum * rng.range(0.35, 1.65),
            lum * rng.range(0.35, 1.65),
        ];
        if i % 23 == 5 {
            px[1] = -0.004 * rng.unit();
        }
        for c in px {
            rgb.push(f16_round(c as f32));
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

fn tone(highlights: f64, clip_stops: f64) -> ToneParams {
    ToneParams::new(highlights, 0.0, clip_luma(clip_stops), 0.0).expect("tone")
}

#[test]
fn the_gain_matches_the_cpu() {
    // The device's C7 over its own C6 (the identity matrix, so C6 is the camera value itself) is the
    // gain as the device computed it, against the CPU gain at the luma of that value.
    let (w, h) = (96usize, 64usize);
    let img = ramp_image(w, h);
    let n = w * h;
    let sets = [
        (0.5, AUTO_CLIP_STOPS),
        (1.0, AUTO_CLIP_STOPS),
        (1.0, 2.0),
        (0.7, 1.6),
        (0.3, 6.0),
        (1.0, 9.0),
        (0.0, AUTO_CLIP_STOPS),
    ];
    for dev in gpus() {
        let mut worst = 0.0f64;
        let mut acted = 0usize;
        for (r, clip_stops) in sets {
            let t = tone(r, clip_stops);
            let range = t.range();
            let chain = ColorChain::new(&ChainParams::new(IDENTITY, t)).expect("chain");
            let out = render(dev, &chain, &img, &[ColorOutput::Stage(Stop::C7)]);
            let got = planar(&out[0]);
            for i in 0..n {
                let input = pixel(&img, i);
                let want_gain = range.gain(luma(input));
                acted += usize::from(want_gain != 1.0);
                for c in 0..3 {
                    if input[c] < 1e-3 {
                        continue;
                    }
                    let ratio = f64::from(got[c * n + i]) / input[c];
                    let err = (ratio / want_gain - 1.0).abs();
                    worst = worst.max(err);
                    assert!(
                        err <= GAIN_TOLERANCE,
                        "{:?} r {r} clip {clip_stops}: pixel {i} ch {c}: gain {ratio} vs {want_gain} (rel {err:.2e})",
                        dev.mode
                    );
                }
            }
        }
        assert!(acted * 4 > sets.len() * n, "{acted} pixels with a gain");
        eprintln!(
            "{:?}: worst relative error of the highlight range's gain {worst:.2e} (tolerance {GAIN_TOLERANCE:.0e})",
            dev.mode
        );
    }
}

#[test]
fn every_stage_of_a_range_chain_matches_the_cpu_trace() {
    let (w, h) = (67usize, 41usize);
    let img = camera_image(w, h, 0x7A6E);
    let n = w * h;
    let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
    let cases = chains::range_cases();
    assert!(cases.len() >= 9, "{} range cases", cases.len());
    let outputs = [
        ColorOutput::Stage(Stop::C7),
        ColorOutput::Stage(Stop::C8),
        ColorOutput::Stage(Stop::C11),
        ColorOutput::Stage(Stop::DngLinearAwg4),
        ColorOutput::Stage(Stop::C12),
        ColorOutput::Stage(Stop::C13),
    ];
    for dev in gpus() {
        // C7 (relative to the output), C8, C11 (LogC4), DNG (relative), C12 (relative to the largest
        // channel), C13 (code values).
        let mut worst = [0.0f64; 6];
        for (name, params) in &cases {
            let chain = ColorChain::new(params).expect("chain");
            let range = params.tone.range();
            let traces: Vec<_> = cams.iter().map(|&c| chain.trace(c)).collect();
            // Not vacuous: a good part of the pixels must be moved by the range (or held at the cap),
            // except where r = 0 below the cap is the identity by design.
            let moved = traces
                .iter()
                .filter(|t| range.gain(luma(t.c6_linear_awg4)) != 1.0)
                .count();
            if range.expansion_stops > 0.0 || name.contains("over-the-cap") {
                assert!(
                    moved * 8 > n,
                    "{name}: only {moved} of {n} pixels in the range"
                );
            }
            let out = render(dev, &chain, &img, &outputs);
            let (c7, c8, c11, dng, c12, c13) = (
                planar(&out[0]),
                planar(&out[1]),
                planar(&out[2]),
                planar(&out[3]),
                planar(&out[4]),
                planar(&out[5]),
            );
            for i in 0..n {
                let t = &traces[i];
                let want_dng = chain.apply_dng_pixel(cams[i]);
                let mag: f64 = cams[i].iter().map(|v| v.abs()).sum::<f64>()
                    * params
                        .matrix
                        .iter()
                        .flatten()
                        .map(|v| v.abs())
                        .fold(0.0, f64::max);
                let out_mag: f64 = t.c7_linear_awg4.iter().map(|v| v.abs()).sum();
                for ch in 0..3 {
                    let e = (f64::from(c7[ch * n + i]) - t.c7_linear_awg4[ch]).abs();
                    worst[0] = worst[0].max(e / out_mag.max(1e-6));
                    assert!(
                        e <= 6e-6 * mag + C7_RELATIVE * out_mag + 1e-9,
                        "{name} {:?} C7 {i}/{ch}: {} vs {} (camera {:?})",
                        dev.mode,
                        c7[ch * n + i],
                        t.c7_linear_awg4[ch],
                        cams[i]
                    );
                    for (k, (g, want)) in [(c8, t.c8_logc4.0[ch]), (c11, t.c11_logc4.0[ch])]
                        .into_iter()
                        .enumerate()
                    {
                        let e = (f64::from(g[ch * n + i]) - want).abs();
                        worst[1 + k] = worst[1 + k].max(e);
                        assert!(
                            e <= LOGC4_GATE,
                            "{name} {:?} stage {k} {i}/{ch}: {e:.2e}",
                            dev.mode
                        );
                    }
                    // The device's DNG stop is C11 decoded; the inverse of a display rendering that is
                    // not ours (a display-baked look above 0 %, a display LUT) is the CPU's alone.
                    if !chain.dng_inverts_display() {
                        let scale = want_dng[ch].abs().max(0.01);
                        let e = (f64::from(dng[ch * n + i]) - want_dng[ch]).abs();
                        worst[3] = worst[3].max(e / scale);
                        assert!(
                            e <= 2e-4 * scale,
                            "{name} {:?} DNG {i}/{ch}: {} vs {}",
                            dev.mode,
                            dng[ch * n + i],
                            want_dng[ch]
                        );
                    }
                    let top = t
                        .c12_display_linear
                        .iter()
                        .copied()
                        .fold(0.0f64, f64::max)
                        .max(0.02);
                    let e = (f64::from(c12[ch * n + i]) - t.c12_display_linear[ch]).abs();
                    worst[4] = worst[4].max(e / top);
                    assert!(
                        e <= 5e-4 * top,
                        "{name} {:?} C12 {i}/{ch}: {} vs {} (camera {:?})",
                        dev.mode,
                        c12[ch * n + i],
                        t.c12_display_linear[ch],
                        cams[i]
                    );
                    let e = (f64::from(c13[ch * n + i]) - t.c13_encoded[ch]).abs();
                    worst[5] = worst[5].max(e);
                    assert!(e <= CODE10, "{name} {:?} C13 {i}/{ch}: {e:.2e}", dev.mode);
                }
            }
        }
        eprintln!(
            "{:?}: range chains ({}): worst C7 {:.1e} (relative to the output), C8 {:.1e}, C11 {:.1e} (LogC4), DNG {:.1e} (relative), C12 {:.1e} (relative to the largest channel), C13 {:.1e} (code values)",
            dev.mode,
            cases.len(),
            worst[0],
            worst[1],
            worst[2],
            worst[3],
            worst[4],
            worst[5]
        );
    }
}

/// r = 1 against r = 0 at the same clip and toe: every pixel under the knee comes out of the device with
/// the same bits in every output, every pixel well above it (and under the cap) differs at C7.
#[test]
fn below_the_knee_no_bit_changes_on_the_device() {
    let (w, h) = (61usize, 43usize);
    let img = camera_image(w, h, 0x4EE5);
    let n = w * h;
    for target in [OutputTarget::Viewport, OutputTarget::Srgb] {
        let mut low = chains::full(target);
        low.tone = chains::range_tone(0.0, AUTO_CLIP_STOPS);
        let mut high = low.clone();
        high.tone = chains::range_tone(1.0, AUTO_CLIP_STOPS);
        let knee = high.tone.range().knee_stops;
        let (a, b) = (
            ColorChain::new(&low).expect("chain"),
            ColorChain::new(&high).expect("chain"),
        );
        for dev in gpus() {
            let x = render(dev, &a, &img, &WITNESS);
            let y = render(dev, &b, &img, &WITNESS);
            let (mut below, mut above) = (0, 0);
            for i in 0..n {
                let s = c6_stops(&a, pixel(&img, i));
                if s < knee - 1e-3 {
                    below += 1;
                    for (k, (p, q)) in x.iter().zip(&y).enumerate() {
                        assert_eq!(
                            pixel_bits(p, i),
                            pixel_bits(q, i),
                            "{:?} {target:?}: pixel {i} at {s:.3} stops (knee {knee:.3}), output {k}",
                            dev.mode
                        );
                    }
                } else if s > knee + 0.1 && s < HIGHLIGHT_CAP_STOPS - 0.05 {
                    // Over the cap both hold the pixel at the ALEXA clip.
                    above += 1;
                    assert_ne!(
                        pixel_bits(&x[1], i),
                        pixel_bits(&y[1], i),
                        "{:?}: pixel {i} at {s:.2} stops is not moved",
                        dev.mode
                    );
                }
            }
            assert!(
                below > n / 2 && above > n / 10,
                "{below} below, {above} above"
            );
        }
    }
}

/// r = 0 against C7 off (no toe in either): every pixel under the ALEXA cap keeps its bits in every
/// output; the pixels over it (the fused matrix 32 times brighter) are held at the cap.
#[test]
fn r_zero_changes_no_bit_below_the_cap_on_the_device() {
    let (w, h) = (59usize, 37usize);
    let img = camera_image(w, h, 0x0C4B);
    let n = w * h;
    for gain in [1.0, 32.0] {
        let mut off = chains::full(OutputTarget::Srgb);
        off.matrix = chains::fused_matrix().map(|r| r.map(|v| v * gain));
        off.tone = ToneParams::IDENTITY;
        let mut zero = off.clone();
        zero.tone = tone(0.0, AUTO_CLIP_STOPS);
        let (a, b) = (
            ColorChain::new(&off).expect("chain"),
            ColorChain::new(&zero).expect("chain"),
        );
        for dev in gpus() {
            let x = render(dev, &a, &img, &WITNESS);
            let y = render(dev, &b, &img, &WITNESS);
            let (mut below, mut over) = (0, 0);
            for i in 0..n {
                let s = c6_stops(&a, pixel(&img, i));
                if s < HIGHLIGHT_CAP_STOPS - 1e-3 {
                    below += 1;
                    for (k, (p, q)) in x.iter().zip(&y).enumerate() {
                        assert_eq!(
                            pixel_bits(p, i),
                            pixel_bits(q, i),
                            "{:?} x{gain}: pixel {i} at {s:.3} stops, output {k}",
                            dev.mode
                        );
                    }
                } else if s > HIGHLIGHT_CAP_STOPS + 0.05 {
                    over += 1;
                    let c7 = planar(&y[1]);
                    let held = luma([0, 1, 2].map(|c| f64::from(c7[c * n + i])));
                    let cap = clip_luma(HIGHLIGHT_CAP_STOPS);
                    assert!(
                        (held / cap - 1.0).abs() < 1e-5,
                        "{:?}: pixel {i} at {s:.2} stops: luma {held} against the cap {cap}",
                        dev.mode
                    );
                }
            }
            assert!(below > n / 2, "{below} below the cap");
            if gain > 1.0 {
                assert!(over > n / 10, "{over} over the cap");
            }
        }
    }
}

/// C7, exactness (`docs/02`): a range that moves nothing is skipped pixel by pixel, so on an image under
/// the ALEXA cap every output is bit for bit the output of the same chain with C7 off: r = 0, a NaN amount
/// (counts as 0), a clip 1.2 stops over grey (the fade is 0), a clip at the ALEXA's and one far over it
/// (no expansion; the cap is out of reach). The control, r = 1 at the Auto clip, changes C7 and everything
/// after it, and not C6.
#[test]
fn a_range_that_moves_nothing_leaves_every_output_bit_identical() {
    let img = below_cap_image(45, 31, 0x0FF5);
    let identities = [
        ("r 0", tone(0.0, AUTO_CLIP_STOPS)),
        (
            "NaN amount",
            ToneParams::new(f64::NAN, 0.0, clip_luma(AUTO_CLIP_STOPS), 0.0).expect("tone"),
        ),
        ("clip inside the fade's floor", tone(1.0, 1.2)),
        ("clip at the ALEXA's", tone(1.0, HIGHLIGHT_CAP_STOPS)),
        ("clip far over the ALEXA's", tone(1.0, 20.0)),
    ];
    for dev in gpus() {
        for target in [OutputTarget::Viewport, OutputTarget::Srgb] {
            let mut params = chains::full(target);
            params.tone = ToneParams::IDENTITY;
            let plain = render(
                dev,
                &ColorChain::new(&params).expect("chain"),
                &img,
                &WITNESS,
            );
            for (name, t) in &identities {
                let range = t.range();
                assert_eq!(range.expansion_stops, 0.0, "{name}");
                let mut p = params.clone();
                p.tone = *t;
                let got = render(dev, &ColorChain::new(&p).expect("chain"), &img, &WITNESS);
                for (k, (a, b)) in got.iter().zip(&plain).enumerate() {
                    assert!(
                        same_f32_bits(a, b),
                        "{:?} {target:?} {name}: output {k} changed",
                        dev.mode
                    );
                }
            }
            let mut p = params.clone();
            p.tone = tone(1.0, AUTO_CLIP_STOPS);
            let got = render(dev, &ColorChain::new(&p).expect("chain"), &img, &WITNESS);
            assert!(same_f32_bits(&got[0], &plain[0]), "C6 is before the range");
            for k in 1..got.len() {
                assert!(!same_f32_bits(&got[k], &plain[k]), "{target:?}: output {k}");
            }
        }
    }
}

fn same_f32_bits(a: &ColorImage, b: &ColorImage) -> bool {
    match (&a.data, &b.data) {
        (PixelData::F32Planar(x), PixelData::F32Planar(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        _ => a == b,
    }
}

#[test]
fn tiles_of_a_range_render_equal_the_whole_image() {
    let (w, h) = (131usize, 77usize);
    let img = camera_image(w, h, 0x7A11);
    let mut params = chains::full(OutputTarget::DisplayP3);
    params.tone = chains::range_tone(1.0, AUTO_CLIP_STOPS);
    let chain = ColorChain::new(&params).expect("chain");
    let outputs = [
        ColorOutput::Rgba8 { dither: true },
        ColorOutput::Stage(Stop::C7),
    ];
    for dev in gpus() {
        let whole = render(dev, &chain, &img, &outputs);
        for tile in [32usize, 50, 64] {
            let plan = TilePlan::with_halo(w, h, tile, 0);
            let chain2 = chain.clone();
            let img2 = img.clone();
            let results = dev
                .gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let prepared = PreparedChain::new(ctx, &chain2)?;
                    let mut collected = Vec::new();
                    for out in outputs {
                        let mut tiles = Vec::new();
                        prepared.render_tiled(ctx, &img2, &plan, out, 2, token, &mut |t, im| {
                            tiles.push((t, im));
                            Ok(())
                        })?;
                        collected.push(tiles);
                    }
                    Ok(collected)
                })
                .expect("tiled render");
            for (k, tiles) in results.iter().enumerate() {
                for (t, im) in tiles {
                    let c = t.core;
                    for y in 0..c.h {
                        for x in 0..c.w {
                            let li = y * c.w + x;
                            let gi = (c.y + y) * w + c.x + x;
                            assert_eq!(
                                pixel_bits(im, li),
                                pixel_bits(&whole[k], gi),
                                "{:?} tile {tile}: output {k} at ({}, {})",
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
