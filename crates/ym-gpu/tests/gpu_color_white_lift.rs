//! The white lift (`docs/02` C12 step 1b, D-02) on the GPU: the colour kernel against
//! `ym_color::display::ColorChain`.
//!
//! * the gain stage alone (the DNG stop of a chain with an identity matrix and tone): the ratio of
//!   output to input against `WhiteLift::gain`, max abs error stated below;
//! * every stage of chains with a lift (C11, the DNG stop, C12, C13) against the CPU trace, on the
//!   random camera image, in every storage mode;
//! * a lift of 0 is the exact identity on the device too;
//! * tiles of a lifted render equal the whole image (the lift is per pixel, nothing crosses a tile).
//!
//! **Stated errors** (lavapipe, f32 maths against the f64 reference; Metal gets its own column in the
//! CI log): the gain, relative, `GAIN_TOLERANCE` below; C12 display-linear 5e-4 relative of the largest
//! channel (the existing gate of `gpu_color.rs`); the DNG stop 2e-4 relative; C13 one 10-bit code.

mod common;

use common::chains::{self, camera_image, pixel};
use common::gpus;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_color::exposure::white::WhiteLift;
use ym_gpu::cpu::RgbImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

/// The gain's relative error, f32 `log2`/`exp2` against f64.
const GAIN_TOLERANCE: f64 = 5e-6;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;

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

/// A neutral ramp over -4 to +6 stops of luma, with a colour cast on every other pixel, as an f32
/// image the identity-matrix chain takes as AWG4.
fn ramp_image(w: usize, h: usize) -> RgbImage {
    let mut rgb = Vec::with_capacity(3 * w * h);
    for i in 0..w * h {
        let s = -4.0 + 10.0 * i as f64 / (w * h - 1) as f64;
        let y = 0.18 * s.exp2();
        let cast = if i % 2 == 0 {
            [1.0, 1.0, 1.0]
        } else {
            [1.3, 0.9, 0.6]
        };
        for c in cast {
            rgb.push(common::f16_round((y * c) as f32));
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

#[test]
fn the_gain_matches_the_cpu() {
    // The device's DNG stop of a lifted chain over that of the same chain without the lift is the gain
    // as the device computed it. It is compared with the CPU gain at the luma of the device's own
    // unlifted pixel, so what is measured is the f32 `log2`, `exp2` and smoothstep, not the f32 round
    // trip through LogC4 that both renders share (the gain of a narrow strong ramp is steep: a relative
    // 1e-6 in the luma is 1.5e-5 in the gain at amount 3 and span 0.3).
    let (w, h) = (64usize, 48usize);
    let img = ramp_image(w, h);
    let n = w * h;
    for dev in gpus() {
        let mut base = ChainParams::new(identity_matrix(), chains::identity_tone());
        base.output = OutputTarget::Srgb;
        let plain_chain = ColorChain::new(&base).expect("chain");
        let plain = render(
            dev,
            &plain_chain,
            &img,
            &[ColorOutput::Stage(Stop::DngLinearAwg4)],
        );
        let plain = planar(&plain[0]);
        let mut worst = 0.0f64;
        for (amount, anchor) in [(0.4, 1.8), (1.1, 2.8), (2.0, 3.6), (3.0, 1.3), (0.9, 6.0)] {
            let lift = WhiteLift::new(amount, anchor).expect("lift");
            let mut p = base.clone();
            p.white_lift = lift;
            let chain = ColorChain::new(&p).expect("chain");
            let out = render(
                dev,
                &chain,
                &img,
                &[ColorOutput::Stage(Stop::DngLinearAwg4)],
            );
            let got = planar(&out[0]);
            for i in 0..n {
                let lin = [
                    f64::from(plain[i]),
                    f64::from(plain[n + i]),
                    f64::from(plain[2 * n + i]),
                ];
                let want_gain = lift.gain(lin);
                for c in 0..3 {
                    if lin[c] < 1e-3 {
                        continue;
                    }
                    let ratio = f64::from(got[c * n + i]) / lin[c];
                    let err = (ratio / want_gain - 1.0).abs();
                    worst = worst.max(err);
                    assert!(
                        err <= GAIN_TOLERANCE,
                        "{:?} A {amount} anchor {anchor}: pixel {i} ch {c}: gain {ratio} vs {want_gain} (rel {err:.2e})",
                        dev.mode
                    );
                }
            }
        }
        eprintln!(
            "{:?}: worst relative error of the white lift gain {worst:.2e} (tolerance {GAIN_TOLERANCE:.0e})",
            dev.mode
        );
    }
}

#[test]
fn every_stage_of_a_lifted_chain_matches_the_cpu_trace() {
    let (w, h) = (67usize, 41usize);
    let img = camera_image(w, h, 0x11F7);
    let n = w * h;
    let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
    let cases: Vec<_> = chains::cases()
        .into_iter()
        .filter(|(name, _)| name.starts_with("lift/"))
        .collect();
    assert!(cases.len() >= 6, "{} lift cases", cases.len());
    for dev in gpus() {
        let mut worst = [0.0f64; 4];
        for (name, params) in &cases {
            let chain = ColorChain::new(params).expect("chain");
            let traces: Vec<_> = cams.iter().map(|&c| chain.trace(c)).collect();
            // Guard against a vacuous comparison: a good part of the pixels must be in the lift's range.
            let lifted = cams
                .iter()
                .filter(|c| {
                    chain
                        .white_lift()
                        .gain(chain.trace(**c).c11_logc4.to_linear().0)
                        > 1.0 + 1e-9
                })
                .count();
            if !chain.white_lift().is_identity() {
                assert!(
                    lifted * 10 > n,
                    "{name}: only {lifted} of {n} pixels are lifted"
                );
            }
            let out = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Stage(Stop::C11),
                    ColorOutput::Stage(Stop::DngLinearAwg4),
                    ColorOutput::Stage(Stop::C12),
                    ColorOutput::Stage(Stop::C13),
                ],
            );
            let (c11, dng, c12, c13) = (
                planar(&out[0]),
                planar(&out[1]),
                planar(&out[2]),
                planar(&out[3]),
            );
            for i in 0..n {
                let t = &traces[i];
                let want_dng = chain.apply_dng_pixel(cams[i]);
                for ch in 0..3 {
                    let e = (f64::from(c11[ch * n + i]) - t.c11_logc4.0[ch]).abs();
                    worst[0] = worst[0].max(e);
                    assert!(e <= 2e-4, "{name} {:?} C11 {i}/{ch}: {e:.2e}", dev.mode);
                    // The device's DNG stop is C11 decoded and lifted; the inverse of a display
                    // rendering that is not ours (a display-baked look above 0 %) is the CPU's alone.
                    if !chain.dng_inverts_display() {
                        let tol = 2e-4 * want_dng[ch].abs().max(0.01);
                        let e = (f64::from(dng[ch * n + i]) - want_dng[ch]).abs();
                        worst[1] = worst[1].max(e / want_dng[ch].abs().max(0.01));
                        assert!(
                            e <= tol,
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
                    worst[2] = worst[2].max(e / top);
                    assert!(
                        e <= 5e-4 * top,
                        "{name} {:?} C12 {i}/{ch}: {} vs {} (camera {:?})",
                        dev.mode,
                        c12[ch * n + i],
                        t.c12_display_linear[ch],
                        cams[i]
                    );
                    let e = (f64::from(c13[ch * n + i]) - t.c13_encoded[ch]).abs();
                    worst[3] = worst[3].max(e);
                    assert!(e <= CODE10, "{name} {:?} C13 {i}/{ch}: {e:.2e}", dev.mode);
                }
            }
        }
        eprintln!(
            "{:?}: lifted chains ({}): worst C11 {:.1e} (LogC4), DNG {:.1e} (relative), C12 {:.1e} (relative to the largest channel), C13 {:.1e} (code values)",
            dev.mode,
            cases.len(),
            worst[0],
            worst[1],
            worst[2],
            worst[3]
        );
    }
}

#[test]
fn a_lift_of_zero_is_the_identity_on_the_device() {
    let (w, h) = (53usize, 31usize);
    let img = camera_image(w, h, 0x2E20);
    let plain = ColorChain::new(&chains::full(OutputTarget::Viewport)).expect("chain");
    let mut params = chains::full(OutputTarget::Viewport);
    params.white_lift = WhiteLift::new(0.0, 3.0).expect("lift");
    let zero = ColorChain::new(&params).expect("chain");
    let stages = [
        ColorOutput::Stage(Stop::DngLinearAwg4),
        ColorOutput::Stage(Stop::C12),
        ColorOutput::Stage(Stop::C13),
    ];
    for dev in gpus() {
        let a = render(dev, &plain, &img, &stages);
        let b = render(dev, &zero, &img, &stages);
        for (x, y) in a.iter().zip(&b) {
            let (x, y) = (planar(x), planar(y));
            assert!(
                x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
                "{:?}: a lift of 0 changed a bit",
                dev.mode
            );
        }
    }
}

#[test]
fn a_lift_changes_only_what_is_above_one_stop_on_the_device() {
    // Mid grey and the tones below +1 stop come out of the device exactly as without the lift.
    let (w, h) = (64usize, 48usize);
    let img = ramp_image(w, h);
    let n = w * h;
    let mut base = ChainParams::new(identity_matrix(), chains::identity_tone());
    base.output = OutputTarget::Srgb;
    let plain = ColorChain::new(&base).expect("chain");
    base.white_lift = WhiteLift::new(1.4, 2.6).expect("lift");
    let lifted = ColorChain::new(&base).expect("chain");
    for dev in gpus() {
        let a = render(dev, &plain, &img, &[ColorOutput::Stage(Stop::C12)]);
        let b = render(dev, &lifted, &img, &[ColorOutput::Stage(Stop::C12)]);
        let (a, b) = (planar(&a[0]), planar(&b[0]));
        let mut below = 0;
        let mut above = 0;
        for i in 0..n {
            let luma = ym_color::exposure::white::drive_luma(pixel(&img, i));
            let stops = (luma / 0.18).log2();
            for c in 0..3 {
                if stops < 0.999 {
                    below += 1;
                    assert_eq!(
                        a[c * n + i].to_bits(),
                        b[c * n + i].to_bits(),
                        "{:?}: pixel {i} at {stops:.2} stops moved",
                        dev.mode
                    );
                } else if stops > 1.5 {
                    above += 1;
                    assert!(
                        b[c * n + i] > a[c * n + i],
                        "{:?}: pixel {i} at {stops:.2} stops not lifted",
                        dev.mode
                    );
                }
            }
        }
        assert!(below > 100 && above > 100, "{below} below, {above} above");
    }
}
