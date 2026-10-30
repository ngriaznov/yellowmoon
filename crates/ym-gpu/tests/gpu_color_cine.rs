//! C12, the cine rendering (`docs/02-colour-pipeline.md`, *Cine rendering (C12, default)*): the GPU kernel
//! `cine.wgsl` against the CPU reference `ym_color::display::cine::CineRendering::forward`, on the inputs of
//! the fixture `crates/ym-color/tests/fixtures/cine_vectors.json` (neutral ramps, skin, ColorChecker, the AWG4
//! primaries, a hue wheel beyond the spectral locus, exposure ladders, negative channels, zeros, tiny and huge
//! values, seeded random colours), for the sRGB, viewport (Display P3) and Rec.2020 outputs in every storage
//! mode the device supports.
//!
//! The inputs of the fixture are joined by large neutrals and near-neutrals of both signs (1e3 to 1e6 in every
//! channel, `large_inputs`): the luminance floor of step 2 must not turn the rounding noise of a large triple into a
//! luminance, whichever way the device fuses and orders its sums (`docs/02` C12, GPU notes).
//!
//! The chain is an identity matrix with C7 off, so a camera triple goes through the LogC4 encode and decode in
//! f32 before it reaches C12. To measure the kernel and nothing else, the CPU reference is fed the linear AWG4
//! triple the device itself decoded (the DNG stop, which is C11 decoded: exactly the input of C12). The
//! chain-level tests of `gpu_color.rs` cover the LogC4 round trip.
//!
//! **Gate:** the error of a channel is `|gpu - cpu|` in display-linear light, judged relative to the largest
//! channel of its pixel (`max(cpu_r, cpu_g, cpu_b, 1e-3)`): 1e-5 of the pixel's brightest channel, which is
//! 1e-8 absolute in the darkest pixels (under sRGB code 12.9) where a code value does not see it. Judged against
//! the channel itself, a dark channel of a bright saturated colour (a blue of 0.08 next to a green of 0.9) takes
//! the same absolute error as its neighbours and shows it 10 times larger. The printed numbers are the measured
//! worst cases of each output, against the channel and against the pixel.

mod common;

use std::sync::LazyLock;

use common::{TestGpu, f16_round, gpus};
use serde_json::Value;
use ym_color::consts::linalg::IDENTITY;
use ym_color::display::cine::CineRendering;
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_color::gamut::LinearAwg4;
use ym_color::tone::ToneParams;
use ym_gpu::cpu::RgbImage;
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

/// The relative gate of the kernel (module notes).
const GATE: f64 = 1e-5;
/// The floor of the denominator: under it the gate is absolute (`GATE * FLOOR`).
const FLOOR: f64 = 1e-3;

static FIXTURE: LazyLock<Value> = LazyLock::new(|| {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ym-color/tests/fixtures/cine_vectors.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("the fixture")).expect("json")
});

/// The scene triples of the fixture with their names.
fn inputs() -> Vec<(String, [f64; 3])> {
    FIXTURE["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| {
            let x = c["x"].as_array().expect("x");
            (
                c["name"].as_str().expect("name").to_string(),
                [0, 1, 2].map(|i| x[i].as_f64().expect("number")),
            )
        })
        .collect()
}

/// Neutrals and near-neutrals of both signs at 1e3, 1e4, 1e5 and 1e6 in every channel, with their names. A neutral
/// has `dX = dZ = 0` in exact arithmetic, and a large triple's XYZ carries rounding noise of about 1e-7 of its size
/// (1e-2 at 1e5) that differs between a strict and a fusing compiler; a near-neutral differs from the neutral
/// by a relative step of 1e-2 down to the last bit of the f32.
fn large_inputs() -> Vec<(String, [f64; 3])> {
    let mut out = Vec::new();
    for (sign, s) in [(-1.0f64, "-"), (1.0, "+")] {
        for (mag, m) in [(1e3, "1e3"), (1e4, "1e4"), (1e5, "1e5"), (1e6, "1e6")] {
            let v = sign * mag;
            out.push((format!("large/neutral{s}{m}"), [v, v, v]));
            for (e, es) in [
                (1e-2, "1e-2"),
                (1e-3, "1e-3"),
                (1e-5, "1e-5"),
                (1e-7, "1e-7"),
            ] {
                let name = |shape: &str| format!("large/near-neutral{s}{m}/{es}/{shape}");
                out.push((name("green"), [v, v * (1.0 + e), v]));
                out.push((name("blue"), [v, v, v * (1.0 - e)]));
                out.push((name("mixed"), [v * (1.0 + e), v, v * (1.0 - 2.0 * e)]));
            }
        }
    }
    out
}

/// The inputs of the parity test: the fixture and the large neutrals.
fn parity_inputs() -> Vec<(String, [f64; 3])> {
    let mut all = inputs();
    all.extend(large_inputs());
    all
}

fn chain_for(output: OutputTarget) -> ColorChain {
    let mut p = ChainParams::new(IDENTITY, ToneParams::IDENTITY);
    p.output = output;
    p.dither = false;
    ColorChain::new(&p).expect("chain")
}

fn planar(out: &PixelData) -> &[f32] {
    match out {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

/// Renders the stages `outputs` of `chain` over `px` laid out as one row.
fn render(
    dev: &TestGpu,
    chain: &ColorChain,
    px: &[[f32; 3]],
    outputs: &[ColorOutput],
) -> Vec<PixelData> {
    let inter: Vec<f32> = px.iter().flatten().copied().collect();
    let img = RgbImage::from_interleaved(px.len(), 1, &inter).expect("image");
    let chain = chain.clone();
    let outputs = outputs.to_vec();
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let prepared = PreparedChain::new(ctx, &chain)?;
            outputs
                .iter()
                .map(|o| prepared.render(ctx, &img, (0, 0), *o).map(|i| i.data))
                .collect()
        })
        .expect("GPU render")
}

/// The inputs the device can hold in `mode`: f16 storage rounds every sample and cannot hold more than
/// 65504, so the f16 modes take the inputs that fit, rounded.
fn inputs_for(mode: StorageMode, all: Vec<(String, [f64; 3])>) -> Vec<(String, [f32; 3])> {
    all.into_iter()
        .filter_map(|(name, x)| {
            let f = x.map(|v| v as f32);
            if mode == StorageMode::F32 {
                Some((name, f))
            } else if f.iter().all(|v| v.abs() <= 60000.0) {
                Some((name, f.map(f16_round)))
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn the_kernel_matches_the_cpu_reference_on_the_fixture_and_the_large_neutrals() {
    let mut reported = 0;
    for dev in gpus() {
        let named = inputs_for(dev.mode, parity_inputs());
        let px: Vec<[f32; 3]> = named.iter().map(|(_, x)| *x).collect();
        let n = px.len();
        for output in [
            OutputTarget::Srgb,
            OutputTarget::Viewport,
            OutputTarget::Rec2020,
        ] {
            let chain = chain_for(output);
            let out = render(
                dev,
                &chain,
                &px,
                &[
                    ColorOutput::Stage(Stop::DngLinearAwg4),
                    ColorOutput::Stage(Stop::C12),
                ],
            );
            let (lin, disp) = (planar(&out[0]), planar(&out[1]));
            let cine = CineRendering::new(output);
            let (mut worst_rel, mut worst_abs, mut worst_at) = (0.0f64, 0.0f64, 0usize);
            // The same measure over the large neutrals alone: (error, input).
            let mut worst_large = (0.0f64, 0usize);
            let mut worst_channel_rel = 0.0f64;
            for i in 0..n {
                let input = [0, 1, 2].map(|c| f64::from(lin[c * n + i]));
                let want = cine.forward(LinearAwg4(input));
                for c in 0..3 {
                    let got = f64::from(disp[c * n + i]);
                    assert!(
                        got.is_finite() && (0.0..=1.0).contains(&got),
                        "{:?} {output:?} {}: channel {c} is {got}",
                        dev.mode,
                        named[i].0
                    );
                    let d = (got - want[c]).abs();
                    let scale = want[0].max(want[1]).max(want[2]).max(FLOOR);
                    let rel = d / scale;
                    worst_channel_rel = worst_channel_rel.max(d / want[c].abs().max(FLOOR));
                    worst_abs = worst_abs.max(d);
                    if rel > worst_rel {
                        worst_rel = rel;
                        worst_at = i;
                    }
                    if named[i].0.starts_with("large/") && rel > worst_large.0 {
                        worst_large = (rel, i);
                    }
                }
            }
            eprintln!(
                "{:?} {output:?}: {n} inputs, worst error {worst_rel:.2e} of the pixel's brightest channel (at {}), {worst_channel_rel:.2e} of the channel itself, worst absolute {worst_abs:.2e}",
                dev.mode, named[worst_at].0
            );
            eprintln!(
                "{:?} {output:?}: large neutrals alone, worst error {:.2e} of the pixel's brightest channel (at {})",
                dev.mode, worst_large.0, named[worst_large.1].0
            );
            assert!(
                worst_rel <= GATE,
                "{:?} {output:?}: relative error {worst_rel:.3e} at {} (gate {GATE:e})",
                dev.mode,
                named[worst_at].0
            );
            reported += 1;
        }
    }
    if !gpus().is_empty() {
        assert!(reported >= 3);
    }
}

#[test]
fn a_large_negative_neutral_renders_at_the_bottom_of_the_tone_curve_on_the_device() {
    // The parity gate is absolute in the darkest pixels (1e-8), and the tone curve falls as Ys^4.3: a luminance
    // floor decided by rounding noise (Ys of 5e-4 instead of Y_MIN = 1e-9) shows in the gate only from about -1e5
    // (1.6e-8 on the Mac). This test sees it from the first bit: a negative neutral has no luminance, its Ys is
    // Y_MIN exactly, and the gamut stage's floor at Y2_MIN puts every channel at the same 8.1e-14, judged
    // against the value itself (1e-4) and not against the gate. The CPU is fed the linear triple the device decoded,
    // and a triple whose channels the decode left unequal is judged as it is, against the CPU.
    let mut checked = 0;
    for dev in gpus() {
        let named: Vec<(String, [f32; 3])> = inputs_for(dev.mode, large_inputs())
            .into_iter()
            .filter(|(name, _)| name.starts_with("large/neutral-"))
            .collect();
        let px: Vec<[f32; 3]> = named.iter().map(|(_, x)| *x).collect();
        let n = px.len();
        for output in [
            OutputTarget::Srgb,
            OutputTarget::Viewport,
            OutputTarget::Rec2020,
        ] {
            let out = render(
                dev,
                &chain_for(output),
                &px,
                &[
                    ColorOutput::Stage(Stop::DngLinearAwg4),
                    ColorOutput::Stage(Stop::C12),
                ],
            );
            let (lin, disp) = (planar(&out[0]), planar(&out[1]));
            let cine = CineRendering::new(output);
            for (i, (name, _)) in named.iter().enumerate() {
                let input = [0, 1, 2].map(|c| f64::from(lin[c * n + i]));
                let want = cine.forward(LinearAwg4(input));
                let neutral = input[0] == input[1] && input[1] == input[2];
                let got = [0, 1, 2].map(|c| f64::from(disp[c * n + i]));
                let rel = [0, 1, 2].map(|c| (got[c] - want[c]).abs() / want[c]);
                eprintln!(
                    "{:?} {output:?} {name}: {:.4e} against {:.4e} (worst channel {:.1e}), decoded {:?}",
                    dev.mode,
                    got[1],
                    want[1],
                    rel[0].max(rel[1]).max(rel[2]),
                    input[1]
                );
                for c in 0..3 {
                    assert!(
                        rel[c] <= 1e-4,
                        "{:?} {output:?} {name}: channel {c} is {:e}, the CPU has {:e}",
                        dev.mode,
                        got[c],
                        want[c]
                    );
                    // A neutral decoded to equal channels is at the bottom of the tone curve for the CPU too.
                    assert!(
                        !neutral || want[c] < 1e-13,
                        "{name}: the CPU's neutral is {:e}",
                        want[c]
                    );
                }
                checked += 1;
            }
        }
    }
    if !gpus().is_empty() {
        assert!(checked >= 12);
    }
}

#[test]
fn the_neutral_axis_is_grey_and_reaches_one_on_the_device() {
    for dev in gpus() {
        let stops: Vec<f64> = (-80..=130).map(|i| f64::from(i) * 0.1).collect();
        let px: Vec<[f32; 3]> = stops
            .iter()
            .map(|s| {
                let g = (0.18 * s.exp2()) as f32;
                [g; 3]
            })
            .filter(|p| dev.mode == StorageMode::F32 || p[0] <= 60000.0)
            .collect();
        let n = px.len();
        for output in [
            OutputTarget::Srgb,
            OutputTarget::Viewport,
            OutputTarget::Rec2020,
        ] {
            let out = render(
                dev,
                &chain_for(output),
                &px,
                &[ColorOutput::Stage(Stop::C12)],
            );
            let d = planar(&out[0]);
            let mut last = [0.0f64; 3];
            for i in 0..n {
                let v = [0, 1, 2].map(|c| f64::from(d[c * n + i]));
                // Grey to grey; the f32 rounding of the colour terms leaves a few 1e-7 of the value.
                assert!(
                    (v[0] - v[1]).abs() <= 2e-6 * v[1].max(1e-3)
                        && (v[2] - v[1]).abs() <= 2e-6 * v[1].max(1e-3),
                    "{:?} {output:?} px {i}: {v:?}",
                    dev.mode
                );
                // Non-decreasing to within the f32 noise of the curve.
                for c in 0..3 {
                    assert!(
                        v[c] >= last[c] - 2e-6 * last[c].max(1e-3),
                        "{:?} {output:?} px {i}",
                        dev.mode
                    );
                }
                last = v;
            }
            // From the white level (scene 243) up the output is 1 to f32 rounding, from +10.5 stops exactly.
            let top = [0, 1, 2].map(|c| f64::from(d[c * n + n - 1]));
            for v in top {
                assert!(
                    (v - 1.0).abs() <= 1e-6,
                    "{:?} {output:?}: white {v}",
                    dev.mode
                );
            }
        }
    }
}

#[test]
fn the_encoded_picture_follows_the_cpu_chain_within_a_code_value() {
    // C13 after C12: 8-bit codes (no dither) against the CPU chain's `encode_u8` of the same f32 input, for
    // the three outputs. The LogC4 round trip in f32 moves a dark code by at most one value.
    for dev in gpus() {
        let named = inputs_for(dev.mode, inputs());
        let px: Vec<[f32; 3]> = named
            .iter()
            .filter(|(_, x)| x.iter().all(|v| v.abs() <= 1000.0))
            .map(|(_, x)| *x)
            .collect();
        let n = px.len();
        for output in [
            OutputTarget::Srgb,
            OutputTarget::Viewport,
            OutputTarget::Rec2020,
        ] {
            let chain = chain_for(output);
            let out = render(dev, &chain, &px, &[ColorOutput::Rgba8 { dither: false }]);
            let PixelData::Rgba8(bytes) = &out[0] else {
                panic!("not RGBA8");
            };
            let mut worst = 0i32;
            for (i, p) in px.iter().enumerate() {
                let want = chain.encode_u8(chain.apply_to_display_linear(p.map(f64::from)), i, 0);
                for c in 0..3 {
                    worst = worst.max((i32::from(bytes[4 * i + c]) - i32::from(want[c])).abs());
                }
            }
            eprintln!(
                "{:?} {output:?}: {n} colours, worst 8-bit code difference {worst}",
                dev.mode
            );
            assert!(worst <= 1, "{:?} {output:?}: {worst} codes", dev.mode);
        }
    }
}
