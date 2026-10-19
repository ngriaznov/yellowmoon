//! The built-in Signature look (`data/looks/builtin/signature.json`: a LogC4 luma curve and CDL
//! saturation 1.45, baked into a 65-point cube) through the existing GPU look path, against the CPU
//! reference `ym_color::display::ColorChain`.
//!
//! A parametric look is only a cube once loaded, so the colour kernel needs nothing new for it: this
//! test is the proof. It runs the real baked cube (outputs leave `[0, 1]` for saturated colours,
//! inputs outside `[0, 1]` clamp at the cube's edge) over a 17-stop random camera image at three
//! intensities (0 %, 60 %, 100 %) and three outputs, in every storage mode the device supports.
//!
//! Gates (the same as `gpu_color.rs`, `docs/02` section Tests): the look's LogC4 output (C10) within
//! 2e-4 of the CPU's, the displayed result (C13) within one 10-bit code value. Intensity 0 is the
//! chain without a look, bit for bit on both devices.

mod common;

use std::sync::Arc;

use common::chains::{self, camera_image, pixel};
use common::{TestGpu, gpus};
use ym_color::consts::Vec3;
use ym_color::display::{ChainParams, ColorChain, LookSetting, OutputTarget};
use ym_color::look::{LookTransform, ParametricLook};
use ym_gpu::cpu::RgbImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

const SIGNATURE_JSON: &str = include_str!("../../../data/looks/builtin/signature.json");

/// C10 gate, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;

fn signature() -> Arc<LookTransform> {
    Arc::new(
        ParametricLook::from_json(SIGNATURE_JSON)
            .expect("signature.json parses")
            .to_transform()
            .expect("bakes"),
    )
}

fn chain_with(target: OutputTarget, look: Option<LookSetting>) -> ChainParams {
    let mut p = chains::base(target);
    p.look = look;
    p
}

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

/// The largest `|gpu - cpu|` over all pixels and channels.
fn worst(gpu: &[f32], cpu: &[Vec3]) -> f64 {
    let n = cpu.len();
    let mut w = 0.0_f64;
    for (i, c) in cpu.iter().enumerate() {
        for ch in 0..3 {
            w = w.max((f64::from(gpu[ch * n + i]) - c[ch]).abs());
        }
    }
    w
}

#[test]
fn signature_through_the_gpu_matches_the_cpu() {
    let (w, h) = (73usize, 47usize);
    let img = camera_image(w, h, 0x51_6E);
    let n = w * h;
    let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
    let look = signature();
    for dev in gpus() {
        let mut worst_c10 = 0.0_f64;
        let mut worst_c13 = 0.0_f64;
        for target in [
            OutputTarget::Viewport,
            OutputTarget::Srgb,
            OutputTarget::Rec2020,
        ] {
            for pct in [0.0, 60.0, 100.0] {
                let params = chain_with(target, Some(LookSetting::new(Arc::clone(&look), pct)));
                let chain = ColorChain::new(&params).expect("chain");
                let traces: Vec<_> = cams.iter().map(|&c| chain.trace(c)).collect();
                // The look must actually move pixels at 60 and 100 %, or this proves nothing.
                if pct > 0.0 {
                    let moved = traces
                        .iter()
                        .filter(|t| {
                            (0..3).any(|c| (t.c10_logc4.0[c] - t.c9_logc4.0[c]).abs() > 1e-3)
                        })
                        .count();
                    assert!(
                        moved * 2 > n,
                        "{target:?} {pct} %: only {moved} of {n} pixels move"
                    );
                }
                let gpu = render(
                    dev,
                    &chain,
                    &img,
                    &[ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C13)],
                );
                let cpu_c10: Vec<Vec3> = traces.iter().map(|t| t.c10_logc4.0).collect();
                let cpu_c13: Vec<Vec3> = traces.iter().map(|t| t.c13_encoded).collect();
                let d10 = worst(planar(&gpu[0]), &cpu_c10);
                let d13 = worst(planar(&gpu[1]), &cpu_c13);
                assert!(
                    d10 <= LOGC4_GATE,
                    "{:?} {target:?} {pct} %: C10 |GPU - CPU| = {d10:.3e}",
                    dev.mode
                );
                assert!(
                    d13 <= CODE10,
                    "{:?} {target:?} {pct} %: C13 |GPU - CPU| = {d13:.3e}",
                    dev.mode
                );
                worst_c10 = worst_c10.max(d10);
                worst_c13 = worst_c13.max(d13);
            }
        }
        eprintln!(
            "{:?}: Signature, 9 chains, worst |GPU - CPU|: C10 {worst_c10:.2e} LogC4 (gate {LOGC4_GATE:.0e}), C13 {worst_c13:.2e} (gate {CODE10:.1e})",
            dev.mode
        );
    }
}

/// Intensity 0 is Neutral: the same bits as a chain with no look, on the GPU as on the CPU.
#[test]
fn signature_at_zero_percent_is_the_chain_without_a_look_bit_for_bit() {
    let (w, h) = (41usize, 29usize);
    let img = camera_image(w, h, 0x7E5A);
    let look = signature();
    let none = ColorChain::new(&chain_with(OutputTarget::Srgb, None)).expect("chain");
    let zero = ColorChain::new(&chain_with(
        OutputTarget::Srgb,
        Some(LookSetting::new(look, 0.0)),
    ))
    .expect("chain");
    for dev in gpus() {
        let outs = [ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C13)];
        let a = render(dev, &none, &img, &outs);
        let b = render(dev, &zero, &img, &outs);
        for k in 0..2 {
            assert_eq!(
                planar(&a[k]),
                planar(&b[k]),
                "{:?}: output {k} differs between no look and 0 %",
                dev.mode
            );
        }
    }
}
