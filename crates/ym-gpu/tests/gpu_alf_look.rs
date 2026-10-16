//! Looks read from ARRI Look Files (`ym_color::alf`) through the existing GPU look path, against the
//! CPU reference `ym_color::display::ColorChain`.
//!
//! An imported look file is a `LookTransform` like any other (an input encoding, a CDL and a LUT),
//! so the colour kernel needs nothing new for it; this test is the proof. The files are written by
//! our own writer from synthetic LUTs and CDLs (no ARRI data): an `.alf4` (LogC4 look with a CDL and a
//! 33-point cube, the size of ARRI's library), an `.alf4c` (its CMT) and an `.aml` (a LogC3-to-Rec.709
//! look, which replaces the display transform). Each is read back by the parser, so the cube the
//! kernel gets is the one decoded from 16-bit codes.
//!
//! Gates (the same as `gpu_color.rs` and `gpu_signature_look.rs`, `docs/02` section Tests): the look's
//! LogC4 output (C10) within 2e-4 of the CPU's for the log-to-log looks, the displayed result (C13)
//! within one 10-bit code value, at 25, 60 and 100 % intensity and for three outputs, in every storage
//! mode the device supports.

mod common;

use std::sync::Arc;

use common::chains::{self, camera_image, pixel};
use common::{TestGpu, gpus};
use ym_color::alf::{self, AlfFile};
use ym_color::cdl::Cdl;
use ym_color::consts::Vec3;
use ym_color::display::{ChainParams, ColorChain, LookSetting, OutputTarget};
use ym_color::look::LookTransform;
use ym_color::lut::{Lut, Lut3d};
use ym_gpu::cpu::RgbImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

/// C10 gate, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;

/// A 33-point cube of our own, the size ARRI's look files have.
fn cube() -> Lut3d {
    Lut3d::bake(33, [0.0; 3], [1.0; 3], |c| {
        let t = chains::tweak(c);
        [t[0].powf(1.1), t[1].powf(1.05), t[2].powf(1.1)]
    })
    .expect("bake")
}

fn cdl() -> Cdl {
    Cdl::new([1.1, 1.0, 0.9], [0.02, 0.0, -0.02], [1.0, 1.1, 1.0], 1.1).expect("cdl")
}

/// A look as the app builds it from a look file: written, read back, and turned into a transform.
fn from_file(file: &AlfFile) -> Arc<LookTransform> {
    let bytes = file.to_bytes().expect("write");
    let look = alf::parse(&bytes).expect("read back");
    assert!(look.cdl.is_some(), "the look file's CDL survives");
    Arc::new(
        LookTransform::new(look.input, look.cdl, Some(Lut::from_cube(look.lut))).expect("look"),
    )
}

/// (name, look, whether it is a log-to-log look with a C10 of its own).
fn looks() -> Vec<(&'static str, Arc<LookTransform>, bool)> {
    let lut = cube();
    let cdl = cdl();
    vec![
        (
            "alf4",
            from_file(&AlfFile::alf4(&lut, Some(&cdl), "", "x.cube")),
            true,
        ),
        (
            "alf4c",
            from_file(&AlfFile::alf4c(
                Some(&lut),
                Some(&cdl),
                Some(&Lut3d::identity(5).expect("identity")),
                Some(&Lut3d::identity(5).expect("identity")),
                "",
            )),
            true,
        ),
        (
            "aml",
            from_file(&AlfFile::aml(&lut, Some(&cdl), "x.cube")),
            false,
        ),
    ]
}

fn chain_with(target: OutputTarget, look: &Arc<LookTransform>, pct: f64) -> ChainParams {
    let mut p = chains::base(target);
    p.look = Some(LookSetting::new(Arc::clone(look), pct));
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
fn looks_from_look_files_through_the_gpu_match_the_cpu() {
    let (w, h) = (61usize, 43usize);
    let img = camera_image(w, h, 0xA1F4);
    let n = w * h;
    let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
    for dev in gpus() {
        for (name, look, log_to_log) in looks() {
            let mut worst_c10: Option<f64> = None;
            let mut worst_c13 = 0.0_f64;
            for target in [
                OutputTarget::Viewport,
                OutputTarget::Srgb,
                OutputTarget::Rec2020,
            ] {
                for pct in [25.0, 60.0, 100.0] {
                    let params = chain_with(target, &look, pct);
                    let chain = ColorChain::new(&params).expect("chain");
                    let traces: Vec<_> = cams.iter().map(|&c| chain.trace(c)).collect();
                    // The look must actually move pixels, or this proves nothing.
                    let without = ColorChain::new(&chains::base(target)).expect("chain");
                    let moved = cams
                        .iter()
                        .zip(&traces)
                        .filter(|(c, t)| {
                            let plain = without.trace(**c);
                            (0..3).any(|k| (t.c13_encoded[k] - plain.c13_encoded[k]).abs() > 2e-3)
                        })
                        .count();
                    assert!(
                        moved * 2 > n,
                        "{name} {target:?} {pct} %: only {moved} of {n} pixels move"
                    );
                    let mut outputs = vec![ColorOutput::Stage(Stop::C13)];
                    if log_to_log {
                        outputs.push(ColorOutput::Stage(Stop::C10));
                    }
                    let gpu = render(dev, &chain, &img, &outputs);
                    let cpu_c13: Vec<Vec3> = traces.iter().map(|t| t.c13_encoded).collect();
                    let d13 = worst(planar(&gpu[0]), &cpu_c13);
                    assert!(
                        d13 <= CODE10,
                        "{:?} {name} {target:?} {pct} %: C13 |GPU - CPU| = {d13:.3e}",
                        dev.mode
                    );
                    worst_c13 = worst_c13.max(d13);
                    if log_to_log {
                        let cpu_c10: Vec<Vec3> = traces.iter().map(|t| t.c10_logc4.0).collect();
                        let d10 = worst(planar(&gpu[1]), &cpu_c10);
                        assert!(
                            d10 <= LOGC4_GATE,
                            "{:?} {name} {target:?} {pct} %: C10 |GPU - CPU| = {d10:.3e}",
                            dev.mode
                        );
                        worst_c10 = Some(worst_c10.unwrap_or(0.0).max(d10));
                    }
                }
            }
            // A look that replaces the display transform has no log-to-log output to compare.
            let c10 = worst_c10.map_or(
                "none (it replaces the display transform)".to_string(),
                |d| format!("{d:.2e} LogC4 (gate {LOGC4_GATE:.0e})"),
            );
            eprintln!(
                "{:?}: {name}, 9 chains, worst |GPU - CPU|: C10 {c10}, C13 {worst_c13:.2e} (gate {CODE10:.1e})",
                dev.mode
            );
        }
    }
}
