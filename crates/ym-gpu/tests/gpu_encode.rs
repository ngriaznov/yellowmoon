//! G-03: the device entry points of the kernels (`encode_*`) chained on the device against the host
//! wrappers they are the layer under.
//!
//! The chain keeps a tile's planes on the device from the mosaic to the end of the raw stages: the
//! demosaic's output image is the diffraction deconvolution's input image, in one command list, with
//! no readback in between. Nothing in that hand-off may change a bit, and the odd widths here
//! (rows padded to an even stride on the device) are what could:
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | adaptive ratio into diffraction, one list, odd and even sizes, shadow matching off and on | bit-identical to the host wrappers run one after the other | 0 differing samples |
//! | log-guided into diffraction, one list, odd and even sizes | bit-identical to the host wrappers | 0 differing samples |
//! | a device-to-device run | one submission, one readback | |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); the Metal runners run the same file (`mac-gpu.yml`).

mod common;

use common::exact::{assert_all_finite, assert_bit_equal};
use common::{TestGpu, gpus};
use ym_gpu::cpu::demosaic::DemosaicInputs;
use ym_gpu::cpu::demosaic::adaptive_ratio::Params;
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage, synth};
use ym_gpu::device::StorageMode;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::adaptive_ratio::{self, Balance, PreparedDemosaic};
use ym_gpu::kernels::diffraction::{self, PreparedDiffraction};
use ym_gpu::kernels::log_guided;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;
use ym_noise::model::NoiseModel;

/// The stage must change a scene by at least this much (normalised units) for a comparison of its
/// device and host runs to mean something.
const MIN_EFFECT: f32 = 1e-3;

/// The device with `f32` storage (the log-guided entry point needs it).
fn f32_device_or_skip() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// A noisy natural scene sampled through `cfa`, normalised units.
fn scene(width: usize, height: usize, cfa: Cfa, seed: u64) -> Mosaic {
    let mut m = synth::mosaic_from_rgb(&synth::natural(width, height, seed), cfa);
    for v in &mut m.data {
        *v *= 0.8;
    }
    synth::add_noise(&mut m.data, 0.004, seed ^ 0x5eed);
    for v in &mut m.data {
        *v = v.max(-0.05);
    }
    m
}

fn noise(a: f64, b: f64, sigma_row: f64, q: f64) -> NoiseDn {
    NoiseDn::from_model(&NoiseModel {
        a,
        b,
        delta: [0.0; 4],
        sigma_row,
        q,
    })
}

/// An active diffraction plan (a small pixel pitch at a small aperture: a wide kernel).
fn diffraction_plan(gains: [f32; 3], noise: &NoiseDn) -> DiffractionPlan {
    let params = DiffractionParams::new(3.76, 11.0)
        .with_scale(1.0)
        .with_strength(1.0);
    DiffractionPlan::new(&params, gains, noise, SPEC_LUMA).expect("an active plan")
}

/// What the two paths of one case produce, and what the device path cost.
struct Outcome {
    host: RgbImage,
    device: RgbImage,
    /// The largest change the deconvolution made to the demosaiced image (the stage must act, or the
    /// comparison proves little).
    effect: f32,
    submissions: u64,
    readbacks: u64,
}

fn max_change(a: &RgbImage, b: &RgbImage) -> f32 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn check(label: &str, o: &Outcome) {
    assert_all_finite(label, &o.device.data);
    assert_bit_equal(label, &o.device.data, &o.host.data);
    assert_eq!(
        (o.device.width, o.device.height, o.device.place),
        (o.host.width, o.host.height, o.host.place),
        "{label}: size and placement"
    );
    println!(
        "{label}: the stage changes the image by up to {:.3e}",
        o.effect
    );
    assert!(
        o.effect > MIN_EFFECT,
        "{label}: the stage did nothing ({:e})",
        o.effect
    );
    assert_eq!(
        o.submissions, 1,
        "{label}: the device-to-device run is one submission"
    );
    assert_eq!(o.readbacks, 1, "{label}: and one readback");
}

/// The adaptive ratio demosaic, then the diffraction deconvolution: the host wrappers one after the
/// other, and the `encode_*` entry points in one command list.
fn adaptive_into_diffraction(
    dev: &TestGpu,
    m: &Mosaic,
    gains: [f32; 3],
    noise: NoiseDn,
) -> Outcome {
    let m = m.clone();
    let plan = diffraction_plan(gains, &noise);
    dev.gpu
        .run(JobSpec::new(Priority::Export), move |ctx, _| {
            let inputs = DemosaicInputs {
                gains,
                noise,
                ca: None,
            };
            let params = Params::default();
            let diff = PreparedDiffraction::new(ctx, &plan)?;

            // The host path: two runs, the planes cross the host in between.
            let demosaiced =
                adaptive_ratio::demosaic(ctx, &m, Balance::Unbalanced, &inputs, &params)?;
            let host = diff.run(ctx, &demosaiced)?;
            let effect = max_change(&host, &demosaiced);

            // The device path: upload once, record both kernels, read once.
            let frame = PreparedDemosaic::new(Balance::Unbalanced, &inputs, &params)?;
            let before = ctx.counters();
            let input = GpuImage::upload(ctx, "mosaic", m.width, m.height, 1, &m.data, m.place)?;
            let mut passes = PassList::new(ctx, "adaptive ratio into diffraction");
            let rgb = adaptive_ratio::encode_demosaic(
                &mut passes,
                &input,
                m.cfa,
                Rect::whole(m.width, m.height),
                &frame,
            )?;
            let out = diffraction::encode_sharpen(
                &mut passes,
                &diff,
                &rgb,
                Rect::whole(m.width, m.height),
            )?;
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let samples = out.take(&done, read)?;
            let used = ctx.counters().since(before);
            let device = RgbImage {
                width: out.width,
                height: out.height,
                place: out.place,
                data: samples,
            };
            Ok(Outcome {
                host,
                device,
                effect,
                submissions: used.submissions,
                readbacks: used.readbacks,
            })
        })
        .expect("the runs")
}

#[test]
fn adaptive_ratio_into_diffraction_on_the_device_equals_the_host_wrappers() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let gains = [1.9, 1.0, 1.4];
    let low = noise(2e-5, 1e-8, 0.0, 1.0 / 65535.0);
    let high = noise(5e-4, 4e-5, 1e-3, 0.0);
    // Odd widths and heights give the device images a padded row; the second set has shadow
    // matching (the high-noise model), which changes the passes and the scratch layout.
    for (w, h, cfa, seed, nz, label) in [
        (83, 61, Cfa::RGGB, 3, low, "83 x 61, low noise"),
        (96, 64, Cfa::GRBG, 5, low, "96 x 64, low noise"),
        (71, 50, Cfa::BGGR, 7, high, "71 x 50, shadow matching"),
        (64, 47, Cfa::GBRG, 9, high, "64 x 47, shadow matching"),
    ] {
        let m = scene(w, h, cfa, seed);
        let o = adaptive_into_diffraction(dev, &m, gains, nz);
        check(label, &o);
    }
}

/// The log-guided demosaic, then the diffraction deconvolution.
fn log_guided_into_diffraction(dev: &TestGpu, m: &Mosaic, nz: NoiseDn) -> Outcome {
    let m = m.clone();
    let plan = diffraction_plan([1.0; 3], &nz);
    dev.gpu
        .run(JobSpec::new(Priority::Export), move |ctx, _| {
            let diff = PreparedDiffraction::new(ctx, &plan)?;
            let demosaiced = log_guided::demosaic(ctx, &m, nz, 0)?;
            let host = diff.run(ctx, &demosaiced)?;
            let effect = max_change(&host, &demosaiced);

            let before = ctx.counters();
            let input = GpuImage::upload(ctx, "mosaic", m.width, m.height, 1, &m.data, m.place)?;
            let mut passes = PassList::new(ctx, "log-guided into diffraction");
            let rgb = log_guided::encode_demosaic(&mut passes, &input, m.cfa, nz, 0)?;
            let out = diffraction::encode_sharpen(
                &mut passes,
                &diff,
                &rgb,
                Rect::whole(m.width, m.height),
            )?;
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let samples = out.take(&done, read)?;
            let used = ctx.counters().since(before);
            Ok(Outcome {
                host,
                device: RgbImage {
                    width: out.width,
                    height: out.height,
                    place: out.place,
                    data: samples,
                },
                effect,
                submissions: used.submissions,
                readbacks: used.readbacks,
            })
        })
        .expect("the runs")
}

#[test]
fn log_guided_into_diffraction_on_the_device_equals_the_host_wrappers() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let nz = noise(2e-5, 1e-8, 0.0, 1.0 / 65535.0);
    for (w, h, cfa, seed, label) in [
        (83, 61, Cfa::RGGB, 3, "83 x 61"),
        (96, 64, Cfa::GRBG, 5, "96 x 64"),
        (45, 71, Cfa::BGGR, 11, "45 x 71"),
    ] {
        let m = scene(w, h, cfa, seed);
        let o = log_guided_into_diffraction(dev, &m, nz);
        check(label, &o);
    }
}
