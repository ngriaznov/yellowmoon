//! The input of L1 and L2 with the colour of the light it holds (`kernels::lens_light`,
//! `shaders/lens_light.wgsl`, `lens_light_colour.wgsl`), against its CPU reference (`ym_lens::lit_input`,
//! `Restore::light_colour`; `docs/21` section 4, *As built (the light's colour)*).
//!
//! * **Parity.** The kernel matches the CPU reference within [`PARITY_BOUND`] of the largest value on an image
//!   whose pixels cover every case of the rule: nothing saturated, green alone, red and blue, two channels, every
//!   channel at the clip pattern, a recovered neutral core, and the steps of the ramp between them.
//! * **Untouched pixels are exact**: a pixel the rule leaves alone has the input's bits.
//! * **A tile equals the whole frame bit for bit** (D5 of `docs/17-gpu.md`): the rule is a function of one pixel.
//! * **Poison**: a poisoned and a plain device give the same bits and nothing is NaN.

mod common;

use common::lens::{amounts, gpu, plain_gpu, serial};
use ym_gpu::device::GpuContext;
use ym_gpu::kernels::lens_light::light;
use ym_gpu::queue::{JobSpec, Priority};
use ym_lens::{Axis, FrameInfo, ImageRef, LensKind, LensPlan, LightPass, lit_input};

/// The largest difference between the kernel and the CPU reference, relative to the largest value of the
/// image. The arithmetic is a handful of `f32` operations on values of a few units: the first measurement
/// on Mesa lavapipe was 0 (identical to the last bit, the rule has no accumulation); the bound is a few
/// ulps and is never loosened.
const PARITY_BOUND: f32 = 4e-7;

/// The same on a hardware adapter (fast-math fuses the products and sums).
const PARITY_BOUND_HARDWARE: f32 = 2e-6;

const G0: [f32; 3] = [2.61, 1.0, 1.95];

fn plan_for(w: usize, h: usize) -> LensPlan {
    let frame = FrameInfo {
        width: w,
        height: h,
        axis: Axis::X,
        f_number: None,
        clip: G0,
    };
    LensPlan::new(
        common::lens::library(),
        &amounts(LensKind::Vintage, 0.0, 60.0, 0.0),
        frame,
        common::lens::camera(),
    )
    .expect("plan")
}

/// An image of patches, each a case of the rule, with a texture under the clip levels around them: every
/// combination of a channel at 0.4, 0.88, 0.93, 0.97, 1.0 and 1.3 clip levels.
fn cases(w: usize, h: usize) -> Vec<f32> {
    let levels = [0.4f32, 0.88, 0.93, 0.97, 1.0, 1.3];
    let n = w * h;
    let mut data = vec![0.0f32; 3 * n];
    let patch = 3;
    let across = w / patch;
    for y in 0..h {
        for x in 0..w {
            let k = (y / patch) * across + x / patch;
            let (a, b, c) = (k % 6, (k / 6) % 6, (k / 36) % 6);
            let jitter = 1.0 + 0.002 * f32::from((x * 7 + y * 13) as u16 % 5);
            for (ch, lv) in [(0, a), (1, b), (2, c)] {
                data[ch * n + y * w + x] = levels[lv] * G0[ch] * jitter;
            }
        }
    }
    // The clip pattern, a recovered neutral core and an unclipped texture in the last rows.
    for x in 0..w {
        let (y0, y1, y2) = (h - 3, h - 2, h - 1);
        for (ch, g) in G0.iter().enumerate() {
            data[ch * n + y0 * w + x] = *g;
            data[ch * n + y1 * w + x] = 3.1;
            data[ch * n + y2 * w + x] = 0.3 + 0.001 * x as f32;
        }
    }
    data
}

fn run<R: Send + 'static>(
    dev: &common::TestGpu,
    job: impl FnOnce(&GpuContext) -> R + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| Ok(job(ctx)))
        .expect("GPU job")
}

fn on_device(dev: &common::TestGpu, data: &[f32], w: usize, h: usize, pass: LightPass) -> Vec<f32> {
    let data = data.to_vec();
    run(dev, move |ctx| {
        light(ctx, &data, w, h, pass).expect("the light pass")
    })
}

#[test]
fn the_kernel_matches_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = gpu() else {
        return;
    };
    let bound = if dev.gpu.info().is_software {
        PARITY_BOUND
    } else {
        PARITY_BOUND_HARDWARE
    };
    let (w, h) = (270usize, 216usize);
    let plan = plan_for(w, h);
    let pass = plan.light_pass().expect("Edges act");
    let data = cases(w, h);
    let image = ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data: &data,
    };
    let want = lit_input(&plan, &image).expect("saturated pixels");
    let got = on_device(dev, &data, w, h, pass);
    let largest = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (mut worst, mut changed, mut exact) = (0.0f32, 0usize, 0usize);
    for i in 0..data.len() {
        worst = worst.max((got[i] - want[i]).abs());
        if want[i] != data[i] {
            changed += 1;
        } else {
            exact += 1;
            assert_eq!(
                got[i].to_bits(),
                data[i].to_bits(),
                "an untouched sample {i} has the input's bits"
            );
        }
        assert!(got[i].is_finite());
    }
    println!(
        "light: worst difference {worst:e} of values up to {largest:e} ({:e} relative), {changed} samples changed, {exact} untouched",
        worst / largest
    );
    assert!(changed > 1000, "the cases change samples: {changed}");
    assert!(
        worst <= bound * largest,
        "worst difference {worst} of values up to {largest}"
    );
}

#[test]
fn a_tile_equals_the_whole_frame_bit_for_bit() {
    let _guard = serial();
    let Some(dev) = gpu() else {
        return;
    };
    let (w, h) = (270usize, 216usize);
    let plan = plan_for(w, h);
    let pass = plan.light_pass().expect("Edges act");
    let data = cases(w, h);
    let whole = on_device(dev, &data, w, h, pass);
    // A tile of 97 x 61 pixels at (53, 41).
    let (tx, ty, tw, th) = (53usize, 41usize, 97usize, 61usize);
    let mut tile = Vec::with_capacity(3 * tw * th);
    for c in 0..3 {
        for y in ty..ty + th {
            tile.extend_from_slice(&data[c * w * h + y * w + tx..c * w * h + y * w + tx + tw]);
        }
    }
    let got = on_device(dev, &tile, tw, th, pass);
    for c in 0..3 {
        for y in 0..th {
            for x in 0..tw {
                let a = got[c * tw * th + y * tw + x];
                let b = whole[c * w * h + (ty + y) * w + tx + x];
                assert_eq!(a.to_bits(), b.to_bits(), "channel {c} at ({x}, {y})");
            }
        }
    }
}

#[test]
fn poison_changes_nothing() {
    let _guard = serial();
    let Some(poisoned) = gpu() else {
        return;
    };
    let Some(plain) = plain_gpu() else {
        return;
    };
    let (w, h) = (90usize, 72usize);
    let plan = plan_for(w, h);
    let pass = plan.light_pass().expect("Edges act");
    let data = cases(w, h);
    let a = on_device(poisoned, &data, w, h, pass);
    let b = {
        let data = data.clone();
        plain
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                light(ctx, &data, w, h, pass)
            })
            .expect("plain GPU job")
    };
    assert!(a.iter().all(|v| v.is_finite()));
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
