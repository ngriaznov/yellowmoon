//! The application's wgpu device imported into burn (`docs/17-gpu.md` sections 14 and 12; feature `burn`).
//!
//! `ym-gpu` opens the device (its `ml` feature requests it the way cubecl requests one: every adapter feature but
//! `MAPPABLE_PRIMARY_BUFFERS` and the experimental ones, the adapter's limits), burn registers that device and queue
//! through `Device::wgpu_options().setup(..)` (`import_device`), and then, on the one queue:
//!
//! 1. a tensor op runs on the imported device,
//! 2. the `jdd` golden tile runs through `BurnJdd::with_device` and matches PyTorch within 2e-3 (the
//!    `burn_backend.rs` tolerance),
//! 3. a `ym-gpu` kernel (the Lanczos-3 resize, flat field exact) runs on the same device,
//! 4. burn runs again afterwards, and gives the same bytes as before the kernel.
//!
//! Without the experimental opt-in: the device is requested without the experimental features and burn accepts
//! it, so no `unsafe` is needed for it. Runs on any wgpu adapter, Mesa lavapipe included
//! (`WGPU_BACKEND=vulkan`); Metal on the Apple runners. Without an adapter the test is skipped with a message,
//! unless `YM_REQUIRE_GPU=1` (the `docs/13` convention).
#![cfg(feature = "burn")]

mod common;

use burn::tensor::{Device, Tensor, TensorData};
use common::{load_tensors, read, tile_from};
use ym_gpu::cpu::RgbImage;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{JobSpec, Priority};
use ym_ml::backends::burn_backend::{BurnJdd, import_device};
use ym_ml::model::{ExecContext, TileModel, TileOutput, TileShape};

/// The application's device, or `None` with a notice when there is no adapter.
fn start() -> Option<Gpu> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            eprintln!("skipped: no wgpu adapter ({e})");
            None
        }
    }
}

fn tensor_op(device: &Device) -> Vec<f32> {
    Tensor::<1>::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), device)
        .mul_scalar(2.0)
        .add_scalar(1.0)
        .into_data()
        .try_to_vec::<f32>()
        .expect("the tensor's data")
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn burn_runs_on_the_device_of_the_image_pipeline_and_they_share_one_queue() {
    let Some(gpu) = start() else { return };
    let device = match import_device(&gpu) {
        Ok(d) => d,
        Err(e) => panic!("burn did not accept the application's device: {e}"),
    };

    // 1. A tensor op on the imported device.
    assert_eq!(tensor_op(&device), [3.0, 5.0, 7.0]);

    // 2. A golden tile of the network, on the same device.
    let tiles = load_tensors(&read("golden_tiles.safetensors"));
    let weights = read("golden_small.safetensors");
    let (shape, input) = tiles
        .get("small.in.0")
        .expect("the first small golden tile");
    let ts = TileShape {
        height: shape[1],
        width: shape[2],
    };
    let model = BurnJdd::with_device(&weights, device.clone())
        .expect("weights on the imported device")
        .with_tile_shape(ts);
    let tin = tile_from(ts, input);
    let mut before = TileOutput::zeros(ts);
    model
        .run(&tin, &mut before, &ExecContext::cpu())
        .expect("burn on the imported device");
    let want = &tiles["small.out.0"].1;
    let d_torch = max_diff(before.as_slice(), want);
    eprintln!(
        "burn on the application's device ({} / {}): golden tile vs torch {d_torch:e}",
        gpu.info().name,
        gpu.info().backend
    );
    assert!(d_torch <= 2e-3, "burn vs torch {d_torch:e}");

    // 3. A kernel of ym-gpu on the same queue: a flat field stays exactly flat through the resize.
    let (w, h) = (96_usize, 64_usize);
    let mut flat = RgbImage::new(w, h);
    for c in 0..3 {
        flat.plane_mut(c).fill(0.5);
    }
    let resized = gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            ym_gpu::kernels::resize::resize(ctx, &flat, w / 2, h / 2)
        })
        .expect("a ym-gpu kernel on the device burn imported");
    assert_eq!((resized.width, resized.height), (w / 2, h / 2));
    assert!(
        resized.data.iter().all(|&v| v == 0.5),
        "the flat field is exact"
    );

    // 4. Burn again, after the kernel: nothing of the kernel's work changes burn's answer.
    assert_eq!(tensor_op(&device), [3.0, 5.0, 7.0]);
    let mut after = TileOutput::zeros(ts);
    model
        .run(&tin, &mut after, &ExecContext::cpu())
        .expect("burn again");
    assert!(
        before.as_slice() == after.as_slice(),
        "burn gave other bytes after a ym-gpu job on the same queue"
    );
}
