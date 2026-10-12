//! The colour pass on the GPU (Mesa lavapipe in CI) against the CPU reference: the frames agree
//! within the parity tolerances of `ym-gpu`. Skipped, loudly, when no adapter opens, unless
//! `YM_REQUIRE_GPU=1` (the `gpu-lavapipe` job) turns that into a failure.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportRect,
    ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in render_gpu: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

#[test]
fn gpu_and_cpu_frames_agree_within_the_parity_tolerance() {
    let Some(gpu) = gpu() else { return };
    let info = gpu.info().clone();
    eprintln!("GPU render tests on:\n{info}");
    let white = light("sony-ilce-7rm4", 4500.0);
    let scene = Scene::new(900, 600, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let cpu =
        Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let on_gpu = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    assert!(on_gpu.gpu_info().is_some());
    let cancel = CancelToken::new();
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    // Plain, then with grade controls, a curve and a straighten.
    let variants: Vec<(&str, PhotoSource)> = vec![
        ("plain", src.clone()),
        (
            "graded",
            edit(&src, |s| {
                s.contrast = 30.0;
                s.brightness = 10.0;
                s.saturation = 20.0;
                s.color_balance.highlights = [0.4, -0.3];
                s.curve.luma = vec![[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]];
                s.curve.blue = vec![[0.0, 0.05], [1.0, 0.95]];
                s.crop = Some(ym_core::settings::Crop {
                    rect: ym_core::settings::NormRect {
                        x: 0.05,
                        y: 0.05,
                        width: 0.9,
                        height: 0.9,
                    },
                    angle_deg: 4.0,
                    aspect: ym_core::settings::CropAspect::Free,
                });
            }),
        ),
    ];
    for (name, v) in variants {
        let a = cpu
            .viewport_frame(&v, &st, &FrameOptions::default(), &cancel)
            .expect("cpu frame");
        let b = on_gpu
            .viewport_frame(&v, &st, &FrameOptions::default(), &cancel)
            .expect("gpu frame");
        assert_eq!(a.report.device, ComputeDevice::Cpu);
        assert_eq!(
            b.report.device,
            ComputeDevice::Gpu,
            "{name}: {:?}",
            b.report.notes
        );
        assert_eq!((a.width, a.height), (b.width, b.height));
        let (mut worst, mut sum) = (0u8, 0u64);
        for (x, y) in a.rgba.iter().zip(&b.rgba) {
            let d = x.abs_diff(*y);
            worst = worst.max(d);
            sum += u64::from(d);
        }
        let mean = sum as f64 / a.rgba.len() as f64;
        eprintln!("{name}: worst difference {worst} codes, mean {mean:.4}");
        // One code of 8 bits at most (the GPU's f16 storage and its own dither lookup agree with
        // the CPU's to better than the 1e-3 gate of ym-gpu), almost always zero.
        assert!(worst <= 2, "{name}: worst difference {worst} codes");
        assert!(mean < 0.25, "{name}: mean difference {mean}");
    }
    // The Before view runs on the CPU whatever is attached.
    let mut before = st.clone();
    before.before = true;
    let b = on_gpu
        .viewport_frame(&src, &before, &FrameOptions::default(), &cancel)
        .expect("before");
    assert_eq!(b.report.device, ComputeDevice::Cpu);
    // Thumbnails go through the same pass.
    let a = cpu.thumbnail_png(&src, &cancel).expect("thumbnail");
    let b = on_gpu
        .thumbnail_png(&src, &cancel)
        .expect("thumbnail on gpu");
    assert_eq!(&a[..8], &b[..8]);
}
