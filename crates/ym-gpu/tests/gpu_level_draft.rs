//! The draft of a zoom (`kernels::level_draft`, `shaders/level_draft.wgsl`) against its CPU reference
//! (`kernels::level_draft::resample_rgba8`).
//!
//! | Case | Asserted | Measured (Mesa lavapipe) |
//! |---|---|---|
//! | six maps (identity, half and double size, a shift of a third of a pixel, a pan that reads past every edge) over random RGBA8 pictures of odd sizes | every code within **1** of the CPU's (the arithmetic is the CPU's `f32` bilinear mix; a fused multiply-add may move a rounding by one code), and at least 99.9 % exact | see the printed numbers |
//! | the identity map | the source, byte for byte | identical |
//!
//! On lavapipe in `gpu-lavapipe` (`YM_REQUIRE_GPU=1`) and on Metal in `mac-gpu.yml`; skipped, loudly, without an adapter.

mod common;

use common::{Rng, gpus};
use ym_gpu::device::GpuError;
use ym_gpu::image::IMAGE_USAGE;
use ym_gpu::kernels::level_draft::{DraftMap, encode_level_draft, resample_rgba8};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::{PassList, upload_bytes};

/// A case: its name, the source size, the map, the frame size.
type Case = (&'static str, (usize, usize), DraftMap, (usize, usize));

fn picture(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    (0..4 * w * h)
        .map(|i| {
            if i % 4 == 3 {
                255
            } else {
                (rng.unit() * 256.0).floor().min(255.0) as u8
            }
        })
        .collect()
}

fn on_device(
    dev: &common::TestGpu,
    src: Vec<u8>,
    src_size: (usize, usize),
    map: DraftMap,
    dst_size: (usize, usize),
) -> Vec<u8> {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut passes = PassList::new(ctx, "draft test");
            let s = passes.keep(ctx.pool().acquire("src", src.len() as u64, IMAGE_USAGE)?);
            upload_bytes(ctx, &s, 0, &src)?;
            let bytes = 4 * (dst_size.0 * dst_size.1) as u64;
            let d = passes.keep(ctx.pool().acquire("dst", bytes, IMAGE_USAGE)?);
            encode_level_draft(&mut passes, &s, src_size, map, &d, dst_size)?;
            let read = passes.read(&d, 0, bytes)?;
            let mut done = passes.submit()?.finish()?;
            Ok::<_, GpuError>(done.take(read))
        })
        .expect("GPU job")
}

#[test]
fn the_draft_kernel_is_the_cpu_reference_within_one_code() {
    for dev in gpus() {
        let cases: [Case; 6] = [
            (
                "identity",
                (37, 23),
                DraftMap {
                    sx: 1.0,
                    tx: 0.0,
                    sy: 1.0,
                    ty: 0.0,
                },
                (37, 23),
            ),
            (
                "half size",
                (64, 48),
                DraftMap {
                    sx: 2.0,
                    tx: 0.0,
                    sy: 2.0,
                    ty: 0.0,
                },
                (32, 24),
            ),
            (
                "double size",
                (21, 17),
                DraftMap {
                    sx: 0.5,
                    tx: 0.0,
                    sy: 0.5,
                    ty: 0.0,
                },
                (42, 34),
            ),
            (
                "a third of a pixel",
                (50, 31),
                DraftMap {
                    sx: 1.0,
                    tx: 1.0 / 3.0,
                    sy: 1.0,
                    ty: -1.0 / 3.0,
                },
                (49, 30),
            ),
            (
                "pan past every edge",
                (40, 30),
                DraftMap {
                    sx: 0.8,
                    tx: -9.5,
                    sy: 1.3,
                    ty: 7.25,
                },
                (60, 40),
            ),
            (
                "odd scale",
                (97, 61),
                DraftMap {
                    sx: 1.37,
                    tx: 3.1,
                    sy: 0.71,
                    ty: 2.2,
                },
                (53, 71),
            ),
        ];
        for (i, (what, src_size, map, dst_size)) in cases.into_iter().enumerate() {
            let src = picture(src_size.0, src_size.1, 11 + i as u64);
            let want = resample_rgba8(&src, src_size, map, dst_size);
            let got = on_device(dev, src.clone(), src_size, map, dst_size);
            assert_eq!(got.len(), want.len(), "{what}");
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            let exact =
                got.iter().zip(&want).filter(|(a, b)| a == b).count() as f64 / got.len() as f64;
            let line = format!(
                "[gpu_level_draft] {what}: worst {worst} code, {:.4} % exact",
                exact * 100.0
            );
            println!("{line}");
            let _ = std::io::Write::write_fmt(&mut std::io::stderr(), format_args!("{line}\n"));
            assert!(worst <= 1, "{what}: {worst} codes");
            assert!(exact >= 0.999, "{what}: {exact}");
            if what == "identity" {
                assert_eq!(got, src, "the identity map is the source");
            }
        }
    }
}
