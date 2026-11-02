//! The mask view's overlay on the device against its CPU reference (`docs/23-nodes.md` section 5.8):
//! `out = display + 0.55 m (accent - display)` in display-linear light, in place.
//!
//! | Case | Asserted | Gate |
//! |---|---|---|
//! | a picture that reaches 8 and goes below 0, every `m` from 0 to 1 | device against `overlay_planar` (the formula in `f64`) | 4e-6 absolute |
//! | `m = 0` | the picture's own bits | bit for bit |
//! | a window of a larger picture | the pixels inside tinted, the pixels outside and the letterbox plane untouched | bit for bit |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`).

mod common;

use common::{Rng, TestGpu, gpus};
use ym_color::encode::OutputSpace;
use ym_color::mask::{OVERLAY_WEIGHT, accent_display_linear};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::tint::overlay_planar;
use ym_gpu::device::StorageMode;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::tint::{encode_tint, tint_host};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The largest difference, absolute, between the device's `f32` arithmetic and the CPU's `f64`
/// on pictures up to 8 (an `f32` ulp there is 1e-6).
const GATE: f64 = 4e-6;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

fn picture(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let bright = if (x + y) % 17 == 0 { 8.0 } else { 1.0 };
            let r = (0.45 + 0.4 * (fx * 9.0).sin() * (fy * 4.0).cos()) * bright;
            let g = 0.15 + 0.7 * fx * fy + 0.03 * rng.range(-1.0, 1.0);
            let b = if (x * 3 + y) % 23 == 0 {
                -0.05
            } else {
                0.9 - 0.6 * fx
            };
            img.set_pixel(x, y, [r as f32, g as f32, b as f32]);
        }
    }
    img
}

fn accent() -> [f64; 3] {
    accent_display_linear(OutputSpace::DisplayP3)
}

fn accent32() -> [f32; 3] {
    accent().map(|c| c as f32)
}

#[test]
fn the_device_overlay_is_the_formula_to_the_rounding_of_f32() {
    let (w, h) = (53, 37);
    let img = picture(w, h, 0x7127);
    let m: Vec<f32> = (0..w * h)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => 1.0,
            _ => (i as f32 * 0.37).fract(),
        })
        .collect();
    let mut want = img.clone();
    overlay_planar(&mut want, &m, accent());
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let (image, m) = (img.clone(), m.clone());
        let got = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                tint_host(ctx, &image, &m, accent32(), OVERLAY_WEIGHT as f32)
            })
            .expect("tint");
        let worst = got
            .data
            .iter()
            .zip(&want.data)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max);
        eprintln!("[gpu_tint] overlay worst difference {worst:.2e} (gate {GATE:.0e})");
        assert!(worst <= GATE, "worst {worst:e}");
        // A pixel with m = 0 is the picture's own, to the bit.
        for i in (0..w * h).filter(|i| i % 7 == 0) {
            for c in 0..3 {
                assert_eq!(got.data[c * w * h + i], img.data[c * w * h + i], "{i} {c}");
            }
        }
    }
    if !seen {
        eprintln!("[gpu_tint] no f32 device here");
    }
}

#[test]
fn a_window_tints_its_pixels_and_leaves_the_rest_and_the_letterbox_plane() {
    let (w, h) = (40, 30);
    let img = picture(w, h, 0x51);
    // A four-plane picture: the fourth plane marks the letterbox (a recognisable pattern).
    let mut data = img.data.clone();
    data.extend((0..w * h).map(|i| if i % 5 == 0 { 1.0f32 } else { 0.0 }));
    let m = vec![1.0f32; w * h];
    let window = Rect::new(7, 5, 20, 11);
    for dev in f32_devices() {
        let (upload, m) = (data.clone(), m.clone());
        let out = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let picture = GpuImage::upload(
                    ctx,
                    "picture",
                    w,
                    h,
                    4,
                    &upload,
                    ym_gpu::cpu::Placement::whole(w, h),
                )?;
                let mask = GpuImage::upload(
                    ctx,
                    "mask",
                    w,
                    h,
                    1,
                    &m,
                    ym_gpu::cpu::Placement::whole(w, h),
                )?;
                let mut passes = PassList::new(ctx, "tint window");
                encode_tint(
                    &mut passes,
                    &picture,
                    window,
                    &mask,
                    accent32(),
                    OVERLAY_WEIGHT as f32,
                )?;
                let read = picture.encode_read(&mut passes)?;
                let done = passes.submit()?.finish()?;
                picture.take(&done, read)
            })
            .expect("tint window");
        let n = w * h;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let inside = window.x <= x && x < window.x1() && window.y <= y && y < window.y1();
                for c in 0..3 {
                    let got = out[c * n + i];
                    let was = img.data[c * n + i];
                    if inside {
                        let want = f64::from(was)
                            + OVERLAY_WEIGHT * (accent()[c] - f64::from(was));
                        assert!((f64::from(got) - want).abs() <= GATE, "({x}, {y}) {c}");
                    } else {
                        assert_eq!(got, was, "({x}, {y}) {c} outside the window");
                    }
                }
                assert_eq!(out[3 * n + i], data[3 * n + i], "letterbox plane at ({x}, {y})");
            }
        }
    }
}
