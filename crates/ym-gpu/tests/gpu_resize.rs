//! G-08: the Lanczos-3 resize of the export tail (`kernels::resize`, `shaders/resize.wgsl`) against
//! the CPU resize (`ym_export::resize::resize_lanczos3`, whose taps are `kernels::resize::Taps`).
//!
//! The CPU reference here is that function on planar images: the same taps, `f64` sums, the
//! horizontal pass first with an `f32` intermediate image, then the vertical pass.
//!
//! * **Parity.** The kernel is within [`VALUE_BOUND`] of the reference on natural and hard-edged
//!   scenes, shrinks (also by a large factor), enlargements, odd sizes and the identity size.
//! * **A flat field is exact** (bit for bit), for any value and any size, by the difference form.
//! * **Step overshoot** under 10 % of the step, like the CPU resize.
//! * **Rows are independent** (D5 of the plan): the horizontal pass over bands of rows and the
//!   vertical pass over bands of output rows, each reading a gathered window with its own first row,
//!   give the whole image's bits.
//! * Bad ranges are refused, a wide row works (index arithmetic at 8658 px), the number of threads
//!   of the software adapter changes no bit, and the kernel's largest binding fits the design size.
//! * Poison: every test device fills its buffers with NaN, so a read of what nobody wrote fails.

mod common;
#[path = "g06/mod.rs"]
mod g06;
#[path = "g08/mod.rs"]
mod g08;

use common::{TestGpu, gpus};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::{Placement, RgbImage};
use ym_gpu::device::GpuError;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::resize::{
    PreparedTaps, Taps, encode_resize_h, encode_resize_v, max_binding_bytes, resize,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// Largest absolute difference to the CPU reference over [`cases`], white = 1. First measured
/// maximum on Mesa lavapipe: 3.58e-7 (three ulps of a value near 1, on the 64 x 64 enlargement to
/// 190 x 190: the f32 sum of the taps against the CPU's f64 sum, and the f32 intermediate image on
/// both sides). The bound is twice that and is never loosened. The plan's proposal was 1e-5.
const VALUE_BOUND: f32 = 7.2e-7;

fn run<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Export), move |ctx, _| job(ctx))
        .expect("GPU job")
}

fn cpu_resize(img: &RgbImage, out_w: usize, out_h: usize) -> Vec<f32> {
    g08::cpu_resize(&img.data, img.width, img.height, out_w, out_h)
}

fn gpu_resize(dev: &TestGpu, img: &RgbImage, out_w: usize, out_h: usize) -> Vec<f32> {
    let img = img.clone();
    run(dev, move |ctx| resize(ctx, &img, out_w, out_h)).data
}

/// A scene with hard edges (the case that rings), smooth gradients and noise.
fn scene(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut img = synth::natural(w, h, seed);
    let n = w * h;
    for y in 0..h {
        for x in 0..w {
            if (x / 7 + y / 5) % 9 == 0 {
                for c in 0..3 {
                    img.data[c * n + y * w + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                }
            }
        }
    }
    img
}

/// (input, output): shrinks of 1.1 to 8.7 x, enlargements, the identity, odd and tiny sizes.
fn cases() -> Vec<((usize, usize), (usize, usize))> {
    vec![
        ((97, 61), (40, 25)),
        ((97, 61), (97, 61)),
        ((97, 61), (300, 200)),
        ((97, 61), (1, 1)),
        ((97, 61), (50, 61)),
        ((61, 97), (61, 33)),
        ((1, 1), (1, 1)),
        ((1, 9), (4, 1)),
        ((9, 1), (1, 4)),
        ((255, 129), (64, 33)),
        ((64, 64), (190, 190)),
        ((333, 222), (301, 199)),
        ((2048, 1365), (1024, 683)),
        ((1200, 300), (138, 35)),
    ]
}

#[test]
fn matches_the_cpu_resize_on_scenes_of_every_kind_of_size() {
    for dev in gpus() {
        let mut worst = (0.0f32, String::new());
        for (i, &((w, h), (ow, oh))) in cases().iter().enumerate() {
            let img = scene(w, h, 100 + i as u64);
            let want = cpu_resize(&img, ow, oh);
            let got = gpu_resize(dev, &img, ow, oh);
            assert_eq!(got.len(), want.len());
            let e = g06::compare(&got, &want);
            assert!(
                e.max.is_finite() && e.max <= VALUE_BOUND,
                "{w}x{h} -> {ow}x{oh}: max error {:e} at sample {}, bound {VALUE_BOUND:e}",
                e.max,
                e.at
            );
            if e.max >= worst.0 {
                worst = (e.max, format!("{w}x{h} -> {ow}x{oh}"));
            }
        }
        eprintln!(
            "[G-08] resize on {}: worst error {:e} ({})",
            dev.gpu.info().name,
            worst.0,
            worst.1
        );
    }
}

#[test]
fn a_flat_field_comes_out_exact() {
    for dev in gpus() {
        for value in [0.0f32, 0.18, 1.0, 0.123_456_79, 7.5, -0.3] {
            for &((w, h), (ow, oh)) in &cases() {
                let mut img = RgbImage::new(w, h);
                img.data.fill(value);
                let got = gpu_resize(dev, &img, ow, oh);
                assert!(
                    got.iter().all(|&v| v.to_bits() == value.to_bits()),
                    "flat {value} {w}x{h} -> {ow}x{oh} is not exact"
                );
            }
        }
        // Three different flat planes stay three flat planes.
        let mut img = RgbImage::new(50, 40);
        let n = 50 * 40;
        img.data[..n].fill(0.25);
        img.data[n..2 * n].fill(0.5);
        img.data[2 * n..].fill(0.75);
        let got = gpu_resize(dev, &img, 17, 13);
        let m = 17 * 13;
        assert!(got[..m].iter().all(|&v| v == 0.25));
        assert!(got[m..2 * m].iter().all(|&v| v == 0.5));
        assert!(got[2 * m..].iter().all(|&v| v == 0.75));
    }
}

#[test]
fn the_identity_size_keeps_the_pixels_within_rounding() {
    for dev in gpus() {
        let img = scene(64, 48, 7);
        let got = gpu_resize(dev, &img, 64, 48);
        let e = g06::compare(&got, &img.data);
        assert!(e.max < 1e-6, "identity: {:e}", e.max);
    }
}

#[test]
fn a_hard_edge_overshoots_by_less_than_ten_percent() {
    // 0 -> 1 across the middle of 256 columns, shrunk 4x: Lanczos-3 rings.
    let (w, h) = (256usize, 8usize);
    let mut img = RgbImage::new(w, h);
    let n = w * h;
    for c in 0..3 {
        for y in 0..h {
            for x in w / 2..w {
                img.data[c * n + y * w + x] = 1.0;
            }
        }
    }
    for dev in gpus() {
        let got = gpu_resize(dev, &img, 64, 2);
        let (lo, hi) = got
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(hi > 1.0, "a Lanczos filter overshoots at an edge: {hi}");
        assert!(hi <= 1.10 && lo >= -0.10, "overshoot {hi}, undershoot {lo}");
    }
}

/// Rows in bands: the horizontal pass over bands of rows into one image and the vertical pass over
/// bands of output rows, each from a window that starts at its own first input row, give the bits
/// of the whole.
#[test]
fn bands_of_rows_are_the_whole_image() {
    for dev in gpus() {
        for &((w, h), (ow, oh)) in &[
            ((301usize, 257usize), (117usize, 101usize)),
            ((97, 61), (200, 150)),
            ((64, 513), (64, 120)),
        ] {
            let img = scene(w, h, 5);
            let whole = gpu_resize(dev, &img, ow, oh);
            let banded = run(dev, {
                let img = img.clone();
                move |ctx| {
                    let xs = PreparedTaps::new(ctx, Taps::new(w, ow))?;
                    let ys = PreparedTaps::new(ctx, Taps::new(h, oh))?;
                    let src = GpuImage::upload_rgb(ctx, "src", &img)?;
                    // Horizontal pass in bands of 37 rows, into the rows they came from.
                    let mid = GpuImage::alloc(ctx, "mid", ow, h, 3, Placement::whole(ow, h))?;
                    let mut y = 0;
                    while y < h {
                        let n = 37.min(h - y);
                        let mut passes = PassList::new(ctx, "h band");
                        encode_resize_h(&mut passes, &xs, &src, y..y + n, &mid, y)?;
                        passes.submit()?.finish()?;
                        y += n;
                    }
                    // Vertical pass in bands of 19 output rows; each reads a window of `mid` that
                    // is a copy of exactly the rows it needs, so its row 0 is input row `first`.
                    let dst = GpuImage::alloc(ctx, "dst", ow, oh, 3, Placement::whole(ow, oh))?;
                    let mid_host = mid.read(ctx)?;
                    let mut o = 0;
                    while o < oh {
                        let n = 19.min(oh - o);
                        let span = ys.taps().input_span(o..o + n);
                        let rows = span.len();
                        let mut win = vec![0.0f32; 3 * ow * rows];
                        for c in 0..3 {
                            for r in 0..rows {
                                let from = c * ow * h + (span.start + r) * ow;
                                win[c * ow * rows + r * ow..][..ow]
                                    .copy_from_slice(&mid_host[from..from + ow]);
                            }
                        }
                        let window = GpuImage::upload(
                            ctx,
                            "window",
                            ow,
                            rows,
                            3,
                            &win,
                            Placement::whole(ow, rows),
                        )?;
                        let mut passes = PassList::new(ctx, "v band");
                        encode_resize_v(&mut passes, &ys, &window, span.start, o..o + n, &dst, o)?;
                        passes.submit()?.finish()?;
                        o += n;
                    }
                    dst.read(ctx)
                }
            });
            common::exact::assert_bit_equal(
                &format!("{w}x{h} -> {ow}x{oh} in bands"),
                &banded,
                &whole,
            );
        }
    }
}

#[test]
fn bad_ranges_are_refused() {
    let Some(dev) = common::gpu() else { return };
    run(dev, |ctx| {
        let xs = PreparedTaps::new(ctx, Taps::new(40, 20))?;
        let ys = PreparedTaps::new(ctx, Taps::new(30, 15))?;
        let src = GpuImage::alloc(ctx, "src", 40, 30, 3, Placement::whole(40, 30))?;
        let wrong_width = GpuImage::alloc(ctx, "w", 21, 30, 3, Placement::whole(21, 30))?;
        let dst = GpuImage::alloc(ctx, "dst", 20, 30, 3, Placement::whole(20, 30))?;
        let one_plane = GpuImage::alloc(ctx, "p", 20, 30, 1, Placement::whole(20, 30))?;
        let mut passes = PassList::new(ctx, "refusals");
        let refused = |r: Result<(), GpuError>, what: &str| {
            assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{what}: {r:?}");
        };
        refused(
            encode_resize_h(&mut passes, &xs, &wrong_width, 0..30, &dst, 0),
            "wrong source width",
        );
        refused(
            encode_resize_h(&mut passes, &xs, &src, 0..30, &wrong_width, 0),
            "wrong destination width",
        );
        refused(
            encode_resize_h(&mut passes, &xs, &src, 0..31, &dst, 0),
            "rows past the source",
        );
        refused(
            encode_resize_h(&mut passes, &xs, &src, 0..30, &dst, 1),
            "rows past the destination",
        );
        refused(
            encode_resize_h(&mut passes, &xs, &src, 0..0, &dst, 0),
            "no rows",
        );
        refused(
            encode_resize_h(&mut passes, &xs, &src, 0..30, &one_plane, 0),
            "one plane",
        );
        // Vertical: the window must hold every input row the outputs read.
        let window = GpuImage::alloc(ctx, "win", 20, 10, 3, Placement::whole(20, 10))?;
        let span = ys.taps().input_span(5..8);
        assert!(span.len() > 10, "the test needs a window that is too small");
        refused(
            encode_resize_v(&mut passes, &ys, &window, span.start, 5..8, &dst, 0),
            "a window that is too small",
        );
        refused(
            encode_resize_v(&mut passes, &ys, &dst, 0, 14..16, &dst, 0),
            "outputs past the resize",
        );
        refused(
            encode_resize_v(&mut passes, &ys, &wrong_width, 0, 0..2, &dst, 0),
            "wrong width",
        );
        Ok(())
    });
}

/// A row 8658 px wide (the widest Sony frame) squeezed to 2000: the indices are what is tested.
#[test]
fn a_wide_row_works() {
    for dev in gpus() {
        let (w, h) = (8658usize, 6usize);
        let img = scene(w, h, 11);
        let (ow, oh) = (2000usize, 3usize);
        let want = cpu_resize(&img, ow, oh);
        let got = gpu_resize(dev, &img, ow, oh);
        let e = g06::compare(&got, &want);
        assert!(e.max <= VALUE_BOUND, "{:e} at {}", e.max, e.at);
    }
}

#[test]
fn the_number_of_threads_of_the_software_adapter_changes_no_bit() {
    if common::gpu().is_none() {
        return;
    }
    g06::assert_thread_count_independent(
        "the_number_of_threads_of_the_software_adapter_changes_no_bit",
        || {
            let dev = common::gpu().expect("device");
            let mut h = g06::FNV_START;
            for (i, &((w, ht), (ow, oh))) in cases().iter().enumerate() {
                let img = scene(w, ht, 300 + i as u64);
                h = g06::fnv(h, &gpu_resize(dev, &img, ow, oh));
            }
            h
        },
    );
}

#[test]
fn the_largest_binding_fits_the_design_size_for_a_slab() {
    // A slab of the export tail: 512 rows of the widest frame, in and out.
    common::limits::assert_max_binding_bytes("resize", Rect::new(0, 0, 8658, 1024), |r| {
        max_binding_bytes(r)
    });
}
