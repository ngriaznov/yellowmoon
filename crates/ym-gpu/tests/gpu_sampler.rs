//! G-04: the geometry sampler (`kernels::sampler`, `shaders/sampler.wgsl`) against its CPU port
//! (`cpu::sampler`, a port of `ym-core`'s `sample_to_camera`).
//!
//! What is compared, and with which bound:
//!
//! * **Copy is exact.** On an exact grid (no straighten angle, one buffer pixel per output pixel) the
//!   pixels are the CPU's bit for bit, for all 8 EXIF orientations x 4 quarter turns, with crops, and
//!   with a letterbox (a window larger than the canvas).
//! * **Catmull-Rom and the box** match the CPU port within [`VALUE_BOUND`] on rotated, scaled and
//!   magnified grids, again for every orientation and turn. The bound is twice the first measured
//!   maximum (recorded in the constant's note) and is never loosened.
//! * **The outside mask is identical** in every case (the host hands the kernel the CPU predicate's
//!   own intervals), as is the zero in the outside pixels.
//! * **Windows are independent** (D5 of the plan): a tile of the output equals the same pixels of the
//!   whole output bit for bit, for exact and for general grids, and so does sampling from a cut-out
//!   source image with another origin.
//! * **The far corner** of an 8658 x 5772 frame is sampled as exactly as the near one (D4).
//! * Poison: a poisoned and a plain device give the same bits and nothing is NaN.

mod common;

use common::{Rng, TestGpu, gpus};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::sampler::{Kernel, Sampled, SamplerGrid, sample_to_camera};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::GpuError;
use ym_gpu::kernels::sampler::{max_binding_bytes, sample};
use ym_gpu::queue::{JobSpec, Priority};

/// Largest absolute difference between the kernel and the CPU port over the rotated / scaled cases.
/// The first measured maximum on Mesa lavapipe (192 cases, every orientation and turn, Catmull-Rom
/// and box) was 5.96e-7, ten ulps of a value near 0.5: the positions are exact to a fraction's
/// precision, what is left is the f32 rounding of the Catmull-Rom weights and sums (the CPU port
/// computes its `t` in f64 and rounds once). The bound is twice that and is never loosened. The plan's
/// proposal was 5e-6.
const VALUE_BOUND: f64 = 1.2e-6;
/// The same on a hardware adapter (fast-math contracts the weights and sums into fused multiply-adds
/// and divides with a reciprocal): the plan's proposal, 5e-6, until the Metal run gives a number.
const VALUE_BOUND_HARDWARE: f64 = 5e-6;

fn run<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("GPU job")
}

fn gpu_sample(
    dev: &TestGpu,
    img: &RgbImage,
    origin: (usize, usize),
    grid: SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Sampled {
    let img = img.clone();
    run(dev, move |ctx| sample(ctx, &img, origin, &grid, off, w, h))
}

/// A smooth colour field with texture and a little noise: gradients of order 1/size per pixel plus
/// noise of 0.02, in [0, 1].
fn scene(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let r = 0.5 + 0.35 * (fx * 7.0).sin() * (fy * 3.0).cos() + 0.02 * rng.range(-1.0, 1.0);
            let g = 0.2 + 0.6 * fx * fy + 0.02 * rng.range(-1.0, 1.0);
            let b = 0.9 - 0.5 * fx + 0.2 * (fy * 11.0).sin() + 0.02 * rng.range(-1.0, 1.0);
            img.set_pixel(x, y, [r as f32, g as f32, b as f32]);
        }
    }
    img
}

fn swaps(exif: u16, turns: u8) -> bool {
    ((5..=8).contains(&exif)) != (turns % 2 == 1)
}

/// The grid of an exact output: one buffer pixel per output pixel, the canvas is the buffer with its
/// sides swapped by the turns. Returns the canvas size too.
fn exact_grid(bw: usize, bh: usize, exif: u16, turns: u8) -> (SamplerGrid, usize, usize) {
    let (cw, ch) = if swaps(exif, turns) {
        (bh, bw)
    } else {
        (bw, bh)
    };
    let grid = SamplerGrid::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        (cw as f64, ch as f64),
        (bw as f64, bh as f64),
        exif,
        turns,
        true,
    );
    (grid, cw, ch)
}

/// A grid for an output pixel `q` -> canvas `R(-angle) (scale q + t - c) + c` (`c` the canvas centre).
#[allow(clippy::too_many_arguments)] // a test fixture: every parameter of the geometry
fn general_grid(
    bw: usize,
    bh: usize,
    exif: u16,
    turns: u8,
    angle_deg: f64,
    scale: f64,
    t: (f64, f64),
) -> (SamplerGrid, usize, usize) {
    let (cw, ch) = if swaps(exif, turns) {
        (bh, bw)
    } else {
        (bw, bh)
    };
    let (s, c) = (-angle_deg).to_radians().sin_cos();
    let (cx, cy) = (cw as f64 * 0.5, ch as f64 * 0.5);
    // p = R (scale q + t - c) + c
    let to_canvas = [
        [c * scale, -s * scale, c * (t.0 - cx) - s * (t.1 - cy) + cx],
        [s * scale, c * scale, s * (t.0 - cx) + c * (t.1 - cy) + cy],
    ];
    let grid = SamplerGrid::new(
        to_canvas,
        (cw as f64, ch as f64),
        (bw as f64, bh as f64),
        exif,
        turns,
        SamplerGrid::is_unrotated_angle(angle_deg),
    );
    (grid, cw, ch)
}

fn max_diff(a: &RgbImage, b: &RgbImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0, f64::max)
}

fn bit_equal(a: &RgbImage, b: &RgbImage) -> bool {
    a.data.len() == b.data.len()
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn assert_masks_equal(what: &str, gpu: &Sampled, cpu: &Sampled) {
    assert_eq!(
        gpu.outside, cpu.outside,
        "{what}: the outside mask differs from the CPU's"
    );
    if let Some(mask) = &gpu.outside {
        let n = gpu.img.plane_len();
        for c in 0..3 {
            let plane = gpu.img.plane(c);
            for i in 0..n {
                if mask[i] != 0 {
                    assert_eq!(
                        plane[i].to_bits(),
                        0,
                        "{what}: an outside pixel is not zero"
                    );
                }
            }
        }
    }
}

fn all_finite(img: &RgbImage) -> bool {
    img.data.iter().all(|v| v.is_finite())
}

#[test]
fn exact_grids_copy_bit_for_bit_in_every_orientation_and_turn() {
    let Some(dev) = gpus().first() else { return };
    let (bw, bh) = (37usize, 23usize);
    let img = scene(bw, bh, 1);
    for exif in 1..=8u16 {
        for turns in 0..4u8 {
            let (grid, cw, ch) = exact_grid(bw, bh, exif, turns);
            assert_eq!(grid.kernel(), Kernel::Copy);
            // The whole canvas, a crop of it, and a window that reaches past the canvas (a
            // letterbox: the mask is set where the window leaves the photo).
            for (off, w, h) in [
                ((0, 0), cw, ch),
                ((3, 2), cw - 5, ch - 4),
                ((0, 0), cw + 6, ch + 3),
            ] {
                let cpu = sample_to_camera(&img, (0, 0), &grid, off, w, h);
                let gpu = gpu_sample(dev, &img, (0, 0), grid, off, w, h);
                let what = format!("exif {exif} turns {turns} window {off:?} {w}x{h}");
                assert!(bit_equal(&gpu.img, &cpu.img), "{what}: pixels differ");
                assert_masks_equal(&what, &gpu, &cpu);
            }
        }
    }
}

#[test]
fn rotated_scaled_and_magnified_grids_match_the_cpu_port() {
    let Some(dev) = gpus().first() else { return };
    let (bw, bh) = (211usize, 157usize);
    let img = scene(bw, bh, 2);
    let bound = if dev.gpu.info().is_software {
        VALUE_BOUND
    } else {
        VALUE_BOUND_HARDWARE
    };
    let mut worst = 0.0f64;
    let (mut cases, mut boxes, mut bicubics, mut with_mask) = (0usize, 0usize, 0usize, 0usize);
    for exif in 1..=8u16 {
        for turns in 0..4u8 {
            // (angle, scale, translation as a fraction of the canvas): a shrink by 1.9 and 3.1 (the box
            // with 2 and 3 sub-samples), a rotation at 1:1, a small shrink and a magnification.
            for (angle, scale, t) in [
                (0.0, 1.9, (0.05, 0.0)),
                (7.3, 1.0, (-0.1, 0.12)),
                (-31.0, 3.1, (0.0, 0.0)),
                (44.0, 0.7, (0.2, 0.1)),
                (0.0, 0.45, (0.3, 0.3)),
                (-12.5, 1.2, (0.0, -0.1)),
            ] {
                let (cw, ch) = if swaps(exif, turns) {
                    (bh, bw)
                } else {
                    (bw, bh)
                };
                let (grid, _, _) = general_grid(
                    bw,
                    bh,
                    exif,
                    turns,
                    angle,
                    scale,
                    (t.0 * cw as f64, t.1 * ch as f64),
                );
                let (w, h) = (
                    ((cw as f64 / scale) as usize).clamp(8, 180),
                    ((ch as f64 / scale) as usize).clamp(8, 150),
                );
                let cpu = sample_to_camera(&img, (0, 0), &grid, (0, 0), w, h);
                let gpu = gpu_sample(dev, &img, (0, 0), grid, (0, 0), w, h);
                let what = format!("exif {exif} turns {turns} angle {angle} scale {scale} {w}x{h}");
                assert!(all_finite(&gpu.img), "{what}: non-finite");
                assert_masks_equal(&what, &gpu, &cpu);
                let d = max_diff(&gpu.img, &cpu.img);
                assert!(d <= bound, "{what}: max error {d:e} above {bound:e}");
                worst = worst.max(d);
                cases += 1;
                match grid.kernel() {
                    Kernel::Box(_) => boxes += 1,
                    Kernel::Bicubic => bicubics += 1,
                    Kernel::Copy => {}
                }
                with_mask += usize::from(cpu.outside.is_some());
            }
        }
    }
    eprintln!(
        "sampler vs CPU port: {cases} cases ({bicubics} Catmull-Rom, {boxes} box, {with_mask} with a letterbox), worst error {worst:.3e}"
    );
    assert!(
        boxes > 40 && bicubics > 40,
        "the matrix must cover both kernels"
    );
    assert!(with_mask > 40, "and grids with a letterbox");
}

#[test]
fn a_tile_equals_the_same_pixels_of_the_whole_output_bit_for_bit() {
    let Some(dev) = gpus().first() else { return };
    let (bw, bh) = (173usize, 131usize);
    let img = scene(bw, bh, 3);
    for (name, (grid, cw, ch), w, h) in [
        ("exact", exact_grid(bw, bh, 6, 1), 131usize, 173usize),
        (
            "rotated",
            general_grid(bw, bh, 1, 0, 9.0, 1.0, (4.5, -3.25)),
            190,
            150,
        ),
        (
            "box 2",
            general_grid(bw, bh, 3, 2, -3.0, 1.7, (0.0, 0.0)),
            100,
            80,
        ),
        (
            "box 3",
            general_grid(bw, bh, 8, 0, 20.0, 2.6, (10.0, 5.0)),
            70,
            52,
        ),
        (
            "magnified",
            general_grid(bw, bh, 1, 0, -5.0, 0.37, (0.0, 0.0)),
            300,
            220,
        ),
    ] {
        let _ = (cw, ch);
        let whole = gpu_sample(dev, &img, (0, 0), grid, (0, 0), w, h);
        // Tiles of two sizes, on odd offsets.
        for tile in [37usize, 64] {
            let mut x0 = 0;
            while x0 < w {
                let mut y0 = 0;
                while y0 < h {
                    let (tw, th) = (tile.min(w - x0), tile.min(h - y0));
                    let t = gpu_sample(dev, &img, (0, 0), grid, (x0, y0), tw, th);
                    for c in 0..3 {
                        for y in 0..th {
                            for x in 0..tw {
                                let a = t.img.plane(c)[y * tw + x];
                                let b = whole.img.plane(c)[(y0 + y) * w + x0 + x];
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "{name}: tile {tile} at ({x0}, {y0}) differs at ({x}, {y}) of plane {c}"
                                );
                            }
                        }
                    }
                    let mask_t = t.outside.as_ref();
                    for y in 0..th {
                        for x in 0..tw {
                            let a = mask_t.is_some_and(|m| m[y * tw + x] != 0);
                            let b = whole
                                .outside
                                .as_ref()
                                .is_some_and(|m| m[(y0 + y) * w + x0 + x] != 0);
                            assert_eq!(a, b, "{name}: mask at ({}, {})", x0 + x, y0 + y);
                        }
                    }
                    y0 += tile;
                }
                x0 += tile;
            }
        }
    }
}

#[test]
fn a_cut_out_source_with_another_origin_gives_the_same_bits() {
    // The source is the group image the grid samples; a tile reads a window of it with its own
    // origin (the integer origin is folded into the position tables in f64).
    let Some(dev) = gpus().first() else { return };
    let (bw, bh) = (240usize, 180usize);
    let img = scene(bw, bh, 4);
    let (grid, _, _) = general_grid(bw, bh, 1, 0, 6.0, 1.0, (0.0, 0.0));
    let (w, h) = (120usize, 90usize);
    let whole = gpu_sample(dev, &img, (0, 0), grid, (0, 0), w, h);
    // The window of the output at (40, 30), 60 x 50, reads buffer pixels within this rectangle.
    let (off, tw, th) = ((40usize, 30usize), 60usize, 50usize);
    let rect = Rect::new(10, 5, 200, 160);
    let part = img.extract(rect);
    let tile = gpu_sample(dev, &part, (rect.x, rect.y), grid, off, tw, th);
    let reference = gpu_sample(dev, &img, (0, 0), grid, off, tw, th);
    assert!(
        bit_equal(&tile.img, &reference.img),
        "another source origin changed the bits"
    );
    for c in 0..3 {
        for y in 0..th {
            for x in 0..tw {
                assert_eq!(
                    tile.img.plane(c)[y * tw + x].to_bits(),
                    whole.img.plane(c)[(off.1 + y) * w + off.0 + x].to_bits()
                );
            }
        }
    }
}

#[test]
fn the_far_corner_of_a_50_mp_frame_is_as_exact_as_the_near_one() {
    // D4: no f32 coordinate of order 1e4. The grid maps a small output window onto the far corner of
    // an 8658 x 5772 buffer (a slightly rotated, slightly shrunk view of it); the source is the 240 x
    // 200 piece of the buffer around that corner, with its origin. The error against the CPU port
    // (f64 positions) must be as small as for the same window at the near corner.
    let Some(dev) = gpus().first() else { return };
    let (bw, bh) = (8658usize, 5772usize);
    let piece = |ox: usize, oy: usize| {
        let img = scene(240, 200, 5);
        (img, (ox, oy))
    };
    let (angle, scale) = (0.6f64, 1.07f64);
    let (s, c) = (-angle).to_radians().sin_cos();
    let (cx, cy) = (bw as f64 * 0.5, bh as f64 * 0.5);
    let grid = SamplerGrid::new(
        [
            [c * scale, -s * scale, c * (0.0 - cx) - s * (0.0 - cy) + cx],
            [s * scale, c * scale, s * (0.0 - cx) + c * (0.0 - cy) + cy],
        ],
        (bw as f64, bh as f64),
        (bw as f64, bh as f64),
        1,
        0,
        false,
    );
    let mut errors = Vec::new();
    for corner in [(0usize, 0usize), (bw - 240, bh - 200)] {
        let (img, origin) = piece(corner.0, corner.1);
        // The output window whose positions fall in the piece: find it from the CPU grid.
        let mut found = None;
        'search: for oy in (0..bh).step_by(8) {
            for ox in (0..bw).step_by(8) {
                if let Some((bx, by)) = grid.source(ox, oy)
                    && bx > (corner.0 + 40) as f64
                    && by > (corner.1 + 40) as f64
                    && bx < (corner.0 + 120) as f64
                    && by < (corner.1 + 100) as f64
                {
                    found = Some((ox, oy));
                    break 'search;
                }
            }
        }
        let off = found.expect("a window in the piece");
        let (w, h) = (40usize, 30usize);
        let cpu = sample_to_camera(&img, origin, &grid, off, w, h);
        let gpu = gpu_sample(dev, &img, origin, grid, off, w, h);
        assert_masks_equal("far corner", &gpu, &cpu);
        let d = max_diff(&gpu.img, &cpu.img);
        eprintln!("corner {corner:?}: window at {off:?}, max error {d:.3e}");
        errors.push(d);
    }
    let bound = if dev.gpu.info().is_software {
        VALUE_BOUND
    } else {
        VALUE_BOUND_HARDWARE
    };
    assert!(
        errors.iter().all(|&e| e <= bound),
        "{errors:?} above {bound:e}"
    );
}

#[test]
fn poison_and_a_plain_device_give_the_same_bits() {
    let Some(_dev) = gpus().first() else { return };
    let (bw, bh) = (97usize, 71usize);
    let img = scene(bw, bh, 6);
    for (grid, w, h) in [
        (
            general_grid(bw, bh, 6, 3, 14.0, 1.0, (3.0, 2.0)).0,
            90usize,
            70usize,
        ),
        (general_grid(bw, bh, 2, 1, -9.0, 2.3, (0.0, 0.0)).0, 44, 38),
    ] {
        let img = img.clone();
        let out = common::exact::assert_poison_neutral("sampler", move |gpu| {
            let img = img.clone();
            let s = gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    sample(ctx, &img, (0, 0), &grid, (0, 0), w, h)
                })
                .expect("sample");
            let mut v = s.img.data.clone();
            v.extend(
                s.outside
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|&m| f32::from(m)),
            );
            v
        });
        assert!(out.is_some());
    }
}

#[test]
fn the_binding_stays_inside_the_design_size() {
    common::limits::assert_max_binding_bytes(
        "sampler",
        Rect::new(0, 0, 2000, 1250),
        max_binding_bytes,
    );
}

#[test]
fn bad_inputs_are_refused() {
    let Some(dev) = gpus().first() else { return };
    let img = scene(16, 16, 7);
    let (grid, _, _) = exact_grid(16, 16, 1, 0);
    for (w, h) in [(0usize, 4usize), (4, 0)] {
        let img = img.clone();
        let r = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                sample(ctx, &img, (0, 0), &grid, (0, 0), w, h).map(|_| ())
            });
        assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
    }
}

#[test]
fn tiny_sources_and_tiny_outputs_match_the_cpu_port() {
    // 1 x 1, 1 x N, N x 1 and 2 x 2 buffers, outputs of 1 x 1, 1 x N, N x 1 and 3 x 3 at odd offsets,
    // through every kernel (exact copy, Catmull-Rom by a rotation, the box by a shrink, a
    // magnification): no panic, the CPU's mask, the CPU's pixels within the bound.
    let Some(dev) = gpus().first() else { return };
    let bound = if dev.gpu.info().is_software {
        VALUE_BOUND
    } else {
        VALUE_BOUND_HARDWARE
    };
    let (mut cases, mut copies, mut bicubics, mut boxes) = (0usize, 0usize, 0usize, 0usize);
    for (bw, bh) in [(1usize, 1usize), (1, 5), (5, 1), (2, 2)] {
        let img = scene(bw, bh, 21);
        for (exif, turns) in [(1u16, 0u8), (6, 1), (3, 2), (5, 3), (8, 0)] {
            let (cw, ch) = if swaps(exif, turns) {
                (bh, bw)
            } else {
                (bw, bh)
            };
            let mut grids = vec![exact_grid(bw, bh, exif, turns).0];
            for (angle, scale) in [
                (0.0, 2.4),
                (12.0, 1.0),
                (-30.0, 3.2),
                (0.0, 0.4),
                (5.0, 0.6),
            ] {
                grids.push(general_grid(bw, bh, exif, turns, angle, scale, (0.25, -0.5)).0);
            }
            for grid in grids {
                for (off, w, h) in [
                    ((0usize, 0usize), 1usize, 1usize),
                    ((1, 1), 1, 1),
                    ((0, 0), 1, 6),
                    ((0, 1), 6, 1),
                    ((1, 0), 3, 3),
                    ((0, 0), cw, ch),
                ] {
                    let cpu = sample_to_camera(&img, (0, 0), &grid, off, w, h);
                    let gpu = gpu_sample(dev, &img, (0, 0), grid, off, w, h);
                    let what = format!(
                        "{bw}x{bh} exif {exif} turns {turns} {:?} off {off:?} {w}x{h}",
                        grid.kernel()
                    );
                    assert!(all_finite(&gpu.img), "{what}: non-finite");
                    assert_masks_equal(&what, &gpu, &cpu);
                    let d = max_diff(&gpu.img, &cpu.img);
                    if grid.kernel() == Kernel::Copy {
                        assert!(bit_equal(&gpu.img, &cpu.img), "{what}: copy differs");
                    } else {
                        assert!(d <= bound, "{what}: max error {d:e} above {bound:e}");
                    }
                    cases += 1;
                    match grid.kernel() {
                        Kernel::Copy => copies += 1,
                        Kernel::Bicubic => bicubics += 1,
                        Kernel::Box(_) => boxes += 1,
                    }
                }
            }
        }
    }
    eprintln!("tiny sources: {cases} cases ({copies} copy, {bicubics} Catmull-Rom, {boxes} box)");
    assert!(
        copies > 50 && bicubics > 50 && boxes > 50,
        "the matrix must cover all three kernels"
    );
}
