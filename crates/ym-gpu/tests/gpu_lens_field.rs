//! L2, the lens's field optics (`kernels::lens_field`, `shaders/lens_field.wgsl`), against its CPU
//! reference (`ym_lens::apply_field` with the reshaping off, `ym_lens::field`).
//!
//! What is compared, and with which bound:
//!
//! * **Parity.** The kernel matches the CPU reference within [`PARITY_BOUND`] over every lens, several
//!   Edges and Anamorphic amounts, both axes and odd frame sizes, in whole frames and in tiles. The CPU
//!   decides a pixel's tap count and mip level from `omega` in `f64`, the kernel in `f32`; a pixel whose
//!   `omega` lies within the `f32` rounding of a decision threshold (an integer, or half a pixel) may
//!   decide the other way, which changes its value by a fraction of the local variation of the image. Those
//!   pixels are counted, not hidden: at most [`FLIP_SHARE`] of the pixels may differ by more than
//!   [`PARITY_BOUND`].
//! * **Tiles equal the whole frame bit for bit** (D5 of `docs/17-gpu.md`), at 512 and 1024 px, and so does
//!   a source cut out with another origin.
//! * **The far corner** of an 8658 x 5772 frame is as exact as the near one (D4).
//! * **Poison**: a poisoned and a plain device give the same bits and nothing is NaN.
//! * The binding stays inside the design size (K4).

mod common;

macro_rules! note {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr(),
            format_args!("[{}] {line}\n", file!().rsplit('/').next().unwrap_or("")),
        );
    }};
}

use common::lens::{amounts, crop, noise, plan, planes_of, serial, texture};
use common::{TestGpu, exact};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::GpuError;
use ym_gpu::kernels::lens_field::{FieldTablesBuffer, field, max_binding_bytes};
use ym_gpu::queue::{JobSpec, Priority};
use ym_lens::{Axis, ImageRef, LensKind, LensPlan, PixelRect, apply_field};

/// The largest difference between the kernel and the CPU reference, over the pixels that did not
/// decide a tap count or a level the other way (see the module notes).
///
/// The first measured maximum on Mesa lavapipe (84 cases, 71.6 million samples, every lens, Edges and
/// Anamorphic amounts that put the attachment's terms alone and with the lens's, both axes) was
/// 3.04e-6, about 25 ulps of a value near 1: the sums of up to 289 products in `f32` against `f64`.
/// The bound is twice that and is never loosened. On hardware (fast-math fuses the weights and sums) the
/// bound is [`PARITY_BOUND_HARDWARE`] until the Metal run gives a number.
const PARITY_BOUND: f32 = 6.1e-6;

/// The same on a hardware adapter, until the Metal run gives a number.
const PARITY_BOUND_HARDWARE: f32 = 5e-5;

/// The share of pixels that may differ by more than the bound because of a flipped decision (none did
/// on lavapipe).
const FLIP_SHARE: f64 = 1e-6;

/// The far-corner matrix has the parity bound's value on lavapipe: a position error of order 1e-3 px at
/// 8658 px would show at once, since the scene's gradient is of order 0.5 per pixel.
const FAR_CORNER_BOUND: f32 = 6.1e-6;

/// The far corner on a hardware adapter, until the Metal run gives a number.
const FAR_CORNER_BOUND_HARDWARE: f32 = 5e-5;

fn run<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("GPU job")
}

/// The kernel over `out` of a frame, from `input` (planar, `size`) placed at `at`.
fn gpu_field(
    dev: &TestGpu,
    plan: &LensPlan,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    out: Rect,
) -> Vec<f32> {
    let f = plan.frame();
    let pass = plan.field_pass(f.width, f.height).expect("a field pass");
    let data = plan.field_tables_data().expect("tables");
    let input = input.to_vec();
    run(dev, move |ctx| {
        let tables = FieldTablesBuffer::upload(ctx, &data)?;
        field(ctx, &tables, &pass, &input, size, at, out)
    })
}

/// The CPU reference over `out`: L2 alone (the planes are those of a noise lens plane, which has no
/// out-of-focus background, so the reshaping does nothing).
fn cpu_field(
    plan: &LensPlan,
    planes: &ym_lens::LensPlanes,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    frame: (usize, usize),
    out: Rect,
) -> Vec<f32> {
    let image = ImageRef {
        width: size.0,
        height: size.1,
        x0: at.0,
        y0: at.1,
        frame_width: frame.0,
        frame_height: frame.1,
        data: input,
    };
    apply_field(
        plan,
        planes,
        1,
        &image,
        PixelRect::new(out.x, out.y, out.w, out.h),
    )
    .expect("the CPU reference")
}

/// The input a rectangle `out` needs: `out` grown by the pass's halo, cut from the whole image.
fn input_for(plan: &LensPlan, whole: &[f32], frame: (usize, usize), out: Rect) -> (Vec<f32>, Rect) {
    let halo = plan.field_pass(frame.0, frame.1).map_or(0, |p| p.halo());
    let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, frame.0, frame.1);
    let rect = Rect::new(r.x, r.y, r.w, r.h);
    (crop(whole, frame.0, frame.1, rect), rect)
}

/// What a comparison found.
struct Found {
    worst: f32,
    over: usize,
    total: usize,
}

fn compare(got: &[f32], want: &[f32], bound: f32) -> Found {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    let mut over = 0;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        assert!(d.is_finite(), "a NaN or infinity in the device's output");
        worst = worst.max(d);
        if d > bound {
            over += 1;
        }
    }
    Found {
        worst,
        over,
        total: got.len(),
    }
}

#[test]
fn the_kernel_matches_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else { return };
    let bound = if dev.gpu.info().is_software {
        PARITY_BOUND
    } else {
        PARITY_BOUND_HARDWARE
    };
    let mut cases = 0;
    let mut worst_all = 0.0f32;
    let mut over_all = 0usize;
    let mut total_all = 0usize;
    // Every lens at four amounts on a landscape frame; fewer on an odd-sized portrait frame and on the
    // y axis, which swap the roles of the field coordinates.
    type Sweep<'a> = (usize, usize, Axis, &'a [LensKind], &'a [(f64, f64)]);
    let sweeps: [Sweep; 3] = [
        (
            600,
            400,
            Axis::X,
            &LensKind::ALL,
            &[(100.0, 0.0), (60.0, 100.0), (0.0, 100.0), (30.0, 40.0)],
        ),
        (
            451,
            677,
            Axis::X,
            &[LensKind::Vintage, LensKind::Detuned],
            &[(100.0, 0.0), (60.0, 100.0)],
        ),
        (
            640,
            480,
            Axis::Y,
            &[LensKind::Petzval, LensKind::Triplet],
            &[(0.0, 100.0), (80.0, 70.0)],
        ),
    ];
    for (w, h, axis, kinds, amount_pairs) in sweeps {
        let image = noise(w, h, 7);
        for &kind in kinds {
            for &(edges, anamorphic) in amount_pairs {
                let p = plan(amounts(kind, 0.0, edges, anamorphic), w, h, axis);
                let planes = planes_of(&p);
                let out = Rect::whole(w, h);
                let want = cpu_field(&p, &planes, &image, (w, h), (0, 0), (w, h), out);
                let got = gpu_field(dev, &p, &image, (w, h), (0, 0), out);
                let f = compare(&got, &want, bound);
                worst_all = worst_all.max(f.worst);
                over_all += f.over;
                total_all += f.total;
                cases += 1;
                assert!(
                    f.over as f64 <= FLIP_SHARE * f.total as f64,
                    "{kind} {w}x{h} {axis:?} edges {edges} anamorphic {anamorphic}: {} of {} samples differ by more than {bound} (worst {})",
                    f.over,
                    f.total,
                    f.worst
                );
            }
        }
    }
    note!(
        "L2 parity: {cases} cases, {total_all} samples, worst difference {worst_all:e}, {over_all} over {bound:e} (flipped decisions)"
    );
}

/// A tile of the output, from the input cut to its halo, as the device sees it.
fn tile(dev: &TestGpu, p: &LensPlan, whole: &[f32], frame: (usize, usize), out: Rect) -> Vec<f32> {
    let (input, rect) = input_for(p, whole, frame, out);
    gpu_field(dev, p, &input, (rect.w, rect.h), (rect.x, rect.y), out)
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else { return };
    let (w, h) = (1200usize, 800usize);
    let image = noise(w, h, 3);
    for (kind, edges, anamorphic, axis) in [
        (LensKind::Vintage, 100.0, 60.0, Axis::X),
        (LensKind::Petzval, 70.0, 0.0, Axis::X),
        (LensKind::Triplet, 40.0, 100.0, Axis::Y),
    ] {
        let p = plan(amounts(kind, 0.0, edges, anamorphic), w, h, axis);
        let pass = p.field_pass(w, h).expect("a field pass");
        assert!(pass.pad > 0 || anamorphic == 0.0, "the fringe's pad is used with the attachment");
        let whole = exact::assert_tiles_equal_whole(
            &format!("L2 {kind} edges {edges} anamorphic {anamorphic} {axis:?}"),
            w,
            h,
            3,
            &[512, 1024],
            |rect| tile(dev, &p, &image, (w, h), rect),
        );
        // The whole frame differs from the input: the lens did something.
        assert!(exact::max_abs_diff(&whole, &image) > 1e-3);
    }
}

#[test]
fn a_source_cut_out_with_another_origin_gives_the_same_bits() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else { return };
    let (w, h) = (700usize, 500usize);
    let image = noise(w, h, 5);
    let p = plan(amounts(LensKind::Vintage, 0.0, 90.0, 80.0), w, h, Axis::X);
    let out = Rect::new(333, 211, 129, 97);
    let halo = p.field_pass(w, h).expect("pass").halo();
    // The whole image, with more than the halo around, and exactly the halo's rectangle.
    let from_whole = gpu_field(dev, &p, &image, (w, h), (0, 0), out);
    let (input, rect) = input_for(&p, &image, (w, h), out);
    let from_cut = gpu_field(dev, &p, &input, (rect.w, rect.h), (rect.x, rect.y), out);
    assert!(rect.x > 0 && rect.y > 0 && rect.w < w && halo > 0);
    exact::assert_bit_equal("a cut-out source", &from_cut, &from_whole);
}

#[test]
fn the_far_corner_of_a_50_mp_frame_is_as_exact_as_the_near_one() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else { return };
    let (w, h) = (8658usize, 5772usize);
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 100.0), w, h, Axis::X);
    let planes = planes_of(&p);
    let pass = p.field_pass(w, h).expect("pass");
    let halo = pass.halo();
    let bound = if dev.gpu.info().is_software {
        FAR_CORNER_BOUND
    } else {
        FAR_CORNER_BOUND_HARDWARE
    };
    let mut worsts = Vec::new();
    for (name, out) in [
        ("near corner", Rect::new(0, 0, 80, 80)),
        ("centre", Rect::new(w / 2 - 40, h / 2 - 40, 80, 80)),
        ("far corner", Rect::new(w - 80, h - 80, 80, 80)),
    ] {
        let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, w, h);
        let rect = Rect::new(r.x, r.y, r.w, r.h);
        let mut input = Vec::with_capacity(3 * rect.w * rect.h);
        for c in 0..3 {
            for y in rect.y..rect.y1() {
                for x in rect.x..rect.x1() {
                    input.push(texture(c, x, y));
                }
            }
        }
        let want = cpu_field(&p, &planes, &input, (rect.w, rect.h), (rect.x, rect.y), (w, h), out);
        let got = gpu_field(dev, &p, &input, (rect.w, rect.h), (rect.x, rect.y), out);
        let f = compare(&got, &want, bound);
        note!("L2 {name}: worst difference {:e} over {} samples", f.worst, f.total);
        assert!(
            f.over == 0,
            "{name}: {} of {} samples differ by more than {bound}",
            f.over,
            f.total
        );
        worsts.push(f.worst);
    }
    // Position errors at the far corner would show as differences orders above the near corner's.
    assert!(
        worsts[2] <= 4.0 * worsts[0].max(1e-7),
        "the far corner ({:e}) is not as exact as the near one ({:e})",
        worsts[2],
        worsts[0]
    );
}

#[test]
fn poison_and_a_plain_device_give_the_same_bits() {
    let _guard = serial();
    let Some(_dev) = common::lens::gpu() else { return };
    let (w, h) = (300usize, 220usize);
    let image = noise(w, h, 11);
    for (edges, anamorphic, axis) in [(100.0, 100.0, Axis::X), (60.0, 30.0, Axis::Y)] {
        let p = plan(amounts(LensKind::Detuned, 0.0, edges, anamorphic), w, h, axis);
        let pass = p.field_pass(w, h).expect("pass");
        let data = p.field_tables_data().expect("tables");
        let out = Rect::new(20, 10, 250, 190);
        let image = image.clone();
        let got = common::lens::assert_poison_neutral("L2", move |gpu| {
            let (pass, data, image) = (pass, data.clone(), image.clone());
            gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let tables = FieldTablesBuffer::upload(ctx, &data)?;
                field(ctx, &tables, &pass, &image, (w, h), (0, 0), out)
            })
            .expect("field")
        });
        assert!(got.is_some());
    }
}

#[test]
fn the_binding_stays_inside_the_design_size() {
    let _guard = serial();
    let p = plan(amounts(LensKind::Petzval, 0.0, 100.0, 100.0), 6000, 4000, Axis::X);
    let pass = p.field_pass(6000, 4000).expect("pass");
    common::limits::assert_max_binding_bytes("lens field", Rect::new(0, 0, 1024, 1024), |r| {
        max_binding_bytes(r, &pass)
    });
}

#[test]
fn a_source_that_does_not_cover_the_halo_is_refused() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else { return };
    let (w, h) = (400usize, 300usize);
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 0.0), w, h, Axis::X);
    let pass = p.field_pass(w, h).expect("pass");
    let data = p.field_tables_data().expect("tables");
    let image = noise(w, h, 2);
    // The output is the middle of the frame and the source is exactly the output: no halo.
    let out = Rect::new(100, 100, 100, 100);
    let small = crop(&image, w, h, out);
    let result = dev.gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let tables = FieldTablesBuffer::upload(ctx, &data)?;
        field(ctx, &tables, &pass, &small, (100, 100), (100, 100), out)
    });
    assert!(
        matches!(result, Err(GpuError::InvalidInput(_))),
        "{result:?}"
    );
}
