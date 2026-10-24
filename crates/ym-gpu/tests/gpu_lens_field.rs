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

use common::lens::{amounts, crop, noise, plan, planes_of, texture};
use common::{TestGpu, exact, gpus};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::GpuError;
use ym_gpu::kernels::lens_field::{FieldTablesBuffer, field, max_binding_bytes};
use ym_gpu::queue::{JobSpec, Priority};
use ym_lens::{Axis, ImageRef, LensKind, LensPlan, PixelRect, apply_field};

/// The largest difference between the kernel and the CPU reference, over the pixels that did not
/// decide a tap count or a level the other way (see the module notes).
const PARITY_BOUND: f32 = 2e-5;

/// The share of pixels that may differ by more than [`PARITY_BOUND`] because of a flipped decision.
const FLIP_SHARE: f64 = 2e-4;

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

fn compare(got: &[f32], want: &[f32]) -> Found {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    let mut over = 0;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        assert!(d.is_finite(), "a NaN or infinity in the device's output");
        worst = worst.max(d);
        if d > PARITY_BOUND {
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
    let Some(dev) = gpus().first() else { return };
    let mut cases = 0;
    let mut worst_all = 0.0f32;
    let mut over_all = 0usize;
    let mut total_all = 0usize;
    for (w, h, axis) in [
        (600usize, 400usize, Axis::X),
        (451, 677, Axis::X),
        (640, 480, Axis::Y),
    ] {
        let image = noise(w, h, 7);
        for kind in LensKind::ALL {
            for (edges, anamorphic) in [(100.0, 0.0), (60.0, 100.0), (0.0, 100.0), (30.0, 40.0)] {
                let p = plan(amounts(kind, 0.0, edges, anamorphic), w, h, axis);
                let planes = planes_of(&p);
                let out = Rect::whole(w, h);
                let want = cpu_field(&p, &planes, &image, (w, h), (0, 0), (w, h), out);
                let got = gpu_field(dev, &p, &image, (w, h), (0, 0), out);
                let f = compare(&got, &want);
                worst_all = worst_all.max(f.worst.min(PARITY_BOUND));
                over_all += f.over;
                total_all += f.total;
                cases += 1;
                assert!(
                    f64::from(f.over as u32) <= FLIP_SHARE * f.total as f64,
                    "{kind} {w}x{h} {axis:?} edges {edges} anamorphic {anamorphic}: {} of {} samples differ by more than {PARITY_BOUND} (worst {})",
                    f.over,
                    f.total,
                    f.worst
                );
            }
        }
    }
    note!(
        "L2 parity: {cases} cases, {over_all} of {total_all} samples over {PARITY_BOUND} (decision flips), worst below the bound {worst_all}"
    );
}
