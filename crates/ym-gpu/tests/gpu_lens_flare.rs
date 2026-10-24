//! L3, the lens's highlight optics (`kernels::lens_flare`, `shaders/lens_flare.wgsl`), against its CPU
//! reference (`ym_lens::apply_flares`, `ym_lens::l3`).
//!
//! What is compared, and with which bound:
//!
//! * **Parity.** The kernel matches the CPU reference within [`PARITY_BOUND`] over every lens, Flares and
//!   Anamorphic amounts that switch the streaks and the attachment's ghosts on and off, both axes, on a
//!   scene of lights of every size and colour. The CPU works in `f64` and the kernel in `f32`.
//! * **Tiles equal the whole frame bit for bit** (D5 of `docs/17-gpu.md`), at 512 and 1024 px.
//! * **The far corner** of an 8658 x 5772 frame is as exact as the near one (D4).
//! * **Poison**: a poisoned and a plain device give the same bits and nothing is NaN.
//! * The binding stays inside the design size (K4); Flares 0 has no scale data at all.

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

use common::lens::{
    Light, amounts, lights, lit, lit_image, lit_planes, plan_clipped, serial, texture,
};
use common::{TestGpu, exact};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::GpuError;
use ym_gpu::kernels::lens_flare::{flare, max_binding_bytes};
use ym_gpu::queue::{JobSpec, Priority};
use ym_lens::{
    Axis, FrameInfo, ImageRef, LensKind, LensPlan, LensPlanes, PixelRect, apply_flares,
    flares_halo_px,
};

/// The largest difference between the kernel and the CPU reference, relative to the largest value of the
/// flare terms alone (the output with `Z` zero). The first measured maximum on Mesa lavapipe (19 cases:
/// every lens at Flares 100 with Anamorphic 0 and 100, both axes, lights of every size and colour) was
/// 9.3e-6, a few dozen ulps of `f32` in the sums of up to 289 core taps and the items' profiles against
/// the CPU's `f64`. The bound is twice that and is never loosened.
const PARITY_BOUND: f32 = 2e-5;

/// The same on a hardware adapter (fast-math fuses the products and sums), until the Metal run gives a
/// number.
const PARITY_BOUND_HARDWARE: f32 = 1e-4;

/// The same on the 50 MP frame of the far-corner test. The streak is bright since `docs/21`'s flare strengths
/// (a light's energy times 2^12 for the small lights), and 2 to 3 standard deviations across from a light its
/// value is read in the Gaussian's tail, where the light's centre, stored in `f32` (6e-8 half-diagonals, 3e-4 px
/// of 50 MP), moves the value by up to 1e-4 of itself. Measured 1.17e-4 of the region's peak in the corner
/// beside a light (the 500 x 350 cases above stay under 7e-6); the bound is twice that and never loosened.
const PARITY_BOUND_50MP: f32 = 2.5e-4;

/// The same on a hardware adapter.
const PARITY_BOUND_50MP_HARDWARE: f32 = 5e-4;

fn run<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("GPU job")
}

/// The kernel over `out`, from `input` (planar, `size`) placed at `at` and `z` (exactly `out`).
#[allow(clippy::too_many_arguments)]
fn gpu_flare(
    dev: &TestGpu,
    plan: &LensPlan,
    planes: &LensPlanes,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    z: &[f32],
    out: Rect,
) -> Vec<f32> {
    let f = plan.frame();
    let plane_data = planes.flare_plane_data().expect("planes");
    let lens = plan.flare_lens_data();
    let scale = plan
        .flare_scale_data(planes, f.width, f.height)
        .expect("scale data");
    let mix = plan.flare_mix();
    let (input, z) = (input.to_vec(), z.to_vec());
    run(dev, move |ctx| {
        flare(
            ctx,
            &plane_data,
            &lens,
            &scale,
            mix,
            &input,
            size,
            at,
            &z,
            out,
        )
    })
}

fn cpu_flare(
    plan: &LensPlan,
    planes: &LensPlanes,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    z: &[f32],
    out: Rect,
) -> Vec<f32> {
    let f = plan.frame();
    let image = ImageRef {
        width: size.0,
        height: size.1,
        x0: at.0,
        y0: at.1,
        frame_width: f.width,
        frame_height: f.height,
        data: input,
    };
    apply_flares(
        plan,
        planes,
        1,
        &image,
        z,
        PixelRect::new(out.x, out.y, out.w, out.h),
    )
    .expect("the CPU reference")
}

/// A picture to stand for `Z` over `out`: a texture of absolute coordinates.
fn z_over(out: Rect) -> Vec<f32> {
    let mut z = Vec::with_capacity(3 * out.w * out.h);
    for c in 0..3 {
        for y in out.y..out.y1() {
            for x in out.x..out.x1() {
                z.push(0.5 * texture(c, x + 3, y + 7));
            }
        }
    }
    z
}

/// `out` grown by the core's reach, clamped to the frame, cut from a whole image.
fn region(plan: &LensPlan, planes: &LensPlanes, out: Rect) -> Rect {
    let f = plan.frame();
    let halo = flares_halo_px(plan, planes, 1);
    let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, f.width, f.height);
    Rect::new(r.x, r.y, r.w, r.h)
}

fn crop(whole: &[f32], w: usize, h: usize, r: Rect) -> Vec<f32> {
    common::lens::crop(whole, w, h, r)
}

struct Found {
    worst: f32,
    scale: f32,
}

fn compare(got: &[f32], want: &[f32]) -> Found {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "a NaN or infinity in the device's output");
        worst = worst.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    Found { worst, scale }
}

#[test]
fn the_kernel_matches_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let bound = if dev.gpu.info().is_software {
        PARITY_BOUND
    } else {
        PARITY_BOUND_HARDWARE
    };
    let (w, h) = (500usize, 350usize);
    let lights = lights();
    let image = lit_image(w, h, &lights);
    let out = Rect::whole(w, h);
    // Z is zero, so the output is the flare terms alone and the comparison is of them (a picture under
    // them would hide their errors behind its own values); one case below mixes with a picture.
    let nothing = vec![0.0f32; 3 * w * h];
    let mut cases = 0;
    let mut times = Vec::new();
    let mut worst_all = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut largest = 0.0f32;
    let mut with_grids = 0;
    for axis in [Axis::X, Axis::Y] {
        for kind in LensKind::ALL {
            for (flares, anamorphic) in [(100.0, 0.0), (100.0, 100.0)] {
                if axis == Axis::Y && kind != LensKind::Vintage && kind != LensKind::Detuned {
                    continue;
                }
                let p = plan_clipped(amounts(kind, flares, 0.0, anamorphic), w, h, axis);
                let planes = lit_planes(&p, &lights);
                let scale = p.flare_scale_data(&planes, w, h).expect("scale data");
                with_grids += usize::from(scale.coarse.iter().any(|c| *c));
                let t = std::time::Instant::now();
                let want = cpu_flare(&p, &planes, &image, (w, h), (0, 0), &nothing, out);
                let cpu_s = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let got = gpu_flare(dev, &p, &planes, &image, (w, h), (0, 0), &nothing, out);
                let gpu_s = t.elapsed().as_secs_f64();
                times.push((cpu_s, gpu_s));
                let f = compare(&got, &want);
                worst_all = worst_all.max(f.worst);
                worst_rel = worst_rel.max(f.worst / f.scale.max(1e-9));
                largest = largest.max(f.scale);
                cases += 1;
                assert!(
                    f.worst <= bound * f.scale.max(1e-9) || f.worst <= 1e-9,
                    "{kind} {axis:?} flares {flares} anamorphic {anamorphic}: worst difference {} (values up to {})",
                    f.worst,
                    f.scale
                );
            }
        }
    }
    // The grids of the slowly varying ghosts are part of what is compared.
    assert!(with_grids > 0, "no case draws a ghost on a grid of nodes");
    // Mixed with a picture at Flares 40: (1 - a) Z + F terms.
    let z = z_over(out);
    let p = plan_clipped(amounts(LensKind::Classic, 40.0, 0.0, 60.0), w, h, Axis::X);
    let planes = lit_planes(&p, &lights);
    let want = cpu_flare(&p, &planes, &image, (w, h), (0, 0), &z, out);
    let got = gpu_flare(dev, &p, &planes, &image, (w, h), (0, 0), &z, out);
    let f = compare(&got, &want);
    cases += 1;
    assert!(
        f.worst <= bound * f.scale.max(1e-9),
        "mixed with a picture: {} of values up to {}",
        f.worst,
        f.scale
    );
    let (cpu_total, gpu_total) = times
        .iter()
        .fold((0.0, 0.0), |(a, b), (c, g)| (a + c, b + g));
    note!(
        "L3 timing over {} cases of {w}x{h}: CPU reference {cpu_total:.1} s, device (lavapipe, with uploads) {gpu_total:.1} s",
        times.len()
    );
    note!(
        "L3 parity: {cases} cases ({with_grids} with ghosts on grids of nodes); flare terms alone: worst difference {worst_all:e} of values up to {largest:e} ({worst_rel:e} relative); mixed with a picture: {:e}",
        f.worst
    );
}

/// The ghosts on grids of nodes, alone: a lens of big soft ghosts with no glow, veil, starburst or streak, so
/// that the flare terms are the ghosts' (the shipped lenses' glow is a thousand times brighter than their
/// ghosts, which hides an error in them from the other test). The bound is the same share of the terms' peak.
#[test]
fn the_ghosts_on_the_grids_match_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let bound = if dev.gpu.info().is_software {
        PARITY_BOUND
    } else {
        PARITY_BOUND_HARDWARE
    };
    let lib = ym_lens::test_support::ghost_library(0.02).expect("the synthetic ghost lenses");
    let (w, h) = (600usize, 400usize);
    let lights = lights();
    let image = lit_image(w, h, &lights);
    let out = Rect::whole(w, h);
    let nothing = vec![0.0f32; 3 * w * h];
    let (mut worst_rel, mut largest) = (0.0f32, 0.0f32);
    for axis in [Axis::X, Axis::Y] {
        for anamorphic in [0.0, 100.0] {
            let frame = FrameInfo {
                width: w,
                height: h,
                axis,
                f_number: Some(8.0),
                clip: [1.0; 3],
            };
            let p = LensPlan::new(
                &lib,
                &amounts(LensKind::Classic, 100.0, 0.0, anamorphic),
                frame,
                common::lens::camera(),
            )
            .expect("plan");
            let planes = lit_planes(&p, &lights);
            let scale = p.flare_scale_data(&planes, w, h).expect("scale data");
            let levels = scale.coarse.iter().filter(|c| **c).count();
            assert!(
                levels >= 2,
                "{axis:?} {anamorphic}: grids in use: {:?}",
                scale.coarse
            );
            let want = cpu_flare(&p, &planes, &image, (w, h), (0, 0), &nothing, out);
            let got = gpu_flare(dev, &p, &planes, &image, (w, h), (0, 0), &nothing, out);
            let f = compare(&got, &want);
            worst_rel = worst_rel.max(f.worst / f.scale.max(1e-9));
            largest = largest.max(f.scale);
            assert!(
                f.worst <= bound * f.scale.max(1e-9),
                "{axis:?} anamorphic {anamorphic}: worst difference {} (values up to {})",
                f.worst,
                f.scale
            );
        }
    }
    note!(
        "L3 ghosts on grids: 4 cases, worst difference {worst_rel:e} of the peak (peaks up to {largest:e})"
    );
}

/// A tile of the output, from the input cut to the core's reach, as the device sees it.
fn tile(
    dev: &TestGpu,
    p: &LensPlan,
    planes: &LensPlanes,
    whole: &[f32],
    frame: (usize, usize),
    out: Rect,
) -> Vec<f32> {
    let r = region(p, planes, out);
    let input = crop(whole, frame.0, frame.1, r);
    gpu_flare(
        dev,
        p,
        planes,
        &input,
        (r.w, r.h),
        (r.x, r.y),
        &z_over(out),
        out,
    )
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let (w, h) = (1200usize, 800usize);
    let lights = lights();
    let image = lit_image(w, h, &lights);
    for (kind, flares, anamorphic, axis) in [
        (LensKind::Vintage, 100.0, 100.0, Axis::X),
        (LensKind::Detuned, 60.0, 0.0, Axis::X),
        (LensKind::Classic, 100.0, 70.0, Axis::Y),
    ] {
        let p = plan_clipped(amounts(kind, flares, 0.0, anamorphic), w, h, axis);
        let planes = lit_planes(&p, &lights);
        let whole = exact::assert_tiles_equal_whole(
            &format!("L3 {kind} flares {flares} anamorphic {anamorphic} {axis:?}"),
            w,
            h,
            3,
            &[512, 1024],
            |rect| tile(dev, &p, &planes, &image, (w, h), rect),
        );
        // The flares did something.
        let z = z_over(Rect::whole(w, h));
        assert!(exact::max_abs_diff(&whole, &z) > 1e-4);
    }
}

#[test]
fn the_far_corner_of_a_50_mp_frame_is_as_exact_as_the_near_one() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let (w, h) = (8658usize, 5772usize);
    // Lights near both corners and the centre, so every region has items to draw.
    let lights: Vec<Light> = vec![
        (0.012, 0.02, 0.003, [1.0, 1.0, 1.0]),
        (0.5, 0.5, 0.004, [1.0, 0.8, 0.5]),
        (0.986, 0.978, 0.003, [1.0, 1.0, 1.0]),
        (0.6, 0.3, 0.002, [1.0, 1.0, 1.0]),
    ];
    let p = plan_clipped(amounts(LensKind::Vintage, 100.0, 0.0, 100.0), w, h, Axis::X);
    let planes = lit_planes(&p, &lights);
    let nothing = |out: Rect| vec![0.0f32; 3 * out.w * out.h];
    let mut results = Vec::new();
    for (name, out) in [
        ("near corner", Rect::new(0, 0, 90, 90)),
        ("centre", Rect::new(w / 2 - 45, h / 2 - 45, 90, 90)),
        ("far corner", Rect::new(w - 90, h - 90, 90, 90)),
    ] {
        let r = region(&p, &planes, out);
        let mut input = Vec::with_capacity(3 * r.w * r.h);
        for c in 0..3 {
            for y in r.y..r.y1() {
                for x in r.x..r.x1() {
                    input.push(lit(c, x, y, w, h, &lights));
                }
            }
        }
        let z = nothing(out);
        let want = cpu_flare(&p, &planes, &input, (r.w, r.h), (r.x, r.y), &z, out);
        let got = gpu_flare(dev, &p, &planes, &input, (r.w, r.h), (r.x, r.y), &z, out);
        let f = compare(&got, &want);
        note!(
            "L3 {name}: worst difference {:e} of values up to {:e}",
            f.worst,
            f.scale
        );
        assert!(f.scale > 0.0, "{name}: the flares drew nothing here");
        results.push(f.worst / f.scale);
    }
    let bound = if dev.gpu.info().is_software {
        PARITY_BOUND_50MP
    } else {
        PARITY_BOUND_50MP_HARDWARE
    };
    for (name, rel) in ["near corner", "centre", "far corner"].iter().zip(&results) {
        assert!(*rel <= bound, "{name}: relative difference {rel:e}");
    }
    assert!(
        results[2] <= 4.0 * results[0].max(1e-6),
        "the far corner ({:e}) is not as exact as the near one ({:e})",
        results[2],
        results[0]
    );
}

#[test]
fn poison_and_a_plain_device_give_the_same_bits() {
    let _guard = serial();
    let Some(_dev) = common::lens::gpu() else {
        return;
    };
    let (w, h) = (300usize, 220usize);
    let lights = lights();
    let image = lit_image(w, h, &lights);
    for (anamorphic, axis) in [(100.0, Axis::X), (30.0, Axis::Y)] {
        let p = plan_clipped(
            amounts(LensKind::Detuned, 100.0, 0.0, anamorphic),
            w,
            h,
            axis,
        );
        let planes = lit_planes(&p, &lights);
        let plane_data = planes.flare_plane_data().expect("planes");
        let lens = p.flare_lens_data();
        let scale = p.flare_scale_data(&planes, w, h).expect("scale");
        let mix = p.flare_mix();
        let out = Rect::new(20, 10, 250, 190);
        let z = z_over(out);
        let image = image.clone();
        let got = common::lens::assert_poison_neutral("L3", move |gpu| {
            let (plane_data, lens, scale, z, image) = (
                plane_data.clone(),
                lens.clone(),
                scale.clone(),
                z.clone(),
                image.clone(),
            );
            gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                flare(
                    ctx,
                    &plane_data,
                    &lens,
                    &scale,
                    mix,
                    &image,
                    (w, h),
                    (0, 0),
                    &z,
                    out,
                )
            })
            .expect("flare")
        });
        assert!(got.is_some());
    }
}

#[test]
fn the_binding_stays_inside_the_design_size() {
    let _guard = serial();
    common::limits::assert_max_binding_bytes("lens flare", Rect::new(0, 0, 1024, 1024), |r| {
        max_binding_bytes(r, 8)
    });
}

#[test]
fn flares_at_zero_have_no_scale_data() {
    let _guard = serial();
    let (w, h) = (300usize, 200usize);
    let lights = lights();
    let on = plan_clipped(amounts(LensKind::Vintage, 50.0, 0.0, 0.0), w, h, Axis::X);
    let planes = lit_planes(&on, &lights);
    assert!(on.flare_scale_data(&planes, w, h).is_some());
    let off = plan_clipped(amounts(LensKind::Vintage, 0.0, 80.0, 40.0), w, h, Axis::X);
    assert!(off.flare_scale_data(&planes, w, h).is_none());
    // A Flares move keeps the data (and its key): the scale data does not depend on the amount.
    let more = plan_clipped(amounts(LensKind::Vintage, 90.0, 0.0, 0.0), w, h, Axis::X);
    assert_eq!(on.flare_shape_key(), more.flare_shape_key());
    assert_eq!(
        on.flare_scale_data(&planes, w, h),
        more.flare_scale_data(&planes, w, h)
    );
    // Anamorphic does.
    let anamorphic = plan_clipped(amounts(LensKind::Vintage, 50.0, 0.0, 30.0), w, h, Axis::X);
    assert_ne!(on.flare_shape_key(), anamorphic.flare_shape_key());
}
