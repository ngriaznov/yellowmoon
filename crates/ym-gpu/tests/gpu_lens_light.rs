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
//! * **The extra light of the clipped lights** (`encode_headroom`, modes 1 and 2) and **L2**
//!   with an extra light (`lens_field`'s `extra`) match their CPU references (`ym_lens::l2_input`, `apply_field`).

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

// ---------------------------------------------------------------------------------------------------------
// The extra light of the small clipped lights (`light::l2_input`, mode 1 and 2 of the kernel) and L2 with an
// extra light (`field_with_extra`)

use ym_gpu::cpu::tiling::Rect;
use ym_gpu::kernels::lens_field::{FieldTablesBuffer, field_with_extra};
use ym_gpu::kernels::lens_light::headroom;
use ym_lens::{LensPlanes, PixelRect, apply_field, l2_input};

/// The largest difference between the headroom kernel and its CPU reference, relative to the largest value of
/// the image. A few `f32` operations and an `exp2` on values up to 100: the first measurement on Mesa lavapipe
/// is in the test's output; the bound is a few times it and is never loosened.
const HEADROOM_BOUND: f32 = 2e-6;

/// The same for the pass after L1 (mode 2). Where L1 changed a pixel by about a thousandth of a clip level the
/// share `m` of the pixels L1 left as they are is a smoothstep over `L1_UNCHANGED_RAMP` (1e-4 to 1e-3), whose
/// slope is 1700 per unit: an `f32` rounding of the change (1e-7 of values up to 5) moves `m` by up to 5e-4. The
/// first measurement on Mesa lavapipe was 2.2e-4 of the largest value.
const HEADROOM_AFTER_L1_BOUND: f32 = 5e-4;

/// The frame of the headroom tests: a plan with the light's clip levels, shipped Vintage with Edges 70.
const HW: usize = 300;
const HH: usize = 200;

fn headroom_plan(edges: f64, anamorphic: f64) -> LensPlan {
    let frame = FrameInfo {
        width: HW,
        height: HH,
        axis: Axis::X,
        f_number: None,
        clip: G0,
    };
    LensPlan::new(
        common::lens::library(),
        &amounts(LensKind::Vintage, 0.0, edges, anamorphic),
        frame,
        common::lens::camera(),
    )
    .expect("plan")
}

/// A dim textured ground with stars: a clipped one (the neutral recovered core), a green-clipped white one, the
/// cases of the saturation rule, and a window too large to be a point.
fn stars() -> Vec<f32> {
    let n = HW * HH;
    let mut data = vec![0.0f32; 3 * n];
    for y in 0..HH {
        for x in 0..HW {
            for c in 0..3 {
                data[c * n + y * HW + x] = 0.02 + 0.03 * common::lens::texture(c, x, y);
            }
        }
    }
    let mut paint = |cx: f64, cy: f64, r: f64, v: [f32; 3]| {
        for y in 0..HH {
            for x in 0..HW {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                if dx * dx + dy * dy <= r * r {
                    for c in 0..3 {
                        data[c * n + y * HW + x] = v[c];
                    }
                }
            }
        }
    };
    paint(60.5, 50.5, 1.6, [3.1, 3.0, 3.0]);
    paint(200.5, 60.5, 1.4, [1.4, 1.1, 1.7]);
    paint(240.5, 150.5, 1.5, [2.4, 1.0, 1.9]);
    paint(100.5, 140.5, 30.0, [3.0, 3.0, 3.0]);
    // A pixel the clip test ramps over, and a coloured one.
    paint(30.5, 170.5, 1.2, [2.62, 0.97, 1.9]);
    paint(160.5, 30.5, 1.2, [2.2, 1.2, 0.2]);
    data
}

fn image(w: usize, h: usize, data: &[f32]) -> ImageRef<'_> {
    ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: HW,
        frame_height: HH,
        data,
    }
}

#[test]
fn the_headroom_kernel_matches_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = gpu() else {
        return;
    };
    let bound = if dev.gpu.info().is_software {
        HEADROOM_BOUND
    } else {
        10.0 * HEADROOM_BOUND
    };
    let plan = headroom_plan(70.0, 0.0);
    let pass = plan.light_pass().expect("Edges act");
    let data = stars();
    let raw = image(HW, HH, &data);
    let planes = LensPlanes::build(&plan, &raw, None, &|| false).expect("planes");
    let map = planes.light_map_data();
    assert!(map.ev.iter().any(|v| *v > 0.0), "the stars are lights");
    // Mode 1: the raw picture, at the scale of the lens plane.
    let want = l2_input(&plan, &planes, plan.lens_plane_scale(), &raw, None).expect("a headroom input");
    assert!(want.own.is_none());
    let (got, none) = {
        let data = data.clone();
        let map = map.clone();
        let k = plan.lens_plane_scale();
        run(dev, move |ctx| {
            headroom(ctx, &data, (HW, HH), (0, 0), None, pass, &map, k).expect("the headroom pass")
        })
    };
    assert!(none.is_empty());
    let largest = want.extra.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want.extra)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "extra light, mode 1: worst difference {worst:e} of values up to {largest:e} ({:e} relative)",
        worst / largest
    );
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(largest > 5.0, "the clipped stars hold headroom: {largest}");
    assert!(got.iter().all(|v| *v >= 0.0));
    assert!(worst <= bound * largest, "worst {worst} of {largest}");
    // Mode 2: after L1, which changed a block of pixels (a blurred patch) and left the others as they are.
    let lit = lit_input(&plan, &raw).expect("saturated pixels");
    let (bx, by, bw, bh) = (30usize, 20usize, 230usize, 140usize);
    let mut l1 = Vec::with_capacity(3 * bw * bh);
    for c in 0..3 {
        for y in by..by + bh {
            for x in bx..bx + bw {
                let v = lit[c * HW * HH + y * HW + x];
                // L1 changed the first star's pixels by 0.4 % (a reshaped pixel), the second's by 0.025 % (the
                // blend between changed and left alone) and left the third's and the rest as they are.
                let changed = (50..70).contains(&x) && (40..60).contains(&y);
                let blend = (190..210).contains(&x) && (50..70).contains(&y);
                l1.push(if changed {
                    v * 1.004
                } else if blend {
                    v * 1.000_25
                } else {
                    v
                });
            }
        }
    }
    let l1_image = ImageRef {
        width: bw,
        height: bh,
        x0: bx,
        y0: by,
        frame_width: HW,
        frame_height: HH,
        data: &l1,
    };
    let want = l2_input(&plan, &planes, plan.lens_plane_scale(), &raw, Some(&l1_image))
        .expect("a headroom input after L1");
    let want_own = want.own.as_ref().expect("an own light after L1");
    let (got, got_own) = {
        let (data, l1, map) = (data.clone(), l1.clone(), map.clone());
        let k = plan.lens_plane_scale();
        run(dev, move |ctx| {
            headroom(
                ctx,
                &data,
                (HW, HH),
                (0, 0),
                Some((&l1, (bw, bh), (bx, by))),
                pass,
                &map,
                k,
            )
            .expect("the headroom pass after L1")
        })
    };
    let largest = want.extra.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    let (wy, wo) = (diff(&got, &want.extra), diff(&got_own, want_own));
    println!(
        "extra light, mode 2: worst difference {wy:e} (extra) and {wo:e} (picture) of values up to {largest:e}"
    );
    let bound = if dev.gpu.info().is_software {
        HEADROOM_AFTER_L1_BOUND
    } else {
        10.0 * HEADROOM_AFTER_L1_BOUND
    };
    assert!(wy <= bound * largest && wo <= bound * largest);
}

#[test]
fn l2_with_an_extra_light_matches_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = gpu() else {
        return;
    };
    let plan = headroom_plan(70.0, 0.0);
    let pass = plan.light_pass().expect("Edges act");
    let data = stars();
    let raw = image(HW, HH, &data);
    let planes = LensPlanes::build(&plan, &raw, None, &|| false).expect("planes");
    let k = plan.lens_plane_scale();
    let want = apply_field(&plan, &planes, k, &raw, PixelRect::new(0, 0, HW, HH)).expect("the CPU lens");
    let map = planes.light_map_data();
    let field_pass = plan.field_pass(HW, HH).expect("a field pass");
    let tables_data = plan.field_tables_data().expect("tables");
    let whole = {
        let (data, map) = (data.clone(), map.clone());
        run(dev, move |ctx| {
            let (extra, _) = headroom(ctx, &data, (HW, HH), (0, 0), None, pass, &map, k).expect("headroom");
            let tables = FieldTablesBuffer::upload(ctx, &tables_data).expect("tables");
            field_with_extra(
                ctx,
                &tables,
                &field_pass,
                &data,
                (HW, HH),
                (0, 0),
                (&extra, (HW, HH), (0, 0)),
                Rect::new(0, 0, HW, HH),
            )
            .expect("L2 with an extra light")
        })
    };
    let largest = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (mut worst, mut at) = (0.0f32, 0usize);
    for (i, (a, b)) in whole.iter().zip(&want).enumerate() {
        assert!(a.is_finite() && *a >= 0.0, "sample {i} is {a}");
        let d = (a - b).abs();
        if d > worst {
            (worst, at) = (d, i);
        }
    }
    println!(
        "L2 with an extra light: worst difference {worst:e} (sample {at}) of values up to {largest:e} ({:e} relative)",
        worst / largest
    );
    assert!(largest > 3.0, "the stars' headroom reaches the output: {largest}");
    assert!(worst <= 2e-4 * largest, "worst difference {worst} of values up to {largest}");
}

#[test]
fn a_tile_of_the_headroom_and_l2_equals_the_frame_bit_for_bit_and_poison_changes_nothing() {
    let _guard = serial();
    let Some(dev) = gpu() else {
        return;
    };
    let plan = headroom_plan(70.0, 0.0);
    let pass = plan.light_pass().expect("Edges act");
    let data = stars();
    let raw = image(HW, HH, &data);
    let planes = LensPlanes::build(&plan, &raw, None, &|| false).expect("planes");
    let k = plan.lens_plane_scale();
    let map = planes.light_map_data();
    let field_pass = plan.field_pass(HW, HH).expect("a field pass");
    let tables_data = plan.field_tables_data().expect("tables");
    let halo = field_pass.halo();
    // The chain over `out` of the frame: the window of the raw picture it reads, the headroom over it, L2.
    let chain = |gpu: &ym_gpu::device::Gpu, out: Rect| -> Vec<f32> {
        let x0 = out.x.saturating_sub(halo);
        let y0 = out.y.saturating_sub(halo);
        let x1 = (out.x1() + halo).min(HW);
        let y1 = (out.y1() + halo).min(HH);
        let (ww, wh) = (x1 - x0, y1 - y0);
        let mut window = Vec::with_capacity(3 * ww * wh);
        for c in 0..3 {
            for y in y0..y1 {
                window.extend_from_slice(&data[c * HW * HH + y * HW + x0..c * HW * HH + y * HW + x1]);
            }
        }
        let (map, field_pass, tables_data) = (map.clone(), field_pass.clone(), tables_data.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let (extra, _) = headroom(ctx, &window, (ww, wh), (x0, y0), None, pass, &map, k).expect("headroom");
            let tables = FieldTablesBuffer::upload(ctx, &tables_data).expect("tables");
            field_with_extra(
                ctx,
                &tables,
                &field_pass,
                &window,
                (ww, wh),
                (x0, y0),
                (&extra, (ww, wh), (x0, y0)),
                out,
            )
        })
        .expect("GPU job")
    };
    let whole = chain(&dev.gpu, Rect::new(0, 0, HW, HH));
    for out in [
        Rect::new(40, 20, 97, 61),
        Rect::new(180, 110, 100, 80),
        Rect::new(0, 0, 64, 64),
    ] {
        let tile = chain(&dev.gpu, out);
        for c in 0..3 {
            for y in 0..out.h {
                for x in 0..out.w {
                    let a = tile[c * out.w * out.h + y * out.w + x];
                    let b = whole[c * HW * HH + (out.y + y) * HW + out.x + x];
                    assert_eq!(a.to_bits(), b.to_bits(), "tile {out:?}: channel {c} at ({x}, {y})");
                }
            }
        }
    }
    // A plain device gives the poisoned device's bits.
    if let Some(plain) = plain_gpu() {
        let again = chain(plain, Rect::new(0, 0, HW, HH));
        assert!(whole.iter().all(|v| v.is_finite()));
        assert_eq!(
            whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
