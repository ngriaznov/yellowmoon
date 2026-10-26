//! L1, the lens's bokeh reshaping (`kernels::lens_reshape`, `shaders/lens_reshape.wgsl`), against its CPU
//! reference (`ym_lens::reshape`, `ym_lens::l1`).
//!
//! What is compared, and with which bound:
//!
//! * **Parity.** The kernel matches the CPU reference within [`PARITY_BOUND`] over every lens that has a swirl or a
//!   cat's-eye, several Edges and Anamorphic amounts (the swirl and the cat's-eye are Anamorphic's; Edges is in the
//!   pairs to show it changes nothing here), both axes, on scenes with out-of-focus balls over a
//!   textured background, with a sharp subject and its strands, and on a night scene. The CPU reference decides a
//!   pixel's blur, the length of its line and its tap count in `f64`, the kernel in `f32`; a pixel whose blur or
//!   length lies within the `f32` rounding of a decision threshold (an integer tap count, the acceptance of the
//!   map, a flag of the sharp-edge test) may decide the other way, which changes its value by a fraction of the
//!   local variation of the picture. Those pixels are counted, not hidden: at most [`FLIP_SHARE`] of the samples
//!   may differ by more than [`PARITY_BOUND`], and none by more than [`FLIP_BOUND`].
//! * **Tiles equal the whole frame bit for bit** (D5 of `docs/17-gpu.md`), at 512 and 1024 px.
//! * **The far corner** of an 8658 x 5772 frame is as exact as the near one (D4).
//! * **Poison**: a poisoned and a plain device give the same bits and nothing is NaN.
//! * **Identity**: Anamorphic at 0 gives no pass at all, whatever Edges is.
//! * The binding stays inside the design size (K4).

mod common;
#[path = "../../ym-lens/tests/scenes/mod.rs"]
mod scenes;

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

use std::sync::OnceLock;

use common::lens::{amounts, crop, plan, serial};
use common::{TestGpu, exact};
use scenes::{Noise, Scene, balls_over_texture_scene, soft_night_scene, subject_scene};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::GpuError;
use ym_gpu::kernels::lens_reshape::reshape;
use ym_gpu::queue::{JobSpec, Priority};
use ym_lens::{Axis, BlurMapInput, ImageRef, LensKind, LensPlan, LensPlanes, PixelRect};

/// The largest difference between the kernel and the CPU reference over the pixels that did not decide a blur,
/// a length or a flag the other way (see the module notes).
///
/// The first measured maximum on Mesa lavapipe (45 cases, 90 million samples, every lens, Edges and
/// Anamorphic amounts that put each term of the stage alone and together, both axes, three scenes) was 6.4e-6 for
/// all but a handful of pixels, which decided a tap count the other way. The bound is 2e-5 (white = 1, balls up
/// to 4); on hardware (fast-math fuses the weights and sums) it is [`PARITY_BOUND_HARDWARE`] until the Metal run
/// gives a number. The comparison adds [`RELATIVE_BOUND`] of the value.
const PARITY_BOUND: f32 = 2.0e-5;

/// The same on a hardware adapter, until the Metal run gives a number.
const PARITY_BOUND_HARDWARE: f32 = 1.0e-4;

/// The part of the bound that grows with the value (`f32` sums of values above white).
const RELATIVE_BOUND: f32 = 1.0e-5;

/// The bound for the device in hand.
fn bound(dev: &TestGpu) -> f32 {
    if dev.gpu.info().is_software {
        PARITY_BOUND
    } else {
        PARITY_BOUND_HARDWARE
    }
}

/// The share of samples that may differ by more than the bound because of a flipped decision (a few pixels in
/// 90 million did).
const FLIP_SHARE: f64 = 5e-6;

/// No sample may differ by more than this: a flipped decision moves a pixel by a fraction of the local variation
/// (the largest seen was 9.4e-5, one pixel's tap count; `docs/21` section 14 allows 1e-3 of white).
const FLIP_BOUND: f32 = 5.0e-4;

const W: usize = 1000;
const H: usize = 667;

fn run<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("GPU job")
}

/// The scenes of the tests, made once.
fn textured() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| {
        balls_over_texture_scene(
            W,
            H,
            8.0,
            &[12.0, 17.5, 9.0, 14.0, 20.0, 11.0, 13.0, 10.0],
            Noise::NONE,
            4,
        )
    })
}

fn subject() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| subject_scene(W, H, [5.0, 9.0, 14.0], Noise::NONE, 1))
}

fn night() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| {
        soft_night_scene(
            W,
            H,
            &[
                (14.0, 0.25, 1.4),
                (9.0, 0.5, 0.9),
                (18.0, 0.15, 2.0),
                (11.0, 0.7, 1.2),
                (7.0, 0.3, 1.1),
                (16.0, 0.4, 1.6),
                (12.0, 0.2, 1.0),
            ],
            Noise::LOW,
            3,
            2.2,
        )
    })
}

fn whole(scene: &Scene) -> ImageRef<'_> {
    ImageRef {
        width: scene.img.w,
        height: scene.img.h,
        x0: 0,
        y0: 0,
        frame_width: scene.img.w,
        frame_height: scene.img.h,
        data: &scene.img.data,
    }
}

/// The planes of a plan made from a scene whose frame is the plan's frame (the lens plane is the scene).
fn planes_of_scene(p: &LensPlan, scene: &Scene) -> LensPlanes {
    assert_eq!(p.lens_plane_scale(), 1, "the scenes are the lens plane");
    LensPlanes::build(p, &whole(scene), scene.focus, &|| false).expect("planes")
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
        // A ball carries more light than white (up to 14 in these scenes): its drawn values have the `f32`
        // error of their size.
        if d > bound + RELATIVE_BOUND * w.abs() {
            over += 1;
        }
    }
    Found {
        worst,
        over,
        total: got.len(),
    }
}

/// The input a rectangle `out` needs: `out` grown by the pass's halo, cut from the whole image.
fn input_for(halo: usize, whole: &[f32], frame: (usize, usize), out: Rect) -> (Vec<f32>, Rect) {
    let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, frame.0, frame.1);
    let rect = Rect::new(r.x, r.y, r.w, r.h);
    (crop(whole, frame.0, frame.1, rect), rect)
}

/// The CPU reference over `out` of a frame at scale `k`, from the whole image.
fn cpu_reshape(
    p: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    image: &[f32],
    frame: (usize, usize),
    out: Rect,
) -> Vec<f32> {
    let halo = ym_lens::reshape_halo(p, planes, k);
    let (input, rect) = input_for(halo, image, frame, out);
    let input = ImageRef {
        width: rect.w,
        height: rect.h,
        x0: rect.x,
        y0: rect.y,
        frame_width: frame.0,
        frame_height: frame.1,
        data: &input,
    };
    ym_lens::reshape(
        p,
        planes,
        k,
        &input,
        PixelRect::new(out.x, out.y, out.w, out.h),
    )
    .expect("the CPU reference")
}

/// The kernel over `out` of a frame at scale `k`, from the whole image cut to the pass's halo.
fn gpu_reshape(
    dev: &TestGpu,
    p: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    image: &[f32],
    frame: (usize, usize),
    out: Rect,
) -> Vec<f32> {
    let plane_data = planes.bokeh_plane_data().expect("a blur map");
    let scale = p
        .bokeh_scale_data(planes, k, frame.0, frame.1)
        .expect("a pass");
    let (input, rect) = input_for(scale.pass.halo(), image, frame, out);
    run(dev, move |ctx| {
        reshape(
            ctx,
            &plane_data,
            &scale,
            &input,
            (rect.w, rect.h),
            (rect.x, rect.y),
            out,
        )
    })
}

/// The scenes' images reduced `k` times by box means (the frame at scale `k`).
fn reduce(data: &[f32], w: usize, h: usize, k: usize) -> (Vec<f32>, usize, usize) {
    let (rw, rh) = (w.div_ceil(k), h.div_ceil(k));
    let mut out = vec![0.0f32; 3 * rw * rh];
    for c in 0..3 {
        for y in 0..rh {
            for x in 0..rw {
                let (mut acc, mut cnt) = (0.0f64, 0.0f64);
                for dy in 0..k {
                    for dx in 0..k {
                        let (sx, sy) = (x * k + dx, y * k + dy);
                        if sx < w && sy < h {
                            acc += f64::from(data[c * w * h + sy * w + sx]);
                            cnt += 1.0;
                        }
                    }
                }
                out[c * rw * rh + y * rw + x] = (acc / cnt) as f32;
            }
        }
    }
    (out, rw, rh)
}

/// One comparison of the kernel and the reference on the whole frame: the sweeps' cell.
struct Cell {
    found: Found,
    changed: f32,
    cpu_s: f64,
    gpu_s: f64,
}

fn compare_case(
    dev: &TestGpu,
    p: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    image: &[f32],
    frame: (usize, usize),
) -> Option<Cell> {
    let pass = p.bokeh_pass(planes, k, frame.0, frame.1)?;
    assert!(pass.halo() > 0);
    let out = Rect::whole(frame.0, frame.1);
    let t = std::time::Instant::now();
    let want = cpu_reshape(p, planes, k, image, frame, out);
    let cpu_s = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let got = gpu_reshape(dev, p, planes, k, image, frame, out);
    let gpu_s = t.elapsed().as_secs_f64();
    Some(Cell {
        found: compare(&got, &want, bound(dev)),
        changed: exact::max_abs_diff(&want, image),
        cpu_s,
        gpu_s,
    })
}

#[test]
fn the_kernel_matches_the_cpu_reference() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let mut cases = 0;
    let mut worst_all = 0.0f32;
    let (mut over_all, mut total_all) = (0usize, 0usize);
    let (mut cpu_s, mut gpu_s) = (0.0f64, 0.0f64);
    type Sweep<'a> = (&'a str, &'a Scene, Axis, &'a [LensKind], &'a [(f64, f64)]);
    let sweeps: [Sweep; 5] = [
        (
            "balls over texture",
            textured(),
            Axis::X,
            &LensKind::ALL,
            &[(0.0, 100.0), (0.0, 35.0), (60.0, 70.0), (100.0, 100.0)],
        ),
        (
            "balls over texture, y axis",
            textured(),
            Axis::Y,
            &[LensKind::Vintage, LensKind::Petzval],
            &[(100.0, 100.0), (30.0, 50.0)],
        ),
        (
            "sharp subject",
            subject(),
            Axis::X,
            &[LensKind::Vintage, LensKind::Detuned, LensKind::Default],
            &[(0.0, 100.0), (100.0, 100.0)],
        ),
        (
            "night",
            night(),
            Axis::X,
            &[LensKind::Vintage, LensKind::Triplet, LensKind::Default],
            &[(100.0, 100.0), (0.0, 50.0)],
        ),
        (
            "night, y axis",
            night(),
            Axis::Y,
            &[LensKind::Petzval],
            &[(80.0, 60.0)],
        ),
    ];
    for (name, scene, axis, kinds, amount_pairs) in sweeps {
        for &kind in kinds {
            for &(edges, anamorphic) in amount_pairs {
                let p = plan(amounts(kind, 0.0, edges, anamorphic), W, H, axis);
                let planes = planes_of_scene(&p, scene);
                let Some(cell) = compare_case(dev, &p, &planes, 1, &scene.img.data, (W, H)) else {
                    continue;
                };
                cases += 1;
                worst_all = worst_all.max(cell.found.worst);
                over_all += cell.found.over;
                total_all += cell.found.total;
                cpu_s += cell.cpu_s;
                gpu_s += cell.gpu_s;
                note!(
                    "L1 {name} {kind} edges {edges} anamorphic {anamorphic}: worst {:e}, {} over {:e}; the picture moved by up to {:.3}",
                    cell.found.worst,
                    cell.found.over,
                    bound(dev),
                    cell.changed
                );
                assert!(
                    cell.found.over as f64 <= FLIP_SHARE * cell.found.total as f64
                        && cell.found.worst <= FLIP_BOUND,
                    "{name} {kind} {axis:?} edges {edges} anamorphic {anamorphic}: {} of {} samples differ by more than {} (worst {})",
                    cell.found.over,
                    cell.found.total,
                    bound(dev),
                    cell.found.worst
                );
            }
        }
    }
    assert!(cases >= 20, "the sweep ran {cases} cases");
    note!(
        "L1 timing over {cases} cases: CPU reference {cpu_s:.1} s, device (lavapipe, with uploads) {gpu_s:.1} s"
    );
    note!(
        "L1 parity: {cases} cases, {total_all} samples, worst difference {worst_all:e}, {over_all} over {:e}",
        bound(dev)
    );
}

/// A map of the frame whose cells (2 x 2 pixels) left of `split_x` are in focus and right of it blurred by
/// `radius_px`.
fn split_map(split_x: usize, radius_px: f64) -> BlurMapInput {
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    let (mut b, mut p) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
    for y in 0..ch {
        for x in 0..cw {
            let sharp = 2 * x + 1 < split_x;
            b[y * cw + x] = if sharp { 0.0 } else { (radius_px / hd) as f32 };
            p[y * cw + x] = if sharp { 0.02 } else { 0.95 };
        }
    }
    BlurMapInput {
        width: cw,
        height: ch,
        plane_width: W,
        plane_height: H,
        b,
        p,
    }
}

#[test]
fn the_kernel_matches_the_reference_on_the_grain_and_a_thin_bright_end() {
    // Three scenes of the grain and thin-end rules (L1, round 4): a smooth field with sensor grain (the grain a gather
    // takes out is given back, in square roots, gated by the noise), a pale bar whose body is in focus and whose end
    // lies in the wedge (the stretch is held back on a thin bright thing there), both beside a sharp part, and a
    // clean field with a faint diagonal line (the window's spread and the residual inside the gates' ramps).
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let n = W * H;
    let mut rng = scenes::Rng::new(77);
    let mut field = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for y in 0..H {
            for x in 0..W {
                let smooth = 0.25 + 0.03 * (x as f64 / 110.0).sin() * (y as f64 / 80.0).cos();
                field[c * n + y * W + x] = (smooth + 0.01 * rng.normal()).max(0.0) as f32;
            }
        }
    }
    let mut bar = vec![0.1f32; 3 * n];
    for c in 0..3 {
        for y in 318..349 {
            let across = (-(((y as f64 + 0.5) - 333.5) / 3.0).powi(2) / 2.0).exp();
            for x in 200..350 {
                let along = if x < 330 {
                    1.0
                } else {
                    0.5 * (1.0 + (std::f64::consts::PI * (x as f64 - 330.0) / 20.0).cos())
                };
                bar[c * n + y * W + x] += (0.5 * across * along) as f32;
            }
        }
    }
    // A clean photo (grain 0.0012, under the detector's floor, so the gates are the floor's multiples) with a faint
    // diagonal line of 0.01: the window's spread and the residual of its pixels are in the gates' ramps.
    let mut rng = scenes::Rng::new(78);
    let mut line = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for y in 0..H {
            for x in 0..W {
                let smooth = 0.25 + 0.03 * (x as f64 / 110.0).sin() * (y as f64 / 80.0).cos();
                let d = ((x as f64 - y as f64) - 300.0) / std::f64::consts::SQRT_2;
                let faint = if (100..560).contains(&y) && x >= 560 {
                    0.01 * (-(d * d) / 2.0).exp()
                } else {
                    0.0
                };
                line[c * n + y * W + x] = (smooth + faint + 0.0012 * rng.normal()).max(0.0) as f32;
            }
        }
    }
    let mut cases = 0;
    let mut worst_all = 0.0f32;
    for (name, image) in [("grain", &field), ("thin end", &bar), ("faint line", &line)] {
        for (kind, edges) in [(LensKind::Default, 0.0), (LensKind::Petzval, 70.0)] {
            let p = plan(amounts(kind, 0.0, edges, 100.0), W, H, Axis::X);
            let planes = LensPlanes::build_with_map(
                &p,
                &ImageRef {
                    width: W,
                    height: H,
                    x0: 0,
                    y0: 0,
                    frame_width: W,
                    frame_height: H,
                    data: image,
                },
                None,
                Some(&split_map(330, 12.0)),
                &|| false,
            )
            .expect("planes");
            let Some(cell) = compare_case(dev, &p, &planes, 1, image, (W, H)) else {
                continue;
            };
            cases += 1;
            worst_all = worst_all.max(cell.found.worst);
            note!(
                "L1 {name} {kind} edges {edges}: worst {:e}, {} over {:e}; the picture moved by up to {:.3}",
                cell.found.worst,
                cell.found.over,
                bound(dev),
                cell.changed
            );
            assert!(
                cell.found.over as f64 <= FLIP_SHARE * cell.found.total as f64
                    && cell.found.worst <= FLIP_BOUND,
                "{name} {kind}: {} of {} samples differ by more than {} (worst {})",
                cell.found.over,
                cell.found.total,
                bound(dev),
                cell.found.worst
            );
        }
    }
    assert_eq!(cases, 6);
    note!("L1 grain and thin end parity: {cases} cases, worst difference {worst_all:e}");
}

/// A larger scene (1400 x 900, whose lens plane is its half) for the tiles: balls over a blurred texture, some
/// of which lie across the 512 and 1024 px tile boundaries.
fn big() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| {
        balls_over_texture_scene(
            1400,
            900,
            10.0,
            &[
                24.0, 30.0, 18.0, 26.0, 36.0, 22.0, 20.0, 28.0, 16.0, 32.0, 19.0, 25.0,
            ],
            Noise::NONE,
            9,
        )
    })
}

/// The planes of a plan whose lens plane is `scene` itself (a frame `k_L` times its size).
fn planes_from_plane(p: &LensPlan, scene: &Scene) -> LensPlanes {
    planes_from_image(p, &scene.img.data, scene.img.w, scene.img.h)
}

fn planes_from_image(p: &LensPlan, data: &[f32], w: usize, h: usize) -> LensPlanes {
    let image = ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data,
    };
    LensPlanes::build(p, &image, None, &|| false).expect("planes")
}

/// A lens plane of `w x h`: a texture blurred by a disc of `bg` px with flat bright discs `(x, y, radius)` on it.
fn plane_with_balls(w: usize, h: usize, bg: f64, balls: &[(f64, f64, f64)], seed: u64) -> Vec<f32> {
    let tex = scenes::texture(w, h, 30.0, seed, 0, w);
    let n = w * h;
    let mut img = scenes::Rgb::new(w, h);
    for c in 0..3 {
        img.plane_mut(c)
            .copy_from_slice(&scenes::disc_blur(tex.plane(c), w, h, bg));
    }
    for &(cx, cy, r) in balls {
        let (x0, x1) = (
            (cx - r - 2.0).floor().max(0.0) as usize,
            ((cx + r + 2.0).ceil() as usize).min(w),
        );
        let (y0, y1) = (
            (cy - r - 2.0).floor().max(0.0) as usize,
            ((cy + r + 2.0).ceil() as usize).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0);
                if a > 0.0 {
                    for (c, tone) in [4.0, 3.4, 2.6].into_iter().enumerate() {
                        let v = &mut img.data[c * n + y * w + x];
                        *v = (f64::from(*v) * (1.0 - a) + tone * a) as f32;
                    }
                }
            }
        }
    }
    img.data
}

/// A smooth picture of any position of a frame, so a region far from the origin needs no image of the frame.
fn smooth(c: usize, x: usize, y: usize) -> f32 {
    let (xf, yf, cf) = (x as f64, y as f64, c as f64);
    (0.35
        + 0.2 * (0.013 * xf + 1.1 * cf).sin() * (0.011 * yf - 0.4 * cf).cos()
        + 0.1 * (0.0071 * (xf + yf) + cf).sin()) as f32
}

/// The planes of a plan whose lens plane is `scene` reduced `k_L` times (`k_L` the plan's).
fn planes_of_big(p: &LensPlan, scene: &Scene) -> LensPlanes {
    let k = p.lens_plane_scale();
    let (plane, pw, ph) = reduce(&scene.img.data, scene.img.w, scene.img.h, k);
    let image = ImageRef {
        width: pw,
        height: ph,
        x0: 0,
        y0: 0,
        frame_width: pw,
        frame_height: ph,
        data: &plane,
    };
    LensPlanes::build(p, &image, scene.focus, &|| false).expect("planes")
}

fn tile_of(
    dev: &TestGpu,
    p: &LensPlan,
    planes: &LensPlanes,
    image: &[f32],
    frame: (usize, usize),
    out: Rect,
) -> Vec<f32> {
    gpu_reshape(dev, p, planes, 1, image, frame, out)
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let scene = big();
    let (w, h) = (scene.img.w, scene.img.h);
    for (kind, edges, anamorphic, axis) in [
        (LensKind::Vintage, 100.0, 100.0, Axis::X),
        (LensKind::Petzval, 70.0, 40.0, Axis::Y),
    ] {
        let p = plan(amounts(kind, 0.0, edges, anamorphic), w, h, axis);
        let planes = planes_of_big(&p, scene);
        let bp = planes.bokeh().expect("the blur map");
        assert!(bp.active(), "the scene is reshaped");
        let pass = p.bokeh_pass(&planes, 1, w, h).expect("a pass");
        // Some ball lies across a tile boundary, so a ball is drawn by two tiles.
        let frame = (w as f64, h as f64);
        let crossing = |tile: f64| {
            bp.balls().iter().any(|b| {
                let (cx, cy, r) = (b.x * frame.0, b.y * frame.1, b.radius * pass.half_diag);
                [(cx, r), (cy, r)]
                    .iter()
                    .any(|&(c, r)| (c / tile).floor() != ((c + 2.0 * r) / tile).floor())
            })
        };
        assert!(
            crossing(512.0) && crossing(1024.0),
            "a ball across a tile boundary"
        );
        let whole = exact::assert_tiles_equal_whole(
            &format!("L1 {kind} edges {edges} anamorphic {anamorphic} {axis:?}"),
            w,
            h,
            3,
            &[512, 1024],
            |rect| tile_of(dev, &p, &planes, &scene.img.data, (w, h), rect),
        );
        // The reshaping did something, and the whole frame is what the reference makes.
        assert!(exact::max_abs_diff(&whole, &scene.img.data) > 1e-2);
        let want = cpu_reshape(&p, &planes, 1, &scene.img.data, (w, h), Rect::whole(w, h));
        let f = compare(&whole, &want, bound(dev));
        // Where the worst difference lies, and how large the value there is (the flipped decisions are at the balls).
        let (at, _) = whole
            .iter()
            .zip(&want)
            .enumerate()
            .map(|(i, (g, r))| (i, (g - r).abs()))
            .fold((0, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
        let (plane, within) = (at / (w * h), at % (w * h));
        note!(
            "L1 {kind} {w}x{h}: tiles of 512 and 1024 equal the whole frame; against the reference worst {:e}, {} over {:e}; the worst at channel {plane}, ({}, {}), value {:.3} (the reference's), the input's {:.3}",
            f.worst,
            f.over,
            bound(dev),
            within % w,
            within / w,
            want[at],
            scene.img.data[at]
        );
        assert!(f.over as f64 <= FLIP_SHARE * f.total as f64 && f.worst <= FLIP_BOUND);
    }
}

#[test]
fn a_source_cut_out_with_another_origin_gives_the_same_bits() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let scene = textured();
    let p = plan(amounts(LensKind::Vintage, 0.0, 90.0, 80.0), W, H, Axis::X);
    let planes = planes_of_scene(&p, scene);
    let out = Rect::new(333, 211, 129, 197);
    let frame = (W, H);
    // The whole image, with more than the halo around, and exactly the halo's rectangle.
    let plane_data = planes.bokeh_plane_data().expect("map");
    let scale = p.bokeh_scale_data(&planes, 1, W, H).expect("pass");
    let halo = scale.pass.halo();
    let (input, rect) = input_for(halo, &scene.img.data, frame, out);
    let from_cut = {
        let (plane_data, scale) = (plane_data.clone(), scale.clone());
        run(dev, move |ctx| {
            reshape(
                ctx,
                &plane_data,
                &scale,
                &input,
                (rect.w, rect.h),
                (rect.x, rect.y),
                out,
            )
        })
    };
    let all = scene.img.data.clone();
    let from_whole = run(dev, move |ctx| {
        reshape(ctx, &plane_data, &scale, &all, (W, H), (0, 0), out)
    });
    assert!(rect.x > 0 && rect.y > 0 && rect.w < W && halo > 0);
    exact::assert_bit_equal("a cut-out source", &from_cut, &from_whole);
}

#[test]
fn a_reduced_frame_reads_the_coarser_levels_of_the_map() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let scene = textured();
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 100.0), W, H, Axis::X);
    let planes = planes_of_scene(&p, scene);
    for k in [2usize, 4] {
        let (image, rw, rh) = reduce(&scene.img.data, W, H, k);
        let pass = p.bokeh_pass(&planes, k, rw, rh).expect("a pass");
        assert!(pass.level > 0, "the cells are under a pixel apart at 1/{k}");
        let out = Rect::whole(rw, rh);
        let want = cpu_reshape(&p, &planes, k, &image, (rw, rh), out);
        let got = gpu_reshape(dev, &p, &planes, k, &image, (rw, rh), out);
        let f = compare(&got, &want, bound(dev));
        note!(
            "L1 at 1/{k} (level {} of the map, {rw}x{rh}): worst {:e}, {} over {:e}",
            pass.level,
            f.worst,
            f.over,
            bound(dev)
        );
        assert!(
            f.over as f64 <= FLIP_SHARE * f.total as f64 && f.worst <= FLIP_BOUND,
            "1/{k}: {} of {} over the bound (worst {})",
            f.over,
            f.total,
            f.worst
        );
        assert!(
            exact::max_abs_diff(&want, &image) > 1e-2,
            "the reshaping did something"
        );
    }
}

#[test]
fn poison_and_a_plain_device_give_the_same_bits() {
    let _guard = serial();
    let Some(_dev) = common::lens::gpu() else {
        return;
    };
    let scene = textured();
    for (kind, edges, anamorphic, axis) in [
        (LensKind::Vintage, 100.0, 100.0, Axis::X),
        (LensKind::Detuned, 60.0, 30.0, Axis::Y),
    ] {
        let p = plan(amounts(kind, 0.0, edges, anamorphic), W, H, axis);
        let planes = planes_of_scene(&p, scene);
        let plane_data = planes.bokeh_plane_data().expect("map");
        let scale = p.bokeh_scale_data(&planes, 1, W, H).expect("pass");
        let out = Rect::new(40, 30, 600, 420);
        let (input, rect) = input_for(scale.pass.halo(), &scene.img.data, (W, H), out);
        let got = common::lens::assert_poison_neutral("L1", move |gpu| {
            let (plane_data, scale, input) = (plane_data.clone(), scale.clone(), input.clone());
            gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                reshape(
                    ctx,
                    &plane_data,
                    &scale,
                    &input,
                    (rect.w, rect.h),
                    (rect.x, rect.y),
                    out,
                )
            })
            .expect("reshape")
        });
        assert!(got.is_some());
    }
}

#[test]
fn a_job_between_two_runs_changes_nothing() {
    // D7: what ran before (here a run of another lens on another tile) does not change a run.
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let scene = textured();
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 100.0), W, H, Axis::X);
    let planes = planes_of_scene(&p, scene);
    let out = Rect::new(100, 80, 400, 300);
    let first = gpu_reshape(dev, &p, &planes, 1, &scene.img.data, (W, H), out);
    let other = plan(amounts(LensKind::Petzval, 0.0, 40.0, 60.0), W, H, Axis::Y);
    let other_planes = planes_of_scene(&other, scene);
    let _ = gpu_reshape(
        dev,
        &other,
        &other_planes,
        1,
        &scene.img.data,
        (W, H),
        Rect::new(300, 200, 500, 400),
    );
    let second = gpu_reshape(dev, &p, &planes, 1, &scene.img.data, (W, H), out);
    exact::assert_bit_equal("a run after another", &second, &first);
}

#[test]
fn nothing_runs_when_the_reshaping_has_nothing_to_do() {
    let _guard = serial();
    let scene = textured();
    // Anamorphic at 0, with Edges at 0 or at 100 (the lens's swirl is Anamorphic's): no blur map is needed, no pass
    // exists.
    for edges in [0.0, 100.0] {
        let p = plan(amounts(LensKind::Vintage, 60.0, edges, 0.0), W, H, Axis::X);
        assert!(!p.needs_blur_map());
        let planes = planes_of_scene(&p, scene);
        assert!(planes.bokeh().is_none());
        assert!(p.bokeh_pass(&planes, 1, W, H).is_none());
        assert!(p.bokeh_scale_data(&planes, 1, W, H).is_none());
        assert!(planes.bokeh_plane_data().is_none());
    }
    // A scene with no out-of-focus background: the planes are inactive, there is no pass.
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 100.0), W, H, Axis::X);
    let sharp = scenes::sharp_scene(W, H, Noise::NONE, 2);
    let planes = LensPlanes::build(&p, &whole(&sharp), None, &|| false).expect("planes");
    assert!(planes.bokeh().is_some_and(|b| !b.active()));
    assert!(p.bokeh_pass(&planes, 1, W, H).is_none());
    assert!(planes.bokeh_plane_data().is_none());
}

#[test]
fn the_binding_stays_inside_the_design_size() {
    let _guard = serial();
    let (w, h) = (6000usize, 4000usize);
    let p = plan(amounts(LensKind::Petzval, 0.0, 100.0, 100.0), w, h, Axis::X);
    // The lens plane of a 6000 x 4000 frame is 1500 x 1000.
    let scene =
        balls_over_texture_scene(1500, 1000, 7.0, &[14.0, 20.0, 11.0, 17.0], Noise::NONE, 6);
    let planes = planes_from_plane(&p, &scene);
    let pass = p.bokeh_pass(&planes, 1, w, h).expect("a pass");
    note!(
        "L1 largest binding of a 1024 px tile of a 6000 x 4000 frame: {:.1} MiB (halo {} px)",
        ym_gpu::kernels::lens_reshape::max_binding_bytes(Rect::new(2000, 1500, 1024, 1024), &pass)
            as f64
            / f64::from(1u32 << 20),
        pass.halo()
    );
    common::limits::assert_max_binding_bytes("lens reshape", Rect::new(0, 0, 1024, 1024), |r| {
        ym_gpu::kernels::lens_reshape::max_binding_bytes(r, &pass)
    });
}

#[test]
fn the_binding_stays_inside_the_design_size_at_the_blur_cap_of_a_50_mp_frame() {
    // The longest lines: the lens plane's balls are blurred by more than `b_cap` (0.03 half-diagonals), so the
    // vertical line (Anamorphic 100), the tangential one (the Petzval's field swirl at Edges 100, held to
    // `TANGENT_REACH_MAX`), the feather and the guide all reach their ends, on an 8658 x 5772 frame (lens plane
    // 1083 x 722). The 1024 px tile of the export and its window must still be one binding of the design size.
    let _guard = serial();
    let (w, h) = (8658usize, 5772usize);
    let p = plan(amounts(LensKind::Petzval, 0.0, 100.0, 100.0), w, h, Axis::X);
    assert_eq!(p.lens_plane_scale(), 8);
    let scene =
        balls_over_texture_scene(1083, 722, 12.0, &[24.0, 26.0, 22.0, 25.0], Noise::NONE, 8);
    let planes = planes_from_plane(&p, &scene);
    let pass = p.bokeh_pass(&planes, 1, w, h).expect("a pass");
    let reach = ym_lens::consts::TANGENT_REACH_MAX * pass.half_diag;
    note!(
        "L1 at the blur cap on 8658 x 5772: tangential half-length at most {:.0} px, halo {} px (vertical {}, tangential {}, feather {}), largest binding of a 1024 px tile {:.1} MiB, of a 512 px tile {:.1} MiB",
        reach,
        pass.halo(),
        pass.v_halo,
        pass.t_halo,
        pass.feather,
        ym_gpu::kernels::lens_reshape::max_binding_bytes(Rect::new(3000, 2000, 1024, 1024), &pass)
            as f64
            / f64::from(1u32 << 20),
        ym_gpu::kernels::lens_reshape::max_binding_bytes(Rect::new(3000, 2000, 512, 512), &pass)
            as f64
            / f64::from(1u32 << 20),
    );
    assert!(
        pass.t_halo as f64 <= reach + ym_lens::consts::GATHER_PAD as f64 + 1.0,
        "the tangential halo {} is held to the reach {reach}",
        pass.t_halo
    );
    common::limits::assert_max_binding_bytes(
        "lens reshape at the cap",
        Rect::new(3000, 2000, 1024, 1024),
        |r| ym_gpu::kernels::lens_reshape::max_binding_bytes(r, &pass),
    );
}

#[test]
fn a_source_that_does_not_cover_the_halo_is_refused() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let scene = textured();
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 100.0), W, H, Axis::X);
    let planes = planes_of_scene(&p, scene);
    let plane_data = planes.bokeh_plane_data().expect("map");
    let scale = p.bokeh_scale_data(&planes, 1, W, H).expect("pass");
    // The output is the middle of the frame and the source is exactly the output: no halo.
    let out = Rect::new(300, 200, 100, 100);
    let small = crop(&scene.img.data, W, H, out);
    let result = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            reshape(
                ctx,
                &plane_data,
                &scale,
                &small,
                (100, 100),
                (300, 200),
                out,
            )
        });
    assert!(
        matches!(result, Err(GpuError::InvalidInput(_))),
        "{result:?}"
    );
}

#[test]
fn the_far_corner_of_a_50_mp_frame_is_as_exact_as_the_near_one() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let (w, h) = (8658usize, 5772usize);
    // The lens plane of this frame is 1082 x 722 (k_L = 8); balls at the corners and in the middle.
    let (pw, ph) = (1082usize, 722usize);
    let balls = [
        (22.0, 20.0, 12.0),
        (541.0, 361.0, 14.0),
        (1058.0, 704.0, 12.0),
        (200.0, 500.0, 16.0),
        (900.0, 150.0, 10.0),
        (700.0, 600.0, 13.0),
        (330.0, 120.0, 11.0),
    ];
    let plane = plane_with_balls(pw, ph, 4.0, &balls, 21);
    let p = plan(amounts(LensKind::Vintage, 0.0, 100.0, 100.0), w, h, Axis::X);
    assert_eq!(p.lens_plane_scale(), 8);
    let planes = planes_from_image(&p, &plane, pw, ph);
    assert!(
        planes.bokeh().is_some_and(|b| b.active()),
        "the plane is reshaped"
    );
    assert!(
        planes.bokeh().is_some_and(|b| b.balls().len() >= 5),
        "the balls are found"
    );
    let plane_data = planes.bokeh_plane_data().expect("map");
    let scale = p.bokeh_scale_data(&planes, 1, w, h).expect("pass");
    let halo = scale.pass.halo();
    let mut worsts = Vec::new();
    for (name, out) in [
        ("near corner", Rect::new(0, 0, 64, 64)),
        ("centre", Rect::new(w / 2 - 32, h / 2 - 32, 64, 64)),
        ("far corner", Rect::new(w - 64, h - 64, 64, 64)),
    ] {
        let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, w, h);
        let rect = Rect::new(r.x, r.y, r.w, r.h);
        let mut input = Vec::with_capacity(3 * rect.w * rect.h);
        for c in 0..3 {
            for y in rect.y..rect.y1() {
                for x in rect.x..rect.x1() {
                    input.push(smooth(c, x, y));
                }
            }
        }
        let image = ImageRef {
            width: rect.w,
            height: rect.h,
            x0: rect.x,
            y0: rect.y,
            frame_width: w,
            frame_height: h,
            data: &input,
        };
        let want = ym_lens::reshape(
            &p,
            &planes,
            1,
            &image,
            PixelRect::new(out.x, out.y, out.w, out.h),
        )
        .expect("the CPU reference");
        let got = {
            let (plane_data, scale, input) = (plane_data.clone(), scale.clone(), input.clone());
            run(dev, move |ctx| {
                reshape(
                    ctx,
                    &plane_data,
                    &scale,
                    &input,
                    (rect.w, rect.h),
                    (rect.x, rect.y),
                    out,
                )
            })
        };
        let f = compare(&got, &want, bound(dev));
        let original = crop(
            &input,
            rect.w,
            rect.h,
            Rect::new(out.x - rect.x, out.y - rect.y, out.w, out.h),
        );
        let moved = exact::max_abs_diff(&want, &original);
        note!(
            "L1 {name}: worst difference {:e} over {} samples; the reference moved the picture by up to {moved:.3}",
            f.worst,
            f.total
        );
        assert!(
            f.over == 0,
            "{name}: {} of {} samples differ by more than {}",
            f.over,
            f.total,
            bound(dev)
        );
        assert!(moved > 0.02, "{name}: the pass did something");
        worsts.push(f.worst);
    }
    // Position errors at the far corner would show as differences orders above the near corner's.
    assert!(
        worsts[2] <= 4.0 * worsts[0].max(1e-6),
        "the far corner ({:e}) is not as exact as the near one ({:e})",
        worsts[2],
        worsts[0]
    );
}

// ---------------------------------------------------------------------------------------------------------
// Timing on a real lens plane (ignored: it needs the dump of `lens_sheet.rs --dump-planes`)
// ---------------------------------------------------------------------------------------------------------

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// An Anamorphic move on a real photo with bokeh, CPU reference against the device (Mesa lavapipe: relative
/// numbers only, the CPU is the reference's own at the dev profile).
///
/// ```text
/// YM_LENS_PLANE_DIR=<folder with lens_plane.f32 and lens_plane.json> WGPU_BACKEND=vulkan \
///   cargo test -p ym-gpu --test gpu_lens_reshape -- --ignored --nocapture l1_timing
/// ```
#[test]
#[ignore = "needs YM_LENS_PLANE_DIR, a dump of a real lens plane"]
fn l1_timing_of_an_anamorphic_move_on_a_real_plane() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let Some(dir) = std::env::var_os("YM_LENS_PLANE_DIR").map(std::path::PathBuf::from) else {
        note!("YM_LENS_PLANE_DIR is not set");
        return;
    };
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("lens_plane.json")).expect("json"))
            .expect("parse");
    let num = |k: &str| json[k].as_u64().expect(k) as usize;
    let (pw, ph, fw, fh) = (
        num("width"),
        num("height"),
        num("frame_width"),
        num("frame_height"),
    );
    let plane = read_f32(&dir.join("lens_plane.f32"));
    assert_eq!(plane.len(), 3 * pw * ph);
    let make = |anamorphic: f64| {
        plan(
            amounts(LensKind::Vintage, 0.0, 0.0, anamorphic),
            fw,
            fh,
            Axis::X,
        )
    };
    let p = make(100.0);
    assert_eq!(p.lens_plane_scale(), num("k_L"));
    let t = std::time::Instant::now();
    let planes = planes_from_image(&p, &plane, pw, ph);
    let bp = planes.bokeh().expect("a blur map");
    note!(
        "real plane {pw}x{ph} of a {fw}x{fh} frame: planes {:.1} s; active {}, {} balls",
        t.elapsed().as_secs_f64(),
        bp.active(),
        bp.balls().len()
    );
    assert!(bp.active());
    // The fit-view group: the frame at scale k_L, the lens plane itself; and a 1024 px tile at full resolution,
    // the plane scaled up (bilinear).
    let k = num("k_L");
    let full = |c: usize, x: usize, y: usize| -> f32 {
        let fx = ((x as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, (pw - 1) as f64);
        let fy = ((y as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, (ph - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(pw - 1), (y0 + 1).min(ph - 1));
        let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
        let q = &plane[c * pw * ph..(c + 1) * pw * ph];
        let top = q[y0 * pw + x0] * (1.0 - tx) + q[y0 * pw + x1] * tx;
        let bot = q[y1 * pw + x0] * (1.0 - tx) + q[y1 * pw + x1] * tx;
        top * (1.0 - ty) + bot * ty
    };
    for (what, scale_k, frame, out, image_at) in [
        (
            "fit-view group (1/4)",
            k,
            (pw, ph),
            Rect::whole(pw, ph),
            None,
        ),
        (
            "100 % tile of 1024",
            1,
            (fw, fh),
            Rect::new(fw / 2 - 512, fh / 2 - 512, 1024, 1024),
            Some(&full),
        ),
    ] {
        let mut cpu_ms = Vec::new();
        let mut gpu_ms = Vec::new();
        for anamorphic in [100.0, 99.0, 98.0] {
            let p = make(anamorphic);
            let planes = planes_from_image(&p, &plane, pw, ph);
            let scale = p
                .bokeh_scale_data(&planes, scale_k, frame.0, frame.1)
                .expect("pass");
            let halo = scale.pass.halo();
            let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, frame.0, frame.1);
            let rect = Rect::new(r.x, r.y, r.w, r.h);
            let mut input = Vec::with_capacity(3 * rect.w * rect.h);
            for c in 0..3 {
                for y in rect.y..rect.y1() {
                    for x in rect.x..rect.x1() {
                        input.push(match image_at {
                            Some(f) => f(c, x, y),
                            None => plane[c * pw * ph + y * pw + x],
                        });
                    }
                }
            }
            let image = ImageRef {
                width: rect.w,
                height: rect.h,
                x0: rect.x,
                y0: rect.y,
                frame_width: frame.0,
                frame_height: frame.1,
                data: &input,
            };
            let t = std::time::Instant::now();
            let want = ym_lens::reshape(
                &p,
                &planes,
                scale_k,
                &image,
                PixelRect::new(out.x, out.y, out.w, out.h),
            )
            .expect("reference");
            cpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let plane_data = planes.bokeh_plane_data().expect("map");
            let t = std::time::Instant::now();
            let got = {
                let input = input.clone();
                run(dev, move |ctx| {
                    reshape(
                        ctx,
                        &plane_data,
                        &scale,
                        &input,
                        (rect.w, rect.h),
                        (rect.x, rect.y),
                        out,
                    )
                })
            };
            gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let f = compare(&got, &want, bound(dev));
            note!(
                "  {what}, Anamorphic {anamorphic}: halo {halo} px, worst {:e}, {} over {:e}",
                f.worst,
                f.over,
                bound(dev)
            );
        }
        note!(
            "L1 {what} on the real plane: CPU reference {:.0} / {:.0} / {:.0} ms, device on lavapipe (with uploads of the map, the shapes and the picture) {:.0} / {:.0} / {:.0} ms",
            cpu_ms[0],
            cpu_ms[1],
            cpu_ms[2],
            gpu_ms[0],
            gpu_ms[1],
            gpu_ms[2]
        );
    }
}

/// The same on a synthetic frame blurred everywhere (the worst case for the gathers: every pixel has a line):
/// a 6000 x 4000 frame whose 1500 x 1000 lens plane is a texture blurred by 10 px with balls on it.
#[test]
#[ignore = "timing"]
fn l1_timing_of_an_anamorphic_move_on_a_fully_blurred_scene() {
    let _guard = serial();
    let Some(dev) = common::lens::gpu() else {
        return;
    };
    let (pw, ph, fw, fh, k) = (1500usize, 1000usize, 6000usize, 4000usize, 4usize);
    let scene = balls_over_texture_scene(
        pw,
        ph,
        10.0,
        &[
            14.0, 18.0, 11.0, 16.0, 22.0, 13.0, 15.0, 12.0, 20.0, 17.0, 19.0, 10.0,
        ],
        Noise::NONE,
        5,
    );
    let plane = &scene.img.data;
    let make = |anamorphic: f64| {
        plan(
            amounts(LensKind::Vintage, 0.0, 0.0, anamorphic),
            fw,
            fh,
            Axis::X,
        )
    };
    let p = make(100.0);
    let planes = planes_from_image(&p, plane, pw, ph);
    assert!(planes.bokeh().is_some_and(|b| b.active()));
    let full = |c: usize, x: usize, y: usize| -> f32 {
        let fx = ((x as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, (pw - 1) as f64);
        let fy = ((y as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, (ph - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(pw - 1), (y0 + 1).min(ph - 1));
        let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
        let q = &plane[c * pw * ph..(c + 1) * pw * ph];
        let top = q[y0 * pw + x0] * (1.0 - tx) + q[y0 * pw + x1] * tx;
        let bot = q[y1 * pw + x0] * (1.0 - tx) + q[y1 * pw + x1] * tx;
        top * (1.0 - ty) + bot * ty
    };
    for (what, scale_k, frame, out, image_at) in [
        (
            "fit-view group (1/4)",
            k,
            (pw, ph),
            Rect::whole(pw, ph),
            None,
        ),
        (
            "100 % tile of 1024",
            1,
            (fw, fh),
            Rect::new(fw / 2 - 512, fh / 2 - 512, 1024, 1024),
            Some(&full),
        ),
    ] {
        let (mut cpu_ms, mut gpu_ms) = (Vec::new(), Vec::new());
        for anamorphic in [100.0, 99.0] {
            let p = make(anamorphic);
            let scale = p
                .bokeh_scale_data(&planes, scale_k, frame.0, frame.1)
                .expect("pass");
            let halo = scale.pass.halo();
            let r = PixelRect::new(out.x, out.y, out.w, out.h).grown(halo, frame.0, frame.1);
            let rect = Rect::new(r.x, r.y, r.w, r.h);
            let mut input = Vec::with_capacity(3 * rect.w * rect.h);
            for c in 0..3 {
                for y in rect.y..rect.y1() {
                    for x in rect.x..rect.x1() {
                        input.push(match image_at {
                            Some(f) => f(c, x, y),
                            None => plane[c * pw * ph + y * pw + x],
                        });
                    }
                }
            }
            let image = ImageRef {
                width: rect.w,
                height: rect.h,
                x0: rect.x,
                y0: rect.y,
                frame_width: frame.0,
                frame_height: frame.1,
                data: &input,
            };
            let t = std::time::Instant::now();
            let want = ym_lens::reshape(
                &p,
                &planes,
                scale_k,
                &image,
                PixelRect::new(out.x, out.y, out.w, out.h),
            )
            .expect("reference");
            cpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let plane_data = planes.bokeh_plane_data().expect("map");
            let t = std::time::Instant::now();
            let got = {
                let input = input.clone();
                run(dev, move |ctx| {
                    reshape(
                        ctx,
                        &plane_data,
                        &scale,
                        &input,
                        (rect.w, rect.h),
                        (rect.x, rect.y),
                        out,
                    )
                })
            };
            gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let f = compare(&got, &want, bound(dev));
            note!(
                "  {what}, Anamorphic {anamorphic}: halo {halo} px, worst {:e}, {} over {:e}",
                f.worst,
                f.over,
                bound(dev)
            );
        }
        note!(
            "L1 {what} on the blurred scene: CPU reference {:.0} / {:.0} ms, device on lavapipe (with uploads) {:.0} / {:.0} ms",
            cpu_ms[0],
            cpu_ms[1],
            gpu_ms[0],
            gpu_ms[1]
        );
    }
}
