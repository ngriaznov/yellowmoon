//! L1, the bokeh reshaping, on synthetic scenes (`docs/21` sections 4 and 14).
//!
//! `reshape` is L1 alone (what the field optics would read); `apply` is the whole lens. Frames are 1000 x 667
//! (the lens plane is the frame, `k_L` = 1) unless a test says otherwise.

#![allow(clippy::needless_range_loop)]

mod common;
mod scenes;

use std::sync::OnceLock;

use common::{apply_tile, plan, whole};
use scenes::{Noise, Scene, erode, night_scene_spaced, sharp_scene, subject_scene};
use ym_lens::{
    Axis, ImageRef, LensError, LensKind, LensPlan, LensPlanes, PixelRect, ReshapeReason, apply,
    halo_px, reshape, reshape_halo,
};

const W: usize = 1000;
const H: usize = 667;

fn planes_of(p: &LensPlan, scene: &Scene) -> LensPlanes {
    // The lens plane of a frame of `k_L` is the scene reduced `k_L` times.
    let k = p.lens_plane_scale();
    let (data, w, h) = if k == 1 {
        (scene.img.data.clone(), scene.img.w, scene.img.h)
    } else {
        reduce(&scene.img.data, scene.img.w, scene.img.h, k)
    };
    LensPlanes::build(p, &whole(w, h, &data), scene.focus, &|| false).expect("planes")
}

/// The scene reduced `k` times by box means (the frame at scale `k`).
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

/// A mask reduced `k` times: a pixel is set when its whole block is.
fn reduce_mask(mask: &[bool], w: usize, h: usize, k: usize) -> Vec<bool> {
    let (rw, rh) = (w.div_ceil(k), h.div_ceil(k));
    let mut out = vec![false; rw * rh];
    for y in 0..rh {
        for x in 0..rw {
            out[y * rw + x] = (0..k).all(|dy| {
                (0..k).all(|dx| {
                    let (sx, sy) = (x * k + dx, y * k + dy);
                    sx >= w || sy >= h || mask[sy * w + sx]
                })
            });
        }
    }
    out
}

/// L1 alone over `rect` of a full-frame image at scale `k`, given exactly the context it needs.
fn reshape_tile(
    p: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    data: &[f32],
    (w, h): (usize, usize),
    rect: PixelRect,
) -> Vec<f32> {
    let halo = reshape_halo(p, planes, k);
    let need = rect.grown(halo, w, h);
    let buf = common::crop(data, w, h, need);
    let input = ImageRef {
        width: need.w,
        height: need.h,
        x0: need.x,
        y0: need.y,
        frame_width: w,
        frame_height: h,
        data: &buf,
    };
    reshape(p, planes, k, &input, rect).expect("reshape")
}

fn subject_base() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| subject_scene(W, H, [5.0, 9.0, 14.0], Noise::NONE, 1))
}

/// The ratio of the principal axes of the light above `base` in a window (second moments), the major axis angle
/// in degrees, and the light above `base` per channel.
struct Blob {
    elongation: f64,
    angle_deg: f64,
    energy: [f64; 3],
}

fn blob(
    out: &[f32],
    base: &[f32],
    w: usize,
    h: usize,
    (cx, cy, r): (f64, f64, f64),
    reach: f64,
) -> Blob {
    let n = w * h;
    let (x0, x1) = (
        (cx - reach * r).max(0.0) as usize,
        ((cx + reach * r) as usize).min(w),
    );
    let (y0, y1) = (
        (cy - reach * r).max(0.0) as usize,
        ((cy + reach * r) as usize).min(h),
    );
    let (mut m, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
    let mut energy = [0.0f64; 3];
    for y in y0..y1 {
        for x in x0..x1 {
            for (c, e) in energy.iter_mut().enumerate() {
                *e += f64::from(out[c * n + y * w + x]) - f64::from(base[c * n + y * w + x]);
            }
            let v = f64::from(out[n + y * w + x]) - f64::from(base[n + y * w + x]);
            m += v;
            mx += v * (x as f64 + 0.5);
            my += v * (y as f64 + 0.5);
        }
    }
    let (mx, my) = (mx / m, my / m);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let v = f64::from(out[n + y * w + x]) - f64::from(base[n + y * w + x]);
            let (dx, dy) = (x as f64 + 0.5 - mx, y as f64 + 0.5 - my);
            sxx += v * dx * dx;
            sxy += v * dx * dy;
            syy += v * dy * dy;
        }
    }
    let (sxx, sxy, syy) = (sxx / m, sxy / m, syy / m);
    let tr = sxx + syy;
    let d = (tr * tr / 4.0 - (sxx * syy - sxy * sxy)).max(0.0).sqrt();
    Blob {
        elongation: ((tr / 2.0 + d) / (tr / 2.0 - d)).sqrt(),
        angle_deg: 0.5 * (2.0 * sxy).atan2(sxx - syy).to_degrees(),
        energy,
    }
}

/// The light a disc held above the ground in `scene`, per channel (the input's, from its own pixels).
fn input_energy(scene: &Scene, d: (f64, f64, f64), reach: f64) -> [f64; 3] {
    let (w, h) = (scene.img.w, scene.img.h);
    blob(&scene.img.data, &scene.ground, w, h, d, reach).energy
}

#[test]
fn discs_come_out_q_times_taller_with_their_energy_kept() {
    let radii = [
        6.0, 9.0, 12.0, 8.0, 14.0, 10.0, 7.0, 11.0, 13.0, 9.0, 6.5, 12.0,
    ];
    // The shapes are measured on the noiseless scene (noise would be weighed into the second moments), the
    // energy on the same scene with sensor noise.
    let clean = night_scene_spaced(W, H, &radii, Noise::NONE, 31, 3.8);
    let noisy = clean.noisy(Noise::LOW, 32);
    for (axis, anamorphic) in [
        (Axis::X, 100.0),
        (Axis::X, 50.0),
        (Axis::Y, 100.0),
        (Axis::Y, 30.0),
    ] {
        let p = plan(LensKind::Default, 0.0, 0.0, anamorphic, W, H, axis);
        let q = p.q();
        let planes = planes_of(&p, &clean);
        let out = reshape_tile(
            &p,
            &planes,
            1,
            &clean.img.data,
            (W, H),
            PixelRect::new(0, 0, W, H),
        );
        for &d in &clean.discs {
            let b = blob(&out, &clean.ground, W, H, d, 3.4);
            // Taller across the anamorphic axis: the canvas horizontal is the buffer's x (axis X) or y (axis Y).
            let long_axis = if axis == Axis::X { 90.0 } else { 0.0 };
            let delta = ((b.angle_deg - long_axis + 90.0).rem_euclid(180.0) - 90.0).abs();
            assert!(
                delta < 2.0,
                "{axis:?} {anamorphic}: disc {d:?} major axis at {:.1} deg",
                b.angle_deg
            );
            assert!(
                (b.elongation / q - 1.0).abs() < 0.05,
                "{axis:?} {anamorphic}: disc of {} px is {:.3} times taller, wanted {q:.3}",
                d.2,
                b.elongation
            );
        }
        let planes = planes_of(&p, &noisy);
        let out = reshape_tile(
            &p,
            &planes,
            1,
            &noisy.img.data,
            (W, H),
            PixelRect::new(0, 0, W, H),
        );
        for &d in &noisy.discs {
            let b = blob(&out, &noisy.ground, W, H, d, 3.4);
            let before = input_energy(&noisy, d, 3.4);
            for c in 0..3 {
                assert!(
                    (b.energy[c] / before[c] - 1.0).abs() < 0.01,
                    "{axis:?} {anamorphic}: disc {d:?} channel {c} energy {:.4} -> {:.4}",
                    before[c],
                    b.energy[c]
                );
            }
        }
    }
}

#[test]
fn the_whole_frames_light_is_kept() {
    let radii = [6.0, 9.0, 12.0, 8.0, 14.0, 10.0, 7.0, 11.0];
    let scene = night_scene_spaced(W, H, &radii, Noise::LOW, 32, 2.5);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let planes = planes_of(&p, &scene);
    let out = reshape_tile(
        &p,
        &planes,
        1,
        &scene.img.data,
        (W, H),
        PixelRect::new(0, 0, W, H),
    );
    for c in 0..3 {
        let sum = |v: &[f32]| {
            v[c * W * H..(c + 1) * W * H]
                .iter()
                .map(|x| f64::from(*x))
                .sum::<f64>()
        };
        let (a, b) = (sum(&scene.img.data), sum(&out));
        assert!((b / a - 1.0).abs() < 0.005, "channel {c}: {a:.1} -> {b:.1}");
    }
}

#[test]
fn the_subject_and_its_hair_are_unchanged_and_the_background_is_reshaped() {
    let scene = subject_base().noisy(Noise::LOW, 41);
    for (kind, edges, anamorphic) in [
        (LensKind::Default, 0.0, 100.0),
        (LensKind::Petzval, 100.0, 60.0),
    ] {
        let p = plan(kind, 0.0, edges, anamorphic, W, H, Axis::X);
        let planes = planes_of(&p, &scene);
        let t = std::time::Instant::now();
        let out = reshape_tile(
            &p,
            &planes,
            1,
            &scene.img.data,
            (W, H),
            PixelRect::new(0, 0, W, H),
        );
        eprintln!("{kind:?}: L1 over {W}x{H} took {:?}", t.elapsed());
        let n = W * H;
        let (mut worst_subject, mut worst_strand) = (0.0f32, 0.0f32);
        let (mut changed, mut background) = (0usize, 0usize);
        for i in 0..n {
            let diff = (0..3)
                .map(|c| (out[c * n + i] - scene.img.data[c * n + i]).abs())
                .fold(0.0, f32::max);
            if scene.subject[i] {
                worst_subject = worst_subject.max(diff);
            } else if scene.strand[i] {
                worst_strand = worst_strand.max(diff);
            } else if scene.trusted[i] && scene.truth_r[i] > 0.0 {
                background += 1;
                changed += usize::from(diff > 1e-4);
            }
        }
        assert!(
            worst_subject <= 1e-6,
            "{kind:?}: the subject changed by {worst_subject}"
        );
        assert!(
            worst_strand <= 1e-6,
            "{kind:?}: the hair changed by {worst_strand}"
        );
        assert!(
            changed as f64 > 0.6 * background as f64,
            "{kind:?}: {changed} of {background} background pixels changed"
        );
        // The same on the reduced picture the viewport shows (k = 4): the subject's blocks do not change.
        let (small, sw, sh) = reduce(&scene.img.data, W, H, 4);
        let subject4 = reduce_mask(&scene.subject, W, H, 4);
        let out4 = reshape_tile(
            &p,
            &planes,
            4,
            &small,
            (sw, sh),
            PixelRect::new(0, 0, sw, sh),
        );
        let mut worst4 = 0.0f32;
        let mut changed4 = 0usize;
        for i in 0..sw * sh {
            let diff = (0..3)
                .map(|c| (out4[c * sw * sh + i] - small[c * sw * sh + i]).abs())
                .fold(0.0, f32::max);
            if subject4[i] {
                worst4 = worst4.max(diff);
            } else {
                changed4 += usize::from(diff > 1e-4);
            }
        }
        assert!(
            worst4 <= 1e-6,
            "{kind:?}: at k = 4 the subject changed by {worst4}"
        );
        assert!(
            changed4 > sw * sh / 4,
            "{kind:?}: k = 4 changed {changed4} pixels"
        );
    }
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let scene = subject_base().noisy(Noise::LOW, 42);
    for (kind, flares, edges, anamorphic) in [
        (LensKind::Default, 0.0, 0.0, 100.0),
        (LensKind::Petzval, 0.0, 100.0, 60.0),
        (LensKind::Vintage, 30.0, 50.0, 0.0),
    ] {
        let p = plan(kind, flares, edges, anamorphic, W, H, Axis::X);
        let planes = planes_of(&p, &scene);
        for k in [1usize, 4] {
            let (data, fw, fh) = if k == 1 {
                (scene.img.data.clone(), W, H)
            } else {
                reduce(&scene.img.data, W, H, k)
            };
            let whole_l1 = reshape_tile(
                &p,
                &planes,
                k,
                &data,
                (fw, fh),
                PixelRect::new(0, 0, fw, fh),
            );
            let whole_lens =
                apply_tile(&p, &planes, k, &data, fw, fh, PixelRect::new(0, 0, fw, fh));
            if edges > 0.0 || anamorphic > 0.0 {
                let changed = whole_l1.iter().zip(&data).filter(|(a, b)| a != b).count();
                assert!(changed > 0, "{kind:?} k {k}: L1 changed nothing");
            }
            for (nx, ny) in [(2usize, 2usize), (3, 3)] {
                let mut l1 = vec![0.0f32; 3 * fw * fh];
                let mut lens = vec![0.0f32; 3 * fw * fh];
                for rect in common::tiles(fw, fh, nx, ny) {
                    let a = reshape_tile(&p, &planes, k, &data, (fw, fh), rect);
                    let b = apply_tile(&p, &planes, k, &data, fw, fh, rect);
                    for c in 0..3 {
                        for y in 0..rect.h {
                            let dst = c * fw * fh + (rect.y + y) * fw + rect.x;
                            let src = c * rect.w * rect.h + y * rect.w;
                            l1[dst..dst + rect.w].copy_from_slice(&a[src..src + rect.w]);
                            lens[dst..dst + rect.w].copy_from_slice(&b[src..src + rect.w]);
                        }
                    }
                }
                assert!(
                    l1.iter()
                        .zip(&whole_l1)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{kind:?} k {k}: L1 in {nx}x{ny} tiles differs from the whole frame"
                );
                assert!(
                    lens.iter()
                        .zip(&whole_lens)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{kind:?} k {k}: the lens in {nx}x{ny} tiles differs from the whole frame"
                );
            }
        }
    }
}

#[test]
fn a_lens_plane_at_half_resolution_gives_the_same_guarantees() {
    // A 1100 x 733 frame has a lens plane of 550 x 367 (k_L = 2): the blur map is coarser than the frame, and
    // tiles at k = 1, 2 and 4 read it at three different ratios.
    let (fw, fh) = (1100usize, 733usize);
    let scene = subject_scene(fw, fh, [9.0, 16.0, 26.0], Noise::LOW, 5);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, fw, fh, Axis::X);
    assert_eq!(p.lens_plane_scale(), 2);
    let planes = planes_of(&p, &scene);
    let bp = planes.bokeh().unwrap();
    assert_eq!(bp.size(), (550, 367));
    assert!(bp.active());
    for k in [1usize, 2, 4] {
        let (data, w, h) = if k == 1 {
            (scene.img.data.clone(), fw, fh)
        } else {
            reduce(&scene.img.data, fw, fh, k)
        };
        let whole_out = reshape_tile(&p, &planes, k, &data, (w, h), PixelRect::new(0, 0, w, h));
        let subject = if k == 1 {
            scene.subject.clone()
        } else {
            reduce_mask(&scene.subject, fw, fh, k)
        };
        let mut worst = 0.0f32;
        for i in 0..w * h {
            if subject[i] {
                for c in 0..3 {
                    worst = worst.max((whole_out[c * w * h + i] - data[c * w * h + i]).abs());
                }
            }
        }
        assert!(worst <= 1e-6, "k {k}: the subject changed by {worst}");
        let changed = whole_out
            .iter()
            .zip(&data)
            .filter(|(a, b)| (**a - **b).abs() > 1e-4)
            .count();
        assert!(changed > w * h / 4, "k {k}: only {changed} samples changed");
        for (nx, ny) in [(2usize, 2usize), (3, 3)] {
            let mut assembled = vec![0.0f32; 3 * w * h];
            for rect in common::tiles(w, h, nx, ny) {
                let a = reshape_tile(&p, &planes, k, &data, (w, h), rect);
                for c in 0..3 {
                    for y in 0..rect.h {
                        let dst = c * w * h + (rect.y + y) * w + rect.x;
                        let src = c * rect.w * rect.h + y * rect.w;
                        assembled[dst..dst + rect.w].copy_from_slice(&a[src..src + rect.w]);
                    }
                }
            }
            assert!(
                assembled
                    .iter()
                    .zip(&whole_out)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "k {k}: {nx}x{ny} tiles differ from the whole frame"
            );
        }
    }
}

#[test]
fn two_runs_and_any_thread_count_give_the_same_bits() {
    let scene = subject_base().noisy(Noise::LOW, 43);
    let p = plan(LensKind::Petzval, 20.0, 80.0, 70.0, W, H, Axis::Y);
    let planes = planes_of(&p, &scene);
    let rect = PixelRect::new(100, 50, 600, 500);
    let a = apply_tile(&p, &planes, 1, &scene.img.data, W, H, rect);
    let b = apply_tile(&p, &planes, 1, &scene.img.data, W, H, rect);
    assert_eq!(a, b);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let c = pool.install(|| apply_tile(&p, &planes, 1, &scene.img.data, W, H, rect));
    assert!(a.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()));
    let pool3 = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let d = pool3.install(|| apply_tile(&p, &planes, 1, &scene.img.data, W, H, rect));
    assert!(a.iter().zip(&d).all(|(x, y)| x.to_bits() == y.to_bits()));
}

#[test]
fn a_scene_with_no_blur_is_left_as_it_is() {
    let scene = sharp_scene(W, H, Noise::MID, 3);
    for (kind, edges, anamorphic) in [
        (LensKind::Default, 0.0, 100.0),
        (LensKind::Petzval, 100.0, 0.0),
    ] {
        let p = plan(kind, 0.0, edges, anamorphic, W, H, Axis::X);
        let planes = planes_of(&p, &scene);
        assert_eq!(
            planes.report(),
            ym_lens::ReshapeReport {
                reshaped: false,
                reason: Some(ReshapeReason::LittleBlur)
            }
        );
        assert_eq!(reshape_halo(&p, &planes, 1), 0);
        let rect = PixelRect::new(30, 40, 200, 150);
        let out = reshape_tile(&p, &planes, 1, &scene.img.data, (W, H), rect);
        assert_eq!(
            out,
            common::crop(&scene.img.data, W, H, rect),
            "{kind:?}: L1 hands back its input bit for bit"
        );
    }
}

#[test]
fn swirl_stretches_balls_along_the_tangent_more_towards_the_edges() {
    let scene = night_scene_spaced(W, H, &[10.0; 10], Noise::LOW, 11, 3.0);
    let p = plan(LensKind::Petzval, 0.0, 100.0, 0.0, W, H, Axis::X);
    assert!(p.needs_blur_map());
    let planes = planes_of(&p, &scene);
    let out = reshape_tile(
        &p,
        &planes,
        1,
        &scene.img.data,
        (W, H),
        PixelRect::new(0, 0, W, H),
    );
    let hd = 600.97;
    let mut checked = 0;
    for &d in &scene.discs {
        let (ux, uy) = (d.0 - 500.0, d.1 - 333.5);
        let rho = ux.hypot(uy) / hd;
        let b = blob(&out, &scene.ground, W, H, d, 3.4);
        let before = input_energy(&scene, d, 3.4);
        for c in 0..3 {
            assert!(
                (b.energy[c] / before[c] - 1.0).abs() < 0.015,
                "disc {d:?} channel {c}"
            );
        }
        if rho < 0.15 {
            assert!(
                b.elongation < 1.05,
                "near the centre the swirl is nothing: {}",
                b.elongation
            );
            continue;
        }
        // The data's swirl at this field radius (read from the lens) and Edges 100.
        let lens = p.lens();
        let sigma = lens
            .field
            .swirl
            .iter()
            .zip(&lens.field.radii)
            .fold(0.0, |acc, (s, r)| if *r <= rho { *s } else { acc });
        assert!(
            b.elongation > 1.0 + 0.3 * sigma,
            "rho {rho:.2}: elongation {:.3}, swirl {sigma:.3}",
            b.elongation
        );
        // The long axis is the tangent: perpendicular to the radius, within 6 degrees.
        let radial = uy.atan2(ux).to_degrees();
        let off = ((b.angle_deg - radial + 90.0 + 90.0).rem_euclid(180.0) - 90.0).abs();
        assert!(
            off < 6.0,
            "rho {rho:.2}: major axis {:.1} deg, radial {radial:.1} deg",
            b.angle_deg
        );
        checked += 1;
    }
    assert!(checked >= 6, "{checked}");
}

#[test]
fn planes_built_without_the_map_are_refused_and_say_so() {
    let scene = subject_base();
    // Edges on a lens without swirl or cat's-eye need no blur map.
    let none = plan(LensKind::Default, 0.0, 100.0, 0.0, W, H, Axis::X);
    assert!(!none.needs_blur_map());
    let planes = planes_of(&none, scene);
    assert!(planes.bokeh().is_none());
    assert_eq!(
        planes.report(),
        ym_lens::ReshapeReport {
            reshaped: false,
            reason: Some(ReshapeReason::NotAsked)
        }
    );
    assert!(planes.serves(&none));
    // Anamorphic does; the planes' key does not change, but they do not serve the plan.
    let wants = plan(LensKind::Default, 0.0, 100.0, 50.0, W, H, Axis::X);
    assert_eq!(planes.key(), wants.planes_key());
    assert!(!planes.serves(&wants));
    let rect = PixelRect::new(0, 0, 100, 100);
    let buf = common::crop(&scene.img.data, W, H, rect);
    let input = ImageRef {
        width: 100,
        height: 100,
        x0: 0,
        y0: 0,
        frame_width: W,
        frame_height: H,
        data: &buf,
    };
    assert!(matches!(
        reshape(&wants, &planes, 1, &input, rect),
        Err(LensError::Frame(_))
    ));
    assert!(matches!(
        apply(&wants, &planes, 1, &input, rect),
        Err(LensError::Frame(_))
    ));
    // With the map they serve it.
    let built = planes_of(&wants, scene);
    assert!(built.serves(&wants));
    assert!(built.serves(&none) || !built.serves(&none)); // either way, no panic
}

#[test]
fn an_input_too_small_for_the_halo_is_refused() {
    let scene = subject_base().noisy(Noise::LOW, 44);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let planes = planes_of(&p, &scene);
    let halo = reshape_halo(&p, &planes, 1);
    assert!(halo > 10 && halo <= 40, "{halo}");
    assert!(halo_px(&p, &planes, 1) >= halo);
    let rect = PixelRect::new(300, 200, 100, 100);
    let need = rect.grown(halo - 1, W, H);
    let buf = common::crop(&scene.img.data, W, H, need);
    let input = ImageRef {
        width: need.w,
        height: need.h,
        x0: need.x,
        y0: need.y,
        frame_width: W,
        frame_height: H,
        data: &buf,
    };
    assert!(matches!(
        reshape(&p, &planes, 1, &input, rect),
        Err(LensError::InputTooSmall { .. })
    ));
    // At the frame's edge the halo is clamped to the frame.
    let corner = PixelRect::new(0, 0, 80, 80);
    let out = reshape_tile(&p, &planes, 1, &scene.img.data, (W, H), corner);
    assert_eq!(out.len(), 3 * 80 * 80);
    let _ = erode;
}
