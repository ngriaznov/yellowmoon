//! L1, the bokeh reshaping, on synthetic scenes (`docs/21` sections 4 and 14).
//!
//! `reshape` is L1 alone (what the field optics would read); `apply` is the whole lens. Frames are 1000 x 667
//! (the lens plane is the frame, `k_L` = 1) unless a test says otherwise.

#![allow(clippy::needless_range_loop)]

mod common;
mod scenes;

use std::sync::OnceLock;

use common::{apply_tile, plan, whole};
use scenes::{
    Noise, Scene, erode, night_scene_spaced, sharp_scene, soft_night_scene, subject_scene,
};
use ym_lens::{
    Axis, BlurMapInput, ImageRef, LensError, LensKind, LensPlan, LensPlanes, PixelRect,
    ReshapeReason, apply, halo_px, reshape, reshape_halo,
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
fn discs_come_out_q_times_taller_with_their_surface_brightness_kept() {
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
            // The ball's own pixels are stretched, so it keeps its surface brightness and carries `q` times the light
            // over its `q` times larger area.
            for c in 0..3 {
                assert!(
                    (b.energy[c] / (q * before[c]) - 1.0).abs() < 0.03,
                    "{axis:?} {anamorphic}: disc {d:?} channel {c} energy {:.4} -> {:.4}, wanted {q:.2} times",
                    before[c],
                    b.energy[c]
                );
            }
        }
    }
}

#[test]
fn soft_edged_and_clipped_balls_come_out_twice_as_tall_at_anamorphic_100() {
    // The strongest cue of an anamorphic lens: every out-of-focus highlight is an oval twice as tall as wide, soft
    // or sharp at the rim, clipped or not. (radius, rim width over radius, level)
    let balls = [
        (7.0, 0.0, 0.5),
        (9.0, 0.3, 0.8),
        (11.0, 0.6, 1.6),
        (13.0, 0.5, 1.6),
        (8.0, 0.4, 1.2),
        (10.0, 0.0, 0.6),
        (12.0, 0.4, 0.45),
        (6.0, 0.3, 1.6),
    ];
    let scene = soft_night_scene(W, H, &balls, Noise::NONE, 21, 3.8);
    for axis in [Axis::X, Axis::Y] {
        let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, axis);
        assert!((p.q() - 2.0).abs() < 1e-12);
        let planes = planes_of(&p, &scene);
        assert!(planes.bokeh().unwrap().active());
        let out = reshape_tile(
            &p,
            &planes,
            1,
            &scene.img.data,
            (W, H),
            PixelRect::new(0, 0, W, H),
        );
        for &d in &scene.discs {
            let b = blob(&out, &scene.ground, W, H, d, 3.6);
            let long_axis = if axis == Axis::X { 90.0 } else { 0.0 };
            let delta = ((b.angle_deg - long_axis + 90.0).rem_euclid(180.0) - 90.0).abs();
            assert!(
                delta < 3.0,
                "{axis:?}: disc {d:?} major axis at {:.1} deg",
                b.angle_deg
            );
            assert!(
                b.elongation >= 1.9,
                "{axis:?}: the ball of {} px is only {:.2} times taller",
                d.2,
                b.elongation
            );
        }
    }
}

#[test]
fn tiles_equal_the_whole_frame_with_soft_balls() {
    let balls = [
        (7.0, 0.0, 0.5),
        (9.0, 0.3, 0.8),
        (11.0, 0.6, 1.6),
        (13.0, 0.5, 1.6),
        (8.0, 0.4, 1.2),
        (10.0, 0.0, 0.6),
        (12.0, 0.4, 0.45),
        (6.0, 0.3, 1.6),
        (9.0, 0.2, 0.9),
        (7.5, 0.5, 1.4),
    ];
    let scene = soft_night_scene(W, H, &balls, Noise::LOW, 22, 2.6);
    let p = plan(LensKind::Petzval, 0.0, 100.0, 100.0, W, H, Axis::X);
    let planes = planes_of(&p, &scene);
    assert!(!planes.bokeh().unwrap().balls().is_empty());
    for k in [1usize, 4] {
        let (data, fw, fh) = if k == 1 {
            (scene.img.data.clone(), W, H)
        } else {
            reduce(&scene.img.data, W, H, k)
        };
        let whole = reshape_tile(
            &p,
            &planes,
            k,
            &data,
            (fw, fh),
            PixelRect::new(0, 0, fw, fh),
        );
        assert!(
            whole.iter().zip(&data).any(|(a, b)| a != b),
            "k {k}: nothing changed"
        );
        for (nx, ny) in [(2usize, 2usize), (3, 3), (5, 4)] {
            let mut assembled = vec![0.0f32; 3 * fw * fh];
            for rect in common::tiles(fw, fh, nx, ny) {
                let a = reshape_tile(&p, &planes, k, &data, (fw, fh), rect);
                for c in 0..3 {
                    for y in 0..rect.h {
                        let dst = c * fw * fh + (rect.y + y) * fw + rect.x;
                        let src = c * rect.w * rect.h + y * rect.w;
                        assembled[dst..dst + rect.w].copy_from_slice(&a[src..src + rect.w]);
                    }
                }
            }
            assert!(
                assembled
                    .iter()
                    .zip(&whole)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "k {k}: {nx}x{ny} tiles differ from the whole frame"
            );
        }
    }
}

#[test]
fn the_whole_frames_light_grows_by_the_stretched_balls_extra_light_only() {
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
    // The gathers keep the light of the ground; the balls keep their surface brightness over `q` times the area, so
    // the picture gains `q - 1` times the light the balls held, and nothing else.
    let q = p.q();
    for c in 0..3 {
        let sum = |v: &[f32]| {
            v[c * W * H..(c + 1) * W * H]
                .iter()
                .map(|x| f64::from(*x))
                .sum::<f64>()
        };
        let balls: f64 = scene
            .discs
            .iter()
            .map(|&d| input_energy(&scene, d, 3.4)[c])
            .sum();
        let (a, b) = (sum(&scene.img.data), sum(&out));
        let want = a + (q - 1.0) * balls;
        assert!(
            (b / want - 1.0).abs() < 0.01,
            "channel {c}: {a:.1} -> {b:.1}, wanted {want:.1}"
        );
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
        (LensKind::Vintage, 30.0, 50.0, 40.0),
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
            // L1 acts with Anamorphic alone; Edges never reshapes bokeh.
            if anamorphic > 0.0 {
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
        (LensKind::Petzval, 0.0, 100.0),
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

/// The stretch of a ball at `(cx, cy)` of the 1000 x 667 frame, as the shapes make it (`docs/21` section 4, L1, *As
/// built (swirl with Anamorphic)*): the long axis is the sum of the attachment's stretch across the anamorphic axis
/// (vertical here), `(q - 1) r`, and the tangential one, `(q_t - 1) r` with `q_t = sqrt(1 + 4 (A sigma)^2 / 3)`, the
/// sign of the second chosen to add. Returns the aspect `f` (the patch's long over short axis) and the long axis's
/// angle in degrees (image coordinates, y down), from `q`, the swirl `sigma` and the amount `A`.
fn vector_sum_stretch(cx: f64, cy: f64, q: f64, sigma: f64, amount: f64) -> (f64, f64) {
    let (ux, uy) = (cx - 500.0, cy - 333.5);
    let dist = ux.hypot(uy);
    let (tx, ty) = (-uy / dist, ux / dist);
    let q_t = (1.0 + 4.0 / 3.0 * (amount * sigma).powi(2)).sqrt();
    let v = q - 1.0;
    let sgn = if v * ty < 0.0 { -1.0 } else { 1.0 };
    let (sx, sy) = (sgn * (q_t - 1.0) * tx, v + sgn * (q_t - 1.0) * ty);
    (1.0 + sx.hypot(sy), sy.atan2(sx).to_degrees())
}

/// The difference of two axis angles, in degrees, modulo 180 (0 to 90).
fn axis_gap_deg(a: f64, b: f64) -> f64 {
    ((a - b + 90.0).rem_euclid(180.0) - 90.0).abs()
}

/// A plan of the synthetic Petzval with `swirl`, `astigmatism` and `cats_eye` as its three tables, on an attachment
/// whose `q_max` is `q_max`, at the given Edges and Anamorphic.
fn tables_plan(
    (swirl, astigmatism, cats_eye): (&[f64; 9], &[f64; 9], &[f64; 9]),
    q_max: f64,
    (edges, anamorphic): (f64, f64),
) -> LensPlan {
    use serde_json::json;
    let dir = tempfile::tempdir().expect("temp dir");
    ym_lens::test_support::write_synthetic_lenses_with(
        dir.path(),
        &|lens| {
            lens["field"]["swirl"] = json!(swirl);
            lens["field"]["astigmatism"] = json!(astigmatism);
            lens["field"]["cats_eye"] = json!(cats_eye);
        },
        &|att| att["q_max"] = json!(q_max),
    )
    .expect("write");
    let lib = ym_lens::LensLibrary::from_dir(dir.path()).expect("loads");
    let amounts = ym_lens::LensAmounts {
        kind: LensKind::Petzval,
        flares: 0.0,
        edges,
        anamorphic,
    };
    LensPlan::new(
        &lib,
        &amounts,
        common::frame(W, H, Axis::X),
        common::camera(),
    )
    .expect("plan")
}

/// Where the balls of the swirl tests stand, in the 1000 x 667 frame: the right-hand side, where the tangent is the
/// axis and the two stretches add, the top, where they are at right angles, and the four corners. Far enough
/// apart that a ball's patch (3.9 to 1 at most, 35 px) holds no other ball, near enough to the corners to be at
/// 0.85 of the field and still whole inside the frame.
const SWIRL_SPOTS: [(f64, f64); 13] = [
    (650.0, 333.5),
    (750.0, 333.5),
    (850.0, 333.5),
    (950.0, 333.5),
    (500.0, 60.0),
    (300.0, 90.0),
    (150.0, 560.0),
    (780.0, 100.0),
    (780.0, 580.0),
    (940.0, 600.0),
    (60.0, 70.0),
    (940.0, 70.0),
    (60.0, 600.0),
];

/// The shipped Petzval's swirl (`data/lenses/petzval/lens.json`, pinned by the test of the tables).
const PETZVAL_SWIRL: [f64; 9] = [0.0, 0.0, 0.0, 0.6, 1.5, 1.85, 2.1, 2.3, 2.45];

#[test]
fn swirl_stretches_balls_along_the_tangent_mixed_with_the_attachments_stretch_towards_the_edges() {
    // The swirl is Anamorphic's: at Anamorphic 100 a ball is `(q - 1) r` taller across the axis and `(q_t - 1) r`
    // longer along the tangent, the long axis the sum of the two (vertical at the sides, where they add, tilted
    // between). The shipped Petzval's swirl, 2 to 1 at half the field and 3 to 1 at the corner for the tangent
    // alone, on balls of 7 px with no cat's-eye (its mask would thin them: tested below).
    let p = tables_plan((&PETZVAL_SWIRL, &[0.0; 9], &[0.0; 9]), 2.0, (0.0, 100.0));
    assert!(p.needs_blur_map());
    let q = p.q();
    assert!((q - 2.0).abs() < 1e-12);
    let balls: Vec<_> = SWIRL_SPOTS
        .iter()
        .map(|&(x, y)| (x, y, 7.0, 0.15, 1.5))
        .collect();
    let scene = balls_on_ground(&balls, Noise::LOW, 5);
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
    let (mut checked, mut corner_three) = (0, 0);
    for &d in &scene.discs {
        let rho = (d.0 - 500.0).hypot(d.1 - 333.5) / hd;
        let sigma = table_at(&p.lens().field.radii, &p.lens().field.swirl, rho);
        let (wanted, angle) = vector_sum_stretch(d.0, d.1, q, sigma, 1.0);
        // The window holds the whole patch (`1.3` of the ball's radius is its take-out disc) and no other ball.
        let b = blob(&out, &scene.ground, W, H, d, 1.3 * wanted + 1.5);
        let before = input_energy(&scene, d, 3.4);
        eprintln!(
            "rho {rho:.2} swirl {sigma:.2}: elongation {:.3}, vector sum {wanted:.3}, axis {:.1} deg (wanted {angle:.1})",
            b.elongation, b.angle_deg
        );
        // The ball's own pixels are stretched: its light grows with its length.
        for c in 0..3 {
            assert!(
                (b.energy[c] / before[c] / b.elongation - 1.0).abs() < 0.05,
                "disc {d:?} channel {c}: light {:.3} -> {:.3}, elongation {:.3}",
                before[c],
                b.energy[c],
                b.elongation
            );
        }
        assert!(
            (b.elongation / wanted - 1.0).abs() < 0.04,
            "rho {rho:.2}: elongation {:.3}, wanted {wanted:.3} (swirl {sigma:.3})",
            b.elongation
        );
        assert!(
            axis_gap_deg(b.angle_deg, angle) < 8.0,
            "rho {rho:.2}: major axis {:.1} deg, wanted {angle:.1} deg",
            b.angle_deg
        );
        if sigma > 0.5 {
            checked += 1;
        }
        // Towards the corners the balls are at least 3 to 1 along the vector sum.
        if rho > 0.8 && wanted >= 3.0 {
            corner_three += 1;
        }
    }
    assert!(checked >= 5, "{checked}");
    assert_eq!(corner_three, 4, "the four balls near the corners");
}

#[test]
fn the_cats_eye_cuts_the_ovals_at_anamorphic_and_the_ovals_stay_as_tall() {
    // The same balls with the shipped Petzval's cat's-eye: the second circle takes a slice off the oval across its
    // short axis (the radial direction), so the oval is narrower than without, and still along the same axis. The
    // mask works on the take-out disc (1.3 radii here), each copy shifted by half the offset times that: it cuts
    // the glow around the ball first and the ball itself only where the offset is 0.45 radii or more, so the oval
    // shows it near the corners (0.7 to 0.8 radii at 0.85 of the field) and not halfway out (0.3).
    let cats = [0.0, 0.0, 0.0, 0.12, 0.3, 0.48, 0.62, 0.72, 0.8];
    let with = tables_plan((&PETZVAL_SWIRL, &[0.0; 9], &cats), 2.0, (0.0, 100.0));
    let without = tables_plan((&PETZVAL_SWIRL, &[0.0; 9], &[0.0; 9]), 2.0, (0.0, 100.0));
    let balls: Vec<_> = SWIRL_SPOTS
        .iter()
        .map(|&(x, y)| (x, y, 7.0, 0.15, 1.5))
        .collect();
    let scene = balls_on_ground(&balls, Noise::LOW, 5);
    let run = |p: &LensPlan| {
        let planes = planes_of(p, &scene);
        reshape_tile(
            p,
            &planes,
            1,
            &scene.img.data,
            (W, H),
            PixelRect::new(0, 0, W, H),
        )
    };
    let (a, b) = (run(&with), run(&without));
    let mut cut_balls = 0;
    for &d in &scene.discs {
        let rho = (d.0 - 500.0).hypot(d.1 - 333.5) / 600.97;
        let (cut, round) = (
            blob(&a, &scene.ground, W, H, d, 6.0),
            blob(&b, &scene.ground, W, H, d, 6.0),
        );
        if rho < 0.65 {
            // The offset is under 0.45 radii here: the glow is cut, not the ball.
            assert!(
                (cut.elongation / round.elongation - 1.0).abs() < 0.03,
                "ball {d:?}: {:.3} with the cat's-eye, {:.3} without",
                cut.elongation,
                round.elongation
            );
        } else if rho > 0.8 {
            assert!(
                cut.elongation > 1.1 * round.elongation,
                "ball {d:?} (rho {rho:.2}): {:.3} with the cat's-eye, {:.3} without",
                cut.elongation,
                round.elongation
            );
            assert!(axis_gap_deg(cut.angle_deg, round.angle_deg) < 8.0);
            assert!(
                cut.energy[1] < round.energy[1],
                "the slice takes light off the oval"
            );
            cut_balls += 1;
        }
    }
    assert_eq!(cut_balls, 4, "the four balls near the corners");
}

#[test]
fn the_shipped_petzvals_swirl_is_two_to_one_halfway_and_three_to_one_at_the_corner_and_the_vintages_is_a_third()
 {
    // `data/lenses/README.md`, *What is traced and what is tuned*, and `docs/21` L1, *As built (swirl with
    // Anamorphic)*: at Anamorphic 100 the Petzval's tangential line is `k sigma b` long, `sigma` blur radii (the
    // half-length of the box line with the spread of the gather), so a disc becomes an oval `sqrt(1 + 4 sigma^2 / 3)`
    // times as long as wide: 2 to 1 at half the field (sigma 1.5), 3 to 1 at the corner (sigma 2.45). It has no
    // astigmatism (the absolute part is 0), a cat's-eye of 0.8 radii at the corner, and the field optics' own spread
    // (`psf_half_width`) is about 0.3 to 0.45 of what it was outside the centre, so that Edges softens less. The
    // Vintage's swirl is a third of it.
    let root = common::repo_root().join("data");
    if !root.join("lenses/index.json").exists() {
        eprintln!("skipped: no lens data");
        return;
    }
    let lib = ym_lens::LensLibrary::load(&root).expect("data/lenses loads");
    let petzval = &lib.get(LensKind::Petzval).expect("petzval").field;
    let vintage = &lib.get(LensKind::Vintage).expect("vintage").field;
    assert_eq!(
        petzval.radii,
        [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    );
    let sigma = &petzval.swirl;
    let aspect = |s: f64| (1.0 + 4.0 / 3.0 * s * s).sqrt();
    assert_eq!(sigma[..3], [0.0; 3], "nothing inside rho 0.25");
    assert!(sigma[3] > 0.0 && sigma[3] < 1.0, "rho 0.375: {}", sigma[3]);
    assert!(
        (aspect(sigma[4]) - 2.0).abs() < 0.01,
        "rho 0.5: {}",
        sigma[4]
    );
    assert!((aspect(sigma[8]) - 3.0).abs() < 0.01, "rho 1: {}", sigma[8]);
    assert!((sigma[8] - 2.45).abs() < 1e-9, "rho 1: {}", sigma[8]);
    for i in 1..9 {
        assert!(sigma[i] >= sigma[i - 1], "monotone at {i}");
    }
    // No absolute part: a small light of the outer field is an oval of the same aspect as a large one.
    assert_eq!(petzval.astigmatism, [0.0; 9]);
    assert_eq!(vintage.astigmatism, [0.0; 9]);
    // The cat's-eye grows outwards to 0.8 radii at the corner.
    let cat = &petzval.cats_eye;
    assert_eq!(cat[..3], [0.0; 3]);
    assert!((cat[8] - 0.8).abs() < 1e-9, "{}", cat[8]);
    for i in 1..9 {
        assert!(cat[i] >= cat[i - 1], "monotone at {i}");
    }
    // The isotropic spread of L2, now about 0.3 (corner) to 0.45 (half the field) of the old table's.
    let old = [
        0.0015, 0.0019, 0.0028, 0.0041, 0.0053, 0.0067, 0.0086, 0.0113, 0.012,
    ];
    assert_eq!(petzval.psf_half_width[0], old[0], "the centre is as it was");
    for i in 4..9 {
        let ratio = petzval.psf_half_width[i] / old[i];
        assert!(
            (0.29..=0.46).contains(&ratio),
            "rho {}: {} of the old width",
            petzval.radii[i],
            ratio
        );
    }
    // The Vintage keeps a third of the swirl, and its own (traced) cat's-eye.
    for (v, p) in vintage.swirl.iter().zip(sigma) {
        assert!((v - p / 3.0).abs() < 1e-5, "{v} against a third of {p}");
    }
    assert!((vintage.cats_eye[8] - 0.721_268).abs() < 1e-9);
    // The traced lenses keep none: their spread functions already hold their astigmatism (L2).
    for kind in [
        LensKind::Default,
        LensKind::Classic,
        LensKind::Triplet,
        LensKind::Sonnar,
        LensKind::Detuned,
    ] {
        let lens = lib.get(kind).expect("lens");
        assert_eq!(lens.astigmatism_max(), 0.0, "{kind:?}");
        assert!(lens.info.version >= 2, "{kind:?}: data version");
    }
    assert!(lib.get(LensKind::Petzval).unwrap().info.version >= 3);
    assert!(lib.get(LensKind::Vintage).unwrap().info.version >= 4);
}

#[test]
fn edges_alone_reshape_no_bokeh_on_any_lens_and_the_swirl_follows_anamorphic() {
    // The swirl and the cat's-eye are the Anamorphic slider's. At Anamorphic 0 no lens, whatever its tables, draws a
    // tangential stretch or a cat's-eye: L1 needs no blur map, hands its input back bit for bit and has no pass;
    // with Anamorphic above 0 the pass takes its swirl amount from Anamorphic alone, and Edges changes nothing in it.
    let root = common::repo_root().join("data");
    if !root.join("lenses/index.json").exists() {
        eprintln!("skipped: no lens data");
        return;
    }
    let lib = ym_lens::LensLibrary::load(&root).expect("data/lenses loads");
    let make = |kind: LensKind, edges: f64, anamorphic: f64| {
        let amounts = ym_lens::LensAmounts {
            kind,
            flares: 0.0,
            edges,
            anamorphic,
        };
        LensPlan::new(
            &lib,
            &amounts,
            common::frame(W, H, Axis::X),
            common::camera(),
        )
        .expect("plan")
    };
    let scene = night_scene_spaced(W, H, &[10.0; 10], Noise::LOW, 11, 3.0);
    let rect = PixelRect::new(0, 0, W, H);
    for kind in LensKind::ALL {
        let p = make(kind, 100.0, 0.0);
        assert!(!p.needs_blur_map(), "{kind:?}: Edges 100 needs no blur map");
        assert_eq!(p.anamorphic(), 0.0);
        let planes = planes_of(&p, &scene);
        assert!(planes.bokeh().is_none(), "{kind:?}");
        assert_eq!(reshape_halo(&p, &planes, 1), 0, "{kind:?}");
        assert!(p.bokeh_pass(&planes, 1, W, H).is_none(), "{kind:?}");
        let out = reshape_tile(&p, &planes, 1, &scene.img.data, (W, H), rect);
        assert_eq!(
            out,
            common::crop(&scene.img.data, W, H, rect),
            "{kind:?}: L1 hands back its input bit for bit at Anamorphic 0"
        );
    }
    // Anamorphic 100: the Petzval's pass swirls by the whole of its tables, and Edges is not in it.
    let a100 = make(LensKind::Petzval, 0.0, 100.0);
    let planes = planes_of(&a100, &scene);
    let pass = a100.bokeh_pass(&planes, 1, W, H).expect("a pass");
    assert!(pass.tangential && pass.vertical);
    assert_eq!(pass.swirl, 1.0);
    let with_edges = make(LensKind::Petzval, 100.0, 100.0);
    let planes_e = planes_of(&with_edges, &scene);
    assert_eq!(
        with_edges.bokeh_pass(&planes_e, 1, W, H),
        Some(pass),
        "Edges is not part of L1's pass"
    );
    assert_eq!(a100.bokeh_shape_key(), with_edges.bokeh_shape_key());
    let half = make(LensKind::Petzval, 0.0, 50.0);
    let planes_h = planes_of(&half, &scene);
    assert_eq!(
        half.bokeh_pass(&planes_h, 1, W, H).expect("pass").swirl,
        0.5
    );
    // L1's picture is the same bit for bit at Edges 0 and Edges 100.
    let (lo, hi) = (
        reshape_tile(&a100, &planes, 1, &scene.img.data, (W, H), rect),
        reshape_tile(&with_edges, &planes_e, 1, &scene.img.data, (W, H), rect),
    );
    assert!(
        lo.iter().zip(&hi).all(|(a, b)| a.to_bits() == b.to_bits()),
        "Edges changes nothing in L1"
    );
    assert_ne!(
        lo,
        common::crop(&scene.img.data, W, H, rect),
        "and L1 acted"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// The field swirl (docs/21 section 4, L1, "As built (field swirl)"): the tangential half-length is
// `A (k sigma(rho) b + a(rho) H)`, `A` = Anamorphic / 100: the swirl grows with the blur, the astigmatism does not.
// ---------------------------------------------------------------------------------------------------------------

/// A plan of a synthetic Petzval whose swirl is `swirl rho^2` and whose astigmatism is `astigmatism rho^2`
/// half-diagonals, with no cat's-eye, at Anamorphic `anamorphic` on an attachment that stretches next to nothing
/// (`q_max` 1.000001: a vertical line of 0.02 px at a blur of 12 px, under the gathers' minimum), so that the
/// tangential gather is seen alone.
fn field_swirl_plan(swirl: f64, astigmatism: f64, anamorphic: f64) -> LensPlan {
    let rho2: Vec<f64> = (0..9).map(|i| (i as f64 / 8.0).powi(2)).collect();
    let table = |scale: f64| -> [f64; 9] { std::array::from_fn(|i| scale * rho2[i]) };
    tables_plan(
        (&table(swirl), &table(astigmatism), &[0.0; 9]),
        1.000_001,
        (0.0, anamorphic),
    )
}

/// A table read between the lens's nine radii, as the gathers read it.
fn table_at(radii: &[f64], values: &[f64], rho: f64) -> f64 {
    let at = radii.iter().position(|r| *r >= rho).unwrap_or(8).max(1);
    let (r0, r1) = (radii[at - 1], radii[at]);
    values[at - 1] + ((rho - r0) / (r1 - r0)).clamp(0.0, 1.0) * (values[at] - values[at - 1])
}

#[test]
fn astigmatism_makes_small_balls_of_the_outer_field_tangential_lemons_by_the_formula() {
    // `a(rho) = 0.0283 rho^2`, no swirl: a ball of radius r at rho 0.9 is blurred along its tangent by a box line
    // of half-length `A a H / k` = 8.6 px, whatever its own blur, so a ball of 8 px becomes a lemon 1.7 times as long
    // as wide, one of 10 px 1.5 times, and the same balls near the centre stay round. (A lens with astigmatism in
    // its table, which no shipped lens has now, shows it with Anamorphic.)
    let p = field_swirl_plan(0.0, 0.0283, 100.0);
    assert!(p.needs_blur_map(), "Anamorphic alone asks for the blur map");
    let hd = 600.97;
    // (x, y, r): four balls at rho 0.9 in the corners, four inside.
    let balls = [
        (950.0, 633.5, 8.0),
        (50.0, 33.5, 10.0),
        (950.0, 33.5, 9.0),
        (50.0, 633.5, 7.0),
        (830.0, 333.5, 8.0),
        (170.0, 333.5, 8.0),
        (500.0, 93.5, 8.0),
        (500.0, 573.5, 8.0),
    ];
    let balls: Vec<_> = balls
        .iter()
        .map(|&(x, y, r)| (x, y, r, 0.15, 1.5))
        .collect();
    let scene = balls_on_ground(&balls, Noise::LOW, 5);
    let planes = planes_of(&p, &scene);
    let out = reshape_tile(
        &p,
        &planes,
        1,
        &scene.img.data,
        (W, H),
        PixelRect::new(0, 0, W, H),
    );
    let lens = p.lens();
    let mut lemons = 0;
    for &d in &scene.discs {
        let (ux, uy) = (d.0 - 500.0, d.1 - 333.5);
        let rho = ux.hypot(uy) / hd;
        let b = blob(&out, &scene.ground, W, H, d, 3.6);
        let a = table_at(&lens.field.radii, &lens.field.astigmatism, rho);
        // The ball's radius as the detector measured it (the region above 40 % of the way to its plateau), which
        // is what the stretch is made from.
        let ball = planes
            .bokeh()
            .expect("map")
            .balls()
            .iter()
            .find(|k| (k.x * W as f64 - d.0).hypot(k.y * H as f64 - d.1) < 3.0)
            .expect("the ball is found");
        let r = ball.radius * hd;
        assert!((r / d.2 - 1.0).abs() < 0.12, "radius {r} against {}", d.2);
        let wanted = (1.0 + 4.0 / 3.0 * (a * hd / (1.527_525 * r)).powi(2)).sqrt();
        eprintln!(
            "rho {rho:.2}, r {r:.2}: elongation {:.3}, formula {wanted:.3}",
            b.elongation
        );
        assert!(
            (b.elongation / wanted - 1.0).abs() < 0.03,
            "rho {rho:.2}, r {r:.2}: elongation {:.3}, wanted {wanted:.3} (a {a:.4})",
            b.elongation
        );
        if wanted > 1.3 {
            // The long axis is the tangent: perpendicular to the radius, within 6 degrees.
            let radial = uy.atan2(ux).to_degrees();
            let off = ((b.angle_deg - radial + 90.0 + 90.0).rem_euclid(180.0) - 90.0).abs();
            assert!(
                off < 6.0,
                "rho {rho:.2}: major axis {:.1} deg, radial {radial:.1} deg",
                b.angle_deg
            );
            lemons += 1;
        } else if rho < 0.45 {
            assert!(
                b.elongation < 1.05,
                "near the centre the balls stay round: rho {rho:.2}: {}",
                b.elongation
            );
        }
    }
    assert_eq!(lemons, 4, "the four balls at rho 0.9");
}

/// The second moments, along the unit vector `u` and across it, of the light the green channel has above `base`
/// within `half` pixels of `(cx, cy)`, and the light itself.
fn response_moments(
    out: &[f32],
    base: f32,
    (cx, cy): (f64, f64),
    half: i64,
    u: (f64, f64),
) -> (f64, f64, f64) {
    let n = W * H;
    let (mut m, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
    for y in cy as i64 - half..=cy as i64 + half {
        for x in cx as i64 - half..=cx as i64 + half {
            let v = f64::from(out[n + y as usize * W + x as usize] - base);
            m += v;
            mx += v * (x as f64 + 0.5);
            my += v * (y as f64 + 0.5);
        }
    }
    let (mx, my) = (mx / m, my / m);
    let (mut along, mut across) = (0.0f64, 0.0f64);
    for y in cy as i64 - half..=cy as i64 + half {
        for x in cx as i64 - half..=cx as i64 + half {
            let v = f64::from(out[n + y as usize * W + x as usize] - base);
            let (dx, dy) = (x as f64 + 0.5 - mx, y as f64 + 0.5 - my);
            along += v * (dx * u.0 + dy * u.1).powi(2);
            across += v * (-dx * u.1 + dy * u.0).powi(2);
        }
    }
    (along / m, across / m, m)
}

#[test]
fn the_gather_smears_a_blurred_field_along_the_tangent_by_the_swirl_plus_the_astigmatism() {
    // A smooth bump (a Gaussian of 3 px, too faint to be a ball) on a field the map calls blurred everywhere: the
    // response of the tangential gather. A biweight line of half-length L adds a variance of L^2 / 7 along the
    // tangent and nothing across it, with L = e (k sigma b + a H) (`tangent_half_length`).
    let at: (f64, f64) = (948.0, 628.0);
    let hd = 600.97;
    let rho = (at.0 - 500.0).hypot(at.1 - 333.5) / hd;
    let (rx, ry) = ((at.0 - 500.0) / (rho * hd), (at.1 - 333.5) / (rho * hd));
    let tangent = (-ry, rx);
    let base = 0.5f32;
    let n = W * H;
    let mut data = vec![base; 3 * n];
    for c in 0..3 {
        for y in 0..H {
            for x in 0..W {
                let d2 = (x as f64 + 0.5 - at.0).powi(2) + (y as f64 + 0.5 - at.1).powi(2);
                data[c * n + y * W + x] += (0.12 * (-d2 / (2.0 * 9.0)).exp()) as f32;
            }
        }
    }
    let spread = |swirl: f64, astigmatism: f64, anamorphic: f64, blur_px: f64| -> (f64, f64) {
        let p = field_swirl_plan(swirl, astigmatism, anamorphic);
        let planes = planes_all_blurred(&p, &data, blur_px);
        assert!(planes.bokeh().is_some_and(|b| b.active()));
        let out = reshape_tile(&p, &planes, 1, &data, (W, H), PixelRect::new(0, 0, W, H));
        let lens = p.lens();
        let sigma = table_at(&lens.field.radii, &lens.field.swirl, rho);
        let a = table_at(&lens.field.radii, &lens.field.astigmatism, rho);
        let len = anamorphic / 100.0 * (1.527_525 * sigma * blur_px + a * hd);
        let (along, across, _) = response_moments(&out, base, at, 70, tangent);
        // The bump's own 9 px^2 is in both; the line adds to the first only.
        ((along - across) / (len * len / 7.0), across)
    };
    // The astigmatism alone: whatever the blur, the line is the same.
    let (r12, across12) = spread(0.0, 0.04, 100.0, 12.0);
    eprintln!(
        "astigmatism alone, blur 12 px: variance along the tangent over L^2/7 {r12:.4}, across {across12:.3}"
    );
    let (r20, across20) = spread(0.0, 0.04, 100.0, 16.0);
    assert!(
        (r12 - 1.0).abs() < 0.03,
        "astigmatism at a blur of 12 px: {r12:.4} of L^2/7"
    );
    assert!(
        (r20 - 1.0).abs() < 0.03,
        "astigmatism at a blur of 16 px: {r20:.4} of L^2/7"
    );
    assert!(
        (across12 - 9.0).abs() < 0.5 && (across20 - 9.0).abs() < 0.5,
        "across: {across12} {across20}"
    );
    // The swirl alone grows with the blur; the two together add; Anamorphic scales both.
    let (s12, _) = spread(1.2, 0.0, 100.0, 12.0);
    let (s20, _) = spread(1.2, 0.0, 100.0, 16.0);
    assert!(
        (s12 - 1.0).abs() < 0.03 && (s20 - 1.0).abs() < 0.03,
        "swirl: {s12:.4} {s20:.4}"
    );
    let (both, _) = spread(1.2, 0.04, 100.0, 12.0);
    assert!((both - 1.0).abs() < 0.03, "both: {both:.4}");
    let (half, _) = spread(1.2, 0.04, 50.0, 12.0);
    assert!((half - 1.0).abs() < 0.03, "Anamorphic 50: {half:.4}");
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
    assert!(halo > 10 && halo <= 120, "{halo}");
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

#[test]
fn tiles_equal_the_whole_frame_with_an_external_blur_map() {
    // The map of another source (the learned model) goes through the same pipeline: tiles are the whole frame.
    let scene = subject_scene(W, H, [5.0, 9.0, 14.0], Noise::LOW, 7);
    let hd = 600.97f64;
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    let (mut b, mut p) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
    for cy in 0..ch {
        for cx in 0..cw {
            let r = f64::from(scene.truth_r[(2 * cy).min(H - 1) * W + (2 * cx).min(W - 1)]);
            let (v, pr) = if r > 0.0 {
                (r / hd, 0.97)
            } else {
                (1.0 / hd, 0.02)
            };
            b[cy * cw + cx] = v as f32;
            p[cy * cw + cx] = pr as f32;
        }
    }
    let map = BlurMapInput {
        width: cw,
        height: ch,
        plane_width: W,
        plane_height: H,
        b,
        p,
    };
    let p = plan(LensKind::Default, 0.0, 100.0, 100.0, W, H, Axis::X);
    let planes = LensPlanes::build_with_map(
        &p,
        &whole(W, H, &scene.img.data),
        scene.focus,
        Some(&map),
        &|| false,
    )
    .expect("planes");
    assert!(planes.bokeh().expect("bokeh").active());
    let data = &scene.img.data;
    let whole_frame = reshape_tile(&p, &planes, 1, data, (W, H), PixelRect::new(0, 0, W, H));
    assert!(
        whole_frame.iter().zip(data).any(|(a, b)| a != b),
        "nothing changed"
    );
    for (nx, ny) in [(2usize, 2usize), (3, 3), (5, 4)] {
        let mut assembled = vec![0.0f32; 3 * W * H];
        for rect in common::tiles(W, H, nx, ny) {
            let a = reshape_tile(&p, &planes, 1, data, (W, H), rect);
            for c in 0..3 {
                for y in 0..rect.h {
                    let dst = c * W * H + (rect.y + y) * W + rect.x;
                    let src = c * rect.w * rect.h + y * rect.w;
                    assembled[dst..dst + rect.w].copy_from_slice(&a[src..src + rect.w]);
                }
            }
        }
        assert!(
            assembled
                .iter()
                .zip(&whole_frame)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{nx}x{ny} tiles differ from the whole frame"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Balls beside bright things, balls at the edge of a map, balls across tiles (docs/21 section 4, L1, "As built").
// ---------------------------------------------------------------------------------------------------------------

/// A dark scene (ground 0.03) with soft-rimmed balls `(cx, cy, radius, rim width over radius, level)` added to it
/// (the rim is a smoothstep centred on the radius, as in `soft_night_scene`), with sensor noise.
fn balls_on_ground(balls: &[(f64, f64, f64, f64, f64)], noise: Noise, seed: u64) -> Scene {
    let mut scene = night_scene_spaced(W, H, &[], Noise::NONE, seed, 2.0);
    let n = W * H;
    for v in scene.img.data.iter_mut().chain(scene.ground.iter_mut()) {
        *v += 0.03;
    }
    for (k, &(cx, cy, r, soft, level)) in balls.iter().enumerate() {
        let tint = [
            1.0,
            0.85 - 0.05 * (k % 3) as f64,
            0.6 + 0.05 * (k % 4) as f64,
        ];
        let reach = r * (1.0 + 0.5 * soft) + 2.0;
        let (x0, x1) = (
            (cx - reach).floor().max(0.0) as usize,
            ((cx + reach).ceil() as usize).min(W),
        );
        let (y0, y1) = (
            (cy - reach).floor().max(0.0) as usize,
            ((cy + reach).ceil() as usize).min(H),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                let width = (soft * r).max(1.0);
                let t = ((r + 0.5 * width - d) / width).clamp(0.0, 1.0);
                let a = t * t * (3.0 - 2.0 * t);
                for c in 0..3 {
                    scene.img.data[c * n + y * W + x] += (level * tint[c] * a) as f32;
                }
            }
        }
        scene.discs.push((cx, cy, r));
    }
    scene.focus = None;
    scene.noisy(noise, seed + 1)
}

/// A learned-style blur map of the 1000 x 667 frame: cells of 2 x 2 pixels at radius `radius_px`, out of focus with
/// probability 0.95 below `subject_above`/unsure above `unsure_above`, with `p = 0.02` (a sure subject) in the rows
/// above `subject_above` and `p = 0.5` (unsure, radius 0) between that and `unsure_above`, as a network gives it
/// near a subject's outline.
fn learned_map(radius_px: f64, subject_above: usize, unsure_above: usize) -> BlurMapInput {
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    let (mut b, mut p) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
    for cy in 0..ch {
        for cx in 0..cw {
            let y = 2 * cy;
            let (bv, pv) = if y < subject_above {
                (1.0 / hd, 0.02)
            } else if y < unsure_above {
                (1.0 / hd, 0.5)
            } else {
                (radius_px / hd, 0.95)
            };
            b[cy * cw + cx] = bv as f32;
            p[cy * cw + cx] = pv as f32;
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

fn planes_with_map(p: &LensPlan, scene: &Scene, map: &BlurMapInput) -> LensPlanes {
    LensPlanes::build_with_map(
        p,
        &whole(W, H, &scene.img.data),
        scene.focus,
        Some(map),
        &|| false,
    )
    .expect("planes")
}

/// The light above `base` per channel and the elongation (second moments of the green) over the box
/// `[x0, x1) x [y0, y1)`.
fn box_blob(
    out: &[f32],
    base: &[f32],
    (x0, x1, y0, y1): (usize, usize, usize, usize),
) -> ([f64; 3], f64) {
    let n = W * H;
    let mut energy = [0.0f64; 3];
    let (mut m, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
    for y in y0..y1 {
        for x in x0..x1 {
            for (c, e) in energy.iter_mut().enumerate() {
                *e += f64::from(out[c * n + y * W + x]) - f64::from(base[c * n + y * W + x]);
            }
            let v = f64::from(out[n + y * W + x]) - f64::from(base[n + y * W + x]);
            m += v;
            mx += v * (x as f64 + 0.5);
            my += v * (y as f64 + 0.5);
        }
    }
    let (mx, my) = (mx / m, my / m);
    let (mut sxx, mut syy) = (0.0f64, 0.0f64);
    for y in y0..y1 {
        for x in x0..x1 {
            let v = f64::from(out[n + y * W + x]) - f64::from(base[n + y * W + x]);
            sxx += v * (x as f64 + 0.5 - mx).powi(2);
            syy += v * (y as f64 + 0.5 - my).powi(2);
        }
    }
    (energy, (syy / sxx).sqrt())
}

#[test]
fn a_small_soft_ball_beside_a_larger_one_survives_as_an_oval_with_its_brightness() {
    // The small ball (r 7, soft rim, 5 times its ground) has the big one's edge in its ring over a fourteenth of the
    // ring (more than the twentieth a clear ball allows): a light beside a bright thing (the curl of a hook, a
    // neighbour in a cluster), still a light. In a learned
    // map (blurred all around) both are found, and both come out twice as tall with the surface brightness they had.
    let balls = [
        (463.0, 330.0, 7.0, 0.3, 0.14),
        (501.0, 330.0, 22.0, 0.3, 1.2),
    ];
    let scene = balls_on_ground(&balls, Noise::LOW, 51);
    let map = learned_map(9.0, 0, 0);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let planes = planes_with_map(&p, &scene, &map);
    let bp = planes.bokeh().expect("bokeh");
    assert!(bp.active());
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    for &(cx, cy, r, _, _) in &balls {
        let found = bp.balls().iter().find(|b| {
            ((b.x * W as f64 - cx).powi(2) + (b.y * H as f64 - cy).powi(2)).sqrt() < 0.3 * r
        });
        let b = found.unwrap_or_else(|| panic!("the ball of {r} px at ({cx}, {cy}) was not found"));
        assert!(
            (b.radius * hd / r - 1.0).abs() < 0.2,
            "found at radius {:.1}, wanted {r}",
            b.radius * hd
        );
    }
    let out = reshape_tile(
        &p,
        &planes,
        1,
        &scene.img.data,
        (W, H),
        PixelRect::new(0, 0, W, H),
    );
    // The small ball's own window: the side away from the big ball, and the oval's whole height.
    let win = (452, 475, 296, 364);
    let (e_out, elong) = box_blob(&out, &scene.ground, win);
    let (e_in, _) = box_blob(&scene.img.data, &scene.ground, win);
    assert!(
        (1.85..=2.15).contains(&elong),
        "the small ball is {elong:.2} times taller, wanted 2"
    );
    for c in 0..3 {
        assert!(
            (e_out[c] / (2.0 * e_in[c]) - 1.0).abs() < 0.03,
            "channel {c}: the small ball's light {:.3} -> {:.3}, wanted twice",
            e_in[c],
            e_out[c]
        );
    }
}

#[test]
fn a_ball_beside_a_bright_thing_is_found_with_the_classical_detector_too() {
    let balls = [
        (463.0, 330.0, 7.0, 0.3, 0.14),
        (501.0, 330.0, 22.0, 0.3, 1.2),
        (300.0, 200.0, 9.0, 0.2, 0.8),
        (700.0, 450.0, 11.0, 0.2, 0.9),
        (200.0, 500.0, 10.0, 0.3, 0.7),
    ];
    let scene = balls_on_ground(&balls, Noise::LOW, 52);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let planes = planes_of(&p, &scene);
    let bp = planes.bokeh().expect("bokeh");
    assert_eq!(bp.balls().len(), 5, "every ball is found");
}

#[test]
fn a_ball_is_drawn_whole_where_the_map_is_unsure_and_hidden_where_a_subject_is() {
    // A ball of 14 px whose oval (28 px up, 28 px down) crosses the lines at which a 2 x 2 split of the frame cuts
    // it. The map above the ball is first unsure (p 0.5, radius 0), then a sure subject (p 0.02) from 20 px above
    // the centre up: the first draws the oval whole, the second hides the part of it that lies over the subject
    // (but not the ball's own disc).
    let (cx, cy) = (503.0, 345.0);
    let scene = balls_on_ground(&[(cx, cy, 14.0, 0.2, 1.0)], Noise::LOW, 53);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let half = |out: &[f32]| {
        let top = box_blob(
            out,
            &scene.ground,
            (cx as usize - 20, cx as usize + 20, 305, 345),
        );
        let bottom = box_blob(
            out,
            &scene.ground,
            (cx as usize - 20, cx as usize + 20, 345, 385),
        );
        (top.0[1], bottom.0[1])
    };
    for (name, subject, unsure) in [("unsure", 0, 330), ("subject", 325, 330)] {
        let map = learned_map(14.0, subject, unsure);
        let planes = planes_with_map(&p, &scene, &map);
        assert_eq!(planes.bokeh().expect("bokeh").balls().len(), 1, "{name}");
        let whole_frame = reshape_tile(
            &p,
            &planes,
            1,
            &scene.img.data,
            (W, H),
            PixelRect::new(0, 0, W, H),
        );
        let (top, bottom) = half(&whole_frame);
        let ratio = top / bottom;
        if subject == 0 {
            assert!(
                (0.95..=1.05).contains(&ratio),
                "{name}: the oval's top half has {ratio:.3} of the bottom half's light"
            );
            let (e_out, elong) = box_blob(
                &whole_frame,
                &scene.ground,
                (cx as usize - 20, cx as usize + 20, 305, 385),
            );
            let (e_in, _) = box_blob(
                &scene.img.data,
                &scene.ground,
                (cx as usize - 20, cx as usize + 20, 305, 385),
            );
            assert!(
                (1.85..=2.15).contains(&elong),
                "{name}: {elong:.2} times taller"
            );
            assert!(
                (e_out[1] / (2.0 * e_in[1]) - 1.0).abs() < 0.03,
                "{name}: energy {:.3} -> {:.3}, wanted twice",
                e_in[1],
                e_out[1]
            );
        } else {
            assert!(
                (0.55..=0.92).contains(&ratio),
                "{name}: the top half, partly over a subject, has {ratio:.3} of the bottom half's light"
            );
        }
        // Tiles agree with the whole frame to the bit, for splits through the oval.
        for (nx, ny) in [(2usize, 2usize), (3, 3)] {
            let mut assembled = vec![0.0f32; 3 * W * H];
            for rect in common::tiles(W, H, nx, ny) {
                let a = reshape_tile(&p, &planes, 1, &scene.img.data, (W, H), rect);
                for c in 0..3 {
                    for y in 0..rect.h {
                        let dst = c * W * H + (rect.y + y) * W + rect.x;
                        let src = c * rect.w * rect.h + y * rect.w;
                        assembled[dst..dst + rect.w].copy_from_slice(&a[src..src + rect.w]);
                    }
                }
            }
            assert!(
                assembled
                    .iter()
                    .zip(&whole_frame)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: {nx}x{ny} tiles differ from the whole frame"
            );
        }
    }
}

#[test]
fn a_very_bright_soft_ball_with_a_glow_leaves_no_pillar_beside_its_oval() {
    // A ball 270 times brighter than its ground, with a rim 0.8 radii wide and a glow around that dies away over
    // 10 px: whatever of its light stays in the picture is stretched with the rest of the blur and stands beside
    // the oval as a vertical pillar. The take-out reaches to where the light has died away, so (nearly) all of it is
    // stretched with the ball, glow included: nothing is added above the patch, and the surface brightness is kept
    // (twice the light, to 8 % with so wide a rim and a glow: part of the glow stays in the picture, stretched by the gather).
    let (cx, cy) = (503.0, 340.0);
    let mut scene = balls_on_ground(&[(cx, cy, 12.0, 0.8, 8.0)], Noise::NONE, 54);
    let n = W * H;
    for y in 0..H {
        for x in 0..W {
            let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
            if d > 12.0 {
                let glow = (0.3 * (-(d - 12.0) / 10.0).exp()) as f32;
                for c in 0..3 {
                    scene.img.data[c * n + y * W + x] += glow;
                }
            }
        }
    }
    let scene = scene.noisy(Noise::LOW, 55);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let map = learned_map(12.0, 0, 0);
    let planes = planes_with_map(&p, &scene, &map);
    assert_eq!(planes.bokeh().expect("bokeh").balls().len(), 1);
    // The patch is the take-out disc, glow included, twice as tall.
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    let top = (cy - 2.0 * planes.bokeh().expect("bokeh").balls()[0].take_out_radius * hd) as usize;
    let out = reshape_tile(
        &p,
        &planes,
        1,
        &scene.img.data,
        (W, H),
        PixelRect::new(0, 0, W, H),
    );
    let win = (
        cx as usize - 70,
        cx as usize + 70,
        cy as usize - 120,
        cy as usize + 120,
    );
    let (e_out, _) = box_blob(&out, &scene.ground, win);
    let (e_in, _) = box_blob(&scene.img.data, &scene.ground, win);
    // The light in a strip above the patch's top: the glow that was left in the picture would be stretched up into
    // it and stand there as a pillar.
    let strip = |img: &[f32]| {
        let mut sum = 0.0f64;
        for y in (top - 14)..(top - 6) {
            for x in (cx as usize - 6)..(cx as usize + 6) {
                let i = W * H + y * W + x;
                sum += f64::from(img[i]) - f64::from(scene.ground[i]);
            }
        }
        sum / (12 * 8) as f64
    };
    let (above_in, above_out) = (strip(&scene.img.data), strip(&out));
    // The glow past the take-out is the picture's own, stretched with the rest of the blur: less than a tenth of the
    // ground's (0.03) is added, where a pillar would add a large part of the ball's (8).
    assert!(
        above_out - above_in <= 0.003,
        "the strip above the patch holds {above_out:.4}, the picture had {above_in:.4}: a pillar"
    );
    assert!(
        (e_out[1] / (2.0 * e_in[1]) - 1.0).abs() < 0.08,
        "energy {:.3} -> {:.3}, wanted twice",
        e_in[1],
        e_out[1]
    );
}

#[test]
fn a_soft_ball_with_a_glow_is_stretched_as_it_is_with_no_ring_and_its_edge_scaled_by_q() {
    // The owner's complaint was a harder dark circle around a squeezed ball: the ball is not drawn again as a
    // synthetic ellipse, its own pixels (soft rim, glow and all) are stretched. A soft ball of 14 px with a rim
    // 0.6 radii wide and a glow of a tenth of its level on a flat ground, noiseless: the profile out from the centre
    // along the long axis and across it falls without a dip (a ring would be one), and the width of its edge along
    // the long axis is `q` times the one across, which is the input's.
    let (cx, cy, r, level, ground) = (503.0, 340.0, 14.0, 1.2, 0.03);
    let mut scene = balls_on_ground(&[(cx, cy, r, 0.6, level)], Noise::NONE, 56);
    let n = W * H;
    for y in 0..H {
        for x in 0..W {
            let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
            let glow = (0.1 * level * (-(d - r).max(0.0) / 9.0).exp()) as f32;
            for c in 0..3 {
                scene.img.data[c * n + y * W + x] += glow;
            }
        }
    }
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let map = learned_map(r, 0, 0);
    let planes = planes_with_map(&p, &scene, &map);
    assert_eq!(planes.bokeh().expect("bokeh").balls().len(), 1);
    let out = reshape_tile(
        &p,
        &planes,
        1,
        &scene.img.data,
        (W, H),
        PixelRect::new(0, 0, W, H),
    );
    let at = |img: &[f32], x: usize, y: usize| f64::from(img[W * H + y * W + x]);
    // The profiles from the centre out: up the column (the long axis for an anamorphic axis X) and along the row.
    let up = |img: &[f32]| -> Vec<f64> {
        (0..70)
            .map(|i| at(img, cx as usize, cy as usize - i))
            .collect()
    };
    let side = |img: &[f32]| -> Vec<f64> {
        (0..40)
            .map(|i| at(img, cx as usize + i, cy as usize))
            .collect()
    };
    // The distance from the centre at which a profile falls to a share of its peak above the ground, linearly
    // interpolated.
    let falls_to = |profile: &[f64], share: f64| -> f64 {
        let peak = profile.iter().copied().fold(0.0, f64::max);
        let target = ground + share * (peak - ground);
        let top = profile
            .iter()
            .position(|v| *v == peak)
            .expect("the peak is in the profile");
        let i = top
            + profile[top..]
                .iter()
                .position(|v| *v < target)
                .expect("falls");
        (i - 1) as f64 + (profile[i - 1] - target) / (profile[i - 1] - profile[i])
    };
    // Inside the ball the profile is the lens's rim profile (this test lens brightens towards the rim); beyond its
    // edge, where a ring would be, the light only falls, to the ground.
    for (name, profile) in [("long axis", up(&out)), ("across", side(&out))] {
        let edge = falls_to(&profile, 0.5).ceil() as usize;
        for i in edge + 1..profile.len() {
            assert!(
                profile[i] <= profile[i - 1] + 0.005 * level,
                "{name}: the profile rises from {:.4} to {:.4} at {i} px, beyond the edge at {edge} px: a ring",
                profile[i - 1],
                profile[i]
            );
        }
    }
    let width = |profile: &[f64]| falls_to(profile, 0.1) - falls_to(profile, 0.5);
    let (along, across) = (width(&up(&out)), width(&side(&out)));
    let before = width(&side(&scene.img.data));
    // (The lens's rim profile lifts the rim a little, so the span from half to a tenth of the peak is narrower.)
    assert!(
        (across / before - 1.0).abs() < 0.3,
        "the edge across the stretch is {across:.2} px wide, the ball's was {before:.2}"
    );
    assert!(
        (along / across / p.q() - 1.0).abs() < 0.15,
        "the edge along the stretch is {along:.2} px wide, {:.2} times the one across (wanted {:.2})",
        along / across,
        p.q()
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Soft gathers (`docs/21` section 4, L1, "As built (soft gathers)"): a smooth bright thing in focus is no ball, a
// speck of sharp detail leaves no step, a bright edge beside a blurred field is not dragged.
// ---------------------------------------------------------------------------------------------------------------

/// A learned-style map of the frame: cells of 2 x 2 pixels, out of focus (`p = 0.95`, radius `radius_px`) except in
/// the cells within `r` of `(cx, cy)`, where the network is unsure (`p = 0.3`, radius 0): a bright thing it cannot
/// tell from one in focus.
fn map_with_unsure_disc(radius_px: f64, (cx, cy, r): (f64, f64, f64)) -> BlurMapInput {
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    let (mut b, mut p) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
    for y in 0..ch {
        for x in 0..cw {
            let d = ((2 * x + 1) as f64 - cx).hypot((2 * y + 1) as f64 - cy);
            let (bv, pv) = if d <= r {
                (0.0, 0.3)
            } else {
                (radius_px / hd, 0.95)
            };
            b[y * cw + x] = bv as f32;
            p[y * cw + x] = pv as f32;
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
fn a_smooth_bright_blob_the_map_does_not_call_blurred_is_not_a_ball() {
    // A bright blob with no plateau and a wide soft rim (a bud, a cloud: the rings of its profile read a rim of
    // more than 0.9 of its radius) in a garden the map calls blurred. Where the map says the blob itself is blurred
    // too, it is a defocused light and a ball; where the map is not sure of the blob, it is a smooth bright thing,
    // and the blur around it is no reason to call it a light.
    let scene = cone_blob((500.0, 330.0, 50.0), (3.0, 3.5), Noise::LOW, 61);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let near = |bp: &ym_lens::BokehPlanes| {
        bp.balls()
            .iter()
            .filter(|b| (b.x * W as f64 - 500.0).hypot(b.y * H as f64 - 330.0) < 12.0)
            .map(|b| b.softness)
            .collect::<Vec<_>>()
    };
    let blurred_all = planes_with_map(&p, &scene, &learned_map(9.0, 0, 0));
    let found = near(blurred_all.bokeh().expect("bokeh"));
    assert_eq!(
        found.len(),
        1,
        "the blob is a ball where the map calls it blurred"
    );
    assert!(found[0] > 0.9, "the blob's rim is wide: {:.2}", found[0]);
    let unsure = planes_with_map(&p, &scene, &map_with_unsure_disc(9.0, (500.0, 330.0, 60.0)));
    assert!(
        near(unsure.bokeh().expect("bokeh")).is_empty(),
        "no ball where the map does not call the blob blurred"
    );
}

/// A dark scene (ground 0.03) with a bright cone, `level (1 - d / reach)^power` out to `reach` pixels: a smooth
/// bright thing with no plateau, whose rings' profile reads a rim about as wide as its radius.
fn cone_blob(
    (cx, cy, reach): (f64, f64, f64),
    (level, power): (f64, f64),
    noise: Noise,
    seed: u64,
) -> Scene {
    let mut scene = balls_on_ground(&[], Noise::NONE, seed);
    let n = W * H;
    for y in (cy - reach).max(0.0) as usize..((cy + reach) as usize + 1).min(H) {
        for x in (cx - reach).max(0.0) as usize..((cx + reach) as usize + 1).min(W) {
            let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
            let a = (1.0 - d / reach).max(0.0).powf(power);
            for (c, tint) in [1.0, 0.95, 0.9].iter().enumerate() {
                scene.img.data[c * n + y * W + x] += (level * tint * a) as f32;
            }
        }
    }
    scene.focus = None;
    scene.noisy(noise, seed + 1)
}

/// A frame of one value (all channels) with a few rectangles `(x0, x1, y0, y1, value)` over it.
fn flat_with(base: f32, rects: &[(usize, usize, usize, usize, f32)]) -> Vec<f32> {
    let n = W * H;
    let mut data = vec![base; 3 * n];
    for &(x0, x1, y0, y1, v) in rects {
        for c in 0..3 {
            for y in y0..y1 {
                for x in x0..x1 {
                    data[c * n + y * W + x] = v;
                }
            }
        }
    }
    data
}

/// Planes of a frame the learned map calls blurred everywhere (`radius_px`), so that the stretch acts on all of it.
fn planes_all_blurred(p: &LensPlan, data: &[f32], radius_px: f64) -> LensPlanes {
    LensPlanes::build_with_map(
        p,
        &whole(W, H, data),
        None,
        Some(&learned_map(radius_px, 0, 0)),
        &|| false,
    )
    .expect("planes")
}

#[test]
fn a_small_sharp_thing_in_a_blurred_field_leaves_no_step_around_it() {
    // A dark 5 x 5 square on a field the map calls blurred (12 px, so the lines are 27 px long) with a soft bright
    // band below it. The sharp-edge test flags the square, and what it flags holds the stretch back around it, and
    // keeps its pixels out of the lines of the others: smoothly. The first build set the blur to 0 on the flagged
    // pixels (the square dilated by 2 px), so the lines of the pixels beside the hole, which reach the band, lost
    // the hole's samples and the lines of the pixels in it did not: a step of the band's light at the edge of a
    // 9 x 9 hole, and stripes of the width of the hole above and below it.
    let n = W * H;
    let mut data = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for y in 0..H {
            let band = 0.6 * (-((y as f64 - 352.0) / 9.0).powi(2) / 2.0).exp();
            for x in 0..W {
                data[c * n + y * W + x] = (0.3 + band) as f32;
            }
        }
        for y in 328..333 {
            for x in 498..503 {
                data[c * n + y * W + x] = 0.05;
            }
        }
    }
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let planes = planes_all_blurred(&p, &data, 12.0);
    assert!(planes.bokeh().is_some_and(|b| b.active()));
    let out = reshape_tile(&p, &planes, 1, &data, (W, H), PixelRect::new(0, 0, W, H));
    // Along rows between the square and the band, across the square's columns: the step from one pixel to the next.
    let mut worst = 0.0f32;
    for y in [336usize, 340, 344] {
        for x in 484..516 {
            let step = (out[n + y * W + x + 1] - out[n + y * W + x]).abs();
            worst = worst.max(step);
        }
    }
    // (With the sharp places counted pixel by pixel, a hole of the square dilated, the step is 0.116; with the
    // density over a 15 x 15 window it is 0.027.)
    assert!(
        worst < 0.04,
        "the largest step beside the square is {worst:.4}"
    );
    // The square itself is still there: it is sharp, and was not smeared into the field.
    let centre = out[n + 330 * W + 500];
    assert!(centre < 0.1, "the square's centre is {centre:.3}");
}

#[test]
fn a_sharp_bright_subject_beside_a_blurred_field_is_not_dragged_into_it() {
    // A bright block in focus (the map reads 0 on it) on a dark field the map calls blurred: its pixels keep
    // their values and the field touching it keeps its own, instead of taking the block's light up and down
    // along the lines.
    let data = flat_with(0.05, &[(300, 700, 300, 500, 1.0)]);
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    let (mut b, mut pr) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
    for y in 0..ch {
        for x in 0..cw {
            let (px, py) = (2 * x + 1, 2 * y + 1);
            let inside = (298..702).contains(&px) && (298..502).contains(&py);
            b[y * cw + x] = if inside { 0.0 } else { (12.0 / hd) as f32 };
            pr[y * cw + x] = if inside { 0.02 } else { 0.95 };
        }
    }
    let map = BlurMapInput {
        width: cw,
        height: ch,
        plane_width: W,
        plane_height: H,
        b,
        p: pr,
    };
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
    let planes = LensPlanes::build_with_map(&p, &whole(W, H, &data), None, Some(&map), &|| false)
        .expect("planes");
    assert!(planes.bokeh().is_some_and(|b| b.active()));
    let out = reshape_tile(&p, &planes, 1, &data, (W, H), PixelRect::new(0, 0, W, H));
    let n = W * H;
    // The block's interior is untouched.
    for (x, y) in [(310, 310), (500, 400), (690, 490)] {
        assert_eq!(out[n + y * W + x], data[n + y * W + x]);
    }
    // The field above and to the side, at the block's edge, keeps (nearly) its value.
    for (x, y) in [
        (500, 298),
        (500, 296),
        (702, 400),
        (704, 400),
        (500, 502),
        (500, 504),
    ] {
        let (got, was) = (out[n + y * W + x], data[n + y * W + x]);
        assert!(
            (got - was).abs() < 0.03,
            "({x}, {y}): {was:.3} became {got:.3}"
        );
    }
}
