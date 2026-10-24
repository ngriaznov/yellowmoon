//! The bokeh detector on synthetic scenes with known blur (`docs/21` sections 7 and 14).
//!
//! The scenes are drawn analytically with a fixed seed (`tests/scenes`): textured regions blurred by discs of
//! known radii, a sharp subject with strands against a blurred background, isolated bright discs on a dark noisy
//! ground, an evenly sharp scene, flat regions. The frames are 1000 x 667 px, so the lens plane is the frame
//! (`k_L` = 1) and its half-diagonal is 601 px: `b_min` is 2.4 px and `b_cap` 18 px.

#![allow(clippy::needless_range_loop)]

mod common;
mod scenes;

use std::sync::OnceLock;

use common::{plan, whole};
use scenes::{
    Noise, Scene, anchor_scene, balls_over_texture_scene, flat_scene, night_scene,
    night_scene_spaced, sharp_scene, soft_night_scene, subject_scene,
};
use ym_lens::consts::{B_MIN, MAX_BALLS};
use ym_lens::{
    Axis, BlurMapInput, BokehPlanes, LensError, LensKind, LensPlan, LensPlanes, ReshapeReason,
    ReshapeReport,
};

const W: usize = 1000;
const H: usize = 667;
/// The half-diagonal of the 1000 x 667 lens plane in pixels.
const HD: f64 = 600.97;

fn planes_of(scene: &Scene) -> (LensPlan, LensPlanes) {
    let (w, h) = (scene.img.w, scene.img.h);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, w, h, Axis::X);
    assert_eq!(p.lens_plane_scale(), 1, "the frame is the lens plane");
    let planes = LensPlanes::build(&p, &whole(w, h, &scene.img.data), scene.focus, &|| false)
        .expect("planes");
    (p, planes)
}

fn bokeh(planes: &LensPlanes) -> &BokehPlanes {
    planes.bokeh().expect("the plan reshapes bokeh")
}

/// The subject scene without noise, made once.
fn subject_base() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| subject_scene(W, H, [5.0, 9.0, 14.0], Noise::NONE, 1))
}

const NOISES: [(&str, Noise); 4] = [
    ("none", Noise::NONE),
    ("low", Noise::LOW),
    ("mid", Noise::MID),
    ("high", Noise::HIGH),
];

#[test]
fn isolated_bright_discs_are_measured_within_15_percent() {
    let radii = [
        6.0, 8.0, 10.0, 12.0, 14.0, 7.0, 9.0, 11.0, 13.0, 15.0, 6.5, 16.0,
    ];
    for (name, noise) in NOISES {
        let scene = night_scene(W, H, &radii, noise, 7);
        let (_p, planes) = planes_of(&scene);
        let bp = bokeh(&planes);
        let (w, h) = bp.size();
        assert_eq!((w, h), (W, H));
        assert_eq!(
            bp.balls().len(),
            radii.len(),
            "noise {name}: every disc is found, nothing else"
        );
        for &(cx, cy, r) in &scene.discs {
            let ball = bp
                .balls()
                .iter()
                .find(|b| (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy) < 0.6 * r)
                .unwrap_or_else(|| {
                    panic!("noise {name}: disc of {r} px at ({cx:.0}, {cy:.0}) not found")
                });
            let est = ball.radius * HD;
            assert!(
                (est / r - 1.0).abs() < 0.15,
                "noise {name}: disc {r} px measured {est:.2}"
            );
            // The measure is in fact much better than the 15 % asked for (the smallest disc reads a few per cent
            // large: the region at 40 % of the rim's one pixel ramp is half a pixel wider than the disc).
            assert!(
                (est / r - 1.0).abs() < 0.05,
                "noise {name}: disc {r} px measured {est:.2}"
            );
            let centre_err = (ball.x * w as f64 - cx).hypot(ball.y * h as f64 - cy);
            assert!(centre_err < 0.5, "centre off by {centre_err}");
            assert!(ball.energy.iter().all(|e| *e > 0.0));
        }
        // The map carries each ball's radius over its disc, with confidence.
        for &(cx, cy, r) in &scene.discs {
            let i = cy as usize * w + cx as usize;
            let b = f64::from(bp.blur()[i]) * HD;
            assert!(
                (b / r - 1.0).abs() < 0.15,
                "map at the disc of {r} px: {b:.2}"
            );
            assert!(bp.confidence()[i] > 0.9);
        }
        assert!(
            bp.active(),
            "noise {name}: balls on a dark ground are out-of-focus background"
        );
        assert_eq!(
            planes.report(),
            ReshapeReport {
                reshaped: true,
                reason: None
            }
        );
    }
}

#[test]
fn balls_are_ordered_brightest_first_and_capped() {
    let radii = [6.0, 8.0, 10.0, 12.0, 14.0, 7.0, 9.0, 11.0];
    let scene = night_scene(W, H, &radii, Noise::LOW, 5);
    let (_p, planes) = planes_of(&scene);
    let balls = bokeh(&planes).balls();
    assert_eq!(balls.len(), radii.len());
    assert!(
        balls.windows(2).all(|w| w[0].peak >= w[1].peak),
        "brightest first"
    );
    assert!(balls.len() <= MAX_BALLS);
    // A ball's fit is the ground it sits on: a dark plane, and its energy is its disc times its brightness.
    for b in balls {
        assert!(
            b.fit.iter().all(|f| f[0] < 0.02 && f[0] >= 0.0),
            "{:?}",
            b.fit
        );
        assert!(b.take_out_radius > 1.15 * b.radius);
    }
}

#[test]
fn a_lone_round_light_and_small_sharp_lights_are_not_balls() {
    // One round light (the moon) and two: more likely sharp than blurred, so no balls and no reshaping.
    for count in [1usize, 2] {
        let scene = night_scene(W, H, &[11.0, 9.0][..count], Noise::LOW, 3);
        let (_p, planes) = planes_of(&scene);
        let bp = bokeh(&planes);
        assert!(bp.balls().is_empty(), "{count} light(s)");
        assert!(!bp.active());
        assert_eq!(
            planes.report(),
            ReshapeReport {
                reshaped: false,
                reason: Some(ReshapeReason::LittleBlur)
            }
        );
    }
    // Small sharp lamps (a few pixels across) are below the smallest ball.
    let scene = night_scene(W, H, &[1.5; 24], Noise::LOW, 3);
    let (_p, planes) = planes_of(&scene);
    assert!(bokeh(&planes).balls().is_empty());
    assert!(!planes.report().reshaped);
}

#[test]
fn of_two_lights_closer_than_1_2_radii_the_dimmer_is_dropped() {
    // Discs overlapping in pairs, one dimmer than the other; the balls kept are never closer than 1.2 radii.
    let mut scene = night_scene_spaced(W, H, &[10.0, 12.0, 9.0, 8.0, 11.0], Noise::NONE, 9, 3.0);
    let n = W * H;
    for k in 0..scene.discs.len() {
        let (cx, cy, r) = scene.discs[k];
        // A dimmer twin 0.9 radii away.
        for y in 0..H {
            for x in 0..W {
                let d = ((x as f64 + 0.5 - cx - 0.9 * r).powi(2) + (y as f64 + 0.5 - cy).powi(2))
                    .sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0) as f32;
                if a > 0.0 {
                    for c in 0..3 {
                        scene.img.data[c * n + y * W + x] += 0.1 * a;
                    }
                }
            }
        }
    }
    let (_p, planes) = planes_of(&scene);
    let balls = bokeh(&planes).balls();
    for (i, a) in balls.iter().enumerate() {
        for b in &balls[i + 1..] {
            let d = (a.x - b.x).hypot((a.y - b.y) * H as f64 / W as f64 * 0.0 + (a.y - b.y));
            let d_px = ((a.x - b.x) * W as f64).hypot((a.y - b.y) * H as f64);
            let _ = d;
            assert!(
                d_px >= 1.2 * a.radius.max(b.radius) * HD,
                "balls {d_px:.1} px apart, radii {} {}",
                a.radius * HD,
                b.radius * HD
            );
        }
    }
}

#[test]
fn balls_over_a_blurred_texture_are_found_and_lights_in_a_sharp_scene_are_not() {
    // Discs of the same size as the blur around them are confirmed by the edges; lights over a ground blurred a
    // little less are balls too (a light is at its own depth). But big lights in a scene whose ground is sharp
    // are in focus, and are not balls.
    let radii = [9.0; 6];
    let scene = balls_over_texture_scene(W, H, 9.0, &radii, Noise::LOW, 4);
    let (_p, planes) = planes_of(&scene);
    let bp = bokeh(&planes);
    assert!(bp.balls().len() >= 5, "found {}", bp.balls().len());
    assert!(
        bp.balls().iter().filter(|b| b.supported).count() >= 4,
        "the edges around them confirm the size"
    );
    // Every light is found, and measured within 10 %. (A bright patch of the blurred texture can be a ball too.)
    let (w, h) = bp.size();
    for &(cx, cy, r) in &scene.discs {
        let ball = bp
            .balls()
            .iter()
            .find(|b| (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy) < 0.6 * r)
            .unwrap_or_else(|| panic!("the light of {r} px at ({cx:.0}, {cy:.0}) was not found"));
        assert!(
            (ball.radius * HD / r - 1.0).abs() < 0.1,
            "{}",
            ball.radius * HD
        );
    }
    let scene = balls_over_texture_scene(W, H, 0.6, &[12.0, 12.0, 12.0, 12.0], Noise::LOW, 4);
    let (_p, planes) = planes_of(&scene);
    assert!(
        bokeh(&planes).balls().is_empty(),
        "four big lights in a sharp scene are lamps in focus"
    );
}

#[test]
fn soft_edged_clipped_and_dim_balls_are_found_with_their_size() {
    // Real bokeh balls are soft at the rim, sometimes clipped flat, sometimes dim: (radius, rim width over the
    // radius, level; a level above 1 is clipped at 1).
    let balls = [
        (7.0, 0.0, 0.5),
        (9.0, 0.3, 0.8),
        (11.0, 0.6, 1.6),
        (13.0, 0.5, 1.6),
        (15.0, 0.2, 0.35),
        (8.0, 0.4, 1.2),
        (10.0, 0.0, 0.6),
        (12.0, 0.4, 0.45),
        (6.0, 0.3, 1.6),
    ];
    let scene = soft_night_scene(W, H, &balls, Noise::NONE, 11, 2.6);
    let (_p, planes) = planes_of(&scene);
    let bp = bokeh(&planes);
    let (w, h) = bp.size();
    let mut found = 0;
    for &(cx, cy, r) in &scene.discs {
        let Some(ball) = bp
            .balls()
            .iter()
            .find(|b| (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy) < 0.6 * r)
        else {
            panic!("the ball of {r} px at ({cx:.0}, {cy:.0}) was not found");
        };
        found += 1;
        let est = ball.radius * HD;
        assert!(
            (est / r - 1.0).abs() < 0.2,
            "ball of {r} px measured {est:.2}"
        );
        assert!(ball.energy.iter().all(|e| *e > 0.0));
        // The disc taken out covers the soft rim.
        assert!(ball.take_out_radius > 1.15 * ball.radius);
    }
    assert_eq!(found, balls.len());
    assert!(bp.active(), "a night of balls is out-of-focus background");
}

#[test]
fn a_sharp_frame_with_specular_highlights_has_no_balls() {
    // Small bright round highlights on a sharp textured scene: in focus, however round.
    let mut scene = sharp_scene(W, H, Noise::LOW, 3);
    let n = W * H;
    // A photograph's sharp detail has contrast: stretch the texture's tones about their mean.
    let mean = scene.img.data.iter().map(|v| f64::from(*v)).sum::<f64>() / (3 * n) as f64;
    for v in &mut scene.img.data {
        *v = (mean + 3.0 * (f64::from(*v) - mean)).max(0.002) as f32;
    }
    let spots = [
        (150.0, 100.0, 3.5),
        (400.0, 220.0, 4.5),
        (700.0, 140.0, 5.0),
        (250.0, 420.0, 3.8),
        (560.0, 480.0, 6.0),
        (830.0, 380.0, 4.2),
        (120.0, 580.0, 5.5),
        (660.0, 600.0, 3.6),
    ];
    for &(cx, cy, r) in &spots {
        for y in 0..H {
            for x in 0..W {
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0);
                if a > 0.0 {
                    for c in 0..3 {
                        let v = &mut scene.img.data[c * n + y * W + x];
                        *v = (f64::from(*v) * (1.0 - a) + 2.5 * a) as f32;
                    }
                }
            }
        }
    }
    let (_p, planes) = planes_of(&scene);
    let bp = bokeh(&planes);
    assert!(
        bp.balls().is_empty(),
        "{} balls in a sharp scene",
        bp.balls().len()
    );
    assert!(!bp.active());
    assert_eq!(
        planes.report(),
        ReshapeReport {
            reshaped: false,
            reason: Some(ReshapeReason::LittleBlur)
        }
    );
}

/// Where the detector is confident, the share of pixels within `tol` of the truth, the median error, and the
/// share of the trusted background that is confident.
fn accuracy(scene: &Scene, bp: &BokehPlanes, min_conf: f32, tol: f64) -> (f64, f64, f64) {
    let (mut n, mut ok, mut trusted) = (0usize, 0usize, 0usize);
    let mut errs = Vec::new();
    for i in 0..W * H {
        if scene.trusted[i] && scene.truth_r[i] > 0.0 {
            trusted += 1;
            if bp.confidence()[i] >= min_conf && bp.blur()[i] > 0.0 {
                n += 1;
                let e = (f64::from(bp.blur()[i]) * HD / f64::from(scene.truth_r[i]) - 1.0).abs();
                errs.push(e);
                if e <= tol {
                    ok += 1;
                }
            }
        }
    }
    errs.sort_by(f64::total_cmp);
    (
        ok as f64 / n.max(1) as f64,
        errs[errs.len() / 2],
        n as f64 / trusted as f64,
    )
}

#[test]
fn the_blur_map_follows_the_truth_where_confident() {
    for (name, noise) in NOISES {
        let scene = subject_base().noisy(noise, 21);
        let (_p, planes) = planes_of(&scene);
        let bp = bokeh(&planes);
        assert!(bp.active(), "noise {name}");
        let (within, median, covered) = accuracy(&scene, bp, 0.6, 0.25);
        eprintln!(
            "noise {name}: confident pixels within 25 %: {:.1} %, median error {:.1} %, confident share {:.1} %",
            100.0 * within,
            100.0 * median,
            100.0 * covered
        );
        assert!(
            within >= 0.85,
            "noise {name}: {:.1} % within 25 %",
            100.0 * within
        );
        assert!(
            median <= 0.15,
            "noise {name}: median error {:.1} %",
            100.0 * median
        );
        assert!(
            covered >= 0.3,
            "noise {name}: only {:.1} % of the background is confident",
            100.0 * covered
        );
        // The largest blur is held at b_cap by the stretch; the map itself reports what it measured.
        assert!(bp.max_blur() <= 0.03 + 1e-9);
    }
}

#[test]
fn sharp_regions_and_strands_stay_below_b_min() {
    for (name, noise) in NOISES {
        let scene = subject_base().noisy(noise, 22);
        let (_p, planes) = planes_of(&scene);
        let bp = bokeh(&planes);
        let visible = visible_strands(&scene, noise);
        let (mut subject, mut strand) = (0usize, 0usize);
        for i in 0..W * H {
            if scene.subject[i] || (scene.strand[i] && visible[i]) {
                let b = f64::from(bp.blur()[i]);
                if b >= B_MIN {
                    if scene.subject[i] {
                        subject += 1;
                    } else {
                        strand += 1;
                    }
                }
            }
        }
        let (ns, nt) = (
            scene.subject.iter().filter(|s| **s).count(),
            visible.iter().filter(|s| **s).count(),
        );
        eprintln!(
            "noise {name}: {subject} of {ns} subject pixels and {strand} of {nt} strand pixels at or above b_min"
        );
        assert_eq!(
            subject, 0,
            "noise {name}: the sharp subject is never blurred"
        );
        assert_eq!(strand, 0, "noise {name}: nor the hair");
    }
}

#[test]
fn the_focus_point_anchors_the_subject() {
    // A smooth, soft-edged subject in a blurred textured scene: nothing in it says it is sharp but the focus point.
    let scene = anchor_scene(W, H, Noise::LOW, 6);
    let (p, planes) = planes_of(&scene);
    let bp = bokeh(&planes);
    let above = |bp: &BokehPlanes| {
        (0..W * H)
            .filter(|&i| scene.subject[i] && f64::from(bp.blur()[i]) >= B_MIN)
            .count()
    };
    let total = scene.subject.iter().filter(|s| **s).count();
    let with = above(bp);
    // Without the focus point the smooth subject reads as blurred as its surroundings.
    let no_focus = LensPlanes::build(&p, &whole(W, H, &scene.img.data), None, &|| false).unwrap();
    let without = above(bokeh(&no_focus));
    eprintln!(
        "subject pixels read as blurred: {with} with the focus point, {without} without, of {total}"
    );
    assert!(
        without as f64 > 0.15 * total as f64,
        "the test scene must make the anchor matter: {without} of {total}"
    );
    assert!(
        with == 0,
        "with the focus point the subject is in focus: {with} of {total} pixels are blurred"
    );
    // The anchor is confident.
    let centre = (H / 2) * W + W / 2;
    assert!(bp.confidence()[centre] >= 0.99);
    assert!(bp.blur()[centre] == 0.0);
    // Away from the subject the background is blurred as the truth says.
    let mut far = scene.clone();
    for i in 0..W * H {
        let (x, y) = ((i % W) as f64, (i / W) as f64);
        if (x - 500.0).hypot(y - 333.5) < 190.0 {
            far.trusted[i] = false;
        }
    }
    let (within, _, covered) = accuracy(&far, bp, 0.6, 0.25);
    assert!(within > 0.85 && covered > 0.3, "{within} {covered}");
}

#[test]
fn an_evenly_sharp_scene_is_not_reshaped() {
    for (name, noise) in NOISES {
        let scene = sharp_scene(W, H, noise, 3);
        let (_p, planes) = planes_of(&scene);
        let bp = bokeh(&planes);
        assert!(!bp.active(), "noise {name}");
        assert_eq!(
            planes.report(),
            ReshapeReport {
                reshaped: false,
                reason: Some(ReshapeReason::LittleBlur)
            },
            "noise {name}"
        );
        assert!(
            bp.blurred_fraction() < 0.005,
            "noise {name}: {}",
            bp.blurred_fraction()
        );
        assert!(bp.balls().is_empty());
        // What is left of false readings is a few specks well under the 2 % that would ask for reshaping.
        let above = bp.blur().iter().filter(|b| **b >= B_MIN as f32).count();
        assert!(
            above < W * H / 50,
            "noise {name}: {above} pixels read as blurred"
        );
    }
}

#[test]
fn flat_regions_are_left_alone() {
    let scene = flat_scene(W, H, Noise::LOW, 3);
    let (_p, planes) = planes_of(&scene);
    let bp = bokeh(&planes);
    assert!(bp.active(), "the blurred band is a third of the frame");
    let (mut flat_blurred, mut flat) = (0usize, 0usize);
    for y in 0..H {
        for x in 0..W {
            let fx = x as f64 / W as f64;
            // Flat parts, away from the band (a blur's width and the spread's reach).
            if !(0.33..=0.77).contains(&fx) {
                flat += 1;
                if bp.blur()[y * W + x] > 0.0 {
                    flat_blurred += 1;
                }
            }
        }
    }
    assert!(
        (flat_blurred as f64) < 0.02 * flat as f64,
        "{flat_blurred} of {flat} flat pixels read as blurred"
    );
    let (within, _, covered) = accuracy(&scene, bp, 0.6, 0.25);
    assert!(within > 0.85 && covered > 0.1, "{within} {covered}");
}

#[test]
fn two_runs_and_any_thread_count_give_the_same_bits() {
    let scene = subject_base().noisy(Noise::LOW, 23);
    let (p, a) = planes_of(&scene);
    let b = LensPlanes::build(&p, &whole(W, H, &scene.img.data), scene.focus, &|| false).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let c = pool.install(|| {
        LensPlanes::build(&p, &whole(W, H, &scene.img.data), scene.focus, &|| false).unwrap()
    });
    for other in [&b, &c] {
        let (x, y) = (bokeh(&a), bokeh(other));
        assert_eq!(bits(x.blur()), bits(y.blur()));
        assert_eq!(bits(x.confidence()), bits(y.confidence()));
        assert_eq!(x.balls(), y.balls());
        assert_eq!(x.edge_count(), y.edge_count());
        assert_eq!(
            x.blurred_fraction().to_bits(),
            y.blurred_fraction().to_bits()
        );
    }
    // The same with balls.
    let night = night_scene(W, H, &[6.0, 8.0, 10.0, 12.0], Noise::MID, 7);
    let (p, a) = planes_of(&night);
    let c = pool
        .install(|| LensPlanes::build(&p, &whole(W, H, &night.img.data), None, &|| false).unwrap());
    assert_eq!(bits(bokeh(&a).blur()), bits(bokeh(&c).blur()));
    assert_eq!(bokeh(&a).balls(), bokeh(&c).balls());
}

/// Strand pixels that stand out of their surroundings: darker (linear) than the mean of the non-strand pixels
/// around them by at least 0.1 and eight times the sensor noise at that level. A faint hair over a dark cell, or
/// buried in noise, cannot be told from the background by any method.
fn visible_strands(scene: &Scene, noise: Noise) -> Vec<bool> {
    let lum = scene.img.luminance();
    let mut out = vec![false; W * H];
    for y in 5..H - 5 {
        for x in 5..W - 5 {
            let i = y * W + x;
            if !scene.strand[i] {
                continue;
            }
            let (mut sum, mut n) = (0.0f64, 0.0f64);
            for dy in -5i64..=5 {
                for dx in -5i64..=5 {
                    let j = ((y as i64 + dy) as usize) * W + (x as i64 + dx) as usize;
                    if !scene.strand[j] {
                        sum += f64::from(lum[j]);
                        n += 1.0;
                    }
                }
            }
            let bg = sum / n.max(1.0);
            let sigma = (noise.shot * bg + noise.read * noise.read).sqrt();
            out[i] = n > 0.0 && bg - f64::from(lum[i]) >= 0.1f64.max(8.0 * sigma);
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cancel_ends_the_build_at_any_step() {
    let scene = subject_base().noisy(Noise::LOW, 24);
    let night = night_scene(W, H, &[6.0, 8.0, 10.0, 12.0], Noise::LOW, 7);
    for scene in [&scene, &night] {
        let p = plan(LensKind::Default, 0.0, 0.0, 100.0, W, H, Axis::X);
        let polls = std::cell::Cell::new(0usize);
        let ok = LensPlanes::build(&p, &whole(W, H, &scene.img.data), scene.focus, &|| {
            polls.set(polls.get() + 1);
            false
        });
        assert!(ok.is_ok());
        let total = polls.get();
        assert!(total >= 12, "the detector polls between its steps: {total}");
        for stop_at in [1, 3, total / 2, total - 1, total] {
            let seen = std::cell::Cell::new(0usize);
            let r = LensPlanes::build(&p, &whole(W, H, &scene.img.data), scene.focus, &|| {
                seen.set(seen.get() + 1);
                seen.get() >= stop_at
            });
            assert!(
                matches!(r, Err(LensError::Cancelled)),
                "cancelled at poll {stop_at} of {total}"
            );
            // It stops at the poll that asked, not later.
            assert_eq!(seen.get(), stop_at, "polling stops once cancelled");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// A blur map from another source (the learned model)
// ---------------------------------------------------------------------------------------------------------------

/// The scene's truth as a dense map, one cell per 2 x 2 pixels: `b` the true radius (a pixel of sharp detail reads
/// 1 px), `p` the probability that the cell is out of focus: `blurred` where the truth is a blur, 0.02 elsewhere.
fn external_map(scene: &Scene, blurred: f64) -> BlurMapInput {
    let (w, h) = (scene.img.w, scene.img.h);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (mut b, mut p) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
    for cy in 0..ch {
        for cx in 0..cw {
            let r = f64::from(scene.truth_r[(2 * cy).min(h - 1) * w + (2 * cx).min(w - 1)]);
            let (v, pr) = if r > 0.0 {
                (r / HD, blurred)
            } else {
                (1.0 / HD, 0.02)
            };
            b[cy * cw + cx] = v as f32;
            p[cy * cw + cx] = pr as f32;
        }
    }
    BlurMapInput {
        width: cw,
        height: ch,
        plane_width: w,
        plane_height: h,
        b,
        p,
    }
}

fn planes_with_map(scene: &Scene, map: &BlurMapInput) -> LensPlanes {
    let (w, h) = (scene.img.w, scene.img.h);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, w, h, Axis::X);
    LensPlanes::build_with_map(
        &p,
        &whole(w, h, &scene.img.data),
        scene.focus,
        Some(map),
        &|| false,
    )
    .expect("planes")
}

#[test]
fn an_external_map_gives_the_blur_and_keeps_the_subject_sharp() {
    let scene = subject_base().noisy(Noise::LOW, 31);
    let map = external_map(&scene, 0.97);
    let planes = planes_with_map(&scene, &map);
    let bp = bokeh(&planes);
    assert!(bp.active());
    let (within, median, covered) = accuracy(&scene, bp, 0.6, 0.25);
    eprintln!(
        "external map: within 25 % {:.1} %, median error {:.1} %, confident share {:.1} %",
        100.0 * within,
        100.0 * median,
        100.0 * covered
    );
    assert!(within >= 0.85, "{:.1} % within 25 %", 100.0 * within);
    assert!(median <= 0.15, "median error {:.1} %", 100.0 * median);
    assert!(
        covered >= 0.5,
        "{:.1} % of the background is confident",
        100.0 * covered
    );
    // The sharp subject (the map says so, and the focus point too) is never blurred.
    let stretched = (0..W * H)
        .filter(|&i| scene.subject[i] && f64::from(bp.blur()[i]) >= B_MIN)
        .count();
    assert_eq!(stretched, 0, "{stretched} subject pixels are blurred");
    // Same input, same bits.
    let again = planes_with_map(&scene, &map);
    assert_eq!(bits(bokeh(&again).blur()), bits(bp.blur()));
}

#[test]
fn balls_are_found_against_an_external_map_and_a_sharp_map_vetoes_them() {
    // Lights over a ground the map reads as blurred are balls; the same lights over a ground it reads as sharp
    // are lamps in focus, whatever the pixels look like.
    let radii = [9.0; 6];
    let scene = balls_over_texture_scene(W, H, 9.0, &radii, Noise::LOW, 4);
    let planes = planes_with_map(&scene, &external_map(&scene, 0.97));
    let bp = bokeh(&planes);
    assert!(bp.active());
    assert!(bp.balls().len() >= 5, "found {}", bp.balls().len());
    let (w, h) = bp.size();
    for &(cx, cy, r) in &scene.discs {
        let ball = bp
            .balls()
            .iter()
            .find(|b| (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy) < 0.6 * r)
            .unwrap_or_else(|| panic!("the light of {r} px at ({cx:.0}, {cy:.0}) was not found"));
        assert!(
            (ball.radius * HD / r - 1.0).abs() < 0.1,
            "{}",
            ball.radius * HD
        );
    }
    // The map says the whole ground is sharp.
    let mut sharp = scene.clone();
    sharp.truth_r = vec![0.0; W * H];
    let planes = planes_with_map(&sharp, &external_map(&sharp, 0.97));
    assert!(bokeh(&planes).balls().is_empty());
    assert!(!bokeh(&planes).active());
}

#[test]
fn a_map_that_is_unsure_does_not_reshape_and_a_fairly_sure_one_does_not_count_for_the_gate() {
    // Below p = 0.7 no cell is blurred, so nothing is stretched; between 0.7 and 0.9 the cells are blurred, but they
    // do not make the photo one with bokeh (the gate counts cells of p >= 0.9: the low-confidence blobs of a
    // featureless sky stay out of it); from 0.9 up it is a photo with bokeh.
    let scene = subject_base().noisy(Noise::LOW, 32);
    for (p, active) in [(0.6, false), (0.85, false), (0.95, true)] {
        let planes = planes_with_map(&scene, &external_map(&scene, p));
        let bp = bokeh(&planes);
        eprintln!(
            "p {p}: {} balls, blurred fraction {:.3}",
            bp.balls().len(),
            bp.blurred_fraction()
        );
        assert_eq!(bp.active(), active, "p {p}");
        // (A ball the plane holds carries its own radius into the map: only without one is the map the source's.)
        if bp.balls().is_empty() {
            let stretched = bp.blur().iter().filter(|b| **b >= B_MIN as f32).count();
            assert_eq!(stretched > 0, p >= 0.7, "p {p}: {stretched} blurred pixels");
        }
        if !active {
            assert_eq!(
                planes.report(),
                ReshapeReport {
                    reshaped: false,
                    reason: Some(ReshapeReason::LittleBlur)
                }
            );
        }
    }
}
