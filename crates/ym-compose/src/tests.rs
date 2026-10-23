//! End-to-end tests of auto crop on synthetic pictures (`docs/20-auto-crop.md` §13): the
//! straightener on façades and horizons turned by known angles, the proposer on rendered subjects,
//! determinism and timing.

use std::time::Instant;

use crate::testutil::*;
use crate::*;

/// The straightener's tolerance in degrees (`docs/20` §13).
const TOLERANCE_DEG: f64 = 0.15;

/// Rolls of the tests: within ±(0.5..9)°, both signs.
const ROLLS: [f64; 10] = [-9.0, -6.4, -3.7, -1.2, -0.5, 0.5, 0.9, 2.3, 5.1, 9.0];

fn straighten_of(rgb: &[u8]) -> Straighten {
    let p = Picture::new(W, H, rgb).unwrap();
    analyse(&p, &CameraHint::default()).straighten()
}

/// The answer levels a picture turned clockwise by `roll` degrees: the straighten angle is
/// `−roll` within the tolerance, with the right sign.
fn assert_levels(name: &str, roll: f64, answer: Straighten) {
    match answer {
        Straighten::Rotate { angle_deg } => {
            assert!(
                (angle_deg + roll).abs() <= TOLERANCE_DEG,
                "{name} turned by {roll}°: answered {angle_deg}°"
            );
            assert_eq!(
                angle_deg.signum(),
                -roll.signum(),
                "{name} turned by {roll}°"
            );
        }
        other => panic!("{name} turned by {roll}°: answered {other:?}"),
    }
}

/// A façade filling the picture, camera pitched up 8° (its verticals lean apart by about ±4° at
/// the sides).
fn facade_only() -> impl Fn(f64, f64) -> [f64; 3] {
    facade(768.0, 8.0, -9.0, 40.0, false)
}

/// A façade on the left, the open sea on the right, the same pitched camera.
fn facade_and_sea() -> impl Fn(f64, f64) -> [f64; 3] {
    facade(768.0, 8.0, -14.0, 4.0, true)
}

#[test]
fn a_facade_with_converging_verticals_is_levelled() {
    for (k, roll) in ROLLS.iter().enumerate() {
        let rgb = render(W, H, *roll, 0.01, 100 + k as u64, facade_only());
        assert_levels("façade", *roll, straighten_of(&rgb));
    }
}

#[test]
fn a_sea_horizon_is_levelled() {
    for (k, roll) in ROLLS.iter().enumerate() {
        let rgb = render(W, H, *roll, 0.01, 200 + k as u64, open_sea);
        assert_levels("sea", *roll, straighten_of(&rgb));
    }
}

#[test]
fn a_facade_and_a_horizon_together_are_levelled() {
    for (k, roll) in [-7.2, -2.0, 0.6, 4.4, 8.5].iter().enumerate() {
        let rgb = render(W, H, *roll, 0.01, 300 + k as u64, facade_and_sea());
        assert_levels("façade and sea", *roll, straighten_of(&rgb));
    }
}

#[test]
fn level_scenes_are_level() {
    for (k, roll) in [0.0, 0.1, -0.12].iter().enumerate() {
        for (name, rgb) in [
            (
                "façade",
                render(W, H, *roll, 0.01, 400 + k as u64, facade_only()),
            ),
            ("sea", render(W, H, *roll, 0.01, 410 + k as u64, open_sea)),
            (
                "both",
                render(W, H, *roll, 0.01, 420 + k as u64, facade_and_sea()),
            ),
        ] {
            assert_eq!(
                straighten_of(&rgb),
                Straighten::Level,
                "{name} turned by {roll}°"
            );
        }
    }
}

#[test]
fn texture_without_straight_lines_is_unsure() {
    for seed in [5, 6, 7] {
        let rgb = render(W, H, 0.0, 0.01, 500 + seed, pebbles(seed));
        assert_eq!(straighten_of(&rgb), Straighten::Unsure, "pebbles {seed}");
    }
}

#[test]
fn a_tilt_of_15_degrees_is_unsure() {
    for (k, roll) in [15.0, -15.0].iter().enumerate() {
        for (name, rgb) in [
            (
                "façade",
                render(W, H, *roll, 0.01, 600 + k as u64, facade_only()),
            ),
            ("sea", render(W, H, *roll, 0.01, 610 + k as u64, open_sea)),
            (
                "both",
                render(W, H, *roll, 0.01, 620 + k as u64, facade_and_sea()),
            ),
        ] {
            assert_eq!(
                straighten_of(&rgb),
                Straighten::Unsure,
                "{name} turned by {roll}°"
            );
        }
    }
}

/// A textured orange disc of `radius` px at `(x, y)` (centre-relative) on a plain field, with an
/// optional bright bar `(x0, y0, x1, y1)`.
fn subject_scene(
    x: f64,
    y: f64,
    radius: f64,
    bar: Option<(f64, f64, f64, f64)>,
) -> impl Fn(f64, f64) -> [f64; 3] {
    move |u, v| {
        if let Some(c) = in_subject(u, v, x, y, radius) {
            return c;
        }
        if let Some((x0, y0, x1, y1)) = bar
            && (x0..=x1).contains(&u)
            && (y0..=y1).contains(&v)
        {
            return [1.0, 0.95, 0.6];
        }
        field(u, v)
    }
}

fn request(quarter_turns: u8, ratio: Ratio, angle_deg: f64) -> FrameRequest {
    FrameRequest {
        quarter_turns,
        ratio,
        angle_deg,
    }
}

/// Whether `r` holds the box `(x0, y0, x1, y1)` of canvas fractions.
fn holds(r: &Rect, (x0, y0, x1, y1): (f64, f64, f64, f64)) -> bool {
    let e = 2e-6;
    x0 >= r.x - e && y0 >= r.y - e && x1 <= r.x + r.width + e && y1 <= r.y + r.height + e
}

#[test]
fn a_rendered_centred_subject_is_kept() {
    let rgb = render(W, H, 0.0, 0.01, 31, subject_scene(0.0, 0.0, 60.0, None));
    let a = analyse(&Picture::new(W, H, &rgb).unwrap(), &CameraHint::default());
    assert!(!a.subject_from_camera());
    for ratio in [Ratio::Canvas, Ratio::Fixed(1.0), Ratio::Fixed(0.8)] {
        let p = a.propose(&request(0, ratio, 0.0));
        assert!(p.keep, "{ratio:?}: {p:?}");
    }
}

#[test]
fn a_rendered_distraction_at_an_edge_is_cropped_away_without_losing_the_subject() {
    // The subject left of centre; a bright bar 12 px from the right edge.
    let rgb = render(
        W,
        H,
        0.0,
        0.01,
        32,
        subject_scene(-61.0, 0.0, 60.0, Some((344.0, -80.0, 372.0, 80.0))),
    );
    let a = analyse(&Picture::new(W, H, &rgb).unwrap(), &CameraHint::default());
    let subject = (
        (384.0 - 121.0) / 768.0,
        196.0 / 512.0,
        (384.0 - 1.0) / 768.0,
        316.0 / 512.0,
    );
    let bar_left = (384.0 + 344.0) / 768.0;
    for ratio in [Ratio::Canvas, Ratio::Fixed(1.0)] {
        let p = a.propose(&request(0, ratio, 0.0));
        let r = p.frames[0].rect;
        // The whole photo holds the bar, so Original crops; the centred square already leaves it
        // out and holds the subject, so Square keeps it.
        assert_eq!(p.keep, ratio != Ratio::Canvas, "{ratio:?}: {p:?}");
        assert!(
            r.x + r.width <= bar_left,
            "{ratio:?}: the bar is still in {r:?}"
        );
        assert!(holds(&r, subject), "{ratio:?}: the subject is cut by {r:?}");
    }
}

#[test]
fn a_rendered_subject_the_centred_square_slices_is_brought_inside() {
    // A disc across the left edge of the centred square (which spans 128 to 640 px across).
    let rgb = render(
        W,
        H,
        0.0,
        0.01,
        34,
        subject_scene(-256.0, -40.0, 45.0, None),
    );
    let a = analyse(&Picture::new(W, H, &rgb).unwrap(), &CameraHint::default());
    let p = a.propose(&request(0, Ratio::Fixed(1.0), 0.0));
    let disc = (
        (384.0 - 256.0 - 45.0) / 768.0,
        (256.0 - 40.0 - 45.0) / 512.0,
        (384.0 - 256.0 + 45.0) / 768.0,
        (256.0 - 40.0 + 45.0) / 512.0,
    );
    assert!(!p.keep, "{p:?}");
    assert!(holds(&p.frames[0].rect, disc), "{p:?}");
}

/// A part of the subject the camera did not focus on is never cut away as a distraction: the
/// skeleton's skull when the camera focused on its ribs.
#[test]
fn a_rendered_second_part_of_the_subject_is_kept_whole() {
    // A large disc (the ribs, where the camera focused) and a smaller one above it (the skull),
    // well inside a portrait picture.
    let scene = {
        let ribs = subject_scene(0.0, 90.0, 120.0, None);
        move |u: f64, v: f64| in_subject(u, v, 0.0, -170.0, 70.0).unwrap_or_else(|| ribs(u, v))
    };
    let rgb = render(H, W, 0.0, 0.01, 36, scene);
    let hint = CameraHint {
        point: Some([0.5, (384.0 + 90.0) / 768.0]),
        box_size: None,
        placed: false,
    };
    let a = analyse(&Picture::new(H, W, &rgb).unwrap(), &hint);
    let skull = (
        (256.0 - 70.0) / 512.0,
        (384.0 - 240.0) / 768.0,
        (256.0 + 70.0) / 512.0,
        (384.0 - 100.0) / 768.0,
    );
    for ratio in [Ratio::Canvas, Ratio::Fixed(1.0), Ratio::Fixed(0.8)] {
        let p = a.propose(&request(0, ratio, 0.0));
        assert!(
            holds(&p.frames[0].rect, skull),
            "{ratio:?}: the skull is cut by {:?}",
            p.frames[0].rect
        );
    }
}

#[test]
fn a_rendered_camera_box_is_never_cut() {
    // The subject far left; the camera focused on a small dark target to the right.
    let target = |u: f64, v: f64| (u - 154.0).abs() <= 12.0 && v.abs() <= 12.0;
    let scene = {
        let base = subject_scene(-192.0, -85.0, 45.0, None);
        move |u: f64, v: f64| {
            if target(u, v) {
                [0.1, 0.1, 0.3]
            } else {
                base(u, v)
            }
        }
    };
    let rgb = render(W, H, 0.0, 0.01, 35, scene);
    let picture = Picture::new(W, H, &rgb).unwrap();
    let hint = CameraHint {
        point: Some([0.7, 0.5]),
        box_size: Some([0.04, 0.06]),
        placed: false,
    };
    let a = analyse(&picture, &hint);
    assert!(a.subject_from_camera());
    let grown = (0.7 - 0.024, 0.5 - 0.036, 0.7 + 0.024, 0.5 + 0.036);
    for turns in 0..4u8 {
        let [x0, y0] = crate::frame::picture_to_canvas(turns, [grown.0, grown.1]);
        let [x1, y1] = crate::frame::picture_to_canvas(turns, [grown.2, grown.3]);
        let b = (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1));
        for ratio in [
            Ratio::Canvas,
            Ratio::Fixed(1.0),
            Ratio::Fixed(0.8),
            Ratio::Fixed(16.0 / 9.0),
        ] {
            let p = a.propose(&request(turns, ratio, 0.0));
            for f in &p.frames {
                assert!(
                    holds(&f.rect, b),
                    "turns {turns} {ratio:?}: {f:?} cuts the box"
                );
            }
        }
    }
}

#[test]
fn quarter_turns_move_a_rendered_subject_as_geometry_does() {
    // A subject near the picture's left edge: after one clockwise quarter turn it is near the
    // canvas's top edge, so the square slides up; after three, down; after two, right.
    let rgb = render(W, H, 0.0, 0.01, 36, subject_scene(-330.0, 0.0, 40.0, None));
    let a = analyse(&Picture::new(W, H, &rgb).unwrap(), &CameraHint::default());
    let square = Ratio::Fixed(1.0);
    let first = |turns| a.propose(&request(turns, square, 0.0)).frames[0].rect;
    assert!(first(0).x < 0.02, "{:?}", first(0));
    assert!(first(1).y < 0.02, "{:?}", first(1));
    assert!(first(2).x + first(2).width > 0.98, "{:?}", first(2));
    assert!(first(3).y + first(3).height > 0.98, "{:?}", first(3));
}

#[test]
fn two_runs_give_the_same_bytes() {
    let rgb = render(W, H, 2.3, 0.01, 37, facade_and_sea());
    let rgb2 = render(
        W,
        H,
        0.0,
        0.01,
        38,
        subject_scene(-61.0, 0.0, 60.0, Some((344.0, -80.0, 372.0, 80.0))),
    );
    let hint = CameraHint {
        point: Some([0.42, 0.5]),
        box_size: Some([0.05, 0.07]),
        placed: true,
    };
    let answer = |rgb: &[u8], hint: &CameraHint| {
        let a = analyse(&Picture::new(W, H, rgb).unwrap(), hint);
        let mut out = serde_json::to_string(&a.straighten()).unwrap();
        for turns in 0..4u8 {
            for ratio in [
                Ratio::Canvas,
                Ratio::Fixed(1.0),
                Ratio::Fixed(0.8),
                Ratio::Fixed(16.0 / 9.0),
            ] {
                let p = a.propose(&request(turns, ratio, a.angle_deg()));
                out.push_str(&serde_json::to_string(&p).unwrap());
            }
        }
        out
    };
    for (rgb, hint) in [(&rgb, CameraHint::default()), (&rgb2, hint)] {
        let first = answer(rgb, &hint);
        let second = answer(rgb, &hint);
        assert_eq!(first, second);
        assert!(first.contains("\"frames\""));
    }
}

#[test]
fn timing_of_one_analysis_and_three_proposals() {
    // Printed, not asserted: run with `cargo test -p ym-compose --release timing -- --nocapture`.
    let rgb = render(W, H, 2.3, 0.01, 39, facade_and_sea());
    let picture = Picture::new(W, H, &rgb).unwrap();
    let hint = CameraHint::default();
    let t0 = Instant::now();
    let straighten = crate::straighten::straighten(&picture);
    let t1 = Instant::now();
    let maps = crate::subject::subject_maps(&picture, &hint);
    let t2 = Instant::now();
    let a = analyse(&picture, &hint);
    let t3 = Instant::now();
    for ratio in [Ratio::Canvas, Ratio::Fixed(1.0), Ratio::Fixed(0.8)] {
        let _ = a.propose(&request(0, ratio, a.angle_deg()));
    }
    let t4 = Instant::now();
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    eprintln!(
        "ym-compose timing on {W} x {H}: straightener {:.1} ms, maps {:.1} ms, analyse {:.1} ms, \
         three proposals {:.2} ms ({straighten:?}, {} map cells)",
        ms(t0, t1),
        ms(t1, t2),
        ms(t2, t3),
        ms(t3, t4),
        maps.width * maps.height
    );
}
