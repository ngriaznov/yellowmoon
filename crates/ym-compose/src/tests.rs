//! End-to-end tests of auto crop on synthetic pictures (`docs/20-auto-crop.md` §13): the
//! straightener on façades and horizons turned by known angles, determinism and timing. The
//! proposer's own tests are in `frame.rs`.

use std::time::Instant;

use crate::testutil::*;
use crate::*;

/// A crop of the whole picture.
const WHOLE: [f64; 4] = [0.0, 0.0, 1.0, 1.0];

/// The straightener's tolerance in degrees (`docs/20` §13).
const TOLERANCE_DEG: f64 = 0.15;

/// Rolls of the tests: within ±(0.5..9)°, both signs.
const ROLLS: [f64; 10] = [-9.0, -6.4, -3.7, -1.2, -0.5, 0.5, 0.9, 2.3, 5.1, 9.0];

fn straighten_of(rgb: &[u8]) -> Straighten {
    let p = Picture::new(W, H, rgb).unwrap();
    analyse(&p, WHOLE).straighten()
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

fn request(quarter_turns: u8, ratio: Ratio, angle_deg: f64) -> FrameRequest {
    FrameRequest {
        quarter_turns,
        ratio,
        angle_deg,
    }
}

#[test]
fn two_runs_give_the_same_bytes() {
    let rgb = render(W, H, 2.3, 0.01, 37, facade_and_sea());
    let answer = |crop: [f64; 4]| {
        let a = analyse(&Picture::new(W, H, &rgb).unwrap(), crop);
        let mut out = serde_json::to_string(&a.straighten()).unwrap();
        for turns in 0..4u8 {
            for ratio in [
                Ratio::Canvas,
                Ratio::Free,
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
    for crop in [WHOLE, [0.12, 0.05, 0.81, 0.93]] {
        let first = answer(crop);
        assert_eq!(first, answer(crop));
        assert!(first.contains("\"frames\""));
    }
}

#[test]
fn timing_of_one_analysis_and_three_proposals() {
    // Printed, not asserted: run with `cargo test -p ym-compose --release timing -- --nocapture`.
    let rgb = render(W, H, 2.3, 0.01, 39, facade_and_sea());
    let picture = Picture::new(W, H, &rgb).unwrap();
    let t0 = Instant::now();
    let a = analyse(&picture, [0.1, 0.1, 0.9, 0.9]);
    let t1 = Instant::now();
    for ratio in [Ratio::Canvas, Ratio::Fixed(1.0), Ratio::Fixed(0.8)] {
        let _ = a.propose(&request(0, ratio, a.angle_deg()));
    }
    let t2 = Instant::now();
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    eprintln!(
        "ym-compose timing on {W} x {H}: analyse (the straightener) {:.1} ms, three proposals \
         {:.2} ms ({:?})",
        ms(t0, t1),
        ms(t1, t2),
        a.straighten()
    );
}
