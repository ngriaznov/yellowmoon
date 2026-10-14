//! The inverse of the ACES 2 output transform (`ym_color::aces2::Aces2Transform::inverse_ap0_linear`):
//! against OpenColorIO's inverse of the same built-in transform, and as a round trip of our own
//! forward transform. The DNG export uses it for looks and display LUTs that carry their own display
//! rendering (`docs/09`, "What a DNG carries").
//!
//! Vectors: `tests/data/aces2/srgb_inverse.json`, made by `tools/vectors/aces2_inverse_vectors.py`
//! (display-linear Rec.709 in, linear AP0 out, OpenColorIO 2.6 float32). Run with `-- --nocapture`
//! for the measured numbers.

use serde_json::Value;
use ym_color::aces2::{Aces2Output, Aces2Transform};

fn triplets(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .expect("array of triplets")
        .iter()
        .map(|row| {
            let r = row.as_array().expect("triplet");
            let f = |i: usize| r[i].as_f64().expect("number");
            [f(0), f(1), f(2)]
        })
        .collect()
}

struct Vectors {
    groups: Vec<(String, usize, usize)>,
    display: Vec<[f64; 3]>,
    ap0: Vec<[f64; 3]>,
}

fn load() -> Vectors {
    let path = format!(
        "{}/tests/data/aces2/srgb_inverse.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(doc["schema"], "yellowmoon/aces2-inverse-vectors/1");
    let groups = doc["groups"]
        .as_array()
        .expect("groups")
        .iter()
        .map(|g| {
            (
                g["name"].as_str().expect("name").to_owned(),
                g["start"].as_u64().expect("start") as usize,
                g["count"].as_u64().expect("count") as usize,
            )
        })
        .collect();
    let v = Vectors {
        groups,
        display: triplets(&doc["input_display_linear"]),
        ap0: triplets(&doc["expected_ap0"]),
    };
    assert_eq!(v.display.len(), v.ap0.len());
    assert_eq!(
        v.display.len(),
        doc["count"].as_u64().expect("count") as usize
    );
    v
}

fn group(v: &Vectors, name: &str) -> std::ops::Range<usize> {
    let (_, start, count) = v
        .groups
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("no group {name}"));
    *start..*start + *count
}

fn srgb() -> &'static Aces2Transform {
    Aces2Transform::shared(Aces2Output::Srgb)
}

/// The largest relative difference `|a - b| / max(|b|, floor)` over the three channels.
fn rel_diff(a: [f64; 3], b: [f64; 3], floor: f64) -> f64 {
    (0..3)
        .map(|c| (a[c] - b[c]).abs() / b[c].abs().max(floor))
        .fold(0.0, f64::max)
}

#[test]
fn the_inverse_agrees_with_opencolorio_inside_the_display_range() {
    let v = load();
    let t = srgb();
    // Display values up to 0.9 are far from the tone scale's roof; there the two inverses are two
    // float32/float64 evaluations of one algorithm. Relative to the larger of the value and 1e-3 so
    // that the darkest blacks (scene 1e-4) are not judged on rounding noise.
    let mut worst = (0.0_f64, 0usize);
    for name in ["neutrals", "cube", "random", "boundary"] {
        for i in group(&v, name) {
            if v.display[i].iter().copied().fold(0.0, f64::max) > 0.9 {
                continue;
            }
            let ours = t.inverse_ap0_linear(v.display[i]);
            let d = rel_diff(ours, v.ap0[i], 1e-3);
            if d > worst.0 {
                worst = (d, i);
            }
        }
    }
    println!(
        "worst relative difference to OpenColorIO: {:.3e} at vector {} ({:?} -> ours {:?}, OCIO {:?})",
        worst.0,
        worst.1,
        v.display[worst.1],
        t.inverse_ap0_linear(v.display[worst.1]),
        v.ap0[worst.1]
    );
    assert!(worst.0 < 5e-4, "worst {worst:?}");
}

#[test]
fn the_neutral_ramp_matches_opencolorio_up_to_the_roof_region() {
    let v = load();
    let t = srgb();
    let mut worst = 0.0_f64;
    for i in group(&v, "neutrals") {
        let ours = t.inverse_ap0_linear(v.display[i]);
        let expected = v.ap0[i];
        let d = (ours[1] - expected[1]).abs() / expected[1].abs().max(1e-4);
        worst = worst.max(d);
    }
    println!("neutral ramp, worst relative difference: {worst:.3e}");
    assert!(worst < 2e-4, "{worst}");
}

#[test]
fn near_display_white_the_inverse_grows_but_stays_finite() {
    let v = load();
    let t = srgb();
    let mut prev = 0.0;
    for i in group(&v, "white_ramp") {
        let ours = t.inverse_ap0_linear(v.display[i]);
        assert!(ours.iter().all(|c| c.is_finite()), "{:?}", v.display[i]);
        // Scene value of a neutral: monotone, and OpenColorIO's value to within a factor close to 1
        // below the roof (its own float32 evaluation is steep there).
        assert!(ours[1] >= prev, "not monotone at {:?}", v.display[i]);
        prev = ours[1];
        println!(
            "display {:.4}: ours {:.3} OCIO {:.3}",
            v.display[i][0], ours[1], v.ap0[i][1]
        );
    }
    // 0.99 is scene 58.6 in OpenColorIO.
    let at_099 = t.inverse_ap0_linear([0.99; 3])[1];
    assert!((at_099 / 58.56 - 1.0).abs() < 0.02, "{at_099}");
}

#[test]
fn forward_of_inverse_returns_the_display_value() {
    let v = load();
    let t = srgb();
    let mut worst = (0.0_f64, 0usize);
    let mut total = 0usize;
    let mut over = 0usize;
    for name in ["neutrals", "cube", "random", "boundary"] {
        for i in group(&v, name) {
            let d = v.display[i];
            if d.iter().copied().fold(0.0, f64::max) > 0.95 {
                continue;
            }
            let scene = t.inverse_ap0_linear(d);
            let back = t.transform_ap0_linear(scene);
            let err = (0..3).map(|c| (back[c] - d[c]).abs()).fold(0.0, f64::max);
            total += 1;
            if err > 1e-3 {
                over += 1;
            }
            if err > worst.0 {
                worst = (err, i);
            }
        }
    }
    println!(
        "round trip display -> scene -> display: worst {:.3e} at {:?}, {over} of {total} above 1e-3",
        worst.0, v.display[worst.1]
    );
    assert!(worst.0 < 1e-6, "worst {worst:?}");
    assert_eq!(over, 0, "{over} of {total}");
}

#[test]
fn inverse_of_forward_returns_the_scene_value() {
    let t = srgb();
    // Scene colours a photograph holds: neutral ramp and moderately saturated colours, 1e-3 to 12.
    let mut worst = 0.0_f64;
    let mut n = 0;
    for e in 0..40 {
        let y = 1e-3 * (12.0 / 1e-3_f64).powf(f64::from(e) / 39.0);
        for tint in [
            [1.0, 1.0, 1.0],
            [1.0, 0.8, 0.6],
            [0.6, 0.8, 1.0],
            [1.0, 0.5, 0.5],
            [0.7, 1.0, 0.7],
        ] {
            let ap0 = [y * tint[0], y * tint[1], y * tint[2]];
            let display = t.transform_ap0_linear(ap0);
            // Skip pixels the forward transform put at the peak: not a bijection there.
            if display.iter().any(|&c| c > 0.97) {
                continue;
            }
            let back = t.inverse_ap0_linear(display);
            worst = worst.max(rel_diff(back, ap0, 1e-3));
            n += 1;
        }
    }
    println!("scene -> display -> scene: worst relative error {worst:.3e} over {n} colours");
    assert!(n > 100);
    assert!(worst < 1e-9, "{worst}");
}

#[test]
fn black_gives_black_and_junk_gives_a_finite_triple() {
    let t = srgb();
    assert_eq!(t.inverse_ap0_linear([0.0; 3]), [0.0; 3]);
    for junk in [
        [f64::NAN, 0.5, 0.5],
        [f64::INFINITY, 0.2, 0.2],
        [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.1],
        [-0.3, 0.4, 2.5],
        [1e30, 1e30, 1e30],
        [1.0, 1.0, 1.0],
        [1e-30, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ] {
        let s = t.inverse_ap0_linear(junk);
        assert!(s.iter().all(|c| c.is_finite()), "{junk:?} -> {s:?}");
    }
    // A value that cannot come out of the transform clamps to the peak: same as white.
    let white = t.inverse_ap0_linear([1.0; 3]);
    assert_eq!(t.inverse_ap0_linear([2.0; 3]), white);
    assert_eq!(
        t.inverse_ap0_linear([-1.0, f64::NAN, 5.0]),
        t.inverse_ap0_linear([0.0, 0.0, 1.0])
    );
}

#[test]
fn neutrals_stay_neutral_and_the_ramp_is_monotone() {
    let t = srgb();
    let mut prev = 0.0;
    for i in 1..=990 {
        let d = f64::from(i) / 1000.0;
        let s = t.inverse_ap0_linear([d; 3]);
        assert!(
            (s[0] - s[1]).abs() < 2e-3 * s[1] + 1e-9 && (s[2] - s[1]).abs() < 2e-3 * s[1] + 1e-9,
            "{d}: {s:?}"
        );
        assert!(s[1] >= prev, "not monotone at {d}");
        prev = s[1];
    }
}

#[test]
fn every_output_has_an_inverse_that_round_trips_a_grey() {
    for output in Aces2Output::ALL {
        let t = Aces2Transform::shared(output);
        for d in [0.01, 0.1, 0.3, 0.6, 0.85] {
            let scene = t.inverse_ap0_linear([d; 3]);
            let back = t.transform_ap0_linear(scene);
            for c in back {
                assert!(
                    (c - d).abs() < 2e-3,
                    "{output:?}: {d} -> {scene:?} -> {back:?}"
                );
            }
        }
        // 18 % grey is where it always was (through the unplaced transform).
        let grey = t.transform_ap0_linear([0.18; 3]);
        let back = t.inverse_ap0_linear(grey);
        assert!((back[1] - 0.18).abs() < 1e-3, "{output:?}: {back:?}");
    }
}
