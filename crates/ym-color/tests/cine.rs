//! The cine rendering (`docs/02-colour-pipeline.md`, C12): the CPU reference against the fixture
//! `tests/fixtures/cine_vectors.json` (made from `training/cine/cine_ref.py`, which reads no ARRI data)
//! and the properties the spec lists.

use std::sync::LazyLock;

use serde_json::Value;
use ym_color::display::OutputTarget;
use ym_color::display::cine::{CineRendering, D_MIN, consts, white_scene_level};
use ym_color::gamut::LinearAwg4;

static FIXTURE: LazyLock<Value> = LazyLock::new(|| {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cine_vectors.json"
    );
    let text = std::fs::read_to_string(path).expect("the cine fixture is in the repository");
    serde_json::from_str(&text).expect("the fixture is JSON")
});

fn triple(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("an array of three numbers");
    [
        a[0].as_f64().expect("number"),
        a[1].as_f64().expect("number"),
        a[2].as_f64().expect("number"),
    ]
}

fn cases() -> &'static Vec<Value> {
    FIXTURE["cases"].as_array().expect("cases")
}

fn max_abs(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
}

fn render(output: OutputTarget, x: [f64; 3]) -> [f64; 3] {
    CineRendering::new(output).forward(LinearAwg4(x))
}

fn target(name: &str) -> OutputTarget {
    match name {
        "rec709" => OutputTarget::Srgb,
        "p3" => OutputTarget::DisplayP3,
        "rec2020" => OutputTarget::Rec2020,
        other => panic!("unknown primaries {other}"),
    }
}

// ---- the constants --------------------------------------------------------------------------------------

#[test]
fn cine_constants_equal_the_json() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../training/cine/constants.json"
    );
    let text = std::fs::read_to_string(path).expect("constants.json is in the repository");
    let json: Value = serde_json::from_str(&text).expect("constants.json is JSON");
    let json = json.as_object().expect("an object");
    assert_eq!(
        json.len(),
        consts::TABLE.len(),
        "the JSON and the Rust table have a different number of constants; run training/cine/gen_rust_consts.py"
    );
    for (name, value) in consts::TABLE {
        let expected = json
            .get(name)
            .unwrap_or_else(|| panic!("{name} is in the Rust table and not in constants.json"))
            .as_f64()
            .expect("a number");
        assert_eq!(
            value.to_bits(),
            expected.to_bits(),
            "{name}: Rust {value:e}, JSON {expected:e}; run training/cine/gen_rust_consts.py"
        );
    }
}

#[test]
fn the_harmonic_arrays_hold_the_named_constants_in_order() {
    let by_name = |n: &str| {
        consts::TABLE
            .iter()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no constant {n}"))
    };
    for (group, array) in [
        ("gh_b0", consts::GH_B0),
        ("gh_bh", consts::GH_BH),
        ("dt_b0", consts::DT_B0),
        ("dt_bh", consts::DT_BH),
        ("rh_b0", consts::RH_B0),
        ("rh_bh", consts::RH_BH),
    ] {
        for (i, suffix) in ["c1", "s1", "c2", "s2", "c3", "s3"].iter().enumerate() {
            assert_eq!(
                array[i],
                by_name(&format!("{group}_{suffix}")),
                "{group}_{suffix}"
            );
        }
    }
}

#[test]
fn the_fixture_was_made_with_these_constants() {
    let in_fixture = FIXTURE["constants"].as_object().expect("constants");
    assert_eq!(in_fixture.len(), consts::TABLE.len());
    for (name, value) in consts::TABLE {
        let v = in_fixture[name].as_f64().expect("number");
        assert_eq!(
            value.to_bits(),
            v.to_bits(),
            "{name}: the fixture is stale; run training/cine/make_vectors.py"
        );
    }
}

// ---- forward ---------------------------------------------------------------------------------------------

#[test]
fn forward_matches_the_fixture_to_1e_9() {
    assert!(cases().len() >= 1900, "{} cases", cases().len());
    let mut worst = 0.0f64;
    for c in cases() {
        let name = c["name"].as_str().expect("name");
        let x = triple(&c["x"]);
        for key in ["rec709", "p3", "rec2020"] {
            let got = render(target(key), x);
            let want = triple(&c[key]);
            let err = max_abs(got, want);
            worst = worst.max(err);
            assert!(
                err < 1e-9,
                "{name} {key}: {got:?} against {want:?} (error {err:e})"
            );
        }
    }
    eprintln!("forward: largest difference to the fixture {worst:e}");
}

#[test]
fn viewport_and_display_p3_are_the_same_colours() {
    for c in cases().iter().step_by(7) {
        let x = triple(&c["x"]);
        assert_eq!(
            render(OutputTarget::Viewport, x),
            render(OutputTarget::DisplayP3, x)
        );
    }
}

#[test]
fn neutral_axis_reaches_exactly_one_from_ten_point_four_stops() {
    let grey = 0.18;
    for output in OutputTarget::ALL {
        let r = CineRendering::new(output);
        // At the white level itself, to rounding in all three channels.
        let at = r.forward(LinearAwg4([grey * 10.4f64.exp2(); 3]));
        for c in at {
            assert!((c - 1.0).abs() <= 4e-16, "{output:?}: {c}");
        }
        // From a hair above it, exactly 1.0; and it is the white level of the model.
        for stops in [10.41, 10.5, 11.0, 12.0, 16.0, 22.0] {
            let v = r.forward(LinearAwg4([grey * f64::exp2(stops); 3]));
            assert_eq!(v, [1.0, 1.0, 1.0], "{output:?} at +{stops} stops");
        }
        // Non-decreasing along a fine ramp, from far below to far above.
        let mut last = [0.0f64; 3];
        let mut stops = -12.0;
        while stops <= 14.0 {
            let v = r.forward(LinearAwg4([grey * f64::exp2(stops); 3]));
            for i in 0..3 {
                assert!(
                    v[i] >= last[i],
                    "{output:?} falls at {stops} stops: {last:?} then {v:?}"
                );
            }
            last = v;
            stops += 0.01;
        }
    }
    // `kap` is listed to 9 digits, which moves the start of the flat top by about 2e-8 stops.
    assert!(((white_scene_level() / grey).log2() - 10.4).abs() < 1e-6);
}

#[test]
fn middle_grey_renders_where_the_spec_says() {
    // The spec: middle grey 0.18 at display 0.1109, sRGB code 93.6, with no placement.
    let v = render(OutputTarget::Srgb, [0.18; 3]);
    assert!((v[0] - 0.1109).abs() < 5e-4, "{v:?}");
    let code = ym_color::encode::OutputSpace::Srgb.encode_clamped(v)[1] * 255.0;
    assert!((code - 93.6).abs() < 0.1, "{code}");
}

#[test]
fn grey_maps_to_grey_within_1e_12() {
    for output in OutputTarget::ALL {
        let r = CineRendering::new(output);
        let mut stops = -24.0;
        while stops <= 24.0 {
            let g = 0.18 * f64::exp2(stops);
            let v = r.forward(LinearAwg4([g; 3]));
            assert!(
                (v[0] - v[1]).abs() < 1e-12 && (v[2] - v[1]).abs() < 1e-12,
                "{output:?} at {stops} stops: {v:?}"
            );
            stops += 0.125;
        }
    }
}

#[test]
fn output_is_finite_and_in_unit_range_over_a_wide_grid() {
    let values = [
        -1e300,
        -1e30,
        -1e12,
        -1e6,
        -100.0,
        -1.0,
        -0.18,
        -1e-3,
        -1e-12,
        -1e-30,
        0.0,
        1e-300,
        1e-30,
        1e-12,
        1e-6,
        1e-3,
        0.05,
        0.18,
        0.5,
        1.0,
        5.0,
        100.0,
        243.0,
        1e4,
        1e6,
        1e12,
        1e30,
        1e150,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut checked = 0u64;
    for output in OutputTarget::ALL {
        let r = CineRendering::new(output);
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    let v = r.forward(LinearAwg4([a, b, c]));
                    for ch in v {
                        assert!(
                            ch.is_finite() && (0.0..=1.0).contains(&ch),
                            "{output:?} {a:e} {b:e} {c:e} gave {v:?}"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 100_000, "{checked}");
}

#[test]
fn the_rendering_is_a_pure_function() {
    let r = CineRendering::new(OutputTarget::Viewport);
    let x = LinearAwg4([0.31, 0.12, 0.05]);
    assert_eq!(r.forward(x), r.forward(x));
    assert_eq!(
        r.forward(x),
        CineRendering::new(OutputTarget::Viewport).forward(x)
    );
}

// ---- inverse ---------------------------------------------------------------------------------------------

/// The display colour the inverse is asked for, held as the spec says.
fn held(d: [f64; 3]) -> [f64; 3] {
    d.map(|c| c.clamp(D_MIN, 1.0))
}

#[test]
fn inverse_matches_the_fixture_by_display_colour() {
    let list = FIXTURE["inverse"].as_array().expect("inverse");
    assert_eq!(list.len(), 374);
    let (mut worst, mut worst_fraction, mut unreachable, mut gaps) =
        (0.0f64, 0.0f64, 0usize, 0usize);
    let mut failures = Vec::new();
    for c in list {
        let primaries = c["primaries"].as_str().expect("primaries");
        let note = c["note"].as_str().expect("note");
        let r = CineRendering::new(target(primaries));
        let d = triple(&c["d"]);
        let inv = r.inverse(d);
        assert!(
            inv.scene.0.iter().all(|v| v.is_finite()),
            "{primaries} {note}: {inv:?}"
        );
        // The display colour the scene colour renders to: the request itself if it was reachable, else
        // the nearest reachable colour, which the fixture holds as `roundtrip`.
        let back = r.forward(inv.scene);
        let want = triple(&c["roundtrip"]);
        let err = max_abs(back, want);
        let fraction = c["chroma_fraction"].as_f64().expect("fraction");
        let df = (inv.chroma_fraction - fraction).abs();
        if c["ray_is_interval"].as_bool() == Some(false) {
            // The reachable set along the chroma ray has gaps: the nearest-reachable answer depends on the
            // solver (the reference's bisection and the product's march from the neutral colour may end on
            // different intervals; `docs/02`, Inverse, step 4). Only require a colour the rendering makes,
            // and the reference's display colour where both find the request reachable.
            gaps += 1;
            let again = r.inverse(back);
            if !(again.chroma_fraction >= 1.0 && max_abs(r.forward(again.scene), back) < 1e-6) {
                failures.push(format!(
                    "{primaries} {note}: {back:?} is not a colour the rendering makes"
                ));
            }
            if fraction >= 1.0 && inv.chroma_fraction >= 1.0 && err >= 1e-6 {
                failures.push(format!("{primaries} {note}: {back:?} against {want:?}"));
            }
            continue;
        }
        if err.is_nan() || err >= 1e-6 || df >= 1e-3 {
            failures.push(format!(
                "{primaries} {note}: d {d:?} rendered back to {back:?}, fixture {want:?} (error {err:e}); chroma fraction {} against {fraction}",
                inv.chroma_fraction
            ));
            continue;
        }
        worst = worst.max(err);
        worst_fraction = worst_fraction.max(df);
        if fraction < 1.0 {
            unreachable += 1;
        }
    }
    eprintln!(
        "inverse: {} display colours, {unreachable} unreachable, {gaps} on rays with gaps; largest display error {worst:e}, largest chroma-fraction difference {worst_fraction:e}",
        list.len(),
    );
    assert!(
        failures.is_empty(),
        "{} failures, the first ones:\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        unreachable > 0,
        "the fixture has no unreachable colour left"
    );
}

#[test]
fn inverse_of_the_rec709_renders_returns_the_render() {
    // Every forward case: invert its Rec.709 render and render it again.
    let r = CineRendering::new(OutputTarget::Srgb);
    let mut worst = 0.0f64;
    let mut unreachable = 0usize;
    for c in cases() {
        let name = c["name"].as_str().expect("name");
        let d = triple(&c["rec709"]);
        let inv = r.inverse(d);
        let back = r.forward(inv.scene);
        let reachable = c["reachable709"].as_bool().expect("bool");
        if reachable {
            let err = max_abs(back, held(d));
            worst = worst.max(err);
            assert!(err < 1e-6, "{name}: {d:?} came back as {back:?}");
            assert!(
                (inv.chroma_fraction - 1.0).abs() < 1e-12,
                "{name}: {}",
                inv.chroma_fraction
            );
        } else {
            unreachable += 1;
            let want = triple(&c["roundtrip709"]);
            let err = max_abs(back, want);
            assert!(
                err < 1e-5,
                "{name}: nearest reachable {back:?} against {want:?}"
            );
        }
    }
    eprintln!("inverse of the renders: largest error {worst:e}, {unreachable} not reachable");
}

#[test]
fn inverse_is_always_finite_and_forward_of_it_is_a_colour_the_rendering_makes() {
    let r = CineRendering::new(OutputTarget::Viewport);
    let edge = [
        0.0, 1e-9, 1e-6, 1e-3, 0.25, 0.5, 0.75, 0.999, 1.0, 1.5, -0.2,
    ];
    let mut n_partial = 0;
    for &a in &edge {
        for &b in &edge {
            for &c in &edge {
                let inv = r.inverse([a, b, c]);
                assert!(
                    inv.scene.0.iter().all(|v| v.is_finite()),
                    "{a} {b} {c}: {inv:?}"
                );
                assert!((0.0..=1.0).contains(&inv.chroma_fraction));
                let back = r.forward(inv.scene);
                if inv.chroma_fraction >= 1.0 {
                    assert!(
                        max_abs(back, held([a, b, c])) < 1e-6,
                        "{a} {b} {c}: {back:?}"
                    );
                } else {
                    n_partial += 1;
                }
            }
        }
    }
    assert!(n_partial > 0);
    // NaN counts as the smallest channel.
    let inv = r.inverse([f64::NAN, 0.5, 0.5]);
    assert!(inv.scene.0.iter().all(|v| v.is_finite()));
}

#[test]
fn the_inverse_of_grey_is_grey_and_the_inverse_of_white_is_the_white_level() {
    for output in OutputTarget::ALL {
        let r = CineRendering::new(output);
        for g in [1e-5, 0.01, 0.1109, 0.5, 0.9, 0.999] {
            let inv = r.inverse([g; 3]);
            let s = inv.scene.0;
            assert!(
                (s[0] - s[1]).abs() < 1e-6 * s[1] && (s[2] - s[1]).abs() < 1e-6 * s[1],
                "{g}: {s:?}"
            );
            assert_eq!(inv.chroma_fraction, 1.0);
        }
        let white = r.inverse([1.0; 3]);
        let level = white_scene_level();
        for c in white.scene.0 {
            assert!(
                (c / level - 1.0).abs() < 1e-3,
                "{output:?}: {c} against {level}"
            );
        }
    }
}

#[test]
fn the_inverse_runs_at_a_useful_speed() {
    // 1.7 megapixel-like sample: the speed is reported for the milestone and checked only loosely, because
    // the machine is shared and the test profile is not the release profile.
    let r = CineRendering::new(OutputTarget::Srgb);
    let mut display = Vec::new();
    for c in cases() {
        if c["reachable709"].as_bool() == Some(true) {
            display.push(triple(&c["rec709"]));
        }
    }
    let _ = r.inverse([0.5; 3]); // builds the table
    let start = std::time::Instant::now();
    let mut acc = 0.0;
    let mut n = 0u64;
    for _ in 0..3 {
        for d in &display {
            acc += r.inverse(*d).scene.0[0];
            n += 1;
        }
    }
    let per = start.elapsed().as_secs_f64() / n as f64;
    eprintln!(
        "inverse: {n} colours, {:.2} us each, {:.3} Mpx/s on one thread (checksum {acc:.3})",
        per * 1e6,
        1e-6 / per
    );
    assert!(per < 5e-4, "{per} s per colour");
}
