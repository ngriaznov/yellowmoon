//! M1-10 (C10): look application: Neutral, LogC4 / LogC3 / LogC3-to-Rec.709 inputs, the intensity
//! blend, and parametric looks (JSON schema, curves, 65-point bake).

use ym_color::cdl::Cdl;
use ym_color::consts::{Vec3, linalg};
use ym_color::cube::parse_lut;
use ym_color::gamut::{LinearAwg4, LinearRec709, matrices};
use ym_color::grade::{E1, E2};
use ym_color::logc::{LogC4, logc3_ei800_decode, logc3_ei800_encode, logc4_decode, logc4_encode};
use ym_color::look::{
    DISPLAY_GAMMA, LookError, LookInput, LookTransform, ParametricLook, SCHEMA_ID,
    blend_display_linear, blend_logc4, rec709_gamma24_decode,
};
use ym_color::lut::{BAKE_SIZE, Lut, Lut3d, LutError};

fn cube_of(f: impl FnMut(Vec3) -> Vec3, size: usize) -> Lut {
    Lut::from_cube(Lut3d::bake(size, [0.0; 3], [1.0; 3], f).unwrap())
}

/// A smooth, non-trivial per-pixel function standing in for a creative LUT (our own).
fn tweak(c: Vec3) -> Vec3 {
    [
        (c[0] * 1.05 + 0.02).min(1.0) * 0.98 + 0.01 * c[1],
        c[1] * 0.97 + 0.03 * c[2] * c[2],
        (c[2] + 0.04 * (c[0] - c[1])).clamp(0.0, 1.0),
    ]
}

fn assert_close(a: Vec3, b: Vec3, tol: f64, what: &str) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() <= tol,
            "{what}: {a:?} vs {b:?} (channel {k})"
        );
    }
}

#[test]
fn neutral_is_the_identity() {
    let n = LookTransform::neutral();
    assert!(n.is_identity() && !n.includes_display_rendering());
    for v in [[0.0, 0.0, 0.0], [0.2784, 0.2784, 0.2784], [1.3, -0.2, 0.5]] {
        assert_eq!(n.apply_logc4(LogC4(v), 100.0).unwrap().0, v);
        assert_eq!(n.apply_logc4(LogC4(v), 37.0).unwrap().0, v);
    }
    // An identity CDL alone is still the identity; an identity LUT is not "no LUT" but changes nothing visible.
    let id = LookTransform::new(LookInput::LogC4, Some(Cdl::identity()), None).unwrap();
    assert!(id.is_identity());
}

#[test]
fn logc4_look_is_cdl_then_lut_directly() {
    let cdl = Cdl::new([1.1, 1.0, 0.9], [0.02, 0.0, -0.02], [1.0, 1.1, 1.0], 1.1).unwrap();
    let lut = cube_of(tweak, 17);
    let look = LookTransform::new(LookInput::LogC4, Some(cdl), Some(lut.clone())).unwrap();
    let v = LogC4([0.42, 0.31, 0.55]);
    let want = lut.sample(cdl.apply(v.0));
    assert_eq!(look.eval_logc4(v).unwrap().0, want);
    assert_eq!(look.apply_own_encoding(v.0), want);
}

#[test]
fn intensity_blends_in_logc4() {
    let lut = cube_of(tweak, 17);
    let look = LookTransform::new(LookInput::LogC4, None, Some(lut)).unwrap();
    let v = LogC4([0.62, 0.21, 0.4]);
    let full = look.eval_logc4(v).unwrap();
    assert_eq!(look.apply_logc4(v, 0.0).unwrap(), v);
    assert_eq!(look.apply_logc4(v, 100.0).unwrap(), full);
    for pct in [10.0, 25.0, 50.0, 80.0] {
        let k = pct / 100.0;
        let got = look.apply_logc4(v, pct).unwrap().0;
        let want = [0, 1, 2].map(|c| v.0[c] + k * (full.0[c] - v.0[c]));
        assert_close(got, want, 1e-15, "blend");
    }
    // Out of range intensities clamp.
    assert_eq!(look.apply_logc4(v, 250.0).unwrap(), full);
    assert_eq!(look.apply_logc4(v, -10.0).unwrap(), v);
    assert_eq!(blend_logc4(v, full, 100.0), full);
}

/// The LogC3 path written out step by step with the building blocks, independent of `look.rs`.
fn logc3_reference(v: Vec3, lut: &Lut, cdl: Option<&Cdl>) -> Vec3 {
    let m = matrices();
    let lin4 = v.map(logc4_decode);
    let lin3 = linalg::mul_vec(&m.awg4_to_awg3, lin4);
    let mut l3 = lin3.map(logc3_ei800_encode);
    if let Some(c) = cdl {
        l3 = c.apply(l3);
    }
    let out3 = lut.sample(l3);
    let lin3_out = out3.map(logc3_ei800_decode);
    let lin4_out = linalg::mul_vec(&m.awg3_to_awg4, lin3_out);
    lin4_out.map(logc4_encode)
}

#[test]
fn logc3_look_goes_through_awg3_and_logc3() {
    let lut = cube_of(tweak, 33);
    let cdl = Cdl::new([1.05, 1.0, 0.95], [0.0, 0.01, 0.0], [1.0; 3], 1.0).unwrap();
    let look = LookTransform::new(LookInput::LogC3Ei800, Some(cdl), Some(lut.clone())).unwrap();
    for v in [
        [0.2784, 0.2784, 0.2784],
        [0.45, 0.3, 0.2],
        [0.15, 0.4, 0.6],
        [0.6, 0.55, 0.65],
    ] {
        let got = look.eval_logc4(LogC4(v)).unwrap().0;
        assert_close(
            got,
            logc3_reference(v, &lut, Some(&cdl)),
            1e-12,
            "logc3 chain",
        );
    }
    // The blend happens in LogC4 after converting back.
    let v = LogC4([0.45, 0.3, 0.2]);
    let out = look.eval_logc4(v).unwrap();
    let got = look.apply_logc4(v, 40.0).unwrap().0;
    let want = [0, 1, 2].map(|c| v.0[c] + 0.4 * (out.0[c] - v.0[c]));
    assert_close(got, want, 1e-15, "logc3 blend");
}

#[test]
fn logc3_identity_lut_is_an_exact_round_trip_up_to_the_published_step() {
    // A LUT that returns its input: the look is LogC4 -> LogC3 -> LogC4, i.e. the identity.
    let id = cube_of(|c| c, 65);
    let look = LookTransform::new(LookInput::LogC3Ei800, None, Some(id)).unwrap();
    let mut worst = 0.0_f64;
    for k in 0..=200 {
        let t = 0.05 + 0.5 * f64::from(k) / 200.0;
        let v = [t, 0.9 * t + 0.02, 0.7 * t + 0.05];
        let got = look.eval_logc4(LogC4(v)).unwrap().0;
        for c in 0..3 {
            worst = worst.max((got[c] - v[c]).abs());
        }
    }
    println!("LogC3 identity look worst LogC4 error: {worst:e}");
    // 65 point linear interpolation of a straight line is exact; what remains is LogC3's 2.5e-7 step.
    assert!(worst < 5e-6, "{worst:e}");
}

#[test]
fn logc3_to_rec709_look_returns_display_linear_rec709() {
    let lut = cube_of(
        |c| {
            let y = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).clamp(0.0, 1.0);
            [y.powf(0.8), (y * 0.98).powf(0.8), (y * 0.95).powf(0.8)]
        },
        33,
    );
    let look = LookTransform::new(LookInput::LogC3ToRec709, None, Some(lut.clone())).unwrap();
    assert!(look.includes_display_rendering() && !look.is_identity());
    let v = LogC4([0.2783958365482653; 3]);
    // The chain: LogC4 -> LogC3 (0.18 grey is 400/1023) -> LUT -> gamma 2.4 decode.
    let l3 = v.to_logc3_ei800().0;
    assert!((l3[0] - 400.0 / 1023.0).abs() < 2e-8);
    let encoded = lut.sample(l3);
    let want = encoded.map(|c| c.powf(DISPLAY_GAMMA));
    let got = look.eval_display_rec709(v).unwrap().0;
    assert_close(got, want, 1e-15, "display decode");
    assert_eq!(rec709_gamma24_decode(encoded), LinearRec709(want));
    // Negative LUT output clamps to zero rather than producing NaN.
    assert_eq!(rec709_gamma24_decode([-0.1, 0.0, 0.5]).0[0], 0.0);

    // The pipeline blends with its own C12 output in display-linear light.
    let own = [0.18, 0.2, 0.19];
    assert_eq!(blend_display_linear(own, got, 0.0), own);
    assert_eq!(blend_display_linear(own, got, 100.0), got);
    let half = blend_display_linear(own, got, 50.0);
    for c in 0..3 {
        assert!((half[c] - 0.5 * (own[c] + got[c])).abs() < 1e-15);
    }
    // Other gamuts: convert the look output before blending.
    let p3 = LinearRec709(got).to_display_p3();
    assert!(p3.0.iter().all(|c| c.is_finite()));

    // Log-to-log API refuses display-referred looks, and vice versa, at every intensity.
    assert_eq!(look.eval_logc4(v), Err(LookError::IncludesDisplayRendering));
    assert_eq!(
        look.apply_logc4(v, 0.0),
        Err(LookError::IncludesDisplayRendering)
    );
    assert_eq!(
        LookTransform::neutral().eval_display_rec709(v),
        Err(LookError::NoDisplayRendering)
    );
    assert_eq!(
        LookTransform::new(LookInput::LogC3ToRec709, None, None).unwrap_err(),
        LookError::MissingLut
    );
}

#[test]
fn looks_run_from_parsed_cube_files() {
    let lut = parse_lut(&cube_of(tweak, 9).cube.to_cube_string("t")).unwrap();
    let look = LookTransform::new(LookInput::LogC4, None, Some(lut)).unwrap();
    assert!(look.eval_logc4(LogC4([0.3, 0.4, 0.5])).is_ok());
    assert_eq!(look.lut().unwrap().cube.size(), 9);
    assert!(look.cdl().is_none());
}

// ---- Parametric looks --------------------------------------------------------------------------

fn spec(body: &str) -> String {
    format!(r#"{{"schema":"{SCHEMA_ID}","name":"t"{body}}}"#)
}

fn hue_chroma(v: Vec3) -> (f64, f64) {
    let p = v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2];
    let q = v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2];
    (q.atan2(p).to_degrees().rem_euclid(360.0), p.hypot(q))
}

fn luma709(v: Vec3) -> f64 {
    0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
}

#[test]
fn the_documented_example_parses_and_bakes() {
    let text = include_str!("data/looks/example_parametric.json");
    let look = ParametricLook::from_json(text).unwrap();
    assert_eq!(look.name(), "Test warm");
    assert!(!look.is_identity());
    let t = look.to_transform().unwrap();
    assert_eq!(t.lut().unwrap().cube.size(), BAKE_SIZE);
    assert_eq!(BAKE_SIZE, 65);
    assert_eq!(t.input(), LookInput::LogC4);
    // A baked 65^3 cube is within a small error of the exact evaluation.
    let mut worst = 0.0_f64;
    let mut seed = 99_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    for _ in 0..20_000 {
        let v = [next(), next(), next()];
        let exact = look.evaluate(v);
        let baked = t.eval_logc4(LogC4(v)).unwrap().0;
        for c in 0..3 {
            worst = worst.max((exact[c] - baked[c]).abs());
        }
    }
    println!("65^3 bake of the example look: max |baked - exact| = {worst:e} LogC4 code values");
    // The worst points are the most saturated reds near the blue-channel floor; typical error is ~1e-4.
    assert!(worst < 1.0e-2, "{worst:e}");
}

#[test]
fn neutral_json_is_the_identity_look() {
    let look = ParametricLook::from_json(&spec("")).unwrap();
    assert!(look.is_identity());
    assert!(look.to_transform().unwrap().is_identity());
    assert_eq!(look.evaluate([0.3, 0.6, 0.1]), [0.3, 0.6, 0.1]);
    assert!(ParametricLook::neutral("n").is_identity());
}

#[test]
fn json_validation() {
    let cases: Vec<(String, &str)> = vec![
        ("not json".into(), "Json"),
        (
            r#"{"schema":"yellowmoon.parametric-look/1"}"#.into(),
            "Json",
        ),
        (spec(r#","bogus":1"#), "Json"),
        (
            spec(r#","cdl":{"slope":[1,1,1],"offset":[0,0,0],"power":[1,1,1]}"#),
            "Json",
        ),
        (r#"{"schema":"v2","name":"x"}"#.into(), "UnsupportedSchema"),
        (spec(r#","input":"logc3_ei800""#), "UnsupportedInput"),
        (
            spec(r#","cdl":{"slope":[-1,1,1],"offset":[0,0,0],"power":[1,1,1],"saturation":1}"#),
            "Cdl",
        ),
        (
            spec(r#","cdl":{"slope":[1,1,1],"offset":[0,0,0],"power":[0,1,1],"saturation":1}"#),
            "Cdl",
        ),
        (
            spec(r#","hue_vs_sat":[{"hue_deg":360,"value":1}]"#),
            "BadCurve",
        ),
        (
            spec(r#","hue_vs_sat":[{"hue_deg":-1,"value":1}]"#),
            "BadCurve",
        ),
        (
            spec(r#","hue_vs_sat":[{"hue_deg":30,"value":1},{"hue_deg":30,"value":1.1}]"#),
            "BadCurve",
        ),
        (
            spec(r#","hue_vs_sat":[{"hue_deg":30,"value":9}]"#),
            "BadCurve",
        ),
        (
            spec(r#","hue_vs_sat":[{"hue_deg":30,"value":-0.1}]"#),
            "BadCurve",
        ),
        (
            spec(r#","hue_vs_hue":[{"hue_deg":30,"value":181}]"#),
            "BadCurve",
        ),
        (
            spec(r#","luma_vs_sat":[{"luma":1.01,"value":1}]"#),
            "BadCurve",
        ),
        (
            spec(r#","luma_vs_sat":[{"luma":0.5,"value":1},{"luma":0.2,"value":1}]"#),
            "BadCurve",
        ),
        (
            spec(r#","hue_vs_hue":[{"hue_deg":10,"value":1e999}]"#),
            "Json",
        ),
    ];
    for (text, want) in cases {
        let err = ParametricLook::from_json(&text).expect_err(&text);
        assert!(
            format!("{err:?}").starts_with(want),
            "{text}: expected {want}, got {err:?}"
        );
        assert!(!err.to_string().is_empty());
    }
    // Too many points.
    let pts: Vec<String> = (0..65)
        .map(|i| format!(r#"{{"hue_deg":{},"value":1}}"#, f64::from(i) * 5.0))
        .collect();
    let big = spec(&format!(r#","hue_vs_sat":[{}]"#, pts.join(",")));
    assert!(matches!(
        ParametricLook::from_json(&big),
        Err(LookError::BadCurve {
            section: "hue_vs_sat",
            ..
        })
    ));
}

#[test]
fn hue_vs_sat_scales_chroma_and_keeps_luma_and_hue() {
    let look =
        ParametricLook::from_json(&spec(r#","hue_vs_sat":[{"hue_deg":0,"value":1.5}]"#)).unwrap();
    // One point is a constant: chroma x1.5 at every hue.
    for v in [
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.3, 0.2, 0.6],
        [0.7, 0.65, 0.3],
    ] {
        let out = look.evaluate(v);
        let (h0, c0) = hue_chroma(v);
        let (h1, c1) = hue_chroma(out);
        assert!((c1 / c0 - 1.5).abs() < 1e-12, "{v:?}");
        assert!((h1 - h0).abs() < 1e-9 || (h1 - h0).abs() > 359.999999);
        assert!((luma709(out) - luma709(v)).abs() < 1e-14);
    }
    // Neutrals stay neutral.
    assert_eq!(look.evaluate([0.4, 0.4, 0.4]), [0.4, 0.4, 0.4]);
    // A curve that is 0 everywhere desaturates fully.
    let gray =
        ParametricLook::from_json(&spec(r#","hue_vs_sat":[{"hue_deg":0,"value":0}]"#)).unwrap();
    let out = gray.evaluate([0.6, 0.3, 0.1]);
    assert!((out[0] - out[1]).abs() < 1e-12 && (out[1] - out[2]).abs() < 1e-12);
    assert!((out[0] - luma709([0.6, 0.3, 0.1])).abs() < 1e-12);
}

#[test]
fn hue_vs_sat_follows_its_control_points() {
    // Boost red (hue 0), leave green (120) and blue (240) alone.
    let look = ParametricLook::from_json(&spec(
        r#","hue_vs_sat":[{"hue_deg":0,"value":2.0},{"hue_deg":120,"value":1.0},{"hue_deg":240,"value":1.0}]"#,
    ))
    .unwrap();
    // Pure red-axis colour: v = m + p E1 with q = 0 has hue 0.
    let red = [0.5 + 0.1 * E1[0], 0.5 + 0.1 * E1[1], 0.5 + 0.1 * E1[2]];
    let (h, c) = hue_chroma(red);
    assert!(h.abs() < 1e-9 || (h - 360.0).abs() < 1e-9);
    let out = look.evaluate(red);
    assert!((hue_chroma(out).1 / c - 2.0).abs() < 1e-9);
    // Hue 120 degrees: p = cos, q = sin of 120.
    let t = 120.0_f64.to_radians();
    let green: Vec3 = [0, 1, 2].map(|k| 0.5 + 0.1 * (t.cos() * E1[k] + t.sin() * E2[k]));
    let out = look.evaluate(green);
    assert!((hue_chroma(out).1 / hue_chroma(green).1 - 1.0).abs() < 1e-9);
    // Between the points the factor is between the neighbours' values (no overshoot).
    for deg in (0..360).step_by(7) {
        let t = f64::from(deg).to_radians();
        let v: Vec3 = [0, 1, 2].map(|k| 0.5 + 0.1 * (t.cos() * E1[k] + t.sin() * E2[k]));
        let factor = hue_chroma(look.evaluate(v)).1 / hue_chroma(v).1;
        assert!(
            (1.0 - 1e-9..=2.0 + 1e-9).contains(&factor),
            "hue {deg}: {factor}"
        );
    }
}

#[test]
fn hue_curves_wrap_around_at_360() {
    let look = ParametricLook::from_json(&spec(
        r#","hue_vs_sat":[{"hue_deg":10,"value":1.8},{"hue_deg":200,"value":0.6}]"#,
    ))
    .unwrap();
    let factor_at = |deg: f64| {
        let t = deg.to_radians();
        let v: Vec3 = [0, 1, 2].map(|k| 0.5 + 0.1 * (t.cos() * E1[k] + t.sin() * E2[k]));
        hue_chroma(look.evaluate(v)).1 / hue_chroma(v).1
    };
    // Continuous across 0 / 360 and exact at the points.
    assert!((factor_at(359.9999) - factor_at(0.0001)).abs() < 1e-4);
    assert!((factor_at(10.0) - 1.8).abs() < 1e-9);
    assert!((factor_at(200.0) - 0.6).abs() < 1e-9);
    // The shortest way round from 200 to 10 passes through 0: values there lie between 0.6 and 1.8.
    for deg in [250.0, 300.0, 350.0, 0.0, 5.0] {
        let f = factor_at(deg);
        assert!((0.6 - 1e-9..=1.8 + 1e-9).contains(&f), "hue {deg}: {f}");
    }
}

/// A look with only a `hue_vs_sat` curve through `points` (hue in degrees, chroma factor).
fn hue_vs_sat_look(points: &[(f64, f64)]) -> ParametricLook {
    let pts: Vec<String> = points
        .iter()
        .map(|(h, v)| format!(r#"{{"hue_deg":{h},"value":{v}}}"#))
        .collect();
    ParametricLook::from_json(&spec(&format!(r#","hue_vs_sat":[{}]"#, pts.join(",")))).unwrap()
}

/// The chroma factor a `hue_vs_sat`-only look applies at hue `deg` (chroma 0.1 about mid grey).
fn chroma_factor(look: &ParametricLook, deg: f64) -> f64 {
    let t = deg.to_radians();
    let v: Vec3 = [0, 1, 2].map(|k| 0.5 + 0.1 * (t.cos() * E1[k] + t.sin() * E2[k]));
    hue_chroma(look.evaluate(v)).1 / hue_chroma(v).1
}

#[test]
fn hue_curves_match_an_independent_periodic_fritsch_carlson() {
    // Reference values from a separate implementation (numpy, written for this review): tangents
    // from the neighbouring secants taken cyclically, one limiter pass round the circle, cubic
    // Hermite segments including the closing one from the last point to the first plus 360 degrees.
    type Case = (&'static [(f64, f64)], &'static [(f64, f64)]);
    let cases: [Case; 3] = [
        (
            &[(20.0, 1.25), (120.0, 0.85), (250.0, 0.9)],
            &[
                (0.0, 1.23070095584888),
                (5.0, 1.23886265361857),
                (15.0, 1.24869993172508),
                (25.0, 1.2471),
                (35.0, 1.2257),
                (100.0, 0.8916),
                (199.0, 0.861220732817478),
                (250.0, 0.9),
                (300.0, 1.03734638142922),
                (340.0, 1.18081474738279),
                (355.0, 1.22062756030951),
            ],
        ),
        // The limiter is active on the first segment (100 to 110 degrees): the tangent at 110
        // degrees falls from 0.15 to 0.03.
        (
            &[(100.0, 0.0), (110.0, 0.1), (120.0, 3.0), (300.0, 1.5)],
            &[
                (0.0, 0.81787109375),
                (5.0, 0.755572001139323),
                (15.0, 0.632804870605469),
                (25.0, 0.514411926269531),
                (35.0, 0.40271250406901),
                (100.0, 0.0),
                (105.0, 0.0125),
                (110.0, 0.1),
                (115.0, 1.5875),
                (119.0, 2.9215),
                (150.0, 2.92578125),
                (199.0, 2.55907286844136),
                (250.0, 2.01384066358025),
                (300.0, 1.5),
                (340.0, 1.06640625),
                (355.0, 0.880393981933594),
            ],
        ),
        (
            &[(10.0, 1.8), (200.0, 0.6)],
            &[
                (0.0, 1.78803175249338),
                (5.0, 1.79694687563607),
                (15.0, 1.79755066336201),
                (25.0, 1.77874325703455),
                (35.0, 1.74314039947514),
                (100.0, 1.24732468289838),
                (199.0, 0.600099373086456),
                (250.0, 0.850356197842459),
                (300.0, 1.35717484225524),
                (340.0, 1.70107877060859),
                (355.0, 1.77362100549562),
            ],
        ),
    ];
    for (points, expected) in cases {
        let look = hue_vs_sat_look(points);
        for &(deg, want) in expected {
            let got = chroma_factor(&look, deg);
            assert!(
                (got - want).abs() < 1e-9,
                "{points:?} at {deg} degrees: {got} vs {want}"
            );
        }
    }
}

#[test]
fn hue_curves_are_c1_at_every_control_point_including_the_seam() {
    // The curve is periodic, so the first point has two neighbours like any other: the slope just
    // before and just after each control point agrees (checked by one-sided finite differences).
    let sets: [&[(f64, f64)]; 3] = [
        &[(20.0, 1.25), (120.0, 0.85), (250.0, 0.9)],
        &[(100.0, 0.0), (110.0, 0.1), (120.0, 3.0), (300.0, 1.5)],
        &[(0.0, 1.4), (90.0, 0.7), (180.0, 1.1), (270.0, 0.9)],
    ];
    let h = 1e-5;
    for points in sets {
        let look = hue_vs_sat_look(points);
        for &(deg, _) in points {
            let at = chroma_factor(&look, deg);
            let left = (at - chroma_factor(&look, deg - h)) / h;
            let right = (chroma_factor(&look, deg + h) - at) / h;
            assert!(
                (left - right).abs() < 1e-6,
                "{points:?}: slope {left} before and {right} after {deg} degrees"
            );
        }
        // And the factor is the same one full turn apart.
        for deg in [3.0, 77.0, 181.5, 333.3] {
            assert!((chroma_factor(&look, deg) - chroma_factor(&look, deg + 360.0)).abs() < 1e-12);
        }
    }
}

#[test]
fn hue_vs_hue_rotates_and_keeps_chroma_and_luma() {
    let look =
        ParametricLook::from_json(&spec(r#","hue_vs_hue":[{"hue_deg":0,"value":30}]"#)).unwrap();
    for v in [[0.55, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.25, 0.6]] {
        let out = look.evaluate(v);
        let (h0, c0) = hue_chroma(v);
        let (h1, c1) = hue_chroma(out);
        assert!((c1 - c0).abs() < 1e-12);
        let turn = (h1 - h0 + 540.0).rem_euclid(360.0) - 180.0;
        assert!((turn - 30.0).abs() < 1e-9, "{v:?}: turned {turn}");
        assert!((luma709(out) - luma709(v)).abs() < 1e-14);
    }
    // Positive rotation turns red towards yellow: hue 0 -> 30.
    let red = [0.6, 0.45, 0.45];
    assert!(hue_chroma(red).0 < 1e-6 || hue_chroma(red).0 > 359.9999);
    assert!((hue_chroma(look.evaluate(red)).0 - 30.0).abs() < 1e-9);
}

#[test]
fn luma_vs_sat_depends_on_luma() {
    let look = ParametricLook::from_json(&spec(
        r#","luma_vs_sat":[{"luma":0.2,"value":0.5},{"luma":0.6,"value":1.5}]"#,
    ))
    .unwrap();
    let dark = [0.2, 0.19, 0.17];
    let bright = [0.7, 0.62, 0.58];
    let f_dark = hue_chroma(look.evaluate(dark)).1 / hue_chroma(dark).1;
    let f_bright = hue_chroma(look.evaluate(bright)).1 / hue_chroma(bright).1;
    assert!((f_dark - 0.5).abs() < 0.05, "{f_dark}");
    assert!((f_bright - 1.5).abs() < 1e-9, "{f_bright}");
    // Flat beyond the end points.
    let very_dark = [0.0, 0.0, 0.05];
    assert!((hue_chroma(look.evaluate(very_dark)).1 / hue_chroma(very_dark).1 - 0.5).abs() < 1e-9);
}

#[test]
fn cdl_comes_first_and_curves_see_its_output() {
    let json = spec(
        r#","cdl":{"slope":[1.2,1,1],"offset":[0,0,0],"power":[1,1,1],"saturation":1},"hue_vs_sat":[{"hue_deg":0,"value":0}]"#,
    );
    let look = ParametricLook::from_json(&json).unwrap();
    let out = look.evaluate([0.4, 0.4, 0.4]);
    // Slope warms the grey first (R = 0.48), then the zero chroma curve flattens it to its luma.
    let after_cdl = [0.48, 0.4, 0.4];
    for c in out {
        assert!((c - luma709(after_cdl)).abs() < 1e-12);
    }
}

#[test]
fn bake_sizes_and_domain() {
    let look =
        ParametricLook::from_json(include_str!("data/looks/example_parametric.json")).unwrap();
    for n in [2, 9, 33, 65] {
        let cube = look.bake(n).unwrap();
        assert_eq!(cube.size(), n);
        assert_eq!(cube.domain_min(), [0.0; 3]);
        assert_eq!(cube.domain_max(), [1.0; 3]);
    }
    assert!(look.bake(1).is_err());
    // Sizes beyond what a .cube file may declare are refused before any allocation.
    for n in [130, 100_000, usize::MAX] {
        assert!(
            matches!(look.bake(n), Err(LookError::Lut(LutError::TooLarge { .. }))),
            "bake({n})"
        );
    }
    // The baked cube is exact at its grid nodes.
    let cube = look.bake(33).unwrap();
    let v = [8.0 / 32.0, 20.0 / 32.0, 3.0 / 32.0];
    let exact = look.evaluate(v);
    let got = cube.sample(v);
    for c in 0..3 {
        assert!((got[c] - exact[c]).abs() < 1e-12);
    }
}

#[test]
fn a_gray_ramp_through_a_chroma_only_look_is_untouched() {
    let look = ParametricLook::from_json(&spec(
        r#","hue_vs_sat":[{"hue_deg":40,"value":1.8}],"hue_vs_hue":[{"hue_deg":200,"value":25}],"luma_vs_sat":[{"luma":0.3,"value":2}]"#,
    ))
    .unwrap();
    let t = look.to_transform().unwrap();
    for k in 0..=100 {
        let v = f64::from(k) / 100.0;
        let out = t.eval_logc4(LogC4([v; 3])).unwrap().0;
        assert_close(out, [v; 3], 1e-12, "grey ramp");
    }
    // And LinearAwg4 grey 0.18 passes through the whole encode / look chain unchanged.
    let gray = LinearAwg4([0.18; 3]).to_logc4();
    assert_close(t.eval_logc4(gray).unwrap().0, gray.0, 1e-12, "grey card");
}

#[test]
fn finite_in_finite_out_through_every_input_kind() {
    let lut = cube_of(tweak, 9);
    for input in [LookInput::LogC4, LookInput::LogC3Ei800] {
        let look = LookTransform::new(input, Some(Cdl::identity()), Some(lut.clone())).unwrap();
        for v in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-0.3, 0.5, 2.0],
            [5.0, -5.0, 0.2784],
        ] {
            let out = look.apply_logc4(LogC4(v), 60.0).unwrap().0;
            assert!(
                out.iter().all(|c| c.is_finite()),
                "{input:?} {v:?}: {out:?}"
            );
        }
    }
    let rec = LookTransform::new(LookInput::LogC3ToRec709, None, Some(lut)).unwrap();
    for v in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-0.3, 0.5, 2.0]] {
        let out = rec.eval_display_rec709(LogC4(v)).unwrap().0;
        assert!(out.iter().all(|c| c.is_finite() && *c >= 0.0));
    }
}
