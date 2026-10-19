//! The `luma_curve` field of parametric looks and the built-in Signature look
//! (`data/looks/builtin/signature.json`, `docs/08-looks.md`): a LogC4 luma curve applied as
//! `v += T(L) - L`, then ASC CDL saturation 1.45, baked into a 65-point cube like every parametric
//! look (so the CPU and the GPU need no kernel of their own for it).
//!
//! The constants of Signature were chosen by eye (owner's decisions of 2026-10-14 and 2026-10-19); nothing here fits
//! them to anything. What is tested: the file parses to what the spec says, mid grey and neutrals
//! stay where they are, the look is monotone, the 65-point cube follows the exact look to one 8-bit
//! display code, and intensity 0 and 100 are exactly Neutral and exactly the look.

use std::sync::Arc;

use ym_color::consts::linalg::{self, Mat3};
use ym_color::consts::{REC709_LUMA_WEIGHTS, Vec3};
use ym_color::curve::Curve;
use ym_color::display::{ChainParams, ColorChain, LookSetting, OutputTarget};
use ym_color::gamut::matrices;
use ym_color::logc::{LogC4, logc4_decode, logc4_encode};
use ym_color::look::{LookError, LookInput, LookTransform, ParametricLook, SCHEMA_ID};
use ym_color::lut::BAKE_SIZE;
use ym_color::tone::ToneParams;

/// The shipped file, read at compile time so the test cannot drift from it.
const SIGNATURE_JSON: &str = include_str!("../../../data/looks/builtin/signature.json");

fn signature() -> ParametricLook {
    ParametricLook::from_json(SIGNATURE_JSON).expect("signature.json parses")
}

fn spec(body: &str) -> String {
    format!(r#"{{"schema":"{SCHEMA_ID}","name":"t"{body}}}"#)
}

/// A small seeded generator in `[0, 1)` (the same LCG as the other look tests).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

fn grey_code() -> f64 {
    logc4_encode(0.18)
}

// ---- The file --------------------------------------------------------------------------------

#[test]
fn the_file_is_the_documented_look() {
    let look = signature();
    assert_eq!(look.name(), "Signature");
    assert!(!look.is_identity());

    // The 15 points of strength B of the richer look (2026-10-19), and CDL saturation 1.45 with no
    // slope, offset or power.
    let curve = look.luma_curve().expect("a luma curve");
    assert_eq!(curve.points().len(), 15);
    assert_eq!((curve.points()[0].x, curve.points()[0].y), (0.0, 0.0));
    assert_eq!((curve.points()[14].x, curve.points()[14].y), (1.0, 1.0));
    assert_eq!(
        (curve.points()[8].x, curve.points()[8].y),
        (0.278396, 0.278396),
        "the pivot at mid grey"
    );
    assert!(curve.points().windows(2).all(|w| w[1].x > w[0].x));
    assert!(
        curve.points().windows(2).all(|w| w[1].y > w[0].y),
        "the tone curve rises everywhere"
    );
    let cdl = look.cdl();
    assert_eq!(cdl.slope, [1.0; 3]);
    assert_eq!(cdl.offset, [0.0; 3]);
    assert_eq!(cdl.power, [1.0; 3]);
    assert_eq!(cdl.saturation, 1.45);

    let t = look.to_transform().expect("bakes");
    assert_eq!(t.input(), LookInput::LogC4);
    assert!(!t.is_identity() && !t.includes_display_rendering());
    assert_eq!(t.lut().expect("a cube").cube.size(), BAKE_SIZE);
    assert_eq!(BAKE_SIZE, 65);
}

/// Values of the same curve from an independent implementation (the research prototype's
/// Fritsch-Carlson spline in NumPy, built from the 15 points of the file). `T(x)` at points that
/// are not control points, so a wrong tangent rule or a wrong segment shows.
#[test]
fn the_curve_matches_an_independent_fritsch_carlson() {
    let look = signature();
    let curve = look.luma_curve().expect("curve");
    let reference = [
        (0.05, 0.051_963_027_995),
        (0.1, 0.097_375_278_148),
        (0.15, 0.131_039_532_110),
        (0.2, 0.178_839_486_607),
        (0.3, 0.307_966_051_363),
        (0.4, 0.442_248_031_414),
        (0.5, 0.555_594_702_383),
        (0.6, 0.654_202_874_989),
        (0.7, 0.745_597_508_054),
        (0.9, 0.915_668_934_751),
        (0.99, 0.991_473_620_644),
    ];
    for (x, want) in reference {
        let got = curve.eval(x);
        assert!(
            (got - want).abs() < 1e-9,
            "T({x}) = {got}, reference {want}"
        );
    }
}

/// Whole-look values of the same independent implementation: the tone shift from `L` (the LogC4
/// code of the AWG4 luminance), then the CDL saturation with Rec.709 weights on the codes.
#[test]
fn the_look_matches_the_prototype_on_colour_triples() {
    let look = signature();
    let reference: [(Vec3, Vec3); 12] = [
        (
            [0.2784, 0.2784, 0.2784],
            [0.278_401_380_022, 0.278_401_380_022, 0.278_401_380_022],
        ),
        (
            [0.5, 0.4, 0.3],
            [0.588_881_313_069, 0.443_881_313_069, 0.298_881_313_069],
        ),
        (
            [0.3, 0.5, 0.35],
            [0.288_829_407_110, 0.578_829_407_110, 0.361_329_407_110],
        ),
        (
            [0.3, 0.38, 0.52],
            [0.293_773_958_030, 0.409_773_958_030, 0.612_773_958_030],
        ),
        (
            [0.45, 0.2, 0.55],
            [0.535_624_027_917, 0.173_124_027_917, 0.680_624_027_917],
        ),
        (
            [0.6, 0.45, 0.35],
            [0.712_170_072_898, 0.494_670_072_898, 0.349_670_072_898],
        ),
        (
            [0.15, 0.2, 0.18],
            [0.110_865_092_856, 0.183_365_092_856, 0.154_365_092_856],
        ),
        (
            [0.7, 0.65, 0.6],
            [0.768_609_412_174, 0.696_109_412_174, 0.623_609_412_174],
        ),
        (
            [0.4, 0.2, 0.1],
            [0.482_286_778_147, 0.192_286_778_147, 0.047_286_778_147],
        ),
        (
            [0.1, 0.1, 0.1],
            [0.097_375_278_148, 0.097_375_278_148, 0.097_375_278_148],
        ),
        (
            [0.9, 0.85, 0.8],
            [0.940_456_439_720, 0.867_956_439_720, 0.795_456_439_720],
        ),
        (
            [0.35, 0.6, 0.2],
            [0.329_413_617_631, 0.691_913_617_631, 0.111_913_617_631],
        ),
    ];
    for (input, want) in reference {
        // The prototype holds `T` flat below 0 where the Curve tab continues with slope 1, so the
        // comparison is for colours whose luminance code is inside [0, 1].
        let l = LogC4(input).luma();
        assert!((0.0..=1.0).contains(&l), "{input:?}: L = {l}");
        let got = look.evaluate(input);
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 1e-9,
                "{input:?}: {got:?}, reference {want:?}"
            );
        }
    }
}

// ---- Mid grey, neutrals, monotony --------------------------------------------------------------

#[test]
fn mid_grey_is_unchanged_to_1e_minus_6() {
    let look = signature();
    let g = grey_code();
    assert!(
        (g - 0.278_395_836_548_265_3).abs() < 1e-15,
        "LogC4 of 0.18: {g}"
    );
    let out = look.evaluate([g; 3]);
    let shift = out[0] - g;
    println!("mid grey: exact look moves LogC4 {g:.9} by {shift:.2e}");
    for c in out {
        assert!((c - g).abs() < 1e-6, "{out:?}");
    }
    // The cube is a little less exact between its grid points; it is still far inside a display code.
    let baked = look.to_transform().expect("bakes");
    let b = baked.eval_logc4(LogC4([g; 3])).expect("log look").0;
    println!("mid grey: 65^3 cube moves it by {:.2e}", b[0] - g);
    assert!((b[0] - g).abs() < 5e-4, "{b:?}");
}

#[test]
fn neutrals_stay_neutral_exactly_and_in_the_cube() {
    let look = signature();
    let baked = look.to_transform().expect("bakes");
    for i in 0..=400 {
        let v = f64::from(i) / 400.0;
        let exact = look.evaluate([v; 3]);
        assert!(
            (exact[0] - exact[1]).abs() < 1e-12 && (exact[1] - exact[2]).abs() < 1e-12,
            "{v}: {exact:?}"
        );
        let b = baked.eval_logc4(LogC4([v; 3])).expect("log look").0;
        assert!(
            (b[0] - b[1]).abs() < 1e-12 && (b[1] - b[2]).abs() < 1e-12,
            "cube at {v}: {b:?}"
        );
    }
}

#[test]
fn the_neutral_axis_rises_strictly_exactly_and_in_the_cube() {
    let look = signature();
    let baked = look.to_transform().expect("bakes");
    let mut prev_exact = f64::NEG_INFINITY;
    let mut prev_baked = f64::NEG_INFINITY;
    for i in 0..=2000 {
        let v = f64::from(i) / 2000.0;
        let e = look.evaluate([v; 3])[1];
        let b = baked.eval_logc4(LogC4([v; 3])).expect("log look").0[1];
        assert!(e > prev_exact, "exact look not rising at {v}");
        assert!(b > prev_baked, "cube not rising at {v}");
        prev_exact = e;
        prev_baked = b;
    }
    // The tone curve T itself never goes down, anywhere on [0, 1].
    let curve = look.luma_curve().expect("curve");
    let mut prev = f64::NEG_INFINITY;
    for i in 0..=5000 {
        let t = curve.eval(f64::from(i) / 5000.0);
        assert!(t >= prev);
        prev = t;
    }
}

/// Raising one channel of a colour never lowers the luma-curve result in that channel, and a ramp
/// of any hue at constant ratio stays increasing in luminance (the saturation step mixes channels,
/// so only the combination, not each channel on its own, is required to rise).
#[test]
fn constant_ratio_ramps_rise_in_luminance() {
    let look = signature();
    let ratios: [Vec3; 6] = [
        [1.0, 0.5, 0.25],
        [0.25, 1.0, 0.5],
        [0.5, 0.25, 1.0],
        [1.0, 0.8, 0.8],
        [0.7, 1.0, 0.75],
        [0.8, 0.8, 1.0],
    ];
    for ratio in ratios {
        let mut prev = f64::NEG_INFINITY;
        for i in 1..=400 {
            let lin = f64::from(i) / 400.0 * 8.0;
            let v = [
                logc4_encode(lin * ratio[0]),
                logc4_encode(lin * ratio[1]),
                logc4_encode(lin * ratio[2]),
            ];
            if v.iter().any(|&c| !(0.0..=1.0).contains(&c)) {
                continue;
            }
            let out = look.evaluate(v);
            let y = REC709_LUMA_WEIGHTS[0] * out[0]
                + REC709_LUMA_WEIGHTS[1] * out[1]
                + REC709_LUMA_WEIGHTS[2] * out[2];
            assert!(y > prev, "ratio {ratio:?} at {lin}");
            prev = y;
        }
    }
}

// ---- The curve as a look field ---------------------------------------------------------------

#[test]
fn the_luma_curve_moves_all_three_channels_by_the_same_shift_of_t_of_l_minus_l() {
    // T lifts 0.5 to 0.6 and holds the ends.
    let look = ParametricLook::from_json(&spec(
        r#","luma_curve":[{"x":0,"y":0},{"x":0.5,"y":0.6},{"x":1,"y":1}]"#,
    ))
    .expect("parses");
    let curve = look.luma_curve().expect("curve");
    let v = [0.55, 0.45, 0.35];
    let l = LogC4(v).luma();
    let want = curve.eval(l) - l;
    let out = look.evaluate(v);
    for c in 0..3 {
        assert!((out[c] - (v[c] + want)).abs() < 1e-14, "{out:?}");
    }
    // The luminance of a neutral at x = 0.5 goes to exactly y = 0.6.
    let grey = look.evaluate([0.5; 3]);
    for c in grey {
        assert!((c - 0.6).abs() < 1e-12, "{grey:?}");
    }
}

/// `L` is the LogC4 code of the AWG4 luminance, not the Rec.709 luma of the codes (that is the
/// `luma` of `luma_vs_sat`): for a saturated colour the two differ a lot.
#[test]
fn the_curve_reads_the_awg4_luminance_not_the_rec709_luma_of_the_codes() {
    let look = ParametricLook::from_json(&spec(
        r#","luma_curve":[{"x":0,"y":0},{"x":0.5,"y":0.6},{"x":1,"y":1}]"#,
    ))
    .expect("parses");
    let curve = look.luma_curve().expect("curve");
    let v = [0.8, 0.3, 0.1];
    let l_awg4 = LogC4(v).luma();
    let l_709 = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    assert!((l_awg4 - l_709).abs() > 0.05, "{l_awg4} vs {l_709}");
    let shift = look.evaluate(v)[0] - v[0];
    assert!((shift - (curve.eval(l_awg4) - l_awg4)).abs() < 1e-14);
    assert!((shift - (curve.eval(l_709) - l_709)).abs() > 1e-3);
}

/// The curve adds one shift to the three codes, so the differences between the channels in LogC4
/// (the log ratios of the colour, which is what hue and chroma are made of) are not touched.
#[test]
fn the_curve_keeps_the_colour_differences_in_logc4() {
    let look = signature();
    for v in [
        [0.55, 0.45, 0.35],
        [0.4, 0.5, 0.6],
        [0.7, 0.4, 0.45],
        [0.3, 0.35, 0.3],
    ] {
        let moved = look.apply_luma_curve(v);
        for c in 1..3 {
            let (a, b) = (v[c] - v[0], moved[c] - moved[0]);
            assert!((a - b).abs() < 1e-15, "{v:?}: {a} vs {b}");
        }
        assert!(moved != v, "{v:?} is moved by the curve");
    }
    // Far above black a common shift of the codes is nearly a common gain of the linear light
    // (LogC4 is 2^code times a constant there, less a small offset), so the linear ratios of a
    // bright colour stay within a few percent.
    for v in [[0.75, 0.7, 0.65], [0.7, 0.75, 0.68]] {
        let moved = look.apply_luma_curve(v);
        for c in 1..3 {
            let before = logc4_decode(v[c]) / logc4_decode(v[0]);
            let after = logc4_decode(moved[c]) / logc4_decode(moved[0]);
            assert!(
                (after / before - 1.0).abs() < 0.01,
                "{v:?}: {before} to {after}"
            );
        }
    }
}

#[test]
fn outside_the_unit_range_the_curve_continues_with_slope_one() {
    // T(0) = 0.05 and T(1) = 0.9, so below 0 the shift stays +0.05 and above 1 it stays -0.1.
    let look = ParametricLook::from_json(&spec(
        r#","luma_curve":[{"x":0,"y":0.05},{"x":0.5,"y":0.5},{"x":1,"y":0.9}]"#,
    ))
    .expect("parses");
    let low = look.apply_luma_curve([-0.05, -0.05, -0.05]);
    let l_low = LogC4([-0.05; 3]).luma();
    assert!(l_low < 0.0);
    assert!((low[0] - (-0.05 + 0.05)).abs() < 1e-12, "{low:?}");
    let high = look.apply_luma_curve([1.2, 1.2, 1.2]);
    assert!(LogC4([1.2; 3]).luma() > 1.0);
    assert!((high[0] - (1.2 - 0.1)).abs() < 1e-12, "{high:?}");
    // Inputs far outside any LogC4 signal give a finite answer (no NaN from inf - inf).
    for v in [[1e6; 3], [-1e6; 3], [1e300, 0.0, -1e300]] {
        let out = look.evaluate(v);
        assert!(out.iter().all(|c| c.is_finite()), "{v:?} -> {out:?}");
    }
}

#[test]
fn the_curve_runs_before_the_cdl_and_the_colour_curves() {
    let json = spec(
        r#","luma_curve":[{"x":0,"y":0},{"x":0.5,"y":0.6},{"x":1,"y":1}],"cdl":{"slope":[1.1,1,1],"offset":[0,0,0],"power":[1,1,1],"saturation":1.3}"#,
    );
    let look = ParametricLook::from_json(&json).expect("parses");
    let v = [0.5, 0.45, 0.4];
    let by_hand = look.cdl().apply(look.apply_luma_curve(v));
    assert_eq!(look.evaluate(v), by_hand);
    let swapped = look.apply_luma_curve(look.cdl().apply(v));
    assert!((swapped[0] - by_hand[0]).abs() > 1e-4, "the order matters");
}

#[test]
fn a_curve_on_the_diagonal_is_the_identity_look() {
    let look = ParametricLook::from_json(&spec(
        r#","luma_curve":[{"x":0,"y":0},{"x":0.4,"y":0.4},{"x":1,"y":1}]"#,
    ))
    .expect("parses");
    assert!(look.is_identity());
    assert!(look.luma_curve().is_none());
    assert!(look.to_transform().expect("bakes").is_identity());
    // A curve that is on y = x between its points but stops short of the ends is not the identity:
    // the curve is flat outside the first and last point.
    let short = ParametricLook::from_json(&spec(
        r#","luma_curve":[{"x":0.2,"y":0.2},{"x":0.8,"y":0.8}]"#,
    ))
    .expect("parses");
    assert!(!short.is_identity());
}

#[test]
fn the_curve_field_is_validated() {
    let pts = |n: usize| -> String {
        let p: Vec<String> = (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1).max(1) as f64;
                format!(r#"{{"x":{x},"y":{}}}"#, x * 0.9)
            })
            .collect();
        format!(r#","luma_curve":[{}]"#, p.join(","))
    };
    assert!(ParametricLook::from_json(&spec(&pts(16))).is_ok());
    for (text, why) in [
        (spec(&pts(17)), "17 points"),
        (
            spec(r#","luma_curve":[{"x":0.5,"y":0.5}]"#),
            "one point is not a curve",
        ),
        (
            spec(r#","luma_curve":[{"x":0,"y":0},{"x":1,"y":1.1}]"#),
            "y above 1",
        ),
        (
            spec(r#","luma_curve":[{"x":-0.1,"y":0},{"x":1,"y":1}]"#),
            "x below 0",
        ),
        (
            spec(r#","luma_curve":[{"x":0,"y":0},{"x":0.5,"y":0.5},{"x":0.5,"y":0.6}]"#),
            "x does not increase",
        ),
        (
            spec(r#","luma_curve":[{"x":0,"y":0},{"x":0.5,"y":0.5,"z":1}]"#),
            "an unknown field in a point",
        ),
        (
            spec(r#","luma_curve":[{"x":0,"y":0},{"x":0.5}]"#),
            "a point without y",
        ),
    ] {
        let err = ParametricLook::from_json(&text).expect_err(why);
        match (&err, why) {
            (LookError::BadCurve { section, reason }, _) => {
                assert_eq!(*section, "luma_curve", "{why}");
                assert!(!reason.is_empty());
            }
            (LookError::Json(_), "an unknown field in a point" | "a point without y") => {}
            other => panic!("{why}: {other:?}"),
        }
        assert!(!err.to_string().is_empty());
    }
}

#[test]
fn the_curve_survives_a_json_round_trip() {
    use ym_color::look::ParametricLookSpec;
    let parsed: ParametricLookSpec = serde_json::from_str(SIGNATURE_JSON).expect("spec");
    assert_eq!(parsed.luma_curve.len(), 15);
    let text = serde_json::to_string(&parsed).expect("serialises");
    let again = ParametricLook::from_json(&text).expect("parses again");
    assert_eq!(again, signature());
    // A look without the field does not write one.
    let plain: ParametricLookSpec = serde_json::from_str(&spec("")).expect("spec");
    assert!(
        !serde_json::to_string(&plain)
            .expect("json")
            .contains("luma_curve")
    );
}

// ---- Intensity ----------------------------------------------------------------------------------

#[test]
fn intensity_zero_is_neutral_and_one_hundred_is_the_look_exactly() {
    let baked = Arc::new(signature().to_transform().expect("bakes"));
    let mut rng = Rng(7);
    for _ in 0..500 {
        let v = LogC4([rng.next(), rng.next(), rng.next()]);
        assert_eq!(baked.apply_logc4(v, 0.0).expect("log look"), v);
        assert_eq!(
            baked.apply_logc4(v, 100.0).expect("log look"),
            baked.eval_logc4(v).expect("log look"),
            "100 % is the look itself"
        );
        let half = baked.apply_logc4(v, 50.0).expect("log look").0;
        let full = baked.eval_logc4(v).expect("log look").0;
        for c in 0..3 {
            assert!((half[c] - (v.0[c] + 0.5 * (full[c] - v.0[c]))).abs() < 1e-15);
        }
    }
}

fn plain_chain(look: Option<LookSetting>) -> ColorChain {
    // Camera RGB is linear AWG4 (identity matrix), C7 off, no grade, no curves: only the look and the
    // display transform (ACES 2 with the placement) act, sRGB output, no dither.
    let identity: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tone = ToneParams::IDENTITY;
    let mut p = ChainParams::new(identity, tone);
    p.look = look;
    p.output = OutputTarget::Srgb;
    p.dither = false;
    ColorChain::new(&p).expect("chain")
}

#[test]
fn in_the_chain_intensity_zero_is_the_bits_of_no_look_and_one_hundred_is_the_cube() {
    let sig = Arc::new(signature().to_transform().expect("bakes"));
    let none = plain_chain(None);
    let zero = plain_chain(Some(LookSetting::new(Arc::clone(&sig), 0.0)));
    let neutral = plain_chain(Some(LookSetting::new(
        Arc::new(LookTransform::neutral()),
        100.0,
    )));
    let full = plain_chain(Some(LookSetting::new(Arc::clone(&sig), 100.0)));
    let mut rng = Rng(11);
    let mut moved = 0;
    for _ in 0..400 {
        let cam: Vec3 = [
            2.0_f64.powf(rng.next() * 8.0 - 5.0) * rng.next(),
            2.0_f64.powf(rng.next() * 8.0 - 5.0) * rng.next(),
            2.0_f64.powf(rng.next() * 8.0 - 5.0) * rng.next(),
        ];
        let a = none.trace(cam);
        assert_eq!(zero.trace(cam), a, "intensity 0 is exactly no look");
        assert_eq!(neutral.trace(cam), a, "Neutral is exactly no look");
        let f = full.trace(cam);
        assert_eq!(
            f.c10_logc4,
            sig.eval_logc4(f.c9_logc4).expect("log look"),
            "C10 at 100 % is the cube"
        );
        if f.c13_encoded != a.c13_encoded {
            moved += 1;
        }
    }
    assert!(moved > 300, "the look changes the picture ({moved} of 400)");
}

// ---- The bake against the exact look, in display codes ---------------------------------------

fn rec709_to_awg4() -> Mat3 {
    linalg::mul(&matrices().xyz_to_awg4, &matrices().rec709_to_xyz)
}

/// Worst `|cube - exact|` in 8-bit sRGB display codes over `scene` (linear AWG4 triples), through
/// the real chain: exact look = the exact LogC4 values re-entered as scene light with no look;
/// cube = the baked look at 100 %.
fn worst_display_code_error(scene: &[Vec3]) -> (f64, Vec3) {
    let look = signature();
    let cube = plain_chain(Some(LookSetting::new(
        Arc::new(look.to_transform().expect("bakes")),
        100.0,
    )));
    let reference = plain_chain(None);
    let mut worst = 0.0_f64;
    let mut at = [0.0; 3];
    for &awg4 in scene {
        let v = [
            logc4_encode(awg4[0]),
            logc4_encode(awg4[1]),
            logc4_encode(awg4[2]),
        ];
        // The cube only exists on [0, 1]^3; scene values outside it are not looks' business.
        if v.iter().any(|c| !(0.0..=1.0).contains(c)) {
            continue;
        }
        let exact = look.evaluate(v);
        let exact_light = [
            logc4_decode(exact[0]),
            logc4_decode(exact[1]),
            logc4_decode(exact[2]),
        ];
        let want = reference.trace(exact_light).c13_encoded;
        let got = cube.trace(awg4).c13_encoded;
        for c in 0..3 {
            let d = (got[c] - want[c]).abs() * 255.0;
            if d > worst {
                worst = d;
                at = awg4;
            }
        }
    }
    (worst, at)
}

fn neutral_ramp() -> Vec<Vec3> {
    (0..=560)
        .map(|i| {
            let stops = -8.0 + f64::from(i) / 40.0;
            let y = 0.18 * stops.exp2();
            [y; 3]
        })
        .collect()
}

/// Hues at four saturations and eight exposures, in linear Rec.709 (so inside every display
/// gamut), moved to AWG4: a swept stand-in for a colour chart.
fn colour_sweep() -> Vec<Vec3> {
    let m = rec709_to_awg4();
    let mut out = Vec::new();
    for h in 0..48 {
        let hue = f64::from(h) * 7.5_f64.to_radians();
        // A pure hue with the largest channel 1 and the smallest 0 (HSV with S = V = 1).
        let sector = hue / 60.0_f64.to_radians();
        let f = sector - sector.floor();
        let (r, g, b) = match sector.floor() as i32 {
            0 => (1.0, f, 0.0),
            1 => (1.0 - f, 1.0, 0.0),
            2 => (0.0, 1.0, f),
            3 => (0.0, 1.0 - f, 1.0),
            4 => (f, 0.0, 1.0),
            _ => (1.0, 0.0, 1.0 - f),
        };
        for s in [0.15, 0.4, 0.7, 1.0] {
            for stop in -4..=3 {
                let gain = 0.18 * f64::from(stop).exp2();
                let c = [
                    gain * (1.0 - s + s * r),
                    gain * (1.0 - s + s * g),
                    gain * (1.0 - s + s * b),
                ];
                out.push(linalg::mul_vec(&m, c));
            }
        }
    }
    out
}

/// Random colours anywhere in the Rec.709 cube at random exposures: the harsh case, with many
/// colours on the gamut boundary.
fn random_scene(n: usize, seed: u64) -> Vec<Vec3> {
    let m = rec709_to_awg4();
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let gain = 0.18 * (rng.next() * 9.0 - 6.0).exp2();
            let c = [rng.next() * gain, rng.next() * gain, rng.next() * gain];
            linalg::mul_vec(&m, c)
        })
        .collect()
}

#[test]
fn the_baked_cube_follows_the_exact_look_to_one_display_code_on_synthetic_scenes() {
    let scenes: [(&str, Vec<Vec3>); 3] = [
        ("neutral ramp -8 to +6 stops", neutral_ramp()),
        ("hue sweep, 4 saturations x 8 exposures", colour_sweep()),
        (
            "20000 random Rec.709 colours",
            random_scene(20_000, 0x51_67),
        ),
    ];
    for (name, scene) in scenes {
        let (worst, at) = worst_display_code_error(&scene);
        println!(
            "65^3 cube vs exact look, {name}: worst {worst:.3} display codes (8 bit sRGB) at {at:?}"
        );
        assert!(worst <= 1.0, "{name}: {worst:.3} codes at {at:?}");
    }
}

/// The cube against the exact look in LogC4 code values, for the record. Near-neutral colours (the
/// ones a photo is made of) are followed to about 6e-4. The largest errors are strongly saturated
/// colours with one channel far above the others (here a blue code of 0.93 next to a red of 0.34, a
/// ratio of hundreds in linear light): the AWG4 luminance row has a negative blue entry, so `L` falls and bends quickly
/// there, finer than the 1/64 grid. No display shows such a colour.
#[test]
fn the_baked_cube_error_in_logc4_values() {
    let look = signature();
    let baked = look.to_transform().expect("bakes");
    let mut rng = Rng(99);
    let (mut worst, mut worst_at) = (0.0_f64, [0.0; 3]);
    let mut worst_neutralish = 0.0_f64;
    for _ in 0..30_000 {
        let v = [rng.next(), rng.next(), rng.next()];
        let exact = look.evaluate(v);
        let got = baked.eval_logc4(LogC4(v)).expect("log look").0;
        let e = (0..3)
            .map(|c| (exact[c] - got[c]).abs())
            .fold(0.0, f64::max);
        if e > worst {
            worst = e;
            worst_at = v;
        }
        let hi = v.iter().copied().fold(f64::MIN, f64::max);
        let lo = v.iter().copied().fold(f64::MAX, f64::min);
        if hi - lo < 0.1 {
            worst_neutralish = worst_neutralish.max(e);
        }
    }
    println!(
        "65^3 cube of Signature: worst |cube - exact| {worst:.2e} at {worst_at:?}; near-neutral colours {worst_neutralish:.2e} (LogC4 code values)"
    );
    assert!(worst < 5e-2, "{worst:e}");
    assert!(worst_neutralish < 2e-3, "{worst_neutralish:e}");
}

// ---- The curve type as the Curve tab has it ---------------------------------------------------

#[test]
fn the_look_curve_is_the_curve_tab_spline() {
    // The same points through `ym_color::curve::Curve` give the same T the look uses.
    let look = signature();
    let from_look = look.luma_curve().expect("curve");
    let points: Vec<(f64, f64)> = from_look.points().iter().map(|p| (p.x, p.y)).collect();
    let again = Curve::new(&points).expect("curve");
    for i in 0..=1000 {
        let x = f64::from(i) / 1000.0;
        assert_eq!(again.eval(x), from_look.eval(x));
    }
}
