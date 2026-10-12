//! M1-09/C11: the Fritsch-Carlson curve: identity, interpolation, monotonicity, endpoint behaviour,
//! the 16-point limit, baked tables and luma application.

use proptest::prelude::*;
use ym_color::consts::LOGC4_GRAY;
use ym_color::curve::{
    BakedCurves, Curve, CurveError, CurveSet, MAX_CONTROL_POINTS, Point, TABLE_SIZE,
};
use ym_color::logc::LogC4;

fn curve(pts: &[(f64, f64)]) -> Curve {
    Curve::new(pts).unwrap()
}

#[test]
fn identity_curve_and_table() {
    let c = Curve::identity();
    assert!(c.is_identity());
    for k in 0..=1000 {
        let x = f64::from(k) / 1000.0;
        assert_eq!(c.eval(x), x);
    }
    let t = c.bake();
    assert_eq!(t.values().len(), TABLE_SIZE);
    assert_eq!(TABLE_SIZE, 1024);
    for (i, v) in t.values().iter().enumerate() {
        assert!((v - i as f64 / 1023.0).abs() < 1e-15);
    }
    // Explicit identity points through a third control point are still the identity.
    let c3 = curve(&[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]);
    for k in 0..=100 {
        let x = f64::from(k) / 100.0;
        assert!((c3.eval(x) - x).abs() < 1e-15);
    }
    // A default curve set is the identity everywhere.
    let set = CurveSet::default();
    assert!(set.is_identity());
    let baked = set.bake();
    let v = LogC4([0.7, -0.1, 1.3]);
    assert_eq!(baked.apply(v), v);
}

#[test]
fn passes_through_every_control_point() {
    let pts = [
        (0.0, 0.05),
        (0.15, 0.1),
        (0.4, 0.45),
        (0.6, 0.62),
        (0.85, 0.9),
        (1.0, 0.97),
    ];
    let c = curve(&pts);
    for (x, y) in pts {
        assert!((c.eval(x) - y).abs() < 1e-15, "({x}, {y})");
    }
    assert_eq!(c.points().len(), 6);
    assert_eq!(c.points()[2], Point { x: 0.4, y: 0.45 });
}

#[test]
fn rejects_bad_input() {
    assert_eq!(
        Curve::new::<(f64, f64)>(&[]),
        Err(CurveError::TooFewPoints(0))
    );
    assert_eq!(
        Curve::new(&[(0.0, 0.0), (0.0, 1.0)]),
        Err(CurveError::NotIncreasing { index: 1 })
    );
    assert_eq!(
        Curve::new(&[(0.5, 0.0), (0.2, 1.0)]),
        Err(CurveError::NotIncreasing { index: 1 })
    );
    assert!(matches!(
        Curve::new(&[(0.0, 0.0), (1.0, 1.5)]),
        Err(CurveError::OutOfRange { index: 1, .. })
    ));
    assert!(matches!(
        Curve::new(&[(-0.1, 0.0), (1.0, 1.0)]),
        Err(CurveError::OutOfRange { index: 0, .. })
    ));
    assert!(matches!(
        Curve::new(&[(0.0, f64::NAN), (1.0, 1.0)]),
        Err(CurveError::OutOfRange { .. })
    ));
}

#[test]
fn sixteen_point_limit() {
    let make = |n: usize| -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let x = i as f64 / (n as f64 - 1.0);
                (x, x * x)
            })
            .collect()
    };
    assert_eq!(MAX_CONTROL_POINTS, 16);
    assert!(Curve::new(&make(16)).is_ok());
    assert_eq!(Curve::new(&make(17)), Err(CurveError::TooManyPoints(17)));
    assert_eq!(Curve::new(&make(40)), Err(CurveError::TooManyPoints(40)));
    // A 16-point curve behaves.
    let c = curve(&make(16));
    for k in 0..=500 {
        let x = f64::from(k) / 500.0;
        assert!((c.eval(x) - x * x).abs() < 5e-3);
    }
}

#[test]
fn monotone_data_gives_a_monotone_curve() {
    // A steep S-curve, a plateau and a near-vertical step: the cases that make a plain cubic overshoot.
    let sets: [&[(f64, f64)]; 4] = [
        &[
            (0.0, 0.0),
            (0.25, 0.05),
            (0.5, 0.5),
            (0.75, 0.95),
            (1.0, 1.0),
        ],
        &[(0.0, 0.0), (0.3, 0.3), (0.6, 0.3), (1.0, 1.0)],
        &[(0.0, 0.0), (0.49, 0.1), (0.5, 0.9), (1.0, 1.0)],
        &[(0.0, 0.2), (0.1, 0.2), (0.2, 0.9), (0.9, 0.95), (1.0, 1.0)],
    ];
    for pts in sets {
        let c = curve(pts);
        let mut prev = c.eval(0.0);
        for k in 1..=20_000 {
            let y = c.eval(f64::from(k) / 20_000.0);
            assert!(y >= prev - 1e-14, "{pts:?}: decreased at {k}");
            prev = y;
        }
    }
}

#[test]
fn never_overshoots_the_data_of_a_segment() {
    let pts = [
        (0.0, 0.0),
        (0.2, 0.7),
        (0.4, 0.2),
        (0.6, 0.9),
        (0.8, 0.1),
        (1.0, 1.0),
    ];
    let c = curve(&pts);
    for w in pts.windows(2) {
        let (lo, hi) = (w[0].1.min(w[1].1), w[0].1.max(w[1].1));
        for k in 0..=200 {
            let x = w[0].0 + (w[1].0 - w[0].0) * f64::from(k) / 200.0;
            let y = c.eval(x);
            assert!(
                y >= lo - 1e-12 && y <= hi + 1e-12,
                "x={x}: {y} outside [{lo}, {hi}]"
            );
        }
    }
}

#[test]
fn is_c1_continuous_at_the_control_points() {
    let pts = [(0.0, 0.0), (0.2, 0.1), (0.5, 0.55), (0.7, 0.6), (1.0, 1.0)];
    let c = curve(&pts);
    let h = 1e-6;
    for &(x, _) in &pts[1..4] {
        let left = (c.eval(x) - c.eval(x - h)) / h;
        let right = (c.eval(x + h) - c.eval(x)) / h;
        assert!(
            (left - right).abs() < 1e-3,
            "slope jump at {x}: {left} vs {right}"
        );
    }
}

#[test]
fn endpoint_behaviour() {
    // Raised black point: flat below the first control point, continuous at it.
    let c = curve(&[(0.1, 0.2), (1.0, 1.0)]);
    assert_eq!(c.eval(0.0), 0.2);
    assert_eq!(c.eval(0.05), 0.2);
    assert!((c.eval(0.1) - 0.2).abs() < 1e-15);
    // Lowered white point: flat above the last control point.
    let c = curve(&[(0.0, 0.0), (0.9, 0.8)]);
    assert_eq!(c.eval(1.0), 0.8);
    assert!((c.eval(0.9) - 0.8).abs() < 1e-15);
    // Two points are a straight line.
    let c = curve(&[(0.0, 0.1), (1.0, 0.7)]);
    for k in 0..=10 {
        let x = f64::from(k) / 10.0;
        assert!((c.eval(x) - (0.1 + 0.6 * x)).abs() < 1e-15);
    }
    assert_eq!(c.eval(f64::NAN), 0.1);
}

#[test]
fn table_matches_the_exact_curve() {
    let c = curve(&[
        (0.0, 0.0),
        (0.25, 0.18),
        (0.5, 0.5),
        (0.8, 0.85),
        (1.0, 1.0),
    ]);
    let t = c.bake();
    assert_eq!(t.values().len(), 1024);
    let mut worst = 0.0_f64;
    for k in 0..=100_000 {
        let x = f64::from(k) / 100_000.0;
        worst = worst.max((t.lookup(x) - c.eval(x)).abs());
    }
    println!("max |table - curve| with linear interpolation: {worst:e}");
    assert!(worst < 2e-6, "{worst:e}");
    // Exact at table nodes.
    for i in [0, 1, 17, 511, 1022, 1023] {
        assert!((t.values()[i] - c.eval(i as f64 / 1023.0)).abs() < 1e-15);
    }
    // f32 form for the 1D texture.
    assert_eq!(t.to_f32().len(), 1024);
}

#[test]
fn extension_outside_the_table_has_unit_slope() {
    let c = curve(&[(0.0, 0.1), (0.5, 0.4), (1.0, 0.9)]);
    let t = c.bake();
    // Continuous at 0 and 1.
    assert!((t.eval_extended(-1e-12) - t.eval_extended(0.0)).abs() < 1e-11);
    assert!((t.eval_extended(1.0 + 1e-12) - t.eval_extended(1.0)).abs() < 1e-11);
    // Slope 1 beyond, carrying the offset of the end value.
    assert!((t.eval_extended(-0.5) - (-0.5 + 0.1)).abs() < 1e-15);
    assert!((t.eval_extended(1.75) - (1.75 - 1.0 + 0.9)).abs() < 1e-15);
    assert_eq!(t.lookup(-3.0), t.values()[0]);
    assert_eq!(t.lookup(9.0), t.values()[1023]);
    assert_eq!(t.lookup(f64::NAN), t.values()[0]);
}

#[test]
fn luma_curve_is_applied_as_a_shift() {
    // A grey pixel: the luma curve acts like the per-channel curve.
    let pts = [(0.0, 0.0), (0.3, 0.2), (0.7, 0.8), (1.0, 1.0)];
    let baked = CurveSet {
        luma: curve(&pts),
        ..Default::default()
    }
    .bake();
    let v = LOGC4_GRAY;
    let out = baked.apply(LogC4([v; 3])).0;
    let want = baked.luma.eval_extended(v);
    for c in out {
        assert!((c - want).abs() < 1e-12);
    }
    // A coloured pixel moves all channels by the same amount: chroma differences are kept.
    let rgb = LogC4([0.55, 0.40, 0.25]);
    let out = baked.apply(rgb).0;
    let shift = out[0] - rgb.0[0];
    assert!((out[1] - rgb.0[1] - shift).abs() < 1e-14 && (out[2] - rgb.0[2] - shift).abs() < 1e-14);
    assert!((shift - (baked.luma.eval_extended(rgb.luma()) - rgb.luma())).abs() < 1e-14);
}

#[test]
fn channel_curves_act_per_channel() {
    let pts = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)];
    let baked = CurveSet {
        g: curve(&pts),
        ..Default::default()
    }
    .bake();
    let v = LogC4([0.2, 0.5, 0.9]);
    let out = baked.apply(v).0;
    assert_eq!(out[0], 0.2);
    assert_eq!(out[2], 0.9);
    assert!((out[1] - baked.g.lookup(0.5)).abs() < 1e-15);
    assert!(out[1] > 0.55);
}

#[test]
fn baked_curves_keep_the_type() {
    let b: BakedCurves = CurveSet::default().bake();
    assert!(b.luma.is_identity());
}

proptest! {
    /// Any set of control points: the curve interpolates them and stays within the data range
    /// (no overshoot beyond the global min and max).
    #[test]
    fn random_control_points(
        raw in proptest::collection::vec((0.0f64..=1.0, 0.0f64..=1.0), 2..=16),
    ) {
        let mut pts = raw;
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
        prop_assume!(pts.len() >= 2);
        let c = Curve::new(&pts).unwrap();
        let lo = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        for &(x, y) in &pts {
            prop_assert!((c.eval(x) - y).abs() < 1e-12);
        }
        for k in 0..=200 {
            let y = c.eval(f64::from(k) / 200.0);
            prop_assert!(y >= lo - 1e-12 && y <= hi + 1e-12);
        }
    }

    /// Monotone (non-decreasing) control data always produce a non-decreasing curve.
    #[test]
    fn monotone_random_data(
        xs in proptest::collection::vec(0.0f64..=1.0, 2..=16),
        ys in proptest::collection::vec(0.0f64..=1.0, 16),
    ) {
        let mut x = xs;
        x.sort_by(f64::total_cmp);
        x.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        prop_assume!(x.len() >= 2);
        let mut y: Vec<f64> = ys[..x.len()].to_vec();
        y.sort_by(f64::total_cmp);
        let pts: Vec<(f64, f64)> = x.into_iter().zip(y).collect();
        let c = Curve::new(&pts).unwrap();
        let mut prev = c.eval(0.0);
        for k in 1..=500 {
            let v = c.eval(f64::from(k) / 500.0);
            prop_assert!(v >= prev - 1e-12);
            prev = v;
        }
    }
}

// ---- Independent implementation ------------------------------------------------------------

/// Control points and the values of an independent numpy implementation of Fritsch-Carlson
/// (the Wikipedia formulation) at `FC_EVAL_X`, for five curves including a plateau and a near-vertical step.
type FcCase = (&'static [(f64, f64)], [f64; 13]);
const FC_EVAL_X: [f64; 13] = [
    0.0, 0.03, 0.11, 0.25, 0.37, 0.48, 0.495, 0.5, 0.62, 0.7, 0.81, 0.93, 1.0,
];
const FC_CASES: [FcCase; 5] = [
    (
        &[
            (0.0, 0.0),
            (0.25, 0.05),
            (0.5, 0.5),
            (0.75, 0.95),
            (1.0, 1.0),
        ],
        [
            0.0,
            0.002857014282564874,
            0.00863413813828006,
            0.05,
            0.2266843290588914,
            0.46221644885277,
            0.4908812581437306,
            0.5,
            0.7522913885610233,
            0.8987728510267342,
            0.9763596253482755,
            0.9944679753100735,
            1.0,
        ],
    ),
    (
        &[
            (0.0, 0.1),
            (0.2, 0.7),
            (0.4, 0.2),
            (0.6, 0.9),
            (0.8, 0.1),
            (1.0, 1.0),
        ],
        [
            0.1,
            0.201475,
            0.5116749999999999,
            0.621875,
            0.23037499999999994,
            0.4463999999999999,
            0.523771875,
            0.55,
            0.8776,
            0.5000000000000003,
            0.10438750000000002,
            0.6133375000000001,
            1.0,
        ],
    ),
    (
        &[(0.05, 0.3), (0.3, 0.3), (0.35, 0.9), (0.9, 0.95)],
        [
            0.3,
            0.3,
            0.30000000000000004,
            0.3,
            0.9052551560942024,
            0.9276360777152868,
            0.9299181702649301,
            0.9306418907854375,
            0.9431239919320289,
            0.9472622430889243,
            0.9495223758614447,
            0.95,
            0.95,
        ],
    ),
    (
        &[(0.0, 0.0), (0.49, 0.1), (0.5, 0.9), (1.0, 1.0)],
        [
            0.0,
            0.00010533949139769302,
            0.001337605330440788,
            0.0134683908802848,
            0.04312402199507382,
            0.09400234793196136,
            0.5000153055406209,
            0.9,
            0.9560363833244183,
            0.9782558235887111,
            0.9942939070709584,
            0.9995706103546637,
            1.0,
        ],
    ),
    (
        &[
            (0.0, 0.0),
            (0.1, 0.02),
            (0.3, 0.35),
            (0.55, 0.5),
            (0.7, 0.8),
            (0.85, 0.82),
            (1.0, 1.0),
        ],
        [
            0.0,
            0.0024893296847301955,
            0.027150822705856878,
            0.27229482968059926,
            0.40117280000000005,
            0.4400072,
            0.44853154999999995,
            0.4518,
            0.6549970607080889,
            0.8,
            0.8145847978881782,
            0.8987868370574444,
            1.0,
        ],
    ),
];

#[test]
fn matches_an_independent_fritsch_carlson_implementation() {
    for (points, want) in FC_CASES {
        let c = Curve::new(points).unwrap();
        for (x, w) in FC_EVAL_X.iter().zip(want) {
            assert!(
                (c.eval(*x) - w).abs() < 1e-12,
                "{points:?} at {x}: {} vs {w}",
                c.eval(*x)
            );
        }
    }
}
