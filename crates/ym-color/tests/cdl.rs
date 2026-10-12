//! ASC CDL v1.2 behaviour. Expected values were computed independently in double precision from
//! the ASC formulas (ASC CDL Transfer Functions and Interchange Syntax, release 1.2, 2009) and, for
//! outputs inside [0, 1], cross-checked against OpenColorIO 2.6 `CDLTransform` in `CDL_ASC` style
//! (float32 with fast pow, agreeing to ~2e-5).

use ym_color::cdl::Cdl;

fn assert_rgb(got: [f64; 3], want: [f64; 3], tol: f64) {
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() < tol,
            "channel {k}: {got:?} vs {want:?}"
        );
    }
}

#[test]
fn slope_offset_power() {
    // in (0.5, 0.4, 0.3), slope (1.2, 1.0, 0.8), offset (0.01, 0, -0.02), power (0.9, 1.0, 1.1), sat 1:
    // R = (0.5 * 1.2 + 0.01)^0.9 = 0.61^0.9, G = 0.4, B = (0.3 * 0.8 - 0.02)^1.1 = 0.22^1.1.
    let cdl = Cdl::new([1.2, 1.0, 0.8], [0.01, 0.0, -0.02], [0.9, 1.0, 1.1], 1.0).unwrap();
    assert_rgb(
        cdl.apply([0.5, 0.4, 0.3]),
        [0.640_909_710_246_052_1, 0.4, 0.189_088_426_947_216_14],
        1e-12,
    );
}

#[test]
fn saturation_uses_rec709_weights() {
    let v = [0.2, 0.5, 0.8];
    let luma = 0.2126 * 0.2 + 0.7152 * 0.5 + 0.0722 * 0.8; // 0.45788
    let zero = Cdl {
        saturation: 0.0,
        ..Cdl::identity()
    };
    assert_rgb(zero.apply(v), [luma; 3], 1e-15);
    assert!((luma - 0.45788).abs() < 1e-15);
    let boost = Cdl {
        saturation: 1.5,
        ..Cdl::identity()
    };
    assert_rgb(boost.apply(v), [0.07106, 0.52106, 0.97106], 1e-12);
    // Saturation keeps the Rec.709 luma.
    let out = boost.apply(v);
    assert!((0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2] - luma).abs() < 1e-15);
}

#[test]
fn all_four_parameters_together() {
    // slope (0.9, 1.1, 1.3), offset (-0.05, 0.02, 0.1), power (1.2, 0.8, 1.0), sat 1.25.
    let cdl = Cdl::new([0.9, 1.1, 1.3], [-0.05, 0.02, 0.1], [1.2, 0.8, 1.0], 1.25).unwrap();
    assert_rgb(
        cdl.apply([0.7, 0.3, 0.1]),
        [0.541_167_684, 0.430_717_607, 0.178_502_656],
        5e-9,
    );
}

#[test]
fn power_sees_no_negative_base() {
    // Offset pushes red below zero: it is clamped at 0 before the power (and before saturation).
    let cdl = Cdl::new([2.0, 0.5, 1.0], [-0.2, 0.0, 0.0], [1.0; 3], 0.7).unwrap();
    // After SOP: (0, 0.05, 0.9) with luma 0.0357 + 0.0650 = 0.1007...; sat 0.7.
    assert_rgb(
        cdl.apply([0.05, 0.1, 0.9]),
        [0.030_222, 0.065_222, 0.660_222],
        1e-9,
    );
    // A pure power with negative input stays at zero, never NaN.
    let p = Cdl::new([1.0; 3], [0.0; 3], [2.0, 0.5, 3.0], 1.0).unwrap();
    assert_eq!(p.apply([-0.3, -1.0, -0.0]), [0.0, 0.0, 0.0]);
}

#[test]
fn values_above_one_are_not_clamped() {
    // (0.9, 0.8, 0.7) * 1.5 + 0.1, ^0.8: above 1. ASC-clamp style would return 1.0.
    let cdl = Cdl::new([1.5; 3], [0.1; 3], [0.8; 3], 1.0).unwrap();
    let out = cdl.apply([0.9, 0.8, 0.7]);
    assert_rgb(out, [1.346_152_933, 1.233_544_104, 1.118_299_865], 5e-9);
    assert!(out.iter().all(|&c| c > 1.0));
}

#[test]
fn negative_results_after_saturation_pass_through() {
    // Strong saturation can push a channel below zero; there is no clamp after saturation.
    let cdl = Cdl::new([1.0; 3], [-0.3, -0.1, 0.0], [2.0; 3], 1.3).unwrap();
    let out = cdl.apply([0.1, 0.0, 0.6]);
    assert_rgb(out, [-0.007_797_6, -0.007_797_6, 0.460_202_4], 1e-9);
}

#[test]
fn slope_zero_and_power_one_shortcut_agree_with_powf() {
    let a = Cdl::new([1.3, 0.0, 0.7], [0.05, 0.2, -0.1], [1.0, 1.0, 1.0], 0.9).unwrap();
    let b = Cdl::new(
        [1.3, 0.0, 0.7],
        [0.05, 0.2, -0.1],
        [1.0 + 1e-12, 1.0, 1.0],
        0.9,
    )
    .unwrap();
    assert_rgb(a.apply([0.4, 0.9, 0.5]), b.apply([0.4, 0.9, 0.5]), 1e-11);
}

#[test]
fn cdl_is_monotone_per_channel_without_saturation_change() {
    // With saturation 1 each channel is a monotone function of its own input (slope >= 0, power > 0).
    let cdl = Cdl::new([1.4, 0.9, 2.0], [-0.1, 0.05, 0.0], [0.7, 1.8, 1.0], 1.0).unwrap();
    let mut prev = [f64::NEG_INFINITY; 3];
    for k in 0..=1000 {
        let x = f64::from(k) / 500.0 - 0.5;
        let out = cdl.apply([x; 3]);
        for c in 0..3 {
            assert!(out[c] >= prev[c]);
            prev[c] = out[c];
        }
    }
}
