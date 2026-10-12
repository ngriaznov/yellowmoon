//! M1-09 (C7): highlight roll-off and low-light toe: exact formulas, continuity at the knee and
//! monotonicity over the whole parameter range.

use proptest::prelude::*;
use ym_color::gamut::LinearAwg4;
use ym_color::tone::{ToneError, ToneParams};

fn p(h: f64, l: f64, c: f64, sigma: f64) -> ToneParams {
    ToneParams::new(h, l, c, sigma).unwrap()
}

#[test]
fn shoulder_matches_the_documented_formula() {
    let c = 12.0;
    for r in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let t = p(r, 0.0, c, 0.0);
        let k = c * (-(0.5 + 2.0 * r)).exp2();
        let cp = 1.5 * c;
        assert!((t.knee() - k).abs() < 1e-14);
        assert!((t.shoulder_limit() - cp).abs() < 1e-14);
        for n in [0.0, 0.1 * k, k, 1.0001 * k, 2.0 * k, c, 3.0 * c, 100.0 * c] {
            let want = if n <= k {
                n
            } else {
                k + (cp - k) * ((n - k) / (cp - k)).tanh()
            };
            assert!((t.curve(n) - want).abs() < 1e-13, "r={r} n={n}");
        }
    }
}

#[test]
fn toe_matches_the_documented_formula_and_r_zero_disables_it() {
    let sigma = 0.004;
    let off = p(0.5, 0.0, 16.0, sigma);
    for n in [-1.0, -0.001, 0.0, 1e-5, 0.01, 0.18, 5.0] {
        assert_eq!(off.toe(n), n, "r = 0 must be exact identity");
    }
    for r in [0.1, 0.5, 1.0] {
        let t = p(0.5, r, 16.0, sigma);
        let tau = 4.0 * r * sigma;
        assert!((t.toe_tau() - tau).abs() < 1e-18);
        for n in [1e-4, 1e-3, tau, 3.0 * tau, 0.05, 0.18] {
            let want = n * n * n / (n * n + tau * tau);
            assert!((t.toe(n) - want).abs() < 1e-15, "r={r} n={n}");
        }
        // About n above 3 tau (within 10%), and strongly reduced below tau.
        assert!(t.toe(3.0 * tau) > 0.89 * 3.0 * tau);
        assert!(t.toe(0.5 * tau) < 0.25 * 0.5 * tau);
    }
}

#[test]
fn shoulder_is_continuous_with_unit_slope_at_the_knee() {
    for (r, c) in [(0.0, 8.0), (0.5, 16.0), (1.0, 32.0), (0.3, 1.0)] {
        let t = p(r, 0.0, c, 0.0);
        let k = t.knee();
        let eps = 1e-6 * k;
        // Value continuity.
        assert!((t.curve(k - eps) - (k - eps)).abs() < 1e-12);
        assert!((t.curve(k + eps) - (k + eps)).abs() < 1e-9 * k);
        // Slope continuity: 1 on both sides.
        let left = (t.curve(k) - t.curve(k - eps)) / eps;
        let right = (t.curve(k + eps) - t.curve(k)) / eps;
        assert!((left - 1.0).abs() < 1e-9, "left slope {left}");
        assert!((right - 1.0).abs() < 1e-5, "right slope {right}");
        // Second-order contact: the departure from the identity is O(eps^2).
        let dev = (t.curve(k + 1e-3 * k) - (k + 1e-3 * k)).abs();
        assert!(dev < 1e-5 * k, "deviation {dev}");
    }
}

#[test]
fn shoulder_never_reaches_the_asymptote_and_compresses() {
    let t = p(0.5, 0.0, 10.0, 0.0);
    for n in [10.0, 15.0, 20.0, 1e3, 1e9] {
        let f = t.curve(n);
        assert!(f <= 15.0 + 1e-12, "f({n}) = {f}");
        assert!(f <= n);
    }
    // 1e9 is still below c' only by rounding.
    assert!((t.curve(1e9) - 15.0).abs() < 1e-9);
    // The knee moves down with r: more roll-off at the same input.
    let a = p(0.0, 0.0, 10.0, 0.0).curve(10.0);
    let b = p(1.0, 0.0, 10.0, 0.0).curve(10.0);
    assert!(b < a);
}

#[test]
fn monotone_in_n_over_the_full_parameter_range() {
    // docs/15 M1-09: monotonicity of C7 over the full parameter ranges.
    let mut checked = 0_u64;
    for hi in 0..=10 {
        for lo in 0..=10 {
            for &c in &[0.25, 1.0, 4.0, 16.0, 64.0] {
                for &sigma in &[0.0, 1e-5, 1e-3, 1e-2, 0.1] {
                    let t = p(f64::from(hi) / 10.0, f64::from(lo) / 10.0, c, sigma);
                    let mut prev = t.curve(-2.0);
                    // Sweep n from -2 to 10 c on a fine, partly log-spaced grid.
                    let mut n = -2.0;
                    while n < 10.0 * c {
                        n += if n < 0.0 {
                            0.0137
                        } else {
                            (n.abs() * 0.01).max(1e-4)
                        };
                        let f = t.curve(n);
                        assert!(
                            f >= prev - 1e-14 * prev.abs().max(1.0),
                            "hi={hi} lo={lo} c={c} sigma={sigma} n={n}: {f} < {prev}"
                        );
                        prev = f;
                        checked += 1;
                    }
                }
            }
        }
    }
    println!("C7 monotonicity: {checked} sample steps checked");
}

#[test]
fn derivative_is_non_negative() {
    // Stronger than monotone samples: central differences of f(n) are >= 0.
    for hi in [0.0, 0.5, 1.0] {
        for lo in [0.0, 0.5, 1.0] {
            let t = p(hi, lo, 16.0, 0.004);
            let mut n = 1e-6;
            while n < 200.0 {
                let h = n * 1e-6;
                let d = (t.curve(n + h) - t.curve(n - h)) / (2.0 * h);
                assert!(d >= -1e-9, "n={n} d={d}");
                n *= 1.01;
            }
        }
    }
}

#[test]
fn scaling_preserves_channel_ratios_and_order() {
    let t = p(0.7, 0.6, 16.0, 0.003);
    for rgb in [
        [0.18, 0.18, 0.18],
        [12.0, 3.0, 0.5],
        [30.0, 28.0, 2.0],
        [0.002, 0.001, 0.0005],
        [0.0, 0.0, 0.0],
    ] {
        let out = t.apply(LinearAwg4(rgb)).0;
        let n = rgb[0].max(rgb[1]).max(rgb[2]);
        if n > 0.0 {
            let s = t.scale(n);
            for k in 0..3 {
                assert!((out[k] - rgb[k] * s).abs() < 1e-15 * (1.0 + rgb[k]));
            }
            // f(n) is the output's maximum channel.
            let out_n = out[0].max(out[1]).max(out[2]);
            assert!((out_n - t.curve(n)).abs() < 1e-12 * (1.0 + n));
        }
        assert!(out[0] >= out[1] || rgb[0] < rgb[1]);
        assert!((out[0] - out[1]) * (rgb[0] - rgb[1]) >= 0.0);
    }
}

#[test]
fn toe_off_and_values_below_the_knee_are_untouched() {
    let t = p(0.5, 0.0, 16.0, 0.01);
    let rgb = [0.5, 0.18, 0.05];
    assert_eq!(t.apply(LinearAwg4(rgb)).0, rgb);
}

#[test]
fn negative_and_non_finite_pixels() {
    let t = p(0.5, 0.5, 16.0, 0.01);
    // Negative-only pixels shrink towards zero but keep their sign and ratios.
    let out = t.apply(LinearAwg4([-0.1, -0.02, -0.5])).0;
    for (o, i) in out.iter().zip([-0.1, -0.02, -0.5]) {
        assert!(*o <= 0.0 && o.abs() <= f64::abs(i));
    }
    // A negative channel next to a bright one is scaled by the bright channel's factor, not clipped.
    let out = t.apply(LinearAwg4([1.0, 0.5, -0.05])).0;
    assert!(out[2] < 0.0);
    // Non-finite input passes through.
    let nan = t.apply(LinearAwg4([f64::NAN, 0.1, 0.1])).0;
    assert!(nan[0].is_nan());
    let inf = t.apply(LinearAwg4([f64::INFINITY, 0.1, 0.1])).0;
    assert!(inf[0].is_infinite());
    // n = 0 with a toe maps to exact zero.
    assert_eq!(t.apply(LinearAwg4([0.0, 0.0, 0.0])).0, [0.0, 0.0, 0.0]);
}

#[test]
fn constructor_validates() {
    assert!(matches!(
        ToneParams::new(0.5, 0.5, -1.0, 0.0),
        Err(ToneError::InvalidClipLevel(_))
    ));
    assert!(matches!(
        ToneParams::new(0.5, 0.5, 1.0, f64::INFINITY),
        Err(ToneError::InvalidNoiseFloor(_))
    ));
}

proptest! {
    /// Raising one channel never lowers its output (the map is monotone along each channel axis).
    #[test]
    fn each_output_channel_is_monotone_in_its_input(
        r in 0.0f64..40.0, g in 0.0f64..40.0, b in 0.0f64..40.0,
        d in 0.0f64..5.0, ch in 0usize..3,
        hi in 0.0f64..=1.0, lo in 0.0f64..=1.0, sigma in 0.0f64..0.05,
    ) {
        let t = ToneParams::new(hi, lo, 16.0, sigma).unwrap();
        let rgb = [r, g, b];
        let mut rgb2 = rgb;
        rgb2[ch] += d;
        let o1 = t.apply_rgb(rgb);
        let o2 = t.apply_rgb(rgb2);
        prop_assert!(o2[ch] >= o1[ch] - 1e-12, "{o1:?} -> {o2:?}");
    }

    #[test]
    fn curve_never_exceeds_input_or_limit(n in 0.0f64..1000.0, hi in 0.0f64..=1.0, lo in 0.0f64..=1.0) {
        let t = ToneParams::new(hi, lo, 16.0, 0.003).unwrap();
        let f = t.curve(n);
        prop_assert!(f <= n + 1e-12);
        prop_assert!(f <= t.shoulder_limit() + 1e-12);
        prop_assert!(f >= 0.0);
    }
}

#[test]
fn more_roll_off_never_raises_the_output() {
    // Monotone in the parameters too: a larger highlight amount lowers (or keeps) the output at every
    // n, and a larger low-light amount lowers (or keeps) the output at every n.
    for &c in &[1.0, 16.0] {
        let mut n = 0.0005;
        while n < 40.0 * c {
            let mut prev = f64::INFINITY;
            for step in 0..=20 {
                let r = f64::from(step) / 20.0;
                let f = p(r, 0.0, c, 0.003).curve(n);
                assert!(f <= prev + 1e-14, "highlights r={r} n={n}");
                prev = f;
            }
            let mut prev = f64::INFINITY;
            for step in 0..=20 {
                let r = f64::from(step) / 20.0;
                let f = p(0.5, r, c, 0.003).curve(n);
                assert!(f <= prev + 1e-14, "low light r={r} n={n}");
                prev = f;
            }
            n *= 1.05;
        }
    }
}

#[test]
fn tiny_values_never_turn_into_nan() {
    // n^2 + tau^2 underflows to zero in f64 below about 1e-154; the factor must stay finite and the
    // limits must hold: m << tau crushes to zero, m >> tau is untouched, m == 0 is black.
    let t = p(0.5, 0.25, 16.0, 1e-200);
    let tau = t.toe_tau();
    assert!(tau > 0.0 && tau * tau == 0.0, "tau^2 underflows: {tau:e}");
    for n in [0.0, 1e-300, 1e-200, 1e-170, 1e-160, 1e-100, 1e-3, 1.0] {
        let f = t.curve(n);
        assert!(f.is_finite() && (0.0..=n).contains(&f), "f({n:e}) = {f:e}");
        let s = t.scale(n);
        assert!(
            s.is_finite() && (0.0..=1.0).contains(&s),
            "scale({n:e}) = {s:e}"
        );
        let out = t.apply(LinearAwg4([n, 0.5 * n, 0.25 * n])).0;
        assert!(out.iter().all(|v| v.is_finite()), "{n:e}: {out:?}");
    }
    assert_eq!(t.curve(1.0), 1.0);
    assert_eq!(t.curve(0.0), 0.0);
    // Half way between: n = tau gives n / 2.
    assert!((t.curve(tau) / tau - 0.5).abs() < 1e-12);
}
