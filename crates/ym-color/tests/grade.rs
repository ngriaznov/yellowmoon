//! M1-09 (C9): the primary grade. Formulas against an independent transcription of the table in
//! docs/02, the properties of the contrast curve, of its colour step and of the saturation (design
//! and owner's answers of 2026-10-19), and monotonicity of every control (what "monotonic" means per
//! control is documented in `ym_color::grade`).

use proptest::prelude::*;
use ym_color::consts::{LOGC4_GRAY, LOGC4_STOP, Vec3};
use ym_color::gamut::{LinearAwg4, awg4_luminance};
use ym_color::grade::{
    BALANCE_CENTER_STOPS, BRIGHTNESS_SIGMA, COLOR_BALANCE_STRENGTH, CONTRAST_BLACK, CONTRAST_COLOR,
    CONTRAST_EXPONENT, CONTRAST_NEUTRAL_CHROMA, CONTRAST_TOP, ColorBalance, E1, E2, Grade,
    GradeParams, SATURATION_BOOST, SATURATION_CODE_LIMIT, SATURATION_SMALL_R, ZONE_CENTER_STOPS,
    ZONE_CONTROL_SIGMA, ZONE_SIGMA, bump, contrast_curve, contrast_pixel, contrast_slope,
    grade_luma, logc4_at_stops, saturate, saturation_chroma_scale,
};
use ym_color::logc::{LogC4, logc4_decode, logc4_encode};

/// The AWG4 luminance row, typed out (`[0.2545, 0.7815, -0.0360]`).
const Y_ROW: Vec3 = [0.254524176404027, 0.781477732712002, -0.036001909116029];

/// The documented constants, typed out: Contrast's colour step `beta` and the saturation boost
/// `alpha` (`docs/02` C9).
const BETA: f64 = 0.5;
const ALPHA: f64 = 1.5;

/// Straight transcription of the C9 table, independent of `Grade` (no precomputation, no shared
/// helpers): the contrast curve written out per segment, the saturation with `1 - (1 - r)^F` as the
/// spec writes it.
fn reference(p: &GradeParams, v: Vec3, l: f64) -> Vec3 {
    let g = 0.2783958365482653;
    let s = 0.9071358748778103 / 14.0;
    let black = 0.09286412512218964;
    let mut v = v;
    let e1 = [
        1.0 / 1.5_f64.sqrt(),
        -0.5 / 1.5_f64.sqrt(),
        -0.5 / 1.5_f64.sqrt(),
    ];
    let e2 = [0.0, 1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()];
    if p.contrast != 0.0 {
        let k = 2.0_f64.powf(0.6 * p.contrast / 100.0);
        let h = |t: f64| k * t / (1.0 + (k - 1.0) * t);
        let t = |x: f64| {
            if x > 1.0 {
                1.0 + (x - 1.0) / k
            } else if x >= g {
                g + (1.0 - g) * h((x - g) / (1.0 - g))
            } else if x >= black {
                g - (g - black) * h((g - x) / (g - black))
            } else {
                black + (x - black) / k
            }
        };
        // u = v + (T(L) - L); p = T on each channel; u's chroma scaled toward |c_p| by BETA.
        let u = v.map(|x| x + t(l) - l);
        let p = v.map(t);
        let chroma = |x: Vec3| {
            [
                e1[0] * x[0] + e1[1] * x[1] + e1[2] * x[2],
                e2[0] * x[0] + e2[1] * x[1] + e2[2] * x[2],
            ]
        };
        let (cu, cp) = (chroma(u), chroma(p));
        let len_u = (cu[0] * cu[0] + cu[1] * cu[1]).sqrt();
        let len_p = (cp[0] * cp[0] + cp[1] * cp[1]).sqrt();
        let f = if len_u < 1e-9 {
            1.0
        } else {
            1.0 + BETA * (len_p / len_u - 1.0)
        };
        v = [0, 1, 2].map(|i| u[i] + (f - 1.0) * (cu[0] * e1[i] + cu[1] * e2[i]));
    }
    let mut add = (p.brightness / 100.0) * s * (-(l - g).powi(2) / (2.0 * (2.5 * s).powi(2))).exp();
    for (x, stops) in [
        (p.shadows, -3.0),
        (p.highlights, 3.0),
        (p.black, -6.0),
        (p.white, 6.0),
    ] {
        let center = logc4_encode(0.18 * f64::powf(2.0, stops));
        add += (x / 100.0) * 1.5 * s * (-(l - center).powi(2) / (2.0 * (1.5 * s).powi(2))).exp();
    }
    for x in &mut v {
        *x += add;
    }
    for (bal, stops) in [
        (p.balance_shadows, -3.0),
        (p.balance_midtones, 0.0),
        (p.balance_highlights, 3.0),
    ] {
        let center = logc4_encode(0.18 * f64::powf(2.0, stops));
        let zone = (-(l - center).powi(2) / (2.0 * (1.5 * s).powi(2))).exp();
        for k in 0..3 {
            v[k] += 0.03 * (bal.a * e1[k] + bal.b * e2[k]) * zone;
        }
    }
    if p.saturation != 0.0 {
        let rgb = v.map(logc4_decode);
        let y = Y_ROW[0] * rgb[0] + Y_ROW[1] * rgb[1] + Y_ROW[2] * rgb[2];
        let lambda = if p.saturation < 0.0 {
            Some(1.0 + p.saturation / 100.0)
        } else {
            let f = 1.0 + ALPHA * p.saturation / 100.0;
            let r = (y - rgb[0].min(rgb[1]).min(rgb[2])) / y;
            if y <= 0.0 || r >= 1.0 {
                None
            } else if r < 1e-9 {
                Some(f)
            } else {
                Some((1.0 - (1.0 - r).powf(f)) / r)
            }
        };
        if let Some(lambda) = lambda {
            v = rgb.map(|c| logc4_encode(y + lambda * (c - y)));
        }
    }
    v
}

fn all_controls(x: f64) -> GradeParams {
    GradeParams {
        contrast: x,
        brightness: x,
        shadows: x,
        highlights: x,
        black: x,
        white: x,
        balance_shadows: ColorBalance {
            a: x / 150.0,
            b: -x / 300.0,
        },
        balance_midtones: ColorBalance {
            a: -x / 300.0,
            b: x / 150.0,
        },
        balance_highlights: ColorBalance {
            a: x / 150.0,
            b: x / 150.0,
        },
        saturation: x,
    }
}

#[test]
fn constants_are_the_documented_ones() {
    assert!((LOGC4_STOP - 0.06480).abs() < 5e-6);
    assert!((LOGC4_GRAY - 0.2784).abs() < 1e-4);
    assert_eq!(BRIGHTNESS_SIGMA, 2.5 * LOGC4_STOP);
    assert_eq!(ZONE_CONTROL_SIGMA, 1.5 * LOGC4_STOP);
    assert_eq!(ZONE_SIGMA, 1.5 * LOGC4_STOP);
    assert_eq!(COLOR_BALANCE_STRENGTH, 0.03);
    assert!((bump(1.0, 1.0, 0.3) - 1.0).abs() < 1e-16);
    assert!((bump(1.3, 1.0, 0.3) - (-0.5_f64).exp()).abs() < 1e-15);
    assert_eq!(CONTRAST_COLOR, BETA);
    assert_eq!(CONTRAST_NEUTRAL_CHROMA, 1e-9);
    assert_eq!(SATURATION_BOOST, ALPHA);
}

// ---- Contrast: the tone curve on the luma (design of 2026-10-19) --------------------------------

/// The slopes the control's range spans: `2^-0.6` to `2^0.6`.
const K_MIN: f64 = 0.659_753_955_386_447_1;
const K_MAX: f64 = 1.515_716_566_510_398;

/// Central-difference slope of `T_k` at `x` from the left and from the right (one-sided, step `h`).
fn one_sided_slopes(x: f64, k: f64, h: f64) -> (f64, f64) {
    let left = (contrast_curve(x, k) - contrast_curve(x - h, k)) / h;
    let right = (contrast_curve(x + h, k) - contrast_curve(x, k)) / h;
    (left, right)
}

#[test]
fn contrast_constants_and_slope() {
    assert_eq!(CONTRAST_BLACK, logc4_encode(0.0));
    assert!((CONTRAST_BLACK - 0.092_864_125_1).abs() < 1e-10);
    assert_eq!(CONTRAST_TOP, 1.0);
    assert_eq!(CONTRAST_EXPONENT, 0.6);
    assert_eq!(contrast_slope(0.0), 1.0);
    assert!((contrast_slope(100.0) - K_MAX).abs() < 1e-15);
    assert!((contrast_slope(-100.0) - K_MIN).abs() < 1e-15);
    assert!((K_MIN * K_MAX - 1.0).abs() < 1e-15);
    // Clamped like every control.
    assert_eq!(contrast_slope(400.0), contrast_slope(100.0));
    assert_eq!(contrast_slope(f64::NAN), 1.0);
}

#[test]
fn the_contrast_curve_fixes_black_grey_and_the_top_code_value() {
    for c in (-100..=100).step_by(5) {
        let k = contrast_slope(f64::from(c));
        assert_eq!(contrast_curve(LOGC4_GRAY, k), LOGC4_GRAY, "c={c}");
        assert_eq!(contrast_curve(CONTRAST_BLACK, k), CONTRAST_BLACK, "c={c}");
        assert_eq!(contrast_curve(CONTRAST_TOP, k), CONTRAST_TOP, "c={c}");
        // And it stays inside [black, top] between them.
        for i in 0..=100 {
            let x = CONTRAST_BLACK + (CONTRAST_TOP - CONTRAST_BLACK) * f64::from(i) / 100.0;
            let t = contrast_curve(x, k);
            assert!(
                (CONTRAST_BLACK..=CONTRAST_TOP).contains(&t),
                "c={c} x={x}: {t}"
            );
        }
    }
    // k = 1 is the identity everywhere.
    for x in [-1.0, 0.0, 0.05, 0.2, 0.2784, 0.5, 1.0, 3.0] {
        assert!((contrast_curve(x, 1.0) - x).abs() < 1e-15, "{x}");
    }
}

#[test]
fn the_contrast_curve_has_slope_k_at_grey_and_one_over_k_at_both_ends() {
    let h = 1e-7;
    for k in [K_MIN, 0.8, 0.95, 1.05, 1.3, K_MAX] {
        for (x, want) in [
            (LOGC4_GRAY, k),
            (CONTRAST_BLACK, 1.0 / k),
            (CONTRAST_TOP, 1.0 / k),
        ] {
            let (left, right) = one_sided_slopes(x, k, h);
            // One-sided differences of a C2-almost-everywhere curve: error ~ |T''| h.
            assert!(
                (left - want).abs() < 1e-5,
                "k={k} x={x}: left {left} vs {want}"
            );
            assert!(
                (right - want).abs() < 1e-5,
                "k={k} x={x}: right {right} vs {want}"
            );
        }
        // Beyond the ends the curve is the straight line of slope 1 / k.
        for x in [-0.5, 0.0, 0.05, 1.2, 2.0, 7.0] {
            let want = if x > CONTRAST_TOP {
                CONTRAST_TOP + (x - CONTRAST_TOP) / k
            } else {
                CONTRAST_BLACK + (x - CONTRAST_BLACK) / k
            };
            assert!((contrast_curve(x, k) - want).abs() < 1e-15, "k={k} x={x}");
        }
    }
}

proptest! {
    /// Increasing and C1 for every slope the control can give, everywhere (below black and above the
    /// top code value included): the curve rises between any two points, and the slopes from the left
    /// and the right agree.
    #[test]
    fn the_contrast_curve_is_increasing_and_c1(
        k in K_MIN..=K_MAX, x in -0.6f64..2.5, dx in 1e-6f64..0.5,
    ) {
        let (t0, t1) = (contrast_curve(x, k), contrast_curve(x + dx, k));
        prop_assert!(t1 > t0, "k={k}: T({x}) = {t0}, T({}) = {t1}", x + dx);
        // The slope is between min(k, 1/k) and max(k, 1/k), so the rise is too.
        let (lo, hi) = (k.min(1.0 / k), k.max(1.0 / k));
        prop_assert!(t1 - t0 >= lo * dx * (1.0 - 1e-8) && t1 - t0 <= hi * dx * (1.0 + 1e-8));
        let h = 1e-6;
        let (left, right) = one_sided_slopes(x, k, h);
        prop_assert!((left - right).abs() < 1e-4, "k={k} x={x}: slopes {left} and {right}");
        // And at the three points where the pieces meet (|T''| <= 8.5, so one-sided differences over
        // 1e-7 agree to about 1e-6 where the slope is continuous).
        for j in [CONTRAST_BLACK, LOGC4_GRAY, CONTRAST_TOP] {
            let (left, right) = one_sided_slopes(j, k, 1e-7);
            prop_assert!((left - right).abs() < 1e-5, "k={k} at {j}: slopes {left} and {right}");
        }
    }
}

#[test]
fn contrast_offsets_all_three_channels_by_the_change_of_the_luma() {
    // The hue-keeping part (`contrast_pixel` with the colour step at 0): one offset on all channels.
    for c in [-100.0, -60.0, -30.0, 30.0, 60.0, 100.0] {
        let k = contrast_slope(c);
        for lin in [
            [0.18, 0.18, 0.18],
            [0.4, 0.2, 0.05],
            [0.02, 0.03, 0.2],
            [3.0, 1.5, 0.9],
            [0.004, 0.002, 0.001],
            [900.0, 400.0, 200.0],
        ] {
            let lin = LinearAwg4(lin);
            let (v, l) = (lin.to_logc4().0, grade_luma(lin));
            let out = contrast_pixel(v, l, k, 0.0);
            let want = contrast_curve(l, k) - l;
            for ch in 0..3 {
                assert!(
                    (out[ch] - v[ch] - want).abs() < 1e-15,
                    "c={c} {lin:?} ch {ch}"
                );
            }
            // On the log segment a common offset is a common gain on `x + 64 / a` (a gain on x where x
            // is well above 0.029): the channel ratios of that quantity are kept.
            let shifted = |code: f64| logc4_decode(code) + 64.0 / 2231.8263090676883;
            if v.iter().all(|&x| x >= 0.0) && out.iter().all(|&x| x >= 0.0) {
                let gain = shifted(out[0]) / shifted(v[0]);
                for ch in 1..3 {
                    let gk = shifted(out[ch]) / shifted(v[ch]);
                    assert!(
                        (gk / gain - 1.0).abs() < 1e-12,
                        "c={c} {lin:?}: {gk} vs {gain}"
                    );
                }
            }
        }
    }
    // The grade's Contrast is that with the colour step `CONTRAST_COLOR`, and it moves the mean of
    // the three channels by the change of the luma.
    for c in [-100.0, -60.0, -30.0, 30.0, 60.0, 100.0] {
        let g = Grade::new(&GradeParams {
            contrast: c,
            ..Default::default()
        });
        let k = contrast_slope(c);
        for lin in [[0.4, 0.2, 0.05], [0.02, 0.03, 0.2], [3.0, 1.5, 0.9]] {
            let lin = LinearAwg4(lin);
            let (v, l) = (lin.to_logc4().0, grade_luma(lin));
            let out = g.apply_linear(lin).0;
            assert_eq!(out, contrast_pixel(v, l, k, CONTRAST_COLOR), "c={c}");
            let mean = |x: Vec3| (x[0] + x[1] + x[2]) / 3.0;
            assert!((mean(out) - mean(v) - (contrast_curve(l, k) - l)).abs() < 1e-15);
        }
    }
    // A neutral grey card stays at grey, a neutral black at black, a neutral at the top code at the top.
    let g = Grade::new(&GradeParams {
        contrast: 73.0,
        ..Default::default()
    });
    for x in [LOGC4_GRAY, CONTRAST_BLACK, CONTRAST_TOP] {
        assert_eq!(g.apply(LogC4([x; 3]), x).0, [x; 3]);
    }
    // +100 at a luma half way from grey to the top: g + (1 - g) h(1/2) with h(1/2) = k / (1 + k).
    let l = 0.5 * (LOGC4_GRAY + 1.0);
    let k = K_MAX;
    let want = LOGC4_GRAY + (1.0 - LOGC4_GRAY) * k / (1.0 + k);
    let g = Grade::new(&GradeParams {
        contrast: 100.0,
        ..Default::default()
    });
    assert!((g.apply(LogC4([l; 3]), l).0[0] - want).abs() < 1e-15);
}

// ---- Contrast's colour step (the owner's answer of 2026-10-19) ---------------------------------

fn chroma(x: Vec3) -> [f64; 2] {
    [
        E1[0] * x[0] + E1[1] * x[1] + E1[2] * x[2],
        E2[0] * x[0] + E2[1] * x[1] + E2[2] * x[2],
    ]
}

fn length(c: [f64; 2]) -> f64 {
    c[0].hypot(c[1])
}

fn mean(x: Vec3) -> f64 {
    (x[0] + x[1] + x[2]) / 3.0
}

fn largest(x: Vec3) -> f64 {
    x.iter().fold(1.0f64, |m, c| m.max(c.abs()))
}

/// The hue-keeping result `u = v + (T(L) - L)`, written out.
fn offset_only(v: Vec3, l: f64, k: f64) -> Vec3 {
    let d = contrast_curve(l, k) - l;
    v.map(|x| x + d)
}

/// LogC4 codes from below black to above the top code value, so every piece of the curve is hit.
fn any_code() -> impl Strategy<Value = Vec3> {
    prop::array::uniform3(-0.3f64..1.6)
}

proptest! {
    /// The colour step only scales `u`'s chroma: its direction (the hue in the C9 plane) and the mean
    /// of the three channels (the position on the neutral axis) are `u`'s, and the length is
    /// `f |c_u|` with `f = 1 + beta (|c_p| / |c_u| - 1)`, which lies between `1 - beta (1 - min(k, 1/k))`
    /// and `1 + beta (max(k, 1/k) - 1)`.
    #[test]
    fn the_colour_step_keeps_the_hue_and_the_mean_and_scales_the_chroma(
        v in any_code(), l in -0.3f64..1.6, c in -100.0f64..=100.0, beta in 0.0f64..=1.0,
    ) {
        let k = contrast_slope(c);
        let u = offset_only(v, l, k);
        let out = contrast_pixel(v, l, k, beta);
        let (cu, co) = (chroma(u), chroma(out));
        let eps = 1e-14 * largest(out);
        // Parallel and the same way round: the hue is u's.
        prop_assert!(
            (cu[0] * co[1] - cu[1] * co[0]).abs() <= eps * (length(cu) + length(co)),
            "{v:?} l={l} c={c} beta={beta}: {cu:?} vs {co:?}"
        );
        prop_assert!(cu[0] * co[0] + cu[1] * co[1] >= -eps * eps);
        prop_assert!((mean(out) - mean(u)).abs() <= eps, "{v:?}: mean {} vs {}", mean(out), mean(u));
        if length(chroma(v)) >= 1e-6 {
            let f = length(co) / length(cu);
            let cp = chroma(v.map(|x| contrast_curve(x, k)));
            let want = 1.0 + beta * (length(cp) / length(chroma(v)) - 1.0);
            prop_assert!((f - want).abs() < 1e-8, "{v:?} c={c} beta={beta}: f {f} vs {want}");
            let (lo, hi) = (k.min(1.0 / k), k.max(1.0 / k));
            prop_assert!(f >= 1.0 - beta * (1.0 - lo) - 1e-8 && f <= 1.0 + beta * (hi - 1.0) + 1e-8);
        }
    }

    /// `beta = 0` is the hue-keeping offset alone, `beta = 1` gives `u` the chroma length of the
    /// curve on each channel, and the grade's Contrast is the step at `CONTRAST_COLOR`.
    #[test]
    fn beta_zero_is_the_offset_and_beta_one_the_per_channel_length(
        v in any_code(), l in -0.3f64..1.6, c in -100.0f64..=100.0,
    ) {
        let k = contrast_slope(c);
        prop_assert_eq!(contrast_pixel(v, l, k, 0.0), offset_only(v, l, k));
        let full = contrast_pixel(v, l, k, 1.0);
        let cp = chroma(v.map(|x| contrast_curve(x, k)));
        if length(chroma(v)) >= 1e-6 {
            prop_assert!(
                (length(chroma(full)) - length(cp)).abs() <= 1e-12 * largest(full),
                "{v:?} c={c}: {} vs {}", length(chroma(full)), length(cp)
            );
        }
        if c != 0.0 {
            let g = Grade::new(&GradeParams { contrast: c, ..Default::default() });
            prop_assert_eq!(g.apply(LogC4(v), l).0, contrast_pixel(v, l, k, CONTRAST_COLOR));
        }
    }

    /// Neutrals are untouched by the colour step: a neutral (and a chroma below
    /// `CONTRAST_NEUTRAL_CHROMA`) gets the offset alone, so a neutral stays neutral.
    #[test]
    fn neutrals_get_the_offset_alone(
        x in -0.3f64..1.6, l in -0.3f64..1.6, c in -100.0f64..=100.0, tiny in -6.9e-10f64..6.9e-10,
    ) {
        let k = contrast_slope(c);
        let out = contrast_pixel([x; 3], l, k, CONTRAST_COLOR);
        prop_assert_eq!(out, offset_only([x; 3], l, k));
        prop_assert!(out[0] == out[1] && out[1] == out[2]);
        // |c| = |tiny| sqrt(2/3) < 1e-9 for a chroma along one channel.
        let v = [x + tiny, x, x];
        prop_assert_eq!(contrast_pixel(v, l, k, CONTRAST_COLOR), offset_only(v, l, k));
    }
}

#[test]
fn contrast_adds_colour_in_the_midtones_as_a_curve_on_each_channel_would() {
    // Channels between 0.21 and 0.55, where the curve's slope is above 1 for every c > 0 and below 1
    // for every c < 0: more contrast adds colour, less takes it away.
    let mid = [
        [0.40, 0.30, 0.24],
        [0.22, 0.33, 0.50],
        [0.30, 0.45, 0.28],
        [0.54, 0.40, 0.21],
    ];
    // Deep shadows (channels 0.10 to 0.14, where the slope is below 1 for c > 0): more contrast
    // takes colour away there, as the curve on each channel would.
    let deep = [[0.14, 0.11, 0.10], [0.10, 0.12, 0.14]];
    for c in [10.0, 30.0, 60.0, 100.0, -10.0, -30.0, -60.0, -100.0] {
        let g = Grade::new(&GradeParams {
            contrast: c,
            ..Default::default()
        });
        for (set, more_in_midtones) in [(&mid[..], true), (&deep[..], false)] {
            for &v in set {
                let l = mean(v);
                let f = length(chroma(g.apply(LogC4(v), l).0)) / length(chroma(v));
                let gains = (f > 1.0) == (c > 0.0);
                assert_eq!(gains, more_in_midtones, "c={c} {v:?}: f {f}");
            }
        }
    }
    // At the ends of the range the factor is within 0.83 to 1.26.
    let (lo, hi) = (contrast_slope(-100.0), contrast_slope(100.0));
    assert!((1.0 - CONTRAST_COLOR * (1.0 - lo) - 0.8299).abs() < 1e-4);
    assert!((1.0 + CONTRAST_COLOR * (hi - 1.0) - 1.2579).abs() < 1e-4);
}

#[test]
fn brightness_and_zone_bumps_peak_where_documented() {
    // Brightness: +S at grey (L = g), nothing far away.
    let g = Grade::new(&GradeParams {
        brightness: 100.0,
        ..Default::default()
    });
    let d = g.apply(LogC4([0.3; 3]), LOGC4_GRAY).0[0] - 0.3;
    assert!((d - LOGC4_STOP).abs() < 1e-15, "{d}");
    let far = g.apply(LogC4([0.9; 3]), 0.9).0[0] - 0.9;
    assert!(far.abs() < 1e-3 * LOGC4_STOP);
    // The four zone controls peak at 1.5 S at their centres.
    for (field, stops) in [
        ("shadows", -3.0),
        ("highlights", 3.0),
        ("black", -6.0),
        ("white", 6.0),
    ] {
        let mut p = GradeParams::default();
        match field {
            "shadows" => p.shadows = 100.0,
            "highlights" => p.highlights = 100.0,
            "black" => p.black = 100.0,
            _ => p.white = 100.0,
        }
        let center = logc4_at_stops(stops);
        let g = Grade::new(&p);
        let d = g.apply(LogC4([0.4; 3]), center).0[0] - 0.4;
        assert!((d - 1.5 * LOGC4_STOP).abs() < 1e-15, "{field}: {d}");
        // Half a width away the bump is exp(-1/8) of the peak.
        let d2 = g
            .apply(LogC4([0.4; 3]), center + 0.5 * ZONE_CONTROL_SIGMA)
            .0[0]
            - 0.4;
        assert!((d2 / d - (-0.125_f64).exp()).abs() < 1e-14);
    }
}

#[test]
fn matches_an_independent_transcription_of_the_table() {
    let lv = [
        ([0.15, 0.30, 0.55], 0.28),
        ([0.45, 0.45, 0.45], 0.45),
        ([0.10, 0.12, 0.09], 0.11),
        ([0.80, 0.55, 0.30], 0.60),
        ([-0.02, 0.2, 0.9], 0.3),
    ];
    for x in [-100.0, -37.0, 0.0, 12.5, 100.0] {
        let p = all_controls(x);
        let g = Grade::new(&p);
        for (v, l) in lv {
            let got = g.apply(LogC4(v), l).0;
            let want = reference(&p, v, l);
            // The saturation's `1 - (1 - r)^F` in the transcription loses a few digits to
            // cancellation where `Grade` uses expm1 and ln_1p; 1e-11 code values is far below anything
            // a picture shows.
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-11,
                    "x={x} v={v:?}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

proptest! {
    #[test]
    fn matches_reference_on_random_params(
        c in -100.0f64..=100.0, b in -100.0f64..=100.0, s in -100.0f64..=100.0, h in -100.0f64..=100.0,
        k in -100.0f64..=100.0, w in -100.0f64..=100.0, sat in -100.0f64..=100.0,
        a1 in -0.7f64..=0.7, b1 in -0.7f64..=0.7, a2 in -0.7f64..=0.7, b2 in -0.7f64..=0.7,
        a3 in -0.7f64..=0.7, b3 in -0.7f64..=0.7,
        r in 0.0f64..1.0, gg in 0.0f64..1.0, bb in 0.0f64..1.0, l in 0.0f64..1.0,
    ) {
        let p = GradeParams {
            contrast: c, brightness: b, shadows: s, highlights: h, black: k, white: w,
            balance_shadows: ColorBalance { a: a1, b: b1 },
            balance_midtones: ColorBalance { a: a2, b: b2 },
            balance_highlights: ColorBalance { a: a3, b: b3 },
            saturation: sat,
        };
        let got = Grade::new(&p).apply(LogC4([r, gg, bb]), l).0;
        let want = reference(&p, [r, gg, bb], l);
        for i in 0..3 {
            prop_assert!((got[i] - want[i]).abs() < 1e-10, "{got:?} vs {want:?}");
        }
    }
}

#[test]
fn color_balance_displacement() {
    // At the shadows centre the shadows wheel displaces by 0.03 (a e1 + b e2); other zones are far.
    let center = logc4_at_stops(-3.0);
    let p = GradeParams {
        balance_shadows: ColorBalance { a: 0.6, b: -0.3 },
        ..Default::default()
    };
    let g = Grade::new(&p);
    let v = [0.2, 0.2, 0.2];
    let out = g.apply(LogC4(v), center).0;
    for k in 0..3 {
        let d = out[k] - v[k];
        let want = COLOR_BALANCE_STRENGTH * (0.6 * E1[k] - 0.3 * E2[k]);
        assert!((d - want).abs() < 1e-15);
    }
    // The displacement keeps the mean (both axes sum to zero).
    assert!(((out[0] + out[1] + out[2]) - (v[0] + v[1] + v[2])).abs() < 1e-15);
    // The zone is a bump: at the highlights centre the shadows wheel is nearly silent.
    let hi = g.apply(LogC4(v), logc4_at_stops(3.0)).0;
    assert!((hi[0] - v[0]).abs() < 1e-2 * COLOR_BALANCE_STRENGTH);
    // A vector outside the unit disc is clamped to radius 1.
    let big = Grade::new(&GradeParams {
        balance_midtones: ColorBalance { a: 3.0, b: 4.0 },
        ..Default::default()
    });
    let unit = Grade::new(&GradeParams {
        balance_midtones: ColorBalance { a: 0.6, b: 0.8 },
        ..Default::default()
    });
    assert_eq!(
        big.apply(LogC4(v), LOGC4_GRAY),
        unit.apply(LogC4(v), LOGC4_GRAY)
    );
}

#[test]
fn saturation_scales_the_chroma_about_the_luminance() {
    // A pixel with every channel above 0: Y = 0.17496.
    let lin = [0.40, 0.12, 0.05];
    let y = awg4_luminance(lin);
    assert!((y - (0.2545241764 * 0.40 + 0.7814777327 * 0.12 - 0.0360019091 * 0.05)).abs() < 1e-9);
    let v = LinearAwg4(lin).to_logc4().0;
    let l = 0.33; // L plays no part in Saturation
    let sat = |s: f64| {
        let out = Grade::new(&GradeParams {
            saturation: s,
            ..Default::default()
        })
        .apply(LogC4(v), l)
        .0;
        out.map(logc4_decode)
    };
    let close = |a: Vec3, b: Vec3| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-13);
    // -100: grey at the same luminance.
    assert!(close(sat(-100.0), [y; 3]), "{:?}", sat(-100.0));
    // -50: half way to it.
    assert!(close(sat(-50.0), lin.map(|c| y + 0.5 * (c - y))));
    // +60: F = 1.6 and r from the darkest channel (blue).
    let f = 1.0 + SATURATION_BOOST * 0.6;
    let r = (y - 0.05) / y;
    let r2 = 1.0 - (1.0 - r).powf(f);
    let lambda = r2 / r;
    assert!(lambda > 1.0 && lambda < f, "{lambda}");
    let want = lin.map(|c| y + lambda * (c - y));
    assert!(close(sat(60.0), want), "{:?} vs {want:?}", sat(60.0));
    // The darkest channel lands at Y (1 - r').
    assert!((sat(60.0)[2] - y * (1.0 - r2)).abs() < 1e-13);
    // The luminance is the same at every setting.
    for s in [-100.0, -40.0, 25.0, 60.0, 100.0] {
        let out = sat(s);
        assert!((awg4_luminance(out) - y).abs() < 1e-13 * y, "s={s}");
    }
    // The pivot does not depend on L.
    let g = Grade::new(&GradeParams {
        saturation: 63.0,
        ..Default::default()
    });
    for l in [0.0, 0.2, 0.9, 5.0] {
        assert_eq!(g.apply(LogC4(v), l), g.apply(LogC4(v), 0.33));
    }
}

#[test]
fn saturation_acts_on_the_pixel_the_controls_before_it_produced() {
    // With Contrast and Brightness on, -100 is the grey of the luminance after them.
    let lin = LinearAwg4([0.55, 0.20, 0.35]);
    let (v, l) = (lin.to_logc4().0, grade_luma(lin));
    for p in [
        GradeParams {
            contrast: 60.0,
            ..Default::default()
        },
        GradeParams {
            brightness: 80.0,
            ..Default::default()
        },
    ] {
        let before = Grade::new(&p).apply(LogC4(v), l).0;
        let y = awg4_luminance(before.map(logc4_decode));
        let out = Grade::new(&GradeParams {
            saturation: -100.0,
            ..p
        })
        .apply(LogC4(v), l)
        .0;
        for c in out.map(logc4_decode) {
            assert!((c - y).abs() < 1e-13, "{out:?} vs {y}");
        }
    }
}

#[test]
fn l_comes_from_linear_values_before_c8() {
    let lin = LinearAwg4([0.4, 0.2, 0.05]);
    let l = grade_luma(lin);
    let y = 0.2545241764 * 0.4 + 0.7814777327 * 0.2 - 0.0360019091 * 0.05;
    assert!((l - logc4_encode(y)).abs() < 1e-9);
    // The same as the LogC4 round trip's luma.
    assert!((lin.to_logc4().luma() - l).abs() < 1e-12);
    let p = GradeParams {
        contrast: 40.0,
        saturation: 25.0,
        shadows: 30.0,
        ..Default::default()
    };
    let g = Grade::new(&p);
    assert_eq!(g.apply_linear(lin), g.apply(lin.to_logc4(), l));
}

#[test]
fn out_of_range_controls_are_clamped() {
    let wild = GradeParams {
        contrast: 900.0,
        brightness: -900.0,
        saturation: f64::NAN,
        ..Default::default()
    };
    let edge = GradeParams {
        contrast: 100.0,
        brightness: -100.0,
        saturation: 0.0,
        ..Default::default()
    };
    let v = LogC4([0.4, 0.6, 0.2]);
    assert_eq!(
        Grade::new(&wild).apply(v, 0.45),
        Grade::new(&edge).apply(v, 0.45)
    );
}

// ---- Monotonicity ------------------------------------------------------------------------------

/// Sample pixels: neutral ramp and coloured pixels, each with its luma.
fn sample_pixels() -> Vec<(Vec3, f64)> {
    let mut out = Vec::new();
    // From below black to above the top code value.
    for k in 0..=48 {
        let v = 0.05 + 0.95 * f64::from(k) / 40.0;
        out.push(([v, v, v], v));
    }
    for lin in [
        [0.18, 0.05, 0.02],
        [0.02, 0.3, 0.1],
        [0.05, 0.1, 0.8],
        [3.0, 1.0, 0.2],
        [0.01, 0.01, 0.2],
        [0.5, 0.5, 0.01],
    ] {
        let rgb = LinearAwg4(lin);
        out.push((rgb.to_logc4().0, grade_luma(rgb)));
    }
    out
}

/// Which way Contrast moves every channel of a pixel whose luma is `l` as the control rises: away
/// from grey between black and the top code value (+1 above grey, -1 below), toward the end beyond
/// them (the line of slope `1 / k`); at grey, black and the top code value nothing moves.
fn contrast_direction(l: f64) -> f64 {
    let side = if l > LOGC4_GRAY {
        1.0
    } else if l < LOGC4_GRAY {
        -1.0
    } else {
        0.0
    };
    if (CONTRAST_BLACK..=CONTRAST_TOP).contains(&l) {
        side
    } else {
        -side
    }
}

fn sweep(set: impl Fn(&mut GradeParams, f64)) -> Vec<Grade> {
    (-100..=100)
        .step_by(5)
        .map(|x| {
            let mut p = GradeParams::default();
            set(&mut p, f64::from(x));
            Grade::new(&p)
        })
        .collect()
}

#[test]
fn every_scalar_control_is_monotone_in_its_value() {
    // Brightness, Shadows, Highlights, Black, White: every channel is non-decreasing in the control.
    type Setter = fn(&mut GradeParams, f64);
    let setters: [(&str, Setter); 5] = [
        ("brightness", |p, x| p.brightness = x),
        ("shadows", |p, x| p.shadows = x),
        ("highlights", |p, x| p.highlights = x),
        ("black", |p, x| p.black = x),
        ("white", |p, x| p.white = x),
    ];
    for (name, set) in setters {
        let grades = sweep(set);
        for (v, l) in sample_pixels() {
            let mut prev = grades[0].apply(LogC4(v), l).0;
            for g in &grades[1..] {
                let cur = g.apply(LogC4(v), l).0;
                for k in 0..3 {
                    assert!(cur[k] >= prev[k] - 1e-15, "{name}: v={v:?} l={l}");
                }
                prev = cur;
            }
        }
    }
    // Contrast: for a luma between black and the top code value, the mean of the three channels (and
    // every channel of a neutral) moves away from grey as the value rises (up above grey, down below
    // it); beyond the ends the line of slope 1 / k pulls the other way (toward the end), and at grey
    // and at the ends nothing moves. The colour step leaves the mean where it is, so a single channel
    // of a coloured pixel is not promised to follow it.
    let grades = sweep(|p, x| p.contrast = x);
    let mut directions = [0usize; 3];
    let mean = |x: Vec3| (x[0] + x[1] + x[2]) / 3.0;
    for (v, l) in sample_pixels() {
        let dir = contrast_direction(l);
        directions[(dir + 1.0) as usize] += 1;
        let neutral = v[0] == v[1] && v[1] == v[2];
        let mut prev = grades[0].apply(LogC4(v), l).0;
        for g in &grades[1..] {
            let cur = g.apply(LogC4(v), l).0;
            assert!(
                (mean(cur) - mean(prev)) * dir >= -1e-15,
                "contrast: {v:?} l={l}: mean {} -> {}",
                mean(prev),
                mean(cur)
            );
            for k in 0..3 {
                assert!(
                    !neutral || (cur[k] - prev[k]) * dir >= -1e-15,
                    "contrast: {v:?} l={l} channel {k}: {} -> {}",
                    prev[k],
                    cur[k]
                );
            }
            prev = cur;
        }
    }
    assert!(directions[0] > 5 && directions[2] > 5, "{directions:?}");
    // Saturation: the distance of each channel from the luminance Y (in linear AWG4) grows with the
    // control; Y does not move.
    let grades = sweep(|p, x| p.saturation = x);
    for (v, l) in sample_pixels() {
        let y = awg4_luminance(v.map(logc4_decode));
        let mut prev = [0.0; 3];
        for (i, g) in grades.iter().enumerate() {
            let o = g.apply(LogC4(v), l).0.map(logc4_decode);
            assert!(
                (awg4_luminance(o) - y).abs() <= 1e-13 * y.abs().max(1e-3),
                "saturation moved Y: {v:?}"
            );
            for k in 0..3 {
                let dist = (o[k] - y).abs();
                if i > 0 {
                    assert!(
                        dist >= prev[k] - 1e-13 * y.abs().max(1e-3),
                        "saturation: {v:?} channel {k}"
                    );
                }
                prev[k] = dist;
            }
        }
    }
}

#[test]
fn color_balance_displacement_grows_with_radius() {
    for center_stops in [-3.0, 0.0, 3.0, -1.0] {
        let l = logc4_at_stops(center_stops);
        for (da, db) in [(1.0, 0.0), (0.0, 1.0), (-0.6, 0.8), (0.3, -0.2)] {
            let mut prev = 0.0;
            for step in 0..=20 {
                let r = f64::from(step) / 20.0;
                let bal = ColorBalance {
                    a: r * da,
                    b: r * db,
                };
                let p = GradeParams {
                    balance_shadows: bal,
                    balance_midtones: bal,
                    balance_highlights: bal,
                    ..Default::default()
                };
                let v = [0.3, 0.3, 0.3];
                let o = Grade::new(&p).apply(LogC4(v), l).0;
                let len =
                    ((o[0] - v[0]).powi(2) + (o[1] - v[1]).powi(2) + (o[2] - v[2]).powi(2)).sqrt();
                assert!(len >= prev - 1e-15);
                prev = len;
            }
        }
    }
}

#[test]
fn neutral_ramp_stays_a_monotone_tone_map_for_each_control_over_its_full_range() {
    // With one control at a time, v -> grade(v, L = v) is strictly increasing for neutrals.
    type Setter = fn(&mut GradeParams, f64);
    let setters: [(&str, Setter); 6] = [
        ("contrast", |p, x| p.contrast = x),
        ("brightness", |p, x| p.brightness = x),
        ("shadows", |p, x| p.shadows = x),
        ("highlights", |p, x| p.highlights = x),
        ("black", |p, x| p.black = x),
        ("white", |p, x| p.white = x),
    ];
    let mut worst_slope = f64::INFINITY;
    for (name, set) in setters {
        for x in (-100..=100).step_by(10) {
            let mut p = GradeParams::default();
            set(&mut p, f64::from(x));
            let g = Grade::new(&p);
            let mut prev = g.apply(LogC4([0.0; 3]), 0.0).0[0];
            let mut prev_v = 0.0;
            for k in 1..=2000 {
                let v = f64::from(k) / 2000.0 * 1.2;
                let o = g.apply(LogC4([v; 3]), v).0[0];
                let slope = (o - prev) / (v - prev_v);
                assert!(slope > 0.0, "{name}={x}: slope {slope} at v={v}");
                worst_slope = worst_slope.min(slope);
                prev = o;
                prev_v = v;
            }
        }
    }
    println!("C9 single-control worst neutral-ramp slope: {worst_slope:.4}");
    // Analytic bounds: brightness >= 1 - 0.4 e^-0.5 = 0.757, zone controls >= 1 - e^-0.5 = 0.393,
    // contrast >= 2^-0.6 = 0.66.
    assert!(worst_slope > 0.39, "{worst_slope}");
}

#[test]
fn combined_extremes_report() {
    // Not a spec requirement: how far can the neutral ramp slope fall with every tonal control at an
    // extreme? Printed for the report. Each control alone is monotone (previous test); stacking all
    // four zone controls and brightness at -100 with contrast -100 can fold the ramp below the code
    // value of linear zero (v < 0.0929), where only out-of-gamut negatives live.
    let black_code = logc4_encode(0.0);
    for sign in [-1.0, 1.0] {
        for contrast in [-100.0, 0.0, 100.0] {
            let p = GradeParams {
                contrast,
                brightness: sign * 100.0,
                shadows: sign * 100.0,
                highlights: sign * 100.0,
                black: sign * 100.0,
                white: sign * 100.0,
                ..Default::default()
            };
            let g = Grade::new(&p);
            let mut prev = g.apply(LogC4([0.0; 3]), 0.0).0[0];
            let mut pv = 0.0;
            let (mut worst_all, mut worst_valid) = (f64::INFINITY, f64::INFINITY);
            for k in 1..=4000 {
                let v = f64::from(k) / 4000.0 * 1.2;
                let o = g.apply(LogC4([v; 3]), v).0[0];
                let slope = (o - prev) / (v - pv);
                worst_all = worst_all.min(slope);
                if v >= black_code {
                    worst_valid = worst_valid.min(slope);
                }
                prev = o;
                pv = v;
            }
            println!(
                "C9 all tonal controls {:+}, contrast {contrast:+}: worst neutral slope {worst_all:.3} (v >= black code: {worst_valid:.3})",
                sign * 100.0
            );
            assert!(worst_all.is_finite());
        }
    }
}

#[test]
fn gray_card_unchanged_by_color_only_controls() {
    // Colour balance at zero and saturation leave a neutral at its place when the grade has no tonal change.
    let g = Grade::new(&GradeParams {
        saturation: 80.0,
        ..Default::default()
    });
    // Through linear and back: the same code to the last bits.
    let v = [0.35, 0.35, 0.35];
    let out = g.apply(LogC4(v), 0.35).0;
    for c in out {
        assert!((c - 0.35).abs() < 1e-15, "{out:?}");
    }
    // And sanity of the decode round trip used by the helper.
    assert!((logc4_decode(logc4_encode(0.18)) - 0.18).abs() < 1e-14);
}

proptest! {
    /// Finite inputs and in-range controls never produce NaN or infinity.
    #[test]
    fn finite_in_finite_out(
        c in -100.0f64..=100.0, b in -100.0f64..=100.0, sat in -100.0f64..=100.0,
        r in -0.5f64..2.0, g in -0.5f64..2.0, bl in -0.5f64..2.0, l in -0.5f64..2.0,
    ) {
        let p = GradeParams { contrast: c, brightness: b, saturation: sat, shadows: c, white: b, ..Default::default() };
        let out = Grade::new(&p).apply(LogC4([r, g, bl]), l).0;
        prop_assert!(out.iter().all(|v| v.is_finite()));
        let lin = Grade::new(&p).apply_linear(LinearAwg4([r.abs() * 50.0, g.abs(), bl])).0;
        prop_assert!(lin.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn infinite_color_balance_vectors_stay_finite_and_inside_the_unit_disc() {
    for (a, b) in [
        (f64::INFINITY, 0.0),
        (0.0, f64::NEG_INFINITY),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NAN, f64::INFINITY),
        (1e308, -1e308),
        (f64::MAX, f64::MAX),
    ] {
        let c = ColorBalance { a, b }.clamped();
        assert!(c.a.is_finite() && c.b.is_finite(), "({a}, {b}) -> {c:?}");
        assert!(c.a.hypot(c.b) <= 1.0 + 1e-12, "({a}, {b}) -> {c:?}");
        let p = GradeParams {
            balance_midtones: ColorBalance { a, b },
            ..Default::default()
        };
        let out = Grade::new(&p).apply(LogC4([0.3, 0.4, 0.5]), LOGC4_GRAY).0;
        assert!(out.iter().all(|v| v.is_finite()), "({a}, {b}): {out:?}");
    }
    // An infinite component points along its axis at full radius.
    let c = ColorBalance {
        a: f64::INFINITY,
        b: 0.0,
    }
    .clamped();
    assert!((c.a - 1.0).abs() < 1e-12 && c.b == 0.0);
}

// ---- Against the grade of pipeline 0.4.2 (contrast about grey, subtractive saturation) ----------

/// The grade as it was before the design of 2026-10-19, copied from the old `Grade::apply`: contrast
/// as a gain about grey on every channel, saturation about the brightest channel `M`. At `c = 0` and
/// `s = 0` the new code must give the same bytes.
fn legacy_apply(p: &GradeParams, v: Vec3, l: f64) -> Vec3 {
    let clamp = |x: f64| {
        if x.is_nan() {
            0.0
        } else {
            x.clamp(-100.0, 100.0)
        }
    };
    let contrast_gain = (CONTRAST_EXPONENT * clamp(p.contrast) / 100.0).exp2();
    let brightness = clamp(p.brightness) / 100.0;
    let zone_amounts = [p.shadows, p.highlights, p.black, p.white].map(|x| clamp(x) / 100.0);
    let zone_centers = ZONE_CENTER_STOPS.map(logc4_at_stops);
    let balance = [
        p.balance_shadows.clamped(),
        p.balance_midtones.clamped(),
        p.balance_highlights.clamped(),
    ];
    let balance_centers = BALANCE_CENTER_STOPS.map(logc4_at_stops);
    let saturation_gain = 1.0 + clamp(p.saturation) / 100.0;

    let mut out = v;
    if contrast_gain != 1.0 {
        for c in &mut out {
            *c = LOGC4_GRAY + (*c - LOGC4_GRAY) * contrast_gain;
        }
    }
    let lift = brightness * LOGC4_STOP * bump(l, LOGC4_GRAY, BRIGHTNESS_SIGMA);
    let mut shift = lift;
    for (amount, center) in zone_amounts.iter().zip(&zone_centers) {
        if *amount != 0.0 {
            shift += amount * ZONE_CONTROL_SIGMA * bump(l, *center, ZONE_CONTROL_SIGMA);
        }
    }
    for c in &mut out {
        *c += shift;
    }
    for (bal, center) in balance.iter().zip(&balance_centers) {
        if bal.a != 0.0 || bal.b != 0.0 {
            let w = COLOR_BALANCE_STRENGTH * bump(l, *center, ZONE_SIGMA);
            for k in 0..3 {
                out[k] += w * (bal.a * E1[k] + bal.b * E2[k]);
            }
        }
    }
    if saturation_gain != 1.0 {
        let m = out[0].max(out[1]).max(out[2]);
        for c in &mut out {
            *c = m + (*c - m) * saturation_gain;
        }
    }
    out
}

fn bits(v: Vec3) -> [u64; 3] {
    v.map(f64::to_bits)
}

/// Every control but Contrast and Saturation (which are 0), at random values.
fn any_params() -> impl Strategy<Value = GradeParams> {
    let c = || -100.0f64..=100.0;
    let w = || -0.7f64..=0.7;
    ((c(), c(), c(), c(), c()), (w(), w(), w(), w(), w(), w())).prop_map(
        |((brightness, shadows, highlights, black, white), (a1, b1, a2, b2, a3, b3))| GradeParams {
            contrast: 0.0,
            brightness,
            shadows,
            highlights,
            black,
            white,
            balance_shadows: ColorBalance { a: a1, b: b1 },
            balance_midtones: ColorBalance { a: a2, b: b2 },
            balance_highlights: ColorBalance { a: a3, b: b3 },
            saturation: 0.0,
        },
    )
}

fn any_logc4() -> impl Strategy<Value = Vec3> {
    prop::array::uniform3(-0.2f64..1.3)
}

/// A linear AWG4 pixel: mostly positive, from 2^-12 to 2^6 of grey, with some channels below 0.
fn any_linear() -> impl Strategy<Value = Vec3> {
    (
        -12.0f64..6.0,
        prop::array::uniform3(-0.15f64..1.0),
        prop::array::uniform3(0.0f64..1.0),
    )
        .prop_map(|(stops, mix, below)| {
            let lum = 0.18 * stops.exp2();
            let mut rgb = mix.map(|m| lum * (0.05 + m));
            // About one pixel in eight has a channel below 0 (out-of-gamut colours).
            if below[0] < 0.125 {
                rgb[(below[1] * 3.0) as usize % 3] = -lum * 0.05 * below[2];
            }
            rgb
        })
}

#[test]
fn the_legacy_copy_differs_from_the_new_grade_when_contrast_or_saturation_is_on() {
    // Guards the byte-identity test below against being vacuous.
    let v = [0.5, 0.3, 0.1];
    for p in [
        GradeParams {
            saturation: 50.0,
            ..Default::default()
        },
        GradeParams {
            contrast: 50.0,
            ..Default::default()
        },
    ] {
        let new = Grade::new(&p).apply(LogC4(v), 0.33).0;
        let old = legacy_apply(&p, v, 0.33);
        assert_ne!(bits(new), bits(old), "{p:?}");
    }
    // The old saturation kept the brightest channel and lowered the others (darker); the new one
    // keeps the luminance.
    let p = GradeParams {
        saturation: 50.0,
        ..Default::default()
    };
    let y = |v: Vec3| awg4_luminance(v.map(logc4_decode));
    let new = Grade::new(&p).apply(LogC4(v), 0.33).0;
    let old = legacy_apply(&p, v, 0.33);
    assert!(y(old) < 0.9 * y(v));
    assert!((y(new) - y(v)).abs() < 1e-13);
    // The old contrast moved black and the top code value; the new one keeps both.
    let p = GradeParams {
        contrast: 60.0,
        ..Default::default()
    };
    for x in [CONTRAST_BLACK, CONTRAST_TOP] {
        assert_eq!(Grade::new(&p).apply(LogC4([x; 3]), x).0, [x; 3]);
        assert!((legacy_apply(&p, [x; 3], x)[0] - x).abs() > 0.02);
    }
}

proptest! {
    /// Contrast 0 and Saturation 0 render exactly as before the change, for every other control and
    /// any pixel.
    #[test]
    fn contrast_and_saturation_zero_are_byte_identical_to_the_old_grade(
        p in any_params(), v in any_logc4(), l in -0.2f64..1.3,
        negative_zero in any::<(bool, bool)>(),
    ) {
        let p = GradeParams {
            contrast: if negative_zero.0 { -0.0 } else { 0.0 },
            saturation: if negative_zero.1 { -0.0 } else { 0.0 },
            ..p
        };
        let got = Grade::new(&p).apply(LogC4(v), l).0;
        prop_assert_eq!(bits(got), bits(legacy_apply(&p, v, l)));
    }

    /// The luminance stays and the chroma keeps its direction (the hue): `rgb' - Y` is `rgb - Y`
    /// times one factor, at least 0 (exactly 0 at -100), for any pixel and any setting.
    #[test]
    fn saturation_keeps_the_luminance_and_the_hue(rgb in any_linear(), s in -100.0f64..=100.0) {
        let y = awg4_luminance(rgb);
        let out = saturate(LinearAwg4(rgb), s).0;
        let scale = rgb.iter().fold(0.0f64, |a, c| a.max(c.abs()));
        prop_assert!((awg4_luminance(out) - y).abs() <= 1e-14 * scale, "{rgb:?} s={s}: {out:?}");
        // The common factor, from the channel farthest from Y.
        let far = (0..3)
            .max_by(|&a, &b| (rgb[a] - y).abs().total_cmp(&(rgb[b] - y).abs()))
            .unwrap_or(0);
        let lambda = (out[far] - y) / (rgb[far] - y);
        prop_assert!(lambda >= -1e-12, "{rgb:?} s={s}: lambda {lambda}");
        for k in 0..3 {
            prop_assert!(
                (out[k] - y - lambda * (rgb[k] - y)).abs() <= 1e-13 * scale,
                "{rgb:?} s={s}: {out:?}"
            );
        }
        if s < 0.0 {
            prop_assert!((lambda - (1.0 + s / 100.0)).abs() < 1e-12);
        } else {
            prop_assert!(lambda >= 1.0 - 1e-12 && lambda <= 1.0 + SATURATION_BOOST * s / 100.0 + 1e-12);
        }
    }

    /// For s > 0 no channel goes below zero: the darkest lands at `Y (1 - r') >= 0` (or above `Y`
    /// where every channel is above it), and a pixel with `Y <= 0` or a channel at or below 0 is
    /// left exactly as it is.
    #[test]
    fn positive_saturation_never_takes_a_channel_below_zero(rgb in any_linear(), s in 0.0f64..=100.0) {
        let y = awg4_luminance(rgb);
        let min = rgb[0].min(rgb[1]).min(rgb[2]);
        let out = saturate(LinearAwg4(rgb), s).0;
        if y <= 0.0 || min <= 0.0 {
            prop_assert_eq!(bits(out), bits(rgb));
        } else {
            let min_out = out[0].min(out[1]).min(out[2]);
            prop_assert!(min_out >= -1e-15 * y, "{rgb:?} s={s}: {out:?}");
            let r = (y - min) / y;
            if r >= SATURATION_SMALL_R {
                let f = 1.0 + SATURATION_BOOST * s / 100.0;
                let r2 = 1.0 - (1.0 - r).powf(f);
                prop_assert!((min_out - y * (1.0 - r2)).abs() <= 1e-12 * y, "{rgb:?} s={s}");
            } else {
                prop_assert!(min_out >= y * (1.0 - 1e-12));
            }
        }
    }

    /// The chroma grows with s, channel by channel, and the luminance does not move.
    #[test]
    fn saturation_is_monotone_in_its_value(rgb in any_linear(), s in -100.0f64..100.0, ds in 0.0f64..50.0) {
        let y = awg4_luminance(rgb);
        let a = saturate(LinearAwg4(rgb), s).0;
        let b = saturate(LinearAwg4(rgb), (s + ds).min(100.0)).0;
        let scale = rgb.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        for k in 0..3 {
            prop_assert!((b[k] - y).abs() >= (a[k] - y).abs() - 1e-13 * scale, "{rgb:?} {s} +{ds}");
        }
    }

    /// Neutral pixels are unchanged (up to rounding), at any setting.
    #[test]
    fn neutral_pixels_are_unchanged_by_saturation(x in -0.5f64..500.0, s in -100.0f64..=100.0) {
        let out = saturate(LinearAwg4([x; 3]), s).0;
        for c in out {
            prop_assert!((c - x).abs() <= 1e-14 * x.abs().max(1e-3), "{x} s={s}: {out:?}");
        }
    }

    /// s = -100 is the grey of the pixel's own luminance, whatever came before.
    #[test]
    fn minus_100_is_the_grey_of_the_same_luminance(
        p in any_params(), v in any_logc4(), l in -0.2f64..1.3,
    ) {
        let without = Grade::new(&p).apply(LogC4(v), l).0;
        let with = Grade::new(&GradeParams { saturation: -100.0, ..p }).apply(LogC4(v), l).0;
        let y = awg4_luminance(without.map(logc4_decode));
        for c in with {
            prop_assert!((c - logc4_encode(y)).abs() < 1e-13);
        }
    }
}

#[test]
fn the_chroma_scale_falls_from_f_to_one_as_the_colour_gets_stronger() {
    for s in [1.0, 10.0, 30.0, 60.0, 100.0] {
        let f = 1.0 + SATURATION_BOOST * s / 100.0;
        // The limits: F for a neutral (and below it), 1 at a channel on 0.
        assert_eq!(saturation_chroma_scale(0.0, f), f);
        assert_eq!(saturation_chroma_scale(-0.5, f), f);
        assert_eq!(saturation_chroma_scale(-1e30, f), f);
        assert_eq!(saturation_chroma_scale(SATURATION_SMALL_R / 2.0, f), f);
        assert_eq!(saturation_chroma_scale(1.0, f), 1.0);
        assert_eq!(saturation_chroma_scale(7.0, f), 1.0);
        // Continuous at the small-r switch and at r = 1.
        assert!((saturation_chroma_scale(SATURATION_SMALL_R, f) - f).abs() < 1e-8);
        assert!((saturation_chroma_scale(1.0 - 1e-12, f) - 1.0).abs() < 1e-9);
        // Muted colours gain more than strong ones: non-increasing in r, between 1 and F.
        let mut prev = f;
        for i in 0..=2000 {
            let r = f64::from(i) / 2000.0 * (1.0 - 1e-9);
            let lambda = saturation_chroma_scale(r, f);
            assert!(lambda <= prev + 1e-15, "s={s} r={r}: {lambda} > {prev}");
            assert!(
                (1.0 - 1e-15..=f + 1e-15).contains(&lambda),
                "s={s} r={r}: {lambda}"
            );
            // The same number as the spec's 1 - (1 - r)^F over r, where that does not cancel.
            if r > 1e-3 {
                let direct = (1.0 - (1.0 - r).powf(f)) / r;
                assert!((lambda - direct).abs() < 1e-12, "s={s} r={r}");
            }
            prev = lambda;
        }
    }
    // And non-decreasing in s at every r.
    for i in 0..=100 {
        let r = f64::from(i) / 100.0 * 0.999;
        let mut prev = 1.0;
        for s in 0..=100 {
            let lambda = saturation_chroma_scale(r, 1.0 + SATURATION_BOOST * f64::from(s) / 100.0);
            assert!(lambda >= prev - 1e-15, "r={r} s={s}");
            prev = lambda;
        }
    }
}

#[test]
fn deep_blues_with_every_channel_above_the_luminance_get_the_plain_factor() {
    // The blue weight of Y is negative: this pixel has Y = 0.0002 and every channel above it, so r is
    // about -100. The spec's r' / r would scale its chroma by about 150 at s = 100; it gets F = 2.5.
    let rgb = [0.02, 0.02, 0.57];
    let y = awg4_luminance(rgb);
    assert!(y > 0.0 && y < 3e-4, "{y}");
    let out = saturate(LinearAwg4(rgb), 100.0).0;
    let f = 1.0 + ALPHA;
    for k in 0..3 {
        assert!((out[k] - (y + f * (rgb[k] - y))).abs() < 1e-15, "{out:?}");
    }
    // A blue with Y <= 0 is left alone.
    let rgb = [0.01, 0.003, 0.2];
    assert!(awg4_luminance(rgb) <= 0.0);
    assert_eq!(saturate(LinearAwg4(rgb), 60.0).0, rgb);
}

#[test]
fn pixels_saturation_leaves_alone_keep_their_codes_bit_for_bit() {
    let g = Grade::new(&GradeParams {
        saturation: 70.0,
        ..Default::default()
    });
    for lin in [
        [0.4, -0.01, 0.1],  // a channel below 0
        [0.3, 0.0, 0.2],    // a channel at 0
        [0.01, 0.003, 0.2], // Y <= 0
        [-0.01, -0.02, -0.005],
    ] {
        let v = LinearAwg4(lin).to_logc4().0;
        assert_eq!(bits(g.apply(LogC4(v), 0.3).0), bits(v), "{lin:?}");
    }
    // Codes above the limit are passed on (the f32 kernel could not decode them).
    for v in [[SATURATION_CODE_LIMIT + 0.1, 0.5, 0.4], [9.0, 9.5, 8.0]] {
        for s in [-100.0, -30.0, 30.0, 100.0] {
            let g = Grade::new(&GradeParams {
                saturation: s,
                ..Default::default()
            });
            assert_eq!(bits(g.apply(LogC4(v), 0.3).0), bits(v));
        }
    }
    // At the limit itself saturation still acts.
    let v = [SATURATION_CODE_LIMIT, 7.0, 6.5];
    assert_ne!(bits(g.apply(LogC4(v), 0.3).0), bits(v));
}

#[test]
fn neutral_ramp_is_unchanged_for_every_saturation_value() {
    for s in -100..=100 {
        let g = Grade::new(&GradeParams {
            saturation: f64::from(s),
            ..Default::default()
        });
        for k in 0..=60 {
            let x = -0.1 + 1.4 * f64::from(k) / 60.0;
            for l in [0.0, x, 0.8] {
                let out = g.apply(LogC4([x; 3]), l).0;
                for c in out {
                    assert!((c - x).abs() < 1e-14, "s={s} x={x}: {out:?}");
                }
            }
        }
    }
}

#[test]
fn saturation_keeps_the_luminance_of_saturated_patches() {
    let patches: [Vec3; 10] = [
        [0.80, 0.05, 0.03],
        [0.04, 0.60, 0.06],
        [0.03, 0.08, 0.70],
        [0.55, 0.50, 0.02],
        [0.03, 0.45, 0.50],
        [0.60, 0.04, 0.55],
        [0.35, 0.18, 0.12],
        [0.020, 0.010, 0.004],
        [6.0, 1.2, 0.3],
        [0.40, 0.02, -0.003],
    ];
    for lin in patches {
        let v = LinearAwg4(lin).to_logc4().0;
        let l = grade_luma(LinearAwg4(lin));
        let y = awg4_luminance(lin);
        let mut prev_chroma = 0.0;
        for s in (-100..=100).step_by(10) {
            let out = Grade::new(&GradeParams {
                saturation: f64::from(s),
                ..Default::default()
            })
            .apply(LogC4(v), l)
            .0
            .map(logc4_decode);
            // Neither darker nor lighter at any setting.
            assert!((awg4_luminance(out) - y).abs() < 1e-12 * y, "{lin:?} s={s}");
            // The chroma grows with the control.
            let chroma: f64 = out.iter().map(|c| (c - y).powi(2)).sum::<f64>().sqrt();
            assert!(chroma >= prev_chroma - 1e-12, "{lin:?} s={s}");
            prev_chroma = chroma;
            // For s > 0 no channel that was above 0 goes below it.
            if s > 0 && lin.iter().all(|&c| c > 0.0) {
                assert!(out.iter().all(|&c| c >= 0.0), "{lin:?} s={s}: {out:?}");
            }
        }
    }
}
