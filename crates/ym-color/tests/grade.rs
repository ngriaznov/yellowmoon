//! M1-09 (C9): the primary grade. Formulas against an independent transcription of the table in
//! docs/02, and monotonicity of every control (what "monotonic" means per control is documented in
//! `ym_color::grade`).

use proptest::prelude::*;
use ym_color::consts::{LOGC4_GRAY, LOGC4_STOP, Vec3};
use ym_color::gamut::LinearAwg4;
use ym_color::grade::{
    BALANCE_CENTER_STOPS, BRIGHTNESS_SIGMA, COLOR_BALANCE_STRENGTH, CONTRAST_EXPONENT,
    ColorBalance, E1, E2, Grade, GradeParams, ZONE_CENTER_STOPS, ZONE_CONTROL_SIGMA, ZONE_SIGMA,
    bump, grade_luma, logc4_at_stops,
};
use ym_color::logc::{LogC4, logc4_decode, logc4_encode};

/// Straight transcription of the C9 table, independent of `Grade` (no precomputation).
fn reference(p: &GradeParams, v: Vec3, l: f64) -> Vec3 {
    let g = 0.2783958365482653;
    let s = 0.9071358748778103 / 14.0;
    let mut v = v;
    let c = p.contrast;
    for x in &mut v {
        *x = g + (*x - g) * 2.0_f64.powf(0.6 * c / 100.0);
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
    let e1 = [
        1.0 / 1.5_f64.sqrt(),
        -0.5 / 1.5_f64.sqrt(),
        -0.5 / 1.5_f64.sqrt(),
    ];
    let e2 = [0.0, 1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()];
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
    // Saturation is subtractive: about the brightest channel of the pixel at this point.
    let m = v[0].max(v[1]).max(v[2]);
    for x in &mut v {
        *x = m + (*x - m) * (1.0 + p.saturation / 100.0);
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
}

#[test]
fn contrast_formula_and_fixed_point() {
    let g = Grade::new(&GradeParams {
        contrast: 100.0,
        ..Default::default()
    });
    // Grey is the fixed point.
    let out = g.apply(LogC4([LOGC4_GRAY; 3]), LOGC4_GRAY).0;
    for c in out {
        assert!((c - LOGC4_GRAY).abs() < 1e-15);
    }
    // 2^0.6 about grey for +100.
    let out = g.apply(LogC4([0.5, 0.2, 0.7]), 0.4).0;
    let gain = 2.0_f64.powf(0.6);
    for (o, i) in out.iter().zip([0.5, 0.2, 0.7]) {
        assert!((o - (LOGC4_GRAY + (i - LOGC4_GRAY) * gain)).abs() < 1e-14);
    }
    // -100 is the inverse gain.
    let g = Grade::new(&GradeParams {
        contrast: -100.0,
        ..Default::default()
    });
    let out = g.apply(LogC4([0.5; 3]), 0.5).0;
    assert!((out[0] - (LOGC4_GRAY + (0.5 - LOGC4_GRAY) / gain)).abs() < 1e-14);
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
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-13,
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
            prop_assert!((got[i] - want[i]).abs() < 1e-12);
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
fn saturation_pivots_on_the_brightest_channel() {
    // M = 0.5 is the red channel; L (0.33) plays no part.
    let v = [0.5, 0.3, 0.1];
    let l = 0.33;
    let m = 0.5;
    let sat = |s: f64| {
        Grade::new(&GradeParams {
            saturation: s,
            ..Default::default()
        })
        .apply(LogC4(v), l)
        .0
    };
    // -100: every channel equals M, a neutral at the pixel's brightest channel.
    assert_eq!(sat(-100.0), [m, m, m]);
    // +100: the distance below M doubles, M stays.
    let double = sat(100.0);
    assert_eq!(double[0], m);
    for k in 0..3 {
        assert!((double[k] - (m + 2.0 * (v[k] - m))).abs() < 1e-15);
    }
    assert!((double[1] - 0.1).abs() < 1e-15 && (double[2] - (-0.3)).abs() < 1e-15);
    // -50: halfway to M.
    let half = sat(-50.0);
    for k in 0..3 {
        assert!((half[k] - (m + 0.5 * (v[k] - m))).abs() < 1e-15);
    }
    // More saturation makes the colour denser and darker, less makes it paler and lighter.
    assert!(double[1] < v[1] && double[2] < v[2]);
    assert!(half[1] > v[1] && half[2] > v[2]);
    assert!(half[1] < m && half[2] < m);
    // The pivot does not depend on L.
    for l in [0.0, 0.2, 0.9, 5.0] {
        let g = Grade::new(&GradeParams {
            saturation: 63.0,
            ..Default::default()
        });
        assert_eq!(g.apply(LogC4(v), l), g.apply(LogC4(v), 0.33));
    }
}

#[test]
fn saturation_pivots_on_the_brightest_channel_after_the_controls_before_it() {
    // M is the brightest channel at this point of C9: with contrast on, it is the contrasted one.
    let v = [0.55, 0.20, 0.35];
    let p = GradeParams {
        contrast: 60.0,
        saturation: -100.0,
        ..Default::default()
    };
    let out = Grade::new(&p).apply(LogC4(v), 0.4).0;
    let gain = (CONTRAST_EXPONENT * 0.6).exp2();
    let m = LOGC4_GRAY + (0.55 - LOGC4_GRAY) * gain;
    for c in out {
        assert!((c - m).abs() < 1e-14, "{out:?} vs {m}");
    }
    // And a lift from brightness moves M with the pixel.
    let p = GradeParams {
        brightness: 80.0,
        saturation: -100.0,
        ..Default::default()
    };
    let l = LOGC4_GRAY;
    let out = Grade::new(&p).apply(LogC4(v), l).0;
    let m = 0.55 + 0.8 * LOGC4_STOP;
    for c in out {
        assert!((c - m).abs() < 1e-14, "{out:?} vs {m}");
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
    for k in 0..=40 {
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
    // Contrast: moves away from grey as the value rises (above grey up, below grey down).
    let grades = sweep(|p, x| p.contrast = x);
    for (v, l) in sample_pixels() {
        let mut prev = grades[0].apply(LogC4(v), l).0;
        for g in &grades[1..] {
            let cur = g.apply(LogC4(v), l).0;
            for k in 0..3 {
                if v[k] > LOGC4_GRAY {
                    assert!(cur[k] >= prev[k] - 1e-15, "contrast up: {v:?}");
                } else if v[k] < LOGC4_GRAY {
                    assert!(cur[k] <= prev[k] + 1e-15, "contrast down: {v:?}");
                }
            }
            prev = cur;
        }
    }
    // Saturation: the distance of each channel from M (the brightest channel) grows with the
    // control; M and neutral pixels do not move.
    let grades = sweep(|p, x| p.saturation = x);
    for (v, l) in sample_pixels() {
        let m = v[0].max(v[1]).max(v[2]);
        let mut prev = [0.0; 3];
        for (i, g) in grades.iter().enumerate() {
            let o = g.apply(LogC4(v), l).0;
            assert_eq!(o[0].max(o[1]).max(o[2]), m, "saturation moved M: {v:?}");
            for k in 0..3 {
                let dist = (o[k] - m).abs();
                if i > 0 {
                    assert!(dist >= prev[k] - 1e-15, "saturation: {v:?} channel {k}");
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
    let v = [0.35, 0.35, 0.35];
    assert_eq!(g.apply(LogC4(v), 0.35).0, v);
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

// ---- Subtractive saturation (owner's decision of 2026-10-13, docs/16) ---------------------------

/// The grade as it was before that decision, copied from the old `Grade::apply`: saturation about
/// the luma `L`. At `s = 0` the new code must give the same bytes.
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
        for c in &mut out {
            *c = l + (*c - l) * saturation_gain;
        }
    }
    out
}

fn bits(v: Vec3) -> [u64; 3] {
    v.map(f64::to_bits)
}

fn max3(v: Vec3) -> f64 {
    v[0].max(v[1]).max(v[2])
}

/// Every control but Saturation (which is 0), at random values.
fn any_params() -> impl Strategy<Value = GradeParams> {
    let c = || -100.0f64..=100.0;
    let w = || -0.7f64..=0.7;
    (
        (c(), c(), c(), c(), c(), c()),
        (w(), w(), w(), w(), w(), w()),
    )
        .prop_map(
            |(
                (contrast, brightness, shadows, highlights, black, white),
                (a1, b1, a2, b2, a3, b3),
            )| {
                GradeParams {
                    contrast,
                    brightness,
                    shadows,
                    highlights,
                    black,
                    white,
                    balance_shadows: ColorBalance { a: a1, b: b1 },
                    balance_midtones: ColorBalance { a: a2, b: b2 },
                    balance_highlights: ColorBalance { a: a3, b: b3 },
                    saturation: 0.0,
                }
            },
        )
}

fn any_logc4() -> impl Strategy<Value = Vec3> {
    prop::array::uniform3(-0.2f64..1.3)
}

#[test]
fn the_legacy_copy_differs_from_the_new_grade_when_saturation_is_on() {
    // Guards the byte-identity test below against being vacuous.
    let p = GradeParams {
        saturation: 50.0,
        ..Default::default()
    };
    let v = [0.5, 0.3, 0.1];
    let new = Grade::new(&p).apply(LogC4(v), 0.33).0;
    let old = legacy_apply(&p, v, 0.33);
    assert_ne!(bits(new), bits(old));
    // The old pivot was L: the old code moved the pixel's brightest channel.
    assert!((old[0] - 0.5).abs() > 1e-3 && new[0] == 0.5);
}

proptest! {
    /// Saturation 0 renders exactly as before the change, for every other control and any pixel.
    #[test]
    fn saturation_zero_is_byte_identical_to_the_old_grade(
        p in any_params(), v in any_logc4(), l in -0.2f64..1.3, negative_zero in any::<bool>(),
    ) {
        let p = GradeParams { saturation: if negative_zero { -0.0 } else { 0.0 }, ..p };
        let got = Grade::new(&p).apply(LogC4(v), l).0;
        prop_assert_eq!(bits(got), bits(legacy_apply(&p, v, l)));
    }

    /// With only Saturation moved, the brightest channel stays and no channel crosses it.
    #[test]
    fn saturation_keeps_the_brightest_channel_and_the_others_move_one_way(
        p in any_params(), s in -100.0f64..=100.0, v in any_logc4(), l in -0.2f64..1.3,
    ) {
        let without = Grade::new(&p).apply(LogC4(v), l).0;
        let with = Grade::new(&GradeParams { saturation: s, ..p }).apply(LogC4(v), l).0;
        let m = max3(without);
        prop_assert_eq!(max3(with), m);
        let gain = 1.0 + s / 100.0;
        for k in 0..3 {
            // The formula, from the pixel the controls before Saturation produced.
            prop_assert!(((with[k] - m) - (without[k] - m) * gain).abs() < 1e-12);
            if without[k] == m {
                prop_assert_eq!(with[k], m);
            }
            if s > 0.0 {
                prop_assert!(with[k] <= without[k] + 1e-13, "s={s}: {without:?} -> {with:?}");
            } else if s < 0.0 {
                prop_assert!(with[k] >= without[k] - 1e-13 && with[k] <= m + 1e-13, "s={s}: {without:?} -> {with:?}");
            }
        }
    }

    /// Neutral pixels (after the tonal controls, which keep a neutral neutral) are unchanged.
    #[test]
    fn neutral_pixels_are_unchanged_by_saturation_at_any_setting(
        p in any_params(), s in -100.0f64..=100.0, x in -0.2f64..1.3, l in -0.2f64..1.3,
    ) {
        let p = GradeParams {
            balance_shadows: ColorBalance::default(),
            balance_midtones: ColorBalance::default(),
            balance_highlights: ColorBalance::default(),
            ..p
        };
        let without = Grade::new(&p).apply(LogC4([x; 3]), l).0;
        let with = Grade::new(&GradeParams { saturation: s, ..p }).apply(LogC4([x; 3]), l).0;
        prop_assert_eq!(without[0], without[1]);
        prop_assert_eq!(without[1], without[2]);
        prop_assert_eq!(with, without);
    }

    /// s = -100 makes every channel equal to M, whatever came before.
    #[test]
    fn minus_100_collapses_every_channel_onto_the_brightest(
        p in any_params(), v in any_logc4(), l in -0.2f64..1.3,
    ) {
        let without = Grade::new(&p).apply(LogC4(v), l).0;
        let with = Grade::new(&GradeParams { saturation: -100.0, ..p }).apply(LogC4(v), l).0;
        prop_assert_eq!(with, [max3(without); 3]);
    }
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
                assert_eq!(g.apply(LogC4([x; 3]), l).0, [x; 3], "s={s} x={x}");
            }
        }
    }
}

#[test]
fn saturation_does_not_raise_luma_on_saturated_patches() {
    // Rec.709 luma of the linear values the LogC4 triple stands for.
    let luma = |v: Vec3| {
        0.2126 * logc4_decode(v[0]) + 0.7152 * logc4_decode(v[1]) + 0.0722 * logc4_decode(v[2])
    };
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
        let neutral = luma([max3(v); 3]);
        let base = luma(v);
        let mut prev = f64::INFINITY;
        // Luma falls as the control rises, everywhere on the scale.
        for s in (-100..=100).step_by(10) {
            let out = Grade::new(&GradeParams {
                saturation: f64::from(s),
                ..Default::default()
            })
            .apply(LogC4(v), l)
            .0;
            let y = luma(out);
            assert!(y <= prev + 1e-12, "{lin:?}: luma rose at s={s}");
            prev = y;
            if s > 0 {
                assert!(
                    y < base,
                    "{lin:?}: luma did not fall at s={s}: {y} vs {base}"
                );
            } else if s < 0 {
                assert!(y > base && y <= neutral + 1e-12, "{lin:?}: s={s}");
            }
        }
        // s = -100 is the neutral at the brightest channel, the lightest the pixel gets.
        let out = Grade::new(&GradeParams {
            saturation: -100.0,
            ..Default::default()
        })
        .apply(LogC4(v), l)
        .0;
        assert!((luma(out) - neutral).abs() < 1e-12 * neutral.abs().max(1.0));
    }
}
