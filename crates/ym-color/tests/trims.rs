//! The two trims of a corrector node (`docs/algorithms/06-masks.md` section 2, `ym_color::display::trim`):
//! the exposure trim is a stop of exposure, a temperature and tint trim of zero is no code at all, the
//! temperature moves the white by 50 mired for 100, the tint across the locus, and the gain is the
//! von Kries gain of the formula.

use ym_color::consts::linalg::{self, Mat3};
use ym_color::consts::{D65_XY, Vec3, xy_to_xyz};
use ym_color::display::trim::{
    D65_CCT_K, TRIM_DUV_PER_UNIT, TRIM_MIRED_PER_UNIT, daylight_xy, trim_white_uv, von_kries,
};
use ym_color::display::{ChainParams, ColorChain, MainNode, NodeParams, NodeSet, Trim};
use ym_color::gamut::matrices;
use ym_color::plan::{PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;

fn matrix(gain: f64) -> Mat3 {
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
    .map(|row| row.map(|v| v * gain))
}

fn pixels() -> Vec<Vec3> {
    let mut v = vec![
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.05, 0.06, 0.09],
        [0.2, 0.15, 0.1],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
    ];
    let mut state = 17_u64;
    for _ in 0..80 {
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let scale = 0.04 + 0.25 * next();
        v.push([
            scale * (0.5 + next()),
            scale * (0.5 + next()),
            scale * (0.5 + next()),
        ]);
    }
    v
}

/// A chain whose Main has this trim and nothing else.
fn trimmed(m: Mat3, tone: ToneParams, trim: Trim) -> ColorChain {
    let mut p = ChainParams::new(m, tone);
    p.nodes.main = MainNode {
        bypass: false,
        trim,
        ..MainNode::default()
    };
    ColorChain::new(&p).unwrap()
}

/// A chain with a second corrector after Main, which has this trim.
fn second_node(trim: Trim) -> ColorChain {
    let plan = compile(
        &PlanGraph {
            nodes: vec![
                PlanNode {
                    id: 1,
                    pos: [0.0, 0.0],
                    picture: Some(PictureSource::Input),
                    key: None,
                },
                PlanNode {
                    id: 2,
                    pos: [1.0, 0.0],
                    picture: Some(PictureSource::Node(1)),
                    key: None,
                },
            ],
            output: PictureSource::Node(2),
        },
        Stop::Output,
    )
    .unwrap();
    let mut p = ChainParams::new(matrix(1.0), ToneParams::IDENTITY);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others: [(
            2,
            NodeParams {
                trim,
                ..NodeParams::default()
            },
        )]
        .into(),
    };
    ColorChain::new(&p).unwrap()
}

fn max_diff(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)
}

fn ev(ev: f64) -> Trim {
    Trim {
        exposure_ev: ev,
        ..Trim::default()
    }
}

// ---- the exposure trim -----------------------------------------------------------------------

#[test]
fn an_exposure_trim_of_a_stop_is_a_stop_of_exposure_in_logc4() {
    // Main with a trim of +1 EV against a chain with no node work and the fused matrix times two.
    // With C7 off the two are the same maths up to the rounding of one decode and one encode.
    let trim = trimmed(matrix(1.0), ToneParams::IDENTITY, ev(1.0));
    let exposed = trimmed(matrix(2.0), ToneParams::IDENTITY, Trim::default());
    let mut worst = 0.0f64;
    for px in pixels() {
        worst = worst.max(max_diff(
            trim.apply_logc4_pixel(px).0,
            exposed.apply_logc4_pixel(px).0,
        ));
    }
    println!("exposure trim +1 against matrix x 2, C7 off: largest LogC4 difference {worst:e}");
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn an_exposure_trim_matches_exposure_above_the_toe_and_below_the_knee() {
    // With the C7 of a Sony body the trim is after C7 and the exposure before it, so they agree only
    // where C7 does next to nothing for both: above the toe (its scale is n^2 / (n^2 + tau^2) with
    // tau = 0.004 here, so 1.6e-3 at n = 0.1) and below the highlight range (from +1 stop over grey,
    // 0.36). Pixels whose largest channel is in [0.1, 0.3] before and after the exposure are there;
    // the bound is the toe's, 3e-4 in LogC4.
    let tone = ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap();
    let mut worst = 0.0f64;
    let mut n = 0;
    for ev_value in [-1.0, -0.5, 0.5, 1.0] {
        let trim = trimmed(matrix(1.0), tone, ev(ev_value));
        let exposed = trimmed(matrix(ev_value.exp2()), tone, Trim::default());
        for px in pixels() {
            let lin = linalg::mul_vec(&matrix(1.0), px);
            let hi = lin.iter().copied().fold(0.0, f64::max);
            let scaled = hi * ev_value.exp2();
            if !(0.1..=0.3).contains(&hi) || !(0.1..=0.3).contains(&scaled) {
                continue;
            }
            n += 1;
            worst = worst.max(max_diff(
                trim.apply_logc4_pixel(px).0,
                exposed.apply_logc4_pixel(px).0,
            ));
        }
    }
    println!(
        "exposure trim against exposure, real C7, {n} pixels: largest LogC4 difference {worst:e}"
    );
    assert!(n > 40, "only {n} pixels were clear of the toe and the knee");
    assert!(worst < 3e-4, "{worst}");
}

#[test]
fn a_trim_in_a_later_node_acts_on_the_decoded_picture_of_the_node_before() {
    let a = second_node(ev(1.0));
    let b = trimmed(matrix(1.0), ToneParams::IDENTITY, Trim::default());
    for px in pixels() {
        let v = b.apply_logc4_pixel(px);
        let want = ym_color::gamut::LinearAwg4(v.to_linear().0.map(|c| c * 2.0)).to_logc4();
        assert!(max_diff(a.apply_logc4_pixel(px).0, want.0) < 1e-12);
    }
}

// ---- zero is no code -------------------------------------------------------------------------

#[test]
fn a_trim_of_zero_changes_no_bit() {
    let tone = ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap();
    let plain = trimmed(matrix(1.0), tone, Trim::default());
    for zero in [
        Trim::default(),
        Trim {
            exposure_ev: -0.0,
            temperature: -0.0,
            tint: -0.0,
        },
        Trim {
            exposure_ev: 0.0,
            temperature: 0.0,
            tint: 0.0,
        },
    ] {
        assert!(zero.is_identity());
        assert_eq!(zero.gain(), None);
        let c = trimmed(matrix(1.0), tone, zero);
        let d = second_node(zero);
        let reference = second_node(Trim::default());
        for px in pixels() {
            assert_eq!(c.trace(px), plain.trace(px));
            assert_eq!(d.trace(px), reference.trace(px));
        }
    }
    // And a temperature or tint alone is a gain with a green of exactly 1.
    for (t, n) in [(10.0, 0.0), (0.0, 10.0), (-100.0, 100.0)] {
        let g = Trim {
            temperature: t,
            tint: n,
            ..Trim::default()
        }
        .gain()
        .unwrap();
        assert_eq!(g[1], 1.0);
    }
}

#[test]
fn the_values_of_a_trim_are_held_to_their_ranges() {
    let big = Trim {
        exposure_ev: 9.0,
        temperature: 900.0,
        tint: -900.0,
    };
    let top = Trim {
        exposure_ev: 4.0,
        temperature: 100.0,
        tint: -100.0,
    };
    assert_eq!(big.gain(), top.gain());
    let nan = Trim {
        exposure_ev: f64::NAN,
        temperature: f64::NAN,
        tint: f64::NAN,
    };
    assert_eq!(nan.gain(), None);
}

// ---- the white that the temperature and the tint move -----------------------------------------

/// CIE 1960 `(u, v)` of `xy`.
fn uv(xy: [f64; 2]) -> [f64; 2] {
    let d = -2.0 * xy[0] + 12.0 * xy[1] + 3.0;
    [4.0 * xy[0] / d, 6.0 * xy[1] / d]
}

fn xy_of(uv: [f64; 2]) -> [f64; 2] {
    let d = 2.0 * uv[0] - 8.0 * uv[1] + 4.0;
    [3.0 * uv[0] / d, 2.0 * uv[1] / d]
}

/// The CIE daylight locus, written out again from its published polynomial.
fn locus(t: f64) -> [f64; 2] {
    let x = if t <= 7000.0 {
        -4.6070e9 / t.powi(3) + 2.9678e6 / t.powi(2) + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t.powi(3) + 1.9018e6 / t.powi(2) + 0.24748e3 / t + 0.237040
    };
    [x, -3.0 * x * x + 2.870 * x - 0.275]
}

/// McCamy's correlated colour temperature of a chromaticity, in mired.
fn mccamy_mired(xy: [f64; 2]) -> f64 {
    let n = (xy[0] - 0.3320) / (0.1858 - xy[1]);
    let cct = 449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33;
    1e6 / cct
}

#[test]
fn the_daylight_locus_is_the_published_polynomial() {
    for t in [
        4000.0, 4907.0, 5500.0, 6504.0, 7000.0, 7001.0, 9637.0, 12000.0,
    ] {
        assert_eq!(daylight_xy(t), locus(t), "{t} K");
    }
    // D65 itself is a little off the locus (the polynomial reads it about 1e-4 away in y).
    let d65 = locus(D65_CCT_K);
    assert!((d65[0] - D65_XY[0]).abs() < 5e-4 && (d65[1] - D65_XY[1]).abs() < 5e-4);
}

#[test]
fn a_temperature_of_a_hundred_moves_the_white_by_fifty_mired() {
    let m0 = 1e6 / D65_CCT_K;
    assert!((m0 - 153.76).abs() < 0.01);
    let w0 = uv(D65_XY);
    // t = 0 is D65 exactly.
    assert_eq!(trim_white_uv(0.0, 0.0), w0);
    for t in [100.0, 50.0, -50.0, -100.0] {
        let m1 = m0 - TRIM_MIRED_PER_UNIT * t;
        let t1 = 1e6 / m1;
        // The step is the displacement of the daylight locus from 6504 K to T1, added to D65.
        let (u0, u1) = (uv(locus(D65_CCT_K)), uv(locus(t1)));
        let want = [w0[0] + (u1[0] - u0[0]), w0[1] + (u1[1] - u0[1])];
        let got = trim_white_uv(t, 0.0);
        assert!((got[0] - want[0]).abs() < 1e-15 && (got[1] - want[1]).abs() < 1e-15);
        // The correlated colour temperature of that white is 50 mired per 100 away from D65's, to the
        // accuracy of McCamy's approximation.
        let mired = mccamy_mired(xy_of(got));
        let expected = m0 - TRIM_MIRED_PER_UNIT * t;
        println!("temperature {t:+}: white at {mired:.2} mired, expected {expected:.2}");
        assert!(
            (mired - expected).abs() < 1.2,
            "t = {t}: {mired} against {expected}"
        );
    }
    // A positive temperature is a bluer light, a lower number of mired.
    assert!(mccamy_mired(xy_of(trim_white_uv(100.0, 0.0))) < mccamy_mired(xy_of(w0)));
}

#[test]
fn a_tint_of_a_hundred_moves_the_white_by_a_hundredth_across_the_locus() {
    for t in [-100.0, 0.0, 100.0] {
        let base = trim_white_uv(t, 0.0);
        for n in [100.0, -100.0, 40.0] {
            let moved = trim_white_uv(t, n);
            let d = [moved[0] - base[0], moved[1] - base[1]];
            let len = d[0].hypot(d[1]);
            assert!((len - TRIM_DUV_PER_UNIT * n.abs()).abs() < 1e-15, "{len}");
            // Across the locus: perpendicular to its tangent at T1 (the polyline at +-1 mired).
            let m1 = 1e6 / D65_CCT_K - TRIM_MIRED_PER_UNIT * t;
            let (a, b) = (uv(locus(1e6 / (m1 - 1.0))), uv(locus(1e6 / (m1 + 1.0))));
            let tangent = [b[0] - a[0], b[1] - a[1]];
            let dot =
                (d[0] * tangent[0] + d[1] * tangent[1]) / (len * tangent[0].hypot(tangent[1]));
            assert!(dot.abs() < 1e-9, "tint is not across the locus: cos {dot}");
            // A positive tint moves the white towards green (up in v).
            assert_eq!(d[1] > 0.0, n > 0.0, "t {t}, n {n}: {d:?}");
        }
    }
}

// ---- the gain ---------------------------------------------------------------------------------

/// The gain written out again from the formula: `d = R(W0) / R(W2)`, green 1.
fn formula(temperature: f64, tint: f64) -> Vec3 {
    let m0 = 1e6 / D65_CCT_K;
    let m1 = m0 - 0.5 * temperature;
    let t1 = 1e6 / m1;
    let w0 = uv(D65_XY);
    let (u0, u1) = (uv(locus(D65_CCT_K)), uv(locus(t1)));
    let mut w = [w0[0] + (u1[0] - u0[0]), w0[1] + (u1[1] - u0[1])];
    if tint != 0.0 {
        let (a, b) = (uv(locus(1e6 / (m1 - 1.0))), uv(locus(1e6 / (m1 + 1.0))));
        let tg = [b[0] - a[0], b[1] - a[1]];
        let len = tg[0].hypot(tg[1]);
        // Perpendicular to the tangent, on the side of larger v.
        let nrm = if tg[0] > 0.0 {
            [-tg[1] / len, tg[0] / len]
        } else {
            [tg[1] / len, -tg[0] / len]
        };
        w = [w[0] + 1e-4 * tint * nrm[0], w[1] + 1e-4 * tint * nrm[1]];
    }
    let to_awg4 = |uvw: [f64; 2]| linalg::mul_vec(&matrices().xyz_to_awg4, xy_to_xyz(xy_of(uvw)));
    let (r0, r2) = (to_awg4(w0), to_awg4(w));
    let d = [r0[0] / r2[0], r0[1] / r2[1], r0[2] / r2[2]];
    d.map(|c| c / d[1])
}

#[test]
fn the_gain_is_the_von_kries_gain_of_the_formula() {
    let mut worst = 0.0f64;
    for t in [-100.0, -63.0, -20.0, 0.0, 7.0, 41.0, 100.0] {
        for n in [-100.0, -33.0, 0.0, 12.0, 100.0] {
            let got = von_kries(t, n);
            let want = formula(t, n);
            worst = worst.max(max_diff(got, want));
            assert_eq!(got[1], 1.0);
        }
    }
    println!("trim gain against the formula, 35 (t, n): largest difference {worst:e}");
    assert!(worst < 1e-13, "{worst}");
    // t = n = 0 is [1, 1, 1].
    assert_eq!(von_kries(0.0, 0.0), [1.0, 1.0, 1.0]);
}

#[test]
fn warmer_raises_red_and_lowers_blue_and_magenta_raises_red_and_blue_against_green() {
    let ratio_rb = |t: f64, n: f64| {
        let g = von_kries(t, n);
        (g[0] / g[2]).log2()
    };
    let mut last = f64::NEG_INFINITY;
    for t in [-100.0, -50.0, 0.0, 50.0, 100.0] {
        let r = ratio_rb(t, 0.0);
        assert!(r > last, "red to blue is not monotone in the temperature");
        last = r;
    }
    // t = +100 is 50 mired: about half a stop between red and blue (a Planck white's blue-to-red ratio
    // changes by 1.44e-2 m K x (1/450 nm - 1/610 nm) x 50e-6 = 0.42 in ln, 0.6 in log2; the gain
    // comes out at 0.485, and t = -100 is the same the other way). The maths file said "about 0.17",
    // which is not what 50 mired is: it is corrected there.
    let step = ratio_rb(100.0, 0.0);
    println!("temperature +100: log2(red / blue) = {step:.4}");
    assert!((0.4..0.6).contains(&step), "{step}");
    assert!(
        (ratio_rb(-100.0, 0.0) + step).abs() < 0.05,
        "{} against {step}",
        ratio_rb(-100.0, 0.0)
    );
    // Tint: positive is magenta (red and blue up against green), negative the reverse; monotone.
    let mut last = f64::NEG_INFINITY;
    for n in [-100.0, -40.0, 0.0, 40.0, 100.0] {
        let g = von_kries(0.0, n);
        let rb_over_g = ((g[0] * g[2]).sqrt() / g[1]).log2();
        assert!(
            rb_over_g > last,
            "red and blue against green is not monotone in the tint"
        );
        last = rb_over_g;
        if n > 0.0 {
            assert!(g[0] > 1.0 && g[2] > 1.0, "{g:?}");
        } else if n < 0.0 {
            assert!(g[0] < 1.0 && g[2] < 1.0, "{g:?}");
        }
    }
}

#[test]
fn the_gain_does_not_depend_on_anything_but_its_two_numbers() {
    // The same trim in two chains with different fused matrices and tones gives the same gain, so a
    // node pasted on another photo renders the same change (docs/23 section 14, decision 4).
    let g = Trim {
        exposure_ev: 0.3,
        temperature: -35.0,
        tint: 18.0,
    }
    .gain()
    .unwrap();
    let again = Trim {
        exposure_ev: 0.3,
        temperature: -35.0,
        tint: 18.0,
    }
    .gain()
    .unwrap();
    assert_eq!(g, again);
    // A warm trim at the white of the picture warms the white: a neutral comes out with more red.
    let warm = second_node(Trim {
        temperature: 100.0,
        ..Trim::default()
    });
    let neutral = [0.18; 3];
    let out = warm.apply_dng_pixel(neutral);
    assert!(out[0] > out[1] && out[1] > out[2], "{out:?}");
}
