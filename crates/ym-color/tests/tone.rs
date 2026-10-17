//! C7 (`docs/02-colour-pipeline.md`, C7): the highlight range and the low-light toe.
//!
//! The range: the formulas against an independent f64 composition, the ALEXA clip constant, exact
//! identity below the knee, continuity of the slope at the knee and at `u = 1`, `s'` non-decreasing
//! with a slope of at least 1 up to the cap, the Sony clip landing on its target, the cap, `r = 0` the
//! identity below the cap, the fade at low clip levels, channel ratios kept, every output channel
//! non-decreasing in each input channel below the cap (proptest), non-finite and non-positive values,
//! and the display codes `docs/02` quotes. The toe: as before (formula, `r = 0` exact, tiny values).

use proptest::prelude::*;
use ym_color::consts::linalg::IDENTITY;
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_color::gamut::LinearAwg4;
use ym_color::tone::{
    ALEXA_CLIP_EI, HIGHLIGHT_CAP_STOPS, HIGHLIGHT_GREY, HIGHLIGHT_KNEE_DEPTH_STOPS,
    HIGHLIGHT_KNEE_FLOOR_STOPS, HIGHLIGHT_MIN_SPAN_STOPS, HIGHLIGHT_OFF_STOPS, HighlightRange,
    ToneError, ToneParams, alexa_clip_stops, luma,
};

fn p(h: f64, l: f64, c: f64, sigma: f64) -> ToneParams {
    ToneParams::new(h, l, c, sigma).unwrap()
}

/// The clip luma `stops` over middle grey.
fn clip_at(stops: f64) -> f64 {
    HIGHLIGHT_GREY * stops.exp2()
}

/// A grey pixel `stops` over middle grey.
fn grey(stops: f64) -> [f64; 3] {
    [clip_at(stops); 3]
}

/// The documented formula, written out again from `docs/02` C7 (division form of the logarithm, no
/// shared code with the crate).
fn reference_stops(r: f64, clip_luma: f64, s: f64) -> f64 {
    let h_s = (clip_luma / 0.18).log2();
    let h_a = ((1.0 - 256.0 / 65535.0) * 800.0 / 4.0).log2();
    let s_k = f64::max(1.0, h_s - 1.0);
    let w = ((h_s - s_k - 0.25) / 0.75).clamp(0.0, 1.0);
    let e = w * r.clamp(0.0, 1.0) * f64::max(h_a - h_s, 0.0);
    let mut out = s;
    if s > s_k && e > 0.0 {
        let u = (s - s_k) / (h_s - s_k);
        let phi = if u <= 1.0 { u * u } else { 2.0 * u - 1.0 };
        out = s + e * phi;
    }
    out.min(h_a)
}

fn reference_apply(r: f64, clip_luma: f64, rgb: [f64; 3]) -> [f64; 3] {
    let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if !(y > 0.0) {
        return rgb;
    }
    let s = (y / 0.18).log2();
    let gain = (reference_stops(r, clip_luma, s) - s).exp2();
    rgb.map(|c| c * gain)
}

/// Clip levels over the whole useful span: inside the fade, at its ends, the Auto clip, near and over
/// the cap.
const CLIP_STOPS: [f64; 16] = [
    0.25, 0.9, 1.0, 1.2, 1.25, 1.4, 1.75, 2.0, 2.5, 3.0, 3.58, 4.5, 6.0, 7.0, 7.638, 9.0,
];

#[test]
fn the_cap_is_the_alexa_clip_at_ei_800_and_the_constants_are_the_spec() {
    assert!((alexa_clip_stops(ALEXA_CLIP_EI) - HIGHLIGHT_CAP_STOPS).abs() <= 1e-15);
    // Linear 35.86 over black: ARRI's LogC3 EI 800 clip.
    assert!((clip_at(HIGHLIGHT_CAP_STOPS) - 35.86).abs() < 0.005);
    assert_eq!(HIGHLIGHT_GREY, 0.18);
    assert_eq!(HIGHLIGHT_KNEE_DEPTH_STOPS, 1.0);
    assert_eq!(HIGHLIGHT_KNEE_FLOOR_STOPS, 1.0);
    assert_eq!(HIGHLIGHT_MIN_SPAN_STOPS, 0.25);
    assert!(HIGHLIGHT_OFF_STOPS.is_finite() && (HIGHLIGHT_OFF_STOPS as f32).is_finite());
    // Lower EIs clip lower: 180 is about 5.47 stops (the auto exposure's highlight guard level, 8.0).
    assert!((clip_at(alexa_clip_stops(180.0)) - 8.0).abs() < 0.1);
}

#[test]
fn the_range_matches_the_documented_formula() {
    let mut worst = 0.0f64;
    for r in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
        for h in CLIP_STOPS {
            let c = clip_at(h);
            let range = p(r, 0.0, c, 0.0).range();
            assert_eq!(range, HighlightRange::new(r, c));
            assert!((range.clip_stops - h).abs() < 1e-12);
            assert_eq!(range.cap_stops, HIGHLIGHT_CAP_STOPS);
            let mut s = -6.0;
            while s < 14.0 {
                let want = reference_stops(r, c, s);
                let got = range.stops(s);
                worst = worst.max((got - want).abs());
                assert!((got - want).abs() < 1e-12, "r {r} h {h} s {s}: {got} vs {want}");
                // The pixel: gain against the reference, three channels.
                let rgb = [clip_at(s) * 1.3, clip_at(s) * 0.9, clip_at(s) * 0.7];
                let want = reference_apply(r, c, rgb);
                let got = p(r, 0.0, c, 0.0).apply_rgb(rgb);
                for k in 0..3 {
                    assert!(
                        (got[k] - want[k]).abs() <= 1e-12 * want[k].abs(),
                        "r {r} h {h} s {s}: {got:?} vs {want:?}"
                    );
                }
                s += 0.0371;
            }
        }
    }
    println!("C7 range against the reference composition: worst |s' - ref| {worst:.1e} stops");
}

#[test]
fn below_the_knee_the_pixel_is_returned_bit_for_bit() {
    for r in [0.5, 1.0] {
        for h in CLIP_STOPS {
            let t = p(r, 0.0, clip_at(h), 0.0);
            let knee = t.range().knee_stops;
            assert!(knee >= HIGHLIGHT_KNEE_FLOOR_STOPS);
            let mut s = -20.0;
            while s <= knee.min(HIGHLIGHT_CAP_STOPS) {
                for rgb in [
                    grey(s),
                    [clip_at(s) * 1.4, clip_at(s) * 0.95, clip_at(s) * 0.2],
                ] {
                    if luma(rgb).log2() - HIGHLIGHT_GREY.log2() > knee {
                        continue;
                    }
                    let out = t.apply_rgb(rgb);
                    assert_eq!(out.map(f64::to_bits), rgb.map(f64::to_bits), "h {h} s {s}");
                    assert_eq!(t.highlight_gain(rgb), 1.0);
                }
                s += 0.013;
            }
            // Middle grey and +1 stop never move, whatever the clip.
            assert_eq!(t.apply_rgb(grey(0.0)), grey(0.0));
            assert_eq!(t.apply_rgb(grey(1.0)), grey(1.0));
        }
    }
}

#[test]
fn the_slope_is_continuous_at_the_knee_and_at_the_clip() {
    for r in [0.25, 0.5, 1.0] {
        for h in [1.6, 2.0, 3.58, 5.0, 6.5] {
            let range = HighlightRange::new(r, clip_at(h));
            assert!(range.expansion_stops > 0.0, "r {r} h {h}");
            let k = range.knee_stops;
            let eps = 1e-6;
            // At the knee: slope 1 on both sides, and the value leaves the identity at second order.
            let left = (range.stops(k) - range.stops(k - eps)) / eps;
            let right = (range.stops(k + eps) - range.stops(k)) / eps;
            assert!((left - 1.0).abs() < 1e-8, "knee left slope {left}");
            assert!((right - 1.0).abs() < 1e-4, "knee right slope {right}");
            assert!((range.stops(k + 1e-3) - (k + 1e-3)).abs() < 1e-5);
            // At u = 1 (the Sony clip), unless the cap cuts in first: 1 + 2E / span on both sides.
            let c = range.clip_stops;
            if range.stops(c + eps) < HIGHLIGHT_CAP_STOPS {
                let want = 1.0 + 2.0 * range.expansion_stops / (c - k);
                let left = (range.stops(c) - range.stops(c - eps)) / eps;
                let right = (range.stops(c + eps) - range.stops(c)) / eps;
                assert!((left - want).abs() < 1e-4 * want, "clip left slope {left} vs {want}");
                assert!((right - want).abs() < 1e-4 * want, "clip right slope {right} vs {want}");
            }
        }
    }
}

#[test]
fn s_prime_never_falls_and_rises_at_least_as_fast_as_s_up_to_the_cap() {
    // docs/15 M1-09: monotonicity of C7 over the full parameter ranges (amounts x clip levels x noise
    // floors), on the tone itself and on a grey pixel through the whole stage.
    let mut checked = 0_u64;
    for hi in 0..=10 {
        let r = f64::from(hi) / 10.0;
        for h in CLIP_STOPS {
            let range = HighlightRange::new(r, clip_at(h));
            let mut s = -4.0;
            let mut prev = range.stops(s);
            while s < 16.0 {
                let next = s + 0.0079;
                let t = range.stops(next);
                assert!(t >= prev, "r {r} h {h} s {next}: {t} < {prev}");
                if t < HIGHLIGHT_CAP_STOPS {
                    assert!(t - prev >= (next - s) * (1.0 - 1e-9), "slope under 1 at {next}");
                } else {
                    assert_eq!(t, HIGHLIGHT_CAP_STOPS, "flat at the cap");
                }
                prev = t;
                s = next;
                checked += 1;
            }
            for lo in [0.0, 0.5, 1.0] {
                for sigma in [0.0, 1e-4, 1e-3, 1e-2] {
                    let t = p(r, lo, clip_at(h), sigma);
                    let mut x = -2.0;
                    let mut prev = t.apply_rgb([x; 3])[0];
                    while x < 400.0 {
                        x += if x < 0.0 { 0.0137 } else { (x * 0.01).max(1e-4) };
                        let f = t.apply_rgb([x; 3])[0];
                        assert!(
                            f >= prev - 1e-14 * prev.abs().max(1.0),
                            "r {r} h {h} lo {lo} sigma {sigma} x {x}: {f} < {prev}"
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
fn the_sony_clip_lands_on_its_target() {
    for r in [0.0, 0.3, 0.5, 0.8, 1.0] {
        for h in CLIP_STOPS {
            let c = clip_at(h);
            let range = HighlightRange::new(r, c);
            let target = (range.clip_stops + range.expansion_stops).min(HIGHLIGHT_CAP_STOPS);
            assert!((range.target_stops() - target).abs() < 1e-12);
            // f(clip) = target: the clipped neutral comes out at 0.18 * 2^T, to 1e-12 relative.
            let out = p(r, 0.0, c, 0.0).apply_rgb([c; 3]);
            let want = clip_at(target);
            for v in out {
                assert!(
                    (v - want).abs() <= 1e-12 * want,
                    "r {r} h {h}: {v} vs {want}"
                );
            }
            // r = 1 with a full fade puts the clip on the ALEXA clip.
            if r == 1.0 && h >= 2.0 {
                assert!((range.target_stops() - HIGHLIGHT_CAP_STOPS).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn nothing_leaves_the_stage_above_the_alexa_clip() {
    let cap = clip_at(HIGHLIGHT_CAP_STOPS);
    for r in [0.0, 0.5, 1.0] {
        for h in CLIP_STOPS {
            let t = p(r, 0.0, clip_at(h), 0.0);
            for s in [7.0, 7.6, 7.65, 8.0, 10.0, 30.0, 300.0, 1000.0] {
                for rgb in [grey(s), [clip_at(s) * 2.0, clip_at(s), clip_at(s) * 0.1]] {
                    let out = t.apply_rgb(rgb);
                    let y = luma(out);
                    assert!(y <= cap * (1.0 + 1e-12), "r {r} h {h} s {s}: luma {y}");
                    if s > HIGHLIGHT_CAP_STOPS + 1e-9 {
                        assert!((y - cap).abs() <= 1e-12 * cap, "held at the cap: {y}");
                    }
                }
            }
            // Even the largest finite pixel lands on the cap, finite.
            let out = t.apply_rgb([f64::MAX; 3]);
            assert!(out.iter().all(|v| v.is_finite()));
            assert!((luma(out) - cap).abs() <= 1e-9 * cap, "{out:?}");
        }
    }
    // When the Sony clip is at or over the ALEXA's (an Exposure about +4 EV over Auto), the stage is the
    // identity below the cap and the hard clip above it.
    for h in [HIGHLIGHT_CAP_STOPS, 8.5, 22.0] {
        let range = HighlightRange::new(1.0, clip_at(h));
        assert_eq!(range.expansion_stops, 0.0);
        assert_eq!(range.stops(7.0), 7.0);
        assert_eq!(range.stops(9.0), HIGHLIGHT_CAP_STOPS);
    }
}

#[test]
fn r_zero_is_the_identity_below_the_cap() {
    for h in CLIP_STOPS {
        let t = p(0.0, 0.0, clip_at(h), 0.0);
        assert_eq!(t.range().expansion_stops, 0.0);
        let mut s = -10.0;
        while s < HIGHLIGHT_CAP_STOPS - 1e-9 {
            for rgb in [grey(s), [clip_at(s) * 1.2, clip_at(s) * 0.8, clip_at(s) * 0.6]] {
                if luma(rgb) >= clip_at(HIGHLIGHT_CAP_STOPS) {
                    continue;
                }
                assert_eq!(t.apply_rgb(rgb), rgb, "h {h} s {s}");
            }
            s += 0.017;
        }
    }
}

#[test]
fn the_expansion_fades_out_as_the_clip_nears_middle_grey() {
    let span = HIGHLIGHT_CAP_STOPS;
    // Nothing at or under 1.25 stops, all of it from 2 stops, linear in between.
    for h in [0.0, 0.5, 1.0, 1.1, 1.25] {
        assert_eq!(HighlightRange::new(1.0, clip_at(h)).expansion_stops, 0.0, "h {h}");
    }
    for h in [2.0, 2.5, 3.58, 5.0] {
        let e = HighlightRange::new(1.0, clip_at(h)).expansion_stops;
        assert!((e - (span - h)).abs() < 1e-12, "h {h}: {e}");
    }
    for h in [1.4, 1.625, 1.9] {
        let range = HighlightRange::new(1.0, clip_at(h));
        let w = (h - 1.0 - 0.25) / 0.75;
        assert!((range.expansion_stops - w * (span - h)).abs() < 1e-12, "h {h}");
        assert_eq!(range.knee_stops, 1.0, "the knee stays at +1 stop");
    }
    // Continuous in the clip level: no jump anywhere on a fine sweep.
    let mut prev = HighlightRange::new(0.7, clip_at(0.5)).expansion_stops;
    let mut h = 0.5;
    while h < 9.0 {
        h += 1e-4;
        let e = HighlightRange::new(0.7, clip_at(h)).expansion_stops;
        assert!((e - prev).abs() < 1e-3, "jump at h {h}: {prev} -> {e}");
        prev = e;
    }
}

#[test]
fn the_gain_keeps_the_channel_ratios_and_the_toe_follows_it() {
    let t = p(0.8, 0.0, clip_at(3.58), 0.0);
    for rgb in [
        [4.0, 3.0, 0.5],
        [12.0, 9.0, 2.0],
        [0.4, 2.5, 1.0],
        [30.0, 28.0, 2.0],
        [1.0, 0.5, -0.05],
    ] {
        let g = t.highlight_gain(rgb);
        assert!(g > 1.0, "{rgb:?} is above the knee");
        let out = t.apply_rgb(rgb);
        for k in 0..3 {
            assert_eq!(out[k], rgb[k] * g);
        }
    }
    // The toe acts on the gained pixel: its norm after the range.
    let tt = p(0.8, 1.0, clip_at(3.58), 0.5);
    let rgb = [6.0, 4.0, 1.0];
    let g = tt.highlight_gain(rgb);
    let gained = rgb.map(|c| c * g);
    let n = gained[0];
    let tau = tt.toe_tau();
    let out = tt.apply_rgb(rgb);
    for k in 0..3 {
        let want = gained[k] * (n * n / (n * n + tau * tau));
        assert!((out[k] - want).abs() <= 1e-15 * want.abs());
    }
}

#[test]
fn non_finite_and_non_positive_values() {
    let t = p(1.0, 0.5, clip_at(3.58), 0.01);
    // A non-finite channel passes through unchanged.
    for rgb in [
        [f64::NAN, 0.1, 0.1],
        [f64::INFINITY, 0.1, 0.1],
        [5.0, f64::NEG_INFINITY, 5.0],
        [f64::NAN; 3],
    ] {
        let out = t.apply_rgb(rgb);
        assert_eq!(out.map(f64::to_bits), rgb.map(f64::to_bits), "{rgb:?}");
    }
    // The range: luma at or under zero, or not finite, has a gain of 1; NaN stops stay NaN.
    let range = t.range();
    for y in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(range.gain(y), 1.0, "luma {y}");
    }
    assert!(range.stops(f64::NAN).is_nan());
    assert_eq!(range.stops(f64::INFINITY), HIGHLIGHT_CAP_STOPS);
    assert_eq!(range.stops(f64::NEG_INFINITY), f64::NEG_INFINITY);
    // A negative luma next to a bright channel: no range gain, only the toe.
    let toe_only = p(0.0, 0.5, clip_at(3.58), 0.01);
    let rgb = [30.0, -10.0, -10.0];
    assert!(luma(rgb) < 0.0);
    assert_eq!(t.apply_rgb(rgb), toe_only.apply_rgb(rgb));
    // Negative-only pixels shrink towards zero with their sign and ratios.
    let out = t.apply(LinearAwg4([-0.1, -0.02, -0.5])).0;
    for (o, i) in out.iter().zip([-0.1, -0.02, -0.5]) {
        assert!(*o <= 0.0 && o.abs() <= f64::abs(i));
    }
    // n = 0 with a toe maps to exact zero.
    assert_eq!(t.apply_rgb([0.0; 3]), [0.0; 3]);
}

#[test]
fn the_identity_changes_no_bit_of_any_finite_pixel() {
    let t = ToneParams::IDENTITY;
    assert_eq!(t.range(), HighlightRange::NONE);
    assert_eq!(t.clip_level, f64::INFINITY);
    for rgb in [
        [0.0; 3],
        [0.18; 3],
        [-0.3, 1e-300, 5.0],
        [36.0, 40.0, 1e3],
        [1e30, 1e30, 1e30],
        [f64::MAX, f64::MAX, 1.0],
        [f64::MIN_POSITIVE, 0.0, 0.0],
    ] {
        let out = t.apply_rgb(rgb);
        assert_eq!(out.map(f64::to_bits), rgb.map(f64::to_bits), "{rgb:?}");
    }
    let mut s = -1000.0;
    while s < 1100.0 {
        assert_eq!(HighlightRange::NONE.stops(s), s);
        s += 0.37;
    }
    // A huge but finite clip level is not the identity: the ALEXA clip still caps (use IDENTITY to turn
    // the stage off).
    let huge = p(0.5, 0.0, 1.0e6, 0.0);
    assert_eq!(huge.range().expansion_stops, 0.0);
    assert!(luma(huge.apply_rgb([100.0; 3])) < 36.0);
    // A clip level that `new` refuses gives no range when the struct is filled in by hand.
    for c in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(HighlightRange::new(1.0, c), HighlightRange::NONE);
    }
}

#[test]
fn the_display_codes_of_docs_02_are_reproduced() {
    // docs/02 C7, Display result: a neutral at Auto (the clip 3.58 stops over grey), ACES 2 with the
    // placement, no look, sRGB codes, at r = 0, 0.5 and 1.
    let stops = [1.0, 2.0, 2.58, 3.08, 3.33, 3.58];
    let want: [(f64, [i32; 6]); 3] = [
        (0.0, [158, 192, 208, 220, 225, 229]),
        (0.5, [158, 192, 208, 229, 240, 248]),
        (1.0, [158, 192, 208, 236, 248, 254]),
    ];
    for (r, codes) in want {
        let mut params = ChainParams::new(IDENTITY, p(r, 0.0, clip_at(3.58), 0.0));
        params.output = OutputTarget::Srgb;
        params.dither = false;
        let chain = ColorChain::new(&params).unwrap();
        let got: Vec<i32> = stops
            .iter()
            .map(|&s| (chain.apply_pixel(grey(s))[1] * 255.0).round() as i32)
            .collect();
        println!("r = {r}: codes {got:?} (docs/02: {codes:?})");
        for (g, w) in got.iter().zip(codes) {
            assert!((g - w).abs() <= 1, "r = {r}: {got:?} vs {codes:?}");
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
            // Below the knee the stage is the toe alone.
            let out = t.apply_rgb([n, 0.5 * n, 0.25 * n]);
            assert!((out[0] - want).abs() < 1e-15, "r={r} n={n}");
        }
        // About n above 3 tau (within 10%), and strongly reduced below tau.
        assert!(t.toe(3.0 * tau) > 0.89 * 3.0 * tau);
        assert!(t.toe(0.5 * tau) < 0.25 * 0.5 * tau);
    }
}

#[test]
fn more_low_light_never_raises_the_output() {
    for &c in &[1.0, 16.0] {
        let mut n = 0.0005;
        while n < 40.0 * c {
            let mut prev = f64::INFINITY;
            for step in 0..=20 {
                let r = f64::from(step) / 20.0;
                let f = p(0.5, r, c, 0.003).apply_rgb([n; 3])[0];
                assert!(f <= prev + 1e-14, "low light r={r} n={n}");
                prev = f;
            }
            n *= 1.05;
        }
    }
}

#[test]
fn more_highlight_range_never_lowers_the_output() {
    // A larger amount moves the clip further up: the output of every tone rises or stays.
    for h in [1.6, 2.0, 3.58, 6.0] {
        let mut s = -2.0;
        while s < 12.0 {
            let mut prev = f64::NEG_INFINITY;
            for step in 0..=20 {
                let r = f64::from(step) / 20.0;
                let f = p(r, 0.0, clip_at(h), 0.0).apply_rgb(grey(s))[0];
                assert!(f >= prev, "h {h} s {s} r {r}: {f} < {prev}");
                prev = f;
            }
            s += 0.05;
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
        let f = t.toe(n);
        assert!(f.is_finite() && (0.0..=n).contains(&f), "f({n:e}) = {f:e}");
        let out = t.apply(LinearAwg4([n, 0.5 * n, 0.25 * n])).0;
        assert!(out.iter().all(|v| v.is_finite()), "{n:e}: {out:?}");
    }
    assert_eq!(t.toe(1.0), 1.0);
    assert_eq!(t.toe(0.0), 0.0);
    // Half way between: n = tau gives n / 2.
    assert!((t.toe(tau) / tau - 0.5).abs() < 1e-12);
}

#[test]
fn constructor_validates() {
    assert!(matches!(
        ToneParams::new(0.5, 0.5, -1.0, 0.0),
        Err(ToneError::InvalidClipLevel(_))
    ));
    assert!(matches!(
        ToneParams::new(0.5, 0.5, f64::INFINITY, 0.0),
        Err(ToneError::InvalidClipLevel(_))
    ));
    assert!(matches!(
        ToneParams::new(0.5, 0.5, 1.0, f64::INFINITY),
        Err(ToneError::InvalidNoiseFloor(_))
    ));
    let t = ToneParams::new(f64::NAN, 3.0, 2.0, 0.0).unwrap();
    assert_eq!((t.highlights, t.low_light), (0.0, 1.0));
}

proptest! {
    /// Below the cap, raising one channel never lowers any output channel (a channel times a
    /// non-negative, non-decreasing function of a luma with positive weights, then the toe); at the
    /// cap, the raised channel's own output still never falls.
    #[test]
    fn every_output_channel_is_monotone_in_each_input_below_the_cap(
        r in 0.0f64..40.0, g in 0.0f64..40.0, b in 0.0f64..40.0,
        d in 0.0f64..5.0, ch in 0usize..3,
        hi in 0.0f64..=1.0, lo in 0.0f64..=1.0, sigma in 0.0f64..0.05,
        clip_stops in 0.5f64..9.0,
    ) {
        let t = ToneParams::new(hi, lo, clip_at(clip_stops), sigma).unwrap();
        let rgb = [r, g, b];
        let mut rgb2 = rgb;
        rgb2[ch] += d;
        let o1 = t.apply_rgb(rgb);
        let o2 = t.apply_rgb(rgb2);
        prop_assert!(o2[ch] >= o1[ch] - 1e-12 * o1[ch].abs().max(1.0), "{o1:?} -> {o2:?}");
        let cap = clip_at(HIGHLIGHT_CAP_STOPS) * (1.0 - 1e-9);
        let range = t.range();
        if luma(rgb2.map(|c| c * range.gain(luma(rgb2)))) < cap {
            for k in 0..3 {
                prop_assert!(o2[k] >= o1[k] - 1e-12 * o1[k].abs().max(1.0), "channel {k}: {o1:?} -> {o2:?}");
            }
        }
    }

    /// The ratios of the channels are the input's, and the gain is at least 1 below the cap.
    #[test]
    fn the_range_scales_all_three_channels_by_one_number(
        r in 0.01f64..60.0, g in 0.01f64..60.0, b in 0.01f64..60.0,
        hi in 0.0f64..=1.0, clip_stops in 0.5f64..9.0,
    ) {
        let t = ToneParams::new(hi, 0.0, clip_at(clip_stops), 0.0).unwrap();
        let out = t.apply_rgb([r, g, b]);
        let k = out[0] / r;
        prop_assert!((out[1] / g - k).abs() <= 1e-14 * k);
        prop_assert!((out[2] / b - k).abs() <= 1e-14 * k);
        if luma([r, g, b]) < clip_at(HIGHLIGHT_CAP_STOPS) * (1.0 - 1e-9) {
            prop_assert!(k >= 1.0);
        }
    }
}
