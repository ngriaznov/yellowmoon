//! C7 (`docs/02-colour-pipeline.md`, C7): the highlight range and the low-light toe.
//!
//! The range: the formulas against an independent f64 composition, the ALEXA clip constant, exact
//! identity below the knee, continuity of the slope at the knee and at `u = 1`, `s'` non-decreasing
//! with a slope of at least 1 up to the cap, the Sony clip landing on its target, the cap, `r = 0` the
//! identity below the cap, the fade at low clip levels, channel ratios kept, every output channel
//! non-decreasing in each input channel below the cap (proptest), non-finite and non-positive values,
//! and the display codes `docs/02` quotes. The toe: as before (formula, `r = 0` exact, tiny values).
//!
//! The achromatic gate (owner decision 2026-10-29): the gate itself, a grey pixel (and every pixel up to
//! purity 0.6) getting the old result bit for bit against a copy of the old formula kept in this file, a
//! saturated pixel getting less gain, the ALEXA's clip still holding for every pixel, the output staying
//! monotone along an exposure ramp of one colour, the gain continuous in the pixel, and the kernel's
//! inputs (the gate's edges) finite.

use proptest::prelude::*;
use ym_color::consts::linalg::IDENTITY;
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_color::gamut::LinearAwg4;
use ym_color::tone::{
    ACHROMATIC_GATE_FROM, ACHROMATIC_GATE_TO, ALEXA_CLIP_EI, AchromaticGate, HIGHLIGHT_CAP_STOPS,
    HIGHLIGHT_GREY, HIGHLIGHT_KNEE_DEPTH_STOPS, HIGHLIGHT_KNEE_FLOOR_STOPS,
    HIGHLIGHT_MIN_SPAN_STOPS, HIGHLIGHT_OFF_STOPS, HighlightRange, MIN_GATE_SPAN, PURITY_MAX,
    ToneError, ToneParams, alexa_clip_stops, luma, purity,
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

/// The gate written out again from `docs/02` C7: `1 - smoothstep(0.6, 0.9, p)` with
/// `p = (max - min) / max` of the pixel (no shared code with the crate).
fn reference_gate(rgb: [f64; 3]) -> f64 {
    let max = rgb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = rgb.iter().copied().fold(f64::INFINITY, f64::min);
    let p = if max > 0.0 {
        ((max - min) / max).min(2.0)
    } else {
        0.0
    };
    let t = ((p - 0.6) / (0.9 - 0.6)).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// The documented formula, written out again from `docs/02` C7 (division form of the logarithm, no
/// shared code with the crate). `g` is the achromatic gate of the pixel (1 for a grey one).
fn reference_stops(r: f64, clip_luma: f64, s: f64, g: f64) -> f64 {
    let h_s = (clip_luma / 0.18).log2();
    let h_a = ((1.0_f64 - 256.0 / 65535.0) * 800.0 / 4.0).log2();
    let s_k = f64::max(1.0, h_s - 2.0);
    let w = ((h_s - s_k - 0.25) / 1.75).clamp(0.0, 1.0);
    let e = w * r.clamp(0.0, 1.0) * f64::max(h_a - h_s, 0.0);
    let mut out = s;
    if s > s_k && e > 0.0 {
        let u = (s - s_k) / (h_s - s_k);
        let phi = if u <= 1.0 { u * u } else { 2.0 * u - 1.0 };
        out = s + g * e * phi;
    }
    out.min(h_a)
}

fn reference_apply(r: f64, clip_luma: f64, rgb: [f64; 3]) -> [f64; 3] {
    let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if y.is_nan() || y <= 0.0 {
        return rgb;
    }
    let s = (y / 0.18).log2();
    let gain = (reference_stops(r, clip_luma, s, reference_gate(rgb)) - s).exp2();
    rgb.map(|c| c * gain)
}

/// The stage as it was before the gate, copied from the code of that day (`HighlightRange::stops` and
/// `gain`, `PreparedTone::apply_rgb` with the toe off): what a grey pixel must still get, bit for bit.
fn legacy_apply(r: f64, clip_luma: f64, rgb: [f64; 3]) -> [f64; 3] {
    let range = HighlightRange::new(r, clip_luma);
    let y = luma(rgb);
    let gain = if !y.is_finite() || y <= 0.0 {
        1.0
    } else {
        let s = y.log2() - HIGHLIGHT_GREY.log2();
        let mut out = s;
        if range.expansion_stops > 0.0 && s > range.knee_stops {
            let u = (s - range.knee_stops) / (range.clip_stops - range.knee_stops);
            let phi = if u <= 1.0 { u * u } else { 2.0 * u - 1.0 };
            out = s + range.expansion_stops * phi;
        }
        let t = if out > range.cap_stops {
            range.cap_stops
        } else {
            out
        };
        if t == s { 1.0 } else { (t - s).exp2() }
    };
    if gain == 1.0 {
        rgb
    } else {
        rgb.map(|c| c * gain)
    }
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
    assert_eq!(HIGHLIGHT_KNEE_DEPTH_STOPS, 2.0);
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
                let want = reference_stops(r, c, s, 1.0);
                let got = range.stops(s);
                worst = worst.max((got - want).abs());
                assert!(
                    (got - want).abs() < 1e-12,
                    "r {r} h {h} s {s}: {got} vs {want}"
                );
                // The gated tone against the reference, for a few gate weights.
                for g in [0.0, 0.25, 0.5, 1.0] {
                    let want = reference_stops(r, c, s, g);
                    let got = range.stops_gated(s, g);
                    assert!(
                        (got - want).abs() < 1e-12,
                        "r {r} h {h} s {s} g {g}: {got} vs {want}"
                    );
                }
                // The pixel: gain against the reference (the gate of its purity included), three channels,
                // for a pixel the gate leaves alone, one it half closes and one it closes.
                for rgb in [
                    [clip_at(s) * 1.3, clip_at(s) * 0.9, clip_at(s) * 0.7],
                    [clip_at(s) * 1.5, clip_at(s) * 0.9, clip_at(s) * 0.3],
                    [clip_at(s) * 2.0, clip_at(s) * 0.3, clip_at(s) * 0.1],
                ] {
                    let want = reference_apply(r, c, rgb);
                    let got = p(r, 0.0, c, 0.0).apply_rgb(rgb);
                    for k in 0..3 {
                        assert!(
                            (got[k] - want[k]).abs() <= 1e-12 * want[k].abs(),
                            "r {r} h {h} s {s}: {got:?} vs {want:?}"
                        );
                    }
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
fn the_knee_sits_two_stops_under_the_clip_and_grey_under_it_does_not_move() {
    // The owner's "gentler range" (2026-10-30): `s_k = max(1, h_S - 2)`.
    for h in [3.0, 3.25, 3.58, 3.88, 5.0, 7.0] {
        let range = HighlightRange::new(DEFAULT_RANGE, clip_at(h));
        assert!((range.knee_stops - (h - 2.0)).abs() < 1e-12, "h {h}");
        assert!(range.expansion_stops > 0.0);
    }
    for h in [1.4, 2.0, 2.5, 2.99] {
        assert_eq!(
            HighlightRange::new(DEFAULT_RANGE, clip_at(h)).knee_stops,
            1.0,
            "h {h}"
        );
    }
    // At Auto (3.58) the knee is at +1.58. A grey pixel under it comes out of C7 with the bits it went in
    // with, at the default amount and at every other, and renders as it does with the range off (r = 0);
    // just over it the pixel gains.
    let knee = 3.58 - 2.0;
    for r in [DEFAULT_RANGE, 0.25, 1.0] {
        let t = p(r, 0.0, clip_at(3.58), 0.0);
        assert!((t.range().knee_stops - knee).abs() < 1e-12);
        let mut s = -10.0;
        while s < knee - 1e-9 {
            assert_eq!(bits(t.apply_rgb(grey(s))), bits(grey(s)), "r {r} s {s}");
            // The cine rendering sees the same value, so the display code is the one of the range off (r = 0).
            let shown = |tone: ToneParams| {
                let mut params = ChainParams::new(IDENTITY, tone);
                params.output = OutputTarget::Srgb;
                params.dither = false;
                ColorChain::new(&params).unwrap().apply_pixel(grey(s))[1]
            };
            assert_eq!(
                shown(t).to_bits(),
                shown(p(0.0, 0.0, clip_at(3.58), 0.0)).to_bits(),
                "r {r} s {s}"
            );
            s += 0.0173;
        }
        assert!(
            t.apply_rgb(grey(knee + 0.2))[0] > grey(knee + 0.2)[0],
            "r {r}"
        );
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
                assert!(
                    (left - want).abs() < 1e-4 * want,
                    "clip left slope {left} vs {want}"
                );
                assert!(
                    (right - want).abs() < 1e-4 * want,
                    "clip right slope {right} vs {want}"
                );
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
                    assert!(
                        t - prev >= (next - s) * (1.0 - 1e-9),
                        "slope under 1 at {next}"
                    );
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
                        x += if x < 0.0 {
                            0.0137
                        } else {
                            (x * 0.01).max(1e-4)
                        };
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
            if r == 1.0 && h >= 3.0 {
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
            // Even a pixel near the largest finite number lands on the cap, finite.
            let out = t.apply_rgb([1e300, 1e300, 1e300]);
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
            for rgb in [
                grey(s),
                [clip_at(s) * 1.2, clip_at(s) * 0.8, clip_at(s) * 0.6],
            ] {
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
    // Nothing at or under 1.25 stops, all of it from 3 stops (the knee depth, 2, over the knee's floor, 1),
    // linear in between.
    for h in [0.0, 0.5, 1.0, 1.1, 1.25] {
        assert_eq!(
            HighlightRange::new(1.0, clip_at(h)).expansion_stops,
            0.0,
            "h {h}"
        );
    }
    for h in [3.0, 3.58, 5.0] {
        let e = HighlightRange::new(1.0, clip_at(h)).expansion_stops;
        assert!((e - (span - h)).abs() < 1e-12, "h {h}: {e}");
    }
    for h in [1.4, 1.625, 1.9, 2.0, 2.5, 2.9] {
        let range = HighlightRange::new(1.0, clip_at(h));
        let w = (h - 1.0 - 0.25) / 1.75;
        assert!(
            (range.expansion_stops - w * (span - h)).abs() < 1e-12,
            "h {h}"
        );
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
    // Pixels of purity under 0.9, which the achromatic gate does not close.
    for rgb in [
        [4.0, 3.4, 2.8],
        [12.0, 10.0, 7.5],
        [1.2, 2.5, 1.9],
        [30.0, 28.0, 22.0],
        [3.0, 2.2, 1.8],
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
    let rgb = [6.0, 5.0, 3.5];
    let g = tt.highlight_gain(rgb);
    assert!(g > 1.0);
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
    // docs/02 C7, Display result: a neutral at Auto (the clip 3.58 stops over grey), the cine rendering, no
    // look, sRGB codes, at r = 0, 0.3, 0.6 (the default) and 1, exactly as docs/02 quotes them. The knee is
    // at +1.58 stops, so the +1 stop column is untouched at every amount.
    let stops = [1.0, 1.58, 2.0, 2.58, 3.08, 3.33, 3.58];
    let want: [(f64, [i32; 7]); 4] = [
        (0.0, [131, 152, 166, 184, 198, 205, 210]),
        (0.3, [131, 152, 168, 193, 215, 225, 234]),
        (0.6, [131, 152, 170, 201, 228, 240, 247]),
        (1.0, [131, 152, 172, 211, 242, 250, 254]),
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
        assert_eq!(got, codes, "r = {r}");
    }
}

/// The shipped default of `rolloff.highlights` (`ym_core::settings::ROLLOFF_HIGHLIGHTS_DEFAULT`; the owner,
/// 2026-10-30, "gentler range").
const DEFAULT_RANGE: f64 = 0.6;

/// The display code (0 to 255, not rounded) of the green channel of a pixel `stops` over middle grey in luma,
/// of colour `ratios` (scaled to luma 1), through C7 at amount `r` and the cine rendering, sRGB, no look.
fn display_code(r: f64, clip_stops: f64, ratios: [f64; 3], stops: f64) -> f64 {
    let mut params = ChainParams::new(IDENTITY, p(r, 0.0, clip_at(clip_stops), 0.0));
    params.output = OutputTarget::Srgb;
    params.dither = false;
    let chain = ColorChain::new(&params).unwrap();
    let k = clip_at(stops) / luma(ratios);
    chain.apply_pixel(ratios.map(|c| c * k))[1] * 255.0
}

/// Peak display slope (codes per stop) and the largest second difference (codes per 0.02 stop squared) of the
/// green channel along an exposure ramp of one colour, from +1.5 to +6 stops in luma in steps of 0.02.
fn ramp_slope(r: f64, clip_stops: f64, ratios: [f64; 3]) -> (f64, f64) {
    const STEP: f64 = 0.02;
    let g: Vec<f64> = (0..)
        .map(|i| 1.5 + STEP * f64::from(i))
        .take_while(|&s| s <= 6.0)
        .map(|s| display_code(r, clip_stops, ratios, s))
        .collect();
    let peak = g
        .windows(2)
        .map(|w| (w[1] - w[0]) / STEP)
        .fold(f64::NEG_INFINITY, f64::max);
    let kink = g
        .windows(3)
        .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
        .fold(0.0, f64::max);
    (peak, kink)
}

#[test]
fn bright_skin_keeps_its_texture_at_the_default_range() {
    // The banding report of 2026-10-30 (the owner's "gentler range"): with the knee one stop under the camera's
    // clip and the full range, the last stop before the clip was stretched about 3.5 times, so a skin texture
    // turned into a hard edge and then a flat white. On the report's photo (the clip 3.88 stops over grey) the
    // display slope of the neutral axis peaked at 88 codes per stop and that of the forehead colour at 91; the
    // knee two stops under the clip and the amount 0.6 bring them to 50 and 52. At the clip of the local photos
    // (3.58) the old setting peaked at 100 and 104, the new one at 56 and 58. The old knee with the new amount
    // is still over (69 to 81), so the bound is 60 codes per stop: under it with both changes, over it with the
    // old knee or the old amount.
    const PEAK_BOUND_CODES_PER_STOP: f64 = 60.0;
    let report_clip = (2.657_371_628_193_014_4_f64 / 0.18).log2();
    // The forehead of the report, camera colour ratios scaled to luma 1 (purity 0.30), and a neutral.
    let skin = [1.216, 0.951, 0.846];
    for clip in [report_clip, 3.58] {
        for (name, ratios) in [("neutral", [1.0; 3]), ("skin", skin)] {
            let (peak, kink) = ramp_slope(DEFAULT_RANGE, clip, ratios);
            println!(
                "clip {clip:.2}, {name}: peak slope {peak:.1} codes per stop, largest second difference {kink:.3}"
            );
            assert!(
                peak < PEAK_BOUND_CODES_PER_STOP,
                "clip {clip:.2}, {name}: the display slope peaks at {peak:.1} codes per stop"
            );
            assert!(
                kink < 0.2,
                "clip {clip:.2}, {name}: a kink of {kink:.3} codes"
            );
        }
        // A clipped neutral lands under pure white: about 247 to 248 of 255, not 253.5.
        let clipped = display_code(DEFAULT_RANGE, clip, [1.0; 3], clip);
        println!("clip {clip:.2}: the clipped neutral lands at {clipped:.1}");
        assert!(
            (244.0..250.0).contains(&clipped),
            "clip {clip:.2}: the clip lands at {clipped}"
        );
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

// ---- The achromatic gate (owner decision 2026-10-29) ------------------------------------------------

fn bits(v: [f64; 3]) -> [u64; 3] {
    v.map(f64::to_bits)
}

/// A colour of purity `p` (its largest channel 1, its smallest `1 - p`), scaled so its luma is `stops` over
/// middle grey. Purity is exactly `p` for `p` in 0..=1 and above 1 for a negative smallest channel.
fn coloured(p: f64, stops: f64) -> [f64; 3] {
    let base = [1.0, 1.0 - 0.5 * p, 1.0 - p];
    let k = clip_at(stops) / luma(base);
    base.map(|c| c * k)
}

#[test]
fn the_gate_is_a_smoothstep_of_the_purity() {
    assert_eq!((ACHROMATIC_GATE_FROM, ACHROMATIC_GATE_TO), (0.6, 0.9));
    let gate = AchromaticGate::DEFAULT;
    assert_eq!(AchromaticGate::default(), gate);
    assert_eq!((gate.from(), gate.to()), (0.6, 0.9));
    assert!((gate.inv_span() - 1.0 / 0.3).abs() < 1e-12);

    // Purity: 0 for a grey pixel, 1 when the smallest channel is 0, above 1 for a negative channel,
    // held to PURITY_MAX, and scale-invariant.
    assert_eq!(purity([0.18; 3]), 0.0);
    assert_eq!(purity([5.0; 3]), 0.0);
    assert_eq!(purity([1.0, 0.0, 0.0]), 1.0);
    assert_eq!(purity([2.0, 1.0, 1.0]), 0.5);
    assert!((purity([1.0, -0.1, 0.5]) - 1.1).abs() < 1e-15);
    assert_eq!(purity([1.0, -3.0, 0.5]), PURITY_MAX);
    assert_eq!(PURITY_MAX, 2.0);
    for rgb in [[0.8, 0.07, 0.04], [1.0, 0.8, 0.6], [3.0, 2.9, -0.2]] {
        for k in [1e-3, 0.37, 1.0, 5.0, 1e3] {
            assert!(
                (purity(rgb.map(|c| c * k)) - purity(rgb)).abs() < 1e-14,
                "{rgb:?} x {k}"
            );
        }
    }
    // Where the largest channel is not above 0 (or not a number) the purity is 0: the luma is not above 0
    // either, and the stages that use the gate are the identity there.
    for rgb in [
        [0.0; 3],
        [-1.0, -2.0, -3.0],
        [f64::NAN, -1.0, -1.0],
        [-0.0, -0.0, -0.0],
    ] {
        assert_eq!(purity(rgb), 0.0, "{rgb:?}");
    }
    // Every input a stage can see gives a finite purity.
    for rgb in [
        [f64::MAX, -f64::MAX, 0.0],
        [1e-300, -1e300, 0.0],
        [f64::MIN_POSITIVE, -1.0, 0.0],
        [1e30, 1e-30, 0.0],
    ] {
        assert!(purity(rgb).is_finite() && purity(rgb) >= 0.0, "{rgb:?}");
    }

    // The weight: 1 up to the first edge (bit for bit), 0 from the second, the smoothstep between.
    assert_eq!(gate.weight_of_purity(0.0).to_bits(), 1.0_f64.to_bits());
    assert_eq!(gate.weight_of_purity(0.6).to_bits(), 1.0_f64.to_bits());
    assert_eq!(gate.weight_of_purity(0.9), 0.0);
    assert_eq!(gate.weight_of_purity(PURITY_MAX), 0.0);
    assert!((gate.weight_of_purity(0.75) - 0.5).abs() < 1e-15);
    let mut prev = 1.0;
    let mut q = 0.0;
    while q <= PURITY_MAX {
        let w = gate.weight_of_purity(q);
        assert!(
            (0.0..=1.0).contains(&w) && w <= prev,
            "p {q}: {w} after {prev}"
        );
        let t = ((q - 0.6) / 0.3_f64).clamp(0.0, 1.0);
        assert!((w - (1.0 - t * t * (3.0 - 2.0 * t))).abs() < 1e-15);
        prev = w;
        q += 0.001;
    }
    // The pixel's weight is the weight of its purity, and 1.0 exactly for grey.
    for rgb in [
        [0.8, 0.07, 0.04],
        [1.0, 0.8, 0.6],
        [1.0, 0.5, 0.3],
        [1.0, 0.9, 0.2],
        [2.0, 1.0, -0.3],
    ] {
        assert_eq!(gate.weight(rgb), gate.weight_of_purity(purity(rgb)));
        assert!(
            (gate.weight(rgb) - reference_gate(rgb)).abs() < 1e-14,
            "{rgb:?}"
        );
    }
    for y in [1e-9, 0.18, 1.0, 36.0, 1e6] {
        assert_eq!(gate.weight([y; 3]).to_bits(), 1.0_f64.to_bits());
    }

    // OFF is 1 for every pixel; the constructor refuses nonsense.
    assert!(AchromaticGate::OFF.is_off() && !gate.is_off());
    let mut q = 0.0;
    while q <= PURITY_MAX {
        assert_eq!(
            AchromaticGate::OFF.weight_of_purity(q).to_bits(),
            1.0_f64.to_bits()
        );
        q += 0.01;
    }
    assert_eq!(AchromaticGate::new(0.6, 0.9), Ok(gate));
    assert!(AchromaticGate::new(0.0, 0.5).is_ok());
    for (a, b) in [
        (0.5, 0.5),
        (0.9, 0.6),
        (-0.1, 0.5),
        (0.1, f64::NAN),
        (f64::NAN, 0.5),
        (0.1, f64::INFINITY),
        (0.1, 3.5),
    ] {
        assert!(
            matches!(
                AchromaticGate::new(a, b),
                Err(ToneError::InvalidGate { .. })
            ),
            "{a} {b}"
        );
    }
    // The tone parameters carry it.
    assert_eq!(p(1.0, 0.0, 2.0, 0.0).gate, gate);
    assert_eq!(ToneParams::IDENTITY.gate, gate);
    assert_eq!(
        p(1.0, 0.0, 2.0, 0.0).with_gate(AchromaticGate::OFF).gate,
        AchromaticGate::OFF
    );
}

#[test]
fn a_gate_with_a_tiny_span_is_refused_so_the_kernels_inverse_span_stays_finite() {
    // `inv_span` is what the GPU kernel multiplies by: a span near 1e-39 overflows f32 to infinity, and the
    // purity at the `from` edge would then give 0 * infinity = NaN. The span must be at least MIN_GATE_SPAN.
    assert_eq!(MIN_GATE_SPAN, 1e-6);
    for (a, b) in [
        (0.5, 0.5 + 5e-7),
        (0.5, 0.5 + 1e-9),
        (0.9, 0.9 + 1e-12),
        (0.0, 1e-30),
        (0.5, f64::from_bits(0.5_f64.to_bits() + 1)),
    ] {
        assert!(
            matches!(
                AchromaticGate::new(a, b),
                Err(ToneError::InvalidGate { .. })
            ),
            "{a} {b} (span {:e})",
            b - a
        );
    }
    // From the smallest span that is accepted on, the inverse span is finite in f32, and so is the weight at
    // the `from` edge worked out the way the kernel does it.
    for (a, b) in [(0.5, 0.5 + 2e-6), (0.0, 1e-6), (0.6, 0.9), (0.0, 3.0)] {
        let gate = AchromaticGate::new(a, b).expect("accepted");
        let inv = gate.inv_span() as f32;
        assert!(inv.is_finite() && inv > 0.0, "{a} {b}: {inv}");
        let at_from = (a as f32 - a as f32) * inv;
        assert!(at_from.is_finite(), "{a} {b}");
    }
}

#[test]
fn a_grey_pixel_gets_the_old_result_bit_for_bit_at_every_exposure() {
    // The grey ramp from -8 to +10 stops, every highlight amount and clip level, against a copy of the
    // stage as it was before the gate (`legacy_apply`); and the same for a pixel that is grey to the last
    // digit of the fused matrix, and for one of purity 0.5 (the gate is 1 up to 0.6).
    let mut checked = 0_u64;
    for r in [0.0, 0.3, 0.5, 1.0] {
        for h in CLIP_STOPS {
            let c = clip_at(h);
            let t = p(r, 0.0, c, 0.0);
            let mut s = -8.0;
            while s <= 10.0 {
                let y = clip_at(s);
                for rgb in [
                    grey(s),
                    [y, y * (1.0 + 1e-15), y * (1.0 - 1e-15)],
                    [y * 1.2, y, y * 0.6],
                ] {
                    assert_eq!(
                        bits(t.apply_rgb(rgb)),
                        bits(legacy_apply(r, c, rgb)),
                        "r {r} h {h} s {s} {rgb:?}"
                    );
                    checked += 1;
                }
                s += 0.0137;
            }
        }
    }
    println!("C7 gate: {checked} grey and near-grey pixels equal the old stage bit for bit");
}

#[test]
fn with_the_gate_off_the_stage_is_the_old_one_for_every_colour() {
    let mut checked = 0_u64;
    for r in [0.5, 1.0] {
        for h in [2.0, 3.58, 5.0] {
            let c = clip_at(h);
            let t = p(r, 0.0, c, 0.0).with_gate(AchromaticGate::OFF);
            let mut s = -4.0;
            while s <= 12.0 {
                for q in [0.0, 0.2, 0.5, 0.8, 0.95, 1.0, 1.3] {
                    let rgb = coloured(q, s);
                    assert_eq!(
                        bits(t.apply_rgb(rgb)),
                        bits(legacy_apply(r, c, rgb)),
                        "q {q} s {s}"
                    );
                    checked += 1;
                }
                s += 0.0331;
            }
        }
    }
    println!("C7 gate off: {checked} coloured pixels equal the old stage bit for bit");
}

#[test]
fn a_saturated_red_gets_less_gain_than_before_and_a_neutral_the_same() {
    // Auto: the Sony clip 3.58 stops over grey, the knee 2.58; both pixels at +3 stops over grey.
    let t = p(1.0, 0.0, clip_at(3.58), 0.0);
    let before = t.with_gate(AchromaticGate::OFF);
    let neutral = grey(3.0);
    let red = {
        let k = clip_at(3.0) / luma([0.80, 0.07, 0.04]);
        [0.80 * k, 0.07 * k, 0.04 * k]
    };
    assert!(purity(red) > 0.9);
    let gain_before = before.highlight_gain(neutral);
    assert!(
        gain_before > 1.2,
        "the old gain of a neutral at +3 stops: {gain_before}"
    );
    assert_eq!(t.highlight_gain(neutral).to_bits(), gain_before.to_bits());
    assert!(before.highlight_gain(red) > 1.2);
    // A purity of 0.9 or more is out of the gate: the red keeps its scene exposure, exactly.
    assert_eq!(t.highlight_gain(red), 1.0);
    assert_eq!(bits(t.apply_rgb(red)), bits(red));
    // In between: a gain between the two, falling with the purity.
    let mut prev = f64::INFINITY;
    for q in [0.0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] {
        let g = t.highlight_gain(coloured(q, 3.0));
        assert!(
            g <= prev && g >= 1.0 && g <= gain_before * (1.0 + 1e-12),
            "q {q}: {g}"
        );
        if q > 0.6 && q < 0.9 {
            assert!(g < gain_before && g > 1.0, "q {q}: {g}");
        }
        prev = g;
    }
    println!(
        "C7 at +3 stops, clip 3.58: neutral gain {gain_before:.4}, purity 0.5 {:.4}, saturated red {:.4}",
        t.highlight_gain(coloured(0.5, 3.0)),
        t.highlight_gain(red)
    );
}

#[test]
fn the_gain_falls_with_the_purity_from_plus_one_to_plus_eight_stops() {
    // A sweep of purity 0 to 1.5 (a negative smallest channel above 1) at fixed luma, +1 to +8 stops, for
    // the Auto clip and a high one: the gain never rises with the purity, is the old gain up to 0.6 and 1
    // (nothing but the cap) from 0.9.
    for h in [3.58, 5.0] {
        let c = clip_at(h);
        let t = p(1.0, 0.0, c, 0.0);
        let old = t.with_gate(AchromaticGate::OFF);
        for si in 0..=14 {
            let s = 1.0 + 0.5 * f64::from(si);
            let mut prev = f64::INFINITY;
            let mut q = 0.0;
            while q <= 1.5 {
                let rgb = coloured(q, s);
                let g = t.highlight_gain(rgb);
                assert!(
                    g <= prev * (1.0 + 1e-14),
                    "h {h} s {s} q {q}: {g} after {prev}"
                );
                if q <= 0.6 {
                    assert_eq!(
                        g.to_bits(),
                        old.highlight_gain(rgb).to_bits(),
                        "h {h} s {s} q {q}"
                    );
                }
                if q >= 0.9 - 1e-9 {
                    // Only the ALEXA's clip is left: the gain is 1 below the cap, the cap's hold above.
                    let want = if s > HIGHLIGHT_CAP_STOPS {
                        (HIGHLIGHT_CAP_STOPS - s).exp2()
                    } else {
                        1.0
                    };
                    assert!((g - want).abs() < 1e-12, "h {h} s {s} q {q}: {g} vs {want}");
                }
                prev = g;
                q += 0.0125;
            }
        }
    }
}

#[test]
fn the_alexa_clip_holds_for_every_pixel_whatever_its_purity() {
    // The cap acts after the gate: no output luma is above 0.18 * 2^h_A, however saturated the pixel,
    // with a negative channel too, and a pixel above the clip is held on it with its ratios kept.
    let cap = clip_at(HIGHLIGHT_CAP_STOPS);
    for r in [0.0, 0.5, 1.0] {
        for h in [2.0, 3.58, 6.0, 9.0] {
            let t = p(r, 0.0, clip_at(h), 0.0);
            for si in 0..=70 {
                let s = -2.0 + 0.5 * f64::from(si);
                for q in [0.0, 0.3, 0.5, 0.7, 0.8, 0.95, 1.0, 1.4, 1.9] {
                    let rgb = coloured(q, s);
                    let out = t.apply_rgb(rgb);
                    let y = luma(out);
                    assert!(
                        y <= cap * (1.0 + 1e-12),
                        "r {r} h {h} s {s} q {q}: luma {y}"
                    );
                    if s > HIGHLIGHT_CAP_STOPS + 1e-9 {
                        assert!((y - cap).abs() <= 1e-12 * cap, "held at the cap: {y}");
                        // ratios kept
                        assert!((out[1] / out[0] - rgb[1] / rgb[0]).abs() < 1e-12);
                    }
                }
            }
        }
    }
}

#[test]
fn along_an_exposure_ramp_of_one_colour_the_gate_is_constant_and_the_output_rises() {
    // Purity does not change when a pixel is scaled, so the gate is the same at every exposure and the
    // output only ever rises with it: every channel, the luma, and the stage's stops.
    for r in [0.5, 1.0] {
        for h in [2.0, 3.58, 5.0] {
            let c = clip_at(h);
            let t = p(r, 0.0, c, 0.0);
            for q in [0.0, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.4] {
                let base = coloured(q, 0.0);
                let w0 = t.gate.weight(base);
                let mut prev = [f64::NEG_INFINITY; 3];
                let mut prev_luma = f64::NEG_INFINITY;
                let mut s = -6.0;
                while s <= 14.0 {
                    let k = clip_at(s) / clip_at(0.0);
                    let rgb = base.map(|v| v * k);
                    assert!((t.gate.weight(rgb) - w0).abs() < 1e-12, "q {q} s {s}");
                    let out = t.apply_rgb(rgb);
                    // (A negative channel grows more negative with the exposure, as it must.)
                    for ch in (0..3).filter(|&ch| base[ch] >= 0.0) {
                        assert!(
                            out[ch] >= prev[ch] - 1e-12 * prev[ch].abs().max(1e-30),
                            "r {r} h {h} q {q} s {s} channel {ch}: {} after {}",
                            out[ch],
                            prev[ch]
                        );
                    }
                    assert!(luma(out) >= prev_luma - 1e-12 * prev_luma.abs().max(1e-30));
                    prev = out;
                    prev_luma = luma(out);
                    s += 0.0173;
                }
            }
        }
    }
}

#[test]
fn the_gain_is_continuous_in_the_pixel() {
    // From grey to saturated at a fixed luma, across the knee and the clip: no step anywhere on a fine
    // sweep (the steepest part of the gate is 5 per unit of purity; times the expansion in stops).
    for h in [3.58, 6.0] {
        let t = p(1.0, 0.0, clip_at(h), 0.0);
        for s in [2.0, 2.6, 3.0, 3.58, 4.5, 7.0, 8.0] {
            let mut prev = t.highlight_gain(coloured(0.0, s));
            let mut worst = 0.0f64;
            let mut q = 0.0;
            while q <= 1.6 {
                q += 1e-4;
                let g = t.highlight_gain(coloured(q, s));
                worst = worst.max((g - prev).abs() / prev);
                prev = g;
            }
            assert!(worst < 6e-3, "h {h} s {s}: a relative step of {worst}");
        }
    }
}

#[test]
fn with_the_gate_a_channel_can_lower_the_output_but_never_where_the_gate_is_one() {
    // docs/02 C7: the per-channel monotonicity across colours no longer holds exactly. A channel that
    // raises the purity of a pixel closes the gate, and the gain it loses can be more than the channel
    // gains. Found by a search on a grid; the same search with the gate off, or between pixels the gate
    // leaves alone, finds nothing.
    let t = p(1.0, 0.0, clip_at(3.58), 0.0);
    let off = t.with_gate(AchromaticGate::OFF);
    let mut falls = 0_u32;
    let mut falls_off = 0_u32;
    let mut falls_in_one = 0_u32;
    let mut worst = 0.0f64;
    for si in 0..=30 {
        let y = clip_at(2.0 + 0.1 * f64::from(si));
        for a in 0..=10 {
            for b in 0..=10 {
                let rgb = [
                    y * 1.5,
                    y * (0.5 + 0.05 * f64::from(a)),
                    y * (0.3 + 0.07 * f64::from(b)),
                ];
                let up = [rgb[0] * 1.0005, rgb[1], rgb[2]];
                let (o1, o2) = (t.apply_rgb(rgb), t.apply_rgb(up));
                if o2[0] < o1[0] * (1.0 - 1e-12) && luma(o1) < clip_at(HIGHLIGHT_CAP_STOPS) * 0.99 {
                    falls += 1;
                    worst = worst.max(1.0 - o2[0] / o1[0]);
                    if purity(rgb) <= ACHROMATIC_GATE_FROM && purity(up) <= ACHROMATIC_GATE_FROM {
                        falls_in_one += 1;
                    }
                }
                let (f1, f2) = (off.apply_rgb(rgb), off.apply_rgb(up));
                if f2[0] < f1[0] * (1.0 - 1e-12) && luma(f1) < clip_at(HIGHLIGHT_CAP_STOPS) * 0.99 {
                    falls_off += 1;
                }
            }
        }
    }
    println!(
        "C7 gate: {falls} grid pixels lose output when their red rises (worst {worst:.2e}); gate off {falls_off}"
    );
    assert!(falls > 0, "the documented trade-off exists");
    assert_eq!(falls_off, 0);
    assert_eq!(falls_in_one, 0);
}

proptest! {
    /// Below the cap, raising one channel never lowers any output channel (a channel times a
    /// non-negative, non-decreasing function of a luma with positive weights, then the toe); at the
    /// cap, the raised channel's own output still never falls. This holds for a gate that is 1
    /// (`AchromaticGate::OFF`, and the default gate for a grey pixel); with the default gate a channel
    /// that raises the purity of a pixel can lower its gain by more than the channel rises, see
    /// `with_the_gate_a_channel_can_lower_the_output_but_never_where_the_gate_is_one`.
    #[test]
    fn every_output_channel_is_monotone_in_each_input_below_the_cap(
        r in 0.0f64..40.0, g in 0.0f64..40.0, b in 0.0f64..40.0,
        d in 0.0f64..5.0, ch in 0usize..3,
        hi in 0.0f64..=1.0, lo in 0.0f64..=1.0, sigma in 0.0f64..0.05,
        clip_stops in 0.5f64..9.0,
    ) {
        let t = ToneParams::new(hi, lo, clip_at(clip_stops), sigma).unwrap().with_gate(AchromaticGate::OFF);
        let rgb = [r, g, b];
        let mut rgb2 = rgb;
        rgb2[ch] += d;
        let o1 = t.apply_rgb(rgb);
        let o2 = t.apply_rgb(rgb2);
        prop_assert!(o2[ch] >= o1[ch] - 1e-12 * o1[ch].abs().max(1.0), "{o1:?} -> {o2:?}");
        let cap = clip_at(HIGHLIGHT_CAP_STOPS) * (1.0 - 1e-9);
        let range = t.range();
        if luma(rgb2.map(|c| c * range.gain_for(rgb2, &t.gate))) < cap {
            for k in 0..3 {
                prop_assert!(o2[k] >= o1[k] - 1e-12 * o1[k].abs().max(1.0), "channel {k}: {o1:?} -> {o2:?}");
            }
        }
    }

    /// Between pixels the gate leaves alone (purity up to 0.6 for both) every output channel is
    /// monotone in each input channel below the cap, as before the gate.
    #[test]
    fn where_the_gate_is_one_every_output_channel_is_monotone_in_each_input(
        y in 0.05f64..60.0, a in 0.65f64..1.35, b in 0.65f64..1.35,
        d in 0.0f64..0.02, ch in 0usize..3,
        hi in 0.0f64..=1.0, clip_stops in 0.5f64..9.0,
    ) {
        let t = ToneParams::new(hi, 0.0, clip_at(clip_stops), 0.0).unwrap();
        let rgb = [y, y * a, y * b];
        let mut rgb2 = rgb;
        rgb2[ch] += d * y;
        prop_assert!(purity(rgb) <= ACHROMATIC_GATE_FROM && purity(rgb2) <= ACHROMATIC_GATE_FROM);
        let (o1, o2) = (t.apply_rgb(rgb), t.apply_rgb(rgb2));
        let cap = clip_at(HIGHLIGHT_CAP_STOPS) * (1.0 - 1e-9);
        if luma(o2) < cap {
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
