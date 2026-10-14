//! The white lift (`docs/02` C12 step 1b, D-02): the scene-linear gain on the tones above +1 stop
//! over middle grey, its estimator on the 1/8 preview and its place in the CPU chain and the DNG.
//!
//! * the stage: exactly 1 up to +1 stop, mid grey and everything below untouched, ratio-preserving,
//!   monotone along every ray and on the neutral axis, `A = 0` the exact identity;
//! * the estimator: deterministic, 0 for a low-key photo, a lift for a high-key one, the offset;
//! * the chain: identity at 0, nothing below the knee, ARRI's own highlights untouched, the DNG's
//!   pixels equal the scene values the app feeds ACES 2.
//!
//! Every pixel and preview here is fixed or seeded; the LUTs are our own synthetic cubes.

use std::sync::Arc;

use proptest::prelude::*;
use ym_color::aces2::{Aces2Output, Aces2Transform};
use ym_color::consts::Vec3;
use ym_color::consts::linalg::{self, Mat3};
use ym_color::display::{
    ArriDisplayLut, ChainParams, ColorChain, DisplayLutInput, DisplayLutOutput, DisplayTransform,
    LookSetting, OutputTarget, aces2_input_matrix,
};
use ym_color::exposure::PreviewRgb;
use ym_color::exposure::white::drive_luma;
use ym_color::exposure::white::{
    KNEE_EV, MAX_AMOUNT_EV, MAX_AUTO_AMOUNT_EV, TARGET_CODE, WhiteEstimate, WhiteLift,
    estimate_white,
};
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

const GREY: f64 = 0.18;

fn stops(s: f64) -> f64 {
    GREY * s.exp2()
}

fn identity_matrix() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn tone() -> ToneParams {
    ToneParams::new(0.5, 0.0, 2.16, 0.0).unwrap()
}

/// A chain with the identity matrix, so the camera triple is the AWG4 triple, with `lift`.
fn chain_with(lift: WhiteLift) -> ColorChain {
    let mut p = ChainParams::new(identity_matrix(), tone());
    p.output = OutputTarget::Srgb;
    p.white_lift = lift;
    ColorChain::new(&p).unwrap()
}

/// SplitMix64: a tiny deterministic generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Seeded pixels: neutrals and colours over -6 to +5 stops (luminance).
fn pixels(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let y = stops(-6.0 + 11.0 * rng.unit());
            [
                y * (0.4 + 1.2 * rng.unit()),
                y * (0.4 + 1.2 * rng.unit()),
                y * (0.4 + 1.2 * rng.unit()),
            ]
        })
        .collect()
}

fn lift(amount: f64, anchor: f64) -> WhiteLift {
    WhiteLift::new(amount, anchor).unwrap()
}

// ---- the stage ---------------------------------------------------------------------------------

#[test]
fn an_amount_of_zero_is_the_exact_identity_and_equals_none() {
    let none = WhiteLift::NONE;
    assert!(none.is_identity());
    assert_eq!(WhiteLift::new(0.0, 3.7).unwrap(), none);
    assert_eq!(WhiteLift::default(), none);
    for px in pixels(200, 1) {
        assert_eq!(none.apply(px), px);
        assert_eq!(none.gain(px), 1.0);
    }
    // and in the chain, bit for bit
    let plain = chain_with(WhiteLift::NONE);
    let zero = chain_with(lift(0.0, 3.0));
    for px in pixels(200, 2) {
        assert_eq!(plain.apply_pixel(px), zero.apply_pixel(px));
        assert_eq!(plain.apply_dng_pixel(px), zero.apply_dng_pixel(px));
    }
}

#[test]
fn bad_values_are_refused() {
    assert!(WhiteLift::new(-0.1, 2.0).is_err());
    assert!(WhiteLift::new(MAX_AMOUNT_EV + 0.01, 2.0).is_err());
    assert!(WhiteLift::new(f64::NAN, 2.0).is_err());
    assert!(WhiteLift::new(1.0, f64::INFINITY).is_err());
    assert!(WhiteLift::new(MAX_AMOUNT_EV, 2.0).is_ok());
}

#[test]
fn below_plus_one_stop_the_gain_is_exactly_one_and_mid_grey_is_untouched() {
    for (amount, anchor) in [(0.3, 1.5), (1.0, 2.5), (2.0, 3.6), (3.0, 1.1), (0.8, 5.0)] {
        let l = lift(amount, anchor);
        assert_eq!(l.gain_for_luminance(GREY), 1.0, "mid grey");
        let mut s = -12.0;
        while s <= KNEE_EV {
            let y = stops(s);
            assert_eq!(
                l.gain_for_luminance(y),
                1.0,
                "{s} stops, A {amount}, anchor {anchor}"
            );
            let px = [y, y * 0.8, y * 1.1];
            // to 1e-9 in scene-linear (it is in fact exact)
            for c in 0..3 {
                assert!((l.apply(px)[c] - px[c]).abs() <= 1e-9 * px[c].max(1e-30));
            }
            s += 0.125;
        }
        // non-positive and non-finite luminance
        for y in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY] {
            assert_eq!(l.gain_for_luminance(y), 1.0);
        }
    }
}

#[test]
fn the_gain_is_two_to_the_amount_above_the_anchor_and_smooth_between() {
    let l = lift(1.2, 2.8);
    let top = l.gain_for_luminance(stops(2.8));
    assert!((top - 1.2_f64.exp2()).abs() < 1e-12, "{top}");
    for s in [2.8, 3.5, 6.0, 20.0] {
        assert!((l.gain_for_luminance(stops(s)) - 1.2_f64.exp2()).abs() < 1e-12);
    }
    // the midpoint of the ramp is half the amount (smoothstep(0.5) = 0.5)
    let mid = l.gain_for_luminance(stops(KNEE_EV + 0.5 * l.span_ev()));
    assert!((mid - 0.6_f64.exp2()).abs() < 1e-12, "{mid}");
    // the gain never decreases with luminance and has no step
    let mut prev = 1.0;
    let mut s = KNEE_EV - 0.5;
    while s < 4.0 {
        let g = l.gain_for_luminance(stops(s));
        assert!(g >= prev - 1e-15, "{s}: {g} < {prev}");
        assert!(g - prev < 0.02, "a step of {} at {s} stops", g - prev);
        prev = g;
        s += 0.01;
    }
    // a narrow ramp is still at least MIN_SPAN wide
    assert!((lift(1.0, 1.0).span_ev() - 0.25).abs() < 1e-15);
    assert!((lift(1.0, 0.0).span_ev() - 0.25).abs() < 1e-15);
}

#[test]
fn colour_ratios_are_kept_exactly() {
    let l = lift(1.4, 2.6);
    for px in pixels(500, 3) {
        let out = l.apply(px);
        let g = l.gain(px);
        for c in 0..3 {
            assert!((out[c] - px[c] * g).abs() <= 1e-15 * out[c].abs().max(1e-300));
        }
        // ratios between channels
        if px[1] > 0.0 && out[1] > 0.0 {
            assert!((out[0] / out[1] - px[0] / px[1]).abs() <= 1e-12 * (px[0] / px[1]).abs());
            assert!((out[2] / out[1] - px[2] / px[1]).abs() <= 1e-12 * (px[2] / px[1]).abs());
        }
    }
}

#[test]
fn the_neutral_axis_and_every_ray_are_monotone() {
    for (amount, anchor) in [(0.5, 1.6), (1.15, 2.78), (2.0, 4.0), (3.0, 1.0)] {
        let l = lift(amount, anchor);
        // the neutral axis: every channel and the luminance only ever rise
        let mut prev = [0.0; 3];
        let mut s = -10.0;
        while s < 8.0 {
            let y = stops(s);
            let out = l.apply([y, y, y]);
            for c in 0..3 {
                assert!(out[c] >= prev[c], "{s} stops: {out:?} below {prev:?}");
            }
            prev = out;
            s += 0.01;
        }
        // along a ray (the colour kept, the level scaled), every channel rises with the level
        for base in [
            [1.0, 0.6, 0.3],
            [0.2, 1.0, 0.5],
            [0.3, 0.4, 1.0],
            [1.0, 1.0, 0.05],
        ] {
            let mut prev = [0.0; 3];
            let mut s = -8.0;
            while s < 6.0 {
                let k = stops(s);
                let out = l.apply([base[0] * k, base[1] * k, base[2] * k]);
                for c in 0..3 {
                    assert!(out[c] >= prev[c] - 1e-15, "ray {base:?} at {s}");
                }
                prev = out;
                s += 0.01;
            }
        }
    }
}

#[test]
fn a_channel_alone_never_lowers_its_own_output() {
    // The gain follows a luma with positive weights, so a channel raised with the other two held can
    // only raise its own output: strictly per-channel monotone, for every colour, not only inside the
    // gamut of photographs (the AWG4 Y row, with its negative blue weight, would fail this for
    // saturated blues: a drive by `Y` gave a slope of -0.31 at [0.46, 0.49, 2.67]).
    for (amount, anchor) in [(0.5, 1.6), (1.15, 2.78), (2.0, 4.0), (2.0, 1.5), (3.0, 1.0)] {
        let l = lift(amount, anchor);
        let mut rng = Rng(77);
        for _ in 0..20_000 {
            let px = [4.0 * rng.unit(), 4.0 * rng.unit(), 4.0 * rng.unit()];
            for c in 0..3 {
                let h = 1e-6 * px[c].max(1e-3);
                let mut up = px;
                up[c] += h;
                let d = (l.apply(up)[c] - l.apply(px)[c]) / h;
                assert!(
                    d >= -1e-9,
                    "{px:?} channel {c}: slope {d} (A {amount}, anchor {anchor})"
                );
            }
        }
    }
}

proptest! {
    #[test]
    fn the_gain_is_at_least_one_and_at_most_two_to_the_amount(
        amount in 0.0f64..3.0, anchor in -1.0f64..6.0, s in -20.0f64..20.0,
    ) {
        let l = WhiteLift::new(amount, anchor).unwrap();
        let g = l.gain_for_luminance(stops(s));
        prop_assert!(g >= 1.0);
        prop_assert!(g <= amount.exp2() * (1.0 + 1e-12));
        if s <= KNEE_EV { prop_assert_eq!(g, 1.0); }
    }
}

// ---- the chain ---------------------------------------------------------------------------------

#[test]
fn the_chain_changes_nothing_below_the_knee_and_lifts_above_it() {
    let plain = chain_with(WhiteLift::NONE);
    let lifted = chain_with(lift(1.0, 2.8));
    for s in [-8.0, -3.0, -1.0, 0.0, 0.5, 1.0] {
        let y = stops(s);
        for px in [[y, y, y], [y * 1.2, y, y * 0.7]] {
            if drive_luma(px) / GREY > KNEE_EV.exp2() {
                continue;
            }
            assert_eq!(
                plain.apply_to_display_linear(px),
                lifted.apply_to_display_linear(px),
                "{s}"
            );
        }
    }
    // mid grey lands on the same display value (to 1e-6, here exactly)
    let g = [GREY; 3];
    assert!(
        (plain.apply_to_display_linear(g)[1] - lifted.apply_to_display_linear(g)[1]).abs() < 1e-6
    );
    // above it the picture is brighter, and by the neutral axis' own ratio
    for s in [1.5, 2.0, 2.8, 4.0] {
        let y = stops(s);
        let a = plain.apply_to_display_linear([y, y, y]);
        let b = lifted.apply_to_display_linear([y, y, y]);
        assert!(b[1] > a[1], "{s}: {a:?} {b:?}");
    }
}

#[test]
fn the_dng_pixels_are_the_scene_values_the_app_feeds_aces_2() {
    // The DNG holds C11 decoded and lifted; Adobe's renderer puts the placement back (BaselineExposure)
    // and runs ACES 2: so the app's display value is ACES 2 of `aces2_input_matrix * dng pixel`.
    let aces = Aces2Transform::new(Aces2Output::Srgb);
    for l in [
        WhiteLift::NONE,
        lift(0.4, 1.8),
        lift(1.15, 2.78),
        lift(2.0, 3.6),
    ] {
        let c = chain_with(l);
        for px in pixels(300, 9) {
            let dng = c.apply_dng_pixel(px);
            let app = c.apply_to_display_linear(px);
            let ap0 = linalg::mul_vec(&aces2_input_matrix(), dng);
            let again = aces.transform_ap0_linear(ap0);
            for k in 0..3 {
                assert!(
                    (app[k] - again[k]).abs() < 1e-12,
                    "{px:?}: {app:?} vs {again:?} (lift {l:?})"
                );
            }
        }
    }
    // and the lift is in the DNG: the same pixel above the knee is brighter, below it identical
    let plain = chain_with(WhiteLift::NONE);
    let lifted = chain_with(lift(1.0, 2.8));
    let hi = stops(3.0);
    assert!(lifted.apply_dng_pixel([hi; 3])[1] > 1.9 * plain.apply_dng_pixel([hi; 3])[1]);
    let lo = stops(0.5);
    assert_eq!(
        lifted.apply_dng_pixel([lo; 3]),
        plain.apply_dng_pixel([lo; 3])
    );
}

fn cube(size: usize, f: impl FnMut(Vec3) -> Vec3) -> Arc<Lut> {
    Arc::new(Lut::from_cube(
        Lut3d::bake(size, [0.0; 3], [1.0; 3], f).unwrap(),
    ))
}

#[test]
fn an_imported_display_lut_and_a_display_look_keep_arris_highlights() {
    let display = |l: WhiteLift| {
        let lut = cube(9, |c| [c[0] * 0.9, c[1] * 0.95, c[2]]);
        let arri =
            ArriDisplayLut::new(lut, DisplayLutInput::LogC4, DisplayLutOutput::REC709_G24).unwrap();
        let mut p = ChainParams::new(identity_matrix(), tone());
        p.output = OutputTarget::Srgb;
        p.display = DisplayTransform::arri(arri);
        p.white_lift = l;
        ColorChain::new(&p).unwrap()
    };
    let (a, b) = (display(WhiteLift::NONE), display(lift(1.5, 2.8)));
    assert!(
        b.white_lift().is_identity(),
        "a display LUT ignores the lift"
    );
    for px in pixels(200, 11) {
        assert_eq!(a.apply_pixel(px), b.apply_pixel(px));
        assert_eq!(a.apply_dng_pixel(px), b.apply_dng_pixel(px));
    }
    // A LogC3-to-Rec.709 look: at 100 % it is theirs (no lift), at 0 % it is our C12 with the lift.
    let baked = |intensity: f64, l: WhiteLift| {
        let lut = cube(17, |c| [c[0].powf(0.9), c[1].powf(0.9), c[2].powf(0.9)]);
        let look =
            LookTransform::new(LookInput::LogC3ToRec709, None, Some((*lut).clone())).unwrap();
        let mut p = ChainParams::new(identity_matrix(), tone());
        p.output = OutputTarget::Srgb;
        p.look = Some(LookSetting::new(Arc::new(look), intensity));
        p.white_lift = l;
        ColorChain::new(&p).unwrap()
    };
    let full = (baked(100.0, WhiteLift::NONE), baked(100.0, lift(1.5, 2.8)));
    let zero_plain = chain_with(WhiteLift::NONE);
    let zero_lift = chain_with(lift(1.5, 2.8));
    let own = (baked(0.0, WhiteLift::NONE), baked(0.0, lift(1.5, 2.8)));
    for px in pixels(200, 12) {
        assert_eq!(
            full.0.apply_pixel(px),
            full.1.apply_pixel(px),
            "theirs at 100 %"
        );
        assert_eq!(
            own.0.apply_pixel(px),
            zero_plain.apply_pixel(px),
            "ours at 0 %"
        );
        assert_eq!(
            own.1.apply_pixel(px),
            zero_lift.apply_pixel(px),
            "ours with the lift at 0 %"
        );
    }
}

// ---- the estimator -----------------------------------------------------------------------------

/// A preview of `w x h` pixels: neutral AWG4 values (the matrix is the identity), a smooth ramp of
/// stops from `lo` to `hi` left to right, with a seeded ripple.
fn ramp_preview(w: usize, h: usize, lo: f64, hi: f64, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let s = lo + (hi - lo) * x as f64 / (w - 1) as f64 + 0.1 * (rng.unit() - 0.5);
            let v = stops(s) as f32;
            rgb.extend_from_slice(&[v, v, v]);
        }
    }
    rgb
}

fn estimate(rgb: &[f32], w: usize, h: usize, look: Option<&LookTransform>) -> WhiteEstimate {
    let preview = PreviewRgb {
        width: w,
        height: h,
        rgb,
        clip_mask: None,
    };
    estimate_white(&preview, &identity_matrix(), &tone(), look).unwrap()
}

#[test]
fn a_low_key_photo_gets_no_lift() {
    // Nothing above +0.7 stops: no tone to lift.
    let (w, h) = (120, 80);
    let rgb = ramp_preview(w, h, -5.0, 0.7, 5);
    let e = estimate(&rgb, w, h, None);
    assert_eq!(e.amount_ev, 0.0, "{e:?}");
    assert_eq!(e.high_key, 0.0);
    assert!(e.lift(0.0).is_identity());
}

#[test]
fn a_high_key_photo_gets_a_lift_that_reaches_the_target() {
    let (w, h) = (160, 90);
    let rgb = ramp_preview(w, h, -3.0, 3.2, 6);
    let e = estimate(&rgb, w, h, None);
    assert!(
        e.amount_ev > 0.2 && e.amount_ev <= MAX_AUTO_AMOUNT_EV,
        "{e:?}"
    );
    assert!(e.high_key > 0.5, "{e:?}");
    assert!(e.code_after > e.code_before, "{e:?}");
    assert!(e.code_after <= TARGET_CODE + 0.01, "{e:?}");
    // the percentile's tone lands on the target (the amount was not held at its limit)
    assert!((e.code_after - e.target_code).abs() < 0.05, "{e:?}");
    // the lift it describes does lift the percentile's tone there, on the neutral axis of the chain
    let l = e.lift(0.0);
    assert_eq!(l.amount_ev(), e.amount_ev);
    // (a chain without the C7 shoulder, which the estimator's neutral axis does not have either)
    let mut p = ChainParams::new(
        identity_matrix(),
        ToneParams::new(0.0, 0.0, 1e12, 0.0).unwrap(),
    );
    p.output = OutputTarget::Srgb;
    p.white_lift = l;
    let chain = ColorChain::new(&p).unwrap();
    let y = GREY * e.highlight_ev.exp2();
    let code = 255.0 * chain.apply_pixel([y, y, y])[1];
    assert!(
        (code - e.code_after).abs() < 0.05,
        "{code} vs {}",
        e.code_after
    );
}

#[test]
fn a_brighter_photo_never_ends_darker_than_a_dimmer_one() {
    // The amount itself falls once a photo is bright enough (it needs less to reach the same target),
    // but the code the brightest tones end at never falls as the photo gets brighter.
    let (w, h) = (160, 90);
    let mut prev = -1.0;
    let mut lifted = 0;
    for top in [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0, 3.6, 4.5] {
        let e = estimate(&ramp_preview(w, h, -3.0, top, 8), w, h, None);
        assert!(e.code_after >= prev - 1e-9, "top {top}: {e:?}");
        assert!(
            e.code_after <= TARGET_CODE + 0.01 || e.code_before > TARGET_CODE,
            "{e:?}"
        );
        prev = e.code_after;
        lifted += usize::from(e.amount_ev > 0.0);
    }
    assert!(lifted >= 3, "{lifted}");
}

#[test]
fn the_offset_adds_to_the_amount_and_below_zero_gives_no_lift() {
    let e = WhiteEstimate {
        amount_ev: 0.8,
        anchor_ev: 2.5,
        ..WhiteEstimate::none()
    };
    assert_eq!(
        e.lift(0.0).amount_ev(),
        0.8,
        "offset 0 is the automatic amount"
    );
    assert!((e.lift(0.5).amount_ev() - 1.3).abs() < 1e-15);
    assert!((e.lift(-0.3).amount_ev() - 0.5).abs() < 1e-15);
    assert!(e.lift(-0.8).is_identity(), "offset of -A gives no lift");
    assert!(e.lift(-2.0).is_identity(), "below -A gives no lift");
    assert_eq!(
        e.lift(100.0).amount_ev(),
        MAX_AMOUNT_EV,
        "held to the limit"
    );
    assert_eq!(e.lift(f64::NAN).amount_ev(), 0.8, "a bad offset is 0");
    assert_eq!(e.lift(0.5).anchor_ev(), 2.5);
    // a photo with no lift can still be given one
    let none = WhiteEstimate::none();
    assert!(none.lift(0.0).is_identity());
    assert!((none.lift(0.6).amount_ev() - 0.6).abs() < 1e-15);
}

#[test]
fn the_estimate_is_deterministic() {
    let (w, h) = (200, 120);
    let rgb = ramp_preview(w, h, -4.0, 3.4, 21);
    let first = estimate(&rgb, w, h, None);
    for _ in 0..3 {
        assert_eq!(estimate(&rgb, w, h, None), first);
    }
    // the same pixels in a copy at another address
    let copy = rgb.clone();
    assert_eq!(estimate(&copy, w, h, None), first);
    // and run from several threads at once: no shared state
    let results: Vec<WhiteEstimate> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| s.spawn(|| estimate(&rgb, w, h, None)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(results.iter().all(|r| *r == first));
}

#[test]
fn hostile_previews_give_finite_numbers() {
    let ok = |e: &WhiteEstimate| {
        e.amount_ev.is_finite()
            && (0.0..=MAX_AUTO_AMOUNT_EV).contains(&e.amount_ev)
            && e.anchor_ev.is_finite()
    };
    // empty, one pixel, all black, all NaN, huge values
    let empty = estimate(&[], 0, 0, None);
    assert_eq!(empty.amount_ev, 0.0);
    for (w, h, v) in [
        (1, 1, 0.5f32),
        (7, 5, 0.0),
        (9, 9, f32::NAN),
        (8, 8, 1e30),
        (6, 6, -3.0),
    ] {
        let rgb = vec![v; w * h * 3];
        let e = estimate(&rgb, w, h, None);
        assert!(ok(&e), "{w}x{h} of {v}: {e:?}");
    }
    // a wrong buffer length is an error, not a panic
    let preview = PreviewRgb {
        width: 4,
        height: 4,
        rgb: &[0.0; 10],
        clip_mask: None,
    };
    assert!(estimate_white(&preview, &identity_matrix(), &tone(), None).is_err());
    // a non-finite matrix is no lift
    let rgb = vec![0.5f32; 4 * 4 * 3];
    let preview = PreviewRgb {
        width: 4,
        height: 4,
        rgb: &rgb,
        clip_mask: None,
    };
    let mut m = identity_matrix();
    m[1][1] = f64::NAN;
    assert_eq!(
        estimate_white(&preview, &m, &tone(), None)
            .unwrap()
            .amount_ev,
        0.0
    );
}

/// The cost of one estimate on the 1/8 preview of a 61 MP frame (1188 x 792), for `docs/02`:
/// `cargo test -p ym-color --test white_lift --release -- --ignored --nocapture timing`.
#[test]
#[ignore = "timing only: run it with --ignored --nocapture"]
fn timing_of_the_estimate() {
    let (w, h) = (1188, 792);
    let rgb = ramp_preview(w, h, -6.0, 3.8, 99);
    let look = None;
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let e = estimate(&rgb, w, h, look);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        assert!(e.amount_ev >= 0.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "white lift estimate of a {w} x {h} preview: median {:.1} ms, best {:.1} ms",
        times[2], times[0]
    );
}
