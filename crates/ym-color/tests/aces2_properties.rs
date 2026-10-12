//! Structural and numerical properties of the ACES 2 output transform and its tables.
//!
//! The agreement with OpenColorIO is checked in `aces2_vectors.rs`; these tests hold the
//! properties that must stay true whatever the reference does: neutrals stay neutral, the
//! tone scale is monotonic, tables are well formed and deterministic, outputs are in range,
//! and the transform has no seam at hue 0 or at the corners of the hue table.

use ym_color::aces2::{
    Aces2Output, Aces2Params, Aces2Transform, BASE_INDEX, MAX_PEAK_LUMINANCE_NITS,
    MIN_PEAK_LUMINANCE_NITS, Primaries, TABLE_SIZE, TOTAL_TABLE_SIZE, UPPER_WRAP_INDEX,
};

const GREY: f64 = 0.18;

/// SplitMix64, a tiny deterministic generator (no external dependency).
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

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

fn all() -> Vec<&'static Aces2Transform> {
    Aces2Output::ALL
        .iter()
        .map(|&o| Aces2Transform::shared(o))
        .collect()
}

#[test]
fn neutrals_stay_neutral_and_the_ramp_is_monotonic() {
    for t in all() {
        let mut prev = [-1.0; 3];
        let mut stops = -12.0;
        while stops <= 14.0 {
            let v = GREY * 2.0_f64.powf(stops);
            let out = t.transform_ap0([v, v, v]);
            assert!(
                (out[0] - out[1]).abs() < 1e-9 && (out[1] - out[2]).abs() < 1e-9,
                "{:?}: grey {v} -> {out:?}",
                t.output()
            );
            for k in 0..3 {
                assert!(
                    out[k] >= prev[k],
                    "{:?}: ramp decreases at {stops} stops",
                    t.output()
                );
            }
            prev = out;
            stops += 0.125;
        }
        // The brightest grey reaches the display peak.
        assert!(
            prev.iter().all(|&c| c > 0.999_999),
            "{:?}: top of ramp {prev:?}",
            t.output()
        );
    }
}

#[test]
fn mid_grey_lands_on_ten_nits() {
    // 18% grey is anchored at 10.013 nit of 100 (docs: c_d); the flare term moves it by 1e-5.
    for t in all() {
        let lin = t.transform_ap0_linear([GREY, GREY, GREY]);
        for c in lin {
            assert!((c - 0.1).abs() < 2e-4, "{:?}: {lin:?}", t.output());
        }
    }
}

#[test]
fn black_negative_and_non_finite_inputs_are_safe() {
    for t in all() {
        assert_eq!(t.transform_ap0([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        assert_eq!(t.transform_ap0([-1.0, -2.0, -3.0]), [0.0, 0.0, 0.0]);
        for bad in [
            [f64::NAN, 0.18, 0.18],
            [f64::INFINITY, f64::INFINITY, f64::INFINITY],
            [f64::NEG_INFINITY, 0.18, f64::INFINITY],
            [f64::MAX, f64::MAX, f64::MAX],
            [1e300, -1e300, 0.0],
            [f64::MIN_POSITIVE, f64::MIN_POSITIVE, 0.0],
        ] {
            let out = t.transform_ap0(bad);
            assert!(
                out.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
                "{:?}: {bad:?} -> {out:?}",
                t.output()
            );
        }
    }
}

#[test]
fn bright_input_at_the_tone_scale_roof_does_not_go_black() {
    // Scene values of about 128x white put the tone-scaled lightness exactly on the display
    // maximum. OpenColorIO's float32 code returns NaN, hence black, for this input (display_p3 sweep
    // sample 1789196); the answer is white.
    // These are float32 values, written out exactly.
    let input = [
        109.496_276_855_468_75,
        135.589_599_609_375,
        239.389_389_038_085_94,
    ];
    for t in all() {
        let out = t.transform_ap0(input);
        assert!(out.iter().all(|&c| c > 0.99), "{:?}: {out:?}", t.output());
    }
    // The same band around it, finely sampled, must stay continuous and bright.
    let t = Aces2Transform::shared(Aces2Output::DisplayP3);
    for i in 0..2_000 {
        let k = 0.99 + 0.02 * i as f64 / 2_000.0;
        let out = t.transform_ap0([input[0] * k, input[1] * k, input[2] * k]);
        assert!(
            out.iter().all(|&c| c.is_finite() && c > 0.95),
            "scale {k}: {out:?}"
        );
    }
}

#[test]
fn lightness_exactly_on_the_display_maximum_stays_finite() {
    // Build colours whose tone-scaled lightness is the display maximum to the last bits, with a
    // little chroma, and walk across the rounding boundary in single-ulp steps. In the reference
    // the gamut mapper takes pow() of a tiny negative number here and returns NaN.
    let params = Aces2Params::new(100.0, Primaries::P3_D65);
    // Scene value (1.0 = 100 nit white) whose tone scale output is exactly 100 nit.
    let (mut lo, mut hi) = (10.0_f64, 1000.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if params.tone_scale.forward(mid) < 100.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let j_roof = params.input.y_to_j(hi * 100.0);
    let mut worst_white = 1.0_f64;
    for hue in [20.0, 140.0, 237.9, 300.0] {
        for m in [0.01, 1.0, 5.0] {
            for k in -400_i32..=400 {
                let j = j_roof * (1.0 + k as f64 * f64::EPSILON);
                let rgb = params.input.jmh_to_rgb([j, m, hue]);
                let out = params.output_transform_fwd(rgb);
                assert!(
                    out.iter().all(|c| c.is_finite()),
                    "NaN at hue {hue} M {m} ulp offset {k}: {out:?}"
                );
                // At the roof the gamut shrinks to the white point, so the pixel must be white.
                worst_white = worst_white.min(out[0].min(out[1]).min(out[2]));
            }
        }
    }
    assert!(
        worst_white > 0.999,
        "darkest channel near the roof: {worst_white}"
    );
}

#[test]
fn outputs_stay_in_range_for_wild_inputs() {
    let mut rng = Rng(7);
    for t in all() {
        for _ in 0..20_000 {
            let mag = 10.0_f64.powf(rng.range(-8.0, 4.5));
            let rgb = [
                mag * rng.range(-0.5, 2.0),
                mag * rng.range(-0.5, 2.0),
                mag * rng.range(-0.5, 2.0),
            ];
            let out = t.transform_ap0(rgb);
            assert!(
                out.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
                "{:?}: {rgb:?} -> {out:?}",
                t.output()
            );
            // `transform_ap0` turns a NaN into black (`f64::max` ignores NaN), so a NaN produced
            // by the maths itself would pass the check above. Look at the raw result as well.
            let raw = t.params().output_transform_fwd(rgb);
            assert!(
                raw.iter().all(|c| c.is_finite()),
                "{:?}: {rgb:?} -> raw {raw:?}",
                t.output()
            );
        }
    }
}

#[test]
fn tone_scale_does_not_invert_brighter_scene_values() {
    // Scaling a colour up never makes the luminance of the result go down.
    let mut rng = Rng(11);
    for t in all() {
        for _ in 0..2_000 {
            let base = [
                rng.range(0.0, 1.0),
                rng.range(0.0, 1.0),
                rng.range(0.0, 1.0),
            ];
            let mut prev = -1.0;
            for stop in 0..8 {
                let k = GREY * 2.0_f64.powi(stop - 4);
                let lin = t.transform_ap0_limiting_linear([base[0] * k, base[1] * k, base[2] * k]);
                // Rec.709 luma is close enough for a monotonicity check in any of the gamuts.
                let y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
                assert!(
                    y >= prev - 1e-9,
                    "{:?}: {base:?} at stop {stop}: {y} < {prev}",
                    t.output()
                );
                prev = y;
            }
        }
    }
}

#[test]
fn rec2020_overshoot_is_bounded_by_the_primaries() {
    // The P3 cube is not entirely inside Rec.2020: P3 red sits just above the red-green edge, so
    // the P3 to Rec.2020 matrix has a -0.00121 entry and cyan (0, 1, 1) lands on blue = 1.00121.
    // The reference leaves that unclamped: up to 1.00121^(1/2.4) = 1.000504 encoded, half a 10-bit
    // code value. `transform_ap0` clamps it, `transform_ap0_unclamped` keeps it. Nothing else may
    // exceed 1.0.
    let t = Aces2Transform::shared(Aces2Output::Rec2020);
    let mut rng = Rng(5);
    let mut worst = 0.0_f64;
    for _ in 0..50_000 {
        let mag = 10.0_f64.powf(rng.range(-1.0, 3.0));
        let rgb = [
            mag * rng.range(0.0, 1.0),
            mag * rng.range(0.0, 1.0),
            mag * rng.range(0.0, 1.0),
        ];
        for c in t.transform_ap0_unclamped(rgb) {
            worst = worst.max(c - 1.0);
        }
        assert!(t.transform_ap0(rgb).iter().all(|&c| c <= 1.0));
    }
    assert!(worst < 5.05e-4, "overshoot {worst}");

    // A saturated, bright cyan shows the overshoot (vector #619 of rec2020.json: OCIO gives 1.00034).
    let cyan_ap0 = [0.23877106606960297, 8.09337329864502, 9.483109474182129];
    assert!(t.transform_ap0_unclamped(cyan_ap0).iter().any(|&c| c > 1.0));
    assert!(t.transform_ap0(cyan_ap0).iter().all(|&c| c <= 1.0));

    for output in [
        Aces2Output::Viewport,
        Aces2Output::Srgb,
        Aces2Output::DisplayP3,
    ] {
        let t = Aces2Transform::shared(output);
        for _ in 0..5_000 {
            let mag = 10.0_f64.powf(rng.range(-1.0, 3.0));
            let rgb = [
                mag * rng.range(0.0, 1.0),
                mag * rng.range(0.0, 1.0),
                mag * rng.range(0.0, 1.0),
            ];
            assert!(
                t.transform_ap0_unclamped(rgb)
                    .iter()
                    .all(|&c| c <= 1.0 + 1e-12)
            );
        }
    }
}

fn assert_table_well_formed(p: &Aces2Params) {
    assert_tables(p, true);
}

/// `uniform_spacing`: also require the 100 nit SDR hue tables' spacing (about one degree). Tables
/// of other peaks and gamuts may place two corner hues almost on top of each other (the algorithm
/// keeps every corner hue), so their spacing varies more.
fn assert_tables(p: &Aces2Params, uniform_spacing: bool) {
    for (name, table) in [
        ("reach_m", &p.reach_m_table),
        ("hue", &p.hue_table),
        ("cusp_j", &p.cusp_j_table),
        ("cusp_m", &p.cusp_m_table),
        ("upper_gamma_inv", &p.upper_hull_gamma_inv_table),
    ] {
        assert_eq!(table.len(), TOTAL_TABLE_SIZE, "{name}");
        assert!(
            table.iter().all(|v| v.is_finite()),
            "{name} has non-finite entries"
        );
    }

    // Hues: strictly increasing, 1 degree on average, starting at 0, with wrap entries.
    let h = &p.hue_table;
    assert_eq!(h[BASE_INDEX], 0.0);
    for i in 1..TOTAL_TABLE_SIZE {
        assert!(
            h[i] > h[i - 1],
            "hue table not increasing at {i}: {} {}",
            h[i - 1],
            h[i]
        );
    }
    assert!(h[TABLE_SIZE] < 360.0);
    assert_eq!(h[0], h[TABLE_SIZE] - 360.0);
    assert_eq!(h[UPPER_WRAP_INDEX], h[BASE_INDEX] + 360.0);
    assert_eq!(h[UPPER_WRAP_INDEX + 1], h[BASE_INDEX + 1] + 360.0);
    for i in BASE_INDEX..UPPER_WRAP_INDEX {
        let step = h[i + 1] - h[i];
        assert!(step > 0.0 && step < 4.0, "hue spacing {step} at {i}");
        if uniform_spacing {
            assert!((0.3..=1.8).contains(&step), "hue spacing {step} at {i}");
        }
    }

    // Wrap entries of the value tables repeat the nominal ones.
    for (name, t) in [
        ("reach_m", &p.reach_m_table),
        ("cusp_j", &p.cusp_j_table),
        ("cusp_m", &p.cusp_m_table),
        ("upper_gamma_inv", &p.upper_hull_gamma_inv_table),
    ] {
        assert_eq!(t[0], t[TABLE_SIZE], "{name} lower wrap");
        assert_eq!(t[UPPER_WRAP_INDEX], t[BASE_INDEX], "{name} upper wrap");
        assert_eq!(
            t[UPPER_WRAP_INDEX + 1],
            t[BASE_INDEX + 1],
            "{name} second upper wrap"
        );
    }

    // Physical sense.
    for i in BASE_INDEX..UPPER_WRAP_INDEX {
        assert!(
            p.cusp_j_table[i] > 0.0 && p.cusp_j_table[i] < p.limit_j_max,
            "cusp J {i}"
        );
        assert!(p.cusp_m_table[i] > 0.0, "cusp M {i}");
        assert!(p.reach_m_table[i] > p.cusp_m_table[i] * 0.5, "reach M {i}");
        let gamma_inv = p.upper_hull_gamma_inv_table[i];
        assert!(
            gamma_inv > 0.2 && gamma_inv < 10.0,
            "upper hull gamma {gamma_inv} at {i}"
        );
    }
}

#[test]
fn tables_are_well_formed_and_deterministic() {
    for limiting in [Primaries::P3_D65, Primaries::REC709] {
        let a = Aces2Params::new(100.0, limiting);
        let b = Aces2Params::new(100.0, limiting);
        assert_table_well_formed(&a);
        // Bit-identical rebuilds.
        for (x, y) in [
            (&a.reach_m_table, &b.reach_m_table),
            (&a.hue_table, &b.hue_table),
            (&a.cusp_j_table, &b.cusp_j_table),
            (&a.cusp_m_table, &b.cusp_m_table),
            (&a.upper_hull_gamma_inv_table, &b.upper_hull_gamma_inv_table),
        ] {
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(a.hue_linearity_search_range, b.hue_linearity_search_range);
        assert_eq!(a.limit_j_max.to_bits(), b.limit_j_max.to_bits());
    }
}

#[test]
fn cached_transforms_use_the_same_tables_as_a_fresh_build() {
    for output in Aces2Output::ALL {
        let cached = Aces2Transform::new(output);
        let fresh = Aces2Params::new(100.0, output.limiting_primaries());
        assert_eq!(
            cached.params().cusp_m_table,
            fresh.cusp_m_table,
            "{output:?}"
        );
        assert_eq!(cached.params().hue_table, fresh.hue_table, "{output:?}");
        // A second construction is the same object content, and shared() agrees.
        assert_eq!(
            Aces2Transform::shared(output).params().reach_m_table,
            fresh.reach_m_table
        );
    }
    // Viewport, Display P3 and Rec.2020 all render with P3 limiting, so they share tables, but the
    // Rec.2020 output encodes differently.
    let v = Aces2Transform::shared(Aces2Output::Viewport);
    let p3 = Aces2Transform::shared(Aces2Output::DisplayP3);
    let r2020 = Aces2Transform::shared(Aces2Output::Rec2020);
    let px = [0.4, 0.1, 0.02];
    assert_eq!(v.transform_ap0(px), p3.transform_ap0(px));
    assert_ne!(v.transform_ap0(px), r2020.transform_ap0(px));
}

/// Largest jump between neighbouring samples of a hue sweep.
fn max_hue_step(
    t: &Aces2Transform,
    j: f64,
    m: f64,
    h_start: f64,
    h_end: f64,
    steps: usize,
) -> (f64, f64) {
    let input = Aces2Params::new(100.0, Primaries::P3_D65).input;
    let mut prev: Option<[f64; 3]> = None;
    let mut worst = 0.0_f64;
    let mut worst_h = h_start;
    for i in 0..=steps {
        let h = h_start + (h_end - h_start) * i as f64 / steps as f64;
        let rgb = input.jmh_to_rgb([j, m, h]);
        let out = t.transform_ap0_limiting_linear(rgb);
        if let Some(p) = prev {
            let step = (0..3).map(|k| (out[k] - p[k]).abs()).fold(0.0, f64::max);
            if step > worst {
                worst = step;
                worst_h = h;
            }
        }
        prev = Some(out);
    }
    (worst, worst_h)
}

#[test]
fn no_seam_at_hue_zero_or_at_the_table_corners() {
    for output in [Aces2Output::Viewport, Aces2Output::Srgb] {
        let t = Aces2Transform::shared(output);
        for (j, m) in [
            (30.0, 8.0),
            (60.0, 25.0),
            (45.0, 60.0),
            (80.0, 40.0),
            (20.0, 90.0),
        ] {
            // Across the 0/360 seam in 0.002 degree steps.
            let (seam, at) = max_hue_step(t, j, m, -1.0, 1.0, 1000);
            // A reference stretch of the same length elsewhere.
            let (away, _) = max_hue_step(t, j, m, 150.0, 152.0, 1000);
            assert!(
                seam < away * 3.0 + 1e-6,
                "{output:?} J {j} M {m}: seam step {seam} at {at} vs {away} elsewhere"
            );
        }
    }
}

#[test]
fn hue_sweep_is_continuous_everywhere() {
    // 0.01 degree steps around the whole circle; the largest linear-light step must be small and
    // spikes (discontinuities) must not stand out from the typical step by orders of magnitude.
    for output in [Aces2Output::Viewport, Aces2Output::Srgb] {
        let t = Aces2Transform::shared(output);
        for (j, m) in [(40.0, 10.0), (60.0, 30.0), (50.0, 80.0), (85.0, 20.0)] {
            let (worst, at) = max_hue_step(t, j, m, 0.0, 360.0, 36_000);
            assert!(
                worst < 1e-3,
                "{output:?} J {j} M {m}: step {worst} at hue {at}"
            );
        }
    }
}

#[test]
fn peak_luminance_outside_the_supported_range_is_limited() {
    // The reference maths has a negative tone scale roof below 46.4 nits (`f64::clamp` panics on
    // it) and never terminates for an infinite peak; the constructor limits the peak instead.
    for peak in [
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        -1.0,
        0.0,
        1e-9,
        1.0,
        10.0,
        46.0,
        1e5,
        1e9,
        f64::MAX,
    ] {
        let p = Aces2Params::new(peak, Primaries::P3_D65);
        assert!(
            (MIN_PEAK_LUMINANCE_NITS..=MAX_PEAK_LUMINANCE_NITS).contains(&p.peak_luminance_nits),
            "peak {peak} -> {}",
            p.peak_luminance_nits
        );
        let out = p.output_transform_fwd([0.18, 0.1, 0.05]);
        assert!(out.iter().all(|c| c.is_finite()), "peak {peak}: {out:?}");
    }
    // NaN falls back to the SDR peak; in-range values are kept as given.
    assert_eq!(
        Aces2Params::new(f64::NAN, Primaries::REC709).peak_luminance_nits,
        100.0
    );
    assert_eq!(
        Aces2Params::new(1000.0, Primaries::REC709).peak_luminance_nits,
        1000.0
    );
}

#[test]
fn every_supported_peak_gives_well_formed_tables_and_a_monotonic_neutral_ramp() {
    // OpenColorIO's range is 1 to 10000 nits; below 48 the tone scale roof is not positive. The
    // peak dependent maths is compared to OpenColorIO in `aces2_hdr_vectors.rs`.
    for peak in [48.0, 60.0, 100.0, 203.0, 500.0, 1000.0, 4000.0, 10_000.0] {
        for limiting in [Primaries::P3_D65, Primaries::REC709] {
            let p = Aces2Params::new(peak, limiting);
            assert_tables(&p, false);
            let mut prev = -1.0;
            let mut stops = -12.0;
            while stops <= 16.0 {
                let v = GREY * 2.0_f64.powf(stops);
                let out = p.output_transform_fwd([v, v, v]);
                assert!(
                    out.iter().all(|c| c.is_finite()),
                    "peak {peak}: grey {v} -> {out:?}"
                );
                assert!(
                    (out[0] - out[1]).abs() < 1e-9 * out[0].max(1.0)
                        && (out[1] - out[2]).abs() < 1e-9 * out[0].max(1.0),
                    "peak {peak}: grey {v} -> {out:?}"
                );
                assert!(
                    out[0] >= prev,
                    "peak {peak}: ramp decreases at {stops} stops"
                );
                prev = out[0];
                stops += 0.25;
            }
            // The top of the ramp is the display peak (1.0 = 100 nits), plus the overshoot of the
            // tone scale's flare term that the final clamp removes (0.4% to 1.5% from 60 to
            // 10 000 nits, 10% at 48 nits where the roof is 5.6 times white).
            let peak_linear = peak / 100.0;
            assert!(
                prev > 0.999 * peak_linear && prev < 1.11 * peak_linear,
                "peak {peak}: top of ramp {prev}"
            );
        }
    }
}
