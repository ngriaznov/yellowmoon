//! M2-03: property tests for the cRAW test encoder and the quantisation map.
//!
//! For random synthetic mosaics (uniform noise, gradients, clipped highlights, spikes and a
//! mix of block spreads that reaches every delta shift) and random camera curves, encode with the
//! test-only encoder, decode, and check that every TRUE value lies within `qstep / 2` of the
//! decoded value (nearest rounding) or in `[decoded, decoded + qstep)` (floor rounding), in
//! output DN.

use half::f16;
use proptest::prelude::*;
use ym_raw::decode::craw::{build_curve, decode_rows};
use ym_raw::decode::{DecodeOptions, decode_bytes};
use ym_raw::testgen::{ArwBuilder, CrawEncoder, RawPayload, Rounding, SplitMix64};

/// Four stored knots: increasing 12-bit values shifted into the top 14 bits.
fn knots() -> impl Strategy<Value = [u16; 4]> {
    (
        proptest::collection::vec(0u16..=4095, 4),
        proptest::collection::vec(0u16..4, 4),
    )
        .prop_map(|(mut values, low)| {
            values.sort_unstable();
            [0, 1, 2, 3].map(|i| (values[i] << 2) | low[i])
        })
}

/// A mosaic of `w x h` true values in `0..=top` whose 16-pixel blocks have random bases and
/// spreads, so the deltas cover every shift from 0 to 4 and flat blocks occur.
fn mosaic(rng: &mut SplitMix64, w: usize, h: usize, top: u16, kind: u8) -> Vec<u16> {
    let mut out = vec![0u16; w * h];
    let spreads = [0u64, 3, 40, 300, 1500, u64::from(top) + 1];
    for row in 0..h {
        for group in 0..w / 32 {
            for half in 0..2 {
                let base = rng.below(u64::from(top) + 1);
                let spread = spreads[rng.below(spreads.len() as u64) as usize];
                for i in 0..16 {
                    let x = group * 32 + half + 2 * i;
                    let value = match kind {
                        0 => rng.below(u64::from(top) + 1),
                        1 => (base + rng.below(spread + 1)).min(u64::from(top)),
                        2 => {
                            if rng.below(16) == 0 {
                                rng.below(u64::from(top) + 1)
                            } else {
                                u64::from(top)
                            }
                        }
                        _ => {
                            if rng.below(8) == 0 {
                                u64::from(top)
                            } else {
                                base.min(spread)
                            }
                        }
                    };
                    out[row * w + x] = value as u16;
                }
            }
        }
    }
    out
}

fn check(encoder: &CrawEncoder, truth: &[u16], w: usize, h: usize) -> Result<(), TestCaseError> {
    let curve = build_curve(encoder.knots).expect("increasing knots");
    let payload = encoder.encode(truth, w as u32, h as u32);
    let (data, qstep) = decode_rows(&payload[0], w, h, &curve, true).expect("decodes");
    prop_assert_eq!(qstep.len(), data.len());
    for (i, ((t, d), q)) in truth.iter().zip(&data).zip(&qstep).enumerate() {
        let q: f16 = *q;
        let q = q.to_f32();
        prop_assert!(q > 0.0, "pixel {i}: qstep is zero");
        let (t, d) = (f32::from(*t), f32::from(*d));
        match encoder.rounding {
            Rounding::Nearest => prop_assert!(
                2.0 * (t - d).abs() <= q,
                "pixel {i}: true {t} decoded {d} qstep {q}"
            ),
            Rounding::Floor => prop_assert!(
                d <= t && t - d < q,
                "pixel {i}: true {t} decoded {d} qstep {q}"
            ),
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Every true value is within half a quantisation step of the decoded one.
    #[test]
    fn true_values_lie_within_half_a_qstep(seed in any::<u64>(), stored in knots(), kind in 0u8..4) {
        let encoder = CrawEncoder::new().knots(stored).rounding(Rounding::Nearest);
        let mut rng = SplitMix64(seed);
        let truth = mosaic(&mut rng, 96, 8, encoder.max_value(), kind);
        check(&encoder, &truth, 96, 8)?;
    }

    /// The same with the camera's own curve, which is what real files carry.
    #[test]
    fn the_real_camera_curve_keeps_the_bound(seed in any::<u64>(), kind in 0u8..4) {
        let encoder = CrawEncoder::new();
        let mut rng = SplitMix64(seed);
        let truth = mosaic(&mut rng, 128, 8, encoder.max_value(), kind);
        check(&encoder, &truth, 128, 8)?;
    }

    /// An encoder that truncates puts the true value at or above the decoded one, less than a
    /// step away.
    #[test]
    fn truncating_encoders_stay_within_one_qstep(seed in any::<u64>(), stored in knots(), kind in 0u8..4) {
        let encoder = CrawEncoder::new().knots(stored).rounding(Rounding::Floor);
        let mut rng = SplitMix64(seed);
        let truth = mosaic(&mut rng, 96, 8, encoder.max_value(), kind);
        check(&encoder, &truth, 96, 8)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// The same bound holds through a whole file: container, strip, probe, decode.
    #[test]
    fn the_bound_holds_through_a_file(seed in any::<u64>(), kind in 0u8..4) {
        let (w, h) = (96u32, 8u32);
        let encoder = CrawEncoder::new();
        let mut rng = SplitMix64(seed);
        let truth = mosaic(&mut rng, w as usize, h as usize, encoder.max_value(), kind);
        let file = ArwBuilder::new("ILCE-7M3", w, h).craw(encoder).build(&truth);
        let raw = decode_bytes(&file, &DecodeOptions::default()).expect("decodes");
        let qstep = raw.qstep.expect("cRAW has qstep");
        for (i, ((t, d), q)) in truth.iter().zip(&raw.data.data).zip(&qstep.data).enumerate() {
            prop_assert!(
                2.0 * (f32::from(*t) - f32::from(*d)).abs() <= q.to_f32(),
                "pixel {i}: true {t} decoded {d} qstep {q}"
            );
        }
    }
}

#[test]
fn qstep_is_the_curve_slope_times_the_block_step_in_flat_curve_segments() {
    // With every knot at the far end of the index range the curve has slope 1 everywhere, so one
    // 11-bit code is two output DN and a block with max - min = 1000 (shift 3) has steps of 16.
    let encoder = CrawEncoder::new().knots([4095 << 2; 4]);
    let curve = encoder.curve();
    assert_eq!(curve[4094], 4094);
    let mut truth = vec![1000u16; 32];
    // Even columns form block 0: one pixel high, one low, the rest in between.
    truth[0] = 1000;
    truth[2] = 0;
    truth[4] = 2000;
    let payload = encoder.encode(&truth, 32, 1);
    let (data, qstep) = decode_rows(&payload[0], 32, 1, &curve, true).expect("decodes");
    // max code 1000 (value 2000), min code 0: difference 1000, shift 3, so the pixels that are
    // neither max nor min have a step of 8 codes = 16 DN.
    let q = |x: usize| qstep[x].to_f32();
    assert_eq!(q(0), 16.0, "a delta pixel: 8 codes of 2 DN");
    assert_eq!(q(2), 2.0, "the minimum is exact: one code");
    assert_eq!(q(4), 2.0, "the maximum is exact: one code");
    assert_eq!(data[2], 0);
    assert_eq!(data[4], 2000);
}

#[test]
fn the_default_encoder_matches_the_files_real_cameras_write() {
    // The knots of every real cRAW file we have seen (A300 to A7 III) decode to this curve.
    let encoder = CrawEncoder::new();
    assert_eq!(encoder.knots, [8000, 10400, 12900, 14100]);
    let curve = encoder.curve();
    assert_eq!(curve[2000], 2000);
    assert_eq!(curve[2600], 2000 + 2 * 600);
    assert_eq!(curve[3225], 3200 + 4 * 625);
    assert_eq!(curve[3525], 5700 + 8 * 300);
    assert_eq!(curve[4094], 8100 + 16 * (4094 - 3525));
}

#[test]
fn pixels_in_the_top_bucket_get_a_wider_qstep() {
    // Slope 1 everywhere: the output of code c is 2c. A block with max code 2047 and min code 0
    // has shift 4, so deltas count in steps of 16 codes and the largest delta (127) decodes to
    // code 2032, while pixels up to code 2047 land in that bucket.
    let encoder = CrawEncoder::new().knots([4095 << 2; 4]);
    let curve = encoder.curve();
    let mut truth = vec![1000u16; 32];
    truth[0] = 4094; // code 2047: the maximum, exact
    truth[2] = 0; // code 0: the minimum, exact
    truth[4] = 4092; // code 2046: top bucket, decodes to code 2032
    let payload = encoder.encode(&truth, 32, 1);
    let (data, qstep) = decode_rows(&payload[0], 32, 1, &curve, true).expect("decodes");
    assert_eq!(data[4], 4064);
    // The ordinary step would be 2 * 16 = 32 DN; the pixel is 28 DN away, which needs more than
    // 32 / 2. Twice the distance to the block maximum plus the maximum's own rounding slack.
    assert_eq!(qstep[4].to_f32(), 2.0 * 30.0 + 2.0);
    assert!(2.0 * (4092.0 - f32::from(data[4])) <= qstep[4].to_f32());
}

// ---------------------------------------------------------------------------------------------
// Exact values (review): the property tests above only bound the error from one side, so a map
// that was uniformly too wide, or a decoder that read the blocks from the wrong place, would pass
// them. This model reads the 128-bit blocks bit by bit and applies the definition of docs/04
// literally, in f64 and without lookup tables, and the decoder has to agree with it everywhere,
// also on arbitrary bytes (which the camera never writes but a damaged file contains).
// ---------------------------------------------------------------------------------------------

/// The curve output for an 11-bit code, continuing the last slope above code 2047.
fn model_curve(curve: &[u16], code: i64) -> f64 {
    if code <= 2047 {
        f64::from(curve[(code as usize) << 1])
    } else {
        let top = f64::from(curve[2047 << 1]);
        top + (top - f64::from(curve[2046 << 1])) * (code - 2047) as f64
    }
}

/// `(value, qstep)` of every pixel of the rows in `payload`, from the written definition.
fn model(payload: &[u8], w: usize, h: usize, curve: &[u16]) -> (Vec<u16>, Vec<f64>) {
    let bit = |row: usize, block_start: usize, i: usize| -> u32 {
        // Bit `i` of the block that starts at byte `block_start` of the row; bits past the end of
        // the row read as zero (LibRaw reads the byte after the block for a fifteenth delta).
        let byte = block_start + i / 8;
        if byte >= w {
            return 0;
        }
        u32::from(payload[row * w + byte] >> (i % 8)) & 1
    };
    let field = |row: usize, start: usize, from: usize, bits: usize| -> i64 {
        (0..bits).fold(0i64, |acc, k| {
            acc | (i64::from(bit(row, start, from + k)) << k)
        })
    };
    let mut values = vec![0u16; w * h];
    let mut steps = vec![0f64; w * h];
    for row in 0..h {
        for group in 0..w / 32 {
            for half in 0..2 {
                let start = group * 32 + half * 16;
                let (max, min) = (field(row, start, 0, 11), field(row, start, 11, 11));
                let (imax, imin) = (field(row, start, 22, 4), field(row, start, 26, 4));
                // shift = max(0, bitlen(max - min) - 7), at most 4.
                let spread = (max - min).max(0);
                let bitlen = (64 - (spread as u64).leading_zeros()) as i64;
                let shift = (bitlen - 7).clamp(0, 4);
                let mut next_delta_at = 30;
                for i in 0..16usize {
                    let x = group * 32 + half + 2 * i;
                    let exact = i as i64 == imax || i as i64 == imin;
                    let (code, delta) = if i as i64 == imax {
                        (max, None)
                    } else if i as i64 == imin {
                        (min, None)
                    } else {
                        let d = field(row, start, next_delta_at, 7);
                        next_delta_at += 7;
                        (((d << shift) + min).min(2047), Some(d))
                    };
                    let k = if exact { 1 } else { 1i64 << shift };
                    let up = model_curve(curve, code + k) - model_curve(curve, code);
                    let down = if code >= k {
                        model_curve(curve, code) - model_curve(curve, code - k)
                    } else {
                        0.0
                    };
                    let mut step = up.max(down);
                    if delta == Some(127) {
                        // Everything from this code up to the block maximum lands in this bucket.
                        let wide = 2.0
                            * (model_curve(curve, max) - model_curve(curve, code)).max(0.0)
                            + (model_curve(curve, max + 1) - model_curve(curve, max));
                        step = step.max(wide);
                    }
                    values[row * w + x] = curve[(code as usize) << 1];
                    steps[row * w + x] = step;
                }
            }
        }
    }
    (values, steps)
}

fn assert_matches_model(payload: &[u8], w: usize, h: usize, curve: &[u16]) {
    let (data, qstep) = decode_rows(payload, w, h, curve, true).expect("decodes");
    let (want_values, want_steps) = model(payload, w, h, curve);
    assert_eq!(data, want_values, "decoded values");
    for (i, (got, want)) in qstep.iter().zip(&want_steps).enumerate() {
        // f16 holds every integer up to 2048 exactly; the steps here stay below that.
        assert!(
            *want < 2048.0,
            "pixel {i}: the model step {want} is beyond f16's exact range"
        );
        assert_eq!(f64::from(got.to_f32()), *want, "qstep of pixel {i}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Blocks written by the encoder: every qstep is exactly the definition.
    #[test]
    fn qstep_equals_the_written_definition_on_encoded_images(
        seed in any::<u64>(),
        stored in knots(),
        kind in 0u8..4,
        floor in any::<bool>(),
    ) {
        let rounding = if floor { Rounding::Floor } else { Rounding::Nearest };
        let encoder = CrawEncoder::new().knots(stored).rounding(rounding);
        let mut rng = SplitMix64(seed);
        let truth = mosaic(&mut rng, 96, 4, encoder.max_value(), kind);
        let curve = build_curve(encoder.knots).expect("increasing knots");
        let payload = encoder.encode(&truth, 96, 4);
        assert_matches_model(&payload[0], 96, 4, &curve);
    }

    /// Arbitrary bytes, including blocks whose maximum is below their minimum and blocks whose
    /// maximum and minimum share a position: the decoder and the model still agree.
    #[test]
    fn qstep_equals_the_written_definition_on_arbitrary_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 64 * 3),
        stored in knots(),
    ) {
        let curve = build_curve(stored).expect("increasing knots");
        assert_matches_model(&bytes, 64, 3, &curve);
    }
}

#[test]
fn a_degenerate_curve_still_gives_finite_steps() {
    // All four knots at zero: slope 16 everywhere, the steepest curve the format can describe.
    // The widened top bucket must not overflow half precision.
    let curve = build_curve([0, 0, 0, 0]).expect("a flat start is increasing");
    let mut payload = vec![0u8; 32];
    // Both blocks: max code 2047, min code 0, so shift 4 and deltas that reach 127.
    for half in 0..2 {
        let word: u128 = 2047 | (3 << 22) | (4 << 26) | (0x7Fu128 << 30) | (0x7Fu128 << 37);
        payload[half * 16..half * 16 + 16].copy_from_slice(&word.to_le_bytes());
    }
    let (_, qstep) = decode_rows(&payload, 32, 1, &curve, true).expect("decodes");
    assert!(
        qstep
            .iter()
            .all(|q| q.to_f32().is_finite() && q.to_f32() > 0.0)
    );
}
