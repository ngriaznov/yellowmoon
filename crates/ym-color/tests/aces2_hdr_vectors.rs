//! The peak-dependent maths of the ACES 2 output transform against OpenColorIO.
//!
//! Yellowmoon V1 renders 100 nit SDR only, but `Aces2Params::new` takes any peak between
//! `MIN_PEAK_LUMINANCE_NITS` and `MAX_PEAK_LUMINANCE_NITS` (the WGSL port and HDR export in V2 will
//! use it). `tests/data/aces2/hdr_p3.json` (written by `tools/vectors/aces2_vectors.py`) holds
//! OpenColorIO's float32 output, P3-D65 limited, for five peaks. The gate is a fraction of the
//! display peak: |ours - expected| / (peak / 100 nit) <= 1e-4 (a tenth of a 10-bit code value
//! near white; measured worst case 1.1e-5).

use serde_json::Value;
use ym_color::aces2::{Aces2Params, Primaries};

const GATE_FRACTION_OF_PEAK: f64 = 1e-4;

fn triplets(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .expect("array of triplets")
        .iter()
        .map(|row| {
            let r = row.as_array().expect("triplet");
            // Float32 values written as their shortest decimal; recover the f32.
            let f = |i: usize| r[i].as_f64().expect("number") as f32 as f64;
            [f(0), f(1), f(2)]
        })
        .collect()
}

#[test]
fn hdr_peaks_match_ocio() {
    let path = format!(
        "{}/tests/data/aces2/hdr_p3.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(doc["schema"], "yellowmoon/aces2-hdr-vectors/1");
    assert_eq!(doc["limiting_primaries"], "P3-D65");
    assert_eq!(doc["gate"]["max"].as_f64(), Some(GATE_FRACTION_OF_PEAK));

    let peaks = doc["peaks"].as_array().expect("peaks");
    assert!(peaks.len() >= 5, "{} peaks", peaks.len());
    for entry in peaks {
        let peak = entry["peak_luminance_nits"].as_f64().expect("peak");
        let input = triplets(&entry["input_ap0"]);
        let expected = triplets(&entry["expected_limiting_linear"]);
        assert_eq!(input.len(), expected.len());
        assert!(input.len() > 100, "peak {peak}: {} vectors", input.len());

        let params = Aces2Params::new(peak, Primaries::P3_D65);
        assert_eq!(params.peak_luminance_nits, peak);
        let peak_linear = peak / 100.0;

        let mut worst = (0.0_f64, 0_usize);
        for (i, (rgb, want)) in input.iter().zip(&expected).enumerate() {
            // The same clamp to [0, peak] the reference chain ends with.
            let got = params
                .output_transform_fwd(*rgb)
                .map(|c| c.clamp(0.0, peak_linear));
            let err = (0..3)
                .map(|k| (got[k] - want[k]).abs() / peak_linear)
                .fold(0.0, f64::max);
            if err > worst.0 {
                worst = (err, i);
            }
        }
        println!(
            "peak {peak:>7} nit: {} vectors, worst error {:.2e} of the peak (#{})",
            input.len(),
            worst.0,
            worst.1
        );
        assert!(
            worst.0 <= GATE_FRACTION_OF_PEAK,
            "peak {peak}: vector #{} input {:?} ours {:?} expected {:?}: {:.3e}",
            worst.1,
            input[worst.1],
            params.output_transform_fwd(input[worst.1]),
            expected[worst.1],
            worst.0
        );
    }
}
