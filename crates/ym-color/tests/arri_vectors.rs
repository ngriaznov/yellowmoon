//! M1-01: our LogC / AWG chain against OpenColorIO's built-in ARRI transforms (formulas only, no
//! ARRI data files). Vectors from tools/vectors/lut_arri_builtin_vectors.py (float32 results).

use serde_json::Value;
use ym_color::logc::{LogC3Ei800, LogC4};

fn load() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/lut/arri_builtin_vectors.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn vec3(v: &Value) -> [f64; 3] {
    let a = v.as_array().unwrap();
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

/// Worst error relative to the output scale `max(|y|, floor)`.
fn worst_relative(case: &Value, floor: f64, ours: impl Fn([f64; 3]) -> [f64; 3]) -> (f64, f64) {
    let (mut rel, mut abs) = (0.0_f64, 0.0_f64);
    for (i, o) in case["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .zip(case["outputs"].as_array().unwrap())
    {
        let (i, o) = (vec3(i), vec3(o));
        let got = ours(i);
        for c in 0..3 {
            let d = (got[c] - o[c]).abs();
            abs = abs.max(d);
            rel = rel.max(d / o[c].abs().max(floor));
        }
    }
    (rel, abs)
}

#[test]
fn logc4_awg4_to_ap0_matches_opencolorio() {
    let doc = load();
    let case = &doc["cases"]["logc4_awg4_to_ap0"];
    assert_eq!(case["style"], "ARRI_LOGC4_to_ACES2065-1");
    let (rel, abs) = worst_relative(case, 1e-2, |v| LogC4(v).to_linear().to_ap0().0);
    println!(
        "LogC4 / AWG4 -> AP0 vs OCIO {}: worst relative {rel:e}, absolute {abs:e}",
        doc["ocio_version"]
    );
    // OCIO's log op uses float32 exp / pow approximations (~2.5e-4 relative, measured); the tight
    // comparison with colour-science in float64 is in tests/logc.rs and tests/consts_gamut.rs.
    assert!(rel < 1e-3, "{rel:e}");
}

#[test]
fn logc3_awg3_to_ap0_matches_opencolorio_within_the_rounded_constants() {
    let doc = load();
    let case = &doc["cases"]["logc3_ei800_awg3_to_ap0"];
    assert_eq!(case["style"], "ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1");
    let (rel, abs) = worst_relative(case, 1e-2, |v| LogC3Ei800(v).to_linear().to_ap0().0);
    println!("LogC3 EI800 / AWG3 -> AP0 vs OCIO: worst relative {rel:e}, absolute {abs:e}");
    // OCIO uses ARRI's constants at full precision (0.2471896383, 0.3855369987, ...); the LogC3 VFX
    // document and docs/02 print them rounded to 6 digits, which is what we implement. Together with
    // OCIO's own float32 approximations the two agree to about 1e-3 relative on this range.
    assert!(rel < 1e-3, "{rel:e}");
}
