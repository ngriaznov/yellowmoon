//! M1-02 runtime: the spectral engine against `tests/data/spectral_reference.json`.
//!
//! The reference file is written by `python3 -m training.sensor_map.spectral_reference` from the shipped
//! A7R IV profile and the ALEXA estimate. Its `interpolated` fields are what the documented table
//! interpolation gives (same arithmetic as this crate, so 1e-9); its `exact` fields come straight from
//! the spectra and can only be matched within the `tolerances` recorded in the file (a 10 K table cannot
//! follow a blackbody below 2000 K or the three-decimal staircase of colour-science's D series to 1e-6).
//! The largest differences are printed (`cargo test -p ym-color --test spectral -- --nocapture`).

use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};
use ym_color::spectral::{
    AlexaWhite, ClassProbabilities, LightClass, Locus, SpectralCurves, SpectralError,
};

const PROFILE_ID: &str = "sony-ilce-7rm4";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference() -> Value {
    serde_json::from_str(include_str!("data/spectral_reference.json")).expect("reference parses")
}

fn profile_value(id: &str) -> Value {
    let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("profile parses")
}

fn tables(id: &str) -> (Locus, AlexaWhite) {
    let v = profile_value(id);
    let locus: Locus = serde_json::from_value(v["locus"].clone()).expect("locus section");
    let white: AlexaWhite =
        serde_json::from_value(v["alexa_white"].clone()).expect("alexa_white section");
    (locus, white)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}

fn vec_f<const N: usize>(v: &Value) -> [f64; N] {
    let a = v.as_array().expect("array");
    assert_eq!(a.len(), N);
    std::array::from_fn(|i| f(&a[i]))
}

fn max_abs_diff<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    a.iter().zip(&b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// Deterministic SplitMix64 in [0, 1).
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

#[test]
fn reference_file_belongs_to_the_shipped_files() {
    let r = reference();
    let profile =
        std::fs::read(repo_root().join("data/cameras/profiles/sony-ilce-7rm4/profile.json"))
            .unwrap();
    assert_eq!(
        sha256_hex(&profile),
        r["sources"]["profile_sha256"].as_str().unwrap(),
        "profile.json changed: rerun `python3 -m training.sensor_map.spectral_reference`"
    );
    let alexa =
        std::fs::read(repo_root().join("data/alexa/alexa_alev3_estimate_380_780_5.json")).unwrap();
    assert_eq!(
        sha256_hex(&alexa),
        r["sources"]["alexa_curves_sha256"].as_str().unwrap()
    );
    assert_eq!(r["camera_locus"]["profile"], PROFILE_ID);
}

#[test]
fn every_shipped_profile_has_valid_spectral_tables() {
    let dir = repo_root().join("data/cameras/profiles");
    let mut n = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let id = entry.unwrap().file_name().into_string().unwrap();
        let (locus, white) = tables(&id);
        locus.validate().unwrap_or_else(|e| panic!("{id}: {e}"));
        white.validate().unwrap_or_else(|e| panic!("{id}: {e}"));
        n += 1;
    }
    assert_eq!(n, 11);
}

/// Which tolerance of the file applies to an off-grid temperature of a table.
fn off_grid_tolerance(r: &Value, kind: &str, quantity: &str, t_k: f64) -> f64 {
    let t = &r["tolerances"]["off_grid_vs_exact_abs"];
    if kind == "blackbody" && t_k >= 2000.0 {
        f(&t[format!("blackbody_{quantity}_above_2000k")])
    } else {
        f(&t[format!("{kind}_{quantity}")])
    }
}

#[test]
fn w_a_tables_reproduce_the_reference() {
    let r = reference();
    let (_, white) = tables(PROFILE_ID);
    let interp_tol = f(&r["tolerances"]["interpolated_abs"]);
    let grid_tol = f(&r["tolerances"]["on_grid_abs"]);
    let mut worst = std::collections::BTreeMap::<String, f64>::new();
    let mut note = |k: &str, v: f64| {
        let e = worst.entry(k.to_string()).or_insert(0.0);
        *e = e.max(v);
    };
    for (section, grid_point) in [("grid_points", true), ("off_grid", false)] {
        for kind in ["blackbody", "daylight"] {
            for case in r["alexa_w_a"][section][kind].as_array().unwrap() {
                let t = f(&case["cct_k"]);
                let got = if kind == "blackbody" {
                    white.blackbody_white(t)
                } else {
                    white.daylight_white(t)
                };
                let interpolated = vec_f::<3>(&case["interpolated"]);
                let exact = vec_f::<3>(&case["exact"]);
                let d_interp = max_abs_diff(got, interpolated);
                let d_exact = max_abs_diff(got, exact);
                assert!(
                    d_interp <= interp_tol,
                    "{kind} {t} K: {got:?} vs interpolated {interpolated:?} ({d_interp:e})"
                );
                assert_eq!(got[1], 1.0, "green must stay exactly 1");
                note(&format!("w_A {kind} interpolated"), d_interp);
                if grid_point {
                    assert!(
                        d_exact <= grid_tol,
                        "{kind} {t} K on grid: {d_exact:e} > {grid_tol:e}"
                    );
                    note(&format!("w_A {kind} on grid vs exact"), d_exact);
                } else {
                    let tol = off_grid_tolerance(&r, kind, "w_a", t);
                    assert!(
                        d_exact <= tol,
                        "{kind} {t} K off grid: {d_exact:e} > {tol:e}"
                    );
                    note(&format!("w_A {kind} off grid vs exact"), d_exact);
                }
            }
        }
    }
    for (k, v) in worst {
        eprintln!("max |difference| {k}: {v:.3e}");
    }
}

#[test]
fn camera_locus_reproduces_the_reference() {
    let r = reference();
    let (locus, _) = tables(PROFILE_ID);
    let interp_tol = f(&r["tolerances"]["interpolated_abs"]);
    let grid_tol = f(&r["tolerances"]["on_grid_abs"]);
    let mut worst = std::collections::BTreeMap::<String, f64>::new();
    let mut note = |k: &str, v: f64| {
        let e = worst.entry(k.to_string()).or_insert(0.0);
        *e = e.max(v);
    };
    for (section, grid_point) in [("grid_points", true), ("off_grid", false)] {
        for kind in ["blackbody", "daylight"] {
            for case in r["camera_locus"][section][kind].as_array().unwrap() {
                let t = f(&case["cct_k"]);
                let (chroma, xy) = if kind == "blackbody" {
                    (
                        locus.blackbody.lerp_mired(t),
                        locus.xy.blackbody.lerp_mired(t),
                    )
                } else {
                    (
                        locus.daylight.lerp_mired(t),
                        locus.xy.daylight.lerp_mired(t),
                    )
                };
                let d_chroma = max_abs_diff(chroma, vec_f(&case["ln_chroma_interpolated"]));
                let d_xy = max_abs_diff(xy, vec_f(&case["xy_interpolated"]));
                assert!(
                    d_chroma <= interp_tol && d_xy <= interp_tol,
                    "{kind} {t} K: {d_chroma:e} {d_xy:e}"
                );
                note(&format!("locus {kind} interpolated"), d_chroma.max(d_xy));
                let e_chroma = max_abs_diff(chroma, vec_f(&case["ln_chroma_exact"]));
                let e_xy = max_abs_diff(xy, vec_f(&case["xy_exact"]));
                if grid_point {
                    assert!(
                        e_chroma <= grid_tol && e_xy <= grid_tol,
                        "{kind} {t} K on grid: {e_chroma:e} {e_xy:e}"
                    );
                    note(
                        &format!("locus {kind} on grid vs exact"),
                        e_chroma.max(e_xy),
                    );
                } else {
                    let tol = off_grid_tolerance(&r, kind, "locus", t);
                    assert!(
                        e_chroma <= tol && e_xy <= tol,
                        "{kind} {t} K off grid: {e_chroma:e} {e_xy:e} > {tol:e}"
                    );
                    note(
                        &format!("locus {kind} off grid vs exact"),
                        e_chroma.max(e_xy),
                    );
                }
                // The public accessors pick the same table by the 4000 K rule.
                if (kind == "blackbody" && t < 4000.0) || (kind == "daylight" && t >= 4000.0) {
                    assert_eq!(locus.chroma_at(t), chroma);
                    assert_eq!(locus.xy_at(t), xy);
                }
            }
        }
    }
    for (k, v) in worst {
        eprintln!("max |difference| {k}: {v:.3e}");
    }
}

#[test]
fn cct_and_duv_readout_reproduces_the_reference() {
    let r = reference();
    let (locus, _) = tables(PROFILE_ID);
    let mut worst_cct = 0.0_f64;
    let mut worst_duv = 0.0_f64;
    for case in r["cct_duv"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let got = locus.rgb_to_cct_duv(vec_f(&case["camera_rgb"])).unwrap();
        let (cct, duv) = (f(&case["cct_k"]), f(&case["duv"]));
        let d_cct = (got.cct_k - cct).abs() / cct;
        let d_duv = (got.duv - duv).abs();
        assert!(
            d_cct <= 1e-9 && d_duv <= 1e-9,
            "{name}: {got:?} vs ({cct}, {duv})"
        );
        worst_cct = worst_cct.max(d_cct);
        worst_duv = worst_duv.max(d_duv);
        // Lights of the locus family agree with CIE colorimetry: the table readout stays within about a
        // kelvin of the generating temperature (the D series' staircase) and within 3e-4 Duv of CIE's
        // value measured from the same locus light (the file's stated accuracy).
        if name.starts_with("blackbody")
            || name.starts_with("CIE D series")
            || name.starts_with("daylight")
        {
            let t = f(&case["generating_cct_k"]);
            let tint = name.contains("green");
            if !tint {
                assert!(
                    (got.cct_k - t).abs() < 1.0,
                    "{name}: {} K vs generating {t} K",
                    got.cct_k
                );
            }
            let cie = f(&case["cie_duv_vs_locus_magenta_positive"]);
            assert!(
                (got.duv - cie).abs() < 3e-4,
                "{name}: Duv {} vs CIE {cie}",
                got.duv
            );
        }
    }
    eprintln!(
        "cct/duv readout: max relative CCT difference {worst_cct:.2e}, max Duv difference {worst_duv:.2e} (vs runtime_ref)"
    );
}

#[test]
fn cct_and_duv_inverse_reproduces_the_reference() {
    let r = reference();
    let (locus, _) = tables(PROFILE_ID);
    for case in r["cct_duv"]["inverse_cases"].as_array().unwrap() {
        let (cct, duv) = (f(&case["cct_k"]), f(&case["duv"]));
        let rgb = locus.cct_duv_to_rgb(cct, duv);
        let d = max_abs_diff(rgb, vec_f(&case["camera_rgb"]));
        assert!(d <= 1e-9, "{cct} K, {duv}: {rgb:?} ({d:e})");
        assert_eq!(rgb[1], 1.0);
        let back = locus.rgb_to_cct_duv(rgb).unwrap();
        let (rt_cct, rt_duv) = (f(&case["roundtrip_cct_k"]), f(&case["roundtrip_duv"]));
        assert!((back.cct_k - rt_cct).abs() <= 1e-9 * rt_cct && (back.duv - rt_duv).abs() <= 1e-9);
        // README: the round trip is below 1e-6 K and 1e-8 Duv, except at 1600 K where the chord
        // curvature moves the foot point by 0.3 K.
        let cct_limit = if cct < 1700.0 { 0.35 } else { 1e-6 };
        assert!(
            (back.cct_k - cct).abs() <= cct_limit,
            "{cct} K: came back as {} K",
            back.cct_k
        );
        assert!(
            (back.duv - duv).abs() <= if cct < 1700.0 { 2e-6 } else { 1e-8 },
            "{cct} K {duv}: {}",
            back.duv
        );
    }
}

#[test]
fn class_w_a_lists_reproduce_the_reference() {
    let r = reference();
    let (_, white) = tables(PROFILE_ID);
    let mut worst = 0.0_f64;
    for case in r["class_w_a"].as_array().unwrap() {
        let class = match case["class"].as_str().unwrap() {
            "fluorescent" => LightClass::Fluorescent,
            "led" => LightClass::Led,
            other => panic!("unexpected class {other}"),
        };
        let mired = f(&case["mired"]);
        let got = white.class_white(class, mired);
        let d = max_abs_diff(got, vec_f(&case["w"]));
        assert!(d <= 1e-9, "{class:?} at {mired} mired: {got:?} ({d:e})");
        assert_eq!(got[1], 1.0);
        worst = worst.max(d);
    }
    eprintln!("class w_A lists: max difference {worst:.2e}");
}

#[test]
fn locus_round_trips_over_the_whole_range_on_every_profile() {
    let dir = repo_root().join("data/cameras/profiles");
    let mut rng = Rng(0x5EED_0001);
    let mut worst_bb = (0.0_f64, 0.0_f64);
    let mut worst_dy = (0.0_f64, 0.0_f64);
    for entry in std::fs::read_dir(dir).unwrap() {
        let id = entry.unwrap().file_name().into_string().unwrap();
        let (locus, _) = tables(&id);
        for _ in 0..300 {
            let t = 1500.0 + rng.next_f64() * 23500.0;
            let rgb = locus.rgb_at(t);
            let back = locus
                .rgb_to_cct_duv(rgb)
                .unwrap_or_else(|e| panic!("{id} {t}: {e}"));
            let dt = (back.cct_k - t).abs();
            if t < 4000.0 {
                worst_bb = (worst_bb.0.max(dt), worst_bb.1.max(back.duv.abs()));
            } else {
                worst_dy = (worst_dy.0.max(dt), worst_dy.1.max(back.duv.abs()));
            }
            // A table point reads back exactly (the readout inverts the interpolation on its chord)
            // up to 16000 K. Above, the three-decimal staircase of the D series retraces itself and
            // two temperatures can share a camera RGB (a chord is 10 K wide).
            let limit = if t < 16000.0 { 1e-6 } else { 10.5 };
            assert!(dt <= limit, "{id}: {t} K came back as {} K", back.cct_k);
            assert!(back.duv.abs() < 1e-9, "{id}: {t} K has Duv {}", back.duv);
        }
    }
    eprintln!(
        "locus round trip over 11 profiles x 300 temperatures: blackbody max {:.3e} K / {:.3e} Duv, daylight max {:.3e} K / {:.3e} Duv",
        worst_bb.0, worst_bb.1, worst_dy.0, worst_dy.1
    );
}

#[test]
fn duv_sign_is_positive_towards_magenta_on_the_real_locus() {
    let (locus, _) = tables(PROFILE_ID);
    let mut rng = Rng(0x5EED_0002);
    for _ in 0..400 {
        // The whole range, including the staircase above 16000 K where some chords run backwards.
        let t = 1500.0 + rng.next_f64() * 23500.0;
        // The two branches meet at 4000 K but do not join smoothly: CIE daylight sits about 0.03 log-chroma
        // units on the green side of the blackbody locus there. A tinted light near 4000 K can therefore
        // be closer to the other branch (the documented smaller-distance rule: a 1.5 % green excess at
        // 3900 K reads as 4000 K with Duv +0.001), so the sign check uses tiny tints in that zone and
        // larger ones elsewhere.
        let g = if (3000.0..5500.0).contains(&t) {
            1.0 + 0.002 + 0.002 * rng.next_f64()
        } else {
            1.0 + 0.02 + 0.08 * rng.next_f64()
        };
        let mut rgb = locus.rgb_at(t);
        rgb[1] = 1.0 / g; // less green: towards magenta
        assert!(locus.rgb_to_cct_duv(rgb).unwrap().duv > 0.0, "{t} K");
        rgb[1] = g; // more green: away from magenta
        assert!(locus.rgb_to_cct_duv(rgb).unwrap().duv < 0.0, "{t} K");
    }
}

#[test]
fn readout_rejects_non_positive_channels() {
    let (locus, _) = tables(PROFILE_ID);
    assert!(matches!(
        locus.rgb_to_cct_duv([0.5, 0.0, 0.5]),
        Err(SpectralError::InvalidRgb(_))
    ));
    assert!(matches!(
        locus.rgb_to_cct_duv([-0.5, 1.0, 0.5]),
        Err(SpectralError::InvalidRgb(_))
    ));
}

#[test]
fn scene_white_follows_the_c3_rules_on_the_real_profile() {
    let (_, white) = tables(PROFILE_ID);
    // Daylight and tungsten classes: the blackbody/daylight model at the scene CCT.
    for t in [2300.0, 3200.0, 3999.0, 4000.0, 5500.0, 6504.0, 9000.0] {
        for class in [LightClass::Daylight, LightClass::Tungsten] {
            let w = white.scene_white(t, &ClassProbabilities::only(class));
            let expect = white.locus_white(1e6 / (1e6 / t));
            assert!(max_abs_diff(w, expect) < 1e-15, "{t} K {class:?}");
        }
    }
    // The model switches tables at 4000 K, and the two agree closely there (CIE D4000 vs blackbody).
    let below = white.locus_white(3999.99);
    let above = white.locus_white(4000.0);
    assert!(max_abs_diff(below, above) < 0.05, "{below:?} vs {above:?}");
    // LED and fluorescent classes use their reference lights, so they differ from the model.
    let model = white.scene_white(4100.0, &ClassProbabilities::only(LightClass::Daylight));
    for class in [LightClass::Fluorescent, LightClass::Led] {
        let w = white.scene_white(4100.0, &ClassProbabilities::only(class));
        assert_eq!(w[1], 1.0);
        assert!(w[0] > 0.0 && w[2] > 0.0);
        assert!(
            max_abs_diff(w, model) > 1e-4,
            "{class:?} should not equal the model"
        );
    }
    // A half-and-half mix is the average.
    let half = ClassProbabilities::new(1.0, 0.0, 0.0, 1.0).unwrap();
    let mix = white.scene_white(5000.0, &half);
    let a = white.scene_white(5000.0, &ClassProbabilities::only(LightClass::Daylight));
    let b = white.scene_white(5000.0, &ClassProbabilities::only(LightClass::Led));
    assert!(max_abs_diff(mix, [0.5 * (a[0] + b[0]), 1.0, 0.5 * (a[2] + b[2])]) < 1e-15);
}

#[test]
fn curve_files_load_and_match_their_provenance() {
    let root = repo_root().join("data");
    let mut n = 0;
    for entry in std::fs::read_dir(root.join("cameras/sony")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let curves =
            SpectralCurves::from_path(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(curves.manufacturer(), "Sony");
        assert_eq!(curves.len(), 81, "{}", path.display());
        assert_eq!(curves.wavelengths_nm().first(), Some(&380.0));
        assert_eq!(curves.wavelengths_nm().last(), Some(&780.0));
        assert!(
            curves
                .wavelengths_nm()
                .windows(2)
                .all(|w| w[1] - w[0] == 5.0)
        );
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        assert_eq!(stem, format!("Sony_{}_380_780_5", curves.model()));
        // Clipping the measurement noise and normalising gives peak 1 on every channel.
        let norm = curves.clipped_at_zero().peak_normalized().unwrap();
        for ch in 0..3 {
            let peak = norm.samples().iter().fold(0.0_f64, |m, s| m.max(s[ch]));
            assert_eq!(peak, 1.0);
            assert!(norm.samples().iter().all(|s| s[ch] >= 0.0));
        }
        n += 1;
    }
    assert_eq!(n, 11);
    let alexa =
        SpectralCurves::from_path(root.join("alexa/alexa_alev3_estimate_380_780_5.json")).unwrap();
    assert_eq!(alexa.manufacturer(), "ARRI");
    assert_eq!(alexa.len(), 81);
    let alexa = alexa.peak_normalized().unwrap();
    assert!(
        alexa
            .samples()
            .iter()
            .all(|s| s.iter().all(|v| v.is_finite()))
    );
}

#[test]
fn extreme_inputs_never_panic_and_stay_well_defined() {
    let (locus, white) = tables(PROFILE_ID);
    let specials = [
        0.0,
        -0.0,
        1e-308,
        1e-300,
        0.5,
        1.0,
        1500.0,
        4000.0,
        25000.0,
        1e300,
        f64::MAX,
        f64::MIN_POSITIVE,
        -1.0,
        -1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let probs = ClassProbabilities::new(0.25, 0.25, 0.25, 0.25).unwrap();
    for &t in &specials {
        // Grids clamp (a NaN clamps to the start) and never return a non-finite table value.
        for v in locus
            .chroma_at(t)
            .into_iter()
            .chain(locus.xy_at(t))
            .chain(locus.uv_at(t))
            .chain([locus.duv_scale_at(t)])
        {
            assert!(v.is_finite(), "locus at {t}: {v}");
        }
        for v in white
            .locus_white(t)
            .into_iter()
            .chain(white.scene_white(t, &probs))
            .chain(locus.rgb_at(t))
        {
            assert!(v.is_finite() && v > 0.0, "white at {t}: {v}");
        }
        for &duv in &specials {
            // (CCT, Duv) to RGB may overflow for absurd Duv but must not panic.
            let _ = locus.cct_duv_to_rgb(t, duv);
        }
        for class in LightClass::ALL {
            // `t` doubles as a mired coordinate: clamped at the ends, NaN gives the first entry.
            let w = white.class_white(class, t);
            assert!(
                w.iter().all(|v| v.is_finite() && *v > 0.0),
                "{class:?} at {t} mired: {w:?}"
            );
        }
        for &g in &specials {
            // Camera RGB: either a valid readout or InvalidRgb.
            if let Ok(r) = locus.rgb_to_cct_duv([t, g, 1.0]) {
                assert!(r.cct_k.is_finite() && r.duv.is_finite(), "{t} {g}: {r:?}");
            }
        }
    }
}

fn profile_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(repo_root().join("data/cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    ids
}

#[test]
fn duv_has_the_right_size_and_sign_on_every_profile() {
    // An independent construction, not the reference file: move a locus point by a known log-chroma
    // distance along the unit normal that points towards magenta (the tangent from a central difference
    // over +-30 K, rotated by -90 degrees) and read it back. Duv must be that distance times the Duv
    // scale (CIE uv per unit of log-chroma), positive towards magenta and negative towards green.
    let mut worst_ratio = 0.0_f64;
    let mut worst_cct = 0.0_f64;
    for id in profile_ids() {
        let (locus, _) = tables(&id);
        // Every temperature is at least 30 K from the 4000 K junction, so one branch serves all three
        // points of the difference.
        for t in [
            1600.0, 2000.0, 2500.0, 3200.0, 3800.0, 4500.0, 5600.0, 6500.0, 8000.0, 10000.0,
            11000.0,
        ] {
            let q = locus.chroma_at(t);
            let (lo, hi) = (locus.chroma_at(t - 30.0), locus.chroma_at(t + 30.0));
            let tangent = [hi[0] - lo[0], hi[1] - lo[1]];
            let len = tangent[0].hypot(tangent[1]);
            let normal = [tangent[1] / len, -tangent[0] / len];
            assert!(
                normal[0] > 0.0 && normal[1] > 0.0,
                "{id} {t} K: the normal does not point towards magenta ({normal:?})"
            );
            for delta in [0.03, 0.005, -0.005, -0.03] {
                let p = [q[0] + delta * normal[0], q[1] + delta * normal[1]];
                let rgb = [p[0].exp(), 1.0, p[1].exp()];
                let got = locus.rgb_to_cct_duv(rgb).unwrap();
                let expected = delta * locus.duv_scale_at(t);
                let ratio = got.duv / expected;
                worst_ratio = worst_ratio.max((ratio - 1.0).abs());
                worst_cct = worst_cct.max((got.cct_k / t - 1.0).abs());
                assert!(
                    (ratio - 1.0).abs() < 0.01,
                    "{id} {t} K, {delta}: Duv {} against {expected} (ratio {ratio})",
                    got.duv
                );
                assert!(
                    (got.cct_k / t - 1.0).abs() < 0.01,
                    "{id} {t} K, {delta}: CCT {}",
                    got.cct_k
                );
            }
        }
    }
    eprintln!(
        "Duv size on 11 profiles, 11 temperatures, 4 tints: worst |ratio - 1| {worst_ratio:.2e}, worst |CCT/T - 1| {worst_cct:.2e}"
    );
}

#[test]
fn class_lists_return_their_own_entry_at_its_mired_on_every_profile() {
    for id in profile_ids() {
        let (_, white) = tables(&id);
        for (class, list) in [
            (LightClass::Fluorescent, &white.fluorescent),
            (LightClass::Led, &white.led),
        ] {
            assert!(list.len() >= 2, "{id} {class:?}");
            for e in list {
                assert_eq!(
                    white.class_white(class, e.mired),
                    e.w,
                    "{id} {class:?} {}",
                    e.light
                );
            }
            // Halfway (in mired) between two entries is the average of the two: linear in mired.
            for pair in list.windows(2) {
                let m = 0.5 * (pair[0].mired + pair[1].mired);
                let got = white.class_white(class, m);
                let expect: [f64; 3] = std::array::from_fn(|k| 0.5 * (pair[0].w[k] + pair[1].w[k]));
                assert!(
                    max_abs_diff(got, expect) < 1e-12,
                    "{id} {class:?} {}",
                    pair[0].light
                );
            }
        }
    }
}

#[test]
fn the_4000_k_junction_is_documented_behaviour() {
    // docs/03 "The 4000 K junction": the blackbody and daylight branches meet at 4000 K but are not
    // collinear (CIE D4000 sits about 0.04 log-chroma units on the green side of the Planckian locus
    // there). The readout takes the nearer branch, so CCT and Duv are continuous within a branch and
    // not across 4000 K. This pins the documented numbers so a change to either rule shows up here and
    // in the docs, and so the illuminant estimator's author finds the limit in a test as well.
    let (locus, _) = tables(PROFILE_ID);
    let bb_end = locus.blackbody.values()[locus.blackbody.len() - 1];
    let dy_start = locus.daylight.values()[0];
    let gap = (bb_end[0] - dy_start[0]).hypot(bb_end[1] - dy_start[1]);
    assert!(
        (0.03..0.06).contains(&gap),
        "gap between the branches at 4000 K: {gap}"
    );

    // Lights on either branch read back as themselves right up to the junction.
    for t in [3990.0, 3999.0, 4000.0, 4010.0] {
        let back = locus.rgb_to_cct_duv(locus.rgb_at(t)).unwrap();
        assert!((back.cct_k - t).abs() < 1e-6, "{t}: {back:?}");
        assert!(back.duv.abs() < 1e-9, "{t}: {back:?}");
    }

    // A 1.5 % green excess on a 3894 K blackbody light is nearer to the daylight branch's first chord:
    // it reads as 4000 K with a small *positive* (magenta) Duv, where a smooth locus would say about
    // 3890 K and a negative Duv. The same tint at 3800 K stays on the blackbody branch.
    let green = |t: f64| {
        let mut rgb = locus.rgb_at(t);
        rgb[1] = 1.015;
        locus.rgb_to_cct_duv(rgb).unwrap()
    };
    let at_junction = green(3894.0);
    assert!((at_junction.cct_k - 4000.0).abs() < 1.0, "{at_junction:?}");
    assert!(
        (0.0005..0.0015).contains(&at_junction.duv),
        "{at_junction:?}"
    );
    let away = green(3800.0);
    assert!(away.cct_k < 3800.0 && away.duv < -0.001, "{away:?}");

    // The same limit seen from above: a magenta tint of 0.03 log-chroma units (Duv about +0.0025) on a
    // 4100 K daylight light. The foot of the perpendicular would lie below the daylight table's first
    // entry, so the readout clamps to 4000 K and projects on the first chord alone: the size comes out
    // at a fraction of the tint and, on this body, with the wrong sign. The same tint at 4600 K is read
    // correctly.
    let tinted = |t: f64, delta: f64| {
        let q = locus.chroma_at(t);
        let (lo, hi) = (locus.chroma_at(t - 20.0), locus.chroma_at(t + 20.0));
        let tangent = [hi[0] - lo[0], hi[1] - lo[1]];
        let len = tangent[0].hypot(tangent[1]);
        let p = [
            q[0] + delta * tangent[1] / len,
            q[1] - delta * tangent[0] / len,
        ];
        let read = locus.rgb_to_cct_duv([p[0].exp(), 1.0, p[1].exp()]).unwrap();
        (read, delta * locus.duv_scale_at(t))
    };
    let (read, expected) = tinted(4100.0, 0.03);
    assert!((read.cct_k - 4000.0).abs() < 1.0, "{read:?}");
    assert!(read.duv < 0.25 * expected, "{read:?} against {expected}");
    let (read, expected) = tinted(4600.0, 0.03);
    assert!((read.cct_k - 4600.0).abs() < 10.0, "{read:?}");
    assert!((read.duv / expected - 1.0).abs() < 0.01, "{read:?}");
}
