//! M1-02: the shipped `profile.json` tables against the spectra they were computed from.
//!
//! Needs the git-ignored fixtures of `python3 -m training.sensor_map.export_fixtures` (CIE light SPDs are
//! CC BY-SA data and are not committed) and skips with a message when they are missing. CI runs the
//! exporter first and sets `YM_REQUIRE_FIXTURES=1`.
//!
//! For the 45 reference lights this renders the lights through the shipped ALEXA estimate and the Sony
//! curves in Rust and compares with the tables:
//!
//! * `alexa_white.fluorescent` / `.led`: `w_A` of each reference light (class lists), and `w_A` of the
//!   blackbody and CIE daylight reference lights against the 10 K tables;
//! * `locus`: the camera log-chroma of the same lights against the locus tables;
//! * the `mired` of every `sensor_map.classes` entry against `1e6 / CCT` of its light.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::sensor_map::SensorMapTables;
use ym_color::spectral::fixtures::{Fixtures, Light};
use ym_color::spectral::{AlexaWhite, LightClass, Locus, SpectralCurves};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Profile {
    curves: SpectralCurves,
    locus: Locus,
    white: AlexaWhite,
    maps: SensorMapTables,
}

fn load_profile(id: &str) -> Profile {
    let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let curves_path = repo_root()
        .join("data/cameras")
        .join(v["spectral_curves"].as_str().unwrap());
    Profile {
        // The locus is the camera white in the raw scale (docs/02 C1): the curves as published, with the
        // measurement noise below zero clipped and NOT normalised per channel. (The maps and `w_A` come from
        // peak-normalised curves, which only works because every response there is divided by the white's.)
        curves: SpectralCurves::from_path(curves_path)
            .unwrap()
            .clipped_at_zero(),
        locus: serde_json::from_value(v["locus"].clone()).unwrap(),
        white: serde_json::from_value(v["alexa_white"].clone()).unwrap(),
        maps: serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap(),
    }
}

fn alexa() -> SpectralCurves {
    SpectralCurves::from_path(repo_root().join("data/alexa/alexa_alev3_estimate_380_780_5.json"))
        .unwrap()
        .peak_normalized()
        .unwrap()
}

/// `[r, 1, b]`: the response to a light, normalised to green.
fn white_of(curves: &SpectralCurves, light: &Light) -> [f64; 3] {
    let r = curves.response(&light.spd).unwrap();
    [r[0] / r[1], 1.0, r[2] / r[1]]
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn log_chroma(curves: &SpectralCurves, light: &Light) -> [f64; 2] {
    let w = white_of(curves, light);
    [w[0].ln(), w[2].ln()]
}

#[test]
fn class_lists_and_locus_tables_match_the_spectra() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let s_a = alexa();
    let ids = [
        "sony-ilce-7rm4",
        "sony-ilce-7sm2",
        "sony-dsc-rx100m4",
        "sony-ilce-9",
        "sony-ilce-6400",
        "sony-ilce-7m3",
        "sony-ilce-7m4",
        "sony-ilce-7cm2",
        "sony-ilce-7rm2",
        "sony-ilce-7rm3",
        "sony-ilce-7sm3",
    ];
    let mut worst = [0.0_f64; 4]; // class w_A, bb/daylight w_A, locus chroma, mired
    for id in ids {
        let p = load_profile(id);
        for light in fx.lights_with_prefix("ref:") {
            let class = light.name.split(':').nth(1).unwrap();
            let t = light.cct_k;
            let w = white_of(&s_a, light);
            match class {
                "fluorescent" | "led" => {
                    let list = if class == "led" {
                        &p.white.led
                    } else {
                        &p.white.fluorescent
                    };
                    let name = light.name.rsplit(':').next().unwrap();
                    let entry = list
                        .iter()
                        .find(|e| e.light == name)
                        .unwrap_or_else(|| panic!("{id}: {name} missing"));
                    worst[0] = worst[0].max(max_diff(&entry.w, &w));
                    assert!(
                        (entry.mired - 1e6 / t).abs() < 1e-6,
                        "{id} {name}: mired {} vs {}",
                        entry.mired,
                        1e6 / t
                    );
                }
                "tungsten" | "daylight" => {
                    // These lights sit on the 10 K tables' grid (multiples of 250 K / 500 K).
                    let table = if class == "tungsten" {
                        p.white.blackbody_white(t)
                    } else {
                        p.white.daylight_white(t)
                    };
                    worst[1] = worst[1].max(max_diff(&table, &w));
                    let chroma = log_chroma(&p.curves, light);
                    let tabled = if class == "tungsten" {
                        p.locus.blackbody.lerp_mired(t)
                    } else {
                        p.locus.daylight.lerp_mired(t)
                    };
                    worst[2] = worst[2].max(max_diff(&tabled, &chroma));
                }
                other => panic!("unexpected class {other}"),
            }
            let class_enum = match class {
                "fluorescent" => LightClass::Fluorescent,
                "led" => LightClass::Led,
                "tungsten" => LightClass::Tungsten,
                _ => LightClass::Daylight,
            };
            let name = light.name.rsplit(':').next().unwrap();
            let entry = p
                .maps
                .class(class_enum)
                .iter()
                .find(|e| e.light == name)
                .unwrap_or_else(|| panic!("{id}: map {name} missing"));
            worst[3] = worst[3].max((entry.mired - 1e6 / t).abs());
        }
    }
    eprintln!(
        "tables vs spectra (11 profiles x 45 reference lights): class w_A {:.2e}, blackbody/daylight w_A {:.2e}, locus log-chroma {:.2e}, entry mired {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    // Tables are rounded to 9 decimals; the on-grid tolerance of the reference file is 1e-6.
    assert!(
        worst[0] < 1e-6 && worst[1] < 1e-6 && worst[2] < 1e-6,
        "{worst:?}"
    );
    assert!(worst[3] < 1e-6, "{worst:?}");
}

#[test]
fn the_exported_cmf_is_the_cie_1931_observer() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    assert_eq!(fx.wavelengths_nm.first(), Some(&380.0));
    assert_eq!(fx.wavelengths_nm.last(), Some(&780.0));
    // Y integrates to 106.86 over 380..780 nm at 5 nm for the CIE 1931 2 degree observer.
    let sum_y: f64 = fx.cmf.iter().map(|c| c[1]).sum();
    assert!((sum_y - 21.371).abs() < 0.01, "sum of y-bar is {sum_y}");
}
