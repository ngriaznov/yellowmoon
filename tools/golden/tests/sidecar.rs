//! The sidecars `training/synth` writes, read by the Rust types: the two sides agree on the format, and the ground
//! truth in them obeys its own rules. The real-data test runs when `testdata/synthetic/scenes` exists (made by
//! `tools/fetch/synthetic.sh`); without it the test says so and passes, unless `YM_REQUIRE_SYNTHETIC=1`.

use std::path::PathBuf;

use golden::scene::{Scene, list_sidecars};
use serde_json::json;

fn scenes_dir() -> PathBuf {
    let root = std::env::var_os("YM_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata"));
    root.join("synthetic/scenes")
}

#[test]
fn the_truth_types_read_a_minimal_object() {
    let rgb = json!([0.1, 0.2, 0.3]);
    let reference = json!({ "native_balanced": rgb, "awg3": rgb, "awg4": rgb, "logc4": rgb });
    let truth = json!({
        "light": { "token": "D6500", "name": "CIE D 6500 K", "class": "daylight", "class_probabilities": { "daylight": 1.0 },
                   "cct_k": 6500.0, "w_true_camera_rgb": [0.34, 1.0, 0.73] },
        "as_shot": { "gains_true": [2.9, 1.0, 1.4], "log_gain_error_rb": [0.01, -0.02] },
        "noise": { "iso": 400, "bits": 14, "a_dn_per_electron": 2.5, "b_dn2": 20.0, "a_normalised": 1.6e-4,
                   "b_normalised": 8e-8, "gain_regime": "high gain" },
        "probes": [{ "name": "neutral-0.18", "group": "neutral", "role": "grey_reference", "rect": [1, 2, 3, 4],
                     "interior": [1, 2, 1, 2], "irradiance": 1.0, "camera_dn_above_black": [1.0, 2.0, 3.0],
                     "clipped_channels": [false, false, false], "alexa": reference, "sony_through_arri": reference }],
    });
    let sidecar: golden::scene::Sidecar = serde_json::from_value(json!({
        "id": "x", "tier": "synthetic",
        "mosaic": { "file": "x.bin", "width": 2, "height": 2, "cfa": "RGGB", "bits": 14, "black": 512, "white": 16383, "sha256": "" },
        "writer": {
            "make": "SONY", "model": "ILCE-7RM4", "model_id": 375, "lens_model": null, "focal_length_mm": 35.0, "iso": 400,
            "exposure_time": [1, 500], "f_number": [80, 10], "exposure_comp_ev": 0.0, "capture_time": "2026:10:12 12:00:00",
            "sr2": { "black": [512, 512, 512, 512], "white": [16383, 16383, 16383], "wb_rggb": [3128, 1024, 1024, 1369] },
            "preview": { "file": "x.jpg", "width": 2, "height": 2 }, "variants": [], "software": "s"
        },
        "truth": truth,
    }))
    .unwrap();
    let t = sidecar.truth().unwrap();
    assert_eq!(t.light.class, "daylight");
    assert_eq!(t.probes[0].role, "grey_reference");
    assert_eq!(t.probes[0].alexa.logc4, [0.1, 0.2, 0.3]);
    assert_eq!(t.noise.gain_regime, "high gain");
    // a sidecar without truth parses, and asking for it is an error, not a panic
    let mut bare = sidecar.clone();
    bare.truth = serde_json::Value::Null;
    assert!(bare.truth().is_err());
}

#[test]
fn every_generated_sidecar_parses_and_its_truth_obeys_its_rules() {
    let dir = scenes_dir();
    let Ok(sidecars) = list_sidecars(&dir) else {
        assert!(
            std::env::var_os("YM_REQUIRE_SYNTHETIC").is_none(),
            "no synthetic scenes in {} (run tools/fetch/synthetic.sh)",
            dir.display()
        );
        eprintln!("skipped: no scenes in {}", dir.display());
        return;
    };
    assert!(!sidecars.is_empty() || std::env::var_os("YM_REQUIRE_SYNTHETIC").is_none());
    for path in sidecars {
        let scene = Scene::load(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        let id = &scene.sidecar.id;
        let truth = scene
            .sidecar
            .truth()
            .unwrap_or_else(|e| panic!("{id}: {e:#}"));
        let (w, h) = (scene.width(), scene.height());
        let m = &scene.sidecar.mosaic;
        assert_eq!(truth.noise.bits, m.bits, "{id}");
        assert_eq!(truth.noise.iso, scene.sidecar.writer.iso, "{id}");
        let span = f64::from(m.white - m.black);
        assert!(
            (truth.noise.a_normalised - truth.noise.a_dn_per_electron / span).abs()
                <= 1e-8 * truth.noise.a_normalised.abs().max(1e-9) + 1e-12,
            "{id}"
        );
        assert_eq!(truth.light.w_true_camera_rgb[1], 1.0, "{id}");
        assert!(
            ["daylight", "tungsten", "fluorescent", "led", "mixed"]
                .contains(&truth.light.class.as_str()),
            "{id}"
        );
        let total: f64 = truth.light.class_probabilities.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "{id}: class probabilities sum to {total}"
        );

        let mut names = std::collections::HashSet::new();
        let mut greys = 0;
        for p in &truth.probes {
            assert!(
                names.insert(p.name.clone()),
                "{id}: duplicate probe {}",
                p.name
            );
            let [x, y, pw, ph] = p.rect;
            let [ix, iy, iw, ih] = p.interior;
            assert!(
                x + pw <= w && y + ph <= h,
                "{id}: {} outside the image",
                p.name
            );
            assert!(
                ix >= x && iy >= y && ix + iw <= x + pw && iy + ih <= y + ph,
                "{id}: {} interior outside its rect",
                p.name
            );
            let rho = p
                .name
                .strip_prefix("neutral-")
                .and_then(|r| r.parse::<f64>().ok());
            if let Some(rho) = rho.filter(|_| truth.light.class != "mixed") {
                // neutral stays neutral through the ARRI matrix and the gamut steps, in both references (in the mixed
                // scene only the local white balance keeps a neutral patch neutral: `alexa_local_white`)
                for r in [&p.alexa, &p.sony_through_arri] {
                    for c in r.awg4 {
                        assert!((c - rho).abs() < 1e-7, "{id}: {} awg4 {:?}", p.name, r.awg4);
                    }
                }
            }
            if p.role == "grey_reference" {
                greys += 1;
                if truth.light.class != "mixed" {
                    assert!(
                        (p.alexa.logc4[1] - 0.2783958365).abs() < 1e-7,
                        "{id}: grey LogC4 {:?}",
                        p.alexa.logc4
                    );
                }
                // the metered grey is at 0.12 of the range, in the mosaic's own scale
                assert!(
                    (p.camera_dn_above_black[1] / span - 0.12).abs() < 1e-6,
                    "{id}"
                );
                // and the mosaic holds it: the mean of the two green sites of the interior (noise averages out)
                let [gx, gy, gw, gh] = p.interior;
                let (mut sum, mut n) = (0.0, 0u32);
                for yy in gy..gy + gh {
                    for xx in gx..gx + gw {
                        if scene.cfa.color_at(xx as usize, yy as usize)
                            == ym_raw::layout::CfaColor::Green
                        {
                            sum += f64::from(scene.mosaic[(yy * w + xx) as usize])
                                - f64::from(m.black);
                            n += 1;
                        }
                    }
                }
                let mean = sum / f64::from(n);
                assert!(
                    (mean / p.camera_dn_above_black[1] - 1.0).abs() < 0.05,
                    "{id}: grey mean {mean} vs {}",
                    p.camera_dn_above_black[1]
                );
            }
        }
        assert_eq!(greys, 1, "{id}: exactly one grey reference");
    }
}
