//! The committed profile (`data/dng/profile.json`) and the fit, against Adobe's renderer.
//!
//! The file tests need nothing. The renderer tests need `dng_validate` (`tools/dng/build_dng_validate.sh`; they say so and
//! pass without it unless `YM_REQUIRE_EXTERNAL_TOOLS` is set). The independent-reader test also needs python3 with numpy,
//! tifffile, Pillow and rawpy, and exiftool.

use std::path::PathBuf;
use std::process::Command;

use dngfit::adobe::{Adobe, write_dng_image, write_dng_image_with};
use dngfit::fit::{self, FitConfig};
use dngfit::profile_file::{self, ProfileFile};
use golden::ymrun::workspace_root;
use ym_export::dng::DngProfile;

fn profile_path() -> PathBuf {
    workspace_root().join("data/dng/profile.json")
}

fn adobe() -> Option<Adobe> {
    match Adobe::new(None) {
        Ok(a) => Some(a),
        Err(e) => {
            assert!(
                std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
                "dng_validate is required: {e:#}"
            );
            eprintln!("SKIPPED: {e:#}");
            None
        }
    }
}

#[test]
fn the_committed_profile_loads_and_is_in_the_form_the_tool_writes() {
    let path = profile_path();
    let file = profile_file::load(&path).unwrap();
    assert_eq!(file.name, "Yellowmoon");
    assert_eq!(
        file.fit_offset_ev, -4.0,
        "the curve covers scene values up to 16"
    );
    let [h, s, v] = file.look_table.dims;
    assert!(
        h <= 36 && s <= 32 && (2..=16).contains(&v) && file.look_table.encoding == 1,
        "{:?}",
        file.look_table.dims
    );
    // the curve: from (0, 0) to (1, T(1)), nowhere decreasing, and 18 % grey (u = 0.18 / 16) in the shadows
    let c = &file.tone_curve;
    assert_eq!(c[0], [0.0, 0.0]);
    assert_eq!(c.last().unwrap()[0], 1.0);
    assert!(c.windows(2).all(|w| w[1][1] >= w[0][1]));
    let top = c.last().unwrap()[1];
    assert!(
        (f64::from(top) - fit::neutral_target_linear(16.0)).abs() < 1e-3,
        "{top}"
    );
    assert!(c.len() <= 128, "{} points", c.len());
    // zero-saturation entries keep the value scale at 1 (the DNG specification)
    for (k, e) in file.look_table.data.chunks_exact(3).enumerate() {
        if k % s as usize == 0 {
            assert_eq!(e[2], 1.0, "entry {k}");
        }
    }
    // the file is exactly what the tool writes for it (nobody edited the numbers by hand)
    let text = std::fs::read_to_string(&path).unwrap();
    let canonical = file.to_json_text();
    if canonical != text {
        let at = canonical
            .bytes()
            .zip(text.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(canonical.len().min(text.len()));
        let from = at.saturating_sub(40);
        panic!(
            "data/dng/profile.json is not in the form `dngfit fit` writes; first difference at byte {at}:\n  file:      {:?}\n  canonical: {:?}",
            &text[from..(at + 40).min(text.len())],
            &canonical[from..(at + 40).min(canonical.len())]
        );
    }
    // the fit metrics it records are the ones the tool measured
    assert!(file.fit["delta_e_2000_against_aces2_via_dng_validate"].is_array());
    assert!(file.fit["dng_validate"].as_str().unwrap().contains("1.7"));
}

#[test]
fn the_committed_profile_matches_aces_2_through_adobes_renderer() {
    let Some(adobe) = adobe() else { return };
    let file = profile_file::load(&profile_path()).unwrap();
    let profile = file.to_profile();
    let sets = fit::test_sets(None);
    let results = fit::evaluate_sets(&adobe, &profile, file.fit_offset_ev, &sets).unwrap();
    let by = |prefix: &str| {
        results
            .iter()
            .find(|r| r.name.starts_with(prefix))
            .unwrap()
            .stats
    };
    let neutral = by("neutral ramp");
    assert!(neutral.max < 0.1 && neutral.median < 0.02, "{neutral:?}");
    let sweep = by("hue and chroma sweep");
    assert!(
        sweep.median < 0.15 && sweep.p95 < 1.5 && sweep.max < 6.0,
        "{sweep:?}"
    );
    let random = by("random photograph-like");
    assert!(
        random.median < 0.15 && random.p95 < 1.2 && random.p99 < 2.5,
        "{random:?}"
    );

    // negative control: the same curve without the look table is clearly worse
    let tone_only = DngProfile {
        look_table: ym_export::dng::LookTable::identity(),
        ..profile
    };
    let worse = fit::evaluate_sets(&adobe, &tone_only, file.fit_offset_ev, &sets).unwrap();
    let worse_random = worse
        .iter()
        .find(|r| r.name.starts_with("random photograph-like"))
        .unwrap()
        .stats;
    assert!(
        worse_random.median > 4.0 * random.median,
        "{worse_random:?} against {random:?}"
    );
}

#[test]
fn a_dng_with_the_profile_passes_dng_validate_without_a_complaint() {
    let Some(adobe) = adobe() else { return };
    let file = profile_file::load(&profile_path()).unwrap();
    // a gradient scene that reaches the white level
    let (w, h) = (160_usize, 96_usize);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / (w - 1) as f32;
            let u = y as f32 / (h - 1) as f32;
            px.extend_from_slice(&[
                0.004 * 4096_f32.powf(t),
                0.004 * 4096_f32.powf(u),
                0.2 + 16.0 * t * u,
            ]);
        }
    }
    let (bytes, report) =
        write_dng_image(w, h, &px, &file.to_profile(), file.fit_offset_ev).unwrap();
    assert!(
        report.baseline_exposure_ev >= -1e-9,
        "BaselineExposure {}",
        report.baseline_exposure_ev
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("profile.dng");
    std::fs::write(&path, bytes).unwrap();
    let r = adobe.render_file_verbose(&path).unwrap();
    assert!(r.problems().is_empty(), "{:?}\n{}", r.problems(), r.log);
    assert_eq!((r.width, r.height), (w, h));
}

#[test]
fn independent_readers_accept_a_dng_with_the_fitted_profile() {
    let python_ok = Command::new("python3")
        .args(["-I", "-c", "import numpy, tifffile, PIL, rawpy"])
        .status()
        .is_ok_and(|s| s.success());
    let exiftool_ok = Command::new("exiftool")
        .arg("-ver")
        .output()
        .is_ok_and(|o| o.status.success());
    if !(python_ok && exiftool_ok) {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "python3 with numpy, tifffile, Pillow, rawpy and exiftool are required"
        );
        eprintln!("SKIPPED: needs python3 (numpy, tifffile, Pillow, rawpy) and exiftool");
        return;
    }
    let file = profile_file::load(&profile_path()).unwrap();
    // many pixels above 16, so that x_white is the photo's own 99.99th percentile in the script's rule too
    let (w, h) = (96_usize, 64_usize);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / (w - 1) as f32;
            let big = if y < 8 { 20.0 } else { 0.0 };
            px.extend_from_slice(&[
                0.01 + 3.0 * t + big,
                0.02 + y as f32 / 40.0,
                0.05 + 2.0 * t * t + big,
            ]);
        }
    }
    let mut meta = ym_export::Metadata::default();
    meta.exif.make = Some("SONY".into());
    meta.exif.model = Some("ILCE-1".into());
    meta.xmp.app_version = "0.0.1".into();
    meta.xmp.ym.pipeline_version = Some("1.0.0".into());
    let (bytes, report) =
        write_dng_image_with(w, h, &px, &file.to_profile(), file.fit_offset_ev, &meta).unwrap();
    assert!(report.x_white >= 20.0);
    let dir = tempfile::tempdir().unwrap();
    let dng = dir.path().join("p.dng");
    let raw = dir.path().join("p.f32");
    std::fs::write(&dng, bytes).unwrap();
    std::fs::write(
        &raw,
        px.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    let curve: Vec<String> = file
        .tone_curve
        .iter()
        .flatten()
        .map(|v| format!("{v}"))
        .collect();
    let dims: Vec<String> = file
        .look_table
        .dims
        .iter()
        .map(ToString::to_string)
        .collect();
    let script = workspace_root().join("tools/dng/check_dng.py");
    let out = Command::new("python3")
        .arg("-I")
        .arg(&script)
        .arg(&dng)
        .arg("--raw-f32")
        .arg(&raw)
        .args(["--width", &w.to_string(), "--height", &h.to_string()])
        .args(["--baseline-offset-ev", &file.fit_offset_ev.to_string()])
        .args(["--tone-curve", &curve.join(" ")])
        .args(["--look-dims", &dims.join(" ")])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "check_dng.py failed:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn the_fit_is_repeatable() {
    let Some(adobe) = adobe() else { return };
    let cfg = FitConfig {
        tone_points: 32,
        tone_iterations: 3,
        dims: [12, 6, 8],
        ..FitConfig::default()
    };
    let run = || {
        let (mut curve, tone, _) = fit::fit_tone(&adobe, &cfg).unwrap();
        fit::round_curve(&mut curve);
        let (mut table, _) = fit::fit_look(&cfg, &tone, 2, 1e-4, 30);
        fit::round_table(&mut table);
        ProfileFile::from_profile(
            &DngProfile {
                tone_curve: curve,
                look_table: table,
            },
            cfg.offset_ev(),
            "t",
            serde_json::Value::Null,
        )
        .to_json_text()
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b);
    let parsed: ProfileFile = serde_json::from_str(&a).unwrap();
    parsed.validate().unwrap();
}
