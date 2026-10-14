//! The session on the real ARW files in `testdata/real/files`, when they are there (local
//! evaluation only: the files are not ours to commit, and CI does not have them). Prints one line
//! per file; run with `cargo test -p ym-core --test real_files -- --nocapture`.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::{base_env, data_root};
use ym_core::auto::{IlluminantModels, IlluminantSource};
use ym_core::session::{OpenOptions, PhotoStatus, Session};

fn real_files() -> Vec<PathBuf> {
    let dir = data_root().parent().unwrap().join("testdata/real/files");
    let Ok(read) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = read
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw")))
        .collect();
    files.sort();
    files
}

fn run(label: &str, models: IlluminantModels, files: &[PathBuf]) -> Vec<(String, f64, f64, f64)> {
    let mut env = base_env();
    env.illuminants = std::sync::Arc::new(models);
    env.decode_threads = 2;
    let session = Session::new(env);
    let started = Instant::now();
    session.open(files, OpenOptions::default());
    assert!(
        session.wait_idle(Duration::from_secs(300)),
        "the pool did not finish"
    );
    eprintln!(
        "[{label}] {} files developed in {:.1} s",
        files.len(),
        started.elapsed().as_secs_f64()
    );
    let mut rows = Vec::new();
    for photo in session.state().photos {
        let name = photo.file_name.clone();
        assert_eq!(photo.status, PhotoStatus::Ready, "{name}");
        let view = session.view(photo.id).unwrap();
        let s = view.estimates.summary();
        let source = match &view.estimates.illuminant.source {
            IlluminantSource::Model { prior_used, .. } => format!("model(prior={prior_used})"),
            IlluminantSource::AsShotFallback => "as-shot".to_owned(),
            IlluminantSource::DefaultDaylight => "default".to_owned(),
        };
        eprintln!(
            "[{label}] {name:<22} {:<10} {:<14} ISO {:>5}  {:>5.0} K tint {:>5.1} conf {:.2} {}  EV {:>5.2} (camera gain {:>5.2}, guard {:.2}, key {:>5.2})  a {:.2e} b {:.1e} {:?}  sharpen {:.0}/{:.1} NR {:.0}",
            view.camera.matched.body,
            source,
            photo.iso,
            s.kelvin,
            s.tint,
            s.light_confidence,
            s.light_class.name(),
            s.exposure_ev,
            view.estimates.exposure.meter.ev_before_guard,
            view.estimates.exposure.meter.guard_reduction_ev,
            // What the old metering would have chosen (reported only, it moves nothing).
            (0.18 / view.estimates.exposure.meter.mean_luminance).log2(),
            s.noise.a,
            s.noise.b,
            view.estimates.noise.source,
            s.sharpen.amount,
            s.sharpen.radius_px,
            s.noise_reduction.luminance,
        );
        assert!(
            s.exposure_ev.is_finite() && s.exposure_ev.abs() <= 4.0,
            "{name}: EV {}",
            s.exposure_ev
        );
        assert!(
            (2000.0..=11000.0).contains(&s.kelvin),
            "{name}: {} K",
            s.kelvin
        );
        assert!(
            s.noise.a > 0.0 && s.noise.a < 1e-2,
            "{name}: a {}",
            s.noise.a
        );
        assert!(s.noise.b >= 0.0 && s.noise.b.is_finite());
        rows.push((name, s.kelvin, s.tint, s.exposure_ev));
    }
    rows
}

#[test]
fn the_real_files_develop_with_sensible_estimates() {
    let files = real_files();
    if files.is_empty() {
        eprintln!("no files in testdata/real/files; skipping");
        return;
    }
    // The as-shot fallback (no model in the data folder for these profiles).
    let fallback = run("as-shot", IlluminantModels::none(), &files);
    // The illuminant models of the data folder (every profile that has one).
    let models = IlluminantModels::load(&data_root(), &common::profiles());
    if models.is_empty() {
        eprintln!("no illuminant models in data/models/illum; only the as-shot fallback ran");
        return;
    }
    eprintln!(
        "{} illuminant models loaded, {} missing",
        models.len(),
        models.missing().len()
    );
    let with_model = run("model", models, &files);
    for ((name, k0, _, _), (_, k1, _, _)) in fallback.iter().zip(&with_model) {
        // Both read the same scene. The model uses the camera's white balance as its prior, so a
        // camera that got it roughly right cannot be far apart from it in kelvin terms.
        assert!(
            (k0 - k1).abs() < 2500.0,
            "{name}: as-shot {k0} K against the model's {k1} K"
        );
    }
}

/// Whether the environment demands the synthetic set (`YM_REQUIRE_SYNTHETIC=1`, the `synthetic-set`
/// CI job): a missing set is then a failure, not a skip. Set, not empty, not `0`: the one rule of
/// every test that reads the variable (`docs/13-testing.md`).
fn synthetic_required() -> bool {
    std::env::var_os("YM_REQUIRE_SYNTHETIC").is_some_and(|v| !v.is_empty() && v != "0")
}

/// The folder `tools/fetch/synthetic.sh` writes the ARW files to: `$YM_TESTDATA/synthetic/arw`, else
/// `<workspace root>/testdata/synthetic/arw`.
fn synthetic_arw_dir() -> PathBuf {
    let testdata = std::env::var_os("YM_TESTDATA")
        .filter(|v| !v.is_empty())
        .map_or_else(
            || data_root().parent().unwrap().join("testdata"),
            PathBuf::from,
        );
    testdata.join("synthetic/arw")
}

/// Says why the synthetic set cannot be used: a skip, or with `YM_REQUIRE_SYNTHETIC=1` a failure.
fn synthetic_missing(why: &str) {
    assert!(
        !synthetic_required(),
        "{why} and YM_REQUIRE_SYNTHETIC is set; run tools/fetch/synthetic.sh first"
    );
    eprintln!("{why}; skipping (YM_REQUIRE_SYNTHETIC=1 makes this a failure)");
}

#[test]
fn every_file_of_the_synthetic_variant_set_develops() {
    // `testdata/synthetic/arw/*.ARW` (tools/fetch/synthetic.sh): every raw variant of the decoders,
    // with gradient, noise, clipped and dark content. Dark and clipped frames are the extremes
    // the estimates must survive. (The files directly in `testdata/synthetic/` are leftovers of an
    // older generator and are not read.)
    let dir = synthetic_arw_dir();
    let Ok(read) = std::fs::read_dir(&dir) else {
        synthetic_missing(&format!("no {}", dir.display()));
        return;
    };
    let count = read
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("arw"))
        })
        .count();
    if count == 0 {
        synthetic_missing(&format!("no ARW files in {}", dir.display()));
        return;
    }
    let mut env = base_env();
    env.decode_threads = 2;
    let session = Session::new(env);
    let outcome = session.open(&[dir], OpenOptions::default());
    assert_eq!(outcome.added.len(), count);
    assert!(session.wait_idle(Duration::from_secs(600)));
    let (mut ready, mut skipped) = (0, 0);
    for photo in session.state().photos {
        match &photo.status {
            PhotoStatus::Ready => {
                ready += 1;
                let e = session.estimates(photo.id).unwrap();
                let s = e.summary();
                assert!(
                    s.exposure_ev.is_finite() && s.exposure_ev.abs() <= 4.0,
                    "{}",
                    photo.file_name
                );
                assert!(
                    (2000.0..=11000.0).contains(&s.kelvin),
                    "{}: {}",
                    photo.file_name,
                    s.kelvin
                );
                assert!(
                    s.noise.a > 0.0 && s.noise.a.is_finite(),
                    "{}",
                    photo.file_name
                );
                eprintln!(
                    "{:<46} EV {:>5.2} {:>5.0} K  a {:.1e}  {:?}",
                    photo.file_name, s.exposure_ev, s.kelvin, s.noise.a, e.noise.source
                );
            }
            PhotoStatus::Skipped { reason } => {
                skipped += 1;
                eprintln!("{:<46} SKIPPED: {reason}", photo.file_name);
            }
            other => panic!("{}: {other:?}", photo.file_name),
        }
    }
    eprintln!("{ready} ready, {skipped} skipped of {count}");
    assert_eq!(skipped, 0, "every variant decodes");
}

#[test]
fn old_metric_median_luminance_against_the_embedded_jpegs_of_the_real_files_informational() {
    // INFORMATIONAL: this is NOT the M1-08 gate. It computes the metric the gate used before
    // Mykyta's decision of 2026-10-13 (`docs/16`): the median luminance in linear light of our
    // render at `EV_auto` against the median of the camera's embedded JPEG. That metric mixes in
    // the camera's tone curve and the scene's key, so its errors are not all exposure errors
    // (nine real files under the metered rule and ACES 2 without the placement: median -0.28 EV,
    // `docs/02` Evaluation; since 2026-10-14 `EV_auto` is the camera's gain, +1.11 EV, and this
    // test reads 9 of 9 within +-0.5 EV on the nine files of `testdata/real/files`, errors -0.45
    // (DSLR-A300) to +0.41 (A7 II); `autoexp.py` G1 on the twelve local files of `docs/02` Gate
    // status: 12 of 12, median -0.02, std 0.27. The display placement does not touch this metric).
    // The gate is G3, `EV_auto` against the camera engine's own gain, measured by
    // `python3 -I tools/eval/autoexp.py testdata/real/files testdata/readme-photos/src --label real12 --gate`
    // (`tools/eval/README.md`, `docs/02` Auto exposure Gate status, `docs/13` Gates): 12 of 12
    // within +-0.5 EV, in-sample (the constant is those files' median gain). This test prints the
    // figures for comparison and asserts nothing about them but that the report is complete; it
    // never decides the gate.
    use ym_color::exposure::{
        GATE_REQUIRED_FRACTION, GATE_TOLERANCE_EV, GateEntry, PreviewRgb, evaluate_gate,
        exposure_error_ev, jpeg_median_luminance, preview_median_luminance,
    };
    let files = real_files();
    if files.is_empty() {
        eprintln!("no files in testdata/real/files; skipping");
        return;
    }
    let mut env = base_env();
    env.decode_threads = 2;
    let session = Session::new(env);
    session.open(&files, OpenOptions::default());
    assert!(session.wait_idle(Duration::from_secs(300)));
    let mut entries = Vec::new();
    for photo in session.state().photos {
        let view = session.view(photo.id).unwrap();
        let raw = session.raw_image(photo.id).unwrap();
        let preview = ym_core::auto::build_preview(&raw).unwrap();
        // The picture the exposure is metered on, with the vignetting gain the render applies.
        let balanced = preview.metered(view.estimates.initial_gains().unwrap());
        let matrix = *view
            .estimates
            .color
            .metering_chain(ym_core::auto::tables_of(&view.profile))
            .unwrap()
            .matrix();
        let ours = preview_median_luminance(
            &PreviewRgb {
                width: preview.width,
                height: preview.height,
                rgb: &balanced,
                clip_mask: None,
            },
            &matrix,
            view.estimates.exposure.ev,
        )
        .unwrap()
        .unwrap();
        let Some(jpeg) = ym_raw::metadata::read_preview(&view.path).unwrap() else {
            eprintln!("{}: no embedded preview", photo.file_name);
            continue;
        };
        let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg));
        let pixels = decoder.decode().unwrap();
        let reference = jpeg_median_luminance(&pixels).unwrap().unwrap();
        eprintln!(
            "[old metric] {:<22} EV_auto {:>5.2}  our median Y {:.4}  JPEG median Y {:.4}  error {:>5.2} EV",
            photo.file_name,
            view.estimates.exposure.ev,
            ours,
            reference,
            exposure_error_ev(ours, reference)
        );
        entries.push(GateEntry {
            name: photo.file_name,
            ours_median_y: ours,
            reference_median_y: reference,
        });
    }
    let report = evaluate_gate(&entries, GATE_TOLERANCE_EV, GATE_REQUIRED_FRACTION);
    eprintln!(
        "[old metric, informational; the gate is G3 in tools/eval/autoexp.py] {} of {} within {GATE_TOLERANCE_EV} EV of the embedded JPEG by median luminance in linear light ({:.0} %); outliers: {:?}",
        report.within,
        report.evaluated,
        100.0 * report.fraction,
        report
            .outliers
            .iter()
            .map(|o| (&o.name, (o.error_ev * 100.0).round() / 100.0))
            .collect::<Vec<_>>()
    );
    assert_eq!(report.evaluated + report.skipped.len(), entries.len());
}
