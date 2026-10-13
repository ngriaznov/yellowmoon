//! The runner against the real `ym` binary and the synthetic set: a smoke test that two files render, hash, store and
//! check identical, and that `truth` reads a DNG back and finds the neutral patch neutral.
//!
//! Needs `ym` (`cargo build -p ym-cli`) and `testdata/synthetic` (`tools/fetch/synthetic.sh`). Without them the tests
//! say so and pass, unless `YM_REQUIRE_SYNTHETIC=1` is set.

#![cfg(unix)]

use std::path::PathBuf;

use golden::manifest::{Manifest, SetChoice, select};
use golden::runner::{Config, Metric, Status, check, update};
use golden::truth;
use golden::ymrun::{YmBinary, find_ym, workspace_root};

fn bench() -> Option<(YmBinary, PathBuf, Manifest)> {
    let require = std::env::var_os("YM_REQUIRE_SYNTHETIC").is_some();
    let testdata = std::env::var_os("YM_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("testdata"));
    let manifest = Manifest::load(&workspace_root().join("tools/golden/manifest.json")).unwrap();
    let ym = find_ym(None);
    let have_files =
        testdata.join("synthetic/arw").is_dir() && testdata.join("synthetic/scenes").is_dir();
    match (ym, have_files) {
        (Ok(path), true) => Some((YmBinary { path, data: None }, testdata, manifest)),
        (ym, files) => {
            assert!(
                !require,
                "YM_REQUIRE_SYNTHETIC is set but ym ({:?}) or the synthetic set ({files}) is missing",
                ym.is_ok()
            );
            eprintln!("SKIPPED: needs the ym binary and testdata/synthetic");
            None
        }
    }
}

#[test]
fn two_real_renders_are_stored_and_checked_identical() {
    let Some((ym, testdata, manifest)) = bench() else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let ctx = Config {
        ym,
        testdata,
        refs: dir.path().join("refs"),
        expected: dir.path().join("expected.json"),
        jobs: 2,
        tolerance: 0.5,
        metric: Metric::Max,
        quiet: true,
    };
    let entries: Vec<_> = select(
        &manifest,
        SetChoice::Synthetic,
        &["D6500__ILCE-7RM4__iso400__rggb__14b".to_owned()],
    )
    .into_iter()
    .filter(|e| e.variant.as_deref() == Some("lossless-l") || e.variant.as_deref() == Some("craw"))
    .collect();
    assert_eq!(entries.len(), 2);
    update(&ctx, &manifest, &entries, None, false).unwrap();
    let report = check(&ctx, &entries, false).unwrap();
    assert_eq!(report.count(Status::Identical), 2, "{:?}", report.outcomes);
}

#[test]
fn the_color_truth_of_one_scene_reads_back_and_beats_the_baseline() {
    let Some((ym, testdata, manifest)) = bench() else {
        return;
    };
    let entries = select(
        &manifest,
        SetChoice::Synthetic,
        &["chart__D6500__ILCE-7RM4__iso400__rggb__14b".to_owned()],
    );
    let scenes = truth::one_per_scene(&entries);
    assert_eq!(scenes.len(), 1);
    let r = truth::scene_result(&ym, &testdata, scenes[0]).unwrap();
    let chart = r.render["chart"];
    assert_eq!(chart.count, 190, "the chart patches that are not clipped");
    // first measurement: 0.61 (docs/13); the bound only catches a collapse of the colour chain
    assert!(chart.median.is_finite() && chart.median < 2.0, "{chart:?}");
    // the sensor map is what makes the product better than Sony's camera RGB through the same ARRI matrix (1.75)
    let baseline = r.baseline["chart"];
    assert!(
        chart.median < baseline.median,
        "render {chart:?} against the baseline {baseline:?}"
    );
    assert!(r.exposure_scale > 0.0);
    // the neutral ramp stays neutral: its median ΔE00 is below the chart's worst patch
    assert!(r.render["neutral"].median <= chart.max);
}
