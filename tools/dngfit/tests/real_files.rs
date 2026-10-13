//! The DNG gates on real Sony raw files, when `testdata/real/files` has some (local evaluation only: `tools/fetch/real-eval.sh`
//! fetches them from public GitHub repositories whose licences are unclear or restricted, and they are never committed or
//! copied). The files: an ARW 1 file (DSLR-A300), a cRAW file (NEX-5N) and a portrait cRAW frame of an A7 III (ILCE-7M3, Exif
//! orientation "Rotate 270 CW": the file on which a DNG whose orientation was applied twice was found, so it guards the
//! orientation of the DNG against `dng_validate`'s render). Slow: each file is developed three times by a debug build of `ym`
//! and rendered by `dng_validate`.
//!
//! `cargo test -p dngfit --test real_files -- --ignored --nocapture`

use std::path::PathBuf;

use dngfit::adobe::Adobe;
use dngfit::check::{GateOptions, MEDIAN_LIMIT, Reprofile, check_file};
use dngfit::profile_file;
use golden::manifest::Entry;
use golden::ymrun::{YmBinary, find_ym, workspace_root};

fn entry_of(path: &std::path::Path) -> Entry {
    Entry {
        id: path.file_name().unwrap().to_string_lossy().into_owned(),
        tier: "local".to_owned(),
        set: String::new(),
        file: path.to_string_lossy().into_owned(),
        sidecar: None,
        settings: serde_json::json!({}),
        variant: None,
        light: None,
        body: None,
        cfa: None,
        bits: None,
    }
}

#[test]
#[ignore = "renders real photographs (minutes in a debug build); run with --ignored"]
fn real_files_pass_both_dng_gates() {
    let dir = workspace_root().join("testdata/real/files");
    let files: Vec<PathBuf> = ["DSLR-A300.ARW", "NEX-5N.ARW", "ILCE-7M3.ARW"]
        .iter()
        .map(|n| dir.join(n))
        .filter(|p| p.is_file())
        .collect();
    if files.is_empty() {
        assert!(
            std::env::var_os("YM_REQUIRE_REAL_FILES").is_none(),
            "YM_REQUIRE_REAL_FILES is set but testdata/real/files is empty"
        );
        eprintln!("SKIPPED: no real files in {}", dir.display());
        return;
    }
    let (Ok(ym), Ok(adobe)) = (find_ym(None), Adobe::new(None)) else {
        eprintln!("SKIPPED: needs the ym binary and dng_validate");
        return;
    };
    let profile = profile_file::load(&workspace_root().join("data/dng/profile.json")).unwrap();
    let opts = GateOptions {
        ym: YmBinary {
            path: ym,
            data: None,
        },
        testdata: workspace_root().join("testdata"),
        adobe,
        reprofile: Reprofile::IfIdentity(profile),
        jobs: 1,
        with_png: true,
        median_limit: MEDIAN_LIMIT,
        quiet: true,
    };
    for file in &files {
        let r = check_file(&opts, &entry_of(file)).unwrap();
        eprintln!(
            "{}: {}x{} median {:.3} against the JPEG, {:.3} against the PNG",
            r.id,
            r.width,
            r.height,
            r.vs_jpeg.unwrap().median,
            r.vs_png.unwrap().median
        );
        assert!(r.passed, "{}: {:?}", r.id, r.reasons);
        assert!(
            r.validation_problems.is_empty(),
            "{:?}",
            r.validation_problems
        );
    }
}
