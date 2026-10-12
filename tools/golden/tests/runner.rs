//! `golden check`, `update` and the colour truth against a stand-in `ym` (a shell script that copies the picture named in
//! the settings), so the logic of the runner is tested without rendering anything. The real binary is exercised by
//! `tests/real_ym.rs` when `ym` and the synthetic set exist.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use golden::expected::Expected;
use golden::image::{Rgb16, write_png16};
use golden::manifest::{Manifest, SetChoice, select};
use golden::runner::{Config, Metric, Refused, Status, UpdateKind, check, reference_path, update};
use golden::ymrun::YmBinary;
use predicates::prelude::*;
use serde_json::{Value, json};

/// A scratch tree: a stub `ym`, test data with two inputs and pictures to choose from.
struct Bench {
    dir: tempfile::TempDir,
}

impl Bench {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let b = Self { dir };
        fs::create_dir_all(b.testdata().join("in")).unwrap();
        for name in ["a.ARW", "b.ARW"] {
            fs::write(b.testdata().join("in").join(name), b"arw").unwrap();
        }
        fs::create_dir_all(b.root().join("pics")).unwrap();
        // The stub prints what `ym render --json` prints and copies the picture the settings name.
        let script = r#"#!/bin/sh
out=""; settings=""; input=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift ;;
    --settings) settings="$2"; shift ;;
    render|--json|--quiet|--overwrite) ;;
    --format|--bits|--colour-space|--name|--quality) shift ;;
    *) input="$1" ;;
  esac
  shift
done
pic=$(sed -n 's/.*"picture":"\([^"]*\)".*/\1/p' "$settings")
if [ "$pic" = "FAIL" ]; then
  echo '{"input":"'"$input"'","output":null,"status":"error","error":{"kind":"look","message":"File missing or changed. Import again."}}'
  exit 1
fi
mkdir -p "$out"
name=$(basename "$input" .ARW)
cp "$pic" "$out/$name.png"
echo '{"input":"'"$input"'","output":"'"$out/$name.png"'","status":"ok","pipeline_version":"'"${STUB_VERSION:-0.1.0}"'","error":null}'
"#;
        let ym = b.root().join("ym");
        fs::write(&ym, script).unwrap();
        fs::set_permissions(&ym, fs::Permissions::from_mode(0o755)).unwrap();
        b
    }

    fn root(&self) -> &Path {
        self.dir.path()
    }

    fn testdata(&self) -> PathBuf {
        self.root().join("testdata")
    }

    /// A picture of one flat colour, 16 x 16, stored under pics/.
    fn picture(&self, name: &str, rgb: [u16; 3]) -> String {
        let p = Rgb16 {
            width: 16,
            height: 16,
            data: (0..256).flat_map(|_| rgb).collect(),
        };
        let path = self.root().join("pics").join(format!("{name}.png"));
        write_png16(&path, &p).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn write_manifest(&self, pictures: [&str; 2]) -> PathBuf {
        let entry = |id: &str, file: &str, pic: &str| json!({ "id": id, "tier": "synthetic", "set": "default", "file": file, "settings": { "picture": pic } });
        let doc = json!({
            "entries": [
                entry("synthetic/a", "in/a.ARW", pictures[0]),
                entry("synthetic/b", "in/b.ARW", pictures[1]),
                json!({ "id": "synthetic/gone", "tier": "synthetic", "set": "default", "file": "in/gone.ARW", "settings": {} }),
            ]
        });
        let path = self.root().join("manifest.json");
        fs::write(&path, doc.to_string()).unwrap();
        path
    }

    fn config(&self) -> Config {
        Config {
            ym: YmBinary {
                path: self.root().join("ym"),
                data: None,
            },
            testdata: self.testdata(),
            refs: self.testdata().join("golden/refs"),
            expected: self.root().join("expected.json"),
            jobs: 2,
            tolerance: 0.5,
            metric: Metric::Max,
            quiet: true,
        }
    }

    fn cmd(&self, sub: &str) -> Command {
        let mut c = Command::cargo_bin("golden").unwrap();
        c.arg(sub)
            .arg("--manifest")
            .arg(self.root().join("manifest.json"))
            .arg("--testdata")
            .arg(self.testdata())
            .arg("--ym")
            .arg(self.root().join("ym"))
            .arg("--expected")
            .arg(self.root().join("expected.json"))
            .arg("--quiet");
        c
    }
}

fn selected(m: &Manifest) -> Vec<&golden::manifest::Entry> {
    select(m, SetChoice::Synthetic, &[])
}

fn by_id<'a>(report: &'a golden::runner::CheckReport, id: &str) -> &'a golden::runner::Outcome {
    report.outcomes.iter().find(|o| o.id == id).unwrap()
}

#[test]
fn an_update_then_a_check_is_identical_and_a_missing_input_is_skipped() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let manifest = Manifest::load(&b.write_manifest([&grey, &red])).unwrap();
    let ctx = b.config();
    let entries = selected(&manifest);

    let up = update(&ctx, &manifest, &entries, None, false).unwrap();
    assert_eq!(up.count(UpdateKind::Added), 2);
    assert_eq!(up.missing, ["synthetic/gone"]);
    assert!(up.written);
    let expected = Expected::load(&ctx.expected).unwrap();
    assert_eq!(expected.entries.len(), 2);
    assert_eq!(expected.log.len(), 1);
    assert_eq!(expected.log[0].added, 2);
    assert!(reference_path(&ctx.refs, "synthetic/a").is_file());

    let report = check(&ctx, &entries, false).unwrap();
    assert_eq!(report.count(Status::Identical), 2);
    assert_eq!(report.count(Status::Missing), 1);
    assert!(!report.failed());
    assert_eq!(report.checked(), 2);
    assert_eq!(by_id(&report, "synthetic/a").delta_e_2000.unwrap().max, 0.0);

    // nothing changed: a second update does not touch expected.json
    let before = fs::read_to_string(&ctx.expected).unwrap();
    let again = update(&ctx, &manifest, &entries, None, false).unwrap();
    assert_eq!(again.count(UpdateKind::Unchanged), 2);
    assert!(!again.written);
    assert_eq!(fs::read_to_string(&ctx.expected).unwrap(), before);
}

#[test]
fn a_small_change_passes_and_a_large_one_fails_with_numbers() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let mut manifest = Manifest::load(&b.write_manifest([&grey, &red])).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();

    // a is one 16-bit code away (invisible), b becomes a different colour
    let nearly = b.picture("nearly", [30001, 30000, 30000]);
    let green = b.picture("green", [10000, 40000, 10000]);
    manifest.entries[0].settings = json!({ "picture": nearly });
    manifest.entries[1].settings = json!({ "picture": green });
    let report = check(&ctx, &selected(&manifest), false).unwrap();
    let a = by_id(&report, "synthetic/a");
    let c = by_id(&report, "synthetic/b");
    assert_eq!(a.status, Status::Within, "{}", a.message);
    assert!(a.delta_e_2000.unwrap().max < 0.01);
    assert_eq!(c.status, Status::Exceeded, "{}", c.message);
    assert!(c.delta_e_2000.unwrap().max > 20.0);
    assert!(c.message.contains("golden update --note"));
    assert!(report.failed());

    // a looser tolerance lets the same change through; p99 is the other statistic
    let mut loose = ctx.clone();
    loose.tolerance = 200.0;
    assert!(!check(&loose, &selected(&manifest), false).unwrap().failed());
    let mut p99 = ctx.clone();
    p99.metric = Metric::P99;
    assert!(check(&p99, &selected(&manifest), false).unwrap().failed());
}

#[test]
fn a_change_needs_a_note_and_the_note_is_kept() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let mut manifest = Manifest::load(&b.write_manifest([&grey, &red])).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    let before = fs::read_to_string(&ctx.expected).unwrap();

    let blue = b.picture("blue", [10000, 10000, 40000]);
    manifest.entries[1].settings = json!({ "picture": blue });
    let entries = selected(&manifest);
    for note in [None, Some("  ")] {
        let err = update(&ctx, &manifest, &entries, note, false).unwrap_err();
        assert!(err.downcast_ref::<Refused>().is_some(), "{err:#}");
        assert!(format!("{err:#}").contains("synthetic/b"));
        assert_eq!(
            fs::read_to_string(&ctx.expected).unwrap(),
            before,
            "nothing is written"
        );
    }
    let up = update(
        &ctx,
        &manifest,
        &entries,
        Some("the blue one is meant"),
        false,
    )
    .unwrap();
    assert_eq!(up.count(UpdateKind::Changed), 1);
    let measured = up.outcomes.iter().find(|o| o.id == "synthetic/b").unwrap();
    assert!(
        measured.delta_e_2000.unwrap().max > 20.0,
        "the old reference was there to measure against"
    );
    let expected = Expected::load(&ctx.expected).unwrap();
    assert_eq!(expected.log.len(), 2);
    assert_eq!(expected.log[1].note, "the blue one is meant");
    assert_eq!((expected.log[1].changed, expected.log[1].added), (1, 0));
    assert!(expected.log[1].max_delta_e_2000 > 20.0);
    assert!(!check(&ctx, &entries, false).unwrap().failed());
}

#[test]
fn a_changed_picture_without_a_reference_cannot_be_measured() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let mut manifest = Manifest::load(&b.write_manifest([&grey, &red])).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    let blue = b.picture("blue", [10000, 10000, 40000]);
    manifest.entries[0].settings = json!({ "picture": blue });

    // the cache is gone: no reference picture
    fs::remove_dir_all(&ctx.refs).unwrap();
    let report = check(&ctx, &selected(&manifest), false).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::NoReference);
    // an unchanged file does not need one
    assert_eq!(by_id(&report, "synthetic/b").status, Status::Identical);

    // a reference that is not the one expected.json describes (a cache from another version)
    fs::create_dir_all(reference_path(&ctx.refs, "synthetic/a").parent().unwrap()).unwrap();
    fs::copy(&red, reference_path(&ctx.refs, "synthetic/a")).unwrap();
    let report = check(&ctx, &selected(&manifest), false).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::StaleReference);
}

#[test]
fn identical_pictures_can_refill_the_reference_cache() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let manifest = Manifest::load(&b.write_manifest([&grey, &red])).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    fs::remove_dir_all(&ctx.refs).unwrap();
    check(&ctx, &selected(&manifest), false).unwrap();
    assert!(
        !reference_path(&ctx.refs, "synthetic/a").exists(),
        "not without the option"
    );
    check(&ctx, &selected(&manifest), true).unwrap();
    assert!(reference_path(&ctx.refs, "synthetic/a").is_file());
    assert!(reference_path(&ctx.refs, "synthetic/b").is_file());
}

#[test]
fn a_file_missing_from_expected_json_and_a_failing_render_are_failures() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let mut manifest = Manifest::load(&b.write_manifest([&grey, &grey])).unwrap();
    let ctx = b.config();
    // nothing expected yet
    let report = check(&ctx, &selected(&manifest), false).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::NotExpected);
    assert!(report.failed());

    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    manifest.entries[0].settings = json!({ "picture": "FAIL" });
    let report = check(&ctx, &selected(&manifest), false).unwrap();
    let a = by_id(&report, "synthetic/a");
    assert_eq!(a.status, Status::RenderFailed);
    assert!(
        a.message.contains("File missing or changed"),
        "{}",
        a.message
    );
    // an update stops before it writes anything
    let before = fs::read_to_string(&ctx.expected).unwrap();
    let err = update(&ctx, &manifest, &selected(&manifest), Some("x"), false).unwrap_err();
    assert!(err.downcast_ref::<Refused>().is_some());
    assert_eq!(fs::read_to_string(&ctx.expected).unwrap(), before);
}

#[test]
fn a_size_change_is_a_failure() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let mut manifest = Manifest::load(&b.write_manifest([&grey, &grey])).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    let wide = Rgb16 {
        width: 32,
        height: 16,
        data: vec![30000; 32 * 16 * 3],
    };
    let wide_path = b.root().join("pics/wide.png");
    write_png16(&wide_path, &wide).unwrap();
    manifest.entries[0].settings = json!({ "picture": wide_path });
    let report = check(&ctx, &selected(&manifest), false).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::SizeChanged);
}

#[test]
fn pruning_drops_entries_the_manifest_no_longer_lists() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let manifest = Manifest::load(&b.write_manifest([&grey, &grey])).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    let mut e = Expected::load(&ctx.expected).unwrap();
    e.entries
        .insert("synthetic/old".into(), e.entries["synthetic/a"].clone());
    e.save(&ctx.expected).unwrap();
    let up = update(&ctx, &manifest, &selected(&manifest), None, true).unwrap();
    assert_eq!(up.pruned, ["synthetic/old"]);
    assert!(
        !Expected::load(&ctx.expected)
            .unwrap()
            .entries
            .contains_key("synthetic/old")
    );
}

#[test]
fn the_binary_exits_with_the_documented_codes() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    b.write_manifest([&grey, &red]);

    // nothing expected yet: the files fail
    b.cmd("check")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("NOT EXPECTED"));
    // an update writes them
    b.cmd("update")
        .assert()
        .success()
        .stdout(predicate::str::contains("2 added"));
    b.cmd("check")
        .arg("--report")
        .arg(b.root().join("report.json"))
        .assert()
        .success()
        .stdout(predicate::str::contains("2 identical"));
    let report: Value =
        serde_json::from_str(&fs::read_to_string(b.root().join("report.json")).unwrap()).unwrap();
    assert_eq!(report["outcomes"].as_array().unwrap().len(), 3);
    assert_eq!(report["tolerance"], json!(0.5));

    // --filter that matches nothing: exit 3, or 0 with --allow-empty
    b.cmd("check")
        .args(["--filter", "no-such-file"])
        .assert()
        .code(3);
    b.cmd("check")
        .args(["--filter", "no-such-file", "--allow-empty"])
        .assert()
        .success();
    // only the missing input selected: nothing was checked
    b.cmd("check").args(["--filter", "gone"]).assert().code(3);
    b.cmd("check")
        .args(["--filter", "gone", "--allow-empty"])
        .assert()
        .success();
    // --require-all makes the missing input a failure
    b.cmd("check").arg("--require-all").assert().code(1);
    // an unknown ym is a setup error
    let mut c = Command::cargo_bin("golden").unwrap();
    c.args(["check", "--ym", "/nonexistent/ym", "--manifest"])
        .arg(b.root().join("manifest.json"))
        .assert()
        .code(2)
        .stderr(predicate::str::contains("/nonexistent/ym"));
    // an update that would change a picture without a note is refused with 1
    let blue = b.picture("blue", [10000, 10000, 40000]);
    let doc = fs::read_to_string(b.root().join("manifest.json"))
        .unwrap()
        .replace(&grey, &blue);
    fs::write(b.root().join("manifest.json"), doc).unwrap();
    b.cmd("update")
        .assert()
        .code(1)
        .stderr(predicate::str::contains("--note"));
    b.cmd("update")
        .args(["--note", "blue on purpose"])
        .assert()
        .success();
    b.cmd("check").assert().success();
}
