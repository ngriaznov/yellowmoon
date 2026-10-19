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
use golden::runner::{
    Config, GpuGate, Metric, Reference, Refused, Status, UpdateKind, check, check_gpu,
    reference_path, update,
};
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
        // The stub prints what `ym render --json` prints and copies the picture the settings name, and refuses a
        // render without `--apple-auto off` (the golden pictures are Yellowmoon's own Auto everywhere). Under
        // YELLOWMOON_GPU=require (the GPU mode of the runner) it says `device: gpu`, copies `picture_gpu` when the
        // settings have one, and `fallback` makes it say what a quiet fallback looks like: `device` (the JSON says cpu)
        // or `note` (the JSON says gpu and carries the fallback note). `ym doctor --json` answers with an adapter unless
        // a file called no-adapter lies next to the script.
        let script = r#"#!/bin/sh
if [ "$1" = "doctor" ]; then
  if [ -f "$(dirname "$0")/no-adapter" ]; then
    echo '{"gpu":{"available":false,"reason":"none"}}'
  else
    echo '{"gpu":{"available":true,"name":"Stub GPU","backend":"Stub","software":false}}'
  fi
  exit 0
fi
out=""; settings=""; input=""; apple=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift ;;
    --settings) settings="$2"; shift ;;
    render|--json|--quiet|--overwrite|--cpu|--require-gpu) ;;
    --apple-auto) apple="$2"; shift ;;
    --format|--bits|--colour-space|--name|--quality) shift ;;
    *) input="$1" ;;
  esac
  shift
done
# The golden pictures are Yellowmoon's own Auto on every machine (docs/19 section 7).
if [ "$apple" != "off" ]; then echo "golden renders without --apple-auto off" >&2; exit 2; fi
pic=$(sed -n 's/.*"picture":"\([^"]*\)".*/\1/p' "$settings")
dev=cpu
notes='[]'
if [ "$YELLOWMOON_GPU" = "require" ]; then
  gpic=$(sed -n 's/.*"picture_gpu":"\([^"]*\)".*/\1/p' "$settings")
  if [ -n "$gpic" ]; then pic="$gpic"; fi
  dev=gpu
  fallback=$(sed -n 's/.*"fallback":"\([^"]*\)".*/\1/p' "$settings")
  if [ "$fallback" = "device" ]; then dev=cpu; fi
  if [ "$fallback" = "note" ]; then notes='["The graphics card could not render this photo, so the CPU did: stub."]'; fi
fi
if [ "$pic" = "FAIL" ]; then
  echo '{"input":"'"$input"'","output":null,"status":"error","error":{"kind":"look","message":"File missing or changed. Import again."}}'
  exit 1
fi
mkdir -p "$out"
name=$(basename "$input" .ARW)
cp "$pic" "$out/$name.png"
echo '{"input":"'"$input"'","output":"'"$out/$name.png"'","status":"ok","device":"'"$dev"'","notes":'"$notes"',"pipeline_version":"'"${STUB_VERSION:-0.1.0}"'","error":null}'
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

    /// A flat picture whose first pixel has `delta` more red.
    fn speck(&self, name: &str, rgb: [u16; 3], delta: u16) -> String {
        let mut data: Vec<u16> = (0..256).flat_map(|_| rgb).collect();
        data[0] += delta;
        let path = self.root().join("pics").join(format!("{name}.png"));
        write_png16(
            &path,
            &Rgb16 {
                width: 16,
                height: 16,
                data,
            },
        )
        .unwrap();
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

// ---------------------------------------------------------------------------------------------------------------
// GPU mode (`golden check --device gpu`)
// ---------------------------------------------------------------------------------------------------------------

/// The stored CPU pictures of a bench: an update with the CPU render, which the stub answers with `picture`.
fn stored(b: &Bench, pictures: [&str; 2]) -> (Manifest, Config) {
    let manifest = Manifest::load(&b.write_manifest(pictures)).unwrap();
    let ctx = b.config();
    update(&ctx, &manifest, &selected(&manifest), None, false).unwrap();
    (manifest, ctx)
}

#[test]
fn a_device_picture_equal_to_the_cpu_one_is_identical_and_says_where_it_was_made() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let (manifest, ctx) = stored(&b, [&grey, &red]);
    let report = check_gpu(
        &ctx,
        &GpuGate::PLAN,
        Reference::Stored,
        &selected(&manifest),
    )
    .unwrap();
    assert_eq!(report.count(Status::Identical), 2, "{:?}", report.outcomes);
    assert!(!report.failed());
    for o in report
        .outcomes
        .iter()
        .filter(|o| o.status != Status::Missing)
    {
        assert_eq!(o.device.as_deref(), Some("gpu"), "{}", o.id);
        assert!(o.notes.is_empty());
    }
    let gpu = report.gpu.as_ref().expect("the GPU part of the report");
    assert_eq!(gpu.reference, Reference::Stored);
    assert_eq!(gpu.gate, GpuGate::PLAN);
}

#[test]
fn a_device_picture_within_the_gates_passes_and_each_gate_can_fail_it() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    let (mut manifest, ctx) = stored(&b, [&grey, &red]);

    // The device is two codes of 16 bits away on one channel (invisible) in a, one pixel of b is 100 codes off
    // (a sparkle: invisible to the 1/4 downsample, but more than 8 codes at 3 of 768 samples).
    let nearly = b.picture("nearly", [30002, 30000, 30000]);
    let spark = b.speck("spark", [40000, 10000, 10000], 100);
    manifest.entries[0].settings = json!({ "picture": grey, "picture_gpu": nearly });
    manifest.entries[1].settings = json!({ "picture": red, "picture_gpu": spark });
    let entries = selected(&manifest);
    let report = check_gpu(&ctx, &GpuGate::PLAN, Reference::Stored, &entries).unwrap();
    let a = by_id(&report, "synthetic/a");
    assert_eq!(a.status, Status::Within, "{}", a.message);
    assert!(a.delta_e_2000.unwrap().max < 0.01);
    assert_eq!(a.codes.unwrap().over, 0);
    assert_eq!(a.codes.unwrap().max, 2);
    let c = by_id(&report, "synthetic/b");
    assert_eq!(c.status, Status::Exceeded, "{}", c.message);
    assert_eq!(c.codes.unwrap().over, 1);
    assert_eq!(c.codes.unwrap().max, 100);
    assert!(c.message.contains("share over codes"), "{}", c.message);
    assert!(report.failed());

    // The share gate is the only one that stops b; with a share of 1e-2 it passes.
    let loose = GpuGate {
        codes_share: 1e-2,
        ..GpuGate::PLAN
    };
    let report = check_gpu(&ctx, &loose, Reference::Stored, &entries).unwrap();
    assert!(!report.failed(), "{:?}", report.outcomes);

    // A different colour fails the colour gates, and a gate of 0 fails even the two codes.
    let green = b.picture("green", [10000, 40000, 10000]);
    manifest.entries[1].settings = json!({ "picture": red, "picture_gpu": green });
    let report = check_gpu(
        &ctx,
        &GpuGate::PLAN,
        Reference::Stored,
        &selected(&manifest),
    )
    .unwrap();
    let c = by_id(&report, "synthetic/b");
    assert_eq!(c.status, Status::Exceeded);
    assert!(
        c.message.contains("median") && c.message.contains("max"),
        "{}",
        c.message
    );
    let zero = GpuGate {
        median: 0.0,
        p99: 0.0,
        max: 0.0,
        ..GpuGate::PLAN
    };
    let report = check_gpu(&ctx, &zero, Reference::Stored, &selected(&manifest)).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::Exceeded);

    // A gate that is not a number would let everything through: refused.
    let nan = GpuGate {
        max: f64::NAN,
        ..GpuGate::PLAN
    };
    assert!(check_gpu(&ctx, &nan, Reference::Stored, &selected(&manifest)).is_err());
}

#[test]
fn a_render_that_ran_on_the_cpu_or_says_it_fell_back_cannot_pass() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let (mut manifest, ctx) = stored(&b, [&grey, &grey]);
    // Both pictures are the stored ones, so only the report of where they were made can fail them.
    manifest.entries[0].settings = json!({ "picture": grey, "fallback": "device" });
    manifest.entries[1].settings = json!({ "picture": grey, "fallback": "note" });
    let report = check_gpu(
        &ctx,
        &GpuGate::PLAN,
        Reference::Stored,
        &selected(&manifest),
    )
    .unwrap();
    let a = by_id(&report, "synthetic/a");
    assert_eq!(a.status, Status::FellBack, "{}", a.message);
    assert_eq!(a.device.as_deref(), Some("cpu"));
    assert!(a.message.contains("not \"gpu\""), "{}", a.message);
    let c = by_id(&report, "synthetic/b");
    assert_eq!(c.status, Status::FellBack, "{}", c.message);
    assert_eq!(c.device.as_deref(), Some("gpu"));
    assert!(c.message.contains("graphics card"), "{}", c.message);
    assert_eq!(c.notes.len(), 1);
    assert!(report.failed());
    assert!(Status::FellBack.failed());
}

#[test]
fn the_stored_reference_must_exist_and_the_render_reference_needs_none() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let (mut manifest, ctx) = stored(&b, [&grey, &grey]);
    let nearly = b.picture("nearly", [30002, 30000, 30000]);
    manifest.entries[0].settings = json!({ "picture": grey, "picture_gpu": nearly });
    let entries = selected(&manifest);

    fs::remove_dir_all(&ctx.refs).unwrap();
    let report = check_gpu(&ctx, &GpuGate::PLAN, Reference::Stored, &entries).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::NoReference);
    assert_eq!(by_id(&report, "synthetic/b").status, Status::Identical);

    // A cache from another version is not the picture expected.json describes.
    fs::create_dir_all(reference_path(&ctx.refs, "synthetic/a").parent().unwrap()).unwrap();
    let red = b.picture("red", [40000, 10000, 10000]);
    fs::copy(&red, reference_path(&ctx.refs, "synthetic/a")).unwrap();
    let report = check_gpu(&ctx, &GpuGate::PLAN, Reference::Stored, &entries).unwrap();
    assert_eq!(by_id(&report, "synthetic/a").status, Status::StaleReference);

    // The render reference does not read expected.json or the pictures at all: a CPU render of the same run.
    fs::remove_file(&ctx.expected).unwrap();
    let report = check_gpu(&ctx, &GpuGate::PLAN, Reference::Render, &entries).unwrap();
    let a = by_id(&report, "synthetic/a");
    assert_eq!(a.status, Status::Within, "{}", a.message);
    assert_eq!(a.codes.unwrap().max, 2);
    assert_eq!(by_id(&report, "synthetic/b").status, Status::Identical);
}

#[test]
fn the_binary_runs_the_gpu_mode_on_a_subset_and_refuses_what_cannot_be_trusted() {
    let b = Bench::new();
    let grey = b.picture("grey", [30000; 3]);
    let red = b.picture("red", [40000, 10000, 10000]);
    b.write_manifest([&grey, &red]);
    b.cmd("update").assert().success();
    let report_path = b.root().join("gpu.json");

    b.cmd("check")
        .args(["--device", "gpu", "--report"])
        .arg(&report_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("GPU run on Stub GPU (Stub)"))
        .stdout(predicate::str::contains(
            "2 of 2 rendered files say device gpu",
        ));
    let report: Value = serde_json::from_str(&fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(report["metric"], "gpu");
    assert_eq!(report["gpu"]["adapter"]["name"], "Stub GPU");
    assert_eq!(report["gpu"]["reference"], "stored");
    assert_eq!(report["outcomes"][0]["device"], "gpu");

    // A subset file keeps the listed ids; comments and blank lines are ignored.
    let subset = b.root().join("subset.txt");
    fs::write(&subset, "# one file\n\nsynthetic/b  # the red one\n").unwrap();
    b.cmd("check")
        .args(["--device", "gpu", "--subset"])
        .arg(&subset)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "1 of 1 rendered files say device gpu",
        ));
    // An id the manifest does not have is a setup error, not a smaller subset.
    fs::write(&subset, "synthetic/typo\n").unwrap();
    b.cmd("check")
        .args(["--device", "gpu", "--subset"])
        .arg(&subset)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("synthetic/typo"));

    // The CPU pictures are not written from a device run.
    b.cmd("check")
        .args(["--device", "gpu", "--write-missing-refs"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--write-missing-refs"));
    // No adapter: a setup error before anything is rendered.
    fs::write(b.root().join("no-adapter"), "").unwrap();
    b.cmd("check")
        .args(["--device", "gpu"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("no graphics adapter"));
    // The CPU mode does not need one.
    b.cmd("check").assert().success();

    // An id that the selected set leaves out is a setup error as well: the subset would shrink without a word.
    let manifest_path = b.root().join("manifest.json");
    let mut doc: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    doc["entries"][1]["set"] = json!("full");
    fs::write(&manifest_path, doc.to_string()).unwrap();
    fs::write(&subset, "synthetic/a\nsynthetic/b\n").unwrap();
    fs::remove_file(b.root().join("no-adapter")).unwrap();
    b.cmd("check")
        .args(["--device", "gpu", "--subset"])
        .arg(&subset)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("not in the selected set"));
    // With the set that holds both, the same subset runs both files.
    b.cmd("check")
        .args(["--device", "gpu", "--set", "synthetic-full", "--subset"])
        .arg(&subset)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "2 of 2 rendered files say device gpu",
        ));
}
