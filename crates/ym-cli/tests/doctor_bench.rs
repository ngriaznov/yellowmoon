//! `ym doctor --bench FILE` (`docs/10-cli.md`): the timings of the render path on one photo, as a report that names
//! no file. Runs on the CPU reference (`YELLOWMOON_GPU=off`) everywhere, and on the device where an adapter opens
//! (Mesa lavapipe in CI, Metal on the Apple runners).

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::Sandbox;
use serde_json::Value;

fn bench(sb: &Sandbox, arw: &std::path::Path, policy: &str, extra: &[&str]) -> (Value, String) {
    let out = sb
        .ym()
        .env("YELLOWMOON_GPU", policy)
        .args(["doctor", "--bench"])
        .arg(arw)
        .args(extra)
        .assert()
        .success();
    let text = String::from_utf8(out.get_output().stdout.clone()).expect("utf-8");
    let json = if extra.contains(&"--json") {
        serde_json::from_str(text.trim()).expect("one JSON object")
    } else {
        Value::Null
    };
    (json, text)
}

fn adapter_opens(sb: &Sandbox) -> bool {
    let out = sb
        .ym()
        .env_remove("YELLOWMOON_GPU")
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    v["gpu"]["available"] == true
}

#[test]
fn the_report_of_the_cpu_path_has_every_part_and_names_no_file() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("a-name-nobody-should-see.ARW", 400, 300, 3);
    let (v, _) = bench(&sb, &arw, "off", &["--json", "--repeat", "2"]);
    assert_eq!(v["bench"], 1);
    assert_eq!(v["photo"]["camera"], "ILCE-7RM4");
    assert_eq!(
        (v["photo"]["width"].as_u64(), v["photo"]["height"].as_u64()),
        (Some(400), Some(300))
    );
    assert_eq!(v["gpu"]["policy"], "off");
    assert_eq!(v["gpu"]["uses_gpu"], false);
    assert_eq!(
        v["compile_ms"],
        serde_json::json!({}),
        "no device, no kernels to compile"
    );
    let views = v["views"].as_array().expect("views");
    let names: Vec<&str> = views.iter().filter_map(|x| x["name"].as_str()).collect();
    assert_eq!(
        names,
        [
            "fit, 1440 x 960 cell",
            "fit, 2880 x 1920 cell (retina)",
            "100 %, 1440 x 960 cell",
            "colour-only edit, 1440 x 960 cell",
            "thumbnail"
        ]
    );
    for view in views {
        assert!(
            view["cold_ms"].as_f64().is_some_and(|ms| ms > 0.0),
            "{view}"
        );
    }
    // A colour-only edit does no raw work (the raw group is cached).
    assert_eq!(views[3]["raw_work"], 0);
    let exports = v["exports"].as_array().expect("exports");
    let formats: Vec<&str> = exports.iter().filter_map(|x| x["name"].as_str()).collect();
    assert_eq!(formats, ["jpeg", "tiff 16", "png 16"]);
    for e in exports {
        assert_eq!(e["device"], "cpu", "{e}");
        assert_eq!(
            (e["width"].as_u64(), e["height"].as_u64()),
            (Some(400), Some(300))
        );
        assert!(e["bytes"].as_u64().is_some_and(|b| b > 0));
    }
    assert_eq!(v["repeat"]["runs"], 2);
    assert_eq!(v["repeat"]["identical"], true, "{}", v["repeat"]);
    assert!(
        v["memory"]["host_ledger_peak_bytes"]
            .as_u64()
            .is_some_and(|b| b > 0)
    );
    assert_eq!(v["memory"]["device_pool_peak_in_use_bytes"], Value::Null);
    // The privacy promise of doctor: no file name, no folder.
    let text = v.to_string();
    assert!(!text.contains("a-name-nobody"), "{text}");
    assert!(
        !text.contains(&sb.root().to_string_lossy().into_owned()),
        "{text}"
    );
    assert!(
        !text.contains(&arw.to_string_lossy().into_owned()),
        "{text}"
    );
}

#[test]
fn the_text_report_has_the_headline_lines() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("a.ARW", 320, 240, 5);
    let (_, text) = bench(&sb, &arw, "off", &["--repeat", "1"]);
    for part in [
        "Timings of one photo on this machine. It names no file and no folder.",
        "GPU policy: off",
        "Photo: ILCE-7RM4 320 x 240",
        "fit, 1440 x 960 cell",
        "Export jpeg",
        "Export png 16",
        "Repeat: 1 exports of the long edge 2000 px, bytes identical",
        "Peak memory: host ledger",
    ] {
        assert!(text.contains(part), "{part:?} is missing from\n{text}");
    }
    assert!(!text.contains("a.ARW"));
}

#[test]
fn on_the_device_the_exports_say_gpu_the_kernels_are_timed_and_the_bytes_repeat() {
    let sb = Sandbox::new();
    if !adapter_opens(&sb) {
        eprintln!("skipped: no graphics adapter");
        assert!(
            std::env::var_os("YM_REQUIRE_GPU").is_none(),
            "YM_REQUIRE_GPU is set but no adapter opens"
        );
        return;
    }
    let arw = sb.arw_sized("a.ARW", 400, 300, 3);
    let (v, _) = bench(&sb, &arw, "on", &["--json", "--repeat", "2"]);
    assert_eq!(v["gpu"]["uses_gpu"], true);
    for set in ["view", "raw", "tail"] {
        assert!(
            v["compile_ms"][set].as_f64().is_some_and(|ms| ms >= 0.0),
            "{set}: {}",
            v["compile_ms"]
        );
    }
    for e in v["exports"].as_array().expect("exports") {
        assert_eq!(e["device"], "gpu", "{e}");
        assert!(
            !e["notes"].to_string().contains("graphics card"),
            "a fallback note: {e}"
        );
    }
    assert_eq!(v["repeat"]["identical"], true, "{}", v["repeat"]);
    assert!(
        v["memory"]["device_pool_peak_in_use_bytes"]
            .as_u64()
            .is_some_and(|b| b > 0)
    );
    assert!(
        v["device_counters"]["dispatches"]
            .as_u64()
            .is_some_and(|n| n > 0)
    );
    // Where the adapter has timestamp queries the GPU time of the passes is reported (lavapipe and Metal have them).
    if v["gpu"]["adapter"]["timestamp_query"] == true {
        let passes = v["exports"][0]["gpu_passes"].as_array().expect("passes");
        assert!(
            !passes.is_empty(),
            "timestamps are on but no pass was timed"
        );
        assert!(
            passes
                .iter()
                .all(|p| p["total_ms"].as_f64().is_some_and(|t| t >= 0.0))
        );
    }
}

#[test]
fn bad_input_is_named_and_a_strict_policy_without_a_device_is_exit_4() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    // --repeat without --bench is a usage error.
    sb.ym().args(["doctor", "--repeat", "3"]).assert().code(2);
    // Not a photo.
    let junk = sb.root().join("junk.txt");
    std::fs::write(&junk, "not a raw file").expect("write");
    let out = sb
        .ym()
        .env("YELLOWMOON_GPU", "off")
        .args(["doctor", "--bench"])
        .arg(&junk)
        .assert()
        .failure();
    let err = String::from_utf8_lossy(&out.get_output().stderr).into_owned();
    assert!(
        err.contains("can't open") || err.contains("not a photo") || err.contains("Try an ARW"),
        "{err}"
    );
    // A device that cannot open when the setting needs one.
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .env("YELLOWMOON_GPU", "require")
        .args(["doctor", "--bench"])
        .arg(&arw)
        .assert()
        .code(4);
}

/// Once the photo is developed the first Ctrl-C only sets a flag (`runtime::handle_interrupts`), so every phase of the bench
/// has to look at it: the run ends with 130 at the next phase or tile, and the folder its exports were written to
/// (`$TMPDIR/ym-bench-<pid>`) is gone. A bench that ignored the flag would go on through the whole repeat check.
#[cfg(unix)]
#[test]
fn ctrl_c_during_the_bench_ends_it_with_130_and_removes_its_scratch_folder() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 3000, 2000, 5);
    let tmp = sb.root().join("tmp");
    std::fs::create_dir_all(&tmp).expect("tmp");
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
        .args(["doctor", "--bench"])
        .arg(&arw)
        // A thousand exports: the run is still going whenever the signal arrives.
        .args(["--repeat", "1000"])
        .env("YELLOWMOON_HOME", sb.home())
        .env("YELLOWMOON_DATA", common::data_dir())
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .env("YELLOWMOON_GPU", "off")
        .env("TMPDIR", &tmp)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ym");
    // Past the development of the photo (the flag is looked at from there on) and into the views or the exports.
    std::thread::sleep(Duration::from_secs(8));
    let kill = std::process::Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("kill");
    assert!(kill.success());
    let signalled = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break Some(status);
        }
        if signalled.elapsed() > Duration::from_secs(90) {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let status =
        status.expect("the bench stops within 90 s of Ctrl-C (it went on, and was killed)");
    assert_eq!(status.code(), Some(130), "{status:?}");
    let left: Vec<_> = std::fs::read_dir(&tmp)
        .expect("tmp")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .collect();
    assert!(left.is_empty(), "the scratch folder is left: {left:?}");
}
