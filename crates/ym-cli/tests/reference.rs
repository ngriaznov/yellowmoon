//! `ym render --reference` (`docs/34-reference-grade.md` section 8, task RG-09): the flags, the node the fit writes, the report, and
//! every error text. The scene is the deterministic fake (`YELLOWMOON_TEST_SCENE=fake`), so no weights are needed; the one test of
//! what a copy without the scene model says removes them on purpose. That app and `ym` write the same bytes is `parity.rs`.

mod common;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::{FAKE_SCENE, Sandbox, json_records, text};
use serde_json::{Value, json};

/// A PNG of `w` x `h` with a smooth colour ramp.
fn picture(path: &Path, w: u32, h: u32) {
    let mut rgb = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgb.extend([
                (60 + 140 * x / w.max(1)) as u8,
                (160 - 90 * y / h.max(1)) as u8,
                (90 + 100 * (x + y) / (w + h).max(1)) as u8,
            ]);
        }
    }
    let file = std::fs::File::create(path).expect("create");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("header")
        .write_image_data(&rgb)
        .expect("data");
}

fn graph(nodes: Vec<Value>) -> Value {
    let mut items = vec![json!({ "id": 1 })];
    items.extend(nodes);
    json!({ "nodes": { "items": items } })
}

/// A run of `ym render` on one photo with the fake scene and the CPU.
fn render(sb: &Sandbox, photo: &Path, args: &[&str]) -> std::process::Output {
    sb.ym()
        .env(FAKE_SCENE.0, FAKE_SCENE.1)
        .args([
            "render",
            "--cpu",
            "--auto-provider",
            "yellowmoon",
            "--json",
            "--format",
            "png",
        ])
        .arg(photo)
        .args(args)
        .arg("--out")
        .arg(sb.root().join("out"))
        .output()
        .expect("ym runs")
}

fn reference(sb: &Sandbox) -> PathBuf {
    let file = sb.root().join("still.png");
    picture(&file, 320, 256);
    file
}

fn arg(p: &Path) -> &str {
    p.to_str().expect("utf-8")
}

#[test]
fn the_flags_need_a_reference() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    for flag in [
        vec!["--reference-strength", "50"],
        vec!["--no-skin-protection"],
        vec!["--reference-node", "2"],
    ] {
        let out = render(&sb, &arw, &flag);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{flag:?}: {}",
            text(&out.stderr)
        );
        assert!(text(&out.stderr).contains("--reference"), "{flag:?}");
    }
    let still = reference(&sb);
    let out = render(
        &sb,
        &arw,
        &["--reference", arg(&still), "--reference-strength", "101"],
    );
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn a_reference_that_cannot_be_used_ends_the_run_before_any_photo_with_the_sentence_of_the_card() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let say = |file: &Path| {
        let out = render(&sb, &arw, &["--reference", arg(file)]);
        assert_eq!(out.status.code(), Some(2), "{file:?}");
        assert!(
            out.stdout.is_empty(),
            "nothing is reported for a photo: {}",
            text(&out.stdout)
        );
        text(&out.stderr)
    };
    // A missing file.
    let said = say(&sb.root().join("nothing.jpg"));
    assert!(
        said.contains(
            "The reference can't be used: this file was not found. Check the path and try again."
        ),
        "{said}"
    );
    // A raw file, by its extension and as a file that is there.
    let said = say(&arw);
    assert!(
        said.contains("The reference can't be used: Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."),
        "{said}"
    );
    // Not a picture at all.
    let text_file = sb.root().join("notes.txt");
    std::fs::write(&text_file, b"not a picture").expect("write");
    assert!(say(&text_file).contains("Yellowmoon reads JPEG, PNG and TIFF pictures."));
    // A PNG that stops after its signature.
    let broken = sb.root().join("broken.png");
    std::fs::write(
        &broken,
        [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3],
    )
    .expect("write");
    let said = say(&broken);
    assert!(
        said.contains(
            "The reference can't be used: This file can't be read. It may be damaged. Choose another picture."
        ),
        "{said}"
    );
    // Black from edge to edge: no picture inside the bars.
    let black = sb.root().join("black.png");
    {
        let file = std::fs::File::create(&black).expect("create");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 320, 256);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .expect("header")
            .write_image_data(&vec![0u8; 320 * 256 * 3])
            .expect("data");
    }
    let said = say(&black);
    assert!(
        said.contains(
            "The reference can't be used: There's no picture inside the black bars. Choose another picture."
        ),
        "{said}"
    );
    // Too small.
    let small = sb.root().join("small.png");
    picture(&small, 200, 150);
    let said = say(&small);
    assert!(
        said.contains("The reference can't be used: This picture is too small to match. Use one that is at least 256 pixels on its short side."),
        "{said}"
    );
}

#[test]
fn a_graph_with_two_reference_grade_nodes_needs_the_node_named() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let still = reference(&sb);
    let doc = graph(vec![
        json!({ "id": 2, "kind": "reference_grade", "pos": [1, 0] }),
        json!({ "id": 3, "kind": "reference_grade", "pos": [2, 0] }),
    ]);
    // The links are the default ones, so neither node is connected; the choice is made before any fit.
    let file = sb.root().join("two.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    let out = render(
        &sb,
        &arw,
        &["--settings", arg(&file), "--reference", arg(&still)],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        text(&out.stderr).contains(
            "The graph has more than one Reference Grade node. Say which with --reference-node."
        ),
        "{}",
        text(&out.stderr)
    );
    // A node that is not a Reference Grade node.
    let out = render(
        &sb,
        &arw,
        &[
            "--settings",
            arg(&file),
            "--reference",
            arg(&still),
            "--reference-node",
            "1",
        ],
    );
    assert_ne!(out.status.code(), Some(0));
    let out = render(
        &sb,
        &arw,
        &[
            "--settings",
            arg(&file),
            "--reference",
            arg(&still),
            "--reference-node",
            "9",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        text(&out.stderr).contains("Node 9 is not a Reference Grade node."),
        "{}",
        text(&out.stderr)
    );
}

#[test]
fn a_copy_without_the_scene_model_says_so_for_the_photo_and_exits_1() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let still = reference(&sb);
    let data = sb.data_without_the_scene_weights();
    let out = sb
        .ym_data(&data)
        .args([
            "render",
            "--cpu",
            "--auto-provider",
            "yellowmoon",
            "--json",
            "--format",
            "png",
        ])
        .arg(&arw)
        .args(["--reference", arg(&still)])
        .arg("--out")
        .arg(sb.root().join("out"))
        .output()
        .expect("ym runs");
    assert_eq!(out.status.code(), Some(1), "{}", text(&out.stderr));
    let records = json_records(&out.stdout);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["status"], "error");
    assert_eq!(records[0]["error"]["kind"], "mask");
    assert_eq!(
        records[0]["error"]["message"],
        "The scene model is missing from this install. Reinstall Yellowmoon."
    );
    assert!(
        common::names_in(&sb.root().join("out")).is_empty(),
        "nothing is exported"
    );
}

#[test]
fn a_fit_adds_a_node_after_main_keeps_the_nodes_fields_and_reports_what_it_found() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("a.ARW", 256, 192, 11);
    let still = reference(&sb);

    // None: a node is added after Main, as Add node does; the strength, the skin protection and the report.
    let out = render(
        &sb,
        &arw,
        &[
            "--reference",
            arg(&still),
            "--reference-strength",
            "60",
            "--no-skin-protection",
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let records = json_records(&out.stdout);
    assert_eq!(records.len(), 1, "{}", text(&out.stdout));
    let r = &records[0];
    assert_eq!(r["status"], "ok", "{r}");
    let report = &r["reference"];
    assert_eq!(report["node"], 2);
    assert!(
        matches!(report["status"].as_str(), Some("fitted" | "already_close")),
        "{report}"
    );
    assert!(report["text"].as_str().is_some_and(|t| t.len() > 10));
    assert_eq!(report["strength"].as_f64(), Some(60.0));
    assert_eq!(report["skin_protection"], false);
    assert!(report["start"] == "neutral" || report["start"] == "look");
    assert!(report["time_ms"].as_u64().is_some());
    assert!(report["candidates"].as_u64().is_some_and(|n| n > 0));
    let elements = report["elements"].as_array().expect("elements");
    let names: Vec<&str> = elements
        .iter()
        .filter_map(|e| e["element"].as_str())
        .collect();
    assert_eq!(
        names,
        ["sky", "vegetation", "water", "ground", "built", "skin"]
    );
    for e in elements {
        for key in ["photo_share", "reference_share"] {
            let share = e[key].as_f64().expect("a share");
            assert!((0.0..=1.0).contains(&share), "{e}");
        }
    }
    // The settings of the report are the fitted ones: the node is in them, after Main, and the links go through it.
    let items = r["settings"]["nodes"]["items"].as_array().expect("items");
    let node = items.iter().find(|n| n["id"] == 2).expect("node 2");
    assert_eq!(node["kind"], "reference_grade");
    assert_eq!(node["strength"].as_f64(), Some(60.0));
    assert_eq!(node["skin_protection"], false);
    let links = r["settings"]["nodes"]["links"].as_array().expect("links");
    assert!(
        links
            .iter()
            .any(|l| l["from"]["node"] == 1 && l["to"]["node"] == 2)
    );
    assert!(
        links
            .iter()
            .any(|l| l["from"]["node"] == 2 && l["to"]["node"] == "output")
    );

    // One: the node of the file is fitted; what the person set stays (the trim, the name, the strength), and the fit's own fields are
    // written over. With the fitted file read back through `--settings`, the same bytes come out without `--reference`.
    let doc = graph(vec![json!({
        "id": 2, "kind": "reference_grade", "name": "Look", "pos": [1, 0], "strength": 50,
        "trim": { "exposure_ev": 0.25 } })]);
    let mut doc = doc;
    doc["nodes"]["links"] = json!([
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ]);
    let file = sb.root().join("one.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    let out = render(
        &sb,
        &arw,
        &["--settings", arg(&file), "--reference", arg(&still)],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let r = json_records(&out.stdout).remove(0);
    let node = r["settings"]["nodes"]["items"]
        .as_array()
        .expect("items")
        .iter()
        .find(|n| n["id"] == 2)
        .expect("node 2")
        .clone();
    assert_eq!(node["name"], "Look");
    assert_eq!(node["strength"].as_f64(), Some(50.0));
    assert_eq!(node["trim"]["exposure_ev"].as_f64(), Some(0.25));
    assert_eq!(r["reference"]["node"], 2);
    let pinned = sb.root().join("pinned.json");
    std::fs::write(&pinned, r["settings"].to_string()).expect("write");
    let again = sb
        .ym()
        .env(FAKE_SCENE.0, FAKE_SCENE.1)
        .args([
            "render",
            "--cpu",
            "--auto-provider",
            "yellowmoon",
            "--format",
            "png",
            "--name",
            "again",
        ])
        .arg(&arw)
        .args(["--settings", arg(&pinned), "--out"])
        .arg(sb.root().join("out"))
        .output()
        .expect("ym runs");
    assert_eq!(again.status.code(), Some(0), "{}", text(&again.stderr));
    let fitted_file = PathBuf::from(r["output"].as_str().expect("output"));
    let again_file = PathBuf::from(text(&again.stdout).trim());
    // The pixels are the same (the files differ in the metadata only: the name of the run, the document's own sentence of settings).
    let pixels = |file: &Path| {
        let decoder = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(file).expect("open"),
        ));
        let mut reader = decoder.read_info().expect("a PNG");
        let mut data = vec![0; reader.output_buffer_size().expect("a size")];
        let info = reader.next_frame(&mut data).expect("a frame");
        data.truncate(info.buffer_size());
        data
    };
    assert!(
        pixels(&fitted_file) == pixels(&again_file),
        "the pinned settings render the fitted picture"
    );

    // Two, with the node named: the other is left alone.
    let doc = {
        let mut d = graph(vec![
            json!({ "id": 2, "kind": "reference_grade", "pos": [1, 0] }),
            json!({ "id": 3, "kind": "reference_grade", "pos": [2, 0], "strength": 40 }),
        ]);
        d["nodes"]["links"] = json!([
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
            { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ]);
        d
    };
    let file = sb.root().join("two-linked.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    let out = render(
        &sb,
        &arw,
        &[
            "--settings",
            arg(&file),
            "--reference",
            arg(&still),
            "--reference-node",
            "3",
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let r = json_records(&out.stdout).remove(0);
    assert_eq!(r["reference"]["node"], 3);
    assert_eq!(r["reference"]["strength"].as_f64(), Some(40.0));
    let two = r["settings"]["nodes"]["items"]
        .as_array()
        .expect("items")
        .iter()
        .find(|n| n["id"] == 2)
        .expect("node 2")
        .clone();
    assert_eq!(
        two["hue_curves"],
        json!({ "hue_vs_hue": [], "hue_vs_sat": [] })
    );
}

#[test]
fn ctrl_c_during_the_fit_stops_it_and_exports_nothing() {
    use std::io::{BufRead, BufReader};
    let sb = Sandbox::new();
    let arw = sb.arw_sized("a.ARW", 256, 192, 11);
    let still = reference(&sb);
    // A plain `std` command with the sandbox's environment, so that the pipes can be read while it runs.
    let base = sb.ym();
    let mut command = std::process::Command::new(base.get_program());
    for (key, value) in base.get_envs() {
        match value {
            Some(value) => command.env(key, value),
            None => command.env_remove(key),
        };
    }
    let mut child = command
        .current_dir(sb.root())
        .env(FAKE_SCENE.0, FAKE_SCENE.1)
        .args([
            "render",
            "--cpu",
            "--auto-provider",
            "yellowmoon",
            "--verbose",
            "--format",
            "png",
        ])
        .arg(&arw)
        .args(["--reference", arg(&still), "--out"])
        .arg(sb.root().join("out"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("ym starts");
    // Read stderr until `--verbose` says the fit has begun, then give the search a moment to run.
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut seen = String::new();
    let deadline = Instant::now() + Duration::from_secs(120);
    while !seen.contains("fitting to") {
        let left = deadline.saturating_duration_since(Instant::now());
        let line = rx.recv_timeout(left).expect("the fit starts");
        seen.push_str(&line);
        seen.push('\n');
    }
    std::thread::sleep(Duration::from_millis(700));
    let started = Instant::now();
    std::process::Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("send Ctrl-C");
    let status = child.wait().expect("ym ends");
    reader.join().expect("reader");
    seen.extend(rx.try_iter().map(|l| l + "\n"));
    assert_eq!(status.code(), Some(130), "{seen}");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "it stops between candidates"
    );
    assert!(
        common::names_in(&sb.root().join("out")).is_empty(),
        "nothing is exported"
    );
    assert!(
        seen.contains("The fit was cancelled. Nothing was exported."),
        "{seen}"
    );
}

#[test]
fn doctor_lists_the_scene_models_weights_with_their_state() {
    let sb = Sandbox::new();
    // This copy of the data folder: the weights are there or not as the checkout has them; the folder without them says so.
    let data = sb.data_without_the_scene_weights();
    let out = sb
        .ym_data(&data)
        .args(["doctor", "--json"])
        .output()
        .expect("ym runs");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).expect("JSON");
    assert_eq!(v["scene_model"]["available"], false);
    assert_eq!(v["scene_model"]["skin"], false);
    assert_eq!(v["scene_model"]["segmenter"]["state"], "missing");
    assert_eq!(v["scene_model"]["segmenter"]["sha256"], Value::Null);
    assert_eq!(
        v["scene_model"]["segmenter"]["file"],
        "eomt/eomt_coco_panoptic_small.f16.safetensors"
    );
    assert_eq!(
        v["scene_model"]["skin_network"]["file"],
        "mediapipe/selfie_multiclass_256.f32.safetensors"
    );
    let text_out = sb.ym_data(&data).arg("doctor").output().expect("ym runs");
    let report = text(&text_out.stdout);
    assert!(
        report.contains("Scene model")
            && report.contains("not available (the segmenter's weights are missing)"),
        "{report}"
    );
    // Where the weights are (the checkout's own data folder), the row says ready, with the two hashes.
    let ready = sb.ym().arg("doctor").output().expect("ym runs");
    let report = text(&ready.stdout);
    let row = report
        .lines()
        .find(|l| l.starts_with("Scene model"))
        .expect("the row");
    assert!(
        row.contains("ready") || row.contains("not available"),
        "{row}"
    );
}
