//! End-to-end tests of `ym looks …` against an isolated app folder: list, import (with our own
//! synthetic `.cube` files; no ARRI file is involved), verify, default and bake.

mod common;

use common::{Sandbox, head, json_lines, lines, names_in, text};
use predicates::prelude::*;
use serde_json::Value;
use ym_color::alf::AlfFile;
use ym_color::cdl::Cdl;
use ym_color::lut::Lut3d;

/// An identity 2-point cube with the red channel scaled by `red` and the given title.
fn cube(title: &str, red: f64) -> String {
    let mut s = format!("TITLE \"{title}\"\nLUT_3D_SIZE 2\n");
    for b in 0..2 {
        for g in 0..2 {
            for r in 0..2 {
                s.push_str(&format!("{} {g} {b}\n", f64::from(r) * red));
            }
        }
    }
    s
}

fn write_cube(sb: &Sandbox, name: &str, title: &str, red: f64) -> std::path::PathBuf {
    let dir = sb.root().join("cubes");
    std::fs::create_dir_all(&dir).expect("folder");
    let path = dir.join(name);
    std::fs::write(&path, cube(title, red)).expect("write");
    path
}

#[test]
fn list_shows_the_built_in_looks_with_signature_as_the_default() {
    let sb = Sandbox::new();
    let out = sb.ym().args(["looks", "list"]).assert().success();
    let l = lines(&out.get_output().stdout);
    assert!(
        l[0].starts_with("ID")
            && l[0].contains("NAME")
            && l[0].contains("LIBRARY")
            && l[0].contains("INPUT"),
        "{l:?}"
    );
    // GROUP, then LIBRARY, then INPUT.
    let col = |header: &str| l[0].find(header).expect(header);
    assert!(
        col("GROUP") < col("LIBRARY") && col("LIBRARY") < col("INPUT"),
        "{l:?}"
    );
    assert!(
        l[1].starts_with("builtin:neutral")
            && l[1].contains("Neutral")
            && l[1].contains("built in")
            && l[1].contains("Built in")
            && l[1].contains("logc4")
            && !l[1].ends_with("default"),
        "{l:?}"
    );
    assert!(
        l[2].starts_with("builtin:signature")
            && l[2].contains("Signature")
            && l[2].contains("built in")
            && l[2].contains("Built in")
            && l[2].contains("logc4")
            && l[2].ends_with("default"),
        "{l:?}"
    );
    assert_eq!(l.len(), 3);
    // Listing creates nothing in the app folder.
    assert!(names_in(&sb.home()).is_empty());

    let out = sb.ym().args(["looks", "list", "--json"]).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["looks"][0]["id"], "builtin:neutral");
    assert_eq!(v["looks"][0]["is_default"], false);
    assert_eq!(v["looks"][1]["id"], "builtin:signature");
    assert_eq!(v["looks"][1]["is_default"], true);
    assert_eq!(v["looks"][1]["name"], "Signature");
    assert_eq!(v["looks"][1]["missing"], false);
    assert_eq!(v["looks"][1]["group"]["kind"], "built_in");
    // A built-in look is in the built-in library and has no file: no format, no size.
    for look in v["looks"].as_array().expect("looks") {
        assert_eq!(look["library"], "built_in", "{look}");
        assert!(
            look["format"].is_null() && look["lut_size"].is_null(),
            "{look}"
        );
        assert!(
            look.get("format").is_some() && look.get("lut_size").is_some(),
            "{look}"
        );
    }
    assert_eq!(v["display_luts"], serde_json::json!([]));
}

#[test]
fn a_cube_file_needs_the_input_question_answered() {
    let sb = Sandbox::new();
    let cube = write_cube(&sb, "warm.cube", "Warm", 0.9);
    sb.ym()
        .args(["looks", "import"])
        .arg(&cube)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "--input logc4, --input logc3 or --input logc3-rec709",
        ));
    assert!(
        !sb.home().join("looks").join("manifest.json").exists(),
        "nothing was imported"
    );

    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&cube)
        .assert()
        .success()
        .stdout(predicate::str::is_match(r"^user:[a-z0-9-]+\n$").expect("regex"))
        .stderr(predicate::str::contains("Imported 1 look."));
    // Importing it again changes nothing.
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&cube)
        .assert()
        .success()
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains(
            "Imported 0 looks, 1 already imported.",
        ));
}

#[test]
fn a_renamed_copy_of_an_imported_lut_is_not_imported_again() {
    let sb = Sandbox::new();
    let warm = write_cube(&sb, "warm.cube", "Warm", 0.9);
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&warm)
        .assert()
        .success()
        .stderr(predicate::str::contains("Imported 1 look."));
    // The same bytes under another name: the same LUT, said in plain words, nothing listed twice.
    let copy = sb.root().join("cubes").join("cool.cube");
    std::fs::copy(&warm, &copy).expect("copy");
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&copy)
        .assert()
        .success()
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains(
            "1 file is the same as a LUT that is already imported under another name, so it was not added again.",
        ))
        .stderr(predicate::str::contains(
            "Imported 0 looks, 1 already imported.",
        ));
    let out = sb.ym().args(["looks", "list", "--json"]).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["looks"].as_array().expect("looks").len(), 3, "{v}");
}

/// A look file written by our own writer from a synthetic LUT (no ARRI file is involved).
fn write_look_file(sb: &Sandbox, name: &str, kind: &str, cdl: Option<&Cdl>) -> std::path::PathBuf {
    let dir = sb.root().join("looks");
    std::fs::create_dir_all(&dir).expect("folder");
    let lut = Lut3d::bake(5, [0.0; 3], [1.0; 3], |c| {
        [c[0] * 0.8 + 0.1, c[1], c[2] * 0.9]
    })
    .expect("bake");
    let file = match kind {
        "alf4" => AlfFile::alf4(&lut, cdl, "", "look.cube"),
        "aml" => AlfFile::aml(&lut, cdl, "look.cube"),
        "alf4c" => AlfFile::alf4c(Some(&lut), cdl, Some(&lut), Some(&lut), ""),
        other => panic!("{other}"),
    };
    let path = dir.join(name);
    std::fs::write(&path, file.to_bytes().expect("write")).expect("write");
    path
}

#[test]
fn arri_look_files_import_without_the_input_question_and_keep_what_they_say() {
    let sb = Sandbox::new();
    let cdl = Cdl::new([1.2, 1.0, 0.9], [0.0; 3], [1.0; 3], 1.1).expect("cdl");
    let alf4 = write_look_file(&sb, "Warm Pull.alf4", "alf4", Some(&cdl));
    let aml = write_look_file(&sb, "Night.aml", "aml", None);
    // No `--input`: the files say what they expect.
    sb.ym()
        .args(["looks", "import"])
        .arg(&alf4)
        .arg(&aml)
        .assert()
        .success()
        .stdout("user:warm-pull\nuser:night\n")
        .stderr(predicate::str::contains("Imported 2 looks."));
    let out = sb.ym().args(["looks", "list", "--json"]).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    let looks = v["looks"].as_array().expect("looks");
    let find = |id: &str| looks.iter().find(|l| l["id"] == id).expect(id);
    assert_eq!(find("user:warm-pull")["input"], "logc4");
    assert_eq!(find("user:warm-pull")["name"], "Warm Pull");
    assert_eq!(find("user:warm-pull")["group"]["kind"], "user");
    assert_eq!(find("user:night")["input"], "logc3_to_rec709");
    // The person's own look files are in Your LUTs, with the kind of file and the size of their table.
    for (id, format) in [("user:warm-pull", "alf4"), ("user:night", "aml")] {
        assert_eq!(find(id)["library"], "user", "{id}");
        assert_eq!(find(id)["format"], format, "{id}");
        assert_eq!(find(id)["lut_size"], 5, "{id}");
    }
    // The table says the same, with the words of the CLI.
    let out = sb.ym().args(["looks", "list"]).assert().success();
    let shown = text(&out.get_output().stdout);
    assert!(
        shown.contains("logc3-rec709") && shown.contains("Warm Pull"),
        "{shown}"
    );
    // The copies keep the extension of the kind of file and verify clean.
    let names: Vec<String> = walk(&sb.home().join("looks"))
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    assert!(names.contains(&"Warm Pull.alf4".to_owned()), "{names:?}");
    assert!(names.contains(&"Night.aml".to_owned()), "{names:?}");
    sb.ym()
        .args(["looks", "verify"])
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Checked 2 imported files: all fine.",
        ));
    // An answer to the question is not needed and changes nothing.
    sb.ym()
        .args(["looks", "import", "--input", "logc3"])
        .arg(&alf4)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Imported 0 looks, 1 already imported.",
        ));

    // A look file renders: with the imported look, the picture is not the default's.
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args(["render", "--json", "--overwrite", "--look", "warm-pull"])
        .arg(&arw)
        .assert()
        .success();
    let o = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(o["look"]["id"], "user:warm-pull");
}

#[test]
fn an_alf4c_says_its_display_transforms_are_not_used_and_a_bad_look_file_says_why() {
    let sb = Sandbox::new();
    let alf4c = write_look_file(&sb, "Custom.alf4c", "alf4c", None);
    sb.ym()
        .args(["looks", "import"])
        .arg(&alf4c)
        .assert()
        .success()
        .stdout("user:custom\n")
        .stderr(predicate::str::contains(
            "Custom.alf4c: its own display transforms are not used; Yellowmoon's display rendering is.",
        ));
    // A damaged file: refused in plain words, nothing is added, exit 1.
    let mut bytes = std::fs::read(&alf4c).expect("read");
    let at = bytes.len() - 7;
    bytes[at] ^= 0x08;
    let damaged = sb.root().join("looks").join("Damaged.alf4c");
    std::fs::write(&damaged, bytes).expect("write");
    sb.ym()
        .args(["looks", "import"])
        .arg(&damaged)
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "Damaged.alf4c cannot be used. Its checksum does not match, so the file is damaged. Download it again.",
        ));
    // An .alf4c with no creative look, and an .aml made for another display.
    let lut = Lut3d::identity(3).expect("identity");
    let no_cmt = sb.root().join("looks").join("NoCmt.alf4c");
    std::fs::write(
        &no_cmt,
        AlfFile::alf4c(None, None, Some(&lut), Some(&lut), "")
            .to_bytes()
            .expect("write"),
    )
    .expect("write");
    sb.ym()
        .args(["looks", "import"])
        .arg(&no_cmt)
        .assert()
        .code(1)
        .stderr(predicate::str::contains("no creative look (CMT)"));
    let AlfFile::Aml(mut wide) = AlfFile::aml(&lut, None, "x") else {
        unreachable!()
    };
    wide.colorspace = "com.apple.proapps.colorspace.rec2100.d65.pq".to_owned();
    let pq = sb.root().join("looks").join("Wide.aml");
    std::fs::write(&pq, AlfFile::Aml(wide).to_bytes().expect("write")).expect("write");
    sb.ym()
        .args(["looks", "import"])
        .arg(&pq)
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "It is made for a Rec.2020 PQ display. Yellowmoon can use ARRI Look Files made for Rec.709.",
        ));
    // Only the first one is in the library.
    let out = sb.ym().args(["looks", "list", "--json"]).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["looks"].as_array().expect("looks").len(), 3);
}

#[test]
fn an_imported_look_is_listed_used_by_name_and_set_as_the_default() {
    let sb = Sandbox::new();
    let cube = write_cube(&sb, "warm.cube", "Warm", 0.9);
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&cube)
        .assert()
        .success();
    let out = sb.ym().args(["looks", "list", "--json"]).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    let looks = v["looks"].as_array().expect("looks");
    assert_eq!(looks.len(), 3, "Neutral, Signature and the import");
    let warm = &looks[2];
    let id = warm["id"].as_str().expect("id").to_owned();
    assert!(id.starts_with("user:"), "{id}");
    assert_eq!(
        warm["name"], "warm",
        "the name of an imported .cube is its file name"
    );
    assert_eq!(warm["group"]["kind"], "user");
    assert_eq!(warm["input"], "logc4");
    assert_eq!(warm["library"], "user");
    assert_eq!(warm["format"], "cube");
    assert!(warm["lut_size"].as_u64().is_some_and(|n| n >= 2), "{warm}");
    // The table has the LIBRARY column: Your LUTs.
    let out = sb.ym().args(["looks", "list"]).assert().success();
    assert!(
        lines(&out.get_output().stdout)
            .iter()
            .any(|l| l.starts_with(&id) && l.contains("Your LUTs") && l.contains("imported")),
        "{}",
        text(&out.get_output().stdout)
    );

    // --look takes the id, the slug and the name in any case.
    let arw = sb.arw("a.ARW");
    for query in [id.as_str(), id.trim_start_matches("user:"), "WARM"] {
        let out = sb
            .ym()
            .args(["render", "--json", "--overwrite", "--look", query])
            .arg(&arw)
            .assert()
            .success();
        let o = json_lines(&out.get_output().stdout).remove(0);
        assert_eq!(o["look"]["id"], id.as_str(), "--look {query}");
        assert!(
            o["output"]
                .as_str()
                .expect("output")
                .ends_with(&format!("a_{}.jpg", id.trim_start_matches("user:"))),
            "{}",
            o["output"]
        );
    }

    // Make it the default; a render without --look then starts from it.
    sb.ym()
        .args(["looks", "default", "warm"])
        .assert()
        .success()
        .stdout(format!("{id}\n"));
    let out = sb.ym().args(["looks", "list"]).assert().success();
    assert!(
        lines(&out.get_output().stdout)
            .iter()
            .any(|l| l.starts_with(&id) && l.ends_with("default")),
        "{}",
        text(&out.get_output().stdout)
    );
    let out = sb
        .ym()
        .args(["render", "--json", "--overwrite"])
        .arg(&arw)
        .assert()
        .success();
    assert_eq!(
        json_lines(&out.get_output().stdout)[0]["look"]["id"],
        id.as_str()
    );
    // The preference file is the only other thing written, and it holds the default look.
    let prefs: Value = serde_json::from_str(
        &std::fs::read_to_string(sb.home().join("preferences.json")).expect("prefs"),
    )
    .expect("JSON");
    assert_eq!(prefs["default_look"], id.as_str());
    sb.ym()
        .args(["looks", "default", "neutral"])
        .assert()
        .success();
}

#[test]
fn verify_finds_a_look_whose_file_has_gone_and_renders_with_it_exit_3() {
    let sb = Sandbox::new();
    let cube = write_cube(&sb, "warm.cube", "Warm", 0.9);
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&cube)
        .assert()
        .success();
    sb.ym()
        .args(["looks", "verify"])
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Checked 1 imported file: all fine.",
        ));

    // Remove the imported file behind the app's back.
    let looks_dir = sb.home().join("looks");
    let mut removed = 0;
    for entry in walk(&looks_dir) {
        if entry.extension().is_some_and(|e| e == "cube") {
            std::fs::remove_file(entry).expect("remove");
            removed += 1;
        }
    }
    assert_eq!(removed, 1);
    sb.ym()
        .args(["looks", "verify"])
        .assert()
        .code(1)
        .stdout(predicate::str::contains("missing"))
        .stderr(
            predicate::str::contains("1 of 1 imported file need attention")
                .or(predicate::str::contains("need attention")),
        );
    let arw = sb.arw("a.ARW");
    sb.ym()
        .args(["render", "--look", "warm"])
        .arg(&arw)
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "File missing or changed. Import again.",
        ));
    sb.ym().args(["looks", "default", "warm"]).assert().code(3);
    // The list marks it.
    sb.ym()
        .args(["looks", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("missing: import again"));
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

#[test]
fn import_of_something_that_is_not_a_look_fails_with_a_message() {
    let sb = Sandbox::new();
    let notes = sb.root().join("notes.txt");
    std::fs::write(&notes, "x").expect("write");
    sb.ym()
        .args(["looks", "import"])
        .arg(&notes)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "is not a look file (.cube, .alf4, .alf4c or .aml), a folder or a zip",
        ));
    sb.ym()
        .args(["looks", "import", "/nonexistent/looks.zip"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("was not found"));
}

#[test]
fn a_name_that_matches_no_look_exits_3_with_the_command_to_list_them() {
    let sb = Sandbox::new();
    sb.ym()
        .args(["looks", "default", "no-such"])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "No look called \"no-such\". Run `ym looks list`",
        ));
    // Two files with the same name become two looks, told apart by their ids.
    let a = write_cube(&sb, "film.cube", "Film", 0.9);
    let other = sb.root().join("more");
    std::fs::create_dir_all(&other).expect("folder");
    let b = other.join("film.cube");
    std::fs::write(&b, cube("Film", 0.8)).expect("write");
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&a)
        .arg(&b)
        .assert()
        .success()
        .stderr(predicate::str::contains("Imported 2 looks."));
    let out = sb.ym().args(["looks", "list", "--json"]).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    let ids: Vec<&str> = v["looks"]
        .as_array()
        .expect("looks")
        .iter()
        .map(|l| l["id"].as_str().expect("id"))
        .collect();
    assert_eq!(
        ids,
        [
            "builtin:neutral",
            "builtin:signature",
            "user:film",
            "user:film-2"
        ]
    );
}

#[test]
fn bake_writes_a_cube_and_a_contact_sheet() {
    let sb = Sandbox::new();
    let look = sb.root().join("tester.json");
    std::fs::write(
        &look,
        r#"{
  "schema": "yellowmoon.parametric-look/1",
  "name": "Tester",
  "input": "logc4",
  "cdl": { "slope": [1.08, 1.0, 0.92], "offset": [0, 0, 0], "power": [1, 1, 1], "saturation": 1.15 }
}"#,
    )
    .expect("write");
    let a = sb.arw("a.ARW");
    let b = sb.arw_sized("b.ARW", 120, 160, 4);
    let out_dir = sb.root().join("baked");
    let out = sb
        .ym()
        .args(["looks", "bake"])
        .arg(&look)
        .arg("--preview")
        .arg(&a)
        .arg(&b)
        .arg("--out")
        .arg(&out_dir)
        .assert()
        .success();
    assert_eq!(
        lines(&out.get_output().stdout),
        [
            out_dir.join("tester.cube").display().to_string(),
            out_dir.join("tester_contact.png").display().to_string()
        ]
    );
    let cube_text = std::fs::read_to_string(out_dir.join("tester.cube")).expect("cube");
    assert!(
        cube_text.starts_with("TITLE \"Tester\"\nLUT_3D_SIZE 33\n"),
        "{}",
        &cube_text[..60]
    );
    assert_eq!(cube_text.lines().count(), 4 + 33 * 33 * 33);
    // The sheet is a PNG with a row per photo and two columns.
    let png = out_dir.join("tester_contact.png");
    assert_eq!(head(&png, 4), [0x89, b'P', b'N', b'G']);
    let bytes = std::fs::read(&png).expect("png");
    let w = u32::from_be_bytes(bytes[16..20].try_into().expect("w"));
    let h = u32::from_be_bytes(bytes[20..24].try_into().expect("h"));
    assert!(w > 2 * 100 && h > 2 * 100, "{w} x {h}");
    // Nothing was written to the app folder, and the cube is a look the importer accepts: a
    // developer can bake, check the sheet and import the result as it is.
    assert!(names_in(&sb.home()).is_empty());
    sb.ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(out_dir.join("tester.cube"))
        .assert()
        .success()
        .stdout("user:tester\n");
}

#[test]
fn bake_without_previews_only_writes_the_cube_next_to_the_look() {
    let sb = Sandbox::new();
    let look = sb.root().join("plain.json");
    std::fs::write(
        &look,
        r#"{ "schema": "yellowmoon.parametric-look/1", "name": "Plain", "input": "logc4" }"#,
    )
    .expect("write");
    sb.ym()
        .args(["looks", "bake"])
        .arg(&look)
        .assert()
        .success();
    assert!(sb.root().join("plain.cube").exists());
    assert!(!sb.root().join("plain_contact.png").exists());
    // A look file that is not a parametric look is a usage error.
    let bad = sb.root().join("bad.json");
    std::fs::write(&bad, r#"{ "name": "x" }"#).expect("write");
    sb.ym()
        .args(["looks", "bake"])
        .arg(&bad)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("bad.json"));
    sb.ym()
        .args(["looks", "bake", "/nonexistent/look.json"])
        .assert()
        .code(2);
}

/// `ym looks default x | head -n 1`: the reader is gone before the id is written. The command has
/// done its work, so it ends with 0 and without a panic message (`println!` would panic, exit 101).
#[cfg(unix)]
#[test]
fn a_reader_that_goes_away_does_not_make_a_looks_command_panic() {
    use std::process::Stdio;

    let sb = Sandbox::new();
    let cube = write_cube(&sb, "warm.cube", "Warm", 0.9);
    for args in [
        vec!["looks", "default", "neutral"],
        vec!["looks", "verify"],
        vec!["looks", "list"],
    ] {
        let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
            .args(&args)
            .env("YELLOWMOON_HOME", sb.home())
            .env("YELLOWMOON_DATA", common::data_dir())
            .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn ym");
        drop(child.stdout.take());
        let output = child.wait_with_output().expect("wait");
        let stderr = text(&output.stderr);
        assert!(!stderr.contains("panicked"), "{args:?}: {stderr}");
        assert_eq!(output.status.code(), Some(0), "{args:?}: {stderr}");
    }
    // The importer prints the id of every look it adds, also to a reader that has gone.
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
        .args(["looks", "import", "--input", "logc4"])
        .arg(&cube)
        .env("YELLOWMOON_HOME", sb.home())
        .env("YELLOWMOON_DATA", common::data_dir())
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ym");
    drop(child.stdout.take());
    let output = child.wait_with_output().expect("wait");
    let stderr = text(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    sb.ym()
        .args(["looks", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("user:warm"));
}

#[test]
fn a_damaged_look_file_in_a_folder_is_said_and_the_rest_still_imports_with_exit_1() {
    let sb = Sandbox::new();
    let good = write_look_file(&sb, "Good.alf4", "alf4", None);
    let mut bytes = std::fs::read(&good).expect("read");
    let at = bytes.len() - 7;
    bytes[at] ^= 0x08;
    std::fs::write(sb.root().join("looks").join("Bad.alf4"), bytes).expect("write");

    let out = sb
        .ym()
        .args(["looks", "import"])
        .arg(sb.root().join("looks"))
        .assert()
        .code(1)
        .stdout("user:good\n")
        .stderr(predicate::str::contains(
            "Bad.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again.",
        ));
    // The good one is in the library, the damaged one is not.
    let _ = out;
    let list = sb.ym().args(["looks", "list"]).assert().success();
    let l = lines(&list.get_output().stdout);
    assert!(l.iter().any(|row| row.starts_with("user:good")), "{l:?}");
    assert!(l.iter().all(|row| !row.starts_with("user:bad")), "{l:?}");
    // `--quiet` keeps the refusal: it is an error, not a remark.
    let quiet = sb.root().join("quiet");
    std::fs::create_dir_all(&quiet).expect("folder");
    std::fs::copy(
        sb.root().join("looks").join("Bad.alf4"),
        quiet.join("Bad.alf4"),
    )
    .expect("copy");
    std::fs::copy(
        sb.root().join("looks").join("Good.alf4"),
        quiet.join("Fine.alf4"),
    )
    .expect("copy");
    sb.ym()
        .args(["--quiet", "looks", "import"])
        .arg(&quiet)
        .assert()
        .code(1)
        .stderr(predicate::str::contains("Bad.alf4 cannot be used."));
    // A folder without a refusal still exits 0.
    let fine = sb.root().join("fine");
    std::fs::create_dir_all(&fine).expect("folder");
    std::fs::copy(
        sb.root().join("looks").join("Good.alf4"),
        fine.join("Third.alf4"),
    )
    .expect("copy");
    sb.ym()
        .args(["looks", "import"])
        .arg(&fine)
        .assert()
        .success();
}
