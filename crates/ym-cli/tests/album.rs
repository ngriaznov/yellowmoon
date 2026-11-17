//! `ym render <album.ymalbum>` and `ym info <album.ymalbum>` (`docs/31` section 8, P-14): the photos in album order with their
//! stored settings, the presets and flags over them, the refusals, a photo that is not found, and the listing.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use common::{Sandbox, data_dir, field_patch, json_lines, lines, node, text};
use predicates::prelude::*;
use serde_json::{Value, json};
use ym_core::app::{Core, CoreOptions};
use ym_core::session::{OpenOptions, PhotoId};
use ym_core::settings::SettingsPatch;

/// Saves an album of `files` (in that order) with the core, `edit` giving each photo its settings.
pub fn make_album(files: &[&Path], album: &Path, edit: impl Fn(&Core, &[PhotoId])) {
    let home = tempfile::tempdir().expect("tempdir");
    let mut options = CoreOptions::new(data_dir(), home.path().to_path_buf());
    options.thumbnails = false;
    let core = Core::open(options).expect("core");
    let paths: Vec<PathBuf> = files.iter().map(|p| p.to_path_buf()).collect();
    let mut ids = Vec::new();
    for p in &paths {
        let o = core
            .session
            .open(std::slice::from_ref(p), OpenOptions::default());
        ids.extend(o.added);
    }
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    edit(&core, &ids);
    core.album_save(album).expect("album saved");
}

fn patch(fields: Value) -> SettingsPatch {
    serde_json::from_value(field_patch(fields)).expect("patch")
}

fn render_args() -> [&'static str; 8] {
    [
        "render",
        "--json",
        "--cpu",
        "--auto-provider",
        "yellowmoon",
        "--format",
        "png",
        "--bits",
    ]
}

fn render(sb: &Sandbox, extra: &[&str], album: &Path) -> assert_cmd::assert::Assert {
    sb.ym()
        .args(render_args())
        .arg("8")
        .arg("--out")
        .arg(sb.root().join("out"))
        .args(extra)
        .arg(album)
        .assert()
}

/// Two photos saved in the order b, a (not the order of their names), with manual exposures.
fn two_photo_album(sb: &Sandbox) -> (PathBuf, PathBuf, PathBuf) {
    let a = sb.arw("a.ARW");
    let b = sb.arw_sized("b.ARW", 128, 96, 7);
    let album = sb.root().join("trip.ymalbum");
    make_album(&[&b, &a], &album, |core, ids| {
        core.session
            .update_settings(
                ids[0],
                &patch(json!({ "exposure": { "kind": "manual", "ev": 0.75 } })),
            )
            .expect("settings");
        core.session
            .update_settings(ids[1], &patch(json!({ "contrast": 25.0 })))
            .expect("settings");
    });
    (a, b, album)
}

#[test]
fn an_album_renders_in_album_order_with_each_photos_stored_settings() {
    let sb = Sandbox::new();
    let (_a, _b, album) = two_photo_album(&sb);
    let out = render(&sb, &[], &album).success();
    let r = json_lines(&out.get_output().stdout);
    assert_eq!(r.len(), 2);
    assert!(
        r[0]["input"].as_str().expect("input").ends_with("b.ARW"),
        "{r:?}"
    );
    assert!(r[1]["input"].as_str().expect("input").ends_with("a.ARW"));
    assert_eq!(
        node(&r[0], "input")["exposure"],
        json!({ "kind": "manual", "ev": 0.75 })
    );
    assert_eq!(node(&r[1], 1)["grade"]["contrast"]["value"], 25.0);
    assert_ne!(
        node(&r[1], "input")["exposure"]["ev"],
        0.75,
        "a's own settings"
    );
}

#[test]
fn presets_and_flags_go_over_the_stored_settings_and_a_flag_has_the_last_word() {
    let sb = Sandbox::new();
    let (_a, _b, album) = two_photo_album(&sb);
    let doc = common::field_doc(
        json!({ "exposure": { "kind": "manual", "ev": -1.0 }, "contrast": 40.0 }),
    );
    let file = sb.root().join("dark.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    sb.ym()
        .args(["presets", "save", "Dark", "--from"])
        .arg(&file)
        .assert()
        .success();
    let out = render(&sb, &["--preset", "dark", "--contrast", "5"], &album).success();
    let r = json_lines(&out.get_output().stdout);
    for rec in &r {
        assert_eq!(
            node(rec, "input")["exposure"],
            json!({ "kind": "manual", "ev": -1.0 }),
            "the preset's"
        );
        assert_eq!(
            node(rec, 1)["grade"]["contrast"]["value"],
            5.0,
            "the flag's"
        );
        assert_eq!(rec["presets"][0]["name"], "Dark");
    }
}

#[test]
fn an_album_with_settings_or_with_photos_is_a_usage_error() {
    let sb = Sandbox::new();
    let (a, _b, album) = two_photo_album(&sb);
    let settings = sb.root().join("s.json");
    std::fs::write(&settings, "{}").expect("write");
    sb.ym()
        .args(["render", "--cpu", "--settings"])
        .arg(&settings)
        .arg(&album)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "The album already has the settings of its photos.",
        ));
    sb.ym()
        .args(["render", "--cpu"])
        .arg(&album)
        .arg(&a)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "Give an album on its own, or photos.",
        ));
    sb.ym()
        .args(["info"])
        .arg(&album)
        .arg(&a)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "Give an album on its own, or photos.",
        ));
    assert!(!sb.root().join("out").exists());
}

#[test]
fn a_file_that_is_not_an_album_or_is_too_new_is_exit_2_with_its_sentence() {
    let sb = Sandbox::new();
    let junk = sb.root().join("junk.ymalbum");
    std::fs::write(&junk, "hello").expect("write");
    sb.ym()
        .args(["render", "--cpu"])
        .arg(&junk)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "This file isn't a Yellowmoon album.",
        ));
    let newer = sb.root().join("newer.ymalbum");
    std::fs::write(
        &newer,
        json!({ "yellowmoon_album": 2, "photos": [] }).to_string(),
    )
    .expect("write");
    sb.ym()
        .args(["info"])
        .arg(&newer)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "This album was made by a newer version of Yellowmoon. Update to open it.",
        ));
    sb.ym()
        .args(["render", "--cpu"])
        .arg(sb.root().join("nowhere.ymalbum"))
        .assert()
        .code(2);
}

#[test]
fn a_photo_that_is_not_found_fails_alone_and_none_found_is_exit_5() {
    let sb = Sandbox::new();
    let (a, b, album) = two_photo_album(&sb);
    // b has moved and is not where either path says.
    std::fs::rename(&b, sb.root().join("moved.ARW")).expect("move");
    let out = render(&sb, &[], &album).code(1);
    let r = json_lines(&out.get_output().stdout);
    assert_eq!(r.len(), 2);
    assert_eq!(r[0]["status"], "error");
    assert_eq!(r[0]["error"]["kind"], "not_found");
    assert!(
        r[0]["error"]["message"]
            .as_str()
            .expect("message")
            .starts_with("Not found: "),
        "{r:?}"
    );
    assert_eq!(r[1]["status"], "ok");
    assert!(r[1]["input"].as_str().expect("input").ends_with("a.ARW"));
    std::fs::rename(&a, sb.root().join("moved-a.ARW")).expect("move");
    render(&sb, &[], &album).code(5);
}

#[test]
fn a_file_that_changed_since_the_album_was_saved_renders_and_says_so() {
    let sb = Sandbox::new();
    let (a, _b, album) = two_photo_album(&sb);
    // Another file at a's place: the same name, other bytes.
    sb.arw_sized("a.ARW", 160, 120, 99);
    assert!(a.is_file());
    let out = render(&sb, &[], &album).success();
    let r = json_lines(&out.get_output().stdout);
    let notes = |v: &Value| -> Vec<String> {
        v["notes"]
            .as_array()
            .expect("notes")
            .iter()
            .map(|n| n.as_str().expect("note").to_owned())
            .collect()
    };
    let changed = "This file has changed since the album was saved.".to_owned();
    assert!(notes(&r[1]).contains(&changed), "{r:?}");
    assert!(!notes(&r[0]).contains(&changed), "{r:?}");
}

#[test]
fn info_lists_the_album_and_the_photos_that_were_found() {
    let sb = Sandbox::new();
    let (_a, b, album) = two_photo_album(&sb);
    std::fs::rename(&b, sb.root().join("moved.ARW")).expect("move");
    let out = sb
        .ym()
        .args(["info", "--auto-provider", "yellowmoon"])
        .arg(&album)
        .assert()
        .success();
    let l = lines(&out.get_output().stdout);
    assert!(
        l[0].starts_with("Album: ") && l[0].ends_with("trip.ymalbum"),
        "{l:?}"
    );
    assert!(l[1].starts_with("Written by: Yellowmoon "), "{l:?}");
    assert_eq!(l[2], "Photos: 2");
    assert!(
        l[3].contains("1. b.ARW") && l[3].ends_with("not found"),
        "{l:?}"
    );
    assert!(
        l[4].contains("2. a.ARW") && l[4].ends_with("found"),
        "{l:?}"
    );
    assert!(text(&out.get_output().stdout).contains("a.ARW"));
    let out = sb
        .ym()
        .args(["info", "--json", "--auto-provider", "yellowmoon"])
        .arg(&album)
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout);
    assert_eq!(v.len(), 2, "the album's line, then the photo found");
    assert_eq!(v[0]["photos"][0]["found"], false);
    assert_eq!(v[0]["photos"][1]["found"], true);
    assert_eq!(v[0]["photos"][1]["fingerprint_matches"], true);
    assert_eq!(v[0]["photos"][1]["picked"], false);
    assert!(v[0]["written_by"]["version"].is_string());
}
