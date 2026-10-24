//! `ym lenses list` and the Lens options of `ym render` (`docs/21` sections 2 and 8), against a
//! synthetic lens folder (`ym_lens::test_support`): `data/lenses` is the lens data of the app, and the
//! tests do not depend on what it holds.

mod common;

use common::{Sandbox, json_lines, lines, text};
use predicates::prelude::*;
use serde_json::{Value, json};
use ym_core::app::{Core, CoreOptions};
use ym_lens::{LensInfo, LensKind};

/// `ym render --json` with `args` on the sandbox's photo, over the data folder `data`, and the
/// `settings` object of its answer.
fn settings_of(sb: &Sandbox, data: &std::path::Path, args: &[&str]) -> Value {
    let arw = sb.photos().join("a.ARW");
    let out = sb
        .ym_data(data)
        .args(["render", "--json", "--apple-auto", "off", "--out"])
        .arg(sb.root().join("out"))
        .args(args)
        .arg(&arw)
        .assert()
        .success();
    let mut objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 1, "{objects:?}");
    objects.remove(0)["settings"].clone()
}

fn lens_of(sb: &Sandbox, data: &std::path::Path, args: &[&str]) -> Value {
    settings_of(sb, data, args)["lens"].clone()
}

#[test]
fn the_list_shows_the_seven_lenses_and_its_json_is_the_cores() {
    let sb = Sandbox::new();
    let data = sb.data_with_lenses(true);
    let out = sb.ym_data(&data).args(["lenses", "list"]).assert().success();
    let l = lines(&out.get_output().stdout);
    assert_eq!(l.len(), 8, "{l:?}");
    for header in ["ID", "NAME", "FLARES", "EDGES", "DESIGN", "LINE"] {
        assert!(l[0].contains(header), "{header}: {l:?}");
    }
    let ids: Vec<&str> = l[1..]
        .iter()
        .map(|row| row.split_whitespace().next().expect("an id"))
        .collect();
    assert_eq!(
        ids,
        [
            "default", "classic", "vintage", "petzval", "triplet", "sonnar", "detuned"
        ]
    );

    // The JSON is what the core answers to the app's `lens_list`, field for field.
    let out = sb
        .ym_data(&data)
        .args(["lenses", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    // Listing writes nothing.
    assert!(common::names_in(&sb.home()).is_empty());
    let listed: Vec<LensInfo> = serde_json::from_value(v["lenses"].clone()).expect("LensInfo list");
    let mut options = CoreOptions::new(data.clone(), sb.home());
    options.thumbnails = false;
    options.decode_threads = 1;
    let core = Core::open(options).expect("the core opens");
    assert_eq!(listed, core.lens_list());
    assert_eq!(listed.len(), 7);
    assert_eq!(v["lenses"][0]["kind"], "default");
    for key in [
        "kind",
        "name",
        "line",
        "start_flares",
        "start_edges",
        "version",
        "design_source",
    ] {
        assert!(v["lenses"][2].get(key).is_some(), "{key}: {}", v["lenses"][2]);
    }
}

#[test]
fn a_missing_lens_folder_is_a_plain_message_and_exit_code_2() {
    let sb = Sandbox::new();
    let data = sb.data_with_lenses(false);
    sb.ym_data(&data)
        .args(["lenses", "list"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("The lens data could not be loaded"));
    // --lens needs the starting amounts; the amounts alone do not.
    sb.arw("a.ARW");
    sb.ym_data(&data)
        .args(["render", "--lens", "vintage"])
        .arg(sb.photos().join("a.ARW"))
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--lens can't be used"));
}

#[test]
fn a_lens_brings_its_starting_amounts_and_an_amount_overrides_them() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    let data = sb.data_with_lenses(true);
    let library = ym_lens::LensLibrary::from_dir(&data.join("lenses")).expect("library");
    let vintage = library
        .lenses()
        .iter()
        .find(|l| l.kind == LensKind::Vintage)
        .expect("vintage");
    let (flares, edges) = (vintage.start_flares, vintage.start_edges);
    assert!(
        flares > 0.0 && edges > 0.0,
        "the test lens starts above 0: {vintage:?}"
    );

    // No option: the default lens, no amounts.
    assert_eq!(
        lens_of(&sb, &data, &[]),
        json!({ "kind": "default", "flares": 0.0, "edges": 0.0, "anamorphic": 0.0 })
    );
    // The lens alone: its starting Flares and Edges, as the card sends them; Anamorphic stays.
    assert_eq!(
        lens_of(&sb, &data, &["--lens", "vintage"]),
        json!({ "kind": "vintage", "flares": flares, "edges": edges, "anamorphic": 0.0 })
    );
    // An amount beats the lens's start, for that amount only.
    assert_eq!(
        lens_of(&sb, &data, &["--lens", "vintage", "--flares", "10"]),
        json!({ "kind": "vintage", "flares": 10.0, "edges": edges, "anamorphic": 0.0 })
    );
    assert_eq!(
        lens_of(
            &sb,
            &data,
            &["--lens", "Vintage", "--edges", "0", "--anamorphic", "35.5"]
        ),
        json!({ "kind": "vintage", "flares": flares, "edges": 0.0, "anamorphic": 35.5 })
    );
    // An amount without a lens changes that amount of the Default lens (or of the settings file's).
    assert_eq!(
        lens_of(&sb, &data, &["--flares", "10"]),
        json!({ "kind": "default", "flares": 10.0, "edges": 0.0, "anamorphic": 0.0 })
    );
    // Default starts at 0 and 0.
    assert_eq!(
        lens_of(&sb, &data, &["--lens", "default", "--anamorphic", "50"]),
        json!({ "kind": "default", "flares": 0.0, "edges": 0.0, "anamorphic": 50.0 })
    );
}

#[test]
fn the_options_and_a_settings_file_say_the_same_and_the_options_win() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    let data = sb.data_with_lenses(true);
    let file = sb.root().join("lens.json");
    std::fs::write(
        &file,
        json!({ "lens": { "kind": "petzval", "flares": 60, "edges": 45, "anamorphic": 30 } })
            .to_string(),
    )
    .expect("write");
    let from_file = lens_of(&sb, &data, &["--settings", file.to_str().expect("utf-8")]);
    assert_eq!(
        from_file,
        json!({ "kind": "petzval", "flares": 60.0, "edges": 45.0, "anamorphic": 30.0 })
    );
    // The same settings given as options.
    let from_options = lens_of(
        &sb,
        &data,
        &[
            "--lens",
            "petzval",
            "--flares",
            "60",
            "--edges",
            "45",
            "--anamorphic",
            "30",
        ],
    );
    assert_eq!(from_options, from_file);
    // An option beats the file: the amount alone, and the lens with its starting amounts.
    let beaten = lens_of(
        &sb,
        &data,
        &["--settings", file.to_str().expect("utf-8"), "--flares", "5"],
    );
    assert_eq!(
        beaten,
        json!({ "kind": "petzval", "flares": 5.0, "edges": 45.0, "anamorphic": 30.0 })
    );
    // A file may hold only the parts it changes, and an amount out of range is limited with a warning.
    std::fs::write(&file, r#"{ "lens": { "flares": 250 } }"#).expect("write");
    let arw = sb.photos().join("a.ARW");
    let out = sb
        .ym_data(&data)
        .args(["render", "--json", "--out"])
        .arg(sb.root().join("out"))
        .arg("--settings")
        .arg(&file)
        .arg(&arw)
        .assert()
        .success();
    let o = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(
        o["settings"]["lens"],
        json!({ "kind": "default", "flares": 100.0, "edges": 0.0, "anamorphic": 0.0 })
    );
    assert!(text(&out.get_output().stderr).contains("lens"));
    // An unknown lens or field in the file is refused.
    for bad in [
        r#"{ "lens": { "kind": "leica" } }"#,
        r#"{ "lens": { "grain": 3 } }"#,
    ] {
        std::fs::write(&file, bad).expect("write");
        sb.ym_data(&data)
            .args(["render", "--out"])
            .arg(sb.root().join("out"))
            .arg("--settings")
            .arg(&file)
            .arg(&arw)
            .assert()
            .code(2);
    }
}

#[test]
fn lens_options_are_checked_like_every_number_option() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let data = sb.data_with_lenses(true);
    for args in [
        ["--flares", "101"],
        ["--edges", "-1"],
        ["--anamorphic", "lots"],
        ["--lens", "leica"],
    ] {
        sb.ym_data(&data)
            .arg("render")
            .args(args)
            .arg(&arw)
            .assert()
            .code(2);
    }
    // The help names the lenses and the amounts under their own heading.
    let out = sb.ym().args(["render", "--help"]).assert().success();
    let help = text(&out.get_output().stdout);
    for word in ["Lens:", "--lens <LENS>", "--flares <N>", "--edges <N>", "--anamorphic <N>", "detuned"] {
        assert!(help.contains(word), "{word}");
    }
}
