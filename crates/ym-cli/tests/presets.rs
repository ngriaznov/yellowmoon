//! `ym presets …` and `ym render --preset` against an isolated app folder (`docs/30` section 11, P-03): each subcommand, `--json`,
//! the exit codes, a preset read from a file without importing it, the layering (`--settings`, then the presets in the order
//! given, then the flags), a missing look that warns and exits 0, and `render --json`'s `presets`.

mod common;

use std::path::{Path, PathBuf};

use common::{Sandbox, field_doc, json_lines, lines, names_in, node, text};
use predicates::prelude::*;
use serde_json::{Value, json};

/// A settings file in the sandbox.
fn settings_file(sb: &Sandbox, name: &str, doc: &Value) -> PathBuf {
    let path = sb.root().join(name);
    std::fs::write(&path, doc.to_string()).expect("write the settings");
    path
}

/// Saves a preset from a settings document with `ym presets save`.
fn save(sb: &Sandbox, name: &str, doc: &Value) {
    let file = settings_file(sb, &format!("{name}.json"), doc);
    sb.ym()
        .args(["presets", "save", name, "--from"])
        .arg(&file)
        .assert()
        .success();
}

fn exposure(ev: f64) -> Value {
    json!({ "kind": "manual", "ev": ev })
}

/// Light: a manual exposure and contrast.
fn light_doc() -> Value {
    field_doc(json!({ "exposure": exposure(0.5), "contrast": 20.0 }))
}

/// Detail: a manual sharpening.
fn detail_doc() -> Value {
    field_doc(json!({ "sharpen": { "kind": "manual", "amount": 90.0, "radius_px": 1.0 } }))
}

/// The JSON records of `ym render --json --cpu … ` for one photo with the arguments given.
fn render_json(sb: &Sandbox, arw: &Path, args: &[&str]) -> Vec<Value> {
    let out = sb
        .ym()
        .args(["render", "--json", "--cpu", "--auto-provider", "yellowmoon"])
        .args(["--format", "png", "--bits", "8", "--out"])
        .arg(sb.root().join("out"))
        .arg(arw)
        .args(args)
        .assert()
        .success();
    json_lines(&out.get_output().stdout)
}

#[test]
fn an_empty_folder_lists_no_presets_and_creates_nothing() {
    let sb = Sandbox::new();
    let out = sb.ym().args(["presets", "list"]).assert().success();
    let l = lines(&out.get_output().stdout);
    assert_eq!(l.len(), 1, "{l:?}");
    assert!(l[0].starts_with("NAME") && l[0].contains("GROUPS") && l[0].contains("LOOK"));
    let out = sb
        .ym()
        .args(["presets", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["presets"], json!([]));
    assert_eq!(v["damaged"], json!([]));
    assert!(
        v["folder"].as_str().expect("folder").ends_with("presets"),
        "{v}"
    );
    assert!(names_in(&sb.home()).is_empty(), "listing writes nothing");
}

#[test]
fn save_makes_a_preset_of_the_groups_that_differ_and_list_shows_it() {
    let sb = Sandbox::new();
    let file = settings_file(&sb, "warm.json", &light_doc());
    let out = sb
        .ym()
        .args(["presets", "save", "Warm evening", "--from"])
        .arg(&file)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Saved the preset Warm evening (light).",
        ));
    assert_eq!(text(&out.get_output().stdout).trim(), "Warm evening");
    assert_eq!(
        names_in(&sb.home().join("presets")),
        [".lock", "Warm evening.ympreset"]
    );

    let out = sb.ym().args(["presets", "list"]).assert().success();
    let l = lines(&out.get_output().stdout);
    assert_eq!(l.len(), 2, "{l:?}");
    assert!(
        l[1].starts_with("Warm evening") && l[1].contains("light") && l[1].trim_end().ends_with('-'),
        "{l:?}"
    );

    let out = sb
        .ym()
        .args(["presets", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(
        v["presets"],
        json!([{
            "name": "Warm evening",
            "file": "Warm evening.ympreset",
            "groups": ["light"],
            "look": null,
            "look_missing": false,
        }])
    );
}

#[test]
fn groups_are_chosen_with_dash_dash_groups_and_unknown_ones_are_usage_errors() {
    let sb = Sandbox::new();
    let file = settings_file(&sb, "both.json", &field_doc(json!({
        "exposure": exposure(0.5),
        "sharpen": { "kind": "manual", "amount": 90.0, "radius_px": 1.0 },
    })));
    // Both groups differ; only Detail is asked for.
    sb.ym()
        .args(["presets", "save", "Crisp", "--groups", "detail", "--from"])
        .arg(&file)
        .assert()
        .success();
    let out = sb
        .ym()
        .args(["presets", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["presets"][0]["groups"], json!(["detail"]));
    // The UI's spelling of the group is accepted, and the list may be in any order.
    sb.ym()
        .args(["presets", "save", "Both", "--groups", "light,colour", "--from"])
        .arg(&file)
        .assert()
        .success();
    let out = sb
        .ym()
        .args(["presets", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["presets"][0]["name"], "Both");
    assert_eq!(v["presets"][0]["groups"], json!(["light", "color"]));
    sb.ym()
        .args(["presets", "save", "Bad", "--groups", "tone", "--from"])
        .arg(&file)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("is not a group"));
}

#[test]
fn nothing_that_differs_from_the_defaults_needs_the_groups_to_be_chosen() {
    let sb = Sandbox::new();
    let file = settings_file(&sb, "plain.json", &field_doc(json!({})));
    sb.ym()
        .args(["presets", "save", "Plain", "--from"])
        .arg(&file)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "Nothing in this file differs from the default settings. Choose the groups with --groups.",
        ));
    assert!(!sb.home().join("presets").exists(), "nothing was written");
    sb.ym()
        .args(["presets", "save", "Plain", "--groups", "look", "--from"])
        .arg(&file)
        .assert()
        .success();
}

#[test]
fn a_name_that_exists_is_a_usage_error_unless_replace_is_given() {
    let sb = Sandbox::new();
    save(&sb, "Warm evening", &light_doc());
    let other = settings_file(
        &sb,
        "other.json",
        &field_doc(json!({ "exposure": exposure(-1.0) })),
    );
    let before = std::fs::read(sb.home().join("presets/Warm evening.ympreset")).expect("file");
    // The same name in another case is the same name.
    sb.ym()
        .args(["presets", "save", "warm EVENING", "--from"])
        .arg(&other)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "A preset called warm EVENING already exists. Use --replace to write over it.",
        ));
    assert_eq!(
        std::fs::read(sb.home().join("presets/Warm evening.ympreset")).expect("file"),
        before,
        "the preset is as it was"
    );
    sb.ym()
        .args(["presets", "save", "Warm evening", "--replace", "--from"])
        .arg(&other)
        .assert()
        .success();
    assert_ne!(
        std::fs::read(sb.home().join("presets/Warm evening.ympreset")).expect("file"),
        before
    );
}

#[test]
fn a_file_that_is_not_there_and_a_name_that_is_not_valid_are_usage_errors() {
    let sb = Sandbox::new();
    sb.ym()
        .args(["presets", "save", "X", "--from", "nowhere.json"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("There is no file with this name"));
    let file = settings_file(&sb, "warm.json", &light_doc());
    sb.ym()
        .args(["presets", "save", "   ", "--from"])
        .arg(&file)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Give the preset a name."));
    let long = "x".repeat(65);
    sb.ym()
        .args(["presets", "save", &long, "--from"])
        .arg(&file)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Use 64 characters or fewer."));
}

#[test]
fn export_import_and_remove_work_on_the_same_folder_as_the_app() {
    let sb = Sandbox::new();
    save(&sb, "Warm evening", &light_doc());
    let file = sb.root().join("shared.ympreset");
    sb.ym()
        .args(["presets", "export", "warm evening"])
        .arg(&file)
        .assert()
        .success();
    assert_eq!(
        std::fs::read(&file).expect("exported"),
        std::fs::read(sb.home().join("presets/Warm evening.ympreset")).expect("saved"),
        "the same bytes"
    );
    // An export over a file that is there writes over it, like `cp`.
    std::fs::write(&file, "old").expect("write");
    sb.ym()
        .args(["presets", "export", "Warm evening"])
        .arg(&file)
        .assert()
        .success();
    assert!(text(&std::fs::read(&file).expect("read")).starts_with('{'));
    // A preset that is not there.
    sb.ym()
        .args(["presets", "export", "Nothing"])
        .arg(sb.root().join("x.ympreset"))
        .assert()
        .code(3)
        .stderr(predicate::str::contains("There is no preset called Nothing."));

    // Remove moves it to the Trash (the file is gone from the folder).
    sb.ym()
        .args(["presets", "remove", "WARM EVENING"])
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Moved the preset Warm evening to the Trash.",
        ));
    assert!(!sb.home().join("presets/Warm evening.ympreset").exists());
    sb.ym()
        .args(["presets", "remove", "Warm evening"])
        .assert()
        .code(3);

    // Import puts it back; the name that exists is refused unless --replace.
    let out = sb
        .ym()
        .args(["presets", "import"])
        .arg(&file)
        .assert()
        .success()
        .stderr(predicate::str::contains("Added the preset Warm evening."));
    assert_eq!(text(&out.get_output().stdout).trim(), "Warm evening");
    sb.ym()
        .args(["presets", "import"])
        .arg(&file)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "A preset called Warm evening already exists. Use --replace to write over it.",
        ));
    sb.ym()
        .args(["presets", "import", "--replace"])
        .arg(&file)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Replaced the preset Warm evening.",
        ));
}

#[test]
fn import_says_why_a_file_is_refused() {
    let sb = Sandbox::new();
    let junk = sb.root().join("junk.ympreset");
    std::fs::write(&junk, "hello").expect("write");
    sb.ym()
        .args(["presets", "import"])
        .arg(&junk)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("This file isn't a Yellowmoon preset."));
    let newer = sb.root().join("newer.ympreset");
    std::fs::write(
        &newer,
        json!({ "yellowmoon_preset": 2, "name": "N", "groups": ["light"], "settings": {} })
            .to_string(),
    )
    .expect("write");
    sb.ym()
        .args(["presets", "import"])
        .arg(&newer)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "This preset was made by a newer version of Yellowmoon. Update to use it.",
        ));
    // The good file of the same call still comes in, and the run says some failed.
    save(&sb, "Good", &light_doc());
    let good = sb.root().join("good.ympreset");
    sb.ym()
        .args(["presets", "export", "Good"])
        .arg(&good)
        .assert()
        .success();
    sb.ym()
        .args(["presets", "remove", "Good"])
        .assert()
        .success();
    sb.ym()
        .args(["presets", "import"])
        .arg(&junk)
        .arg(&good)
        .assert()
        .code(2);
    assert!(sb.home().join("presets/Good.ympreset").exists());
}

#[test]
fn a_damaged_file_is_a_line_after_the_table_and_never_stops_the_list() {
    let sb = Sandbox::new();
    save(&sb, "Warm evening", &light_doc());
    std::fs::write(sb.home().join("presets/Broken.ympreset"), "{").expect("write");
    let out = sb.ym().args(["presets", "list"]).assert().success();
    let l = lines(&out.get_output().stdout);
    assert_eq!(l.len(), 3, "{l:?}");
    assert_eq!(l[2], "Broken.ympreset: This file isn't a Yellowmoon preset.");
    let out = sb
        .ym()
        .args(["presets", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(
        v["damaged"],
        json!([{ "file": "Broken.ympreset", "reason": "This file isn't a Yellowmoon preset." }])
    );
}

// ---------------------------------------------------------------------------------------------
// render --preset
// ---------------------------------------------------------------------------------------------

#[test]
fn a_preset_is_applied_to_every_photo_and_the_json_names_it() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    save(&sb, "Warm evening", &light_doc());
    let plain = render_json(&sb, &arw, &[]).remove(0);
    assert_eq!(plain["presets"], json!([]));
    let with = render_json(&sb, &arw, &["--preset", "warm"]).remove(0);
    assert_eq!(
        with["presets"],
        json!([{ "name": "Warm evening", "file": null, "missing_look": null }])
    );
    assert_eq!(
        node(&with, "input")["exposure"],
        json!({ "kind": "manual", "ev": 0.5 })
    );
    assert_eq!(node(&with, 1)["grade"]["contrast"], json!({ "kind": "manual", "value": 20.0 }));
    // Another picture than the plain one.
    assert_ne!(
        std::fs::read(plain["output"].as_str().expect("output")).expect("file"),
        std::fs::read(with["output"].as_str().expect("output")).expect("file"),
    );
}

#[test]
fn a_name_matches_ignoring_case_exactly_first_and_then_as_a_prefix() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    save(&sb, "Warm", &light_doc());
    save(&sb, "Warm evening", &detail_doc());
    // Exact wins over the prefix of the other.
    let r = render_json(&sb, &arw, &["--preset", "WARM"]).remove(0);
    assert_eq!(r["presets"][0]["name"], "Warm");
    // A prefix of one name.
    let r = render_json(&sb, &arw, &["--preset", "warm e"]).remove(0);
    assert_eq!(r["presets"][0]["name"], "Warm evening");
    // A prefix of both is exit 2, and the matches are listed; before any photo is opened.
    save(&sb, "Cool morning", &light_doc());
    save(&sb, "Cool noon", &light_doc());
    sb.ym()
        .args(["render", "--cpu", "--preset", "cool"])
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "cool matches more than one preset: Cool morning, Cool noon.",
        ));
    // Nothing matches: exit 3.
    sb.ym()
        .args(["render", "--cpu", "--preset", "nothing"])
        .arg(&arw)
        .assert()
        .code(3)
        .stderr(predicate::str::contains("There is no preset called nothing."));
    assert!(!sb.default_export().exists(), "no photo was opened");
}

#[test]
fn a_path_to_a_file_works_without_importing_it() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    save(&sb, "Warm evening", &light_doc());
    let file = sb.root().join("warm.ympreset");
    sb.ym()
        .args(["presets", "export", "Warm evening"])
        .arg(&file)
        .assert()
        .success();
    sb.ym()
        .args(["presets", "remove", "Warm evening"])
        .assert()
        .success();
    let r = render_json(&sb, &arw, &["--preset", file.to_str().expect("utf-8")]).remove(0);
    assert_eq!(r["presets"][0]["name"], "Warm evening");
    assert_eq!(
        r["presets"][0]["file"].as_str().map(PathBuf::from),
        Some(file.clone()),
        "the file the preset came from"
    );
    assert_eq!(
        node(&r, "input")["exposure"],
        json!({ "kind": "manual", "ev": 0.5 })
    );
    // It was not imported.
    assert!(!sb.home().join("presets/Warm evening.ympreset").exists());
    // A path that is not a file is exit 3; a file that is not a preset is exit 2; both before any photo is opened.
    sb.ym()
        .args(["render", "--cpu", "--preset", "nowhere.ympreset"])
        .arg(&arw)
        .assert()
        .code(3);
    let junk = sb.root().join("junk.ympreset");
    std::fs::write(&junk, "hello").expect("write");
    sb.ym()
        .args(["render", "--cpu", "--preset"])
        .arg(&junk)
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("This file isn't a Yellowmoon preset."));
}

#[test]
fn the_layering_is_settings_then_presets_in_order_then_the_flags() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    // Two presets that share a group: the later one wins on it.
    save(
        &sb,
        "Bright",
        &field_doc(json!({ "exposure": exposure(1.0), "contrast": 40.0 })),
    );
    save(&sb, "Dark", &field_doc(json!({ "exposure": exposure(-1.0) })));
    save(&sb, "Crisp", &detail_doc());
    let settings = settings_file(
        &sb,
        "base.json",
        &field_doc(json!({ "exposure": exposure(0.25), "saturation": 30.0 })),
    );
    let base = settings.to_str().expect("utf-8");

    // --settings, then a preset: Light (all seven fields) is the preset's, so the settings' saturation (Colour) stays.
    let r = render_json(&sb, &arw, &["--settings", base, "--preset", "Bright"]).remove(0);
    assert_eq!(node(&r, "input")["exposure"], json!({ "kind": "manual", "ev": 1.0 }));
    assert_eq!(node(&r, 1)["grade"]["saturation"]["value"], 30.0);
    assert_eq!(node(&r, 1)["grade"]["contrast"]["value"], 40.0);

    // Two presets: `a b` is not `b a`.
    let r = render_json(&sb, &arw, &["--preset", "Bright", "--preset", "Dark"]).remove(0);
    assert_eq!(node(&r, "input")["exposure"], json!({ "kind": "manual", "ev": -1.0 }));
    assert_eq!(
        r["presets"].as_array().expect("presets").len(),
        2,
        "in the order given"
    );
    assert_eq!(r["presets"][0]["name"], "Bright");
    assert_eq!(r["presets"][1]["name"], "Dark");
    let r = render_json(&sb, &arw, &["--preset", "Dark", "--preset", "Bright"]).remove(0);
    assert_eq!(node(&r, "input")["exposure"], json!({ "kind": "manual", "ev": 1.0 }));

    // Stacking is the same as one document: Light then Detail keeps both.
    let stacked = render_json(&sb, &arw, &["--preset", "Bright", "--preset", "Crisp"]).remove(0);
    let doc = settings_file(
        &sb,
        "stacked.json",
        &field_doc(json!({
            "exposure": exposure(1.0),
            "contrast": 40.0,
            "sharpen": { "kind": "manual", "amount": 90.0, "radius_px": 1.0 },
        })),
    );
    let by_hand = render_json(&sb, &arw, &["--settings", doc.to_str().expect("utf-8")]).remove(0);
    assert_eq!(stacked["settings"], by_hand["settings"]);
    assert_eq!(
        std::fs::read(stacked["output"].as_str().expect("output")).expect("file"),
        std::fs::read(by_hand["output"].as_str().expect("output")).expect("file"),
        "the same bytes as the settings document made by hand"
    );

    // A flag has the last word, over the settings and over every preset.
    let r = render_json(
        &sb,
        &arw,
        &["--preset", "Bright", "--preset", "Dark", "--exposure", "2"],
    )
    .remove(0);
    assert_eq!(node(&r, "input")["exposure"], json!({ "kind": "manual", "ev": 2.0 }));
    let r = render_json(
        &sb,
        &arw,
        &["--settings", base, "--preset", "Bright", "--contrast", "-10"],
    )
    .remove(0);
    assert_eq!(node(&r, 1)["grade"]["contrast"]["value"], -10.0);
    assert_eq!(node(&r, 1)["grade"]["saturation"]["value"], 30.0, "the settings'");
}

#[test]
fn auto_stays_auto_so_a_preset_never_gives_a_photo_another_photos_exposure() {
    let sb = Sandbox::new();
    // The second photo is taken a day later, so it is not in the first one's burst and has its own automatic exposure.
    let bright = sb.arw("bright.ARW");
    let dark = sb.arw_with_white("dark.ARW", 3_000);
    // A preset of Light made from a file whose exposure is Auto.
    save(
        &sb,
        "Contrasty",
        &field_doc(json!({ "exposure": { "kind": "auto" }, "contrast": 35.0 })),
    );
    let out = sb
        .ym()
        .args(["render", "--json", "--cpu", "--auto-provider", "yellowmoon", "--preset", "Contrasty"])
        .args(["--format", "png", "--bits", "8", "--out"])
        .arg(sb.root().join("out"))
        .arg(&dark)
        .arg(&bright)
        .assert()
        .success();
    let records = json_lines(&out.get_output().stdout);
    assert_eq!(records.len(), 2);
    // Each photo's exposure is its own automatic one (the record pins an Auto exposure to the number it resolved to), whatever
    // the preset's file holds, and the contrast is the preset's.
    for r in &records {
        assert_eq!(
            node(r, "input")["exposure"]["ev"],
            r["auto"]["input"]["exposure_ev"],
            "the exposure is the photo's own automatic one"
        );
    }
    for r in &records {
        assert_eq!(node(r, 1)["grade"]["contrast"]["value"], 35.0);
    }
}

/// A preset file whose look is not in the library.
fn missing_look_preset(sb: &Sandbox) -> PathBuf {
    let file = sb.root().join("film.ympreset");
    let doc = json!({
        "yellowmoon_preset": 1,
        "name": "Warm evening",
        "groups": ["look", "light"],
        "look": { "id": "user:film-warm", "name": "Film warm" },
        "settings": { "nodes": { "items": [
            { "id": "input", "exposure": exposure(0.5) },
            { "id": 1, "grade": {} },
            { "id": "output", "look": "user:film-warm", "look_intensity": 70.0 },
        ] } },
    });
    std::fs::write(&file, doc.to_string()).expect("write");
    file
}

#[test]
fn a_missing_look_warns_once_applies_the_rest_and_exits_zero() {
    let sb = Sandbox::new();
    let a = sb.arw("a.ARW");
    let b = sb.arw_sized("b.ARW", 128, 96, 7);
    let file = missing_look_preset(&sb);
    let out = sb
        .ym()
        .args(["render", "--json", "--cpu", "--auto-provider", "yellowmoon", "--preset"])
        .arg(&file)
        .args(["--format", "png", "--bits", "8", "--out"])
        .arg(sb.root().join("out"))
        .arg(&a)
        .arg(&b)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Warm evening uses the look Film warm, which isn't imported. Everything else was applied.",
        ));
    let err = text(&out.get_output().stderr);
    assert_eq!(
        err.matches("which isn't imported").count(),
        1,
        "said once, not once a photo:\n{err}"
    );
    let records = json_lines(&out.get_output().stdout);
    assert_eq!(records.len(), 2);
    for r in &records {
        assert_eq!(r["presets"][0]["missing_look"], "Film warm");
        // The rest applied; the look is the photo's own (the default look), not the missing one.
        assert_eq!(node(r, "input")["exposure"], json!({ "kind": "manual", "ev": 0.5 }));
        assert_ne!(r["look"]["id"], "user:film-warm");
        assert_eq!(r["look"]["intensity"], 100);
    }
    // `presets list` says the look is not imported.
    sb.ym()
        .args(["presets", "import"])
        .arg(&file)
        .assert()
        .success();
    let out = sb.ym().args(["presets", "list"]).assert().success();
    assert!(
        lines(&out.get_output().stdout)[1].contains("Film warm (not imported)"),
        "{}",
        text(&out.get_output().stdout)
    );
    let out = sb
        .ym()
        .args(["presets", "list", "--json"])
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["presets"][0]["look"], json!({ "id": "user:film-warm", "name": "Film warm" }));
    assert_eq!(v["presets"][0]["look_missing"], true);
}

#[test]
fn a_preset_does_not_change_what_a_run_without_one_does() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let first = render_json(&sb, &arw, &["--overwrite"]).remove(0);
    let again = render_json(&sb, &arw, &["--overwrite"]).remove(0);
    assert_eq!(
        std::fs::read(first["output"].as_str().expect("output")).expect("file"),
        std::fs::read(again["output"].as_str().expect("output")).expect("file")
    );
    // Rendering with presets writes nothing in the presets folder (rule 4).
    assert!(!sb.home().join("presets").exists());
}
