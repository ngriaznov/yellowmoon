//! `docs/10-cli.md` and `ym render --help` are tested against each other and against the code
//! (`docs/23` section 11): the table of *Where the flags land*, the closing paragraph of the help, the
//! example files of *Settings*, the sentence for the form of Yellowmoon 0.1 and the sentences of the `mask` error for
//! Subject and People.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use common::Sandbox;
use predicates::prelude::*;
use serde_json::Value;
use ym_core::settings::{DevelopSettings, OLD_FORM_SENTENCE};
use ym_looks::library::LookId;

fn repo_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the repository")
        .to_path_buf()
}

fn read(path: &str) -> String {
    std::fs::read_to_string(repo_root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The flags that set a field of the Input, as `cli.rs` has them.
const INPUT_FLAGS: &[&str] = &[
    "--exposure",
    "--exposure-offset",
    "--wb",
    "--light",
    "--lens",
    "--flares",
    "--edges",
    "--anamorphic",
    "--nr",
    "--sharpen",
    "--rolloff",
    "--no-neural",
    "--no-highlight-recovery",
    "--no-lens",
    "--no-ca",
    "--auto-crop",
];
/// The flags that set a field of Main (node 1).
const MAIN_FLAGS: &[&str] = &[
    "--contrast",
    "--brightness",
    "--highlights",
    "--shadows",
    "--white",
    "--black",
    "--saturation",
];
/// The flags that set a field of the Output.
const OUTPUT_FLAGS: &[&str] = &["--look", "--intensity", "--output-sharpening"];
/// The options that touch no node: the whole graph (`--settings`), the camera profile of the run, the
/// Output and Run groups and the options every command has.
const OTHER_FLAGS: &[&str] = &[
    "--settings",
    "--recursive",
    "--profile",
    "--format",
    "--quality",
    "--bits",
    "--compression",
    "--colour-space",
    "--color-space",
    "--long-edge",
    "--out",
    "--name",
    "--skip-existing",
    "--overwrite",
    "--jobs",
    "--json",
    "--cpu",
    "--require-gpu",
    "--auto-provider",
    "--apple-auto",
    "--auto-instructions",
    "--apple-auto-instructions",
    "--quiet",
    "--verbose",
    "--help",
];

fn flags_in(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == b'-'
            && bytes[i + 1] == b'-'
            && bytes[i + 2].is_ascii_lowercase()
            && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'-'))
        {
            let end = text[i + 2..]
                .find(|c: char| !(c.is_ascii_lowercase() || c == '-'))
                .map_or(text.len(), |n| i + 2 + n);
            out.insert(text[i..end].trim_end_matches('-').to_owned());
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// The text of `ym render --help`.
fn help() -> String {
    let sb = Sandbox::new();
    let out = sb.ym().args(["render", "--help"]).assert().success();
    String::from_utf8_lossy(&out.get_output().stdout).into_owned()
}

/// The long options the help lists (the option lines, not the examples or the closing paragraph).
fn help_options(help: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for line in help.lines() {
        let t = line.trim_start();
        if line.starts_with("  ") && (t.starts_with("--") || t.starts_with('-')) {
            let first = t.split_whitespace().next().unwrap_or("");
            let first = if first.ends_with(',') {
                t.split_whitespace().nth(1).unwrap_or("")
            } else {
                first
            };
            found.extend(flags_in(first));
        }
    }
    found
}

/// The closing paragraph of the help: from "Settings are a graph of nodes." to its end.
fn help_paragraph(help: &str) -> String {
    let start = help
        .find("Settings are a graph of nodes.")
        .expect("the closing paragraph of the help");
    help[start..].trim_end().to_owned()
}

/// The table of *Where the flags land* as `flag -> (node, field)`.
fn doc_table(doc: &str) -> BTreeMap<String, (String, String)> {
    let start = doc
        .find("### Where the flags land")
        .expect("the table of Where the flags land");
    let section = &doc[start..];
    let end = section[4..].find("\n### ").map_or(section.len(), |n| n + 4);
    let mut rows = BTreeMap::new();
    for line in section[..end].lines() {
        if !line.starts_with("| `--") {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        assert_eq!(cells.len(), 3, "a row of three cells: {line}");
        let node = if cells[1].starts_with("Input") {
            "Input"
        } else if cells[1].starts_with("Main") {
            "Main"
        } else if cells[1].starts_with("Output") {
            "Output"
        } else {
            "the whole graph"
        };
        for flag in flags_in(cells[0]) {
            rows.insert(flag, (node.to_owned(), cells[2].to_owned()));
        }
    }
    rows
}

#[test]
fn the_cli_documentation_and_the_help_agree() {
    let help = help();
    let doc = read("docs/10-cli.md");

    // 1. Every long option of `ym render` is classified: it sets a field of a node or it touches
    //    none. A new flag has to be put in one of the lists, and so in the table.
    let classified: BTreeSet<&str> = INPUT_FLAGS
        .iter()
        .chain(MAIN_FLAGS)
        .chain(OUTPUT_FLAGS)
        .chain(OTHER_FLAGS)
        .copied()
        .collect();
    let listed = help_options(&help);
    assert!(listed.len() > 30, "the help lists the options: {listed:?}");
    for flag in &listed {
        assert!(
            classified.contains(flag.as_str()),
            "{flag} is an option of `ym render` that this test does not know: say which node it sets, and put it in \
             docs/10 (Where the flags land) and in the help's closing paragraph, or list it as an option that sets no node"
        );
    }
    for flag in INPUT_FLAGS.iter().chain(MAIN_FLAGS).chain(OUTPUT_FLAGS) {
        assert!(
            listed.contains(*flag),
            "{flag} is not an option of the help"
        );
    }

    // 2. The table of docs/10 has a row for every settings flag, with its node and a field, and every
    //    row names a real option.
    let table = doc_table(&doc);
    for (flags, node) in [
        (INPUT_FLAGS, "Input"),
        (MAIN_FLAGS, "Main"),
        (OUTPUT_FLAGS, "Output"),
    ] {
        for flag in flags {
            let (found, field) = table
                .get(*flag)
                .unwrap_or_else(|| panic!("{flag} has no row in Where the flags land"));
            assert_eq!(found, node, "{flag} sets a field of the {node}");
            assert!(
                field.contains('`'),
                "{flag} says which field it sets: {field}"
            );
        }
    }
    assert_eq!(table["--settings"].0, "the whole graph");
    for flag in table.keys() {
        assert!(
            listed.contains(flag.as_str()),
            "the table names {flag}, which is not an option of `ym render`"
        );
        assert!(
            INPUT_FLAGS.contains(&flag.as_str())
                || MAIN_FLAGS.contains(&flag.as_str())
                || OUTPUT_FLAGS.contains(&flag.as_str())
                || flag == "--settings",
            "the table has a row for {flag}, which sets no field of a node"
        );
    }

    // 3. The closing paragraph of the help lists the same flags for the Input, Main and the Output, and
    //    docs/10 quotes it word for word.
    let paragraph = help_paragraph(&help);
    assert!(
        doc.contains(&paragraph),
        "docs/10 quotes the help's closing paragraph"
    );
    let list = |opener: &str| -> BTreeSet<String> {
        let at = paragraph
            .find(opener)
            .unwrap_or_else(|| panic!("{opener} in the paragraph"));
        let rest = &paragraph[at + opener.len()..];
        flags_in(&rest[..rest.find(')').expect("a closing bracket")])
    };
    for (opener, flags) in [
        ("the Input (", INPUT_FLAGS),
        ("node 1, Main (", MAIN_FLAGS),
        ("the Output (", OUTPUT_FLAGS),
    ] {
        let want: BTreeSet<String> = flags.iter().map(|f| (*f).to_owned()).collect();
        assert_eq!(list(opener), want, "the paragraph's flags for {opener}");
    }
    // The `--settings` text of the help names the node-first form.
    assert!(help.contains("a graph of nodes (`ym render --json` prints one)"));
    // The example in the paragraph is a document the core reads.
    let example = paragraph
        .split("for example ")
        .nth(1)
        .and_then(|rest| rest.split("; --json").next())
        .expect("the example of the paragraph");
    let (s, clamped) =
        DevelopSettings::from_json(example, &LookId::neutral()).expect("the example reads");
    assert!(clamped.is_empty());
    assert_eq!(
        s.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(20.0)
    );
}

/// The JSON blocks (` ```json `) of a stretch of a document.
fn json_blocks(section: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = section;
    while let Some(start) = rest.find("```json\n") {
        let body = &rest[start + 8..];
        let end = body.find("```").expect("the end of a block");
        blocks.push(body[..end].to_owned());
        rest = &body[end + 3..];
    }
    blocks
}

#[test]
fn the_example_files_of_docs_10_are_settings_files_the_core_reads() {
    let doc = read("docs/10-cli.md");
    let start = doc
        .find("### The node-first form")
        .expect("the node-first form");
    let end = doc[start..]
        .find("\n### The help text")
        .map_or(doc.len(), |n| start + n);
    let blocks = json_blocks(&doc[start..end]);
    assert_eq!(
        blocks.len(),
        4,
        "the complete small file, the per-file example, the Brush row and the Object row"
    );

    // The complete small file: read over the default graph, checked, and written back unchanged.
    let value: Value = serde_json::from_str(&blocks[0]).expect("the example is JSON");
    let (settings, clamped) =
        DevelopSettings::from_value(value, &LookId::neutral()).expect("the example reads");
    assert!(clamped.is_empty(), "{clamped:?}");
    let written = serde_json::to_value(&settings).expect("writes");
    let (again, clamped) =
        DevelopSettings::from_value(written.clone(), &LookId::neutral()).expect("reads again");
    assert!(clamped.is_empty());
    assert_eq!(again, settings, "written back unchanged");
    assert_eq!(serde_json::to_value(&again).expect("writes"), written);
    // What the example says.
    assert_eq!(
        settings.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(0.3)
    );
    assert_eq!(
        settings.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(20.0)
    );
    assert_eq!(
        settings.nodes.corrector(2).map(|c| c.name.as_str()),
        Some("Sky")
    );
    assert_eq!(
        settings.nodes.corrector(2).map(|c| c.masks.items.len()),
        Some(1)
    );
    assert_eq!(settings.output().look_intensity, 80.0);
    assert_eq!(settings.nodes.links.len(), 3);

    // The per-file example: an object keyed by ARW names, each value a document of that form.
    let per_file: Value = serde_json::from_str(&blocks[1]).expect("the per-file example is JSON");
    let map = per_file.as_object().expect("an object");
    assert!(map.len() >= 2);
    for (name, doc) in map {
        assert!(name.to_ascii_lowercase().ends_with(".arw"), "{name}");
        DevelopSettings::from_value(doc.clone(), &LookId::neutral())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }

    // The Brush row: a row of a corrector's masks that the core reads and can make, with its strokes.
    let row: Value = serde_json::from_str(&blocks[2]).expect("the Brush example is JSON");
    let doc = serde_json::json!({ "nodes": { "items": [
        { "id": 2, "kind": "corrector", "masks": { "items": [row] } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } });
    let (settings, clamped) =
        DevelopSettings::from_value(doc, &LookId::neutral()).expect("the Brush example reads");
    assert!(clamped.is_empty(), "{clamped:?}");
    let item = &settings.nodes.corrector(2).expect("node 2").masks.items[0];
    let ym_core::settings::MaskShape::Brush(brush) = &item.shape else {
        panic!("the example is a Brush row");
    };
    assert_eq!(brush.strokes.len(), 2);
    assert!(brush.strokes[1].erase && brush.strokes[0].points.len() == 3);
    assert_eq!((brush.grow, brush.feather, brush.refine), (0.2, 0.3, 0.5));
    assert_eq!(
        ym_core::pipeline::unmakeable_mask(&settings),
        None,
        "this version makes a Brush mask"
    );
    // The Object row: its points and its box are what the core reads and `ym` makes the mask from; a copy of Yellowmoon without the
    // model refuses it with the sentence `docs/10` quotes (`ym` and the app write the same bytes, which `parity.rs` shows).
    let row: Value = serde_json::from_str(&blocks[3]).expect("the Object example is JSON");
    let doc = serde_json::json!({ "nodes": { "items": [
        { "id": 2, "kind": "corrector", "masks": { "items": [row] } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } });
    let (settings, clamped) =
        DevelopSettings::from_value(doc, &LookId::neutral()).expect("the Object example reads");
    assert!(clamped.is_empty(), "{clamped:?}");
    let item = &settings.nodes.corrector(2).expect("node 2").masks.items[0];
    let ym_core::settings::MaskShape::Object(object) = &item.shape else {
        panic!("the example is an Object row");
    };
    assert_eq!(object.points, [(0.52, 0.41, true), (0.6, 0.55, false)]);
    assert_eq!(object.box_, Some([0.4, 0.3, 0.7, 0.6]));
    assert_eq!(
        ym_core::pipeline::unmakeable_mask_with(
            &settings,
            ym_core::settings::MaskSupport::bare().with_object(true)
        ),
        None,
        "this version makes an Object mask where the model is"
    );
    assert_eq!(
        ym_core::pipeline::unmakeable_mask(&settings).as_deref(),
        Some(ym_core::settings::NO_OBJECT_MODEL_SENTENCE),
        "and says so where it is not"
    );
    assert!(
        read("docs/10-cli.md").contains(&format!(
            "\"{}\"",
            ym_core::settings::NO_OBJECT_MODEL_SENTENCE
        )),
        "docs/10 quotes the sentence"
    );
    // And `ym` takes both files as `--settings` does: they are read, and checked, before any photo is
    // opened; the photo named here is not there, so the run ends for want of a photo (exit 5), never as a
    // settings error (exit 2).
    let sb = Sandbox::new();
    for (label, text) in [("small", &blocks[0]), ("per-file", &blocks[1])] {
        let file = sb.root().join(format!("{label}.json"));
        std::fs::write(&file, text).expect("write");
        sb.ym()
            .args(["render", "--cpu", "--settings"])
            .arg(&file)
            .arg(sb.root().join("not-there.ARW"))
            .assert()
            .code(predicate::ne(2));
    }
}

#[test]
fn the_mask_error_sentences_for_subject_and_people_are_the_same_in_docs_10_docs_23_and_the_code() {
    // `docs/23` section 3.10 gives the sentence for node 2 and a Subject row; `docs/10` quotes it; the core builds it for
    // every node and both kinds, and `ym` prints it (`render.rs` runs the binary).
    let subject = "Node 2 has a Subject mask, and this computer cannot make one. Use a Mac that supports Subject masks, or remove the mask.";
    assert!(
        read("docs/10-cli.md").contains(subject),
        "docs/10 quotes the sentence"
    );
    assert!(
        read("docs/23-nodes.md").contains(subject),
        "docs/23 section 3.10 quotes the sentence"
    );
    assert_eq!(
        ym_core::settings::unmakeable_mask_sentence(2, ym_core::settings::MaskKind::Subject)
            .as_deref(),
        Some(subject)
    );
    assert_eq!(
        ym_core::settings::unmakeable_mask_sentence(7, ym_core::settings::MaskKind::People)
            .as_deref(),
        Some(
            "Node 7 has a People mask, and this computer cannot make one. Use a Mac that supports People masks, or remove the mask."
        )
    );
    // The row's own words in the app (`docs/23` section 7.6), by the core's function and by the page's.
    for (kind, text) in [
        (
            ym_core::settings::MaskKind::Subject,
            "Subject masks need a Mac that supports them",
        ),
        (
            ym_core::settings::MaskKind::People,
            "People masks need a Mac that supports them",
        ),
    ] {
        assert_eq!(
            ym_core::settings::unsupported_row_sentence(kind).as_deref(),
            Some(text)
        );
        assert!(read("docs/23-nodes.md").contains(text), "docs/23: {text}");
        assert!(
            read("src/state/maskRows.ts").contains(text),
            "the page says: {text}"
        );
    }
}

#[test]
fn the_sentence_for_the_form_of_yellowmoon_0_1_is_the_same_in_the_docs_and_in_the_code() {
    // docs/10 and docs/23 section 3.12 say it; the core and `ym` print it.
    assert!(
        read("docs/10-cli.md").contains(&format!("> {OLD_FORM_SENTENCE}")),
        "docs/10 quotes the sentence"
    );
    assert!(
        read("docs/23-nodes.md").contains(OLD_FORM_SENTENCE),
        "docs/23 section 3.12 quotes the sentence"
    );
    // `ym render --settings` with a flat file (the form of 0.1): a usage error, exit 2, the sentence on
    // stderr, nothing rendered.
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let file = sb.root().join("old.json");
    std::fs::write(
        &file,
        r#"{ "look": "builtin:neutral", "contrast": { "kind": "manual", "value": 20 } }"#,
    )
    .expect("write");
    sb.ym()
        .args(["render", "--settings"])
        .arg(&file)
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(OLD_FORM_SENTENCE));
    assert!(!sb.default_export().exists(), "nothing was rendered");
    // The same through stdin, and inside a per-file entry.
    sb.ym()
        .args(["render", "--settings", "-"])
        .arg(&arw)
        .write_stdin(r#"{ "exposure": { "kind": "auto" } }"#)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(OLD_FORM_SENTENCE));
    sb.ym()
        .args(["render", "--settings", "-"])
        .arg(&arw)
        .write_stdin(r#"{ "a.ARW": { "orientation": 90 } }"#)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(OLD_FORM_SENTENCE));
}
