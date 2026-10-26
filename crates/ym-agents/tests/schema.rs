//! The answer's form (`docs/22` section 6.2): the schema's enums and guides equal the Swift source's
//! (`crates/ym-apple/swift/AppleAuto.swift`) and the table of `docs/22`, so the three cannot drift.

use std::path::PathBuf;

use serde_json::Value;
use ym_agents::schema::{FIELDS, Field, schema_text};

fn repo(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

/// The enums of the Swift source: type name to case names, in order.
fn swift_enums(source: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    let mut lines = source.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(rest) = line.trim().strip_prefix("enum ") else {
            continue;
        };
        let name = rest
            .split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap_or("")
            .to_owned();
        let mut cases = Vec::new();
        for body in lines.by_ref() {
            let body = body.trim();
            if body == "}" {
                break;
            }
            if let Some(list) = body.strip_prefix("case ") {
                cases.extend(list.split(',').map(|c| c.trim().to_owned()));
            }
        }
        out.push((name, cases));
    }
    out
}

/// The `@Generable struct Answer` of the Swift source: property name, type name, guide.
fn swift_answer(source: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut in_answer = false;
    let mut guide = String::new();
    for line in source.lines() {
        let line = line.trim();
        if line.starts_with("struct Answer") {
            in_answer = true;
            continue;
        }
        if !in_answer {
            continue;
        }
        if line == "}" {
            break;
        }
        if let Some(rest) = line.strip_prefix("@Guide(description: \"") {
            guide = rest.trim_end_matches("\")").to_owned();
        } else if let Some(rest) = line.strip_prefix("var ") {
            let (name, ty) = rest.split_once(':').unwrap();
            out.push((
                name.trim().to_owned(),
                ty.trim().trim_end_matches('?').to_owned(),
                std::mem::take(&mut guide),
            ));
        }
    }
    out
}

/// The Swift type behind each JSON field's name.
fn swift_type_of(field: &Field) -> &'static str {
    match field.json {
        "scene" | "plan" => "String",
        "exposure" => "Exposure",
        "highlights" => "Highlights",
        "shadows" => "Shadows",
        "blacks" => "Blacks",
        "contrast" => "Contrast",
        "warmth" => "Warmth",
        "tint" => "Tint",
        "saturation" => "Saturation",
        "shadowColour" => "ShadowColour",
        "highlightColour" => "HighlightColour",
        "shadowColourStrength" | "highlightColourStrength" => "Strength",
        other => panic!("{other}"),
    }
}

#[test]
fn the_schema_equals_the_swift_source() {
    let source = std::fs::read_to_string(repo("crates/ym-apple/swift/AppleAuto.swift")).unwrap();
    let enums = swift_enums(&source);
    let answer = swift_answer(&source);
    assert_eq!(answer.len(), FIELDS.len(), "the same fields");
    for (field, (name, ty, guide)) in FIELDS.iter().zip(&answer) {
        assert_eq!(&field.json, name, "the same field in the same place");
        assert_eq!(swift_type_of(field), ty, "{name}");
        assert_eq!(field.description, guide, "{name}: the guide");
        if field.cases.is_empty() {
            continue;
        }
        let (_, cases) = enums
            .iter()
            .find(|(n, _)| n == ty)
            .unwrap_or_else(|| panic!("enum {ty}"));
        let ours: Vec<&str> = field.cases.iter().map(|(c, _)| *c).collect();
        assert_eq!(&ours, cases, "{name}: the cases");
    }
}

/// The table of `docs/22` section 6.2: JSON field, step, values with their codes, description.
/// One row of the spec's table: the JSON field, its cases with the codes the table gives, its guide.
type DocRow = (String, Vec<(String, Option<i8>)>, String);

fn doc_table() -> Vec<DocRow> {
    let text = std::fs::read_to_string(repo("docs/22-auto-providers.md")).unwrap();
    let start = text.find("| JSON field | Step in `AutoSteps`").unwrap();
    let mut rows = Vec::new();
    for line in text[start..].lines().skip(2) {
        if !line.starts_with('|') {
            break;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        let field = cells[0].trim_matches('`').to_owned();
        let values = cells[2];
        let mut cases = Vec::new();
        // "`muchDarker` (−3), `darker` (−2), ..." or "`muchDarker`, `darker`, ... (−2 … 2)".
        let mut rest = values;
        while let Some(open) = rest.find('`') {
            let after = &rest[open + 1..];
            let Some(close) = after.find('`') else { break };
            let name = after[..close].to_owned();
            let tail = after[close + 1..].trim_start();
            let code = tail.strip_prefix('(').and_then(|t| {
                let inner = &t[..t.find(')')?];
                inner.replace('\u{2212}', "-").parse::<i8>().ok()
            });
            cases.push((name, code));
            rest = &after[close + 1..];
        }
        let description = cells[3].to_owned();
        rows.push((field, cases, description));
    }
    rows
}

#[test]
fn the_schema_equals_the_table_in_the_spec() {
    let rows = doc_table();
    assert_eq!(rows.len(), FIELDS.len(), "rows: {rows:?}");
    for (field, (name, cases, description)) in FIELDS.iter().zip(&rows) {
        assert_eq!(&field.json, name);
        assert_eq!(&field.description, description, "{name}: the description");
        if field.cases.is_empty() {
            // `scene` and `plan` are "a string".
            assert!(cases.is_empty(), "{name}: {cases:?}");
            continue;
        }
        let ours: Vec<&str> = field.cases.iter().map(|(c, _)| *c).collect();
        let theirs: Vec<&str> = cases.iter().map(|(c, _)| c.as_str()).collect();
        assert_eq!(ours, theirs, "{name}: the cases");
        // Where the table gives a code for a name, ours is the same.
        for ((_, ours), (case, theirs)) in field.cases.iter().zip(cases) {
            if let Some(theirs) = theirs {
                assert_eq!(ours, theirs, "{name} {case}");
            }
        }
    }
}

#[test]
fn the_text_is_valid_json_and_names_every_field() {
    let schema: Value = serde_json::from_str(schema_text()).unwrap();
    let props = schema["properties"].as_object().unwrap();
    assert_eq!(props.len(), FIELDS.len());
    for field in &FIELDS {
        assert!(props.contains_key(field.json));
        assert_eq!(props[field.json]["description"], field.description);
    }
}
