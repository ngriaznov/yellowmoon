//! The JSON objects of `--json` (`docs/10-cli.md` section JSON output) and the text lines that
//! say the same.
//!
//! `render --json` and `info --json` share the pieces built here: `camera`, `look`, `auto` and
//! `settings` come from the same code, so the two commands cannot drift apart. The numbers are
//! those `ym-core` computes; nothing is worked out again.

use serde_json::{Value, json};
use ym_core::camera::PhotoCamera;
use ym_core::session::PhotoView;
use ym_core::settings::ResolvedSettings;
use ym_core::version::PIPELINE_VERSION;
use ym_profiles::detect::MatchKind;
use ym_raw::metadata::RawMetadata;

use crate::text::escape_json_controls;

/// A JSON object whose keys keep the order they were added in, so the output reads like the
/// example in `docs/10` (`input`, `output`, `status`, `camera`, …) and not alphabetically.
///
/// `serde_json`'s own `Map` sorts its keys unless the `preserve_order` feature is on, and that
/// feature would be switched on for every crate in the `ym` binary (cargo unifies features), which
/// changes the key order of the `ym:Settings` an export writes into its XMP and so the bytes of the
/// file: the CLI and the app must write the same bytes. So the order is kept here, in the one place
/// that needs it. Values inside (`camera`, `settings`, …) are ordinary sorted `serde_json` values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Object(Vec<(&'static str, Value)>);

impl Object {
    /// An empty object.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a key at the end. A key that is already there is replaced in place.
    pub fn insert(&mut self, key: &'static str, value: impl Into<Value>) {
        let value = value.into();
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => *v = value,
            None => self.0.push((key, value)),
        }
    }

    /// Adds all the keys of `other` at the end.
    pub fn extend(&mut self, other: Object) {
        for (k, v) in other.0 {
            self.insert(k, v);
        }
    }

    /// The value of a key.
    #[cfg(test)]
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    /// The object as one line of JSON, keys in the order they were added. DEL and the C1
    /// characters are written as `\u` escapes ([`escape_json_controls`]): the line is safe to
    /// show on a terminal and parses to the same values.
    #[must_use]
    pub fn to_line(&self) -> String {
        let mut out = String::from("{");
        for (i, (key, value)) in self.0.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&Value::from(*key).to_string());
            out.push(':');
            out.push_str(&value.to_string());
        }
        out.push('}');
        escape_json_controls(&out)
    }
}

/// A number as JSON: a whole value without the `.0` (`100`, as `docs/10` shows), else as it is.
#[must_use]
pub fn number(x: f64) -> Value {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 1e15 {
        Value::from(x as i64)
    } else {
        Value::from(x)
    }
}

/// How the profile was chosen, as one word: `exact`, `sensor_family`, `default` or `forced`.
#[must_use]
pub fn match_word(kind: &MatchKind) -> &'static str {
    match kind {
        MatchKind::Exact => "exact",
        MatchKind::SensorFamily { .. } => "sensor_family",
        MatchKind::Default => "default",
        MatchKind::Forced { .. } => "forced",
    }
}

/// The `camera` object: the body, the profile used and why.
///
/// `profile` is the profile id (`sony-ilce-7rm4`) as in `docs/10`; `profile_name` is the short name
/// the app shows (`A7R IV`).
#[must_use]
pub fn camera_json(camera: &PhotoCamera, meta: &RawMetadata) -> Value {
    let m = &camera.matched;
    json!({
        "make": meta.make,
        "model": meta.model,
        "model_id": m.model_id,
        "body": m.body,
        "firmware": meta.software,
        "arw_version": meta.arw_version_name(),
        "profile": m.profile.to_string(),
        "profile_name": camera.copy.profile,
        "profile_version": m.profile_version.to_string(),
        "match": match_word(&m.how),
        "stand_in": camera.stand_in,
        "reduced": camera.reduced,
    })
}

/// The `{ "kind", "message" }` object of a failure.
#[must_use]
pub fn error_value(kind: &str, message: &str) -> Value {
    json!({ "kind": kind, "message": message })
}

/// `camera`, `look`, `auto` and `settings` of a developed photo, ready to be merged into an object.
///
/// `settings` is the document an export records as `ym:Settings`
/// ([`ResolvedSettings::pinned_settings`]): feed it back with `--settings` to get the same file.
#[must_use]
pub fn develop_fields(view: &PhotoView, resolved: &ResolvedSettings) -> Object {
    let mut out = Object::new();
    out.insert("camera", camera_json(&view.camera, &view.meta));
    out.insert(
        "look",
        json!({
            "id": view.settings.look.to_string(),
            "intensity": number(view.settings.look_intensity),
        }),
    );
    out.insert(
        "auto",
        serde_json::to_value(&resolved.estimates).unwrap_or(Value::Null),
    );
    out.insert(
        "settings",
        serde_json::to_value(resolved.pinned_settings()).unwrap_or(Value::Null),
    );
    out
}

/// The failure object of `render --json`: `status: "error"`, no output, and whatever is known of
/// the photo.
#[must_use]
pub fn render_error(input: &str, known: Option<&Object>, kind: &str, message: &str) -> Object {
    let mut obj = Object::new();
    obj.insert("input", input);
    obj.insert("output", Value::Null);
    obj.insert("status", "error");
    if let Some(known) = known {
        obj.extend(known.clone());
    }
    obj.insert("pipeline_version", PIPELINE_VERSION);
    obj.insert("error", error_value(kind, message));
    obj
}

/// The success object of `render --json`. `timings_ms` is (decode, develop, encode).
#[must_use]
pub fn render_ok(
    input: &str,
    output: &str,
    status: &str,
    known: &Object,
    timings_ms: (f64, f64, f64),
    notes: &[String],
) -> Object {
    let mut obj = Object::new();
    obj.insert("input", input);
    obj.insert("output", output);
    obj.insert("status", status);
    obj.extend(known.clone());
    let ms = |x: f64| x.round().max(0.0) as u64;
    obj.insert(
        "timings_ms",
        json!({ "decode": ms(timings_ms.0), "develop": ms(timings_ms.1), "encode": ms(timings_ms.2) }),
    );
    obj.insert("notes", json!(notes));
    obj.insert("pipeline_version", PIPELINE_VERSION);
    obj.insert("error", Value::Null);
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_numbers_have_no_decimal_point() {
        assert_eq!(number(100.0).to_string(), "100");
        assert_eq!(number(80.5).to_string(), "80.5");
        assert_eq!(number(-0.0).to_string(), "0");
        assert_eq!(number(f64::NAN), Value::Null);
    }

    #[test]
    fn match_words_are_stable() {
        assert_eq!(match_word(&MatchKind::Exact), "exact");
        assert_eq!(match_word(&MatchKind::Default), "default");
        assert_eq!(
            match_word(&MatchKind::SensorFamily {
                via: "Sony A7 III".into()
            }),
            "sensor_family"
        );
    }

    #[test]
    fn keys_keep_the_order_they_were_added_in() {
        let mut o = Object::new();
        o.insert("input", "a");
        o.insert("output", Value::Null);
        o.insert("status", "ok");
        o.insert("input", "b");
        assert_eq!(o.to_line(), r#"{"input":"b","output":null,"status":"ok"}"#);
        assert_eq!(o.get("status"), Some(&json!("ok")));
        assert!(o.get("nope").is_none());
    }

    #[test]
    fn a_line_holds_no_control_character_and_reads_back_the_same() {
        // serde_json escapes the C0 controls; DEL and the C1 characters (U+0080 to U+009F, which
        // a terminal may obey) are written as \u escapes by `to_line`.
        let hostile = "a\u{1b}b\u{7f}c\u{85}d\u{9b}e\u{2014}f";
        let mut o = Object::new();
        o.insert("input", hostile);
        o.insert("error", error_value("open", hostile));
        let line = o.to_line();
        assert!(
            !line.chars().any(|c| c.is_control()),
            "a control character in {line:?}"
        );
        for escape in ["\\u001b", "\\u007f", "\\u0085", "\\u009b"] {
            assert!(line.contains(escape), "{escape} is not in {line}");
        }
        let back: Value = serde_json::from_str(&line).expect("valid JSON");
        assert_eq!(back["input"], hostile);
        assert_eq!(back["error"]["message"], hostile);
        // A key with such a character is escaped too.
        let line = Object(vec![("a\u{9b}", json!(1))]).to_line();
        assert_eq!(line, "{\"a\\u009b\":1}");
    }

    #[test]
    fn a_failure_has_no_output_and_names_the_error() {
        let v = render_error("/a/b.ARW", None, "look", "File missing or changed.");
        assert_eq!(v.get("status"), Some(&json!("error")));
        assert_eq!(v.get("output"), Some(&Value::Null));
        assert_eq!(
            v.get("error"),
            Some(&json!({"kind":"look","message":"File missing or changed."}))
        );
        assert!(v.get("camera").is_none());
        assert!(
            v.to_line()
                .starts_with(r#"{"input":"/a/b.ARW","output":null,"status":"error""#)
        );
    }

    #[test]
    fn a_success_follows_the_order_of_docs_10() {
        let mut known = Object::new();
        known.insert("camera", json!({}));
        known.insert("look", json!({}));
        let v = render_ok(
            "/a.ARW",
            "/o.jpg",
            "ok",
            &known,
            (310.4, 2410.0, 179.6),
            &[],
        );
        let line = v.to_line();
        let at = |key: &str| line.find(&format!("\"{key}\":")).unwrap_or(usize::MAX);
        let order = [
            "input",
            "output",
            "status",
            "camera",
            "look",
            "timings_ms",
            "notes",
            "pipeline_version",
            "error",
        ];
        for pair in order.windows(2) {
            assert!(
                at(pair[0]) < at(pair[1]),
                "{} before {}: {line}",
                pair[0],
                pair[1]
            );
        }
        let parsed: Value = serde_json::from_str(&line).expect("valid JSON");
        assert_eq!(parsed["timings_ms"]["decode"], 310);
        assert_eq!(parsed["timings_ms"]["encode"], 180);
        assert_eq!(parsed["error"], Value::Null);
    }
}
