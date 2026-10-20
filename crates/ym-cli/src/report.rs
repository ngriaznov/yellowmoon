//! The JSON objects of `--json` (`docs/10-cli.md` section JSON output) and the text lines that
//! say the same.
//!
//! `render --json` and `info --json` share the pieces built here: `camera`, `look`, `auto` and
//! `settings` come from the same code, so the two commands cannot drift apart. The numbers are
//! those `ym-core` computes; nothing is worked out again.

use serde_json::{Value, json};
use ym_core::apple_auto::{
    AppleAuto, AutoSteps, FailureKind, HighlightColor, PROMPT_VERSION, ShadowColor, TABLE_VERSION,
    steps_json, strength_name,
};
use ym_core::camera::PhotoCamera;
use ym_core::session::PhotoView;
use ym_core::settings::ResolvedSettings;
use ym_core::version::PIPELINE_VERSION;
use ym_profiles::detect::MatchKind;
use ym_raw::metadata::RawMetadata;

use crate::text::{escape_json_controls, line_text};

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

/// When `apple_auto` goes into a photo's object: `info` always has it, `render` only for a photo
/// Apple Auto acted on (`docs/19` section 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleAutoField {
    /// Every photo (`ym info --json`).
    Always,
    /// A photo that was asked: answered, failed or still pending (`ym render --json`).
    WhenActive,
}

/// The per-photo `apple_auto` object of `ym info --json` and `ym render --json` (`docs/19` section
/// 8): `{ "state", "reason", "model", "os", "instructions", "instructions_sha256", "prompt",
/// "table" }`, with `steps`, `scene` and `plan` when the photo was answered. `reason` is the wire
/// name of why Apple Auto is not available (`unavailable`) or why the request failed (`failed`),
/// else null; `model`, `os` and the instructions name the model that answered or was asked and what
/// it read, null otherwise. `steps` are the photo's own answer, as `ym:AppleAuto` records them, a
/// step the model left empty `null`; the steps in effect can be its burst's (`docs/19` section 5).
#[must_use]
pub fn apple_auto_value(apple: &AppleAuto) -> Value {
    let (state, reason, info) = match apple {
        AppleAuto::Off => ("off", Value::Null, None),
        AppleAuto::Unavailable { reason } => (
            "unavailable",
            serde_json::to_value(reason).unwrap_or(Value::Null),
            None,
        ),
        AppleAuto::Pending => ("pending", Value::Null, None),
        AppleAuto::Answered { info, .. } => ("answered", Value::Null, Some(info)),
        AppleAuto::Failed { kind, info, .. } => (
            "failed",
            serde_json::to_value(kind).unwrap_or(Value::Null),
            Some(info),
        ),
    };
    let mut value = json!({
        "state": state,
        "reason": reason,
        "model": info.map(|i| i.model.as_str()),
        "os": info.map(|i| i.os.as_str()),
        "instructions": info.map(|i| i.instructions.instructions),
        "instructions_sha256": info.map(|i| i.instructions.instructions_sha256.as_str()),
        "prompt": PROMPT_VERSION,
        "table": TABLE_VERSION,
    });
    if let (AppleAuto::Answered { answer, .. }, Value::Object(map)) = (apple, &mut value) {
        map.insert("steps".to_owned(), steps_json(&answer.steps));
        map.insert("scene".to_owned(), json!(answer.scene));
        map.insert("plan".to_owned(), json!(answer.plan));
    }
    value
}

/// What the steps do, in words, in the order of the table (exposure, highlights, shadows, black,
/// contrast, warmth, tint, saturation, the shadows' colour, the highlights' colour), and only the
/// ones that were set and change something: `Brighter +0.3 EV · highlights -20 · cooler 10 mired ·
/// teal shadows (medium)`, or `No change` when every step that was set is keep or none. The numbers
/// are the table's (`docs/19` section 5).
#[must_use]
pub fn steps_text(steps: AutoSteps) -> String {
    let Some(v) = steps.values() else {
        return format!("steps out of range ({steps:?})");
    };
    let mut parts: Vec<String> = Vec::new();
    let changed = |step: Option<i8>| step.is_some_and(|s| s != 0);
    if let Some(e) = steps.exposure.filter(|e| *e != 0) {
        let word = if e > 0 { "brighter" } else { "darker" };
        parts.push(format!("{word} {:+.1} EV", v.exposure_ev));
    }
    if changed(steps.highlights) {
        parts.push(format!("highlights {:+.0}", v.highlights));
    }
    if changed(steps.shadows) {
        parts.push(format!("shadows {:+.0}", v.shadows));
    }
    if changed(steps.blacks) {
        parts.push(format!("black {:+.0}", v.black));
    }
    if changed(steps.contrast) {
        parts.push(format!("contrast {:+.0}", v.contrast));
    }
    if let Some(w) = steps.white.filter(|w| *w != 0) {
        // A warmer picture is a lower mired of the user's white (`AppleValues::white_shift_mired`).
        let word = if w > 0 { "warmer" } else { "cooler" };
        parts.push(format!("{word} {:.0} mired", v.white_shift_mired.abs()));
    }
    if changed(steps.tint) {
        parts.push(format!("tint {:+.0}", v.tint_shift));
    }
    if changed(steps.saturation) {
        parts.push(format!("saturation {:+.0}", v.saturation));
    }
    if let (Some(c), Some(s)) = (steps.shadow_color, steps.shadow_color_strength)
        && c != ShadowColor::None
    {
        parts.push(format!("{} shadows ({})", c.name(), strength_name(s)));
    }
    if let (Some(c), Some(s)) = (steps.highlight_color, steps.highlight_color_strength)
        && c != HighlightColor::None
    {
        parts.push(format!("{} highlights ({})", c.name(), strength_name(s)));
    }
    if parts.is_empty() {
        return "No change".to_owned();
    }
    let text = parts.join(" · ");
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// A failure in words: the kind's words, with Apple's message for a failure of no known kind.
fn failure_text(kind: FailureKind, message: &str) -> String {
    if kind == FailureKind::Other && !message.trim().is_empty() {
        format!("{}: {}", kind.words(), line_text(message.trim()))
    } else {
        kind.words().to_owned()
    }
}

/// The **Apple Auto** row of `ym info` (`docs/19` section 8). `in_effect` is the steps the photo's
/// Auto values use (`AutoEstimates::apple_steps`: its own answer, or its burst's), `None` when Auto
/// is the rule.
///
/// * answered: what the steps do, the scene and the plan, `Brighter +0.3 EV · contrast +15 · scene
///   "Backlit portrait at dusk" · plan "Warm, soft contrast"` (`No change · …` for keep on every
///   step), with `(shared with its burst)` when the burst's steps differ from the photo's own;
/// * failed: `Failed (macOS asked Yellowmoon to wait); Yellowmoon's own Auto`, or the burst's
///   steps when the burst has them;
/// * `Off`, `Not available (Apple Intelligence is off in System Settings)`, and
///   `Waiting for Apple Intelligence` for a photo still waiting for its answer.
#[must_use]
pub fn apple_auto_text(apple: &AppleAuto, in_effect: Option<AutoSteps>) -> String {
    match apple {
        AppleAuto::Off => "Off".to_owned(),
        AppleAuto::Unavailable { reason } => format!("Not available ({})", reason.words()),
        AppleAuto::Pending => "Waiting for Apple Intelligence".to_owned(),
        AppleAuto::Answered { answer, .. } => {
            let steps = in_effect.unwrap_or(answer.steps);
            let mut text = steps_text(steps);
            if steps != answer.steps {
                text.push_str(" (shared with its burst)");
            }
            let scene = answer.scene.trim();
            if !scene.is_empty() {
                text.push_str(&format!(" · scene \"{}\"", line_text(scene)));
            }
            let plan = answer.plan.trim();
            if !plan.is_empty() {
                text.push_str(&format!(" · plan \"{}\"", line_text(plan)));
            }
            text
        }
        AppleAuto::Failed { kind, message, .. } => {
            let why = failure_text(*kind, message);
            match in_effect {
                Some(steps) => format!("Failed ({why}); its burst's answer: {}", steps_text(steps)),
                None => format!("Failed ({why}); Yellowmoon's own Auto"),
            }
        }
    }
}

/// `camera`, `look`, `auto`, `auto_source`, `apple_auto` and `settings` of a developed photo, ready
/// to be merged into an object.
///
/// `auto_source` is which Auto made the automatic values (`rule` or `apple`, as `ym:Auto` records
/// it); `apple_auto` is [`apple_auto_value`], always or only for a photo Apple Auto acted on
/// (`field`). `settings` is the document an export records as `ym:Settings`
/// ([`ResolvedSettings::pinned_settings`]): feed it back with `--settings` to get the same file.
#[must_use]
pub fn develop_fields(
    view: &PhotoView,
    resolved: &ResolvedSettings,
    field: AppleAutoField,
) -> Object {
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
    out.insert("auto_source", resolved.estimates.source.name());
    let apple = &resolved.estimates.apple;
    let acted = !matches!(apple, AppleAuto::Off | AppleAuto::Unavailable { .. });
    if field == AppleAutoField::Always || acted {
        out.insert("apple_auto", apple_auto_value(apple));
    }
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

    /// Steps with the four steps of prompt version 1 set, every other step keep and no colour.
    fn steps(exposure: i8, white: i8, contrast: i8, saturation: i8) -> AutoSteps {
        AutoSteps {
            exposure: Some(exposure),
            white: Some(white),
            contrast: Some(contrast),
            saturation: Some(saturation),
            ..AutoSteps::keep()
        }
    }

    fn info() -> ym_core::apple_auto::AdvisorInfo {
        ym_core::apple_auto::AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)")
    }

    fn answered(s: AutoSteps, scene: &str, plan: &str) -> AppleAuto {
        AppleAuto::Answered {
            answer: ym_core::apple_auto::AdvisorAnswer {
                steps: s,
                scene: scene.to_owned(),
                plan: plan.to_owned(),
            },
            info: info(),
        }
    }

    #[test]
    fn the_steps_read_as_what_they_do_in_the_order_of_docs_19() {
        assert_eq!(steps_text(steps(0, 0, 0, 0)), "No change");
        assert_eq!(
            steps_text(steps(1, 0, 1, 0)),
            "Brighter +0.3 EV · contrast +15"
        );
        assert_eq!(
            steps_text(steps(-3, -2, -1, 2)),
            "Darker -1.0 EV · contrast -15 · cooler 20 mired · saturation +30"
        );
        assert_eq!(steps_text(steps(0, 1, 0, 0)), "Warmer 10 mired");
        assert_eq!(steps_text(steps(0, 0, 0, -2)), "Saturation -30");
        assert!(steps_text(steps(5, 0, 0, 0)).starts_with("steps out of range"));
        // Every control, in the order of the table, and the colours with their strengths.
        let all = AutoSteps {
            exposure: Some(-1),
            highlights: Some(-2),
            shadows: Some(1),
            blacks: Some(-1),
            contrast: Some(1),
            white: Some(-1),
            tint: Some(1),
            saturation: Some(-1),
            shadow_color: Some(ShadowColor::Teal),
            shadow_color_strength: Some(2),
            highlight_color: Some(HighlightColor::Warm),
            highlight_color_strength: Some(1),
        };
        assert_eq!(
            steps_text(all),
            "Darker -0.3 EV · highlights -40 · shadows +15 · black -15 · contrast +15 · cooler 10 mired · tint +5 · saturation -15 · teal shadows (medium) · warm highlights (subtle)"
        );
        // Only what was set is listed; none and a colour without a strength are not.
        let some = AutoSteps {
            tint: Some(-2),
            shadow_color: Some(ShadowColor::None),
            highlight_color: Some(HighlightColor::Gold),
            ..AutoSteps::default()
        };
        assert_eq!(steps_text(some), "Tint -10");
        let keep_only = AutoSteps {
            blacks: Some(0),
            ..AutoSteps::default()
        };
        assert_eq!(steps_text(keep_only), "No change");
    }

    #[test]
    fn the_apple_auto_row_says_what_happened() {
        use ym_core::apple_auto::UnavailableReason;
        let a = answered(
            steps(1, 0, 1, 0),
            "Backlit portrait at dusk",
            "Warm and soft",
        );
        assert_eq!(
            apple_auto_text(&a, Some(steps(1, 0, 1, 0))),
            "Brighter +0.3 EV · contrast +15 · scene \"Backlit portrait at dusk\" · plan \"Warm and soft\""
        );
        // A burst's shared steps are the ones in effect, and the row says so.
        assert_eq!(
            apple_auto_text(&a, Some(steps(0, 0, 1, 0))),
            "Contrast +15 (shared with its burst) · scene \"Backlit portrait at dusk\" · plan \"Warm and soft\""
        );
        // A scene with a control character in it does nothing to the terminal.
        let hostile = answered(steps(0, 0, 0, 0), "a\u{1b}[2Jb", "");
        assert_eq!(
            apple_auto_text(&hostile, Some(steps(0, 0, 0, 0))),
            "No change · scene \"a\\x1b[2Jb\""
        );
        assert_eq!(apple_auto_text(&AppleAuto::Off, None), "Off");
        assert_eq!(
            apple_auto_text(
                &AppleAuto::Unavailable {
                    reason: UnavailableReason::AppleIntelligenceOff
                },
                None
            ),
            "Not available (Apple Intelligence is off in System Settings)"
        );
        let failed = |kind, message: &str| AppleAuto::Failed {
            kind,
            message: message.to_owned(),
            info: ym_core::apple_auto::AdvisorInfo::new("m", "o"),
        };
        assert_eq!(
            apple_auto_text(&failed(FailureKind::RateLimited, "busy"), None),
            "Failed (macOS asked Yellowmoon to wait); Yellowmoon's own Auto"
        );
        assert_eq!(
            apple_auto_text(&failed(FailureKind::Other, "The model said no."), None),
            "Failed (an error: The model said no.); Yellowmoon's own Auto"
        );
        assert_eq!(
            apple_auto_text(&failed(FailureKind::Timeout, ""), Some(steps(-1, 0, 0, 0))),
            "Failed (no answer within 30 seconds); its burst's answer: Darker -0.3 EV"
        );
    }

    #[test]
    fn the_apple_auto_object_has_steps_scene_and_plan_only_when_answered() {
        use ym_core::apple_auto::{InstructionsId, UnavailableReason};
        let sha = InstructionsId::default().instructions_sha256;
        let v = apple_auto_value(&answered(
            AutoSteps {
                exposure: Some(1),
                white: Some(-1),
                contrast: Some(1),
                saturation: Some(0),
                ..AutoSteps::default()
            },
            "Dusk",
            "Cool and moody",
        ));
        assert_eq!(
            v,
            json!({
                "state": "answered", "reason": null, "model": "AFM 3 Core", "os": "27.0.1 (27A404)",
                "instructions": "default", "instructions_sha256": sha,
                "prompt": 3, "table": 2,
                "steps": {
                    "exposure": 1, "highlights": null, "shadows": null, "blacks": null,
                    "contrast": 1, "white": -1, "tint": null, "saturation": 0,
                    "shadow_color": null, "shadow_color_strength": null,
                    "highlight_color": null, "highlight_color_strength": null
                },
                "scene": "Dusk",
                "plan": "Cool and moody"
            })
        );
        let v = apple_auto_value(&AppleAuto::Unavailable {
            reason: UnavailableReason::NotAMac,
        });
        assert_eq!(
            v,
            json!({ "state": "unavailable", "reason": "not_a_mac", "model": null, "os": null,
                    "instructions": null, "instructions_sha256": null, "prompt": 3, "table": 2 })
        );
        let v = apple_auto_value(&AppleAuto::Failed {
            kind: FailureKind::RateLimited,
            message: "busy".to_owned(),
            info: info().with_instructions("Moody."),
        });
        assert_eq!(v["state"], "failed");
        assert_eq!(v["reason"], "rate_limited");
        assert_eq!(v["model"], "AFM 3 Core");
        assert_eq!(v["instructions"], "edited");
        assert_eq!(
            v["instructions_sha256"],
            InstructionsId::of("Moody.").instructions_sha256
        );
        assert!(v.get("steps").is_none() && v.get("scene").is_none() && v.get("plan").is_none());
        assert_eq!(
            apple_auto_value(&AppleAuto::Off),
            json!({ "state": "off", "reason": null, "model": null, "os": null,
                    "instructions": null, "instructions_sha256": null, "prompt": 3, "table": 2 })
        );
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
