//! The answer's form (`docs/22` section 6.2) and what comes back (section 6.3).
//!
//! The form is `docs/19` section 4's: the same fields, cases and guides, given to the programs as a JSON
//! Schema with every numeric step and colour as a string enum with Apple's case names (the model reads
//! names, not numbers). The text is built by hand, in the order the model should think in (the order of
//! [`FIELDS`]), because a `serde_json` map would sort its keys.

use std::sync::OnceLock;

use serde_json::{Map, Value};
use ym_core::apple_auto::{AutoSteps, HighlightColor, ShadowColor};

use crate::process::clip;

/// Where a field's value goes in the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// `AdvisorAnswer::scene`.
    Scene,
    /// `AdvisorAnswer::plan`.
    Plan,
    /// `AutoSteps::exposure`.
    Exposure,
    /// `AutoSteps::highlights`.
    Highlights,
    /// `AutoSteps::shadows`.
    Shadows,
    /// `AutoSteps::blacks`.
    Blacks,
    /// `AutoSteps::contrast`.
    Contrast,
    /// `AutoSteps::white`: the model's warmth.
    White,
    /// `AutoSteps::tint`.
    Tint,
    /// `AutoSteps::saturation`.
    Saturation,
    /// `AutoSteps::shadow_color`, by code.
    ShadowColor,
    /// `AutoSteps::shadow_color_strength`.
    ShadowStrength,
    /// `AutoSteps::highlight_color`, by code.
    HighlightColor,
    /// `AutoSteps::highlight_color_strength`.
    HighlightStrength,
}

/// One field of the answer's form.
#[derive(Debug, Clone, Copy)]
pub struct Field {
    /// The name in the JSON object (the British spelling of `docs/19`).
    pub json: &'static str,
    /// Where its value goes.
    pub target: Target,
    /// The case names with their codes, in order; empty for the two strings.
    pub cases: &'static [(&'static str, i8)],
    /// The guide the model reads (`docs/19` section 4, verbatim).
    pub description: &'static str,
}

const EXPOSURE: &[(&str, i8)] = &[
    ("muchDarker", -3),
    ("darker", -2),
    ("slightlyDarker", -1),
    ("keep", 0),
    ("slightlyBrighter", 1),
    ("brighter", 2),
    ("muchBrighter", 3),
];
const TONES: &[(&str, i8)] = &[
    ("muchDarker", -2),
    ("darker", -1),
    ("keep", 0),
    ("brighter", 1),
    ("muchBrighter", 2),
];
const BLACKS: &[(&str, i8)] = &[
    ("muchDeeper", -2),
    ("deeper", -1),
    ("keep", 0),
    ("lifted", 1),
    ("muchLifted", 2),
];
const AMOUNT: &[(&str, i8)] = &[
    ("muchLess", -2),
    ("less", -1),
    ("keep", 0),
    ("more", 1),
    ("muchMore", 2),
];
const WARMTH: &[(&str, i8)] = &[
    ("muchCooler", -2),
    ("cooler", -1),
    ("keep", 0),
    ("warmer", 1),
    ("muchWarmer", 2),
];
const TINT: &[(&str, i8)] = &[
    ("muchGreener", -2),
    ("greener", -1),
    ("keep", 0),
    ("moreMagenta", 1),
    ("muchMoreMagenta", 2),
];
const SHADOW_COLOURS: &[(&str, i8)] = &[
    ("none", 0),
    ("teal", 1),
    ("blue", 2),
    ("green", 3),
    ("amber", 4),
    ("magenta", 5),
];
const HIGHLIGHT_COLOURS: &[(&str, i8)] = &[
    ("none", 0),
    ("warm", 1),
    ("gold", 2),
    ("cool", 3),
    ("pink", 4),
];
const STRENGTHS: &[(&str, i8)] = &[("subtle", 1), ("medium", 2), ("strong", 3)];

/// The fields in the order the model should think in (`docs/22` section 6.2).
pub const FIELDS: [Field; 14] = [
    Field {
        json: "scene",
        target: Target::Scene,
        cases: &[],
        description: "What the photo shows and how it is lit, in at most eight words.",
    },
    Field {
        json: "plan",
        target: Target::Plan,
        cases: &[],
        description: "The look you will give this photo and how you will get it, in at most fifteen words.",
    },
    Field {
        json: "exposure",
        target: Target::Exposure,
        cases: EXPOSURE,
        description: "Brightness of the main subject.",
    },
    Field {
        json: "highlights",
        target: Target::Highlights,
        cases: TONES,
        description: "The brightest tones: darker to bring back sky and bright detail, brighter for an airy, luminous look.",
    },
    Field {
        json: "shadows",
        target: Target::Shadows,
        cases: TONES,
        description: "The dark tones: brighter to open up detail in the shadows, darker for depth and mood.",
    },
    Field {
        json: "blacks",
        target: Target::Blacks,
        cases: BLACKS,
        description: "The black level: deeper for rich, solid blacks, lifted for a soft, faded, filmic black.",
    },
    Field {
        json: "contrast",
        target: Target::Contrast,
        cases: AMOUNT,
        description: "Overall contrast: more for punch and depth, less for a soft, gentle look.",
    },
    Field {
        json: "warmth",
        target: Target::White,
        cases: WARMTH,
        description: "Colour of the light: warmer for golden, cosy or sunny light, cooler for blue hour, night or a cold mood; also to remove an unwanted cast.",
    },
    Field {
        json: "tint",
        target: Target::Tint,
        cases: TINT,
        description: "Green against magenta: more magenta to remove a green cast from fluorescent light or foliage, greener to remove a magenta cast.",
    },
    Field {
        json: "saturation",
        target: Target::Saturation,
        cases: AMOUNT,
        description: "Strength of the colours: more for vivid, rich colour, less for a muted, restrained or moody look; keep skin natural.",
    },
    Field {
        json: "shadowColour",
        target: Target::ShadowColor,
        cases: SHADOW_COLOURS,
        description: "A colour for the shadows, as colourists do: teal or blue for a cinematic or cool mood, amber for warmth, none for a neutral grade.",
    },
    Field {
        json: "shadowColourStrength",
        target: Target::ShadowStrength,
        cases: STRENGTHS,
        description: "How strongly the shadows are coloured.",
    },
    Field {
        json: "highlightColour",
        target: Target::HighlightColor,
        cases: HIGHLIGHT_COLOURS,
        description: "A colour for the highlights: warm or gold for sunlit, golden or nostalgic light (with teal shadows this gives the classic cinematic look), cool for a cold, clean feel, none for neutral.",
    },
    Field {
        json: "highlightColourStrength",
        target: Target::HighlightStrength,
        cases: STRENGTHS,
        description: "How strongly the highlights are coloured.",
    },
];

/// The schema as one compact JSON string, properties in [`FIELDS`]' order, `scene` and `plan` required.
/// ASCII only, no double quote or backslash inside a description, about 3 KB.
#[must_use]
pub fn schema_text() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        let mut text = String::from(
            "{\"type\":\"object\",\"title\":\"Answer\",\"additionalProperties\":false,\
             \"required\":[\"scene\",\"plan\"],\"properties\":{",
        );
        for (i, field) in FIELDS.iter().enumerate() {
            if i > 0 {
                text.push(',');
            }
            text.push_str(&format!("\"{}\":{{\"type\":\"string\"", field.json));
            if !field.cases.is_empty() {
                text.push_str(",\"enum\":[");
                for (n, (name, _)) in field.cases.iter().enumerate() {
                    if n > 0 {
                        text.push(',');
                    }
                    text.push_str(&format!("\"{name}\""));
                }
                text.push(']');
            }
            text.push_str(&format!(",\"description\":\"{}\"}}", field.description));
        }
        text.push_str("}}");
        text
    })
}

/// What an answer holds once it is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// The steps; a field the model left out is `None`.
    pub steps: AutoSteps,
    /// What the photo shows.
    pub scene: String,
    /// The look the model gives it.
    pub plan: String,
}

/// The compiled schema, built once.
fn validator() -> Result<&'static jsonschema::Validator, String> {
    static VALIDATOR: OnceLock<Result<jsonschema::Validator, String>> = OnceLock::new();
    VALIDATOR
        .get_or_init(|| {
            let schema: Value = serde_json::from_str(schema_text()).map_err(|e| e.to_string())?;
            jsonschema::draft202012::new(&schema).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Reads the program's `structured_output` (`docs/22` section 6.3): a JSON string is parsed once; every
/// key whose value is `null` and every key that is not in the schema is dropped; the rest is validated
/// against the schema; each name is mapped to its step. The error is the validator's first message,
/// cut to 300 characters.
pub fn parse_answer(output: &Value) -> Result<Parsed, String> {
    let parsed_text;
    let object = match output {
        Value::Object(map) => map,
        Value::String(text) => {
            parsed_text = serde_json::from_str::<Value>(text)
                .map_err(|e| clip(&format!("the answer is not JSON: {e}"), 300))?;
            parsed_text
                .as_object()
                .ok_or_else(|| "the answer is not an object".to_owned())?
        }
        _ => return Err("the answer is not an object".to_owned()),
    };

    let mut kept = Map::new();
    for field in &FIELDS {
        if let Some(value) = object.get(field.json) {
            if !value.is_null() {
                kept.insert(field.json.to_owned(), value.clone());
            }
        }
    }
    let kept = Value::Object(kept);

    let validator = validator()?;
    if let Some(error) = validator.iter_errors(&kept).next() {
        return Err(clip(&error.to_string(), 300));
    }

    let mut steps = AutoSteps::default();
    let (mut scene, mut plan) = (String::new(), String::new());
    for field in &FIELDS {
        let Some(text) = kept.get(field.json).and_then(Value::as_str) else {
            continue;
        };
        if matches!(field.target, Target::Scene) {
            scene = text.to_owned();
            continue;
        }
        if matches!(field.target, Target::Plan) {
            plan = text.to_owned();
            continue;
        }
        let Some(&(_, code)) = field.cases.iter().find(|(name, _)| *name == text) else {
            // The validator has checked the enum; this cannot happen, but it must not be trusted.
            return Err(clip(&format!("{text:?} is not a case of {}", field.json), 300));
        };
        match field.target {
            Target::Exposure => steps.exposure = Some(code),
            Target::Highlights => steps.highlights = Some(code),
            Target::Shadows => steps.shadows = Some(code),
            Target::Blacks => steps.blacks = Some(code),
            Target::Contrast => steps.contrast = Some(code),
            Target::White => steps.white = Some(code),
            Target::Tint => steps.tint = Some(code),
            Target::Saturation => steps.saturation = Some(code),
            Target::ShadowColor => steps.shadow_color = ShadowColor::from_code(i32::from(code)),
            Target::ShadowStrength => steps.shadow_color_strength = Some(code),
            Target::HighlightColor => {
                steps.highlight_color = HighlightColor::from_code(i32::from(code));
            }
            Target::HighlightStrength => steps.highlight_color_strength = Some(code),
            Target::Scene | Target::Plan => {}
        }
    }
    Ok(Parsed { steps, scene, plan })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_schema_is_compact_ascii_valid_json_with_the_fields_in_order() {
        let text = schema_text();
        assert!(text.is_ascii());
        assert!(text.len() < 4096, "{} bytes", text.len());
        assert!(!text.contains('\\'));
        let value: Value = serde_json::from_str(text).unwrap();
        assert_eq!(value["required"], json!(["scene", "plan"]));
        assert_eq!(value["additionalProperties"], json!(false));
        // Properties in the table's order: the text has them in that order.
        let mut at = 0;
        for field in &FIELDS {
            let found = text[at..].find(&format!("\"{}\":{{", field.json));
            assert!(found.is_some(), "{} is out of order", field.json);
            at += found.unwrap();
        }
        assert!(jsonschema::draft202012::new(&value).is_ok());
    }

    #[test]
    fn a_full_answer_maps_to_its_steps() {
        let answer = json!({
            "scene": "Harbour in morning light", "plan": "Cool, crisp morning",
            "exposure": "slightlyBrighter", "highlights": "darker", "shadows": "muchBrighter",
            "blacks": "lifted", "contrast": "more", "warmth": "muchCooler", "tint": "moreMagenta",
            "saturation": "less", "shadowColour": "teal", "shadowColourStrength": "medium",
            "highlightColour": "gold", "highlightColourStrength": "strong"
        });
        let parsed = parse_answer(&answer).unwrap();
        assert_eq!(parsed.scene, "Harbour in morning light");
        assert_eq!(
            parsed.steps,
            AutoSteps {
                exposure: Some(1),
                highlights: Some(-1),
                shadows: Some(2),
                blacks: Some(1),
                contrast: Some(1),
                white: Some(-2),
                tint: Some(1),
                saturation: Some(-1),
                shadow_color: Some(ShadowColor::Teal),
                shadow_color_strength: Some(2),
                highlight_color: Some(HighlightColor::Gold),
                highlight_color_strength: Some(3),
            }
        );
    }

    #[test]
    fn nulls_and_unknown_keys_are_dropped_and_a_string_answer_is_parsed() {
        let answer = json!({"scene": "s", "plan": "p", "exposure": null, "note": "extra", "keepers": 1});
        let parsed = parse_answer(&answer).unwrap();
        assert_eq!(parsed.steps, AutoSteps::default());
        let as_text = Value::String(r#"{"scene":"s","plan":"p","contrast":"less"}"#.to_owned());
        assert_eq!(parse_answer(&as_text).unwrap().steps.contrast, Some(-1));
    }

    #[test]
    fn bad_answers_are_refused_with_the_validators_words() {
        for bad in [
            json!({"scene": "s", "plan": "p", "exposure": "muchBrightest"}),
            json!({"scene": "s", "plan": "p", "exposure": 2}),
            json!({"plan": "p"}),
            json!({"scene": "s"}),
            json!({"scene": 5, "plan": "p"}),
            json!([1, 2]),
            Value::String("not json".into()),
        ] {
            let error = parse_answer(&bad).unwrap_err();
            assert!(!error.is_empty() && error.chars().count() <= 300, "{bad}: {error}");
        }
    }

    #[test]
    fn every_case_name_maps_to_its_code_and_back() {
        for field in FIELDS.iter().filter(|f| !f.cases.is_empty()) {
            for (name, code) in field.cases {
                let mut answer = json!({"scene": "s", "plan": "p"});
                answer[field.json] = json!(name);
                let parsed = parse_answer(&answer).unwrap();
                let got = match field.target {
                    Target::Exposure => parsed.steps.exposure,
                    Target::Highlights => parsed.steps.highlights,
                    Target::Shadows => parsed.steps.shadows,
                    Target::Blacks => parsed.steps.blacks,
                    Target::Contrast => parsed.steps.contrast,
                    Target::White => parsed.steps.white,
                    Target::Tint => parsed.steps.tint,
                    Target::Saturation => parsed.steps.saturation,
                    Target::ShadowColor => parsed.steps.shadow_color.map(|c| c.code() as i8),
                    Target::ShadowStrength => parsed.steps.shadow_color_strength,
                    Target::HighlightColor => parsed.steps.highlight_color.map(|c| c.code() as i8),
                    Target::HighlightStrength => parsed.steps.highlight_color_strength,
                    Target::Scene | Target::Plan => unreachable!(),
                };
                assert_eq!(got, Some(*code), "{} {name}", field.json);
            }
        }
    }
}
