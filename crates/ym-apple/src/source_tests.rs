//! Reads the Swift bridge, its C header and `src/bridge.rs` and checks them against
//! `docs/19-apple-auto.md` and the core, so that the words the model sees, the answer's schema, the
//! steps, the codes and the C names cannot drift apart. Runs on every platform: no Swift compiler.

use std::path::Path;

use ym_core::apple_auto::{
    DEFAULT_INSTRUCTIONS, FailureKind, HighlightColor, PROMPT_VERSION, ShadowColor,
    UnavailableReason,
};

use crate::abi::{FAILURE_CODES, PLAN_BYTES, SCENE_BYTES, UNAVAILABLE_CODES, UNSET, YmAppleAnswer};

fn read(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn swift() -> String {
    read("swift/AppleAuto.swift")
}

fn header() -> String {
    read("swift/ym_apple.h")
}

fn spec() -> String {
    read("../../docs/19-apple-auto.md")
}

/// Section `n` of the spec: from its `## n.` heading to the next `## ` heading.
fn section(spec: &str, n: u32) -> String {
    let heading = format!("\n## {n}. ");
    let start = spec
        .find(&heading)
        .unwrap_or_else(|| panic!("docs/19 has no section {n}"))
        + 1;
    let rest = &spec[start..];
    let end = rest[1..].find("\n## ").map_or(rest.len(), |i| i + 1);
    rest[..end].to_owned()
}

/// Runs of whitespace as one space.
fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `MacosTooOld` → `macos_too_old`, the wire names of `serde(rename_all = "snake_case")`.
fn snake(debug_name: &str) -> String {
    let mut out = String::new();
    for (i, c) in debug_name.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// `macos_too_old` → `macosTooOld`, the Swift names of the codes.
fn camel(snake_name: &str) -> String {
    let mut out = String::new();
    let mut upper = false;
    for c in snake_name.chars() {
        if c == '_' {
            upper = true;
        } else if upper {
            out.push(c.to_ascii_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// The body of a top-level Swift declaration that starts with `header` (up to the `}` that closes
/// it at the start of a line).
fn block<'a>(source: &'a str, header: &str) -> &'a str {
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("the Swift has no `{header}`"));
    let rest = &source[start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{header}` is not closed"));
    &rest[..end]
}

/// One field of the answer as section 4 describes it.
struct Field {
    name: String,
    /// The enum's cases in order with their codes; empty for a `String` field.
    cases: Vec<(String, i32)>,
    guide: String,
}

/// What section 4 says the bridge sends.
struct Request {
    instructions: String,
    prompt: String,
    options: String,
    fields: Vec<Field>,
}

/// `` `muchDarker` −3, `darker` −2, … `` as `(name, code)` pairs (the minus may be U+2212).
fn coded_cases(cell: &str) -> Vec<(String, i32)> {
    cell.split(',')
        .map(|item| {
            let item = item.trim();
            let (name, code) = item
                .rsplit_once(' ')
                .unwrap_or_else(|| panic!("`{item}` is not `<case> <code>`"));
            let code = code
                .replace('\u{2212}', "-")
                .trim_start_matches('+')
                .parse()
                .unwrap_or_else(|_| panic!("`{item}` has no code"));
            (name.trim().trim_matches('`').to_owned(), code)
        })
        .collect()
}

fn request(spec: &str) -> Request {
    let s = section(spec, 4);
    // The default instructions run over several lines, so section 4 gives them in a ```text block,
    // its lines joined with a line feed.
    let mut after_fence = s.lines().skip_while(|l| l.trim() != "```text");
    assert!(
        after_fence.next().is_some(),
        "section 4 gives the instructions in a ```text block"
    );
    let instructions = after_fence
        .take_while(|l| l.trim() != "```")
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        s.matches("```text").count() == 1,
        "section 4 has one ```text block, the instructions"
    );
    let prompt = s
        .lines()
        .find_map(|l| l.strip_prefix("**Prompt:** \""))
        .and_then(|rest| rest.split('"').next())
        .expect("section 4 gives the prompt")
        .to_owned();
    let options = s
        .lines()
        .find_map(|l| l.strip_prefix("**Options:** `"))
        .and_then(|rest| rest.split('`').next())
        .expect("section 4 gives the options")
        .to_owned();
    let fields = s
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| {
            let cells: Vec<&str> = l
                .trim()
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect();
            assert_eq!(cells.len(), 3, "{l}");
            let cases = if cells[1] == "`String`" {
                Vec::new()
            } else {
                coded_cases(cells[1])
            };
            Field {
                name: cells[0].trim_matches('`').to_owned(),
                cases,
                guide: cells[2].trim_matches('"').to_owned(),
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(fields.len(), 14, "section 4's answer has fourteen fields");
    Request {
        instructions,
        prompt,
        options,
        fields,
    }
}

#[test]
fn only_the_system_model_on_this_mac_is_used() {
    let swift = swift();
    assert!(!swift.contains("PrivateCloudCompute"));
    assert!(swift.contains("SystemLanguageModel.default"));
    // No model is constructed (`SystemLanguageModel(useCase:)`, any other `…LanguageModel(`), and
    // no type of ours is a model.
    assert!(!swift.contains("LanguageModel("));
    assert!(!swift.contains(": LanguageModel"));
    // Every session is on the system model, without tools: no model argument, so the
    // initialiser's default, `SystemLanguageModel.default`.
    let sessions: Vec<&str> = swift.split("LanguageModelSession(").skip(1).collect();
    assert!(!sessions.is_empty(), "the Swift opens a session");
    for after in sessions {
        assert!(
            squeeze(after).starts_with("tools: [], instructions: {"),
            "a session that is not on the default system model without tools: {}",
            &after[..after.len().min(120)]
        );
    }
    assert!(!swift.contains("model:"), "a model is passed explicitly");
    assert!(!swift.contains(": Tool"));
    // Nothing that reaches the network (hard rule 7).
    for word in ["URLSession", "http:", "https:", "Network"] {
        assert!(!swift.contains(word), "{word}");
    }
}

#[test]
fn the_default_instructions_are_section_4s_and_reach_the_swift_by_parameter() {
    let swift = swift();
    let request = request(&spec());
    assert_eq!(
        request.instructions, DEFAULT_INSTRUCTIONS,
        "ym_core::apple_auto::DEFAULT_INSTRUCTIONS is not the text of docs/19 section 4"
    );
    // The Swift does not hold the instructions: it takes them from ym_apple_auto_ask's parameter
    // and opens the session with them.
    assert!(
        !swift.contains(&request.instructions[..60]),
        "the Swift holds the instructions"
    );
    let ask = block(&swift, "public func ymAppleAutoAsk(");
    assert!(squeeze(ask).contains("_ instructions: UnsafePointer<CChar>?,"));
    assert!(ask.contains("String(cString: $0)"));
    assert!(ask.contains("instructions: instructionsText"));
    let model = block(&swift, "private func askModel(");
    assert!(model.contains("instructions: String"));
    assert!(model.contains("LanguageModelSession(tools: [], instructions: { instructions })"));
}

#[test]
fn the_swift_sends_what_section_4_says() {
    let swift = swift();
    let request = request(&spec());
    assert!(
        swift.contains(&format!("\"{}\"", request.prompt)),
        "the prompt is not in the Swift verbatim"
    );
    assert!(
        swift.contains(&request.options),
        "the options `{}` are not in the Swift verbatim",
        request.options
    );
    assert!(swift.contains("includeSchemaInPrompt: true"));
    assert!(swift.contains("Attachment(image, orientation: .up)"));
    // The answer is streamed, so a cut answer keeps the fields filled until then.
    assert!(swift.contains("session.streamResponse("));
    assert!(swift.contains("generating: Answer.self"));

    let lines: Vec<&str> = swift.lines().map(str::trim).collect();
    let filled = block(&swift, "private func filledFields(");
    let mut last_line = 0;
    for field in &request.fields {
        let guide = format!("@Guide(description: \"{}\")", field.guide);
        let at = lines
            .iter()
            .position(|l| *l == guide)
            .unwrap_or_else(|| panic!("no {guide} in the Swift"));
        let declaration = lines[at + 1];
        let prefix = format!("var {}: ", field.name);
        let swift_type = declaration
            .strip_prefix(&prefix)
            .unwrap_or_else(|| panic!("{guide} is not on `{prefix}…`: {declaration}"));
        // Fields are generated in the order they are declared.
        assert!(at > last_line, "{} is out of order", field.name);
        last_line = at;

        if field.cases.is_empty() {
            assert_eq!(swift_type, "String", "{}", field.name);
            assert!(
                filled.contains(&format!("forProperty: \"{}\"", field.name)),
                "filledFields does not read {}",
                field.name
            );
            continue;
        }
        // Every value is optional: a field the model leaves empty sets nothing.
        let swift_type = swift_type
            .strip_suffix('?')
            .unwrap_or_else(|| panic!("{} is not optional: {swift_type}", field.name));
        let declared = block(&swift, &format!("enum {swift_type} {{"));
        let names: Vec<&str> = field.cases.iter().map(|(n, _)| n.as_str()).collect();
        let cases_line = format!("case {}", names.join(", "));
        assert!(
            declared.lines().any(|l| l.trim() == cases_line),
            "enum {swift_type} does not declare `{cases_line}`"
        );
        assert_eq!(
            declared.matches("case ").count(),
            1,
            "enum {swift_type} declares other cases"
        );
        // The codes: section 4's numbers, case by case.
        let mapping = block(&swift, &format!("extension {swift_type}: AnswerCode {{"));
        for (case, code) in &field.cases {
            let line = format!("case .{case}: return {code}");
            assert!(
                mapping.lines().any(|l| l.trim() == line),
                "extension {swift_type} lacks `{line}`"
            );
        }
        assert_eq!(
            mapping.matches("case .").count(),
            field.cases.len(),
            "extension {swift_type} maps other cases"
        );
        // The answer's reader takes this field by its name and type.
        assert!(
            filled.contains(&format!(
                "code(of: {swift_type}.self, \"{}\", in: content)",
                field.name
            )),
            "filledFields does not read {} as {swift_type}",
            field.name
        );
    }
}

#[test]
fn the_codes_of_section_4_are_the_cores() {
    // The steps run from −n to +n with keep 0 in the core's ranges; the colours and strengths are
    // the core's codes.
    let request = request(&spec());
    let field = |name: &str| {
        request
            .fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no field {name}"))
    };
    for (name, n) in [
        ("exposure", 3),
        ("highlights", 2),
        ("shadows", 2),
        ("blacks", 2),
        ("contrast", 2),
        ("warmth", 2),
        ("tint", 2),
        ("saturation", 2),
    ] {
        let codes: Vec<i32> = field(name).cases.iter().map(|(_, c)| *c).collect();
        assert_eq!(codes, (-n..=n).collect::<Vec<_>>(), "{name}");
        assert_eq!(field(name).cases[n as usize].0, "keep", "{name}");
    }
    let shadows: Vec<(String, i32)> = ShadowColor::ALL
        .iter()
        .map(|c| (to_camel(c.name()), c.code()))
        .collect();
    assert_eq!(field("shadowColour").cases, shadows);
    let highlights: Vec<(String, i32)> = HighlightColor::ALL
        .iter()
        .map(|c| (to_camel(c.name()), c.code()))
        .collect();
    assert_eq!(field("highlightColour").cases, highlights);
    for name in ["shadowColourStrength", "highlightColourStrength"] {
        let cases = &field(name).cases;
        assert_eq!(
            cases
                .iter()
                .map(|(n, c)| (n.as_str(), *c))
                .collect::<Vec<_>>(),
            [("subtle", 1), ("medium", 2), ("strong", 3)],
            "{name}"
        );
        for (n, c) in cases {
            assert_eq!(ym_core::apple_auto::strength_name(*c as i8), n);
        }
    }
}

/// A name of the core (`teal`) as the Swift case (`teal`); one word each, so the same.
fn to_camel(name: &str) -> String {
    camel(name)
}

#[test]
fn the_swift_prompt_version_is_the_cores() {
    let swift = swift();
    let version: u32 = swift
        .lines()
        .find_map(|l| l.trim().strip_prefix("let promptVersion = "))
        .expect("the Swift declares promptVersion")
        .trim()
        .parse()
        .expect("promptVersion is a number");
    assert_eq!(version, PROMPT_VERSION);
}

/// Codes and their snake-case names, in the order the spec lists them.
type Codes = Vec<(i32, String)>;

/// The `n name` pairs of a list such as "3 `macos_too_old`, 4 `device_not_eligible`" or
/// "1 unavailable, 2 rate limited", names in snake case.
fn coded_list(list: &str) -> Codes {
    list.split(',')
        .map(|item| {
            let item = item.trim();
            let (code, name) = item
                .split_once(' ')
                .unwrap_or_else(|| panic!("`{item}` is not `<code> <name>`"));
            let code = code
                .parse()
                .unwrap_or_else(|_| panic!("`{item}` has no code"));
            (code, name.trim().trim_matches('`').replace(' ', "_"))
        })
        .collect()
}

/// The codes section 11 gives: (availability codes, ask error codes).
fn spec_codes() -> (Codes, Codes) {
    let s = squeeze(&section(&spec(), 11));
    let availability = s
        .split("in table order from 3: ")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("section 11 lists the availability codes");
    let errors = s
        .split("else an error code (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("section 11 lists the error codes");
    (coded_list(availability), coded_list(errors))
}

#[test]
fn the_codes_are_those_of_section_11() {
    let (availability, errors) = spec_codes();
    let rust_availability: Codes = UNAVAILABLE_CODES
        .iter()
        .map(|(code, reason)| (*code, snake(&format!("{reason:?}"))))
        .collect();
    assert_eq!(rust_availability, availability);
    let rust_errors: Codes = FAILURE_CODES
        .iter()
        .map(|(code, kind)| (*code, snake(&format!("{kind:?}"))))
        .collect();
    assert_eq!(rust_errors, errors);
    // The snake names are the core's wire names (a spot check of the conversion).
    assert_eq!(
        snake(&format!("{:?}", UnavailableReason::NotAMac)),
        "not_a_mac"
    );
    assert_eq!(
        snake(&format!("{:?}", FailureKind::ContextTooLarge)),
        "context_too_large"
    );

    let swift = swift();
    let swift_availability = block(&swift, "private enum AvailabilityCode {");
    assert!(swift_availability.contains("static let available: Int32 = 0"));
    for (code, name) in &availability {
        let line = format!("static let {}: Int32 = {code}", camel(name));
        assert!(
            swift_availability.contains(&line),
            "AvailabilityCode lacks `{line}`"
        );
    }
    let swift_errors = block(&swift, "private enum AskCode {");
    assert!(swift_errors.contains("static let answered: Int32 = 0"));
    for (code, name) in &errors {
        let line = format!("static let {}: Int32 = {code}", camel(name));
        assert!(swift_errors.contains(&line), "AskCode lacks `{line}`");
    }
}

/// The names of the C functions `src/bridge.rs` declares.
fn rust_functions() -> Vec<String> {
    read("src/bridge.rs")
        .split("fn ym_apple_auto_")
        .skip(1)
        .map(|rest| {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            format!("ym_apple_auto_{name}")
        })
        .collect()
}

#[test]
fn the_header_the_swift_and_the_rust_name_the_same_four_functions() {
    let functions = rust_functions();
    assert_eq!(
        functions,
        [
            "ym_apple_auto_availability",
            "ym_apple_auto_model",
            "ym_apple_auto_os",
            "ym_apple_auto_ask"
        ]
    );
    let header = squeeze(&header());
    let swift = swift();
    for name in &functions {
        assert!(
            header.contains(&format!("int32_t {name}(")),
            "ym_apple.h lacks {name}"
        );
        assert!(
            swift.contains(&format!("@_cdecl(\"{name}\")")),
            "AppleAuto.swift lacks @_cdecl(\"{name}\")"
        );
    }
    assert_eq!(header.matches("int32_t ym_apple_auto_").count(), 4);
    assert_eq!(swift.matches("@_cdecl(").count(), 4);

    // The header's declarations are section 11's signatures.
    let spec = squeeze(&section(&spec(), 11));
    let signatures: Vec<&str> = spec
        .split('`')
        .filter(|s| s.starts_with("int32_t ym_apple_auto_"))
        .collect();
    assert_eq!(signatures.len(), 4, "section 11 gives four signatures");
    for signature in signatures {
        assert!(
            header.contains(&format!("{signature};")),
            "ym_apple.h does not declare `{signature}`"
        );
    }
}

#[test]
fn the_answer_layout_agrees_everywhere() {
    let header = header();
    assert!(header.contains(&format!("#define YM_APPLE_SCENE_BYTES {SCENE_BYTES}")));
    assert!(header.contains(&format!("#define YM_APPLE_PLAN_BYTES {PLAN_BYTES}")));
    assert!(header.contains("#define YM_APPLE_UNSET INT32_MIN"));
    assert_eq!(UNSET, i32::MIN);
    let ints = [
        "exposure",
        "highlights",
        "shadows",
        "blacks",
        "contrast",
        "white",
        "tint",
        "saturation",
        "shadow_color",
        "shadow_color_strength",
        "highlight_color",
        "highlight_color_strength",
    ];
    let declared: String = ints.iter().map(|f| format!("int32_t {f}; ")).collect();
    assert!(squeeze(&header).contains(&format!(
        "typedef struct YmAppleAnswer {{ {declared}char scene[YM_APPLE_SCENE_BYTES]; \
         char plan[YM_APPLE_PLAN_BYTES]; }} YmAppleAnswer;"
    )));
    assert!(header.contains(&format!(
        "sizeof(YmAppleAnswer) == {}",
        size_of::<YmAppleAnswer>()
    )));
    let spec = squeeze(&section(&spec(), 11));
    assert!(
        spec.contains(&format!(
            "YmAppleAnswer {{ int32_t {}; char scene[{SCENE_BYTES}]; char plan[{PLAN_BYTES}]; }}",
            ints.join(", ")
        )),
        "section 11 does not give the struct"
    );
    let swift = swift();
    let layout = block(&swift, "private enum AnswerLayout {");
    let offsets = [
        std::mem::offset_of!(YmAppleAnswer, exposure),
        std::mem::offset_of!(YmAppleAnswer, highlights),
        std::mem::offset_of!(YmAppleAnswer, shadows),
        std::mem::offset_of!(YmAppleAnswer, blacks),
        std::mem::offset_of!(YmAppleAnswer, contrast),
        std::mem::offset_of!(YmAppleAnswer, white),
        std::mem::offset_of!(YmAppleAnswer, tint),
        std::mem::offset_of!(YmAppleAnswer, saturation),
        std::mem::offset_of!(YmAppleAnswer, shadow_color),
        std::mem::offset_of!(YmAppleAnswer, shadow_color_strength),
        std::mem::offset_of!(YmAppleAnswer, highlight_color),
        std::mem::offset_of!(YmAppleAnswer, highlight_color_strength),
    ];
    let mut expected: Vec<(String, usize)> = ints
        .iter()
        .zip(offsets)
        .map(|(f, o)| (camel(f), o))
        .collect();
    expected.extend([
        (
            "scene".to_owned(),
            std::mem::offset_of!(YmAppleAnswer, scene),
        ),
        ("sceneBytes".to_owned(), SCENE_BYTES),
        ("plan".to_owned(), std::mem::offset_of!(YmAppleAnswer, plan)),
        ("planBytes".to_owned(), PLAN_BYTES),
    ]);
    for (field, offset) in expected {
        let line = format!("static let {field} = {offset}");
        assert!(layout.contains(&line), "AnswerLayout lacks `{line}`");
    }
    assert!(layout.contains("static let unset = Int32.min"));
    // Every field is written, and cleared to unset before the request.
    let write = block(&swift, "private func writeAnswer(");
    for f in ints {
        assert!(
            write.contains(&format!("(AnswerLayout.{0}, filled.{0})", camel(f))),
            "writeAnswer does not write {f}"
        );
    }
}

#[test]
fn every_foundation_models_use_is_behind_macos_27() {
    // Each declaration that names a Foundation Models type, and each C entry point that reaches
    // one, is marked or checked for macOS 27 (the binary starts on macOS 14).
    let swift = swift();
    let lines: Vec<&str> = swift.lines().collect();
    let mut checked = 0;
    for (i, line) in lines.iter().enumerate() {
        // A top-level declaration whose first line opens its body.
        let declares = !line.starts_with([' ', '}', '/', '@'])
            && !line.starts_with("import ")
            && line.trim_end().ends_with('{');
        if !declares {
            continue;
        }
        let attributes: Vec<&str> = lines[..i]
            .iter()
            .rev()
            .take_while(|l| l.starts_with('@') || l.starts_with("///"))
            .copied()
            .collect();
        let body = block(&swift, line);
        let uses_models = attributes.iter().any(|a| a.starts_with("@Generable"))
            || [
                "SystemLanguageModel",
                "LanguageModelSession",
                "LanguageModelError",
                "GenerationOptions",
                "Attachment(",
                "@Guide(",
            ]
            .iter()
            .any(|word| body.contains(word));
        if uses_models {
            checked += 1;
            assert!(
                attributes.contains(&"@available(macOS 27.0, *)")
                    || body.contains("guard #available(macOS 27.0, *) else"),
                "`{line}` uses Foundation Models without @available(macOS 27.0, *)"
            );
        }
    }
    // Answer, its eleven enums, currentAvailability, askModel and askCode at least.
    assert!(
        checked >= 15,
        "only {checked} declarations use Foundation Models"
    );
    // The C entry points that reach the model check the system first.
    for entry in [
        "ymAppleAutoAvailability",
        "ymAppleAutoModel",
        "ymAppleAutoAsk",
    ] {
        let body = block(&swift, &format!("public func {entry}("));
        assert!(
            body.contains("guard #available(macOS 27.0, *) else"),
            "{entry} does not check for macOS 27"
        );
    }
}
