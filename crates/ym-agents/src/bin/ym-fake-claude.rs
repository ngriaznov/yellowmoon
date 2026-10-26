//! A test double for Claude Code (`claude`) 2.1.310 (`docs/22` section 15.2): it prints what the real
//! program prints for the probes and for one request, checks what it is given and exits 99 when the
//! request is one the real program would not take. Not shipped; `cargo test -p ym-agents` builds it.
//! The scenarios and the variables are in `src/fakes.rs`.

#[path = "../fakes.rs"]
mod fakes;

use base64::Engine;
use fakes::{
    Recorder, answer_for, decode_jpeg, first_word, flood, hang, has, maybe_sleeper, must_fail, out,
    read_stdin, reject, request_index, requests_so_far, require, scenario, slow, twist, value_of,
    version,
};
use serde_json::{Value, json};

fn main() {
    maybe_sleeper();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenario = scenario();
    let recorder = Recorder::start();
    fakes::maybe_raw(&recorder, &scenario);

    if args == ["--version"] {
        if scenario == "probe_hang" {
            hang();
        }
        let default = if scenario == "too_old" {
            "2.1.100"
        } else {
            "2.1.310"
        };
        println!("{} (Claude Code)", version(default));
        return;
    }
    if args == ["auth", "status"] {
        auth_status(&scenario);
        return;
    }
    if has(&args, "-p") {
        request(&args, &scenario, &recorder);
        return;
    }
    reject(&format!("arguments not understood: {args:?}"));
}

/// `claude auth status`: JSON on standard output, exit 0 signed in and 1 signed out.
fn auth_status(scenario: &str) {
    if matches!(scenario, "probe_hang" | "signin_hang") {
        hang();
    }
    let signed_out =
        scenario == "signed_out" || (scenario == "late_signout" && requests_so_far() >= 1);
    let api = std::env::var("YM_FAKE_API_PROVIDER").unwrap_or_else(|_| "firstParty".to_owned());
    if signed_out {
        out(
            json!({"loggedIn": false, "authMethod": "none", "apiProvider": api})
                .to_string()
                .as_bytes(),
        );
        out(b"\n");
        std::process::exit(1);
    }
    let method = std::env::var("YM_FAKE_AUTH_METHOD").unwrap_or_else(|_| "oauth_token".to_owned());
    out(json!({
        "loggedIn": true,
        "authMethod": method,
        "apiProvider": api,
        "analyticsDisabled": false,
        "projectsDirectory": "/home/fake/.claude/projects",
        "configDirectory": "/home/fake/.claude"
    })
    .to_string()
    .as_bytes());
    out(b"\n");
}

/// One request: `claude -p …` with a stream-json line on standard input.
fn request(args: &[String], scenario: &str, recorder: &Recorder) {
    // The argument list of docs/22 section 7.2.
    require(args, "-p");
    if value_of(args, "--input-format").map(String::as_str) != Some("stream-json") {
        reject("--input-format stream-json is missing");
    }
    if value_of(args, "--output-format").map(String::as_str) != Some("stream-json") {
        reject("--output-format stream-json is missing");
    }
    for flag in [
        "--verbose",
        "--safe-mode",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--strict-mcp-config",
    ] {
        require(args, flag);
    }
    if value_of(args, "--tools").map(String::as_str) != Some("") {
        reject("--tools must be followed by an empty argument");
    }
    for flag in ["--max-turns", "--model", "--effort"] {
        if value_of(args, flag).is_none_or(String::is_empty) {
            reject(&format!("{flag} needs a value"));
        }
    }
    if args
        .iter()
        .any(|a| a.starts_with("--dangerously") || a == "--bare")
    {
        reject("an option Yellowmoon never passes");
    }
    let Some(prompt_file) = value_of(args, "--system-prompt-file") else {
        reject("--system-prompt-file is missing");
    };
    let Ok(instructions) = std::fs::read_to_string(prompt_file) else {
        reject("--system-prompt-file does not name a readable file");
    };
    let Some(schema) = value_of(args, "--json-schema") else {
        reject("--json-schema is missing");
    };
    fakes::check_schema(schema);
    let cwd = std::env::current_dir().unwrap_or_default();
    if std::fs::read_dir(&cwd).map(|d| d.count()).unwrap_or(1) != 0 {
        reject("the working folder is not empty");
    }

    if scenario == "crash" {
        eprintln!("fatal: key sk-ant-FAKE0123456789 was rejected");
        std::process::exit(3);
    }
    if scenario == "slow" {
        slow();
    }
    // `backpressure`: write 2 MB before reading a byte of standard input.
    if scenario == "backpressure" {
        flood(2);
    }

    let counting = scenario == "late_signout" || std::env::var_os("YM_FAKE_FAILS").is_some();
    let index = if counting { request_index() } else { 0 };

    let stdin = read_stdin();
    recorder.write("stdin.bin", &stdin);
    let line = stdin
        .split(|b| *b == b'\n')
        .find(|l| !l.is_empty())
        .unwrap_or(&[]);
    let Ok(message) = serde_json::from_slice::<Value>(line) else {
        reject("standard input does not start with a JSON line");
    };
    let content = message
        .pointer("/message/content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if message.get("type") != Some(&json!("user")) || content.len() < 2 {
        reject("the line is not a user message with an image and a text block");
    }
    let image = &content[0];
    if image.get("type") != Some(&json!("image"))
        || image.pointer("/source/type") != Some(&json!("base64"))
        || image.pointer("/source/media_type") != Some(&json!("image/jpeg"))
    {
        reject("the first block is not a base64 image/jpeg block");
    }
    if content[1].get("type") != Some(&json!("text")) {
        reject("the second block is not a text block");
    }
    let data = image
        .pointer("/source/data")
        .and_then(Value::as_str)
        .unwrap_or("");
    let Ok(jpeg) = base64::engine::general_purpose::STANDARD.decode(data) else {
        reject("the image data is not base64");
    };
    let picture = match decode_jpeg(&jpeg) {
        Ok(p) => p,
        Err(why) => reject(&why),
    };

    fakes::delay();
    match scenario {
        "signed_out" => error_result("Not logged in · Please run /login", None, &[]),
        "late_signout" if index >= 1 => {
            error_result("Not logged in · Please run /login", None, &[])
        }
        "unauthenticated" => error_result(
            "API Error: request failed",
            None,
            &[
                json!({"type": "system", "subtype": "api_retry", "attempt": 1, "error": "authentication_failed"}),
            ],
        ),
        "rate_limit" => error_result(
            "API Error: rate limited",
            Some(429),
            &[
                json!({"type": "system", "subtype": "api_retry", "attempt": 1, "error": "rate_limit"}),
            ],
        ),
        _ if must_fail(index) => error_result("fake failure", Some(500), &[]),
        "no_structured" => success(
            None,
            "Here is my opinion of the photo, in words.",
            "end_turn",
        ),
        "refused" => success(None, "I can't help with that.", "refusal"),
        "huge" => {
            flood(6);
            success(
                Some(answer(&picture, &instructions, scenario)),
                "",
                "end_turn",
            );
        }
        _ => success(
            Some(answer(&picture, &instructions, scenario)),
            "",
            "end_turn",
        ),
    }
}

/// The scenario's answer object.
fn answer(picture: &fakes::Picture, instructions: &str, scenario: &str) -> Value {
    Value::Object(twist(
        answer_for(picture, &first_word(instructions)),
        scenario,
    ))
}

/// The events and the success `result` line.
fn success(structured: Option<Value>, text: &str, stop_reason: &str) {
    out(b"{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"fake\",\"model\":\"claude-fake-1\",\"tools\":[\"StructuredOutput\"]}\n");
    out(b"\n");
    out(b"this line is not JSON\n");
    if let Some(structured) = &structured {
        let event = json!({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "StructuredOutput", "input": structured}]}});
        out(format!("{event}\n").as_bytes());
    }
    let mut result = json!({
        "type": "result", "subtype": "success", "is_error": false, "result": text,
        "stop_reason": stop_reason, "terminal_reason": "completed", "num_turns": 2,
        "duration_ms": 120, "total_cost_usd": 0.001,
        "usage": {"input_tokens": 1500, "output_tokens": 120},
        "modelUsage": {
            "claude-fake-aux": {"inputTokens": 10, "outputTokens": 2},
            "claude-fake-1": {"inputTokens": 1500, "outputTokens": 120}
        },
        "permission_denials": [], "session_id": "fake", "uuid": "fake"
    });
    if let Some(structured) = structured {
        result["structured_output"] = structured;
    }
    out(format!("{result}\n").as_bytes());
}

/// An error `result` line (after the `api_retry` events) and exit code 1.
fn error_result(text: &str, status: Option<u16>, events: &[Value]) {
    for event in events {
        out(format!("{event}\n").as_bytes());
    }
    let result = json!({
        "type": "result", "subtype": "success", "is_error": true, "result": text,
        "api_error_status": status, "terminal_reason": "api_error", "num_turns": 1,
        "session_id": "fake", "uuid": "fake"
    });
    out(format!("{result}\n").as_bytes());
    std::process::exit(1);
}
