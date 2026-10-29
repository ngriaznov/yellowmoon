//! A test double for Antigravity (`agy`) 1.4.7 (`docs/22` section 15.2): it prints what the real program
//! is documented to print for the probes and for one request, checks what it is given and exits 99
//! when the request is one the real program would not take. Not shipped; `cargo test -p ym-agents`
//! builds it. The scenarios and the variables are in `src/fakes.rs`.

#[path = "../fakes.rs"]
mod fakes;

use fakes::{
    Recorder, answer_for, counting, decode_jpeg, first_word, flood, hang, has, maybe_sleeper,
    must_fail, out, planned, reject, request_index, requests_so_far, require, scenario, slow,
    twist, value_of, version,
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
            "1.0.9"
        } else {
            "1.4.7"
        };
        println!("{}", version(default));
        return;
    }
    if has(&args, "-p") {
        let prompt = value_of(&args, "-p").cloned().unwrap_or_default();
        if prompt == "/usage" {
            usage(&args, &scenario);
        } else {
            request(&args, &prompt, &scenario, &recorder);
        }
        return;
    }
    reject(&format!("arguments not understood: {args:?}"));
}

/// The message of a signed-out run, on standard error, exit 1.
fn signed_out() -> ! {
    eprintln!("Error: authentication required. Run 'agy' to log in.");
    std::process::exit(1);
}

/// `agy -p /usage --output-format json`: the sign-in check.
fn usage(args: &[String], scenario: &str) {
    if matches!(scenario, "probe_hang" | "signin_hang") {
        hang();
    }
    if value_of(args, "--output-format").map(String::as_str) != Some("json") {
        reject("--output-format json is missing");
    }
    if scenario == "signed_out" || (scenario == "late_signout" && requests_so_far() >= 1) {
        signed_out();
    }
    out(json!({"status": "SUCCESS", "response": "quota: plenty"})
        .to_string()
        .as_bytes());
    out(b"\n");
}

/// One request: `agy -p <prompt> …` in a folder that holds `photo.jpg`.
fn request(args: &[String], prompt: &str, scenario: &str, recorder: &Recorder) {
    // The argument list of docs/22 section 8.2.
    if value_of(args, "--output-format").map(String::as_str) != Some("json") {
        reject("--output-format json is missing");
    }
    require(args, "--sandbox");
    require(args, "--disable-slash-commands");
    if args
        .iter()
        .any(|a| a.starts_with("--dangerously") || a == "--continue")
    {
        reject("an option Yellowmoon never passes");
    }
    for flag in ["--effort", "--print-timeout"] {
        if value_of(args, flag).is_none_or(String::is_empty) {
            reject(&format!("{flag} needs a value"));
        }
    }
    let Some(schema_path) = value_of(args, "--json-schema") else {
        reject("--json-schema is missing");
    };
    let Ok(schema_text) = std::fs::read_to_string(schema_path) else {
        reject("--json-schema does not name a readable file");
    };
    fakes::check_schema(&schema_text);
    if !prompt.contains("@photo.jpg") {
        reject("the prompt does not mention @photo.jpg");
    }
    if prompt.starts_with('-') || prompt.starts_with('/') {
        reject("the prompt begins with - or /");
    }
    // The working folder holds exactly the photograph.
    let cwd = std::env::current_dir().unwrap_or_default();
    let names: Vec<String> = std::fs::read_dir(&cwd)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    if names != ["photo.jpg"] {
        reject(&format!(
            "the working folder holds {names:?}, not exactly photo.jpg"
        ));
    }
    let Ok(jpeg) = std::fs::read(cwd.join("photo.jpg")) else {
        reject("photo.jpg cannot be read");
    };
    let picture = match decode_jpeg(&jpeg) {
        Ok(p) => p,
        Err(why) => reject(&why),
    };

    // The first word of the instructions: they follow the first line and a blank line.
    let instructions = prompt.split_once("\n\n").map_or("", |(_, rest)| rest);
    let word = first_word(instructions);

    if scenario == "crash" {
        eprintln!("fatal: key sk-ant-FAKE0123456789 was rejected");
        std::process::exit(3);
    }
    if matches!(scenario, "slow" | "slow_deaf") {
        slow();
    }
    if scenario == "backpressure" {
        flood(2);
    }

    let index = if counting(scenario) {
        request_index()
    } else {
        0
    };

    fakes::delay();
    // What `YM_FAKE_PLAN` says this request does comes before the scenario.
    match planned(index).as_deref() {
        Some("fail") => {
            eprintln!("Error: fake failure");
            std::process::exit(1);
        }
        Some("rate") => {
            eprintln!(
                "AGY_ERROR: {{\"status\":\"RESOURCE_EXHAUSTED\",\"code\":429,\"retryable\":true,\"id\":\"fake\"}}"
            );
            std::process::exit(3);
        }
        Some("bad") => {
            let answer = Value::Object(twist(answer_for(&picture, &word), "malformed"));
            envelope("SUCCESS", Some(answer), None, &[]);
            return;
        }
        Some("signedout") => signed_out(),
        Some("refused") => {
            reject("the agy double has no refusal: use fail, rate, bad or signedout")
        }
        _ => {}
    }
    match scenario {
        "signed_out" => signed_out(),
        "late_signout" if index >= 1 => signed_out(),
        "exit1" => {
            eprintln!("Error: something failed");
            std::process::exit(1);
        }
        "exit2" => {
            eprintln!("flag provided but not defined: -p");
            std::process::exit(2);
        }
        "rate_limit" => {
            eprintln!(
                "AGY_ERROR: {{\"status\":\"RESOURCE_EXHAUSTED\",\"code\":429,\"retryable\":true,\"id\":\"fake\"}}"
            );
            std::process::exit(3);
        }
        "unauthenticated" => {
            eprintln!(
                "AGY_ERROR: {{\"status\":\"UNAUTHENTICATED\",\"code\":401,\"retryable\":false,\"id\":\"fake\"}}"
            );
            std::process::exit(3);
        }
        _ if must_fail(index) => {
            eprintln!("Error: fake failure");
            std::process::exit(1);
        }
        "status_error" => envelope("ERROR", None, Some("the model is unavailable"), &[]),
        "print_timeout" => envelope("INTERRUPTED", None, None, &[]),
        "no_structured" => envelope("SUCCESS", None, None, &[]),
        "denied" => {
            let answer = Value::Object(twist(answer_for(&picture, &word), scenario));
            envelope(
                "SUCCESS",
                Some(answer),
                None,
                &[json!({"tool": "run_command", "reason": "denied by the sandbox"})],
            );
        }
        "huge" => {
            flood(6);
            let answer = Value::Object(twist(answer_for(&picture, &word), scenario));
            envelope("SUCCESS", Some(answer), None, &[]);
        }
        _ => {
            let answer = Value::Object(twist(answer_for(&picture, &word), scenario));
            envelope("SUCCESS", Some(answer), None, &[]);
        }
    }
    let _ = recorder;
}

/// The envelope on standard output, after a banner line (which the reader must skip), and exit 0.
fn envelope(status: &str, structured: Option<Value>, error: Option<&str>, denied: &[Value]) {
    out(b"Antigravity CLI 1.4.7\n");
    let mut envelope = json!({
        "conversation_id": "fake-conversation",
        "status": status,
        "response": if structured.is_some() { "{...}" } else { "Here is my opinion of the photo, in words." },
        "duration_seconds": 0.1,
        "num_turns": 1,
        "usage": {"input_tokens": 10500, "output_tokens": 120, "total_tokens": 10620}
    });
    if let Some(structured) = structured {
        envelope["structured_output"] = structured;
    }
    if let Some(error) = error {
        envelope["error"] = json!(error);
    }
    if !denied.is_empty() {
        envelope["denied_actions"] = json!(denied);
    }
    out(format!("{envelope}\n").as_bytes());
}
