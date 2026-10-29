//! Claude Code as an Auto provider (`docs/22` section 7): the person's own `claude`, given the picture as a
//! base64 image block on standard input, with no tools, one request, and its answer in the
//! `structured_output` of the last `result` event.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use serde_json::Value;
use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorPicture, AutoAdvisor, Availability, FailureKind,
};
use ym_core::auto_provider::{AdvisorDetails, ChangeListener};

use crate::advisor::{Core, Fail, Source};
use crate::discovery::Locations;
use crate::jpeg;
use crate::probe::Level;
use crate::process::{CancelToken, EnvProfile, Outcome, RunOutput, RunSpec, message, run};
use crate::schema;
use crate::tool::{Tool, Tuning};

/// The request line after the picture (`docs/22` section 7.2): fixed, whatever the instructions say.
const REQUEST_LINE: &str = "Grade this photo.";

/// Claude Code (`claude`) as an [`AutoAdvisor`].
pub struct ClaudeAdvisor {
    core: Core,
}

impl ClaudeAdvisor {
    /// The person's Claude Code, found as `docs/22` section 4 says. No probe has run yet.
    #[must_use]
    pub fn system() -> Arc<Self> {
        Self::custom(
            Source::System(Locations::from_env()),
            Tuning::for_tool(Tool::Claude),
        )
    }

    /// This program, with discovery skipped (the tests, and `exec:` of the test hook).
    #[must_use]
    pub fn with_program(path: PathBuf) -> Arc<Self> {
        Self::custom(Source::Program(path), Tuning::for_tool(Tool::Claude))
    }

    /// Any source and tuning (the tests shorten the limits and name a double's scenario).
    #[doc(hidden)]
    #[must_use]
    pub fn custom(source: Source, tuning: Tuning) -> Arc<Self> {
        Arc::new(Self {
            core: Core::new(Tool::Claude, source, tuning),
        })
    }

    /// Starts the first probe (`StartProbe::Now`): everything, the sign-in check included.
    #[doc(hidden)]
    pub fn start(&self) {
        self.core.refresh(Level::Full);
    }

    /// The request of `docs/22` section 7.2, as its parts: the arguments, the instructions' file and
    /// the standard-input line.
    fn run_request(
        &self,
        program: &Path,
        cancel: CancelToken,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, Fail> {
        let tuning = self.core.tuning();
        let jpeg = jpeg::encode(picture).map_err(|e| Fail::new(FailureKind::Other, e.to_string()))?;
        let workdir = tuning.workdir().map_err(|e| {
            Fail::new(
                FailureKind::Other,
                message(&format!("could not make a temporary folder: {e}")),
            )
        })?;
        let instructions_file = workdir.input().join("instructions.txt");
        std::fs::write(&instructions_file, instructions.as_bytes()).map_err(|e| {
            Fail::new(
                FailureKind::Other,
                message(&format!("could not write the instructions: {e}")),
            )
        })?;

        let spec = RunSpec::new(program, workdir.cwd(), EnvProfile::Claude)
            .args(arguments(
                &self.core.model_alias(),
                &self.core.effort(),
                &instructions_file,
            ))
            .stdin(stdin_line(&jpeg))
            .timeout(tuning.request_timeout)
            .cancel(cancel)
            .extra_env(tuning.extra_env.clone());
        let output = run(spec);
        // The folder is deleted when `workdir` drops, after the run has ended.
        interpret(&output)
    }
}

/// The argument list of `docs/22` section 7.2, one by one.
fn arguments(model: &str, effort: &str, instructions_file: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--safe-mode",
        "--tools",
        "",
        "--no-session-persistence",
        "--max-turns",
        "3",
        "--model",
        model,
        "--effort",
        effort,
        "--system-prompt-file",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    args.push(instructions_file.as_os_str().to_owned());
    args.extend(
        ["--json-schema", schema::schema_text(), "--disable-slash-commands", "--strict-mcp-config"]
            .iter()
            .map(OsString::from),
    );
    args
}

/// The one line of standard input: a user message with the picture, then the request line.
fn stdin_line(jpeg: &[u8]) -> Vec<u8> {
    let data = base64::engine::general_purpose::STANDARD.encode(jpeg);
    format!(
        "{{\"type\":\"user\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"image\",\
         \"source\":{{\"type\":\"base64\",\"media_type\":\"image/jpeg\",\"data\":\"{data}\"}}}},\
         {{\"type\":\"text\",\"text\":\"{REQUEST_LINE}\"}}]}}}}\n"
    )
    .into_bytes()
}

/// Reads what the program printed (`docs/22` section 7.3). The exit code alone decides nothing: a
/// signed-out run has `subtype: "success"` and `is_error: true`.
fn interpret(output: &RunOutput) -> Result<AdvisorAnswer, Fail> {
    match &output.outcome {
        Outcome::SpawnFailed(why) => {
            return Err(Fail::new(
                FailureKind::CliFailed,
                format!("Claude Code could not be started: {why}"),
            ));
        }
        Outcome::TimedOut => return Err(Fail::new(FailureKind::Timeout, "")),
        Outcome::Cancelled => return Err(Fail::Cancelled),
        Outcome::OutputTooLarge => {
            return Err(Fail::new(
                FailureKind::CliFailed,
                "Claude Code produced too much output",
            ));
        }
        Outcome::Exited => {}
    }

    // Newline-delimited JSON events: blank and unparsable lines are skipped, the last `result` is the
    // answer, and the `api_retry` events say what the program had to retry.
    let mut result: Option<Value> = None;
    let mut retry_errors: Vec<String> = Vec::new();
    for line in output.stdout.split(|b| *b == b'\n') {
        let Ok(event) = serde_json::from_slice::<Value>(line) else {
            continue;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("result") => result = Some(event),
            Some("system") if event.get("subtype").and_then(Value::as_str) == Some("api_retry") => {
                if let Some(error) = event.get("error").and_then(Value::as_str) {
                    retry_errors.push(error.to_owned());
                }
            }
            _ => {}
        }
    }
    let Some(result) = result else {
        let code = output
            .exit_code
            .map_or_else(|| "none".to_owned(), |c| c.to_string());
        return Err(Fail::new(
            FailureKind::CliFailed,
            format!("exit code {code}: {}", output.stderr_message()),
        ));
    };

    let text = result.get("result").and_then(Value::as_str).unwrap_or("");
    let status = result.get("api_error_status").and_then(Value::as_u64);
    let retried = |name: &str| retry_errors.iter().any(|e| e == name);
    if result.get("is_error").and_then(Value::as_bool) == Some(true) {
        if text.starts_with("Not logged in")
            || retried("authentication_failed")
            || matches!(status, Some(401 | 403))
        {
            return Err(Fail::new(FailureKind::NotSignedIn, ""));
        }
        if status == Some(429) || retried("rate_limit") {
            return Err(Fail::new(FailureKind::RateLimited, ""));
        }
        return Err(Fail::new(FailureKind::CliFailed, tail_or(text)));
    }

    match result.get("structured_output").filter(|v| !v.is_null()) {
        Some(structured) => {
            let parsed =
                schema::parse_answer(structured).map_err(|e| Fail::new(FailureKind::BadAnswer, e))?;
            AdvisorAnswer {
                steps: parsed.steps,
                scene: parsed.scene,
                plan: parsed.plan,
                served_model: served_model(&result),
            }
            .checked()
            .map_err(|e| Fail::new(e.kind, e.message))
        }
        None if result.get("stop_reason").and_then(Value::as_str) == Some("refusal") => {
            Err(Fail::new(FailureKind::Refused, ""))
        }
        None => Err(Fail::new(
            FailureKind::BadAnswer,
            "Claude Code answered in words instead of the answer form",
        )),
    }
}

/// An error text for a message: redacted and cut.
fn tail_or(text: &str) -> String {
    if text.trim().is_empty() {
        "Claude Code reported an error".to_owned()
    } else {
        message(text)
    }
}

/// The id of the model that answered: the key of `modelUsage` with the most output tokens (the
/// object has one key in practice).
fn served_model(result: &Value) -> Option<String> {
    result
        .get("modelUsage")?
        .as_object()?
        .iter()
        .max_by_key(|(_, usage)| usage.get("outputTokens").and_then(Value::as_u64).unwrap_or(0))
        .map(|(model, _)| model.clone())
}

impl AutoAdvisor for ClaudeAdvisor {
    fn availability(&self) -> Availability {
        self.core.availability()
    }

    fn ask(
        &self,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        self.core
            .execute(|program, cancel| self.run_request(program, cancel, picture, instructions))
    }

    fn concurrency(&self) -> usize {
        2
    }

    fn refresh(&self) {
        self.core.refresh(Level::Full);
    }

    fn refresh_asked(&self) {
        self.core.refresh_asked();
    }

    fn settle(&self, timeout: Duration) {
        self.core.settle(timeout);
    }

    fn cancel(&self) {
        self.core.cancel();
    }

    fn wait_idle(&self, timeout: Duration) -> bool {
        self.core.wait_idle(timeout)
    }

    fn on_change(&self, listener: ChangeListener) {
        self.core.on_change(listener);
    }

    fn details(&self) -> AdvisorDetails {
        self.core.details()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn exited(stdout: &str, code: i32) -> RunOutput {
        RunOutput {
            outcome: Outcome::Exited,
            exit_code: Some(code),
            stdout: stdout.as_bytes().to_vec(),
            stderr: b"stderr text".to_vec(),
            elapsed: Duration::ZERO,
        }
    }

    fn result_line(extra: Value) -> String {
        let mut base = json!({"type": "result", "subtype": "success", "is_error": false, "result": ""});
        for (k, v) in extra.as_object().unwrap() {
            base[k] = v.clone();
        }
        format!("{base}\n")
    }

    fn answer() -> Value {
        json!({"scene": "s", "plan": "p", "exposure": "brighter"})
    }

    fn kind(output: &RunOutput) -> FailureKind {
        match interpret(output) {
            Err(Fail::Failed(kind, _)) => kind,
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    #[test]
    fn the_stdin_line_is_the_specs() {
        let line = String::from_utf8(stdin_line(b"\xFF\xD8\xFF")).unwrap();
        assert!(line.starts_with(r#"{"type":"user","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":""#));
        assert!(line.ends_with("\"}},{\"type\":\"text\",\"text\":\"Grade this photo.\"}]}}\n"));
        assert_eq!(line.matches('\n').count(), 1);
        let value: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(value["message"]["content"][1]["text"], "Grade this photo.");
    }

    #[test]
    fn the_arguments_are_the_specs_one_by_one() {
        let args = arguments("sonnet", "low", Path::new("/t/in/instructions.txt"));
        let text: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(&text[..9], ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--safe-mode", "--tools", ""]);
        assert!(text.windows(2).any(|w| w == ["--model", "sonnet"]));
        assert!(text.windows(2).any(|w| w == ["--effort", "low"]));
        assert!(text.windows(2).any(|w| w == ["--system-prompt-file", "/t/in/instructions.txt"]));
        assert!(text.windows(2).any(|w| w[0] == "--json-schema" && w[1] == schema::schema_text()));
        for flag in ["--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config"] {
            assert!(text.iter().any(|a| a == flag), "{flag}");
        }
        assert!(!text.iter().any(|a| a.contains("dangerously") || a == "--bare"));
    }

    #[test]
    fn a_good_answer_names_the_model_with_most_output() {
        let out = exited(
            &format!(
                "{}{}",
                "{\"type\":\"system\",\"subtype\":\"init\"}\nnot json\n\n",
                result_line(json!({
                    "structured_output": answer(),
                    "modelUsage": {"a": {"outputTokens": 3}, "claude-sonnet-5-5": {"outputTokens": 90}}
                }))
            ),
            0,
        );
        let answer = interpret(&out).unwrap();
        assert_eq!(answer.served_model.as_deref(), Some("claude-sonnet-5-5"));
        assert_eq!(answer.steps.exposure, Some(2));
    }

    #[test]
    fn the_failures_map_as_the_table_says() {
        let signed_out = exited(&result_line(json!({"is_error": true, "result": "Not logged in · Please run /login"})), 1);
        assert_eq!(kind(&signed_out), FailureKind::NotSignedIn);
        let auth_retry = exited(
            &format!(
                "{}\n{}",
                json!({"type": "system", "subtype": "api_retry", "error": "authentication_failed"}),
                result_line(json!({"is_error": true, "result": "API Error"}))
            ),
            1,
        );
        assert_eq!(kind(&auth_retry), FailureKind::NotSignedIn);
        let http_401 = exited(&result_line(json!({"is_error": true, "result": "x", "api_error_status": 401})), 1);
        assert_eq!(kind(&http_401), FailureKind::NotSignedIn);
        let limited = exited(&result_line(json!({"is_error": true, "result": "x", "api_error_status": 429})), 1);
        assert_eq!(kind(&limited), FailureKind::RateLimited);
        let retry_limited = exited(
            &format!(
                "{}\n{}",
                json!({"type": "system", "subtype": "api_retry", "error": "rate_limit"}),
                result_line(json!({"is_error": true, "result": "x"}))
            ),
            1,
        );
        assert_eq!(kind(&retry_limited), FailureKind::RateLimited);
        let other = exited(&result_line(json!({"is_error": true, "result": "boom"})), 1);
        assert_eq!(kind(&other), FailureKind::CliFailed);
        // A retry that worked is not a failure.
        let fine = exited(
            &format!(
                "{}\n{}",
                json!({"type": "system", "subtype": "api_retry", "error": "rate_limit"}),
                result_line(json!({"structured_output": answer()}))
            ),
            0,
        );
        assert!(interpret(&fine).is_ok());
    }

    #[test]
    fn answers_without_the_form_are_bad_or_refused() {
        let words = exited(&result_line(json!({"result": "in words", "stop_reason": "end_turn"})), 0);
        assert_eq!(kind(&words), FailureKind::BadAnswer);
        let refused = exited(&result_line(json!({"stop_reason": "refusal"})), 0);
        assert_eq!(kind(&refused), FailureKind::Refused);
        let malformed = exited(&result_line(json!({"structured_output": {"scene": "s", "plan": "p", "exposure": "muchBrightest"}})), 0);
        assert_eq!(kind(&malformed), FailureKind::BadAnswer);
        let empty = exited(&result_line(json!({"structured_output": {"scene": "s", "plan": "p"}})), 0);
        assert_eq!(kind(&empty), FailureKind::BadAnswer, "no value set");
    }

    #[test]
    fn no_result_line_is_a_cli_failure_with_the_redacted_stderr() {
        let mut out = exited("garbage\n", 3);
        out.stderr = b"fatal: key sk-ant-FAKE0123456789 rejected".to_vec();
        let Err(Fail::Failed(FailureKind::CliFailed, message)) = interpret(&out) else {
            panic!("expected cli_failed");
        };
        assert!(message.contains("exit code 3"), "{message}");
        assert!(!message.contains("FAKE0123456789"), "{message}");
        assert!(message.contains("rejected"), "{message}");
    }

    #[test]
    fn the_other_outcomes() {
        let mut out = exited("", 0);
        for (outcome, want) in [
            (Outcome::TimedOut, FailureKind::Timeout),
            (Outcome::OutputTooLarge, FailureKind::CliFailed),
            (Outcome::SpawnFailed("no such file".into()), FailureKind::CliFailed),
        ] {
            out.outcome = outcome;
            assert_eq!(kind(&out), want);
        }
        out.outcome = Outcome::Cancelled;
        assert_eq!(interpret(&out), Err(Fail::Cancelled));
    }
}
