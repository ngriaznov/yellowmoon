//! Antigravity (`agy`) as an Auto provider (`docs/22` section 8): the person's own `agy`, given the picture
//! as `photo.jpg` in an otherwise empty working folder and named as `@photo.jpg` in its prompt, in a
//! sandbox, never with permissions skipped. Experimental: everything here is from Google's documentation,
//! no request to `agy` has been run (`docs/22` section 19).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorPicture, AutoAdvisor, Availability, FailureKind,
};
use ym_core::auto_provider::{AdvisorDetails, ChangeListener};

use crate::advisor::{Core, Fail, Source};
use crate::discovery::Locations;
use crate::jpeg;
use crate::probe::Level;
use crate::process::{
    CancelToken, EnvProfile, Outcome, RunOutput, RunSpec, Workdir, message, run,
};
use crate::schema;
use crate::tool::{Tool, Tuning};

/// The program's own limit on a turn: shorter than ours, so that it ends the turn itself, returns what it
/// has and exits 0 (`docs/22` D12).
const PRINT_TIMEOUT: &str = "110s";

/// Antigravity (`agy`) as an [`AutoAdvisor`].
pub struct AgyAdvisor {
    core: Core,
}

impl AgyAdvisor {
    /// The person's Antigravity, found as `docs/22` section 4 says. No probe has run yet.
    #[must_use]
    pub fn system() -> Arc<Self> {
        Self::custom(
            Source::System(Locations::from_env()),
            Tuning::for_tool(Tool::Agy),
        )
    }

    /// This program, with discovery skipped (the tests, and `exec:` of the test hook).
    #[must_use]
    pub fn with_program(path: PathBuf) -> Arc<Self> {
        Self::custom(Source::Program(path), Tuning::for_tool(Tool::Agy))
    }

    /// Any source and tuning (the tests shorten the limits and name a double's scenario).
    #[doc(hidden)]
    #[must_use]
    pub fn custom(source: Source, tuning: Tuning) -> Arc<Self> {
        Arc::new(Self {
            core: Core::new(Tool::Agy, source, tuning),
        })
    }

    /// Starts the first probe (`StartProbe::Now`): up to the version. The sign-in check, which makes the
    /// program contact its service, runs only when asked for (`docs/22` D7).
    pub(crate) fn start(&self) {
        self.core.refresh(Level::Light);
    }

    /// The request of `docs/22` section 8.2.
    fn run_request(
        &self,
        program: &Path,
        cancel: CancelToken,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, Fail> {
        let jpeg = jpeg::encode(picture).map_err(|e| Fail::new(FailureKind::Other, e.to_string()))?;
        let io_failure = |what: &str, e: std::io::Error| {
            Fail::new(FailureKind::Other, message(&format!("could not {what}: {e}")))
        };
        let workdir = Workdir::new().map_err(|e| io_failure("make a temporary folder", e))?;
        std::fs::write(workdir.cwd().join("photo.jpg"), &jpeg)
            .map_err(|e| io_failure("write the picture", e))?;
        let schema_file = workdir.input().join("schema.json");
        std::fs::write(&schema_file, schema::schema_text())
            .map_err(|e| io_failure("write the schema", e))?;

        let tuning = self.core.tuning();
        let spec = RunSpec::new(program, workdir.cwd(), EnvProfile::Agy)
            .args(arguments(
                &prompt(instructions),
                &schema_file,
                &self.core.effort(),
                self.core.model_named().as_deref(),
            ))
            .timeout(tuning.request_timeout)
            .cancel(cancel)
            .extra_env(tuning.extra_env.clone());
        let output = run(spec);
        interpret(&output, tuning.agy_own_limit, self.version().as_deref())
    }

    /// The version the program printed, for the message of a usage error.
    fn version(&self) -> Option<String> {
        self.core.details().tool_version
    }
}

/// The prompt (`docs/22` section 8.2): the first line is fixed so that the person's own text never begins
/// with `-` or `/`; the answer form follows the instructions.
fn prompt(instructions: &str) -> String {
    format!(
        "You will grade one photograph. Follow these instructions.\n\n{instructions}\n\n\
         The photograph is @photo.jpg in the current folder. Look only at that file. Do not open, \
         read, write or run anything else, and use no tool but the one that shows you the \
         photograph. Grade this photo. Answer only with the JSON object of the answer form below, \
         as your final structured answer.\n\nAnswer form (JSON Schema):\n{}",
        schema::schema_text()
    )
}

/// The argument list of `docs/22` section 8.2, one by one. `--model` only when a developer named one.
fn arguments(prompt: &str, schema_file: &Path, effort: &str, model: Option<&str>) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "-p".into(),
        prompt.into(),
        "--output-format".into(),
        "json".into(),
        "--json-schema".into(),
        schema_file.as_os_str().to_owned(),
        "--effort".into(),
        effort.into(),
        "--print-timeout".into(),
        PRINT_TIMEOUT.into(),
        "--disable-slash-commands".into(),
        "--sandbox".into(),
    ];
    if let Some(model) = model {
        args.push("--model".into());
        args.push(model.into());
    }
    args
}

/// Reads what the program did (`docs/22` section 8.3): the exit code first, then the envelope.
fn interpret(
    output: &RunOutput,
    own_limit: Duration,
    version: Option<&str>,
) -> Result<AdvisorAnswer, Fail> {
    match &output.outcome {
        Outcome::SpawnFailed(why) => {
            return Err(Fail::new(
                FailureKind::CliFailed,
                format!("Antigravity could not be started: {why}"),
            ));
        }
        Outcome::TimedOut => return Err(Fail::new(FailureKind::Timeout, "")),
        Outcome::Cancelled => return Err(Fail::Cancelled),
        Outcome::OutputTooLarge => {
            return Err(Fail::new(
                FailureKind::CliFailed,
                "Antigravity produced too much output",
            ));
        }
        Outcome::Exited => {}
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let signed_out = || stderr.to_lowercase().contains("authentication required");
    match output.exit_code {
        Some(0) => {}
        Some(2) => {
            let v = version.map_or(String::new(), |v| format!(" (version {v})"));
            return Err(Fail::new(
                FailureKind::CliFailed,
                format!("Antigravity didn't accept Yellowmoon's options{v}. Update Yellowmoon."),
            ));
        }
        Some(1) if signed_out() => return Err(Fail::new(FailureKind::NotSignedIn, "")),
        Some(3) => return Err(exit_three(&stderr)),
        code => {
            let code = code.map_or_else(|| "none".to_owned(), |c| c.to_string());
            return Err(Fail::new(
                FailureKind::CliFailed,
                format!("exit code {code}: {}", output.stderr_message()),
            ));
        }
    }

    // Exit 0: the envelope is the last non-empty line that parses as a JSON object; banner lines come
    // before it.
    let envelope = output
        .stdout
        .split(|b| *b == b'\n')
        .rev()
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .find_map(|line| serde_json::from_slice::<Value>(line).ok().filter(Value::is_object));
    let Some(envelope) = envelope else {
        return Err(Fail::new(
            FailureKind::CliFailed,
            "Antigravity printed no answer",
        ));
    };
    match envelope.get("status").and_then(Value::as_str) {
        Some("SUCCESS") => {}
        Some("ERROR" | "CANCELED" | "INTERRUPTED" | "INVALID" | "WAITING" | "RUNNING") => {
            // Past our own clock's 105 s the program's `--print-timeout` fired: it is a time-out.
            if output.elapsed >= own_limit {
                return Err(Fail::new(FailureKind::Timeout, ""));
            }
            let error = envelope.get("error").and_then(Value::as_str).unwrap_or("");
            return Err(Fail::new(
                FailureKind::CliFailed,
                if error.trim().is_empty() {
                    "Antigravity reported an error".to_owned()
                } else {
                    message(error)
                },
            ));
        }
        _ => {
            return Err(Fail::new(
                FailureKind::CliFailed,
                "Antigravity's answer has no status",
            ));
        }
    }
    match envelope.get("structured_output").filter(|v| !v.is_null()) {
        Some(structured) => {
            let parsed =
                schema::parse_answer(structured).map_err(|e| Fail::new(FailureKind::BadAnswer, e))?;
            AdvisorAnswer {
                steps: parsed.steps,
                scene: parsed.scene,
                plan: parsed.plan,
                // The envelope names no model: the session shows the slug that was passed, or `default`.
                served_model: None,
            }
            .checked()
            .map_err(|e| Fail::new(e.kind, e.message))
        }
        None => Err(Fail::new(
            FailureKind::BadAnswer,
            "Antigravity answered in words instead of the answer form",
        )),
    }
}

/// Exit 3: a turn that ended on an agent or model API error, with an `AGY_ERROR:` line on standard
/// error. Its field names are unverified, so the line is searched as text, not by key.
fn exit_three(stderr: &str) -> Fail {
    let Some(line) = stderr.lines().rev().find(|l| l.contains("AGY_ERROR:")) else {
        return Fail::new(
            FailureKind::CliFailed,
            format!("exit code 3: {}", crate::process::tail_message(stderr)),
        );
    };
    let upper = line.to_uppercase();
    let has_word = |word: &str| {
        upper
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .any(|w| w == word)
    };
    if upper.contains("RESOURCE_EXHAUSTED") || has_word("429") || upper.contains("RATE LIMIT") {
        Fail::new(FailureKind::RateLimited, "")
    } else if upper.contains("UNAUTHENTICATED")
        || has_word("401")
        || has_word("403")
        || upper.contains("AUTHENTICATION")
    {
        Fail::new(FailureKind::NotSignedIn, "")
    } else {
        Fail::new(FailureKind::CliFailed, message(line))
    }
}

impl AutoAdvisor for AgyAdvisor {
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
        1
    }

    fn refresh(&self) {
        self.core.refresh(Level::Full);
    }

    fn settle(&self, timeout: Duration) {
        self.core.settle(timeout);
    }

    fn cancel(&self) {
        self.core.cancel();
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

    fn exited(stdout: &str, code: i32, stderr: &str, elapsed: Duration) -> RunOutput {
        RunOutput {
            outcome: Outcome::Exited,
            exit_code: Some(code),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
            elapsed,
        }
    }

    fn run_of(stdout: &str, code: i32, stderr: &str) -> Result<AdvisorAnswer, Fail> {
        interpret(
            &exited(stdout, code, stderr, Duration::from_secs(5)),
            Duration::from_secs(105),
            Some("1.4.7"),
        )
    }

    fn kind(result: Result<AdvisorAnswer, Fail>) -> FailureKind {
        match result {
            Err(Fail::Failed(kind, _)) => kind,
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    fn envelope(extra: Value) -> String {
        let mut base = json!({"conversation_id": "c", "status": "SUCCESS", "response": ""});
        for (k, v) in extra.as_object().unwrap() {
            base[k] = v.clone();
        }
        format!("agy 1.4.7\n{base}\n")
    }

    #[test]
    fn the_prompt_names_the_photo_and_starts_with_a_fixed_line() {
        let p = prompt("-/ Make it moody.");
        assert!(p.starts_with("You will grade one photograph. Follow these instructions.\n\n-/ Make it moody."));
        assert!(p.contains("@photo.jpg"));
        assert!(p.contains(schema::schema_text()));
        assert!(p.len() < 8 * 1024);
    }

    #[test]
    fn the_arguments_are_the_specs_and_model_only_when_named() {
        let schema_file = Path::new("/t/in/schema.json");
        let text = |args: Vec<OsString>| -> Vec<String> {
            args.iter().map(|a| a.to_string_lossy().into_owned()).collect()
        };
        let plain = text(arguments("PROMPT", schema_file, "low", None));
        assert_eq!(
            plain,
            [
                "-p", "PROMPT", "--output-format", "json", "--json-schema", "/t/in/schema.json",
                "--effort", "low", "--print-timeout", "110s", "--disable-slash-commands", "--sandbox"
            ]
        );
        let named = text(arguments("PROMPT", schema_file, "low", Some("gemini-x")));
        assert_eq!(&named[named.len() - 2..], ["--model", "gemini-x"]);
        assert!(!plain.iter().any(|a| a.contains("dangerously") || a == "--continue"));
    }

    #[test]
    fn a_good_envelope_maps_to_the_steps_after_a_banner() {
        let out = envelope(json!({"structured_output": {"scene": "s", "plan": "p", "contrast": "less"}}));
        let answer = run_of(&out, 0, "").unwrap();
        assert_eq!(answer.steps.contrast, Some(-1));
        assert_eq!(answer.served_model, None);
    }

    #[test]
    fn the_exit_codes_map_as_the_table_says() {
        assert_eq!(kind(run_of("", 2, "flag provided but not defined")), FailureKind::CliFailed);
        assert_eq!(kind(run_of("", 1, "Error: authentication required. Run 'agy' to log in.")), FailureKind::NotSignedIn);
        assert_eq!(kind(run_of("", 1, "something else")), FailureKind::CliFailed);
        assert_eq!(
            kind(run_of("", 3, "AGY_ERROR: {\"status\":\"RESOURCE_EXHAUSTED\",\"code\":429}")),
            FailureKind::RateLimited
        );
        assert_eq!(
            kind(run_of("", 3, "AGY_ERROR: {\"status\":\"UNAUTHENTICATED\",\"code\":401}")),
            FailureKind::NotSignedIn
        );
        assert_eq!(kind(run_of("", 3, "AGY_ERROR: {\"status\":\"INTERNAL\",\"code\":500}")), FailureKind::CliFailed);
        assert_eq!(kind(run_of("", 3, "no line")), FailureKind::CliFailed);
        assert_eq!(kind(run_of("", 7, "weird")), FailureKind::CliFailed);
    }

    #[test]
    fn the_statuses_map_as_the_table_says() {
        assert_eq!(kind(run_of("not json at all\n", 0, "")), FailureKind::CliFailed);
        assert_eq!(kind(run_of(&envelope(json!({})), 0, "")), FailureKind::BadAnswer);
        assert_eq!(kind(run_of(&envelope(json!({"status": "ERROR", "error": "boom"})), 0, "")), FailureKind::CliFailed);
        let bad = envelope(json!({"structured_output": {"scene": "s", "plan": "p", "exposure": "muchBrightest"}}));
        assert_eq!(kind(run_of(&bad, 0, "")), FailureKind::BadAnswer);
        // INTERRUPTED after our own 105 s is the program's own time limit.
        let slow = exited(&envelope(json!({"status": "INTERRUPTED"})), 0, "", Duration::from_secs(111));
        assert_eq!(kind(interpret(&slow, Duration::from_secs(105), None)), FailureKind::Timeout);
        assert_eq!(kind(run_of(&envelope(json!({"status": "INTERRUPTED"})), 0, "")), FailureKind::CliFailed);
        assert_eq!(kind(run_of(&envelope(json!({"status": "WHATEVER"})), 0, "")), FailureKind::CliFailed);
    }

    #[test]
    fn the_other_outcomes() {
        let mut out = exited("", 0, "", Duration::ZERO);
        for (outcome, want) in [
            (Outcome::TimedOut, FailureKind::Timeout),
            (Outcome::OutputTooLarge, FailureKind::CliFailed),
            (Outcome::SpawnFailed("no such file".into()), FailureKind::CliFailed),
        ] {
            out.outcome = outcome;
            assert_eq!(kind(interpret(&out, Duration::from_secs(105), None)), want);
        }
        out.outcome = Outcome::Cancelled;
        assert_eq!(interpret(&out, Duration::from_secs(105), None), Err(Fail::Cancelled));
    }
}
