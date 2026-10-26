//! The test doubles' own checks (`docs/22` section 15.2): what they print for the probes, and that a
//! request the real program would not take exits 99.

mod common;

use std::io::Write;
use std::process::{Command, Stdio};

use common::{FAKE_AGY, FAKE_CLAUDE, dark_flat};
use tempfile::TempDir;

fn run(program: &str, args: &[&str], scenario: &str, stdin: &[u8]) -> std::process::Output {
    let dir = TempDir::new().unwrap();
    let mut child = Command::new(program)
        .args(args)
        .current_dir(dir.path())
        .env("YM_FAKE_SCENARIO", scenario)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let bytes = stdin.to_vec();
    std::thread::spawn(move || {
        let _ = input.write_all(&bytes);
    });
    child.wait_with_output().unwrap()
}

#[test]
fn the_probes_print_what_the_real_programs_print() {
    let out = run(FAKE_CLAUDE, &["--version"], "signed_in", b"");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "2.1.310 (Claude Code)"
    );
    let out = run(FAKE_CLAUDE, &["auth", "status"], "signed_in", b"");
    assert!(out.status.success());
    let status: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(status["loggedIn"], true);
    assert_eq!(status["apiProvider"], "firstParty");
    let out = run(FAKE_CLAUDE, &["auth", "status"], "signed_out", b"");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"loggedIn\":false"));

    let out = run(FAKE_AGY, &["--version"], "signed_in", b"");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "1.4.7");
    let out = run(
        FAKE_AGY,
        &["-p", "/usage", "--output-format", "json"],
        "signed_in",
        b"",
    );
    assert!(out.status.success());
    let out = run(
        FAKE_AGY,
        &["-p", "/usage", "--output-format", "json"],
        "signed_out",
        b"",
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("authentication required"));
}

#[test]
fn a_bad_claude_argument_list_exits_99() {
    for args in [
        vec!["-p"],
        vec!["-p", "--input-format", "stream-json"],
        // Everything but `--tools ""`: a non-empty tool list is not the request.
        vec![
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--safe-mode",
            "--tools",
            "Read",
            "--no-session-persistence",
            "--max-turns",
            "3",
            "--model",
            "sonnet",
            "--effort",
            "low",
            "--disable-slash-commands",
            "--strict-mcp-config",
        ],
        vec!["--surprise"],
    ] {
        let out = run(FAKE_CLAUDE, &args, "signed_in", b"{}\n");
        assert_eq!(
            out.status.code(),
            Some(99),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!out.stderr.is_empty());
    }
}

#[test]
fn a_bad_agy_argument_list_exits_99() {
    // No photograph in the working folder, no sandbox, no schema: none of these is the request.
    for args in [
        vec!["-p", "grade @photo.jpg", "--output-format", "json"],
        vec!["-p", "grade", "--output-format", "json", "--sandbox"],
        vec!["--surprise"],
    ] {
        let out = run(FAKE_AGY, &args, "signed_in", b"");
        assert_eq!(out.status.code(), Some(99), "{args:?}");
    }
}

#[test]
fn a_claude_request_without_an_image_exits_99() {
    let dir = TempDir::new().unwrap();
    let schema = ym_agents::schema::schema_text();
    let instructions = dir.path().join("instructions.txt");
    std::fs::write(&instructions, "Plain.").unwrap();
    let work = TempDir::new().unwrap();
    let args = [
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
        "sonnet",
        "--effort",
        "low",
        "--system-prompt-file",
        instructions.to_str().unwrap(),
        "--json-schema",
        schema,
        "--disable-slash-commands",
        "--strict-mcp-config",
    ];
    let line =
        br#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#;
    let mut child = Command::new(FAKE_CLAUDE)
        .args(args)
        .current_dir(work.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&[line.as_slice(), b"\n"].concat())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(99));
    let _ = dark_flat();
}
