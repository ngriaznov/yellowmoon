//! The variables of a parent Claude Code session are removed from a child's environment (`docs/22`
//! section 5). One test in this file, because it sets environment variables of this process.

mod common;

use std::ffi::OsString;

use common::FAKE_CLAUDE;
use tempfile::TempDir;
use ym_agents::process::{EnvProfile, RunSpec, run};

fn env_of(profile: EnvProfile) -> String {
    let dir = TempDir::new().unwrap();
    let spec = RunSpec::new(FAKE_CLAUDE, std::env::temp_dir(), profile)
        .args(["--ym-fake-raw"])
        .extra_env(vec![
            (
                OsString::from("YM_FAKE_RECORD"),
                dir.path().as_os_str().to_owned(),
            ),
            (
                OsString::from("YM_FAKE_SCENARIO"),
                OsString::from("signed_in"),
            ),
        ]);
    assert!(run(spec).success());
    std::fs::read_to_string(dir.path().join("env.txt")).unwrap()
}

#[test]
fn a_parent_sessions_variables_do_not_reach_the_child() {
    // SAFETY: this is the only test of this binary; no other thread reads the environment.
    unsafe {
        for name in [
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_CODE_CHILD_SESSION",
            "CLAUDE_CODE_REMOTE_SESSION_ID",
            "CLAUDE_CODE_REMOTE",
        ] {
            std::env::set_var(name, "1");
        }
        std::env::set_var("CLAUDE_CODE_USE_BEDROCK", "1");
    }

    let env = env_of(EnvProfile::Claude);
    for name in [
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_CHILD_SESSION",
        "CLAUDE_CODE_REMOTE*",
    ] {
        assert!(
            env.lines().any(|l| l == format!("{name}=unset")),
            "{name} reached the child:\n{env}"
        );
    }
    assert!(
        env.lines()
            .any(|l| l == "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"),
        "{env}"
    );

    // `agy` and the shell lookup change nothing but NO_COLOR for agy: the person's own settings stay.
    let env = env_of(EnvProfile::Inherit);
    assert!(env.lines().any(|l| l == "CLAUDECODE=1"), "{env}");
    assert!(env.lines().any(|l| l == "NO_COLOR=unset") || std::env::var_os("NO_COLOR").is_some());
}
