//! Process exit codes, as documented in `docs/10-cli.md` section Exit codes, and the error type
//! that carries one out of a command.

use std::process::ExitCode;

/// Every file rendered.
pub const OK: u8 = 0;
/// Some files failed (others succeeded).
pub const SOME_FAILED: u8 = 1;
/// Usage error. Also the code for a missing data folder.
pub const USAGE: u8 = 2;
/// A requested look is missing.
pub const LOOK_MISSING: u8 = 3;
/// `--require-gpu` was given and there is no usable GPU.
pub const NO_GPU: u8 = 4;
/// No input files were found.
pub const NO_INPUT: u8 = 5;
/// Interrupted with Ctrl-C (the conventional `128 + SIGINT`).
pub const INTERRUPTED: u8 = 130;

/// Converts one of the codes above into a process exit code.
pub fn code(value: u8) -> ExitCode {
    ExitCode::from(value)
}

/// A command that cannot go on: the message for stderr and the exit code to end with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// The exit code.
    pub code: u8,
    /// What went wrong and what to do next, in plain words.
    pub message: String,
}

impl Failure {
    /// A failure with an exit code.
    pub fn new(code: u8, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// A usage error (exit code 2).
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(USAGE, message)
    }

    /// Something failed (exit code 1).
    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(SOME_FAILED, message)
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Failure {}
