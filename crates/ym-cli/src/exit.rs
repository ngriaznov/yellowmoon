//! Process exit codes, as documented in `docs/10-cli.md` section Exit codes.

#![allow(
    dead_code,
    reason = "the whole code table lives here; commands start using each code as they are implemented"
)]

use std::process::ExitCode;

/// Every file rendered.
pub const OK: u8 = 0;
/// Some files failed (others succeeded).
pub const SOME_FAILED: u8 = 1;
/// Usage error. Also the code for a missing data folder and for subcommands that are not
/// implemented yet.
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
