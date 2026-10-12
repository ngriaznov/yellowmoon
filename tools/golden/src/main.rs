//! Golden-image runner (`docs/13-testing.md`, `docs/10-cli.md` section Tests).
//!
//! Will call `yellowmoon render --json` on the golden set and compare the results with stored
//! references by CIEDE2000. Not implemented yet: every run prints a notice and exits with code 2.

use std::process::ExitCode;

use clap::Parser;

/// Renders the golden set through the CLI and compares it with the stored references.
#[derive(Debug, Parser)]
#[command(name = "golden", version)]
struct Args {}

fn main() -> anyhow::Result<ExitCode> {
    let Args {} = Args::parse();
    eprintln!("Not implemented yet.");
    Ok(ExitCode::from(2))
}
