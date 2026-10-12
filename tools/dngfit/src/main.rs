//! DNG profile fitting tool (`docs/09-export.md` section DNG with the look baked in).
//!
//! Will fit the embedded DNG profile (colour matrices and tone curve) so that raw converters
//! render the baked LinearRaw export like our ACES 2 output. Not implemented yet: every run prints
//! a notice and exits with code 2.

use std::process::ExitCode;

use clap::Parser;

/// Fits the embedded DNG profile for the LinearRaw export.
#[derive(Debug, Parser)]
#[command(name = "dngfit", version)]
struct Args {}

fn main() -> anyhow::Result<ExitCode> {
    let Args {} = Args::parse();
    eprintln!("Not implemented yet.");
    Ok(ExitCode::from(2))
}
