//! The `ym` command-line tool: a second front end on the same core as the app
//! (`docs/10-cli.md`).
//!
//! The commands translate options into `ym-core` calls and print what comes back; the image
//! processing, the estimates and the export code are the app's own, so a file rendered here is the
//! file the app would write.

mod cli;
mod console;
mod develop;
mod doctor;
mod exit;
mod info;
mod inputs;
mod lenses_cmd;
mod looks_cmd;
mod render;
mod report;
mod runtime;
mod text;

use std::process::ExitCode;

use clap::Parser;

use crate::cli::{Cli, Command};
use crate::console::Console;

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.quiet, cli.verbose);
    runtime::install_interrupt_handler();
    exit::code(run(cli))
}

/// The crates of this workspace. The log is about what Yellowmoon does: the graphics stack and the
/// libraries below it have their own opinions about drivers that are not installed, and a missing
/// EGL driver is not an error on a machine that has Vulkan.
const OUR_CRATES: [&str; 12] = [
    "ym_agents",
    "ym_cli",
    "ym_color",
    "ym_core",
    "ym_export",
    "ym_gpu",
    "ym_lens",
    "ym_looks",
    "ym_ml",
    "ym_noise",
    "ym_profiles",
    "ym_raw",
];

/// Logs go to stderr: errors of our crates by default, nothing with `--quiet`, their info messages
/// (stages, timings) with `--verbose`. `RUST_LOG` overrides all of it.
fn init_logging(quiet: bool, verbose: bool) {
    let level = if verbose { "info" } else { "error" };
    let directives = if quiet {
        "off".to_owned()
    } else {
        std::iter::once("off".to_owned())
            .chain(OUR_CRATES.iter().map(|c| format!("{c}={level}")))
            .collect::<Vec<_>>()
            .join(",")
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(directives));
    // A second initialisation (a test harness) is not an error worth stopping for.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_target(false)
        .without_time()
        .try_init();
}

/// Runs the chosen command and returns its exit code.
fn run(cli: Cli) -> u8 {
    let json = match &cli.command {
        Command::Render(args) => args.json,
        Command::Info(args) => args.json,
        _ => false,
    };
    let mut console = Console::new(cli.quiet, cli.verbose, json);
    match cli.command {
        Command::Render(args) => render::run(&args, &mut console),
        Command::Info(args) => info::run(&args, &mut console),
        Command::Looks(command) => looks_cmd::run(&command, &mut console),
        Command::Lenses(command) => lenses_cmd::run(&command, &mut console),
        Command::Doctor(args) => doctor::run(&args, &mut console),
    }
}
