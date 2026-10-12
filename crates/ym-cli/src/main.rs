//! The `yellowmoon` command-line tool: a second front end on the same core as the app
//! (`docs/10-cli.md`).

mod cli;
mod exit;
mod info;

use std::process::ExitCode;

use clap::Parser;

use crate::cli::{Cli, Command, LooksCommand};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => exit::code(code),
        Err(error) => {
            eprintln!("Error: {error:#}");
            exit::code(exit::SOME_FAILED)
        }
    }
}

/// Runs the chosen command and returns its exit code.
fn run(cli: Cli) -> anyhow::Result<u8> {
    match cli.command {
        Command::Info(args) => Ok(info::run(&args)),
        Command::Render(_)
        | Command::Looks(
            LooksCommand::List
            | LooksCommand::Import(_)
            | LooksCommand::Verify
            | LooksCommand::Default(_)
            | LooksCommand::Bake(_),
        )
        | Command::Doctor => Ok(not_implemented()),
    }
}

/// Reports a command whose feature has not landed yet.
fn not_implemented() -> u8 {
    eprintln!("Not implemented yet.");
    exit::USAGE
}
