//! `ym lenses …`: the lenses `ym render --lens` can give a photo (`docs/21` section 8,
//! `docs/10-cli.md`).
//!
//! `list` reads `data/lenses` with the library the app and `ym render` use
//! ([`ym_lens::LensLibrary`]), so the lenses, their lines and their starting amounts are the ones
//! `Core::lens_list` answers to the app.

use std::io::{self, Write};

use serde_json::json;
use ym_core::data_dir::find_data_dir;
use ym_lens::{LensInfo, LensLibrary};

use crate::cli::{LensesCommand, LensesListArgs};
use crate::console::Console;
use crate::exit::{self, Failure};
use crate::looks_cmd::table;
use crate::text::{json_line, line_text};

/// Runs a `ym lenses` command and returns the exit code.
pub fn run(command: &LensesCommand, console: &mut Console) -> u8 {
    let result = match command {
        LensesCommand::List(args) => list(args),
    };
    match result {
        Ok(code) => code,
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// The lens library of the data folder, or the failure that says what to do.
///
/// # Errors
///
/// A usage failure (exit code 2, as for a data folder that is missing) when `data/lenses` is
/// missing or damaged.
pub fn open_library() -> Result<LensLibrary, Failure> {
    let data_root = find_data_dir().map_err(|e| Failure::usage(e.to_string()))?;
    LensLibrary::load(&data_root).map_err(|e| {
        Failure::usage(format!(
            "The lens data could not be loaded ({e}). Reinstall Yellowmoon, or set YELLOWMOON_DATA to a data folder that has them."
        ))
    })
}

/// A number of the starting amounts as text: `40`, or `37.5`.
fn amount_text(v: f64) -> String {
    v.to_string()
}

/// The text of `ym lenses list`: one row per lens.
fn list_text(lenses: &[LensInfo]) -> String {
    let mut rows = vec![vec![
        "ID".to_owned(),
        "NAME".to_owned(),
        "FLARES".to_owned(),
        "EDGES".to_owned(),
        "DESIGN".to_owned(),
        "LINE".to_owned(),
    ]];
    for l in lenses {
        rows.push(vec![
            l.kind.as_str().to_owned(),
            line_text(&l.name),
            amount_text(l.start_flares),
            amount_text(l.start_edges),
            line_text(&l.design_source),
            line_text(&l.line),
        ]);
    }
    table(&rows)
}

fn list(args: &LensesListArgs) -> Result<u8, Failure> {
    let library = open_library()?;
    let lenses = library.lenses();
    let mut out = io::stdout().lock();
    let written = if args.json {
        let value = json!({ "lenses": lenses });
        writeln!(out, "{}", json_line(&value))
    } else {
        write!(out, "{}", list_text(lenses))
    };
    match written.and_then(|()| out.flush()) {
        Ok(()) => Ok(exit::OK),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(exit::OK),
        Err(e) => Err(Failure::failed(format!("Couldn't write the list ({e})."))),
    }
}
