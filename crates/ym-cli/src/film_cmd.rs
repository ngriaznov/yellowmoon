//! `ym film …`: the stocks and papers a Film node can use (`docs/24-film.md` section 4.5, `docs/10-cli.md`).
//!
//! `list` reads `data/film/params/names.json` with the catalogue the core checks a Film node against, so the ids
//! are the ones a `--settings` file may name.

use std::io::{self, Write};

use serde_json::json;
use ym_core::data_dir::find_data_dir;
use ym_core::film::{Catalogue, FILM_CREDIT};
use ym_film::{StockKind, StockName};

use crate::cli::{FilmCommand, FilmListArgs};
use crate::console::Console;
use crate::exit::{self, Failure};
use crate::looks_cmd::table;
use crate::text::{json_line, line_text};

/// Runs a `ym film` command and returns the exit code.
pub fn run(command: &FilmCommand, console: &mut Console) -> u8 {
    let result = match command {
        FilmCommand::List(args) => list(args),
    };
    match result {
        Ok(code) => code,
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// The catalogue of the data folder, or the failure that says what to do.
///
/// # Errors
///
/// A usage failure (exit code 2, as for a data folder that is missing) when `data/film` is missing or damaged.
fn open_catalogue() -> Result<Catalogue, Failure> {
    let data_root = find_data_dir().map_err(|e| Failure::usage(e.to_string()))?;
    Catalogue::load(&data_root).map_err(|_| {
        Failure::usage(
            "Yellowmoon's film data is missing. Reinstall the app, or set YELLOWMOON_DATA to a data folder that has it."
                .to_owned(),
        )
    })
}

/// The kind as the list says it, and as the JSON names it.
fn kind_words(kind: StockKind) -> (&'static str, &'static str) {
    match kind {
        StockKind::ColourNegative => ("colour negative", "colour_negative"),
        StockKind::CineNegative => ("cine negative", "cine_negative"),
        StockKind::Slide => ("slide", "slide"),
        StockKind::Paper => ("paper", "paper"),
        StockKind::PrintFilm => ("print film", "print_film"),
    }
}

/// The rows a Film node can use: the films that are offered, then the papers that are, in the order of `names.json`.
fn offered(catalogue: &Catalogue) -> Vec<&StockName> {
    let films = catalogue.rows().iter().filter(|r| catalogue.is_film(&r.id));
    let papers = catalogue
        .rows()
        .iter()
        .filter(|r| catalogue.is_paper(&r.id));
    films.chain(papers).collect()
}

/// The text of `ym film list`: one row per stock or paper, then the credit line.
fn list_text(catalogue: &Catalogue) -> String {
    let mut rows = vec![vec![
        "ID".to_owned(),
        "NAME".to_owned(),
        "KIND".to_owned(),
        "ISO".to_owned(),
        "MATCHING PAPER".to_owned(),
    ]];
    for r in offered(catalogue) {
        rows.push(vec![
            line_text(&r.id),
            line_text(&r.name),
            kind_words(r.kind).0.to_owned(),
            r.iso.map_or_else(|| "-".to_owned(), |n| n.to_string()),
            catalogue
                .matching_paper(&r.id)
                .map_or_else(|| "-".to_owned(), |p| line_text(&p)),
        ]);
    }
    let mut out = table(&rows);
    out.push('\n');
    out.push_str(FILM_CREDIT);
    out.push('\n');
    out
}

/// The JSON of `ym film list`: `{ "stocks": [...], "papers": [...], "profile_version", "credit" }`.
fn list_value(catalogue: &Catalogue) -> serde_json::Value {
    let rows = offered(catalogue);
    let stock = |r: &&StockName| {
        json!({
            "id": r.id,
            "name": r.name,
            "kind": kind_words(r.kind).1,
            "maker": r.group,
            "iso": r.iso,
            "paper": catalogue.matching_paper(&r.id),
        })
    };
    let paper = |r: &&StockName| json!({ "id": r.id, "name": r.name, "maker": r.group });
    json!({
        "stocks": rows.iter().filter(|r| catalogue.is_film(&r.id)).map(stock).collect::<Vec<_>>(),
        "papers": rows.iter().filter(|r| catalogue.is_paper(&r.id)).map(paper).collect::<Vec<_>>(),
        "profile_version": catalogue.profile_version(),
        "credit": FILM_CREDIT,
    })
}

fn list(args: &FilmListArgs) -> Result<u8, Failure> {
    let catalogue = open_catalogue()?;
    let mut out = io::stdout().lock();
    let written = if args.json {
        writeln!(out, "{}", json_line(&list_value(&catalogue)))
    } else {
        write!(out, "{}", list_text(&catalogue))
    };
    match written.and_then(|()| out.flush()) {
        Ok(()) => Ok(exit::OK),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(exit::OK),
        Err(e) => Err(Failure::failed(format!("Couldn't write the list ({e})."))),
    }
}
