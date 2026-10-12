//! What the tool says on stderr: warnings, the summary, verbose notes and the progress line.
//!
//! stdout carries data only (the paths of the files written, or JSON objects with `--json`).
//! Everything meant for a person goes to stderr, so `ym render … | xargs …` and `> results.jsonl`
//! keep working. `--quiet` leaves errors only; `--verbose` adds stages, timings and notes.
//!
//! The progress line is drawn only when stderr is a terminal and neither `--json` nor `--quiet` was
//! given. Any other line printed meanwhile clears it first and draws it again afterwards.

use std::io::{self, IsTerminal, Write};

/// Where messages for a person go.
#[derive(Debug)]
pub struct Console {
    quiet: bool,
    verbose: bool,
    /// The progress line is wanted and stderr is a terminal.
    progress: bool,
    /// What the progress line says now, when it is on screen.
    line: Option<String>,
}

impl Console {
    /// A console for a run. `json` turns the progress line off (stdout is data then).
    pub fn new(quiet: bool, verbose: bool, json: bool) -> Self {
        Self {
            quiet,
            verbose,
            progress: !quiet && !json && io::stderr().is_terminal(),
            line: None,
        }
    }

    /// Whether `--verbose` was given.
    pub fn is_verbose(&self) -> bool {
        self.verbose
    }

    fn clear(&mut self) {
        if self.line.is_some() {
            let mut err = io::stderr().lock();
            let _ = write!(err, "\r\x1b[K");
            let _ = err.flush();
        }
    }

    fn redraw(&mut self) {
        if let Some(line) = &self.line {
            let mut err = io::stderr().lock();
            let _ = write!(err, "\r\x1b[K{line}");
            let _ = err.flush();
        }
    }

    fn print(&mut self, text: &str) {
        self.clear();
        let mut err = io::stderr().lock();
        let _ = writeln!(err, "{text}");
        let _ = err.flush();
        drop(err);
        self.redraw();
    }

    /// Something worth saying that is not an error: a summary line. Hidden by `--quiet`.
    pub fn info(&mut self, text: &str) {
        if !self.quiet {
            self.print(text);
        }
    }

    /// A warning. Hidden by `--quiet`.
    pub fn warn(&mut self, text: &str) {
        if !self.quiet {
            self.print(&format!("Warning: {text}"));
        }
    }

    /// An error about one file or one step. Always shown.
    pub fn error(&mut self, text: &str) {
        self.print(&format!("Error: {text}"));
    }

    /// The failure a command ends with: an error, except that an interruption the user asked for
    /// is not one ("Interrupted." and exit code 130, not "Error: Interrupted.").
    pub fn failure(&mut self, failure: &crate::exit::Failure) {
        if failure.code == crate::exit::INTERRUPTED {
            self.info(&failure.message);
        } else {
            self.error(&failure.message);
        }
    }

    /// A detail only `--verbose` wants.
    pub fn detail(&mut self, text: &str) {
        if self.verbose {
            self.print(text);
        }
    }

    /// Draws or updates the progress line (terminals only).
    pub fn progress(&mut self, text: String) {
        if !self.progress {
            return;
        }
        self.line = Some(text);
        self.redraw();
    }

    /// Removes the progress line.
    pub fn finish_progress(&mut self) {
        self.clear();
        self.line = None;
    }
}

/// Writes one line of data to stdout and carries on when nobody is reading it any more.
///
/// `println!` panics when the reader has gone away (`ym looks import … | head -n 1`), which would
/// print a backtrace and exit with 101 for a command that did its work. The files and the library
/// are the product (`docs/10`), so a closed pipe only means the line was not wanted.
pub fn data_line(text: &str) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{text}").and_then(|()| out.flush());
}

/// Shortens `text` to at most `max` characters, ending in an ellipsis when it was cut.
///
/// The progress line is redrawn in place with a carriage return, which only works while it fits
/// on one terminal row: a long file name would wrap it and leave a copy of every update behind.
pub fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// A bar of `width` cells for a fraction 0 to 1: `[#####-----]`.
pub fn bar(fraction: f64, width: usize) -> String {
    let filled = ((fraction.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("[{}{}]", "#".repeat(filled), "-".repeat(width - filled))
}

/// "2 photos", "1 photo".
pub fn photos(n: usize) -> String {
    if n == 1 {
        "1 photo".to_owned()
    } else {
        format!("{n} photos")
    }
}

/// Seconds as "850 ms", "4.2 s", "1 min 12 s".
pub fn duration_text(seconds: f64) -> String {
    if seconds < 1.0 {
        format!("{} ms", (seconds * 1000.0).round() as u64)
    } else if seconds < 60.0 {
        format!("{seconds:.1} s")
    } else {
        let whole = seconds.round() as u64;
        format!("{} min {} s", whole / 60, whole % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bar_fills_in_proportion() {
        assert_eq!(bar(0.0, 10), "[----------]");
        assert_eq!(bar(0.5, 10), "[#####-----]");
        assert_eq!(bar(1.0, 10), "[##########]");
        assert_eq!(bar(7.0, 4), "[####]");
        assert_eq!(bar(-1.0, 4), "[----]");
    }

    #[test]
    fn long_names_are_shortened_to_fit_one_row() {
        assert_eq!(shorten("DSC00001.ARW", 20), "DSC00001.ARW");
        assert_eq!(shorten("abcdefghij", 10), "abcdefghij");
        assert_eq!(shorten("abcdefghijk", 10), "abcdefghi…");
        assert_eq!(
            shorten("ärgerliches-bild-ünd-mehr.ARW", 8).chars().count(),
            8
        );
        assert_eq!(shorten("anything", 0), "…");
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(duration_text(0.85), "850 ms");
        assert_eq!(duration_text(4.24), "4.2 s");
        assert_eq!(duration_text(72.0), "1 min 12 s");
        assert_eq!(photos(1), "1 photo");
        assert_eq!(photos(3), "3 photos");
    }
}
