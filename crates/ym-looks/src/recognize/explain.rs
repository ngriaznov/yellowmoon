//! Plain-language explanations for the wizard's Recognise step and the CLI (`docs/08-looks.md` rule 5).
//!
//! Short sentences that say what is wrong and what to do next, British spelling, no jargon.

use super::{Ignored, SourceInfo, SourceKind};

/// "zip", "folder" or "file".
fn noun(source: &SourceInfo) -> &'static str {
    match source.kind {
        SourceKind::Zip => "zip",
        SourceKind::Folder => "folder",
        SourceKind::File => "file",
    }
}

/// "1 file" or "3 files".
#[must_use]
pub fn count(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
    }
}

/// Why a source with no `.cube` files cannot be imported, and where to get the right one.
pub(super) fn no_cube_files(source: &SourceInfo, ignored: &Ignored) -> String {
    let noun = noun(source);
    let alf = match (ignored.alf4 > 0, ignored.alf2 > 0) {
        (true, true) => Some(".alf4 and .alf2"),
        (true, false) => Some(".alf4"),
        (false, true) => Some(".alf2"),
        (false, false) => None,
    };
    if let Some(alf) = alf {
        return format!(
            "This {noun} has ARRI Look Files ({alf}) but no .cube files. \
             Download the 'LUT (.cube)' version from ARRI's page."
        );
    }
    if ignored.pdf > 0 {
        return format!(
            "This {noun} has documents (PDF) but no .cube files. \
             Choose the one that holds the LUTs."
        );
    }
    format!(
        "This {noun} has no .cube files, so there is nothing to import. \
         Choose a .cube file, or a folder or zip that holds .cube files."
    )
}

/// Every `.cube` file was rejected.
pub(super) fn all_rejected(source: &SourceInfo, n: usize, example: &str) -> String {
    let noun = noun(source);
    if n == 1 {
        format!("The one .cube file in this {noun} cannot be used. {example}")
    } else {
        format!("None of the {n} .cube files in this {noun} can be used. For example: {example}")
    }
}

/// One file was rejected.
pub(super) fn rejected(name: &str, reason: &str) -> String {
    format!("{name} cannot be used. {reason}")
}

/// More rejected files than are listed one by one.
pub(super) fn more_rejected(n: usize) -> String {
    format!(
        "{} more could not be used either.",
        count(n, "other file", "other files")
    )
}

/// A package with fewer files than it should have.
pub(super) fn incomplete(
    source: &SourceInfo,
    title: &str,
    found: usize,
    expected: usize,
    missing: &[String],
) -> String {
    let noun = noun(source);
    let mut text = format!(
        "{title} should have {expected} files, but this {noun} has {found}. \
         The download may be incomplete. Download it again from ARRI's page."
    );
    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().take(5).map(String::as_str).collect();
        text.push_str(&format!(" Missing: {}", shown.join(", ")));
        if missing.len() > shown.len() {
            text.push_str(&format!(" and {} more", missing.len() - shown.len()));
        }
        text.push('.');
    }
    text
}

/// A package with more files than expected.
pub(super) fn more_than_expected(title: &str, found: usize, expected: usize) -> String {
    format!("{title} has {found} files, more than the {expected} expected. They are imported too.")
}

/// Files in a fingerprinted package that are not the ones ARRI published.
pub(super) fn unexpected_files(title: &str, n: usize) -> String {
    format!(
        "{} in {title} are not the ones ARRI published. They are imported as they are.",
        count(n, "file", "files")
    )
}

/// Files that need the generic question.
pub(super) fn needs_answer(n: usize, suggestion: Option<&str>) -> String {
    let mut text = format!(
        "{} not from an ARRI package we know. Say what {} as input: LogC4, LogC3, \
         or LogC3 with Rec.709 baked in.",
        if n == 1 {
            "1 file is".to_string()
        } else {
            format!("{n} files are")
        },
        if n == 1 { "it expects" } else { "they expect" }
    );
    if let Some(s) = suggestion {
        text.push_str(&format!(" The names suggest {s}."));
    }
    text
}

/// Files that name ARRI but match none of the five packages.
pub(super) fn unknown_arri(titles: &[String]) -> String {
    format!(
        "These files look like they come from ARRI, but not from a package Yellowmoon knows \
         ({}). They may be for another camera. You can still import them as your own LUTs \
         if you say what they expect as input.",
        titles.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zip() -> SourceInfo {
        SourceInfo {
            kind: SourceKind::Zip,
            name: "x.zip".into(),
        }
    }

    #[test]
    fn alf4_only_message_matches_the_spec_wording() {
        let ignored = Ignored {
            alf4: 87,
            ..Ignored::default()
        };
        assert_eq!(
            no_cube_files(&zip(), &ignored),
            "This zip has ARRI Look Files (.alf4) but no .cube files. \
             Download the 'LUT (.cube)' version from ARRI's page."
        );
    }

    #[test]
    fn other_empty_sources_get_their_own_words() {
        let pdf = Ignored {
            pdf: 2,
            ..Ignored::default()
        };
        assert!(no_cube_files(&zip(), &pdf).contains("PDF"));
        let none = no_cube_files(&zip(), &Ignored::default());
        assert!(none.contains("no .cube files"));
        let folder = SourceInfo {
            kind: SourceKind::Folder,
            name: "f".into(),
        };
        assert!(no_cube_files(&folder, &Ignored::default()).starts_with("This folder"));
    }

    #[test]
    fn incomplete_lists_a_few_missing_names() {
        let missing: Vec<String> = (1..=8).map(|i| format!("{i}.cube")).collect();
        let text = incomplete(&zip(), "Look Library · LogC4", 79, 87, &missing);
        assert!(text.contains("should have 87 files"), "{text}");
        assert!(text.contains("has 79"), "{text}");
        assert!(text.contains("Missing: 1.cube, 2.cube, 3.cube, 4.cube, 5.cube and 3 more."));
    }

    #[test]
    fn counts_pluralise() {
        assert_eq!(count(1, "file", "files"), "1 file");
        assert_eq!(count(2, "file", "files"), "2 files");
        assert!(needs_answer(1, None).starts_with("1 file is"));
        assert!(needs_answer(4, Some("LogC4")).contains("4 files are"));
        assert!(needs_answer(4, Some("LogC4")).contains("suggest LogC4"));
    }
}
