//! Plain-language explanations for the wizard's Recognise step and the CLI (`docs/08-looks.md` rule 5).
//!
//! Short sentences that say what is wrong and what to do next, British spelling, no jargon.

use crate::library::LookFormat;

use super::{Ignored, RecognizedFile, SourceInfo, SourceKind};

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

/// Why a source with no `.cube` files and no look files cannot be imported, and what to choose.
pub(super) fn nothing_to_import(source: &SourceInfo, ignored: &Ignored) -> String {
    let noun = noun(source);
    if ignored.xml > 0 && source.kind == SourceKind::File {
        // One loose file: it is the ALF-1 file, it does not "have" any.
        return "This file is one of ARRI's older look files (ALF-1, for ALEXA Classic and XT \
                cameras). Yellowmoon cannot read it. It reads .cube, .alf4, .alf4c and .aml files."
            .to_string();
    }
    if ignored.xml > 0 {
        return format!(
            "This {noun} has .xml files. Yellowmoon cannot read ARRI's older look files \
             (ALF-1, for ALEXA Classic and XT cameras). It reads .cube, .alf4, .alf4c and .aml \
             files."
        );
    }
    if ignored.pdf > 0 {
        return format!(
            "This {noun} has documents (PDF) but no look files or LUTs. \
             Choose the one that holds the LUTs."
        );
    }
    format!(
        "This {noun} has no .cube, .alf4, .alf4c or .aml files, so there is nothing to import. \
         Choose a look file or a LUT, or a folder or zip that holds them."
    )
}

/// What the files are, for a sentence about all of them: ".cube files", ".alf4 files" or
/// "look files" for a mix.
fn files_word(files: &[RecognizedFile]) -> String {
    let mut formats: Vec<LookFormat> = Vec::new();
    for f in files {
        if !formats.contains(&f.format) {
            formats.push(f.format);
        }
    }
    match formats.as_slice() {
        [one] => format!("{} files", one.noun()),
        _ => "look files".to_string(),
    }
}

/// Every file was rejected. A single file that was chosen by itself gets just its own reason.
pub(super) fn all_rejected(source: &SourceInfo, files: &[RecognizedFile], example: &str) -> String {
    if source.kind == SourceKind::File {
        return example.to_string();
    }
    let noun = noun(source);
    let what = files_word(files);
    if files.len() == 1 {
        let single = what.strip_suffix('s').unwrap_or(&what);
        format!("The one {single} in this {noun} cannot be used. {example}")
    } else {
        format!(
            "None of the {} {what} in this {noun} can be used. For example: {example}",
            files.len()
        )
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

/// `.alf4c` files carry display transforms of their own; Yellowmoon uses its own.
pub(super) fn display_transforms_ignored(names: &[&str]) -> String {
    match names {
        [one] => format!(
            "{one}: its own display transforms are not used; Yellowmoon's display rendering is."
        ),
        many => format!(
            "{} .alf4c files carry display transforms of their own. They are not used; \
             Yellowmoon's display rendering is.",
            many.len()
        ),
    }
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

    fn folder() -> SourceInfo {
        SourceInfo {
            kind: SourceKind::Folder,
            name: "f".into(),
        }
    }

    fn recognized(name: &str, format: LookFormat) -> RecognizedFile {
        RecognizedFile {
            rel_path: name.into(),
            name: name.into(),
            size: 1,
            sha256: String::new(),
            lut_size: None,
            format,
            outcome: super::super::Outcome::Rejected { reason: "x".into() },
        }
    }

    #[test]
    fn empty_sources_get_their_own_words() {
        let none = nothing_to_import(&zip(), &Ignored::default());
        assert_eq!(
            none,
            "This zip has no .cube, .alf4, .alf4c or .aml files, so there is nothing to import. \
             Choose a look file or a LUT, or a folder or zip that holds them."
        );
        assert!(nothing_to_import(&folder(), &Ignored::default()).starts_with("This folder"));
        let pdf = Ignored {
            pdf: 2,
            ..Ignored::default()
        };
        assert!(nothing_to_import(&zip(), &pdf).contains("PDF"));
    }

    #[test]
    fn arris_older_look_files_are_named_and_not_read() {
        let xml = Ignored {
            xml: 3,
            ..Ignored::default()
        };
        let text = nothing_to_import(&zip(), &xml);
        assert!(
            text.contains("ALF-1") && text.contains("ALEXA Classic"),
            "{text}"
        );
        assert!(text.contains(".cube, .alf4, .alf4c and .aml"), "{text}");
        assert!(text.starts_with("This zip has .xml files."), "{text}");
    }

    #[test]
    fn one_loose_older_look_file_is_not_said_to_have_xml_files() {
        let one = SourceInfo {
            kind: SourceKind::File,
            name: "Look.xml".into(),
        };
        let xml = Ignored {
            xml: 1,
            ..Ignored::default()
        };
        let text = nothing_to_import(&one, &xml);
        assert!(
            text.starts_with("This file is one of ARRI's older look files (ALF-1"),
            "{text}"
        );
        assert!(!text.contains("has .xml files"), "{text}");
        assert!(
            text.contains("It reads .cube, .alf4, .alf4c and .aml files."),
            "{text}"
        );
    }

    #[test]
    fn rejections_name_what_the_files_are() {
        let alf = [recognized("a.alf4", LookFormat::Alf4)];
        assert_eq!(
            all_rejected(&zip(), &alf, "a.alf4 cannot be used. Bad."),
            "The one .alf4 file in this zip cannot be used. a.alf4 cannot be used. Bad."
        );
        // One file chosen by itself is not "in this file": it just says why it cannot be used.
        let file = SourceInfo {
            kind: SourceKind::File,
            name: "a.alf4".into(),
        };
        assert_eq!(
            all_rejected(&file, &alf, "a.alf4 cannot be used. Bad."),
            "a.alf4 cannot be used. Bad."
        );
        let cubes = [
            recognized("a.cube", LookFormat::Cube),
            recognized("b.cube", LookFormat::Cube),
        ];
        assert!(
            all_rejected(&folder(), &cubes, "e")
                .starts_with("None of the 2 .cube files in this folder")
        );
        let mixed = [
            recognized("a.cube", LookFormat::Cube),
            recognized("b.aml", LookFormat::Aml),
        ];
        assert!(
            all_rejected(&zip(), &mixed, "e").starts_with("None of the 2 look files in this zip")
        );
    }

    #[test]
    fn alf4c_display_transforms_are_said_to_be_set_aside() {
        assert_eq!(
            display_transforms_ignored(&["Mine.alf4c"]),
            "Mine.alf4c: its own display transforms are not used; Yellowmoon's display rendering is."
        );
        let text = display_transforms_ignored(&["a.alf4c", "b.alf4c"]);
        assert!(
            text.starts_with("2 .alf4c files carry display transforms of their own."),
            "{text}"
        );
        assert!(
            text.ends_with("Yellowmoon's display rendering is."),
            "{text}"
        );
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
