//! Recognition rules that identify ARRI packages and the LUT input encoding (`docs/08-looks.md` section Recognition rules).
//!
//! This module is pure: it takes facts about files (name, size, SHA-256, `TITLE`, LUT size) and
//! the package list, and decides what each file is. Reading zips and folders is the importer's
//! job ([`crate::import`]).
//!
//! The rules, in order, for each valid `.cube` file:
//!
//! 1. **Hash.** If its SHA-256 is in a package's fingerprints, that package identifies it exactly,
//!    including its group and name (taken from the fingerprinted path).
//! 2. **Names and headers.** Otherwise the packages' markers are tried against the evidence: the
//!    source's name, the file's path and its `TITLE`. `LogC4` or `LogC3` in the evidence picks the
//!    encoding; `-to-` plus a display name (`Gamma24_Rec709`, `P3`, `Rec2020`, `HLG`, `PQ`) makes
//!    it a display LUT (no group folder) or a Rec.709-baked look (group folder); a four-digit
//!    group number in a folder name makes it a look; `LUT_3D_SIZE` 65 or 33 must agree with the
//!    package. Exactly one package must fit.
//! 3. **The package as a whole.** A file that fits no package by its own names may still belong to
//!    one by the evidence of the whole source: a zip or folder whose own name is the package's
//!    (`ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip`), holding at least half the files the
//!    package should have, all at its LUT size. That is strong enough that the layout (group
//!    folders, one folder per look, a single top-level folder, a flat list) and look names that
//!    happen to hold words like `P3` or `PQ` are only a tie-breaker, never a requirement
//!    (`match_by_package`).
//! 4. **Nothing fits.** The file gets the generic question (what does it expect as input?), with a
//!    suggestion when the names hint at one.
//!
//! **Look files.** ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) carry their own input and LUT size,
//! so they need no question. One whose hash is in a package's fingerprints, or that sits in a zip
//! or folder named for a look-file package (the ALF4 library, the LogC3 look-file library, rule 3
//! above, per format), is that package's look. Any other look file is the person's own look: it is
//! imported as it is, with the input the file says ([`Outcome::OwnLookFile`]).
//!
//! A source with no `.cube` files or look files, or only unusable ones, or ARRI files from a
//! package Yellowmoon does not know, gets a plain explanation instead ([`Issue`]).

use std::collections::{HashMap, HashSet};

use serde::Serialize;
use ym_color::look::LookInput;

use crate::library::LookFormat;

mod explain;
pub mod names;
pub mod packages;

pub use explain::count as plural;
pub use packages::{
    ARRI_LOOK_FILES_URL, Fingerprint, PACKAGES_PATH, Package, PackageDb, PackageDbError,
    PackageInput, PackageRole,
};

use names::{
    file_name, file_stem, fold_alnum, folder_components, group_of, has_display_rendering,
    look_code, look_name, parse_group_folder, split_strength,
};

/// What kind of thing the user chose to import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// One `.cube` file.
    File,
    /// A folder (walked recursively).
    Folder,
    /// A zip archive.
    Zip,
}

/// The thing being imported, for the explanations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceInfo {
    /// File, folder or zip.
    pub kind: SourceKind,
    /// Its file name, which is itself evidence (`ARRI_Look_Library_LogC4.zip`).
    pub name: String,
}

/// What was set aside without being read as a LUT or a look file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct Ignored {
    /// `.xml` files: ARRI's older look files (ALF-1, for ALEXA Classic and XT), which are not read.
    pub xml: usize,
    /// PDF documents.
    pub pdf: usize,
    /// macOS metadata: `__MACOSX` folders, `._` files, `.DS_Store`.
    pub macos_metadata: usize,
    /// Anything else that is not a `.cube` file.
    pub other: usize,
    /// Zip entries whose paths would leave the folder (`../x`, absolute paths) or are links.
    pub unsafe_entries: usize,
}

/// What a valid `.cube` file looked like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubeSummary {
    /// The `TITLE` line, if any.
    pub title: Option<String>,
    /// Points per axis of the 3D table.
    pub lut_size: usize,
    /// Whether a 1D shaper precedes the cube.
    pub has_shaper: bool,
}

/// What a valid ARRI Look File says about itself: no LUT data, only facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookFileFacts {
    /// The encoding the look is made for.
    pub input: LookInput,
    /// An `.alf4c` with display transforms of its own, which are not used.
    pub display_transforms_ignored: bool,
}

/// What the importer learned about one file: no LUT data, only facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFacts {
    /// Path inside the source, `/`-separated.
    pub rel_path: String,
    /// Size in bytes.
    pub size: u64,
    /// SHA-256, lower-case hex.
    pub sha256: String,
    /// What kind of file it is: the container of a look file that read (whatever its extension
    /// says), else what the extension says.
    pub format: LookFormat,
    /// What the file is, or why it was rejected (a sentence for the user). For a look file the
    /// summary has its LUT size and no `TITLE`.
    pub check: Result<CubeSummary, String>,
    /// What a valid ARRI Look File says about itself; `None` for a `.cube` and for a file that was
    /// rejected.
    pub look_file: Option<LookFileFacts>,
}

/// How a file was identified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// Its hash is in the package's fingerprints.
    Fingerprint,
    /// Its names and header fit the package's patterns.
    Pattern,
}

/// What a file is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    /// An ARRI look.
    ArriLook {
        /// The package's id.
        package: String,
        /// Group number (0 when the path has none).
        group_no: u32,
        /// Group name.
        group_name: String,
        /// What it expects as input.
        input: LookInput,
        /// Display name.
        name: String,
        /// The file name the copy in the library gets: the fingerprinted name for a file
        /// identified by hash (whatever the user renamed it to), else the name it came with.
        file_name: String,
        /// How it was identified.
        evidence: Evidence,
    },
    /// An ARRI display LUT. It is recognised so that the source can be explained, and it is never
    /// imported: Yellowmoon uses its own display rendering (`docs/02-colour-pipeline.md`, C12).
    ArriDisplay {
        /// The package's id.
        package: String,
        /// How it was identified.
        evidence: Evidence,
    },
    /// An ARRI Look File that is not part of a known package: the person's own look. The file says
    /// what it expects as input, so nothing is asked.
    OwnLookFile {
        /// What the file says it is made for.
        input: LookInput,
        /// An `.alf4c` with display transforms of its own, which are not used.
        display_transforms_ignored: bool,
    },
    /// Not from a known package: the generic question applies.
    NeedsAnswer {
        /// What the names and header hint at, if anything.
        suggested_input: Option<LookInput>,
        /// Why the file matched no package, when that is worth saying.
        note: Option<String>,
    },
    /// Not a usable LUT.
    Rejected {
        /// A sentence saying why.
        reason: String,
    },
}

/// One file and what it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecognizedFile {
    /// Path inside the source, `/`-separated.
    pub rel_path: String,
    /// File name.
    pub name: String,
    /// Size in bytes.
    pub size: u64,
    /// SHA-256, lower-case hex.
    pub sha256: String,
    /// Points per axis, for valid files.
    pub lut_size: Option<usize>,
    /// `.cube`, or the container of a look file.
    pub format: LookFormat,
    /// What it is.
    pub outcome: Outcome,
}

impl RecognizedFile {
    /// Whether the importer can take the file without being told anything. An ARRI display LUT is
    /// recognised too, but it is not taken: see [`RecognizedFile::is_display_lut`].
    #[must_use]
    pub fn is_recognized(&self) -> bool {
        matches!(
            self.outcome,
            Outcome::ArriLook { .. } | Outcome::OwnLookFile { .. }
        )
    }

    /// Whether the file is one of ARRI's display LUTs, which are left out of every import.
    #[must_use]
    pub fn is_display_lut(&self) -> bool {
        matches!(self.outcome, Outcome::ArriDisplay { .. })
    }

    /// Whether the file is one of the person's own look files (`.alf4`, `.alf4c`, `.aml`).
    #[must_use]
    pub fn is_own_look_file(&self) -> bool {
        matches!(self.outcome, Outcome::OwnLookFile { .. })
    }

    /// Whether the file needs the generic input question answered.
    #[must_use]
    pub fn needs_answer(&self) -> bool {
        matches!(self.outcome, Outcome::NeedsAnswer { .. })
    }
}

/// Whether a package has all the files it should.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageStatus {
    /// Exactly the expected files.
    Complete,
    /// Fewer than expected: the download may be incomplete.
    Incomplete,
    /// More than expected (the package may have grown since the patterns were written).
    MoreThanExpected,
}

/// A package found in the source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageFinding {
    /// The package's id (`look-library-logc4`).
    pub id: String,
    /// The name shown in the wizard.
    pub title: String,
    /// Looks or display LUTs.
    pub role: PackageRole,
    /// Whether the files were identified by hash or by names and headers.
    pub evidence: Evidence,
    /// Files found.
    pub found: usize,
    /// Files expected, when known.
    pub expected: Option<usize>,
    /// Complete or not.
    pub status: PackageStatus,
    /// Paths of fingerprinted files that are not in the source (empty without fingerprints).
    pub missing: Vec<String>,
    /// Files identified by pattern in a fingerprinted package: not the ones ARRI published.
    pub unexpected: usize,
    /// The LUT sizes seen, sorted.
    pub lut_sizes: Vec<usize>,
    /// Distinct look groups seen (looks only).
    pub groups: usize,
}

/// How serious an [`Issue`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Worth knowing; nothing to do.
    Info,
    /// The user needs to answer the generic question.
    Question,
    /// Something is wrong; the message says what to do.
    Problem,
}

/// What an [`Issue`] is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    /// No `.cube` files or look files at all.
    NoLookFiles,
    /// Every file was rejected.
    AllRejected,
    /// One file was rejected.
    Rejected,
    /// Fewer files than the package should have.
    Incomplete,
    /// More files than the package should have.
    MoreThanExpected,
    /// Files of a fingerprinted package that differ from the published ones.
    UnexpectedFiles,
    /// Files that need the generic input question.
    NeedsAnswer,
    /// ARRI files from a package Yellowmoon does not know.
    UnknownArriPackage,
    /// `.alf4c` files whose display transforms are not used.
    DisplayTransformsIgnored,
    /// ARRI display LUTs, which are never imported: Yellowmoon uses its own display rendering. A
    /// problem when the source holds nothing else that can be imported, else a remark.
    DisplayRenderings,
}

/// Something the wizard or CLI should tell the user, in plain words.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Issue {
    /// How serious it is.
    pub severity: Severity,
    /// What it is about.
    pub kind: IssueKind,
    /// The sentence to show.
    pub message: String,
}

/// The overall state of a recognized source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Everything is identified and complete: ready to import.
    Ready,
    /// Some files need the generic question answered first.
    NeedsAnswer,
    /// A package is missing files; importing would give a partial set.
    Incomplete,
    /// Nothing can be imported; the issues say why.
    Blocked,
}

/// The result of recognising a source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Recognition {
    /// What was chosen.
    pub source: SourceInfo,
    /// Every `.cube` file and look file, in path order.
    pub files: Vec<RecognizedFile>,
    /// The packages found.
    pub packages: Vec<PackageFinding>,
    /// What was set aside.
    pub ignored: Ignored,
    /// Explanations, most important first.
    pub issues: Vec<Issue>,
}

impl Recognition {
    /// Files identified as ARRI looks or display LUTs.
    #[must_use]
    pub fn recognized_count(&self) -> usize {
        self.files.iter().filter(|f| f.is_recognized()).count()
    }

    /// Files that need the generic question.
    #[must_use]
    pub fn needs_answer_count(&self) -> usize {
        self.files.iter().filter(|f| f.needs_answer()).count()
    }

    /// ARRI display LUTs, which are recognised and left out of every import.
    #[must_use]
    pub fn display_lut_count(&self) -> usize {
        self.files.iter().filter(|f| f.is_display_lut()).count()
    }

    /// Files that were rejected.
    #[must_use]
    pub fn rejected_count(&self) -> usize {
        self.files.len()
            - self.recognized_count()
            - self.needs_answer_count()
            - self.display_lut_count()
    }

    /// Whether any file needs the generic question answered before import.
    #[must_use]
    pub fn needs_answer(&self) -> bool {
        self.needs_answer_count() > 0
    }

    /// The overall state: see [`Verdict`].
    #[must_use]
    pub fn verdict(&self) -> Verdict {
        if self.recognized_count() + self.needs_answer_count() == 0 {
            Verdict::Blocked
        } else if self
            .packages
            .iter()
            .any(|p| p.status == PackageStatus::Incomplete)
        {
            Verdict::Incomplete
        } else if self.needs_answer() {
            Verdict::NeedsAnswer
        } else {
            Verdict::Ready
        }
    }

    /// The messages that are problems.
    pub fn problems(&self) -> impl Iterator<Item = &Issue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Problem)
    }

    /// The finding for a package id, if that package was found.
    #[must_use]
    pub fn package(&self, id: &str) -> Option<&PackageFinding> {
        self.packages.iter().find(|p| p.id == id)
    }
}

/// At most this many rejected files get their own message.
const MAX_REJECTED_MESSAGES: usize = 12;

/// What the pattern rules decided for one file.
enum Pattern<'a> {
    Match(&'a Package),
    NoMatch {
        suggestion: Option<LookInput>,
        note: Option<String>,
    },
}

/// The look group of a file: the four-digit group folder it sits in (`3000 Film/x.cube`), else the
/// first digit of the look code its name starts with. ARRI's Look Library is a flat list of files
/// called `3110 Film A.cube` or `3250-Cross.cube` (code 3110 is in group 3000, "Film", by
/// [`PackageDb::group_names`]), so the folders say nothing and the file names do.
fn group_for(db: &PackageDb, path: &str) -> Option<(u32, String)> {
    group_of(path).or_else(|| {
        let code = look_code(file_stem(path))?;
        let number = code / 1000 * 1000;
        if number == 0 {
            return None;
        }
        let name = db
            .group_names
            .get(&number)
            .cloned()
            .unwrap_or_else(|| format!("Group {number}"));
        Some((number, name))
    })
}

/// The group of a look file: the first digit of the look code in its name decides
/// (`LL-1120-Black-and-White` is in group 1000), named as the package list names the group; a
/// file with no code takes the group folder it sits in.
fn look_file_group(db: &PackageDb, path: &str) -> Option<(u32, String)> {
    let number = look_code(file_stem(path)).map(|code| code / 1000 * 1000);
    let Some(number) = number.filter(|n| *n != 0) else {
        return group_of(path);
    };
    let name = db
        .group_names
        .get(&number)
        .cloned()
        .or_else(|| group_of(path).filter(|(n, _)| *n == number).map(|(_, n)| n))
        .unwrap_or_else(|| format!("Group {number}"));
    Some((number, name))
}

/// [`group_for`] for a file of `format`: look files are grouped by their code first.
fn group_for_format(db: &PackageDb, path: &str, format: LookFormat) -> Option<(u32, String)> {
    if format.is_look_file() {
        look_file_group(db, path)
    } else {
        group_for(db, path)
    }
}

/// Evidence text for a file: the source's name, its path and its `TITLE`, folded.
fn evidence_text(source: &SourceInfo, file: &FileFacts, summary: &CubeSummary) -> String {
    let title = summary.title.as_deref().unwrap_or("");
    fold_alnum(&format!("{} {} {}", source.name, file.rel_path, title))
}

/// Whether anything about the file names a display rendering (`-to-` plus `Rec709`, `P3`, …): its
/// own name, the folders it sits in, its `TITLE`, or the name of the zip or folder it came in.
///
/// A Rec.709-baked look is display-referred; taking it for a log-to-log look would apply it in the
/// wrong encoding. So the evidence is read as widely as the encoding evidence is, not from the
/// file name alone: a package whose zip is called "... LogC3 to Rec709" may not repeat that in
/// every file name.
fn names_display_rendering(
    db: &PackageDb,
    source: &SourceInfo,
    file: &FileFacts,
    summary: &CubeSummary,
) -> bool {
    let source_name = match source.kind {
        // Drop the extension (".zip", ".cube") so it cannot run into the last word.
        SourceKind::Zip | SourceKind::File => file_stem(&source.name),
        SourceKind::Folder => source.name.as_str(),
    };
    std::iter::once(file_stem(&file.rel_path))
        .chain(folder_components(&file.rel_path))
        .chain(summary.title.as_deref())
        .chain(std::iter::once(source_name))
        .any(|text| has_display_rendering(text, &db.display_names))
}

fn lut_size_fits(package: &Package, size: usize) -> bool {
    package.expected.lut_sizes.is_empty() || package.expected.lut_sizes.contains(&size)
}

fn suggest_input(
    evidence: &str,
    has_group: bool,
    has_display: bool,
    lut_size: usize,
) -> Option<LookInput> {
    let c4 = evidence.contains("logc4");
    let c3 = evidence.contains("logc3");
    match (c4, c3) {
        (true, false) => Some(LookInput::LogC4),
        (false, true) if has_display => Some(LookInput::LogC3ToRec709),
        (false, true) => Some(LookInput::LogC3Ei800),
        (false, false) if has_group => match lut_size {
            65 => Some(LookInput::LogC4),
            33 => Some(LookInput::LogC3Ei800),
            _ => None,
        },
        _ => None,
    }
}

fn match_pattern<'a>(
    db: &'a PackageDb,
    source: &SourceInfo,
    file: &FileFacts,
    summary: &CubeSummary,
) -> Pattern<'a> {
    let evidence = evidence_text(source, file, summary);
    let has_group = group_for(db, &file.rel_path).is_some();
    let has_display = names_display_rendering(db, source, file, summary);
    let fits: Vec<&Package> = db
        .packages
        .iter()
        .filter(|p| {
            p.format.is_cube()
                && !p.markers.encoding.is_empty()
                && p.markers.encoding.matches_folded(&evidence)
                && p.markers.display_rendering.allows(has_display)
                && p.markers.group_folders.allows(has_group)
                && lut_size_fits(p, summary.lut_size)
        })
        .collect();
    match fits.as_slice() {
        [one] => Pattern::Match(one),
        [] => Pattern::NoMatch {
            suggestion: suggest_input(&evidence, has_group, has_display, summary.lut_size),
            note: None,
        },
        several => Pattern::NoMatch {
            suggestion: None,
            note: Some(format!(
                "It could be from {} packages ({}), so it is not assumed to be any of them.",
                several.len(),
                several
                    .iter()
                    .map(|p| p.title.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        },
    }
}

fn outcome_for(db: &PackageDb, package: &Package, path: &str, evidence: Evidence) -> Outcome {
    outcome_in_group(
        db,
        package,
        path,
        evidence,
        group_for_format(db, path, package.format),
    )
}

/// The display name of a look from its file stem. A `.cube` or `.alf4` has a title; an `.aml` of
/// the LogC3 library comes in three strengths, so `...-Black-and-White-Low` is "Black and White ·
/// Low" (a stem that ends in anything else, such as `Beauty-How`, keeps all its words).
fn title_of(db: &PackageDb, format: LookFormat, stem: &str) -> String {
    if format == LookFormat::Aml {
        let (base, strength) = split_strength(stem);
        if let Some(strength) = strength {
            return format!("{} · {strength}", look_name(base, &db.display_names));
        }
    }
    look_name(stem, &db.display_names)
}

/// [`outcome_for`] with the group decided by the caller (`None`: ungrouped).
fn outcome_in_group(
    db: &PackageDb,
    package: &Package,
    path: &str,
    evidence: Evidence,
    group: Option<(u32, String)>,
) -> Outcome {
    let stem = file_stem(path);
    if package.role == PackageRole::DisplayLuts {
        Outcome::ArriDisplay {
            package: package.id.clone(),
            evidence,
        }
    } else {
        let (group_no, group_name) = group.unwrap_or_else(|| (0, "Ungrouped".to_string()));
        Outcome::ArriLook {
            package: package.id.clone(),
            group_no,
            group_name,
            input: package.look_input().unwrap_or(LookInput::LogC4),
            name: title_of(db, package.format, stem),
            file_name: file_name(path).to_string(),
            evidence,
        }
    }
}

/// Fewest files a source may hold, as a share of what its package should have, and still be read
/// as that package when its name and LUT size say so: half, rounded up. Fewer files than the
/// package should have are then reported as an incomplete download (with the numbers); a handful
/// of files, such as ARRI's small WVO package, are not taken for a big package.
fn minimum_files(expected: usize) -> usize {
    expected.div_ceil(2).max(1)
}

/// The group a file gets when its folders are not ARRI's four-digit ones: its innermost folder
/// that holds at least two of the files (so one folder per look is no group), not counting the
/// folders every file shares (a single top-level folder is the download's wrapper, not a group).
/// `None` for a flat list.
fn lenient_groups(paths: &[&str]) -> Vec<Option<(u32, String)>> {
    let folders: Vec<Vec<&str>> = paths.iter().map(|p| folder_components(p)).collect();
    let Some(first) = folders.first() else {
        return Vec::new();
    };
    let shared = folders.iter().fold(first.len(), |shared, f| {
        shared.min(f.iter().zip(first).take_while(|(a, b)| a == b).count())
    });
    // How many files sit at or below each folder.
    let mut below: HashMap<String, usize> = HashMap::new();
    for f in &folders {
        for depth in shared + 1..=f.len() {
            *below.entry(f[..depth].join("/")).or_default() += 1;
        }
    }
    folders
        .iter()
        .map(|f| {
            (shared + 1..=f.len()).rev().find_map(|depth| {
                if below.get(&f[..depth].join("/")).copied().unwrap_or(0) < 2 {
                    return None;
                }
                let name = f[depth - 1];
                Some(parse_group_folder(name).unwrap_or_else(|| {
                    let tidy = name.replace('_', " ");
                    let tidy = tidy.split_whitespace().collect::<Vec<_>>().join(" ");
                    (
                        0,
                        if tidy.is_empty() {
                            "Group".to_string()
                        } else {
                            tidy
                        },
                    )
                }))
            })
        })
        .collect()
}

/// The folders every file sits in, outermost first (`ARRI Look Library for LogC4 (3D-LUTs)`).
fn shared_folders(files: &[FileFacts]) -> Vec<&str> {
    let mut folders = files.iter().map(|f| folder_components(&f.rel_path));
    let Some(mut shared) = folders.next() else {
        return Vec::new();
    };
    for other in folders {
        let same = shared
            .iter()
            .zip(&other)
            .take_while(|(a, b)| a == b)
            .count();
        shared.truncate(same);
    }
    shared
}

/// Looks of one package and group that share a name (ARRI's `1110 Black and White` and
/// `1111 Black and White` are two looks) get their look code added to the name, so the list of
/// looks does not show the same name twice.
///
/// A file whose hash identified it gets the code if *any* two files of its package share its name,
/// whether or not the other one is in this source: the name (and with it the id) of a look is the
/// same whether the whole package is imported or one loose file of it, so the same look is never
/// two entries (`arri:1000:black-and-white-1111` either way). Files recognised by their names alone
/// have only the source to go by.
fn tell_same_names_apart(db: &PackageDb, files: &mut [RecognizedFile]) {
    let key = |package: &str, group_no: u32, name: &str| {
        (package.to_string(), group_no, name.to_lowercase())
    };
    type Counts = HashMap<(String, u32, String), usize>;
    // The files of this source.
    let mut count: Counts = HashMap::new();
    for f in files.iter() {
        if let Outcome::ArriLook {
            package,
            group_no,
            name,
            ..
        } = &f.outcome
        {
            *count.entry(key(package, *group_no, name)).or_default() += 1;
        }
    }
    // Every file a package lists, for the packages that a file of this source was identified by.
    let mut listed: Counts = HashMap::new();
    let mut counted: HashSet<&str> = HashSet::new();
    for f in files.iter() {
        if let Outcome::ArriLook {
            package,
            evidence: Evidence::Fingerprint,
            ..
        } = &f.outcome
            && counted.insert(package.as_str())
            && let Some(package) = db.package(package)
        {
            for fp in &package.fingerprints {
                if let Outcome::ArriLook {
                    package: id,
                    group_no,
                    name,
                    ..
                } = outcome_for(db, package, &fp.path, Evidence::Fingerprint)
                {
                    *listed.entry(key(&id, group_no, &name)).or_default() += 1;
                }
            }
        }
    }
    for f in files.iter_mut() {
        if let Outcome::ArriLook {
            package,
            group_no,
            name,
            file_name: library_name,
            evidence,
            ..
        } = &mut f.outcome
        {
            let k = key(package, *group_no, name);
            let here = count.get(&k).copied().unwrap_or(0);
            let in_package = if *evidence == Evidence::Fingerprint {
                listed.get(&k).copied().unwrap_or(0)
            } else {
                0
            };
            if here.max(in_package) > 1
                && let Some(code) = look_code(file_stem(library_name))
            {
                *name = with_code(name, code);
            }
        }
    }
}

/// `name` with a look code added: "Black and White 1110"; for a name with a strength, before it:
/// "Black and White 1110 · Low".
fn with_code(name: &str, code: u32) -> String {
    match name.rsplit_once(" · ") {
        Some((title, strength)) => format!("{title} {code} · {strength}"),
        None => format!("{name} {code}"),
    }
}

/// Whether a file was identified by its hash (which nothing overrides).
fn identified_by_hash(file: &RecognizedFile) -> bool {
    matches!(
        file.outcome,
        Outcome::ArriLook {
            evidence: Evidence::Fingerprint,
            ..
        }
    )
}

/// Rule 3: reads a whole zip or folder as one package. The name of the source must be the
/// package's (`zip_name` patterns, which include the encoding words and rule out the other
/// encodings and packages), and at least [`minimum_files`] valid `.cube` files must have a LUT
/// size the package allows. Layout is not asked for: group folders, a display name in a look's
/// name and the like decide only between two packages that both qualify.
///
/// When the name fits one package only, the name wins over what the files' own names suggested
/// (a flat `LogC3_to_Rec709` list inside "Look Library LogC3 to Rec709" is that look library, not
/// a set of display LUTs). Files identified by hash keep what the hash says. When the name fits
/// several packages, only files that fit nothing yet are assigned, to the package that stands out
/// (the file count closest to what it should have, then the layout that fits its markers best).
fn match_by_package(
    db: &PackageDb,
    source: &SourceInfo,
    files: &[FileFacts],
    recognized: &mut [RecognizedFile],
    package_of: &mut [Option<usize>],
) {
    if source.kind == SourceKind::File {
        return;
    }
    // The names that can say what the package is: the zip or folder, and the folders every file
    // sits in (ARRI's zips hold one folder named for the package, which survives a renamed zip).
    let wrapper = shared_folders(files);
    let names: Vec<&str> = std::iter::once(source.name.as_str())
        .chain(wrapper.iter().copied())
        .collect();
    let named: Vec<usize> = db
        .packages
        .iter()
        .enumerate()
        .filter(|(_, package)| {
            package.expected.cube_files.is_some()
                && !package.markers.encoding.is_empty()
                && names.iter().any(|name| {
                    let folded = fold_alnum(name.strip_suffix(".zip").unwrap_or(name));
                    package.matches_zip_name(name)
                        && package.markers.encoding.matches_folded(&folded)
                })
        })
        .map(|(i, _)| i)
        .collect();
    let decisive = named.len() == 1;
    // (package index, files that fit it)
    let mut qualified: Vec<(usize, Vec<usize>)> = Vec::new();
    for &i in &named {
        let package = &db.packages[i];
        let Some(expected) = package.expected.cube_files else {
            continue;
        };
        // A package holds files of its own kind: `.cube` LUTs, or look files of one container.
        let members: Vec<usize> = (0..files.len())
            .filter(|&j| match &files[j].check {
                Ok(summary) => {
                    files[j].format == package.format
                        && lut_size_fits(package, summary.lut_size)
                        && (package_of[j] == Some(i)
                            || recognized[j].needs_answer()
                            || recognized[j].is_own_look_file()
                            || (decisive && !identified_by_hash(&recognized[j])))
                }
                Err(_) => false,
            })
            .collect();
        if members.len() >= minimum_files(expected) {
            qualified.push((i, members));
        }
    }
    let chosen = match qualified.as_slice() {
        [] => return,
        [one] => one,
        several => {
            let layout = |(i, members): &(usize, Vec<usize>)| {
                let markers = &db.packages[*i].markers;
                members
                    .iter()
                    .filter(|&&j| {
                        let Ok(summary) = &files[j].check else {
                            return false;
                        };
                        markers.group_folders.allows(
                            group_for_format(db, &files[j].rel_path, db.packages[*i].format)
                                .is_some(),
                        ) && markers
                            .display_rendering
                            .allows(names_display_rendering(db, source, &files[j], summary))
                    })
                    .count()
            };
            let rank = |q: &(usize, Vec<usize>)| {
                let expected = db.packages[q.0].expected.cube_files.unwrap_or(0);
                (expected.abs_diff(q.1.len()), std::cmp::Reverse(layout(q)))
            };
            let best = several.iter().map(rank).min();
            let mut top = several.iter().filter(|q| Some(rank(q)) == best);
            match (top.next(), top.next()) {
                (Some(one), None) => one,
                _ => return,
            }
        }
    };
    let (index, members) = chosen;
    let package = &db.packages[*index];
    let paths: Vec<&str> = members
        .iter()
        .map(|&j| files[j].rel_path.as_str())
        .collect();
    let groups = lenient_groups(&paths);
    for (&j, lenient) in members.iter().zip(groups) {
        if package_of[j] != Some(*index) {
            let path = &files[j].rel_path;
            let group = group_for_format(db, path, package.format).or(lenient);
            recognized[j].outcome = outcome_in_group(db, package, path, Evidence::Pattern, group);
        }
        package_of[j] = Some(*index);
    }
}

/// Picks the fingerprint a file with a known hash stands for, among `candidates` (package index,
/// fingerprint index) that all have this hash.
///
/// A fingerprint that nobody has claimed yet is preferred, so identical files published under two
/// names are both accounted for; among those, the one whose path or file name the file still
/// carries, then the first. When every candidate is already claimed (the source holds more copies
/// than the package lists) the first one is reused: it is the same content, and importing it again
/// changes nothing.
fn claim_fingerprint(
    db: &PackageDb,
    candidates: &[(usize, usize)],
    claimed: &HashSet<(usize, usize)>,
    rel_path: &str,
) -> Option<(usize, usize)> {
    let free: Vec<(usize, usize)> = candidates
        .iter()
        .copied()
        .filter(|c| !claimed.contains(c))
        .collect();
    let pool = if free.is_empty() { candidates } else { &free };
    let path_of = |&(i, j): &(usize, usize)| db.packages[i].fingerprints[j].path.as_str();
    pool.iter()
        .find(|c| path_of(c).eq_ignore_ascii_case(rel_path))
        .or_else(|| {
            pool.iter()
                .find(|c| file_name(path_of(c)).eq_ignore_ascii_case(file_name(rel_path)))
        })
        .or_else(|| pool.first())
        .copied()
}

/// Decides what each file is and whether the packages are complete.
#[must_use]
pub fn recognize(
    db: &PackageDb,
    source: &SourceInfo,
    files: &[FileFacts],
    ignored: &Ignored,
) -> Recognition {
    // Rule 2: hash lookup. A hash can belong to more than one fingerprint (the same LUT published
    // under two names), so each hash maps to a list; every source file takes a fingerprint of its
    // own where it can, and `claimed` remembers which ones are taken.
    let mut by_hash: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for (i, package) in db.packages.iter().enumerate() {
        for (j, fp) in package.fingerprints.iter().enumerate() {
            by_hash.entry(fp.sha256.as_str()).or_default().push((i, j));
        }
    }
    let mut claimed: HashSet<(usize, usize)> = HashSet::new();

    let mut recognized: Vec<RecognizedFile> = Vec::with_capacity(files.len());
    // The index of the package each recognized file went to.
    let mut package_of: Vec<Option<usize>> = Vec::with_capacity(files.len());
    for file in files {
        let mut out = RecognizedFile {
            rel_path: file.rel_path.clone(),
            name: file_name(&file.rel_path).to_string(),
            size: file.size,
            sha256: file.sha256.clone(),
            lut_size: None,
            format: file.format,
            outcome: Outcome::Rejected {
                reason: String::new(),
            },
        };
        let mut package_index = None;
        match &file.check {
            Err(reason) => {
                out.outcome = Outcome::Rejected {
                    reason: reason.clone(),
                };
            }
            Ok(summary) => {
                out.lut_size = Some(summary.lut_size);
                let known = by_hash
                    .get(file.sha256.as_str())
                    .and_then(|list| claim_fingerprint(db, list, &claimed, &file.rel_path));
                if let Some((i, j)) = known {
                    claimed.insert((i, j));
                    // The fingerprinted path says what the file is, wherever the user keeps it.
                    let fp = &db.packages[i].fingerprints[j];
                    out.outcome = outcome_for(db, &db.packages[i], &fp.path, Evidence::Fingerprint);
                    package_index = Some(i);
                } else if let Some(look) = file.look_file {
                    // A look file knows its own input; it is a package's only by the package rule.
                    out.outcome = Outcome::OwnLookFile {
                        input: look.input,
                        display_transforms_ignored: look.display_transforms_ignored,
                    };
                } else {
                    match match_pattern(db, source, file, summary) {
                        Pattern::Match(package) => {
                            out.outcome =
                                outcome_for(db, package, &file.rel_path, Evidence::Pattern);
                            package_index = db.packages.iter().position(|p| p.id == package.id);
                        }
                        Pattern::NoMatch { suggestion, note } => {
                            out.outcome = Outcome::NeedsAnswer {
                                suggested_input: suggestion,
                                note,
                            };
                        }
                    }
                }
            }
        }
        recognized.push(out);
        package_of.push(package_index);
    }

    match_by_package(db, source, files, &mut recognized, &mut package_of);
    tell_same_names_apart(db, &mut recognized);

    let packages = findings(db, source, &recognized, &package_of, &claimed);
    let issues = issues_for(db, source, ignored, &recognized, &packages);
    Recognition {
        source: source.clone(),
        files: recognized,
        packages,
        ignored: *ignored,
        issues,
    }
}

fn findings(
    db: &PackageDb,
    _source: &SourceInfo,
    files: &[RecognizedFile],
    package_of: &[Option<usize>],
    claimed: &HashSet<(usize, usize)>,
) -> Vec<PackageFinding> {
    let mut out = Vec::new();
    for (index, package) in db.packages.iter().enumerate() {
        // A display-LUT package is recognised but not offered: `issues_for` says so in one sentence.
        if package.role == PackageRole::DisplayLuts {
            continue;
        }
        let members: Vec<&RecognizedFile> = files
            .iter()
            .zip(package_of)
            .filter(|(_, p)| **p == Some(index))
            .map(|(f, _)| f)
            .collect();
        if members.is_empty() {
            continue;
        }
        let by_pattern = members
            .iter()
            .filter(|f| {
                matches!(
                    &f.outcome,
                    Outcome::ArriLook {
                        evidence: Evidence::Pattern,
                        ..
                    }
                )
            })
            .count();
        let (found, expected, missing, unexpected) = if package.is_fingerprinted() {
            // One count per published file that is in the source, however many files share content.
            let mut found = 0_usize;
            let mut missing = Vec::new();
            for (j, fp) in package.fingerprints.iter().enumerate() {
                if claimed.contains(&(index, j)) {
                    found += 1;
                } else {
                    missing.push(fp.path.clone());
                }
            }
            (found, Some(package.fingerprints.len()), missing, by_pattern)
        } else {
            (members.len(), package.expected.cube_files, Vec::new(), 0)
        };
        let status = match expected {
            Some(e) if found < e => PackageStatus::Incomplete,
            Some(e) if found > e => PackageStatus::MoreThanExpected,
            _ => PackageStatus::Complete,
        };
        let mut lut_sizes: Vec<usize> = members.iter().filter_map(|f| f.lut_size).collect();
        lut_sizes.sort_unstable();
        lut_sizes.dedup();
        let groups: HashSet<u32> = members
            .iter()
            .filter_map(|f| match &f.outcome {
                Outcome::ArriLook { group_no, .. } => Some(*group_no),
                _ => None,
            })
            .collect();
        out.push(PackageFinding {
            id: package.id.clone(),
            title: package.title.clone(),
            role: package.role,
            evidence: if package.is_fingerprinted() && by_pattern == 0 {
                Evidence::Fingerprint
            } else {
                Evidence::Pattern
            },
            found,
            expected,
            status,
            missing,
            unexpected,
            lut_sizes,
            groups: groups.len(),
        });
    }
    out
}

fn issues_for(
    db: &PackageDb,
    source: &SourceInfo,
    ignored: &Ignored,
    files: &[RecognizedFile],
    packages: &[PackageFinding],
) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut push = |severity, kind, message: String| {
        issues.push(Issue {
            severity,
            kind,
            message,
        });
    };

    if files.is_empty() {
        push(
            Severity::Problem,
            IssueKind::NoLookFiles,
            explain::nothing_to_import(source, ignored),
        );
        return issues;
    }

    let rejected: Vec<(&RecognizedFile, &str)> = files
        .iter()
        .filter_map(|f| match &f.outcome {
            Outcome::Rejected { reason } => Some((f, reason.as_str())),
            _ => None,
        })
        .collect();
    if rejected.len() == files.len() {
        let example = rejected
            .first()
            .map(|(f, reason)| explain::rejected(&f.name, reason))
            .unwrap_or_default();
        push(
            Severity::Problem,
            IssueKind::AllRejected,
            explain::all_rejected(source, files, &example),
        );
        return issues;
    }
    // ARRI's display LUTs are recognised and never imported (Yellowmoon uses its own rendering).
    let display_luts = files.iter().filter(|f| f.is_display_lut()).count();
    if display_luts > 0 {
        let importable = files
            .iter()
            .filter(|f| f.is_recognized() || f.needs_answer())
            .count();
        if importable == 0 {
            push(
                Severity::Problem,
                IssueKind::DisplayRenderings,
                explain::display_renderings_only(source),
            );
        } else {
            push(
                Severity::Info,
                IssueKind::DisplayRenderings,
                explain::display_renderings_left_out(display_luts),
            );
        }
    }
    for (file, reason) in rejected.iter().take(MAX_REJECTED_MESSAGES) {
        push(
            Severity::Problem,
            IssueKind::Rejected,
            explain::rejected(&file.name, reason),
        );
    }
    if rejected.len() > MAX_REJECTED_MESSAGES {
        push(
            Severity::Problem,
            IssueKind::Rejected,
            explain::more_rejected(rejected.len() - MAX_REJECTED_MESSAGES),
        );
    }

    for finding in packages {
        match (finding.status, finding.expected) {
            (PackageStatus::Incomplete, Some(expected)) => push(
                Severity::Problem,
                IssueKind::Incomplete,
                explain::incomplete(
                    source,
                    &finding.title,
                    finding.found,
                    expected,
                    &finding.missing,
                ),
            ),
            (PackageStatus::MoreThanExpected, Some(expected)) => push(
                Severity::Info,
                IssueKind::MoreThanExpected,
                explain::more_than_expected(&finding.title, finding.found, expected),
            ),
            _ => {}
        }
        if finding.unexpected > 0 {
            push(
                Severity::Info,
                IssueKind::UnexpectedFiles,
                explain::unexpected_files(&finding.title, finding.unexpected),
            );
        }
    }

    let setting_aside: Vec<&str> = files
        .iter()
        .filter(|f| {
            matches!(
                f.outcome,
                Outcome::OwnLookFile {
                    display_transforms_ignored: true,
                    ..
                }
            )
        })
        .map(|f| f.name.as_str())
        .collect();
    if !setting_aside.is_empty() {
        push(
            Severity::Info,
            IssueKind::DisplayTransformsIgnored,
            explain::display_transforms_ignored(&setting_aside),
        );
    }

    let asking: Vec<&RecognizedFile> = files.iter().filter(|f| f.needs_answer()).collect();
    if !asking.is_empty() {
        let mut suggestions: Vec<LookInput> = asking
            .iter()
            .filter_map(|f| match &f.outcome {
                Outcome::NeedsAnswer {
                    suggested_input, ..
                } => *suggested_input,
                _ => None,
            })
            .collect();
        suggestions.dedup();
        let suggestion = match suggestions.as_slice() {
            [one] => Some(match one {
                LookInput::LogC4 => "LogC4",
                LookInput::LogC3Ei800 => "LogC3",
                LookInput::LogC3ToRec709 => "LogC3 with Rec.709 baked in",
            }),
            _ => None,
        };
        push(
            Severity::Question,
            IssueKind::NeedsAnswer,
            explain::needs_answer(asking.len(), suggestion),
        );
        // Names that say ARRI, but no package fits: say so plainly.
        let names_arri = asking.iter().any(|f| {
            let text = fold_alnum(&format!("{} {}", source.name, f.rel_path));
            text.contains("arri") || text.contains("alexa")
        });
        if names_arri {
            let mut titles: Vec<String> = Vec::new();
            for package in &db.packages {
                // A package that comes as 3D LUTs and as look files has one title.
                if !titles.contains(&package.title) {
                    titles.push(package.title.clone());
                }
            }
            if !titles.is_empty() {
                push(
                    Severity::Problem,
                    IssueKind::UnknownArriPackage,
                    explain::unknown_arri(&titles),
                );
            }
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A package list with small numbers, built from the real file's shape.
    fn db() -> PackageDb {
        PackageDb::from_json(
            r#"{
            "version": 1,
            "packages": [
              { "id": "ll4", "title": "Look Library · LogC4", "role": "looks", "input": "logc4",
                "arri_title": "Look Library for LogC4 (3D-LUTs)",
                "zip_name": [ { "all_of": ["look", "logc4"], "none_of": ["logc3"] } ],
                "expected": { "cube_files": 3, "lut_sizes": [5] },
                "markers": { "encoding": { "any_of": ["logc4"], "none_of": ["logc3"] },
                             "display_rendering": "absent", "group_folders": "required" } },
              { "id": "ll3", "title": "Look Library · LogC3", "role": "looks", "input": "logc3",
                "expected": { "cube_files": 3, "lut_sizes": [3] },
                "markers": { "encoding": { "any_of": ["logc3"], "none_of": ["logc4"] },
                             "display_rendering": "absent", "group_folders": "required" } },
              { "id": "ll3r", "title": "Look Library · LogC3 to Rec.709", "role": "looks", "input": "logc3_to_rec709",
                "expected": { "cube_files": 2 },
                "markers": { "encoding": { "any_of": ["logc3"], "none_of": ["logc4"] },
                             "display_rendering": "required", "group_folders": "required" } },
              { "id": "dl4", "title": "Display LUTs · LogC4", "role": "display_luts", "input": "logc4",
                "expected": { "cube_files": 2 },
                "markers": { "encoding": { "any_of": ["logc4"], "none_of": ["logc3"] },
                             "display_rendering": "required", "group_folders": "absent" } }
            ] }"#,
        )
        .unwrap()
    }

    fn zip(name: &str) -> SourceInfo {
        SourceInfo {
            kind: SourceKind::Zip,
            name: name.to_string(),
        }
    }

    fn facts(path: &str, size: usize, title: Option<&str>, salt: u8) -> FileFacts {
        FileFacts {
            rel_path: path.to_string(),
            size: 1000,
            sha256: format!("{salt:02x}").repeat(32),
            format: LookFormat::Cube,
            check: Ok(CubeSummary {
                title: title.map(str::to_string),
                lut_size: size,
                has_shaper: false,
            }),
            look_file: None,
        }
    }

    #[test]
    fn logc4_look_package_is_recognized_by_names_and_sizes() {
        let files = vec![
            facts("Looks/3000 Film/Film_Warm_LogC4.cube", 5, None, 1),
            facts("Looks/3000 Film/Film_Cool_LogC4.cube", 5, None, 2),
            facts("Looks/4000 Mono/Mono_LogC4.cube", 5, None, 3),
        ];
        let r = recognize(
            &db(),
            &zip("Look_Library_LogC4.zip"),
            &files,
            &Ignored::default(),
        );
        assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
        assert_eq!(r.packages.len(), 1);
        let p = &r.packages[0];
        assert_eq!(p.id, "ll4");
        assert_eq!((p.found, p.expected), (3, Some(3)));
        assert_eq!(p.status, PackageStatus::Complete);
        assert_eq!(p.groups, 2);
        match &r.files[0].outcome {
            Outcome::ArriLook {
                group_no,
                group_name,
                input,
                name,
                evidence,
                ..
            } => {
                assert_eq!((*group_no, group_name.as_str()), (3000, "Film"));
                assert_eq!(*input, LookInput::LogC4);
                assert_eq!(name, "Film Warm");
                assert_eq!(*evidence, Evidence::Pattern);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_title_header_can_carry_the_encoding() {
        let files = vec![facts("3000 Film/Warm.cube", 3, Some("ARRI LogC3 Warm"), 1)];
        let r = recognize(&db(), &zip("looks.zip"), &files, &Ignored::default());
        match &r.files[0].outcome {
            Outcome::ArriLook { package, input, .. } => {
                assert_eq!(package, "ll3");
                assert_eq!(*input, LookInput::LogC3Ei800);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn logc3_with_a_display_name_is_a_rec709_baked_look() {
        let files = vec![
            facts("3000 Film/Warm_LogC3_to_Rec709.cube", 3, None, 1),
            facts("3000 Film/Cool_LogC3_to_Rec709.cube", 3, None, 2),
        ];
        let r = recognize(&db(), &zip("x.zip"), &files, &Ignored::default());
        assert_eq!(r.packages.len(), 1);
        assert_eq!(r.packages[0].id, "ll3r");
        match &r.files[0].outcome {
            Outcome::ArriLook { input, name, .. } => {
                assert_eq!(*input, LookInput::LogC3ToRec709);
                assert_eq!(name, "Warm");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn display_luts_have_no_group_folder() {
        let files = vec![
            facts("LogC4_to_Gamma24_Rec709.cube", 5, None, 1),
            facts("LogC4_to_P3_D65.cube", 5, None, 2),
        ];
        let r = recognize(
            &db(),
            &zip("Display_LUTs_LogC4.zip"),
            &files,
            &Ignored::default(),
        );
        assert!(
            matches!(&r.files[0].outcome, Outcome::ArriDisplay { package, .. } if package == "dl4"),
            "{:?}",
            r.files[0].outcome
        );
        assert_eq!(r.display_lut_count(), 2);
    }

    #[test]
    fn a_package_of_display_luts_has_nothing_to_import() {
        let files = vec![
            facts("LogC4_to_Gamma24_Rec709.cube", 5, None, 1),
            facts("LogC4_to_P3_D65.cube", 5, None, 2),
        ];
        let r = recognize(
            &db(),
            &zip("Display_LUTs_LogC4.zip"),
            &files,
            &Ignored::default(),
        );
        // Recognised, and said in one sentence; not a package of looks, and not rejected.
        assert_eq!(r.verdict(), Verdict::Blocked);
        assert!(r.packages.is_empty());
        assert_eq!((r.recognized_count(), r.rejected_count()), (0, 0));
        let problem = r.problems().next().expect("a problem");
        assert_eq!(problem.kind, IssueKind::DisplayRenderings);
        assert_eq!(
            problem.message,
            "This package has only display renderings. Yellowmoon uses its own rendering, so there is nothing to import."
        );
        // A single display LUT chosen by itself.
        let one = recognize(
            &db(),
            &SourceInfo {
                kind: SourceKind::File,
                name: "LogC4_to_Gamma24_Rec709.cube".to_string(),
            },
            &files[..1],
            &Ignored::default(),
        );
        assert_eq!(one.verdict(), Verdict::Blocked);
        assert_eq!(
            one.problems().next().map(|i| i.message.as_str()),
            Some(
                "This file is a display rendering. Yellowmoon uses its own rendering, so there is nothing to import."
            )
        );
    }

    #[test]
    fn display_luts_beside_looks_are_left_out_with_a_remark() {
        let files = vec![
            facts("3000 Film/Warm_LogC4.cube", 5, None, 1),
            facts("3000 Film/Cool_LogC4.cube", 5, None, 4),
            facts("1000 Colour/Plain_LogC4.cube", 5, None, 5),
            facts("LogC4_to_Gamma24_Rec709.cube", 5, None, 2),
            facts("LogC4_to_P3_D65.cube", 5, None, 3),
        ];
        let r = recognize(&db(), &zip("everything.zip"), &files, &Ignored::default());
        assert_eq!(r.display_lut_count(), 2, "{:?}", r.files);
        assert_eq!(r.recognized_count(), 3);
        assert_eq!(r.verdict(), Verdict::Ready);
        assert!(r.problems().next().is_none(), "{:?}", r.issues);
        let remark = r
            .issues
            .iter()
            .find(|i| i.kind == IssueKind::DisplayRenderings)
            .expect("a remark");
        assert_eq!(remark.severity, Severity::Info);
        assert_eq!(
            remark.message,
            "2 files are display renderings, so they were left out. Yellowmoon uses its own rendering."
        );
        assert!(r.packages.iter().all(|p| p.role == PackageRole::Looks));
    }

    #[test]
    fn lut_size_must_agree_with_the_package() {
        // Named LogC4 but 3-point: the size rule keeps it out of the LogC4 package.
        let files = vec![facts("3000 Film/Warm_LogC4.cube", 3, None, 1)];
        let r = recognize(&db(), &zip("x.zip"), &files, &Ignored::default());
        assert!(r.files[0].needs_answer());
        assert!(r.packages.is_empty());
    }

    #[test]
    fn unknown_files_get_the_question_with_a_suggestion() {
        let files = vec![facts("my_look_LogC4.cube", 33, None, 1)];
        let r = recognize(
            &db(),
            &zip("my_look_LogC4.cube"),
            &files,
            &Ignored::default(),
        );
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        match &r.files[0].outcome {
            Outcome::NeedsAnswer {
                suggested_input, ..
            } => assert_eq!(*suggested_input, Some(LookInput::LogC4)),
            other => panic!("{other:?}"),
        }
        let q = r
            .issues
            .iter()
            .find(|i| i.kind == IssueKind::NeedsAnswer)
            .unwrap();
        assert_eq!(q.severity, Severity::Question);
        assert!(q.message.contains("suggest LogC4"), "{}", q.message);
        // No ARRI marker, so no "unknown ARRI package" warning.
        assert!(
            r.issues
                .iter()
                .all(|i| i.kind != IssueKind::UnknownArriPackage)
        );
    }

    #[test]
    fn conflicting_encodings_are_not_guessed() {
        let files = vec![facts(
            "3000 Film/Warm_LogC4.cube",
            5,
            Some("LogC3 version"),
            1,
        )];
        let r = recognize(&db(), &zip("x.zip"), &files, &Ignored::default());
        assert!(r.files[0].needs_answer());
        match &r.files[0].outcome {
            Outcome::NeedsAnswer {
                suggested_input, ..
            } => assert_eq!(*suggested_input, None),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn arri_named_files_from_an_unknown_package_are_explained() {
        let files = vec![facts("ARRI_ALEXA35_Custom/Thing.cube", 17, None, 1)];
        let r = recognize(
            &db(),
            &zip("ARRI_ALEXA35_Custom.zip"),
            &files,
            &Ignored::default(),
        );
        let issue = r
            .issues
            .iter()
            .find(|i| i.kind == IssueKind::UnknownArriPackage)
            .expect("explanation");
        assert!(
            issue.message.contains("another camera"),
            "{}",
            issue.message
        );
        assert!(issue.message.contains("Look Library · LogC4"));
    }

    #[test]
    fn incomplete_packages_are_flagged() {
        let files = vec![
            facts("3000 Film/A_LogC4.cube", 5, None, 1),
            facts("3000 Film/B_LogC4.cube", 5, None, 2),
        ];
        let r = recognize(
            &db(),
            &zip("Look_Library_LogC4.zip"),
            &files,
            &Ignored::default(),
        );
        assert_eq!(r.verdict(), Verdict::Incomplete);
        assert_eq!(r.packages[0].status, PackageStatus::Incomplete);
        let issue = r.problems().next().unwrap();
        assert_eq!(issue.kind, IssueKind::Incomplete);
        assert!(
            issue.message.contains("should have 3 files"),
            "{}",
            issue.message
        );
    }

    #[test]
    fn extra_files_are_only_noted() {
        let files: Vec<_> = (0..4)
            .map(|i| facts(&format!("3000 Film/L{i}_LogC4.cube"), 5, None, i + 1))
            .collect();
        let r = recognize(&db(), &zip("x.zip"), &files, &Ignored::default());
        assert_eq!(r.packages[0].status, PackageStatus::MoreThanExpected);
        assert_eq!(r.verdict(), Verdict::Ready);
        assert!(r.problems().next().is_none());
    }

    #[test]
    fn fingerprints_identify_files_whatever_they_are_called() {
        let mut db = db();
        let hashes: Vec<String> = (1..=3_u8).map(|i| format!("{i:02x}").repeat(32)).collect();
        db.packages[0].fingerprints = hashes
            .iter()
            .enumerate()
            .map(|(i, h)| Fingerprint {
                path: format!("3000 Film/Look_{i}.cube"),
                size: 1000,
                lut_size: 5,
                sha256: h.clone(),
            })
            .collect();
        // Names say nothing, the LUT size is odd, and the zip is called "stuff.zip".
        let files = vec![
            facts("a.cube", 99, None, 1),
            facts("b.cube", 99, None, 2),
            facts("c.cube", 99, None, 3),
        ];
        let r = recognize(&db, &zip("stuff.zip"), &files, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
        assert_eq!(r.packages[0].evidence, Evidence::Fingerprint);
        match &r.files[1].outcome {
            Outcome::ArriLook {
                group_no,
                name,
                evidence,
                ..
            } => {
                assert_eq!(*group_no, 3000);
                assert_eq!(name, "Look 1");
                assert_eq!(*evidence, Evidence::Fingerprint);
            }
            other => panic!("{other:?}"),
        }

        // One file short: the missing path is named.
        let r = recognize(&db, &zip("stuff.zip"), &files[..2], &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Incomplete);
        assert_eq!(r.packages[0].missing, vec!["3000 Film/Look_2.cube"]);
        let message = &r.problems().next().unwrap().message;
        assert!(message.contains("Look_2.cube"), "{message}");
    }

    /// The real download's name (ARRI's own text, with a `_data` suffix).
    const REAL_NAME: &str = "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip";

    fn names_of(paths: &[&str]) -> Vec<FileFacts> {
        paths
            .iter()
            .enumerate()
            .map(|(i, p)| facts(p, 5, None, i as u8 + 1))
            .collect()
    }

    fn look(r: &Recognition, i: usize) -> (&str, u32, String, String) {
        match &r.files[i].outcome {
            Outcome::ArriLook {
                package,
                group_no,
                group_name,
                name,
                input,
                ..
            } => {
                assert_eq!(*input, LookInput::LogC4);
                (package, *group_no, group_name.clone(), name.clone())
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_zip_named_for_the_package_is_recognized_whatever_its_layout() {
        // None of these has a four-digit group folder, which the per-file rules ask for.
        let layouts: [(&str, &[&str]); 4] = [
            (
                "flat",
                &[
                    "ARRI_LogC4-to-LogC4_Neutral_v1_65.cube",
                    "ARRI_LogC4-to-LogC4_Warm_v1_65.cube",
                    "ARRI_LogC4-to-LogC4_Cool_v1_65.cube",
                ],
            ),
            (
                "one top-level folder",
                &[
                    "ARRI Look Library LogC4/Neutral.cube",
                    "ARRI Look Library LogC4/Warm.cube",
                    "ARRI Look Library LogC4/Cool.cube",
                ],
            ),
            (
                "one folder per look",
                &["Neutral/Neutral.cube", "Warm/Warm.cube", "Cool/Cool.cube"],
            ),
            (
                "named groups",
                &["Film/Neutral.cube", "Film/Warm.cube", "Mono/Cool.cube"],
            ),
        ];
        for (layout, paths) in layouts {
            let files = names_of(paths);
            let r = recognize(&db(), &zip(REAL_NAME), &files, &Ignored::default());
            assert_eq!(r.verdict(), Verdict::Ready, "{layout}: {:?}", r.issues);
            assert_eq!(r.packages.len(), 1, "{layout}");
            assert_eq!(r.packages[0].id, "ll4", "{layout}");
            assert_eq!(r.packages[0].status, PackageStatus::Complete, "{layout}");
            assert!(
                r.files.iter().all(RecognizedFile::is_recognized),
                "{layout}"
            );
        }
        // The names come from the files, not from the log-to-log prefix or the version and mesh.
        let flat = names_of(&[
            "ARRI_LogC4-to-LogC4_Neutral_v1_65.cube",
            "ARRI_LogC4-to-LogC4_Warm_v1_65.cube",
            "ARRI_LogC4-to-LogC4_Cool_v1_65.cube",
        ]);
        let r = recognize(&db(), &zip(REAL_NAME), &flat, &Ignored::default());
        assert_eq!(
            look(&r, 0),
            ("ll4", 0, "Ungrouped".into(), "Neutral".into())
        );
    }

    #[test]
    fn folders_that_hold_several_looks_become_groups_only_when_they_are_groups() {
        let files = names_of(&[
            "Pack/Film/A.cube",
            "Pack/Film/B.cube",
            "Pack/Mono/C.cube",
            "Pack/Mono/D.cube",
        ]);
        let r = recognize(&db(), &zip(REAL_NAME), &files, &Ignored::default());
        assert_eq!(look(&r, 0).2, "Film");
        assert_eq!(look(&r, 3).2, "Mono");
        assert_eq!(r.packages[0].groups, 1, "unnumbered groups share number 0");
        // Four-digit folders keep their numbers, and win over the other rule.
        let files = names_of(&["3000 Film/A.cube", "3000 Film/B.cube", "4000 Mono/C.cube"]);
        let r = recognize(&db(), &zip(REAL_NAME), &files, &Ignored::default());
        assert_eq!(look(&r, 2).1, 4000);
        assert_eq!(r.packages[0].groups, 2);
    }

    #[test]
    fn a_flat_list_with_look_codes_is_a_look_package_by_the_folder_name_alone() {
        // ARRI's real layout: one folder named for the package, flat files `<code> <title>.cube`,
        // no encoding word in them. These two packages have no zip_name patterns here, so only the
        // per-file rules can recognise them: the encoding comes from the folder, the group from the
        // first digit of the look code, and "to Rec709" in the folder says display-referred.
        let log = vec![
            facts(
                "ARRI Look Library LogC3 (log-to-log)/1110 Gray Test.cube",
                3,
                None,
                1,
            ),
            facts(
                "ARRI Look Library LogC3 (log-to-log)/3250 Cross.cube",
                3,
                None,
                2,
            ),
            facts(
                "ARRI Look Library LogC3 (log-to-log)/8120 20s.cube",
                3,
                None,
                3,
            ),
        ];
        let r = recognize(&db(), &zip("download.zip"), &log, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
        assert_eq!(r.packages[0].id, "ll3");
        assert_eq!(r.packages[0].groups, 3);
        match &r.files[2].outcome {
            Outcome::ArriLook {
                group_no,
                name,
                evidence,
                ..
            } => {
                assert_eq!((*group_no, name.as_str()), (8000, "20s"));
                assert_eq!(*evidence, Evidence::Pattern);
            }
            other => panic!("{other:?}"),
        }
        let baked = vec![
            facts(
                "ARRI Look Library LogC3 to Rec709 3D-LUTs/1110-Gray-Test.cube",
                3,
                None,
                1,
            ),
            facts(
                "ARRI Look Library LogC3 to Rec709 3D-LUTs/8120-20s.cube",
                3,
                None,
                2,
            ),
        ];
        let r = recognize(&db(), &zip("download.zip"), &baked, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
        assert_eq!(r.packages[0].id, "ll3r");
        match &r.files[1].outcome {
            Outcome::ArriLook {
                group_no, input, ..
            } => {
                assert_eq!(*group_no, 8000);
                assert_eq!(*input, LookInput::LogC3ToRec709);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn looks_that_share_a_title_keep_their_code() {
        let files = vec![
            facts(
                "ARRI Look Library LogC3 (log-to-log)/1110 Gray Test.cube",
                3,
                None,
                1,
            ),
            facts(
                "ARRI Look Library LogC3 (log-to-log)/1111 Gray Test.cube",
                3,
                None,
                2,
            ),
            facts(
                "ARRI Look Library LogC3 (log-to-log)/2110 Other.cube",
                3,
                None,
                3,
            ),
        ];
        let r = recognize(&db(), &zip("download.zip"), &files, &Ignored::default());
        let names: Vec<&str> = r
            .files
            .iter()
            .map(|f| match &f.outcome {
                Outcome::ArriLook { name, .. } => name.as_str(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(names, ["Gray Test 1110", "Gray Test 1111", "Other"]);
    }

    #[test]
    fn look_names_with_p3_pq_hlg_in_them_are_not_display_renderings() {
        // Per-file rules: a group folder and these names, and the log-to-log zip name: a look.
        let files = names_of(&[
            "3000 Film/ARRI_LogC4-to-LogC4_Pale to Pqrs_v1_65.cube",
            "3000 Film/ARRI_LogC4-to-LogC4_Warm P3 Glow_v1_65.cube",
            "3000 Film/ARRI_LogC4-to-LogC4_PQ Noir_v1_65.cube",
        ]);
        let r = recognize(&db(), &zip("misc.zip"), &files, &Ignored::default());
        assert_eq!(r.packages.len(), 1, "{:?}", r.issues);
        assert_eq!(r.packages[0].id, "ll4");
        assert_eq!(look(&r, 0).3, "Pale to Pqrs");
        assert_eq!(look(&r, 1).3, "Warm P3 Glow");
        // Package rule: the same names with no group folders at all.
        let files = names_of(&[
            "Pale to Pqrs/ARRI_LogC4-to-LogC4_Pale to Pqrs_v1_65.cube",
            "Warm P3 Glow/ARRI_LogC4-to-LogC4_Warm P3 Glow_v1_65.cube",
            "HLG Haze/ARRI_LogC4-to-LogC4_HLG Haze_v1_65.cube",
        ]);
        let r = recognize(&db(), &zip(REAL_NAME), &files, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
        assert_eq!(r.packages[0].id, "ll4");
    }

    #[test]
    fn the_package_rule_needs_the_name_the_lut_size_and_enough_files() {
        let flat = ["A.cube", "B.cube", "C.cube"];
        let sized = |size: usize| -> Vec<FileFacts> {
            flat.iter()
                .enumerate()
                .map(|(i, p)| facts(p, size, None, i as u8 + 1))
                .collect()
        };
        // Another name: the question, not a guess.
        let r = recognize(&db(), &zip("my looks.zip"), &sized(5), &Ignored::default());
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        assert!(r.packages.is_empty());
        // The LogC3 name does not make it the LogC4 package.
        let r = recognize(
            &db(),
            &zip("ARRI Look Library LogC3 (log-to-log).zip"),
            &sized(5),
            &Ignored::default(),
        );
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        // The right name, the wrong LUT size.
        let r = recognize(&db(), &zip(REAL_NAME), &sized(9), &Ignored::default());
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        assert!(r.packages.is_empty());
        // Too few files for a package that should have 3: one file is no package.
        let one = sized(5)[..1].to_vec();
        let r = recognize(&db(), &zip(REAL_NAME), &one, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        // Two of three is read as the package, short: the download may be incomplete.
        let two = sized(5)[..2].to_vec();
        let r = recognize(&db(), &zip(REAL_NAME), &two, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Incomplete);
        assert!(
            r.problems()
                .next()
                .unwrap()
                .message
                .contains("should have 3 files")
        );
        // More than expected is taken with a note.
        let many: Vec<FileFacts> = (0..5)
            .map(|i| facts(&format!("L{i}.cube"), 5, None, i + 1))
            .collect();
        let r = recognize(&db(), &zip(REAL_NAME), &many, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Ready);
        assert_eq!(r.packages[0].status, PackageStatus::MoreThanExpected);
        // A single .cube file is never a package, whatever it is called.
        let single = SourceInfo {
            kind: SourceKind::File,
            name: "ARRI Look Library LogC4.cube".to_string(),
        };
        let r = recognize(&db(), &single, &sized(5)[..3], &Ignored::default());
        assert!(r.packages.is_empty());
    }

    #[test]
    fn the_layout_only_breaks_a_tie_between_two_packages_that_both_fit() {
        // Two packages with one name pattern. Every file's TITLE names LogC3 as well as the zip's
        // LogC4, so the per-file rules (which read the titles) fit neither; the package rule reads
        // the name of the zip and fits both. The files have no group folders: the package that
        // does not ask for them is the better fit.
        let mut both = db();
        both.packages[1].zip_name = both.packages[0].zip_name.clone();
        both.packages[1].markers.encoding = both.packages[0].markers.encoding.clone();
        both.packages[1].expected = both.packages[0].expected.clone();
        both.packages[1].markers.group_folders = packages::Presence::Absent;
        let mixed: Vec<FileFacts> = ["A.cube", "B.cube", "C.cube"]
            .iter()
            .enumerate()
            .map(|(i, p)| facts(p, 5, Some("LogC3 and LogC4"), i as u8 + 1))
            .collect();
        let r = recognize(&both, &zip(REAL_NAME), &mixed, &Ignored::default());
        assert_eq!(r.packages.len(), 1, "{:?}", r.issues);
        assert_eq!(r.packages[0].id, "ll3");
        // Nothing separates them: no guess, the question.
        both.packages[1].markers.group_folders = both.packages[0].markers.group_folders;
        let r = recognize(&both, &zip(REAL_NAME), &mixed, &Ignored::default());
        assert!(r.packages.is_empty());
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        // A count that is closer to what a package should have decides before the layout does.
        both.packages[1].expected.cube_files = Some(5);
        both.packages[1].markers.group_folders = packages::Presence::Absent;
        let r = recognize(&both, &zip(REAL_NAME), &mixed, &Ignored::default());
        assert_eq!(
            r.packages[0].id, "ll4",
            "3 files: 0 away from 3, 2 away from 5"
        );
    }

    #[test]
    fn empty_and_all_rejected_sources_are_explained() {
        let xml = Ignored {
            xml: 5,
            ..Ignored::default()
        };
        let r = recognize(&db(), &zip("x.zip"), &[], &xml);
        assert_eq!(r.verdict(), Verdict::Blocked);
        assert_eq!(r.issues[0].kind, IssueKind::NoLookFiles);
        assert!(
            r.issues[0].message.contains("ALF-1"),
            "{}",
            r.issues[0].message
        );

        let bad = FileFacts {
            rel_path: "bad.cube".into(),
            size: 3,
            sha256: "0".repeat(64),
            format: LookFormat::Cube,
            check: Err("Line 1: unknown keyword.".into()),
            look_file: None,
        };
        let r = recognize(&db(), &zip("x.zip"), &[bad], &Ignored::default());
        assert_eq!(r.verdict(), Verdict::Blocked);
        assert_eq!(r.issues[0].kind, IssueKind::AllRejected);
        assert!(r.issues[0].message.contains("bad.cube cannot be used"));
        assert_eq!(r.rejected_count(), 1);
    }

    #[test]
    fn one_bad_file_among_good_ones_is_listed() {
        let mut files = vec![facts("a.cube", 33, None, 1)];
        files.push(FileFacts {
            rel_path: "bad.cube".into(),
            size: 3,
            sha256: "0".repeat(64),
            format: LookFormat::Cube,
            check: Err("It is not a text file, so it is not a .cube LUT.".into()),
            look_file: None,
        });
        let r = recognize(&db(), &zip("x.zip"), &files, &Ignored::default());
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        assert!(r.issues.iter().any(|i| i.kind == IssueKind::Rejected));
        assert_eq!(
            (
                r.recognized_count(),
                r.needs_answer_count(),
                r.rejected_count()
            ),
            (0, 1, 1)
        );
    }
}
