//! Deciding where an incoming file goes in the library: new look, same look again, updated look.
//!
//! Identity is the look id plus the file's SHA-256:
//!
//! * the id is already in the manifest with the **same hash**: nothing to add (re-importing a
//!   package is idempotent);
//! * the id is in the manifest with a **different hash** and the same file path, for ARRI files:
//!   ARRI re-released the file, so the entry is replaced in place. This only applies to entries
//!   that came from an earlier import: two files of one import that end up with the same id and
//!   path (names that differ only in punctuation, say) must not replace each other;
//! * the same ARRI look in two forms, a `.cube` and a look file (the LogC4 library comes as 3D LUTs
//!   and as `.alf4` files and both give the same ids): the `.cube` is the better one (the library's
//!   look files hold 33 points where its cubes hold 65), so a look file never replaces a `.cube`
//!   that an earlier import put there, and a `.cube` replaces the look file of an earlier import.
//!   The exception is a `.cube` that the library no longer holds intact (deleted or altered): a
//!   look file is then no reason to keep that entry as it is, so the look file takes its place
//!   ([`Slot::Repair`]) and importing the 3D-LUT version again brings the `.cube` back;
//! * the same ARRI look in two forms inside **one** import (a folder that holds both libraries):
//!   the same rule, so the `.cube` is kept and the look file is left out. [`crate::import`] plans the
//!   `.cube` files of a package before its look files, so the look file always meets the `.cube`;
//! * **the same bytes are never imported twice.** A file whose SHA-256 (with the same input) is
//!   already in the library is the same LUT whatever it is called now and whichever group it sits
//!   in: a copy of one of the person's own LUTs under a new name, or a loose file that is already
//!   one of ARRI's looks, is "already imported" ([`Slot::Same`]). When an ARRI package now claims a
//!   file that an earlier import put under "Your LUTs", that entry becomes the ARRI look (its ARRI
//!   id, name, group and place in the library) and no second entry is made ([`Slot::Claim`]). ARRI
//!   files are never merged with each other: ARRI's names decide which looks a package has, even if
//!   two of them hold the same LUT. Two files of one import are not, and a file that was identified
//!   by its hash (its id is ARRI's own, so it is the look that id says) is not folded into an
//!   earlier ARRI entry with another id either, whichever import that entry came from; only a file
//!   recognised by its name alone is taken for the ARRI look already there;
//! * any other collision (a user LUT with the same name but different content, two names that
//!   slugify alike, a path already taken) gets a number: `film`, `film-2`, `film-3`, so no file of
//!   the user's is ever overwritten.

use std::collections::HashSet;

use ym_color::look::LookInput;

use crate::library::{LookFormat, LookId, sanitize_file_stem, slugify, truncate_to_bytes};
use crate::manifest::{Manifest, ManifestGroup, ManifestLook, Source};
use crate::recognize::names::{file_name, file_stem};
use crate::recognize::{Evidence, Outcome};

/// What the recogniser (or the user's answer) says an incoming file is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spec {
    /// The id before any collision number.
    pub(crate) id: LookId,
    /// Display name.
    pub(crate) name: String,
    /// The group it is listed under.
    pub(crate) group: ManifestGroup,
    /// What the LUT expects as input.
    pub(crate) input: LookInput,
    /// Folder below `looks/`: `arri/3000-film` or `user`.
    pub(crate) folder: String,
    /// File stem to write, already sanitised and including any encoding suffix.
    pub(crate) stem: String,
    /// The kind of file, which gives the extension.
    pub(crate) format: LookFormat,
    /// Package id, for ARRI files.
    pub(crate) package: Option<String>,
    /// Whether the file came from an ARRI package (updates replace in place).
    pub(crate) arri: bool,
    /// Whether the file was identified by its SHA-256. Its id is then ARRI's own, which says which
    /// looks the package has: an ARRI file that holds the same bytes under another id is another
    /// look and is never folded into it ([`same_bytes`]).
    pub(crate) by_hash: bool,
    /// The file name it had.
    pub(crate) original_name: String,
}

/// Longest group part of an `arri/<group_no>-<group>` folder name, in bytes.
const MAX_GROUP_FOLDER_BYTES: usize = 64;

fn encoding_suffix(input: LookInput) -> &'static str {
    match input {
        LookInput::LogC4 => "",
        LookInput::LogC3Ei800 => "-logc3",
        LookInput::LogC3ToRec709 => "-logc3-rec709",
    }
}

impl Spec {
    /// The spec for a recognized file, or `None` for a rejected one. `answer` is the user's reply
    /// to the generic question and applies to files that need it; `format` is the kind of file.
    pub(crate) fn from_outcome(
        outcome: &Outcome,
        rel_path: &str,
        answer: Option<LookInput>,
        format: LookFormat,
    ) -> Option<Self> {
        let original_name = file_name(rel_path).to_string();
        let raw_stem = file_stem(rel_path);
        match outcome {
            Outcome::ArriLook {
                package,
                group_no,
                group_name,
                input,
                name,
                file_name: library_name,
                evidence,
                ..
            } => {
                let raw_stem = file_stem(library_name);
                let suffix = encoding_suffix(*input);
                let slug = slugify(name);
                let slug = if slug.is_empty() {
                    "look".to_string()
                } else {
                    slug
                };
                // A folder name is limited to 255 bytes too; group names are short in practice.
                let group_slug = slugify(group_name);
                let group_slug = truncate_to_bytes(&group_slug, MAX_GROUP_FOLDER_BYTES)
                    .trim_end_matches('-')
                    .to_string();
                let group_slug = if group_slug.is_empty() {
                    "group".to_string()
                } else {
                    group_slug
                };
                Some(Self {
                    id: LookId::arri(*group_no, &format!("{slug}{suffix}")),
                    name: name.clone(),
                    group: ManifestGroup::Arri {
                        no: *group_no,
                        name: group_name.clone(),
                    },
                    input: *input,
                    folder: format!("arri/{group_no:04}-{group_slug}"),
                    stem: format!("{}{suffix}", sanitize_file_stem(raw_stem)),
                    format,
                    package: Some(package.clone()),
                    arri: true,
                    by_hash: *evidence == Evidence::Fingerprint,
                    original_name,
                })
            }
            Outcome::NeedsAnswer { .. } => {
                let input = answer?;
                let name = raw_stem.trim();
                let name = if name.is_empty() { "LUT" } else { name };
                Some(Self {
                    id: LookId::user(&slugify(name)),
                    name: name.to_string(),
                    group: ManifestGroup::User,
                    input,
                    folder: "user".to_string(),
                    stem: sanitize_file_stem(raw_stem),
                    format,
                    package: None,
                    arri: false,
                    by_hash: false,
                    original_name,
                })
            }
            // A look file of the person's own: its name is the file's, its input the file's.
            Outcome::OwnLookFile { input, .. } => {
                let name = raw_stem.trim();
                let name = if name.is_empty() { "Look" } else { name };
                Some(Self {
                    id: LookId::user(&slugify(name)),
                    name: name.to_string(),
                    group: ManifestGroup::User,
                    input: *input,
                    folder: "user".to_string(),
                    stem: sanitize_file_stem(raw_stem),
                    format,
                    package: None,
                    arri: false,
                    by_hash: false,
                    original_name,
                })
            }
            // Display LUTs are not imported: Yellowmoon uses its own display rendering.
            Outcome::ArriDisplay { .. } | Outcome::Rejected { .. } => None,
        }
    }
}

/// What to do with an incoming file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Slot {
    /// A new entry.
    Add { id: LookId, file: String },
    /// An ARRI file that changed: replace the entry's content in place.
    Replace { id: LookId, file: String },
    /// Already imported with the same content.
    Existing { id: LookId, file: String },
    /// The same ARRI look is already imported as a `.cube` and this is a look file: the `.cube`
    /// stays and nothing is written. `file` is the `.cube` entry's file. `same_import` says the
    /// `.cube` is in this very import (a folder that holds both libraries) rather than in an
    /// earlier one.
    Superseded {
        id: LookId,
        file: String,
        same_import: bool,
        /// Where the look file itself would go: used when the `.cube` is gone or altered in the
        /// library, so that it is no reason to keep the entry as it is ([`Slot::Repair`]).
        own_file: String,
    },
    /// The look is imported as a `.cube` that the library no longer holds intact, and this is its
    /// look file: the entry moves to `file` (the look file) and the damaged `.cube` at `old_file`
    /// goes. Decided by [`crate::import::import`] from a look at the disk, never by [`assign`].
    Repair {
        id: LookId,
        file: String,
        old_file: String,
    },
    /// The same ARRI look is imported as a look file and this is its `.cube`: the entry moves to
    /// `file` and the look file `old_file` goes.
    Upgrade {
        id: LookId,
        file: String,
        old_file: String,
    },
    /// These exact bytes (with the same input) are already in the library under `id`, in whatever
    /// group and under whatever name: nothing is added. The entry is not touched.
    Same {
        id: LookId,
        file: String,
        /// The entry's own name, for the report.
        name: String,
    },
    /// An ARRI package claims bytes that an earlier import put under "Your LUTs": that entry
    /// becomes the ARRI one (`id` and `file` are the ARRI ones) and `old` is the entry that goes.
    Claim {
        id: LookId,
        file: String,
        old: Claimed,
    },
}

/// The user's entry that an ARRI package has claimed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Claimed {
    pub(crate) id: LookId,
    /// Its file, which goes once the new list is saved.
    pub(crate) file: String,
}

/// The stem with a collision number.
fn numbered_stem(stem: &str, n: u32) -> String {
    if n <= 1 {
        stem.to_string()
    } else {
        format!("{stem}-{n}")
    }
}

/// The hash, file and kind of the entry with this id.
fn existing<'a>(manifest: &'a Manifest, id: &LookId) -> Option<(&'a str, &'a str, LookFormat)> {
    manifest
        .look(id)
        .map(|l| (l.sha256.as_str(), l.file.as_str(), l.format))
}

fn file_taken(manifest: &Manifest, file: &str) -> bool {
    manifest
        .looks
        .iter()
        .map(|l| l.file.as_str())
        .any(|f| f.eq_ignore_ascii_case(file))
}

/// Most collision numbers tried before falling back to a hash-based name.
const MAX_COLLISION_NUMBER: u32 = 10_000;

/// The entry that already holds these exact bytes for the same input, if `spec` should be folded
/// into it (the rules are in the module documentation), as the slot to use for a file that would
/// otherwise be a new entry `id` at `file`.
///
/// ARRI files are only compared with entries of earlier imports (`touched` are this import's):
/// two files of one package that hold the same LUT are two looks, because ARRI's names say so.
fn same_bytes(
    manifest: &Manifest,
    spec: &Spec,
    sha256: &str,
    touched: &HashSet<LookId>,
    id: &LookId,
    file: &str,
) -> Option<Slot> {
    let usable = |entry: &LookId| !spec.arri || !touched.contains(entry);
    let claim = |l: &ManifestLook| Slot::Claim {
        id: id.clone(),
        file: file.to_string(),
        old: Claimed {
            id: l.id.clone(),
            file: l.file.clone(),
        },
    };
    let same_look = |l: &ManifestLook| Slot::Same {
        id: l.id.clone(),
        file: l.file.clone(),
        name: l.name.clone(),
    };
    let mut same = manifest
        .looks
        .iter()
        .filter(|l| l.sha256 == sha256 && usable(&l.id))
        .filter(|l| l.input == spec.input);
    if spec.arri {
        // A look of ARRI's that is there already, whatever its name (unless this file's own id is
        // ARRI's: it was identified by its hash, so it is the look its id says and not another
        // look that happens to hold the same LUT); else the person's own entry that this package
        // now claims.
        let all: Vec<&ManifestLook> = same.collect();
        let arri = if spec.by_hash {
            None
        } else {
            all.iter().find(|l| l.group != ManifestGroup::User)
        };
        arri.map(|l| same_look(l)).or_else(|| {
            all.iter()
                .find(|l| l.group == ManifestGroup::User)
                .map(|l| claim(l))
        })
    } else {
        same.next().map(same_look)
    }
}

/// Chooses the id and file for `spec` with content hash `sha256`, given the manifest as changed
/// so far in this import. `touched` holds the ids this import has already assigned (added,
/// replaced or matched): such an entry belongs to another file of the same import, so it is never
/// replaced, only matched when the content is the same.
///
/// A file that would become a new entry is first compared by its bytes with what the library
/// holds ([`Slot::Same`], [`Slot::Claim`]).
pub(crate) fn assign(
    manifest: &Manifest,
    spec: &Spec,
    sha256: &str,
    touched: &HashSet<LookId>,
) -> Slot {
    let slot = assign_by_id(manifest, spec, sha256, touched);
    if let Slot::Add { id, file } = &slot
        && let Some(same) = same_bytes(manifest, spec, sha256, touched, id, file)
    {
        return same;
    }
    slot
}

fn assign_by_id(manifest: &Manifest, spec: &Spec, sha256: &str, touched: &HashSet<LookId>) -> Slot {
    let extension = spec.format.extension();
    for n in 1..=MAX_COLLISION_NUMBER {
        let id = spec.id.numbered(n);
        let file = format!(
            "{}/{}.{extension}",
            spec.folder,
            numbered_stem(&spec.stem, n)
        );
        match existing(manifest, &id) {
            Some((sha, found, _)) if sha == sha256 => {
                return Slot::Existing {
                    id,
                    file: found.to_string(),
                };
            }
            Some((_, found, _))
                if spec.arri && !touched.contains(&id) && found.eq_ignore_ascii_case(&file) =>
            {
                return Slot::Replace { id, file };
            }
            // The same look in its other form: the `.cube` is the one to keep, wherever it came
            // from (an earlier import or a file of this one: a `.cube` is planned first).
            Some((_, found, found_format))
                if spec.arri
                    && found_format != spec.format
                    && (found_format.is_cube() || spec.format.is_cube())
                    && if spec.format.is_cube() {
                        !touched.contains(&id) && !file_taken(manifest, &file)
                    } else {
                        found_format.is_cube()
                    } =>
            {
                return if spec.format.is_cube() {
                    Slot::Upgrade {
                        id,
                        file,
                        old_file: found.to_string(),
                    }
                } else {
                    Slot::Superseded {
                        same_import: touched.contains(&id),
                        id,
                        file: found.to_string(),
                        own_file: file,
                    }
                };
            }
            Some(_) => {}
            None if file_taken(manifest, &file) => {}
            None => return Slot::Add { id, file },
        }
    }
    // Ten thousand files with the same name: name this one after its content.
    let short = &sha256[..sha256.len().min(12)];
    Slot::Add {
        id: spec.id.with_slug(&format!("{}-{short}", spec.id.slug())),
        file: format!("{}/{}-{short}.{extension}", spec.folder, spec.stem),
    }
}

/// Everything about a file that goes in its manifest entry.
pub(crate) struct EntryData<'a> {
    pub(crate) spec: &'a Spec,
    pub(crate) id: LookId,
    pub(crate) file: String,
    pub(crate) sha256: String,
    pub(crate) lut_size: usize,
    pub(crate) imported: &'a str,
    pub(crate) url: Option<&'a str>,
    /// The person's entry that this one takes the place of ([`Slot::Claim`]).
    pub(crate) replaces: Option<&'a LookId>,
}

/// Inserts or replaces the entry in `manifest`. Returns true when something changed.
pub(crate) fn apply(manifest: &mut Manifest, data: &EntryData<'_>) -> bool {
    if let Some(old) = data.replaces {
        manifest.looks.retain(|l| &l.id != old);
    }
    let source = Some(Source {
        package: data.spec.package.clone(),
        url: data.spec.package.as_ref().and(data.url).map(str::to_string),
        imported: data.imported.to_string(),
        original_name: data.spec.original_name.clone(),
    });
    let entry = ManifestLook {
        id: data.id.clone(),
        name: data.spec.name.clone(),
        group: data.spec.group.clone(),
        input: data.spec.input,
        file: data.file.clone(),
        format: data.spec.format,
        sha256: data.sha256.clone(),
        lut_size: data.lut_size,
        source,
    };
    match manifest.looks.iter_mut().find(|l| l.id == data.id) {
        Some(slot) => {
            if slot.sha256 == entry.sha256 {
                // The same file again keeps its import date and source, but takes a changed answer
                // to the input question.
                let changed = slot.input != entry.input;
                slot.input = entry.input;
                changed
            } else {
                *slot = entry;
                true
            }
        }
        None => {
            manifest.looks.push(entry);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_spec(name: &str) -> Spec {
        Spec::from_outcome(
            &Outcome::NeedsAnswer {
                suggested_input: None,
                note: None,
            },
            &format!("{name}.cube"),
            Some(LookInput::LogC4),
            LookFormat::Cube,
        )
        .unwrap()
    }

    fn arri_spec(input: LookInput, name: &str) -> Spec {
        Spec::from_outcome(
            &Outcome::ArriLook {
                package: "p".into(),
                group_no: 3000,
                group_name: "Film".into(),
                input,
                name: name.into(),
                file_name: format!("{name}.cube"),
                evidence: crate::recognize::Evidence::Pattern,
            },
            &format!("3000 Film/{name}.cube"),
            None,
            LookFormat::Cube,
        )
        .unwrap()
    }

    fn put(manifest: &mut Manifest, spec: &Spec, sha: &str) -> Slot {
        let slot = assign(manifest, spec, sha, &HashSet::new());
        let (id, file) = match &slot {
            Slot::Add { id, file }
            | Slot::Replace { id, file }
            | Slot::Existing { id, file }
            | Slot::Superseded { id, file, .. }
            | Slot::Upgrade { id, file, .. }
            | Slot::Repair { id, file, .. }
            | Slot::Same { id, file, .. }
            | Slot::Claim { id, file, .. } => (id.clone(), file.clone()),
        };
        let replaces = match &slot {
            Slot::Claim { old, .. } => Some(old.id.clone()),
            _ => None,
        };
        // A look that is already there is not written again (what the importer does).
        if !matches!(slot, Slot::Same { .. } | Slot::Superseded { .. }) {
            apply(
                manifest,
                &EntryData {
                    spec,
                    id,
                    file,
                    sha256: sha.to_string(),
                    lut_size: 33,
                    imported: "2026-01-01T00:00:00Z",
                    url: Some("https://example.invalid/"),
                    replaces: replaces.as_ref(),
                },
            );
        }
        slot
    }

    fn sha(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    #[test]
    fn specs_name_ids_folders_and_files() {
        let c4 = arri_spec(LookInput::LogC4, "Film Warm");
        assert_eq!(c4.id.as_str(), "arri:3000:film-warm");
        assert_eq!(c4.folder, "arri/3000-film");
        assert_eq!(c4.stem, "Film Warm");
        let c3 = arri_spec(LookInput::LogC3Ei800, "Film Warm");
        assert_eq!(c3.id.as_str(), "arri:3000:film-warm-logc3");
        assert_eq!(c3.stem, "Film Warm-logc3");
        let r709 = arri_spec(LookInput::LogC3ToRec709, "Film Warm");
        assert_eq!(r709.id.as_str(), "arri:3000:film-warm-logc3-rec709");
        let user = user_spec("My Look v2");
        assert_eq!(user.id.as_str(), "user:my-look-v2");
        assert_eq!(user.folder, "user");
        assert!(!user.arri);
        assert!(
            Spec::from_outcome(
                &Outcome::NeedsAnswer {
                    suggested_input: None,
                    note: None
                },
                "x.cube",
                None,
                LookFormat::Cube
            )
            .is_none(),
            "no answer, no spec"
        );
    }

    #[test]
    fn an_arri_display_lut_has_no_spec_because_it_is_never_imported() {
        let outcome = Outcome::ArriDisplay {
            package: "d".into(),
            evidence: crate::recognize::Evidence::Pattern,
        };
        assert!(
            Spec::from_outcome(&outcome, "LogC4_to_Rec709.cube", None, LookFormat::Cube).is_none()
        );
        assert!(
            Spec::from_outcome(
                &outcome,
                "LogC4_to_Rec709.cube",
                Some(LookInput::LogC4),
                LookFormat::Cube
            )
            .is_none(),
            "an answer to the question does not make it a look"
        );
    }

    #[test]
    fn same_content_is_idempotent() {
        let mut m = Manifest::default();
        let spec = arri_spec(LookInput::LogC4, "Film");
        assert!(matches!(put(&mut m, &spec, &sha(1)), Slot::Add { .. }));
        assert!(matches!(put(&mut m, &spec, &sha(1)), Slot::Existing { .. }));
        assert_eq!(m.looks.len(), 1);
    }

    #[test]
    fn arri_files_are_replaced_in_place_when_they_change() {
        let mut m = Manifest::default();
        let spec = arri_spec(LookInput::LogC4, "Film");
        put(&mut m, &spec, &sha(1));
        let slot = put(&mut m, &spec, &sha(2));
        assert!(matches!(slot, Slot::Replace { .. }));
        assert_eq!(m.looks.len(), 1);
        assert_eq!(m.looks[0].sha256, sha(2));
    }

    #[test]
    fn two_files_of_one_import_never_replace_each_other() {
        let mut m = Manifest::default();
        let spec = arri_spec(LookInput::LogC4, "Film");
        let mut touched = HashSet::new();
        // The first file takes the id; the second has the same id and path but other content.
        let first = assign(&m, &spec, &sha(1), &touched);
        let Slot::Add { id, file } = &first else {
            panic!("{first:?}")
        };
        touched.insert(id.clone());
        apply(
            &mut m,
            &EntryData {
                spec: &spec,
                id: id.clone(),
                file: file.clone(),
                sha256: sha(1),
                lut_size: 33,
                imported: "t",
                url: None,
                replaces: None,
            },
        );
        let second = assign(&m, &spec, &sha(2), &touched);
        assert!(
            matches!(&second, Slot::Add { id, .. } if id.as_str() == "arri:3000:film-2"),
            "{second:?}"
        );
        // Without the memory of this import it *would* be taken for a re-release of the first.
        let across_imports = assign(&m, &spec, &sha(2), &HashSet::new());
        assert!(matches!(across_imports, Slot::Replace { .. }));
    }

    #[test]
    fn user_files_are_never_overwritten() {
        let mut m = Manifest::default();
        let spec = user_spec("look");
        put(&mut m, &spec, &sha(1));
        let second = put(&mut m, &spec, &sha(2));
        let third = put(&mut m, &spec, &sha(3));
        assert!(
            matches!(&second, Slot::Add { id, file } if id.as_str() == "user:look-2" && file == "user/look-2.cube")
        );
        assert!(matches!(&third, Slot::Add { id, .. } if id.as_str() == "user:look-3"));
        // The same content again finds its numbered entry.
        assert!(
            matches!(put(&mut m, &spec, &sha(2)), Slot::Existing { id, .. } if id.as_str() == "user:look-2")
        );
        assert_eq!(m.looks.len(), 3);
    }

    fn look_file_spec(format: LookFormat, name: &str) -> Spec {
        let mut spec = arri_spec(LookInput::LogC4, name);
        spec.format = format;
        spec.stem = format!("LL-3010-{name}");
        spec
    }

    #[test]
    fn own_look_files_are_users_looks_with_the_input_of_the_file() {
        for (format, input) in [
            (LookFormat::Alf4, LookInput::LogC4),
            (LookFormat::Alf4c, LookInput::LogC4),
            (LookFormat::Aml, LookInput::LogC3ToRec709),
        ] {
            let spec = Spec::from_outcome(
                &Outcome::OwnLookFile {
                    input,
                    display_transforms_ignored: false,
                },
                &format!("Mine/Warm Pull.{}", format.extension()),
                None,
                format,
            )
            .unwrap();
            assert_eq!(spec.id.as_str(), "user:warm-pull");
            assert_eq!(spec.name, "Warm Pull");
            assert_eq!(spec.folder, "user");
            assert_eq!(spec.format, format);
            assert!(!spec.arri);
            assert!(spec.group == ManifestGroup::User && spec.input == input);
            // The file keeps its kind's extension and is never overwritten by another.
            let mut m = Manifest::default();
            let first = put(&mut m, &spec, &sha(1));
            assert!(
                matches!(&first, Slot::Add { file, .. } if *file == format!("user/Warm Pull.{}", format.extension()))
            );
            let second = put(&mut m, &spec, &sha(2));
            assert!(matches!(&second, Slot::Add { id, .. } if id.as_str() == "user:warm-pull-2"));
        }
    }

    #[test]
    fn a_cube_beats_the_look_file_of_the_same_look_whichever_came_first() {
        let cube = arri_spec(LookInput::LogC4, "Film");
        let alf = look_file_spec(LookFormat::Alf4, "Film");

        // The `.cube` is there: the look file is left out.
        let mut m = Manifest::default();
        put(&mut m, &cube, &sha(1));
        let slot = assign(&m, &alf, &sha(2), &HashSet::new());
        assert!(
            matches!(&slot, Slot::Superseded { id, file, same_import: false, .. } if id.as_str() == "arri:3000:film" && file == "arri/3000-film/Film.cube"),
            "{slot:?}"
        );

        // The look file is there: the `.cube` takes its place and names the file that goes.
        let mut m = Manifest::default();
        put(&mut m, &alf, &sha(2));
        assert_eq!(m.looks[0].file, "arri/3000-film/LL-3010-Film.alf4");
        let slot = assign(&m, &cube, &sha(1), &HashSet::new());
        assert!(
            matches!(&slot, Slot::Upgrade { id, file, old_file }
                if id.as_str() == "arri:3000:film"
                    && file == "arri/3000-film/Film.cube"
                    && old_file == "arri/3000-film/LL-3010-Film.alf4"),
            "{slot:?}"
        );
        // Applying it moves the entry: one look, now a cube.
        let Slot::Upgrade { id, file, .. } = slot else {
            unreachable!()
        };
        apply(
            &mut m,
            &EntryData {
                spec: &cube,
                id,
                file,
                sha256: sha(1),
                lut_size: 65,
                imported: "later",
                url: None,
                replaces: None,
            },
        );
        assert_eq!(m.looks.len(), 1);
        assert_eq!(m.looks[0].format, LookFormat::Cube);
        assert_eq!(m.looks[0].file, "arri/3000-film/Film.cube");
        assert_eq!(m.looks[0].lut_size, 65);

        // The same look file again is the same entry; a look file never replaces a look file of
        // another kind (a `.cube` is the only one that outranks).
        let mut m = Manifest::default();
        put(&mut m, &alf, &sha(2));
        assert!(matches!(put(&mut m, &alf, &sha(2)), Slot::Existing { .. }));
        let aml = look_file_spec(LookFormat::Aml, "Film");
        let slot = assign(&m, &aml, &sha(3), &HashSet::new());
        assert!(
            matches!(&slot, Slot::Add { id, .. } if id.as_str() == "arri:3000:film-2"),
            "{slot:?}"
        );

        // Within one import the first of two files with the same id keeps it.
        let mut m = Manifest::default();
        put(&mut m, &alf, &sha(2));
        let mut touched = HashSet::new();
        touched.insert(alf.id.clone());
        let slot = assign(&m, &cube, &sha(1), &touched);
        assert!(
            matches!(&slot, Slot::Add { id, .. } if id.as_str() == "arri:3000:film-2"),
            "{slot:?}"
        );

        // A user's look of the same name is nobody's "same look": it gets a number.
        let mut m = Manifest::default();
        put(&mut m, &cube, &sha(1));
        let own = Spec::from_outcome(
            &Outcome::OwnLookFile {
                input: LookInput::LogC4,
                display_transforms_ignored: false,
            },
            "Film.alf4",
            None,
            LookFormat::Alf4,
        )
        .unwrap();
        assert!(
            matches!(assign(&m, &own, &sha(9), &HashSet::new()), Slot::Add { id, .. } if id.as_str() == "user:film")
        );
    }

    #[test]
    fn a_taken_path_stops_a_cube_taking_the_place_of_a_look_file() {
        // Another entry already sits where the `.cube` would go: the look file stays, the `.cube`
        // gets a number rather than overwriting that file.
        let cube = arri_spec(LookInput::LogC4, "Film");
        let alf = look_file_spec(LookFormat::Alf4, "Film");
        let mut m = Manifest::default();
        put(&mut m, &alf, &sha(2));
        let blocker = Spec {
            id: LookId::user("film"),
            stem: "Film".to_string(),
            folder: "arri/3000-film".to_string(),
            ..user_spec("film")
        };
        put(&mut m, &blocker, &sha(7));
        assert_eq!(m.looks.len(), 2);
        let slot = assign(&m, &cube, &sha(1), &HashSet::new());
        assert!(
            matches!(slot, Slot::Add { ref id, .. } if id.as_str() == "arri:3000:film-2"),
            "{slot:?}"
        );
    }

    #[test]
    fn the_three_look_library_packages_do_not_collide() {
        let mut m = Manifest::default();
        for (i, input) in [
            LookInput::LogC4,
            LookInput::LogC3Ei800,
            LookInput::LogC3ToRec709,
        ]
        .into_iter()
        .enumerate()
        {
            let spec = arri_spec(input, "Film");
            assert!(matches!(
                put(&mut m, &spec, &sha(i as u8 + 1)),
                Slot::Add { .. }
            ));
        }
        let files: Vec<_> = m.looks.iter().map(|l| l.file.clone()).collect();
        let mut unique = files.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(files.len(), unique.len(), "{files:?}");
    }

    #[test]
    fn a_taken_path_forces_a_number_even_with_a_new_id() {
        let mut m = Manifest::default();
        // A user LUT whose stem happens to equal another's path in a different case.
        put(&mut m, &user_spec("Look"), &sha(1));
        let other = Spec {
            id: LookId::user("look-b"),
            stem: "look".to_string(),
            ..user_spec("look")
        };
        let slot = assign(&m, &other, &sha(2), &HashSet::new());
        assert!(
            matches!(slot, Slot::Add { ref file, .. } if file == "user/look-2.cube"),
            "{slot:?}"
        );
    }

    #[test]
    fn user_input_answer_updates_an_existing_entry() {
        let mut m = Manifest::default();
        let spec = user_spec("look");
        put(&mut m, &spec, &sha(1));
        let mut again = spec.clone();
        again.input = LookInput::LogC3Ei800;
        let slot = assign(&m, &again, &sha(1), &HashSet::new());
        assert!(matches!(slot, Slot::Existing { .. }));
        let (id, file) = match slot {
            Slot::Existing { id, file } => (id, file),
            _ => unreachable!(),
        };
        let changed = apply(
            &mut m,
            &EntryData {
                spec: &again,
                id,
                file,
                sha256: sha(1),
                lut_size: 33,
                imported: "later",
                url: None,
                replaces: None,
            },
        );
        assert!(changed);
        assert_eq!(m.looks[0].input, LookInput::LogC3Ei800);
        assert_eq!(
            m.looks[0].source.as_ref().unwrap().imported,
            "2026-01-01T00:00:00Z"
        );
    }

    fn with_input(spec: &Spec, input: LookInput) -> Spec {
        let mut spec = spec.clone();
        spec.input = input;
        spec
    }

    #[test]
    fn a_renamed_copy_of_an_own_lut_is_the_same_entry() {
        let mut m = Manifest::default();
        put(&mut m, &user_spec("mine"), &sha(1));
        let slot = assign(&m, &user_spec("renamed"), &sha(1), &HashSet::new());
        assert!(
            matches!(&slot, Slot::Same { id, file, name }
                if id.as_str() == "user:mine" && file == "user/mine.cube" && name == "mine"),
            "{slot:?}"
        );
        // Other content under the new name is another LUT, and so is the same content for
        // another input.
        assert!(
            matches!(assign(&m, &user_spec("renamed"), &sha(2), &HashSet::new()), Slot::Add { id, .. } if id.as_str() == "user:renamed")
        );
        let other_input = with_input(&user_spec("renamed"), LookInput::LogC3Ei800);
        assert!(
            matches!(assign(&m, &other_input, &sha(1), &HashSet::new()), Slot::Add { id, .. } if id.as_str() == "user:renamed")
        );
        // The same name and the same bytes is still the entry, with the new answer.
        assert!(matches!(
            assign(
                &m,
                &with_input(&user_spec("mine"), LookInput::LogC3Ei800),
                &sha(1),
                &HashSet::new()
            ),
            Slot::Existing { .. }
        ));
        // Two copies in one import: the first is added, the second is the first.
        let mut touched = HashSet::new();
        touched.insert(LookId::user("mine"));
        let slot = assign(&m, &user_spec("copy"), &sha(1), &touched);
        assert!(matches!(slot, Slot::Same { .. }), "{slot:?}");
    }

    #[test]
    fn a_loose_file_that_is_one_of_arris_looks_is_that_look() {
        let mut m = Manifest::default();
        put(&mut m, &arri_spec(LookInput::LogC4, "Film"), &sha(1));
        let slot = assign(&m, &user_spec("my film"), &sha(1), &HashSet::new());
        assert!(
            matches!(&slot, Slot::Same { id, name, .. } if id.as_str() == "arri:3000:film" && name == "Film"),
            "{slot:?}"
        );
    }

    #[test]
    fn an_arri_package_claims_an_own_lut_and_there_is_one_entry() {
        let mut m = Manifest::default();
        put(&mut m, &user_spec("mine"), &sha(1));
        put(&mut m, &user_spec("other"), &sha(2));
        let film = arri_spec(LookInput::LogC4, "Film");
        let slot = assign(&m, &film, &sha(1), &HashSet::new());
        let Slot::Claim { id, file, old } = &slot else {
            panic!("{slot:?}")
        };
        assert_eq!(id.as_str(), "arri:3000:film");
        assert_eq!(file, "arri/3000-film/Film.cube");
        assert_eq!(
            (old.id.as_str(), old.file.as_str()),
            ("user:mine", "user/mine.cube")
        );
        apply(
            &mut m,
            &EntryData {
                spec: &film,
                id: id.clone(),
                file: file.clone(),
                sha256: sha(1),
                lut_size: 65,
                imported: "later",
                url: Some("https://example.invalid/"),
                replaces: Some(&old.id),
            },
        );
        assert_eq!(m.looks.len(), 2);
        let moved = m.look(&LookId::arri(3000, "film")).unwrap();
        assert!(matches!(&moved.group, ManifestGroup::Arri { no: 3000, .. }));
        assert_eq!(moved.name, "Film");
        assert_eq!(moved.file, "arri/3000-film/Film.cube");
        assert_eq!(moved.source.as_ref().unwrap().package.as_deref(), Some("p"));
        assert!(m.look(&LookId::user("mine")).is_none());
        assert!(m.look(&LookId::user("other")).is_some());

        // Only for the same input: the same bytes read as another encoding are not that look.
        let mut m = Manifest::default();
        put(
            &mut m,
            &with_input(&user_spec("mine"), LookInput::LogC3Ei800),
            &sha(1),
        );
        assert!(matches!(
            assign(&m, &film, &sha(1), &HashSet::new()),
            Slot::Add { .. }
        ));
    }

    #[test]
    fn arri_files_are_not_merged_by_their_bytes_within_an_import() {
        // Two looks of one package that hold the same LUT are two looks: ARRI's names say so.
        let mut m = Manifest::default();
        let alpha = arri_spec(LookInput::LogC4, "Alpha");
        let beta = arri_spec(LookInput::LogC4, "Beta");
        put(&mut m, &alpha, &sha(1));
        let mut touched = HashSet::new();
        touched.insert(alpha.id.clone());
        let slot = assign(&m, &beta, &sha(1), &touched);
        assert!(
            matches!(&slot, Slot::Add { id, .. } if id.as_str() == "arri:3000:beta"),
            "{slot:?}"
        );
        // A later import of a file under another ARRI name with the same bytes is that look.
        let slot = assign(&m, &beta, &sha(1), &HashSet::new());
        assert!(
            matches!(&slot, Slot::Same { id, .. } if id.as_str() == "arri:3000:alpha"),
            "{slot:?}"
        );
    }

    #[test]
    fn a_look_file_meets_the_cube_of_its_own_import_first() {
        let cube = arri_spec(LookInput::LogC4, "Film");
        let alf = look_file_spec(LookFormat::Alf4, "Film");
        let mut m = Manifest::default();
        put(&mut m, &cube, &sha(1));
        // The `.cube` was settled by this import: the look file is left out, and says so.
        let mut touched = HashSet::new();
        touched.insert(cube.id.clone());
        let slot = assign(&m, &alf, &sha(2), &touched);
        assert!(
            matches!(
                &slot,
                Slot::Superseded {
                    same_import: true,
                    ..
                }
            ),
            "{slot:?}"
        );
        // The same look file after an earlier import's `.cube` is the other kind of note.
        let slot = assign(&m, &alf, &sha(2), &HashSet::new());
        assert!(
            matches!(
                &slot,
                Slot::Superseded {
                    same_import: false,
                    ..
                }
            ),
            "{slot:?}"
        );
        // A `.cube` still never replaces an entry that this import settled.
        let mut m = Manifest::default();
        put(&mut m, &alf, &sha(2));
        let mut touched = HashSet::new();
        touched.insert(alf.id.clone());
        let slot = assign(&m, &cube, &sha(1), &touched);
        assert!(matches!(slot, Slot::Add { .. }), "{slot:?}");
    }
}
