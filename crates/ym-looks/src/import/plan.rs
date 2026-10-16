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
//!   that an earlier import put there, and a `.cube` replaces the look file of an earlier import;
//! * any other collision (a user LUT with the same name but different content, two names that
//!   slugify alike, a path already taken) gets a number: `film`, `film-2`, `film-3`, so no file of
//!   the user's is ever overwritten.

use std::collections::HashSet;

use ym_color::look::LookInput;

use crate::library::{
    DisplayInput, LookFormat, LookId, sanitize_file_stem, slugify, truncate_to_bytes,
};
use crate::manifest::{Manifest, ManifestDisplayLut, ManifestGroup, ManifestLook, Source};
use crate::recognize::Outcome;
use crate::recognize::names::{file_name, file_stem};

/// What kind of entry an incoming file becomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpecKind {
    /// An entry in `looks`.
    Look {
        group: ManifestGroup,
        input: LookInput,
    },
    /// An entry in `display_luts`.
    Display { input: DisplayInput, target: String },
}

/// What the recogniser (or the user's answer) says an incoming file is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spec {
    /// The id before any collision number.
    pub(crate) id: LookId,
    /// Display name.
    pub(crate) name: String,
    pub(crate) kind: SpecKind,
    /// Folder below `looks/`: `arri/3000-film`, `arri-display` or `user`.
    pub(crate) folder: String,
    /// File stem to write, already sanitised and including any encoding suffix.
    pub(crate) stem: String,
    /// The kind of file, which gives the extension.
    pub(crate) format: LookFormat,
    /// Package id, for ARRI files.
    pub(crate) package: Option<String>,
    /// Whether the file came from an ARRI package (updates replace in place).
    pub(crate) arri: bool,
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
                    kind: SpecKind::Look {
                        group: ManifestGroup::Arri {
                            no: *group_no,
                            name: group_name.clone(),
                        },
                        input: *input,
                    },
                    folder: format!("arri/{group_no:04}-{group_slug}"),
                    stem: format!("{}{suffix}", sanitize_file_stem(raw_stem)),
                    format,
                    package: Some(package.clone()),
                    arri: true,
                    original_name,
                })
            }
            Outcome::ArriDisplay {
                package,
                input,
                target,
                file_name: library_name,
                ..
            } => {
                let raw_stem = file_stem(library_name);
                let suffix = match input {
                    DisplayInput::LogC4 => "",
                    DisplayInput::LogC3 => "-logc3",
                };
                Some(Self {
                    id: LookId::arri_display(*input, target),
                    name: target.clone(),
                    kind: SpecKind::Display {
                        input: *input,
                        target: target.clone(),
                    },
                    folder: "arri-display".to_string(),
                    stem: format!("{}{suffix}", sanitize_file_stem(raw_stem)),
                    format,
                    package: Some(package.clone()),
                    arri: true,
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
                    kind: SpecKind::Look {
                        group: ManifestGroup::User,
                        input,
                    },
                    folder: "user".to_string(),
                    stem: sanitize_file_stem(raw_stem),
                    format,
                    package: None,
                    arri: false,
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
                    kind: SpecKind::Look {
                        group: ManifestGroup::User,
                        input: *input,
                    },
                    folder: "user".to_string(),
                    stem: sanitize_file_stem(raw_stem),
                    format,
                    package: None,
                    arri: false,
                    original_name,
                })
            }
            Outcome::Rejected { .. } => None,
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
    /// stays and nothing is written. `file` is the `.cube` entry's file.
    Superseded { id: LookId, file: String },
    /// The same ARRI look is imported as a look file and this is its `.cube`: the entry moves to
    /// `file` and the look file `old_file` goes.
    Upgrade {
        id: LookId,
        file: String,
        old_file: String,
    },
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
        .or_else(|| {
            manifest
                .display_lut(id)
                .map(|d| (d.sha256.as_str(), d.file.as_str(), LookFormat::Cube))
        })
}

fn file_taken(manifest: &Manifest, file: &str) -> bool {
    manifest
        .looks
        .iter()
        .map(|l| l.file.as_str())
        .chain(manifest.display_luts.iter().map(|d| d.file.as_str()))
        .any(|f| f.eq_ignore_ascii_case(file))
}

/// Most collision numbers tried before falling back to a hash-based name.
const MAX_COLLISION_NUMBER: u32 = 10_000;

/// Chooses the id and file for `spec` with content hash `sha256`, given the manifest as changed
/// so far in this import. `touched` holds the ids this import has already assigned (added,
/// replaced or matched): such an entry belongs to another file of the same import, so it is never
/// replaced, only matched when the content is the same.
pub(crate) fn assign(
    manifest: &Manifest,
    spec: &Spec,
    sha256: &str,
    touched: &HashSet<LookId>,
) -> Slot {
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
            // The same look in its other form: the `.cube` is the one to keep.
            Some((_, found, found_format))
                if spec.arri
                    && !touched.contains(&id)
                    && found_format != spec.format
                    && (found_format.is_cube() || spec.format.is_cube())
                    && (!spec.format.is_cube() || !file_taken(manifest, &file)) =>
            {
                return if spec.format.is_cube() {
                    Slot::Upgrade {
                        id,
                        file,
                        old_file: found.to_string(),
                    }
                } else {
                    Slot::Superseded {
                        id,
                        file: found.to_string(),
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
}

/// Inserts or replaces the entry in `manifest`. Returns true when something changed.
pub(crate) fn apply(manifest: &mut Manifest, data: &EntryData<'_>) -> bool {
    let source = Some(Source {
        package: data.spec.package.clone(),
        url: data.spec.package.as_ref().and(data.url).map(str::to_string),
        imported: data.imported.to_string(),
        original_name: data.spec.original_name.clone(),
    });
    match &data.spec.kind {
        SpecKind::Look { group, input } => {
            let entry = ManifestLook {
                id: data.id.clone(),
                name: data.spec.name.clone(),
                group: group.clone(),
                input: *input,
                file: data.file.clone(),
                format: data.spec.format,
                sha256: data.sha256.clone(),
                lut_size: data.lut_size,
                source,
            };
            match manifest.looks.iter_mut().find(|l| l.id == data.id) {
                Some(slot) => {
                    if slot.sha256 == entry.sha256 {
                        // The same file again keeps its import date and source, but takes a
                        // changed answer to the input question.
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
        SpecKind::Display { input, target } => {
            let entry = ManifestDisplayLut {
                id: data.id.clone(),
                input: *input,
                target: target.clone(),
                file: data.file.clone(),
                sha256: data.sha256.clone(),
            };
            match manifest.display_luts.iter_mut().find(|d| d.id == data.id) {
                Some(slot) => {
                    let changed = slot.sha256 != entry.sha256;
                    if changed {
                        *slot = entry;
                    }
                    changed
                }
                None => {
                    manifest.display_luts.push(entry);
                    true
                }
            }
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
            | Slot::Superseded { id, file }
            | Slot::Upgrade { id, file, .. } => (id.clone(), file.clone()),
        };
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
            },
        );
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
        again.kind = SpecKind::Look {
            group: ManifestGroup::User,
            input: LookInput::LogC3Ei800,
        };
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
            },
        );
        assert!(changed);
        assert_eq!(m.looks[0].input, LookInput::LogC3Ei800);
        assert_eq!(
            m.looks[0].source.as_ref().unwrap().imported,
            "2026-01-01T00:00:00Z"
        );
    }
}
