//! Which library each imported look is listed under (`docs/08-looks.md` section Look model,
//! `Look::library`): ARRI's five look downloads each have their own, the person's own files are in
//! one, whatever they are.
//!
//! The packages are made up: ARRI-like names, our own synthetic LUTs (`support`).

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use support::{
    ALF_LUT_SIZE, Kind, alf4_bytes, alf4c_bytes, aml_bytes, cube_text, no_progress, open_library,
    small_db, write_alf4_package_zip, write_aml_package_zip, write_package_zip,
};
use tempfile::TempDir;
use ym_color::look::LookInput;
use ym_looks::import::{ImportOptions, import_path};
use ym_looks::library::{Library, LookFormat, LookGroup, LookLibrary};
use ym_looks::recognize::PackageDb;

/// Looks per package; the LogC3 to Rec.709 look-file library has three strengths of each (`3 * N` files).
const N: usize = 6;

fn import(library: &mut Library, db: &PackageDb, path: &Path, input: Option<LookInput>) {
    let options = ImportOptions {
        input,
        imported_at: Some("2026-10-16T09:00:00Z".to_string()),
        ..ImportOptions::default()
    };
    import_path(library, db, path, &options, &mut no_progress())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// A kind of file and the size of its table: what a look is kept as.
type Kept = (Option<LookFormat>, Option<usize>);

/// How many looks a library holds, and the kinds of file and table sizes they come as.
type Census = BTreeMap<LookLibrary, (usize, Vec<Kept>)>;

fn census(library: &Library) -> Census {
    let mut out = Census::new();
    for look in library.looks() {
        let entry = out.entry(look.library()).or_default();
        entry.0 += 1;
        let kind = (look.format(), look.lut_size());
        if !entry.1.contains(&kind) {
            entry.1.push(kind);
        }
    }
    out
}

#[test]
fn every_arri_download_lands_in_its_own_library() {
    let dir = TempDir::new().expect("dir");
    let db = small_db(N);
    let mut library = open_library(&dir.path().join("home"));
    for kind in [Kind::LookLogC4, Kind::LookLogC3, Kind::LookLogC3Rec709] {
        let zip = write_package_zip(dir.path(), kind, N);
        import(&mut library, &db, &zip, None);
    }
    // ARRI's LogC3 to Rec.709 look files come in a zip of their own: three strengths of each look.
    let aml = write_aml_package_zip(dir.path(), N);
    import(&mut library, &db, &aml, None);

    let census = census(&library);
    // Neutral is always there; the imports are in ARRI's four libraries and nothing is in Your LUTs.
    assert_eq!(
        census.get(&LookLibrary::BuiltIn).map(|c| c.0),
        Some(1),
        "{census:?}"
    );
    assert_eq!(
        census.get(&LookLibrary::ArriLogC4).map(|c| c.0),
        Some(N),
        "{census:?}"
    );
    assert_eq!(
        census.get(&LookLibrary::ArriLogC3).map(|c| c.0),
        Some(N),
        "{census:?}"
    );
    assert_eq!(
        census.get(&LookLibrary::ArriLogC3ToRec709).map(|c| c.0),
        Some(N),
        "{census:?}"
    );
    assert_eq!(
        census.get(&LookLibrary::ArriAlf2).map(|c| c.0),
        Some(3 * N),
        "{census:?}"
    );
    assert_eq!(census.get(&LookLibrary::User), None, "{census:?}");

    // The kind of file and the size of the table come with the look.
    assert_eq!(
        census[&LookLibrary::ArriLogC4].1,
        vec![(Some(LookFormat::Cube), Some(Kind::LookLogC4.lut_size()))]
    );
    assert_eq!(
        census[&LookLibrary::ArriLogC3].1,
        vec![(Some(LookFormat::Cube), Some(Kind::LookLogC3.lut_size()))]
    );
    assert_eq!(
        census[&LookLibrary::ArriLogC3ToRec709].1,
        vec![(
            Some(LookFormat::Cube),
            Some(Kind::LookLogC3Rec709.lut_size())
        )]
    );
    assert_eq!(
        census[&LookLibrary::ArriAlf2].1,
        vec![(Some(LookFormat::Aml), Some(ALF_LUT_SIZE))]
    );

    // Each library's looks have the input its section is named for.
    for look in library.looks() {
        let expected = match look.library() {
            LookLibrary::ArriLogC4 => LookInput::LogC4,
            LookLibrary::ArriLogC3 => LookInput::LogC3Ei800,
            LookLibrary::ArriLogC3ToRec709 | LookLibrary::ArriAlf2 => LookInput::LogC3ToRec709,
            LookLibrary::BuiltIn | LookLibrary::User => continue,
        };
        assert_eq!(look.input, expected, "{}", look.id);
        assert!(matches!(look.group, LookGroup::Arri { .. }), "{}", look.id);
    }
}

#[test]
fn the_logc4_look_files_are_in_the_logc4_library_with_the_cube_ids() {
    let dir = TempDir::new().expect("dir");
    let db = small_db(N);

    // Alone, the look files are the LogC4 library, as 33-point look files.
    let mut alone = open_library(&dir.path().join("alone"));
    let alf4 = write_alf4_package_zip(dir.path(), N);
    import(&mut alone, &db, &alf4, None);
    let census_alone = census(&alone);
    assert_eq!(
        census_alone[&LookLibrary::ArriLogC4].0,
        N,
        "{census_alone:?}"
    );
    assert_eq!(
        census_alone[&LookLibrary::ArriLogC4].1,
        vec![(Some(LookFormat::Alf4), Some(ALF_LUT_SIZE))]
    );
    // Neutral, and nothing else.
    assert_eq!(
        census_alone.len(),
        2,
        "nothing else is imported: {census_alone:?}"
    );

    // After the cubes of the same library, the same looks stay one entry each, as cubes.
    let mut both = open_library(&dir.path().join("both"));
    let cubes = write_package_zip(dir.path(), Kind::LookLogC4, N);
    import(&mut both, &db, &cubes, None);
    import(&mut both, &db, &alf4, None);
    let census_both = census(&both);
    assert_eq!(census_both[&LookLibrary::ArriLogC4].0, N, "{census_both:?}");
    assert_eq!(
        census_both[&LookLibrary::ArriLogC4].1,
        vec![(Some(LookFormat::Cube), Some(Kind::LookLogC4.lut_size()))],
        "the cube wins"
    );
}

#[test]
fn the_persons_own_files_are_all_in_your_luts_whatever_they_are() {
    let dir = TempDir::new().expect("dir");
    let db = small_db(N);
    let mut library = open_library(&dir.path().join("home"));
    let own = dir.path().join("own");
    std::fs::create_dir_all(&own).expect("folder");
    // (file name, bytes, the answer to the input question, expected format, expected input)
    type Own = (
        &'static str,
        Vec<u8>,
        Option<LookInput>,
        LookFormat,
        LookInput,
    );
    let files: Vec<Own> = vec![
        (
            "Warm.cube",
            cube_text(5, 1, "Warm").into_bytes(),
            Some(LookInput::LogC3Ei800),
            LookFormat::Cube,
            LookInput::LogC3Ei800,
        ),
        (
            "Pull.alf4",
            alf4_bytes(3, 2, None),
            None,
            LookFormat::Alf4,
            LookInput::LogC4,
        ),
        (
            "Custom.alf4c",
            alf4c_bytes(3, 3, None),
            None,
            LookFormat::Alf4c,
            LookInput::LogC4,
        ),
        (
            "Night.aml",
            aml_bytes(3, 4, None),
            None,
            LookFormat::Aml,
            LookInput::LogC3ToRec709,
        ),
    ];
    for (name, bytes, input, _, _) in &files {
        let path = own.join(name);
        std::fs::write(&path, bytes).expect("write");
        import(&mut library, &db, &path, *input);
    }
    let looks: Vec<_> = library
        .looks()
        .iter()
        .filter(|l| l.library() == LookLibrary::User)
        .collect();
    assert_eq!(looks.len(), 4, "{:?}", census(&library));
    for (name, _, _, format, input) in &files {
        let stem = name.split('.').next().expect("stem").to_lowercase();
        let look = looks
            .iter()
            .find(|l| l.id.slug() == stem)
            .unwrap_or_else(|| panic!("{stem} is in Your LUTs"));
        assert_eq!(look.library(), LookLibrary::User);
        assert_eq!(look.format(), Some(*format), "{name}");
        assert_eq!(look.input, *input, "{name}");
        assert!(matches!(look.group, LookGroup::User), "{name}");
    }
    // A user's `.aml` of LogC3 to Rec.709 is not in ARRI's ALF2 library: that is ARRI's own.
    assert!(!census(&library).contains_key(&LookLibrary::ArriAlf2));
}

#[test]
fn the_built_in_looks_are_in_the_built_in_library_and_have_no_file() {
    let dir = TempDir::new().expect("dir");
    let builtin = dir.path().join("builtin");
    std::fs::create_dir_all(&builtin).expect("folder");
    let library = Library::open(dir.path().join("home"), Some(builtin.as_path())).expect("open");
    let built_in: Vec<_> = library
        .looks()
        .iter()
        .filter(|l| l.library() == LookLibrary::BuiltIn)
        .collect();
    assert!(!built_in.is_empty(), "Neutral is always there");
    for look in built_in {
        assert_eq!(
            (look.format(), look.lut_size()),
            (None, None),
            "{}",
            look.id
        );
        assert!(matches!(look.group, LookGroup::BuiltIn));
    }
}
