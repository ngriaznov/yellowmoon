//! Hostile and broken inputs: nothing may panic, hang or write anything.

mod support;

use std::fs;

use proptest::prelude::*;
use support::{cube_text, no_progress, small_db, tree, write_zip};
use ym_looks::import::{ImportError, inspect};

proptest! {
    // A fixed seed: the same cases on every run, so a failure is always reproducible.
    #![proptest_config(ProptestConfig {
        cases: 64,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x594d_4c4f_4f4b_5333),
        ..ProptestConfig::default()
    })]

    #[test]
    fn random_bytes_are_never_a_crash(bytes in proptest::collection::vec(any::<u8>(), 0..4000), as_zip in any::<bool>()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if as_zip { "x.zip" } else { "x.cube" });
        fs::write(&path, &bytes).unwrap();
        let result = inspect(&path, &small_db(1), None, &mut no_progress());
        match result {
            Ok(recognition) => {
                // A .cube file of random bytes is rejected, never accepted.
                prop_assert!(recognition.recognized_count() + recognition.needs_answer_count() == 0);
            }
            Err(e) => prop_assert!(matches!(e, ImportError::BadZip(_)), "{e}"),
        }
        prop_assert!(tree(dir.path()) == vec![path.file_name().unwrap().to_string_lossy().into_owned()]);
    }

    #[test]
    fn random_text_cubes_are_rejected_not_fatal(text in "[ -~\\n]{0,400}") {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.cube");
        fs::write(&path, text).unwrap();
        let recognition = inspect(&path, &small_db(1), None, &mut no_progress()).unwrap();
        prop_assert_eq!(recognition.files.len(), 1);
    }
}

#[test]
fn every_truncation_of_a_zip_is_handled() {
    let dir = tempfile::tempdir().unwrap();
    let full_path = dir.path().join("full.zip");
    write_zip(
        &full_path,
        &[
            (
                "3000 Film/a_LogC4.cube".to_string(),
                cube_text(2, 1, "a").into_bytes(),
            ),
            (
                "3000 Film/b_LogC4.cube".to_string(),
                cube_text(2, 2, "b").into_bytes(),
            ),
            ("notes.pdf".to_string(), b"pdf".to_vec()),
        ],
    );
    let full = fs::read(&full_path).unwrap();
    assert!(full.len() < 4000, "{}", full.len());
    let db = small_db(2);
    let cut_path = dir.path().join("cut.zip");
    let mut opened = 0;
    for len in 0..full.len() {
        fs::write(&cut_path, &full[..len]).unwrap();
        match inspect(&cut_path, &db, None, &mut no_progress()) {
            Ok(_) => opened += 1,
            Err(e) => assert!(matches!(e, ImportError::BadZip(_)), "{len}: {e}"),
        }
    }
    assert_eq!(
        opened, 0,
        "a zip without its central directory cannot be read"
    );
    fs::write(&cut_path, &full).unwrap();
    assert!(inspect(&cut_path, &db, None, &mut no_progress()).is_ok());
}

#[test]
fn a_flipped_byte_in_a_zip_is_an_error_or_a_rejected_file_never_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("z.zip");
    write_zip(
        &path,
        &[("a.cube".to_string(), cube_text(3, 1, "a").into_bytes())],
    );
    let original = fs::read(&path).unwrap();
    let probe = dir.path().join("flipped.zip");
    for i in 0..original.len() {
        let mut bytes = original.clone();
        bytes[i] ^= 0xff;
        fs::write(&probe, &bytes).unwrap();
        // Must return (Ok or Err) without panicking.
        let _ = inspect(&probe, &small_db(1), None, &mut no_progress());
    }
}
