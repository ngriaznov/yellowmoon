//! The closed catalogue of file messages (`docs/26` P1-2, P1-3; `docs/04` Errors).
//!
//! A deterministic sweep of damaged files (truncations at many fractions, header and garbage
//! overwrites, zeros, bit flips, patched size tags, a text file, a JPEG, a zero-byte file) of every
//! decodable variant is decoded from memory and from disk. Every error's `Display` must be one of
//! the catalogue sentences and must not contain the words of the technical detail.

use std::collections::BTreeSet;

use ym_raw::decode::{DecodeOptions, decode, decode_bytes};
use ym_raw::error::{
    MESSAGE_CUT_SHORT, MESSAGE_DAMAGED, MESSAGE_DENIED, MESSAGE_EMPTY, MESSAGE_MACOS_BLOCKED,
    MESSAGE_NOT_FOUND, MESSAGE_UNREADABLE, RawError,
};
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::damage::{foreign, sweep, sweep_damages, valid_files};

/// Every sentence a damaged or unreadable file may show, besides the fixed ones of the other
/// variants (`NotArw`, `Arq`, `Unsupported`).
const CATALOGUE: [&str; 7] = [
    MESSAGE_EMPTY,
    MESSAGE_CUT_SHORT,
    MESSAGE_DAMAGED,
    MESSAGE_NOT_FOUND,
    MESSAGE_DENIED,
    MESSAGE_MACOS_BLOCKED,
    MESSAGE_UNREADABLE,
];

const NOT_ARW: &str = "This isn't an ARW file. Yellowmoon opens Sony ARW files only.";
const ARQ: &str = "Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead.";

const BANNED: [&str; 8] = [
    "((",
    "os error",
    "Huffman",
    "entropy",
    "restart",
    "lossless JPEG",
    "ends before the image",
    "runs past",
];

/// Checks one error against the closed list and the banned words; returns its sentence.
fn check(what: &str, error: &RawError) -> String {
    let text = error.to_string();
    for banned in BANNED {
        assert!(
            !text.contains(banned),
            "{what}: {text:?} contains {banned:?}"
        );
    }
    let known = CATALOGUE.contains(&text.as_str())
        || text == NOT_ARW
        || text == ARQ
        || (matches!(error, RawError::Unsupported { .. })
            && (text.contains("aren't supported yet")
                || text.starts_with("Yellowmoon can't open ")));
    assert!(known, "{what}: {text:?} is not in the catalogue");
    match error {
        RawError::Corrupt { detail } => {
            assert!(!detail.is_empty(), "{what}: empty detail");
            assert!(
                [MESSAGE_EMPTY, MESSAGE_CUT_SHORT, MESSAGE_DAMAGED].contains(&text.as_str()),
                "{what}: {text:?}"
            );
        }
        RawError::Io(_) => assert!(error.detail().is_some(), "{what}"),
        _ => {}
    }
    text
}

#[test]
fn every_message_of_a_damaged_file_is_in_the_catalogue() {
    for (name, valid) in valid_files() {
        decode_bytes(&valid, &DecodeOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    // The sweep is shared with the damaged-folder tests (`ym_raw::testgen::damage`).
    let files = sweep();
    assert!(files.len() >= 1576, "{} files in the sweep", files.len());

    let mut seen = BTreeSet::new();
    let mut ok = 0;
    for (what, bytes) in &files {
        match decode_bytes(bytes, &DecodeOptions::default()) {
            Ok(_) => ok += 1,
            Err(error) => {
                seen.insert(check(what, &error));
            }
        }
        if let Err(error) = probe_bytes(bytes) {
            check(what, &error);
        }
    }
    // The sweep reaches all three sentences of a damaged file, and decodes some files that only
    // have damaged pixel values.
    for sentence in [MESSAGE_EMPTY, MESSAGE_CUT_SHORT, MESSAGE_DAMAGED] {
        assert!(
            seen.contains(sentence),
            "the sweep never produced {sentence:?}"
        );
    }
    assert!(ok > 0, "some damage lies in pixel values only");
}

#[test]
fn the_same_sweep_from_disk_gives_the_same_sentences() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut files = foreign();
    let (name, valid) = valid_files().swap_remove(0);
    for damage in sweep_damages(valid.len()).into_iter().step_by(5) {
        files.push((format!("{name} / {}", damage.label()), damage.apply(&valid)));
    }
    for (i, (what, bytes)) in files.iter().enumerate() {
        let path = dir.path().join(format!("f{i}.ARW"));
        std::fs::write(&path, bytes).expect("write");
        let from_disk = decode(&path).map(|_| ()).map_err(|e| e.to_string());
        let from_memory = decode_bytes(bytes, &DecodeOptions::default())
            .map(|_| ())
            .map_err(|e| e.to_string());
        assert_eq!(from_disk, from_memory, "{what}");
        if let Err(error) = decode(&path) {
            check(what, &error);
        }
    }
}

#[test]
fn a_zero_byte_file_is_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.ARW");
    std::fs::write(&path, b"").expect("write");
    let error = decode(&path).expect_err("empty");
    assert_eq!(
        error.to_string(),
        "This file is empty. Copy it from the card again, or skip it."
    );
    assert_eq!(error.kind(), "corrupt");
    assert!(error.detail().is_some());
}

#[test]
fn a_missing_file_and_a_folder_have_io_sentences() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = decode(dir.path().join("gone.ARW")).expect_err("missing");
    assert!(matches!(missing, RawError::Io(_)));
    assert_eq!(
        check("missing", &missing),
        "Yellowmoon can't find this file any more. It may have been moved, renamed or its drive disconnected."
    );
    assert!(missing.detail().is_some_and(|d| d.contains("os error")));
    // A folder opens on Unix and fails on the first read; either way the sentence is in the list.
    if let Err(error) = decode(dir.path()) {
        check("folder", &error);
    }
}

#[cfg(unix)]
#[test]
fn a_file_without_read_permission_says_so() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("locked.ARW");
    std::fs::write(&path, b"x").expect("write");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    if std::fs::File::open(&path).is_ok() {
        return; // running as root: the permission bits do not apply
    }
    let error = decode(&path).expect_err("locked");
    assert_eq!(check("locked", &error), MESSAGE_DENIED);
}
