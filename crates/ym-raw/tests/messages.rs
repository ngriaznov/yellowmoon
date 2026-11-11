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
use ym_raw::layout::ChromaSubsampling;
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::{
    Arw1, ArwBuilder, CrawEncoder, LosslessL, LosslessMs, Packed12, SplitMix64, gradient_mosaic,
    gradient_rgb, tiny_jpeg,
};

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

/// The small valid files, one per decodable variant (the seeds of `decode_hostile.rs`).
fn seeds() -> Vec<(&'static str, Vec<u8>)> {
    let m = gradient_mosaic(96, 64, 512, 16000, 1);
    let m12 = gradient_mosaic(96, 64, 128, 4000, 2);
    let encoder = CrawEncoder::new();
    let mc = gradient_mosaic(96, 64, 512, encoder.max_value(), 3);
    let ms = |sub, tile: u32| {
        let rgb = gradient_rgb(70, 50, 1100, 9000, 4);
        ArwBuilder::new("ILCE-7RM5", 70, 50)
            .lossless_ms(LosslessMs::new(sub, rgb, 70, 50, [2.0, 1.0, 1.6]).tile(tile, tile))
            .build(&[])
    };
    vec![
        ("unc14", ArwBuilder::new("ILCE-7RM4", 96, 64).build(&m)),
        (
            "unc12",
            ArwBuilder::new("ILCE-7M2", 96, 64)
                .uncompressed(12)
                .build(&m12),
        ),
        (
            "packed12",
            ArwBuilder::new("DSLR-A900", 96, 64)
                .payload(Packed12)
                .build(&m12),
        ),
        (
            "arw1",
            ArwBuilder::new("DSLR-A300", 96, 64)
                .payload(Arw1)
                .black([0; 4])
                .white([3840; 3])
                .build(&m12),
        ),
        (
            "craw",
            ArwBuilder::new("ILCE-7M3", 96, 64).craw(encoder).build(&mc),
        ),
        (
            "lossless_l",
            ArwBuilder::new("ILCE-1", 96, 64)
                .payload(LosslessL::new().tile(32, 32))
                .build(&m),
        ),
        (
            "lossless_l_restarts",
            ArwBuilder::new("ILCE-1", 96, 64)
                .payload(LosslessL::new().tile(64, 64).restart_rows(2).predictor(4))
                .build(&m),
        ),
        ("ms420", ms(ChromaSubsampling::Yuv420, 32)),
        ("ms422", ms(ChromaSubsampling::Yuv422, 32)),
        ("ms444", ms(ChromaSubsampling::Yuv444, 64)),
    ]
}

/// `(entry offset, field type)` of every directory entry with `tag` in a little-endian TIFF,
/// following next-IFD links, SubIFDs and the EXIF pointer.
fn entries(file: &[u8], tag: u16) -> Vec<(usize, u16)> {
    fn u16at(f: &[u8], at: usize) -> Option<u16> {
        Some(u16::from_le_bytes(f.get(at..at + 2)?.try_into().ok()?))
    }
    fn u32at(f: &[u8], at: usize) -> Option<u32> {
        Some(u32::from_le_bytes(f.get(at..at + 4)?.try_into().ok()?))
    }
    let mut found = Vec::new();
    let mut seen = Vec::new();
    let mut pending = vec![u32at(file, 4).unwrap_or(0) as usize];
    while let Some(at) = pending.pop() {
        if at < 8 || seen.contains(&at) {
            continue;
        }
        seen.push(at);
        let Some(count) = u16at(file, at) else {
            continue;
        };
        for i in 0..usize::from(count) {
            let e = at + 2 + 12 * i;
            let (Some(t), Some(ty), Some(n)) =
                (u16at(file, e), u16at(file, e + 2), u32at(file, e + 4))
            else {
                break;
            };
            if t == tag {
                found.push((e, ty));
            }
            if (t == 0x014A || t == 0x8769) && n == 1 {
                pending.push(u32at(file, e + 8).unwrap_or(0) as usize);
            }
        }
        pending.push(u32at(file, at + 2 + 12 * usize::from(count)).unwrap_or(0) as usize);
    }
    found
}

/// Writes `value` into the first value slot of every entry with `tag` (SHORT or LONG).
fn patch(file: &mut [u8], tag: u16, value: u32) {
    for (e, ty) in entries(file, tag) {
        match ty {
            3 => file[e + 8..e + 10].copy_from_slice(&(value as u16).to_le_bytes()),
            4 => file[e + 8..e + 12].copy_from_slice(&value.to_le_bytes()),
            _ => {}
        }
    }
}

/// The damaged files built from one valid file: `(what, bytes)`.
fn damaged(valid: &[u8], rng: &mut SplitMix64) -> Vec<(String, Vec<u8>)> {
    let len = valid.len();
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    // Truncations at many fractions, and a few bytes short of the end.
    for k in 0..=24 {
        out.push((format!("cut {k}/24"), valid[..len * k / 24].to_vec()));
    }
    for k in [1, 7, 100, 500] {
        out.push((
            format!("{k} bytes short"),
            valid[..len.saturating_sub(k)].to_vec(),
        ));
    }
    // Header overwrites.
    for n in [1usize, 2, 4, 8, 16] {
        let mut f = valid.to_vec();
        for b in &mut f[..n] {
            *b = rng.next_u64() as u8;
        }
        out.push((format!("header garbage {n}"), f));
        let mut f = valid.to_vec();
        f[..n].fill(0);
        out.push((format!("header zeros {n}"), f));
    }
    // Garbage and zeros in the body.
    for k in 1..=6usize {
        let at = len * k / 7;
        let n = (len / 50).max(8).min(len - at);
        let mut f = valid.to_vec();
        for b in &mut f[at..at + n] {
            *b = rng.next_u64() as u8;
        }
        out.push((format!("garbage at {k}/7"), f));
        let mut f = valid.to_vec();
        f[at..at + n].fill(0);
        out.push((format!("zeros at {k}/7"), f));
    }
    // Bit flips.
    for round in 0..8 {
        let mut f = valid.to_vec();
        for _ in 0..=rng.below(4) {
            let limit = if round % 2 == 0 { len.min(2048) } else { len };
            let at = rng.below(limit as u64) as usize;
            f[at] ^= 1 << rng.below(8);
        }
        out.push((format!("bit flips {round}"), f));
    }
    // Tags that size buffers, set to extremes.
    for tag in [
        0x0100u16, 0x0101, 0x0102, 0x0103, 0x0115, 0x0116, 0x0117, 0x0111, 0x0142, 0x0143, 0x0144,
        0x0145, 0x0212, 0x7010,
    ] {
        for value in [0u32, 1, 3, 255, 65535, 0x7FFF_FFFF, u32::MAX] {
            let mut f = valid.to_vec();
            patch(&mut f, tag, value);
            out.push((format!("tag {tag:#06x} = {value}"), f));
        }
    }
    out
}

/// Files that are not ARW at all, and the empty file.
fn foreign() -> Vec<(String, Vec<u8>)> {
    let jpeg = tiny_jpeg(32, 24, 7);
    vec![
        ("empty".to_owned(), Vec::new()),
        ("one byte".to_owned(), vec![b'I']),
        ("seven bytes".to_owned(), b"II*\0\x08\0\0".to_vec()),
        (
            "text".to_owned(),
            b"This is not a photo. It is a note about a photo.\n".repeat(40),
        ),
        ("jpeg".to_owned(), jpeg),
        ("zeros".to_owned(), vec![0; 4096]),
    ]
}

#[test]
fn every_message_of_a_damaged_file_is_in_the_catalogue() {
    let mut rng = SplitMix64(0x00C0_FFEE);
    let mut files = foreign();
    for (name, valid) in seeds() {
        decode_bytes(&valid, &DecodeOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (what, bytes) in damaged(&valid, &mut rng) {
            files.push((format!("{name} / {what}"), bytes));
        }
    }
    assert!(files.len() >= 192, "{} files in the sweep", files.len());

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
    let mut rng = SplitMix64(0xFEED);
    let mut files = foreign();
    let (name, valid) = seeds().swap_remove(0);
    for (what, bytes) in damaged(&valid, &mut rng).into_iter().step_by(5) {
        files.push((format!("{name} / {what}"), bytes));
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
