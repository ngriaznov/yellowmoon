//! The lens tables of the NEX-5N and NEX-7, which only the enciphered SR2 sub-IFD carries
//! (`docs/04-raw-decoding.md` section Lens tags).
//!
//! These tests read real files from `testdata/real/files` (`tools/fetch/real-eval.sh`, local evaluation
//! only: the files are never in the repository or in CI). Without them a test says SKIPPED and passes,
//! unless `YM_REQUIRE_REAL_FILES` is set.

use std::path::{Path, PathBuf};

use ym_raw::metadata::{LensTable, LensTags, probe};
use ym_raw::sr2;
use ym_raw::tiff::{FileSource, Tiff};

fn real_file(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/real/files")
        .join(name);
    if path.is_file() {
        return Some(path);
    }
    assert!(
        std::env::var_os("YM_REQUIRE_REAL_FILES").is_none(),
        "YM_REQUIRE_REAL_FILES is set but {} is missing",
        path.display()
    );
    eprintln!("SKIPPED: {} is not on this machine", path.display());
    None
}

fn raw(t: &Option<LensTable>) -> &[i16] {
    &t.as_ref().expect("table").raw
}

#[test]
fn the_nex_7_lens_tables_are_read_from_the_sr2_sub_ifd() {
    let Some(path) = real_file("NEX-7.ARW") else {
        return;
    };
    let lens = probe(&path)
        .expect("probe")
        .lens_corrections
        .expect("lens tables");
    // 11 knots (APS-C), the count first, the rest of the 17 or 33 values zero padding.
    assert_eq!(raw(&lens.vignetting).len(), 17);
    assert_eq!(
        &raw(&lens.vignetting)[..12],
        [
            11, 0, 96, 288, 608, 1088, 1664, 2432, 3392, 4576, 5984, 7680
        ]
    );
    assert_eq!(raw(&lens.chromatic_aberration).len(), 33);
    assert_eq!(
        &raw(&lens.chromatic_aberration)[..6],
        [22, 1316, 1340, 1340, 1316, 1272]
    );
    assert_eq!(
        &raw(&lens.chromatic_aberration)[12..16],
        [-544, -496, -448, -368]
    );
    assert_eq!(
        &raw(&lens.distortion)[..12],
        [11, 16, 5, 0, 0, 0, 5, 16, 38, 74, 128, 205]
    );
}

#[test]
fn the_nex_5n_lens_tables_are_read_from_the_sr2_sub_ifd() {
    let Some(path) = real_file("NEX-5N.ARW") else {
        return;
    };
    let lens = probe(&path)
        .expect("probe")
        .lens_corrections
        .expect("lens tables");
    assert_eq!(
        &raw(&lens.vignetting)[..12],
        [
            11, 0, 96, 320, 672, 1120, 1728, 2432, 3200, 4096, 5184, 6496
        ]
    );
    assert_eq!(
        &raw(&lens.chromatic_aberration)[..6],
        [22, 1268, 1264, 1248, 1220, 1176]
    );
    assert_eq!(
        &raw(&lens.distortion)[..12],
        [11, 16, 5, 0, 0, 0, 5, 16, 38, 72, 126, 203]
    );
}

/// The SR2 tags 0x797D, 0x7980 and 0x7982 are the same tables as the raw sub-IFD's 0x7032, 0x7035 and
/// 0x7037: on a body that has both (the A7 III here) the two copies are equal, so the tag numbers and the
/// layout are checked against data that was read before.
#[test]
fn the_sr2_copy_equals_the_raw_sub_ifd_copy_where_a_file_has_both() {
    let Some(path) = real_file("ILCE-7M3.ARW") else {
        return;
    };
    let from_raw: LensTags = probe(&path)
        .expect("probe")
        .lens_corrections
        .expect("raw copy");
    let source = FileSource::open(&path).expect("open");
    let (tiff, first) = Tiff::open(&source).expect("tiff");
    let ifd0 = tiff.read_ifd(first).expect("ifd0");
    let from_sr2 = sr2::read(&tiff, &ifd0)
        .expect("sr2")
        .lens
        .expect("sr2 copy");
    assert_eq!(raw(&from_raw.vignetting)[0], 16);
    assert_eq!(from_sr2, from_raw);
}

/// A body whose files carry no tables anywhere still reports none.
#[test]
fn a_file_without_lens_tables_still_has_none() {
    let Some(path) = real_file("DSLR-A300.ARW") else {
        return;
    };
    assert!(probe(&path).expect("probe").lens_corrections.is_none());
}
