//! Fuzzes the whole metadata probe (container, EXIF, maker note, SR2, raw IFD selection, storage
//! checks) and the preview reader. Seed the corpus with valid files from
//! `cargo run -p ym-raw --features testgen --example gen_seeds -- <dir>`.
//! Property: no panic, no unbounded allocation, for any bytes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_raw::metadata::{probe_bytes, read_preview_from};
use ym_raw::tiff::MemSource;

fuzz_target!(|data: &[u8]| {
    if let Ok(meta) = probe_bytes(data) {
        let _ = meta.layout();
        let _ = meta.arw_version_name();
        // A reported preview must be readable from the same bytes.
        if let Ok(Some(jpeg)) = read_preview_from(&MemSource(data), &meta) {
            assert!(jpeg.len() as u64 == meta.preview.map_or(0, |p| p.length));
        }
    }
});
