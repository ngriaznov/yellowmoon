//! Fuzzes the complete decoder entry point on arbitrary bytes: container, probe and every pixel
//! decoder. Seed with the files from `gen_seeds` (one per variant). Property: no panic, no
//! unbounded allocation; a successful decode returns a plane whose size matches the metadata, and
//! an error is one of the typed errors.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_raw::decode::{DecodeOptions, decode_bytes};

fuzz_target!(|data: &[u8]| {
    match decode_bytes(data, &DecodeOptions::default()) {
        Ok(image) => {
            assert_eq!(image.data.data.len(), image.data.width * image.data.height);
            assert_eq!(image.pixel_height(), image.meta.height as usize);
            assert_eq!(image.pixel_width(), image.meta.width as usize);
            if let Some(qstep) = &image.qstep {
                assert_eq!(qstep.data.len(), image.data.data.len());
            }
        }
        Err(error) => {
            // The message must render without panicking.
            let _ = error.to_string();
            let _ = error.kind();
        }
    }
});
