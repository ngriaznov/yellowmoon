//! Property tests for the probe: whatever the bytes, it returns a result and never panics, and
//! valid files survive being rebuilt. `cargo-fuzz` (crates/ym-raw/fuzz) runs the same property for
//! much longer; these cases are the quick, deterministic version for every `cargo test`.

use proptest::prelude::*;
use ym_raw::error::RawError;
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::{ArwBuilder, ContainerOnly, gradient_mosaic};

fn valid_file(variant: u8) -> Vec<u8> {
    let mosaic = gradient_mosaic(48, 32, 512, 16_000, 5);
    match variant % 4 {
        0 => ArwBuilder::new("ILCE-7RM4", 48, 32).build(&mosaic),
        1 => ArwBuilder::new("ILCE-1", 48, 32)
            .raw_ifd_levels(true)
            .iso(51_200)
            .build(&mosaic),
        2 => ArwBuilder::new("ILCE-7M4", 700, 520)
            .container_only(ContainerOnly::lossless_l())
            .build(&[]),
        _ => ArwBuilder::new("ILCE-7M5", 48, 32)
            .container_only(ContainerOnly::arw6(12))
            .build(&[]),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        let _ = probe_bytes(&bytes);
    }

    #[test]
    fn arbitrary_bytes_behind_a_tiff_header_never_panic(
        tail in proptest::collection::vec(any::<u8>(), 0..2048),
        big_endian in any::<bool>(),
    ) {
        let mut bytes = if big_endian { b"MM\0*".to_vec() } else { b"II*\0".to_vec() };
        bytes.extend_from_slice(&if big_endian { 8u32.to_be_bytes() } else { 8u32.to_le_bytes() });
        bytes.extend_from_slice(&tail);
        let _ = probe_bytes(&bytes);
    }

    #[test]
    fn mutated_valid_files_never_panic_and_errors_are_typed(
        variant in 0u8..4,
        edits in proptest::collection::vec((any::<proptest::sample::Index>(), any::<u8>()), 1..24),
    ) {
        let mut file = valid_file(variant);
        for (index, value) in edits {
            let at = index.index(file.len());
            file[at] = value;
        }
        match probe_bytes(&file) {
            Ok(meta) => {
                // Whatever survives is internally consistent.
                prop_assert!(meta.width > 0 && meta.height > 0);
                prop_assert!(meta.black.iter().zip(meta.white.iter()).all(|(b, w)| b < w));
            }
            Err(RawError::NotArw | RawError::Arq | RawError::Corrupt { .. } | RawError::Unsupported { .. }) => {}
            Err(RawError::Io(e)) => prop_assert!(false, "in-memory probe cannot fail with I/O: {e}"),
        }
    }

    #[test]
    fn truncated_valid_files_are_rejected(variant in 0u8..4, keep in 0.0f64..0.999) {
        let file = valid_file(variant);
        let len = (file.len() as f64 * keep) as usize;
        prop_assert!(probe_bytes(&file[..len]).is_err());
    }
}
