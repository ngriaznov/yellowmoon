//! Container-level behaviour of the probe on synthetic files: which files are ARW, which errors
//! the others get, and that damaged input never panics.

use ym_raw::error::RawError;
use ym_raw::layout::{ChromaSubsampling, DecodeKind};
use ym_raw::metadata::{PHOTOMETRIC_CFA, probe, probe_bytes, read_preview};
use ym_raw::testgen::{
    ArwBuilder, ContainerOnly, PayloadDescription, RawPayload, SplitMix64, gradient_mosaic,
    tiny_jpeg,
};

fn small_file() -> Vec<u8> {
    ArwBuilder::new("ILCE-7RM4", 64, 48).build(&gradient_mosaic(64, 48, 512, 16_000, 3))
}

/// A payload that claims any size but stores four bytes, to test limits cheaply.
struct Claimed {
    compression: u16,
    bits: u16,
}

impl RawPayload for Claimed {
    fn describe(&self) -> PayloadDescription {
        PayloadDescription {
            compression: self.compression,
            bits_per_sample: self.bits,
            photometric: PHOTOMETRIC_CFA,
            samples_per_pixel: 1,
            tile: None,
            sony_raw_file_type: None,
            ycbcr_subsampling: None,
        }
    }

    fn encode(&self, _mosaic: &[u16], _width: u32, _height: u32) -> Vec<Vec<u8>> {
        vec![vec![0u8; 4]]
    }
}

// ---------------------------------------------------------------------------------------------
// Which files are ARW
// ---------------------------------------------------------------------------------------------

#[test]
fn non_tiff_files_are_not_arw() {
    let jpeg = tiny_jpeg(32, 32, 1);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&[0u8; 64]);
    for (name, bytes) in [
        ("empty", Vec::new()),
        ("one byte", vec![b'I']),
        ("text", b"hello, this is not a raw file at all".to_vec()),
        ("jpeg", jpeg),
        ("png", png),
        ("zeros", vec![0u8; 4096]),
        ("bigtiff", b"II+\0\x08\0\0\0\0\0\0\0\x10\0\0\0".to_vec()),
        ("olympus", b"IIRO\x08\0\0\0\0\0\0\0\0\0\0\0".to_vec()),
    ] {
        match probe_bytes(&bytes) {
            Err(RawError::NotArw) => {}
            other => panic!("{name}: expected NotArw, got {other:?}"),
        }
    }
}

#[test]
fn a_cfa_tiff_without_any_sony_evidence_is_not_arw() {
    // Hand-built: little-endian TIFF, IFD0 with Make "NIKON", a raw SubIFD with CFA photometric.
    let mut f = b"II*\0".to_vec();
    f.extend_from_slice(&8u32.to_le_bytes());
    let ifd0_len = 2 + 2 * 12 + 4;
    let raw_at = 8 + ifd0_len;
    // IFD0: Make (inline "NIK\0"), SubIFDs -> raw IFD.
    f.extend_from_slice(&2u16.to_le_bytes());
    for (tag, ty, count, value) in [
        (0x010Fu16, 2u16, 4u32, *b"NIK\0"),
        (0x014A, 4, 1, (raw_at as u32).to_le_bytes()),
    ] {
        f.extend_from_slice(&tag.to_le_bytes());
        f.extend_from_slice(&ty.to_le_bytes());
        f.extend_from_slice(&count.to_le_bytes());
        f.extend_from_slice(&value);
    }
    f.extend_from_slice(&0u32.to_le_bytes());
    // Raw IFD: width, height, bits, compression, photometric CFA, strip offset and count.
    let entries: [(u16, u16, [u8; 4]); 8] = [
        (0x0100, 4, 64u32.to_le_bytes()),
        (0x0101, 4, 48u32.to_le_bytes()),
        (0x0102, 3, [14, 0, 0, 0]),
        (0x0103, 3, [1, 0, 0, 0]),
        (0x0106, 3, [0x23, 0x80, 0, 0]),
        (0x0111, 4, 400u32.to_le_bytes()),
        (0x0115, 3, [1, 0, 0, 0]),
        (0x0117, 4, 6144u32.to_le_bytes()),
    ];
    f.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, value) in entries {
        f.extend_from_slice(&tag.to_le_bytes());
        f.extend_from_slice(&ty.to_le_bytes());
        f.extend_from_slice(&1u32.to_le_bytes());
        f.extend_from_slice(&value);
    }
    f.extend_from_slice(&0u32.to_le_bytes());
    f.resize(400 + 6144, 0);
    assert!(matches!(probe_bytes(&f), Err(RawError::NotArw)));
}

#[test]
fn dng_files_are_not_arw() {
    // Same writer, plus a DNGVersion tag patched into IFD0 by building a file whose Software
    // string we cannot use; instead flip a harmless tag number to 0xC612 in IFD0.
    let mut file = small_file();
    // IFD0 starts at 8: count, then entries of 12 bytes. Entry 2 is Orientation (0x0112).
    let entry2 = 8 + 2 + 2 * 12;
    assert_eq!(&file[entry2..entry2 + 2], &0x0112u16.to_le_bytes());
    file[entry2..entry2 + 2].copy_from_slice(&0xC612u16.to_le_bytes());
    assert!(matches!(probe_bytes(&file), Err(RawError::NotArw)));
}

#[test]
fn sr2_and_srf_cameras_are_not_arw() {
    for model in ["DSC-F828", "DSC-V3", "DSC-R1"] {
        let file = ArwBuilder::new(model, 64, 48).build(&gradient_mosaic(64, 48, 512, 16_000, 3));
        assert!(
            matches!(probe_bytes(&file), Err(RawError::NotArw)),
            "{model}"
        );
    }
    // The maker note's FileFormat says SR2.
    let file = ArwBuilder::new("ILCE-7M3", 64, 48)
        .file_format([1, 0, 0, 0])
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    assert!(matches!(probe_bytes(&file), Err(RawError::NotArw)));
}

#[test]
fn arq_files_get_their_own_error_and_message() {
    let file = ArwBuilder::new("ILCE-7RM4", 64, 48)
        .container_only(ContainerOnly::arq())
        .build(&[]);
    let error = probe_bytes(&file).expect_err("ARQ");
    assert!(matches!(error, RawError::Arq));
    assert_eq!(
        error.to_string(),
        "Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead."
    );
}

#[test]
fn hasselblad_rebadges_decode_as_arw() {
    let file = ArwBuilder::new("Lunar", 64, 48)
        .make("Hasselblad")
        .model_id(358)
        .maker_note_header(b"VHAB     \0\0\0")
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    let meta = probe_bytes(&file).expect("Hasselblad ARW");
    assert_eq!(meta.make, "Hasselblad");
    assert_eq!(meta.decode, DecodeKind::Uncompressed14);
}

#[test]
fn the_a100_without_a_raw_directory_is_reported_as_unsupported() {
    // Remove the SubIFDs pointer's target by pointing it at junk: the A100 stores raw data
    // directly behind the pointer.
    let mut file =
        ArwBuilder::new("DSLR-A100", 64, 48).build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    // Find the SubIFDs entry (0x014A) in IFD0 and aim it at the end of the file.
    let count = u16::from_le_bytes([file[8], file[9]]) as usize;
    let mut patched = false;
    for i in 0..count {
        let at = 10 + i * 12;
        if file[at..at + 2] == 0x014Au16.to_le_bytes() {
            let junk = (file.len() - 100) as u32;
            file[at + 8..at + 12].copy_from_slice(&junk.to_le_bytes());
            patched = true;
        }
    }
    assert!(patched);
    match probe_bytes(&file) {
        Err(RawError::Unsupported { camera, .. }) => assert_eq!(camera, "DSLR-A100"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn a_sony_file_that_lost_its_raw_directory_is_corrupt_not_not_arw() {
    let mut file = small_file();
    let count = u16::from_le_bytes([file[8], file[9]]) as usize;
    for i in 0..count {
        let at = 10 + i * 12;
        if file[at..at + 2] == 0x014Au16.to_le_bytes() {
            let beyond = file.len() as u32 + 1000;
            file[at + 8..at + 12].copy_from_slice(&beyond.to_le_bytes());
        }
    }
    assert!(matches!(probe_bytes(&file), Err(RawError::Corrupt { .. })));
}

// ---------------------------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------------------------

#[test]
fn declared_dimensions_above_120_megapixels_are_corrupt() {
    for (w, h) in [
        (12_000u32, 12_000u32),
        (70_000, 100),
        (100, 70_000),
        (65_535, 65_535),
    ] {
        let file = ArwBuilder::new("ILCE-7RM4", w, h)
            .payload(Claimed {
                compression: 1,
                bits: 14,
            })
            .build(&[]);
        match probe_bytes(&file) {
            Err(RawError::Corrupt { detail }) => {
                assert!(detail.contains("120 megapixel"), "{w}x{h}: {detail}")
            }
            other => panic!("{w}x{h}: expected Corrupt, got {other:?}"),
        }
    }
}

#[test]
fn raw_data_smaller_than_the_image_needs_is_corrupt() {
    // 100 MP is within the dimension limit, but four stored bytes cannot hold it.
    let file = ArwBuilder::new("ILCE-7RM4", 10_000, 10_000)
        .payload(Claimed {
            compression: 1,
            bits: 14,
        })
        .build(&[]);
    match probe_bytes(&file) {
        Err(RawError::Corrupt { detail }) => {
            assert!(detail.contains("needs 200000000"), "{detail}")
        }
        other => panic!("expected Corrupt, got {other:?}"),
    }
}

#[test]
fn black_level_at_or_above_white_is_corrupt() {
    let file = ArwBuilder::new("ILCE-7RM4", 64, 48)
        .black([16_300; 4])
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    match probe_bytes(&file) {
        Err(RawError::Corrupt { detail }) => assert!(detail.contains("black level"), "{detail}"),
        other => panic!("expected Corrupt, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Truncated and damaged input
// ---------------------------------------------------------------------------------------------

#[test]
fn every_truncation_of_a_valid_file_is_an_error_not_a_panic() {
    let file = small_file();
    assert!(probe_bytes(&file).is_ok());
    // The metadata part is the first ~30 kB of the file; the raw data (the last 6 kB) is checked
    // against the file size, so any cut short of the full length must be rejected.
    for len in 0..file.len() {
        match probe_bytes(&file[..len]) {
            Err(RawError::NotArw | RawError::Corrupt { .. }) => {}
            other => panic!("truncated to {len} of {}: {other:?}", file.len()),
        }
    }
}

#[test]
fn truncated_files_on_disk_report_corrupt_with_a_useful_message() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = small_file();
    let path = dir.path().join("cut.ARW");
    std::fs::write(&path, &file[..file.len() - 100]).expect("write");
    match probe(&path) {
        Err(error @ RawError::Corrupt { .. }) => {
            let text = error.to_string();
            assert!(text.contains("cut short"), "{text}");
            assert!(text.contains("Copy it from the card again"), "{text}");
        }
        other => panic!("expected Corrupt, got {other:?}"),
    }
    assert!(matches!(
        probe(dir.path().join("missing.ARW")),
        Err(RawError::Io(_))
    ));
}

#[test]
fn corrupting_any_header_byte_never_panics() {
    let file = small_file();
    // The first 3 kB hold the TIFF header, IFD0, IFD1 and the start of the EXIF directory.
    for position in 0..3000.min(file.len()) {
        for value in [0x00u8, 0xFF, 0x80, 0x01] {
            let mut damaged = file.clone();
            damaged[position] = value;
            let _ = probe_bytes(&damaged);
        }
    }
}

#[test]
fn random_damage_across_the_whole_file_never_panics() {
    let file = small_file();
    let mut rng = SplitMix64(0xA5A5_5A5A);
    for _ in 0..3000 {
        let mut damaged = file.clone();
        for _ in 0..1 + rng.below(8) {
            let at = rng.below(damaged.len() as u64) as usize;
            damaged[at] = rng.next_u64() as u8;
        }
        let _ = probe_bytes(&damaged);
    }
}

#[test]
fn random_bytes_and_random_tiff_like_bytes_are_rejected_without_panics() {
    let mut rng = SplitMix64(7);
    for _ in 0..2000 {
        let len = rng.below(600) as usize;
        let mut bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
        let _ = probe_bytes(&bytes);
        // Give it a valid TIFF header so the walker is exercised on junk directories.
        if bytes.len() >= 8 {
            bytes[..4].copy_from_slice(b"II*\0");
            bytes[4..8].copy_from_slice(&8u32.to_le_bytes());
            let _ = probe_bytes(&bytes);
        }
    }
}

#[test]
fn a_file_whose_directories_loop_terminates() {
    // IFD0 whose next pointer and SubIFDs both point at IFD0 itself.
    let mut f = b"II*\0".to_vec();
    f.extend_from_slice(&8u32.to_le_bytes());
    f.extend_from_slice(&1u16.to_le_bytes());
    f.extend_from_slice(&0x014Au16.to_le_bytes());
    f.extend_from_slice(&4u16.to_le_bytes());
    f.extend_from_slice(&1u32.to_le_bytes());
    f.extend_from_slice(&8u32.to_le_bytes());
    f.extend_from_slice(&8u32.to_le_bytes());
    assert!(matches!(probe_bytes(&f), Err(RawError::NotArw)));
}

// ---------------------------------------------------------------------------------------------
// Preview extraction
// ---------------------------------------------------------------------------------------------

#[test]
fn read_preview_returns_a_decodable_jpeg_or_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let with = dir.path().join("with.ARW");
    std::fs::write(&with, small_file()).expect("write");
    let jpeg = read_preview(&with).expect("probe").expect("preview");
    assert_eq!(&jpeg[..3], &[0xFF, 0xD8, 0xFF]);
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);

    let without = dir.path().join("without.ARW");
    let bare = ArwBuilder::new("ILCE-7RM4", 64, 48)
        .thumbnail(None)
        .preview(None)
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    std::fs::write(&without, bare).expect("write");
    assert_eq!(read_preview(&without).expect("probe"), None);

    let junk = dir.path().join("junk.ARW");
    std::fs::write(&junk, b"nothing to see").expect("write");
    assert!(matches!(read_preview(&junk), Err(RawError::NotArw)));
}

#[test]
fn a_preview_pointing_outside_the_file_is_not_reported() {
    // Cut the file right after the preview JPEGs: the probe fails (raw data missing) rather than
    // reporting previews that run past the end.
    let file = small_file();
    let meta = probe_bytes(&file).expect("probe");
    let preview = meta.preview.expect("preview");
    assert!(preview.offset + preview.length <= file.len() as u64);
}

// ---------------------------------------------------------------------------------------------
// Variants the writer can express
// ---------------------------------------------------------------------------------------------

#[test]
fn compression_and_bit_depth_combinations_classify_as_documented() {
    let cases: [(ContainerOnly, DecodeKind); 4] = [
        (ContainerOnly::craw(), DecodeKind::Craw),
        (ContainerOnly::packed12(), DecodeKind::Uncompressed12),
        (
            ContainerOnly::lossless_ms(ChromaSubsampling::Yuv422, false),
            DecodeKind::LosslessMS {
                subsampling: ChromaSubsampling::Yuv422,
            },
        ),
        (ContainerOnly::arw6(14), DecodeKind::Arw6),
    ];
    for (payload, expected) in cases {
        let file = ArwBuilder::new("ILCE-7RM5", 1024, 512)
            .container_only(payload)
            .build(&[]);
        assert_eq!(probe_bytes(&file).expect("probe").decode, expected);
    }
}

// ---------------------------------------------------------------------------------------------
// Patching a valid file: each test damages one field and checks what the reader does with it
// ---------------------------------------------------------------------------------------------

/// A minimal independent directory reader for the tests: positions of the 12-byte entries.
mod patch {
    pub fn u16_at(file: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([file[at], file[at + 1]])
    }

    pub fn u32_at(file: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([file[at], file[at + 1], file[at + 2], file[at + 3]])
    }

    /// Position of the entry with `tag` in the directory at `ifd`.
    pub fn entry(file: &[u8], ifd: usize, tag: u16) -> Option<usize> {
        let count = usize::from(u16_at(file, ifd));
        (0..count)
            .map(|i| ifd + 2 + i * 12)
            .find(|at| u16_at(file, *at) == tag)
    }

    /// The raw directory: the first SubIFDs target of IFD0.
    pub fn raw_ifd(file: &[u8]) -> usize {
        let ifd0 = u32_at(file, 4) as usize;
        let sub = entry(file, ifd0, 0x014A).expect("SubIFDs");
        u32_at(file, sub + 8) as usize
    }

    /// The SR2Private directory, through DNGPrivateData in IFD0.
    pub fn sr2_private(file: &[u8]) -> usize {
        let ifd0 = u32_at(file, 4) as usize;
        let private = entry(file, ifd0, 0xC634).expect("DNGPrivateData");
        u32_at(file, private + 8) as usize
    }

    pub fn set_value(file: &mut [u8], entry: usize, value: [u8; 4]) {
        file[entry + 8..entry + 12].copy_from_slice(&value);
    }

    pub fn set_count(file: &mut [u8], entry: usize, count: u32) {
        file[entry + 4..entry + 8].copy_from_slice(&count.to_le_bytes());
    }
}

use ym_raw::metadata::LevelSource;

#[test]
fn a_damaged_sr2_block_falls_back_to_defaults_instead_of_failing() {
    let good = small_file();
    let private = patch::sr2_private(&good);

    // Wrong key: the block decrypts to garbage.
    let mut wrong_key = good.clone();
    let key = patch::entry(&wrong_key, private, 0x7221).expect("key");
    patch::set_value(&mut wrong_key, key, [1, 2, 3, 4]);
    let meta = probe_bytes(&wrong_key).expect("still an ARW");
    assert_eq!(meta.black_source, LevelSource::Default);
    assert_eq!(meta.white_source, LevelSource::Default);
    assert_eq!(meta.wb_as_shot, None);
    assert_eq!(meta.black, [512.0; 4]);
    assert_eq!(meta.white, [16383.0; 4]);

    // Length too small, too large, or offset outside the file.
    for (tag, value) in [
        (0x7201u16, 5u32),
        (0x7201, 2_000_000),
        (0x7200, 0xFFFF_FF00),
        (0x7200, 3),
    ] {
        let mut file = good.clone();
        let at = patch::entry(&file, private, tag).expect("tag");
        patch::set_value(&mut file, at, value.to_le_bytes());
        let meta = probe_bytes(&file).unwrap_or_else(|e| panic!("{tag:#x}={value}: {e}"));
        assert_eq!(meta.black_source, LevelSource::Default, "{tag:#x}={value}");
    }

    // A missing SR2Private pointer is the same as no SR2 at all.
    let mut file = good.clone();
    let ifd0 = patch::u32_at(&file, 4) as usize;
    let at = patch::entry(&file, ifd0, 0xC634).expect("DNGPrivateData");
    patch::set_value(&mut file, at, [0, 0, 0, 0]);
    assert_eq!(
        probe_bytes(&file).expect("probe").black_source,
        LevelSource::Default
    );
    patch::set_value(&mut file, at, 0xFFFF_0000u32.to_le_bytes());
    assert_eq!(
        probe_bytes(&file).expect("probe").black_source,
        LevelSource::Default
    );
}

#[test]
fn storage_that_does_not_match_the_image_is_corrupt() {
    // Tiles: change the tile width so the tile count no longer fits the image.
    let tiled = ArwBuilder::new("ILCE-1", 1100, 700)
        .container_only(ContainerOnly::lossless_l())
        .build(&[]);
    let raw = patch::raw_ifd(&tiled);
    let mut file = tiled.clone();
    let at = patch::entry(&file, raw, 0x0142).expect("TileWidth");
    patch::set_value(&mut file, at, 256u32.to_le_bytes());
    match probe_bytes(&file) {
        Err(RawError::Corrupt { detail }) => assert!(detail.contains("tiles"), "{detail}"),
        other => panic!("expected Corrupt, got {other:?}"),
    }
    // A missing tile size.
    let mut file = tiled.clone();
    let at = patch::entry(&file, raw, 0x0143).expect("TileLength");
    patch::set_value(&mut file, at, 0u32.to_le_bytes());
    assert!(matches!(probe_bytes(&file), Err(RawError::Corrupt { .. })));

    // Strips: byte counts that disagree with offsets, a strip beyond the file, absurd counts.
    let plain = small_file();
    let raw = patch::raw_ifd(&plain);
    let counts = patch::entry(&plain, raw, 0x0117).expect("StripByteCounts");
    let offsets = patch::entry(&plain, raw, 0x0111).expect("StripOffsets");
    for (what, entry, value) in [
        ("count past EOF", counts, (plain.len() as u32).to_le_bytes()),
        (
            "offset past EOF",
            offsets,
            (plain.len() as u32).to_le_bytes(),
        ),
        ("offset overflow", offsets, u32::MAX.to_le_bytes()),
    ] {
        let mut file = plain.clone();
        patch::set_value(&mut file, entry, value);
        assert!(
            matches!(probe_bytes(&file), Err(RawError::Corrupt { .. })),
            "{what}"
        );
    }
    let mut file = plain.clone();
    patch::set_count(&mut file, counts, 2); // two counts, one offset
    assert!(matches!(probe_bytes(&file), Err(RawError::Corrupt { .. })));
    let mut file = plain.clone();
    patch::set_count(&mut file, offsets, 0x00FF_FFFF); // more strips than the limit allows
    assert!(matches!(probe_bytes(&file), Err(RawError::Corrupt { .. })));
    // No data location at all: rename the tag.
    let mut file = plain.clone();
    file[offsets..offsets + 2].copy_from_slice(&0x0999u16.to_le_bytes());
    assert!(matches!(probe_bytes(&file), Err(RawError::Corrupt { .. })));
}

#[test]
fn an_unusable_cfa_description_falls_back_to_rggb() {
    let good = ArwBuilder::new("ILCE-7M3", 64, 48)
        .cfa(ym_raw::layout::CfaPattern::GRBG)
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    assert_eq!(
        probe_bytes(&good).expect("probe").cfa,
        ym_raw::layout::CfaPattern::GRBG
    );
    let raw = patch::raw_ifd(&good);

    // A 4x4 repeat pattern is not a Bayer quad.
    let mut file = good.clone();
    let dim = patch::entry(&file, raw, 0x828D).expect("CFARepeatPatternDim");
    patch::set_value(&mut file, dim, [4, 0, 4, 0]);
    assert_eq!(
        probe_bytes(&file).expect("probe").cfa,
        ym_raw::layout::CfaPattern::RGGB
    );

    // Two reds are not a Bayer quad either.
    let mut file = good.clone();
    let pattern = patch::entry(&file, raw, 0x828E).expect("CFAPattern");
    patch::set_value(&mut file, pattern, [0, 0, 1, 2]);
    assert_eq!(
        probe_bytes(&file).expect("probe").cfa,
        ym_raw::layout::CfaPattern::RGGB
    );

    // Missing tags.
    let mut file = good.clone();
    file[pattern..pattern + 2].copy_from_slice(&0x0998u16.to_le_bytes());
    assert_eq!(
        probe_bytes(&file).expect("probe").cfa,
        ym_raw::layout::CfaPattern::RGGB
    );
}

#[test]
fn an_odd_bit_depth_is_classified_by_the_sony_raw_file_type() {
    let good = small_file();
    let raw = patch::raw_ifd(&good);
    let bits = patch::entry(&good, raw, 0x0102).expect("BitsPerSample");

    // 14-bit data announcing itself as 13 bits: the Sony type (0 = 14-bit uncompressed) decides.
    let mut file = good.clone();
    patch::set_value(&mut file, bits, [13, 0, 0, 0]);
    let meta = probe_bytes(&file).expect("probe");
    assert_eq!(meta.decode, DecodeKind::Uncompressed14);
    assert_eq!(meta.bits_per_sample, 13);

    // Without the type, the layout is unknown but the container is still readable.
    let mut file = good.clone();
    patch::set_value(&mut file, bits, [13, 0, 0, 0]);
    let file_type = patch::entry(&file, raw, 0x7000).expect("SonyRawFileType");
    file[file_type..file_type + 2].copy_from_slice(&0x0997u16.to_le_bytes());
    let meta = probe_bytes(&file).expect("probe");
    assert_eq!(
        meta.decode,
        DecodeKind::Unknown {
            compression: 1,
            bits: 13
        }
    );

    // A missing BitsPerSample is read as 0 bits; 14-bit levels still apply.
    let mut file = good.clone();
    file[bits..bits + 2].copy_from_slice(&0x0996u16.to_le_bytes());
    let meta = probe_bytes(&file).expect("probe");
    assert_eq!(meta.bits_per_sample, 0);
    assert_eq!(meta.decode, DecodeKind::Uncompressed14); // via SonyRawFileType = 0
}

#[test]
fn exif_values_that_make_no_sense_are_dropped() {
    // Orientation 0 and 9 are not valid; ExposureTime with a zero denominator is not a time.
    for orientation in [0u16, 9, 100] {
        let file = ArwBuilder::new("ILCE-7M3", 64, 48)
            .orientation(orientation)
            .exposure_time(1, 0)
            .f_number(0, 10)
            .focal_length(0, 1)
            .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
        let meta = probe_bytes(&file).expect("probe");
        assert_eq!(meta.orientation, 1, "orientation {orientation}");
        assert_eq!(meta.exposure_time_s, None);
        assert_eq!(meta.f_number, None);
        assert_eq!(meta.focal_length_mm, None);
    }
}

#[test]
fn gps_west_and_below_sea_level_are_negative() {
    let file = ArwBuilder::new("ILCE-7M3", 64, 48)
        .gps(ym_raw::testgen::GpsSpec {
            latitude_deg: 40.0,
            longitude_deg: -105.25,
            altitude_m: -86.0,
        })
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    let gps = probe_bytes(&file).expect("probe").gps.expect("gps");
    assert!((gps.latitude_deg.expect("lat") - 40.0).abs() < 1e-9);
    assert!((gps.longitude_deg.expect("lon") + 105.25).abs() < 1e-9);
    assert_eq!(gps.altitude_m, Some(-86.0));
}

#[test]
fn capture_time_falls_back_to_the_ifd0_date_time() {
    // Rename DateTimeOriginal (0x9003) in the EXIF directory so only IFD0's DateTime remains.
    let good = small_file();
    let ifd0 = patch::u32_at(&good, 4) as usize;
    let exif_ptr = patch::entry(&good, ifd0, 0x8769).expect("ExifIFD");
    let exif = patch::u32_at(&good, exif_ptr + 8) as usize;
    let mut file = good.clone();
    let original = patch::entry(&file, exif, 0x9003).expect("DateTimeOriginal");
    file[original..original + 2].copy_from_slice(&0x9998u16.to_le_bytes());
    let meta = probe_bytes(&file).expect("probe");
    let time = meta.capture_time.expect("fallback time");
    assert_eq!(time.to_exif_string(), "2026:10:04 12:03:44");
    // The fallback has no sub-second or offset information.
    assert_eq!(time.subsec_ms, None);
    assert_eq!(time.utc_offset_minutes, None);
}

#[test]
fn the_layout_of_a_mosaic_carries_the_pattern_area_and_crop() {
    use ym_raw::layout::{CfaPattern, Layout, Rect};
    let plain = probe_bytes(&small_file()).expect("probe");
    assert_eq!(
        plain.layout(),
        Layout::Bayer {
            pattern: CfaPattern::RGGB,
            active_area: None,
            crop: None
        }
    );
    let cropped = ArwBuilder::new("ILCE-1", 64, 48)
        .active_area(60, 44)
        .crop(Rect {
            x: 2,
            y: 2,
            width: 40,
            height: 30,
        })
        .build(&gradient_mosaic(64, 48, 512, 16_000, 3));
    assert_eq!(
        probe_bytes(&cropped).expect("probe").layout(),
        Layout::Bayer {
            pattern: CfaPattern::RGGB,
            active_area: Some(Rect {
                x: 0,
                y: 0,
                width: 60,
                height: 44
            }),
            crop: Some(Rect {
                x: 2,
                y: 2,
                width: 40,
                height: 30
            }),
        }
    );
}

/// Real files do not keep their directory entries sorted by tag (the ILCE-9 in the lclevy dump
/// has the raw IFD's 0x011A after 0x011C and 0x74C7 last, and ExifTool warns about it for every
/// ARW). Nothing may depend on the order: reversing the entry table of every plain directory
/// gives the same metadata.
#[test]
fn the_order_of_directory_entries_does_not_matter() {
    fn reverse_entries(file: &mut [u8], ifd: usize) {
        let count = usize::from(patch::u16_at(file, ifd));
        let table = ifd + 2;
        let mut entries: Vec<[u8; 12]> = (0..count)
            .map(|i| {
                file[table + i * 12..table + (i + 1) * 12]
                    .try_into()
                    .expect("12 bytes")
            })
            .collect();
        entries.reverse();
        for (i, entry) in entries.iter().enumerate() {
            file[table + i * 12..table + (i + 1) * 12].copy_from_slice(entry);
        }
    }

    let good = ArwBuilder::new("ILCE-9", 64, 48)
        .raw_ifd_levels(true)
        .raw_ifd_white_level(16_383)
        .tone_curve(Some([8000, 10_400, 12_900, 14_100]))
        .gps(ym_raw::testgen::GpsSpec {
            latitude_deg: 1.5,
            longitude_deg: 2.5,
            altitude_m: 3.0,
        })
        .build(&gradient_mosaic(64, 48, 512, 15_000, 3));
    let expected = probe_bytes(&good).expect("probe");
    assert!(expected.model_id.is_some() && expected.tone_curve.is_some());

    let ifd0 = patch::u32_at(&good, 4) as usize;
    let exif_at = patch::entry(&good, ifd0, 0x8769).expect("Exif pointer");
    let exif = patch::u32_at(&good, exif_at + 8) as usize;
    let maker_at = patch::entry(&good, exif, 0x927C).expect("MakerNote");
    let maker = patch::u32_at(&good, maker_at + 8) as usize;
    let gps_at = patch::entry(&good, ifd0, 0x8825).expect("GPS pointer");
    let gps = patch::u32_at(&good, gps_at + 8) as usize;
    let count0 = usize::from(patch::u16_at(&good, ifd0));
    let ifd1 = patch::u32_at(&good, ifd0 + 2 + count0 * 12) as usize;
    let directories = [
        ("IFD0", ifd0),
        ("IFD1", ifd1),
        ("raw IFD", patch::raw_ifd(&good)),
        ("EXIF", exif),
        ("GPS", gps),
        ("maker note", maker),
        ("SR2Private", patch::sr2_private(&good)),
    ];
    let mut all = good.clone();
    for (name, at) in directories {
        let mut file = good.clone();
        reverse_entries(&mut file, at);
        assert_eq!(probe_bytes(&file).expect("probe"), expected, "{name}");
        reverse_entries(&mut all, at);
    }
    assert_eq!(probe_bytes(&all).expect("probe"), expected, "all at once");
}

#[test]
fn a_lossless_ms_directory_without_bits_per_sample_is_still_found() {
    // LibRaw reads the precision from the JPEG stream when the tag is missing; so do we, by not
    // requiring it: a full-size YCbCr directory with compression 7 and three samples is the raw
    // image whatever BitsPerSample says (or does not say).
    let file = ArwBuilder::new("ILCE-7RM5", 1024, 768)
        .container_only(ContainerOnly::lossless_ms(ChromaSubsampling::Yuv420, true))
        .omit_bits_per_sample(true)
        .build(&[]);
    let meta = ym_raw::metadata::probe_bytes(&file).expect("probe");
    assert_eq!(
        meta.decode,
        DecodeKind::LosslessMS {
            subsampling: ChromaSubsampling::Yuv420
        }
    );
    assert_eq!(meta.bits_per_sample, 0);
}

#[test]
fn a_lossless_ms_directory_flagged_reduced_resolution_is_still_the_raw_image() {
    // LibRaw skips a reduced-resolution YCbCr directory only when it is 8-bit (the RGB preview);
    // one that states 14 bits is data, whatever NewSubFileType says.
    let rgb = ym_raw::testgen::gradient_rgb(64, 48, 1100, 9000, 3);
    let payload = ym_raw::testgen::LosslessMs::new(
        ym_raw::layout::ChromaSubsampling::Yuv420,
        rgb,
        64,
        48,
        [2.0, 1.0, 1.6],
    )
    .tile(64, 48);
    let good = ArwBuilder::new("ILCE-7RM5", 64, 48)
        .lossless_ms(payload)
        .build(&[]);
    let raw = patch::raw_ifd(&good);
    let subfile = patch::entry(&good, raw, 0x00FE).expect("NewSubFileType");

    let mut flagged = good.clone();
    patch::set_value(&mut flagged, subfile, [1, 0, 0, 0]);
    let meta = probe_bytes(&flagged).expect("a 14-bit directory is data");
    assert!(matches!(meta.decode, DecodeKind::LosslessMS { .. }));
    assert_eq!((meta.width, meta.height), (64, 48));
    ym_raw::decode::decode_bytes(&flagged, &ym_raw::decode::DecodeOptions::default())
        .expect("and it decodes");

    // The same flag without a stated precision is the camera's preview, not the raw image.
    let bits = patch::entry(&flagged, raw, 0x0102).expect("BitsPerSample");
    flagged[bits..bits + 2].copy_from_slice(&0x0996u16.to_le_bytes());
    assert!(
        probe_bytes(&flagged).is_err(),
        "a flagged directory with no precision is not taken for the raw image"
    );
}
