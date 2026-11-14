//! DNG input: every shape the reader accepts round-trips a known plane exactly, every shape it
//! refuses gives its sentence, and damaged files give errors, not panics.

use ym_raw::decode::{DecodeOptions, RawImage, decode_bytes};
use ym_raw::dng::{DngKind, GainMap};
use ym_raw::error::{DngRefusal, RawError};
use ym_raw::layout::{CfaPattern, DecodeKind, Layout, Rect};
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::dng::{DngBuilder, DngCompression, DngStorage, opcode_list};
use ym_raw::testgen::{SplitMix64, tiny_jpeg};

/// A deterministic plane of `n` samples below `2^bits`.
fn plane(n: usize, bits: u16, seed: u64) -> Vec<u16> {
    let mut rng = SplitMix64(seed);
    let top = 1u64 << bits;
    // Smooth ramps plus noise exercise every predictor; the extremes are included.
    let mut out: Vec<u16> = (0..n)
        .map(|i| (((i as u64 * 7) % top / 2) + rng.below(top / 8 + 1)).min(top - 1) as u16)
        .collect();
    out[0] = 0;
    out[n - 1] = (top - 1) as u16;
    out
}

fn decode(bytes: &[u8]) -> RawImage {
    decode_bytes(bytes, &DecodeOptions::default()).expect("a DNG that decodes")
}

fn refusal(bytes: &[u8]) -> DngRefusal {
    match probe_bytes(bytes) {
        Err(RawError::UnsupportedDng(why)) => why,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_cfa_dng_round_trips_in_all_four_orders() {
    for cfa in [
        CfaPattern::RGGB,
        CfaPattern::BGGR,
        CfaPattern::GRBG,
        CfaPattern::GBRG,
    ] {
        let mut b = DngBuilder::new("Acme", "Cam 1", 40, 30);
        b.cfa = cfa;
        let samples = plane(40 * 30, 12, 1);
        let image = decode(&b.build(&samples));
        assert_eq!(image.data.data, samples, "{}", cfa.name());
        assert_eq!((image.data.width, image.data.height), (40, 30));
        let Layout::Bayer { pattern, .. } = image.layout else {
            panic!("a mosaic")
        };
        assert_eq!(pattern, cfa);
        assert_eq!(image.meta.decode, DecodeKind::DngCfa);
        assert_eq!(image.black, [64.0; 4]);
        assert_eq!(image.white, [4095.0; 4]);
        assert_eq!(image.meta.make, "Acme");
        assert_eq!(image.meta.model, "Cam 1");
        assert!(!image.is_reduced());
    }
}

#[test]
fn uncompressed_samples_of_every_bit_depth_and_byte_order() {
    for bits in [8u16, 9, 10, 11, 12, 13, 14, 15, 16] {
        for big_endian in [false, true] {
            let mut b = DngBuilder::new("Acme", "Cam 1", 33, 17);
            b.bits = bits;
            b.big_endian = big_endian;
            b.black = vec![0.0];
            b.white = Some(vec![(1u32 << bits) - 1]);
            let samples = plane(33 * 17, bits, u64::from(bits));
            let image = decode(&b.build(&samples));
            assert_eq!(
                image.data.data, samples,
                "{bits} bits, big endian {big_endian}"
            );
        }
    }
}

#[test]
fn strips_and_tiles_cover_the_image_with_padded_edges() {
    let samples = plane(38 * 29, 14, 3);
    for storage in [
        DngStorage::Strips(29),
        DngStorage::Strips(8),
        DngStorage::Strips(1),
        DngStorage::Tiles(16, 16),
        DngStorage::Tiles(40, 32),
        DngStorage::Tiles(8, 4),
    ] {
        for compression in [
            DngCompression::None,
            DngCompression::LosslessJpeg {
                components: 2,
                predictor: 1,
            },
        ] {
            let mut b = DngBuilder::new("Acme", "Cam 1", 38, 29);
            b.bits = 14;
            b.black = vec![256.0];
            b.white = Some(vec![16383]);
            b.storage = storage;
            b.compression = compression;
            // Tiles must be a multiple of the component count wide for the JPEG layout.
            let image = decode(&b.build(&samples));
            assert_eq!(image.data.data, samples, "{storage:?} {compression:?}");
        }
    }
}

#[test]
fn lossless_jpeg_with_every_predictor_and_component_count() {
    for predictor in 1u8..=7 {
        for components in 1usize..=4 {
            for bits in [12u16, 14, 16] {
                let mut b = DngBuilder::new("Acme", "Cam 1", 48, 20);
                b.bits = bits;
                b.black = vec![0.0];
                b.white = Some(vec![(1u32 << bits) - 1]);
                b.compression = DngCompression::LosslessJpeg {
                    components,
                    predictor,
                };
                b.storage = DngStorage::Tiles(24, 10);
                let samples = plane(48 * 20, bits, u64::from(predictor) * 10 + components as u64);
                let image = decode(&b.build(&samples));
                assert_eq!(
                    image.data.data, samples,
                    "predictor {predictor}, {components} components, {bits} bits"
                );
            }
        }
    }
}

#[test]
fn linear_raw_becomes_camera_rgb_with_levels_in_red_green_green_blue_order() {
    for compression in [
        DngCompression::None,
        DngCompression::LosslessJpeg {
            components: 3,
            predictor: 6,
        },
    ] {
        let mut b = DngBuilder::new("Acme", "Phone", 21, 15);
        b.kind = DngKind::LinearRaw;
        b.bits = 16;
        b.compression = compression;
        b.black_repeat = Some((1, 1));
        b.black = vec![100.0, 200.0, 300.0];
        b.white = Some(vec![60000, 61000, 62000]);
        b.storage = DngStorage::Strips(6);
        let samples = plane(21 * 15 * 3, 16, 9);
        let image = decode(&b.build(&samples));
        assert_eq!(image.data.data, samples);
        assert_eq!((image.data.width, image.data.height), (63, 15));
        assert_eq!(image.pixel_width(), 21);
        assert!(image.is_reduced());
        assert_eq!(image.meta.decode, DecodeKind::DngLinear);
        assert!(matches!(image.layout, Layout::Ycbcr { .. }));
        assert_eq!(image.black, [100.0, 200.0, 200.0, 300.0]);
        assert_eq!(image.white, [60000.0, 61000.0, 61000.0, 62000.0]);
    }
}

#[test]
fn the_black_level_pattern_follows_the_cfa_and_the_active_area() {
    let mut b = DngBuilder::new("Acme", "Cam 1", 24, 20);
    b.black_repeat = Some((2, 2));
    b.black = vec![10.0, 20.0, 30.0, 40.0];
    let samples = plane(24 * 20, 12, 5);
    let image = decode(&b.build(&samples));
    // Raster order of the readout.
    assert_eq!(image.black, [10.0, 20.0, 30.0, 40.0]);

    // An active area that starts on an odd row and column: the pattern (and the CFA) are
    // relative to it, so the readout positions swap.
    b.active_area = Some([1, 1, 19, 23]);
    let image = decode(&b.build(&samples));
    assert_eq!(image.black, [40.0, 30.0, 20.0, 10.0]);
    let Layout::Bayer {
        pattern,
        active_area,
        ..
    } = image.layout
    else {
        panic!("mosaic")
    };
    assert_eq!(pattern, CfaPattern::BGGR);
    assert_eq!(
        active_area,
        Some(Rect {
            x: 1,
            y: 1,
            width: 22,
            height: 18
        })
    );
}

#[test]
fn black_level_deltas_come_off_the_samples() {
    let (w, h) = (16usize, 12usize);
    let mut b = DngBuilder::new("Acme", "Cam 1", w, h);
    b.black = vec![50.0];
    b.black_delta_h = (0..w).map(|x| (x % 3) as f64).collect();
    b.black_delta_v = (0..h).map(|y| -((y % 2) as f64)).collect();
    let samples = plane(w * h, 12, 6);
    let image = decode(&b.build(&samples));
    for y in 0..h {
        for x in 0..w {
            let delta = ((x % 3) as i32) - ((y % 2) as i32);
            let want = (i32::from(samples[y * w + x]) - delta).clamp(0, 65535);
            assert_eq!(i32::from(image.data.data[y * w + x]), want, "({x},{y})");
        }
    }
}

#[test]
fn a_linearization_table_maps_every_sample_and_sets_the_default_white() {
    let mut b = DngBuilder::new("Acme", "Cam 1", 16, 8);
    b.bits = 10;
    b.linearization = Some((0..1024u32).map(|i| (i * i / 16) as u16).collect());
    b.black = vec![0.0];
    b.white = None;
    let samples = plane(16 * 8, 10, 7);
    let bytes = b.build(&samples);
    let image = decode(&bytes);
    let want: Vec<u16> = samples
        .iter()
        .map(|v| (u32::from(*v) * u32::from(*v) / 16) as u16)
        .collect();
    assert_eq!(image.data.data, want);
    // No WhiteLevel: the largest table value.
    assert_eq!(image.white, [(1023u32 * 1023 / 16) as f32; 4]);
}

#[test]
fn gain_maps_scale_the_samples_above_black_and_never_past_white() {
    let (w, h) = (16usize, 16usize);
    let mut b = DngBuilder::new("Acme", "Cam 1", w, h);
    b.black = vec![64.0];
    let samples: Vec<u16> = (0..w * h).map(|i| 64 + (i % 200) as u16 * 10).collect();
    // One map over the whole image, a vertical ramp from 1 to 2.
    b.gain_maps = vec![GainMap {
        top: 0,
        left: 0,
        bottom: h as u32,
        right: w as u32,
        plane: 0,
        planes: 1,
        row_pitch: 1,
        col_pitch: 1,
        points_v: 2,
        points_h: 1,
        spacing_v: 1.0,
        spacing_h: 1.0,
        origin_v: 0.0,
        origin_h: 0.0,
        map_planes: 1,
        gains: vec![1.0, 2.0],
    }];
    let image = decode(&b.build(&samples));
    for y in 0..h {
        let g = 1.0 + y as f32 / (h - 1) as f32;
        for x in 0..w {
            let v = f32::from(samples[y * w + x]);
            let want = (64.0 + (v - 64.0) * g).round().clamp(0.0, 4095.0) as u16;
            let got = image.data.data[y * w + x];
            assert!(got.abs_diff(want) <= 1, "({x},{y}) {got} vs {want}");
        }
    }
    assert!(image.data.data.iter().all(|v| *v <= 4095));
}

#[test]
fn four_gain_maps_with_a_pitch_of_two_each_serve_one_cfa_position() {
    let (w, h) = (12usize, 10usize);
    let mut b = DngBuilder::new("Acme", "Cam 1", w, h);
    b.black = vec![0.0];
    let samples = vec![1000u16; w * h];
    let gains = [1.0f32, 1.25, 1.5, 2.0];
    b.gain_maps = (0..4)
        .map(|k| GainMap {
            top: (k / 2) as u32,
            left: (k % 2) as u32,
            bottom: h as u32,
            right: w as u32,
            plane: 0,
            planes: 1,
            row_pitch: 2,
            col_pitch: 2,
            points_v: 1,
            points_h: 1,
            spacing_v: 1.0,
            spacing_h: 1.0,
            origin_v: 0.0,
            origin_h: 0.0,
            map_planes: 1,
            gains: vec![gains[k]],
        })
        .collect();
    let image = decode(&b.build(&samples));
    for y in 0..h {
        for x in 0..w {
            let want = (1000.0 * gains[(y % 2) * 2 + x % 2]) as u16;
            assert_eq!(image.data.data[y * w + x], want, "({x},{y})");
        }
    }
}

#[test]
fn an_orientation_a_default_crop_and_the_as_shot_neutral_are_read() {
    let mut b = DngBuilder::new("Acme", "Cam 1", 40, 30);
    b.orientation = 6;
    b.active_area = Some([2, 4, 28, 38]);
    b.crop = Some(([3, 5], [20, 10]));
    b.as_shot_neutral = Some([0.5, 1.0, 0.25]);
    b.iso = Some(800);
    b.lens_model = Some("Test 35mm F2".to_owned());
    let meta = probe_bytes(&b.build(&plane(40 * 30, 12, 8))).expect("probe");
    assert_eq!(meta.orientation, 6);
    assert_eq!(
        meta.active_area,
        Some(Rect {
            x: 4,
            y: 2,
            width: 34,
            height: 26
        })
    );
    // The crop is relative to the active area: its origin is added.
    assert_eq!(
        meta.crop,
        Some(Rect {
            x: 7,
            y: 7,
            width: 20,
            height: 10
        })
    );
    assert_eq!(meta.iso, Some(800));
    assert_eq!(meta.lens_model.as_deref(), Some("Test 35mm F2"));
    assert_eq!(meta.exposure_time_s, Some(1.0 / 125.0));
    // Gains are the inverse of the neutral, green about 1024, per CFA position.
    let wb = meta.wb_as_shot.expect("white balance");
    assert_eq!(wb[1], wb[2]);
    assert!((wb[1] - 1024.0).abs() < 1.0);
    assert!((wb[0] - 2048.0).abs() < 1.0);
    assert!((wb[3] - 4096.0).abs() < 1.0);
}

#[test]
fn baseline_exposure_and_the_other_opcode_lists_change_nothing() {
    let samples = plane(24 * 16, 12, 11);
    let plain = DngBuilder::new("Acme", "Cam 1", 24, 16);
    let mut busy = plain.clone();
    busy.baseline_exposure = Some(1.5);
    busy.other_opcode_lists = [
        Some(opcode_list(&[(4, 0, vec![0; 16])])),
        Some(opcode_list(&[(1, 0, vec![0; 16])])),
    ];
    let a = decode(&plain.build(&samples));
    let b = decode(&busy.build(&samples));
    assert_eq!(a.data, b.data);
    assert_eq!((a.black, a.white), (b.black, b.white));
}

#[test]
fn the_preview_is_the_embedded_jpeg() {
    let mut b = DngBuilder::new("Acme", "Cam 1", 24, 16);
    b.preview = Some(tiny_jpeg(16, 16, 3));
    let bytes = b.build(&plane(24 * 16, 12, 12));
    let meta = probe_bytes(&bytes).expect("probe");
    let jpeg = meta.preview.expect("a preview");
    assert_eq!(&bytes[jpeg.offset as usize..][..3], &[0xFF, 0xD8, 0xFF]);
    // And none without one.
    let meta =
        probe_bytes(&DngBuilder::new("Acme", "Cam 1", 24, 16).build(&plane(24 * 16, 12, 12)))
            .expect("probe");
    assert!(meta.preview.is_none());
}

#[test]
fn colour_tags_are_kept_as_written() {
    let mut b = DngBuilder::new("Acme", "Cam 1", 24, 16);
    b.color_matrix = [
        Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Some([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
    ];
    b.illuminant = [Some(17), Some(21)];
    b.analog_balance = Some([1.1, 1.0, 0.9]);
    b.noise_profile = Some(vec![1e-4, 1e-6]);
    let meta = probe_bytes(&b.build(&plane(24 * 16, 12, 13))).expect("probe");
    let dng = meta.dng.expect("DNG data");
    assert_eq!(dng.version, [1, 4, 0, 0]);
    assert_eq!(dng.kind, DngKind::Cfa);
    assert_eq!(dng.calibration_illuminant, [Some(17), Some(21)]);
    assert_eq!(dng.color_matrix[1].expect("matrix 2")[0], 2.0);
    assert_eq!(dng.analog_balance, Some([1.1, 1.0, 0.9]));
    assert_eq!(dng.noise_profile, Some(vec![1e-4, 1e-6]));
    assert_eq!(dng.unique_camera_model.as_deref(), Some("Acme Cam 1"));
}

#[test]
fn what_cannot_be_read_is_refused_with_its_sentence() {
    let samples = plane(24 * 16, 12, 14);
    let base = DngBuilder::new("Acme", "Cam 1", 24, 16);
    let with = |f: &dyn Fn(&mut DngBuilder)| {
        let mut b = base.clone();
        f(&mut b);
        b.build(&samples)
    };
    assert_eq!(
        refusal(&with(&|b| b.compression_tag = Some(34892))),
        DngRefusal::LossyJpeg
    );
    assert_eq!(
        refusal(&with(&|b| b.compression_tag = Some(52546))),
        DngRefusal::JpegXl
    );
    assert_eq!(
        refusal(&with(&|b| b.compression_tag = Some(8))),
        DngRefusal::Deflate
    );
    assert_eq!(
        refusal(&with(&|b| b.compression_tag = Some(32946))),
        DngRefusal::Deflate
    );
    assert_eq!(
        refusal(&with(&|b| b.compression_tag = Some(5))),
        DngRefusal::Compression(5)
    );
    assert_eq!(
        refusal(&with(&|b| b.sample_format = Some(3))),
        DngRefusal::FloatData
    );
    assert_eq!(
        refusal(&with(&|b| b.sample_format = Some(2))),
        DngRefusal::FloatData
    );
    assert_eq!(
        refusal(&with(&|b| b.cfa_dim = Some([6, 6]))),
        DngRefusal::NotBayer
    );
    assert_eq!(
        refusal(&with(&|b| b.default_scale = Some([2.0, 1.0]))),
        DngRefusal::Scaled
    );
    assert_eq!(refusal(&with(&|b| b.bits = 20)), DngRefusal::SampleLayout);
    assert_eq!(refusal(&with(&|b| b.bits = 4)), DngRefusal::SampleLayout);
    assert_eq!(
        refusal(&with(&|b| b.backward_version = Some([1, 8, 0, 0]))),
        DngRefusal::NewerVersion
    );
    // A DNG of a developed picture.
    assert_eq!(
        refusal(&with(&|b| b.photometric_tag = Some(2))),
        DngRefusal::NotRaw
    );
    // An opcode the reader cannot apply and may not skip, and one it may skip.
    assert_eq!(
        refusal(&with(&|b| b.extra_opcodes = vec![(7, 0, vec![0; 8])])),
        DngRefusal::Opcode(7)
    );
    assert!(probe_bytes(&with(&|b| b.extra_opcodes = vec![(7, 1, vec![0; 8])])).is_ok());
    // The sentence.
    let text = DngRefusal::JpegXl.message();
    assert_eq!(
        RawError::UnsupportedDng(DngRefusal::JpegXl).to_string(),
        text
    );
    assert_eq!(
        RawError::UnsupportedDng(DngRefusal::JpegXl).kind(),
        "unsupported"
    );
}

#[test]
fn a_four_colour_or_odd_cfa_is_not_bayer() {
    // Two reds, one green, one blue is not a Bayer quad: CFAPlaneColor maps 0 -> red, so a pattern
    // [0, 0, 1, 2] fails the check.
    let mut b = DngBuilder::new("Acme", "Cam 1", 24, 16);
    b.cfa = CfaPattern::RGGB;
    let mut bytes = b.build(&plane(24 * 16, 12, 15));
    // Find the CFAPattern value (00 01 01 02 inline in its entry) and make it 00 00 01 02.
    let at = bytes
        .windows(12)
        .position(|w| w[..2] == [0x8E, 0x82] && w[8..12] == [0, 1, 1, 2])
        .expect("the CFAPattern entry");
    bytes[at + 9] = 0;
    assert_eq!(refusal(&bytes), DngRefusal::NotBayer);
}

#[test]
fn damaged_files_give_errors_and_never_panic() {
    let samples = plane(48 * 32, 12, 16);
    let mut variants = Vec::new();
    for (compression, storage) in [
        (DngCompression::None, DngStorage::Strips(8)),
        (
            DngCompression::LosslessJpeg {
                components: 2,
                predictor: 4,
            },
            DngStorage::Tiles(16, 16),
        ),
    ] {
        let mut b = DngBuilder::new("Acme", "Cam 1", 48, 32);
        b.compression = compression;
        b.storage = storage;
        b.black_delta_h = vec![1.0; 48];
        b.gain_maps = vec![GainMap {
            top: 0,
            left: 0,
            bottom: 32,
            right: 48,
            plane: 0,
            planes: 1,
            row_pitch: 1,
            col_pitch: 1,
            points_v: 2,
            points_h: 2,
            spacing_v: 1.0,
            spacing_h: 1.0,
            origin_v: 0.0,
            origin_h: 0.0,
            map_planes: 1,
            gains: vec![1.0, 1.1, 1.2, 1.3],
        }];
        variants.push(b.build(&samples));
    }
    let mut rng = SplitMix64(99);
    for file in &variants {
        // A file cut before the end of its last directory's picture data is an error; one cut only
        // in the last values (the software string, say) still decodes. Neither panics.
        for keep in (0..file.len()).step_by(37) {
            let result = decode_bytes(&file[..keep], &DecodeOptions::default());
            if keep + 800 < file.len() {
                assert!(result.is_err(), "kept {keep} of {}", file.len());
            }
        }
        // Byte flips never panic; whatever decodes is the right size.
        for _ in 0..600 {
            let mut copy = file.clone();
            for _ in 0..1 + rng.below(6) {
                let at = rng.below(copy.len() as u64) as usize;
                copy[at] = rng.next_u64() as u8;
            }
            if let Ok(image) = decode_bytes(&copy, &DecodeOptions::default()) {
                assert_eq!(image.data.data.len(), image.data.width * image.data.height);
            }
        }
    }
}

#[test]
fn an_arw_with_dng_tags_in_its_raw_directory_is_still_an_arw() {
    // The Sony ARW writer puts DefaultCropOrigin and WhiteLevel in the raw IFD but no DNGVersion:
    // it is not taken for a DNG.
    let bytes = ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", 64, 48).build(&vec![600u16; 64 * 48]);
    let meta = probe_bytes(&bytes).expect("an ARW");
    assert!(meta.dng.is_none());
    assert_eq!(meta.model_id, Some(375));
}
