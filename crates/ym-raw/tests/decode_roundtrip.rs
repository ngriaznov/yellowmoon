//! End-to-end decoder tests on synthetic ARW files (M0-05, M2-09): every variant is written by the
//! test-only encoders in `ym_raw::testgen`, read back through `ym_raw::decode`, and compared with
//! the mosaic (or RGB image) it came from. Lossless variants must match exactly.

use std::path::PathBuf;

use ym_raw::decode::ycbcr::{CHROMA_CENTER, ycc_to_rgb};
use ym_raw::decode::{
    ChromaUpsampling, DecodeOptions, RawImage, YcbcrOptions, decode, decode_bytes, decode_with,
};
use ym_raw::error::RawError;
use ym_raw::layout::{ChromaSubsampling, DecodeKind, Layout};
use ym_raw::testgen::{
    Arw1, ArwBuilder, ContainerOnly, CrawEncoder, HuffmanChoice, LosslessL, LosslessMs, Packed12,
    SplitMix64, gradient_mosaic, gradient_rgb,
};

fn must_decode(bytes: &[u8]) -> RawImage {
    decode_bytes(bytes, &DecodeOptions::default()).unwrap_or_else(|e| panic!("decode failed: {e}"))
}

fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write test file");
    path
}

#[test]
fn uncompressed_14_bit_words_come_back_as_written() {
    let (w, h) = (96, 64);
    let mosaic = gradient_mosaic(w, h, 512, 16000, 1);
    let file = ArwBuilder::new("ILCE-7RM4", w, h).build(&mosaic);
    let raw = must_decode(&file);
    assert_eq!(raw.meta.decode, DecodeKind::Uncompressed14);
    assert_eq!(raw.data.data, mosaic);
    assert_eq!((raw.data.width, raw.data.height), (w as usize, h as usize));
    assert!(matches!(raw.layout, Layout::Bayer { .. }));
    assert!(raw.qstep.is_none());
    assert!(!raw.is_reduced());
    assert_eq!((raw.channels(), raw.pixel_width()), (1, w as usize));
    assert_eq!(raw.black, [512.0; 4]);
    assert_eq!(raw.white, [16300.0; 4]);
    assert_eq!(raw.wb_as_shot, Some([2043.0, 1024.0, 1024.0, 1755.0]));
}

#[test]
fn uncompressed_12_bit_words_come_back_as_written() {
    let (w, h) = (64, 48);
    let mosaic = gradient_mosaic(w, h, 128, 4000, 2);
    let file = ArwBuilder::new("ILCE-7RM4", w, h)
        .uncompressed(12)
        .build(&mosaic);
    let raw = must_decode(&file);
    assert_eq!(raw.meta.decode, DecodeKind::Uncompressed12);
    assert_eq!(raw.data.data, mosaic);
    // 14-bit SR2 levels are shifted to the 12-bit scale of the data.
    assert_eq!(raw.black, [128.0; 4]);
    assert_eq!(raw.white, [4075.0; 4]);
}

#[test]
fn packed_12_bit_round_trips() {
    let (w, h) = (64, 40);
    let mosaic = gradient_mosaic(w, h, 128, 4000, 3);
    let file = ArwBuilder::new("DSLR-A900", w, h)
        .payload(Packed12)
        .build(&mosaic);
    let raw = must_decode(&file);
    assert_eq!(raw.meta.decode, DecodeKind::Uncompressed12);
    assert_eq!(raw.meta.compression, 32767);
    assert_eq!(raw.data.data, mosaic);
}

#[test]
fn arw1_huffman_samples_round_trip() {
    let (w, h) = (96, 64);
    let mosaic = gradient_mosaic(w, h, 128, 4000, 12);
    let file = ArwBuilder::new("DSLR-A300", w, h)
        .payload(Arw1)
        .build(&mosaic);
    let raw = must_decode(&file);
    assert_eq!(raw.meta.decode, DecodeKind::Arw1);
    assert_eq!(raw.data.data, mosaic);
    assert_eq!((raw.data.width, raw.data.height), (96, 64));
}

#[test]
fn lossless_l_round_trips_for_every_predictor() {
    // 600 x 520: two tile columns and rows with partial tiles at the right and bottom edges.
    let (w, h) = (600, 520);
    let mosaic = gradient_mosaic(w, h, 512, 16000, 4);
    for predictor in 1..=7u8 {
        let file = ArwBuilder::new("ILCE-1", w, h)
            .payload(LosslessL::new().predictor(predictor))
            .build(&mosaic);
        let raw = must_decode(&file);
        assert_eq!(raw.meta.decode, DecodeKind::LosslessL);
        assert_eq!(raw.data.data, mosaic, "predictor {predictor}");
    }
}

#[test]
fn lossless_l_round_trips_with_restart_markers_and_other_tilings() {
    let (w, h) = (256, 192);
    let mosaic = gradient_mosaic(w, h, 512, 16000, 5);
    for (predictor, restart_rows, tile, huffman) in [
        (1, 0, (512, 512), HuffmanChoice::Optimal),
        (1, 1, (128, 128), HuffmanChoice::Optimal),
        (4, 3, (256, 64), HuffmanChoice::Skewed),
        (7, 2, (64, 64), HuffmanChoice::Optimal),
        (2, 0, (96, 128), HuffmanChoice::Skewed),
    ] {
        let file = ArwBuilder::new("ILCE-7M4", w, h)
            .payload(
                LosslessL::new()
                    .predictor(predictor)
                    .restart_rows(restart_rows)
                    .tile(tile.0, tile.1)
                    .huffman(huffman),
            )
            .build(&mosaic);
        let raw = must_decode(&file);
        assert_eq!(
            raw.data.data, mosaic,
            "predictor {predictor}, restart rows {restart_rows}, tile {tile:?}"
        );
    }
}

#[test]
fn lossless_l_quad_order_is_r_g1_g2_b_across_two_rows() {
    // Distinct values per CFA position make a wrong component-to-quad order visible.
    let (w, h) = (64, 64);
    let mosaic: Vec<u16> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            1000 + 1000 * ((y & 1) * 2 + (x & 1)) as u16 + (x / 2) as u16
        })
        .collect();
    let file = ArwBuilder::new("ILCE-1", w, h)
        .payload(LosslessL::new().tile(64, 64))
        .build(&mosaic);
    let raw = must_decode(&file);
    assert_eq!(raw.data.data, mosaic);
    assert_eq!(raw.data.data[0], 1000);
    assert_eq!(raw.data.data[1], 2000);
    assert_eq!(raw.data.data[w as usize], 3000);
    assert_eq!(raw.data.data[w as usize + 1], 4000);
}

fn craw_mosaic(w: u32, h: u32, seed: u64, encoder: &CrawEncoder) -> Vec<u16> {
    let top = encoder.max_value();
    gradient_mosaic(w, h, 512, top.min(16000), seed)
}

#[test]
fn craw_decodes_to_the_curve_values_with_a_quantisation_map() {
    let (w, h) = (128, 64);
    let encoder = CrawEncoder::new();
    let mosaic = craw_mosaic(w, h, 6, &encoder);
    let file = ArwBuilder::new("ILCE-7M3", w, h)
        .craw(encoder.clone())
        .build(&mosaic);
    let raw = must_decode(&file);
    assert_eq!(raw.meta.decode, DecodeKind::Craw);
    assert_eq!(raw.tone_curve, Some(encoder.knots));
    // Levels stay on the 14-bit scale (512 / 15360 for the usual SR2 values).
    assert_eq!(raw.black, [512.0; 4]);
    let qstep = raw.qstep.as_ref().expect("cRAW has a quantisation map");
    assert_eq!(
        (qstep.width, qstep.height),
        (raw.data.width, raw.data.height)
    );
    for (i, (decoded, truth)) in raw.data.data.iter().zip(&mosaic).enumerate() {
        let step = qstep.data[i].to_f32();
        assert!(step >= 2.0, "pixel {i}: qstep {step}");
        assert!(
            2.0 * (f32::from(*decoded) - f32::from(*truth)).abs() <= step,
            "pixel {i}: true {truth} decoded {decoded} qstep {step}"
        );
    }
}

#[test]
fn craw_decoding_is_deterministic_and_never_dithered() {
    let (w, h) = (64, 16);
    let encoder = CrawEncoder::new();
    let mosaic = craw_mosaic(w, h, 7, &encoder);
    let file = ArwBuilder::new("ILCE-7M3", w, h)
        .craw(encoder.clone())
        .build(&mosaic);
    let a = must_decode(&file);
    let b = must_decode(&file);
    assert_eq!(a.data, b.data);
    // Every decoded value is a point of the curve (an even index).
    let curve = encoder.curve();
    let on_grid: std::collections::HashSet<u16> = (0..2048).map(|c| curve[2 * c]).collect();
    assert!(a.data.data.iter().all(|v| on_grid.contains(v)));
}

#[test]
fn craw_without_a_tone_curve_is_refused() {
    let (w, h) = (64, 16);
    let encoder = CrawEncoder::new();
    let mosaic = craw_mosaic(w, h, 8, &encoder);
    let file = ArwBuilder::new("ILCE-7M3", w, h)
        .craw(encoder)
        .tone_curve(None)
        .build(&mosaic);
    match decode_bytes(&file, &DecodeOptions::default()) {
        Err(RawError::Corrupt { detail }) => assert!(detail.contains("tone curve"), "{detail}"),
        other => panic!("expected Corrupt, got {other:?}"),
    }
}

fn factors(sub: ChromaSubsampling) -> (usize, usize) {
    match sub {
        ChromaSubsampling::Yuv444 => (1, 1),
        ChromaSubsampling::Yuv422 => (2, 1),
        ChromaSubsampling::Yuv420 => (2, 2),
    }
}

/// The RGB the camera-side planes decode to, computed independently of the decoder's tile
/// handling: the chroma of a pixel is looked up in the full-frame chroma plane, replicated or
/// interpolated as documented.
fn expected_rgb(payload: &LosslessMs, mode: ChromaUpsampling) -> Vec<u16> {
    let (w, h) = (payload.width as usize, payload.height as usize);
    let (fh, fv) = factors(payload.subsampling);
    let (y, cb, cr, cw, ch) = payload.planes();
    let at = |plane: &[u16], cx: usize, cy: usize| {
        i32::from(plane[cy.min(ch - 1) * cw + cx.min(cw - 1)])
    };
    let horizontal = |plane: &[u16], x: usize, cy: usize| -> i32 {
        let cx = x / fh;
        if mode == ChromaUpsampling::Replicate || x.is_multiple_of(fh) {
            at(plane, cx, cy)
        } else {
            (at(plane, cx, cy) + at(plane, cx + 1, cy) + 1) / 2
        }
    };
    let chroma = |plane: &[u16], x: usize, yy: usize| -> i32 {
        let cy = yy / fv;
        if mode == ChromaUpsampling::Replicate || yy.is_multiple_of(fv) {
            horizontal(plane, x, cy)
        } else {
            (horizontal(plane, x, cy) + horizontal(plane, x, cy + 1) + 1) / 2
        }
    };
    let mut out = Vec::with_capacity(w * h * 3);
    for yy in 0..h {
        for x in 0..w {
            let rgb = ycc_to_rgb(
                y[yy * w + x],
                chroma(&cb, x, yy) - CHROMA_CENTER,
                chroma(&cr, x, yy) - CHROMA_CENTER,
            );
            out.extend_from_slice(&rgb);
        }
    }
    out
}

fn ms_payload(sub: ChromaSubsampling, w: u32, h: u32, tile: u32, gains: [f32; 3]) -> LosslessMs {
    let rgb = gradient_rgb(w, h, 1100, 9000, 21);
    LosslessMs::new(sub, rgb, w, h, gains).tile(tile, tile)
}

fn ms_file(
    sub: ChromaSubsampling,
    w: u32,
    h: u32,
    tile: (u32, u32),
    gains: [f32; 3],
) -> (Vec<u8>, Vec<u16>) {
    let rgb = gradient_rgb(w, h, 1100, 9000, 21);
    let payload = LosslessMs::new(sub, rgb.clone(), w, h, gains).tile(tile.0, tile.1);
    let file = ArwBuilder::new("ILCE-7RM5", w, h)
        .lossless_ms(payload)
        .build(&[]);
    (file, rgb)
}

#[test]
fn lossless_ms_decodes_to_exactly_what_the_planes_say() {
    for sub in [
        ChromaSubsampling::Yuv420,
        ChromaSubsampling::Yuv422,
        ChromaSubsampling::Yuv444,
    ] {
        // 200 x 136 with 64-pixel tiles: partial tiles on both edges.
        let payload = ms_payload(sub, 200, 136, 64, [2.1, 1.0, 1.7]);
        let file = ArwBuilder::new("ILCE-7RM5", 200, 136)
            .lossless_ms(payload.clone())
            .build(&[]);
        for mode in [ChromaUpsampling::Replicate, ChromaUpsampling::Linear] {
            let raw = decode_bytes(
                &file,
                &DecodeOptions {
                    ycbcr: YcbcrOptions {
                        chroma: mode,
                        undo_white_balance: false,
                    },
                },
            )
            .expect("decode");
            assert_eq!(raw.meta.decode, DecodeKind::LosslessMS { subsampling: sub });
            assert_eq!(raw.layout, Layout::Ycbcr { subsampling: sub });
            assert!(raw.is_reduced());
            assert_eq!(
                (raw.channels(), raw.pixel_width(), raw.pixel_height()),
                (3, 200, 136)
            );
            assert_eq!(raw.data.width, 600);
            assert!(raw.qstep.is_none());
            let want = expected_rgb(&payload, mode);
            let first_diff = raw.data.data.iter().zip(&want).position(|(a, b)| a != b);
            assert!(
                first_diff.is_none(),
                "{sub:?} {mode:?}: first difference at sample {first_diff:?}"
            );
        }
    }
}

#[test]
fn lossless_ms_returns_unbalanced_camera_rgb() {
    for sub in [
        ChromaSubsampling::Yuv420,
        ChromaSubsampling::Yuv422,
        ChromaSubsampling::Yuv444,
    ] {
        let payload = ms_payload(sub, 200, 136, 64, [2.1, 1.0, 1.7]);
        let rgb = payload.rgb.clone();
        let file = ArwBuilder::new("ILCE-7RM5", 200, 136)
            .lossless_ms(payload)
            .build(&[]);
        let raw = must_decode(&file);
        assert_eq!(raw.wb_as_shot, Some([2150.0, 1024.0, 1024.0, 1741.0]));
        // Pedestal is twice the sensor black (1024); the gains were divided out again.
        assert_eq!(raw.black, [1024.0; 4]);
        assert!(raw.white[0] < raw.white[1] && raw.white[3] < raw.white[1]);
        // The RGB comes back to what the camera was given, up to the chroma subsampling (noise of
        // about 120 DN per channel in the test image) and the rounding of the gains.
        let (tolerance, mean_limit) = if sub == ChromaSubsampling::Yuv444 {
            (4u16, 1.0)
        } else {
            (600, 60.0)
        };
        let mut worst = 0u16;
        let mut sum = 0u64;
        for (got, want) in raw.data.data.iter().zip(&rgb) {
            let d = got.abs_diff(*want);
            worst = worst.max(d);
            sum += u64::from(d);
        }
        assert!(worst <= tolerance, "{sub:?}: worst error {worst}");
        let mean = sum as f64 / rgb.len() as f64;
        assert!(mean < mean_limit, "{sub:?}: mean error {mean}");
    }
}

#[test]
fn lossless_ms_levels_follow_the_gains() {
    let gains = [2.0f32, 1.0, 1.5];
    let (file, _) = ms_file(ChromaSubsampling::Yuv420, 64, 64, (64, 64), gains);
    let raw = must_decode(&file);
    // white = pedestal + (17536 - 1024) / gain.
    assert!(
        (raw.white[0] - (1024.0 + 16512.0 / 2.0)).abs() < 0.01,
        "{:?}",
        raw.white
    );
    assert!((raw.white[1] - (1024.0 + 16512.0)).abs() < 0.01);
    assert_eq!(raw.white[1], raw.white[2]);
    assert!((raw.white[3] - (1024.0 + 16512.0 / 1.5)).abs() < 0.01);
}

#[test]
fn lossless_ms_without_white_balance_data_keeps_the_balanced_values() {
    let (w, h) = (64u32, 64u32);
    let rgb = gradient_rgb(w, h, 1100, 9000, 22);
    let payload =
        LosslessMs::new(ChromaSubsampling::Yuv444, rgb, w, h, [2.0, 1.0, 1.5]).tile(64, 64);
    let file = ArwBuilder::new("ILCE-7RM5", w, h)
        .payload(payload)
        .wb(None)
        .build(&[]);
    let raw = must_decode(&file);
    assert!(raw.wb_as_shot.is_none());
    assert_eq!(raw.white[0], raw.white[1]);
}

#[test]
fn files_cut_short_are_corrupt_never_partial() {
    let (w, h) = (128, 64);
    let mosaic = gradient_mosaic(w, h, 512, 16000, 9);
    let files = [
        ArwBuilder::new("ILCE-7RM4", w, h).build(&mosaic),
        ArwBuilder::new("ILCE-7M3", w, h)
            .craw(CrawEncoder::new())
            .build(&craw_mosaic(w, h, 9, &CrawEncoder::new())),
        ArwBuilder::new("ILCE-1", w, h)
            .payload(LosslessL::new().tile(64, 64))
            .build(&mosaic),
    ];
    for file in files {
        for keep in [file.len() - 1, file.len() - 100, file.len() * 3 / 4] {
            let result = decode_bytes(&file[..keep], &DecodeOptions::default());
            assert!(
                matches!(
                    result,
                    Err(RawError::Corrupt { .. }) | Err(RawError::NotArw)
                ),
                "cut to {keep} of {}: {result:?}",
                file.len()
            );
        }
    }
}

#[test]
fn damaged_tile_data_is_corrupt() {
    let (w, h) = (128, 64);
    let mosaic = gradient_mosaic(w, h, 512, 16000, 10);
    let mut file = ArwBuilder::new("ILCE-1", w, h)
        .payload(LosslessL::new().tile(64, 64))
        .build(&mosaic);
    // Overwrite the middle of the tile area with noise: the streams no longer parse or end early.
    let mut rng = SplitMix64(3);
    let n = file.len();
    for byte in &mut file[n - 3000..n - 1500] {
        *byte = rng.next_u64() as u8;
    }
    let result = decode_bytes(&file, &DecodeOptions::default());
    match result {
        Err(RawError::Corrupt { .. }) => {}
        // Noise can still be a valid (different) stream; then it must not equal the original.
        Ok(raw) => assert_ne!(raw.data.data, mosaic),
        Err(other) => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn arw6_and_unknown_layouts_are_unsupported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = ArwBuilder::new("ILCE-7M5", 64, 64)
        .container_only(ContainerOnly::arw6(12))
        .build(&[]);
    let path = write(&dir, "a7v.ARW", &file);
    match decode(&path) {
        Err(RawError::Unsupported { variant, camera }) => {
            assert_eq!(variant, DecodeKind::Arw6);
            assert_eq!(camera, "ILCE-7M5");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
    // A compression value nobody knows is a recognised Sony file in an unrecognised layout.
    let file = ArwBuilder::new("DSLR-A700", 64, 64)
        .container_only(ContainerOnly::other(12345, 12))
        .build(&[]);
    match decode_bytes(&file, &DecodeOptions::default()) {
        Err(RawError::Unsupported { variant, .. }) => {
            assert!(matches!(
                variant,
                DecodeKind::Unknown {
                    compression: 12345,
                    ..
                }
            ));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn decode_reads_files_from_disk_like_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (w, h) = (64, 64);
    let mosaic = gradient_mosaic(w, h, 512, 16000, 11);
    let file = ArwBuilder::new("ILCE-1", w, h)
        .payload(LosslessL::new().tile(64, 64))
        .build(&mosaic);
    let path = write(&dir, "lossless.ARW", &file);
    let from_disk = decode(&path).expect("decode from disk");
    let from_bytes = must_decode(&file);
    assert_eq!(from_disk.data, from_bytes.data);
    let again = decode_with(&path, &DecodeOptions::default()).expect("decode_with");
    assert_eq!(again.data, from_disk.data);
}

// ---------------------------------------------------------------------------------------------
// Several strips (review): every file the writer makes has one strip, but the decoders loop over
// strips of `RowsPerStrip` rows, so this path needs a file of its own.
// ---------------------------------------------------------------------------------------------

/// Rewrites the raw image's strip tables so that it is stored as `rows_per_strip`-row strips, by
/// appending the new offset and byte-count arrays to the file.
fn split_into_strips(
    file: &[u8],
    row_bytes: usize,
    height: usize,
    rows_per_strip: usize,
) -> Vec<u8> {
    let mut out = file.to_vec();
    let u32at =
        |f: &[u8], at: usize| u32::from_le_bytes(f[at..at + 4].try_into().expect("4 bytes"));
    // The raw directory is the one whose StripByteCounts says width * height * k bytes.
    let counts_entry = {
        let mut found = None;
        let mut pending = vec![u32at(file, 4) as usize];
        let mut seen = Vec::new();
        while let Some(at) = pending.pop() {
            if at < 8 || seen.contains(&at) {
                continue;
            }
            seen.push(at);
            let n = usize::from(u16::from_le_bytes([file[at], file[at + 1]]));
            for i in 0..n {
                let e = at + 2 + 12 * i;
                let tag = u16::from_le_bytes([file[e], file[e + 1]]);
                if tag == 0x0117 && u32at(file, e + 8) as usize == row_bytes * height {
                    found = Some(at);
                }
                if tag == 0x014A {
                    let k = u32at(file, e + 4) as usize;
                    if k == 1 {
                        pending.push(u32at(file, e + 8) as usize);
                    } else {
                        let table = u32at(file, e + 8) as usize;
                        pending.extend((0..k).map(|j| u32at(file, table + 4 * j) as usize));
                    }
                }
            }
            pending.push(u32at(file, at + 2 + 12 * n) as usize);
        }
        found.expect("the raw directory")
    };
    let entry = |tag: u16| {
        let n = usize::from(u16::from_le_bytes([
            file[counts_entry],
            file[counts_entry + 1],
        ]));
        (0..n)
            .map(|i| counts_entry + 2 + 12 * i)
            .find(|e| u16::from_le_bytes([file[*e], file[*e + 1]]) == tag)
    };
    let first_offset = u32at(file, entry(0x0111).expect("StripOffsets") + 8) as usize;
    let strips = height.div_ceil(rows_per_strip);
    let table_at = out.len();
    for k in 0..strips {
        out.extend_from_slice(
            &((first_offset + k * rows_per_strip * row_bytes) as u32).to_le_bytes(),
        );
    }
    let counts_at = out.len();
    for k in 0..strips {
        let rows = rows_per_strip.min(height - k * rows_per_strip);
        out.extend_from_slice(&((rows * row_bytes) as u32).to_le_bytes());
    }
    for (tag, at) in [(0x0111u16, table_at), (0x0117, counts_at)] {
        let e = entry(tag).expect("strip tag");
        out[e + 2..e + 4].copy_from_slice(&4u16.to_le_bytes()); // LONG
        out[e + 4..e + 8].copy_from_slice(&(strips as u32).to_le_bytes());
        out[e + 8..e + 12].copy_from_slice(&(at as u32).to_le_bytes());
    }
    let rps = entry(0x0116).expect("RowsPerStrip");
    out[rps + 2..rps + 4].copy_from_slice(&4u16.to_le_bytes());
    out[rps + 4..rps + 8].copy_from_slice(&1u32.to_le_bytes());
    out[rps + 8..rps + 12].copy_from_slice(&(rows_per_strip as u32).to_le_bytes());
    out
}

#[test]
fn files_stored_in_several_strips_decode_like_single_strip_files() {
    let (w, h) = (96usize, 61usize);
    let encoder = CrawEncoder::new();
    let cases: Vec<(&str, Vec<u8>, usize)> = vec![
        (
            "uncompressed 14-bit",
            ArwBuilder::new("ILCE-7RM4", w as u32, h as u32)
                .build(&gradient_mosaic(w as u32, h as u32, 512, 16000, 31)),
            w * 2,
        ),
        (
            "packed 12-bit",
            ArwBuilder::new("DSLR-A900", w as u32, h as u32)
                .payload(Packed12)
                .build(&gradient_mosaic(w as u32, h as u32, 128, 4000, 32)),
            w * 3 / 2,
        ),
        (
            "cRAW",
            ArwBuilder::new("ILCE-7M3", w as u32, h as u32)
                .craw(encoder.clone())
                .build(&gradient_mosaic(
                    w as u32,
                    h as u32,
                    512,
                    encoder.max_value(),
                    33,
                )),
            w,
        ),
    ];
    for (name, file, row_bytes) in cases {
        let whole = must_decode(&file);
        for rows_per_strip in [1usize, 7, 20, 31, 60] {
            let split = split_into_strips(&file, row_bytes, h, rows_per_strip);
            let raw = decode_bytes(&split, &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("{name}, {rows_per_strip} rows per strip: {e}"));
            assert_eq!(
                raw.data, whole.data,
                "{name}, {rows_per_strip} rows per strip"
            );
            assert_eq!(
                raw.qstep, whole.qstep,
                "{name}, {rows_per_strip} rows per strip"
            );
        }
    }
}

/// A smooth colour ramp (no noise): chroma varies slowly, so interpolating it must beat copying it.
fn smooth_rgb(w: u32, h: u32) -> Vec<u16> {
    let mut out = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            out.push((2000 + 40 * x + 10 * y) as u16);
            out.push((3000 + 15 * x + 45 * y) as u16);
            out.push((2500 + 25 * x + 25 * y) as u16);
        }
    }
    out
}

#[test]
fn the_shipped_default_returns_a_smooth_image_to_within_rounding() {
    // The default path (linear chroma, white balance divided out) is the one no oracle reads:
    // LibRaw replicates chroma and leaves the balance in. With co-sited chroma, which is what
    // linear upsampling assumes, a colour ramp survives subsampling, interpolation, the YCbCr
    // matrices and the gains to within rounding (2 DN is what it measures), in every
    // subsampling, with partial tiles and odd sizes.
    for (sub, tolerance) in [
        (ChromaSubsampling::Yuv444, 4u16),
        (ChromaSubsampling::Yuv422, 4),
        (ChromaSubsampling::Yuv420, 4),
    ] {
        let (w, h) = (101u32, 77u32);
        let rgb = smooth_rgb(w, h);
        let payload = LosslessMs::new(sub, rgb.clone(), w, h, [2.1, 1.0, 1.7])
            .tile(32, 32)
            .co_sited(true);
        let file = ArwBuilder::new("ILCE-7RM5", w, h)
            .lossless_ms(payload)
            .build(&[]);
        let raw = must_decode(&file);
        let mut worst = (0u16, 0usize);
        for (i, (got, want)) in raw.data.data.iter().zip(&rgb).enumerate() {
            let d = got.abs_diff(*want);
            if d > worst.0 {
                worst = (d, i);
            }
        }
        // The last row and column have no interpolation partner (the chroma repeats), the rest is
        // interior; both must be close on a ramp.
        assert!(
            worst.0 <= tolerance,
            "{sub:?}: worst error {} DN at sample {} (pixel {}, channel {})",
            worst.0,
            worst.1,
            worst.1 / 3,
            worst.1 % 3
        );
    }
}
