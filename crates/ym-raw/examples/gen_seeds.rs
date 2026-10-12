//! Writes a small set of valid synthetic ARW files, one per container and decoder variant, to seed
//! the fuzz targets:
//!
//! ```text
//! cargo run -p ym-raw --features testgen --example gen_seeds -- crates/ym-raw/fuzz/corpus/probe
//! cargo run -p ym-raw --features testgen --example gen_seeds -- --fuzz-root crates/ym-raw/fuzz/corpus
//! ```
//!
//! The first form writes the files for one corpus folder. The second fills the corpus folders of
//! every target (`probe`, `tiff_walker`, `decode_any`, `ljpeg`, `craw`, `ycbcr`).

use ym_raw::layout::ChromaSubsampling;
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::ljpeg_enc::{EncodeParams, HuffmanChoice, encode};
use ym_raw::testgen::{
    ArwBuilder, ContainerOnly, CrawEncoder, GpsSpec, LosslessL, LosslessMs, Packed12, RawPayload,
    SonyIsoTable, Sr2Variant, ThumbnailIfd, gradient_mosaic, gradient_rgb,
};

fn write_all(dir: &str, seeds: &[(String, Vec<u8>)]) {
    std::fs::create_dir_all(dir).expect("create the output folder");
    for (name, bytes) in seeds {
        std::fs::write(format!("{dir}/{name}"), bytes).expect("write a seed");
        println!("wrote {dir}/{name}");
    }
}

/// Valid lossless JPEG streams of the shapes the decoders read.
fn ljpeg_seeds() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let planes = |p: &EncodeParams| -> Vec<Vec<u16>> {
        (0..p.sampling.len())
            .map(|c| {
                let (w, h) = p.plane_size(c);
                let mut v = 3000u32;
                (0..w * h)
                    .map(|i| {
                        v = (v * 1103 + 12345 + i as u32 * 7) % 16000;
                        (v & 0x3FFF) as u16
                    })
                    .collect()
            })
            .collect()
    };
    for (name, sampling, predictor, restart_rows, huffman, quirk) in [
        (
            "l4_p1",
            vec![(1u8, 1u8); 4],
            1u8,
            0usize,
            HuffmanChoice::Optimal,
            false,
        ),
        (
            "l4_p4_rst",
            vec![(1, 1); 4],
            4,
            2,
            HuffmanChoice::Optimal,
            false,
        ),
        (
            "l4_p6_skewed",
            vec![(1, 1); 4],
            6,
            0,
            HuffmanChoice::Skewed,
            false,
        ),
        (
            "ms420",
            vec![(2, 2), (1, 1), (1, 1)],
            1,
            0,
            HuffmanChoice::Optimal,
            true,
        ),
        (
            "ms422",
            vec![(2, 1), (1, 1), (1, 1)],
            1,
            0,
            HuffmanChoice::Optimal,
            false,
        ),
        ("c1_p7", vec![(1, 1)], 7, 1, HuffmanChoice::Optimal, false),
    ] {
        let mut p = EncodeParams {
            precision: 14,
            predictor,
            point_transform: 0,
            restart_interval: 0,
            sony_420_quirk: quirk,
            huffman,
            width: 32,
            height: 16,
            sampling,
        };
        p.restart_interval = restart_rows * p.mcus_x();
        out.push((format!("{name}.jpg"), encode(&planes(&p), &p)));
    }
    out
}

/// Seeds for the cRAW target: width byte, height byte, the four knots, then block data.
fn craw_seeds() -> Vec<(String, Vec<u8>)> {
    let encoder = CrawEncoder::new();
    let mosaic = gradient_mosaic(64, 2, 512, encoder.max_value().min(16000), 5);
    let payload = encoder.encode(&mosaic, 64, 2).remove(0);
    let mut seed = vec![1u8, 1];
    for knot in encoder.knots {
        seed.extend_from_slice(&knot.to_le_bytes());
    }
    seed.extend_from_slice(&payload);
    vec![("craw_gradient.bin".to_owned(), seed)]
}

/// Seeds for the M/S target: a subsampling byte, then the tile payload of a valid file.
fn ycbcr_seeds() -> Vec<(String, Vec<u8>)> {
    [
        ChromaSubsampling::Yuv420,
        ChromaSubsampling::Yuv422,
        ChromaSubsampling::Yuv444,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, sub)| {
        let rgb = gradient_rgb(64, 64, 1100, 9000, 1);
        let payload = LosslessMs::new(sub, rgb, 64, 64, [2.0, 1.0, 1.6]).tile(32, 32);
        let file = ArwBuilder::new("ILCE-7RM5", 64, 64)
            .lossless_ms(payload)
            .build(&[]);
        let meta = probe_bytes(&file).expect("valid template");
        let start = meta.storage.chunks().0[0] as usize;
        let mut seed = vec![i as u8];
        seed.extend_from_slice(&file[start..]);
        (format!("ms_{}.bin", sub.name().replace(':', "")), seed)
    })
    .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, fuzz_root) = match &args[..] {
        [flag, root] if flag == "--fuzz-root" => (None, Some(root.clone())),
        [dir] => (Some(dir.clone()), None),
        _ => {
            eprintln!("usage: gen_seeds DIR | gen_seeds --fuzz-root CORPUS_ROOT");
            std::process::exit(2);
        }
    };
    let mosaic = gradient_mosaic(64, 48, 512, 16_000, 1);
    let seeds: Vec<(&str, Vec<u8>)> = vec![
        (
            "unc12_words.arw",
            ArwBuilder::new("ILCE-7M2", 64, 48)
                .uncompressed(12)
                .build(&gradient_mosaic(64, 48, 128, 4000, 2)),
        ),
        (
            "packed12.arw",
            ArwBuilder::new("DSLR-A900", 64, 48)
                .payload(Packed12)
                .build(&gradient_mosaic(64, 48, 128, 4000, 3)),
        ),
        (
            "craw_real.arw",
            ArwBuilder::new("ILCE-7M3", 64, 48)
                .craw(CrawEncoder::new())
                .build(&gradient_mosaic(64, 48, 512, 16_000, 4)),
        ),
        (
            "lossless_l_real.arw",
            ArwBuilder::new("ILCE-1", 128, 96)
                .payload(LosslessL::new().tile(64, 64))
                .build(&gradient_mosaic(128, 96, 512, 16_000, 5)),
        ),
        (
            "lossless_l_rst_p4.arw",
            ArwBuilder::new("ILCE-1", 128, 96)
                .payload(LosslessL::new().tile(64, 64).predictor(4).restart_rows(2))
                .build(&gradient_mosaic(128, 96, 512, 16_000, 6)),
        ),
        (
            "lossless_ms_420.arw",
            ArwBuilder::new("ILCE-7RM5", 96, 64)
                .lossless_ms(
                    LosslessMs::new(
                        ChromaSubsampling::Yuv420,
                        gradient_rgb(96, 64, 1100, 9000, 7),
                        96,
                        64,
                        [2.0, 1.0, 1.6],
                    )
                    .tile(32, 32),
                )
                .build(&[]),
        ),
        (
            "a7r4.arw",
            ArwBuilder::new("ILCE-7RM4", 64, 48).build(&mosaic),
        ),
        (
            "a1_raw_ifd_levels.arw",
            ArwBuilder::new("ILCE-1", 64, 48)
                .raw_ifd_levels(true)
                .iso(102_400)
                .active_area(60, 44)
                .build(&mosaic),
        ),
        (
            "a7m3_legacy_sr2.arw",
            ArwBuilder::new("ILCE-7M3", 64, 48)
                .sr2(Sr2Variant::Legacy)
                .thumbnail_ifd(ThumbnailIfd::Ifd0)
                .iso_table(SonyIsoTable::Tag9405)
                .gps(GpsSpec {
                    latitude_deg: 51.5,
                    longitude_deg: -0.12,
                    altitude_m: 11.0,
                })
                .build(&mosaic),
        ),
        (
            // The older maker-note layout with the 12-byte header.
            "a7m3_sony_dsc_header.arw",
            ArwBuilder::new("ILCE-7M3", 64, 48)
                .maker_note_header(b"SONY DSC \0\0\0")
                .build(&mosaic),
        ),
        (
            // Laid out like the real ILCE-9 file of github.com/lclevy/sony_raw.
            "a9_like_the_dump.arw",
            ArwBuilder::new("ILCE-9", 64, 48)
                .iso(2500)
                .sony_iso_word(2890)
                .white([15360; 3])
                .wb(Some([2364, 1024, 1024, 1728]))
                .raw_ifd_levels(true)
                .raw_ifd_white_level(16383)
                .file_format([3, 3, 3, 0])
                .tone_curve(Some([8000, 10_400, 12_900, 14_100]))
                .thumbnail_ifd(ThumbnailIfd::Ifd0)
                .lossless_thumbnail(true)
                .build(&mosaic),
        ),
        (
            "craw.arw",
            ArwBuilder::new("ILCE-7M4", 64, 48)
                .container_only(ContainerOnly::craw())
                .tone_curve(Some([1000, 4000, 8000, 12000]))
                .build(&[]),
        ),
        (
            "lossless_l.arw",
            ArwBuilder::new("ILCE-1", 1100, 700)
                .container_only(ContainerOnly::lossless_l())
                .build(&[]),
        ),
        (
            "lossless_ms.arw",
            ArwBuilder::new("ILCE-7RM5", 640, 480)
                .container_only(ContainerOnly::lossless_ms(ChromaSubsampling::Yuv422, false))
                .build(&[]),
        ),
        (
            "arw6.arw",
            ArwBuilder::new("ILCE-7M5", 64, 48)
                .container_only(ContainerOnly::arw6(12))
                .build(&[]),
        ),
        (
            "arq.arw",
            ArwBuilder::new("ILCE-7RM4", 64, 48)
                .container_only(ContainerOnly::arq())
                .build(&[]),
        ),
    ];
    let seeds: Vec<(String, Vec<u8>)> = seeds
        .into_iter()
        .map(|(name, bytes)| (name.to_owned(), bytes))
        .collect();
    if let Some(dir) = dir {
        write_all(&dir, &seeds);
    }
    if let Some(root) = fuzz_root {
        for target in ["probe", "tiff_walker", "decode_any"] {
            write_all(&format!("{root}/{target}"), &seeds);
        }
        write_all(&format!("{root}/ljpeg"), &ljpeg_seeds());
        write_all(&format!("{root}/craw"), &craw_seeds());
        write_all(&format!("{root}/ycbcr"), &ycbcr_seeds());
    }
}
