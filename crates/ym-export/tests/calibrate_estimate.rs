//! Measures the compression ratios behind `estimate_bytes` on natural photographs.
//!
//! Ignored by default because it needs the Kodak test images in `testdata/demosaic/kodak/` (downloaded
//! by the demosaic fetch script; git-ignored). Run with
//! `cargo test -p ym-export --test calibrate_estimate -- --ignored --nocapture`.
//! The Kodak images are 768 x 512 film scans, so they carry more fine detail per pixel than a 24 to
//! 61 MP camera frame: the measured sizes are an upper bound for a full-size export.

mod common;

use std::path::PathBuf;

use ym_export::*;

fn load_kodak() -> Vec<(String, u32, u32, Vec<f32>)> {
    let dir = common::repo_root().join("testdata/demosaic/kodak");
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let decoder =
                png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&p).unwrap()));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            assert_eq!(info.color_type, png::ColorType::Rgb);
            assert_eq!(info.bit_depth, png::BitDepth::Eight);
            let pixels: Vec<f32> = buf[..info.buffer_size()]
                .iter()
                .map(|&v| f32::from(v) / 255.0)
                .collect();
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                info.width,
                info.height,
                pixels,
            )
        })
        .collect()
}

#[test]
#[ignore = "needs testdata/demosaic/kodak; run with --ignored --nocapture to print the table"]
fn compression_ratios_on_the_kodak_images() {
    let images = load_kodak();
    if images.is_empty() {
        eprintln!("no Kodak images in testdata/demosaic/kodak");
        return;
    }
    let meta = common::metadata();
    // `noise` is the standard deviation, in code units of 0..1, added before a 16-bit export (a real
    // 16-bit frame is not made of 256 distinct levels like an 8-bit scan scaled up).
    let encode = |format: ExportFormat, noise: f32| -> (f64, f64) {
        // (bits per pixel, ratio to the raw size)
        let mut bpp = 0.0;
        let mut ratio = 0.0;
        for (_, w, h, px) in &images {
            let mut rng = common::Rng(7);
            let noisy: Vec<f32> = px
                .iter()
                .map(|&v| {
                    // Sum of two uniforms: a triangular distribution with the requested deviation.
                    let n = (rng.unit() + rng.unit() - 1.0) * noise * 2.449;
                    (v + n).clamp(0.0, 1.0)
                })
                .collect();
            let req = ExportRequest {
                pixels: ImageRef::new(*w, *h, &noisy, PixelEncoding::DisplayEncoded).unwrap(),
                format,
                color_space: ColorSpace::Srgb,
                size: Size::Full,
                metadata: &meta,
                remove_location: false,
                dng: None,
            };
            let bytes = encode_to_vec(&req, &ExportHooks::none()).unwrap().0.len() as f64;
            let n = f64::from(*w) * f64::from(*h);
            bpp += bytes * 8.0 / n;
            let raw_bits = match format {
                ExportFormat::Tiff { bits, .. } | ExportFormat::Png { bits } => {
                    f64::from(bits.bits())
                }
                _ => 8.0,
            };
            ratio += bytes / (n * 3.0 * raw_bits / 8.0);
        }
        (bpp / images.len() as f64, ratio / images.len() as f64)
    };
    println!("{} Kodak images", images.len());
    for q in [50_u8, 60, 75, 85, 89, 90, 92, 95, 100] {
        // Fractional values, as a rendered image has them, so that the dither has something to do.
        let (bpp, _) = encode(
            ExportFormat::Jpeg {
                quality: JpegQuality::new(q).unwrap(),
            },
            0.29 / 255.0,
        );
        println!("jpeg q{q:<3} {bpp:6.2} bits/pixel");
    }
    for (bits, noise) in [
        (BitDepth::Eight, 0.0),
        (BitDepth::Eight, 0.29 / 255.0),
        (BitDepth::Sixteen, 0.0),
        (BitDepth::Sixteen, 1.0 / 16384.0),
        (BitDepth::Sixteen, 1.0 / 4096.0),
        (BitDepth::Sixteen, 1.0 / 1024.0),
    ] {
        let mut line = format!("{:>2}-bit, noise sigma {noise:.1e}:", bits.bits());
        for c in [TiffCompression::Lzw, TiffCompression::Zip] {
            let (_, r) = encode(
                ExportFormat::Tiff {
                    bits,
                    compression: c,
                },
                noise,
            );
            line.push_str(&format!("  tiff {c:?} {r:.3}"));
        }
        let (_, r) = encode(ExportFormat::Png { bits }, noise);
        line.push_str(&format!("  png {r:.3}  (of raw size)"));
        println!("{line}");
    }
}
