//! The measurements behind the size estimate (`ym_export::estimate`, `docs/09-export.md`, Size estimate).
//!
//! Every test here is ignored by default (they need local data and take minutes). Run one with
//! `cargo test -p ym-export --test calibrate_estimate -- --ignored --nocapture <name>`.
//!
//! * `measure_renders`, `measure_library_detail` and `report_estimate_errors` are the calibration on Yellowmoon
//!   renders: the first two write the CSVs that `tests/py/fit_estimate.py` fits the model from, the third reports
//!   the error of the estimate as it is in the code. They need `YM_ESTIMATE_RENDERS`, a folder of 16-bit sRGB PNG
//!   renders (`ym render --format png --bits 16`).
//! * `compression_ratios_on_the_kodak_images` is the original calibration, kept as a cross-check: it needs the Kodak
//!   test images in `testdata/demosaic/kodak/` (downloaded by the demosaic fetch script; git-ignored). They are
//!   768 x 512 film scans, so they carry more fine detail per pixel than a 24 to 61 MP camera frame.

mod common;

use std::path::PathBuf;

use ym_export::estimate::DetailStat;
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

// ---- calibration on renders -------------------------------------------------------------------------------------

/// Reads a 16-bit RGB PNG (a Yellowmoon render made with `ym render --format png --bits 16`) as
/// display-encoded `f32` in 0..1.
fn load_png16(path: &std::path::Path) -> (u32, u32, Vec<f32>) {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgb, "{}", path.display());
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen, "{}", path.display());
    let pixels: Vec<f32> = buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|b| f32::from(u16::from_be_bytes([b[0], b[1]])) / 65535.0)
        .collect();
    (info.width, info.height, pixels)
}

/// Area average over `k x k` cells (the remainder at the right and bottom edges is dropped).
fn box_down(w: usize, h: usize, px: &[f32], k: usize) -> (usize, usize, Vec<f32>) {
    if k <= 1 {
        return (w, h, px.to_vec());
    }
    let (tw, th) = (w / k, h / k);
    let mut out = vec![0.0f32; tw * th * 3];
    let inv = 1.0 / (k * k) as f32;
    for ty in 0..th {
        for tx in 0..tw {
            let mut acc = [0.0f32; 3];
            for y in ty * k..(ty + 1) * k {
                let row = &px[(y * w + tx * k) * 3..(y * w + tx * k + k) * 3];
                for p in row.chunks_exact(3) {
                    acc[0] += p[0];
                    acc[1] += p[1];
                    acc[2] += p[2];
                }
            }
            let o = (ty * tw + tx) * 3;
            out[o] = acc[0] * inv;
            out[o + 1] = acc[1] * inv;
            out[o + 2] = acc[2] * inv;
        }
    }
    (tw, th, out)
}

/// Candidate detail statistics of an image (all on encoded 0..1 values).
fn candidate_stats(w: usize, h: usize, px: &[f32]) -> [f64; 6] {
    // per-pixel gradient magnitude: mean over channels of (|dx| + |dy|) / 2, forward differences
    let mut grads: Vec<f32> = Vec::with_capacity((w - 1) * (h - 1));
    let mut lap_sum = 0.0f64;
    let mut hf_sum = 0.0f64;
    let mut luma_sum = 0.0f64;
    let at = |x: usize, y: usize, c: usize| px[(y * w + x) * 3 + c];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut g = 0.0f32;
            for c in 0..3 {
                let v = at(x, y, c);
                g += ((at(x + 1, y, c) - v).abs() + (at(x, y + 1, c) - v).abs()) * 0.5;
                let lap =
                    at(x - 1, y, c) + at(x + 1, y, c) + at(x, y - 1, c) + at(x, y + 1, c) - 4.0 * v;
                lap_sum += f64::from(lap.abs());
                let mut m = 0.0f32;
                for dy in 0..3 {
                    for dx in 0..3 {
                        m += at(x + dx - 1, y + dy - 1, c);
                    }
                }
                let d = f64::from(v - m / 9.0);
                hf_sum += d * d;
                luma_sum += f64::from(v);
            }
            grads.push(g / 3.0);
        }
    }
    let n = grads.len() as f64;
    let mean = grads.iter().map(|&g| f64::from(g)).sum::<f64>() / n;
    grads.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = f64::from(grads[grads.len() / 2]);
    let p90 = f64::from(grads[grads.len() * 9 / 10]);
    [
        mean,
        median,
        p90,
        lap_sum / (n * 3.0),
        (hf_sum / (n * 3.0)).sqrt(),
        luma_sum / (n * 3.0),
    ]
}

fn all_variants() -> Vec<(String, ExportFormat)> {
    let mut v = Vec::new();
    for q in [50_u8, 60, 75, 80, 85, 89, 90, 92, 95, 100] {
        v.push((
            format!("jpeg,{q}"),
            ExportFormat::Jpeg {
                quality: JpegQuality::new(q).unwrap(),
            },
        ));
    }
    for bits in [BitDepth::Eight, BitDepth::Sixteen] {
        for (name, compression) in [("lzw", TiffCompression::Lzw), ("zip", TiffCompression::Zip)] {
            v.push((
                format!("tiff_{name},{}", bits.bits()),
                ExportFormat::Tiff { bits, compression },
            ));
        }
        v.push((format!("png,{}", bits.bits()), ExportFormat::Png { bits }));
    }
    v
}

/// Re-encodes every 16-bit PNG render in `$YM_ESTIMATE_RENDERS` into all formats at every size of
/// `$YM_ESTIMATE_SIZES` (comma separated long edges in pixels, `full` for the render's own size; default `full`)
/// and writes one CSV row per (image, size, format) to `$YM_ESTIMATE_CSV`: the size of the file, the estimate
/// the code gives for it (column `old_estimate`: the estimate of the code that ran, without a preview; the refit
/// of October 2026 was measured while that was still the table of the Kodak scans, which is the "Before" column
/// of `docs/09`) and the detail statistics. The fit is done in Python from this file (`tests/py/fit_estimate.py`,
/// `docs/09`, size estimate).
#[test]
#[ignore = "needs YM_ESTIMATE_RENDERS (a folder of 16-bit sRGB PNG renders) and YM_ESTIMATE_CSV; takes minutes"]
fn measure_renders() {
    use rayon::prelude::*;
    use std::fmt::Write as _;
    let (Some(dir), Some(csv)) = (
        std::env::var_os("YM_ESTIMATE_RENDERS"),
        std::env::var_os("YM_ESTIMATE_CSV"),
    ) else {
        eprintln!("set YM_ESTIMATE_RENDERS and YM_ESTIMATE_CSV");
        return;
    };
    let sizes: Vec<Size> = std::env::var("YM_ESTIMATE_SIZES")
        .unwrap_or_else(|_| "full".into())
        .split(',')
        .map(|s| match s.trim() {
            "full" => Size::Full,
            px => Size::LongEdge {
                px: px.parse().unwrap(),
            },
        })
        .collect();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();
    let meta = common::metadata();
    // `YM_ESTIMATE_FORMATS=tiff,png` limits the run to the variants whose label starts with one of the words.
    let only: Option<Vec<String>> = std::env::var("YM_ESTIMATE_FORMATS")
        .ok()
        .map(|v| v.split(',').map(|w| w.trim().to_string()).collect());
    let variants: Vec<(String, ExportFormat)> = all_variants()
        .into_iter()
        .filter(|(label, _)| {
            only.as_ref()
                .is_none_or(|o| o.iter().any(|w| label.starts_with(w.as_str())))
        })
        .collect();
    let mut out = String::from(
        "image,width,height,out_width,out_height,format,param,bytes,old_estimate,\
         t240_gmean,t240_gmed,t240_g90,t240_lap,t240_hf,t240_luma,\
         t480_gmean,t480_gmed,t480_g90,t480_lap,t480_hf,t480_luma,\
         full_gmean,full_gmed,full_g90,full_lap,full_hf,full_luma\n",
    );
    for path in files {
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .trim_end_matches("_neutral")
            .to_string();
        let started = std::time::Instant::now();
        let (w, h, px) = load_png16(&path);
        let (wu, hu) = (w as usize, h as usize);
        let long = wu.max(hu);
        let stats_at = |target: usize| {
            let k = long.div_ceil(target);
            let (tw, th, t) = box_down(wu, hu, &px, k);
            candidate_stats(tw, th, &t)
        };
        let (s240, s480) = (stats_at(240), stats_at(480));
        let full = if long > 3000 {
            // The full-resolution statistic on a 2000 x 2000 crop from the middle (the whole frame is slow here).
            let (cx, cy) = (wu / 2 - 1000, hu / 2 - 1000);
            let mut crop = Vec::with_capacity(2000 * 2000 * 3);
            for y in cy..cy + 2000 {
                crop.extend_from_slice(&px[(y * wu + cx) * 3..(y * wu + cx + 2000) * 3]);
            }
            candidate_stats(2000, 2000, &crop)
        } else {
            candidate_stats(wu, hu, &px)
        };
        for size in &sizes {
            let (ow, oh) = size.output_dimensions(w, h);
            // Resize once (the writers would do it for every format), then encode the display-linear result.
            let source = ImageRef::new(w, h, &px, PixelEncoding::DisplayEncoded).unwrap();
            let resized = (ow != w || oh != h).then(|| {
                ym_export::resize::resize_lanczos3(
                    &source,
                    ow,
                    oh,
                    ColorSpace::Srgb.output_space(),
                    &ExportHooks::none(),
                )
                .unwrap()
            });
            let pixels = resized.as_ref().map_or(source, |r| r.as_ref());
            let rows: Vec<u64> = variants
                .par_iter()
                .map(|(_, format)| {
                    let req = ExportRequest {
                        pixels,
                        format: *format,
                        color_space: ColorSpace::Srgb,
                        size: Size::Full,
                        metadata: &meta,
                        remove_location: false,
                        dng: None,
                    };
                    encode_to_vec(&req, &ExportHooks::none()).unwrap().0.len() as u64
                })
                .collect();
            for (bytes, (label, format)) in rows.iter().zip(&variants) {
                let old = estimate_bytes(w, h, *format, *size);
                let _ = write!(out, "{name},{w},{h},{ow},{oh},{label},{bytes},{old}");
                for s in s240.iter().chain(&s480).chain(&full) {
                    let _ = write!(out, ",{s:.8}");
                }
                out.push('\n');
            }
        }
        eprintln!(
            "{name}: {w} x {h}, {:.1} s",
            started.elapsed().as_secs_f64()
        );
    }
    std::fs::write(csv, out).unwrap();
}

/// The detail statistic the app would compute for each render, from its 8-bit thumbnail (the render box-averaged to
/// at most 240 px on the long edge and quantised to 8 bits, like the thumbnails of `ym-core`) and, as a cross-check,
/// straight from the full-size `f32` picture. Writes `image,detail_u8,detail_f32` to `$YM_ESTIMATE_DETAIL_CSV`.
#[test]
#[ignore = "needs YM_ESTIMATE_RENDERS and YM_ESTIMATE_DETAIL_CSV"]
fn measure_library_detail() {
    let (Some(dir), Some(csv)) = (
        std::env::var_os("YM_ESTIMATE_RENDERS"),
        std::env::var_os("YM_ESTIMATE_DETAIL_CSV"),
    ) else {
        eprintln!("set YM_ESTIMATE_RENDERS and YM_ESTIMATE_DETAIL_CSV");
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();
    let mut out = String::from("image,detail_u8,detail_f32\n");
    for path in files {
        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .trim_end_matches("_neutral")
            .to_string();
        let (w, h, px) = load_png16(&path);
        let (wu, hu) = (w as usize, h as usize);
        let k = wu
            .max(hu)
            .div_ceil(ym_export::estimate::ANALYSIS_LONG_EDGE_PX as usize);
        let (tw, th, thumb) = box_down(wu, hu, &px, k);
        let thumb8: Vec<u8> = thumb
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        if let Some(thumbs) = std::env::var_os("YM_ESTIMATE_THUMBS") {
            // For trying other statistics offline: the 8-bit thumbnail as raw RGB, the size in the name.
            let file = format!("{name}__{tw}x{th}.rgb8");
            std::fs::write(PathBuf::from(thumbs).join(file), &thumb8).unwrap();
        }
        let d8 = DetailStat::measure_rgb8(tw as u32, th as u32, &thumb8).unwrap();
        let d32 = DetailStat::measure_encoded_f32(w, h, &px).unwrap();
        out.push_str(&format!(
            "{name},{:.8},{:.8}\n",
            d8.gradient(),
            d32.gradient()
        ));
        eprintln!(
            "{name}: u8 thumbnail {:.5}, f32 full {:.5}",
            d8.gradient(),
            d32.gradient()
        );
    }
    std::fs::write(csv, out).unwrap();
}

/// Median and 90th percentile of |estimate / actual - 1| and of estimate / actual for every format variant, for the
/// estimate as it is in the code, read from the CSVs of `measure_renders` and `measure_library_detail`. The figures
/// in `docs/09` come from here.
#[test]
#[ignore = "needs YM_ESTIMATE_CSV and YM_ESTIMATE_DETAIL_CSV"]
fn report_estimate_errors() {
    use std::collections::BTreeMap;
    let (Some(csv), Some(detail_csv)) = (
        std::env::var_os("YM_ESTIMATE_CSV"),
        std::env::var_os("YM_ESTIMATE_DETAIL_CSV"),
    ) else {
        eprintln!("set YM_ESTIMATE_CSV and YM_ESTIMATE_DETAIL_CSV");
        return;
    };
    let details: BTreeMap<String, f64> = std::fs::read_to_string(detail_csv)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].to_string(), f[1].parse().unwrap())
        })
        .collect();
    let text = std::fs::read_to_string(csv).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (c_image, c_w, c_h, c_ow, c_oh, c_fmt, c_param, c_bytes, c_old) = (
        col("image"),
        col("width"),
        col("height"),
        col("out_width"),
        col("out_height"),
        col("format"),
        col("param"),
        col("bytes"),
        col("old_estimate"),
    );
    // (format, param, full size only) -> ratios of old / with detail / typical detail to actual
    type Ratios = (Vec<f64>, Vec<f64>, Vec<f64>);
    let mut table: BTreeMap<(String, u32, bool), Ratios> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    // Bytes of the full-size JPEGs at quality 92 of the real renders: actual, old estimate, with detail, typical.
    let mut totals = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let (w, h): (u32, u32) = (f[c_w].parse().unwrap(), f[c_h].parse().unwrap());
        let (ow, oh): (u32, u32) = (f[c_ow].parse().unwrap(), f[c_oh].parse().unwrap());
        let param: u32 = f[c_param].parse().unwrap();
        if !seen.insert((f[c_image].to_string(), ow, oh, f[c_fmt].to_string(), param)) {
            continue; // an export size at or above the render's own repeats the full size
        }
        let bits = |p: u32| {
            if p == 16 {
                BitDepth::Sixteen
            } else {
                BitDepth::Eight
            }
        };
        let format = match f[c_fmt] {
            "jpeg" => ExportFormat::Jpeg {
                quality: JpegQuality::new(param as u8).unwrap(),
            },
            "tiff_lzw" => ExportFormat::Tiff {
                bits: bits(param),
                compression: TiffCompression::Lzw,
            },
            "tiff_zip" => ExportFormat::Tiff {
                bits: bits(param),
                compression: TiffCompression::Zip,
            },
            "png" => ExportFormat::Png { bits: bits(param) },
            other => panic!("{other}"),
        };
        let actual: f64 = f[c_bytes].parse().unwrap();
        let old: f64 = f[c_old].parse().unwrap();
        let detail = DetailStat::from_gradient(details[f[c_image]]).unwrap();
        let size = if (ow, oh) == (w, h) {
            Size::Full
        } else {
            Size::LongEdge { px: ow.max(oh) }
        };
        let with = estimate_bytes_with_detail(w, h, format, size, Some(detail)) as f64;
        let typical = estimate_bytes(w, h, format, size) as f64;
        let real = !f[c_image].starts_with("chart");
        if real && f[c_fmt] == "jpeg" && param == 92 && (ow, oh) == (w, h) {
            totals.0 += actual;
            totals.1 += old;
            totals.2 += with;
            totals.3 += typical;
        }
        for key in [
            (f[c_fmt].to_string(), param, true),
            (f[c_fmt].to_string(), param, false),
        ] {
            // `true`: the nine real renders at every export size; `false`: everything.
            if key.2 && !real {
                continue;
            }
            let entry = table.entry(key).or_default();
            entry.0.push(old / actual);
            entry.1.push(with / actual);
            entry.2.push(typical / actual);
        }
    }
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Quantiles with linear interpolation between ranks (numpy's default).
        let quantile = |sorted: &[f64], q: f64| {
            let pos = q * (sorted.len() - 1) as f64;
            let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
            sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
        };
        let median = quantile(v, 0.5);
        let mut dev: Vec<f64> = v.iter().map(|r| (r - 1.0).abs()).collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = quantile(&dev, 0.9);
        format!("median {median:5.2} p90|e-1| {p90:5.2}")
    };
    println!(
        "real renders, full size, JPEG q92: actual {:.1} MB, old estimate {:.1} MB, with detail {:.1} MB, typical detail {:.1} MB",
        totals.0 / 1e6,
        totals.1 / 1e6,
        totals.2 / 1e6,
        totals.3 / 1e6
    );
    println!("estimate / actual, real renders only (R) and real + synthetic (A); n = rows");
    for ((fmt, param, real_only), (mut old, mut with, mut typical)) in table {
        println!(
            "{} {fmt:<9} {param:>3} n={:<3} old: {} | with detail: {} | typical detail: {}",
            if real_only { "R" } else { "A" },
            old.len(),
            stats(&mut old),
            stats(&mut with),
            stats(&mut typical),
        );
    }
}
