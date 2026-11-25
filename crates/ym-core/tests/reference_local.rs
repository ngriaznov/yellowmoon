//! The letterbox rule and the loader on **local pictures** (`docs/34-reference-grade.md` sections 6.5, 16.5 and 16.6, RG-05). Ignored
//! by default and never run in CI: the pictures are the owner's evaluation set (film stills and renders of public raw photographs),
//! which are **never committed, never cached by CI, never uploaded** (section 6.6). Point `YM_REFERENCE_PICTURES` at a folder
//! (searched recursively for `.jpg` and `.jpeg`) and run
//! `cargo test -p ym-core --test reference_local -- --ignored --nocapture`.
//!
//! For each picture it checks that
//!
//! 1. the picture as it is loses nothing (these are frames without bars, many of them night frames with dark edges that the POC's two
//!    thresholds alone would cut: a line of a dark sky is not a bar);
//! 2. the same picture inside synthetic black bars (a letterbox of 12.5 % each, a pillarbox of 8 %), re-encoded as a JPEG at a low
//!    quality with 4:2:0 sampling (the ringing of a real bar), is trimmed back to exactly its own size.
//!
//! A line per picture says what happened and the totals are printed at the end.

use std::path::{Path, PathBuf};

use ym_color::reduce::reduced_size;
use ym_core::reference::{ANALYSIS_LONG_EDGE_PX, ReferenceError, ReferenceNote, ReferencePicture};
use zune_jpeg::JpegDecoder;

fn jpegs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            jpegs(&p, out);
        } else if p
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("jpg") || x.eq_ignore_ascii_case("jpeg"))
        {
            out.push(p);
        }
    }
}

fn decode(path: &Path) -> Option<(usize, usize, Vec<u8>)> {
    let bytes = std::fs::read(path).ok()?;
    let mut d = JpegDecoder::new(std::io::Cursor::new(bytes));
    let rgb = d.decode().ok()?;
    let info = d.info()?;
    let (w, h) = (usize::from(info.width), usize::from(info.height));
    (rgb.len() == w * h * 3).then_some((w, h, rgb))
}

fn barred(rgb: &[u8], w: usize, h: usize, top: usize, left: usize) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = (w + 2 * left, h + 2 * top);
    let mut d = vec![0u8; ow * oh * 3];
    for y in 0..h {
        let at = ((y + top) * ow + left) * 3;
        d[at..at + w * 3].copy_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    (d, ow, oh)
}

fn jpeg(rgb: &[u8], w: usize, h: usize, quality: u8) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut out, quality);
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
    enc.encode(rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb)
        .expect("encode");
    out
}

#[test]
#[ignore = "local pictures only: set YM_REFERENCE_PICTURES to a folder of JPEGs"]
fn local_pictures_lose_nothing_and_their_synthetic_bars_are_trimmed_exactly() {
    let Some(dir) = std::env::var_os("YM_REFERENCE_PICTURES") else {
        eprintln!("YM_REFERENCE_PICTURES is not set");
        return;
    };
    let mut files = Vec::new();
    jpegs(Path::new(&dir), &mut files);
    files.sort();
    let (mut tried, mut cut, mut wrong, mut small, mut empty, mut unread) = (0, 0, 0, 0, 0, 0);
    for path in &files {
        let name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let Some((w, h, rgb)) = decode(path) else {
            unread += 1;
            continue;
        };
        // 1. As it is.
        match ReferencePicture::from_file(path) {
            Ok(p) => {
                tried += 1;
                if p.notes().contains(&ReferenceNote::BlackBarsRemoved) {
                    cut += 1;
                    println!("CUT  {name} ({w} x {h}) lost bars it does not have");
                }
            }
            Err(ReferenceError::TooSmall) => {
                small += 1;
                continue;
            }
            Err(ReferenceError::Empty) => {
                empty += 1;
                println!("EMPTY {name} ({w} x {h})");
                continue;
            }
            Err(e) => panic!("{name}: {e}"),
        }
        // 2. With bars.
        let (top, left) = (h / 8, w * 8 / 100);
        let expect = reduced_size(w, h, ANALYSIS_LONG_EDGE_PX);
        for (label, (t, l)) in [
            ("letterbox", (top, 0)),
            ("pillarbox", (0, left)),
            ("window box", (top / 2, left / 2)),
        ] {
            let (d, ow, oh) = barred(&rgb, w, h, t, l);
            let bytes = jpeg(&d, ow, oh, 60);
            match ReferencePicture::from_bytes(&bytes) {
                Ok(p) => {
                    let got = (p.analysis().width, p.analysis().height);
                    if got != expect {
                        wrong += 1;
                        println!("WRONG {name} {label}: {got:?}, expected {expect:?}");
                    }
                }
                Err(e) => {
                    wrong += 1;
                    println!("WRONG {name} {label}: {e}");
                }
            }
        }
    }
    println!(
        "pictures {tried}, losing bars they do not have {cut}, too small {small}, empty {empty}, unreadable {unread}; \
         synthetic bars not trimmed exactly {wrong} of {}",
        tried * 3
    );
    assert_eq!(cut, 0, "a picture without bars lost some");
}

/// What loading a big reference costs on this machine: a synthetic 6000 x 4000 JPEG (4:2:0, quality 90) and PNG, from the bytes to
/// the two reduced pictures. Printed, not asserted.
#[test]
#[ignore = "timing only: run with --ignored --nocapture"]
fn the_cost_of_loading_a_24_megapixel_reference() {
    let (w, h) = (6000usize, 4000usize);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let t = (x * 255 / w) as u8;
            let u = (y * 255 / h) as u8;
            rgb.extend_from_slice(&[t, u, t ^ u]);
        }
    }
    let jpg = jpeg(&rgb, w, h, 90);
    let mut png_bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png_bytes, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Fast);
        let mut writer = enc.write_header().expect("header");
        writer.write_image_data(&rgb).expect("data");
        writer.finish().expect("finish");
    }
    for (name, bytes) in [("jpeg", &jpg), ("png", &png_bytes)] {
        let start = std::time::Instant::now();
        let p = ReferencePicture::from_bytes(bytes).expect("a reference");
        println!(
            "{name}: {} MB file, 24 MP -> scene {} x {}, analysis {} x {} in {:.2} s on {} cores",
            bytes.len() / 1_000_000,
            p.scene().width,
            p.scene().height,
            p.analysis().width,
            p.analysis().height,
            start.elapsed().as_secs_f64(),
            std::thread::available_parallelism().map_or(1, usize::from)
        );
    }
}
