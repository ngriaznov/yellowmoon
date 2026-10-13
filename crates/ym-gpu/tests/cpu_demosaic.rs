//! R6 classical demosaic: port fidelity, receptive fields, PSNR comparison (`docs/05` R6, `docs/13` demosaic rows).
//!
//! * `menon_matches_the_python_reference`: our Menon port against fixtures produced by the
//!   `colour-demosaicing` package (`tests/cpu_data/gen_menon_fixtures.py`).
//! * `receptive_fields_are_within_the_declared_halos`: perturbs single photosites and measures how
//!   far the output changes; the declared halo must cover it.
//! * `psnr_on_synthetic_images`: ranking of the methods on procedural images (always runs).
//! * `psnr_on_kodak_and_mcmaster` (ignored): the public sets, fetched into
//!   `testdata/demosaic/{kodak,mcmaster}` by `tools/fetch/demosaic-sets.sh` (pinned commits, SHA-256 checked; override the root
//!   with `YM_TESTDATA`); run with
//!   `cargo test -p ym-gpu --test cpu_demosaic --release -- --ignored --nocapture`.

use std::path::{Path, PathBuf};

use ym_gpu::cpu::demosaic::{self, Method};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, Mosaic, RgbImage};

fn fixtures() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/menon_fixtures.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture file"))
        .expect("fixture json")
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

#[test]
fn menon_matches_the_python_reference() {
    let fx = fixtures();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["pattern"].as_str().unwrap();
        let (w, h) = (
            case["width"].as_u64().unwrap() as usize,
            case["height"].as_u64().unwrap() as usize,
        );
        let cfa = Cfa::from_name(name).unwrap();
        let m = Mosaic::from_data(w, h, cfa, floats(&case["cfa"])).unwrap();
        for (key, refine) in [("menon", true), ("menon_no_refine", false)] {
            let want = floats(&case[key]); // HWC interleaved
            let got = demosaic::menon(&m, refine).to_interleaved();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 3e-5, "{name} {key}: max |rust - python| = {worst}");
        }
    }
}

/// Largest Chebyshev distance from the perturbed photosite to any changed output sample.
fn measured_radius(method: Method, cfa: Cfa, px: usize, py: usize) -> usize {
    let (w, h) = (96, 96);
    let img = synth::natural(w, h, 11);
    let m = synth::mosaic_from_rgb(&img, cfa);
    let base = demosaic::demosaic_whole(&m, method);
    let mut m2 = m.clone();
    m2.data[py * w + px] += 0.37;
    let out = demosaic::demosaic_whole(&m2, method);
    let mut worst = 0usize;
    for y in 0..h {
        for x in 0..w {
            if base.pixel(x, y) != out.pixel(x, y) {
                worst = worst.max(x.abs_diff(px).max(y.abs_diff(py)));
            }
        }
    }
    worst
}

#[test]
fn receptive_fields_are_within_the_declared_halos() {
    for method in Method::ALL {
        let mut worst = 0;
        for cfa in Cfa::ALL {
            for (px, py) in [(48, 48), (49, 48), (48, 49), (49, 49)] {
                worst = worst.max(measured_radius(method, cfa, px, py));
            }
        }
        println!(
            "{:>9}: measured receptive radius {worst}, declared halo {}",
            method.name(),
            method.halo()
        );
        assert!(
            worst <= method.halo(),
            "{method:?}: measured {worst} > declared {}",
            method.halo()
        );
        assert!(
            worst + 2 >= method.halo(),
            "{method:?}: declared halo {} is far above the measured {worst}",
            method.halo()
        );
    }
}

fn mosaic_of(img: &RgbImage) -> Mosaic {
    synth::mosaic_from_rgb(img, Cfa::RGGB)
}

fn synthetic_set() -> Vec<(&'static str, RgbImage)> {
    vec![
        ("zone plate", synth::zone_plate(192, 192)),
        ("edges", synth::edges(192, 192)),
        ("colour patches", synth::color_patches(192, 192)),
        ("fine text", synth::fine_text(192, 192, 1)),
        ("natural 1", synth::natural(192, 192, 1)),
        ("natural 2", synth::natural(192, 192, 2)),
        ("natural 3", synth::natural(192, 192, 3)),
    ]
}

/// PSNR (dB) of each method on one ground-truth image, border of 10 px excluded.
fn psnr_row(truth: &RgbImage) -> [f64; 3] {
    let m = mosaic_of(truth);
    let mut row = [0.0; 3];
    for (i, method) in Method::ALL.iter().enumerate() {
        row[i] = psnr(&demosaic::demosaic(&m, *method), truth, 10, 1.0);
    }
    row
}

#[test]
fn psnr_on_synthetic_images() {
    println!(
        "{:<16} {:>9} {:>9} {:>9}",
        "image", "bilinear", "menon", "lmmse"
    );
    let mut mean = [0.0; 3];
    let set = synthetic_set();
    for (name, img) in &set {
        let row = psnr_row(img);
        println!("{name:<16} {:>9.2} {:>9.2} {:>9.2}", row[0], row[1], row[2]);
        for i in 0..3 {
            mean[i] += row[i] / set.len() as f64;
        }
    }
    println!(
        "{:<16} {:>9.2} {:>9.2} {:>9.2}",
        "mean", mean[0], mean[1], mean[2]
    );
    // Both directional methods beat bilinear on average over the procedural set.
    assert!(
        mean[1] > mean[0] + 1.0,
        "menon {} vs bilinear {}",
        mean[1],
        mean[0]
    );
    assert!(
        mean[2] > mean[0] + 1.0,
        "lmmse {} vs bilinear {}",
        mean[2],
        mean[0]
    );
}

fn load_png(path: &Path) -> RgbImage {
    let file = std::fs::File::open(path).expect("open png");
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let channels = info.color_type.samples();
    let rgb: Vec<f32> = buf[..info.width as usize * info.height as usize * channels]
        .chunks(channels)
        .flat_map(|p| {
            if channels >= 3 {
                [p[0], p[1], p[2]]
            } else {
                [p[0], p[0], p[0]]
            }
        })
        .map(|v| f32::from(v) / 255.0)
        .collect();
    RgbImage::from_interleaved(info.width as usize, info.height as usize, &rgb).unwrap()
}

fn load_tiff(path: &Path) -> RgbImage {
    let file = std::fs::File::open(path).expect("open tiff");
    let mut decoder =
        tiff::decoder::Decoder::new(std::io::BufReader::new(file)).expect("tiff decoder");
    let (w, h) = decoder.dimensions().expect("tiff dims");
    let data = match decoder.read_image().expect("tiff data") {
        tiff::decoder::DecodingResult::U8(v) => v,
        other => panic!("unexpected tiff sample type {other:?}"),
    };
    let rgb: Vec<f32> = data.iter().map(|&v| f32::from(v) / 255.0).collect();
    RgbImage::from_interleaved(w as usize, h as usize, &rgb).unwrap()
}

fn load_set(dir: &Path) -> Vec<(String, RgbImage)> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().to_string();
            match p.extension()?.to_str()? {
                "png" => Some((name, load_png(&p))),
                "tif" | "tiff" => Some((name, load_tiff(&p))),
                _ => None,
            }
        })
        .collect()
}

#[test]
#[ignore = "needs testdata/demosaic/{kodak,mcmaster}; run with --ignored --nocapture (see the file header)"]
fn psnr_on_kodak_and_mcmaster() {
    let root = std::env::var_os("YM_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"))
        .join("demosaic");
    for set in ["kodak", "mcmaster"] {
        let images = load_set(&root.join(set));
        if images.is_empty() {
            println!("{set}: no images in {} (skipped)", root.join(set).display());
            continue;
        }
        println!("== {set}: {} images", images.len());
        println!(
            "{:<16} {:>9} {:>9} {:>9}",
            "image", "bilinear", "menon", "lmmse"
        );
        let mut mean = [0.0; 3];
        for (name, img) in &images {
            let row = psnr_row(img);
            println!("{name:<16} {:>9.2} {:>9.2} {:>9.2}", row[0], row[1], row[2]);
            for i in 0..3 {
                mean[i] += row[i] / images.len() as f64;
            }
        }
        println!(
            "{:<16} {:>9.2} {:>9.2} {:>9.2}",
            "mean", mean[0], mean[1], mean[2]
        );
    }
}
