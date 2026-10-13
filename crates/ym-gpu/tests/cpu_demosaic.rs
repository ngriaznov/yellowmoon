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
//!
//! The RA-05 gate of the R6 default (linear light at signal scales 1, 0.1 and 0.01, zipper and
//! false-colour metrics, noisy inputs, CPU time per 50 MP) is `examples/ra05_gate.rs` and
//! `examples/ra05_timing.rs`, run by `tools/eval/demosaic/gate.sh`; its metrics are tested in
//! `tests/cpu_demosaic_metrics.rs`.

use std::path::{Path, PathBuf};

use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use ym_noise::model::NoiseModel;

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
        // The adaptive ratio demosaic declares the spec's maximum (13, with shadow matching), and
        // this test runs it with unit gains and a zero noise model: plain, with the shadow gate
        // closed. Its reach in each configuration is measured in `tests/cpu_adaptive_ratio.rs`.
        if method == Method::AdaptiveRatio {
            // Still a lower bound, so that a perturbation that stopped reaching anything shows.
            assert!(
                worst >= 6,
                "{method:?}: the perturbation test saw only {worst} px"
            );
        } else {
            assert!(
                worst + 2 >= method.halo(),
                "{method:?}: declared halo {} is far above the measured {worst}",
                method.halo()
            );
        }
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

/// The nominal low-noise model of `docs/algorithms/01-adaptive-ratio-demosaic.md` section 6.3
/// (`a = 2e-5`, `b = 1e-8`, `q = 1 / 65535`) with unit gains. The adaptive ratio demosaic derives
/// its offsets K and T from the noise model, so its PSNR depends on it; with the zero model that
/// `demosaic()` passes, K = T = 1 DN (pure ratio interpolation) is not its operating point.
fn nominal_inputs() -> DemosaicInputs<'static> {
    DemosaicInputs {
        gains: [1.0; 3],
        noise: NoiseDn::from_model(&NoiseModel {
            a: 2e-5,
            b: 1e-8,
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 1.0 / 65535.0,
        }),
        ca: None,
    }
}

/// PSNR (dB) of each method of `Method::ALL` (in that order) on one ground-truth image, border of
/// 10 px excluded. The adaptive ratio demosaic runs with [`nominal_inputs`]; the others do not
/// read the noise model.
fn psnr_row(truth: &RgbImage) -> Vec<f64> {
    let m = mosaic_of(truth);
    Method::ALL
        .iter()
        .map(|method| {
            let out = if *method == Method::AdaptiveRatio {
                demosaic::demosaic_with(&m, *method, &nominal_inputs())
            } else {
                demosaic::demosaic(&m, *method)
            };
            psnr(&out, truth, 10, 1.0)
        })
        .collect()
}

/// The header line of the PSNR tables: one column per method of `Method::ALL`.
fn psnr_header() -> String {
    let mut line = format!("{:<16}", "image");
    for method in Method::ALL {
        line.push_str(&format!(" {:>10}", method.name()));
    }
    line
}

/// One line of the PSNR tables.
fn psnr_line(name: &str, row: &[f64]) -> String {
    let mut line = format!("{name:<16}");
    for v in row {
        line.push_str(&format!(" {v:>10.2}"));
    }
    line
}

/// Index of `method` in `Method::ALL` (the column of `psnr_row`).
fn column(method: Method) -> usize {
    Method::ALL
        .iter()
        .position(|&m| m == method)
        .expect("a method of Method::ALL")
}

#[test]
fn psnr_on_synthetic_images() {
    println!("{}", psnr_header());
    let mut mean = vec![0.0; Method::ALL.len()];
    let set = synthetic_set();
    for (name, img) in &set {
        let row = psnr_row(img);
        println!("{}", psnr_line(name, &row));
        for (m, v) in mean.iter_mut().zip(&row) {
            *m += v / set.len() as f64;
        }
    }
    println!("{}", psnr_line("mean", &mean));
    // The directional methods beat bilinear on average over the procedural set.
    let bilinear = mean[column(Method::Bilinear)];
    for method in [Method::Menon, Method::Lmmse, Method::AdaptiveRatio] {
        let got = mean[column(method)];
        assert!(
            got > bilinear + 1.0,
            "{} {got} vs bilinear {bilinear}",
            method.name()
        );
    }
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
        println!("{}", psnr_header());
        let mut mean = vec![0.0; Method::ALL.len()];
        for (name, img) in &images {
            let row = psnr_row(img);
            println!("{}", psnr_line(name, &row));
            for (m, v) in mean.iter_mut().zip(&row) {
                *m += v / images.len() as f64;
            }
        }
        println!("{}", psnr_line("mean", &mean));
    }
}
