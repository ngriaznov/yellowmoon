//! The CPU stages R5 to R8 against independent NumPy implementations (`tests/cpu_data/gen_reference.py`).
//!
//! These stages have no published reference code to compare with (R5 is clean-room from a
//! description, R6's bilateral filter, R7's colour guided filter and R8's space-variant
//! Richardson-Lucy are written from the spec), so the tiled-versus-whole and perturbation tests in
//! the other files can only show self-consistency. Here the inputs and outputs of plain-loop NumPy
//! versions of the spec text are stored (`reference.bin`, little-endian f32) and the Rust stages
//! must reproduce them: R5 to 1e-5 on all four Bayer phases, the filters to 1e-4 (their Rust
//! versions truncate negligible weights and interpolate kernels from a bank).

use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::detail;
use ym_gpu::cpu::highlights;
use ym_gpu::cpu::sharpen::{self, SharpenParams};
use ym_gpu::cpu::{Cfa, ClipMask, Mosaic, NoiseSigma, RgbImage};

const BIN: &[u8] = include_bytes!("cpu_data/reference.bin");
const META: &str = include_str!("cpu_data/reference.json");

fn meta(name: &str) -> serde_json::Value {
    let all: serde_json::Value = serde_json::from_str(META).expect("reference.json");
    all["arrays"][name].clone()
}

/// The float32 array `name` and its metadata entry.
fn array(name: &str) -> (Vec<f32>, serde_json::Value) {
    let m = meta(name);
    assert!(!m.is_null(), "no array {name} in reference.json");
    let len: usize = m["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .product();
    let start = m["offset"].as_u64().unwrap() as usize * 4;
    let data = BIN[start..start + 4 * len]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, m)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn highlight_reconstruction_matches_the_reference_for_every_bayer_phase() {
    for cfa in Cfa::ALL {
        let name = cfa.name();
        let (mosaic, m) = array(&format!("r5_{name}_mosaic"));
        let (w, h) = (
            m["width"].as_u64().unwrap() as usize,
            m["height"].as_u64().unwrap() as usize,
        );
        let (mask, _) = array(&format!("r5_{name}_mask"));
        let (want, _) = array(&format!("r5_{name}_out"));
        let (want_offsets, _) = array(&format!("r5_{name}_offsets"));

        let mut mos = Mosaic::from_data(w, h, cfa, mosaic.clone()).unwrap();
        let clip = ClipMask {
            width: w,
            height: h,
            data: mask.iter().map(|&v| v as u8).collect(),
        };
        let got = highlights::reconstruct(&mut mos, &clip).unwrap();
        let d = max_abs_diff(&mos.data, &want);
        println!(
            "R5 {name}: {} clipped photosites, max |rust - numpy| = {d:.2e}",
            clip.count()
        );
        assert!(clip.count() > 40, "{name}: the scene must clip");
        assert!(d < 1e-5, "{name}: {d}");
        // Something was actually reconstructed.
        assert!(
            mos.data.iter().zip(&mosaic).any(|(a, b)| a > b),
            "{name}: nothing changed"
        );
        for c in 0..3 {
            assert!(
                (got.offset[c] as f32 - want_offsets[c]).abs() < 1e-5,
                "{name} offset {c}"
            );
            assert_eq!(
                got.count[c] as f32,
                want_offsets[3 + c],
                "{name} reference count {c}"
            );
        }
    }
}

fn rgb_input() -> (RgbImage, Vec<f32>) {
    let (rgb, m) = array("rgb");
    let (w, h) = (
        m["width"].as_u64().unwrap() as usize,
        m["height"].as_u64().unwrap() as usize,
    );
    (RgbImage::from_planar(w, h, rgb).unwrap(), array("sigma").0)
}

#[test]
fn color_guided_filter_matches_the_reference() {
    let (img, sigma) = rgb_input();
    for color in [30.0f32, 100.0] {
        let (want, _) = array(&format!("guided_{}", color as i32));
        let got = detail::chroma_guided_filter(&img, &NoiseSigma::Plane(&sigma), color);
        let d = max_abs_diff(&got.data, &want);
        println!("R7 guided filter C={color}: max |rust - numpy| = {d:.2e}");
        assert!(d < 1e-5, "C={color}: {d}");
        // The reference moved something.
        assert!(max_abs_diff(&want, &img.data) > 0.01);
    }
}

#[test]
fn bilateral_denoise_matches_the_reference() {
    let (img, sigma) = rgb_input();
    let (want, _) = array("bilateral");
    let got = denoise::bilateral_opponent(
        &img,
        &NoiseSigma::Plane(&sigma),
        &BilateralParams::default(),
    );
    let d = max_abs_diff(&got.data, &want);
    println!("R6 bilateral: max |rust - numpy| = {d:.2e}");
    // The Rust filter skips weights below exp(-9) of the kernel peak; the reference keeps them.
    assert!(d < 1e-4, "{d}");
    assert!(max_abs_diff(&want, &img.data) > 0.01);
}

#[test]
fn richardson_lucy_matches_the_reference() {
    let (img, _) = rgb_input();
    for amount in [90.0f32, 150.0] {
        let (want, m) = array(&format!("rl_{}", amount as i32));
        let params = SharpenParams {
            amount,
            radius_px: m["radius"].as_f64().unwrap() as f32,
        };
        let noise = m["noise"].as_f64().unwrap() as f32;
        let got = sharpen::richardson_lucy(&img, &params, &NoiseSigma::Scalar(noise));
        let d = max_abs_diff(&got.data, &want);
        println!("R8 RL A={amount}: max |rust - numpy| = {d:.2e}");
        // Kernels come from a 33-level bank between rho and 1.25 rho; the reference evaluates them exactly.
        assert!(d < 1e-4, "A={amount}: {d}");
        assert!(
            max_abs_diff(&want, &img.data) > 0.02,
            "the reference sharpened nothing"
        );
    }
}
