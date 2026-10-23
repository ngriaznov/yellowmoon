//! The crop student against PyTorch (`training/crop/export.py` writes the fixture): the letterbox
//! gives Pillow's bytes exactly, and the network's numbers agree within 2e-4 (`f32` in a different
//! order of operations; the largest difference seen is printed).

use std::path::PathBuf;

use serde::Deserialize;
use ym_ml::crop_net::{CropNet, letterbox};

#[derive(Deserialize)]
struct Fixture {
    weights_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    width: usize,
    height: usize,
    seed: i64,
    letterbox_sha256: String,
    numbers: [f64; 4],
    #[serde(rename = "box")]
    crop: [f64; 4],
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture() -> Fixture {
    let text = std::fs::read_to_string(repo().join("crates/ym-ml/tests/data/cropnet_parity.json"))
        .expect("fixture");
    serde_json::from_str(&text).expect("fixture json")
}

/// `training/crop/export.py`'s `pattern`: gradients, edges and a bright block, integers only.
fn pattern(width: usize, height: usize, seed: i64) -> Vec<u8> {
    let (w, h) = (width as i64, height as i64);
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3i64 {
                let v = (x * (c + 1) * 7 + y * (3 + seed) + ((x * y) >> 5) * (11 + c) + seed * 37)
                    % 256;
                let block = x * 5 >= w * 2 && x * 5 < w * 3 && y * 3 >= h && y * 3 < h * 2;
                out.push(if block { 255 - (v >> 2) } else { v } as u8);
            }
        }
    }
    out
}

fn net() -> CropNet {
    CropNet::load(&repo().join("data/models/crop")).expect("data/models/crop loads")
}

#[test]
fn the_shipped_weights_are_the_fixture_s() {
    assert_eq!(net().sha256(), fixture().weights_sha256);
}

#[test]
fn the_letterbox_is_pillow_s_to_the_byte() {
    for case in fixture().cases {
        let (square, _) = letterbox(
            &pattern(case.width, case.height, case.seed),
            case.width,
            case.height,
        )
        .expect("letterbox");
        assert_eq!(
            ym_ml::model::sha256_hex(&square),
            case.letterbox_sha256,
            "{} x {}",
            case.width,
            case.height
        );
    }
}

#[test]
fn the_network_agrees_with_pytorch() {
    let net = net();
    let mut worst = 0.0f64;
    for case in fixture().cases {
        let rgb = pattern(case.width, case.height, case.seed);
        let (square, _) = letterbox(&rgb, case.width, case.height).unwrap();
        let numbers = net.numbers(&square).unwrap();
        let crop = net.crop(&rgb, case.width, case.height).unwrap();
        for i in 0..4 {
            worst = worst.max((numbers[i] - case.numbers[i]).abs());
            worst = worst.max((crop[i] - case.crop[i]).abs());
        }
    }
    println!("largest difference from PyTorch: {worst:.2e}");
    assert!(worst < 2e-4, "{worst}");
}

#[test]
fn the_same_picture_gives_the_same_bits() {
    let net = net();
    let rgb = pattern(768, 512, 3);
    let a = net.crop(&rgb, 768, 512).unwrap();
    let b = net.crop(&rgb, 768, 512).unwrap();
    assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    assert!(a[0] >= 0.0 && a[0] < a[2] && a[2] <= 1.0 && a[1] >= 0.0 && a[1] < a[3] && a[3] <= 1.0);
}

#[test]
fn a_damaged_or_foreign_file_is_refused() {
    let bytes = std::fs::read(repo().join("data/models/crop/cropnet.safetensors")).unwrap();
    assert!(CropNet::from_bytes(&bytes, Some(&"0".repeat(64))).is_err());
    assert!(CropNet::from_bytes(&bytes[..1000], None).is_err());
}

#[test]
#[ignore = "timing; run with --ignored --nocapture in a release build"]
fn timing_of_one_crop() {
    let net = net();
    let rgb = pattern(768, 512, 0);
    let t = std::time::Instant::now();
    for _ in 0..5 {
        std::hint::black_box(net.crop(&rgb, 768, 512).unwrap());
    }
    println!(
        "one crop: {:.1} ms",
        t.elapsed().as_secs_f64() * 1000.0 / 5.0
    );
}
