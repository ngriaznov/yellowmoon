//! Parity of the CPU runtime with PyTorch on the golden tiles (`training/jdd/golden_tiles.py`).
//!
//! The fixtures use seeded random weights of two small instances of the architecture (no trained weights
//! exist yet). Gate: max abs difference <= 2e-3 in the output's linear units (docs/07); the arithmetic is
//! f32 on both sides, so the real difference is ~1e-6.

mod common;

use common::{load_tensors, read, tile_from};
use ym_ml::cpu_runtime::{CpuJdd, bilinear_weights};
use ym_ml::model::{ExecContext, MlError, TileModel, TileShape, sha256_hex};

const TOL: f32 = 2e-3;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Run every `<key>.in.<i>` golden tile and require exactly `min_tiles` of them (12 + 8 = the 20 fixed inputs
/// of docs/07).
fn check_model(weights_file: &str, key: &str, min_tiles: usize) {
    let weights = read(weights_file);
    let model = CpuJdd::from_bytes(&weights, None).expect("weights load");
    assert_eq!(model.id().sha256, sha256_hex(&weights));
    let tiles = load_tensors(&read("golden_tiles.safetensors"));
    let mut n = 0;
    while let Some((shape, input)) = tiles.get(&format!("{key}.in.{n}")) {
        let (want_shape, want) = &tiles[&format!("{key}.out.{n}")];
        let ts = TileShape {
            height: shape[1],
            width: shape[2],
        };
        assert_eq!(want_shape, &vec![3, 2 * ts.height, 2 * ts.width]);
        let out = model.run_dynamic(&tile_from(ts, input)).expect("run");
        let d = max_abs_diff(out.as_slice(), want);
        assert!(
            d <= TOL,
            "{key} tile {n} ({ts}): max abs diff {d:e} > {TOL:e}"
        );
        assert!(
            want.iter().any(|v| v.abs() > 0.05),
            "golden output is not informative"
        );
        n += 1;
    }
    assert_eq!(
        n, min_tiles,
        "expected {min_tiles} golden tiles for {key}, found {n}"
    );
}

#[test]
fn small_model_matches_pytorch() {
    check_model("golden_small.safetensors", "small", 12);
}

#[test]
fn k3_model_matches_pytorch() {
    // 3x3 up-convolutions and 3x3 head: the other code paths of the runtime.
    check_model("golden_k3.safetensors", "k3", 8);
}

#[test]
fn header_describes_the_architecture() {
    let m = CpuJdd::from_bytes(&read("golden_small.safetensors"), None).unwrap();
    assert_eq!(m.config().channels, [8, 16, 24]);
    assert_eq!((m.config().up_kernel, m.config().head_kernel), (1, 1));
    assert_eq!(m.halo_half_px(), 32);
    assert_eq!(m.id().name, "jdd");
    assert_eq!(m.id().version, "golden-small");
    let k3 = CpuJdd::from_bytes(&read("golden_k3.safetensors"), None).unwrap();
    assert_eq!((k3.config().up_kernel, k3.config().head_kernel), (3, 3));
    assert!(k3.halo_half_px() >= 32 && k3.halo_half_px().is_multiple_of(8));
}

#[test]
fn bilinear_buffer_in_the_file_equals_the_built_in_kernel() {
    let t = load_tensors(&read("golden_small.safetensors"));
    let (shape, w) = &t["bilinear.weight"];
    assert_eq!(shape, &vec![12, 4, 3, 3]);
    assert!(max_abs_diff(w, &bilinear_weights()) < 1e-7);
}

#[test]
fn checksum_and_header_errors() {
    let bytes = read("golden_small.safetensors");
    let good = sha256_hex(&bytes);
    assert!(CpuJdd::from_bytes(&bytes, Some(&good)).is_ok());
    let bad = "0".repeat(64);
    assert!(matches!(
        CpuJdd::from_bytes(&bytes, Some(&bad)),
        Err(MlError::Checksum { .. })
    ));
    // Not a safetensors file, and a safetensors file without the jdd header.
    assert!(matches!(
        CpuJdd::from_bytes(b"nope", None),
        Err(MlError::Weights(_))
    ));
    let tiles = read("golden_tiles.safetensors");
    assert!(matches!(
        CpuJdd::from_bytes(&tiles, None),
        Err(MlError::Weights(_))
    ));
}

#[test]
fn static_shape_contract_of_the_trait() {
    let model = CpuJdd::from_bytes(&read("golden_small.safetensors"), None)
        .unwrap()
        .with_tile_shape(TileShape::square(32));
    let tiles = load_tensors(&read("golden_tiles.safetensors"));
    let (shape, input) = &tiles["small.in.0"];
    let ts = TileShape {
        height: shape[1],
        width: shape[2],
    };
    assert_eq!(model.input_shape(), ts);
    let tin = tile_from(ts, input);
    let mut out = ym_ml::model::TileOutput::zeros(ts);
    model.run(&tin, &mut out, &ExecContext::cpu()).unwrap();
    let (_, want) = &tiles["small.out.0"];
    assert!(max_abs_diff(out.as_slice(), want) <= TOL);
    // wrong shapes are rejected by `run` but accepted by `run_dynamic`
    let other = TileShape {
        height: 32,
        width: 48,
    };
    let bad = ym_ml::model::TileInputs::zeros(other);
    let mut bad_out = ym_ml::model::TileOutput::zeros(other);
    assert!(matches!(
        model.run(&bad, &mut bad_out, &ExecContext::cpu()),
        Err(MlError::Shape(_))
    ));
    assert!(model.run_dynamic(&bad).is_ok());
    assert!(
        model
            .run_dynamic(&ym_ml::model::TileInputs::zeros(TileShape {
                height: 30,
                width: 32
            }))
            .is_err()
    );
}

/// The weights file with `from` replaced by `to` (same length, so the safetensors offsets stay valid).
fn patched(name: &str, from: &str, to: &str) -> Vec<u8> {
    assert_eq!(from.len(), to.len());
    let mut bytes = read(name);
    let pos = bytes
        .windows(from.len())
        .position(|w| w == from.as_bytes())
        .unwrap_or_else(|| panic!("`{from}` not found in the header of {name}"));
    bytes[pos..pos + from.len()].copy_from_slice(to.as_bytes());
    bytes
}

#[test]
fn malformed_headers_are_errors_not_panics() {
    let f = "golden_small.safetensors";
    // sanity: the patch helper itself keeps a valid file valid
    assert!(
        CpuJdd::from_bytes(
            &patched(f, "\"halo_half_px\":\"32\"", "\"halo_half_px\":\"32\""),
            None
        )
        .is_ok()
    );
    for (what, from, to) in [
        // halo not a multiple of the tile multiple, zero, or not a number
        (
            "halo 30",
            "\"halo_half_px\":\"32\"",
            "\"halo_half_px\":\"30\"",
        ),
        (
            "halo 00",
            "\"halo_half_px\":\"32\"",
            "\"halo_half_px\":\"00\"",
        ),
        (
            "halo xx",
            "\"halo_half_px\":\"32\"",
            "\"halo_half_px\":\"xx\"",
        ),
        // zero input scale (a division by zero at the output), zero channel count, odd kernel
        (
            "plane_scale 0",
            "\\\"plane_scale\\\":0.25",
            "\\\"plane_scale\\\":0.00",
        ),
        (
            "channels 0",
            "\\\"channels\\\":[8,16,24]",
            "\\\"channels\\\":[0,16,24]",
        ),
        ("up_kernel 5", "\\\"up_kernel\\\":1", "\\\"up_kernel\\\":5"),
    ] {
        let r =
            std::panic::catch_unwind(|| CpuJdd::from_bytes(&patched(f, from, to), None).is_err());
        assert_eq!(
            r.ok(),
            Some(true),
            "{what}: must be an error, not a panic or a success"
        );
    }
}

#[test]
fn forward_planar_rejects_bad_shapes_instead_of_panicking() {
    let m = CpuJdd::from_bytes(&read("golden_small.safetensors"), None).unwrap();
    let ok = vec![0.1f32; 7 * 8 * 8];
    assert!(m.forward_planar(&ok, 8, 8).is_ok());
    assert!(matches!(
        m.forward_planar(&ok, 8, 6),
        Err(MlError::Shape(_))
    )); // 6 is not a multiple of 4
    assert!(matches!(
        m.forward_planar(&ok[..ok.len() - 1], 8, 8),
        Err(MlError::Shape(_))
    ));
    assert!(matches!(
        m.forward_planar(&ok, 0, 8),
        Err(MlError::Shape(_))
    ));
    assert!(matches!(
        m.forward_planar(&[], 0, 0),
        Err(MlError::Shape(_))
    ));
    assert!(matches!(
        m.forward_planar(&ok, usize::MAX - 3, 8),
        Err(MlError::Shape(_))
    ));
}
