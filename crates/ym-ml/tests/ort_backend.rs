//! ONNX Runtime backend parity (feature `ort`): the golden tiles through `OrtJdd` must match PyTorch and
//! the CPU runtime within 2e-3. Needs the ONNX Runtime shared library: set `ORT_DYLIB_PATH` (for example to
//! the `libonnxruntime.so*` inside the Python `onnxruntime` wheel). Without it the test is skipped with a
//! message, unless `YM_REQUIRE_ORT=1`.
#![cfg(feature = "ort")]

mod common;

use std::collections::HashMap;

use common::{load_tensors, read, tile_from};
use ym_ml::backends::ort_backend::OrtJdd;
use ym_ml::cpu_runtime::CpuJdd;
use ym_ml::model::{ExecContext, TileModel, TileOutput, TileShape};

fn ort_library() -> Option<String> {
    let p = std::env::var("ORT_DYLIB_PATH").ok()?;
    std::path::Path::new(&p).exists().then_some(p)
}

#[test]
fn ort_matches_pytorch_and_the_cpu_runtime_on_all_golden_tiles() {
    let Some(lib) = ort_library() else {
        assert!(
            std::env::var("YM_REQUIRE_ORT").is_err(),
            "YM_REQUIRE_ORT is set but ORT_DYLIB_PATH does not point at libonnxruntime"
        );
        eprintln!("skipped: ORT_DYLIB_PATH not set");
        return;
    };
    let tiles = load_tensors(&read("golden_tiles.safetensors"));
    let mut total = 0;
    // The golden ONNX files have dynamic height and width, so one session per distinct tile shape runs
    // every tile (the shipped export is static-shape).
    for (key, onnx_file, weights_file) in [
        ("small", "golden_small.onnx", "golden_small.safetensors"),
        ("k3", "golden_k3.onnx", "golden_k3.safetensors"),
    ] {
        let cpu = CpuJdd::from_bytes(&read(weights_file), None).unwrap();
        let halo = cpu.halo_half_px();
        let onnx = common::data_dir().join(onnx_file);
        let mut sessions: HashMap<TileShape, OrtJdd> = HashMap::new();
        let mut n = 0;
        while let Some((shape, input)) = tiles.get(&format!("{key}.in.{n}")) {
            let ts = TileShape {
                height: shape[1],
                width: shape[2],
            };
            let ort = sessions.entry(ts).or_insert_with(|| {
                OrtJdd::with_library(&lib, &onnx, ts, halo).expect("load ONNX model")
            });
            let tin = tile_from(ts, input);
            let mut out = TileOutput::zeros(ts);
            ort.run(&tin, &mut out, &ExecContext::cpu())
                .expect("ort run");
            let want = &tiles[&format!("{key}.out.{n}")].1;
            let d_torch = max_diff(out.as_slice(), want);
            let d_cpu = max_diff(out.as_slice(), cpu.run_dynamic(&tin).unwrap().as_slice());
            assert!(
                d_torch <= 2e-3,
                "{key} tile {n} ({ts}): ort vs torch {d_torch:e}"
            );
            assert!(d_cpu <= 2e-3, "{key} tile {n} ({ts}): ort vs cpu {d_cpu:e}");
            n += 1;
        }
        total += n;
        // a tile of another shape than the session was built for is rejected
        let (&ts, ort) = sessions.iter().next().expect("at least one session");
        let other = TileShape {
            height: ts.height + 4,
            width: ts.width,
        };
        let bad = ym_ml::model::TileInputs::zeros(other);
        let mut bad_out = TileOutput::zeros(other);
        assert!(ort.run(&bad, &mut bad_out, &ExecContext::cpu()).is_err());
    }
    assert_eq!(total, 20, "the 20 golden tiles of docs/07");
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
