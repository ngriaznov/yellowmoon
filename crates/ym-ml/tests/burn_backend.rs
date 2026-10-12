//! burn backend parity (feature `burn`): the golden tiles through `BurnJdd` on burn's wgpu backend must
//! match PyTorch and the CPU runtime within 2e-3. Runs on any wgpu adapter, including Mesa lavapipe
//! (`WGPU_BACKEND=vulkan`). Without an adapter the test is skipped with a message unless
//! `YM_REQUIRE_GPU=1` (the `docs/13` convention).
#![cfg(feature = "burn")]

mod common;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{load_tensors, read, tile_from};
use ym_ml::backends::burn_backend::BurnJdd;
use ym_ml::cpu_runtime::CpuJdd;
use ym_ml::model::{ExecContext, TileModel, TileOutput, TileShape};

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A burn model for tiles of `ts`, or the reason there is none (no wgpu adapter, bad weights).
fn build(weights: &[u8], ts: TileShape) -> Result<BurnJdd, String> {
    match catch_unwind(AssertUnwindSafe(|| BurnJdd::from_bytes(weights))) {
        Ok(Ok(m)) => Ok(m.with_tile_shape(ts)),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("wgpu device initialisation panicked (no adapter?)".to_string()),
    }
}

#[test]
fn burn_matches_pytorch_and_the_cpu_runtime_on_all_golden_tiles() {
    let tiles = load_tensors(&read("golden_tiles.safetensors"));
    let mut total = 0;
    for (key, weights_file) in [
        ("small", "golden_small.safetensors"),
        ("k3", "golden_k3.safetensors"),
    ] {
        let weights = read(weights_file);
        let cpu = CpuJdd::from_bytes(&weights, None).unwrap();
        // The static-shape contract of the trait: one model per distinct tile shape.
        let mut models: HashMap<TileShape, BurnJdd> = HashMap::new();
        let mut n = 0;
        while let Some((shape, input)) = tiles.get(&format!("{key}.in.{n}")) {
            let ts = TileShape {
                height: shape[1],
                width: shape[2],
            };
            if let Entry::Vacant(slot) = models.entry(ts) {
                match build(&weights, ts) {
                    Ok(m) => {
                        slot.insert(m);
                    }
                    Err(why) => {
                        assert!(
                            std::env::var("YM_REQUIRE_GPU").is_err(),
                            "YM_REQUIRE_GPU is set but burn failed: {why}"
                        );
                        eprintln!("skipped: {why}");
                        return;
                    }
                }
            }
            let model = &models[&ts];
            let tin = tile_from(ts, input);
            let mut out = TileOutput::zeros(ts);
            model
                .run(&tin, &mut out, &ExecContext::cpu())
                .expect("burn run");
            let want = &tiles[&format!("{key}.out.{n}")].1;
            let d_torch = max_diff(out.as_slice(), want);
            let d_cpu = max_diff(out.as_slice(), cpu.run_dynamic(&tin).unwrap().as_slice());
            eprintln!("{key} tile {n} ({ts}): burn vs torch {d_torch:e}, burn vs cpu {d_cpu:e}");
            assert!(d_torch <= 2e-3, "{key} tile {n}: burn vs torch {d_torch:e}");
            assert!(d_cpu <= 2e-3, "{key} tile {n}: burn vs cpu {d_cpu:e}");
            n += 1;
        }
        total += n;
        let (&ts, model) = models.iter().next().expect("at least one model");
        let other = TileShape {
            height: ts.height + 4,
            width: ts.width,
        };
        let bad = ym_ml::model::TileInputs::zeros(other);
        let mut bad_out = TileOutput::zeros(other);
        assert!(model.run(&bad, &mut bad_out, &ExecContext::cpu()).is_err());
    }
    assert_eq!(total, 20, "the 20 golden tiles of docs/07");
}
