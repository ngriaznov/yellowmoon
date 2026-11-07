//! Whole-network tests with seeded random weights: the loader, shapes, finiteness, the cost count and determinism across thread
//! counts. Stage-by-stage parity with the upstream code is `tests/depth.rs` (the oracle fixtures).

use std::collections::HashMap;
use std::sync::OnceLock;

use safetensors::{Dtype, tensor::TensorView};

use super::*;

fn weights() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| synthetic_weights(1))
}

fn model() -> &'static DepthAnything {
    static MODEL: OnceLock<DepthAnything> = OnceLock::new();
    MODEL
        .get_or_init(|| DepthAnything::from_bytes(weights(), None).expect("synthetic weights load"))
}

/// A seeded picture already normalised (values of about -2 to 2.5).
fn picture(width: usize, height: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..width * height * 3)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            let (x, y) = ((i / 3) % width, (i / 3) / width);
            ((x as f32 * 0.11).sin() + (y as f32 * 0.07).cos()) * 1.2 + noise
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(f)
}

/// The tensors of a weights file, widened to f32, for editing in a test.
fn tensors_of(bytes: &[u8]) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    read_float_tensors(bytes).expect("readable")
}

/// Writes tensors as an f32 safetensors file.
fn write(tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let raw: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(n, (s, d))| {
            (
                n.clone(),
                s.clone(),
                d.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views = raw.iter().map(|(n, s, b)| {
        (
            n.clone(),
            TensorView::new(Dtype::F32, s.clone(), b).expect("view"),
        )
    });
    safetensors::serialize(views, None).expect("serialise")
}

#[test]
fn the_tensor_list_is_the_upstream_state_dict_of_vits() {
    let specs = tensor_specs();
    let names: std::collections::HashSet<&str> = specs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.len(), specs.len(), "names are unique");
    // 12 blocks x 14 tensors, the stem (cls, pos, patch weight and bias) and the final norm.
    let encoder = specs
        .iter()
        .filter(|(n, _)| n.starts_with("pretrained."))
        .count();
    assert_eq!(encoder, 12 * 14 + 4 + 2);
    for name in [
        "pretrained.pos_embed",
        "pretrained.blocks.11.ls2.gamma",
        "depth_head.projects.3.weight",
        "depth_head.resize_layers.3.weight",
        "depth_head.scratch.layer4_rn.weight",
        "depth_head.scratch.refinenet1.resConfUnit1.conv1.weight",
        "depth_head.scratch.output_conv2.2.bias",
    ] {
        assert!(names.contains(name), "{name}");
    }
    assert!(
        !names.contains("depth_head.scratch.layer1_rn.bias"),
        "the layer_rn convolutions have no bias"
    );
    // 24.8 M parameters in the checkpoint (the two unused tensors are the other 74 k).
    let total: usize = specs.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    let unused = 384 + 2 * (64 * 64 * 9 + 64);
    println!(
        "depth network: {total} parameters read, {unused} more in the checkpoint that are never used"
    );
    assert_eq!(total + unused, 24_785_089);
}

#[test]
fn the_cost_count_is_the_sum_of_every_weight_times_the_positions_it_is_applied_at() {
    // An independent count: each weight tensor times the number of positions it runs at.
    for (width, height) in [(518usize, 518usize), (784, 518), (56, 84), (70, 42)] {
        let (ph, pw) = ((height / 14) as u64, (width / 14) as u64);
        let n = ph * pw;
        let (h2, w2) = (ph.div_ceil(2), pw.div_ceil(2));
        let mut total = 0u64;
        for (name, shape) in tensor_specs() {
            let numel: u64 = shape.iter().product::<usize>() as u64;
            if !name.ends_with(".weight") || shape.len() == 1 {
                continue;
            }
            // (name fragment, positions the weight runs at), first match wins; the rest are at the picture's size.
            let table: [(&str, u64); 16] = [
                ("pretrained.patch_embed", n),
                ("pretrained.blocks", n + 1),
                ("projects.", n),
                ("resize_layers.0", n),
                ("resize_layers.1", n),
                ("resize_layers.3", h2 * w2),
                ("layer4_rn", h2 * w2),
                ("layer3_rn", n),
                ("layer2_rn", 4 * n),
                ("layer1_rn", 16 * n),
                ("refinenet4.resConfUnit", h2 * w2),
                ("refinenet4.out_conv", n),
                ("refinenet3", n),
                ("refinenet2", 4 * n),
                ("refinenet1", 16 * n),
                ("output_conv1", 64 * n),
            ];
            let mut positions = table
                .iter()
                .find(|(needle, _)| name.contains(needle))
                .map_or((width * height) as u64, |&(_, p)| p);
            // The out_conv of a fusion block runs after its 2x resize.
            if name.contains("refinenet")
                && name.contains(".out_conv")
                && !name.contains("refinenet4")
            {
                positions *= 4;
            }
            // The transposed convolutions apply every weight once per input position (kernel = stride).
            total += numel * positions;
        }
        // The attention scores and the weighted sum.
        let t = n + 1;
        total += 12 * 2 * t * t * 384;
        assert_eq!(total, multiply_adds(width, height), "{width} x {height}");
    }
    assert_eq!(multiply_adds(518, 518), 57_633_767_040);
    assert_eq!(multiply_adds(784, 518), 100_619_427_840);
}

#[test]
fn the_loader_accepts_the_unused_tensors_and_refuses_what_is_wrong() {
    let good = tensors_of(weights());
    // The checkpoint's mask token and the coarsest block's first residual unit are read and ignored.
    let mut with_unused = good.clone();
    with_unused.insert(
        "pretrained.mask_token".into(),
        (vec![1, 384], vec![0.0; 384]),
    );
    for conv in ["conv1", "conv2"] {
        with_unused.insert(
            format!("depth_head.scratch.refinenet4.resConfUnit1.{conv}.weight"),
            (vec![64, 64, 3, 3], vec![0.0; 64 * 64 * 9]),
        );
        with_unused.insert(
            format!("depth_head.scratch.refinenet4.resConfUnit1.{conv}.bias"),
            (vec![64], vec![0.0; 64]),
        );
    }
    DepthAnything::from_bytes(&write(&with_unused), None).expect("the unused tensors are accepted");

    let mut missing = good.clone();
    missing.remove("pretrained.blocks.7.attn.qkv.bias");
    let err = DepthAnything::from_bytes(&write(&missing), None).unwrap_err();
    assert!(
        err.to_string()
            .contains("pretrained.blocks.7.attn.qkv.bias"),
        "{err}"
    );

    let mut wrong = good.clone();
    wrong.insert(
        "depth_head.projects.1.weight".into(),
        (vec![96, 384], vec![0.0; 96 * 384]),
    );
    assert!(matches!(
        DepthAnything::from_bytes(&write(&wrong), None),
        Err(MlError::TensorShape { .. })
    ));

    let mut extra = good.clone();
    extra.insert(
        "pretrained.blocks.12.norm1.weight".into(),
        (vec![384], vec![1.0; 384]),
    );
    let err = DepthAnything::from_bytes(&write(&extra), None).unwrap_err();
    assert!(err.to_string().contains("blocks.12"), "{err}");

    let mut nan = good.clone();
    nan.get_mut("pretrained.norm.bias").expect("present").1[3] = f32::NAN;
    assert!(matches!(
        DepthAnything::from_bytes(&write(&nan), None),
        Err(MlError::Weights(_))
    ));

    assert!(matches!(
        DepthAnything::from_bytes(b"not safetensors at all", None),
        Err(MlError::Weights(_))
    ));
    assert!(matches!(
        DepthAnything::from_bytes(weights(), Some(&"0".repeat(64))),
        Err(MlError::Checksum { .. })
    ));
    let sum = crate::model::sha256_hex(weights());
    let loaded = DepthAnything::from_bytes(weights(), Some(&sum)).expect("matching digest");
    assert_eq!(loaded.sha256(), Some(sum.as_str()));
}

#[test]
fn load_reads_the_folder_and_its_sha256sums() {
    let dir = std::env::temp_dir().join(format!("ym-depth-load-{}", std::process::id()));
    let models = dir.join("models").join("depth");
    std::fs::create_dir_all(&models).expect("mkdir");
    std::fs::write(models.join(WEIGHTS_FILE), weights()).expect("write weights");
    let sums = format!("{}  {WEIGHTS_FILE}\n", crate::model::sha256_hex(weights()));
    std::fs::write(models.join("SHA256SUMS"), sums).expect("write sums");
    let ok = DepthAnything::load_bundled(&dir);
    std::fs::write(
        models.join("SHA256SUMS"),
        format!("{}  {WEIGHTS_FILE}\n", "1".repeat(64)),
    )
    .expect("write sums");
    let bad = DepthAnything::load_bundled(&dir);
    std::fs::remove_dir_all(&dir).expect("cleanup");
    assert!(ok.is_ok());
    assert!(matches!(bad, Err(MlError::Checksum { .. })));
}

#[test]
fn inputs_are_checked() {
    let m = model();
    let ok = picture(28, 28, 1);
    assert!(m.infer(&ok, 28, 28).is_ok());
    assert!(matches!(m.infer(&ok, 28, 30), Err(MlError::Shape(_))));
    assert!(matches!(
        m.infer(&ok[..ok.len() - 1], 28, 28),
        Err(MlError::Shape(_))
    ));
    assert!(matches!(m.infer(&[], 0, 0), Err(MlError::Shape(_))));
    assert!(
        matches!(m.infer(&picture(7, 28, 1), 7, 28), Err(MlError::Shape(_))),
        "under the minimum"
    );
    // One patch is the smallest grid (the stride-2 layer keeps it at one).
    assert!(m.infer(&picture(14, 28, 1), 14, 28).is_ok());
    assert!(m.infer(&picture(14, 14, 1), 14, 14).is_ok());
    let mut bad = ok.clone();
    bad[10] = f32::INFINITY;
    assert!(matches!(m.infer(&bad, 28, 28), Err(MlError::Input(_))));
}

#[test]
fn a_run_has_the_shapes_and_the_values_of_a_depth_map() {
    let m = model();
    // 3 x 5 patches (odd grid: the stride-2 layer gives 2 x 3), 4 x 6 patches, 37 x 37 patches' shortcut is in the oracle tests.
    for (width, height) in [(70usize, 42usize), (84, 56), (28, 28)] {
        let rgb = picture(width, height, 7);
        let (map, trace) = m.infer_traced(&rgb, width, height).expect("runs");
        assert_eq!((map.width, map.height), (width, height));
        assert_eq!(map.data.len(), width * height);
        assert!(map.data.iter().all(|v| v.is_finite() && *v >= 0.0));
        let range = map.data.iter().cloned().fold(0.0f32, f32::max);
        assert!(range > 0.0, "{width}x{height}: the map is not all zero");
        let (ph, pw) = (height / 14, width / 14);
        let n = ph * pw;
        assert_eq!(trace.encoder.embedded.len(), (n + 1) * 384);
        assert_eq!(trace.encoder.blocks.len(), 12);
        assert!(
            trace
                .encoder
                .blocks
                .iter()
                .all(|b| b.len() == (n + 1) * 384)
        );
        assert_eq!(trace.taps.len(), 4);
        assert!(trace.taps.iter().all(|t| t.len() == n * 384));
        let (h2, w2) = (ph.div_ceil(2), pw.div_ceil(2));
        let want_layers = [16 * n * 48, 4 * n * 96, n * 192, h2 * w2 * 384];
        for (got, want) in trace.head.layers.iter().zip(want_layers) {
            assert_eq!(got.len(), want);
        }
        let want_paths = [n * 64, 4 * n * 64, 16 * n * 64, 64 * n * 64];
        for (got, want) in trace.head.paths.iter().zip(want_paths) {
            assert_eq!(got.len(), want);
        }
        assert_eq!(trace.head.output_conv1.len(), 64 * n * 32);
        // The traced run is the same run.
        let plain = m.infer(&rgb, width, height).expect("runs");
        assert_eq!(bits(&plain.data), bits(&map.data));
    }
}

#[test]
fn a_different_picture_gives_a_different_map() {
    let m = model();
    let a = m.infer(&picture(56, 56, 1), 56, 56).expect("runs");
    let b = m.infer(&picture(56, 56, 2), 56, 56).expect("runs");
    assert_ne!(bits(&a.data), bits(&b.data));
}

#[test]
fn threads_do_not_change_a_bit() {
    let m = model();
    // A size that makes several tasks of every kind (rows of the output stage, query blocks, row tiles).
    let (width, height) = (126, 84);
    let rgb = picture(width, height, 3);
    let run = |threads: usize| {
        in_pool(threads, || {
            let (map, trace) = m.infer_traced(&rgb, width, height).expect("runs");
            let mut all = bits(&map.data);
            all.extend(bits(&trace.encoder.embedded));
            for b in &trace.encoder.blocks {
                all.extend(bits(b));
            }
            for t in &trace.taps {
                all.extend(bits(t));
            }
            for l in trace.head.layers.iter().chain(&trace.head.paths) {
                all.extend(bits(l));
            }
            all.extend(bits(&trace.head.output_conv1));
            all
        })
    };
    let reference = run(1);
    for threads in [2, 8] {
        assert_eq!(reference, run(threads), "{threads} threads");
    }
    assert_eq!(reference, run(4), "a second pool");
}

#[test]
fn the_positional_embedding_is_the_trained_grid_at_37_by_37_and_resized_otherwise() {
    let m = model();
    let enc = &m.encoder;
    let at_37 = enc.position_embedding(37, 37).expect("pos");
    assert_eq!(at_37.len(), 37 * 37 * 384);
    let raw = tensors_of(weights());
    let grid = &raw["pretrained.pos_embed"].1[384..];
    // The f16 weights widened are what the loader holds.
    assert_eq!(bits(&at_37), bits(grid));
    // 37 x 56 is resized on both axes with the +0.1 offset: even the rows of the unchanged height move a little.
    let wide = enc.position_embedding(37, 56).expect("pos");
    assert_eq!(wide.len(), 37 * 56 * 384);
    assert_ne!(bits(&wide[..384 * 37]), bits(&at_37[..384 * 37]));
}

#[test]
fn cancel_stops_the_run() {
    let m = model();
    let rgb = picture(56, 56, 1);
    let calls = std::cell::Cell::new(0);
    let cancel = || {
        calls.set(calls.get() + 1);
        calls.get() > 3
    };
    assert!(matches!(
        m.infer_with(&rgb, 56, 56, &cancel),
        Err(MlError::Cancelled)
    ));
    assert!(m.infer_with(&rgb, 56, 56, &|| false).is_ok());
}
