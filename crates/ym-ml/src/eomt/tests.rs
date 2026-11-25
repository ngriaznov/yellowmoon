//! Whole-network tests with seeded random weights: the loader, shapes, finiteness, the cost count and determinism across thread
//! counts and runs; the post-processing on hand-made outputs. Parity with the original model is `tests/eomt.rs` (the fixtures).

use std::collections::HashMap;
use std::sync::OnceLock;

use super::network::{CLASS_OUTPUTS, QUERIES};
use super::*;

fn weights() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| synthetic_weights(1))
}

fn model() -> &'static Eomt {
    static MODEL: OnceLock<Eomt> = OnceLock::new();
    MODEL.get_or_init(|| Eomt::from_bytes(weights(), None).expect("synthetic weights load"))
}

/// A seeded 8-bit picture with smooth structure and noise.
fn picture(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..width * height * 3)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            let (x, y, c) = ((i / 3) % width, (i / 3) / width, i % 3);
            let v = 128.0
                + 70.0 * (x as f32 * 0.05 + c as f32).sin()
                + 40.0 * (y as f32 * 0.03).cos()
                + 20.0 * noise;
            v.clamp(0.0, 255.0) as u8
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

fn rewrite(
    edit: impl FnOnce(&mut HashMap<String, (Vec<usize>, Vec<f32>)>),
    metadata: Option<HashMap<String, String>>,
) -> Vec<u8> {
    let mut tensors = read_float_tensors(weights()).expect("readable");
    edit(&mut tensors);
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
    safetensors::serialize(views, metadata).expect("serialises")
}

fn header(bytes: &[u8]) -> HashMap<String, String> {
    let (_, meta) = SafeTensors::read_metadata(bytes).expect("a safetensors file");
    meta.metadata().clone().expect("metadata")
}

#[test]
fn the_synthetic_weights_have_every_tensor_and_the_parameter_count_of_the_card() {
    let tensors = read_float_tensors(weights()).expect("readable");
    let specs = tensor_specs();
    assert_eq!(tensors.len(), 244);
    assert_eq!(specs.len(), 244);
    let total: usize = specs.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    assert_eq!(total, PARAMETERS);
    for (name, shape) in &specs {
        assert_eq!(&tensors[name].0, shape, "{name}");
    }
}

#[test]
fn a_file_with_the_wrong_header_or_tensors_is_refused() {
    let good = header(weights());
    // No header at all.
    let bytes = rewrite(|_| {}, None);
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Weights(_))
    ));
    // Another format.
    let mut other = good.clone();
    other.insert("format".into(), "yellowmoon-depth".into());
    let bytes = rewrite(|_| {}, Some(other));
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Unsupported(_))
    ));
    // A class table that is not ours.
    let mut other = good.clone();
    let mut config: serde_json::Value = serde_json::from_str(&good["config"]).expect("json");
    config["id2label"][5] = "train-wreck".into();
    other.insert("config".into(), config.to_string());
    let bytes = rewrite(|_| {}, Some(other));
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Weights(_))
    ));
    // A patch size of 14.
    let mut other = good.clone();
    let mut config: serde_json::Value = serde_json::from_str(&good["config"]).expect("json");
    config["patch_size"] = 14.into();
    other.insert("config".into(), config.to_string());
    let bytes = rewrite(|_| {}, Some(other));
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Unsupported(_))
    ));
    // A masked-attention checkpoint.
    let bytes = rewrite(
        |t| {
            t.get_mut("attn_mask_probs").expect("tensor").1[1] = 0.5;
        },
        Some(good.clone()),
    );
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Unsupported(_))
    ));
    // A tensor the network does not know, and one that is missing.
    let bytes = rewrite(
        |t| {
            t.insert("mystery.weight".into(), (vec![1], vec![1.0]));
        },
        Some(good.clone()),
    );
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Weights(_))
    ));
    let bytes = rewrite(
        |t| {
            t.remove("query.weight");
        },
        Some(good.clone()),
    );
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::Weights(_))
    ));
    // A wrong shape.
    let bytes = rewrite(
        |t| {
            t.insert(
                "query.weight".into(),
                (vec![100, 384], vec![0.0; 100 * 384]),
            );
        },
        Some(good),
    );
    assert!(matches!(
        Eomt::from_bytes(&bytes, None),
        Err(MlError::TensorShape { .. })
    ));
    // The checksum.
    assert!(matches!(
        Eomt::from_bytes(weights(), Some(&"0".repeat(64))),
        Err(MlError::Checksum { .. })
    ));
}

#[test]
fn the_network_gives_finite_outputs_of_the_right_shape() {
    let (input, (th, tw)) = input_tensor(&picture(96, 64, 3), 96, 64);
    assert_eq!((th, tw), (427, 640));
    let raw = model().infer(&input).expect("runs");
    assert_eq!(raw.class_logits.len(), QUERIES * CLASS_OUTPUTS);
    assert_eq!(raw.mask_logits.len(), QUERIES * MASK_PX * MASK_PX);
    assert!(raw.class_logits.iter().all(|v| v.is_finite()));
    assert!(raw.mask_logits.iter().all(|v| v.is_finite()));
    // The synthetic weights are not a model, but the outputs are not constant.
    let spread = |v: &[f32]| {
        v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
    };
    assert!(spread(&raw.class_logits) > 0.1);
    assert!(spread(&raw.mask_logits) > 0.1);
    let labels = model().label_map(&raw, th, tw);
    assert_eq!((labels.width, labels.height), (640, 427));
    assert_eq!(labels.classes.len(), 640 * 427);
}

#[test]
fn a_wrong_input_is_refused() {
    let m = model();
    assert!(matches!(m.infer(&[0.0; 10]), Err(MlError::Shape(_))));
    let mut input = vec![0.0f32; IMAGE_PX * IMAGE_PX * 3];
    input[7] = f32::NAN;
    assert!(matches!(m.infer(&input), Err(MlError::Input(_))));
    assert!(matches!(m.segment(&[0; 10], 4, 4), Err(MlError::Shape(_))));
    assert!(matches!(m.segment(&[], 0, 4), Err(MlError::Shape(_))));
}

#[test]
fn a_cancelled_run_stops() {
    let input = vec![0.1f32; IMAGE_PX * IMAGE_PX * 3];
    let result = model().infer_with(&input, &|| true);
    assert!(matches!(result, Err(MlError::Cancelled)));
}

#[test]
fn threads_and_repeats_do_not_change_a_bit() {
    let rgb = picture(160, 120, 9);
    let run = |threads: usize| {
        in_pool(threads, || {
            let (input, (th, tw)) = input_tensor(&rgb, 160, 120);
            let raw = model().infer(&input).expect("runs");
            let labels = model().label_map(&raw, th, tw);
            (
                bits(&input),
                bits(&raw.class_logits),
                bits(&raw.mask_logits),
                labels,
            )
        })
    };
    let one = run(1);
    assert_eq!(run(2), one, "2 threads");
    assert_eq!(run(8), one, "8 threads");
    assert_eq!(run(1), one, "a second run");
}

/// Class logits that make the query `q` choose `class` with a probability well over 0.8.
fn choose(raw: &mut Raw, q: usize, class: usize) {
    let row = &mut raw.class_logits[q * CLASS_OUTPUTS..(q + 1) * CLASS_OUTPUTS];
    row.fill(0.0);
    row[class] = 12.0;
}

/// A raw output where every query says "none" and every mask is far below 0.
fn empty_raw() -> Raw {
    let mut raw = Raw {
        class_logits: vec![0.0; QUERIES * CLASS_OUTPUTS],
        mask_logits: vec![-10.0; QUERIES * MASK_PX * MASK_PX],
    };
    for q in 0..QUERIES {
        choose(&mut raw, q, 133);
    }
    raw
}

/// Sets the mask logits of query `q` to `value` where `pred(x, y)` holds on the 160 x 160 grid.
fn paint(raw: &mut Raw, q: usize, value: f32, pred: impl Fn(usize, usize) -> bool) {
    let plane = &mut raw.mask_logits[q * MASK_PX * MASK_PX..(q + 1) * MASK_PX * MASK_PX];
    for y in 0..MASK_PX {
        for x in 0..MASK_PX {
            if pred(x, y) {
                plane[y * MASK_PX + x] = value;
            }
        }
    }
}

#[test]
fn the_label_map_follows_the_panoptic_rules() {
    let mut raw = empty_raw();
    // Query 3: sky (class 119 is a stuff class), the left half. Query 7: person (0), a block in the left half with a stronger mask.
    choose(&mut raw, 3, 119);
    paint(&mut raw, 3, 8.0, |x, _| x < 80);
    choose(&mut raw, 7, 0);
    paint(&mut raw, 7, 9.0, |x, y| {
        (20..60).contains(&x) && (20..60).contains(&y)
    });
    // Query 9: a class with a probability under 0.8 is not kept: 0.5 for class 2 and 0.5 for class 4 gives 0.5.
    raw.class_logits[9 * CLASS_OUTPUTS..(9 + 1) * CLASS_OUTPUTS].fill(0.0);
    raw.class_logits[9 * CLASS_OUTPUTS + 2] = 3.0;
    raw.class_logits[9 * CLASS_OUTPUTS + 4] = 3.0;
    paint(&mut raw, 9, 9.0, |x, _| x >= 100);
    // Query 11: "none" with a strong mask is not kept either.
    paint(&mut raw, 11, 9.0, |_, y| y >= 140);
    let kept = kept_queries(&raw.class_logits);
    assert_eq!(
        kept.iter().map(|k| (k.query, k.label)).collect::<Vec<_>>(),
        vec![(3, 119), (7, 0)]
    );
    let map = Eomt::label_map_for_tests(&raw, 640, 640);
    let at = |x: usize, y: usize| map[y * 640 + x];
    // The person's block wins where both masks hold (its score times its probability is larger than the sky's).
    assert_eq!(at(160, 160), 0);
    // The sky holds the left half outside the block.
    assert_eq!(at(20, 20), 119);
    assert_eq!(at(300, 500), 119);
    // Nothing holds the right half.
    assert_eq!(at(500, 100), UNLABELLED);
    assert_eq!(at(500, 600), UNLABELLED);
}

#[test]
fn a_query_that_owns_less_than_four_fifths_of_its_mask_is_not_a_segment() {
    let mut raw = empty_raw();
    // Query 0 (class 1) has a mask over columns 0 to 79 of the 160 x 160 grid, query 1 (class 2) a slightly stronger one over
    // columns 0 to 47 and query 2 (class 3) the right half: query 0 wins only columns 48 to 79, which is 40 % of its own mask,
    // so it is not a segment and those pixels are unlabelled. Queries 1 and 2 own their masks.
    choose(&mut raw, 0, 1);
    paint(&mut raw, 0, 9.9, |x, _| x < 80);
    choose(&mut raw, 1, 2);
    paint(&mut raw, 1, 10.0, |x, _| x < 48);
    choose(&mut raw, 2, 3);
    paint(&mut raw, 2, 10.0, |x, _| x >= 80);
    let map = Eomt::label_map_for_tests(&raw, 640, 640);
    assert_eq!(map[100 * 640 + 100], 2);
    assert_eq!(map[100 * 640 + 250], UNLABELLED);
    assert_eq!(map[100 * 640 + 500], 3);
    assert!(!map.contains(&1));
}

#[test]
fn nothing_kept_is_all_unlabelled() {
    let raw = empty_raw();
    let map = Eomt::label_map_for_tests(&raw, 300, 500);
    assert_eq!(map.len(), 300 * 500);
    assert!(map.iter().all(|&c| c == UNLABELLED));
}

#[test]
fn the_label_map_is_cut_to_the_fit_size() {
    let mut raw = empty_raw();
    choose(&mut raw, 0, 5);
    paint(&mut raw, 0, 6.0, |_, _| true);
    let map = Eomt::label_map_for_tests(&raw, 427, 640);
    assert_eq!(map.len(), 427 * 640);
    assert!(map.iter().all(|&c| c == 5));
}

impl Eomt {
    fn label_map_for_tests(raw: &Raw, h: usize, w: usize) -> Vec<u8> {
        model().label_map(raw, h, w).classes
    }
}

#[test]
fn nearest_resizing_takes_the_pixel_that_contains_the_centre() {
    let map = LabelMap {
        width: 4,
        height: 2,
        classes: vec![1, 2, 3, 4, 5, 6, 7, 8],
    };
    let big = map.resized_nearest(8, 4);
    assert_eq!(big.classes[..8], [1, 1, 2, 2, 3, 3, 4, 4]);
    assert_eq!(big.classes[16..24], [5, 5, 6, 6, 7, 7, 8, 8]);
    // A centre on the border of two source pixels takes the later one.
    let small = map.resized_nearest(2, 1);
    assert_eq!(small.classes, vec![6, 8]);
    assert_eq!(map.resized_nearest(4, 2), map);
    assert_eq!(map.elements().len(), 8);
}

#[test]
fn the_cost_is_about_sixty_seven_billion_multiply_adds() {
    let g = multiply_adds() as f64 / 1e9;
    assert!((60.0..75.0).contains(&g), "{g} G");
}
