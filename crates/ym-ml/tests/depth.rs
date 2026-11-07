//! The Depth Anything V2-Small runtime against the upstream PyTorch code (`docs/algorithms/06-masks.md` section 8.6, D-03).
//!
//! The fixtures are made by `training/depth/oracle.py` (the upstream `DepthAnythingV2('vits')` on CPU in f32 with the published
//! f16 weights widened to f32) for the public photos of the oracle set, and fetched by `tools/fetch/depth.sh --fixtures` into
//! `testdata/depth/oracle-1/` (git-ignored). The weights are `data/models/depth/` (`tools/fetch/depth.sh`). Without them the
//! gates print a plain "SKIPPED" line and pass; `YM_REQUIRE_DEPTH=1` makes a missing file a failure (CI sets it).
//! `YM_DEPTH_PHOTOS=a,b` runs only those photos (of the 3 full-trace photos and the 17 depth-only ones together).
//!
//! # The fixture format this test reads (a proposal for `oracle.py`; change it here and there together)
//!
//! `manifest.json`: `{"photos": {"<id>": {...}}, "depth_only": {"<id>": {...}}}`, 20 photos in all. `photos` are the 3 full-trace
//! photos (`building`, `cow`, `dog`; `"full_trace": true`) whose `<id>.npz` (uncompressed, `ZIP_STORED`) holds every key below;
//! `depth_only` are the other 17, whose npz holds only `picture`, `input` and `depth`. The resize, the depth map from upstream's
//! tensor and the whole path are gated on all 20; the stage by stage gate on the 3 full-trace photos. Arrays are float32 in
//! PyTorch's layouts, `N = ph * pw` patches of a `h x w` network input (`h = 14 ph`, `w = 14 pw`); `embedded` to `output_conv1`
//! exist only in the full-trace files:
//!
//! | key | shape | what |
//! |---|---|---|
//! | `picture` | `(H0, W0, 3)` | the 8-bit photo divided by 255, RGB (what [`prepare_input`] takes) |
//! | `input` | `(1, 3, h, w)` | upstream's `image2tensor` tensor made from that picture (OpenCV resize and normalisation) |
//! | `embedded` | `(1, 1 + N, 384)` | `prepare_tokens_with_masks(input)`: patch tokens with the class token and positional embedding |
//! | `block_00` .. `block_11` | `(1, 1 + N, 384)` | the stream after each block |
//! | `tap_0` .. `tap_3` | `(1, N, 384)` | `get_intermediate_layers(input, [2, 5, 8, 11])`: normed patch tokens |
//! | `layer_1` .. `layer_4` | `(1, C, ., .)` | after `projects` and `resize_layers`: C = 48, 96, 192, 384 |
//! | `path_4` .. `path_1` | `(1, 64, ., .)` | the fusion blocks' outputs |
//! | `output_conv1` | `(1, 32, 8 ph, 8 pw)` | `scratch.output_conv1(path_1)` |
//! | `depth` | `(1, h, w)` | `model(input)`, the network's output (before `infer_image`'s resize back) |
//!
//! Every gate compares one stage of ours with upstream's from the same input, so that an error is found where it is made.
//! These tests are slow without optimisation: run them with `cargo test -p ym-ml --release --test depth`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ym_color::mask_depth::{DepthMap, depth_band, depth_cut, depth_ramp, depth_ranks};
use ym_ml::depth::{DepthAnything, WEIGHTS_FILE, prepare_input, synthetic_weights};

// -------------------------------------------------------------------------------------------------
// The gates
// -------------------------------------------------------------------------------------------------
//
// Bounds set from the first runs against the fixtures, as the SAM 2 gates were (`tests/sam2.rs`: 4 to 14 times the worst
// measured value, never looser than the number proposed before measuring). A gate that fails is reported, not loosened. The
// relative RMS is the RMS of the difference over the RMS of upstream's tensor.

// "Worst measured" below is over the 20 oracle photos (the stage by stage ones over the 3 full-trace photos), 2026-11-07, on the
// Linux reference machine; each bound is 4 to 14 times it.

/// Our picture-to-tensor against upstream's (OpenCV) tensor, in normalised values. Worst measured: 2.4e-7 largest (bearded-man),
/// 7.3e-8 mean (sky-trees).
const RESIZE_MAX_ABS: f64 = 2e-6;
const RESIZE_MEAN_ABS: f64 = 5e-7;
/// Patch embedding, class token and positional embedding. Worst measured: 7.0e-7 (dog).
const EMBEDDED_REL_RMS: f64 = 5e-6;
/// The stream after any block. Worst measured: 5.9e-6 (dog).
const BLOCK_REL_RMS: f64 = 5e-5;
/// The four tapped layers. Worst measured: 4.1e-6 (cow).
const TAP_REL_RMS: f64 = 2e-5;
/// The head's projected and resized layers. Worst measured: 5.1e-6 (building).
const LAYER_REL_RMS: f64 = 5e-5;
/// The fusion paths and `output_conv1`. Worst measured: 1.9e-6 (building, `path_*`; 1.8e-6 `output_conv1`).
const PATH_REL_RMS: f64 = 1e-5;
/// The depth map from upstream's tensor: relative RMS (worst measured 2.6e-6, sky-trees), and the largest difference over the
/// map's largest value (worst measured 2.4e-5, chair).
const DEPTH_REL_RMS: f64 = 2e-5;
const DEPTH_MAX_REL: f64 = 2e-4;
/// The whole path (our resize into our network) against upstream's map, relative RMS. Worst measured: 2.8e-6 (sky-trees).
const WHOLE_REL_RMS: f64 = 2e-5;

// -------------------------------------------------------------------------------------------------
// Fixture files
// -------------------------------------------------------------------------------------------------

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn weights_dir() -> PathBuf {
    repo().join("data/models/depth")
}

fn fixtures_dir() -> PathBuf {
    repo().join("testdata/depth/oracle-1")
}

struct Npy {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn u16_at(b: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// Reads an uncompressed `.npz` of float32 arrays.
fn read_npz(path: &Path) -> HashMap<String, Npy> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let eocd = (0..=bytes.len() - 22)
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .expect("zip end record");
    let count = u16_at(&bytes, eocd + 10);
    let mut at = u32_at(&bytes, eocd + 16);
    let mut out = HashMap::new();
    for _ in 0..count {
        assert_eq!(
            bytes[at..at + 4],
            [0x50, 0x4b, 0x01, 0x02],
            "zip central header"
        );
        let method = u16_at(&bytes, at + 10);
        let size = u32_at(&bytes, at + 20);
        let (nlen, elen, clen) = (
            u16_at(&bytes, at + 28),
            u16_at(&bytes, at + 30),
            u16_at(&bytes, at + 32),
        );
        let local = u32_at(&bytes, at + 42);
        let name = String::from_utf8(bytes[at + 46..at + 46 + nlen].to_vec()).expect("utf-8 name");
        assert_eq!(method, 0, "{name} is stored, not compressed");
        let (lnlen, lelen) = (u16_at(&bytes, local + 26), u16_at(&bytes, local + 28));
        let body = &bytes[local + 30 + lnlen + lelen..][..size];
        out.insert(name.trim_end_matches(".npy").to_owned(), parse_npy(body));
        at += 46 + nlen + elen + clen;
    }
    out
}

fn parse_npy(b: &[u8]) -> Npy {
    assert_eq!(&b[..6], b"\x93NUMPY", "npy magic");
    let (hlen, start) = if b[6] == 1 {
        (u16_at(b, 8), 10)
    } else {
        (u32_at(b, 8), 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).expect("npy header");
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .expect("descr");
    assert_eq!(descr, "<f4", "float32 array");
    assert!(header.contains("'fortran_order': False"), "C order");
    let shape_text = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(['(', ')']).nth(1))
        .expect("shape");
    let shape = shape_text
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let data = b[start + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Npy { shape, data }
}

/// `(1, C, H, W)` to our `[y][x][channel]`.
fn nchw_to_hwc(a: &Npy) -> Vec<f32> {
    assert_eq!(a.shape.len(), 4, "{:?}", a.shape);
    let (c, h, w) = (a.shape[1], a.shape[2], a.shape[3]);
    let mut out = vec![0.0f32; a.data.len()];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ch] = a.data[(ch * h + y) * w + x];
            }
        }
    }
    out
}

fn require() -> bool {
    std::env::var_os("YM_REQUIRE_DEPTH").is_some_and(|v| v != "0" && !v.is_empty())
}

/// The model, or `None` after printing why not (or a panic when `YM_REQUIRE_DEPTH` is set).
fn model() -> Option<&'static DepthAnything> {
    static MODEL: OnceLock<Option<DepthAnything>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            if !weights_dir().join(WEIGHTS_FILE).exists() {
                let msg = format!(
                    "SKIPPED: no Depth Anything weights in {} (run tools/fetch/depth.sh)",
                    weights_dir().display()
                );
                assert!(!require(), "{msg}");
                println!("{msg}");
                return None;
            }
            Some(
                DepthAnything::load(&weights_dir()).expect("the weights load and match SHA256SUMS"),
            )
        })
        .as_ref()
}

fn photo_ids() -> Option<Vec<String>> {
    if !fixtures_dir().join("manifest.json").exists() {
        let msg = format!(
            "SKIPPED: no Depth Anything oracle fixtures in {} (run tools/fetch/depth.sh --fixtures)",
            fixtures_dir().display()
        );
        assert!(!require(), "{msg}");
        println!("{msg}");
        return None;
    }
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("manifest.json")).expect("manifest"),
    )
    .expect("json");
    // `photos` (the full traces) and `depth_only` together: 20 photos.
    let mut ids: Vec<String> = ["photos", "depth_only"]
        .iter()
        .flat_map(|k| {
            manifest[*k]
                .as_object()
                .unwrap_or_else(|| panic!("manifest has no `{k}`"))
                .keys()
                .cloned()
        })
        .collect();
    ids.sort();
    if let Ok(only) = std::env::var("YM_DEPTH_PHOTOS") {
        let want: Vec<&str> = only.split(',').collect();
        ids.retain(|id| want.contains(&id.as_str()));
    }
    Some(ids)
}

// -------------------------------------------------------------------------------------------------
// Measurements
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Diff {
    max_abs: f64,
    mean_abs: f64,
    /// RMS of the difference over the RMS of the reference.
    rel_rms: f64,
}

fn diff(ours: &[f32], reference: &[f32]) -> Diff {
    assert_eq!(ours.len(), reference.len(), "same number of values");
    let (mut max, mut sum, mut sq, mut ref_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        let d = (f64::from(a) - f64::from(b)).abs();
        max = max.max(d);
        sum += d;
        sq += d * d;
        ref_sq += f64::from(b) * f64::from(b);
    }
    let n = ours.len().max(1) as f64;
    Diff {
        max_abs: max,
        mean_abs: sum / n,
        rel_rms: (sq / n).sqrt() / (ref_sq / n).sqrt().max(1e-30),
    }
}

/// One photo's measured stages: `(name, difference, bound on the relative RMS)`.
struct Photo {
    id: String,
    resize: Diff,
    /// The stage by stage differences; empty for a depth-only photo.
    stages: Vec<(String, Diff, f64)>,
    /// The depth map from upstream's tensor against upstream's map.
    depth: Diff,
    depth_max_rel: f64,
    whole: Diff,
    /// The map of the whole path (`w x h`, row-major) and upstream's, for the cut's gate.
    grid: (usize, usize),
    ours: Vec<f32>,
    theirs: Vec<f32>,
}

fn measure(model: &DepthAnything, id: &str) -> Photo {
    let fx = read_npz(&fixtures_dir().join(format!("{id}.npz")));
    let get = |k: &str| {
        fx.get(k)
            .unwrap_or_else(|| panic!("{id}: fixture has no `{k}`"))
    };
    let input = get("input");
    let (h, w) = (input.shape[2], input.shape[3]);
    let input_hwc = nchw_to_hwc(input);

    // 1. Our resize against upstream's tensor.
    let picture = get("picture");
    let (ph0, pw0) = (picture.shape[0], picture.shape[1]);
    let prepared = prepare_input(&picture.data, pw0, ph0).expect("prepares");
    assert_eq!(
        (prepared.width, prepared.height),
        (w, h),
        "{id}: the network size"
    );
    let resize = diff(&prepared.data, &input_hwc);

    // 2. The network from upstream's tensor, stage by stage (the full-trace photos) or only its output (the others).
    let mut stages: Vec<(String, Diff, f64)> = Vec::new();
    let map = if fx.contains_key("embedded") {
        let (map, trace) = model.infer_traced(&input_hwc, w, h).expect("runs");
        stages.push((
            "embedded".into(),
            diff(&trace.encoder.embedded, &get("embedded").data),
            EMBEDDED_REL_RMS,
        ));
        for (i, b) in trace.encoder.blocks.iter().enumerate() {
            stages.push((
                format!("block {i}"),
                diff(b, &get(&format!("block_{i:02}")).data),
                BLOCK_REL_RMS,
            ));
        }
        for (i, t) in trace.taps.iter().enumerate() {
            stages.push((
                format!("tap {i}"),
                diff(t, &get(&format!("tap_{i}")).data),
                TAP_REL_RMS,
            ));
        }
        for (i, l) in trace.head.layers.iter().enumerate() {
            stages.push((
                format!("layer {}", i + 1),
                diff(l, &nchw_to_hwc(get(&format!("layer_{}", i + 1)))),
                LAYER_REL_RMS,
            ));
        }
        for (i, p) in trace.head.paths.iter().enumerate() {
            let name = format!("path_{}", 4 - i);
            stages.push((
                name.clone(),
                diff(p, &nchw_to_hwc(get(&name))),
                PATH_REL_RMS,
            ));
        }
        stages.push((
            "output_conv1".into(),
            diff(&trace.head.output_conv1, &nchw_to_hwc(get("output_conv1"))),
            PATH_REL_RMS,
        ));
        map
    } else {
        model.infer(&input_hwc, w, h).expect("runs")
    };
    let want = &get("depth").data;
    let depth = diff(&map.data, want);
    let top = want.iter().cloned().fold(0.0f32, f32::max).max(1e-12);

    // 3. The whole path: our resize into our network.
    let whole_map = model.infer(&prepared.data, w, h).expect("runs");
    Photo {
        id: id.to_owned(),
        resize,
        stages,
        depth,
        depth_max_rel: depth.max_abs / f64::from(top),
        whole: diff(&whole_map.data, want),
        grid: (w, h),
        ours: whole_map.data,
        theirs: want.clone(),
    }
}

fn results() -> Option<&'static Vec<Photo>> {
    static RESULTS: OnceLock<Option<Vec<Photo>>> = OnceLock::new();
    RESULTS
        .get_or_init(|| {
            if cfg!(debug_assertions) && !require() && std::env::var_os("YM_DEPTH_FULL").is_none() {
                println!("SKIPPED: the depth parity tests need an optimised build (cargo test -p ym-ml --release --test depth)");
                return None;
            }
            let model = model()?;
            let ids = photo_ids()?;
            Some(ids.iter().map(|id| measure(model, id)).collect())
        })
        .as_ref()
}

// -------------------------------------------------------------------------------------------------
// The gates
// -------------------------------------------------------------------------------------------------

#[test]
fn the_resize_matches_upstreams_tensor() {
    let Some(all) = results() else { return };
    for p in all {
        println!(
            "{}: resize largest difference {:.3e}, mean {:.3e}",
            p.id, p.resize.max_abs, p.resize.mean_abs
        );
        assert!(
            p.resize.max_abs <= RESIZE_MAX_ABS,
            "{}: largest {:.3e}",
            p.id,
            p.resize.max_abs
        );
        assert!(
            p.resize.mean_abs <= RESIZE_MEAN_ABS,
            "{}: mean {:.3e}",
            p.id,
            p.resize.mean_abs
        );
    }
}

#[test]
fn every_stage_of_the_network_matches_upstream_from_the_same_tensor() {
    let Some(all) = results() else { return };
    let mut failures = Vec::new();
    let traced = all.iter().filter(|p| !p.stages.is_empty()).count();
    assert!(
        traced > 0 || std::env::var_os("YM_DEPTH_PHOTOS").is_some(),
        "no full-trace photo in the run"
    );
    println!("stage by stage on {traced} full-trace photos");
    for p in all {
        println!(
            "{}: depth: relative RMS {:.3e} (bound {DEPTH_REL_RMS:.0e}), largest {:.3e}",
            p.id, p.depth.rel_rms, p.depth.max_abs
        );
        if p.depth.rel_rms > DEPTH_REL_RMS {
            failures.push(format!(
                "{} depth: relative RMS {:.3e} over {DEPTH_REL_RMS:.0e}",
                p.id, p.depth.rel_rms
            ));
        }
        for (name, d, bound) in &p.stages {
            println!(
                "{}: {name}: relative RMS {:.3e} (bound {bound:.0e}), largest {:.3e}",
                p.id, d.rel_rms, d.max_abs
            );
            if d.rel_rms > *bound {
                failures.push(format!(
                    "{} {name}: relative RMS {:.3e} over {bound:.0e}",
                    p.id, d.rel_rms
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_depth_map_agrees_in_its_largest_difference_too() {
    let Some(all) = results() else { return };
    for p in all {
        println!(
            "{}: depth largest difference over the map's largest value {:.3e}",
            p.id, p.depth_max_rel
        );
        assert!(
            p.depth_max_rel <= DEPTH_MAX_REL,
            "{}: {:.3e}",
            p.id,
            p.depth_max_rel
        );
    }
}

#[test]
fn the_whole_path_matches_upstreams_map() {
    let Some(all) = results() else { return };
    for p in all {
        println!("{}: whole path relative RMS {:.3e}", p.id, p.whole.rel_rms);
        assert!(
            p.whole.rel_rms <= WHOLE_REL_RMS,
            "{}: {:.3e}",
            p.id,
            p.whole.rel_rms
        );
    }
}

/// The bands of the cut's gate, `(near, far)` at softness 0.5 (`docs/algorithms/06` section 8.6).
const CUT_BANDS: [(f64, f64); 5] = [(0.0, 0.2), (0.0, 0.5), (0.3, 0.6), (0.5, 1.0), (0.8, 1.0)];
/// The least IoU of the cut `S0 > 0.5` of our map and of upstream's (Gate D).
const CUT_MIN_IOU: f64 = 0.99;

/// The cells of `S0 > 0.5` for a band, from the ranks of `map` (the rank and the band of section 5.8, in f64).
fn cut_cells(map: &DepthMap, near: f64, far: f64) -> Vec<bool> {
    let (nu, phi, sigma) = depth_band(near, far, 0.5);
    let q = depth_ramp(sigma);
    (0..map.len())
        .map(|i| depth_cut(map.z(i), nu, phi, q) > 0.5)
        .collect()
}

/// The IoU of two cuts; 1 when both are empty.
fn iou(a: &[bool], b: &[bool]) -> f64 {
    let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    let union = a.iter().zip(b).filter(|(x, y)| **x || **y).count();
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// The share of the cells that sit in the largest group of equal values (a tie group such as sky at exactly 0).
fn largest_tie_share(d: &[f32]) -> (f32, f64) {
    let mut v: Vec<f32> = d.iter().map(|x| x + 0.0).collect();
    v.sort_unstable_by(f32::total_cmp);
    let (mut best, mut best_n, mut run, mut cur) = (0.0f32, 0usize, 0usize, f32::NAN);
    for &x in &v {
        if x.total_cmp(&cur).is_eq() {
            run += 1;
        } else {
            cur = x;
            run = 1;
        }
        if run > best_n {
            (best, best_n) = (x, run);
        }
    }
    (best, best_n as f64 / v.len().max(1) as f64)
}

/// Gate D: the rank and the band applied to our `d` and to upstream's give the same cut (IoU of `S0 > 0.5` at least 0.99) on
/// every photo and band. Prints every IoU and the worst per band; a failure names the photo, the band and the ties.
#[test]
fn the_cut_of_our_map_matches_the_cut_of_upstreams() {
    let Some(all) = results() else { return };
    let mut worst: [(f64, String); 5] = std::array::from_fn(|_| (1.0, String::new()));
    let mut failures = Vec::new();
    for p in all {
        let (w, h) = p.grid;
        let ours = DepthMap::new(1, w, h, p.ours.clone()).expect("our map is finite");
        let theirs = DepthMap::new(2, w, h, p.theirs.clone()).expect("upstream's map is finite");
        let (tie_v, tie_share) = largest_tie_share(&p.theirs);
        let mut line = format!("{:<14}", p.id);
        for (b, &(near, far)) in CUT_BANDS.iter().enumerate() {
            let v = iou(&cut_cells(&ours, near, far), &cut_cells(&theirs, near, far));
            line += &format!("  ({near}, {far}) {v:.5}");
            if v < worst[b].0 {
                worst[b] = (v, p.id.clone());
            }
            if v < CUT_MIN_IOU {
                failures.push(format!(
                    "{}: the cut ({near}, {far}) has IoU {v:.5} (upstream's largest tie group: {:.1} % of the cells at {tie_v:e})",
                    p.id,
                    tie_share * 100.0
                ));
            }
        }
        println!(
            "{line}  | upstream tie group {:.2} % at {tie_v:e}",
            tie_share * 100.0
        );
    }
    for (b, &(near, far)) in CUT_BANDS.iter().enumerate() {
        println!(
            "worst IoU of ({near}, {far}): {:.5} ({})",
            worst[b].0, worst[b].1
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The real weights give the same bits on any number of threads (the synthetic-weights twin is in `src/depth/tests.rs`).
#[test]
fn the_real_weights_give_the_same_bits_on_1_2_4_and_8_threads() {
    if cfg!(debug_assertions) && !require() && std::env::var_os("YM_DEPTH_FULL").is_none() {
        println!("SKIPPED: needs an optimised build (cargo test -p ym-ml --release --test depth)");
        return;
    }
    let Some(model) = model() else { return };
    if !fixtures_dir().join("dog.npz").exists() {
        let msg = "SKIPPED: no `dog` fixture (run tools/fetch/depth.sh --fixtures)";
        assert!(!require(), "{msg}");
        println!("{msg}");
        return;
    }
    let fx = read_npz(&fixtures_dir().join("dog.npz"));
    let picture = &fx["picture"];
    let run = |threads: usize| -> (Vec<u32>, Vec<u32>) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                let prepared = prepare_input(&picture.data, picture.shape[1], picture.shape[0])
                    .expect("prepares");
                let map = model
                    .infer(&prepared.data, prepared.width, prepared.height)
                    .expect("runs");
                (
                    map.data.iter().map(|v| v.to_bits()).collect(),
                    depth_ranks(&map.data),
                )
            })
    };
    let reference = run(1);
    for threads in [2, 4, 8] {
        let (d, k) = run(threads);
        assert!(
            reference.0 == d,
            "dog: the map differs on {threads} threads"
        );
        assert!(
            reference.1 == k,
            "dog: the ranks differ on {threads} threads"
        );
    }
    println!(
        "dog: whole path (d and the ranks k) bit-identical on 1, 2, 4 and 8 threads ({} values)",
        reference.0.len()
    );
}

// -------------------------------------------------------------------------------------------------
// Always run
// -------------------------------------------------------------------------------------------------

#[test]
fn the_public_api_runs_on_synthetic_weights() {
    let model = DepthAnything::from_bytes(&synthetic_weights(5), None).expect("loads");
    let (w, h) = (42usize, 28usize);
    let rgb: Vec<f32> = (0..w * h * 3)
        .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
        .collect();
    let map = model.infer(&rgb, w, h).expect("runs");
    assert_eq!((map.width, map.height, map.data.len()), (w, h, w * h));
    assert!(map.data.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn the_comparison_helpers_measure_what_they_say() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        diff(&a, &a),
        Diff {
            max_abs: 0.0,
            mean_abs: 0.0,
            rel_rms: 0.0
        }
    );
    let b = [1.0f32, 2.0, 3.0, 5.0];
    let d = diff(&b, &a);
    assert_eq!(d.max_abs, 1.0);
    assert_eq!(d.mean_abs, 0.25);
    assert!((d.rel_rms - (0.25f64.sqrt() / (30.0f64 / 4.0).sqrt())).abs() < 1e-12);
    // (1, 2, 2, 3) channels-first to channels-last.
    let npy = Npy {
        shape: vec![1, 2, 2, 3],
        data: (0..12).map(|v| v as f32).collect(),
    };
    let hwc = nchw_to_hwc(&npy);
    assert_eq!(hwc[0], 0.0);
    assert_eq!(hwc[1], 6.0, "channel 1 of pixel (0, 0)");
    assert_eq!(hwc[(3 + 2) * 2 + 1], 11.0);
}

#[test]
fn a_missing_fixture_directory_is_skipped_not_failed() {
    // Without YM_REQUIRE_DEPTH the gates above print SKIPPED and pass; this pins that the fixture path is the documented one.
    assert!(fixtures_dir().ends_with("testdata/depth/oracle-1"));
    assert!(weights_dir().ends_with("data/models/depth"));
}
