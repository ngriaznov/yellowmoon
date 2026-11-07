//! The Depth Anything V2-Small runtime against the upstream PyTorch code (`docs/algorithms/06-masks.md` section 8.6, D-03).
//!
//! The fixtures are made by `training/depth/oracle.py` (the upstream `DepthAnythingV2('vits')` on CPU in f32 with the published
//! f16 weights widened to f32) for the public photos of the oracle set, and fetched by `tools/fetch/depth.sh --fixtures` into
//! `testdata/depth/oracle-1/` (git-ignored). The weights are `data/models/depth/` (`tools/fetch/depth.sh`). Without them the
//! gates print a plain "SKIPPED" line and pass; `YM_REQUIRE_DEPTH=1` makes a missing file a failure (CI sets it).
//! `YM_DEPTH_PHOTOS=a,b` runs only those photos.
//!
//! # The fixture format this test reads (a proposal for `oracle.py`; change it here and there together)
//!
//! `manifest.json`: `{"photos": {"<id>": {...}}}`. `<id>.npz` (uncompressed, `ZIP_STORED`) holds float32 arrays in PyTorch's
//! layouts, `N = ph * pw` patches of a `h x w` network input (`h = 14 ph`, `w = 14 pw`):
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

use ym_ml::depth::{DepthAnything, WEIGHTS_FILE, prepare_input, synthetic_weights};

// -------------------------------------------------------------------------------------------------
// The gates
// -------------------------------------------------------------------------------------------------
//
// PROPOSED bounds, mirroring the SAM 2 gates (`tests/sam2.rs`: 4 to 14 times the worst measured, 20 to 2000 times tighter than the
// starting number). Nothing has been measured yet: the first run against the fixtures sets them (and any gate that fails then
// is reported, not loosened). The relative RMS is the RMS of the difference over the RMS of upstream's tensor.

/// Our picture-to-tensor against upstream's (OpenCV) tensor, in normalised values.
const RESIZE_MAX_ABS: f64 = 5e-3;
const RESIZE_MEAN_ABS: f64 = 5e-4;
/// Patch embedding, class token and positional embedding.
const EMBEDDED_REL_RMS: f64 = 1e-5;
/// The stream after any block.
const BLOCK_REL_RMS: f64 = 5e-5;
/// The four tapped layers.
const TAP_REL_RMS: f64 = 1e-4;
/// The head's projected and resized layers, the fusion paths and `output_conv1`.
const LAYER_REL_RMS: f64 = 1e-4;
const PATH_REL_RMS: f64 = 2e-4;
/// The depth map from upstream's tensor: relative RMS, and the largest difference over the map's largest value.
const DEPTH_REL_RMS: f64 = 5e-4;
const DEPTH_MAX_REL: f64 = 5e-3;
/// The whole path (our resize into our network) against upstream's map.
const WHOLE_REL_RMS: f64 = 5e-3;

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
    let mut ids: Vec<String> = manifest["photos"]
        .as_object()
        .expect("photos")
        .keys()
        .cloned()
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
    stages: Vec<(String, Diff, f64)>,
    depth_max_rel: f64,
    whole: Diff,
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

    // 2. The network from upstream's tensor, stage by stage.
    let (map, trace) = model.infer_traced(&input_hwc, w, h).expect("runs");
    let mut stages: Vec<(String, Diff, f64)> = Vec::new();
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
    let want = &get("depth").data;
    let depth = diff(&map.data, want);
    stages.push(("depth".into(), depth, DEPTH_REL_RMS));
    let top = want.iter().cloned().fold(0.0f32, f32::max).max(1e-12);

    // 3. The whole path: our resize into our network.
    let whole_map = model.infer(&prepared.data, w, h).expect("runs");
    Photo {
        id: id.to_owned(),
        resize,
        stages,
        depth_max_rel: depth.max_abs / f64::from(top),
        whole: diff(&whole_map.data, want),
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
    for p in all {
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
