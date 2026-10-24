//! The bokeh map network against PyTorch (`training/bokeh/export.py` writes the fixture): the encoding of
//! a linear plane, the network's raw output on every fixture case (max abs error stated below; the
//! largest seen is printed), the conversion to `b` and `c`, and determinism (two runs, any thread count).
//!
//! The weights are the shipped ones in `data/models/bokeh/` with the fixture in
//! `crates/ym-ml/tests/data/bokehnet_parity.json`; until they are committed the tests use a smoke-run
//! export in `testdata/bokeh-smoke/` (git-ignored: `bokehnet.safetensors` and `parity.json`), and when
//! neither exists they print a note and pass.

use std::path::PathBuf;

use serde::Deserialize;
use ym_ml::bokeh_net::{
    BokehMap, BokehNet, FRAME_LONG_PX, PlanarRgbRef, box_reduce, encode_linear, pad_encoded,
    reduced_size,
};
use ym_ml::model::{MlError, sha256_hex};

/// The largest difference from PyTorch's (`f32`, folded) output the parity test accepts. The
/// fixture rounds its numbers to 7 decimals, and the two sides add in different orders.
const PARITY_MAX_ABS: f64 = 1e-4;
/// The largest difference from the fixture's encoded probes (`encode_np`, `f64`, 9 decimals).
const PROBE_MAX_ABS: f64 = 1e-6;

#[derive(Deserialize)]
struct Fixture {
    weights_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    kind: String,
    height: usize,
    width: usize,
    seed: Option<i64>,
    /// `(h/2, w/2, 2)` row-major: `(μ, s)` of every cell.
    output: Vec<f64>,
    #[serde(default)]
    encoded_probes: Vec<Probe>,
}

#[derive(Deserialize)]
struct Probe {
    c: usize,
    y: usize,
    x: usize,
    e: f64,
}

struct Located {
    weights: PathBuf,
    fixture: PathBuf,
    label: &'static str,
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn locate() -> Option<Located> {
    let shipped = Located {
        weights: repo().join("data/models/bokeh/bokehnet.safetensors"),
        fixture: repo().join("crates/ym-ml/tests/data/bokehnet_parity.json"),
        label: "data/models/bokeh",
    };
    let smoke = Located {
        weights: repo().join("testdata/bokeh-smoke/bokehnet.safetensors"),
        fixture: repo().join("testdata/bokeh-smoke/parity.json"),
        label: "testdata/bokeh-smoke (a smoke run, not the final model)",
    };
    [shipped, smoke]
        .into_iter()
        .find(|l| l.weights.exists() && l.fixture.exists())
}

/// The network and its fixture, or `None` (with a note) when no weights are available.
fn setup() -> Option<(BokehNet, Fixture, Located)> {
    let Some(found) = locate() else {
        println!(
            "NOTE: no bokeh weights (data/models/bokeh/ or testdata/bokeh-smoke/): test skipped"
        );
        return None;
    };
    let net = BokehNet::from_file(&found.weights).expect("the weights load");
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(&found.fixture).expect("fixture file"))
            .expect("fixture json");
    println!("weights: {}", found.label);
    Some((net, fixture, found))
}

/// `training/bokeh/export.py`'s `pattern_e`: the encoded input from an integer pattern, planar.
fn pattern_e(h: usize, w: usize, seed: i64) -> Vec<f32> {
    let (hi, wi) = (h as i64, w as i64);
    let mut out = Vec::with_capacity(3 * h * w);
    for c in 0..3i64 {
        for y in 0..hi {
            for x in 0..wi {
                let v = (x * (c + 2) * 5 + y * (3 + seed) * 3 + ((x * y) >> 4) * (7 + c) + seed * 41)
                    % 256;
                let block = x * 5 >= wi * 2 && x * 5 < wi * 3 && y * 3 >= hi && y * 3 < hi * 2;
                let v = if block { 255 - v / 4 } else { v };
                out.push(v as f32 / 255.0f32);
            }
        }
    }
    out
}

/// `export.py`'s `pattern_linear`: exactly representable linear RGB, planar `f64`.
fn pattern_linear(h: usize, w: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let v = (1 + ((x * (c + 3) + y * 5 + c * 7) % 61)) as f64 / 16.0;
                let patch = x >= w / 4 && x < w / 4 + w / 3 && y >= h / 4 && y < h / 4 + h / 3;
                out.push(if patch { 40.0 } else { v });
            }
        }
    }
    out
}

/// A deterministic synthetic lens plane: a smooth sky-to-ground gradient, a sharp-edged block, a few
/// bright discs and noise, planar `f32`, linear light.
fn scene(w: usize, h: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut noise = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut out = Vec::with_capacity(3 * w * h);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                let mut v = 0.05 + 0.4 * fy + 0.1 * fx * (1.0 + c as f32 * 0.2);
                if (0.3..0.5).contains(&fx) && (0.4..0.7).contains(&fy) {
                    v += 0.5;
                }
                for (cx, cy) in [(0.7, 0.3), (0.8, 0.6), (0.2, 0.2)] {
                    let d = ((fx - cx) * w as f32).hypot((fy - cy) * w as f32);
                    if d < 0.03 * w as f32 {
                        v += 8.0;
                    }
                }
                out.push(v * (0.9 + 0.2 * noise()));
            }
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn map_bits(m: &BokehMap) -> (Vec<u32>, Vec<u32>) {
    (bits(&m.b), bits(&m.c))
}

#[test]
fn the_weights_are_the_fixture_s_and_the_network_is_the_trained_one() {
    let Some((net, fixture, _)) = setup() else {
        return;
    };
    assert_eq!(net.sha256(), fixture.weights_sha256);
    // `training/bokeh/README.md`: 1,522,122 parameters with the batch norms folded.
    assert_eq!(net.parameters(), 1_522_122);
}

#[test]
fn the_encoding_reproduces_the_fixture_s_probes() {
    let Some((_, fixture, _)) = setup() else {
        return;
    };
    let mut worst = 0.0f64;
    let mut probes = 0;
    for case in fixture.cases.iter().filter(|c| c.kind == "linear") {
        let (h, w) = (case.height, case.width);
        let e = encode_linear(&pattern_linear(h, w), w, h).expect("encode");
        for p in &case.encoded_probes {
            worst = worst.max((e[(p.c * h + p.y) * w + p.x] - p.e).abs());
            probes += 1;
        }
    }
    println!("encoded probes: {probes}, largest difference from encode.py: {worst:.2e}");
    assert!(probes >= 12, "the fixture holds the linear case's probes");
    assert!(worst < PROBE_MAX_ABS, "{worst}");
}

#[test]
fn the_network_agrees_with_pytorch_on_every_fixture_case() {
    let Some((net, fixture, _)) = setup() else {
        return;
    };
    let mut worst = 0.0f64;
    for case in &fixture.cases {
        let (h, w) = (case.height, case.width);
        let e = match case.kind.as_str() {
            "encoded" => pattern_e(h, w, case.seed.expect("seed")),
            "linear" => {
                // Through the encoder, as the app feeds it (64 is a multiple of 32: no padding).
                let encoded = encode_linear(&pattern_linear(h, w), w, h).expect("encode");
                pad_encoded(&encoded, w, h).0
            }
            other => panic!("unknown fixture case kind {other}"),
        };
        let raw = net.forward_padded(&e, w, h).expect("forward");
        let (w2, h2) = (w / 2, h / 2);
        assert_eq!((raw.width, raw.height), (w2, h2));
        assert_eq!(case.output.len(), w2 * h2 * 2);
        let mut case_worst = 0.0f64;
        for i in 0..w2 * h2 {
            case_worst = case_worst
                .max((f64::from(raw.mu[i]) - case.output[2 * i]).abs())
                .max((f64::from(raw.s[i]) - case.output[2 * i + 1]).abs());
        }
        println!(
            "{} {w} x {h}: largest difference from PyTorch {case_worst:.2e}",
            case.kind
        );
        worst = worst.max(case_worst);
    }
    println!("largest difference from PyTorch over all cases: {worst:.2e}");
    assert!(worst < PARITY_MAX_ABS, "{worst}");
}

#[test]
fn a_lens_plane_goes_through_the_whole_pipeline_to_the_fixture_s_numbers() {
    // The linear case as a lens plane (`f32`, exactly representable): `map` reduces (not at 64 px),
    // encodes, runs and converts, and must give `b` and `c` from the fixture's `(μ, s)`.
    let Some((net, fixture, _)) = setup() else {
        return;
    };
    let case = fixture
        .cases
        .iter()
        .find(|c| c.kind == "linear")
        .expect("a linear case");
    let (h, w) = (case.height, case.width);
    let plane: Vec<f32> = pattern_linear(h, w).iter().map(|&v| v as f32).collect();
    let map = net
        .map(&PlanarRgbRef::new(w, h, &plane).expect("plane"))
        .expect("map");
    assert_eq!((map.width, map.height), (w / 2, h / 2));
    assert_eq!((map.plane_width, map.plane_height), (w, h));
    let hd = 0.5 * (w as f64).hypot(h as f64);
    assert!((map.half_diagonal_px - hd).abs() < 1e-12);
    let mut worst = 0.0f64;
    for i in 0..map.width * map.height {
        let (mu, s) = (case.output[2 * i], case.output[2 * i + 1]);
        let b = mu.max(0.0).exp() / hd;
        let c = (-s.clamp(-6.0, 3.0).exp() / 0.25).exp();
        worst = worst
            .max((f64::from(map.b[i]) - b).abs() * hd) // in pixels
            .max((f64::from(map.c[i]) - c).abs());
    }
    println!("map vs fixture: largest difference (b in px, c) {worst:.2e}");
    assert!(worst < 1e-3, "{worst}");
    assert!(map.b.iter().all(|&b| b >= (1.0 / hd) as f32 * 0.999));
    assert!(map.c.iter().all(|&c| (0.0..=1.0).contains(&c)));
}

#[test]
fn the_same_plane_gives_the_same_bits_on_any_number_of_threads() {
    let Some((net, _, _)) = setup() else {
        return;
    };
    // 333 x 211 needs padding to 352 x 224 and a map of 167 x 106; 1100 x 64 is longer than 1024 px,
    // so it is reduced first (to 1024 x 60) and padded (to 1024 x 64).
    for (w, h, seed) in [(333usize, 211usize, 1u32), (1100, 64, 2)] {
        let data = scene(w, h, seed);
        let plane = PlanarRgbRef::new(w, h, &data).expect("plane");
        let first = net.map(&plane).expect("map");
        let again = net.map(&plane).expect("map");
        assert_eq!(map_bits(&first), map_bits(&again), "two runs, {w} x {h}");
        for threads in [1, 2, 3, 5] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let got = pool.install(|| net.map(&plane).expect("map"));
            assert_eq!(
                map_bits(&first),
                map_bits(&got),
                "{threads} threads, {w} x {h}"
            );
        }
        let (rw, rh) = reduced_size(w, h);
        assert_eq!(
            (first.plane_width, first.plane_height, first.width, first.height),
            (rw, rh, rw.div_ceil(2), rh.div_ceil(2))
        );
        let (b, c) = map_bits(&first);
        let bytes: Vec<u8> = b.iter().chain(&c).flat_map(|v| v.to_le_bytes()).collect();
        println!(
            "{w} x {h} -> {rw} x {rh} -> map {} x {}: output SHA-256 {}",
            first.width,
            first.height,
            sha256_hex(&bytes)
        );
    }
}

#[test]
fn a_plane_twice_as_bright_gives_the_same_map() {
    let Some((net, _, _)) = setup() else {
        return;
    };
    let (w, h) = (200, 150);
    let data = scene(w, h, 3);
    let doubled: Vec<f32> = data.iter().map(|v| v * 2.0).collect();
    let a = net
        .map(&PlanarRgbRef::new(w, h, &data).expect("plane"))
        .expect("map");
    let b = net
        .map(&PlanarRgbRef::new(w, h, &doubled).expect("plane"))
        .expect("map");
    assert_eq!(map_bits(&a), map_bits(&b));
}

#[test]
fn the_map_is_the_stages_one_after_the_other() {
    let Some((net, _, _)) = setup() else {
        return;
    };
    let (w, h) = (1100, 64);
    let data = scene(w, h, 4);
    let plane = PlanarRgbRef::new(w, h, &data).expect("plane");
    let (rw, rh) = reduced_size(w, h);
    assert_eq!(rw, FRAME_LONG_PX);
    let reduced = box_reduce(&plane, rw, rh).expect("reduce");
    let encoded = encode_linear(&reduced, rw, rh).expect("encode");
    let (padded, pw, ph) = pad_encoded(&encoded, rw, rh);
    assert_eq!((pw, ph), (1024, 64));
    let by_hand = net
        .forward_padded(&padded, pw, ph)
        .expect("forward")
        .into_map(rw, rh);
    assert_eq!(map_bits(&by_hand), map_bits(&net.map(&plane).expect("map")));
    assert_eq!((by_hand.width, by_hand.height), (512, 30));
}

#[test]
fn bad_planes_are_refused_not_run() {
    let Some((net, _, _)) = setup() else {
        return;
    };
    let mut data = scene(40, 30, 5);
    assert!(PlanarRgbRef::new(40, 30, &data[1..]).is_err());
    assert!(PlanarRgbRef::new(0, 30, &[]).is_err());
    data[17] = f32::NAN;
    let plane = PlanarRgbRef {
        width: 40,
        height: 30,
        data: &data,
    };
    assert!(matches!(net.map(&plane), Err(MlError::Input(_))));
    assert!(net.forward_padded(&[0.0; 3 * 33 * 32], 33, 32).is_err());
    assert!(net.forward_padded(&[0.0; 10], 32, 32).is_err());
}

#[test]
fn a_damaged_or_foreign_file_is_refused() {
    let Some((_, _, found)) = setup() else {
        return;
    };
    let bytes = std::fs::read(&found.weights).expect("weights");
    assert!(BokehNet::from_bytes(&bytes, Some(&"0".repeat(64))).is_err());
    assert!(BokehNet::from_bytes(&bytes[..1000], None).is_err());
    assert!(BokehNet::from_bytes(&bytes[..bytes.len() - 4], None).is_err());
    // The crop model's file is a safetensors file of another format.
    let crop = std::fs::read(repo().join("data/models/crop/cropnet.safetensors")).expect("crop");
    assert!(matches!(
        BokehNet::from_bytes(&crop, None),
        Err(MlError::Weights(_))
    ));
    // Same-length edits of the header: another format, another version, a renamed tensor.
    let edit = |from: &str, to: &str| {
        assert_eq!(from.len(), to.len());
        let at = bytes
            .windows(from.len())
            .position(|w| w == from.as_bytes())
            .expect("text in the header");
        let mut copy = bytes.clone();
        copy[at..at + from.len()].copy_from_slice(to.as_bytes());
        copy
    };
    assert!(matches!(
        BokehNet::from_bytes(&edit("\"ym-bokehnet\"", "\"ym-bokehxxx\""), None),
        Err(MlError::Weights(_))
    ));
    assert!(matches!(
        BokehNet::from_bytes(&edit("\"version\":\"1\"", "\"version\":\"2\""), None),
        Err(MlError::Unsupported(_))
    ));
    assert!(matches!(
        BokehNet::from_bytes(&edit("\"stem.bias\"", "\"stem.biaz\""), None),
        Err(MlError::Weights(_))
    ));
    assert!(matches!(
        BokehNet::from_bytes(&edit("\"relu6\"", "\"silu_\""), None),
        Err(MlError::Unsupported(_))
    ));
}

#[test]
fn the_model_folder_is_checked_against_its_sha256sums() {
    // A folder with the weights and a SHA256SUMS line loads; with a wrong line it does not.
    let Some((_, _, found)) = setup() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let bytes = std::fs::read(&found.weights).expect("weights");
    std::fs::write(dir.path().join("bokehnet.safetensors"), &bytes).expect("write");
    assert!(BokehNet::load(dir.path()).is_err(), "no SHA256SUMS");
    let line = |hash: &str| format!("{hash}  bokehnet.safetensors\n");
    std::fs::write(dir.path().join("SHA256SUMS"), line(&"0".repeat(64))).expect("write");
    assert!(matches!(
        BokehNet::load(dir.path()),
        Err(MlError::Checksum { .. })
    ));
    std::fs::write(dir.path().join("SHA256SUMS"), line(&sha256_hex(&bytes))).expect("write");
    let net = BokehNet::load(dir.path()).expect("loads");
    assert_eq!(net.sha256(), sha256_hex(&bytes));
}

#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn timing_of_one_map() {
    let Some(found) = locate() else {
        println!("NOTE: no bokeh weights: timing skipped");
        return;
    };
    let t = std::time::Instant::now();
    let net = BokehNet::from_file(&found.weights).expect("load");
    println!("load: {:.0} ms ({})", t.elapsed().as_secs_f64() * 1e3, found.label);
    println!("rayon threads: {}", rayon::current_num_threads());
    for (w, h) in [(1024usize, 683usize), (1512, 1006)] {
        let data = scene(w, h, 6);
        let plane = PlanarRgbRef::new(w, h, &data).expect("plane");
        net.map(&plane).expect("warm-up");
        let runs = 3;
        let mut total = 0.0;
        let mut parts = [0.0f64; 4];
        for _ in 0..runs {
            let t = std::time::Instant::now();
            let (_, tm) = net.map_timed(&plane).expect("map");
            total += t.elapsed().as_secs_f64() * 1e3;
            parts[0] += tm.reduce_ms;
            parts[1] += tm.encode_ms;
            parts[2] += tm.forward_ms;
            parts[3] += tm.convert_ms;
        }
        let n = f64::from(runs);
        println!(
            "{w} x {h}: {:.0} ms per map (reduce {:.0}, encode {:.0}, network {:.0}, convert {:.0})",
            total / n,
            parts[0] / n,
            parts[1] / n,
            parts[2] / n,
            parts[3] / n
        );
    }
}
