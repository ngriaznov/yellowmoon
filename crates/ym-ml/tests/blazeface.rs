//! The BlazeFace full-range port against Google's TFLite file (`docs/27` S-08, `docs/07`).
//!
//! The fixtures are `tests/fixtures/blazeface.json`, written by `tools/mediapipe/reference.py` from Google's own interpreter
//! (ai-edge-litert) and Google's own face detector task (mediapipe), never from our code. The weights are
//! `data/models/mediapipe/` (`tools/fetch/mediapipe.sh`); the photos are `testdata/mediapipe/photos/<id>.jpg`
//! (`tools/fetch/mediapipe.sh --photos`, CC0 on Wikimedia Commons, pinned by SHA-256, never committed). Without them the tests
//! print a plain "SKIPPED" line and pass; `YM_REQUIRE_MEDIAPIPE=1` makes a missing file a failure (CI sets it).
//!
//! The gates, with the worst value measured on the Linux reference machine on 2026-11-11 (see `docs/27` S-08, "As built"):
//!
//! | gate | bound | measured |
//! |---|---|---|
//! | network alone, synthetic input: largest difference of a logit | `LOGIT_TOL` | see `docs/27` |
//! | network alone: largest difference of a regressor, relative to the largest regressor | `REGRESSOR_REL_TOL` | see `docs/27` |
//!
//! Run in release mode for speed: `cargo test -p ym-ml --release --test blazeface`.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_ml::mediapipe::blazeface::{
    ANCHOR_COUNT, BlazeFace, Detection, FaceFinder, FinderParams, Letterbox, Resample, Rgb8,
    Window, prepare_tensor, weighted_nms,
};

/// Largest allowed distance between an edge of our box and MediaPipe's task, in pixels or as a fraction of the box (the task
/// rounds its box to whole pixels, our JPEG decoder and OpenCV's fixed-point resampling differ in the last bits).
const TASK_EDGE_PX: f32 = 2.5;
const TASK_EDGE_REL: f32 = 0.02;
/// Smallest IoU and largest score difference against the TFLite interpreter on MediaPipe's own tensor (our JPEG decoder and
/// OpenCV's fixed-point resampling are the only differences).
const INTERPRETER_IOU: f32 = 0.98;
const INTERPRETER_SCORE: f32 = 0.01;
/// Smallest IoU between the finder's default (box-filtered windows) and MediaPipe's task.
const FINDER_IOU: f32 = 0.85;
/// Largest allowed difference of a logit between our network and the TFLite interpreter on the synthetic input.
const LOGIT_TOL: f32 = 1e-4;
/// Largest allowed difference of a regressor, relative to the largest regressor of the fixture.
const REGRESSOR_REL_TOL: f32 = 1e-5;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn required() -> bool {
    std::env::var_os("YM_REQUIRE_MEDIAPIPE").is_some_and(|v| v != "0" && !v.is_empty())
}

fn skip_or_fail(what: &str) -> bool {
    assert!(!required(), "YM_REQUIRE_MEDIAPIPE is set but {what}");
    println!("SKIPPED: {what}");
    false
}

fn load_net() -> Option<BlazeFace> {
    let dir = repo().join("data/models/mediapipe");
    if !dir.join("blaze_face_full_range.f16.safetensors").is_file() {
        skip_or_fail(
            "no BlazeFace weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return None;
    }
    Some(BlazeFace::load(&dir).expect("the weights load and match SHA256SUMS"))
}

fn fixtures() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/blazeface.json");
    serde_json::from_slice(&std::fs::read(path).expect("fixtures")).expect("fixture json")
}

fn f(v: &Value) -> f32 {
    v.as_f64().expect("number") as f32
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().expect("array").iter().map(f).collect()
}

/// The synthetic input of `tools/mediapipe/reference.py::synthetic_input`, step by step in f64.
fn synthetic_input() -> Vec<f32> {
    (0u64..192 * 192 * 3)
        .map(|i| {
            let noise = ((i * 2_654_435_761 + 12345) & 0xFFFF_FFFF) as f64 / 4_294_967_296.0;
            let (c, x, y) = (i % 3, (i / 3) % 192, i / (3 * 192));
            let ramp = ((x + 2 * y + 40 * c) % 97) as f64 / 96.0;
            ((0.6 * ramp + 0.4 * noise) * 2.0 - 1.0) as f32
        })
        .collect()
}

#[test]
fn network_matches_tflite_on_a_synthetic_input() {
    let Some(net) = load_net() else { return };
    let fx = fixtures();
    let s = &fx["synthetic"];
    let raw = net.forward(&synthetic_input()).expect("forward");
    assert_eq!(raw.logits.len(), ANCHOR_COUNT);
    let anchors: Vec<usize> = s["anchors"]
        .as_array()
        .expect("anchors")
        .iter()
        .map(|v| v.as_u64().expect("index") as usize)
        .collect();
    let want_logits = floats(&s["logits"]);
    let reg_max = f(&s["regressor_abs_max"]);
    let (mut worst_logit, mut worst_reg) = (0.0f32, 0.0f32);
    for (k, &a) in anchors.iter().enumerate() {
        worst_logit = worst_logit.max((raw.logits[a] - want_logits[k]).abs());
        let want = floats(&s["regressors"][k]);
        for (j, w) in want.iter().enumerate() {
            worst_reg = worst_reg.max((raw.regressors[a * 16 + j] - w).abs() / reg_max);
        }
    }
    println!(
        "synthetic: worst logit difference {worst_logit:.3e}, worst regressor difference {worst_reg:.3e} of the largest"
    );
    assert!(worst_logit <= LOGIT_TOL, "logit {worst_logit}");
    assert!(worst_reg <= REGRESSOR_REL_TOL, "regressor {worst_reg}");
    // All anchors: the mean absolute value agrees with the interpreter's.
    let mean: f32 = raw.logits.iter().map(|v| v.abs()).sum::<f32>() / ANCHOR_COUNT as f32;
    assert!(
        (mean - f(&s["logit_abs_mean"])).abs() < 1e-3,
        "mean |logit| {mean}"
    );
}

#[test]
fn the_same_bits_on_one_and_four_threads() {
    let Some(net) = load_net() else { return };
    let input = synthetic_input();
    let a = net.forward(&input).expect("forward");
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("pool")
    };
    let one = pool(1).install(|| net.forward(&input).expect("forward"));
    let four = pool(4).install(|| net.forward(&input).expect("forward"));
    assert_eq!(a, one);
    assert_eq!(a, four);
}

fn photo_path(id: &str) -> PathBuf {
    repo()
        .join("testdata/mediapipe/photos")
        .join(format!("{id}.jpg"))
}

fn decode(path: &Path) -> (usize, usize, Vec<u8>) {
    let bytes = std::fs::read(path).expect("photo");
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes));
    let rgb = d.decode().expect("decode");
    let info = d.info().expect("header");
    (usize::from(info.width), usize::from(info.height), rgb)
}

fn photo_ids(fx: &Value) -> Vec<String> {
    fx["photos"]
        .as_object()
        .expect("photos")
        .keys()
        .cloned()
        .collect()
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d = |box_px| Detection {
        box_px,
        keypoints_px: [[0.0; 2]; 6],
        score: 1.0,
    };
    d(a).iou(&d(b))
}

#[test]
fn photos_match_mediapipe() {
    let Some(net) = load_net() else { return };
    let fx = fixtures();
    let mut ran = 0;
    for id in photo_ids(&fx) {
        let path = photo_path(&id);
        if !path.is_file() {
            skip_or_fail(&format!(
                "no photo {} (run tools/fetch/mediapipe.sh --photos)",
                path.display()
            ));
            continue;
        }
        let (w, h, rgb) = decode(&path);
        let p = &fx["photos"][id.as_str()];
        assert_eq!(
            (w as u64, h as u64),
            (p["width"].as_u64().unwrap(), p["height"].as_u64().unwrap())
        );
        let img = Rgb8::new(w, h, &rgb).expect("picture");
        // MediaPipe's own resampling, one pass on the whole picture, threshold and NMS of the reference.
        let got = net
            .detect_window(
                &img,
                Window::whole(w, h),
                Resample::MediaPipe,
                f(&fx["score_threshold"]),
                f(&fx["nms_iou"]),
            )
            .expect("detect");
        let task = p["mediapipe_task"].as_array().expect("task");
        let numpy = p["tflite_weighted_nms"].as_array().expect("nms");
        println!(
            "{id}: {} face(s) ours, {} from MediaPipe's task, {} from the interpreter",
            got.len(),
            task.len(),
            numpy.len()
        );
        assert_eq!(got.len(), task.len(), "{id}: number of faces");
        assert_eq!(
            got.len(),
            numpy.len(),
            "{id}: number of faces (interpreter)"
        );
        for (g, t) in got.iter().zip(task) {
            let tb: Vec<f32> = floats(&t["box"]);
            let tb = [tb[0], tb[1], tb[2], tb[3]];
            let size = tb[2] - tb[0];
            let v = iou(g.box_px, tb);
            // MediaPipe's task rounds the box to whole pixels, so compare edges in pixels, not only the IoU.
            let edge = (0..4)
                .map(|j| (g.box_px[j] - tb[j]).abs())
                .fold(0.0f32, f32::max);
            let kp_err = (0..6)
                .map(|k| {
                    let want = floats(&t["keypoints"][k]);
                    ((g.keypoints_px[k][0] - want[0]).powi(2)
                        + (g.keypoints_px[k][1] - want[1]).powi(2))
                    .sqrt()
                        / size
                })
                .fold(0.0f32, f32::max);
            let ds = (g.score - f(&t["score"])).abs();
            println!(
                "  task: IoU {v:.4}, worst edge {edge:.2} px, worst keypoint error {kp_err:.4} of the box, score difference {ds:.4}"
            );
            assert!(
                edge <= TASK_EDGE_PX.max(TASK_EDGE_REL * size),
                "{id}: edge difference {edge} px with the task"
            );
            assert!(kp_err <= 0.02, "{id}: keypoint error {kp_err}");
            assert!(ds <= 0.02, "{id}: score difference {ds}");
        }
        for (g, t) in got.iter().zip(numpy) {
            let tb = floats(&t["box"]);
            let v = iou(g.box_px, [tb[0], tb[1], tb[2], tb[3]]);
            let ds = (g.score - f(&t["score"])).abs();
            println!("  interpreter: IoU {v:.5}, score difference {ds:.5}");
            assert!(v >= INTERPRETER_IOU, "{id}: IoU with the interpreter {v}");
            assert!(ds <= INTERPRETER_SCORE, "{id}: score difference {ds}");
        }
        ran += 1;
    }
    println!("{ran} photo(s) compared");
}

#[test]
fn finder_defaults_find_the_same_faces() {
    let Some(net) = load_net() else { return };
    let fx = fixtures();
    let finder = FaceFinder::new(net);
    for id in photo_ids(&fx) {
        let path = photo_path(&id);
        if !path.is_file() {
            skip_or_fail(&format!("no photo {}", path.display()));
            continue;
        }
        let (w, h, rgb) = decode(&path);
        let img = Rgb8::new(w, h, &rgb).expect("picture");
        let faces = finder.find(&img, &FinderParams::default()).expect("find");
        let task = fx["photos"][id.as_str()]["mediapipe_task"]
            .as_array()
            .expect("task");
        println!("{id}: {} face(s), MediaPipe {}", faces.len(), task.len());
        assert_eq!(faces.len(), task.len(), "{id}");
        for (face, t) in faces.iter().zip(task) {
            let tb = floats(&t["box"]);
            let r = face.rect;
            let ours = [
                r.x_px as f32,
                r.y_px as f32,
                (r.x_px + r.width_px as i32) as f32,
                (r.y_px + r.height_px as i32) as f32,
            ];
            let v = iou(ours, [tb[0], tb[1], tb[2], tb[3]]);
            let edge = (0..4)
                .map(|j| (ours[j] - tb[j]).abs())
                .fold(0.0f32, f32::max);
            let size = tb[2] - tb[0];
            println!(
                "  IoU {v:.4}, worst edge {edge:.2} px of {size:.0}, score {:.3}",
                face.score
            );
            assert!(v >= FINDER_IOU, "{id}: IoU {v}");
        }
    }
}

#[test]
fn tensor_of_a_flat_picture_is_flat() {
    // No weights needed: a grey picture letterboxed gives the grey inside and -1 in the bars.
    let (w, h) = (400usize, 200usize);
    let data = vec![128u8; w * h * 3];
    let img = Rgb8::new(w, h, &data).expect("picture");
    let lb = Letterbox::new(Window::whole(w, h)).expect("window");
    for resample in [Resample::MediaPipe, Resample::Box] {
        let t = prepare_tensor(&img, &lb, resample);
        let at = |x: usize, y: usize| t[(y * 192 + x) * 3];
        let grey = 128.0 / 255.0 * 2.0 - 1.0;
        assert!(
            (at(96, 96) - grey).abs() < 1e-6,
            "{resample:?}: centre {}",
            at(96, 96)
        );
        assert!(
            (at(96, 5) + 1.0).abs() < 1e-6,
            "{resample:?}: top bar {}",
            at(96, 5)
        );
        assert!((at(96, 186) + 1.0).abs() < 1e-6, "{resample:?}: bottom bar");
    }
}

#[test]
fn weighted_nms_keeps_the_best_score() {
    let d = |x0, score| Detection {
        box_px: [x0, 0.0, x0 + 100.0, 100.0],
        keypoints_px: [[x0 + 50.0, 50.0]; 6],
        score,
    };
    let out = weighted_nms(vec![d(0.0, 0.9), d(10.0, 0.6), d(500.0, 0.7)], 0.3);
    assert_eq!(out.len(), 2);
    assert!((out[0].score - 0.9).abs() < 1e-6);
    let expect = (0.0 * 0.9 + 10.0 * 0.6) / 1.5;
    assert!((out[0].box_px[0] - expect).abs() < 1e-4);
    assert!((out[1].box_px[0] - 500.0).abs() < 1e-4);
}

/// A box-filtered copy of an RGB8 picture `new_w` pixels wide.
fn shrink(w: usize, h: usize, data: &[u8], new_w: usize) -> (usize, usize, Vec<u8>) {
    let new_h = (h * new_w).div_ceil(w);
    let mut out = vec![0u8; new_w * new_h * 3];
    for y in 0..new_h {
        let (y0, y1) = (y * h / new_h, ((y + 1) * h).div_ceil(new_h).min(h));
        for x in 0..new_w {
            let (x0, x1) = (x * w / new_w, ((x + 1) * w).div_ceil(new_w).min(w));
            let mut acc = [0u32; 3];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    for c in 0..3 {
                        acc[c] += u32::from(data[(yy * w + xx) * 3 + c]);
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            for c in 0..3 {
                out[(y * new_w + x) * 3 + c] = (acc[c] / n) as u8;
            }
        }
    }
    (new_w, new_h, out)
}

/// A group picture: width, height, RGB8 pixels and the planted face boxes.
type Group = (usize, usize, Vec<u8>, Vec<[f32; 4]>);

/// A 24 MP picture with the given photos pasted in at the given places (`id`, left, top, width); returns it with the faces
/// MediaPipe's task found in each photo, moved and scaled to the places they now have (`[x0, y0, x1, y1]`).
fn group_picture(fx: &Value, places: &[(&str, usize, usize, usize)]) -> Option<Group> {
    let (cw, ch) = (6000usize, 4000usize);
    let mut canvas = vec![0u8; cw * ch * 3];
    let mut seed = 12345u32;
    for y in 0..ch {
        for x in 0..cw {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 28) as usize;
            let v = (100 + (x * 5 + y * 3) % 50 + noise) as u8;
            canvas[(y * cw + x) * 3..][..3].copy_from_slice(&[v, v - 12, v - 25]);
        }
    }
    let mut faces = Vec::new();
    for &(id, left, top, width) in places {
        let path = photo_path(id);
        if !path.is_file() {
            skip_or_fail(&format!("no photo {}", path.display()));
            return None;
        }
        let (w, h, rgb) = decode(&path);
        let (nw, nh, small) = shrink(w, h, &rgb, width);
        for y in 0..nh {
            canvas[((top + y) * cw + left) * 3..][..nw * 3]
                .copy_from_slice(&small[y * nw * 3..][..nw * 3]);
        }
        let k = nw as f32 / w as f32;
        for t in fx["photos"][id]["mediapipe_task"].as_array().expect("task") {
            let b = floats(&t["box"]);
            faces.push([
                left as f32 + b[0] * k,
                top as f32 + b[1] * k,
                left as f32 + b[2] * k,
                top as f32 + b[3] * k,
            ]);
        }
    }
    Some((cw, ch, canvas, faces))
}

#[test]
fn a_24_megapixel_group_picture_gives_every_planted_face_once() {
    let Some(net) = load_net() else { return };
    let fx = fixtures();
    // Faces from 110 to 390 px (1.8 % to 6.5 % of the long side), and two of 49 and 61 px that are below the default minimum.
    let places = [
        ("mp-portrait", 300, 200, 1400),
        ("mp-two-people", 2300, 300, 2000),
        ("mp-portrait", 4800, 2600, 400),
        ("mp-woman-hands", 1200, 2900, 500),
        ("mp-two-people", 3000, 3300, 450),
    ];
    let Some((w, h, rgb, planted)) = group_picture(&fx, &places) else {
        return;
    };
    let img = Rgb8::new(w, h, &rgb).expect("picture");
    let finder = FaceFinder::new(net);
    let started = std::time::Instant::now();
    let faces = finder.find(&img, &FinderParams::default()).expect("find");
    println!(
        "24 MP group picture: {} face(s) for {} planted, {:.0} ms",
        faces.len(),
        planted.len(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    // Faces under the default minimum (1.5 % of 6000 = 90 px) are not looked for: here the two of the 450 px copy (49 and 61 px).
    let looked_for: Vec<&[f32; 4]> = planted
        .iter()
        .filter(|b| (b[2] - b[0]).max(b[3] - b[1]) >= 90.0)
        .collect();
    assert_eq!(faces.len(), looked_for.len(), "faces {faces:?}");
    for b in &looked_for {
        let best = faces
            .iter()
            .map(|f| {
                let r = f.rect;
                iou(
                    [
                        r.x_px as f32,
                        r.y_px as f32,
                        (r.x_px + r.width_px as i32) as f32,
                        (r.y_px + r.height_px as i32) as f32,
                    ],
                    **b,
                )
            })
            .fold(0.0f32, f32::max);
        println!("  planted {b:?}: best IoU {best:.3}");
        assert!(best >= 0.8, "no face near {b:?} (best IoU {best})");
    }
    // The scale of a face is its size over the long side.
    for f in &faces {
        assert!(f.scale > 0.015 && f.scale < 0.1, "scale {}", f.scale);
    }
    // Asking for smaller faces finds more or the same, never fewer.
    let small = finder
        .find(
            &img,
            &FinderParams {
                min_face_px: Some(30),
                ..FinderParams::default()
            },
        )
        .expect("find small");
    println!("min face 30 px: {} face(s)", small.len());
    assert!(small.len() >= faces.len());
    // A cap keeps the best.
    let capped = finder
        .find(
            &img,
            &FinderParams {
                max_faces: 2,
                ..FinderParams::default()
            },
        )
        .expect("find capped");
    assert_eq!(capped.len(), 2);
    assert!(capped[0].score >= capped[1].score);
    assert!((capped[0].score - faces[0].score).abs() < 1e-6);
    // The same faces on one thread and on four.
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("pool")
    };
    let one = pool(1).install(|| finder.find(&img, &FinderParams::default()).expect("find"));
    let four = pool(4).install(|| finder.find(&img, &FinderParams::default()).expect("find"));
    assert_eq!(one, faces);
    assert_eq!(four, faces);
}
