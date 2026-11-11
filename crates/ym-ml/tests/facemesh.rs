//! The Face Mesh V2 and Selfie Multiclass ports against Google's TFLite files (`docs/27` S-09 and S-10, `docs/07`).
//!
//! The fixtures are `tests/fixtures/faces.json`, written by `tools/mediapipe/reference_faces.py` from Google's own
//! interpreter (ai-edge-litert) and Google's own `FaceLandmarker` task (mediapipe), never from our code. The weights are
//! `data/models/mediapipe/` (`tools/fetch/mediapipe.sh`); the photos are `testdata/mediapipe/photos/<id>.jpg`
//! (`tools/fetch/mediapipe.sh --photos`, pinned by SHA-256, never committed). Without them the tests print a plain
//! "SKIPPED" line and pass; `YM_REQUIRE_MEDIAPIPE=1` makes a missing file a failure (CI sets it).
//!
//! The gates and the worst value measured on the Linux reference machine on 2026-11-11 (`docs/27`, "As built (S-09)" and
//! "As built (S-10)"):
//!
//! | gate | bound | measured |
//! |---|---|---|
//! | mesh alone, synthetic input: largest landmark difference over the largest landmark (220.7 px) | `MESH_REL_TOL` 1e-5 | 2.6e-6 |
//! | mesh alone: presence logit | `PRESENCE_TOL` 2e-3 | 1.8e-5 |
//! | segmenter alone, synthetic input: largest probability difference | `SELFIE_PROB_TOL` 1e-4 | 5.0e-6 |
//! | one pass on the reference's crop: mean / worst landmark distance over the crop's side | 0.0010 / 0.0040 | 0.00041 / 0.00185 |
//! | segmenter on the same crop: mean difference / share within 0.05 / argmax agreement | 0.004 / 0.99 / 0.99 | 0.0011 / 1.0 / 0.9961 |
//! | our finder and two passes against the reference's own chain: mean distance over the face | `REFERENCE_MEAN_TOL` 0.004 | 0.0014 |
//! | box only (no eye keypoints): unturned first pass and two refinements: mean distance over the face | `BOX_ONLY_MEAN_TOL` 0.01 | 0.0047 |
//! | the same against MediaPipe's task (its own detector, one pass): mean distance over the face | `TASK_MEAN_TOL` 0.02 | 0.0126 |
//!
//! Run in release mode for speed: `cargo test -p ym-ml --release --test facemesh`.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Resample, Rgb8};
use ym_ml::mediapipe::facemesh::{
    FaceLandmarker, FaceMeshNet, LANDMARKS, LandmarkerParams, roi_from_landmarks,
};
use ym_ml::mediapipe::roi::Roi;
use ym_ml::mediapipe::selfie::{CLASSES, INPUT_PX, SelfieNet, SelfieSegmenter};

/// The network alone on the synthetic input: largest landmark difference relative to the largest landmark (up to 223 px),
/// and the presence logit.
const MESH_REL_TOL: f32 = 1e-5;
const PRESENCE_TOL: f32 = 2e-3;
/// Selfie Multiclass alone on the synthetic input: largest probability difference.
const SELFIE_PROB_TOL: f32 = 1e-4;
/// A photo's first pass on the same crop (OpenCV's fixed-point resampling is the only difference): the mean and the worst
/// landmark distance as a fraction of the crop's side.
const PASS_MEAN_TOL: f32 = 0.0010;
const PASS_WORST_TOL: f32 = 0.0040;
/// The segmenter on the same crop: mean absolute probability difference, share of samples within 0.05, argmax agreement.
const SEG_MEAN_TOL: f32 = 0.004;
const SEG_WITHIN: f32 = 0.99;
const SEG_ARGMAX: f32 = 0.99;
/// The whole chain (our finder, two passes) against MediaPipe's own task (its own detector, one pass): the mean distance of
/// the landmarks as a fraction of the face's size.
const TASK_MEAN_TOL: f32 = 0.02;
/// The same chain against the reference's own second pass (same detections and crops, OpenCV's resampling): the mean
/// distance as a fraction of the face's size.
const REFERENCE_MEAN_TOL: f32 = 0.004;
/// The box-only path (unturned first pass, two refinements) against the reference's second pass: the mean distance over the
/// face.
const BOX_ONLY_MEAN_TOL: f32 = 0.01;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn required() -> bool {
    std::env::var_os("YM_REQUIRE_MEDIAPIPE").is_some_and(|v| v != "0" && !v.is_empty())
}

fn skip_or_fail(what: &str) {
    assert!(!required(), "YM_REQUIRE_MEDIAPIPE is set but {what}");
    println!("SKIPPED: {what}");
}

fn dir() -> PathBuf {
    repo().join("data/models/mediapipe")
}

fn have(file: &str) -> bool {
    dir().join(file).is_file()
}

fn load_mesh() -> Option<FaceLandmarker> {
    if !have("face_mesh_v2.f16.safetensors") {
        skip_or_fail(
            "no Face Mesh V2 weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return None;
    }
    Some(FaceLandmarker::new(
        FaceMeshNet::load(&dir()).expect("the weights load and match SHA256SUMS"),
    ))
}

fn load_selfie() -> Option<SelfieSegmenter> {
    if !have("selfie_multiclass_256.f32.safetensors") {
        skip_or_fail(
            "no Selfie Multiclass weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return None;
    }
    Some(SelfieSegmenter::new(
        SelfieNet::load(&dir()).expect("the weights load and match SHA256SUMS"),
    ))
}

fn fixtures() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/faces.json");
    serde_json::from_slice(&std::fs::read(path).expect("fixtures")).expect("fixture json")
}

fn f(v: &Value) -> f32 {
    v.as_f64().expect("number") as f32
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().expect("array").iter().map(f).collect()
}

/// `tools/mediapipe/reference_faces.py::synthetic_input`, step by step in f64.
fn synthetic_input() -> Vec<f32> {
    (0u64..256 * 256 * 3)
        .map(|i| {
            let noise = ((i * 2_654_435_761 + 12345) & 0xFFFF_FFFF) as f64 / 4_294_967_296.0;
            let (c, x, y) = (i % 3, (i / 3) % 256, i / (3 * 256));
            let ramp = ((x + 2 * y + 40 * c) % 97) as f64 / 96.0;
            (0.6 * ramp + 0.4 * noise) as f32
        })
        .collect()
}

fn roi_of(v: &Value) -> Roi {
    Roi {
        center_px: [
            v["center"][0].as_f64().expect("x"),
            v["center"][1].as_f64().expect("y"),
        ],
        side_px: v["side"].as_f64().expect("side"),
        rotation_rad: v["rotation"].as_f64().expect("rotation"),
    }
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

/// Calls `body` for every photo of the fixture that is on disk.
fn each_photo(mut body: impl FnMut(&str, &Value, usize, usize, &[u8])) -> usize {
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
        body(&id, p, w, h, &rgb);
        ran += 1;
    }
    ran
}

#[test]
fn face_mesh_matches_tflite_on_a_synthetic_input() {
    let Some(lm) = load_mesh() else { return };
    let fx = fixtures();
    let s = &fx["mesh_synthetic"];
    let raw = lm.net().forward(&synthetic_input()).expect("forward");
    assert_eq!(raw.landmarks.len(), LANDMARKS);
    let want = floats(&s["landmarks"]);
    let max = f(&s["landmark_abs_max"]);
    let worst = raw
        .landmarks
        .iter()
        .flatten()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let presence = (raw.presence_logit - f(&s["presence_logit"])).abs();
    println!(
        "mesh synthetic: worst landmark difference {worst:.3e} of {max} (relative {:.3e}), presence logit {presence:.3e}",
        worst / max
    );
    assert!(worst / max <= MESH_REL_TOL, "landmark {worst}");
    assert!(presence <= PRESENCE_TOL, "presence {presence}");
}

#[test]
fn selfie_multiclass_matches_tflite_on_a_synthetic_input() {
    let Some(seg) = load_selfie() else { return };
    let fx = fixtures();
    let s = &fx["selfie_synthetic"];
    let probs = seg.net().forward(&synthetic_input()).expect("forward");
    assert_eq!(probs.len(), INPUT_PX * INPUT_PX * CLASSES);
    // Every pixel is a probability distribution.
    for px in probs.chunks_exact(CLASSES) {
        let sum: f32 = px.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    let stride = s["stride"].as_u64().expect("stride") as usize;
    let want = floats(&s["probs"]);
    let n = INPUT_PX / stride;
    let mut worst = 0.0f32;
    for j in 0..n {
        for i in 0..n {
            for c in 0..CLASSES {
                let got = probs[((j * stride) * INPUT_PX + i * stride) * CLASSES + c];
                worst = worst.max((got - want[(j * n + i) * CLASSES + c]).abs());
            }
        }
    }
    let means = floats(&s["class_means"]);
    let mut worst_mean = 0.0f32;
    for c in 0..CLASSES {
        let m: f32 =
            probs.chunks_exact(CLASSES).map(|p| p[c]).sum::<f32>() / (INPUT_PX * INPUT_PX) as f32;
        worst_mean = worst_mean.max((m - means[c]).abs());
    }
    println!(
        "selfie synthetic: worst probability difference {worst:.3e}, worst class mean {worst_mean:.3e}"
    );
    assert!(worst <= SELFIE_PROB_TOL, "probability {worst}");
    assert!(worst_mean <= SELFIE_PROB_TOL, "class mean {worst_mean}");
}

#[test]
fn the_same_bits_on_one_and_four_threads() {
    let (Some(lm), Some(seg)) = (load_mesh(), load_selfie()) else {
        return;
    };
    let input = synthetic_input();
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("pool")
    };
    let a = lm.net().forward(&input).expect("forward");
    assert_eq!(
        a,
        pool(1).install(|| lm.net().forward(&input).expect("forward"))
    );
    assert_eq!(
        a,
        pool(4).install(|| lm.net().forward(&input).expect("forward"))
    );
    let b = seg.net().forward(&input).expect("forward");
    assert_eq!(
        b,
        pool(1).install(|| seg.net().forward(&input).expect("forward"))
    );
    assert_eq!(
        b,
        pool(4).install(|| seg.net().forward(&input).expect("forward"))
    );
}

/// The distance of each stored landmark from ours, as a fraction of `size`: (mean, worst).
fn landmark_error(got: &[[f32; 3]], want: &Value, stride: usize, size: f32) -> (f32, f32) {
    let want = want.as_array().expect("landmarks");
    let (mut sum, mut worst) = (0.0f32, 0.0f32);
    for (k, w) in want.iter().enumerate() {
        let g = got[k * stride];
        let d = (g[0] - f(&w[0])).hypot(g[1] - f(&w[1])) / size;
        sum += d;
        worst = worst.max(d);
    }
    (sum / want.len() as f32, worst)
}

#[test]
fn both_passes_match_tflite_on_the_same_crops() {
    let Some(lm) = load_mesh() else { return };
    let fx = fixtures();
    let stride = fx["landmark_stride"].as_u64().expect("stride") as usize;
    let mut faces = 0;
    let ran = each_photo(|id, p, w, h, rgb| {
        let img = Rgb8::new(w, h, rgb).expect("picture");
        for (k, face) in p["faces"].as_array().expect("faces").iter().enumerate() {
            let roi1 = roi_of(&face["roi1"]);
            let first = lm.pass(&img, &roi1, Resample::MediaPipe).expect("pass");
            let (mean1, worst1) = landmark_error(
                &first.points_px,
                &face["pass1"],
                stride,
                roi1.side_px as f32,
            );
            // The second crop from our first landmarks against the reference's.
            let roi2 = roi_from_landmarks(&first.points_px, 1.5).expect("roi");
            let want2 = roi_of(&face["roi2"]);
            let centre = (roi2.center_px[0] - want2.center_px[0])
                .hypot(roi2.center_px[1] - want2.center_px[1]);
            let second = lm.pass(&img, &roi2, Resample::MediaPipe).expect("pass");
            let (mean2, worst2) = landmark_error(
                &second.points_px,
                &face["pass2"],
                stride,
                roi2.side_px as f32,
            );
            println!(
                "{id} face {k}: pass 1 mean {mean1:.5} worst {worst1:.5} of the crop; roi2 centre off {centre:.3} px, side {:.3} px; pass 2 mean {mean2:.5} worst {worst2:.5}",
                (roi2.side_px - want2.side_px).abs()
            );
            for (mean, worst) in [(mean1, worst1), (mean2, worst2)] {
                assert!(mean <= PASS_MEAN_TOL, "{id} mean {mean}");
                assert!(worst <= PASS_WORST_TOL, "{id} worst {worst}");
            }
            assert!(first.presence > 0.99 && second.presence > 0.99);
            faces += 1;
        }
    });
    if ran > 0 {
        assert!(faces >= 3);
    }
}

#[test]
fn the_segmenter_matches_tflite_on_the_same_crops() {
    let (Some(lm), Some(seg)) = (load_mesh(), load_selfie()) else {
        return;
    };
    let fx = fixtures();
    let stride = fx["selfie_stride"].as_u64().expect("stride") as usize;
    each_photo(|id, p, w, h, rgb| {
        let img = Rgb8::new(w, h, rgb).expect("picture");
        for (k, face) in p["faces"].as_array().expect("faces").iter().enumerate() {
            // Our own chain to the segmenter's crop, from the reference's first crop.
            let first = lm
                .pass(&img, &roi_of(&face["roi1"]), Resample::MediaPipe)
                .expect("pass");
            let roi2 = roi_from_landmarks(&first.points_px, 1.5).expect("roi");
            let second = lm.pass(&img, &roi2, Resample::MediaPipe).expect("pass");
            let roi = roi_from_landmarks(
                &second.points_px,
                fx["selfie_scale"].as_f64().expect("scale"),
            )
            .expect("roi");
            let map = seg
                .segment(&img, &roi, Resample::MediaPipe)
                .expect("segment");
            let want = floats(&face["selfie_probs"]);
            let n = INPUT_PX / stride;
            let (mut sum, mut within, mut agree, mut count) = (0.0f32, 0usize, 0usize, 0usize);
            for j in 0..n {
                for i in 0..n {
                    let got =
                        &map.probs[((j * stride) * INPUT_PX + i * stride) * CLASSES..][..CLASSES];
                    let exp = &want[(j * n + i) * CLASSES..][..CLASSES];
                    let diff = got
                        .iter()
                        .zip(exp)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    sum += got.iter().zip(exp).map(|(a, b)| (a - b).abs()).sum::<f32>()
                        / CLASSES as f32;
                    within += usize::from(diff <= 0.05);
                    let arg = |v: &[f32]| {
                        v.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|x| x.0)
                    };
                    agree += usize::from(arg(got) == arg(exp));
                    count += 1;
                }
            }
            let (mean, within, agree) = (
                sum / count as f32,
                within as f32 / count as f32,
                agree as f32 / count as f32,
            );
            println!(
                "{id} face {k}: segmenter mean difference {mean:.5}, within 0.05 {within:.4}, argmax agreement {agree:.4}"
            );
            assert!(mean <= SEG_MEAN_TOL, "{id} mean {mean}");
            assert!(within >= SEG_WITHIN, "{id} within {within}");
            assert!(agree >= SEG_ARGMAX, "{id} argmax {agree}");
        }
    });
}

#[test]
fn the_whole_chain_agrees_with_mediapipes_task() {
    let Some(lm) = load_mesh() else { return };
    if !have("blaze_face_full_range.f16.safetensors") {
        skip_or_fail(
            "no BlazeFace weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return;
    }
    let finder = FaceFinder::new(BlazeFace::load(&dir()).expect("blazeface"));
    let fx = fixtures();
    let stride = fx["landmark_stride"].as_u64().expect("stride") as usize;
    each_photo(|id, p, w, h, rgb| {
        let img = Rgb8::new(w, h, rgb).expect("picture");
        let found = finder.find(&img, &FinderParams::default()).expect("find");
        let faces = p["faces"].as_array().expect("faces");
        assert_eq!(found.len(), faces.len(), "{id}: faces found");
        for det in &found {
            let Some(got) = lm
                .landmarks(&img, det, &LandmarkerParams::default())
                .expect("landmarks")
            else {
                panic!("{id}: no landmarks for a face the finder found");
            };
            assert_eq!(got.points_px.len(), LANDMARKS);
            // The reference face whose centre is nearest.
            let cx = f64::from(det.rect.x_px) + f64::from(det.rect.width_px) / 2.0;
            let cy = f64::from(det.rect.y_px) + f64::from(det.rect.height_px) / 2.0;
            let reference = faces
                .iter()
                .min_by(|a, b| {
                    let d = |face: &Value| {
                        let b = &face["detection"]["box"];
                        ((b[0].as_f64().unwrap() + b[2].as_f64().unwrap()) / 2.0 - cx)
                            .hypot((b[1].as_f64().unwrap() + b[3].as_f64().unwrap()) / 2.0 - cy)
                    };
                    d(a).total_cmp(&d(b))
                })
                .expect("a face");
            if reference["task"].is_null() {
                continue;
            }
            let size = f32::from(det.rect.width_px.max(det.rect.height_px) as u16);
            let (mean, worst) = landmark_error(&got.points_px, &reference["task"], stride, size);
            let (mean_ref, _) = landmark_error(&got.points_px, &reference["pass2"], stride, size);
            println!(
                "{id}: against the task mean {mean:.5} worst {worst:.5} of the face; against the reference's own second pass {mean_ref:.5}"
            );
            assert!(mean <= TASK_MEAN_TOL, "{id} mean {mean}");
            assert!(
                mean_ref <= REFERENCE_MEAN_TOL,
                "{id} mean against the reference {mean_ref}"
            );
        }
    });
}

#[test]
fn a_crop_without_a_face_has_a_low_presence() {
    let Some(lm) = load_mesh() else { return };
    // A smooth gradient picture: no face.
    let (w, h) = (320usize, 320usize);
    let data: Vec<f32> = (0..w * h * 3)
        .map(|i| (((i / 3) % w) as f32 / w as f32) * 0.5 + 0.2)
        .collect();
    let img = ym_ml::mediapipe::blazeface::RgbF32::new(w, h, &data).expect("picture");
    let roi = Roi {
        center_px: [160.0, 160.0],
        side_px: 200.0,
        rotation_rad: 0.0,
    };
    let got = lm.pass(&img, &roi, Resample::Box).expect("pass");
    println!("presence of a gradient: {:.4}", got.presence);
    assert!(got.presence < 0.5);
}

#[test]
fn the_chain_gives_the_same_bits_on_one_and_four_threads() {
    let Some(lm) = load_mesh() else { return };
    let Some(seg) = load_selfie() else { return };
    each_photo(|_, p, w, h, rgb| {
        let img = Rgb8::new(w, h, rgb).expect("picture");
        let face = &p["faces"][0];
        let roi = roi_of(&face["roi1"]);
        let pool = |n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("pool")
        };
        let run = || {
            let l = lm.pass(&img, &roi, Resample::Box).expect("pass");
            let s = seg.segment(&img, &roi, Resample::Box).expect("segment");
            (l, s)
        };
        assert_eq!(pool(1).install(run), pool(4).install(run));
    });
}

/// The path of `FaceParser`, which has only a box: an unturned first pass and two refinements, against the reference's own
/// second pass (the faces are tilted by up to 29 degrees: one refinement leaves 0.84 % on the most tilted, two 0.16 %).
#[test]
fn the_box_only_path_finds_the_same_landmarks() {
    let Some(lm) = load_mesh() else { return };
    let fx = fixtures();
    let stride = fx["landmark_stride"].as_u64().expect("stride") as usize;
    each_photo(|id, p, w, h, rgb| {
        for face in p["faces"].as_array().expect("faces") {
            let b = &face["detection"]["box"];
            let bx = [
                b[0].as_f64().unwrap(),
                b[1].as_f64().unwrap(),
                b[2].as_f64().unwrap() - b[0].as_f64().unwrap(),
                b[3].as_f64().unwrap() - b[1].as_f64().unwrap(),
            ];
            let img = Rgb8::new(w, h, rgb).expect("picture");
            let got = lm
                .landmarks_from_box(&img, bx, 2, &LandmarkerParams::default())
                .expect("landmarks")
                .expect("a face");
            let (mean, worst) = landmark_error(
                &got.points_px,
                &face["pass2"],
                stride,
                bx[2].max(bx[3]) as f32,
            );
            println!(
                "{id}: box only, two refinements: mean {mean:.5} worst {worst:.5} of the face; rotation {:.1} deg",
                got.roi.rotation_rad.to_degrees()
            );
            assert!(mean <= BOX_ONLY_MEAN_TOL, "{id} mean {mean}");
        }
    });
}
