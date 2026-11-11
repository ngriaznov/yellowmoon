//! The real face models through the core (`docs/27-skin-retouch.md` S-11b): the public test portraits of
//! `tools/fetch/mediapipe.sh --photos` are made into raw files, and the core finds their faces, numbers them, and retouches
//! them (and nothing else) at an export.
//!
//! Weights: `data/models/mediapipe/` (`tools/fetch/mediapipe.sh`); photos: `testdata/mediapipe/photos/<id>.jpg`. Without them the
//! tests print a plain "SKIPPED" line and pass; `YM_REQUIRE_MEDIAPIPE=1` makes a missing file a failure (CI sets it).
//!
//! A photo becomes a raw file by putting its linear sRGB values on the camera's white (`0.9 * linear * white / peak`, as
//! `Scene::signal` does): the colours are not the photo's (the camera's matrix is not sRGB), the faces are where they were.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::*;
use serde_json::json;
use ym_color::display::OutputTarget;
use ym_core::pipeline::{FaceModel, FullRequest, FullTarget, NeutralOnly, RenderOptions, Renderer};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken;
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

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

fn have_weights() -> bool {
    [
        "blaze_face_full_range.f16.safetensors",
        "face_mesh_v2.f16.safetensors",
        "selfie_multiclass_256.f32.safetensors",
    ]
    .iter()
    .all(|f| data_root().join("models/mediapipe").join(f).is_file())
}

struct Photo {
    id: String,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
    /// The faces of the photo (`MediaPipe`'s test data): how many the finder finds in it.
    faces: usize,
}

/// The photos of the manifest, with the number of faces each holds (from the manifest's text).
fn photos() -> Vec<Photo> {
    let tsv =
        std::fs::read_to_string(repo().join("tools/fetch/mediapipe-photos.tsv")).expect("manifest");
    let mut out = Vec::new();
    for line in tsv
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let id = line.split('\t').next().expect("id").to_owned();
        let faces = if id == "mp-two-people" { 2 } else { 1 };
        let path = repo()
            .join("testdata/mediapipe/photos")
            .join(format!("{id}.jpg"));
        if !path.is_file() {
            skip_or_fail(&format!(
                "no photo {} (run tools/fetch/mediapipe.sh --photos)",
                path.display()
            ));
            continue;
        }
        let mut d =
            zune_jpeg::JpegDecoder::new(std::io::Cursor::new(std::fs::read(&path).expect("photo")));
        let rgb = d.decode().expect("decode");
        let info = d.info().expect("header");
        out.push(Photo {
            id,
            w: usize::from(info.width),
            h: usize::from(info.height),
            rgb,
            faces,
        });
    }
    out
}

/// A raw file of the photo, opened.
fn developed(p: &Photo) -> Developed {
    let white = light(PROFILE, 5500.0);
    let (w, h) = (p.w as u32 & !1, p.h as u32 & !1);
    let scene = Scene::new(w, h, white, Layout::Uniform, 0.5).noise(0.0, 0.0);
    let peak = white.iter().copied().fold(0.0, f64::max);
    let decode = |v: u8| {
        let e = f64::from(v) / 255.0;
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    };
    let range = scene.white_level - scene.black;
    let mut mosaic = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let c = [0usize, 1, 1, 2][(y & 1) * 2 + (x & 1)];
            let lin = decode(p.rgb[(y * p.w + x) * 3 + c]);
            let v = (0.9 * lin * white[c] / peak).clamp(0.0, 1.0);
            mosaic.push((scene.black + v * range).round() as u16);
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn renderer(model: Arc<FaceModel>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            face_model: Some(model),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn node(body: serde_json::Value) -> DevelopSettings {
    let mut n = json!({ "id": 2, "kind": "face_refinement", "pos": [1.0, 0.0] });
    n.as_object_mut()
        .unwrap()
        .extend(body.as_object().unwrap().clone());
    let link = |a: serde_json::Value, b: serde_json::Value| json!({ "from": { "node": a, "port": "picture" }, "to": { "node": b, "port": "picture" } });
    serde_json::from_value::<DevelopSettings>(json!({ "nodes": {
        "items": [ { "id": 1 }, n ],
        "links": [ link(json!("input"), json!(1)), link(json!(1), json!(2)), link(json!(2), json!("output")) ] } }))
    .expect("a document")
    .validated()
    .expect("a valid graph")
    .0
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x1, y1) = (
        (a[0] + a[2]).min(b[0] + b[2]),
        (a[1] + a[3]).min(b[1] + b[3]),
    );
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    inter / (a[2] * a[3] + b[2] * b[3] - inter)
}

#[test]
fn the_core_finds_the_faces_of_the_public_portraits_and_retouches_them() {
    if !have_weights() {
        skip_or_fail("no face weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)");
        return;
    }
    let all = photos();
    if all.is_empty() {
        return;
    }
    for p in &all {
        let d = developed(p);
        let src = d.source();
        let with_node = src.with_settings(node(json!({})));
        // A floor of 24 px finds every face of the photo; the finder on the JPEG itself is the reference for where.
        let r = renderer(FaceModel::probe_with(&data_root(), 24).expect("the model"));
        let found = r
            .faces_found(&with_node, &CancelToken::new())
            .expect("the faces");
        let finder = FaceFinder::new(
            BlazeFace::load(&data_root().join("models/mediapipe")).expect("blazeface"),
        );
        let reference = finder
            .find(
                &Rgb8::new(p.w, p.h, &p.rgb).expect("picture"),
                &FinderParams::default(),
            )
            .expect("find");
        println!(
            "{}: {} faces through the core, {} on the JPEG",
            p.id,
            found.len(),
            reference.len()
        );
        assert_eq!(found.len(), p.faces, "{}: {found:?}", p.id);
        let numbers: Vec<u32> = found.iter().map(|f| f.number).collect();
        assert_eq!(numbers, (1..=p.faces as u32).collect::<Vec<_>>());
        assert!(
            found
                .windows(2)
                .all(|w| w[0].x + w[0].width / 2.0 <= w[1].x + w[1].width / 2.0)
        );
        for f in &found {
            let b = [f.x, f.y, f.width, f.height];
            let best = reference
                .iter()
                .map(|d| {
                    iou(
                        b,
                        [
                            f64::from(d.rect.x_px) / p.w as f64,
                            f64::from(d.rect.y_px) / p.h as f64,
                            f64::from(d.rect.width_px) / p.w as f64,
                            f64::from(d.rect.height_px) / p.h as f64,
                        ],
                    )
                })
                .fold(0.0, f64::max);
            assert!(best > 0.6, "{}: face {} IoU {best}", p.id, f.number);
        }
        // The default floor (200 px across in the photo) keeps only the faces that big.
        let strict = renderer(FaceModel::probe(&data_root()).expect("the model"));
        let kept = strict
            .faces_found(&with_node, &CancelToken::new())
            .expect("the faces");
        let big = found
            .iter()
            .filter(|f| (f.width * p.w as f64).max(f.height * p.h as f64) >= 200.0)
            .count();
        assert_eq!(kept.len(), big, "{}: the floor of 200 px", p.id);
        // An export retouches the faces and nothing else, bit for bit.
        let plain = r
            .render_full(
                &src.with_settings(DevelopSettings::default()),
                &FullRequest {
                    target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                },
                &CancelToken::new(),
            )
            .expect("render");
        let retouched = r
            .render_full(
                &with_node,
                &FullRequest {
                    target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                },
                &CancelToken::new(),
            )
            .expect("render");
        let (mut inside, mut outside) = (0usize, 0usize);
        for y in 0..plain.height {
            for x in 0..plain.width {
                let i = (y * plain.width + x) * 3;
                if plain.rgb[i..i + 3] == retouched.rgb[i..i + 3] {
                    continue;
                }
                let near = found.iter().any(|f| {
                    let (w, h) = (plain.width as f64, plain.height as f64);
                    let long = (f.width * w).max(f.height * h);
                    let (px, py) = (x as f64, y as f64);
                    px >= f.x * w - 0.6 * long
                        && px < (f.x + f.width) * w + 0.6 * long
                        && py >= f.y * h - 0.6 * long
                        && py < (f.y + f.height) * h + 0.6 * long
                });
                if near {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
        }
        println!(
            "{}: {inside} values changed near the faces, {outside} elsewhere",
            p.id
        );
        assert_eq!(outside, 0, "{}: nothing changes away from the faces", p.id);
        assert!(inside > 100, "{}: the faces were retouched", p.id);
    }
}
