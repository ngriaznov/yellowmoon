//! The teeth segmenter on public portraits (`docs/27` S-17): SAM 2.1 small prompted from the face mesh, on the photos of
//! `tools/fetch/teeth-photos.sh` (CC0 or public domain; never committed).
//!
//! Weights: `data/models/mediapipe/` (`tools/fetch/mediapipe.sh`) and `data/models/sam2/` (`tools/fetch/sam2.sh`); photos:
//! `testdata/teeth/photos/<id>.jpg` (or under `YM_TESTDATA`). Every test is `#[ignore]` (SAM's encoder takes about half a minute per
//! face on four cores); `cargo test -p ym-core --test teeth_real -- --ignored`. Without the files they print a plain "SKIPPED" line
//! and pass; `YM_REQUIRE_TEETH=1` makes a missing file a failure.
//!
//! What they hold the segmenter to:
//!
//! * a face with an open mouth and visible teeth gets a mask that lies in the mouth and covers a good share of its opening;
//! * a smile with the teeth hidden, and a face washed out to white, get **no** teeth (and so no whitening), never a guess;
//! * the answer does not depend on how the photo was processed (a warm cast, a crushed curve): the mask of the original and of
//!   the processed picture agree. This is what the colour gates of S-16 could not do.

#[path = "../examples/teeth_support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use support::{decode, shrink, to_linear_awg4};
use ym_core::pipeline::object::ObjectModel;
use ym_core::pipeline::teeth::SamTeeth;
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::face_parts::mouth_interior_mask;
use ym_retouch::mediapipe::MediaPipeFaces;
use ym_retouch::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, TeethSegmenter, face_extent_px,
    teeth_from_segment,
};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn testdata() -> PathBuf {
    std::env::var_os("YM_TESTDATA").map_or_else(|| repo().join("testdata"), PathBuf::from)
}

fn required() -> bool {
    std::env::var_os("YM_REQUIRE_TEETH").is_some_and(|v| v != "0" && !v.is_empty())
}

fn skip_or_fail(what: &str) {
    assert!(!required(), "YM_REQUIRE_TEETH is set but {what}");
    println!("SKIPPED: {what}");
}

/// A photo of the manifest and what is expected of it.
struct Photo {
    id: String,
    expect: String,
}

fn photos() -> Vec<Photo> {
    let tsv =
        std::fs::read_to_string(repo().join("tools/fetch/teeth-photos.tsv")).expect("manifest");
    tsv.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            let expect = cols[3]
                .rsplit("expect ")
                .next()
                .expect("expectation")
                .trim();
            Photo {
                id: cols[0].to_owned(),
                expect: expect.to_owned(),
            }
        })
        .collect()
}

struct Env {
    finder: FaceFinder,
    engine: Arc<MediaPipeFaces>,
    seg: SamTeeth,
}

/// The networks, or `None` (after a SKIPPED line) when a weights file is missing.
fn env() -> Option<&'static Env> {
    static ENV: OnceLock<Option<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let models = repo().join("data/models");
        let mp = models.join("mediapipe");
        let ok = [
            "blaze_face_full_range.f16.safetensors",
            "face_mesh_v2.f16.safetensors",
            "selfie_multiclass_256.f32.safetensors",
        ]
        .iter()
        .all(|f| mp.join(f).is_file());
        if !ok {
            skip_or_fail("no face weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)");
            return None;
        }
        let Some(sam) = ObjectModel::probe(&repo().join("data")) else {
            skip_or_fail("no SAM weights in data/models/sam2 (run tools/fetch/sam2.sh)");
            return None;
        };
        let engine = MediaPipeFaces::load(&mp).expect("mesh and segmenter weights");
        Some(Env {
            finder: FaceFinder::new(BlazeFace::load(&mp).expect("blazeface")),
            seg: SamTeeth::new(Arc::clone(&engine), sam),
            engine,
        })
    })
    .as_ref()
}

/// A photo at the analysis size of the product (1400 px on the long edge at most) and its widest face.
fn load(env: &Env, id: &str) -> Option<(LinearAwg4Image, Face)> {
    let path = testdata().join("teeth/photos").join(format!("{id}.jpg"));
    if !path.is_file() {
        skip_or_fail(&format!(
            "no photo {} (run tools/fetch/teeth-photos.sh)",
            path.display()
        ));
        return None;
    }
    // The picture the analysis looks at: the long edge 1400 px at most (`FACE_ANALYSIS_LONG_EDGE_PX`).
    let (w, h, rgb) = decode(&path);
    let (w, h, rgb) = shrink(w, h, &rgb, 1400);
    let mut faces = env
        .finder
        .find(
            &Rgb8::new(w, h, &rgb).expect("picture"),
            &FinderParams::default(),
        )
        .expect("find");
    faces.sort_by_key(|d| std::cmp::Reverse(d.rect.width_px));
    let d = faces.first().unwrap_or_else(|| panic!("{id}: no face"));
    Some((
        to_linear_awg4(w, h, &rgb),
        Face {
            rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
            scale: d.scale,
        },
    ))
}

/// The final teeth mask of 5.6 (the segmenter's answer cut to the mouth) and the opening's size in pixels.
fn final_mask(
    env: &Env,
    image: &LinearAwg4Image,
    face: &Face,
    answer: &BoundedMask,
) -> (Mask, usize) {
    let size = (image.width_px, image.height_px);
    let a = env
        .engine
        .analyse(image, face)
        .expect("analyse")
        .expect("a mesh");
    let mouth = mouth_interior_mask(&a.geometry, size)
        .expect("mouth")
        .expect("an open mouth");
    let mouth_full = mouth.to_full(size).expect("mouth");
    let opening = mouth_full.data.iter().filter(|&&v| v >= 128).count();
    let extent = face_extent_px(size, face);
    (
        teeth_from_segment(&mouth_full, &answer.to_full(size).expect("answer"), extent)
            .expect("teeth"),
        opening,
    )
}

fn on(mask: &Mask) -> impl Iterator<Item = usize> + '_ {
    mask.data
        .iter()
        .enumerate()
        .filter(|(_, v)| **v >= 128)
        .map(|(i, _)| i)
}

fn check_teeth(env: &Env, id: &str) {
    let Some((image, face)) = load(env, id) else {
        return;
    };
    let t = std::time::Instant::now();
    let answer = env
        .seg
        .teeth(&image, &face)
        .expect("the segmenter")
        .unwrap_or_else(|| panic!("{id}: SAM found no teeth"));
    let first = t.elapsed();
    let (mask, opening) = final_mask(env, &image, &face, &answer);
    let area = on(&mask).count();
    let cover = area as f64 / opening as f64;
    println!(
        "{id}: teeth {area} px of an opening of {opening} px ({:.0} %), {:.1} s",
        100.0 * cover,
        first.as_secs_f64()
    );
    assert!(
        (0.25..=1.0).contains(&cover),
        "{id}: the teeth cover {cover:.2} of the opening"
    );
    // The mask is in the mouth: its pixels are inside the opening grown by the feather.
    let a = env
        .engine
        .analyse(&image, &face)
        .expect("analyse")
        .expect("a mesh");
    let size = (image.width_px, image.height_px);
    let opening_mask = mouth_interior_mask(&a.geometry, size)
        .expect("mouth")
        .expect("open")
        .to_full(size)
        .expect("mouth");
    let w = size.0 as i64;
    let margin = 0.01 * face_extent_px(size, &face).max(4.0);
    for i in on(&mask) {
        let (x, y) = (i as i64 % w, i as i64 / w);
        let r = margin.ceil() as i64;
        let near = (-r..=r).any(|dy| (-r..=r).any(|dx| opening_mask.get(x + dx, y + dy) > 0));
        assert!(near, "{id}: teeth at ({x}, {y}) are outside the mouth");
    }
    // The second call is the cached answer: the same bytes, at once.
    let t = std::time::Instant::now();
    let again = env.seg.teeth(&image, &face).expect("the segmenter");
    assert_eq!(
        again.as_ref(),
        Some(&answer),
        "{id}: the cache answers the same"
    );
    assert!(
        t.elapsed().as_millis() < 500,
        "{id}: the second call runs SAM again"
    );
}

fn run_slice(from: usize, to: usize) {
    let Some(env) = env() else {
        return;
    };
    let teeth: Vec<Photo> = photos()
        .into_iter()
        .filter(|p| p.expect == "teeth")
        .collect();
    for p in teeth.iter().skip(from).take(to - from) {
        check_teeth(env, &p.id);
    }
}

#[test]
#[ignore = "SAM's encoder: about half a minute per face"]
fn teeth_are_found_in_public_portraits_a() {
    run_slice(0, 6);
}

#[test]
#[ignore = "SAM's encoder: about half a minute per face"]
fn teeth_are_found_in_public_portraits_b() {
    run_slice(6, 12);
}

#[test]
#[ignore = "SAM's encoder: about half a minute per face"]
fn teeth_are_found_in_public_portraits_c() {
    run_slice(12, 18);
}

#[test]
#[ignore = "SAM's encoder: about half a minute per face"]
fn a_hidden_smile_and_a_blown_out_face_get_no_teeth() {
    let Some(env) = env() else {
        return;
    };
    for p in photos().iter().filter(|p| p.expect != "teeth") {
        let Some((image, face)) = load(env, &p.id) else {
            continue;
        };
        let answer = env.seg.teeth(&image, &face).expect("the segmenter");
        println!(
            "{} ({}): {}",
            p.id,
            p.expect,
            answer.as_ref().map_or("no teeth", |_| "TEETH")
        );
        assert!(
            answer.is_none(),
            "{} ({}): no whitening, not a guess",
            p.id,
            p.expect
        );
    }
}

/// A grade: a warm cast, a crushed curve and a gain, on the linear picture.
fn processed(image: &LinearAwg4Image) -> LinearAwg4Image {
    let mut out = image.clone();
    for p in &mut out.data {
        let tinted = [p[0] * 1.3, p[1], p[2] * 0.7];
        *p = tinted.map(|c| 1.4 * c.max(0.0).powf(1.35));
    }
    out
}

#[test]
#[ignore = "SAM's encoder: about half a minute per face"]
fn the_teeth_do_not_depend_on_how_the_photo_was_processed() {
    let Some(env) = env() else {
        return;
    };
    // Yellow teeth, white teeth, shaded teeth.
    for id in ["pub-6ad5a2ea", "pub-935c7333", "pub-a736a39d"] {
        let Some((image, face)) = load(env, id) else {
            return;
        };
        let size = (image.width_px, image.height_px);
        let plain = env
            .seg
            .teeth(&image, &face)
            .expect("the segmenter")
            .unwrap_or_else(|| panic!("{id}: no teeth"));
        let graded = processed(&image);
        let other = env
            .seg
            .teeth(&graded, &face)
            .expect("the segmenter")
            .unwrap_or_else(|| panic!("{id}: no teeth after the grade"));
        let (a, b) = (
            plain.to_full(size).expect("a"),
            other.to_full(size).expect("b"),
        );
        let (mut inter, mut union) = (0usize, 0usize);
        for (x, y) in a.data.iter().zip(&b.data) {
            let (x, y) = (*x >= 128, *y >= 128);
            inter += usize::from(x && y);
            union += usize::from(x || y);
        }
        let iou = inter as f64 / union.max(1) as f64;
        println!("{id}: the mask of the graded picture agrees with the plain one at IoU {iou:.3}");
        assert!(
            iou >= 0.7,
            "{id}: IoU {iou:.3} between the plain and the graded picture"
        );
    }
}
