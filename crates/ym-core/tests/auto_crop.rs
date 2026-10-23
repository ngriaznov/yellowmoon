//! Auto crop in the session (`docs/20-auto-crop.md` §3, §13): the proposal for every aspect and
//! orientation is a set of frames inside the canvas with the aspect's ratio, made from the crop
//! model's crop of the picture, two sessions give the same answer, a missing model is the reason
//! given, and the setting's `auto` flag reads and writes as the spec says.
//!
//! Synthetic scenes only.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::*;
use ym_core::auto_crop::{CropSuggestion, SubjectSource};
use ym_core::pipeline::{AdvisorPictures, RenderOptions};
use ym_core::session::{OpenOptions, PhotoId, Session, SessionError};
use ym_core::settings::{Crop, CropAspect, NormRect, Orientation};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A session whose Auto crop has its picture renderer, as the app's has.
fn session() -> Session {
    let env = base_env();
    let pictures =
        AdvisorPictures::new(env.reference_look.clone(), &RenderOptions::default()).unwrap();
    Session::new(env.with_pictures(Arc::new(pictures)))
}

/// A 3:2 patchwork with the camera's focus point a third in from the left, as Sony writes it.
fn write_photo(dir: &Path) -> PathBuf {
    let scene = Scene::new(600, 400, light(PROFILE, 5200.0), Layout::Patchwork, 0.3)
        .noise(2e-4, 1e-6)
        .seed(11);
    let path = dir.join("focus.ARW");
    scene
        .builder(MODEL)
        .focus_location([600, 400, 200, 180])
        .focus_frame_size([60, 60, 257])
        .write_to(&path, &scene.mosaic())
        .unwrap();
    path
}

fn open_one(session: &Session, path: &Path) -> PhotoId {
    let id = session
        .open(&[path.to_path_buf()], OpenOptions::default())
        .added[0];
    settle(session);
    id
}

/// Width over height of a frame in pixels of the canvas of a `w x h` photo turned by `orientation`.
fn pixel_ratio(rect: &NormRect, w: f64, h: f64, orientation: Orientation) -> f64 {
    let (cw, ch) = match orientation {
        Orientation::Deg90 | Orientation::Deg270 => (h, w),
        _ => (w, h),
    };
    rect.width * cw / (rect.height * ch)
}

#[test]
fn every_aspect_and_turn_gives_frames_inside_the_canvas_with_its_ratio() {
    let dir = tempfile::tempdir().unwrap();
    let session = session();
    let id = open_one(&session, &write_photo(dir.path()));
    let summary = session.summary(id).unwrap();
    let (w, h) = (f64::from(summary.width_px), f64::from(summary.height_px));
    for orientation in [
        Orientation::Deg0,
        Orientation::Deg90,
        Orientation::Deg180,
        Orientation::Deg270,
    ] {
        let turned = matches!(orientation, Orientation::Deg90 | Orientation::Deg270);
        let canvas = if turned { h / w } else { w / h };
        for (aspect, want) in [
            (CropAspect::Original, Some(canvas)),
            // Free is the crop model's own shape.
            (CropAspect::Free, None),
            (CropAspect::Square, Some(1.0)),
            (
                CropAspect::FourByFive,
                Some(if canvas >= 1.0 { 5.0 / 4.0 } else { 4.0 / 5.0 }),
            ),
            (
                CropAspect::SixteenByNine,
                Some(if canvas >= 1.0 {
                    16.0 / 9.0
                } else {
                    9.0 / 16.0
                }),
            ),
        ] {
            let s = session.suggest_crop(id, aspect, orientation).unwrap();
            assert!(
                !s.frames.is_empty() && s.frames.len() <= 3,
                "{aspect:?} {orientation:?}: {s:?}"
            );
            // The photo carries the camera's focus point, but the crop model reads the picture.
            assert_eq!(s.subject, SubjectSource::Picture, "{s:?}");
            for f in &s.frames {
                let r = f.rect;
                assert!(
                    r.x >= -1e-9
                        && r.y >= -1e-9
                        && r.x + r.width <= 1.0 + 1e-9
                        && r.y + r.height <= 1.0 + 1e-9
                        && r.width > 0.0
                        && r.height > 0.0,
                    "{aspect:?} {orientation:?}: {r:?}"
                );
                // The canvas's own shape at the photo's pixel size is exact to the rounding of the
                // picture's size (768 px on its long edge, `docs/19` §3).
                if let Some(want) = want {
                    let ratio = pixel_ratio(&r, w, h, orientation);
                    assert!(
                        (ratio / want - 1.0).abs() < 0.01,
                        "{aspect:?} {orientation:?}: ratio {ratio} for {want}"
                    );
                }
                assert!((0.0..=1.0).contains(&f.score), "{s:?}");
            }
            // Closest to the crop model's crop first.
            assert!(
                s.frames.windows(2).all(|p| p[0].score >= p[1].score),
                "{s:?}"
            );
        }
    }
}

#[test]
fn the_answer_is_the_same_in_another_session() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_photo(dir.path());
    let answers: Vec<String> = (0..2)
        .map(|_| {
            let session = session();
            let id = open_one(&session, &path);
            let s: CropSuggestion = session
                .suggest_crop(id, CropAspect::FourByFive, Orientation::Deg90)
                .unwrap();
            serde_json::to_string(&s).unwrap()
        })
        .collect();
    assert_eq!(answers[0], answers[1]);
    // And again in the same session, from the kept analysis.
    let session = session();
    let id = open_one(&session, &path);
    let first = session
        .suggest_crop(id, CropAspect::FourByFive, Orientation::Deg90)
        .unwrap();
    let again = session
        .suggest_crop(id, CropAspect::FourByFive, Orientation::Deg90)
        .unwrap();
    assert_eq!(first, again);
    assert_eq!(serde_json::to_string(&first).unwrap(), answers[0]);
}

#[test]
fn a_missing_crop_model_is_the_reason_given() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let pictures =
        AdvisorPictures::new(env.reference_look.clone(), &RenderOptions::default()).unwrap();
    let mut env = env.with_pictures(Arc::new(pictures));
    env.crop_model = Err("the crop model could not be loaded (test)".to_owned());
    let session = Session::new(env);
    let id = open_one(&session, &write_photo(dir.path()));
    let err = session
        .suggest_crop(id, CropAspect::Original, Orientation::Deg0)
        .unwrap_err();
    assert!(matches!(err, SessionError::AutoCrop(_)), "{err:?}");
    assert!(
        err.to_string().contains("could not be loaded (test)"),
        "{err}"
    );
}

#[test]
fn unknown_photos_and_sessions_without_a_renderer_are_refused() {
    let session = session();
    assert!(matches!(
        session.suggest_crop(PhotoId(99), CropAspect::Original, Orientation::Deg0),
        Err(SessionError::UnknownPhoto(_))
    ));
    let dir = tempfile::tempdir().unwrap();
    let bare = Session::new(base_env());
    let id = open_one(&bare, &write_photo(dir.path()));
    let err = bare
        .suggest_crop(id, CropAspect::Original, Orientation::Deg0)
        .unwrap_err();
    assert!(matches!(err, SessionError::AutoCrop(_)), "{err:?}");
    assert!(
        err.to_string()
            .starts_with("Couldn't suggest a frame for this photo:"),
        "{err}"
    );
}

#[test]
fn crop_auto_is_absent_when_false_and_written_when_true() {
    let manual = Crop {
        rect: NormRect::FULL,
        angle_deg: 1.5,
        aspect: CropAspect::Original,
        auto: false,
    };
    let text = serde_json::to_string(&manual).unwrap();
    assert!(!text.contains("auto"), "{text}");
    let read: Crop = serde_json::from_str(&text).unwrap();
    assert!(!read.auto);
    let auto = Crop {
        auto: true,
        ..manual
    };
    let text = serde_json::to_string(&auto).unwrap();
    assert!(text.contains(r#""auto":true"#), "{text}");
    assert_eq!(serde_json::from_str::<Crop>(&text).unwrap(), auto);
}
