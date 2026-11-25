//! A photo of the filmstrip as a reference, a reference's scene, and what is never kept (`docs/34-reference-grade.md` sections 6.4,
//! 6.6 and 10, RG-05).
//!
//! * **The snapshot** (`Renderer::reference_snapshot`): the photo rendered through its own current settings on the CPU reference path
//!   of a renderer without a device, held as a [`ReferencePicture`]. The photo being graded is refused.
//! * **The scene** (`Renderer::reference_scene`): the same segmenter and rules as a photo's scene, on the reference's scene picture.
//!   These tests use a fake scene model; the real networks are `crates/ym-ml/tests/scene.rs`.
//! * **Never kept**: with a reference alive, an export of a photo in every format writes nothing of it (the file's bytes, its name, a
//!   marker in its metadata), the original is untouched and the log has no word of it.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use ym_color::mask_element::soft_masks_of_picture;
use ym_core::export::{ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    NO_SCENE_MODEL_SENTENCE, NeutralOnly, RenderError, RenderOptions, Renderer, SceneModel,
    SnapshotError,
};
use ym_core::reference::{
    ANALYSIS_LONG_EDGE_PX, ReferenceError, ReferencePicture, ReferenceSource, SCENE_LONG_EDGE_PX,
};
use ym_core::session::PhotoId;
use ym_core::settings::SettingsPatch;
use ym_export::{ExportOptions, FormatKind};
use ym_gpu::queue::CancelToken;
use ym_ml::scene::{GROUND, SKY};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
const WIDTH: u32 = 640;
const HEIGHT: u32 = 420;

fn renderer(model: Option<Arc<SceneModel>>, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            scene_model: model,
            data_root: Some(data_root()),
            threads,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn photo() -> Developed {
    let scene = Scene::new(
        WIDTH,
        HEIGHT,
        light(PROFILE, 4800.0),
        Layout::Patchwork,
        0.5,
    )
    .noise(2e-4, 2e-6)
    .seed(21);
    Developed::new(&scene, MODEL)
}

fn mean_of(p: &ReferencePicture) -> f64 {
    let a = p.analysis();
    a.linear
        .iter()
        .map(|px| (px[0] + px[1] + px[2]) / 3.0)
        .sum::<f64>()
        / a.linear.len() as f64
}

// ---------------------------------------------------------------------------------------------
// The snapshot
// ---------------------------------------------------------------------------------------------

#[test]
fn a_strip_photo_is_a_snapshot_through_its_own_settings() {
    let d = photo();
    let src = d.source();
    let r = renderer(None, 2);
    let cancel = CancelToken::new();

    // The photo being graded cannot be its own reference: the core says so, with the text of section 9.3.
    let same = r.reference_snapshot(src.id, &src, &cancel).unwrap_err();
    assert!(
        matches!(same, SnapshotError::Reference(ReferenceError::SamePhoto)),
        "{same:?}"
    );
    assert_eq!(
        same.to_string(),
        "That's the photo you're grading. Choose another."
    );

    // Another photo is taken: whole, never enlarged, at most the scene and analysis sizes.
    let graded = PhotoId(src.id.0 + 1000);
    let snap = r
        .reference_snapshot(graded, &src, &cancel)
        .expect("a snapshot");
    assert_eq!(snap.source(), ReferenceSource::Photo(src.id));
    assert!(snap.notes().is_empty() && !snap.is_grey());
    assert_eq!(
        (snap.scene().width, snap.scene().height),
        (WIDTH as usize, HEIGHT as usize)
    );
    assert_eq!(snap.analysis().width, ANALYSIS_LONG_EDGE_PX);
    assert!(
        snap.analysis().width <= ANALYSIS_LONG_EDGE_PX && snap.scene().width <= SCENE_LONG_EDGE_PX
    );

    // It is the picture Auto's analysis takes (the CPU reference path, the photo's own settings, sRGB 8 bits): at a size where
    // neither reduces, the same bytes.
    let advisor = r
        .advisor_picture(&src, &cancel)
        .expect("the advisor's picture");
    assert_eq!(
        (advisor.width, advisor.height),
        (snap.scene().width, snap.scene().height)
    );
    assert!(
        snap.scene().srgb8() == advisor.rgb,
        "the snapshot is the advisor's path"
    );

    // The photo's own settings are in it: the same photo at two manual exposures is two pictures, the brighter one brighter.
    d.session
        .update_settings(d.id, &SettingsPatch::exposure_ev(-1.0))
        .expect("settings");
    let dim_src = d.source();
    d.session
        .update_settings(d.id, &SettingsPatch::exposure_ev(2.0))
        .expect("settings");
    let bright_src = d.source();
    assert_ne!(bright_src.settings, dim_src.settings);
    let dim = r
        .reference_snapshot(graded, &dim_src, &cancel)
        .expect("dim");
    let brighter = r
        .reference_snapshot(graded, &bright_src, &cancel)
        .expect("brighter");
    assert!(
        mean_of(&brighter) > mean_of(&dim) * 2.0,
        "{} against {}",
        mean_of(&brighter),
        mean_of(&dim)
    );
    assert_ne!(brighter.digest(), dim.digest());
    // A snapshot is a value: the photo's later edit changed nothing in the one taken before.
    assert!(
        (mean_of(&snap) - mean_of(&r.reference_snapshot(graded, &src, &cancel).expect("again")))
            .abs()
            < 1e-15
    );

    // The same bits on a second call and on another number of threads.
    let again = r.reference_snapshot(graded, &src, &cancel).expect("again");
    assert!(again.analysis() == snap.analysis() && again.scene() == snap.scene());
    assert_eq!(again.digest(), snap.digest());
    let one = renderer(None, 1)
        .reference_snapshot(graded, &src, &cancel)
        .expect("one thread");
    let four = renderer(None, 4)
        .reference_snapshot(graded, &src, &cancel)
        .expect("four threads");
    assert!(one.analysis() == snap.analysis() && four.analysis() == snap.analysis());
}

#[test]
fn a_cancelled_snapshot_is_cancelled_and_leaves_nothing_behind() {
    let d = photo();
    let src = d.source();
    let r = renderer(None, 2);
    let cancel = CancelToken::new();
    cancel.cancel();
    let err = r
        .reference_snapshot(PhotoId(src.id.0 + 1), &src, &cancel)
        .unwrap_err();
    assert!(
        matches!(err, SnapshotError::Render(RenderError::Cancelled)),
        "{err:?}"
    );
    // The renderer works for the next one.
    assert!(
        r.reference_snapshot(PhotoId(src.id.0 + 1), &src, &CancelToken::new())
            .is_ok()
    );
}

// ---------------------------------------------------------------------------------------------
// The scene of a reference
// ---------------------------------------------------------------------------------------------

/// A fake scene that counts its runs and keeps the last picture it was given: sky above a third, ground below.
fn recording_scene(
    runs: &Arc<AtomicUsize>,
    seen: &Arc<std::sync::Mutex<Vec<(usize, usize, Vec<u8>)>>>,
) -> Arc<SceneModel> {
    let (runs, seen) = (Arc::clone(runs), Arc::clone(seen));
    SceneModel::fake(
        "reference-fake",
        true,
        Arc::new(move |w, h, rgb, _| {
            runs.fetch_add(1, Ordering::SeqCst);
            seen.lock().expect("lock").push((w, h, rgb.to_vec()));
            let mut bits = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    bits[y * w + x] = if (y as f64 + 0.5) / (h as f64) < 0.33 {
                        SKY
                    } else {
                        GROUND
                    };
                }
            }
            Ok(bits)
        }),
    )
}

fn colourful(w: usize, h: usize, seed: u8) -> Vec<u8> {
    (0..w * h * 3)
        .map(|i| ((i * 7 + usize::from(seed) * 31 + (i / (3 * w)) * 3) % 251) as u8)
        .collect()
}

fn png_of(w: usize, h: usize, rgb: &[u8], text: Option<(&str, &str)>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    if let Some((k, v)) = text {
        enc.add_text_chunk(k.to_owned(), v.to_owned())
            .expect("text");
    }
    let mut writer = enc.write_header().expect("header");
    writer.write_image_data(rgb).expect("data");
    writer.finish().expect("finish");
    out
}

#[test]
fn a_references_scene_is_made_from_its_scene_picture_by_the_photos_machinery() {
    let (w, h) = (1500usize, 1000usize);
    let reference =
        ReferencePicture::from_bytes(&png_of(w, h, &colourful(w, h, 1), None)).expect("reference");
    assert_eq!(
        (reference.scene().width, reference.scene().height),
        (SCENE_LONG_EDGE_PX, 683)
    );
    let runs = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let r = renderer(Some(recording_scene(&runs, &seen)), 2);
    let cancel = CancelToken::new();
    let made = r.reference_scene(&reference, &cancel).expect("a scene");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert!(made.skin_available);
    let (sw, sh, rgb) = seen.lock().expect("lock")[0].clone();
    assert_eq!(
        (sw, sh),
        (reference.scene().width, reference.scene().height),
        "the segmenter read the scene picture"
    );
    assert!(rgb == reference.scene().srgb8(), "as sRGB display codes");
    assert_eq!((made.scene.width, made.scene.height), (sw, sh));
    assert!(
        made.scene.share(0) > 0.30 && made.scene.share(0) < 0.36,
        "sky share {}",
        made.scene.share(0)
    );

    // The key names the picture: equal references and models, equal keys; another picture, another key.
    let again = r.reference_scene(&reference, &cancel).expect("again");
    assert_eq!(again.scene.key, made.scene.key);
    assert!(again.scene.bits == made.scene.bits);
    let other =
        ReferencePicture::from_bytes(&png_of(w, h, &colourful(w, h, 2), None)).expect("other");
    assert_ne!(
        r.reference_scene(&other, &cancel).expect("other").scene.key,
        made.scene.key
    );

    // The fit's soft masks on the analysis picture, from the guide of its own lightness (RG-06 reads them like this).
    let a = reference.analysis();
    let guide: Vec<f32> = a
        .linear
        .iter()
        .map(|p| (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f32)
        .collect();
    let wanted = made.scene.covering(0.02);
    assert!(wanted[0] && wanted[3], "sky and ground are wanted");
    let planes = soft_masks_of_picture(&made.scene, &wanted, &guide, a.width, a.height);
    let sky = planes[0].as_ref().expect("a sky plane");
    assert_eq!(sky.len(), a.width * a.height);
    assert!(sky[a.width * 5 + 3] > 0.9 && sky[a.width * (a.height - 5) + 3] < 0.1);

    // A cancelled call is cancelled; no model is the plain sentence.
    let stopped = CancelToken::new();
    stopped.cancel();
    assert!(matches!(
        r.reference_scene(&reference, &stopped),
        Err(RenderError::Cancelled)
    ));
    let none = renderer(None, 1)
        .reference_scene(&reference, &cancel)
        .unwrap_err();
    assert!(
        matches!(&none, RenderError::Mask(s) if s == NO_SCENE_MODEL_SENTENCE),
        "{none:?}"
    );
}

#[test]
fn a_snapshot_has_a_scene_too() {
    let d = photo();
    let src = d.source();
    let runs = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let r = renderer(Some(recording_scene(&runs, &seen)), 2);
    let snap = r
        .reference_snapshot(PhotoId(src.id.0 + 1), &src, &CancelToken::new())
        .expect("snapshot");
    let made = r
        .reference_scene(&snap, &CancelToken::new())
        .expect("scene");
    assert_eq!(
        (made.scene.width, made.scene.height),
        (snap.scene().width, snap.scene().height)
    );
}

// ---------------------------------------------------------------------------------------------
// Never kept
// ---------------------------------------------------------------------------------------------

fn files_under(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let mut v = Vec::new();
    walk(root, &mut v);
    v.sort();
    v
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// With a reference held, a photo's export in every format and the log carry nothing of it: not its file's name, not a marker in its
/// metadata, not its pixels; the original file of the photo and the reference file are not touched; nothing is written to the
/// reference's folder.
#[test]
fn nothing_of_a_reference_reaches_a_written_file_or_the_log() {
    const MARKER: &str = "YM-REFERENCE-MARKER-4f1c9a7e2d-do-not-keep";
    const NAME: &str = "private-film-still-name.png";
    captured_log();
    let refs = tempfile::tempdir().expect("references folder");
    let (rw, rh) = (700usize, 400usize);
    let pixels = colourful(rw, rh, 9);
    let reference_bytes = png_of(rw, rh, &pixels, Some(("Comment", MARKER)));
    assert!(contains(&reference_bytes, MARKER.as_bytes()));
    let reference_path = refs.path().join(NAME);
    std::fs::write(&reference_path, &reference_bytes).expect("write the reference");
    let before_refs = files_under(refs.path());
    let modified = std::fs::metadata(&reference_path)
        .expect("meta")
        .modified()
        .expect("mtime");

    let src_dir = tempfile::tempdir().expect("source folder");
    let scene = Scene::new(
        WIDTH,
        HEIGHT,
        light(PROFILE, 5500.0),
        Layout::Patchwork,
        0.4,
    )
    .noise(2e-4, 1e-5);
    scene
        .builder(MODEL)
        .capture_time("2026:10:04 12:03:44", "07", Some("+02:00"))
        .write_to(&src_dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write the ARW");
    let original = std::fs::read(src_dir.path().join("DSC00001.ARW")).expect("original");
    let session = ym_core::session::Session::new(base_env());
    let opened = session.open(
        &[src_dir.path().to_path_buf()],
        ym_core::session::OpenOptions::default(),
    );
    settle(&session);
    let id = opened.added[0];

    // The reference is read, and held while the photo is exported.
    let reference = ReferencePicture::from_file(&reference_path).expect("reference");
    assert_eq!(reference.source(), ReferenceSource::File);
    let home = tempfile::tempdir().expect("home");
    let out = tempfile::tempdir().expect("out");
    let ctx = export_context(home.path());
    let photo = ExportPhoto::from_session(&session, id).expect("export photo");
    for format in [
        FormatKind::Jpeg,
        FormatKind::Png,
        FormatKind::Tiff,
        FormatKind::Dng,
    ] {
        let options = ExportOptions {
            format,
            destination: Some(out.path().to_path_buf()),
            ..ExportOptions::default()
        };
        export_photo(
            &ctx,
            &photo,
            &ExportParams::new(options),
            &ExportControl::none(),
        )
        .unwrap_or_else(|e| panic!("{format:?}: {e}"));
    }
    assert!(
        !reference.analysis().linear.is_empty(),
        "the reference was alive throughout"
    );

    let written: Vec<PathBuf> = [out.path(), home.path(), src_dir.path()]
        .into_iter()
        .flat_map(files_under)
        .collect();
    assert!(
        written.len() >= 5,
        "the exports and the original: {written:?}"
    );
    let first_pixels: Vec<u8> = pixels[..96].to_vec();
    for f in &written {
        let bytes = std::fs::read(f).expect("read");
        assert!(
            !contains(&bytes, MARKER.as_bytes()),
            "{} holds the marker",
            f.display()
        );
        assert!(
            !contains(&bytes, NAME.as_bytes()),
            "{} holds the name",
            f.display()
        );
        assert!(
            !contains(&bytes, &first_pixels),
            "{} holds reference pixels",
            f.display()
        );
        assert!(
            !contains(&bytes, &reference_bytes[..64]),
            "{} holds the reference's header",
            f.display()
        );
    }
    // The original and the reference file are as they were, and nothing was written beside the reference.
    assert_eq!(
        std::fs::read(src_dir.path().join("DSC00001.ARW")).expect("original"),
        original
    );
    assert_eq!(
        std::fs::read(&reference_path).expect("reference"),
        reference_bytes
    );
    assert_eq!(
        std::fs::metadata(&reference_path)
            .expect("meta")
            .modified()
            .expect("mtime"),
        modified
    );
    assert_eq!(files_under(refs.path()), before_refs);
    // The settings of the photo have no field for it, and the log does not know its name or marker.
    let settings =
        serde_json::to_string(&session.view(id).expect("view").settings).expect("settings json");
    assert!(
        !settings.contains(MARKER)
            && !settings.contains(NAME)
            && !settings.contains("reference_path")
    );
    let log = captured_log();
    assert!(
        !log.contains(MARKER) && !log.contains(NAME),
        "the log names the reference"
    );
    // And a debug print of the reference is no way to carry its pixels out.
    let text = format!("{reference:?}");
    assert!(!text.contains(MARKER) && text.len() < 600);
}
