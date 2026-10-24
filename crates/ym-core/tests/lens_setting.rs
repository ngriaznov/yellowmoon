//! The lens setting (`docs/21` sections 2 and 5) in the core: the lens library loaded once with the
//! data folder, `Core::lens_list`, a missing or damaged `data/lenses` that is no crash, and the
//! identity every later lens stage must keep: all three amounts at 0 give the bytes of a render
//! without the field.
mod common;

use std::path::Path;
use std::sync::Arc;

use common::*;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions, Renderer,
    ViewportRect, ViewportState,
};
use ym_core::session::OpenOptions;
use ym_core::settings::{DevelopSettings, LensKind, LensSettings};
use ym_gpu::queue::CancelToken;
use ym_lens::LensLibrary;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn patchwork() -> Scene {
    Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4)
}

fn state(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn with_lens(src: &PhotoSource, lens: LensSettings) -> PhotoSource {
    let mut s: DevelopSettings = src.settings.clone();
    s.lens = lens;
    src.with_settings(s)
}

/// A synthetic lens folder (`ym_lens::test_support`) loaded as a library.
fn synthetic_library(dir: &Path) -> Arc<LensLibrary> {
    ym_lens::test_support::write_synthetic_lenses(dir).expect("write the lenses");
    Arc::new(LensLibrary::from_dir(dir).expect("the synthetic lenses load"))
}

fn renderer_with(library: Option<Arc<LensLibrary>>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            lens: library,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// The pictures one photo gives: the fit view, the full-resolution render and the thumbnail.
struct Pictures {
    frame: Vec<u8>,
    frame_notes: Vec<String>,
    full: Vec<f32>,
    full_notes: Vec<String>,
    thumbnail: Vec<u8>,
}

fn pictures(r: &Renderer, src: &PhotoSource) -> Pictures {
    let frame = r
        .viewport_frame(
            src,
            &state(300.0, 200.0),
            &FrameOptions {
                no_cache: true,
                ..FrameOptions::default()
            },
            &CancelToken::new(),
        )
        .expect("frame");
    let full = r
        .render_full(
            src,
            &FullRequest {
                target: FullTarget::DisplayLinear(ym_color::display::OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("full render");
    let thumbnail = r
        .thumbnail_png(src, &CancelToken::new())
        .expect("thumbnail");
    Pictures {
        frame: frame.rgba,
        frame_notes: frame.report.notes,
        full: full.rgb,
        full_notes: full.report.notes,
        thumbnail,
    }
}

#[test]
fn all_three_amounts_at_zero_are_the_render_without_the_field() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    assert_eq!(src.settings.lens, LensSettings::default());
    let dir = tempfile::tempdir().expect("dir");
    let library = synthetic_library(dir.path());
    // With the library and without it: the same bytes, whatever the kind says.
    for (what, renderer) in [
        ("with the lenses", renderer_with(Some(Arc::clone(&library)))),
        ("without the lenses", renderer_with(None)),
    ] {
        let none = pictures(&renderer, &src);
        assert!(none.frame_notes.is_empty(), "{what}: {:?}", none.frame_notes);
        assert!(none.full_notes.is_empty(), "{what}: {:?}", none.full_notes);
        for kind in LensKind::ALL {
            let zero = with_lens(
                &src,
                LensSettings {
                    kind,
                    ..LensSettings::default()
                },
            );
            assert!(zero.settings.lens.is_identity());
            let got = pictures(&renderer, &zero);
            assert!(got.frame == none.frame, "{what}, {kind}: the viewport frame");
            assert!(got.full == none.full, "{what}, {kind}: the full render");
            assert!(got.thumbnail == none.thumbnail, "{what}, {kind}: the thumbnail");
            assert!(
                got.frame_notes.is_empty() && got.full_notes.is_empty(),
                "{what}, {kind}: no note"
            );
        }
    }
}

#[test]
fn the_before_view_has_no_lens() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = with_lens(
        &d.source(),
        LensSettings {
            kind: LensKind::Petzval,
            flares: 70.0,
            edges: 60.0,
            anamorphic: 50.0,
        },
    );
    let before = src.for_before_view();
    assert_eq!(before.settings.lens, LensSettings::default());
    // And the Before view of a photo with a lens is the Before view of the same photo without one.
    let plain = d.source().for_before_view();
    let renderer = renderer_with(None);
    let a = pictures(&renderer, &before);
    let b = pictures(&renderer, &plain);
    assert!(a.frame == b.frame && a.full == b.full && a.thumbnail == b.thumbnail);
}

#[test]
fn a_lens_setting_without_lens_data_renders_as_no_lens_and_says_so() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    let renderer = renderer_with(None);
    let none = pictures(&renderer, &src);
    let lensed = with_lens(
        &src,
        LensSettings {
            kind: LensKind::Vintage,
            flares: 60.0,
            edges: 50.0,
            anamorphic: 40.0,
        },
    );
    let got = pictures(&renderer, &lensed);
    assert!(got.frame == none.frame, "the viewport frame");
    assert!(got.full == none.full, "the full render");
    assert!(got.thumbnail == none.thumbnail, "the thumbnail");
    for notes in [&got.frame_notes, &got.full_notes] {
        assert_eq!(
            notes,
            &["The lens data could not be loaded, so this photo has no lens effect."],
            "the render says why there is no lens"
        );
    }
}

/// Copies a folder and everything in it.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create folder");
    for entry in std::fs::read_dir(from).expect("read folder") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy file");
        }
    }
}

/// A copy of the shipped data folder with its `lenses` replaced by the folder `lenses` (or left out).
fn data_root_with(lenses: Option<&Path>) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("data root");
    for entry in std::fs::read_dir(data_root()).expect("the data folder") {
        let entry = entry.expect("entry");
        if entry.file_name() == "lenses" {
            continue;
        }
        let target = root.path().join(entry.file_name());
        if entry.file_type().expect("type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy file");
        }
    }
    if let Some(lenses) = lenses {
        copy_tree(lenses, &root.path().join("lenses"));
    }
    root
}

fn open_core(data: &Path) -> (Core, tempfile::TempDir) {
    let app = tempfile::tempdir().expect("app root");
    let mut options = CoreOptions::new(data.to_path_buf(), app.path().to_path_buf());
    options.decode_threads = 2;
    options.thumbnails = false;
    (Core::open(options).expect("the core opens"), app)
}

#[test]
fn the_lens_list_is_the_seven_lenses_in_menu_order() {
    let lenses = tempfile::tempdir().expect("lenses");
    let library = synthetic_library(lenses.path());
    let data = data_root_with(Some(lenses.path()));
    let (core, _app) = open_core(data.path());
    let list = core.lens_list();
    assert_eq!(list, library.lenses());
    assert_eq!(
        list.iter().map(|l| l.kind).collect::<Vec<_>>(),
        LensKind::ALL.to_vec()
    );
    assert!(
        list.iter().all(|l| !l.name.is_empty() && !l.line.is_empty()),
        "{list:?}"
    );
    // The renderer holds the same library, for the stages to come.
    assert!(core.renderer.options().lens.is_some());
    assert_eq!(
        core.lens_library().expect("the library").lenses(),
        library.lenses()
    );
    // The session and the library of the session are the one that was loaded.
    assert!(core.session.lens_library().is_ok());
}

#[test]
fn a_missing_or_damaged_lens_folder_is_no_crash() {
    // Missing: the core opens, the list is empty and says why.
    let data = data_root_with(None);
    let (core, _app) = open_core(data.path());
    assert!(core.lens_list().is_empty());
    let why = core.lens_library().err().expect("a reason");
    assert!(why.contains("lens data"), "{why}");
    assert!(core.renderer.options().lens.is_none());

    // Damaged: a file changed behind the checksums.
    let lenses = tempfile::tempdir().expect("lenses");
    synthetic_library(lenses.path());
    let victim = lenses.path().join("vintage").join("lens.json");
    let mut text = std::fs::read_to_string(&victim).expect("read");
    text.push_str("\n ");
    std::fs::write(&victim, text).expect("damage");
    let data = data_root_with(Some(lenses.path()));
    let (core, _app) = open_core(data.path());
    assert!(core.lens_list().is_empty());
    assert!(core.lens_library().is_err());
    assert!(core.renderer.options().lens.is_none());

    // A photo with a lens setting still opens and develops, as no lens.
    let dir = tempfile::tempdir().expect("photos");
    patchwork().write(dir.path(), "DSC00001.ARW", MODEL);
    let out = core
        .session
        .open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert!(core
        .session
        .wait_idle(std::time::Duration::from_secs(120)));
    assert_eq!(out.added.len(), 1);
}
