//! Photos at the edges: tiny and odd sizes, black and white frames, every layout at once, hostile
//! settings. Nothing may panic and every frame must be finite and the size it was asked for.
mod common;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, PhotoSource, Renderer, ViewportMode, ViewportRect,
    ViewportState,
};
use ym_core::settings::{Crop, CropAspect, NormRect, Orientation};
use ym_gpu::queue::CancelToken;
use ym_raw::testgen::{ArwBuilder, LensTables};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn exercise(src: &PhotoSource, what: &str) {
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    let (w, h) = r
        .final_size(src, &cancel)
        .unwrap_or_else(|e| panic!("{what}: size: {e}"));
    assert!(w >= 1 && h >= 1, "{what}: {w}x{h}");
    for (cw, ch, percent) in [
        (64.0, 48.0, None),
        (7.0, 5.0, None),
        (64.0, 48.0, Some(100.0)),
        (50.0, 50.0, Some(800.0)),
    ] {
        let mut st = ViewportState::new(ViewportRect::cell(cw, ch, 1.0));
        if let Some(p) = percent {
            st.set_mode(ViewportMode::Percent { value: p }, (w, h));
        }
        let f = r
            .viewport_frame(src, &st, &FrameOptions::default(), &cancel)
            .unwrap_or_else(|e| panic!("{what}: frame {cw}x{ch} {percent:?}: {e}"));
        assert_eq!((f.width, f.height), (cw as usize, ch as usize), "{what}");
        assert!(f.rgba.chunks(4).all(|p| p[3] == 255), "{what}: alpha");
    }
    let png = r
        .thumbnail_png(src, &cancel)
        .unwrap_or_else(|e| panic!("{what}: thumbnail: {e}"));
    assert_eq!(&png[1..4], b"PNG");
    let hist = r
        .histogram(src, &cancel)
        .unwrap_or_else(|e| panic!("{what}: histogram: {e}"));
    assert_eq!(hist.red.len(), 64);
    let full = r
        .render_full(
            src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &cancel,
        )
        .unwrap_or_else(|e| panic!("{what}: full: {e}"));
    assert_eq!((full.width, full.height), (w, h), "{what}");
    assert!(full.rgb.iter().all(|v| v.is_finite()), "{what}: finite");
    let dng = r
        .render_full(
            src,
            &FullRequest {
                target: FullTarget::DngLinearAwg4,
            },
            &cancel,
        )
        .unwrap_or_else(|e| panic!("{what}: dng: {e}"));
    assert!(dng.rgb.iter().all(|v| v.is_finite()), "{what}: finite dng");
    let _ = r.sample_pixel(
        src,
        &ViewportState::new(ViewportRect::cell(64.0, 48.0, 1.0)),
        (3.0, 3.0),
        &cancel,
    );
}

#[test]
fn tiny_and_odd_sizes_render_without_panicking() {
    for (w, h) in [
        (8u32, 8u32),
        (16, 10),
        (33, 17),
        (64, 48),
        (130, 6),
        (6, 130),
        (257, 255),
    ] {
        let scene =
            Scene::new(w, h, light(PROFILE, 5200.0), Layout::Patchwork, 0.3).noise(1e-4, 1e-6);
        let d = Developed::new(&scene, MODEL);
        exercise(&d.source(), &format!("{w}x{h}"));
    }
}

#[test]
fn black_and_white_frames_render_finite_pictures() {
    for (level, name) in [(0.0, "black"), (6.0, "white (every photosite clipped)")] {
        let scene = Scene::new(200, 150, light(PROFILE, 5500.0), Layout::Uniform, level);
        let d = Developed::new(&scene, MODEL);
        exercise(&d.source(), name);
    }
}

#[test]
fn every_layout_with_lens_tables_and_a_turn_renders() {
    let scene =
        Scene::new(240, 160, light(PROFILE, 4300.0), Layout::Patchwork, 0.8).noise(3e-4, 2e-6);
    for exif in [1u16, 6] {
        let dir = tempfile::tempdir().expect("dir");
        scene
            .builder(MODEL)
            .orientation(exif)
            .lens_tables(LensTables::default())
            .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
            .expect("write");
        let d = Developed::open(dir);
        let mut s = d.source().settings.clone();
        s.orientation = Orientation::Deg270;
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.05,
                y: 0.1,
                width: 0.7,
                height: 0.8,
            },
            angle_deg: -13.0,
            aspect: CropAspect::Free,
            auto: false,
        });
        exercise(&d.source().with_settings(s), &format!("exif {exif}"));
    }
}

#[test]
fn an_extreme_straighten_and_a_sliver_of_a_crop_still_render() {
    let scene = Scene::new(300, 200, light(PROFILE, 5500.0), Layout::Patchwork, 0.4);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    for (rect, angle) in [
        (
            NormRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            45.0,
        ),
        (
            NormRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            -45.0,
        ),
        (
            NormRect {
                x: 0.4,
                y: 0.4,
                width: 0.001,
                height: 1.0,
            },
            0.0,
        ),
        (
            NormRect {
                x: 0.99,
                y: 0.99,
                width: 0.01,
                height: 0.01,
            },
            30.0,
        ),
    ] {
        let mut s = d.source().settings.clone();
        s.crop = Some(Crop {
            rect,
            angle_deg: angle,
            aspect: CropAspect::Free,
            auto: false,
        });
        let src = d.source().with_settings(s);
        let (w, h) = r.final_size(&src, &cancel).expect("size");
        assert!(w >= 1 && h >= 1);
        let f = r
            .viewport_frame(
                &src,
                &ViewportState::new(ViewportRect::cell(100.0, 80.0, 2.0)),
                &FrameOptions::default(),
                &cancel,
            )
            .expect("frame");
        assert_eq!((f.width, f.height), (200, 160));
        r.render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::DisplayP3),
            },
            &cancel,
        )
        .expect("full");
    }
}

#[test]
fn an_unreadable_photo_is_an_error_not_a_panic() {
    // A frame whose levels have no range cannot be developed: the session skips it, so there is
    // nothing to render; the renderer still refuses a source built from such an image.
    let scene = Scene::new(64, 48, light(PROFILE, 5500.0), Layout::Uniform, 0.3);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .black([20000; 4])
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let session = ym_core::session::Session::new(base_env());
    let out = session.open(
        &[dir.path().to_path_buf()],
        ym_core::session::OpenOptions::default(),
    );
    settle(&session);
    assert!(PhotoSource::from_session(&session, out.added[0]).is_err());
    let _ = ArwBuilder::new(MODEL, 8, 8);
}
