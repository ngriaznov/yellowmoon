//! The preview API: pixel readouts, Pick white, thumbnails, look tiles, the histogram, render
//! stats, cancellation and the memory budget.
mod common;

use std::sync::Arc;

use common::*;
use ym_color::cdl::Cdl;
use ym_color::display::{OutputTarget, View};
use ym_color::look::{LookInput, LookTransform};
use ym_color::warp::{WarpPin, chroma_of};
use ym_core::pipeline::color::{GradeRequest, color_plan, grade_key, grade_plan};
use ym_core::pipeline::{
    ChromaSample, FrameOptions, FullRequest, FullTarget, LookProblem, LookProvider, NeutralOnly,
    PhotoSource, RenderError, RenderOptions, Renderer, ResolvedLook, ViewportMode, ViewportRect,
    ViewportState,
};
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{EventSink, OpenOptions, Session, SessionEvent};
use ym_core::settings::{AutoOr, DevelopSettings, ExposureValue, Orientation, SettingsPatch};
use ym_gpu::queue::CancelToken;
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn state(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn decode_png(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("a PNG");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("frame");
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    buf.truncate(info.buffer_size());
    (info.width as usize, info.height as usize, buf)
}

fn patchwork() -> Scene {
    Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4)
}

#[test]
fn the_pixel_readout_agrees_with_the_frame_and_the_letterbox_has_none() {
    let d = Developed::new(&patchwork(), MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    // A 300 x 300 cell: the 3:2 photo is letterboxed above and below (100 px of image at 50 %).
    let st = state(300.0, 300.0);
    let frame = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    assert_eq!(
        r.sample_pixel(&src, &st, (150.0, 5.0), &CancelToken::new())
            .expect("sample"),
        None
    );
    assert_eq!(
        r.sample_pixel(&src, &st, (150.0, 295.0), &CancelToken::new())
            .expect("sample"),
        None
    );
    // Patch centres: 50 px wide cells, the 150 px tall image starts at y = 75 (37.5 px rows).
    for (x, y) in [(75usize, 131usize), (175, 169), (225, 94), (25, 94)] {
        let s = r
            .sample_pixel(
                &src,
                &st,
                (x as f64 + 0.5, y as f64 + 0.5),
                &CancelToken::new(),
            )
            .expect("sample")
            .expect("inside");
        let m = mean_rgba(&frame, x - 3, y - 3, 7, 7);
        for (c, v) in [s.r, s.g, s.b].into_iter().enumerate() {
            assert!(
                (f64::from(v) - m[c]).abs() <= 2.0,
                "({x},{y}) channel {c}: {v} vs {:.1}",
                m[c]
            );
        }
    }
    // Before mode shows the camera default rendering, which is not what the final view shows.
    let mut before = st.clone();
    before.before = true;
    let fin = r
        .sample_pixel(&src, &st, (150.0, 150.0), &CancelToken::new())
        .expect("s")
        .expect("in");
    let camera_default = r
        .sample_pixel(&src, &before, (150.0, 150.0), &CancelToken::new())
        .expect("s")
        .expect("in");
    assert_ne!(
        (fin.r, fin.g, fin.b),
        (camera_default.r, camera_default.g, camera_default.b)
    );
}

#[test]
fn a_click_maps_back_to_the_pick_white_space_through_orientation_and_turns() {
    let scene = patchwork();
    let r = Renderer::neutral_only().expect("render thread");
    for exif in [1u16, 3, 6, 8] {
        let dir = tempfile::tempdir().expect("dir");
        scene
            .builder(MODEL)
            .orientation(exif)
            .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
            .expect("write");
        let d = Developed::open(dir);
        let src = d.source();
        let (cw, ch) = if matches!(exif, 6 | 8) {
            (200.0, 300.0)
        } else {
            (300.0, 200.0)
        };
        let st = state(cw, ch);
        for (u, v) in [(0.25, 0.25), (0.8, 0.4), (0.5, 0.9)] {
            let p = r
                .pick_point_at(&src, &st, (u * cw, v * ch), &CancelToken::new())
                .expect("pick")
                .expect("inside");
            // The cell is the photo as the camera oriented it, so the click is the point.
            assert!(
                (p[0] - u).abs() < 0.01 && (p[1] - v).abs() < 0.01,
                "exif {exif}: {p:?} vs ({u},{v})"
            );
        }
        // With the user's turn the click is in the turned photo; the pick stays in the camera's.
        let turned = edit(&src, |s| s.orientation = Orientation::Deg90);
        let (tw, th) = (ch, cw);
        let p = r
            .pick_point_at(
                &turned,
                &state(tw, th),
                (0.25 * tw, 0.75 * th),
                &CancelToken::new(),
            )
            .expect("pick")
            .expect("inside");
        // Canvas (u, v) = (0.25, 0.75) is oriented (v, 1 - u) = (0.75, 0.75).
        assert!(
            (p[0] - 0.75).abs() < 0.01 && (p[1] - 0.75).abs() < 0.01,
            "exif {exif} turned: {p:?}"
        );
    }
    // Outside the photo (the letterbox) there is nothing to pick.
    let d = Developed::new(&scene, MODEL);
    assert_eq!(
        r.pick_point_at(
            &d.source(),
            &state(300.0, 300.0),
            (150.0, 3.0),
            &CancelToken::new()
        )
        .expect("pick"),
        None
    );
}

#[test]
fn pick_white_turns_a_wrong_as_shot_balance_neutral_end_to_end() {
    // The light is 3500 K but the camera's white balance says daylight: Auto renders a cast.
    let true_light = light(PROFILE, 3500.0);
    let scene = Scene::new(600, 400, true_light, Layout::Patchwork, 0.4);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .wb(Some(wb_levels(light(PROFILE, 6500.0))))
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let d = Developed::open(dir);
    let r = Renderer::neutral_only().expect("render thread");
    let st = state(300.0, 200.0);
    let patch = |f: &ym_core::pipeline::RenderedFrame| mean_rgba(f, 70, 55, 30, 30); // patch (1, 1)
    let before = r
        .viewport_frame(
            &d.source(),
            &st,
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    let cast = patch(&before);
    assert!(
        (cast[0] - cast[2]).abs() > 8.0,
        "the wrong white balance should show: {cast:?}"
    );
    // Pick the patch: the session samples it and the chain takes the sample as the scene white.
    let point = [(1.0 + 0.5) / 6.0, (1.0 + 0.5) / 4.0];
    d.session
        .update_settings(
            d.id,
            &SettingsPatch {
                white_pick: Some(Some(point)),
                ..SettingsPatch::default()
            },
        )
        .expect("pick white");
    let src = d.source();
    let after = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    let m = patch(&after);
    assert!(
        (m[0] - m[1]).abs() < 2.0 && (m[2] - m[1]).abs() < 2.0,
        "picked patch is neutral: {m:?}"
    );
}

#[test]
fn thumbnails_are_small_pngs_with_the_photos_shape() {
    let scene = Scene::new(900, 600, light(PROFILE, 5500.0), Layout::Patchwork, 0.4);
    let r = Renderer::neutral_only().expect("render thread");
    let d = Developed::new(&scene, MODEL);
    let png = r
        .thumbnail_png(&d.source(), &CancelToken::new())
        .expect("thumbnail");
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let (w, h, px) = decode_png(&png);
    assert_eq!((w, h), (240, 160));
    let spread = px.iter().copied().max().unwrap() - px.iter().copied().min().unwrap();
    assert!(spread > 40, "a patchwork has contrast");
    // A portrait photo (EXIF 6) gives a portrait thumbnail, a crop its own shape.
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .orientation(6)
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let p = Developed::open(dir);
    let (w, h, _) = decode_png(
        &r.thumbnail_png(&p.source(), &CancelToken::new())
            .expect("png"),
    );
    assert_eq!((w, h), (160, 240));
    let cropped = edit(&d.source(), |s| {
        s.crop = Some(ym_core::settings::Crop {
            rect: ym_core::settings::NormRect {
                x: 0.0,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
            angle_deg: 0.0,
            aspect: ym_core::settings::CropAspect::Free,
            auto: false,
        })
    });
    let (w, h, _) = decode_png(&r.thumbnail_png(&cropped, &CancelToken::new()).expect("png"));
    assert_eq!(
        (w, h),
        (180, 240),
        "half the width of a 3:2 photo, long edge 240"
    );
    // A photo smaller than the thumbnail is not enlarged.
    let tiny = Developed::new(
        &Scene::new(160, 120, light(PROFILE, 5500.0), Layout::Uniform, 0.3),
        MODEL,
    );
    let (w, h, _) = decode_png(
        &r.thumbnail_png(&tiny.source(), &CancelToken::new())
            .expect("png"),
    );
    assert_eq!((w, h), (160, 120));
}

#[test]
fn the_session_gets_its_thumbnails_from_the_render_graph() {
    let renderer = Renderer::neutral_only().expect("render thread");
    let (sink, rx) = EventSink::channel();
    let env = base_env()
        .with_events(sink)
        .with_thumbnails(renderer.thumbnail_renderer());
    let session = Session::new(env);
    let dir = tempfile::tempdir().expect("dir");
    for (i, level) in [0.2, 0.3, 0.4].into_iter().enumerate() {
        Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Uniform, level).write(
            dir.path(),
            &format!("DSC0000{}.ARW", i + 1),
            MODEL,
        );
    }
    let out = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let mut thumbs = std::collections::HashMap::new();
    while let Ok(e) = rx.try_recv() {
        if let SessionEvent::ThumbnailReady(t) = e {
            thumbs.insert(t.id, t.png_bytes);
        }
    }
    assert_eq!(thumbs.len(), 3, "a thumbnail per photo");
    // Brighter scenes give brighter thumbnails.
    let mean = |id| {
        let (_, _, px) = decode_png(&thumbs[&id]);
        px.iter().map(|v| f64::from(*v)).sum::<f64>() / px.len() as f64
    };
    assert!(mean(out.added[0]) < mean(out.added[1]) && mean(out.added[1]) < mean(out.added[2]));
    // At most three mosaics were ever held, however many photos developed.
    assert!(session.max_cached() <= 3);
}

struct TestLooks {
    warm: Arc<LookTransform>,
}

impl TestLooks {
    fn new() -> Self {
        let cdl = Cdl::new([1.25, 1.0, 0.8], [0.0; 3], [1.0; 3], 1.0).expect("cdl");
        Self {
            warm: Arc::new(LookTransform::new(LookInput::LogC4, Some(cdl), None).expect("look")),
        }
    }
}

impl LookProvider for TestLooks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        match id.as_str() {
            "builtin:neutral" => NeutralOnly.look(id),
            "user:warm" => Ok(ResolvedLook {
                transform: Arc::clone(&self.warm),
                fingerprint: "warm-v1".to_owned(),
            }),
            _ => Err(LookProblem(
                "File missing or changed. Import again.".to_owned(),
            )),
        }
    }
}

#[test]
fn look_tiles_apply_one_look_to_the_current_render_and_share_the_raw_group() {
    let d = Developed::new(&patchwork(), MODEL);
    let r =
        Renderer::new(Arc::new(TestLooks::new()), RenderOptions::default()).expect("render thread");
    let src = d.source();
    let neutral = r.thumbnail_png(&src, &CancelToken::new()).expect("neutral");
    let after_neutral = r.counters(src.id);
    let warm_id = LookId::parse("user:warm").expect("id");
    let warm = r
        .look_thumbnail_png(&src, &warm_id, &CancelToken::new())
        .expect("warm");
    let after_tile = r.counters(src.id);
    let (w, h, a) = decode_png(&neutral);
    let (w2, h2, b) = decode_png(&warm);
    assert_eq!((w, h), (w2, h2));
    let mean =
        |px: &[u8], c: usize| px.chunks(3).map(|p| f64::from(p[c])).sum::<f64>() / (w * h) as f64;
    assert!(mean(&b, 0) > mean(&a, 0) + 3.0, "the warm look adds red");
    assert!(mean(&b, 2) < mean(&a, 2) - 3.0, "and takes blue away");
    // The raw stages ran for the thumbnail and not again for the tile: the tile only built its
    // grade plan and ran one more colour pass.
    assert_eq!(
        after_tile.raw_work(),
        after_neutral.raw_work(),
        "{after_tile:?}"
    );
    assert_eq!(after_tile.sigma_planes, after_neutral.sigma_planes);
    assert_eq!(after_tile.color_plans, after_neutral.color_plans);
    assert_eq!(after_tile.grade_plans, after_neutral.grade_plans + 1);
    assert_eq!(after_tile.color_passes, after_neutral.color_passes + 1);
    // A look that is gone is an error with the library's words, not a Neutral tile.
    let gone = LookId::parse("user:gone").expect("id");
    match r.look_thumbnail_png(&src, &gone, &CancelToken::new()) {
        Err(RenderError::Look(m)) => assert_eq!(m, "File missing or changed. Import again."),
        other => panic!("{other:?}"),
    }
    // In the viewport the same missing look renders as Neutral and says so.
    let st = state(300.0, 200.0);
    let missing = edit(&src, |s| s.look = gone.clone());
    let f = r
        .viewport_frame(&missing, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    let base = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    assert_eq!(f.rgba, base.rgba);
    assert!(
        f.report
            .notes
            .iter()
            .any(|n| n.contains("File missing or changed"))
    );
}

#[test]
fn the_histogram_of_a_flat_field_is_one_spike_per_channel() {
    let d = Developed::new(
        &Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Uniform, 0.1),
        MODEL,
    );
    let r = Renderer::neutral_only().expect("render thread");
    let h = r
        .histogram(&d.source(), &CancelToken::new())
        .expect("histogram");
    for ch in [&h.red, &h.green, &h.blue] {
        assert_eq!(ch.len(), 64);
        let nonzero = ch.iter().filter(|v| **v > 0.0).count();
        assert!(nonzero <= 2, "{nonzero} bins");
    }
    let peak = h
        .red
        .iter()
        .chain(&h.green)
        .chain(&h.blue)
        .copied()
        .fold(0.0, f64::max);
    assert_eq!(peak, 1.0);
    // The wire shape is the TypeScript one.
    let json = serde_json::to_value(&h).expect("json");
    assert!(json["red"].is_array() && json["green"].is_array() && json["blue"].is_array());
    // More exposure moves the spike to a higher bin.
    let bin = |h: &Vec<f64>| h.iter().position(|v| *v > 0.5).expect("a spike");
    let src = d.source();
    let lo = r
        .histogram(
            &edit(&src, |s| {
                s.exposure = AutoOr::Manual(ExposureValue { ev: -1.0 })
            }),
            &CancelToken::new(),
        )
        .expect("h");
    let hi = r
        .histogram(
            &edit(&src, |s| {
                s.exposure = AutoOr::Manual(ExposureValue { ev: 1.0 })
            }),
            &CancelToken::new(),
        )
        .expect("h");
    assert!(bin(&lo.green) < bin(&hi.green));
}

#[test]
fn render_stats_say_which_path_ran_how_long_and_in_which_space() {
    let d = Developed::new(&patchwork(), MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    assert_eq!(r.last_stats(src.id), None);
    let f = r
        .viewport_frame(
            &src,
            &state(300.0, 200.0),
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    let stats = r.last_stats(src.id).expect("stats");
    assert_eq!(stats, f.report.stats());
    let text = stats.status_text();
    assert!(
        text.starts_with("Classical · ") && text.ends_with(" ms · Display P3"),
        "{text}"
    );
    assert_eq!(f.report.path, ym_core::pipeline::RenderPath::Classical);
    // The reason the network did not run is recorded, and goes away when the user switched it off.
    assert!(
        f.report
            .notes
            .iter()
            .any(|n| n.contains("No trained demosaic model")),
        "{:?}",
        f.report.notes
    );
    let off = edit(&src, |s| s.decode.neural_demosaic = false);
    let f = r
        .viewport_frame(
            &off,
            &state(300.0, 200.0),
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    assert!(f.report.notes.is_empty(), "{:?}", f.report.notes);
    // JSON of the event payload: the TypeScript names.
    let json = serde_json::to_value(stats).expect("json");
    assert_eq!(json["path"], "classical");
    assert_eq!(json["space"], "display_p3");
    assert!(json["ms"].is_number());
}

#[test]
fn a_cancelled_request_stops_with_cancelled_and_does_no_work() {
    let d = Developed::new(&patchwork(), MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let token = CancelToken::new();
    token.cancel();
    let src = d.source();
    match r.viewport_frame(&src, &state(300.0, 200.0), &FrameOptions::default(), &token) {
        Err(RenderError::Cancelled) => {}
        other => panic!("{other:?}"),
    }
    let c = r.counters(src.id);
    assert_eq!(
        (c.raw_groups, c.sigma_planes, c.color_passes),
        (0, 0, 0),
        "{c:?}"
    );
}

#[test]
fn a_full_render_that_does_not_fit_the_budget_is_refused_and_one_that_does_releases_it() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    let req = FullRequest {
        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
    };
    let tiny = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            budget: MemoryBudget::default().with_cpu_bytes(1 << 20),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    match tiny.render_full(&src, &req, &CancelToken::new()) {
        Err(RenderError::Memory) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(tiny.ledger().used(), 0);
    let r = Renderer::neutral_only().expect("render thread");
    let full = r
        .render_full(&src, &req, &CancelToken::new())
        .expect("full render");
    assert_eq!((full.width, full.height), (600, 400));
    assert_eq!(r.ledger().used(), 0, "the reservation is released");
    assert!(
        r.ledger().peak() > (600 * 400 * 12) as u64,
        "the output was counted"
    );
    // The DNG stop point is linear AWG4: a neutral patch is neutral and positive.
    let dng = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DngLinearAwg4,
            },
            &CancelToken::new(),
        )
        .expect("dng render");
    let i = (250 * 600 + 350) * 3; // the middle of the neutral patch (3, 2)
    let (a, b, c) = (dng.rgb[i], dng.rgb[i + 1], dng.rgb[i + 2]);
    assert!(
        a > 0.0 && (a - b).abs() < 0.02 * b.max(1e-3) && (c - b).abs() < 0.02 * b.max(1e-3),
        "{a} {b} {c}"
    );
}

#[test]
fn full_renders_apply_crop_straighten_and_turns() {
    let d = Developed::new(&patchwork(), MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = edit(&d.source(), |s| {
        s.orientation = Orientation::Deg90;
        s.crop = Some(ym_core::settings::Crop {
            rect: ym_core::settings::NormRect {
                x: 0.1,
                y: 0.2,
                width: 0.5,
                height: 0.6,
            },
            angle_deg: 0.0,
            aspect: ym_core::settings::CropAspect::Free,
            auto: false,
        });
    });
    let full = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("full");
    // The canvas is 400 x 600 after the turn; the crop is 0.5 x 0.6 of it.
    assert_eq!((full.width, full.height), (200, 360));
    // A straightened crop leaves zeros in the empty corners of the canvas.
    let tilted = edit(&d.source(), |s| {
        s.crop = Some(ym_core::settings::Crop {
            rect: ym_core::settings::NormRect::FULL,
            angle_deg: 20.0,
            aspect: ym_core::settings::CropAspect::Original,
            auto: false,
        });
    });
    let t = r
        .render_full(
            &tilted,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("tilted");
    assert_eq!((t.width, t.height), (600, 400));
    assert_eq!(&t.rgb[..3], &[0.0, 0.0, 0.0]);
    let mid = (200 * 600 + 300) * 3;
    assert!(t.rgb[mid + 1] > 0.0);
    // The 100 % view of the same state is the same picture.
    let mut st = state(600.0, 400.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (600, 400));
    let f = r
        .viewport_frame(&tilted, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    assert_eq!(
        &f.rgba[..4],
        &[255, 255, 255, 255],
        "the viewport paints the empty corner"
    );
}

#[test]
fn a_viewport_frame_reports_its_zoom_and_the_fit_zoom_in_the_render_stats() {
    // The webview used to estimate these from the photo's size and the cell; only the core knows
    // the lens crop, the straighten angle and the crop that lie between them.
    let scene = Scene::new(900, 600, light(PROFILE, 5500.0), Layout::Patchwork, 0.4);
    let r = Renderer::neutral_only().expect("render thread");
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let cancel = CancelToken::new();
    let image = r.final_size(&src, &cancel).expect("final size");
    let mut st = state(300.0, 200.0);
    let fit = (300.0 / image.0 as f64).min(200.0 / image.1 as f64) * 100.0;

    let frame = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    let stats = frame.report.stats();
    assert!(
        (stats.zoom_percent.expect("zoom") - fit).abs() < 1e-9,
        "{stats:?}"
    );
    assert!(
        (stats.fit_percent.expect("fit") - fit).abs() < 1e-9,
        "{stats:?}"
    );
    // The header's status text is the last viewport frame's, zoom included.
    assert_eq!(r.last_stats(src.id), Some(stats));

    // 100 %: the zoom is 100, the Fit zoom does not move.
    st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    let frame = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    let stats = frame.report.stats();
    assert!(
        (stats.zoom_percent.expect("zoom") - 100.0).abs() < 1e-9,
        "{stats:?}"
    );
    assert!(
        (stats.fit_percent.expect("fit") - fit).abs() < 1e-9,
        "{stats:?}"
    );
    // A frame served from the cache says the same (the zoom belongs to the request).
    let again = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    assert_eq!(again.report.stats().zoom_percent, stats.zoom_percent);

    // A fixed zoom below Fit is Fit: the frame says what it drew, and the Fit zoom beside it.
    st.set_mode(ViewportMode::Percent { value: 1.0 }, image);
    let frame = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    let stats = frame.report.stats();
    assert!(
        (stats.zoom_percent.expect("zoom") - fit).abs() < 1e-9,
        "{stats:?}"
    );

    // A crop changes the final image and with it the Fit zoom, which the core alone can say.
    let cropped = edit(&src, |s| {
        s.crop = Some(ym_core::settings::Crop {
            rect: ym_core::settings::NormRect {
                x: 0.0,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
            angle_deg: 0.0,
            aspect: ym_core::settings::CropAspect::Free,
            auto: false,
        });
    });
    let st = state(300.0, 200.0);
    let image = r.final_size(&cropped, &cancel).expect("final size");
    let fit = (300.0 / image.0 as f64).min(200.0 / image.1 as f64) * 100.0;
    let stats = r
        .viewport_frame(&cropped, &st, &FrameOptions::default(), &cancel)
        .expect("frame")
        .report
        .stats();
    assert!(
        (stats.fit_percent.expect("fit") - fit).abs() < 1e-9,
        "{stats:?}"
    );

    // Anything but a viewport frame has no zoom.
    let thumb = r.spawn_thumbnail(src.clone()).wait().expect("job");
    assert!(thumb.is_ok());
    assert_eq!(
        r.last_stats(src.id).map(|s| s.zoom_percent.is_some()),
        Some(true)
    );
}

#[test]
fn a_thumbnail_keeps_its_detail_statistic_for_the_size_estimate_and_a_look_tile_does_not() {
    use ym_export::DetailStat;
    let d = Developed::new(&patchwork(), MODEL);
    let r =
        Renderer::new(Arc::new(TestLooks::new()), RenderOptions::default()).expect("render thread");
    let src = d.source();
    assert_eq!(r.thumbnail_detail(src.id), None, "no thumbnail yet");

    // The statistic is the one of the PNG the thumbnail hands out.
    let png = r
        .thumbnail_png(&src, &CancelToken::new())
        .expect("thumbnail");
    let (w, h, rgb) = decode_png(&png);
    let want = DetailStat::measure_rgb8(w as u32, h as u32, &rgb).expect("measurable");
    assert_eq!(r.thumbnail_detail(src.id), Some(want));
    assert!(want.gradient() > 0.0 && want.gradient().is_finite());

    // A look tile is the photo through another look: it leaves the figure alone, even for a tile
    // of a picture that differs (here by 2.5 stops more exposure).
    let brighter = edit(&src, |s| {
        s.exposure = AutoOr::Manual(ExposureValue { ev: 2.5 });
    });
    let warm = LookId::parse("user:warm").expect("id");
    r.look_thumbnail_png(&brighter, &warm, &CancelToken::new())
        .expect("tile");
    assert_eq!(r.thumbnail_detail(src.id), Some(want));

    // A newer thumbnail (the filmstrip follows a change) replaces it.
    let png = r
        .thumbnail_png(&brighter, &CancelToken::new())
        .expect("thumbnail");
    let (w, h, rgb) = decode_png(&png);
    let newer = DetailStat::measure_rgb8(w as u32, h as u32, &rgb).expect("measurable");
    assert_ne!(newer, want, "the brighter picture has other gradients");
    assert_eq!(r.thumbnail_detail(src.id), Some(newer));

    // It goes with the photo's caches, and a closed session's photo gets none.
    r.forget(src.id);
    assert_eq!(r.thumbnail_detail(src.id), None);
    r.thumbnail_png(&src, &CancelToken::new()).expect("again");
    assert!(r.thumbnail_detail(src.id).is_some());
    r.retire_through(src.id);
    assert_eq!(r.thumbnail_detail(src.id), None);
    assert!(r.thumbnail_png(&src, &CancelToken::new()).is_err());
    assert_eq!(r.thumbnail_detail(src.id), None);
}

// ---------------------------------------------------------------------------------------------
// Colour warp (docs/02 C9b): the render, the grade key, the chroma scatter and Pick colour
// ---------------------------------------------------------------------------------------------

fn with_pins(src: &PhotoSource, pins: &[WarpPin]) -> PhotoSource {
    edit(src, |s| {
        s.color_warp.pins = pins.iter().copied().map(Into::into).collect();
    })
}

/// The most saturated colour of the photo at the warp's input, from the chroma scatter.
fn most_colourful(r: &Renderer, src: &PhotoSource) -> [f64; 2] {
    let scatter = r.chroma_scatter(src, &CancelToken::new()).expect("scatter");
    scatter
        .points
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .max_by(|a, b| a[0].hypot(a[1]).total_cmp(&b[0].hypot(b[1])))
        .expect("points")
}

/// A pin that moves the colour `c` by 0.05 across the plane, at right angles to its hue.
fn moving_pin(c: [f64; 2]) -> WarpPin {
    let r = c[0].hypot(c[1]);
    WarpPin::moving(c, [c[0] - 0.05 * c[1] / r, c[1] + 0.05 * c[0] / r])
}

/// The grade chain the renderer makes for `src` (Neutral-only looks, the cine rendering), built here.
fn chain_of(src: &PhotoSource) -> std::sync::Arc<ym_color::display::ColorChain> {
    let color = color_plan(src).expect("colour plan");
    let req = GradeRequest {
        target: OutputTarget::Viewport,
        dither: false,
        looks: &NeutralOnly,
    };
    grade_plan(src, &color, &req).expect("grade plan").chain
}

#[test]
fn a_colour_warp_moves_the_picture_and_locked_pins_alone_change_nothing() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    let colour = most_colourful(&Renderer::neutral_only().expect("renderer"), &src);
    assert!(colour[0].hypot(colour[1]) > 0.05, "{colour:?}");
    // Each render on a renderer of its own and without the frame cache: the chain itself decides.
    let frame = |src: &PhotoSource, before: bool| {
        let mut st = state(300.0, 200.0);
        st.before = before;
        Renderer::neutral_only()
            .expect("renderer")
            .viewport_frame(
                src,
                &st,
                &FrameOptions {
                    no_cache: true,
                    ..FrameOptions::default()
                },
                &CancelToken::new(),
            )
            .expect("frame")
            .rgba
    };
    let none = frame(&src, false);
    let moved = frame(&with_pins(&src, &[moving_pin(colour)]), false);
    assert_ne!(
        moved, none,
        "a pin that moves a colour of the photo changes the picture"
    );
    let changed = moved.iter().zip(&none).filter(|(a, b)| a != b).count();
    assert!(
        changed < none.len() / 2,
        "only the pin's colours move: {changed}"
    );
    // Locked pins alone, and a pin whose target is its source, are byte for byte no warp.
    for (what, pins) in [
        (
            "locked pins",
            vec![
                WarpPin::locked_at(colour),
                WarpPin::locked_at([-colour[0], -colour[1]]),
            ],
        ),
        ("a pin that stays", vec![WarpPin::moving(colour, colour)]),
    ] {
        assert!(frame(&with_pins(&src, &pins), false) == none, "{what}");
    }
    // Before is the camera default: the warp is not in it.
    assert!(frame(&with_pins(&src, &[moving_pin(colour)]), true) == frame(&src, true));
}

#[test]
fn the_grade_key_follows_the_warp_as_the_chain_prepares_it() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    let color = color_plan(&src).expect("colour plan");
    let req = GradeRequest {
        target: OutputTarget::Viewport,
        dither: true,
        looks: &NeutralOnly,
    };
    let key = |pins: &[WarpPin]| grade_key(&with_pins(&src, pins), &color, &req);
    let none = key(&[]);
    let a = moving_pin([0.1, 0.02]);
    let b = moving_pin([0.1, 0.03]);
    assert_ne!(key(&[a]), none, "a pin that moves re-renders");
    assert_ne!(key(&[a]), key(&[b]), "so does moving it");
    assert_ne!(
        key(&[a]),
        key(&[a, WarpPin::locked_at([0.0, 0.1])]),
        "and a lock"
    );
    // What changes nothing keeps the key, so the cached frame is served.
    assert_eq!(key(&[WarpPin::locked_at([0.1, 0.0])]), none);
    assert_eq!(key(&[WarpPin::moving([0.1, 0.0], [0.1, 0.0])]), none);
    // The plan's chain carries the pins (the device path reads them from its parameters).
    let plan = grade_plan(&with_pins(&src, &[a]), &color, &req).expect("plan");
    assert_eq!(plan.chain.params().warp.pins, vec![a]);
    assert_eq!(plan.key, key(&[a]));
}

#[test]
fn the_chroma_scatter_is_a_fixed_sample_of_the_photo_at_the_warps_input() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    let r = Renderer::neutral_only().expect("renderer");
    let scatter = |r: &Renderer, src: &PhotoSource| {
        r.chroma_scatter(src, &CancelToken::new())
            .expect("scatter")
            .points
    };
    let points = scatter(&r, &src);
    // 600 x 400 is read at 192 x 128 (as the histogram): every 12th pixel, 2048 points.
    assert_eq!(points.len(), 2 * 2048);
    for v in &points {
        assert!(
            ((v * 1e4).round() - v * 1e4).abs() < 1e-6,
            "{v} on the 1e-4 grid"
        );
        assert!(v.abs() < 1.0, "{v}");
    }
    // The same photo and settings give the same points, from this renderer or another.
    assert_eq!(scatter(&r, &src), points);
    assert_eq!(
        scatter(&Renderer::neutral_only().expect("renderer"), &src),
        points
    );
    // The scheduled call answers the same.
    let queued = r
        .spawn_chroma_scatter(src.clone())
        .wait()
        .expect("job")
        .expect("scatter");
    assert_eq!(queued.points, points);
    // The JSON is `{ points: number[] }`.
    assert_eq!(
        serde_json::to_value(&queued).expect("json")["points"]
            .as_array()
            .map(Vec::len),
        Some(points.len())
    );
    // The points are the warp's input: a pin does not move them.
    let colour = most_colourful(&r, &src);
    assert_eq!(scatter(&r, &with_pins(&src, &[moving_pin(colour)])), points);
    // Everything before the warp does: a white balance for a warmer light makes the picture bluer.
    let wb = |kelvin: f64| {
        scatter(
            &r,
            &edit(&src, |s| {
                s.white_balance =
                    AutoOr::Manual(ym_core::settings::WhiteBalanceValue { kelvin, tint: 0.0 });
            }),
        )
    };
    let mean_b = |p: &[f64]| p.iter().skip(1).step_by(2).sum::<f64>() / (p.len() / 2) as f64;
    let (warm, cool) = (wb(3000.0), wb(8000.0));
    assert_eq!(warm.len(), points.len());
    assert!(
        mean_b(&warm) + 0.01 < mean_b(&cool),
        "green against blue: {} at 3000 K, {} at 8000 K",
        mean_b(&warm),
        mean_b(&cool)
    );
}

#[test]
fn pick_colour_reads_the_warp_input_of_the_pixel_under_the_pointer() {
    let d = Developed::new(&patchwork(), MODEL);
    let src = d.source();
    let r = Renderer::neutral_only().expect("renderer");
    // A 300 x 300 cell: the 3:2 photo is letterboxed above and below.
    let st = state(300.0, 300.0);
    let chroma = |src: &PhotoSource, st: &ViewportState, at: (f64, f64)| {
        r.sample_chroma(src, st, at, &CancelToken::new())
            .expect("sample")
    };
    assert_eq!(chroma(&src, &st, (150.0, 5.0)), None);
    assert_eq!(chroma(&src, &st, (150.0, 295.0)), None);
    let chain = chain_of(&src);
    let q = |c: f64| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut colourful = None;
    // Neutral patches, and the coloured patches 8 (green) and 13 (blue) of the patchwork: 50 px
    // columns, 37.5 px rows from y = 75.
    for at in [
        (75.5, 131.5),
        (175.5, 169.5),
        (225.5, 94.5),
        (25.5, 94.5),
        (125.5, 131.5),
        (75.5, 169.5),
    ] {
        let rgb = r
            .camera_rgb_at(&src, &st, at, &CancelToken::new())
            .expect("rgb")
            .expect("inside");
        // The pixel is the one the readout reads...
        let pixel = r
            .sample_pixel(&src, &st, at, &CancelToken::new())
            .expect("pixel")
            .expect("inside");
        let shown = chain.apply_view_pixel(View::Final, rgb);
        assert_eq!(
            (pixel.r, pixel.g, pixel.b),
            (q(shown[0]), q(shown[1]), q(shown[2]))
        );
        // ... and its chroma at the warp's input is the answer, exactly.
        let [a, b] = chroma_of(chain.trace(rgb).c9_logc4);
        let got = chroma(&src, &st, at).expect("inside");
        assert_eq!(got, ChromaSample { a, b }, "{at:?}");
        if a.hypot(b) > 0.05 {
            colourful = Some((at, got));
        }
    }
    // A pin on that colour changes what is shown there, not what Pick colour reads; nor does the
    // Before view.
    let (at, got) = colourful.expect("a coloured patch under one of the points");
    let pinned = with_pins(&src, &[moving_pin([got.a, got.b])]);
    assert_eq!(chroma(&pinned, &st, at), Some(got));
    assert_ne!(
        r.sample_pixel(&pinned, &st, at, &CancelToken::new())
            .expect("pixel"),
        r.sample_pixel(&src, &st, at, &CancelToken::new())
            .expect("pixel")
    );
    let mut before = st.clone();
    before.before = true;
    assert_eq!(chroma(&src, &before, at), Some(got));
    // JSON: `{ a, b }`.
    assert_eq!(
        serde_json::to_value(got).expect("json"),
        serde_json::json!({ "a": got.a, "b": got.b })
    );
}
