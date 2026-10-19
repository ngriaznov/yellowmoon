//! Regression tests for the defects found when part 2 was reviewed: an export holding up the
//! viewport, the current photo's caches pushed out by thumbnails, half-pixel resamples at 100 % and
//! in cropped exports, the header's status text overwritten by other renders, thumbnails of crops
//! made from too coarse an image, and Pick white through a vignetting table.
mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions, Renderer,
    ViewportMode, ViewportRect, ViewportState,
};
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{OpenOptions, Session};
use ym_core::settings::{Crop, CropAspect, DevelopSettings, NormRect};
use ym_gpu::queue::CancelToken;
use ym_raw::testgen::LensTables;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn cell(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn srgb_export() -> FullRequest {
    FullRequest {
        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
    }
}

/// A photo of vertical bars four photosites wide (and a few single-photosite lines), so that any
/// resampling at a fractional offset shows as a softer edge.
fn barred(w: u32, h: u32) -> Developed {
    let white = light(PROFILE, 5500.0);
    let scene = Scene::new(w, h, white, Layout::Uniform, 0.4);
    let peak = white.iter().copied().fold(0.0, f64::max);
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let c = [0usize, 1, 1, 2][((y & 1) * 2 + (x & 1)) as usize];
            let lum = if x % 16 < 4 || x % 37 == 0 { 0.45 } else { 0.1 };
            let s = 0.9 * lum * white[c] / peak;
            data.push((512.0 + s * (16300.0 - 512.0)).round() as u16);
        }
    }
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &data)
        .expect("write");
    Developed::open(dir)
}

// ---------------------------------------------------------------------------------------------
// Export and the viewport
// ---------------------------------------------------------------------------------------------

#[test]
fn a_viewport_frame_does_not_wait_for_an_export_in_progress() {
    let scene =
        Scene::new(1800, 1200, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    // Small export tiles: many pieces for the export to give way between.
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            full_tile_px: 256,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let st = cell(300.0, 200.0);
    // Warm the fit view: the frame below is then a cheap colour pass.
    r.viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("warm");
    let export = r.spawn_full(src.clone(), srgb_export());
    // The export is in progress once it holds its memory reservation.
    let started = Instant::now();
    while r.ledger().used() == 0 {
        assert!(started.elapsed() < Duration::from_secs(60), "never started");
        assert!(export.try_wait().is_none(), "finished before it was seen");
        std::thread::sleep(Duration::from_millis(1));
    }
    let frame = r.spawn_viewport_frame(
        src.clone(),
        st.clone(),
        FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        },
    );
    let f = frame.wait().expect("job").expect("frame");
    assert_eq!((f.width, f.height), (300, 200));
    assert!(
        export.try_wait().is_none() && r.ledger().used() > 0,
        "the frame was served while the export was still running"
    );
    // Giving way changes nothing about the pixels of the export.
    let full = export.wait().expect("job").expect("export");
    let direct = r
        .render_full(&src, &srgb_export(), &CancelToken::new())
        .expect("direct");
    assert!(
        full.rgb == direct.rgb,
        "an export that yields is the same picture"
    );
    assert_eq!(r.ledger().used(), 0);
}

#[test]
fn an_export_waits_for_queued_thumbnails_and_frames_before_it_starts() {
    let scene = Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    // Hold the foreground with a job that is waiting for a signal.
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let (up_tx, up_rx) = std::sync::mpsc::channel::<()>();
    let gate = r
        .scheduler()
        .submit(ym_core::scheduler::Priority::Thumbnail, None, move |_| {
            up_tx.send(()).expect("up");
            rx.recv().expect("gate");
        });
    up_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("running");
    let export = r.spawn_full(src.clone(), srgb_export());
    std::thread::sleep(Duration::from_millis(150));
    assert!(export.try_wait().is_none());
    assert_eq!(
        r.ledger().used(),
        0,
        "the export has not begun while a thumbnail is queued"
    );
    assert_eq!(r.scheduler().foreground_load(), 1);
    tx.send(()).expect("open");
    gate.wait().expect("gate job");
    let full = export.wait().expect("job").expect("export");
    assert_eq!((full.width, full.height), (600, 400));
    assert_eq!(r.scheduler().foreground_load(), 0);
}

#[test]
fn cancelling_an_export_that_is_waiting_for_the_foreground_stops_it() {
    let scene = Scene::new(300, 200, light(PROFILE, 5500.0), Layout::Uniform, 0.4);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let (up_tx, up_rx) = std::sync::mpsc::channel::<()>();
    let gate = r
        .scheduler()
        .submit(ym_core::scheduler::Priority::Viewport, None, move |_| {
            up_tx.send(()).expect("up");
            rx.recv().expect("gate");
        });
    up_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("running");
    let export = r.spawn_full(src, srgb_export());
    std::thread::sleep(Duration::from_millis(50));
    export.cancel();
    // Either the export worker had already taken the job and was waiting for the foreground (the
    // job's own `Cancelled`), or, on a machine too busy to have scheduled the worker within the
    // 50 ms, the job was still in the queue and the scheduler dropped it (`JobError::Cancelled`).
    // Both stop the export while the gate is still closed; what must never happen is a hang
    // (`wait` returning only after the gate opens) or a finished render.
    match export.wait() {
        Ok(Err(ym_core::pipeline::RenderError::Cancelled))
        | Err(ym_gpu::queue::JobError::Cancelled) => {}
        other => panic!("{other:?}"),
    }
    tx.send(()).expect("open");
    gate.wait().expect("gate job");
}

// ---------------------------------------------------------------------------------------------
// The renderer's caches
// ---------------------------------------------------------------------------------------------

#[test]
fn the_current_photos_images_survive_the_thumbnails_of_the_other_photos() {
    let dir = tempfile::tempdir().expect("dir");
    for i in 1..=4 {
        Scene::new(
            400,
            300,
            light(PROFILE, 5500.0),
            Layout::Uniform,
            0.1 + 0.1 * i as f64,
        )
        .write(dir.path(), &format!("DSC0000{i}.ARW"), MODEL);
    }
    let session = Session::new(base_env());
    let out = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let sources: Vec<PhotoSource> = out
        .added
        .iter()
        .map(|id| PhotoSource::from_session(&session, *id).expect("source"))
        .collect();
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            budget: MemoryBudget {
                max_photos: 2,
                ..MemoryBudget::default()
            },
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let st = cell(200.0, 150.0);
    r.navigate(sources[0].id);
    r.viewport_frame(
        &sources[0],
        &st,
        &FrameOptions::default(),
        &CancelToken::new(),
    )
    .expect("frame");
    // The filmstrip's thumbnails of every other photo, one after the other.
    for s in &sources[1..] {
        r.thumbnail_png(s, &CancelToken::new()).expect("thumbnail");
    }
    assert_eq!(
        r.counters(sources[0].id).raw_groups,
        1,
        "the current photo's graph is still there"
    );
    let again = r
        .viewport_frame(
            &sources[0],
            &st,
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    assert_eq!(
        again.report.runs.frame_hits, 1,
        "and so is its finished frame"
    );
    // Leaving the photo makes its graph evictable like any other.
    r.navigate(sources[3].id);
    r.thumbnail_png(&sources[1], &CancelToken::new())
        .expect("thumbnail");
    r.thumbnail_png(&sources[2], &CancelToken::new())
        .expect("thumbnail");
    assert_eq!(
        r.counters(sources[0].id).raw_groups,
        0,
        "the old graph went"
    );
}

#[test]
fn the_status_text_belongs_to_the_last_viewport_frame() {
    let d = Developed::new(
        &Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4),
        MODEL,
    );
    let src = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    let st = cell(300.0, 200.0);
    let f = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    let stats = r.last_stats(src.id).expect("stats");
    assert_eq!(stats, f.report.stats());
    // Nothing else the same photo is asked for replaces it.
    r.thumbnail_png(&src, &cancel).expect("thumbnail");
    r.histogram(&src, &cancel).expect("histogram");
    r.sample_pixel(&src, &st, (150.0, 100.0), &cancel)
        .expect("sample");
    r.pick_point_at(&src, &st, (150.0, 100.0), &cancel)
        .expect("pick");
    r.render_full(&src, &srgb_export(), &cancel)
        .expect("export");
    r.look_thumbnail_png(
        &src,
        &ym_looks::library::LookId::parse("builtin:neutral").expect("id"),
        &cancel,
    )
    .expect("look tile");
    assert_eq!(r.last_stats(src.id), Some(stats));
    // A cached frame is a viewport frame too: it reports its own (tiny) time.
    let hit = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    assert_eq!(hit.report.runs.frame_hits, 1);
    assert_eq!(r.last_stats(src.id), Some(hit.report.stats()));
}

// ---------------------------------------------------------------------------------------------
// Pixel exactness at full resolution
// ---------------------------------------------------------------------------------------------

#[test]
fn at_100_percent_a_pan_or_an_odd_cell_moves_whole_pixels_and_never_resamples() {
    let d = barred(800, 600);
    let src = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let frame = |st: &ViewportState| {
        r.viewport_frame(&src, st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame")
    };
    let hundred = |w: f64, h: f64| {
        let mut st = cell(w, h);
        st.set_mode(ViewportMode::Percent { value: 100.0 }, (800, 600));
        st
    };
    let even = frame(&hundred(300.0, 200.0));
    // A cell one pixel wider has its origin between two pixels: it rounds to the same pixel grid,
    // so the first 300 columns are the same pixels.
    let odd = frame(&hundred(301.0, 200.0));
    assert_eq!(even.report.scale, 1);
    for y in 0..200 {
        let a = &even.rgba[y * 300 * 4..(y + 1) * 300 * 4];
        let b = &odd.rgba[y * 301 * 4..y * 301 * 4 + 300 * 4];
        assert_eq!(a, b, "row {y}");
    }
    // A pan by a fraction of a pixel stays on the pixel; a pan by 0.7 px moves one whole pixel.
    let mut small = hundred(300.0, 200.0);
    small.centre[0] += 0.3 / 800.0;
    assert_eq!(frame(&small).rgba, even.rgba, "0.3 px is no pixel");
    let mut one = hundred(300.0, 200.0);
    one.centre[0] += 0.7 / 800.0;
    let moved = frame(&one);
    for y in 0..200 {
        // The view moved right by one pixel: its column x is the old column x + 1. (The dither of
        // the 8-bit frame is fixed to the cell, not to the photo, so a code may differ by one.)
        let a = &even.rgba[(y * 300 + 1) * 4..(y * 300 + 300) * 4];
        let b = &moved.rgba[y * 300 * 4..(y * 300 + 299) * 4];
        let worst = a
            .iter()
            .zip(b)
            .map(|(p, q)| (i32::from(*p) - i32::from(*q)).abs())
            .max()
            .expect("pixels");
        assert!(worst <= 1, "row {y}: {worst} codes");
    }
    // The edges of the bars are as sharp as the photo's: some pixel is dark and its neighbour is
    // bright (a half-pixel resample would put a mid value between them).
    let row = &even.rgba[100 * 300 * 4..101 * 300 * 4];
    let g: Vec<i32> = row.chunks(4).map(|p| i32::from(p[1])).collect();
    let sharpest = g
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .max()
        .expect("pixels");
    let (lo, hi) = (*g.iter().min().expect("p"), *g.iter().max().expect("p"));
    assert!(
        sharpest * 10 >= (hi - lo) * 7,
        "edge {sharpest} of range {}",
        hi - lo
    );
}

#[test]
fn a_crop_at_a_fractional_origin_is_a_window_of_whole_pixels_in_the_export_and_at_100_percent() {
    let d = barred(1000, 700);
    let src = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    let whole = r.render_full(&src, &srgb_export(), &cancel).expect("whole");
    assert_eq!((whole.width, whole.height), (1000, 700));
    // x 257.3, y 135.17, width 498.1, height 357.91: none of them is a whole pixel.
    let cropped = edit(&src, |s| {
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.2573,
                y: 0.1931,
                width: 0.4981,
                height: 0.5113,
            },
            angle_deg: 0.0,
            aspect: CropAspect::Free,
        })
    });
    let full = r
        .render_full(&cropped, &srgb_export(), &cancel)
        .expect("cropped");
    assert_eq!((full.width, full.height), (498, 358));
    // The window starts on the nearest whole pixel and holds the uncropped export's pixels.
    let (x0, y0) = (257usize, 135usize);
    let mut worst = 0.0f32;
    for y in 0..full.height {
        for x in 0..full.width {
            for c in 0..3 {
                let a = full.rgb[(y * full.width + x) * 3 + c];
                let b = whole.rgb[((y0 + y) * 1000 + x0 + x) * 3 + c];
                worst = worst.max((a - b).abs());
            }
        }
    }
    assert!(
        worst <= 1e-5,
        "the cropped export differs from the window by {worst}"
    );
    // The viewport at 100 % in a cell of the crop's size shows the same pixels.
    let mut st = cell(498.0, 358.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (498, 358));
    let f = r
        .viewport_frame(&cropped, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    assert_eq!(f.report.scale, 1);
    let uncropped_view = {
        let mut s = cell(498.0, 358.0);
        s.set_mode(ViewportMode::Percent { value: 100.0 }, (1000, 700));
        s.centre = [(x0 as f64 + 249.0) / 1000.0, (y0 as f64 + 179.0) / 700.0];
        s.set_rect(s.rect, (1000, 700));
        r.viewport_frame(&src, &s, &FrameOptions::default(), &cancel)
            .expect("frame")
    };
    assert_eq!(
        f.rgba, uncropped_view.rgba,
        "a crop is a window on the same pixels"
    );
}

#[test]
fn a_cell_that_is_absurdly_large_or_not_a_number_is_drawn_at_a_sane_size() {
    use ym_core::pipeline::viewport::MAX_CELL_PX;
    let huge = ViewportRect::cell(1e12, 1e15, 3.0);
    assert_eq!(huge.device_size(), (MAX_CELL_PX, MAX_CELL_PX));
    assert_eq!(
        ViewportRect::cell(f64::NAN, -5.0, 2.0).device_size(),
        (1, 1)
    );
    assert_eq!(
        ViewportRect::cell(100.0, 50.0, f64::NAN).device_size(),
        (100, 50)
    );
    assert_eq!(
        ViewportRect::cell(7680.0, 5120.0, 1.0).device_size(),
        (7680, 5120)
    );
}

// ---------------------------------------------------------------------------------------------
// Thumbnails of crops, Pick white
// ---------------------------------------------------------------------------------------------

#[test]
fn a_thumbnail_of_a_small_crop_is_made_from_an_image_that_has_the_detail() {
    let scene =
        Scene::new(2400, 1600, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let cancel = CancelToken::new();
    let plain = Renderer::neutral_only().expect("render thread");
    plain.thumbnail_png(&src, &cancel).expect("thumbnail");
    let c = plain.counters(src.id);
    assert_eq!(
        (c.raw_groups, c.full_chains),
        (1, 0),
        "the whole photo: the 1/4 image, {c:?}"
    );
    // A crop to a quarter of the width shows 600 px: the 240 px thumbnail needs twice that, i.e.
    // the full-resolution image, not the 1/4 image of the whole photo.
    let cropped = edit(&src, |s| {
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.3,
                y: 0.3,
                width: 0.25,
                height: 0.25,
            },
            angle_deg: 0.0,
            aspect: CropAspect::Free,
        })
    });
    let r = Renderer::neutral_only().expect("render thread");
    r.thumbnail_png(&cropped, &cancel).expect("thumbnail");
    let c = r.counters(src.id);
    assert_eq!(
        (c.raw_groups, c.full_chains),
        (0, 1),
        "tiles of the crop, {c:?}"
    );
}

#[test]
fn pick_white_sample_has_no_vignetting_gain_in_it() {
    // A flat field darkened by a lens (raw = flat * attenuation(r)) with the table in the file.
    use ym_gpu::cpu::vignette::VignetteProfile;
    let (w, h) = (800usize, 600usize);
    let knots: Vec<i16> = (0..16).map(|i| (i * i * 50) as i16).collect();
    let mut tag = vec![16i16];
    tag.extend(&knots);
    let profile = VignetteProfile::from_tag(&knots).expect("profile");
    let white = light(PROFILE, 5500.0);
    let scene = Scene::new(w as u32, h as u32, white, Layout::Uniform, 0.3);
    let mut mosaic = scene.mosaic();
    let hd = 0.5 * (w as f64).hypot(h as f64);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (
                x as f64 + 0.5 - w as f64 / 2.0,
                y as f64 + 0.5 - h as f64 / 2.0,
            );
            let g = profile.attenuation_at(dx.hypot(dy) / hd);
            let i = y * w + x;
            mosaic[i] = (scene.black + (f64::from(mosaic[i]) - scene.black) * g).round() as u16;
        }
    }
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .lens_tables(LensTables {
            vignetting: Some(tag),
            chromatic_aberration: None,
            distortion: None,
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write");
    let d = Developed::open(dir);
    // The session's sample is the one the chain uses (the render has no sample of its own). Pick a
    // spot near the corner, where the gain is large, and the middle, where it is 1.
    let sample_at = |point: [f64; 2]| {
        d.session
            .update_settings(
                d.id,
                &ym_core::settings::SettingsPatch {
                    white_pick: Some(Some(point)),
                    ..ym_core::settings::SettingsPatch::default()
                },
            )
            .expect("pick white");
        d.source().pick.expect("the session took a sample").sample
    };
    let corner_at = [0.9, 0.88];
    let corner = sample_at(corner_at);
    let middle = sample_at([0.5, 0.5]);
    let g = profile.attenuation_at(
        ((corner_at[0] - 0.5) * w as f64).hypot((corner_at[1] - 0.5) * h as f64) / hd,
    );
    assert!(g < 0.8, "the spot is darkened by the lens ({g:.2})");
    for c in 0..3 {
        // The sample is the data as the lens left it (darker by `g`), with no gain put back in...
        assert!(
            (corner[c] / middle[c] - g).abs() < 0.03 * g,
            "channel {c}: corner {corner:?}, middle {middle:?}, attenuation {g:.3}"
        );
        // ...and the lens carries no colour: the three channels are darkened alike.
        assert!(
            ((corner[c] / corner[1]) / (middle[c] / middle[1]) - 1.0).abs() < 0.03,
            "channel {c} against green: corner {corner:?}, middle {middle:?}"
        );
    }
}

#[test]
fn pick_white_on_a_photo_smaller_than_the_window_is_an_answer_not_a_panic() {
    let scene = Scene::new(4, 4, light(PROFILE, 5500.0), Layout::Uniform, 0.3);
    let dir = tempfile::tempdir().expect("dir");
    scene.write(dir.path(), "DSC00001.ARW", MODEL);
    let session = Session::new(base_env());
    let out = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let id = out.added[0];
    let patch = ym_core::settings::SettingsPatch {
        white_pick: Some(Some([0.5, 0.5])),
        ..ym_core::settings::SettingsPatch::default()
    };
    // The 5 x 5 window of the sample is kept inside the photo, so a 4 x 4 photo gives a sample of
    // what there is (this is the branch that runs); a pick that cannot be taken is refused in words
    // and leaves no sample behind. (The render's own sample, which refused a photo under 5 x 5, is
    // gone: the chain never read it.)
    match session.update_settings(id, &patch) {
        Ok(_) => {
            let sample = session
                .view(id)
                .expect("a photo that took a pick can be viewed")
                .picked_white()
                .expect("the sample of the pick");
            assert!(
                sample.iter().all(|v| v.is_finite() && *v > 0.0),
                "{sample:?}"
            );
        }
        Err(e) => {
            assert!(!e.to_string().is_empty(), "an error says what went wrong");
            if let Ok(view) = session.view(id) {
                assert!(
                    view.picked_white().is_none(),
                    "no sample after a refused pick"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The DNG's geometry (`docs/09`: the pixels keep the camera's orientation, a crop without a
// straighten angle becomes a window for the default crop tags)
// ---------------------------------------------------------------------------------------------

fn patchwork_with_exif(exif: u16) -> Developed {
    let scene =
        Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(1e-4, 1e-6);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .orientation(exif)
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    Developed::open(dir)
}

fn full(r: &Renderer, src: &PhotoSource, target: FullTarget) -> ym_core::pipeline::FullImage {
    r.render_full(src, &FullRequest { target }, &CancelToken::new())
        .expect("render")
}

fn pixel(img: &ym_core::pipeline::FullImage, x: usize, y: usize) -> [f32; 3] {
    let i = (y * img.width + x) * 3;
    [img.rgb[i], img.rgb[i + 1], img.rgb[i + 2]]
}

/// R8 capture sharpening off by the setting. A DNG is made without R8 (`docs/09`, "What a DNG carries"), so the
/// picture of another format that it is compared with has none either: these tests are about where the pixels
/// are, not how sharp they are.
fn without_r8(s: &mut DevelopSettings) {
    s.sharpen = ym_core::settings::AutoOr::Manual(ym_core::settings::SharpenValue {
        amount: 0.0,
        radius_px: 1.0,
    });
}

fn assert_same_pixel(a: [f32; 3], b: [f32; 3], tol: f32, what: &str) {
    for c in 0..3 {
        assert!((a[c] - b[c]).abs() <= tol, "{what}: {a:?} vs {b:?}");
    }
}

#[test]
fn the_dng_raw_pixels_keep_the_cameras_orientation_and_the_crop_becomes_a_window() {
    use ym_core::pipeline::PixelWindow;
    use ym_core::settings::Orientation;
    let r = Renderer::neutral_only().expect("render thread");
    let crop = |s: &mut DevelopSettings| {
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.1,
                y: 0.2,
                width: 0.5,
                height: 0.6,
            },
            angle_deg: 0.0,
            aspect: CropAspect::Free,
        });
    };
    // Camera on its side (EXIF 6, a quarter turn clockwise) and the user turns it another quarter:
    // the net turn is a half turn. Canvas 600 x 400, crop (60, 80, 300, 240) on it.
    let d = patchwork_with_exif(6);
    let src = edit(&d.source(), |s| {
        s.orientation = Orientation::Deg90;
        crop(s);
        without_r8(s);
    });
    let baked = full(&r, &src, FullTarget::DisplayLinear(OutputTarget::Srgb));
    assert_eq!((baked.width, baked.height), (300, 240));
    assert_eq!(baked.window, None);
    let preview = full(&r, &src, FullTarget::DngPreview);
    assert_eq!((preview.width, preview.height), (300, 240));
    for y in 0..240 {
        for x in 0..300 {
            assert_same_pixel(
                pixel(&baked, x, y),
                pixel(&preview, 299 - x, 239 - y),
                1e-5,
                &format!("({x},{y}) of the half-turned preview"),
            );
        }
    }
    let raw = full(&r, &src, FullTarget::DngLinearAwg4);
    assert_eq!(
        (raw.width, raw.height),
        (600, 400),
        "the whole lens-cropped buffer"
    );
    assert_eq!(
        raw.window,
        Some(PixelWindow {
            x: 240,
            y: 80,
            width: 300,
            height: 240
        }),
        "the crop seen from the sensor"
    );
    // Only the camera's turn (EXIF 6): the net turn is a quarter turn clockwise. Canvas 400 x 600,
    // crop (40, 120, 200, 360); in the buffer (120, 160, 360, 200).
    let src = edit(&d.source(), |s| {
        crop(s);
        without_r8(s);
    });
    let baked = full(&r, &src, FullTarget::DisplayLinear(OutputTarget::Srgb));
    assert_eq!((baked.width, baked.height), (200, 360));
    let preview = full(&r, &src, FullTarget::DngPreview);
    assert_eq!((preview.width, preview.height), (360, 200));
    for y in 0..360 {
        for x in 0..200 {
            // The preview turned a quarter clockwise is the baked picture: (x', y') -> (h' - 1 - y', x').
            assert_same_pixel(
                pixel(&baked, x, y),
                pixel(&preview, y, 199 - x),
                1e-5,
                &format!("({x},{y}) of the quarter-turned preview"),
            );
        }
    }
    let raw = full(&r, &src, FullTarget::DngLinearAwg4);
    assert_eq!((raw.width, raw.height), (600, 400));
    assert_eq!(
        raw.window,
        Some(PixelWindow {
            x: 120,
            y: 160,
            width: 360,
            height: 200
        })
    );
    // No crop: the window is absent and the raw pixels are the buffer.
    let plain = full(&r, &d.source(), FullTarget::DngLinearAwg4);
    assert_eq!((plain.width, plain.height, plain.window), (600, 400, None));
}

#[test]
fn a_dng_with_a_straighten_angle_bakes_the_angle_and_the_crop_but_not_the_turns() {
    use ym_core::settings::Orientation;
    let r = Renderer::neutral_only().expect("render thread");
    // A mirrored camera orientation (EXIF 2) reverses the sense of the angle in the buffer.
    let d = patchwork_with_exif(2);
    let src = edit(&d.source(), |s| {
        s.orientation = Orientation::Deg180;
        without_r8(s);
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.05,
                y: 0.1,
                width: 0.9,
                height: 0.8,
            },
            angle_deg: 9.0,
            aspect: CropAspect::Free,
        });
    });
    let baked = full(&r, &src, FullTarget::DisplayLinear(OutputTarget::Srgb));
    let preview = full(&r, &src, FullTarget::DngPreview);
    let raw = full(&r, &src, FullTarget::DngLinearAwg4);
    assert_eq!((baked.width, baked.height), (540, 320));
    assert_eq!(
        (preview.width, preview.height),
        (540, 320),
        "a half turn and a mirror keep the shape"
    );
    assert_eq!(
        (raw.width, raw.height),
        (540, 320),
        "the crop is baked into the raw pixels"
    );
    assert_eq!(raw.window, None);
    // The net map is a mirror (EXIF 2 mirrors left to right, the half turn then flips both axes,
    // so top to bottom): the preview is the baked picture mirrored vertically.
    let mut worst = 0.0f32;
    for y in 0..320 {
        for x in 0..540 {
            let (a, b) = (pixel(&baked, x, y), pixel(&preview, x, 319 - y));
            for c in 0..3 {
                worst = worst.max((a[c] - b[c]).abs());
            }
        }
    }
    assert!(worst <= 1e-4, "the straightened preview differs by {worst}");
    // The same without the crop: the canvas keeps its size and the turned corners are empty (zero)
    // in the preview as in the baked picture.
    let whole = edit(&src, |s| {
        s.crop = Some(Crop {
            rect: NormRect::FULL,
            angle_deg: 9.0,
            aspect: CropAspect::Original,
        });
    });
    let preview = full(&r, &whole, FullTarget::DngPreview);
    assert_eq!((preview.width, preview.height), (600, 400));
    assert!(
        [
            pixel(&preview, 0, 0),
            pixel(&preview, 599, 0),
            pixel(&preview, 0, 399),
            pixel(&preview, 599, 399)
        ]
        .iter()
        .filter(|p| **p == [0.0, 0.0, 0.0])
        .count()
            >= 2,
        "two opposite corners of a straightened canvas are empty"
    );
}

// ---------------------------------------------------------------------------------------------
// Cache transparency
// ---------------------------------------------------------------------------------------------

/// A noisy photo with clipped highlights, a vignetting table and a distortion table: every stage
/// group has something to do.
fn busy_photo() -> Developed {
    let white = light(PROFILE, 4300.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 1.5)
        .noise(3e-4, 2e-6)
        .seed(5);
    let mut vign = vec![16i16];
    vign.extend((0..16).map(|i| (i * i * 20) as i16));
    let mut dist = vec![16i16];
    dist.extend((0..16).map(|i| 30 + 14 * i));
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .iso(1600)
        .lens_tables(LensTables {
            vignetting: Some(vign),
            chromatic_aberration: None,
            distortion: Some(dist),
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    Developed::open(dir)
}

#[test]
fn a_renderer_that_remembers_draws_the_pictures_of_one_that_does_not() {
    use ym_core::settings::{
        AutoOr, ExposureValue, NoiseReductionValue, Orientation, SharpenValue, WhiteBalanceValue,
    };
    use ym_raw::testgen::SplitMix64;
    let d = busy_photo();
    let base = d.source();
    let shared = Renderer::neutral_only().expect("render thread");
    let mut rng = SplitMix64(0x5EED);
    let mut unit = move || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut s = base.settings.clone();
    let mut estimates = Arc::clone(&base.estimates);
    let mut st = cell(240.0, 160.0);
    let image = shared.final_size(&base, &CancelToken::new()).expect("size");
    for step in 0..60 {
        let what: String;
        match (unit() * 24.0) as u32 {
            0 => {
                s.contrast = ym_core::settings::PercentSetting::manual_value(unit() * 80.0 - 40.0);
                what = "contrast".into();
            }
            1 => {
                s.brightness = unit() * 80.0 - 40.0;
                what = "brightness".into();
            }
            2 => {
                s.shadows = ym_core::settings::PercentSetting::manual_value(unit() * 100.0 - 50.0);
                s.highlights =
                    ym_core::settings::PercentSetting::manual_value(unit() * 100.0 - 50.0);
                what = "shadows and highlights".into();
            }
            3 => {
                s.saturation =
                    ym_core::settings::PercentSetting::manual_value(unit() * 100.0 - 40.0);
                what = "saturation".into();
            }
            4 => {
                s.color_balance.midtones = [unit() - 0.5, unit() - 0.5];
                what = "colour balance".into();
            }
            5 => {
                s.rolloff.highlights = unit();
                s.rolloff.low_light = unit();
                what = "highlight range and toe".into();
            }
            6 => {
                s.exposure = if unit() < 0.3 {
                    AutoOr::Auto
                } else {
                    AutoOr::Manual(ExposureValue {
                        ev: unit() * 4.0 - 2.0,
                    })
                };
                what = "exposure".into();
            }
            7 => {
                s.white_balance = if unit() < 0.3 {
                    AutoOr::Auto
                } else {
                    AutoOr::Manual(WhiteBalanceValue {
                        kelvin: 2500.0 + 50.0 * (unit() * 120.0).floor(),
                        tint: unit() * 60.0 - 30.0,
                    })
                };
                what = "white balance".into();
            }
            8 => {
                use ym_color::wb::LightSource::*;
                s.light_source =
                    [Auto, Day, Shade, Tungsten, Led, Fluorescent][(unit() * 6.0) as usize % 6];
                what = "light source".into();
            }
            9 => {
                s.look_intensity = (unit() * 100.0).round();
                what = "look intensity".into();
            }
            10 => {
                s.curve.luma = vec![
                    [0.0, 0.0],
                    [0.3 + 0.4 * unit(), 0.2 + 0.6 * unit()],
                    [1.0, 1.0],
                ];
                what = "luma curve".into();
            }
            11 => {
                s.curve.green = vec![[0.0, 0.0], [0.5, 0.35 + 0.3 * unit()], [1.0, 1.0]];
                what = "green curve".into();
            }
            12 => {
                s.decode.highlight_recovery = unit() < 0.5;
                what = "highlight recovery".into();
            }
            13 => {
                s.decode.lens_corrections = unit() < 0.5;
                what = "lens corrections".into();
            }
            14 => {
                s.decode.chromatic_aberration = unit() < 0.5;
                what = "chromatic aberration".into();
            }
            15 => {
                s.noise_reduction = if unit() < 0.3 {
                    AutoOr::Auto
                } else {
                    AutoOr::Manual(NoiseReductionValue {
                        luminance: (unit() * 100.0).round(),
                        color: (unit() * 100.0).round(),
                        detail: (unit() * 100.0).round(),
                    })
                };
                what = "noise reduction".into();
            }
            16 => {
                s.sharpen = if unit() < 0.3 {
                    AutoOr::Auto
                } else {
                    AutoOr::Manual(SharpenValue {
                        amount: (unit() * 300.0).round(),
                        radius_px: 0.3 + 2.2 * unit(),
                    })
                };
                what = "sharpening".into();
            }
            17 => {
                s.orientation = [
                    Orientation::Deg0,
                    Orientation::Deg90,
                    Orientation::Deg180,
                    Orientation::Deg270,
                ][(unit() * 4.0) as usize % 4];
                what = "orientation".into();
            }
            18 => {
                s.crop = (unit() < 0.7).then(|| Crop {
                    rect: NormRect {
                        x: 0.3 * unit(),
                        y: 0.3 * unit(),
                        width: 0.4 + 0.3 * unit(),
                        height: 0.4 + 0.3 * unit(),
                    },
                    angle_deg: if unit() < 0.5 {
                        0.0
                    } else {
                        20.0 * unit() - 10.0
                    },
                    aspect: CropAspect::Free,
                });
                what = "crop".into();
            }
            19 => {
                // The burst median moves the exposure the estimates serve.
                estimates = Arc::new(
                    base.estimates
                        .with_shared_ev(base.estimates.exposure.ev + unit() - 0.5),
                );
                what = "shared exposure".into();
            }
            20 => {
                let mut noise = base.estimates.noise.clone();
                noise.model.a *= 0.5 + unit();
                estimates = Arc::new(base.estimates.with_noise(noise));
                what = "pooled noise".into();
            }
            21 => {
                // Another view: fit, 50 %, 100 % at a random pan.
                st = cell(240.0 + (unit() * 40.0).floor(), 160.0);
                match (unit() * 3.0) as u32 {
                    0 => {}
                    1 => st.set_mode(ViewportMode::Percent { value: 50.0 }, image),
                    _ => st.set_mode(ViewportMode::Percent { value: 100.0 }, image),
                }
                st.centre = [0.3 + 0.4 * unit(), 0.3 + 0.4 * unit()];
                st.set_rect(st.rect, image);
                what = "view".into();
            }
            22 => {
                st.before = unit() < 0.4;
                what = "before".into();
            }
            _ => {
                s = base.settings.clone();
                what = "reset".into();
            }
        }
        let mut src = base.with_settings(s.clone());
        src.estimates = Arc::clone(&estimates);
        let a = shared
            .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame");
        let fresh = Renderer::neutral_only().expect("render thread");
        let b = fresh
            .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("fresh frame");
        assert_eq!(
            (a.width, a.height, a.report.scale),
            (b.width, b.height, b.report.scale),
            "step {step} ({what})"
        );
        assert!(
            a.rgba == b.rgba,
            "step {step} ({what}): the remembering renderer drew another picture ({:?} against {:?})",
            a.report.runs,
            b.report.runs
        );
        if step % 15 == 14 {
            // The other entry points read the same caches: a thumbnail, the histogram and an
            // export are what a fresh renderer makes of the same state.
            let cancel = CancelToken::new();
            assert_eq!(
                shared.thumbnail_png(&src, &cancel).expect("thumbnail"),
                fresh.thumbnail_png(&src, &cancel).expect("thumbnail"),
                "step {step} ({what}): thumbnail"
            );
            assert_eq!(
                shared.histogram(&src, &cancel).expect("histogram"),
                fresh.histogram(&src, &cancel).expect("histogram"),
                "step {step} ({what}): histogram"
            );
            let (e1, e2) = (
                shared
                    .render_full(&src, &srgb_export(), &cancel)
                    .expect("export"),
                fresh
                    .render_full(&src, &srgb_export(), &cancel)
                    .expect("export"),
            );
            assert!(e1.rgb == e2.rgb, "step {step} ({what}): export");
        }
    }
}

#[test]
fn threads_asking_for_different_pictures_of_one_photo_each_get_their_own() {
    // Eight states of one photo (raw group, colour group, grade and geometry all vary), the
    // pictures a fresh renderer makes of them, then four threads hammering one renderer with all
    // kinds of requests in a different order each, while another thread moves between photos and a
    // queued export runs. Every picture must be the one the state asks for.
    use ym_core::settings::AutoOr;
    let d = busy_photo();
    let base = d.source();
    let states: Vec<(PhotoSource, ViewportState)> = (0..8)
        .map(|i| {
            let mut s = base.settings.clone();
            match i % 4 {
                0 => s.contrast = ym_core::settings::PercentSetting::manual_value(10.0 * i as f64),
                1 => {
                    s.exposure = AutoOr::Manual(ym_core::settings::ExposureValue {
                        ev: 0.25 * i as f64 - 1.0,
                    })
                }
                2 => s.decode.highlight_recovery = i % 2 == 0,
                _ => {
                    s.crop = Some(Crop {
                        rect: NormRect {
                            x: 0.05 * i as f64,
                            y: 0.1,
                            width: 0.6,
                            height: 0.6,
                        },
                        angle_deg: i as f64,
                        aspect: CropAspect::Free,
                    })
                }
            }
            let mut st = cell(200.0, 140.0);
            if i % 3 == 0 {
                st.set_mode(ViewportMode::Percent { value: 100.0 }, (480, 320));
                st.centre = [0.35 + 0.04 * i as f64, 0.5];
                st.set_rect(st.rect, (480, 320));
            }
            (base.with_settings(s), st)
        })
        .collect();
    let expected: Vec<Vec<u8>> = states
        .iter()
        .map(|(src, st)| {
            Renderer::neutral_only()
                .expect("render thread")
                .viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
                .expect("frame")
                .rgba
        })
        .collect();
    let r = Renderer::neutral_only().expect("render thread");
    let export = r.spawn_full(base.clone(), srgb_export());
    std::thread::scope(|scope| {
        for t in 0..4usize {
            let (r, states, expected) = (&r, &states, &expected);
            scope.spawn(move || {
                for k in 0..16usize {
                    let i = (t * 5 + k * 3 + k / 2) % states.len();
                    let (src, st) = &states[i];
                    let f = r
                        .viewport_frame(
                            src,
                            st,
                            &FrameOptions {
                                no_cache: k % 3 == 0,
                                coarse_only: k % 5 == 4,
                            },
                            &CancelToken::new(),
                        )
                        .expect("frame");
                    if !(k % 5 == 4 && !f.sharp) {
                        assert!(f.rgba == expected[i], "thread {t} request {k}: state {i}");
                    }
                    if k % 4 == 1 {
                        r.thumbnail_png(src, &CancelToken::new())
                            .expect("thumbnail");
                    }
                    if k % 6 == 2 {
                        r.prepare_tiles(src, st, &CancelToken::new())
                            .expect("tiles");
                    }
                }
            });
        }
        // Someone browsing: moves between the photo and another id.
        let r = &r;
        let id = base.id;
        scope.spawn(move || {
            for k in 0..40u32 {
                r.navigate(if k % 2 == 0 {
                    id
                } else {
                    ym_core::session::PhotoId(id.0 + 50)
                });
                std::thread::sleep(Duration::from_millis(5));
            }
        });
    });
    let e = export.wait().expect("job").expect("export");
    let direct = Renderer::neutral_only()
        .expect("render thread")
        .render_full(&base, &srgb_export(), &CancelToken::new())
        .expect("direct");
    assert!(
        e.rgb == direct.rgb,
        "the export that ran among the others is the same picture"
    );
}

// ---------------------------------------------------------------------------------------------
// Notes
// ---------------------------------------------------------------------------------------------

struct OneLook;

impl ym_core::pipeline::LookProvider for OneLook {
    fn look(
        &self,
        id: &ym_looks::library::LookId,
    ) -> Result<ym_core::pipeline::ResolvedLook, ym_core::pipeline::LookProblem> {
        NeutralOnly.look(id).map_err(|_| {
            ym_core::pipeline::LookProblem("File missing or changed. Import again.".to_owned())
        })
    }

    fn display_transform(
        &self,
        id: &ym_looks::library::LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), ym_core::pipeline::LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

#[test]
fn a_look_that_was_replaced_by_neutral_says_so_on_every_frame_not_only_the_first() {
    let d = Developed::new(
        &Scene::new(300, 200, light(PROFILE, 5500.0), Layout::Patchwork, 0.4),
        MODEL,
    );
    let gone = ym_looks::library::LookId::parse("user:gone").expect("id");
    let src = edit(&d.source(), |s| s.look = gone.clone());
    let r = Renderer::new(Arc::new(OneLook), RenderOptions::default()).expect("render thread");
    let st = cell(150.0, 100.0);
    let says = |f: &ym_core::pipeline::RenderedFrame| {
        f.report
            .notes
            .iter()
            .filter(|n| n.contains("File missing or changed"))
            .count()
    };
    let first = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    assert_eq!(says(&first), 1);
    // The same frame from the cache, and another frame that finds the cached grade plan.
    let again = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    assert_eq!(again.report.runs.frame_hits, 1);
    assert_eq!(says(&again), 1, "{:?}", again.report.notes);
    let other = r
        .viewport_frame(
            &src,
            &cell(160.0, 100.0),
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    assert_eq!(other.report.runs.grade_plans, 0);
    assert_eq!(says(&other), 1, "{:?}", other.report.notes);
}

#[test]
fn a_lens_table_that_cannot_be_used_is_mentioned_on_every_frame_that_ignores_it() {
    let scene = Scene::new(300, 200, light(PROFILE, 5500.0), Layout::Uniform, 0.4);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .lens_tables(LensTables {
            vignetting: None,
            chromatic_aberration: None,
            // A distortion table with two knots where the tag promised sixteen.
            distortion: Some(vec![16, 5, 7]),
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let d = Developed::open(dir);
    let r = Renderer::neutral_only().expect("render thread");
    let st = cell(150.0, 100.0);
    let a = r
        .viewport_frame(
            &d.source(),
            &st,
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    let b = r
        .viewport_frame(
            &d.source(),
            &st,
            &FrameOptions {
                no_cache: true,
                ..FrameOptions::default()
            },
            &CancelToken::new(),
        )
        .expect("frame");
    assert_eq!(
        b.report.runs.raw_work(),
        0,
        "the second frame used the cached plan"
    );
    let ignored = |f: &ym_core::pipeline::RenderedFrame| {
        f.report
            .notes
            .iter()
            .filter(|n| n.contains("Distortion table ignored"))
            .count()
    };
    assert_eq!(ignored(&a), 1, "{:?}", a.report.notes);
    assert_eq!(ignored(&b), 1, "{:?}", b.report.notes);
}

#[test]
fn closing_the_session_cancels_the_thumbnails_waiting_in_the_render_queue() {
    let dir = tempfile::tempdir().expect("dir");
    for i in 1..=6 {
        Scene::new(300, 200, light(PROFILE, 5500.0), Layout::Uniform, 0.3).write(
            dir.path(),
            &format!("DSC0000{i}.ARW"),
            MODEL,
        );
    }
    let r = Renderer::neutral_only().expect("render thread");
    // Hold the render worker so that every thumbnail waits in the queue.
    let (open, gate) = {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (up_tx, up_rx) = std::sync::mpsc::channel::<()>();
        let h = r
            .scheduler()
            .submit(ym_core::scheduler::Priority::Viewport, None, move |_| {
                up_tx.send(()).expect("up");
                rx.recv().expect("gate");
            });
        up_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("running");
        (tx, h)
    };
    let session = Session::new(base_env().with_thumbnails(r.thumbnail_renderer()));
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    // Wait until the decode threads have queued their thumbnails behind the gate.
    let started = Instant::now();
    while r.scheduler().pending() < 2 {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "no thumbnail queued"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    session.close();
    // The hook polls the session's flag, so within a moment every thumbnail that waits has been
    // cancelled (a cancelled job leaves the queue when the worker reaches it). Let it reach them.
    std::thread::sleep(Duration::from_millis(200));
    open.send(()).expect("open");
    gate.wait().expect("gate job");
    let started = Instant::now();
    while r.scheduler().foreground_load() > 0 {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the queue did not drain"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(r.cached_bytes(), 0, "no cancelled thumbnail was rendered");
}

// ---------------------------------------------------------------------------------------------
// Colour fringes of the reduced views
// ---------------------------------------------------------------------------------------------

#[test]
fn a_sharp_edge_has_no_red_and_blue_fringe_in_the_half_resolution_view() {
    // A neutral edge, a dark side and a bright side, with the blur of a lens (a cosine ramp over
    // three photosites). The plain mean of a 2 x 2 block's red and blue photosites sits half a
    // photosite either side of the block's middle, so red and blue came out 85 display codes
    // apart at the edge pixel (a cyan line on every rising edge, an orange one on every falling
    // edge) while a demosaic of the same data has none.
    let (w, h) = (800u32, 600u32);
    let white = light(PROFILE, 5500.0);
    let scene = Scene::new(w, h, white, Layout::Uniform, 0.4);
    let peak = white.iter().copied().fold(0.0, f64::max);
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let c = [0usize, 1, 1, 2][((y & 1) * 2 + (x & 1)) as usize];
            let t = ((x as f64 + 0.5 - 399.5) / 3.0).clamp(0.0, 1.0);
            let lum = 0.03 + 0.47 * 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
            let s = 0.9 * lum * white[c] / peak;
            data.push((512.0 + s * (16300.0 - 512.0)).round() as u16);
        }
    }
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &data)
        .expect("write");
    let d = Developed::open(dir);
    let src = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    for (cw, ch, scale, limit) in [
        (400.0, 300.0, 2, 30),
        (200.0, 150.0, 4, 20),
        (100.0, 75.0, 4, 15),
    ] {
        let f = r
            .viewport_frame(
                &src,
                &cell(cw, ch),
                &FrameOptions::default(),
                &CancelToken::new(),
            )
            .expect("frame");
        assert_eq!(f.report.scale, scale);
        let row = (f.height / 2) * f.width;
        let worst = (0..f.width)
            .map(|x| {
                let p = &f.rgba[(row + x) * 4..(row + x) * 4 + 3];
                (i32::from(p[0]) - i32::from(p[2])).abs()
            })
            .max()
            .expect("pixels");
        assert!(
            worst <= limit,
            "cell {cw}: red and blue differ by {worst} codes at the edge"
        );
    }
}
