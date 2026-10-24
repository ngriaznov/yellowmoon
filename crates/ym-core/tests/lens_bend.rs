//! L4, the lens's bend (`docs/21` section 4, L4; `docs/01` *Geometry*): the anamorphic bend, the lens's
//! own distortion and the zoom that keeps the corners filled, applied while the geometry samples the
//! picture.
//!
//! * The sampler's warp is `ym_lens::LensWarp` bit for bit, and the geometry of a photo carries it.
//! * At every Anamorphic amount, on the four quarter turns, at three straighten angles and at full
//!   resolution and 3, 8 and 16 times smaller, no pixel of the canvas reads outside the photo.
//! * A synthetic photo of lines: a horizontal line near the top bows as the map says, a vertical line
//!   stays within 0.5 px of straight, in the finished picture.
//! * Pick (`pick_point_at`) goes through the map and back.
//! * Anamorphic 0 on a lens without distortion is no bend at all (the geometry is the plain one and a
//!   full-resolution grid is still a pixel copy); the Before view has none.
//! * The DNG's upright pictures are the baked picture, bent, and an export's tiles equal the whole frame.
mod common;

use std::sync::{Arc, OnceLock};

use common::*;
use ym_color::display::OutputTarget;
use ym_core::pipeline::geometry::{Geometry, canvas_to_oriented, oriented_to_canvas, sampler_warp};
use ym_core::pipeline::{
    FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions, Renderer,
    ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, LensKind, LensSettings, Orientation};
use ym_gpu::cpu::sampler::Kernel;
use ym_gpu::queue::CancelToken;
use ym_lens::{Axis, CameraResponse, FrameInfo, LensAmounts, LensLibrary, LensPlan};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn library() -> Arc<LensLibrary> {
    static LIBRARY: OnceLock<Arc<LensLibrary>> = OnceLock::new();
    Arc::clone(LIBRARY.get_or_init(|| {
        Arc::new(LensLibrary::load(&data_root()).expect("the shipped lens data load"))
    }))
}

fn options_with(library: Arc<LensLibrary>, full_tile_px: usize) -> RenderOptions {
    let defaults = RenderOptions::default();
    RenderOptions {
        lens: Some(library),
        data_root: Some(data_root()),
        full_tile_px: if full_tile_px == 0 {
            defaults.full_tile_px
        } else {
            full_tile_px
        },
        ..defaults
    }
}

fn renderer() -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options_with(library(), 0)).expect("render thread")
}

fn lens(kind: LensKind, flares: f64, edges: f64, anamorphic: f64) -> LensSettings {
    LensSettings {
        kind,
        flares,
        edges,
        anamorphic,
    }
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn with_lens(src: &PhotoSource, l: LensSettings) -> PhotoSource {
    edit(src, |s| s.lens = l)
}

fn turned(src: &PhotoSource, quarter_turns: u8) -> PhotoSource {
    edit(src, |s| {
        s.orientation = match quarter_turns % 4 {
            0 => Orientation::Deg0,
            1 => Orientation::Deg90,
            2 => Orientation::Deg180,
            _ => Orientation::Deg270,
        }
    })
}

/// A photo of patches with a little noise, `w x h`, camera orientation `exif`: texture for the tests
/// that compare pictures.
fn textured(w: u32, h: u32, exif: u16) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Patchwork, 0.35).noise(2e-4, 1e-6);
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .orientation(exif)
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

/// A flat photo of `w x h`, camera orientation `exif`, whose file carries no lens-correction tables
/// (the synthetic files carry the camera's by default, and R9 then resamples the frame by a fraction
/// of a percent): the buffer is the mosaic, and a position in the photo is a position in the file.
fn bare(w: u32, h: u32, exif: u16) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.4);
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .orientation(exif)
        .lens_tables(ym_raw::testgen::LensTables {
            vignetting: None,
            chromatic_aberration: None,
            distortion: None,
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

/// A flat photo of `w x h`, camera orientation `exif`.
fn flat(w: u32, h: u32, exif: u16) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.4);
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .orientation(exif)
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn token() -> CancelToken {
    CancelToken::new()
}

fn geometry_of(r: &Renderer, src: &PhotoSource) -> Geometry {
    r.geometry(src, &token()).expect("geometry")
}

// ---------------------------------------------------------------------------------------------
// The map
// ---------------------------------------------------------------------------------------------

/// The warp the sampler reads is `ym_lens::LensWarp` bit for bit, for every lens and amount, on the
/// full-resolution canvas and on reduced ones.
#[test]
fn the_samplers_warp_is_the_lens_warp_bit_for_bit() {
    let lib = library();
    let camera = CameraResponse::neutral();
    let mut compared = 0usize;
    for kind in LensKind::ALL {
        for (edges, anamorphic) in [(0.0, 100.0), (100.0, 0.0), (100.0, 100.0), (40.0, 50.0)] {
            for (w, h) in [(6048usize, 4024usize), (4024, 6048), (640, 480)] {
                let amounts = LensAmounts::from(lens(kind, 0.0, edges, anamorphic));
                let frame = FrameInfo {
                    width: w,
                    height: h,
                    axis: Axis::X,
                    f_number: None,
                    clip: [1.0; 3],
                };
                let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
                let full = plan.warp(w as f64, h as f64);
                if full.is_identity() {
                    continue;
                }
                let sampler = sampler_warp(&full).expect("a bend");
                for (k, view) in [
                    (1usize, full.clone()),
                    (3, full.rescaled(w as f64 / 3.0, h as f64 / 3.0)),
                    (16, full.rescaled((w / 16) as f64, (h / 16) as f64)),
                ] {
                    let canvas = view.canvas_size();
                    for (fx, fy) in [
                        (0.0, 0.0),
                        (0.31, 0.77),
                        (0.5, 0.5),
                        (1.0, 1.0),
                        (0.93, 0.02),
                    ] {
                        let (x, y) = (fx * canvas.0, fy * canvas.1);
                        let a = sampler.canvas_to_photo(canvas, x, y);
                        let b = view.canvas_to_photo(x, y);
                        assert_eq!(
                            (a.0.to_bits(), a.1.to_bits()),
                            (b.0.to_bits(), b.1.to_bits()),
                            "{kind:?} e{edges} a{anamorphic} {w}x{h} /{k}: inverse at ({x}, {y})"
                        );
                        let a = sampler.photo_to_canvas(canvas, x, y);
                        let b = view.photo_to_canvas(x, y);
                        assert_eq!(
                            (a.0.to_bits(), a.1.to_bits()),
                            (b.0.to_bits(), b.1.to_bits()),
                            "{kind:?} e{edges} a{anamorphic} {w}x{h} /{k}: forward at ({x}, {y})"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 200, "{compared}");
}

/// The geometry of a photo carries the warp of its lens plan, made once on the full-resolution
/// canvas: the same numbers at every scale, the zoom of `LensWarp`.
#[test]
fn the_geometry_of_a_photo_carries_the_zoom_of_its_full_resolution_canvas() {
    let r = renderer();
    let d = flat(640, 480, 1);
    let src = with_lens(&d.source(), lens(LensKind::Default, 0.0, 0.0, 100.0));
    let g = geometry_of(&r, &src);
    let warp = g.warp.expect("Anamorphic 100 bends");
    assert!(warp.zoom > 1.0, "{warp:?}");
    assert!(!warp.axis_y);
    // The plan's own warp on that canvas.
    let frame = FrameInfo {
        width: 640,
        height: 480,
        axis: Axis::X,
        f_number: None,
        clip: [1.0; 3],
    };
    let plan = LensPlan::new(
        &library(),
        &LensAmounts::from(src.settings.lens),
        frame,
        &CameraResponse::neutral(),
    )
    .expect("plan");
    let own = plan.warp(640.0, 480.0);
    assert_eq!(warp.zoom, own.zoom());
    assert_eq!(warp.mu, own.mu());
    // A quarter turn swaps the canvas: the zoom is fitted on the 480 x 640 canvas, with the axis
    // along its horizontal.
    let t = geometry_of(&r, &turned(&src, 1));
    let tw = t.warp.expect("bends");
    let frame_y = FrameInfo {
        axis: Axis::Y,
        ..frame
    };
    let plan_y = LensPlan::new(
        &library(),
        &LensAmounts::from(src.settings.lens),
        frame_y,
        &CameraResponse::neutral(),
    )
    .expect("plan");
    assert_eq!(tw.zoom, plan_y.warp(480.0, 640.0).zoom());
    assert_ne!(tw.zoom, warp.zoom, "another canvas, another zoom");
    assert!(
        !tw.axis_y,
        "the canvas's horizontal is the axis in the baked geometry"
    );
}

// ---------------------------------------------------------------------------------------------
// Every pixel of the canvas has a source
// ---------------------------------------------------------------------------------------------

/// The sides of the reduced views, as `plan.out_size_at(k)` rounds them (and as the geometry tests do).
fn reduced(w: usize, h: usize, k: usize) -> (usize, usize) {
    if k == 1 {
        (w, h)
    } else {
        ((w + k / 2) / k, (h + k / 2) / k)
    }
}

/// No output pixel of the canvas lacks a source (`docs/21` section 14): for Anamorphic from 10 to 100,
/// Edges 0 and 100, the four quarter turns and three straighten angles, at full resolution and 3, 8
/// and 16 times smaller, every pixel the inside test accepts reads a position inside the photo, and
/// at the angle 0 that is every pixel. (Pixels the straightened canvas leaves empty are outside the
/// photo by design.)
#[test]
fn no_pixel_of_the_canvas_lacks_a_source_at_any_scale() {
    let r = renderer();
    // 1210 x 806: its reductions round (403 x 269, 151 x 101, 76 x 50).
    let (w, h) = (1210usize, 806usize);
    let d = flat(w as u32, h as u32, 1);
    let base = d.source();
    let mut closest = [f64::INFINITY; 4];
    let mut checked = 0usize;
    for (edges, amounts) in [
        (0.0, [10.0, 25.0, 50.0, 75.0, 100.0].as_slice()),
        (100.0, [0.0, 50.0, 100.0].as_slice()),
    ] {
        for &anamorphic in amounts {
            let src = with_lens(&base, lens(LensKind::Sonnar, 0.0, edges, anamorphic));
            for turns in 0..4u8 {
                let g0 = geometry_of(&r, &turned(&src, turns));
                let warp = g0.warp.expect("this lens bends");
                for angle in [0.0, 3.7, -11.2] {
                    let g = Geometry {
                        angle_deg: angle,
                        ..g0
                    };
                    let (sin, cos) = angle.to_radians().sin_cos();
                    for (si, k) in [1usize, 3, 8, 16].into_iter().enumerate() {
                        let buf = reduced(w, h, k);
                        let (cw, ch) = g.canvas_size(buf.0, buf.1);
                        let canvas = (cw as f64, ch as f64);
                        let g = if k == 1 { g.at_full_resolution() } else { g };
                        let grid = g.grid(buf, (cw, ch));
                        // A lattice at full resolution (a million pixels a case is slow), every
                        // pixel on the reduced canvases.
                        let step = if k == 1 { 7 } else { 1 };
                        let mut inside_by = f64::INFINITY;
                        for y in (0..ch).step_by(step).chain([ch - 1]) {
                            for x in (0..cw).step_by(step).chain([cw - 1]) {
                                // The canvas position of the output pixel, from the formula of the
                                // straighten map, independent of the grid.
                                let (qx, qy) = (
                                    x as f64 + 0.5 - canvas.0 * 0.5,
                                    y as f64 + 0.5 - canvas.1 * 0.5,
                                );
                                let px = canvas.0 * 0.5 + cos * qx + sin * qy;
                                let py = canvas.1 * 0.5 - sin * qx + cos * qy;
                                let on_canvas =
                                    px >= 0.0 && py >= 0.0 && px <= canvas.0 && py <= canvas.1;
                                let source = grid.source(x, y);
                                assert_eq!(
                                    source.is_some(),
                                    on_canvas,
                                    "e{edges} a{anamorphic} turns {turns} angle {angle} /{k}: pixel ({x}, {y})"
                                );
                                if !on_canvas {
                                    continue;
                                }
                                // Where the bent canvas reads, before the clamp.
                                let (sx, sy) = warp.canvas_to_photo(canvas, px, py);
                                inside_by = inside_by
                                    .min(sx)
                                    .min(canvas.0 - sx)
                                    .min(sy)
                                    .min(canvas.1 - sy);
                                checked += 1;
                                if angle == 0.0 {
                                    assert!(source.is_some());
                                }
                            }
                        }
                        // Full resolution: strictly inside, to the tolerance of the inside test.
                        // A reduced canvas has sides rounded to whole pixels (its half-diagonal
                        // is not the full canvas's over k), which moves the edge by a fraction of
                        // a reduced pixel at most; the sampler clamps what is left.
                        let allowed = if k == 1 { -1e-6 } else { -0.5 };
                        assert!(
                            inside_by >= allowed,
                            "e{edges} a{anamorphic} turns {turns} angle {angle} /{k}: reads {} px outside the photo",
                            -inside_by
                        );
                        closest[si] = closest[si].min(inside_by);
                    }
                }
            }
        }
    }
    eprintln!(
        "lens bend: {checked} pixels read; the closest any comes to the photo's edge (px of the view; negative is outside): full {:.3}, /3 {:.3}, /8 {:.3}, /16 {:.3}",
        closest[0], closest[1], closest[2], closest[3]
    );
    assert!(checked > 1_000_000);
}

// ---------------------------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------------------------

/// A 640 x 480 photo with dark lines 4 px wide: vertical at x = 160, 320 and 480, horizontal at
/// y = 60, 240 and 420 (continuous positions of their centres).
fn lines() -> Developed {
    let (w, h) = (640u32, 480u32);
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.5);
    let mut mosaic = scene.mosaic();
    let dark = (scene.black + 0.04 * (scene.white_level - scene.black)) as u16;
    for y in 0..h {
        for x in 0..w {
            let on_vertical = [160u32, 320, 480]
                .iter()
                .any(|&c| x + 2 >= c && x + 2 < c + 4);
            let on_horizontal = [60u32, 240, 420]
                .iter()
                .any(|&c| y + 2 >= c && y + 2 < c + 4);
            if on_vertical || on_horizontal {
                mosaic[(y * w + x) as usize] = dark;
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn export(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Viewport),
        },
        &token(),
    )
    .expect("full render")
}

/// The dark line's centre across a row (`along_x`) or a column of the picture's green channel,
/// within `half` px of `around`: the centroid of how far the samples fall under the level on both
/// sides of the window. `None` when there is no line.
fn centroid(img: &FullImage, line: usize, along_x: bool, around: f64, half: usize) -> Option<f64> {
    let len = if along_x { img.width } else { img.height };
    let lo = (around.round() as isize - half as isize).max(0) as usize;
    let hi = ((around.round() as isize + half as isize) as usize).min(len - 1);
    if hi < lo + 8 {
        return None;
    }
    let g = |i: usize| {
        let (x, y) = if along_x { (i, line) } else { (line, i) };
        f64::from(img.rgb[(y * img.width + x) * 3 + 1])
    };
    let level = (g(lo) + g(lo + 1) + g(hi) + g(hi - 1)) / 4.0;
    if level <= 0.0 {
        return None;
    }
    let (mut sum, mut moment) = (0.0, 0.0);
    for i in lo..=hi {
        let depth = (level - g(i)).max(0.0);
        sum += depth;
        moment += depth * (i as f64 + 0.5);
    }
    (sum > 0.05 * level).then(|| moment / sum)
}

/// True when the photo column `x` is clear of the vertical lines (a column through one is dark all
/// the way down, and says nothing about a horizontal line).
fn away_from_verticals(x: f64) -> bool {
    [160.0, 320.0, 480.0].iter().all(|&c| (x - c).abs() > 16.0)
}

/// The extremes of a set of numbers.
fn span(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        })
}

/// In the finished picture of a photo of lines, at Anamorphic 100 and Edges 0, a vertical line is
/// straight to 0.5 px, at the place the map puts it, and a horizontal line near the top bows as the
/// map says (and by a few pixels, so the check means something).
///
/// The map's input is the photo as the lens stage gets it, not the mosaic: the raw stages move the
/// lines by a fraction of a pixel (R9 resamples the frame), so the lines are first measured in the
/// picture of the same photo without a lens, and those are the places the map is applied to.
#[test]
fn a_horizontal_line_near_the_top_bows_and_a_vertical_stays_straight() {
    let r = renderer();
    let d = lines();
    let src = with_lens(&d.source(), lens(LensKind::Default, 0.0, 0.0, 100.0));
    let g = geometry_of(&r, &src);
    let warp = g.warp.expect("Anamorphic 100 bends");
    let canvas = (640.0, 480.0);
    let img = export(&r, &src);
    let base = export(&r, &with_lens(&d.source(), LensSettings::default()));
    assert_eq!((img.width, img.height), (640, 480));
    assert_eq!((base.width, base.height), (640, 480));

    // The verticals: x = 160, 320, 480 in the photo, away from the horizontal lines' rows.
    let rows: Vec<usize> = (30..450)
        .step_by(10)
        .filter(|&row| {
            [60.0, 240.0, 420.0]
                .iter()
                .all(|&c| (row as f64 - c).abs() >= 40.0)
        })
        .collect();
    for line_x in [160.0, 320.0, 480.0] {
        // Where the raw stages leave the line: the mean over the rows.
        let in_photo = rows
            .iter()
            .map(|&row| centroid(&base, row, true, line_x, 10).expect("the line, no lens"))
            .sum::<f64>()
            / rows.len() as f64;
        let want = warp.photo_to_canvas(canvas, in_photo, 240.0).0;
        // The map keeps a vertical line vertical: the canvas x does not depend on the photo's y.
        for y in [0.0, 120.0, 479.0] {
            assert!((warp.photo_to_canvas(canvas, in_photo, y).0 - want).abs() < 1e-9);
        }
        let found: Vec<f64> = rows
            .iter()
            .map(|&row| centroid(&img, row, true, want, 10).expect("the line is in the picture"))
            .collect();
        let (lo, hi) = span(&found);
        assert!(
            hi - lo <= 0.5,
            "the vertical line at {line_x} is not straight: {lo:.3} to {hi:.3}"
        );
        let mean = found.iter().sum::<f64>() / found.len() as f64;
        assert!(
            (mean - want).abs() <= 0.5,
            "the vertical line at {line_x} is at {mean:.3}, the map says {want:.3} (it was at {in_photo:.3} without the lens)"
        );
        eprintln!(
            "lens bend: vertical {line_x}: {in_photo:.3} -> {mean:.3} (map {want:.3}), spread {:.3}",
            hi - lo
        );
    }

    // A horizontal line along its length: the bent line against the map of the unbent one, which is
    // measured column by column in the picture without a lens.
    let along = |photo_y: f64, step: usize| -> (Vec<f64>, Vec<f64>) {
        let (mut expected, mut measured) = (Vec::new(), Vec::new());
        for s in (60..=580u32)
            .step_by(step)
            .filter(|&s| away_from_verticals(f64::from(s)))
        {
            let y_in_photo =
                centroid(&base, s as usize, false, photo_y, 12).expect("the line, no lens");
            let (cx, cy) = warp.photo_to_canvas(canvas, f64::from(s) + 0.5, y_in_photo);
            // The column the bent line crosses at that place (its slope is below 0.1).
            let y = centroid(&img, cx.floor() as usize, false, cy, 12)
                .expect("the line is in the picture");
            assert!(
                (y - cy).abs() <= 0.5,
                "the horizontal line of the photo at x = {s}, y = {photo_y}: it is at {y:.3}, the map says {cy:.3}"
            );
            expected.push(cy);
            measured.push(y);
        }
        (expected, measured)
    };
    let (_, top) = along(60.0, 20);
    let (lo, hi) = span(&top);
    eprintln!(
        "lens bend: horizontal line near the top spans {:.2} px",
        hi - lo
    );
    assert!(
        hi - lo >= 3.0,
        "the horizontal line near the top must bow, it spans {:.2} px",
        hi - lo
    );
    // The line through the middle stays straight.
    let (_, middle) = along(240.0, 40);
    let (lo, hi) = span(&middle);
    assert!(
        hi - lo <= 0.5,
        "the middle line is not straight: {lo:.3} to {hi:.3}"
    );
}

// ---------------------------------------------------------------------------------------------
// Pick
// ---------------------------------------------------------------------------------------------

fn cell(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

/// A click goes through the bend to a point of the oriented photo, and the forward map takes that point
/// back to the click: for the camera's orientations that swap the sides, a user's turn, and a lens that
/// bends along either axis.
#[test]
fn a_click_goes_through_the_bend_and_back() {
    let r = renderer();
    let mut worst = 0.0f64;
    for (exif, user_turns) in [(1u16, 0u8), (6, 0), (1, 1), (8, 1), (3, 2)] {
        let d = bare(600, 400, exif);
        let src = turned(
            &with_lens(&d.source(), lens(LensKind::Default, 0.0, 100.0, 100.0)),
            user_turns,
        );
        let g = geometry_of(&r, &src);
        let warp = g.warp.expect("bends");
        let (cw, ch) = g.canvas_size(600, 400);
        // A cell half the canvas' size: css px are canvas px over 2.
        let st = cell(cw as f64 / 2.0, ch as f64 / 2.0);
        for (u, v) in [
            (0.5, 0.5),
            (0.1, 0.12),
            (0.93, 0.88),
            (0.25, 0.8),
            (0.02, 0.5),
        ] {
            let css = (u * cw as f64 / 2.0, v * ch as f64 / 2.0);
            // Where the click is on the final image, which is the canvas (no crop, no angle).
            let (fx, fy) = st
                .final_point_at((cw, ch), css)
                .expect("the click is on the picture");
            let p = r
                .pick_point_at(&src, &st, css, &token())
                .expect("pick")
                .expect("inside");
            // The oriented photo's fractions back to the canvas, then forward through the bend.
            let c = oriented_to_canvas(user_turns, p);
            let (x, y) =
                warp.photo_to_canvas((cw as f64, ch as f64), c[0] * cw as f64, c[1] * ch as f64);
            let err = (x - fx).abs().max((y - fy).abs());
            worst = worst.max(err);
            assert!(
                err < 1e-6,
                "exif {exif} turns {user_turns}: click {css:?} = canvas ({fx:.4}, {fy:.4}) -> photo {p:?} -> canvas ({x:.4}, {y:.4}) of {cw}x{ch}"
            );
            // And the pick is not the click: the bend moved it (at the sides and corners).
            if (u - 0.5).abs() > 0.3 {
                let straight = canvas_to_oriented(user_turns, [fx / cw as f64, fy / ch as f64]);
                assert!(
                    (straight[0] - p[0]).abs() + (straight[1] - p[1]).abs() > 1e-3,
                    "exif {exif}: the pick ignored the bend"
                );
            }
        }
    }
    eprintln!("lens bend: pick round trip, worst error {worst:.2e} px");
}

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

/// A lens folder like the shipped one except that the Default lens has no distortion.
fn undistorted_library() -> (tempfile::TempDir, Arc<LensLibrary>) {
    let dir = tempfile::tempdir().expect("temp dir");
    let lenses = dir.path().join("lenses");
    ym_lens::test_support::write_synthetic_lenses(&lenses).expect("synthetic lenses");
    let path = lenses.join("default").join("lens.json");
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
    let distortion = json
        .pointer_mut("/field/distortion")
        .expect("a distortion block");
    *distortion = serde_json::json!({ "k1": 0.0, "k2": 0.0 });
    std::fs::write(&path, serde_json::to_vec_pretty(&json).expect("json")).expect("write");
    ym_lens::test_support::write_sums(&lenses).expect("sums");
    let library = Arc::new(LensLibrary::from_dir(&lenses).expect("the library loads"));
    (dir, library)
}

/// Anamorphic 0 on a lens without distortion is no bend: the geometry is the plain one, a
/// full-resolution grid is still a pixel copy, and the sampler's kernel is the copy. Edges on a lens
/// with distortion, or any Anamorphic, bends; the Before view never does.
#[test]
fn without_anamorphic_and_distortion_the_geometry_is_the_plain_one() {
    let (_dir, flat_lens) = undistorted_library();
    let d = flat(640, 480, 1);
    let plain = Geometry::new(1, &d.source().settings);
    let r =
        Renderer::new(Arc::new(NeutralOnly), options_with(flat_lens, 0)).expect("render thread");
    for l in [
        lens(LensKind::Default, 60.0, 100.0, 0.0),
        lens(LensKind::Default, 0.0, 0.0, 0.0),
        lens(LensKind::Default, 100.0, 0.0, 0.0),
    ] {
        let src = with_lens(&d.source(), l);
        let g = geometry_of(&r, &src);
        assert_eq!(g, plain, "{l:?}");
        assert_eq!(g.warp, None);
        let grid = g.at_full_resolution().grid((640, 480), (640, 480));
        assert!(grid.is_pixel_copy());
        assert_eq!(grid.sampler_grid().kernel(), Kernel::Copy);
    }
    // Anamorphic bends, even on that lens.
    let bent = geometry_of(
        &r,
        &with_lens(&d.source(), lens(LensKind::Default, 0.0, 0.0, 30.0)),
    );
    assert!(bent.warp.is_some());
    let grid = bent.at_full_resolution().grid((640, 480), (640, 480));
    assert!(!grid.is_pixel_copy());
    assert_eq!(
        grid.sampler_grid().kernel(),
        Kernel::Bicubic,
        "Catmull-Rom at full resolution"
    );

    // The shipped lens has distortion: Edges alone bends it, Flares alone does not.
    let r = renderer();
    let e = geometry_of(
        &r,
        &with_lens(&d.source(), lens(LensKind::Sonnar, 0.0, 100.0, 0.0)),
    );
    assert!(e.warp.is_some(), "Edges bends a lens with distortion");
    let f = geometry_of(
        &r,
        &with_lens(&d.source(), lens(LensKind::Sonnar, 100.0, 0.0, 0.0)),
    );
    assert_eq!(f.warp, None, "Flares alone do not");

    // The Before view has no lens, so no bend, whatever the settings say.
    let src = with_lens(&d.source(), lens(LensKind::Sonnar, 60.0, 100.0, 100.0));
    assert!(geometry_of(&r, &src).warp.is_some());
    assert_eq!(geometry_of(&r, &src.for_before_view()).warp, None);
}

/// At full resolution the bend never costs a box filter: the largest shrink of any lens at the largest
/// amounts stays under Catmull-Rom's 1.25 (`docs/21` section 4, L4: "Catmull-Rom everywhere").
#[test]
fn at_full_resolution_every_lens_is_catmull_rom() {
    let r = renderer();
    let d = flat(640, 480, 1);
    for kind in LensKind::ALL {
        for turns in [0u8, 1] {
            let src = turned(
                &with_lens(&d.source(), lens(kind, 0.0, 100.0, 100.0)),
                turns,
            );
            let g = geometry_of(&r, &src);
            let (cw, ch) = g.canvas_size(640, 480);
            let grid = g.at_full_resolution().grid((640, 480), (cw, ch));
            assert_ne!(grid.sampler_grid().kernel(), Kernel::Copy, "{kind:?}");
            assert_eq!(
                grid.sampler_grid().kernel(),
                Kernel::Bicubic,
                "{kind:?} turns {turns}: {:?}",
                g.warp
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The DNG and the tiles
// ---------------------------------------------------------------------------------------------

/// A DNG's preview is the baked picture in the buffer's own orientation: with a bend that turns the
/// canvas (a camera orientation that swaps the sides, a user's turn) and a straighten angle, the
/// upright render holds the baked render's pixels, turned back.
#[test]
fn the_dng_preview_is_the_baked_picture_unturned() {
    let r = renderer();
    // 320 x 240 under camera orientation 6 (the picture is 240 x 320), and a user's quarter turn.
    for (exif, user_turns, angle) in [
        (1u16, 0u8, 0.0),
        (6, 0, 0.0),
        (1, 1, 0.0),
        (6, 1, 4.0),
        (8, 0, -3.0),
    ] {
        let d = textured(320, 240, exif);
        // No capture sharpening on either side: a DNG's renders are made without R8.
        let mut src = edit(&d.source(), |s| {
            s.sharpen = ym_core::settings::AutoOr::Manual(ym_core::settings::SharpenValue {
                amount: 0.0,
                radius_px: 1.0,
            });
        });
        src = turned(
            &with_lens(&src, lens(LensKind::Default, 0.0, 100.0, 100.0)),
            user_turns,
        );
        if angle != 0.0 {
            src = edit(&src, |s| {
                s.crop = Some(ym_core::settings::Crop {
                    rect: ym_core::settings::NormRect::FULL,
                    angle_deg: angle,
                    aspect: ym_core::settings::CropAspect::Free,
                    auto: false,
                });
            });
        }
        let g = geometry_of(&r, &src);
        assert!(g.warp.is_some());
        let baked = r
            .render_full(
                &src,
                &FullRequest {
                    target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                },
                &token(),
            )
            .expect("baked");
        let upright = r
            .render_full(
                &src,
                &FullRequest {
                    target: FullTarget::DngPreview,
                },
                &token(),
            )
            .expect("preview");
        let (uw, uh) = (upright.width, upright.height);
        let swapped = (5..=8).contains(&exif) != (user_turns % 2 == 1);
        assert_eq!(
            (baked.width, baked.height),
            if swapped { (uh, uw) } else { (uw, uh) }
        );
        // The upright pixel (x, y) of the camera's image lands, after the camera's orientation and
        // the user's turns, on the baked pixel the independent forward maps say.
        let mut worst = 0.0f32;
        let mut compared = 0usize;
        for y in (0..uh).step_by(3) {
            for x in (0..uw).step_by(3) {
                let (mut px, mut py) = match exif {
                    2 => (uw - 1 - x, y),
                    3 => (uw - 1 - x, uh - 1 - y),
                    4 => (x, uh - 1 - y),
                    5 => (y, x),
                    6 => (uh - 1 - y, x),
                    7 => (uh - 1 - y, uw - 1 - x),
                    8 => (y, uw - 1 - x),
                    _ => (x, y),
                };
                let (mut ow, mut oh) = if (5..=8).contains(&exif) {
                    (uh, uw)
                } else {
                    (uw, uh)
                };
                for _ in 0..user_turns {
                    (px, py) = (oh - 1 - py, px);
                    std::mem::swap(&mut ow, &mut oh);
                }
                for c in 0..3 {
                    let a = upright.rgb[(y * uw + x) * 3 + c];
                    let b = baked.rgb[(py * baked.width + px) * 3 + c];
                    worst = worst.max((a - b).abs());
                }
                compared += 1;
            }
        }
        eprintln!(
            "lens bend: DNG preview vs baked, exif {exif} turns {user_turns} angle {angle}: {compared} pixels, worst {worst:.2e}"
        );
        assert!(
            worst <= 2e-4,
            "exif {exif} turns {user_turns} angle {angle}: the upright picture differs from the baked one by {worst}"
        );
    }
}

/// An export in small tiles is the export in one: a tile's source rectangle follows the bent edge of the
/// tile, so the sampler never reads a pixel that was not cut out for it.
#[test]
fn an_exports_tiles_equal_the_whole_frame_with_a_bend() {
    let d = textured(400, 300, 1);
    let l = lens(LensKind::Default, 0.0, 100.0, 100.0);
    let whole =
        Renderer::new(Arc::new(NeutralOnly), options_with(library(), 1024)).expect("render thread");
    let a = export(&whole, &with_lens(&d.source(), l));
    for tile in [64usize, 96, 160] {
        let tiled = Renderer::new(Arc::new(NeutralOnly), options_with(library(), tile))
            .expect("render thread");
        let b = export(&tiled, &with_lens(&d.source(), l));
        assert_eq!((a.width, a.height), (b.width, b.height));
        let differing = a
            .rgb
            .iter()
            .zip(&b.rgb)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "tile {tile}: {differing} samples differ from the whole frame"
        );
    }
    // The bend did something.
    let plain = export(&whole, &d.source());
    let moved = a
        .rgb
        .iter()
        .zip(&plain.rgb)
        .filter(|(x, y)| (*x - *y).abs() > 1e-3)
        .count();
    assert!(moved > a.rgb.len() / 10, "{moved}");
}
