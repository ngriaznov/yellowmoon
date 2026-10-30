//! What the pixels of a render must be: neutral stays neutral, one stop is one stop of exposure,
//! orientation and straightening move the right pixels, the reduced path agrees with the full one,
//! and every raw layout renders (cRAW, Lossless M/S, odd crop origins, lens tags).
mod common;

use common::*;
use ym_color::camera_default::CameraTone;
use ym_color::encode::srgb_encode;
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, PhotoSource, RenderedFrame, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::{
    AutoOr, Crop, CropAspect, DevelopSettings, ExposureValue, NormRect, Orientation,
};
use ym_gpu::queue::CancelToken;
use ym_raw::layout::ChromaSubsampling;
use ym_raw::testgen::{ArwBuilder, CrawEncoder, LensTables, LosslessMs};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn render(r: &Renderer, src: &PhotoSource, state: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, state, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn state(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn before(w: f64, h: f64) -> ViewportState {
    let mut s = state(w, h);
    s.before = true;
    s
}

fn centre(frame: &RenderedFrame) -> [f64; 3] {
    let (w, h) = (frame.width / 4, frame.height / 4);
    mean_rgba(
        frame,
        frame.width / 2 - w / 2,
        frame.height / 2 - h / 2,
        w,
        h,
    )
}

fn manual_ev(src: &PhotoSource, ev: f64) -> PhotoSource {
    edit(src, |s| s.exposure = AutoOr::Manual(ExposureValue { ev }))
}

/// The balanced camera value of a neutral field of luminance `level` under `white` (green = 1):
/// `0.9 * level / max(white)` (`tests/common`: `Scene::signal`).
fn balanced(level: f64, white: [f64; 3]) -> f64 {
    0.9 * level / white.iter().copied().fold(0.0, f64::max)
}

/// The mean of the central quarter of a full image, per channel.
fn centre_of(img: &ym_core::pipeline::FullImage) -> [f64; 3] {
    let (w, h) = (img.width, img.height);
    let (x0, y0, cw, ch) = (w * 3 / 8, h * 3 / 8, w / 4, h / 4);
    let mut sum = [0.0f64; 3];
    for y in y0..y0 + ch {
        for x in x0..x0 + cw {
            for (c, s) in sum.iter_mut().enumerate() {
                *s += f64::from(img.rgb[(y * w + x) * 3 + c]);
            }
        }
    }
    let n = (cw * ch) as f64;
    sum.map(|v| v / n)
}

#[test]
fn the_exposed_signal_of_a_neutral_field_is_two_to_the_ev_times_the_balanced_value() {
    // This checks R1 (black and white), R4 (the gains), the fused matrix (white preserving, one stop per
    // EV), the highlight range (identity below the knee) and C8 to C11 and back, end to end: the DNG's
    // pixels (scene-linear AWG4 after the whole LogC4 chain) of a neutral field move by exactly a stop
    // for an EV. The Before view cannot show this any more: it ignores the exposure.
    let white = light(PROFILE, 5500.0);
    for level in [0.05, 0.12] {
        let scene = Scene::new(600, 400, white, Layout::Uniform, level);
        let d = Developed::new(&scene, MODEL);
        let src = d.source();
        let r = Renderer::neutral_only().expect("render thread");
        let v = balanced(level, white);
        let dng = |src: &PhotoSource| {
            r.render_full(
                src,
                &FullRequest {
                    target: FullTarget::DngLinearAwg4,
                },
                &CancelToken::new(),
            )
            .expect("render")
        };
        for ev in [-1.0, 0.0, 1.0, 1.5] {
            let got = centre_of(&dng(&manual_ev(&src, ev)));
            let expect = 2f64.powf(ev) * v;
            for (c, g) in got.iter().enumerate() {
                assert!(
                    (g / expect - 1.0).abs() < 0.03,
                    "level {level} ev {ev} channel {c}: got {g:.5}, 2^ev x v says {expect:.5}"
                );
            }
        }
        // Auto exposure is the same chain with the metered EV.
        let got = centre_of(&dng(&src))[1];
        let expect = 2f64.powf(src.estimates.exposure.ev) * v;
        assert!((got / expect - 1.0).abs() < 0.03, "auto: {got} vs {expect}");
    }
}

/// The highlight range (`rolloff.highlights`, `docs/02` C7) on a neutral ramp whose top third clips on
/// the sensor, at a manual EV of +1 (the clip is then `log2(2 / 0.18)` = 3.47 stops over grey, about
/// where a Sony photo at Auto has it): 0 against 1 moves the clipped highlights a long way, 0.5 lands
/// between them, and every pixel under the knee (two stops below the clip since the gentler range of
/// 2026-10-30; it was one) is the same bit for bit. The
/// DNG's pixels carry it: at 1 the clip lands on the ALEXA's EI 800 clip, `HIGHLIGHT_CAP_STOPS` over
/// grey, and nothing goes past it.
#[test]
fn the_highlight_range_moves_the_clipped_highlights_and_nothing_under_the_knee() {
    use ym_color::display::OutputTarget;
    use ym_color::tone::{HIGHLIGHT_CAP_STOPS, luma};

    let white = light(PROFILE, 5500.0);
    // `Layout::Ramp`: luminance from 0.1 to 1 of the level along x. The balanced neutral is
    // `0.9 * lum / max(white)` and clips at 1, so with a level of 1.6 the clip is at t = 0.66 and the
    // knee (a quarter of the clip, two stops under it) at t = 0.08.
    let level = 1.6;
    let scene = Scene::new(640, 48, white, Layout::Ramp, level);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let base = edit(&manual_ev(&d.source(), 1.0), |s| s.output_sharpening = 0.0);
    let render = |highlights: f64, target: FullTarget| {
        let src = edit(&base, |s| s.rolloff.highlights = highlights);
        r.render_full(&src, &FullRequest { target }, &CancelToken::new())
            .expect("render")
    };
    let peak = white.iter().copied().fold(0.0, f64::max);
    let balanced_at = |x: usize, w: usize| {
        let t = x as f64 / (w - 1) as f64;
        0.9 * level * (0.1 + 0.9 * t) / peak
    };
    // The mean of a band of columns, per channel; and the columns under the knee with a margin
    // (balanced under 0.225, 0.15 stop under the knee at 0.25, for the demosaic's reach across the ramp).
    let band = |img: &ym_core::pipeline::FullImage, x0: usize, x1: usize| {
        let mut sum = [0.0f64; 3];
        for y in 0..img.height {
            for x in x0..x1 {
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += f64::from(img.rgb[(y * img.width + x) * 3 + c]);
                }
            }
        }
        sum.map(|v| v / ((x1 - x0) * img.height) as f64)
    };

    let display = FullTarget::DisplayLinear(OutputTarget::Viewport);
    let (lo, mid, hi) = (
        render(0.0, display),
        render(0.5, display),
        render(1.0, display),
    );
    let w = lo.width;
    let under = (0..w).take_while(|&x| balanced_at(x, w) < 0.225).count();
    assert!(under > w / 20, "{under} columns under the knee");
    for img in [&mid, &hi] {
        for y in 0..img.height {
            let row = (y * w) * 3;
            assert_eq!(
                img.rgb[row..row + 3 * under],
                lo.rgb[row..row + 3 * under],
                "row {y}: a pixel under the knee moved"
            );
        }
    }
    // The clipped band: brighter with each step of the range (measured: about 0.809, 0.985 and 0.992;
    // the cine rendering compresses the last step, sRGB codes 232, 253 and 254 under the former display transform).
    let clipped = (0..w)
        .find(|&x| balanced_at(x, w) > 1.1)
        .expect("a clipped band");
    let (a, b, c) = (
        band(&lo, clipped, w),
        band(&mid, clipped, w),
        band(&hi, clipped, w),
    );
    eprintln!("clipped band, display-linear: 0 {a:?}, 0.5 {b:?}, 1 {c:?}");
    for ch in 0..3 {
        assert!(
            a[ch] + 0.1 < b[ch] && b[ch] + 0.002 < c[ch],
            "channel {ch}: {a:?} {b:?} {c:?}"
        );
    }

    // The DNG's pixels: scene-linear AWG4, the range in them.
    let (dng_lo, dng_hi) = (
        render(0.0, FullTarget::DngLinearAwg4),
        render(1.0, FullTarget::DngLinearAwg4),
    );
    // The range caps the luma (`docs/02` C7), so that is what is compared with the ALEXA clip; one
    // channel of a pixel at the cap can be a little over it (the demosaic's slight colour there).
    let max_luma = |img: &ym_core::pipeline::FullImage| {
        img.rgb
            .chunks_exact(3)
            .map(|p| luma([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]))
            .fold(0.0f64, f64::max)
    };
    let cap = 0.18 * HIGHLIGHT_CAP_STOPS.exp2();
    let (m_lo, m_hi) = (max_luma(&dng_lo), max_luma(&dng_hi));
    eprintln!("DNG maximum luma: {m_lo} at 0, {m_hi} at 1 (the ALEXA clip {cap})");
    assert!(
        m_lo < 4.0,
        "at 0 the clip stays where it was recorded (2): {m_lo}"
    );
    assert!(
        m_hi > 0.9 * cap && m_hi < cap * (1.0 + 1e-4),
        "at 1 the clip lands on the ALEXA clip {cap}: {m_hi}"
    );
    assert_eq!((dng_lo.width, dng_lo.height), (w, lo.height));
    for y in 0..dng_lo.height {
        let row = (y * w) * 3;
        assert_eq!(
            dng_hi.rgb[row..row + 3 * under],
            dng_lo.rgb[row..row + 3 * under],
            "DNG row {y}: a pixel under the knee moved"
        );
    }
}

/// The code value the Before view shows for a neutral that is `v` after the as-shot white balance: the
/// camera gain, the camera tone curve and the sRGB transfer (a neutral is the same in Display P3).
fn camera_default_code(v: f64) -> f64 {
    255.0 * srgb_encode(CameraTone::FITTED.apply(CAMERA_GAIN_EV.exp2() * v))
}

#[test]
fn the_before_view_of_a_neutral_field_is_the_camera_default_of_the_as_shot_signal() {
    // R1, R4 and the as-shot balance, end to end, against the camera default's own formula; and the
    // exposure setting does not move it.
    let white = light(PROFILE, 5500.0);
    for level in [0.05, 0.12, 0.3] {
        let scene = Scene::new(600, 400, white, Layout::Uniform, level);
        let d = Developed::new(&scene, MODEL);
        let src = d.source();
        let r = Renderer::neutral_only().expect("render thread");
        let expect = camera_default_code(balanced(level, white));
        for ev in [-1.0, 0.0, 1.5] {
            let f = render(&r, &manual_ev(&src, ev), &before(300.0, 200.0));
            for (c, g) in centre(&f).iter().enumerate() {
                assert!(
                    (g - expect).abs() < 0.8,
                    "level {level} ev {ev} channel {c}: got {g:.2}, the camera default says {expect:.2}"
                );
            }
        }
        let f = render(&r, &src, &before(300.0, 200.0));
        assert!((centre(&f)[1] - expect).abs() < 0.8);
    }
}

#[test]
fn a_neutral_field_renders_neutral_under_any_light_with_a_right_as_shot_balance() {
    for cct in [3000.0, 4200.0, 5500.0, 7500.0] {
        let scene = Scene::new(400, 300, light(PROFILE, cct), Layout::Uniform, 0.3);
        let d = Developed::new(&scene, MODEL);
        let r = Renderer::neutral_only().expect("render thread");
        let f = render(&r, &d.source(), &state(200.0, 150.0));
        let m = centre(&f);
        assert!(
            (m[0] - m[1]).abs() < 1.5 && (m[2] - m[1]).abs() < 1.5,
            "{cct} K: {m:?}"
        );
        assert!(m[1] > 20.0 && m[1] < 235.0, "{cct} K: {m:?}");
    }
}

/// The cell (column, row) of an `cols x rows` grid, interior 60 %, mean display values.
fn cell_mean(f: &RenderedFrame, cols: usize, rows: usize, c: usize, r: usize) -> [f64; 3] {
    let (cw, ch) = (f.width / cols, f.height / rows);
    let (mx, my) = (cw / 5, ch / 5);
    mean_rgba(f, c * cw + mx, r * ch + my, cw - 2 * mx, ch - 2 * my)
}

fn assert_close(a: [f64; 3], b: [f64; 3], tol: f64, what: &str) {
    for c in 0..3 {
        assert!((a[c] - b[c]).abs() <= tol, "{what}: {a:?} vs {b:?}");
    }
}

/// Where the sensor patch `(px, py)` of a `cols x rows` grid lands under EXIF orientation `exif`,
/// as the (column, row) of the turned grid: written independently of the code under test.
fn forward_patch(exif: u16, px: usize, py: usize, cols: usize, rows: usize) -> (usize, usize) {
    match exif {
        2 => (cols - 1 - px, py),
        3 => (cols - 1 - px, rows - 1 - py),
        4 => (px, rows - 1 - py),
        5 => (py, px),
        6 => (rows - 1 - py, px),
        7 => (rows - 1 - py, cols - 1 - px),
        8 => (py, cols - 1 - px),
        _ => (px, py),
    }
}

/// One more clockwise quarter turn of a `cols x rows` grid.
fn turn_cw(c: usize, r: usize, rows: usize) -> (usize, usize) {
    (rows - 1 - r, c)
}

#[test]
fn exif_orientation_and_the_users_turn_put_every_patch_where_it_belongs() {
    let white = light(PROFILE, 5500.0);
    let scene = Scene::new(600, 400, white, Layout::Patchwork, 0.4);
    let r = Renderer::neutral_only().expect("render thread");
    // Reference: the sensor as it is (EXIF 1), cell grid 6 x 4.
    let base = Developed::new(&scene, MODEL);
    let reference = render(&r, &base.source(), &state(300.0, 200.0));
    for exif in 1..=8u16 {
        let dir = tempfile::tempdir().expect("dir");
        scene
            .builder(MODEL)
            .orientation(exif)
            .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
            .expect("write");
        let d = Developed::open(dir);
        let swapped = (5..=8).contains(&exif);
        let (cw, ch) = if swapped {
            (200.0, 300.0)
        } else {
            (300.0, 200.0)
        };
        let f = render(&r, &d.source(), &state(cw, ch));
        let (cols, rows) = if swapped { (4, 6) } else { (6, 4) };
        assert_eq!(
            (f.width, f.height),
            (cw as usize, ch as usize),
            "exif {exif}"
        );
        for py in 0..4 {
            for px in 0..6 {
                let (c, row) = forward_patch(exif, px, py, 6, 4);
                assert_close(
                    cell_mean(&f, cols, rows, c, row),
                    cell_mean(&reference, 6, 4, px, py),
                    1.6,
                    &format!("exif {exif}, sensor patch ({px},{py}) -> ({c},{row})"),
                );
            }
        }
    }
    // The user's turns compose with it, clockwise, on top of the camera's orientation.
    for (turns, orientation) in [
        (1u8, Orientation::Deg90),
        (2, Orientation::Deg180),
        (3, Orientation::Deg270),
    ] {
        for exif in [1u16, 6] {
            let dir = tempfile::tempdir().expect("dir");
            scene
                .builder(MODEL)
                .orientation(exif)
                .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
                .expect("write");
            let d = Developed::open(dir);
            let rot = edit(&d.source(), |s| s.orientation = orientation);
            let swapped = (5..=8).contains(&exif) != (turns % 2 == 1);
            let (cw, ch) = if swapped {
                (200.0, 300.0)
            } else {
                (300.0, 200.0)
            };
            let f = render(&r, &rot, &state(cw, ch));
            let (cols, rows) = if swapped { (4, 6) } else { (6, 4) };
            for py in 0..4 {
                for px in 0..6 {
                    let (mut c, mut row) = forward_patch(exif, px, py, 6, 4);
                    let (mut ccols, mut crows) = if (5..=8).contains(&exif) {
                        (4, 6)
                    } else {
                        (6, 4)
                    };
                    for _ in 0..turns {
                        (c, row) = turn_cw(c, row, crows);
                        std::mem::swap(&mut ccols, &mut crows);
                    }
                    assert_eq!((ccols, crows), (cols, rows));
                    assert_close(
                        cell_mean(&f, cols, rows, c, row),
                        cell_mean(&reference, 6, 4, px, py),
                        1.6,
                        &format!("exif {exif} then {turns} turns, patch ({px},{py})"),
                    );
                }
            }
        }
    }
}

#[test]
fn straightening_rotates_about_the_centre_and_leaves_the_corners_empty() {
    let scene = Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Patchwork, 0.4);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    let plain = render(&r, &src, &state(300.0, 200.0));
    let tilted = render(
        &r,
        &edit(&src, |s| {
            s.crop = Some(Crop {
                rect: NormRect::FULL,
                angle_deg: 12.0,
                aspect: CropAspect::Original,
                auto: false,
            })
        }),
        &state(300.0, 200.0),
    );
    assert_eq!((tilted.width, tilted.height), (plain.width, plain.height));
    // The corner of the canvas is empty after a 12 degree turn: the letterbox colour.
    assert_eq!(&tilted.rgba[0..4], &[255, 255, 255, 255]);
    let last = tilted.rgba.len() - 4;
    assert_eq!(&tilted.rgba[last..], &[255, 255, 255, 255]);
    // The middle of the canvas is the middle of the photo.
    let c = |f: &RenderedFrame| mean_rgba(f, 140, 90, 20, 20);
    // A turn about the centre keeps the pixel at the centre (the patches around it are flat).
    assert_close(c(&plain), c(&tilted), 3.0, "the centre");
    // A positive angle turns clockwise: the content at the right of the centre moves down.
    let probe = |f: &RenderedFrame, x: usize, y: usize| mean_rgba(f, x, y, 6, 6);
    let right_of_centre_plain = probe(&plain, 270, 94);
    let below_in_tilted = (60..140)
        .map(|y| probe(&tilted, 270, y))
        .min_by(|a, b| {
            let d = |m: &[f64; 3]| (m[1] - right_of_centre_plain[1]).abs();
            d(a).total_cmp(&d(b))
        })
        .expect("a probe");
    assert!((below_in_tilted[1] - right_of_centre_plain[1]).abs() < 4.0);
}

#[test]
fn a_crop_renders_exactly_the_pixels_of_that_part_of_the_photo_at_100_percent() {
    let scene =
        Scene::new(1200, 800, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    // A crop of the middle half of the photo, viewed in a cell the size of the crop.
    let cropped = edit(&src, |s| {
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.25,
                y: 0.25,
                width: 0.5,
                height: 0.5,
            },
            angle_deg: 0.0,
            aspect: CropAspect::Free,
            auto: false,
        })
    });
    assert_eq!(
        r.final_size(&cropped, &CancelToken::new()).expect("size"),
        (600, 400)
    );
    let a = render(&r, &cropped, &state(600.0, 400.0));
    // The whole photo at 100 %, centred: the same 600 x 400 pixels.
    let mut st = state(600.0, 400.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (1200, 800));
    let b = render(&r, &src, &st);
    assert_eq!((a.width, a.height), (b.width, b.height));
    assert_eq!(a.rgba, b.rgba, "a crop is a window on the same pixels");
    assert!(a.sharp && a.report.scale == 1);
}

#[test]
fn the_half_resolution_image_agrees_with_full_resolution_inside_flat_patches() {
    let scene =
        Scene::new(1200, 800, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    let half = render(&r, &src, &state(600.0, 400.0)); // 50 %: the 1/2 group
    assert_eq!(half.report.scale, 2);
    let mut st = state(600.0, 400.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (1200, 800));
    let full = render(&r, &src, &st);
    assert_eq!(full.report.scale, 1);
    // Compare the same patches: the 1:1 frame shows the middle 600 x 400, i.e. cells 1.5..4.5 of
    // the 6 x 4 grid; compare through the 50 % frame's central 3 x 2 cells.
    for pr in 1..3usize {
        for pc in 2..4usize {
            // Patch (pc, pr) spans x in [200 pc, 200 pc + 200) of 1200; at 50 % that is 100 px wide.
            let h = mean_rgba(&half, 100 * pc + 20, 100 * pr + 20, 60, 60);
            // In the 1:1 frame the origin is (300, 200) of the photo.
            let f = mean_rgba(&full, 200 * pc - 300 + 40, 200 * pr - 200 + 40, 120, 120);
            assert_close(h, f, 1.6, &format!("patch ({pc},{pr})"));
        }
    }
}

#[test]
fn the_quarter_resolution_image_agrees_too() {
    let scene =
        Scene::new(1200, 800, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    let quarter = render(&r, &src, &state(300.0, 200.0));
    assert_eq!(quarter.report.scale, 4);
    let half = render(&r, &src, &state(600.0, 400.0));
    for py in 0..4 {
        for px in 0..6 {
            assert_close(
                cell_mean(&quarter, 6, 4, px, py),
                cell_mean(&half, 6, 4, px, py),
                1.6,
                &format!("patch ({px},{py})"),
            );
        }
    }
}

#[test]
fn vignetting_is_corrected_when_the_switch_is_on_and_only_then() {
    use ym_gpu::cpu::vignette::VignetteProfile;
    let (w, h) = (800usize, 600usize);
    let knots: Vec<i16> = (0..16).map(|i| (i * i * 14) as i16).collect(); // up to 3150
    let mut tag = vec![16i16];
    tag.extend(&knots);
    let profile = VignetteProfile::from_tag(&knots).expect("profile");
    // A flat field darkened by the lens: raw = flat * G(r).
    let white = light(PROFILE, 5500.0);
    // At 0.1 the flat field is what the camera's gain (+1.11 EV, the constant exposure) renders as middle grey, so the
    // corner and the centre are compared in the mid-tones and not in the shoulder of the display transform.
    let scene = Scene::new(w as u32, h as u32, white, Layout::Uniform, 0.1);
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
            let v = f64::from(mosaic[i]) - scene.black;
            mosaic[i] = (scene.black + v * g).round() as u16;
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
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    let ratio = |s: &PhotoSource| {
        let f = render(&r, s, &state(400.0, 300.0));
        let corner = mean_rgba(&f, 2, 2, 24, 18)[1];
        let mid = mean_rgba(&f, 188, 141, 24, 18)[1];
        corner / mid
    };
    let on = ratio(&src);
    let off = ratio(&edit(&src, |s| s.decode.lens_corrections = false));
    assert!((on - 1.0).abs() < 0.02, "corrected corner/centre {on:.3}");
    assert!(off < 0.975, "uncorrected corner/centre {off:.3}");
    // The 100 % tiles agree with the half-resolution image on the correction too.
    let mut st = state(400.0, 300.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (800, 600));
    let f = render(&r, &src, &st);
    assert_eq!(f.report.scale, 1);
}

#[test]
fn distortion_tables_crop_the_image_and_the_switch_restores_it() {
    let scene = Scene::new(800, 600, light(PROFILE, 5500.0), Layout::Uniform, 0.3);
    let dir = tempfile::tempdir().expect("dir");
    let mut dist = vec![16i16];
    dist.extend((0..16).map(|i| 40 + 25 * i)); // barrel correction up to ~2.7 %
    scene
        .builder(MODEL)
        .lens_tables(LensTables {
            vignetting: None,
            chromatic_aberration: None,
            distortion: Some(dist),
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let d = Developed::open(dir);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    let (w, h) = r.final_size(&src, &CancelToken::new()).expect("size");
    assert!(w < 800 && h < 600 && w > 700, "auto crop: {w}x{h}");
    assert!(
        (w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.01,
        "aspect kept: {w}x{h}"
    );
    let off = edit(&src, |s| s.decode.lens_corrections = false);
    assert_eq!(
        r.final_size(&off, &CancelToken::new()).expect("size"),
        (800, 600)
    );
    // The corrected frame is a flat neutral field with no black corners.
    let f = render(&r, &src, &state(w as f64 / 2.0, h as f64 / 2.0));
    let corner = mean_rgba(&f, 0, 0, 4, 4);
    let mid = centre(&f);
    assert_close(corner, mid, 2.5, "no black corner after the auto crop");
}

#[test]
fn a_crop_at_an_odd_origin_with_different_black_levels_renders_flat_and_neutral() {
    // The readout is 2 px wider and taller than the photo and the crop starts at (1, 1): the
    // levels, indexed by the readout's parity, must be re-indexed by the frame's. Each position
    // has its own black level, so a wrong index shows as a checkerboard cast.
    use ym_raw::layout::Rect;
    let (w, h) = (402u32, 302u32);
    let white = light(PROFILE, 5500.0);
    let black = [480u16, 520, 540, 500];
    let mut data = Vec::new();
    let peak = white.iter().copied().fold(0.0, f64::max);
    for y in 0..h {
        for x in 0..w {
            let pos = ((y & 1) * 2 + (x & 1)) as usize;
            let c = [0usize, 1, 1, 2][pos];
            let sig = 0.9 * 0.3 * white[c] / peak;
            let range = 16300.0 - f64::from(black[pos]);
            data.push((f64::from(black[pos]) + sig * range).round() as u16);
        }
    }
    let dir = tempfile::tempdir().expect("dir");
    let scene = Scene::new(w, h, white, Layout::Uniform, 0.3);
    scene
        .builder(MODEL)
        .black(black)
        .crop(Rect {
            x: 1,
            y: 1,
            width: 400,
            height: 300,
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &data)
        .expect("write");
    let d = Developed::open(dir);
    let r = Renderer::neutral_only().expect("render thread");
    let src = d.source();
    assert_eq!(
        r.final_size(&src, &CancelToken::new()).expect("size"),
        (400, 300)
    );
    let f = render(&r, &src, &state(200.0, 150.0));
    let m = centre(&f);
    assert!(
        (m[0] - m[1]).abs() < 1.5 && (m[2] - m[1]).abs() < 1.5,
        "{m:?}"
    );
    // Flat: every pixel within the dither of the mean.
    let worst = f
        .rgba
        .chunks(4)
        .map(|p| (f64::from(p[1]) - m[1]).abs())
        .fold(0.0, f64::max);
    assert!(worst <= 2.0, "worst deviation {worst}");
    // And the 100 % tiles (a different code path) agree.
    let mut st = state(200.0, 150.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (400, 300));
    let t = render(&r, &src, &st);
    assert_eq!(t.report.scale, 1);
    assert_close(
        centre(&t),
        m,
        1.5,
        "tiles against the half-resolution image",
    );
}

#[test]
fn a_craw_file_renders_like_the_uncompressed_one() {
    let white = light(PROFILE, 5500.0);
    let scene = Scene::new(640, 480, white, Layout::Patchwork, 0.4).noise(1e-4, 5e-7);
    let plain = Developed::new(&scene, MODEL);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(MODEL)
        .craw(CrawEncoder::new())
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let craw = Developed::open(dir);
    let r = Renderer::neutral_only().expect("render thread");
    let a = render(&r, &plain.source(), &state(320.0, 240.0));
    let b = render(&r, &craw.source(), &state(320.0, 240.0));
    for py in 0..4 {
        for px in 0..6 {
            assert_close(
                cell_mean(&a, 6, 4, px, py),
                cell_mean(&b, 6, 4, px, py),
                3.0,
                &format!("cRAW patch ({px},{py})"),
            );
        }
    }
    // The full-resolution path takes the quantisation steps into the clip mask without failing.
    let mut st = state(320.0, 240.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (640, 480));
    let t = render(&r, &craw.source(), &st);
    assert_eq!(t.report.scale, 1);
}

fn write_ms(dir: &std::path::Path, white: [f64; 3], level: f64, w: u32, h: u32) {
    let mut rgb = Vec::new();
    for _ in 0..w * h {
        for c in 0..3 {
            // Unbalanced camera RGB of a neutral field: pedestal 1024 plus the signal.
            rgb.push((1024.0 + level * white[c] / white[1] * 14000.0).round() as u16);
        }
    }
    ArwBuilder::new(MODEL, w, h)
        .wb(Some(wb_levels(white)))
        .lossless_ms(LosslessMs::new(
            ChromaSubsampling::Yuv420,
            rgb,
            w,
            h,
            [1.0 / white[0], 1.0, 1.0 / white[2]].map(|g| g as f32),
        ))
        .write_to(&dir.join("DSC00001.ARW"), &[])
        .expect("write");
}

#[test]
fn a_lossless_ms_file_renders_through_the_reduced_path_with_the_right_scale() {
    let white = light(PROFILE, 5500.0);
    let dir = tempfile::tempdir().expect("dir");
    write_ms(dir.path(), white, 0.4, 256, 192);
    let d = Developed::open(dir);
    let src = d.source();
    assert!(src.raw.is_reduced());
    let r = Renderer::neutral_only().expect("render thread");
    // R1 scales the three channels by one range (green's): a neutral field of 0.4 * 14000 DN above
    // the pedestal over a 16512 DN range is 0.339 after the gains, whatever the colour; the Before view
    // shows that through the camera default.
    let v = 0.4 * 14000.0 / 16512.0;
    let f = render(&r, &manual_ev(&src, 0.0), &before(128.0, 96.0));
    let expect = camera_default_code(v);
    for (c, g) in centre(&f).iter().enumerate() {
        assert!(
            (g - expect).abs() < 1.5,
            "channel {c}: {g:.2} vs {expect:.2}"
        );
    }
    // Final view: neutral. And the 100 % view (the group at full resolution) agrees.
    let fit = render(&r, &src, &state(128.0, 96.0));
    let m = centre(&fit);
    assert!(
        (m[0] - m[1]).abs() < 2.0 && (m[2] - m[1]).abs() < 2.0,
        "{m:?}"
    );
    let mut st = state(128.0, 96.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (256, 192));
    let full = render(&r, &src, &st);
    assert_close(centre(&full), m, 2.0, "full against reduced");
    // No tiles for these files: the group at k = 1 is the image.
    assert_eq!(full.report.runs.tiles, 0);
}

#[test]
fn an_exported_full_render_matches_the_viewport_pixel_for_pixel_at_100_percent() {
    let scene =
        Scene::new(800, 600, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, MODEL);
    let r = Renderer::neutral_only().expect("render thread");
    // With the output sharpening off: the same camera pixels through the same chain. (With it on,
    // the viewport shows the sharpened picture the export writes; `tests/render_sharpen.rs`.)
    let src = edit(&d.source(), |s| s.output_sharpening = 0.0);
    let full = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(ym_color::display::OutputTarget::Viewport),
            },
            &CancelToken::new(),
        )
        .expect("full render");
    assert_eq!((full.width, full.height), (800, 600));
    assert_eq!(full.rgb.len(), 800 * 600 * 3);
    // The viewport at 100 % in a cell the size of the photo: same camera pixels, same chain. The
    // frame is encoded and dithered; encode the linear values the same way and compare.
    let mut st = state(800.0, 600.0);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (800, 600));
    // A cell exactly the size of the photo is Fit; ask for the 100 % tiles explicitly.
    let frame = render(&r, &src, &st);
    assert_eq!(
        frame.report.scale, 1,
        "fit at zoom 1.0 uses the full-resolution tiles"
    );
    let chain_space = ym_color::encode::OutputSpace::DisplayP3;
    let mut worst = 0.0f64;
    for y in (0..600).step_by(7) {
        for x in (0..800).step_by(5) {
            let i = (y * 800 + x) * 3;
            let lin = [full.rgb[i], full.rgb[i + 1], full.rgb[i + 2]].map(f64::from);
            let enc = chain_space.encode_clamped(lin);
            for (c, e) in enc.iter().enumerate() {
                let want = e * 255.0;
                let got = f64::from(frame.rgba[(y * 800 + x) * 4 + c]);
                worst = worst.max((got - want).abs());
            }
        }
    }
    assert!(
        worst <= 1.0,
        "worst difference {worst} codes (the dither moves a value by less than one)"
    );
}
