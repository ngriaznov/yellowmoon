//! N-10 (`docs/23-nodes.md` sections 3.3, 3.4 and 4, `docs/algorithms/06-masks.md` sections 3 and 4): the
//! shapes (Linear, Radial), the ranges (Colour, Brightness), how the rows combine, the key input and the
//! mask blend, on the CPU reference.
//!
//! The formulas are written out again here in a different shape from the module's (the independent
//! evaluation of the maths file's text), and the chain is held to hand-made compositions of the pieces.

use std::collections::BTreeMap;

use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::curve::CurveSet;
use ym_color::display::{
    ChainParams, ColorChain, Frame, Layout, MainNode, NodeParams, NodeSet, Trim, View,
};
use ym_color::gamut::LinearAwg4;
use ym_color::grade::{E1, E2, Grade, GradeParams, grade_luma};
use ym_color::logc::{LogC4, logc4_encode};
use ym_color::mask::{
    MIN_RAMP_PIXELS, MaskKind, MaskOp, MaskRow, NodeMasks, OpKind, PhotoFrame, PhotoMap, PhotoSpot,
    PixelAt, RANGE_RAMP_MIN, fold, smooth_step,
};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;
use ym_color::warp::chroma_of;

// ---- helpers ------------------------------------------------------------------------------------

fn spot(x: f64, y: f64, frame: PhotoFrame) -> PixelAt<'static> {
    Some(PhotoSpot::new([x, y], frame))
}

/// A photo of `w x h` pixels rendered with output pixels of `pixel` photo pixels.
fn frame(w: f64, h: f64, pixel: f64) -> PhotoFrame {
    PhotoFrame::new(w, h, pixel)
}

/// `S(t)` written out: the smoothstep of the maths file.
fn s_of(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn linear(center: [f64; 2], angle_deg: f64, feather: f64) -> MaskRow {
    MaskRow::new(MaskKind::Linear {
        center,
        angle_deg,
        feather,
    })
}

fn radial(center: [f64; 2], radii: [f64; 2], angle_deg: f64, feather: f64) -> MaskRow {
    MaskRow::new(MaskKind::Radial {
        center,
        radii,
        angle_deg,
        feather,
    })
}

fn colour(center: [f64; 2], width: f64, softness: f64) -> MaskRow {
    MaskRow::new(MaskKind::ColorRange {
        center,
        width,
        softness,
    })
}

fn brightness(low: f64, high: f64, softness: f64) -> MaskRow {
    MaskRow::new(MaskKind::BrightnessRange {
        low,
        high,
        softness,
    })
}

fn masks(rows: Vec<MaskRow>) -> NodeMasks {
    NodeMasks {
        rows,
        invert: false,
    }
}

/// Equal to within rounding (the shapes are evaluated from positions that carry a few ulps).
fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn grey() -> LogC4 {
    LogC4([0.3, 0.3, 0.3])
}

/// `m` of rows at a place on a photo, key `None`, an input that does not matter.
fn m_at(rows: Vec<MaskRow>, x: f64, y: f64, f: PhotoFrame) -> f64 {
    masks(rows).prepare().mask(None, grey(), 0.3, spot(x, y, f))
}

/// The independent evaluation of docs/algorithms/06 section 4.1: `p` in the isotropic frame, the signed
/// distance `s` along `n = (-sin, cos)`, `h = max(0.5 f, one pixel)`.
fn linear_ref(
    (px, py): (f64, f64),
    c: [f64; 2],
    angle_deg: f64,
    feather: f64,
    (w, h): (f64, f64),
    pixel_px: f64,
) -> f64 {
    let l = w.max(h);
    let p = ((px - c[0]) * w / l, (py - c[1]) * h / l);
    let t = angle_deg.to_radians();
    let n = (-t.sin(), t.cos());
    let s = p.0 * n.0 + p.1 * n.1;
    // The transition's width, centred on the line: the feather, and at least the narrowest ramp.
    let width = feather.max(MIN_RAMP_PIXELS * pixel_px / l);
    1.0 - s_of((s + 0.5 * width) / width)
}

/// `1 - S(t)`.
fn s_of_complement(t: f64) -> f64 {
    1.0 - s_of(t)
}

/// Section 4.2: the pixel rotated by `-angle`, `d` in units of the radii, the ramp `q`.
fn radial_ref(
    (px, py): (f64, f64),
    c: [f64; 2],
    r: [f64; 2],
    angle_deg: f64,
    feather: f64,
    (w, h): (f64, f64),
    pixel_px: f64,
) -> f64 {
    let l = w.max(h);
    let p = ((px - c[0]) * w / l, (py - c[1]) * h / l);
    let t = angle_deg.to_radians();
    let (u, v) = (
        p.0 * t.cos() + p.1 * t.sin(),
        -p.0 * t.sin() + p.1 * t.cos(),
    );
    let d = ((u / r[0]).powi(2) + (v / r[1]).powi(2)).sqrt();
    // |grad d| per unit of L, from the partial derivatives of d(x, y) (the chain rule through the rotation),
    // held to its bounds 1/rmax and 1/rmin: how many d-units lie across one unit of length at the pixel.
    let (sn, cs) = (t.sin(), t.cos());
    let dx = (u * cs / (r[0] * r[0]) - v * sn / (r[1] * r[1])) / d.max(1e-12);
    let dy = (u * sn / (r[0] * r[0]) + v * cs / (r[1] * r[1])) / d.max(1e-12);
    let grad = dx
        .hypot(dy)
        .clamp(1.0 / r[0].max(r[1]), 1.0 / r[0].min(r[1]));
    // The ramp: the feather, and at least the narrowest ramp across the outline; centred on the contour the
    // feather alone has, d = 1 - f/2.
    let q = feather.max(MIN_RAMP_PIXELS * (pixel_px / l) * grad);
    s_of_complement((d - (1.0 - 0.5 * feather - 0.5 * q)) / q)
}

// ---- shapes -------------------------------------------------------------------------------------

#[test]
fn linear_matches_the_maths_file_on_a_grid() {
    let mut worst = 0.0f64;
    let mut n = 0;
    for (w, h, pixel) in [
        (6000.0, 4000.0, 1.0),
        (4000.0, 6000.0, 4.0),
        (1500.0, 1000.0, 0.25),
    ] {
        for &(c, angle, feather) in &[
            ([0.5, 0.5], 0.0, 0.2),
            ([0.3, 0.6], 30.0, 0.5),
            ([0.7, 0.2], -75.0, 0.05),
            ([0.5, 0.5], 90.0, 1.0),
            ([0.4, 0.5], 180.0, 0.0),
            ([0.5, 0.8], -135.0, 0.3),
        ] {
            let f = frame(w, h, pixel);
            for i in 0..=24 {
                for j in 0..=24 {
                    let (x, y) = (f64::from(i) / 24.0, f64::from(j) / 24.0);
                    let got = m_at(vec![linear(c, angle, feather)], x, y, f);
                    let want = linear_ref((x, y), c, angle, feather, (w, h), pixel);
                    worst = worst.max((got - want).abs());
                    n += 1;
                }
            }
        }
    }
    println!("Linear against the maths file: {n} points, largest difference {worst:.2e}");
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn linear_has_its_named_values() {
    let f = frame(1000.0, 1000.0, 1.0);
    // Horizontal line through the middle, feather 0.2: half-width 0.1 of the long edge.
    let row = linear([0.5, 0.5], 0.0, 0.2);
    // On the centre line: 0.5. At the start line (0.1 above) 1, at the end line (0.1 below) 0.
    assert!(near(m_at(vec![row], 0.5, 0.5, f), 0.5));
    assert!(near(m_at(vec![row], 0.3, 0.4, f), 1.0));
    assert!(near(m_at(vec![row], 0.3, 0.6, f), 0.0));
    // Beyond the lines it stays 1 above and 0 below.
    assert_eq!(m_at(vec![row], 0.9, 0.05, f), 1.0);
    assert_eq!(m_at(vec![row], 0.9, 0.95, f), 0.0);
    // Along the line nothing changes.
    assert_eq!(m_at(vec![row], 0.0, 0.5, f), m_at(vec![row], 1.0, 0.5, f));
    // An angle of 90 degrees (clockwise) turns the line to vertical: n = (-1, 0) points left, so s grows
    // to the left and the mask is 1 on the right of the line and 0 on its left.
    let row = linear([0.5, 0.5], 90.0, 0.2);
    assert!(near(m_at(vec![row], 0.6, 0.3, f), 1.0));
    assert!(near(m_at(vec![row], 0.4, 0.3, f), 0.0));
    assert!(near(m_at(vec![row], 0.5, 0.9, f), 0.5));
    // The line is a line of the photo's pixels, not of its fractions: on a wide photo a diagonal of 45
    // degrees has 1 : 1 pixels, so a step of 0.1 of the width and 0.15 of the height are equal.
    let wide = frame(3000.0, 2000.0, 1.0);
    let row = linear([0.5, 0.5], 45.0, 0.1);
    let a = m_at(vec![row], 0.5 + 0.1, 0.5, wide);
    let b = m_at(vec![row], 0.5, 0.5 + 0.1 * 3000.0 / 2000.0, wide);
    // (0.1 W, 0) along n = (-sin 45, cos 45) is -0.07 L; (0, 0.15 H) is +0.07 L.
    assert!((a + b - 1.0).abs() < 1e-12, "{a} {b}");
}

#[test]
fn radial_matches_the_maths_file_on_a_grid() {
    let mut worst = 0.0f64;
    let mut n = 0;
    for (w, h, pixel) in [(6000.0, 4000.0, 1.0), (4000.0, 6000.0, 4.0)] {
        for &(c, r, angle, feather) in &[
            ([0.5, 0.5], [0.25, 0.18], 0.0, 0.3),
            ([0.4, 0.6], [0.2, 0.1], 35.0, 0.0),
            ([0.7, 0.3], [0.05, 0.3], -80.0, 1.0),
            ([0.5, 0.5], [0.4, 0.4], 0.0, 0.5),
            ([0.2, 0.8], [0.01, 0.02], 130.0, 0.2),
        ] {
            let f = frame(w, h, pixel);
            for i in 0..=30 {
                for j in 0..=30 {
                    let (x, y) = (f64::from(i) / 30.0, f64::from(j) / 30.0);
                    let got = m_at(vec![radial(c, r, angle, feather)], x, y, f);
                    let want = radial_ref((x, y), c, r, angle, feather, (w, h), pixel);
                    worst = worst.max((got - want).abs());
                    n += 1;
                }
            }
        }
    }
    println!("Radial against the maths file: {n} points, largest difference {worst:.2e}");
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn radial_has_its_named_values() {
    // A square photo, so that fractions are lengths. Radii 0.2 and 0.1, feather 0.5: the outline is at
    // d = 1 and the inner ring at d = 0.5.
    let f = frame(1000.0, 1000.0, 1.0);
    let row = radial([0.5, 0.5], [0.2, 0.1], 0.0, 0.5);
    assert_eq!(m_at(vec![row], 0.5, 0.5, f), 1.0, "the centre");
    assert!(
        near(m_at(vec![row], 0.5 + 0.1, 0.5, f), 1.0),
        "the inner ring"
    );
    assert!(near(m_at(vec![row], 0.5 + 0.2, 0.5, f), 0.0), "the outline");
    assert_eq!(m_at(vec![row], 0.95, 0.5, f), 0.0, "beyond");
    // Halfway along the ramp (d = 0.75): 0.5.
    assert!(near(m_at(vec![row], 0.5 + 0.15, 0.5, f), 0.5));
    // The second axis: radius 0.1 along y.
    assert!(near(m_at(vec![row], 0.5, 0.5 + 0.1, f), 0.0));
    assert!(near(m_at(vec![row], 0.5, 0.5 + 0.075, f), 0.5));
    // Rotated by 90 degrees the first axis is vertical (clockwise, y down).
    let turned = radial([0.5, 0.5], [0.2, 0.1], 90.0, 0.5);
    assert!(near(m_at(vec![turned], 0.5, 0.5 + 0.2, f), 0.0));
    assert!(near(m_at(vec![turned], 0.5, 0.5 + 0.15, f), 0.5));
    assert!(near(m_at(vec![turned], 0.5 + 0.1, 0.5, f), 0.0));
    // Feather 1: a smooth cone from the centre, 0.5 at half the radius.
    let cone = radial([0.5, 0.5], [0.2, 0.2], 0.0, 1.0);
    assert_eq!(m_at(vec![cone], 0.5, 0.5, f), 1.0);
    assert!(near(m_at(vec![cone], 0.6, 0.5, f), 0.5));
    // A circle is a circle on any photo: 0.225 of the long edge to the right and below are the same
    // distance in pixels (d = 0.75, halfway along the ramp).
    let wide = frame(3000.0, 2000.0, 1.0);
    let c = radial([0.5, 0.5], [0.3, 0.3], 0.0, 0.5);
    let right = m_at(vec![c], 0.5 + 0.225, 0.5, wide);
    let down = m_at(vec![c], 0.5, 0.5 + 0.225 * 3000.0 / 2000.0, wide);
    assert!((right - down).abs() < 1e-14, "{right} {down}");
    assert!((right - 0.5).abs() < 1e-12);
}

#[test]
fn an_edge_is_a_ramp_across_it_of_the_minimum_width_centred_on_it_at_feather_0() {
    // docs/17 K8: no edge depends on a rounding. At Feather 0 a shape's edge is a ramp of MIN_RAMP_PIXELS output
    // pixels across it, centred on the edge, in the render's own pixels: the same mask at two scales has the
    // ramp of each. (k is the reduction of the render: pixel = k photo pixels.)
    const { assert!(MIN_RAMP_PIXELS >= 1.0) } // never narrower than one output pixel
    for k in [1.0, 2.0, 4.0] {
        let f = frame(6000.0, 4000.0, k);
        let ramp = MIN_RAMP_PIXELS * k / 6000.0;
        // Linear: the transition spans [-ramp/2, +ramp/2] around the line, never less than one pixel.
        let row = linear([0.5, 0.5], 0.0, 0.0);
        let at = |dy_l: f64| m_at(vec![row], 0.5, 0.5 + dy_l * 6000.0 / 4000.0, f);
        assert!(near(at(-0.5 * ramp), 1.0));
        assert!(near(at(0.5 * ramp), 0.0));
        assert!(near(at(0.0), 0.5));
        let mid_in = at(-0.25 * ramp);
        let mid_out = at(0.25 * ramp);
        assert!(mid_in > 0.5 && mid_in < 1.0 && mid_out > 0.0 && mid_out < 0.5);
        // Radial: the same ramp across the outline, centred on it, as wide in length at the ends of both
        // axes (an ellipse of radii 0.2 and 0.1).
        let r = radial([0.5, 0.5], [0.2, 0.1], 0.0, 0.0);
        let on = |dx_l: f64, dy_l: f64| m_at(vec![r], 0.5 + dx_l, 0.5 + dy_l * 6000.0 / 4000.0, f);
        for (outline, along_x) in [(0.1, false), (0.2, true)] {
            let value = |off: f64| {
                if along_x {
                    on(outline + off, 0.0)
                } else {
                    on(0.0, outline + off)
                }
            };
            assert!(near(value(-0.5 * ramp), 1.0), "k {k}, {outline}");
            assert!(near(value(0.0), 0.5), "k {k}, {outline}: {}", value(0.0));
            assert!(near(value(0.5 * ramp), 0.0), "k {k}, {outline}");
            // Inside and outside the outline the mask is partly on and partly off: half in, half out.
            assert!(value(-0.25 * ramp) > 0.5 && value(-0.25 * ramp) < 1.0);
            assert!(value(0.25 * ramp) > 0.0 && value(0.25 * ramp) < 0.5);
        }
        // A feather of a pixel or more runs inwards from the outline as before: 1 at the inner ring, 0 at the outline.
        let soft = radial([0.5, 0.5], [0.2, 0.1], 0.0, 0.5);
        assert!(near(
            m_at(vec![soft], 0.5, 0.5 + 0.1 * 6000.0 / 4000.0, f),
            0.0
        ));
        assert!(near(
            m_at(vec![soft], 0.5, 0.5 + 0.05 * 6000.0 / 4000.0, f),
            1.0
        ));
    }
}

#[test]
fn a_shape_with_no_position_selects_nothing() {
    for row in [
        linear([0.5, 0.5], 0.0, 0.2),
        radial([0.5, 0.5], [0.2, 0.2], 0.0, 0.2),
    ] {
        let ops = masks(vec![row]).prepare();
        assert_eq!(ops.mask(None, grey(), 0.3, None), 0.0);
    }
}

// ---- ranges -------------------------------------------------------------------------------------

/// A LogC4 value with the chroma `(a, b)` along `e1`, `e2` around a grey: `v = g + a e1 + b e2`.
fn with_chroma(g: f64, a: f64, b: f64) -> LogC4 {
    LogC4([
        g + a * E1[0] + b * E2[0],
        g + a * E1[1] + b * E2[1],
        g + a * E1[2] + b * E2[2],
    ])
}

#[test]
fn the_colour_range_follows_the_maths_file() {
    let c0 = [0.12, -0.05];
    let (width, softness): (f64, f64) = (0.06, 0.5);
    let q = (softness * width).max(RANGE_RAMP_MIN);
    assert!(
        (q - 0.03).abs() < 1e-15,
        "the ramp is the softness, over the minimum"
    );
    let m = masks(vec![colour(c0, width, softness)]).prepare();
    let s_at = |d: f64, g: f64| {
        // A colour at chroma distance d from the centre along e1.
        let v = with_chroma(g, c0[0] + d, c0[1]);
        m.mask(None, v, 0.0, None)
    };
    let want = |d: f64| 1.0 - s_of((d - width) / q);
    for g in [0.1, 0.3, 0.7] {
        // The colour of the centre, and inside the width: 1, whatever the tone.
        assert!((s_at(0.0, g) - 1.0).abs() < 1e-15);
        assert!((s_at(width, g) - 1.0).abs() < 1e-12);
        // Beyond width + q: 0.
        assert!(s_at(width + 0.1, g).abs() < 1e-15);
        for d in [0.0, 0.03, 0.06, 0.07, 0.08, 0.085, 0.09, 0.12] {
            assert!((s_at(d, g) - want(d)).abs() < 1e-12, "d {d}, g {g}");
        }
    }
    // The chroma only: a neutral shadow and a neutral highlight have the same s.
    let n = masks(vec![colour([0.0, 0.0], 0.05, 0.5)]).prepare();
    assert_eq!(
        n.mask(None, LogC4([0.1; 3]), 0.1, None),
        n.mask(None, LogC4([0.9; 3]), 0.9, None)
    );
    assert_eq!(n.mask(None, LogC4([0.4; 3]), 0.4, None), 1.0);
    // The picked pixel's own chroma is the centre: s = 1.
    let v = LogC4([0.41, 0.33, 0.22]);
    let [a, b] = chroma_of(v);
    let picked = masks(vec![colour([a, b], 0.005, 0.0)]).prepare();
    assert_eq!(picked.mask(None, v, 0.0, None), 1.0);
    // The minimum ramp: at softness 0 the edge is RANGE_RAMP_MIN wide, never a step.
    let hard = masks(vec![colour([0.0, 0.0], 0.1, 0.0)]).prepare();
    let at = |d: f64| hard.mask(None, with_chroma(0.3, d, 0.0), 0.0, None);
    assert_eq!(at(0.1), 1.0);
    assert!((at(0.1 + 0.5 * RANGE_RAMP_MIN) - 0.5).abs() < 1e-9);
    assert_eq!(at(0.1 + RANGE_RAMP_MIN), 0.0);
}

#[test]
fn the_brightness_range_follows_the_maths_file() {
    let (low, high, softness) = (0.3, 0.7, 0.4);
    let q = (0.25_f64 * softness).max(RANGE_RAMP_MIN);
    let m = masks(vec![brightness(low, high, softness)]).prepare();
    let at = |l: f64| m.mask(None, grey(), l, None);
    let want = |l: f64| s_of((l - (low - q)) / q) * (1.0 - s_of((l - high) / q));
    for i in 0..=100 {
        let l = f64::from(i) / 100.0 * 1.2 - 0.1;
        assert!((at(l) - want(l)).abs() < 1e-14, "{l}");
    }
    // A trapezoid: 0 under low - q, 1 from low to high, 0 over high + q.
    assert_eq!(at(low - q - 0.01), 0.0);
    assert_eq!(at(low), 1.0);
    assert_eq!(at(0.5), 1.0);
    assert_eq!(at(high), 1.0);
    assert_eq!(at(high + q + 0.01), 0.0);
    assert!((at(low - 0.5 * q) - 0.5).abs() < 1e-12);
    assert!((at(high + 0.5 * q) - 0.5).abs() < 1e-12);
    // Softness 0: the ramp is RANGE_RAMP_MIN.
    let hard = masks(vec![brightness(0.4, 0.6, 0.0)]).prepare();
    assert!((hard.mask(None, grey(), 0.4 - 0.5 * RANGE_RAMP_MIN, None) - 0.5).abs() < 1e-12);
    // low above high is both at their mean; numbers outside [0, 1] are clamped.
    let swapped = masks(vec![brightness(0.8, 0.4, 0.0)]).prepare();
    let mean = masks(vec![brightness(0.6, 0.6, 0.0)]).prepare();
    for l in [0.3, 0.55, 0.6, 0.65, 0.9] {
        assert_eq!(
            swapped.mask(None, grey(), l, None),
            mean.mask(None, grey(), l, None)
        );
    }
    let wild = masks(vec![brightness(-3.0, 9.0, 0.0)]).prepare();
    let tame = masks(vec![brightness(0.0, 1.0, 0.0)]).prepare();
    for l in [-0.1, 0.0, 0.5, 1.0, 1.1] {
        assert_eq!(
            wild.mask(None, grey(), l, None),
            tame.mask(None, grey(), l, None)
        );
    }
}

#[test]
fn the_brightness_range_reads_the_luma_of_the_nodes_input() {
    // `luma_of` is encode(Y(decode(v))) without the round trip when the linear values are known.
    let lin = [0.18, 0.18, 0.18];
    let v = LinearAwg4(lin).to_logc4();
    let l = ym_color::mask::PreparedMasks::luma_of(v, Some(lin));
    assert!((l - logc4_encode(0.18)).abs() < 1e-3, "grey is {l}"); // Y of equal channels is 0.18
    let from_code = ym_color::mask::PreparedMasks::luma_of(v, None);
    assert!((l - from_code).abs() < 1e-12);
}

// ---- rows: Invert and Opacity ---------------------------------------------------------------------

#[test]
fn invert_then_opacity_per_row() {
    let f = frame(1000.0, 1000.0, 1.0);
    let base = radial([0.5, 0.5], [0.2, 0.2], 0.0, 0.5);
    let s = |x: f64| m_at(vec![base], x, 0.5, f);
    for x in [0.5, 0.58, 0.63, 0.68, 0.72, 0.9] {
        let inv = MaskRow {
            invert: true,
            ..base
        };
        assert!((m_at(vec![inv], x, 0.5, f) - (1.0 - s(x))).abs() < 1e-15);
        let half = MaskRow {
            opacity: 0.5,
            ..base
        };
        assert!((m_at(vec![half], x, 0.5, f) - 0.5 * s(x)).abs() < 1e-15);
        // Invert first, then Opacity: an inverted row at 40 % is 0.4 (1 - s), not 1 - 0.4 s.
        let both = MaskRow {
            invert: true,
            opacity: 0.4,
            ..base
        };
        assert!((m_at(vec![both], x, 0.5, f) - 0.4 * (1.0 - s(x))).abs() < 1e-15);
    }
}

#[test]
fn raster_kinds_select_nothing_until_their_tasks() {
    let f = frame(1000.0, 1000.0, 1.0);
    let raster = MaskRow::new(MaskKind::Raster { plane: 0 });
    // s = 0: an added row adds nothing, so a node whose only mask is empty changes nothing.
    assert_eq!(m_at(vec![raster], 0.5, 0.5, f), 0.0);
    // The row's Invert applies to s = 0, as for an Object row with no point (`docs/23` section 4.2).
    let inverted = MaskRow {
        invert: true,
        ..raster
    };
    assert_eq!(m_at(vec![inverted], 0.5, 0.5, f), 1.0);
    assert_eq!(MaskOp::of(&raster).kind, OpKind::Raster);
}

// ---- combination --------------------------------------------------------------------------------

/// A row whose `s` is 1 at the middle of a square photo (a Linear line along the bottom edge, whose start
/// side is the whole photo above it), so that its amount is its Opacity.
fn all(opacity: f64) -> MaskRow {
    MaskRow {
        opacity,
        ..linear([0.5, 1.0], 0.0, 0.0)
    }
}

fn mask_of(rows: Vec<MaskRow>, invert: bool, key: Option<f64>) -> f64 {
    let f = frame(1000.0, 1000.0, 1.0);
    NodeMasks { rows, invert }
        .prepare()
        .mask(key, grey(), 0.3, spot(0.5, 0.5, f))
}

#[test]
fn adds_are_the_screen_and_subtracts_multiply() {
    // Two half-covered pixels give three quarters, nothing exceeds 1.
    assert_eq!(mask_of(vec![all(0.5), all(0.5)], false, None), 0.75);
    assert_eq!(mask_of(vec![all(1.0), all(0.5)], false, None), 1.0);
    // The order of adds does not matter.
    let (a, b, c) = (0.3, 0.55, 0.8);
    let abc = mask_of(vec![all(a), all(b), all(c)], false, None);
    let cab = mask_of(vec![all(c), all(a), all(b)], false, None);
    assert!((abc - cab).abs() < 1e-15);
    assert!((abc - (1.0 - (1.0 - a) * (1.0 - b) * (1.0 - c))).abs() < 1e-15);
    // A subtract multiplies by 1 - a: it cuts exactly its own coverage.
    let sub = |o: f64| MaskRow {
        subtract: true,
        ..all(o)
    };
    assert!((mask_of(vec![all(0.8), sub(0.25)], false, None) - 0.8 * 0.75).abs() < 1e-15);
    assert!((mask_of(vec![all(0.6), sub(1.0)], false, None)).abs() < 1e-15);
    // Add, subtract, add: in list order.
    let m = mask_of(vec![all(0.5), sub(0.5), all(0.5)], false, None);
    let want = 1.0 - (1.0 - 0.5 * 0.5) * (1.0 - 0.5);
    assert!((m - want).abs() < 1e-15);
    // A lone subtract starts from the whole photo: m = 1 - a.
    assert!((mask_of(vec![sub(0.3)], false, None) - 0.7).abs() < 1e-15);
    assert_eq!(mask_of(vec![sub(1.0)], false, None), 0.0);
    assert_eq!(mask_of(vec![sub(0.0)], false, None), 1.0);
}

#[test]
fn regions_and_ranges_fold_apart_and_multiply() {
    let f = frame(1000.0, 1000.0, 1.0);
    // A neutral pixel: the Colour range at the origin selects it (1); one far off selects nothing (0).
    let near = colour([0.0, 0.0], 0.05, 0.5);
    let far = colour([0.25, 0.25], 0.005, 0.0);
    let run = |rows: Vec<MaskRow>, invert: bool| {
        NodeMasks { rows, invert }
            .prepare()
            .mask(None, grey(), 0.3, spot(0.5, 0.5, f))
    };
    // A node with only ranges is the range alone (R = 1).
    assert_eq!(run(vec![near], false), 1.0);
    assert_eq!(run(vec![far], false), 0.0);
    // m = R K: a range keeps only its part of what the regions gave.
    let r = MaskRow {
        opacity: 0.6,
        ..all(1.0)
    };
    assert!((run(vec![r, near], false) - 0.6).abs() < 1e-15);
    assert_eq!(run(vec![r, far], false), 0.0);
    // The ranges combine among themselves as the regions do: two half-selecting ranges add up.
    let half_near = MaskRow {
        opacity: 0.5,
        ..near
    };
    assert_eq!(run(vec![half_near, half_near], false), 0.75);
    // A subtracted range (a lone one starts from the whole range group).
    let sub_near = MaskRow {
        subtract: true,
        ..near
    };
    assert_eq!(run(vec![sub_near], false), 0.0);
    assert_eq!(run(vec![sub_near, r], false), 0.0);
    let sub_far = MaskRow {
        subtract: true,
        ..far
    };
    assert_eq!(run(vec![sub_far], false), 1.0);
    // The node's Invert: m -> 1 - m, after everything.
    assert!((run(vec![r, near], true) - 0.4).abs() < 1e-15);
    // No rows: the whole photo; with the node's Invert, nothing.
    assert_eq!(run(vec![], false), 1.0);
    assert_eq!(run(vec![], true), 0.0);
}

#[test]
fn the_key_input_is_an_implicit_first_region_row() {
    // A node with a key input and no rows is its source's m, bit for bit.
    for key in [0.0, 0.123_456_789_012_345_6, 0.5, 1.0] {
        assert_eq!(mask_of(vec![], false, Some(key)), key);
    }
    // With the node's Invert: 1 - m_source.
    assert_eq!(mask_of(vec![], true, Some(0.3)), 1.0 - 0.3);
    // A Radial subtracted from it: m_source (1 - a).
    let sub_all = |o: f64| MaskRow {
        subtract: true,
        ..all(o)
    };
    assert!((mask_of(vec![sub_all(0.25)], false, Some(0.8)) - 0.8 * 0.75).abs() < 1e-15);
    // A Radial added to it: the screen.
    assert!((mask_of(vec![all(0.5)], false, Some(0.5)) - 0.75).abs() < 1e-15);
    // A Colour range: m_source K.
    let f = frame(1000.0, 1000.0, 1.0);
    let near = colour([0.0, 0.0], 0.05, 0.5);
    let half = MaskRow {
        opacity: 0.5,
        ..near
    };
    let m = NodeMasks {
        rows: vec![half],
        invert: false,
    }
    .prepare()
    .mask(Some(0.6), grey(), 0.3, spot(0.5, 0.5, f));
    assert!((m - 0.3).abs() < 1e-15);
    // The key is a region: it folds with the regions, not with the ranges. A lone subtract after it
    // does not start from 1.
    assert!(
        (mask_of(vec![sub_all(0.5)], false, Some(0.4)) - 0.2).abs() < 1e-15,
        "{}",
        mask_of(vec![sub_all(0.5)], false, Some(0.4))
    );
}

#[test]
fn a_node_blends_only_when_it_has_rows_an_invert_or_a_key() {
    let none = masks(vec![]).prepare();
    assert!(!none.blends(false));
    assert!(none.blends(true));
    assert!(masks(vec![all(1.0)]).prepare().blends(false));
    let inverted = NodeMasks {
        rows: vec![],
        invert: true,
    }
    .prepare();
    assert!(inverted.blends(false));
    assert!(masks(vec![]).is_empty());
    assert!(!inverted_is_empty());
}

fn inverted_is_empty() -> bool {
    NodeMasks {
        rows: vec![],
        invert: true,
    }
    .is_empty()
}

#[test]
fn fold_is_exact_for_the_first_row() {
    assert_eq!(fold(None, false, 0.3), 0.3);
    assert_eq!(fold(None, true, 0.3), 1.0 - 0.3);
    assert_eq!(fold(Some(0.5), false, 0.5), 0.75);
    assert_eq!(fold(Some(0.5), true, 0.5), 0.25);
    assert_eq!(smooth_step(f64::NAN), 0.0);
    assert_eq!(smooth_step(-1.0), 0.0);
    assert_eq!(smooth_step(0.5), 0.5);
}

// ---- the chain ------------------------------------------------------------------------------------

fn matrix() -> Mat3 {
    let gain = 0.25_f64.exp2();
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
    .map(|row| row.map(|v| v * gain))
}

fn base() -> ChainParams {
    let tone = ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap();
    ChainParams::new(matrix(), tone)
}

/// A node that visibly changes the picture: contrast and saturation and an exposure trim.
fn visible() -> NodeParams {
    NodeParams {
        trim: Trim {
            exposure_ev: 0.8,
            ..Trim::default()
        },
        grade: GradeParams {
            contrast: 40.0,
            saturation: -35.0,
            ..GradeParams::default()
        },
        ..NodeParams::default()
    }
}

/// What the node does to a LogC4 value, from the stage modules (no mask).
fn made(node: &NodeParams, v: LogC4) -> LogC4 {
    let mut v = v;
    let mut lin = v.to_linear().0;
    if let Some(g) = node.trim.gain() {
        lin = [lin[0] * g[0], lin[1] * g[1], lin[2] * g[2]];
        v = LinearAwg4(lin).to_logc4();
    }
    node.curves.bake().apply(
        ym_color::warp::Warp::new(&node.warp)
            .apply(Grade::new(&node.grade).apply(v, grade_luma(LinearAwg4(lin)))),
    )
}

fn blend(v_in: LogC4, out: LogC4, m: f64) -> LogC4 {
    if m == 1.0 {
        out
    } else if m == 0.0 {
        v_in
    } else {
        LogC4([0, 1, 2].map(|k| v_in.0[k] + m * (out.0[k] - v_in.0[k])))
    }
}

struct Spec {
    id: u32,
    picture: PictureSource,
    key: Option<u32>,
    params: NodeParams,
}

fn chain_of(specs: Vec<Spec>, output: u32, stop: Stop) -> ColorChain {
    let mut nodes = vec![PlanNode {
        id: MAIN_NODE,
        pos: [0.0, 0.0],
        picture: Some(PictureSource::Input),
        key: None,
    }];
    let mut others = BTreeMap::new();
    for (i, s) in specs.into_iter().enumerate() {
        nodes.push(PlanNode {
            id: s.id,
            pos: [1.0 + i as f64, 0.0],
            picture: Some(s.picture),
            key: s.key,
        });
        others.insert(s.id, s.params);
    }
    let plan = compile(
        &PlanGraph {
            nodes,
            output: PictureSource::Node(output),
        },
        stop,
    )
    .unwrap();
    let mut p = base();
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others,
    };
    ColorChain::new(&p).unwrap()
}

fn pixels() -> Vec<Vec3> {
    vec![
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.02, 0.03, 0.08],
        [0.9, 0.8, 0.1],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
        [2.0, 1.5, 1.0],
        [0.002, 0.003, 0.004],
    ]
}

/// Main's output for a camera pixel when Main is at its defaults: C8.
fn c8_of(px: Vec3) -> (Vec3, LogC4) {
    let p = base();
    let c6 = ym_color::consts::linalg::mul_vec(&p.matrix, px);
    let c7 = p.tone.apply_rgb(c6);
    (c7, LinearAwg4(c7).to_logc4())
}

#[test]
fn a_node_blends_by_its_mask_in_logc4() {
    let f = frame(1000.0, 1000.0, 1.0);
    let rows = vec![radial([0.5, 0.5], [0.2, 0.2], 0.0, 0.5)];
    let mut node = visible();
    node.masks = masks(rows.clone());
    let plain = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: visible(),
        }],
        2,
        Stop::Output,
    );
    let masked = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: node,
        }],
        2,
        Stop::Output,
    );
    assert!(masked.uses_photo_position());
    assert!(!plain.uses_photo_position());
    let prepared = masks(rows).prepare();
    for px in pixels() {
        let (_, c8) = c8_of(px);
        let want = made(&visible(), c8);
        for (x, y) in [
            (0.5, 0.5),
            (0.58, 0.5),
            (0.65, 0.5),
            (0.69, 0.5),
            (0.9, 0.9),
        ] {
            let m = prepared.mask(None, c8, 0.0, spot(x, y, f));
            let got = masked.apply_logc4_pixel_at(px, spot(x, y, f));
            let exp = blend(c8, want, m);
            for k in 0..3 {
                assert!(
                    (got.0[k] - exp.0[k]).abs() < 1e-12,
                    "{px:?} at ({x}, {y}): m {m}, got {got:?}, want {exp:?}"
                );
            }
            // m = 1 is the node's own bits, m = 0 the input's.
            if m == 1.0 {
                assert_eq!(got, plain.apply_logc4_pixel_at(px, spot(x, y, f)));
            }
            if m == 0.0 {
                assert_eq!(got, c8);
            }
        }
        // A pixel with no known place: the shapes select nothing, the node does nothing.
        assert_eq!(masked.apply_logc4_pixel(px), c8, "{px:?}");
    }
}

#[test]
fn a_node_without_masks_or_a_key_keeps_its_own_bits() {
    let f = frame(1000.0, 1000.0, 1.0);
    // Rows that are all off are not in the list, so a node with none is the plain node; an Invert alone
    // is mask 0.
    let plain = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: visible(),
        }],
        2,
        Stop::Output,
    );
    let empty = NodeParams {
        masks: NodeMasks::default(),
        ..visible()
    };
    let same = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: empty,
        }],
        2,
        Stop::Output,
    );
    let inverted = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: NodeParams {
                masks: NodeMasks {
                    rows: vec![],
                    invert: true,
                },
                ..visible()
            },
        }],
        2,
        Stop::Output,
    );
    for px in pixels() {
        let (_, c8) = c8_of(px);
        let at = spot(0.3, 0.4, f);
        assert_eq!(
            plain.apply_logc4_pixel_at(px, at),
            same.apply_logc4_pixel_at(px, at)
        );
        assert_eq!(plain.apply_logc4_pixel(px), made(&visible(), c8));
        // The whole photo, inverted, is nothing: the node leaves its input as it is.
        assert_eq!(inverted.apply_logc4_pixel_at(px, at), c8);
    }
}

#[test]
fn a_key_link_carries_the_final_mask_of_another_node() {
    let f = frame(1000.0, 1000.0, 1.0);
    // Node 2 makes a mask only (its picture goes nowhere): a Linear, then node 3 is keyed by it.
    let line = masks(vec![linear([0.5, 0.5], 0.0, 0.4)]);
    let key_node = NodeParams {
        masks: line.clone(),
        ..visible()
    };
    let keyed = chain_of(
        vec![
            Spec {
                id: 2,
                picture: PictureSource::Input,
                key: None,
                params: key_node,
            },
            Spec {
                id: 3,
                picture: PictureSource::Node(1),
                key: Some(2),
                params: visible(),
            },
        ],
        3,
        Stop::Output,
    );
    // The key-only node is live (its mask is used) and is in the plan.
    assert_eq!(keyed.params().nodes.plan.node_ids(), vec![1, 2, 3]);
    let prepared = line.prepare();
    for px in pixels() {
        let (_, c8) = c8_of(px);
        for y in [0.1, 0.4, 0.5, 0.6, 0.9] {
            let m = prepared.mask(None, c8, 0.0, spot(0.5, y, f));
            let got = keyed.apply_logc4_pixel_at(px, spot(0.5, y, f));
            let exp = blend(c8, made(&visible(), c8), m);
            for k in 0..3 {
                assert!((got.0[k] - exp.0[k]).abs() < 1e-12, "{px:?} y {y}");
            }
        }
    }
    // Node 3 with a Radial subtracted from the key: m = m_2 (1 - a).
    let hole = MaskRow {
        subtract: true,
        ..radial([0.5, 0.5], [0.1, 0.1], 0.0, 0.0)
    };
    let with_hole = chain_of(
        vec![
            Spec {
                id: 2,
                picture: PictureSource::Input,
                key: None,
                params: NodeParams {
                    masks: line.clone(),
                    ..visible()
                },
            },
            Spec {
                id: 3,
                picture: PictureSource::Node(1),
                key: Some(2),
                params: NodeParams {
                    masks: masks(vec![hole]),
                    ..visible()
                },
            },
        ],
        3,
        Stop::Output,
    );
    let hole_only = masks(vec![hole]).prepare();
    for px in pixels() {
        let (_, c8) = c8_of(px);
        for (x, y) in [(0.5, 0.5), (0.5, 0.58), (0.52, 0.4), (0.9, 0.2), (0.3, 0.5)] {
            let sp = spot(x, y, f);
            let m2 = prepared.mask(None, c8, 0.0, sp);
            let m3 = hole_only.mask(Some(m2), c8, 0.0, sp);
            let got = with_hole.apply_logc4_pixel_at(px, sp);
            let exp = blend(c8, made(&visible(), c8), m3);
            for k in 0..3 {
                assert!((got.0[k] - exp.0[k]).abs() < 1e-12, "{px:?} at ({x}, {y})");
            }
        }
    }
}

#[test]
fn a_bypassed_nodes_key_output_still_carries_its_mask() {
    let f = frame(1000.0, 1000.0, 1.0);
    let line = masks(vec![linear([0.5, 0.5], 0.0, 0.4)]);
    let build = |bypass: bool| {
        chain_of(
            vec![
                Spec {
                    id: 2,
                    picture: PictureSource::Node(1),
                    key: None,
                    params: NodeParams {
                        bypass,
                        masks: line.clone(),
                        ..visible()
                    },
                },
                Spec {
                    id: 3,
                    picture: PictureSource::Node(1),
                    key: Some(2),
                    params: NodeParams {
                        grade: GradeParams {
                            saturation: 60.0,
                            ..GradeParams::default()
                        },
                        ..NodeParams::default()
                    },
                },
            ],
            3,
            Stop::Output,
        )
    };
    let (a, b) = (build(false), build(true));
    for px in pixels() {
        for y in [0.2, 0.5, 0.8] {
            let sp = spot(0.5, y, f);
            assert_eq!(
                a.apply_logc4_pixel_at(px, sp),
                b.apply_logc4_pixel_at(px, sp)
            );
        }
    }
    // A bypassed node blends nothing itself: its picture is its input.
    let alone = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: NodeParams {
                bypass: true,
                masks: line,
                ..visible()
            },
        }],
        2,
        Stop::Output,
    );
    for px in pixels() {
        let (_, c8) = c8_of(px);
        assert_eq!(alone.apply_logc4_pixel_at(px, spot(0.5, 0.3, f)), c8);
    }
}

#[test]
fn the_mask_stop_shows_the_nodes_final_mask() {
    let f = frame(1000.0, 1000.0, 1.0);
    let rows = vec![
        radial([0.5, 0.5], [0.3, 0.3], 0.0, 0.4),
        MaskRow {
            subtract: true,
            ..linear([0.5, 0.5], 20.0, 0.2)
        },
        brightness(0.2, 0.9, 0.3),
    ];
    let node = NodeMasks {
        rows: rows.clone(),
        invert: true,
    };
    let c = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: NodeParams {
                masks: node.clone(),
                ..visible()
            },
        }],
        2,
        Stop::Mask(2),
    );
    let prepared = node.prepare();
    for px in pixels() {
        let (c7, c8) = c8_of(px);
        let l_in = ym_color::mask::PreparedMasks::luma_of(c8, None);
        // Main is a default node, so its output is its input, C8, and the node reads that.
        let _ = c7;
        for (x, y) in [(0.5, 0.5), (0.6, 0.45), (0.2, 0.9), (0.8, 0.1)] {
            let m = prepared.mask(None, c8, l_in, spot(x, y, f));
            let got = c.apply_logc4_pixel_at(px, spot(x, y, f));
            for k in 0..3 {
                assert!((got.0[k] - m).abs() < 1e-12, "{px:?} at ({x}, {y})");
            }
        }
    }
    // A node that has no mask shows 1: the whole photo.
    let plain = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: visible(),
        }],
        2,
        Stop::Mask(2),
    );
    assert_eq!(plain.apply_logc4_pixel(pixels()[1]), LogC4([1.0; 3]));
}

#[test]
fn keys_cross_the_split_between_passes() {
    // Main fans out to nine nodes (2..=10), each read by one of 20..=28; the keys of 20..=28 are
    // chained. Nine pictures alive at once need two passes (docs/23 section 5.2), and the keys cross
    // the split in buffers. The picture the Output reads is node 28's.
    let fan = 9u32;
    let mut specs = Vec::new();
    for i in 0..fan {
        specs.push(Spec {
            id: 2 + i,
            picture: PictureSource::Node(1),
            key: None,
            params: NodeParams {
                trim: Trim {
                    exposure_ev: 0.1 * f64::from(i + 1),
                    ..Trim::default()
                },
                ..NodeParams::default()
            },
        });
    }
    let row = |i: u32| {
        radial(
            [0.2 + 0.07 * f64::from(i), 0.5],
            [0.25, 0.2],
            10.0 * f64::from(i),
            0.5,
        )
    };
    for i in 0..fan {
        specs.push(Spec {
            id: 20 + i,
            picture: PictureSource::Node(2 + i),
            key: (i > 0).then(|| 20 + i - 1),
            params: NodeParams {
                grade: GradeParams {
                    contrast: 25.0,
                    ..GradeParams::default()
                },
                masks: masks(vec![row(i)]),
                ..NodeParams::default()
            },
        });
    }
    let c = chain_of(specs, 20 + fan - 1, Stop::Output);
    assert!(
        c.params().nodes.plan.passes.len() >= 2,
        "{} passes",
        c.params().nodes.plan.passes.len()
    );
    let f = frame(1000.0, 1000.0, 1.0);
    for px in pixels() {
        let (_, c8) = c8_of(px);
        for (x, y) in [(0.5, 0.5), (0.3, 0.4), (0.7, 0.55), (0.9, 0.9)] {
            let sp = spot(x, y, f);
            // m of node 20+i: its Radial and the key of the node before (a screen: an add).
            let mut m = None;
            for i in 0..fan {
                m = Some(masks(vec![row(i)]).prepare().mask(m, c8, 0.0, sp));
            }
            let m = m.unwrap();
            // The picture is node 28's input (node 10's output: exposure 0.9) through its node.
            let node10 = NodeParams {
                trim: Trim {
                    exposure_ev: 0.1 * f64::from(fan),
                    ..Trim::default()
                },
                ..NodeParams::default()
            };
            let v_in = made(&node10, c8);
            let grade = NodeParams {
                grade: GradeParams {
                    contrast: 25.0,
                    ..GradeParams::default()
                },
                ..NodeParams::default()
            };
            let exp = blend(v_in, made(&grade, v_in), m);
            let got = c.apply_logc4_pixel_at(px, sp);
            for k in 0..3 {
                assert!(
                    (got.0[k] - exp.0[k]).abs() < 1e-12,
                    "{px:?} at ({x}, {y}): m {m}"
                );
            }
        }
    }
}

// ---- frames of pixels -----------------------------------------------------------------------------

/// The positions of a `w x h` frame that covers the photo: the centres of its pixels.
struct Even {
    w: usize,
    h: usize,
}

impl PhotoMap for Even {
    fn position(&self, x: usize, y: usize) -> [f64; 2] {
        [
            (x as f64 + 0.5) / self.w as f64,
            (y as f64 + 0.5) / self.h as f64,
        ]
    }
}

#[test]
fn a_frame_with_positions_equals_its_pixels_one_by_one_on_any_number_of_threads() {
    let (w, h) = (23usize, 17usize);
    let map = Even { w, h };
    let f = frame(w as f64, h as f64, 1.0);
    let mut node = visible();
    node.masks = masks(vec![
        radial([0.5, 0.5], [0.4, 0.3], 20.0, 0.4),
        MaskRow {
            subtract: true,
            ..linear([0.7, 0.5], -40.0, 0.2)
        },
        brightness(0.25, 0.8, 0.3),
    ]);
    let c = chain_of(
        vec![Spec {
            id: 2,
            picture: PictureSource::Node(1),
            key: None,
            params: node,
        }],
        2,
        Stop::Output,
    );
    // A planar frame of varied camera pixels.
    let px = pixels();
    let mut planar = vec![0.0f32; 3 * w * h];
    for i in 0..w * h {
        let p = px[i % px.len()];
        let s = 0.5 + (i % 7) as f32 * 0.2;
        for k in 0..3 {
            planar[k * w * h + i] = p[k] as f32 * s;
        }
    }
    let frame_in = Frame::planar(w, h, &planar).unwrap().with_photo(&map, f);
    let mut single = vec![0.0f32; 3 * w * h];
    c.render_encoded_f32(View::Final, &frame_in, Layout::Interleaved, &mut single, 1)
        .unwrap();
    for threads in [2, 4] {
        let mut many = vec![0.0f32; 3 * w * h];
        c.render_encoded_f32(
            View::Final,
            &frame_in,
            Layout::Interleaved,
            &mut many,
            threads,
        )
        .unwrap();
        assert_eq!(single, many, "{threads} threads");
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let rgb = [0, 1, 2].map(|k| f64::from(planar[k * w * h + i]));
            let want = c.apply_view_pixel_at(
                View::Final,
                rgb,
                spot(map.position(x, y)[0], map.position(x, y)[1], f),
            );
            for k in 0..3 {
                assert_eq!(single[3 * i + k], want[k] as f32, "pixel ({x}, {y})");
            }
        }
    }
    // Without the positions the shapes select nothing, and the render differs.
    let bare = Frame::planar(w, h, &planar).unwrap();
    let mut flat = vec![0.0f32; 3 * w * h];
    c.render_encoded_f32(View::Final, &bare, Layout::Interleaved, &mut flat, 1)
        .unwrap();
    assert_ne!(flat, single);
}

#[test]
fn only_nodes_with_a_shape_ask_for_positions() {
    let with = |rows: Vec<MaskRow>| {
        chain_of(
            vec![Spec {
                id: 2,
                picture: PictureSource::Node(1),
                key: None,
                params: NodeParams {
                    masks: masks(rows),
                    ..visible()
                },
            }],
            2,
            Stop::Output,
        )
    };
    assert!(!with(vec![]).uses_photo_position());
    assert!(!with(vec![colour([0.0, 0.0], 0.05, 0.5)]).uses_photo_position());
    assert!(!with(vec![brightness(0.2, 0.9, 0.5)]).uses_photo_position());
    // A raster row needs the places of the pixels to make its plane, and reads the plane.
    let raster = with(vec![MaskRow::new(MaskKind::Raster { plane: 0 })]);
    assert!(raster.uses_photo_position());
    // It reads a plane only when the chain's RasterSet holds one: with an empty set (a row with no plane yet) there is
    // nothing to make, and the device must not be asked (`tests/mask_raster.rs` checks the non-empty set).
    assert!(raster.rasters().rows.is_empty());
    assert!(!raster.uses_rasters());
    assert!(!with(vec![linear([0.5, 0.5], 0.0, 0.2)]).uses_rasters());
    assert!(with(vec![linear([0.5, 0.5], 0.0, 0.2)]).uses_photo_position());
    assert!(with(vec![radial([0.5, 0.5], [0.2, 0.2], 0.0, 0.2)]).uses_photo_position());
}

#[test]
fn the_default_graph_is_unchanged_by_masks_that_are_not_there() {
    // Main with no masks and the trace of 0.9.0: the stages are as they were.
    let c = ColorChain::new(&base()).unwrap();
    for px in pixels() {
        let t = c.trace(px);
        assert_eq!(t.c9_logc4, t.c8_logc4);
        assert_eq!(t.c11_logc4, t.c8_logc4);
        assert_eq!(c.apply_logc4_pixel(px), t.c8_logc4);
    }
    let _ = CurveSet::default();
}
