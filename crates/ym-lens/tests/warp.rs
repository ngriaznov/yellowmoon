//! L4: the anamorphic bend, the lens's distortion and the zoom (`docs/21` section 4, L4).

mod common;

use common::{camera, library, plan};
use ym_lens::{Axis, LensKind, LensPlan, LensWarp};

const CANVAS: (f64, f64) = (6000.0, 4000.0);

fn warp(kind: LensKind, edges: f64, anamorphic: f64) -> (LensPlan, LensWarp) {
    let p = plan(kind, 0.0, edges, anamorphic, 6000, 4000, Axis::X);
    let w = p.warp(CANVAS.0, CANVAS.1);
    (p, w)
}

/// Does canvas point `(x, y)` map inside the photo with a margin of 1 px?
fn inside(w: &LensWarp, x: f64, y: f64) -> bool {
    let (px, py) = w.canvas_to_photo(x, y);
    px >= 1.0 - 1e-9
        && px <= CANVAS.0 - 1.0 + 1e-9
        && py >= 1.0 - 1e-9
        && py <= CANVAS.1 - 1.0 + 1e-9
}

fn edge_points() -> Vec<(f64, f64)> {
    let mut v = Vec::new();
    for i in 0..=200 {
        let t = i as f64 / 200.0;
        v.extend([
            (t * CANVAS.0, 0.0),
            (t * CANVAS.0, CANVAS.1),
            (0.0, t * CANVAS.1),
            (CANVAS.0, t * CANVAS.1),
        ]);
    }
    v
}

#[test]
fn identity_at_amounts_zero() {
    for kind in LensKind::ALL {
        let (_, w) = warp(kind, 0.0, 0.0);
        assert!(w.is_identity());
        assert_eq!(w.zoom(), 1.0);
        assert_eq!(w.canvas_to_photo(123.5, 77.25), (123.5, 77.25));
        assert_eq!(w.key(), "warp:none");
    }
    // Flares alone do not move a pixel either.
    let p = plan(LensKind::Vintage, 80.0, 0.0, 0.0, 6000, 4000, Axis::X);
    assert!(p.warp(6000.0, 4000.0).is_identity());
}

#[test]
fn the_round_trip_is_exact_to_a_billionth_of_a_pixel() {
    for kind in [
        LensKind::Default,
        LensKind::Vintage,
        LensKind::Petzval,
        LensKind::Detuned,
    ] {
        for (edges, anamorphic) in [
            (0.0, 100.0),
            (0.0, 37.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (55.0, 80.0),
        ] {
            let (_, w) = warp(kind, edges, anamorphic);
            assert!(!w.is_identity());
            let mut worst = 0.0f64;
            for i in 0..=30 {
                for j in 0..=20 {
                    let (x, y) = (i as f64 * CANVAS.0 / 30.0, j as f64 * CANVAS.1 / 20.0);
                    let (cx, cy) = w.photo_to_canvas(x, y);
                    let (px, py) = w.canvas_to_photo(cx, cy);
                    worst = worst.max((px - x).abs()).max((py - y).abs());
                }
            }
            assert!(
                worst < 1e-9,
                "{kind} edges {edges} anamorphic {anamorphic}: {worst:e} px"
            );
            // And the other way, from the canvas.
            let (px, py) = w.canvas_to_photo(1234.5, 2345.5);
            let (cx, cy) = w.photo_to_canvas(px, py);
            assert!((cx - 1234.5).abs() < 1e-9 && (cy - 2345.5).abs() < 1e-9);
        }
    }
}

#[test]
fn the_zoom_leaves_no_corner_empty_and_is_minimal() {
    for kind in [
        LensKind::Default,
        LensKind::Vintage,
        LensKind::Petzval,
        LensKind::Detuned,
    ] {
        for (edges, anamorphic) in [
            (0.0, 100.0),
            (0.0, 30.0),
            (100.0, 100.0),
            (100.0, 0.0),
            (60.0, 60.0),
        ] {
            let (_, w) = warp(kind, edges, anamorphic);
            let z = w.zoom();
            assert!(
                (1.0..1.3).contains(&z),
                "{kind} edges {edges} anamorphic {anamorphic}: zoom {z}"
            );
            for (x, y) in edge_points() {
                assert!(
                    inside(&w, x, y),
                    "{kind} edges {edges} anamorphic {anamorphic}: ({x}, {y}) maps outside"
                );
            }
            if z > 1.0 {
                // Minimal: a zoom of 0.999 z, which is the same map read at canvas points 1/0.999 times as far
                // from the centre, leaves an edge point outside the photo.
                let (cx, cy) = (CANVAS.0 / 2.0, CANVAS.1 / 2.0);
                let outside = edge_points().iter().any(|&(x, y)| {
                    let (sx, sy) = (cx + (x - cx) / 0.999, cy + (y - cy) / 0.999);
                    !inside(&w, sx, sy)
                });
                assert!(
                    outside,
                    "{kind} edges {edges} anamorphic {anamorphic}: zoom {z} is not minimal"
                );
            }
        }
    }
}

#[test]
fn the_zoom_is_shared_by_every_scale() {
    let (p, w) = warp(LensKind::Vintage, 40.0, 100.0);
    let small = w.rescaled(1500.0, 1000.0);
    assert_eq!(small.zoom(), w.zoom());
    assert_eq!(small.canvas_size(), (1500.0, 1000.0));
    // A point a quarter of the way across lands at a quarter of the position.
    let (x, y) = w.canvas_to_photo(1200.0, 800.0);
    let (xs, ys) = small.canvas_to_photo(300.0, 200.0);
    assert!((x / 4.0 - xs).abs() < 1e-9 && (y / 4.0 - ys).abs() < 1e-9);
    assert_ne!(w.key(), small.key());
    let _ = p;
}

#[test]
fn a_horizontal_line_near_the_top_bows_and_a_vertical_one_stays_straight() {
    // Anamorphic 100 is q = 2 on the synthetic attachment. Edges 0: no distortion, only the bend.
    let (_, w) = warp(LensKind::Default, 0.0, 100.0);
    // A line a tenth of the way down the photo, seen on the canvas at its middle and its ends.
    let y_photo = 400.0;
    let mid = w.photo_to_canvas(3000.0, y_photo).1;
    let left = w.photo_to_canvas(300.0, y_photo).1;
    let right = w.photo_to_canvas(5700.0, y_photo).1;
    assert!(
        (mid - left).abs() > 0.5 && (mid - right).abs() > 0.5,
        "mid {mid}, ends {left} and {right}"
    );
    assert!((left - right).abs() < 1e-6, "the bow is symmetric");
    // Near the top the line is pulled towards the centre line at its ends... (y' = y (1 - beta x^2) with y < 0).
    assert!(
        left > mid,
        "ends are lower than the middle at the top: {left} vs {mid}"
    );
    // The same near the bottom, mirrored.
    let bottom_mid = w.photo_to_canvas(3000.0, 3600.0).1;
    let bottom_left = w.photo_to_canvas(300.0, 3600.0).1;
    assert!(bottom_left < bottom_mid);
    // A vertical line a tenth of the way across stays at one column.
    let x_photo = 600.0;
    let xs: Vec<f64> = (0..=10)
        .map(|i| w.photo_to_canvas(x_photo, 100.0 + 380.0 * f64::from(i)).0)
        .collect();
    let spread =
        xs.iter().copied().fold(f64::MIN, f64::max) - xs.iter().copied().fold(f64::MAX, f64::min);
    assert!(spread < 0.5, "the vertical line moves by {spread} px");
    // Things near the left and right edges look wider: the horizontal stretch grows with x.
    let (x_in, _) = w.photo_to_canvas(4500.0, 2000.0);
    let (x_mid, _) = w.photo_to_canvas(3500.0, 2000.0);
    assert!(
        x_in - x_mid > 1000.0 * w.zoom(),
        "stretch: {}",
        x_in - x_mid
    );
}

#[test]
fn the_lens_distortion_scales_with_edges_and_the_bend_with_anamorphic() {
    // Edges alone: a barrel/pincushion that keeps lines through the centre straight.
    let (_, w) = warp(LensKind::Vintage, 100.0, 0.0);
    let (x1, y1) = w.photo_to_canvas(3000.0, 2000.0);
    assert!(
        (x1 - 3000.0).abs() < 1e-9 && (y1 - 2000.0).abs() < 1e-9,
        "the centre stays"
    );
    let (_, half) = warp(LensKind::Vintage, 50.0, 0.0);
    let corner = w.photo_to_canvas(0.0, 0.0);
    let corner_half = half.photo_to_canvas(0.0, 0.0);
    // Radial displacement (before the zoom) is about twice as large at Edges 100 as at 50.
    let d = |c: (f64, f64), z: f64| ((c.0 - 3000.0) / z).hypot((c.1 - 2000.0) / z);
    let full = d(corner, w.zoom());
    let part = d(corner_half, half.zoom());
    let r0 = 3000.0f64.hypot(2000.0);
    assert!(
        ((full - r0) / (part - r0) - 2.0).abs() < 0.2,
        "{full} {part} {r0}"
    );
    // No zoom is needed for a pure barrel of the wrong sign, and the map never leaves the photo empty.
    for (x, y) in edge_points() {
        assert!(inside(&w, x, y));
    }
    // Different amounts, different keys.
    assert_ne!(w.key(), half.key());
    let _ = (library(), camera());
}
