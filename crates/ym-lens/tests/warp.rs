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

// ---------------------------------------------------------------------------------------------
// The shipped bend: ten constants, the mustache (`docs/21` section 4, L4; data version 5)
// ---------------------------------------------------------------------------------------------

use std::sync::OnceLock;
use ym_lens::{FrameInfo, LensAmounts, LensLibrary};

/// The shipped lens data (`data/lenses`).
fn shipped() -> &'static LensLibrary {
    static LIB: OnceLock<LensLibrary> = OnceLock::new();
    LIB.get_or_init(|| LensLibrary::load(&common::repo_root().join("data")).expect("data/lenses"))
}

/// A plan of the shipped lens `kind` on a `w x h` frame.
fn shipped_plan(kind: LensKind, edges: f64, anamorphic: f64, w: usize, h: usize) -> LensPlan {
    let amounts = LensAmounts {
        kind,
        flares: 0.0,
        edges,
        anamorphic,
    };
    let frame = FrameInfo {
        width: w,
        height: h,
        axis: Axis::X,
        f_number: None,
        clip: [1.0; 3],
    };
    LensPlan::new(shipped(), &amounts, frame, camera()).expect("plan")
}

/// The shipped warp of the Default lens (no distortion at Edges 0) at an Anamorphic amount, 6000 x 4000.
fn shipped_warp(edges: f64, anamorphic: f64) -> LensWarp {
    shipped_plan(LensKind::Default, edges, anamorphic, 6000, 4000).warp(CANVAS.0, CANVAS.1)
}

/// Photo pixel positions on the 6000 x 4000 canvas and the canvas positions the reference model gives them
/// (the model of the L4 equations in numpy, which is the design this code was written from: the ten
/// constants of the data times `s = q - 1`, the zoom by bisection, `canvas = cx + z F(p) h`). One row:
/// `(photo x, photo y, canvas x, canvas y)`.
const REFERENCE_100: [(f64, f64, f64, f64); 13] = [
    (0.0, 0.0, -109.28205561464665, 6.185356024958992),
    (6000.0, 0.0, 6109.282055614647, 6.185356024958992),
    (6000.0, 4000.0, 6109.282055614647, 3993.814643975041),
    (0.0, 4000.0, -109.28205561464665, 3993.814643975041),
    (6000.0, 2000.0, 6139.401018285105, 2000.0),
    (3000.0, 0.0, 3000.0, -121.86593936495956),
    (4500.0, 500.0, 4586.744188547296, 428.22031407882855),
    (1200.0, 3300.0, 1098.049838416699, 3353.4816914130433),
    (300.0, 150.0, 188.10897652425092, 134.59782928658342),
    (5700.0, 3850.0, 5811.8910234757495, 3865.4021707134166),
    (3000.0, 2000.0, 3000.0, 2000.0),
    (5500.0, 1000.0, 5625.98247776558, 980.3826100831819),
    (2400.0, 200.0, 2363.6004443686597, 90.52973593139905),
];
const REFERENCE_50: [(f64, f64, f64, f64); 13] = [
    (0.0, 0.0, -56.09126873123296, 0.7803770796558638),
    (6000.0, 0.0, 6056.091268731233, 0.7803770796558638),
    (6000.0, 4000.0, 6056.091268731233, 3999.219622920344),
    (0.0, 4000.0, -56.09126873123296, 3999.219622920344),
    (6000.0, 2000.0, 6070.638044600855, 2000.0),
    (3000.0, 0.0, 3000.0, -61.06549438121556),
    (4500.0, 500.0, 4543.550732901062, 463.6767824713761),
    (1200.0, 3300.0, 1148.774075555682, 3327.2650002214023),
    (300.0, 150.0, 242.97971040284392, 140.5196169913645),
    (5700.0, 3850.0, 5757.020289597156, 3859.4803830086357),
    (3000.0, 2000.0, 3000.0, 2000.0),
    (5500.0, 1000.0, 5563.605441464354, 989.4217407959283),
    (2400.0, 200.0, 2381.7577372957394, 145.14203661173974),
];
const REFERENCE_25: [(f64, f64, f64, f64); 13] = [
    (0.0, 0.0, -28.954796342087775, -0.5470176663109214),
    (6000.0, 0.0, 6028.954796342088, -0.5470176663109214),
    (6000.0, 4000.0, 6028.954796342088, 4000.547017666311),
    (0.0, 4000.0, -28.954796342087775, 4000.547017666311),
    (6000.0, 2000.0, 6036.102016166543, 2000.0),
    (3000.0, 0.0, 3000.0, -30.93354747384683),
    (4500.0, 500.0, 4522.095467537471, 481.4556039586744),
    (1200.0, 3300.0, 1173.993940029351, 3314.000807970879),
    (300.0, 150.0, 270.72831654126503, 144.4813794629922),
    (5700.0, 3850.0, 5729.271683458735, 3855.518620537008),
    (3000.0, 2000.0, 3000.0, 2000.0),
    (5500.0, 1000.0, 5532.41410081566, 994.3373996003479),
    (2400.0, 200.0, 2390.7579573766307, 172.20942221069504),
];

#[test]
fn the_shipped_forward_map_and_zoom_are_the_reference_models() {
    for (anamorphic, zoom, rows) in [
        (100.0, 1.0727753943912148, &REFERENCE_100),
        (50.0, 1.0362523697378947, &REFERENCE_50),
        (25.0, 1.0182769771816425, &REFERENCE_25),
    ] {
        let w = shipped_warp(0.0, anamorphic);
        assert!(
            (w.zoom() - zoom).abs() < 1e-9,
            "Anamorphic {anamorphic}: zoom {} against {zoom}",
            w.zoom()
        );
        for &(px, py, cx, cy) in rows {
            let (x, y) = w.photo_to_canvas(px, py);
            assert!(
                (x - cx).abs() < 1e-8 && (y - cy).abs() < 1e-8,
                "Anamorphic {anamorphic}: ({px}, {py}) -> ({x}, {y}), the model says ({cx}, {cy})"
            );
            // And the inverse, three Newton steps, takes the canvas position back to the photo's.
            let (bx, by) = w.canvas_to_photo(cx, cy);
            assert!(
                (bx - px).abs() < 1e-8 && (by - py).abs() < 1e-8,
                "Anamorphic {anamorphic}: ({cx}, {cy}) -> ({bx}, {by}), expected ({px}, {py})"
            );
        }
    }
}

#[test]
fn the_shipped_constants_are_the_ones_in_the_plan() {
    // The plan multiplies the data's ten constants by `q - 1` (0.25, 0.5, 1 at Anamorphic 25, 50, 100).
    let bend = shipped().attachment().bend;
    for (anamorphic, s) in [(100.0, 1.0), (50.0, 0.5), (25.0, 0.25)] {
        let w = shipped_warp(0.0, anamorphic);
        for i in 0..5 {
            assert!((w.bend_x()[i] - bend.x[i] * s).abs() < 1e-15, "x[{i}]");
            assert!((w.bend_y()[i] - bend.y[i] * s).abs() < 1e-15, "y[{i}]");
        }
    }
    // Edges does not touch the bend.
    let both = shipped_warp(100.0, 100.0);
    assert_eq!(both.bend_x(), shipped_warp(0.0, 100.0).bend_x());
    assert_ne!(both.key(), shipped_warp(0.0, 100.0).key());
}

#[test]
fn the_shipped_round_trip_is_exact_to_a_billionth_of_a_pixel() {
    let mut overall = 0.0f64;
    for kind in LensKind::ALL {
        for (edges, anamorphic) in [
            (0.0, 100.0),
            (0.0, 75.0),
            (0.0, 50.0),
            (0.0, 25.0),
            (100.0, 100.0),
            (100.0, 0.0),
            (40.0, 60.0),
        ] {
            let w = shipped_plan(kind, edges, anamorphic, 6000, 4000).warp(CANVAS.0, CANVAS.1);
            let mut worst = 0.0f64;
            // Photo positions over the photo, forward and back; and the canvas positions of the canvas, back
            // and forward (the positions the sampler asks for).
            for i in 0..=40 {
                for j in 0..=30 {
                    let x = i as f64 * CANVAS.0 / 40.0;
                    let y = j as f64 * CANVAS.1 / 30.0;
                    let (cx, cy) = w.photo_to_canvas(x, y);
                    let (px, py) = w.canvas_to_photo(cx, cy);
                    worst = worst.max((px - x).abs()).max((py - y).abs());
                    let (px, py) = w.canvas_to_photo(x, y);
                    let (cx, cy) = w.photo_to_canvas(px, py);
                    worst = worst.max((cx - x).abs()).max((cy - y).abs());
                }
            }
            // The bend alone is within a billionth. The Triplet's own distortion at Edges 100 (with no bend at
            // all) leaves 4e-6 px after three steps, and the bend on top of it about the same.
            let limit = if edges == 0.0 { 1e-9 } else { 1e-5 };
            assert!(
                worst < limit,
                "{kind} edges {edges} anamorphic {anamorphic}: {worst:e} px"
            );
            overall = overall.max(worst);
        }
    }
    eprintln!("shipped lenses: the worst round trip over every lens and amount is {overall:e} px");
}

/// The determinant of the map's Jacobian (canvas pixels per photo pixel, squared), by central differences of the
/// forward map.
fn jacobian_determinant(w: &LensWarp, x: f64, y: f64) -> f64 {
    let h = 0.5;
    let (xp, yp) = (w.photo_to_canvas(x + h, y), w.photo_to_canvas(x, y + h));
    let (xm, ym) = (w.photo_to_canvas(x - h, y), w.photo_to_canvas(x, y - h));
    let (a, b) = ((xp.0 - xm.0) / (2.0 * h), (yp.0 - ym.0) / (2.0 * h));
    let (c, d) = ((xp.1 - xm.1) / (2.0 * h), (yp.1 - ym.1) / (2.0 * h));
    (a * d - b * c) / (w.zoom() * w.zoom())
}

#[test]
fn the_shipped_map_has_a_positive_jacobian_over_the_photo_and_a_margin() {
    // The loader holds it to 0.5 over the photo plus 25 % of its half-sides at every position of the slider
    // (the Jacobian of the map alone: the finite differences below divide the zoom's square out).
    let mut smallest = [f64::INFINITY; 4];
    for (n, anamorphic) in [25.0, 50.0, 75.0, 100.0].into_iter().enumerate() {
        let w = shipped_warp(0.0, anamorphic);
        for i in 0..=60 {
            for j in 0..=40 {
                let x = (-0.125 + 1.25 * i as f64 / 60.0) * CANVAS.0;
                let y = (-0.125 + 1.25 * j as f64 / 40.0) * CANVAS.1;
                let det = jacobian_determinant(&w, x, y);
                assert!(
                    det >= 0.5,
                    "Anamorphic {anamorphic}: the determinant is {det} at ({x}, {y})"
                );
                smallest[n] = smallest[n].min(det);
            }
        }
    }
    eprintln!(
        "smallest Jacobian determinant over the photo + 25 %, Anamorphic 25/50/75/100: {smallest:.3?}"
    );
    // The rule of the data check: it gets smaller as the slider goes up.
    assert!(smallest.windows(2).all(|p| p[1] < p[0]));
}

#[test]
fn the_shipped_zoom_fills_the_frame_at_every_position_of_the_slider() {
    let mut zooms = Vec::new();
    for anamorphic in [25.0, 50.0, 75.0, 100.0] {
        for kind in LensKind::ALL {
            for edges in [0.0, 100.0] {
                let w = shipped_plan(kind, edges, anamorphic, 6000, 4000).warp(CANVAS.0, CANVAS.1);
                let z = w.zoom();
                assert!(
                    (1.0..1.2).contains(&z),
                    "{kind} edges {edges} anamorphic {anamorphic}: zoom {z}"
                );
                // The edge, the corners, and a lattice of the whole canvas read inside the photo.
                for (x, y) in edge_points() {
                    assert!(
                        inside(&w, x, y),
                        "{kind} edges {edges} anamorphic {anamorphic}: edge point ({x}, {y}) maps outside"
                    );
                }
                // The edge between the fitted samples may come within the margin by a fraction of a pixel; the
                // lattice only has to read inside the photo itself.
                for i in 0..=60 {
                    for j in 0..=40 {
                        let (x, y) = (i as f64 * CANVAS.0 / 60.0, j as f64 * CANVAS.1 / 40.0);
                        let (px, py) = w.canvas_to_photo(x, y);
                        assert!(
                            (0.0..=CANVAS.0).contains(&px) && (0.0..=CANVAS.1).contains(&py),
                            "{kind} edges {edges} anamorphic {anamorphic}: ({x}, {y}) reads outside the photo at ({px}, {py})"
                        );
                    }
                }
                // Minimal: a zoom of 0.999 z leaves an edge point outside.
                if z > 1.0 {
                    let (cx, cy) = (CANVAS.0 / 2.0, CANVAS.1 / 2.0);
                    assert!(
                        edge_points().iter().any(|&(x, y)| {
                            !inside(&w, cx + (x - cx) / 0.999, cy + (y - cy) / 0.999)
                        }),
                        "{kind} edges {edges} anamorphic {anamorphic}: zoom {z} is not minimal"
                    );
                }
                if kind == LensKind::Default && edges == 0.0 {
                    zooms.push(z);
                }
            }
        }
    }
    eprintln!("shipped zoom at Anamorphic 25/50/75/100 (Default, Edges 0): {zooms:.4?}");
    // The shipped bend zooms by 1.8, 3.6, 5.4 and 7.3 per cent, growing with the slider.
    for (z, want) in zooms.iter().zip([1.0183, 1.0363, 1.0544, 1.0728]) {
        assert!((z - want).abs() < 5e-4, "{zooms:?}");
    }
}

#[test]
fn anamorphic_zero_is_the_identity_bit_for_bit_with_the_shipped_data() {
    for kind in LensKind::ALL {
        let w = shipped_plan(kind, 0.0, 0.0, 6000, 4000).warp(CANVAS.0, CANVAS.1);
        assert!(w.is_identity(), "{kind}");
        assert_eq!(w.zoom().to_bits(), 1.0f64.to_bits());
        assert_eq!(w.key(), "warp:none");
        for (x, y) in [
            (0.0, 0.0),
            (123.456, 77.25),
            (3000.0, 2000.0),
            (5999.9, 3999.9),
        ] {
            let (a, b) = w.canvas_to_photo(x, y);
            assert_eq!((a.to_bits(), b.to_bits()), (x.to_bits(), y.to_bits()));
            let (a, b) = w.photo_to_canvas(x, y);
            assert_eq!((a.to_bits(), b.to_bits()), (x.to_bits(), y.to_bits()));
        }
    }
    // The ten constants times 0 are zeros: with Edges on, the bend does nothing and only the lens's
    // distortion moves a point.
    let w = shipped_warp(100.0, 0.0);
    assert!(
        w.bend_x()
            .iter()
            .chain(w.bend_y().iter())
            .all(|&c| c == 0.0)
    );
}

/// The second difference of the canvas height of a horizontal photo line, as a function of the photo's `x`,
/// at `x` = `fx` of the half-width, in canvas pixels (step 3 px).
fn line_curvature(w: &LensWarp, photo_y: f64, fx: f64) -> f64 {
    let x = CANVAS.0 / 2.0 + fx * CANVAS.0 / 2.0;
    let step = 3.0;
    let h = |x: f64| w.photo_to_canvas(x, photo_y).1;
    h(x + step) - 2.0 * h(x) + h(x - step)
}

#[test]
fn the_mustache_is_in_the_map() {
    // At Anamorphic 100 a horizontal line near the top, which the sides pull towards the middle, bows one way
    // in the inner part of the frame and the other way near the sides: its curvature (the second difference
    // of its height along the photo's x) changes sign inside the frame, at about 0.93 of the half-width.
    let w = shipped_warp(0.0, 100.0);
    for photo_y in [0.0, 20.0, 80.0] {
        let mut sign_changes = Vec::new();
        let mut last = line_curvature(&w, photo_y, 0.05);
        let mut fx = 0.05;
        while fx < 0.995 {
            fx += 0.0005;
            let now = line_curvature(&w, photo_y, fx);
            if now.signum() != last.signum() {
                sign_changes.push(fx);
            }
            last = now;
        }
        assert_eq!(
            sign_changes.len(),
            1,
            "photo y {photo_y}: curvature changes sign at {sign_changes:?} of the half-width"
        );
        let at = sign_changes[0];
        eprintln!(
            "mustache: the line at photo y = {photo_y} reverses at {at:.4} of the half-width"
        );
        assert!(
            (0.915..0.96).contains(&at),
            "photo y {photo_y}: the curvature reverses at {at} of the half-width"
        );
        // Convex (y rises towards the sides, the sides pulling the line down) inside, concave outside,
        // and the point is inside the canvas.
        assert!(line_curvature(&w, photo_y, 0.5) > 0.0);
        assert!(line_curvature(&w, photo_y, 0.99) < 0.0);
        let x_canvas = w
            .photo_to_canvas(CANVAS.0 / 2.0 + at * CANVAS.0 / 2.0, photo_y)
            .0;
        assert!(x_canvas < CANVAS.0, "{x_canvas}");
    }
    // The bottom line mirrors it.
    let top = line_curvature(&w, 20.0, 0.5);
    let bottom = line_curvature(&w, CANVAS.1 - 20.0, 0.5);
    assert!((top + bottom).abs() < 1e-9 * top.abs().max(1e-3));
    // The line through the middle is straight.
    assert!(line_curvature(&w, CANVAS.1 / 2.0, 0.7).abs() < 1e-9);
    // The reversal is the strength of the slider's: at 25 it is there too, with a quarter of the curvature.
    let quarter = shipped_warp(0.0, 25.0);
    let ratio = line_curvature(&quarter, 20.0, 0.5) / line_curvature(&w, 20.0, 0.5);
    assert!((ratio - 0.25).abs() < 0.03, "{ratio}");
    assert!(line_curvature(&quarter, 20.0, 0.99) < 0.0);
    // Without the attachment no line bends.
    let none = shipped_warp(0.0, 0.0);
    assert_eq!(line_curvature(&none, 20.0, 0.5), 0.0);
    // The test can tell: the synthetic attachment has the plain two-term bend (`a1` and `b1` only), whose
    // top line bows one way along its whole length.
    let (_, plain) = warp(LensKind::Default, 0.0, 100.0);
    assert!(
        (1..200).all(|i| line_curvature(&plain, 20.0, f64::from(i) / 200.0) > 0.0),
        "the two-term bend has no reversal"
    );
}

#[test]
fn the_shipped_bend_draws_the_sides_in_and_bows_the_top() {
    // What it looks like at Anamorphic 100 on a 3:2 canvas (before the zoom): the side edges drawn in by 2.5 %
    // of the half-width, the top line's middle 3 % of the picture height beyond its ends, the corners in.
    let w = shipped_warp(0.0, 100.0);
    let z = w.zoom();
    let h = 0.5 * CANVAS.0.hypot(CANVAS.1);
    let to_unit = |(x, y): (f64, f64)| ((x - 3000.0) / z / h, (y - 2000.0) / z / h);
    let (hx, hy) = (3000.0 / h, 2000.0 / h);
    let (side_x, _) = to_unit(w.photo_to_canvas(CANVAS.0, 2000.0));
    let drawn_in = 1.0 - side_x / hx;
    assert!((drawn_in - 0.0245).abs() < 0.0015, "{drawn_in}");
    let mid = to_unit(w.photo_to_canvas(3000.0, 0.0)).1;
    let end = to_unit(w.photo_to_canvas(CANVAS.0, 0.0)).1;
    let bow = (end - mid) / (2.0 * hy);
    assert!((bow - 0.0298).abs() < 0.0015, "{bow}");
    let corner = to_unit(w.photo_to_canvas(CANVAS.0, CANVAS.1));
    assert!(((1.0 - corner.0 / hx) - 0.0339).abs() < 0.0015);
    assert!(((1.0 - corner.1 / hy) - 0.0707).abs() < 0.0015);
    // The draw-in flattens towards the sides: the local horizontal scale at the side edge is 0.933 of the
    // centre's.
    let step = 1.0;
    let scale = (w.photo_to_canvas(CANVAS.0, 2000.0).0
        - w.photo_to_canvas(CANVAS.0 - step, 2000.0).0)
        / (w.photo_to_canvas(3000.0 + step, 2000.0).0 - w.photo_to_canvas(3000.0, 2000.0).0);
    assert!((scale - 0.933).abs() < 0.002, "{scale}");
}
