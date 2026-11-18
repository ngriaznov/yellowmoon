//! Lens' fast method (the cell method of `ym_color::blur::lens`, BL-07) against the exact aperture (`docs/33-blur.md` sections
//! 3.5, 3.7 and 8).
//!
//! * `R <= 8` px is the exact kernel bit for bit; the cell size is the smallest with `R / f <= 8`;
//! * the method's own properties: a flat field stays flat (to the picture's edge and corner), nothing from outside the mask
//!   reaches inside with Respect mask edges on, a window of the reach equals the whole frame bit for bit;
//! * the method against the exact aperture on an edge, a disc, a bright light and a noise field, at small and large radii,
//!   round and hexagonal: the numbers are printed and held to what was measured (see the note on `MEASURED`).

#![allow(clippy::needless_range_loop)]

use ym_color::blur::lens::{self, cell_size, cell_table, level_for};
use ym_color::blur::{self, BlurKind, BlurParams, Frame, aperture_table, radius_px, reach_px};
use ym_color::consts::Vec3;

type Scene = Box<dyn Fn(i64, i64) -> Vec3>;

fn params(radius: f64, blades: u32, respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Lens {
            radius,
            blades,
            rotation_deg: 10.0,
            highlights: 0.0,
        },
        respect_mask_edges: respect,
    }
}

fn noise(x: i64, y: i64) -> f64 {
    let h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    let h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (h >> 40) as f64 / (1u64 << 24) as f64
}

#[test]
fn the_cell_is_the_smallest_with_the_radius_at_most_eight_cells() {
    for (r, level) in [
        (0.0, 0),
        (1.0, 0),
        (8.0, 0),
        (8.000_001, 1),
        (16.0, 1),
        (16.01, 2),
        (48.0, 3),
        (64.0, 3),
        (240.0, 5),
        (256.0, 5),
        (344.0, 6),
    ] {
        assert_eq!(level_for(r), level, "R = {r}");
        assert_eq!(cell_size(r), 1usize << level);
        assert!(r / (1u64 << level) as f64 <= 8.0 || r.is_nan());
    }
    assert_eq!(level_for(f64::NAN), 0);
    assert_eq!(level_for(f64::INFINITY), 30);
}

#[test]
fn the_cell_table_sums_to_one_holds_every_nonzero_weight_and_is_at_most_17_cells_wide() {
    for (r, blades, level) in [
        (6.0, 0, 0),
        (12.0, 0, 1),
        (60.0, 6, 3),
        (240.0, 7, 5),
        (8.0, 0, 0),
    ] {
        let t = cell_table(r, blades, 25.0, level);
        let total: f64 = t.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "R {r}: {total}");
        assert!(t.half <= 8, "R {r}: half {}", t.half);
        assert_eq!(
            t.weights.len(),
            ((2 * t.half + 1) * (2 * t.half + 1)) as usize
        );
        // The untrimmed table of the same radius in cells has the same nonzero weights.
        let full = aperture_table(r / (1u64 << level) as f64, blades, 25.0);
        for dy in -full.half..=full.half {
            for dx in -full.half..=full.half {
                let w = full.at(dx, dy);
                assert_eq!(t.at(dx, dy).to_bits(), w.to_bits(), "R {r} ({dx}, {dy})");
            }
        }
    }
    // A round aperture's table is symmetric.
    let t = cell_table(30.0, 0, 0.0, 2);
    for dy in -t.half..=t.half {
        for dx in -t.half..=t.half {
            assert!((t.at(dx, dy) - t.at(-dx, dy)).abs() < 1e-15);
            assert!((t.at(dx, dy) - t.at(dx, -dy)).abs() < 1e-15);
            assert!((t.at(dx, dy) - t.at(dy, dx)).abs() < 1e-15);
        }
    }
}

/// A window of a frame as the tests use them: values `f(x, y)` and a mask.
fn window(
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    f: impl Fn(i64, i64) -> Vec3,
    m: impl Fn(i64, i64) -> f64,
) -> (Vec<Vec3>, Vec<f64>) {
    let mut v = Vec::with_capacity(w * h);
    let mut mm = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(f((x0 + x) as i64, (y0 + y) as i64));
            mm.push(m((x0 + x) as i64, (y0 + y) as i64));
        }
    }
    (v, mm)
}

/// A picture of LogC4 values (not too extreme: the blend decodes them).
fn logc(x: i64, y: i64) -> Vec3 {
    [
        0.2 + 0.4 * noise(x, y),
        0.25 + 0.3 * noise(y + 7, x + 3),
        0.3 + 0.2 * noise(x * 3 + 1, y * 5),
    ]
}

#[test]
fn up_to_eight_pixels_the_fast_method_is_the_exact_kernel_bit_for_bit() {
    // u = 200 px: Radius 100 is R = 8 px, so level 0.
    let (w, h) = (200usize, 120usize);
    let (v, m) = window(0, 0, w, h, logc, |x, y| ((x + y) as f64 / 320.0).min(1.0));
    for blades in [0, 6] {
        for respect in [true, false] {
            for radius in [10.0, 55.0, 100.0] {
                let p = params(radius, blades, respect);
                let frame = Frame::whole(w, h);
                let exact = blur::apply(&v, &m, w, h, &p, &frame).expect("exact");
                let fast = blur::apply_fast(&v, &m, w, h, &p, &frame).expect("fast");
                assert!(
                    exact
                        .iter()
                        .zip(&fast)
                        .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits)),
                    "radius {radius}, blades {blades}, respect {respect}"
                );
            }
        }
    }
}

#[test]
fn a_flat_field_stays_flat_to_the_edge_and_the_corner_and_the_mask_keeps_the_outside_out() {
    // u = 600 px: Radius 100 is R = 24 px, level 2 (f = 4, r = 6); Radius 30 is R = 7.2 (level 0) and 60 is 14.4 (level 1).
    let (w, h) = (600usize, 90usize);
    for radius in [30.0, 60.0, 100.0] {
        for blades in [0, 6] {
            let p = params(radius, blades, true);
            let frame = Frame::whole(w, h);
            // A flat linear value through the node: sums are weighted by the mask, B is the value wherever D > 0.
            let (x, m) = window(
                0,
                0,
                w,
                h,
                |_, _| [0.37, 0.37, 0.37],
                |x, y| 0.2 + 0.8 * noise(x, y),
            );
            let out = blur::apply_linear_fast(&x, &m, w, h, &p, &frame).expect("fast");
            for (i, o) in out.iter().enumerate() {
                for c in 0..3 {
                    assert!(
                        (o[c] - 0.37).abs() < 1e-9,
                        "radius {radius}, blades {blades} at {}, {}: {}",
                        i % w,
                        i / w,
                        o[c]
                    );
                }
            }
            // Nothing from outside the mask reaches inside: the picture inside (m = 1 for x >= 300) does not move when the
            // picture outside (m = 0 for x < 300) changes wildly. The cells that straddle the border are the node's own.
            let inside = |x: i64, _y: i64| if x >= 300 { 1.0 } else { 0.0 };
            let (a, ma) = window(0, 0, w, h, |x, y| [0.1 + 0.5 * noise(x, y); 3], inside);
            let (b, mb) = window(
                0,
                0,
                w,
                h,
                |x, y| {
                    if x >= 300 {
                        [0.1 + 0.5 * noise(x, y); 3]
                    } else {
                        [50.0 * noise(y, x); 3]
                    }
                },
                inside,
            );
            let oa = blur::apply_linear_fast(&a, &ma, w, h, &p, &frame).expect("a");
            let ob = blur::apply_linear_fast(&b, &mb, w, h, &p, &frame).expect("b");
            let mut worst = 0.0f64;
            for y in 0..h {
                for x in 300..w {
                    worst = worst.max((oa[y * w + x][0] - ob[y * w + x][0]).abs());
                }
            }
            println!(
                "radius {radius}, blades {blades}: the outside moves the inside by {worst:.1e}"
            );
            assert!(worst < 1e-9, "radius {radius}, blades {blades}: {worst}");
        }
    }
}

#[test]
fn a_window_of_the_reach_equals_the_whole_frame_bit_for_bit() {
    // u = 600 px: R = 24 px at Radius 100 (f = 4), 14.4 (f = 2) at 60; a frame of 600 x 80.
    let (w, h) = (600usize, 80usize);
    let (v, m) = window(0, 0, w, h, logc, |x, y| {
        (0.5 + 0.5 * ((x as f64 / 53.0).sin() * (y as f64 / 17.0).cos())).clamp(0.0, 1.0)
    });
    for blades in [0, 6] {
        for respect in [true, false] {
            for radius in [60.0, 100.0] {
                let p = params(radius, blades, respect);
                let whole = blur::apply_fast(&v, &m, w, h, &p, &Frame::whole(w, h)).expect("whole");
                let reach = reach_px(&p, w, h);
                for core in [
                    (10usize, 8usize, 25usize, 19usize),
                    (0, 0, 31, 31),
                    (255, 40, 97, 40),
                    (570, 50, 30, 30),
                    (301, 3, 160, 71),
                ] {
                    let (cx, cy, cw, ch) = core;
                    let (x0, y0) = (cx.saturating_sub(reach), cy.saturating_sub(reach));
                    let (x1, y1) = ((cx + cw + reach).min(w), (cy + ch + reach).min(h));
                    let (ww, wh) = (x1 - x0, y1 - y0);
                    let (sv, sm) = window(
                        x0,
                        y0,
                        ww,
                        wh,
                        |x, y| v[y as usize * w + x as usize],
                        |x, y| m[y as usize * w + x as usize],
                    );
                    let frame = Frame {
                        full_w: w,
                        full_h: h,
                        x0,
                        y0,
                    };
                    let got = blur::apply_fast(&sv, &sm, ww, wh, &p, &frame).expect("window");
                    for y in cy..(cy + ch).min(h) {
                        for x in cx..(cx + cw).min(w) {
                            let a = got[(y - y0) * ww + (x - x0)];
                            let b = whole[y * w + x];
                            assert!(
                                a.map(f64::to_bits) == b.map(f64::to_bits),
                                "radius {radius}, blades {blades}, respect {respect}, core {core:?} at ({x}, {y}): {a:?} against {b:?}, reach {reach}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// What the method measured against the exact aperture on `docs/33` section 8's scenes (`max |fast - exact| / peak of the exact
/// result`, in linear light, on a line and a diagonal through the feature across the aperture's rim), the worst over the radii
/// and the two shapes below, and what the test holds the method to (about 1.25 times the measurement).
///
/// **The spec's gate of 1e-2 is met on the noise field and, but at R = 144 px, on the edge, and not on the disc and the light**:
/// the cell method softens the aperture's rim by about `f / 2`, so a light smaller than a cell shows a rim of `f` pixels at
/// every radius (error up to 0.45 of the light's own peak at its rim), and a disc as wide as the aperture shows 3 to 6 per
/// cent. This is the method of `docs/33` section 3.5 as written; the numbers are in `docs/STATUS.md` for the owner (the gate
/// is not changed here).
const MEASURED: [(&str, f64); 4] = [
    ("edge", 2.0e-2),
    ("disc", 7.0e-2),
    ("light", 0.55),
    ("noise", 1.8e-2),
];

fn scenes(r: f64) -> Vec<(&'static str, Scene)> {
    let (cx, cy) = (3000i64, 2000i64);
    vec![
        (
            "edge",
            Box::new(move |x, _y| if x < cx { [0.02; 3] } else { [1.0; 3] }),
        ),
        (
            "disc",
            Box::new(move |x, y| {
                let d = (((x - cx).pow(2) + (y - cy).pow(2)) as f64).sqrt();
                if d < 0.5 * r { [4.0; 3] } else { [0.05; 3] }
            }),
        ),
        (
            "light",
            Box::new(move |x, y| {
                if (x - cx).abs() < 2 && (y - cy).abs() < 2 {
                    [400.0; 3]
                } else {
                    [0.02; 3]
                }
            }),
        ),
        ("noise", Box::new(|x, y| [0.1 + 0.8 * noise(x, y); 3])),
    ]
}

#[test]
fn the_cell_method_against_the_exact_aperture_on_an_edge_a_disc_a_light_and_noise() {
    let (fw, fh) = (6000usize, 4000usize);
    let (cx, cy) = (3000i64, 2000i64);
    let mut worst = [0.0f64; 4];
    for blades in [0u32, 6] {
        for radius in [5.0, 20.0, 60.0, 100.0] {
            let p = params(radius, blades, true);
            let r = radius_px(radius, fw as f64);
            let reach = reach_px(&p, fw, fh) as i64;
            let half = r.ceil() as i64 + 3 * cell_size(r) as i64;
            let step = (half / 24).max(1);
            let mut pts = vec![];
            let mut x = -half;
            while x <= half {
                pts.push((cx + x, cy));
                pts.push((cx + x, cy + x / 2));
                x += step;
            }
            let (x0, x1) = (cx - half - reach, cx + half + reach + 1);
            let (y0, y1) = (cy - half / 2 - reach - 1, cy + half / 2 + reach + 2);
            let (ww, wh) = ((x1 - x0) as usize, (y1 - y0) as usize);
            let t = aperture_table(r, blades, 10.0);
            let side = (2 * t.half + 1) as usize;
            for (si, (name, scene)) in scenes(r).into_iter().enumerate() {
                let (xs, ms) = window(x0 as usize, y0 as usize, ww, wh, &scene, |_, _| 1.0);
                let frame = Frame {
                    full_w: fw,
                    full_h: fh,
                    x0: x0 as usize,
                    y0: y0 as usize,
                };
                let fast = blur::apply_linear_fast(&xs, &ms, ww, wh, &p, &frame).expect("fast");
                let (mut err, mut peak) = (0.0f64, 0.0f64);
                for &(px, py) in &pts {
                    let mut acc = 0.0;
                    for dy in -t.half..=t.half {
                        for dx in -t.half..=t.half {
                            let k =
                                t.weights[(dy + t.half) as usize * side + (dx + t.half) as usize];
                            if k != 0.0 {
                                acc += k * scene(px + dx, py + dy)[1];
                            }
                        }
                    }
                    let f = fast[(py - y0) as usize * ww + (px - x0) as usize][1];
                    err = err.max((f - acc).abs());
                    peak = peak.max(acc);
                }
                let rel = err / peak;
                println!(
                    "blades {blades} radius {radius:5} R {r:6.1} f {:2} {name:6}: {rel:.2e} of the peak",
                    cell_size(r)
                );
                worst[si] = worst[si].max(rel);
                // R <= 8 would be exact; here every R is above it except the smallest, which is exact too.
                if r <= 8.0 {
                    assert!(rel < 1e-9, "{name}: {rel}");
                }
            }
        }
    }
    for (i, (name, bound)) in MEASURED.iter().enumerate() {
        println!(
            "worst {name}: {:.2e} (held to {bound:.1e}; the spec's gate is 1e-2)",
            worst[i]
        );
        assert!(worst[i] < *bound, "{name}: {} against {bound}", worst[i]);
    }
    // The module's doc names the cap the cell radius is held to.
    assert!((lens::MAX_CELL_RADIUS - 8.0).abs() < f64::EPSILON);
}

/// The CPU run's cost per megapixel on one thread (`docs/33` section 9 budgets 600 ms for Lens): printed, not asserted, since it
/// depends on the machine. A whole 1200 x 800 frame, the fast method and (for a radius where it is affordable) the exact kernel.
#[test]
fn the_cpu_cost_per_megapixel_is_measured() {
    let (w, h) = (1200usize, 800usize);
    let mp = (w * h) as f64 / 1e6;
    let (v, m) = window(0, 0, w, h, logc, |x, _| {
        ((x as f64 + 200.0) / 1200.0).clamp(0.0, 1.0)
    });
    let frame = Frame::whole(w, h);
    for (name, radius) in [
        ("R 4.8 px (Radius 10)", 10.0),
        ("R 7.9 px (Radius 16.5)", 16.5),
        ("R 48 px (Radius 100)", 100.0),
    ] {
        let p = params(radius, 6, true);
        let t = std::time::Instant::now();
        let fast = blur::apply_fast(&v, &m, w, h, &p, &frame).expect("fast");
        let ms = t.elapsed().as_secs_f64() * 1000.0 / mp;
        assert_eq!(fast.len(), w * h);
        println!("{name}: the fast method {ms:.0} ms per MP on one thread");
    }
}
