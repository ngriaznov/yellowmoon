//! Lens' fast method (the row segments of `ym_color::blur::lens`, BL-07b) against the exact aperture (`docs/33-blur.md` sections
//! 3.5, 3.7 and 8).
//!
//! * `R <= 8` px is the exact kernel bit for bit; the row table sums to 1 and holds the aperture's area;
//! * the method's own properties: a flat field stays flat (to the picture's edge and corner), nothing from outside the mask
//!   reaches inside with Respect mask edges on, a window of the reach equals the whole frame bit for bit;
//! * the method against the exact aperture (exact pixel coverage) on an edge, a disc, a bright light and a noise field, at
//!   `R` 12 to 240 px, round and hexagonal, and on a light for 5 to 9 blades at many rotations: held to the gate of 1e-2.

#![allow(clippy::needless_range_loop)]

use ym_color::blur::lens::{SubRow, reach, row_table, uses_rows};
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
fn up_to_eight_pixels_is_the_table_above_it_the_rows_and_the_reach_is_the_radius_plus_one() {
    for (r, rows) in [
        (0.0, false),
        (1.0, false),
        (8.0, false),
        (8.000_001, true),
        (48.0, true),
        (240.0, true),
    ] {
        assert_eq!(uses_rows(r), rows, "R = {r}");
    }
    assert!(!uses_rows(f64::NAN));
    for (r, want) in [(0.0, 1), (7.2, 9), (8.0, 9), (240.0, 241), (343.5, 345)] {
        assert_eq!(reach(r), want, "R = {r}");
    }
    assert_eq!(reach(f64::NAN), 1);
}

#[test]
fn the_row_table_sums_to_one_holds_the_apertures_area_and_stays_inside_its_rows() {
    for (r, blades, rot) in [
        (12.0, 0, 0.0),
        (60.0, 6, 25.0),
        (240.0, 7, 0.0),
        (9.3, 5, 17.0),
        (30.0, 4, 45.0),
        (30.0, 8, 22.5),
    ] {
        let t = row_table(r, blades, rot);
        assert!(!t.is_empty(), "R {r}");
        println!(
            "R {r} blades {blades} rotation {rot}: {} sub-rows over {} rows = {:.2} a row",
            t.len(),
            2 * r.ceil() as usize + 1,
            t.len() as f64 / (2 * r.ceil() as usize + 1) as f64
        );
        let area: f64 = t
            .iter()
            .map(|e| e.w * ((e.kb as f64 + e.fb) - (e.ka as f64 + e.fa)))
            .sum();
        assert!((area - 1.0).abs() < 1e-12, "R {r}: {area}");
        let half = r.ceil() as i32 + 1;
        let mut last = i32::MIN;
        for e in &t {
            assert!(e.w > 0.0 && (0.0..1.0).contains(&e.fa) && (0.0..1.0).contains(&e.fb));
            assert!(e.dy.abs() <= half && e.dy >= last, "R {r}: rows ascend");
            assert!(e.ka.abs() <= half + 1 && e.kb.abs() <= half + 1);
            last = e.dy;
        }
        // The table's sub-rows weigh what the exact table's rows do: the row `dy`'s share of the area.
        let exact = aperture_table(r, blades, rot);
        for dy in -exact.half..=exact.half {
            let want: f64 = (-exact.half..=exact.half).map(|dx| exact.at(dx, dy)).sum();
            let got: f64 = t
                .iter()
                .filter(|e| i64::from(e.dy) == dy)
                .map(|e: &SubRow| e.w * ((e.kb as f64 + e.fb) - (e.ka as f64 + e.fa)))
                .sum();
            assert!(
                (got - want).abs() < 2e-3,
                "R {r} blades {blades} row {dy}: {got} against {want}"
            );
        }
    }
    // A round aperture's table is symmetric in x (its chord's ends are mirrors).
    for e in row_table(30.0, 0, 0.0) {
        let (a, b) = (e.ka as f64 + e.fa, e.kb as f64 + e.fb);
        assert!(
            (a + b - 1.0).abs() < 1e-12,
            "a chord of a disc is centred: {a} {b}"
        );
    }
    assert!(row_table(0.0, 0, 0.0).is_empty() && row_table(f64::NAN, 6, 0.0).is_empty());
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

/// The gate of `docs/33` section 3.7 for Lens: the fast method against the exact aperture, `max |fast - exact| / the peak of
/// the exact result`, in linear light, on a line and a diagonal through the feature across the aperture's rim, `<= 1e-2`.
const GATE: f64 = 1e-2;

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

/// The error of the fast method against the exact aperture on the four scenes at `radius` (the panel's number, `u = 6000` px):
/// `[edge, disc, light, noise]`.
fn measure(radius: f64, blades: u32, rotation_deg: f64) -> [f64; 4] {
    let (fw, fh) = (6000usize, 4000usize);
    let (cx, cy) = (3000i64, 2000i64);
    let p = BlurParams {
        kind: BlurKind::Lens {
            radius,
            blades,
            rotation_deg,
            highlights: 0.0,
        },
        respect_mask_edges: true,
    };
    let r = radius_px(radius, fw as f64);
    let reach = reach_px(&p, fw, fh) as i64;
    let half = r.ceil() as i64 + 3;
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
    let t = aperture_table(r, blades, rotation_deg);
    let side = (2 * t.half + 1) as usize;
    let mut out = [0.0; 4];
    for (si, (_, scene)) in scenes(r).into_iter().enumerate() {
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
                    let k = t.weights[(dy + t.half) as usize * side + (dx + t.half) as usize];
                    if k != 0.0 {
                        acc += k * scene(px + dx, py + dy)[1];
                    }
                }
            }
            let f = fast[(py - y0) as usize * ww + (px - x0) as usize][1];
            err = err.max((f - acc).abs());
            peak = peak.max(acc);
        }
        out[si] = err / peak;
    }
    out
}

#[test]
fn the_row_segments_against_the_exact_aperture_on_an_edge_a_disc_a_light_and_noise() {
    let mut worst = [0.0f64; 4];
    for blades in [0u32, 6] {
        for radius in [5.0, 20.0, 60.0, 100.0] {
            let r = radius_px(radius, 6000.0);
            let e = measure(radius, blades, 10.0);
            println!(
                "blades {blades} radius {radius:5} R {r:6.1}: edge {:.2e} disc {:.2e} light {:.2e} noise {:.2e} of the peak",
                e[0], e[1], e[2], e[3]
            );
            for i in 0..4 {
                worst[i] = worst[i].max(e[i]);
            }
        }
    }
    for (i, name) in ["edge", "disc", "light", "noise"].iter().enumerate() {
        println!("worst {name}: {:.2e} (the gate is {GATE:.0e})", worst[i]);
        assert!(worst[i] < GATE, "{name}: {} against {GATE}", worst[i]);
    }
}

#[test]
fn a_light_of_4_by_4_pixels_is_within_the_gate_for_5_to_9_blades_at_any_rotation() {
    // Radius 6.25 on the 6000 px frame is R = 15 px.
    let mut worst = 0.0f64;
    for blades in [5u32, 6, 7, 8, 9] {
        for rot in [0.0, 7.0, 22.5, 45.0, 80.0] {
            let e = measure(6.25, blades, rot);
            worst = worst.max(e[2]);
            assert!(
                e[2] < GATE,
                "blades {blades} rotation {rot}: light {:.2e}",
                e[2]
            );
        }
    }
    println!("5 to 9 blades, 5 rotations, R 15 px: the light's worst {worst:.2e}");
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
        ("R 4.8 px (Radius 10), the exact table", 10.0),
        ("R 7.9 px (Radius 16.5), the exact table", 16.5),
        ("R 9.6 px (Radius 20), rows", 20.0),
        ("R 48 px (Radius 100), rows", 100.0),
    ] {
        let p = params(radius, 6, true);
        let t = std::time::Instant::now();
        let fast = blur::apply_fast(&v, &m, w, h, &p, &frame).expect("fast");
        let ms = t.elapsed().as_secs_f64() * 1000.0 / mp;
        assert_eq!(fast.len(), w * h);
        println!("{name}: the fast method {ms:.0} ms per MP on one thread");
    }
}
