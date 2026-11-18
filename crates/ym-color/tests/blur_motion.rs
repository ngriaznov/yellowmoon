//! Motion's fast method against the oracle (`docs/33-blur.md` 3.5, 3.7, 8, 11 item 50; BL-08): the sparse gather of the exact
//! line is the oracle's sum bit for bit (so it is far inside the bound of 3e-3), a window of the reach equals the frame, the
//! mask gate, the energy and the symmetry hold, and the spec's design of cells is measured to show why it was not built.

use ym_color::blur::{
    self, BlurKind, BlurParams, Frame, Plane4, length_px, motion, reach_px, soft::spline,
};
use ym_color::consts::Vec3;

fn motion_params(length: f64, angle_deg: f64, respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Motion { length, angle_deg },
        respect_mask_edges: respect,
    }
}

fn hash01(i: usize) -> f64 {
    ((i.wrapping_mul(2_654_435_761).wrapping_add(12345)) % 4_294_967_296) as f64 / 4_294_967_296.0
}

/// The test scenes in linear light: an edge, a disc (peak 1.5 on 0.05), a light (a 3 x 3 square of 300 on 0.02) and noise.
fn scene(name: &str, w: usize, h: usize) -> Vec<Vec3> {
    let mut v = vec![[0.05; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let p = &mut v[y * w + x];
            match name {
                "edge" => *p = [if x < w * 3 / 7 { 0.05 } else { 1.0 }; 3],
                "disc" => {
                    let (dx, dy) = (x as f64 - w as f64 * 0.45, y as f64 - h as f64 * 0.5);
                    if dx.hypot(dy) < h as f64 * 0.2 {
                        *p = [1.5; 3];
                    }
                }
                "light" => {
                    *p = [0.02; 3];
                    let (cx, cy) = (w * 11 / 20, h * 9 / 20);
                    if x.abs_diff(cx) <= 1 && y.abs_diff(cy) <= 1 {
                        *p = [300.0; 3];
                    }
                }
                _ => {
                    for (c, o) in p.iter_mut().enumerate() {
                        *o = 0.02 + 1.98 * hash01((y * 131 + x * 31 + c * 17) * 3 + c);
                    }
                }
            }
        }
    }
    v
}

fn ones(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// The largest difference of the blurred value `N / D` between two sums, over the largest value of the first.
fn relative_difference(a: (&[Vec3], &[f64]), b: (&[Vec3], &[f64])) -> f64 {
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    for i in 0..a.1.len() {
        for c in 0..3 {
            let p = a.0[i][c] / (a.1[i] + blur::EPS);
            let q = b.0[i][c] / (b.1[i] + blur::EPS);
            worst = worst.max((p - q).abs());
            peak = peak.max(p.abs());
        }
    }
    worst / peak
}

fn fast_against_oracle(name: &str, w: usize, h: usize, length: f64, angle: f64) -> f64 {
    let (x, m) = (scene(name, w, h), ones(w * h));
    let params = motion_params(length, angle, true);
    let frame = Frame::whole(w, h);
    let (n0, d0) = blur::sums(&x, &m, w, h, &params, &frame).expect("oracle");
    let (n1, d1) = blur::sums_fast(&x, &m, w, h, &params, &frame).expect("fast");
    relative_difference((&n0, &d0), (&n1, &d1))
}

#[test]
fn the_fast_method_is_within_three_thousandths_of_the_oracle_and_in_fact_the_oracle() {
    let mut worst = 0.0f64;
    // Length 10 to 100 on frames of 300 to 640 px: lines of 2.4 to 24 px, at angles along both axes, the diagonals and odd ones.
    for (w, h, length) in [
        (300usize, 200usize, 10.0),
        (300, 200, 100.0),
        (500, 340, 55.0),
        (640, 400, 20.0),
    ] {
        for angle in [0.0, 20.0, 45.0, 90.0, 137.0, 179.9, -33.0] {
            for name in ["edge", "disc", "light", "noise"] {
                let e = fast_against_oracle(name, w, h, length, angle);
                assert!(e <= 3e-3, "{name} {w}x{h} length {length} angle {angle}: {e}");
                worst = worst.max(e);
            }
        }
    }
    println!("motion fast against the oracle: worst {worst:.2e} (gate 3e-3)");
    assert_eq!(worst, 0.0);
}

#[test]
fn the_taps_are_the_nonzero_weights_of_the_oracles_table_in_its_order() {
    for (len, angle) in [
        (0.0, 0.0),
        (1e-9, 5.0),
        (0.7, 10.0),
        (7.3, 0.0),
        (16.0, 45.0),
        (33.7, 120.0),
        (64.0, 90.0),
    ] {
        let t = blur::line_table(len, angle);
        let taps = motion::taps(len, angle);
        let sum: f64 = taps.iter().map(|k| k.w).sum();
        assert!((sum - 1.0).abs() < 1e-12, "{len} {angle}: {sum}");
        let side = (2 * t.half + 1) as usize;
        let mut it = taps.iter();
        for (i, &w) in t.weights.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let k = it.next().expect("a tap");
            assert_eq!(k.w.to_bits(), w.to_bits());
            assert_eq!(i64::from(k.dx), (i % side) as i64 - t.half);
            assert_eq!(i64::from(k.dy), (i / side) as i64 - t.half);
        }
        assert!(it.next().is_none());
        // A line touches about l (|cos| + |sin|) + 2 pixels, a few more than that at most, never the whole square.
        let a = angle.to_radians();
        let bound = len * (a.cos().abs() + a.sin().abs()) + 4.0;
        assert!(taps.len() as f64 <= bound.max(1.0), "{len} {angle}: {}", taps.len());
    }
}

#[test]
fn a_window_with_its_reach_equals_the_frame_bit_for_bit() {
    for (w, h, length, angle) in [
        (300usize, 210usize, 100.0, 20.0),
        (257, 130, 60.0, 90.0),
        (500, 90, 100.0, 0.0),
        (333, 222, 45.0, 135.0),
    ] {
        let x = scene("noise", w, h);
        let m: Vec<f64> = (0..w * h)
            .map(|i| ((i % w) as f64 / w as f64).clamp(0.0, 1.0))
            .collect();
        for respect in [true, false] {
            let params = motion_params(length, angle, respect);
            let whole = blur::sums_fast(&x, &m, w, h, &params, &Frame::whole(w, h)).expect("whole");
            let reach = reach_px(&params, w, h);
            for (cx, cy, cw, ch) in [
                (10usize, 8usize, 25usize, 19usize),
                (0, 0, 31, 31),
                (97, 40, 61, 30),
                (w - 37, h - 21, 60, 40),
                (w / 2 - 30, 0, 61, 15),
            ] {
                if cx >= w || cy >= h {
                    continue;
                }
                let (x0, y0) = (cx.saturating_sub(reach), cy.saturating_sub(reach));
                let (x1, y1) = ((cx + cw + reach).min(w), (cy + ch + reach).min(h));
                let (ww, wh) = (x1 - x0, y1 - y0);
                let idx = || (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * w + x));
                let sub_x: Vec<Vec3> = idx().map(|i| x[i]).collect();
                let sub_m: Vec<f64> = idx().map(|i| m[i]).collect();
                let frame = Frame {
                    full_w: w,
                    full_h: h,
                    x0,
                    y0,
                };
                let got = blur::sums_fast(&sub_x, &sub_m, ww, wh, &params, &frame).expect("window");
                for y in cy..(cy + ch).min(h) {
                    for xx in cx..(cx + cw).min(w) {
                        let (i, j) = ((y - y0) * ww + (xx - x0), y * w + xx);
                        assert!(
                            got.0[i].map(f64::to_bits) == whole.0[j].map(f64::to_bits)
                                && got.1[i].to_bits() == whole.1[j].to_bits(),
                            "{w}x{h} length {length} angle {angle} respect {respect} at ({xx}, {y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn the_reach_holds_what_the_fast_method_reads() {
    for long in [
        200usize, 256, 300, 512, 1000, 1500, 2048, 4000, 6000, 8600, 9000,
    ] {
        for length in [0.5, 1.0, 5.0, 10.0, 20.0, 33.3, 50.0, 99.0, 100.0] {
            let params = motion_params(length, 30.0, true);
            let l = length_px(length, long as f64);
            let (reach, reads) = (reach_px(&params, long, long * 2 / 3), motion::reads_px(l));
            assert!(reach >= reads, "{long} px length {length}: {reach} < {reads}");
            let half = blur::line_table(l, 30.0).half as usize;
            assert_eq!(reads, half, "{long} px length {length}");
            assert!(motion::taps(l, 30.0).iter().all(|t| {
                (t.dx.unsigned_abs() as usize) <= reads && (t.dy.unsigned_abs() as usize) <= reads
            }));
        }
    }
}

#[test]
fn nothing_from_outside_the_mask_reaches_inside_and_the_same_test_fails_with_the_switch_off() {
    let (w, h) = (400, 280);
    let m: Vec<f64> = (0..w * h)
        .map(|i| if i % w < 180 { 0.0 } else { 1.0 })
        .collect();
    let a = scene("noise", w, h);
    let b: Vec<Vec3> = a
        .iter()
        .enumerate()
        .map(|(i, p)| if i % w < 180 { [50.0, 0.0, 9.0] } else { *p })
        .collect();
    for (respect, moves) in [(true, false), (false, true)] {
        let params = motion_params(100.0, 15.0, respect);
        let f = Frame::whole(w, h);
        let ra = blur::apply_linear(&a, &m, w, h, &params, &f).expect("a");
        let rb = blur::apply_linear(&b, &m, w, h, &params, &f).expect("b");
        let moved = (0..w * h)
            .filter(|i| i % w >= 180)
            .any(|i| (0..3).any(|c| (ra[i][c] - rb[i][c]).abs() > 1e-9));
        assert_eq!(moved, moves, "respect {respect}");
    }
}

#[test]
fn a_flat_field_is_flat_and_a_point_keeps_its_energy() {
    let (w, h) = (512, 512);
    let m = ones(w * h);
    let flat = vec![[0.18, 0.4, 0.07]; w * h];
    let f = Frame::whole(w, h);
    for angle in [0.0, 33.0, 90.0] {
        let params = motion_params(100.0, angle, true);
        let out = blur::apply_fast(&flat, &m, w, h, &params, &f);
        assert!(out.is_ok());
        let lin = blur::apply_linear(&flat, &m, w, h, &params, &f).expect("flat");
        for p in lin.iter().step_by(37) {
            assert!((p[0] - 0.18).abs() < 1e-9 && (p[1] - 0.4).abs() < 1e-9);
        }
        let mut pt = vec![[0.0; 3]; w * h];
        pt[256 * w + 256] = [1000.0, 500.0, 100.0];
        let (n, d) = blur::sums_fast(&pt, &m, w, h, &params, &f).expect("point");
        let _ = d;
        for c in 0..3 {
            let e = n.iter().map(|p| p[c]).sum::<f64>() / pt[256 * w + 256][c] - 1.0;
            assert!(e.abs() < 1e-9, "energy, angle {angle}, channel {c}: {e}");
        }
    }
}

#[test]
fn the_mirrored_picture_gives_the_mirrored_result_at_0_and_90_degrees() {
    let (w, h) = (256, 192);
    let x = scene("noise", w, h);
    let m: Vec<f64> = (0..w * h)
        .map(|i| ((i % w) as f64 / w as f64).clamp(0.0, 1.0))
        .collect();
    let mirror_x = |v: &[Vec3]| -> Vec<Vec3> {
        (0..w * h)
            .map(|i| v[(i / w) * w + (w - 1 - i % w)])
            .collect()
    };
    let mirror_m = |v: &[f64]| -> Vec<f64> {
        (0..w * h)
            .map(|i| v[(i / w) * w + (w - 1 - i % w)])
            .collect()
    };
    for angle in [0.0, 90.0] {
        let params = motion_params(100.0, angle, true);
        let f = Frame::whole(w, h);
        let a = blur::apply_linear(&x, &m, w, h, &params, &f).expect("a");
        let b = blur::apply_linear(&mirror_x(&x), &mirror_m(&m), w, h, &params, &f).expect("b");
        let a = mirror_x(&a);
        let d = a
            .iter()
            .zip(&b)
            .flat_map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).abs()))
            .fold(0.0, f64::max);
        assert!(d < 1e-9, "angle {angle}: {d}");
    }
}

#[test]
fn black_huge_negative_and_subnormal_stay_finite() {
    let (w, h) = (300, 200);
    let params = motion_params(100.0, 40.0, true);
    let m = ones(w * h);
    for value in [0.0, 1e-310, -3.0, 1e12, f64::MAX / 1e10] {
        let x = vec![[value; 3]; w * h];
        let (n, d) = blur::sums_fast(&x, &m, w, h, &params, &Frame::whole(w, h)).expect("sums");
        assert!(n.iter().all(|q| q.iter().all(|v| v.is_finite())), "{value:e}");
        assert!(d.iter().all(|v| v.is_finite()));
    }
}

// ---- the spec's design of cells, measured (docs/33 11 item 50) ----

/// The design `docs/33` 3.5 first gave Motion: the weighted planes reduced to cells of `f x f` pixels (the quadratic B-spline
/// of Soft), `ceil(l / f)` equally weighted bilinear taps along the line over the cells (the best spacing of at most one cell),
/// and the B-spline read-back. A frame with no edge handling beyond zero outside (the ratio `N / D` renormalises).
fn cells_method(p: &[Plane4], w: usize, h: usize, l: f64, angle_deg: f64, f: usize) -> Vec<Plane4> {
    let fi = f as i64;
    let (cw, ch) = (w.div_ceil(f) + 2, h.div_ceil(f) + 2); // cells -1 ..= ceil(w / f)
    let mut cells = vec![[0.0; 4]; cw * ch];
    for cj in 0..ch {
        for ci in 0..cw {
            let (i, j) = (ci as i64 - 1, cj as i64 - 1);
            let mut acc = [0.0; 4];
            for y in ((j - 1) * fi).max(0)..((j + 2) * fi).min(h as i64) {
                let ky = spline(y, j, fi);
                for x in ((i - 1) * fi).max(0)..((i + 2) * fi).min(w as i64) {
                    let k = ky * spline(x, i, fi);
                    for c in 0..4 {
                        acc[c] += k * p[y as usize * w + x as usize][c];
                    }
                }
            }
            cells[cj * cw + ci] = acc.map(|a| a / (f * f) as f64);
        }
    }
    let a = angle_deg.to_radians();
    let dir = [a.cos(), -a.sin()];
    let n = (l / f as f64).ceil().max(1.0) as usize;
    let at = |ci: i64, cj: i64| -> Plane4 {
        let (a, b) = (ci + 1, cj + 1);
        if a < 0 || b < 0 || a >= cw as i64 || b >= ch as i64 {
            [0.0; 4]
        } else {
            cells[b as usize * cw + a as usize]
        }
    };
    // The line of every cell, bilinear reads at cell coordinates (a cell's centre is at pixel (I + 1/2) f - 1/2).
    let mut lined = vec![[0.0; 4]; cw * ch];
    for cj in 0..ch {
        for ci in 0..cw {
            let mut acc = [0.0; 4];
            for k in 0..n {
                let t = -l / 2.0 + (k as f64 + 0.5) * l / n as f64;
                let u = (ci as f64 - 1.0) + t * dir[0] / f as f64;
                let v = (cj as f64 - 1.0) + t * dir[1] / f as f64;
                let (u0, v0) = (u.floor(), v.floor());
                let (fx, fy) = (u - u0, v - v0);
                let (i0, j0) = (u0 as i64, v0 as i64);
                let q = [at(i0, j0), at(i0 + 1, j0), at(i0, j0 + 1), at(i0 + 1, j0 + 1)];
                let wts = [
                    (1.0 - fx) * (1.0 - fy),
                    fx * (1.0 - fy),
                    (1.0 - fx) * fy,
                    fx * fy,
                ];
                for c in 0..4 {
                    for s in 0..4 {
                        acc[c] += wts[s] * q[s][c] / n as f64;
                    }
                }
            }
            lined[cj * cw + ci] = acc;
        }
    }
    let mut out = vec![[0.0; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let (i0, j0) = ((x / f) as i64, (y / f) as i64);
            let mut acc = [0.0; 4];
            for j in j0 - 1..=j0 + 1 {
                for i in i0 - 1..=i0 + 1 {
                    let (a, b) = (i + 1, j + 1);
                    if a < 0 || b < 0 || a >= cw as i64 || b >= ch as i64 {
                        continue;
                    }
                    let k = spline(y as i64, j, fi) * spline(x as i64, i, fi);
                    for c in 0..4 {
                        acc[c] += k * lined[b as usize * cw + a as usize][c];
                    }
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

#[test]
fn the_specs_cells_miss_the_bound_on_a_light_and_on_noise_which_is_why_the_taps_are_exact() {
    let (w, h) = (800usize, 560usize);
    let frame = Frame::whole(w, h);
    let mut report = Vec::new();
    // l / f = 16 at f = 4 (64 px, Length 100 on 800 px) and 12.5 at f = 8 (100 px).
    for (l, f, angle) in [(64.0, 4usize, 0.0), (64.0, 4, 30.0)] {
        for name in ["edge", "disc", "light", "noise"] {
            let x = scene(name, w, h);
            let m = ones(w * h);
            let params = motion_params(l / (0.08 * w as f64) * 100.0, angle, true);
            let (n0, d0) = blur::sums(&x, &m, w, h, &params, &frame).expect("oracle");
            // The weighted planes of the oracle's own `omega` (m = 1, no highlights: omega = 1).
            let planes: Vec<Plane4> = x.iter().map(|v| [v[0], v[1], v[2], 1.0]).collect();
            let c = cells_method(&planes, w, h, l, angle, f);
            let n1: Vec<Vec3> = c.iter().map(|q| [q[0], q[1], q[2]]).collect();
            let d1: Vec<f64> = c.iter().map(|q| q[3]).collect();
            let e = relative_difference((&n0, &d0), (&n1, &d1));
            report.push((name, angle, e));
            println!("cells of {f}, l = {l}, angle {angle}, {name}: {e:.2e} (gate 3e-3)");
        }
    }
    let worst = |name: &str| {
        report
            .iter()
            .filter(|r| r.0 == name)
            .fold(0.0f64, |a, r| a.max(r.2))
    };
    assert!(worst("light") > 3e-3, "light {}", worst("light"));
    assert!(worst("noise") > 3e-3, "noise {}", worst("noise"));
}
