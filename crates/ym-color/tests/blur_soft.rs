//! Soft's fast method against the oracle (`docs/33-blur.md` 3.5, 3.7, 8; BL-06): the direct sum up to sigma 4 px is the
//! oracle's own, the coarse grid beyond is within 5e-3 of it in linear light on an edge, a disc, a light and a noise field,
//! a window of the reach equals the frame bit for bit, the mask gate and the energy hold.

use ym_color::blur::{
    self, BlurKind, BlurParams, Frame, radius_px, reach_px,
    soft::{CELL_SIGMA_MAX, cell_level, cell_sigma},
};
use ym_color::consts::Vec3;

fn soft(radius: f64, respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Soft { radius },
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

/// The largest difference of the blurred value `N / D` between the fast method and the oracle, over the largest oracle value.
fn fast_against_oracle(name: &str, w: usize, h: usize, radius: f64) -> f64 {
    let (x, m) = (scene(name, w, h), ones(w * h));
    let params = soft(radius, true);
    let frame = Frame::whole(w, h);
    let (n0, d0) = blur::sums(&x, &m, w, h, &params, &frame).expect("oracle");
    let (n1, d1) = blur::sums_fast(&x, &m, w, h, &params, &frame).expect("fast");
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    for i in 0..w * h {
        for c in 0..3 {
            let a = n0[i][c] / (d0[i] + blur::EPS);
            let b = n1[i][c] / (d1[i] + blur::EPS);
            worst = worst.max((a - b).abs());
            peak = peak.max(a.abs());
        }
    }
    worst / peak
}

#[test]
fn the_cell_rule_is_the_smallest_power_of_two_with_a_cell_sigma_of_four_or_less() {
    assert_eq!(cell_level(0.0), 0);
    assert_eq!(cell_level(4.0), 0);
    assert_eq!(cell_level(4.01), 1);
    for sigma in [4.5, 8.0, 8.1, 16.0, 25.0, 40.0, 120.0, 172.0] {
        let j = cell_level(sigma);
        assert!(j >= 1);
        let f = 1usize << j;
        assert!(cell_sigma(sigma, f) <= CELL_SIGMA_MAX, "{sigma}");
        if j > 1 {
            assert!(cell_sigma(sigma, f / 2) > CELL_SIGMA_MAX, "{sigma}");
        }
    }
    // The spec's frames: sigma 120 px (6000 px) uses f = 32, sigma 172 px (8600 px) f = 64.
    assert_eq!(1usize << cell_level(120.0), 32);
    assert_eq!(1usize << cell_level(172.0), 64);
}

#[test]
fn up_to_sigma_four_the_fast_method_is_the_oracle() {
    // sigma = radius / 100 * 0.04 * long / 2; 4 px on a 200 px frame at Radius 100.
    for name in ["edge", "light", "noise"] {
        let (w, h) = (200, 140);
        let (x, m) = (scene(name, w, h), ones(w * h));
        let params = soft(100.0, true);
        let frame = Frame::whole(w, h);
        assert!(radius_px(100.0, 200.0) / 2.0 <= 4.0);
        let a = blur::apply(&x, &m, w, h, &params, &frame).expect("a");
        let b = blur::apply_fast(&x, &m, w, h, &params, &frame).expect("b");
        assert!(
            a.iter()
                .zip(&b)
                .all(|(p, q)| p.map(f64::to_bits) == q.map(f64::to_bits))
        );
    }
}

#[test]
fn the_coarse_grid_is_within_five_thousandths_of_the_direct_sum() {
    let mut worst = 0.0f64;
    // Frame long edges 400, 800, 1200: sigma 8, 16, 24 at Radius 100; and Radius 50 on 1200 (12 px).
    for (w, h, radius) in [
        (400, 280, 100.0),
        (800, 560, 100.0),
        (1200, 800, 100.0),
        (1200, 800, 50.0),
        (600, 400, 70.0),
        (1000, 700, 65.0),
        (1600, 1000, 100.0),
    ] {
        for name in ["edge", "disc", "light", "noise"] {
            let e = fast_against_oracle(name, w, h, radius);
            println!(
                "soft fast {name} {w}x{h} radius {radius} sigma {:.1}: {e:.2e}",
                radius_px(radius, w as f64) / 2.0
            );
            assert!(e <= 5e-3, "{name} {w}x{h} radius {radius}: {e}");
            worst = worst.max(e);
        }
    }
    println!("soft fast worst: {worst:.2e}");
}

#[test]
fn a_window_with_its_reach_equals_the_frame_bit_for_bit() {
    for (w, h, radius) in [
        (300usize, 210usize, 100.0),
        (257, 130, 60.0),
        (500, 90, 100.0),
    ] {
        let x = scene("noise", w, h);
        let m: Vec<f64> = (0..w * h)
            .map(|i| ((i % w) as f64 / w as f64).clamp(0.0, 1.0))
            .collect();
        for respect in [true, false] {
            let params = soft(radius, respect);
            let whole = blur::sums_fast(&x, &m, w, h, &params, &Frame::whole(w, h)).expect("whole");
            let reach = reach_px(&params, w, h);
            for core in [
                (10usize, 8usize, 25usize, 19usize),
                (0, 0, 31, 31),
                (97, 40, 61, 30),
                (w - 37, h - 21, 60, 40),
                (w / 2 - 30, 0, 61, 15),
            ] {
                let (cx, cy, cw, ch) = core;
                if cx >= w || cy >= h {
                    continue;
                }
                let (x0, y0) = (cx.saturating_sub(reach), cy.saturating_sub(reach));
                let (x1, y1) = ((cx + cw + reach).min(w), (cy + ch + reach).min(h));
                let (ww, wh) = (x1 - x0, y1 - y0);
                let sub = |f: &dyn Fn(usize) -> f64| -> Vec<f64> {
                    (y0..y1)
                        .flat_map(|y| (x0..x1).map(move |x| y * w + x))
                        .map(f)
                        .collect()
                };
                let sub_x: Vec<Vec3> = (y0..y1)
                    .flat_map(|y| (x0..x1).map(move |x| y * w + x))
                    .map(|i| x[i])
                    .collect();
                let sub_m = sub(&|i| m[i]);
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
                            "{w}x{h} radius {radius} respect {respect} core {core:?} at ({xx}, {y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn nothing_from_outside_the_mask_reaches_inside_on_the_coarse_grid() {
    let (w, h) = (400, 280);
    let m: Vec<f64> = (0..w * h)
        .map(|i| if i % w < 180 { 0.0 } else { 1.0 })
        .collect();
    let a = scene("noise", w, h);
    // Change the picture wildly where m = 0.
    let b: Vec<Vec3> = a
        .iter()
        .enumerate()
        .map(|(i, p)| if i % w < 180 { [50.0, 0.0, 9.0] } else { *p })
        .collect();
    for (respect, moves) in [(true, false), (false, true)] {
        let params = soft(100.0, respect);
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
fn a_flat_field_is_flat_and_a_point_keeps_its_energy_on_the_coarse_grid() {
    let (w, h) = (512, 512);
    let m = ones(w * h);
    let flat = vec![[0.18, 0.4, 0.07]; w * h];
    let params = soft(100.0, true);
    let f = Frame::whole(w, h);
    let out = blur::apply_linear(&flat, &m, w, h, &params, &f).expect("flat");
    for p in out.iter().step_by(37) {
        assert!((p[0] - 0.18).abs() < 1e-9 && (p[1] - 0.4).abs() < 1e-9);
    }
    // Radius 30 (sigma 3 x 512 x 0.04 / 2 ...): a point well inside keeps its sum.
    let params = soft(30.0, true);
    let mut pt = vec![[0.0; 3]; w * h];
    pt[256 * w + 256] = [1000.0, 500.0, 100.0];
    let out = blur::apply_linear(&pt, &m, w, h, &params, &f).expect("point");
    let sum = |v: &[Vec3], c: usize| v.iter().map(|p| p[c]).sum::<f64>();
    for c in 0..3 {
        let e = sum(&out, c) / sum(&pt, c) - 1.0;
        assert!(
            e.abs() < 1e-9,
            "energy of the point's blur, channel {c}: {e}"
        );
    }
}

#[test]
fn the_reach_holds_what_the_fast_method_reads() {
    let mut worst = 0.0f64;
    for long in [
        200usize, 256, 300, 512, 1000, 1500, 2048, 3000, 4000, 4096, 6000, 8600, 9000,
    ] {
        for radius in [1.0, 5.0, 10.0, 20.0, 33.3, 50.0, 70.0, 99.0, 100.0] {
            let params = soft(radius, true);
            let sigma = radius_px(radius, long as f64) / 2.0;
            let reach = reach_px(&params, long, long * 2 / 3);
            let reads = ym_color::blur::soft::reads_px(sigma);
            assert!(
                reach >= reads,
                "{long} px radius {radius}: reach {reach} < {reads}"
            );
            worst = worst.max(reads as f64 / reach as f64);
        }
    }
    println!("soft: the fast method reads at most {worst:.3} of the reach");
}

#[test]
fn the_mirrored_picture_gives_the_mirrored_result_on_the_coarse_grid() {
    // A frame whose sides are multiples of the cell (f = 2 at sigma 5.1): the cells mirror with the picture.
    let (w, h) = (256, 192);
    let x = scene("noise", w, h);
    let m: Vec<f64> = (0..w * h)
        .map(|i| ((i % w) as f64 / w as f64).clamp(0.0, 1.0))
        .collect();
    let mirror = |v: &[Vec3]| -> Vec<Vec3> {
        (0..w * h)
            .map(|i| v[(i / w) * w + (w - 1 - i % w)])
            .collect()
    };
    let mirror_m = |v: &[f64]| -> Vec<f64> {
        (0..w * h)
            .map(|i| v[(i / w) * w + (w - 1 - i % w)])
            .collect()
    };
    for radius in [100.0, 90.0] {
        let params = soft(radius, true);
        let f = Frame::whole(w, h);
        assert!(ym_color::blur::soft::cell_level(radius_px(radius, w as f64) / 2.0) >= 1);
        let a = blur::apply_linear(&x, &m, w, h, &params, &f).expect("a");
        let b = blur::apply_linear(&mirror(&x), &mirror_m(&m), w, h, &params, &f).expect("b");
        let a = mirror(&a);
        let d = a
            .iter()
            .zip(&b)
            .flat_map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).abs()))
            .fold(0.0, f64::max);
        assert!(d < 1e-9, "radius {radius}: {d}");
    }
}

/// The node in linear light by the fast method: `blend_linear` of `sums_fast` (the clean-up of the mask as the node's).
fn fast_linear(x: &[Vec3], m: &[f64], w: usize, h: usize, params: &BlurParams) -> Vec<Vec3> {
    let (n, d) = blur::sums_fast(x, m, w, h, params, &Frame::whole(w, h)).expect("fast");
    (0..x.len())
        .map(|i| blur::blend_linear(x[i], m[i].clamp(0.0, 1.0), n[i], d[i]))
        .collect()
}

fn max_diff(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn on_the_coarse_grid_a_mask_at_half_weighs_half_and_the_result_is_continuous_in_m() {
    let (w, h) = (400, 300);
    let x = scene("noise", w, h);
    let params = soft(100.0, true);
    assert!(cell_level(radius_px(100.0, w as f64) / 2.0) >= 1);
    let one = fast_linear(&x, &vec![1.0; w * h], w, h, &params);
    let half = fast_linear(&x, &vec![0.5; w * h], w, h, &params);
    let want: Vec<Vec3> = x
        .iter()
        .zip(&one)
        .map(|(a, b)| [0, 1, 2].map(|c| a[c] + 0.5 * (b[c] - a[c])))
        .collect();
    assert!(max_diff(&half, &want) < 1e-9);
    // A sweep of a uniform mask moves the result by no more than the step times the range.
    let a = fast_linear(&x, &vec![0.4; w * h], w, h, &params);
    let b = fast_linear(&x, &vec![0.4 + 1e-6; w * h], w, h, &params);
    let range = max_diff(&one, &x);
    assert!(max_diff(&a, &b) <= 1e-6 * range * 1.001 + 1e-12);
    // One pixel with a tiny mask and no other weight in reach: a tiny change, never a jump, and nothing not finite.
    for tiny in [1e-300, 1e-100, 1e-15, 1e-9, 1e-6] {
        let mut m = vec![0.0; w * h];
        let c = 150 * w + 200;
        m[c] = tiny;
        let o = fast_linear(&x, &m, w, h, &params);
        assert!(o.iter().all(|q| q.iter().all(|v| v.is_finite())), "{tiny}");
        let d = (0..3)
            .map(|ch| (o[c][ch] - x[c][ch]).abs())
            .fold(0.0, f64::max);
        assert!(d <= tiny * 600.0 + 1e-12, "{tiny:e}: {d:e}");
    }
    // Opacity 0.5 of a hard mask is a mask of 0.5 inside: still no light from where it is 0.
    let m: Vec<f64> = (0..w * h)
        .map(|i| if i % w >= 180 { 0.5 } else { 0.0 })
        .collect();
    let y: Vec<Vec3> = x
        .iter()
        .enumerate()
        .map(|(i, p)| if i % w < 180 { [1e4; 3] } else { *p })
        .collect();
    let (ra, rb) = (
        fast_linear(&x, &m, w, h, &params),
        fast_linear(&y, &m, w, h, &params),
    );
    let d = (0..w * h)
        .filter(|i| m[*i] > 0.0)
        .map(|i| {
            (0..3)
                .map(|c| (ra[i][c] - rb[i][c]).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    assert!(d < 1e-9, "{d}");
}

#[test]
fn black_huge_negative_and_subnormal_stay_finite_on_the_coarse_grid() {
    let (w, h) = (300, 200);
    let params = soft(100.0, true);
    assert!(cell_level(radius_px(100.0, w as f64) / 2.0) >= 1);
    let m = ones(w * h);
    for value in [0.0, 1e-310, -3.0, 1e12, f64::MAX / 1e10] {
        let x = vec![[value; 3]; w * h];
        let o = fast_linear(&x, &m, w, h, &params);
        assert!(
            o.iter().all(|q| q.iter().all(|v| v.is_finite())),
            "{value:e}"
        );
    }
    let mut x = vec![[0.0; 3]; w * h];
    x[100 * w + 100] = [1e9, 1e9, 1e9];
    let o = fast_linear(&x, &m, w, h, &params);
    assert!(o.iter().all(|q| q.iter().all(|v| v.is_finite())));
}
