//! The raster masks' CPU reference (`docs/algorithms/06` section 5): the resample, Grow, Feather and the guided
//! filter against independent implementations on known planes, the Brush strokes, the skips, the tiles with
//! their halos, and the chain reading a plane.

use std::sync::Arc;

use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, Frame, Layout, View};
use ym_color::display::{ChainParams, NodeParams, NodeSet};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoFrame, PhotoMap, RasterPlanes};
use ym_color::mask_raster::{
    Area, BrushStrokes, CpuPlanes, Guided, PlaneJob, RasterRow, RasterSet, RasterSource,
    RasterSteps, SourcePlane, Stroke, box_mean, brush_plane, draw_brush, feather, feather_boxes,
    grow, guided, quantise, resample_at, resample_taps,
};
use ym_color::plan::{PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;

// ---- helpers ------------------------------------------------------------------------------------

/// A deterministic pseudo-random sequence in [0, 1) (no dependencies).
fn noise(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn plane_of(w: usize, h: usize, f: impl Fn(usize, usize) -> f64) -> SourcePlane {
    SourcePlane {
        key: 1,
        width: w,
        height: h,
        data: (0..w * h).map(|i| quantise(f(i % w, i / w))).collect(),
    }
}

/// The identity position: pixel `(x, y)` of a `w x h` window is at its own place in the photo.
fn identity_position(w: usize, h: usize) -> impl Fn(usize, usize) -> Option<[f64; 2]> + Sync {
    move |x, y| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64])
}

fn row(src: SourcePlane, grow: f64, feather: f64, refine: f64) -> RasterRow {
    RasterRow {
        source: RasterSource::Plane(Arc::new(src)),
        grow,
        feather,
        refine,
    }
}

fn codes(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&c| f64::from(c) / 65535.0).collect()
}

// ---- the sizes ----------------------------------------------------------------------------------

#[test]
fn the_sizes_are_photo_relative_and_round_to_whole_pixels() {
    // 6000 px: Grow 100 is 60 px, Feather 100 is sigma 120, Refine edges 100 is a window of 48 px.
    let s = RasterSteps::of(1.0, 1.0, 1.0, 6000.0);
    assert_eq!(s.grow, 60);
    let g = s.guided.expect("a guided filter");
    assert_eq!((g.radius, g.step, g.samples), (48, 3, 16));
    assert!((g.eps - 1e-5).abs() < 1e-17);
    // The same row on a 1500 px fit view is a quarter of that.
    let s = RasterSteps::of(1.0, 1.0, 1.0, 1500.0);
    assert_eq!(s.grow, 15);
    let g = s.guided.expect("a guided filter");
    assert_eq!((g.radius, g.step, g.samples), (12, 1, 12));
    // Negative Grow is an erosion of the same size.
    assert_eq!(RasterSteps::of(-0.5, 0.0, 0.0, 6000.0).grow, -30);
    // Refine edges 0.5: eps = 10^-3.5.
    let g = RasterSteps::of(0.0, 0.0, 0.5, 6000.0)
        .guided
        .expect("a guided filter");
    assert!((g.eps - 10f64.powf(-3.5)).abs() < 1e-17);
    assert_eq!(g.radius, 24);
    // A radius of at most 16 is not subsampled; 17 is.
    let at = |u: f64| RasterSteps::of(0.0, 0.0, 1.0, u).guided;
    assert_eq!(at(2000.0).map(|g| (g.radius, g.step)), Some((16, 1)));
    assert_eq!(at(2125.0).map(|g| (g.radius, g.step, g.samples)), Some((17, 2, 9)));
    // Every step under its threshold is skipped; a number that is not a number is 0.
    let none = RasterSteps::of(0.0, 0.0, 0.0, 6000.0);
    assert_eq!((none.grow, none.feather, none.guided), (0, None, None));
    assert_eq!(none.halo(), 0);
    assert_eq!(RasterSteps::of(f64::NAN, f64::NAN, f64::NAN, 6000.0), none);
    // Feather under half a pixel of sigma, Refine edges under a pixel of window.
    assert!(RasterSteps::of(0.0, 0.2, 0.0, 100.0).feather.is_none());
    assert!(RasterSteps::of(0.0, 0.5, 0.0, 100.0).feather.is_some());
    assert!(RasterSteps::of(0.0, 0.0, 0.1, 100.0).guided.is_none());
}

#[test]
fn the_halo_is_the_sum_of_the_radii() {
    let s = RasterSteps::of(0.5, 0.5, 0.5, 1500.0);
    let boxes = s.feather.expect("feather");
    let g = s.guided.expect("guided");
    assert_eq!(s.halo(), s.grow.unsigned_abs() as usize + boxes.iter().sum::<usize>() + 2 * g.radius);
    let big = RasterSteps::of(0.0, 0.0, 1.0, 6000.0);
    assert_eq!(big.halo(), (2 * 16 + 3) * 3);
    let set = RasterSet {
        rows: vec![
            row(plane_of(4, 4, |_, _| 0.0), 0.0, 0.0, 0.0),
            row(plane_of(4, 4, |_, _| 0.0), 1.0, 0.0, 0.0),
        ],
    };
    assert_eq!(set.halo(6000.0), 60);
    assert!(!set.needs_guide(6000.0));
}

// ---- resample -----------------------------------------------------------------------------------

#[test]
fn a_resample_at_a_pixel_centre_is_the_pixel_and_between_them_the_mean() {
    let src = plane_of(4, 4, |x, y| (x + 4 * y) as f64 / 15.0);
    // The centre of pixel (2, 1) of 4 x 4 is at ((2 + 0.5) / 4, (1 + 0.5) / 4).
    let v = resample_at(&src, [2.5 / 4.0, 1.5 / 4.0], 1, 1.0);
    assert!((v - 6.0 / 15.0).abs() < 1e-4, "{v}");
    // Halfway between pixels (1, 1) and (2, 1): their mean.
    let v = resample_at(&src, [2.0 / 4.0, 1.5 / 4.0], 1, 1.0);
    assert!((v - 5.5 / 15.0).abs() < 1e-4, "{v}");
    // Outside the plane the edge pixel is held, and the result stays in [0, 1].
    assert!((resample_at(&src, [-0.2, 0.0], 1, 1.0) - 0.0).abs() < 1e-4);
    assert!((resample_at(&src, [1.2, 1.2], 1, 1.0) - 1.0).abs() < 1e-4);
}

#[test]
fn the_resample_never_overshoots() {
    // A step edge: bilinear and the box average are convex combinations, so no value leaves [0, 1].
    let src = plane_of(64, 64, |x, _| if x < 32 { 0.0 } else { 1.0 });
    for taps in [1, 2, 5] {
        for i in 0..400 {
            let x = i as f64 / 400.0;
            let v = resample_at(&src, [x, 0.5], taps, 2.7);
            assert!((0.0..=1.0).contains(&v), "taps {taps} x {x}: {v}");
        }
    }
}

#[test]
fn the_box_average_starts_above_a_shrink_of_one_and_a_quarter() {
    assert_eq!(resample_taps(2048, 2048.0), 1);
    assert_eq!(resample_taps(2048, 6000.0), 1);
    assert_eq!(resample_taps(2048, 1700.0), 1); // 1.20
    assert_eq!(resample_taps(2048, 1600.0), 2); // 1.28
    assert_eq!(resample_taps(2048, 1500.0), 2); // 1.37
    assert_eq!(resample_taps(2048, 160.0), 13); // 12.8
    assert_eq!(resample_taps(2048, 16.0), 16); // capped
}

#[test]
fn a_shrunk_resample_is_the_mean_of_the_source_under_the_pixel() {
    // A source that alternates every pixel: one sample at a pixel centre sees 0 or 1, the box average under
    // a pixel four source pixels wide sees one half.
    let src = plane_of(256, 256, |x, _| if x % 2 == 0 { 0.0 } else { 1.0 });
    let shrink = 4.0;
    let taps = resample_taps(256, 64.0);
    assert_eq!(taps, 4);
    for ix in 0..64 {
        let pos = [(ix as f64 + 0.5) / 64.0, 0.5];
        let v = resample_at(&src, pos, taps, shrink);
        assert!((v - 0.5).abs() < 0.13, "pixel {ix}: {v}");
    }
}

// ---- grow ---------------------------------------------------------------------------------------

#[test]
fn grow_is_a_square_maximum_and_an_erosion_a_square_minimum() {
    let (w, h) = (9, 7);
    let g = Area::whole(w, h);
    let mut p = vec![0.0; w * h];
    p[3 * w + 4] = 1.0; // one pixel at (4, 3)
    let grown = grow(&p, &g, &g, 2);
    for y in 0..h {
        for x in 0..w {
            let inside = (x as i64 - 4).abs() <= 2 && (y as i64 - 3).abs() <= 2;
            assert_eq!(grown[y * w + x], if inside { 1.0 } else { 0.0 }, "({x}, {y})");
        }
    }
    // Eroding a 5 x 5 square by 1 leaves its 3 x 3 core; by 2, one pixel.
    let mut sq = vec![0.0; w * h];
    for y in 1..6 {
        for x in 2..7 {
            sq[y * w + x] = 1.0;
        }
    }
    let eroded = grow(&sq, &g, &g, -1);
    let count = |v: &[f64]| v.iter().filter(|&&x| x == 1.0).count();
    assert_eq!(count(&eroded), 9);
    assert_eq!(count(&grow(&sq, &g, &g, -2)), 1);
    // Zero is the plane itself, bit for bit.
    assert_eq!(grow(&sq, &g, &g, 0), sq);
}

#[test]
fn a_window_at_the_frame_edge_is_cut_to_the_frame() {
    // The eroding window does not see beyond the frame: a plane of ones stays ones to its edge.
    let g = Area::whole(6, 6);
    let ones = vec![1.0; 36];
    assert_eq!(grow(&ones, &g, &g, -3), ones);
    // And the mean of a window at the edge is divided by the samples it kept.
    let m = box_mean(&ones, &g, &g, 3);
    assert!(m.iter().all(|&v| v == 1.0));
}

// ---- feather ------------------------------------------------------------------------------------

#[test]
fn the_feather_boxes_have_a_gaussians_variance() {
    // Sum of the variances of the boxes (a box of width w has (w^2 - 1) / 12) against sigma^2.
    for sigma in [1.0, 2.0, 5.0, 12.0, 36.0, 120.0] {
        let boxes = feather_boxes(sigma);
        let var: f64 = boxes
            .iter()
            .map(|&r| {
                let w = 2.0 * r as f64 + 1.0;
                (w * w - 1.0) / 12.0
            })
            .sum();
        let want = sigma * sigma;
        let err = (var - want).abs() / want;
        assert!(err < 0.35 / sigma.max(1.0).powf(0.5) + 0.12, "sigma {sigma}: {var} against {want}");
        if sigma >= 5.0 {
            assert!(err < 0.08, "sigma {sigma}: {var} against {want}");
        }
    }
}

#[test]
fn a_feathered_impulse_has_the_variance_of_its_boxes_and_sums_to_one() {
    let (w, h) = (101, 101);
    let g = Area::whole(w, h);
    let mut p = vec![0.0; w * h];
    p[50 * w + 50] = 1.0;
    let sigma = 6.0;
    let boxes = feather_boxes(sigma);
    let out = feather(&p, &g, &g, boxes);
    let total: f64 = out.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
    let mut var_x = 0.0;
    for y in 0..h {
        for x in 0..w {
            var_x += out[y * w + x] * (x as f64 - 50.0).powi(2);
        }
    }
    // The variance along x of the 2-D response is that of the three horizontal boxes.
    let want: f64 = boxes
        .iter()
        .map(|&r| {
            let width = 2.0 * r as f64 + 1.0;
            (width * width - 1.0) / 12.0
        })
        .sum();
    assert!((var_x - want).abs() < 1e-9, "{var_x} against {want}");
    assert!((var_x - sigma * sigma).abs() / (sigma * sigma) < 0.1);
    // The response is symmetric and does not reach beyond the summed half-widths.
    let reach: usize = boxes.iter().sum();
    for y in 0..h {
        for x in 0..w {
            if (x as i64 - 50).unsigned_abs() as usize > reach {
                assert_eq!(out[y * w + x], 0.0);
            }
        }
    }
}

// ---- the guided filter against a direct implementation ------------------------------------------

/// The guided filter written straight from the formulas, with the means taken by explicit double loops over
/// each window (the independent reference of `guided`), for a window of half-side `r` samples on the full grid.
fn direct_guided(p: &[f64], gd: &[f64], w: usize, h: usize, r: usize, eps: f64) -> Vec<f64> {
    let mean = |v: &dyn Fn(usize) -> f64, x: usize, y: usize| {
        let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
        let mut sum = 0.0;
        let mut n = 0.0;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                sum += v(yy * w + xx);
                n += 1.0;
            }
        }
        sum / n
    };
    let mut a = vec![0.0; w * h];
    let mut b = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mg = mean(&|i| gd[i], x, y);
            let mp = mean(&|i| p[i], x, y);
            let mgg = mean(&|i| gd[i] * gd[i], x, y);
            let mgp = mean(&|i| gd[i] * p[i], x, y);
            let var = mgg - mg * mg;
            let cov = mgp - mg * mp;
            a[y * w + x] = cov / (var + eps);
            b[y * w + x] = mp - a[y * w + x] * mg;
        }
    }
    let mut q = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let ma = mean(&|i| a[i], x, y);
            let mb = mean(&|i| b[i], x, y);
            q[y * w + x] = (ma * gd[y * w + x] + mb).clamp(0.0, 1.0);
        }
    }
    q
}

#[test]
fn the_guided_filter_is_the_formulas_on_a_textured_guide() {
    let (w, h) = (37, 29);
    let g = Area::whole(w, h);
    let mut rnd = noise(7);
    let guide: Vec<f64> = (0..w * h).map(|_| rnd()).collect();
    let p: Vec<f64> = (0..w * h)
        .map(|i| if (i % w) < 17 { 1.0 } else { 0.0 })
        .collect();
    for (r, eps) in [(1, 1e-2), (3, 3.2e-4), (7, 1e-5), (16, 1e-3)] {
        let f = Guided {
            radius: r,
            step: 1,
            samples: r,
            eps,
        };
        let got = guided(&p, &guide, &g, &g, &f);
        let want = direct_guided(&p, &guide, w, h, r, eps);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-10, "r {r}: {worst}");
    }
}

#[test]
fn a_step_edge_with_a_matching_guide_stays_a_step() {
    // The mask is a hard step and the guide has the same step: the filter returns the step to within 1e-3
    // (Refine edges 100: eps 1e-5, a window larger than the edge's neighbourhood).
    let (w, h) = (96, 40);
    let g = Area::whole(w, h);
    let step = |x: usize| if x < 48 { 0.2 } else { 0.8 };
    let guide: Vec<f64> = (0..w * h).map(|i| step(i % w)).collect();
    let p: Vec<f64> = (0..w * h).map(|i| if i % w < 48 { 0.0 } else { 1.0 }).collect();
    let f = Guided {
        radius: 8,
        step: 1,
        samples: 8,
        eps: 1e-5,
    };
    let q = guided(&p, &guide, &g, &g, &f);
    let worst = q
        .iter()
        .zip(&p)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-3, "{worst}");
}

#[test]
fn a_soft_mask_is_pulled_to_the_edge_of_the_guide() {
    // A blurred mask over a guide with a sharp edge comes back sharper at the guide's edge.
    let (w, h) = (96, 24);
    let g = Area::whole(w, h);
    let guide: Vec<f64> = (0..w * h).map(|i| if i % w < 50 { 0.1 } else { 0.9 }).collect();
    let hard: Vec<f64> = (0..w * h).map(|i| if i % w < 40 { 0.0 } else { 1.0 }).collect();
    let soft = feather(&hard, &g, &g, [5, 5, 5]);
    let f = Guided {
        radius: 12,
        step: 1,
        samples: 12,
        eps: 1e-4,
    };
    let q = guided(&soft, &guide, &g, &g, &f);
    // Just left of the guide's edge the mask is lower than the blurred one was, and the step across the edge
    // is steeper than the blurred mask's.
    let at = |v: &[f64], x: usize| v[12 * w + x];
    assert!(at(&q, 46) < at(&soft, 46) - 0.05, "{} {}", at(&q, 46), at(&soft, 46));
    let jump = |v: &[f64]| at(v, 51) - at(v, 48);
    assert!(jump(&q) > jump(&soft) + 0.1, "{} against {}", jump(&q), jump(&soft));
    assert!(at(&q, 56) > 0.9);
}

/// The Fast Guided Filter written out with explicit loops (independent of `guided`): block means of `step x step`
/// cells aligned at multiples of `step` of the frame, the coefficients with a window of `samples` cells, and
/// their bilinear upsampling.
fn direct_fast_guided(p: &[f64], gd: &[f64], w: usize, h: usize, f: &Guided) -> Vec<f64> {
    let s = f.step;
    let (cw, ch) = (w.div_ceil(s), h.div_ceil(s));
    let block = |v: &[f64]| {
        let mut out = vec![0.0; cw * ch];
        for j in 0..ch {
            for i in 0..cw {
                let mut sum = 0.0;
                let mut n = 0.0;
                for y in j * s..((j + 1) * s).min(h) {
                    for x in i * s..((i + 1) * s).min(w) {
                        sum += v[y * w + x];
                        n += 1.0;
                    }
                }
                out[j * cw + i] = sum / n;
            }
        }
        out
    };
    let (gc, pc) = (block(gd), block(p));
    let r = f.samples;
    let mean = |v: &dyn Fn(usize) -> f64, x: usize, y: usize| {
        let (x0, x1) = (x.saturating_sub(r), (x + r).min(cw - 1));
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(ch - 1));
        let mut sum = 0.0;
        let mut n = 0.0;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                sum += v(yy * cw + xx);
                n += 1.0;
            }
        }
        sum / n
    };
    let mut a = vec![0.0; cw * ch];
    let mut b = vec![0.0; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let mg = mean(&|i| gc[i], x, y);
            let mp = mean(&|i| pc[i], x, y);
            let mgg = mean(&|i| gc[i] * gc[i], x, y);
            let mgp = mean(&|i| gc[i] * pc[i], x, y);
            let var = mgg - mg * mg;
            let cov = mgp - mg * mp;
            a[y * cw + x] = cov / (var + f.eps);
            b[y * cw + x] = mp - a[y * cw + x] * mg;
        }
    }
    let (abar, bbar): (Vec<f64>, Vec<f64>) = (0..cw * ch)
        .map(|i| (mean(&|k| a[k], i % cw, i / cw), mean(&|k| b[k], i % cw, i / cw)))
        .unzip();
    let mut q = vec![0.0; w * h];
    for y in 0..h {
        let vy = (y as f64 + 0.5) / s as f64 - 0.5;
        let (j0, ty) = (vy.floor(), vy - vy.floor());
        let rr = |j: f64| (j as i64).clamp(0, ch as i64 - 1) as usize;
        for x in 0..w {
            let ux = (x as f64 + 0.5) / s as f64 - 0.5;
            let (i0, tx) = (ux.floor(), ux - ux.floor());
            let cc = |i: f64| (i as i64).clamp(0, cw as i64 - 1) as usize;
            let lerp = |v: &[f64]| {
                let top = v[rr(j0) * cw + cc(i0)] * (1.0 - tx) + v[rr(j0) * cw + cc(i0 + 1.0)] * tx;
                let bot = v[rr(j0 + 1.0) * cw + cc(i0)] * (1.0 - tx)
                    + v[rr(j0 + 1.0) * cw + cc(i0 + 1.0)] * tx;
                top * (1.0 - ty) + bot * ty
            };
            q[y * w + x] = (lerp(&abar) * gd[y * w + x] + lerp(&bbar)).clamp(0.0, 1.0);
        }
    }
    q
}

#[test]
fn above_sixteen_pixels_the_guided_filter_is_the_subsampled_form() {
    let (w, h) = (83, 61);
    let g = Area::whole(w, h);
    let mut rnd = noise(11);
    let guide: Vec<f64> = (0..w * h)
        .map(|i| 0.5 + 0.4 * (((i % w) as f64) / 9.0).sin() + 0.1 * rnd())
        .collect();
    let p: Vec<f64> = (0..w * h)
        .map(|i| (((i % w) as f64 - 40.0).hypot((i / w) as f64 - 30.0) < 22.0) as u8 as f64)
        .collect();
    // radius 40 -> step 3, samples 13 (round(40 / 3)); radius 20 -> step 2, samples 10.
    for radius in [20usize, 40] {
        let step = radius.div_ceil(16);
        let f = Guided {
            radius,
            step,
            samples: (radius as f64 / step as f64).round() as usize,
            eps: 1e-4,
        };
        let got = guided(&p, &guide, &g, &g, &f);
        let want = direct_fast_guided(&p, &guide, w, h, &f);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-10, "radius {radius}: {worst}");
        // And it is close to the full guided filter of the same radius (the speed-up costs little).
        let full = direct_guided(&p, &guide, w, h, radius, 1e-4);
        let mean_diff =
            got.iter().zip(&full).map(|(a, b)| (a - b).abs()).sum::<f64>() / (w * h) as f64;
        assert!(mean_diff < 0.03, "radius {radius}: the subsampled form differs by {mean_diff}");
    }
}

// ---- a row's plane: skips, tiles, halos ---------------------------------------------------------

fn textured_row(grow: f64, feather: f64, refine: f64) -> RasterRow {
    // A plane of per-pixel noise (so that a missing pixel of the receptive field shows), with a few runs of 1.
    let mut rnd = noise(3);
    let cells: Vec<f64> = (0..64 * 48).map(|_| rnd()).collect();
    row(
        plane_of(64, 48, move |x, y| {
            let v = cells[y * 64 + x];
            if v > 0.6 { 1.0 } else { v * 0.5 }
        }),
        grow,
        feather,
        refine,
    )
}

fn guide_of(w: usize, h: usize) -> Vec<f64> {
    let mut rnd = noise(5);
    (0..w * h)
        .map(|i| 0.5 + 0.35 * (((i % w) as f64) / 7.0).sin() * (((i / w) as f64) / 11.0).cos() + 0.05 * rnd())
        .collect()
}

#[test]
fn a_row_with_no_step_is_the_resample_quantised() {
    let (w, h) = (80usize, 60usize);
    let r = textured_row(0.0, 0.0, 0.0);
    let pos = identity_position(w, h);
    let photo = PhotoFrame::new(w as f64, h as f64, 1.0);
    let job = PlaneJob {
        photo,
        grown: Area::whole(w, h),
        frame: Area::whole(w, h),
        window: Area::whole(w, h),
        position: &pos,
        guide: None,
    };
    let plane = r.plane(&job);
    let RasterSource::Plane(src) = &r.source else {
        unreachable!()
    };
    let taps = resample_taps(64, 80.0);
    for y in 0..h {
        for x in 0..w {
            let want = resample_at(src, pos(x, y).expect("inside"), taps, 64.0 / 80.0);
            assert_eq!(plane[y * w + x], quantise(want), "({x}, {y})");
        }
    }
}

#[test]
fn a_tile_over_its_halo_equals_the_whole_frame_bit_for_bit() {
    let (w, h) = (150usize, 110usize);
    // Grow, Feather and Refine edges together, the last subsampled (u_px 150 -> radius 1 ... use a big
    // scale through the photo frame's pixel so that the radii are large).
    let u_px = 2200.0;
    let photo = PhotoFrame::new(w as f64, h as f64, w as f64 / u_px);
    let pos = identity_position(w, h);
    let guide = guide_of(w, h);
    for (grow_amount, feather_amount, refine_amount) in [(0.4, 0.0, 0.0), (0.0, 0.3, 0.0), (0.3, 0.3, 0.5), (0.0, 0.0, 1.0)] {
        let r = textured_row(grow_amount, feather_amount, refine_amount);
        let steps = RasterSteps::of(grow_amount, feather_amount, refine_amount, u_px);
        let halo = steps.halo();
        assert!(halo > 8 && halo < 80, "{halo}");
        let frame = Area::whole(w, h);
        let whole = r.plane(&PlaneJob {
            photo,
            grown: frame,
            frame,
            window: frame,
            position: &pos,
            guide: Some(&guide),
        });
        for tile in [Area { x: 33, y: 20, w: 40, h: 37 }, Area { x: 0, y: 0, w: 61, h: 50 }, Area { x: 100, y: 70, w: 50, h: 40 }] {
            let grown = tile.grown(halo, &frame);
            // The tile's own view of the position and guide planes: pixel (x, y) of the grown area.
            let pos_g = |x: usize, y: usize| pos((grown.x as usize) + x, (grown.y as usize) + y);
            let guide_g: Vec<f64> = (0..grown.h)
                .flat_map(|y| (0..grown.w).map(move |x| (x, y)))
                .map(|(x, y)| guide[(grown.y as usize + y) * w + grown.x as usize + x])
                .collect();
            let cut = r.plane(&PlaneJob {
                photo,
                grown,
                frame,
                window: tile,
                position: &pos_g,
                guide: Some(&guide_g),
            });
            for y in 0..tile.h {
                for x in 0..tile.w {
                    let a = cut[y * tile.w + x];
                    let b = whole[(tile.y as usize + y) * w + tile.x as usize + x];
                    assert_eq!(a, b, "grow {grow_amount} feather {feather_amount} refine {refine_amount}: tile {tile:?} pixel ({x}, {y})");
                }
            }
        }
    }
}

#[test]
fn a_halo_that_is_one_pixel_short_changes_the_tile() {
    // The test above has teeth: with a halo one pixel too small the tile differs somewhere.
    let (w, h) = (120usize, 90usize);
    let u_px = 1500.0;
    let photo = PhotoFrame::new(w as f64, h as f64, w as f64 / u_px);
    let pos = identity_position(w, h);
    let r = textured_row(0.3, 0.3, 0.0);
    let halo = RasterSteps::of(0.3, 0.3, 0.0, u_px).halo();
    let frame = Area::whole(w, h);
    let whole = r.plane(&PlaneJob {
        photo,
        grown: frame,
        frame,
        window: frame,
        position: &pos,
        guide: None,
    });
    let tile = Area { x: 40, y: 30, w: 30, h: 30 };
    let grown = tile.grown(halo - 1, &frame);
    let pos_g = |x: usize, y: usize| pos(grown.x as usize + x, grown.y as usize + y);
    let cut = r.plane(&PlaneJob {
        photo,
        grown,
        frame,
        window: tile,
        position: &pos_g,
        guide: None,
    });
    let differs = (0..tile.h).any(|y| {
        (0..tile.w).any(|x| cut[y * tile.w + x] != whole[(tile.y as usize + y) * w + tile.x as usize + x])
    });
    assert!(differs, "one pixel short must show");
}

#[test]
fn a_pixel_outside_the_photo_is_zero_before_the_steps() {
    let (w, h) = (40usize, 40usize);
    let photo = PhotoFrame::new(w as f64, h as f64, 1.0);
    let r = row(plane_of(8, 8, |_, _| 1.0), 0.0, 0.0, 0.0);
    let pos = |x: usize, y: usize| (x >= 10).then(|| [(x as f64 + 0.5) / 40.0, (y as f64 + 0.5) / 40.0]);
    let frame = Area::whole(w, h);
    let plane = r.plane(&PlaneJob {
        photo,
        grown: frame,
        frame,
        window: frame,
        position: &pos,
        guide: None,
    });
    assert!(plane[5 * w + 3] == 0 && plane[5 * w + 30] == 65535);
}

// ---- the chain reads the plane ------------------------------------------------------------------

struct Positions(usize, usize);
impl PhotoMap for Positions {
    fn position(&self, x: usize, y: usize) -> [f64; 2] {
        [(x as f64 + 0.5) / self.0 as f64, (y as f64 + 0.5) / self.1 as f64]
    }
}

fn chain_with_raster(rows: Vec<MaskRow>, set: RasterSet) -> ColorChain {
    let graph = PlanGraph {
        nodes: vec![PlanNode {
            id: 1,
            pos: [0.0, 0.0],
            picture: Some(PictureSource::Input),
            key: None,
        }],
        output: PictureSource::Node(1),
    };
    let plan = compile(&graph, Stop::Output).expect("a plan");
    let mut p = ChainParams::new(
        ym_color::consts::linalg::IDENTITY,
        ToneParams::IDENTITY,
    );
    p.nodes = NodeSet {
        plan,
        main: ym_color::display::MainNode {
            masks: NodeMasks {
                rows,
                invert: false,
            },
            ..Default::default()
        },
        others: Default::default(),
    };
    p.grade.contrast = 40.0;
    p.rasters = set;
    ColorChain::new(&p).expect("a chain")
}

#[test]
fn a_raster_row_reads_its_plane_through_the_frame() {
    let (w, h) = (16usize, 8usize);
    let set = RasterSet {
        rows: vec![row(plane_of(4, 4, |_, _| 1.0), 0.0, 0.0, 0.0)],
    };
    let chain = chain_with_raster(vec![MaskRow::new(MaskKind::Raster { plane: 0 })], set);
    assert!(chain.uses_rasters() && chain.uses_photo_position());
    assert_eq!(chain.rasters().rows.len(), 1);
    let cam: Vec<f32> = (0..3 * w * h).map(|i| 0.05 + (i % 29) as f32 * 0.01).collect();
    let map = Positions(w, h);
    let photo = PhotoFrame::new(w as f64, h as f64, 1.0);
    let render = |planes: Option<&CpuPlanes>| {
        let mut f = Frame::planar(w, h, &cam)
            .expect("a frame")
            .with_photo(&map, photo);
        if let Some(p) = planes {
            f = f.with_rasters(p);
        }
        let mut out = vec![0.0f32; 3 * w * h];
        chain
            .render_encoded_f32(View::Final, &f, Layout::Interleaved, &mut out, 1)
            .expect("a render");
        out
    };
    // No planes: the row selects nothing and the node changes nothing.
    let none = render(None);
    let full = CpuPlanes {
        width: w,
        height: h,
        planes: vec![vec![65535; w * h]],
    };
    let all = render(Some(&full));
    assert_ne!(none, all, "the plane acts");
    // A plane of zero is the same as no plane.
    let zero = CpuPlanes {
        width: w,
        height: h,
        planes: vec![vec![0; w * h]],
    };
    assert_eq!(render(Some(&zero)), none);
    // Half a plane (the left half 1) changes the left half only.
    let half = CpuPlanes {
        width: w,
        height: h,
        planes: vec![(0..w * h).map(|i| if i % w < w / 2 { 65535 } else { 0 }).collect()],
    };
    let part = render(Some(&half));
    for y in 0..h {
        for x in 0..w {
            let i = 3 * (y * w + x);
            let (a, b, c) = (none[i], part[i], all[i]);
            if x < w / 2 {
                assert_eq!(b, c, "({x}, {y})");
            } else {
                assert_eq!(b, a, "({x}, {y})");
            }
        }
    }
    // The planes are read as u16 codes over 65535.
    assert!((half.value(0, 0, 0) - 1.0).abs() < 1e-15 && half.value(0, w - 1, 0) == 0.0);
    assert_eq!(half.value(3, 0, 0), 0.0, "a plane the render does not have");
}

#[test]
fn the_guide_is_the_inputs_luma_with_no_node_in_it() {
    let set = RasterSet::default();
    let plain = chain_with_raster(vec![], set.clone());
    let graded = {
        let mut p = ChainParams::new(ym_color::consts::linalg::IDENTITY, ToneParams::IDENTITY);
        p.grade.contrast = 80.0;
        p.grade.saturation = 60.0;
        ColorChain::new(&p).expect("a chain")
    };
    for cam in [[0.2, 0.3, 0.1], [0.8, 0.1, 0.4], [0.0, 0.0, 0.0], [5.0, 4.0, 3.0]] {
        let a = plain.guide_pixel(cam);
        assert!((0.0..=1.0).contains(&a));
        assert_eq!(a, graded.guide_pixel(cam), "{cam:?}");
        // It is the C7 luma code the Brightness range of a node that reads the Input sees.
        let t = plain.trace(cam);
        let l = ym_color::logc::logc4_luma_of_linear(ym_color::gamut::LinearAwg4(t.c7_linear_awg4));
        assert_eq!(a, l.clamp(0.0, 1.0));
    }
    let _: Vec3 = [0.0; 3];
}

// ---- Brush --------------------------------------------------------------------------------------

fn stroke(size: f64, hardness: f64, erase: bool, points: &[[f64; 2]]) -> Stroke {
    Stroke {
        size,
        hardness,
        erase,
        points: points.to_vec(),
    }
}

fn dab(plane: &SourcePlane, x: f64, y: f64) -> f64 {
    plane.at((x * plane.width as f64) as i64, (y * plane.height as f64) as i64)
}

#[test]
fn a_single_point_is_one_stamp_with_the_hardness_ramp() {
    // D = 0.1 of the long edge = 204.8 plane pixels, R = 102.4. Hardness 0.5: the ramp is 51.2 pixels.
    for hardness in [0.0, 0.5, 0.9] {
        let b = BrushStrokes::new(vec![stroke(0.1, hardness, false, &[[0.5, 0.5]])]);
        let p = draw_brush(&b, [1.0, 1.0]);
        assert_eq!((p.width, p.height), (2048, 2048));
        let (cx, cy) = (1024.0, 1024.0);
        let radius = 102.4;
        let ramp = ((1.0 - hardness) * radius).max(1.0);
        let inner = radius - ramp;
        for d in [0.0, inner * 0.5, inner, inner + 0.25 * ramp, inner + 0.5 * ramp, inner + 0.75 * ramp, radius - 0.5, radius + 2.0] {
            // The pixel whose centre is d to the right of the stamp's centre.
            let x = (cx + d - 0.5).round() as i64;
            let centre_d = (x as f64 + 0.5 - cx).hypot(0.5 - 0.0 + (1024.0_f64 - 0.5 - cy).abs() * 0.0);
            let _ = centre_d;
            let y = 1024i64;
            let dist = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
            let want = if dist <= inner {
                1.0
            } else if dist >= radius {
                0.0
            } else {
                let t = (dist - inner) / (radius - inner);
                1.0 - t * t * (3.0 - 2.0 * t)
            };
            let got = p.at(x, y);
            assert!((got - want).abs() < 2e-5, "hardness {hardness} d {d}: {got} against {want}");
        }
        // Nothing outside the brush.
        assert_eq!(p.at(1024 + 110, 1024), 0.0);
        assert_eq!(p.at(10, 10), 0.0);
    }
}

#[test]
fn a_hard_brush_has_a_ramp_of_one_plane_pixel() {
    let b = BrushStrokes::new(vec![stroke(0.1, 1.0, false, &[[0.5, 0.5]])]);
    let p = draw_brush(&b, [1.0, 1.0]);
    let radius = 102.4;
    let row: Vec<f64> = (0..2048).map(|x| p.at(x, 1024)).collect();
    // Inside radius - 1 it is 1, beyond the radius 0, and between the two a smooth fall of about a pixel.
    let first_below = row[1024..].iter().position(|&v| v < 1.0).expect("a fall") + 1024;
    let last_above = row[1024..].iter().rposition(|&v| v > 0.0).expect("a fall") + 1024;
    assert!(last_above - first_below <= 2, "{first_below} {last_above}");
    assert!(((first_below as f64 + 0.5) - 1024.0 - radius).abs() < 2.0);
}

#[test]
fn paint_then_erase_removes_what_was_painted_and_the_order_matters() {
    let paint = stroke(0.05, 1.0, false, &[[0.5, 0.5]]);
    let erase = stroke(0.02, 1.0, true, &[[0.5, 0.5]]);
    let painted = draw_brush(&BrushStrokes::new(vec![paint.clone()]), [1.0, 1.0]);
    let then_erase = draw_brush(&BrushStrokes::new(vec![paint.clone(), erase.clone()]), [1.0, 1.0]);
    let erase_then = draw_brush(&BrushStrokes::new(vec![erase.clone(), paint.clone()]), [1.0, 1.0]);
    assert!(dab(&painted, 0.5, 0.5) > 0.99);
    // Painted, then a smaller eraser at the centre: a hole.
    assert!(dab(&then_erase, 0.5, 0.5) < 0.01);
    // The disc beyond the eraser is still painted.
    assert!(then_erase.at(1024 + 45, 1024) > 0.99);
    // Erasing before painting erases nothing.
    assert_eq!(erase_then.data, painted.data);
    assert_ne!(then_erase.data, erase_then.data);
}

#[test]
fn strokes_overlap_by_the_screen_rule() {
    // Two soft stamps on one spot: 1 - (1 - a)(1 - a), never more than 1.
    let one = draw_brush(&BrushStrokes::new(vec![stroke(0.2, 0.0, false, &[[0.5, 0.5]])]), [1.0, 1.0]);
    let two = draw_brush(
        &BrushStrokes::new(vec![
            stroke(0.2, 0.0, false, &[[0.5, 0.5]]),
            stroke(0.2, 0.0, false, &[[0.5, 0.5]]),
        ]),
        [1.0, 1.0],
    );
    for x in [1100i64, 1150, 1200, 1250] {
        let a = one.at(x, 1024);
        let both = two.at(x, 1024);
        assert!((both - (1.0 - (1.0 - a) * (1.0 - a))).abs() < 3e-5, "{x}: {both}");
        assert!(both <= 1.0);
    }
}

#[test]
fn the_path_is_stamped_every_fifteen_hundredths_of_the_diameter() {
    // A horizontal stroke of D = 0.02 (40.96 px, spacing 6.14 px) over 0.5 of the width (1024 px): the
    // stamps are evenly spaced, so the painted bar is even along its length: no scallops.
    let b = BrushStrokes::new(vec![stroke(0.02, 0.0, false, &[[0.25, 0.5], [0.75, 0.5]])]);
    let p = draw_brush(&b, [1.0, 1.0]);
    let along: Vec<f64> = (600..1400).map(|x| p.at(x, 1024)).collect();
    let (lo, hi) = along
        .iter()
        .fold((1.0f64, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
    assert!(hi - lo < 0.02, "the bar is uneven: {lo} to {hi}");
    // It starts and ends at the stroke's ends (one radius beyond them).
    assert!(p.at(1024 - 512 - 8, 1024) > 0.3);
    assert!(p.at(1024 - 512 - 25, 1024) < 0.05);
    assert!(p.at(1024 + 512 + 8, 1024) > 0.3);
    assert!(p.at(1024 + 512 + 25, 1024) < 0.05);
    // A path through a corner is the same as the two legs drawn one after the other as strokes.
    let bent = draw_brush(&BrushStrokes::new(vec![stroke(0.02, 1.0, false, &[[0.3, 0.3], [0.6, 0.3], [0.6, 0.6]])]), [1.0, 1.0]);
    assert!(bent.at((0.45 * 2048.0) as i64, (0.3 * 2048.0) as i64) > 0.99);
    assert!(bent.at((0.6 * 2048.0) as i64, (0.45 * 2048.0) as i64) > 0.99);
    assert!(bent.at((0.45 * 2048.0) as i64, (0.45 * 2048.0) as i64) < 0.01);
}

#[test]
fn a_plane_has_the_proportions_of_the_photo() {
    let b = BrushStrokes::new(vec![stroke(0.1, 1.0, false, &[[0.5, 0.5]])]);
    let p = draw_brush(&b, [1.0, 2.0 / 3.0]);
    assert_eq!((p.width, p.height), (2048, 1365));
    // The dab is a circle in photo pixels: as wide as it is tall.
    let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (i64::MAX, 0, i64::MAX, 0);
    for y in 0..p.height as i64 {
        for x in 0..p.width as i64 {
            if p.at(x, y) > 0.5 {
                x_lo = x_lo.min(x);
                x_hi = x_hi.max(x);
                y_lo = y_lo.min(y);
                y_hi = y_hi.max(y);
            }
        }
    }
    let (dw, dh) = (x_hi - x_lo, y_hi - y_lo);
    assert!((dw - dh).abs() <= 1, "{dw} x {dh}");
    assert!((dw as f64 - 204.8).abs() < 3.0);
    // Portrait.
    let q = draw_brush(&b, [2.0 / 3.0, 1.0]);
    assert_eq!((q.width, q.height), (1365, 2048));
}

#[test]
fn the_same_strokes_make_the_same_bytes_and_come_from_the_cache() {
    let mk = || {
        BrushStrokes::new(vec![
            stroke(0.04, 0.3, false, &[[0.1, 0.2], [0.4, 0.5], [0.8, 0.3]]),
            stroke(0.01, 0.8, true, &[[0.3, 0.4], [0.5, 0.45]]),
        ])
    };
    let (a, b) = (mk(), mk());
    assert_eq!(a.key, b.key);
    assert_eq!(draw_brush(&a, [1.0, 0.75]).data, draw_brush(&b, [1.0, 0.75]).data);
    let c = BrushStrokes::new(vec![stroke(0.04, 0.3, false, &[[0.1, 0.2], [0.4, 0.5], [0.8, 0.31]])]);
    assert_ne!(a.key, c.key);
    let first = brush_plane(&a, [1.0, 0.75]);
    let second = brush_plane(&b, [1.0, 0.75]);
    assert!(Arc::ptr_eq(&first, &second), "the second call is the cache's");
    let other = brush_plane(&a, [1.0, 0.5]);
    assert!(!Arc::ptr_eq(&first, &other));
    assert!(ym_color::mask_raster::brush_cache_len() <= 8);
}

#[test]
fn a_row_of_strokes_goes_through_the_steps_like_any_source() {
    // A Brush row's plane with Feather: a soft edge where the dab had a hard one.
    let (w, h) = (200usize, 200usize);
    let photo = PhotoFrame::new(w as f64, h as f64, 1.0); // u_px 200 -> sigma = f * 4
    let b = Arc::new(BrushStrokes::new(vec![stroke(0.3, 1.0, false, &[[0.5, 0.5]])]));
    let make = |feather: f64| RasterRow {
        source: RasterSource::Brush(Arc::clone(&b)),
        grow: 0.0,
        feather,
        refine: 0.0,
    };
    let pos = identity_position(w, h);
    let frame = Area::whole(w, h);
    let plane = |r: &RasterRow| {
        r.plane(&PlaneJob {
            photo,
            grown: frame,
            frame,
            window: frame,
            position: &pos,
            guide: None,
        })
    };
    let hard = plane(&make(0.0));
    let soft = plane(&make(1.0));
    let ramp = |v: &[u16]| v[100 * w..101 * w].iter().filter(|&&c| c > 1000 && c < 64500).count();
    assert!(ramp(&soft) > ramp(&hard) + 4, "{} {}", ramp(&soft), ramp(&hard));
    assert_eq!(codes(&hard).iter().filter(|&&v| v > 0.99).count() > 1000, true);
}
