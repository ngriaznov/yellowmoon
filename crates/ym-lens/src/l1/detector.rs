//! The bokeh detector (`docs/21` section 7): the blur the photo already holds, measured from the photo alone.
//!
//! The work is done once per photo on the lens plane (or its half when the plane is large), in `f64` where it
//! sums and in `f32` where it stores, in a fixed order, so that the maps are the same bits on every platform and
//! for any number of threads.
//!
//! 1. **Blur at edges.** The luminance goes up a ladder of Gaussians (read in linear light, where defocus is a
//!    convolution; noise is judged in the square-root domain, where it is nearly constant). At each strong edge (a maximum of the gradient along its direction) the ratio of the
//!    gradient magnitudes between two ladder levels gives the radius of the disc that blurred the edge
//!    ([`super::disc`]); every edge is read at the levels that resolve it, each reading weighted by how well
//!    its ratio is conditioned and how far it stands above the noise.
//! 2. **The subject anchor.** The camera's focus point marks its neighbourhood as in focus.
//! 3. **Bright balls** are found and measured by their size ([`super::balls`]); their edges, which a sharp rim
//!    would make read sharp, are dropped and their radius takes their place.
//! 4. **Spread.** The sparse readings are spread over the plane by edge-aware normalised convolution
//!    ([`super::spread`]).
//! 5. **Caution.** The blur is 0 where the evidence is thin or inconsistent, where it is below the smallest
//!    radius, in the anchor, and next to a strong sharp edge: when unsure, nothing is reshaped.

use rayon::prelude::*;

use crate::consts::{B_CAP, B_MIN, MIN_EDGE_SAMPLES};
use crate::error::LensError;
use crate::image::ImageRef;

use super::balls::{BallInput, BallPx, find_balls};
use super::disc::{LEVELS, PAIRS, RadiusEstimate, level_sigma, radius_from_ratio};
use super::ladder::{blur_wide, gradient_at, gradient_magnitude, quantile};
use super::spread::{GuidePyramid, spread};

/// A plane whose long edge is at least this many pixels is detected on its 2x reduction.
const REDUCE_FROM: usize = 1536;
/// The smallest disc radius the detector reports, in pixels, whatever `b_min` is on a small plane: the native
/// blur of a sharp edge on the binned plane reads as a disc of about one pixel, and the margin keeps it from
/// being reshaped.
const RADIUS_FLOOR_PX: f64 = 2.2;
/// The noise of the plane is judged at the finest scale; this allows for noise correlated over a few pixels.
const NOISE_ALLOWANCE: f64 = 2.5;
/// A reading is dropped when a reading of at least this weight, from a finer pair, within `CONSISTENCY_REACH`
/// of its radius, is below `CONSISTENCY_RATIO` of its radius.
const CONSISTENCY_WEIGHT: f64 = 0.15;
const CONSISTENCY_REACH: f64 = 1.5;
const CONSISTENCY_RATIO: f64 = 0.55;
/// An edge is a gradient maximum above this many noise units.
const EDGE_KAPPA: f64 = 5.0;
/// ... and above this absolute gradient (square-root units per pixel), for noiseless data.
const EDGE_ABS_MIN: f64 = 0.002;
/// Relative error of a ratio that no noise explains (the step-edge model against real edges).
const MODEL_ERR: f64 = 0.05;
/// A gradient ratio of the coarsest pair above this (and at or below the table's last entry) is an edge wider
/// than the table; it reads as `b_cap`.
const WIDE_RHO_MIN: f64 = 0.9;
/// A reading of the coarsest pair from this fraction of `b_cap` up is taken as `b_cap`.
const WIDE_FROM: f64 = 0.6;
/// Sensitivity assumed for such a reading (it is a bound, not a measure).
const WIDE_SENSITIVITY: f64 = 3.0;
/// A reading whose log radius has this variance has weight 1.
const WEIGHT_REF_VAR: f64 = 0.04;
/// Readings lighter than this are dropped.
const MIN_SAMPLE_WEIGHT: f64 = 0.02;
/// Colour tolerance of the spread (square-root units): textured backgrounds have many colours, so the guide only
/// stops the spread at large differences.
const GUIDE_TOLERANCE: f64 = 0.25;
/// A level of the spread is trusted where the readings in its window weigh this much (a window of 9 cells
/// holds a kernel sum of about 4, so this is a fraction of a reading of full weight).
const EVIDENCE_NEED: f64 = 0.25;
/// Variance of the log radius at which the confidence is halved (35 % spread).
const VARIANCE_HALF: f64 = 0.12;
/// Below this confidence the blur is taken as 0.
const CONFIDENCE_MIN: f64 = 0.3;
/// The anchor around the focus point, in half-diagonals, and its weight per pixel.
const ANCHOR_RADIUS: f64 = 0.05;
const ANCHOR_WEIGHT: f64 = 2.0;
/// The ball footprint that carries the ball's radius into the map, in radii.
const BALL_FOOTPRINT: f64 = 2.6;
/// Total evidence weight a ball adds, spread over its disc and ring.
const BALL_WEIGHT: f64 = 30.0;
/// A sharp edge flags the places around it when its gradient is this many noise units.
const SHARP_STRENGTH: f64 = 5.0;
/// A blurred region must cover at least this many discs of its own mean blur radius.
const SMALL_REGION: f64 = 2.0;
/// Dilation, in pixels, of the places that show a strong sharp edge.
const SHARP_DILATION: i64 = 2;

/// The plane the detector works on.
pub(super) struct WorkPlane {
    pub w: usize,
    pub h: usize,
    /// How many pixels of the lens plane one pixel of this plane covers (1 or 2).
    pub reduce: usize,
    pub rgb: [Vec<f32>; 3],
    pub y: Vec<f32>,
}

/// What the detector found.
pub(super) struct Detected {
    pub w: usize,
    pub h: usize,
    /// Blur radius per pixel in half-diagonals (0 where nothing is reshaped).
    pub b: Vec<f32>,
    /// Confidence of the value above (of 0, too).
    pub conf: Vec<f32>,
    pub balls: Vec<BallPx>,
    /// Edges measured (readings of weight at least 0.1, plus the perimeters of the balls).
    pub edges: usize,
    /// Noise of the square-root luminance per pixel.
    pub noise_v: f64,
    /// Half-diagonal of the working plane in pixels.
    pub hd: f64,
    pub reduce: usize,
}

/// Luminance of a linear camera RGB triple.
#[inline]
pub(super) fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.25 * r + 0.5 * g + 0.25 * b
}

/// The lens plane as the detector's working plane: itself, or 2x2 box means when it is large.
pub(super) fn work_plane(plane: &ImageRef) -> WorkPlane {
    let (w0, h0) = (plane.width, plane.height);
    let reduce = if w0.max(h0) >= REDUCE_FROM { 2 } else { 1 };
    let (w, h) = (w0.div_ceil(reduce), h0.div_ceil(reduce));
    let src = [plane.plane(0), plane.plane(1), plane.plane(2)];
    let mut rgb = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];
    for (c, out) in rgb.iter_mut().enumerate() {
        if reduce == 1 {
            out.copy_from_slice(src[c]);
            continue;
        }
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let (mut acc, mut n) = (0.0f32, 0.0f32);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (sx, sy) = (2 * x + dx, 2 * y + dy);
                        if sx < w0 && sy < h0 {
                            acc += src[c][sy * w0 + sx];
                            n += 1.0;
                        }
                    }
                }
                *o = acc / n;
            }
        });
    }
    let y: Vec<f32> = (0..w * h)
        .map(|i| luminance(rgb[0][i], rgb[1][i], rgb[2][i]))
        .collect();
    WorkPlane {
        w,
        h,
        reduce,
        rgb,
        y,
    }
}

/// Noise of a plane of square-root luminance (Immerkaer's Laplacian estimate, from a low quantile so that
/// texture inflates it little).
pub(super) fn noise_sigma(v: &[f32], w: usize, h: usize) -> f64 {
    if w < 3 || h < 3 {
        return 0.0015;
    }
    let mut vals = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let at = |dx: usize, dy: usize| f64::from(v[(y + dy - 1) * w + (x + dx - 1)]);
            let n = 4.0 * at(1, 1) - 2.0 * (at(0, 1) + at(2, 1) + at(1, 0) + at(1, 2))
                + (at(0, 0) + at(2, 0) + at(0, 2) + at(2, 2));
            vals.push(n.abs() as f32);
        }
    }
    let p = quantile(&mut vals, 0.3);
    (p / (0.385 * 6.0)).max(0.0015)
}

/// The scale ladder of a plane: level 0 is the plane, level `m >= 1` is blurred by `level_sigma(m)` pixels.
fn build_ladder(
    v: &[f32],
    w: usize,
    h: usize,
    cancel: &dyn Fn() -> bool,
) -> Result<Vec<Vec<f32>>, LensError> {
    let mut levels: Vec<Vec<f32>> = Vec::with_capacity(LEVELS);
    levels.push(v.to_vec());
    for m in 0..LEVELS - 1 {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        // Gaussians add in quadrature: the step from level m to m + 1.
        let step = (level_sigma(m + 1).powi(2) - level_sigma(m).powi(2)).sqrt();
        let next = blur_wide(&levels[m], w, h, step);
        levels.push(next);
    }
    Ok(levels)
}

/// One reading of the blur: a pixel, its weight and the log of the disc radius in pixels.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sample {
    pub idx: u32,
    pub w: f32,
    pub ln_r: f32,
}

/// Floor of the luminance when normalising a gradient by the noise it carries.
fn noise_floor(noise_v: f64) -> f64 {
    noise_v * noise_v
}

/// The gradient magnitude of a ladder level divided by twice the square root of its luminance: what the
/// gradient of the square-root luminance would be. Sensor noise is nearly constant in that domain, so this is
/// the contrast-to-noise scale of an edge whatever its brightness. The edges themselves are read on the
/// linear ladder, where a defocus blur is exactly a convolution.
#[inline]
fn normalised(g: f32, level: f32, floor: f64) -> f64 {
    f64::from(g) / (2.0 * (f64::from(level).max(0.0) + floor).sqrt())
}

/// Noise scale of the normalised gradient at ladder level `m`: white noise of the plane's level through the
/// Gaussian and the central difference, times an allowance for the noise being correlated over a few pixels
/// (demosaicing, binning). It comes from the plane's noise and not from the gradients themselves, which in a
/// textured photo are mostly signal.
fn gradient_noise(m: usize, noise_v: f64) -> f64 {
    let s = level_sigma(m);
    let white = if m == 0 {
        noise_v * std::f64::consts::FRAC_1_SQRT_2
    } else {
        noise_v / ((8.0 * std::f64::consts::PI).sqrt() * s * s)
    };
    (NOISE_ALLOWANCE * white).max(1e-7)
}

/// The edge readings of the ladder, and the places that show a strong sharp edge.
fn edge_samples(
    ladder: &[Vec<f32>],
    w: usize,
    h: usize,
    noise_v: f64,
    r_min_px: f64,
    r_wide_px: f64,
    cancel: &dyn Fn() -> bool,
) -> Result<(Vec<Sample>, Vec<bool>), LensError> {
    let mut samples = Vec::new();
    let mut sharp = vec![false; w * h];
    // The smallest radius among the weighty readings of the finer pairs, per pixel.
    let mut fine_min = vec![f32::INFINITY; w * h];
    let floor = noise_floor(noise_v);
    let mut gj = gradient_magnitude(&ladder[PAIRS[0].0], w, h);
    let mut nj = gradient_noise(PAIRS[0].0, noise_v);
    for (pair, &(j, k)) in PAIRS.iter().enumerate() {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let gk = gradient_magnitude(&ladder[k], w, h);
        let nk = gradient_noise(k, noise_v);
        let lj = &ladder[j];
        let lk = &ladder[k];
        let thr = (EDGE_KAPPA * nj).max(EDGE_ABS_MIN);
        // Rows in parallel, each returning its readings in column order.
        let rows: Vec<(Vec<Sample>, Vec<usize>)> = (1..h - 1)
            .into_par_iter()
            .map(|y| {
                let mut out = Vec::new();
                let mut sharp_here = Vec::new();
                for x in 1..w - 1 {
                    let i = y * w + x;
                    let g = gj[i];
                    // `normalised(g, level) >= thr` without the square root.
                    if f64::from(g) * f64::from(g)
                        < 4.0 * thr * thr * (f64::from(lj[i]).max(0.0) + floor)
                    {
                        continue;
                    }
                    let gn = normalised(g, lj[i], floor);
                    // Non-maximum suppression along the gradient direction (4 sectors).
                    let (gx, gy) = gradient_at(lj, w, h, x, y);
                    let (ax, ay) = (gx.abs(), gy.abs());
                    let (dx, dy): (i64, i64) = if ay <= 0.4142 * ax {
                        (1, 0)
                    } else if ax <= 0.4142 * ay {
                        (0, 1)
                    } else if gx * gy > 0.0 {
                        (1, 1)
                    } else {
                        (1, -1)
                    };
                    let a = gj[((y as i64 + dy) as usize) * w + (x as i64 + dx) as usize];
                    let b = gj[((y as i64 - dy) as usize) * w + (x as i64 - dx) as usize];
                    if g < a || g <= b {
                        continue;
                    }
                    // The second level's peak around this pixel.
                    let mut gk_peak = 0.0f32;
                    let mut at_peak = i;
                    for yy in y - 1..=y + 1 {
                        for xx in x - 1..=x + 1 {
                            if gk[yy * w + xx] > gk_peak {
                                gk_peak = gk[yy * w + xx];
                                at_peak = yy * w + xx;
                            }
                        }
                    }
                    if gk_peak <= 0.0 {
                        continue;
                    }
                    let gkn = normalised(gk_peak, lk[at_peak], floor);
                    let rho = f64::from(g) / f64::from(gk_peak);
                    let mut est = match radius_from_ratio(pair, rho) {
                        Some(e) => e,
                        // An edge the coarsest pair cannot resolve is wider than the table.
                        None if pair == PAIRS.len() - 1
                            && rho.is_finite()
                            && rho > WIDE_RHO_MIN =>
                        {
                            RadiusEstimate {
                                r: r_wide_px,
                                sensitivity: WIDE_SENSITIVITY,
                            }
                        }
                        None => continue,
                    };
                    // A wide edge is blurred at least as much as the stretch ever uses (`r_wide_px`), which is
                    // all the map needs of it: the pair's ratio there is too flat to give more, and its
                    // sensitivity would give the reading no weight at all.
                    if pair == PAIRS.len() - 1 && est.r >= WIDE_FROM * r_wide_px {
                        est = RadiusEstimate {
                            r: r_wide_px,
                            sensitivity: est.sensitivity.min(WIDE_SENSITIVITY),
                        };
                    }
                    let eps2 = (nj / gn).powi(2) + (nk / gkn).powi(2) + MODEL_ERR * MODEL_ERR;
                    let var = est.sensitivity * est.sensitivity * eps2;
                    let weight = (WEIGHT_REF_VAR / var).min(1.0);
                    if pair == 0 && est.r < 0.5 * r_min_px && gn > SHARP_STRENGTH * nj {
                        sharp_here.push(i);
                    }
                    if weight < MIN_SAMPLE_WEIGHT {
                        continue;
                    }
                    out.push(Sample {
                        idx: i as u32,
                        w: weight as f32,
                        ln_r: libm::log(est.r) as f32,
                    });
                }
                (out, sharp_here)
            })
            .collect();
        let mut found: Vec<Sample> = Vec::new();
        for (out, sh) in rows {
            found.extend(out);
            for i in sh {
                sharp[i] = true;
            }
        }
        // Scale consistency: a region blurred by a disc holds no edge much sharper than the disc, so a reading
        // with sharper readings of weight around it measures a texture of many edges, not a blurred edge.
        if pair > 0 {
            found = found
                .into_par_iter()
                .filter(|s| {
                    let r = f64::from(s.ln_r).exp();
                    let reach = ((CONSISTENCY_REACH * r + 1.0).ceil() as i64).min(48);
                    let limit = (CONSISTENCY_RATIO * r) as f32;
                    let (x, y) = ((s.idx as usize % w) as i64, (s.idx as usize / w) as i64);
                    for yy in (y - reach).max(0)..=(y + reach).min(h as i64 - 1) {
                        for xx in (x - reach).max(0)..=(x + reach).min(w as i64 - 1) {
                            if fine_min[yy as usize * w + xx as usize] < limit {
                                return false;
                            }
                        }
                    }
                    true
                })
                .collect();
        }
        for s in &found {
            if f64::from(s.w) >= CONSISTENCY_WEIGHT {
                let r = f64::from(s.ln_r).exp() as f32;
                let m = &mut fine_min[s.idx as usize];
                *m = m.min(r);
            }
        }
        samples.extend(found);
        gj = gk;
        nj = nk;
    }
    Ok((samples, sharp))
}

/// The three sums per pixel of a list of readings.
fn accumulate(samples: &[Sample], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut w, mut s, mut q) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    for sm in samples {
        let i = sm.idx as usize;
        let (wt, l) = (f64::from(sm.w), f64::from(sm.ln_r));
        w[i] += wt;
        s[i] += wt * l;
        q[i] += wt * l * l;
    }
    (
        w.into_iter().map(|v| v as f32).collect(),
        s.into_iter().map(|v| v as f32).collect(),
        q.into_iter().map(|v| v as f32).collect(),
    )
}

/// The guide: the square root of each channel, smoothed by (1, 2, 1) / 4 both ways.
fn guide_image(rgb: &[Vec<f32>; 3], w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; w * h];
    for c in 0..3 {
        let root: Vec<f32> = rgb[c].iter().map(|v| v.max(0.0).sqrt()).collect();
        let mut tmp = vec![0.0f32; w * h];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let l = root[y * w + x.saturating_sub(1)];
                let r = root[y * w + (x + 1).min(w - 1)];
                *o = 0.25 * l + 0.5 * root[y * w + x] + 0.25 * r;
            }
        });
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let u = tmp[y.saturating_sub(1) * w + x];
                let d = tmp[(y + 1).min(h - 1) * w + x];
                o[c] = 0.25 * u + 0.5 * tmp[y * w + x] + 0.25 * d;
            }
        });
    }
    out
}

/// Mean, variance and cover at a pixel of a spread.
#[inline]
fn moments(sp: &super::spread::Spread, i: usize) -> Option<(f64, f64, f64)> {
    let c = f64::from(sp.cover[i]);
    if c <= 1e-9 {
        return None;
    }
    Some((f64::from(sp.mean[i]), f64::from(sp.var[i]), c))
}

/// Runs the detector on a lens plane.
pub(super) fn detect(
    plane: &ImageRef,
    focus: Option<(f64, f64)>,
    cancel: &dyn Fn() -> bool,
) -> Result<Detected, LensError> {
    let wp = work_plane(plane);
    let (w, h) = (wp.w, wp.h);
    let n = w * h;
    let hd = 0.5 * ((w * w + h * h) as f64).sqrt();
    let r_min_px = (B_MIN * hd).max(RADIUS_FLOOR_PX);
    let r_max_px = B_CAP * hd * 1.3;
    if w < 16 || h < 16 {
        return Ok(Detected {
            w,
            h,
            b: vec![0.0; n],
            conf: vec![0.0; n],
            balls: Vec::new(),
            edges: 0,
            noise_v: 0.0015,
            hd,
            reduce: wp.reduce,
        });
    }
    let v: Vec<f32> = wp.y.iter().map(|&y| y.max(0.0).sqrt()).collect();
    let noise_v = noise_sigma(&v, w, h);
    drop(v);
    let ladder = build_ladder(&wp.y, w, h, cancel)?;
    if cancel() {
        return Err(LensError::Cancelled);
    }

    // 1. Edge readings.
    let (mut samples, mut sharp) =
        edge_samples(&ladder, w, h, noise_v, r_min_px, B_CAP * hd, cancel)?;
    drop(ladder);
    let measured_edges = samples.iter().filter(|s| s.w >= 0.1).count();

    // 2. The subject anchor.
    let mut anchor = vec![false; n];
    if let Some((fx, fy)) = focus {
        let (cx, cy) = (fx * w as f64, fy * h as f64);
        let radius = (ANCHOR_RADIUS * hd).max(6.0);
        let mut inside = vec![false; n];
        let (x0, x1) = (
            (cx - radius).floor().max(0.0) as usize,
            ((cx + radius).ceil() as usize + 1).min(w),
        );
        let (y0, y1) = (
            (cy - radius).floor().max(0.0) as usize,
            ((cy + radius).ceil() as usize + 1).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                if dx * dx + dy * dy <= radius * radius {
                    inside[y * w + x] = true;
                    anchor[y * w + x] = true;
                }
            }
        }
        samples.retain(|s| !inside[s.idx as usize]);
        let ln_sharp = libm::log(0.4) as f32;
        for (i, _) in inside.iter().enumerate().filter(|(_, b)| **b) {
            samples.push(Sample {
                idx: i as u32,
                w: ANCHOR_WEIGHT as f32,
                ln_r: ln_sharp,
            });
        }
    }

    // 3. Bright balls.
    let (ev_w, ev_s, _) = accumulate(&samples, n);
    let balls = find_balls(
        &BallInput {
            w,
            h,
            rgb: [&wp.rgb[0], &wp.rgb[1], &wp.rgb[2]],
            y: &wp.y,
            noise_v,
            r_max_px,
            ev_w: &ev_w,
            ev_s: &ev_s,
            anchor: &anchor,
            sharp: &sharp,
        },
        cancel,
    )?;
    // Their rims are not edges of the blur: drop the readings around them and put their radius there.
    if !balls.is_empty() {
        let mut inside = vec![false; n];
        for b in &balls {
            let reach = 1.5 * b.r_out;
            let (x0, x1) = (
                (b.cx - reach).floor().max(0.0) as usize,
                ((b.cx + reach).ceil() as usize + 1).min(w),
            );
            let (y0, y1) = (
                (b.cy - reach).floor().max(0.0) as usize,
                ((b.cy + reach).ceil() as usize + 1).min(h),
            );
            for y in y0..y1 {
                for x in x0..x1 {
                    let (dx, dy) = (x as f64 + 0.5 - b.cx, y as f64 + 0.5 - b.cy);
                    if dx * dx + dy * dy <= reach * reach {
                        inside[y * w + x] = true;
                    }
                }
            }
        }
        samples.retain(|s| !inside[s.idx as usize]);
        for (i, f) in inside.iter().enumerate() {
            if *f {
                sharp[i] = false;
            }
        }
        for b in &balls {
            let reach = 1.6 * b.r;
            let per_pixel = BALL_WEIGHT / (std::f64::consts::PI * reach * reach);
            let ln_r = libm::log(b.r) as f32;
            let (x0, x1) = (
                (b.cx - reach).floor().max(0.0) as usize,
                ((b.cx + reach).ceil() as usize + 1).min(w),
            );
            let (y0, y1) = (
                (b.cy - reach).floor().max(0.0) as usize,
                ((b.cy + reach).ceil() as usize + 1).min(h),
            );
            for y in y0..y1 {
                for x in x0..x1 {
                    let (dx, dy) = (x as f64 + 0.5 - b.cx, y as f64 + 0.5 - b.cy);
                    if dx * dx + dy * dy <= reach * reach {
                        samples.push(Sample {
                            idx: (y * w + x) as u32,
                            w: per_pixel as f32,
                            ln_r,
                        });
                    }
                }
            }
        }
    }
    let ball_perimeters: f64 = balls.iter().map(|b| 2.0 * std::f64::consts::PI * b.r).sum();
    if cancel() {
        return Err(LensError::Cancelled);
    }

    // 4. Spread.
    let sigma_g = (3.0 * noise_v).max(GUIDE_TOLERANCE);
    let guide = GuidePyramid::new(guide_image(&wp.rgb, w, h), w, h, sigma_g);
    let (w0, s0, q0) = accumulate(&samples, n);
    let spread_out = spread(&guide, &w0, &s0, &q0, EVIDENCE_NEED);
    if cancel() {
        return Err(LensError::Cancelled);
    }

    // 5. Caution, and the maps.
    let mut b = vec![0.0f32; n];
    let mut conf = vec![0.0f32; n];
    let mut r_est = vec![0.0f64; n];
    for i in 0..n {
        if let Some((mu, var, cover)) = moments(&spread_out, i) {
            let c = cover / (1.0 + var / VARIANCE_HALF);
            conf[i] = c as f32;
            r_est[i] = libm::exp(mu);
        }
    }
    // Places next to a strong sharp edge are not reshaped.
    let mut forced = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            if !sharp[y * w + x] {
                continue;
            }
            for dy in -SHARP_DILATION..=SHARP_DILATION {
                for dx in -SHARP_DILATION..=SHARP_DILATION {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                        forced[ny as usize * w + nx as usize] = true;
                    }
                }
            }
        }
    }
    // The balls' radius over their footprint, except where the map says confidently sharp.
    let mut best = vec![f64::INFINITY; n];
    for ball in &balls {
        let reach = BALL_FOOTPRINT * ball.r;
        let (x0, x1) = (
            (ball.cx - reach).floor().max(0.0) as usize,
            ((ball.cx + reach).ceil() as usize + 1).min(w),
        );
        let (y0, y1) = (
            (ball.cy - reach).floor().max(0.0) as usize,
            ((ball.cy + reach).ceil() as usize + 1).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * w + x;
                let (dx, dy) = (x as f64 + 0.5 - ball.cx, y as f64 + 0.5 - ball.cy);
                let d = (dx * dx + dy * dy).sqrt();
                if d > reach || anchor[i] {
                    continue;
                }
                let in_disc = d <= ball.r_out;
                let sharp_here = forced[i] || (f64::from(conf[i]) >= 0.5 && r_est[i] < r_min_px);
                if !in_disc && sharp_here {
                    continue;
                }
                let nd = d / ball.r;
                if nd < best[i] {
                    best[i] = nd;
                    r_est[i] = ball.r;
                    conf[i] = 0.95;
                    forced[i] = false;
                }
            }
        }
    }
    for i in 0..n {
        if anchor[i] {
            conf[i] = 1.0;
            continue;
        }
        let ok = f64::from(conf[i]) >= CONFIDENCE_MIN && r_est[i] >= r_min_px && !forced[i];
        if ok {
            b[i] = (r_est[i] / hd) as f32;
        } else if forced[i] {
            conf[i] = conf[i].max(0.5);
        }
    }
    // A blurred region smaller than a couple of its own blur discs is a speck of misread edges, not defocus.
    let keep: Vec<bool> = best.iter().map(|d| d.is_finite()).collect();
    drop_small_regions(&mut b, &keep, w, h, hd);
    Ok(Detected {
        w,
        h,
        b,
        conf,
        balls,
        edges: measured_edges + ball_perimeters as usize,
        noise_v,
        hd,
        reduce: wp.reduce,
    })
}

/// Zeroes the connected regions (4-connected) of `b > 0` whose area is below `SMALL_REGION` discs of their own
/// mean radius, except those that touch a pixel of `keep` (a ball's footprint).
fn drop_small_regions(b: &mut [f32], keep: &[bool], w: usize, h: usize, hd: f64) {
    let n = w * h;
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    for start in 0..n {
        if seen[start] || b[start] <= 0.0 {
            continue;
        }
        members.clear();
        stack.push(start);
        seen[start] = true;
        let (mut sum, mut kept) = (0.0f64, false);
        while let Some(i) = stack.pop() {
            members.push(i);
            sum += f64::from(b[i]);
            kept |= keep[i];
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if !seen[j] && b[j] > 0.0 {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        let mean_r = sum / members.len() as f64 * hd;
        if !kept && (members.len() as f64) < SMALL_REGION * std::f64::consts::PI * mean_r * mean_r {
            for &i in &members {
                b[i] = 0.0;
            }
        }
    }
}

/// Fraction of the frame at `b >= 1.5 b_min` with confidence (the "little out-of-focus background" test).
pub(super) fn blurred_fraction(d: &Detected) -> f64 {
    let thr = (crate::consts::MIN_BLUR_FROM * B_MIN) as f32;
    let count =
        d.b.iter()
            .zip(&d.conf)
            .filter(|(b, c)| **b >= thr && **c >= 0.5)
            .count();
    count as f64 / (d.w * d.h).max(1) as f64
}

/// True when the detection has enough to reshape: enough edges and enough of the frame out of focus.
pub(super) fn enough(d: &Detected) -> bool {
    d.edges >= MIN_EDGE_SAMPLES && blurred_fraction(d) >= crate::consts::MIN_BLUR_AREA
}
