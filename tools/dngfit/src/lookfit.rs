//! The look table as a least-squares fit to scattered targets.
//!
//! The renderer's table stage turns a colour with HSV `(h, s, v)` into `(h + dh, s * ds, v * dv)`, with `(dh, ds, dv)`
//! interpolated trilinearly from the grid nodes (value scale in the encoded domain). What the table has to produce at a
//! colour does not depend on the table: it is the colour that the tone curve then maps to the cine rendering's output
//! ([`crate::model::before_tone`]). So each test colour `p` gives a *required modifier* `(dh, ds, dv)` at its position
//! `(h, s, encoded v)`, and the table is the trilinear field that fits those requirements best: a *linear* least-squares
//! problem in the entries, which this module solves by conjugate gradients with a weight that follows how much a
//! modifier error moves the colour, plus a pull towards the node-exact solution so that nodes with no colour near them
//! (outside the scene data's gamut) stay sensible.
//!
//! Fitting the nodes alone (a colour on every node) is exact at the nodes and leaves the error of the interpolation
//! between them, which is large in the first value division, where the cine rendering's toe bends fast.

use rayon::prelude::*;
use ym_color::consts::linalg;
use ym_color::encode::{srgb_decode, srgb_encode};

use crate::model::{Matrices, ToneTable, before_tone, hsv_to_rgb, rgb_to_hsv};
use crate::target::cine_srgb;

/// One required modifier at a position of the table's grid.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Position in grid units: hue 0 to `H`, saturation 0 to `S - 1`, encoded value 0 to `V - 1`.
    pub pos: [f64; 3],
    /// The required `(hue shift in degrees, saturation scale, value scale)`.
    pub target: [f64; 3],
    /// Weight of each of the three components.
    pub weight: [f64; 3],
}

/// The grid cell and weights of `pos`: eight `(node index, weight)` pairs (value-major, hue, saturation order).
#[must_use]
pub fn corners(dims: [usize; 3], pos: [f64; 3]) -> [(usize, f64); 8] {
    let [hd, sd, vd] = dims;
    let h = pos[0].clamp(0.0, hd as f64 - 1e-9);
    let s = pos[1].clamp(0.0, (sd - 1) as f64);
    let v = pos[2].clamp(0.0, (vd - 1) as f64);
    let h0 = (h as usize).min(hd - 1);
    let s0 = (s as usize).min(sd.saturating_sub(2));
    let v0 = if vd < 2 { 0 } else { (v as usize).min(vd - 2) };
    let h1 = if h0 + 1 >= hd { 0 } else { h0 + 1 };
    let (fh, fs) = (h - h0 as f64, s - s0 as f64);
    let fv = if vd < 2 { 0.0 } else { v - v0 as f64 };
    let v1 = if vd < 2 { 0 } else { v0 + 1 };
    let mut out = [(0usize, 0.0f64); 8];
    let mut n = 0;
    for (vi, wv) in [(v0, 1.0 - fv), (v1, fv)] {
        for (hi, wh) in [(h0, 1.0 - fh), (h1, fh)] {
            for (si, ws) in [(s0, 1.0 - fs), (s0 + 1, fs)] {
                out[n] = ((vi * hd + hi) * sd + si, wv * wh * ws);
                n += 1;
            }
        }
    }
    out
}

/// The required modifiers on a regular grid of colours in HSV, `density` times as fine as the table's grid in each
/// direction (hue and saturation) and in the encoded value, left out where the colour needs a negative AWG4 component
/// (the scene data has no such colour). `gain` is the renderer's gain `2^offset`.
#[must_use]
pub fn grid_samples(dims: [u32; 3], gain: f64, tone: &ToneTable, density: usize) -> Vec<Sample> {
    let m = Matrices::new();
    let [hd, sd, vd] = dims.map(|d| d as usize);
    let (nh, ns, nv) = (hd * density, (sd - 1) * density + 1, (vd - 1) * density + 1);
    let cells: Vec<(usize, usize, usize)> = (0..nv)
        .flat_map(|v| (0..nh).flat_map(move |h| (1..ns).map(move |s| (v, h, s))))
        .collect();
    cells
        .par_iter()
        .filter_map(|&(vi, hi, si)| {
            // jitter-free grid, offset by half a cell in hue so that no sample sits exactly on the wrap
            let h_pos = (hi as f64 + 0.5) / density as f64;
            let s_pos = si as f64 / density as f64;
            let v_pos = vi as f64 / density as f64;
            let h = 6.0 * h_pos / hd as f64;
            let s = s_pos / (sd - 1) as f64;
            let ve = v_pos / (vd - 1) as f64;
            let v = srgb_decode(ve);
            if v <= 0.0 || s <= 0.0 {
                return None;
            }
            let p = hsv_to_rgb([h, s, v]);
            let x = linalg::mul_vec(&m.prophoto_to_awg4, p).map(|c| c / gain);
            if x.iter().any(|&c| c < 0.0) {
                return None;
            }
            let target = cine_srgb(x);
            let q = before_tone(target, tone, &m);
            let [h2, s2, v2] = rgb_to_hsv(q);
            let dh_sixths = {
                let mut d = (h2 - h) % 6.0;
                if d > 3.0 {
                    d -= 6.0;
                } else if d <= -3.0 {
                    d += 6.0;
                }
                d
            };
            let dh = if s2 < 1e-9 { 0.0 } else { dh_sixths * 60.0 };
            let ds = s2 / s;
            let dv = if v2 <= 0.0 {
                0.0
            } else {
                srgb_encode(v2.min(1.0)) / ve
            };
            // How far a unit error of each modifier moves the colour (in encoded display units, up to a constant).
            let w_h = (s * ve * std::f64::consts::PI / 180.0).powi(2);
            let w_s = (ve * s).powi(2);
            let w_v = ve.powi(2);
            Some(Sample {
                pos: [h_pos, s_pos, v_pos],
                target: [dh, ds, dv],
                weight: [w_h.max(1e-12), w_s.max(1e-12), w_v.max(1e-12)],
            })
        })
        .collect()
}

/// The trilinear field `Σ w_k e_k` of component `c` at every sample, and its transpose.
struct Operator<'a> {
    dims: [usize; 3],
    corners: Vec<[(usize, f64); 8]>,
    samples: &'a [Sample],
}

impl Operator<'_> {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.corners
            .par_iter()
            .map(|c| c.iter().map(|&(k, w)| w * x[k]).sum())
            .collect()
    }

    fn apply_t(&self, r: &[f64]) -> Vec<f64> {
        let n = self.dims[0] * self.dims[1] * self.dims[2];
        // A fixed chunk size: the order of the additions, and so the result, does not depend on the thread count.
        let chunk = 8192;
        let parts: Vec<Vec<f64>> = self
            .corners
            .par_chunks(chunk)
            .zip(r.par_chunks(chunk))
            .map(|(cs, rs)| {
                let mut acc = vec![0.0; n];
                for (c, &ri) in cs.iter().zip(rs) {
                    for &(k, w) in c {
                        acc[k] += w * ri;
                    }
                }
                acc
            })
            .collect();
        let mut out = vec![0.0; n];
        for p in parts {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out
    }
}

/// What the least-squares solve did.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct SolveReport {
    /// Samples used.
    pub samples: usize,
    /// Conjugate-gradient iterations per component.
    pub iterations: usize,
    /// Weighted RMS of the residual per component before and after, as the weights see it.
    pub rms_before: [f64; 3],
    /// After.
    pub rms_after: [f64; 3],
}

/// Fits the entries (`dh`, `ds`, `dv` per node, value-major then hue then saturation) to `samples`.
///
/// `start` holds the node-exact solution; `fixed` marks entries that must keep their value in a component (the value
/// scale at zero saturation is 1 by the DNG specification); `pull` is the weight of the pull towards `start`, relative
/// to the largest diagonal of the normal equations.
#[must_use]
pub fn solve(
    dims: [u32; 3],
    samples: &[Sample],
    start: &[[f64; 3]],
    fixed: &[[bool; 3]],
    pull: f64,
    iterations: usize,
) -> (Vec<[f64; 3]>, SolveReport) {
    let d = dims.map(|x| x as usize);
    let n = d[0] * d[1] * d[2];
    assert_eq!(start.len(), n);
    let op = Operator {
        dims: d,
        corners: samples.iter().map(|s| corners(d, s.pos)).collect(),
        samples,
    };
    let mut result = start.to_vec();
    let mut before = [0.0; 3];
    let mut after = [0.0; 3];
    for c in 0..3 {
        let w: Vec<f64> = op.samples.iter().map(|s| s.weight[c]).collect();
        let t: Vec<f64> = op.samples.iter().map(|s| s.target[c]).collect();
        let x0: Vec<f64> = start.iter().map(|e| e[c]).collect();
        let free: Vec<bool> = fixed.iter().map(|f| !f[c]).collect();
        // diagonal of AᵀWA, for the preconditioner and the size of the pull
        let mut diag = vec![0.0; n];
        for (cs, &wi) in op.corners.iter().zip(&w) {
            for &(k, a) in cs {
                diag[k] += wi * a * a;
            }
        }
        let lambda = pull * diag.iter().copied().fold(0.0, f64::max);
        let rms = |x: &[f64]| -> f64 {
            let ax = op.apply(x);
            let (num, den) = ax
                .iter()
                .zip(&t)
                .zip(&w)
                .fold((0.0, 0.0), |(n, d), ((a, t), w)| {
                    (n + w * (a - t).powi(2), d + w)
                });
            (num / den.max(1e-300)).sqrt()
        };
        before[c] = rms(&x0);
        // (AᵀWA + λI) x = AᵀW t + λ x0, on the free entries; the fixed ones keep x0
        let normal = |v: &[f64]| -> Vec<f64> {
            let av = op.apply(v);
            let wav: Vec<f64> = av.iter().zip(&w).map(|(a, w)| a * w).collect();
            let mut out = op.apply_t(&wav);
            for k in 0..n {
                out[k] += lambda * v[k];
                if !free[k] {
                    out[k] = 0.0;
                }
            }
            out
        };
        let wt: Vec<f64> = t.iter().zip(&w).map(|(t, w)| t * w).collect();
        let mut rhs = op.apply_t(&wt);
        // contribution of the fixed entries moves to the right-hand side
        let mut fixed_part = vec![0.0; n];
        for k in 0..n {
            if !free[k] {
                fixed_part[k] = x0[k];
            }
        }
        if fixed_part.iter().any(|&v| v != 0.0) {
            let ap = normal_full(&op, &w, &fixed_part, lambda);
            for k in 0..n {
                rhs[k] -= ap[k];
            }
        }
        for k in 0..n {
            rhs[k] += lambda * x0[k];
            if !free[k] {
                rhs[k] = 0.0;
            }
        }
        // preconditioned conjugate gradients on the free entries, starting from x0 (free part)
        let precond = |k: usize| 1.0 / (diag[k] + lambda).max(1e-300);
        let mut xk: Vec<f64> = (0..n).map(|k| if free[k] { x0[k] } else { 0.0 }).collect();
        let ax = normal(&xk);
        let mut r: Vec<f64> = (0..n).map(|k| rhs[k] - ax[k]).collect();
        let mut z: Vec<f64> = (0..n).map(|k| r[k] * precond(k)).collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        for _ in 0..iterations {
            let ap = normal(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap <= 0.0 || rz <= 0.0 {
                break;
            }
            let alpha = rz / pap;
            for k in 0..n {
                xk[k] += alpha * p[k];
                r[k] -= alpha * ap[k];
            }
            for k in 0..n {
                z[k] = r[k] * precond(k);
            }
            let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..n {
                p[k] = z[k] + beta * p[k];
            }
        }
        let full: Vec<f64> = (0..n)
            .map(|k| if free[k] { xk[k] } else { x0[k] })
            .collect();
        after[c] = rms(&full);
        for k in 0..n {
            result[k][c] = full[k];
        }
    }
    (
        result,
        SolveReport {
            samples: samples.len(),
            iterations,
            rms_before: before,
            rms_after: after,
        },
    )
}

/// `AᵀWA x + λ x` without masking (used for the fixed part of the right-hand side).
fn normal_full(op: &Operator<'_>, w: &[f64], x: &[f64], lambda: f64) -> Vec<f64> {
    let ax = op.apply(x);
    let wax: Vec<f64> = ax.iter().zip(w).map(|(a, w)| a * w).collect();
    let mut out = op.apply_t(&wax);
    for (o, v) in out.iter_mut().zip(x) {
        *o += lambda * v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_have_weights_that_sum_to_one_and_hit_nodes_exactly() {
        let dims = [6, 4, 3];
        for pos in [
            [0.0, 0.0, 0.0],
            [2.5, 1.5, 0.5],
            [5.9, 3.0, 2.0],
            [3.0, 2.0, 1.0],
        ] {
            let c = corners(dims, pos);
            let sum: f64 = c.iter().map(|&(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-12, "{pos:?}");
            assert!(c.iter().all(|&(k, _)| k < 6 * 4 * 3));
        }
        // a node position gives that node weight 1
        let c = corners(dims, [3.0, 2.0, 1.0]);
        let node = (6 + 3) * 4 + 2;
        let w: f64 = c.iter().filter(|&&(k, _)| k == node).map(|&(_, w)| w).sum();
        assert!((w - 1.0).abs() < 1e-12);
        // hue wraps: just under H interpolates between the last hue node and node 0
        let c = corners(dims, [5.5, 0.0, 0.0]);
        let to_zero: f64 = c.iter().filter(|&&(k, _)| k == 0).map(|&(_, w)| w).sum();
        assert!((to_zero - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_linear_field_is_recovered_from_scattered_samples() {
        // field e(h, s, v) = 2 + 0.5 s + 0.25 v on a 4 x 5 x 3 grid (no hue dependence, so the wrap is harmless)
        let dims = [4u32, 5, 3];
        let field = |s: f64, v: f64| 2.0 + 0.5 * s + 0.25 * v;
        let mut samples = Vec::new();
        for i in 0..400 {
            let t = i as f64;
            let pos = [
                (t * 0.37) % 4.0,
                4.0 * ((t * 0.173) % 1.0),
                2.0 * ((t * 0.291) % 1.0),
            ];
            samples.push(Sample {
                pos,
                target: [field(pos[1], pos[2]), 1.0, 1.0],
                weight: [1.0, 1.0, 1.0],
            });
        }
        let n = 4 * 5 * 3;
        let start = vec![[0.0, 1.0, 1.0]; n];
        let fixed = vec![[false; 3]; n];
        let (sol, report) = solve(dims, &samples, &start, &fixed, 1e-9, 200);
        assert!(report.rms_after[0] < 1e-6, "{report:?}");
        for v in 0..3usize {
            for h in 0..4usize {
                for s in 0..5usize {
                    let e = sol[(v * 4 + h) * 5 + s][0];
                    assert!((e - field(s as f64, v as f64)).abs() < 1e-4, "{e}");
                }
            }
        }
    }

    #[test]
    fn fixed_entries_keep_their_value_and_the_rest_adapts() {
        let dims = [3u32, 3, 2];
        let n = 18;
        let mut samples = Vec::new();
        for i in 0..300 {
            let t = i as f64;
            samples.push(Sample {
                pos: [
                    (t * 0.31) % 3.0,
                    2.0 * ((t * 0.137) % 1.0),
                    (t * 0.211) % 1.0,
                ],
                target: [0.0, 1.0, 1.0 + 0.2 * 2.0 * ((t * 0.137) % 1.0)],
                weight: [1.0, 1.0, 1.0],
            });
        }
        let start = vec![[0.0, 1.0, 1.0]; n];
        // the value scale at saturation 0 is fixed at 1 (every node whose index % 3 == 0)
        let fixed: Vec<[bool; 3]> = (0..n).map(|k| [false, false, k % 3 == 0]).collect();
        let (sol, _) = solve(dims, &samples, &start, &fixed, 1e-9, 200);
        for (k, e) in sol.iter().enumerate() {
            if k % 3 == 0 {
                assert_eq!(e[2], 1.0);
            }
        }
        // saturation 2 nodes carry the 0.4 slope: dv = 1.4 there
        assert!((sol[2][2] - 1.4).abs() < 1e-3, "{}", sol[2][2]);
    }
}
