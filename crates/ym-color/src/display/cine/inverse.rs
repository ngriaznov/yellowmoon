//! The inverse of the cine rendering (`docs/02`, C12, *Inverse*): display-linear RGB back to scene AWG4.
//!
//! The forward map is not invertible in closed form (level, hue and chroma are coupled), so the inverse is
//! numerical, in four parts: the display side in closed form; the core (steps 4 to 6) by a first-use table
//! and a damped chord iteration with an exact Jacobian; the scene side in closed form; and the
//! nearest-reachable rule for display colours the rendering cannot produce.

use std::f64::consts::PI;
use std::sync::OnceLock;

use super::real::Dual3;
use super::{
    CineRendering, D_MIN, DELTA, EPS_FLOOR, Y_MIN, Y2_MIN, consts, core, derived, luminance_floor,
    tone,
};
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::gamut::LinearAwg4;

/// The residual of `(ln Y2, ca2, cb2)` below which the core has been inverted (the reference uses 1e-9).
const RESIDUAL_TOL: f64 = 1e-7;
/// The largest chroma after the limit that the scene side can undo: `chi_k + (chi_m - chi_k) tanh 3`.
fn chroma_cap() -> f64 {
    consts::CHI_K + (consts::CHI_M - consts::CHI_K) * 3.0_f64.tanh()
}

/// What [`CineRendering::inverse`] returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CineInverse {
    /// The scene colour. One pre-image among possibly several; compare display colours, not scene ones.
    pub scene: LinearAwg4,
    /// The share of the requested chroma that was kept: 1 for a display colour the rendering produces,
    /// less for one it cannot (the nearest reachable colour keeps the requested luminance and hue and
    /// lowers the chroma), 0 for the neutral colour of that luminance.
    pub chroma_fraction: f64,
}

// ---- the scene side, closed form ------------------------------------------------------------------------

/// The scene luminance `Ys` with `tone(Ys) = t` (0 for `t <= 0`); the flat top starts at `tone_inverse(1)`.
pub(super) fn tone_inverse(t: f64) -> f64 {
    let d = derived();
    if t <= 0.0 {
        return 0.0;
    }
    let t = t.min(1.0);
    let z0 = 1.0 - consts::WC;
    let z = if t <= z0 {
        t
    } else {
        z0 + 2.0 * consts::WC * (1.0 - (1.0 - (t - z0) / consts::WC).max(0.0).sqrt())
    };
    let h = (z / consts::KAP).min(1.0 - 1e-16);
    let q = h.powf(1.0 / d.b);
    (d.c * q / (1.0 - q)).powf(1.0 / d.a)
}

/// The inverse of the smooth cube root, by Newton on `r^3 - P r^2 - P d^2 = 0` (`P = |q|^3 / (1 + d^2)`)
/// from the upper bound `P + (P d^2)^(1/3)`, where the cubic is convex and increasing, so the iteration
/// is monotone.
fn smooth_cbrt_inverse(q: f64) -> f64 {
    let d2 = DELTA * DELTA;
    let p = q.abs().powi(3) / (1.0 + d2);
    let mut r = p + (p * d2).cbrt();
    for _ in 0..40 {
        let h = r * r * (r - p) - p * d2;
        let dh = r * (3.0 * r - 2.0 * p);
        if dh.is_nan() || dh <= 0.0 {
            break;
        }
        let step = h / dh;
        r -= step;
        if step.abs() <= 1e-15 * r {
            break;
        }
    }
    r.copysign(q)
}

/// Scene AWG4 from the limited polar coordinates `(ln Ys, ca, cb)`, or `None` when they have no scene
/// colour: the chroma is beyond the part of the limiter that can be undone, or the luminance floor has no
/// root.
fn scene_of(q: Vec3) -> Option<Vec3> {
    let d = derived();
    let (ca, cb) = (q[1], q[2]);
    let limited = ca.hypot(cb);
    if limited.is_nan() || limited > chroma_cap() || q[0].is_nan() || q[0] <= Y_MIN.ln() - 1e-9 {
        return None;
    }
    let (mut ca, mut cb) = (ca, cb);
    if limited > consts::CHI_K {
        let t = ((limited - consts::CHI_K) / (consts::CHI_M - consts::CHI_K)).min(1.0 - 1e-12);
        let chi = consts::CHI_K + (consts::CHI_M - consts::CHI_K) * t.atanh();
        ca *= chi / limited;
        cb *= chi / limited;
    }
    let ym = q[0].exp();
    let x = d.xn * ym * smooth_cbrt_inverse(1.0 + ca);
    let z = d.zn * ym * smooth_cbrt_inverse(1.0 - cb);
    // The luminance floor, `Ys = smoothmax(Y, Yf(Y))`, by Newton from `Y = Ys`, an upper bound of the
    // root (the smooth maximum is never below `Y`).
    let mut y = ym;
    for _ in 0..40 {
        let dx = x - d.xn * y;
        let dz = z - d.zn * y;
        let sq = (dx * dx + dz * dz + EPS_FLOOR).sqrt();
        let yf = consts::RHO * sq;
        let dyf = -consts::RHO * (d.xn * dx + d.zn * dz) / sq;
        let w = consts::WFRAC * yf;
        let dw = consts::WFRAC * dyf;
        let rt = ((y - yf) * (y - yf) + w * w).sqrt();
        let g = 0.5 * (y + yf + rt) - ym;
        if g.abs() <= 1e-13 * ym {
            break;
        }
        let dg = 0.5
            * (1.0
                + dyf
                + if rt > 0.0 {
                    ((y - yf) * (1.0 - dyf) + w * dw) / rt
                } else {
                    0.0
                });
        let dg = if dg.abs() < 1e-6 {
            1e-6_f64.copysign(dg)
        } else {
            dg
        };
        let step = (g / dg).clamp(-0.5 * y.abs() - 1e-12, 0.5 * y.abs() + 1e-12);
        y -= step;
    }
    let back = (luminance_floor(x, y, z) - ym).abs();
    if back.is_nan() || back > 1e-9 * ym {
        return None;
    }
    Some(linalg::mul_vec(&d.xyz_to_awg4, [x, y, z]))
}

// ---- the display side, closed form ----------------------------------------------------------------------

/// Undoes the soft ceiling of step 7 on one channel: `v = z / (1 + kh)` with `z = o` for `o <= 1 - kh`.
fn ceiling_inverse(o: f64) -> f64 {
    let kh = derived().kh;
    let z0 = 1.0 - kh;
    let z = if o <= z0 {
        o
    } else {
        z0 + 2.0 * kh * (1.0 - (1.0 - (o - z0) / kh).max(0.0).sqrt())
    };
    z / (1.0 + kh)
}

/// Display RGB to the display XYZ before the gamut stage: each channel is held to `[D_MIN, 1]`, the
/// ceiling is undone, the shared luminance `Y2` is found by Newton on the convex sum of the floors'
/// inverses and the floors are undone.
fn gamut_inverse(display: Vec3, to_xyz: &Mat3) -> Vec3 {
    let kl = derived().kl;
    let yw = to_xyz[1];
    let hold = |c: f64| {
        if c.is_nan() {
            D_MIN
        } else {
            c.clamp(D_MIN, 1.0)
        }
    };
    let v = display.map(|c| ceiling_inverse(hold(c)));
    let y0 = yw[0] * v[0] + yw[1] * v[1] + yw[2] * v[2];
    let mut y2 = y0;
    if v.iter().any(|&c| c < kl * y0) {
        // Newton in t = ln Y2 on the convex, decreasing g(t) = sum y_i Finv(v_i e^-t) - 1, started to the
        // right of the root.
        let mut t = y0.ln();
        for _ in 0..40 {
            let e = (-t).exp();
            let (mut g, mut dg) = (-1.0, 0.0);
            for i in 0..3 {
                let s = v[i] * e;
                if s >= kl {
                    g += yw[i] * s;
                    dg -= yw[i] * s;
                } else {
                    g += yw[i] * kl * (1.0 + (s / kl).ln());
                    dg -= yw[i] * kl;
                }
            }
            let step = g / dg;
            t -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        y2 = t.exp();
    }
    let rgb = v.map(|c| {
        let s = c / y2;
        y2 * if s >= kl {
            s
        } else {
            kl * (1.0 + (s / kl).ln())
        }
    });
    linalg::mul_vec(to_xyz, rgb)
}

// ---- the core: chord iteration ---------------------------------------------------------------------------

fn residual(q: Vec3, goal: Vec3) -> Vec3 {
    let o = core::<f64>(q[0], q[1], q[2]);
    [o[0] - goal[0], o[1] - goal[1], o[2] - goal[2]]
}

/// The largest absolute component; infinite for a non-finite vector.
fn norm_inf(r: Vec3) -> f64 {
    let m = r[0].abs().max(r[1].abs()).max(r[2].abs());
    if m.is_finite() { m } else { f64::INFINITY }
}

/// The inverse of a 3x3 matrix, `None` when it is singular.
fn inverse3(a: &Mat3) -> Option<Mat3> {
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if det.is_nan() || det.abs() <= 1e-300 {
        return None;
    }
    let id = 1.0 / det;
    Some([
        [
            c00 * id,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id,
        ],
        [
            c01 * id,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id,
        ],
        [
            c02 * id,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id,
        ],
    ])
}

/// The exact 3x3 Jacobian of the core at `q`, by forward-mode dual numbers.
fn jacobian(q: Vec3) -> Mat3 {
    let o = core::<Dual3>(
        Dual3::input(q[0], 0),
        Dual3::input(q[1], 1),
        Dual3::input(q[2], 2),
    );
    [o[0].d, o[1].d, o[2].d]
}

/// A damped chord iteration for `core(q) = goal` from `q`. The Jacobian is computed at the start and
/// again only when a step fails to cut the largest residual by a factor of four; a step is limited to 0.7
/// and halved up to six times while the residual does not fall. Returns the last `q` and whether the
/// residual is below [`RESIDUAL_TOL`].
fn newton(mut q: Vec3, goal: Vec3, max_iterations: usize) -> (Vec3, bool) {
    let mut r = residual(q, goal);
    let mut cost = norm_inf(r);
    let mut jinv: Option<Mat3> = None;
    let mut fresh = false;
    for _ in 0..max_iterations {
        if cost < RESIDUAL_TOL {
            return (q, true);
        }
        let ji = match jinv {
            Some(ji) => ji,
            None => {
                let Some(ji) = inverse3(&jacobian(q)) else {
                    return (q, false);
                };
                jinv = Some(ji);
                fresh = true;
                ji
            }
        };
        let step = [
            -(ji[0][0] * r[0] + ji[0][1] * r[1] + ji[0][2] * r[2]),
            -(ji[1][0] * r[0] + ji[1][1] * r[1] + ji[1][2] * r[2]),
            -(ji[2][0] * r[0] + ji[2][1] * r[1] + ji[2][2] * r[2]),
        ];
        let size = step[0].abs().max(step[1].abs()).max(step[2].abs());
        let mut scale = if size > 0.7 { 0.7 / size } else { 1.0 };
        let mut accepted = false;
        for _ in 0..6 {
            let qn = [
                q[0] + scale * step[0],
                q[1] + scale * step[1],
                q[2] + scale * step[2],
            ];
            let rn = residual(qn, goal);
            let cn = norm_inf(rn);
            if cn < cost {
                let good = cn < 0.25 * cost;
                q = qn;
                r = rn;
                cost = cn;
                accepted = true;
                if !good {
                    jinv = None;
                }
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            if fresh {
                return (q, cost < RESIDUAL_TOL);
            }
            jinv = None;
        }
        fresh = false;
    }
    (q, cost < RESIDUAL_TOL)
}

/// A damped least-squares (Levenberg-Marquardt) iteration for `core(q) = goal` with the exact Jacobian.
/// Slower and more robust than [`newton`]: it follows the descent of the squared residual and so reaches
/// pre-images far from the start (a very bright, strongly coloured scene colour for a display colour that
/// the table cannot place). It is the last resort before the nearest-reachable rule, as the reference's.
fn levenberg(mut q: Vec3, goal: Vec3, max_iterations: usize) -> (Vec3, bool) {
    let finite = |r: Vec3| r.map(|v| if v.is_finite() { v } else { 1e3 });
    let cost_of = |r: Vec3| r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let mut r = finite(residual(q, goal));
    let mut cost = cost_of(r);
    let mut damping = 1e-6;
    for _ in 0..max_iterations {
        if norm_inf(r) < RESIDUAL_TOL * 1e-2 {
            break;
        }
        let j = jacobian(q).map(|row| row.map(|v| if v.is_finite() { v } else { 0.0 }));
        // JtJ and Jt r.
        let mut a = [[0.0; 3]; 3];
        let mut g = [0.0; 3];
        for (row, &res) in j.iter().zip(&r) {
            for p in 0..3 {
                g[p] += row[p] * res;
                for k in 0..3 {
                    a[p][k] += row[p] * row[k];
                }
            }
        }
        let scale = (a[0][0] + a[1][1] + a[2][2]).max(1e-30);
        for (p, row) in a.iter_mut().enumerate() {
            row[p] += damping * scale;
        }
        let Some(inv) = inverse3(&a) else {
            damping = (damping * 10.0).min(1e6);
            continue;
        };
        let step = linalg::mul_vec(&inv, g).map(|v| (-v).clamp(-0.7, 0.7));
        let qn = [q[0] + step[0], q[1] + step[1], q[2] + step[2]];
        let rn = finite(residual(qn, goal));
        let cn = cost_of(rn);
        if cn < cost {
            q = qn;
            r = rn;
            cost = cn;
            damping = (damping * 0.3).max(1e-12);
        } else {
            damping = (damping * 10.0).min(1e6);
        }
    }
    (q, norm_inf(r) < RESIDUAL_TOL)
}

/// A start for the iteration without the table: the damped fixed point `Y <- Tinv(Y2 e^-lr)`,
/// `chi <- chi e^-gh / G0`, `theta <- theta2 - dt` (damping 0.7, six steps; the terms at the current
/// estimate).
fn fixed_point_start(y2: f64, ca2: f64, cb2: f64) -> Vec3 {
    let d = derived();
    let chroma = ca2.hypot(cb2);
    let theta2 = cb2.atan2(ca2);
    let mut y = tone_inverse(y2.min(1.0));
    if y.is_nan() || y <= 0.0 {
        y = 1e-6;
    }
    let mut chi = chroma / d.g_peak;
    let mut theta = theta2;
    let damp = 0.7;
    for _ in 0..6 {
        let t = super::terms::<f64>(y.ln(), chi * theta.cos(), chi * theta.sin());
        let yn = tone_inverse((y2 * (-t.lr).exp()).min(1.0));
        let yn = if yn > 0.0 { yn } else { y };
        y = ((1.0 - damp) * y.max(1e-300).ln() + damp * yn.max(1e-300).ln()).exp();
        chi = (1.0 - damp) * chi + damp * chroma * (-t.gh).exp() / t.g0;
        let turn = (theta2 - t.dt - theta + PI).rem_euclid(2.0 * PI) - PI;
        theta += damp * turn;
    }
    [y.max(1e-300).ln(), chi * theta.cos(), chi * theta.sin()]
}

// ---- the table -------------------------------------------------------------------------------------------

/// Nodes along the level axis `s`, the hue angle and the chroma.
const NS: usize = 40;
const NT: usize = 64;
const NR: usize = 28;
/// The largest chroma `rho` the table holds.
const R_MAX: f64 = 2.2;

/// The level axis: the scene luminance of the neutral colour whose display luminance is `y2`.
fn axis_s(y2: f64) -> f64 {
    tone_inverse(y2.clamp(1e-7, 1.0)).ln()
}

/// A start for the iteration, interpolated from 71,680 pre-images of the core.
struct Table {
    s_lo: f64,
    ds: f64,
    dr: f64,
    /// `(ln Ys - s, ca, cb)` of the pre-image of node `(i, j, k)` at `(i * NT + j) * NR + k`.
    data: Vec<[f32; 3]>,
}

impl Table {
    /// Builds the table from the forward model alone: each (level, hue) ray is marched outwards from the
    /// neutral colour, each node solved by the iteration from the one before it; a node with no
    /// pre-image keeps the last good one. Rays are independent, so the result does not depend on
    /// `threads`.
    fn build(threads: usize) -> Table {
        let s_lo = axis_s(1e-7);
        let s_hi = axis_s(1.0);
        let ds = (s_hi - s_lo) / (NS - 1) as f64;
        let dr = R_MAX / (NR - 1) as f64;
        let rays = NS * NT;
        let per_thread = rays.div_ceil(threads.max(1));
        let ray_nodes = |ray: usize| -> [[f32; 3]; NR] {
            let (i, j) = (ray / NT, ray % NT);
            let s = s_lo + i as f64 * ds;
            let y2 = tone(s.exp()).max(1e-12);
            let theta = 2.0 * PI * j as f64 / NT as f64 - PI;
            let mut out = [[0.0f32; 3]; NR];
            let mut previous = [s, 0.0, 0.0];
            for (k, node) in out.iter_mut().enumerate() {
                let rho = k as f64 * dr;
                let goal = [y2.ln(), rho * theta.cos(), rho * theta.sin()];
                let (mut q, mut ok) = newton(previous, goal, 20);
                ok = ok && scene_of(q).is_some();
                if !ok {
                    let (q2, ok2) = newton(fixed_point_start(y2, goal[1], goal[2]), goal, 40);
                    if ok2 && scene_of(q2).is_some() {
                        q = q2;
                        ok = true;
                    }
                }
                if ok {
                    previous = q;
                }
                *node = [
                    (previous[0] - s) as f32,
                    previous[1] as f32,
                    previous[2] as f32,
                ];
            }
            out
        };
        let mut data = vec![[0.0f32; 3]; rays * NR];
        std::thread::scope(|scope| {
            let handles: Vec<_> = data
                .chunks_mut(per_thread * NR)
                .enumerate()
                .map(|(n, chunk)| {
                    let first = n * per_thread;
                    let ray_nodes = &ray_nodes;
                    scope.spawn(move || {
                        for (r, nodes) in chunk.chunks_mut(NR).enumerate() {
                            nodes.copy_from_slice(&ray_nodes(first + r));
                        }
                    })
                })
                .collect();
            for h in handles {
                if let Err(panic) = h.join() {
                    std::panic::resume_unwind(panic);
                }
            }
        });
        Table { s_lo, ds, dr, data }
    }

    /// Trilinear interpolation (hue periodic) of the start `(ln Ys, ca, cb)` for a display colour.
    fn lookup(&self, y2: f64, ca2: f64, cb2: f64) -> Vec3 {
        let s = axis_s(y2);
        let fs = ((s - self.s_lo) / self.ds).clamp(0.0, (NS - 1) as f64 - 1e-9);
        let ft = (cb2.atan2(ca2) + PI) / (2.0 * PI) * NT as f64;
        let fr = (ca2.hypot(cb2) / self.dr).clamp(0.0, (NR - 1) as f64 - 1e-9);
        // Truncation is the floor here: all three coordinates are non-negative after the clamps.
        let i0 = fs as usize;
        let k0 = fr as usize;
        let t_floor = ft.floor();
        let j0 = (t_floor as isize).rem_euclid(NT as isize) as usize;
        let j1 = (j0 + 1) % NT;
        let (ws, wt, wr) = (fs - i0 as f64, ft - t_floor, fr - k0 as f64);
        let mut acc = [0.0f64; 3];
        for (di, a) in [(0, 1.0 - ws), (1, ws)] {
            for (j, b) in [(j0, 1.0 - wt), (j1, wt)] {
                for (dk, c) in [(0, 1.0 - wr), (1, wr)] {
                    let node = self.data[((i0 + di) * NT + j) * NR + k0 + dk];
                    let w = a * b * c;
                    acc[0] += w * f64::from(node[0]);
                    acc[1] += w * f64::from(node[1]);
                    acc[2] += w * f64::from(node[2]);
                }
            }
        }
        [acc[0] + s, acc[1], acc[2]]
    }
}

static TABLE: OnceLock<Table> = OnceLock::new();

/// The table, built on first use by every available core.
fn table() -> &'static Table {
    TABLE.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        Table::build(threads)
    })
}

// ---- nearest reachable -----------------------------------------------------------------------------------

/// The nearest-reachable solve: keep the display luminance `y2` and the hue angle of `(ca2, cb2)` and
/// find the largest chroma fraction `lambda` in `[0, 1]` that has a pre-image. Marches `lambda` in
/// eighths from the neutral colour with the iteration (each step started from the previous solution) and
/// bisects 24 times in the first eighth that fails. `start` is a start for the full target.
fn solve_ray(y2: f64, ca2: f64, cb2: f64, start: Option<Vec3>) -> (Vec3, f64) {
    let ly2 = y2.max(1e-300).ln();
    let neutral = [axis_s(y2), 0.0, 0.0];
    let (q0, ok0) = newton(neutral, [ly2, 0.0, 0.0], 30);
    let mut last = if ok0 { q0 } else { neutral };
    if let Some(qs) = start {
        let (q, ok) = newton(qs, [ly2, ca2, cb2], 12);
        if ok && scene_of(q).is_some() {
            return (q, 1.0);
        }
    }
    let steps = 8;
    let mut lambda_ok = 0.0;
    for k in 1..=steps {
        let lambda = f64::from(k) / f64::from(steps);
        let (q, ok) = newton(last, [ly2, lambda * ca2, lambda * cb2], 12);
        if ok && scene_of(q).is_some() {
            last = q;
            lambda_ok = lambda;
        } else {
            let (mut lo, mut hi) = (lambda_ok, lambda);
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                let (qm, okm) = newton(last, [ly2, mid * ca2, mid * cb2], 12);
                if okm && scene_of(qm).is_some() {
                    lo = mid;
                    last = qm;
                } else {
                    hi = mid;
                }
            }
            return (last, lo);
        }
    }
    (last, 1.0)
}

/// The fast inverse of the cine rendering for one display colour.
pub(super) fn invert(r: &CineRendering, display: Vec3) -> CineInverse {
    let d = derived();
    let xyz2 = gamut_inverse(display, &r.to_xyz);
    let y2 = xyz2[1].max(Y2_MIN);
    let ca2 = (xyz2[0] / (d.xn * y2)).cbrt() - 1.0;
    let cb2 = 1.0 - (xyz2[2] / (d.zn * y2)).cbrt();
    let goal = [y2.ln(), ca2, cb2];
    let q0 = table().lookup(y2, ca2, cb2);
    let reached = |x: Vec3, fraction: f64| CineInverse {
        scene: LinearAwg4(x),
        chroma_fraction: fraction,
    };
    let (q, ok) = newton(q0, goal, 6);
    if ok && let Some(x) = scene_of(q) {
        return reached(x, 1.0);
    }
    // The fallback: the damped fixed point as the start, with more iterations.
    let (q, ok) = newton(fixed_point_start(y2, ca2, cb2), goal, 40);
    if ok && let Some(x) = scene_of(q) {
        return reached(x, 1.0);
    }
    // The last resort for a pre-image far from every start: a damped least-squares iteration from the
    // same fixed-point start.
    let (q, ok) = levenberg(fixed_point_start(y2, ca2, cb2), goal, 60);
    if ok && let Some(x) = scene_of(q) {
        return reached(x, 1.0);
    }
    // Not reachable (or not found): the nearest reachable colour.
    let (q, lambda) = solve_ray(y2, ca2, cb2, Some(q0));
    match scene_of(q) {
        Some(x) => reached(x, lambda),
        None => {
            // The neutral colour of that luminance always has a pre-image.
            let g = tone_inverse(y2.min(1.0));
            reached([g, g, g], 0.0)
        }
    }
}
