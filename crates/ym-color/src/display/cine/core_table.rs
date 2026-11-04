//! The core of the cine rendering's inverse, per pixel (`docs/02` C12 *Inverse*, *Per-pixel return*; `docs/24`
//! section 3.8c, F-04d).
//!
//! The Film node returns a print's display colour to LogC4 for every pixel, by the same rule the direct chain uses
//! ([`CineRendering::inverse_continuous`](super::CineRendering::inverse_continuous)): the pre-image on the sheet of
//! the forward map that is connected to the neutral colour of the same display luminance, and for a colour that
//! sheet does not reach, the pre-image of the nearest colour it does. Four parts, three of them closed form:
//!
//! 1. the display side ([`DisplayCoordinates`]), closed form up to one Newton iteration for the shared luminance;
//! 2. the **core inverse** `(ln Y2, ca2, cb2) -> (ln Ys, ca, cb)`, the only part that needs a table: [`CoreTable`];
//! 3. the nearest-reachable rule, an exact clamp of the chroma to the table's reachable boundary;
//! 4. the scene side ([`scene_from_coordinates`](super::scene_from_coordinates)), closed form.
//!
//! **The table** is indexed by the display colour's level `s = ln Tinv(Y2)`, its hue angle `theta` and its chroma
//! `rho`, and holds the pre-image `(ln Ys - s, ca, cb)`. Along each ray `(s, theta)` the nodes sit at fractions
//! `u` of the ray's reachable length `rho_max(s, theta)` (also a table), so the table covers the reachable set
//! exactly: `u = 1` is the boundary, whose pre-image is the nearest-reachable answer for every colour beyond it. The
//! nodes crowd towards `u = 1` (`u = 1 - (1 - k / (n - 1))^2`) because the pre-image has a square-root fold there.
//! It depends on the rendering's constants alone (not on the output: the output enters in the display side), is
//! built once per process from the forward model, and is the same bytes on any number of threads.
//!
//! **The solve** starts from the trilinear interpolation of the table and takes a fixed number of damped Newton
//! steps with the exact Jacobian (dual numbers through steps 4 to 6): no data-dependent loop count, and a step is
//! kept only when it lowers the residual.

use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use super::inverse::{
    axis_s, display_side, inverse3, jacobian, jump, jump_bound, newton, norm_inf, residual,
    scene_of, tone_inverse,
};
use super::{CineCoordinates, CineRendering, scene_from_coordinates, tone};
use crate::consts::Vec3;
use crate::gamut::LinearAwg4;

/// The largest chroma `rho` the table holds.
const R_MAX: f64 = 4.0;
/// The step in `rho` of the march that finds a ray's reachable length.
const MARCH_STEP: f64 = 0.04;
/// The pre-image movement per step of the march that the step length aims at.
const MARCH_MOVE: f64 = 0.03;
/// The first step of the march.
const MARCH_START: f64 = 0.001;
/// Bisections of the last march step.
const BISECTIONS: usize = 24;
/// The residual of the core below which a colour counts as reached.
const SOLVED: f64 = 1e-8;
/// The residual below which a Newton step is skipped (it would not move).
pub const CONVERGED: f64 = 1e-13;
/// The largest change of any coordinate in one Newton step (a larger step is scaled down to it), and the clamp of a
/// least-squares step.
pub const STEP_LIMIT: f64 = 0.7;
/// The factors of the Newton step that are tried in turn; the first that lowers the residual is kept.
pub const STEP_SCALES: [f64; 3] = [1.0, 0.5, 0.25];
/// The damping `mu` of the least-squares candidates, tried in turn (times a third of the trace of `J^T J`).
pub const LM_DAMPING: [f64; 3] = [1e-3, 1e-1, 10.0];
/// The floor of the trace of `J^T J` in the least-squares damping.
pub const LM_TRACE_FLOOR: f64 = 1e-30;

/// The settings of [`CoreTable`]'s Newton steps as `(name, value)` pairs in f64, for a GPU kernel: `ym-gpu` writes them
/// into the shader text (`FX_<NAME>`), so the kernel and `refine` apply the same numbers. The list also holds the
/// fixed iteration counts the kernel runs where the CPU loops to a tolerance (the CPU runs up to 40), which are
/// chosen against the CPU code (`docs/24` section 3.8c).
#[must_use]
pub fn kernel_refine_scalars() -> Vec<(&'static str, f64)> {
    vec![
        ("CONVERGED", CONVERGED),
        ("STEP_LIMIT", STEP_LIMIT),
        ("STEP_SCALE_0", STEP_SCALES[0]),
        ("STEP_SCALE_1", STEP_SCALES[1]),
        ("STEP_SCALE_2", STEP_SCALES[2]),
        ("LM_MU_0", LM_DAMPING[0]),
        ("LM_MU_1", LM_DAMPING[1]),
        ("LM_MU_2", LM_DAMPING[2]),
        ("LM_TRACE_FLOOR", LM_TRACE_FLOOR),
        ("Y2_ITERATIONS", KERNEL_Y2_ITERATIONS as f64),
        ("CBRT_ITERATIONS", KERNEL_CBRT_ITERATIONS as f64),
        ("FLOOR_ITERATIONS", KERNEL_FLOOR_ITERATIONS as f64),
    ]
}

/// Fixed Newton iterations of the kernel for the shared luminance `Y2` of the display step.
pub const KERNEL_Y2_ITERATIONS: u32 = 10;
/// Fixed Newton iterations of the kernel for the smooth cube root's inverse.
pub const KERNEL_CBRT_ITERATIONS: u32 = 8;
/// Fixed Newton iterations of the kernel for the luminance floor's root.
pub const KERNEL_FLOOR_ITERATIONS: u32 = 10;
/// The most the pre-image may move between two nodes for them to be on the same sheet (the march uses
/// `jump_bound`).
const NODE_JUMP: f64 = 0.15;

/// The size of a [`CoreTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreDims {
    /// Nodes along the level `s`.
    pub ns: usize,
    /// Nodes around the hue angle (periodic).
    pub nt: usize,
    /// Nodes along `u`, the fraction of the ray's reachable length.
    pub nk: usize,
}

impl Default for CoreDims {
    fn default() -> Self {
        Self {
            ns: DEFAULT_NS,
            nt: DEFAULT_NT,
            nk: DEFAULT_NK,
        }
    }
}

/// Default nodes along the level.
pub const DEFAULT_NS: usize = 40;
/// Default nodes around the hue angle.
pub const DEFAULT_NT: usize = 64;
/// Default nodes along the radius.
pub const DEFAULT_NK: usize = 24;
/// Default number of Newton steps after the table's start.
pub const DEFAULT_STEPS: usize = 5;

/// The display colour as the core sees it: the shared luminance and the cube-root chroma coordinates of the
/// display XYZ before the gamut stage (step 1 of the inverse, undone).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayCoordinates {
    /// The display luminance `Y2` (at least `Y2_MIN`).
    pub y2: f64,
    /// The red-green cube-root coordinate.
    pub ca2: f64,
    /// The blue-yellow cube-root coordinate.
    pub cb2: f64,
}

impl CineRendering {
    /// Step 1 of the inverse in closed form: each channel is held to `[D_MIN, 1]` (NaN counts as `D_MIN`), the
    /// soft ceiling is undone, the shared luminance `Y2` is found by Newton on the convex sum of the floors'
    /// inverses and the floors are undone; then `(Y2, ca2, cb2)`.
    #[must_use]
    pub fn display_coordinates(&self, display: Vec3) -> DisplayCoordinates {
        let (y2, ca2, cb2) = display_side(self, display);
        DisplayCoordinates { y2, ca2, cb2 }
    }

    /// The per-pixel return: the same pre-image as [`CineRendering::inverse_coordinates`] (the sheet connected to
    /// the neutral colour; the nearest reachable colour's for one it does not reach), from `table` by a start and
    /// a fixed number of Newton steps, with no continuation. See the module documentation.
    #[must_use]
    pub fn return_coordinates(&self, table: &CoreTable, display: Vec3) -> CineCoordinates {
        table.invert(self.display_coordinates(display))
    }

    /// [`CineRendering::return_coordinates`] followed by the scene side: the scene colour and the share of the
    /// requested chroma that was kept (the Film node's stage 10 before LogC4).
    #[must_use]
    pub fn return_scene(&self, table: &CoreTable, display: Vec3) -> (LinearAwg4, f64) {
        table.scene(self.display_coordinates(display))
    }
}

/// The table of the core's connected-sheet pre-image (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct CoreTable {
    dims: CoreDims,
    steps: usize,
    s_lo: f64,
    /// The span of the level axis, `s_hi - s_lo`.
    s_span: f64,
    /// The natural logarithm of the reachable length `rho_max` of ray `(i, j)`, at `i * nt + j` (at least `ln 1e-9`).
    ln_reach: Vec<f32>,
    /// `(ln Ys - s, ca, cb)` of node `(i, j, k)` at `(i * nt + j) * nk + k`.
    data: Vec<[f32; 3]>,
}

static SHARED: OnceLock<CoreTable> = OnceLock::new();

impl CoreTable {
    /// The process-wide table, built on first use on every available core.
    #[must_use]
    pub fn shared() -> &'static CoreTable {
        SHARED.get_or_init(|| {
            let threads = std::thread::available_parallelism().map_or(1, usize::from);
            CoreTable::build(CoreDims::default(), threads)
        })
    }

    /// Builds a table of `dims` on `threads` threads. The bytes do not depend on `threads`: rays are independent
    /// and each is made by one thread.
    #[must_use]
    pub fn build(dims: CoreDims, threads: usize) -> CoreTable {
        let s_lo = axis_s(1e-7);
        let s_hi = axis_s(1.0);
        let s_span = s_hi - s_lo;
        let rays = dims.ns * dims.nt;
        let mut data = vec![[0.0_f32; 3]; rays * dims.nk];
        let mut ln_reach = vec![0.0_f32; rays];
        let threads = threads.clamp(1, rays);
        {
            let jobs: Vec<(&mut [f32], &mut [[f32; 3]])> = ln_reach
                .chunks_mut(1)
                .zip(data.chunks_mut(dims.nk))
                .collect();
            let queue = Mutex::new(jobs.into_iter().enumerate());
            let work = |ray: usize, r: &mut [f32], nodes: &mut [[f32; 3]]| {
                let (i, j) = (ray / dims.nt, ray % dims.nt);
                let s = level_node(s_lo, s_span, i, dims.ns);
                let theta = 2.0 * PI * j as f64 / dims.nt as f64 - PI;
                r[0] = build_ray(s, theta, nodes).max(1e-9).ln() as f32;
            };
            if threads == 1 {
                let mut q = queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for (ray, (r, nodes)) in q.by_ref() {
                    work(ray, r, nodes);
                }
            } else {
                std::thread::scope(|scope| {
                    for _ in 0..threads {
                        let (queue, work) = (&queue, &work);
                        scope.spawn(move || {
                            loop {
                                let next = queue
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .next();
                                let Some((ray, (r, nodes))) = next else { break };
                                work(ray, r, nodes);
                            }
                        });
                    }
                });
            }
        }
        CoreTable {
            dims,
            steps: DEFAULT_STEPS,
            s_lo,
            s_span,
            ln_reach,
            data,
        }
    }

    /// The same table taking `steps` Newton steps after the start (0: the table alone).
    #[must_use]
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// The table's size.
    #[must_use]
    pub fn dims(&self) -> CoreDims {
        self.dims
    }

    /// The Newton steps taken after the start.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The natural logarithm of the reachable length `rho_max` of every ray, `ns * nt` values, hue fastest: the
    /// boundary of the nearest-reachable rule.
    #[must_use]
    pub fn ln_reach(&self) -> &[f32] {
        &self.ln_reach
    }

    /// The nodes `(ln Ys - s, ca, cb)`, `ns * nt * nk`, radius fastest.
    #[must_use]
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    /// The level `s` of the first row and the span to the last row (the rows crowd towards the last).
    #[must_use]
    pub fn level_axis(&self) -> (f64, f64) {
        (self.s_lo, self.s_span)
    }

    /// The start for a display colour: the pre-image interpolated from the table, the reachable length of the ray
    /// through the colour and the chroma `rho`. Each of the four rays around the colour `(s, theta)` is read at the
    /// colour's own chroma (at the fraction `rho / rho_max` of that ray, held at its end) and the four values are
    /// blended bilinearly: the rays' reachable lengths differ by a factor of two or more in the dark and near
    /// white, and a blend of the nodes at equal fractions would mix pre-images of different chromas. The reachable
    /// length of the colour's own ray is the blend of the logarithms of the four.
    fn start(&self, c: DisplayCoordinates) -> (Vec3, f64, f64) {
        let CoreDims { ns, nt, nk } = self.dims;
        // The level of the neutral colour with this display luminance (unclamped: the start's `ln Ys` follows it),
        // and its clamped position on the axis.
        let s_full = tone_inverse(c.y2.min(1.0)).max(1e-300).ln();
        let t = ((s_full - self.s_lo) / self.s_span).clamp(0.0, 1.0);
        let fs = ((1.0 - (1.0 - t).sqrt()) * (ns - 1) as f64).clamp(0.0, (ns - 1) as f64 - 1e-9);
        let rho = c.ca2.hypot(c.cb2);
        let theta = c.cb2.atan2(c.ca2);
        let ft = (theta + PI) / (2.0 * PI) * nt as f64;
        let t_floor = ft.floor();
        let j0 = (t_floor as isize).rem_euclid(nt as isize) as usize;
        let j1 = (j0 + 1) % nt;
        let (i0, wt, ws) = (fs as usize, ft - t_floor, fs - fs.floor());
        let mut acc = [0.0_f64; 3];
        let mut ln_reach = 0.0;
        for (di, a) in [(0, 1.0 - ws), (1, ws)] {
            for (j, b) in [(j0, 1.0 - wt), (j1, wt)] {
                let ray = (i0 + di) * nt + j;
                let w = a * b;
                let ln_r = f64::from(self.ln_reach[ray]);
                ln_reach += w * ln_r;
                let u = (rho / ln_r.exp()).min(1.0);
                let fk =
                    ((nk - 1) as f64 * (1.0 - (1.0 - u).sqrt())).clamp(0.0, (nk - 1) as f64 - 1e-9);
                let k0 = fk as usize;
                let wk = fk - fk.floor();
                let (n0, n1) = (self.data[ray * nk + k0], self.data[ray * nk + k0 + 1]);
                for m in 0..3 {
                    acc[m] += w * (f64::from(n0[m]) + wk * (f64::from(n1[m]) - f64::from(n0[m])));
                }
            }
        }
        ([acc[0] + s_full, acc[1], acc[2]], ln_reach.exp(), rho)
    }

    /// The reachable length of the ray of constant luminance and hue through `c`, as the table interpolates it:
    /// a chroma `rho2` above it is beyond what the connected sheet reaches.
    #[must_use]
    pub fn reachable_length(&self, c: DisplayCoordinates) -> f64 {
        self.start(c).1
    }

    /// The core inverse and the nearest-reachable rule for a display colour (see the module documentation).
    #[must_use]
    pub fn invert(&self, c: DisplayCoordinates) -> CineCoordinates {
        self.solve(c).0
    }

    /// [`CoreTable::invert`] and, when the pre-image has a scene colour, that colour (the scene side is then not
    /// done twice).
    fn solve(&self, c: DisplayCoordinates) -> (CineCoordinates, Option<Vec3>) {
        let (q0, rho_max, rho) = self.start(c);
        if rho > rho_max {
            // Beyond the reachable boundary of the ray: the pre-image of its end.
            let found = CineCoordinates {
                coordinates: q0,
                chroma_fraction: rho_max / rho,
            };
            return (found, scene_of(q0));
        }
        let goal = [c.y2.max(1e-300).ln(), c.ca2, c.cb2];
        let q = refine(q0, goal, self.steps);
        // A colour at the very edge of the reachable set (within the table's accuracy of `rho_max`) may have no
        // pre-image on the connected sheet (or none that has a scene colour) while the table's length says it has:
        // the iteration then ends at its least residual, which is the nearest reachable colour's pre-image to the
        // table's accuracy, and the colour is reported as slightly short of reached.
        let scene = scene_of(q);
        let converged = norm_inf(residual(q, goal)) <= SOLVED && scene.is_some();
        let found = CineCoordinates {
            coordinates: q,
            chroma_fraction: if converged { 1.0 } else { 1.0 - 1e-3 },
        };
        (found, scene)
    }

    /// The whole per-pixel return: [`CoreTable::invert`] and the scene side, as scene AWG4 and the share of the
    /// requested chroma that was kept. For a display colour the rendering produces this is the scene colour that
    /// renders to it; for one it cannot, the nearest reachable colour's.
    #[must_use]
    pub fn scene(&self, c: DisplayCoordinates) -> (LinearAwg4, f64) {
        let (found, scene) = self.solve(c);
        let scene = scene.unwrap_or_else(|| scene_from_coordinates(found.coordinates).0);
        (LinearAwg4(scene), found.chroma_fraction)
    }
}

/// `steps` damped Newton steps for `core(q) = goal` from `q`, each with the exact Jacobian at the current point.
/// A step is tried as the Newton step (limited to 0.7 in every coordinate, then halved up to twice) and, when none
/// of those lowers the largest residual, as damped least squares `-(J^T J + mu I)^-1 J^T r` with three growing
/// `mu` (a short step along the descent direction that stays near a fold, where the Newton step is large and
/// points away); the first that lowers the residual is kept, and otherwise the point stays. The loop count is
/// fixed (a step is skipped once the residual is below [`CONVERGED`], where it would not move; a kernel runs them
/// all); the candidates are tried in a fixed order.
fn refine(mut q: Vec3, goal: Vec3, steps: usize) -> Vec3 {
    let mut r = residual(q, goal);
    let mut cost = norm_inf(r);
    for _ in 0..steps {
        if cost < CONVERGED {
            break;
        }
        let j = jacobian(q);
        let better = |qn: Vec3| -> Option<(Vec3, f64)> {
            let rn = residual(qn, goal);
            let cn = norm_inf(rn);
            (cn < cost).then_some((rn, cn))
        };
        let mut next: Option<(Vec3, Vec3, f64)> = None;
        if let Some(ji) = inverse3(&j) {
            let step = [0, 1, 2].map(|i| -(ji[i][0] * r[0] + ji[i][1] * r[1] + ji[i][2] * r[2]));
            let size = step[0].abs().max(step[1].abs()).max(step[2].abs());
            let scale = if size > STEP_LIMIT {
                STEP_LIMIT / size
            } else {
                1.0
            };
            for s in STEP_SCALES.map(|f| f * scale) {
                let qn = [0, 1, 2].map(|i| q[i] + s * step[i]);
                if let Some((rn, cn)) = better(qn) {
                    next = Some((qn, rn, cn));
                    break;
                }
            }
        }
        if next.is_none() {
            // Damped least squares.
            let mut a = [[0.0_f64; 3]; 3];
            let mut g = [0.0_f64; 3];
            for (row, &res) in j.iter().zip(&r) {
                for p in 0..3 {
                    g[p] += row[p] * res;
                    for k in 0..3 {
                        a[p][k] += row[p] * row[k];
                    }
                }
            }
            let scale = (a[0][0] + a[1][1] + a[2][2]).max(LM_TRACE_FLOOR) / 3.0;
            for mu in LM_DAMPING {
                let mut m = a;
                for (p, row) in m.iter_mut().enumerate() {
                    row[p] += mu * scale;
                }
                let Some(inv) = inverse3(&m) else { continue };
                let step = [0, 1, 2].map(|i| {
                    (-(inv[i][0] * g[0] + inv[i][1] * g[1] + inv[i][2] * g[2]))
                        .clamp(-STEP_LIMIT, STEP_LIMIT)
                });
                let qn = [0, 1, 2].map(|i| q[i] + step[i]);
                if let Some((rn, cn)) = better(qn) {
                    next = Some((qn, rn, cn));
                    break;
                }
            }
        }
        if let Some((qn, rn, cn)) = next {
            (q, r, cost) = (qn, rn, cn);
        }
    }
    q
}

/// The level of row `i` of `n`: the rows crowd towards the white level, where the pre-image changes fastest.
fn level_node(s_lo: f64, span: f64, i: usize, n: usize) -> f64 {
    let f = 1.0 - i as f64 / (n - 1) as f64;
    s_lo + span * (1.0 - f * f)
}

/// The nodes of one ray `(s, theta)` and its reachable length. The march from the neutral colour in steps of
/// [`MARCH_STEP`] follows the connected sheet (a step is refused when the solution jumps further than a continuous
/// branch would, as in `solve_connected`); the first refused step is bisected. The nodes at fractions `u` of the
/// reachable length start from the march's piecewise-linear pre-image and are solved by Newton.
fn build_ray(s: f64, theta: f64, nodes: &mut [[f32; 3]]) -> f64 {
    let nk = nodes.len();
    let y2 = tone(s.exp()).max(1e-12);
    let ly2 = y2.ln();
    let (ct, st) = (theta.cos(), theta.sin());
    let goal_at = |rho: f64| [ly2, rho * ct, rho * st];
    let neutral = [s, 0.0, 0.0];
    let (q0, ok0) = newton(neutral, goal_at(0.0), 30);
    let start = if ok0 { q0 } else { neutral };
    let try_at = |from: Vec3, rho: f64, step: f64| -> Option<Vec3> {
        let (q, ok) = newton(from, goal_at(rho), 20);
        (ok && scene_of(q).is_some() && jump(q, from) <= jump_bound(step / rho.max(step), rho))
            .then_some(q)
    };
    // The march: `samples` are (rho, pre-image) along the ray.
    let mut samples: Vec<(f64, Vec3)> = vec![(0.0, start)];
    let mut last = start;
    let mut ok_rho = 0.0;
    let mut rho_max = R_MAX;
    // The step grows geometrically from `MARCH_START` (a quarter of the distance travelled) up to `MARCH_STEP`, and
    // follows the pre-image: it shrinks where the pre-image moves fast (near the neutral colour, and approaching a
    // fold) so that the nodes' starts are good and a long step cannot hop to another sheet. A refused step is
    // retried at a quarter of its length before it is taken for the end of the ray.
    let mut rho = 0.0_f64;
    let mut step = MARCH_START;
    'march: loop {
        let mut attempt = step;
        let (q, rho_next) = loop {
            let rho_next = (rho + attempt).min(R_MAX);
            if rho_next <= rho {
                break 'march;
            }
            if let Some(q) = try_at(last, rho_next, rho_next - ok_rho) {
                break (q, rho_next);
            }
            if attempt <= 4.0 * MARCH_START {
                // The end of the ray: bisect the refused step.
                let (mut lo, mut hi) = (ok_rho, rho_next);
                for _ in 0..BISECTIONS {
                    let mid = 0.5 * (lo + hi);
                    if let Some(q) = try_at(last, mid, mid - lo) {
                        lo = mid;
                        last = q;
                    } else {
                        hi = mid;
                    }
                }
                rho_max = lo;
                samples.push((lo, last));
                break 'march;
            }
            attempt *= 0.25;
        };
        let moved = jump(q, last);
        last = q;
        ok_rho = rho_next;
        rho = rho_next;
        samples.push((rho, q));
        step = (attempt * (MARCH_MOVE / moved.max(1e-12)).clamp(0.5, 2.0))
            .min(MARCH_STEP)
            .min((0.25 * rho).max(MARCH_START))
            .max(MARCH_START);
    }
    if rho_max == R_MAX {
        rho_max = ok_rho;
    }
    let edge = last;
    // The nodes.
    let along = |rho: f64| -> Vec3 {
        let at = samples
            .partition_point(|&(r, _)| r <= rho)
            .clamp(1, samples.len().max(2) - 1);
        if samples.len() < 2 {
            return samples[0].1;
        }
        let ((r0, a), (r1, b)) = (samples[at - 1], samples[at]);
        let w = if r1 > r0 {
            ((rho - r0) / (r1 - r0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        [0, 1, 2].map(|k| a[k] + w * (b[k] - a[k]))
    };
    let mut previous = start;
    for (k, node) in nodes.iter_mut().enumerate() {
        let q = if k == 0 {
            start
        } else if k == nk - 1 {
            edge
        } else {
            let f = 1.0 - k as f64 / (nk - 1) as f64;
            let rho = rho_max * (1.0 - f * f);
            let guess = along(rho);
            let (q, ok) = newton(guess, goal_at(rho), 20);
            if ok
                && scene_of(q).is_some()
                && jump(q, guess) <= NODE_JUMP
                && jump(q, previous) <= 1.0
            {
                q
            } else {
                guess
            }
        };
        previous = q;
        *node = [(q[0] - s) as f32, q[1] as f32, q[2] as f32];
    }
    rho_max
}
