//! Baking: the per-pixel colour part of the chain as tables (`docs/24-film.md` section 3.8).
//!
//! The CPU reference ([`FilmChain`]) evaluates 81 wavelengths several times a pixel. The app and `ym` evaluate the
//! same colour part from four kinds of table instead, which this module bakes in f64 from the chain's own public
//! stages and stores as f32:
//!
//! | Table | Maps | Stage |
//! |---|---|---|
//! | [`T1`] | chromaticity (2D) to raw exposure per unit scale (3) | 2 |
//! | [`Curves`] | log exposure to density, 1D, 4,096 entries, 9 curves: the film's with and without couplers, the paper's | 4, 5, 8 |
//! | T3a ([`Table3`]) | negative density (3D) to print log-exposure (3), the **lights** inside | 7 |
//! | T3b ([`Table3`]) | print density (3D) to LogC4 / AWG4 (3), the inverse of the cine rendering inside | 9, 10 |
//!
//! A slide film has no T3a; its T3b reads the film's own dyes.
//!
//! **What stays per pixel.** Stage 1 (the scene light: a Bradford step and the chroma compression, shared with the
//! reference), and stage 5, the DIR couplers, as a *3 x 3 step*: a few Newton iterations on the baked curves
//! without couplers and the matrix `M` of `couplers.json`. The couplers are not baked into a table: they act on
//! the three layers at once, and their spatial diffusion (F-05) cannot be a table anyway.
//!
//! **Exact where it can be.** T1 is a bilinear table over the same grid as the upsampling table (steps of 1/255),
//! and the upsampling table mixes its nodes' spectra bilinearly, so T1 is exact up to f32 rounding. The 3D tables
//! interpolate **tetrahedrally** from f32 storage, with the formula of the look LUTs (`ym_color::lut::Lut3d`,
//! `docs/02` C10), never trilinearly. The f32 `vec4` storage (the fourth entry is 0) is what the GPU reads.
//! [`Table3::sample`] is its CPU twin: f32 storage, f64 arithmetic.
//!
//! **T3b's axes are warped.** The inverse of the cine rendering is logarithmic at the paper white (the soft clip),
//! so T3b's nodes sit at the squares of a uniform grid ([`T3B_POWER`]): ten times the resolution near white, and
//! the lookup takes a square root of each density. The curvature of `exp` at large density is what this costs.
//!
//! **Without a re-bake.** The film exposure multiplies T1's result ([`BakedFilm::set_exposure_ev`]) and the print
//! exposure shifts the log-exposure after T3a ([`BakedFilm::set_print_stops`]). The lights are inside T3a, so they
//! re-bake it, at [`DRAFT_SIZE`] while a slider moves and at the full size on request.
//!
//! **Determinism.** Every node is independent and the work is cut into contiguous runs of nodes, so any thread
//! count gives the same bytes.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use sha2::{Digest, Sha256};

use crate::Vec3;
use crate::chain::FilmChain;
use crate::error::FilmError;
use crate::exposure::ExposureStage;
use crate::math::{pow10, solve3};
use crate::params::{Output, load_couplers};
use crate::profile::Profile;
use crate::scene::{SceneLight, SceneStage};

const LOG10_2: f64 = std::f64::consts::LOG10_2;

/// The version of the baking. It is part of every cache key (and of the export's `ym:FilmData`); raise it when a
/// change to the engine alters what a table holds without altering its inputs.
pub const ENGINE_VERSION: u32 = 1;

/// Entries of every 1D curve table.
pub const CURVE_ENTRIES: usize = 4096;

/// The size of the draft T3a, used while a lights slider moves.
pub const DRAFT_SIZE: usize = 17;

/// The size of T3a when the caller does not choose (`docs/24` section 3.8, *As built*).
pub const DEFAULT_T3A_SIZE: usize = 49;

/// The size of T3b when the caller does not choose. None of 33, 49 and 65 meets the baking gate; 65 is the
/// largest of them (`docs/24` section 3.8, *As built*).
pub const DEFAULT_T3B_SIZE: usize = 65;

/// The sizes of the 3D tables that section 3.8 compares.
pub const CANDIDATE_SIZES: [usize; 3] = [33, 49, 65];

/// T1's node spacing is `1 / T1_GRID`: the upsampling table's own spacing (256 nodes over 0 to 1).
pub const T1_GRID: f64 = 255.0;
/// T1's nodes along x (chromaticity 0 to 0.7647; the spectral locus reaches 0.7347).
pub const T1_NX: usize = 196;
/// T1's nodes along y (chromaticity 0 to 0.8588; the spectral locus reaches 0.8338).
pub const T1_NY: usize = 220;

/// The margin of the 3D axes beyond the largest density, as a fraction of it.
const AXIS_MARGIN: f64 = 0.02;
/// The most iterations of the baked coupler step.
const COUPLER_ITERATIONS: usize = 16;
/// T3b's axis warp: nodes at the squares of a uniform grid, because the inverse of the cine rendering is singular
/// (logarithmic) at the paper white.
pub const T3B_POWER: f64 = 2.0;

/// Cache sizes: T1 keeps enough entries for the stock picker's tiles (`docs/24` section 4.4); the others keep the
/// eight of section 3.8.
pub const T1_CACHE_ENTRIES: usize = 24;
/// Entries of the curves, T3a and T3b caches.
pub const CACHE_ENTRIES: usize = 8;

/// A 32-byte SHA-256.
pub type Digest32 = [u8; 32];

// ---------------------------------------------------------------------------------------------------------
// The lights
// ---------------------------------------------------------------------------------------------------------

/// The enlarger's filter shifts from neutral, in Kodak CC units (`FilmParams::light_*_cc`). They are inside T3a.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Lights {
    /// Yellow.
    pub yellow_cc: f64,
    /// Magenta.
    pub magenta_cc: f64,
    /// Cyan.
    pub cyan_cc: f64,
}

impl Lights {
    /// The lights a chain was built with.
    #[must_use]
    pub fn of(chain: &FilmChain) -> Self {
        let p = chain.params();
        Self {
            yellow_cc: p.light_yellow_cc,
            magenta_cc: p.light_magenta_cc,
            cyan_cc: p.light_cyan_cc,
        }
    }

    fn bits(self) -> [u64; 3] {
        [
            self.yellow_cc.to_bits(),
            self.magenta_cc.to_bits(),
            self.cyan_cc.to_bits(),
        ]
    }
}

/// What to bake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakeSettings {
    /// The lights inside T3a.
    pub lights: Lights,
    /// Nodes per side of T3a (ignored for a slide). [`DRAFT_SIZE`] gives a draft.
    pub t3a_size: usize,
    /// Nodes per side of T3b.
    pub t3b_size: usize,
}

impl BakeSettings {
    /// The chain's own lights and the default sizes ([`DEFAULT_T3A_SIZE`], [`DEFAULT_T3B_SIZE`]).
    #[must_use]
    pub fn of(chain: &FilmChain) -> Self {
        Self {
            lights: Lights::of(chain),
            t3a_size: DEFAULT_T3A_SIZE,
            t3b_size: DEFAULT_T3B_SIZE,
        }
    }

    /// The same with a draft T3a.
    #[must_use]
    pub fn draft(self) -> Self {
        Self {
            t3a_size: DRAFT_SIZE,
            ..self
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The tables
// ---------------------------------------------------------------------------------------------------------

/// T1: chromaticity to raw exposure of the three layers for a light of luminance `0.18` at the film's mid-grey
/// placement (so a neutral at scene 0.18 gives `1, 1, 1` before the exposure gain). Bilinear, f32 `vec4`.
#[derive(Debug, Clone, PartialEq)]
pub struct T1 {
    nx: usize,
    ny: usize,
    step: f64,
    data: Vec<f32>,
}

impl T1 {
    /// Nodes along x and y.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// The node spacing in chromaticity.
    #[must_use]
    pub fn step(&self) -> f64 {
        self.step
    }

    /// The nodes as f32 `vec4` (the fourth is 0), x fastest.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn at(&self, ix: usize, iy: usize) -> Vec3 {
        let o = 4 * (ix + self.nx * iy);
        [
            f64::from(self.data[o]),
            f64::from(self.data[o + 1]),
            f64::from(self.data[o + 2]),
        ]
    }

    /// Bilinear sample at chromaticity `(x, y)`; held at the edge outside the table.
    #[must_use]
    pub fn sample(&self, x: f64, y: f64) -> Vec3 {
        let fx = grid_pos(x / self.step, self.nx);
        let fy = grid_pos(y / self.step, self.ny);
        let ix = (fx.floor() as usize).min(self.nx - 2);
        let iy = (fy.floor() as usize).min(self.ny - 2);
        let (tx, ty) = (fx - ix as f64, fy - iy as f64);
        let (a, b, c, d) = (
            self.at(ix, iy),
            self.at(ix + 1, iy),
            self.at(ix, iy + 1),
            self.at(ix + 1, iy + 1),
        );
        let mut out = [0.0; 3];
        for k in 0..3 {
            let lo = a[k] + (b[k] - a[k]) * tx;
            let hi = c[k] + (d[k] - c[k]) * tx;
            out[k] = lo + (hi - lo) * ty;
        }
        out
    }
}

/// `p` clamped into `0 ..= n - 1` (NaN is 0).
fn grid_pos(p: f64, n: usize) -> f64 {
    if p.is_nan() {
        0.0
    } else {
        p.clamp(0.0, (n - 1) as f64)
    }
}

/// One baked 1D curve: `CURVE_ENTRIES` f32 values over `[lo, hi]`, linear between them, held at the edge.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve1 {
    lo: f64,
    hi: f64,
    data: Vec<f32>,
}

impl Curve1 {
    fn bake(lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> Self {
        let n = CURVE_ENTRIES;
        let data = (0..n)
            .map(|i| f(lo + (hi - lo) * i as f64 / (n - 1) as f64) as f32)
            .collect();
        Self { lo, hi, data }
    }

    /// The axis ends.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        (self.lo, self.hi)
    }

    /// The entries.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The value at `x`.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.data.len();
        let pos = grid_pos((x - self.lo) / (self.hi - self.lo) * (n - 1) as f64, n);
        let i = (pos.floor() as usize).min(n - 2);
        let t = pos - i as f64;
        let (a, b) = (f64::from(self.data[i]), f64::from(self.data[i + 1]));
        a + (b - a) * t
    }

    /// The slope of the piece `x` lies in (0 outside the axis).
    #[must_use]
    pub fn slope(&self, x: f64) -> f64 {
        if x.is_nan() || x < self.lo || x > self.hi {
            return 0.0;
        }
        let n = self.data.len();
        let pos = (x - self.lo) / (self.hi - self.lo) * (n - 1) as f64;
        let i = (pos.floor() as usize).min(n - 2);
        let h = (self.hi - self.lo) / (n - 1) as f64;
        (f64::from(self.data[i + 1]) - f64::from(self.data[i])) / h
    }
}

/// The 1D tables: three channels each of the film's curves with the couplers (the profile's), the film's curves
/// without the couplers (the argument is the effective exposure), and the paper's. 9 x 4,096 f32.
#[derive(Debug, Clone, PartialEq)]
pub struct Curves {
    /// Stage 4: the profile's `density_curves`.
    pub film: [Curve1; 3],
    /// The curves with the couplers taken out; stage 5 evaluates them at `log E - amount * M D`.
    pub film0: [Curve1; 3],
    /// Stage 8: the paper's curves (none for a slide).
    pub paper: Option<[Curve1; 3]>,
}

/// A 3D table: `size^3` nodes of f32 `vec4`, red fastest, over `[0, hi]` per channel; tetrahedral interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct Table3 {
    size: usize,
    hi: Vec3,
    /// The axis warp: node `i` of `n` sits at density `hi * (i / (n - 1))^power`. 1 is a uniform axis.
    power: f64,
    data: Vec<f32>,
}

impl Table3 {
    /// Bakes `f` at the nodes of a `size`^3 table over `[0, hi]` per channel on `threads` threads. The bytes do
    /// not depend on `threads`.
    #[must_use]
    pub fn bake(
        size: usize,
        hi: Vec3,
        power: f64,
        threads: usize,
        f: impl Fn(Vec3) -> Vec3 + Sync,
    ) -> Self {
        bake_table3(size, hi, power, threads, f)
    }

    /// The axis warp (1 for a uniform axis, 2 for nodes at the squares of a uniform grid).
    #[must_use]
    pub fn power(&self) -> f64 {
        self.power
    }

    /// Nodes per side.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The upper end of each axis (the lower end is 0).
    #[must_use]
    pub fn axis_max(&self) -> Vec3 {
        self.hi
    }

    /// The nodes as f32 `vec4` (the fourth is 0), red fastest: the layout of an f32 storage buffer.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn at(&self, r: usize, g: usize, b: usize) -> Vec3 {
        let o = 4 * (r + self.size * (g + self.size * b));
        [
            f64::from(self.data[o]),
            f64::from(self.data[o + 1]),
            f64::from(self.data[o + 2]),
        ]
    }

    /// Tetrahedral interpolation, the formula of `ym_color::lut::Lut3d::sample`: the cell is split into six
    /// tetrahedra by the order of the fractions, so the result is continuous across cells and exact at nodes. A
    /// point outside the axes is held at the edge; NaN counts as 0.
    #[must_use]
    pub fn sample(&self, p: Vec3) -> Vec3 {
        let n = self.size;
        let mut idx = [0_usize; 3];
        let mut frac = [0.0_f64; 3];
        for c in 0..3 {
            let v = if p[c].is_nan() {
                0.0
            } else {
                p[c].clamp(0.0, self.hi[c])
            };
            let u = v / self.hi[c];
            let u = if self.power == 1.0 {
                u
            } else if self.power == 2.0 {
                u.sqrt()
            } else {
                u.powf(1.0 / self.power)
            };
            let pos = u * (n - 1) as f64;
            let i = (pos.floor() as usize).min(n - 2);
            idx[c] = i;
            frac[c] = pos - i as f64;
        }
        let [i, j, k] = idx;
        let [fr, fg, fb] = frac;
        let c000 = self.at(i, j, k);
        let c111 = self.at(i + 1, j + 1, k + 1);
        let (p1, p2, w0, w1, w2) = if fr > fg {
            if fg > fb {
                (self.at(i + 1, j, k), self.at(i + 1, j + 1, k), fr, fg, fb)
            } else if fr > fb {
                (self.at(i + 1, j, k), self.at(i + 1, j, k + 1), fr, fb, fg)
            } else {
                (self.at(i, j, k + 1), self.at(i + 1, j, k + 1), fb, fr, fg)
            }
        } else if fb > fg {
            (self.at(i, j, k + 1), self.at(i, j + 1, k + 1), fb, fg, fr)
        } else if fb > fr {
            (self.at(i, j + 1, k), self.at(i, j + 1, k + 1), fg, fb, fr)
        } else {
            (self.at(i, j + 1, k), self.at(i + 1, j + 1, k), fg, fr, fb)
        };
        let mut out = [0.0; 3];
        for c in 0..3 {
            out[c] =
                c000[c] + w0 * (p1[c] - c000[c]) + w1 * (p2[c] - p1[c]) + w2 * (c111[c] - p2[c]);
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------
// Parallel filling
// ---------------------------------------------------------------------------------------------------------

/// Fills `out` (`4 * n` floats, `n` items) with `f(item)`, cutting the items into `threads` contiguous runs. The
/// items are independent, so the bytes do not depend on `threads`.
fn fill<F>(out: &mut [f32], threads: usize, f: F)
where
    F: Fn(usize) -> [f32; 4] + Sync,
{
    let n = out.len() / 4;
    let threads = threads.clamp(1, n.max(1));
    let run = |first: usize, chunk: &mut [f32]| {
        for (k, px) in chunk.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&f(first + k));
        }
    };
    if threads == 1 {
        run(0, out);
        return;
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(4 * per).enumerate() {
            let run = &run;
            s.spawn(move || run(t * per, chunk));
        }
    });
}

fn f32x4(v: Vec3) -> [f32; 4] {
    let g = |x: f64| if x.is_finite() { x as f32 } else { 0.0 };
    [g(v[0]), g(v[1]), g(v[2]), 0.0]
}

fn bake_table3(
    size: usize,
    hi: Vec3,
    power: f64,
    threads: usize,
    f: impl Fn(Vec3) -> Vec3 + Sync,
) -> Table3 {
    let mut data = vec![0.0_f32; 4 * size * size * size];
    let last = (size - 1) as f64;
    let at = |i: usize, c: usize| {
        let u = i as f64 / last;
        hi[c] * if power == 1.0 { u } else { u.powf(power) }
    };
    fill(&mut data, threads, |n| {
        let (r, g, b) = (n % size, (n / size) % size, n / (size * size));
        f32x4(f([at(r, 0), at(g, 1), at(b, 2)]))
    });
    Table3 {
        size,
        hi,
        power,
        data,
    }
}

// ---------------------------------------------------------------------------------------------------------
// The baked film
// ---------------------------------------------------------------------------------------------------------

/// The values after each baked stage, like [`crate::chain::Taps`] (the chain's `display` is the final LogC4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakedTaps {
    /// After stage 2: `log10` raw exposure.
    pub log_e_film: Vec3,
    /// After stage 5: density with the couplers acting.
    pub density_film: Vec3,
    /// After T3a and the print exposure: the paper's `log10` exposure (zero for a slide).
    pub log_e_print: Vec3,
    /// After stage 8: the paper's dye densities (zero for a slide).
    pub density_print: Vec3,
    /// After T3b: LogC4 / AWG4.
    pub logc4: Vec3,
}

/// The chain's colour part from tables. Cheap to clone (the tables are shared).
#[derive(Debug, Clone)]
pub struct BakedFilm {
    scene: SceneStage,
    x_mid: f64,
    film_gain: f64,
    amount: f64,
    matrix: [[f64; 3]; 3],
    exposure_ev: f64,
    print_stops: f64,
    t1: Arc<T1>,
    curves: Arc<Curves>,
    t3a: Option<Arc<Table3>>,
    t3b: Arc<Table3>,
    output: Output,
}

impl BakedFilm {
    /// Film exposure, in stops (no re-bake).
    pub fn set_exposure_ev(&mut self, ev: f64) {
        self.exposure_ev = ev;
    }

    /// Print exposure, in stops (no re-bake).
    pub fn set_print_stops(&mut self, stops: f64) {
        self.print_stops = stops;
    }

    /// The same film with another T3a (a refined one replacing a draft).
    #[must_use]
    pub fn with_t3a(&self, t3a: Arc<Table3>) -> Self {
        Self {
            t3a: Some(t3a),
            ..self.clone()
        }
    }

    /// The output primaries T3b was baked for.
    #[must_use]
    pub fn output(&self) -> Output {
        self.output
    }

    /// Whether this is a slide (no enlarger, no paper).
    #[must_use]
    pub fn is_slide(&self) -> bool {
        self.t3a.is_none()
    }

    /// T1.
    #[must_use]
    pub fn t1(&self) -> &Arc<T1> {
        &self.t1
    }

    /// The 1D tables.
    #[must_use]
    pub fn curves(&self) -> &Arc<Curves> {
        &self.curves
    }

    /// T3a (none for a slide).
    #[must_use]
    pub fn t3a(&self) -> Option<&Arc<Table3>> {
        self.t3a.as_ref()
    }

    /// T3b.
    #[must_use]
    pub fn t3b(&self) -> &Arc<Table3> {
        &self.t3b
    }

    /// Stage 2 from T1: the `log10` raw exposure of a scene light.
    #[must_use]
    pub fn log_exposure(&self, light: &SceneLight) -> Vec3 {
        let e1 = self.t1.sample(light.x, light.y);
        let gain = self.exposure_ev * LOG10_2 + self.film_gain;
        let scale = light.luminance / 0.18 * pow10(self.x_mid + gain);
        e1.map(|v| (scale * v).max(1e-30).log10())
    }

    /// Stage 5 on the baked curves: the density with the couplers acting, a damped Newton step on the three
    /// layers (the 3 x 3 matrix `M` and the baked curves without couplers).
    #[must_use]
    pub fn film_density(&self, log_e: Vec3) -> Vec3 {
        let c0 = &self.curves.film0;
        let mut d = [
            c0[0].eval(log_e[0]),
            c0[1].eval(log_e[1]),
            c0[2].eval(log_e[2]),
        ];
        if self.amount == 0.0 {
            return d;
        }
        let (a, m) = (self.amount, &self.matrix);
        let resid = |d: &Vec3| -> Vec3 {
            let mut g = [0.0; 3];
            for i in 0..3 {
                let inh: f64 = (0..3).map(|j| m[i][j] * d[j]).sum();
                g[i] = c0[i].eval(log_e[i] - a * inh) - d[i];
            }
            g
        };
        let norm = |g: &Vec3| g[0].abs() + g[1].abs() + g[2].abs();
        let mut g = resid(&d);
        for _ in 0..COUPLER_ITERATIONS {
            if norm(&g) < 1e-12 {
                break;
            }
            let mut jac = [[0.0; 3]; 3];
            for i in 0..3 {
                let inh: f64 = (0..3).map(|j| m[i][j] * d[j]).sum();
                let s = c0[i].slope(log_e[i] - a * inh);
                for j in 0..3 {
                    jac[i][j] = -s * a * m[i][j] - if i == j { 1.0 } else { 0.0 };
                }
            }
            let Some(step) = solve3(jac, [-g[0], -g[1], -g[2]]) else {
                break;
            };
            let mut t = 1.0;
            let mut improved = false;
            for _ in 0..12 {
                let cand = [d[0] + t * step[0], d[1] + t * step[1], d[2] + t * step[2]];
                let gc = resid(&cand);
                if norm(&gc) < norm(&g) {
                    d = cand;
                    g = gc;
                    improved = true;
                    break;
                }
                t *= 0.5;
            }
            if !improved {
                break;
            }
        }
        d
    }

    /// Stage 7 from T3a, plus the print exposure: the paper's `log10` exposure (zero for a slide).
    #[must_use]
    pub fn print_log_exposure(&self, density_film: Vec3) -> Vec3 {
        match &self.t3a {
            Some(t) => {
                let v = t.sample(density_film);
                let s = self.print_stops * LOG10_2;
                [v[0] + s, v[1] + s, v[2] + s]
            }
            None => [0.0; 3],
        }
    }

    /// Stage 8 from the paper's baked curves (zero for a slide).
    #[must_use]
    pub fn paper_density(&self, log_e_print: Vec3) -> Vec3 {
        match &self.curves.paper {
            Some(p) => [
                p[0].eval(log_e_print[0]),
                p[1].eval(log_e_print[1]),
                p[2].eval(log_e_print[2]),
            ],
            None => [0.0; 3],
        }
    }

    /// Stages 1 to 8: what the scan reads (the print's densities, or a slide's film densities).
    #[must_use]
    pub fn scan_input(&self, logc4: Vec3) -> Vec3 {
        let (_, taps) = self.run_tapped(logc4);
        if self.is_slide() {
            taps.density_film
        } else {
            taps.density_print
        }
    }

    /// The whole baked chain on one LogC4 / AWG4 pixel, with the values after each stage.
    #[must_use]
    pub fn run_tapped(&self, logc4: Vec3) -> (Vec3, BakedTaps) {
        let light = self.scene.run(logc4);
        let log_e_film = self.log_exposure(&light);
        let density_film = self.film_density(log_e_film);
        let (log_e_print, density_print, out) = if self.is_slide() {
            ([0.0; 3], [0.0; 3], self.t3b.sample(density_film))
        } else {
            let p = self.print_log_exposure(density_film);
            let dp = self.paper_density(p);
            (p, dp, self.t3b.sample(dp))
        };
        (
            out,
            BakedTaps {
                log_e_film,
                density_film,
                log_e_print,
                density_print,
                logc4: out,
            },
        )
    }

    /// The whole baked chain on one LogC4 / AWG4 pixel.
    #[must_use]
    pub fn run(&self, logc4: Vec3) -> Vec3 {
        self.run_tapped(logc4).0
    }
}

// ---------------------------------------------------------------------------------------------------------
// Keys and the cache
// ---------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Hash(Sha256);

impl Hash {
    fn str(&mut self, s: &str) {
        self.0.update((s.len() as u64).to_le_bytes());
        self.0.update(s.as_bytes());
    }
    fn f64s<'a>(&mut self, v: impl IntoIterator<Item = &'a f64>) {
        let mut n = 0_u64;
        for x in v {
            self.0.update(x.to_bits().to_le_bytes());
            n += 1;
        }
        self.0.update(n.to_le_bytes());
    }
    fn finish(self) -> Digest32 {
        self.0.finalize().into()
    }
}

/// The SHA-256 of a profile's content (its numbers, in a fixed order). It identifies the data in cache keys.
#[must_use]
pub fn profile_digest(p: &Profile) -> Digest32 {
    let mut h = Hash::default();
    h.str(&p.id);
    h.str(&p.reference_illuminant);
    h.str(&p.viewing_illuminant);
    h.f64s(p.log_sensitivity.iter().flatten());
    h.f64s(&p.window_params);
    h.f64s(p.channel_density.iter().flatten());
    h.f64s(&p.base_density);
    h.f64s(&p.midscale_neutral_density);
    h.f64s(&p.log_exposure);
    h.f64s(p.density_curves.iter().flatten());
    h.finish()
}

/// A stock in a key: its id and the SHA-256 of its data.
type StockKey = (String, Digest32);

fn stock_key(p: &Profile) -> StockKey {
    (p.id.clone(), profile_digest(p))
}

#[derive(Clone, PartialEq, Eq)]
struct T1Key {
    film: StockKey,
    spectra: String,
    use_window: bool,
    /// Stage 2 evaluated at two chromaticities: a change to the window reading or the sensitivities changes it.
    probe: [u64; 6],
    engine: u32,
}

#[derive(Clone, PartialEq, Eq)]
struct CurvesKey {
    film: StockKey,
    paper: Option<StockKey>,
    matrix: [u64; 9],
    engine: u32,
}

#[derive(Clone, PartialEq, Eq)]
struct T3aKey {
    film: StockKey,
    paper: StockKey,
    lights: [u64; 3],
    /// The calibration of the chain (neutral filters and the brightness match), which sit inside the table.
    calibration: [u64; 4],
    size: usize,
    engine: u32,
}

#[derive(Clone, PartialEq, Eq)]
struct T3bKey {
    medium: StockKey,
    slide: bool,
    output: u8,
    size: usize,
    engine: u32,
}

fn matrix_bits(m: &[[f64; 3]; 3]) -> [u64; 9] {
    let mut out = [0_u64; 9];
    for (k, v) in m.iter().flatten().enumerate() {
        out[k] = v.to_bits();
    }
    out
}

struct Lru<K, V> {
    cap: usize,
    items: Vec<(K, V)>,
}

impl<K: PartialEq, V: Clone> Lru<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            items: Vec::new(),
        }
    }

    /// The value for `k`, which becomes the most recently used.
    fn get(&mut self, k: &K) -> Option<V> {
        let at = self.items.iter().position(|(key, _)| key == k)?;
        let item = self.items.remove(at);
        let v = item.1.clone();
        self.items.push(item);
        Some(v)
    }

    fn put(&mut self, k: K, v: V) {
        if let Some(at) = self.items.iter().position(|(key, _)| *key == k) {
            self.items.remove(at);
        }
        self.items.push((k, v));
        while self.items.len() > self.cap {
            self.items.remove(0);
        }
    }
}

struct Caches {
    t1: Lru<T1Key, Arc<T1>>,
    curves: Lru<CurvesKey, Arc<Curves>>,
    t3a: Lru<T3aKey, Arc<Table3>>,
    t3b: Lru<T3bKey, Arc<Table3>>,
    built: BakeCounts,
}

/// How many tables a [`Baker`] has built (not fetched from its cache), and how many it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BakeCounts {
    /// T1 tables built.
    pub t1: u64,
    /// Curve sets built.
    pub curves: u64,
    /// T3a tables built (drafts included).
    pub t3a: u64,
    /// T3b tables built.
    pub t3b: u64,
}

/// Bakes tables for chains and keeps them in memory: an LRU per kind (T1: [`T1_CACHE_ENTRIES`]; the others:
/// [`CACHE_ENTRIES`]). Nothing is written to disk. Every key holds the profiles' SHA-256 and [`ENGINE_VERSION`].
pub struct Baker {
    couplers: [[f64; 3]; 3],
    spectra_id: String,
    threads: usize,
    caches: Mutex<Caches>,
}

impl std::fmt::Debug for Baker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Baker")
            .field("threads", &self.threads)
            .field("spectra_id", &self.spectra_id)
            .finish_non_exhaustive()
    }
}

impl Baker {
    /// A baker reading the coupler matrix from `<data_dir>/film/params/`, working on `threads` threads (at least
    /// 1). `spectra_id` names the upsampling data the chains use (for example its SHA-256), and is part of T1's
    /// key.
    ///
    /// # Errors
    ///
    /// [`FilmError`] when `couplers.json` is missing or damaged.
    pub fn new(
        data_dir: &Path,
        spectra_id: impl Into<String>,
        threads: usize,
    ) -> Result<Self, FilmError> {
        let couplers = load_couplers(&data_dir.join("film").join("params"))?;
        Ok(Self {
            couplers,
            spectra_id: spectra_id.into(),
            threads: threads.max(1),
            caches: Mutex::new(Caches {
                t1: Lru::new(T1_CACHE_ENTRIES),
                curves: Lru::new(CACHE_ENTRIES),
                t3a: Lru::new(CACHE_ENTRIES),
                t3b: Lru::new(CACHE_ENTRIES),
                built: BakeCounts::default(),
            }),
        })
    }

    /// The threads a bake uses.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// A baker like this one with another thread count and an empty cache.
    #[must_use]
    pub fn with_threads(&self, threads: usize) -> Self {
        Self {
            couplers: self.couplers,
            spectra_id: self.spectra_id.clone(),
            threads: threads.max(1),
            caches: Mutex::new(Caches {
                t1: Lru::new(T1_CACHE_ENTRIES),
                curves: Lru::new(CACHE_ENTRIES),
                t3a: Lru::new(CACHE_ENTRIES),
                t3b: Lru::new(CACHE_ENTRIES),
                built: BakeCounts::default(),
            }),
        }
    }

    /// How many tables this baker has built so far.
    #[must_use]
    pub fn built(&self) -> BakeCounts {
        self.lock().built
    }

    fn lock(&self) -> MutexGuard<'_, Caches> {
        // A panic while the lock was held leaves the caches in a state that is still valid.
        self.caches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// T1 for the chain's film (cached).
    #[must_use]
    pub fn t1(&self, chain: &FilmChain) -> Arc<T1> {
        let ex = chain.exposure_stage();
        let x_mid = ex.mid_log_exposure();
        let probe = |x: f64, y: f64| {
            ex.raw(
                &SceneLight {
                    x,
                    y,
                    luminance: 0.18,
                },
                -x_mid,
            )
            .map(f64::to_bits)
        };
        let (a, b) = (probe(0.3127, 0.3290), probe(0.45, 0.40));
        let key = T1Key {
            film: stock_key(chain.film()),
            spectra: self.spectra_id.clone(),
            use_window: chain.params().use_window,
            probe: [a[0], a[1], a[2], b[0], b[1], b[2]],
            engine: ENGINE_VERSION,
        };
        if let Some(t) = self.lock().t1.get(&key) {
            return t;
        }
        let t = Arc::new(bake_t1(ex, self.threads));
        let mut c = self.lock();
        c.built.t1 += 1;
        c.t1.put(key, Arc::clone(&t));
        t
    }

    /// The 1D tables for the chain's film and paper (cached).
    #[must_use]
    pub fn curves(&self, chain: &FilmChain) -> Arc<Curves> {
        let key = CurvesKey {
            film: stock_key(chain.film()),
            paper: chain.paper().map(stock_key),
            matrix: matrix_bits(&self.couplers),
            engine: ENGINE_VERSION,
        };
        if let Some(t) = self.lock().curves.get(&key) {
            return t;
        }
        let t = Arc::new(bake_curves(chain, &self.couplers));
        let mut c = self.lock();
        c.built.curves += 1;
        c.curves.put(key, Arc::clone(&t));
        t
    }

    /// T3a for the chain's film and paper with `lights`, `size` nodes a side. `None` for a slide. A draft
    /// (`size <= DRAFT_SIZE`) is built every time and not cached.
    #[must_use]
    pub fn t3a(&self, chain: &FilmChain, lights: Lights, size: usize) -> Option<Arc<Table3>> {
        let paper = chain.paper()?;
        let cal = Calibration::of(chain);
        let key = T3aKey {
            film: stock_key(chain.film()),
            paper: stock_key(paper),
            lights: lights.bits(),
            calibration: [
                cal.neutral[0].to_bits(),
                cal.neutral[1].to_bits(),
                cal.shift.to_bits(),
                cal.p_ref.to_bits(),
            ],
            size,
            engine: ENGINE_VERSION,
        };
        let cached = size > DRAFT_SIZE;
        if cached && let Some(t) = self.lock().t3a.get(&key) {
            return Some(t);
        }
        let t = Arc::new(bake_t3a(chain, &cal, lights, size, self.threads));
        let mut c = self.lock();
        c.built.t3a += 1;
        if cached {
            c.t3a.put(key, Arc::clone(&t));
        }
        Some(t)
    }

    /// T3b for the chain's paper (or a slide's film) and output primaries (cached).
    #[must_use]
    pub fn t3b(&self, chain: &FilmChain, size: usize) -> Arc<Table3> {
        let slide = chain.paper().is_none();
        let medium = chain.paper().unwrap_or_else(|| chain.film());
        let key = T3bKey {
            medium: stock_key(medium),
            slide,
            output: match chain.params().output {
                Output::Srgb => 0,
                Output::DisplayP3 => 1,
            },
            size,
            engine: ENGINE_VERSION,
        };
        if let Some(t) = self.lock().t3b.get(&key) {
            return t;
        }
        let t = Arc::new(bake_t3b(chain, size, self.threads));
        let mut c = self.lock();
        c.built.t3b += 1;
        c.t3b.put(key, Arc::clone(&t));
        t
    }

    /// Every table for the chain, as one [`BakedFilm`] at the chain's exposure and print exposure.
    ///
    /// # Errors
    ///
    /// [`FilmError::Data`] when `settings` asks for a table smaller than 2 nodes a side.
    pub fn bake(&self, chain: &FilmChain, settings: &BakeSettings) -> Result<BakedFilm, FilmError> {
        if settings.t3a_size < 2 || settings.t3b_size < 2 {
            return Err(FilmError::data(
                "bake settings",
                "a 3D table needs at least 2 nodes a side",
            ));
        }
        let p = chain.params();
        Ok(BakedFilm {
            scene: chain.scene_stage().clone(),
            x_mid: chain.exposure_stage().mid_log_exposure(),
            film_gain: Calibration::of(chain).film_gain,
            amount: p.coupler_amount,
            matrix: self.couplers,
            exposure_ev: p.exposure_ev,
            print_stops: p.print_exposure_stops,
            t1: self.t1(chain),
            curves: self.curves(chain),
            t3a: self.t3a(chain, settings.lights, settings.t3a_size),
            t3b: self.t3b(chain, settings.t3b_size),
            output: p.output,
        })
    }
}

// ---------------------------------------------------------------------------------------------------------
// The bakes
// ---------------------------------------------------------------------------------------------------------

/// What the chain's calibration left, recovered from its public stages (the numbers are the chain's own).
struct Calibration {
    /// Yellow and magenta of the neutral filtering.
    neutral: [f64; 2],
    /// Where the grey's red layer lands on the paper's axis.
    shift: f64,
    /// The red layer's light on the paper from the grey at the neutral filtering.
    p_ref: f64,
    /// A slide's gain on the exposure (log10); 0 for a negative.
    film_gain: f64,
}

impl Calibration {
    fn of(chain: &FilmChain) -> Self {
        let grey = ym_color::logc::logc4_encode_rgb([0.18; 3]);
        let light = chain.scene_stage().run(grey);
        let neutral = chain.neutral_filters_cc();
        let (p_ref, film_gain) = match chain.enlarger_stage() {
            Some(enl) => {
                // As the chain's own grey does: gain 0, the couplers at the chain's amount.
                let log_e = chain.exposure_stage().log_exposure(&light, 0.0);
                let d = chain
                    .develop_stage()
                    .couple(log_e, chain.params().coupler_amount);
                (enl.light_on_paper(d, [neutral[0], neutral[1], 0.0])[0], 0.0)
            }
            None => {
                // A slide: the gain is what the chain's tap at the grey shows beyond the exposure setting.
                let probe = chain.exposure_stage().log_exposure(&light, 0.0);
                let tap = chain.run_tapped(grey).1.log_e_film;
                (
                    1.0,
                    tap[0] - probe[0] - chain.params().exposure_ev * LOG10_2,
                )
            }
        };
        Self {
            neutral,
            shift: chain.print_shift(),
            p_ref,
            film_gain,
        }
    }
}

fn bake_t1(ex: &ExposureStage, threads: usize) -> T1 {
    let x_mid = ex.mid_log_exposure();
    let step = 1.0 / T1_GRID;
    let mut data = vec![0.0_f32; 4 * T1_NX * T1_NY];
    fill(&mut data, threads, |n| {
        let (ix, iy) = (n % T1_NX, n / T1_NX);
        let light = SceneLight {
            x: ix as f64 * step,
            y: iy as f64 * step,
            luminance: 0.18,
        };
        // Luminance 0.18 and a gain of -x_mid make the scale 1: raw exposure per unit scale.
        f32x4(ex.raw(&light, -x_mid))
    });
    T1 {
        nx: T1_NX,
        ny: T1_NY,
        step,
        data,
    }
}

fn densities_max(p: &Profile) -> Vec3 {
    let mut m = [0.0_f64; 3];
    for row in &p.density_curves {
        for c in 0..3 {
            m[c] = m[c].max(row[c]);
        }
    }
    m
}

fn with_margin(m: Vec3) -> Vec3 {
    m.map(|v| (v * (1.0 + AXIS_MARGIN)).max(1e-3))
}

fn bake_curves(chain: &FilmChain, matrix: &[[f64; 3]; 3]) -> Curves {
    let film = chain.film();
    let dev = chain.develop_stage();
    let (x0, x1) = (
        film.log_exposure.first().copied().unwrap_or(-3.0),
        film.log_exposure.last().copied().unwrap_or(4.0),
    );
    // The curves without couplers are read at `log E - amount * M D`: up to the largest row of `M` times the
    // largest density below the exposure axis. The effective exposure never exceeds `log E`.
    let dmax = densities_max(film);
    let row = matrix
        .iter()
        .map(|r| r.iter().sum::<f64>())
        .fold(0.0_f64, f64::max);
    let d_all = dmax[0].max(dmax[1]).max(dmax[2]);
    let lo0 = x0 - (row * d_all * 1.05 + 0.05).max(0.0);
    let mk =
        |lo: f64, hi: f64, c: usize, f: &dyn Fn(f64) -> Vec3| Curve1::bake(lo, hi, |x| f(x)[c]);
    let film_c = |x: f64| dev.curves([x; 3]);
    let film0_c = |x: f64| dev.couple([x; 3], 0.0);
    let paper = chain.enlarger_stage().map(|enl| {
        let p = chain.paper();
        let (p0, p1) = p.map_or((-3.0, 4.0), |p| {
            (
                p.log_exposure.first().copied().unwrap_or(-3.0),
                p.log_exposure.last().copied().unwrap_or(4.0),
            )
        });
        let f = |x: f64| enl.paper_density([x; 3]);
        [mk(p0, p1, 0, &f), mk(p0, p1, 1, &f), mk(p0, p1, 2, &f)]
    });
    Curves {
        film: [
            mk(x0, x1, 0, &film_c),
            mk(x0, x1, 1, &film_c),
            mk(x0, x1, 2, &film_c),
        ],
        film0: [
            mk(lo0, x1, 0, &film0_c),
            mk(lo0, x1, 1, &film0_c),
            mk(lo0, x1, 2, &film0_c),
        ],
        paper,
    }
}

fn bake_t3a(
    chain: &FilmChain,
    cal: &Calibration,
    lights: Lights,
    size: usize,
    threads: usize,
) -> Table3 {
    let hi = with_margin(densities_max(chain.film()));
    let cc = [
        cal.neutral[0] + lights.yellow_cc,
        cal.neutral[1] + lights.magenta_cc,
        lights.cyan_cc,
    ];
    let enl = chain.enlarger_stage();
    bake_table3(size, hi, 1.0, threads, |d| {
        let Some(enl) = enl else { return [0.0; 3] };
        let p = enl.light_on_paper(d, cc);
        // The chain adds `stops * log10 2` to this at run time; the shift of the brightness match is in the table.
        p.map(|v| (v.max(1e-300) / cal.p_ref).log10() + cal.shift)
    })
}

fn bake_t3b(chain: &FilmChain, size: usize, threads: usize) -> Table3 {
    let medium = chain.paper().unwrap_or_else(|| chain.film());
    let hi = with_margin(densities_max(medium));
    let scan = chain.scan_stage();
    bake_table3(size, hi, T3B_POWER, threads, |d| {
        chain.return_stage(scan.display(d))
    })
}
