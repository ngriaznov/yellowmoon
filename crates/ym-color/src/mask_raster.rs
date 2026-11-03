//! Raster masks on the CPU reference (`docs/23-nodes.md` section 4.3, `docs/algorithms/06-masks.md` section 5):
//! the photo-frame plane `S0` of a row, its resample to the output grid, Grow, Feather and Refine edges, and
//! the Brush strokes that make a Brush row's `S0`.
//!
//! # The pass
//!
//! A raster row (Object, Subject, People, Background, Brush) has a **source plane** `S0`
//! ([`SourcePlane`]: u16 codes over the photo frame, made once and kept) and three sizes, `grow`, `feather`
//! and `refine`. For a render whose output grid has `u_px` pixels along the photo's long edge
//! ([`PhotoFrame::pixel`] is `1 / u_px`), [`RasterSteps`] turns them into whole pixels, and
//! [`RasterRow::plane`] runs the steps over a window of the grid:
//!
//! 1. **resample** `S0` at each pixel's place in the photo ([`resample_at`]; bilinear, or the mean of `n x n`
//!    bilinear samples when the grid is coarser than `S0` by more than [`SHRINK_BOX_FROM`]);
//! 2. **grow**: a separable maximum (or minimum) over a square of half-side `r_g`;
//! 3. **feather**: three cascaded box means whose widths give the variance of a Gaussian of `sigma_f` pixels;
//! 4. **refine**: the guided filter of He, Sun and Tang with the Input's luma as the guide, in the Fast
//!    Guided Filter's subsampled form above [`GUIDED_SUBSAMPLE_SAMPLES`] samples;
//! 5. the result is quantised to a u16 code ([`quantise`]): the plane the colour pass reads, on the CPU and
//!    on the device alike, so the planes are bytes and not approximations.
//!
//! # Windows, frames and halos
//!
//! Every plane lives on an [`Area`] of the output grid in **absolute** pixel coordinates. A window mean is
//! over the pixels of `[x - r, x + r]` that lie inside the **frame** (the whole output image: a window at
//! its edge is clipped and divided by the number of samples it kept). Every window fold, the sums of the box
//! means and the maximum and minimum of Grow alike, is a **blocked** fold with absolute block origins (van
//! Herk and Gil-Werman: blocks of the window's width, a suffix of one block joined to a prefix of the next, the
//! scans restarting at each block boundary, see `fold_line`), so it costs three operations a pixel at any radius
//! and the value of a window is made from the pixels of the window only, in an order fixed by their absolute
//! positions. A tile of a frame is computed over itself grown by [`RasterSet::halo`] pixels (clipped to the
//! frame) and cut: the value of a pixel then depends only on the absolute pixels of its receptive field, never on
//! where the tile began, so a tile equals the whole frame bit for bit (`docs/17` D4 and D5).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::thread;

use crate::mask::{PhotoFrame, RasterPlanes};

/// Grow at 100 % moves an edge by this fraction of the long edge (`docs/algorithms/06` section 5.1).
pub const GROW_MAX: f64 = 0.01;
/// Feather at 100 % is a Gaussian of this standard deviation, as a fraction of the long edge.
pub const FEATHER_MAX: f64 = 0.02;
/// Refine edges at 100 % is a window of this half-side, as a fraction of the long edge.
pub const GUIDED_RADIUS_MAX: f64 = 0.008;
/// A feather whose standard deviation is under this many output pixels is skipped.
pub const FEATHER_MIN_SIGMA_PX: f64 = 0.5;
/// The most samples one side of the guided filter's window has before it is subsampled (the Fast Guided
/// Filter of He and Sun).
pub const GUIDED_SUBSAMPLE_SAMPLES: usize = 16;
/// The grid is coarser than `S0` by more than this much: the resample averages instead of sampling once.
pub const SHRINK_BOX_FROM: f64 = 1.25;
/// The most samples along one side of the resample's box average.
pub const SHRINK_BOX_MAX: usize = 16;
/// The long edge of a Brush row's `S0`, in pixels.
pub const BRUSH_PLANE_PX: usize = 2048;
/// The most raster planes one render holds (`docs/23` section 5.5: 96 MiB at 1024 px squared).
pub const MAX_RASTER_PLANES: usize = 48;
/// A u16 plane's full scale.
pub const PLANE_ONE: f64 = 65535.0;

// ---------------------------------------------------------------------------------------------
// Areas
// ---------------------------------------------------------------------------------------------

/// A rectangle of the output grid, in absolute pixels: the frame, a tile, a tile grown by its halo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Area {
    /// The first column.
    pub x: i64,
    /// The first row.
    pub y: i64,
    /// The width.
    pub w: usize,
    /// The height.
    pub h: usize,
}

impl Area {
    /// The area `(0, 0)` to `(w, h)`.
    #[must_use]
    pub fn whole(w: usize, h: usize) -> Self {
        Self { x: 0, y: 0, w, h }
    }

    /// One past the last column.
    #[must_use]
    pub fn x1(&self) -> i64 {
        self.x + self.w as i64
    }

    /// One past the last row.
    #[must_use]
    pub fn y1(&self) -> i64 {
        self.y + self.h as i64
    }

    /// Whether the area has no pixel.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// Whether `other` lies inside this area.
    #[must_use]
    pub fn contains(&self, other: &Area) -> bool {
        other.x >= self.x && other.y >= self.y && other.x1() <= self.x1() && other.y1() <= self.y1()
    }

    /// The pixels in both areas.
    #[must_use]
    pub fn intersect(&self, other: &Area) -> Area {
        let (x, y) = (self.x.max(other.x), self.y.max(other.y));
        let (x1, y1) = (self.x1().min(other.x1()), self.y1().min(other.y1()));
        Area {
            x,
            y,
            w: (x1 - x).max(0) as usize,
            h: (y1 - y).max(0) as usize,
        }
    }

    /// This area grown by `halo` pixels on every side and cut to `frame`.
    #[must_use]
    pub fn grown(&self, halo: usize, frame: &Area) -> Area {
        let h = halo as i64;
        Area {
            x: self.x - h,
            y: self.y - h,
            w: self.w + 2 * halo,
            h: self.h + 2 * halo,
        }
        .intersect(frame)
    }
}

// ---------------------------------------------------------------------------------------------
// The source plane
// ---------------------------------------------------------------------------------------------

/// The photo-frame raster `S0` of a row: u16 codes (a code over 65535 is the value) on a grid of
/// `width x height` square pixels over the photo frame, row-major.
#[derive(Clone, PartialEq, Eq)]
pub struct SourcePlane {
    /// What the plane was made from, for cache keys: equal keys are equal planes.
    pub key: u64,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The codes.
    pub data: Vec<u16>,
}

impl std::fmt::Debug for SourcePlane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourcePlane")
            .field("key", &self.key)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl SourcePlane {
    /// The plane's value at pixel `(x, y)` in [0, 1]; the indices are held to the plane.
    #[must_use]
    pub fn at(&self, x: i64, y: i64) -> f64 {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        f64::from(self.data[y * self.width + x]) / PLANE_ONE
    }
}

/// What makes a row's `S0`.
#[derive(Debug, Clone)]
pub enum RasterSource {
    /// A plane made already (a mask the core found, or a test's).
    Plane(Arc<SourcePlane>),
    /// The strokes of a Brush row: drawn into [`BRUSH_PLANE_PX`] on the long edge for the photo's aspect when
    /// the plane is wanted ([`RasterSource::plane`]) and kept in a small cache.
    Brush(Arc<BrushStrokes>),
    /// A Background row: the photo's blur map with the row's Level and Softness (`docs/algorithms/06` section
    /// 5.4), made when the plane is wanted and kept in a small cache under its own key.
    Background(Arc<BackgroundSource>),
}

impl PartialEq for RasterSource {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl RasterSource {
    /// What identifies the source: equal keys are the same strokes or the same plane.
    #[must_use]
    pub fn key(&self) -> u64 {
        match self {
            Self::Plane(p) => p.key,
            Self::Brush(b) => b.key,
            Self::Background(b) => b.key(),
        }
    }

    /// The plane for a photo of this aspect (`(W, H) / L`, one of them 1). A made plane is returned as it is.
    #[must_use]
    pub fn plane(&self, aspect: [f64; 2]) -> Arc<SourcePlane> {
        self.plane_among(aspect, 0)
    }

    /// [`RasterSource::plane`] for a render with `live` raster rows: the caches that keep the plane keep at least
    /// `max(floor, live)` planes (and stay within a byte cap), so that a plan with more than eight rows does not
    /// draw them again on every frame.
    #[must_use]
    pub fn plane_among(&self, aspect: [f64; 2], live: usize) -> Arc<SourcePlane> {
        match self {
            Self::Plane(p) => Arc::clone(p),
            Self::Brush(b) => brush_plane_among(b, aspect, live),
            Self::Background(b) => background_plane_among(b, live),
        }
    }
}

/// One row of a chain's raster planes: where `S0` comes from and the sizes of the steps after it.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterRow {
    /// `S0`.
    pub source: RasterSource,
    /// Grow, -1 to 1.
    pub grow: f64,
    /// Feather, 0 to 1.
    pub feather: f64,
    /// Refine edges, 0 to 1.
    pub refine: f64,
}

/// The raster rows of a chain, in plane order: row `i` is plane `i` (`MaskKind::Raster { plane: i }`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RasterSet {
    /// The rows.
    pub rows: Vec<RasterRow>,
}

impl RasterSet {
    /// Whether the chain has no raster row.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The steps of every row at a render whose output has `u_px` pixels along the photo's long edge.
    #[must_use]
    pub fn steps(&self, u_px: f64) -> Vec<RasterSteps> {
        self.rows
            .iter()
            .map(|r| RasterSteps::of(r.grow, r.feather, r.refine, u_px))
            .collect()
    }

    /// The receptive field of the planes in output pixels: the largest [`RasterSteps::halo`] of the rows. A tile
    /// is computed over itself grown by this, then cut.
    #[must_use]
    pub fn halo(&self, u_px: f64) -> usize {
        self.steps(u_px)
            .iter()
            .map(RasterSteps::halo)
            .max()
            .unwrap_or(0)
    }

    /// Whether any row refines its edges, so that the pass needs the Input's luma as a guide.
    #[must_use]
    pub fn needs_guide(&self, u_px: f64) -> bool {
        self.steps(u_px).iter().any(|s| s.guided.is_some())
    }
}

// ---------------------------------------------------------------------------------------------
// The sizes of the steps
// ---------------------------------------------------------------------------------------------

/// The guided filter's window at a render's scale (`docs/algorithms/06` section 5.1, step 5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Guided {
    /// The window's half-side in output pixels, `round(refine * 0.008 u_px)`, at least 1.
    pub radius: usize,
    /// The subsampling `s = ceil(radius / 16)` above 16 pixels, else 1.
    pub step: usize,
    /// The window's half-side in samples of the subsampled grid, `round(radius / s)`.
    pub samples: usize,
    /// The regularisation `10^(-2 - 3 refine)`.
    pub eps: f64,
}

/// The steps 3 to 5 of a raster row in whole pixels at a render's scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterSteps {
    /// The half-side of Grow's square: positive for a dilation, negative for an erosion, 0 skips the step.
    pub grow: i64,
    /// The half-widths of Feather's three boxes, `None` when the step is skipped.
    pub feather: Option<[usize; 3]>,
    /// The guided filter, `None` when Refine edges is skipped.
    pub guided: Option<Guided>,
}

/// `x` held to [0, 1]; NaN reads as 0.
fn unit(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

/// The half-widths of the three cascaded boxes whose summed variance is the Gaussian `sigma`'s (the standard
/// "almost Gaussian" choice of Wells and Kovesi): `w = sqrt(4 sigma^2 + 1)` rounded down to an odd number
/// `w_l`, `w_u = w_l + 2`, and the first `m = round((12 sigma^2 - 3 w_l^2 - 12 w_l - 9) / (-4 w_l - 4))`
/// boxes are `w_l` wide and the rest `w_u`. A box of width `w` has half-width `(w - 1) / 2`.
#[must_use]
pub fn feather_boxes(sigma: f64) -> [usize; 3] {
    let ideal = (4.0 * sigma * sigma + 1.0).sqrt();
    let mut wl = ideal.floor() as i64;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m = ((12.0 * sigma * sigma - 3.0 * (wl * wl) as f64 - 12.0 * wl as f64 - 9.0)
        / (-4.0 * wl as f64 - 4.0))
        .round()
        .clamp(0.0, 3.0) as usize;
    let half = |w: i64| ((w - 1) / 2) as usize;
    let mut out = [half(wu); 3];
    for b in out.iter_mut().take(m) {
        *b = half(wl);
    }
    out
}

impl RasterSteps {
    /// The sizes of a row's steps at `u_px` output pixels along the photo's long edge. Numbers outside their
    /// ranges are held to them; a number that is not a number is 0.
    #[must_use]
    pub fn of(grow: f64, feather: f64, refine: f64, u_px: f64) -> Self {
        let u = if u_px.is_finite() && u_px > 0.0 {
            u_px
        } else {
            1.0
        };
        let g = if grow.is_nan() {
            0.0
        } else {
            grow.clamp(-1.0, 1.0)
        };
        let grow_px = (g.abs() * GROW_MAX * u).round() as i64;
        let grow_px = if g < 0.0 { -grow_px } else { grow_px };
        let sigma = unit(feather) * FEATHER_MAX * u;
        let feather = (sigma >= FEATHER_MIN_SIGMA_PX).then(|| feather_boxes(sigma));
        let rho = unit(refine);
        let radius = (rho * GUIDED_RADIUS_MAX * u).round() as usize;
        let guided = (radius >= 1).then(|| {
            let step = if radius > GUIDED_SUBSAMPLE_SAMPLES {
                radius.div_ceil(GUIDED_SUBSAMPLE_SAMPLES)
            } else {
                1
            };
            Guided {
                radius,
                step,
                samples: ((radius as f64 / step as f64).round() as usize).max(1),
                eps: 10f64.powf(-2.0 - 3.0 * rho),
            }
        });
        Self {
            grow: grow_px,
            feather,
            guided,
        }
    }

    /// How far the row's output reaches into the grid around a pixel: the sum of the radii of its steps.
    /// Grow's half-side, Feather's three half-widths and the guided filter's two nested windows (and, when it
    /// is subsampled, the block means and the bilinear upsampling at `step` pixels a sample).
    #[must_use]
    pub fn halo(&self) -> usize {
        let feather = self.feather.map_or(0, |b| b.iter().sum());
        let guided = self.guided.map_or(0, |g| {
            if g.step == 1 {
                2 * g.radius
            } else {
                (2 * g.samples + 3) * g.step
            }
        });
        self.grow.unsigned_abs() as usize + feather + guided
    }
}

// ---------------------------------------------------------------------------------------------
// Quantising
// ---------------------------------------------------------------------------------------------

/// A value in [0, 1] as a u16 code: `floor(v 65535 + 0.5)`. NaN and values outside [0, 1] are held to the
/// nearest end.
#[must_use]
pub fn quantise(v: f64) -> u16 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * PLANE_ONE + 0.5).floor() as u16
}

// ---------------------------------------------------------------------------------------------
// Step 2: resample
// ---------------------------------------------------------------------------------------------

/// The number of samples along one side of the resample's box average for `S0` of `long_edge` pixels on a grid
/// of `u_px` pixels along the photo's long edge: 1 up to a shrink of [`SHRINK_BOX_FROM`], else `ceil(shrink)`
/// (at most [`SHRINK_BOX_MAX`]).
#[must_use]
pub fn resample_taps(long_edge: usize, u_px: f64) -> usize {
    let shrink = long_edge as f64 / u_px;
    if shrink.is_nan() || shrink <= SHRINK_BOX_FROM {
        1
    } else {
        (shrink.ceil() as usize).clamp(2, SHRINK_BOX_MAX)
    }
}

/// `S0` at the place `pos` (fractions of the photo frame) for a grid whose pixel is `shrink` pixels of `S0`
/// wide: one bilinear sample (`taps` = 1) or the mean of `taps x taps` bilinear samples spread evenly over the
/// pixel's footprint, rows first. The indices of the four taps are held to the plane.
#[must_use]
pub fn resample_at(src: &SourcePlane, pos: [f64; 2], taps: usize, shrink: f64) -> f64 {
    let u = pos[0] * src.width as f64 - 0.5;
    let v = pos[1] * src.height as f64 - 0.5;
    let bilinear = |u: f64, v: f64| {
        let (fu, fv) = (u.floor(), v.floor());
        let (tu, tv) = (u - fu, v - fv);
        let (i, j) = (fu as i64, fv as i64);
        let top = src.at(i, j) * (1.0 - tu) + src.at(i + 1, j) * tu;
        let bottom = src.at(i, j + 1) * (1.0 - tu) + src.at(i + 1, j + 1) * tu;
        top * (1.0 - tv) + bottom * tv
    };
    if taps <= 1 {
        return unit(bilinear(u, v));
    }
    let n = taps as f64;
    let mut sum = 0.0;
    for b in 0..taps {
        let dv = ((b as f64 + 0.5) / n - 0.5) * shrink;
        for a in 0..taps {
            let du = ((a as f64 + 0.5) / n - 0.5) * shrink;
            sum += bilinear(u + du, v + dv);
        }
    }
    unit(sum / (n * n))
}

// ---------------------------------------------------------------------------------------------
// Planes of f64
// ---------------------------------------------------------------------------------------------

/// The number of threads a plane operation uses.
fn workers(pixels: usize) -> usize {
    if pixels < 1 << 14 {
        1
    } else {
        thread::available_parallelism().map_or(1, usize::from)
    }
}

/// Fills `out` (`rows` rows of `width`) by calling `f(row, row_slice)` for every row, on several threads for a
/// large plane. Every row is computed from its own inputs, so the result does not depend on the thread count.
fn par_rows<F>(out: &mut [f64], width: usize, f: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    if width == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / width;
    let n = workers(out.len()).min(rows);
    if n <= 1 {
        for (y, row) in out.chunks_mut(width).enumerate() {
            f(y, row);
        }
        return;
    }
    let per = rows.div_ceil(n);
    thread::scope(|scope| {
        for (c, chunk) in out.chunks_mut(per * width).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (i, row) in chunk.chunks_mut(width).enumerate() {
                    f(c * per + i, row);
                }
            });
        }
    });
}

/// How a window is folded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fold {
    /// The mean over the samples inside the frame.
    Mean,
    /// The largest.
    Max,
    /// The smallest.
    Min,
}

impl Fold {
    /// What a missing half of a window contributes.
    fn identity(self) -> f64 {
        match self {
            Fold::Mean => 0.0,
            Fold::Max => f64::NEG_INFINITY,
            Fold::Min => f64::INFINITY,
        }
    }

    /// Two partial results joined: the sum (which is exact against the identity 0), the larger or the smaller.
    fn join(self, a: f64, b: f64) -> f64 {
        match self {
            Fold::Mean => a + b,
            Fold::Max => a.max(b),
            Fold::Min => a.min(b),
        }
    }
}

/// One line of a window fold: `dst[i]` is the fold of `[i - r, i + r]` (absolute: the line's first sample is at
/// `a0`) over the samples that lie inside `f0..f1` (the frame) and inside the line, as a *blocked* fold.
///
/// The line is cut into blocks of `w = 2 r + 1` samples that start at absolute multiples of `w`. A window of
/// `w` samples is either one block or the end of one block and the start of the next, so it is a *suffix* of the
/// first and a *prefix* of the second (van Herk 1992, Gil and Werman 1993): `suffix[i - r] + prefix[i + r]`.
/// Both are scans that restart at a block boundary, in a fixed order (prefixes upwards, suffixes downwards), so a
/// window's value is made from the samples of the window only and in an order that depends on their absolute
/// positions alone: it does not depend on where the line was cut, which is what makes a tile equal the frame bit
/// for bit (`docs/17` D4). It costs three operations a sample at any radius. The mean divides by the number of
/// samples that were there.
fn fold_line(line: &[f64], a0: i64, f0: i64, f1: i64, r: usize, fold: Fold, dst: &mut [f64]) {
    dst.fill(0.0);
    let lo_av = a0.max(f0);
    let hi_av = (a0 + line.len() as i64).min(f1);
    if lo_av >= hi_av {
        return;
    }
    let w = 2 * r as i64 + 1;
    let (il, ih) = ((lo_av - a0) as usize, (hi_av - a0) as usize);
    let mut prefix = vec![0.0; ih];
    let mut suffix = vec![0.0; ih];
    for k in il..ih {
        prefix[k] = if k == il || (a0 + k as i64).rem_euclid(w) == 0 {
            line[k]
        } else {
            fold.join(prefix[k - 1], line[k])
        };
    }
    for k in (il..ih).rev() {
        suffix[k] = if k == ih - 1 || (a0 + k as i64 + 1).rem_euclid(w) == 0 {
            line[k]
        } else {
            fold.join(suffix[k + 1], line[k])
        };
    }
    let (block_lo, block_hi) = (lo_av.div_euclid(w), (hi_av - 1).div_euclid(w));
    let id = fold.identity();
    for (k, d) in dst.iter_mut().enumerate().take(ih).skip(il) {
        let abs = a0 + k as i64;
        let (lo_w, hi_w) = (abs - r as i64, abs + r as i64);
        // The window's end in its second block; past the line it is the line's last prefix when that is the
        // same block, and nothing otherwise.
        let upper = if hi_w < hi_av {
            prefix[(hi_w - a0) as usize]
        } else if hi_w.div_euclid(w) == block_hi {
            prefix[ih - 1]
        } else {
            id
        };
        let total = if lo_w.rem_euclid(w) == 0 {
            // The window is one whole block.
            upper
        } else {
            let lower = if lo_w >= lo_av {
                suffix[(lo_w - a0) as usize]
            } else if lo_w.div_euclid(w) == block_lo {
                suffix[il]
            } else {
                id
            };
            fold.join(lower, upper)
        };
        *d = if fold == Fold::Mean {
            total / ((hi_w.min(hi_av - 1) - lo_w.max(lo_av)) + 1) as f64
        } else {
            total
        };
    }
}

/// The fold of every row of a plane `width` wide whose first column is the absolute column `a0`.
fn fold_rows(
    src: &[f64],
    width: usize,
    a0: i64,
    f0: i64,
    f1: i64,
    r: usize,
    fold: Fold,
) -> Vec<f64> {
    let mut out = vec![0.0; src.len()];
    par_rows(&mut out, width, |y, dst| {
        fold_line(&src[y * width..(y + 1) * width], a0, f0, f1, r, fold, dst);
    });
    out
}

/// A plane of `w x h` turned over its diagonal (`h x w`).
fn transpose(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; src.len()];
    par_rows(&mut out, h, |x, row| {
        for (y, d) in row.iter_mut().enumerate() {
            *d = src[y * w + x];
        }
    });
    out
}

/// One separable half of a window fold over a plane of area `g` inside `frame`: along x (`horizontal`) or y.
fn fold_axis(
    src: &[f64],
    g: &Area,
    frame: &Area,
    r: usize,
    horizontal: bool,
    fold: Fold,
) -> Vec<f64> {
    if g.is_empty() {
        return vec![0.0; g.w * g.h];
    }
    if horizontal {
        fold_rows(src, g.w, g.x, frame.x, frame.x1(), r, fold)
    } else {
        let t = transpose(src, g.w, g.h);
        transpose(
            &fold_rows(&t, g.h, g.y, frame.y, frame.y1(), r, fold),
            g.h,
            g.w,
        )
    }
}

/// The mean of the square window of half-side `r` around every pixel (a horizontal pass, then a vertical one),
/// over the samples inside `frame`.
#[must_use]
pub fn box_mean(src: &[f64], g: &Area, frame: &Area, r: usize) -> Vec<f64> {
    let h = fold_axis(src, g, frame, r, true, Fold::Mean);
    fold_axis(&h, g, frame, r, false, Fold::Mean)
}

/// Grow (`radius > 0`) or erode (`radius < 0`): the largest or smallest value of the square window of
/// half-side `|radius|` around every pixel, over the samples inside `frame`. A radius of 0 returns the plane.
#[must_use]
pub fn grow(src: &[f64], g: &Area, frame: &Area, radius: i64) -> Vec<f64> {
    if radius == 0 {
        return src.to_vec();
    }
    let fold = if radius > 0 { Fold::Max } else { Fold::Min };
    let r = radius.unsigned_abs() as usize;
    let h = fold_axis(src, g, frame, r, true, fold);
    fold_axis(&h, g, frame, r, false, fold)
}

/// Feather: three cascaded box means of the half-widths `boxes` (each a horizontal and a vertical pass).
#[must_use]
pub fn feather(src: &[f64], g: &Area, frame: &Area, boxes: [usize; 3]) -> Vec<f64> {
    let mut plane = src.to_vec();
    for r in boxes {
        plane = box_mean(&plane, g, frame, r);
    }
    plane
}

/// The cell `cell` (absolute, a multiple of `step` pixels wide) as a range of absolute pixels cut to `lo..hi`.
fn cell_pixels(cell: i64, step: usize, lo: i64, hi: i64) -> std::ops::Range<i64> {
    let s = step as i64;
    (cell * s).max(lo)..((cell + 1) * s).min(hi)
}

/// `floor(a / b)` for a positive `b`.
fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// `ceil(a / b)` for a positive `b`.
fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// The guided filter (He, Sun and Tang, 2010) of `p` with the guide `guide`, both over the area `g` inside
/// `frame`. Above [`GUIDED_SUBSAMPLE_SAMPLES`] pixels the window is the Fast Guided Filter's: the guide and `p`
/// are averaged over `step x step` blocks aligned at absolute multiples of `step`, `a` and `b` are found on that
/// grid with a window of `samples` cells, and bilinearly upsampled to the pixel before `q = a G + b`.
#[must_use]
pub fn guided(p: &[f64], guide: &[f64], g: &Area, frame: &Area, f: &Guided) -> Vec<f64> {
    if f.step == 1 {
        let (a, b) = guided_coefficients(p, guide, g, frame, f.samples, f.eps);
        let (abar, bbar) = (
            box_mean(&a, g, frame, f.samples),
            box_mean(&b, g, frame, f.samples),
        );
        return abar
            .iter()
            .zip(&bbar)
            .zip(guide)
            .map(|((a, b), gv)| unit(a * gv + b))
            .collect();
    }
    let s = f.step as i64;
    // The coarse grid: cells of `step` pixels at absolute multiples, covering the area and the frame.
    let cg = Area {
        x: floor_div(g.x, s),
        y: floor_div(g.y, s),
        w: (ceil_div(g.x1(), s) - floor_div(g.x, s)) as usize,
        h: (ceil_div(g.y1(), s) - floor_div(g.y, s)) as usize,
    };
    let cf = Area {
        x: floor_div(frame.x, s),
        y: floor_div(frame.y, s),
        w: (ceil_div(frame.x1(), s) - floor_div(frame.x, s)) as usize,
        h: (ceil_div(frame.y1(), s) - floor_div(frame.y, s)) as usize,
    };
    let block = |src: &[f64]| {
        let mut out = vec![0.0; cg.w * cg.h];
        par_rows(&mut out, cg.w, |j, row| {
            let ys = cell_pixels(
                cg.y + j as i64,
                f.step,
                g.y.max(frame.y),
                g.y1().min(frame.y1()),
            );
            for (i, d) in row.iter_mut().enumerate() {
                let xs = cell_pixels(
                    cg.x + i as i64,
                    f.step,
                    g.x.max(frame.x),
                    g.x1().min(frame.x1()),
                );
                let mut sum = 0.0;
                let mut n = 0usize;
                for y in ys.clone() {
                    for x in xs.clone() {
                        sum += src[(y - g.y) as usize * g.w + (x - g.x) as usize];
                        n += 1;
                    }
                }
                *d = if n == 0 { 0.0 } else { sum / n as f64 };
            }
        });
        out
    };
    let (gc, pc) = (block(guide), block(p));
    let (a, b) = guided_coefficients(&pc, &gc, &cg, &cf, f.samples, f.eps);
    let (abar, bbar) = (
        box_mean(&a, &cg, &cf, f.samples),
        box_mean(&b, &cg, &cf, f.samples),
    );
    let mut out = vec![0.0; g.w * g.h];
    par_rows(&mut out, g.w, |y, row| {
        let ay = g.y + y as i64;
        let vy = (ay as f64 + 0.5) / f.step as f64 - 0.5;
        let (j0f, ty) = (vy.floor(), vy - vy.floor());
        let at_y = |j: i64| (j.clamp(cf.y, cf.y1() - 1).clamp(cg.y, cg.y1() - 1) - cg.y) as usize;
        let (r0, r1) = (at_y(j0f as i64), at_y(j0f as i64 + 1));
        for (x, d) in row.iter_mut().enumerate() {
            let ax = g.x + x as i64;
            let ux = (ax as f64 + 0.5) / f.step as f64 - 0.5;
            let (i0f, tx) = (ux.floor(), ux - ux.floor());
            let at_x =
                |i: i64| (i.clamp(cf.x, cf.x1() - 1).clamp(cg.x, cg.x1() - 1) - cg.x) as usize;
            let (c0, c1) = (at_x(i0f as i64), at_x(i0f as i64 + 1));
            let lerp = |plane: &[f64]| {
                let top = plane[r0 * cg.w + c0] * (1.0 - tx) + plane[r0 * cg.w + c1] * tx;
                let bottom = plane[r1 * cg.w + c0] * (1.0 - tx) + plane[r1 * cg.w + c1] * tx;
                top * (1.0 - ty) + bottom * ty
            };
            *d = unit(lerp(&abar) * guide[y * g.w + x] + lerp(&bbar));
        }
    });
    out
}

/// `a` and `b` of the guided filter at every pixel of `g`: `a = cov / (var + eps)`, `b = mean_p - a mean_g`, from
/// the window means of the guide, `p`, the guide squared and the guide times `p`.
fn guided_coefficients(
    p: &[f64],
    guide: &[f64],
    g: &Area,
    frame: &Area,
    r: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let gg: Vec<f64> = guide.iter().map(|v| v * v).collect();
    let gp: Vec<f64> = guide.iter().zip(p).map(|(a, b)| a * b).collect();
    let (mg, mp) = (box_mean(guide, g, frame, r), box_mean(p, g, frame, r));
    let (mgg, mgp) = (box_mean(&gg, g, frame, r), box_mean(&gp, g, frame, r));
    let mut a = vec![0.0; p.len()];
    let mut b = vec![0.0; p.len()];
    for i in 0..p.len() {
        let var = mgg[i] - mg[i] * mg[i];
        let cov = mgp[i] - mg[i] * mp[i];
        a[i] = cov / (var + eps);
        b[i] = mp[i] - a[i] * mg[i];
    }
    (a, b)
}

// ---------------------------------------------------------------------------------------------
// A row's plane
// ---------------------------------------------------------------------------------------------

/// The places of a window's pixels in the photo frame: `(x, y)` relative to the window gives the fractions of
/// the photo, or `None` for a pixel outside the photo (the empty corner of a straightened photo).
pub type PositionAt<'a> = &'a (dyn Fn(usize, usize) -> Option<[f64; 2]> + Sync);

/// What a render knows for the planes of one window.
pub struct PlaneJob<'a> {
    /// The photo's aspect and the size of one output pixel in units of its long edge.
    pub photo: PhotoFrame,
    /// The window the planes are computed over: the tile grown by the halo, cut to the frame.
    pub grown: Area,
    /// The whole output image: window sums are clipped to it.
    pub frame: Area,
    /// The part of `grown` the planes are returned for (the tile).
    pub window: Area,
    /// Where each pixel of `grown` is in the photo.
    pub position: PositionAt<'a>,
    /// The Input's luma at every pixel of `grown`, row-major, in [0, 1]; present when a row refines its edges.
    pub guide: Option<&'a [f64]>,
}

impl PlaneJob<'_> {
    /// `u_px`: the output pixels along the photo's long edge.
    #[must_use]
    pub fn u_px(&self) -> f64 {
        1.0 / self.photo.pixel
    }
}

impl RasterRow {
    /// The row's plane over `job.window` as u16 codes (row-major), through the five steps.
    ///
    /// # Panics
    ///
    /// When `job.window` is not inside `job.grown`: a mistake of the caller. A row that refines its edges in a job
    /// that has no guide skips that step.
    #[must_use]
    pub fn plane(&self, job: &PlaneJob<'_>) -> Vec<u16> {
        self.plane_among(job, 0)
    }

    /// [`RasterRow::plane`] in a set of `live` rows ([`RasterSource::plane_among`]).
    ///
    /// # Panics
    ///
    /// As [`RasterRow::plane`].
    #[must_use]
    pub fn plane_among(&self, job: &PlaneJob<'_>, live: usize) -> Vec<u16> {
        assert!(
            job.grown.contains(&job.window),
            "the window of a raster plane is inside the area it is computed over"
        );
        let steps = RasterSteps::of(self.grow, self.feather, self.refine, job.u_px());
        let src = self.source.plane_among(job.photo.aspect, live);
        let g = job.grown;
        // Step 2: the resample (a pixel outside the photo is 0).
        let taps = resample_taps(src.width.max(src.height), job.u_px());
        let shrink = src.width.max(src.height) as f64 / job.u_px();
        let mut plane = vec![0.0; g.w * g.h];
        par_rows(&mut plane, g.w, |y, row| {
            for (x, d) in row.iter_mut().enumerate() {
                *d = (job.position)(x, y).map_or(0.0, |pos| resample_at(&src, pos, taps, shrink));
            }
        });
        // Steps 3 to 5.
        if steps.grow != 0 {
            plane = grow(&plane, &g, &job.frame, steps.grow);
        }
        if let Some(boxes) = steps.feather {
            plane = feather(&plane, &g, &job.frame, boxes);
        }
        // A row that refines its edges needs the guide (`RasterSet::needs_guide` says so, and the caller makes it for
        // exactly those renders); a job without one skips the step.
        if let (Some(gf), Some(guide)) = (&steps.guided, job.guide) {
            plane = guided(&plane, guide, &g, &job.frame, gf);
        }
        // The window, quantised.
        let w = job.window;
        let mut out = Vec::with_capacity(w.w * w.h);
        for y in 0..w.h {
            let row = (w.y - g.y) as usize + y;
            for x in 0..w.w {
                let col = (w.x - g.x) as usize + x;
                out.push(quantise(plane[row * g.w + col]));
            }
        }
        out
    }
}

/// The raster planes of a window on the CPU: one u16 plane for every row of the chain, each `width x height`
/// (the window), row-major. The CPU chain reads it through [`RasterPlanes`]; the device uploads the same codes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuPlanes {
    /// The window's width.
    pub width: usize,
    /// The window's height.
    pub height: usize,
    /// The planes, in the order of [`RasterSet::rows`].
    pub planes: Vec<Vec<u16>>,
}

impl CpuPlanes {
    /// These planes (of a window) placed in an `extent_w x extent_h` extent with the window's first pixel at `at`,
    /// 0 around them: the planes of a view whose frame has paper in it (the photo ends inside the picture).
    ///
    /// # Panics
    ///
    /// When the window does not fit in the extent at `at`: a mistake of the caller.
    #[must_use]
    pub fn placed(&self, at: (usize, usize), extent_w: usize, extent_h: usize) -> CpuPlanes {
        assert!(
            at.0 + self.width <= extent_w && at.1 + self.height <= extent_h,
            "a window of planes lies inside the extent they are placed in"
        );
        if (self.width, self.height) == (extent_w, extent_h) {
            return self.clone();
        }
        let planes = self
            .planes
            .iter()
            .map(|p| {
                let mut out = vec![0u16; extent_w * extent_h];
                for y in 0..self.height {
                    let dst = (at.1 + y) * extent_w + at.0;
                    out[dst..dst + self.width]
                        .copy_from_slice(&p[y * self.width..(y + 1) * self.width]);
                }
                out
            })
            .collect();
        CpuPlanes {
            width: extent_w,
            height: extent_h,
            planes,
        }
    }
}

impl RasterPlanes for CpuPlanes {
    fn value(&self, plane: usize, x: usize, y: usize) -> f64 {
        match self.planes.get(plane) {
            Some(p) if x < self.width && y < self.height => {
                f64::from(p[y * self.width + x]) / PLANE_ONE
            }
            _ => 0.0,
        }
    }
}

impl RasterSet {
    /// Every row's plane over `job.window`.
    #[must_use]
    pub fn planes(&self, job: &PlaneJob<'_>) -> CpuPlanes {
        CpuPlanes {
            width: job.window.w,
            height: job.window.h,
            planes: self
                .rows
                .iter()
                .map(|r| r.plane_among(job, self.rows.len()))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Brush
// ---------------------------------------------------------------------------------------------

/// One stroke of a Brush row (`docs/algorithms/06` section 5.5).
#[derive(Debug, Clone, PartialEq)]
pub struct Stroke {
    /// The brush's diameter as a fraction of the photo's long edge.
    pub size: f64,
    /// The hardness, 0 to 1.
    pub hardness: f64,
    /// An eraser stroke.
    pub erase: bool,
    /// The path in the photo frame, fractions of its width and height.
    pub points: Vec<[f64; 2]>,
}

/// The strokes of a Brush row, in the order they were drawn, with the key that names them.
#[derive(Clone, PartialEq)]
pub struct BrushStrokes {
    /// What identifies the strokes (a hash of their content): the cache key.
    pub key: u64,
    /// The strokes.
    pub strokes: Vec<Stroke>,
}

/// The stamps spaced along a stroke are at `0.15 D` (a fraction of the brush's diameter) along its path.
pub const BRUSH_SPACING: f64 = 0.15;
/// The closest two stamps of a stroke are, in plane pixels.
pub const BRUSH_MIN_SPACING_PX: f64 = 0.5;
/// The most pixels all the stamps of a Brush may touch (a file with absurd strokes cannot hang the render):
/// the plane stops being drawn when the budget is spent, the same way on every machine.
pub const BRUSH_WORK_LIMIT: u64 = 4_000_000_000;

impl std::fmt::Debug for BrushStrokes {
    /// The key and the counts, never the points (a file may hold a hundred thousand).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrushStrokes")
            .field("key", &self.key)
            .field("strokes", &self.strokes.len())
            .field(
                "points",
                &self.strokes.iter().map(|s| s.points.len()).sum::<usize>(),
            )
            .finish()
    }
}

/// The key the strokes `(size, hardness, erase, points)` hash to (FNV-1a over the bit patterns: stable, cheap, and
/// only a cache key). [`BrushStrokes::new`] uses it, and a caller that only needs the key (the render graph's cache
/// key) can ask for it without building the strokes.
pub fn strokes_key<'a>(strokes: impl IntoIterator<Item = (f64, f64, bool, &'a [[f64; 2]])>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bits: u64| {
        for byte in bits.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let mut count = 0u64;
    let mut body: Vec<(f64, f64, bool, &[[f64; 2]])> = Vec::new();
    for s in strokes {
        count += 1;
        body.push(s);
    }
    feed(count);
    for (size, hardness, erase, points) in body {
        feed(size.to_bits());
        feed(hardness.to_bits());
        feed(u64::from(erase));
        feed(points.len() as u64);
        for p in points {
            feed(p[0].to_bits());
            feed(p[1].to_bits());
        }
    }
    h
}

impl BrushStrokes {
    /// The strokes and the key their content hashes to ([`strokes_key`]).
    #[must_use]
    pub fn new(strokes: Vec<Stroke>) -> Self {
        let key = strokes_key(
            strokes
                .iter()
                .map(|s| (s.size, s.hardness, s.erase, s.points.as_slice())),
        );
        Self { key, strokes }
    }
}

/// The weight of a stamp of radius `radius` and inner radius `inner` at distance `d` from its centre: 1 inside
/// the inner radius, a smoothstep down to 0 at the radius.
fn stamp_weight(d: f64, radius: f64, inner: f64) -> f32 {
    if d <= inner {
        return 1.0;
    }
    if d >= radius {
        return 0.0;
    }
    let t = ((d - inner) / (radius - inner)).clamp(0.0, 1.0);
    (1.0 - t * t * (3.0 - 2.0 * t)) as f32
}

/// Draws the strokes of a Brush row into a plane of [`BRUSH_PLANE_PX`] pixels on the long edge, with the
/// proportions of a photo of this `aspect` (`(W, H) / L`, one of them 1), in list order, in f32:
///
/// * a stroke's diameter is `D = size` of the long edge, `D_px = D BRUSH_PLANE_PX` plane pixels, radius
///   `R = D_px / 2`;
/// * the path is resampled by arc length at `max(0.15 D_px, 0.5)` pixels: a stamp at the first point and at
///   every `spacing` along the polyline, and one at the last point; a single point is one stamp;
/// * a stamp's ramp is `max((1 - hardness) R, 1)` pixels wide, ending at the radius; its weight is 1 inside
///   `R - ramp` and falls as a smoothstep to 0 at `R` (so a hard brush has a ramp of one plane pixel);
/// * a paint stamp makes `S = 1 - (1 - S)(1 - w)` and an erase stamp `S = S (1 - w)`.
///
/// The result is the plane's codes, `round(S 65535)`.
#[must_use]
pub fn draw_brush(brush: &BrushStrokes, aspect: [f64; 2]) -> SourcePlane {
    let long = BRUSH_PLANE_PX;
    let width = ((aspect[0].clamp(0.0, 1.0) * long as f64).round() as usize).clamp(1, long);
    let height = ((aspect[1].clamp(0.0, 1.0) * long as f64).round() as usize).clamp(1, long);
    let mut s = vec![0.0f32; width * height];
    let mut work = 0u64;
    'strokes: for stroke in &brush.strokes {
        let d_px = unit(stroke.size) * long as f64;
        let radius = (0.5 * d_px).max(0.5);
        let ramp = ((1.0 - unit(stroke.hardness)) * radius)
            .max(1.0)
            .min(radius);
        let inner = radius - ramp;
        let spacing = (BRUSH_SPACING * d_px).max(BRUSH_MIN_SPACING_PX);
        let pts: Vec<[f64; 2]> = stroke
            .points
            .iter()
            .map(|p| [unit(p[0]) * width as f64, unit(p[1]) * height as f64])
            .collect();
        let mut stamps: Vec<[f64; 2]> = Vec::new();
        match pts.as_slice() {
            [] => {}
            [only] => stamps.push(*only),
            _ => {
                // Stamps at arc lengths 0, spacing, 2 spacing, ... and at the end of the path.
                stamps.push(pts[0]);
                let mut carried = 0.0;
                let mut last = pts[0];
                for p in &pts[1..] {
                    let len = (p[0] - last[0]).hypot(p[1] - last[1]);
                    if len > 0.0 {
                        let mut at = spacing - carried;
                        while at <= len {
                            let t = at / len;
                            stamps.push([
                                last[0] + (p[0] - last[0]) * t,
                                last[1] + (p[1] - last[1]) * t,
                            ]);
                            at += spacing;
                        }
                        carried = len - (at - spacing);
                        last = *p;
                    }
                }
                if carried > 1e-9 {
                    stamps.push(last);
                }
            }
        }
        let side = (2.0 * radius).ceil() as u64 + 2;
        for c in stamps {
            work += side * side;
            if work > BRUSH_WORK_LIMIT {
                break 'strokes;
            }
            let x0 = ((c[0] - radius).floor() as i64).max(0);
            let x1 = ((c[0] + radius).ceil() as i64 + 1).min(width as i64);
            let y0 = ((c[1] - radius).floor() as i64).max(0);
            let y1 = ((c[1] + radius).ceil() as i64 + 1).min(height as i64);
            for y in y0..y1 {
                let dy = y as f64 + 0.5 - c[1];
                for x in x0..x1 {
                    let dx = x as f64 + 0.5 - c[0];
                    let w = stamp_weight(dx.hypot(dy), radius, inner);
                    if w > 0.0 {
                        let v = &mut s[y as usize * width + x as usize];
                        *v = if stroke.erase {
                            *v * (1.0 - w)
                        } else {
                            1.0 - (1.0 - *v) * (1.0 - w)
                        };
                    }
                }
            }
        }
    }
    SourcePlane {
        key: brush.key,
        width,
        height,
        data: s.iter().map(|v| quantise(f64::from(*v))).collect(),
    }
}

/// How many Brush planes the cache keeps at least (8 MiB each at 2048 px).
const BRUSH_CACHE_PLANES: usize = 8;

/// The most bytes the Brush cache holds beyond its [`BRUSH_CACHE_PLANES`] (a plan with many live Brush rows keeps
/// a plane for each, as long as they fit this).
const BRUSH_CACHE_BYTES: usize = 256 << 20;

type BrushCache = Mutex<VecDeque<((u64, usize, usize), Arc<SourcePlane>)>>;

fn brush_cache() -> &'static BrushCache {
    static CACHE: OnceLock<BrushCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// The plane of a Brush row for a photo of this aspect, from the small in-memory cache (never on disk) when
/// the same strokes were drawn for the same proportions lately, else drawn now ([`draw_brush`]).
#[must_use]
pub fn brush_plane(brush: &BrushStrokes, aspect: [f64; 2]) -> Arc<SourcePlane> {
    brush_plane_among(brush, aspect, 0)
}

/// [`brush_plane`] for a render with `live` raster rows: the cache keeps at least `max(8, live)` planes, within
/// [`BRUSH_CACHE_BYTES`] beyond the first eight.
#[must_use]
pub fn brush_plane_among(brush: &BrushStrokes, aspect: [f64; 2], live: usize) -> Arc<SourcePlane> {
    let long = BRUSH_PLANE_PX;
    let width = ((aspect[0].clamp(0.0, 1.0) * long as f64).round() as usize).clamp(1, long);
    let height = ((aspect[1].clamp(0.0, 1.0) * long as f64).round() as usize).clamp(1, long);
    let key = (brush.key, width, height);
    {
        let mut cache = brush_cache().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
            let hit = cache.remove(i);
            if let Some((k, plane)) = hit {
                let out = Arc::clone(&plane);
                cache.push_back((k, plane));
                return out;
            }
        }
    }
    let plane = Arc::new(draw_brush(brush, aspect));
    let mut cache = brush_cache().lock().unwrap_or_else(PoisonError::into_inner);
    cache.push_back((key, Arc::clone(&plane)));
    let keep = BRUSH_CACHE_PLANES.max(live);
    let mut bytes: usize = cache.iter().map(|(_, p)| p.data.len() * 2).sum();
    while cache.len() > keep || (bytes > BRUSH_CACHE_BYTES && cache.len() > BRUSH_CACHE_PLANES) {
        if let Some((_, old)) = cache.pop_front() {
            bytes -= old.data.len() * 2;
        }
    }
    plane
}

/// The planes the Brush cache holds now (for tests).
#[must_use]
pub fn brush_cache_len() -> usize {
    brush_cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len()
}

// ---------------------------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------------------------

/// The narrowest ramp of Background's threshold, in probability (`docs/algorithms/06` section 5.4).
pub const BLUR_RAMP_MIN: f64 = 0.05;

/// The blur map of a photo as Background reads it: `P`, the probability that a cell is out of focus
/// (`P(b >= 2 b_min)`, `docs/21` section 7), over the **photo frame** (the photo as the camera oriented it), one
/// f32 in [0, 1] for each cell, row-major. A cell is 2 x 2 pixels of the 1024 px lens plane, so the map is about
/// 512 px on the long edge.
#[derive(Clone, PartialEq)]
pub struct BlurMap {
    /// What the map was made from: equal keys are the same cells. The core gives every map it makes a key of its
    /// own, so a new map (another photo, other raw parameters) never reads the old one's planes.
    pub key: u64,
    /// Cells across.
    pub width: usize,
    /// Cells down.
    pub height: usize,
    /// `P` per cell.
    pub p: Vec<f32>,
}

impl std::fmt::Debug for BlurMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlurMap")
            .field("key", &self.key)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// What makes a Background row's `S0`: the photo's blur map and the row's Level and Softness.
#[derive(Debug, Clone, PartialEq)]
pub struct BackgroundSource {
    /// The photo's blur map.
    pub map: Arc<BlurMap>,
    /// Level `lambda`, 0 to 1: where the map's decision falls (0.5 is `P = 0.5`).
    pub level: f64,
    /// Softness `sigma`, 0 to 1: the ramp's width in `P`.
    pub softness: f64,
}

impl BackgroundSource {
    /// What identifies the plane: the map and the two numbers (FNV-1a over their bits), never Feather, Grow or
    /// Refine edges, which act after the plane.
    #[must_use]
    pub fn key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for bits in [
            self.map.key,
            unit(self.level).to_bits(),
            unit(self.softness).to_bits(),
        ] {
            for byte in bits.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}

/// `S0` of a Background row from the cells' probabilities (`docs/algorithms/06` section 5.4):
///
/// `S0 = S((P - (1 - level) + w / 2) / w)`, `w = max(softness, BLUR_RAMP_MIN)`,
///
/// `S` the smoothstep, rounded once to a u16 code. Level 0.5 puts the edge at `P = 0.5`, the map's own decision;
/// a cell with `P = 0` (an evenly sharp photo) is 0 for any level below 1 and any softness.
#[must_use]
pub fn background_s0(p: &[f32], level: f64, softness: f64) -> Vec<u16> {
    let w = unit(softness).max(BLUR_RAMP_MIN);
    let edge = 1.0 - unit(level);
    p.iter()
        .map(|&v| {
            let t = (f64::from(v) - edge + 0.5 * w) / w;
            quantise(crate::mask::smooth_step(t))
        })
        .collect()
}

/// How many Background planes the cache keeps (about 350 KB each).
const BACKGROUND_CACHE_PLANES: usize = 16;

type BackgroundCache = Mutex<VecDeque<(u64, Arc<SourcePlane>)>>;

fn background_cache() -> &'static BackgroundCache {
    static CACHE: OnceLock<BackgroundCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// The plane of a Background row, from the small in-memory cache (never on disk) when the same map was cut with
/// the same Level and Softness lately, else made now ([`background_s0`]): moving Level or Softness makes a plane
/// from the cached map, and Feather, Grow and Refine edges, which are not in the key, never do.
#[must_use]
pub fn background_plane(src: &BackgroundSource) -> Arc<SourcePlane> {
    background_plane_among(src, 0)
}

/// [`background_plane`] for a render with `live` raster rows: the cache keeps at least `max(16, live)` planes.
#[must_use]
pub fn background_plane_among(src: &BackgroundSource, live: usize) -> Arc<SourcePlane> {
    let key = src.key();
    {
        let mut cache = background_cache()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = cache.iter().position(|(k, _)| *k == key)
            && let Some((k, plane)) = cache.remove(i)
        {
            let out = Arc::clone(&plane);
            cache.push_back((k, plane));
            return out;
        }
    }
    let plane = Arc::new(SourcePlane {
        key,
        width: src.map.width.max(1),
        height: src.map.height.max(1),
        data: if src.map.p.len() == src.map.width * src.map.height && !src.map.p.is_empty() {
            background_s0(&src.map.p, src.level, src.softness)
        } else {
            vec![0; src.map.width.max(1) * src.map.height.max(1)]
        },
    });
    let mut cache = background_cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    cache.push_back((key, Arc::clone(&plane)));
    while cache.len() > BACKGROUND_CACHE_PLANES.max(live) {
        cache.pop_front();
    }
    plane
}
