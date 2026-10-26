//! L2: field optics, the CPU reference (`docs/21` section 4, L2).
//!
//! # What a pixel does
//!
//! An output pixel at field position `(u, v)` (the pixel centre minus the frame centre, over the frame's
//! half-diagonal `H`) gathers its input with a kernel:
//!
//! * the **oval field radius** `rho' = sqrt((alpha x_a)^2 + y_a^2)` with `x_a`, `y_a` the field coordinates
//!   along and across the anamorphic axis and `alpha = 1 + kappa (q - 1)`; it selects the spread function
//!   tables (held at the last table beyond 1) and the width `w`, which is read by a monotone cubic through
//!   the nine radii and carried on beyond 1 along its end slope ([`FieldTables::half_width_at`]), and, over
//!   the scale `n` ([`FieldTables::radius_scale`]), the relative illumination `T`, a concave curve through
//!   the nine values carried straight on beyond 1 ([`FieldTables::illumination_at`]), so that no ray through
//!   the frame meets a corner or a plateau and the falloff of a flat field has no inflection;
//! * the **edge amount** `e' = e + (1 - e) kappa (q - 1)`: Edges, plus the attachment's own share, so that at
//!   Edges 0 the anamorphic terms act alone (`docs/21` section 4, L2);
//! * the **half-width** `omega = cap(e' (1 + S) w(rho')) H` pixels, with the side softness
//!   `S = a_s (q - 1) x_a^2` and [`soft_cap`] bending the width towards `w_max` without a corner;
//! * the **gain** `1 - e' (1 - T(rho' / n))`.
//!
//! The kernel is the projected channel table (`sum_j w_c(l_j) P_j`), blended linearly between the two
//! neighbouring radii, resampled area-weighted to `n x n` cells (`n = min(17, 2 ceil(omega) + 1)`), turned so
//! that table x points radially outwards and table y anticlockwise, and scaled to `+-omega`. Each cell is a
//! point mass at its centre, read from the input by bilinear interpolation. Because the channel tables have
//! their own centroids, the channels are shifted against each other: that is the lateral colour, of which the
//! projection keeps [`LATERAL_COLOR_SCALE`] of the traced offsets from green (red and blue tables are
//! translated, green stays). With the
//! attachment the axis part of each channel's centroid offset is multiplied by `1 + h (q - 1)`: after the
//! gather, channel `c` is read `h (q - 1)` times its centroid offset's axis part further along the axis, by
//! linear interpolation between neighbouring pixels (to first order the same as translating the kernel, and a
//! single gather instead of three). The gain is applied last.
//!
//! A kernel with `omega` below half a pixel is a copy: the pixel is the input pixel times the gain.
//!
//! # Wide kernels
//!
//! A kernel wider than the tap budget at one tap a pixel reads a 2x, 4x, ... box-reduced copy of the input (a
//! mip level whose blocks are aligned to absolute frame coordinates, so that any tiling reads the same texels)
//! until its cells are at most one texel apart. The input is replicated beyond the frame edge.
//!
//! # Determinism
//!
//! A pixel depends only on its absolute position, the plan and the input in its receptive field
//! ([`halo`]); sums run in a fixed order, and only `+ - * /`, `sqrt` and truncation are used.

use rayon::prelude::*;

use crate::consts::{
    COPY_BELOW_PX, EXTRA_CHANGE_FLOOR, EXTRA_CHANGE_RAMP, EXTRA_INFLOW_RAMP, EXTRA_KERNEL_RAMP,
    LATERAL_COLOR_SCALE, MAX_TAPS_SIDE, PSF_SIZE, PSF_WAVELENGTHS, RADII, SIDE_TABLE_RADIUS,
    T_OVERSHOOT, TAP_SPACING_MAX_PX, W_KNEE, W_MAX,
};
use crate::data::LensData;
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::{Axis, LensPlan};
use crate::simd::F4;
use crate::spectral::CameraResponse;

/// Number of table sizes `n = 3, 5, ..., MAX_TAPS_SIDE`.
pub(crate) const SIZES: usize = (MAX_TAPS_SIDE - 3) / 2 + 1;

/// The most mip levels a kernel can ask for (a safety bound; real kernels need 0 to 4).
const MAX_LEVEL: u32 = 12;

/// Knobs of the CPU reference that tests turn; production code uses [`Options::default`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    /// Largest spacing of the taps in texels (default [`TAP_SPACING_MAX_PX`]).
    pub spacing_max: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            spacing_max: TAP_SPACING_MAX_PX,
        }
    }
}

/// Bends a kernel half-width `x` (half-diagonals) towards [`W_MAX`] without a corner: `x` up to [`W_KNEE`],
/// then `K + (W - K) u / (1 + u)` with `u = (x - K) / (W - K)`. It is continuous with a continuous first
/// derivative at the knee (1 on both sides), strictly increasing, at most `x` and below `W_MAX`, so the
/// halo bounds taken from `min(x, W_MAX)` still hold. Only `+ - * /`.
pub(crate) fn soft_cap(x: f64) -> f64 {
    if x <= W_KNEE {
        x
    } else {
        let r = W_MAX - W_KNEE;
        let u = (x - W_KNEE) / r;
        W_KNEE + r * u / (1.0 + u)
    }
}

/// The values made monotone by pooling adjacent violators (equal weights): the closest non-decreasing (or,
/// with `increasing` false, non-increasing) sequence in the least-squares sense. A monotone input is
/// returned bit for bit.
fn pool_monotone(values: &[f64; RADII], increasing: bool) -> [f64; RADII] {
    let sign = if increasing { 1.0 } else { -1.0 };
    // Blocks of (sum, count), merged while a block's mean is below the one before it.
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(RADII);
    for v in values {
        blocks.push((sign * v, 1));
        while blocks.len() >= 2 {
            let (s1, n1) = blocks[blocks.len() - 1];
            let (s0, n0) = blocks[blocks.len() - 2];
            if s0 / n0 as f64 <= s1 / n1 as f64 {
                break;
            }
            blocks.truncate(blocks.len() - 2);
            blocks.push((s0 + s1, n0 + n1));
        }
    }
    let mut out = *values;
    let mut at = 0;
    for (sum, n) in blocks {
        for slot in &mut out[at..at + n] {
            *slot = if n == 1 {
                *slot
            } else {
                sign * (sum / n as f64)
            };
        }
        at += n;
    }
    out
}

/// The slopes of the monotone cubic through `(x, y)` (Fritsch and Butland's rule, the end slopes of a
/// three-point formula held to the data's direction): the cubic Hermite curve with these slopes is `C^1` and
/// monotone wherever `y` is.
fn monotone_slopes(x: &[f64; RADII], y: &[f64; RADII]) -> [f64; RADII] {
    let mut h = [0.0; RADII - 1];
    let mut m = [0.0; RADII - 1];
    for k in 0..RADII - 1 {
        h[k] = x[k + 1] - x[k];
        m[k] = (y[k + 1] - y[k]) / h[k];
    }
    let mut d = [0.0; RADII];
    for k in 1..RADII - 1 {
        if m[k - 1] * m[k] > 0.0 {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k]);
        }
    }
    let end = |h0: f64, h1: f64, m0: f64, m1: f64| -> f64 {
        let slope = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        if slope * m0 <= 0.0 {
            0.0
        } else if m0 * m1 <= 0.0 && slope.abs() > 3.0 * m0.abs() {
            3.0 * m0
        } else {
            slope
        }
    };
    d[0] = end(h[0], h[1], m[0], m[1]);
    d[RADII - 1] = end(h[RADII - 2], h[RADII - 3], m[RADII - 2], m[RADII - 3]);
    d
}

/// The cubic Hermite curve through `(0, y0)` and `(1, y1)` with slopes `d0 h` and `d1 h` (`h` the interval
/// in the curve's own units), at `t`.
fn hermite(y0: f64, y1: f64, d0: f64, d1: f64, h: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * d0
        + (3.0 * t2 - 2.0 * t3) * y1
        + (t3 - t2) * h * d1
}

/// The closest concave sequence to `values` at `radii` (in the least-squares sense on the secants): adjacent
/// intervals whose secant slopes rise are pooled into one straight run between the same end values, so a
/// sequence that is concave already is returned bit for bit.
fn pool_concave(radii: &[f64; RADII], values: &[f64; RADII]) -> [f64; RADII] {
    // Blocks of (rise, run, intervals); concave means the secants fall from block to block.
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(RADII - 1);
    for k in 0..RADII - 1 {
        blocks.push((values[k + 1] - values[k], radii[k + 1] - radii[k], 1));
        while blocks.len() >= 2 {
            let (y1, x1, n1) = blocks[blocks.len() - 1];
            let (y0, x0, n0) = blocks[blocks.len() - 2];
            if y0 / x0 >= y1 / x1 {
                break;
            }
            blocks.truncate(blocks.len() - 2);
            blocks.push((y0 + y1, x0 + x1, n0 + n1));
        }
    }
    let mut out = *values;
    let mut at = 0;
    for (rise, run, n) in blocks {
        let secant = rise / run;
        for i in 1..n {
            out[at + i] = values[at] + secant * (radii[at + i] - radii[at]);
        }
        at += n;
    }
    out
}

/// The slopes for the concave curve through `(x, y)` with `y` concave: [`monotone_slopes`] held between the
/// secants either side of each node (the later one at most, the earlier one at least), the first slope at most
/// 0 and the last at most the last secant, so that [`concave_piece`] has a falling slope on every interval.
/// An interval that has the same secant as a neighbour is a straight run (pooling makes such runs): its two
/// slopes are the secant, since a curve whose slope starts above the secant and ends at it cannot have the
/// interval's rise without a corner.
fn concave_slopes(x: &[f64; RADII], y: &[f64; RADII]) -> [f64; RADII] {
    let mut d = monotone_slopes(x, y);
    let mut secant = [0.0; RADII - 1];
    for (k, m) in secant.iter_mut().enumerate() {
        *m = (y[k + 1] - y[k]) / (x[k + 1] - x[k]);
    }
    for (k, slope) in d.iter_mut().enumerate() {
        let lo = if k < RADII - 1 {
            secant[k]
        } else {
            f64::NEG_INFINITY
        };
        let hi = if k > 0 { secant[k - 1] } else { 0.0 };
        let hi = if k == RADII - 1 { hi.min(0.0) } else { hi };
        *slope = slope.clamp(lo.min(hi), hi);
    }
    // (Equal to the rounding: a pooled run's secants are computed from its rounded node values.)
    let same = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + a.abs());
    for k in 0..RADII - 1 {
        let before = k > 0 && same(secant[k - 1], secant[k]);
        let after = k < RADII - 2 && same(secant[k], secant[k + 1]);
        if before || after {
            d[k] = secant[k];
            d[k + 1] = secant[k];
        }
    }
    d
}

/// The concave `C^1` curve through `(0, y0)` and `(h, y1)` whose slope falls from `d0` to `d1`
/// (`d0 >= (y1 - y0) / h >= d1`), at `x = t h`: a quadratic whose slope falls linearly from `d0` to `d1`
/// when the secant is the mean of the two slopes, else a straight run at the slope of one end and a quadratic
/// at the other (Schumaker's single extra knot, at the place that keeps the area). Its second derivative is
/// never above 0, so a gain made of it, along a row of a flat field, cannot bend back.
fn concave_piece(y0: f64, y1: f64, d0: f64, d1: f64, h: f64, t: f64) -> f64 {
    let x = t * h;
    let dd = d0 - d1;
    if dd <= 1e-9 {
        return y0 + d0 * x;
    }
    let m = (y1 - y0) / h;
    if m <= 0.5 * (d0 + d1) {
        // The slope falls to d1 over the first h1, then stays there.
        let h1 = (2.0 * h * (m - d1) / dd).clamp(0.0, h);
        if x <= h1 {
            if h1 <= 0.0 {
                return y0;
            }
            return y0 + d0 * x - dd * x * x / (2.0 * h1);
        }
        y0 + 0.5 * h1 * (d0 + d1) + d1 * (x - h1)
    } else {
        // The slope stays at d0 for the first h - h2, then falls to d1 over the last h2.
        let h2 = (2.0 * h * (d0 - m) / dd).clamp(0.0, h);
        let h1 = h - h2;
        if x <= h1 {
            return y0 + d0 * x;
        }
        let u = x - h1;
        y0 + d0 * x - dd * u * u / (2.0 * h2)
    }
}

/// The lens's spread functions projected on a camera's three channels, at every table size.
#[derive(Clone, Debug)]
pub(crate) struct FieldTables {
    radii: [f64; RADII],
    /// `w(rho)` at the radii, made non-decreasing (the traced Detuned dips once).
    half_width: [f64; RADII],
    /// `T(rho)` at the radii (the data are non-increasing).
    illumination: [f64; RADII],
    /// The slopes of the monotone cubics through `half_width` and `illumination`.
    half_width_slope: [f64; RADII],
    illumination_slope: [f64; RADII],
    /// Centroid of each channel table in table units (x radial, y tangential), indexed `[radius][channel]`.
    centroid: [[[f64; 2]; 3]; RADII],
    /// Largest centroid distance from the table centre, in table units.
    pub(crate) centroid_max: f64,
    /// `tables[s]` holds, for `n = 3 + 2 s`, the `n x n` tables indexed `[radius][cell]`, each cell the weights
    /// of the three channels and a zero (so a cell is one 4-lane vector).
    tables: Vec<Vec<V4>>,
}

/// Four `f32` lanes: the three channels of a texel or of a cell's weights, and a padding lane.
type V4 = [f32; 4];

/// Centre of cell `k` of `n` cells over `[-1, 1]`.
fn cell_centre(k: usize, n: usize) -> f64 {
    -1.0 + (2 * k + 1) as f64 / n as f64
}

/// Moves the mass of a `PSF_SIZE` x `PSF_SIZE` table by `(dx, dy)` cells: each cell's mass is shared between the
/// four cells around its new place by bilinear weights, and what would leave the table lands on its outermost
/// row or column (the tables are cut at the widest spread the runtime draws, so there is mass out there to
/// keep: nothing is dropped and the sum stays 1). A zero shift returns the table as it was.
fn translate(table: &mut [f64], dx: f64, dy: f64) {
    if dx == 0.0 && dy == 0.0 {
        return;
    }
    let m = PSF_SIZE as i64;
    let (ix, iy) = (floor_i64(dx), floor_i64(dy));
    let (fx, fy) = (dx - ix as f64, dy - iy as f64);
    let mut out = vec![0.0; table.len()];
    for j in 0..m {
        for i in 0..m {
            let mass = table[(j * m + i) as usize];
            for (ox, oy, w) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let x = (i + ix + ox).clamp(0, m - 1);
                let y = (j + iy + oy).clamp(0, m - 1);
                out[(y * m + x) as usize] += mass * w;
            }
        }
    }
    table.copy_from_slice(&out);
}

/// Area-weighted resampling of a 33 x 33 mass table to `n x n` cells over the same span. Mass is kept.
fn resample(src: &[f64], n: usize) -> Vec<f64> {
    let m = PSF_SIZE;
    if n == m {
        return src.to_vec();
    }
    // Fraction of source cell k that target cell a covers, with both cut on the common grid of 33 n steps.
    let frac = |a: usize, k: usize| -> f64 {
        let (a0, a1) = (a * m, (a + 1) * m);
        let (k0, k1) = (k * n, (k + 1) * n);
        let overlap = a1.min(k1).saturating_sub(a0.max(k0));
        overlap as f64 / n as f64
    };
    // Along x: tmp[j][a] = sum_k src[j][k] F(a, k).
    let mut tmp = vec![0.0; m * n];
    for j in 0..m {
        for a in 0..n {
            let mut s = 0.0;
            for k in 0..m {
                s += src[j * m + k] * frac(a, k);
            }
            tmp[j * n + a] = s;
        }
    }
    // Along y: out[b][a] = sum_j F(b, j) tmp[j][a].
    let mut out = vec![0.0; n * n];
    for b in 0..n {
        for a in 0..n {
            let mut s = 0.0;
            for j in 0..m {
                s += frac(b, j) * tmp[j * n + a];
            }
            out[b * n + a] = s;
        }
    }
    out
}

impl FieldTables {
    /// Projects a lens's tables on the camera's channels and resamples them to every size.
    pub(crate) fn new(lens: &LensData, camera: &CameraResponse) -> Self {
        let cells = PSF_SIZE * PSF_SIZE;
        let mut proj = vec![0.0f64; RADII * 3 * cells];
        let mut centroid = [[[0.0; 2]; 3]; RADII];
        let table_centroid = |t: &[f64]| -> [f64; 2] {
            let (mut cx, mut cy) = (0.0, 0.0);
            for (i, m) in t.iter().enumerate() {
                cx += m * cell_centre(i % PSF_SIZE, PSF_SIZE);
                cy += m * cell_centre(i / PSF_SIZE, PSF_SIZE);
            }
            [cx, cy]
        };
        for r in 0..RADII {
            for c in 0..3 {
                let dst = &mut proj[(r * 3 + c) * cells..][..cells];
                for j in 0..PSF_WAVELENGTHS {
                    let w = camera.weights()[c][j];
                    for (d, s) in dst.iter_mut().zip(lens.psf(r, j)) {
                        *d += w * f64::from(*s);
                    }
                }
                let sum: f64 = dst.iter().sum();
                for d in dst.iter_mut() {
                    *d /= sum;
                }
                centroid[r][c] = table_centroid(dst);
            }
            // The lateral colour kept: red and blue move so that their centroids' offsets from green's are
            // `LATERAL_COLOR_SCALE` times the traced ones (table units are half-widths; a cell is 2 / 33).
            let green = centroid[r][1];
            for c in [0, 2] {
                let cell = 2.0 / PSF_SIZE as f64;
                let dx = (LATERAL_COLOR_SCALE - 1.0) * (centroid[r][c][0] - green[0]) / cell;
                let dy = (LATERAL_COLOR_SCALE - 1.0) * (centroid[r][c][1] - green[1]) / cell;
                let dst = &mut proj[(r * 3 + c) * cells..][..cells];
                translate(dst, dx, dy);
                centroid[r][c] = table_centroid(dst);
            }
        }
        let mut tables = Vec::with_capacity(SIZES);
        for s in 0..SIZES {
            let n = 3 + 2 * s;
            let nn = n * n;
            let mut t = vec![[0.0f32; 4]; RADII * nn];
            for r in 0..RADII {
                for c in 0..3 {
                    let mut cell = resample(&proj[(r * 3 + c) * cells..][..cells], n);
                    // Each kernel sums to 1 (to the rounding of the sum).
                    let sum: f64 = cell.iter().sum();
                    for v in &mut cell {
                        *v /= sum;
                    }
                    for (q, v) in cell.iter().enumerate() {
                        t[r * nn + q][c] = *v as f32;
                    }
                }
            }
            tables.push(t);
        }
        let mut radii = [0.0; RADII];
        let mut half_width = [0.0; RADII];
        let mut illumination = [0.0; RADII];
        radii.copy_from_slice(&lens.field.radii);
        half_width.copy_from_slice(&lens.field.psf_half_width);
        illumination.copy_from_slice(&lens.field.relative_illumination);
        let half_width = pool_monotone(&half_width, true);
        let illumination = pool_concave(&radii, &pool_monotone(&illumination, false));
        let centroid_max = centroid
            .iter()
            .flatten()
            .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
            .fold(0.0, f64::max);
        FieldTables {
            radii,
            half_width,
            illumination,
            half_width_slope: monotone_slopes(&radii, &half_width),
            illumination_slope: concave_slopes(&radii, &illumination),
            centroid,
            centroid_max,
            tables,
        }
    }

    /// The tables as the device reads them (`crate::device::FieldTablesData`): a header of
    /// [`crate::device::FIELD_HEADER`] four-lane vectors (the radii, `w(rho)` and `T(rho)` padded to 12 values
    /// each, then the channels' centroids `(x, y, 0, 0)` indexed `[radius * 3 + channel]`, then the slopes of
    /// the cubics through `w` and through `T`, padded likewise) followed by the cell tables for
    /// `n = 3, 5, ..., 17`, each indexed `[radius][cell]`.
    pub(crate) fn flatten(&self) -> Vec<[f32; 4]> {
        let mut out = Vec::new();
        let padded_column = |column: &[f64; RADII], out: &mut Vec<[f32; 4]>| {
            let mut padded = [0.0f32; 12];
            for (d, v) in padded.iter_mut().zip(column.iter()) {
                *d = *v as f32;
            }
            out.extend(padded.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]));
        };
        for column in [&self.radii, &self.half_width, &self.illumination] {
            padded_column(column, &mut out);
        }
        for r in 0..RADII {
            for c in 0..3 {
                out.push([
                    self.centroid[r][c][0] as f32,
                    self.centroid[r][c][1] as f32,
                    0.0,
                    0.0,
                ]);
            }
        }
        for column in [&self.half_width_slope, &self.illumination_slope] {
            padded_column(column, &mut out);
        }
        for t in &self.tables {
            out.extend_from_slice(t);
        }
        out
    }

    /// The relative illumination `T(rho_t)` at the table radius `rho_t` on segment `seg` with blend `t`
    /// (`segment(rho_t)`): the nine `T(rho)`, made concave, joined by [`concave_piece`] (`C^1`, its second
    /// derivative never above 0), and beyond `rho_t = 1` carried straight on at the end slope, so that along
    /// any row of a flat field the gain is monotone and its slope never lessens towards the edge of the
    /// frame (`docs/21` section 4, L2, *As built (the field law)*).
    pub(crate) fn illumination_at(&self, seg: usize, t: f64, rho_t: f64) -> f64 {
        if rho_t > 1.0 {
            let end = RADII - 1;
            return self.illumination[end] + self.illumination_slope[end] * (rho_t - 1.0);
        }
        let (a, b) = (self.radii[seg], self.radii[seg + 1]);
        concave_piece(
            self.illumination[seg],
            self.illumination[seg + 1],
            self.illumination_slope[seg],
            self.illumination_slope[seg + 1],
            b - a,
            t,
        )
    }

    /// What the oval field radius is divided by to read the relative illumination, for a frame whose far edge
    /// along the axis lies at `xa_max` half-diagonals and whose far edge across it at `ya_max`, under the stretch
    /// `alpha`. At least 1 (so that a frame without the attachment reads the tables as they are), at least what
    /// puts the oval radius of the middle of the long edge at [`SIDE_TABLE_RADIUS`] (or at the unstretched
    /// side's own radius where that is farther out, so that the scale grows from 1 with the attachment), and
    /// enough that the corner of the frame lies no farther past the last radius than [`T_OVERSHOOT`] of the way
    /// to the place where the end slope would take `T` to 0.
    pub(crate) fn radius_scale(&self, alpha: f64, xa_max: f64, ya_max: f64) -> f64 {
        let side = alpha * xa_max;
        let corner = (side * side + ya_max * ya_max).sqrt();
        let end = RADII - 1;
        let fall = -self.illumination_slope[end];
        let to_corner = if fall > 0.0 {
            corner / (1.0 + T_OVERSHOOT * self.illumination[end] / fall)
        } else {
            0.0
        };
        1.0f64
            .max(side / SIDE_TABLE_RADIUS.max(xa_max))
            .max(to_corner)
    }

    /// The half-width `w(rho')` of the spread functions (half-diagonals) on segment `seg` with blend `t`:
    /// the monotone cubic through the nine `w(rho)`, and beyond `rho' = 1` a line with the end slope.
    pub(crate) fn half_width_at(&self, seg: usize, t: f64, rho: f64) -> f64 {
        let (a, b) = (self.radii[seg], self.radii[seg + 1]);
        let near = hermite(
            self.half_width[seg],
            self.half_width[seg + 1],
            self.half_width_slope[seg],
            self.half_width_slope[seg + 1],
            b - a,
            t,
        );
        if rho <= 1.0 {
            return near;
        }
        near + self.half_width_slope[RADII - 1] * (rho - 1.0)
    }

    /// The segment `s` (0 to 7) and blend `t` (0 to 1) of field radius `rho` (held at 1 beyond it).
    fn segment(&self, rho: f64) -> (usize, f64) {
        let rho = rho.clamp(0.0, 1.0);
        let mut s = 0;
        while s + 2 < RADII && rho > self.radii[s + 1] {
            s += 1;
        }
        let (a, b) = (self.radii[s], self.radii[s + 1]);
        (s, ((rho - a) / (b - a)).clamp(0.0, 1.0))
    }

    /// The `n x n` table of `radius`: `n^2` cells of channel weights, row-major (x radial, y tangential).
    fn table(&self, n: usize, radius: usize) -> &[V4] {
        let len = n * n;
        &self.tables[(n - 3) / 2][radius * len..][..len]
    }
}

/// Everything the pixels of one call share.
#[derive(Clone, Copy)]
pub(crate) struct Geometry<'a> {
    tables: &'a FieldTables,
    /// Frame size at this scale.
    fw: f64,
    fh: f64,
    /// Half-diagonal in pixels.
    h: f64,
    axis: Axis,
    alpha: f64,
    /// What the oval field radius is divided by to read the relative illumination
    /// ([`FieldTables::radius_scale`]).
    radius_scale: f64,
    side: f64,
    e: f64,
    fringe: f64,
    opts: Options,
    /// Most mip levels any pixel of the frame can ask for.
    lmax: u32,
    /// Context the gather of a pixel reads, in pixels.
    gather_halo: usize,
    /// Pixels the fringe shift reads beyond the output (0 without the attachment's fringe).
    pad: usize,
}

/// One pixel's kernel parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelKernel {
    /// The gain `1 - e' (1 - T)`.
    pub gain: f64,
    /// The kernel's half-width in pixels (0 when the kernel is a copy).
    pub omega: f64,
    /// Lower table of the blend (the radius index).
    pub seg: usize,
    /// Weight of the upper table.
    pub t: f64,
    /// Radial unit vector in the frame (x right, y down): x component.
    pub cs: f64,
    /// Radial unit vector in the frame: y component.
    pub sn: f64,
}

/// Taps per side for a half-width of `omega` texels.
fn taps_for(omega: f64) -> usize {
    ((2.0 * omega.ceil() + 1.0) as usize).min(MAX_TAPS_SIDE)
}

/// The mip level that brings the cells of an `omega`-pixel kernel within `spacing_max` texels of each other.
fn level_for(omega: f64, spacing_max: f64) -> u32 {
    let mut level = 0;
    let mut w = omega;
    while level < MAX_LEVEL && 2.0 * w / taps_for(w) as f64 > spacing_max {
        w *= 0.5;
        level += 1;
    }
    level
}

/// `floor(x)` as an integer, without the library call of `f64::floor` on baseline x86-64.
#[inline(always)]
fn floor_i64(x: f64) -> i64 {
    let t = x as i64;
    t - i64::from((t as f64) > x)
}

impl<'a> Geometry<'a> {
    /// The geometry of `plan` for a frame of `fw x fh` pixels at the scale of the caller.
    ///
    /// `None` when the plan has no field optics (nothing to do and nothing to read).
    pub(crate) fn new(plan: &'a LensPlan, fw: usize, fh: usize, opts: Options) -> Option<Self> {
        let tables = plan.field_tables()?;
        let e = plan.field_amount();
        let (fwf, fhf) = (fw as f64, fh as f64);
        let h = 0.5 * (fwf * fwf + fhf * fhf).sqrt();
        let axis = plan.frame().axis;
        let (alpha, side, fringe) = (plan.alpha(), plan.side_softness(), plan.fringe_factor());
        // Upper bounds on the half-width, over the whole frame.
        let along = match axis {
            Axis::X => fwf,
            Axis::Y => fhf,
        };
        let across = match axis {
            Axis::X => fhf,
            Axis::Y => fwf,
        };
        let xa_max = 0.5 * along / h;
        let ya_max = 0.5 * across / h;
        let radius_scale = tables.radius_scale(alpha, xa_max, ya_max);
        // `w` does not fall with the radius, so its value at the frame's farthest point bounds every pixel's.
        let rho_far = ((alpha * xa_max) * (alpha * xa_max) + ya_max * ya_max).sqrt();
        let (seg_far, t_far) = tables.segment(rho_far);
        let w_far = tables.half_width_at(seg_far, t_far, rho_far);
        let omega_max = soft_cap(e * (1.0 + side * xa_max * xa_max) * w_far) * h;
        let mut lmax = 0;
        let mut o = COPY_BELOW_PX;
        while o < omega_max + 0.25 {
            lmax = lmax.max(level_for(o.min(omega_max), opts.spacing_max));
            o += 0.25;
        }
        let (gather_halo, pad) = if omega_max < COPY_BELOW_PX {
            (0, 0)
        } else {
            // A tap lies within sqrt(2) omega of the pixel centre; it reads texels covering P +- 1.5 texels at
            // the coarsest level.
            let reach = omega_max * std::f64::consts::SQRT_2;
            let margin = 1.5 * f64::from(1u32 << lmax);
            let gather_halo = reach.ceil() as usize + margin.ceil() as usize + 3;
            // The fringe shift is the boost times the centroid offset, at most `centroid_max` omega.
            let shift = (fringe - 1.0).max(0.0) * tables.centroid_max * omega_max;
            let pad = if shift > 0.0 {
                shift.ceil() as usize + 2
            } else {
                0
            };
            (gather_halo, pad)
        };
        Some(Geometry {
            tables,
            fw: fwf,
            fh: fhf,
            h,
            axis,
            alpha,
            radius_scale,
            side,
            e,
            fringe,
            opts,
            lmax,
            gather_halo,
            pad,
        })
    }

    /// Pixels of context an output rectangle needs around it.
    pub(crate) fn halo(&self) -> usize {
        self.gather_halo + self.pad
    }

    /// The numbers the device kernels take (`crate::device::FieldPass`).
    pub(crate) fn pass(&self) -> crate::device::FieldPass {
        crate::device::FieldPass {
            width: self.fw as usize,
            height: self.fh as usize,
            half_diag: self.h,
            axis_y: self.axis == Axis::Y,
            e: self.e,
            alpha: self.alpha,
            radius_scale: self.radius_scale,
            side: self.side,
            fringe_boost: self.fringe - 1.0,
            lmax: self.lmax,
            gather_halo: self.gather_halo,
            pad: self.pad,
        }
    }

    /// The kernel of the pixel at integer frame position `(i, j)`.
    pub(crate) fn pixel(&self, i: usize, j: usize) -> PixelKernel {
        let u = (i as f64 + 0.5 - 0.5 * self.fw) / self.h;
        let v = (j as f64 + 0.5 - 0.5 * self.fh) / self.h;
        let mut k = self.kernel_at(u, v);
        if k.omega < COPY_BELOW_PX {
            k.omega = 0.0;
        }
        k
    }

    /// The kernel at the field position `(u, v)` (frame coordinates in half-diagonals from the centre, x to the
    /// right and y down), with the half-width as the law gives it: a half-width below [`COPY_BELOW_PX`] is not
    /// yet turned into a copy (that is [`Geometry::pixel`]'s, for pixels). The law's gain and width are `C^1`
    /// and monotone along every ray from the centre (`docs/21` section 4, L2).
    pub(crate) fn kernel_at(&self, u: f64, v: f64) -> PixelKernel {
        let (xa, ya) = match self.axis {
            Axis::X => (u, v),
            Axis::Y => (v, u),
        };
        let ax = self.alpha * xa;
        let rho = (ax * ax + ya * ya).sqrt();
        let tb = self.tables;
        let (seg, t) = tb.segment(rho);
        // The gain reads the relative illumination at the oval radius over `radius_scale`; the kernel's shape
        // and width read the oval radius itself.
        let rho_t = rho / self.radius_scale;
        let (seg_t, t_t) = tb.segment(rho_t);
        let illum = tb.illumination_at(seg_t, t_t, rho_t);
        let gain = 1.0 - self.e * (1.0 - illum);
        let w = tb.half_width_at(seg, t, rho);
        let softness = 1.0 + self.side * xa * xa;
        let omega = soft_cap(self.e * softness * w) * self.h;
        let rr = (u * u + v * v).sqrt();
        let (cs, sn) = if rr > 1e-12 {
            (u / rr, v / rr)
        } else {
            (1.0, 0.0)
        };
        PixelKernel {
            gain,
            omega,
            seg,
            t,
            cs,
            sn,
        }
    }

    /// The blended centroid of channel `c` in table units.
    #[cfg(test)]
    fn centroid(&self, k: &PixelKernel, c: usize) -> [f64; 2] {
        let a = self.tables.centroid[k.seg][c];
        let b = self.tables.centroid[k.seg + 1][c];
        [
            (1.0 - k.t) * a[0] + k.t * b[0],
            (1.0 - k.t) * a[1] + k.t * b[1],
        ]
    }

    /// The centroid offset of channel `c` in pixels (frame x, y), without the attachment's fringe boost.
    #[cfg(test)]
    pub(crate) fn centroid_px(&self, k: &PixelKernel, c: usize) -> [f64; 2] {
        let [cx, cy] = self.centroid(k, c);
        // Table x is radial (cs, sn), table y is tangential (sn, -cs).
        [
            k.omega * (cx * k.cs + cy * k.sn),
            k.omega * (cx * k.sn - cy * k.cs),
        ]
    }

    /// The taps of channel `c` of a pixel, as `(dx, dy, weight)` with offsets from the pixel centre in
    /// pixels, for tests and reference checks. Positions and weights are those the gather uses at its mip
    /// level (the tap count `n` is that level's).
    #[cfg(test)]
    pub(crate) fn taps(&self, k: &PixelKernel, c: usize) -> Vec<(f64, f64, f64)> {
        if k.omega == 0.0 {
            return vec![(0.0, 0.0, 1.0)];
        }
        let level = level_for(k.omega, self.opts.spacing_max).min(self.lmax);
        let n = taps_for(k.omega / f64::from(1u32 << level));
        let (lo, hi) = (self.tables.table(n, k.seg), self.tables.table(n, k.seg + 1));
        let mut out = Vec::with_capacity(n * n);
        for b in 0..n {
            for a in 0..n {
                let (tx, ty) = (cell_centre(a, n), cell_centre(b, n));
                let ox = k.omega * (tx * k.cs + ty * k.sn);
                let oy = k.omega * (tx * k.sn - ty * k.cs);
                let q = b * n + a;
                out.push((
                    ox,
                    oy,
                    (1.0 - k.t) * f64::from(lo[q][c]) + k.t * f64::from(hi[q][c]),
                ));
            }
        }
        out
    }

    /// The shift, in pixels along the axis, at which channel `c` is read after the gather: the boost
    /// `h (q - 1)` times the axis part of the channel's centroid offset (0 for a copy).
    pub(crate) fn fringe_shift(&self, k: &PixelKernel, c: usize) -> f64 {
        let boost = self.fringe - 1.0;
        if boost == 0.0 || k.omega == 0.0 {
            return 0.0;
        }
        let a = self.tables.centroid[k.seg][c];
        let b = self.tables.centroid[k.seg + 1][c];
        let cx = (1.0 - k.t) * a[0] + k.t * b[0];
        let cy = (1.0 - k.t) * a[1] + k.t * b[1];
        boost
            * k.omega
            * match self.axis {
                Axis::X => cx * k.cs + cy * k.sn,
                Axis::Y => cx * k.sn - cy * k.cs,
            }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The input pyramid

/// One level of the input pyramid: texel `(bx, by)` is the mean of the input over the aligned block of
/// `2^L x 2^L` frame pixels starting at `(bx 2^L, by 2^L)`. A texel is its three channels and a zero.
struct Level {
    bx0: i64,
    by0: i64,
    w: usize,
    h: usize,
    data: Vec<V4>,
}

struct Pyramid {
    levels: Vec<Level>,
}

fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

impl Pyramid {
    /// Builds the levels `0..=lmax` over `need` (frame coordinates, may lie outside the frame), rounded
    /// outwards to blocks of `2^lmax`. Pixels outside the frame replicate the edge; pixels outside the input
    /// (which a covering input makes unread) take the nearest input pixel.
    fn build(input: &ImageRef, need: [i64; 4], lmax: u32) -> Pyramid {
        let b = 1i64 << lmax;
        let x0 = floor_div(need[0], b) * b;
        let y0 = floor_div(need[1], b) * b;
        let x1 = ceil_div(need[2], b) * b;
        let y1 = ceil_div(need[3], b) * b;
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);

        let (fw, fh) = (input.frame_width as i64, input.frame_height as i64);
        let (ix0, iy0) = (input.x0 as i64, input.y0 as i64);
        let (iw, ih) = (input.width, input.height);
        let planes = [input.plane(0), input.plane(1), input.plane(2)];
        let mut base = vec![[0.0f32; 4]; w * h];
        base.par_chunks_mut(w).enumerate().for_each(|(row, d)| {
            let y = (y0 + row as i64)
                .clamp(0, fh - 1)
                .clamp(iy0, iy0 + ih as i64 - 1);
            let at = (y - iy0) as usize * iw;
            for (col, px) in d.iter_mut().enumerate() {
                let x = (x0 + col as i64)
                    .clamp(0, fw - 1)
                    .clamp(ix0, ix0 + iw as i64 - 1);
                let i = at + (x - ix0) as usize;
                *px = [planes[0][i], planes[1][i], planes[2][i], 0.0];
            }
        });
        let mut levels = vec![Level {
            bx0: x0,
            by0: y0,
            w,
            h,
            data: base,
        }];
        for l in 1..=lmax {
            let prev = &levels[(l - 1) as usize];
            let (pw, w2, h2) = (prev.w, prev.w / 2, prev.h / 2);
            let mut data = vec![[0.0f32; 4]; w2 * h2];
            data.par_chunks_mut(w2).enumerate().for_each(|(row, d)| {
                let r0 = &prev.data[2 * row * pw..][..pw];
                let r1 = &prev.data[(2 * row + 1) * pw..][..pw];
                for (col, px) in d.iter_mut().enumerate() {
                    let (a, b) = (r0[2 * col], r0[2 * col + 1]);
                    let (c, dd) = (r1[2 * col], r1[2 * col + 1]);
                    for k in 0..3 {
                        px[k] = ((a[k] + b[k]) + (c[k] + dd[k])) * 0.25;
                    }
                }
            });
            let level = Level {
                bx0: prev.bx0 / 2,
                by0: prev.by0 / 2,
                w: w2,
                h: h2,
                data,
            };
            levels.push(level);
        }
        Pyramid { levels }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Sampling

/// Fractional bits of the fixed-point tap positions.
const FRAC: u32 = 20;
const FRAC_MASK: i64 = (1 << FRAC) - 1;
const FRAC_SCALE: f32 = 1.0 / (1u32 << FRAC) as f32;

/// A fixed-point value from a real one.
#[inline(always)]
fn fix(x: f64) -> i64 {
    floor_i64(x * f64::from(1u32 << FRAC) + 0.5)
}

#[inline(always)]
fn lerp(a: F4, b: F4, t: F4) -> F4 {
    a.add(t.mul(b.sub(a)))
}

/// Bilinear sample of `lv` at the texel coordinate `(ux, uy)`: fixed point, relative to the level's first
/// texel centre, so texel `(i, j)` is at `(i, j) << FRAC`.
///
/// With `CLAMP` the texel indices are held inside the level (for a pixel whose taps might leave it, which a
/// covering input rules out); without it they are used as they are.
#[inline(always)]
fn sample<const CLAMP: bool>(lv: &Level, ux: i64, uy: i64) -> F4 {
    let (mut ix, mut iy) = (ux >> FRAC, uy >> FRAC);
    let fx = F4::splat((ux & FRAC_MASK) as i32 as f32 * FRAC_SCALE);
    let fy = F4::splat((uy & FRAC_MASK) as i32 as f32 * FRAC_SCALE);
    if CLAMP {
        ix = ix.max(0).min(lv.w as i64 - 2);
        iy = iy.max(0).min(lv.h as i64 - 2);
    }
    let at = iy as usize * lv.w + ix as usize;
    // One slice for both rows: a single bounds check.
    let rows = &lv.data[at..at + lv.w + 2];
    let top = lerp(F4::load(&rows[0]), F4::load(&rows[1]), fx);
    let bot = lerp(F4::load(&rows[lv.w]), F4::load(&rows[lv.w + 1]), fx);
    lerp(top, bot, fy)
}

// ---------------------------------------------------------------------------------------------------------
// The pass

/// L2 on `out` (frame coordinates at the caller's scale): the planar output, `3 * out.w * out.h` samples.
///
/// `input` must cover `out` grown by [`Geometry::halo`] and clamped to the frame. With no field optics
/// (`plan.field_tables()` is `None`) the result is the input's `out` rectangle, bit for bit.
///
/// `extra` is light the picture does not record and the stage spreads (`light::l2_input`: the headroom of a
/// clipped light, and the light's colour where a channel clipped), an image of the rectangle of `input` or
/// more. It is gathered with the same spread functions and added to the gather of `input` by the share `mu`
/// of [`EXTRA_INFLOW_RAMP`] and [`EXTRA_CHANGE_RAMP`] (the larger of how much of the picture's inflow the extra
/// is, and how much the lens changes the pixel's picture): a pixel in a flat bright area, where the extra only
/// moves from pixel to pixel and the lens changes nothing, is as the picture alone gives it, and one beside a
/// clipped star has the extra whole. A pixel with no kernel (a copy) is never given any.
pub(crate) fn apply(
    plan: &LensPlan,
    input: &ImageRef,
    extra: Option<&ImageRef>,
    out: PixelRect,
    opts: Options,
) -> Result<Vec<f32>, LensError> {
    input.validate()?;
    let (fw, fh) = (input.frame_width, input.frame_height);
    if out.x1() > fw || out.y1() > fh {
        return Err(LensError::Frame(format!(
            "the output {out} does not fit the {fw}x{fh} frame"
        )));
    }
    let Some(geo) = Geometry::new(plan, fw, fh, opts) else {
        return copy_rect(input, out);
    };
    let need = out.grown(geo.halo(), fw, fh);
    if !input.rect().contains(&need) {
        return Err(LensError::InputTooSmall {
            needed: need.to_string(),
            have: input.rect().to_string(),
        });
    }
    if let Some(extra) = extra {
        extra.validate()?;
        if !extra.rect().contains(&need) {
            return Err(LensError::InputTooSmall {
                needed: need.to_string(),
                have: extra.rect().to_string(),
            });
        }
    }
    if out.w == 0 || out.h == 0 {
        return Ok(Vec::new());
    }
    // Pass 1: the gather over the output grown by the fringe's reach. The pyramid covers that grown by the
    // gather's halo, even beyond the frame (replicated).
    let gathered_rect = out.grown(geo.pad, fw, fh);
    let halo = geo.gather_halo as i64;
    let region = [
        gathered_rect.x as i64 - halo,
        gathered_rect.y as i64 - halo,
        gathered_rect.x1() as i64 + halo,
        gathered_rect.y1() as i64 + halo,
    ];
    let pyramid = Pyramid::build(input, region, geo.lmax);
    let mut gathered = gather_region(&geo, &pyramid, gathered_rect, None);
    if let Some(extra) = extra {
        let pyramid = Pyramid::build(extra, region, geo.lmax);
        let occupancy = Occupancy::build(extra);
        let more = gather_region(&geo, &pyramid, gathered_rect, Some(&occupancy));
        add_extra(&geo, input, &mut gathered, &more, gathered_rect);
    }

    // Pass 2: the fringe shift of each channel and the gain.
    Ok(finish(&geo, &gathered, gathered_rect, out))
}

/// A summed-area table of the pixels of an image that hold any light, to tell whether a window of it holds some.
struct Occupancy {
    x0: i64,
    y0: i64,
    w: usize,
    h: usize,
    /// `(w + 1) * (h + 1)` counts.
    sat: Vec<u32>,
}

impl Occupancy {
    fn build(img: &ImageRef) -> Self {
        let (w, h) = (img.width, img.height);
        let n = w * h;
        let mut sat = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                let any = (0..3).any(|c| img.data[c * n + y * w + x] != 0.0);
                row += u32::from(any);
                sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
            }
        }
        Occupancy {
            x0: img.x0 as i64,
            y0: img.y0 as i64,
            w,
            h,
            sat,
        }
    }

    /// True when some pixel in `[x0, x1) x [y0, y1)` (frame coordinates, clamped to the image) holds light.
    fn any(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> bool {
        let clamp = |v: i64, o: i64, n: usize| (v - o).clamp(0, n as i64) as usize;
        let (a, b) = (clamp(x0, self.x0, self.w), clamp(x1, self.x0, self.w));
        let (c, d) = (clamp(y0, self.y0, self.h), clamp(y1, self.y0, self.h));
        let at = |x: usize, y: usize| self.sat[y * (self.w + 1) + x];
        at(b, d) + at(a, c) > at(a, d) + at(b, c)
    }
}

/// Adds to the gather `gathered` of the picture the share `mu` of the gather `more` of the extra light, over `rect`
/// (see [`apply`]): the larger of the smoothsteps of `|more| / gathered` over [`EXTRA_INFLOW_RAMP`] and of
/// `|gathered - input| / input` at the pixel over [`EXTRA_CHANGE_RAMP`], each the largest over the channels, times
/// the smoothstep of the kernel's half-width over the half-diagonal over [`EXTRA_KERNEL_RAMP`].
fn add_extra(
    geo: &Geometry,
    input: &ImageRef,
    gathered: &mut [f32],
    more: &[f32],
    rect: PixelRect,
) {
    let plane = rect.w * rect.h;
    let smooth = |v: f64, (from, to): (f64, f64)| {
        let t = ((v - from) / (to - from)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let (r, rest) = gathered.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    let inp = input.width * input.height;
    r.par_chunks_mut(rect.w)
        .zip(g.par_chunks_mut(rect.w))
        .zip(b.par_chunks_mut(rect.w))
        .enumerate()
        .for_each(|(row, ((rr, gr), br))| {
            let j = rect.y + row;
            for col in 0..rect.w {
                let i = rect.x + col;
                if geo.pixel(i, j).omega == 0.0 {
                    continue;
                }
                let at = row * rect.w + col;
                let e = [more[at], more[plane + at], more[2 * plane + at]];
                if e == [0.0; 3] {
                    continue;
                }
                let base = [rr[col], gr[col], br[col]];
                let pi = (j - input.y0) * input.width + (i - input.x0);
                let (mut inflow, mut change) = (0.0f64, 0.0f64);
                for c in 0..3 {
                    let own = f64::from(input.data[c * inp + pi]);
                    let got = f64::from(base[c]);
                    inflow = inflow.max(f64::from(e[c]).abs() / got.max(EXTRA_CHANGE_FLOOR));
                    change = change.max((got - own).abs() / own.max(EXTRA_CHANGE_FLOOR));
                }
                let width = smooth(geo.pixel(i, j).omega / geo.h, EXTRA_KERNEL_RAMP);
                let mu = smooth(inflow, EXTRA_INFLOW_RAMP).max(smooth(change, EXTRA_CHANGE_RAMP))
                    * width;
                if mu > 0.0 {
                    rr[col] = (f64::from(base[0]) + mu * f64::from(e[0])) as f32;
                    gr[col] = (f64::from(base[1]) + mu * f64::from(e[1])) as f32;
                    br[col] = (f64::from(base[2]) + mu * f64::from(e[2])) as f32;
                }
            }
        });
}

/// The gather (without the gain) of every pixel of `rect`: planar, `3 * rect.w * rect.h` samples.
///
/// With an occupancy (the gather of the extra light) a pixel whose window holds none, and a pixel with no kernel,
/// is 0 without gathering.
fn gather_region(
    geo: &Geometry,
    pyr: &Pyramid,
    rect: PixelRect,
    occupancy: Option<&Occupancy>,
) -> Vec<f32> {
    let plane = rect.w * rect.h;
    let mut result = vec![0.0f32; 3 * plane];
    let (r, rest) = result.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    r.par_chunks_mut(rect.w)
        .zip(g.par_chunks_mut(rect.w))
        .zip(b.par_chunks_mut(rect.w))
        .enumerate()
        .for_each(|(row, ((rr, gr), br))| {
            let j = rect.y + row;
            for col in 0..rect.w {
                let i = rect.x + col;
                if let Some(occ) = occupancy {
                    let k = geo.pixel(i, j);
                    if k.omega == 0.0 {
                        continue;
                    }
                    // A tap lies within sqrt(2) omega of the pixel and reads texels 1.5 blocks wide.
                    let level = level_for(k.omega, geo.opts.spacing_max).min(geo.lmax);
                    let reach = (k.omega * std::f64::consts::SQRT_2).ceil() as i64
                        + (1.5 * f64::from(1u32 << level)).ceil() as i64
                        + 3;
                    if !occ.any(
                        i as i64 - reach,
                        j as i64 - reach,
                        i as i64 + reach + 1,
                        j as i64 + reach + 1,
                    ) {
                        continue;
                    }
                }
                let px = gather(geo, pyr, i, j);
                rr[col] = px[0] as f32;
                gr[col] = px[1] as f32;
                br[col] = px[2] as f32;
            }
        });
    result
}

/// The output: every channel of `out` read from `gathered` (the gather over `rect`) at its fringe shift along
/// the axis, times the gain.
fn finish(geo: &Geometry, gathered: &[f32], rect: PixelRect, out: PixelRect) -> Vec<f32> {
    let (gp, op) = (rect.w * rect.h, out.w * out.h);
    let (fw, fh) = (geo.fw as i64, geo.fh as i64);
    let mut result = vec![0.0f32; 3 * op];
    let (r, rest) = result.split_at_mut(op);
    let (g, b) = rest.split_at_mut(op);
    r.par_chunks_mut(out.w)
        .zip(g.par_chunks_mut(out.w))
        .zip(b.par_chunks_mut(out.w))
        .enumerate()
        .for_each(|(row, ((rr, gr), br))| {
            let j = out.y + row;
            for col in 0..out.w {
                let i = out.x + col;
                let k = geo.pixel(i, j);
                let mut px = [0.0f64; 3];
                for (c, v) in px.iter_mut().enumerate() {
                    let plane = &gathered[c * gp..(c + 1) * gp];
                    let shift = geo.fringe_shift(&k, c);
                    *v = if shift == 0.0 {
                        f64::from(plane[(j - rect.y) * rect.w + (i - rect.x)])
                    } else {
                        // Linear interpolation between the two pixels around i + shift (or j + shift).
                        let (along, lo, hi, first, last) = match geo.axis {
                            Axis::X => (i, rect.x, rect.x1(), 0, fw),
                            Axis::Y => (j, rect.y, rect.y1(), 0, fh),
                        };
                        let pos = along as f64 + shift;
                        let i0 = floor_i64(pos);
                        let f = pos - i0 as f64;
                        let at = |q: i64| -> f64 {
                            let q =
                                q.clamp(first, last - 1).clamp(lo as i64, hi as i64 - 1) as usize;
                            f64::from(match geo.axis {
                                Axis::X => plane[(j - rect.y) * rect.w + (q - rect.x)],
                                Axis::Y => plane[(q - rect.y) * rect.w + (i - rect.x)],
                            })
                        };
                        let (a, bb) = (at(i0), at(i0 + 1));
                        a + f * (bb - a)
                    };
                }
                rr[col] = (px[0] * k.gain) as f32;
                gr[col] = (px[1] * k.gain) as f32;
                br[col] = (px[2] * k.gain) as f32;
            }
        });
    result
}

/// The `out` rectangle of `input`, copied.
pub(crate) fn copy_rect(input: &ImageRef, out: PixelRect) -> Result<Vec<f32>, LensError> {
    if !input.rect().contains(&out) {
        return Err(LensError::InputTooSmall {
            needed: out.to_string(),
            have: input.rect().to_string(),
        });
    }
    let mut result = Vec::with_capacity(3 * out.w * out.h);
    for c in 0..3 {
        let plane = input.plane(c);
        for y in out.y..out.y1() {
            let row = (y - input.y0) * input.width + (out.x - input.x0);
            result.extend_from_slice(&plane[row..row + out.w]);
        }
    }
    Ok(result)
}

/// One pixel's gather (all three channels), without the gain.
fn gather(geo: &Geometry, pyr: &Pyramid, i: usize, j: usize) -> [f64; 3] {
    let k = geo.pixel(i, j);
    let base_lv = &pyr.levels[0];
    if k.omega == 0.0 {
        // A copy: read the input pixel itself.
        let at = (j as i64 - base_lv.by0) as usize * base_lv.w + (i as i64 - base_lv.bx0) as usize;
        let px = base_lv.data[at];
        return [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
    }
    let (cx, cy) = (i as f64 + 0.5, j as f64 + 0.5);
    let level = level_for(k.omega, geo.opts.spacing_max).min(geo.lmax);
    let scale = f64::from(1u32 << level);
    let inv = 1.0 / scale;
    let n = taps_for(k.omega * inv);
    let lv = &pyr.levels[level as usize];
    let nf = n as f64;
    // Table (tx, ty) -> frame offset omega (tx r + ty t) with r = (cs, sn), t = (sn, -cs). Positions below
    // are texel coordinates of the level, fixed point, relative to its first texel centre.
    let step = 2.0 * k.omega / nf;
    let first = k.omega * (-1.0 + 1.0 / nf);
    let to_texel = |pos: f64, origin: i64| fix(pos * inv - 0.5) - (origin << FRAC);
    let bx = to_texel(cx + first * (k.cs + k.sn), lv.bx0);
    let by = to_texel(cy + first * (k.sn - k.cs), lv.by0);
    let (ea_x, ea_y) = (fix(step * k.cs * inv), fix(step * k.sn * inv));
    let (eb_x, eb_y) = (fix(step * k.sn * inv), fix(-step * k.cs * inv));

    // The positions the taps take: base + col ea + row eb, col and row in 0..n. Do they stay inside the
    // level? (They do whenever the input covers the halo; the clamped path keeps a short input safe.)
    let last = (n - 1) as i64;
    let span = |b: i64, ea: i64, eb: i64| {
        (
            b + (last * ea).min(0) + (last * eb).min(0),
            b + (last * ea).max(0) + (last * eb).max(0),
        )
    };
    let (x_lo, x_hi) = span(bx, ea_x, eb_x);
    let (y_lo, y_hi) = span(by, ea_y, eb_y);
    let inside = (x_lo >> FRAC) >= 0
        && (x_hi >> FRAC) <= lv.w as i64 - 2
        && (y_lo >> FRAC) >= 0
        && (y_hi >> FRAC) <= lv.h as i64 - 2;

    let (t0, t1) = (F4::splat((1.0 - k.t) as f32), F4::splat(k.t as f32));
    let lo = geo.tables.table(n, k.seg);
    let hi = geo.tables.table(n, k.seg + 1);
    let mut acc = [0.0f64; 3];
    let mut run = |clamp: bool| {
        for row in 0..n {
            let (mut px, mut py) = (bx + row as i64 * eb_x, by + row as i64 * eb_y);
            let wl = &lo[row * n..(row + 1) * n];
            let wh = &hi[row * n..(row + 1) * n];
            let mut sum = F4::splat(0.0);
            for (a, b) in wl.iter().zip(wh) {
                let s = if clamp {
                    sample::<true>(lv, px, py)
                } else {
                    sample::<false>(lv, px, py)
                };
                let w = t0.mul(F4::load(a)).add(t1.mul(F4::load(b)));
                sum = sum.add(w.mul(s));
                px += ea_x;
                py += ea_y;
            }
            let sum = sum.to_array();
            for c in 0..3 {
                acc[c] += f64::from(sum[c]);
            }
        }
    };
    run(!inside);
    acc
}

/// The context `plan` needs around an output rectangle of a `fw x fh` frame (0 without field optics).
pub(crate) fn halo(plan: &LensPlan, fw: usize, fh: usize) -> usize {
    Geometry::new(plan, fw, fh, Options::default()).map_or(0, |g| g.halo())
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::amounts::{LensAmounts, LensKind};
    use crate::data::LensLibrary;
    use crate::plan::FrameInfo;
    use crate::test_support::write_synthetic_lenses;

    fn library() -> &'static LensLibrary {
        static LIB: OnceLock<LensLibrary> = OnceLock::new();
        LIB.get_or_init(|| {
            let dir = tempfile::tempdir().expect("temp dir");
            write_synthetic_lenses(dir.path()).expect("write lenses");
            LensLibrary::from_dir(dir.path()).expect("load lenses")
        })
    }

    /// The A7R IV's projection: its channels differ, so the lateral colour shows.
    fn camera() -> &'static CameraResponse {
        static CAM: OnceLock<CameraResponse> = OnceLock::new();
        CAM.get_or_init(|| {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
            );
            let curves = ym_color::spectral::SpectralCurves::from_path(path).expect("curves");
            CameraResponse::from_curves(&curves).expect("projection")
        })
    }

    fn plan(
        kind: LensKind,
        edges: f64,
        anamorphic: f64,
        w: usize,
        h: usize,
        axis: Axis,
    ) -> LensPlan {
        let amounts = LensAmounts {
            kind,
            flares: 0.0,
            edges,
            anamorphic,
        };
        let frame = FrameInfo {
            width: w,
            height: h,
            axis,
            f_number: None,
            clip: [1.0; 3],
        };
        LensPlan::new(library(), &amounts, frame, camera()).expect("plan")
    }

    /// A deterministic image with texture at every scale: planar RGB values in 0.05 to 1.
    fn noise(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..3 * w * h)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                0.05 + 0.95 * ((state >> 40) as f32 / (1u64 << 24) as f32)
            })
            .collect()
    }

    fn whole(w: usize, h: usize, data: &[f32]) -> ImageRef<'_> {
        ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data,
        }
    }

    #[test]
    fn resample_keeps_mass_and_is_exact_at_33() {
        let src: Vec<f64> = (0..33 * 33).map(|i| 1.0 + f64::from(i % 7)).collect();
        let total: f64 = src.iter().sum();
        assert_eq!(resample(&src, 33), src);
        for n in [3, 5, 9, 17] {
            let r = resample(&src, n);
            assert_eq!(r.len(), n * n);
            assert!(
                (r.iter().sum::<f64>() - total).abs() < 1e-9 * total,
                "n = {n}"
            );
        }
        // A uniform table stays uniform.
        let flat = vec![1.0 / 1089.0; 1089];
        for v in resample(&flat, 5) {
            assert!((v - 1.0 / 25.0).abs() < 1e-12);
        }
    }

    #[test]
    fn level_choice_keeps_the_tap_budget() {
        assert_eq!(level_for(0.6, 1.0), 0);
        assert_eq!(level_for(8.0, 1.0), 0);
        assert_eq!(level_for(8.5, 1.0), 0);
        assert_eq!(level_for(8.6, 1.0), 1);
        assert_eq!(level_for(17.0, 1.0), 1);
        assert_eq!(level_for(17.5, 1.0), 2);
        assert_eq!(level_for(62.0, 1.0), 3);
        for omega in [0.7, 3.2, 9.0, 21.0, 43.0, 62.0] {
            let l = level_for(omega, 1.0);
            let n = taps_for(omega / f64::from(1u32 << l));
            assert!(n <= MAX_TAPS_SIDE);
            assert!(2.0 * omega / f64::from(1u32 << l) / n as f64 <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn kernels_sum_to_one_and_follow_the_projected_centroids() {
        let p = plan(LensKind::Vintage, 100.0, 0.0, 1500, 1000, Axis::X);
        let geo = Geometry::new(&p, 1500, 1000, Options::default()).unwrap();
        let mut checked = 0;
        for (i, j) in [
            (750, 500),
            (1200, 500),
            (300, 120),
            (1400, 950),
            (750, 40),
            (60, 500),
        ] {
            let k = geo.pixel(i, j);
            if k.omega == 0.0 {
                continue;
            }
            checked += 1;
            for c in 0..3 {
                let taps = geo.taps(&k, c);
                let sum: f64 = taps.iter().map(|t| t.2).sum();
                assert!(
                    (sum - 1.0).abs() < 2e-7,
                    "sum {sum} at ({i},{j}) channel {c}"
                );
                let (mx, my) = (
                    taps.iter().map(|t| t.0 * t.2).sum::<f64>(),
                    taps.iter().map(|t| t.1 * t.2).sum::<f64>(),
                );
                let want = geo.centroid_px(&k, c);
                // The area-weighted resampling moves the centroid by a small part of a cell.
                let cell = 2.0 * k.omega
                    / taps_for(k.omega / f64::from(1u32 << level_for(k.omega, 1.0))) as f64;
                assert!(
                    (mx - want[0]).abs() < 0.2 * cell + 1e-6,
                    "x {mx} vs {} at ({i},{j}) c{c}",
                    want[0]
                );
                assert!(
                    (my - want[1]).abs() < 0.2 * cell + 1e-6,
                    "y {my} vs {} at ({i},{j}) c{c}",
                    want[1]
                );
            }
        }
        assert!(checked >= 4);
    }

    #[test]
    fn red_moves_outwards_and_blue_inwards() {
        // Lateral colour of the synthetic lenses: long wavelengths shift radially outwards of short ones, with
        // green between them (the lateral colour is kept about green's centroid, `LATERAL_COLOR_SCALE`).
        let p = plan(LensKind::Vintage, 100.0, 0.0, 1500, 1000, Axis::X);
        let geo = Geometry::new(&p, 1500, 1000, Options::default()).unwrap();
        // Right of the centre: radial is +x. Below it: radial is +y. Up-left: radial points up-left.
        for (i, j, sx, sy) in [
            (1300, 500, 1.0, 0.0),
            (750, 900, 0.0, 1.0),
            (200, 100, -1.0, -1.0),
        ] {
            let k = geo.pixel(i, j);
            let dot = |c: usize| {
                let v = geo.centroid_px(&k, c);
                v[0] * sx + v[1] * sy
            };
            let (r, g, b) = (dot(0), dot(1), dot(2));
            assert!(
                r > g + 0.005,
                "red {r} is not outwards of green {g} at ({i},{j})"
            );
            assert!(
                g > b + 0.005,
                "blue {b} is not inwards of green {g} at ({i},{j})"
            );
        }
    }

    #[test]
    fn the_lateral_colour_kept_is_the_tuned_share_of_the_traced_one() {
        // Green keeps its place; red and blue keep `LATERAL_COLOR_SCALE` of their offsets from it.
        let kind = LensKind::Classic;
        let lens = shipped().get(kind).expect("lens");
        let p = shipped_plan(kind, 100.0, 0.0, 1500, 1000, Axis::X);
        let tables = p.field_tables().unwrap();
        let cell = 2.0 / PSF_SIZE as f64;
        for r in [2, 5, 8] {
            let traced: Vec<[f64; 2]> = (0..3)
                .map(|c| {
                    let (mut cx, mut cy, mut total) = (0.0, 0.0, 0.0);
                    for j in 0..PSF_WAVELENGTHS {
                        let w = camera().weights()[c][j];
                        for (i, v) in lens.psf(r, j).iter().enumerate() {
                            let m = w * f64::from(*v);
                            total += m;
                            cx += m * cell_centre(i % PSF_SIZE, PSF_SIZE);
                            cy += m * cell_centre(i / PSF_SIZE, PSF_SIZE);
                        }
                    }
                    [cx / total, cy / total]
                })
                .collect();
            assert!((tables.centroid[r][1][0] - traced[1][0]).abs() < 1e-12);
            for c in [0, 2] {
                let want: [f64; 2] = std::array::from_fn(|a| {
                    traced[1][a] + LATERAL_COLOR_SCALE * (traced[c][a] - traced[1][a])
                });
                // The translation splits mass between cells, so the centroid is exact to rounding, but
                // for the mass that would leave the table and is kept on its edge (the Classic's tables
                // are cut at the widest spread): a few hundredths of a cell.
                for (axis, (got, want)) in tables.centroid[r][c].iter().zip(want).enumerate() {
                    assert!(
                        (got - want).abs() < 0.05 * cell,
                        "radius {r} channel {c} axis {axis}: {got} against {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn table_y_is_anticlockwise() {
        // A table whose mass sits at +y (tangential) must land anticlockwise of the radial direction on screen
        // (y down): to the right of the centre, anticlockwise is up; below the centre it is to the right.
        let p = plan(LensKind::Default, 100.0, 0.0, 1500, 1000, Axis::X);
        let mut tables = p.field_tables().unwrap().clone();
        for s in 0..SIZES {
            let n = 3 + 2 * s;
            let len = n * n;
            let mut one = vec![[0.0f32; 4]; len];
            // Cell (x = middle, y = last): table position (0, +1 - 1/n).
            one[(n - 1) * n + n / 2] = [1.0, 1.0, 1.0, 0.0];
            for r in 0..RADII {
                tables.tables[s][r * len..(r + 1) * len].copy_from_slice(&one);
            }
        }
        let geo = Geometry {
            tables: &tables,
            ..Geometry::new(&p, 1500, 1000, Options::default()).unwrap()
        };
        let right = geo.pixel(1300, 500);
        let t = geo.taps(&right, 0);
        let heavy = t.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        assert!(
            heavy.1 < -0.5 * right.omega && heavy.0.abs() < 0.2 * right.omega,
            "right: {heavy:?}"
        );
        let below = geo.pixel(750, 900);
        let t = geo.taps(&below, 0);
        let heavy = t.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        assert!(
            heavy.0 > 0.5 * below.omega && heavy.1.abs() < 0.2 * below.omega,
            "below: {heavy:?}"
        );
    }

    /// The red-blue displacement of the kernels' centroids (the fringe on a dark edge against a bright one), in
    /// half-diagonals, the largest over a grid of the frame, including the attachment's doubling of the axis part.
    fn largest_fringe(geo: &Geometry, fw: usize, fh: usize) -> f64 {
        let mut worst = 0.0f64;
        for j in (0..fh).step_by(25).chain([fh - 1]) {
            for i in (0..fw).step_by(25).chain([fw - 1]) {
                let k = geo.pixel(i, j);
                let at = |c: usize| {
                    let m = geo.centroid_px(&k, c);
                    let shift = geo.fringe_shift(&k, c);
                    match geo.axis {
                        Axis::X => [m[0] + shift, m[1]],
                        Axis::Y => [m[0], m[1] + shift],
                    }
                };
                let (r, b) = (at(0), at(2));
                worst = worst.max((r[0] - b[0]).hypot(r[1] - b[1]) / geo.h);
            }
        }
        worst
    }

    /// The lateral colour of every shipped lens stays within what a strong real lens shows: at Edges 100 with
    /// Anamorphic 100 (which doubles the axis part) the red-blue displacement is at most 0.15 % of the
    /// half-diagonal anywhere in the frame (the traced lenses gave 0.35 to 0.45 %), and at Edges 100 alone at most
    /// half of that. A soft colour edge remains: the worst lens still has about a pixel on a 1500 px frame.
    #[test]
    fn the_lateral_colour_of_every_shipped_lens_stays_small() {
        let mut worst_both = 0.0f64;
        for kind in LensKind::ALL {
            let mut row = format!("{kind:8}");
            for (edges, anamorphic, limit) in [(100.0, 100.0, 0.0015), (100.0, 0.0, 0.00075)] {
                let p = shipped_plan(kind, edges, anamorphic, 1500, 1000, Axis::X);
                let geo = Geometry::new(&p, 1500, 1000, Options::default()).expect("field optics");
                let worst = largest_fringe(&geo, 1500, 1000);
                row += &format!(
                    "  E{edges} A{anamorphic}: {:.3} % H ({:.2} px)",
                    100.0 * worst,
                    worst * geo.h
                );
                assert!(
                    worst <= limit,
                    "{kind} edges {edges} anamorphic {anamorphic}: red and blue are {:.3} % of the half-diagonal apart",
                    100.0 * worst
                );
                if anamorphic > 0.0 {
                    worst_both = worst_both.max(worst);
                }
            }
            println!("{row}");
        }
        assert!(worst_both > 0.0005, "the fringe is gone: {worst_both}");
    }

    /// The shipped lenses (`data/lenses`), the real ones: the field law is checked on them, not on the
    /// synthetic tables.
    fn shipped() -> &'static LensLibrary {
        static LIB: OnceLock<LensLibrary> = OnceLock::new();
        LIB.get_or_init(|| {
            LensLibrary::from_dir(std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../data/lenses"
            )))
            .expect("shipped lenses")
        })
    }

    fn shipped_plan(
        kind: LensKind,
        edges: f64,
        anamorphic: f64,
        w: usize,
        h: usize,
        axis: Axis,
    ) -> LensPlan {
        let amounts = LensAmounts {
            kind,
            flares: 0.0,
            edges,
            anamorphic,
        };
        let frame = FrameInfo {
            width: w,
            height: h,
            axis,
            f_number: None,
            clip: [1.0; 3],
        };
        LensPlan::new(shipped(), &amounts, frame, camera()).expect("plan")
    }

    /// The gain and the width `omega / W_MAX` along the ray at `angle` from the frame's centre to its edge,
    /// `n + 1` samples, with the step `ds` in half-diagonals.
    fn ray_profile(geo: &Geometry, angle: f64, n: usize) -> (Vec<f64>, Vec<f64>, f64) {
        let (c, s) = (angle.cos(), angle.sin());
        let (um, vm) = (0.5 * geo.fw / geo.h, 0.5 * geo.fh / geo.h);
        let s_max = (um / c.abs().max(1e-12)).min(vm / s.abs().max(1e-12));
        let ds = s_max / n as f64;
        let (mut gain, mut width) = (Vec::new(), Vec::new());
        for i in 0..=n {
            let r = ds * i as f64;
            let k = geo.kernel_at(r * c, r * s);
            gain.push(k.gain);
            width.push(k.omega / geo.h / W_MAX);
        }
        (gain, width, ds)
    }

    /// The largest change of the slope between neighbouring samples of a curve sampled at step `ds`, as the
    /// slope's change per sample over `ds`.
    fn largest_slope_step(values: &[f64], ds: f64) -> (f64, usize) {
        let mut worst = (0.0f64, 0usize);
        for i in 1..values.len() - 1 {
            let step = ((values[i + 1] - values[i]) - (values[i] - values[i - 1])) / ds;
            if step.abs() > worst.0 {
                worst = (step.abs(), i);
            }
        }
        worst
    }

    /// The tolerances of the `C^1` check, in the change of slope per sample: the law's curvature times the
    /// step (the gain's slope is at most about 2 and bends over 0.1 or more, so below 0.05; the width over
    /// `W_MAX` bends where it meets the cap, below 0.3), against the corners the checks are for, which change
    /// the slope by 0.5 to 4 at once (`the_check_sees_the_corners_of_the_old_law`).
    const GAIN_SLOPE_STEP: f64 = 0.05;
    const WIDTH_SLOPE_STEP: f64 = 0.3;

    /// Every lens, many amounts, both orientations and both axes: along any ray from the centre the gain
    /// never rises and the width never falls (to 1e-12), and both are `C^1` (slope steps under
    /// [`GAIN_SLOPE_STEP`] and [`WIDTH_SLOPE_STEP`] at 2000 samples a ray).
    #[test]
    fn the_field_law_is_c1_and_monotone_along_every_ray() {
        let n = 2000;
        let mut checked = 0usize;
        let mut worst_gain = (0.0f64, String::new());
        let mut worst_width = (0.0f64, String::new());
        for kind in LensKind::ALL {
            for (edges, anamorphic) in [
                (0.0, 100.0),
                (0.0, 40.0),
                (10.0, 0.0),
                (35.0, 60.0),
                (60.0, 100.0),
                (100.0, 0.0),
                (100.0, 30.0),
                (100.0, 100.0),
            ] {
                for (fw, fh, axis) in [
                    (1500, 1000, Axis::X),
                    (1000, 1500, Axis::Y),
                    (1000, 1500, Axis::X),
                    (1500, 1000, Axis::Y),
                    (1600, 900, Axis::X),
                ] {
                    let p = shipped_plan(kind, edges, anamorphic, fw, fh, axis);
                    let geo = Geometry::new(&p, fw, fh, Options::default()).expect("field optics");
                    for a in 0..16 {
                        let angle = std::f64::consts::PI * (a as f64 + 0.37) / 8.0;
                        let (gain, width, ds) = ray_profile(&geo, angle, n);
                        let what = format!(
                            "{kind} edges {edges} anamorphic {anamorphic} {fw}x{fh} {axis:?} ray {a}"
                        );
                        for i in 0..n {
                            assert!(
                                gain[i + 1] <= gain[i] + 1e-12,
                                "{what}: the gain rises at sample {i}"
                            );
                            assert!(
                                width[i + 1] >= width[i] - 1e-12,
                                "{what}: the width falls at sample {i}"
                            );
                        }
                        let (g, gi) = largest_slope_step(&gain, ds);
                        let (w, wi) = largest_slope_step(&width, ds);
                        if g > worst_gain.0 {
                            worst_gain = (g, format!("{what} sample {gi}"));
                        }
                        if w > worst_width.0 {
                            worst_width = (w, format!("{what} sample {wi}"));
                        }
                        checked += 1;
                    }
                }
            }
        }
        println!(
            "{checked} rays; largest slope step: gain {:.4} ({}), width {:.4} ({})",
            worst_gain.0, worst_gain.1, worst_width.0, worst_width.1
        );
        assert!(checked > 4000);
        assert!(
            worst_gain.0 < GAIN_SLOPE_STEP,
            "the gain's slope jumps by {:.4}: {}",
            worst_gain.0,
            worst_gain.1
        );
        assert!(
            worst_width.0 < WIDTH_SLOPE_STEP,
            "the width's slope jumps by {:.4}: {}",
            worst_width.0,
            worst_width.1
        );
    }

    /// The check has teeth: the law this build replaced (the tables read linearly between radii, `rho'` held at 1,
    /// the width cut at `W_MAX`) fails it by a wide margin on the same rays.
    #[test]
    fn the_check_sees_the_corners_of_the_old_law() {
        let n = 2000;
        for (kind, edges, gain_floor, width_floor) in [
            (LensKind::Default, 40.0, 10.0 * GAIN_SLOPE_STEP, 0.0),
            (
                LensKind::Vintage,
                100.0,
                10.0 * GAIN_SLOPE_STEP,
                10.0 * WIDTH_SLOPE_STEP,
            ),
        ] {
            let p = shipped_plan(kind, edges, 100.0, 1500, 1000, Axis::X);
            let geo = Geometry::new(&p, 1500, 1000, Options::default()).expect("field optics");
            let radii = p.lens().field.radii.clone();
            let lin = |y: &[f64], rho: f64| {
                let rho = rho.min(1.0);
                let s = (0..8).find(|s| rho <= radii[s + 1]).unwrap_or(7);
                let u = (rho - radii[s]) / (radii[s + 1] - radii[s]);
                (1.0 - u) * y[s] + u * y[s + 1]
            };
            let ds = 0.5 * geo.fw / geo.h / n as f64;
            let e = p.field_amount();
            let (mut gain, mut width) = (Vec::new(), Vec::new());
            for i in 0..=n {
                let x = ds * i as f64;
                let rho = p.alpha() * x;
                gain.push(1.0 - e * (1.0 - lin(&p.lens().field.relative_illumination, rho)));
                let w = lin(&p.lens().field.psf_half_width, rho);
                width.push((e * (1.0 + p.side_softness() * x * x) * w).min(W_MAX) / W_MAX);
            }
            let (g, _) = largest_slope_step(&gain, ds);
            let (w, _) = largest_slope_step(&width, ds);
            println!("{kind} old law: largest slope step: gain {g:.3}, width {w:.3}");
            assert!(
                g > gain_floor,
                "{kind}: the old gain's worst slope step is {g}"
            );
            assert!(
                w > width_floor,
                "{kind}: the old width's worst slope step is {w}"
            );
            // The new law on the same ray.
            let (new_gain, new_width, nds) = ray_profile(&geo, 0.0, n);
            let (g2, _) = largest_slope_step(&new_gain, nds);
            let (w2, _) = largest_slope_step(&new_width, nds);
            println!("{kind} new law: largest slope step: gain {g2:.3}, width {w2:.3}");
            assert!(g2 < GAIN_SLOPE_STEP, "{kind}: {g2}");
            assert!(w2 < WIDTH_SLOPE_STEP, "{kind}: {w2}");
        }
    }

    /// The gain along a row (`v` fixed) or a column (`u` fixed) of the frame, from the centre line out to the edge,
    /// at `n + 1` samples.
    fn gain_line(geo: &Geometry, fixed: f64, along_u: bool, n: usize) -> Vec<f64> {
        let max = if along_u {
            0.5 * geo.fw / geo.h
        } else {
            0.5 * geo.fh / geo.h
        };
        (0..=n)
            .map(|i| {
                let x = max * i as f64 / n as f64;
                if along_u {
                    geo.kernel_at(x, fixed).gain
                } else {
                    geo.kernel_at(fixed, x).gain
                }
            })
            .collect()
    }

    /// The flat-field falloff has no band: along every row and column, from the centre line out, the gain
    /// never rises and its slope never lessens (second differences of one sign, to rounding) for every lens, at
    /// Anamorphic 0, 50 and 100, at several Edges, on landscape, portrait, square and panorama frames with the
    /// axis along either side (`docs/21` section 4, L2). The law this build replaced bent back at `rho' = 1`,
    /// 56 px from the side of a 1500 px frame at Anamorphic 100, which a white field shows as a grey band.
    #[test]
    fn the_gain_is_concave_along_every_row_and_column_of_a_flat_field() {
        let n = 1500;
        let mut worst = (0.0f64, String::new());
        let mut lines = 0usize;
        for kind in LensKind::ALL {
            for anamorphic in [0.0, 50.0, 100.0] {
                for edges in [10.0, 40.0, 70.0, 100.0] {
                    for (fw, fh, axis) in [
                        (1500, 1000, Axis::X),
                        (1000, 1500, Axis::Y),
                        (1000, 1500, Axis::X),
                        (1500, 1000, Axis::Y),
                        (1600, 900, Axis::X),
                        (1000, 1000, Axis::X),
                        (3000, 1000, Axis::X),
                    ] {
                        let p = shipped_plan(kind, edges, anamorphic, fw, fh, axis);
                        let geo =
                            Geometry::new(&p, fw, fh, Options::default()).expect("field optics");
                        for along_u in [true, false] {
                            // The other coordinate runs from the centre line to the far edge.
                            let across = if along_u {
                                0.5 * geo.fh / geo.h
                            } else {
                                0.5 * geo.fw / geo.h
                            };
                            for r in 0..=12 {
                                let g = gain_line(&geo, across * r as f64 / 12.0, along_u, n);
                                let what = format!(
                                    "{kind} edges {edges} anamorphic {anamorphic} {fw}x{fh} {axis:?} \
                                     {} {r}",
                                    if along_u { "row" } else { "column" }
                                );
                                for i in 1..g.len() - 1 {
                                    assert!(g[i] <= g[i - 1] + 1e-12, "{what}: rises at {i}");
                                    let d2 = g[i + 1] - 2.0 * g[i] + g[i - 1];
                                    if d2 > worst.0 {
                                        worst = (d2, format!("{what} sample {i}"));
                                    }
                                }
                                assert!(g[n] <= g[n - 1] + 1e-12, "{what}: rises at the edge");
                                lines += 1;
                            }
                        }
                    }
                }
            }
        }
        println!(
            "{lines} lines; the largest second difference (it must be 0): {:.2e} ({})",
            worst.0, worst.1
        );
        assert!(lines > 10_000);
        assert!(
            worst.0 < 1e-12,
            "the gain bends back by {:.3e}: {}",
            worst.0,
            worst.1
        );
    }

    /// The scale that puts the side of the frame at [`SIDE_TABLE_RADIUS`] of the tables: 1 without the attachment
    /// (the tables are read as they are), the oval radius of the middle of the long edge over it from about
    /// Anamorphic 45 on, more where the corner would otherwise be left with less than `1 - T_OVERSHOOT` of the
    /// traced end value; and the corner's gain is never below that, so no lens is cut to black or to a negative
    /// gain.
    #[test]
    fn the_table_radius_scale_keeps_the_side_at_the_tables_side_radius_and_the_corner_above_zero() {
        for kind in LensKind::ALL {
            let p0 = shipped_plan(kind, 100.0, 0.0, 1500, 1000, Axis::X);
            let g0 = Geometry::new(&p0, 1500, 1000, Options::default()).expect("field optics");
            assert_eq!(g0.radius_scale, 1.0, "{kind}: scale without the attachment");
            let tb = g0.tables;
            let end = tb.illumination[RADII - 1];
            for (fw, fh, axis) in [
                (1500, 1000, Axis::X),
                (1000, 1500, Axis::Y),
                (1000, 1000, Axis::X),
                (3000, 1000, Axis::X),
            ] {
                for anamorphic in [30.0, 70.0, 100.0] {
                    let p = shipped_plan(kind, 100.0, anamorphic, fw, fh, axis);
                    let geo = Geometry::new(&p, fw, fh, Options::default()).expect("field optics");
                    let e = p.field_amount();
                    assert_eq!(e, 1.0);
                    let (um, vm) = (0.5 * geo.fw / geo.h, 0.5 * geo.fh / geo.h);
                    let xm = if axis == Axis::X { um } else { vm };
                    // The middle of the long edge is at the side radius of the tables or inside it.
                    assert!(
                        p.alpha() * xm / geo.radius_scale <= SIDE_TABLE_RADIUS.max(xm) + 1e-12,
                        "{kind} {fw}x{fh} A{anamorphic}: the side is past the tables' side radius"
                    );
                    // The corner's relative illumination, at the scaled radius.
                    let corner = geo.kernel_at(um, vm).gain;
                    let floor = (1.0 - T_OVERSHOOT) * end;
                    assert!(
                        corner >= floor - 1e-9,
                        "{kind} {fw}x{fh} A{anamorphic}: the corner's gain {corner} is below {floor}"
                    );
                    // The middle of the long edge: the traced end value, or lighter where the corner asked
                    // for a larger scale.
                    let side = if axis == Axis::X {
                        geo.kernel_at(um, 0.0).gain
                    } else {
                        geo.kernel_at(0.0, vm).gain
                    };
                    if p.alpha() * xm > 1.0 {
                        assert!(
                            side >= end - 1e-9,
                            "{kind} {fw}x{fh} A{anamorphic}: the side is darker ({side}) than T(1) {end}"
                        );
                    }
                }
            }
        }
    }

    /// `pool_concave` leaves a concave sequence as it is, bit for bit, and makes any other concave with the
    /// same end values and the same total: the Triplet's last three intervals rise in secant slope.
    #[test]
    fn concave_pooling_changes_only_what_is_not_concave() {
        let radii: [f64; RADII] = std::array::from_fn(|k| k as f64 / 8.0);
        let concave = [1.0, 1.0, 0.983, 0.954, 0.914, 0.843, 0.717, 0.55, 0.353];
        assert_eq!(pool_concave(&radii, &concave), concave);
        let triplet = [1.0, 0.988, 0.913, 0.817, 0.676, 0.514, 0.345, 0.187, 0.042];
        let pooled = pool_concave(&radii, &triplet);
        assert_eq!(pooled[0], triplet[0]);
        assert_eq!(pooled[8], triplet[8]);
        let secants: Vec<f64> = pooled.windows(2).map(|p| p[1] - p[0]).collect();
        for w in secants.windows(2) {
            assert!(w[1] <= w[0] + 1e-15, "not concave: {secants:?}");
        }
        // Only the interior of the pooled run moved (nodes 5 to 7 of this sequence: the run starts at node 4).
        for k in 0..5 {
            assert_eq!(pooled[k], triplet[k], "node {k}");
        }
        for k in 5..8 {
            assert!((pooled[k] - triplet[k]).abs() < 0.02, "node {k}");
        }
    }

    /// `concave_piece` passes through both ends with the given slopes and has a slope that never rises, for
    /// the three cases (the secant below, at and above the mean of the end slopes) and the straight run.
    #[test]
    fn concave_pieces_hit_their_ends_and_never_bend_back() {
        // Over h = 0.5: the secant below the mean of the end slopes, at it, above it, a straight run, a steep
        // end, and a near straight one.
        for (y0, y1, d0, d1) in [
            (1.0, 0.9, 0.0, -0.3),
            (1.0, 0.925, 0.0, -0.3),
            (1.0, 0.95, 0.0, -0.3),
            (1.0, 0.9, -0.2, -0.2),
            (1.0, 0.8, -0.05, -1.0),
            (0.9, 0.15, -1.4, -1.6),
            (0.5, 0.2, -0.1, -0.7),
        ] {
            let h = 0.5;
            let m = (y1 - y0) / h;
            assert!(
                d0 >= m - 1e-12 && m >= d1 - 1e-12,
                "the case is a valid one"
            );
            assert!((concave_piece(y0, y1, d0, d1, h, 0.0) - y0).abs() < 1e-12);
            assert!((concave_piece(y0, y1, d0, d1, h, 1.0) - y1).abs() < 1e-12);
            let n = 4000;
            let v: Vec<f64> = (0..=n)
                .map(|i| concave_piece(y0, y1, d0, d1, h, i as f64 / n as f64))
                .collect();
            let ds = h / n as f64;
            assert!(((v[1] - v[0]) / ds - d0).abs() < 1e-3, "start slope");
            assert!(((v[n] - v[n - 1]) / ds - d1).abs() < 1e-3, "end slope");
            for i in 1..n {
                let (a, b) = (v[i] - v[i - 1], v[i + 1] - v[i]);
                assert!(b <= a + 1e-12, "the slope rises at {i} of {n}");
            }
        }
    }

    /// No plateau where the lens's own law is still falling: at Edges 40 and Anamorphic 100, on the long axis
    /// of a landscape frame, the gain keeps falling to the last pixel (the old law held `rho'` at 1 and the last
    /// 60 to 100 pixels of a 1500 px frame were flat). The Triplet's `T` is almost 0 there, so its gain, which
    /// cannot fall below `1 - e'`, has little left to fall.
    #[test]
    fn the_gain_does_not_flatten_towards_the_side_of_the_frame() {
        for kind in LensKind::ALL {
            let p = shipped_plan(kind, 40.0, 100.0, 1500, 1000, Axis::X);
            let geo = Geometry::new(&p, 1500, 1000, Options::default()).expect("field optics");
            let g = |i: usize| geo.pixel(i, 500).gain;
            // The slope over the last 20 px, and over 20 px 130 px in.
            let last = (g(1479) - g(1499)) / 20.0;
            let inner = (g(1349) - g(1369)) / 20.0;
            assert!(inner > 0.0, "{kind}: the gain falls towards the side");
            let share = if kind == LensKind::Triplet { 0.05 } else { 0.3 };
            assert!(
                last > share * inner,
                "{kind}: the last 20 px fall by {last:.5} a pixel, 130 px in {inner:.5}"
            );
        }
    }

    /// The box-reduced image the reference reads: texel `(bx, by)` of level `l` is the mean of the aligned block.
    fn ref_texel(img: &[f32], w: usize, h: usize, l: u32, bx: i64, by: i64, c: usize) -> f64 {
        let b = 1i64 << l;
        let mut sum = 0.0;
        for y in by * b..(by + 1) * b {
            for x in bx * b..(bx + 1) * b {
                let (xc, yc) = (
                    x.clamp(0, w as i64 - 1) as usize,
                    y.clamp(0, h as i64 - 1) as usize,
                );
                sum += f64::from(img[c * w * h + yc * w + xc]);
            }
        }
        sum / (b * b) as f64
    }

    fn ref_bilinear(img: &[f32], w: usize, h: usize, l: u32, px: f64, py: f64, c: usize) -> f64 {
        let s = f64::from(1u32 << l);
        let (u, v) = (px / s - 0.5, py / s - 0.5);
        let (i0, j0) = (u.floor(), v.floor());
        let (fx, fy) = (u - i0, v - j0);
        let (i0, j0) = (i0 as i64, j0 as i64);
        let t = |dx, dy| ref_texel(img, w, h, l, i0 + dx, j0 + dy, c);
        (1.0 - fy) * ((1.0 - fx) * t(0, 0) + fx * t(1, 0))
            + fy * ((1.0 - fx) * t(0, 1) + fx * t(1, 1))
    }

    /// The reference gather of one pixel, without the gain: the taps of `Geometry::taps` read from
    /// independently built texels.
    fn ref_gather(img: &[f32], w: usize, h: usize, geo: &Geometry, i: usize, j: usize) -> [f64; 3] {
        let k = geo.pixel(i, j);
        if k.omega == 0.0 {
            return std::array::from_fn(|c| f64::from(img[c * w * h + j * w + i]));
        }
        let level = level_for(k.omega, geo.opts.spacing_max).min(geo.lmax);
        std::array::from_fn(|c| {
            geo.taps(&k, c)
                .iter()
                .map(|(dx, dy, wt)| {
                    wt * ref_bilinear(
                        img,
                        w,
                        h,
                        level,
                        i as f64 + 0.5 + dx,
                        j as f64 + 0.5 + dy,
                        c,
                    )
                })
                .sum()
        })
    }

    /// The reference result of one pixel: each channel's gather read at its fringe shift along the axis (linear
    /// interpolation between the pixels around it, edge pixels replicated), times the gain.
    fn ref_pixel(img: &[f32], w: usize, h: usize, geo: &Geometry, i: usize, j: usize) -> [f64; 3] {
        let k = geo.pixel(i, j);
        std::array::from_fn(|c| {
            let shift = geo.fringe_shift(&k, c);
            let g = |x: i64, y: i64| {
                let (x, y) = (
                    x.clamp(0, w as i64 - 1) as usize,
                    y.clamp(0, h as i64 - 1) as usize,
                );
                ref_gather(img, w, h, geo, x, y)[c]
            };
            let v = if shift == 0.0 {
                g(i as i64, j as i64)
            } else {
                let (along, other) = match geo.axis {
                    Axis::X => (i, j),
                    Axis::Y => (j, i),
                };
                let pos = along as f64 + shift;
                let i0 = pos.floor();
                let f = pos - i0;
                let at = |q: i64| match geo.axis {
                    Axis::X => g(q, other as i64),
                    Axis::Y => g(other as i64, q),
                };
                let (a, b) = (at(i0 as i64), at(i0 as i64 + 1));
                a + f * (b - a)
            };
            v * k.gain
        })
    }

    fn check_against_reference(p: &LensPlan, w: usize, h: usize, opts: Options, tol: f64) {
        let data = noise(w, h, 7);
        let out = apply(
            p,
            &whole(w, h, &data),
            None,
            PixelRect::new(0, 0, w, h),
            opts,
        )
        .unwrap();
        let geo = Geometry::new(p, w, h, opts).unwrap();
        let mut compared = 0;
        let mut wide = 0;
        let mut shifted = 0;
        for j in (0..h).step_by(37).chain([0, h - 1]) {
            for i in (0..w).step_by(41).chain([0, w - 1]) {
                let want = ref_pixel(&data, w, h, &geo, i, j);
                let k = geo.pixel(i, j);
                if k.omega > 0.0 {
                    wide += 1;
                    if (0..3).any(|c| geo.fringe_shift(&k, c).abs() > 0.015) {
                        shifted += 1;
                    }
                }
                for c in 0..3 {
                    let got = f64::from(out[c * w * h + j * w + i]);
                    assert!(
                        (got - want[c]).abs() < tol,
                        "pixel ({i},{j}) channel {c}: {got} vs {}",
                        want[c]
                    );
                }
                compared += 1;
            }
        }
        assert!(
            compared > 100 && wide > 50,
            "{compared} compared, {wide} with a kernel"
        );
        let fringe = p.fringe_factor() > 1.0;
        assert_eq!(
            shifted > 20,
            fringe,
            "{shifted} pixels read at a fringe shift"
        );
    }

    #[test]
    fn gather_matches_an_independent_reference() {
        // Kernels of a few pixels, read at level 0.
        check_against_reference(
            &plan(LensKind::Petzval, 100.0, 0.0, 600, 400, Axis::X),
            600,
            400,
            Options::default(),
            3e-5,
        );
    }

    #[test]
    fn gather_matches_the_reference_with_the_attachment() {
        // Anamorphic 100: oval radius, side softness and the fringe's per-channel shifts.
        check_against_reference(
            &plan(LensKind::Vintage, 60.0, 100.0, 600, 400, Axis::X),
            600,
            400,
            Options::default(),
            3e-5,
        );
        check_against_reference(
            &plan(LensKind::Triplet, 0.0, 100.0, 1500, 1000, Axis::Y),
            1500,
            1000,
            Options::default(),
            3e-5,
        );
    }

    #[test]
    fn gather_matches_the_reference_on_reduced_levels() {
        // A tight spacing makes even small kernels read the 2x, 4x and 8x copies.
        let opts = Options { spacing_max: 0.45 };
        let p = plan(LensKind::Petzval, 100.0, 40.0, 600, 400, Axis::X);
        let geo = Geometry::new(&p, 600, 400, opts).unwrap();
        assert!(geo.lmax >= 2, "lmax {}", geo.lmax);
        check_against_reference(&p, 600, 400, opts, 3e-5);
    }

    #[test]
    fn wide_kernels_use_reduced_levels_at_full_budget() {
        let p = plan(LensKind::Petzval, 100.0, 0.0, 6000, 4000, Axis::X);
        let geo = Geometry::new(&p, 6000, 4000, Options::default()).unwrap();
        // w_max = 0.012 half-diagonals of 3606 px is 43 px; level 3 brings its cells within a texel.
        assert_eq!(geo.lmax, 3);
        assert!(geo.halo() > 50 && geo.halo() < 80, "halo {}", geo.halo());
    }

    #[test]
    fn small_kernels_are_a_copy_times_the_gain() {
        let p = plan(LensKind::Default, 50.0, 0.0, 600, 400, Axis::X);
        let data = noise(600, 400, 3);
        let out = apply(
            &p,
            &whole(600, 400, &data),
            None,
            PixelRect::new(290, 190, 20, 20),
            Options::default(),
        )
        .unwrap();
        let geo = Geometry::new(&p, 600, 400, Options::default()).unwrap();
        for j in 190..210 {
            for i in 290..310 {
                let k = geo.pixel(i, j);
                assert_eq!(
                    k.omega, 0.0,
                    "centre kernels of Edges 50 are below half a pixel"
                );
                for c in 0..3 {
                    let want = (f64::from(data[c * 600 * 400 + j * 600 + i]) * k.gain) as f32;
                    assert_eq!(out[c * 400 + (j - 190) * 20 + (i - 290)], want);
                }
            }
        }
    }

    /// A whole frame of `w` x `h` pixels.
    fn whole_image(w: usize, h: usize, data: &[f32]) -> ImageRef<'_> {
        ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data,
        }
    }

    /// A smooth picture of 0.3 to 1 with a lit patch of the extra light `value` at `(cx, cy)`, radius `r`, and the
    /// extra light alone.
    fn picture_and_extra(
        w: usize,
        h: usize,
        cx: f64,
        cy: f64,
        r: f64,
        value: f64,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = w * h;
        let mut data = vec![0.0f32; 3 * n];
        let mut extra = vec![0.0f32; 3 * n];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let t = (x as f64 / w as f64 + y as f64 / h as f64) * 0.5;
                    data[c * n + y * w + x] = (0.3 + 0.7 * t) as f32;
                    let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                    if dx * dx + dy * dy <= r * r {
                        extra[c * n + y * w + x] = value as f32;
                    }
                }
            }
        }
        (data, extra)
    }

    #[test]
    fn an_extra_light_where_the_lens_changes_nothing_leaves_the_picture_as_it_is() {
        // A flat extra light of 1 % of a smooth picture, Vintage with Edges 100: where the spread changes the light of
        // a pixel by under the ramp's start (every pixel but those at the frame's border, where the replicated
        // edge bends the gradient) the output is the picture's alone, bit for bit, and anywhere it adds at most
        // the extra's own 1 %.
        let (w, h) = (600, 400);
        let p = plan(LensKind::Vintage, 100.0, 0.0, w, h, Axis::X);
        let (data, _) = picture_and_extra(w, h, 0.0, 0.0, 0.0, 0.0);
        let extra: Vec<f32> = data.iter().map(|v| v * 0.01).collect();
        let rect = PixelRect::new(0, 0, w, h);
        let plain = apply(
            &p,
            &whole_image(w, h, &data),
            None,
            rect,
            Options::default(),
        )
        .unwrap();
        let with = apply(
            &p,
            &whole_image(w, h, &data),
            Some(&whole_image(w, h, &extra)),
            rect,
            Options::default(),
        )
        .unwrap();
        let n = w * h;
        let mut inner = 0usize;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let i = c * n + y * w + x;
                    assert!(
                        with[i] >= plain[i] && with[i] <= plain[i] * 1.0101,
                        "sample {i}"
                    );
                    if (120..w - 120).contains(&x) && (120..h - 120).contains(&y) {
                        assert_eq!(
                            with[i].to_bits(),
                            plain[i].to_bits(),
                            "sample {i} at ({x}, {y})"
                        );
                        inner += 1;
                    }
                }
            }
        }
        assert!(inner > n / 2);
    }

    #[test]
    fn an_isolated_extra_light_reaches_the_pixels_round_it_and_not_the_ones_without_a_kernel() {
        // A star's headroom: 30 clip levels on a disc of 1.5 px near the right edge, where the spread is wide,
        // and the same at the centre, where the kernels are a copy. The first reaches its neighbours and
        // adds up to the light it holds times the weight the lens gives them; the second changes nothing.
        let (w, h) = (600, 400);
        let rect = PixelRect::new(0, 0, w, h);
        let n = w * h;
        for (kind, edges, cx, cy, wide) in [
            (LensKind::Vintage, 100.0, 520.0, 200.0, true),
            (LensKind::Default, 50.0, 300.0, 200.0, false),
        ] {
            let p = plan(kind, edges, 0.0, w, h, Axis::X);
            let geo = Geometry::new(&p, w, h, Options::default()).unwrap();
            let (data, extra) = picture_and_extra(w, h, cx, cy, 1.5, 30.0);
            let plain = apply(
                &p,
                &whole_image(w, h, &data),
                None,
                rect,
                Options::default(),
            )
            .unwrap();
            let with = apply(
                &p,
                &whole_image(w, h, &data),
                Some(&whole_image(w, h, &extra)),
                rect,
                Options::default(),
            )
            .unwrap();
            let (mut added, mut changed) = (0.0f64, 0usize);
            for i in 0..3 * n {
                assert!(with[i] >= plain[i], "the extra light only adds");
                let d = f64::from(with[i] - plain[i]);
                added += d;
                changed += usize::from(d > 0.0);
            }
            let (ix, iy) = (cx as usize, cy as usize);
            if wide {
                assert!(geo.pixel(ix, iy).omega > 0.0);
                assert!(
                    changed > 50,
                    "the star reaches its neighbours: {changed} samples"
                );
                // Over three channels: the light of the disc (9 pixels at 30) spread by the lens's weights, of
                // which what the share leaves out near the disc is a fraction.
                assert!(added > 0.2 * 3.0 * 9.0 * 30.0, "added {added}");
                assert!(added < 1.01 * 3.0 * 9.0 * 30.0 * 1.2, "added {added}");
            } else {
                assert_eq!(geo.pixel(ix, iy).omega, 0.0);
                assert_eq!(added, 0.0, "a copy is given none");
            }
        }
    }

    #[test]
    fn the_occupancy_tells_a_window_with_light_from_one_without() {
        let (w, h) = (40, 30);
        let mut data = vec![0.0f32; 3 * w * h];
        data[2 * w * h + 12 * w + 25] = 0.5;
        let img = ImageRef {
            width: w,
            height: h,
            x0: 10,
            y0: 100,
            frame_width: 200,
            frame_height: 200,
            data: &data,
        };
        let occ = Occupancy::build(&img);
        // The pixel is at frame (35, 112).
        assert!(occ.any(35, 112, 36, 113));
        assert!(occ.any(0, 0, 200, 200));
        assert!(occ.any(30, 108, 40, 118));
        assert!(!occ.any(36, 112, 60, 113));
        assert!(!occ.any(10, 100, 35, 130));
        assert!(!occ.any(35, 113, 36, 130));
        assert!(!occ.any(-50, -50, 5, 5), "outside the image");
    }

    #[test]
    fn mean_tap_count_is_reported() {
        // Not an assertion about speed: a guard that the tap budget holds (at most 17 x 17 a pixel).
        let p = plan(LensKind::Default, 100.0, 0.0, 3000, 2000, Axis::X);
        let geo = Geometry::new(&p, 3000, 2000, Options::default()).unwrap();
        let (mut total, mut count) = (0usize, 0usize);
        for j in (0..2000).step_by(10) {
            for i in (0..3000).step_by(10) {
                let k = geo.pixel(i, j);
                if k.omega > 0.0 {
                    let n = taps_for(k.omega / f64::from(1u32 << level_for(k.omega, 1.0)));
                    assert!(n <= MAX_TAPS_SIDE);
                    total += n * n;
                }
                count += 1;
            }
        }
        println!("mean taps per pixel: {:.1}", total as f64 / count as f64);
        assert!(total as f64 / (count as f64) <= (MAX_TAPS_SIDE * MAX_TAPS_SIDE) as f64);
    }
}
