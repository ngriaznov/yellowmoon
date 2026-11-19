//! Method A of `docs/algorithms/05-single-frame-denoise.md`: block matching and 3D DCT hard thresholding.
//!
//! The CPU reference of the method, with the repository's reading of the text (items G1 to G5 and
//! A-a to A-d of that file).
//!
//! # Use
//!
//! ```
//! use ym_gpu::cpu::RgbImage;
//! use ym_gpu::cpu::bm_dct::{self, BmDctParams};
//!
//! let img = RgbImage::new(32, 24);
//! let p = BmDctParams { luma: 30.0, chroma: 30.0, radius: 1, blend: 0.0, level: None };
//! let out = bm_dct::denoise(&img, &p);
//! assert_eq!((out.width, out.height), (32, 24));
//! ```
//!
//! # Maths
//!
//! The image goes to BT.709 Y/Cb/Cr ([`ycbcr::to_ycbcr`]); everything after is `f64`. From the
//! strengths, `τY = 0.1 L`, `τC = 0.06 C`, `θ = 0.2 max(L, C)` and the reference step
//! `s = 8 − 2R` ([`BmDctParams`]). Y is filtered when `τY > 1e-6`; Cb and Cr when `τC > 1e-6`, or,
//! with the photo's noise given ([`LevelNoise`]), when it can make a threshold above 0. A plane that
//! is not filtered is the input plane. For every reference block (8 × 8):
//!
//! 1. **Candidates**: the 25 blocks at `ref + (dx, dy)`, `dx, dy ∈ {−8, −4, 0, 4, 8}`, numbered
//!    `5 · row + column` (12 is the reference). A block not wholly inside the frame is not a candidate.
//! 2. **Distance** on Y: `D = Σ K(i, j) · |Y_ref − Y_cand|`, with `K = k ⊗ k` the Kaiser window of
//!    [`kaiser`] (N = 8, β = 2), summed in raster order.
//! 3. **Group**: slot 0 is the reference; slots 1 to 7 take the 7 other candidates of smallest `D`
//!    (a tie goes to the smaller index); a slot whose `D > θ`, or for which no candidate is left,
//!    takes the reference block. Cb and Cr use the group found on Y.
//! 4. **3D transform** per filtered channel: an 8-point DCT-II along the rows, then the columns of
//!    each block, then across the 8 blocks. Each 1D pass is the orthonormal DCT-II times √8, so the
//!    DC term is the plain sum and a forward and inverse round trip gains `8³ = 512`.
//! 5. **Hard threshold**: a coefficient with `|c| < τ` becomes 0, the DC term included. The
//!    optional highlight tweak of the text is omitted (reading A-b). With [`LevelNoise`], Cb and Cr
//!    take the threshold of their reference block, `τC = chroma_scale · σC(m)`, where `m` is the
//!    reference block's mean Y, Cb and Cr (each summed in raster order) and `σC` the chroma noise
//!    of the working encoding at that colour ([`LevelNoise::chroma_sigma`]): the encoding stretches
//!    the shadows, so one threshold set at middle grey leaves colour noise in dark colours.
//! 6. **Inverse** 3D transform (across the blocks, then columns, then rows).
//! 7. **Aggregation**: for every pixel of every slot, `acc_ch += K(i, j) · v / 512` and
//!    `w += K(i, j)` (once per pixel, not per channel). The output is `acc_ch / w`.
//!
//! For `B > 0` the output's Y is blended with the input's, `d + (x − d) B / 100`; Cb and Cr keep
//! their filtered values, so the blend brings back fine luma texture and no colour noise. The
//! result goes back to RGB ([`ycbcr::from_ycbcr`]). When neither channel is filtered, or the frame
//! is narrower or lower than 8 px, the input comes back bit for bit.
//!
//! # Grid and edges (reading A-a, G4)
//!
//! Reference corners are the multiples of `s` from 0 up to `W − 8` of the *frame*, plus `W − 8`
//! when it is not one of them (the same for y), so every pixel of the frame is covered. The grid
//! is anchored to the frame's origin ([`RgbImage::place`]), never to the buffer's, and the
//! candidate test uses the frame's size.
//!
//! # Order and determinism (reading A-c, G3)
//!
//! References are processed in the passes of A.5, which never overlap: the regular grid in passes
//! `(i, j)` (references at `(24 m + i s, 24 n + j s)`) in raster order, `j` the outer loop and `i`
//! the inner one; then the extra column `x = W − 8` in passes `j` (`y = 24 n + j s`); then the extra
//! row `y = H − 8` in passes `i` (`x = 24 m + i s`); then the corner `(W − 8, H − 8)`. Two
//! references of one pass are at least 24 px apart and a reference writes only inside its 24 × 24
//! footprint, so a pixel receives at most one reference per pass, and the slots of a group are
//! added in slot order: each pixel's sums have one fixed order. Inside a pass the rows of
//! references run in parallel (rayon), each into its own band of the accumulators, so the bits do
//! not depend on the number of threads. The DCT's cosine table comes from the `libm` crate, not the
//! platform's library, so they do not depend on the platform either.
//!
//! # Receptive field and tiles (reading A-d)
//!
//! A reference at `r` reads the pixels `[r − 8, r + 16)` (its candidates, clipped to the frame)
//! and writes the blocks of its group, inside the same span. A pixel `p` is written by the
//! references in `[p − 15, p + 8]`, which read `[p − 23, p + 23]`: the halo is [`HALO_PX`] = 23 px
//! on each side. A buffer that is a window of the frame computes only the references whose read
//! span (clipped to the frame) lies inside the buffer; the others touch only pixels within 23 px
//! of a buffer edge that is not a frame edge, which the caller drops. The pixels it keeps equal the
//! whole frame's bit for bit (tested): each reference is computed from the same samples in the same
//! order, and the skipped references never reach them. A pixel no reference reached (only in that
//! dropped margin) keeps its input value.
//!
//! Inputs are not clamped (reading G1); they must be finite. The working memory is the three
//! Y/Cb/Cr planes and four accumulator planes of the buffer, all `f64`.

use std::array::from_fn;
use std::f64::consts::{PI, SQRT_2};

use rayon::prelude::*;

use super::denoise_space::{srgb_decode, srgb_slope};
use super::ycbcr::{self, CB_DIV, CR_DIV, KB, KG, KR, Planes};
use super::{Placement, RgbImage};

/// The halo of method A in pixels (reading A-d): a tile needs this much context on every side
/// that is not a frame edge for its inner pixels to equal the whole frame's.
pub const HALO_PX: usize = 23;

/// Side of a block in pixels.
const B: usize = 8;
/// Number of candidate blocks per reference.
const CANDIDATES: usize = 25;
/// Index of the reference among the candidates.
const REF: usize = 12;
/// Farthest a candidate's corner lies from its reference's, in pixels.
const REACH: usize = 8;
/// Side of a reference's footprint, and the spacing of the references of one pass (A.5).
const PASS_PX: usize = 24;
/// A threshold at or below this leaves its channels unfiltered (A.1).
const SKIP_TAU: f64 = 1e-6;
/// Gain of a forward and inverse 3D transform, `8³`.
const ROUND_TRIP_GAIN: f64 = 512.0;
/// Largest radius the text defines (`R ∈ {0, 1, 2, 3}`).
const MAX_RADIUS: usize = 3;

/// An 8 × 8 block, rows of 8 samples.
type Block = [[f64; B]; B];
/// The 8 blocks of a group, in slot order.
type Group = [Block; 8];

/// Strength inputs of method A (spec section 0 and A.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BmDctParams {
    /// Luma strength `L`, in [0, 100].
    pub luma: f64,
    /// Chroma strength `C`, in [0, 100].
    pub chroma: f64,
    /// Radius `R`, 0 to 3; a larger value is read as 3 (the step `8 − 2R` must stay positive).
    pub radius: usize,
    /// Blend `B` with the input, in [0, 100]; 0 means no blend.
    pub blend: f64,
    /// The photo's noise, for a chroma threshold that follows each reference block's level
    /// ([`LevelNoise`]); `None` keeps the one threshold `τC = 0.06 C`.
    pub level: Option<LevelNoise>,
}

/// The photo's noise in the working encoding at any level, for method A's chroma threshold: the
/// threshold of a reference block is `τC = chroma_scale · σC`, where `σC` is the chroma noise at the
/// block's mean colour. A balanced channel `c` at linear level `v` has the variance
/// `g_c a v + g_c² b`; the encoding `srgb(G v)` scales its standard deviation by `G srgb'(G v)`;
/// Cb and Cr weigh the channels (taken as independent) and `σC = √((σCb² + σCr²) / 2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelNoise {
    /// `a` of the noise model (variance `a x + b` on the normalised raw scale), at least 0.
    pub a: f64,
    /// `b` of the noise model, at least 0.
    pub b: f64,
    /// The R4 gains `g_c`.
    pub gains: [f64; 3],
    /// The encoding's gain `G = 2^WORKING_GAIN_EV`.
    pub encoding_gain: f64,
    /// `τC / σC`: the threshold in standard deviations times the transform's scale `8^1.5`.
    pub chroma_scale: f64,
}

impl LevelNoise {
    /// The chroma noise `σC` in the working encoding at the colour whose Y/Cb/Cr (working encoding)
    /// is `ycc`.
    #[must_use]
    pub fn chroma_sigma(&self, ycc: [f64; 3]) -> f64 {
        let [y, cb, cr] = ycc;
        let r = y + CR_DIV * cr;
        let b = y + CB_DIV * cb;
        let g = (y - KR * r - KB * b) / KG;
        let gain = self.encoding_gain;
        let var: [f64; 3] = from_fn(|c| {
            let e = [r, g, b][c];
            let v = (srgb_decode(e) / gain).max(0.0);
            let s = gain * srgb_slope(gain * v);
            let gc = self.gains[c];
            (gc * self.a * v + gc * gc * self.b) * s * s
        });
        let weigh =
            |w: [f64; 3]| w[0] * w[0] * var[0] + w[1] * w[1] * var[1] + w[2] * w[2] * var[2];
        let (wb, wr) = chroma_weights();
        let vb = weigh(wb);
        let vr = weigh(wr);
        ((vb + vr) / 2.0).sqrt()
    }

    /// Whether the threshold can be above 0 anywhere.
    fn filters(&self) -> bool {
        self.chroma_scale > 0.0 && (self.a > 0.0 || self.b > 0.0)
    }
}

/// The weights of R, G and B in Cb and in Cr (BT.709, [`ycbcr::to_ycbcr`]), for [`LevelNoise::chroma_sigma`]
/// and the device kernel (`kernels::bm_dct`), which writes their squares into its shader.
pub(crate) fn chroma_weights() -> ([f64; 3], [f64; 3]) {
    (
        [-KR / CB_DIV, -KG / CB_DIV, (1.0 - KB) / CB_DIV],
        [(1.0 - KR) / CR_DIV, -KG / CR_DIV, -KB / CR_DIV],
    )
}

impl BmDctParams {
    /// The hard threshold of luma, `τY = 0.1 L`.
    pub fn luma_threshold(&self) -> f64 {
        0.1 * self.luma
    }

    /// The hard threshold of chroma, `τC = 0.06 C`.
    pub fn chroma_threshold(&self) -> f64 {
        0.06 * self.chroma
    }

    /// The block-distance threshold, `θ = 0.2 max(L, C)`.
    pub fn distance_threshold(&self) -> f64 {
        0.2 * self.luma.max(self.chroma)
    }

    /// The reference step, `s = 8 − 2R` px, with `R` above 3 read as 3.
    pub fn step_px(&self) -> usize {
        B - 2 * self.radius.min(MAX_RADIUS)
    }
}

/// The Kaiser window of A.2, `k(n) = I0(2 √(1 − (2n/7 − 1)²)) / I0(2)`, from the Bessel series.
///
/// `[0.4387, 0.6813, 0.8768, 0.9858, 0.9858, 0.8768, 0.6813, 0.4387]`; the 2D window is
/// `K(i, j) = k(i) k(j)` and sums to about 35.59. `2n/7 − 1` is computed as `(2n − 7) / 7`, so the
/// window is exactly symmetric.
pub fn kaiser() -> [f64; 8] {
    let i0_beta = bessel_i0(2.0);
    from_fn(|n| {
        let t = (2.0 * n as f64 - 7.0) / 7.0;
        bessel_i0(2.0 * (1.0 - t * t).max(0.0).sqrt()) / i0_beta
    })
}

/// The modified Bessel function of the first kind and order 0, `I0(x) = Σ ((x/2)^k / k!)²`, for the
/// small arguments of [`kaiser`] (`|x| ≤ 2`, where 30 terms are far past `f64` precision).
fn bessel_i0(x: f64) -> f64 {
    let q = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=30u32 {
        term *= q / f64::from(k * k);
        sum += term;
    }
    sum
}

/// The 2D window `K(i, j) = k(i) k(j)`.
fn window_2d() -> Block {
    let k = kaiser();
    from_fn(|i| from_fn(|j| k[i] * k[j]))
}

/// The 8-point DCT-II of A.3 step 4, unnormalised: the orthonormal transform times √8.
///
/// `X[k] = c_k Σ_n x[n] cos(π (2n + 1) k / 16)` with `c_0 = 1` and `c_k = √2`, so `X[0]` is the plain
/// sum. The inverse is the transpose of that matrix, and `inverse(forward(x)) = 8 x`. Computed with
/// the even and odd halves (`x[n] ± x[7 − n]`), 32 multiplications a transform.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dct8 {
    /// `even[m][n]`: row `2m` of the matrix, columns 0 to 3.
    even: [[f64; 4]; 4],
    /// `odd[m][n]`: row `2m + 1` of the matrix, columns 0 to 3.
    odd: [[f64; 4]; 4],
}

impl Dct8 {
    /// The cosine table. `libm::cos` keeps it the same on every platform.
    pub(crate) fn new() -> Dct8 {
        let m = |k: usize, n: usize| {
            let c = if k == 0 { 1.0 } else { SQRT_2 };
            c * libm::cos(PI * (2 * n + 1) as f64 * k as f64 / 16.0)
        };
        Dct8 {
            even: from_fn(|a| from_fn(|n| m(2 * a, n))),
            odd: from_fn(|a| from_fn(|n| m(2 * a + 1, n))),
        }
    }

    /// The forward transform of 8 samples.
    #[inline]
    pub(crate) fn forward(&self, x: &[f64; 8]) -> [f64; 8] {
        let sum: [f64; 4] = from_fn(|n| x[n] + x[7 - n]);
        let diff: [f64; 4] = from_fn(|n| x[n] - x[7 - n]);
        let dot = |row: &[f64; 4], v: &[f64; 4]| {
            row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]
        };
        let mut out = [0.0; 8];
        for a in 0..4 {
            out[2 * a] = dot(&self.even[a], &sum);
            out[2 * a + 1] = dot(&self.odd[a], &diff);
        }
        out
    }

    /// The inverse transform of 8 coefficients (the transpose: `inverse(forward(x)) = 8 x`).
    #[inline]
    pub(crate) fn inverse(&self, c: &[f64; 8]) -> [f64; 8] {
        let (e, o) = (&self.even, &self.odd);
        let mut out = [0.0; 8];
        for n in 0..4 {
            let even = e[0][n] * c[0] + e[1][n] * c[2] + e[2][n] * c[4] + e[3][n] * c[6];
            let odd = o[0][n] * c[1] + o[1][n] * c[3] + o[2][n] * c[5] + o[3][n] * c[7];
            out[n] = even + odd;
            out[7 - n] = even - odd;
        }
        out
    }

    /// The 2D forward transform of a block in place: rows, then columns.
    fn forward_2d(&self, b: &mut Block) {
        for row in b.iter_mut() {
            *row = self.forward(row);
        }
        for j in 0..B {
            let col = self.forward(&from_fn(|i| b[i][j]));
            for (row, v) in b.iter_mut().zip(col) {
                row[j] = v;
            }
        }
    }

    /// The 2D inverse transform of a block in place: columns, then rows.
    fn inverse_2d(&self, b: &mut Block) {
        for j in 0..B {
            let col = self.inverse(&from_fn(|i| b[i][j]));
            for (row, v) in b.iter_mut().zip(col) {
                row[j] = v;
            }
        }
        for row in b.iter_mut() {
            *row = self.inverse(row);
        }
    }

    /// The 3D forward transform of a group in place: rows, columns, then across the 8 blocks.
    pub(crate) fn forward_3d(&self, g: &mut Group) {
        for b in g.iter_mut() {
            self.forward_2d(b);
        }
        self.across(g, Dct8::forward);
    }

    /// The 3D inverse transform of a group in place: across the blocks, then columns, then rows.
    /// `inverse_3d(forward_3d(g)) = 512 g`.
    pub(crate) fn inverse_3d(&self, g: &mut Group) {
        self.across(g, Dct8::inverse);
        for b in g.iter_mut() {
            self.inverse_2d(b);
        }
    }

    /// Applies `f` to the 8 samples at each position across the blocks of a group.
    fn across(&self, g: &mut Group, f: impl Fn(&Dct8, &[f64; 8]) -> [f64; 8]) {
        for i in 0..B {
            for j in 0..B {
                let line = f(self, &from_fn(|k| g[k][i][j]));
                for (b, v) in g.iter_mut().zip(line) {
                    b[i][j] = v;
                }
            }
        }
    }
}

/// Half width of the soft threshold's band, relative to `tau` (RC-15, `docs/algorithms/05` A-f): a
/// coefficient below `tau (1 - SOFT_BAND)` is zeroed, one above `tau (1 + SOFT_BAND)` is kept, and
/// in between it is kept by the smoothstep of its place in the band. The device forms a coefficient
/// with a relative error of a few 1e-7 of the group's samples, so a band of 2 % turns what was a
/// jump of the whole coefficient into a change of about 1e-5 of it.
pub(crate) const SOFT_BAND: f64 = 0.02;

/// The weight a coefficient of magnitude `a` keeps at threshold `tau`.
pub(crate) fn keep_weight(a: f64, tau: f64) -> f64 {
    let e = SOFT_BAND;
    let (lo, hi) = (tau * (1.0 - e), tau * (1.0 + e));
    if a <= lo {
        0.0
    } else if a >= hi {
        1.0
    } else {
        let t = (a - lo) / (hi - lo);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Corner offset `(dx, dy)` of candidate `idx` from its reference.
const fn offset(idx: usize) -> (isize, isize) {
    ((idx % 5) as isize * 4 - 8, (idx / 5) as isize * 4 - 8)
}

/// The part of the half width of the band, in units of the distance `D`, that does not depend on `D`
/// (RC-15, `docs/algorithms/05` A-g). `D` is a sum of 64 window-weighted differences of the encoded
/// luma, which the device forms in `f32` and this file in `f64`.
pub(crate) const GROUP_BAND_ABS: f64 = 1e-3;
/// The part of the half width that is a share of the larger of the two values compared.
pub(crate) const GROUP_BAND_REL: f64 = 5e-4;

/// The key of a register no candidate has reached (`+∞` on this side, a large finite number on the
/// device).
const EMPTY_KEY: f64 = f64::INFINITY;

/// Weights of the 26 sources of a register: the 25 candidates and, last, "nothing yet".
const SOURCES: usize = CANDIDATES + 1;

/// How far a compare-exchange of two keys `diff` apart goes, `scale` the larger of the two: 0 for
/// `diff <= -h`, 1 for `diff >= h`, the smoothstep of the position between (0.5 at 0), with the half
/// width `h = ` [`GROUP_BAND_ABS`] ` + ` [`GROUP_BAND_REL`] ` * scale`.
pub(crate) fn soft_step(diff: f64, scale: f64) -> f64 {
    let h = GROUP_BAND_ABS + GROUP_BAND_REL * scale;
    if diff >= h {
        1.0
    } else if diff <= -h {
        0.0
    } else {
        let t = (diff + h) / (2.0 * h);
        t * t * (3.0 - 2.0 * t)
    }
}

/// The weights of the group: row `k` holds, for slot `k`, how much of each candidate block the slot's
/// block is made of (`docs/algorithms/05` A-g). Slot 0 is the reference. `d` holds the candidates'
/// distances (`+∞` for a block that is not a candidate), `theta` the distance threshold.
///
/// The 7 nearest candidates in order of distance are found by an insertion into seven registers
/// whose compare-exchange is soft: a candidate whose distance is within the band ([`soft_step`]) of a
/// register's takes a share of it and leaves the rest, so the weights are continuous in the
/// distances, and two candidates at the same distance share their slots equally whatever the
/// rounding of either. Away from a tie every row has one weight of 1 and the others 0 and the
/// slots are the 7 candidates of smallest distance in order. A slot whose distance is above `theta`
/// (within the band: a share of it) or that no candidate reaches is the reference.
fn select(d: &[f64; CANDIDATES], theta: f64) -> [[f64; CANDIDATES]; 8] {
    // Vector `v` of a register: its weights over the sources. Register `q` is `vec[id[q]]`; `id[7]`
    // is the carry, the candidate that is being inserted (or what it has exchanged for).
    let mut vecs = [[0.0f64; SOURCES]; 8];
    let mut id: [usize; 8] = from_fn(|q| q);
    let mut key = [EMPTY_KEY; 7];
    for v in &mut vecs[..7] {
        v[CANDIDATES] = 1.0;
    }
    for (idx, &di) in d.iter().enumerate() {
        if idx == REF || !di.is_finite() {
            continue;
        }
        vecs[id[7]] = [0.0; SOURCES];
        vecs[id[7]][idx] = 1.0;
        let mut kc = di;
        for q in 0..7 {
            if kc == EMPTY_KEY {
                break;
            }
            let kr = key[q];
            let s = if kr == EMPTY_KEY {
                1.0
            } else {
                soft_step(kr - kc, kr.max(kc))
            };
            if s == 0.0 {
                continue;
            }
            if s == 1.0 {
                id.swap(q, 7);
                key[q] = kc;
                kc = kr;
                continue;
            }
            let om = 1.0 - s;
            let (rv, cv) = (id[q], id[7]);
            let (a, b) = if rv < cv {
                let (lo, hi) = vecs.split_at_mut(cv);
                (&mut lo[rv], &mut hi[0])
            } else {
                let (lo, hi) = vecs.split_at_mut(rv);
                (&mut hi[0], &mut lo[cv])
            };
            for (x, y) in a.iter_mut().zip(b.iter_mut()) {
                (*x, *y) = (om * *x + s * *y, om * *y + s * *x);
            }
            (key[q], kc) = (om * kr + s * kc, om * kc + s * kr);
        }
    }
    let mut w = [[0.0f64; CANDIDATES]; 8];
    w[0][REF] = 1.0;
    for q in 0..7 {
        let v = &vecs[id[q]];
        let u = if key[q] == EMPTY_KEY {
            1.0
        } else {
            soft_step(key[q] - theta, key[q].max(theta))
        };
        let keep = 1.0 - u;
        for (i, wi) in w[q + 1].iter_mut().enumerate() {
            *wi = keep * v[i];
        }
        w[q + 1][REF] = u + keep * v[CANDIDATES];
    }
    w
}

/// Per-worker scratch of one reference.
struct Scratch {
    /// The blocks the slots are made of: slot, buffer position (top-left) and weight, slots in order
    /// and candidates in index order inside a slot (the order of the device's sums); `len` of them.
    parts: [(usize, usize, usize, f64); 8 * CANDIDATES],
    /// How many of `parts` are in use.
    len: usize,
    /// The group of each channel (Y, Cb, Cr); after [`Ctx::filter`], the filtered blocks times 512.
    groups: [Group; 3],
}

impl Scratch {
    /// Zeroed scratch on the heap (12 KiB).
    fn boxed() -> Box<Scratch> {
        Box::new(Scratch {
            parts: [(0, 0, 0, 0.0); 8 * CANDIDATES],
            len: 0,
            groups: [[[[0.0; B]; B]; 8]; 3],
        })
    }
}

/// Everything one run needs: the Y/Cb/Cr planes of the buffer and the derived parameters.
struct Ctx {
    /// `[Y, Cb, Cr]` of the buffer.
    planes: Planes,
    /// Buffer width in pixels.
    width: usize,
    /// Buffer height in pixels.
    height: usize,
    /// The buffer's place in the frame.
    place: Placement,
    /// `K(i, j)`.
    window: Block,
    /// The cosine table.
    dct: Dct8,
    /// Hard threshold per channel.
    tau: [f64; 3],
    /// Whether each channel is filtered.
    active: [bool; 3],
    /// The noise for a chroma threshold per reference block, if it follows the level.
    level: Option<LevelNoise>,
    /// Block-distance threshold.
    theta: f64,
    /// Reference step in pixels.
    step: usize,
}

impl Ctx {
    /// Whether the method changes anything: a frame of at least 8 × 8 px and a channel to filter.
    fn runs(place: &Placement, p: &BmDctParams) -> bool {
        place.frame_width >= B && place.frame_height >= B && Ctx::active(p).contains(&true)
    }

    /// Whether Y, Cb and Cr are filtered.
    fn active(p: &BmDctParams) -> [bool; 3] {
        let chroma = match &p.level {
            Some(l) => l.filters(),
            None => p.chroma_threshold() > SKIP_TAU,
        };
        [p.luma_threshold() > SKIP_TAU, chroma, chroma]
    }

    /// A context for the `[Y, Cb, Cr]` planes of a `width` × `height` buffer at `place` (a frame of
    /// at least 8 × 8 px).
    fn new(planes: Planes, width: usize, height: usize, place: Placement, p: &BmDctParams) -> Ctx {
        let tau = [
            p.luma_threshold(),
            p.chroma_threshold(),
            p.chroma_threshold(),
        ];
        Ctx {
            planes,
            width,
            height,
            place,
            window: window_2d(),
            dct: Dct8::new(),
            tau,
            active: Ctx::active(p),
            level: p.level,
            theta: p.distance_threshold(),
            step: p.step_px(),
        }
    }

    /// Whether a reference at frame column `x` reads only columns of the buffer.
    fn fits_x(&self, x: usize) -> bool {
        let lo = x.saturating_sub(REACH);
        let hi = (x + PASS_PX - REACH).min(self.place.frame_width);
        lo >= self.place.x0 && hi <= self.place.x0 + self.width
    }

    /// Whether a reference at frame row `y` reads only rows of the buffer.
    fn fits_y(&self, y: usize) -> bool {
        let lo = y.saturating_sub(REACH);
        let hi = (y + PASS_PX - REACH).min(self.place.frame_height);
        lo >= self.place.y0 && hi <= self.place.y0 + self.height
    }

    /// The passes in order (reading A-c), each the frame rows and the frame columns of its
    /// references (a pass is every pair), keeping the references the buffer can compute.
    fn passes(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let (fw, fh) = (self.place.frame_width, self.place.frame_height);
        let s = self.step;
        let phases = PASS_PX / s;
        let grid = |len: usize, phase: usize| (phase * s..=len - B).step_by(PASS_PX);
        let xs = |i: usize| -> Vec<usize> { grid(fw, i).filter(|&x| self.fits_x(x)).collect() };
        let ys = |j: usize| -> Vec<usize> { grid(fh, j).filter(|&y| self.fits_y(y)).collect() };
        let (last_x, last_y) = (fw - B, fh - B);
        let (extra_x, extra_y) = (last_x % s != 0, last_y % s != 0);
        let column: Vec<usize> = Some(last_x)
            .filter(|&x| self.fits_x(x))
            .into_iter()
            .collect();
        let row: Vec<usize> = Some(last_y)
            .filter(|&y| self.fits_y(y))
            .into_iter()
            .collect();

        let mut passes = Vec::new();
        for j in 0..phases {
            for i in 0..phases {
                passes.push((ys(j), xs(i)));
            }
        }
        if extra_x {
            for j in 0..phases {
                passes.push((ys(j), column.clone()));
            }
        }
        if extra_y {
            for i in 0..phases {
                passes.push((row.clone(), xs(i)));
            }
        }
        if extra_x && extra_y {
            passes.push((row.clone(), column.clone()));
        }
        passes.retain(|(ys, xs)| !ys.is_empty() && !xs.is_empty());
        passes
    }

    /// The accumulators `[Y, Cb, Cr, w]` of every buffer pixel after all passes.
    fn accumulate(&self) -> Vec<[f64; 4]> {
        let mut acc = vec![[0.0; 4]; self.width * self.height];
        for (ys, xs) in self.passes() {
            self.run_pass(&mut acc, &ys, &xs);
        }
        acc
    }

    /// One pass: the rows of references in parallel, each into its own band of rows.
    fn run_pass(&self, acc: &mut [[f64; 4]], ys: &[usize], xs: &[usize]) {
        let (y0, fh) = (self.place.y0, self.place.frame_height);
        // Buffer rows of each reference row's footprint: inside the buffer (`fits_y`), ascending and
        // disjoint (the rows of a pass are 24 px apart).
        let mut rest = acc;
        let mut at = 0;
        let mut bands = Vec::with_capacity(ys.len());
        for &y in ys {
            let r0 = y.saturating_sub(REACH) - y0;
            let r1 = (y + PASS_PX - REACH).min(fh) - y0;
            let (_, tail) = std::mem::take(&mut rest).split_at_mut((r0 - at) * self.width);
            let (band, tail) = tail.split_at_mut((r1 - r0) * self.width);
            bands.push((y, r0, band));
            rest = tail;
            at = r1;
        }
        bands
            .into_par_iter()
            .for_each_init(Scratch::boxed, |scratch, (y, r0, band)| {
                for &x in xs {
                    self.filter(x, y, scratch);
                    self.scatter(scratch, band, r0);
                }
            });
    }

    /// The distances of the 25 candidates of the reference at frame `(rx, ry)` (`+∞` for a block not
    /// wholly inside the frame) and their buffer positions. The reference must fit the buffer.
    fn distances(&self, rx: usize, ry: usize) -> ([f64; CANDIDATES], [(usize, usize); CANDIDATES]) {
        let (fw, fh) = (self.place.frame_width, self.place.frame_height);
        let (x0, y0) = (self.place.x0, self.place.y0);
        let y = &self.planes[0];
        let w = self.width;
        // The reference fits the buffer (`fits_x`, `fits_y`), so it and every in-frame candidate
        // lie at or after the buffer's corner.
        let (bx, by) = (rx - x0, ry - y0);
        let mut d = [f64::INFINITY; CANDIDATES];
        let mut pos = [(bx, by); CANDIDATES];
        d[REF] = 0.0;
        for idx in 0..CANDIDATES {
            if idx == REF {
                continue;
            }
            let (dx, dy) = offset(idx);
            let (Some(cx), Some(cy)) = (rx.checked_add_signed(dx), ry.checked_add_signed(dy))
            else {
                continue;
            };
            if cx + B > fw || cy + B > fh {
                continue;
            }
            let (cx, cy) = (cx - x0, cy - y0);
            let mut sum = 0.0;
            for (i, k_row) in self.window.iter().enumerate() {
                let r = &y[(by + i) * w + bx..][..B];
                let c = &y[(cy + i) * w + cx..][..B];
                for ((k, a), b) in k_row.iter().zip(r).zip(c) {
                    sum += k * (a - b).abs();
                }
            }
            d[idx] = if sum.is_nan() { f64::INFINITY } else { sum };
            pos[idx] = (cx, cy);
        }
        (d, pos)
    }

    /// The parts of the group of the reference at frame `(rx, ry)`, in `s.parts`, and the buffer
    /// position of the reference block.
    fn group(&self, rx: usize, ry: usize, s: &mut Scratch) -> (usize, usize) {
        let (d, pos) = self.distances(rx, ry);
        let weights = select(&d, self.theta);
        s.len = 0;
        for (k, row) in weights.iter().enumerate() {
            for (idx, &wt) in row.iter().enumerate() {
                if wt != 0.0 {
                    s.parts[s.len] = (k, pos[idx].0, pos[idx].1, wt);
                    s.len += 1;
                }
            }
        }
        pos[REF]
    }

    /// Groups, transforms, thresholds and inverts the blocks of the reference at frame `(rx, ry)`.
    fn filter(&self, rx: usize, ry: usize, s: &mut Scratch) {
        let pos0 = self.group(rx, ry, s);
        let w = self.width;
        let chroma_tau = self.level.map(|l| {
            // The reference block's mean Y, Cb and Cr, each summed in raster order.
            let (cx, cy) = pos0;
            let mean: [f64; 3] = from_fn(|ch| {
                let mut sum = 0.0;
                for i in 0..B {
                    for v in &self.planes[ch][(cy + i) * w + cx..][..B] {
                        sum += v;
                    }
                }
                sum / (B * B) as f64
            });
            l.chroma_scale * l.chroma_sigma(mean)
        });
        for ch in 0..3 {
            if !self.active[ch] {
                continue;
            }
            let plane = &self.planes[ch];
            let g = &mut s.groups[ch];
            for b in g.iter_mut() {
                *b = [[0.0; B]; B];
            }
            // A slot's block is the weighted sum of its parts, in index order: one part of weight 1
            // (away from a tie) is the candidate's block itself.
            for &(k, cx, cy, wt) in &s.parts[..s.len] {
                for (i, row) in g[k].iter_mut().enumerate() {
                    for (v, p) in row.iter_mut().zip(&plane[(cy + i) * w + cx..][..B]) {
                        *v += wt * p;
                    }
                }
            }
            self.dct.forward_3d(g);
            let tau = match chroma_tau {
                Some(t) if ch > 0 => t,
                _ => self.tau[ch],
            };
            for v in g.iter_mut().flatten().flatten() {
                *v *= keep_weight(v.abs(), tau);
            }
            self.dct.inverse_3d(g);
        }
    }

    /// Adds the filtered group in `s` to `band`, the accumulators of the buffer rows from `r0` on,
    /// part by part in slot order, each slot's share of a candidate's position by its weight.
    fn scatter(&self, s: &Scratch, band: &mut [[f64; 4]], r0: usize) {
        let w = self.width;
        for &(k, cx, cy, wt) in &s.parts[..s.len] {
            for (i, k_row) in self.window.iter().enumerate() {
                let out = &mut band[(cy + i - r0) * w + cx..][..B];
                for (j, (a, &kk)) in out.iter_mut().zip(k_row).enumerate() {
                    for (ch, g) in s.groups.iter().enumerate() {
                        if self.active[ch] {
                            a[ch] += wt * (kk * g[k][i][j] / ROUND_TRIP_GAIN);
                        }
                    }
                    a[3] += kk * wt;
                }
            }
        }
    }

    /// The output planes: `acc / w` for the filtered channels where a reference reached the pixel,
    /// the input elsewhere; then the blend of Y with the input, `d + (x − d) t` with `t = B / 100`
    /// (Cb and Cr keep the filtered values).
    fn finish(self, acc: &[[f64; 4]], blend: f64) -> Planes {
        let t = (blend / 100.0).min(1.0);
        let mut out = self.planes;
        for (i, a) in acc.iter().enumerate() {
            if a[3] > 0.0 {
                for (ch, plane) in out.iter_mut().enumerate() {
                    if self.active[ch] {
                        let d = a[ch] / a[3];
                        plane[i] = if ch == 0 && t > 0.0 {
                            d + (plane[i] - d) * t
                        } else {
                            d
                        };
                    }
                }
            }
        }
        out
    }
}

/// What the device kernel (`kernels::bm_dct`) takes from this reference for one buffer, so that its
/// schedule is this file's: the passes in order (each the frame rows and columns of its references, as
/// [`Ctx::passes`] makes them for the buffer), the filtered channels, the thresholds and the noise of
/// the chroma threshold per reference block.
#[derive(Clone, Debug)]
pub(crate) struct DevicePlan {
    /// The passes in order, each `(rows, columns)` in frame coordinates.
    pub(crate) passes: Vec<(Vec<usize>, Vec<usize>)>,
    /// Whether Y, Cb and Cr are filtered.
    pub(crate) active: [bool; 3],
    /// The hard threshold of each channel (Cb and Cr: the one threshold `0.06 C`, which `level`
    /// replaces when it is `Some`).
    pub(crate) tau: [f64; 3],
    /// The block-distance threshold.
    pub(crate) theta: f64,
    /// The noise of the chroma threshold per reference block ([`LevelNoise`]), if Cb and Cr take it.
    pub(crate) level: Option<LevelNoise>,
}

/// The [`DevicePlan`] of a `width` × `height` buffer at `place`, or `None` when the method changes
/// nothing there (a frame under 8 px, or no channel to filter: `denoise` returns its input).
pub(crate) fn device_plan(
    width: usize,
    height: usize,
    place: Placement,
    p: &BmDctParams,
) -> Option<DevicePlan> {
    if !Ctx::runs(&place, p) {
        return None;
    }
    let ctx = Ctx::new(
        [Vec::new(), Vec::new(), Vec::new()],
        width,
        height,
        place,
        p,
    );
    Some(DevicePlan {
        passes: ctx.passes(),
        active: ctx.active,
        tau: ctx.tau,
        theta: ctx.theta,
        level: ctx.level,
    })
}

/// The constants of this reference the device kernel writes into its shader and uniform block, so
/// that they are never typed twice: `K(i, j)`, the rows `2m` and `2m + 1` of the DCT matrix (columns 0
/// to 3, [`Dct8`]) and the gain of a forward and inverse 3D transform (512).
pub(crate) fn device_constants() -> (Block, [[f64; 4]; 4], [[f64; 4]; 4], f64) {
    let dct = Dct8::new();
    (window_2d(), dct.even, dct.odd, ROUND_TRIP_GAIN)
}

/// Whether method A filters any channel at `p` (a frame of at least 8 × 8 px assumed).
#[must_use]
pub fn filters_any(p: &BmDctParams) -> bool {
    Ctx::active(p).contains(&true)
}

/// Denoises `img` with method A.
///
/// `img` is the whole frame, or a window of it (`img.place`) with at least [`HALO_PX`] px of context
/// around the pixels the caller keeps (except where the window meets the frame's edge): those pixels
/// equal the whole frame's bit for bit. The output has the size and place of `img`. A frame
/// narrower or lower than 8 px, or strengths that filter no channel, return `img` unchanged.
pub fn denoise(img: &RgbImage, p: &BmDctParams) -> RgbImage {
    if !Ctx::runs(&img.place, p) {
        return img.clone();
    }
    let ctx = Ctx::new(ycbcr::to_ycbcr(img), img.width, img.height, img.place, p);
    let acc = ctx.accumulate();
    ycbcr::from_ycbcr(&ctx.finish(&acc, p.blend), img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{Rect, metrics, synth};

    fn params(luma: f64, chroma: f64, radius: usize) -> BmDctParams {
        BmDctParams {
            luma,
            chroma,
            radius,
            blend: 0.0,
            level: None,
        }
    }

    fn bits(img: &RgbImage) -> Vec<u32> {
        img.data.iter().map(|v| v.to_bits()).collect()
    }

    fn noisy(w: usize, h: usize, sigma: f32, seed: u64) -> RgbImage {
        let mut img = synth::natural(w, h, seed);
        synth::add_noise(&mut img.data, sigma, seed + 100);
        img
    }

    fn random_group(seed: u64) -> Group {
        let mut rng = synth::Rng::new(seed);
        from_fn(|_| from_fn(|_| from_fn(|_| f64::from(rng.range(-1.0, 1.0)))))
    }

    fn naive_dct(x: &[f64; 8]) -> [f64; 8] {
        from_fn(|k| {
            let c = if k == 0 { 1.0 } else { SQRT_2 };
            (0..8)
                .map(|n| c * x[n] * (PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos())
                .sum()
        })
    }

    fn naive_idct(c: &[f64; 8]) -> [f64; 8] {
        from_fn(|n| {
            (0..8)
                .map(|k| {
                    let s = if k == 0 { 1.0 } else { SQRT_2 };
                    s * c[k] * (PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos()
                })
                .sum()
        })
    }

    /// A context on a Y plane alone (Cb, Cr zero), for the grouping tests.
    fn ctx_on_luma(y: Vec<f64>, w: usize, h: usize, place: Placement, luma: f64) -> Ctx {
        let n = y.len();
        Ctx::new(
            [y, vec![0.0; n], vec![0.0; n]],
            w,
            h,
            place,
            &params(luma, 0.0, 1),
        )
    }

    /// `Y = scale · (x / 64 + y / 16)` in frame coordinates, with `scale` a power of two: exact in
    /// `f64`, so equal offsets give equal distances bit for bit, `D(dx, dy) = scale · |dx / 64 + dy / 16| · ΣK`.
    fn dyadic_luma(w: usize, h: usize, place: Placement, scale: f64) -> Vec<f64> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (place.x0 + i % w, place.y0 + i / w);
                scale * (x as f64 / 64.0 + y as f64 / 16.0)
            })
            .collect()
    }

    #[test]
    fn kaiser_window_matches_the_spec_and_sums_to_35_59() {
        let spec = [
            0.4387, 0.6813, 0.8768, 0.9858, 0.9858, 0.8768, 0.6813, 0.4387,
        ];
        let k = kaiser();
        for (n, (a, b)) in k.iter().zip(spec).enumerate() {
            assert!((a - b).abs() < 5e-5, "k({n}) = {a}, spec {b}");
            assert_eq!(k[n].to_bits(), k[7 - n].to_bits(), "symmetric at {n}");
        }
        let sum: f64 = window_2d().iter().flatten().sum();
        assert!((sum - 35.59).abs() <= 0.01, "{sum}");
    }

    #[test]
    fn dct_of_a_constant_block_is_its_sum_in_the_dc_term_alone() {
        let dct = Dct8::new();
        let c = dct.forward(&[1.0; 8]);
        assert_eq!(c[0], 8.0);
        assert!(c[1..].iter().all(|v| v.abs() < 1e-12), "{c:?}");

        let mut g: Group = [[[1.0; B]; B]; 8];
        dct.forward_3d(&mut g);
        assert_eq!(g[0][0][0], 512.0);
        let rest = g
            .iter()
            .flatten()
            .flatten()
            .skip(1)
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(rest < 1e-9, "{rest}");
    }

    #[test]
    fn dct_round_trip_gains_8_in_1d_and_512_in_3d() {
        let dct = Dct8::new();
        let mut rng = synth::Rng::new(5);
        let x: [f64; 8] = from_fn(|_| f64::from(rng.range(-1.0, 1.0)));
        let back = dct.inverse(&dct.forward(&x));
        for (a, b) in back.iter().zip(&x) {
            assert!((a - 8.0 * b).abs() < 1e-12, "{a} vs {}", 8.0 * b);
        }

        let g = random_group(7);
        let mut t = g;
        dct.forward_3d(&mut t);
        dct.inverse_3d(&mut t);
        let peak = g
            .iter()
            .flatten()
            .flatten()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = t
            .iter()
            .flatten()
            .flatten()
            .zip(g.iter().flatten().flatten())
            .fold(0.0f64, |m, (a, b)| m.max((a - 512.0 * b).abs()));
        let relative = worst / (512.0 * peak);
        assert!(relative < 1e-9, "relative error {relative}");
    }

    #[test]
    fn dct_matches_a_naive_cosine_sum() {
        let dct = Dct8::new();
        let mut rng = synth::Rng::new(9);
        for _ in 0..50 {
            let x: [f64; 8] = from_fn(|_| f64::from(rng.range(-2.0, 2.0)));
            for (a, b) in dct.forward(&x).iter().zip(naive_dct(&x)) {
                assert!((a - b).abs() < 1e-12, "forward {a} vs {b}");
            }
            for (a, b) in dct.inverse(&x).iter().zip(naive_idct(&x)) {
                assert!((a - b).abs() < 1e-12, "inverse {a} vs {b}");
            }
        }
        // The 3D transform is the naive 1D one along the rows, the columns and the blocks in turn.
        let g = random_group(11);
        let mut t = g;
        dct.forward_3d(&mut t);
        let rows: Group = g.map(|b| b.map(|r| naive_dct(&r)));
        let cols: Group = rows.map(|b| {
            let mut out = b;
            for j in 0..B {
                let c = naive_dct(&from_fn(|i| b[i][j]));
                for (row, v) in out.iter_mut().zip(c) {
                    row[j] = v;
                }
            }
            out
        });
        for i in 0..B {
            for j in 0..B {
                let line = naive_dct(&from_fn(|k| cols[k][i][j]));
                for (k, v) in line.iter().enumerate() {
                    assert!((t[k][i][j] - v).abs() < 1e-11, "({k}, {i}, {j})");
                }
            }
        }
    }

    /// The candidate of each slot of weights that are all 0 or 1 (a test fails on a blend).
    fn hard(w: &[[f64; CANDIDATES]; 8]) -> [usize; 8] {
        from_fn(|k| {
            let at: Vec<usize> = (0..CANDIDATES).filter(|&i| w[k][i] != 0.0).collect();
            assert_eq!(at.len(), 1, "slot {k}: {:?}", w[k]);
            assert_eq!(w[k][at[0]], 1.0, "slot {k}");
            at[0]
        })
    }

    #[test]
    fn grouping_takes_the_nearest_candidates_by_distance() {
        let (w, h) = (48, 48);
        let place = Placement::whole(w, h);
        let ctx = ctx_on_luma(dyadic_luma(w, h, place, 1.0), w, h, place, 50.0);
        let (d, pos) = ctx.distances(20, 20);
        // |dx/64 + dy/16|: 11 and 13 at 1/16; 9, 10, 14, 15 at 1/8; 8 and 16 at 3/16 (θ = 10 is
        // above all of them: ΣK · 3/16 ≈ 6.7).
        let sum_k: f64 = window_2d().iter().flatten().sum();
        assert_eq!(d[9], d[15]);
        assert_eq!(d[9], sum_k / 8.0);
        assert_eq!(d[11], sum_k / 16.0);
        assert_eq!(pos[11], (16, 20));
        // Distances apart by more than the band: the slots are the 7 nearest in order of distance.
        let mut apart = [f64::INFINITY; CANDIDATES];
        for (n, idx) in [12, 11, 13, 9, 10, 14, 15, 8, 16].into_iter().enumerate() {
            apart[idx] = n as f64 * 0.01;
        }
        assert_eq!(hard(&select(&apart, 10.0)), [12, 11, 13, 9, 10, 14, 15, 8]);
        // Equal distances share their slots: 9 and 15 at the same D take half of each of two slots,
        // in whatever way the two distances round.
        let mut tied = apart;
        tied[15] = tied[9];
        let w = select(&tied, 10.0);
        assert_eq!((w[3][9], w[3][15], w[4][9], w[4][15]), (0.5, 0.5, 0.5, 0.5));
        let mut off = tied;
        off[15] += 1e-9;
        let v = select(&off, 10.0);
        for k in 0..8 {
            for i in 0..CANDIDATES {
                assert!((w[k][i] - v[k][i]).abs() < 1e-5, "{k} {i}");
            }
        }
    }

    #[test]
    fn grouping_replaces_slots_over_theta_with_the_reference() {
        // A slot without a candidate takes the reference.
        let mut d = [f64::INFINITY; CANDIDATES];
        d[REF] = 0.0;
        (d[3], d[20], d[7]) = (2.0, 1.0, 1.5);
        assert_eq!(hard(&select(&d, 10.0)), [12, 20, 7, 3, 12, 12, 12, 12]);
        // Over theta the reference takes the slot.
        assert_eq!(hard(&select(&d, 1.7)), [12, 20, 7, 12, 12, 12, 12, 12]);
        assert_eq!(hard(&select(&d, 1.2)), [12, 20, 12, 12, 12, 12, 12, 12]);
        // Within the band of theta a slot is shared: the half at the middle.
        let w = select(&d, 1.5);
        assert!((w[2][7] - 0.5).abs() < 1e-12 && (w[2][REF] - 0.5).abs() < 1e-12);
        // A distance that is not a number is no candidate.
        d[4] = f64::NAN;
        assert_eq!(hard(&select(&d, 10.0)), [12, 20, 7, 3, 12, 12, 12, 12]);
    }

    #[test]
    fn grouping_never_takes_a_candidate_outside_the_frame() {
        // Top-left corner of the frame: only dx, dy >= 0 are candidates. 11 (dx = -4) would be the
        // nearest if it were one. θ = 20 is above every distance (at most ΣK · 5/16 ≈ 11.1).
        let (w, h) = (40, 40);
        let place = Placement::whole(w, h);
        let ctx = ctx_on_luma(dyadic_luma(w, h, place, 0.5), w, h, place, 100.0);
        let (d, _) = ctx.distances(0, 0);
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 20, 21] {
            assert_eq!(d[idx], f64::INFINITY, "{idx}");
        }
        let weights = select(&d, ctx.theta);
        for row in &weights {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 20, 21] {
                assert_eq!(row[idx], 0.0, "{idx}");
            }
        }

        // The bottom-right corner of the frame, seen through a window of it: frame coordinates.
        let (fw, fh) = (35, 37);
        let (bw, bh) = (30, 30);
        let place = Placement {
            x0: fw - bw,
            y0: fh - bh,
            frame_width: fw,
            frame_height: fh,
        };
        let ctx = ctx_on_luma(dyadic_luma(bw, bh, place, 0.5), bw, bh, place, 100.0);
        let (d, pos) = ctx.distances(fw - 8, fh - 8);
        assert_eq!(pos[REF], (bw - 8, bh - 8));
        for idx in [13, 14, 17, 18, 19, 22, 23, 24] {
            assert_eq!(d[idx], f64::INFINITY, "{idx}");
        }
        for row in &select(&d, ctx.theta) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn the_weights_are_continuous_in_the_distances() {
        // Every weight moves by at most about (change / band) times a constant when a distance moves,
        // including through a tie and through theta.
        let mut base = [f64::INFINITY; CANDIDATES];
        for (n, idx) in [12, 3, 7, 8, 9, 13, 14, 18, 19, 20, 22]
            .into_iter()
            .enumerate()
        {
            base[idx] = 0.5 + 0.0005 * n as f64;
        }
        for step in 0..400 {
            let mut a = base;
            a[8] = 0.5 + 0.0005 * 3.0 + (step as f64 - 200.0) * 1e-5;
            let mut b = a;
            b[8] += 1e-10;
            let (wa, wb) = (select(&a, 0.503), select(&b, 0.503));
            for k in 0..8 {
                for i in 0..CANDIDATES {
                    assert!((wa[k][i] - wb[k][i]).abs() < 2e-5, "{step} {k} {i}");
                }
            }
        }
    }

    #[test]
    fn zero_strengths_return_the_input_bit_for_bit() {
        let img = noisy(40, 30, 0.03, 1);
        let mut p = params(0.0, 0.0, 1);
        assert_eq!(denoise(&img, &p), img);
        p.blend = 40.0;
        assert_eq!(denoise(&img, &p), img);
        // Thresholds at or below 1e-6 are off (A.1).
        assert_eq!(denoise(&img, &params(5e-6, 5e-6, 0)), img);
    }

    fn max_plane_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn luma_only_leaves_cb_and_cr_as_they_were() {
        let img = noisy(48, 40, 0.03, 2);
        let out = denoise(&img, &params(30.0, 0.0, 1));
        let [y0, cb0, cr0] = ycbcr::to_ycbcr(&img);
        let [y1, cb1, cr1] = ycbcr::to_ycbcr(&out);
        assert!(max_plane_diff(&cb0, &cb1) < 1e-6);
        assert!(max_plane_diff(&cr0, &cr1) < 1e-6);
        assert!(max_plane_diff(&y0, &y1) > 1e-2, "luma was filtered");
    }

    #[test]
    fn chroma_only_leaves_y_as_it_was() {
        let img = noisy(48, 40, 0.03, 3);
        let out = denoise(&img, &params(0.0, 30.0, 1));
        let [y0, cb0, cr0] = ycbcr::to_ycbcr(&img);
        let [y1, cb1, cr1] = ycbcr::to_ycbcr(&out);
        assert!(max_plane_diff(&y0, &y1) < 1e-6);
        assert!(max_plane_diff(&cb0, &cb1) > 1e-2, "Cb was filtered");
        assert!(max_plane_diff(&cr0, &cr1) > 1e-2, "Cr was filtered");
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let mut img = RgbImage::new(37, 29);
        for (c, v) in [0.5f32, 0.4, 0.3].into_iter().enumerate() {
            img.plane_mut(c).fill(v);
        }
        let out = denoise(&img, &params(30.0, 30.0, 1));
        assert!(out.max_abs_diff(&img) < 1e-6, "{}", out.max_abs_diff(&img));
    }

    /// Gradients, a rectangle, a disc and a diagonal edge, values in 0.1 to 0.9.
    fn smooth_scene(w: usize, h: usize) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                let mut rgb = [
                    0.15 + 0.5 * fx + 0.1 * fy,
                    0.2 + 0.45 * fy + 0.15 * (1.0 - fx),
                    0.3 + 0.35 * fx * fy,
                ];
                if (0.3..0.6).contains(&fx) && (0.2..0.7).contains(&fy) {
                    rgb = [rgb[0] + 0.2, rgb[1] - 0.1, rgb[2] + 0.25];
                }
                let (dx, dy) = (fx - 0.72, fy - 0.3);
                if dx * dx + dy * dy < 0.15 * 0.15 {
                    rgb = [0.85, 0.3, 0.15];
                }
                if fx + fy > 1.5 {
                    rgb = [0.2, 0.25, 0.7];
                }
                img.set_pixel(x, y, rgb.map(|v| v.clamp(0.1, 0.9)));
            }
        }
        img
    }

    #[test]
    fn denoising_a_noisy_smooth_scene_gains_at_least_5_db() {
        let clean = smooth_scene(128, 128);
        let sigma = 10.0 / 255.0;
        let mut noisy = clean.clone();
        synth::add_noise(&mut noisy.data, sigma as f32, 42);
        // τ / 8^1.5 = 2.7 σ: L = 2.7 σ 8^1.5 / 0.1, C = 2.7 σ 8^1.5 / 0.06.
        let scaled = 2.7 * sigma * 8f64.powf(1.5);
        let (luma, chroma) = (scaled / 0.1, scaled / 0.06);
        let before = metrics::psnr(&noisy, &clean, 0, 1.0);
        for radius in 0..=2 {
            let out = denoise(&noisy, &params(luma, chroma, radius));
            let after = metrics::psnr(&out, &clean, 0, 1.0);
            println!(
                "R = {radius} (L = {luma:.2}, C = {chroma:.2}): PSNR {before:.2} dB -> {after:.2} dB \
                 (+{:.2})",
                after - before
            );
            assert!(after - before >= 5.0, "R = {radius}: {before} -> {after}");
        }
    }

    #[test]
    fn output_is_the_same_bits_on_1_and_4_threads() {
        let img = noisy(101, 77, 0.03, 4);
        let p = BmDctParams {
            blend: 20.0,
            ..params(30.0, 30.0, 1)
        };
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| denoise(&img, &p))
        };
        let one = run(1);
        assert_ne!(bits(&one), bits(&img));
        assert_eq!(bits(&one), bits(&run(4)));
    }

    #[test]
    fn tiles_with_the_halo_equal_the_whole_frame_bit_for_bit() {
        let (fw, fh) = (131, 113);
        let img = noisy(fw, fh, 0.03, 5);
        let tiles = [
            Rect::new(0, 0, 55, 49),
            Rect::new(17, 9, 61, 57),
            Rect::new(33, 29, 67, 70),
            Rect::new(71, 53, fw - 71, fh - 53),
            Rect::new(3, 61, 49, fh - 61),
            Rect::new(0, 0, fw, fh),
        ];
        for radius in 0..=3 {
            let p = BmDctParams {
                blend: 25.0,
                ..params(30.0, 40.0, radius)
            };
            let whole = denoise(&img, &p);
            for t in tiles {
                let out = denoise(&img.extract(t), &p);
                // Keep HALO_PX px inside every tile edge that is not a frame edge.
                let lo = |a0: usize| if a0 == 0 { 0 } else { HALO_PX };
                let hi = |a1: usize, len: usize, f: usize| {
                    if a1 == f { len } else { len - HALO_PX }
                };
                let (cx0, cx1) = (lo(t.x), hi(t.x1(), t.w, fw));
                let (cy0, cy1) = (lo(t.y), hi(t.y1(), t.h, fh));
                assert!(cx0 < cx1 && cy0 < cy1, "{t:?} has no core");
                for c in 0..3 {
                    for y in cy0..cy1 {
                        for x in cx0..cx1 {
                            let a = out.plane(c)[y * t.w + x];
                            let b = whole.plane(c)[(t.y + y) * fw + t.x + x];
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "R = {radius}, {t:?}, channel {c} at ({x}, {y}): {a} vs {b}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_frames_run_and_frames_under_8_px_are_returned_unchanged() {
        let p = params(30.0, 30.0, 1);
        for (w, h) in [(8, 8), (9, 13), (31, 17)] {
            let img = noisy(w, h, 0.05, 6);
            let out = denoise(&img, &p);
            assert_eq!((out.width, out.height), (w, h));
            assert!(out.data.iter().all(|v| v.is_finite()), "{w} x {h}");
            assert_ne!(bits(&out), bits(&img), "{w} x {h} was filtered");
        }
        for (w, h) in [(7, 20), (20, 7), (3, 3)] {
            let img = noisy(w, h, 0.05, 7);
            assert_eq!(bits(&denoise(&img, &p)), bits(&img), "{w} x {h}");
        }
    }

    #[test]
    fn every_pixel_receives_a_contribution_at_odd_sizes() {
        for (w, h) in [(8, 8), (9, 13), (31, 17), (37, 29), (53, 61)] {
            let img = noisy(w, h, 0.05, 8);
            for radius in 0..=3 {
                let ctx = Ctx::new(
                    ycbcr::to_ycbcr(&img),
                    w,
                    h,
                    img.place,
                    &params(30.0, 30.0, radius),
                );
                let acc = ctx.accumulate();
                let empty = acc.iter().position(|a| a[3].is_nan() || a[3] <= 0.0);
                assert_eq!(
                    empty, None,
                    "{w} x {h}, R = {radius}: a pixel without weight"
                );
            }
        }
    }

    fn level(chroma_scale: f64) -> LevelNoise {
        LevelNoise {
            a: 1e-4,
            b: 1e-6,
            gains: [1.0; 3],
            encoding_gain: crate::cpu::denoise_space::WORKING_GAIN_EV.exp2(),
            chroma_scale,
        }
    }

    #[test]
    fn the_level_noise_at_grey_is_the_working_noise_and_grows_into_the_shadows() {
        use crate::cpu::denoise_space::{WORKING_GAIN_EV, encode};
        use crate::cpu::frame_denoise::working_noise;
        use ym_noise::model::NoiseModel;
        let (a, b, gains) = (4e-4, 6e-7, [2.0_f32, 1.0, 1.5]);
        let l = LevelNoise {
            a,
            b,
            gains: gains.map(f64::from),
            encoding_gain: WORKING_GAIN_EV.exp2(),
            chroma_scale: 1.0,
        };
        let grey = 0.18 / WORKING_GAIN_EV.exp2();
        let at = |v: f64| l.chroma_sigma([encode(v, WORKING_GAIN_EV), 0.0, 0.0]);
        let (_, sc) = working_noise(&NoiseModel::new(a, b), gains);
        assert!((at(grey) / sc - 1.0).abs() < 1e-6, "{} {sc}", at(grey));
        assert!(
            at(grey / 16.0) > 1.3 * at(grey),
            "{} {}",
            at(grey / 16.0),
            at(grey)
        );
        assert!(at(grey * 4.0) < at(grey));
        // A colour below black in one channel reads that channel at 0, not as a negative level.
        assert!(at(-0.01).is_finite() && l.chroma_sigma([0.05, 0.2, -0.3]).is_finite());
    }

    #[test]
    fn the_blend_brings_back_the_luma_and_leaves_the_colour_denoised() {
        let img = noisy(64, 48, 0.05, 7);
        let base = params(30.0, 30.0, 1);
        let plain = denoise(&img, &base);
        let blended = denoise(
            &img,
            &BmDctParams {
                blend: 40.0,
                ..base
            },
        );
        let [yp, cbp, crp] = ycbcr::to_ycbcr(&plain);
        let [yb, cbb, crb] = ycbcr::to_ycbcr(&blended);
        let [yi, _, _] = ycbcr::to_ycbcr(&img);
        for i in 0..yp.len() {
            assert!(
                (cbp[i] - cbb[i]).abs() < 1e-6 && (crp[i] - crb[i]).abs() < 1e-6,
                "{i}"
            );
            let want = yp[i] + 0.4 * (yi[i] - yp[i]);
            assert!((yb[i] - want).abs() < 1e-6, "{i}: {} {want}", yb[i]);
        }
    }

    #[test]
    fn a_threshold_that_follows_the_level_filters_dark_colour_noise_harder() {
        // A dark and a light neutral half in the working encoding with the same noise in every channel.
        let (w, h) = (96, 64);
        let mut img = RgbImage::from_planar(
            w,
            h,
            (0..3 * w * h)
                .map(|i| if i % w < w / 2 { 0.08 } else { 0.75 })
                .collect(),
        )
        .unwrap();
        synth::add_noise(&mut img.data, 0.02, 11);
        // The chroma noise std over the inner part of each half.
        let residual = |out: &RgbImage| {
            let [_, cb, cr] = ycbcr::to_ycbcr(out);
            let side = |x0: usize| {
                let v: Vec<f64> = (12..h - 12)
                    .flat_map(|y| (x0 + 12..x0 + w / 2 - 12).map(move |x| y * w + x))
                    .flat_map(|i| [cb[i], cr[i]])
                    .collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                (v.iter().map(|d| (d - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
            };
            (side(0), side(w / 2))
        };
        let k = 3.0 * 22.627_416_997_969_52;
        let l = level(k);
        // One threshold, the light half's: both halves keep about as much colour noise.
        let bright_tau = k * l.chroma_sigma([0.75, 0.0, 0.0]);
        let one = BmDctParams {
            luma: 0.0,
            chroma: bright_tau / 0.06,
            radius: 1,
            blend: 0.0,
            level: None,
        };
        let (dark, light) = residual(&denoise(&img, &one));
        assert!(dark > 0.8 * light, "{dark} {light}");
        // The level's threshold: the dark half, three times noisier in the model, keeps far less.
        let (dark, light) = residual(&denoise(
            &img,
            &BmDctParams {
                level: Some(l),
                ..one
            },
        ));
        assert!(dark < 0.5 * light, "{dark} {light}");
    }

    #[test]
    fn radius_above_3_is_read_as_3() {
        assert_eq!(params(1.0, 1.0, 0).step_px(), 8);
        assert_eq!(params(1.0, 1.0, 3).step_px(), 2);
        assert_eq!(params(1.0, 1.0, 9).step_px(), 2);
    }

    /// Timing on a 2048 × 1365 frame: `cargo test --release -p ym-gpu --lib bm_dct -- --ignored
    /// --nocapture`. Prints CPU seconds per megapixel (one thread) and the wall time on all threads.
    #[test]
    #[ignore = "timing; run in release with --ignored --nocapture"]
    fn timing_on_a_2048_by_1365_frame() {
        let (w, h) = (2048, 1365);
        let img = noisy(w, h, 0.03, 9);
        let mp = (w * h) as f64 / 1e6;
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for radius in 0..=3 {
            let p = params(30.0, 30.0, radius);
            let t = std::time::Instant::now();
            let a = one.install(|| denoise(&img, &p));
            let single = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let b = denoise(&img, &p);
            let all = t.elapsed().as_secs_f64();
            assert_eq!(bits(&a), bits(&b));
            println!(
                "R = {radius}, L = C = 30, {w} x {h}: 1 thread {single:.2} s ({:.2} CPU-s/MP); \
                 {} threads {all:.2} s wall",
                single / mp,
                rayon::current_num_threads()
            );
        }
    }
}
