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
//! let p = BmDctParams { luma: 30.0, chroma: 30.0, radius: 1, blend: 0.0 };
//! let out = bm_dct::denoise(&img, &p);
//! assert_eq!((out.width, out.height), (32, 24));
//! ```
//!
//! # Maths
//!
//! The image goes to BT.709 Y/Cb/Cr ([`ycbcr::to_ycbcr`]); everything after is `f64`. From the
//! strengths, `τY = 0.1 L`, `τC = 0.06 C`, `θ = 0.2 max(L, C)` and the reference step
//! `s = 8 − 2R` ([`BmDctParams`]). Y is filtered when `τY > 1e-6`, Cb and Cr when `τC > 1e-6`; a
//! plane that is not filtered is the input plane. For every reference block (8 × 8):
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
//!    optional highlight tweak of the text is omitted (reading A-b).
//! 6. **Inverse** 3D transform (across the blocks, then columns, then rows).
//! 7. **Aggregation**: for every pixel of every slot, `acc_ch += K(i, j) · v / 512` and
//!    `w += K(i, j)` (once per pixel, not per channel). The output is `acc_ch / w`.
//!
//! The result goes back to RGB ([`ycbcr::from_ycbcr`]) and is blended with the input,
//! `lerp(out, input, B / 100)` for `B > 0` ([`ycbcr::blend`]). When neither channel is filtered,
//! or the frame is narrower or lower than 8 px, the input comes back bit for bit.
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

use super::ycbcr::{self, Planes};
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

/// Corner offset `(dx, dy)` of candidate `idx` from its reference.
const fn offset(idx: usize) -> (isize, isize) {
    ((idx % 5) as isize * 4 - 8, (idx / 5) as isize * 4 - 8)
}

/// The slots of a group from the candidates' distances (`+∞` for a block that is not a
/// candidate): slot 0 the reference, then the 7 other candidates of smallest distance in order (a
/// tie goes to the smaller index), and the reference for any slot whose distance exceeds `theta`
/// or for which no candidate is left.
fn select(d: &[f64; CANDIDATES], theta: f64) -> [usize; 8] {
    let mut best = [(f64::INFINITY, REF); 7];
    let mut n = 0;
    for (idx, &di) in d.iter().enumerate() {
        if idx == REF || !di.is_finite() {
            continue;
        }
        // After every kept entry of distance <= di: a tie keeps the smaller index first.
        let at = best[..n].iter().position(|&(bd, _)| bd > di).unwrap_or(n);
        if at >= best.len() {
            continue;
        }
        let end = n.min(best.len() - 1);
        best.copy_within(at..end, at + 1);
        best[at] = (di, idx);
        n = (n + 1).min(best.len());
    }
    let mut slots = [REF; 8];
    for (slot, &(bd, idx)) in slots[1..].iter_mut().zip(&best[..n]) {
        if bd <= theta {
            *slot = idx;
        }
    }
    slots
}

/// Per-worker scratch of one reference.
struct Scratch {
    /// Buffer position (top-left) of each slot's block.
    pos: [(usize, usize); 8],
    /// The group of each channel (Y, Cb, Cr); after [`Ctx::filter`], the filtered blocks times 512.
    groups: [Group; 3],
}

impl Scratch {
    /// Zeroed scratch on the heap (12 KiB).
    fn boxed() -> Box<Scratch> {
        Box::new(Scratch {
            pos: [(0, 0); 8],
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
    /// Block-distance threshold.
    theta: f64,
    /// Reference step in pixels.
    step: usize,
}

impl Ctx {
    /// Whether the method changes anything: a frame of at least 8 × 8 px and a channel to filter.
    fn runs(place: &Placement, p: &BmDctParams) -> bool {
        place.frame_width >= B
            && place.frame_height >= B
            && (p.luma_threshold() > SKIP_TAU || p.chroma_threshold() > SKIP_TAU)
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
            active: tau.map(|t| t > SKIP_TAU),
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

    /// The slots (candidate indices) of the group of the reference at frame `(rx, ry)` and the
    /// buffer positions of their blocks.
    fn group(&self, rx: usize, ry: usize) -> ([usize; 8], [(usize, usize); 8]) {
        let (d, pos) = self.distances(rx, ry);
        let slots = select(&d, self.theta);
        (slots, slots.map(|idx| pos[idx]))
    }

    /// Groups, transforms, thresholds and inverts the blocks of the reference at frame `(rx, ry)`.
    fn filter(&self, rx: usize, ry: usize, s: &mut Scratch) {
        let (_, pos) = self.group(rx, ry);
        s.pos = pos;
        let w = self.width;
        for ch in 0..3 {
            if !self.active[ch] {
                continue;
            }
            let plane = &self.planes[ch];
            let g = &mut s.groups[ch];
            for (b, &(cx, cy)) in g.iter_mut().zip(&pos) {
                for (i, row) in b.iter_mut().enumerate() {
                    row.copy_from_slice(&plane[(cy + i) * w + cx..][..B]);
                }
            }
            self.dct.forward_3d(g);
            let tau = self.tau[ch];
            for v in g.iter_mut().flatten().flatten() {
                if v.abs() < tau {
                    *v = 0.0;
                }
            }
            self.dct.inverse_3d(g);
        }
    }

    /// Adds the filtered group in `s` to `band`, the accumulators of the buffer rows from `r0` on,
    /// in slot order.
    fn scatter(&self, s: &Scratch, band: &mut [[f64; 4]], r0: usize) {
        let w = self.width;
        for (k, &(cx, cy)) in s.pos.iter().enumerate() {
            for (i, k_row) in self.window.iter().enumerate() {
                let out = &mut band[(cy + i - r0) * w + cx..][..B];
                for (j, (a, &kk)) in out.iter_mut().zip(k_row).enumerate() {
                    for (ch, g) in s.groups.iter().enumerate() {
                        if self.active[ch] {
                            a[ch] += kk * g[k][i][j] / ROUND_TRIP_GAIN;
                        }
                    }
                    a[3] += kk;
                }
            }
        }
    }

    /// The output planes: `acc / w` for the filtered channels where a reference reached the pixel,
    /// the input elsewhere.
    fn finish(self, acc: &[[f64; 4]]) -> Planes {
        let mut out = self.planes;
        for (i, a) in acc.iter().enumerate() {
            if a[3] > 0.0 {
                for (ch, plane) in out.iter_mut().enumerate() {
                    if self.active[ch] {
                        plane[i] = a[ch] / a[3];
                    }
                }
            }
        }
        out
    }
}

/// What the device kernel (`kernels::bm_dct`) takes from this reference for one buffer, so that its
/// schedule is this file's: the passes in order (each the frame rows and columns of its references, as
/// [`Ctx::passes`] makes them for the buffer), the filtered channels and the thresholds.
#[derive(Clone, Debug)]
pub(crate) struct DevicePlan {
    /// The passes in order, each `(rows, columns)` in frame coordinates.
    pub(crate) passes: Vec<(Vec<usize>, Vec<usize>)>,
    /// Whether Y, Cb and Cr are filtered.
    pub(crate) active: [bool; 3],
    /// The hard threshold of each channel.
    pub(crate) tau: [f64; 3],
    /// The block-distance threshold.
    pub(crate) theta: f64,
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
    })
}

/// The constants of this reference the device kernel writes into its shader and uniform block, so
/// that they are never typed twice: `K(i, j)`, the rows `2m` and `2m + 1` of the DCT matrix (columns 0
/// to 3, [`Dct8`]) and the gain of a forward and inverse 3D transform (512).
pub(crate) fn device_constants() -> (Block, [[f64; 4]; 4], [[f64; 4]; 4], f64) {
    let dct = Dct8::new();
    (window_2d(), dct.even, dct.odd, ROUND_TRIP_GAIN)
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
    let planes = ctx.finish(&acc);
    ycbcr::blend(ycbcr::from_ycbcr(&planes, img), img, p.blend)
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

    #[test]
    fn grouping_takes_the_nearest_candidates_by_distance_then_index() {
        let (w, h) = (48, 48);
        let place = Placement::whole(w, h);
        let ctx = ctx_on_luma(dyadic_luma(w, h, place, 1.0), w, h, place, 50.0);
        // |dx/64 + dy/16|: 11 and 13 at 1/16; 9, 10, 14, 15 at 1/8; 8 and 16 at 3/16 (θ = 10 is
        // above all of them: ΣK · 3/16 ≈ 6.7).
        let (slots, pos) = ctx.group(20, 20);
        assert_eq!(slots, [12, 11, 13, 9, 10, 14, 15, 8]);
        assert_eq!(pos[0], (20, 20));
        assert_eq!(pos[1], (16, 20));
        assert_eq!(pos[7], (24, 16));
        let (d, _) = ctx.distances(20, 20);
        let sum_k: f64 = window_2d().iter().flatten().sum();
        assert_eq!(d[9], d[15]);
        assert_eq!(d[9], sum_k / 8.0);
        assert_eq!(d[11], sum_k / 16.0);
    }

    #[test]
    fn grouping_replaces_slots_over_theta_with_the_reference() {
        let (w, h) = (48, 48);
        let place = Placement::whole(w, h);
        // θ = 0.2 · 27.5 = 5.5 lies between ΣK/8 ≈ 4.45 and ΣK · 3/16 ≈ 6.67.
        let ctx = ctx_on_luma(dyadic_luma(w, h, place, 1.0), w, h, place, 27.5);
        assert_eq!(ctx.group(20, 20).0, [12, 11, 13, 9, 10, 14, 15, 12]);
        // A slot without a candidate takes the reference too.
        let mut d = [f64::INFINITY; CANDIDATES];
        d[REF] = 0.0;
        (d[3], d[20], d[7]) = (2.0, 1.0, 1.0);
        assert_eq!(select(&d, 10.0), [12, 7, 20, 3, 12, 12, 12, 12]);
        assert_eq!(select(&d, 1.5), [12, 7, 20, 12, 12, 12, 12, 12]);
        d[4] = f64::NAN;
        assert_eq!(select(&d, 10.0), [12, 7, 20, 3, 12, 12, 12, 12]);
    }

    #[test]
    fn grouping_never_takes_a_candidate_outside_the_frame() {
        // Top-left corner of the frame: only dx, dy >= 0 are candidates. 11 (dx = -4) would be the
        // nearest if it were one. θ = 20 is above every distance (at most ΣK · 5/16 ≈ 11.1).
        let (w, h) = (40, 40);
        let place = Placement::whole(w, h);
        let ctx = ctx_on_luma(dyadic_luma(w, h, place, 0.5), w, h, place, 100.0);
        assert_eq!(ctx.group(0, 0).0, [12, 13, 14, 17, 18, 19, 22, 23]);
        let (d, _) = ctx.distances(0, 0);
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 20, 21] {
            assert_eq!(d[idx], f64::INFINITY, "{idx}");
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
        let (slots, pos) = ctx.group(fw - 8, fh - 8);
        assert_eq!(slots, [12, 11, 10, 7, 6, 5, 2, 1]);
        assert_eq!(pos[0], (bw - 8, bh - 8));
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
