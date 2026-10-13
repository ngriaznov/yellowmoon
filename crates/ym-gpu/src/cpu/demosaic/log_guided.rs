//! Log-guided fast demosaic (`Method::LogGuided`), specified in
//! `docs/algorithms/02-log-guided-demosaic.md` (RA-06).
//!
//! A fast Bayer demosaic for previews and thumbnails. It builds a luminance guide in the log
//! domain (the geometric mean of the three channels, with a read-noise offset `p`), then fills each
//! missing channel from the mean of its neighbours scaled by the local guide ratio. Green at red
//! and blue sites blends a horizontal and a vertical estimate with inverse-cube structure weights.
//! The receptive field is exactly [`HALO_PX`] = 5 pixels.
//!
//! Integration: `docs/algorithms/INTEGRATION.md` sections 3 to 5. The input is the **balanced**
//! mosaic (after R4 and R5): [`demosaic`] is called on it directly, nothing is divided out. The
//! output is balanced camera RGB with the input's size and [`Placement`](super::super::Placement).
//! The noise model arrives as [`NoiseDn`] (through [`DemosaicInputs`](super::DemosaicInputs) when
//! the method is selected with [`Method::LogGuided`](super::Method::LogGuided)); only its
//! signal-independent part `r` is used, to set the offset `p` (spec section 2.2).
//!
//! # Two forms
//!
//! * [`demosaic_float`] (and [`demosaic`], its default-detail shorthand): the **all-float form**
//!   of spec section 3.9, the production path. It needs no headroom: values above 1.0 keep their
//!   value. This is what `Method::LogGuided` runs.
//! * [`demosaic_fixed`]: the **fixed-point form** of sections 3.1 to 3.8, 16-bit integers with two
//!   cubic polynomials ([`log2_fixed`], [`exp2_fixed`]). It is normative for the `expected` field of
//!   the vectors and is kept for them and as an integer-only fallback. It needs the per-frame
//!   headroom `H` (spec section 2.4, [`LogGuidedParams::for_mosaic`]); tiles must all use the
//!   frame's `H`.
//!
//! # Where this port follows the reference to the letter, and where it deviates
//!
//! * Order of operations is the oracle's: the guide filters accumulate left to right in `f64`
//!   (no fused multiply-add anywhere), `log2` and `exp2` are `f64` library results rounded to
//!   `f32`, every plane is stored as `f32` (the fixed form as `u16`), the box means are `f64` sums
//!   divided by 25 and rounded to `f32`, and the reconstruction is `f64` in the order of section 3.7.
//! * The 5x5 box sum is computed separably (a vertical 5-sum of the `f32` structure planes in `f64`,
//!   then a horizontal 5-sum) instead of 25 sequential additions. The summands are `f32` values of
//!   at least 1 (the offset `p`), so as long as they stay below about 1.6e7 (a mosaic value of
//!   several hundred times white) every partial sum is exactly representable in `f64` and the
//!   order cannot change the result; beyond that it could change the last bit of an `f64`, which
//!   the rounding to `f32` hides.
//! * The structure weights are only evaluated at red and blue sites (the only place section 3.7
//!   reads them), which does not change any output.
//! * `log2` and `exp2` come from the platform's libm. Two platforms can differ by an ulp of `f64`
//!   there; after the rounding to `f32` that is invisible in practice, and the tests state the
//!   tolerance they hold (the vectors' own `tolerance_float` of 1e-4).
//!
//! There are no reductions that depend on thread order: every output sample is a fixed function of
//! the mosaic within [`HALO_PX`] pixels, so the banded runner (`filters::run_mosaic_bands`) and a
//! whole-buffer call give identical bytes.

use super::super::noise::{NoiseDn, WORKING_SCALE};
use super::super::{CfaColor, Mosaic, RgbImage, mirror};

/// Receptive-field radius in pixels (Chebyshev): output pixel `p` depends on mosaic photosites
/// within this distance of `p` (spec section 1.4; verified exactly by the perturbation tests).
pub const HALO_PX: usize = 5;

/// Smallest `guide_detail`; smaller values are clamped (spec section 2.1).
pub const GUIDE_DETAIL_MIN: i32 = -50;
/// Largest `guide_detail`; larger values are clamped (spec section 2.1).
pub const GUIDE_DETAIL_MAX: i32 = 100;
/// Largest `headroom_log2` (headroom 16); larger values are clamped (spec section 2.1).
pub const HEADROOM_LOG2_MAX: u8 = 4;
/// Upper bound of the offset `p` (spec section 2.2).
const OFFSET_MAX: f64 = 32767.0;
/// The largest code of the 16-bit working scale.
const CODE_MAX: f64 = 65535.0;
/// Sum of the kernel taps: the kernel is `K / 1024`.
const KERNEL_DIVISOR: f64 = 1024.0;

/// Parameters of the log-guided demosaic (spec section 2.1) other than the noise model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogGuidedParams {
    /// Sharpness of the luminance guide, clamped to `-50..=100`. 0 is the plain geometric-mean
    /// guide and the default; the parameter is internal and not exposed in the UI (spec 2.3).
    pub guide_detail: i32,
    /// Fixed-point form only: the headroom is `H = 2^headroom_log2`, clamped to `0..=4`.
    pub headroom_log2: u8,
}

impl LogGuidedParams {
    /// The headroom exponent used when the frame maximum is not available: `H = 4` (spec 2.4).
    pub const DEFAULT_HEADROOM_LOG2: u8 = 2;

    /// The default rule of spec section 2.4: `H` is the smallest power of two that is at least
    /// `max_sample`, capped at 16. A non-finite `max_sample` gives the fallback
    /// [`DEFAULT_HEADROOM_LOG2`](Self::DEFAULT_HEADROOM_LOG2).
    #[must_use]
    pub fn headroom_log2_for(max_sample: f32) -> u8 {
        if !max_sample.is_finite() {
            return Self::DEFAULT_HEADROOM_LOG2;
        }
        let max = f64::from(max_sample);
        (0..HEADROOM_LOG2_MAX)
            .find(|&n| max <= f64::from(1u32 << n))
            .unwrap_or(HEADROOM_LOG2_MAX)
    }

    /// Detail 0 and the headroom for the largest finite sample of `m`, which is the frame's
    /// balanced maximum after R5 when `m` is the whole frame. Compute it once per frame, before
    /// tiling, and pass the same value to every tile (spec 2.4). A mosaic without a finite sample
    /// gets the fallback headroom.
    #[must_use]
    pub fn for_mosaic(m: &Mosaic) -> LogGuidedParams {
        let max = m
            .data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(None, |acc: Option<f32>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            });
        LogGuidedParams {
            guide_detail: 0,
            headroom_log2: max.map_or(Self::DEFAULT_HEADROOM_LOG2, Self::headroom_log2_for),
        }
    }
}

impl Default for LogGuidedParams {
    /// Detail 0 and the fallback headroom `H = 4`.
    fn default() -> LogGuidedParams {
        LogGuidedParams {
            guide_detail: 0,
            headroom_log2: LogGuidedParams::DEFAULT_HEADROOM_LOG2,
        }
    }
}

/// The values derived from the parameters and the noise model (spec section 2.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Derived {
    /// Log offset and structure regulariser, an integer in `1..=32767` in the units of the form
    /// that was asked for (`H` applied for the fixed form, `H = 1` for the float form).
    pub p: u32,
    /// Outer tap of the guide kernel, `-trunc(2.048 * guide_detail)`.
    pub k: i32,
    /// The guide kernel `(k, 256, 512 - 2k, 256, k)`; the taps sum to 1024.
    pub kernel: [i32; 5],
    /// The headroom `H` (1, 2, 4, 8 or 16).
    pub headroom: f64,
    /// Read-noise sigma in DN at the working scale `S = 65535`, `r_dn`. An invalid noise value
    /// (negative, `NaN` or infinite) counts as 0 (spec 2.2).
    pub read_noise_dn: f64,
}

/// Derives `p`, the guide kernel and the headroom from the noise model and the parameters
/// (spec section 2.2). `guide_detail` and `headroom_log2` are clamped to their ranges. The float
/// form uses `headroom_log2 = 0`.
#[must_use]
pub fn derive(noise: NoiseDn, guide_detail: i32, headroom_log2: u8) -> Derived {
    let detail = guide_detail.clamp(GUIDE_DETAIL_MIN, GUIDE_DETAIL_MAX);
    // Integer division truncates toward zero: k = -trunc(2.048 * P) exactly.
    let k = -((2048 * detail) / 1000);
    let headroom = f64::from(1u32 << headroom_log2.min(HEADROOM_LOG2_MAX));
    // `NoiseDn::r` is S * sqrt(b + sigma_row^2 + q^2 / 12): a negative variance gives NaN and an
    // infinite one infinity, both read as 0 here (the oracle zeroes the variance).
    let read_noise_dn = if noise.r.is_finite() && noise.r > 0.0 {
        noise.r
    } else {
        0.0
    };
    let p = (read_noise_dn / (2.0 * headroom))
        .floor()
        .clamp(1.0, OFFSET_MAX) as u32;
    Derived {
        p,
        k,
        kernel: [k, 256, 512 - 2 * k, 256, k],
        headroom,
        read_noise_dn,
    }
}

// ---------------------------------------------------------------------------------------------
// Fixed-point log2 / exp2 (spec sections 3.2 and 3.4)
// ---------------------------------------------------------------------------------------------

/// `L(y)`, about `4096 * log2(y)`, for `1 <= y <= 65535` (spec section 3.2; `log2_fixed(0)` is
/// defined as 0 and never reached). A cubic on the mantissa: the error is between -4.98 and +2.98
/// codes (1/4096 stop) for every `y`, and `L(1) = 0`, `L(2) = 4096`, `L(65535) = 65535`. Doubling
/// `y` adds exactly 4096. It is not strictly monotone: in the top octave it steps back by one code
/// at 23 values of `y`.
#[must_use]
pub fn log2_fixed(y: u16) -> u16 {
    if y == 0 {
        return 0;
    }
    let y = u32::from(y);
    let e = 31 - y.leading_zeros(); // index of the highest set bit, 0..15
    let n = y << (15 - e); // 32768 <= n <= 65535
    let m = 2 * n - 65536; // mantissa fraction in Q16, 0..65534
    let m2 = (m * m) >> 16;
    let m3 = (m2 * m) >> 16;
    let poly = ((m * 46597) >> 16) - ((m2 * 19037) >> 16) + ((m3 * 5208) >> 16); // 0..32767
    // 4096 * 15 + 4095 = 65535: fits.
    (4096 * e + (poly >> 3)) as u16
}

/// `C(L)`, about `2^(L / 4096)`, for `0 <= L <= 65535` (spec section 3.4). A cubic on the
/// fraction: `-(1 + 1.18e-4 t) <= C - t <= 1.24e-4 t` with `t = 2^(L / 4096)`,
/// `exp2_fixed(4096 e) = 2^e` exactly, monotone non-decreasing, maximum 65522.
#[must_use]
pub fn exp2_fixed(l: u16) -> u16 {
    let l = u32::from(l);
    let e = l >> 12; // integer part, 0..15
    let f = (l & 4095) << 4; // fraction in Q16, 0..65520
    let f2 = (f * f) >> 16;
    let f3 = (f2 * f) >> 16;
    let q = ((45607 * f) >> 16) + ((14741 * f2) >> 16) + ((5185 * f3) >> 16); // <= 65509
    let s = 1u32 << e;
    // s + (s * q >> 16) <= 65522: fits.
    (s + ((s * q) >> 16)) as u16
}

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

/// A non-finite sample counts as 0 (spec section 4).
#[inline(always)]
fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Native samples keep their value; this is the one the output carries.
#[inline(always)]
fn native(m: &Mosaic, x: usize, y: usize) -> f32 {
    finite_or_zero(m.data[y * m.width + x])
}

/// `max(v, 1)` cubed as an exact product (spec section 3.7).
#[inline(always)]
fn cube_floor1(v: f64) -> f64 {
    let t = v.max(1.0);
    t * t * t
}

/// Where the chroma photosites of row `y` sit: the parity of their `x` and whether they are red
/// (a red row: red and green alternate) or blue (a blue row). The green sites are the other parity.
#[inline(always)]
fn row_layout(m: &Mosaic, y: usize) -> (usize, bool) {
    if m.cfa.color_at(0, y) == CfaColor::Green {
        (1, m.cfa.color_at(1, y) == CfaColor::Red)
    } else {
        (0, m.cfa.color_at(0, y) == CfaColor::Red)
    }
}

/// An output image of the right size and placement, black. The planes of a mosaic whose data does
/// not match its size are not read: it gives this black image (the types of the crate keep the two
/// in step; `Mosaic::from_data` checks it).
fn empty_output(m: &Mosaic) -> RgbImage {
    let mut out = RgbImage::new(m.width, m.height);
    out.place = m.place;
    out
}

/// Whether there is anything to compute: a non-empty buffer whose data matches its size.
fn usable(m: &Mosaic) -> bool {
    m.width > 0 && m.height > 0 && m.data.len() == m.width * m.height
}

// ---------------------------------------------------------------------------------------------
// Fixed-point form (normative for the vectors' `expected`)
// ---------------------------------------------------------------------------------------------

/// Quantises a sample to the working scale of the headroom: `round_half_even(v * S / H)` clamped to
/// `0..=65535`; negatives, `NaN` and infinities give 0 (spec section 3.1).
#[inline]
fn quantise(v: f32, to_dn: f64) -> u16 {
    let d = f64::from(v) * to_dn;
    if !d.is_finite() || d <= 0.0 {
        0
    } else if d >= CODE_MAX {
        65535
    } else {
        d.round_ties_even() as u16
    }
}

/// Rounds a reconstruction result to a code: `v <= 0` or `NaN` give 0, `v >= 65535` gives 65535,
/// otherwise round half to even (spec section 3.7).
#[inline]
fn round_sat_u16(v: f64) -> u16 {
    if v.is_nan() || v <= 0.0 {
        0
    } else if v >= CODE_MAX {
        65535
    } else {
        v.round_ties_even() as u16
    }
}

/// The rounded halving add of two codes, `floor((a + b + 1) / 2)`.
#[inline(always)]
fn rhalf(a: u16, b: u16) -> u32 {
    (u32::from(a) + u32::from(b) + 1) >> 1
}

/// Output of a guide filter: `clamp(floor((acc + 512) / 1024), 0, 65535)`.
#[inline(always)]
fn filter_out(acc: i64) -> u16 {
    let t = acc + 512;
    if t < 0 {
        0
    } else {
        (t / 1024).min(65535) as u16
    }
}

/// The fixed-point form (spec sections 3.1 to 3.8), the normative one for the vectors' `expected`.
///
/// `noise` sets the offset `p` (only its read-noise part is used), `guide_detail` the sharpness of
/// the guide (clamped to `-50..=100`) and `headroom_log2` the headroom `H = 2^headroom_log2`
/// (clamped to `0..=4`). The headroom must cover the largest sample that should survive the
/// interpolation: interpolated samples saturate at `H`, native samples always pass through. Every
/// tile of a frame must use the frame's headroom ([`LogGuidedParams::for_mosaic`]); with it, tiles
/// that carry [`HALO_PX`] pixels of context give the whole-frame result bit for bit.
///
/// A mosaic whose data does not match its size gives a black image.
#[must_use]
pub fn demosaic_fixed(
    m: &Mosaic,
    noise: NoiseDn,
    guide_detail: i32,
    headroom_log2: u8,
) -> RgbImage {
    let mut out = empty_output(m);
    if !usable(m) {
        return out;
    }
    let (w, h) = (m.width, m.height);
    let d = derive(noise, guide_detail, headroom_log2);
    let p = d.p;
    let b = HALO_PX;
    let (wp, hp) = (w + 2 * b, h + 2 * b);
    let to_dn = WORKING_SCALE / d.headroom;
    let from_dn_num = d.headroom;

    // 1. Quantise and mirror-pad.
    let mut a = vec![0u16; wp * hp];
    for (yy, row) in a.chunks_exact_mut(wp).enumerate() {
        let sy = mirror(yy as isize - b as isize, h);
        let src = &m.data[sy * w..(sy + 1) * w];
        for (dst, &v) in row[b..b + w].iter_mut().zip(src) {
            *dst = quantise(v, to_dn);
        }
        for x in (0..b).chain(b + w..wp) {
            row[x] = row[b + mirror(x as isize - b as isize, w)];
        }
    }

    // 2. Log transform: L = log2_fixed(p + ceil(A / 2)) (1 <= y <= 65535).
    let mut l: Vec<u16> = a
        .iter()
        .map(|&v| log2_fixed((p + ((u32::from(v) + 1) >> 1)) as u16))
        .collect();

    // 3. Horizontal 5-tap guide filter, columns 2..wp-2.
    let mut lh = vec![0u16; wp * hp];
    for y in 0..hp {
        let lrow = &l[y * wp..(y + 1) * wp];
        let hrow = &mut lh[y * wp..(y + 1) * wp];
        for (dst, win) in hrow[2..wp - 2].iter_mut().zip(lrow.windows(5)) {
            let acc: i64 = d
                .kernel
                .iter()
                .zip(win)
                .map(|(&k, &v)| i64::from(k) * i64::from(v))
                .sum();
            *dst = filter_out(acc);
        }
    }

    // 4. Vertical 5-tap guide filter and exp2: the guide C, rows and columns 2..size-2.
    // C overwrites L: nothing reads L any more.
    for y in 2..hp - 2 {
        for x in 2..wp - 2 {
            let acc: i64 = (0..5)
                .map(|j| i64::from(d.kernel[j]) * i64::from(lh[(y - 2 + j) * wp + x]))
                .sum();
            l[y * wp + x] = exp2_fixed(filter_out(acc));
        }
    }
    let c = l;

    // 5. Structure measures (second-difference magnitude + p), 3..size-3. They overwrite Lh and a
    // second plane.
    let mut dh = lh;
    let mut dv = vec![0u16; wp * hp];
    for y in 3..hp - 3 {
        for x in 3..wp - 3 {
            let i = y * wp + x;
            let cc = u32::from(c[i]);
            let hh = rhalf(c[i - 1], c[i + 1]);
            let vv = rhalf(c[i - wp], c[i + wp]);
            dh[i] = (hh.abs_diff(cc) + p).min(65535) as u16;
            dv[i] = (vv.abs_diff(cc) + p).min(65535) as u16;
        }
    }

    // 6. 5x5 box mean of both measures, floor(sum / 25), at the output core (5..size-5).
    let mut bh = vec![0u16; wp * hp];
    let mut bv = vec![0u16; wp * hp];
    for y in b..hp - b {
        for x in b..wp - b {
            let (mut sh, mut sv) = (0u32, 0u32);
            for j in 0..5 {
                let row = (y - 2 + j) * wp + x - 2;
                for i in 0..5 {
                    sh += u32::from(dh[row + i]);
                    sv += u32::from(dv[row + i]);
                }
            }
            bh[y * wp + x] = (sh / 25) as u16;
            bv[y * wp + x] = (sv / 25) as u16;
        }
    }

    // 7. Guide-ratio reconstruction on the core.
    let n = w * h;
    let deq = |v: u16| (f64::from(v) * from_dn_num / WORKING_SCALE) as f32;
    let [rp, gp, bp] = out.planes_mut();
    debug_assert_eq!(rp.len(), n);
    for y in 0..h {
        let yy = y + b;
        let (pc, chroma_red) = row_layout(m, y);
        for x in 0..w {
            let xx = x + b;
            let i = yy * wp + xx;
            let cc = f64::from(c[i]);
            let ha = |ix: usize| f64::from(rhalf(a[ix - 1], a[ix + 1]));
            let va = |ix: usize| f64::from(rhalf(a[ix - wp], a[ix + wp]));
            let hc = |ix: usize| f64::from(rhalf(c[ix - 1], c[ix + 1]));
            let vc = |ix: usize| f64::from(rhalf(c[ix - wp], c[ix + wp]));
            let pos = y * w + x;
            let nat = native(m, x, y);
            if x % 2 == pc {
                // A red or blue site: native chroma, green from the structure weights, the other
                // chroma from the four diagonal neighbours.
                let wh = cube_floor1(f64::from(bv[i]));
                let wv = cube_floor1(f64::from(bh[i]));
                let num = wh * ha(i) + wv * va(i);
                let den = wh * hc(i) + wv * vc(i);
                let green = round_sat_u16(num * (cc / den));
                let sa = va(i - 1) + va(i + 1);
                let sc = vc(i - 1) + vc(i + 1);
                let diag = round_sat_u16(sa * (cc / sc));
                let (chroma, other) = if chroma_red {
                    (&mut *rp, &mut *bp)
                } else {
                    (&mut *bp, &mut *rp)
                };
                chroma[pos] = nat;
                gp[pos] = deq(green);
                other[pos] = deq(diag);
            } else {
                // A green site: the colour that owns the row from the horizontal neighbours, the
                // other from the vertical ones.
                let by_row = round_sat_u16((cc / hc(i)) * ha(i));
                let by_col = round_sat_u16((cc / vc(i)) * va(i));
                let (chroma, other) = if chroma_red {
                    (&mut *rp, &mut *bp)
                } else {
                    (&mut *bp, &mut *rp)
                };
                gp[pos] = nat;
                chroma[pos] = deq(by_row);
                other[pos] = deq(by_col);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// All-float form (the production path)
// ---------------------------------------------------------------------------------------------

/// The all-float form with the default guide detail 0 and the noise model of the photo: what
/// [`Method::LogGuided`](super::Method::LogGuided) runs (spec sections 3.9 and 7.2).
///
/// Same as [`demosaic_float`] with `guide_detail = 0`.
#[must_use]
pub fn demosaic(m: &Mosaic, noise: NoiseDn) -> RgbImage {
    demosaic_float(m, noise, 0)
}

/// The all-float form (spec section 3.9), the recommended production path.
///
/// The maths is the fixed form's without fixed point and without headroom: the planes are stored
/// as `f32`, the filters and the box are accumulated in `f64`, the reconstruction is `f64`, and
/// values above 1.0 keep their value. `noise` sets the offset `p` with `H = 1`, `guide_detail` the
/// sharpness of the guide (clamped to `-50..=100`). Interpolated samples are never negative;
/// native samples are copied unchanged (a non-finite sample is written as 0).
///
/// A mosaic whose data does not match its size gives a black image.
#[must_use]
pub fn demosaic_float(m: &Mosaic, noise: NoiseDn, guide_detail: i32) -> RgbImage {
    let mut out = empty_output(m);
    if !usable(m) {
        return out;
    }
    let (w, h) = (m.width, m.height);
    let d = derive(noise, guide_detail, 0);
    let p = f64::from(d.p);
    let kf = d.kernel.map(|k| f64::from(k) / KERNEL_DIVISOR);
    let b = HALO_PX;
    let (wp, hp) = (w + 2 * b, h + 2 * b);

    // 1. A = 65535 * max(v, 0), mirror-padded; no upper clamp and no headroom.
    let mut a = vec![0.0f32; wp * hp];
    for (yy, row) in a.chunks_exact_mut(wp).enumerate() {
        let sy = mirror(yy as isize - b as isize, h);
        let src = &m.data[sy * w..(sy + 1) * w];
        for (dst, &v) in row[b..b + w].iter_mut().zip(src) {
            *dst = (WORKING_SCALE * f64::from(finite_or_zero(v)).max(0.0)) as f32;
        }
        for x in (0..b).chain(b + w..wp) {
            row[x] = row[b + mirror(x as isize - b as isize, w)];
        }
    }

    // 2. L = log2(p + A / 2).
    let mut l: Vec<f32> = a
        .iter()
        .map(|&v| (p + 0.5 * f64::from(v)).log2() as f32)
        .collect();

    // 3. Horizontal 5-tap guide filter, columns 2..wp-2, no rounding.
    let mut lh = vec![0.0f32; wp * hp];
    for y in 0..hp {
        let lrow = &l[y * wp..(y + 1) * wp];
        let hrow = &mut lh[y * wp..(y + 1) * wp];
        for (dst, win) in hrow[2..wp - 2].iter_mut().zip(lrow.windows(5)) {
            let mut acc = 0.0f64;
            for (k, &v) in kf.iter().zip(win) {
                acc += k * f64::from(v);
            }
            *dst = acc as f32;
        }
    }

    // 4. Vertical filter and exp2: C = 2^max(Lv, 0), rows and columns 2..size-2. C overwrites L.
    for y in 2..hp - 2 {
        let rows: [&[f32]; 5] = std::array::from_fn(|j| &lh[(y - 2 + j) * wp..(y - 1 + j) * wp]);
        let crow = &mut l[y * wp..(y + 1) * wp];
        for x in 2..wp - 2 {
            let mut acc = 0.0f64;
            for (k, row) in kf.iter().zip(&rows) {
                acc += k * f64::from(row[x]);
            }
            crow[x] = acc.max(0.0).exp2() as f32;
        }
    }
    let c = l;

    // 5. Structure measures, 3..size-3. They overwrite Lh and a second plane.
    let mut dh = lh;
    let mut dv = vec![0.0f32; wp * hp];
    for y in 3..hp - 3 {
        let (up, mid, down) = (
            &c[(y - 1) * wp..y * wp],
            &c[y * wp..(y + 1) * wp],
            &c[(y + 1) * wp..(y + 2) * wp],
        );
        let (dh_row, dv_row) = (&mut dh[y * wp..(y + 1) * wp], &mut dv[y * wp..(y + 1) * wp]);
        for x in 3..wp - 3 {
            let cc = f64::from(mid[x]);
            let hh = 0.5 * (f64::from(mid[x - 1]) + f64::from(mid[x + 1]));
            let vv = 0.5 * (f64::from(up[x]) + f64::from(down[x]));
            dh_row[x] = ((hh - cc).abs() + p) as f32;
            dv_row[x] = ((vv - cc).abs() + p) as f32;
        }
    }

    // 6 and 7, row by row: the 5x5 box means and the guide-ratio reconstruction.
    //
    // The per-row work is split in dense passes that the compiler can vectorize (vertical sums
    // of the structure measures, the vertical means of the mosaic and of the guide) and two
    // stride-2 loops over the site classes. Slices are cut to exact lengths so the loops carry no
    // bounds checks.
    let [rp, gp, bp] = out.planes_mut();
    // Vertical 5-sums of the structure measures over the columns the box of this row reads
    // (padded columns 3..wp-3, that is w + 4 of them).
    let mut sum_h = vec![0.0f64; w + 4];
    let mut sum_v = vec![0.0f64; w + 4];
    // 0.5 * (up + down) of the mosaic and of the guide at padded columns b-1 ..= b+w (w + 2).
    let mut va_row = vec![0.0f64; w + 2];
    let mut vc_row = vec![0.0f64; w + 2];
    for y in 0..h {
        let yy = y + b;
        for (rows, sums) in [(&dh, &mut sum_h), (&dv, &mut sum_v)] {
            let r: [&[f32]; 5] = std::array::from_fn(|j| &rows[(yy - 2 + j) * wp + 3..][..w + 4]);
            for (i, s) in sums.iter_mut().enumerate() {
                *s = f64::from(r[0][i])
                    + f64::from(r[1][i])
                    + f64::from(r[2][i])
                    + f64::from(r[3][i])
                    + f64::from(r[4][i]);
            }
        }
        // Rows of the mosaic `a` and the guide `c` from padded column b-1: index x + 1 is the
        // pixel x, indices x and x + 2 its left and right neighbours.
        let a_up = &a[(yy - 1) * wp + b - 1..][..w + 2];
        let a_mid = &a[yy * wp + b - 1..][..w + 2];
        let a_dn = &a[(yy + 1) * wp + b - 1..][..w + 2];
        let c_up = &c[(yy - 1) * wp + b - 1..][..w + 2];
        let c_mid = &c[yy * wp + b - 1..][..w + 2];
        let c_dn = &c[(yy + 1) * wp + b - 1..][..w + 2];
        for ((o, &u), &d) in va_row.iter_mut().zip(a_up).zip(a_dn) {
            *o = 0.5 * (f64::from(u) + f64::from(d));
        }
        for ((o, &u), &d) in vc_row.iter_mut().zip(c_up).zip(c_dn) {
            *o = 0.5 * (f64::from(u) + f64::from(d));
        }
        let (pc, chroma_red) = row_layout(m, y);
        let (r_row, g_row, b_row) = (
            &mut rp[y * w..(y + 1) * w],
            &mut gp[y * w..(y + 1) * w],
            &mut bp[y * w..(y + 1) * w],
        );
        let (chroma_row, other_row) = if chroma_red {
            (r_row, b_row)
        } else {
            (b_row, r_row)
        };
        let nat_row = &m.data[y * w..(y + 1) * w];
        let deq = |v: f64| (v / WORKING_SCALE) as f32;
        // Red and blue sites. The structure measures are read at the site itself.
        for x in (pc..w).step_by(2) {
            let cc = f64::from(c_mid[x + 1]);
            let ha = 0.5 * (f64::from(a_mid[x]) + f64::from(a_mid[x + 2]));
            let hc = 0.5 * (f64::from(c_mid[x]) + f64::from(c_mid[x + 2]));
            let (va, vc) = (va_row[x + 1], vc_row[x + 1]);
            let sh = sum_h[x] + sum_h[x + 1] + sum_h[x + 2] + sum_h[x + 3] + sum_h[x + 4];
            let sv = sum_v[x] + sum_v[x + 1] + sum_v[x + 2] + sum_v[x + 3] + sum_v[x + 4];
            let bh = f64::from((sh / 25.0) as f32);
            let bv = f64::from((sv / 25.0) as f32);
            let wh = cube_floor1(bv);
            let wv = cube_floor1(bh);
            let num = wh * ha + wv * va;
            let den = wh * hc + wv * vc;
            let green = num * (cc / den);
            let diag = (va_row[x] + va_row[x + 2]) * (cc / (vc_row[x] + vc_row[x + 2]));
            chroma_row[x] = finite_or_zero(nat_row[x]);
            g_row[x] = deq(green);
            other_row[x] = deq(diag);
        }
        // Green sites.
        for x in (1 - pc..w).step_by(2) {
            let cc = f64::from(c_mid[x + 1]);
            let ha = 0.5 * (f64::from(a_mid[x]) + f64::from(a_mid[x + 2]));
            let hc = 0.5 * (f64::from(c_mid[x]) + f64::from(c_mid[x + 2]));
            let (va, vc) = (va_row[x + 1], vc_row[x + 1]);
            g_row[x] = finite_or_zero(nat_row[x]);
            chroma_row[x] = deq((cc / hc) * ha);
            other_row[x] = deq((cc / vc) * va);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_values_follow_spec_2_2() {
        // P = 100 -> k = -204, 50 -> -102, 25 -> -51, 0 -> 0, -25 -> 51, -50 -> 102.
        for (detail, k) in [
            (100, -204),
            (50, -102),
            (25, -51),
            (0, 0),
            (-25, 51),
            (-50, 102),
        ] {
            let d = derive(NoiseDn::ZERO, detail, 0);
            assert_eq!(d.k, k, "guide_detail {detail}");
            assert_eq!(d.kernel, [k, 256, 512 - 2 * k, 256, k]);
            assert_eq!(d.kernel.iter().sum::<i32>(), 1024);
        }
        // Out-of-range detail is clamped.
        assert_eq!(derive(NoiseDn::ZERO, 1000, 0).k, -204);
        assert_eq!(derive(NoiseDn::ZERO, -1000, 0).k, 102);
        // p = clamp(floor(r_dn / 2H), 1, 32767).
        let noise = NoiseDn::new(0.0, 21.0);
        assert_eq!(derive(noise, 0, 0).p, 10);
        assert_eq!(derive(noise, 0, 2).p, 2);
        assert_eq!(derive(NoiseDn::ZERO, 0, 0).p, 1);
        assert_eq!(derive(NoiseDn::new(0.0, 1e9), 0, 0).p, 32767);
        // An invalid read noise counts as 0; the headroom exponent is clamped to 4.
        for r in [f64::NAN, f64::INFINITY, -3.0] {
            let d = derive(NoiseDn::new(0.0, r), 0, 9);
            assert_eq!(
                (d.p, d.read_noise_dn, d.headroom),
                (1, 0.0, 16.0),
                "r = {r}"
            );
        }
    }

    #[test]
    fn the_default_headroom_rule_is_the_smallest_power_of_two_up_to_16() {
        let n = LogGuidedParams::headroom_log2_for;
        assert_eq!(
            [n(0.0), n(0.7), n(1.0), n(1.0001), n(2.0), n(3.4), n(4.0)],
            [0, 0, 0, 1, 1, 2, 2]
        );
        assert_eq!([n(4.5), n(8.0), n(8.01), n(16.0), n(1e9)], [3, 3, 4, 4, 4]);
        assert_eq!(n(f32::NAN), 2);
        assert_eq!(n(f32::INFINITY), 2);
    }
}
