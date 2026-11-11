//! Blemish removal (`docs/27` section 6): the expensive, cached **map** and the cheap **apply**.
//!
//! The map ([`blemish_map`]) runs a [`BlemishProposer`] over the full-resolution face crop, guards
//! its chroma and turns the difference into a per-pixel linear-light gain and a strength. The apply
//! ([`apply_blemish`]) reads that map at the working resolution through a threshold on the strength
//! (the Blemish amount), so moving the amount never reruns the proposer.
//!
//! Every per-pixel step is a pure free function ([`chroma_confidence`], [`chroma_pull`],
//! [`blemish_gain`], [`gain_strength`], [`threshold_strength`], [`apply_gain`]) so a GPU kernel can
//! follow with the same maths. Image-sized passes run over rows with `rayon` and write disjoint rows;
//! every sum has a fixed order, so results do not depend on the thread count.

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{Result, RetouchError};
use crate::models::BlemishProposer;
use crate::types::{LinearAwg4Image, Mask, Rect, check_render_scale};

/// The face size, in pixels, the proposer works at (`docs/27` 6.1 step 1): larger crops are
/// area-downscaled to it, smaller ones are never upscaled.
pub const TARGET_FACE_PX: f64 = 450.0;

/// The chroma guard runs when the longer side of the (downscaled) crop is at least this.
pub const CHROMA_GUARD_MIN_SIDE_PX: usize = 80;

/// The smallest gain and the largest one.
pub const GAIN_MIN: f64 = 0.5;
/// See [`GAIN_MIN`].
pub const GAIN_MAX: f64 = 2.5;

/// Radius of the max filter and of the box blur that turn the gain into the strength.
const STRENGTH_RADIUS_PX: usize = 3;

/// The blemish map of one face (`docs/27` 6.1 step 7).
#[derive(Debug, Clone, PartialEq)]
pub struct BlemishMap {
    /// The face rectangle in full-resolution image pixels that the map covers.
    pub rect: Rect,
    /// Map width in pixels (the crop width, or less when it was downscaled).
    pub width_px: u32,
    /// Map height in pixels.
    pub height_px: u32,
    /// `width_px * height_px` entries of `(gain.r, gain.g, gain.b, alpha)`, rows top to bottom.
    /// The gain is a linear-light multiplier in 0.5..=2.5; `alpha` is the strength.
    pub data: Vec<[f32; 4]>,
}

impl BlemishMap {
    /// Checks that both sides are at least 1, the data length matches and `rect` is not empty.
    pub fn validate(&self) -> Result<()> {
        let expected = (self.width_px as usize).checked_mul(self.height_px as usize);
        if self.width_px == 0 || self.height_px == 0 || expected != Some(self.data.len()) {
            return Err(RetouchError::Size {
                what: "blemish map",
                width: self.width_px,
                height: self.height_px,
                len: self.data.len(),
            });
        }
        if self.rect.width_px == 0 || self.rect.height_px == 0 {
            return Err(RetouchError::Size {
                what: "blemish map rectangle",
                width: self.rect.width_px,
                height: self.rect.height_px,
                len: self.data.len(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Pure per-pixel kernels
// ---------------------------------------------------------------------------------------------

/// The proposer's encoding of one channel: times `rgb_gain`, then `sqrt` (`docs/27` 6.1 step 2,
/// before the mean normalisation). Negative and NaN values encode as 0.
pub fn encode_channel(value: f32, rgb_gain: f32) -> f64 {
    (f64::from(value) * f64::from(rgb_gain)).max(0.0).sqrt()
}

/// The chroma-guard confidence of one pixel (`docs/27` 6.1 step 4), from the proposer output `out`
/// and its input `inp`, both encoded. 1 when the proposal keeps the colour ratios, near 0 when it
/// tints.
///
/// `r_c = ((max(out_c, 0) + 1e-5) / (max(inp_c, 0) + 1e-5))^2`, then
/// `conf = clamp(1.1 - 10 * (1 - min(r) / max(r)), 1e-5, 1)`, and 1 when `min == max`.
pub fn chroma_confidence(out: [f64; 3], inp: [f64; 3]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in 0..3 {
        let ratio = (out[c].max(0.0) + 1e-5) / (inp[c].max(0.0) + 1e-5);
        let r = ratio * ratio;
        lo = lo.min(r);
        hi = hi.max(r);
    }
    if lo == hi {
        return 1.0;
    }
    (1.1 - 10.0 * (1.0 - lo / hi)).clamp(1e-5, 1.0)
}

/// Rec. 601 luma weights of the chroma guard.
fn guard_luma(p: [f64; 3]) -> f64 {
    0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
}

/// The chroma-guard replacement of one pixel (`docs/27` 6.1 step 4): `out` is the proposer output,
/// `blurred` the box blur of `out * conf` and `conf` the pixel's confidence. Returns the colour of
/// the local average at the luma of `out`: `lerp(blurred * k, out, conf^2)` with
/// `k = luma(out) / luma(blurred)`. A pixel with `conf >= 1`, or with `luma(blurred) <= 0`, is
/// returned unchanged.
pub fn chroma_pull(out: [f64; 3], blurred: [f64; 3], conf: f64) -> [f64; 3] {
    if conf >= 1.0 {
        return out;
    }
    let lb = guard_luma(blurred);
    if lb <= 0.0 || !lb.is_finite() {
        return out;
    }
    let k = guard_luma(out) / lb;
    let w = conf * conf;
    let mut result = [0.0; 3];
    for c in 0..3 {
        let target = blurred[c] * k;
        result[c] = target + (out[c] - target) * w;
    }
    result
}

/// The linear-light gain of one channel at one map pixel (`docs/27` 6.1 step 5).
///
/// `out_c` and `in_c` are the (guarded) proposer output and its input, both encoded; `mask` is the
/// blemish mask byte; `edge_distance_px` is `d = min(x, w - 1 - x, y, h - 1 - y)`. The squared ratio
/// undoes the square root of the encoding; the gain is clamped to 0.5..=2.5, 1 where the mask is 0,
/// and fades to 1 over the two outermost rings (`d < 2`).
pub fn blemish_gain(out_c: f64, in_c: f64, mask: u8, edge_distance_px: u32) -> f64 {
    const FLOOR: f64 = 1.0 / 256.0;
    let ratio = out_c.max(FLOOR) / in_c.max(FLOOR);
    let mut g = (ratio * ratio).clamp(GAIN_MIN, GAIN_MAX);
    g = 1.0 + (f64::from(mask) / 255.0) * (g - 1.0);
    if edge_distance_px < 2 {
        g = 1.0 + (f64::from(edge_distance_px) / 2.0) * (g - 1.0);
    }
    g
}

/// The strength of a gain: `max(|log2 g.r|, |log2 g.g|, |log2 g.b|, 1e-6)` (`docs/27` 6.1 step 6,
/// before the max filter and the blur).
pub fn gain_strength(gain: [f64; 3]) -> f64 {
    gain.iter()
        .map(|g| g.log2().abs())
        .fold(1e-6, |acc, v| if v > acc { v } else { acc })
}

/// The threshold of the apply (`docs/27` 6.2): from the map strength `alpha`, the Blemish amount
/// `a` in 0..=1 and the protection mask byte, the mix `s` between no correction (0) and the full
/// gain (1).
///
/// ```text
/// s = clamp(alpha / max(0.01, 1 - a), 0, 1)
/// if a < 0.25: s = s * 4 * a
/// if a > 0.75: s = lerp(s, 1, 4 * (a - 0.75))
/// s = s * (1 - protect / 255)
/// ```
///
/// Low amounts fix only strong blemishes; high amounts apply every correction. For fixed `alpha`
/// and `protect`, `s` never decreases as `a` grows.
pub fn threshold_strength(alpha: f64, a: f64, protect: u8) -> f64 {
    let a = a.clamp(0.0, 1.0);
    let mut s = (alpha / (1.0 - a).max(0.01)).clamp(0.0, 1.0);
    if a < 0.25 {
        s *= 4.0 * a;
    }
    if a > 0.75 {
        s += (1.0 - s) * (4.0 * (a - 0.75));
    }
    s * (1.0 - f64::from(protect) / 255.0)
}

/// Applies a gain to a pixel: `rgb * lerp(1, g, s)` per channel, in `f64`, stored as `f32`.
pub fn apply_gain(rgb: [f32; 3], gain: [f64; 3], s: f64) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = (f64::from(rgb[c]) * (1.0 + (gain[c] - 1.0) * s)) as f32;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Buffers and filters
// ---------------------------------------------------------------------------------------------

/// Three planes, one per channel, `w * h` values each (the proposer's layout is planar).
#[derive(Debug, Clone)]
struct Planar {
    planes: [Vec<f32>; 3],
}

impl Planar {
    fn pixel(&self, i: usize) -> [f64; 3] {
        [
            f64::from(self.planes[0][i]),
            f64::from(self.planes[1][i]),
            f64::from(self.planes[2][i]),
        ]
    }
}

/// Area-average downscale of `N`-channel `f32` data from `(w, h)` to `(nw, nh)` (both not larger
/// than the source). Each destination pixel averages the source area it covers, partial pixels
/// weighted by their coverage, in a fixed order.
fn area_downscale<const N: usize>(
    src: &[[f32; N]],
    (w, h): (usize, usize),
    (nw, nh): (usize, usize),
) -> Vec<[f32; N]> {
    let (sx, sy) = (w as f64 / nw as f64, h as f64 / nh as f64);
    let mut out = vec![[0.0f32; N]; nw * nh];
    out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
        let (y0, y1) = (y as f64 * sy, (y + 1) as f64 * sy);
        for (x, px) in row.iter_mut().enumerate() {
            let (x0, x1) = (x as f64 * sx, (x + 1) as f64 * sx);
            let mut acc = [0.0f64; N];
            let mut total = 0.0f64;
            let (iy0, iy1) = (y0.floor() as usize, (y1.ceil() as usize).min(h));
            let (ix0, ix1) = (x0.floor() as usize, (x1.ceil() as usize).min(w));
            for iy in iy0..iy1 {
                let wy = (y1.min((iy + 1) as f64) - y0.max(iy as f64)).max(0.0);
                for ix in ix0..ix1 {
                    let wx = (x1.min((ix + 1) as f64) - x0.max(ix as f64)).max(0.0);
                    let wt = wx * wy;
                    total += wt;
                    for c in 0..N {
                        acc[c] += wt * f64::from(src[iy * w + ix][c]);
                    }
                }
            }
            for c in 0..N {
                px[c] = if total > 0.0 {
                    (acc[c] / total) as f32
                } else {
                    0.0
                };
            }
        }
    });
    out
}

/// Box blur of one plane with radius `r` (side `2r + 1`). At the borders the window shrinks to the
/// pixels inside the plane. Sums are prefix sums in `f64`, in a fixed order.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    // Horizontal pass, rows in parallel.
    let mut horizontal = vec![0.0f64; w * h];
    horizontal
        .par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(dst, row)| {
            let mut prefix = vec![0.0f64; w + 1];
            for (i, v) in row.iter().enumerate() {
                prefix[i + 1] = prefix[i] + f64::from(*v);
            }
            for (x, d) in dst.iter_mut().enumerate() {
                let (lo, hi) = (x.saturating_sub(r), (x + r + 1).min(w));
                *d = (prefix[hi] - prefix[lo]) / (hi - lo) as f64;
            }
        });
    // Vertical pass: column prefix sums built row by row, then output rows in parallel.
    let mut prefix = vec![0.0f64; w * (h + 1)];
    for y in 0..h {
        for x in 0..w {
            prefix[(y + 1) * w + x] = prefix[y * w + x] + horizontal[y * w + x];
        }
    }
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (lo, hi) = (y.saturating_sub(r), (y + r + 1).min(h));
        let n = (hi - lo) as f64;
        for (x, o) in row.iter_mut().enumerate() {
            *o = ((prefix[hi * w + x] - prefix[lo * w + x]) / n) as f32;
        }
    });
    out
}

/// Square max filter of radius `r` (side `2r + 1`), ignoring pixels outside the plane.
fn max_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0f32; w * h];
    horizontal
        .par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(dst, row)| {
            for (x, d) in dst.iter_mut().enumerate() {
                let (lo, hi) = (x.saturating_sub(r), (x + r + 1).min(w));
                *d = row[lo..hi].iter().copied().fold(f32::MIN, f32::max);
            }
        });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (lo, hi) = (y.saturating_sub(r), (y + r + 1).min(h));
        for (x, o) in row.iter_mut().enumerate() {
            *o = (lo..hi)
                .map(|yy| horizontal[yy * w + x])
                .fold(f32::MIN, f32::max);
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------------------------

/// Sorted tile origins along one axis of length `len` for output tiles of side `tile`: a regular
/// grid whose last tile moves inward to end at the edge; one origin 0 when `len <= tile`.
fn tile_origins(len: usize, tile: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let mut origins: Vec<usize> = (0..len).step_by(tile).map(|o| o.min(len - tile)).collect();
    origins.dedup();
    origins
}

/// Runs the proposer over `input` (size `w x h`) in tiles (`docs/27` 6.1 step 3).
fn propose(input: &Planar, w: usize, h: usize, proposer: &dyn BlemishProposer) -> Result<Planar> {
    let tile = proposer.output_size_px() as usize;
    let border = proposer.border_px() as usize;
    if tile == 0 {
        return Err(RetouchError::Model(
            "the proposer has an output tile of 0 pixels".into(),
        ));
    }
    let side = tile + 2 * border;
    let mut out = Planar {
        planes: [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]],
    };
    let mut tile_in = vec![0.0f32; 3 * side * side];
    for ty in tile_origins(h, tile) {
        for tx in tile_origins(w, tile) {
            for c in 0..3 {
                for j in 0..side {
                    let sy = (ty as i64 + j as i64 - border as i64).clamp(0, h as i64 - 1) as usize;
                    let dst = c * side * side + j * side;
                    for i in 0..side {
                        let sx =
                            (tx as i64 + i as i64 - border as i64).clamp(0, w as i64 - 1) as usize;
                        tile_in[dst + i] = input.planes[c][sy * w + sx];
                    }
                }
            }
            let tile_out = proposer.run(&tile_in)?;
            if tile_out.len() != 3 * tile * tile {
                return Err(RetouchError::Model(format!(
                    "the proposer returned {} values, expected {}",
                    tile_out.len(),
                    3 * tile * tile
                )));
            }
            for c in 0..3 {
                for j in 0..tile.min(h - ty) {
                    let src = c * tile * tile + j * tile;
                    let n = tile.min(w - tx);
                    let dst = (ty + j) * w + tx;
                    out.planes[c][dst..dst + n].copy_from_slice(&tile_out[src..src + n]);
                }
            }
        }
    }
    Ok(out)
}

/// The chroma guard (`docs/27` 6.1 step 4). Returns `out` unchanged when `max(w, h) < 80`.
fn chroma_guard(input: &Planar, out: &Planar, w: usize, h: usize) -> Planar {
    if w.max(h) < CHROMA_GUARD_MIN_SIDE_PX {
        return out.clone();
    }
    let radius = w.max(h) / CHROMA_GUARD_MIN_SIDE_PX;
    let n = w * h;
    let conf: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| chroma_confidence(out.pixel(i), input.pixel(i)))
        .collect();
    let weighted: Vec<Vec<f32>> = (0..3)
        .map(|c| {
            out.planes[c]
                .iter()
                .zip(&conf)
                .map(|(v, k)| (f64::from(*v) * k) as f32)
                .collect()
        })
        .collect();
    let blurred = Planar {
        planes: [
            box_blur(&weighted[0], w, h, radius),
            box_blur(&weighted[1], w, h, radius),
            box_blur(&weighted[2], w, h, radius),
        ],
    };
    let mut result = out.clone();
    let [p0, p1, p2] = &mut result.planes;
    p0.par_chunks_mut(w)
        .zip(p1.par_chunks_mut(w))
        .zip(p2.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((r0, r1), r2))| {
            if y < radius || y + radius >= h {
                return;
            }
            for x in radius..w.saturating_sub(radius) {
                let i = y * w + x;
                if conf[i] >= 1.0 {
                    continue;
                }
                let pulled = chroma_pull(out.pixel(i), blurred.pixel(i), conf[i]);
                r0[x] = pulled[0] as f32;
                r1[x] = pulled[1] as f32;
                r2[x] = pulled[2] as f32;
            }
        });
    result
}

/// Builds the blemish map of one face (`docs/27` 6.1).
///
/// - `crop`: the full-resolution linear AWG4 crop over `rect`; its size must equal the rectangle's.
/// - `mask`: the blemish mask over the same rectangle, same size as `crop`.
/// - `rect`: the face rectangle in full-resolution pixels, stored in the map for the apply.
/// - `rgb_gain`: per-channel gain before encoding; `[1.0; 3]` when the input is white-balanced.
/// - `face_px`: the face size in pixels of the crop. The crop and mask are area-downscaled by
///   `450 / max(1, face_px)` when that is below 1, and never upscaled.
/// - `proposer`: makes the cleaned copy.
///
/// The map does not depend on the Blemish amount, so it can be cached ([`BlemishMapCache`]).
pub fn blemish_map(
    crop: &LinearAwg4Image,
    mask: &Mask,
    rect: Rect,
    rgb_gain: [f32; 3],
    face_px: f64,
    proposer: &dyn BlemishProposer,
) -> Result<BlemishMap> {
    crop.validate()?;
    mask.validate()?;
    if (crop.width_px, crop.height_px) != mask.size_px() {
        return Err(RetouchError::MaskSizeMismatch {
            left: (crop.width_px, crop.height_px),
            right: mask.size_px(),
        });
    }
    if (crop.width_px, crop.height_px) != (rect.width_px, rect.height_px) {
        return Err(RetouchError::RegionMismatch {
            region: (rect.width_px, rect.height_px),
            mask: (crop.width_px, crop.height_px),
        });
    }

    // Step 1: size.
    let (cw, ch) = (crop.width_px as usize, crop.height_px as usize);
    let face_px = if face_px.is_finite() { face_px } else { 1.0 };
    let k = TARGET_FACE_PX / face_px.max(1.0);
    let (image, mask_bytes, w, h) = if k < 1.0 {
        let nw = ((cw as f64 * k).round() as usize).clamp(1, cw);
        let nh = ((ch as f64 * k).round() as usize).clamp(1, ch);
        let image = area_downscale(&crop.data, (cw, ch), (nw, nh));
        let m: Vec<[f32; 1]> = mask.data.iter().map(|v| [f32::from(*v)]).collect();
        let m = area_downscale(&m, (cw, ch), (nw, nh));
        let bytes: Vec<u8> = m
            .iter()
            .map(|v| v[0].round().clamp(0.0, 255.0) as u8)
            .collect();
        (std::borrow::Cow::Owned(image), bytes, nw, nh)
    } else {
        (
            std::borrow::Cow::Borrowed(&crop.data[..]),
            mask.data.clone(),
            cw,
            ch,
        )
    };

    // Step 2: encode. The mean is a plain sequential f64 sum, so it never depends on threads.
    let mut sum = 0.0f64;
    for p in image.iter() {
        let e = [
            encode_channel(p[0], rgb_gain[0]),
            encode_channel(p[1], rgb_gain[1]),
            encode_channel(p[2], rgb_gain[2]),
        ];
        sum += (e[0] + 2.0 * e[1] + e[2]) / 4.0;
    }
    let mean = sum / (w * h) as f64;
    let norm = if mean > 0.0 && mean.is_finite() {
        0.5 / mean
    } else {
        1.0
    };
    let mut input = Planar {
        planes: [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]],
    };
    for (c, plane) in input.planes.iter_mut().enumerate() {
        plane
            .par_iter_mut()
            .zip(image.par_iter())
            .for_each(|(dst, p)| *dst = (encode_channel(p[c], rgb_gain[c]) * norm) as f32);
    }

    // Steps 3 and 4: propose, then guard the chroma.
    let proposed = propose(&input, w, h, proposer)?;
    let guarded = chroma_guard(&input, &proposed, w, h);

    // Step 5: gain.
    let mut gains = vec![[1.0f32; 3]; w * h];
    gains.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, g) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let d = x.min(w - 1 - x).min(y).min(h - 1 - y) as u32;
            let m = mask_bytes[i];
            for (c, gc) in g.iter_mut().enumerate() {
                *gc = blemish_gain(
                    f64::from(guarded.planes[c][i]),
                    f64::from(input.planes[c][i]),
                    m,
                    d,
                ) as f32;
            }
        }
    });

    // Step 6: strength.
    let strength: Vec<f32> = gains
        .par_iter()
        .map(|g| gain_strength([f64::from(g[0]), f64::from(g[1]), f64::from(g[2])]) as f32)
        .collect();
    let strength = max_filter(&strength, w, h, STRENGTH_RADIUS_PX);
    let alpha = box_blur(&strength, w, h, STRENGTH_RADIUS_PX);

    // Step 7: result, at the size of step 1.
    let data = gains
        .iter()
        .zip(&alpha)
        .map(|(g, a)| [g[0], g[1], g[2], *a])
        .collect();
    Ok(BlemishMap {
        rect,
        width_px: w as u32,
        height_px: h as u32,
        data,
    })
}

// ---------------------------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------------------------

/// Applies one face's blemish map to the working image (`docs/27` 6.2).
///
/// - `a`: the Blemish amount in 0..=1 (`SkinSettings::a`); with `a <= 0` nothing changes.
/// - `render_scale`: the working size over the full-resolution size (0.0001..=1).
/// - `protect`: an optional protection mask in working coordinates (the size of `image`); 255
///   leaves the pixel alone.
///
/// Pixels outside the map are not touched. The map is sampled bilinearly at pixel centres, the
/// `+1` neighbour clamped to the last index.
pub fn apply_blemish(
    image: &mut LinearAwg4Image,
    map: &BlemishMap,
    a: f64,
    render_scale: f32,
    protect: Option<&Mask>,
) -> Result<()> {
    let scale = check_render_scale(render_scale)?;
    image.validate()?;
    map.validate()?;
    if let Some(p) = protect {
        p.validate()?;
        if p.size_px() != (image.width_px, image.height_px) {
            return Err(RetouchError::MaskSizeMismatch {
                left: (image.width_px, image.height_px),
                right: p.size_px(),
            });
        }
    }
    let a = if a.is_nan() { 0.0 } else { a.clamp(0.0, 1.0) };
    if a <= 0.0 {
        return Ok(());
    }
    let w = image.width_px as usize;
    let (mw, mh) = (map.width_px as usize, map.height_px as usize);
    let sx = f64::from(map.width_px) / f64::from(map.rect.width_px);
    let sy = f64::from(map.height_px) / f64::from(map.rect.height_px);
    let (rx, ry) = (f64::from(map.rect.x_px), f64::from(map.rect.y_px));
    // A conservative column range, refined per pixel by the exact test.
    let col_lo = (((rx - 1.0) * scale).floor().max(0.0)) as usize;
    let col_hi = ((((map.rect.right_px() as f64) + 1.0) * scale)
        .ceil()
        .max(0.0) as usize)
        .min(w);

    image
        .data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let v = sy * (py as f64 / scale - ry + 0.5) - 0.5;
            if !(v >= 0.0 && v < mh as f64) {
                return;
            }
            let y0 = v.floor() as usize;
            let y1 = (y0 + 1).min(mh - 1);
            let fy = v - y0 as f64;
            #[allow(clippy::needless_range_loop)] // px is also the protect-mask column
            for px in col_lo..col_hi {
                let u = sx * (px as f64 / scale - rx + 0.5) - 0.5;
                if !(u >= 0.0 && u < mw as f64) {
                    continue;
                }
                let x0 = u.floor() as usize;
                let x1 = (x0 + 1).min(mw - 1);
                let fx = u - x0 as f64;
                let mut sample = [0.0f64; 4];
                for (c, s) in sample.iter_mut().enumerate() {
                    let top = lerp(
                        f64::from(map.data[y0 * mw + x0][c]),
                        f64::from(map.data[y0 * mw + x1][c]),
                        fx,
                    );
                    let bottom = lerp(
                        f64::from(map.data[y1 * mw + x0][c]),
                        f64::from(map.data[y1 * mw + x1][c]),
                        fx,
                    );
                    *s = lerp(top, bottom, fy);
                }
                let protect_byte = protect.map_or(0, |p| p.data[py * w + px]);
                let s = threshold_strength(sample[3], a, protect_byte);
                row[px] = apply_gain(row[px], [sample[0], sample[1], sample[2]], s);
            }
        });
    Ok(())
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------------------------

/// What a set of blemish maps depends on (`docs/27` 6.1, last paragraph): the image, the faces and
/// everything upstream that changes the pixels the proposer sees. It does not include the Blemish
/// amount.
#[derive(Debug, Clone, PartialEq)]
pub struct BlemishMapKey {
    /// The identity of the image, chosen by the caller (for example a hash of the source and the
    /// upstream node settings); the crate treats it as opaque.
    pub image_key: u64,
    /// The number of faces.
    pub face_count: usize,
    /// Each face rectangle, in face order.
    pub face_rects: Vec<Rect>,
    /// A key of everything upstream that changes the pixels (white balance, other nodes); opaque.
    pub upstream_key: u64,
}

/// A cache of blemish maps for one image: the maps of every face under one [`BlemishMapKey`].
///
/// The cache holds one key at a time. Asking with a different key drops every map of the old one
/// (the preview and the export of the same image share a key, so they share maps). The retouch
/// cache of the pipeline holds one of these.
#[derive(Debug, Default)]
pub struct BlemishMapCache {
    key: Option<BlemishMapKey>,
    maps: BTreeMap<usize, Arc<BlemishMap>>,
}

impl BlemishMapCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The map of face `face_index` under `key`, built by `build` when it is not cached. A
    /// different `key` first drops the old maps. A failing `build` caches nothing.
    pub fn get_or_build(
        &mut self,
        key: &BlemishMapKey,
        face_index: usize,
        build: impl FnOnce() -> Result<BlemishMap>,
    ) -> Result<Arc<BlemishMap>> {
        if self.key.as_ref() != Some(key) {
            self.maps.clear();
            self.key = Some(key.clone());
        }
        if let Some(map) = self.maps.get(&face_index) {
            return Ok(Arc::clone(map));
        }
        let map = Arc::new(build()?);
        self.maps.insert(face_index, Arc::clone(&map));
        Ok(map)
    }

    /// The cached map of face `face_index`, if there is one under `key`.
    pub fn get(&self, key: &BlemishMapKey, face_index: usize) -> Option<Arc<BlemishMap>> {
        if self.key.as_ref() != Some(key) {
            return None;
        }
        self.maps.get(&face_index).map(Arc::clone)
    }

    /// The number of cached maps.
    pub fn len(&self) -> usize {
        self.maps.len()
    }

    /// True when no map is cached.
    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    /// Drops everything.
    pub fn clear(&mut self) {
        self.key = None;
        self.maps.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockProposer;
    use proptest::prelude::*;

    /// A deterministic textured image with colour variation.
    fn textured(w: u32, h: u32) -> LinearAwg4Image {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let a = ((x * 7 + y * 13) % 17) as f32 / 17.0;
                let b = ((x * 3 + y * 5) % 11) as f32 / 11.0;
                data.push([0.2 + 0.3 * a, 0.25 + 0.2 * b, 0.15 + 0.3 * a * b]);
            }
        }
        LinearAwg4Image::new(w, h, data).unwrap()
    }

    fn full_mask(w: u32, h: u32) -> Mask {
        Mask::new(w, h, vec![255; (w * h) as usize]).unwrap()
    }

    fn map_for(
        img: &LinearAwg4Image,
        mask: &Mask,
        face_px: f64,
        proposer: &dyn BlemishProposer,
    ) -> BlemishMap {
        let rect = Rect::new(0, 0, img.width_px, img.height_px);
        blemish_map(img, mask, rect, [1.0; 3], face_px, proposer).unwrap()
    }

    #[test]
    fn threshold_matches_hand_values() {
        // a < 0.25: s = clamp(0.05 / 0.9) * 4 * 0.1
        let s = threshold_strength(0.05, 0.1, 0);
        assert!((s - 0.05 / 0.9 * 0.4).abs() < 1e-12, "{s}");
        // 0.25 <= a <= 0.75: only the division. alpha 0.25 / (1 - 0.5) = 0.5
        assert!((threshold_strength(0.25, 0.5, 0) - 0.5).abs() < 1e-12);
        // clamp to 1
        assert_eq!(threshold_strength(0.9, 0.5, 0), 1.0);
        // a > 0.75: 0.05 / 0.1 = 0.5; lerp(0.5, 1, 4 * 0.15 = 0.6) = 0.8
        let s = threshold_strength(0.05, 0.9, 0);
        assert!((s - 0.8).abs() < 1e-12, "{s}");
        // a = 1: the 0.01 floor; alpha 0.001 / 0.01 = 0.1, then lerp(0.1, 1, 1) = 1
        assert_eq!(threshold_strength(0.001, 1.0, 0), 1.0);
        // a = 0: nothing is applied
        assert_eq!(threshold_strength(5.0, 0.0, 0), 0.0);
        // protection mask: half-protected halves it, fully protected zeroes it
        let full = threshold_strength(0.25, 0.5, 0);
        assert!((threshold_strength(0.25, 0.5, 255)).abs() < 1e-12);
        let half = threshold_strength(0.25, 0.5, 51);
        assert!((half - full * (1.0 - 51.0 / 255.0)).abs() < 1e-12);
    }

    #[test]
    fn gain_kernel_clamps_masks_and_fades() {
        // Out of range ratios clamp.
        assert_eq!(blemish_gain(3.0, 1.0, 255, 5), 2.5);
        assert_eq!(blemish_gain(0.1, 1.0, 255, 5), 0.5);
        // Square undoes the sqrt.
        assert!((blemish_gain(1.2, 1.0, 255, 5) - 1.44).abs() < 1e-12);
        // Mask 0 gives exactly 1; half mask is halfway.
        assert_eq!(blemish_gain(3.0, 1.0, 0, 5), 1.0);
        assert!((blemish_gain(3.0, 1.0, 51, 5) - (1.0 + 0.2 * 1.5)).abs() < 1e-12);
        // Border fade: d = 0 is 1, d = 1 is halfway.
        assert_eq!(blemish_gain(3.0, 1.0, 255, 0), 1.0);
        assert!((blemish_gain(3.0, 1.0, 255, 1) - 1.75).abs() < 1e-12);
        assert_eq!(blemish_gain(3.0, 1.0, 255, 2), 2.5);
        // The 1/256 floor.
        assert_eq!(blemish_gain(0.0, 0.0, 255, 5), 1.0);
    }

    #[test]
    fn map_gains_are_clamped_and_one_on_the_ring() {
        let img = textured(100, 100);
        let mask = full_mask(100, 100);
        for (factor, interior, d1) in [(3.0f32, 2.5f32, 1.75f32), (0.5, 0.5, 0.75)] {
            let map = map_for(&img, &mask, 100.0, &MockProposer::scale(64, factor));
            assert_eq!((map.width_px, map.height_px), (100, 100));
            for (i, e) in map.data.iter().enumerate() {
                let (x, y) = (i % 100, i / 100);
                let d = x.min(99 - x).min(y).min(99 - y);
                for v in &e[..3] {
                    assert!((0.5..=2.5).contains(v), "gain {v} out of range");
                    match d {
                        0 => assert_eq!(*v, 1.0),
                        1 => assert!((v - d1).abs() < 1e-6),
                        _ => assert!((v - interior).abs() < 1e-6),
                    }
                }
            }
        }
    }

    #[test]
    fn identity_proposer_gives_unit_gains_and_changes_nothing() {
        let img = textured(120, 90);
        let mask = full_mask(120, 90);
        let map = map_for(&img, &mask, 120.0, &MockProposer::identity(64));
        assert!(
            map.data
                .iter()
                .all(|e| e[0] == 1.0 && e[1] == 1.0 && e[2] == 1.0)
        );
        for a in [0.1, 0.5, 0.9, 1.0] {
            let mut out = img.clone();
            apply_blemish(&mut out, &map, a, 1.0, None).unwrap();
            assert_eq!(out, img, "a = {a}");
        }
    }

    #[test]
    fn a_zero_mask_gives_unit_gains() {
        let img = textured(60, 60);
        let mask = Mask::zeros(60, 60).unwrap();
        let map = map_for(&img, &mask, 60.0, &MockProposer::scale(64, 1.2));
        assert!(map.data.iter().all(|e| e[..3] == [1.0; 3]));
    }

    #[test]
    fn chroma_confidence_values() {
        // Equal scaling: exactly 1.
        assert_eq!(chroma_confidence([0.5; 3], [0.5; 3]), 1.0);
        let scaled = chroma_confidence([0.25, 0.3, 0.2], [0.5, 0.6, 0.4]);
        assert_eq!(scaled, 1.0, "equal scaling keeps conf at 1");
        // A 15 % tint on one channel: min/max = 1 / 1.3225, conf clamps to 1e-5.
        let tinted = chroma_confidence([0.575, 0.5, 0.5], [0.5, 0.5, 0.5]);
        assert_eq!(tinted, 1e-5);
        // A mild one lands strictly between.
        let mild = chroma_confidence([0.505, 0.5, 0.5], [0.5, 0.5, 0.5]);
        assert!(mild > 1e-5 && mild < 1.0, "{mild}");
    }

    #[test]
    fn chroma_pull_keeps_luma_and_takes_the_local_chroma() {
        let out = [0.8, 0.4, 0.3];
        let blurred = [0.3, 0.5, 0.2];
        let pulled = chroma_pull(out, blurred, 1e-5);
        let (lo, lb) = (guard_luma(out), guard_luma(pulled));
        assert!((lo - lb).abs() < 1e-9, "luma {lo} vs {lb}");
        // Same chroma ratios as the local average.
        for c in 0..3 {
            let want = blurred[c] / guard_luma(blurred);
            assert!((pulled[c] / lb - want).abs() < 1e-6);
        }
        // conf 1 and a black average leave the pixel alone.
        assert_eq!(chroma_pull(out, blurred, 1.0), out);
        assert_eq!(chroma_pull(out, [0.0; 3], 0.1), out);
    }

    #[test]
    fn chroma_guard_pulls_a_tint_back_and_leaves_equal_scaling() {
        let (w, h) = (96usize, 96usize);
        // A checker of two colours as the input.
        let mut input = Planar {
            planes: [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]],
        };
        for y in 0..h {
            for x in 0..w {
                let even = (x + y) % 2 == 0;
                let p = if even {
                    [0.6, 0.4, 0.3]
                } else {
                    [0.3, 0.45, 0.6]
                };
                for (plane, v) in input.planes.iter_mut().zip(p) {
                    plane[y * w + x] = v;
                }
            }
        }
        // Tinted: red times 1.15.
        let mut tinted = input.clone();
        tinted.planes[0].iter_mut().for_each(|v| *v *= 1.15);
        let guarded = chroma_guard(&input, &tinted, w, h);
        let radius = w / 80;
        let spread = |p: &Planar| {
            let mut ratios = Vec::new();
            for y in radius..h - radius {
                for x in radius..w - radius {
                    let q = p.pixel(y * w + x);
                    ratios.push(q[0] / q[2]);
                }
            }
            let (lo, hi) = ratios
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), r| (l.min(*r), h.max(*r)));
            hi - lo
        };
        assert!(
            spread(&guarded) < 0.2 * spread(&tinted),
            "chroma spread {} vs {}",
            spread(&guarded),
            spread(&tinted)
        );
        // Luma is kept where the guard acted.
        for y in radius..h - radius {
            for x in radius..w - radius {
                let i = y * w + x;
                let (a, b) = (guard_luma(guarded.pixel(i)), guard_luma(tinted.pixel(i)));
                assert!((a - b).abs() < 1e-4 * b.max(1.0), "luma {a} vs {b}");
            }
        }
        // Equal scaling: left alone, bit for bit.
        let mut scaled = input.clone();
        scaled.planes.iter_mut().for_each(|p| {
            p.iter_mut().for_each(|v| *v *= 0.8);
        });
        let guarded = chroma_guard(&input, &scaled, w, h);
        for c in 0..3 {
            assert_eq!(guarded.planes[c], scaled.planes[c]);
        }
        // Too small for the guard: returned as is.
        let small = chroma_guard(&input, &tinted, 79, 79);
        assert_eq!(small.planes[0], tinted.planes[0]);
    }

    #[test]
    fn tile_origins_cover_the_axis() {
        assert_eq!(tile_origins(10, 64), vec![0]);
        assert_eq!(tile_origins(64, 64), vec![0]);
        assert_eq!(tile_origins(100, 64), vec![0, 36]);
        assert_eq!(tile_origins(130, 64), vec![0, 64, 66]);
        assert_eq!(tile_origins(128, 64), vec![0, 64]);
    }

    #[test]
    fn tiling_leaves_no_seams() {
        // A pointwise proposer gives the same map for any tile size, also when the image is smaller
        // than a tile or not a multiple of it.
        for (w, h) in [(97u32, 131u32), (30, 20), (64, 64), (200, 40)] {
            let img = textured(w, h);
            let mask = full_mask(w, h);
            let face_px = f64::from(w.max(h));
            let reference = map_for(&img, &mask, face_px, &MockProposer::tint(1000, 1, 1.05));
            for tile in [16u32, 33, 64, 100] {
                let map = map_for(&img, &mask, face_px, &MockProposer::tint(tile, 1, 1.05));
                assert_eq!(map, reference, "{w}x{h} with tile {tile}");
            }
        }
    }

    #[test]
    fn a_failing_or_wrong_sized_proposer_is_an_error() {
        struct Bad(usize);
        impl BlemishProposer for Bad {
            fn output_size_px(&self) -> u32 {
                8
            }
            fn border_px(&self) -> u32 {
                0
            }
            fn run(&self, _input: &[f32]) -> Result<Vec<f32>> {
                Ok(vec![0.0; self.0])
            }
        }
        let img = textured(10, 10);
        let mask = full_mask(10, 10);
        let rect = Rect::new(0, 0, 10, 10);
        assert!(matches!(
            blemish_map(&img, &mask, rect, [1.0; 3], 10.0, &Bad(5)),
            Err(RetouchError::Model(_))
        ));
        assert!(blemish_map(&img, &mask, rect, [1.0; 3], 10.0, &Bad(3 * 64)).is_ok());
        // Mismatched sizes are refused, not panics.
        let other = full_mask(9, 10);
        assert!(blemish_map(&img, &other, rect, [1.0; 3], 10.0, &Bad(3 * 64)).is_err());
        assert!(
            blemish_map(
                &img,
                &mask,
                Rect::new(0, 0, 9, 10),
                [1.0; 3],
                10.0,
                &Bad(192)
            )
            .is_err()
        );
    }

    #[test]
    fn the_map_never_upscales() {
        let p = MockProposer::identity(64);
        // A small face: k = 9, the map is the crop.
        let img = textured(50, 40);
        let map = map_for(&img, &full_mask(50, 40), 50.0, &p);
        assert_eq!((map.width_px, map.height_px), (50, 40));
        // Exactly 450: k = 1, no resize.
        let img = textured(150, 120);
        let map = map_for(&img, &full_mask(150, 120), 450.0, &p);
        assert_eq!((map.width_px, map.height_px), (150, 120));
        // A big face: area downscale by k = 0.5.
        let img = textured(200, 300);
        let map = map_for(&img, &full_mask(200, 300), 900.0, &p);
        assert_eq!((map.width_px, map.height_px), (100, 150));
        assert_eq!(map.rect, Rect::new(0, 0, 200, 300));
        // Degenerate face sizes never upscale or divide by zero.
        let map = map_for(&textured(20, 20), &full_mask(20, 20), 0.0, &p);
        assert_eq!((map.width_px, map.height_px), (20, 20));
        let map = map_for(&textured(20, 20), &full_mask(20, 20), f64::NAN, &p);
        assert_eq!((map.width_px, map.height_px), (20, 20));
    }

    #[test]
    fn area_downscale_averages() {
        let src: Vec<[f32; 1]> = (0..16).map(|i| [i as f32]).collect();
        let out = area_downscale(&src, (4, 4), (2, 2));
        assert_eq!(out, vec![[2.5], [4.5], [10.5], [12.5]]);
        // Non-integer factor keeps the mean of a constant.
        let flat = vec![[0.7f32, 0.7, 0.7]; 7 * 5];
        let out = area_downscale(&flat, (7, 5), (3, 2));
        assert!(out.iter().all(|p| p.iter().all(|v| (v - 0.7).abs() < 1e-6)));
    }

    #[test]
    fn filters_behave() {
        let (w, h) = (9usize, 7usize);
        let mut src = vec![0.0f32; w * h];
        src[3 * w + 4] = 1.0;
        let m = max_filter(&src, w, h, 2);
        for y in 0..h {
            for x in 0..w {
                let inside = x.abs_diff(4) <= 2 && y.abs_diff(3) <= 2;
                assert_eq!(m[y * w + x], if inside { 1.0 } else { 0.0 });
            }
        }
        let b = box_blur(&src, w, h, 1);
        assert!((b[3 * w + 4] - 1.0 / 9.0).abs() < 1e-7);
        assert!((b.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let flat = box_blur(&vec![0.3; w * h], w, h, 3);
        assert!(flat.iter().all(|v| (v - 0.3).abs() < 1e-6));
        assert_eq!(box_blur(&src, w, h, 0), src);
    }

    fn constant_map(rect: Rect, mw: u32, mh: u32, gain: f32, alpha: f32) -> BlemishMap {
        BlemishMap {
            rect,
            width_px: mw,
            height_px: mh,
            data: vec![[gain, gain, gain, alpha]; (mw * mh) as usize],
        }
    }

    #[test]
    fn apply_maps_through_render_scale() {
        // Working image half the size of the full image; the face rect covers full (20..60, 20..60),
        // which is working (10..30, 10..30).
        let map = constant_map(Rect::new(20, 20, 40, 40), 40, 40, 2.0, 1.0);
        let mut img = LinearAwg4Image::filled(50, 50, [0.1, 0.2, 0.3]).unwrap();
        apply_blemish(&mut img, &map, 1.0, 0.5, None).unwrap();
        for y in 0..50usize {
            for x in 0..50usize {
                let p = img.data[y * 50 + x];
                if (10..30).contains(&x) && (10..30).contains(&y) {
                    assert_eq!(p, [0.2, 0.4, 0.6], "inside at {x},{y}");
                } else {
                    assert_eq!(p, [0.1, 0.2, 0.3], "outside at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn apply_samples_bilinearly_with_the_last_index_clamped() {
        // Two map columns at 1:1: pixel 0 reads column 0, pixel 1 reads column 1.
        let map = BlemishMap {
            rect: Rect::new(0, 0, 2, 1),
            width_px: 2,
            height_px: 1,
            data: vec![[2.0, 2.0, 2.0, 1.0], [0.5, 0.5, 0.5, 1.0]],
        };
        let mut img = LinearAwg4Image::filled(2, 1, [1.0; 3]).unwrap();
        apply_blemish(&mut img, &map, 1.0, 1.0, None).unwrap();
        assert_eq!(img.data, vec![[2.0; 3], [0.5; 3]]);
        // A map twice as wide as the rect: pixel centres fall between map entries.
        let map = BlemishMap {
            rect: Rect::new(0, 0, 1, 1),
            width_px: 2,
            height_px: 1,
            data: vec![[2.0, 2.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        };
        let mut img = LinearAwg4Image::filled(1, 1, [1.0; 3]).unwrap();
        apply_blemish(&mut img, &map, 1.0, 1.0, None).unwrap();
        // u = 2 * (0 + 0.5) - 0.5 = 0.5: halfway between 2 and 1.
        assert_eq!(img.data[0], [1.5; 3]);
    }

    #[test]
    fn apply_threshold_protection_and_errors() {
        let map = constant_map(Rect::new(0, 0, 4, 4), 4, 4, 2.0, 0.25);
        // a = 0.5: s = 0.25 / 0.5 = 0.5; gain = lerp(1, 2, 0.5) = 1.5.
        let mut img = LinearAwg4Image::filled(4, 4, [1.0; 3]).unwrap();
        apply_blemish(&mut img, &map, 0.5, 1.0, None).unwrap();
        assert!(img.data.iter().all(|p| *p == [1.5; 3]));
        // A protection mask of 255 on the left half.
        let mut protect = Mask::zeros(4, 4).unwrap();
        for y in 0..4 {
            for x in 0..2 {
                protect.data[y * 4 + x] = 255;
            }
        }
        let mut img = LinearAwg4Image::filled(4, 4, [1.0; 3]).unwrap();
        apply_blemish(&mut img, &map, 0.5, 1.0, Some(&protect)).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(img.data[y * 4 + x], if x < 2 { [1.0; 3] } else { [1.5; 3] });
            }
        }
        // a = 0 changes nothing; bad inputs are errors.
        let mut img = LinearAwg4Image::filled(4, 4, [1.0; 3]).unwrap();
        apply_blemish(&mut img, &map, 0.0, 1.0, None).unwrap();
        assert!(img.data.iter().all(|p| *p == [1.0; 3]));
        assert!(apply_blemish(&mut img, &map, 0.5, 0.0, None).is_err());
        let wrong = Mask::zeros(3, 4).unwrap();
        assert!(apply_blemish(&mut img, &map, 0.5, 1.0, Some(&wrong)).is_err());
        let bad_map = BlemishMap {
            data: vec![],
            ..map.clone()
        };
        assert!(apply_blemish(&mut img, &bad_map, 0.5, 1.0, None).is_err());
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let img = textured(180, 140);
        let mut mask = full_mask(180, 140);
        for (i, m) in mask.data.iter_mut().enumerate() {
            *m = ((i * 31) % 256) as u8;
        }
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                // Tint and scale mixed so the guard and the gain both act; k < 1 so the area
                // downscale runs too (face_px 900 -> 0.5).
                let map = map_for(&img, &mask, 900.0, &MockProposer::tint(48, 0, 1.15));
                let mut out = img.clone();
                apply_blemish(&mut out, &map, 0.8, 1.0, None).unwrap();
                (map, out)
            })
        };
        let (m1, o1) = run(1);
        let (m4, o4) = run(4);
        assert_eq!(m1.width_px, 90);
        let bits =
            |m: &BlemishMap| -> Vec<u32> { m.data.iter().flatten().map(|v| v.to_bits()).collect() };
        assert_eq!(bits(&m1), bits(&m4));
        let pix = |i: &LinearAwg4Image| -> Vec<u32> {
            i.data.iter().flatten().map(|v| v.to_bits()).collect()
        };
        assert_eq!(pix(&o1), pix(&o4));
    }

    #[test]
    fn the_cache_reuses_and_invalidates() {
        let key = BlemishMapKey {
            image_key: 7,
            face_count: 2,
            face_rects: vec![Rect::new(0, 0, 4, 4), Rect::new(10, 10, 4, 4)],
            upstream_key: 1,
        };
        let mut cache = BlemishMapCache::new();
        assert!(cache.is_empty());
        let mut builds = 0;
        let mut get = |cache: &mut BlemishMapCache, key: &BlemishMapKey, i: usize| {
            cache
                .get_or_build(key, i, || {
                    builds += 1;
                    Ok(constant_map(Rect::new(0, 0, 4, 4), 4, 4, 1.0, 0.0))
                })
                .unwrap()
        };
        let a = get(&mut cache, &key, 0);
        let b = get(&mut cache, &key, 0);
        assert!(Arc::ptr_eq(&a, &b));
        get(&mut cache, &key, 1);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&key, 1).is_some());
        // Each part of the key invalidates.
        for changed in [
            BlemishMapKey {
                image_key: 8,
                ..key.clone()
            },
            BlemishMapKey {
                face_count: 1,
                ..key.clone()
            },
            BlemishMapKey {
                face_rects: vec![Rect::new(1, 0, 4, 4), Rect::new(10, 10, 4, 4)],
                ..key.clone()
            },
            BlemishMapKey {
                upstream_key: 2,
                ..key.clone()
            },
        ] {
            assert!(cache.get(&changed, 0).is_none());
            let mut c = BlemishMapCache::new();
            get(&mut c, &key, 0);
            get(&mut c, &changed, 0);
            assert_eq!(c.len(), 1);
            assert!(c.get(&key, 0).is_none());
        }
        // A failing build caches nothing.
        let mut c = BlemishMapCache::new();
        assert!(
            c.get_or_build(&key, 0, || Err(RetouchError::Model("x".into())))
                .is_err()
        );
        assert!(c.is_empty());
        cache.clear();
        assert!(cache.is_empty() && cache.get(&key, 0).is_none());
        assert_eq!(builds, 1 + 1 + 4 * 2);
    }

    proptest! {
        #[test]
        fn kernels_are_finite(
            out in prop::array::uniform3(-2.0f64..4.0),
            inp in prop::array::uniform3(-2.0f64..4.0),
            blurred in prop::array::uniform3(0.0f64..4.0),
            m in any::<u8>(),
            d in 0u32..10,
            alpha in 0.0f64..5.0,
            a in -0.5f64..1.5,
            protect in any::<u8>(),
            rgb in prop::array::uniform3(-1.0f32..8.0),
        ) {
            let conf = chroma_confidence(out, inp);
            prop_assert!(conf.is_finite() && (1e-5..=1.0).contains(&conf));
            let pulled = chroma_pull(out, blurred, conf);
            prop_assert!(pulled.iter().all(|v| v.is_finite()));
            let g: Vec<f64> = (0..3).map(|c| blemish_gain(out[c], inp[c], m, d)).collect();
            prop_assert!(g.iter().all(|v| (0.5..=2.5).contains(v) || *v == 1.0));
            let g = [g[0], g[1], g[2]];
            prop_assert!(gain_strength(g).is_finite() && gain_strength(g) >= 1e-6);
            let s = threshold_strength(alpha, a, protect);
            prop_assert!((0.0..=1.0).contains(&s));
            prop_assert!(apply_gain(rgb, g, s).iter().all(|v| v.is_finite()));
        }

        #[test]
        fn strength_is_monotonic_in_the_amount(
            alpha in 0.0f64..3.0,
            a1 in 0.0f64..1.0,
            a2 in 0.0f64..1.0,
            protect in any::<u8>(),
        ) {
            let (lo, hi) = if a1 <= a2 { (a1, a2) } else { (a2, a1) };
            let (s_lo, s_hi) = (
                threshold_strength(alpha, lo, protect),
                threshold_strength(alpha, hi, protect),
            );
            prop_assert!(s_lo <= s_hi + 1e-12, "s({lo}) = {s_lo} > s({hi}) = {s_hi}");
        }
    }
}
