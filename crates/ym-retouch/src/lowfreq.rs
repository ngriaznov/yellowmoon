//! Dark circles, even skin, skin grading and teeth whitening (`docs/27` section 7).
//!
//! One low-frequency operator, per face, on the working image: `T` is a blur of the image at the
//! texture radius, `src - T` is the texture that is kept, and a mask-normalised blur of `T` gives
//! the regional average that dark circles (colour) and even skin (luminance) move the skin toward.
//! The asymmetric blend of 7.5 lets the skin brighten and almost never darken. Skin grading (7.7:
//! contrast, shadow desaturation, shine removal) reads the same `T` and a wider regional average of
//! it; teeth whitening (7.8) works per pixel in its own mask.
//!
//! The pure per-pixel kernels are free functions ([`blend_weight`], [`blend_pixel`],
//! [`masked_blur_combine`], [`dark_circles_target`], [`even_skin_target`], [`contrast_target`],
//! [`shadow_desaturation_pixel`], [`shine_removal_texture`], [`teeth_pixel`], [`LowFreqParams`]) so a
//! GPU kernel can follow with the same maths. The image passes run over rows with `rayon` and
//! write disjoint rows, so results never depend on the thread count.

use rayon::prelude::*;

use crate::error::{Result, RetouchError};
use crate::mask::{percentile, smoothstep};
use crate::types::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, check_render_scale, face_size_px,
};

/// Texture radius as a fraction of the face at Texture -100: the radius is
/// `min(TEX_FRAC_MIN + TEX_FRAC_SLOPE * t, TEX_FRAC_MAX) * D` with `t` in 0..=1 (`docs/27` 7.1; S-12
/// tuned it, the brief's value was 0.004).
pub const TEX_FRAC_MIN: f64 = 0.005;
/// Growth of the texture radius over the Texture range, as a fraction of the face (brief: 0.02).
pub const TEX_FRAC_SLOPE: f64 = 0.025;
/// Largest texture radius as a fraction of the face (brief: 0.024).
pub const TEX_FRAC_MAX: f64 = 0.030;
/// Radius of the dark-circle average as a fraction of `D` (`r_dc`; brief: 0.077).
pub const DC_RADIUS_FRAC: f64 = 0.077;
/// Radius of the even-skin average as a fraction of `D` (`r_se`; brief: 0.044).
pub const SE_RADIUS_FRAC: f64 = 0.055;
/// Radius of the skin-grading average as a fraction of `D` (`r_g`; brief: 0.15).
pub const GRADE_RADIUS_FRAC: f64 = 0.15;
/// Exponent of the weight when a pixel would get darker, `(Sum(new) / Sum(src))^n` (brief: 16).
pub const DARKEN_EXPONENT: i32 = 16;
/// Exponent of the roll-off when a pixel would get brighter, `(Sum(src) / Sum(new))^n` (brief: 2).
pub const BRIGHTEN_EXPONENT: i32 = 3;

/// The derived parameters of `docs/27` section 7.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowFreqParams {
    /// `min(TEX_FRAC_MIN + TEX_FRAC_SLOPE * t, TEX_FRAC_MAX)`: the texture radius as a fraction of the
    /// face size.
    pub tex_frac: f64,
    /// `D * tex_frac`, in working pixels.
    pub r_tex_full: f64,
    /// True when the layers are computed at half resolution (`r_tex_full >= 8`).
    pub half: bool,
    /// The downscale factor, 2 when `half`, else 1.
    pub ds: usize,
    /// Texture radius at the layer resolution, at least 1.
    pub r_tex: usize,
    /// Dark-circle radius at the layer resolution, at least 1.
    pub r_dc: usize,
    /// Even-skin radius at the layer resolution, at least 1.
    pub r_se: usize,
    /// Dark-circle blend strength `1 - (1 - dc)^3`.
    pub k_dc: f64,
    /// Dark-circle brightening roll-off `1 - 0.75 * dc^4`.
    pub sp_dc: f64,
    /// Even-skin blend strength `se`.
    pub k_se: f64,
    /// Even-skin brightening roll-off, always 1.
    pub sp_se: f64,
    /// Skin-grading regional radius `max(1, trunc(0.15 * D / ds))` (7.7).
    pub r_g: usize,
}

/// `max(1, trunc(x / ds))` as a radius.
fn radius(x: f64, ds: usize) -> usize {
    let r = (x / ds as f64).trunc();
    if r.is_finite() && r >= 1.0 {
        r.min(1.0e6) as usize
    } else {
        1
    }
}

/// `(k, sp)` of the dark-circle tool for `dc` in 0..=1 (clamped).
pub fn dark_circle_k_sp(dc: f64) -> (f64, f64) {
    let dc = dc.clamp(0.0, 1.0);
    (1.0 - (1.0 - dc).powi(3), 1.0 - 0.75 * dc.powi(4))
}

impl LowFreqParams {
    /// Derives the parameters from the face size `d` in working pixels (`D` of section 3),
    /// `t`, `dc` and `se` of [`SkinSettings`].
    pub fn new(d: f64, t: f64, dc: f64, se: f64) -> Self {
        Self::with_half(d, t, dc, se, None)
    }

    /// As [`LowFreqParams::new`], optionally forcing the half-resolution path on or off (tests).
    pub(crate) fn with_half(d: f64, t: f64, dc: f64, se: f64, half: Option<bool>) -> Self {
        let t = t.clamp(0.0, 1.0);
        let tex_frac = (TEX_FRAC_MIN + TEX_FRAC_SLOPE * t).min(TEX_FRAC_MAX);
        let r_tex_full = d * tex_frac;
        let half = half.unwrap_or(r_tex_full >= 8.0);
        let ds = if half { 2 } else { 1 };
        let (k_dc, sp_dc) = dark_circle_k_sp(dc);
        Self {
            tex_frac,
            r_tex_full,
            half,
            ds,
            r_tex: radius(r_tex_full, ds),
            r_dc: radius(DC_RADIUS_FRAC * d, ds),
            r_se: radius(SE_RADIUS_FRAC * d, ds),
            k_dc,
            sp_dc,
            k_se: se.clamp(0.0, 1.0),
            sp_se: 1.0,
            r_g: radius(GRADE_RADIUS_FRAC * d, ds),
        }
    }
}

/// True for a number above 0 (false for NaN).
fn positive(x: f64) -> bool {
    x > 0.0
}

fn sum3(p: [f64; 3]) -> f64 {
    (p[0] + p[1] + p[2]).max(0.001)
}

/// The weight `w` of 7.5: `(Sum(new) / Sum(src))^16` when the pixel gets darker, else
/// `(1 - sp) + sp * (Sum(src) / Sum(new))^2` (the exponents are [`DARKEN_EXPONENT`] and
/// [`BRIGHTEN_EXPONENT`]), with `Sum(p) = max(r + g + b, 0.001)`.
pub fn blend_weight(src: [f64; 3], new: [f64; 3], sp: f64) -> f64 {
    let (s, n) = (sum3(src), sum3(new));
    if n < s {
        (n / s).powi(DARKEN_EXPONENT)
    } else {
        let q = s / n;
        (1.0 - sp) + sp * q.powi(BRIGHTEN_EXPONENT)
    }
}

/// The blend of 7.5: `lerp(src, new, k * w)`.
pub fn blend_pixel(src: [f64; 3], new: [f64; 3], k: f64, sp: f64) -> [f64; 3] {
    let f = k * blend_weight(src, new, sp);
    [
        src[0] + f * (new[0] - src[0]),
        src[1] + f * (new[1] - src[1]),
        src[2] + f * (new[2] - src[2]),
    ]
}

/// The mask-normalised blur of 7.2 step 3 for one pixel: `m * num / max(den, 1/255) + (1 - m) * x`,
/// with `num = blur(X * m)` and `den = blur(m)` at that pixel and `m = mask / 255`.
pub fn masked_blur_combine(m: f64, num: f64, den: f64, x: f64) -> f64 {
    m * num / den.max(1.0 / 255.0) + (1.0 - m) * x
}

/// 7.3: `new = src - T + C_dc`.
pub fn dark_circles_target(src: [f64; 3], t: [f64; 3], c_dc: [f64; 3]) -> [f64; 3] {
    [
        src[0] - t[0] + c_dc[0],
        src[1] - t[1] + c_dc[1],
        src[2] - t[2] + c_dc[2],
    ]
}

/// 7.4: `new = src * (C_se / max(Y, 0.001))`, the same gain on all channels.
pub fn even_skin_target(src: [f64; 3], y: f64, c_se: f64) -> [f64; 3] {
    let g = c_se / y.max(0.001);
    [src[0] * g, src[1] * g, src[2] * g]
}

/// 7.7 contrast: `Y' = C_g * (Y / C_g)^(1 + 0.5 * gc)`, `new = src * (Y' / max(Y, 0.001))`, then
/// `lerp(src, new, m)`. `y` is the luminance of `T`, `c_g` the regional average, `gc` in -1..=1, `m`
/// the mask in 0..=1. A pixel or level that is not positive is left alone.
pub fn contrast_target(src: [f64; 3], y: f64, c_g: f64, gc: f64, m: f64) -> [f64; 3] {
    if !(positive(y) && positive(c_g)) {
        return src;
    }
    let y_new = c_g * (y / c_g).powf(1.0 + 0.5 * gc);
    let gain = y_new / y.max(0.001);
    [0, 1, 2].map(|c| src[c] + m * (src[c] * gain - src[c]))
}

/// 7.7 shadow desaturation: `w = clamp((C_g - Y) / (0.5 * C_g), 0, 1)`, `L = mean(src)`,
/// `out = L + (1 - 0.6 * gd * w * m) * (src - L)`. The mean of the channels is kept.
pub fn shadow_desaturation_pixel(src: [f64; 3], y: f64, c_g: f64, gd: f64, m: f64) -> [f64; 3] {
    let w = if c_g > 0.0 {
        ((c_g - y) / (0.5 * c_g)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let l = (src[0] + src[1] + src[2]) / 3.0;
    let f = 1.0 - 0.6 * gd * w * m;
    [0, 1, 2].map(|c| l + f * (src[c] - l))
}

/// 7.7 shine removal for one pixel of `T`: the weight `w` of the sheen, and the new low-frequency
/// colour. With `sat = 1 - min(T) / max(T)`, `e = ln(Y / C_g) - 0.1`,
/// `w = smoothstep(0, 0.4, e) * (1 - smoothstep(0.15, 0.35, sat)) * m` and
/// `T' = lerp(T, C * (C_g * e^0.1) / max(mean(C), 0.001), gs * w)`, the caller adds the texture
/// back: `out = src - T + T'`. Returns `T` itself when `w` is 0.
pub fn shine_removal_texture(
    t: [f64; 3],
    y: f64,
    c_g: f64,
    c: [f64; 3],
    gs: f64,
    m: f64,
) -> [f64; 3] {
    if !(positive(y) && positive(c_g)) || gs <= 0.0 {
        return t;
    }
    let e = (y / c_g).ln() - 0.1;
    let tmax = t[0].max(t[1]).max(t[2]);
    let sat = if tmax > 0.0 {
        1.0 - t[0].min(t[1]).min(t[2]) / tmax
    } else {
        1.0
    };
    let w = smoothstep(0.0, 0.4, e) * (1.0 - smoothstep(0.15, 0.35, sat)) * m;
    if !positive(w) {
        return t;
    }
    let level = c_g * e.powf(0.1) / ((c[0] + c[1] + c[2]) / 3.0).max(0.001);
    let f = (gs * w).min(1.0);
    [0, 1, 2].map(|k| t[k] + f * (c[k] * level - t[k]))
}

/// 7.8 teeth whitening for one pixel: chroma toward neutral and a little more light, never past the
/// brightest tooth. `c = L + (1 - 0.7 * tw * m) * (src - L)`, `cap = l_max / max(L, 1e-4)`,
/// `out = c * min(1 + 0.15 * tw * m, max(cap, 1))`; `L = mean(src)`, `l_max` the mask's 99th
/// percentile of `L`.
pub fn teeth_pixel(src: [f64; 3], tw: f64, m: f64, l_max: f64) -> [f64; 3] {
    let l = (src[0] + src[1] + src[2]) / 3.0;
    let f = 1.0 - 0.7 * tw * m;
    let cap = l_max / l.max(1e-4);
    let gain = (1.0 + 0.15 * tw * m).min(cap.max(1.0));
    [0, 1, 2].map(|c| (l + f * (src[c] - l)) * gain)
}

/// Resamples a full-resolution mask to the working image (`docs/27` decision 1): bilinear,
/// pixel-centre aligned, `working = full * render_scale`. The result is a bounded mask in working
/// pixels, clipped to the image and cropped to its non-zero pixels, or `None` when nothing is left.
pub fn mask_to_working(
    mask: &BoundedMask,
    render_scale: f32,
    working_size_px: (u32, u32),
) -> Result<Option<BoundedMask>> {
    let s = check_render_scale(render_scale)?;
    mask.validate()?;
    let (ww, wh) = (i64::from(working_size_px.0), i64::from(working_size_px.1));
    let (rx, ry) = (f64::from(mask.region.x_px), f64::from(mask.region.y_px));
    let (rw, rh) = (
        f64::from(mask.region.width_px),
        f64::from(mask.region.height_px),
    );
    let lo = |r: f64| ((r - 0.5) * s - 0.5).floor() as i64;
    let hi = |r: f64, n: f64| ((r + n + 0.5) * s - 0.5).ceil() as i64 + 1;
    let x0 = lo(rx).clamp(0, ww);
    let x1 = hi(rx, rw).clamp(0, ww);
    let y0 = lo(ry).clamp(0, wh);
    let y1 = hi(ry, rh).clamp(0, wh);
    if x0 >= x1 || y0 >= y1 {
        return Ok(None);
    }
    let (ow, oh) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let mut data = vec![0u8; ow * oh];
    data.par_chunks_mut(ow).enumerate().for_each(|(j, row)| {
        let sy = ((y0 as f64 + j as f64) + 0.5) / s - 0.5 - ry;
        let iy = sy.floor();
        let fy = sy - iy;
        let iy = iy as i64;
        for (i, out) in row.iter_mut().enumerate() {
            let sx = ((x0 as f64 + i as f64) + 0.5) / s - 0.5 - rx;
            let ix = sx.floor();
            let fx = sx - ix;
            let ix = ix as i64;
            let g = |dx: i64, dy: i64| f64::from(mask.mask.get(ix + dx, iy + dy));
            let top = g(0, 0) * (1.0 - fx) + g(1, 0) * fx;
            let bot = g(0, 1) * (1.0 - fx) + g(1, 1) * fx;
            *out = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
        }
    });
    let sub = BoundedMask::new(
        Rect::new(x0 as i32, y0 as i32, ow as u32, oh as u32),
        Mask::new(ow as u32, oh as u32, data)?,
    )?;
    Ok(sub.crop_to_nonzero())
}

/// One box filter of radius `q` along rows, edges replicated.
fn box_rows(src: &[f64], w: usize, q: usize) -> Vec<f64> {
    if q == 0 || w == 0 {
        return src.to_vec();
    }
    let mut out = vec![0.0; src.len()];
    let inv = 1.0 / (2 * q + 1) as f64;
    let last = w as isize - 1;
    let at = |row: &[f64], i: isize| row[i.clamp(0, last) as usize];
    out.par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(o, row)| {
            let qi = q as isize;
            let mut acc = 0.0;
            for i in -qi..=qi {
                acc += at(row, i);
            }
            for x in 0..w as isize {
                o[x as usize] = acc * inv;
                acc += at(row, x + qi + 1) - at(row, x - qi);
            }
        });
    out
}

fn transpose(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; src.len()];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for (y, v) in col.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
}

/// `blur(X, r)` of 7.1: two box filters in a row with radii `r / 2` and `r - r / 2`, edges
/// replicated, on one `w x h` plane.
pub(crate) fn blur_plane(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let (q1, q2) = (r / 2, r - r / 2);
    let a = box_rows(&box_rows(src, w, q1), w, q2);
    let t = transpose(&a, w, h);
    let b = box_rows(&box_rows(&t, h, q1), h, q2);
    transpose(&b, h, w)
}

/// Area downscale by 2; an odd last row or column averages what exists.
pub(crate) fn downscale2(src: &[f64], w: usize, h: usize) -> (Vec<f64>, usize, usize) {
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0; w2 * h2];
    out.par_chunks_mut(w2).enumerate().for_each(|(y2, row)| {
        for (x2, o) in row.iter_mut().enumerate() {
            let (mut acc, mut n) = (0.0, 0.0);
            for y in (2 * y2)..(2 * y2 + 2).min(h) {
                for x in (2 * x2)..(2 * x2 + 2).min(w) {
                    acc += src[y * w + x];
                    n += 1.0;
                }
            }
            *o = acc / n;
        }
    });
    (out, w2, h2)
}

/// The mask-normalised blur of 7.2 step 3 on one plane, given `den = blur(m, r)`.
fn masked_blur_plane(x: &[f64], m: &[f64], den: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let xm: Vec<f64> = x.iter().zip(m).map(|(a, b)| a * b).collect();
    let num = blur_plane(&xm, w, h, r);
    (0..x.len())
        .map(|i| masked_blur_combine(m[i], num[i], den[i], x[i]))
        .collect()
}

/// A tool's mask over the padded region: bytes at full layer resolution and `m` at layer resolution.
struct ToolMask {
    bytes: Vec<u8>,
    m: Vec<f64>,
}

fn paint(bm: &BoundedMask, region: Rect) -> Vec<u8> {
    let (rw, rh) = (region.width_px as usize, region.height_px as usize);
    let mut out = vec![0u8; rw * rh];
    let mw = bm.mask.width_px as usize;
    for y in 0..rh {
        let sy = i64::from(region.y_px) + y as i64 - i64::from(bm.region.y_px);
        if sy < 0 || sy >= i64::from(bm.mask.height_px) {
            continue;
        }
        for x in 0..rw {
            let sx = i64::from(region.x_px) + x as i64 - i64::from(bm.region.x_px);
            if sx >= 0 && sx < mw as i64 {
                out[y * rw + x] = bm.mask.data[sy as usize * mw + sx as usize];
            }
        }
    }
    out
}

/// The masks the tools of one face work in, all cropped to their bounding boxes, in full-resolution
/// pixels (`docs/27` section 5). A tool whose mask is `None` does nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolMasks<'a> {
    /// Dark circles (5.4).
    pub dark_circles: Option<&'a BoundedMask>,
    /// Even skin (5.5).
    pub even_skin: Option<&'a BoundedMask>,
    /// Skin grading (5.6).
    pub grading: Option<&'a BoundedMask>,
    /// Teeth whitening (5.6).
    pub teeth: Option<&'a BoundedMask>,
}

/// Dark circles then even skin on the working image, in place (`docs/27` section 7).
///
/// `full_size_px` is the full-resolution size the masks were built at; the masks are resampled by
/// `render_scale` to the working image ([`mask_to_working`]). A tool is active when its amount is
/// above 0 and its mask exists and is not all zeros. Returns `Ok(false)` and leaves the image
/// untouched when neither tool is active. Pixels where a tool's mask is 0 are not touched by it.
/// See [`retouch_tones`] for the whole of section 7.
pub fn retouch_low_frequency(
    image: &mut LinearAwg4Image,
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    dark_circle_mask: Option<&BoundedMask>,
    even_skin_mask: Option<&BoundedMask>,
) -> Result<bool> {
    retouch_with(
        image,
        full_size_px,
        face,
        settings,
        render_scale,
        dark_circle_mask,
        even_skin_mask,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn retouch_with(
    image: &mut LinearAwg4Image,
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    dark_circle_mask: Option<&BoundedMask>,
    even_skin_mask: Option<&BoundedMask>,
    half_override: Option<bool>,
) -> Result<bool> {
    let masks = ToolMasks {
        dark_circles: dark_circle_mask,
        even_skin: even_skin_mask,
        ..ToolMasks::default()
    };
    tones_with(
        image,
        full_size_px,
        face,
        settings,
        render_scale,
        &masks,
        None,
        None,
        half_override,
    )
}

/// All of `docs/27` section 7 for one face on the working image, in place, in the order of 7.9:
/// dark circles, even skin, skin grading (contrast, shadow desaturation, shine removal), teeth
/// whitening. The first five share one low-frequency layer `T` taken from the image as it is on
/// entry; each reads the previous result as `src`. A tool is active when its amount is not 0
/// (contrast) or above 0 and its mask exists and is not all zeros; with no active tool the image is
/// untouched and the result is `Ok(false)`. A tool never changes a pixel whose own mask is 0.
///
/// `protect` is the protection mask in working pixels (the image's size; the owner's Protect row,
/// 2026-11-11): every tool's weight (`k * w` of 7.5, `m` of 7.7 and 7.8) is multiplied by
/// `1 - protect / 255` at each pixel, and a pixel with `protect == 255` is not touched at all.
#[allow(clippy::too_many_arguments)]
pub fn retouch_tones(
    image: &mut LinearAwg4Image,
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    masks: &ToolMasks<'_>,
    protect: Option<&Mask>,
) -> Result<bool> {
    tones_with(
        image,
        full_size_px,
        face,
        settings,
        render_scale,
        masks,
        protect,
        None,
        None,
    )
}

/// Where an image that is a part of the working image sits: the working pixel of its top-left corner and the size of the
/// whole working image. Pixel `(x, y)` of the part is working pixel `(x + origin_px.0, y + origin_px.1)`.
///
/// A part gives the pixels of the whole image bit for bit when it holds everything a face reads and writes
/// ([`tones_region`]) and its origin is even (the half-resolution path works on 2 x 2 blocks of the whole image's grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// The working pixel of the part's top-left corner.
    pub origin_px: (i32, i32),
    /// The size of the whole working image, in pixels.
    pub working_px: (u32, u32),
}

/// [`retouch_tones`] on a part of the working image (see [`Placement`]); `protect` is the same part of the protection mask.
///
/// # Errors
///
/// As [`retouch_tones`].
#[allow(clippy::too_many_arguments)]
pub fn retouch_tones_at(
    image: &mut LinearAwg4Image,
    place: Placement,
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    masks: &ToolMasks<'_>,
    protect: Option<&Mask>,
) -> Result<bool> {
    tones_with(
        image,
        full_size_px,
        face,
        settings,
        render_scale,
        masks,
        protect,
        Some(place),
        None,
    )
}

/// A full-resolution mask as the working-resolution mask of a part: resampled by `render_scale` to the whole working image,
/// then moved to the part's coordinates and clipped to the part.
fn mask_in_part(
    mask: &BoundedMask,
    render_scale: f32,
    place: Placement,
    part: (u32, u32),
) -> Result<Option<BoundedMask>> {
    let Some(w) = mask_to_working(mask, render_scale, place.working_px)? else {
        return Ok(None);
    };
    if place.origin_px == (0, 0) && part == place.working_px {
        return Ok(Some(w));
    }
    let (ox, oy) = (i64::from(place.origin_px.0), i64::from(place.origin_px.1));
    let x0 = (i64::from(w.region.x_px) - ox).max(0);
    let y0 = (i64::from(w.region.y_px) - oy).max(0);
    let x1 = (w.region.right_px() - ox).min(i64::from(part.0));
    let y1 = (w.region.bottom_px() - oy).min(i64::from(part.1));
    if x0 >= x1 || y0 >= y1 {
        return Ok(None);
    }
    let (nw, nh) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let mw = w.mask.width_px as usize;
    let mut data = Vec::with_capacity(nw * nh);
    for y in y0..y1 {
        let sy = (y + oy - i64::from(w.region.y_px)) as usize;
        let sx = (x0 + ox - i64::from(w.region.x_px)) as usize;
        data.extend_from_slice(&w.mask.data[sy * mw + sx..sy * mw + sx + nw]);
    }
    Ok(Some(BoundedMask::new(
        Rect::new(x0 as i32, y0 as i32, nw as u32, nh as u32),
        Mask::new(nw as u32, nh as u32, data)?,
    )?))
}

/// The padded region the low-frequency pass of one face works in, clipped to a `size` image: the bounding box of the active
/// masks grown by `(r_tex + reach) * ds + ds` pixels, its origin even on the half path. `None` without a mask.
fn padded_region(
    masks: &[(&Option<BoundedMask>, usize)],
    p: &LowFreqParams,
    size: (u32, u32),
) -> Option<Rect> {
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    let mut reach = 0usize;
    for (m, r) in masks {
        if let Some(m) = m {
            x0 = x0.min(i64::from(m.region.x_px));
            y0 = y0.min(i64::from(m.region.y_px));
            x1 = x1.max(m.region.right_px());
            y1 = y1.max(m.region.bottom_px());
            reach = reach.max(*r);
        }
    }
    if x0 == i64::MAX {
        return None;
    }
    let pad = ((p.r_tex + reach) * p.ds + p.ds) as i64;
    let even = |v: i64| if p.half { v & !1 } else { v };
    let x0 = even((x0 - pad).max(0));
    let y0 = even((y0 - pad).max(0));
    let x1 = (x1 + pad).min(i64::from(size.0));
    let y1 = (y1 + pad).min(i64::from(size.1));
    Some(Rect::new(
        x0 as i32,
        y0 as i32,
        (x1 - x0).max(0) as u32,
        (y1 - y0).max(0) as u32,
    ))
}

/// The rectangle of the working image (`working_px` pixels) that the tools of one face read and write: the padded region of
/// the low-frequency pass joined with the teeth mask's. `None` when no tool is active (an amount of 0, or a mask that is
/// missing or empty). A part of the working image that holds this rectangle, with an even origin, is retouched like the whole
/// image ([`retouch_tones_at`]).
///
/// # Errors
///
/// [`RetouchError::RenderScale`] for a scale outside the limits.
pub fn tones_region(
    working_px: (u32, u32),
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    masks: &ToolMasks<'_>,
) -> Result<Option<Rect>> {
    let d = face_size_px(full_size_px, render_scale, face)?;
    let place = Placement {
        origin_px: (0, 0),
        working_px,
    };
    let working = |active: bool, mask: Option<&BoundedMask>| -> Result<Option<BoundedMask>> {
        match mask {
            Some(m) if active && !m.mask.is_all_zero() => {
                mask_in_part(m, render_scale, place, working_px)
            }
            _ => Ok(None),
        }
    };
    let dc_mask = working(settings.dc() > 0.0, masks.dark_circles)?;
    let se_mask = working(settings.se() > 0.0, masks.even_skin)?;
    let g_mask = working(settings.grading_active(), masks.grading)?;
    let teeth_mask = working(settings.tw() > 0.0, masks.teeth)?;
    let p = LowFreqParams::new(d, settings.t(), settings.dc(), settings.se());
    let mut out = padded_region(
        &[(&dc_mask, p.r_dc), (&se_mask, p.r_se), (&g_mask, p.r_g)],
        &p,
        working_px,
    );
    if let Some(t) = &teeth_mask {
        out = Some(join_rects(out, t.region));
    }
    Ok(out)
}

/// The smallest rectangle that holds `a` (when there is one) and `b`.
pub(crate) fn join_rects(a: Option<Rect>, b: Rect) -> Rect {
    let Some(a) = a else { return b };
    let x0 = i64::from(a.x_px).min(i64::from(b.x_px));
    let y0 = i64::from(a.y_px).min(i64::from(b.y_px));
    let x1 = a.right_px().max(b.right_px());
    let y1 = a.bottom_px().max(b.bottom_px());
    Rect::new(x0 as i32, y0 as i32, (x1 - x0) as u32, (y1 - y0) as u32)
}

/// What the device needs for one face's low-frequency tools on a part (`docs/27` S-13): the padded region of
/// [`LowFreqParams`], the amounts, and the tools' masks at the working resolution over that region, all as the CPU pass
/// reads them. The device runs the maths of [`tones_with`] on these; nothing here is a different rule.
#[derive(Debug, Clone)]
pub struct LowPlan {
    /// The padded region of the pass, in pixels of the part (the origin is even on the half path).
    pub rect: Rect,
    /// The derived radii and strengths (7.1).
    pub params: LowFreqParams,
    /// Skin contrast `gc`, -1..=1 (0: off).
    pub gc: f64,
    /// Shadow desaturation `gd`, 0..=1.
    pub gd: f64,
    /// Shine removal `gs`, 0..=1.
    pub gs: f64,
    /// The dark-circle mask over `rect` (one byte a pixel, rows of `rect.width_px`), when the tool is active.
    pub dc: Option<Vec<u8>>,
    /// The even-skin mask over `rect`, when active.
    pub se: Option<Vec<u8>>,
    /// The skin-grading mask over `rect`, when any grading amount is active.
    pub grade: Option<Vec<u8>>,
}

/// What the device needs for teeth whitening (7.8) on a part.
#[derive(Debug, Clone)]
pub struct TeethPlan {
    /// The teeth mask's rectangle in the part (cropped to its non-zero pixels).
    pub rect: Rect,
    /// The mask over `rect`.
    pub bytes: Vec<u8>,
    /// Teeth whitening `tw`, 0..=1.
    pub tw: f64,
}

/// The tools of one face on a part, as the device runs them.
#[derive(Debug, Clone)]
pub struct TonesPlan {
    /// Dark circles, even skin and grading.
    pub low: Option<LowPlan>,
    /// Teeth whitening.
    pub teeth: Option<TeethPlan>,
}

/// [`tones_with`] without the pixels: the masks resampled to the working image and cut to the part, the padded region and
/// the parameters, for the device. `None` when no tool of the face is active on this part.
///
/// # Errors
///
/// [`RetouchError::RenderScale`] for a scale outside the limits.
pub(crate) fn tones_plan(
    size: (u32, u32),
    place: Placement,
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    masks: &ToolMasks<'_>,
) -> Result<Option<TonesPlan>> {
    let d = face_size_px(full_size_px, render_scale, face)?;
    let working = |active: bool, mask: Option<&BoundedMask>| -> Result<Option<BoundedMask>> {
        match mask {
            Some(m) if active && !m.mask.is_all_zero() => {
                mask_in_part(m, render_scale, place, size)
            }
            _ => Ok(None),
        }
    };
    let dc_mask = working(settings.dc() > 0.0, masks.dark_circles)?;
    let se_mask = working(settings.se() > 0.0, masks.even_skin)?;
    let g_mask = working(settings.grading_active(), masks.grading)?;
    let teeth_mask = working(settings.tw() > 0.0, masks.teeth)?;
    let p = LowFreqParams::new(d, settings.t(), settings.dc(), settings.se());
    let low = padded_region(
        &[(&dc_mask, p.r_dc), (&se_mask, p.r_se), (&g_mask, p.r_g)],
        &p,
        size,
    )
    .map(|rect| LowPlan {
        rect,
        params: p,
        gc: settings.gc(),
        gd: settings.gd(),
        gs: settings.gs(),
        dc: dc_mask.as_ref().map(|bm| paint(bm, rect)),
        se: se_mask.as_ref().map(|bm| paint(bm, rect)),
        grade: g_mask.as_ref().map(|bm| paint(bm, rect)),
    });
    let teeth = teeth_mask.map(|bm| TeethPlan {
        rect: bm.region,
        bytes: bm.mask.data,
        tw: settings.tw(),
    });
    if low.is_none() && teeth.is_none() {
        return Ok(None);
    }
    Ok(Some(TonesPlan { low, teeth }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tones_with(
    image: &mut LinearAwg4Image,
    full_size_px: (u32, u32),
    face: &Face,
    settings: &SkinSettings,
    render_scale: f32,
    masks: &ToolMasks<'_>,
    protect: Option<&Mask>,
    place: Option<Placement>,
    half_override: Option<bool>,
) -> Result<bool> {
    image.validate()?;
    if let Some(p) = protect {
        p.validate()?;
        if p.size_px() != (image.width_px, image.height_px) {
            return Err(RetouchError::MaskSizeMismatch {
                left: (image.width_px, image.height_px),
                right: p.size_px(),
            });
        }
    }
    let d = face_size_px(full_size_px, render_scale, face)?;
    let size = (image.width_px, image.height_px);
    let place = place.unwrap_or(Placement {
        origin_px: (0, 0),
        working_px: size,
    });
    let working = |active: bool, mask: Option<&BoundedMask>| -> Result<Option<BoundedMask>> {
        match mask {
            Some(m) if active && !m.mask.is_all_zero() => {
                mask_in_part(m, render_scale, place, size)
            }
            _ => Ok(None),
        }
    };
    let dc_mask = working(settings.dc() > 0.0, masks.dark_circles)?;
    let se_mask = working(settings.se() > 0.0, masks.even_skin)?;
    let g_mask = working(settings.grading_active(), masks.grading)?;
    let teeth_mask = working(settings.tw() > 0.0, masks.teeth)?;
    let low = dc_mask.is_some() || se_mask.is_some() || g_mask.is_some();
    if low {
        low_frequency_pass(
            image,
            d,
            settings,
            [&dc_mask, &se_mask, &g_mask],
            protect,
            half_override,
        );
    }
    if let Some(bm) = &teeth_mask {
        whiten_teeth(image, bm, settings.tw(), protect);
    }
    Ok(low || teeth_mask.is_some())
}

/// `1 - protect / 255` at a working pixel (1 without a protection mask).
fn protect_keep(protect: Option<&Mask>, x: usize, y: usize) -> f64 {
    protect.map_or(1.0, |p| {
        1.0 - f64::from(p.data[y * p.width_px as usize + x]) / 255.0
    })
}

/// Teeth whitening (7.8) in the working-resolution teeth mask: `L_max` is the 99th percentile of
/// `L` over the pixels where the mask is not 0, taken from the image as it is.
fn whiten_teeth(image: &mut LinearAwg4Image, bm: &BoundedMask, tw: f64, protect: Option<&Mask>) {
    let iw = image.width_px as usize;
    let (x0, y0) = (bm.region.x_px as usize, bm.region.y_px as usize);
    let (rw, rh) = (bm.region.width_px as usize, bm.region.height_px as usize);
    let mean = |p: &[f32; 3]| (f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2])) / 3.0;
    let mut ls = Vec::new();
    for y in 0..rh {
        for x in 0..rw {
            if bm.mask.data[y * rw + x] != 0 {
                ls.push(mean(&image.data[(y0 + y) * iw + x0 + x]));
            }
        }
    }
    let l_max = percentile(&mut ls, 0.99);
    image.data[y0 * iw..(y0 + rh) * iw]
        .par_chunks_mut(iw)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..rw {
                let byte = bm.mask.data[y * rw + x];
                let keep = protect_keep(protect, x0 + x, y0 + y);
                if byte == 0 || keep == 0.0 {
                    continue;
                }
                let px = &mut row[x0 + x];
                let src = [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
                let out = teeth_pixel(src, tw, keep * f64::from(byte) / 255.0, l_max);
                *px = [out[0] as f32, out[1] as f32, out[2] as f32];
            }
        });
}

fn low_frequency_pass(
    image: &mut LinearAwg4Image,
    d: f64,
    settings: &SkinSettings,
    [dc_mask, se_mask, g_mask]: [&Option<BoundedMask>; 3],
    protect: Option<&Mask>,
    half_override: Option<bool>,
) {
    let (gc, gd, gs) = (settings.gc(), settings.gd(), settings.gs());
    let size = (image.width_px, image.height_px);
    let p = LowFreqParams::with_half(d, settings.t(), settings.dc(), settings.se(), half_override);
    let ds = p.ds;

    // The padded, clipped working region; its origin is even on the half path so that its 2 x 2
    // blocks sit on the same grid as the whole image's.
    let Some(region) = padded_region(
        &[(dc_mask, p.r_dc), (se_mask, p.r_se), (g_mask, p.r_g)],
        &p,
        size,
    ) else {
        return;
    };
    let (x0, y0) = (i64::from(region.x_px), i64::from(region.y_px));
    let y1 = region.bottom_px();
    let (rw, rh) = (region.width_px as usize, region.height_px as usize);
    let iw = image.width_px as usize;

    // Source planes at the layer resolution.
    let mut planes: Vec<Vec<f64>> = (0..3)
        .map(|c| {
            let mut v = Vec::with_capacity(rw * rh);
            for y in 0..rh {
                let row = (y0 as usize + y) * iw + x0 as usize;
                v.extend(image.data[row..row + rw].iter().map(|p| f64::from(p[c])));
            }
            v
        })
        .collect();
    let (mut lw, mut lh) = (rw, rh);
    if p.half {
        for plane in &mut planes {
            let (d2, w2, h2) = downscale2(plane, rw, rh);
            *plane = d2;
            lw = w2;
            lh = h2;
        }
    }
    let t_planes: Vec<Vec<f64>> = planes
        .iter()
        .map(|s| blur_plane(s, lw, lh, p.r_tex))
        .collect();

    let tool_mask = |bm: &BoundedMask| -> ToolMask {
        let bytes = paint(bm, region);
        let full: Vec<f64> = bytes.iter().map(|&b| f64::from(b) / 255.0).collect();
        let m = if p.half {
            downscale2(&full, rw, rh).0
        } else {
            full
        };
        ToolMask { bytes, m }
    };

    // C_dc (RGB) and C_se (one channel) at the layer resolution.
    let dc = dc_mask.as_ref().map(|bm| {
        let tm = tool_mask(bm);
        let den = blur_plane(&tm.m, lw, lh, p.r_dc);
        let c: Vec<Vec<f64>> = t_planes
            .iter()
            .map(|t| masked_blur_plane(t, &tm.m, &den, lw, lh, p.r_dc))
            .collect();
        (tm, c)
    });
    let y_plane: Vec<f64> = (0..lw * lh)
        .map(|i| (t_planes[0][i] + t_planes[1][i] + t_planes[2][i]) / 3.0)
        .collect();
    // C_g (one channel, radius r_g) and, for shine removal, the local skin colour C (RGB, r_se).
    let grade = g_mask.as_ref().map(|bm| {
        let tm = tool_mask(bm);
        let den = blur_plane(&tm.m, lw, lh, p.r_g);
        let c_g = masked_blur_plane(&y_plane, &tm.m, &den, lw, lh, p.r_g);
        let shine: Option<Vec<Vec<f64>>> = (gs > 0.0).then(|| {
            let den = blur_plane(&tm.m, lw, lh, p.r_se);
            t_planes
                .iter()
                .map(|t| masked_blur_plane(t, &tm.m, &den, lw, lh, p.r_se))
                .collect()
        });
        (tm, c_g, shine)
    });
    let se = se_mask.as_ref().map(|bm| {
        let tm = tool_mask(bm);
        let den = blur_plane(&tm.m, lw, lh, p.r_se);
        let c = masked_blur_plane(&y_plane, &tm.m, &den, lw, lh, p.r_se);
        (tm, c)
    });

    // Blend into the image, rows in parallel (7.5, 7.6).
    let rows = &mut image.data[y0 as usize * iw..y1 as usize * iw];
    rows.par_chunks_mut(iw).enumerate().for_each(|(ry, row)| {
        for rx in 0..rw {
            let li = (ry / ds) * lw + rx / ds;
            let ri = ry * rw + rx;
            let keep = protect_keep(protect, x0 as usize + rx, y0 as usize + ry);
            if keep == 0.0 {
                continue;
            }
            let px = &mut row[x0 as usize + rx];
            let mut cur = [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
            let mut touched = false;
            if let Some((tm, c)) = &dc
                && tm.bytes[ri] != 0
            {
                let t = [t_planes[0][li], t_planes[1][li], t_planes[2][li]];
                let cc = [c[0][li], c[1][li], c[2][li]];
                let new = dark_circles_target(cur, t, cc);
                cur = blend_pixel(cur, new, keep * p.k_dc, p.sp_dc);
                touched = true;
            }
            if let Some((tm, c)) = &se
                && tm.bytes[ri] != 0
            {
                let new = even_skin_target(cur, y_plane[li], c[li]);
                cur = blend_pixel(cur, new, keep * p.k_se, p.sp_se);
                touched = true;
            }
            if let Some((tm, c_g, shine)) = &grade
                && tm.bytes[ri] != 0
            {
                let m = keep * f64::from(tm.bytes[ri]) / 255.0;
                let (y, cg) = (y_plane[li], c_g[li]);
                if gc != 0.0 {
                    cur = contrast_target(cur, y, cg, gc, m);
                }
                if gd > 0.0 {
                    cur = shadow_desaturation_pixel(cur, y, cg, gd, m);
                }
                if let Some(c) = shine {
                    let t = [t_planes[0][li], t_planes[1][li], t_planes[2][li]];
                    let cc = [c[0][li], c[1][li], c[2][li]];
                    let t_new = shine_removal_texture(t, y, cg, cc, gs, m);
                    for k in 0..3 {
                        cur[k] += t_new[k] - t[k];
                    }
                }
                touched = true;
            }
            if touched {
                *px = [cur[0] as f32, cur[1] as f32, cur[2] as f32];
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bm(x: i32, y: i32, w: u32, h: u32, v: u8) -> BoundedMask {
        BoundedMask::new(
            Rect::new(x, y, w, h),
            Mask::new(w, h, vec![v; (w * h) as usize]).unwrap(),
        )
        .unwrap()
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as f64 / (1u64 << 31) as f64
    }

    #[test]
    fn params_match_the_spec() {
        let p = LowFreqParams::new(1000.0, 0.5, 0.5, 0.25);
        assert!((p.tex_frac - (TEX_FRAC_MIN + 0.5 * TEX_FRAC_SLOPE)).abs() < 1e-12);
        assert!(p.half && p.ds == 2);
        // D = 1000, half resolution: r_tex = trunc(17.5 / 2), r_dc = trunc(0.077 * 1000 / 2),
        // r_se = trunc(0.055 * 1000 / 2) (S-12: the brief had 0.014, 7 and 0.044, 22).
        assert_eq!((p.r_tex, p.r_dc, p.r_se), (8, 38, 27));
        assert!((p.k_dc - 0.875).abs() < 1e-12);
        assert!((p.sp_dc - (1.0 - 0.75 * 0.0625)).abs() < 1e-12);
        assert_eq!((p.k_se, p.sp_se), (0.25, 1.0));
        let q = LowFreqParams::new(100.0, 1.0, 0.0, 0.0);
        assert_eq!(q.tex_frac, TEX_FRAC_MAX);
        assert!(!q.half && q.r_tex == 3 && q.r_dc == 7);
        assert_eq!(LowFreqParams::new(1.0, 0.0, 0.0, 0.0).r_tex, 1);
        // t = 1 gives exactly the cap; the cap never exceeds it.
        assert_eq!(
            LowFreqParams::new(10.0, 5.0, 0.0, 0.0).tex_frac,
            TEX_FRAC_MAX
        );
    }

    #[test]
    fn blur_is_two_boxes() {
        // An impulse blurred with r = 3 is the convolution of box(1) and box(2).
        let mut x = vec![0.0; 21];
        x[10] = 1.0;
        let b = blur_plane(&x, 21, 1, 3);
        let expect = |i: i64| {
            // box(1) side 3, box(2) side 5.
            let mut n = 0;
            for a in -1i64..=1 {
                for c in -2i64..=2 {
                    if a + c == i {
                        n += 1;
                    }
                }
            }
            f64::from(n) / 15.0
        };
        for (i, v) in b.iter().enumerate() {
            assert!((v - expect(i as i64 - 10)).abs() < 1e-12, "{i}");
        }
    }

    #[test]
    fn masked_blur_fixed_point_and_locality() {
        let (w, h) = (23usize, 17usize);
        let mut seed = 7u64;
        for case in 0..4 {
            let m: Vec<f64> = (0..w * h)
                .map(|i| match case {
                    0 => 0.0,
                    1 => 1.0,
                    2 => f64::from(u8::from((i % w) > 9)),
                    _ => f64::from((lcg(&mut seed) * 255.0) as u8) / 255.0,
                })
                .collect();
            let den = blur_plane(&m, w, h, 4);
            let c = vec![0.37; w * h];
            let out = masked_blur_plane(&c, &m, &den, w, h, 4);
            for v in &out {
                assert!((v - 0.37).abs() < 1e-12);
            }
            let x: Vec<f64> = (0..w * h).map(|_| lcg(&mut seed)).collect();
            let out = masked_blur_plane(&x, &m, &den, w, h, 4);
            for i in 0..w * h {
                if m[i] == 0.0 {
                    assert_eq!(out[i], x[i]);
                }
            }
        }
    }

    #[test]
    fn blend_asymmetry() {
        let src = [0.5, 0.4, 0.3];
        let dark = src.map(|v| v * 0.8);
        let out = blend_pixel(src, dark, 1.0, 1.0);
        let moved = (src[0] - out[0]) / (src[0] - dark[0]);
        assert!(moved < 0.03, "{moved}");
        let bright = src.map(|v| v * 1.2);
        for sp in [1.0, 0.5, 0.25] {
            let out = blend_pixel(src, bright, 0.8, sp);
            let w = (1.0 - sp) + sp * (1.0 / 1.2f64).powi(BRIGHTEN_EXPONENT);
            for c in 0..3 {
                let want = src[c] + 0.8 * w * (bright[c] - src[c]);
                assert!((out[c] - want).abs() < 1e-12);
            }
        }
        // Equal sums fall on the brightening branch with weight 1.
        assert_eq!(blend_weight(src, src, 0.3), 1.0);
    }

    #[test]
    fn mask_resampling() {
        let m = bm(10, 10, 20, 20, 255);
        let same = mask_to_working(&m, 1.0, (100, 100)).unwrap().unwrap();
        assert_eq!(same.region, Rect::new(10, 10, 20, 20));
        assert!(same.mask.data.iter().all(|&v| v == 255));
        let half = mask_to_working(&m, 0.5, (50, 50)).unwrap().unwrap();
        assert_eq!(half.region, Rect::new(5, 5, 10, 10));
        assert!(half.mask.data.iter().all(|&v| v == 255));
        // Clipped to the image, and off-image masks vanish.
        let edge = mask_to_working(&bm(90, 90, 30, 30, 200), 1.0, (100, 100))
            .unwrap()
            .unwrap();
        assert_eq!(edge.region, Rect::new(90, 90, 10, 10));
        assert!(
            mask_to_working(&bm(200, 0, 5, 5, 255), 1.0, (100, 100))
                .unwrap()
                .is_none()
        );
        assert!(mask_to_working(&m, 0.0, (50, 50)).is_err());
    }

    fn skin_image(w: u32, h: u32, f: impl Fn(u32, u32) -> f32) -> LinearAwg4Image {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                data.push([v, v * 0.8, v * 0.6]);
            }
        }
        LinearAwg4Image::new(w, h, data).unwrap()
    }

    fn face(scale: f32) -> Face {
        Face {
            rect: Rect::new(0, 0, 10, 10),
            scale,
        }
    }

    fn settings(dc: f32, se: f32, texture: f32) -> SkinSettings {
        SkinSettings {
            dark_circles: dc,
            even_skin: se,
            texture,
            ..Default::default()
        }
    }

    #[test]
    fn zero_amounts_and_empty_masks_do_nothing() {
        let img = skin_image(64, 64, |x, y| 0.3 + 0.001 * (x + y) as f32);
        let m = bm(10, 10, 30, 30, 255);
        let zero = BoundedMask::new(Rect::new(0, 0, 4, 4), Mask::zeros(4, 4).unwrap()).unwrap();
        let cases: [(SkinSettings, Option<&BoundedMask>, Option<&BoundedMask>); 4] = [
            (settings(0.0, 0.0, 0.0), Some(&m), Some(&m)),
            (settings(50.0, 50.0, 0.0), None, None),
            (settings(50.0, 50.0, 0.0), Some(&zero), Some(&zero)),
            (settings(0.0, 50.0, 0.0), Some(&m), None),
        ];
        for (s, a, b) in cases {
            let mut out = img.clone();
            let done =
                retouch_low_frequency(&mut out, (64, 64), &face(0.5), &s, 1.0, a, b).unwrap();
            assert!(!done);
            assert_eq!(out, img);
        }
    }

    #[test]
    fn locality_outside_masks() {
        let img = skin_image(120, 100, |x, y| {
            0.3 + 0.05 * (((x as f32) / 9.0).sin() + ((y as f32) / 7.0).cos())
        });
        let dcm = bm(20, 20, 30, 15, 255);
        let sem = bm(60, 40, 40, 40, 255);
        let mut out = img.clone();
        let done = retouch_low_frequency(
            &mut out,
            (120, 100),
            &face(0.4),
            &settings(100.0, 100.0, 0.0),
            1.0,
            Some(&dcm),
            Some(&sem),
        )
        .unwrap();
        assert!(done);
        let inside = |m: &BoundedMask, x: i32, y: i32| {
            x >= m.region.x_px
                && y >= m.region.y_px
                && i64::from(x) < m.region.right_px()
                && i64::from(y) < m.region.bottom_px()
        };
        let mut changed = 0;
        for y in 0..100i32 {
            for x in 0..120i32 {
                let i = (y * 120 + x) as usize;
                if !inside(&dcm, x, y) && !inside(&sem, x, y) {
                    assert_eq!(out.data[i], img.data[i]);
                } else if out.data[i] != img.data[i] {
                    changed += 1;
                }
            }
        }
        assert!(changed > 0);
    }

    #[test]
    fn texture_is_preserved() {
        // A flat patch with a slow bump and zero-mean fine noise (period 3 and 4 px).
        let (w, h) = (300u32, 300u32);
        let noise = |x: u32, y: u32| {
            0.01 * ((std::f32::consts::TAU * x as f32 / 3.0).sin()
                + (std::f32::consts::TAU * y as f32 / 4.0).cos())
        };
        let base = |x: u32, y: u32| {
            let (dx, dy) = (x as f32 - 150.0, y as f32 - 150.0);
            0.4 * (1.0 + 0.03 * (-(dx * dx + dy * dy) / 2000.0).exp())
        };
        let img = skin_image(w, h, |x, y| base(x, y) + noise(x, y));
        let m = bm(40, 40, 220, 220, 255);
        for half in [Some(true), Some(false), None] {
            let mut out = img.clone();
            retouch_with(
                &mut out,
                (w, h),
                &face(0.8),
                &settings(0.0, 100.0, 0.0),
                1.0,
                None,
                Some(&m),
                half,
            )
            .unwrap();
            // Amplitude: mean square of horizontal neighbour differences, interior only.
            let amp = |im: &LinearAwg4Image| {
                let mut acc = 0.0f64;
                let mut n = 0.0;
                for y in 100..200usize {
                    for x in 100..199usize {
                        let a = f64::from(im.data[y * w as usize + x][0]);
                        let b = f64::from(im.data[y * w as usize + x + 1][0]);
                        acc += (a - b) * (a - b);
                        n += 1.0;
                    }
                }
                (acc / n).sqrt()
            };
            let (before, after) = (amp(&img), amp(&out));
            assert!(
                (after / before - 1.0).abs() < 0.05,
                "half {half:?}: {before} -> {after}"
            );
            // The bump is flattened, so the change is real.
            assert_ne!(
                out.data[150 * w as usize + 150],
                img.data[150 * w as usize + 150]
            );
        }
    }

    #[test]
    fn half_path_agrees_with_full_path() {
        let (w, h) = (240u32, 240u32);
        let img = skin_image(w, h, |x, y| {
            let (dx, dy) = (x as f32 - 120.0, y as f32 - 120.0);
            0.35 + 0.06 * (-(dx * dx + dy * dy) / 3000.0).exp() + 0.0005 * x as f32 / 10.0
        });
        let m = bm(50, 50, 140, 140, 255);
        let run = |half| {
            let mut out = img.clone();
            retouch_with(
                &mut out,
                (w, h),
                &face(0.8),
                &settings(70.0, 80.0, 0.0),
                1.0,
                Some(&m),
                Some(&m),
                Some(half),
            )
            .unwrap();
            out
        };
        let (a, b) = (run(true), run(false));
        let mut acc = 0.0f64;
        for (p, q) in a.data.iter().zip(&b.data) {
            acc += (0..3).map(|c| f64::from((p[c] - q[c]).abs())).sum::<f64>() / 3.0;
        }
        let mad = acc / a.data.len() as f64;
        assert!(mad < 2e-3, "mean abs diff {mad}");
    }

    #[test]
    fn independent_of_thread_count() {
        let img = skin_image(130, 90, |x, y| {
            0.3 + 0.1 * ((x * 7 + y * 13) % 17) as f32 / 17.0
        });
        let m = bm(15, 10, 90, 70, 200);
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut out = img.clone();
                retouch_low_frequency(
                    &mut out,
                    (130, 90),
                    &face(0.9),
                    &settings(60.0, 90.0, 30.0),
                    1.0,
                    Some(&m),
                    Some(&m),
                )
                .unwrap();
                out
            })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn region_crop_matches_whole_image() {
        // Same result whether the mask is small (cropped region) or the image is processed whole:
        // compare against a run where the image is padded with an unrelated margin.
        let img = skin_image(200, 200, |x, y| {
            0.3 + 0.1 * (x as f32 / 30.0).sin() * (y as f32 / 40.0).cos()
        });
        let m = bm(80, 80, 40, 40, 255);
        let mut a = img.clone();
        retouch_with(
            &mut a,
            (200, 200),
            &face(0.9),
            &settings(0.0, 100.0, 0.0),
            1.0,
            None,
            Some(&m),
            Some(true),
        )
        .unwrap();
        // Reference: a mask that covers the whole image but is zero outside the box.
        let mut full = Mask::zeros(200, 200).unwrap();
        for y in 80..120 {
            for x in 80..120 {
                full.data[y * 200 + x] = 255;
            }
        }
        let fm = BoundedMask::new(Rect::new(0, 0, 200, 200), full).unwrap();
        let mut b = img.clone();
        retouch_with(
            &mut b,
            (200, 200),
            &face(0.9),
            &settings(0.0, 100.0, 0.0),
            1.0,
            None,
            Some(&fm),
            Some(true),
        )
        .unwrap();
        // The crop makes the region of `fm` smaller too (cropped to non-zero), so both agree.
        assert_eq!(a, b);
    }

    // ---- skin grading and teeth (7.7, 7.8) ----

    fn lum(p: [f32; 3]) -> f64 {
        (f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2])) / 3.0
    }

    fn chroma(p: [f32; 3]) -> f64 {
        let l = lum(p);
        (0..3).map(|c| (f64::from(p[c]) - l).abs()).sum::<f64>()
    }

    fn grading(gc: f32, gd: f32, gs: f32) -> SkinSettings {
        SkinSettings {
            skin_contrast: gc,
            shadow_desaturation: gd,
            shine_removal: gs,
            ..Default::default()
        }
    }

    fn run_grading(
        img: &LinearAwg4Image,
        s: &SkinSettings,
        mask: &BoundedMask,
        scale: f32,
    ) -> LinearAwg4Image {
        let mut out = img.clone();
        let masks = ToolMasks {
            grading: Some(mask),
            ..ToolMasks::default()
        };
        let done = retouch_tones(
            &mut out,
            (img.width_px, img.height_px),
            &face(scale),
            s,
            1.0,
            &masks,
            None,
        )
        .unwrap();
        assert!(done);
        out
    }

    /// Standard deviation of the blurred luminance (the low-frequency light) in the interior.
    fn low_spread(img: &LinearAwg4Image) -> f64 {
        let (w, h) = (img.width_px as usize, img.height_px as usize);
        let l: Vec<f64> = img.data.iter().map(|p| lum(*p)).collect();
        let b = blur_plane(&l, w, h, 8);
        let mut v = Vec::new();
        for y in 100..200 {
            for x in 100..200 {
                v.push(b[y * w + x].ln());
            }
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    fn fine_amplitude(img: &LinearAwg4Image) -> f64 {
        let w = img.width_px as usize;
        let (mut acc, mut n) = (0.0f64, 0.0f64);
        for y in 100..200usize {
            for x in 100..199usize {
                let d = f64::from(img.data[y * w + x][0]) - f64::from(img.data[y * w + x + 1][0]);
                acc += d * d;
                n += 1.0;
            }
        }
        (acc / n).sqrt()
    }

    #[test]
    fn contrast_widens_and_narrows_the_low_frequency_spread() {
        let (w, h) = (300u32, 300u32);
        let tau = std::f32::consts::TAU;
        let img = skin_image(w, h, |x, y| {
            let slow = 1.0 + 0.06 * (tau * x as f32 / 90.0).sin() * (tau * y as f32 / 110.0).cos();
            let fine = 0.012 * ((tau * x as f32 / 3.0).sin() + (tau * y as f32 / 4.0).cos());
            0.4 * slow + fine
        });
        let m = bm(40, 40, 220, 220, 255);
        let (base_spread, base_fine) = (low_spread(&img), fine_amplitude(&img));
        let up = run_grading(&img, &grading(100.0, 0.0, 0.0), &m, 0.8);
        let down = run_grading(&img, &grading(-100.0, 0.0, 0.0), &m, 0.8);
        let (su, sd) = (
            low_spread(&up) / base_spread,
            low_spread(&down) / base_spread,
        );
        assert!(su > 1.2, "+100 widens the spread: x{su}");
        assert!(sd < 0.8, "-100 narrows the spread: x{sd}");
        for (name, out) in [("+100", &up), ("-100", &down)] {
            let r = fine_amplitude(out) / base_fine;
            assert!((r - 1.0).abs() < 0.05, "{name}: texture energy x{r}");
        }
        // Half amounts sit between, outside the mask nothing moves.
        let half = run_grading(&img, &grading(50.0, 0.0, 0.0), &m, 0.8);
        let sh = low_spread(&half) / base_spread;
        assert!(1.0 < sh && sh < su, "{sh}");
        assert_eq!(up.data[10], img.data[10]);
    }

    #[test]
    fn shadow_desaturation_acts_only_below_the_skin_level() {
        // A dark half on the left, a bright half on the right, with a colour cast.
        let (w, h) = (300u32, 300u32);
        let img = skin_image(w, h, |x, _| if x < 150 { 0.2 } else { 0.4 });
        let m = bm(40, 40, 220, 220, 255);
        let out = run_grading(&img, &grading(0.0, 100.0, 0.0), &m, 0.8);
        let at = |im: &LinearAwg4Image, x: usize, y: usize| im.data[y * w as usize + x];
        // Next to the step on the dark side: less chroma, the same light.
        for x in 140..146 {
            let (a, b) = (at(&img, x, 150), at(&out, x, 150));
            assert!(
                chroma(b) < 0.8 * chroma(a),
                "x {x}: {} -> {}",
                chroma(a),
                chroma(b)
            );
            assert!((lum(a) - lum(b)).abs() < 1e-6, "luminance kept at x {x}");
        }
        // At or above the skin's level: unchanged. Far on the dark side C_g equals Y: unchanged.
        for x in (60..100).chain(165..255) {
            let (a, b) = (at(&img, x, 150), at(&out, x, 150));
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-6, "x {x}");
            }
        }
        // Less amount, less change.
        let mild = run_grading(&img, &grading(0.0, 30.0, 0.0), &m, 0.8);
        assert!(chroma(at(&mild, 143, 150)) > chroma(at(&out, 143, 150)));
    }

    #[test]
    fn shine_removal_pulls_a_highlight_toward_the_skin() {
        let (w, h) = (300u32, 300u32);
        let skin = [0.4f32, 0.31, 0.25];
        let patch = |x: u32, y: u32| {
            let d2 = (x as f32 - 150.0).powi(2) + (y as f32 - 150.0).powi(2);
            0.4 * (-d2 / (2.0 * 7.0 * 7.0)).exp()
        };
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let s = patch(x, y);
                data.push([skin[0] + s, skin[1] + s, skin[2] + s]);
            }
        }
        let img = LinearAwg4Image::new(w, h, data).unwrap();
        let m = bm(40, 40, 220, 220, 255);
        let full = run_grading(&img, &grading(0.0, 0.0, 100.0), &m, 0.8);
        let half = run_grading(&img, &grading(0.0, 0.0, 50.0), &m, 0.8);
        let centre = |im: &LinearAwg4Image| lum(im.data[150 * w as usize + 150]);
        let (before, after, mid) = (centre(&img), centre(&full), centre(&half));
        let level = lum(skin);
        assert!(before - level > 0.3, "a real highlight: {before}");
        assert!(
            after - level < 0.5 * (before - level),
            "{before} -> {after}, skin {level}"
        );
        assert!(after < mid && mid < before, "{after} {mid} {before}");
        assert!(after.is_finite() && after > 0.0);
        // Skin at its own level, far from the patch, is unchanged.
        for (x, y) in [(60, 60), (240, 70), (70, 230), (230, 230), (150, 80)] {
            let (a, b) = (img.data[y * w as usize + x], full.data[y * w as usize + x]);
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-6, "({x}, {y})");
            }
        }
        // Flat skin has no highlight at all.
        let flat = LinearAwg4Image::filled(w, h, skin).unwrap();
        let out = run_grading(&flat, &grading(0.0, 0.0, 100.0), &m, 0.8);
        for (p, q) in flat.data.iter().zip(&out.data) {
            for c in 0..3 {
                assert!((p[c] - q[c]).abs() < 1e-6);
            }
        }
    }

    fn mouth_picture() -> (LinearAwg4Image, Face, crate::FaceMasks, (u32, u32)) {
        let full = (400u32, 320u32);
        let face = Face {
            rect: Rect::new(100, 40, 200, 250),
            scale: 0.5,
        };
        let mut image = LinearAwg4Image::filled(full.0, full.1, [0.45, 0.35, 0.28]).unwrap();
        crate::mock::paint_mouth(&mut image, &face.rect, 1.0);
        let models = crate::mock::models(vec![]);
        let masks = crate::face_masks(&models, &image, &face, full, None).unwrap();
        (image, face, masks, full)
    }

    #[test]
    fn teeth_whitening_whitens_teeth_and_nothing_else() {
        let (image, face, masks, full) = mouth_picture();
        let teeth = masks.teeth.as_ref().unwrap();
        let tools = ToolMasks {
            teeth: Some(teeth),
            ..ToolMasks::default()
        };
        let s = SkinSettings {
            teeth_whitening: 100.0,
            ..Default::default()
        };
        let mut out = image.clone();
        assert!(retouch_tones(&mut out, full, &face, &s, 1.0, &tools, None).unwrap());
        let (cx, cy, ax, ay) = crate::mock::mouth_ellipse(&face.rect);
        let w = full.0 as usize;
        let idx = |x: f64, y: f64| y as usize * w + x as usize;
        // Yellow teeth lose chroma; the relative cast (chroma over light) drops a lot.
        for dx in [-0.5, 0.0, 0.5] {
            let i = idx(cx + dx * ax, cy);
            let (a, b) = (image.data[i], out.data[i]);
            assert!(
                chroma(b) / lum(b) < 0.6 * chroma(a) / lum(a),
                "{a:?} -> {b:?}"
            );
            assert!(lum(b) >= lum(a) - 1e-6);
        }
        // Lips, gums (the dark gap) and skin are outside the mask: unchanged to the bit.
        for (x, y) in [
            (cx + 1.25 * ax, cy),
            (cx - 1.25 * ax, cy),
            (cx, cy + 0.8 * ay),
            (cx, cy - 0.8 * ay),
            (cx + 0.95 * ax, cy),
            (cx, cy + 1.3 * ay),
            (10.0, 10.0),
        ] {
            assert_eq!(image.data[idx(x, y)], out.data[idx(x, y)], "({x}, {y})");
        }
        // Where the teeth mask is 0 nothing moves at all.
        let tf = teeth.to_full(full).unwrap();
        for (i, (a, b)) in image.data.iter().zip(&out.data).enumerate() {
            if tf.data[i] == 0 {
                assert_eq!(a, b, "pixel {i}");
            }
        }
        // No tooth ends brighter than the brightest tooth was.
        let brightest = image
            .data
            .iter()
            .zip(&tf.data)
            .filter(|(_, m)| **m > 0)
            .map(|(p, _)| lum(*p))
            .fold(0.0f64, f64::max);
        let after = out
            .data
            .iter()
            .zip(&tf.data)
            .filter(|(_, m)| **m > 0)
            .map(|(p, _)| lum(*p))
            .fold(0.0f64, f64::max);
        assert!(after <= brightest + 1e-6, "{after} against {brightest}");
        // Half the amount whitens less.
        let mut half = image.clone();
        let s50 = SkinSettings {
            teeth_whitening: 50.0,
            ..s
        };
        retouch_tones(&mut half, full, &face, &s50, 1.0, &tools, None).unwrap();
        let i = idx(cx, cy);
        assert!(chroma(half.data[i]) > chroma(out.data[i]));
    }

    #[test]
    fn identity_without_amounts_and_masks() {
        let (image, face, masks, full) = mouth_picture();
        let tools = ToolMasks {
            dark_circles: masks.dark_circles.as_ref(),
            even_skin: masks.even_skin.as_ref(),
            grading: masks.grading.as_ref(),
            teeth: masks.teeth.as_ref(),
        };
        let mut out = image.clone();
        let done = retouch_tones(
            &mut out,
            full,
            &face,
            &SkinSettings::default(),
            1.0,
            &tools,
            None,
        )
        .unwrap();
        assert!(!done);
        assert_eq!(out, image);
        // Amounts above 0 but no masks: nothing either.
        let all = SkinSettings {
            dark_circles: 80.0,
            even_skin: 80.0,
            skin_contrast: 80.0,
            shadow_desaturation: 80.0,
            shine_removal: 80.0,
            teeth_whitening: 80.0,
            ..Default::default()
        };
        let done = retouch_tones(
            &mut out,
            full,
            &face,
            &all,
            1.0,
            &ToolMasks::default(),
            None,
        )
        .unwrap();
        assert!(!done);
        assert_eq!(out, image);
    }

    #[test]
    fn grading_locality_and_threads() {
        let (image, face, masks, full) = mouth_picture();
        let tools = ToolMasks {
            dark_circles: masks.dark_circles.as_ref(),
            even_skin: masks.even_skin.as_ref(),
            grading: masks.grading.as_ref(),
            teeth: masks.teeth.as_ref(),
        };
        let all = SkinSettings {
            dark_circles: 70.0,
            even_skin: 70.0,
            skin_contrast: 70.0,
            shadow_desaturation: 70.0,
            shine_removal: 70.0,
            teeth_whitening: 70.0,
            ..Default::default()
        };
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut out = image.clone();
                assert!(retouch_tones(&mut out, full, &face, &all, 1.0, &tools, None).unwrap());
                out
            })
        };
        let one = run(1);
        assert_ne!(one, image);
        assert_eq!(one, run(4));
        // Outside every mask: bit-identical.
        let mut union = Mask::zeros(full.0, full.1).unwrap();
        for bm in [
            &masks.dark_circles,
            &masks.even_skin,
            &masks.grading,
            &masks.teeth,
        ]
        .into_iter()
        .flatten()
        {
            let f = bm.to_full(full).unwrap();
            for (u, v) in union.data.iter_mut().zip(&f.data) {
                *u = (*u).max(*v);
            }
        }
        let mut changed = 0;
        for (i, (a, b)) in image.data.iter().zip(&one.data).enumerate() {
            if union.data[i] == 0 {
                assert_eq!(a, b, "pixel {i}");
            } else if a != b {
                changed += 1;
            }
        }
        assert!(changed > 1000, "{changed}");
    }

    #[test]
    fn grading_half_path_agrees_with_full_path() {
        let (w, h) = (240u32, 240u32);
        let img = skin_image(w, h, |x, y| {
            let (dx, dy) = (x as f32 - 120.0, y as f32 - 120.0);
            0.35 + 0.1 * (-(dx * dx + dy * dy) / 3000.0).exp() + 0.0005 * x as f32 / 10.0
        });
        let m = bm(50, 50, 140, 140, 255);
        let s = grading(60.0, 60.0, 60.0);
        let run = |half| {
            let mut out = img.clone();
            let masks = ToolMasks {
                grading: Some(&m),
                ..ToolMasks::default()
            };
            tones_with(
                &mut out,
                (w, h),
                &face(0.8),
                &s,
                1.0,
                &masks,
                None,
                None,
                Some(half),
            )
            .unwrap();
            out
        };
        let (a, b) = (run(true), run(false));
        let mut acc = 0.0f64;
        for (p, q) in a.data.iter().zip(&b.data) {
            acc += (0..3).map(|c| f64::from((p[c] - q[c]).abs())).sum::<f64>() / 3.0;
        }
        assert!(acc / (a.data.len() as f64) < 3e-3);
        assert_ne!(a, img);
    }

    proptest! {
        #[test]
        fn grading_kernels_are_finite(
            s in prop::array::uniform3(-2.0f64..20.0),
            t in prop::array::uniform3(-1.0f64..10.0),
            c in prop::array::uniform3(-1.0f64..10.0),
            y in -1.0f64..10.0,
            cg in -1.0f64..10.0,
            gc in -1.0f64..=1.0,
            g in 0.0f64..=1.0,
            m in 0.0f64..=1.0,
            lmax in 0.0f64..10.0,
        ) {
            prop_assert!(contrast_target(s, y, cg, gc, m).iter().all(|v| v.is_finite()));
            prop_assert!(shadow_desaturation_pixel(s, y, cg, g, m).iter().all(|v| v.is_finite()));
            prop_assert!(shine_removal_texture(t, y, cg, c, g, m).iter().all(|v| v.is_finite()));
            prop_assert!(teeth_pixel(s, g, m, lmax).iter().all(|v| v.is_finite()));
        }

        #[test]
        fn contrast_is_monotonic_in_the_amount(
            s in prop::array::uniform3(0.01f64..5.0),
            y in 0.01f64..5.0,
            cg in 0.01f64..5.0,
            a in -1.0f64..=1.0,
            b in -1.0f64..=1.0,
        ) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let gain = |gc: f64| contrast_target(s, y, cg, gc, 1.0)[0] / s[0];
            // Above the skin's level more contrast lifts, below it lowers.
            if y >= cg {
                prop_assert!(gain(lo) <= gain(hi) + 1e-9);
            } else {
                prop_assert!(gain(lo) >= gain(hi) - 1e-9);
            }
            // The skin's own level is a fixed point.
            let at_level = contrast_target(s, cg, cg, hi, 1.0);
            prop_assert!(at_level.iter().zip(&s).all(|(o, i)| (o - i).abs() <= 1e-9 * i.abs().max(1.0) || cg < 0.001));
        }

        #[test]
        fn shadow_desaturation_keeps_light_and_is_monotonic(
            s in prop::array::uniform3(0.0f64..5.0),
            y in 0.0f64..5.0,
            cg in 0.01f64..5.0,
            a in 0.0f64..=1.0,
            b in 0.0f64..=1.0,
            m in 0.0f64..=1.0,
        ) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let c = |p: [f64; 3]| { let l = (p[0] + p[1] + p[2]) / 3.0; (0..3).map(|k| (p[k] - l).abs()).sum::<f64>() };
            let (p, q) = (shadow_desaturation_pixel(s, y, cg, lo, m), shadow_desaturation_pixel(s, y, cg, hi, m));
            prop_assert!(c(q) <= c(p) + 1e-9);
            prop_assert!(c(p) <= c(s) + 1e-9);
            prop_assert!(((q[0] + q[1] + q[2]) - (s[0] + s[1] + s[2])).abs() < 1e-9);
            if y >= cg {
                prop_assert!(q.iter().zip(&s).all(|(o, i)| (o - i).abs() < 1e-9));
            }
        }

        #[test]
        fn shine_removal_moves_more_with_the_amount(
            t in prop::array::uniform3(0.01f64..5.0),
            c in prop::array::uniform3(0.01f64..5.0),
            cg in 0.05f64..3.0,
            a in 0.0f64..=1.0,
            b in 0.0f64..=1.0,
            m in 0.0f64..=1.0,
        ) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let y = (t[0] + t[1] + t[2]) / 3.0;
            let dist = |g: f64| { let o = shine_removal_texture(t, y, cg, c, g, m); (0..3).map(|k| (o[k] - t[k]).abs()).sum::<f64>() };
            prop_assert!(dist(lo) <= dist(hi) + 1e-9);
            // Not above the skin's level by 10 %: nothing happens.
            if y <= cg * 1.1 {
                prop_assert!(dist(hi) == 0.0);
            }
        }

        #[test]
        fn teeth_pixel_is_monotonic_and_capped(
            s in prop::array::uniform3(0.0f64..5.0),
            lmax in 0.0f64..5.0,
            a in 0.0f64..=1.0,
            b in 0.0f64..=1.0,
            m in 0.0f64..=1.0,
        ) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let l = |p: [f64; 3]| (p[0] + p[1] + p[2]) / 3.0;
            let c = |p: [f64; 3]| { let m = l(p); (0..3).map(|k| (p[k] - m).abs()).sum::<f64>() };
            let (p, q) = (teeth_pixel(s, lo, m, lmax), teeth_pixel(s, hi, m, lmax));
            // More whitening: less chroma relative to light, no darker.
            prop_assert!(l(q) >= l(p) - 1e-9);
            prop_assert!(l(p) >= l(s) - 1e-9);
            if l(q) > 1e-9 {
                prop_assert!(c(q) / l(q) <= c(p) / l(p).max(1e-9) + 1e-9);
            }
            // Never above the brightest tooth (or the pixel itself when it is already above).
            prop_assert!(l(q) <= l(s).max(lmax) + 1e-9);
            // Zero amount is the identity.
            let id = teeth_pixel(s, 0.0, m, lmax);
            prop_assert!(id.iter().zip(&s).all(|(o, i)| (o - i).abs() < 1e-12));
        }
    }

    proptest! {
        #[test]
        fn kernels_are_finite(
            s in prop::array::uniform3(-2.0f64..20.0),
            n in prop::array::uniform3(-2.0f64..20.0),
            k in 0.0f64..=1.0,
            sp in 0.0f64..=1.0,
            y in -1.0f64..5.0,
            c in -1.0f64..5.0,
            m in 0.0f64..=1.0,
            num in -5.0f64..5.0,
            den in 0.0f64..5.0,
        ) {
            prop_assert!(blend_pixel(s, n, k, sp).iter().all(|v| v.is_finite()));
            let w = blend_weight(s, n, sp);
            prop_assert!(w.is_finite() && (0.0..=1.0 + 1e-12).contains(&w));
            prop_assert!(even_skin_target(s, y, c).iter().all(|v| v.is_finite()));
            prop_assert!(dark_circles_target(s, n, [c; 3]).iter().all(|v| v.is_finite()));
            prop_assert!(masked_blur_combine(m, num, den, y).is_finite());
        }

        #[test]
        fn moved_fraction_is_monotonic_in_the_amount(
            s in prop::array::uniform3(0.01f64..5.0),
            lift in 1.0f64..3.0,
            a in 0.0f64..=1.0,
            b in 0.0f64..=1.0,
        ) {
            let new = s.map(|v| v * lift);
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let frac = |dc: f64| {
                let (k, sp) = dark_circle_k_sp(dc);
                k * blend_weight(s, new, sp)
            };
            prop_assert!(frac(lo) <= frac(hi) + 1e-12);
            // Even skin: k = se, sp = 1.
            let fe = |se: f64| se * blend_weight(s, new, 1.0);
            prop_assert!(fe(lo) <= fe(hi) + 1e-12);
        }

        #[test]
        fn constant_image_is_a_fixed_point_of_the_whole_operator(
            v in 0.05f32..2.0,
            dc in 0.0f32..=100.0,
            se in 1.0f32..=100.0,
        ) {
            let mut img = skin_image(48, 40, |_, _| v);
            let before = img.clone();
            let m = bm(8, 8, 30, 24, 255);
            retouch_low_frequency(
                &mut img, (48, 40), &face(0.9), &settings(dc, se, 0.0), 1.0, Some(&m), Some(&m),
            ).unwrap();
            for (p, q) in img.data.iter().zip(&before.data) {
                for c in 0..3 {
                    prop_assert!((p[c] - q[c]).abs() <= 1e-5 * q[c].abs().max(1.0));
                }
            }
        }
    }
}
