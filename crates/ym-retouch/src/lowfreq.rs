//! Dark circles and even skin (`docs/27` section 7).
//!
//! One low-frequency operator, per face, on the working image: `T` is a blur of the image at the
//! texture radius, `src - T` is the texture that is kept, and a mask-normalised blur of `T` gives
//! the regional average that dark circles (colour) and even skin (luminance) move the skin toward.
//! The asymmetric blend of 7.5 lets the skin brighten and almost never darken.
//!
//! The pure per-pixel kernels are free functions ([`blend_weight`], [`blend_pixel`],
//! [`masked_blur_combine`], [`dark_circles_target`], [`even_skin_target`], [`LowFreqParams`]) so a
//! GPU kernel can follow with the same maths. The image passes run over rows with `rayon` and
//! write disjoint rows, so results never depend on the thread count.

use rayon::prelude::*;

use crate::error::Result;
use crate::types::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, check_render_scale, face_size_px,
};

/// The derived parameters of `docs/27` section 7.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowFreqParams {
    /// `min(0.004 + 0.02 * t, 0.024)`: the texture radius as a fraction of the face size.
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
        let tex_frac = (0.004 + 0.02 * t).min(0.024);
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
            r_dc: radius(0.077 * d, ds),
            r_se: radius(0.044 * d, ds),
            k_dc,
            sp_dc,
            k_se: se.clamp(0.0, 1.0),
            sp_se: 1.0,
        }
    }
}

fn sum3(p: [f64; 3]) -> f64 {
    (p[0] + p[1] + p[2]).max(0.001)
}

/// The weight `w` of 7.5: `(Sum(new) / Sum(src))^16` when the pixel gets darker, else
/// `(1 - sp) + sp * (Sum(src) / Sum(new))^2`, with `Sum(p) = max(r + g + b, 0.001)`.
pub fn blend_weight(src: [f64; 3], new: [f64; 3], sp: f64) -> f64 {
    let (s, n) = (sum3(src), sum3(new));
    if n < s {
        (n / s).powi(16)
    } else {
        let q = s / n;
        (1.0 - sp) + sp * q * q
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

/// Dark circles then even skin on the working image, in place (`docs/27` section 7).
///
/// `full_size_px` is the full-resolution size the masks were built at; the masks are resampled by
/// `render_scale` to the working image ([`mask_to_working`]). A tool is active when its amount is
/// above 0 and its mask exists and is not all zeros. Returns `Ok(false)` and leaves the image
/// untouched when neither tool is active. Pixels where a tool's mask is 0 are not touched by it.
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
    image.validate()?;
    let d = face_size_px(full_size_px, render_scale, face)?;
    let size = (image.width_px, image.height_px);
    let working = |amount: f64, mask: Option<&BoundedMask>| -> Result<Option<BoundedMask>> {
        match mask {
            Some(m) if amount > 0.0 && !m.mask.is_all_zero() => {
                mask_to_working(m, render_scale, size)
            }
            _ => Ok(None),
        }
    };
    let dc_mask = working(settings.dc(), dark_circle_mask)?;
    let se_mask = working(settings.se(), even_skin_mask)?;
    if dc_mask.is_none() && se_mask.is_none() {
        return Ok(false);
    }
    let p = LowFreqParams::with_half(d, settings.t(), settings.dc(), settings.se(), half_override);
    let ds = p.ds;

    // The padded, clipped working region; its origin is even on the half path so that its 2 x 2
    // blocks sit on the same grid as the whole image's.
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    let mut reach = 0usize;
    for (m, r) in [(&dc_mask, p.r_dc), (&se_mask, p.r_se)] {
        if let Some(m) = m {
            x0 = x0.min(i64::from(m.region.x_px));
            y0 = y0.min(i64::from(m.region.y_px));
            x1 = x1.max(m.region.right_px());
            y1 = y1.max(m.region.bottom_px());
            reach = reach.max(r);
        }
    }
    let pad = ((p.r_tex + reach) * ds + ds) as i64;
    let even = |v: i64| if p.half { v & !1 } else { v };
    let x0 = even((x0 - pad).max(0));
    let y0 = even((y0 - pad).max(0));
    let x1 = (x1 + pad).min(i64::from(size.0));
    let y1 = (y1 + pad).min(i64::from(size.1));
    let region = Rect::new(x0 as i32, y0 as i32, (x1 - x0) as u32, (y1 - y0) as u32);
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
            let px = &mut row[x0 as usize + rx];
            let mut cur = [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
            let mut touched = false;
            if let Some((tm, c)) = &dc
                && tm.bytes[ri] != 0
            {
                let t = [t_planes[0][li], t_planes[1][li], t_planes[2][li]];
                let cc = [c[0][li], c[1][li], c[2][li]];
                let new = dark_circles_target(cur, t, cc);
                cur = blend_pixel(cur, new, p.k_dc, p.sp_dc);
                touched = true;
            }
            if let Some((tm, c)) = &se
                && tm.bytes[ri] != 0
            {
                let new = even_skin_target(cur, y_plane[li], c[li]);
                cur = blend_pixel(cur, new, p.k_se, p.sp_se);
                touched = true;
            }
            if touched {
                *px = [cur[0] as f32, cur[1] as f32, cur[2] as f32];
            }
        }
    });
    Ok(true)
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
        assert!((p.tex_frac - 0.014).abs() < 1e-12);
        assert!(p.half && p.ds == 2);
        assert_eq!((p.r_tex, p.r_dc, p.r_se), (7, 38, 22));
        assert!((p.k_dc - 0.875).abs() < 1e-12);
        assert!((p.sp_dc - (1.0 - 0.75 * 0.0625)).abs() < 1e-12);
        assert_eq!((p.k_se, p.sp_se), (0.25, 1.0));
        let q = LowFreqParams::new(100.0, 1.0, 0.0, 0.0);
        assert_eq!(q.tex_frac, 0.024);
        assert!(!q.half && q.r_tex == 2 && q.r_dc == 7);
        assert_eq!(LowFreqParams::new(1.0, 0.0, 0.0, 0.0).r_tex, 1);
        // t = 1 gives exactly the 0.024 cap; the cap never exceeds it.
        assert_eq!(LowFreqParams::new(10.0, 5.0, 0.0, 0.0).tex_frac, 0.024);
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
            let w = (1.0 - sp) + sp * (1.0 / 1.2f64).powi(2);
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
