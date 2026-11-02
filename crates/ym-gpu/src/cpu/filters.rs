//! Shared building blocks of the CPU stages: row-band parallel runners, box means and Gaussian
//! kernels.
//!
//! The band runners are what make the stages tile-consistent *by construction*: a stage is written
//! as a plain whole-image function, and the runner cuts the image into horizontal bands with the
//! stage's halo of context rows, runs the function on each band in parallel and keeps only the
//! core rows. Because bands use [`Mosaic::extract`] / [`RgbImage::extract`], the CFA phase and the
//! placement are right inside every band. The band layout depends only on the image height, never
//! on the thread count.

use std::ops::Range;

use rayon::prelude::*;

use super::{Mosaic, Rect, RgbImage};

/// Default number of core rows per band.
pub const BAND_ROWS: usize = 128;

/// Splits `0..h` into consecutive bands of at most `band_rows` rows (all but the last have exactly
/// `band_rows`).
pub fn bands(h: usize, band_rows: usize) -> Vec<Range<usize>> {
    let band_rows = band_rows.max(1);
    (0..h.div_ceil(band_rows))
        .map(|b| b * band_rows..((b + 1) * band_rows).min(h))
        .collect()
}

/// The extended row range `[start - halo, end + halo)` of a band, clamped to `0..h` and moved down
/// to an even start so the Bayer phase and the superpixel grid line up.
pub fn extended_rows(band: &Range<usize>, halo: usize, h: usize) -> Range<usize> {
    let start = band.start.saturating_sub(halo) & !1;
    let end = (band.end + halo).min(h);
    start..end
}

/// Splits the three planes of `out` into per-band mutable row slices.
fn split_planes<'a>(
    out: &'a mut RgbImage,
    bands: &[Range<usize>],
) -> Vec<(Range<usize>, [&'a mut [f32]; 3])> {
    let w = out.width;
    let [mut r, mut g, mut b] = out.planes_mut();
    let mut result = Vec::with_capacity(bands.len());
    for band in bands {
        let len = band.len() * w;
        let (r1, r2) = r.split_at_mut(len);
        let (g1, g2) = g.split_at_mut(len);
        let (b1, b2) = b.split_at_mut(len);
        result.push((band.clone(), [r1, g1, b1]));
        r = r2;
        g = g2;
        b = b2;
    }
    result
}

fn copy_core(local: &RgbImage, local_core: Range<usize>, dst: [&mut [f32]; 3]) {
    let w = local.width;
    for (c, d) in dst.into_iter().enumerate() {
        let src = local.plane(c);
        d.copy_from_slice(&src[local_core.start * w..local_core.end * w]);
    }
}

/// Runs `f` on bands of a mosaic (each with `halo` rows of context above and below) and assembles
/// the core rows of the results into one image. `f` must return an image of the same size as the
/// band it was given.
pub fn run_mosaic_bands<F>(m: &Mosaic, halo: usize, band_rows: usize, f: F) -> RgbImage
where
    F: Fn(&Mosaic) -> RgbImage + Sync,
{
    let mut out = RgbImage::new(m.width, m.height);
    out.place = m.place;
    let bs = bands(m.height, band_rows);
    let slices = split_planes(&mut out, &bs);
    slices.into_par_iter().for_each(|(band, dst)| {
        let ext = extended_rows(&band, halo, m.height);
        let local = m.extract(Rect::new(0, ext.start, m.width, ext.len()));
        let res = f(&local);
        debug_assert_eq!((res.width, res.height), (local.width, local.height));
        copy_core(&res, band.start - ext.start..band.end - ext.start, dst);
    });
    out
}

/// Like [`run_mosaic_bands`] for an RGB image in, RGB image out.
pub fn run_rgb_bands<F>(img: &RgbImage, halo: usize, band_rows: usize, f: F) -> RgbImage
where
    F: Fn(&RgbImage) -> RgbImage + Sync,
{
    let mut out = RgbImage::new(img.width, img.height);
    out.place = img.place;
    let bs = bands(img.height, band_rows);
    let slices = split_planes(&mut out, &bs);
    slices.into_par_iter().for_each(|(band, dst)| {
        let ext = band.start.saturating_sub(halo)..(band.end + halo).min(img.height);
        let local = img.extract(Rect::new(0, ext.start, img.width, ext.len()));
        let res = f(&local);
        debug_assert_eq!((res.width, res.height), (local.width, local.height));
        copy_core(&res, band.start - ext.start..band.end - ext.start, dst);
    });
    out
}

/// Mean over a `(2r + 1)` square window clipped to the image (each output is the mean of the
/// samples that exist), computed separably with `f64` sums. `src` and `out` are `w * h`; `tmp` is
/// scratch of the same size.
///
/// Every window is summed on its own, in order from its first sample (`docs/17-gpu.md` D4: no running
/// sum). A running sum that adds a sample at one end and takes one off at the other keeps the rounding
/// of every sum it passed through, and when the plane spans many powers of two (the squares of a light
/// on a dark ground) that residue depends on where the sum began: a tile, whose rows and columns begin
/// elsewhere, then differs from the whole frame in the last bits. A pixel's mean here depends on its
/// own window and the window's clipping at the image's edge, so it is the same bits in any crop that
/// holds the window.
pub fn box_mean(src: &[f32], w: usize, h: usize, r: usize, out: &mut [f32], tmp: &mut [f32]) {
    debug_assert!(src.len() == w * h && out.len() == w * h && tmp.len() == w * h);
    if w == 0 || h == 0 {
        return;
    }
    // Horizontal pass: src -> tmp. One tap of the window at a time over the row, so that the loop runs
    // along contiguous samples; the taps of a pixel come in ascending order (the window, left to right).
    let mut sums = vec![0.0f64; w];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        sums.fill(0.0);
        for d in 0..=2 * r {
            // Tap `x + d - r` of pixel `x`, where it exists.
            let first = r.saturating_sub(d);
            let last = w.min((w + r).saturating_sub(d));
            for x in first..last {
                sums[x] += f64::from(row[x + d - r]);
            }
        }
        let trow = &mut tmp[y * w..(y + 1) * w];
        for (x, (t, &s)) in trow.iter_mut().zip(&sums).enumerate() {
            let count = (x + r + 1).min(w) - x.saturating_sub(r);
            *t = (s / count as f64) as f32;
        }
    }
    // Vertical pass: tmp -> out, the rows of the window added to one sum per column, top to bottom.
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), (y + r + 1).min(h));
        sums.fill(0.0);
        for ty in lo..hi {
            for (s, &v) in sums.iter_mut().zip(&tmp[ty * w..(ty + 1) * w]) {
                *s += f64::from(v);
            }
        }
        let inv = 1.0 / (hi - lo) as f64;
        for (o, &s) in out[y * w..(y + 1) * w].iter_mut().zip(&sums) {
            *o = (s * inv) as f32;
        }
    }
}

/// Copies a `w x h` plane into a `(w + 2r) x (h + 2r)` plane with mirrored borders
/// ([`mirror`](super::mirror)), so a window of radius `r` never needs a bounds check.
pub fn pad_mirror(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let pw = w + 2 * r;
    let mut out = vec![0.0f32; pw * (h + 2 * r)];
    for py in 0..h + 2 * r {
        let sy = super::mirror(py as isize - r as isize, h);
        let row = &src[sy * w..(sy + 1) * w];
        let orow = &mut out[py * pw..(py + 1) * pw];
        for (px, o) in orow.iter_mut().enumerate() {
            *o = row[super::mirror(px as isize - r as isize, w)];
        }
    }
    out
}

/// Normalised Gaussian kernel of half-width `radius` (taps `-radius..=radius`).
pub fn gaussian_kernel(sigma: f32, radius: usize) -> Vec<f32> {
    let mut k: Vec<f32> = (0..=2 * radius)
        .map(|i| {
            let d = i as f32 - radius as f32;
            (-0.5 * d * d / (sigma * sigma)).exp()
        })
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_without_gaps() {
        let b = bands(1000, 128);
        assert_eq!(b.first().unwrap().start, 0);
        assert_eq!(b.last().unwrap().end, 1000);
        assert!(b.windows(2).all(|w| w[0].end == w[1].start));
        assert_eq!(extended_rows(&(130..200), 5, 1000), 124..205);
        assert_eq!(extended_rows(&(0..64), 5, 1000), 0..69);
        assert_eq!(extended_rows(&(960..1000), 5, 1000), 954..1000);
    }

    #[test]
    fn box_mean_matches_direct_computation() {
        let (w, h) = (13, 9);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32 * 0.1).collect();
        for r in [0usize, 1, 2, 5, 20] {
            let mut out = vec![0.0; w * h];
            let mut tmp = vec![0.0; w * h];
            box_mean(&src, w, h, r, &mut out, &mut tmp);
            for y in 0..h {
                for x in 0..w {
                    let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
                    let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
                    let mut s = 0.0f64;
                    let mut n = 0.0f64;
                    for yy in y0..=y1 {
                        for xx in x0..=x1 {
                            s += f64::from(src[yy * w + xx]);
                            n += 1.0;
                        }
                    }
                    assert!(
                        (f64::from(out[y * w + x]) - s / n).abs() < 1e-6,
                        "r={r} ({x},{y})"
                    );
                }
            }
        }
    }

    /// A plane that spans many powers of two, as the squares of a bright light on a dark ground do: a window
    /// sum of it is not exact in `f64`, so a running sum keeps a rounding residue that depends on where the
    /// sum began.
    fn wide_plane(w: usize, h: usize) -> Vec<f32> {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) as f64 / u64::MAX as f64
        };
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if (x / 5 + y / 5) % 4 == 1 {
                    (1.0 + next()) as f32 // a light
                } else {
                    (1e-9 * (1.0 + 1000.0 * next())) as f32 // the dark ground
                }
            })
            .collect()
    }

    /// `docs/17-gpu.md` D4 for the CPU reference: the mean at a pixel is the sum of its own window, so a crop of
    /// the plane gives the whole plane's bits wherever the crop holds the window (`r` pixels of context), and
    /// the crop's own edge is the window's only where it is the plane's.
    #[test]
    fn box_mean_of_a_crop_is_the_whole_planes_bit_for_bit() {
        let (w, h) = (97, 83);
        let src = wide_plane(w, h);
        for r in [1usize, 2, 5, 8] {
            let mut whole = vec![0.0f32; w * h];
            let mut tmp = vec![0.0f32; w * h];
            box_mean(&src, w, h, r, &mut whole, &mut tmp);
            for (x0, y0, cw, ch) in [
                (13, 9, 40, 37),
                (30, 0, 67, 83),
                (0, 41, 97, 42),
                (50, 50, 47, 33),
            ] {
                let crop: Vec<f32> = (y0..y0 + ch)
                    .flat_map(|y| src[y * w + x0..y * w + x0 + cw].iter().copied())
                    .collect();
                let mut out = vec![0.0f32; cw * ch];
                let mut tmp = vec![0.0f32; cw * ch];
                box_mean(&crop, cw, ch, r, &mut out, &mut tmp);
                for y in 0..ch {
                    for x in 0..cw {
                        let inside = (x >= r || x0 == 0)
                            && (x + r < cw || x0 + cw == w)
                            && (y >= r || y0 == 0)
                            && (y + r < ch || y0 + ch == h);
                        if inside {
                            assert_eq!(
                                out[y * cw + x].to_bits(),
                                whole[(y0 + y) * w + x0 + x].to_bits(),
                                "r={r} crop ({x0},{y0}) pixel ({x},{y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pad_mirror_matches_mirror_indexing() {
        let (w, h, r) = (5usize, 4usize, 3usize);
        let src: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let p = pad_mirror(&src, w, h, r);
        let pw = w + 2 * r;
        for y in 0..h + 2 * r {
            for x in 0..pw {
                let (sy, sx) = (
                    super::super::mirror(y as isize - r as isize, h),
                    super::super::mirror(x as isize - r as isize, w),
                );
                assert_eq!(p[y * pw + x], src[sy * w + sx]);
            }
        }
    }

    #[test]
    fn gaussian_kernel_sums_to_one_and_is_symmetric() {
        let k = gaussian_kernel(1.3, 4);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for i in 0..4 {
            assert!((k[i] - k[8 - i]).abs() < 1e-7);
        }
    }
}
