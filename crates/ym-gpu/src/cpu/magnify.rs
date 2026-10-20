//! The magnification of a view above 100 % (`docs/17-gpu.md` section 16): the CPU reference of the
//! `magnify` kernel and of `ym-core`'s magnified view.
//!
//! A view magnified above 100 % shows the picture at 100 % made bigger. The colour chain and output
//! sharpening run on the 100 % picture under the view (the *source* here: `nw x nh` pixels of the
//! final image, from `origin`), and each device pixel of the view is a Catmull-Rom sample of that
//! picture ([`super::resample::sample_catmull_rom`]: a = -0.5, pixel `i` centred at `i + 0.5`, the
//! edge pixel replicated, nothing clamped). A device pixel whose centre falls outside the final image,
//! or on a source pixel outside the straightened photo (the source's mask), is the paper.
//!
//! # Taps
//!
//! Everything about a device pixel except the three sums is a function of the f64 position of its
//! centre in the final image, one axis at a time: whether the centre is inside the image along that
//! axis, the source column (row) whose mask sample decides the paper, the four columns (rows) the
//! filter reads with the edge replicated, and their four `f32` weights. [`AxisTap::new`] computes
//! them as the per-pixel code always did. A view whose map from device pixels to the final image is a
//! scale and a translation (every viewport) has one tap per column and one per row
//! ([`MagnifyMap`]): that is what the kernel reads, so the device computes no position, no floor and
//! no weight, and its only arithmetic is the sum, in the order of [`sample_taps`].
//!
//! [`magnify_with`] is the general form (any map, the taps made per pixel) that `ym-core` runs on the
//! CPU; [`magnify`] is the same over a [`MagnifyMap`]. For a separable map the two are the same
//! function of the same positions, so they agree bit for bit (`tests` below, and against the
//! per-pixel [`super::resample::sample_catmull_rom`] the view used before).

use rayon::prelude::*;

use super::resample::catmull_rom_weights;
use super::{CpuError, RgbImage, clamp_index};
use crate::queue::CancelToken;

/// What one column (or one row) of a magnified view reads (module notes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisTap {
    /// The four source columns (rows) of the Catmull-Rom filter, the edge pixel replicated.
    pub idx: [u32; 4],
    /// Their weights, [`catmull_rom_weights`] of the fraction of the position.
    pub weights: [f32; 4],
    /// The source column (row) whose mask sample says whether the device pixel is paper: the pixel
    /// under the centre, clamped to the source.
    pub mask_at: u32,
    /// Whether the centre lies inside the final image along this axis.
    pub inside: bool,
}

impl AxisTap {
    /// The tap of a device pixel whose centre is at `pos` (pixels of the final image, which is `len`
    /// pixels long on this axis) over a source that starts at pixel `origin` of the final image and is
    /// `src_len` pixels long (at least 1).
    #[must_use]
    pub fn new(pos: f64, len: usize, origin: usize, src_len: usize) -> Self {
        let inside = pos >= 0.0 && pos < len as f64;
        let p = pos - origin as f64;
        let mask_at = (p.max(0.0) as usize).min(src_len.saturating_sub(1));
        // `sample_catmull_rom`: u = p - 0.5, i0 = floor(u), t = u - i0 rounded to f32.
        let u = p - 0.5;
        let fl = u.floor();
        let i0 = fl as isize;
        let weights = catmull_rom_weights((u - fl) as f32);
        let idx = std::array::from_fn(|k| clamp_index(i0 - 1 + k as isize, src_len) as u32);
        Self {
            idx,
            weights,
            mask_at: mask_at as u32,
            inside,
        }
    }
}

/// The taps of a view whose map is a scale and a translation: one per column and one per row
/// (module notes).
#[derive(Clone, Debug, PartialEq)]
pub struct MagnifyMap {
    /// One tap per column of the view.
    pub cols: Vec<AxisTap>,
    /// One tap per row of the view.
    pub rows: Vec<AxisTap>,
    /// The size of the source picture the taps read.
    pub src: (usize, usize),
}

impl MagnifyMap {
    /// The taps of a `cols.len() x rows.len()` view whose column `x` has its centre at `cols[x]` and
    /// row `y` at `rows[y]` (pixels of the final image of `final_size`), over a source of `src`
    /// pixels that starts at `origin` of the final image.
    ///
    /// # Errors
    ///
    /// [`CpuError::Mismatch`] for an empty view or source.
    pub fn new(
        cols: &[f64],
        rows: &[f64],
        final_size: (usize, usize),
        origin: (usize, usize),
        src: (usize, usize),
    ) -> Result<Self, CpuError> {
        if cols.is_empty() || rows.is_empty() || src.0 == 0 || src.1 == 0 {
            return Err(CpuError::Mismatch(format!(
                "a magnified view of {} x {} pixels over a {} x {} source",
                cols.len(),
                rows.len(),
                src.0,
                src.1
            )));
        }
        Ok(Self {
            cols: cols
                .iter()
                .map(|&p| AxisTap::new(p, final_size.0, origin.0, src.0))
                .collect(),
            rows: rows
                .iter()
                .map(|&p| AxisTap::new(p, final_size.1, origin.1, src.1))
                .collect(),
            src,
        })
    }

    /// Width of the view.
    #[must_use]
    pub fn width(&self) -> usize {
        self.cols.len()
    }

    /// Height of the view.
    #[must_use]
    pub fn height(&self) -> usize {
        self.rows.len()
    }
}

/// Whether the device pixel of taps `cx` and `ry` is paper: its centre is outside the final image, or
/// the source pixel under it is outside the photo (`outside`, one byte per source pixel, 1 outside).
#[must_use]
pub fn is_paper(outside: Option<&[u8]>, src_width: usize, cx: &AxisTap, ry: &AxisTap) -> bool {
    !(cx.inside && ry.inside)
        || outside.is_some_and(|o| o[ry.mask_at as usize * src_width + cx.mask_at as usize] != 0)
}

/// The Catmull-Rom sample of one plane (`src_width` samples a row) at the taps `cx` and `ry`:
/// `acc += wy_k (wx_0 r_0 + wx_1 r_1 + wx_2 r_2 + wx_3 r_3)` row by row from `acc = 0`, left to
/// right, in `f32`: the operations of [`super::resample::sample_catmull_rom`], in its order.
#[must_use]
pub fn sample_taps(plane: &[f32], src_width: usize, cx: &AxisTap, ry: &AxisTap) -> f32 {
    let [x0, x1, x2, x3] = cx.idx.map(|i| i as usize);
    let wx = &cx.weights;
    let mut acc = 0.0f32;
    for (k, &wyk) in ry.weights.iter().enumerate() {
        let row = &plane[ry.idx[k] as usize * src_width..][..src_width];
        let r = wx[0] * row[x0] + wx[1] * row[x1] + wx[2] * row[x2] + wx[3] * row[x3];
        acc += wyk * r;
    }
    acc
}

fn check_source(src: &RgbImage, outside: Option<&[u8]>) -> Result<(), CpuError> {
    let n = src.width * src.height;
    if src.data.len() != 3 * n {
        return Err(CpuError::BadLength {
            got: src.data.len(),
            expected: 3 * n,
            width: src.width,
            height: src.height,
        });
    }
    if let Some(o) = outside
        && o.len() != n
    {
        return Err(CpuError::BadLength {
            got: o.len(),
            expected: n,
            width: src.width,
            height: src.height,
        });
    }
    if n == 0 {
        return Err(CpuError::Mismatch("an empty source picture".into()));
    }
    Ok(())
}

/// The magnified view: interleaved display-linear RGB (`3 w h`, zero on the paper) and the paper mask
/// (`w h`, 1 on the paper), what `ym-core` encodes. `taps(x, y)` gives the taps of device pixel
/// `(x, y)`. Rows run side by side on the rayon pool; a cancelled `cancel` stops the rows that have
/// not started (the caller looks at it afterwards).
fn magnify_by(
    src: &RgbImage,
    outside: Option<&[u8]>,
    (w, h): (usize, usize),
    cancel: Option<&CancelToken>,
    taps: impl Fn(usize, usize) -> (AxisTap, AxisTap) + Sync,
) -> (Vec<f32>, Vec<u8>) {
    let nw = src.width;
    let planes = [src.plane(0), src.plane(1), src.plane(2)];
    let mut view = vec![0.0f32; 3 * w * h];
    let mut paper = vec![0u8; w * h];
    view.par_chunks_mut(3 * w.max(1))
        .zip(paper.par_chunks_mut(w.max(1)))
        .enumerate()
        .for_each(|(y, (row, mask))| {
            if cancel.is_some_and(CancelToken::is_cancelled) {
                return;
            }
            for x in 0..w {
                let (cx, ry) = taps(x, y);
                if is_paper(outside, nw, &cx, &ry) {
                    mask[x] = 1;
                    continue;
                }
                for (c, plane) in planes.iter().enumerate() {
                    row[3 * x + c] = sample_taps(plane, nw, &cx, &ry);
                }
            }
        });
    (view, paper)
}

/// The CPU reference of a magnified view (module notes) for any map: device pixel `(x, y)` has its
/// centre at `pos(x, y)` in the final image of `final_size`; the source `src` (planar RGB, with its
/// optional mask `outside`) is the 100 % picture from `origin`. Returns the interleaved RGB of the
/// `w x h` view (zero on the paper) and its paper mask (1 on the paper).
///
/// # Errors
///
/// [`CpuError::BadLength`] when the source or the mask has the wrong length, [`CpuError::Mismatch`] for
/// an empty source.
pub fn magnify_with(
    src: &RgbImage,
    outside: Option<&[u8]>,
    view: (usize, usize),
    final_size: (usize, usize),
    origin: (usize, usize),
    cancel: Option<&CancelToken>,
    pos: impl Fn(usize, usize) -> (f64, f64) + Sync,
) -> Result<(Vec<f32>, Vec<u8>), CpuError> {
    check_source(src, outside)?;
    let (nw, nh) = (src.width, src.height);
    Ok(magnify_by(src, outside, view, cancel, |x, y| {
        let (fx, fy) = pos(x, y);
        (
            AxisTap::new(fx, final_size.0, origin.0, nw),
            AxisTap::new(fy, final_size.1, origin.1, nh),
        )
    }))
}

/// [`magnify_with`] over the taps of a [`MagnifyMap`]: what the kernel computes.
///
/// # Errors
///
/// As [`magnify_with`]; [`CpuError::Mismatch`] when the map was made for a source of another size.
pub fn magnify(
    src: &RgbImage,
    outside: Option<&[u8]>,
    map: &MagnifyMap,
) -> Result<(Vec<f32>, Vec<u8>), CpuError> {
    check_source(src, outside)?;
    if map.src != (src.width, src.height) {
        return Err(CpuError::Mismatch(format!(
            "the taps were made for a {} x {} source, this one is {} x {}",
            map.src.0, map.src.1, src.width, src.height
        )));
    }
    Ok(magnify_by(
        src,
        outside,
        (map.width(), map.height()),
        None,
        |x, y| (map.cols[x], map.rows[y]),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::resample::sample_catmull_rom;
    use super::*;

    /// A source with texture, values of both signs and a few above 1.
    fn source(w: usize, h: usize, seed: u32) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for v in &mut img.data {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = (s % 20_000) as f32 / 16_000.0 - 0.05;
        }
        img
    }

    /// The per-pixel code `ym-core`'s magnified view ran before the taps (its `magnify_and_encode`):
    /// the inside test, the mask lookup and `sample_catmull_rom` per channel.
    #[allow(clippy::too_many_arguments)] // the old code's own inputs
    fn per_pixel(
        src: &RgbImage,
        outside: Option<&[u8]>,
        (w, h): (usize, usize),
        (fw, fh): (usize, usize),
        (ox, oy): (usize, usize),
        pos: impl Fn(usize, usize) -> (f64, f64),
    ) -> (Vec<f32>, Vec<u8>) {
        let (nw, nh) = (src.width, src.height);
        let mut view = vec![0.0f32; 3 * w * h];
        let mut letterbox = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = pos(x, y);
                let inside = fx >= 0.0 && fy >= 0.0 && fx < fw as f64 && fy < fh as f64;
                let (px, py) = (fx - ox as f64, fy - oy as f64);
                let empty = !inside
                    || outside.is_some_and(|o| {
                        let (ix, iy) = (
                            (px.max(0.0) as usize).min(nw - 1),
                            (py.max(0.0) as usize).min(nh - 1),
                        );
                        o[iy * nw + ix] != 0
                    });
                if empty {
                    letterbox[y * w + x] = 1;
                    continue;
                }
                for c in 0..3 {
                    view[3 * (y * w + x) + c] = sample_catmull_rom(src.plane(c), nw, nh, px, py);
                }
            }
        }
        (view, letterbox)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn the_taps_are_the_per_pixel_code_bit_for_bit() {
        // (source size, origin, final size, view size, zoom, view origin in the final image): views
        // that start before the image and run past it (paper), sources at the image's edge and inside
        // it, odd zooms, a one-pixel-wide source.
        let cases = [
            ((40, 30), (0, 0), (40, 30), (90, 70), 2.0, (-3.25, -1.5)),
            ((40, 30), (10, 5), (80, 60), (64, 48), 4.0, (12.125, 7.0)),
            ((33, 21), (2, 3), (60, 40), (101, 77), 3.37, (5.0, 4.4)),
            ((17, 19), (0, 0), (17, 19), (50, 70), 1.013, (-0.3, 0.2)),
            ((1, 9), (4, 0), (5, 9), (12, 30), 7.9, (3.9, -0.6)),
            ((64, 48), (0, 0), (64, 48), (130, 98), 2.0, (0.0, 0.0)),
        ];
        for (i, &(src_size, origin, final_size, view, zoom, start)) in cases.iter().enumerate() {
            let src = source(src_size.0, src_size.1, i as u32);
            // A mask with holes: every fifth pixel outside.
            let mask: Vec<u8> = (0..src_size.0 * src_size.1)
                .map(|j| u8::from(j % 5 == 2))
                .collect();
            let pos = |x: usize, y: usize| {
                (
                    (x as f64 + 0.5) / zoom + start.0,
                    (y as f64 + 0.5) / zoom + start.1,
                )
            };
            let cols: Vec<f64> = (0..view.0).map(|x| pos(x, 0).0).collect();
            let rows: Vec<f64> = (0..view.1).map(|y| pos(0, y).1).collect();
            let map = MagnifyMap::new(&cols, &rows, final_size, origin, src_size).unwrap();
            for outside in [None, Some(mask.as_slice())] {
                let old = per_pixel(&src, outside, view, final_size, origin, pos);
                let general =
                    magnify_with(&src, outside, view, final_size, origin, None, pos).unwrap();
                let tapped = magnify(&src, outside, &map).unwrap();
                assert_eq!(bits(&general.0), bits(&old.0), "case {i}: the values");
                assert_eq!(general.1, old.1, "case {i}: the paper");
                assert_eq!(bits(&tapped.0), bits(&old.0), "case {i}: the map's values");
                assert_eq!(tapped.1, old.1, "case {i}: the map's paper");
                assert!(old.1.contains(&0), "case {i}: some picture");
            }
        }
    }

    #[test]
    fn a_tap_replicates_the_edge_and_says_where_the_centre_is() {
        // Centre of pixel 0 of a source at the image's edge: u = 0, i0 = 0, t = 0.
        let t = AxisTap::new(0.5, 10, 0, 10);
        assert_eq!(t.idx, [0, 0, 1, 2]);
        assert_eq!(t.weights, [0.0, 1.0, 0.0, 0.0]);
        assert!(t.inside);
        assert_eq!(t.mask_at, 0);
        // Past the end: the last pixel replicated, outside the image.
        let t = AxisTap::new(10.2, 10, 0, 10);
        assert_eq!(t.idx, [8, 9, 9, 9]);
        assert!(!t.inside);
        assert_eq!(t.mask_at, 9);
        // Before the start, with a source that starts at pixel 3 of the image.
        let t = AxisTap::new(1.0, 10, 3, 4);
        assert_eq!(t.idx, [0, 0, 0, 0]);
        assert!(t.inside);
        assert_eq!(t.mask_at, 0);
        // Not a number: never inside.
        assert!(!AxisTap::new(f64::NAN, 10, 0, 10).inside);
    }

    #[test]
    fn bad_inputs_are_refused() {
        let src = source(4, 3, 1);
        assert!(MagnifyMap::new(&[], &[0.5], (4, 3), (0, 0), (4, 3)).is_err());
        assert!(MagnifyMap::new(&[0.5], &[0.5], (4, 3), (0, 0), (0, 3)).is_err());
        let map = MagnifyMap::new(&[0.5], &[0.5], (4, 3), (0, 0), (5, 3)).unwrap();
        assert!(magnify(&src, None, &map).is_err(), "another source size");
        assert!(magnify(&src, Some(&[0; 5]), &map).is_err(), "a short mask");
    }
}
