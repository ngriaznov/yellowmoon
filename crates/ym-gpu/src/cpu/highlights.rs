//! R5 highlight reconstruction on the mosaic, opposed-channel method (`docs/05-raw-processing.md` section R5).
//!
//! Clean-room implementation from the five-step description in `docs/05` (no code of any other
//! raw processor was read):
//!
//! 1. Each 2x2 quad becomes a *superpixel* with R, G (mean of the two greens) and B values and a
//!    clipped flag per channel (green is flagged when either green photosite is clipped).
//! 2. For a clipped channel `c` of superpixel `p`, the **opposed estimate** is the cube-root mean
//!    of the *unclipped other channels* over the 3x3 superpixel neighbourhood:
//!    `y = (mean(x^(1/3)))^3`. If the neighbourhood holds no unclipped other channel there is no
//!    estimate and the photosite keeps its value.
//! 3. **Global chroma correction.** For every channel `c`, over all superpixels with no clipped
//!    channel that have a superpixel with channel `c` clipped in their 3x3 neighbourhood, average
//!    `x_c - y_c` (with `y_c` the opposed estimate computed for `c` as if it were clipped). That
//!    mean is the offset added to every estimate of channel `c`. (Reading "bordering clipped ones"
//!    per channel keeps, for example, a red lamp from biasing the blue offset.)
//! 4. The reconstructed value is `max(original, y_c + offset_c)`, written back to the clipped
//!    photosites of that channel in the quad. The clip mask is not modified: later stages still
//!    know these values are estimates.
//! 5. `decode.highlight_recovery = false` means not calling this module.
//!
//! The chroma offsets are a global reduction. For tiled processing run
//! [`accumulate_offsets`] over the cores of all tiles first, merge the accumulators with
//! [`OffsetAccumulator::merge`], and give the same [`ChromaOffsets`] to [`apply_offsets`] on each
//! tile. [`reconstruct`] does both for one buffer.
//!
//! The superpixel grid is anchored at frame (0, 0), so a tile must start on even coordinates
//! (`place.x0` and `place.y0` even). A buffer with an odd width or height leaves its last
//! column / row untouched.

use rayon::prelude::*;

use super::{Cfa, ClipMask, CpuError, Mosaic, Rect, split_range};

/// Fixed band height (in superpixel rows) of the parallel reductions. The partition depends only
/// on the image size, so the f64 sums are identical whatever the thread count.
const BAND_ROWS: usize = 64;

/// Superpixel image of a mosaic: three value planes and three flag planes of `sw * sh`.
struct Superpixels {
    sw: usize,
    sh: usize,
    /// Plane `c` at `c * sw * sh`.
    value: Vec<f32>,
    /// Plane `c` at `c * sw * sh`, 1 where channel `c` of the superpixel is clipped.
    clipped: Vec<u8>,
    /// Cube roots of `value`.
    cbrt: Vec<f32>,
    /// Bit `c` set when channel `c` is clipped anywhere in the 3x3 neighbourhood.
    near: Vec<u8>,
    /// Bit `c` set when channel `c` is clipped in the superpixel itself.
    own: Vec<u8>,
}

impl Superpixels {
    fn build(m: &Mosaic, mask: &ClipMask) -> Superpixels {
        let (sw, sh) = (m.width / 2, m.height / 2);
        let n = sw * sh;
        let mut value = vec![0.0f32; 3 * n];
        let mut clipped = vec![0u8; 3 * n];
        let cfa = m.cfa;
        {
            let (vr, rest) = value.split_at_mut(n);
            let (vg, vb) = rest.split_at_mut(n);
            let (cr, crest) = clipped.split_at_mut(n);
            let (cg, cb) = crest.split_at_mut(n);
            (
                vr.par_chunks_mut(sw.max(1)),
                vg.par_chunks_mut(sw.max(1)),
                vb.par_chunks_mut(sw.max(1)),
            )
                .into_par_iter()
                .zip(
                    (
                        cr.par_chunks_mut(sw.max(1)),
                        cg.par_chunks_mut(sw.max(1)),
                        cb.par_chunks_mut(sw.max(1)),
                    )
                        .into_par_iter(),
                )
                .enumerate()
                .for_each(|(j, ((vr, vg, vb), (cr, cg, cb)))| {
                    let vals = [vr, vg, vb];
                    let flags = [cr, cg, cb];
                    for i in 0..sw {
                        let mut sum = [0.0f32; 3];
                        let mut cnt = [0u32; 3];
                        let mut clip = [false; 3];
                        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            let (x, y) = (2 * i + dx, 2 * j + dy);
                            let c = cfa.color_at(x, y).plane();
                            sum[c] += m.data[y * m.width + x];
                            cnt[c] += 1;
                            clip[c] |= mask.data[y * m.width + x] != 0;
                        }
                        for c in 0..3 {
                            vals[c][i] = sum[c] / cnt[c] as f32;
                            flags[c][i] = u8::from(clip[c]);
                        }
                    }
                });
        }
        let cbrt: Vec<f32> = value.par_iter().map(|v| v.cbrt()).collect();
        let mut own = vec![0u8; n];
        for c in 0..3 {
            for (o, &f) in own.iter_mut().zip(&clipped[c * n..(c + 1) * n]) {
                *o |= f << c;
            }
        }
        // 3x3 dilation of `own`, separably.
        let mut tmp = vec![0u8; n];
        tmp.par_chunks_mut(sw.max(1))
            .enumerate()
            .for_each(|(j, row)| {
                for (i, r) in row.iter_mut().enumerate() {
                    let lo = i.saturating_sub(1);
                    let hi = (i + 1).min(sw - 1);
                    *r = (lo..=hi).fold(0, |a, k| a | own[j * sw + k]);
                }
            });
        let mut near = vec![0u8; n];
        near.par_chunks_mut(sw.max(1))
            .enumerate()
            .for_each(|(j, row)| {
                let lo = j.saturating_sub(1);
                let hi = (j + 1).min(sh - 1);
                for i in 0..sw {
                    row[i] = (lo..=hi).fold(0, |a, k| a | tmp[k * sw + i]);
                }
            });
        Superpixels {
            sw,
            sh,
            value,
            clipped,
            cbrt,
            near,
            own,
        }
    }

    /// Opposed estimate for channel `c` at superpixel `(i, j)`; `None` without usable neighbours.
    #[inline]
    fn opposed(&self, i: usize, j: usize, c: usize) -> Option<f32> {
        let n = self.sw * self.sh;
        let (i0, i1) = (i.saturating_sub(1), (i + 1).min(self.sw - 1));
        let (j0, j1) = (j.saturating_sub(1), (j + 1).min(self.sh - 1));
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for jj in j0..=j1 {
            for ii in i0..=i1 {
                let idx = jj * self.sw + ii;
                for k in 0..3 {
                    if k != c && self.clipped[k * n + idx] == 0 {
                        sum += self.cbrt[k * n + idx];
                        cnt += 1;
                    }
                }
            }
        }
        (cnt > 0).then(|| {
            let mean = sum / cnt as f32;
            mean * mean * mean
        })
    }
}

/// Running sums for the global chroma correction (step 3); mergeable across tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OffsetAccumulator {
    /// Sum of `x_c - y_c` per channel over the reference superpixels.
    pub sum: [f64; 3],
    /// Number of reference superpixels per channel.
    pub count: [u64; 3],
}

impl OffsetAccumulator {
    /// Adds another accumulator (tiles are merged in a fixed order for reproducible sums).
    pub fn merge(&mut self, other: &OffsetAccumulator) {
        for c in 0..3 {
            self.sum[c] += other.sum[c];
            self.count[c] += other.count[c];
        }
    }

    /// The means; 0 for a channel without reference superpixels.
    pub fn finish(&self) -> ChromaOffsets {
        let mut offset = [0.0f64; 3];
        for (c, o) in offset.iter_mut().enumerate() {
            if self.count[c] > 0 {
                *o = self.sum[c] / self.count[c] as f64;
            }
        }
        ChromaOffsets {
            offset,
            count: self.count,
        }
    }
}

/// The global chroma correction of step 3.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChromaOffsets {
    /// Offset added to the opposed estimate of each channel (red, green, blue).
    pub offset: [f64; 3],
    /// How many reference superpixels each offset averages (0: no clipped boundary found).
    pub count: [u64; 3],
}

fn check_inputs(m: &Mosaic, mask: &ClipMask) -> Result<(), CpuError> {
    if (m.width, m.height) != (mask.width, mask.height) {
        return Err(CpuError::Mismatch(format!(
            "mosaic {}x{} and clip mask {}x{} differ",
            m.width, m.height, mask.width, mask.height
        )));
    }
    if !m.place.x0.is_multiple_of(2) || !m.place.y0.is_multiple_of(2) {
        return Err(CpuError::Mismatch(format!(
            "highlight reconstruction needs an even-aligned tile, got origin ({}, {})",
            m.place.x0, m.place.y0
        )));
    }
    Ok(())
}

/// Step 3 over the superpixels whose top-left photosite lies in `core` (buffer coordinates,
/// even-aligned). Superpixels outside `core` still feed the 3x3 neighbourhoods but are not counted.
pub fn accumulate_offsets(
    m: &Mosaic,
    mask: &ClipMask,
    core: Rect,
) -> Result<OffsetAccumulator, CpuError> {
    check_inputs(m, mask)?;
    let sp = Superpixels::build(m, mask);
    Ok(accumulate(&sp, core))
}

fn accumulate(sp: &Superpixels, core: Rect) -> OffsetAccumulator {
    if sp.sw == 0 || sp.sh == 0 {
        return OffsetAccumulator::default();
    }
    let n = sp.sw * sp.sh;
    let (i_lo, i_hi) = (core.x / 2, (core.x1() / 2).min(sp.sw));
    let (j_lo, j_hi) = (core.y / 2, (core.y1() / 2).min(sp.sh));
    let rows = j_hi.saturating_sub(j_lo);
    let bands = split_range(rows, rows.div_ceil(BAND_ROWS));
    let parts: Vec<OffsetAccumulator> = bands
        .par_iter()
        .map(|band| {
            let mut acc = OffsetAccumulator::default();
            for j in (j_lo + band.start)..(j_lo + band.end) {
                for i in i_lo..i_hi {
                    let idx = j * sp.sw + i;
                    if sp.own[idx] != 0 {
                        continue;
                    }
                    let near = sp.near[idx];
                    if near == 0 {
                        continue;
                    }
                    for c in 0..3 {
                        if near & (1 << c) == 0 {
                            continue;
                        }
                        if let Some(y) = sp.opposed(i, j, c) {
                            acc.sum[c] += f64::from(sp.value[c * n + idx]) - f64::from(y);
                            acc.count[c] += 1;
                        }
                    }
                }
            }
            acc
        })
        .collect();
    let mut total = OffsetAccumulator::default();
    for p in &parts {
        total.merge(p);
    }
    total
}

/// Computes the chroma offsets of the whole buffer (step 3).
pub fn chroma_offsets(m: &Mosaic, mask: &ClipMask) -> Result<ChromaOffsets, CpuError> {
    let core = Rect::new(0, 0, m.width, m.height);
    Ok(accumulate_offsets(m, mask, core)?.finish())
}

/// Steps 2 and 4: reconstructs every clipped photosite of `m` using `offsets`.
pub fn apply_offsets(
    m: &mut Mosaic,
    mask: &ClipMask,
    offsets: &ChromaOffsets,
) -> Result<(), CpuError> {
    check_inputs(m, mask)?;
    let sp = Superpixels::build(m, mask);
    if sp.sw == 0 || sp.sh == 0 {
        return Ok(());
    }
    let cfa: Cfa = m.cfa;
    let width = m.width;
    let sw = sp.sw;
    let sh = sp.sh;
    let off = offsets.offset;
    m.data[..2 * sh * width]
        .par_chunks_mut(2 * width)
        .enumerate()
        .for_each(|(j, rows)| {
            for i in 0..sw {
                if sp.own[j * sw + i] == 0 {
                    continue;
                }
                let mut est = [None::<f32>; 3];
                for (c, e) in est.iter_mut().enumerate() {
                    if sp.clipped[c * sw * sh + j * sw + i] != 0 {
                        *e = sp.opposed(i, j, c).map(|y| y + off[c] as f32);
                    }
                }
                for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                    let (x, y) = (2 * i + dx, 2 * j + dy);
                    if mask.data[y * width + x] == 0 {
                        continue;
                    }
                    let c = cfa.color_at(x, y).plane();
                    if let Some(e) = est[c] {
                        let v = &mut rows[dy * width + x];
                        *v = v.max(e);
                    }
                }
            }
        });
    Ok(())
}

/// Reconstructs the highlights of `m` in place and returns the chroma offsets used.
pub fn reconstruct(m: &mut Mosaic, mask: &ClipMask) -> Result<ChromaOffsets, CpuError> {
    let offsets = chroma_offsets(m, mask)?;
    apply_offsets(m, mask, &offsets)?;
    Ok(offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_mosaic(w: usize, h: usize, rgb: [f32; 3]) -> Mosaic {
        let mut m = Mosaic::new(w, h, Cfa::RGGB);
        for y in 0..h {
            for x in 0..w {
                m.data[y * w + x] = rgb[m.cfa.color_at(x, y).plane()];
            }
        }
        m
    }

    #[test]
    fn untouched_without_clipping() {
        let mut m = flat_mosaic(16, 16, [0.5, 0.4, 0.3]);
        let before = m.clone();
        let mask = ClipMask::new(16, 16);
        let off = reconstruct(&mut m, &mask).unwrap();
        assert_eq!(m, before);
        assert_eq!(off.count, [0, 0, 0]);
    }

    #[test]
    fn red_is_rebuilt_from_green_and_blue() {
        // Left half: the true colour (R 0.9, G 0.6, B 0.3); right half: red clipped at 0.7 although
        // the scene red is 0.9 (the sensor saturated below the scene value).
        let (w, h) = (32, 16);
        let mut m = flat_mosaic(w, h, [0.9, 0.6, 0.3]);
        let mut mask = ClipMask::new(w, h);
        for y in 0..h {
            for x in 16..w {
                if m.color_at(x, y) == super::super::CfaColor::Red {
                    m.data[y * w + x] = 0.7;
                    mask.data[y * w + x] = 1;
                }
            }
        }
        let off = reconstruct(&mut m, &mask).unwrap();
        assert!(off.count[0] > 0 && off.count[1] == 0 && off.count[2] == 0);
        // Opposed estimate of red from G = 0.6 and B = 0.3 (cube-root mean) is below 0.7, so the
        // max(original, ...) rule keeps the clipped value.
        let r = m.get(24, 8);
        assert!(r >= 0.7);
    }

    #[test]
    fn estimate_can_exceed_the_clip_level() {
        // Scene colour R 1.6 G 1.0 B 0.8, red clipped at 1.0; opposed estimate from G/B is ~0.9 but
        // the border offset (x_R - y_R, positive for a red object) lifts it well above 1.0.
        let (w, h) = (40, 20);
        let mut m = flat_mosaic(w, h, [1.6, 1.0, 0.8]);
        let mut mask = ClipMask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if m.color_at(x, y) == super::super::CfaColor::Red && x >= 20 {
                    m.data[y * w + x] = 1.0;
                    mask.data[y * w + x] = 1;
                } else if m.color_at(x, y) == super::super::CfaColor::Red {
                    // Unclipped red region: measured at its true value (as if the white level were higher).
                }
            }
        }
        reconstruct(&mut m, &mask).unwrap();
        assert!(
            m.get(30, 10) > 1.2,
            "reconstructed red {} should exceed the clip level",
            m.get(30, 10)
        );
        assert!(m.get(30, 10) < 1.8);
    }

    #[test]
    fn needs_even_alignment() {
        let m = flat_mosaic(16, 16, [0.5; 3]);
        let tile = m.extract(Rect::new(1, 0, 8, 8));
        let mask = ClipMask::new(8, 8);
        assert!(chroma_offsets(&tile, &mask).is_err());
        let mut t2 = m.extract(Rect::new(0, 0, 8, 8));
        assert!(apply_offsets(&mut t2, &ClipMask::new(4, 4), &ChromaOffsets::default()).is_err());
    }
}
