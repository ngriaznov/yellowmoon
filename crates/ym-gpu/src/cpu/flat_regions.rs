//! The flat-region detector and the per-channel noise profile of `docs/algorithms/05-single-frame-denoise.md`
//! (sections B/C.3 and B/C.4), shared by methods B and C and by appendix D, with the 2 × 2 box pyramid they run
//! on (B/C.2 step 2, reading BC-a).
//!
//! All arithmetic is `f64` on planes of `f64` (reading G1). Everything here is sequential and deterministic.
//!
//! # Pyramid
//!
//! * Depth ([`pyramid_depth`]): `n = max(0, trunc(log2(min(W, H) / base)))`, computed exactly in integers as the
//!   largest `n` with `base · 2ⁿ ≤ min(W, H)` (0 when `min(W, H) < base`).
//! * [`downsample`]: a `w × h` plane gives `ceil(w / 2) × ceil(h / 2)`; each sample is the mean of a 2 × 2 block,
//!   summed as `((top-left + top-right) + (bottom-left + bottom-right)) / 4`. A block that crosses the right or
//!   bottom edge of an odd-sized plane repeats the last column or row (the replicated edge).
//! * [`upsample`]: bilinear with pixel centres at half-integers. Fine pixel `x` reads the coarse coordinate
//!   `u = (x + 0.5) / 2 − 0.5` (the factor is exactly 2, the box pyramid's, whatever the rounding of the size),
//!   so its weights are 3/4 and 1/4; indices are clamped to the coarse plane. The output has the exact size
//!   asked for (the finer level's).
//!
//! # Detector (B/C.3, reading BC-b)
//!
//! [`detect`] reads the RGB planes of one level (the coarsest): `g = 0.299 R + 0.587 G + 0.114 B`,
//! `m = max(R, G, B)`; the box sizes `b`, `Wv` come from that level's width (the table of B/C.3);
//! `v = max(0, box_b(g²) − box_b(g)²)` (the clamp only removes rounding below zero, so the `2 · best` rule
//! below is never fed a negative best), `V = box_Wv(v)`. Box filters are `k × k` means whose taps mirror at the
//! edges without repeating the edge sample ([`super::mirror`], the house default), so every output averages
//! exactly `k` samples per axis. Cells have side `c = floor(1.5 · Wv)` and are tiled from the origin; a partial
//! cell at the right or bottom is not used. A cell's candidate is the first minimum of `V` in raster order over
//! the cell inset by `floor(c / 4)` on each side; its score is that minimum, plus 1000 when the cell's centre pixel
//! (`origin + floor(c / 2)`) has `m > 220/255` or `g < 30/255`. Candidates are sorted by score (ties: the cell's
//! raster index); the first `clamp(#{score ≤ 2 · best}, 2, 10)` are kept (fewer only when fewer cells exist). A
//! region is the `c × c` square `[x − floor(c / 2), x − floor(c / 2) + c)` around the candidate (same for `y`),
//! clipped to the level, stored normalised by the level's size.
//!
//! # Profile (B/C.4, reading BC-c)
//!
//! [`profile`] maps each region onto a level by rounding (`round(x · width)`, [`Region::to_pixels`]). For a region
//! of `rw × rh` pixels, `a = min(rw, rh)`, `d = max(1, floor(a / 5))`, `G = ceil(a / d)`; `μ_r` is the channel's
//! mean over the whole `rw × rh` region; the 2 × 2 patches (vector order `(0,0), (1,0), (0,1), (1,1)`) have corners
//! `origin + (i d, j d)`, `i, j < G`, each coordinate clipped to the last corner where the patch fits
//! (`x1 − 2`); `C_r = Σ (p − μ_r)(p − μ_r)ᵀ / (G² − 1)`. A region with `a < 2` (no patch fits, or `G = 1`) is
//! skipped. `P` is the mean of `C_r` and `m_flat` the mean of `μ_r` over the regions not skipped; no such region
//! gives `None`. [`regularize`] then applies step 4.
//!
//! # Regularisation (B/C.4 step 4, `n = 4`)
//!
//! In order, each step reading the matrix the step before produced: `d̄ = mean(diag P)`; `P[1][1]` and `P[2][2]`
//! become the mean of their four matrix neighbours, `P[0][0] = P[0][1]`, `P[3][3] = P[3][2]`; two median passes
//! over the 5-point cross clipped to the matrix (5 values inside, 4 on an edge, 3 at a corner), each pass computed
//! from the matrix before it; the median of an even count is the mean of the two middle values; Toeplitz
//! averaging of offsets 0 and 1, where the offset-`k` diagonal is the `2 (4 − k)` entries `(i, i + k)` and
//! `(i + k, i)` (both triangles together, so the matrix stays exactly symmetric) set to their mean; last, the
//! main diagonal set to `d̄`.

use super::{Rect, RgbImage, mirror};

/// Score added to a cell whose centre is near clipping or too dark.
const PENALTY: f64 = 1000.0;
/// At most this many regions are kept.
const MAX_REGIONS: usize = 10;
/// At least this many regions are kept (when the level has that many cells).
const MIN_REGIONS: usize = 2;

/// A flat region: a rectangle in coordinates normalised by the size of the level it was found on, and the score
/// of its cell (lower is flatter).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    /// Left edge, in `[0, 1]`.
    pub x0: f64,
    /// Top edge, in `[0, 1]`.
    pub y0: f64,
    /// Right edge (exclusive), in `[0, 1]`.
    pub x1: f64,
    /// Bottom edge (exclusive), in `[0, 1]`.
    pub y1: f64,
    /// The cell's score: the minimum local variance, plus 1000 for a clipped or dark cell centre.
    pub score: f64,
}

impl Region {
    /// The region on a grid of `width × height` pixels: each edge is `round(edge · size)` (half away from zero),
    /// clipped to the grid.
    pub fn to_pixels(&self, width: usize, height: usize) -> Rect {
        let map = |t: f64, n: usize| ((t * n as f64).round().max(0.0) as usize).min(n);
        let (x0, x1) = (map(self.x0, width), map(self.x1, width));
        let (y0, y1) = (map(self.y0, height), map(self.y1, height));
        Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }
}

/// A channel's regularised noise profile at one pyramid level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    /// The regularised 4 × 4 patch covariance `P` (patch order `(0,0), (1,0), (0,1), (1,1)`).
    pub p: [[f64; 4]; 4],
    /// The mean of the channel over the flat regions, `m_flat`.
    pub m_flat: f64,
}

/// The pyramid depth `n = max(0, trunc(log2(min(width, height) / base)))`, exactly: the largest `n` with
/// `base · 2ⁿ ≤ min(width, height)`.
pub fn pyramid_depth(width: usize, height: usize, base: usize) -> usize {
    let side = width.min(height);
    let (mut n, mut size) = (0usize, base.max(1));
    while let Some(next) = size.checked_mul(2).filter(|&s| s <= side) {
        (n, size) = (n + 1, next);
    }
    n
}

/// One step down the pyramid: the 2 × 2 box mean of a `width × height` plane, of size
/// `ceil(width / 2) × ceil(height / 2)`, the last column or row of an odd size repeating the edge.
/// Returns the plane and its size.
pub fn downsample(plane: &[f64], width: usize, height: usize) -> (Vec<f64>, usize, usize) {
    debug_assert_eq!(plane.len(), width * height);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let mut out = Vec::with_capacity(cw * ch);
    for j in 0..ch {
        let r0 = &plane[2 * j * width..(2 * j + 1) * width];
        let y1 = (2 * j + 1).min(height - 1);
        let r1 = &plane[y1 * width..(y1 + 1) * width];
        for i in 0..cw {
            let (x0, x1) = (2 * i, (2 * i + 1).min(width - 1));
            out.push(((r0[x0] + r0[x1]) + (r1[x0] + r1[x1])) * 0.25);
        }
    }
    (out, cw, ch)
}

/// Bilinear 2× upsampling of a `width × height` plane to `out_width × out_height` (the finer level's size),
/// with half-pixel centres and clamped edges (see the module documentation).
pub fn upsample(
    plane: &[f64],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f64> {
    debug_assert_eq!(plane.len(), width * height);
    if width == 0 || height == 0 {
        return vec![0.0; out_width * out_height];
    }
    // Per output coordinate: the two source indices and the weight of the second.
    let taps = |n_out: usize, n_src: usize| -> Vec<(usize, usize, f64)> {
        (0..n_out)
            .map(|x| {
                let u = (x as f64 + 0.5) * 0.5 - 0.5;
                let i0 = u.floor();
                let f = u - i0;
                let i0 = i0 as isize;
                let clamp = |i: isize| i.clamp(0, n_src as isize - 1) as usize;
                (clamp(i0), clamp(i0 + 1), f)
            })
            .collect()
    };
    let tx = taps(out_width, width);
    let ty = taps(out_height, height);
    // Horizontal pass: width x height -> out_width x height.
    let mut tmp = Vec::with_capacity(out_width * height);
    for row in plane.chunks_exact(width) {
        tmp.extend(tx.iter().map(|&(a, b, f)| row[a] * (1.0 - f) + row[b] * f));
    }
    // Vertical pass.
    let mut out = Vec::with_capacity(out_width * out_height);
    for &(a, b, f) in &ty {
        let (ra, rb) = (
            &tmp[a * out_width..(a + 1) * out_width],
            &tmp[b * out_width..(b + 1) * out_width],
        );
        out.extend(ra.iter().zip(rb).map(|(&p, &q)| p * (1.0 - f) + q * f));
    }
    out
}

/// The RGB planes of `img` taken `depth` steps down the pyramid, with their size.
pub fn coarsest_rgb(img: &RgbImage, depth: usize) -> ([Vec<f64>; 3], usize, usize) {
    let mut planes = super::ycbcr::rgb_planes(img);
    let (mut w, mut h) = (img.width, img.height);
    for _ in 0..depth {
        if w == 0 || h == 0 {
            break;
        }
        let mut size = (w, h);
        planes = planes.map(|p| {
            let (q, cw, ch) = downsample(&p, w, h);
            size = (cw, ch);
            q
        });
        (w, h) = size;
    }
    (planes, w, h)
}

/// Runs the detector on the coarsest level of a pyramid of base `base` built from `img` (read as a whole
/// frame). Returns the pyramid depth and the regions.
pub fn detect_frame(img: &RgbImage, base: usize) -> (usize, Vec<Region>) {
    let depth = pyramid_depth(img.width, img.height, base);
    let (planes, w, h) = coarsest_rgb(img, depth);
    (depth, detect(&planes, w, h))
}

/// The variance box size `b` and smoothing size `Wv` for a level of width `width` (B/C.3).
fn box_sizes(width: usize) -> (usize, usize) {
    match width {
        0..256 => (3, 9),
        256..512 => (5, 9),
        512..1024 => (7, 15),
        _ => (9, 25),
    }
}

/// `size × size` mean (`size` odd) with mirrored taps, separable (horizontal, then vertical). Every output sums
/// exactly `size` samples per axis, in tap order.
fn box_mirror(src: &[f64], width: usize, height: usize, size: usize) -> Vec<f64> {
    let r = (size / 2) as isize;
    let inv = 1.0 / size as f64;
    let mut tmp = vec![0.0; width * height];
    for (row, trow) in src.chunks_exact(width).zip(tmp.chunks_exact_mut(width)) {
        for (x, t) in trow.iter_mut().enumerate() {
            let mut s = 0.0;
            for d in -r..=r {
                s += row[mirror(x as isize + d, width)];
            }
            *t = s * inv;
        }
    }
    let mut out = vec![0.0; width * height];
    for (y, orow) in out.chunks_exact_mut(width).enumerate() {
        for (x, o) in orow.iter_mut().enumerate() {
            let mut s = 0.0;
            for d in -r..=r {
                s += tmp[mirror(y as isize + d, height) * width + x];
            }
            *o = s * inv;
        }
    }
    out
}

/// The flat-region detector (B/C.3) on the RGB planes of one level of `width × height`. Returns 0 to 10 regions
/// sorted by score (none only when the level holds no whole cell).
pub fn detect(rgb: &[Vec<f64>; 3], width: usize, height: usize) -> Vec<Region> {
    let n = width * height;
    if n == 0 || rgb.iter().any(|p| p.len() != n) {
        return Vec::new();
    }
    let (b, wv) = box_sizes(width);
    let c = wv * 3 / 2;
    let (cols, rows) = (width / c, height / c);
    if cols == 0 || rows == 0 {
        return Vec::new();
    }
    let [red, green, blue] = rgb;
    let g: Vec<f64> = (0..n)
        .map(|i| 0.299 * red[i] + 0.587 * green[i] + 0.114 * blue[i])
        .collect();
    let g2: Vec<f64> = g.iter().map(|v| v * v).collect();
    let mean = box_mirror(&g, width, height, b);
    let mean2 = box_mirror(&g2, width, height, b);
    let v: Vec<f64> = mean2
        .iter()
        .zip(&mean)
        .map(|(m2, m)| (m2 - m * m).max(0.0))
        .collect();
    let big_v = box_mirror(&v, width, height, wv);

    let q = c / 4;
    // (score, cell raster index, x and y of the minimum)
    let mut cands: Vec<(f64, usize, usize, usize)> = Vec::with_capacity(cols * rows);
    for cy in 0..rows {
        for cx in 0..cols {
            let (ox, oy) = (cx * c, cy * c);
            let (mut bx, mut by) = (ox + q, oy + q);
            let mut best = big_v[by * width + bx];
            for y in oy + q..oy + c - q {
                for x in ox + q..ox + c - q {
                    let val = big_v[y * width + x];
                    if val < best {
                        (best, bx, by) = (val, x, y);
                    }
                }
            }
            let centre = (oy + c / 2) * width + ox + c / 2;
            let m = red[centre].max(green[centre]).max(blue[centre]);
            let penalised = m > 220.0 / 255.0 || g[centre] < 30.0 / 255.0;
            let score = best + if penalised { PENALTY } else { 0.0 };
            cands.push((score, cy * cols + cx, bx, by));
        }
    }
    cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let best = cands[0].0;
    let within = cands.iter().take_while(|cand| cand.0 <= 2.0 * best).count();
    let keep = within.clamp(MIN_REGIONS, MAX_REGIONS).min(cands.len());

    let half = (c / 2) as isize;
    let span = |m: usize, size: usize| {
        let lo = (m as isize - half).max(0) as usize;
        let hi = ((m as isize - half + c as isize).max(0) as usize).min(size);
        (lo as f64 / size as f64, hi as f64 / size as f64)
    };
    cands[..keep]
        .iter()
        .map(|&(score, _, bx, by)| {
            let (x0, x1) = span(bx, width);
            let (y0, y1) = span(by, height);
            Region {
                x0,
                y0,
                x1,
                y1,
                score,
            }
        })
        .collect()
}

/// The noise profile (B/C.4) of one channel plane of `width × height` over `regions`, regularised; `None` when no
/// region is usable at this size.
pub fn profile(plane: &[f64], width: usize, height: usize, regions: &[Region]) -> Option<Profile> {
    if plane.len() != width * height {
        return None;
    }
    let mut sum_c = [[0.0f64; 4]; 4];
    let mut sum_mu = 0.0f64;
    let mut used = 0usize;
    for region in regions {
        let rect = region.to_pixels(width, height);
        let a = rect.w.min(rect.h);
        if a < 2 {
            continue;
        }
        let d = (a / 5).max(1);
        let grid = a.div_ceil(d);
        if grid * grid <= 1 {
            continue;
        }
        let mut s = 0.0f64;
        for y in rect.y..rect.y1() {
            s += plane[y * width + rect.x..y * width + rect.x1()]
                .iter()
                .sum::<f64>();
        }
        let mu = s / (rect.w * rect.h) as f64;
        let mut c = [[0.0f64; 4]; 4];
        for j in 0..grid {
            let py = (rect.y + j * d).min(rect.y1() - 2);
            for i in 0..grid {
                let px = (rect.x + i * d).min(rect.x1() - 2);
                let k = py * width + px;
                let p = [
                    plane[k] - mu,
                    plane[k + 1] - mu,
                    plane[k + width] - mu,
                    plane[k + width + 1] - mu,
                ];
                for (row, &pi) in c.iter_mut().zip(&p) {
                    for (e, &pj) in row.iter_mut().zip(&p) {
                        *e += pi * pj;
                    }
                }
            }
        }
        let norm = 1.0 / (grid * grid - 1) as f64;
        for (srow, crow) in sum_c.iter_mut().zip(&c) {
            for (s, &v) in srow.iter_mut().zip(crow) {
                *s += v * norm;
            }
        }
        sum_mu += mu;
        used += 1;
    }
    if used == 0 {
        return None;
    }
    let inv = 1.0 / used as f64;
    let p = sum_c.map(|row| row.map(|v| v * inv));
    Some(Profile {
        p: regularize(p),
        m_flat: sum_mu * inv,
    })
}

/// One median pass over the 5-point cross clipped to the 4 × 4 matrix.
fn median_pass(p: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (i, orow) in out.iter_mut().enumerate() {
        for (j, o) in orow.iter_mut().enumerate() {
            let mut vals = [p[i][j]; 5];
            let mut n = 1;
            let neighbours = [
                (i > 0).then(|| p[i - 1][j]),
                (i < 3).then(|| p[i + 1][j]),
                (j > 0).then(|| p[i][j - 1]),
                (j < 3).then(|| p[i][j + 1]),
            ];
            for v in neighbours.into_iter().flatten() {
                vals[n] = v;
                n += 1;
            }
            let v = &mut vals[..n];
            v.sort_by(f64::total_cmp);
            *o = if n % 2 == 1 {
                v[n / 2]
            } else {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            };
        }
    }
    out
}

/// The regularisation of B/C.4 step 4 (`n = 4`; the module documentation states the order and the open
/// points).
pub fn regularize(p: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let d_bar = (p[0][0] + p[1][1] + p[2][2] + p[3][3]) / 4.0;
    let mut q = p;
    for k in 1..3 {
        q[k][k] = (p[k - 1][k] + p[k + 1][k] + p[k][k - 1] + p[k][k + 1]) / 4.0;
    }
    q[0][0] = p[0][1];
    q[3][3] = p[3][2];
    let mut q = median_pass(&median_pass(&q));
    for offset in 0..2 {
        let mut s = 0.0;
        let mut count = 0usize;
        for i in 0..4 - offset {
            s += q[i][i + offset];
            count += 1;
            if offset > 0 {
                s += q[i + offset][i];
                count += 1;
            }
        }
        let mean = s / count as f64;
        for i in 0..4 - offset {
            q[i][i + offset] = mean;
            q[i + offset][i] = mean;
        }
    }
    for (k, row) in q.iter_mut().enumerate() {
        row[k] = d_bar;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth::Rng;

    /// `[x0, x1, y0, y1)` in pixels.
    type Span = (usize, usize, usize, usize);

    fn grey(w: usize, h: usize, v: f64) -> [Vec<f64>; 3] {
        std::array::from_fn(|_| vec![v; w * h])
    }

    /// Fills `[x0, x1) × [y0, y1)` of every plane with `v`.
    fn fill(planes: &mut [Vec<f64>; 3], w: usize, rect: Span, v: f64) {
        for p in planes.iter_mut() {
            for y in rect.2..rect.3 {
                p[y * w + rect.0..y * w + rect.1].fill(v);
            }
        }
    }

    /// High-contrast texture: every pixel an independent grey in [0.2, 0.8).
    fn texture(w: usize, h: usize, seed: u64) -> [Vec<f64>; 3] {
        let mut rng = Rng::new(seed);
        let g: Vec<f64> = (0..w * h)
            .map(|_| 0.2 + 0.6 * f64::from(rng.next_f32()))
            .collect();
        [g.clone(), g.clone(), g]
    }

    fn centre_in(r: &Region, w: usize, h: usize, rect: Span) -> bool {
        let cx = 0.5 * (r.x0 + r.x1) * w as f64;
        let cy = 0.5 * (r.y0 + r.y1) * h as f64;
        cx >= rect.0 as f64 && cx < rect.1 as f64 && cy >= rect.2 as f64 && cy < rect.3 as f64
    }

    #[test]
    fn depth_is_the_truncated_log2_of_the_short_side_over_the_base() {
        assert_eq!(pyramid_depth(100, 1000, 128), 0);
        assert_eq!(pyramid_depth(255, 255, 128), 0);
        assert_eq!(pyramid_depth(256, 300, 128), 1);
        assert_eq!(pyramid_depth(6000, 4000, 128), 4); // log2(31.25) = 4.97
        assert_eq!(pyramid_depth(6000, 4000, 256), 3);
        assert_eq!(pyramid_depth(2048, 2048, 256), 3);
        assert_eq!(pyramid_depth(2047, 2048, 256), 2);
        assert_eq!(pyramid_depth(0, 10, 128), 0);
    }

    #[test]
    fn downsample_takes_ceil_sizes_and_box_means_with_the_edge_repeated() {
        // 3 x 2: [1 2 4 / 3 6 8] -> 2 x 1: [(1+2+3+6)/4, (4+4+8+8)/4].
        let (out, w, h) = downsample(&[1.0, 2.0, 4.0, 3.0, 6.0, 8.0], 3, 2);
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, vec![3.0, 6.0]);
        // 3 x 1: the single row repeats too.
        let (out, w, h) = downsample(&[1.0, 2.0, 4.0], 3, 1);
        assert_eq!((w, h, out), (2, 1, vec![1.5, 4.0]));
        // Sizes down a pyramid of odd sizes.
        let (mut w, mut h) = (7usize, 5usize);
        let mut p = vec![0.5; w * h];
        let mut sizes = vec![];
        for _ in 0..3 {
            let (q, cw, ch) = downsample(&p, w, h);
            (p, w, h) = (q, cw, ch);
            sizes.push((w, h));
        }
        assert_eq!(sizes, vec![(4, 3), (2, 2), (1, 1)]);
        assert_eq!(p, vec![0.5]);
    }

    #[test]
    fn upsample_is_bilinear_with_half_pixel_centres_to_an_odd_size() {
        // 2 x 1 -> 3 x 1: x = 0 reads u = -0.25 (clamped: the first sample), x = 1 reads 0.25, x = 2 reads 0.75.
        let out = upsample(&[1.0, 5.0], 2, 1, 3, 1);
        assert_eq!(out, vec![1.0, 0.75 + 0.25 * 5.0, 0.25 + 0.75 * 5.0]);
        // 2 x 2 -> 4 x 3, separable: rows read v = -0.25 (clamped), 0.25 and 0.75.
        let out = upsample(&[0.0, 4.0, 8.0, 12.0], 2, 2, 4, 3);
        assert_eq!(out.len(), 12);
        assert_eq!(&out[0..4], &[0.0, 1.0, 3.0, 4.0]);
        assert_eq!(&out[4..8], &[2.0, 3.0, 5.0, 6.0]);
        assert_eq!(&out[8..12], &[6.0, 7.0, 9.0, 10.0]);
        // A constant stays constant.
        assert!(upsample(&[0.3; 6], 3, 2, 5, 3).iter().all(|&v| v == 0.3));
    }

    #[test]
    fn regularize_matches_the_hand_computed_matrix() {
        let p = [
            [4.0, 2.0, 1.0, 0.0],
            [2.0, 6.0, 3.0, 1.0],
            [1.0, 3.0, 8.0, 2.0],
            [0.0, 1.0, 2.0, 2.0],
        ];
        // d̄ = (4 + 6 + 8 + 2) / 4 = 5.
        // Diagonal step (from P): P11 = (2 + 3 + 2 + 3) / 4 = 2.5, P22 = (3 + 2 + 3 + 2) / 4 = 2.5,
        // P00 = P01 = 2, P33 = P32 = 2:
        //   [2   2   1   0 ]
        //   [2   2.5 3   1 ]
        //   [1   3   2.5 2 ]
        //   [0   1   2   2 ]
        // Median pass 1, e.g. (0,2): {1, 2, 0, 3} -> (1 + 2) / 2 = 1.5; (1,2): {3, 1, 2.5, 2.5, 1} -> 2.5;
        // (0,3): {0, 1, 1} -> 1:
        //   [2   2   1.5 1  ]
        //   [2   2.5 2.5 1.5]
        //   [1.5 2.5 2.5 2  ]
        //   [1   1.5 2   2  ]
        // Median pass 2, e.g. (0,2): {1.5, 2, 1, 2.5} -> 1.75; (0,3): {1, 1.5, 1.5} -> 1.5;
        // (1,3): {1.5, 1, 2, 2.5} -> 1.75; (2,3): {2, 1.5, 2, 2.5} -> 2:
        //   [2    2    1.75 1.5 ]
        //   [2    2.5  2.5  1.75]
        //   [1.75 2.5  2.5  2   ]
        //   [1.5  1.75 2    2   ]
        // Toeplitz: offset 1 = (2 + 2.5 + 2) * 2 / 6 = 13/6; offsets 2 and 3 stay (1.75, 1.5);
        // last, the diagonal = d̄ = 5.
        let t = 13.0 / 6.0;
        let want = [
            [5.0, t, 1.75, 1.5],
            [t, 5.0, t, 1.75],
            [1.75, t, 5.0, t],
            [1.5, 1.75, t, 5.0],
        ];
        let got = regularize(p);
        for i in 0..4 {
            for j in 0..4 {
                assert!((got[i][j] - want[i][j]).abs() < 1e-12, "({i},{j}): {got:?}");
            }
        }
    }

    #[test]
    fn profile_of_white_noise_on_flat_grey_has_its_variance_on_the_diagonal() {
        let (w, h) = (256usize, 256usize);
        let sigma = 0.02;
        let mut rng = Rng::new(17);
        let plane: Vec<f64> = (0..w * h)
            .map(|_| 0.5 + sigma * f64::from(rng.gaussian()))
            .collect();
        // Ten regions of 52 x 52 (d = 10, G = 6), as the detector gives two levels up.
        let regions: Vec<Region> = (0..10)
            .map(|k| {
                let (x, y) = (8 + 60 * (k % 4), 8 + 60 * (k / 4));
                Region {
                    x0: x as f64 / w as f64,
                    y0: y as f64 / h as f64,
                    x1: (x + 52) as f64 / w as f64,
                    y1: (y + 52) as f64 / h as f64,
                    score: 0.0,
                }
            })
            .collect();
        let prof = profile(&plane, w, h, &regions).expect("usable regions");
        let var = sigma * sigma;
        for i in 0..4 {
            assert!(
                (prof.p[i][i] / var - 1.0).abs() < 0.15,
                "diag {i}: {} vs {var}",
                prof.p[i][i]
            );
            for j in 0..4 {
                if i != j {
                    assert!(
                        prof.p[i][j].abs() < 0.15 * var,
                        "({i},{j}) {}",
                        prof.p[i][j]
                    );
                }
            }
        }
        assert!((prof.m_flat - 0.5).abs() < 0.002, "{}", prof.m_flat);
        // A region of one pixel is not usable; nothing usable gives None.
        let tiny = Region {
            x0: 0.0,
            y0: 0.0,
            x1: 1.0 / w as f64,
            y1: 1.0 / h as f64,
            score: 0.0,
        };
        assert_eq!(profile(&plane, w, h, &[tiny]), None);
    }

    #[test]
    fn detector_finds_the_two_flat_patches_in_texture() {
        let (w, h) = (130usize, 130usize); // b = 3, Wv = 9, c = 13: 10 x 10 cells
        let mut rgb = texture(w, h, 5);
        let a = (13, 52, 13, 52);
        let b = (78, 117, 65, 104);
        fill(&mut rgb, w, a, 0.5);
        fill(&mut rgb, w, b, 0.5);
        let regions = detect(&rgb, w, h);
        assert!((2..=10).contains(&regions.len()), "{}", regions.len());
        assert!(regions.windows(2).all(|p| p[0].score <= p[1].score));
        for r in &regions {
            assert!(centre_in(r, w, h, a) || centre_in(r, w, h, b), "{r:?}");
            // 13 x 13 squares (none clipped here).
            let px = r.to_pixels(w, h);
            assert_eq!((px.w, px.h), (13, 13));
        }
        assert!(regions.iter().any(|r| centre_in(r, w, h, a)));
        assert!(regions.iter().any(|r| centre_in(r, w, h, b)));
    }

    #[test]
    fn detector_avoids_clipped_and_dark_cells() {
        let (w, h) = (130usize, 130usize);
        let bright = (13, 52, 13, 52);
        let dark = (65, 117, 13, 52);
        let good = (39, 78, 78, 117);
        let mut rgb = texture(w, h, 9);
        fill(&mut rgb, w, bright, 0.95);
        fill(&mut rgb, w, dark, 0.05);
        fill(&mut rgb, w, good, 0.5);
        let regions = detect(&rgb, w, h);
        assert!(!regions.is_empty());
        for r in &regions {
            assert!(centre_in(r, w, h, good), "{r:?}");
            assert!(r.score < 1.0);
        }
        // Without the good patch, texture wins over the flat but clipped or dark patches.
        let mut rgb = texture(w, h, 9);
        fill(&mut rgb, w, bright, 0.95);
        fill(&mut rgb, w, dark, 0.05);
        let regions = detect(&rgb, w, h);
        assert!(regions.len() >= 2);
        for r in &regions {
            assert!(
                !centre_in(r, w, h, bright) && !centre_in(r, w, h, dark),
                "{r:?}"
            );
        }
    }

    #[test]
    fn detector_keeps_at_least_two_and_at_most_ten() {
        let (w, h) = (130usize, 130usize);
        // All cells equal: ten kept.
        let regions = detect(&grey(w, h, 0.5), w, h);
        assert_eq!(regions.len(), 10);
        // One flat cell in texture: still two.
        let mut rgb = texture(w, h, 3);
        fill(&mut rgb, w, (52, 65, 52, 65), 0.5);
        let regions = detect(&rgb, w, h);
        assert_eq!(regions.len(), 2);
        // A level smaller than one cell has none.
        assert!(detect(&grey(12, 40, 0.5), 12, 40).is_empty());
        // One cell only: one region.
        assert_eq!(detect(&grey(13, 20, 0.5), 13, 20).len(), 1);
    }
}
