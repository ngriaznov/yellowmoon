//! Small `f64` grey-image helpers for the classical proposer and detector: separable Gaussian,
//! square-window minimum and maximum (erosion, dilation, opening, closing), a disc dilation of
//! masks and 8-connected components. Rows run in parallel and write disjoint rows, so results never
//! depend on the thread count. Windows and kernels replicate (clamp) at the borders.

use rayon::prelude::*;

/// A grey image of `f64`, row-major.
#[derive(Debug, Clone)]
pub(crate) struct Grey {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f64>,
}

impl Grey {
    pub fn new(w: usize, h: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), w * h);
        Self { w, h, data }
    }

    fn transposed(&self) -> Self {
        let mut out = vec![0.0; self.w * self.h];
        // Output row `x` is input column `x`; rows in parallel.
        out.par_chunks_mut(self.h.max(1))
            .enumerate()
            .for_each(|(x, row)| {
                for (y, v) in row.iter_mut().enumerate() {
                    *v = self.data[y * self.w + x];
                }
            });
        Self::new(self.h, self.w, out)
    }

    /// Applies a per-row operation along x, then along y (through a transpose).
    fn separable(&self, op: impl Fn(&[f64], &mut [f64]) + Sync) -> Self {
        let pass = |src: &Self| {
            let mut out = vec![0.0; src.w * src.h];
            out.par_chunks_mut(src.w.max(1))
                .zip(src.data.par_chunks(src.w.max(1)))
                .for_each(|(dst, line)| op(line, dst));
            Self::new(src.w, src.h, out)
        };
        pass(&pass(self).transposed()).transposed()
    }

    /// Gaussian blur with standard deviation `sigma_px` (kernel radius `ceil(3 sigma)`), edges
    /// replicated. `sigma_px <= 0` returns a copy.
    pub fn gaussian(&self, sigma_px: f64) -> Self {
        if sigma_px.is_nan() || sigma_px <= 0.0 || sigma_px.is_infinite() {
            return self.clone();
        }
        let r = (3.0 * sigma_px).ceil() as usize;
        let mut kernel: Vec<f64> = (0..=2 * r)
            .map(|i| {
                let d = i as f64 - r as f64;
                (-0.5 * d * d / (sigma_px * sigma_px)).exp()
            })
            .collect();
        let sum: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= sum);
        self.separable(|line, dst| {
            let n = line.len() as isize;
            for (x, d) in dst.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (i, k) in kernel.iter().enumerate() {
                    let xx = (x as isize + i as isize - r as isize).clamp(0, n - 1);
                    acc += k * line[xx as usize];
                }
                *d = acc;
            }
        })
    }

    /// Maximum over a square window of side `2 r + 1` (grey dilation); the window shrinks at the
    /// borders, which equals replicating the edge.
    pub fn dilate(&self, r: usize) -> Self {
        self.extreme(r, true)
    }

    /// Minimum over a square window of side `2 r + 1` (grey erosion).
    pub fn erode(&self, r: usize) -> Self {
        self.extreme(r, false)
    }

    fn extreme(&self, r: usize, max: bool) -> Self {
        if r == 0 {
            return self.clone();
        }
        let pick = move |a: f64, b: f64| if max { a.max(b) } else { a.min(b) };
        self.separable(move |line, dst| {
            // van Herk / Gil-Werman on the line padded by replication: block prefix and suffix
            // extremes, then one comparison per output, whatever the window.
            let n = line.len();
            let k = 2 * r + 1;
            let padded: Vec<f64> = (0..n + 2 * r)
                .map(|i| line[i.saturating_sub(r).min(n - 1)])
                .collect();
            let m = padded.len();
            let mut pre = padded.clone();
            let mut suf = padded.clone();
            for i in 0..m {
                if i % k != 0 {
                    pre[i] = pick(pre[i - 1], padded[i]);
                }
            }
            for i in (0..m).rev() {
                if i % k != k - 1 && i + 1 < m {
                    suf[i] = pick(suf[i + 1], padded[i]);
                }
            }
            for (x, d) in dst.iter_mut().enumerate() {
                *d = pick(suf[x], pre[x + k - 1]);
            }
        })
    }

    /// Grey closing: dilation then erosion. It fills dark features narrower than the window.
    pub fn closing(&self, r: usize) -> Self {
        self.dilate(r).erode(r)
    }

    /// Grey opening: erosion then dilation. It removes bright features narrower than the window.
    pub fn opening(&self, r: usize) -> Self {
        self.erode(r).dilate(r)
    }
}

/// One 8-connected component of a boolean mask.
#[derive(Debug, Clone)]
pub(crate) struct Component {
    /// Pixel indices (`y * w + x`) in discovery order.
    pub pixels: Vec<usize>,
    /// Bounding box, inclusive: `(x0, y0, x1, y1)`.
    pub bbox: (usize, usize, usize, usize),
}

impl Component {
    /// Width of the bounding box in pixels.
    pub fn width(&self) -> usize {
        self.bbox.2 - self.bbox.0 + 1
    }

    /// Height of the bounding box in pixels.
    pub fn height(&self) -> usize {
        self.bbox.3 - self.bbox.1 + 1
    }
}

/// The 8-connected components of `mask`, in raster order of their first pixel.
pub(crate) fn components(mask: &[bool], w: usize, h: usize) -> Vec<Component> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut pixels = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        while let Some(p) = stack.pop() {
            pixels.push(p);
            let (x, y) = (p % w, p / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let q = ny * w + nx;
                    if mask[q] && !seen[q] {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        out.push(Component {
            pixels,
            bbox: (x0, y0, x1, y1),
        });
    }
    out
}

/// Dilates a mask by a disc of radius `r` pixels (pixel centres within `r`).
pub(crate) fn dilate_disc(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    let ri = r as isize;
    let offsets: Vec<(isize, isize)> = (-ri..=ri)
        .flat_map(|dy| (-ri..=ri).map(move |dx| (dx, dy)))
        .filter(|&(dx, dy)| dx * dx + dy * dy <= ri * ri)
        .collect();
    let mut out = vec![false; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = offsets.iter().any(|&(dx, dy)| {
                let (xx, yy) = (x as isize + dx, y as isize + dy);
                xx >= 0
                    && yy >= 0
                    && (xx as usize) < w
                    && (yy as usize) < h
                    && mask[yy as usize * w + xx as usize]
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_keeps_constants_and_sums() {
        let g = Grey::new(9, 7, vec![0.25; 63]);
        assert!(
            g.gaussian(1.7)
                .data
                .iter()
                .all(|v| (v - 0.25).abs() < 1e-12)
        );
        let mut d = vec![0.0; 21 * 21];
        d[10 * 21 + 10] = 1.0;
        let b = Grey::new(21, 21, d).gaussian(1.5);
        assert!((b.data.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(
            Grey::new(2, 2, vec![1.0; 4]).gaussian(0.0).data,
            vec![1.0; 4]
        );
    }

    #[test]
    fn closing_fills_narrow_dark_and_opening_removes_narrow_bright() {
        let mut d = vec![1.0; 15 * 15];
        d[7 * 15 + 7] = 0.0;
        d[7 * 15 + 8] = 0.0;
        let g = Grey::new(15, 15, d);
        assert!(g.closing(2).data.iter().all(|&v| v == 1.0));
        let inv = Grey::new(15, 15, g.data.iter().map(|v| 1.0 - v).collect());
        assert!(inv.opening(2).data.iter().all(|&v| v == 0.0));
        // A wide dark block survives.
        let mut wide = vec![1.0; 15 * 15];
        for y in 4..11 {
            for x in 4..11 {
                wide[y * 15 + x] = 0.0;
            }
        }
        assert_eq!(Grey::new(15, 15, wide).closing(2).data[7 * 15 + 7], 0.0);
    }

    #[test]
    fn extremes_match_brute_force_with_edges() {
        let (w, h) = (13usize, 9usize);
        let data: Vec<f64> = (0..w * h).map(|i| ((i * 7919) % 101) as f64).collect();
        let g = Grey::new(w, h, data.clone());
        for r in [1usize, 2, 5, 20] {
            let (mx, mn) = (g.dilate(r), g.erode(r));
            for y in 0..h {
                for x in 0..w {
                    let mut win = Vec::new();
                    for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                        for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                            win.push(data[yy * w + xx]);
                        }
                    }
                    let hi = win.iter().copied().fold(f64::MIN, f64::max);
                    let lo = win.iter().copied().fold(f64::MAX, f64::min);
                    assert_eq!(
                        (mx.data[y * w + x], mn.data[y * w + x]),
                        (hi, lo),
                        "r {r} ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn components_and_disc() {
        let (w, h) = (8, 6);
        let mut m = vec![false; w * h];
        for &(x, y) in &[(1, 1), (2, 2), (6, 4), (6, 5)] {
            m[y * w + x] = true;
        }
        let c = components(&m, w, h);
        assert_eq!(c.len(), 2, "diagonal pixels join (8-connectivity)");
        assert_eq!((c[0].width(), c[0].height()), (2, 2));
        assert_eq!(c[1].bbox, (6, 4, 6, 5));
        let d = dilate_disc(&m, w, h, 1);
        assert!(d[w + 2] && d[3 * w + 2] && !d[3 * w + 3]);
        assert!(m.iter().zip(&d).all(|(a, b)| !a | b));
    }
}
