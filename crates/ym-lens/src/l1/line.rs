//! Box averages along the rows or the columns of a picture in constant time per pixel, bit for bit the same
//! whatever rectangle of the frame they are computed on (`docs/21` sections 4 and 9).
//!
//! The stretch averages, at every pixel, the picture along a line of half-length `l` (up to a hundred pixels at
//! full resolution). When every pixel of that line passes the rule "its blur is at least half the pixel's own",
//! the average is the exact mean of the piecewise-linear picture over the line, which a table of running
//! integrals gives in a few reads. The running integrals are kept in fixed point (64-bit integers): a sum of
//! integers does not depend on where the running sum started, so a tile and the whole frame read the same
//! integers and get the same average. Whether the rule holds along the line is read from a sparse table of
//! range minima of the blur.

use rayon::prelude::*;

/// Fractional bits of the fixed-point picture.
const SHIFT: u32 = 26;
/// Values beyond this magnitude saturate (scene-linear values are far below it).
const LIMIT: f32 = 65536.0;

/// A value in fixed point (NaN reads as 0).
#[inline(always)]
fn quant(v: f32) -> i64 {
    if v.is_nan() {
        return 0;
    }
    (f64::from(v.clamp(-LIMIT, LIMIT)) * f64::from(1u32 << SHIFT)).round() as i64
}

/// The running integrals of a planar picture and the range minima of its blur map, along one axis.
pub(super) struct Lines<'a> {
    w: usize,
    h: usize,
    /// True when the lines run along `y` (the columns), false when along `x` (the rows).
    along_y: bool,
    /// The frame coordinate of position 0 along the lines.
    origin: usize,
    picture: &'a [f32],
    blur: &'a [f32],
    /// `t[c][i]`: twice the trapezoid integral of channel `c` from the line's first pixel to pixel `i`.
    t: [Vec<i64>; 3],
    /// `table[k - 1]`: minima of the blur over `2^k` pixels along the line starting at each pixel.
    table: Vec<Vec<f32>>,
}

impl<'a> Lines<'a> {
    /// The tables of a `w x h` planar picture (`3 w h` samples) and its blur map, for lines along `y` when
    /// `along_y`; range minima over windows of up to `2^max_k` pixels.
    pub(super) fn new(
        picture: &'a [f32],
        blur: &'a [f32],
        (w, h): (usize, usize),
        along_y: bool,
        origin: usize,
        max_k: usize,
    ) -> Self {
        let plane = w * h;
        let build = |c: usize| -> Vec<i64> {
            let src = &picture[c * plane..(c + 1) * plane];
            let mut t = vec![0i64; plane];
            if along_y {
                // Row by row: t[y + 1] = t[y] + q[y] + q[y + 1], all columns at once.
                for y in 0..h.saturating_sub(1) {
                    let (head, tail) = t.split_at_mut((y + 1) * w);
                    let prev = &head[y * w..];
                    let (a, b) = (&src[y * w..(y + 1) * w], &src[(y + 1) * w..(y + 2) * w]);
                    for x in 0..w {
                        tail[x] = prev[x] + quant(a[x]) + quant(b[x]);
                    }
                }
            } else {
                t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    let s = &src[y * w..(y + 1) * w];
                    for x in 0..w.saturating_sub(1) {
                        row[x + 1] = row[x] + quant(s[x]) + quant(s[x + 1]);
                    }
                });
            }
            t
        };
        let t = [build(0), build(1), build(2)];
        let mut table: Vec<Vec<f32>> = Vec::with_capacity(max_k);
        for k in 1..=max_k {
            let half = 1usize << (k - 1);
            let prev: &[f32] = if k == 1 { blur } else { &table[k - 2] };
            let mut level = prev.to_vec();
            level.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    let other = if along_y {
                        if y + half < h { Some(prev[(y + half) * w + x]) } else { None }
                    } else if x + half < w {
                        Some(prev[y * w + x + half])
                    } else {
                        None
                    };
                    if let Some(v) = other {
                        *o = o.min(v);
                    }
                }
            });
            table.push(level);
        }
        Lines {
            w,
            h,
            along_y,
            origin,
            picture,
            blur,
            t,
            table,
        }
    }

    /// Index of position `a` along the line through pixel `(x, y)`.
    #[inline(always)]
    fn index(&self, x: usize, y: usize, a: usize) -> usize {
        if self.along_y { a * self.w + x } else { y * self.w + a }
    }

    /// Length of the lines.
    #[inline(always)]
    fn len(&self) -> usize {
        if self.along_y { self.h } else { self.w }
    }

    /// The smallest blur over the pixels `lo ..= hi` (positions along the line through `(x, y)`), or `None`
    /// when the window is longer than the table reaches.
    #[inline]
    pub(super) fn min_blur(&self, x: usize, y: usize, lo: usize, hi: usize) -> Option<f32> {
        let len = hi + 1 - lo;
        let k = (usize::BITS - 1 - len.leading_zeros()) as usize;
        let level: &[f32] = if k == 0 {
            self.blur
        } else if k <= self.table.len() {
            &self.table[k - 1]
        } else {
            return None;
        };
        let a = level[self.index(x, y, lo)];
        let b = level[self.index(x, y, hi + 1 - (1usize << k))];
        Some(a.min(b))
    }

    /// The integral base (an exact integer) and the fractional part of twice the integral of channel `c` up to
    /// the frame position `t` along the line through `(x, y)`; the picture beyond the ends repeats its edge
    /// pixel (the ends of a region that is not the frame's are never reached).
    ///
    /// The position is split into an integer pixel and a fraction of the frame position itself, so that every
    /// rectangle of the frame computes the same fraction.
    #[inline]
    fn integral(&self, c: usize, x: usize, y: usize, t: f64) -> (i64, f64) {
        let n = self.len();
        let plane = self.w * self.h;
        let ch = &self.picture[c * plane..(c + 1) * plane];
        let q = |a: usize| quant(ch[self.index(x, y, a)]) as f64;
        let origin = self.origin as f64;
        if t <= origin {
            return (0, 2.0 * q(0) * (t - origin));
        }
        let last = origin + (n - 1) as f64;
        if t >= last {
            return (self.t[c][self.index(x, y, n - 1)], 2.0 * q(n - 1) * (t - last));
        }
        let floor = t.floor();
        let f = t - floor;
        let j = floor as usize - self.origin;
        let (qj, qj1) = (q(j), q(j + 1));
        (self.t[c][self.index(x, y, j)], 2.0 * qj * f + (qj1 - qj) * f * f)
    }

    /// The mean of channel `c` over `[pos - len, pos + len]` (frame coordinates along the line) on the line
    /// through the region's pixel `(x, y)`.
    #[inline]
    pub(super) fn mean(&self, c: usize, x: usize, y: usize, pos: f64, len: f64) -> f64 {
        let (b1, f1) = self.integral(c, x, y, pos - len);
        let (b2, f2) = self.integral(c, x, y, pos + len);
        // Integer difference first: the bases depend on where the running sums started, their difference does not.
        ((b2 - b1) as f64 + (f2 - f1)) / (f64::from(1u32 << SHIFT) * 4.0 * len)
    }
}

/// The largest `k` with `2^k` at most `len` (for sizing the range tables).
pub(super) fn floor_log2(len: usize) -> usize {
    (usize::BITS - 1 - len.max(1).leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(w: usize, h: usize) -> Vec<f32> {
        (0..3 * w * h).map(|i| ((i * 7919) % 1000) as f32 / 250.0).collect()
    }

    #[test]
    fn the_mean_is_the_integral_of_the_linear_picture() {
        let (w, h) = (30, 20);
        let pic = picture(w, h);
        let blur = vec![1.0f32; w * h];
        for along_y in [true, false] {
            let lines = Lines::new(&pic, &blur, (w, h), along_y, 0, 4);
            let (x, y) = (11, 9);
            let (pos, len) = if along_y { (9.0, 3.4) } else { (11.0, 3.4) };
            // Reference: fine sampling of the linear interpolant.
            let at = |c: usize, t: f64| -> f64 {
                let n = if along_y { h } else { w } as f64;
                let t = t.clamp(0.0, n - 1.0);
                let j = (t.floor() as usize).min(if along_y { h - 2 } else { w - 2 });
                let f = t - j as f64;
                let idx = |a: usize| if along_y { a * w + x } else { y * w + a };
                let v = |a: usize| f64::from(pic[c * w * h + idx(a)]);
                v(j) + f * (v(j + 1) - v(j))
            };
            for c in 0..3 {
                let steps = 100_000;
                let mut acc = 0.0;
                for i in 0..steps {
                    acc += at(c, pos - len + 2.0 * len * (i as f64 + 0.5) / steps as f64);
                }
                let want = acc / steps as f64;
                let got = lines.mean(c, x, y, pos, len);
                assert!((got - want).abs() < 1e-6, "{along_y} {c}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn the_edges_repeat_and_a_constant_stays() {
        let (w, h) = (16, 16);
        let pic = vec![0.75f32; 3 * w * h];
        let blur = vec![1.0f32; w * h];
        let lines = Lines::new(&pic, &blur, (w, h), true, 0, 3);
        // An interval hanging over the frame's edge.
        let m = lines.mean(1, 4, 1, 1.0, 5.0);
        assert!((m - 0.75).abs() < 1e-7, "{m}");
    }

    #[test]
    fn range_minima_match_a_scan() {
        let (w, h) = (24, 17);
        let blur: Vec<f32> = (0..w * h).map(|i| ((i * 31) % 17) as f32).collect();
        let pic = vec![0.0f32; 3 * w * h];
        for along_y in [true, false] {
            let lines = Lines::new(&pic, &blur, (w, h), along_y, 0, 4);
            let (x, y) = (5, 6);
            let n = if along_y { h } else { w };
            for lo in 0..n {
                for hi in lo..n.min(lo + 31) {
                    let want = (lo..=hi)
                        .map(|a| blur[if along_y { a * w + x } else { y * w + a }])
                        .fold(f32::INFINITY, f32::min);
                    match lines.min_blur(x, y, lo, hi) {
                        Some(got) => assert_eq!(got, want, "{along_y} [{lo}, {hi}]"),
                        None => assert!(hi + 1 - lo >= 32),
                    }
                }
            }
        }
    }

    #[test]
    fn a_running_sum_does_not_depend_on_where_it_started() {
        // The same line cut at two different places gives the same mean, bit for bit.
        let (w, h) = (6, 80);
        let pic = picture(w, h);
        let blur = vec![1.0f32; w * h];
        let whole = Lines::new(&pic, &blur, (w, h), true, 0, 3);
        // Rows 20 .. 70 only.
        let sub_h = 50;
        let mut sub = Vec::new();
        for c in 0..3 {
            sub.extend_from_slice(&pic[c * w * h + 20 * w..c * w * h + 70 * w]);
        }
        let sub_blur = vec![1.0f32; w * sub_h];
        let part = Lines::new(&sub, &sub_blur, (w, sub_h), true, 20, 3);
        for c in 0..3 {
            let a = whole.mean(c, 2, 40, 40.3, 6.7);
            let b = part.mean(c, 2, 20, 40.3, 6.7);
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
