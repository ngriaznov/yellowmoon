//! Convolution of a plane with a symmetric-window kernel by FFT (`docs/21` section 4, L3, "the tail plane").
//!
//! The image is zero-padded so that nothing wraps around into the frame: for a kernel reaching `hx` columns and
//! `hy` rows, a transform of at least `w + hx` by `h + hy` is enough to read the frame `[0, w) x [0, h)` of the
//! circular convolution as the linear one. The transform is `rustfft`'s scalar planner (its algorithms do not
//! depend on the CPU's vector extensions), applied to rows in parallel; every row is independent, so the bits do
//! not depend on the number of threads.

use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlannerScalar};
use std::sync::Arc;

type C = Complex<f64>;

/// A kernel stored as a window of `(2 hy + 1) x (2 hx + 1)` samples around its centre.
pub(crate) struct Kernel2d {
    /// Columns on each side of the centre.
    pub(crate) hx: usize,
    /// Rows on each side of the centre.
    pub(crate) hy: usize,
    /// The samples, row-major.
    pub(crate) data: Vec<f64>,
}

/// The smallest `2^a 3^b 5^c` that is at least `n` (a size the FFT does quickly).
pub(crate) fn fast_size(n: usize) -> usize {
    let n = n.max(1);
    let mut best = usize::MAX;
    let mut p2 = 1usize;
    while p2 < 2 * n {
        let mut p3 = p2;
        while p3 < 2 * n {
            let mut p5 = p3;
            while p5 < 2 * n {
                if p5 >= n && p5 < best {
                    best = p5;
                }
                p5 *= 5;
            }
            p3 *= 3;
        }
        p2 *= 2;
    }
    best
}

/// Transposes `src` (`rows x cols`, row-major) into `dst` (`cols x rows`).
fn transpose(src: &[C], dst: &mut [C], rows: usize, cols: usize) {
    const B: usize = 32;
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                for c in c0..(c0 + B).min(cols) {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Runs `fft` on every row of length `len` of `buf`, in parallel.
fn rows(buf: &mut [C], len: usize, fft: &Arc<dyn Fft<f64>>) {
    let scratch = fft.get_inplace_scratch_len();
    buf.par_chunks_mut(len).for_each_init(
        || vec![C::new(0.0, 0.0); scratch],
        |s, row| fft.process_with_scratch(row, s),
    );
}

/// The transforms for one padded size.
pub(crate) struct Convolver {
    nx: usize,
    ny: usize,
    fwd_x: Arc<dyn Fft<f64>>,
    fwd_y: Arc<dyn Fft<f64>>,
    inv_x: Arc<dyn Fft<f64>>,
    inv_y: Arc<dyn Fft<f64>>,
}

impl Convolver {
    /// A convolver for an image of `w x h` and kernels reaching `hx` columns and `hy` rows.
    pub(crate) fn new(w: usize, h: usize, hx: usize, hy: usize) -> Self {
        let nx = fast_size(w + hx);
        let ny = fast_size(h + hy);
        let mut planner = FftPlannerScalar::<f64>::new();
        Convolver {
            nx,
            ny,
            fwd_x: planner.plan_fft_forward(nx),
            fwd_y: planner.plan_fft_forward(ny),
            inv_x: planner.plan_fft_inverse(nx),
            inv_y: planner.plan_fft_inverse(ny),
        }
    }

    /// The padded size `(nx, ny)`.
    #[cfg(test)]
    pub(crate) fn size(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// The forward 2-D transform of `buf` (`ny` rows of `nx`), left transposed (`nx` rows of `ny`).
    fn forward(&self, buf: Vec<C>) -> Vec<C> {
        let mut buf = buf;
        rows(&mut buf, self.nx, &self.fwd_x);
        let mut t = vec![C::new(0.0, 0.0); buf.len()];
        transpose(&buf, &mut t, self.ny, self.nx);
        drop(buf);
        rows(&mut t, self.ny, &self.fwd_y);
        t
    }

    /// The inverse of [`Convolver::forward`] (unnormalised), back in `ny` rows of `nx`.
    fn inverse(&self, spectrum: Vec<C>) -> Vec<C> {
        let mut t = spectrum;
        rows(&mut t, self.ny, &self.inv_y);
        let mut buf = vec![C::new(0.0, 0.0); t.len()];
        transpose(&t, &mut buf, self.nx, self.ny);
        drop(t);
        rows(&mut buf, self.nx, &self.inv_x);
        buf
    }

    /// The spectrum of a kernel, in the layout [`Convolver::forward`] gives.
    pub(crate) fn kernel_spectrum(&self, k: &Kernel2d) -> Vec<C> {
        let mut buf = vec![C::new(0.0, 0.0); self.nx * self.ny];
        let side = 2 * k.hx + 1;
        for (q, v) in k.data.iter().enumerate() {
            let dx = (q % side) as isize - k.hx as isize;
            let dy = (q / side) as isize - k.hy as isize;
            let x = dx.rem_euclid(self.nx as isize) as usize;
            let y = dy.rem_euclid(self.ny as isize) as usize;
            buf[y * self.nx + x] = C::new(*v, 0.0);
        }
        self.forward(buf)
    }

    /// `img` (`w x h`, row-major) convolved with the kernel whose spectrum is `kernel`, cropped to `w x h`.
    pub(crate) fn convolve(&self, img: &[f64], w: usize, h: usize, kernel: &[C]) -> Vec<f64> {
        let mut buf = vec![C::new(0.0, 0.0); self.nx * self.ny];
        for y in 0..h {
            for x in 0..w {
                buf[y * self.nx + x] = C::new(img[y * w + x], 0.0);
            }
        }
        let mut spectrum = self.forward(buf);
        spectrum
            .par_iter_mut()
            .zip(kernel.par_iter())
            .for_each(|(a, b)| *a *= *b);
        let back = self.inverse(spectrum);
        let norm = 1.0 / (self.nx * self.ny) as f64;
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = back[y * self.nx + x].re * norm;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sizes_are_smooth_and_not_smaller() {
        assert_eq!(fast_size(1), 1);
        assert_eq!(fast_size(7), 8);
        assert_eq!(fast_size(11), 12);
        assert_eq!(fast_size(1023), 1024);
        assert_eq!(fast_size(1025), 1080);
        for n in 1..2000 {
            let f = fast_size(n);
            assert!(f >= n);
            let mut m = f;
            for p in [2, 3, 5] {
                while m % p == 0 {
                    m /= p;
                }
            }
            assert_eq!(m, 1, "{f} for {n}");
        }
    }

    /// A direct convolution, zero outside the image.
    fn direct(img: &[f64], w: usize, h: usize, k: &Kernel2d) -> Vec<f64> {
        let side = 2 * k.hx + 1;
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for (q, kv) in k.data.iter().enumerate() {
                    let dx = (q % side) as isize - k.hx as isize;
                    let dy = (q / side) as isize - k.hy as isize;
                    // out[x] = sum_m img[m] ker[x - m]  =>  m = x - d
                    let (mx, my) = (x as isize - dx, y as isize - dy);
                    if mx >= 0 && my >= 0 && (mx as usize) < w && (my as usize) < h {
                        s += img[my as usize * w + mx as usize] * kv;
                    }
                }
                out[y * w + x] = s;
            }
        }
        out
    }

    #[test]
    fn the_fft_matches_a_direct_convolution_without_wrap_around() {
        let (w, h) = (37, 23);
        let img: Vec<f64> = (0..w * h)
            .map(|i| ((i * 7919) % 101) as f64 / 101.0)
            .collect();
        // A lopsided kernel that reaches across most of the frame.
        let (hx, hy) = (30usize, 19usize);
        let side = 2 * hx + 1;
        let data: Vec<f64> = (0..side * (2 * hy + 1))
            .map(|q| (((q * 31) % 17) as f64 - 4.0) / 17.0)
            .collect();
        let k = Kernel2d { hx, hy, data };
        let conv = Convolver::new(w, h, hx, hy);
        let (nx, ny) = conv.size();
        assert!(nx >= w + hx && ny >= h + hy);
        let got = conv.convolve(&img, w, h, &conv.kernel_spectrum(&k));
        let want = direct(&img, w, h, &k);
        for (g, v) in got.iter().zip(&want) {
            assert!((g - v).abs() < 1e-11, "{g} vs {v}");
        }
    }
}
