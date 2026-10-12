//! A small power-of-two 2-D FFT (radix-2, in place, f64) for the circular convolution of the illuminant
//! filter. Row-major `n x n` data in separate real and imaginary slices; the inverse includes the `1 / n^2`.

use std::f64::consts::PI;

/// Precomputed tables for `n x n` transforms, `n` a power of two.
#[derive(Debug, Clone)]
pub(crate) struct Fft2 {
    n: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
    rev: Vec<usize>,
}

impl Fft2 {
    /// Tables for `n x n` transforms. `n` must be a power of two of at least 2 (checked by the model loader).
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n >= 2 && n.is_power_of_two());
        let cos = (0..n / 2)
            .map(|k| (2.0 * PI * k as f64 / n as f64).cos())
            .collect();
        let sin = (0..n / 2)
            .map(|k| (2.0 * PI * k as f64 / n as f64).sin())
            .collect();
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        Self { n, cos, sin, rev }
    }

    /// In-place 1-D transform of one line (`sign` -1 forward, +1 inverse without scaling).
    fn line(&self, re: &mut [f64], im: &mut [f64], sign: f64) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], sign * self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }

    fn transform(&self, re: &mut [f64], im: &mut [f64], sign: f64) {
        let n = self.n;
        debug_assert!(re.len() == n * n && im.len() == n * n);
        for r in 0..n {
            let (lo, hi) = (r * n, (r + 1) * n);
            self.line(&mut re[lo..hi], &mut im[lo..hi], sign);
        }
        let mut col_re = vec![0.0; n];
        let mut col_im = vec![0.0; n];
        for c in 0..n {
            for r in 0..n {
                col_re[r] = re[r * n + c];
                col_im[r] = im[r * n + c];
            }
            self.line(&mut col_re, &mut col_im, sign);
            for r in 0..n {
                re[r * n + c] = col_re[r];
                im[r * n + c] = col_im[r];
            }
        }
    }

    /// Forward 2-D transform, unnormalised (`numpy.fft.fft2`).
    pub(crate) fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        self.transform(re, im, -1.0);
    }

    /// Inverse 2-D transform with the `1 / n^2` factor (`numpy.fft.ifft2`).
    pub(crate) fn inverse(&self, re: &mut [f64], im: &mut [f64]) {
        self.transform(re, im, 1.0);
        let s = 1.0 / (self.n * self.n) as f64;
        for v in re.iter_mut().chain(im.iter_mut()) {
            *v *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft2(n: usize, re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut out_re = vec![0.0; n * n];
        let mut out_im = vec![0.0; n * n];
        for ky in 0..n {
            for kx in 0..n {
                let (mut sr, mut si) = (0.0, 0.0);
                for y in 0..n {
                    for x in 0..n {
                        let ang = -2.0 * PI * ((ky * y + kx * x) as f64) / n as f64;
                        let (c, s) = (ang.cos(), ang.sin());
                        sr += re[y * n + x] * c - im[y * n + x] * s;
                        si += re[y * n + x] * s + im[y * n + x] * c;
                    }
                }
                out_re[ky * n + kx] = sr;
                out_im[ky * n + kx] = si;
            }
        }
        (out_re, out_im)
    }

    fn pseudo(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn forward_matches_the_naive_dft_and_inverse_round_trips() {
        for n in [2usize, 4, 8, 16] {
            let re = pseudo(n * n, 1 + n as u64);
            let im = pseudo(n * n, 99 + n as u64);
            let (want_re, want_im) = naive_dft2(n, &re, &im);
            let fft = Fft2::new(n);
            let (mut a, mut b) = (re.clone(), im.clone());
            fft.forward(&mut a, &mut b);
            for i in 0..n * n {
                assert!((a[i] - want_re[i]).abs() < 1e-11, "n={n} re[{i}]");
                assert!((b[i] - want_im[i]).abs() < 1e-11, "n={n} im[{i}]");
            }
            fft.inverse(&mut a, &mut b);
            for i in 0..n * n {
                assert!(
                    (a[i] - re[i]).abs() < 1e-13 && (b[i] - im[i]).abs() < 1e-13,
                    "round trip n={n}"
                );
            }
        }
    }
}
