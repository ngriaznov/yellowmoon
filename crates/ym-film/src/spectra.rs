//! Spectra from a chromaticity: the input of stage 2.
//!
//! Written from Jakob and Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling", Computer
//! Graphics Forum 38(2), 2019 (a spectrum as a sigmoid of a quadratic in wavelength), and CIE 15 (chromaticity).
//!
//! A [`Spectra`] is fed a chromaticity `(x, y)` and returns the 81 samples (380 to 780 nm, 5 nm) of a smooth emission
//! spectrum whose luminance `Y` is 1. Two implementations:
//!
//! * [`TableSpectra`] reads our upsampling table (`data/film/upsampling/coeffs.f32`, made by `training/film/`):
//!   little-endian f32 `[N][N][4]` = `(c0, c1, c2, Y_s)` over `x, y` in `[0, 1]`; the spectrum is
//!   `sigmoid(c0 t^2 + c1 t + c2)` with `t = (l - 380) / 400` and `sigmoid(z) = 1/2 + z / (2 sqrt(1 + z^2))`; for
//!   unit luminance it is divided by `Y_s`. The four numbers are interpolated bilinearly.
//! * [`ReferenceSpectra`] is a stand-in that needs no table: the reference light's own spectrum for a neutral,
//!   times a smooth sigmoid-of-a-quadratic factor that is solved so that the chromaticity is right. It is for
//!   tests and for working before the table exists.

use std::path::Path;

use crate::cie::Cie;
use crate::error::FilmError;
use crate::math::solve2;
use crate::{N_LAMBDA, Spectrum, Vec3};

/// Something that turns a chromaticity into a spectrum of unit luminance.
pub trait Spectra: Send + Sync {
    /// The 81-sample spectrum for chromaticity `(x, y)` whose luminance (against the 1931 observer, with the
    /// same weights as the engine's) is 1.
    fn spectrum(&self, x: f64, y: f64) -> Spectrum;
}

/// `sigmoid(z) = 1/2 + z / (2 sqrt(1 + z^2))`.
#[must_use]
pub fn sigmoid(z: f64) -> f64 {
    0.5 + z / (2.0 * (1.0 + z * z).sqrt())
}

/// The sigmoid-of-a-quadratic spectrum `s(l)` of the table's format, un-normalised.
#[must_use]
pub fn sigmoid_quadratic(c: [f64; 3]) -> Spectrum {
    let mut out = [0.0; N_LAMBDA];
    for (i, o) in out.iter_mut().enumerate() {
        let t = (i as f64 * 5.0) / 400.0;
        *o = sigmoid((c[0] * t + c[1]) * t + c[2]);
    }
    out
}

/// Our upsampling table.
#[derive(Debug, Clone)]
pub struct TableSpectra {
    n: usize,
    data: Vec<f32>,
}

impl TableSpectra {
    /// Reads `coeffs.f32` from `dir` (normally `data/film/upsampling`).
    ///
    /// # Errors
    ///
    /// [`FilmError::Io`] when the file cannot be read, [`FilmError::Data`] when its size is not `N * N * 4`
    /// floats for a whole `N` or has a value that is not finite.
    pub fn load(dir: &Path) -> Result<Self, FilmError> {
        let path = dir.join("coeffs.f32");
        let bytes = std::fs::read(&path).map_err(|source| FilmError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_bytes(&bytes)
    }

    /// Builds the table from the bytes of `coeffs.f32`.
    ///
    /// # Errors
    ///
    /// As [`TableSpectra::load`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FilmError> {
        let name = "upsampling/coeffs.f32";
        if bytes.len() % 16 != 0 {
            return Err(FilmError::data(name, "its size is not a whole number of (c0, c1, c2, Y) cells"));
        }
        let cells = bytes.len() / 16;
        let n = (cells as f64).sqrt().round() as usize;
        if n < 2 || n * n != cells {
            return Err(FilmError::data(name, format!("{cells} cells do not make a square table")));
        }
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if data.iter().any(|v| !v.is_finite()) {
            return Err(FilmError::data(name, "it holds a value that is not a number"));
        }
        Ok(Self { n, data })
    }

    /// The side of the table.
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    fn cell(&self, ix: usize, iy: usize) -> [f64; 4] {
        let o = (iy * self.n + ix) * 4;
        [
            f64::from(self.data[o]),
            f64::from(self.data[o + 1]),
            f64::from(self.data[o + 2]),
            f64::from(self.data[o + 3]),
        ]
    }
}

impl Spectra for TableSpectra {
    fn spectrum(&self, x: f64, y: f64) -> Spectrum {
        let last = (self.n - 1) as f64;
        let fx = (x.clamp(0.0, 1.0) * last).min(last);
        let fy = (y.clamp(0.0, 1.0) * last).min(last);
        let ix = (fx.floor() as usize).min(self.n - 2);
        let iy = (fy.floor() as usize).min(self.n - 2);
        let (tx, ty) = (fx - ix as f64, fy - iy as f64);
        let (a, b) = (self.cell(ix, iy), self.cell(ix + 1, iy));
        let (c, d) = (self.cell(ix, iy + 1), self.cell(ix + 1, iy + 1));
        let mut v = [0.0; 4];
        for k in 0..4 {
            v[k] = (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty;
        }
        let s = sigmoid_quadratic([v[0], v[1], v[2]]);
        let ys = if v[3].abs() > 1e-12 { v[3] } else { 1.0 };
        let mut out = [0.0; N_LAMBDA];
        for i in 0..N_LAMBDA {
            out[i] = s[i] / ys;
        }
        out
    }
}

/// A stand-in spectra source: `reference(l) * 2 sigmoid(c0 t^2 + c1 t + c2)`, solved for the chromaticity.
///
/// For the reference light's own chromaticity the coefficients are 0 and the spectrum is the light itself. The
/// solve is a small damped Gauss-Newton on `(x, y)` with the minimum-norm step (three unknowns, two equations).
#[derive(Debug, Clone)]
pub struct ReferenceSpectra {
    reference: Spectrum,
    cmf: [Vec3; N_LAMBDA],
}

impl ReferenceSpectra {
    /// A stand-in around the light `reference` (any scale).
    #[must_use]
    pub fn new(cie: &Cie, reference: Spectrum) -> Self {
        Self {
            reference,
            cmf: cie.cmf,
        }
    }

    fn build(&self, c: [f64; 3]) -> Spectrum {
        let q = sigmoid_quadratic(c);
        let mut out = [0.0; N_LAMBDA];
        for i in 0..N_LAMBDA {
            out[i] = self.reference[i] * 2.0 * q[i];
        }
        out
    }

    fn xyz(&self, s: &Spectrum) -> Vec3 {
        let mut xyz = [0.0; 3];
        for i in 0..N_LAMBDA {
            for k in 0..3 {
                xyz[k] += s[i] * self.cmf[i][k];
            }
        }
        xyz
    }

    fn xy(&self, c: [f64; 3]) -> [f64; 2] {
        let xyz = self.xyz(&self.build(c));
        let sum = xyz[0] + xyz[1] + xyz[2];
        [xyz[0] / sum, xyz[1] / sum]
    }
}

impl Spectra for ReferenceSpectra {
    fn spectrum(&self, x: f64, y: f64) -> Spectrum {
        let mut c = [0.0; 3];
        for _ in 0..40 {
            let f = self.xy(c);
            let r = [x - f[0], y - f[1]];
            if r[0].abs() + r[1].abs() < 1e-12 {
                break;
            }
            // Numeric Jacobian, 2 x 3.
            let mut j = [[0.0; 3]; 2];
            for k in 0..3 {
                let mut cc = c;
                cc[k] += 1e-5;
                let g = self.xy(cc);
                j[0][k] = (g[0] - f[0]) / 1e-5;
                j[1][k] = (g[1] - f[1]) / 1e-5;
            }
            // Minimum-norm step: c += J^T (J J^T + mu I)^-1 r.
            let mut a = [[0.0; 2]; 2];
            for p in 0..2 {
                for q in 0..2 {
                    a[p][q] = (0..3).map(|k| j[p][k] * j[q][k]).sum();
                }
                a[p][p] += 1e-9;
            }
            let Some(w) = solve2(a, r) else { break };
            let mut step = [0.0; 3];
            for k in 0..3 {
                step[k] = j[0][k] * w[0] + j[1][k] * w[1];
            }
            let len = (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt();
            let scale = if len > 4.0 { 4.0 / len } else { 1.0 };
            for k in 0..3 {
                c[k] += step[k] * scale;
            }
        }
        let s = self.build(c);
        let yv = self.xyz(&s)[1];
        let mut out = s;
        if yv > 0.0 {
            for v in &mut out {
                *v /= yv;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_reads_and_interpolates() {
        // A 2 x 2 table whose cells all hold c = (0, 0, 0), Y_s = 0.5: the spectrum is 0.5 / 0.5 = 1 everywhere.
        let mut bytes = Vec::new();
        for _ in 0..4 {
            for v in [0.0f32, 0.0, 0.0, 0.5] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let t = TableSpectra::from_bytes(&bytes).expect("table");
        assert_eq!(t.size(), 2);
        let s = t.spectrum(0.3, 0.3);
        assert!(s.iter().all(|v| (v - 1.0).abs() < 1e-12));
        assert!(TableSpectra::from_bytes(&bytes[..15]).is_err());
        assert!(TableSpectra::from_bytes(&bytes[..48]).is_err());
    }
}
