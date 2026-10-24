//! The spectral projection: the lens data are given per wavelength, the photo has three channels.
//!
//! The camera's curves (`data/cameras/sony/`, `docs/03`) are resampled to the eight wavelengths of the spread
//! functions and to the 31-sample grid of coatings and tints (`docs/21` section 6, "Colour"):
//!
//! * a spread function's channel table is `sum_j w_c(l_j) P_j` with the equal-energy weights
//!   `w_c(l_j) = S_c(l_j) / sum_j S_c(l_j)`, which sum to 1 per channel;
//! * a reflectance spectrum `R` on the 31-sample grid has the tint `t_c = sum S_c(l) R(l) / sum S_c(l)`, so a
//!   flat spectrum `R = r` tints every channel by exactly `r`.

use sha2::{Digest, Sha256};
use ym_color::spectral::SpectralCurves;

use crate::consts::{GRID_SAMPLES, GRID_START_NM, GRID_STEP_NM, PSF_WAVELENGTHS, PSF_WAVELENGTHS_NM};
use crate::error::LensError;

/// A camera's three channels seen at the wavelengths of the lens data.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraResponse {
    /// `w_c(l_j)`, indexed `[channel][wavelength]`.
    weights: [[f64; PSF_WAVELENGTHS]; 3],
    /// `S_c` on the 31-sample grid, indexed `[channel][sample]`.
    grid: [[f64; GRID_SAMPLES]; 3],
    /// `sum S_c` on the grid.
    grid_sum: [f64; 3],
    key: String,
}

/// Linear interpolation of `(wavelengths, values)` at `nm`; `None` outside the sampled range.
fn interpolate(wavelengths: &[f64], values: &[f64], nm: f64) -> Option<f64> {
    let (first, last) = (*wavelengths.first()?, *wavelengths.last()?);
    if nm < first || nm > last {
        return None;
    }
    let hi = wavelengths.partition_point(|w| *w < nm).min(wavelengths.len() - 1);
    if hi == 0 || wavelengths[hi] == nm {
        return Some(values[hi]);
    }
    let lo = hi - 1;
    let t = (nm - wavelengths[lo]) / (wavelengths[hi] - wavelengths[lo]);
    Some(values[lo] + t * (values[hi] - values[lo]))
}

impl CameraResponse {
    /// Projects a camera's curves (negative samples are clipped to 0 first).
    ///
    /// # Errors
    ///
    /// [`LensError::Camera`] when the curves do not cover 400 to 700 nm or a channel is zero at the lens
    /// data's wavelengths.
    pub fn from_curves(curves: &SpectralCurves) -> Result<Self, LensError> {
        let curves = curves.clipped_at_zero();
        let wl = curves.wavelengths_nm();
        let mut at_psf = [[0.0; PSF_WAVELENGTHS]; 3];
        let mut grid = [[0.0; GRID_SAMPLES]; 3];
        for c in 0..3 {
            let channel: Vec<f64> = curves.samples().iter().map(|s| s[c]).collect();
            for (j, nm) in PSF_WAVELENGTHS_NM.iter().enumerate() {
                at_psf[c][j] = interpolate(wl, &channel, *nm).ok_or_else(|| {
                    LensError::Camera(format!("the curves do not cover {nm} nm"))
                })?;
            }
            for (k, g) in grid[c].iter_mut().enumerate() {
                let nm = GRID_START_NM + GRID_STEP_NM * k as f64;
                *g = interpolate(wl, &channel, nm).ok_or_else(|| {
                    LensError::Camera(format!("the curves do not cover {nm} nm"))
                })?;
            }
        }
        let mut weights = [[0.0; PSF_WAVELENGTHS]; 3];
        let mut grid_sum = [0.0; 3];
        for c in 0..3 {
            let total: f64 = at_psf[c].iter().sum();
            let sum: f64 = grid[c].iter().sum();
            if total <= 0.0 || sum <= 0.0 {
                return Err(LensError::Camera(format!(
                    "channel {} has no sensitivity at the wavelengths of the lens data",
                    ['R', 'G', 'B'][c]
                )));
            }
            for j in 0..PSF_WAVELENGTHS {
                weights[c][j] = at_psf[c][j] / total;
            }
            grid_sum[c] = sum;
        }
        let mut hasher = Sha256::new();
        for v in weights.iter().flatten().chain(grid.iter().flatten()) {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let key = format!("cam-{}", hex::encode(&hasher.finalize()[..8]));
        Ok(CameraResponse {
            weights,
            grid,
            grid_sum,
            key,
        })
    }

    /// A camera with equal weights at every wavelength and a flat sensitivity: the tint of a spectrum is its
    /// mean. For tests and callers without curves.
    #[must_use]
    pub fn neutral() -> Self {
        CameraResponse {
            weights: [[1.0 / PSF_WAVELENGTHS as f64; PSF_WAVELENGTHS]; 3],
            grid: [[1.0; GRID_SAMPLES]; 3],
            grid_sum: [GRID_SAMPLES as f64; 3],
            key: "neutral".to_owned(),
        }
    }

    /// A short text that identifies the projection: equal keys, equal weights.
    #[must_use]
    pub fn key(&self) -> String {
        self.key.clone()
    }

    /// The weights `w_c(l_j)` indexed `[channel][wavelength]`; each channel's row sums to 1.
    #[must_use]
    pub fn weights(&self) -> &[[f64; PSF_WAVELENGTHS]; 3] {
        &self.weights
    }

    /// The camera's sensitivity on the 31-sample grid, indexed `[channel][sample]`.
    #[must_use]
    pub fn grid(&self) -> &[[f64; GRID_SAMPLES]; 3] {
        &self.grid
    }

    /// The tint of a reflectance spectrum on the 31-sample grid: `t_c = sum S_c R / sum S_c`.
    #[must_use]
    pub fn tint(&self, spectrum: &[f64; GRID_SAMPLES]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for c in 0..3 {
            let dot: f64 = self.grid[c].iter().zip(spectrum).map(|(s, r)| s * r).sum();
            out[c] = dot / self.grid_sum[c];
        }
        out
    }

    /// The per-channel value of a quantity given at the eight PSF wavelengths: `sum_j w_c(l_j) v_j`.
    #[must_use]
    pub fn project(&self, per_wavelength: &[f64; PSF_WAVELENGTHS]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for c in 0..3 {
            out[c] = self.weights[c]
                .iter()
                .zip(per_wavelength)
                .map(|(w, v)| w * v)
                .sum();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_is_linear_and_bounded() {
        let wl = [400.0, 410.0, 430.0];
        let v = [0.0, 1.0, 3.0];
        assert_eq!(interpolate(&wl, &v, 400.0), Some(0.0));
        assert_eq!(interpolate(&wl, &v, 405.0), Some(0.5));
        assert_eq!(interpolate(&wl, &v, 420.0), Some(2.0));
        assert_eq!(interpolate(&wl, &v, 430.0), Some(3.0));
        assert_eq!(interpolate(&wl, &v, 399.0), None);
        assert_eq!(interpolate(&wl, &v, 431.0), None);
    }

    #[test]
    fn neutral_is_flat() {
        let n = CameraResponse::neutral();
        for row in n.weights() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        }
        let t = n.tint(&[0.3; GRID_SAMPLES]);
        for v in t {
            assert!((v - 0.3).abs() < 1e-15);
        }
    }
}
