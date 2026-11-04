//! Stage 2: film exposure. A chromaticity and a luminance become the raw exposure of the three layers.
//!
//! Written from Giorgianni and Madden, *Digital Color Management*, 2nd ed. (a film's exposure is the spectral
//! power integrated against each layer's spectral sensitivity), Hunt, *The Reproduction of Colour* (exposure
//! balance of a film for its reference light), the film data sheets the profiles are read from, and Jakob and
//! Hanika 2019 for the spectra (see [`crate::spectra`]).
//!
//! For a colour with chromaticity `(x, y)` and luminance `Y`:
//! `E_c = (Y / 0.18) * 10^x_mid * R_c(x, y) / R_c(white)`, with `R_c(x, y) = sum_l s_xy(l) S_c(l) W(l)`, `s_xy` the
//! spectrum of unit luminance, `S_c = 10^log_sensitivity_c` and `W` the adaptation window; `white` is the
//! reference light's chromaticity.
//!
//! **Window.** `hanatos2025_adaptation_window_params` are four numbers: the centre and width (nm) of a rising
//! error-function edge at the blue end and of a falling one at the red end, a band-pass multiplied onto the
//! sensitivities: `W(l) = 1/2 erfc((c_b - l) / w_b) * 1/2 erfc((l - c_r) / w_r)`. The reading (the width is the
//! divisor of the error function's argument, the red edge falls) is ours, from the research note and the
//! numbers (blue edges at 400 to 456 nm, red edges at 580 to 630 nm). `W` is renormalised per layer by
//! `k_c = sum ill S_c / sum ill S_c W` so the reference light `ill` gives the same raw values as without it. The
//! surface correction is off.
//!
//! **Balance and placement.** Dividing by `R_c(white)` makes a neutral give three equal raw values. Scene 0.18
//! lands at `log10 E = x_mid` on the profile's axis, where `x_mid` is the point at which the profile's curves and
//! dyes reproduce its documented `midscale_neutral_density` (found by least squares; 0 for papers).

use std::sync::Arc;

use crate::cie::{Cie, Light};
use crate::error::FilmError;
use crate::math::{falling_edge, pow10, rising_edge};
use crate::profile::Profile;
use crate::scene::SceneLight;
use crate::spectra::Spectra;
use crate::{N_LAMBDA, Spectrum, Vec3, lambda_nm};

/// Stage 2 for one film.
pub struct ExposureStage {
    sens: [Spectrum; 3],
    spectra: Arc<dyn Spectra>,
    white_resp: Vec3,
    x_mid: f64,
}

impl std::fmt::Debug for ExposureStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExposureStage")
            .field("x_mid", &self.x_mid)
            .field("white_resp", &self.white_resp)
            .finish_non_exhaustive()
    }
}

/// The spectrum of the film's reference light.
///
/// # Errors
///
/// [`FilmError::ViewingLight`] when the profile names a light the data does not have.
pub fn reference_spectrum(profile: &Profile, cie: &Cie) -> Result<Spectrum, FilmError> {
    Light::from_profile_name(&profile.reference_illuminant)
        .map(|l| cie.light(l))
        .ok_or_else(|| FilmError::ViewingLight {
            id: profile.id.clone(),
            light: profile.reference_illuminant.clone(),
        })
}

impl ExposureStage {
    /// Builds the stage.
    ///
    /// # Errors
    ///
    /// [`FilmError::ViewingLight`] when the reference light is unknown.
    pub fn new(
        profile: &Profile,
        cie: &Cie,
        spectra: Arc<dyn Spectra>,
        use_window: bool,
    ) -> Result<Self, FilmError> {
        let ill = reference_spectrum(profile, cie)?;
        let raw = profile.sensitivity();
        let mut window = [1.0; N_LAMBDA];
        if use_window && profile.window_params.len() == 4 {
            let p = &profile.window_params;
            for (i, w) in window.iter_mut().enumerate() {
                let l = lambda_nm(i);
                *w = rising_edge(l, p[0], p[1]) * falling_edge(l, p[2], p[3]);
            }
        }
        let mut sens = [[0.0; N_LAMBDA]; 3];
        for c in 0..3 {
            let (mut plain, mut windowed) = (0.0, 0.0);
            for i in 0..N_LAMBDA {
                plain += ill[i] * raw[i][c];
                windowed += ill[i] * raw[i][c] * window[i];
            }
            let k = if windowed > 0.0 {
                plain / windowed
            } else {
                1.0
            };
            for i in 0..N_LAMBDA {
                sens[c][i] = raw[i][c] * window[i] * k;
            }
        }
        let white = {
            let xy = cie.xy_of_light(&ill);
            spectra.spectrum(xy[0], xy[1])
        };
        let mut white_resp = [0.0; 3];
        for c in 0..3 {
            white_resp[c] = (0..N_LAMBDA).map(|i| white[i] * sens[c][i]).sum();
        }
        Ok(Self {
            sens,
            spectra,
            white_resp,
            x_mid: mid_log_exposure(profile),
        })
    }

    /// The log10 exposure where scene 0.18 lands on the profile's axis.
    #[must_use]
    pub fn mid_log_exposure(&self) -> f64 {
        self.x_mid
    }

    /// The raw exposure of the three layers; `gain_log10` shifts it (film exposure, brightness match).
    #[must_use]
    pub fn raw(&self, light: &SceneLight, gain_log10: f64) -> Vec3 {
        let s = self.spectra.spectrum(light.x, light.y);
        let scale = light.luminance / 0.18 * pow10(self.x_mid + gain_log10);
        let mut e = [0.0; 3];
        for c in 0..3 {
            let r: f64 = (0..N_LAMBDA).map(|i| s[i] * self.sens[c][i]).sum();
            e[c] = scale * r / self.white_resp[c];
        }
        e
    }

    /// `log10` of [`ExposureStage::raw`], floored at -30 so that no light is `-inf`.
    #[must_use]
    pub fn log_exposure(&self, light: &SceneLight, gain_log10: f64) -> Vec3 {
        self.raw(light, gain_log10).map(|v| v.max(1e-30).log10())
    }
}

/// The exposure at which the profile's curves and dyes give its `midscale_neutral_density`, by least squares over
/// the defined samples; 0 when the fit is poor or there is nothing to fit.
#[must_use]
pub fn mid_log_exposure(profile: &Profile) -> f64 {
    let defined: Vec<usize> = (0..N_LAMBDA)
        .filter(|i| !profile.midscale_neutral_density[*i].is_nan())
        .collect();
    if defined.len() < 20 {
        return 0.0;
    }
    let curves: Vec<crate::develop::MonotoneCubic> = (0..3)
        .map(|c| {
            let (x, y) = profile.curve(c);
            crate::develop::MonotoneCubic::new(x, y)
        })
        .collect();
    let mut best = (f64::INFINITY, 0.0);
    for k in 0..=800 {
        let x = -2.0 + 0.005 * f64::from(k);
        let d = [curves[0].eval(x), curves[1].eval(x), curves[2].eval(x)];
        let mut sum = 0.0;
        for &i in &defined {
            let m = profile.base_density[i]
                + (0..3)
                    .map(|c| d[c] * profile.channel_density[i][c])
                    .sum::<f64>();
            let e = m - profile.midscale_neutral_density[i];
            sum += e * e;
        }
        let rms = (sum / defined.len() as f64).sqrt();
        if rms < best.0 {
            best = (rms, x);
        }
    }
    if best.0 < 0.05 { best.1 } else { 0.0 }
}
