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
//! sensitivities: `W(l) = 1/2 erfc((c_b - l) / (w_b sqrt 2)) * 1/2 erfc((l - c_r) / (w_r sqrt 2))`, that is, each edge
//! is a normal distribution's cumulative curve with the width as its standard deviation. The reading (the red edge
//! falls; the width is a standard deviation) is ours, from the numbers (blue edges at 400 to 456 nm, red edges at 580
//! to 630 nm) and fitted against outputs only: see `docs/24` section 3.7a, F-06. `W` is renormalised per layer by
//! `k_c = sum ill S_c / sum ill S_c W` so the reference light `ill` gives the same raw values as without it. The
//! surface correction is off.
//!
//! **Balance and placement.** Dividing by `R_c(white)` makes a neutral give three equal raw values. Scene 0.18
//! lands at `log10 E = MID_GREY_LOG10` (-0.0095) on every profile's axis, where the oracle puts it (F-06b); the
//! per-stock fit to the profile's `midscale_neutral_density` is kept as [`mid_log_exposure`] for diagnostics.

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

/// The log10 raw exposure of scene 0.18 on the profile's axis, the same for every stock (F-06b). The oracle puts a
/// neutral at 0.9786 of the exposure that the film's reference light gives a unit reflectance, on every profile
/// (`log_e_film` of the grey is -0.0095 on all seven, window off), instead of finding a place per stock from the
/// midscale density. [`mid_log_exposure`] keeps that per-stock reading for the diagnostics.
pub const MID_GREY_LOG10: f64 = -0.0095;

/// The factor between the width in the profile and the divisor of the error function's argument: the width is the standard
/// deviation of a normal edge, `1/2 erfc(z / (w sqrt 2))`, not the divisor of `z` itself. See the module header.
const WIDTH_SCALE: f64 = std::f64::consts::SQRT_2;

/// The window of a profile's four numbers, one factor per sample (all 1 when the profile has none).
#[must_use]
pub fn window_of(params: &[f64]) -> Spectrum {
    let mut window = [1.0; N_LAMBDA];
    if params.len() == 4 {
        for (i, w) in window.iter_mut().enumerate() {
            let l = lambda_nm(i);
            *w = rising_edge(l, params[0], params[1] * WIDTH_SCALE)
                * falling_edge(l, params[2], params[3] * WIDTH_SCALE);
        }
    }
    window
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
        let window = if use_window {
            window_of(&profile.window_params)
        } else {
            [1.0; N_LAMBDA]
        };
        Self::with_window(profile, cie, spectra, &window)
    }

    /// Builds the stage with an explicit window (one factor per sample, 1 for none). The diagnostics of `tests/oracle*.rs`
    /// use it to compare readings of the profile's four window numbers.
    ///
    /// # Errors
    ///
    /// [`FilmError::ViewingLight`] when the reference light is unknown.
    pub fn with_window(
        profile: &Profile,
        cie: &Cie,
        spectra: Arc<dyn Spectra>,
        window: &Spectrum,
    ) -> Result<Self, FilmError> {
        let ill = reference_spectrum(profile, cie)?;
        Self::with_light(profile, cie, spectra, window, &ill)
    }

    /// Builds the stage for an explicit reference light `ill` (the profile's own is [`reference_spectrum`]); the
    /// diagnostics of `tests/oracle*.rs` use it to ask which light the oracle's balance corresponds to.
    ///
    /// # Errors
    ///
    /// Never today; the `Result` keeps the signature of [`ExposureStage::with_window`].
    pub fn with_light(
        profile: &Profile,
        cie: &Cie,
        spectra: Arc<dyn Spectra>,
        window: &Spectrum,
        ill: &Spectrum,
    ) -> Result<Self, FilmError> {
        Self::with_lens(profile, cie, spectra, window, ill, &[1.0; N_LAMBDA])
    }

    /// Builds the stage with a lens transmittance `lens` (one factor per sample) in front of the film: the ultraviolet
    /// and infrared cut of a camera. It multiplies the sensitivities everywhere, the balance included.
    ///
    /// # Errors
    ///
    /// Never today; the `Result` keeps the signature of [`ExposureStage::with_light`].
    pub fn with_lens(
        profile: &Profile,
        cie: &Cie,
        spectra: Arc<dyn Spectra>,
        window: &Spectrum,
        ill: &Spectrum,
        lens: &Spectrum,
    ) -> Result<Self, FilmError> {
        let mut raw = profile.sensitivity();
        for (i, r) in raw.iter_mut().enumerate() {
            for v in r.iter_mut() {
                *v *= lens[i];
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
            let xy = cie.xy_of_light(ill);
            spectra.spectrum(xy[0], xy[1])
        };
        // The balance is the response of the table's smooth white to the windowed sensitivities, so a neutral gives three
        // equal raw values (the gate of `tests/gates.rs`). The oracle does not do this: it divides by the response
        // without the window, so its neutral is off by +0.005, -0.005, -0.037 log10 (Portra 400) with the window on;
        // the print's mid-grey normalisation absorbs that, and we keep the exact balance (F-06b).
        let mut white_resp = [0.0; 3];
        for c in 0..3 {
            white_resp[c] = (0..N_LAMBDA).map(|i| white[i] * sens[c][i]).sum();
        }
        Ok(Self {
            sens,
            spectra,
            white_resp,
            x_mid: MID_GREY_LOG10,
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
