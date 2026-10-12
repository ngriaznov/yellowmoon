//! Illuminant estimator: the scene light as a camera white, class probabilities and a confidence
//! (`docs/02-colour-pipeline.md`, section Illuminant estimate; training code in `training/illum/`).
//!
//! The estimator is the baseline of `docs/02`: an FFCC-style filter on a log-chroma histogram of the 1/8-resolution
//! preview (Barron and Tsai, "Fast Fourier Color Constancy", CVPR 2017; the reference code is Apache-2.0, this is our own
//! implementation), the camera's as-shot white balance as a Gaussian prior, and a small class head. One model per
//! camera profile (`data/models/illum/<profile-id>.json`, under 50 k parameters, [`IlluminantModel::param_count`]).
//!
//! # Pipeline
//!
//! 1. [`histogram`]: for every unmasked pixel the log-chroma `u = ln(G/R)`, `v = ln(G/B)` is splatted (bilinear,
//!    weight `sqrt(Y)`) onto a 64 x 64 grid; a second channel does the same for the local absolute deviation of the
//!    image (the edges). Each channel is normalised to sum 1.
//! 2. Filter: `Z = sum_c X_c (*) F_c + B`, a circular convolution done with FFTs. `Z[i, j]` scores the light at
//!    log-chroma `(lo_u + i * bin_size, lo_v + j * bin_size)`.
//! 3. `P = softmax(Z)`; the histogram estimate is the mean of `P` with covariance `Cov(P) + pad * I`.
//! 4. The as-shot white balance, as a Gaussian `N(a, 0.05^2 I)` on `(u, v)`, multiplies the estimate (a precision-weighted
//!    mean). Without an as-shot value the estimate is the histogram's alone.
//! 5. `w_scene = [exp(-u), 1, exp(-v)]`: the camera RGB of the light, green = 1, in the camera's raw scale (the scale of
//!    the profile `locus`). The confidence is the probability that the angular error is below 2 degrees: a logistic
//!    regression on the class features, the Mahalanobis distance between the histogram estimate and the as-shot value and
//!    the log variances of the two Gaussians, fitted on held-out synthetic scenes.
//! 6. Class head: 25 features (histogram moments, the estimate, the as-shot value, the brightness value `Bv`) go through
//!    one `tanh` layer to a softmax over [`CLASS_NAMES`].
//!
//! The output feeds [`crate::wb::SceneLight`] (`IlluminantEstimate::scene_light`).
//!
//! # Input
//!
//! A [`RawPreview`]: black-subtracted camera RGB at about 1/8 of the sensor resolution, white level 1.0, **before any
//! white balance**, with a mask of the clipped pixels (any raw pixel of the block clipped). [`ShotInfo`] carries the
//! as-shot gains and the exposure settings from the metadata; both are optional.
//!
//! # Parity with the Python reference
//!
//! `training/illum/{features,model}.py` is the reference. The Rust code uses the same formulas in the same order and
//! agrees with it to 1e-9 on the committed vectors (`tests/illum.rs`); the histograms agree to about 1e-14.

mod fft;
mod format;
pub mod histogram;

use std::path::Path;

use crate::consts::Vec3;
use crate::wb::{EstimatorClasses, SceneLight, WbError};

use fft::Fft2;
use format::{CLASS_NAMES as CLASS_NAMES_INNER, N_CLASSES, Parsed};

/// The class names, in the order of the head's outputs and of [`IlluminantEstimate::class_probabilities`].
pub const CLASS_NAMES: [&str; 6] = CLASS_NAMES_INNER;
pub use histogram::{HistogramChannel, HistogramSpec, Histograms};

/// The camera's as-shot prior on `(u, v)`: standard deviation in log-chroma units (`docs/02`).
pub const PRIOR_SIGMA: f64 = 0.05;

/// Errors from loading a model or running the estimator.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IllumError {
    /// The model file could not be read.
    #[error("cannot read the illuminant model {path}: {reason}")]
    Io {
        /// The file.
        path: String,
        /// The operating system's message.
        reason: String,
    },
    /// The model file is not valid JSON of the expected layout.
    #[error("the illuminant model file is not valid: {0}")]
    Json(String),
    /// The model file parsed but its contents are not usable.
    #[error("the illuminant model is not usable: {0}")]
    InvalidModel(String),
    /// The tensors do not match the SHA-256 recorded in the file.
    #[error(
        "the illuminant model does not match its SHA-256 (file says {expected}, tensors give {found})"
    )]
    HashMismatch {
        /// The digest in the file.
        expected: String,
        /// The digest of the tensors.
        found: String,
    },
    /// Histograms handed to [`IlluminantModel::estimate_from_histograms`] do not have the model's shape.
    #[error("the histograms do not fit the model: {0}")]
    InvalidHistograms(String),
    /// The preview buffers do not match the stated size.
    #[error(
        "the preview is {width} x {height} pixels but its buffers hold {rgb_len} values and {mask_len:?} mask bytes"
    )]
    PreviewSize {
        /// Stated width in pixels.
        width: usize,
        /// Stated height in pixels.
        height: usize,
        /// Length of the RGB buffer.
        rgb_len: usize,
        /// Length of the mask buffer, if there is one.
        mask_len: Option<usize>,
    },
}

/// The 1/8-resolution preview the estimator reads: black-subtracted camera RGB before any white balance.
#[derive(Debug, Clone, Copy)]
pub struct RawPreview<'a> {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` values, `R G B` per pixel, row by row, white level = 1.0.
    pub rgb: &'a [f32],
    /// One byte per pixel, non-zero where the pixel is clipped (and so ignored); `None` for no mask.
    pub clip_mask: Option<&'a [u8]>,
}

impl RawPreview<'_> {
    pub(crate) fn validate(&self) -> Result<(), IllumError> {
        let pixels = self.width.saturating_mul(self.height);
        let ok_rgb = self.rgb.len() == pixels.saturating_mul(3);
        let ok_mask = self.clip_mask.is_none_or(|m| m.len() == pixels);
        if ok_rgb && ok_mask {
            Ok(())
        } else {
            Err(IllumError::PreviewSize {
                width: self.width,
                height: self.height,
                rgb_len: self.rgb.len(),
                mask_len: self.clip_mask.map(<[u8]>::len),
            })
        }
    }

    #[inline]
    pub(crate) fn is_clipped(&self, i: usize) -> bool {
        self.clip_mask.is_some_and(|m| m[i] != 0)
    }

    #[inline]
    pub(crate) fn pixel(&self, i: usize) -> [f64; 3] {
        [
            f64::from(self.rgb[3 * i]),
            f64::from(self.rgb[3 * i + 1]),
            f64::from(self.rgb[3 * i + 2]),
        ]
    }
}

/// What the file's metadata says about the shot. Every field is optional; a missing or unusable one is ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShotInfo {
    /// The camera's as-shot white balance gains `[R, G, B]` (Sony tag 0x7313), multipliers of the raw channels.
    pub as_shot_gains: Option<Vec3>,
    /// The f-number.
    pub f_number: Option<f64>,
    /// The exposure time in seconds.
    pub exposure_time_s: Option<f64>,
    /// The ISO speed.
    pub iso: Option<f64>,
}

impl ShotInfo {
    /// The as-shot light's log-chroma `(u, v) = (ln(g_R / g_G), ln(g_B / g_G))` from the gains (a white surface records
    /// `1 / g`), or `None` when the gains are missing, not finite or not positive.
    #[must_use]
    pub fn as_shot_uv(&self) -> Option<[f64; 2]> {
        let g = self.as_shot_gains?;
        g.iter()
            .all(|c| c.is_finite() && *c > 0.0)
            .then(|| [(g[0] / g[1]).ln(), (g[2] / g[1]).ln()])
    }

    /// The brightness value `Bv = log2(N^2 / t) - log2(ISO / 100)`, or `None` if a setting is missing, not positive or so
    /// extreme that the value is not finite.
    #[must_use]
    pub fn bv(&self) -> Option<f64> {
        let (n, t, iso) = (self.f_number?, self.exposure_time_s?, self.iso?);
        let usable = [n, t, iso].iter().all(|v| v.is_finite() && *v > 0.0);
        // Absurd settings (`n * n / t` overflows) give an infinite value, which is as unusable as a missing one.
        usable
            .then(|| (n * n / t).log2() - (iso / 100.0).log2())
            .filter(|bv| bv.is_finite())
    }
}

/// The estimator's output.
#[derive(Debug, Clone, PartialEq)]
pub struct IlluminantEstimate {
    /// Camera RGB of the scene light, green = 1, in the camera's raw scale.
    pub w_scene: Vec3,
    /// The same light as log-chroma `(ln G/R, ln G/B)`.
    pub chroma_uv: [f64; 2],
    /// Covariance of `chroma_uv` (the fitted Gaussian, with the as-shot prior if it was used).
    pub covariance_uv: [[f64; 2]; 2],
    /// The estimate from the histogram alone, before the as-shot prior.
    pub histogram_chroma_uv: [f64; 2],
    /// Probabilities over the six classes (they sum to 1).
    pub classes: EstimatorClasses,
    /// The probability that the angular error of `w_scene` is below 2 degrees, in 0 to 1.
    pub confidence: f64,
    /// Whether the as-shot white balance entered the estimate.
    pub prior_used: bool,
    /// True when the preview had no usable pixel and the result is the as-shot white (or the camera's D55) with
    /// confidence 0 and the prior class frequencies.
    pub fallback: bool,
}

impl IlluminantEstimate {
    /// The scene light for the colour chain: [`SceneLight::from_estimate`] with the profile's locus.
    ///
    /// # Errors
    ///
    /// As [`SceneLight::from_estimate`].
    pub fn scene_light(&self, locus: &crate::spectral::Locus) -> Result<SceneLight, WbError> {
        SceneLight::from_estimate(self.w_scene, &self.classes, self.confidence, locus)
    }

    /// The probabilities as an array in the order of [`CLASS_NAMES`].
    #[must_use]
    pub fn class_probabilities(&self) -> [f64; N_CLASSES] {
        let c = &self.classes;
        [
            c.daylight,
            c.shade,
            c.tungsten,
            c.fluorescent,
            c.led,
            c.mixed,
        ]
    }
}

/// Every intermediate of one estimate, for tests and diagnostics.
#[derive(Debug, Clone)]
pub struct DetailedEstimate {
    /// The result.
    pub estimate: IlluminantEstimate,
    /// The histograms (`None` for a fallback).
    pub histograms: Option<Histograms>,
    /// The filter scores `Z` (`n * n`, empty for a fallback).
    pub scores: Vec<f64>,
    /// The class head's features (empty for a fallback).
    pub features: Vec<f64>,
    /// Covariance of the histogram-only estimate.
    pub covariance_obs: [[f64; 2]; 2],
}

/// One trained model: the filter bank (kept as its FFT), the bias, the Gaussian-fit and prior constants and the head.
#[derive(Debug, Clone)]
pub struct IlluminantModel {
    parsed: Parsed,
    n: usize,
    channels: usize,
    filter_fft_re: Vec<f64>,
    filter_fft_im: Vec<f64>,
    fft: Fft2,
}

fn softmax_in_place(v: &mut [f64]) {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

impl IlluminantModel {
    /// Reads a model file (JSON text) and validates it, including its SHA-256.
    ///
    /// # Errors
    ///
    /// [`IllumError::Json`], [`IllumError::InvalidModel`] or [`IllumError::HashMismatch`].
    pub fn from_json_str(text: &str) -> Result<Self, IllumError> {
        let parsed = format::parse(text)?;
        let n = parsed.spec.n_bins;
        let channels = parsed.spec.channels.len();
        let fft = Fft2::new(n);
        let mut re = parsed.filter.clone();
        let mut im = vec![0.0; re.len()];
        for c in 0..channels {
            let r = c * n * n..(c + 1) * n * n;
            fft.forward(&mut re[r.clone()], &mut im[r]);
        }
        Ok(Self {
            parsed,
            n,
            channels,
            filter_fft_re: re,
            filter_fft_im: im,
            fft,
        })
    }

    /// Reads a model file from disk.
    ///
    /// # Errors
    ///
    /// [`IllumError::Io`] and those of [`IlluminantModel::from_json_str`].
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, IllumError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| IllumError::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        Self::from_json_str(&text)
    }

    /// Reads the model of a camera profile from the data folder: `<data_root>/models/illum/<profile_id>.json`.
    ///
    /// # Errors
    ///
    /// [`IllumError::InvalidModel`] for a profile id that is not a plain file name, [`IllumError::InvalidModel`] when the
    /// file belongs to another profile, and those of [`IlluminantModel::from_path`].
    pub fn load_for_profile(
        data_root: impl AsRef<Path>,
        profile_id: &str,
    ) -> Result<Self, IllumError> {
        let plain = !profile_id.is_empty()
            && profile_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if !plain {
            return Err(IllumError::InvalidModel(format!(
                "{profile_id:?} is not a profile id"
            )));
        }
        let model = Self::from_path(
            data_root
                .as_ref()
                .join("models/illum")
                .join(format!("{profile_id}.json")),
        )?;
        if model.profile_id() != profile_id {
            return Err(IllumError::InvalidModel(format!(
                "the model file of {profile_id} is for {}",
                model.profile_id()
            )));
        }
        Ok(model)
    }

    /// The camera profile this model was trained for (`sony-ilce-7rm4`).
    #[must_use]
    pub fn profile_id(&self) -> &str {
        &self.parsed.profile_id
    }

    /// The model's version string (goes into the export metadata).
    #[must_use]
    pub fn version(&self) -> &str {
        &self.parsed.version
    }

    /// SHA-256 of the tensors (hex), as recorded in the file and verified at load.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.parsed.sha256
    }

    /// Number of stored parameters (tensors and the three scalars `pad`, confidence scale and temperature), as
    /// counted by the training code.
    #[must_use]
    pub fn param_count(&self) -> u64 {
        self.parsed.param_count
    }

    /// The histogram geometry.
    #[must_use]
    pub fn histogram_spec(&self) -> &HistogramSpec {
        &self.parsed.spec
    }

    /// The camera white `[r, 1, b]` of the light `(u, v)`.
    #[must_use]
    pub fn white_of_uv(uv: [f64; 2]) -> Vec3 {
        [(-uv[0]).exp(), 1.0, (-uv[1]).exp()]
    }

    /// Estimates the scene light of a preview.
    ///
    /// # Errors
    ///
    /// [`IllumError::PreviewSize`] when the buffers do not match the stated size.
    pub fn estimate(
        &self,
        preview: &RawPreview<'_>,
        shot: &ShotInfo,
    ) -> Result<IlluminantEstimate, IllumError> {
        Ok(self.estimate_detailed(preview, shot)?.estimate)
    }

    /// As [`IlluminantModel::estimate`], with every intermediate.
    ///
    /// # Errors
    ///
    /// [`IllumError::PreviewSize`] when the buffers do not match the stated size.
    pub fn estimate_detailed(
        &self,
        preview: &RawPreview<'_>,
        shot: &ShotInfo,
    ) -> Result<DetailedEstimate, IllumError> {
        let hist = histogram::compute(&self.parsed.spec, preview)?;
        self.estimate_from_histograms(hist, shot.as_shot_uv(), shot.bv())
    }

    /// The estimate for ready histograms (the part after step 1). A non-finite `as_shot_uv` or `bv` counts as missing.
    ///
    /// # Errors
    ///
    /// [`IllumError::InvalidHistograms`] when `hist` does not hold this model's channels and bins (use the output of
    /// [`histogram::compute`] with [`IlluminantModel::histogram_spec`]).
    pub fn estimate_from_histograms(
        &self,
        hist: Histograms,
        as_shot_uv: Option<[f64; 2]>,
        bv: Option<f64>,
    ) -> Result<DetailedEstimate, IllumError> {
        let n = self.n;
        let channels = self.channels;
        if hist.data.len() != channels * n * n
            || hist.valid_fraction.len() != channels
            || hist.mass.len() != channels
        {
            return Err(IllumError::InvalidHistograms(format!(
                "expected {channels} channels of {n} x {n} bins with one valid fraction and one mass each, found {} values, {} valid fractions and {} masses",
                hist.data.len(),
                hist.valid_fraction.len(),
                hist.mass.len()
            )));
        }
        let as_shot_uv = as_shot_uv.filter(|a| a.iter().all(|v| v.is_finite()));
        let bv = bv.filter(|b| b.is_finite());
        if hist.mass.iter().all(|m| *m <= 0.0) {
            return Ok(self.fallback(as_shot_uv));
        }
        let scores = self.filter_scores(&hist.data);
        let (uv_obs, cov_obs, p_max) = self.gaussian_fit(&scores);
        let (uv_post, cov_post) = match as_shot_uv {
            Some(a) => self.fuse(uv_obs, cov_obs, a),
            None => (uv_obs, cov_obs),
        };
        let features = self.head_features(&hist, uv_obs, cov_obs, uv_post, p_max, as_shot_uv, bv);
        let probs = self.head_probs(&features);
        let confidence = self.confidence(&features, uv_obs, cov_obs, cov_post, as_shot_uv);
        debug_assert_eq!(scores.len(), n * n);
        Ok(DetailedEstimate {
            estimate: IlluminantEstimate {
                w_scene: Self::white_of_uv(uv_post),
                chroma_uv: uv_post,
                covariance_uv: cov_post,
                histogram_chroma_uv: uv_obs,
                classes: EstimatorClasses {
                    daylight: probs[0],
                    shade: probs[1],
                    tungsten: probs[2],
                    fluorescent: probs[3],
                    led: probs[4],
                    mixed: probs[5],
                },
                confidence: if confidence.is_finite() {
                    confidence.clamp(0.0, 1.0)
                } else {
                    0.0
                },
                prior_used: as_shot_uv.is_some(),
                fallback: false,
            },
            histograms: Some(hist),
            scores,
            features,
            covariance_obs: cov_obs,
        })
    }

    fn fallback(&self, as_shot_uv: Option<[f64; 2]>) -> DetailedEstimate {
        let uv = as_shot_uv.unwrap_or(self.parsed.default_uv);
        let var = self.parsed.prior_sigma * self.parsed.prior_sigma;
        let p = self.parsed.class_prior;
        DetailedEstimate {
            estimate: IlluminantEstimate {
                w_scene: Self::white_of_uv(uv),
                chroma_uv: uv,
                covariance_uv: [[var, 0.0], [0.0, var]],
                histogram_chroma_uv: uv,
                classes: EstimatorClasses {
                    daylight: p[0],
                    shade: p[1],
                    tungsten: p[2],
                    fluorescent: p[3],
                    led: p[4],
                    mixed: p[5],
                },
                confidence: 0.0,
                prior_used: as_shot_uv.is_some(),
                fallback: true,
            },
            histograms: None,
            scores: Vec::new(),
            features: Vec::new(),
            covariance_obs: [[var, 0.0], [0.0, var]],
        }
    }

    /// `Z = sum_c ifft2(fft2(X_c) * fft2(F_c)) + B`.
    fn filter_scores(&self, data: &[f64]) -> Vec<f64> {
        let n = self.n;
        let nn = n * n;
        let mut acc_re = vec![0.0; nn];
        let mut acc_im = vec![0.0; nn];
        let mut re = vec![0.0; nn];
        let mut im = vec![0.0; nn];
        for c in 0..self.channels {
            re.copy_from_slice(&data[c * nn..(c + 1) * nn]);
            im.iter_mut().for_each(|v| *v = 0.0);
            self.fft.forward(&mut re, &mut im);
            let (fr, fi) = (
                &self.filter_fft_re[c * nn..(c + 1) * nn],
                &self.filter_fft_im[c * nn..(c + 1) * nn],
            );
            for k in 0..nn {
                acc_re[k] += re[k] * fr[k] - im[k] * fi[k];
                acc_im[k] += re[k] * fi[k] + im[k] * fr[k];
            }
        }
        self.fft.inverse(&mut acc_re, &mut acc_im);
        acc_re
            .iter()
            .zip(&self.parsed.bias)
            .map(|(z, b)| z + b)
            .collect()
    }

    /// Softmax and moment fit: `(mean, covariance, max P)`.
    fn gaussian_fit(&self, scores: &[f64]) -> ([f64; 2], [[f64; 2]; 2], f64) {
        let n = self.n;
        let mut p = scores.to_vec();
        softmax_in_place(&mut p);
        let off = 0.5 * (n - 1) as f64;
        let k: Vec<f64> = (0..n).map(|i| i as f64 - off).collect();
        let mut pi = vec![0.0; n];
        let mut pj = vec![0.0; n];
        let mut eij = 0.0;
        let mut p_max = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let v = p[i * n + j];
                pi[i] += v;
                pj[j] += v;
                eij += v * k[i] * k[j];
                p_max = p_max.max(v);
            }
        }
        let (mut mi, mut mj, mut eii, mut ejj) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            mi += pi[i] * k[i];
            mj += pj[i] * k[i];
            eii += pi[i] * (k[i] * k[i]);
            ejj += pj[i] * (k[i] * k[i]);
        }
        let b = self.parsed.spec.bin_size;
        let pad = self.parsed.pad;
        let cov = [
            [(eii - mi * mi) * b * b + pad, (eij - mi * mj) * b * b],
            [(eij - mi * mj) * b * b, (ejj - mj * mj) * b * b + pad],
        ];
        let uv = [
            self.parsed.spec.lo_u + (mi + off) * b,
            self.parsed.spec.lo_v + (mj + off) * b,
        ];
        (uv, cov, p_max)
    }

    /// The as-shot Gaussian times the estimate: `(mean, covariance)`.
    fn fuse(&self, mu: [f64; 2], cov: [[f64; 2]; 2], a: [f64; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
        let (l00, l11, l01) = (cov[1][1] / det, cov[0][0] / det, -cov[0][1] / det);
        let wp = 1.0 / (self.parsed.prior_sigma * self.parsed.prior_sigma);
        let (t00, t11, t01) = (l00 + wp, l11 + wp, l01);
        let dt = t00 * t11 - t01 * t01;
        let (p00, p11, p01) = (t11 / dt, t00 / dt, -t01 / dt);
        let r0 = l00 * mu[0] + l01 * mu[1] + wp * a[0];
        let r1 = l01 * mu[0] + l11 * mu[1] + wp * a[1];
        (
            [p00 * r0 + p01 * r1, p01 * r0 + p11 * r1],
            [[p00, p01], [p01, p11]],
        )
    }

    /// Per channel `(mean_u, mean_v, std_u, std_v, corr, entropy)` of one normalised histogram (zeros when empty).
    fn moments(&self, hist: &[f64]) -> [f64; 6] {
        let n = self.n;
        let off = 0.5 * (n - 1) as f64;
        let k: Vec<f64> = (0..n).map(|i| i as f64 - off).collect();
        let mut pi = vec![0.0; n];
        let mut pj = vec![0.0; n];
        let mut eij_raw = 0.0;
        let mut ent_raw = 0.0;
        for i in 0..n {
            for j in 0..n {
                let v = hist[i * n + j];
                pi[i] += v;
                pj[j] += v;
                eij_raw += v * k[i] * k[j];
                if v > 0.0 {
                    ent_raw += v * v.ln();
                }
            }
        }
        let mass: f64 = pi.iter().sum();
        if mass <= 0.0 {
            return [0.0; 6];
        }
        let (mut mi, mut mj, mut ei2, mut ej2) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            mi += pi[i] * k[i];
            mj += pj[i] * k[i];
            ei2 += pi[i] * (k[i] * k[i]);
            ej2 += pj[i] * (k[i] * k[i]);
        }
        let (mi, mj) = (mi / mass, mj / mass);
        let vi = (ei2 / mass - mi * mi).max(0.0);
        let vj = (ej2 / mass - mj * mj).max(0.0);
        let eij = eij_raw / mass;
        let (si, sj) = (vi.sqrt(), vj.sqrt());
        let corr = if si > 0.0 && sj > 0.0 {
            (eij - mi * mj) / (si * sj).max(1e-300)
        } else {
            0.0
        };
        let ent = -ent_raw / mass + mass.ln();
        let b = self.parsed.spec.bin_size;
        [
            self.parsed.spec.lo_u + (mi + off) * b,
            self.parsed.spec.lo_v + (mj + off) * b,
            si * b,
            sj * b,
            corr,
            ent,
        ]
    }

    #[allow(clippy::too_many_arguments)]
    fn head_features(
        &self,
        hist: &Histograms,
        mu_obs: [f64; 2],
        cov_obs: [[f64; 2]; 2],
        mu_post: [f64; 2],
        p_max: f64,
        as_shot_uv: Option<[f64; 2]>,
        bv: Option<f64>,
    ) -> Vec<f64> {
        let (a_u, a_v) = as_shot_uv.map_or((mu_obs[0], mu_obs[1]), |a| (a[0], a[1]));
        let has_a = f64::from(u8::from(as_shot_uv.is_some()));
        let det = cov_obs[0][0] * cov_obs[1][1] - cov_obs[0][1] * cov_obs[0][1];
        let mut f = vec![
            mu_post[0],
            mu_post[1],
            a_u,
            a_v,
            has_a,
            if as_shot_uv.is_some() {
                mu_obs[0] - a_u
            } else {
                0.0
            },
            if as_shot_uv.is_some() {
                mu_obs[1] - a_v
            } else {
                0.0
            },
            0.5 * det.ln(),
            bv.unwrap_or(self.parsed.bv_default),
            f64::from(u8::from(bv.is_some())),
            p_max.ln(),
        ];
        let nn = self.n * self.n;
        for c in 0..self.channels {
            let m = self.moments(&hist.data[c * nn..(c + 1) * nn]);
            f.extend_from_slice(&[
                m[0] - mu_post[0],
                m[1] - mu_post[1],
                m[2],
                m[3],
                m[4],
                m[5],
                hist.valid_fraction[c],
            ]);
        }
        f
    }

    /// The probability that the angular error is under 2 degrees: a logistic regression on the class-head features and the
    /// Mahalanobis distance of the histogram estimate from the as-shot value, the log variance of the fused Gaussian and the
    /// log variance of the histogram-only Gaussian.
    fn confidence(
        &self,
        features: &[f64],
        mu_obs: [f64; 2],
        cov_obs: [[f64; 2]; 2],
        cov_post: [[f64; 2]; 2],
        as_shot_uv: Option<[f64; 2]>,
    ) -> f64 {
        let (s0, s1, s01) = (cov_obs[0][0], cov_obs[1][1], cov_obs[0][1]);
        let det = s0 * s1 - s01 * s01;
        let maha = match as_shot_uv {
            Some(a) => {
                let (d0, d1) = (mu_obs[0] - a[0], mu_obs[1] - a[1]);
                ((s1 * d0 * d0 - 2.0 * s01 * d0 * d1 + s0 * d1 * d1) / det)
                    .max(0.0)
                    .sqrt()
            }
            None => 0.0,
        };
        let extras = [
            maha,
            (0.5 * (cov_post[0][0] + cov_post[1][1])).ln(),
            (0.5 * (s0 + s1)).ln(),
        ];
        let p = &self.parsed;
        let n = features.len();
        let mut z = p.conf_w[n + extras.len()];
        for (i, x) in features.iter().chain(&extras).enumerate() {
            z += ((x - p.conf_mean[i]) / p.conf_scale[i]) * p.conf_w[i];
        }
        1.0 / (1.0 + (-z).exp())
    }

    fn head_probs(&self, features: &[f64]) -> [f64; N_CLASSES] {
        let p = &self.parsed;
        let z: Vec<f64> = features
            .iter()
            .zip(p.feature_mean.iter().zip(&p.feature_scale))
            .map(|(f, (m, s))| (f - m) / s)
            .collect();
        let hidden: Vec<f64> = (0..p.hidden)
            .map(|h| {
                let row = &p.w1[h * p.n_features..(h + 1) * p.n_features];
                (row.iter().zip(&z).map(|(w, x)| w * x).sum::<f64>() + p.b1[h]).tanh()
            })
            .collect();
        let mut logits = [0.0; N_CLASSES];
        for (k, l) in logits.iter_mut().enumerate() {
            let row = &p.w2[k * p.hidden..(k + 1) * p.hidden];
            *l = (row.iter().zip(&hidden).map(|(w, x)| w * x).sum::<f64>() + p.b2[k])
                / p.temperature;
        }
        softmax_in_place(&mut logits);
        logits
    }
}
