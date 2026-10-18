//! Auto exposure (`docs/02-colour-pipeline.md`, section "Auto exposure"), burst consistency and the
//! comparison with the camera's embedded JPEG that gates it (M1-08).
//!
//! Everything here is a pure function over plain buffers and small metadata structs: no decoder, no
//! GPU, no file access.
//!
//! # The rule (interim, decision of 2026-10-14)
//!
//! Until the intelligent automatic settings of a later milestone replace it, **auto exposure is the
//! camera's own exposure**: no metering of our own moves it. Step by step ([`auto_exposure`] is given
//! the fused matrix of [`crate::chain`]):
//!
//! 1. Run C1 to C6 at EV 0 on the 1/8-resolution preview and take the linear luminance `Y` with the
//!    AWG4 `Y` row.
//! 2. Ignore pixels in the clip mask, and pixels whose `Y` is below the 1st or above the 99th
//!    percentile of the unmasked pixels.
//! 3. `m = exp(sum w ln Y / sum w)` with the centre weight `w = exp(-r^2 / (2 * 0.35^2))`, `r` the
//!    distance from the centre over the half diagonal. This statistic is only reported
//!    ([`AutoExposure::mean_luminance`]); it does not move the exposure.
//! 4. `EV_auto = clamp(CAMERA_GAIN_EV, -3, +3)`: the median gain of the cameras' own JPEG engines (raw
//!    level to JPEG mid grey, +1.11 EV over 12 photos from 9 Sony bodies). There is **no
//!    exposure-compensation term**: the raw data already carries the photographer's compensation (the
//!    camera exposed that much more or less), so with a constant gain it reaches the picture in full,
//!    as it does in the camera's JPEG.
//! 5. Highlight guard: if more than 0.5 % of the unmasked pixels would exceed linear 8.0 after the
//!    exposure, lower `EV_auto` until they do not, by at most 1 EV. It can only lower the exposure.
//!    8.0 is 5.47 stops over middle grey, about the clip of an ALEXA (ALEV III) at EI 180, not at
//!    EI 800 (7.64 stops, linear 35.9: the cap of C7's highlight range); the level is kept as it is.
//! 6. Frames of the same camera within 2 s of each other and with identical exposure settings share the
//!    median `EV_auto` ([`share_burst_ev`]); with a constant gain that is the constant unless the guard
//!    acted on some frame.
//!
//! # Decisions where the spec is silent
//!
//! * *Percentiles* are the linearly interpolated quantiles of the sorted unmasked luminances (numpy's
//!   default); a pixel is kept when `q01 <= Y <= q99`. The mask is applied first, so clipped pixels
//!   do not move the percentiles. When no pixel lies in the band (two distinct values and nothing
//!   else) every unmasked pixel is metered.
//! * *Luminance floor.* `Y` is raised to [`LUMINANCE_FLOOR`] (1e-4, ten and a half stops below
//!   middle grey) before the logarithm of the reported statistic, so black or negative pixels (noise
//!   after black subtraction) are very dark pixels rather than missing ones.
//! * *Which pixels count.* A pixel with a non-finite channel counts as masked. The highlight guard and
//!   its 0.5 % are over every unmasked pixel (not only those kept by the percentiles, which would remove
//!   the highlights the guard is looking for), and compare `Y` after exposure with 8.0.
//! * *The guard* is solved in closed form: the largest EV at which at most `floor(0.005 n)` pixels have
//!   `Y * 2^EV > 8`, from the `(floor(0.005 n) + 1)`-th brightest luminance; the result is
//!   `max(that, EV_before_guard - 1)` when it is below `EV_before_guard`.
//! * *No usable pixel* (everything masked, or an empty preview): nothing can be reported, and the
//!   exposure is the same camera gain, `EV_auto = clamp(CAMERA_GAIN_EV)`; [`AutoExposure::metered`] is
//!   false.
//! * *Bursts.* A frame joins a group when the previous frame of the same camera and identical exposure
//!   settings is at most 2 s earlier (consecutive gaps, so a 10 fps burst of any length is one group).
//!   The median of an even group is the mean of the two middle values. Frames without a capture time
//!   are alone.
//!
//! # The gate
//!
//! "Within 0.5 EV of the camera's embedded JPEG (median luminance, compared in linear light) for at least
//! 80 % of the golden set." [`preview_median_luminance`] gives the median `Y` of our render at an
//! exposure, [`jpeg_median_luminance`] the median `Y` of the decoded JPEG (sRGB decoded to linear light,
//! Rec.709 luminance), [`exposure_error_ev`] their ratio in stops and [`evaluate_gate`] the pass rate and
//! the outliers for review. Decoding the JPEG and reading the files is the caller's job.
//!
//! # The white lift
//!
//! [`white`] holds the other automatic estimate that is made on the same preview: the per-photo lift of
//! the tones above +1 stop over middle grey (`docs/02`, C12 step 1b), its scene-linear stage
//! ([`white::WhiteLift`]) and its estimator ([`white::estimate_white`]).

pub mod white;

use serde::{Deserialize, Serialize};

use crate::consts::linalg::{Mat3, Vec3};
use crate::gamut::awg4_luminance_row;

/// The camera's gain in stops: the exposure `EV_auto` of every photo until the intelligent automatic settings
/// of a later milestone replace the rule (`docs/02`, Auto exposure).
///
/// It is the median gain of the cameras' own JPEG engines, `log2(0.18 / x)` for the raw level `x` that the
/// embedded JPEG renders as middle grey in linear light, over the 12 local evaluation photos from 9 Sony bodies
/// (std 0.13 EV, range +0.87 to +1.33; `tools/eval/autoexp.py`, `research/auto-exposure-2026-10-14.md`). An ideal
/// ISO 12232 camera would put grey at +0.79; Sony places it higher.
///
/// The scene does not move it and neither does the camera's exposure compensation: the raw data already holds
/// that compensation, so it reaches the picture in full.
pub const CAMERA_GAIN_EV: f64 = 1.11;
/// Scene-linear middle grey: the reference of the reported key statistic ([`AutoExposure::mean_luminance`]) and
/// of the camera gain ([`CAMERA_GAIN_EV`] is measured against it).
pub const TARGET_GREY: f64 = 0.18;
/// Standard deviation of the centre weight, in units of the half diagonal.
pub const CENTRE_SIGMA: f64 = 0.35;
/// `EV_auto` is clamped to plus or minus this many stops before the highlight guard.
pub const EV_LIMIT: f64 = 3.0;
/// Pixels below this quantile of the unmasked luminances are ignored.
pub const PERCENTILE_LOW: f64 = 0.01;
/// Pixels above this quantile of the unmasked luminances are ignored.
pub const PERCENTILE_HIGH: f64 = 0.99;
/// The highlight guard's linear level after exposure: 5.47 stops over middle grey, about the clip of an
/// ALEXA at EI 180, not at EI 800 (whose clip, 7.64 stops, is [`crate::tone::HIGHLIGHT_CAP_STOPS`]).
pub const GUARD_LEVEL: f64 = 8.0;
/// The highlight guard's fraction of unmasked pixels allowed above [`GUARD_LEVEL`].
pub const GUARD_FRACTION: f64 = 0.005;
/// The most the highlight guard lowers `EV_auto`, in stops.
pub const GUARD_MAX_REDUCTION_EV: f64 = 1.0;
/// The largest gap between consecutive frames of a burst, in seconds.
pub const BURST_WINDOW_S: f64 = 2.0;
/// Luminance floor before the logarithm.
pub const LUMINANCE_FLOOR: f64 = 1e-4;
/// Lowest absolute exposure of the Exposure control (stops).
pub const EXPOSURE_MIN_EV: f64 = -4.0;
/// Highest absolute exposure of the Exposure control (stops).
pub const EXPOSURE_MAX_EV: f64 = 4.0;
/// The gate's tolerance: stops between our median luminance and the JPEG's.
pub const GATE_TOLERANCE_EV: f64 = 0.5;
/// The gate's pass rate: the fraction of the golden set that must be within the tolerance.
pub const GATE_REQUIRED_FRACTION: f64 = 0.8;

/// Errors from auto exposure and the gate.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ExposureError {
    /// The pixel buffer is not `width * height * 3` values.
    #[error("a {width} x {height} preview needs {expected} RGB values, found {found}")]
    BufferSize {
        /// Preview width in pixels.
        width: usize,
        /// Preview height in pixels.
        height: usize,
        /// `width * height * 3`.
        expected: usize,
        /// The length given.
        found: usize,
    },
    /// The clip mask is not `width * height` values.
    #[error("a {width} x {height} preview needs a mask of {expected} values, found {found}")]
    MaskSize {
        /// Preview width in pixels.
        width: usize,
        /// Preview height in pixels.
        height: usize,
        /// `width * height`.
        expected: usize,
        /// The length given.
        found: usize,
    },
    /// An 8-bit image buffer is not a whole number of RGB pixels.
    #[error("an RGB image needs a multiple of 3 values, found {0}")]
    NotRgb(usize),
    /// A manual exposure outside the control's range, or not a number.
    #[error("exposure {0} EV must be a number from -4 to 4")]
    EvOutOfRange(f64),
}

/// The `exposure` setting: Auto, or an absolute value in stops (`docs/01`).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExposureSetting {
    /// `EV_auto`.
    #[default]
    Auto,
    /// An absolute exposure, -4 to +4 stops.
    Manual {
        /// Stops.
        ev: f64,
    },
}

impl ExposureSetting {
    /// A manual exposure after checking the control's range (-4 to +4 stops, `docs/01`).
    ///
    /// # Errors
    ///
    /// [`ExposureError::EvOutOfRange`] for a value outside the range or not a number.
    pub fn manual(ev: f64) -> Result<Self, ExposureError> {
        let setting = ExposureSetting::Manual { ev };
        setting.validate()?;
        Ok(setting)
    }

    /// Checks the range of a manual exposure (Auto is always valid).
    ///
    /// # Errors
    ///
    /// [`ExposureError::EvOutOfRange`] for a value outside -4 to +4 or not a number.
    pub fn validate(&self) -> Result<(), ExposureError> {
        if let ExposureSetting::Manual { ev } = *self
            && !(EXPOSURE_MIN_EV..=EXPOSURE_MAX_EV).contains(&ev)
        {
            return Err(ExposureError::EvOutOfRange(ev));
        }
        Ok(())
    }

    /// The exposure in stops, given the automatic one.
    #[must_use]
    pub fn resolve(self, ev_auto: f64) -> f64 {
        match self {
            ExposureSetting::Auto => ev_auto,
            ExposureSetting::Manual { ev } => ev,
        }
    }
}

/// The 1/8-resolution preview that auto exposure meters: g0-balanced camera RGB, interleaved, and the
/// clip mask.
#[derive(Debug, Clone, Copy)]
pub struct PreviewRgb<'a> {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` values, `R G B` per pixel, row by row.
    pub rgb: &'a [f32],
    /// One value per pixel, non-zero where the pixel is in the clip mask; `None` for no mask.
    pub clip_mask: Option<&'a [u8]>,
}

impl PreviewRgb<'_> {
    pub(crate) fn validate(&self) -> Result<(), ExposureError> {
        // Saturating: a size that overflows `usize` can never equal a slice length, so it is refused
        // as a wrong size instead of wrapping (release) or panicking (debug).
        let pixels = self.width.saturating_mul(self.height);
        let expected = pixels.saturating_mul(3);
        if self.rgb.len() != expected {
            return Err(ExposureError::BufferSize {
                width: self.width,
                height: self.height,
                expected,
                found: self.rgb.len(),
            });
        }
        if let Some(mask) = self.clip_mask
            && mask.len() != pixels
        {
            return Err(ExposureError::MaskSize {
                width: self.width,
                height: self.height,
                expected: pixels,
                found: mask.len(),
            });
        }
        Ok(())
    }

    /// The luminance row of the camera RGB, `Y_awg4 * M`, so one dot product per pixel gives `Y`.
    pub(crate) fn luminance_row(matrix_ev0: &Mat3) -> Vec3 {
        let y = awg4_luminance_row();
        std::array::from_fn(|c| {
            y[0] * matrix_ev0[0][c] + y[1] * matrix_ev0[1][c] + y[2] * matrix_ev0[2][c]
        })
    }

    /// `Y` of every pixel at EV 0, or `None` where the pixel is masked or not finite.
    fn luminances(&self, matrix_ev0: &Mat3) -> Vec<Option<f64>> {
        let row = Self::luminance_row(matrix_ev0);
        (0..self.width * self.height)
            .map(|i| {
                if self.clip_mask.is_some_and(|m| m[i] != 0) {
                    return None;
                }
                let p = &self.rgb[3 * i..3 * i + 3];
                let (r, g, b) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
                let y = row[0] * r + row[1] * g + row[2] * b;
                (r.is_finite() && g.is_finite() && b.is_finite() && y.is_finite()).then_some(y)
            })
            .collect()
    }
}

/// The result of [`auto_exposure`] for one photo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoExposure {
    /// `EV_auto` in stops after the highlight guard (before burst sharing).
    pub ev_auto: f64,
    /// `EV_auto` of step 4, before the highlight guard: [`CAMERA_GAIN_EV`] (clamped to the limit).
    pub ev_before_guard: f64,
    /// How far the highlight guard lowered the exposure, 0 to 1 stop.
    pub guard_reduction_ev: f64,
    /// The centre-weighted geometric mean luminance `m` at EV 0 (0 when nothing could be metered). A
    /// diagnostic: it does not move `ev_auto`.
    pub mean_luminance: f64,
    /// Pixels that were not masked and finite.
    pub unmasked_pixels: usize,
    /// Pixels inside the percentile band that entered the mean.
    pub metered_pixels: usize,
    /// False when there was no usable pixel (so `mean_luminance` is 0 and the highlight guard did not run).
    pub metered: bool,
}

/// Linearly interpolated quantile of a sorted slice (numpy's default); `q` in 0 to 1. The interpolation
/// is [`interpolate`].
pub(crate) fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    interpolate(sorted[lo], sorted[hi], pos - lo as f64)
}

/// The interpolation between the neighbours `a <= b` at `frac` (0 to 1) of a quantile: `a + (b - a) * frac`,
/// written from `b` for `frac >= 0.5` as numpy does, and kept inside `[a, b]`.
///
/// The form `(1 - frac) * a + frac * b` is off by an ulp for some equal neighbours, and a percentile that
/// lands one ulp above a run of equal luminances (a flat or black frame, the floor value of many dark
/// pixels) would drop that whole run from the band. This returns exactly `a` when `a == b`. The selection
/// quantiles of `autoset` share it, so they agree with [`quantile`] bit for bit.
pub(crate) fn interpolate(a: f64, b: f64, frac: f64) -> f64 {
    let v = if frac >= 0.5 {
        b - (b - a) * (1.0 - frac)
    } else {
        a + (b - a) * frac
    };
    v.clamp(a, b)
}

/// The measured part of auto exposure: steps 1 to 3 of the rule, and the pixels the highlight guard (step 5) reads.
///
/// [`measure_exposure`] makes it from a preview once. The exposure is then decided by [`Measurement::guarded`] from
/// the step-4 value: the camera gain of the rule, or the value the learned model asks for (`docs/18` section 4.4),
/// raised to the camera's exposure label when that is higher (section 6.1, `ym_core::autoset::camera::camera_floor`).
/// The model needs the mean luminance this measurement reports, so the two steps are kept apart rather than
/// metering the preview twice. [`auto_exposure`] is the rule alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// `Y` of every unmasked pixel, floored at [`LUMINANCE_FLOOR`], sorted ascending. Empty when nothing could be
    /// metered.
    sorted: Vec<f64>,
    /// The centre-weighted geometric mean luminance at EV 0 (0 when nothing could be metered).
    mean_luminance: f64,
    /// Pixels inside the percentile band that entered the mean.
    metered_pixels: usize,
}

/// Steps 1 to 3 of the rule for a preview (see [`auto_exposure`]): the luminances, the percentile band and the
/// centre-weighted mean. None of it depends on the exposure.
///
/// `matrix_ev0` is the fused C1 to C6 matrix at EV 0 (`FusedChain::matrix` of a chain built with
/// `exposure_ev = 0`).
///
/// # Errors
///
/// [`ExposureError::BufferSize`] or [`ExposureError::MaskSize`] for buffers of the wrong length.
pub fn measure_exposure(
    preview: &PreviewRgb<'_>,
    matrix_ev0: &Mat3,
) -> Result<Measurement, ExposureError> {
    preview.validate()?;
    let luminances = preview.luminances(matrix_ev0);
    let mut sorted: Vec<f64> = luminances
        .iter()
        .flatten()
        .map(|y| y.max(LUMINANCE_FLOOR))
        .collect();
    let n = sorted.len();
    if n == 0 {
        return Ok(Measurement {
            sorted,
            mean_luminance: 0.0,
            metered_pixels: 0,
        });
    }
    sorted.sort_unstable_by(f64::total_cmp);
    let (low, high) = (
        quantile(&sorted, PERCENTILE_LOW),
        quantile(&sorted, PERCENTILE_HIGH),
    );

    // Step 3: centre-weighted geometric mean over the band (reported, not used for the exposure).
    let (w, h) = (preview.width as f64, preview.height as f64);
    let half_diagonal_sq = 0.25 * (w * w + h * h);
    let weighted_log_sum = |banded: bool| {
        let mut sum_w = 0.0;
        let mut sum_wl = 0.0;
        let mut count = 0;
        for (i, y) in luminances.iter().enumerate() {
            let Some(y) = *y else { continue };
            let y = y.max(LUMINANCE_FLOOR);
            if banded && (y < low || y > high) {
                continue;
            }
            let (px, py) = (
                (i % preview.width) as f64 + 0.5 - 0.5 * w,
                (i / preview.width) as f64 + 0.5 - 0.5 * h,
            );
            let r_sq = (px * px + py * py) / half_diagonal_sq;
            let weight = (-r_sq / (2.0 * CENTRE_SIGMA * CENTRE_SIGMA)).exp();
            sum_w += weight;
            sum_wl += weight * y.ln();
            count += 1;
        }
        (sum_w, sum_wl, count)
    };
    let (mut sum_w, mut sum_wl, mut metered_pixels) = weighted_log_sum(true);
    if metered_pixels == 0 {
        // Nothing lies between the 1st and 99th percentile: only possible with two distinct values
        // (a preview of two usable pixels). Meter every usable pixel rather than divide 0 by 0.
        (sum_w, sum_wl, metered_pixels) = weighted_log_sum(false);
    }
    let mean_luminance = (sum_wl / sum_w).exp();
    Ok(Measurement {
        sorted,
        mean_luminance,
        metered_pixels,
    })
}

impl Measurement {
    /// Pixels that were not masked and are finite (`AutoExposure::unmasked_pixels`).
    #[must_use]
    pub fn unmasked_pixels(&self) -> usize {
        self.sorted.len()
    }

    /// Whether anything could be metered. False means there is no mean luminance and the guard has nothing to read
    /// (`AutoExposure::metered`).
    #[must_use]
    pub fn metered(&self) -> bool {
        !self.sorted.is_empty()
    }

    /// The centre-weighted geometric mean luminance at EV 0, 0 when nothing could be metered (`docs/02`, step 3).
    #[must_use]
    pub fn mean_luminance(&self) -> f64 {
        self.mean_luminance
    }

    /// Pixels inside the percentile band that entered the mean (`AutoExposure::metered_pixels`).
    #[must_use]
    pub fn metered_pixels(&self) -> usize {
        self.metered_pixels
    }

    /// Step 5, the highlight guard, on the step-4 exposure `ev_before_guard` (stops; the camera gain of the rule or
    /// the model's value, already clamped to plus or minus [`EV_LIMIT`]). The result is the largest exposure at which
    /// at most `floor(0.005 n)` unmasked pixels exceed [`GUARD_LEVEL`], and no lower than `ev_before_guard` minus one
    /// stop; it never raises the exposure. With nothing metered the exposure is `ev_before_guard` itself.
    #[must_use]
    pub fn guarded(&self, ev_before_guard: f64) -> AutoExposure {
        let n = self.sorted.len();
        if n == 0 {
            return AutoExposure {
                ev_auto: ev_before_guard,
                ev_before_guard,
                guard_reduction_ev: 0.0,
                mean_luminance: 0.0,
                unmasked_pixels: 0,
                metered_pixels: 0,
                metered: false,
            };
        }
        // Step 5: the largest EV with at most floor(0.005 n) pixels above the guard level.
        let allowed = (GUARD_FRACTION * n as f64).floor() as usize;
        let threshold_y = self.sorted[n - 1 - allowed.min(n - 1)];
        let ev_guard = (GUARD_LEVEL / threshold_y).log2();
        let ev_auto = if ev_before_guard > ev_guard {
            ev_guard.max(ev_before_guard - GUARD_MAX_REDUCTION_EV)
        } else {
            ev_before_guard
        };
        AutoExposure {
            ev_auto,
            ev_before_guard,
            guard_reduction_ev: ev_before_guard - ev_auto,
            mean_luminance: self.mean_luminance,
            unmasked_pixels: n,
            metered_pixels: self.metered_pixels,
            metered: true,
        }
    }
}

/// The automatic exposure of a preview: steps 1 to 5 of the spec. `EV_auto` is [`CAMERA_GAIN_EV`], lowered by the
/// highlight guard when the picture would clip; the centre-weighted geometric mean is reported and moves nothing.
///
/// `matrix_ev0` is the fused C1 to C6 matrix at EV 0 (`FusedChain::matrix` of a chain built with
/// `exposure_ev = 0`). The camera's exposure compensation is not an input: the raw already holds it and it
/// reaches the picture in full.
///
/// # Errors
///
/// [`ExposureError::BufferSize`] or [`ExposureError::MaskSize`] for buffers of the wrong length.
pub fn auto_exposure(
    preview: &PreviewRgb<'_>,
    matrix_ev0: &Mat3,
) -> Result<AutoExposure, ExposureError> {
    // Step 4: the camera's gain, whatever the scene.
    let measurement = measure_exposure(preview, matrix_ev0)?;
    Ok(measurement.guarded(CAMERA_GAIN_EV.clamp(-EV_LIMIT, EV_LIMIT)))
}

/// The median luminance of the preview at an exposure: `Y` at EV 0 times `2^ev` over **every** pixel
/// (masked pixels included, as the JPEG's median includes them). `None` for an empty preview.
///
/// # Errors
///
/// [`ExposureError::BufferSize`] for a buffer of the wrong length.
pub fn preview_median_luminance(
    preview: &PreviewRgb<'_>,
    matrix_ev0: &Mat3,
    ev: f64,
) -> Result<Option<f64>, ExposureError> {
    let all = PreviewRgb {
        clip_mask: None,
        ..*preview
    };
    all.validate()?;
    let gain = ev.exp2();
    let mut ys: Vec<f64> = all
        .luminances(matrix_ev0)
        .into_iter()
        .flatten()
        .map(|y| y * gain)
        .collect();
    Ok(median(&mut ys))
}

/// The median of a list (the mean of the two middle values for an even count); `None` when empty.
fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    })
}

/// The sRGB electro-optical transfer function (IEC 61966-2-1): code value 0 to 1 to linear light.
#[must_use]
pub fn srgb_to_linear(code: f64) -> f64 {
    if code <= 0.04045 {
        code / 12.92
    } else {
        ((code + 0.055) / 1.055).powf(2.4)
    }
}

/// The median linear luminance of an 8-bit sRGB image (`R G B` interleaved): each channel decoded to
/// linear light, then `Y = 0.2126 R + 0.7152 G + 0.0722 B`. `None` for an empty image.
///
/// # Errors
///
/// [`ExposureError::NotRgb`] when the length is not a multiple of 3.
pub fn jpeg_median_luminance(rgb8: &[u8]) -> Result<Option<f64>, ExposureError> {
    if !rgb8.len().is_multiple_of(3) {
        return Err(ExposureError::NotRgb(rgb8.len()));
    }
    let table: Vec<f64> = (0..=255_u16)
        .map(|v| srgb_to_linear(f64::from(v) / 255.0))
        .collect();
    let mut ys: Vec<f64> = rgb8
        .chunks_exact(3)
        .map(|p| {
            0.2126 * table[usize::from(p[0])]
                + 0.7152 * table[usize::from(p[1])]
                + 0.0722 * table[usize::from(p[2])]
        })
        .collect();
    Ok(median(&mut ys))
}

/// Stops between our median luminance and the reference's: `log2(ours / reference)`. Positive when we
/// are brighter than the JPEG. Both must be positive.
#[must_use]
pub fn exposure_error_ev(ours: f64, reference: f64) -> f64 {
    (ours / reference).log2()
}

/// One photo of the golden set in the gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GateEntry {
    /// The photo's name, for the outlier list.
    pub name: String,
    /// Median luminance of our render at `EV_auto` ([`preview_median_luminance`]).
    pub ours_median_y: f64,
    /// Median luminance of the embedded JPEG ([`jpeg_median_luminance`]).
    pub reference_median_y: f64,
}

/// An entry outside the tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutlier {
    /// The photo's name.
    pub name: String,
    /// `log2(ours / reference)` in stops.
    pub error_ev: f64,
}

/// The gate's result.
#[derive(Debug, Clone, PartialEq)]
pub struct GateReport {
    /// Photos evaluated (entries with a positive, finite pair of medians).
    pub evaluated: usize,
    /// Photos within the tolerance.
    pub within: usize,
    /// `within / evaluated` (0 for an empty set).
    pub fraction: f64,
    /// Whether the fraction reaches the required one.
    pub passes: bool,
    /// Photos outside the tolerance, worst first, for review.
    pub outliers: Vec<GateOutlier>,
    /// Names of entries that could not be evaluated (a median that is not positive and finite).
    pub skipped: Vec<String>,
}

/// Evaluates the gate: the fraction of entries whose median luminance is within `tolerance_ev` stops of
/// the reference, against `required_fraction`; the outliers are listed worst first.
#[must_use]
pub fn evaluate_gate(
    entries: &[GateEntry],
    tolerance_ev: f64,
    required_fraction: f64,
) -> GateReport {
    let mut outliers = Vec::new();
    let mut skipped = Vec::new();
    let mut evaluated = 0;
    let mut within = 0;
    for e in entries {
        let usable = |v: f64| v.is_finite() && v > 0.0;
        if !(usable(e.ours_median_y) && usable(e.reference_median_y)) {
            skipped.push(e.name.clone());
            continue;
        }
        evaluated += 1;
        let error_ev = exposure_error_ev(e.ours_median_y, e.reference_median_y);
        if error_ev.abs() <= tolerance_ev {
            within += 1;
        } else {
            outliers.push(GateOutlier {
                name: e.name.clone(),
                error_ev,
            });
        }
    }
    outliers.sort_by(|a, b| {
        b.error_ev
            .abs()
            .total_cmp(&a.error_ev.abs())
            .then_with(|| a.name.cmp(&b.name))
    });
    let fraction = if evaluated == 0 {
        0.0
    } else {
        within as f64 / evaluated as f64
    };
    GateReport {
        evaluated,
        within,
        fraction,
        passes: evaluated > 0 && fraction >= required_fraction,
        outliers,
        skipped,
    }
}

/// The exposure settings that must match for frames to share an exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotSettings {
    /// Exposure time in seconds.
    pub shutter_s: f64,
    /// Aperture as an f-number (0 when unknown).
    pub f_number: f64,
    /// ISO sensitivity.
    pub iso: f64,
    /// Exposure compensation in stops.
    pub exposure_compensation_ev: f64,
}

/// Two numbers are the same setting: equal, or within a relative 1e-9 (the same rational read twice).
fn same_number(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

impl ShotSettings {
    /// Whether every setting is identical.
    #[must_use]
    pub fn same_as(&self, other: &ShotSettings) -> bool {
        same_number(self.shutter_s, other.shutter_s)
            && same_number(self.f_number, other.f_number)
            && same_number(self.iso, other.iso)
            && same_number(
                self.exposure_compensation_ev,
                other.exposure_compensation_ev,
            )
    }
}

/// What burst grouping needs to know about a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct BurstFrame {
    /// Identifies the camera (make, model and body serial number, joined by the caller). Frames from
    /// different cameras never group.
    pub camera: String,
    /// Capture time in seconds on any common clock (EXIF date and time plus the sub-second field);
    /// `None` when unknown.
    pub capture_time_s: Option<f64>,
    /// The exposure settings.
    pub settings: ShotSettings,
}

/// Group numbers for frames, in order of first appearance (0, 1, 2, ...): frames of one group have the
/// same camera and identical settings, and each is at most [`BURST_WINDOW_S`] after the previous one.
#[must_use]
pub fn burst_groups(frames: &[BurstFrame]) -> Vec<usize> {
    struct Open {
        camera_index: usize,
        settings: ShotSettings,
        last_time_s: f64,
        group: usize,
    }
    let mut groups = vec![usize::MAX; frames.len()];
    let mut order: Vec<usize> = (0..frames.len())
        .filter(|&i| frames[i].capture_time_s.is_some_and(f64::is_finite))
        .collect();
    order.sort_by(|&a, &b| {
        let (ta, tb) = (
            frames[a].capture_time_s.unwrap_or(0.0),
            frames[b].capture_time_s.unwrap_or(0.0),
        );
        ta.total_cmp(&tb).then(a.cmp(&b))
    });
    let mut cameras: Vec<&str> = Vec::new();
    let mut open: Vec<Open> = Vec::new();
    let mut next_group = 0;
    // Group numbers are assigned in time order first, then renumbered by first appearance in the input.
    for &i in &order {
        let frame = &frames[i];
        let time = frame.capture_time_s.unwrap_or(0.0);
        let camera_index = match cameras.iter().position(|c| *c == frame.camera) {
            Some(k) => k,
            None => {
                cameras.push(&frame.camera);
                cameras.len() - 1
            }
        };
        open.retain(|o| time - o.last_time_s <= BURST_WINDOW_S);
        let found = open
            .iter_mut()
            .find(|o| o.camera_index == camera_index && o.settings.same_as(&frame.settings));
        match found {
            Some(o) => {
                o.last_time_s = time;
                groups[i] = o.group;
            }
            None => {
                open.push(Open {
                    camera_index,
                    settings: frame.settings,
                    last_time_s: time,
                    group: next_group,
                });
                groups[i] = next_group;
                next_group += 1;
            }
        }
    }
    for g in &mut groups {
        if *g == usize::MAX {
            *g = next_group;
            next_group += 1;
        }
    }
    // Renumber by first appearance in the input.
    let mut renumber: Vec<Option<usize>> = vec![None; next_group];
    let mut count = 0;
    for g in &mut groups {
        let slot = &mut renumber[*g];
        *g = *slot.get_or_insert_with(|| {
            count += 1;
            count - 1
        });
    }
    groups
}

/// Step 6: every frame gets the median `EV_auto` of its burst group (itself when alone).
///
/// `frames` and `ev_auto` are parallel slices; the result has the same length. For slices of unequal
/// length the extra entries are ignored and the result has the shorter length.
#[must_use]
pub fn share_burst_ev(frames: &[BurstFrame], ev_auto: &[f64]) -> Vec<f64> {
    let count = frames.len().min(ev_auto.len());
    let groups = burst_groups(&frames[..count]);
    let group_count = groups.iter().copied().max().map_or(0, |g| g + 1);
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); group_count];
    for (g, ev) in groups.iter().zip(ev_auto) {
        members[*g].push(*ev);
    }
    let medians: Vec<f64> = members
        .iter_mut()
        .map(|m| median(m).unwrap_or(0.0))
        .collect();
    groups.iter().map(|g| medians[*g]).collect()
}
