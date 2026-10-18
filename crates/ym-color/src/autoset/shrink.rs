//! How a prediction becomes the photo's settings (`docs/18-auto-settings.md` §4.4).
//!
//! A head with mean `mu` and variance `var` gets the weight `w = sigma0² / (sigma0² + var)`, in (0, 1]. The exposure
//! is the one the model asks for, in full: full rescue (owner's decision of 2026-10-18), so it never falls back
//! towards the camera's exposure; its weight is only reported, as the exposure's confidence. The camera floor
//! (`ym_core::autoset::camera::camera_floor`, `docs/18` section 6.1) then raises it to the camera's own exposure
//! label where that is higher, which needs the photo's embedded JPEG and so is decided in `ym-core`. The contrast is
//! `w` times the predicted contrast: a doubtful contrast fades to 0.

use crate::exposure::{CAMERA_GAIN_EV, EV_LIMIT};

use super::{AutosetModel, Prediction};

/// What `autoset` decides for one photo: the exposure before the highlight guard, the contrast, and the weight
/// behind each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Applied {
    /// `EV_auto` before the highlight guard (`docs/02`, step 4): the exposure the model asks for
    /// (`key_target - key`), clamped to plus or minus [`EV_LIMIT`]; the camera gain only when the prediction is
    /// not usable. `ym-core` raises it to the camera's exposure label when that is higher (the camera floor).
    pub ev_before_guard: f64,
    /// The exposure's confidence `sigma0² / (sigma0² + var)`, in (0, 1]; 0 for the rule
    /// (`AutoSummary::ev_confidence`). Reported only: it does not move the exposure.
    pub ev_weight: f64,
    /// The Auto contrast: the contrast weight times the predicted contrast, bounded to the model's contrast limit and
    /// rounded to 0.1.
    pub contrast: f64,
    /// The contrast's weight, in (0, 1]; 0 for the rule. It is the confidence of the contrast
    /// (`AutoSummary::contrast_confidence`).
    pub contrast_weight: f64,
}

/// The shrinkage weight `sigma0² / (sigma0² + variance)` of one head. It is 0 (the rule) when the prediction is not
/// finite, so a broken prediction can never move the photo.
fn weight(prior_sigma: f64, mean: f64, variance: f64) -> f64 {
    if !(mean.is_finite() && variance.is_finite() && variance >= 0.0) {
        return 0.0;
    }
    let prior = prior_sigma * prior_sigma;
    prior / (prior + variance)
}

/// Rounds to a tenth, half away from zero. A negative zero becomes zero, so the value is written the same way
/// everywhere.
fn round_tenth(x: f64) -> f64 {
    let rounded = (x * 10.0).round() / 10.0;
    if rounded == 0.0 { 0.0 } else { rounded }
}

/// The Auto contrast rounded to a tenth, as [`apply`] rounds a photo's own (`docs/18` §4.4). The burst median of the
/// contrasts is rounded the same way, so every value a photo serves is a multiple of 0.1.
#[must_use]
pub fn round_contrast(x: f64) -> f64 {
    round_tenth(x)
}

/// The settings of a prediction (`docs/18` §4.4). `key` is the key `m` the photo was metered at (`Features::key`);
/// the exposure is `key_target - key`, the contrast is shrunk towards 0 by its weight.
#[must_use]
pub fn apply(prediction: &Prediction, key: f64, model: &AutosetModel) -> Applied {
    let sigma = model.prior_sigma();
    let limit = model.contrast_limit();
    let ev_weight = weight(sigma.key_target, prediction.key_target, prediction.key_var);
    let contrast_weight = weight(sigma.contrast, prediction.contrast, prediction.contrast_var);
    let ev = if ev_weight > 0.0 && key.is_finite() {
        prediction.key_target - key
    } else {
        CAMERA_GAIN_EV
    };
    let contrast = if contrast_weight > 0.0 {
        (contrast_weight * prediction.contrast).clamp(-limit, limit)
    } else {
        0.0
    };
    Applied {
        ev_before_guard: ev.clamp(-EV_LIMIT, EV_LIMIT),
        ev_weight,
        contrast: round_tenth(contrast),
        contrast_weight,
    }
}

/// The rule alone, the fallback when there is no usable model or preview (`docs/18` §4.4): the camera gain as the
/// exposure, clamped to plus or minus [`EV_LIMIT`], no contrast, and both weights 0.
#[must_use]
pub fn rule_only() -> Applied {
    Applied {
        ev_before_guard: CAMERA_GAIN_EV.clamp(-EV_LIMIT, EV_LIMIT),
        ev_weight: 0.0,
        contrast: 0.0,
        contrast_weight: 0.0,
    }
}
