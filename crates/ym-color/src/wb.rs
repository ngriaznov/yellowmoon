//! White balance maths: the scene light, residual gains, the user's Kelvin and tint, the light presets,
//! Pick white and the Auto readouts (stages C1, C3 and C4 of `docs/02-colour-pipeline.md`, sections
//! C1 and C3-C4).
//!
//! Two quantities stay separate on purpose:
//!
//! * the **scene light** ([`SceneLight`]) is what the photo was lit by. It drives the residual gains of
//!   C1, the sensor map of C2 and the ALEXA white `w_A(E_scene)` of C3. It comes from the illuminant
//!   estimate, from a light preset ([`LightSource`]) or from Pick white;
//! * the **user's white** ([`WhiteBalance`]) is the light the ALEXA is told to neutralise. It drives the
//!   gains of C4 and the ARRI matrix of C5. In Auto it is the scene light itself, so C3 and C4 cancel.
//!
//! # Class probabilities
//!
//! The illuminant estimator reports six classes (daylight, shade/overcast, tungsten/halogen,
//! fluorescent, LED, mixed); a camera profile has tables for four (daylight, tungsten, fluorescent,
//! LED). [`EstimatorClasses::to_profile_classes`] folds the six into the four:
//!
//! | Estimator class | Profile class |
//! |---|---|
//! | daylight, shade/overcast | daylight |
//! | tungsten/halogen | tungsten |
//! | fluorescent | fluorescent |
//! | LED | LED |
//! | mixed | the class whose reference lights cover the scene CCT: tungsten below 4000 K, daylight from 4000 K (the `E_scene` rule) |
//!
//! # Tint
//!
//! `t` in -50..=50, positive towards magenta. The green gain of C4 is multiplied by `2^(-t/100)`
//! ([`tint_green_gain`]), so +50 removes half a stop of green. A manual white balance at `(K, t)`
//! neutralises the light whose ALEXA white is `[g w_r, 1, g w_b]` with `[w_r, 1, w_b] = w_A(K)` and
//! `g = 2^(-t/100)` ([`tinted_white`]): a positive tint says "the light was greener than the locus, add
//! magenta".
//!
//! The scene light carries its own tint, so that Auto can show it and Manual can keep it
//! ([`scene_tint`]): the camera white's signed distance from the locus, in log-chroma units, is turned
//! into the green gain that would cancel it (`t = -100 d / (sqrt(2) ln 2)`, `d` positive towards
//! magenta), weighted by the probability of the daylight and tungsten classes (the fluorescent and LED
//! class whites already contain their reference lights' distance from the locus). `w_A(E_scene)` of
//! stage C3 is the class-blended `w_A` of the scene CCT with that tint ([`scene_alexa_white`]). The Auto
//! readout then inverts the user's side exactly ([`auto_readout`]), so `Manual { kelvin, tint }` at the
//! values Auto shows renders the same picture as Auto, up to the rounding of Kelvin to 50 K and an
//! overall brightness factor `2^(-t/100)` (the tint scales the green gain, not the red and blue gains).
//!
//! # Apple Auto's nudge
//!
//! Apple Auto may move the user's white of an Auto white balance along the locus by a few mired and
//! its tint by a few tint units (`docs/19` section 5): [`nudged_alexa_white`] is the scene's white at
//! the nudged CCT ([`nudged_cct`]) with the scene's classes and its tint plus the tint nudge
//! ([`nudged_tint`]), and [`nudged_readout`] what the rows then show.

use serde::{Deserialize, Serialize};

use crate::consts::Vec3;
use crate::spectral::{
    AlexaWhite, ClassProbabilities, ClassProbabilityError, KELVIN_SPLIT_K, LightClass, Locus,
    SpectralError,
};

/// Lowest Kelvin of the White balance control (kelvin).
pub const KELVIN_MIN_K: f64 = 2000.0;
/// Highest Kelvin of the White balance control (kelvin).
pub const KELVIN_MAX_K: f64 = 11000.0;
/// Step of the Kelvin control, and the rounding of the Auto readout (kelvin).
pub const KELVIN_STEP_K: f64 = 50.0;
/// Lowest tint of the White balance control.
pub const TINT_MIN: f64 = -50.0;
/// Highest tint of the White balance control.
pub const TINT_MAX: f64 = 50.0;
/// CIE daylight temperature of the Day preset (kelvin).
pub const DAY_PRESET_K: f64 = 5500.0;
/// CIE daylight temperature of the Shade preset (kelvin).
pub const SHADE_PRESET_K: f64 = 7500.0;
/// Blackbody temperature of the Tungsten preset (kelvin).
pub const TUNGSTEN_PRESET_K: f64 = 3200.0;
/// Pick white refuses a sample with a channel above this fraction of that channel's clip level.
pub const PICK_CLIP_FRACTION: f64 = 0.9;
/// Pick white refuses a sample with a channel below this fraction of that channel's clip level (ten
/// stops below clip): there is no colour to read in the noise floor.
pub const PICK_MIN_FRACTION: f64 = 1.0 / 1024.0;

/// Errors from building a scene light or a white balance.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WbError {
    /// A camera white must have three positive, finite channels.
    #[error("the camera white {0:?} must have three positive finite channels")]
    InvalidWhite(Vec3),
    /// Kelvin outside the control's range or not a number.
    #[error("kelvin {0} must be a number from 2000 to 11000")]
    KelvinOutOfRange(f64),
    /// Tint outside the control's range or not a number.
    #[error("tint {0} must be a number from -50 to 50")]
    TintOutOfRange(f64),
    /// Confidence must be a number from 0 to 1.
    #[error("confidence {0} must be a number from 0 to 1")]
    InvalidConfidence(f64),
    /// A scene CCT must be a positive finite temperature.
    #[error("the scene CCT {0} must be a positive finite temperature")]
    InvalidCct(f64),
    /// The class probabilities are not usable.
    #[error(transparent)]
    Classes(#[from] ClassProbabilityError),
    /// The camera RGB could not be read out as a CCT.
    #[error(transparent)]
    Spectral(#[from] SpectralError),
}

/// Why Pick white refused a sample. The messages are the user-facing text.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PickWhiteError {
    /// A channel is above [`PICK_CLIP_FRACTION`] of its clip level.
    #[error("That spot is too bright to use. Pick a grey or white area that isn't blown out.")]
    TooBright,
    /// A channel is not a number or below [`PICK_MIN_FRACTION`] of its clip level.
    #[error("That spot is too dark to use. Pick a grey or white area that is lit.")]
    TooDark,
    /// The initial white or the resulting light is not usable.
    #[error(transparent)]
    Wb(#[from] WbError),
}

/// The Colour tab's light presets (`light_source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightSource {
    /// The scene light as estimated.
    #[default]
    Auto,
    /// CIE daylight 5500 K, class daylight.
    Day,
    /// CIE daylight 7500 K, class daylight.
    Shade,
    /// Blackbody 3200 K, class tungsten.
    Tungsten,
    /// The estimated chromaticity, class LED.
    Led,
    /// The estimated chromaticity, class fluorescent.
    Fluorescent,
}

/// The `white_balance` setting: Auto, or the user's Kelvin and tint.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WhiteBalance {
    /// The user's white is the scene light: C3 and C4 cancel and the scene white renders neutral.
    #[default]
    Auto,
    /// The ALEXA is balanced for a blackbody (below 4000 K) or CIE daylight (from 4000 K) at `kelvin`,
    /// then the green gain is scaled by `2^(-tint/100)`.
    Manual {
        /// Colour temperature, 2000 to 11000 K.
        kelvin: f64,
        /// Tint, -50 to 50, positive towards magenta.
        tint: f64,
    },
}

impl WhiteBalance {
    /// A manual white balance after checking both ranges.
    ///
    /// # Errors
    ///
    /// [`WbError::KelvinOutOfRange`] or [`WbError::TintOutOfRange`].
    pub fn manual(kelvin: f64, tint: f64) -> Result<Self, WbError> {
        let wb = WhiteBalance::Manual { kelvin, tint };
        wb.validate()?;
        Ok(wb)
    }

    /// Checks the Kelvin and tint ranges of a manual white balance (Auto is always valid).
    ///
    /// # Errors
    ///
    /// [`WbError::KelvinOutOfRange`] or [`WbError::TintOutOfRange`].
    pub fn validate(&self) -> Result<(), WbError> {
        if let WhiteBalance::Manual { kelvin, tint } = *self {
            if !(KELVIN_MIN_K..=KELVIN_MAX_K).contains(&kelvin) {
                return Err(WbError::KelvinOutOfRange(kelvin));
            }
            if !(TINT_MIN..=TINT_MAX).contains(&tint) {
                return Err(WbError::TintOutOfRange(tint));
            }
        }
        Ok(())
    }
}

/// The illuminant estimator's six classes, as probabilities (they need not sum to 1).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EstimatorClasses {
    /// Direct daylight.
    pub daylight: f64,
    /// Shade or overcast sky.
    pub shade: f64,
    /// Tungsten or halogen.
    pub tungsten: f64,
    /// Fluorescent tubes.
    pub fluorescent: f64,
    /// LED.
    pub led: f64,
    /// Mixed lighting.
    pub mixed: f64,
}

impl EstimatorClasses {
    /// Folds the six estimator classes into the profile's four (see the module notes). `cct_k` is the
    /// scene CCT, which decides where the `mixed` mass goes.
    ///
    /// # Errors
    ///
    /// [`ClassProbabilityError`] for a negative or non-finite value or six zeros.
    pub fn to_profile_classes(
        &self,
        cct_k: f64,
    ) -> Result<ClassProbabilities, ClassProbabilityError> {
        let mut daylight = self.daylight + self.shade;
        let mut tungsten = self.tungsten;
        if cct_k < KELVIN_SPLIT_K {
            tungsten += self.mixed;
        } else {
            daylight += self.mixed;
        }
        ClassProbabilities::new(daylight, tungsten, self.fluorescent, self.led)
    }
}

/// The light a photo was taken under: its camera RGB, class probabilities and readout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneLight {
    /// Camera RGB of the light (what a white surface records), green = 1.
    pub w_scene: Vec3,
    /// Probabilities over the profile's light classes.
    pub classes: ClassProbabilities,
    /// How sure the estimate is, 0 to 1 (1 for a preset or a picked white).
    pub confidence: f64,
    /// CCT of the light in kelvin: the camera locus readout of `w_scene`, or the preset's temperature.
    pub cct_k: f64,
    /// Distance from the locus in CIE 1960 uv units, positive towards magenta (camera readout, see
    /// [`Locus::rgb_to_cct_duv`]); 0 for a preset on the locus.
    pub duv: f64,
}

/// A camera white normalised to green = 1.
///
/// # Errors
///
/// [`WbError::InvalidWhite`] unless all three channels are positive and finite.
pub fn normalise_green(w: Vec3) -> Result<Vec3, WbError> {
    if !w.iter().all(|c| c.is_finite() && *c > 0.0) {
        return Err(WbError::InvalidWhite(w));
    }
    Ok([w[0] / w[1], 1.0, w[2] / w[1]])
}

impl SceneLight {
    /// A scene light from a camera white (normalised to green = 1 here), class probabilities and a
    /// confidence; the CCT and Duv are read from the profile's locus.
    ///
    /// # Errors
    ///
    /// [`WbError::InvalidWhite`], [`WbError::InvalidConfidence`] or a readout failure.
    pub fn new(
        w_scene: Vec3,
        classes: ClassProbabilities,
        confidence: f64,
        locus: &Locus,
    ) -> Result<Self, WbError> {
        let w_scene = normalise_green(w_scene)?;
        let readout = locus.rgb_to_cct_duv(w_scene)?;
        Self::from_parts(w_scene, classes, confidence, readout.cct_k, readout.duv)
    }

    /// A scene light from the illuminant estimator's raw output: the six classes are folded into the
    /// profile's four with the CCT read from the locus.
    ///
    /// # Errors
    ///
    /// As [`SceneLight::new`], and for unusable class probabilities.
    pub fn from_estimate(
        w_scene: Vec3,
        classes: &EstimatorClasses,
        confidence: f64,
        locus: &Locus,
    ) -> Result<Self, WbError> {
        let w_scene = normalise_green(w_scene)?;
        let readout = locus.rgb_to_cct_duv(w_scene)?;
        let classes = classes.to_profile_classes(readout.cct_k)?;
        Self::from_parts(w_scene, classes, confidence, readout.cct_k, readout.duv)
    }

    /// A scene light whose CCT and Duv are already known (a preset, a stored estimate, a test).
    ///
    /// # Errors
    ///
    /// [`WbError::InvalidWhite`], [`WbError::InvalidConfidence`] or [`WbError::InvalidCct`].
    pub fn from_parts(
        w_scene: Vec3,
        classes: ClassProbabilities,
        confidence: f64,
        cct_k: f64,
        duv: f64,
    ) -> Result<Self, WbError> {
        let w_scene = normalise_green(w_scene)?;
        if !(0.0..=1.0).contains(&confidence) {
            return Err(WbError::InvalidConfidence(confidence));
        }
        if !(cct_k.is_finite() && cct_k > 0.0) {
            return Err(WbError::InvalidCct(cct_k));
        }
        Ok(Self {
            w_scene,
            classes,
            confidence,
            cct_k,
            duv: if duv.is_finite() { duv } else { 0.0 },
        })
    }

    /// The scene light after Pick white: `w_scene = sample (.) w_scene_initial` normalised to green = 1,
    /// class probabilities unchanged, confidence 1 (the user said what is white). See
    /// [`picked_scene_white`] for the checks.
    ///
    /// # Errors
    ///
    /// [`PickWhiteError`] for a sample that is too bright or too dark, or an unusable initial white.
    pub fn with_picked_white(
        &self,
        sample: Vec3,
        w_scene_initial: Vec3,
        locus: &Locus,
    ) -> Result<Self, PickWhiteError> {
        self.with_picked_white_scaled(sample, w_scene_initial, IDENTITY_WHITE_SCALE, locus)
    }

    /// As [`SceneLight::with_picked_white`] for a body whose raw white is on another scale than its profile's:
    /// the mosaic was balanced with `g0 = white_scale / w_scene_initial` ([`initial_gains_scaled`]), so the sample's
    /// clip levels are `g0` and the picked white comes out on the profile's scale. See [`picked_scene_white_scaled`].
    ///
    /// # Errors
    ///
    /// As [`SceneLight::with_picked_white`]; a `white_scale` that is not positive and finite is
    /// [`WbError::InvalidWhite`].
    pub fn with_picked_white_scaled(
        &self,
        sample: Vec3,
        w_scene_initial: Vec3,
        white_scale: Vec3,
        locus: &Locus,
    ) -> Result<Self, PickWhiteError> {
        let w_scene = picked_scene_white_scaled(sample, w_scene_initial, white_scale)?;
        let readout = locus.rgb_to_cct_duv(w_scene).map_err(WbError::from)?;
        Ok(Self {
            w_scene,
            classes: self.classes,
            confidence: 1.0,
            cct_k: readout.cct_k,
            duv: readout.duv,
        })
    }
}

impl LightSource {
    /// The scene light for the sensor map under this preset (`docs/02` section C1, preset table).
    ///
    /// | Preset | Scene light |
    /// |---|---|
    /// | Auto | `estimate` |
    /// | Day, Shade | the camera white of CIE daylight 5500 K / 7500 K on the profile locus, class daylight |
    /// | Tungsten | the camera white of a 3200 K blackbody, class tungsten |
    /// | LED, Fluorescent | the estimated chromaticity and CCT, all probability on class LED / fluorescent |
    #[must_use]
    pub fn scene_light(self, estimate: &SceneLight, locus: &Locus) -> SceneLight {
        let model = |cct_k: f64, class: LightClass| SceneLight {
            w_scene: locus.rgb_at(cct_k),
            classes: ClassProbabilities::only(class),
            confidence: 1.0,
            cct_k,
            duv: 0.0,
        };
        let class_only = |class: LightClass| SceneLight {
            classes: ClassProbabilities::only(class),
            confidence: 1.0,
            ..*estimate
        };
        match self {
            LightSource::Auto => *estimate,
            LightSource::Day => model(DAY_PRESET_K, LightClass::Daylight),
            LightSource::Shade => model(SHADE_PRESET_K, LightClass::Daylight),
            LightSource::Tungsten => model(TUNGSTEN_PRESET_K, LightClass::Tungsten),
            LightSource::Led => class_only(LightClass::Led),
            LightSource::Fluorescent => class_only(LightClass::Fluorescent),
        }
    }
}

/// Per-channel median of a patch of camera RGB samples (Pick white takes the 5x5 median around the
/// click). `None` for an empty patch or one with a non-finite sample.
#[must_use]
pub fn median_rgb(samples: &[Vec3]) -> Option<Vec3> {
    if samples.is_empty() || !samples.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    let median = |channel: usize| {
        let mut values: Vec<f64> = samples.iter().map(|s| s[channel]).collect();
        values.sort_unstable_by(f64::total_cmp);
        let n = values.len();
        if n % 2 == 1 {
            values[n / 2]
        } else {
            0.5 * (values[n / 2 - 1] + values[n / 2])
        }
    };
    Some([median(0), median(1), median(2)])
}

/// Pick white: the scene white from a sample of demosaiced camera RGB that is still balanced with the
/// initial gains `g0` (`docs/02` section C1).
///
/// The raw files clip at 1.0 per channel before the gains, so after `g0 = 1 / w_scene_initial` channel
/// `c` clips at `1 / w_scene_initial[c]`. The sample is refused when a channel is above
/// [`PICK_CLIP_FRACTION`] of that level ([`PickWhiteError::TooBright`]; a sample taken where a
/// vignetting gain was applied should be divided by that gain first). Otherwise the initial gains are
/// undone, `w_scene = sample (.) w_scene_initial`, and the result is normalised to green = 1.
///
/// # Errors
///
/// [`PickWhiteError::TooBright`] first, then [`PickWhiteError::TooDark`] for a channel that is not a
/// number or is below [`PICK_MIN_FRACTION`] of its clip level; an unusable initial white is
/// [`WbError::InvalidWhite`].
pub fn picked_scene_white(sample: Vec3, w_scene_initial: Vec3) -> Result<Vec3, PickWhiteError> {
    picked_scene_white_scaled(sample, w_scene_initial, IDENTITY_WHITE_SCALE)
}

/// Pick white for a body with a white scale (`docs/03`, White scale of stand-in bodies).
///
/// `white_scale` takes the body's raw white to the white of its profile's curves, `w_profile = w_body (.)
/// white_scale`, and `w_scene_initial` is the white **on the profile's scale** (the estimator's and the locus'
/// scale). The mosaic was balanced with `g0 = white_scale / w_scene_initial`, so channel `c` clips at `g0[c]`
/// (the raw clips at 1.0 before the gains) and the sample is refused above [`PICK_CLIP_FRACTION`] of that level;
/// the sample times `w_scene_initial` is then already the picked white on the profile's scale, whatever the
/// scale is. With `white_scale = [1, 1, 1]` this is [`picked_scene_white`] to the last bit.
///
/// # Errors
///
/// As [`picked_scene_white`]; a `white_scale` that is not positive and finite, or whose green is not 1, is
/// [`WbError::InvalidWhite`].
pub fn picked_scene_white_scaled(
    sample: Vec3,
    w_scene_initial: Vec3,
    white_scale: Vec3,
) -> Result<Vec3, PickWhiteError> {
    let w_init = normalise_green(w_scene_initial)?;
    let clip = initial_gains_scaled(w_scene_initial, white_scale)?;
    if sample
        .iter()
        .zip(&clip)
        .any(|(s, c)| *s > PICK_CLIP_FRACTION * c)
    {
        return Err(PickWhiteError::TooBright);
    }
    if sample
        .iter()
        .zip(&clip)
        .any(|(s, c)| !s.is_finite() || *s < PICK_MIN_FRACTION * c)
    {
        return Err(PickWhiteError::TooDark);
    }
    let w = [
        sample[0] * w_init[0],
        sample[1] * w_init[1],
        sample[2] * w_init[2],
    ];
    Ok(normalise_green(w)?)
}

/// The initial gains of the raw stages, `g0 = 1 / w_scene_initial` (green = 1).
///
/// # Errors
///
/// [`WbError::InvalidWhite`] for an unusable white.
pub fn initial_gains(w_scene_initial: Vec3) -> Result<Vec3, WbError> {
    initial_gains_scaled(w_scene_initial, IDENTITY_WHITE_SCALE)
}

/// The identity white scale: a body whose profile was measured on it (or on its sensor).
pub const IDENTITY_WHITE_SCALE: Vec3 = [1.0, 1.0, 1.0];

/// Checks a white scale: finite and positive in every channel, green exactly 1.
fn check_white_scale(white_scale: Vec3) -> Result<Vec3, WbError> {
    if white_scale.iter().all(|c| c.is_finite() && *c > 0.0) && white_scale[1] == 1.0 {
        Ok(white_scale)
    } else {
        Err(WbError::InvalidWhite(white_scale))
    }
}

/// A body's raw white on its profile's scale: `white (.) white_scale` with green = 1 (`docs/03`, White scale of
/// stand-in bodies). Used for the camera's as-shot white; the illuminant estimator's input is scaled the same way
/// channel by channel.
///
/// # Errors
///
/// [`WbError::InvalidWhite`] for an unusable white or white scale.
pub fn scale_white(white: Vec3, white_scale: Vec3) -> Result<Vec3, WbError> {
    let s = check_white_scale(white_scale)?;
    normalise_green([white[0] * s[0], white[1] * s[1], white[2] * s[2]])
}

/// The initial gains of the raw stages for a body with a white scale, `g0 = white_scale / w_scene_initial`
/// (green = 1). `w_scene_initial` is on the profile's scale, so the mosaic (the body's own raw scale) balanced
/// with `g0` is what the profile's curves would have recorded, balanced for that white. With the identity scale
/// this is [`initial_gains`] to the last bit (`1.0 / c`).
///
/// # Errors
///
/// [`WbError::InvalidWhite`] for an unusable white or white scale.
pub fn initial_gains_scaled(w_scene_initial: Vec3, white_scale: Vec3) -> Result<Vec3, WbError> {
    let s = check_white_scale(white_scale)?;
    let w = normalise_green(w_scene_initial)?;
    Ok([s[0] / w[0], s[1] / w[1], s[2] / w[2]])
}

/// The residual gains of stage C1, `g_res = w_scene_initial / w_scene` element-wise with both
/// normalised to green = 1. They are exactly `[1, 1, 1]` when the two whites are equal.
///
/// # Errors
///
/// [`WbError::InvalidWhite`] for an unusable white.
pub fn residual_gains(w_scene_initial: Vec3, w_scene: Vec3) -> Result<Vec3, WbError> {
    let init = normalise_green(w_scene_initial)?;
    let scene = normalise_green(w_scene)?;
    Ok([init[0] / scene[0], 1.0, init[2] / scene[2]])
}

/// The factor tint `t` puts on the green gain of stage C4, `2^(-t/100)`: below 1 for a positive
/// (magenta) tint.
#[must_use]
pub fn tint_green_gain(tint: f64) -> f64 {
    (-tint / 100.0).exp2()
}

/// The gains of stage C4 for a manual white balance: `1 / w_A(E_user)` with the green gain scaled by
/// [`tint_green_gain`]. `E_user` is a blackbody below 4000 K and CIE daylight from there, at `kelvin`
/// clamped to the control's range.
#[must_use]
pub fn manual_user_gains(white: &AlexaWhite, kelvin: f64, tint: f64) -> Vec3 {
    let k = kelvin.clamp(KELVIN_MIN_K, KELVIN_MAX_K);
    let t = tint.clamp(TINT_MIN, TINT_MAX);
    let w = white.locus_white(k);
    [1.0 / w[0], tint_green_gain(t), 1.0 / w[2]]
}

/// Tint units per unit of log-chroma distance: `100 / (sqrt(2) ln 2)`, about 102. A green gain `g` moves
/// `(ln R/G, ln B/G)` by `ln(1/g)` in both coordinates, a distance of `sqrt(2) ln(1/g)`, and a tint `t`
/// is `g = 2^(-t/100)`.
pub const TINT_PER_LOG_CHROMA: f64 = 100.0 / (std::f64::consts::SQRT_2 * std::f64::consts::LN_2);

/// The ALEXA white that a tint at a given `w_A` neutralises: `[g w_r, 1, g w_b]` with `g = 2^(-t/100)`,
/// green = 1. The gains of `Manual { kelvin, tint }` are `1 / [w_r, 1/g, w_b]`, which map exactly this
/// white to neutral (up to an overall factor `g`).
#[must_use]
pub fn tinted_white(w: Vec3, tint: f64) -> Vec3 {
    let g = tint_green_gain(tint);
    [g * w[0], 1.0, g * w[2]]
}

/// The tint of a scene light: the green gain, as a tint value, that cancels the camera white's distance
/// from the locus. `t = -TINT_PER_LOG_CHROMA * d` with `d = duv / duv_scale(cct)` the signed log-chroma
/// distance (positive towards magenta), weighted by the probability of the daylight and tungsten classes
/// (the fluorescent and LED class whites already include their reference lights' distance from the
/// locus) and held in -50 to 50. A greener light gives a positive tint. Zero for a preset on the locus.
///
/// The distance is the camera's, read through the profile's locus, so it is as good as the Duv readout
/// (`docs/03-camera-profiles.md`: not exact for spiky spectra, and not trustworthy beyond 0.002 Duv
/// between about 3800 K and 4300 K).
#[must_use]
pub fn scene_tint(locus: &Locus, scene: &SceneLight) -> f64 {
    let model = scene.classes.get(LightClass::Daylight) + scene.classes.get(LightClass::Tungsten);
    let scale = locus.duv_scale_at(scene.cct_k);
    if !(scale.is_finite() && scale > 0.0 && scene.duv.is_finite()) {
        return 0.0;
    }
    // `+ 0.0` turns a negative zero into 0.
    (-TINT_PER_LOG_CHROMA * model * scene.duv / scale).clamp(TINT_MIN, TINT_MAX) + 0.0
}

/// `w_A(E_scene)` for stage C3: the class-blended `w_A` of the scene CCT ([`AlexaWhite::scene_white`])
/// with the scene's tint ([`scene_tint`], [`tinted_white`]), green = 1.
#[must_use]
pub fn scene_alexa_white(white: &AlexaWhite, locus: &Locus, scene: &SceneLight) -> Vec3 {
    tinted_white(
        white.scene_white(scene.cct_k, &scene.classes),
        scene_tint(locus, scene),
    )
}

/// The lowest CCT a nudged white can reach (kelvin): the warm end of the ALEXA white tables.
pub const NUDGE_MIN_K: f64 = 1500.0;
/// The highest CCT a nudged white can reach (kelvin): the cool end of the ALEXA white tables.
pub const NUDGE_MAX_K: f64 = 25000.0;

/// The CCT of the user's white after Apple Auto's nudge (`docs/19` section 5): the scene CCT moved by
/// `shift_mired` on the mired scale, `1e6 / (1e6 / cct_scene_k + shift_mired)`, held in
/// [`NUDGE_MIN_K`] to [`NUDGE_MAX_K`] (the range of the white tables; a nudge of at most 20 mired never
/// reaches either end from a scene the estimator reports). A negative shift is a higher Kelvin: the
/// camera is balanced for a bluer light and the picture comes out warmer.
#[must_use]
pub fn nudged_cct(cct_scene_k: f64, shift_mired: f64) -> f64 {
    let mired = 1e6 / cct_scene_k + shift_mired;
    if !(mired.is_finite() && mired > 0.0) {
        return if mired > 0.0 {
            NUDGE_MIN_K
        } else {
            NUDGE_MAX_K
        };
    }
    (1e6 / mired).clamp(NUDGE_MIN_K, NUDGE_MAX_K)
}

/// The tint nudge that applies to a scene (`docs/19` section 5): `tint_shift` added to the scene's
/// own tint ([`scene_tint`]) and the sum held in [`TINT_MIN`] to [`TINT_MAX`], returned as the change
/// of the tint. A positive shift is more magenta (it removes a green cast), as a manual tint is.
#[must_use]
pub fn nudged_tint(locus: &Locus, scene: &SceneLight, tint_shift: f64) -> f64 {
    if tint_shift == 0.0 || !tint_shift.is_finite() {
        return 0.0;
    }
    let tint = scene_tint(locus, scene);
    // `+ 0.0` turns a negative zero into 0.
    (tint + tint_shift).clamp(TINT_MIN, TINT_MAX) - tint + 0.0
}

/// The user's white of a nudged Auto white balance (`docs/19` section 5): the scene's class-blended,
/// tinted ALEXA white `w_A(E_scene)` ([`scene_alexa_white`]) moved along the locus to the nudged CCT
/// ([`nudged_cct`]), channel by channel `w_A(E_scene) * w_locus(CCT_user) / w_locus(CCT_scene)`
/// ([`AlexaWhite::locus_white`]), so the scene's class mix and tint are kept, then given the tint
/// nudge ([`nudged_tint`]) as a tint is given to a white ([`tinted_white`]: red and blue times
/// `2^(-dt/100)`). Returns the nudged CCT and `w_A(E_user)`, green = 1. A shift of 0 mired keeps the
/// scene's CCT exactly and a tint shift of 0 leaves the white as the locus move made it.
///
/// For daylight and tungsten light, whose class whites are the locus model, the move along the locus
/// is the class-blended white evaluated at the nudged CCT. The fluorescent and LED class whites are
/// lists of real lights (not ordered in colour, and constant beyond their ends), so evaluating them
/// at another CCT can move the white the wrong way or not at all; moving the scene's own white as the
/// locus moves gives every light the same, monotonic nudge (`docs/02` C3–C4, Apple Auto's nudge).
#[must_use]
pub fn nudged_alexa_white(
    white: &AlexaWhite,
    locus: &Locus,
    scene: &SceneLight,
    shift_mired: f64,
    tint_shift: f64,
) -> (f64, Vec3) {
    let w_scene = scene_alexa_white(white, locus, scene);
    let (cct_k, mut w) = if shift_mired == 0.0 {
        (scene.cct_k, w_scene)
    } else {
        let cct_k = nudged_cct(scene.cct_k, shift_mired);
        let from = white.locus_white(scene.cct_k);
        let to = white.locus_white(cct_k);
        (
            cct_k,
            [
                w_scene[0] * (to[0] / from[0]),
                1.0,
                w_scene[2] * (to[2] / from[2]),
            ],
        )
    };
    let dt = nudged_tint(locus, scene, tint_shift);
    if dt != 0.0 {
        w = tinted_white(w, dt);
    }
    (cct_k, w)
}

/// The values the White balance rows show: Kelvin and tint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WbReadout {
    /// Colour temperature in kelvin, a multiple of 50 within 2000 to 11000 (for Manual, the user's
    /// own value).
    pub kelvin_k: f64,
    /// Tint, -50 to 50, positive towards magenta.
    pub tint: f64,
}

/// The Auto readout for a scene light.
///
/// * Kelvin is the scene CCT rounded to 50 K and held in 2000 to 11000 K.
/// * Tint is the value that reproduces the scene's distance from the locus: with `w_s = w_A(E_scene)`
///   ([`scene_alexa_white`]) and `w_u = w_A(blackbody or daylight at that Kelvin)`, the gains of
///   `Manual { kelvin, tint }` act on the scene white as `[w_s_r / w_u_r, 2^(-t/100), w_s_b / w_u_b]`;
///   they are equal, so the white is neutral, only when the three agree. The tint that comes closest
///   on the green axis is the geometric mean of the red and blue ratios,
///   `t = -50 log2((w_s_r / w_u_r) (w_s_b / w_u_b))`, held in -50 to 50. With the scene white built as
///   above this is the scene's own tint (plus the class whites' offset from the locus for LED and
///   fluorescent light), and 0 for a preset on the locus. A greener light than the locus reads a
///   positive tint, a magenta one a negative tint.
///
/// The tint is computed for the rounded Kelvin that is shown, so the pair is self consistent.
#[must_use]
pub fn auto_readout(white: &AlexaWhite, locus: &Locus, scene: &SceneLight) -> WbReadout {
    readout_of_white(white, scene.cct_k, scene_alexa_white(white, locus, scene))
}

/// The readout of an Auto white balance whose user white Apple Auto nudged (`docs/19` section 5):
/// [`auto_readout`] of the nudged white `w_A(E_user)` ([`nudged_alexa_white`]) at its CCT. The Kelvin
/// is the nudged CCT rounded to 50 K and held in 2000 to 11000 K, and the tint is the one that makes
/// `Manual { kelvin, tint }` reproduce the nudged white (the scene's own tint plus the tint nudge, up
/// to the rounding of the Kelvin, for daylight and tungsten light), so the Automatic card and the
/// Kelvin and Tint sliders' detents show the white in effect.
#[must_use]
pub fn nudged_readout(
    white: &AlexaWhite,
    locus: &Locus,
    scene: &SceneLight,
    shift_mired: f64,
    tint_shift: f64,
) -> WbReadout {
    let (cct_k, w) = nudged_alexa_white(white, locus, scene, shift_mired, tint_shift);
    readout_of_white(white, cct_k, w)
}

/// The Kelvin (`cct_k` rounded to 50 K, held in 2000 to 11000 K) and the tint that come closest to
/// neutralising the ALEXA white `w_s` with `Manual { kelvin, tint }` (see [`auto_readout`]).
fn readout_of_white(white: &AlexaWhite, cct_k: f64, w_s: Vec3) -> WbReadout {
    let kelvin_k =
        ((cct_k / KELVIN_STEP_K).round() * KELVIN_STEP_K).clamp(KELVIN_MIN_K, KELVIN_MAX_K);
    let w_u = white.locus_white(kelvin_k);
    let product = (w_s[0] / w_u[0]) * (w_s[2] / w_u[2]);
    // `+ 0.0` turns a negative zero into 0.
    let tint = (-50.0 * product.log2()).clamp(TINT_MIN, TINT_MAX) + 0.0;
    WbReadout { kelvin_k, tint }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_classes_fold_into_the_profile_classes() {
        let c = EstimatorClasses {
            daylight: 0.2,
            shade: 0.1,
            tungsten: 0.1,
            fluorescent: 0.2,
            led: 0.2,
            mixed: 0.2,
        };
        // Mixed light at 3000 K goes to tungsten, at 5000 K to daylight.
        let low = c.to_profile_classes(3000.0).unwrap().to_array();
        let high = c.to_profile_classes(5000.0).unwrap().to_array();
        let near = |a: [f64; 4], b: [f64; 4]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-15);
        assert!(near(low, [0.3, 0.3, 0.2, 0.2]));
        assert!(near(high, [0.5, 0.1, 0.2, 0.2]));
        assert!(near(
            c.to_profile_classes(KELVIN_SPLIT_K).unwrap().to_array(),
            high
        ));
        assert!(
            EstimatorClasses::default()
                .to_profile_classes(5000.0)
                .is_err()
        );
        let all_mixed = EstimatorClasses {
            mixed: 1.0,
            ..Default::default()
        };
        assert_eq!(
            all_mixed.to_profile_classes(2800.0).unwrap(),
            ClassProbabilities::only(LightClass::Tungsten)
        );
        assert_eq!(
            all_mixed.to_profile_classes(6500.0).unwrap(),
            ClassProbabilities::only(LightClass::Daylight)
        );
    }

    #[test]
    fn manual_white_balance_checks_its_ranges() {
        assert!(WhiteBalance::manual(2000.0, -50.0).is_ok());
        assert!(WhiteBalance::manual(11000.0, 50.0).is_ok());
        assert_eq!(
            WhiteBalance::manual(1999.0, 0.0),
            Err(WbError::KelvinOutOfRange(1999.0))
        );
        assert_eq!(
            WhiteBalance::manual(5000.0, 50.5),
            Err(WbError::TintOutOfRange(50.5))
        );
        assert!(WhiteBalance::manual(f64::NAN, 0.0).is_err());
        assert!(WhiteBalance::manual(5000.0, f64::NAN).is_err());
        assert!(WhiteBalance::Auto.validate().is_ok());
    }

    #[test]
    fn white_balance_serialises_with_a_kind_tag() {
        let json = serde_json::to_string(&WhiteBalance::Auto).unwrap();
        assert_eq!(json, r#"{"kind":"auto"}"#);
        let wb: WhiteBalance =
            serde_json::from_str(r#"{"kind":"manual","kelvin":3200.0,"tint":-5.0}"#).unwrap();
        assert_eq!(
            wb,
            WhiteBalance::Manual {
                kelvin: 3200.0,
                tint: -5.0
            }
        );
        let light: LightSource = serde_json::from_str(r#""fluorescent""#).unwrap();
        assert_eq!(light, LightSource::Fluorescent);
    }

    #[test]
    fn tint_scales_the_green_gain_by_two_to_the_minus_t_over_100() {
        assert_eq!(tint_green_gain(0.0), 1.0);
        assert!((tint_green_gain(50.0) - 0.5_f64.sqrt()).abs() < 1e-15);
        assert!((tint_green_gain(-50.0) - 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((tint_green_gain(100.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn median_takes_each_channel_separately() {
        let patch = [
            [0.1, 0.9, 0.5],
            [0.2, 0.1, 0.5],
            [0.3, 0.5, 0.1],
            [0.9, 0.2, 0.9],
            [0.0, 0.3, 0.2],
        ];
        assert_eq!(median_rgb(&patch), Some([0.2, 0.3, 0.5]));
        assert_eq!(median_rgb(&patch[..4]), Some([0.25, 0.35, 0.5]));
        assert_eq!(median_rgb(&[]), None);
        assert_eq!(median_rgb(&[[0.1, f64::NAN, 0.2]]), None);
    }

    #[test]
    fn residual_gains_are_identity_for_equal_whites_and_normalised() {
        let w = [0.55, 1.0, 0.71];
        assert_eq!(residual_gains(w, w).unwrap(), [1.0, 1.0, 1.0]);
        // Scale in the input does not matter: green is normalised first.
        assert_eq!(
            residual_gains(w, w.map(|c| c * 7.0)).unwrap(),
            [1.0, 1.0, 1.0]
        );
        let g = residual_gains([0.5, 1.0, 0.7], [0.4, 1.0, 0.8]).unwrap();
        assert_eq!(g, [0.5 / 0.4, 1.0, 0.7 / 0.8]);
        assert!(residual_gains([0.5, 0.0, 0.7], w).is_err());
        assert!(residual_gains(w, [f64::NAN, 1.0, 1.0]).is_err());
    }

    #[test]
    fn initial_gains_invert_the_white() {
        let g = initial_gains([0.5, 2.0, 0.8]).unwrap();
        assert_eq!(g, [1.0 / 0.25, 1.0, 1.0 / 0.4]);
    }

    #[test]
    fn picked_white_undoes_the_initial_gains() {
        let w_init = [0.5, 1.0, 0.7];
        // A neutral object under a light of camera white `w`: the g0-balanced sample is w / w_init.
        let w = [0.62, 1.0, 0.55];
        let sample = [0.3 * w[0] / w_init[0], 0.3, 0.3 * w[2] / w_init[2]];
        let got = picked_scene_white(sample, w_init).unwrap();
        for k in 0..3 {
            assert!((got[k] - w[k]).abs() < 1e-15, "{got:?} vs {w:?}");
        }
        assert_eq!(got[1], 1.0);
    }

    #[test]
    fn pick_white_refuses_blown_out_spots_with_the_exact_message() {
        let w_init = [0.5, 1.0, 0.7];
        // Clip levels after g0: [2, 1, 1/0.7].
        assert!(picked_scene_white([1.79, 0.89, 1.27], w_init).is_ok());
        for sample in [[1.81, 0.5, 0.5], [0.5, 0.91, 0.5], [0.5, 0.5, 1.29]] {
            let err = picked_scene_white(sample, w_init).unwrap_err();
            assert_eq!(err, PickWhiteError::TooBright);
            assert_eq!(
                err.to_string(),
                "That spot is too bright to use. Pick a grey or white area that isn't blown out."
            );
        }
        // Too dark, not a number, and bright wins over dark.
        assert_eq!(
            picked_scene_white([0.0, 0.5, 0.5], w_init),
            Err(PickWhiteError::TooDark)
        );
        assert_eq!(
            picked_scene_white([f64::NAN, 0.5, 0.5], w_init),
            Err(PickWhiteError::TooDark)
        );
        assert_eq!(
            picked_scene_white([0.0, 0.95, 0.5], w_init),
            Err(PickWhiteError::TooBright)
        );
        assert!(matches!(
            picked_scene_white([0.5, 0.5, 0.5], [0.5, -1.0, 0.7]),
            Err(PickWhiteError::Wb(WbError::InvalidWhite(_)))
        ));
    }

    #[test]
    fn scene_light_rejects_bad_parts() {
        let ok = ClassProbabilities::only(LightClass::Daylight);
        assert!(SceneLight::from_parts([0.5, 1.0, 0.7], ok, 0.5, 5000.0, 0.0).is_ok());
        assert!(matches!(
            SceneLight::from_parts([0.5, 1.0, 0.7], ok, 1.5, 5000.0, 0.0),
            Err(WbError::InvalidConfidence(_))
        ));
        assert!(matches!(
            SceneLight::from_parts([0.5, 1.0, 0.7], ok, 0.5, 0.0, 0.0),
            Err(WbError::InvalidCct(_))
        ));
        assert!(matches!(
            SceneLight::from_parts([0.5, 1.0, -0.7], ok, 0.5, 5000.0, 0.0),
            Err(WbError::InvalidWhite(_))
        ));
        // Green is normalised.
        let light = SceneLight::from_parts([1.0, 2.0, 1.4], ok, 0.5, 5000.0, 0.0).unwrap();
        assert_eq!(light.w_scene, [0.5, 1.0, 0.7]);
    }
}
