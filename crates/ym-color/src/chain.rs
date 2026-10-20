//! The fused colour chain C1 to C6 as one 3x3 matrix per photo (`docs/02-colour-pipeline.md`, section
//! "Stage order").
//!
//! ```text
//! M = 2^EV * M_awg3->awg4 * M_arri(CCT_user) * diag(g_user) * diag(w_A(E_scene)) * M_map(scene) * diag(g_res)
//! ```
//!
//! | Factor | Stage | Where it comes from |
//! |---|---|---|
//! | `diag(g_res)` | C1 | [`wb::residual_gains`]: scene light against the light the mosaic was balanced for |
//! | `M_map(scene)` | C2 | [`SensorMapTables::blend`] at the scene light's class probabilities and mired |
//! | `diag(w_A(E_scene))` | C3 | [`wb::scene_alexa_white`]: [`AlexaWhite::scene_white`] (blackbody below 4000 K, CIE daylight from there, the fluorescent and LED lists for those classes, blended by probability) with the scene's own tint |
//! | `diag(g_user)` | C4 | Auto: `1 / w_A(E_scene)` (with Apple Auto's nudge, `1 / w_A(E_user)` at the nudged CCT); Manual: `1 / w_A(E_user)` with the tint on the green gain |
//! | `M_arri(CCT_user)` | C5 | [`ArriTable::native_to_awg3`]: 17 CCTs, linear in mired, clamped to 2000 to 11000 K |
//! | `M_awg3->awg4` | C6 | [`gamut::matrices`] |
//! | `2^EV` | C6 | the resolved exposure |
//!
//! Every `w` and `g` is normalised to green = 1 (the tint then scales the green gain). Input is
//! scene-linear camera RGB balanced with the initial gains `g0` of the raw stages; output is linear
//! AWG4.
//!
//! **Auto white balance.** The user's white is the scene light, so C3 and C4 cancel: the product of the
//! two diagonals is set to exactly `[1, 1, 1]` and the scene white (its camera RGB, balanced with `g0`)
//! lands on a neutral AWG4 triple `2^EV * [1, 1, 1]`, because the sensor map preserves white, the ARRI
//! matrices are balanced (rows sum to 1) and AWG3 to AWG4 keeps the white. `g_user` is `1 / w_A(E_scene)`
//! with the class-blended `w_A`, not the blackbody/daylight model at the CCT, so an LED or fluorescent
//! scene is neutralised too (`docs/03-camera-profiles.md`).
//!
//! **Apple Auto's nudge** (`docs/19` section 5) moves the user's white of an Auto white balance (light
//! source Auto, no picked white) by [`ChainSettings::white_shift_mired`] along the locus and its tint
//! by [`ChainSettings::tint_shift`]: `g_user` is `1 / w_A(E_user)`, the scene's tinted, class-blended
//! white moved as the locus moves from `CCT_scene` to `CCT_user = 1e6 / (1e6 / CCT_scene + m)` and
//! given the scene's tint plus the tint nudge ([`wb::nudged_alexa_white`]), the ARRI matrix takes
//! `CCT_user` and the readout is [`wb::nudged_readout`]. With `m = 0` and a tint nudge of 0 none of
//! this runs and the chain is the Auto chain bit for bit.
//!
//! **Manual white balance** leaves the scene light alone and puts the user's `E_user` (blackbody below
//! 4000 K, CIE daylight from there, at the user's Kelvin, then the tint) in C4 and the user's Kelvin in
//! C5. A daylight scene set to 3200 K therefore renders the way an ALEXA balanced for 3200 K tungsten
//! would render it: blue (`tests/chain_spectral.rs` checks the amount against a spectral simulation).
//! Because `w_A(E_scene)` carries the scene's own tint ([`wb::scene_tint`]), `Manual` at the Kelvin and
//! tint that Auto shows renders the Auto picture times the green gain `2^(-t/100)`.

use crate::arri::ArriTable;
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::exposure::{ExposureError, ExposureSetting};
use crate::gamut::{self, LinearAwg4};
use crate::sensor_map::SensorMapTables;
use crate::spectral::{AlexaWhite, Locus};
use crate::tone;
use crate::wb::{
    self, KELVIN_MAX_K, KELVIN_MIN_K, LightSource, PickWhiteError, SceneLight, WbError, WbReadout,
    WhiteBalance,
};

/// Errors from building a chain.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ChainError {
    /// The exposure is not a finite number of stops, or `2^EV` is not a finite positive gain.
    #[error(
        "the exposure {0} EV must be a finite number of stops that gives a finite, non-zero gain"
    )]
    InvalidExposure(f64),
    /// A white or a manual white balance is unusable.
    #[error(transparent)]
    Wb(#[from] WbError),
    /// Pick white refused the sample.
    #[error(transparent)]
    Pick(#[from] PickWhiteError),
    /// A manual exposure outside the control's range.
    #[error(transparent)]
    Exposure(#[from] ExposureError),
}

/// The three tables of a camera profile that the chain reads. `ym-profiles` owns the profile type;
/// build this from its public fields (`profile.sensor_map.classes`, `profile.alexa_white`,
/// `profile.locus`).
#[derive(Debug, Clone, Copy)]
pub struct ProfileTables<'a> {
    /// The class-blended sensor map (stage C2).
    pub sensor_map: &'a SensorMapTables,
    /// `w_A` of the ALEXA estimate (stages C3 and C4).
    pub alexa_white: &'a AlexaWhite,
    /// The camera locus, for the CCT readout of a camera white and the camera white of a preset light.
    pub locus: &'a Locus,
}

/// The settings that decide the chain, each resolved from `DevelopSettings`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainSettings {
    /// The light preset.
    pub light_source: LightSource,
    /// Auto or the user's Kelvin and tint.
    pub white_balance: WhiteBalance,
    /// The exposure in stops, already resolved (`EV_auto` for Auto, the slider's absolute value for
    /// Manual).
    pub exposure_ev: f64,
    /// The Pick white sample (5x5 median of g0-balanced camera RGB) if a white was picked. It replaces
    /// the estimate's scene light before the preset is applied.
    pub picked_white: Option<Vec3>,
    /// The body's white scale (`docs/03`, White scale of stand-in bodies): the factor that takes its raw white
    /// to its profile's, `[1, 1, 1]` for a body whose profile was measured on it. Only Pick white needs it (the
    /// clip level of the sample); the rest of the chain works on the profile's scale already.
    pub white_scale: Vec3,
    /// Apple Auto's nudge of the white in mired (`docs/19` section 5; negative makes the picture
    /// warmer). It acts only while the white balance and the light source are Auto and no white is
    /// picked; 0, the default, is the chain without Apple Auto, bit for bit.
    pub white_shift_mired: f64,
    /// Apple Auto's nudge of the white's tint, in tint units (`docs/19` section 5; positive is more
    /// magenta, as a manual tint is), added to the scene's own tint and held in -50 to 50. It acts
    /// when the white shift does; 0, the default, is the chain without Apple Auto, bit for bit.
    pub tint_shift: f64,
}

impl Default for ChainSettings {
    fn default() -> Self {
        Self {
            light_source: LightSource::Auto,
            white_balance: WhiteBalance::Auto,
            exposure_ev: 0.0,
            picked_white: None,
            white_scale: wb::IDENTITY_WHITE_SCALE,
            white_shift_mired: 0.0,
            tint_shift: 0.0,
        }
    }
}

/// The clip level for the highlight range (stage C7), after C1 to C6.
///
/// A neutral surface brightens until the first camera channel clips. After C1 that surface is
/// `[s, s, s]` and channel `c` clips at `s = 1 / w_scene[c]`, so the first clip is at
/// `balanced = min_c 1 / w_scene[c]` (1 unless the light is redder or bluer than the green channel
/// sees). Mapped through the matrix, that point is the AWG4 triple `awg4`. C7's highlight range reads
/// it as a luma, [`ClipLevel::luma`] (the Rec.709 weights, the norm the stage applies to every pixel):
/// the Sony clip `Y_clip` of the range's formulas (`docs/02`, C7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipLevel {
    /// The smallest per-channel clip of a neutral surface after C1.
    pub balanced: f64,
    /// That point after C1 to C6: `M * [balanced; 3]`.
    pub awg4: Vec3,
    /// `max(awg4)`: its largest channel, a diagnostic.
    pub n: f64,
}

impl ClipLevel {
    /// The luma of the clipped neutral ([`tone::luma`] of [`ClipLevel::awg4`]): `Y_clip`, the clip level
    /// C7's highlight range takes ([`tone::ToneParams::new`]).
    #[must_use]
    pub fn luma(&self) -> f64 {
        tone::luma(self.awg4)
    }
}

/// The clip level of a chain matrix for a scene light (see [`ClipLevel`]).
#[must_use]
pub fn clip_level(matrix: &Mat3, w_scene: Vec3) -> ClipLevel {
    clip_level_scaled(matrix, w_scene, wb::IDENTITY_WHITE_SCALE)
}

/// The clip level for a body with a white scale (`docs/03`, White scale of stand-in bodies). The body's raw
/// clips at 1.0 per channel in its own scale, so a neutral surface of brightness `s` clips channel `c` at
/// `s = white_scale[c] / w_scene[c]`, `w_scene` being on the profile's scale. With the identity scale this is
/// [`clip_level`] to the last bit.
#[must_use]
pub fn clip_level_scaled(matrix: &Mat3, w_scene: Vec3, white_scale: Vec3) -> ClipLevel {
    let balanced = w_scene
        .iter()
        .zip(&white_scale)
        .map(|(w, s)| s / w)
        .fold(f64::INFINITY, f64::min);
    let awg4 = linalg::mul_vec(matrix, [balanced; 3]);
    let n = awg4.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ClipLevel { balanced, awg4, n }
}

/// The factors of the chain, for tests and for debugging a kernel stage by stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainStages {
    /// C1: residual gains, green = 1.
    pub g_res: Vec3,
    /// C2: the sensor map for the scene light.
    pub sensor_map: Mat3,
    /// C3: `w_A(E_scene)`, green = 1.
    pub w_a_scene: Vec3,
    /// C4: the user's gains (tint included).
    pub g_user: Vec3,
    /// The product of the C3 and C4 diagonals as used in the matrix: exactly `[1, 1, 1]` for Auto
    /// without Apple Auto's nudge.
    pub c3_c4: Vec3,
    /// The CCT that picks the ARRI matrix (the scene CCT for Auto, the nudged CCT for a nudged Auto,
    /// the user's Kelvin for Manual).
    pub user_cct_k: f64,
    /// C5: ARRI native to AWG3 at `user_cct_k`.
    pub arri_native_to_awg3: Mat3,
    /// C6: AWG3 to AWG4.
    pub awg3_to_awg4: Mat3,
    /// C6: `2^EV`.
    pub exposure_gain: f64,
}

/// The fused C1 to C6 chain of one photo.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedChain {
    matrix: Mat3,
    stages: ChainStages,
    scene: SceneLight,
    readout: WbReadout,
    clip: ClipLevel,
    exposure_ev: f64,
}

impl FusedChain {
    /// Builds the chain for a photo.
    ///
    /// * `estimate` is the illuminant estimate (the scene light at open, `AutoEstimates`);
    /// * `w_scene_initial` is the white the mosaic was balanced for (`g0 = 1 / w_scene_initial`);
    /// * a Pick white sample in the settings replaces the estimate's chromaticity and CCT (the classes
    ///   stay), then the light preset is applied to the result.
    ///
    /// # Errors
    ///
    /// [`ChainError`] for a non-finite exposure, an unusable white, a manual white balance out of range
    /// or a Pick white sample that [`wb::picked_scene_white`] refuses.
    pub fn build(
        tables: ProfileTables<'_>,
        estimate: &SceneLight,
        w_scene_initial: Vec3,
        settings: &ChainSettings,
    ) -> Result<Self, ChainError> {
        Self::build_with(
            tables,
            ArriTable::builtin(),
            estimate,
            w_scene_initial,
            settings,
        )
    }

    /// As [`FusedChain::build`] with an explicit ARRI table (tests and tools).
    ///
    /// # Errors
    ///
    /// As [`FusedChain::build`].
    pub fn build_with(
        tables: ProfileTables<'_>,
        arri: &ArriTable,
        estimate: &SceneLight,
        w_scene_initial: Vec3,
        settings: &ChainSettings,
    ) -> Result<Self, ChainError> {
        // `2^EV` must be a usable gain: a finite EV beyond about +-1000 stops overflows to infinity or
        // underflows to zero, and the matrix would be all infinities or NaNs.
        let exposure_gain = settings.exposure_ev.exp2();
        if !(settings.exposure_ev.is_finite() && exposure_gain.is_finite() && exposure_gain > 0.0) {
            return Err(ChainError::InvalidExposure(settings.exposure_ev));
        }
        settings.white_balance.validate()?;
        let w_init = wb::normalise_green(w_scene_initial)?;

        // The scene light: estimate, then Pick white, then the preset.
        let base = match settings.picked_white {
            Some(sample) => estimate.with_picked_white_scaled(
                sample,
                w_init,
                settings.white_scale,
                tables.locus,
            )?,
            None => *estimate,
        };
        let scene = settings.light_source.scene_light(&base, tables.locus);

        // C1 and C2.
        let g_res = wb::residual_gains(w_init, scene.w_scene)?;
        let sensor_map = tables.sensor_map.blend(&scene.classes, 1e6 / scene.cct_k);

        // C3 and C4.
        let w_a_scene = wb::scene_alexa_white(tables.alexa_white, tables.locus, &scene);
        let nudged = (settings.white_shift_mired != 0.0 || settings.tint_shift != 0.0)
            && settings.light_source == LightSource::Auto
            && settings.picked_white.is_none();
        let (g_user, c3_c4, user_cct_k, readout) = match settings.white_balance {
            // Apple Auto's nudge (`docs/19` section 5): the user's white is the scene's white moved
            // along the locus to the nudged CCT, keeping the scene's classes, with the scene's tint
            // plus the tint nudge, and the ARRI matrix takes that CCT, as for a manual Kelvin.
            WhiteBalance::Auto if nudged => {
                let (cct_k, w_a_user) = wb::nudged_alexa_white(
                    tables.alexa_white,
                    tables.locus,
                    &scene,
                    settings.white_shift_mired,
                    settings.tint_shift,
                );
                let g = w_a_user.map(|w| 1.0 / w);
                let product = [
                    g[0] * w_a_scene[0],
                    g[1] * w_a_scene[1],
                    g[2] * w_a_scene[2],
                ];
                (
                    g,
                    product,
                    cct_k.clamp(KELVIN_MIN_K, KELVIN_MAX_K),
                    wb::nudged_readout(
                        tables.alexa_white,
                        tables.locus,
                        &scene,
                        settings.white_shift_mired,
                        settings.tint_shift,
                    ),
                )
            }
            WhiteBalance::Auto => (
                w_a_scene.map(|w| 1.0 / w),
                [1.0; 3],
                scene.cct_k,
                wb::auto_readout(tables.alexa_white, tables.locus, &scene),
            ),
            WhiteBalance::Manual { kelvin, tint } => {
                let g = wb::manual_user_gains(tables.alexa_white, kelvin, tint);
                let product = [
                    g[0] * w_a_scene[0],
                    g[1] * w_a_scene[1],
                    g[2] * w_a_scene[2],
                ];
                (
                    g,
                    product,
                    kelvin.clamp(KELVIN_MIN_K, KELVIN_MAX_K),
                    WbReadout {
                        kelvin_k: kelvin,
                        tint,
                    },
                )
            }
        };

        // C5 and C6.
        let arri_native_to_awg3 = arri.native_to_awg3(user_cct_k);
        let awg3_to_awg4 = gamut::matrices().awg3_to_awg4;

        let mut matrix = linalg::mul(&sensor_map, &linalg::diag(g_res));
        matrix = linalg::mul(&linalg::diag(c3_c4), &matrix);
        matrix = linalg::mul(&arri_native_to_awg3, &matrix);
        matrix = linalg::mul(&awg3_to_awg4, &matrix);
        for row in &mut matrix {
            for v in row {
                *v *= exposure_gain;
            }
        }

        let clip = clip_level_scaled(&matrix, scene.w_scene, settings.white_scale);
        Ok(Self {
            matrix,
            stages: ChainStages {
                g_res,
                sensor_map,
                w_a_scene,
                g_user,
                c3_c4,
                user_cct_k,
                arri_native_to_awg3,
                awg3_to_awg4,
                exposure_gain,
            },
            scene,
            readout,
            clip,
            exposure_ev: settings.exposure_ev,
        })
    }

    /// The fused 3x3: g0-balanced camera RGB to linear AWG4.
    #[must_use]
    pub fn matrix(&self) -> &Mat3 {
        &self.matrix
    }

    /// Applies the matrix to one g0-balanced camera RGB triple.
    #[must_use]
    pub fn apply(&self, rgb: Vec3) -> LinearAwg4 {
        LinearAwg4(linalg::mul_vec(&self.matrix, rgb))
    }

    /// The factors that make up the matrix.
    #[must_use]
    pub fn stages(&self) -> &ChainStages {
        &self.stages
    }

    /// The scene light the chain used (after Pick white and the preset).
    #[must_use]
    pub fn scene(&self) -> &SceneLight {
        &self.scene
    }

    /// The Kelvin and tint the White balance rows show: the readout of the scene light for Auto, the
    /// user's own values for Manual.
    #[must_use]
    pub fn readout(&self) -> WbReadout {
        self.readout
    }

    /// The clip level for the highlight range (stage C7).
    #[must_use]
    pub fn clip(&self) -> &ClipLevel {
        &self.clip
    }

    /// The exposure in stops that went into the matrix.
    #[must_use]
    pub fn exposure_ev(&self) -> f64 {
        self.exposure_ev
    }
}

/// What the automatic development knows about a photo's colour, computed once at open and on Re-run
/// (the colour part of `AutoEstimates`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorEstimates {
    /// The illuminant estimate.
    pub scene: SceneLight,
    /// The white the mosaic was balanced for, on the profile's scale: `g0 = white_scale / w_scene_initial`
    /// (`1 / w_scene_initial` for a body whose profile was measured on it). Normally `scene.w_scene` at open.
    pub w_scene_initial: Vec3,
    /// The body's white scale (`docs/03`): `[1, 1, 1]` unless the body borrows a profile it does not match.
    /// It enters the initial gains ([`ColorEstimates::initial_gains`]) and the clip level of Pick white.
    pub white_scale: Vec3,
    /// The automatic exposure in stops (`crate::exposure`).
    pub ev_auto: f64,
    /// Apple Auto's nudge of the Auto white in mired (`docs/19` section 5), 0 without an answer: what
    /// [`ChainSettings::white_shift_mired`] gets.
    pub white_shift_mired: f64,
    /// Apple Auto's nudge of the Auto white's tint (`docs/19` section 5), 0 without an answer: what
    /// [`ChainSettings::tint_shift`] gets.
    pub tint_shift: f64,
}

impl ColorEstimates {
    /// The chain for a set of colour settings: the exposure setting resolves against `ev_auto`.
    ///
    /// # Errors
    ///
    /// As [`FusedChain::build`], and [`ChainError::Exposure`] for a manual exposure outside -4 to +4.
    pub fn chain(
        &self,
        tables: ProfileTables<'_>,
        light_source: LightSource,
        white_balance: WhiteBalance,
        exposure: ExposureSetting,
        picked_white: Option<Vec3>,
    ) -> Result<FusedChain, ChainError> {
        exposure.validate()?;
        FusedChain::build(
            tables,
            &self.scene,
            self.w_scene_initial,
            &ChainSettings {
                light_source,
                white_balance,
                exposure_ev: exposure.resolve(self.ev_auto),
                picked_white,
                white_scale: self.white_scale,
                white_shift_mired: self.white_shift_mired,
                tint_shift: self.tint_shift,
            },
        )
    }

    /// The initial gains `g0` of the raw stages, `white_scale / w_scene_initial` with green = 1
    /// ([`wb::initial_gains_scaled`]).
    ///
    /// # Errors
    ///
    /// [`WbError::InvalidWhite`] for an unusable white or white scale.
    pub fn initial_gains(&self) -> Result<Vec3, WbError> {
        wb::initial_gains_scaled(self.w_scene_initial, self.white_scale)
    }

    /// The chain with every colour setting on Auto and the exposure at 0 EV: the chain that the auto
    /// exposure meters through (with Apple Auto's nudge of the white, when there is one: its readout
    /// is the Auto white balance the rows show).
    ///
    /// # Errors
    ///
    /// As [`FusedChain::build`].
    pub fn metering_chain(&self, tables: ProfileTables<'_>) -> Result<FusedChain, ChainError> {
        FusedChain::build(
            tables,
            &self.scene,
            self.w_scene_initial,
            &ChainSettings {
                white_scale: self.white_scale,
                white_shift_mired: self.white_shift_mired,
                tint_shift: self.tint_shift,
                ..ChainSettings::default()
            },
        )
    }
}
