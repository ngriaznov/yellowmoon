//! The registry (`data/cameras/registry.json`): model ID to body, profile and match kind, plus
//! Model-string aliases (`docs/03-camera-profiles.md` sections "Lookup" and "Profile files"), and the
//! [`ProfileStore`] that loads the registry and every profile it lists, once.
//!
//! Lookup order is exact model ID, then same-sensor sibling, then the default profile (the A7R IV
//! stand-in): see [`crate::detect`]. A Sony file is never refused for lack of a profile.
//!
//! The data root is the folder that holds `cameras/` and `alexa/`. Callers pass `ym_core::data_dir()`'s
//! result; this crate does not locate it itself.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::profile::{Profile, ProfileError, ProfileId, Target};

/// Why a registry (or its agreement with the profile files) was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    /// Two bodies share a model ID.
    #[error("model ID {0} appears twice")]
    DuplicateModelId(u16),
    /// Two bodies share a model string.
    #[error("model string {0:?} appears twice")]
    DuplicateModel(String),
    /// Two profile entries share an id.
    #[error("profile {0} is listed twice")]
    DuplicateProfile(String),
    /// A body, or the default, names a profile that is not listed.
    #[error("{what} uses profile {profile}, which is not listed in `profiles`")]
    UnknownProfile {
        /// Who refers to it (`default_profile` or a body).
        what: String,
        /// The missing profile id.
        profile: String,
    },
    /// A profile has no `exact` body, or more than one.
    #[error("profile {profile} has {count} exact bodies, expected exactly 1")]
    ExactBodies {
        /// The profile id.
        profile: String,
        /// How many bodies claim an exact match.
        count: usize,
    },
    /// A `sensor_family` body's `via` is missing or is not its profile's measured body.
    #[error("{model} (model ID {model_id}): `via` is {via:?}, expected {expected:?}")]
    BadVia {
        /// The model string.
        model: String,
        /// The model ID.
        model_id: u16,
        /// What the entry says.
        via: Option<String>,
        /// What it should say (the display name of the profile's measured body, or none).
        expected: Option<String>,
    },
    /// A `default` body uses a profile other than `default_profile`.
    #[error(
        "{model} (model ID {model_id}) has match `default` but uses profile {profile}, not the default profile"
    )]
    DefaultBodyProfile {
        /// The model string.
        model: String,
        /// The model ID.
        model_id: u16,
        /// The profile it uses.
        profile: String,
    },
    /// `model_strings` and `bodies` disagree.
    #[error("model_strings maps {model:?} to {mapped}, but `bodies` says {actual:?}")]
    ModelStringMismatch {
        /// The model string.
        model: String,
        /// The ID in `model_strings`.
        mapped: u16,
        /// The ID in `bodies`, if the model is listed there.
        actual: Option<u16>,
    },
    /// A body is missing from `model_strings`.
    #[error("{model:?} (model ID {model_id}) is missing from model_strings")]
    ModelStringMissing {
        /// The model string.
        model: String,
        /// The model ID.
        model_id: u16,
    },
    /// A profile entry's path does not point into `profiles/<id>/profile.json`.
    #[error("profile {id} has path {path:?}, expected profiles/{id}/profile.json")]
    BadPath {
        /// The profile id.
        id: String,
        /// The path in the registry.
        path: String,
    },
    /// A body's sensor fields are incomplete or give an implausible pixel pitch.
    #[error("{model} (model ID {model_id}): {reason}")]
    BadSensor {
        /// The model string.
        model: String,
        /// The model ID.
        model_id: u16,
        /// What is wrong.
        reason: String,
    },
    /// A body's white scale is not a usable `[r, 1, b]`, or does not agree with how the body is matched
    /// or with its source (`docs/03`, White scale of stand-in bodies).
    #[error("{model} (model ID {model_id}): white scale: {reason}")]
    BadWhiteScale {
        /// The model string.
        model: String,
        /// The model ID.
        model_id: u16,
        /// What is wrong.
        reason: String,
    },
    /// A profile file's own id differs from the registry's.
    #[error("{path} holds profile {found}, the registry lists it as {expected}")]
    IdMismatch {
        /// The profile file.
        path: String,
        /// The id in the registry.
        expected: String,
        /// The id inside the file.
        found: String,
    },
}

/// How a body relates to its profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyMatch {
    /// The profile was measured on this body.
    Exact,
    /// The body shares its sensor with the profile's measured body.
    SensorFamily,
    /// No profile of its own or a sibling's: the default (A7R IV) is borrowed.
    Default,
}

/// A profile entry of the registry: its id and file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryProfile {
    /// Profile id.
    pub id: ProfileId,
    /// File relative to `data/cameras/`, `profiles/<id>/profile.json`.
    pub path: String,
}

/// Where the registry's model IDs and table come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistrySources {
    /// Where the model IDs and Model strings come from (the values the cameras write).
    pub model_ids: String,
    /// The table in `docs/03` the registry is generated from.
    pub table: String,
    /// Where the sensor geometry of the bodies comes from and how it was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensors: Option<String>,
}

/// How the sensor fields of a body were read (`sensor_source_kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorSourceKind {
    /// From an excerpt of the maker's specification page.
    Maker,
    /// The maker's specification as restated by a retailer or reference site (the maker's page was
    /// not seen).
    Restated,
}

/// The pixel pitch of the smallest and the largest sensor the registry may hold, micrometres: the
/// sanity range of `docs/algorithms/03-diffraction-deconvolution.md` section 2.2.
pub const PIXEL_PITCH_RANGE_UM: std::ops::RangeInclusive<f64> = 0.5..=20.0;

/// The published geometry of a body's sensor (`docs/algorithms/03` section 2.2): what the diffraction
/// deconvolution needs to turn an f-number into a blur in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorGeometry {
    /// Width of the full sensor, millimetres.
    pub sensor_width_mm: f64,
    /// Width in pixels of the largest 3:2 still recorded from the full sensor. A crop mode does not
    /// change it, and it is what a Lossless M or S file was reduced from.
    pub active_width_px: u32,
    /// The pixel pitch, micrometres: the registry's `pixel_pitch_um` when the maker states one, else
    /// `1000 * sensor_width_mm / active_width_px`. Square pixels.
    pub pixel_pitch_um: f64,
}

/// The white scale that changes nothing: a body whose profile was measured on it.
pub const IDENTITY_WHITE_SCALE: [f64; 3] = [1.0, 1.0, 1.0];

/// The largest and the smallest factor a white scale may have in red or blue: the bodies Sony makes
/// differ from one another's curves by tens of percent, not by a factor of four.
pub const WHITE_SCALE_RANGE: std::ops::RangeInclusive<f64> = 0.25..=4.0;

/// The most, in percent, that the files a `presets` scale was fitted on may differ from one another in red or blue:
/// each file's own scale lies within this of the median of the files (`docs/03`, consistency of a body's files). It is
/// the unit-to-unit spread of the presets in the files of bodies with curves of their own (4.7 % at most).
pub const WHITE_SCALE_FILE_AGREEMENT_PERCENT: f64 = 5.0;

fn identity_white_scale() -> [f64; 3] {
    IDENTITY_WHITE_SCALE
}

/// How a body's white scale was found (`white_scale_source.kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhiteScaleKind {
    /// The profile was measured on this body: the scale is `[1, 1, 1]`.
    OwnCurves,
    /// The body shares the sensor of the profile's measured body: the scale is `[1, 1, 1]`.
    SameSensor,
    /// Fitted on the white-balance presets stored in the body's own files.
    Presets,
    /// Taken from another body whose published colour matrix is the same.
    Sibling,
    /// Nothing was measured: the scale is `[1, 1, 1]` and the body's whites are not corrected.
    None,
}

/// The set a file of a `presets` white scale comes from; each has a fetch script that reproduces it
/// (`docs/03`, Provenance per body). Only sets of files whose licence allows any use appear here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PhotoSet {
    /// The owner's A1 photos (`tools/fetch/owner-photos.sh`).
    OwnerPhotos,
    /// Play Raw files of discuss.pixls.us (`tools/fetch/white-scale-photos.sh`).
    PlayRaw,
}

/// One file a `presets` white scale was fitted on: its name, its SHA-256 and the set it comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteScaleFile {
    /// The file name (`A1M01454.ARW`).
    pub name: String,
    /// SHA-256 of the file, 64 lower-case hex digits.
    pub sha256: String,
    /// The set (and so the fetch script) the file comes from.
    pub set: PhotoSet,
}

/// The fit of a `presets` white scale against the presets of its files, and against the published colour
/// matrix where the body has one (`docs/03`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteScaleResidual {
    /// Mean angle between the five Kelvin presets and the locus white at their labels, degrees, with no scale
    /// (the mean over the files kept).
    pub mean_preset_angle_before_deg: f64,
    /// The same with the scale applied.
    pub mean_preset_angle_after_deg: f64,
    /// The colour temperature the profile's locus reads for the Daylight preset, with no scale.
    pub daylight_k_before: f64,
    /// The same with the scale applied.
    pub daylight_k_after: f64,
    /// The largest distance, in percent, of a kept file's own scale from the median of the kept files, in red
    /// and in blue; at most [`WHITE_SCALE_FILE_AGREEMENT_PERCENT`].
    pub file_spread_percent: [f64; 2],
    /// The scale the body's published colour matrix at D65 would give: `[r, 1, b]`. Absent, with the next two
    /// fields, for a body that has no published matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_scale: Option<[f64; 3]>,
    /// The mean preset angle with the matrix scale applied, degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_mean_preset_angle_deg: Option<f64>,
    /// The Daylight preset's reading with the matrix scale applied, kelvin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_daylight_k: Option<f64>,
}

/// Where a body's white scale comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteScaleSource {
    /// How it was found.
    pub kind: WhiteScaleKind,
    /// The files whose presets it was fitted on, for `presets`; empty otherwise.
    pub files: Vec<WhiteScaleFile>,
    /// The fit's residuals, for `presets`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual: Option<WhiteScaleResidual>,
    /// What was done and why, in words.
    pub note: String,
}

/// One camera body of the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    /// Sony maker-note model ID (tag 0xB001).
    pub model_id: u16,
    /// IFD0 Model string, as the camera writes it (`ILCE-7RM4`).
    pub model: String,
    /// Display name (`Sony A7R IV`).
    pub body: String,
    /// The profile this body uses.
    pub profile: ProfileId,
    /// How it relates to that profile.
    #[serde(rename = "match")]
    pub match_kind: BodyMatch,
    /// For `sensor_family`: the display name of the profile's measured body; null otherwise.
    pub via: Option<String>,
    /// Free-text notes (measurement source, the sibling check).
    pub notes: String,
    /// `[r, 1, b]`: the factor that takes this body's raw white to the white of its profile's curves,
    /// `w_profile = w_body * white_scale` (`docs/03`, White scale of stand-in bodies). `[1, 1, 1]` for a
    /// body whose profile was measured on it or on its sensor, and when the file says nothing.
    #[serde(default = "identity_white_scale")]
    pub white_scale: [f64; 3],
    /// Where the white scale comes from; absent in a registry written before the scale existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white_scale_source: Option<WhiteScaleSource>,
    /// Width of the full sensor in millimetres, from the maker's specification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor_width_mm: Option<f64>,
    /// Width in pixels of the largest 3:2 still from the full sensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_width_px: Option<u32>,
    /// The pixel pitch in micrometres when the maker states it (overrides the computed one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_pitch_um: Option<f64>,
    /// How the sensor fields were read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor_source_kind: Option<SensorSourceKind>,
    /// The pages the sensor fields were read from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor_source: Option<String>,
}

impl Body {
    /// Checks the sensor fields: width, pixel count and source come together, and the pixel pitch is
    /// in [`PIXEL_PITCH_RANGE_UM`].
    fn validate_sensor(&self) -> Result<(), RegistryError> {
        let bad = |reason: &str| RegistryError::BadSensor {
            model: self.model.clone(),
            model_id: self.model_id,
            reason: reason.to_string(),
        };
        let complete = self.sensor_width_mm.is_some()
            && self.active_width_px.is_some()
            && self.sensor_source_kind.is_some()
            && self.sensor_source.as_deref().is_some_and(|s| !s.is_empty());
        let any = self.sensor_width_mm.is_some()
            || self.active_width_px.is_some()
            || self.pixel_pitch_um.is_some()
            || self.sensor_source_kind.is_some()
            || self.sensor_source.is_some();
        if !any {
            return Ok(());
        }
        if !complete {
            return Err(bad(
                "sensor_width_mm, active_width_px, sensor_source_kind and sensor_source come together",
            ));
        }
        let Some(sensor) = self.sensor() else {
            return Err(bad("incomplete sensor fields"));
        };
        if !(sensor.sensor_width_mm.is_finite() && sensor.sensor_width_mm > 0.0)
            || sensor.active_width_px == 0
        {
            return Err(bad("the sensor width and the pixel count must be positive"));
        }
        if !PIXEL_PITCH_RANGE_UM.contains(&sensor.pixel_pitch_um) {
            return Err(bad(&format!(
                "pixel pitch {:.3} um is outside {} to {} um",
                sensor.pixel_pitch_um,
                PIXEL_PITCH_RANGE_UM.start(),
                PIXEL_PITCH_RANGE_UM.end()
            )));
        }
        Ok(())
    }

    /// Checks the white scale: green exactly 1, red and blue positive and in [`WHITE_SCALE_RANGE`]; `[1, 1, 1]` for
    /// an `exact` or `sensor_family` body and for kind `none`; a `source` whose kind fits how the body is matched;
    /// files and residuals only where they belong.
    fn validate_white_scale(&self) -> Result<(), RegistryError> {
        let bad = |reason: &str| RegistryError::BadWhiteScale {
            model: self.model.clone(),
            model_id: self.model_id,
            reason: reason.to_string(),
        };
        let [r, g, b] = self.white_scale;
        if g != 1.0
            || ![r, b]
                .iter()
                .all(|v| v.is_finite() && WHITE_SCALE_RANGE.contains(v))
        {
            return Err(bad(&format!(
                "{:?} is not [r, 1, b] with r and b between {} and {}",
                self.white_scale,
                WHITE_SCALE_RANGE.start(),
                WHITE_SCALE_RANGE.end()
            )));
        }
        let identity = self.white_scale == IDENTITY_WHITE_SCALE;
        let Some(source) = &self.white_scale_source else {
            // A registry written before the scale existed: only the identity can stand without a source.
            return if identity {
                Ok(())
            } else {
                Err(bad(
                    "a scale other than [1, 1, 1] needs a white_scale_source",
                ))
            };
        };
        if source.note.trim().is_empty() {
            return Err(bad("the source needs a note"));
        }
        let kind_ok = match self.match_kind {
            BodyMatch::Exact => source.kind == WhiteScaleKind::OwnCurves,
            BodyMatch::SensorFamily => source.kind == WhiteScaleKind::SameSensor,
            BodyMatch::Default => matches!(
                source.kind,
                WhiteScaleKind::Presets | WhiteScaleKind::Sibling | WhiteScaleKind::None
            ),
        };
        if !kind_ok {
            return Err(bad(&format!(
                "source kind {:?} does not fit match {:?}",
                source.kind, self.match_kind
            )));
        }
        if matches!(
            source.kind,
            WhiteScaleKind::OwnCurves | WhiteScaleKind::SameSensor | WhiteScaleKind::None
        ) && !identity
        {
            return Err(bad("this source kind means [1, 1, 1]"));
        }
        if matches!(
            source.kind,
            WhiteScaleKind::Presets | WhiteScaleKind::Sibling
        ) && identity
        {
            return Err(bad("a fitted or inherited scale is not [1, 1, 1]"));
        }
        if (source.kind == WhiteScaleKind::Presets) != source.residual.is_some() {
            return Err(bad("the residuals belong to kind presets, and only to it"));
        }
        if (source.kind == WhiteScaleKind::Presets) == source.files.is_empty() {
            return Err(bad("files are named for kind presets, and only for it"));
        }
        for (i, file) in source.files.iter().enumerate() {
            let hex = file.sha256.len() == 64
                && file
                    .sha256
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c));
            if !hex {
                return Err(bad(&format!(
                    "{}: the SHA-256 is not 64 lower-case hex digits",
                    file.name
                )));
            }
            if !file.name.to_ascii_uppercase().ends_with(".ARW") || file.name.contains('/') {
                return Err(bad(&format!("{}: not a plain ARW file name", file.name)));
            }
            if source.files[..i].iter().any(|o| o.name == file.name) {
                return Err(bad(&format!("{} is listed twice", file.name)));
            }
        }
        if let Some(res) = &source.residual {
            let [sr, sb] = res.file_spread_percent;
            if !(0.0..=WHITE_SCALE_FILE_AGREEMENT_PERCENT).contains(&sr)
                || !(0.0..=WHITE_SCALE_FILE_AGREEMENT_PERCENT).contains(&sb)
            {
                return Err(bad(&format!(
                    "the files disagree: {sr} % and {sb} % from their median against a limit of {WHITE_SCALE_FILE_AGREEMENT_PERCENT} %"
                )));
            }
            let matrix = [
                res.matrix_scale.is_some(),
                res.matrix_mean_preset_angle_deg.is_some(),
                res.matrix_daylight_k.is_some(),
            ];
            if matrix.iter().any(|m| *m) && !matrix.iter().all(|m| *m) {
                return Err(bad(
                    "the matrix reference is all or nothing: matrix_scale, its angle and its Daylight reading",
                ));
            }
        }
        Ok(())
    }

    /// The published sensor geometry, `None` when the registry has none for this body (the EXIF
    /// crop-factor fallback then applies). A body whose fields are incomplete is refused by
    /// [`Registry::validate`], so for a validated registry this is all or nothing.
    #[must_use]
    pub fn sensor(&self) -> Option<SensorGeometry> {
        let (width_mm, width_px) = (self.sensor_width_mm?, self.active_width_px?);
        let computed = 1000.0 * width_mm / f64::from(width_px);
        Some(SensorGeometry {
            sensor_width_mm: width_mm,
            active_width_px: width_px,
            pixel_pitch_um: self.pixel_pitch_um.unwrap_or(computed),
        })
    }
}

/// `data/cameras/registry.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    /// Semver of the registry.
    pub version: semver::Version,
    /// Profile for every body without an entry of its own (the A7R IV stand-in).
    pub default_profile: ProfileId,
    /// Origin of the table.
    pub sources: RegistrySources,
    /// The profiles and their files.
    pub profiles: Vec<RegistryProfile>,
    /// The known bodies, one per model ID.
    pub bodies: Vec<Body>,
    /// Model string to model ID, for files without a usable maker-note ID.
    pub model_strings: BTreeMap<String, u16>,
}

/// Normalises a Model string for comparison: trimmed of whitespace and NULs, a leading `Sony ` removed,
/// ASCII case folded.
pub(crate) fn normalize_model(s: &str) -> String {
    let t = s.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    let t = match t.get(..5) {
        Some(prefix) if prefix.eq_ignore_ascii_case("sony ") => &t[5..],
        _ => t,
    };
    t.trim().to_ascii_lowercase()
}

impl Registry {
    /// Parses and validates the text of a `registry.json`.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Json`] for malformed JSON, [`ProfileError::Registry`] when
    /// [`Registry::validate`] fails.
    pub fn parse(text: &str) -> Result<Self, ProfileError> {
        let registry: Registry =
            serde_json::from_str(text).map_err(|source| ProfileError::Json {
                path: PathBuf::from("<text>"),
                source,
            })?;
        registry.validate()?;
        Ok(registry)
    }

    /// Checks what the schema cannot express (the Rust twin of `validate_profiles.check_registry`):
    /// model IDs, model strings and profile ids unique; the default and every body's profile listed;
    /// every profile has exactly one `exact` body; `via` names that body for `sensor_family` and is null
    /// otherwise; every `default` body uses `default_profile`; `model_strings` agrees with `bodies`
    /// (each body once, nothing else); profile paths are `profiles/<id>/profile.json`; each body's white scale
    /// is usable and agrees with its match kind and source (`docs/03`, White scale of stand-in bodies).
    ///
    /// # Errors
    ///
    /// The first problem found, as a [`RegistryError`].
    pub fn validate(&self) -> Result<(), RegistryError> {
        let mut seen_profiles = Vec::new();
        for p in &self.profiles {
            if seen_profiles.contains(&&p.id) {
                return Err(RegistryError::DuplicateProfile(p.id.to_string()));
            }
            seen_profiles.push(&p.id);
            if p.path != format!("profiles/{}/profile.json", p.id) {
                return Err(RegistryError::BadPath {
                    id: p.id.to_string(),
                    path: p.path.clone(),
                });
            }
        }
        let listed = |id: &ProfileId| self.profiles.iter().any(|p| &p.id == id);
        if !listed(&self.default_profile) {
            return Err(RegistryError::UnknownProfile {
                what: "default_profile".to_string(),
                profile: self.default_profile.to_string(),
            });
        }
        for (i, b) in self.bodies.iter().enumerate() {
            if self.bodies[..i].iter().any(|o| o.model_id == b.model_id) {
                return Err(RegistryError::DuplicateModelId(b.model_id));
            }
            if self.bodies[..i]
                .iter()
                .any(|o| normalize_model(&o.model) == normalize_model(&b.model))
            {
                return Err(RegistryError::DuplicateModel(b.model.clone()));
            }
            if !listed(&b.profile) {
                return Err(RegistryError::UnknownProfile {
                    what: format!("{} (model ID {})", b.model, b.model_id),
                    profile: b.profile.to_string(),
                });
            }
            if b.match_kind == BodyMatch::Default && b.profile != self.default_profile {
                return Err(RegistryError::DefaultBodyProfile {
                    model: b.model.clone(),
                    model_id: b.model_id,
                    profile: b.profile.to_string(),
                });
            }
            b.validate_sensor()?;
            b.validate_white_scale()?;
        }
        for p in &self.profiles {
            let exact: Vec<&Body> = self
                .bodies
                .iter()
                .filter(|b| b.profile == p.id && b.match_kind == BodyMatch::Exact)
                .collect();
            if exact.len() != 1 {
                return Err(RegistryError::ExactBodies {
                    profile: p.id.to_string(),
                    count: exact.len(),
                });
            }
        }
        for b in &self.bodies {
            let expected = if b.match_kind == BodyMatch::SensorFamily {
                self.exact_body(&b.profile).map(|e| e.body.clone())
            } else {
                None
            };
            if b.via != expected {
                return Err(RegistryError::BadVia {
                    model: b.model.clone(),
                    model_id: b.model_id,
                    via: b.via.clone(),
                    expected,
                });
            }
            if self.model_strings.get(&b.model) != Some(&b.model_id) {
                return Err(RegistryError::ModelStringMissing {
                    model: b.model.clone(),
                    model_id: b.model_id,
                });
            }
        }
        for (model, &mapped) in &self.model_strings {
            let actual = self
                .bodies
                .iter()
                .find(|b| &b.model == model)
                .map(|b| b.model_id);
            if actual != Some(mapped) {
                return Err(RegistryError::ModelStringMismatch {
                    model: model.clone(),
                    mapped,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// The body with this Sony model ID.
    #[must_use]
    pub fn body_by_model_id(&self, model_id: u16) -> Option<&Body> {
        self.bodies.iter().find(|b| b.model_id == model_id)
    }

    /// The body a Model string names, through `model_strings`. Case and surrounding whitespace do not
    /// matter, and a leading `Sony ` is ignored.
    #[must_use]
    pub fn body_by_model_string(&self, model: &str) -> Option<&Body> {
        let wanted = normalize_model(model);
        let (_, &id) = self
            .model_strings
            .iter()
            .find(|(k, _)| normalize_model(k) == wanted)?;
        self.body_by_model_id(id)
    }

    /// The one body whose profile was measured on it.
    #[must_use]
    pub fn exact_body(&self, profile: &ProfileId) -> Option<&Body> {
        self.bodies
            .iter()
            .find(|b| &b.profile == profile && b.match_kind == BodyMatch::Exact)
    }

    /// The registry entry of a profile.
    #[must_use]
    pub fn profile_entry(&self, id: &ProfileId) -> Option<&RegistryProfile> {
        self.profiles.iter().find(|p| &p.id == id)
    }
}

/// The registry, the ALEXA target and every camera profile, loaded once from a data root.
///
/// Profiles are shared (`Arc`) so a photo can keep the one it was developed with. Loading validates
/// everything ([`Profile::validate`], [`Registry::validate`]) and checks that each profile file holds the
/// id the registry lists it under.
#[derive(Debug, Clone)]
pub struct ProfileStore {
    data_root: PathBuf,
    registry: Registry,
    target: Target,
    profiles: BTreeMap<ProfileId, Arc<Profile>>,
}

impl ProfileStore {
    /// Loads `cameras/registry.json`, every profile it lists and `alexa/target.json` from `data_root`.
    ///
    /// # Errors
    ///
    /// [`ProfileError`] for a missing or malformed file, a registry or profile that breaks the format's
    /// rules, or a profile file whose id differs from the registry's.
    pub fn load(data_root: impl AsRef<Path>) -> Result<Self, ProfileError> {
        let data_root = data_root.as_ref();
        let cameras = data_root.join("cameras");
        let registry_path = cameras.join("registry.json");
        let text = std::fs::read_to_string(&registry_path).map_err(|source| ProfileError::Io {
            path: registry_path.clone(),
            source,
        })?;
        let registry: Registry =
            serde_json::from_str(&text).map_err(|source| ProfileError::Json {
                path: registry_path.clone(),
                source,
            })?;
        registry.validate()?;
        let mut profiles = BTreeMap::new();
        for entry in &registry.profiles {
            let profile = Profile::from_path(cameras.join(&entry.path))?;
            if profile.id != entry.id {
                return Err(RegistryError::IdMismatch {
                    path: entry.path.clone(),
                    expected: entry.id.to_string(),
                    found: profile.id.to_string(),
                }
                .into());
            }
            profiles.insert(entry.id.clone(), Arc::new(profile));
        }
        let target = Target::from_path(data_root.join("alexa").join("target.json"))?;
        Ok(Self {
            data_root: data_root.to_path_buf(),
            registry,
            target,
            profiles,
        })
    }

    /// The data root this store was loaded from.
    #[must_use]
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// The registry.
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// The emulation target description.
    #[must_use]
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// A profile by id.
    #[must_use]
    pub fn profile(&self, id: &ProfileId) -> Option<&Arc<Profile>> {
        self.profiles.get(id)
    }

    /// A profile by id text, for example from a command-line flag.
    #[must_use]
    pub fn profile_by_name(&self, id: &str) -> Option<&Arc<Profile>> {
        self.profiles.get(id)
    }

    /// The default profile (A7R IV): always present, checked at load.
    ///
    /// # Panics
    ///
    /// Never for a store built by [`ProfileStore::load`]: the registry's default is checked to be listed
    /// and every listed profile is loaded.
    #[must_use]
    pub fn default_profile(&self) -> &Arc<Profile> {
        // Invariant of `load`: `default_profile` is listed and every listed profile was loaded.
        &self.profiles[&self.registry.default_profile]
    }

    /// All profiles, ordered by id.
    pub fn profiles(&self) -> impl Iterator<Item = &Arc<Profile>> {
        self.profiles.values()
    }

    /// The ids of all profiles, ordered.
    pub fn profile_ids(&self) -> impl Iterator<Item = &ProfileId> {
        self.profiles.keys()
    }

    /// Number of cameras with a profile of their own (the "Eleven Sony cameras have their own" of the
    /// camera card): one per profile, since each has exactly one `exact` body.
    #[must_use]
    pub fn own_profile_count(&self) -> usize {
        self.profiles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_registry() -> Registry {
        let id = |s: &str| ProfileId::new(s).unwrap();
        let body =
            |model_id, model: &str, name: &str, profile: &str, kind, via: Option<&str>| Body {
                model_id,
                model: model.to_string(),
                body: name.to_string(),
                profile: id(profile),
                match_kind: kind,
                via: via.map(str::to_string),
                notes: String::new(),
                white_scale: IDENTITY_WHITE_SCALE,
                white_scale_source: None,
                sensor_width_mm: None,
                active_width_px: None,
                pixel_pitch_um: None,
                sensor_source_kind: None,
                sensor_source: None,
            };
        Registry {
            version: semver::Version::new(1, 0, 0),
            default_profile: id("p-a"),
            sources: RegistrySources {
                model_ids: String::new(),
                table: String::new(),
                sensors: None,
            },
            profiles: vec![
                RegistryProfile {
                    id: id("p-a"),
                    path: "profiles/p-a/profile.json".to_string(),
                },
                RegistryProfile {
                    id: id("p-b"),
                    path: "profiles/p-b/profile.json".to_string(),
                },
            ],
            bodies: vec![
                body(1, "AA-1", "Sony A1", "p-a", BodyMatch::Exact, None),
                body(2, "BB-2", "Sony B2", "p-b", BodyMatch::Exact, None),
                body(
                    3,
                    "BB-3",
                    "Sony B3",
                    "p-b",
                    BodyMatch::SensorFamily,
                    Some("Sony B2"),
                ),
                body(4, "NEW-4", "Sony N4", "p-a", BodyMatch::Default, None),
            ],
            model_strings: [("AA-1", 1), ("BB-2", 2), ("BB-3", 3), ("NEW-4", 4)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        }
    }

    #[test]
    fn a_consistent_registry_validates_and_looks_up() {
        let r = tiny_registry();
        r.validate().unwrap();
        assert_eq!(r.body_by_model_id(3).unwrap().model, "BB-3");
        assert!(r.body_by_model_id(99).is_none());
        for spelling in ["BB-3", " bb-3 ", "Sony BB-3", "SONY bb-3\0"] {
            assert_eq!(
                r.body_by_model_string(spelling).unwrap().model_id,
                3,
                "{spelling:?}"
            );
        }
        assert!(r.body_by_model_string("BB-33").is_none());
        assert_eq!(
            r.exact_body(&ProfileId::new("p-b").unwrap())
                .unwrap()
                .model_id,
            2
        );
    }

    #[test]
    fn inconsistent_registries_are_refused() {
        let mut r = tiny_registry();
        r.bodies[3].model_id = 1;
        assert_eq!(r.validate(), Err(RegistryError::DuplicateModelId(1)));

        let mut r = tiny_registry();
        r.bodies[1].profile = ProfileId::new("p-zzz").unwrap();
        assert!(matches!(
            r.validate(),
            Err(RegistryError::UnknownProfile { .. })
        ));

        let mut r = tiny_registry();
        r.bodies[2].match_kind = BodyMatch::Exact;
        r.bodies[2].via = None;
        assert!(matches!(
            r.validate(),
            Err(RegistryError::ExactBodies { count: 2, .. })
        ));

        let mut r = tiny_registry();
        r.bodies[2].via = Some("Sony Q1".to_string());
        assert!(matches!(r.validate(), Err(RegistryError::BadVia { .. })));

        let mut r = tiny_registry();
        r.bodies[3].via = Some("Sony A1".to_string());
        assert!(matches!(r.validate(), Err(RegistryError::BadVia { .. })));

        let mut r = tiny_registry();
        r.bodies[3].profile = ProfileId::new("p-b").unwrap();
        assert!(matches!(
            r.validate(),
            Err(RegistryError::DefaultBodyProfile { .. })
        ));

        let mut r = tiny_registry();
        r.model_strings.insert("BB-3".to_string(), 2);
        assert!(matches!(
            r.validate(),
            Err(RegistryError::ModelStringMissing { .. })
        ));

        let mut r = tiny_registry();
        r.model_strings.insert("EXTRA".to_string(), 1);
        assert!(matches!(
            r.validate(),
            Err(RegistryError::ModelStringMismatch { .. })
        ));

        let mut r = tiny_registry();
        r.profiles[0].path = "profiles/other/profile.json".to_string();
        assert!(matches!(r.validate(), Err(RegistryError::BadPath { .. })));

        let mut r = tiny_registry();
        r.default_profile = ProfileId::new("p-q").unwrap();
        assert!(matches!(
            r.validate(),
            Err(RegistryError::UnknownProfile { .. })
        ));

        let mut r = tiny_registry();
        r.profiles.push(r.profiles[0].clone());
        assert!(matches!(
            r.validate(),
            Err(RegistryError::DuplicateProfile(_))
        ));
    }

    fn source(kind: WhiteScaleKind) -> WhiteScaleSource {
        WhiteScaleSource {
            kind,
            files: Vec::new(),
            residual: None,
            note: "n".to_string(),
        }
    }

    fn file(name: &str) -> WhiteScaleFile {
        WhiteScaleFile {
            name: name.to_string(),
            sha256: "ab".repeat(32),
            set: PhotoSet::PlayRaw,
        }
    }

    fn presets_source() -> WhiteScaleSource {
        WhiteScaleSource {
            files: vec![file("a.ARW")],
            residual: Some(WhiteScaleResidual {
                mean_preset_angle_before_deg: 3.0,
                mean_preset_angle_after_deg: 0.5,
                daylight_k_before: 4400.0,
                daylight_k_after: 5200.0,
                file_spread_percent: [0.4, 2.0],
                matrix_scale: Some([0.9, 1.0, 1.0]),
                matrix_mean_preset_angle_deg: Some(2.0),
                matrix_daylight_k: Some(4700.0),
            }),
            ..source(WhiteScaleKind::Presets)
        }
    }

    #[test]
    fn white_scales_are_checked_against_how_the_body_is_matched() {
        let refused =
            |r: &Registry| matches!(r.validate(), Err(RegistryError::BadWhiteScale { .. }));
        // Without a source only the identity can stand (a registry written before the scale existed).
        let mut r = tiny_registry();
        r.validate().unwrap();
        r.bodies[3].white_scale = [0.9, 1.0, 1.1];
        assert!(refused(&r));
        // A stand-in body with fitted presets.
        r.bodies[3].white_scale_source = Some(presets_source());
        r.validate().unwrap();
        // Sources of the wrong kind for the match.
        let mut bad = r.clone();
        bad.bodies[3].white_scale_source = Some(source(WhiteScaleKind::OwnCurves));
        assert!(refused(&bad));
        let mut bad = r.clone();
        bad.bodies[0].white_scale_source = Some(source(WhiteScaleKind::Presets));
        assert!(refused(&bad));
        let mut bad = r.clone();
        bad.bodies[2].white_scale_source = Some(source(WhiteScaleKind::OwnCurves));
        assert!(refused(&bad));
        // An exact or same-sensor body, and a body that measured nothing, is [1, 1, 1].
        let mut bad = r.clone();
        bad.bodies[0].white_scale = [0.9, 1.0, 1.1];
        bad.bodies[0].white_scale_source = Some(source(WhiteScaleKind::OwnCurves));
        assert!(refused(&bad));
        let mut bad = r.clone();
        bad.bodies[3].white_scale_source = Some(source(WhiteScaleKind::None));
        assert!(refused(&bad));
        // Presets need files and residuals; the others have neither.
        let mut bad = r.clone();
        bad.bodies[3].white_scale_source = Some(WhiteScaleSource {
            files: Vec::new(),
            ..presets_source()
        });
        assert!(refused(&bad));
        let mut bad = r.clone();
        bad.bodies[3].white_scale_source = Some(WhiteScaleSource {
            residual: None,
            ..presets_source()
        });
        assert!(refused(&bad));
        let mut bad = r.clone();
        bad.bodies[3].white_scale_source = Some(WhiteScaleSource {
            files: vec![file("a.ARW")],
            ..source(WhiteScaleKind::Sibling)
        });
        assert!(refused(&bad));
        // The files are pinned and agree: a bad hash, a name that is not an ARW, a repeated name, a spread over the
        // limit, a half-given matrix reference are all refused; no matrix at all is fine.
        let with_files = |files: Vec<WhiteScaleFile>| WhiteScaleSource {
            files,
            ..presets_source()
        };
        for files in [
            vec![WhiteScaleFile {
                sha256: "AB".repeat(32),
                ..file("a.ARW")
            }],
            vec![WhiteScaleFile {
                sha256: "ab".repeat(31),
                ..file("a.ARW")
            }],
            vec![file("a.jpg")],
            vec![file("../a.ARW")],
            vec![file("a.ARW"), file("a.ARW")],
        ] {
            let mut bad = r.clone();
            bad.bodies[3].white_scale_source = Some(with_files(files.clone()));
            assert!(refused(&bad), "{files:?}");
        }
        let mut two = r.clone();
        two.bodies[3].white_scale_source = Some(with_files(vec![file("a.ARW"), file("b.ARW")]));
        two.validate().unwrap();
        for spread in [[5.01, 0.0], [0.0, 7.5], [-0.1, 0.0]] {
            let mut bad = r.clone();
            let mut src = presets_source();
            src.residual.as_mut().unwrap().file_spread_percent = spread;
            bad.bodies[3].white_scale_source = Some(src);
            assert!(refused(&bad), "{spread:?}");
        }
        let mut bad = r.clone();
        let mut src = presets_source();
        src.residual.as_mut().unwrap().matrix_daylight_k = None;
        bad.bodies[3].white_scale_source = Some(src);
        assert!(refused(&bad));
        let mut ok = r.clone();
        let mut src = presets_source();
        let res = src.residual.as_mut().unwrap();
        res.matrix_scale = None;
        res.matrix_mean_preset_angle_deg = None;
        res.matrix_daylight_k = None;
        ok.bodies[3].white_scale_source = Some(src);
        ok.validate().unwrap();
        // A sibling takes a scale and says so.
        let mut ok = r.clone();
        ok.bodies[3].white_scale_source = Some(source(WhiteScaleKind::Sibling));
        ok.validate().unwrap();
        // Unusable numbers.
        for scale in [
            [0.0, 1.0, 1.0],
            [-0.9, 1.0, 1.0],
            [f64::NAN, 1.0, 1.0],
            [1.0, 1.0, f64::INFINITY],
            [0.9, 1.01, 1.0],
            [5.0, 1.0, 1.0],
            [1.0, 1.0, 0.2],
        ] {
            let mut bad = r.clone();
            bad.bodies[3].white_scale = scale;
            assert!(refused(&bad), "{scale:?}");
        }
    }

    #[test]
    fn model_normalisation_is_forgiving_but_not_loose() {
        assert_eq!(normalize_model("  ILCE-7M3\0"), "ilce-7m3");
        assert_eq!(normalize_model("Sony ILCE-7M3"), "ilce-7m3");
        assert_eq!(normalize_model("SONYILCE"), "sonyilce");
        assert_eq!(normalize_model("é"), "é");
        assert_eq!(normalize_model(""), "");
    }
}
