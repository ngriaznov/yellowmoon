//! The `Profile` type loaded from `profile.json`: spectral curve references, sensor-map and white tables,
//! ids and versions (`docs/03-camera-profiles.md` section "Profile files").
//!
//! The types match `schema/profile.schema.json` field for field (unknown fields are refused), and
//! [`Profile::validate`] checks what a schema cannot express: white-preserving matrices, strictly
//! increasing mired in every class list, grids with the documented ranges, green exactly 1 in every
//! `w_A`, reference-light names that agree across sections. The tables themselves are the types of
//! `ym_color::spectral` and `ym_color::sensor_map`, whose reading rules are in
//! `training/sensor_map/README.md`.
//!
//! [`ProfileError`] is the error type of the whole crate.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use ym_color::consts::Mat3;
use ym_color::sensor_map::{SensorMapError, SensorMapTables};
use ym_color::spectral::{
    AlexaWhite, CctDuv, ClassProbabilities, CurveError, LightClass, Locus, SpectralCurves,
    SpectralError,
};

pub mod target;

pub use target::Target;

/// Errors from loading, validating and looking up camera data.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// A data file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A data file is not valid JSON of the documented shape.
    #[error("{path}: {source}")]
    Json {
        /// The file (or `<text>` when parsed from memory).
        path: PathBuf,
        /// The underlying error.
        source: serde_json::Error,
    },
    /// A profile identifier does not match `^[a-z0-9]+(-[a-z0-9]+)*$`.
    #[error("{0:?} is not a valid profile id (lower-case letters, digits and single hyphens)")]
    BadId(String),
    /// A profile file passed the shape checks but breaks a rule of the format.
    #[error("profile {id}: {reason}")]
    Invalid {
        /// The profile id.
        id: String,
        /// What is wrong.
        reason: String,
    },
    /// The registry is inconsistent.
    #[error("registry: {0}")]
    Registry(#[from] crate::registry::RegistryError),
    /// A forced profile id is not in the registry.
    #[error("there is no profile called {requested:?}; the available profiles are {available}")]
    UnknownProfile {
        /// What was asked for.
        requested: String,
        /// The ids that exist, comma separated.
        available: String,
    },
    /// A curve file could not be loaded.
    #[error("spectral curves: {0}")]
    Curves(#[from] CurveError),
}

/// The identifier of a camera profile, for example `sony-ilce-7rm4`.
///
/// Equals the folder name under `data/cameras/profiles/`, the registry's profile id and the id written
/// into export metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ProfileId(String);

impl ProfileId {
    /// Wraps an identifier after checking its shape.
    ///
    /// # Errors
    ///
    /// [`ProfileError::BadId`] unless `id` matches `^[a-z0-9]+(-[a-z0-9]+)*$`.
    pub fn new(id: impl Into<String>) -> Result<Self, ProfileError> {
        let id = id.into();
        let valid = !id.is_empty()
            && id.split('-').all(|part| {
                !part.is_empty()
                    && part
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            });
        if valid {
            Ok(Self(id))
        } else {
            Err(ProfileError::BadId(id))
        }
    }

    /// The identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ProfileId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for ProfileId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl FromStr for ProfileId {
    type Err = ProfileError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<String> for ProfileId {
    type Error = ProfileError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ProfileId> for String {
    fn from(id: ProfileId) -> Self {
        id.0
    }
}

/// How the sensor-map matrices were fitted (informational for the runtime).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorMapFit {
    /// Always true: every matrix maps white to white.
    pub white_preserving: bool,
    /// The fit objective, `cie-lab-after-arri-matrix`.
    pub objective: String,
    /// Number of training reflectances (190).
    pub training_reflectances: u32,
}

/// The `sensor_map` section: per-light 3x3 maps grouped by light class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorMapSection {
    /// `per-light-3x3`.
    pub method: String,
    /// The target description, relative to the data root (`alexa/target.json`).
    pub target: String,
    /// How the maps were fitted.
    pub fit: SensorMapFit,
    /// The fitted maps of each class.
    pub classes: SensorMapTables,
}

/// The `decode` section: per-body decode quirks carried as data (none in V1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeSection {
    /// Quirk names; empty for every V1 profile.
    pub quirks: Vec<String>,
}

/// Where the camera curves came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvesSource {
    /// Repository URL (rawtoaces-data).
    pub repository: String,
    /// Commit of that repository.
    pub commit: String,
    /// File inside it.
    pub file: String,
    /// SHA-256 of the curve file; equals the shipped file's hash.
    pub sha256: String,
}

/// The ALEXA target the maps were fitted against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSource {
    /// Curve file relative to the data root.
    pub file: String,
    /// SHA-256 of that file.
    pub sha256: String,
    /// Caveat shown wherever the target is described.
    pub note: String,
}

/// Names of the reference lights per class, in table order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceLights {
    /// CIE daylight lights.
    pub daylight: Vec<String>,
    /// Blackbody lights.
    pub tungsten: Vec<String>,
    /// Fluorescent lights.
    pub fluorescent: Vec<String>,
    /// LED lights.
    pub led: Vec<String>,
}

/// One same-sensor pairing check recorded for the profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceCheck {
    /// For example `ILCE-7RM4A (387) uses sony-ilce-7rm4`, or `ILCE-7RM5 (390) was considered for
    /// sony-ilce-7rm4` for a pairing that failed.
    pub pairing: String,
    /// What is compared.
    pub check: String,
    /// The outcome: `pass: max |diff| 0.0000 per coefficient (limit 0.02); <tool and version>` or
    /// `fail: ...` (the pairing does not ship).
    pub result: String,
}

/// The `provenance` section: where every number came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// Who measured the curves.
    pub curves: String,
    /// The curve file's origin.
    pub curves_source: CurvesSource,
    /// The target the maps were fitted against.
    pub target: TargetSource,
    /// `training/sensor_map/fit.py@<git sha>`.
    pub fit_script: String,
    /// SHA-256 over the fit sources.
    pub fit_sources_sha256: String,
    /// The objective and the decision that chose it.
    pub sensor_map_objective: String,
    /// Solver, options and library versions.
    pub fit_solver: String,
    /// colour-science version used at fit time. The profile files keep the British key, as the
    /// package is called `colour-science`.
    #[serde(rename = "colour_science")]
    pub color_science: String,
    /// The training reflectance set.
    pub training_reflectances: String,
    /// Reference lights per class.
    pub reference_lights: ReferenceLights,
    /// How the Duv scale was measured.
    pub duv_scale_method: String,
    /// Conventions for light SPDs and CCTs.
    pub light_conventions: String,
    /// States that CIE data was used at fit time only (pending HUMAN M1-H4).
    pub cie_data: String,
    /// Same-sensor pairing checks, with their measured results.
    pub checks: Vec<ProvenanceCheck>,
}

/// A camera profile: everything the colour pipeline needs to know about one body's sensor.
///
/// Loaded from `data/cameras/profiles/<id>/profile.json` by [`ProfileStore`](crate::registry::ProfileStore)
/// or directly with [`Profile::parse`] / [`Profile::from_path`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    /// Profile id; equals the folder name and the registry's id.
    pub id: ProfileId,
    /// Semver of this profile, written into every export's XMP.
    pub version: semver::Version,
    /// Short body name shown in the app, for example `A7R IV`.
    pub display_name: String,
    /// Measured curves, relative to `data/cameras/`.
    pub spectral_curves: String,
    /// The sensor map tables.
    pub sensor_map: SensorMapSection,
    /// The camera locus and Duv scale.
    pub locus: Locus,
    /// `w_A` tables of the ALEXA estimate.
    pub alexa_white: AlexaWhite,
    /// Decode quirks.
    pub decode: DecodeSection,
    /// Where the numbers came from.
    pub provenance: Provenance,
}

fn invalid(id: &ProfileId, reason: impl Into<String>) -> ProfileError {
    ProfileError::Invalid {
        id: id.to_string(),
        reason: reason.into(),
    }
}

impl Profile {
    /// Parses and validates the text of a `profile.json`.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Json`] for malformed or mismatching JSON, [`ProfileError::Invalid`] when
    /// [`Profile::validate`] fails.
    pub fn parse(text: &str) -> Result<Self, ProfileError> {
        Self::parse_labelled(text, Path::new("<text>"))
    }

    fn parse_labelled(text: &str, path: &Path) -> Result<Self, ProfileError> {
        let profile: Profile = serde_json::from_str(text).map_err(|source| ProfileError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        profile.validate()?;
        Ok(profile)
    }

    /// Reads, parses and validates a `profile.json`.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Io`], [`ProfileError::Json`] or [`ProfileError::Invalid`].
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ProfileError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse_labelled(&text, path)
    }

    /// Checks the rules the JSON schema cannot express (the Rust twin of
    /// `training/sensor_map/validate_profiles.py`, minus the file-hash checks, which need the data folder):
    ///
    /// * the spectral tables (`locus`, `alexa_white`) and the sensor-map matrices pass their own
    ///   validation: grid ranges, green exactly 1, strictly increasing mired, rows summing to 1 within
    ///   1e-12;
    /// * `sensor_map.method` is `per-light-3x3` and the fit is white preserving;
    /// * the lights named in `provenance.reference_lights` are the sensor-map entries, in order, and the
    ///   fluorescent and LED `w_A` lists sit at the same mired as the maps;
    /// * formula lights (`blackbody-<T>K`, `daylight-<T>K`) have mired `1e6 / T`;
    /// * `decode.quirks` is empty in V1.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Invalid`] naming the first problem.
    pub fn validate(&self) -> Result<(), ProfileError> {
        let fail = |reason: String| invalid(&self.id, reason);
        self.locus
            .validate()
            .map_err(|e: SpectralError| fail(e.to_string()))?;
        self.alexa_white
            .validate()
            .map_err(|e: SpectralError| fail(e.to_string()))?;
        self.sensor_map
            .classes
            .validate()
            .map_err(|e: SensorMapError| fail(e.to_string()))?;
        if self.sensor_map.method != "per-light-3x3" {
            return Err(fail(format!(
                "sensor_map.method is {:?}, expected \"per-light-3x3\"",
                self.sensor_map.method
            )));
        }
        if !self.sensor_map.fit.white_preserving {
            return Err(fail(
                "sensor_map.fit.white_preserving must be true".to_string(),
            ));
        }
        let refs = &self.provenance.reference_lights;
        for class in LightClass::ALL {
            let entries = self.sensor_map.classes.class(class);
            let names = match class {
                LightClass::Daylight => &refs.daylight,
                LightClass::Tungsten => &refs.tungsten,
                LightClass::Fluorescent => &refs.fluorescent,
                LightClass::Led => &refs.led,
            };
            let lights: Vec<&str> = entries.iter().map(|e| e.light.as_str()).collect();
            if lights != names.iter().map(String::as_str).collect::<Vec<_>>() {
                return Err(fail(format!(
                    "sensor_map.classes.{} lights {lights:?} differ from provenance.reference_lights {names:?}",
                    class.name()
                )));
            }
            // Formula lights carry their generating temperature in the name.
            let prefix = match class {
                LightClass::Daylight => Some("daylight-"),
                LightClass::Tungsten => Some("blackbody-"),
                _ => None,
            };
            if let Some(prefix) = prefix {
                for e in entries {
                    let t: f64 = e
                        .light
                        .strip_prefix(prefix)
                        .and_then(|rest| rest.strip_suffix('K'))
                        .and_then(|rest| rest.parse().ok())
                        .ok_or_else(|| {
                            fail(format!(
                                "sensor_map.classes.{}: cannot read a temperature from {:?}",
                                class.name(),
                                e.light
                            ))
                        })?;
                    if (e.mired - 1e6 / t).abs() >= 1e-6 {
                        return Err(fail(format!(
                            "sensor_map.classes.{}.{}: mired {} is not 1e6/{t}",
                            class.name(),
                            e.light,
                            e.mired
                        )));
                    }
                }
            }
        }
        for (class, list) in [
            (LightClass::Fluorescent, &self.alexa_white.fluorescent),
            (LightClass::Led, &self.alexa_white.led),
        ] {
            let maps = self.sensor_map.classes.class(class);
            let same = list.len() == maps.len()
                && list
                    .iter()
                    .zip(maps)
                    .all(|(w, m)| w.light == m.light && w.mired == m.mired);
            if !same {
                return Err(fail(format!(
                    "alexa_white.{} lights or mired differ from sensor_map.classes.{0}",
                    class.name()
                )));
            }
        }
        if !self.decode.quirks.is_empty() {
            return Err(fail("decode.quirks must be empty in V1".to_string()));
        }
        Ok(())
    }

    /// The sensor map for a scene light: `M_map = sum_k p_k M_k(mired)`, white preserving
    /// (`ym_color::sensor_map`).
    #[must_use]
    pub fn sensor_map(&self, probs: &ClassProbabilities, mired: f64) -> Mat3 {
        self.sensor_map.classes.blend(probs, mired)
    }

    /// `w_A(E_scene)` of stage C3 for a scene light at `cct_k`, blended by class probability
    /// (`ym_color::spectral::AlexaWhite::scene_white`). Green is exactly 1.
    #[must_use]
    pub fn scene_white(&self, cct_k: f64, probs: &ClassProbabilities) -> [f64; 3] {
        self.alexa_white.scene_white(cct_k, probs)
    }

    /// `w_A(E_user)` of stage C4 for the user's CCT: blackbody below 4000 K, CIE daylight from 4000 K.
    #[must_use]
    pub fn user_white(&self, cct_k: f64) -> [f64; 3] {
        self.alexa_white.locus_white(cct_k)
    }

    /// Camera RGB of a light (any scale) to CCT and Duv (positive towards magenta).
    ///
    /// # Errors
    ///
    /// [`SpectralError::InvalidRgb`] when a channel is not a positive finite number.
    pub fn rgb_to_cct_duv(&self, rgb: [f64; 3]) -> Result<CctDuv, SpectralError> {
        self.locus.rgb_to_cct_duv(rgb)
    }

    /// CCT and Duv (positive towards magenta) to camera RGB with green = 1.
    #[must_use]
    pub fn cct_duv_to_rgb(&self, cct_k: f64, duv: f64) -> [f64; 3] {
        self.locus.cct_duv_to_rgb(cct_k, duv)
    }

    /// Path of the measured curves under the data root (`<data_root>/cameras/<spectral_curves>`).
    #[must_use]
    pub fn curves_path(&self, data_root: &Path) -> PathBuf {
        data_root.join("cameras").join(&self.spectral_curves)
    }

    /// Loads the measured curves (for tests and tools; the runtime uses the precomputed tables).
    ///
    /// # Errors
    ///
    /// [`ProfileError::Curves`] when the file is missing or malformed.
    pub fn load_curves(&self, data_root: &Path) -> Result<SpectralCurves, ProfileError> {
        Ok(SpectralCurves::from_path(self.curves_path(data_root))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_ids_follow_the_schema_pattern() {
        for ok in ["sony-ilce-7rm4", "a", "a1", "sony-dsc-rx100m4", "x-1-2"] {
            assert_eq!(ProfileId::new(ok).unwrap().as_str(), ok);
        }
        for bad in [
            "",
            "Sony-ilce",
            "sony--ilce",
            "-sony",
            "sony-",
            "sony ilce",
            "sony_ilce",
            "sony-ilce-7rm4/../x",
            "é",
        ] {
            assert!(
                matches!(ProfileId::new(bad), Err(ProfileError::BadId(_))),
                "{bad:?}"
            );
        }
        let id: ProfileId = "sony-ilce-9".parse().unwrap();
        assert_eq!(id.to_string(), "sony-ilce-9");
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"sony-ilce-9\"");
        assert!(serde_json::from_str::<ProfileId>("\"Bad Id\"").is_err());
    }

    #[test]
    fn malformed_profiles_are_reported_with_context() {
        let err = Profile::parse("{}").unwrap_err();
        assert!(matches!(err, ProfileError::Json { .. }), "{err}");
        let err = Profile::from_path("/nonexistent/profile.json").unwrap_err();
        assert!(matches!(err, ProfileError::Io { .. }), "{err}");
    }
}
