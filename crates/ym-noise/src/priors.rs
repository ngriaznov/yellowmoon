//! Priors keyed by model ID and the wide default prior, loaded from `data/cameras/noise/`
//! (`docs/06-noise.md` section Priors).
//!
//! Two kinds of prior feed the MAP fit of the estimator:
//!
//! * a **public-data prior** for bodies that have a file `<model-id>-<model>.json` in
//!   `data/cameras/noise/` (fitted by `training/noise/public_priors.py`): `log a = alpha log ISO +
//!   beta` per gain regime, sigma 0.25 in log, and `b` per regime, sigma 40%;
//! * a **class prior** for every other body: `a` proportional to ISO around a body-class median,
//!   sigma one stop; `b` log-uniform over two decades.
//!
//! The class medians are derived from physics with the stated assumptions in [`ClassAssumptions`]
//! (no measurement of ours, no numbers copied from a camera database). They are deliberately wide
//! and are replaced by data as soon as a body has a public-data prior. Gain regimes
//! (`gain-regimes.json`) switch the regime of the public-data prior at the conversion-gain ISO.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Schema version written into and expected from the data files.
pub const SCHEMA_VERSION: u32 = 1;

/// Name of the gain-regime table inside `data/cameras/noise/`.
pub const GAIN_REGIMES_FILE: &str = "gain-regimes.json";

/// ISO assumed to be the base gain of a body whose `base_iso` is unknown.
pub const DEFAULT_BASE_ISO: u32 = 100;

/// Prior sigma of `log a` for bodies with a public-data prior (`docs/06`).
pub const PUBLIC_SIGMA_LOG_A: f64 = 0.25;

/// Relative prior sigma of `b` for bodies with a public-data prior (`docs/06`).
pub const PUBLIC_SIGMA_REL_B: f64 = 0.40;

/// Prior sigma of `log a` for the class prior: one stop.
pub const CLASS_SIGMA_LOG_A: f64 = std::f64::consts::LN_2;

/// A normal distribution on the natural logarithm of a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogNormal {
    /// Mean of `ln(parameter)`.
    pub mean: f64,
    /// Standard deviation of `ln(parameter)`.
    pub sigma: f64,
}

/// Prior on the read-noise variance `b` (normalised units squared).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BPrior {
    /// No information.
    Flat,
    /// Gaussian penalty on `b`.
    Gaussian {
        /// Prior mean.
        mean: f64,
        /// Prior standard deviation.
        sigma: f64,
    },
    /// Log-uniform between `min` and `max`: no pull inside, a soft wall outside.
    LogUniform {
        /// Lower end of the range.
        min: f64,
        /// Upper end of the range.
        max: f64,
    },
}

/// Where a prior came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PriorSource {
    /// No prior (the estimator runs on the data alone).
    None,
    /// Fitted from public files for this body and gain regime (`regime` indexes the body file).
    PublicData {
        /// Sony model ID.
        model_id: u32,
        /// Index of the regime in the body file.
        regime: usize,
    },
    /// Class prior derived from physics ([`ClassAssumptions`]).
    Class(BodyClass),
}

/// The prior handed to the estimator's MAP fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoisePrior {
    /// Prior on `ln a`, if any.
    pub log_a: Option<LogNormal>,
    /// Prior on `b`.
    pub b: BPrior,
    /// Standard deviation of the Gaussian prior on each plane offset `delta_c`, normalised units.
    pub delta_sigma: f64,
    /// Where the prior came from.
    pub source: PriorSource,
    /// True when the prior was widened because the ISO looks processed (`docs/06` ISO quirks).
    pub widened: bool,
}

/// Default prior sigma of the plane offsets: about 5 DN at 14 bit (the ELD colour biases are 1 to 4 DN).
pub const DEFAULT_DELTA_SIGMA: f64 = 3.0e-4;

impl NoisePrior {
    /// No prior: the fit relies on the data alone (used to build the public-data priors).
    pub fn none() -> Self {
        Self {
            log_a: None,
            b: BPrior::Flat,
            delta_sigma: DEFAULT_DELTA_SIGMA,
            source: PriorSource::None,
            widened: false,
        }
    }

    /// Centre of the prior on `a`, if there is one.
    pub fn a_center(&self) -> Option<f64> {
        self.log_a.map(|l| l.mean.exp())
    }

    /// Centre of the prior on `b`, if there is one (geometric mean for the log-uniform prior).
    pub fn b_center(&self) -> Option<f64> {
        match self.b {
            BPrior::Flat => None,
            BPrior::Gaussian { mean, .. } => Some(mean),
            BPrior::LogUniform { min, max } => Some((min * max).sqrt()),
        }
    }
}

/// Sensor class used for the physics-based prior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyClass {
    /// 35 mm full frame, 24 to 61 MP.
    FullFrame,
    /// APS-C.
    ApsC,
    /// 1-inch type.
    OneInch,
}

/// The assumptions behind the class prior. Nothing here is a measurement of ours.
///
/// The shot-noise gain in normalised units is `a = g / (W - B)`, with `g` the conversion gain in
/// DN per electron and `W - B` the usable DN range. At the base ISO a camera maps the full well to
/// (about) the top of the ADC range, so `a_base = 1 / FWC`, and `a` grows in proportion to ISO
/// from there. The full-well capacity is not known per body, so it is estimated as
///
/// ```text
/// FWC = areal_full_well_density * pixel_area,   pixel_area = sensor_area / megapixels
/// ```
///
/// with the round-number assumptions below. The prior is one stop wide (sigma `ln 2`), which covers
/// the spread of full-well density between pixel generations; it is a stand-in until a body has
/// a public-data prior (M2-01).
pub struct ClassAssumptions;

impl ClassAssumptions {
    /// Assumed areal full-well density in electrons per square micrometre (an order-of-magnitude
    /// figure for modern back-illuminated CMOS pixels of 3 to 9 micrometres; **an assumption**).
    pub const FULL_WELL_PER_UM2: f64 = 2000.0;
    /// Assumed lowest input-referred read noise in electrons for the `b` range (**an assumption**).
    pub const READ_NOISE_E_MIN: f64 = 1.0;
    /// Assumed highest input-referred read noise in electrons for the `b` range (**an assumption**);
    /// one decade above the minimum, so `b` spans two decades.
    pub const READ_NOISE_E_MAX: f64 = 10.0;

    /// Active sensor area in mm^2 (format definition: 36 x 24, 23.5 x 15.6, 13.2 x 8.8 mm).
    pub fn sensor_area_mm2(class: BodyClass) -> f64 {
        match class {
            BodyClass::FullFrame => 36.0 * 24.0,
            BodyClass::ApsC => 23.5 * 15.6,
            BodyClass::OneInch => 13.2 * 8.8,
        }
    }

    /// Geometric middle of the megapixel range of the class: full frame 24 to 61 MP (`docs/06`),
    /// APS-C 20 to 40 MP, 1-inch 20 MP.
    pub fn typical_megapixels(class: BodyClass) -> f64 {
        match class {
            BodyClass::FullFrame => (24.0f64 * 61.0).sqrt(),
            BodyClass::ApsC => (20.0f64 * 40.0).sqrt(),
            BodyClass::OneInch => 20.0,
        }
    }

    /// Pixel area in square micrometres at the typical megapixel count.
    pub fn pixel_area_um2(class: BodyClass) -> f64 {
        Self::sensor_area_mm2(class) * 1.0e6 / (Self::typical_megapixels(class) * 1.0e6)
    }

    /// Full-well capacity in electrons at the base ISO.
    pub fn full_well_electrons(class: BodyClass) -> f64 {
        Self::FULL_WELL_PER_UM2 * Self::pixel_area_um2(class)
    }

    /// Class median of `a` at ISO 100: `1 / FWC`.
    pub fn a_at_iso100(class: BodyClass) -> f64 {
        1.0 / Self::full_well_electrons(class)
    }
}

impl BodyClass {
    /// Best guess of the class from a Sony model string such as `ILCE-6400` or `DSC-RX100M4`.
    /// Returns `None` for strings it does not recognise.
    ///
    /// Covers the current E-mount bodies and the older Sony and Minolta-mount bodies that public
    /// sample sets contain (NEX, ILCE-3000/5000, SLT, ILCA and DSLR-A; the few full-frame ones
    /// among them are listed by name). Strings are matched on the model as the file stores it.
    pub fn from_model(model: &str) -> Option<Self> {
        let m = model.trim().to_ascii_uppercase();
        let starts = |p: &str| m.starts_with(p);
        // Full-frame A-mount and fixed-lens bodies whose names would otherwise fall into a family
        // prefix below (DSLR-A850/A900, SLT-A99(V), ILCA-99M2, DSC-RX1 and its R variants).
        if starts("DSLR-A850")
            || starts("DSLR-A900")
            || starts("SLT-A99")
            || starts("ILCA-99")
            || (starts("DSC-RX1") && !starts("DSC-RX10"))
        {
            Some(Self::FullFrame)
        } else if starts("DSC-RX100") || starts("DSC-RX10") || starts("DSC-RX0") || starts("ZV-1") {
            Some(Self::OneInch)
        } else if starts("ILCE-6")
            || starts("ILCE-5")
            || starts("ILCE-3")
            || starts("ILCE-QX")
            || starts("NEX-")
            || starts("DSLR-A")
            || starts("SLT-A")
            || starts("ILCA-")
            || starts("DSC-R1")
            || starts("ZV-E10")
            || m == "ILME-FX30"
        {
            Some(Self::ApsC)
        } else if starts("ILCE-7")
            || starts("ILCE-9")
            || starts("ILCE-1")
            || starts("ILX-")
            || m == "ILME-FX3"
            || m == "ZV-E1"
        {
            Some(Self::FullFrame)
        } else {
            None
        }
    }
}

/// The wide class prior at the gain ISO `gain_iso` (`docs/06` Priors, second row).
///
/// `a` is `a_base * gain_iso / 100` with sigma one stop; `b` is log-uniform between
/// `(a * READ_NOISE_E_MIN)^2` and `(a * READ_NOISE_E_MAX)^2` (two decades). `widened` multiplies the
/// width of both by 3.
pub fn class_prior(class: BodyClass, gain_iso: f64, widened: bool) -> NoisePrior {
    let a = ClassAssumptions::a_at_iso100(class) * gain_iso / 100.0;
    let k = if widened { 3.0 } else { 1.0 };
    let lo = (a * ClassAssumptions::READ_NOISE_E_MIN).powi(2);
    let hi = (a * ClassAssumptions::READ_NOISE_E_MAX).powi(2);
    let centre = (lo * hi).sqrt();
    let half = (hi / lo).sqrt().powf(k);
    NoisePrior {
        log_a: Some(LogNormal {
            mean: a.ln(),
            sigma: CLASS_SIGMA_LOG_A * k,
        }),
        b: BPrior::LogUniform {
            min: centre / half,
            max: centre * half,
        },
        delta_sigma: DEFAULT_DELTA_SIGMA,
        source: PriorSource::Class(class),
        widened,
    }
}

/// Maps an extended low ISO to the base gain (`docs/06` ISO quirks): ISOs below `base_iso` use the gain of `base_iso`.
pub fn gain_iso(iso: u32, base_iso: u32) -> u32 {
    iso.max(base_iso)
}

/// What the estimator needs to know about the file to pick a prior.
#[derive(Clone, Debug, Default)]
pub struct CameraContext {
    /// Sony model ID (maker note `0xB001`), if known.
    pub model_id: Option<u32>,
    /// Model string (IFD0 `Model`), if known.
    pub model: Option<String>,
    /// ISO from the maker note (not the saturating EXIF `ISOSpeedRatings`), if known.
    pub iso: Option<u32>,
    /// Sensor class, if the caller knows it; otherwise guessed from the model string.
    pub class: Option<BodyClass>,
}

/// One row of `gain-regimes.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyRegimes {
    /// Sony model ID.
    pub model_id: u32,
    /// Model string, such as `ILCE-1`.
    pub model: String,
    /// Display name, such as `Sony A1`.
    pub body: String,
    /// ISO at which the conversion gain switches (the high-gain regime starts here); `None` = no break.
    pub switch_iso: Option<u32>,
    /// Base ISO of the body when it is not 100.
    pub base_iso: Option<u32>,
    /// ISOs above this look processed (`docs/06` ISO quirks); `None` = not known.
    pub processed_hi_iso_above: Option<u32>,
    /// Where the row comes from.
    pub source: String,
}

/// The table in `gain-regimes.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GainRegimes {
    /// Schema version of the file.
    pub schema_version: u32,
    /// Free-text description of the table and its status.
    pub description: String,
    /// One row per body.
    pub bodies: Vec<BodyRegimes>,
}

impl GainRegimes {
    /// The row for a model ID.
    pub fn body(&self, model_id: u32) -> Option<&BodyRegimes> {
        self.bodies.iter().find(|b| b.model_id == model_id)
    }

    /// Index of the gain regime for `gain_iso`: 1 at or above the switch ISO, otherwise 0.
    pub fn regime_index(&self, model_id: u32, gain_iso: u32) -> usize {
        match self.body(model_id).and_then(|b| b.switch_iso) {
            Some(switch) if gain_iso >= switch => 1,
            _ => 0,
        }
    }
}

/// A fitted `log a = alpha ln ISO + beta` for one gain regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AFit {
    /// Slope of `ln a` against `ln ISO`.
    pub alpha: f64,
    /// Intercept of `ln a`.
    pub beta: f64,
    /// Prior sigma of `ln a` to use (0.25 by default, `docs/06`).
    pub sigma_log: f64,
    /// Number of distinct ISOs behind the fit.
    pub n_isos: u32,
    /// Number of files behind the fit.
    pub n_files: u32,
    /// Root-mean-square residual of `ln a` of the fit (information only).
    pub rms_log_residual: f64,
}

/// A fitted `b = exp(beta) ISO^gamma` for one gain regime (`gamma = 0` is a constant).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BFit {
    /// Exponent of ISO, constrained to [0, 2] by the fit script.
    pub gamma: f64,
    /// Intercept of `ln b`.
    pub beta: f64,
    /// Relative prior sigma of `b` (0.40 by default, `docs/06`).
    pub sigma_rel: f64,
    /// Number of distinct ISOs behind the fit.
    pub n_isos: u32,
    /// Number of files behind the fit.
    pub n_files: u32,
    /// Root-mean-square residual of `ln b` of the fit (information only).
    pub rms_log_residual: f64,
}

/// One gain regime of a body file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegimeFit {
    /// Free-text name such as `low-gain`.
    pub name: String,
    /// First gain ISO of the regime.
    pub iso_min: u32,
    /// Last gain ISO of the regime; `None` = open-ended.
    pub iso_max: Option<u32>,
    /// Fit of `a`, present when the body has files at 5 or more ISOs in the regime.
    pub a: Option<AFit>,
    /// Fit of `b`, present when the body has files at 3 or more ISOs in the regime.
    pub b: Option<BFit>,
}

/// Contents of `<model-id>-<model>.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyPriorFile {
    /// Schema version of the file.
    pub schema_version: u32,
    /// Sony model ID.
    pub model_id: u32,
    /// Model string, such as `ILCE-1`.
    pub model: String,
    /// Display name.
    pub body: String,
    /// Regimes, in increasing ISO.
    pub regimes: Vec<RegimeFit>,
    /// Source files, tool versions and notes (free-form object, see `data/cameras/noise/README.md`).
    pub provenance: serde_json::Value,
}

impl BodyPriorFile {
    /// Index of the regime holding `gain_iso`.
    pub fn regime_for(&self, gain_iso: u32) -> Option<usize> {
        self.regimes
            .iter()
            .position(|r| gain_iso >= r.iso_min && r.iso_max.is_none_or(|m| gain_iso <= m))
    }
}

/// Why loading the noise data failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PriorError {
    /// A file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// A file is not valid JSON of the expected shape.
    #[error("{path} is not a valid noise data file: {source}")]
    Parse {
        /// File that failed.
        path: PathBuf,
        /// Underlying error.
        source: serde_json::Error,
    },
    /// A file has the wrong schema version or contradicts its own name.
    #[error("{path}: {reason}")]
    Invalid {
        /// File that failed.
        path: PathBuf,
        /// What is wrong.
        reason: String,
    },
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, PriorError> {
    let text = fs::read_to_string(path).map_err(|source| PriorError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| PriorError::Parse {
        path: path.to_owned(),
        source,
    })
}

/// Gain-regime table and per-body priors, loaded from `data/cameras/noise/`.
#[derive(Clone, Debug)]
pub struct PriorStore {
    gain: GainRegimes,
    bodies: HashMap<u32, BodyPriorFile>,
}

impl PriorStore {
    /// A store with no table and no body files: every body gets the class prior.
    pub fn empty() -> Self {
        Self {
            gain: GainRegimes {
                schema_version: SCHEMA_VERSION,
                description: String::new(),
                bodies: Vec::new(),
            },
            bodies: HashMap::new(),
        }
    }

    /// A store from a gain-regime table and body files (for tests and tools).
    pub fn from_parts(gain: GainRegimes, bodies: Vec<BodyPriorFile>) -> Self {
        Self {
            gain,
            bodies: bodies.into_iter().map(|b| (b.model_id, b)).collect(),
        }
    }

    /// Loads `gain-regimes.json` (optional) and every `<model-id>-<model>.json` body file from
    /// `dir` (normally `<data>/cameras/noise`). A missing directory gives an empty store.
    ///
    /// # Errors
    ///
    /// Returns [`PriorError`] when a file is unreadable, malformed, has an unsupported schema
    /// version, or its name disagrees with its `model_id`.
    pub fn load_dir(dir: &Path) -> Result<Self, PriorError> {
        if !dir.is_dir() {
            return Ok(Self::empty());
        }
        let mut store = Self::empty();
        let gain_path = dir.join(GAIN_REGIMES_FILE);
        if gain_path.is_file() {
            let g: GainRegimes = read_json(&gain_path)?;
            if g.schema_version != SCHEMA_VERSION {
                return Err(PriorError::Invalid {
                    path: gain_path,
                    reason: format!("unsupported schema_version {}", g.schema_version),
                });
            }
            store.gain = g;
        }
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|source| PriorError::Io {
                path: dir.to_owned(),
                source,
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        paths.sort();
        for path in paths {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if name == GAIN_REGIMES_FILE || name.ends_with(".schema.json") {
                continue;
            }
            let body: BodyPriorFile = read_json(&path)?;
            if body.schema_version != SCHEMA_VERSION {
                return Err(PriorError::Invalid {
                    path,
                    reason: format!("unsupported schema_version {}", body.schema_version),
                });
            }
            let want_prefix = format!("{}-", body.model_id);
            if !name.starts_with(&want_prefix) {
                return Err(PriorError::Invalid {
                    path,
                    reason: format!(
                        "file name should start with `{want_prefix}` (model_id {})",
                        body.model_id
                    ),
                });
            }
            store.bodies.insert(body.model_id, body);
        }
        Ok(store)
    }

    /// The gain-regime table.
    pub fn gain_regimes(&self) -> &GainRegimes {
        &self.gain
    }

    /// The body file for a model ID, if there is one.
    pub fn body(&self, model_id: u32) -> Option<&BodyPriorFile> {
        self.bodies.get(&model_id)
    }

    /// Number of bodies with a public-data prior.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// True when no body has a public-data prior.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The prior for a file (`docs/06` Priors).
    ///
    /// Without an ISO there is nothing to scale with, so the prior is empty. With one, the ISO is
    /// mapped to its gain (extended low ISOs use the base gain), the gain regime is looked up, and
    /// each of `a` and `b` comes from the body's fit when it has one for that regime, from the
    /// class prior otherwise. ISOs above the body's "processed" threshold widen the prior.
    pub fn prior_for(&self, ctx: &CameraContext) -> NoisePrior {
        let Some(iso) = ctx.iso.filter(|&i| i > 0) else {
            return NoisePrior::none();
        };
        let info = ctx.model_id.and_then(|id| self.gain.body(id));
        let base_iso = info.and_then(|b| b.base_iso).unwrap_or(DEFAULT_BASE_ISO);
        let gain = gain_iso(iso, base_iso);
        let widened = info
            .and_then(|b| b.processed_hi_iso_above)
            .is_some_and(|t| iso > t);
        let class = ctx
            .class
            .or_else(|| ctx.model.as_deref().and_then(BodyClass::from_model))
            .unwrap_or(BodyClass::FullFrame);
        let mut prior = class_prior(class, f64::from(gain), widened);
        if let Some(file) = ctx.model_id.and_then(|id| self.bodies.get(&id))
            && let Some(ri) = file.regime_for(gain)
        {
            let regime = &file.regimes[ri];
            let k = if widened { 3.0 } else { 1.0 };
            let mut used = false;
            if let Some(a) = &regime.a {
                prior.log_a = Some(LogNormal {
                    mean: a.alpha * f64::from(gain).ln() + a.beta,
                    sigma: a.sigma_log * k,
                });
                used = true;
            }
            if let Some(b) = &regime.b {
                let mean = (b.beta + b.gamma * f64::from(gain).ln()).exp();
                prior.b = BPrior::Gaussian {
                    mean,
                    sigma: mean * b.sigma_rel * k,
                };
                used = true;
            }
            if used {
                prior.source = PriorSource::PublicData {
                    model_id: file.model_id,
                    regime: ri,
                };
            }
        }
        prior
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn table() -> GainRegimes {
        GainRegimes {
            schema_version: SCHEMA_VERSION,
            description: "test".into(),
            bodies: vec![
                BodyRegimes {
                    model_id: 384,
                    model: "ILCE-1".into(),
                    body: "Sony A1".into(),
                    switch_iso: Some(500),
                    base_iso: None,
                    processed_hi_iso_above: Some(16000),
                    source: "test".into(),
                },
                BodyRegimes {
                    model_id: 392,
                    model: "ILCE-9M3".into(),
                    body: "Sony A9 III".into(),
                    switch_iso: Some(800),
                    base_iso: Some(250),
                    processed_hi_iso_above: Some(16000),
                    source: "test".into(),
                },
            ],
        }
    }

    fn body_file() -> BodyPriorFile {
        BodyPriorFile {
            schema_version: SCHEMA_VERSION,
            model_id: 384,
            model: "ILCE-1".into(),
            body: "Sony A1".into(),
            regimes: vec![
                RegimeFit {
                    name: "low-gain".into(),
                    iso_min: 100,
                    iso_max: Some(499),
                    a: Some(AFit {
                        alpha: 1.0,
                        beta: -10.5,
                        sigma_log: 0.25,
                        n_isos: 6,
                        n_files: 12,
                        rms_log_residual: 0.05,
                    }),
                    b: None,
                },
                RegimeFit {
                    name: "high-gain".into(),
                    iso_min: 500,
                    iso_max: None,
                    a: None,
                    b: Some(BFit {
                        gamma: 1.0,
                        beta: -18.0,
                        sigma_rel: 0.4,
                        n_isos: 3,
                        n_files: 5,
                        rms_log_residual: 0.1,
                    }),
                },
            ],
            provenance: serde_json::json!({"note": "test fixture"}),
        }
    }

    #[test]
    fn class_medians_follow_the_stated_assumptions() {
        // FWC = 2000 e/um2 * pixel area; a(ISO 100) = 1 / FWC.
        let ff = ClassAssumptions::a_at_iso100(BodyClass::FullFrame);
        let area = 864.0 / ((24.0f64 * 61.0).sqrt());
        assert_relative_eq!(ff, 1.0 / (2000.0 * area), max_relative = 1e-12);
        // Smaller sensors have fewer electrons per pixel, so a larger a.
        assert!(ClassAssumptions::a_at_iso100(BodyClass::ApsC) > ff);
        assert!(
            ClassAssumptions::a_at_iso100(BodyClass::OneInch)
                > ClassAssumptions::a_at_iso100(BodyClass::ApsC)
        );
        // Sanity range: tens of thousands of electrons for full frame.
        let fwc = ClassAssumptions::full_well_electrons(BodyClass::FullFrame);
        assert!((20_000.0..120_000.0).contains(&fwc), "{fwc}");
    }

    #[test]
    fn class_prior_scales_with_iso_and_b_spans_two_decades() {
        let p100 = class_prior(BodyClass::FullFrame, 100.0, false);
        let p800 = class_prior(BodyClass::FullFrame, 800.0, false);
        let a100 = p100.a_center().unwrap();
        assert_relative_eq!(p800.a_center().unwrap() / a100, 8.0, max_relative = 1e-12);
        assert_relative_eq!(p100.log_a.unwrap().sigma, std::f64::consts::LN_2);
        match p100.b {
            BPrior::LogUniform { min, max } => {
                assert_relative_eq!(max / min, 100.0, max_relative = 1e-12)
            }
            _ => panic!("class b prior must be log-uniform"),
        }
        let wide = class_prior(BodyClass::FullFrame, 100.0, true);
        assert!(wide.log_a.unwrap().sigma > 2.9 * p100.log_a.unwrap().sigma);
        assert!(wide.widened);
    }

    #[test]
    fn model_strings_map_to_classes() {
        assert_eq!(
            BodyClass::from_model("ILCE-7RM4"),
            Some(BodyClass::FullFrame)
        );
        assert_eq!(BodyClass::from_model("ILCE-1"), Some(BodyClass::FullFrame));
        assert_eq!(
            BodyClass::from_model("ILME-FX3"),
            Some(BodyClass::FullFrame)
        );
        assert_eq!(BodyClass::from_model("ZV-E1"), Some(BodyClass::FullFrame));
        assert_eq!(BodyClass::from_model("ILCE-6400"), Some(BodyClass::ApsC));
        assert_eq!(BodyClass::from_model("ZV-E10"), Some(BodyClass::ApsC));
        assert_eq!(
            BodyClass::from_model("DSC-RX100M4"),
            Some(BodyClass::OneInch)
        );
        assert_eq!(BodyClass::from_model("DSC-HX99"), None);
    }

    #[test]
    fn older_and_a_mount_bodies_get_their_real_class() {
        // Found on the local sample files: ILCE-5000, NEX-5N, NEX-7 and DSLR-A500 all fell through
        // to the full-frame default and got a prior centred 1.7 times too low in `a`.
        use BodyClass::{ApsC, FullFrame, OneInch};
        for (model, class) in [
            ("ILCE-5000", ApsC),
            ("ILCE-5100", ApsC),
            ("ILCE-3000", ApsC),
            ("ILCE-6300", ApsC),
            ("ILCE-QX1", ApsC),
            ("NEX-7", ApsC),
            ("NEX-5N", ApsC),
            ("DSLR-A500", ApsC),
            ("DSLR-A300", ApsC),
            ("SLT-A58", ApsC),
            ("SLT-A77V", ApsC),
            ("ILCA-77M2", ApsC),
            ("DSC-R1", ApsC),
            ("DSLR-A900", FullFrame),
            ("DSLR-A850", FullFrame),
            ("SLT-A99V", FullFrame),
            ("ILCA-99M2", FullFrame),
            ("DSC-RX1", FullFrame),
            ("DSC-RX1RM2", FullFrame),
            ("ILCE-7RM5", FullFrame),
            ("ILCE-1M2", FullFrame),
            ("DSC-RX10M3", OneInch),
            ("DSC-RX100M7", OneInch),
            ("DSC-RX0M2", OneInch),
            ("ZV-1", OneInch),
        ] {
            assert_eq!(BodyClass::from_model(model), Some(class), "{model}");
        }
        // Case and surrounding blanks do not matter.
        assert_eq!(BodyClass::from_model(" nex-7 "), Some(ApsC));
    }

    #[test]
    fn extended_low_iso_uses_the_base_gain() {
        assert_eq!(gain_iso(50, 100), 100);
        assert_eq!(gain_iso(64, 100), 100);
        assert_eq!(gain_iso(160, 100), 160);
        assert_eq!(gain_iso(100, 250), 250);
    }

    #[test]
    fn no_iso_means_no_prior() {
        let store = PriorStore::empty();
        assert_eq!(
            store.prior_for(&CameraContext::default()),
            NoisePrior::none()
        );
    }

    #[test]
    fn unknown_body_gets_the_class_prior_at_the_gain_iso() {
        let store = PriorStore::empty();
        let ctx = CameraContext {
            model_id: Some(9999),
            model: Some("ILCE-6400".into()),
            iso: Some(50),
            class: None,
        };
        let p = store.prior_for(&ctx);
        assert_eq!(p.source, PriorSource::Class(BodyClass::ApsC));
        assert_relative_eq!(
            p.a_center().unwrap(),
            ClassAssumptions::a_at_iso100(BodyClass::ApsC),
            max_relative = 1e-12
        );
    }

    #[test]
    fn gain_regimes_pick_the_regime_and_the_base_gain() {
        let g = table();
        assert_eq!(g.regime_index(384, 499), 0);
        assert_eq!(g.regime_index(384, 500), 1);
        assert_eq!(g.regime_index(1, 8000), 0);
        let store = PriorStore::from_parts(g, vec![]);
        // A9 III: ISO 100 is below its base 250 and uses the gain of 250.
        let p = store.prior_for(&CameraContext {
            model_id: Some(392),
            model: Some("ILCE-9M3".into()),
            iso: Some(100),
            class: None,
        });
        assert_relative_eq!(
            p.a_center().unwrap(),
            ClassAssumptions::a_at_iso100(BodyClass::FullFrame) * 2.5,
            max_relative = 1e-12
        );
    }

    #[test]
    fn public_data_prior_is_used_per_regime_and_widened_for_processed_isos() {
        let store = PriorStore::from_parts(table(), vec![body_file()]);
        let ctx = |iso| CameraContext {
            model_id: Some(384),
            model: Some("ILCE-1".into()),
            iso: Some(iso),
            class: None,
        };
        // Low-gain regime: a from the file, b from the class.
        let p = store.prior_for(&ctx(200));
        assert_eq!(
            p.source,
            PriorSource::PublicData {
                model_id: 384,
                regime: 0
            }
        );
        assert_relative_eq!(
            p.log_a.unwrap().mean,
            200f64.ln() - 10.5,
            max_relative = 1e-12
        );
        assert_relative_eq!(p.log_a.unwrap().sigma, 0.25);
        assert!(matches!(p.b, BPrior::LogUniform { .. }));
        // High-gain regime: b from the file (gamma 1), a from the class.
        let p = store.prior_for(&ctx(1000));
        match p.b {
            BPrior::Gaussian { mean, sigma } => {
                assert_relative_eq!(mean, (-18.0f64 + 1000f64.ln()).exp(), max_relative = 1e-12);
                assert_relative_eq!(sigma / mean, 0.4, max_relative = 1e-12);
            }
            other => panic!("expected a Gaussian b prior, got {other:?}"),
        }
        assert!(!p.widened);
        // Processed Hi ISO: widened.
        let p = store.prior_for(&ctx(25600));
        assert!(p.widened);
        match p.b {
            BPrior::Gaussian { mean, sigma } => {
                assert_relative_eq!(sigma / mean, 1.2, max_relative = 1e-12)
            }
            other => panic!("expected a Gaussian b prior, got {other:?}"),
        }
    }

    #[test]
    fn loader_reads_a_directory_and_checks_names() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        fs::write(
            p.join(GAIN_REGIMES_FILE),
            serde_json::to_string(&table()).unwrap(),
        )
        .unwrap();
        fs::write(
            p.join("384-ilce-1.json"),
            serde_json::to_string(&body_file()).unwrap(),
        )
        .unwrap();
        let store = PriorStore::load_dir(p).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.gain_regimes().bodies.len(), 2);
        assert!(store.body(384).is_some());
        // A file whose name disagrees with its model ID is rejected.
        fs::write(
            p.join("999-wrong.json"),
            serde_json::to_string(&body_file()).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            PriorStore::load_dir(p),
            Err(PriorError::Invalid { .. })
        ));
        // A missing directory is an empty store.
        assert!(PriorStore::load_dir(&p.join("nope")).unwrap().is_empty());
    }

    #[test]
    fn shipped_gain_regime_table_loads_and_matches_the_spec() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/cameras/noise");
        let store = PriorStore::load_dir(&dir).unwrap();
        let g = store.gain_regimes();
        let switch = |id: u32| g.body(id).and_then(|b| b.switch_iso);
        // docs/06 table: A1 500, A1 II 500, A7 III 640, A7 IV ~400, A7 V 1000, A7R IV 320,
        // A7R V 320, A7S III 1600, A7C 640, A7C II 400, A9 500, A9 II 640, A9 III 800.
        for (id, iso) in [
            (384, 500),
            (400, 500),
            (363, 640),
            (388, 400),
            (407, 1000),
            (375, 320),
            (390, 320),
            (383, 1600),
            (381, 640),
            (397, 400),
            (358, 500),
            (376, 640),
            (392, 800),
        ] {
            assert_eq!(switch(id), Some(iso), "model id {id}");
        }
        assert_eq!(g.body(392).unwrap().base_iso, Some(250));
        assert_eq!(g.bodies.len(), 13);
        for b in &g.bodies {
            assert_eq!(b.source, "docs/06 table, to be confirmed on samples");
        }
    }
}
