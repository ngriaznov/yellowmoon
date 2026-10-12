//! Spectral sensitivity curves in the rawtoaces-data JSON format (`data/cameras/sony/*.json`,
//! `data/alexa/*.json`), for tests and tools.
//!
//! The runtime never needs these: the profile carries the tables derived from them. This loader exists
//! so tests and tools can render reflectances through the measured curves and check the shipped tables
//! (`docs/02-colour-pipeline.md`, "Data formats").
//!
//! The format is
//!
//! ```json
//! { "header": { "manufacturer": "Sony", "model": "ILCE-7RM4", ... },
//!   "spectral_data": { "units": "relative", "index": { "main": ["R", "G", "B"] },
//!                      "data": { "main": { "380": [r, g, b], "385": [r, g, b], ... } } } }
//! ```
//!
//! Wavelengths are keys of the `main` object, in nanometres; they are sorted numerically on load.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Errors from loading a curve file.
#[derive(Debug, thiserror::Error)]
pub enum CurveError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The text is not valid JSON of the expected shape.
    #[error("not a rawtoaces curve file: {0}")]
    Json(#[from] serde_json::Error),
    /// `spectral_data.index.main` is not `["R", "G", "B"]`.
    #[error("the channel order is {0:?}, expected [\"R\", \"G\", \"B\"]")]
    ChannelOrder(Vec<String>),
    /// A wavelength key is not a number.
    #[error("wavelength key {0:?} is not a number")]
    BadWavelength(String),
    /// A sample is not finite.
    #[error("the sample at {wavelength_nm} nm is not finite")]
    NonFinite {
        /// Wavelength of the offending sample.
        wavelength_nm: f64,
    },
    /// The file has fewer than two samples.
    #[error("the file holds {0} samples, at least 2 are needed")]
    TooFewSamples(usize),
    /// A channel is zero everywhere, so it cannot be peak-normalised.
    #[error("channel {0} has no positive sample")]
    EmptyChannel(char),
    /// A spectrum passed to [`SpectralCurves::response`] has the wrong length.
    #[error(
        "a spectrum with {found} samples cannot be integrated against {expected} curve samples"
    )]
    LengthMismatch {
        /// Number of curve samples.
        expected: usize,
        /// Length of the spectrum.
        found: usize,
    },
}

#[derive(Deserialize)]
struct Header {
    manufacturer: String,
    model: String,
}

#[derive(Deserialize)]
struct SpectralIndex {
    main: Vec<String>,
}

#[derive(Deserialize)]
struct SpectralData {
    index: SpectralIndex,
    data: BTreeMap<String, BTreeMap<String, [f64; 3]>>,
}

#[derive(Deserialize)]
struct CurveFile {
    header: Header,
    spectral_data: SpectralData,
}

/// Camera (or ALEXA) spectral sensitivities: R, G, B per wavelength.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralCurves {
    manufacturer: String,
    model: String,
    wavelengths_nm: Vec<f64>,
    samples: Vec<[f64; 3]>,
}

impl SpectralCurves {
    /// Parses the text of a curve file.
    ///
    /// # Errors
    ///
    /// Returns a [`CurveError`] for malformed JSON, a channel order other than R, G, B, a non-numeric
    /// wavelength, a non-finite sample or fewer than two samples.
    pub fn parse(text: &str) -> Result<Self, CurveError> {
        let file: CurveFile = serde_json::from_str(text)?;
        let order = &file.spectral_data.index.main;
        if order.iter().map(String::as_str).ne(["R", "G", "B"]) {
            return Err(CurveError::ChannelOrder(order.clone()));
        }
        let main = file.spectral_data.data.get("main").ok_or_else(|| {
            CurveError::Json(serde::de::Error::custom(
                "spectral_data.data has no \"main\" entry",
            ))
        })?;
        let mut rows = Vec::with_capacity(main.len());
        for (key, rgb) in main {
            let wavelength_nm = key
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|w| w.is_finite())
                .ok_or_else(|| CurveError::BadWavelength(key.clone()))?;
            if !rgb.iter().all(|v| v.is_finite()) {
                return Err(CurveError::NonFinite { wavelength_nm });
            }
            rows.push((wavelength_nm, *rgb));
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        if rows.len() < 2 {
            return Err(CurveError::TooFewSamples(rows.len()));
        }
        Ok(Self {
            manufacturer: file.header.manufacturer,
            model: file.header.model,
            wavelengths_nm: rows.iter().map(|r| r.0).collect(),
            samples: rows.iter().map(|r| r.1).collect(),
        })
    }

    /// Reads and parses a curve file.
    ///
    /// # Errors
    ///
    /// Returns a [`CurveError`] when the file cannot be read or parsed.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, CurveError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| CurveError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
    }

    /// The `header.manufacturer` field (`Sony`, `ARRI`).
    #[must_use]
    pub fn manufacturer(&self) -> &str {
        &self.manufacturer
    }

    /// The `header.model` field (`ILCE-7RM4`).
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The sample wavelengths in nanometres, ascending.
    #[must_use]
    pub fn wavelengths_nm(&self) -> &[f64] {
        &self.wavelengths_nm
    }

    /// The `[r, g, b]` sensitivity at each wavelength.
    #[must_use]
    pub fn samples(&self) -> &[[f64; 3]] {
        &self.samples
    }

    /// Number of wavelength samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Always `false`: a parsed file has at least two samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The curves with negative samples set to 0 (the Sony files carry a little measurement noise below
    /// zero; the research scripts clip it before normalising).
    #[must_use]
    pub fn clipped_at_zero(&self) -> Self {
        let mut out = self.clone();
        for s in &mut out.samples {
            for v in s {
                *v = v.max(0.0);
            }
        }
        out
    }

    /// The curves with each channel divided by its own peak, so every channel peaks at 1.
    ///
    /// # Errors
    ///
    /// [`CurveError::EmptyChannel`] when a channel has no positive sample.
    pub fn peak_normalized(&self) -> Result<Self, CurveError> {
        let mut peak = [0.0_f64; 3];
        for s in &self.samples {
            for (p, v) in peak.iter_mut().zip(s) {
                *p = p.max(*v);
            }
        }
        for (p, name) in peak.iter().zip(['R', 'G', 'B']) {
            if *p <= 0.0 {
                return Err(CurveError::EmptyChannel(name));
            }
        }
        let mut out = self.clone();
        for s in &mut out.samples {
            for (v, p) in s.iter_mut().zip(peak) {
                *v /= p;
            }
        }
        Ok(out)
    }

    /// Camera response to a spectrum sampled at the curves' wavelengths: `sum_i spectrum[i] * S[i]` per
    /// channel (5 nm sums as in the research scripts; no wavelength weighting).
    ///
    /// # Errors
    ///
    /// [`CurveError::LengthMismatch`] when `spectrum` does not have one value per curve sample.
    pub fn response(&self, spectrum: &[f64]) -> Result<[f64; 3], CurveError> {
        if spectrum.len() != self.samples.len() {
            return Err(CurveError::LengthMismatch {
                expected: self.samples.len(),
                found: spectrum.len(),
            });
        }
        let mut out = [0.0; 3];
        for (e, s) in spectrum.iter().zip(&self.samples) {
            for (acc, v) in out.iter_mut().zip(s) {
                *acc += e * v;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "header": {"manufacturer": "Test", "model": "T1", "measurement_equipment": null},
        "spectral_data": {
            "units": "relative",
            "index": {"main": ["R", "G", "B"]},
            "data": {"main": {"400": [0.0, 0.5, -0.1], "390": [0.1, 0.25, 0.0], "410": [0.4, 1.0, 0.2]}}
        }
    }"#;

    #[test]
    fn parses_and_sorts_wavelengths() {
        let c = SpectralCurves::parse(SAMPLE).unwrap();
        assert_eq!(c.manufacturer(), "Test");
        assert_eq!(c.model(), "T1");
        assert_eq!(c.wavelengths_nm(), &[390.0, 400.0, 410.0]);
        assert_eq!(c.samples()[0], [0.1, 0.25, 0.0]);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn clip_and_normalise() {
        let c = SpectralCurves::parse(SAMPLE)
            .unwrap()
            .clipped_at_zero()
            .peak_normalized()
            .unwrap();
        assert_eq!(c.samples()[1], [0.0, 0.5, 0.0]);
        assert_eq!(c.samples()[2], [1.0, 1.0, 1.0]);
        assert_eq!(c.samples()[0][0], 0.25);
    }

    #[test]
    fn response_is_a_dot_product() {
        let c = SpectralCurves::parse(SAMPLE).unwrap();
        let r = c.response(&[1.0, 2.0, 3.0]).unwrap();
        assert!((r[0] - (0.1 + 0.0 + 1.2)).abs() < 1e-15);
        assert!((r[1] - (0.25 + 1.0 + 3.0)).abs() < 1e-15);
        assert!(matches!(
            c.response(&[1.0]),
            Err(CurveError::LengthMismatch {
                expected: 3,
                found: 1
            })
        ));
    }

    #[test]
    fn rejects_malformed_files() {
        assert!(matches!(
            SpectralCurves::parse("{}"),
            Err(CurveError::Json(_))
        ));
        let swapped = SAMPLE.replace(r#"["R", "G", "B"]"#, r#"["B", "G", "R"]"#);
        assert!(matches!(
            SpectralCurves::parse(&swapped),
            Err(CurveError::ChannelOrder(_))
        ));
        let bad_key = SAMPLE.replace(r#""400""#, r#""four hundred""#);
        assert!(matches!(
            SpectralCurves::parse(&bad_key),
            Err(CurveError::BadWavelength(_))
        ));
        let single = r#"{"header": {"manufacturer": "T", "model": "M"}, "spectral_data": {"index": {"main": ["R","G","B"]}, "data": {"main": {"400": [1, 1, 1]}}}}"#;
        assert!(matches!(
            SpectralCurves::parse(single),
            Err(CurveError::TooFewSamples(1))
        ));
        let zero = SpectralCurves::parse(
            &SAMPLE
                .replace("0.5, -0.1", "0.0, -0.1")
                .replace("0.25, 0.0", "0.0, 0.0")
                .replace("1.0, 0.2", "0.0, 0.2"),
        )
        .unwrap();
        assert!(matches!(
            zero.peak_normalized(),
            Err(CurveError::EmptyChannel('G'))
        ));
        assert!(matches!(
            SpectralCurves::from_path("/nonexistent/curves.json"),
            Err(CurveError::Io { .. })
        ));
    }
}
