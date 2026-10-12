//! Reader for the spectral test fixtures (`testdata/spectral/`), for tests and tools only.
//!
//! `python3 -m training.sensor_map.export_fixtures` writes the CIE 1931 observer, the reference and gate
//! light SPDs and the reflectance sets as text files into `testdata/spectral/`. They are CC BY-SA 4.0 CIE
//! data (and rawtoaces' Apache-2.0 training patches), so the folder is git-ignored and never shipped;
//! this module only reads it. Tests that need it skip with a message when it is missing (see
//! [`Fixtures::load_or_skip`]); CI runs the exporter first and sets `YM_REQUIRE_FIXTURES=1` so a missing
//! folder fails instead of skipping.
//!
//! All spectra are on the 380 to 780 nm, 5 nm grid (81 samples) of the research scripts. The text
//! formats are described in `training/sensor_map/export_fixtures.py`.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Number of wavelength samples in every fixture spectrum (380 to 780 nm at 5 nm).
pub const SAMPLES: usize = 81;

/// Errors from reading the fixtures.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    /// A file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A file does not have the documented shape.
    #[error("{path}: {reason}")]
    Format {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        reason: String,
    },
}

/// A named spectrum with its class and CCT (a light), on the fixture grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Light {
    /// Name as exported, for example `gate:FL3.1`, `ref:led:LED-B5` or `research:D65`.
    pub name: String,
    /// Light class (`daylight`, `tungsten`, `fluorescent`, `led`, or `test` for the research lights).
    pub class: String,
    /// The CCT used for the mired axis and the ARRI matrix lookup: the generating temperature for
    /// blackbody and D-series lights, the Ohno (2013) CCT for the others.
    pub cct_k: f64,
    /// Peak-normalised spectral power distribution, 81 samples.
    pub spd: Vec<f64>,
}

/// A named set of reflectance spectra.
#[derive(Debug, Clone, PartialEq)]
pub struct Reflectances {
    /// Names of the patches (`ces1`, `train17`, ...).
    pub names: Vec<String>,
    /// One 81-sample spectrum per patch.
    pub spectra: Vec<Vec<f64>>,
}

/// Everything `export_fixtures` writes that a Rust evaluation needs.
#[derive(Debug, Clone)]
pub struct Fixtures {
    /// Wavelengths in nanometres (81 values from 380 to 780).
    pub wavelengths_nm: Vec<f64>,
    /// CIE 1931 2-degree observer `[x, y, z]` per wavelength.
    pub cmf: Vec<[f64; 3]>,
    /// The research test lights (`research:*`), gate lights (`gate:*`) and reference lights
    /// (`ref:<class>:*`).
    pub lights: Vec<Light>,
    /// The 190 rawtoaces training reflectances.
    pub training: Reflectances,
    /// The 99 CIE 2017 CES reflectances.
    pub ces: Reflectances,
    /// The 24 ColorChecker (Ohta) reflectances.
    pub colorchecker: Reflectances,
    /// The 15 NIST CQS VS reflectances.
    pub nist_vs: Reflectances,
    /// `expected.json`: the numbers of the Python reference (`sony_profiles`, `sensor_map_gate`).
    pub expected: Value,
    /// `constants.json`.
    pub constants: Value,
}

/// The default fixture folder: `$YM_TESTDATA/spectral` or `<workspace root>/testdata/spectral`.
#[must_use]
pub fn default_dir() -> PathBuf {
    match std::env::var_os("YM_TESTDATA") {
        Some(root) => PathBuf::from(root).join("spectral"),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/spectral"),
    }
}

fn read(path: &Path) -> Result<String, FixtureError> {
    std::fs::read_to_string(path).map_err(|source| FixtureError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn format_error(path: &Path, reason: impl Into<String>) -> FixtureError {
    FixtureError::Format {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Splits one CSV line (comma separated, optional double quotes without embedded quotes).
fn split_csv(line: &str) -> Vec<&str> {
    line.split(',')
        .map(|f| f.trim().trim_matches('"'))
        .collect()
}

fn numbers(path: &Path, line_no: usize, fields: &[&str]) -> Result<Vec<f64>, FixtureError> {
    fields
        .iter()
        .map(|f| {
            f.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    format_error(
                        path,
                        format!("line {line_no}: {f:?} is not a finite number"),
                    )
                })
        })
        .collect()
}

/// Rows of a CSV file after the header, as (line number, fields).
fn rows(text: &str) -> Vec<(usize, Vec<String>)> {
    text.lines()
        .enumerate()
        .skip(1)
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            (
                i + 1,
                split_csv(l).into_iter().map(str::to_string).collect(),
            )
        })
        .collect()
}

fn parse_reflectances(path: &Path) -> Result<Reflectances, FixtureError> {
    let text = read(path)?;
    let mut out = Reflectances {
        names: Vec::new(),
        spectra: Vec::new(),
    };
    for (line_no, fields) in rows(&text) {
        if fields.len() != 1 + SAMPLES {
            return Err(format_error(
                path,
                format!(
                    "line {line_no}: {} fields, expected {}",
                    fields.len(),
                    1 + SAMPLES
                ),
            ));
        }
        let refs: Vec<&str> = fields[1..].iter().map(String::as_str).collect();
        out.names.push(fields[0].clone());
        out.spectra.push(numbers(path, line_no, &refs)?);
    }
    Ok(out)
}

fn parse_lights(path: &Path) -> Result<Vec<Light>, FixtureError> {
    let text = read(path)?;
    let mut out = Vec::new();
    for (line_no, fields) in rows(&text) {
        if fields.len() != 3 + SAMPLES {
            return Err(format_error(
                path,
                format!(
                    "line {line_no}: {} fields, expected {}",
                    fields.len(),
                    3 + SAMPLES
                ),
            ));
        }
        let refs: Vec<&str> = fields[2..].iter().map(String::as_str).collect();
        let nums = numbers(path, line_no, &refs)?;
        out.push(Light {
            name: fields[0].clone(),
            class: fields[1].clone(),
            cct_k: nums[0],
            spd: nums[1..].to_vec(),
        });
    }
    Ok(out)
}

fn parse_cmf(path: &Path) -> Result<(Vec<f64>, Vec<[f64; 3]>), FixtureError> {
    let text = read(path)?;
    let mut wavelengths = Vec::new();
    let mut cmf = Vec::new();
    for (line_no, fields) in rows(&text) {
        let refs: Vec<&str> = fields.iter().map(String::as_str).collect();
        let nums = numbers(path, line_no, &refs)?;
        if nums.len() != 4 {
            return Err(format_error(
                path,
                format!("line {line_no}: {} fields, expected 4", nums.len()),
            ));
        }
        wavelengths.push(nums[0]);
        cmf.push([nums[1], nums[2], nums[3]]);
    }
    if wavelengths.len() != SAMPLES {
        return Err(format_error(
            path,
            format!("{} rows, expected {SAMPLES}", wavelengths.len()),
        ));
    }
    Ok((wavelengths, cmf))
}

fn parse_json(path: &Path) -> Result<Value, FixtureError> {
    serde_json::from_str(&read(path)?).map_err(|e| format_error(path, e.to_string()))
}

impl Fixtures {
    /// Reads the fixtures from `dir` (the folder `export_fixtures` wrote).
    ///
    /// # Errors
    ///
    /// Returns a [`FixtureError`] when a file is missing or malformed.
    pub fn load(dir: &Path) -> Result<Self, FixtureError> {
        let (wavelengths_nm, cmf) = parse_cmf(&dir.join("cmf_1931_2deg.csv"))?;
        Ok(Self {
            wavelengths_nm,
            cmf,
            lights: parse_lights(&dir.join("lights.csv"))?,
            training: parse_reflectances(&dir.join("reflectances_training_190.csv"))?,
            ces: parse_reflectances(&dir.join("reflectances_ces_99.csv"))?,
            colorchecker: parse_reflectances(&dir.join("reflectances_colorchecker_24.csv"))?,
            nist_vs: parse_reflectances(&dir.join("reflectances_nist_cqs_vs_15.csv"))?,
            expected: parse_json(&dir.join("expected.json"))?,
            constants: parse_json(&dir.join("constants.json"))?,
        })
    }

    /// Loads [`default_dir`], or explains why a test should skip.
    ///
    /// Returns `None` after printing a message when the folder does not exist, unless the environment
    /// variable `YM_REQUIRE_FIXTURES` is set to anything but `0`, in which case it panics (for CI, which
    /// runs `python3 -m training.sensor_map.export_fixtures` first).
    ///
    /// # Panics
    ///
    /// When the folder exists but cannot be read, or is missing while `YM_REQUIRE_FIXTURES` is set.
    #[must_use]
    pub fn load_or_skip() -> Option<Self> {
        let dir = default_dir();
        if !dir.join("MANIFEST.json").exists() {
            let hint = "run `python3 -m training.sensor_map.export_fixtures` from the repository root (needs \
                        `tools/fetch/spectral.sh` first); CI does this before `cargo test`";
            let require = std::env::var_os("YM_REQUIRE_FIXTURES").is_some_and(|v| v != "0");
            assert!(
                !require,
                "spectral fixtures missing in {} and YM_REQUIRE_FIXTURES is set: {hint}",
                dir.display()
            );
            eprintln!(
                "SKIPPED: spectral fixtures not found in {}: {hint}",
                dir.display()
            );
            return None;
        }
        match Self::load(&dir) {
            Ok(f) => Some(f),
            Err(e) => panic!("spectral fixtures in {} are unreadable: {e}", dir.display()),
        }
    }

    /// The light with this exported name (`gate:LED-V2`, `research:D65`, `ref:daylight:daylight-6500K`).
    #[must_use]
    pub fn light(&self, name: &str) -> Option<&Light> {
        self.lights.iter().find(|l| l.name == name)
    }

    /// All lights whose name starts with `prefix` (`gate:`, `research:`, `ref:led:`), in file order.
    pub fn lights_with_prefix<'a>(
        &'a self,
        prefix: &'a str,
    ) -> impl Iterator<Item = &'a Light> + 'a {
        self.lights
            .iter()
            .filter(move |l| l.name.starts_with(prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_splitting_handles_quotes_and_spaces() {
        assert_eq!(split_csv("a,\"b c\", 3"), vec!["a", "b c", "3"]);
    }

    #[test]
    fn missing_folder_is_reported_not_panicked() {
        let err = Fixtures::load(Path::new("/nonexistent/spectral")).unwrap_err();
        assert!(matches!(err, FixtureError::Io { .. }));
    }

    #[test]
    fn malformed_rows_are_rejected() {
        let dir =
            std::env::temp_dir().join(format!("ym-color-fixture-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.csv");
        std::fs::write(&path, "name,380\nx,1,2\n").unwrap();
        assert!(matches!(
            parse_reflectances(&path),
            Err(FixtureError::Format { .. })
        ));
        std::fs::write(
            &path,
            format!(
                "name{}\nx{}\n",
                ",1".repeat(SAMPLES),
                ",nan".repeat(SAMPLES)
            ),
        )
        .unwrap();
        assert!(matches!(
            parse_reflectances(&path),
            Err(FixtureError::Format { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
