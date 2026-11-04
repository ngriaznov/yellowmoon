//! Film and paper profiles, and the list of stock names.
//!
//! A profile is one JSON file of `data/film/spektrafilm/profiles/` (Andrea Volpato's spektrafilm profiles,
//! CC BY-SA 4.0, read unmodified). The format is described in `research/film-spektrafilm-2026-11-01.md` section 4:
//! all numbers are JSON, `null` stands for NaN, the wavelength grid is 380 to 780 nm in 5 nm steps and the common
//! exposure axis is log10 relative exposure from -3 to 4 in 256 steps. The densitometric data come from Kodak
//! and Fujifilm data sheets (the profile's `metadata.datasource` says so); this module only reads and checks them.
//!
//! Reading rules: a NaN in `log_sensitivity` is a sensitivity of 0; a NaN in `channel_density` or `base_density` takes
//! the nearest defined value (undefined outside the measured range; F-06b, `docs/24` section 3.7a).

use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::FilmError;
use crate::math::pow10;
use crate::{LAMBDA_STEP_NM, N_LAMBDA, Spectrum, Vec3, lambda_nm};

/// Number of points of the exposure axis.
pub const N_CURVE: usize = 256;

#[derive(Deserialize)]
struct RawProfile {
    info: RawInfo,
    data: RawData,
}

#[derive(Deserialize)]
struct RawInfo {
    stock: String,
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    target_print: Option<String>,
    reference_illuminant: String,
    viewing_illuminant: String,
}

#[derive(Deserialize)]
struct RawData {
    wavelengths: Vec<Option<f64>>,
    log_sensitivity: Vec<Vec<Option<f64>>>,
    hanatos2025_adaptation_window_params: Vec<Option<f64>>,
    channel_density: Vec<Vec<Option<f64>>>,
    base_density: Vec<Option<f64>>,
    midscale_neutral_density: Vec<Option<f64>>,
    log_exposure: Vec<Option<f64>>,
    density_curves: Vec<Vec<Option<f64>>>,
}

/// What a profile is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileType {
    /// A negative (or a print material, which also responds negatively): density rises with exposure.
    Negative,
    /// A positive (slide film): density falls with exposure.
    Positive,
}

/// One validated profile.
#[derive(Debug, Clone)]
pub struct Profile {
    /// The id (the file name without `.json`).
    pub id: String,
    /// The marketing name from the file's `info.name`.
    pub name: String,
    /// Negative or positive.
    pub kind: ProfileType,
    /// The usual paper (`info.target_print`), for negatives.
    pub target_print: Option<String>,
    /// The light the film is balanced for: `D55`, `T` (tungsten) or `TH-KG3` (the enlarger lamp, for papers).
    pub reference_illuminant: String,
    /// The light the print is viewed in: `D50`, or `K75P` for the cine print films.
    pub viewing_illuminant: String,
    /// `log10` spectral sensitivity per sample and layer (red, green, blue); NaN where undefined.
    pub log_sensitivity: Vec<Vec3>,
    /// The four numbers of the adaptation window (blue centre, blue width, red centre, red width in nm), or none.
    pub window_params: Vec<f64>,
    /// Dye density per unit of density (cyan, magenta, yellow) per sample; undefined entries hold the nearest defined value.
    pub channel_density: Vec<Vec3>,
    /// The support, fog and mask density per sample; undefined entries hold the nearest defined value.
    pub base_density: Spectrum,
    /// The density of a neutral at mid exposure per sample; NaN where undefined.
    pub midscale_neutral_density: Spectrum,
    /// The exposure axis, 256 points of log10 relative exposure from -3 to 4.
    pub log_exposure: Vec<f64>,
    /// Dye density against the axis, per point and layer.
    pub density_curves: Vec<Vec3>,
}

impl Profile {
    /// Reads and validates `<dir>/<id>.json`.
    ///
    /// # Errors
    ///
    /// [`FilmError::Io`] when the file cannot be read, [`FilmError::Data`] (naming the file and the problem)
    /// when it does not parse or a grid, shape or range is wrong.
    pub fn load(dir: &Path, id: &str) -> Result<Self, FilmError> {
        Self::load_file(&dir.join(format!("{id}.json")))
    }

    /// Reads and validates one profile file.
    ///
    /// # Errors
    ///
    /// As [`Profile::load`].
    pub fn load_file(path: &Path) -> Result<Self, FilmError> {
        let bytes = std::fs::read(path).map_err(|source| FilmError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let file = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("profile")
            .to_string();
        Self::parse(&file, &bytes)
    }

    fn parse(file: &str, bytes: &[u8]) -> Result<Self, FilmError> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| FilmError::data(file, format!("it is not valid JSON ({e})")))?;
        let raw: RawProfile = serde_json::from_value(value).map_err(|e| {
            FilmError::data(
                file,
                format!("a field is missing or has the wrong type ({e})"),
            )
        })?;
        let id = raw.info.stock.clone();
        let d = raw.data;
        let bad = |what: String| FilmError::data(file, what);

        if d.wavelengths.len() != N_LAMBDA {
            return Err(bad(format!(
                "the wavelength grid has {} samples, expected {N_LAMBDA}",
                d.wavelengths.len()
            )));
        }
        for (i, w) in d.wavelengths.iter().enumerate() {
            match w {
                Some(w) if (w - lambda_nm(i)).abs() < 1e-6 => {}
                _ => {
                    return Err(bad(format!(
                        "wavelength sample {i} is not {} nm (the grid must be 380 to 780 nm in {LAMBDA_STEP_NM} nm steps)",
                        lambda_nm(i)
                    )));
                }
            }
        }
        let log_sensitivity = rows3(&d.log_sensitivity, N_LAMBDA, "log_sensitivity", file)?;
        range_check(&log_sensitivity, -15.0, 4.0, "log_sensitivity", file)?;
        let channel_density = rows3(&d.channel_density, N_LAMBDA, "channel_density", file)?;
        range_check(&channel_density, -0.5, 4.0, "channel_density", file)?;
        // A dye density the profile leaves undefined (a paper's dyes before 410 nm and after 695 nm) holds its nearest
        // defined value, as the base density does. Zero let a deep shadow leak red and blue through the dyes' ends
        // (Portra Endura at density 2.5: red 1.9 times and blue 4 times the oracle's); holding the edge brings the
        // scan of the oracle's own print densities from a 95th percentile of 4.4 to 0.96 dE00 (F-06b).
        let mut channel_density = channel_density;
        for c in 0..3 {
            let mut column = [f64::NAN; N_LAMBDA];
            for (i, row) in channel_density.iter().enumerate() {
                column[i] = row[c];
            }
            if fill_nearest(&mut column).is_none() {
                column = [0.0; N_LAMBDA];
            }
            for (i, row) in channel_density.iter_mut().enumerate() {
                row[c] = column[i];
            }
        }
        let channel_density: Vec<Vec3> = channel_density;

        if d.base_density.len() != N_LAMBDA {
            return Err(bad(format!(
                "base_density has {} samples, expected {N_LAMBDA}",
                d.base_density.len()
            )));
        }
        let mut base = [f64::NAN; N_LAMBDA];
        for (i, v) in d.base_density.iter().enumerate() {
            if let Some(v) = v {
                if !(-0.2..=4.0).contains(v) {
                    return Err(bad(format!("base_density[{i}] = {v} is outside -0.2 to 4")));
                }
                base[i] = *v;
            }
        }
        if d.midscale_neutral_density.len() != N_LAMBDA {
            return Err(bad(format!(
                "midscale_neutral_density has {} samples, expected {N_LAMBDA}",
                d.midscale_neutral_density.len()
            )));
        }
        let mut mid = [f64::NAN; N_LAMBDA];
        for (m, v) in mid.iter_mut().zip(&d.midscale_neutral_density) {
            if let Some(v) = v {
                *m = *v;
            }
        }
        fill_nearest(&mut base).ok_or_else(|| bad("base_density has no defined value".into()))?;

        if d.log_exposure.len() != N_CURVE {
            return Err(bad(format!(
                "log_exposure has {} points, expected {N_CURVE}",
                d.log_exposure.len()
            )));
        }
        let mut log_exposure = Vec::with_capacity(N_CURVE);
        for (i, v) in d.log_exposure.iter().enumerate() {
            let v = v.ok_or_else(|| bad(format!("log_exposure[{i}] is not a number")))?;
            let want = -3.0 + 7.0 * i as f64 / (N_CURVE - 1) as f64;
            if (v - want).abs() > 1e-6 {
                return Err(bad(format!(
                    "log_exposure[{i}] = {v}, expected {want} (an even axis from -3 to 4)"
                )));
            }
            log_exposure.push(v);
        }
        let dc = rows3(&d.density_curves, N_CURVE, "density_curves", file)?;
        let mut density_curves = Vec::with_capacity(N_CURVE);
        for (i, r) in dc.iter().enumerate() {
            for v in r {
                if !v.is_finite() || !(-0.05..=6.0).contains(v) {
                    return Err(bad(format!(
                        "density_curves[{i}] = {v} is not a density between -0.05 and 6"
                    )));
                }
            }
            density_curves.push(*r);
        }

        let window: Vec<f64> = d
            .hanatos2025_adaptation_window_params
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        if !(window.is_empty() || window.len() == 4) {
            return Err(bad(format!(
                "the adaptation window has {} numbers, expected 0 or 4",
                window.len()
            )));
        }
        if window.iter().any(|v| !v.is_finite()) {
            return Err(bad(
                "the adaptation window holds a value that is not a number".into(),
            ));
        }
        if window.len() == 4 && (window[1] <= 0.0 || window[3] <= 0.0) {
            return Err(bad(
                "the adaptation window has a width that is not positive".into(),
            ));
        }

        let kind = match raw.info.kind.as_str() {
            "negative" => ProfileType::Negative,
            "positive" => ProfileType::Positive,
            other => {
                return Err(bad(format!(
                    "info.type is {other:?}, expected negative or positive"
                )));
            }
        };
        Ok(Self {
            id,
            name: raw.info.name,
            kind,
            target_print: raw
                .info
                .target_print
                .filter(|s| !s.is_empty() && s != "null"),
            reference_illuminant: raw.info.reference_illuminant,
            viewing_illuminant: raw.info.viewing_illuminant,
            log_sensitivity,
            window_params: window,
            channel_density,
            base_density: base,
            midscale_neutral_density: mid,
            log_exposure,
            density_curves,
        })
    }

    /// The linear spectral sensitivity `10^log_sensitivity` per sample and layer (NaN reads as 0).
    #[must_use]
    pub fn sensitivity(&self) -> Vec<Vec3> {
        self.log_sensitivity
            .iter()
            .map(|r| r.map(|v| if v.is_nan() { 0.0 } else { pow10(v) }))
            .collect()
    }

    /// The density curve of layer `c` as `(log exposure, density)` points.
    #[must_use]
    pub fn curve(&self, c: usize) -> (Vec<f64>, Vec<f64>) {
        (
            self.log_exposure.clone(),
            self.density_curves.iter().map(|r| r[c]).collect(),
        )
    }
}

fn rows3(
    rows: &[Vec<Option<f64>>],
    want: usize,
    field: &str,
    file: &str,
) -> Result<Vec<Vec3>, FilmError> {
    if rows.len() != want {
        return Err(FilmError::data(
            file,
            format!("{field} has {} rows, expected {want}", rows.len()),
        ));
    }
    let mut out = Vec::with_capacity(want);
    for (i, r) in rows.iter().enumerate() {
        if r.len() != 3 {
            return Err(FilmError::data(
                file,
                format!("{field}[{i}] has {} columns, expected 3", r.len()),
            ));
        }
        out.push([
            r[0].unwrap_or(f64::NAN),
            r[1].unwrap_or(f64::NAN),
            r[2].unwrap_or(f64::NAN),
        ]);
    }
    Ok(out)
}

fn range_check(rows: &[Vec3], lo: f64, hi: f64, field: &str, file: &str) -> Result<(), FilmError> {
    for (i, r) in rows.iter().enumerate() {
        for v in r {
            if !v.is_nan() && !(lo..=hi).contains(v) {
                return Err(FilmError::data(
                    file,
                    format!("{field}[{i}] = {v} is outside {lo} to {hi}"),
                ));
            }
        }
    }
    Ok(())
}

/// Replaces NaN entries by the nearest defined sample; `None` when none is defined.
fn fill_nearest(a: &mut Spectrum) -> Option<()> {
    let first = a.iter().position(|v| !v.is_nan())?;
    let last = a.iter().rposition(|v| !v.is_nan())?;
    for i in 0..first {
        a[i] = a[first];
    }
    for i in last + 1..a.len() {
        a[i] = a[last];
    }
    let mut prev = a[first];
    for v in a.iter_mut() {
        if v.is_nan() {
            *v = prev;
        } else {
            prev = *v;
        }
    }
    Some(())
}

/// Checks every file listed in `<root>/SHA256SUMS` (paths relative to `root`, normally `data/film/spektrafilm`).
///
/// # Errors
///
/// [`FilmError::Checksum`] naming the first file that differs, [`FilmError::Io`] when one cannot be read.
pub fn verify_checksums(root: &Path) -> Result<usize, FilmError> {
    let sums = root.join("SHA256SUMS");
    let text = std::fs::read_to_string(&sums).map_err(|source| FilmError::Io {
        path: sums.clone(),
        source,
    })?;
    let mut n = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (hash, rel) = line.split_once("  ").ok_or_else(|| {
            FilmError::data("SHA256SUMS", format!("cannot read the line {line:?}"))
        })?;
        let path = root.join(rel.trim());
        let bytes = std::fs::read(&path).map_err(|source| FilmError::Io {
            path: path.clone(),
            source,
        })?;
        let got = Sha256::digest(&bytes);
        let got: String = got.iter().map(|b| format!("{b:02x}")).collect();
        if got != hash {
            return Err(FilmError::Checksum {
                file: rel.trim().to_string(),
            });
        }
        n += 1;
    }
    Ok(n)
}

/// What a stock is, as `names.json` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StockKind {
    /// A still colour negative.
    ColourNegative,
    /// A cine colour negative.
    CineNegative,
    /// A slide film.
    Slide,
    /// A still print paper.
    Paper,
    /// A cine print film.
    PrintFilm,
}

/// One row of `names.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct StockName {
    /// The profile id.
    pub id: String,
    /// The label shown to the person.
    pub name: String,
    /// The kind.
    pub kind: StockKind,
    /// The maker.
    pub group: String,
    /// The ISO number, for films.
    #[serde(default)]
    pub iso: Option<u32>,
    /// The usual paper or print film.
    #[serde(default)]
    pub print: Option<String>,
    /// True when the stock is not offered.
    #[serde(default)]
    pub hidden: bool,
}

/// The parsed `names.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Names {
    /// Every stock.
    pub stocks: Vec<StockName>,
}

impl Names {
    /// Reads `<params>/names.json`.
    ///
    /// # Errors
    ///
    /// [`FilmError::Io`] or [`FilmError::Data`].
    pub fn load(params_dir: &Path) -> Result<Self, FilmError> {
        let path: PathBuf = params_dir.join("names.json");
        let bytes = std::fs::read(&path).map_err(|source| FilmError::Io {
            path: path.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes)
            .map_err(|e| FilmError::data("names.json", format!("it does not parse ({e})")))
    }

    /// The row for `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&StockName> {
        self.stocks.iter().find(|s| s.id == id)
    }
}
