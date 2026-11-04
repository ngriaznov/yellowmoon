//! The CIE tables on the profiles' grid, and the lights built from them.
//!
//! Written from CIE 15 (the 1931 2-degree standard colorimetric observer), CIE S 014-2 (illuminants D50 and D55)
//! and Planck's law. The files in `data/film/cie/` are the CIE's own, in 1 nm steps (the observer from 360 nm,
//! D50 from 300 nm) and 5 nm steps (D55); this module reads the samples at 380, 385, ... 780 nm. The 1 nm files
//! are decimated, not averaged, so the values are the CIE's own. The files use CRLF line ends.

use std::path::Path;

use crate::error::FilmError;
use crate::math::blackbody;
use crate::{N_LAMBDA, Spectrum, Vec3, lambda_nm};

/// The observer and the lights, on the 81-sample grid.
#[derive(Debug, Clone)]
pub struct Cie {
    /// The 1931 2-degree observer, `[x_bar, y_bar, z_bar]` per sample.
    pub cmf: [Vec3; N_LAMBDA],
    /// Illuminant D50, as published (relative power, 100 at 560 nm).
    pub d50: Spectrum,
    /// Illuminant D55, as published.
    pub d55: Spectrum,
}

/// The lights a stock can be balanced or viewed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Light {
    /// Daylight, 5500 K.
    D55,
    /// The viewing light of papers, 5000 K.
    D50,
    /// Tungsten, a 3200 K black body (the reference of tungsten films).
    Tungsten,
}

impl Cie {
    /// Reads the three tables from `dir` (normally `data/film/cie`).
    ///
    /// # Errors
    ///
    /// [`FilmError::Io`] or [`FilmError::Data`] when a file is missing or damaged.
    pub fn load(dir: &Path) -> Result<Self, FilmError> {
        let xyz = read_columns(&dir.join("CIE_xyz_1931_2deg.csv"), 3)?;
        let d50 = read_columns(&dir.join("CIE_std_illum_D50.csv"), 1)?;
        let d55 = read_columns(&dir.join("CIE_illum_D55.csv"), 1)?;
        let mut cmf = [[0.0; 3]; N_LAMBDA];
        let mut s50 = [0.0; N_LAMBDA];
        let mut s55 = [0.0; N_LAMBDA];
        for i in 0..N_LAMBDA {
            let nm = lambda_nm(i).round() as i64;
            let name = |f: &str| format!("data/film/cie/{f}");
            let a = xyz.get(&nm).ok_or_else(|| {
                FilmError::data(name("CIE_xyz_1931_2deg.csv"), format!("no row for {nm} nm"))
            })?;
            cmf[i] = [a[0], a[1], a[2]];
            s50[i] = d50.get(&nm).ok_or_else(|| {
                FilmError::data(name("CIE_std_illum_D50.csv"), format!("no row for {nm} nm"))
            })?[0];
            s55[i] = d55.get(&nm).ok_or_else(|| {
                FilmError::data(name("CIE_illum_D55.csv"), format!("no row for {nm} nm"))
            })?[0];
        }
        Ok(Self {
            cmf,
            d50: s50,
            d55: s55,
        })
    }

    /// The spectrum of `light`: the CIE table or a black body, as published (not normalised).
    #[must_use]
    pub fn light(&self, light: Light) -> Spectrum {
        match light {
            Light::D55 => self.d55,
            Light::D50 => self.d50,
            Light::Tungsten => blackbody(3200.0),
        }
    }

    /// XYZ of a spectrum seen as light: `sum s(l) cmf(l)`, normalised so that `Y = 1` for `s` itself when
    /// `normalise` is true.
    #[must_use]
    pub fn xyz_of_light(&self, s: &Spectrum, normalise: bool) -> Vec3 {
        let mut xyz = [0.0; 3];
        for i in 0..N_LAMBDA {
            for c in 0..3 {
                xyz[c] += s[i] * self.cmf[i][c];
            }
        }
        if normalise && xyz[1] > 0.0 {
            let y = xyz[1];
            for v in &mut xyz {
                *v /= y;
            }
        }
        xyz
    }

    /// Chromaticity `(x, y)` of a light.
    #[must_use]
    pub fn xy_of_light(&self, s: &Spectrum) -> [f64; 2] {
        let xyz = self.xyz_of_light(s, false);
        let sum = xyz[0] + xyz[1] + xyz[2];
        [xyz[0] / sum, xyz[1] / sum]
    }
}

/// Reads `nm,v1[,v2,v3]` rows into a map from integer nm to the values.
fn read_columns(
    path: &Path,
    n: usize,
) -> Result<std::collections::BTreeMap<i64, Vec<f64>>, FilmError> {
    let text = std::fs::read_to_string(path).map_err(|source| FilmError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file = format!(
        "data/film/cie/{}",
        path.file_name().and_then(|f| f.to_str()).unwrap_or("?")
    );
    let mut map = std::collections::BTreeMap::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != n + 1 {
            return Err(FilmError::data(
                &file,
                format!(
                    "line {}: {} fields, expected {}",
                    ln + 1,
                    fields.len(),
                    n + 1
                ),
            ));
        }
        let nm: f64 = fields[0].parse().map_err(|_| {
            FilmError::data(
                &file,
                format!("line {}: {:?} is not a wavelength", ln + 1, fields[0]),
            )
        })?;
        let mut vals = Vec::with_capacity(n);
        for f in &fields[1..] {
            let v: f64 = f
                .parse()
                .ok()
                .filter(|v: &f64| v.is_finite())
                .ok_or_else(|| {
                    FilmError::data(&file, format!("line {}: {f:?} is not a number", ln + 1))
                })?;
            vals.push(v);
        }
        map.insert(nm.round() as i64, vals);
    }
    Ok(map)
}

impl Light {
    /// The light a profile names (`D55`, `T`), or `None` for any other (`TH-KG3`, `K75P`).
    #[must_use]
    pub fn from_profile_name(name: &str) -> Option<Self> {
        match name {
            "D55" => Some(Self::D55),
            "D50" => Some(Self::D50),
            "T" => Some(Self::Tungsten),
            _ => None,
        }
    }
}
