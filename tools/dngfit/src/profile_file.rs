//! `data/dng/profile.json`: the fitted DNG profile, the file `ym-export` embeds (`docs/09`, Profile).
//!
//! The profile is the same for every photo: it depends only on the display transform (ACES 2). What differs per photo
//! is `BaselineExposure`, `log2(x_white)` plus `fit_offset_ev`, which is part of the fit and therefore of this file.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use ym_export::dng::{DngProfile, LookTable};

/// The only version of the file format.
pub const SCHEMA_VERSION: u32 = 1;

/// Rounds every floating-point number in `value` to `decimals` places, so that the text of a report is short and survives
/// a round trip through `serde_json` (whose default parser is not correctly rounded in the last digit).
pub fn round_numbers(value: &mut serde_json::Value, decimals: i32) {
    use serde_json::Value;
    match value {
        Value::Number(n) if n.is_f64() => {
            let scale = 10f64.powi(decimals);
            if let Some(x) = n.as_f64()
                && x.is_finite()
                && let Some(r) = serde_json::Number::from_f64((x * scale).round() / scale)
            {
                *n = r;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|v| round_numbers(v, decimals)),
        Value::Object(o) => o.values_mut().for_each(|v| round_numbers(v, decimals)),
        _ => {}
    }
}

/// The look table as stored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookTableFile {
    /// `[hue, saturation, value]` divisions.
    pub dims: [u32; 3],
    /// `ProfileLookTableEncoding`: 0 linear, 1 sRGB.
    pub encoding: u32,
    /// Hue shift in degrees, saturation scale, value scale per entry; value-major, then hue, then saturation.
    pub data: Vec<f32>,
}

/// The fitted profile and how it was fitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileFile {
    /// Format version ([`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// `ProfileName` of the DNG.
    pub name: String,
    /// What the profile is fitted to.
    pub target: String,
    /// Added to `log2(x_white)` to give `BaselineExposure` (negative: the tone curve's input range covers scene values
    /// up to `2^-fit_offset_ev`).
    pub fit_offset_ev: f64,
    /// `ProfileToneCurve` as (input, output) pairs.
    pub tone_curve: Vec<[f32; 2]>,
    /// `ProfileLookTable`.
    pub look_table: LookTableFile,
    /// How it was fitted and what the fit measured (free-form).
    #[serde(default)]
    pub fit: serde_json::Value,
}

impl ProfileFile {
    /// The profile the DNG writer takes.
    #[must_use]
    pub fn to_profile(&self) -> DngProfile {
        DngProfile {
            tone_curve: self.tone_curve.clone(),
            look_table: LookTable {
                dims: self.look_table.dims,
                data: self.look_table.data.clone(),
                encoding: self.look_table.encoding,
            },
        }
    }

    /// A file for `profile` and `fit_offset_ev`.
    #[must_use]
    pub fn from_profile(
        profile: &DngProfile,
        fit_offset_ev: f64,
        target: &str,
        fit: serde_json::Value,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            name: ym_export::dng::PROFILE_NAME.to_owned(),
            target: target.to_owned(),
            fit_offset_ev,
            tone_curve: profile.tone_curve.clone(),
            look_table: LookTableFile {
                dims: profile.look_table.dims,
                encoding: profile.look_table.encoding,
                data: profile.look_table.data.clone(),
            },
            fit,
        }
    }

    /// Checks everything the DNG writer checks, and the offset.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION,
            "profile schema version {}, expected {SCHEMA_VERSION}",
            self.schema_version
        );
        ensure!(
            self.fit_offset_ev.is_finite() && (-12.0..=4.0).contains(&self.fit_offset_ev),
            "fit_offset_ev {} is outside -12 to 4",
            self.fit_offset_ev
        );
        self.to_profile()
            .validate()
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Reads and validates `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        file.validate()
            .with_context(|| format!("{} is not a usable profile", path.display()))?;
        Ok(file)
    }

    /// The file's text: indented JSON, with the numbers of the curve and the table on a few long lines (one hue row of
    /// the table per line), and a final newline.
    #[must_use]
    pub fn to_json_text(&self) -> String {
        fn num<T: Serialize>(v: &T) -> String {
            serde_json::to_string(v).unwrap_or_else(|_| "null".to_owned())
        }
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"schema_version\": {},\n", self.schema_version));
        out.push_str(&format!("  \"name\": {},\n", num(&self.name)));
        out.push_str(&format!("  \"target\": {},\n", num(&self.target)));
        out.push_str(&format!(
            "  \"fit_offset_ev\": {},\n",
            num(&self.fit_offset_ev)
        ));
        out.push_str("  \"tone_curve\": [\n");
        for (i, p) in self.tone_curve.iter().enumerate() {
            let comma = if i + 1 < self.tone_curve.len() {
                ","
            } else {
                ""
            };
            out.push_str(&format!("    [{}, {}]{comma}\n", num(&p[0]), num(&p[1])));
        }
        out.push_str("  ],\n  \"look_table\": {\n");
        out.push_str(&format!("    \"dims\": {},\n", num(&self.look_table.dims)));
        out.push_str(&format!(
            "    \"encoding\": {},\n",
            self.look_table.encoding
        ));
        out.push_str("    \"data\": [\n");
        let row = (self.look_table.dims[1] as usize * 3).max(3);
        let rows: Vec<&[f32]> = self.look_table.data.chunks(row).collect();
        for (i, r) in rows.iter().enumerate() {
            let comma = if i + 1 < rows.len() { "," } else { "" };
            let values: Vec<String> = r.iter().map(num).collect();
            out.push_str(&format!("      {}{comma}\n", values.join(", ")));
        }
        out.push_str("    ]\n  },\n  \"fit\": ");
        let fit = serde_json::to_string_pretty(&self.fit).unwrap_or_else(|_| "null".to_owned());
        out.push_str(&fit.replace('\n', "\n  "));
        out.push_str("\n}\n");
        out
    }

    /// Writes the file (see [`ProfileFile::to_json_text`]) after validating it.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let text = self.to_json_text();
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_identity_profile_round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dng/profile.json");
        let file = ProfileFile::from_profile(
            &DngProfile::identity(),
            -3.0,
            "test",
            serde_json::json!({"a": 1}),
        );
        file.save(&path).unwrap();
        let back = ProfileFile::load(&path).unwrap();
        assert_eq!(back, file);
        assert_eq!(back.to_profile(), DngProfile::identity());
    }

    #[test]
    fn rounded_reports_survive_a_round_trip_and_the_file_text_is_stable() {
        let mut fit = serde_json::json!({ "a": 0.918_090_308_210_158_5, "b": [1.0 / 3.0, 2, -0.000_000_4], "c": { "d": 123.456_789_012 } });
        round_numbers(&mut fit, 6);
        assert_eq!(fit["a"], serde_json::json!(0.91809));
        assert_eq!(fit["b"][1], serde_json::json!(2));
        let file = ProfileFile::from_profile(&DngProfile::identity(), -4.0, "t", fit);
        let text = file.to_json_text();
        let back: ProfileFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back.to_json_text(), text);
        assert_eq!(back, file);
    }

    #[test]
    fn a_broken_profile_is_refused_with_the_reason() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        let mut file =
            ProfileFile::from_profile(&DngProfile::identity(), 0.0, "t", serde_json::Value::Null);
        file.tone_curve = vec![[0.0, 0.0], [0.0, 1.0]];
        assert!(file.save(&path).is_err());
        file.tone_curve = vec![[0.0, 0.0], [1.0, 1.0]];
        file.fit_offset_ev = 40.0;
        assert!(file.validate().is_err());
        file.fit_offset_ev = 0.0;
        file.schema_version = 9;
        assert!(file.validate().is_err());
        std::fs::write(&path, "{ not json").unwrap();
        assert!(ProfileFile::load(&path).is_err());
    }
}
