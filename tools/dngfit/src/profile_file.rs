//! `data/dng/profile.json`: the fitted DNG profile, the file `ym-export` embeds (`docs/09`, Profile).
//!
//! The schema, the validation and the text format live in `ym-export` ([`ym_export::dng::fitted`]), which embeds the
//! committed file and writes it into every DNG: the tool that fits the profile and the product that uses it share one
//! reader and one writer of the format. This module adds the file system: [`load`] and [`save`].
//!
//! The profile is the same for every photo: it depends only on the display transform (ACES 2). What differs per photo
//! is `BaselineExposure`, `log2(x_white)` plus `fit_offset_ev`, which is part of the fit and therefore of this file.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub use ym_export::dng::fitted::{LookTableFile, ProfileFile, SCHEMA_VERSION};

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

/// Reads and validates the profile file at `path`.
pub fn load(path: &Path) -> Result<ProfileFile> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ProfileFile::from_json_str(&text)
        .with_context(|| format!("{} is not a usable profile", path.display()))
}

/// Writes `file` to `path` (see [`ProfileFile::to_json_text`]) after validating it.
pub fn save(file: &ProfileFile, path: &Path) -> Result<()> {
    file.validate()?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let text = file.to_json_text();
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_export::dng::DngProfile;

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
        save(&file, &path).unwrap();
        let back = load(&path).unwrap();
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
        assert!(save(&file, &path).is_err());
        assert!(!path.exists(), "a refused profile writes nothing");
        file.tone_curve = vec![[0.0, 0.0], [1.0, 1.0]];
        file.fit_offset_ev = 40.0;
        assert!(file.validate().is_err());
        file.fit_offset_ev = 0.0;
        file.schema_version = 9;
        assert!(file.validate().is_err());
        std::fs::write(&path, "{ not json").unwrap();
        assert!(load(&path).is_err());
    }

    /// The tool and the product read the same bytes: the file on disk is the text `ym-export` embeds.
    #[test]
    fn the_committed_file_is_the_one_ym_export_embeds() {
        let path = golden::ymrun::workspace_root().join("data/dng/profile.json");
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, ym_export::dng::fitted::EMBEDDED_PROFILE_JSON);
        let file = load(&path).unwrap();
        assert_eq!(
            file.to_fitted(),
            *ym_export::dng::FittedProfile::embedded(),
            "the embedded profile is the committed one"
        );
    }
}
