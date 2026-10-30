//! The fitted DNG profile: `data/dng/profile.json`, embedded in the binary (`docs/09-export.md`, Profile).
//!
//! `tools/dngfit` fits the tone curve, the look table and the `BaselineExposure` offset to the cine rendering (C12)
//! with Adobe's reference renderer and writes the result as `data/dng/profile.json`. This module is the schema of that
//! file (shared by the fitting tool and the product, so there is one reader and one writer of the format) and the copy
//! of it that `ym` embeds with `include_str!`: [`FittedProfile::embedded`].
//!
//! The profile is the same for every photo: it depends only on the display transform (the cine rendering). What differs
//! per photo is `BaselineExposure`, `log2(x_white)` plus [`FittedProfile::fit_offset_ev`] (there is no display placement:
//! the app shows the scene through the rendering as it is), and the writer floors `x_white` so that it is never negative
//! ([`BaselineExposure::white_floor`]).

use std::fmt;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::profile::{BaselineExposure, DngProfile, FIT_OFFSET_RANGE_EV, LookTable};

/// The only version of the file format.
pub const SCHEMA_VERSION: u32 = 1;

/// The text of `data/dng/profile.json` as committed, compiled into the binary (no file is read at run time, and the
/// bundle needs no resource entry for it).
pub const EMBEDDED_PROFILE_JSON: &str = include_str!("../../../../data/dng/profile.json");

/// Why a profile file could not be used.
#[derive(Debug, thiserror::Error)]
pub enum ProfileFileError {
    /// The text is not JSON of the profile file's shape.
    #[error("The DNG profile can't be read: {0}.")]
    Parse(#[from] serde_json::Error),
    /// The file parsed, but the profile in it is not one the DNG writer accepts.
    #[error("The DNG profile can't be used: {0}.")]
    Invalid(String),
}

/// The look table as stored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookTableFile {
    /// `[hue, saturation, value]` divisions.
    pub dims: [u32; 3],
    /// `ProfileLookTableEncoding`: 0 linear, 1 sRGB.
    pub encoding: u32,
    /// Hue shift in degrees, saturation scale and value scale per entry; value-major, then hue, then saturation.
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

    /// The profile and offset in the form the writer's callers hold.
    #[must_use]
    pub fn to_fitted(&self) -> FittedProfile {
        FittedProfile {
            profile: self.to_profile(),
            fit_offset_ev: self.fit_offset_ev,
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
            name: super::PROFILE_NAME.to_owned(),
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
    ///
    /// # Errors
    ///
    /// [`ProfileFileError::Invalid`] with the reason.
    pub fn validate(&self) -> Result<(), ProfileFileError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ProfileFileError::Invalid(format!(
                "profile schema version {}, expected {SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if !self.fit_offset_ev.is_finite() || !FIT_OFFSET_RANGE_EV.contains(&self.fit_offset_ev) {
            return Err(ProfileFileError::Invalid(format!(
                "fit_offset_ev {} is outside {} to {}",
                self.fit_offset_ev,
                FIT_OFFSET_RANGE_EV.start(),
                FIT_OFFSET_RANGE_EV.end()
            )));
        }
        self.to_profile()
            .validate()
            .map_err(ProfileFileError::Invalid)
    }

    /// Parses the text of a profile file and validates it.
    ///
    /// # Errors
    ///
    /// [`ProfileFileError::Parse`] for text that is not a profile file, [`ProfileFileError::Invalid`] for a profile the
    /// writer would refuse.
    pub fn from_json_str(text: &str) -> Result<Self, ProfileFileError> {
        let file: Self = serde_json::from_str(text)?;
        file.validate()?;
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
}

/// The part of the fitted profile a DNG export uses: the tone curve and look table the writer embeds, and the
/// `BaselineExposure` offset they were fitted with.
#[derive(Clone, PartialEq)]
pub struct FittedProfile {
    /// `ProfileToneCurve` and `ProfileLookTable`.
    pub profile: DngProfile,
    /// Added to `log2(x_white)` to give `BaselineExposure`; `-4` when the curve covers scene values up to 16.
    pub fit_offset_ev: f64,
}

impl FittedProfile {
    /// The profile committed as `data/dng/profile.json`, parsed on first use and validated.
    ///
    /// Every export of the app and the CLI takes its profile from here, so a DNG is the same whichever front end wrote it.
    #[must_use]
    pub fn embedded() -> &'static Self {
        static EMBEDDED: OnceLock<FittedProfile> = OnceLock::new();
        EMBEDDED.get_or_init(|| {
            // Invariant: the committed file parses and validates; `the_embedded_profile_parses_and_validates` runs in CI
            // on every change, and `dngfit fit` refuses to write a file that does not validate.
            Self::parse_embedded()
                .expect("data/dng/profile.json is a valid profile (checked by a test)")
        })
    }

    /// Parses and validates [`EMBEDDED_PROFILE_JSON`] again (what [`FittedProfile::embedded`] does once).
    ///
    /// # Errors
    ///
    /// When the committed file is not a usable profile.
    pub fn parse_embedded() -> Result<Self, ProfileFileError> {
        ProfileFile::from_json_str(EMBEDDED_PROFILE_JSON).map(|f| f.to_fitted())
    }

    /// `BaselineExposure::Auto` with this profile's offset: what the writer is given with this profile.
    ///
    /// The profile is fitted to the cine rendering as the app shows it, with nothing in front of it. Adobe's
    /// renderer applies `2^BaselineExposure` to the linear values before the profile, which undoes the scaling of the
    /// stored values. For a photo whose 99.99th percentile is below the white floor the value is zero.
    #[must_use]
    pub fn baseline_exposure(&self) -> BaselineExposure {
        BaselineExposure::Auto {
            fit_offset_ev: self.fit_offset_ev,
        }
    }
}

impl fmt::Debug for FittedProfile {
    /// A summary: the look table alone holds tens of thousands of numbers.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedProfile")
            .field("fit_offset_ev", &self.fit_offset_ev)
            .field("tone_curve_points", &self.profile.tone_curve.len())
            .field("look_table_dims", &self.profile.look_table.dims)
            .field("look_table_encoding", &self.profile.look_table.encoding)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dng::{DngReport, LookTable};

    #[test]
    fn the_embedded_profile_parses_and_validates() {
        let file = ProfileFile::from_json_str(EMBEDDED_PROFILE_JSON).expect("the committed file");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.name, crate::dng::PROFILE_NAME);
        let fitted = FittedProfile::parse_embedded().expect("the committed file");
        fitted
            .profile
            .validate()
            .expect("a profile the writer accepts");
        assert_eq!(&fitted, FittedProfile::embedded());
        // The shape the fit documents: the curve covers scene values up to 16, with the look table of the fit.
        assert_eq!(fitted.fit_offset_ev, -4.0);
        assert!(fitted.profile.tone_curve.len() > 8);
        assert_eq!(fitted.profile.look_table.encoding, 1);
        assert!(
            fitted.profile.look_table.dims[2] >= 2,
            "a sRGB-encoded table needs at least two value divisions"
        );
        assert_ne!(fitted.profile, DngProfile::identity());
        // Parsing the text and writing it again gives the committed text: the file is in the form the tool writes.
        assert_eq!(file.to_json_text(), EMBEDDED_PROFILE_JSON);
    }

    #[test]
    fn the_embedded_profile_gives_a_baseline_exposure_of_zero_for_ordinary_photos() {
        let fitted = FittedProfile::embedded();
        let baseline = fitted.baseline_exposure();
        assert_eq!(
            baseline,
            BaselineExposure::Auto {
                fit_offset_ev: fitted.fit_offset_ev
            }
        );
        // x_white is floored at 2^-offset (16), where the offset cancels it: nothing is left.
        let floor = baseline.white_floor();
        assert_eq!(floor, 16.0);
        assert_eq!(baseline.resolve_ev(floor), 0.0);
    }

    #[test]
    fn the_debug_text_is_a_summary() {
        let text = format!("{:?}", FittedProfile::embedded());
        assert!(text.len() < 300, "{text}");
        assert!(text.contains("fit_offset_ev"), "{text}");
    }

    #[test]
    fn a_profile_round_trips_through_its_text() {
        let file = ProfileFile::from_profile(
            &DngProfile::identity(),
            -3.0,
            "test",
            serde_json::json!({"a": 1}),
        );
        let back = ProfileFile::from_json_str(&file.to_json_text()).expect("round trip");
        assert_eq!(back, file);
        assert_eq!(back.to_profile(), DngProfile::identity());
        assert_eq!(
            back.to_fitted(),
            FittedProfile {
                profile: DngProfile::identity(),
                fit_offset_ev: -3.0
            }
        );
    }

    #[test]
    fn a_broken_profile_is_refused_with_the_reason() {
        let mut file =
            ProfileFile::from_profile(&DngProfile::identity(), 0.0, "t", serde_json::Value::Null);
        file.validate().expect("the identity validates");
        file.tone_curve = vec![[0.0, 0.0], [0.0, 1.0]];
        assert!(matches!(
            ProfileFile::from_json_str(&file.to_json_text()),
            Err(ProfileFileError::Invalid(_))
        ));
        file.tone_curve = vec![[0.0, 0.0], [1.0, 1.0]];
        file.fit_offset_ev = 40.0;
        assert!(file.validate().is_err());
        file.fit_offset_ev = f64::NAN;
        assert!(file.validate().is_err());
        file.fit_offset_ev = 0.0;
        file.schema_version = 9;
        let e = file.validate().unwrap_err().to_string();
        assert!(e.contains("schema version 9"), "{e}");
        assert!(matches!(
            ProfileFile::from_json_str("{ not json"),
            Err(ProfileFileError::Parse(_))
        ));
        // a look table the writer would refuse is refused here too
        let mut file =
            ProfileFile::from_profile(&DngProfile::identity(), 0.0, "t", serde_json::Value::Null);
        file.look_table.dims = [2, 2, 1];
        file.look_table.data = LookTable {
            dims: [2, 2, 1],
            data: [0.0, 1.0, 1.0].repeat(4),
            encoding: 1,
        }
        .data;
        assert!(file.validate().is_err(), "one value division with sRGB");
    }

    /// The writer, given the embedded profile and a dim photo, writes a `BaselineExposure` of zero and a white level of 16.
    #[test]
    fn a_dim_photo_is_written_with_a_baseline_exposure_of_zero() {
        use crate::pixels::{ExportHooks, ImageRef, PixelEncoding};
        use crate::request::{DngInputs, ExportRequest, encode_to_vec};
        let (w, h) = (24_u32, 16_u32);
        let raw: Vec<f32> = (0..w * h * 3)
            .map(|i| 0.02 + (i % 7) as f32 * 0.1)
            .collect();
        let preview = vec![0.2_f32; raw.len()];
        let fitted = FittedProfile::embedded();
        let meta = crate::request::Metadata::default();
        let request = ExportRequest {
            pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
            format: crate::options::ExportFormat::Dng,
            color_space: crate::options::ColorSpace::Srgb,
            size: crate::options::Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
                orientation: 1,
                crop: None,
                baseline_exposure: fitted.baseline_exposure(),
                profile: &fitted.profile,
            }),
        };
        let (_, summary) = encode_to_vec(&request, &ExportHooks::none()).unwrap();
        let report: DngReport = summary.dng.expect("a DNG report");
        assert_eq!(report.x_white, 16.0);
        assert_eq!(report.baseline_exposure_ev, 0.0);
    }
}
