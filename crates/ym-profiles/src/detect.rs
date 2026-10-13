//! Detection: turns a file's Make, Model and Sony model ID into a profile choice and the reason for it
//! (exact, sensor family, default) (`docs/03-camera-profiles.md` sections "Detection" and "Lookup").
//!
//! * The model ID wins when it disagrees with the Model string (firmware variants and regional "A"
//!   models write odd strings).
//! * Without an ID, the registry's Model-string aliases are tried.
//! * Any other Sony body, and any ARW whose Make is not Sony (the rebadged Hasselblad Lunar and
//!   Stellar), gets the default profile. **A file is never refused for lack of a profile.**
//! * `--profile <id>` forces a profile ([`ProfileStore::detect_forced`]).
//!
//! [`CameraMatch`] does not carry the decoder's `DecodeKind` (it lives in `ym_raw::RawMetadata`);
//! `ym-core` composes the two for the UI and the CLI. The display copy of the camera card is here too
//! ([`CameraCopy`]), so the app and `ym info` say the same thing.

use serde::{Deserialize, Serialize};

use crate::profile::{ProfileError, ProfileId};
use crate::registry::{Body, BodyMatch, ProfileStore};

/// Why a profile was chosen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MatchKind {
    /// The profile was measured on this body.
    Exact,
    /// The body shares its sensor with the profile's measured body, named by `via` (`Sony A7 III`).
    SensorFamily {
        /// Display name of the profile's measured body.
        via: String,
    },
    /// No profile of its own or a sibling's: the default (A7R IV) stand-in.
    Default,
    /// The profile was forced on the command line (`--profile`), and is not the one detection chose.
    ///
    /// Not part of `docs/03`'s `Exact | SensorFamily | Default`; added for the CLI override so a forced
    /// profile is never reported as a measured match.
    Forced {
        /// The profile detection would have chosen.
        detected: ProfileId,
    },
}

/// The result of camera detection: who the body is and which profile it gets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CameraMatch {
    /// IFD0 Make, as read and trimmed (`SONY`).
    pub make: String,
    /// The body's Model string: the registry's name when the body is known (the model ID wins), else
    /// the trimmed string from the file.
    pub model: String,
    /// Sony's maker-note model ID (tag 0xB001) as the file gave it; `None` when the file had no usable one
    /// (the body was then found through the Model string, or not at all).
    pub model_id: Option<u16>,
    /// Display name, for example `Sony A1`, from the registry or built from Make and Model.
    pub body: String,
    /// The profile used.
    pub profile: ProfileId,
    /// That profile's version, written into export metadata.
    pub profile_version: semver::Version,
    /// Why this profile.
    pub how: MatchKind,
}

impl CameraMatch {
    /// Whether the profile is a stand-in rather than one measured on this body (the app's STAND-IN tag).
    #[must_use]
    pub fn is_stand_in(&self) -> bool {
        !matches!(self.how, MatchKind::Exact)
    }
}

/// Display copy for the camera card and the Automatic card (`docs/11-ui.md`), shared by the app and the
/// CLI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CameraCopy {
    /// `Detected` row: body and model, for example `Sony A1 (ILCE-1)`.
    pub detected: String,
    /// `Profile` row: the profile's short name, for example `A7R IV`.
    pub profile: String,
    /// The Automatic card's camera row: `A7R IV` for a measured body, else `<body> · <profile> profile`
    /// (`A1 · A7R IV profile`).
    pub automatic_row: String,
    /// Whether to show the STAND-IN tag.
    pub stand_in: bool,
    /// The explanation under the card, absent for a measured body.
    pub note: Option<String>,
}

fn is_sony(make: &str) -> bool {
    // An absent Make is treated as Sony: the maker-note ID is the better evidence then.
    make.is_empty() || make.to_ascii_lowercase().starts_with("sony")
}

fn clean(s: &str) -> String {
    s.trim_matches(|c: char| c.is_whitespace() || c == '\0')
        .to_string()
}

/// Title-cases each word: `HASSELBLAD` becomes `Hasselblad`.
fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Display name for a body the registry does not know: `Sony ILCE-7M6`, `Hasselblad Lunar`.
fn display_body(make: &str, model: &str) -> String {
    let make = title_case(make);
    match (make.is_empty(), model.is_empty()) {
        (true, true) => "Unknown camera".to_string(),
        (true, false) => model.to_string(),
        (false, true) => make,
        (false, false) => {
            if model
                .to_ascii_lowercase()
                .starts_with(&make.to_ascii_lowercase())
            {
                model.to_string()
            } else {
                format!("{make} {model}")
            }
        }
    }
}

/// The body name without the maker: `Sony A1` becomes `A1`.
fn short_name(body: &str) -> &str {
    match body.get(..5) {
        Some(prefix) if prefix.eq_ignore_ascii_case("sony ") => &body[5..],
        _ => body,
    }
}

/// `Eleven` for 11; digits above twenty.
fn count_in_words(n: usize) -> String {
    const WORDS: [&str; 21] = [
        "Zero",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
        "Eleven",
        "Twelve",
        "Thirteen",
        "Fourteen",
        "Fifteen",
        "Sixteen",
        "Seventeen",
        "Eighteen",
        "Nineteen",
        "Twenty",
    ];
    WORDS
        .get(n)
        .map_or_else(|| n.to_string(), |w| (*w).to_string())
}

/// Detects the body and chooses its profile: [`ProfileStore::detect`] as a free function.
///
/// `make` and `model` are IFD0 Make and Model, `model_id` Sony's maker-note model ID (0xB001) when the
/// file has a usable one. Never fails: a file without a profile of its own gets the default.
#[must_use]
pub fn detect(store: &ProfileStore, make: &str, model: &str, model_id: Option<u16>) -> CameraMatch {
    store.detect(make, model, model_id)
}

impl ProfileStore {
    fn camera_match(
        &self,
        make: String,
        model: String,
        model_id: Option<u16>,
        body: String,
        profile: &ProfileId,
        how: MatchKind,
    ) -> CameraMatch {
        // Every profile id comes from the registry, which `load` checked against the loaded profiles; the
        // default is the fallback so a match can always be built.
        let profile = self
            .profile(profile)
            .unwrap_or_else(|| self.default_profile());
        CameraMatch {
            make,
            model,
            model_id,
            body,
            profile: profile.id.clone(),
            profile_version: profile.version.clone(),
            how,
        }
    }

    /// Detects the body and chooses its profile.
    ///
    /// `make` and `model` are IFD0 Make (0x010F) and Model (0x0110); `model_id` is Sony's maker-note
    /// model ID (0xB001) when the file has one. Rules, in order:
    ///
    /// 1. A Make other than Sony (Hasselblad's rebadged bodies): default profile.
    /// 2. A model ID: the registry body with that ID. An ID the registry does not list is a body we
    ///    have not seen: default profile, even if the Model string looks familiar (the ID wins).
    /// 3. No ID: the registry body the Model string names (case and spacing ignored). An unknown string:
    ///    default profile.
    ///
    /// Never fails: the result always names a profile that exists.
    #[must_use]
    pub fn detect(&self, make: &str, model: &str, model_id: Option<u16>) -> CameraMatch {
        let make = clean(make);
        let model = clean(model);
        let registry = self.registry();
        let known: Option<&Body> = if is_sony(&make) {
            match model_id {
                Some(id) => registry.body_by_model_id(id),
                None => registry.body_by_model_string(&model),
            }
        } else {
            None
        };
        match known {
            Some(body) => {
                let how = match body.match_kind {
                    BodyMatch::Exact => MatchKind::Exact,
                    BodyMatch::SensorFamily => MatchKind::SensorFamily {
                        via: body.via.clone().unwrap_or_default(),
                    },
                    BodyMatch::Default => MatchKind::Default,
                };
                let make = if make.is_empty() {
                    "SONY".to_string()
                } else {
                    make
                };
                self.camera_match(
                    make,
                    body.model.clone(),
                    model_id,
                    body.body.clone(),
                    &body.profile,
                    how,
                )
            }
            None => {
                let body = display_body(&make, &model);
                let default = registry.default_profile.clone();
                self.camera_match(make, model, model_id, body, &default, MatchKind::Default)
            }
        }
    }

    /// Detects the body like [`ProfileStore::detect`], then forces `profile_id` (the CLI's
    /// `--profile`, for comparing a new profile against the default). When it is the profile detection
    /// chose, the detected match is returned unchanged; otherwise `how` is [`MatchKind::Forced`].
    ///
    /// # Errors
    ///
    /// [`ProfileError::UnknownProfile`] listing the valid ids.
    pub fn detect_forced(
        &self,
        make: &str,
        model: &str,
        model_id: Option<u16>,
        profile_id: &str,
    ) -> Result<CameraMatch, ProfileError> {
        let Some(forced) = self.profile_by_name(profile_id.trim()) else {
            return Err(ProfileError::UnknownProfile {
                requested: profile_id.to_string(),
                available: self
                    .profile_ids()
                    .map(ProfileId::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        };
        let mut found = self.detect(make, model, model_id);
        if found.profile != forced.id {
            found.how = MatchKind::Forced {
                detected: found.profile.clone(),
            };
            found.profile = forced.id.clone();
            found.profile_version = forced.version.clone();
        }
        Ok(found)
    }

    /// The profile a match refers to; the default if the match names one this store does not hold (a
    /// match built elsewhere), so a photo can always be developed.
    #[must_use]
    pub fn resolve(&self, found: &CameraMatch) -> &std::sync::Arc<crate::profile::Profile> {
        self.profile(&found.profile)
            .unwrap_or_else(|| self.default_profile())
    }

    /// The camera card and Automatic card copy for a match (`docs/11-ui.md`).
    ///
    /// * Default: "The A1 has no measured profile yet, so it borrows the A7R IV. Eleven Sony cameras
    ///   have their own, and cameras that share their sensors use them."
    /// * Sensor family: "The A7C shares its sensor with the A7 III, so it uses that profile."
    /// * Forced: a sentence for the command-line comparison.
    /// * Exact: no note.
    #[must_use]
    pub fn camera_copy(&self, found: &CameraMatch) -> CameraCopy {
        let profile = self.resolve(found);
        let short = short_name(&found.body);
        let profile_name = profile.display_name.clone();
        // `Sony A1 (ILCE-1)`, but not `Sony ZV-E1 (ZV-E1)` or `Hasselblad Lunar (Lunar)`: the model in
        // brackets is only worth showing when the body name does not already say it.
        let detected = if found.model.is_empty()
            || found
                .body
                .to_ascii_lowercase()
                .contains(&found.model.to_ascii_lowercase())
        {
            found.body.clone()
        } else {
            format!("{} ({})", found.body, found.model)
        };
        let note = match &found.how {
            MatchKind::Exact => None,
            MatchKind::SensorFamily { via } => Some(format!(
                "The {short} shares its sensor with the {}, so it uses that profile.",
                short_name(via)
            )),
            MatchKind::Default => Some(format!(
                "The {short} has no measured profile yet, so it borrows the {profile_name}. {} Sony cameras have their own, and cameras that share their sensors use them.",
                count_in_words(self.own_profile_count())
            )),
            MatchKind::Forced { detected } => {
                let detected_name = self
                    .profile(detected)
                    .map_or_else(|| detected.to_string(), |p| p.display_name.clone());
                Some(format!(
                    "The profile was chosen on the command line, so the {short} uses the {profile_name} instead of the {detected_name}."
                ))
            }
        };
        let automatic_row = if found.how == MatchKind::Exact && short == profile_name {
            profile_name.clone()
        } else {
            format!("{short} · {profile_name} profile")
        };
        CameraCopy {
            detected,
            profile: profile_name,
            automatic_row,
            stand_in: found.is_stand_in(),
            note,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_names_are_built_from_make_and_model() {
        assert_eq!(display_body("SONY", "ILCE-7M6"), "Sony ILCE-7M6");
        assert_eq!(display_body("HASSELBLAD", "Lunar"), "Hasselblad Lunar");
        assert_eq!(
            display_body("Hasselblad", "HASSELBLAD Stellar II"),
            "HASSELBLAD Stellar II"
        );
        assert_eq!(display_body("", "ILCE-9"), "ILCE-9");
        assert_eq!(display_body("SONY", ""), "Sony");
        assert_eq!(display_body("", ""), "Unknown camera");
        assert_eq!(title_case("sony   corporation"), "Sony Corporation");
    }

    #[test]
    fn short_names_and_counts() {
        assert_eq!(short_name("Sony A1"), "A1");
        assert_eq!(short_name("sony A1"), "A1");
        assert_eq!(short_name("Hasselblad Lunar"), "Hasselblad Lunar");
        assert_eq!(short_name("é"), "é");
        assert_eq!(count_in_words(11), "Eleven");
        assert_eq!(count_in_words(0), "Zero");
        assert_eq!(count_in_words(20), "Twenty");
        assert_eq!(count_in_words(21), "21");
    }

    #[test]
    fn make_check() {
        assert!(is_sony("SONY"));
        assert!(is_sony("Sony"));
        assert!(is_sony("Sony Corporation"));
        assert!(is_sony(""));
        assert!(!is_sony("HASSELBLAD"));
        assert!(!is_sony("Canon"));
    }

    #[test]
    fn match_kind_serialises_with_a_tag() {
        let json = serde_json::to_string(&MatchKind::SensorFamily {
            via: "Sony A7 III".to_string(),
        })
        .unwrap();
        assert_eq!(json, r#"{"kind":"sensor_family","via":"Sony A7 III"}"#);
        assert_eq!(
            serde_json::to_string(&MatchKind::Exact).unwrap(),
            r#"{"kind":"exact"}"#
        );
        assert_eq!(
            serde_json::to_string(&MatchKind::Default).unwrap(),
            r#"{"kind":"default"}"#
        );
        let forced: MatchKind =
            serde_json::from_str(r#"{"kind":"forced","detected":"sony-ilce-9"}"#).unwrap();
        assert_eq!(
            forced,
            MatchKind::Forced {
                detected: ProfileId::new("sony-ilce-9").unwrap()
            }
        );
    }
}
