//! The D65 colour matrix of each Sony body, for the Before view (`docs/02-colour-pipeline.md`, section
//! The Before view).
//!
//! `data/cameras/color-matrices.json` lists, per Sony model ID, the body's published D65 colour matrix
//! (XYZ to camera RGB; numbers only, `docs/14-licensing.md` rule 8). The file is embedded in the binary
//! (like the ARRI matrix table and the DNG profile), so the app, the CLI and the tests read the same
//! numbers and nothing can be missing at run time; a test checks it against its JSON schema.
//!
//! Lookup follows the profile registry's rules ([`ColorMatrices::lookup`]): a Make other than Sony
//! gets the fallback; a model ID, when the file has one, decides; without one the Model string does;
//! a body that is not listed gets the fallback, the A7R IV entry, as the profile registry falls back to
//! the A7R IV profile. A body is never refused for lack of a matrix.
//!
//! The rows are normalised and inverted by `ym_color::camera_default`, not here.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

use serde::Deserialize;
use ym_color::camera_default::{CameraDefault, CameraDefaultError};
use ym_color::consts::linalg::Mat3;

use crate::detect::{clean, is_sony};
use crate::registry::normalize_model;

/// The embedded table: `data/cameras/color-matrices.json`.
pub const EMBEDDED_JSON: &str = include_str!("../../../data/cameras/color-matrices.json");

/// Schema version this code reads.
pub const SCHEMA_VERSION: u32 = 1;

/// Why a colour matrix table was refused.
#[derive(Debug, thiserror::Error)]
pub enum ColorMatricesError {
    /// The text is not JSON of the documented shape.
    #[error("colour matrices: {0}")]
    Json(#[from] serde_json::Error),
    /// The table breaks a rule of the format.
    #[error("colour matrices: {0}")]
    Invalid(String),
}

/// One body's matrix as published.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BodyMatrix {
    /// Sony's model ID (maker-note tag 0xB001).
    pub model_id: u16,
    /// The model string the camera writes (`ILCE-7M3`).
    pub model: String,
    /// The D65 colour matrix: XYZ to camera RGB, rows as published (not normalised).
    pub xyz_d65_to_camera: Mat3,
    /// Where the numbers come from when they are not the body's own published ones (the A1 II takes the
    /// A1's, the A3000 the generic E-mount matrix), and who decided it; `None` for a body's own.
    #[serde(default)]
    pub note: Option<String>,
}

/// How a lookup found its entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMatch {
    /// The file's Sony model ID is listed.
    ModelId,
    /// The file has no model ID and its Model string is listed.
    ModelString,
    /// The body is not listed (or is not a Sony): the fallback entry.
    Fallback,
}

/// A lookup result.
#[derive(Debug, Clone, Copy)]
pub struct MatrixLookup<'a> {
    /// The entry to use.
    pub body: &'a BodyMatrix,
    /// How it was found.
    pub how: MatrixMatch,
}

#[derive(Debug, Deserialize)]
struct File {
    schema: u32,
    fallback: String,
    bodies: Vec<BodyMatrix>,
}

/// The table of body colour matrices.
#[derive(Debug, Clone)]
pub struct ColorMatrices {
    bodies: Vec<BodyMatrix>,
    by_id: HashMap<u16, usize>,
    by_model: HashMap<String, usize>,
    fallback: usize,
}

static SHIPPED: LazyLock<ColorMatrices> = LazyLock::new(|| {
    // The embedded file is checked by this crate's tests (`color_matrices` test file and the unit tests
    // below), so a parse failure here is a build that never passed CI.
    ColorMatrices::from_json(EMBEDDED_JSON).expect("the embedded colour matrix table is valid")
});

impl ColorMatrices {
    /// The table embedded in this build.
    #[must_use]
    pub fn shipped() -> &'static ColorMatrices {
        &SHIPPED
    }

    /// Parses and validates a table.
    ///
    /// # Errors
    ///
    /// [`ColorMatricesError`] for malformed JSON, an unknown schema version, a repeated model ID or
    /// Model string, a matrix that cannot be normalised and inverted, or a fallback that is not listed.
    pub fn from_json(text: &str) -> Result<Self, ColorMatricesError> {
        let file: File = serde_json::from_str(text)?;
        if file.schema != SCHEMA_VERSION {
            return Err(ColorMatricesError::Invalid(format!(
                "schema {} is not supported (expected {SCHEMA_VERSION})",
                file.schema
            )));
        }
        let mut by_id = HashMap::new();
        let mut by_model = HashMap::new();
        for (i, body) in file.bodies.iter().enumerate() {
            if by_id.insert(body.model_id, i).is_some() {
                return Err(ColorMatricesError::Invalid(format!(
                    "model ID {} appears twice",
                    body.model_id
                )));
            }
            if by_model.insert(normalize_model(&body.model), i).is_some() {
                return Err(ColorMatricesError::Invalid(format!(
                    "model {:?} appears twice",
                    body.model
                )));
            }
            // A matrix the Before view could not use is refused here, not at the first photo.
            CameraDefault::new(&body.xyz_d65_to_camera, [1.0; 3], [1.0; 3]).map_err(
                |e: CameraDefaultError| ColorMatricesError::Invalid(format!("{}: {e}", body.model)),
            )?;
        }
        let fallback = by_model
            .get(&normalize_model(&file.fallback))
            .copied()
            .ok_or_else(|| {
                ColorMatricesError::Invalid(format!(
                    "the fallback {:?} is not listed",
                    file.fallback
                ))
            })?;
        Ok(Self {
            bodies: file.bodies,
            by_id,
            by_model,
            fallback,
        })
    }

    /// Every listed body, in file order.
    #[must_use]
    pub fn bodies(&self) -> &[BodyMatrix] {
        &self.bodies
    }

    /// The fallback entry (the A7R IV).
    #[must_use]
    pub fn fallback(&self) -> &BodyMatrix {
        &self.bodies[self.fallback]
    }

    /// The matrix of the body a file names, as the profile registry would find the body: a Make other
    /// than Sony (an empty Make counts as Sony) gets the fallback; with a model ID, the entry with that
    /// ID, and an ID the table does not list gets the fallback even if the Model string looks familiar
    /// (the ID wins, `docs/03`); without an ID, the entry the Model string names (case and spacing
    /// ignored); anything else gets the fallback.
    #[must_use]
    pub fn lookup(&self, make: &str, model: &str, model_id: Option<u16>) -> MatrixLookup<'_> {
        let fallback = MatrixLookup {
            body: self.fallback(),
            how: MatrixMatch::Fallback,
        };
        if !is_sony(&clean(make)) {
            return fallback;
        }
        match model_id {
            Some(id) => self.by_id.get(&id).map_or(fallback, |&i| MatrixLookup {
                body: &self.bodies[i],
                how: MatrixMatch::ModelId,
            }),
            None => self
                .by_model
                .get(&normalize_model(model))
                .map_or(fallback, |&i| MatrixLookup {
                    body: &self.bodies[i],
                    how: MatrixMatch::ModelString,
                }),
        }
    }

    /// The path of the shipped JSON under a data root (for the data checks and the docs).
    #[must_use]
    pub fn data_path(data_root: impl Into<PathBuf>) -> PathBuf {
        data_root.into().join("cameras").join("color-matrices.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_table_is_valid_and_has_the_fallback() {
        let t = ColorMatrices::shipped();
        assert!(t.bodies().len() >= 40, "{}", t.bodies().len());
        assert_eq!(t.fallback().model, "ILCE-7RM4");
        assert_eq!(t.fallback().model_id, 375);
    }

    #[test]
    fn lookup_follows_the_registry_rules() {
        let t = ColorMatrices::shipped();
        // The model ID decides, whatever the Model string says.
        let a = t.lookup("SONY", "ILCE-7M3", Some(363));
        assert_eq!(
            (a.body.model.as_str(), a.how),
            ("ILCE-7M3", MatrixMatch::ModelId)
        );
        let b = t.lookup("SONY", "ILCE-7M3", Some(375));
        assert_eq!(b.body.model, "ILCE-7RM4");
        assert_eq!(b.how, MatrixMatch::ModelId);
        // Without an ID the Model string is used, case and a leading maker ignored.
        let c = t.lookup("SONY", " sony ilce-7m3\0", None);
        assert_eq!(
            (c.body.model.as_str(), c.how),
            ("ILCE-7M3", MatrixMatch::ModelString)
        );
        // An ID the table does not list, an unknown string and a Make other than Sony: the fallback.
        for l in [
            t.lookup("SONY", "ILCE-7M3", Some(60000)),
            t.lookup("SONY", "ILCE-9999", None),
            t.lookup("Hasselblad", "ILCE-7M3", Some(363)),
        ] {
            assert_eq!(l.how, MatrixMatch::Fallback);
            assert_eq!(l.body.model, "ILCE-7RM4");
        }
        // An empty Make counts as Sony, as in the registry.
        assert_eq!(t.lookup("", "", Some(363)).body.model, "ILCE-7M3");
    }

    #[test]
    fn a_file_finds_its_own_body_by_the_id_the_camera_writes() {
        // The A7, A7R and A6000 have neighbouring IDs (306, 311, 312) that the first draft mixed up, and
        // the model string must not matter once the file has an ID.
        let t = ColorMatrices::shipped();
        for (id, model) in [(306, "ILCE-7"), (311, "ILCE-7R"), (312, "ILCE-6000")] {
            let l = t.lookup("SONY", model, Some(id));
            assert_eq!(
                (l.body.model.as_str(), l.how),
                (model, MatrixMatch::ModelId),
                "ID {id}"
            );
            // Another body's string behind the same ID changes nothing.
            assert_eq!(t.lookup("SONY", "ILCE-7M3", Some(id)).body.model, model);
        }
        // The NEX-6 (295), NEX-5T (307), A7S (318) and A6500 (360) are listed, not the fallback.
        for (id, model) in [
            (295, "NEX-6"),
            (307, "NEX-5T"),
            (318, "ILCE-7S"),
            (360, "ILCE-6500"),
        ] {
            let l = t.lookup("SONY", "", Some(id));
            assert_eq!(
                (l.body.model.as_str(), l.how),
                (model, MatrixMatch::ModelId)
            );
        }
    }

    #[test]
    fn bodies_without_a_real_entry_use_the_fallback() {
        // These two are left out of the table (data/cameras/color-matrices.json, `excluded`): none
        // published, or the numbers of another body. The A1 II and the A3000 were left out too until
        // the owner's decision of 2026-10-16; they have entries now (the next test).
        let t = ColorMatrices::shipped();
        for (model, id) in [("ILCE-7M5", Some(407)), ("ILCE-7RM6", Some(410))] {
            for l in [t.lookup("SONY", model, id), t.lookup("SONY", model, None)] {
                assert_eq!(l.how, MatrixMatch::Fallback, "{model}");
                assert_eq!(l.body.model, "ILCE-7RM4", "{model}");
            }
        }
    }

    #[test]
    fn the_a1_ii_and_the_a3000_borrow_numbers_by_decision() {
        // Owner's decision 2026-10-16 (`docs/STATUS.md`, question 29): the A1 II shares the A1's sensor and
        // takes the A1's numbers; the A3000 has none of its own and takes the generic E-mount numbers,
        // which are the NEX-5N's (and the A6000's, A5000's and A5100's).
        let t = ColorMatrices::shipped();
        let by_id = |id: u16| t.lookup("SONY", "", Some(id));
        let a1 = by_id(384);
        assert_eq!(a1.body.model, "ILCE-1");
        let a1_ii = by_id(400);
        assert_eq!(
            (a1_ii.body.model.as_str(), a1_ii.how),
            ("ILCE-1M2", MatrixMatch::ModelId)
        );
        assert_eq!(a1_ii.body.xyz_d65_to_camera, a1.body.xyz_d65_to_camera);
        let nex_5n = by_id(288);
        assert_eq!(nex_5n.body.model, "NEX-5N");
        let a3000 = by_id(302);
        assert_eq!(
            (a3000.body.model.as_str(), a3000.how),
            ("ILCE-3000", MatrixMatch::ModelId)
        );
        assert_eq!(a3000.body.xyz_d65_to_camera, nex_5n.body.xyz_d65_to_camera);
        // Without an ID the Model string finds them too, and neither is the fallback any more.
        for (model, want) in [("ILCE-1M2", 400), ("ILCE-3000", 302)] {
            let l = t.lookup("SONY", model, None);
            assert_eq!(
                (l.how, l.body.model_id),
                (MatrixMatch::ModelString, want),
                "{model}"
            );
        }
        // Each says where its numbers come from.
        for body in [a1_ii.body, a3000.body] {
            let note = body.note.as_deref().unwrap_or_default();
            assert!(note.contains("2026-10-16"), "{}: {note:?}", body.model);
        }
        assert!(a1.body.note.is_none() && nex_5n.body.note.is_none());
    }

    #[test]
    fn bad_tables_are_refused() {
        let ok = r#"{"schema":1,"fallback":"A","bodies":[
            {"model_id":1,"model":"A","xyz_d65_to_camera":[[0.7,-0.2,-0.05],[-0.4,1.2,0.24],[-0.07,0.14,0.58]]}]}"#;
        ColorMatrices::from_json(ok).unwrap();
        for (bad, why) in [
            (ok.replace("\"schema\":1", "\"schema\":2"), "schema"),
            (ok.replace("\"fallback\":\"A\"", "\"fallback\":\"B\""), "fallback"),
            (ok.replace("-0.2,", "null,"), ""),
            (ok.replace("0.58]]", "0.58]]},{\"model_id\":1,\"model\":\"B\",\"xyz_d65_to_camera\":[[1,0,0],[0,1,0],[0,0,1]]"), "twice"),
            (ok.replace("0.58]]", "0.58]]},{\"model_id\":2,\"model\":\"a\",\"xyz_d65_to_camera\":[[1,0,0],[0,1,0],[0,0,1]]"), "twice"),
            (ok.replace("[[0.7,-0.2,-0.05],[-0.4,1.2,0.24],[-0.07,0.14,0.58]]", "[[0.5,0.4,0.3],[0.5,0.4,0.3],[0.5,0.4,0.3]]"), "inverse"),
        ] {
            let err = ColorMatrices::from_json(&bad).unwrap_err().to_string();
            assert!(err.contains(why), "{bad}: {err}");
        }
    }
}
