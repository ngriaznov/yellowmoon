//! The Face Refinement node's settings (`docs/27-skin-retouch.md` sections 3 and 10, task S-11).
//!
//! A Face Refinement node is a numbered node of the graph, like a Film node: `picture` and `key` ports, masks, Bypass.
//! Its ids share the number space of the correctors. It carries the settings of the retouching tools per face
//! ([`FaceSettings`]), a switch that makes one set serve every face, and a protection plane painted like a Brush mask.
//!
//! Faces are found automatically and numbered from left to right, 1 to 32. Nothing about the faces themselves is saved:
//! the settings of face N are the Nth entry of [`FaceRefinementNode::faces`], and a face past the end of the list takes
//! face 1's settings ([`FaceRefinementNode::settings_for_face`]).

use serde::{Deserialize, Serialize};
use ym_retouch::SkinSettings;

use super::NodePos;
use super::graph::DEFAULT_FACE_REFINEMENT_NAME;
use super::masks::{BrushMask, MaskSettings};

/// Range of Blemish, Dark circles, Even skin, Shadow desaturation, Shine removal and Whitening.
pub const FACE_AMOUNT_RANGE: (f64, f64) = (0.0, 100.0);
/// Range of Texture and of Skin contrast.
pub const FACE_BIPOLAR_RANGE: (f64, f64) = (-100.0, 100.0);
/// The most faces a node holds settings for (`docs/27` section 11 item 9: at most 32 faces are kept).
pub const MAX_FACES: usize = 32;

/// The tools' amounts for one face (`docs/27` section 3).
///
/// JSON: the field names below; a field left out takes the default (Blemish 30, Dark circles 30, Even skin 25 and
/// everything else 0, the starting point the owner approved at S-H1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FaceSettings {
    /// Blemish removal, 0 to 100.
    pub blemish: f64,
    /// Dark circles, 0 to 100.
    pub dark_circles: f64,
    /// Even skin, 0 to 100.
    pub even_skin: f64,
    /// Texture, -100 to 100 (0 keeps the skin's fine detail as it is).
    pub texture: f64,
    /// Skin grading › Contrast, -100 to 100.
    pub skin_contrast: f64,
    /// Skin grading › Shadow desaturation, 0 to 100.
    pub shadow_desaturation: f64,
    /// Skin grading › Shine removal, 0 to 100.
    pub shine_removal: f64,
    /// Teeth › Whitening, 0 to 100.
    pub teeth_whitening: f64,
}

impl Default for FaceSettings {
    /// Blemish 30, Dark circles 30, Even skin 25, Texture 0 and the grading and teeth at 0: what a new node does and
    /// what Reset returns to (`docs/27` section 11 item 9).
    fn default() -> Self {
        Self {
            blemish: 30.0,
            dark_circles: 30.0,
            even_skin: 25.0,
            texture: 0.0,
            skin_contrast: 0.0,
            shadow_desaturation: 0.0,
            shine_removal: 0.0,
            teeth_whitening: 0.0,
        }
    }
}

impl FaceSettings {
    /// The settings for `ym_retouch` (`f32`, the same names). Out-of-range values are the library's to clamp
    /// ([`SkinSettings::clamped`]); a node read through the settings has them in range already.
    #[must_use]
    pub fn to_skin_settings(&self) -> SkinSettings {
        SkinSettings {
            blemish: self.blemish as f32,
            dark_circles: self.dark_circles as f32,
            even_skin: self.even_skin as f32,
            texture: self.texture as f32,
            skin_contrast: self.skin_contrast as f32,
            shadow_desaturation: self.shadow_desaturation as f32,
            shine_removal: self.shine_removal as f32,
            teeth_whitening: self.teeth_whitening as f32,
        }
    }

    /// Whether every value is 0 (Texture and Contrast included, which is their neutral point too).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.blemish == 0.0
            && self.dark_circles == 0.0
            && self.even_skin == 0.0
            && self.texture == 0.0
            && self.skin_contrast == 0.0
            && self.shadow_desaturation == 0.0
            && self.shine_removal == 0.0
            && self.teeth_whitening == 0.0
    }

    /// Clamps every amount to its range and returns whether any was changed.
    ///
    /// # Errors
    ///
    /// [`SettingsError::NotFinite`](super::SettingsError) for a number that is not a number.
    pub(crate) fn clamp(&mut self, prefix: &str) -> Result<bool, super::SettingsError> {
        let mut changed = false;
        for (name, v, range) in [
            ("blemish", &mut self.blemish, FACE_AMOUNT_RANGE),
            ("dark_circles", &mut self.dark_circles, FACE_AMOUNT_RANGE),
            ("even_skin", &mut self.even_skin, FACE_AMOUNT_RANGE),
            ("texture", &mut self.texture, FACE_BIPOLAR_RANGE),
            ("skin_contrast", &mut self.skin_contrast, FACE_BIPOLAR_RANGE),
            (
                "shadow_desaturation",
                &mut self.shadow_desaturation,
                FACE_AMOUNT_RANGE,
            ),
            ("shine_removal", &mut self.shine_removal, FACE_AMOUNT_RANGE),
            (
                "teeth_whitening",
                &mut self.teeth_whitening,
                FACE_AMOUNT_RANGE,
            ),
        ] {
            let (x, c) = super::clamp_to(&format!("{prefix}.{name}"), *v, range)?;
            *v = x;
            changed |= c;
        }
        Ok(changed)
    }
}

/// A Face Refinement node (`docs/27` section 10, S-11).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FaceRefinementNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed node passes its picture through; its masks and protection still exist.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// "Same for all faces": when true, [`all`](Self::all) serves every face and [`faces`](Self::faces) is kept but
    /// not used.
    pub same_for_all: bool,
    /// The settings every face takes while `same_for_all` is true, and the settings a face takes when the list of
    /// per-face settings is empty.
    pub all: FaceSettings,
    /// The settings per face; the entry at index `n - 1` is face `n`'s (faces are numbered from 1, left to right). At
    /// most [`MAX_FACES`] entries. A face past the end takes face 1's settings.
    pub faces: Vec<FaceSettings>,
    /// The protection plane: strokes painted on the photo, in the format of a Brush mask row. Painted areas are left
    /// alone by every tool (`docs/27` section 11 item 9). One plane for the node, whatever the face.
    pub protect: BrushMask,
    /// The masks, as a corrector's.
    pub masks: MaskSettings,
}

impl Default for FaceRefinementNode {
    /// "Same for all faces" on, the default amounts, no per-face settings, nothing protected.
    fn default() -> Self {
        Self {
            name: DEFAULT_FACE_REFINEMENT_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            same_for_all: true,
            all: FaceSettings::default(),
            faces: Vec::new(),
            protect: BrushMask::default(),
            masks: MaskSettings::default(),
        }
    }
}

impl FaceRefinementNode {
    /// The settings of face `face_number` (1 for the leftmost face; 0 reads as 1).
    ///
    /// * `same_for_all`: [`all`](Self::all), for every face.
    /// * otherwise the entry `face_number - 1` of [`faces`](Self::faces); a face past the end of the list takes face
    ///   1's entry, and when the list is empty `all` (`docs/27` section 11 item 9).
    #[must_use]
    pub fn settings_for_face(&self, face_number: u32) -> FaceSettings {
        if self.same_for_all {
            return self.all;
        }
        let index = (face_number.max(1) - 1) as usize;
        self.faces
            .get(index)
            .or_else(|| self.faces.first())
            .copied()
            .unwrap_or(self.all)
    }

    /// [`settings_for_face`](Self::settings_for_face) as the settings of `ym_retouch`.
    #[must_use]
    pub fn skin_settings_for_face(&self, face_number: u32) -> SkinSettings {
        self.settings_for_face(face_number).to_skin_settings()
    }

    /// Whether the node does nothing to the picture whatever faces are found: bypassed, or every tool of every face
    /// that can be used is off. Painted protection does not matter then.
    #[must_use]
    pub fn is_inert(&self) -> bool {
        if self.bypass {
            return true;
        }
        if self.same_for_all || self.faces.is_empty() {
            return self.all.is_zero();
        }
        self.faces.iter().all(FaceSettings::is_zero)
    }
}
