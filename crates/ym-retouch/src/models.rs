//! The model traits of `docs/27` section 4. The crate bundles no weights; implementations live in
//! other crates, and [`crate::mock`] holds deterministic ones for tests.

use crate::error::Result;
use crate::types::{BoundedMask, Face, LinearAwg4Image, Rect};

/// A part of the face a [`FaceParser`] can find.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FacePart {
    /// Face skin without the ears.
    SkinWithoutEars,
    /// Both eyes.
    Eyes,
    /// The inside of the mouth (teeth, gums, tongue), inside the inner lip line.
    MouthInterior,
}

/// A table of the face mesh: a smooth texture over the mesh's texture coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshTable {
    /// Bright under the eyes.
    UnderEye,
    /// Bright where the skin is not uniform (features, edges); even skin leaves those alone.
    InvertedSkinUniformity,
}

/// Finds the parts of a face.
pub trait FaceParser: Send + Sync {
    /// Soft mask of one face part, full image coordinates; `None` when the part is not found.
    fn part_mask(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
        part: FacePart,
    ) -> Result<Option<BoundedMask>>;
}

/// A dense face mesh with two tables.
pub trait FaceMesh: Send + Sync {
    /// Rasterises the face mesh into an 8-bit mask of one of its tables, in full image coordinates.
    /// `image` is the full-resolution image the mesh is fitted to (an implementation that runs a network needs the picture;
    /// the analytic mock ignores it), `size_px` its size (width, height).
    fn render(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
        table: MeshTable,
        size_px: (u32, u32),
    ) -> Result<BoundedMask>;
}

/// Finds moles and beauty marks.
pub trait BeautyMarkDetector: Send + Sync {
    /// Boxes of moles and beauty marks that blemish removal leaves alone.
    fn detect(&self, image: &LinearAwg4Image, face: &Face) -> Result<Vec<Rect>>;
}

/// Makes a cleaned copy of a face tile.
pub trait BlemishProposer: Send + Sync {
    /// Side of the square output tile, in pixels.
    fn output_size_px(&self) -> u32;
    /// Border consumed on each side: input side = output side + 2 * border.
    fn border_px(&self) -> u32;
    /// Planar RRR..GGG..BBB in, planar out, both in the encoding of `docs/27` 6.1 step 2.
    fn run(&self, input: &[f32]) -> Result<Vec<f32>>;
    /// Like [`BlemishProposer::run`], and a plane of weights too (`output_size_px()` squared values in 0..=1, rows top to
    /// bottom; `None`: 1 everywhere). The map multiplies its strength by the weight (`docs/27` 6.1 step 6), so a
    /// proposer that is less sure of a correction gets it only at a higher Blemish amount (S-16). The default
    /// has no weights.
    fn run_weighted(&self, input: &[f32]) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        Ok((self.run(input)?, None))
    }
    /// A proposer made for a face of `face_px` pixels in the blemish map (`min(face, 450)`), for a proposer whose constants
    /// scale with the face (the classical one). `None` (the default) means this one serves every face.
    fn for_face(&self, _face_px: f64) -> Option<Box<dyn BlemishProposer>> {
        None
    }
}

/// Finds the teeth of a face with an open mouth (`docs/27` 5.6, S-17): SAM 2.1 small prompted from the face mesh.
///
/// The answer is a soft mask of where the teeth are, in full image coordinates, with no colour rule behind it. It is made
/// once per analysis and cached with the other masks; the caller intersects it with the mouth's interior.
pub trait TeethSegmenter: Send + Sync {
    /// The teeth of `face`, as a soft mask; `Ok(None)` when the segmenter found nothing it can stand behind (it logs why), in
    /// which case the face gets no teeth whitening. Only asked for faces whose mouth is open.
    fn teeth(&self, image: &LinearAwg4Image, face: &Face) -> Result<Option<BoundedMask>>;
}

/// The models one retouch run uses.
pub struct Models {
    /// Skin without ears, eyes and the mouth interior. A missing mouth interior only turns
    /// teeth whitening off.
    pub parser: Box<dyn FaceParser>,
    /// The face mesh with its two tables.
    pub mesh: Box<dyn FaceMesh>,
    /// Beauty marks; without it nothing is left alone. A failing detector is logged and ignored.
    pub beauty_marks: Option<Box<dyn BeautyMarkDetector>>,
    /// The blemish proposer.
    pub proposer: Box<dyn BlemishProposer>,
    /// The teeth. Without it (for example when the SAM weights are missing) teeth whitening is off for every face, and the
    /// log says so; there is no fall back to colours. A failing segmenter is logged and turns teeth whitening off for that face.
    pub teeth: Option<Box<dyn TeethSegmenter>>,
}
