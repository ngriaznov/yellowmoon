//! The real `FaceParser` and `FaceMesh`: MediaPipe's Face Mesh V2 and Selfie Multiclass run by `ym-ml` (feature `mediapipe`;
//! `docs/27` S-09 and S-10, decision 12).
//!
//! [`MediaPipeFaces`] owns the two networks and turns a face of a picture into a [`FaceAnalysis`] (the 478 landmarks and the
//! segmenter's face-skin probability); [`MediaPipeParser`] and [`MediaPipeMesh`] are two views of it that implement
//! [`FaceParser`] and [`FaceMesh`], so one analysis (about 110 ms, `docs/27`) serves the three parts and the two tables of
//! a face. The geometry lives in the parts that need no network ([`crate::face_parts`]).
//!
//! **The analysis** of a face (`Face::rect`, no eye keypoints are known here): Face Mesh V2 on the unturned crop of the box
//! (1.5 times its long side), then two more passes, each on the crop made from the last answer, which also levels the eyes
//! (`ym_ml::mediapipe::facemesh`); then Selfie Multiclass on the crop of the landmarks' box times 1.8, turned like the face.
//! The networks take display-referred values: the linear AWG4 picture is converted to sRGB-encoded Rec.709 for them
//! ([`DisplayView`]); the masks are made on the linear picture.
//!
//! **The cache.** The parts of one face (three parser calls, two mesh calls) share one analysis. Analyses are kept (the last
//! [`CACHE_FACES`]) under a key of the face's rectangle, the picture's size and a hash of all the pixels of the region the
//! analysis can read (the rectangle grown by 1.5 of its size on every side), so a changed picture never gets an old answer.
//!
//! # Determinism
//!
//! The same picture and face give the same bytes on any number of threads: the networks are bit-stable, the crops and masks
//! write disjoint rows, and the hash combines rows in order.

use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use ym_ml::mediapipe::blazeface::PixelSource;
use ym_ml::mediapipe::facemesh::{
    FaceLandmarker, FaceMeshNet, LandmarkerParams, roi_from_landmarks,
};
use ym_ml::mediapipe::selfie::{CLASSES, Class, INPUT_PX, SelfieNet, SelfieSegmenter};
use ym_ml::model::MlError;

use crate::error::{Result, RetouchError};
use crate::face_parts::{
    FaceGeometry, SkinClass, eyes_mask, mesh_table_mask, mouth_interior_mask, skin_mask,
};
use crate::models::{FaceMesh, FaceParser, FacePart, MeshTable};
use crate::types::{BoundedMask, Face, LinearAwg4Image, Mask, Rect};

/// How many analyses are kept.
pub const CACHE_FACES: usize = 8;

/// Linear AWG4 to linear Rec.709 (the sRGB primaries), from `ym_color::gamut::matrices().awg4_to_rec709` (the test
/// `tests/mediapipe.rs::the_display_matrix_is_ym_colors` compares them).
pub const AWG4_TO_REC709: [[f32; 3]; 3] = [
    [1.893_123_4, -0.780_881_5, -0.112_241_94],
    [-0.205_700_36, 1.340_257_5, -0.134_557_13],
    [-0.012_705_743, -0.152_184_87, 1.164_890_6],
];

/// The settings of the analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaPipeParams {
    /// The landmarker's crops and thresholds.
    pub landmarker: LandmarkerParams,
    /// Refinement passes after the first (each on the crop made from the last answer). Two level a face tilted by 30 degrees
    /// (0.84 % of the face's size off after one, 0.16 % after two, `docs/27` S-09).
    pub refinements: usize,
    /// The segmenter's crop: the landmarks' box times this.
    pub segmenter_scale: f64,
}

impl Default for MediaPipeParams {
    fn default() -> Self {
        Self {
            landmarker: LandmarkerParams::default(),
            refinements: 2,
            segmenter_scale: 1.8,
        }
    }
}

/// A linear AWG4 picture seen by the networks: sRGB-encoded Rec.709, `0..1`, clamped.
#[derive(Debug, Clone, Copy)]
pub struct DisplayView<'a> {
    image: &'a LinearAwg4Image,
}

impl<'a> DisplayView<'a> {
    /// Wraps a picture.
    pub fn new(image: &'a LinearAwg4Image) -> Self {
        Self { image }
    }
}

/// The sRGB encoding of a linear value in `0..1`, from a table of 4096 steps (linear below the toe) with linear
/// interpolation: the same on every machine, no `powf`.
fn srgb_encode(v: f32) -> f32 {
    use std::sync::OnceLock;
    const N: usize = 4096;
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    let lut = LUT.get_or_init(|| {
        (0..=N)
            .map(|i| {
                let x = i as f64 / N as f64;
                let e = if x <= 0.003_130_8 {
                    12.92 * x
                } else {
                    1.055 * x.powf(1.0 / 2.4) - 0.055
                };
                e as f32
            })
            .collect()
    });
    let x = v.clamp(0.0, 1.0) * N as f32;
    let i = (x as usize).min(N - 1);
    let t = x - i as f32;
    lut[i] * (1.0 - t) + lut[i + 1] * t
}

impl PixelSource for DisplayView<'_> {
    fn width(&self) -> usize {
        self.image.width_px as usize
    }

    fn height(&self) -> usize {
        self.image.height_px as usize
    }

    fn rgb(&self, x: usize, y: usize) -> [f32; 3] {
        let p = self.image.data[y * self.image.width_px as usize + x];
        let m = &AWG4_TO_REC709;
        let lin = |r: usize| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2];
        [
            srgb_encode(lin(0)),
            srgb_encode(lin(1)),
            srgb_encode(lin(2)),
        ]
    }
}

/// What the networks know about one face.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceAnalysis {
    /// The 478 landmarks in continuous image coordinates.
    pub geometry: FaceGeometry,
    /// The segmenter's face-skin probability on its crop.
    pub skin: SkinClass,
    /// The face-presence score of the landmarker's last pass, `0..1`.
    pub presence: f32,
}

struct Cached {
    key: Key,
    analysis: Option<Arc<FaceAnalysis>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    rect: (i32, i32, u32, u32),
    size: (u32, u32),
    hash: u64,
}

/// The two networks and the cache of analyses; share it with `Arc` between a [`MediaPipeParser`] and a [`MediaPipeMesh`].
pub struct MediaPipeFaces {
    landmarker: FaceLandmarker,
    segmenter: SelfieSegmenter,
    params: MediaPipeParams,
    cache: Mutex<Vec<Cached>>,
}

impl std::fmt::Debug for MediaPipeFaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaPipeFaces")
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

fn model_error(e: &MlError) -> RetouchError {
    RetouchError::Model(e.to_string())
}

/// FNV-1a over the bits of every value of the rectangle `region` of the picture, rows in parallel and combined in order.
fn region_hash(image: &LinearAwg4Image, region: Rect) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let w = image.width_px as usize;
    let rows: Vec<u64> = (region.y_px.max(0) as usize
        ..region.bottom_px().min(i64::from(image.height_px)) as usize)
        .into_par_iter()
        .map(|y| {
            let x0 = region.x_px.max(0) as usize;
            let x1 = region.right_px().min(i64::from(image.width_px)) as usize;
            let mut h = OFFSET ^ y as u64;
            for p in &image.data[y * w + x0..y * w + x1] {
                for v in p {
                    h = (h ^ u64::from(v.to_bits())).wrapping_mul(PRIME);
                }
            }
            h
        })
        .collect();
    rows.iter()
        .fold(OFFSET, |acc, &h| (acc ^ h).wrapping_mul(PRIME))
}

impl MediaPipeFaces {
    /// The networks from a model folder (`data/models/mediapipe/`: `face_mesh_v2.f16.safetensors` and
    /// `selfie_multiclass_256.f32.safetensors`, each checked against `SHA256SUMS`).
    ///
    /// # Errors
    ///
    /// [`RetouchError::Model`] when a file is missing, does not match its checksum or is not the network.
    pub fn load(dir: &std::path::Path) -> Result<Arc<Self>> {
        Self::load_with(dir, MediaPipeParams::default())
    }

    /// As [`MediaPipeFaces::load`], with settings.
    ///
    /// # Errors
    ///
    /// As [`MediaPipeFaces::load`].
    pub fn load_with(dir: &std::path::Path, params: MediaPipeParams) -> Result<Arc<Self>> {
        let mesh = FaceMeshNet::load(dir).map_err(|e| model_error(&e))?;
        let selfie = SelfieNet::load(dir).map_err(|e| model_error(&e))?;
        Ok(Self::from_networks(mesh, selfie, params))
    }

    /// From loaded networks.
    pub fn from_networks(
        mesh: FaceMeshNet,
        selfie: SelfieNet,
        params: MediaPipeParams,
    ) -> Arc<Self> {
        Arc::new(Self {
            landmarker: FaceLandmarker::new(mesh),
            segmenter: SelfieSegmenter::new(selfie),
            params,
            cache: Mutex::new(Vec::new()),
        })
    }

    /// The parser view.
    pub fn parser(self: &Arc<Self>) -> MediaPipeParser {
        MediaPipeParser(Arc::clone(self))
    }

    /// The mesh view.
    pub fn mesh(self: &Arc<Self>) -> MediaPipeMesh {
        MediaPipeMesh(Arc::clone(self))
    }

    /// The analysis of a face: the landmarks and the face-skin probability; `None` when the network sees no face in the
    /// box. Cached (see the module documentation).
    ///
    /// # Errors
    ///
    /// [`RetouchError::Model`] when a network fails, [`RetouchError::Size`] for an inconsistent picture.
    pub fn analyse(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
    ) -> Result<Option<Arc<FaceAnalysis>>> {
        image.validate()?;
        let r = face.rect;
        let long = i64::from(r.width_px.max(r.height_px));
        let grow = long * 3 / 2;
        let region = Rect::new(
            (i64::from(r.x_px) - grow).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
            (i64::from(r.y_px) - grow).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
            (i64::from(r.width_px) + 2 * grow).min(i64::from(u32::MAX)) as u32,
            (i64::from(r.height_px) + 2 * grow).min(i64::from(u32::MAX)) as u32,
        );
        let key = Key {
            rect: (r.x_px, r.y_px, r.width_px, r.height_px),
            size: (image.width_px, image.height_px),
            hash: region_hash(image, region),
        };
        {
            let cache = self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(c) = cache.iter().find(|c| c.key == key) {
                return Ok(c.analysis.clone());
            }
        }
        let analysis = self.compute(image, face)?.map(Arc::new);
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.len() >= CACHE_FACES {
            cache.remove(0);
        }
        cache.push(Cached {
            key,
            analysis: analysis.clone(),
        });
        Ok(analysis)
    }

    fn compute(&self, image: &LinearAwg4Image, face: &Face) -> Result<Option<FaceAnalysis>> {
        let view = DisplayView::new(image);
        let r = face.rect;
        let box_px = [
            f64::from(r.x_px),
            f64::from(r.y_px),
            f64::from(r.width_px),
            f64::from(r.height_px),
        ];
        let Some(found) = self
            .landmarker
            .landmarks_from_box(
                &view,
                box_px,
                self.params.refinements,
                &self.params.landmarker,
            )
            .map_err(|e| model_error(&e))?
        else {
            return Ok(None);
        };
        let Some(roi) = roi_from_landmarks(&found.points_px, self.params.segmenter_scale) else {
            return Ok(None);
        };
        let map = self
            .segmenter
            .segment(&view, &roi, self.params.landmarker.resample)
            .map_err(|e| model_error(&e))?;
        let skin_channel = Class::FaceSkin as usize;
        let probs: Vec<f32> = map
            .probs
            .chunks_exact(CLASSES)
            .map(|p| p[skin_channel])
            .collect();
        // The networks count pixel centres as integers; the masks count them as halves.
        let half = 0.5;
        let Some(geometry) = FaceGeometry::new(
            found
                .points_px
                .iter()
                .map(|p| [f64::from(p[0]) + half, f64::from(p[1]) + half])
                .collect(),
        ) else {
            return Ok(None);
        };
        Ok(Some(FaceAnalysis {
            geometry,
            skin: SkinClass {
                center_px: [roi.center_px[0] + half, roi.center_px[1] + half],
                side_px: roi.side_px,
                rotation_rad: roi.rotation_rad,
                size: INPUT_PX,
                probs,
            },
            presence: found.presence,
        }))
    }
}

/// The parts of a face from Face Mesh V2 and Selfie Multiclass (`docs/27` section 1, S-10).
#[derive(Debug, Clone)]
pub struct MediaPipeParser(Arc<MediaPipeFaces>);

impl FaceParser for MediaPipeParser {
    fn part_mask(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
        part: FacePart,
    ) -> Result<Option<BoundedMask>> {
        let Some(a) = self.0.analyse(image, face)? else {
            return Ok(None);
        };
        let size = (image.width_px, image.height_px);
        match part {
            FacePart::SkinWithoutEars => skin_mask(image, &a.geometry, &a.skin),
            FacePart::Eyes => eyes_mask(&a.geometry, size),
            FacePart::MouthInterior => mouth_interior_mask(&a.geometry, size),
        }
    }
}

/// The two tables of the mesh drawn on the face's 478 landmarks (`docs/27` S-09).
#[derive(Debug, Clone)]
pub struct MediaPipeMesh(Arc<MediaPipeFaces>);

impl FaceMesh for MediaPipeMesh {
    fn render(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
        table: MeshTable,
        size_px: (u32, u32),
    ) -> Result<BoundedMask> {
        match self.0.analyse(image, face)? {
            Some(a) => mesh_table_mask(&a.geometry, table, size_px),
            // No face found: an empty table.
            None => BoundedMask::new(Rect::new(0, 0, 1, 1), Mask::zeros(1, 1)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_srgb_table_is_the_encoding() {
        assert_eq!(srgb_encode(0.0), 0.0);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-6);
        assert!(
            (srgb_encode(0.18) - 0.461_356).abs() < 1e-4,
            "{}",
            srgb_encode(0.18)
        );
        assert!((srgb_encode(0.002) - 12.92 * 0.002).abs() < 1e-5);
        assert_eq!(srgb_encode(-3.0), 0.0);
        assert!((srgb_encode(9.0) - 1.0).abs() < 1e-6);
        // Monotone.
        let mut last = -1.0;
        for i in 0..=1000 {
            let e = srgb_encode(i as f32 / 1000.0);
            assert!(e >= last);
            last = e;
        }
    }

    #[test]
    fn the_display_view_converts_grey_to_the_same_grey_in_every_channel() {
        // AWG4 and Rec.709 share the white point: grey stays grey (the rows of the matrix sum to 1).
        for row in &AWG4_TO_REC709 {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        let image = LinearAwg4Image::filled(2, 2, [0.18; 3]).expect("image");
        let v = DisplayView::new(&image);
        let p = v.rgb(1, 1);
        assert!((p[0] - p[1]).abs() < 1e-4 && (p[1] - p[2]).abs() < 1e-4);
        assert!((p[0] - 0.4614).abs() < 1e-3);
        assert_eq!((v.width(), v.height()), (2, 2));
    }

    #[test]
    fn the_region_hash_sees_every_pixel_of_the_region_and_nothing_outside_it() {
        let mut image = LinearAwg4Image::filled(40, 30, [0.2; 3]).expect("image");
        let region = Rect::new(10, 5, 20, 15);
        let a = region_hash(&image, region);
        assert_eq!(a, region_hash(&image, region));
        image.data[3 * 40 + 3][1] = 0.9; // outside
        assert_eq!(a, region_hash(&image, region));
        image.data[5 * 40 + 10][2] = 0.9; // the first pixel of the region
        let b = region_hash(&image, region);
        assert_ne!(a, b);
        image.data[19 * 40 + 29][0] += 1e-7; // the last pixel of the region
        assert_ne!(b, region_hash(&image, region));
        // A region that hangs over the picture's edge is clipped.
        let _ = region_hash(&image, Rect::new(-5, -5, 100, 100));
    }
}
