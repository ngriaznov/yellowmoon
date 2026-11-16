//! The teeth segmenter of the Face Refinement node: SAM 2.1 small prompted from the face mesh (`docs/27` 5.6, S-17).
//!
//! [`SamTeeth`] implements `ym_retouch::TeethSegmenter`. For a face with an open mouth it
//!
//! 1. takes the 478 landmarks of the face's analysis (`ym_retouch::mediapipe::MediaPipeFaces::analyse`, cached);
//! 2. cuts a square **crop around the mouth** ([`MouthCrop`], [`CROP_FACTOR`] times the mouth's width) out of the analysis
//!    picture, as sRGB-encoded Rec.709 8 bits, and encodes it with SAM (the encoder resizes it to 1024 px);
//! 3. prompts the decoder from the mesh ([`teeth_prompt_with`]: the opening's box, positive points in it, negative points on the
//!    lips) and chooses among the four masks it answers ([`select_candidate`]);
//! 4. gives the chosen mask back on the picture ([`crop_mask`]).
//!
//! When the weights are missing or damaged, when SAM answers nothing usable or when the mesh gives no mouth, the answer is
//! `None` and the log says why; the caller turns teeth whitening off for the face. No colour is looked at anywhere.
//!
//! The answer is cached (the last [`CACHE_FACES`] faces) under a key of the face's rectangle, the picture's size and a hash of
//! the pixels the crop reads, so a changed picture never gets an old answer; the encoder runs once per analysis (about 3 to 4
//! s per face on the CPU, `docs/27` S-17 as built).

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use ym_ml::mediapipe::blazeface::PixelSource;
use ym_ml::sam2::{Decoded, Features, Picture, Point, Prompt};
use ym_retouch::mediapipe::{DisplayView, FaceAnalysis, MediaPipeFaces, region_hash};
use ym_retouch::teeth::{
    CROP_FACTOR, CandidateStats, GRID_PX, MouthCrop, PromptShape, PromptStyle, Refusal, Regions,
    TeethPrompt, crop_mask, select_candidate, teeth_prompt_with,
};
use ym_retouch::{
    BoundedMask, Face, LinearAwg4Image, Rect, Result as RetouchResult, RetouchError, TeethSegmenter,
};

use super::object::ObjectModel;

/// How many answers are kept.
pub const CACHE_FACES: usize = 8;
/// The prompt the segmenter uses (S-17 compared the three; the choice is in `docs/27` 5.6).
pub const PROMPT_STYLE: PromptStyle = PromptStyle::Points;
/// The most pixels of a side of the crop handed to SAM (a larger crop is resampled down; SAM works at 1024).
pub const CROP_MAX_PX: usize = 1024;
/// The fewest pixels of a side of the crop handed to SAM.
pub const CROP_MIN_PX: usize = 32;

/// The mouth of a face, encoded by SAM and ready to be prompted.
pub struct EncodedMouth {
    /// The crop, in picture pixels.
    pub crop: MouthCrop,
    /// The mouth's regions on SAM's grid.
    pub regions: Regions,
    /// The analysis the mouth was taken from.
    pub analysis: Arc<FaceAnalysis>,
    features: Features,
    /// How long the crop and the encoder took, in milliseconds.
    pub encode_ms: f64,
}

/// What SAM answered for one prompt, and which mask was chosen.
pub struct Decision {
    /// The prompt that was given.
    pub prompt: TeethPrompt,
    /// The four masks and their predicted IoUs (the single-mask token first).
    pub decoded: Decoded,
    /// The index of the chosen candidate, or why none was.
    pub chosen: std::result::Result<usize, Refusal>,
    /// What every candidate does on the regions.
    pub stats: Vec<CandidateStats>,
}

impl Decision {
    /// The chosen candidate's logits on SAM's grid.
    #[must_use]
    pub fn logits(&self) -> Option<&[f32]> {
        self.chosen.ok().map(|i| self.decoded.candidate_logits(i))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    rect: (i32, i32, u32, u32),
    size: (u32, u32),
    hash: u64,
}

/// The teeth segmenter: SAM 2.1 small and the face analysis of the same picture.
pub struct SamTeeth {
    faces: Arc<MediaPipeFaces>,
    sam: Arc<ObjectModel>,
    style: PromptStyle,
    factor: f64,
    cache: Mutex<Vec<(Key, Option<BoundedMask>)>>,
}

impl std::fmt::Debug for SamTeeth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamTeeth")
            .field("style", &self.style)
            .field("factor", &self.factor)
            .finish_non_exhaustive()
    }
}

fn model_error(e: impl std::fmt::Display) -> RetouchError {
    RetouchError::Model(e.to_string())
}

/// The sRGB 8-bit crop of `crop` out of `image`, `side x side` pixels, sampled bilinearly (a crop that hangs over the picture's
/// edge repeats the edge).
fn crop_rgb8(image: &LinearAwg4Image, crop: &MouthCrop, side: usize) -> Vec<u8> {
    let view = DisplayView::new(image);
    let (w, h) = (view.width() as i64, view.height() as i64);
    let step = crop.side_px / side as f64;
    let at = |x: i64, y: i64| view.rgb(x.clamp(0, w - 1) as usize, y.clamp(0, h - 1) as usize);
    let mut out = vec![0u8; side * side * 3];
    for (j, row) in out.chunks_mut(side * 3).enumerate() {
        // The pixel's centre in picture coordinates, then the four neighbours (pixel centres are at the halves).
        let y = crop.y_px + (j as f64 + 0.5) * step - 0.5;
        let (y0, fy) = (y.floor(), (y - y.floor()) as f32);
        for (i, px) in row.chunks_exact_mut(3).enumerate() {
            let x = crop.x_px + (i as f64 + 0.5) * step - 0.5;
            let (x0, fx) = (x.floor(), (x - x.floor()) as f32);
            let (xi, yi) = (x0 as i64, y0 as i64);
            let (a, b, c, d) = (
                at(xi, yi),
                at(xi + 1, yi),
                at(xi, yi + 1),
                at(xi + 1, yi + 1),
            );
            for k in 0..3 {
                let top = a[k] * (1.0 - fx) + b[k] * fx;
                let bottom = c[k] * (1.0 - fx) + d[k] * fx;
                let v = top * (1.0 - fy) + bottom * fy;
                px[k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

/// A SAM prompt from the segmenter's: the same numbers.
fn sam_prompt(p: &TeethPrompt) -> Prompt {
    Prompt {
        points: p
            .points
            .iter()
            .map(|q| Point {
                x: q.x,
                y: q.y,
                positive: q.positive,
            })
            .collect(),
        bbox: p.bbox,
    }
}

impl SamTeeth {
    /// A segmenter on a face analysis and SAM, with the crop and prompt the spec fixes.
    #[must_use]
    pub fn new(faces: Arc<MediaPipeFaces>, sam: Arc<ObjectModel>) -> Self {
        Self::with(faces, sam, PROMPT_STYLE, CROP_FACTOR)
    }

    /// As [`SamTeeth::new`], with another prompt style and crop size (the evaluation of S-17).
    #[must_use]
    pub fn with(
        faces: Arc<MediaPipeFaces>,
        sam: Arc<ObjectModel>,
        style: PromptStyle,
        factor: f64,
    ) -> Self {
        Self {
            faces,
            sam,
            style,
            factor,
            cache: Mutex::new(Vec::new()),
        }
    }

    /// Cuts the crop around the mouth of `face` and encodes it. `None` when the mesh finds no face or no mouth, or the mouth is
    /// closed.
    ///
    /// # Errors
    ///
    /// [`RetouchError::Model`] when the weights cannot be read or a network fails.
    pub fn encode(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
        factor: f64,
    ) -> RetouchResult<Option<EncodedMouth>> {
        let Some(analysis) = self.faces.analyse(image, face)? else {
            return Ok(None);
        };
        let Some(crop) = MouthCrop::from_geometry(&analysis.geometry, factor) else {
            return Ok(None);
        };
        let Some(regions) = Regions::from_geometry(&analysis.geometry, &crop) else {
            return Ok(None);
        };
        let started = Instant::now();
        let sam = self.sam.sam().map_err(model_error)?;
        let side = (crop.side_px.round() as usize).clamp(CROP_MIN_PX, CROP_MAX_PX);
        let rgb = crop_rgb8(image, &crop, side);
        let picture = Picture::new(side, side, &rgb).map_err(model_error)?;
        let features = self.sam.run(|| sam.encode(&picture)).map_err(model_error)?;
        Ok(Some(EncodedMouth {
            crop,
            regions,
            analysis,
            features,
            encode_ms: started.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    /// Prompts an encoded mouth in `style` and chooses a mask.
    ///
    /// # Errors
    ///
    /// [`RetouchError::Model`] when the decoder fails; a mesh that gives no prompt is [`Refusal::Nothing`].
    pub fn decide(
        &self,
        mouth: &EncodedMouth,
        style: PromptStyle,
    ) -> RetouchResult<Option<Decision>> {
        self.decide_with(mouth, style, &PromptShape::DEFAULT)
    }

    /// [`SamTeeth::decide`] with another [`PromptShape`].
    ///
    /// # Errors
    ///
    /// As [`SamTeeth::decide`].
    pub fn decide_with(
        &self,
        mouth: &EncodedMouth,
        style: PromptStyle,
        shape: &PromptShape,
    ) -> RetouchResult<Option<Decision>> {
        let Some(prompt) = teeth_prompt_with(&mouth.analysis.geometry, &mouth.crop, style, shape)
        else {
            return Ok(None);
        };
        let sam = self.sam.sam().map_err(model_error)?;
        let decoded = self
            .sam
            .run(|| sam.decode_candidates(&mouth.features, &sam_prompt(&prompt)))
            .map_err(model_error)?;
        let candidates: Vec<(&[f32], f32)> = (0..decoded.candidates())
            .map(|i| (decoded.candidate_logits(i), decoded.iou[i]))
            .collect();
        debug_assert!(candidates.iter().all(|(l, _)| l.len() == GRID_PX * GRID_PX));
        let (chosen, stats) = select_candidate(&candidates, &mouth.regions);
        Ok(Some(Decision {
            prompt,
            decoded,
            chosen,
            stats,
        }))
    }

    fn segment(&self, image: &LinearAwg4Image, face: &Face) -> RetouchResult<Option<BoundedMask>> {
        let Some(mouth) = self.encode(image, face, self.factor)? else {
            tracing::info!("the mesh gives no open mouth; no teeth");
            return Ok(None);
        };
        let Some(decision) = self.decide(&mouth, self.style)? else {
            tracing::info!("the mesh gives no prompt for the teeth; no teeth");
            return Ok(None);
        };
        let Ok(chosen) = decision.chosen else {
            if let Err(why) = decision.chosen {
                tracing::info!(reason = %why, "no teeth found; teeth whitening is off for this face");
            }
            return Ok(None);
        };
        tracing::debug!(
            encode_ms = mouth.encode_ms,
            candidate = chosen,
            iou = decision.stats[chosen].iou,
            cover = decision.stats[chosen].cover,
            spill = decision.stats[chosen].spill,
            "the teeth"
        );
        crop_mask(
            decision.decoded.candidate_logits(chosen),
            &mouth.crop,
            (image.width_px, image.height_px),
        )
    }

    fn key(&self, image: &LinearAwg4Image, face: &Face) -> Key {
        // The crop lies inside the face's rectangle grown by its own size on every side (a mouth is inside the face and the
        // crop is a few mouth widths).
        let r = face.rect;
        let long = i64::from(r.width_px.max(r.height_px));
        let clamp = |v: i64| v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let region = Rect::new(
            clamp(i64::from(r.x_px) - long),
            clamp(i64::from(r.y_px) - long),
            (i64::from(r.width_px) + 2 * long).min(i64::from(u32::MAX)) as u32,
            (i64::from(r.height_px) + 2 * long).min(i64::from(u32::MAX)) as u32,
        );
        Key {
            rect: (r.x_px, r.y_px, r.width_px, r.height_px),
            size: (image.width_px, image.height_px),
            hash: region_hash(image, region),
        }
    }
}

impl TeethSegmenter for SamTeeth {
    fn teeth(&self, image: &LinearAwg4Image, face: &Face) -> RetouchResult<Option<BoundedMask>> {
        image.validate()?;
        let key = self.key(image, face);
        {
            let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some((_, answer)) = cache.iter().find(|(k, _)| *k == key) {
                return Ok(answer.clone());
            }
        }
        let answer = self.segment(image, face)?;
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.len() >= CACHE_FACES {
            cache.remove(0);
        }
        cache.push((key, answer.clone()));
        Ok(answer)
    }
}

/// A [`SamTeeth`] that many analyses share (its cache is kept between them).
#[derive(Debug, Clone)]
pub struct SharedTeeth(pub Arc<SamTeeth>);

impl TeethSegmenter for SharedTeeth {
    fn teeth(&self, image: &LinearAwg4Image, face: &Face) -> RetouchResult<Option<BoundedMask>> {
        self.0.teeth(image, face)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crop_is_the_picture_resampled_and_repeats_the_edge() {
        // A picture whose red grows to the right: the crop shows the same ramp, and the part outside the picture repeats it.
        let (w, h) = (40usize, 30usize);
        let data: Vec<[f32; 3]> = (0..w * h)
            .map(|i| [(i % w) as f32 / (w - 1) as f32 * 0.5, 0.1, 0.1])
            .collect();
        let image = LinearAwg4Image::new(w as u32, h as u32, data).expect("image");
        let crop = MouthCrop {
            x_px: -10.0,
            y_px: -10.0,
            side_px: 60.0,
        };
        let rgb = crop_rgb8(&image, &crop, 60);
        let px = |x: usize, y: usize| rgb[(y * 60 + x) * 3];
        // Outside on the left and above: the edge column's value; inside: growing with x.
        assert_eq!(px(0, 0), px(5, 40));
        assert!(px(30, 30) > px(15, 30));
        assert!(px(59, 30) >= px(45, 30));
        // The same crop at another resolution has the same ramp.
        let small = crop_rgb8(&image, &crop, 30);
        assert!((i32::from(small[(15 * 30 + 15) * 3]) - i32::from(px(30, 30))).abs() <= 3);
    }

    #[test]
    fn a_prompt_is_converted_number_for_number() {
        let p = TeethPrompt {
            bbox: Some([0.1, 0.2, 0.3, 0.4]),
            points: vec![ym_retouch::teeth::PromptPoint {
                x: 0.5,
                y: 0.6,
                positive: false,
            }],
        };
        let s = sam_prompt(&p);
        assert_eq!(s.bbox, p.bbox);
        assert_eq!(
            (s.points[0].x, s.points[0].y, s.points[0].positive),
            (0.5, 0.6, false)
        );
    }
}
