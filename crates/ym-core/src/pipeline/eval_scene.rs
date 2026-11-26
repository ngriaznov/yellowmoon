//! EVALUATION ONLY: another scene for a Reference Grade fit (`docs/34-reference-grade.md` section 16.10, the RG-10 ablation).
//!
//! This module exists only with the `eval-scene` cargo feature, which only `tools/eval/refgrade` turns on; it is `doc(hidden)` and
//! nothing in the app, in `ym` or in the product's own code calls it. A build of the app or of `ym` does not contain it.
//!
//! It answers one question: how much of the Rust node's loss of cast transfer against the proof of concept's round 15 is the
//! **scene** (the label map, the guide of the soft masks) and how much is the fit itself. It changes three things, all on the
//! evaluation's own request and never in a render the person sees:
//!
//! * **The photo's labels** ([`PhotoScene::Labels`], variant A1): the photo's element map is the proof of concept's cached label map
//!   (made from the photo's default render through the look), carried to the analysis picture by nearest neighbour; the subject comes
//!   from the same rule and skin from MediaPipe inside `person` on the default render, as the proof of concept made them.
//! * **The segmenter on the default render** ([`PhotoScene::DefaultRender`], variant A2): the product's own segmenter, run on the
//!   photo's default render through the look (whole, 1024 px, 8 bits) in place of the Input's picture.
//! * **The guide**: in both, the guide of the soft masks is the L\*/100 of the photo's default render through the look at the
//!   analysis picture (the render with the node at the identity: the picture the proof of concept's guide was made of).
//! * **The reference's labels** ([`EvalLabels`], A1 only): the reference's element map is the proof of concept's label map, cut by the
//!   reference's black bars as the product cut them and carried to the analysis picture.

use std::sync::{Arc, Mutex, PoisonError};

use rayon::prelude::*;
use ym_color::display::ColorChain;
use ym_color::mask::{NodeMasks, PhotoSpot};
use ym_color::mask_element::ElementScene;
use ym_color::refgrade::N_PARAMS;
use ym_color::refgrade::measure::LabConv;
use ym_color::refgrade::params::to_numbers;
use ym_gpu::queue::CancelToken;

use super::RenderError;
use super::color::PhotoSource;
use super::graph::{ReferenceScene, Renderer};
use super::scene::SceneModel;
use crate::reference::ReferencePicture;
use crate::reference_fit::apply_numbers;
use crate::session::PhotoId;
use crate::settings::ReferenceGradeNode;

/// An element map made elsewhere: the proof of concept's label map of a picture. `elements` holds the segmenter's element number of
/// every pixel (unlabelled 0, sky 1, vegetation 2, water 3, ground 4, architecture 5, interior 6, person 7, other thing 8, other
/// stuff 9), rows top to bottom.
#[derive(Debug, Clone)]
pub struct EvalLabels {
    /// Width of the map in pixels.
    pub width: usize,
    /// Height of the map in pixels.
    pub height: usize,
    /// `width * height` element numbers.
    pub elements: Vec<u8>,
    /// What to cut off the map's edges before it is carried to a grid: top, bottom, left, right, in the map's pixels (a reference's
    /// black bars; all 0 for a photo).
    pub crop: [usize; 4],
}

impl EvalLabels {
    /// The map on a `width x height` grid by nearest neighbour after the crop: the centre of each grid pixel, as the proof of concept's
    /// `resize_labels` took it.
    #[must_use]
    pub fn on_grid(&self, width: usize, height: usize) -> Vec<u8> {
        let (sw, sh) = (self.width, self.height);
        let (cw, ch) = (
            sw.saturating_sub(self.crop[2] + self.crop[3]).max(1),
            sh.saturating_sub(self.crop[0] + self.crop[1]).max(1),
        );
        let mut out = vec![0u8; width * height];
        for y in 0..height {
            let sy = self.crop[0] + (((y as f64 + 0.5) / height as f64) * ch as f64) as usize;
            for x in 0..width {
                let sx = self.crop[2] + (((x as f64 + 0.5) / width as f64) * cw as f64) as usize;
                out[y * width + x] = self.elements[sy.min(sh - 1) * sw + sx.min(sw - 1)];
            }
        }
        out
    }
}

/// How the photo's scene is replaced.
#[derive(Debug, Clone)]
pub enum PhotoScene {
    /// The photo's element map is this label map.
    Labels(EvalLabels),
    /// The segmenter runs on the photo's default render through the look.
    DefaultRender,
}

/// What a photo's default render gives the evaluation: its codes as 8 bits and the guide of the soft masks.
struct Prepared {
    key: (PhotoId, usize, usize),
    /// The label scene (variant A1), made once.
    scene: Option<Arc<ElementScene>>,
    /// L\*/100 of every pixel of the analysis picture, row-major.
    guide: Arc<Vec<f32>>,
}

/// The photo's override and what it has made for the photo being fitted.
pub struct PhotoOverride {
    scene: PhotoScene,
    prepared: Mutex<Option<Prepared>>,
    /// The scene of variant A2 (the segmenter on the default render), made once for the photo.
    segmented: Mutex<Option<(PhotoId, Arc<ElementScene>)>>,
}

impl std::fmt::Debug for PhotoOverride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotoOverride").finish_non_exhaustive()
    }
}

/// What the renderer holds for the evaluation.
#[derive(Default)]
pub struct Slots {
    photo: Option<Arc<PhotoOverride>>,
    reference: Option<Arc<EvalLabels>>,
}

/// The scene to use in place of the fit's (`None`: keep it) and the guide of the soft masks.
pub(crate) type PreparedScene = (Option<Arc<ElementScene>>, Arc<Vec<f32>>);

/// What `fit_photo_on` is given when the evaluation changed the scene: the override and the scene model that makes skin.
pub(crate) type Active<'a> = Option<(&'a Arc<PhotoOverride>, Arc<SceneModel>)>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// An `ElementScene` from the scene model's bits (the key is a fixed number: an evaluation scene is never compared by key).
fn element_scene(bits: ym_ml::scene::SceneBits, key: u64) -> ElementScene {
    ElementScene {
        key,
        width: bits.width,
        height: bits.height,
        bits: bits.bits,
        classes: bits.classes,
    }
}

impl Renderer {
    /// EVALUATION ONLY: replaces the scene of the photos this renderer fits from now on (`None`: the product's own). The photo's
    /// default render and what is made from it are kept per photo, so set it again for each photo.
    pub fn set_eval_photo_scene(&self, scene: Option<PhotoScene>) {
        lock(self.eval_slots()).photo = scene.map(|scene| {
            Arc::new(PhotoOverride {
                scene,
                prepared: Mutex::new(None),
                segmented: Mutex::new(None),
            })
        });
    }

    /// EVALUATION ONLY: the label map the next reference scenes are made from (`None`: the product's own segmenter).
    pub fn set_eval_reference_labels(&self, labels: Option<EvalLabels>) {
        lock(self.eval_slots()).reference = labels.map(Arc::new);
    }

    pub(crate) fn eval_photo(&self) -> Option<Arc<PhotoOverride>> {
        lock(self.eval_slots()).photo.clone()
    }

    /// The scene of a reference picture made from the evaluation's label map, when one is set: the map cut by the bars and carried
    /// to the analysis picture, skin by MediaPipe on that picture.
    pub(crate) fn eval_reference_scene(
        &self,
        model: &SceneModel,
        picture: &ReferencePicture,
    ) -> Result<Option<ReferenceScene>, RenderError> {
        let Some(labels) = lock(self.eval_slots()).reference.clone() else {
            return Ok(None);
        };
        let a = picture.analysis();
        let elements = labels.on_grid(a.width, a.height);
        let rgb = a.srgb8();
        let bits = model.scene_from_elements(elements, &rgb, a.width, a.height)?;
        let skin_available = bits.skin_available;
        Ok(Some(ReferenceScene {
            scene: Arc::new(element_scene(bits, 0xE7A1_0002)),
            skin_available,
        }))
    }
}

impl PhotoOverride {
    /// The scene the fit starts from: the segmenter on the default render (A2), or an empty one that [`PhotoOverride::prepare`]
    /// replaces (A1).
    pub(crate) fn scene_for_fit(
        &self,
        renderer: &Renderer,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Arc<ElementScene>, RenderError> {
        match &self.scene {
            PhotoScene::Labels(_) => Ok(Arc::new(ElementScene {
                key: 0,
                width: 0,
                height: 0,
                bits: Vec::new(),
                classes: Vec::new(),
            })),
            PhotoScene::DefaultRender => {
                if let Some((id, s)) = lock(&self.segmented).as_ref()
                    && *id == src.id
                {
                    return Ok(Arc::clone(s));
                }
                let model = renderer.scene_model().ok_or_else(|| {
                    RenderError::Mask(super::scene::NO_SCENE_MODEL_SENTENCE.to_owned())
                })?;
                let picture = renderer.eval_default_picture(src, cancel)?;
                let bits = model.scene_of(&picture.rgb, picture.width, picture.height, &|| {
                    cancel.is_cancelled()
                })?;
                let made = Arc::new(element_scene(bits, 0xE7A1_0003));
                *lock(&self.segmented) = Some((src.id, Arc::clone(&made)));
                Ok(made)
            }
        }
    }

    /// The photo's default render through the look at the analysis picture (the node at the identity), made once per photo: the
    /// guide of the soft masks, and for A1 the label scene (skin by MediaPipe on that render). Returns the scene to use in place of
    /// the fit's (`None`: keep it) and the guide.
    ///
    /// `camera` and `spot_for` give a pixel's camera values and its place in the photo frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &self,
        model: &SceneModel,
        chain: &ColorChain,
        base: &ReferenceGradeNode,
        node: u32,
        photo: PhotoId,
        (width, height): (usize, usize),
        camera: &(dyn Fn(usize) -> [f64; 3] + Sync),
        spot_for: &(dyn Fn(usize, usize) -> PhotoSpot<'static> + Sync),
    ) -> Result<PreparedScene, RenderError> {
        let key = (photo, width, height);
        if let Some(p) = lock(&self.prepared).as_ref()
            && p.key == key
        {
            return Ok((p.scene.clone(), Arc::clone(&p.guide)));
        }
        // The chain with the node at the identity: the photo's default render through its look.
        let mut identity = base.clone();
        identity.bypass = false;
        identity.strength = 100.0;
        apply_numbers(&mut identity, &to_numbers(&[0.0; N_PARAMS]));
        let params = super::color::reference_node_params(&identity, NodeMasks::default())?;
        let mut chain = chain.clone();
        chain
            .replace_node(node, &params)
            .map_err(|e| RenderError::Color(e.to_string()))?;
        let conv = LabConv::new();
        let codes: Vec<[f64; 3]> = (0..width * height)
            .into_par_iter()
            .map(|i| {
                let spot = spot_for(i % width, i / width);
                chain
                    .state_before_node(camera(i), Some(spot), node)
                    .and_then(|s| chain.codes_from_node(&s, Some(spot), node))
                    .unwrap_or([0.0; 3])
            })
            .collect();
        let guide: Vec<f32> = codes
            .par_iter()
            .map(|&c| (conv.px_of_codes(c).lab[0] / 100.0).clamp(0.0, 1.0) as f32)
            .collect();
        let scene = match &self.scene {
            PhotoScene::Labels(labels) => {
                let rgb: Vec<u8> = codes
                    .iter()
                    .flat_map(|c| c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
                    .collect();
                let elements = labels.on_grid(width, height);
                let bits = model.scene_from_elements(elements, &rgb, width, height)?;
                Some(Arc::new(element_scene(bits, 0xE7A1_0001)))
            }
            PhotoScene::DefaultRender => None,
        };
        let guide = Arc::new(guide);
        *lock(&self.prepared) = Some(Prepared {
            key,
            scene: scene.clone(),
            guide: Arc::clone(&guide),
        });
        Ok((scene, guide))
    }
}
