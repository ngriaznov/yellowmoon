//! The photo's side of a Reference Grade fit (`docs/34-reference-grade.md` sections 2.2 and 5.1, RG-06).
//!
//! [`Renderer::reference_fit_photo`] renders the photo **once** at the analysis size (512 px on the long edge) on the CPU reference
//! path of a renderer with no device, as Auto's analysis picture is made, finds the seeded sample of 8192 pixels, and runs the plan
//! up to the node under fit for each sample pixel. What it keeps is the fixed input of the search: for each sample pixel its
//! [`PlanState`] at the node's input and its place in the photo frame. The node and every per-pixel step after it, the Output's look,
//! the cine rendering and the display encode run for each candidate ([`ColorChain::codes_from_node`], in `crate::reference_fit`).
//! Spatial nodes (Film, Blur, Face Refinement) are not run in the fit's chain: they make their picture before the colour pass, and
//! a pixel with no such picture passes its input through.

use std::sync::Arc;

use rayon::prelude::*;
use ym_color::display::{ColorChain, OutputTarget, PlanState};
use ym_color::mask::{PhotoFrame, PhotoMap, PhotoSpot, RasterPlanes};
use ym_color::mask_element::{ElementJob, ElementScene, FIT_MIN_SHARE, soft_masks};
use ym_color::mask_raster::Area;
use ym_color::refgrade::measure::PixelScene;
use ym_color::refgrade::{ANALYSIS_LONG_EDGE_PX, sample_pixels};
use ym_gpu::cpu::sampler::GridPositions;
use ym_gpu::queue::{CancelToken, Priority};

use super::RenderError;
use super::color::PhotoSource;
use super::frame::{ViewSpec, sample_view_routed};
use super::geometry::Affine;
use super::graph::{Job, Renderer};
use super::thumbs::{thumbnail_scale, thumbnail_size};
use crate::settings::NodeKind;

/// The soft masks of the seven elements over the analysis picture, for the elements that cover enough of it (`None`: no mask).
#[derive(Debug, Clone)]
pub struct FitMasks {
    /// The analysis picture's width.
    pub width: usize,
    /// Sky, vegetation, water, ground, built, skin, subject: `f32` in 0 to 1, row-major.
    pub planes: [Option<Vec<f32>>; 7],
    node: u32,
}

impl RasterPlanes for FitMasks {
    fn value(&self, _plane: usize, _x: usize, _y: usize) -> f64 {
        0.0
    }

    fn element(&self, node: u32, element: usize, x: usize, y: usize) -> f64 {
        if node != self.node {
            return 0.0;
        }
        match self.planes.get(element) {
            Some(Some(p)) => p.get(y * self.width + x).map_or(0.0, |&v| f64::from(v)),
            _ => 0.0,
        }
    }
}

/// The photo as the fit sees it: the fixed input of the search.
pub struct FitPhoto {
    /// The analysis picture's width in pixels.
    pub width: usize,
    /// Its height.
    pub height: usize,
    /// The chain of the photo's own settings for the sRGB display codes, with the node under fit at a placeholder (every candidate
    /// replaces it).
    pub chain: ColorChain,
    /// The node under fit.
    pub node: u32,
    /// The sample's pixels `(x, y)` of the analysis picture, in sample order.
    pub sample: Vec<(usize, usize)>,
    /// The place of each sample pixel in the photo frame.
    pub spots: Vec<PhotoSpot<'static>>,
    /// Each sample pixel's state at the node's input: the plan's steps before the node, already run.
    pub states: Vec<PlanState>,
    /// The photo's scene at the sample pixels.
    pub scene: PixelScene,
    /// The soft masks of the elements with a mask of at least 2 % of the picture, over the analysis picture.
    pub masks: Arc<FitMasks>,
    /// Which of the seven elements have such a mask (sky ... skin, subject).
    pub mask_ok: [bool; 7],
    /// Whether skin could be looked for (the skin network was there when the scene was made).
    pub skin_available: bool,
}

impl std::fmt::Debug for FitPhoto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitPhoto")
            .field("size", &(self.width, self.height))
            .field("node", &self.node)
            .field("sample", &self.sample.len())
            .finish_non_exhaustive()
    }
}

impl Renderer {
    /// The photo's fixed input for a fit of the Reference Grade node `node` (`docs/34` sections 2.2 and 5.1): the analysis picture at
    /// [`ANALYSIS_LONG_EDGE_PX`], the seeded sample, the plan's steps before the node run once per sample pixel, the photo's scene at
    /// the sample and the soft masks of its elements. The photo is rendered through its own settings, look and Output on the CPU
    /// reference path of a renderer without a device (the path Auto's analysis takes), so the bits are the same on every machine.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when this machine has no scene model or the scene cannot be made; [`RenderError::Color`] when the node
    /// is not a Reference Grade node of the photo or does not take part in its picture (it is not connected to the Output);
    /// [`RenderError::Cancelled`]; the errors of the picture.
    pub fn reference_fit_photo(
        &self,
        src: &PhotoSource,
        node: u32,
        cancel: &CancelToken,
    ) -> Result<FitPhoto, RenderError> {
        self.fit_photo_for(src, node, cancel, false)
    }

    /// The photo as [`Renderer::reference_fit_photo`] makes it, with **every pixel of the analysis picture** in the place of the
    /// seeded sample of 8192: the evaluation tool's input (`docs/34` section 10.1), which measures the full renders of the photo
    /// and of a fit, not the fit's sample. The pictures, the chain, the scene and the masks are the fit's own; only `sample`,
    /// `spots`, `states` and `scene` are longer (row-major, one entry per pixel). It costs a plan run per pixel, so it is made once
    /// per photo.
    ///
    /// # Errors
    ///
    /// The errors of [`Renderer::reference_fit_photo`].
    pub fn reference_eval_photo(
        &self,
        src: &PhotoSource,
        node: u32,
        cancel: &CancelToken,
    ) -> Result<FitPhoto, RenderError> {
        self.fit_photo_for(src, node, cancel, true)
    }

    fn fit_photo_for(
        &self,
        src: &PhotoSource,
        node: u32,
        cancel: &CancelToken,
        every_pixel: bool,
    ) -> Result<FitPhoto, RenderError> {
        if src
            .settings
            .nodes
            .kind_of(crate::settings::NodeId::Corrector(node))
            != Some(NodeKind::ReferenceGrade)
        {
            return Err(RenderError::Color(format!(
                "Node {node} is not a Reference Grade node."
            )));
        }
        // The photo's scene (the Input's picture: nothing a node says changes it), made now when the worker has not made it.
        #[cfg(feature = "eval-scene")]
        let eval = self.eval_photo();
        #[cfg(feature = "eval-scene")]
        let scene = match &eval {
            Some(ev) => ev.scene_for_fit(self, src, cancel)?,
            None => self.scene_of(src, cancel)?,
        };
        #[cfg(not(feature = "eval-scene"))]
        let scene = self.scene_of(src, cancel)?;
        let skin_available = self.scene_model().is_some_and(|m| m.skin_available());
        // The photo as the fit sees its nodes: the node under fit takes part (a hue curve stands in for its candidates) and is not
        // bypassed. Everything else is the person's.
        let mut settings = src.settings.clone();
        if let Some(n) = settings.nodes.reference_grade_mut(node) {
            n.bypass = false;
            n.strength = 100.0;
            n.elements = crate::settings::Elements::default();
            n.hue_curves = crate::settings::HueCurves {
                hue_vs_hue: vec![crate::settings::HuePoint {
                    hue_deg: 0.0,
                    value: 1.0,
                }],
                hue_vs_sat: Vec::new(),
            };
        }
        let placeholder = src.with_settings(settings);
        let renderer = self.snapshot_renderer()?;
        #[cfg(feature = "eval-scene")]
        let result = renderer.fit_photo_on(
            &placeholder,
            node,
            &scene,
            skin_available,
            cancel,
            every_pixel,
            eval.as_ref().zip(self.scene_model()),
        );
        #[cfg(not(feature = "eval-scene"))]
        let result = renderer.fit_photo_on(
            &placeholder,
            node,
            &scene,
            skin_available,
            cancel,
            every_pixel,
        );
        renderer.forget(src.id);
        result
    }

    #[cfg_attr(feature = "eval-scene", allow(clippy::too_many_arguments))]
    fn fit_photo_on(
        &self,
        src: &PhotoSource,
        node: u32,
        scene: &Arc<ElementScene>,
        skin_available: bool,
        cancel: &CancelToken,
        every_pixel: bool,
        #[cfg(feature = "eval-scene")] eval: super::eval_scene::Active<'_>,
    ) -> Result<FitPhoto, RenderError> {
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let image = job.final_size()?;
        let (width, height) = thumbnail_size(image, ANALYSIS_LONG_EDGE_PX);
        let spec = ViewSpec {
            out_w: width,
            out_h: height,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / width as f64,
                image.1 as f64 / height as f64,
                0.0,
                0.0,
            ),
            k: thumbnail_scale(image, ANALYSIS_LONG_EDGE_PX),
            target: OutputTarget::Srgb,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            output: super::frame::FrameOutput::Host,
        };
        let grade = job.grade(OutputTarget::Srgb, false)?;
        let view = sample_view_routed(&mut job, &spec)?;
        if cancel.is_cancelled() {
            return Err(RenderError::Cancelled);
        }
        let chain: ColorChain = (*grade.chain).clone();
        let sampled = &view.sampled;
        let plane = width * height;
        let camera = |i: usize| -> [f64; 3] {
            let d = &sampled.img.data;
            [
                f64::from(d[i]),
                f64::from(d[plane + i]),
                f64::from(d[2 * plane + i]),
            ]
        };
        // The sample, its places in the photo and the plan's steps before the node.
        let sample = if every_pixel {
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .collect()
        } else {
            sample_pixels(width, height)
        };
        let spot_for = |x: usize, y: usize| -> PhotoSpot<'static> {
            view.spot_at(x, y).unwrap_or_else(|| {
                PhotoSpot::new(
                    [
                        (x as f64 + 0.5) / width as f64,
                        (y as f64 + 0.5) / height as f64,
                    ],
                    PhotoFrame::new(width as f64, height as f64, 1.0),
                )
            })
        };
        let spots: Vec<PhotoSpot<'static>> = sample.iter().map(|&(x, y)| spot_for(x, y)).collect();
        let states: Vec<Option<PlanState>> = sample
            .par_iter()
            .zip(spots.par_iter())
            .map(|(&(x, y), spot)| {
                chain.state_before_node(camera(y * width + x), Some(*spot), node)
            })
            .collect();
        let states: Option<Vec<PlanState>> = states.into_iter().collect();
        let states = states.ok_or_else(|| {
            RenderError::Color(
                "This Reference Grade node is not connected to the Output, so it does not change the picture.".to_owned(),
            )
        })?;
        // EVALUATION ONLY: the photo's default render through the look gives the guide (and, with labels, the scene).
        #[cfg(feature = "eval-scene")]
        let (scene, guide_override) = match &eval {
            Some((ev, model)) => {
                let base = src
                    .settings
                    .nodes
                    .reference_grade(node)
                    .ok_or_else(|| RenderError::Color("The node is gone.".to_owned()))?;
                let (made, guide) = ev.prepare(
                    model,
                    &chain,
                    base,
                    node,
                    src.id,
                    (width, height),
                    &camera,
                    &spot_for,
                )?;
                (made.unwrap_or_else(|| Arc::clone(scene)), Some(guide))
            }
            None => (Arc::clone(scene), None),
        };
        #[cfg(feature = "eval-scene")]
        let scene = &scene;
        // The photo's scene at the sample pixels, by the place of each pixel in the photo.
        let mut ps = PixelScene::blank(sample.len());
        for (i, spot) in spots.iter().enumerate() {
            let (class, skin, subject) = scene.class_at(spot.pos);
            ps.class[i] = class;
            ps.skin[i] = skin;
            ps.subject[i] = subject;
        }
        // The soft masks of the elements that cover enough of the picture: the guided filter on the L*/100 of the Input's picture.
        let covering = scene.covering(FIT_MIN_SHARE);
        let mut planes: [Option<Vec<f32>>; 7] = Default::default();
        if let (true, Some(grid)) = (covering.iter().any(|&c| c), view.grid.as_ref()) {
            let lightness: Vec<f32> = (0..plane)
                .into_par_iter()
                .map(|i| chain.scene_guide_pixel(camera(i)))
                .collect();
            #[cfg(feature = "eval-scene")]
            let lightness: Vec<f32> = guide_override.map_or(lightness, |g| (*g).clone());
            let map = GridPositions::new(grid, (0, 0));
            let outside = sampled.outside.as_deref();
            let position = |x: usize, y: usize| {
                if outside.is_some_and(|o| o[y * width + x] != 0) {
                    None
                } else {
                    Some(map.position(x, y))
                }
            };
            let whole = Area::whole(width, height);
            planes = soft_masks(
                scene,
                &covering,
                &ElementJob {
                    photo: grid.photo_frame(),
                    grown: whole,
                    frame: whole,
                    window: whole,
                    position: &position,
                    guide: &lightness,
                },
            );
        }
        let mask_ok = std::array::from_fn(|e| covering[e] && planes[e].is_some());
        // The class of the photo's own element at the sample is the scene's; an element with no mask has no local grade.
        Ok(FitPhoto {
            width,
            height,
            chain,
            node,
            sample,
            spots,
            states,
            scene: ps,
            masks: Arc::new(FitMasks {
                width,
                planes,
                node,
            }),
            mask_ok,
            skin_available,
        })
    }
}
