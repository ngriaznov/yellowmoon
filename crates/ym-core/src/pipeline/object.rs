//! Object masks (`docs/23-nodes.md` section 4.4, `docs/algorithms/06-masks.md` section 6): the SAM 2.1 model the core
//! holds, the prompts a click or a box makes, the answers of `mask_find`, and the planes the model makes for the rows
//! of a photo.
//!
//! The model ([`ObjectModel`]) is `ym_ml::sam2` with its weights, loaded the first time something needs it. The
//! renderer keeps one ([`RenderOptions::object_model`](super::RenderOptions)); `mask_kinds` offers Object when it is
//! there. What runs when a photo has an Object row is in `graph/object.rs`: the encoder in the background, the features
//! the photo keeps, `mask_find` and the source of the row's raster `S0`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ym_color::mask_raster::SourcePlane;
use ym_ml::sam2::{self, Sam2};

use super::RenderError;
use super::color::PhotoSource;
use crate::settings::{
    DevelopSettings, MAX_OBJECT_POINTS, NO_OBJECT_MODEL_SENTENCE, NodeGrade, ObjectMask,
};
use crate::vision::VisionKind;
use ym_looks::library::LookId;

/// Coordinates the core stores for an Object row are rounded to this many decimals (a tenth of a pixel of SAM's input),
/// so a document stays short and the stored points are exactly the ones the decoder was given.
pub const OBJECT_POINT_DECIMALS: i32 = 4;

/// The long edge of the picture SAM looks at, in pixels (`docs/algorithms/06` section 6.2).
pub const OBJECT_PICTURE_PX: usize = sam2::IMAGE_PX;

/// The message of the sentence the person reads when "Nothing found" ends a click (`docs/23` section 4.4). The page
/// shows it in the error bar; the core owns the words so the app and the mock say the same.
pub const NOTHING_FOUND_SENTENCE: &str =
    "Nothing found there. Click another part of the object, or draw a box.";

/// What the encoder is doing for a photo, as the renderer's listener hears it ([`Renderer::set_object_listener`]).
///
/// [`Renderer::set_object_listener`]: super::Renderer::set_object_listener
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectPhase {
    /// The photo's features are being made, or will be in a moment: the row says "Getting ready…".
    Preparing,
    /// The encoder is not working for the photo. A frame that waited for the features should be drawn again.
    Idle,
}

/// SAM 2.1 small and where its weights are. The weights load the first time they are asked for
/// ([`ObjectModel::sam`]); a file that is damaged takes the model away for the rest of the run
/// ([`ObjectModel::available`]).
pub struct ObjectModel {
    dir: Option<PathBuf>,
    state: Mutex<ModelState>,
    pool: OnceLock<Option<rayon::ThreadPool>>,
}

enum ModelState {
    /// The weights have not been read yet.
    Untried,
    /// Loaded.
    Ready(Arc<Sam2>),
    /// Could not be read or did not match their checksum (the reason is in the log).
    Damaged,
}

impl std::fmt::Debug for ObjectModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectModel")
            .field("dir", &self.dir)
            .field("available", &self.available())
            .finish_non_exhaustive()
    }
}

impl ObjectModel {
    /// The model of a data folder (`<data_root>/models/sam2/`), when its weights file and its `SHA256SUMS` are there.
    /// Nothing is read: the weights load, and are checked against their checksum, on first use.
    #[must_use]
    pub fn probe(data_root: &Path) -> Option<Arc<Self>> {
        let dir = data_root.join("models").join("sam2");
        (dir.join(sam2::WEIGHTS_FILE).is_file() && dir.join("SHA256SUMS").is_file()).then(|| {
            Arc::new(Self {
                dir: Some(dir),
                state: Mutex::new(ModelState::Untried),
                pool: OnceLock::new(),
            })
        })
    }

    /// A model that is loaded already (tests).
    #[must_use]
    pub fn from_sam(sam: Arc<Sam2>) -> Arc<Self> {
        Arc::new(Self {
            dir: None,
            state: Mutex::new(ModelState::Ready(sam)),
            pool: OnceLock::new(),
        })
    }

    /// Whether Object can be offered: the weights are there and have not been found damaged.
    #[must_use]
    pub fn available(&self) -> bool {
        !matches!(
            *self.state.lock().unwrap_or_else(PoisonError::into_inner),
            ModelState::Damaged
        )
    }

    /// The model, loading the weights when this is the first call.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] with [`NO_OBJECT_MODEL_SENTENCE`] when the weights cannot be read or do not match their
    /// checksum (the reason is logged, and the model is not [`available`](Self::available) from then on).
    pub fn sam(&self) -> Result<Arc<Sam2>, RenderError> {
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match &*st {
            ModelState::Ready(sam) => return Ok(Arc::clone(sam)),
            ModelState::Damaged => {
                return Err(RenderError::Mask(NO_OBJECT_MODEL_SENTENCE.to_owned()));
            }
            ModelState::Untried => {}
        }
        let started = std::time::Instant::now();
        let loaded = match &self.dir {
            Some(dir) => Sam2::load(dir),
            None => Err(ym_ml::model::MlError::Weights("no folder".to_owned())),
        };
        match loaded {
            Ok(sam) => {
                tracing::debug!(
                    ms = started.elapsed().as_secs_f64() * 1000.0,
                    "the Object model loaded"
                );
                let sam = Arc::new(sam);
                *st = ModelState::Ready(Arc::clone(&sam));
                Ok(sam)
            }
            Err(e) => {
                tracing::error!(error = %e, "the Object model could not be loaded");
                *st = ModelState::Damaged;
                Err(RenderError::Mask(NO_OBJECT_MODEL_SENTENCE.to_owned()))
            }
        }
    }

    /// Runs `f` on the threads this model keeps for the encoder: all the cores but one, so that a photo being edited
    /// stays smooth while the encoder works (the bits do not depend on the number of threads,
    /// `docs/algorithms/06` section 6.4).
    pub fn run<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        let pool = self.pool.get_or_init(|| {
            let cores = std::thread::available_parallelism().map_or(1, usize::from);
            rayon::ThreadPoolBuilder::new()
                .num_threads(cores.saturating_sub(1).max(1))
                .thread_name(|i| format!("yellowmoon-sam-{i}"))
                .build()
                .ok()
        });
        match pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }
}

/// What the person did on the photo to find an object (`mask_find`'s `prompt`, `docs/23` section 9), in the photo
/// frame (`docs/23` section 3.6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectPrompt {
    /// A click: `positive` for a part of the object, not positive for a part to take away.
    Point {
        /// Fraction of the photo's width.
        x: f64,
        /// Fraction of the photo's height.
        y: f64,
        /// On the object, or off it.
        positive: bool,
    },
    /// A box dragged around the object.
    Box {
        /// Left, as a fraction of the width.
        x0: f64,
        /// Top, as a fraction of the height.
        y0: f64,
        /// Right.
        x1: f64,
        /// Bottom.
        y1: f64,
    },
}

/// Why a prompt found nothing (the rule of `docs/23` section 4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NothingFound {
    /// Which rule refused it.
    pub reason: sam2::NothingReason,
    /// The object score the decoder gave.
    pub object_score: f32,
    /// The fraction of the photo the best mask covered.
    pub area: f64,
}

/// The answer of `mask_find` (`docs/23` section 9).
#[derive(Debug, Clone, PartialEq)]
pub enum MaskFound {
    /// A mask: the row's new points and box (the prompt added to what the row had) and the time the decoder took.
    Found {
        /// The row's points with the new one at the end (for a point prompt).
        points: Vec<(f64, f64, bool)>,
        /// The row's box (the new one for a box prompt).
        box_: Option<[f64; 4]>,
        /// Milliseconds the decoder took, with the soft mask made.
        took_ms: f64,
    },
    /// No object there: the settings are not to change.
    NothingFound(NothingFound),
}

/// The planes a render uses for the rows that read the Input's picture: for the Object rows of a plan, by the row's
/// prompt ([`prompt_key`]), the `S0` of every enabled row that has a point or a box and whose features exist; and for the
/// Subject and People rows, by the request, the union of Vision's instances (N-19). A row without a plane selects nothing.
#[derive(Debug, Clone, Default)]
pub struct ObjectPlanes {
    planes: HashMap<u64, Arc<SourcePlane>>,
    vision: [Option<Arc<SourcePlane>>; 2],
}

impl ObjectPlanes {
    /// Adds the plane of the rows that have this prompt.
    pub(crate) fn insert(&mut self, prompt: u64, plane: Arc<SourcePlane>) {
        self.planes.insert(prompt, plane);
    }

    /// Adds the plane of the rows of a Vision request.
    pub(crate) fn insert_vision(&mut self, kind: VisionKind, plane: Arc<SourcePlane>) {
        self.vision[vision_index(kind)] = Some(plane);
    }

    /// The plane of the Subject or People rows: `None` when the computer cannot make it, it is not made yet or Vision
    /// found nothing.
    #[must_use]
    pub fn vision_plane(&self, kind: VisionKind) -> Option<&Arc<SourcePlane>> {
        self.vision[vision_index(kind)].as_ref()
    }

    /// Whether no row has a plane.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.planes.is_empty() && self.vision.iter().all(Option::is_none)
    }

    /// The plane of an Object row, by what the row asks (its points and box). `None` for a row with none or whose
    /// features are not there.
    #[must_use]
    pub fn plane_for(&self, o: &ObjectMask) -> Option<&Arc<SourcePlane>> {
        self.planes.get(&prompt_key(&o.points, o.box_))
    }
}

/// Where a request's plane sits in [`ObjectPlanes`].
fn vision_index(kind: VisionKind) -> usize {
    match kind {
        VisionKind::Subject => 0,
        VisionKind::People => 1,
    }
}

/// FNV-1a of a row's prompt: its points in order and its box, bit for bit. Equal prompts are equal masks for one photo.
#[must_use]
pub fn prompt_key(points: &[(f64, f64, bool)], bbox: Option<[f64; 4]>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(points.len() as u64);
    for (x, y, positive) in points {
        mix(x.to_bits());
        mix(y.to_bits());
        mix(u64::from(*positive));
    }
    match bbox {
        Some(b) => {
            mix(1);
            for v in b {
                mix(v.to_bits());
            }
        }
        None => mix(0),
    }
    h
}

/// Whether an Object row asks for a mask: it has a point or a box. A row with neither selects nothing
/// (`docs/23` section 3.3).
#[must_use]
pub fn asks_for_mask(o: &ObjectMask) -> bool {
    !o.points.is_empty() || o.box_.is_some()
}

/// A coordinate rounded to [`OBJECT_POINT_DECIMALS`] and held to the photo.
#[must_use]
pub fn round_coordinate(v: f64) -> f64 {
    let scale = 10f64.powi(OBJECT_POINT_DECIMALS);
    ((v.clamp(0.0, 1.0)) * scale).round() / scale
}

/// A box with its corners in order and its coordinates rounded and held to the photo.
#[must_use]
pub fn round_box(b: [f64; 4]) -> [f64; 4] {
    let [x0, y0, x1, y1] = b.map(round_coordinate);
    [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]
}

/// The prompt of the decoder for a row's points and box.
#[must_use]
pub fn sam_prompt(points: &[(f64, f64, bool)], bbox: Option<[f64; 4]>) -> sam2::Prompt {
    sam2::Prompt {
        points: points
            .iter()
            .map(|&(x, y, positive)| sam2::Point {
                x: x as f32,
                y: y as f32,
                positive,
            })
            .collect(),
        bbox: bbox.map(|b| b.map(|v| v as f32)),
    }
}

/// The most points a row may hold; a prompt that would make one more is refused with the core's sentence.
pub const MAX_POINTS: usize = MAX_OBJECT_POINTS;

/// The photo as SAM looks at it (`docs/algorithms/06` section 5.7): the Input's picture, and nothing after it. The
/// settings keep the Input's raw, light and noise fields and drop everything that is a later step or a view of the
/// result: no crop and no turn (the picture is the photo frame, `docs/23` section 3.6), no lens character (its bend is
/// not in the photo frame), Main and every other node at their identity with no masks, the Neutral look and no output
/// sharpening. So grading a node, changing the look or moving a slider of another node never moves a mask.
#[must_use]
pub fn object_source(src: &PhotoSource) -> PhotoSource {
    let mut s = DevelopSettings::default();
    {
        let input = s.input_mut();
        *input = src.settings.input().clone();
        input.crop = None;
        input.orientation = crate::settings::Orientation::Deg0;
        input.lens = crate::settings::LensSettings::default();
    }
    *s.main_grade_mut() = NodeGrade::neutral();
    let out = s.output_mut();
    out.look = LookId::neutral();
    out.look_intensity = 100.0;
    out.output_sharpening = 0.0;
    src.with_settings(s)
}

/// The plane an Object row's soft mask makes: u16 codes at SAM's `S0` size over the photo frame.
#[must_use]
pub fn plane_of(key: u64, mask: &sam2::SoftMask) -> SourcePlane {
    SourcePlane {
        key,
        width: mask.width,
        height: mask.height,
        data: mask
            .data
            .iter()
            .map(|&v| ym_color::mask_raster::quantise(f64::from(v)))
            .collect(),
    }
}
