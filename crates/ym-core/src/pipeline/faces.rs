//! The Face Refinement node's models and what the face worker tells its listener (`docs/27-skin-retouch.md` section 10,
//! S-11b, and decisions 9, 12 and 13).
//!
//! The model ([`FaceModel`]) is MediaPipe's BlazeFace (the finder), Face Mesh V2 and Selfie Multiclass (`ym_ml::mediapipe`,
//! run through `ym_retouch::mediapipe`), with the classical blemish proposer and mole detector of `ym_retouch`. It loads its
//! weights the first time something needs them. The renderer keeps one ([`RenderOptions::face_model`](super::RenderOptions));
//! the node is live when it is there. What runs when a photo has a Face Refinement node is in `graph/faces.rs`: the worker
//! in the background, the analysis the photo keeps, and the retouch step in `frame/film_pass.rs`.
//!
//! A model can also be a **fake** ([`FaceModel::fake`]): a function from the analysis picture to face boxes, with the
//! deterministic mock models of `ym_retouch` for everything after the finder. The tests of the worker, the cache, the
//! geometry, the random graphs and the app-and-`ym` parity use it, so they need no weights.
//!
//! # Starting values
//!
//! [`FACE_ANALYSIS_LONG_EDGE_PX`], [`MIN_FACE_FULL_PX`] and the finder's score threshold are starting values for S-12.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ym_ml::mediapipe::blazeface::{BlazeFace, DetectedFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::mediapipe::MediaPipeFaces;
use ym_retouch::{ClassicalBeautyMarks, ClassicalProposer, Face, Models, Rect};

use super::RenderError;

/// The version of the weights the analysis reads (BlazeFace full range, Face Mesh V2, Selfie Multiclass), part of the key.
pub const FACES_FORMAT_VERSION: u32 = 1;
/// The version of the rules that turn the finder's boxes into the numbered faces (the size floor, the cap, the order) and of the
/// analysis picture, part of the key. A change of any constant below changes this number.
pub const FACES_RULES_VERSION: u32 = 1;

/// The long edge, in pixels, of the picture the analysis looks at: the photo's final frame (crop, turn and lens kept) at this
/// size at most, never enlarged. A face of [`MIN_FACE_FULL_PX`] in a 24 MP photo (6000 px) is 102 px in it, about three times
/// what the finder needs; the masks and the blemish maps of the faces are built at this size and resampled to the render
/// (`docs/27` decision 13). Starting value for S-12.
pub const FACE_ANALYSIS_LONG_EDGE_PX: usize = 1400;
/// The smallest face that is retouched, in pixels of its longer side in the full-resolution photo (`docs/27` section 11 item 9).
/// Starting value for S-12.
pub const MIN_FACE_FULL_PX: u32 = 200;
/// The most faces a photo keeps (the best by score, then numbered from left to right).
pub const MAX_FACES_KEPT: usize = crate::settings::MAX_FACES;
/// The finder is asked for faces down to this share of [`MIN_FACE_FULL_PX`], so that a face of exactly the minimum is found
/// whatever its box says to within a few pixels. The size rule is applied after, on the box.
const FINDER_SLACK: f64 = 0.8;
/// The finder's score threshold. Starting value for S-12.
pub const FINDER_SCORE_THRESHOLD: f32 = 0.5;
/// The most the finder looks for in one picture: a margin over the cap, so that faces dropped for their size do not use up
/// the cap.
const FINDER_MAX_FACES: usize = 64;

/// The sentence for a photo with a live Face Refinement node on a copy of Yellowmoon that has no face models (the `mask`
/// error of `ym render` and of an export, as [`NO_DEPTH_MODEL_SENTENCE`](crate::settings::NO_DEPTH_MODEL_SENTENCE) is for
/// a Depth row): what went wrong and what to do.
#[must_use]
pub fn no_face_model_sentence(node: u32) -> String {
    format!(
        "Node {node} is a Face Refinement node, and the face models are missing. Reinstall Yellowmoon, or remove the node."
    )
}

/// The sentence the node's panel shows for [`FacePhase::ModelsMissing`].
pub const NO_FACE_MODEL_SENTENCE: &str = "Face Refinement needs the face models, and this copy of Yellowmoon has none. Reinstall the app.";

/// A face as the panel lists it: its number and its box in the oriented photo frame (the picture as it is shown, crop and
/// turn applied), as shares of the frame's width and height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBox {
    /// The face's number: 1 for the leftmost, up to 32.
    pub number: u32,
    /// Left edge, 0 to 1.
    pub x: f64,
    /// Top edge, 0 to 1.
    pub y: f64,
    /// Width, 0 to 1.
    pub width: f64,
    /// Height, 0 to 1.
    pub height: f64,
}

/// Where the face analysis of a photo is (`docs/27` decision 9): what the panel of a Face Refinement node says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacePhase {
    /// The faces are being found, or will be in a moment: the node says "Finding faces…" and the photo is shown without its
    /// effect.
    Finding,
    /// The faces are known (there may be none: "No faces found").
    Ready,
    /// The analysis failed; the sentence says why. Nothing asks again until the picture changes.
    Failed,
    /// This copy of Yellowmoon has no face models: the node passes its picture through.
    ModelsMissing,
}

impl FacePhase {
    /// The phase's wire name: `finding`, `ready`, `failed` or `models_missing`.
    #[must_use]
    pub fn wire(self) -> &'static str {
        match self {
            Self::Finding => "finding",
            Self::Ready => "ready",
            Self::Failed => "failed",
            Self::ModelsMissing => "models_missing",
        }
    }
}

/// What the face worker tells the listener about a photo, and what [`Renderer::face_state`](super::Renderer::face_state)
/// answers.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceState {
    /// Where the analysis is.
    pub phase: FacePhase,
    /// The faces, numbered from left to right; empty unless [`FacePhase::Ready`].
    pub faces: Vec<FaceBox>,
    /// The plain sentence of the failure ([`FacePhase::Failed`]) or of the missing models ([`FacePhase::ModelsMissing`]).
    pub message: Option<String>,
}

impl FaceState {
    /// The faces are being found.
    #[must_use]
    pub fn finding() -> Self {
        Self {
            phase: FacePhase::Finding,
            faces: Vec::new(),
            message: None,
        }
    }

    /// The faces are known.
    #[must_use]
    pub fn ready(faces: Vec<FaceBox>) -> Self {
        Self {
            phase: FacePhase::Ready,
            faces,
            message: None,
        }
    }

    /// The analysis failed with this sentence.
    #[must_use]
    pub fn failed(message: String) -> Self {
        Self {
            phase: FacePhase::Failed,
            faces: Vec::new(),
            message: Some(message),
        }
    }

    /// This copy has no face models.
    #[must_use]
    pub fn models_missing() -> Self {
        Self {
            phase: FacePhase::ModelsMissing,
            faces: Vec::new(),
            message: Some(NO_FACE_MODEL_SENTENCE.to_owned()),
        }
    }
}

/// What the face worker tells its listener ([`Renderer::set_faces_listener`](super::Renderer::set_faces_listener)): it started
/// on a photo, and, once it has ended, where the photo's faces are. Each `Started` is followed by one `Finished`.
#[derive(Debug, Clone, PartialEq)]
pub enum FaceEvent {
    /// The worker has work for the photo: the picture and the networks are on their way. The node says "Finding faces…"
    /// (`viewport_busy.finding`).
    Started,
    /// The worker ended its work for the photo. A frame that showed the node's input for want of faces should be drawn again.
    Finished(FaceState),
}

/// A face box a fake finder gives: shares of the picture's width and height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FakeFace {
    /// Left edge, 0 to 1.
    pub x: f64,
    /// Top edge, 0 to 1.
    pub y: f64,
    /// Width, 0 to 1.
    pub width: f64,
    /// Height, 0 to 1.
    pub height: f64,
    /// The finder's confidence, 0 to 1 (the best 32 are kept).
    pub score: f32,
}

/// What a fake finder is given: the analysis picture (`width x height`, `[y][x][rgb]`, 8 bits, sRGB). It answers face boxes.
pub type FakeFaces = dyn Fn(usize, usize, &[u8]) -> Vec<FakeFace> + Send + Sync;

/// A face the finder found, in pixels of the analysis picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Found {
    pub rect: Rect,
    pub scale: f32,
    pub score: f32,
}

/// The weights of the three networks, loaded.
struct Loaded {
    finder: FaceFinder,
    faces: Arc<MediaPipeFaces>,
}

enum Net {
    /// The MediaPipe networks, read from the model folder when first needed.
    Network {
        dir: PathBuf,
        state: Mutex<NetState>,
    },
    /// A function that stands in for the finder (tests).
    Fake(Arc<FakeFaces>),
}

enum NetState {
    /// The weights have not been read yet.
    Untried,
    /// Loaded.
    Ready(Arc<Loaded>),
    /// Could not be read, did not match their checksum or are not the models (the reason is in the log).
    Damaged,
}

/// The face finder, the face mesh and the segmenter, and where their weights are. The weights load the first time they are
/// asked for (on the first analysis); a file that is damaged takes the model away for the rest of the run
/// ([`FaceModel::available`]).
pub struct FaceModel {
    net: Net,
    /// Names the weights in the analysis' key: the SHA-256 lines of `SHA256SUMS`, or the fake's name.
    weights_id: String,
    /// The smallest face, in pixels of the full-resolution photo ([`MIN_FACE_FULL_PX`], less for a fake).
    min_face_px: u32,
    pool: OnceLock<Option<rayon::ThreadPool>>,
}

impl std::fmt::Debug for FaceModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceModel")
            .field("weights_id", &self.weights_id)
            .field("available", &self.available())
            .finish_non_exhaustive()
    }
}

/// The three weight files of the model folder.
const WEIGHT_FILES: [&str; 3] = [
    "blaze_face_full_range.f16.safetensors",
    "face_mesh_v2.f16.safetensors",
    "selfie_multiclass_256.f32.safetensors",
];

/// The SHA-256 lines `dir/SHA256SUMS` lists for the three weight files, joined; `None` when one is missing.
fn listed_sha256(dir: &Path) -> Option<String> {
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS")).ok()?;
    let mut ids = Vec::new();
    for file in WEIGHT_FILES {
        let hash = sums
            .lines()
            .filter_map(|l| l.split_once("  "))
            .find(|(_, name)| name.trim() == file)
            .map(|(hash, _)| hash.trim().to_owned())?;
        ids.push(hash);
    }
    Some(ids.join("+"))
}

impl FaceModel {
    /// The model of a data folder (`<data_root>/models/mediapipe/`), when its three weight files and a `SHA256SUMS` that lists
    /// them are there. Nothing is read beyond the sums: the weights load, and are checked against their checksums, on first
    /// use, and a file that fails takes the model away ([`FaceModel::available`]).
    #[must_use]
    pub fn probe(data_root: &Path) -> Option<Arc<Self>> {
        Self::probe_with(data_root, MIN_FACE_FULL_PX)
    }

    /// [`FaceModel::probe`] with another smallest face (in pixels of the full-resolution photo), for tests and for tuning
    /// (`docs/27` S-12). The floor is part of the analysis' key.
    #[must_use]
    pub fn probe_with(data_root: &Path, min_face_px: u32) -> Option<Arc<Self>> {
        let dir = data_root.join("models").join("mediapipe");
        if !WEIGHT_FILES.iter().all(|f| dir.join(f).is_file()) {
            return None;
        }
        let weights_id = listed_sha256(&dir)?;
        Some(Arc::new(Self {
            net: Net::Network {
                dir,
                state: Mutex::new(NetState::Untried),
            },
            weights_id: if min_face_px == MIN_FACE_FULL_PX {
                weights_id
            } else {
                format!("{weights_id}:{min_face_px}")
            },
            min_face_px,
            pool: OnceLock::new(),
        }))
    }

    /// A model whose finder is a function (tests): it is given the analysis picture and answers face boxes. `name` names it in
    /// the analysis' key, so two fakes that differ do not share analyses. The meshes, the parser, the proposer and the mole
    /// detector are `ym_retouch`'s mocks. Faces down to `min_face_px` (in pixels of the full-resolution photo) are kept, in
    /// place of [`MIN_FACE_FULL_PX`], because the pictures of the tests are small.
    #[must_use]
    pub fn fake(name: &str, min_face_px: u32, f: Arc<FakeFaces>) -> Arc<Self> {
        Arc::new(Self {
            net: Net::Fake(f),
            weights_id: format!("fake:{name}:{min_face_px}"),
            min_face_px,
            pool: OnceLock::new(),
        })
    }

    /// The environment variable that makes [`Core::open`](crate::app::Core::open) use [`FaceModel::fake_default`] in place of
    /// the weights when it is `fake`: the test hook of `ym` and the app (as `YELLOWMOON_TEST_DEPTH` is for Depth), so that
    /// their parity and their Face Refinement nodes can be tested without the weights.
    pub const TEST_ENV: &'static str = "YELLOWMOON_TEST_FACES";

    /// A fake that finds two faces at fixed places of the picture (a larger one at the left, a smaller one at the right), with
    /// the minimum of 24 px, so any test picture has faces.
    #[must_use]
    pub fn fake_default() -> Arc<Self> {
        Self::fake(
            "default",
            24,
            Arc::new(|_, _, _| {
                vec![
                    FakeFace {
                        x: 0.12,
                        y: 0.18,
                        width: 0.26,
                        height: 0.46,
                        score: 0.9,
                    },
                    FakeFace {
                        x: 0.60,
                        y: 0.26,
                        width: 0.2,
                        height: 0.36,
                        score: 0.8,
                    },
                ]
            }),
        )
    }

    /// Names the weights in the analysis' key.
    #[must_use]
    pub fn weights_id(&self) -> &str {
        &self.weights_id
    }

    /// The smallest face this model keeps, in pixels of the full-resolution photo.
    #[must_use]
    pub fn min_face_px(&self) -> u32 {
        self.min_face_px
    }

    /// Whether the node can be live: the weights are there and have not been found damaged.
    #[must_use]
    pub fn available(&self) -> bool {
        match &self.net {
            Net::Fake(_) => true,
            Net::Network { state, .. } => !matches!(
                *state.lock().unwrap_or_else(PoisonError::into_inner),
                NetState::Damaged
            ),
        }
    }

    /// The networks, loading the weights when this is the first call.
    fn loaded(&self) -> Result<Arc<Loaded>, RenderError> {
        let Net::Network { dir, state } = &self.net else {
            return Err(RenderError::Mask(NO_FACE_MODEL_SENTENCE.to_owned()));
        };
        let mut st = state.lock().unwrap_or_else(PoisonError::into_inner);
        match &*st {
            NetState::Ready(net) => return Ok(Arc::clone(net)),
            NetState::Damaged => return Err(RenderError::Mask(NO_FACE_MODEL_SENTENCE.to_owned())),
            NetState::Untried => {}
        }
        let started = std::time::Instant::now();
        let loaded = BlazeFace::load(dir)
            .map_err(|e| e.to_string())
            .and_then(|net| {
                MediaPipeFaces::load(dir)
                    .map(|faces| Loaded {
                        finder: FaceFinder::new(net),
                        faces,
                    })
                    .map_err(|e| e.to_string())
            });
        match loaded {
            Ok(loaded) => {
                tracing::debug!(
                    ms = started.elapsed().as_secs_f64() * 1000.0,
                    "the face models loaded"
                );
                let loaded = Arc::new(loaded);
                *st = NetState::Ready(Arc::clone(&loaded));
                Ok(loaded)
            }
            Err(e) => {
                tracing::error!(error = %e, "the face models could not be loaded");
                *st = NetState::Damaged;
                Err(RenderError::Mask(NO_FACE_MODEL_SENTENCE.to_owned()))
            }
        }
    }

    /// Runs `f` on the threads this model keeps: all the cores but one, so that a photo being edited stays smooth while it runs
    /// (the bits do not depend on the number of threads).
    fn run<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        let pool = self.pool.get_or_init(|| {
            let cores = std::thread::available_parallelism().map_or(1, usize::from);
            rayon::ThreadPoolBuilder::new()
                .num_threads(cores.saturating_sub(1).max(1))
                .thread_name(|i| format!("yellowmoon-faces-{i}"))
                .build()
                .ok()
        });
        match pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// The faces of the analysis picture (`width x height`, `[y][x][rgb]`, 8 bits, sRGB), best first, as the finder gives them;
    /// the size floor, the cap and the numbering are the caller's ([`select_faces`]). `full_long_px` is the long edge of the
    /// full-resolution photo, which sets the smallest face the finder looks for.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] with [`NO_FACE_MODEL_SENTENCE`] when the weights cannot be read, and with "Faces can't be found: …"
    /// when the finder fails.
    pub(crate) fn find(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        full_long_px: usize,
    ) -> Result<Vec<Found>, RenderError> {
        let failed = |text: String| {
            RenderError::Mask(format!(
                "Faces can't be found: {}.",
                text.trim_end_matches('.')
            ))
        };
        match &self.net {
            Net::Fake(f) => Ok(f(width, height, rgb)
                .into_iter()
                .filter_map(|b| {
                    let rect = Rect::new(
                        (b.x * width as f64).round() as i32,
                        (b.y * height as f64).round() as i32,
                        (b.width * width as f64).round().max(1.0) as u32,
                        (b.height * height as f64).round().max(1.0) as u32,
                    );
                    let long = width.max(height) as f64;
                    let side = f64::from(rect.width_px.max(rect.height_px));
                    (b.score.is_finite()).then_some(Found {
                        rect,
                        scale: (side / long) as f32,
                        score: b.score,
                    })
                })
                .collect()),
            Net::Network { .. } => {
                let net = self.loaded()?;
                // The pictures are `analysis / full` of the photo: the smallest face the finder looks for is the minimum
                // face in the picture's pixels, with some slack (the box is coarse).
                let ratio = width.max(height) as f64 / full_long_px.max(1) as f64;
                let min_face = (f64::from(self.min_face_px) * ratio * FINDER_SLACK).max(24.0);
                let src = Rgb8::new(width, height, rgb).map_err(|e| failed(e.to_string()))?;
                let params = FinderParams {
                    score_threshold: FINDER_SCORE_THRESHOLD,
                    max_faces: FINDER_MAX_FACES,
                    min_face_px: Some(min_face.round() as u32),
                    ..FinderParams::default()
                };
                let found: Vec<DetectedFace> = self
                    .run(|| net.finder.find(&src, &params))
                    .map_err(|e| failed(e.to_string()))?;
                Ok(found
                    .into_iter()
                    .map(|d| Found {
                        // The two lines of decision 12: the finder's rectangle is `ym_retouch::Rect`'s fields.
                        rect: Rect::new(
                            d.rect.x_px,
                            d.rect.y_px,
                            d.rect.width_px,
                            d.rect.height_px,
                        ),
                        scale: d.scale,
                        score: d.score,
                    })
                    .collect())
            }
        }
    }

    /// The models one analysis uses: the real parser and mesh on the loaded networks (or the mocks, for a fake), the classical
    /// mole detector and blemish proposer. Made for each analysis; the networks and their cache are shared.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] with [`NO_FACE_MODEL_SENTENCE`] when the weights cannot be read.
    pub(crate) fn models(&self) -> Result<Models, RenderError> {
        match &self.net {
            Net::Fake(_) => Ok(ym_retouch::mock::models(Vec::new())),
            Net::Network { .. } => {
                let net = self.loaded()?;
                Ok(Models {
                    parser: Box::new(net.faces.parser()),
                    mesh: Box::new(net.faces.mesh()),
                    beauty_marks: Some(Box::new(ClassicalBeautyMarks)),
                    proposer: Box::new(ClassicalProposer::new(ym_retouch::TARGET_FACE_PX)),
                })
            }
        }
    }

    /// Runs `f` on the model's threads (for the masks and maps of the analysis).
    pub(crate) fn on_threads<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        self.run(f)
    }
}

/// The numbered faces of a photo from what the finder found (`docs/27` decision 9): drop the faces under `min_face_px`
/// across in the full-resolution photo (`analysis` is the analysis picture's size, `full` the photo's), keep the best
/// [`MAX_FACES_KEPT`] by score, and number them from left to right by the centre of their box (ties by the top edge, then by the
/// score), 1 to 32. Pure: the same boxes always give the same numbers.
#[must_use]
pub(crate) fn select_faces(
    found: &[Found],
    analysis: (u32, u32),
    full: (u32, u32),
    min_face_px: u32,
) -> Vec<(u32, Face)> {
    let ratio = f64::from(full.0.max(full.1)) / f64::from(analysis.0.max(analysis.1)).max(1.0);
    let mut kept: Vec<&Found> = found
        .iter()
        .filter(|f| {
            f64::from(f.rect.width_px.max(f.rect.height_px)) * ratio >= f64::from(min_face_px)
        })
        .collect();
    // Best first, ties by position so the cap is stable.
    kept.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.rect.x_px.cmp(&b.rect.x_px))
            .then(a.rect.y_px.cmp(&b.rect.y_px))
    });
    kept.truncate(MAX_FACES_KEPT);
    let centre = |f: &Found| 2 * i64::from(f.rect.x_px) + i64::from(f.rect.width_px);
    kept.sort_by(|a, b| {
        centre(a)
            .cmp(&centre(b))
            .then(a.rect.y_px.cmp(&b.rect.y_px))
            .then(b.score.total_cmp(&a.score))
    });
    kept.iter()
        .enumerate()
        .map(|(i, f)| {
            (
                i as u32 + 1,
                Face {
                    rect: f.rect,
                    scale: f.scale,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(x: i32, y: i32, side: u32, score: f32) -> Found {
        Found {
            rect: Rect::new(x, y, side, side),
            scale: side as f32 / 1000.0,
            score,
        }
    }

    #[test]
    fn faces_are_dropped_under_the_floor_capped_and_numbered_left_to_right() {
        // A 1000 px analysis picture of a 6000 px photo: 40 px across is 240 px in the photo, 30 px is 180.
        let all = [
            found(700, 100, 40, 0.9),
            found(100, 300, 50, 0.7),
            found(400, 100, 30, 0.99),
            found(405, 500, 45, 0.8),
        ];
        let faces = select_faces(&all, (1000, 1000), (6000, 6000), 200);
        let order: Vec<(u32, i32)> = faces.iter().map(|(n, f)| (*n, f.rect.x_px)).collect();
        assert_eq!(
            order,
            [(1, 100), (2, 405), (3, 700)],
            "the 30 px face is under 200 px"
        );
        // The same faces in another order are numbered the same.
        let mut shuffled = all;
        shuffled.reverse();
        assert_eq!(
            select_faces(&shuffled, (1000, 1000), (6000, 6000), 200),
            faces
        );
        // More than 32 keep the best 32.
        let many: Vec<Found> = (0..40)
            .map(|i| found(i * 20, 10, 40, 0.5 + i as f32 / 100.0))
            .collect();
        let capped = select_faces(&many, (1000, 1000), (6000, 6000), 200);
        assert_eq!(capped.len(), MAX_FACES_KEPT);
        let lowest = capped.iter().map(|(_, f)| f.rect.x_px).min().unwrap();
        assert_eq!(
            lowest,
            8 * 20,
            "the eight weakest, on the left, were dropped"
        );
        assert!(
            capped
                .windows(2)
                .all(|w| w[0].1.rect.x_px < w[1].1.rect.x_px)
        );
        assert_eq!(capped[0].0, 1);
        assert_eq!(capped[31].0, 32);
    }

    #[test]
    fn the_phases_have_their_wire_names() {
        assert_eq!(FacePhase::Finding.wire(), "finding");
        assert_eq!(FacePhase::ModelsMissing.wire(), "models_missing");
        assert_eq!(FaceState::finding().faces.len(), 0);
        assert!(FaceState::models_missing().message.is_some());
    }
}
