//! The scene model of Reference Grade and what the renderer tells its listener (`docs/34-reference-grade.md` sections 4a.3, 4b
//! and 7.1, RG-04).
//!
//! The model ([`SceneModel`]) is the segmenter `ym_ml::eomt::Eomt` with MediaPipe's Selfie Multiclass for skin
//! (`ym_ml::scene`): the pictures of the Input go in, a scene (the seven element masks as bits) comes out. The renderer keeps
//! one ([`RenderOptions::scene_model`](super::RenderOptions)); [`Core::open`](crate::app::Core::open) finds it in the data
//! folder (`models/eomt/` and `models/mediapipe/`) and loads the weights when a node first needs them. What runs when a
//! photo has a Reference Grade node with a live element grade is in `graph/scene.rs`: the worker in the background, the scene the
//! photo keeps in memory, and what an export does.
//!
//! A model can also be a **fake** ([`SceneModel::fake`]): a function from the picture to a scene. The tests of the worker, the
//! cache, the geometry, the random graphs and the app-and-`ym` parity use it, so they need no weights; `YELLOWMOON_TEST_SCENE=fake`
//! puts [`SceneModel::fake_default`] in the place of the weights for `ym` and the app alike.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use ym_ml::eomt::{self, Eomt};
use ym_ml::mediapipe::selfie::{self, SelfieNet, SelfieSegmenter};
use ym_ml::model::MlError;
use ym_ml::scene::{BUILT, GROUND, SKIN, SKY, SUBJECT, SceneBits, VEGETATION, WATER, make_scene};

use super::RenderError;
use super::resident::{Clock, ModelSlot, SlotError, Unload, Unloadable};

/// The version of the weights files' format the model reads (`yellowmoon-eomt` version 1 and the Selfie Multiclass safetensors),
/// part of the scene's key.
pub const SCENE_FORMAT_VERSION: u32 = 1;

/// The sentence for a node whose Elements need the scene model on a copy of Yellowmoon that has none (the `mask` error of an
/// export, and the note of the viewer): what went wrong and what to do.
pub const NO_SCENE_MODEL_SENTENCE: &str = "This copy of Yellowmoon has no scene model, so the Elements of a Reference Grade node can't be made. Reinstall the app.";

/// The sentence for a node whose Skin grade needs the skin model on a copy of Yellowmoon that has none.
pub const NO_SKIN_MODEL_SENTENCE: &str = "This copy of Yellowmoon has no skin model, so the Skin of a Reference Grade node can't be made. Reinstall the app.";

/// The note of a viewer frame that has no element masks for want of the scene model.
pub const SCENE_MODEL_MISSING_NOTE: &str = "The scene model is missing, so the Elements of a Reference Grade node are not applied. Reinstall the app.";

/// The note of a viewer frame that has no skin mask for want of the skin model.
pub const SKIN_MODEL_MISSING_NOTE: &str = "The skin model is missing, so the Skin of a Reference Grade node is not applied. Reinstall the app.";

/// What a weights file of the scene model is: found and as its checksum says, not there, or there and not as the checksum says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsState {
    /// The file is there and its SHA-256 is the one `SHA256SUMS` lists.
    Ready,
    /// The file (or its line in `SHA256SUMS`) is not there.
    Missing,
    /// The file is there and is not the one `SHA256SUMS` lists (truncated, changed).
    Damaged,
}

impl WeightsState {
    /// The word `ym doctor --json` writes: `ready`, `missing` or `damaged`.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Missing => "missing",
            Self::Damaged => "damaged",
        }
    }
}

/// One weights file of the scene model as `ym doctor` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightsReport {
    /// The file's name under `models/`, with forward slashes (`eomt/eomt_coco_panoptic_small.f16.safetensors`).
    pub name: String,
    /// What the file is.
    pub state: WeightsState,
    /// The SHA-256 of the file when it is there (also when it is damaged).
    pub sha256: Option<String>,
}

/// The state of the scene model's two weights files in a data folder: the segmenter's and MediaPipe's Selfie Multiclass (the skin
/// network, without which the scene has no skin). The files are read and hashed (about 50 MB), not loaded.
#[must_use]
pub fn weights_reports(data_root: &Path) -> [WeightsReport; 2] {
    let one = |dir: &str, file: &str| {
        let folder = data_root.join("models").join(dir);
        let name = format!("{dir}/{file}");
        let path = folder.join(file);
        let listed = listed_sha256(&folder, file);
        if !path.is_file() {
            return WeightsReport {
                name,
                state: WeightsState::Missing,
                sha256: None,
            };
        }
        let sha256 = ym_looks::manifest::hash_file(&path).ok();
        let state = match (&listed, &sha256) {
            (None, _) => WeightsState::Missing,
            (Some(l), Some(h)) if l.eq_ignore_ascii_case(h) => WeightsState::Ready,
            _ => WeightsState::Damaged,
        };
        WeightsReport {
            name,
            state,
            sha256,
        }
    };
    [
        one("eomt", eomt::WEIGHTS_FILE),
        one("mediapipe", selfie::WEIGHTS_FILE),
    ]
}

/// What the scene worker is doing for a photo, as the renderer's listener hears it
/// ([`Renderer::set_scene_listener`](super::Renderer::set_scene_listener)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenePhase {
    /// The worker has work for the photo: the picture and the networks are on their way. The nodes say "Finding…"
    /// (`viewport_busy.finding`).
    Started,
    /// The worker ended its work for the photo (after a `Started`). A frame that had no element masks for want of the scene should
    /// be drawn again.
    Finished {
        /// The plain sentence of the failure ("The Elements can't be made: …"), or `None` when the scene is ready.
        failed: Option<String>,
    },
}

/// What a fake model is given: the picture the segmenter would see (`width x height`, `[y][x][rgb]`, 8 bits) and a question that
/// says whether the run was cancelled. It answers the scene's bits (one byte a pixel, a bit for each element) or the text of its
/// failure.
pub type FakeScene =
    dyn Fn(usize, usize, &[u8], &dyn Fn() -> bool) -> Result<Vec<u8>, String> + Send + Sync;

enum Net {
    /// EoMT small and MediaPipe's Selfie Multiclass, read from the model folders when first needed and let go when idle or when
    /// memory is short.
    Network {
        eomt_dir: PathBuf,
        selfie_dir: PathBuf,
        eomt: ModelSlot<Eomt>,
        selfie: ModelSlot<SelfieSegmenter>,
        /// The Selfie weights are in the folder (and listed): without them the scene has no skin.
        has_selfie: bool,
    },
    /// A function that stands in for them (tests). Its "weights" are a unit value in a slot.
    Fake {
        f: Arc<FakeScene>,
        resident: ModelSlot<()>,
        skin: bool,
    },
}

/// The scene model and where its weights are. The weights load the first time they are asked for (on the first scene); a file
/// that is damaged takes the model away for the rest of the run ([`SceneModel::available`]).
pub struct SceneModel {
    net: Net,
    /// Names the weights in the scene's key: their SHA-256 lines, or the fake's name.
    weights_id: String,
    pool: OnceLock<Option<rayon::ThreadPool>>,
}

impl std::fmt::Debug for SceneModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneModel")
            .field("weights_id", &self.weights_id)
            .field("available", &self.available())
            .finish_non_exhaustive()
    }
}

/// The SHA-256 `dir/SHA256SUMS` lists for `file`.
fn listed_sha256(dir: &Path, file: &str) -> Option<String> {
    std::fs::read_to_string(dir.join("SHA256SUMS"))
        .ok()?
        .lines()
        .filter_map(|l| l.split_once("  "))
        .find(|(_, name)| name.trim() == file)
        .map(|(hash, _)| hash.trim().to_owned())
}

/// Folds the outcomes of two slots into one: the most active wins.
fn worst(a: Unload, b: Unload) -> Unload {
    for u in [Unload::Unloaded, Unload::InUse, Unload::Kept] {
        if a == u || b == u {
            return u;
        }
    }
    Unload::NotLoaded
}

impl Unloadable for SceneModel {
    fn name(&self) -> &'static str {
        "Scene"
    }
    fn sweep(&self) -> Unload {
        match &self.net {
            Net::Network { eomt, selfie, .. } => worst(eomt.sweep(), selfie.sweep()),
            Net::Fake { resident, .. } => resident.sweep(),
        }
    }
    fn trim(&self) -> Unload {
        match &self.net {
            Net::Network { eomt, selfie, .. } => worst(eomt.trim(), selfie.trim()),
            Net::Fake { resident, .. } => resident.trim(),
        }
    }
    fn loaded(&self) -> bool {
        match &self.net {
            Net::Network { eomt, selfie, .. } => eomt.loaded() || selfie.loaded(),
            Net::Fake { resident, .. } => resident.loaded(),
        }
    }
}

impl SceneModel {
    /// The model of a data folder: the segmenter's weights in `<data_root>/models/eomt/` and a `SHA256SUMS` that lists them, and
    /// MediaPipe's Selfie Multiclass in `<data_root>/models/mediapipe/` (without which the scene has no skin). Nothing is read
    /// beyond the sums: the weights load, and are checked against their checksums, on first use, and a file that fails takes the
    /// model away ([`SceneModel::available`]). `None` when the segmenter is not there.
    #[must_use]
    pub fn probe(data_root: &Path) -> Option<Arc<Self>> {
        let eomt_dir = data_root.join("models").join("eomt");
        let selfie_dir = data_root.join("models").join("mediapipe");
        if !eomt_dir.join(eomt::WEIGHTS_FILE).is_file() {
            return None;
        }
        let eomt_id = listed_sha256(&eomt_dir, eomt::WEIGHTS_FILE)?;
        let selfie_id = listed_sha256(&selfie_dir, selfie::WEIGHTS_FILE)
            .filter(|_| selfie_dir.join(selfie::WEIGHTS_FILE).is_file());
        Some(Arc::new(Self {
            weights_id: format!("{eomt_id}+{}", selfie_id.as_deref().unwrap_or("no-skin")),
            net: Net::Network {
                has_selfie: selfie_id.is_some(),
                eomt_dir,
                selfie_dir,
                eomt: ModelSlot::lazy(),
                selfie: ModelSlot::lazy(),
            },
            pool: OnceLock::new(),
        }))
    }

    /// [`SceneModel::probe`] with the clock and the idle time of the unload given (tests).
    #[must_use]
    pub fn probe_with_clock(
        data_root: &Path,
        clock: &Clock,
        idle: std::time::Duration,
    ) -> Option<Arc<Self>> {
        let model = Self::probe(data_root)?;
        let mut model = Arc::into_inner(model)?;
        if let Net::Network { eomt, selfie, .. } = &mut model.net {
            *eomt = ModelSlot::lazy_with(Arc::clone(clock), idle);
            *selfie = ModelSlot::lazy_with(Arc::clone(clock), idle);
        }
        Some(Arc::new(model))
    }

    /// A model that is a function (tests): it is given the picture the segmenter would see and answers the scene's bits. `name`
    /// names it in the scene's key, so two fakes that differ do not share scenes; `skin` says whether the fake makes skin (a
    /// model with no skin weights does not).
    #[must_use]
    pub fn fake(name: &str, skin: bool, f: Arc<FakeScene>) -> Arc<Self> {
        Arc::new(Self {
            net: Net::Fake {
                f,
                resident: ModelSlot::pinned(Arc::new(())),
                skin,
            },
            weights_id: format!("fake:{name}"),
            pool: OnceLock::new(),
        })
    }

    /// The environment variable that makes [`Core::open`](crate::app::Core::open) use [`SceneModel::fake_default`] in place of
    /// the weights when it is `fake`: the test hook of `ym` and the app (as `YELLOWMOON_TEST_DEPTH` is for Depth).
    pub const TEST_ENV: &'static str = "YELLOWMOON_TEST_SCENE";

    /// A fake scene drawn by position: sky across the top third, vegetation and built surfaces side by side in the middle third,
    /// ground below with a patch of water at its right, a disc of subject in the centre with skin on its upper half. It has the
    /// shape and the edges a real scene has, and the same bits for the same size.
    #[must_use]
    pub fn fake_default() -> Arc<Self> {
        Self::fake(
            "default",
            true,
            Arc::new(|w, h, _rgb, _| {
                let mut bits = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                        let mut b = if v < 0.33 {
                            SKY
                        } else if v < 0.66 {
                            if u < 0.5 { VEGETATION } else { BUILT }
                        } else if u > 0.7 && v > 0.8 {
                            WATER
                        } else {
                            GROUND
                        };
                        let d = ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)).sqrt();
                        if d < 0.18 {
                            b |= SUBJECT;
                            if v < 0.5 {
                                b |= SKIN;
                            }
                        }
                        bits[y * w + x] = b;
                    }
                }
                Ok(bits)
            }),
        )
    }

    /// Names the weights in the scene's key (the SHA-256 lines of `SHA256SUMS`, or the fake's name).
    #[must_use]
    pub fn weights_id(&self) -> &str {
        &self.weights_id
    }

    /// Whether the model can be used: the segmenter's weights are there and have not been found damaged.
    #[must_use]
    pub fn available(&self) -> bool {
        match &self.net {
            Net::Fake { .. } => true,
            Net::Network { eomt, .. } => !eomt.damaged(),
        }
    }

    /// Whether the scene can have skin: the Selfie weights are there and have not been found damaged.
    #[must_use]
    pub fn skin_available(&self) -> bool {
        match &self.net {
            Net::Fake { skin, .. } => *skin,
            Net::Network {
                has_selfie, selfie, ..
            } => *has_selfie && !selfie.damaged(),
        }
    }

    /// How many times the segmenter's weights have been read since the model was made (a test's count of loads).
    #[must_use]
    pub fn loads(&self) -> u64 {
        match &self.net {
            Net::Network { eomt, .. } => eomt.loads(),
            Net::Fake { resident, .. } => resident.loads(),
        }
    }

    /// Runs `f` on the threads this model keeps for the networks: all the cores but one, so that a photo being edited stays
    /// smooth while it runs (the bits do not depend on the number of threads, `docs/34` section 4b.1).
    fn run<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        let pool = self.pool.get_or_init(|| {
            let cores = std::thread::available_parallelism().map_or(1, usize::from);
            rayon::ThreadPoolBuilder::new()
                .num_threads(cores.saturating_sub(1).max(1))
                .thread_name(|i| format!("yellowmoon-scene-{i}"))
                .build()
                .ok()
        });
        match pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// The scene of a picture (`width x height`, `[y][x][rgb]`, 8 bits, sRGB, the Input's picture): the five class elements, the
    /// subject and, where the Selfie weights are there, skin. `cancel` is asked between the segmenter's blocks.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` says so; [`RenderError::Mask`] with [`NO_SCENE_MODEL_SENTENCE`] when the weights
    /// cannot be read, and with "The Elements can't be made: …" when a network or the fake fails.
    pub(crate) fn scene_of(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<SceneBits, RenderError> {
        let failed = |text: &str| {
            RenderError::Mask(format!(
                "The Elements can't be made: {}.",
                text.trim_end_matches('.')
            ))
        };
        match &self.net {
            Net::Fake { f, resident, skin } => {
                let _weights = resident
                    .get(|| Ok::<(), ()>(()))
                    .map_err(|_| RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned()))?;
                let bits = f(width, height, rgb, cancel).map_err(|e| {
                    if cancel() {
                        RenderError::Cancelled
                    } else {
                        failed(&e)
                    }
                })?;
                if bits.len() != width * height {
                    return Err(failed("the scene has the wrong size"));
                }
                // The fake draws bits only: its classes are read off them (as `ElementScene::class_at` does).
                let classes = bits
                    .iter()
                    .map(|&b| {
                        (0..5)
                            .find(|&e| b >> e & 1 != 0)
                            .map_or(if b & SKIN != 0 { 7 } else { 0 }, |e| e as u8 + 1)
                    })
                    .collect();
                Ok(SceneBits {
                    width,
                    height,
                    bits,
                    classes,
                    skin_available: *skin,
                })
            }
            Net::Network {
                eomt_dir,
                selfie_dir,
                eomt,
                selfie,
                has_selfie,
            } => {
                let net = eomt
                    .get(|| {
                        let started = std::time::Instant::now();
                        let net = Eomt::load(eomt_dir)?;
                        tracing::debug!(
                            ms = started.elapsed().as_secs_f64() * 1000.0,
                            "the scene segmenter loaded"
                        );
                        Ok::<_, MlError>(net)
                    })
                    .map_err(|e| match e {
                        SlotError::Damaged => RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned()),
                        SlotError::Load(e) => {
                            tracing::error!(error = %e, "the scene segmenter could not be loaded");
                            RenderError::Mask(NO_SCENE_MODEL_SENTENCE.to_owned())
                        }
                    })?;
                // The skin network is optional: without it (a file missing or damaged) the scene has no skin.
                let skin_net = if *has_selfie {
                    selfie
                        .get(|| {
                            let net = SelfieNet::load(selfie_dir)?;
                            Ok::<_, MlError>(SelfieSegmenter::new(net))
                        })
                        .map_err(|e| {
                            if let SlotError::Load(e) = &e {
                                tracing::error!(error = %e, "the skin network could not be loaded");
                            }
                        })
                        .ok()
                } else {
                    None
                };
                let stopped = || cancel();
                self.run(|| make_scene(&net, skin_net.as_deref(), rgb, width, height, &stopped))
                    .map_err(|e| match e {
                        MlError::Cancelled => RenderError::Cancelled,
                        other => failed(&other.to_string()),
                    })
            }
        }
    }

    #[cfg(feature = "eval-scene")]
    /// EVALUATION ONLY (`eval-scene`): the scene of a picture whose segmenter's elements are given, not made: the five class elements
    /// from `elements` (the segmenter's element number of every pixel of the `width x height` grid), the subject by its rule and,
    /// where the Selfie weights are there, skin inside `person` on `rgb` (the picture of that grid, 8 bits, sRGB), exactly as
    /// `ym_ml::scene::scene_from_labels` makes them from a label map. The segmenter is not loaded.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when the skin network fails or `rgb` is not `width * height * 3` bytes.
    pub(crate) fn scene_from_elements(
        &self,
        elements: Vec<u8>,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<SceneBits, RenderError> {
        use ym_ml::eomt::Element;
        use ym_ml::mediapipe::blazeface::{PixelSource, Resample, Rgb8};
        use ym_ml::scene::{element_bit, skin_mask, subject_mask};
        let failed = |text: &str| RenderError::Mask(format!("The Elements can't be made: {text}."));
        if elements.len() != width * height {
            return Err(failed("the elements have the wrong size"));
        }
        let element = |n: u8| match n {
            1 => Element::Sky,
            2 => Element::Vegetation,
            3 => Element::Water,
            4 => Element::Ground,
            5 => Element::Architecture,
            6 => Element::Interior,
            7 => Element::Person,
            8 => Element::OtherThing,
            9 => Element::OtherStuff,
            _ => Element::Unlabelled,
        };
        let mut bits: Vec<u8> = elements.iter().map(|&n| element_bit(element(n))).collect();
        let subject = subject_mask(&elements, width, height);
        let (skin, skin_available) = match &self.net {
            Net::Network {
                selfie_dir,
                selfie,
                has_selfie: true,
                ..
            } => {
                let net = selfie
                    .get(|| {
                        let net = SelfieNet::load(selfie_dir)?;
                        Ok::<_, MlError>(SelfieSegmenter::new(net))
                    })
                    .map_err(|_| failed("the skin network could not be loaded"))?;
                let src = Rgb8::new(width, height, rgb).map_err(|e| failed(&e.to_string()))?;
                let person: Vec<bool> = elements.iter().map(|&n| n == 7).collect();
                let skin = self
                    .run(|| {
                        skin_mask(&person, width, height, &|roi| {
                            net.segment(&src as &dyn PixelSource, roi, Resample::Box)
                        })
                    })
                    .map_err(|e| failed(&e.to_string()))?;
                (skin, true)
            }
            _ => (vec![false; width * height], false),
        };
        for (i, b) in bits.iter_mut().enumerate() {
            if subject[i] {
                *b |= SUBJECT;
            }
            if skin[i] {
                *b |= SKIN;
            }
        }
        Ok(SceneBits {
            width,
            height,
            bits,
            classes: elements,
            skin_available,
        })
    }
}

#[cfg(test)]
mod weights_tests {
    use super::*;

    /// A data folder with `files` (folder, name, bytes) under `models/` and, for each folder, a `SHA256SUMS` that lists the file with
    /// the hash of `listed` bytes.
    fn data(files: &[(&str, &str, &[u8], &[u8])]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("a folder");
        for (dir, name, bytes, listed) in files {
            let folder = root.path().join("models").join(dir);
            std::fs::create_dir_all(&folder).expect("a folder");
            std::fs::write(folder.join(name), bytes).expect("a file");
            let scratch = folder.join("listed.tmp");
            std::fs::write(&scratch, listed).expect("a file");
            let hash = ym_looks::manifest::hash_file(&scratch).expect("a hash");
            std::fs::remove_file(&scratch).expect("remove");
            std::fs::write(folder.join("SHA256SUMS"), format!("{hash}  {name}\n")).expect("sums");
        }
        root
    }

    #[test]
    fn the_two_files_are_ready_missing_or_damaged_by_their_checksums() {
        let none = tempfile::tempdir().expect("a folder");
        let [seg, skin] = weights_reports(none.path());
        assert_eq!(seg.name, format!("eomt/{}", eomt::WEIGHTS_FILE));
        assert_eq!(skin.name, format!("mediapipe/{}", selfie::WEIGHTS_FILE));
        assert_eq!(
            (seg.state, skin.state),
            (WeightsState::Missing, WeightsState::Missing)
        );
        assert_eq!((seg.sha256, skin.sha256), (None, None));

        let good = data(&[
            ("eomt", eomt::WEIGHTS_FILE, b"segmenter", b"segmenter"),
            ("mediapipe", selfie::WEIGHTS_FILE, b"skin", b"skin"),
        ]);
        let [seg, skin] = weights_reports(good.path());
        assert_eq!(
            (seg.state, skin.state),
            (WeightsState::Ready, WeightsState::Ready)
        );
        assert_eq!(seg.sha256.as_deref().map(str::len), Some(64));

        // A file cut short is damaged, with the hash of what is there; a file with no line in the sums is missing.
        let cut = data(&[
            ("eomt", eomt::WEIGHTS_FILE, b"segm", b"segmenter"),
            ("mediapipe", selfie::WEIGHTS_FILE, b"skin", b"skin"),
        ]);
        let [seg, skin] = weights_reports(cut.path());
        assert_eq!(
            (seg.state, skin.state),
            (WeightsState::Damaged, WeightsState::Ready)
        );
        assert!(seg.sha256.is_some());
        std::fs::write(cut.path().join("models/mediapipe/SHA256SUMS"), "").expect("sums");
        assert_eq!(weights_reports(cut.path())[1].state, WeightsState::Missing);
        assert_eq!(WeightsState::Damaged.word(), "damaged");
    }
}
