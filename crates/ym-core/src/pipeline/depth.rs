//! The Depth mask's model and what the renderer tells its listener (`docs/23-nodes.md` sections 3.10a and 4.8,
//! `docs/algorithms/06-masks.md` sections 5.8 and 8).
//!
//! The model ([`DepthModel`]) is `ym_ml::depth::DepthAnything` with its weights, loaded the first time something needs
//! it. The renderer keeps one ([`RenderOptions::depth_model`](super::RenderOptions)); `mask_kinds` offers Depth when it is
//! there. What runs when a photo has a Depth row is in `graph/depth.rs`: the worker in the background, the map the photo
//! keeps, `sample_depth`, `focus_plane` and the source of the row's raster `S0`.
//!
//! A model can also be a **fake** ([`DepthModel::fake`]): a function from the picture to a small map. The tests of the
//! worker, the cache, the geometry, the random graphs and the app-and-`ym` parity use it, so they need no weights.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ym_ml::depth::{self, DepthAnything, WEIGHTS_FILE};
use ym_ml::model::MlError;

use crate::settings::NO_DEPTH_MODEL_SENTENCE;

use super::RenderError;

/// The version of the weights file format the model reads (`format` `yellowmoon-depth`, `version` `1`), part of the map's key.
pub const DEPTH_FORMAT_VERSION: u32 = 1;

/// What the depth worker is doing for a photo, as the renderer's listener hears it
/// ([`Renderer::set_depth_listener`](super::Renderer::set_depth_listener)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepthPhase {
    /// The worker has work for the photo: the picture and the network are on their way, or about to be. The rows say
    /// "Finding…" (`viewport_busy.finding`).
    Started,
    /// The worker ended its work for the photo (after a `Started`). A frame that selected nothing for want of a map should
    /// be drawn again.
    Finished {
        /// The plain sentence of the failure ("The Depth mask can't be made: …"), or `None` when the map is ready.
        failed: Option<String>,
    },
}

/// What a fake model is given: the picture the network would see (`width x height`, `[y][x][rgb]`, 8 bits) and a question that
/// says whether the run was cancelled. It answers a map `(width, height, values)` or the text of its failure.
pub type FakeDepth = dyn Fn(usize, usize, &[u8], &dyn Fn() -> bool) -> Result<(usize, usize, Vec<f32>), String>
    + Send
    + Sync;

/// Why Focus plane has no band (`focus_plane`'s `reason`, `docs/23` section 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusReason {
    /// The file has no focus point: an older body, a panorama (the maker note's point is `0 0`), or a file that is not from a
    /// Sony camera. Answered at once, without waiting for the map.
    NoFocusPoint,
    /// The depth map could not be made.
    MapFailed,
}

impl FocusReason {
    /// The reason's wire name: `no_focus_point` or `map_failed`.
    #[must_use]
    pub fn wire(self) -> &'static str {
        match self {
            Self::NoFocusPoint => "no_focus_point",
            Self::MapFailed => "map_failed",
        }
    }
}

/// The answer of `focus_plane` (`docs/23` section 9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocusPlane {
    /// The band around the plane the camera focused on, two decimals, as Pick depth's.
    Band {
        /// Where the band starts, 0 to 1.
        near: f64,
        /// Where the band ends, 0 to 1.
        far: f64,
        /// The point is exactly the frame's centre, which is also the value Sony writes when it does not know where it
        /// focused: the card says "The camera may not have recorded where it focused. Check the result."
        exact_centre: bool,
    },
    /// No band, and why.
    None(FocusReason),
}

enum Net {
    /// Depth Anything V2-Small, read from the model folder when first needed.
    Network {
        dir: PathBuf,
        state: Mutex<NetState>,
    },
    /// A function that stands in for it (tests).
    Fake(Arc<FakeDepth>),
}

enum NetState {
    /// The weights have not been read yet.
    Untried,
    /// Loaded.
    Ready(Arc<DepthAnything>),
    /// Could not be read, did not match their checksum or are not the model (the reason is in the log).
    Damaged,
}

/// Depth Anything V2-Small and where its weights are. The weights load the first time they are asked for
/// (on the first map); a file that is damaged takes the model away for the rest of the run
/// ([`DepthModel::available`]).
pub struct DepthModel {
    net: Net,
    /// Names the weights in the map's key: their SHA-256 from `SHA256SUMS`, or the fake's name.
    weights_id: String,
    pool: OnceLock<Option<rayon::ThreadPool>>,
}

impl std::fmt::Debug for DepthModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthModel")
            .field("weights_id", &self.weights_id)
            .field("available", &self.available())
            .finish_non_exhaustive()
    }
}

/// The SHA-256 `dir/SHA256SUMS` lists for the weights file.
fn listed_sha256(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("SHA256SUMS"))
        .ok()?
        .lines()
        .filter_map(|l| l.split_once("  "))
        .find(|(_, name)| name.trim() == WEIGHTS_FILE)
        .map(|(hash, _)| hash.trim().to_owned())
}

impl DepthModel {
    /// The model of a data folder (`<data_root>/models/depth/`), when its weights file and a `SHA256SUMS` that lists it are
    /// there. Nothing is read beyond the sums: the weights load, and are checked against their checksum and their header,
    /// on first use, and a file that fails takes the model away ([`DepthModel::available`]).
    #[must_use]
    pub fn probe(data_root: &Path) -> Option<Arc<Self>> {
        let dir = data_root.join("models").join("depth");
        if !dir.join(WEIGHTS_FILE).is_file() {
            return None;
        }
        let weights_id = listed_sha256(&dir)?;
        Some(Arc::new(Self {
            net: Net::Network {
                dir,
                state: Mutex::new(NetState::Untried),
            },
            weights_id,
            pool: OnceLock::new(),
        }))
    }

    /// A model that is loaded already (tests with synthetic weights).
    #[must_use]
    pub fn from_network(network: Arc<DepthAnything>) -> Arc<Self> {
        let weights_id = network
            .sha256()
            .map_or_else(|| "loaded".to_owned(), str::to_owned);
        Arc::new(Self {
            net: Net::Network {
                dir: PathBuf::new(),
                state: Mutex::new(NetState::Ready(network)),
            },
            weights_id,
            pool: OnceLock::new(),
        })
    }

    /// A model that is a function (tests): it is given the picture the network would see and answers a map. `name` names it
    /// in the map's key, so two fakes that differ do not share maps.
    #[must_use]
    pub fn fake(name: &str, f: Arc<FakeDepth>) -> Arc<Self> {
        Arc::new(Self {
            net: Net::Fake(f),
            weights_id: format!("fake:{name}"),
            pool: OnceLock::new(),
        })
    }

    /// The environment variable that makes [`Core::open`](crate::app::Core::open) use [`DepthModel::fake_default`] in place of
    /// the weights when it is `fake`: the test hook of `ym` and the app (as `YELLOWMOON_TEST_VISION` is for Vision), so that
    /// their parity and their Depth rows can be tested without the 50 MB of weights.
    pub const TEST_ENV: &'static str = "YELLOWMOON_TEST_DEPTH";

    /// A fake whose map is the picture's own tones and rows: nearer toward the bottom of the photo and in its bright parts,
    /// at one cell for every 14 pixels of the network's input (`depth::capped_network_size`), so it has the shape and the
    /// aspect of the real map and varies with the Input's picture (a change of the Input's exposure changes it).
    #[must_use]
    pub fn fake_default() -> Arc<Self> {
        Self::fake(
            "default",
            Arc::new(|w, h, rgb, _| {
                let (nw, nh) = depth::capped_network_size(w, h);
                let (cw, ch) = (nw / depth::PATCH, nh / depth::PATCH);
                let mut d = Vec::with_capacity(cw * ch);
                for cy in 0..ch {
                    for cx in 0..cw {
                        let x = (((cx as f64 + 0.5) / cw as f64) * w as f64) as usize;
                        let y = (((cy as f64 + 0.5) / ch as f64) * h as f64) as usize;
                        let p = &rgb[3 * (y.min(h - 1) * w + x.min(w - 1))..][..3];
                        let luma = (0.2126 * f64::from(p[0])
                            + 0.7152 * f64::from(p[1])
                            + 0.0722 * f64::from(p[2]))
                            / 255.0;
                        let row = (cy as f64 + 0.5) / ch as f64;
                        d.push((row + 0.25 * luma) as f32);
                    }
                }
                Ok((cw, ch, d))
            }),
        )
    }

    /// Names the weights in the map's key (`docs/23` section 4.8: "the weights' SHA-256 and the format version").
    #[must_use]
    pub fn weights_id(&self) -> &str {
        &self.weights_id
    }

    /// The SHA-256 of the weights when they load and match their line of `SHA256SUMS` (the file is read once, and a
    /// truncated or changed one takes the model away: `None`, as for an export). `ym doctor` uses it, so the 50 MB file is
    /// read once for its Masks row and its list of weights. A fake has none.
    #[must_use]
    pub fn checked_sha256(&self) -> Option<String> {
        let net = self.network().ok()?;
        net.sha256().map(str::to_owned)
    }

    /// Whether Depth can be offered: the weights are there and have not been found damaged.
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

    /// The network, loading the weights when this is the first call.
    fn network(&self) -> Result<Arc<DepthAnything>, RenderError> {
        let Net::Network { dir, state } = &self.net else {
            return Err(RenderError::Mask(NO_DEPTH_MODEL_SENTENCE.to_owned()));
        };
        let mut st = state.lock().unwrap_or_else(PoisonError::into_inner);
        match &*st {
            NetState::Ready(net) => return Ok(Arc::clone(net)),
            NetState::Damaged => return Err(RenderError::Mask(NO_DEPTH_MODEL_SENTENCE.to_owned())),
            NetState::Untried => {}
        }
        let started = std::time::Instant::now();
        match DepthAnything::load(dir) {
            Ok(net) => {
                tracing::debug!(
                    ms = started.elapsed().as_secs_f64() * 1000.0,
                    "the Depth model loaded"
                );
                let net = Arc::new(net);
                *st = NetState::Ready(Arc::clone(&net));
                Ok(net)
            }
            Err(e) => {
                tracing::error!(error = %e, "the Depth model could not be loaded");
                *st = NetState::Damaged;
                Err(RenderError::Mask(NO_DEPTH_MODEL_SENTENCE.to_owned()))
            }
        }
    }

    /// Runs `f` on the threads this model keeps for the network: all the cores but one, so that a photo being edited stays
    /// smooth while it runs (the bits do not depend on the number of threads, `docs/algorithms/06` section 8.5).
    fn run<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        let pool = self.pool.get_or_init(|| {
            let cores = std::thread::available_parallelism().map_or(1, usize::from);
            rayon::ThreadPoolBuilder::new()
                .num_threads(cores.saturating_sub(1).max(1))
                .thread_name(|i| format!("yellowmoon-depth-{i}"))
                .build()
                .ok()
        });
        match pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// The relative inverse depth of a picture (`width x height`, `[y][x][rgb]`, 8 bits, sRGB): `(cells across, cells down,
    /// values)`, larger is nearer. `cancel` is asked between the network's blocks and stages.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` says so; [`RenderError::Mask`] with [`NO_DEPTH_MODEL_SENTENCE`] when the
    /// weights cannot be read, and with "The Depth mask can't be made: …" when the network or the fake fails.
    pub(crate) fn map_of(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<(usize, usize, Vec<f32>), RenderError> {
        let failed = |text: &str| {
            RenderError::Mask(format!(
                "The Depth mask can't be made: {}.",
                text.trim_end_matches('.')
            ))
        };
        match &self.net {
            // A fake that fails while it is being cancelled was stopped, as the network is (`MlError::Cancelled`).
            Net::Fake(f) => f(width, height, rgb, cancel).map_err(|e| {
                if cancel() {
                    RenderError::Cancelled
                } else {
                    failed(&e)
                }
            }),
            Net::Network { .. } => {
                let net = self.network()?;
                let unit: Vec<f32> = rgb.iter().map(|&v| f32::from(v) / 255.0).collect();
                self.run(|| {
                    let input = depth::prepare_input(&unit, width, height)?;
                    net.infer_with(&input.data, input.width, input.height, cancel)
                        .map(|m| (m.width, m.height, m.data))
                })
                .map_err(|e| match e {
                    MlError::Cancelled => RenderError::Cancelled,
                    other => failed(&other.to_string()),
                })
            }
        }
    }
}
