//! The Film node in the core (`docs/24-film.md` sections 3.8, 3.9 and 4.5): the catalogue of stocks and papers
//! the settings are checked against, and the engine that builds a node's baked chain and runs its spatial
//! passes on tiles ([`FilmEngine`]).
//!
//! **The catalogue** is `data/film/params/names.json`. A copy is built into the program (it is our own
//! Apache-2.0 file, a few KiB), so that the settings can be checked by the same rule in the app, in `ym` and in
//! the tests whatever folder they find their data in; the engine and `ym film list` read the shipped file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ym_film::bake::{BakeSettings, Baker, ENGINE_VERSION};
use ym_film::spatial::{FilmFrame, SpatialModel, SpatialSettings};
use ym_film::{
    CREDIT_LINE, FilmChain, FilmError, FilmParams, Names, Output, PaperChoice, StockKind,
    StockName, TableSpectra,
};

use crate::settings::FilmNode;

/// The `names.json` built into the program.
const BUILT_IN_NAMES: &str = include_str!("../../../data/film/params/names.json");

/// The credit line of the film data, for `ym film list` and the export's XMP.
pub const FILM_CREDIT: &str = CREDIT_LINE;

/// The stocks and papers the Film node knows (`names.json`).
#[derive(Debug, Clone)]
pub struct Catalogue {
    names: Names,
}

/// The catalogue built into the program, or `None` when its copy of `names.json` does not parse (a build error
/// that a test catches; the settings are then not checked against it).
#[must_use]
pub fn catalogue() -> Option<&'static Catalogue> {
    static C: OnceLock<Option<Catalogue>> = OnceLock::new();
    C.get_or_init(|| {
        serde_json::from_str::<Names>(BUILT_IN_NAMES)
            .ok()
            .map(|names| Catalogue { names })
    })
    .as_ref()
}

impl Catalogue {
    /// The catalogue of a data folder (`<data>/film/params/names.json`).
    ///
    /// # Errors
    ///
    /// [`FilmError`] when the file cannot be read or parsed.
    pub fn load(data_root: &Path) -> Result<Self, FilmError> {
        Ok(Self {
            names: Names::load(&data_root.join("film").join("params"))?,
        })
    }

    /// Every row, hidden ones included.
    #[must_use]
    pub fn rows(&self) -> &[StockName] {
        &self.names.stocks
    }

    /// The version of the profile data (`0.3.2`).
    #[must_use]
    pub fn profile_version(&self) -> &str {
        self.names.profile_version()
    }

    /// The number of profiles (every row, hidden ones included).
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.names.stocks.len()
    }

    /// The row for `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&StockName> {
        self.names.get(id)
    }

    /// Whether `id` is a camera film that is offered (a negative or a slide, not hidden).
    #[must_use]
    pub fn is_film(&self, id: &str) -> bool {
        self.get(id).is_some_and(|s| {
            !s.hidden
                && matches!(
                    s.kind,
                    StockKind::ColourNegative | StockKind::CineNegative | StockKind::Slide
                )
        })
    }

    /// Whether `id` is a still paper that is offered.
    #[must_use]
    pub fn is_paper(&self, id: &str) -> bool {
        self.get(id)
            .is_some_and(|s| !s.hidden && s.kind == StockKind::Paper)
    }

    /// Whether `id` is a slide film.
    #[must_use]
    pub fn is_slide(&self, id: &str) -> bool {
        self.get(id).is_some_and(|s| s.kind == StockKind::Slide)
    }

    /// The paper a film is printed on when the settings say `paper: null` (`docs/24` section 4.5): the stock's own
    /// `print` of `names.json`, or, where that is a cine print film (hidden until `docs/24` section 6.4 is settled), the
    /// still paper that stands in for it. `None` for a slide (it has no print) and for what is not a film.
    #[must_use]
    pub fn matching_paper(&self, id: &str) -> Option<String> {
        let row = self.get(id)?;
        match row.kind {
            StockKind::ColourNegative | StockKind::CineNegative => {}
            _ => return None,
        }
        match row.print.as_deref() {
            Some(p) if self.is_paper(p) => Some(p.to_owned()),
            _ => Some(ym_film::chain::FALLBACK_PAPER.to_owned()),
        }
    }

    /// The plain refusal for a stock that is not a camera film that is offered.
    ///
    /// # Errors
    ///
    /// The sentence of `docs/24` section 4.5.
    pub fn check_stock(&self, id: &str) -> Result<(), String> {
        if self.is_film(id) {
            Ok(())
        } else {
            Err(format!(
                "Unknown film stock \"{id}\". Run \"ym film list\" to see the stocks."
            ))
        }
    }

    /// The plain refusal for a paper that is not a still paper that is offered.
    ///
    /// # Errors
    ///
    /// The sentence for papers.
    pub fn check_paper(&self, id: &str) -> Result<(), String> {
        if self.is_paper(id) {
            Ok(())
        } else {
            Err(format!(
                "Unknown film paper \"{id}\". Run \"ym film list\" to see the papers."
            ))
        }
    }
}

/// What `ym doctor` says about the film data of a data folder: how many profiles it holds, their version, and the
/// file whose checksum does not match, if any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmDataStatus {
    /// Profiles with a file in `film/spektrafilm/profiles/`.
    pub profiles: usize,
    /// The data version of `names.json`.
    pub data_version: String,
    /// The first problem found (a missing file, a checksum that differs), in plain words.
    pub problem: Option<String>,
}

/// Reads the film data of `data_root` for the doctor.
///
/// # Errors
///
/// [`FilmError`] when `names.json` cannot be read.
pub fn data_status(data_root: &Path) -> Result<FilmDataStatus, FilmError> {
    let catalogue = Catalogue::load(data_root)?;
    let spektra = data_root.join("film").join("spektrafilm");
    let profiles = catalogue
        .rows()
        .iter()
        .filter(|r| {
            spektra
                .join("profiles")
                .join(format!("{}.json", r.id))
                .is_file()
        })
        .count();
    let problem = ym_film::profile::verify_checksums(&spektra)
        .err()
        .map(|e| e.to_string());
    Ok(FilmDataStatus {
        profiles,
        data_version: catalogue.profile_version().to_owned(),
        problem,
    })
}

// ---------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------

/// A built chain is cached by what it was made from.
#[derive(Clone, PartialEq, Eq, Debug)]
struct ChainKey {
    stock: String,
    paper: Option<String>,
    output: u8,
    /// The bits of the exposure, print exposure and the three lights.
    knobs: [u64; 5],
}

/// A film frame is cached by everything that makes it: the node's film, print, halation and grain, and the output.
#[derive(Clone, PartialEq, Debug)]
struct FrameKey {
    film: crate::settings::FilmStockSettings,
    print: crate::settings::PrintSettings,
    halation: crate::settings::HalationSettings,
    grain: crate::settings::GrainSettings,
    output: u8,
}

/// How many film frames the engine keeps (cheap: they share the tables).
const FRAME_CACHE: usize = 16;

/// How many built chains the engine keeps (a chain holds the profiles and the calibration, not tables).
const CHAIN_CACHE: usize = 8;
/// How many spatial models the engine keeps.
const MODEL_CACHE: usize = 24;

/// The Film node's engine for one data folder: the upsampling table, the baker with its caches (T1, curves,
/// T3a, T3b′, `docs/24` section 3.8) and the chains and spatial models built from them. Cheap to share; the
/// caches are in memory only (hard rule 4).
pub struct FilmEngine {
    data_root: PathBuf,
    spectra: Arc<TableSpectra>,
    baker: Baker,
    chains: Mutex<Vec<(ChainKey, Arc<FilmChain>)>>,
    models: Mutex<Vec<(String, Arc<SpatialModel>)>>,
    frames: Mutex<Vec<(FrameKey, FilmFrame)>>,
    /// The tables of a node's baked film that were built, for tests: (chains, models).
    built: Mutex<(u64, u64)>,
}

impl std::fmt::Debug for FilmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilmEngine")
            .field("data_root", &self.data_root)
            .finish_non_exhaustive()
    }
}

/// The output the baked film is made for (`docs/24` section 2.2: the scan is in the output's primaries).
/// A Rec.2020 export uses the Display P3 tables.
#[must_use]
pub fn film_output(target: ym_color::display::OutputTarget) -> Output {
    use ym_color::display::OutputTarget as T;
    match target {
        T::Srgb => Output::Srgb,
        T::Viewport | T::DisplayP3 | T::Rec2020 => Output::DisplayP3,
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FilmEngine {
    /// The engine of a data folder, shared by every caller in the process (one baker, so the caches serve the
    /// viewport, the thumbnails and the exports alike).
    ///
    /// # Errors
    ///
    /// [`FilmError`] (its plain sentence) when the folder has no film data.
    pub fn shared(data_root: &Path) -> Result<Arc<Self>, FilmError> {
        static ENGINES: OnceLock<Mutex<BTreeMap<PathBuf, Arc<FilmEngine>>>> = OnceLock::new();
        let engines = ENGINES.get_or_init(|| Mutex::new(BTreeMap::new()));
        if let Some(e) = lock(engines).get(data_root) {
            return Ok(Arc::clone(e));
        }
        let engine = Arc::new(Self::open(data_root)?);
        Ok(Arc::clone(
            lock(engines)
                .entry(data_root.to_path_buf())
                .or_insert(engine),
        ))
    }

    /// A new engine with empty caches.
    ///
    /// # Errors
    ///
    /// [`FilmError`] when the upsampling table cannot be read.
    pub fn open(data_root: &Path) -> Result<Self, FilmError> {
        let spectra = TableSpectra::load(&data_root.join("film").join("upsampling"))?;
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let spectra_id = format!("upsampling-{}", spectra.size());
        Ok(Self {
            data_root: data_root.to_path_buf(),
            spectra: Arc::new(spectra),
            baker: Baker::new(spectra_id.clone(), threads),
            chains: Mutex::new(Vec::new()),
            models: Mutex::new(Vec::new()),
            frames: Mutex::new(Vec::new()),
            built: Mutex::new((0, 0)),
        })
    }

    /// The data folder.
    #[must_use]
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// The baker (its counts say how many tables were built).
    #[must_use]
    pub fn baker(&self) -> &Baker {
        &self.baker
    }

    /// How many chains and spatial models were built (not fetched from the caches).
    #[must_use]
    pub fn built(&self) -> (u64, u64) {
        *lock(&self.built)
    }

    fn chain(
        &self,
        node: &FilmNode,
        output: Output,
        paper: Option<&str>,
    ) -> Result<Arc<FilmChain>, FilmError> {
        let knobs = [
            node.film.exposure_ev.to_bits(),
            node.print.exposure_ev.to_bits(),
            node.print.lights.y.to_bits(),
            node.print.lights.m.to_bits(),
            node.print.lights.c.to_bits(),
        ];
        let key = ChainKey {
            stock: node.film.stock.clone(),
            paper: paper.map(str::to_owned),
            output: output as u8,
            knobs,
        };
        {
            let mut chains = lock(&self.chains);
            if let Some(at) = chains.iter().position(|(k, _)| *k == key) {
                let item = chains.remove(at);
                let chain = Arc::clone(&item.1);
                chains.push(item);
                return Ok(chain);
            }
        }
        let params = FilmParams {
            exposure_ev: node.film.exposure_ev,
            print_exposure_stops: node.print.exposure_ev,
            light_yellow_cc: node.print.lights.y,
            light_magenta_cc: node.print.lights.m,
            light_cyan_cc: node.print.lights.c,
            output,
            ..FilmParams::default()
        };
        let choice = match paper {
            Some(id) => PaperChoice::Id(id.to_owned()),
            None => PaperChoice::Matching,
        };
        let chain = Arc::new(FilmChain::new(
            &self.data_root,
            &node.film.stock,
            &choice,
            params,
            Arc::clone(&self.spectra) as Arc<dyn ym_film::Spectra>,
        )?);
        lock(&self.built).0 += 1;
        let mut chains = lock(&self.chains);
        chains.push((key, Arc::clone(&chain)));
        while chains.len() > CHAIN_CACHE {
            chains.remove(0);
        }
        Ok(chain)
    }

    fn model(&self, stock: &str) -> Result<Arc<SpatialModel>, FilmError> {
        {
            let mut models = lock(&self.models);
            if let Some(at) = models.iter().position(|(k, _)| k == stock) {
                let item = models.remove(at);
                let model = Arc::clone(&item.1);
                models.push(item);
                return Ok(model);
            }
        }
        let model = Arc::new(SpatialModel::load(&self.data_root, stock)?);
        lock(&self.built).1 += 1;
        let mut models = lock(&self.models);
        models.push((stock.to_owned(), Arc::clone(&model)));
        while models.len() > MODEL_CACHE {
            models.remove(0);
        }
        Ok(model)
    }

    /// The node's film for a render into `target`: the baked colour passes (cached by what they are made of,
    /// `docs/24` section 3.8) and the spatial model of the stock, with the node's spatial settings.
    ///
    /// # Errors
    ///
    /// [`FilmError`] for a stock whose data is missing or damaged.
    pub fn frame_for(
        &self,
        node: &FilmNode,
        target: ym_color::display::OutputTarget,
    ) -> Result<FilmFrame, FilmError> {
        let output = film_output(target);
        let key = FrameKey {
            film: node.film.clone(),
            print: node.print.clone(),
            halation: node.halation,
            grain: node.grain,
            output: output as u8,
        };
        {
            let mut frames = lock(&self.frames);
            if let Some(at) = frames.iter().position(|(k, _)| *k == key) {
                let item = frames.remove(at);
                let frame = item.1.clone();
                frames.push(item);
                return Ok(frame);
            }
        }
        let frame = self.build_frame(node, output)?;
        let mut frames = lock(&self.frames);
        frames.push((key, frame.clone()));
        while frames.len() > FRAME_CACHE {
            frames.remove(0);
        }
        Ok(frame)
    }

    fn build_frame(&self, node: &FilmNode, output: Output) -> Result<FilmFrame, FilmError> {
        // A slide has no paper, whatever the settings hold.
        let paper = if self.is_slide(&node.film.stock) {
            None
        } else {
            node.print.paper.as_deref()
        };
        let chain = self.chain(node, output, paper)?;
        let baked = self.baker.bake(&chain, &BakeSettings::of(&chain))?;
        let model = self.model(&node.film.stock)?;
        let settings = SpatialSettings {
            halation_amount: node.halation.amount / 100.0,
            halation_size: node.halation.size / 100.0,
            grain_amount: node.grain.amount / 100.0,
            grain_size: node.grain.size / 100.0,
            grain_seed: node.grain.seed,
            diffusion_size: 1.0,
        };
        Ok(FilmFrame::with_shared_model(baked, model, settings))
    }

    #[allow(clippy::unused_self)]
    fn is_slide(&self, stock: &str) -> bool {
        catalogue().is_some_and(|c| c.is_slide(stock))
    }
}

/// The text of `ym:FilmData` (`docs/24` section 6.2): for each live Film node that does something, its stock and the
/// paper it is printed on, then the version of the profile data, the engine's version and the credit; `None` for a
/// photo without one. A slide has no paper. The data version is the one of the built-in copy of `names.json`.
#[must_use]
pub fn film_data_of(settings: &crate::settings::DevelopSettings) -> Option<String> {
    let plan =
        crate::pipeline::color::node_plan(&settings.nodes, ym_color::plan::Stop::Output).ok()?;
    let catalogue = catalogue()?;
    let nodes: Vec<String> = plan
        .node_ids()
        .into_iter()
        .filter_map(|id| settings.nodes.film(id))
        .filter(|f| !f.is_inert())
        .map(|f| {
            let paper = if catalogue.is_slide(&f.film.stock) {
                None
            } else {
                f.print
                    .paper
                    .clone()
                    .or_else(|| catalogue.matching_paper(&f.film.stock))
            };
            match paper {
                Some(p) => format!("{} on {p}", f.film.stock),
                None => f.film.stock.clone(),
            }
        })
        .collect();
    if nodes.is_empty() {
        return None;
    }
    Some(format!(
        "{}; profiles {}; engine {ENGINE_VERSION}; {FILM_CREDIT}",
        nodes.join(", "),
        catalogue.profile_version()
    ))
}
