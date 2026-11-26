//! The fit loop: every photo of the chosen set against every reference, through the core's own API.
//!
//! For each photo the tool opens it in a core as the app does (default settings), adds **one Reference Grade node after Main**, and
//! for each reference loads the still through RG-05's loader (`Core::reference_choose_file`), runs the fit of RG-06
//! (`Core::reference_fit`), and measures the photo as it is and the photo with the fit on **every pixel** of the analysis picture
//! (`Core::reference_eval_photo`: the fit's own picture, chain, scene and masks, with the whole frame in place of the sample). The
//! fit's patch is undone before the next reference, so every fit starts from the photo as the person would have it.
//!
//! Results go to `records.jsonl` as they are made (a killed run is resumed with `--resume`), `failures.jsonl` for a fit that failed
//! and `masks.jsonl` for the element-mask check against label maps.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions as FsOpen};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use ym_color::refgrade::N_PARAMS;
use ym_color::refgrade::measure::{PixelScene, Px, class};
use ym_color::refgrade::search::{CandidateRenderer, Start};
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::PhotoSource;
use ym_core::pipeline::eval_scene::{EvalLabels, PhotoScene};
use ym_core::pipeline::fit::FitPhoto;
use ym_core::reference_fit::{ChainCandidates, FitEvent};
use ym_core::session::{OpenOptions, PhotoId, PhotoStatus};
use ym_core::settings::{DevelopSettings, ReferenceGradeNode, SettingsPatch};
use ym_gpu::queue::CancelToken;

use crate::measure::{HARMS, Pic, SceneRef, harms, sample_ab, transfers};
use crate::report::read_jsonl;
use crate::split::{SplitEntry, read_split, reference_files};

/// The id of the Reference Grade node the tool adds (Main is node 1).
pub const NODE: u32 = 2;

/// What a run does.
#[derive(Debug, Clone)]
pub struct RunOpts {
    /// `split.json`.
    pub split: PathBuf,
    /// The set to run: `dev`, `test` or `all`.
    pub set: String,
    /// The folder of reference stills.
    pub references: PathBuf,
    /// Where `records.jsonl` and the rest are written.
    pub out: PathBuf,
    /// `(index, count)`: this process takes the photos whose position in the chosen list is `index` modulo `count`.
    pub shard: (usize, usize),
    /// Threads of this process (the fit's, the render's and the decode pool's).
    pub threads: usize,
    /// Keep only the photos with these file stems (empty: all).
    pub only: Vec<String>,
    /// Stop after this many photos of the shard (for a trial).
    pub max_photos: Option<usize>,
    /// The `data/` folder (default: as `ym` finds it).
    pub data: Option<PathBuf>,
    /// Skip the fits already in `records.jsonl`.
    pub resume: bool,
    /// Compare the product's element map with each photo's label map when the split names one.
    pub mask_check: bool,
    /// Render each photo's first fit through the real pipeline and compare it with the measured picture.
    pub audit: bool,
    /// The scene the fits see (`docs/34` section 16.10: the ablation of RG-10). The product's own unless the run says otherwise.
    pub scene: SceneMode,
    /// For [`SceneMode::PocLabels`]: the folder of the references' label maps (`<dir>/labels/<folder of the still>__<stem>.png`).
    pub reference_labels: Option<PathBuf>,
}

/// Which scene a run's fits see. Only the evaluation can change it (`ym-core`'s `eval-scene` feature, which only this tool turns on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneMode {
    /// The product as it is: the segmenter on the Input's picture, the Input's picture as the guide.
    Product,
    /// A1: the proof of concept's label maps for the photo and the reference, the photo's default render through the look as the guide.
    PocLabels,
    /// A2: the product's segmenter on the photo's default render through the look, that render as the guide.
    DefaultRender,
}

impl SceneMode {
    /// The name in the records and on the command line.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            SceneMode::Product => "product",
            SceneMode::PocLabels => "poc-labels",
            SceneMode::DefaultRender => "default-render",
        }
    }

    /// Reads a name.
    ///
    /// # Errors
    ///
    /// A name that is none of the three.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "product" => Ok(SceneMode::Product),
            "poc-labels" => Ok(SceneMode::PocLabels),
            "default-render" => Ok(SceneMode::DefaultRender),
            other => bail!("--scene is product, poc-labels or default-render, not {other}"),
        }
    }
}

/// The reference's label map (`<dir>/labels/<folder of the still>__<stem>.png`, the folder after the still's symlinks are followed) and
/// the bars the product's loader cuts off the still.
pub(crate) fn reference_labels(dir: &Path, still: &Path) -> Result<EvalLabels> {
    let real = fs::canonicalize(still).unwrap_or_else(|_| still.to_path_buf());
    let group = real
        .parent()
        .and_then(|d| d.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = real
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file = dir.join("labels").join(format!("{group}__{stem}.png"));
    let (width, height, elements) = read_label_png(&file)?;
    let crop = ym_core::reference::eval_bars_of_file(still)
        .map_err(|e| anyhow!("{}: {e}", still.display()))?;
    Ok(EvalLabels {
        width,
        height,
        elements,
        crop,
    })
}

/// The photos of a run: the split's entries of the set, filtered, then this shard's share.
///
/// # Errors
///
/// A split that cannot be read.
pub fn selected(opts: &RunOpts) -> Result<Vec<SplitEntry>> {
    let split = read_split(&opts.split)?;
    let chosen: Vec<SplitEntry> = split
        .into_iter()
        .filter(|e| opts.set == "all" || e.set == opts.set)
        .filter(|e| opts.only.is_empty() || opts.only.iter().any(|x| *x == e.name()))
        .collect();
    let (i, n) = opts.shard;
    let mut mine: Vec<SplitEntry> = chosen
        .into_iter()
        .enumerate()
        .filter(|(k, _)| k % n == i)
        .map(|(_, e)| e)
        .collect();
    if let Some(m) = opts.max_photos {
        mine.truncate(m);
    }
    Ok(mine)
}

/// The patch the app sends for "Add node, Reference Grade" on a photo whose settings are `now`: the node (id [`NODE`]) added after Main
/// and the Output's links moved to it, **leaving everything else as the photo has it** (the Input, Main, the Output and its look, Auto's
/// settings). The photo's settings are the core's own (`Core::session.view`: what a photo opens with, the Output's look included), so the
/// tool builds no graph of its own and cannot drift from the app (it did once: `docs/34` section 16.10.1).
///
/// # Errors
///
/// A graph with no Output node, or a patch the core does not read.
pub(crate) fn add_node_patch(now: &DevelopSettings) -> Result<SettingsPatch> {
    let doc = serde_json::to_value(now)?;
    let correctors = doc["nodes"]["items"].as_array().map_or(1, |items| {
        items
            .iter()
            .filter(|n| n["id"] != "input" && n["id"] != "output")
            .count()
    });
    let links: Vec<Value> = {
        let mut moved: Vec<Value> = Vec::new();
        for l in &now.nodes.links {
            let mut l = serde_json::to_value(l)?;
            if l["to"]["node"] == "output" {
                l["to"]["node"] = json!(NODE);
            }
            moved.push(l);
        }
        moved.push(
            json!({ "from": { "node": NODE, "port": "picture" }, "to": { "node": "output", "port": "picture" } }),
        );
        moved
    };
    Ok(serde_json::from_value(json!({
        "nodes": { NODE.to_string(): { "kind": "reference_grade", "pos": [correctors as f64, 0.0] } },
        "links": links,
    }))?)
}

/// Opens the core of a run: the data folder, a private app folder, `threads` threads, no thumbnails.
pub(crate) fn open_core(opts: &RunOpts, app: &Path) -> Result<Core> {
    let data = match &opts.data {
        Some(d) => d.clone(),
        None => ym_core::data_dir::find_data_dir()?,
    };
    let mut options = CoreOptions::new(data, app.to_path_buf());
    options.thumbnails = false;
    options.decode_threads = opts.threads.max(1);
    options.render.threads = opts.threads.max(1);
    Core::open(options).map_err(|e| anyhow!("the core would not open: {e}"))
}

/// Opens one photo and waits for it to develop.
pub(crate) fn open_photo(core: &Core, raw: &Path) -> Result<PhotoId> {
    let out = core
        .session
        .open(&[raw.to_path_buf()], OpenOptions::default());
    let id = *out
        .added
        .first()
        .ok_or_else(|| anyhow!("{}: the photo did not open", raw.display()))?;
    if !core.session.wait_idle(Duration::from_secs(900)) {
        bail!("{}: the photo did not finish developing", raw.display());
    }
    match core.session.status(id) {
        Some(PhotoStatus::Ready) => Ok(id),
        other => bail!("{}: the photo is not ready ({other:?})", raw.display()),
    }
}

/// The class names of the element-mask check, with the element numbers each one is.
pub const MASK_ELEMENTS: [(&str, &[u8]); 6] = [
    ("sky", &[class::SKY]),
    ("vegetation", &[class::VEGETATION]),
    ("water", &[class::WATER]),
    ("ground", &[class::GROUND]),
    ("built", &[class::ARCHITECTURE, class::INTERIOR]),
    ("person", &[class::PERSON]),
];

/// Reads an 8-bit indexed label png: `(width, height, element number per pixel)`.
///
/// # Errors
///
/// A file that is not an 8-bit indexed png.
pub fn read_label_png(path: &Path) -> Result<(usize, usize, Vec<u8>)> {
    let bytes = fs::read(path).with_context(|| format!("{}", path.display()))?;
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::IDENTITY);
    let mut reader = dec.read_info()?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| anyhow!("label png too large"))?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    if info.bit_depth != png::BitDepth::Eight || info.color_type != png::ColorType::Indexed {
        bail!("{}: expected an 8-bit indexed label png", path.display());
    }
    buf.truncate(w * h);
    Ok((w, h, buf))
}

/// Nearest-neighbour resize of a label map to `dw` by `dh` (the proof of concept's: the centre of each destination pixel).
#[must_use]
pub fn resize_labels(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy = ((((y as f64 + 0.5) / dh as f64) * sh as f64) as usize).min(sh - 1);
        for x in 0..dw {
            let sx = ((((x as f64 + 0.5) / dw as f64) * sw as f64) as usize).min(sw - 1);
            out[y * dw + x] = src[sy * sw + sx];
        }
    }
    out
}

/// The agreement of the product's element map with a label map of the same size.
///
/// `equal` is the share of pixels with the same element number; for each of six elements `agree` is the share of pixels where
/// membership is the same and `iou` the intersection over the union of the two masks.
#[must_use]
pub fn mask_agreement(product: &[u8], labels: &[u8]) -> Value {
    let n = product.len().max(1) as f64;
    let equal = product.iter().zip(labels).filter(|(a, b)| a == b).count() as f64 / n;
    let mut per = serde_json::Map::new();
    for (name, ids) in MASK_ELEMENTS {
        let (mut agree, mut inter, mut union) = (0usize, 0usize, 0usize);
        for (a, b) in product.iter().zip(labels) {
            let (x, y) = (ids.contains(a), ids.contains(b));
            agree += usize::from(x == y);
            inter += usize::from(x && y);
            union += usize::from(x || y);
        }
        per.insert(
            name.into(),
            json!({
                "agree": agree as f64 / n,
                "iou": if union == 0 { Value::Null } else { json!(inter as f64 / union as f64) },
                "product_share": product.iter().filter(|a| ids.contains(a)).count() as f64 / n,
                "labels_share": labels.iter().filter(|a| ids.contains(a)).count() as f64 / n,
            }),
        );
    }
    json!({ "equal": equal, "elements": per })
}

/// One fit's record, in the proof of concept's format.
#[allow(clippy::too_many_arguments)]
fn measure_fit(
    entry: &SplitEntry,
    reference: &str,
    p0: &[Px],
    fit: &[Px],
    scene: &PixelScene,
    refpx: &[Px],
    refscene: &PixelScene,
    p0_sample: &[[f64; 2]],
    seconds: f64,
    start: Start,
) -> Value {
    let p0p = Pic {
        px: p0,
        scene: Some(SceneRef {
            class: &scene.class,
            skin: &scene.skin,
        }),
    };
    let fitp = Pic {
        px: fit,
        scene: p0p.scene,
    };
    let refp = Pic {
        px: refpx,
        scene: Some(SceneRef {
            class: &refscene.class,
            skin: &refscene.skin,
        }),
    };
    let h = harms(&p0p, &fitp, &refp);
    let t = transfers(&p0p, &fitp, &refp, p0_sample);
    let mut harm_json = serde_json::Map::new();
    for (k, id) in HARMS.iter().enumerate() {
        harm_json.insert((*id).into(), json!({ "value": h[k].0, "flag": h[k].1 }));
    }
    json!({
        "photo": entry.name(), "set": entry.set, "tags": entry.tags, "primary_tag": entry.primary_tag,
        "reference": reference, "fit_seconds": seconds, "harms": harm_json,
        "any_harm": h.iter().any(|x| x.1),
        "T1": t[0], "T2": t[1], "T3": t[2],
        "search_start": match start { Start::Neutral => "neutral", Start::Look => "look" },
    })
}

/// The photo as the person sees it with the fit applied (the real pipeline, whole, 8-bit), against the measured fit picture `px`
/// (the fit's own render of every analysis pixel): the means of L\*, a\* and b\* of both and the mean CIE76 distance at matching
/// places (nearest pixel of the larger picture).
fn audit_render(
    core: &Core,
    id: PhotoId,
    px: &[Px],
    eval: &FitPhoto,
    cancel: &CancelToken,
) -> Result<Value> {
    let src = PhotoSource::from_session(&core.session, id).map_err(|e| anyhow!("{e}"))?;
    let pic = core
        .renderer
        .advisor_picture(&src, cancel)
        .map_err(|e| anyhow!("{e}"))?;
    let conv = ym_color::refgrade::measure::LabConv::new();
    let (w, h) = (eval.width, eval.height);
    let (mut ma, mut mb) = ([0.0f64; 3], [0.0f64; 3]);
    let mut de = 0.0;
    for y in 0..h {
        let sy = (((y as f64 + 0.5) / h as f64) * pic.height as f64) as usize;
        for x in 0..w {
            let sx = (((x as f64 + 0.5) / w as f64) * pic.width as f64) as usize;
            let o = 3 * (sy.min(pic.height - 1) * pic.width + sx.min(pic.width - 1));
            let q = conv.px_of_codes([
                f64::from(pic.rgb[o]) / 255.0,
                f64::from(pic.rgb[o + 1]) / 255.0,
                f64::from(pic.rgb[o + 2]) / 255.0,
            ]);
            let m = &px[y * w + x];
            for c in 0..3 {
                ma[c] += m.lab[c];
                mb[c] += q.lab[c];
            }
            de += ((m.lab[0] - q.lab[0]).powi(2)
                + (m.lab[1] - q.lab[1]).powi(2)
                + (m.lab[2] - q.lab[2]).powi(2))
            .sqrt();
        }
    }
    let n = (w * h) as f64;
    Ok(json!({
        "measured_mean_lab": ma.map(|v| v / n), "real_mean_lab": mb.map(|v| v / n),
        "mean_delta_e76": de / n, "real_size": [pic.width, pic.height],
    }))
}

fn append(path: &Path, v: &Value) -> Result<()> {
    let mut f = FsOpen::new().create(true).append(true).open(path)?;
    writeln!(f, "{v}")?;
    Ok(())
}

/// Runs the shard.
///
/// # Errors
///
/// A split or a folder that cannot be read, or a core that will not open. A photo or a fit that fails is logged to
/// `failures.jsonl` and the run goes on.
pub fn run(opts: &RunOpts) -> Result<()> {
    if opts.shard.1 == 0 || opts.shard.0 >= opts.shard.1 {
        bail!("--shard i/n needs i < n");
    }
    fs::create_dir_all(&opts.out)?;
    let photos = selected(opts)?;
    let refs = reference_files(&opts.references)?;
    if refs.is_empty() {
        bail!("{}: no references", opts.references.display());
    }
    let ref_names: Vec<String> = refs
        .iter()
        .map(|p| {
            p.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
        .collect();
    let records_path = opts.out.join("records.jsonl");
    let failures_path = opts.out.join("failures.jsonl");
    let masks_path = opts.out.join("masks.jsonl");
    let done: BTreeSet<(String, String)> = if opts.resume && records_path.is_file() {
        read_jsonl(&records_path)?
            .iter()
            .map(|r| {
                (
                    r["photo"].as_str().unwrap_or("").to_string(),
                    r["reference"].as_str().unwrap_or("").to_string(),
                )
            })
            .collect()
    } else {
        if records_path.exists() {
            bail!(
                "{} exists: use --resume to carry on, or another folder",
                records_path.display()
            );
        }
        BTreeSet::new()
    };
    let masks_done: BTreeSet<String> = if opts.resume && masks_path.is_file() {
        read_jsonl(&masks_path)?
            .iter()
            .map(|r| r["photo"].as_str().unwrap_or("").to_string())
            .collect()
    } else {
        BTreeSet::new()
    };
    // The process's pool: this many threads everywhere (a second call in one process, as the tests make, keeps the first pool).
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads.max(1))
        .build_global();
    let app = tempfile::tempdir()?;
    let core = open_core(opts, app.path())?;
    let scene_ok = core.renderer.scene_model().is_some_and(|m| m.available());
    let skin_ok = core
        .renderer
        .scene_model()
        .is_some_and(|m| m.skin_available());
    eprintln!(
        "refgrade-eval: shard {}/{}, {} photos x {} references, {} threads; scene model {}, skin {}",
        opts.shard.0,
        opts.shard.1,
        photos.len(),
        refs.len(),
        opts.threads,
        if scene_ok { "present" } else { "MISSING" },
        if skin_ok { "present" } else { "MISSING" },
    );
    if !scene_ok {
        bail!(
            "the scene model is missing from the data folder: the fit would run without elements"
        );
    }
    fs::write(
        opts.out.join("scene.json"),
        serde_json::to_string_pretty(&json!({ "scene_model": scene_ok, "skin": skin_ok }))?,
    )?;
    let audit = opts.audit;
    let started = Instant::now();
    let total_fits = photos.len() * refs.len();
    let mut fits_done = 0usize;
    let cancel = CancelToken::new();
    for (pi, entry) in photos.iter().enumerate() {
        let name = entry.name();
        let todo: Vec<usize> = (0..refs.len())
            .filter(|&k| !done.contains(&(name.clone(), ref_names[k].clone())))
            .collect();
        let want_mask = opts.mask_check
            && !masks_done.contains(&name)
            && entry.label_file().is_some_and(|f| f.is_file());
        if todo.is_empty() && !want_mask {
            fits_done += refs.len();
            continue;
        }
        eprintln!("== {name} ({}) [{}/{}]", entry.set, pi + 1, photos.len());
        let t_photo = Instant::now();
        let outcome = (|| -> Result<()> {
            let id = open_photo(&core, &entry.raw)?;
            let result = (|| -> Result<()> {
                let now = core.session.view(id)?.settings;
                core.update_settings(id, &add_node_patch(&now)?)?;
                // The ablation: another scene for this photo's fits (never set in a run of the product as it is).
                match opts.scene {
                    SceneMode::Product => {}
                    SceneMode::PocLabels => {
                        let file = entry
                            .label_file()
                            .filter(|f| f.is_file())
                            .ok_or_else(|| anyhow!("{name}: no label map for the ablation"))?;
                        let (width, height, elements) = read_label_png(&file)?;
                        core.renderer
                            .set_eval_photo_scene(Some(PhotoScene::Labels(EvalLabels {
                                width,
                                height,
                                elements,
                                crop: [0; 4],
                            })));
                    }
                    SceneMode::DefaultRender => {
                        core.renderer
                            .set_eval_photo_scene(Some(PhotoScene::DefaultRender));
                    }
                }
                let t0 = Instant::now();
                let eval: FitPhoto = core
                    .reference_eval_photo(id, NODE, &cancel)
                    .map_err(|e| anyhow!("{e}"))?;
                let scene_seconds = t0.elapsed().as_secs_f64();
                let base: ReferenceGradeNode = core
                    .session
                    .view(id)?
                    .settings
                    .nodes
                    .reference_grade(NODE)
                    .ok_or_else(|| anyhow!("the node is gone"))?
                    .clone();
                let mut candidates = ChainCandidates::new(&eval, &base);
                let mut p0 = vec![Px::default(); candidates.sample_len()];
                candidates
                    .render(&[0.0; N_PARAMS], &mut p0)
                    .map_err(|_| anyhow!("the photo could not be rendered"))?;
                let p0_sample = sample_ab(&p0);
                if want_mask && let Some(file) = entry.label_file() {
                    match read_label_png(&file) {
                        Ok((lw, lh, labels)) => {
                            let at = resize_labels(&labels, lw, lh, eval.width, eval.height);
                            let mut m = mask_agreement(&eval.scene.class, &at);
                            m["photo"] = json!(name);
                            m["set"] = json!(entry.set);
                            m["size"] = json!([eval.width, eval.height]);
                            m["labels_size"] = json!([lw, lh]);
                            append(&masks_path, &m)?;
                        }
                        Err(e) => append(
                            &failures_path,
                            &json!({ "photo": name, "what": "labels", "error": e.to_string() }),
                        )?,
                    }
                }
                for &k in &todo {
                    let rname = &ref_names[k];
                    let one = (|| -> Result<Value> {
                        if opts.scene == SceneMode::PocLabels {
                            let dir = opts.reference_labels.as_deref().ok_or_else(|| {
                                anyhow!("--scene poc-labels needs --reference-labels")
                            })?;
                            core.renderer
                                .set_eval_reference_labels(Some(reference_labels(dir, &refs[k])?));
                        }
                        core.reference_choose_file(id, NODE, &refs[k], Some(rname))
                            .map_err(|e| anyhow!("{e}"))?;
                        let t = Instant::now();
                        let mut prep: Option<f64> = None;
                        let fit = core
                            .reference_fit(id, NODE, &cancel, &mut |e| {
                                if prep.is_none() && matches!(e, FitEvent::Search { .. }) {
                                    prep = Some(t.elapsed().as_secs_f64());
                                }
                            })
                            .map_err(|e| anyhow!("{e}"))?;
                        let seconds = t.elapsed().as_secs_f64();
                        let refm = core
                            .reference_measured(id, NODE, &cancel)
                            .map_err(|e| anyhow!("{e}"))?;
                        let mut px = vec![Px::default(); candidates.sample_len()];
                        candidates
                            .render(&fit.result.x, &mut px)
                            .map_err(|_| anyhow!("the fit could not be rendered"))?;
                        let mut rec = measure_fit(
                            entry,
                            rname,
                            &p0,
                            &px,
                            &eval.scene,
                            &refm.px,
                            &refm.scene,
                            &p0_sample,
                            seconds,
                            fit.result.start,
                        );
                        rec["scene_mode"] = json!(opts.scene.name());
                        rec["objective"] = json!(fit.result.total);
                        rec["identity"] = json!(fit.result.identity);
                        rec["evals"] = json!(fit.result.evals);
                        rec["scene_seconds"] = json!(scene_seconds);
                        rec["prep_seconds"] = json!(prep);
                        // The first fit of each photo is also rendered through the real pipeline, as the person would see it, and
                        // compared with the measured picture: the audit that the numbers describe what the product makes.
                        if audit && k == todo[0] {
                            rec["audit"] = audit_render(&core, id, &px, &eval, &cancel)?;
                        }
                        // Every fit starts from the photo as the person has it: the fit's patch is taken back.
                        core.update_settings(id, &fit.undo)?;
                        core.reference_clear(id, NODE);
                        Ok(rec)
                    })();
                    match one {
                        Ok(rec) => {
                            append(&records_path, &rec)?;
                            eprintln!(
                                "   {rname}: {:.1} s, start {}, any harm {}, T1 {}, T2 {}, T3 {}",
                                rec["fit_seconds"].as_f64().unwrap_or(0.0),
                                rec["search_start"].as_str().unwrap_or("-"),
                                rec["any_harm"],
                                fmt_t(&rec["T1"]),
                                fmt_t(&rec["T2"]),
                                fmt_t(&rec["T3"]),
                            );
                        }
                        Err(e) => {
                            eprintln!("   {rname}: FAILED: {e}");
                            core.reference_clear(id, NODE);
                            append(
                                &failures_path,
                                &json!({ "photo": name, "reference": rname, "what": "fit", "error": e.to_string() }),
                            )?;
                        }
                    }
                    fits_done += 1;
                    write_progress(
                        &opts.out,
                        opts.shard,
                        pi,
                        photos.len(),
                        fits_done,
                        total_fits,
                        started,
                    )?;
                }
                Ok(())
            })();
            core.renderer.set_eval_photo_scene(None);
            core.renderer.set_eval_reference_labels(None);
            core.remove_photos(&[id]);
            result
        })();
        if let Err(e) = outcome {
            eprintln!("   FAILED: {e:#}");
            append(
                &failures_path,
                &json!({ "photo": name, "what": "photo", "error": format!("{e:#}") }),
            )?;
            fits_done += todo.len();
        }
        eprintln!("   photo took {:.0} s", t_photo.elapsed().as_secs_f64());
        write_progress(
            &opts.out,
            opts.shard,
            pi + 1,
            photos.len(),
            fits_done,
            total_fits,
            started,
        )?;
    }
    // The shard's own metrics.json (sorted), as the proof of concept's runs wrote.
    let mut records = read_jsonl(&records_path).unwrap_or_default();
    crate::report::write_summaries(&opts.out, "shard", &mut records)?;
    let _ = File::create(opts.out.join("DONE"));
    Ok(())
}

/// Renders each photo of the shard's list through the real pipeline with its default settings (no node), whole, as an 8-bit png in
/// `opts.out`: the product's default picture, for a side-by-side check against another tool's default renders of the same photos.
///
/// # Errors
///
/// A split that cannot be read or a core that will not open; a photo that fails is skipped with a line on stderr.
pub fn default_renders(opts: &RunOpts) -> Result<()> {
    fs::create_dir_all(&opts.out)?;
    let photos = selected(opts)?;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads.max(1))
        .build_global();
    let app = tempfile::tempdir()?;
    let core = open_core(opts, app.path())?;
    let cancel = CancelToken::new();
    for entry in &photos {
        let name = entry.name();
        let one = (|| -> Result<()> {
            let id = open_photo(&core, &entry.raw)?;
            let r = (|| -> Result<()> {
                let src =
                    PhotoSource::from_session(&core.session, id).map_err(|e| anyhow!("{e}"))?;
                let pic = core
                    .renderer
                    .advisor_picture(&src, &cancel)
                    .map_err(|e| anyhow!("{e}"))?;
                let png =
                    ym_core::pipeline::thumbs::encode_png_rgb8(pic.width, pic.height, &pic.rgb)
                        .map_err(|e| anyhow!("{e}"))?;
                fs::write(opts.out.join(format!("{name}.png")), png)?;
                Ok(())
            })();
            core.remove_photos(&[id]);
            r
        })();
        match one {
            Ok(()) => eprintln!("{name}: rendered"),
            Err(e) => eprintln!("{name}: FAILED: {e:#}"),
        }
    }
    Ok(())
}

fn fmt_t(v: &Value) -> String {
    v.as_f64().map_or("-".into(), |x| format!("{x:.2}"))
}

fn write_progress(
    out: &Path,
    shard: (usize, usize),
    photos_done: usize,
    photos: usize,
    fits_done: usize,
    fits: usize,
    started: Instant,
) -> Result<()> {
    let el = started.elapsed().as_secs_f64();
    let eta = if fits_done == 0 {
        f64::NAN
    } else {
        el / fits_done as f64 * (fits - fits_done) as f64
    };
    fs::write(
        out.join("progress.txt"),
        format!(
            "shard {}/{}: photos {photos_done}/{photos}, fits {fits_done}/{fits}, elapsed {:.0} s, eta {:.0} s\n",
            shard.0, shard.1, el, eta
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_resize_takes_the_centre_of_each_pixel() {
        // 4 x 2 to 2 x 1: columns 1 and 3 (the centres 1.0 and 3.0 floor to 1 and 3), row 1 (centre 1.0).
        let src = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(resize_labels(&src, 4, 2, 2, 1), vec![5, 7]);
        // The identity.
        assert_eq!(resize_labels(&src, 4, 2, 4, 2), src.to_vec());
        // Larger.
        assert_eq!(resize_labels(&[1, 2], 2, 1, 4, 1), vec![1, 1, 2, 2]);
    }

    #[test]
    fn the_agreement_counts_equal_pixels_and_each_elements_overlap() {
        // sky = 1, vegetation = 2, architecture = 5, interior = 6.
        let product = [1u8, 1, 2, 5, 6, 0, 7, 7];
        let labels = [1u8, 2, 2, 6, 6, 0, 7, 8];
        let m = mask_agreement(&product, &labels);
        assert!((m["equal"].as_f64().unwrap_or(0.0) - 5.0 / 8.0).abs() < 1e-12);
        // Sky: product {0, 1}, labels {0}: they agree on 7 of 8 pixels, IoU 1/2.
        assert!((m["elements"]["sky"]["agree"].as_f64().unwrap_or(0.0) - 7.0 / 8.0).abs() < 1e-12);
        assert!((m["elements"]["sky"]["iou"].as_f64().unwrap_or(0.0) - 0.5).abs() < 1e-12);
        // Built is architecture or interior: pixels 3 and 4 in both.
        assert!((m["elements"]["built"]["iou"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-12);
        // Water is in neither: no union, no IoU, full agreement.
        assert!(m["elements"]["water"]["iou"].is_null());
        assert_eq!(m["elements"]["water"]["agree"], 1.0);
        assert!((m["elements"]["person"]["iou"].as_f64().unwrap_or(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_shard_takes_every_nth_photo_of_the_filtered_list() {
        let dir = tempfile::tempdir().expect("dir");
        let items: Vec<Value> = (0..10)
            .map(|i| {
                json!({ "raw": format!("/x/p{i}.ARW"), "set": if i < 7 { "dev" } else { "test" }, "tags": ["a"] })
            })
            .collect();
        let path = dir.path().join("split.json");
        fs::write(&path, json!({ "items": items }).to_string()).expect("write");
        let mk = |shard, set: &str, only: Vec<String>| RunOpts {
            split: path.clone(),
            set: set.into(),
            references: dir.path().to_path_buf(),
            out: dir.path().to_path_buf(),
            shard,
            threads: 1,
            only,
            max_photos: None,
            data: None,
            resume: false,
            mask_check: false,
            audit: false,
            scene: SceneMode::Product,
            reference_labels: None,
        };
        let names = |o: &RunOpts| -> Vec<String> {
            selected(o)
                .expect("selected")
                .iter()
                .map(SplitEntry::name)
                .collect()
        };
        assert_eq!(names(&mk((0, 3), "dev", vec![])), ["p0", "p3", "p6"]);
        assert_eq!(names(&mk((1, 3), "dev", vec![])), ["p1", "p4"]);
        assert_eq!(names(&mk((2, 3), "dev", vec![])), ["p2", "p5"]);
        assert_eq!(names(&mk((0, 1), "test", vec![])), ["p7", "p8", "p9"]);
        assert_eq!(
            names(&mk((0, 1), "all", vec!["p4".into(), "p8".into()])),
            ["p4", "p8"]
        );
        // Every photo is in exactly one shard.
        let mut all: Vec<String> = (0..3)
            .flat_map(|i| names(&mk((i, 3), "all", vec![])))
            .collect();
        all.sort();
        assert_eq!(all.len(), 10);
    }

    /// The tool's graph for a photo is the photo's own settings plus the node, and the Output keeps its look (the RG-10 run once lost it).
    #[test]
    fn the_tools_graph_is_the_photos_default_settings_plus_the_node() {
        // A photo as the app opens it: the default settings with the default look on the Output.
        let mut now = DevelopSettings::default();
        let look: SettingsPatch = serde_json::from_value(
            json!({ "nodes": { "output": { "look": "builtin:signature" } } }),
        )
        .expect("a patch");
        now.apply_patch(&look, None).expect("the look");
        assert_eq!(now.output().look.to_string(), "builtin:signature");
        let mut after = now.clone();
        after
            .apply_patch(&add_node_patch(&now).expect("a patch"), None)
            .expect("the node");
        // The Output keeps the photo's look; so does everything else of the Input, Main and the Output.
        assert_eq!(after.output().look, now.output().look);
        assert_eq!(after.output().look.to_string(), "builtin:signature");
        assert!(after.nodes.reference_grade(NODE).is_some());
        let json = |s: &DevelopSettings, id: &str| {
            serde_json::to_value(s).expect("json")["nodes"]["items"]
                .as_array()
                .expect("items")
                .iter()
                .find(|n| n["id"] == id)
                .cloned()
                .expect("a node")
        };
        assert_eq!(json(&after, "input"), json(&now, "input"));
        assert_eq!(json(&after, "output")["look"], json(&now, "output")["look"]);
        let main = |s: &DevelopSettings| {
            serde_json::to_value(s).expect("json")["nodes"]["items"]
                .as_array()
                .expect("items")
                .iter()
                .find(|n| n["id"] == 1)
                .cloned()
                .expect("main")
        };
        assert_eq!(main(&after), main(&now));
        // Main, then the node, then the Output.
        let links = serde_json::to_value(&after.nodes.links).expect("links");
        let has = |from: Value, to: Value| {
            links
                .as_array()
                .expect("links")
                .iter()
                .any(|l| l["from"]["node"] == from && l["to"]["node"] == to)
        };
        assert!(has(json!("input"), json!(1)));
        assert!(has(json!(1), json!(NODE)));
        assert!(has(json!(NODE), json!("output")));
        assert!(!has(json!(1), json!("output")));
    }
}
