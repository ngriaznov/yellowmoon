//! `autoset_dump`: the dataset rows of `docs/18-auto-settings.md` section 6, one JSON object per line.
//!
//! ```text
//! YELLOWMOON_DATA=$PWD/data cargo run -p ym-core --release --example autoset_dump -- [--source S] [--licence L] \
//!     [--no-aug] FILE_OR_FOLDER...
//! ```
//!
//! A release build does not look for the repository's `data/` folder, so `YELLOWMOON_DATA` names it.
//!
//! For each ARW file the first row is the photo: its 39 features and key (`ym_core::autoset::photo_features`), the
//! shot's metadata, and the label tables of the camera's embedded JPEG: `g3` (the two quantile tables of the exposure
//! label), `label_ev` (`L`, section 6.1) and, where there is a level, `contrast_fit` (section 6.2). Six augmentation
//! rows follow (`aug_k` = -2, -1, -0.5, 0.5, 1, 2: the same file exposed `2^k` times more, see
//! `ym_core::autoset::scaled_preview`), unless `--no-aug`.
//!
//! The features and the colour come from the photo's session estimates, the ones the app and `ym render` use. One
//! photo is open at a time and the session is closed after each, so memory stays flat. Progress goes to stderr. A
//! file that cannot be processed is reported as `skip <file>: <reason>`, writes no rows, and the run goes on; the exit
//! code is 1 when a file was skipped and 0 otherwise. Nothing is written next to the inputs; the files are read only.
//!
//! The contrast fit renders each photo with the app's default look, Signature (`preferences::DEFAULT_LOOK`), taken from
//! the data folder's look library and assigned the way a fresh app session assigns it. The look is checked when the dump
//! starts: when it is missing or cannot be used the dump exits with an error before it writes anything, so the rows
//! cannot mix looks.

use std::fmt::Write as _;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use ym_color::display::{ColorChain, OutputTarget};
use ym_color::exposure::{EXPOSURE_MAX_EV, EXPOSURE_MIN_EV, PreviewRgb, auto_exposure};
use ym_core::auto::{Preview, build_preview, tables_of};
use ym_core::autoset::camera::{ASPECT_TOLERANCE, adobe_rgb_of, camera_exposure, decode_jpeg};
use ym_core::autoset::image::{SRGB_Y_ROW, encoded_luma, fit_to_preview, rgb_of, to_common};
use ym_core::autoset::labels::quantile_table;
use ym_core::autoset::{photo_features, scaled_preview};
use ym_core::data_dir::find_data_dir;
use ym_core::pipeline::color::{GradeRequest, LookProvider, color_plan, grade_plan};
use ym_core::pipeline::{DisplayChoice, PhotoSource};
use ym_core::preferences::DEFAULT_LOOK;
use ym_core::session::{OpenOptions, PhotoView, Session, SessionEnv};
use ym_core::settings::ExposureSetting;
use ym_looks::library::{Library, LookId};
use ym_raw::decode::RawImage;
use ym_raw::makernote::CreativeStyle;
use ym_raw::metadata::RawMetadata;

/// The augmentation exposures of section 6, in stops.
const AUGMENTATIONS: [f64; 6] = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];
/// The contrast grid of section 6.2.
const CONTRAST_GRID: [f64; 9] = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0];
/// The quantile probabilities of the contrast tables (section 6.2).
const CONTRAST_P: [f64; 7] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];
/// The name of the exposure tables' grid in the rows (`training/autoset/rows.py`).
const GRID_NAME: &str = "linspace(0.001, 0.999, 999)";
/// How long one photo may take to develop before it is skipped.
const DEVELOP_TIMEOUT: Duration = Duration::from_secs(900);
const USAGE: &str = "usage: autoset_dump [--source S] [--licence L] [--no-aug] FILE_OR_FOLDER...  (JSON lines on stdout)";

/// The command line.
struct Options {
    inputs: Vec<PathBuf>,
    source: Option<String>,
    licence: Option<String>,
    augment: bool,
}

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Options {
            inputs: Vec::new(),
            source: None,
            licence: None,
            augment: true,
        };
        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--source" => options.source = Some(args.next().ok_or("--source needs a value")?),
                "--licence" => {
                    options.licence = Some(args.next().ok_or("--licence needs a value")?)
                }
                "--no-aug" => options.augment = false,
                "-h" | "--help" => return Err(USAGE.to_owned()),
                other if other.starts_with("--") => {
                    return Err(format!("unknown option {other}\n{USAGE}"));
                }
                _ => options.inputs.push(PathBuf::from(arg)),
            }
        }
        if options.inputs.is_empty() {
            return Err(USAGE.to_owned());
        }
        Ok(options)
    }
}

/// The look library of the data folder, and the look every row is rendered with: the default look of a fresh app session
/// (`preferences::DEFAULT_LOOK`, Signature). The data folder is the library's root, so the look comes from the repository
/// and from nothing else: there are no imported looks and no preferences to read.
struct DatasetLooks {
    library: Library,
    default: LookId,
}

impl DatasetLooks {
    /// Opens the library of `data_root` and checks the default look: it must be in the built-in folder, enabled, and
    /// baked into a usable transform.
    ///
    /// # Errors
    ///
    /// A message for the user when the look is missing or cannot be used.
    fn open(data_root: &Path) -> Result<Self, String> {
        let builtin = data_root.join("looks").join("builtin");
        let library = Library::open(data_root, builtin.is_dir().then_some(builtin.as_path()))
            .map_err(|error| {
                format!(
                    "cannot open the look library in {} ({error})",
                    data_root.display()
                )
            })?;
        let default = LookId::parse(DEFAULT_LOOK)
            .map_err(|_| format!("the default look {DEFAULT_LOOK} is not a valid look id"))?;
        let file = builtin.join(format!("{}.json", default.slug()));
        match library.get(&default) {
            None => {
                return Err(format!(
                    "The default look {default} is missing or cannot be read: {} is not in the data folder. The \
                     dataset needs it for the contrast fit, so restore it from the repository, then run the dump again.",
                    file.display()
                ));
            }
            Some(look) if !look.is_enabled() => {
                return Err(format!(
                    "The default look {default} cannot be used: its file {} is missing or changed. Restore it from the \
                     repository, then run the dump again.",
                    file.display()
                ));
            }
            Some(_) => {}
        }
        if let Err(error) = library.transform(&default) {
            return Err(format!(
                "The default look {default} cannot be used ({error}). Restore {} from the repository, then run the dump \
                 again.",
                file.display()
            ));
        }
        Ok(Self { library, default })
    }
}

fn main() -> ExitCode {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let data_root = match find_data_dir() {
        Ok(root) => root,
        Err(error) => {
            eprintln!("cannot find the data folder: {error}");
            return ExitCode::from(2);
        }
    };
    // Before any file is read: every row of the dataset must be rendered with the same look.
    let looks = match DatasetLooks::open(&data_root) {
        Ok(looks) => looks,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let (files, problems) = collect_files(&options.inputs);
    for problem in &problems {
        eprintln!("{problem}");
    }
    let env = match SessionEnv::load(&data_root) {
        Ok(env) => env.with_default_look(looks.default.clone()),
        Err(error) => {
            eprintln!("cannot load the data folder: {error}");
            return ExitCode::from(2);
        }
    };
    let session = Session::new(env);
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut skipped = problems.len();
    for file in &files {
        let started = Instant::now();
        match dump_file(&session, file, &options, &looks) {
            Ok(lines) => {
                for line in &lines {
                    if writeln!(out, "{line}").is_err() {
                        eprintln!("cannot write the rows (stdout is closed)");
                        return ExitCode::from(2);
                    }
                }
                eprintln!(
                    "ok {} rows {} {:.1} s",
                    file.display(),
                    lines.len(),
                    started.elapsed().as_secs_f64()
                );
            }
            Err(reason) => {
                skipped += 1;
                eprintln!("skip {}: {reason}", file.display());
            }
        }
        session.close();
    }
    if out.flush().is_err() {
        eprintln!("cannot write the rows (stdout is closed)");
        return ExitCode::from(2);
    }
    if skipped > 0 {
        eprintln!("{skipped} file(s) skipped");
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// The ARW files the inputs name: a folder gives its ARW files, sorted; a file is taken as it is. A path that does not
/// exist comes back as a skip message.
fn collect_files(inputs: &[PathBuf]) -> (Vec<PathBuf>, Vec<String>) {
    let mut files = Vec::new();
    let mut problems = Vec::new();
    for input in inputs {
        if input.is_dir() {
            match std::fs::read_dir(input) {
                Ok(entries) => {
                    let mut found: Vec<PathBuf> = entries
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| {
                            p.is_file()
                                && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw"))
                        })
                        .collect();
                    found.sort();
                    files.extend(found);
                }
                Err(error) => problems.push(format!(
                    "skip {}: the folder cannot be read ({error})",
                    input.display()
                )),
            }
        } else if input.is_file() {
            files.push(input.clone());
        } else {
            problems.push(format!("skip {}: the path does not exist", input.display()));
        }
    }
    (files, problems)
}

// ---------------------------------------------------------------------------------------------
// One photo
// ---------------------------------------------------------------------------------------------

/// The JSON lines of one file (the photo, then its augmentations), or the reason it cannot be processed.
fn dump_file(
    session: &Session,
    path: &Path,
    options: &Options,
    looks: &DatasetLooks,
) -> Result<Vec<String>, String> {
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let bytes =
        std::fs::read(path).map_err(|error| format!("the file cannot be read ({error})"))?;
    let sha256 = hex(&Sha256::digest(&bytes));
    let opened = session.open(
        std::slice::from_ref(&path.to_path_buf()),
        OpenOptions::default(),
    );
    if !session.wait_idle(DEVELOP_TIMEOUT) {
        return Err("it did not finish developing in 15 minutes".to_owned());
    }
    let Some(&id) = opened.added.first() else {
        return Err("it was not opened as a photo (not an ARW file, or unreadable)".to_owned());
    };
    let view = session.view(id).map_err(|error| error.to_string())?;
    if view.settings.look != looks.default {
        return Err(format!(
            "it was opened with the look {} instead of the default look {}",
            view.settings.look, looks.default
        ));
    }
    let raw = session.raw_image(id).map_err(|error| error.to_string())?;
    let preview = build_preview(&raw).map_err(|error| error.to_string())?;
    let meta: &RawMetadata = &view.meta;
    let color = &view.estimates.color;
    let tables = tables_of(&view.profile);
    let base = photo_features(
        meta,
        &preview,
        color,
        tables,
        view.estimates.exposure.meter.mean_luminance,
    )
    .map_err(|error| error.to_string())?;
    let labels = labels_of(&name, &bytes, &view, &raw, &preview, &looks.library)?;

    let identity = Identity {
        file: name.clone(),
        sha256,
        source: options.source.clone(),
        licence: options.licence.clone(),
        body: (!meta.model.is_empty()).then(|| meta.model.clone()),
        capture_time: meta.capture_time.as_ref().map(|t| {
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                t.year, t.month, t.day, t.hour, t.minute, t.second
            )
        }),
        meta: shot_meta(meta),
    };
    let mut lines = vec![row_json(
        &identity,
        0.0,
        &base.values,
        base.key,
        Some(labels),
    )?];

    if options.augment {
        let g0 = color.initial_gains().map_err(|error| error.to_string())?;
        let metering = color
            .metering_chain(tables)
            .map_err(|error| error.to_string())?;
        for k in AUGMENTATIONS {
            let scaled = scaled_preview(&preview, k);
            let balanced = scaled.metered(g0);
            let meter = auto_exposure(
                &PreviewRgb {
                    width: scaled.width,
                    height: scaled.height,
                    rgb: &balanced,
                    clip_mask: Some(&scaled.clip),
                },
                metering.matrix(),
            )
            .map_err(|error| error.to_string())?;
            // The same scene exposed `2^k` times more: the exposure time is multiplied by `2^k`, which lowers the
            // brightness value of the shot (`ShotInfo::bv`) by `k`, so `key_bv` is unchanged, as section 3 says.
            let mut shot = meta.clone();
            shot.exposure_time_s = shot.exposure_time_s.map(|t| t * k.exp2());
            let features = photo_features(&shot, &scaled, color, tables, meter.mean_luminance)
                .map_err(|error| error.to_string())?;
            lines.push(row_json(
                &identity,
                k,
                &features.values,
                features.key,
                None,
            )?);
        }
    }
    Ok(lines)
}

// ---------------------------------------------------------------------------------------------
// The label tables (section 6.1 and 6.2)
// ---------------------------------------------------------------------------------------------

/// The label part of a file's row: the tables, `L` and the contrast fit.
#[derive(Serialize)]
struct LabelsOut {
    g3: Option<G3Out>,
    label_ev: Option<f64>,
    contrast_fit: Option<ContrastOut>,
}

impl LabelsOut {
    /// A file with no label: no table, no level, no fit.
    fn none() -> Self {
        Self {
            g3: None,
            label_ev: None,
            contrast_fit: None,
        }
    }
}

#[derive(Serialize)]
struct G3Out {
    grid: &'static str,
    scene_q: Vec<f64>,
    jpeg_q: Vec<f64>,
}

#[derive(Serialize)]
struct ContrastOut {
    grid: Vec<f64>,
    p: Vec<f64>,
    ours: Vec<Vec<f64>>,
    jpeg: Vec<f64>,
}

/// The label tables of a photo, from its embedded JPEG. A file without a usable JPEG keeps its features and gets no
/// label (the note says why); a failure of the colour chain itself fails the file. The level and the tables are
/// `ym_core::autoset::camera::camera_exposure`, which the app uses too.
fn labels_of(
    name: &str,
    bytes: &[u8],
    view: &PhotoView,
    raw: &Arc<RawImage>,
    preview: &Preview,
    looks: &dyn LookProvider,
) -> Result<LabelsOut, String> {
    let meta: &RawMetadata = &view.meta;
    let Some(jpeg) = embedded_jpeg(bytes, meta) else {
        eprintln!("note {name}: no embedded JPEG, so no exposure label");
        return Ok(LabelsOut::none());
    };
    // Our scene luminance is metered on the metered, g0-balanced preview through the metering chain at EV 0.
    let g0 = view
        .estimates
        .initial_gains()
        .map_err(|error| error.to_string())?;
    let metering = view
        .estimates
        .color
        .metering_chain(tables_of(&view.profile))
        .map_err(|error| error.to_string())?;
    let balanced = preview.metered(g0);
    let camera = match camera_exposure(
        jpeg,
        adobe_rgb_of(meta),
        preview,
        &balanced,
        metering.matrix(),
    ) {
        Ok(Some(camera)) => camera,
        Ok(None) => {
            eprintln!("note {name}: no usable pixel for the exposure label");
            return Ok(LabelsOut::none());
        }
        Err(error) => {
            eprintln!("note {name}: {error}, so no exposure label");
            return Ok(LabelsOut::none());
        }
    };
    let (pw, ph) = (preview.width, preview.height);
    let contrast_fit = match camera.level {
        Some(level) => {
            let (mut jpeg_display, common) = jpeg_display_luma(jpeg, (pw, ph))?;
            Some(contrast_fit(
                name,
                view,
                raw,
                &balanced,
                FitContext {
                    looks,
                    preview: (pw, ph),
                    common,
                },
                level,
                &mut jpeg_display,
            )?)
        }
        None => None,
    };
    Ok(LabelsOut {
        g3: Some(G3Out {
            grid: GRID_NAME,
            scene_q: camera.scene_q,
            jpeg_q: camera.jpeg_q,
        }),
        label_ev: camera.level,
        contrast_fit,
    })
}

/// The JPEG's display luma (`encoded_luma`, the Rec.709 weights on the encoded values) fitted to the preview and at the
/// common size, as the contrast fit reads it (section 6.2), with that size.
fn jpeg_display_luma(
    jpeg: &[u8],
    preview: (usize, usize),
) -> Result<(Vec<f64>, (usize, usize)), String> {
    let (jw, jh, jrgb) = decode_jpeg(jpeg).map_err(|error| error.to_string())?;
    let (rgb, w, h) = fit_to_preview(&jrgb, jw, jh, preview, ASPECT_TOLERANCE);
    let common = (preview.0.min(w), preview.1.min(h));
    let display = to_common(encoded_luma(&rgb), w, h, common.0, common.1);
    Ok((display, common))
}

/// What the contrast fit needs besides the photo: the look provider (the dataset's default look) and the sizes of the
/// luminance images: the preview's, and the common size both images are brought to.
#[derive(Clone, Copy)]
struct FitContext<'a> {
    looks: &'a dyn LookProvider,
    preview: (usize, usize),
    common: (usize, usize),
}

/// The contrast table (section 6.2): the display luma quantiles of our render at each contrast of the grid, at exposure
/// `L`, and those of the JPEG, at the common size.
fn contrast_fit(
    name: &str,
    view: &PhotoView,
    raw: &Arc<RawImage>,
    balanced: &[f32],
    context: FitContext<'_>,
    level: f64,
    jpeg_display: &mut Vec<f64>,
) -> Result<ContrastOut, String> {
    let ((pw, ph), (cw, ch)) = (context.preview, context.common);
    // The exposure control takes -4 to +4 stops; a level beyond that is rendered at the limit, and the note says so.
    let ev = level.clamp(EXPOSURE_MIN_EV, EXPOSURE_MAX_EV);
    if ev != level {
        eprintln!(
            "note {name}: the level {level:.3} EV is rendered at {ev} EV for the contrast fit (the control's limit)"
        );
    }
    let mut settings = view.settings.clone();
    settings.exposure = ExposureSetting::manual_ev(ev);
    // The chain of the sRGB export, with the photo's own look (the default look of a fresh app session, Signature):
    // ACES 2, the estimated white lift, the sRGB output. A look that cannot be applied falls back to Neutral with a
    // warning; that is a failed file, never a silent mix of looks.
    let source = PhotoSource::from_view(view, Arc::clone(raw)).with_settings(settings);
    let colour = color_plan(&source).map_err(|error| error.to_string())?;
    let display = DisplayChoice::Aces2;
    let request = GradeRequest {
        target: OutputTarget::Srgb,
        dither: false,
        display: &display,
        looks: context.looks,
    };
    let plan = grade_plan(&source, &colour, &request).map_err(|error| error.to_string())?;
    if !plan.warnings.is_empty() {
        return Err(format!(
            "the look could not be applied ({})",
            plan.warnings.join("; ")
        ));
    }
    let base = plan.chain.params().clone();

    let mut ours = Vec::with_capacity(CONTRAST_GRID.len());
    for &contrast in &CONTRAST_GRID {
        let mut params = base.clone();
        params.grade.contrast = contrast;
        let chain = ColorChain::new(&params).map_err(|error| error.to_string())?;
        let mut luma = to_common(display_luma(&chain, balanced, pw * ph), pw, ph, cw, ch);
        ours.push(quantile_table(&mut luma, &CONTRAST_P));
    }
    let jpeg = quantile_table(jpeg_display, &CONTRAST_P);
    Ok(ContrastOut {
        grid: CONTRAST_GRID.to_vec(),
        p: CONTRAST_P.to_vec(),
        ours,
        jpeg,
    })
}

/// The display luma (Rec.709 weights on the encoded output) of every pixel of the metered preview through the chain,
/// NaN where the camera values are not finite.
fn display_luma(chain: &ColorChain, balanced: &[f32], pixels: usize) -> Vec<f64> {
    (0..pixels)
        .into_par_iter()
        .map(|i| match rgb_of(balanced, i) {
            None => f64::NAN,
            Some(rgb) => {
                let encoded = chain.apply_pixel(rgb);
                SRGB_Y_ROW[0] * encoded[0] + SRGB_Y_ROW[1] * encoded[1] + SRGB_Y_ROW[2] * encoded[2]
            }
        })
        .collect()
}

/// The embedded JPEG's bytes, from the file (`meta.preview`), when they lie inside the file.
fn embedded_jpeg<'a>(bytes: &'a [u8], meta: &RawMetadata) -> Option<&'a [u8]> {
    let jpeg = meta.preview?;
    let start = usize::try_from(jpeg.offset).ok()?;
    let end = start.checked_add(usize::try_from(jpeg.length).ok()?)?;
    bytes.get(start..end)
}

// ---------------------------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------------------------

/// What the rows of one file share.
struct Identity {
    file: String,
    sha256: String,
    source: Option<String>,
    licence: Option<String>,
    body: Option<String>,
    capture_time: Option<String>,
    meta: MetaOut,
}

/// The shot's metadata, as the dataset keeps it (`docs/18` section 6, `meta`; the spellings of
/// `training/autoset/labels.py::is_flagged`).
#[derive(Serialize)]
struct MetaOut {
    dro: Option<String>,
    creative_style: Option<String>,
    hdr: Option<String>,
    flash: Option<u8>,
    metering: Option<String>,
    ec: Option<f64>,
    iso: Option<u32>,
    exposure_time_s: Option<f64>,
    f_number: Option<f64>,
    bv_exif: Option<f64>,
    exposure_program: Option<String>,
    scene_mode: Option<String>,
    scene_capture_type: Option<String>,
}

fn shot_meta(meta: &RawMetadata) -> MetaOut {
    MetaOut {
        dro: meta.dro_mode.as_ref().and_then(snake),
        creative_style: meta.creative_style.as_ref().and_then(|style| match style {
            CreativeStyle::Other(text) => Some(text.to_lowercase()),
            named => snake(named),
        }),
        // The HDR tag is not parsed yet (`docs/18` section 6): null for every file.
        hdr: None,
        flash: meta.flash.map(|flash| u8::from(flash.fired)),
        metering: meta.metering_mode.as_ref().and_then(snake),
        ec: meta.exposure_comp_ev,
        iso: meta.iso,
        exposure_time_s: meta.exposure_time_s,
        f_number: meta.f_number,
        bv_exif: meta.brightness_value,
        exposure_program: meta.exposure_program.as_ref().and_then(snake),
        scene_mode: meta.scene_mode.as_ref().and_then(snake),
        scene_capture_type: meta.scene_capture_type.as_ref().and_then(snake),
    }
}

/// The snake_case name of an enum value as its serde form gives it: a unit variant is its name, and a variant with a
/// number (`Other(7)`, `Level(3)`) is `name_number`.
fn snake<T: Serialize>(value: &T) -> Option<String> {
    match serde_json::to_value(value).ok()? {
        serde_json::Value::String(name) => Some(name),
        serde_json::Value::Object(map) => {
            let (name, number) = map.into_iter().next()?;
            Some(format!("{name}_{number}"))
        }
        _ => None,
    }
}

/// One JSON object of the dataset, with the keys in the order of the spec's schema. The label keys are left out of
/// augmentation rows (`labels` is `None`).
#[derive(Serialize)]
struct Row<'a> {
    file: &'a str,
    sha256: &'a str,
    source: Option<&'a str>,
    licence: Option<&'a str>,
    body: Option<&'a str>,
    capture_time: Option<&'a str>,
    aug_k: f64,
    features: Vec<f64>,
    key: f64,
    meta: &'a MetaOut,
    #[serde(flatten)]
    labels: Option<LabelsOut>,
}

fn row_json(
    identity: &Identity,
    aug_k: f64,
    features: &[f64],
    key: f64,
    labels: Option<LabelsOut>,
) -> Result<String, String> {
    let row = Row {
        file: &identity.file,
        sha256: &identity.sha256,
        source: identity.source.as_deref(),
        licence: identity.licence.as_deref(),
        body: identity.body.as_deref(),
        capture_time: identity.capture_time.as_deref(),
        aug_k,
        features: features.to_vec(),
        key,
        meta: &identity.meta,
        labels,
    };
    serde_json::to_string(&row).map_err(|error| error.to_string())
}

/// Lower-case hexadecimal of bytes.
fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        })
}
