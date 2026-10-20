//! `ym info <file…> [--json]`: what the file says about itself and what the automatic
//! development makes of it, without rendering (`docs/10-cli.md`).
//!
//! Each file is read twice over: its metadata (container, EXIF, Sony maker note, SR2: camera,
//! exposure, lens, dimensions, the raw layout and levels, which lens-correction tables it carries,
//! where its preview is) and, when the file can be developed, the estimates the app shows on the
//! Automatic card: the camera profile and why it was chosen, the exposure, white balance and light
//! class, what Apple Auto did (`docs/19` section 8), the noise model, the Auto sharpening and noise
//! reduction, and the look and settings a `render` would start from (it waits for Apple Auto's
//! answers when Apple Auto acts, as `render` does). A file that has metadata but cannot be developed yet (ARW 6.0, a
//! layout without a decoder) still prints its metadata and says why the rest is missing.
//!
//! Text output goes to stdout, one block per file. In `--json` mode every file produces one JSON
//! object on its own line of stdout, also when it fails (`"status": "error"`): the object of
//! `render --json` without `output` and `timings_ms`, plus the metadata blocks. In text mode a
//! failure is reported on stderr as `Error: <path>: <message>`.
//!
//! Exit code: 0 when every file was read, 1 when at least one failed, 2 when the data folder is
//! missing. Usage errors are code 2 and are reported by clap before this module runs.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_core::camera::PhotoCamera;
use ym_core::session::{OpenOptions, PhotoId, PhotoView};
use ym_core::settings::ResolvedSettings;
use ym_core::version::PIPELINE_VERSION;
use ym_raw::error::RawError;
use ym_raw::layout::{DecodeKind, Rect};
use ym_raw::metadata::{LensTags, LevelSource, RawMetadata, probe};

use crate::cli::{AppleAutoArg, InfoArgs};
use crate::console::Console;
use crate::exit::{self, Failure};
use crate::report::{
    AppleAutoField, Object, apple_auto_text, camera_json, develop_fields, error_value, match_word,
};
use crate::runtime::{CoreSettings, Runtime, interrupted, read_instructions};
use crate::text::{lossy, path_text};

/// Why `run_to` stopped early.
#[derive(Debug)]
enum Stop {
    /// stdout could not be written.
    Io(io::Error),
    /// The command cannot go on.
    Failure(Failure),
}

impl From<io::Error> for Stop {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<Failure> for Stop {
    fn from(f: Failure) -> Self {
        Self::Failure(f)
    }
}

/// Runs `ym info` and returns the exit code.
///
/// A reader that goes away (`ym info *.ARW --json | head -n 1`) ends the run quietly instead of
/// panicking like `println!` would.
pub fn run(args: &InfoArgs, console: &mut Console) -> u8 {
    let mut stdout = io::stdout().lock();
    match run_to(args, console, &mut stdout) {
        Ok(code) => code,
        Err(Stop::Io(error)) if error.kind() == io::ErrorKind::BrokenPipe => exit::OK,
        Err(Stop::Io(error)) => {
            console.error(&format!("Couldn't write the result ({error})."));
            exit::SOME_FAILED
        }
        Err(Stop::Failure(failure)) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// One file given on the command line.
struct Entry {
    path: PathBuf,
    probe: Result<RawMetadata, RawError>,
    photo: Option<PhotoId>,
}

/// [`run`] with an explicit output stream. Fails only when `out` cannot be written or the data
/// folder is missing.
fn run_to(args: &InfoArgs, console: &mut Console, out: &mut dyn Write) -> Result<u8, Stop> {
    let mut entries: Vec<Entry> = args
        .files
        .iter()
        .map(|path| Entry {
            path: path.clone(),
            probe: probe_file(path),
            photo: None,
        })
        .collect();

    // The estimates need the core; files that cannot even be read do not.
    let apple_auto = AppleAutoArg::mode(args.apple_auto);
    let apple_auto_instructions = args
        .apple_auto_instructions
        .as_deref()
        .map(read_instructions)
        .transpose()?;
    let runtime = if entries.iter().any(|e| e.probe.is_ok()) {
        Some(Runtime::open(&CoreSettings {
            jobs: None,
            forced_profile: args.profile.clone(),
            apple_auto: Some(apple_auto),
            apple_auto_instructions,
        })?)
    } else {
        None
    };
    if let Some(rt) = &runtime {
        if let Some(notice) = rt.apple_auto_notice(apple_auto) {
            console.warn(&notice);
        }
        let mut opened: HashMap<PathBuf, Option<PhotoId>> = HashMap::new();
        for e in entries.iter_mut().filter(|e| e.probe.is_ok()) {
            let key = std::fs::canonicalize(&e.path).unwrap_or_else(|_| e.path.clone());
            e.photo = *opened.entry(key).or_insert_with(|| {
                is_arw_name(&e.path)
                    .then(|| {
                        rt.core
                            .session
                            .open(std::slice::from_ref(&e.path), OpenOptions::default())
                            .added
                            .first()
                            .copied()
                    })
                    .flatten()
            });
        }
        let finished = rt.wait_developed(|w| {
            console.progress(match w.answers {
                Some((answered, n)) => format!("Asking Apple Intelligence {answered}/{n}"),
                None => format!("Reading {}/{}", w.ready, w.total),
            });
        });
        console.finish_progress();
        if !finished {
            console.info("Interrupted.");
            return Ok(exit::INTERRUPTED);
        }
    }

    let mut failures = 0usize;
    let mut printed_text = false;
    for entry in &entries {
        if interrupted() {
            return Ok(exit::INTERRUPTED);
        }
        match &entry.probe {
            Err(error) => {
                failures += 1;
                if args.json {
                    writeln!(out, "{}", json_error(&entry.path, error).to_line())?;
                } else {
                    out.flush()?;
                    console.error(&format!("{}: {error}", path_text(&entry.path)));
                }
            }
            Ok(meta) => {
                let rt = runtime
                    .as_ref()
                    .expect("a runtime exists when a file probed");
                let camera = PhotoCamera::new(
                    &rt.core.session.env().profiles,
                    meta,
                    args.profile.as_deref(),
                );
                let developed = match entry.photo {
                    Some(id) => develop(rt, id),
                    None => Err("Only files with the extension .ARW are developed.".to_owned()),
                };
                let shown: Result<Developed<'_>, &str> = match &developed {
                    Ok((view, resolved)) => Ok((view, resolved)),
                    Err(reason) => Err(reason.as_str()),
                };
                if args.json {
                    writeln!(
                        out,
                        "{}",
                        json_ok(&entry.path, meta, &camera, shown).to_line()
                    )?;
                } else {
                    if printed_text {
                        writeln!(out)?;
                    }
                    write!(out, "{}", render_text(&entry.path, meta, &camera, shown))?;
                    printed_text = true;
                }
            }
        }
    }
    out.flush()?;
    Ok(if failures == 0 {
        exit::OK
    } else {
        exit::SOME_FAILED
    })
}

/// The session's view of a developed photo with its settings resolved, or why there is none.
fn develop(rt: &Runtime, id: PhotoId) -> Result<(PhotoView, ResolvedSettings), String> {
    let view = rt.core.session.view(id).map_err(|e| e.to_string())?;
    let resolved = view.resolved().map_err(|e| e.to_string())?;
    Ok((view, resolved))
}

fn is_arw_name(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("arw"))
}

/// Probes one input. A folder is an error, not a read attempt.
fn probe_file(path: &Path) -> Result<RawMetadata, RawError> {
    if path.is_dir() {
        return Err(RawError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "this is a folder, not a file",
        )));
    }
    probe(path)
}

/// The `status: "error"` object for a file that could not be read.
pub fn json_error(path: &Path, error: &RawError) -> Object {
    let mut obj = Object::new();
    obj.insert("input", absolute(path));
    obj.insert("status", "error");
    obj.insert("error", error_value(error.kind(), &error.to_string()));
    obj.insert("pipeline_version", PIPELINE_VERSION);
    obj
}

/// The path as an absolute one (symbolic links are not resolved), for the `input` of the JSON.
fn absolute(path: &Path) -> String {
    lossy(&std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// The `status: "ok"` object for a file: the shape of `render --json` without `output` and
/// `timings_ms`, plus the metadata blocks.
///
/// `developed` is the photo's view and resolved settings, or the reason it was not developed
/// (then `look`, `auto` and `settings` are null and the reason is in `notes`).
pub fn json_ok(
    path: &Path,
    meta: &RawMetadata,
    camera: &PhotoCamera,
    developed: Result<Developed<'_>, &str>,
) -> Object {
    let mut obj = Object::new();
    obj.insert("input", absolute(path));
    obj.insert("status", json!("ok"));
    match developed {
        Ok((view, resolved)) => {
            obj.extend(develop_fields(view, resolved, AppleAutoField::Always));
            obj.insert("notes", json!([]));
        }
        Err(reason) => {
            obj.insert("camera", camera_json(camera, meta));
            obj.insert("look", Value::Null);
            obj.insert("auto", Value::Null);
            obj.insert("auto_source", Value::Null);
            obj.insert("apple_auto", Value::Null);
            obj.insert("settings", Value::Null);
            obj.insert("notes", json!([reason]));
        }
    }
    obj.insert(
        "exposure",
        json!({
            "iso": meta.iso,
            "iso_source": meta.iso_source.map(|s| s.name()),
            "shutter_s": meta.exposure_time_s,
            "aperture": meta.f_number,
            "exposure_comp_ev": meta.exposure_comp_ev,
            "focal_length_mm": meta.focal_length_mm,
            "focal_length_35mm": meta.focal_length_35mm,
            "lens_model": meta.lens_model,
            "captured": meta.capture_time.map(|t| t.to_iso8601()),
            "orientation": meta.orientation,
        }),
    );
    obj.insert(
        "dimensions",
        json!({
            "width": meta.width,
            "height": meta.height,
            "active_area": meta.active_area.map(rect_json),
            "crop": meta.crop.map(rect_json),
        }),
    );
    obj.insert(
        "raw",
        json!({
            "decode": meta.decode.id(),
            "description": meta.decode.describe(),
            "compression": meta.compression,
            "bits_per_sample": meta.bits_per_sample,
            "samples_per_pixel": meta.samples_per_pixel,
            "cfa": meta.cfa.name(),
            // Lossless M and S are camera-processed YCbCr: no mosaic, no demosaic, a REDUCED tag.
            "reduced": is_reduced(meta.decode),
            "subsampling": match meta.decode {
                DecodeKind::LosslessMS { subsampling } => Some(subsampling.name()),
                _ => None,
            },
            // False for ARW 6.0 and layouts no decoder reads yet.
            "supported": is_supported(meta.decode),
        }),
    );
    obj.insert("black", json!(meta.black));
    obj.insert("white", json!(meta.white));
    obj.insert("wb_as_shot", json!(meta.wb_as_shot));
    obj.insert(
        "levels_source",
        json!({
            "black": meta.black_source.name(),
            "white": meta.white_source.name(),
            "wb": meta.wb_source.map(LevelSource::name),
        }),
    );
    obj.insert(
        "lens_corrections",
        json!(meta.lens_corrections.as_ref().map(lens_json)),
    );
    obj.insert("tone_curve", json!(meta.tone_curve));
    obj.insert(
        "preview",
        json!(
            meta.preview
                .map(|p| json!({ "offset": p.offset, "length": p.length }))
        ),
    );
    obj.insert(
        "gps",
        json!(meta.gps.as_ref().map(|g| json!({
            "latitude_deg": g.latitude_deg,
            "longitude_deg": g.longitude_deg,
            "altitude_m": g.altitude_m,
            "utc_time": g.utc_time,
        }))),
    );
    obj.insert("pipeline_version", json!(PIPELINE_VERSION));
    obj.insert("error", Value::Null);
    obj
}

/// The Profile row: the short name of the profile and how it was chosen.
fn profile_text(camera: &PhotoCamera) -> String {
    let m = &camera.matched;
    let how = match &m.how {
        ym_profiles::detect::MatchKind::Exact => "measured on this body".to_owned(),
        ym_profiles::detect::MatchKind::SensorFamily { via } => {
            format!("stand-in: same sensor as {via}")
        }
        ym_profiles::detect::MatchKind::Default => "stand-in: no profile of its own".to_owned(),
        ym_profiles::detect::MatchKind::Forced { .. } => "forced with --profile".to_owned(),
    };
    format!(
        "{} {}  ({how}; {})",
        camera.copy.profile,
        m.profile_version,
        match_word(&m.how)
    )
}

/// The rows of what the automatic development makes of a photo.
fn develop_rows(view: &PhotoView, resolved: &ResolvedSettings) -> Vec<(&'static str, String)> {
    let e = &resolved.estimates;
    vec![
        (
            "Look",
            format!(
                "{} at {}%",
                view.settings.look,
                trim_number(view.settings.look_intensity, 1)
            ),
        ),
        (
            "Auto exposure",
            format!("{} EV", format_signed_ev(e.exposure_ev)),
        ),
        (
            "Auto light",
            format!(
                "{}, {}% sure",
                e.light_class.name(),
                (e.light_confidence * 100.0).round()
            ),
        ),
        (
            "Auto white",
            format!(
                "{} K, tint {}",
                trim_number(e.kelvin, 0),
                trim_number(e.tint, 1)
            ),
        ),
        // What Apple Auto did to the values above (`docs/19` section 8).
        (
            "Apple Auto",
            apple_auto_text(&e.apple, view.estimates.apple_steps),
        ),
        (
            "Noise model",
            format!("a = {:.3e}, b = {:.3e}", e.noise.a, e.noise.b),
        ),
        (
            "Auto sharpen",
            format!(
                "amount {}, radius {} px",
                trim_number(e.sharpen.amount, 0),
                trim_number(e.sharpen.radius_px, 2)
            ),
        ),
        (
            "Auto noise red.",
            format!(
                "luminance {}, colour {}, detail {}",
                trim_number(e.noise_reduction.luminance, 0),
                trim_number(e.noise_reduction.color, 0),
                trim_number(e.noise_reduction.detail, 0)
            ),
        ),
    ]
}

/// "+0.3", "-1.2", "0.0": a signed number of stops with one decimal.
fn format_signed_ev(ev: f64) -> String {
    if ev >= 0.0 {
        format!("+{ev:.2}")
    } else {
        format!("{ev:.2}")
    }
}

/// True for the camera-processed files (Lossless M and S) that take the REDUCED path.
fn is_reduced(decode: DecodeKind) -> bool {
    matches!(decode, DecodeKind::LosslessMS { .. })
}

/// True when `ym-raw` has a decoder for the layout.
fn is_supported(decode: DecodeKind) -> bool {
    !matches!(decode, DecodeKind::Arw6 | DecodeKind::Unknown { .. })
}

fn rect_json(r: Rect) -> Value {
    json!({ "x": r.x, "y": r.y, "width": r.width, "height": r.height })
}

fn lens_json(lens: &LensTags) -> Value {
    json!({
        "vignetting": lens.vignetting.is_some(),
        "chromatic_aberration": lens.chromatic_aberration.is_some(),
        "distortion": lens.distortion.is_some(),
    })
}

/// What the session worked out for a photo that developed: its view and its settings resolved.
pub type Developed<'a> = (&'a PhotoView, &'a ResolvedSettings);

/// The text block for one file. `camera` is the profile match; `developed` is `Ok` with the
/// photo's estimates when it developed, else the reason it did not.
pub fn render_text(
    path: &Path,
    meta: &RawMetadata,
    camera: &PhotoCamera,
    developed: Result<Developed<'_>, &str>,
) -> String {
    let mut rows: Vec<(&str, String)> = Vec::new();

    let mut body = if meta.make.is_empty() {
        meta.model.clone()
    } else {
        format!("{} {}", meta.make, meta.model)
    };
    if let Some(id) = meta.model_id {
        body.push_str(&format!(" (model ID {id})"));
    }
    rows.push(("Camera", body));
    rows.push(("Profile", profile_text(camera)));
    if let Some(software) = &meta.software {
        let version = meta
            .arw_version_name()
            .map(|v| format!(", {v}"))
            .unwrap_or_default();
        rows.push(("Firmware", format!("{software}{version}")));
    }
    if let Some(lens) = &meta.lens_model {
        rows.push(("Lens", lens.clone()));
    }

    let mut exposure = Vec::new();
    if let Some(t) = meta.exposure_time_s {
        exposure.push(format_shutter(t));
    }
    if let Some(f) = meta.f_number {
        exposure.push(format!("f/{}", trim_number(f, 1)));
    }
    if let Some(iso) = meta.iso {
        exposure.push(format!("ISO {iso}"));
    }
    if let Some(ev) = meta.exposure_comp_ev {
        exposure.push(format!("{} EV", format_signed(ev)));
    }
    if let Some(mm) = meta.focal_length_mm {
        exposure.push(format!("{} mm", trim_number(mm, 1)));
    }
    if !exposure.is_empty() {
        rows.push(("Exposure", exposure.join("  ")));
    }
    if let Some(time) = meta.capture_time {
        let offset = time
            .utc_offset_minutes
            .map(|m| format!(" {}", offset_text(m)))
            .unwrap_or_default();
        rows.push((
            "Taken",
            format!(
                "{:04}-{:02}-{:02} {:02}:{:02}:{:02}{offset}",
                time.year, time.month, time.day, time.hour, time.minute, time.second
            ),
        ));
    }

    // cRAW files say 12 bits although they decode to a 14-bit scale; the tag would mislead. The
    // camera-processed Lossless M and S files are YCbCr, with a subsampling instead of a CFA.
    let depth = if matches!(
        meta.decode,
        DecodeKind::Craw | DecodeKind::LosslessMS { .. }
    ) {
        String::new()
    } else {
        format!(", {} bits", meta.bits_per_sample)
    };
    let arrangement = match meta.decode {
        DecodeKind::LosslessMS { subsampling } => subsampling.name().to_owned(),
        _ => meta.cfa.name(),
    };
    rows.push((
        "Image",
        format!(
            "{} x {}{depth}, {} ({arrangement})",
            meta.width,
            meta.height,
            meta.decode.describe(),
        ),
    ));
    match meta.decode {
        DecodeKind::LosslessMS { subsampling } => rows.push((
            "Reduced",
            format!(
                "yes: camera-processed {} YCbCr, not a mosaic; no demosaic, colour emulation is approximate",
                subsampling.name()
            ),
        )),
        DecodeKind::Arw6 | DecodeKind::Unknown { .. } => {
            rows.push(("Decoder", "not available yet".to_owned()));
        }
        _ => {}
    }
    if let Some(area) = meta.active_area {
        rows.push(("Active area", rect_text(area)));
    }
    if let Some(crop) = meta.crop {
        rows.push(("Camera crop", rect_text(crop)));
    }
    rows.push((
        "Black level",
        format!(
            "{}  ({})",
            numbers_text(&meta.black),
            meta.black_source.name()
        ),
    ));
    rows.push((
        "White level",
        format!(
            "{}  ({})",
            numbers_text(&meta.white),
            meta.white_source.name()
        ),
    ));
    if let Some(wb) = meta.wb_as_shot {
        let source = meta.wb_source.map(LevelSource::name).unwrap_or("unknown");
        rows.push((
            "White balance",
            format!("{}  (as shot, {source})", numbers_text(&wb)),
        ));
    }
    rows.push(("Lens data", lens_text(meta.lens_corrections.as_ref())));
    rows.push((
        "Preview",
        match meta.preview {
            Some(p) => format!("JPEG, {}", size_text(p.length)),
            None => "none".to_owned(),
        },
    ));
    if let Some(gps) = &meta.gps
        && let (Some(lat), Some(lon)) = (gps.latitude_deg, gps.longitude_deg)
    {
        rows.push(("Location", format!("{lat:.5}, {lon:.5}")));
    }

    match developed {
        Ok((view, resolved)) => rows.extend(develop_rows(view, resolved)),
        Err(reason) => {
            // ARW 6.0 and unknown layouts already said so above.
            if !matches!(meta.decode, DecodeKind::Arw6 | DecodeKind::Unknown { .. }) {
                rows.push(("Development", format!("not available: {reason}")));
            }
        }
    }

    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    let mut text = format!("{}\n", path_text(path));
    for (label, value) in rows {
        text.push_str(&format!("  {label:<width$}  {value}\n"));
    }
    text
}

/// The shutter speed the way ExifTool prints it: `1/N s` up to a quarter of a second, one
/// decimal place above (`0.3 s`, `0.8 s`, `1 s`, `30 s`). A reciprocal of a longer exposure would
/// read as a different speed (0.8 s is not 1/1 s).
fn format_shutter(seconds: f64) -> String {
    if seconds > 0.0 && seconds < 0.250_01 {
        format!("1/{} s", (1.0 / seconds + 0.5).floor() as u64)
    } else if seconds > 0.0 {
        format!("{} s", trim_number(seconds, 1))
    } else {
        "0 s".to_owned()
    }
}

/// Rounds to `decimals` places and drops a trailing `.0`.
fn trim_number(value: f64, decimals: usize) -> String {
    let text = format!("{value:.decimals$}");
    text.strip_suffix(".0").map_or(text.clone(), str::to_owned)
}

fn format_signed(value: f64) -> String {
    if value.abs() < 0.05 {
        "0".to_owned()
    } else if value > 0.0 {
        format!("+{}", trim_number(value, 1))
    } else {
        trim_number(value, 1)
    }
}

fn offset_text(minutes: i16) -> String {
    let sign = if minutes < 0 { '-' } else { '+' };
    let abs = minutes.unsigned_abs();
    format!("{sign}{:02}:{:02}", abs / 60, abs % 60)
}

fn numbers_text(values: &[f32; 4]) -> String {
    values
        .iter()
        .map(|v| trim_number(f64::from(*v), 1))
        .collect::<Vec<_>>()
        .join(" ")
}

fn rect_text(r: Rect) -> String {
    format!("{} x {} at {}, {}", r.width, r.height, r.x, r.y)
}

fn lens_text(lens: Option<&LensTags>) -> String {
    let Some(lens) = lens else {
        return "none".to_owned();
    };
    let mut present = Vec::new();
    if lens.vignetting.is_some() {
        present.push("vignetting");
    }
    if lens.chromatic_aberration.is_some() {
        present.push("chromatic aberration");
    }
    if lens.distortion.is_some() {
        present.push("distortion");
    }
    present.join(", ")
}

fn size_text(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1000 {
        format!("{:.0} kB", bytes as f64 / 1e3)
    } else {
        format!("{bytes} bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutter_speeds_read_naturally() {
        assert_eq!(format_shutter(1.0 / 250.0), "1/250 s");
        assert_eq!(format_shutter(1.0 / 8000.0), "1/8000 s");
        assert_eq!(format_shutter(0.25), "1/4 s");
        assert_eq!(format_shutter(1.0), "1 s");
        assert_eq!(format_shutter(30.0), "30 s");
        assert_eq!(format_shutter(1.3), "1.3 s");
        assert_eq!(format_shutter(0.0), "0 s");
    }

    /// Sony's series between a quarter of a second and one second (0.3", 0.4", 0.5", 0.6", 0.8")
    /// used to print as 1/3, 1/3, 1/2, 1/2 and 1/1 s: a different speed from the one shot.
    #[test]
    fn shutter_speeds_between_a_quarter_and_one_second_are_not_reciprocals() {
        for (seconds, text) in [
            (0.3, "0.3 s"),
            (0.4, "0.4 s"),
            (0.5, "0.5 s"),
            (0.6, "0.6 s"),
            (0.8, "0.8 s"),
            (0.77, "0.8 s"),
            (0.999, "1 s"),
        ] {
            assert_eq!(format_shutter(seconds), text, "{seconds}");
        }
        // The 1/N speeds from a quarter of a second down keep their reciprocal form.
        for denominator in [4u32, 5, 6, 8, 10, 13, 15, 20, 25, 30, 60, 125, 4000, 8000] {
            assert_eq!(
                format_shutter(1.0 / f64::from(denominator)),
                format!("1/{denominator} s")
            );
        }
    }

    /// A writer whose reader has gone away.
    struct Closed;

    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn write_arw(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        ym_raw::testgen::ArwBuilder::new("ILCE-7M3", 32, 32)
            .write_to(&path, &[1000u16; 32 * 32])
            .expect("write");
        path
    }

    #[test]
    fn a_closed_stdout_is_reported_to_the_caller_not_a_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_arw(dir.path(), "a.ARW");
        for json in [false, true] {
            let args = InfoArgs {
                files: vec![path.clone()],
                json,
                profile: None,
                apple_auto: None,
                apple_auto_instructions: None,
            };
            let mut console = Console::new(true, false, true);
            let result = run_to(&args, &mut console, &mut Closed);
            match result {
                Err(Stop::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::BrokenPipe, "json {json}"),
                other => panic!("json {json}: {other:?}"),
            }
        }
    }

    #[test]
    fn the_blank_line_separates_printed_blocks_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = write_arw(dir.path(), "good.ARW");
        let bad = dir.path().join("bad.txt");
        std::fs::write(&bad, "x").expect("write");
        let args = InfoArgs {
            files: vec![bad, good.clone()],
            json: false,
            profile: None,
            apple_auto: None,
            apple_auto_instructions: None,
        };
        let mut out = Vec::new();
        let mut console = Console::new(true, false, true);
        let code = run_to(&args, &mut console, &mut out).expect("write to memory");
        assert_eq!(code, exit::SOME_FAILED);
        // The failed file wrote nothing to stdout, so the first block starts with no blank line.
        let text = String::from_utf8(out).expect("utf-8");
        assert!(text.starts_with(&good.display().to_string()), "{text:?}");
    }

    #[test]
    fn numbers_drop_a_pointless_decimal() {
        assert_eq!(trim_number(4.0, 1), "4");
        assert_eq!(trim_number(2.8, 1), "2.8");
        assert_eq!(trim_number(35.04, 1), "35");
        assert_eq!(format_signed(-0.7), "-0.7");
        assert_eq!(format_signed(0.3), "+0.3");
        assert_eq!(format_signed(0.01), "0");
    }

    #[test]
    fn offsets_and_sizes() {
        assert_eq!(offset_text(120), "+02:00");
        assert_eq!(offset_text(-330), "-05:30");
        assert_eq!(size_text(999), "999 bytes");
        assert_eq!(size_text(4_776), "5 kB");
        assert_eq!(size_text(3_200_000), "3.2 MB");
    }
}
