//! `yellowmoon info <file…> [--json]`: what the file says about itself, without rendering
//! (`docs/10-cli.md`).
//!
//! Only metadata is shown for now: camera, exposure, lens, dimensions, the raw layout and
//! levels, which lens-correction tables the file carries and where its preview is. The
//! automatic estimates (white balance, exposure, light class, noise) and the camera profile are
//! added when the colour and profile crates land; `--json` already uses the object shape from
//! `docs/10` ("`info --json` prints the same object without `output` and `timings_ms`") with the
//! fields that exist today.
//!
//! Text output goes to stdout, one block per file. In `--json` mode every file produces one JSON
//! object on its own line of stdout, also when it fails (`"status": "error"`). In text mode a
//! failure is reported on stderr as `<path>: <message>`.
//!
//! Exit code: 0 when every file was read, 1 when at least one failed. Usage errors are code 2 and
//! are reported by clap before this module runs.

use std::io::{self, Write};
use std::path::Path;

use serde_json::{Value, json};
use ym_core::version::PIPELINE_VERSION;
use ym_raw::error::RawError;
use ym_raw::layout::{DecodeKind, Rect};
use ym_raw::metadata::{LensTags, LevelSource, RawMetadata, probe};

use crate::cli::InfoArgs;
use crate::exit;

/// Runs `yellowmoon info` and returns the exit code.
///
/// A reader that goes away (`yellowmoon info *.ARW --json | head -n 1`) ends the run quietly
/// instead of panicking like `println!` would.
pub fn run(args: &InfoArgs) -> u8 {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    match run_to(args, &mut stdout, &mut stderr) {
        Ok(code) => code,
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => exit::OK,
        Err(error) => {
            let _ = writeln!(stderr, "Error: couldn't write the result ({error}).");
            exit::SOME_FAILED
        }
    }
}

/// [`run`] with explicit output streams. Fails only when `out` cannot be written.
fn run_to(args: &InfoArgs, out: &mut dyn Write, err: &mut dyn Write) -> io::Result<u8> {
    let mut failures = 0usize;
    let mut printed_text = false;
    for path in &args.files {
        let result = probe_file(path);
        if result.is_err() {
            failures += 1;
        }
        if args.json {
            writeln!(out, "{}", json_line(path, &result))?;
        } else {
            match &result {
                Ok(meta) => {
                    if printed_text {
                        writeln!(out)?;
                    }
                    write!(out, "{}", render_text(path, meta))?;
                    printed_text = true;
                }
                Err(error) => {
                    // Keep the two streams in order when both go to the same terminal.
                    out.flush()?;
                    let _ = writeln!(err, "{}: {error}", path.display());
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

/// The JSON text for one file: the success object or the error object.
pub fn json_line(path: &Path, result: &Result<RawMetadata, RawError>) -> String {
    let value = match result {
        Ok(meta) => json_ok(path, meta),
        Err(error) => json_error(path, error),
    };
    value.to_string()
}

/// The `status: "error"` object for a file that could not be read.
pub fn json_error(path: &Path, error: &RawError) -> Value {
    json!({
        "input": path.display().to_string(),
        "status": "error",
        "error": { "kind": error.kind(), "message": error.to_string() },
        "pipeline_version": PIPELINE_VERSION,
    })
}

/// The `status: "ok"` object for a file.
pub fn json_ok(path: &Path, meta: &RawMetadata) -> Value {
    json!({
        "input": path.display().to_string(),
        "status": "ok",
        "camera": {
            "make": meta.make,
            "model": meta.model,
            "model_id": meta.model_id,
            // The registry (docs/03) names the body once ym-profiles lands; until then it is the
            // Model string from the file.
            "body": meta.model,
            "firmware": meta.software,
            "arw_version": meta.arw_version_name(),
        },
        "exposure": {
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
        },
        "dimensions": {
            "width": meta.width,
            "height": meta.height,
            "active_area": meta.active_area.map(rect_json),
            "crop": meta.crop.map(rect_json),
        },
        "raw": {
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
        },
        "black": meta.black,
        "white": meta.white,
        "wb_as_shot": meta.wb_as_shot,
        "levels_source": {
            "black": meta.black_source.name(),
            "white": meta.white_source.name(),
            "wb": meta.wb_source.map(LevelSource::name),
        },
        "lens_corrections": meta.lens_corrections.as_ref().map(lens_json),
        "tone_curve": meta.tone_curve,
        "preview": meta.preview.map(|p| json!({ "offset": p.offset, "length": p.length })),
        "gps": meta.gps.as_ref().map(|g| json!({
            "latitude_deg": g.latitude_deg,
            "longitude_deg": g.longitude_deg,
            "altitude_m": g.altitude_m,
            "utc_time": g.utc_time,
        })),
        "pipeline_version": PIPELINE_VERSION,
        "error": Value::Null,
    })
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

/// The text block for one file.
pub fn render_text(path: &Path, meta: &RawMetadata) -> String {
    let mut rows: Vec<(&str, String)> = Vec::new();

    let mut camera = if meta.make.is_empty() {
        meta.model.clone()
    } else {
        format!("{} {}", meta.make, meta.model)
    };
    if let Some(id) = meta.model_id {
        camera.push_str(&format!(" (model ID {id})"));
    }
    rows.push(("Camera", camera));
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

    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    let mut text = format!("{}\n", path.display());
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

    #[test]
    fn a_closed_stdout_is_reported_to_the_caller_not_a_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.ARW");
        ym_raw::testgen::ArwBuilder::new("ILCE-7M3", 32, 32)
            .write_to(&path, &[1000u16; 32 * 32])
            .expect("write");
        for json in [false, true] {
            let args = InfoArgs {
                files: vec![path.clone()],
                json,
            };
            let mut err = Vec::new();
            let result = run_to(&args, &mut Closed, &mut err);
            assert_eq!(
                result.expect_err("closed pipe").kind(),
                io::ErrorKind::BrokenPipe,
                "json {json}"
            );
        }
    }

    #[test]
    fn the_blank_line_separates_printed_blocks_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = dir.path().join("good.ARW");
        ym_raw::testgen::ArwBuilder::new("ILCE-7M3", 32, 32)
            .write_to(&good, &[1000u16; 32 * 32])
            .expect("write");
        let bad = dir.path().join("bad.txt");
        std::fs::write(&bad, "x").expect("write");
        let args = InfoArgs {
            files: vec![bad, good.clone()],
            json: false,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run_to(&args, &mut out, &mut err).expect("write to memory");
        assert_eq!(code, exit::SOME_FAILED);
        // The failed file wrote nothing to stdout, so the first block starts with no blank line.
        let text = String::from_utf8(out).expect("utf-8");
        assert!(text.starts_with(&good.display().to_string()), "{text:?}");
        assert!(String::from_utf8_lossy(&err).contains("bad.txt"));
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
