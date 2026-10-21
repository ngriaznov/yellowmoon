//! The DNG gates (`docs/09-export.md`, section Gates; `docs/13-testing.md`):
//!
//! 1. every exported DNG passes Adobe's `dng_validate` with no warning and no error;
//! 2. the picture `dng_validate` renders from the DNG, against our JPEG of the same photo: CIEDE2000 median at most 1.5,
//!    or 2.5 for a file at ISO 12800 or above (see [`median_limit_for`]).
//!
//! Both run on the golden set. For each file the real `ym` binary writes the DNG (`--format dng`) and the JPEG (quality
//! 92, sRGB) with the same settings; `dng_validate` renders the DNG to sRGB; the two pictures are compared pixel by
//! pixel (`dng_validate` turns its render upright by the DNG's `Orientation` tag and crops it to the default crop; see
//! [`turn_after_render`]).
//!
//! **As written.** `ym` embeds `data/dng/profile.json` and writes it into every DNG (`ym-export` floors `x_white` at
//! `2^-fit_offset_ev`, `ym-core` passes the profile; see `docs/09`), so the gates check the DNGs exactly as `ym` wrote them
//! ([`Reprofile::Never`], what `tools/dng/run_gates.sh` runs).
//!
//! **Re-profiling** is for a `ym` that still writes the identity profile (an old build, a bisect). With a profile given,
//! [`reprofile`] reads the pixels back from that DNG and writes them again with the product's DNG writer, the fitted
//! profile and its `BaselineExposure` rule. The last rows of such a file are white anchors for the writer's white level, and
//! are left out of the comparison.
//!
//! A `ym` that writes the fitted profile (a look table with more than the identity's 2 x 2 x 2 divisions) must not be
//! re-profiled: that would scale its pixels by the wrong exposure, so [`Reprofile::IfIdentity`] (the default of
//! `dngfit gate`) checks such a file as it is, and [`Reprofile::Always`] refuses it with a message.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use golden::dngread::read_linear_raw;
use golden::image::{DeltaStats, Rgb16, read_png16, stats_of};
use golden::manifest::Entry;
use golden::ymrun::{Output, YmBinary};
use rayon::prelude::*;
use serde::Serialize;
use ym_color::consts::linalg;
use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::encode::srgb_decode;
use ym_color::gamut;

use crate::adobe::{Adobe, write_dng_image};
use crate::profile_file::ProfileFile;

/// The gate: the median ΔE00 of Adobe's render against our JPEG may not exceed this (`docs/09`).
pub const MEDIAN_LIMIT: f64 = 1.5;

/// The ISO from which a file is held to [`HIGH_ISO_MEDIAN_LIMIT`].
pub const HIGH_ISO: u32 = 12_800;

/// The gate for a file at [`HIGH_ISO`] or above. The denoised picture keeps a fine colour grain there, and JPEG coding of
/// that grain is most of the distance to Adobe's render: at ISO 12800 the synthetic chart measures 2.125 against the JPEG
/// and 0.774 against the 16-bit PNG.
pub const HIGH_ISO_MEDIAN_LIMIT: f64 = 2.5;

/// The ISO a golden id names (`chart__D6500__ILCE-7RM4__iso12800__…`), if it names one.
#[must_use]
pub fn iso_of(id: &str) -> Option<u32> {
    id.split("__")
        .find_map(|part| part.strip_prefix("iso")?.parse().ok())
}

/// The median limit for the file `id` when the gate's limit is `base`: [`HIGH_ISO_MEDIAN_LIMIT`] at [`HIGH_ISO`] or
/// above (or `base`, if that is larger), else `base`.
#[must_use]
pub fn median_limit_for(base: f64, id: &str) -> f64 {
    if iso_of(id).is_some_and(|iso| iso >= HIGH_ISO) {
        base.max(HIGH_ISO_MEDIAN_LIMIT)
    } else {
        base
    }
}

/// The largest number of pixels compared per file; larger pictures are sampled on a regular stride.
const MAX_COMPARED: usize = 3_000_000;

/// Whether the DNGs `ym` writes are written again with a fitted profile before they are checked.
#[derive(Clone, Debug)]
pub enum Reprofile {
    /// Check every DNG exactly as `ym` wrote it (`--as-written`).
    Never,
    /// Write every DNG again with this profile (`--profile`); a DNG that already carries a fitted profile is an error.
    Always(ProfileFile),
    /// Write a DNG again with this profile when `ym` wrote the identity profile, and check it as it is when `ym`
    /// already wrote a fitted one: the default, which keeps working once `ym-core` writes the fitted profile.
    IfIdentity(ProfileFile),
}

impl Reprofile {
    /// The profile to write a DNG again with, or `None` to check it as `ym` wrote it, given the DNG's
    /// `ProfileLookTableDims`.
    ///
    /// # Errors
    ///
    /// For [`Reprofile::Always`] when the DNG already carries a fitted profile.
    pub fn profile_for(&self, look_table_dims: Option<[u32; 3]>) -> Result<Option<&ProfileFile>> {
        let fitted_already = carries_fitted_profile(look_table_dims);
        match self {
            Self::Never => Ok(None),
            Self::Always(profile) => {
                ensure!(
                    !fitted_already,
                    "ym already writes a fitted profile ({look_table_dims:?} look table), and writing it again would scale \
                     its pixels by the wrong exposure: check it as written (--as-written, or no --profile)"
                );
                Ok(Some(profile))
            }
            Self::IfIdentity(profile) => Ok((!fitted_already).then_some(profile)),
        }
    }
}

/// Whether a DNG carries a profile other than the writer's identity profile (a look table that is not 2 x 2 x 2).
#[must_use]
pub fn carries_fitted_profile(look_table_dims: Option<[u32; 3]>) -> bool {
    look_table_dims.is_some_and(|d| d != ym_export::dng::LookTable::identity().dims)
}

/// What to run and with what.
#[derive(Clone, Debug)]
pub struct GateOptions {
    /// The `ym` binary.
    pub ym: YmBinary,
    /// `testdata/`.
    pub testdata: PathBuf,
    /// Adobe's renderer.
    pub adobe: Adobe,
    /// Whether to write the DNGs again with the fitted profile.
    pub reprofile: Reprofile,
    /// Files processed at the same time.
    pub jobs: usize,
    /// Also compare with the 16-bit PNG (no JPEG coding, no dither).
    pub with_png: bool,
    /// The median ΔE00 limit ([`median_limit_for`] raises it for a file at ISO 12800 or above).
    pub median_limit: f64,
    /// No progress lines.
    pub quiet: bool,
}

/// How one file did.
#[derive(Clone, Debug, Serialize)]
pub struct FileResult {
    /// The manifest id.
    pub id: String,
    /// Both gates passed.
    pub passed: bool,
    /// The DNG was written again with the fitted profile (`false`: it was checked as `ym` wrote it).
    pub reprofiled: bool,
    /// Why not, in words.
    pub reasons: Vec<String>,
    /// Warnings and errors `dng_validate` printed (gate 1).
    pub validation_problems: Vec<String>,
    /// Adobe's render against our JPEG (gate 2).
    pub vs_jpeg: Option<DeltaStats>,
    /// Adobe's render against our 16-bit PNG, when asked for.
    pub vs_png: Option<DeltaStats>,
    /// Picture size.
    pub width: usize,
    /// Picture size.
    pub height: usize,
    /// Wall time.
    pub seconds: f64,
}

/// The result of a gate run.
#[derive(Clone, Debug, Serialize)]
pub struct GateReport {
    /// `reprofiled`, `as written` or `mixed`.
    pub dng: String,
    /// The median limit.
    pub median_limit: f64,
    /// One result per file.
    pub files: Vec<FileResult>,
    /// Golden entries of the selection whose input file is not on disk, so they were not checked (the caller fills this
    /// in; `dngfit gate --require-all` turns it into a failure).
    #[serde(default)]
    pub not_on_disk: Vec<String>,
}

impl GateReport {
    /// Whether every file passed (and there was at least one).
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.files.is_empty() && self.files.iter().all(|f| f.passed)
    }

    /// The median of the per-file medians (JPEG).
    #[must_use]
    pub fn median_of_medians(&self) -> Option<f64> {
        let v: Vec<f64> = self
            .files
            .iter()
            .filter_map(|f| f.vs_jpeg.map(|s| s.median))
            .collect();
        ym_color::deltae::percentile(&v, 50.0)
    }

    /// The largest per-file median (JPEG).
    #[must_use]
    pub fn worst_median(&self) -> Option<f64> {
        self.files
            .iter()
            .filter_map(|f| f.vs_jpeg.map(|s| s.median))
            .reduce(f64::max)
    }
}

/// The result of [`reprofile`].
#[derive(Clone, Debug)]
pub struct Reprofiled {
    /// The new DNG.
    pub bytes: Vec<u8>,
    /// Picture size.
    pub width: usize,
    /// Picture size.
    pub height: usize,
    /// Rows at the bottom that are white anchors and not part of the picture.
    pub anchor_rows: usize,
    /// The `x_white` the writer chose.
    pub x_white: f32,
    /// The `Orientation` of the source DNG.
    pub orientation: u16,
}

/// Writes the pixels of a DNG again with `profile` (see the module docs).
pub fn reprofile(dng: &[u8], profile: &ProfileFile) -> Result<Reprofiled> {
    let raw = read_linear_raw(dng)?;
    let be = raw.baseline_exposure_ev.unwrap_or(0.0);
    // A `ym` that writes the identity profile writes BaselineExposure = log2(x_white) (no profile offset), so
    // x = stored / 65535 * 2^BE.
    let scale = be.exp2();
    let mut pixels: Vec<f32> = raw
        .data
        .iter()
        .map(|&v| (f64::from(v) / 65535.0 * scale) as f32)
        .collect();
    let (w, h) = (raw.width, raw.height);
    // Enough pixels at the white level that the writer's 99.99th percentile is the white level.
    let anchors = w * h / 2000 + 3;
    let anchor_rows = anchors.div_ceil(w).min(h.saturating_sub(1));
    let white = profile.fit_offset_ev.exp2().recip() as f32;
    for p in pixels[(h - anchor_rows) * w * 3..].iter_mut() {
        *p = white;
    }
    let (bytes, report) =
        write_dng_image(w, h, &pixels, &profile.to_profile(), profile.fit_offset_ev)?;
    Ok(Reprofiled {
        bytes,
        width: w,
        height: h,
        anchor_rows,
        x_white: report.x_white,
        orientation: raw.orientation.unwrap_or(1),
    })
}

/// The orientation to turn Adobe's render by to make it upright, given the `Orientation` of the DNG `ym` wrote.
///
/// `dng_validate` honours the `Orientation` tag of the file it renders: the render of a DNG as `ym` wrote it is upright
/// already (a portrait photo renders tall), so it is not turned again. A DNG written again by [`reprofile`] carries no
/// `Orientation`, so its render is in the pixels' orientation and is turned by that of the DNG `ym` wrote.
#[must_use]
pub fn turn_after_render(reprofiled: bool, ym_dng_orientation: u16) -> u16 {
    if reprofiled { ym_dng_orientation } else { 1 }
}

/// Turns a picture stored in the DNG's orientation into the upright picture (EXIF orientations 1 to 8).
#[must_use]
pub fn upright(
    width: usize,
    height: usize,
    rgb: &[f32],
    orientation: u16,
) -> (usize, usize, Vec<f32>) {
    let (w, h) = (width, height);
    let swap = (5..=8).contains(&orientation);
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![0.0_f32; ow * oh * 3];
    for y in 0..oh {
        for x in 0..ow {
            // the stored pixel (column, row) that lands on output (x, y)
            let (c, r) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                8 => (w - 1 - y, x),
                _ => (x, y),
            };
            let src = (r * w + c) * 3;
            let dst = (y * ow + x) * 3;
            out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
    (ow, oh, out)
}

/// Decodes a JPEG into 8-bit RGB.
pub fn decode_jpeg(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    let mut dec = zune_jpeg::JpegDecoder::new(Cursor::new(bytes));
    let pixels = dec
        .decode()
        .map_err(|e| anyhow::anyhow!("decoding the JPEG: {e:?}"))?;
    let (w, h) = dec
        .dimensions()
        .context("the JPEG decoder gave no dimensions")?;
    ensure!(pixels.len() == w * h * 3, "the JPEG is not 3 channels");
    Ok((w, h, pixels))
}

/// CIEDE2000 between two pictures given as sRGB code values (0 to 1), on a regular stride when the picture is large,
/// leaving out the pixels `valid` rules out. The pictures must have the same size.
#[must_use]
pub fn compare(a: &[f32], b: &[f32], valid: Option<&[bool]>) -> DeltaStats {
    let m = gamut::matrices().rec709_to_xyz;
    let white = d65_white_xyz();
    let n = a.len() / 3;
    let stride = n.div_ceil(MAX_COMPARED).max(1);
    let lab = |rgb: &[f32]| {
        let lin = [
            srgb_decode(f64::from(rgb[0])),
            srgb_decode(f64::from(rgb[1])),
            srgb_decode(f64::from(rgb[2])),
        ];
        xyz_to_lab(linalg::mul_vec(&m, lin), white)
    };
    let des: Vec<f64> = (0..n)
        .step_by(stride)
        .filter(|&i| valid.is_none_or(|v| v[i]))
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&i| delta_e_2000(lab(&a[i * 3..i * 3 + 3]), lab(&b[i * 3..i * 3 + 3])))
        .collect();
    stats_of(&des)
}

fn to_unit(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&v| f32::from(v) / 255.0).collect()
}

fn png_to_unit(p: &Rgb16) -> Vec<f32> {
    p.data.iter().map(|&v| f32::from(v) / 65535.0).collect()
}

/// Runs both gates on one file.
pub fn check_file(opts: &GateOptions, entry: &Entry) -> Result<FileResult> {
    let started = Instant::now();
    let input = opts.testdata.join(&entry.file);
    ensure!(input.is_file(), "{} is not on disk", input.display());
    let scratch = tempfile::tempdir().context("creating a scratch folder")?;
    let work = |name: &str| scratch.path().join(name);

    let dng = opts.ym.render(
        &input,
        &entry.settings,
        Output::Dng,
        &work("dng"),
        &work("w1"),
    )?;
    let jpeg = opts.ym.render(
        &input,
        &entry.settings,
        Output::Jpeg,
        &work("jpg"),
        &work("w2"),
    )?;
    let png = if opts.with_png {
        Some(opts.ym.render(
            &input,
            &entry.settings,
            Output::Png16,
            &work("png"),
            &work("w3"),
        )?)
    } else {
        None
    };

    let dng_bytes =
        std::fs::read(&dng.output).with_context(|| format!("reading {}", dng.output.display()))?;
    let written = read_linear_raw(&dng_bytes)?;
    let profile = opts.reprofile.profile_for(written.look_table_dims)?;
    let (dng_path, anchor_rows, source_orientation) = match profile {
        Some(profile) => {
            let r = reprofile(&dng_bytes, profile)?;
            let path = work("reprofiled.dng");
            std::fs::write(&path, &r.bytes)?;
            (path, r.anchor_rows, r.orientation)
        }
        None => (dng.output.clone(), 0, written.orientation.unwrap_or(1)),
    };
    let reprofiled = profile.is_some();
    let rendered = opts.adobe.render_file_verbose(&dng_path)?;
    let mut validation_problems = rendered.problems();
    if reprofiled {
        // The file that was rendered is the writer's re-run on the same pixels, with no EXIF, XMP, crop or orientation:
        // the DNG `ym` wrote (the one users get) is validated too, so gate 1 is about the real export.
        validation_problems.extend(
            opts.adobe
                .validate_file(&dng.output)?
                .into_iter()
                .map(|p| format!("ym's own DNG: {p}")),
        );
    }
    let orientation = turn_after_render(reprofiled, source_orientation);
    let (w, h, adobe_rgb) = upright(
        rendered.width,
        rendered.height,
        &rendered.samples,
        orientation,
    );

    let jpeg_bytes = std::fs::read(&jpeg.output)?;
    let (jw, jh, jpeg_px) = decode_jpeg(&jpeg_bytes)?;
    let mut reasons = Vec::new();
    for p in &validation_problems {
        reasons.push(format!("dng_validate: {p}"));
    }
    ensure!(
        (jw, jh) == (w, h),
        "the JPEG is {jw} x {jh} and the DNG renders {w} x {h}"
    );
    // The anchor rows are the bottom rows of the DNG as stored; the same turn that makes the render upright moves them.
    let valid = if anchor_rows == 0 {
        None
    } else {
        let (sw, sh) = (rendered.width, rendered.height);
        let mut stored = vec![1.0_f32; sw * sh * 3];
        for p in stored[(sh - anchor_rows) * sw * 3..].iter_mut() {
            *p = 0.0;
        }
        let (_, _, turned) = upright(sw, sh, &stored, orientation);
        Some(
            turned
                .chunks_exact(3)
                .map(|c| c[0] > 0.5)
                .collect::<Vec<bool>>(),
        )
    };
    let vs_jpeg = compare(&adobe_rgb, &to_unit(&jpeg_px), valid.as_deref());
    let limit = median_limit_for(opts.median_limit, &entry.id);
    if vs_jpeg.median > limit {
        reasons.push(format!(
            "the median ΔE00 against the JPEG is {:.3}, over the limit of {limit}",
            vs_jpeg.median
        ));
    }
    let vs_png = match &png {
        Some(png) => {
            let p = read_png16(&png.output)?;
            ensure!((p.width, p.height) == (w, h), "the PNG has another size");
            Some(compare(&adobe_rgb, &png_to_unit(&p), valid.as_deref()))
        }
        None => None,
    };
    Ok(FileResult {
        id: entry.id.clone(),
        passed: reasons.is_empty(),
        reprofiled,
        reasons,
        validation_problems,
        vs_jpeg: Some(vs_jpeg),
        vs_png,
        width: w,
        height: h,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Runs the gates on `entries`, up to `opts.jobs` at a time.
pub fn run(opts: &GateOptions, entries: &[&Entry]) -> GateReport {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<FileResult>>> =
        Mutex::new((0..entries.len()).map(|_| None).collect());
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..opts.jobs.clamp(1, entries.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(entry) = entries.get(i) else { break };
                    let r = check_file(opts, entry).unwrap_or_else(|e| FileResult {
                        id: entry.id.clone(),
                        passed: false,
                        reprofiled: false,
                        reasons: vec![format!("{e:#}")],
                        validation_problems: Vec::new(),
                        vs_jpeg: None,
                        vs_png: None,
                        width: 0,
                        height: 0,
                        seconds: 0.0,
                    });
                    let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                    if !opts.quiet {
                        eprintln!(
                            "[{n}/{}] {}: {}",
                            entries.len(),
                            r.id,
                            r.vs_jpeg.map_or_else(
                                || "failed".to_owned(),
                                |s| format!("median ΔE00 {:.3}", s.median)
                            )
                        );
                    }
                    if let Ok(mut g) = results.lock() {
                        g[i] = Some(r);
                    }
                }
            });
        }
    });
    let files: Vec<FileResult> = results
        .into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect();
    // a file that failed before it was rendered says nothing about how the DNGs were written
    let checked = || files.iter().filter(|f| f.vs_jpeg.is_some());
    let dng = match (
        checked().any(|f| f.reprofiled),
        checked().all(|f| f.reprofiled),
    ) {
        (true, true) => "reprofiled",
        (true, false) => "mixed",
        (false, _) => "as written",
    };
    GateReport {
        dng: dng.to_owned(),
        median_limit: opts.median_limit,
        files,
        not_on_disk: Vec::new(),
    }
}

/// Runs `dng_validate` on DNG files (gate 1 alone) and returns the problems per file.
pub fn validate_files(adobe: &Adobe, files: &[PathBuf]) -> Vec<(PathBuf, Result<Vec<String>>)> {
    files
        .iter()
        .map(|f| {
            (
                f.clone(),
                adobe.render_file_verbose(f).map(|r| r.problems()),
            )
        })
        .collect()
}

/// Checks that `path` is a DNG file by extension, for the command line.
#[must_use]
pub fn is_dng(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("dng"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .flat_map(|i| [(i % w) as f32, (i / w) as f32, 0.5])
            .collect()
    }

    #[test]
    fn files_at_iso_12800_and_above_get_the_high_iso_limit() {
        assert_eq!(
            iso_of("synthetic/chart__D6500__ILCE-7RM4__iso12800__rggb__14b__lossless-l"),
            Some(12_800)
        );
        assert_eq!(
            iso_of("synthetic/chart__bb2850__ILCE-1__iso400__rggb__14b__craw"),
            Some(400)
        );
        assert_eq!(iso_of("/photos/DSC01234.ARW"), None);
        let at = |id| median_limit_for(MEDIAN_LIMIT, id);
        assert_eq!(
            at("synthetic/chart__D6500__ILCE-7RM4__iso12800__rggb__14b__lossless-l"),
            2.5
        );
        assert_eq!(
            at("synthetic/chart__D6500__ILCE-7RM4__iso25600__rggb__14b__lossless-l"),
            2.5
        );
        assert_eq!(
            at("synthetic/chart__D6500__ILCE-7RM4__iso3200__rggb__14b__lossless-l"),
            1.5
        );
        assert_eq!(at("/photos/DSC01234.ARW"), 1.5);
        assert_eq!(median_limit_for(3.0, "x__iso12800__y"), 3.0);
    }

    #[test]
    fn orientations_turn_the_stored_picture_upright() {
        // A 3 x 2 picture whose pixels are (column, row).
        let (w, h) = (3, 2);
        let src = picture(w, h);
        let at = |img: &[f32], width: usize, x: usize, y: usize| {
            (img[(y * width + x) * 3], img[(y * width + x) * 3 + 1])
        };
        // 1: unchanged
        let (ow, oh, o) = upright(w, h, &src, 1);
        assert_eq!((ow, oh), (3, 2));
        assert_eq!(o, src);
        // 2: mirrored left-right
        let (_, _, o) = upright(w, h, &src, 2);
        assert_eq!(at(&o, 3, 0, 0), (2.0, 0.0));
        // 3: rotated half a turn
        let (_, _, o) = upright(w, h, &src, 3);
        assert_eq!(at(&o, 3, 0, 0), (2.0, 1.0));
        // 6: stored top row becomes the right column (rotate 90 degrees clockwise)
        let (ow, oh, o) = upright(w, h, &src, 6);
        assert_eq!((ow, oh), (2, 3));
        assert_eq!(
            at(&o, 2, 1, 0),
            (0.0, 0.0),
            "stored top-left is at the top right"
        );
        assert_eq!(
            at(&o, 2, 0, 0),
            (0.0, 1.0),
            "stored bottom-left is at the top left"
        );
        // 8: stored top row becomes the left column (rotate 90 degrees counter-clockwise)
        let (_, _, o) = upright(w, h, &src, 8);
        assert_eq!(
            at(&o, 2, 0, 2),
            (0.0, 0.0),
            "stored top-left is at the bottom left"
        );
        // 5 and 7 are the diagonal reflections
        let (_, _, o) = upright(w, h, &src, 5);
        assert_eq!(at(&o, 2, 1, 0), (0.0, 1.0));
        let (_, _, o) = upright(w, h, &src, 7);
        assert_eq!(at(&o, 2, 0, 0), (2.0, 1.0));
        // every orientation is a permutation of the pixels
        for orientation in 1..=8 {
            let (_, _, o) = upright(w, h, &src, orientation);
            let mut a: Vec<_> = src.chunks(3).map(|c| (c[0] as i32, c[1] as i32)).collect();
            let mut b: Vec<_> = o.chunks(3).map(|c| (c[0] as i32, c[1] as i32)).collect();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "orientation {orientation}");
        }
    }

    #[test]
    fn only_a_reprofiled_render_is_turned_upright_by_the_gate() {
        for orientation in 1..=8 {
            assert_eq!(turn_after_render(false, orientation), 1, "as written");
            assert_eq!(turn_after_render(true, orientation), orientation);
        }
    }

    /// The reason for [`turn_after_render`]: Adobe's renderer turns the picture by the file's `Orientation` tag.
    #[test]
    fn dng_validate_renders_a_file_with_an_orientation_upright() {
        use crate::adobe::{Adobe, write_dng_image_exposure};
        use ym_export::dng::{BaselineExposure, DngProfile};
        let adobe = match Adobe::new(None) {
            Ok(a) => a,
            Err(e) => {
                assert!(
                    std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
                    "dng_validate is required: {e:#}"
                );
                eprintln!("SKIPPED: {e:#}");
                return;
            }
        };
        let (w, h) = (24_usize, 8_usize);
        let px = vec![0.2_f32; w * h * 3];
        let dir = tempfile::tempdir().unwrap();
        for (orientation, want) in [(1_u16, (w, h)), (6, (h, w)), (3, (w, h)), (8, (h, w))] {
            let (bytes, _) = write_dng_image_exposure(
                w,
                h,
                &px,
                &DngProfile::identity(),
                BaselineExposure::default(),
                orientation,
                &ym_export::Metadata::default(),
            )
            .unwrap();
            let path = dir.path().join(format!("o{orientation}.dng"));
            std::fs::write(&path, bytes).unwrap();
            let (rw, rh, _) = adobe.render_file(&path).unwrap();
            assert_eq!((rw, rh), want, "Orientation {orientation}");
        }
    }

    #[test]
    fn comparing_a_picture_with_itself_gives_zero_and_a_shift_does_not() {
        let a: Vec<f32> = (0..64 * 3).map(|i| (i % 255) as f32 / 255.0).collect();
        assert_eq!(compare(&a, &a, None).max, 0.0);
        let b: Vec<f32> = a.iter().map(|v| (v + 0.05).min(1.0)).collect();
        let s = compare(&a, &b, None);
        assert!(s.median > 1.0 && s.max < 30.0, "{s:?}");
        // pixels that are ruled out are not compared
        let mut c = a.clone();
        for v in &mut c[7 * 8 * 3..] {
            *v = 1.0 - *v;
        }
        let valid: Vec<bool> = (0..64).map(|i| i < 56).collect();
        assert_eq!(compare(&a, &c, Some(&valid)).max, 0.0);
        assert!(compare(&a, &c, None).max > 5.0);
    }

    #[test]
    fn a_dng_with_a_fitted_profile_is_not_written_again() {
        use crate::profile_file::ProfileFile;
        use ym_export::dng::DngProfile;
        let file =
            ProfileFile::from_profile(&DngProfile::identity(), -4.0, "t", serde_json::Value::Null);
        // what `ym` writes today and what it will write once ym-core embeds data/dng/profile.json
        let identity = Some([2, 2, 2]);
        let fitted = Some([36, 16, 16]);
        assert!(!carries_fitted_profile(identity));
        assert!(!carries_fitted_profile(None));
        assert!(carries_fitted_profile(fitted));

        let default = Reprofile::IfIdentity(file.clone());
        assert!(default.profile_for(identity).unwrap().is_some());
        assert!(default.profile_for(fitted).unwrap().is_none());
        let always = Reprofile::Always(file);
        assert!(always.profile_for(identity).unwrap().is_some());
        let e = always.profile_for(fitted).unwrap_err().to_string();
        assert!(e.contains("already writes a fitted profile"), "{e}");
        assert!(Reprofile::Never.profile_for(identity).unwrap().is_none());
        assert!(Reprofile::Never.profile_for(fitted).unwrap().is_none());
    }

    #[test]
    fn problems_are_the_warnings_the_errors_and_a_missing_completion() {
        let clean = crate::adobe::FileRender {
            width: 1,
            height: 1,
            samples: vec![0.0; 3],
            log: "Validating...\nValidation complete\n".to_owned(),
        };
        assert!(clean.problems().is_empty());
        let bad = crate::adobe::FileRender {
            log: "*** Warning: something\nValidation complete\n".to_owned(),
            ..clean.clone()
        };
        assert_eq!(bad.problems().len(), 1);
        let cut = crate::adobe::FileRender {
            log: "Validating...\n".to_owned(),
            ..clean
        };
        assert_eq!(cut.problems().len(), 1);
    }
}
