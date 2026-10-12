//! The comparison of one file against the oracles, and the per-variant tally.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use ym_raw::decode::craw::build_curve;
use ym_raw::decode::{ChromaUpsampling, DecodeOptions, RawImage, YcbcrOptions, decode_with};
use ym_raw::layout::{CfaColor, CfaPattern, DecodeKind, Layout};
use ym_raw::metadata::COMPRESSION_NONE;

use crate::libraw::{Dump, Runner};
use crate::rawler_side;
use ym_raw::metadata::RawStorage;

/// How many mismatching positions a report lists.
const FIRST_MISMATCHES: usize = 5;

/// Outcome of one comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every sample is identical.
    Identical,
    /// Not identical but inside the documented tolerance of that oracle (rawler's cRAW dither).
    WithinTolerance,
    /// At least one sample or level differs.
    Mismatch,
    /// The oracle could not decode the file; not counted as a mismatch.
    OracleError,
    /// Not compared on purpose (reported for information only).
    Informational,
    /// The oracle was switched off.
    Skipped,
}

impl Outcome {
    fn name(self) -> &'static str {
        match self {
            Self::Identical => "identical",
            Self::WithinTolerance => "within_tolerance",
            Self::Mismatch => "mismatch",
            Self::OracleError => "oracle_error",
            Self::Informational => "informational",
            Self::Skipped => "skipped",
        }
    }
}

/// Counts per variant and oracle.
#[derive(Default)]
pub struct Tally {
    /// variant -> (files, ours failed, libraw outcomes, rawler outcomes)
    rows: BTreeMap<String, Row>,
}

#[derive(Default)]
struct Row {
    files: usize,
    ours_error: usize,
    /// Files that are not a supported variant but should have decoded (corrupt, unreadable).
    unreadable: usize,
    libraw: BTreeMap<&'static str, usize>,
    libjpeg: BTreeMap<&'static str, usize>,
    rawler: BTreeMap<&'static str, usize>,
    levels_mismatch: usize,
}

impl Tally {
    fn row(&mut self, variant: &str) -> &mut Row {
        self.rows.entry(variant.to_owned()).or_default()
    }

    /// Files that ym-raw decoded (as opposed to refused as unsupported or unreadable).
    pub fn ours_decoded(&self) -> usize {
        self.rows.values().map(|r| r.files - r.ours_error).sum()
    }

    /// Files for which LibRaw produced a decode that was compared (identical, inside tolerance or
    /// mismatching); `oracle_error` and skipped files are not counted.
    pub fn libraw_compared(&self) -> usize {
        self.rows
            .values()
            .map(|r| {
                ["identical", "within_tolerance", "mismatch"]
                    .iter()
                    .map(|k| r.libraw.get(k).copied().unwrap_or(0))
                    .sum::<usize>()
            })
            .sum()
    }

    /// Number of files with a pixel or level mismatch against either oracle.
    pub fn mismatched(&self) -> usize {
        self.rows
            .values()
            .map(|r| {
                r.libraw.get("mismatch").copied().unwrap_or(0)
                    + r.libjpeg.get("mismatch").copied().unwrap_or(0)
                    + r.rawler.get("mismatch").copied().unwrap_or(0)
                    + r.levels_mismatch
                    + r.unreadable
            })
            .sum()
    }

    /// The `summary` object of the JSON report.
    pub fn summary(&self, total: usize) -> Value {
        let by_variant: BTreeMap<&String, Value> = self
            .rows
            .iter()
            .map(|(variant, row)| {
                (
                    variant,
                    json!({
                        "files": row.files,
                        "ours_error": row.ours_error,
                        "unreadable": row.unreadable,
                        "libraw": row.libraw,
                        "libjpeg": row.libjpeg,
                        "rawler": row.rawler,
                        "levels_mismatch": row.levels_mismatch,
                    }),
                )
            })
            .collect();
        let compared: usize = self
            .rows
            .values()
            .map(|r| {
                r.libraw
                    .iter()
                    .chain(r.libjpeg.iter())
                    .chain(r.rawler.iter())
                    .filter(|(k, _)| matches!(**k, "identical" | "within_tolerance" | "mismatch"))
                    .count()
            })
            .sum();
        json!({
            "total": total,
            "compared": compared,
            "mismatched": self.mismatched(),
            "by_variant": by_variant,
        })
    }

    /// A Markdown table for the log.
    pub fn table(&self) -> String {
        let mut out = String::from(
            "| variant | files | LibRaw identical | LibRaw mismatch | LibRaw error | libjpeg identical | libjpeg mismatch | rawler identical | rawler within dither | rawler mismatch | rawler error | rawler info | level mismatch |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
        );
        for (variant, row) in &self.rows {
            let l = |k: &str| row.libraw.get(k).copied().unwrap_or(0);
            let j = |k: &str| row.libjpeg.get(k).copied().unwrap_or(0);
            let r = |k: &str| row.rawler.get(k).copied().unwrap_or(0);
            out.push_str(&format!(
                "| {variant} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                row.files,
                l("identical"),
                l("mismatch"),
                l("oracle_error"),
                j("identical"),
                j("mismatch"),
                r("identical"),
                r("within_tolerance"),
                r("mismatch"),
                r("oracle_error"),
                r("informational"),
                row.levels_mismatch,
            ));
        }
        out
    }
}

/// The name of the variant used in the report.
pub fn variant_name(image: &RawImage) -> &'static str {
    match image.meta.decode {
        DecodeKind::Uncompressed14 => "uncompressed14",
        DecodeKind::Uncompressed12 if image.meta.compression == COMPRESSION_NONE => {
            "uncompressed12"
        }
        DecodeKind::Uncompressed12 => "packed12",
        DecodeKind::Craw => "craw",
        DecodeKind::Arw1 => "arw1",
        DecodeKind::LosslessL => "lossless_l",
        DecodeKind::LosslessMS { .. } => "lossless_ms",
        DecodeKind::Arw6 => "arw6",
        DecodeKind::Unknown { .. } => "unknown",
    }
}

/// Values in the order red, green 1, green 2, blue from a four-value array in CFA raster order.
fn rggb(pattern: CfaPattern, positions: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    let mut greens = 0;
    for (color, value) in pattern.0.iter().zip(positions) {
        match color {
            CfaColor::Red => out[0] = value,
            CfaColor::Green => {
                greens += 1;
                out[greens] = value;
            }
            CfaColor::Blue => out[3] = value,
        }
    }
    out
}

/// The levels of an image in red, green 1, green 2, blue order.
fn levels_rggb(image: &RawImage) -> ([f32; 4], [f32; 4], Option<[f32; 4]>) {
    if image.is_reduced() {
        // Already in this order for camera-processed data.
        return (
            image.black,
            image.white,
            image.wb_as_shot.map(|w| rggb(image.meta.cfa, w)),
        );
    }
    let pattern = image.meta.cfa;
    (
        rggb(pattern, image.black),
        rggb(pattern, image.white),
        image.wb_as_shot.map(|w| rggb(pattern, w)),
    )
}

/// The JSON sidecar written by `oracle dump`.
pub fn sidecar(file: &Path, image: &RawImage, bytes: &[u8]) -> Value {
    let meta = &image.meta;
    let (black, white, wb) = levels_rggb(image);
    let qstep = image.qstep.as_ref().map(|q| {
        let (mut lo, mut hi) = (f32::MAX, 0f32);
        for v in &q.data {
            let v = v.to_f32();
            lo = lo.min(v);
            hi = hi.max(v);
        }
        json!({ "min": lo, "max": hi })
    });
    json!({
        "file": file.display().to_string(),
        "make": meta.make,
        "model": meta.model,
        "decode": meta.decode.id(),
        "layout": match image.layout {
            Layout::Bayer { pattern, .. } => json!({ "kind": "bayer", "cfa": pattern.name() }),
            Layout::Ycbcr { subsampling } => json!({ "kind": "ycbcr", "subsampling": subsampling.name() }),
        },
        "width": image.pixel_width(),
        "height": image.pixel_height(),
        "channels": image.channels(),
        "samples_per_row": image.data.width,
        "black_rggb": black,
        "white_rggb": white,
        "wb_as_shot_rggb": wb,
        "tone_curve": image.tone_curve,
        "qstep": qstep,
        "sha256": hex::encode(Sha256::digest(bytes)),
        "data_format": "little-endian u16",
    })
}

/// The oracles' rows that correspond to ours. ARW 1 files store 8 rows more than `ImageLength`;
/// LibRaw and rawler return them, we drop them.
fn matching_rows<'a>(
    ours: &RawImage,
    theirs_height: usize,
    theirs: &'a [u16],
    row_samples: usize,
) -> Option<&'a [u16]> {
    let height = ours.data.height;
    if theirs_height == height {
        Some(theirs)
    } else if ours.meta.decode == DecodeKind::Arw1
        && theirs_height == height + ym_raw::decode::arw1::EXTRA_ROWS
    {
        theirs.get(..height * row_samples)
    } else {
        None
    }
}

/// Result of comparing two sample arrays of the same length.
struct Diff {
    samples: usize,
    mismatched: usize,
    max_abs: u32,
    first: Vec<Value>,
}

fn diff(ours: &[u16], theirs: &[u16], row_samples: usize) -> Diff {
    let mut d = Diff {
        samples: ours.len(),
        // A dump that is shorter or longer than ours has samples nobody compared: they count as
        // mismatches instead of vanishing in the zip below.
        mismatched: ours.len().abs_diff(theirs.len()),
        max_abs: 0,
        first: Vec::new(),
    };
    for (i, (a, b)) in ours.iter().zip(theirs).enumerate() {
        if a != b {
            d.mismatched += 1;
            d.max_abs = d.max_abs.max(u32::from(a.abs_diff(*b)));
            if d.first.len() < FIRST_MISMATCHES {
                d.first.push(json!({
                    "x": i % row_samples.max(1),
                    "y": i / row_samples.max(1),
                    "ours": a,
                    "theirs": b,
                }));
            }
        }
    }
    d
}

fn diff_json(d: &Diff) -> Value {
    json!({
        "samples": d.samples,
        "mismatched": d.mismatched,
        "max_abs_diff": d.max_abs,
        "first": d.first,
    })
}

/// 12-bit samples in 16-bit words (TIFF compression 1): no real body is known to write them, and
/// LibRaw and rawler read them as 14-bit data, so their levels are not comparable.
fn is_twelve_bit_words(image: &RawImage) -> bool {
    image.meta.decode == DecodeKind::Uncompressed12 && image.meta.compression == COMPRESSION_NONE
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 * a.abs().max(b.abs()).max(1.0)
}

/// Compares ours with LibRaw's dump.
fn against_libraw(
    ours: &RawImage,
    pseudo_raw: Option<&RawImage>,
    dump: &Dump,
) -> (Outcome, bool, Value) {
    let meta = &dump.meta;
    let theirs = &dump.pixels;
    let reduced = ours.is_reduced();
    let image = if reduced {
        pseudo_raw.unwrap_or(ours)
    } else {
        ours
    };
    let (width, height) = (
        meta["width"].as_u64().unwrap_or(0) as usize,
        meta["height"].as_u64().unwrap_or(0) as usize,
    );
    let channels = meta["channels"].as_u64().unwrap_or(1) as usize;
    let mut report = json!({
        "libraw_width": width, "libraw_height": height, "libraw_channels": channels,
        "kind": meta["kind"],
        "input_patched": meta["input_patched"],
    });

    let mut outcome;
    let rows = (width * channels == image.data.width)
        .then(|| matching_rows(image, height, theirs, image.data.width))
        .flatten();
    if let Some(rows) = rows {
        let d = diff(&image.data.data, rows, image.data.width);
        outcome = if d.mismatched == 0 {
            Outcome::Identical
        } else {
            Outcome::Mismatch
        };
        report["pixels"] = diff_json(&d);
        if rows.len() < theirs.len() {
            // ARW 1 files hold 8 rows beyond ImageLength; LibRaw returns them, we do not.
            report["extra_rows_ignored"] = json!((theirs.len() - rows.len()) / image.data.width);
        }
    } else {
        outcome = Outcome::Mismatch;
        report["error"] = json!(format!(
            "size differs: ours {} x {} samples, LibRaw {} x {} x {}",
            image.data.width, image.data.height, width, height, channels
        ));
    }

    // Levels: LibRaw lists R, G, B, G2. For camera-processed data the levels compared are those
    // of the pseudo-raw decode (balanced, pedestal 1024), which is what LibRaw reports.
    let to_rggb = |key: &str| -> Option<[f32; 4]> {
        let list = meta[key].as_array()?;
        let v: Vec<f32> = list
            .iter()
            .filter_map(|x| x.as_f64())
            .map(|x| x as f32)
            .collect();
        (v.len() >= 4).then(|| [v[0], v[1], v[3], v[2]])
    };
    let (black, white, wb) = levels_rggb(image);
    let mut levels_ok = true;
    let mut levels = json!({});
    // LibRaw does not rescale the 14-bit-scale levels of an SR2 block for 12-bit words (it reads
    // those files as 14-bit); ym-raw does, so their black and white levels are not compared.
    let compare_levels = !is_twelve_bit_words(ours);
    if !compare_levels {
        levels["note"] =
            json!("12-bit words: LibRaw keeps the 14-bit-scale SR2 levels, ym-raw shifts them");
    }
    if let Some(b) = to_rggb("black_level_per_channel").filter(|_| compare_levels) {
        let ok = black.iter().zip(b).all(|(a, b)| approx_eq(*a, b));
        levels["black"] = json!({ "ours": black, "libraw": b, "equal": ok });
        levels_ok &= ok;
    }
    if !compare_levels {
        // nothing more to compare for the white level
    } else if reduced {
        // LibRaw's `maximum` is the ceiling of the balanced data (17536).
        if let Some(maximum) = meta["white_level"].as_f64() {
            let ok = white.iter().all(|w| approx_eq(*w, maximum as f32));
            levels["white"] = json!({ "ours": white, "libraw_maximum": maximum, "equal": ok });
            levels_ok &= ok;
        }
    } else if let Some(w) = to_rggb("camera_white_level_per_channel") {
        let ok = white.iter().zip(w).all(|(a, b)| approx_eq(*a, b));
        levels["white"] = json!({ "ours": white, "libraw": w, "equal": ok });
        levels_ok &= ok;
    }
    if let (Some(ours_wb), Some(w)) = (wb, to_rggb("camera_whitebalance")) {
        // LibRaw's second green may repeat the first; compare red, green and blue.
        let ok = approx_eq(ours_wb[0], w[0])
            && approx_eq(ours_wb[1], w[1])
            && approx_eq(ours_wb[3], w[3]);
        levels["white_balance"] = json!({ "ours": ours_wb, "libraw": w, "equal": ok });
        levels_ok &= ok;
    }
    report["levels"] = levels;
    if !levels_ok && outcome == Outcome::Identical {
        outcome = Outcome::Mismatch;
    }
    (outcome, !levels_ok, report)
}

/// Compares ours with libjpeg's decode of the Lossless L tiles (pixels only).
fn against_libjpeg(ours: &RawImage, dump: &Dump) -> (Outcome, Value) {
    let meta = &dump.meta;
    let (width, height) = (
        meta["width"].as_u64().unwrap_or(0) as usize,
        meta["height"].as_u64().unwrap_or(0) as usize,
    );
    let mut report = json!({ "libjpeg_width": width, "libjpeg_height": height });
    if width != ours.data.width
        || height != ours.data.height
        || dump.pixels.len() != ours.data.data.len()
    {
        report["error"] = json!(format!(
            "size differs: ours {} x {}, libjpeg {} x {}",
            ours.data.width, ours.data.height, width, height
        ));
        return (Outcome::Mismatch, report);
    }
    let d = diff(&ours.data.data, &dump.pixels, ours.data.width);
    report["pixels"] = diff_json(&d);
    (
        if d.mismatched == 0 {
            Outcome::Identical
        } else {
            Outcome::Mismatch
        },
        report,
    )
}

/// Why rawler may legitimately differ on a tiled lossless file: it assumes 512 x 512 tiles and
/// was written for the streams Sony cameras write (predictor 1, no restart markers).
fn rawler_limitation(path: &Path, image: &RawImage) -> Option<String> {
    let RawStorage::Tiles {
        tile_width,
        tile_length,
        offsets,
        byte_counts,
    } = &image.meta.storage
    else {
        return None;
    };
    if (*tile_width, *tile_length) != (512, 512) {
        return Some(format!(
            "rawler assumes 512 x 512 tiles, this file has {tile_width} x {tile_length}"
        ));
    }
    if !image.meta.width.is_multiple_of(*tile_width)
        || !image.meta.height.is_multiple_of(*tile_length)
    {
        return Some(format!(
            "rawler needs whole tiles; {} x {} is not a multiple of {tile_width} x {tile_length}",
            image.meta.width, image.meta.height
        ));
    }
    let (offset, count) = (*offsets.first()?, *byte_counts.first()?);
    let mut file = std::fs::File::open(path).ok()?;
    let mut bytes = vec![0u8; count.min(4096) as usize];
    {
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::Start(offset)).ok()?;
        file.read_exact(&mut bytes).ok()?;
    }
    let header = ym_raw::decode::ljpeg::parse_header(&bytes).ok()?;
    (header.predictor != 1 || header.restart_interval != 0).then(|| {
        format!(
            "the stream uses predictor {} and restart interval {}; Sony cameras write predictor 1 without restarts",
            header.predictor, header.restart_interval
        )
    })
}

/// Compares ours with rawler.
fn against_rawler(
    path: &Path,
    ours: &RawImage,
    pseudo_raw: Option<&RawImage>,
) -> (Outcome, bool, Value) {
    let theirs = match rawler_side::decode(path) {
        Ok(t) => t,
        Err(e) => {
            let mut report = json!({ "error": e });
            if let Some(why) = rawler_limitation(path, ours) {
                report["limitation"] = json!(why);
                return (Outcome::Informational, false, report);
            }
            return (Outcome::OracleError, false, report);
        }
    };
    let reduced = ours.is_reduced();
    let image = if reduced {
        pseudo_raw.unwrap_or(ours)
    } else {
        ours
    };
    let mut report = json!({
        "rawler_width": theirs.width, "rawler_height": theirs.height, "rawler_cpp": theirs.cpp,
    });
    let rows = (theirs.width * theirs.cpp == image.data.width)
        .then(|| matching_rows(image, theirs.height, &theirs.data, image.data.width))
        .flatten();
    let Some(rows) = rows else {
        report["error"] = json!(format!(
            "size differs: ours {} x {} samples, rawler {} x {} x {}",
            image.data.width, image.data.height, theirs.width, theirs.height, theirs.cpp
        ));
        return (Outcome::Mismatch, false, report);
    };

    let (black, white, wb) = levels_rggb(ours);
    let mut levels_ok = true;
    let mut levels = json!({});
    if is_twelve_bit_words(ours) {
        levels["note"] =
            json!("12-bit words: rawler keeps the 14-bit-scale SR2 levels, ym-raw shifts them");
    } else if !reduced {
        if theirs.black.len() >= 4 {
            let b = [
                theirs.black[0],
                theirs.black[1],
                theirs.black[2],
                theirs.black[3],
            ];
            let ok = black.iter().zip(b).all(|(a, b)| approx_eq(*a, b));
            levels["black"] = json!({ "ours": black, "rawler": b, "equal": ok });
            levels_ok &= ok;
        }
        if ours.meta.decode == DecodeKind::Arw1 {
            // rawler keeps its 14-bit-scale white level (15360) for the 12-bit ARW 1 samples;
            // LibRaw and the SR2 block say 3840.
            levels["white"] = json!({ "ours": white[0], "rawler": theirs.white.first(), "note": "not compared: rawler does not rescale the white level of ARW 1 data" });
        } else if let Some(w) = theirs.white.first() {
            let ok = approx_eq(white[0], *w as f32);
            levels["white"] = json!({ "ours": white[0], "rawler": w, "equal": ok });
            levels_ok &= ok;
        }
        if let Some(ours_wb) = wb {
            // rawler normalises to green = 1.
            let g = (ours_wb[1] + ours_wb[2]) / 2.0;
            let mine = [
                ours_wb[0] / ours_wb[1],
                g / ours_wb[1],
                ours_wb[3] / ours_wb[1],
            ];
            let ok = approx_eq(mine[0], theirs.wb[0])
                && approx_eq(mine[1], theirs.wb[1])
                && approx_eq(mine[2], theirs.wb[2]);
            levels["white_balance"] =
                json!({ "ours_normalised": mine, "rawler": theirs.wb, "equal": ok });
            levels_ok &= ok;
        }
    }
    report["levels"] = levels;

    let d = diff(&image.data.data, rows, image.data.width);
    report["pixels"] = diff_json(&d);
    let outcome = if reduced {
        // rawler's YCbCr matrix and chroma interpolation are not ours; the numbers are information.
        let mut sum = 0u64;
        for (a, b) in image.data.data.iter().zip(rows) {
            sum += u64::from(a.abs_diff(*b));
        }
        report["mean_abs_diff"] = json!(sum as f64 / image.data.data.len().max(1) as f64);
        Outcome::Informational
    } else if d.mismatched == 0 {
        Outcome::Identical
    } else if ours.meta.decode == DecodeKind::Craw {
        let (inside, worst_steps) = craw_dither_check(ours, rows);
        report["craw"] = json!({
            "outside_dither_window": inside,
            "max_abs_diff_in_curve_steps": worst_steps,
        });
        if inside == 0 {
            Outcome::WithinTolerance
        } else {
            Outcome::Mismatch
        }
    } else {
        Outcome::Mismatch
    };
    let outcome = if !levels_ok && matches!(outcome, Outcome::Identical | Outcome::WithinTolerance)
    {
        Outcome::Mismatch
    } else {
        outcome
    };
    // A difference on a stream shape rawler was not written for is information, not a failure.
    if outcome == Outcome::Mismatch
        && let Some(why) = rawler_limitation(path, ours)
    {
        report["limitation"] = json!(why);
        return (Outcome::Informational, false, report);
    }
    (outcome, !levels_ok, report)
}

/// rawler's cRAW dither: the output is `base + ((delta * r + 1024) >> 12)` for a pseudo-random
/// `r` in `0..2048`, with `base = c - (delta + 2) / 4` and `delta = curve[i + 1] - curve[i - 1]`
/// around the curve index `i = 2 * code` of the undithered value `c` (rawler `LookupTable`).
/// Returns how many samples lie outside that window and the largest difference in units of the
/// local curve step per code.
fn craw_dither_check(ours: &RawImage, theirs: &[u16]) -> (usize, f64) {
    let Some(knots) = ours.tone_curve else {
        return (usize::MAX, f64::NAN);
    };
    let Ok(curve) = build_curve(knots) else {
        return (usize::MAX, f64::NAN);
    };
    let on_grid: Vec<u16> = (0..2048).map(|code| curve[2 * code]).collect();
    let mut outside = 0usize;
    let mut worst = 0f64;
    for (&c, &t) in ours.data.data.iter().zip(theirs) {
        let Ok(code) = on_grid.binary_search(&c) else {
            outside += 1;
            continue;
        };
        let i = 2 * code;
        let lower = if i > 0 { curve[i - 1] } else { curve[i] };
        let upper = if i < curve.len() - 1 {
            curve[i + 1]
        } else {
            curve[i]
        };
        let delta = u32::from(upper - lower);
        let base = if c == 0 {
            0
        } else {
            u32::from(c) - (delta + 2) / 4
        };
        let lo = base;
        let hi = base + ((delta * 2047 + 1024) >> 12);
        let t32 = u32::from(t);
        if t32 < lo || t32 > hi {
            outside += 1;
        }
        if delta > 0 {
            worst = worst.max(f64::from(c.abs_diff(t)) / f64::from(delta));
        }
    }
    (outside, worst)
}

/// Compares one file with the oracles and returns its report entry.
pub fn compare_file(
    path: &Path,
    libraw: Option<&Runner>,
    libjpeg: Option<&Runner>,
    with_rawler: bool,
    tally: &mut Tally,
) -> Value {
    let started = Instant::now();
    let ours = decode_with(path, &DecodeOptions::default());
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ours = match ours {
        Ok(image) => image,
        Err(e) => {
            // Unsupported variants (ARW 6.0) are expected outcomes, anything else is reported.
            let variant = match &e {
                ym_raw::error::RawError::Unsupported { variant, .. } => variant.id().to_owned(),
                _ => "unreadable".to_owned(),
            };
            let row = tally.row(&variant);
            row.files += 1;
            row.ours_error += 1;
            if variant == "unreadable" {
                row.unreadable += 1;
            }
            return json!({
                "path": path.display().to_string(),
                "variant": variant,
                "ours": { "status": "error", "kind": e.kind(), "message": e.to_string() },
                "libraw": { "outcome": Outcome::Skipped.name() },
                "libjpeg": { "outcome": Outcome::Skipped.name() },
                "rawler": { "outcome": Outcome::Skipped.name() },
            });
        }
    };
    let variant = variant_name(&ours);
    // LibRaw's pseudo-raw path has no white-balance undo and replicates chroma.
    let pseudo = ours.is_reduced().then(|| {
        decode_with(
            path,
            &DecodeOptions {
                ycbcr: YcbcrOptions {
                    chroma: ChromaUpsampling::Replicate,
                    undo_white_balance: false,
                },
            },
        )
        .ok()
    });
    let pseudo = pseudo.flatten();

    let mut levels_mismatch = false;
    let (libraw_outcome, libraw_report) = match libraw {
        None => (Outcome::Skipped, json!({})),
        Some(runner) => match runner.dump(path) {
            Ok(Some(dump)) => {
                let (o, lm, r) = against_libraw(&ours, pseudo.as_ref(), &dump);
                levels_mismatch |= lm;
                (o, r)
            }
            Ok(None) => (Outcome::Skipped, json!({})),
            Err(e) => {
                // LibRaw refusing a file is information; failing to start LibRaw is not.
                (Outcome::OracleError, json!({ "error": format!("{e:#}") }))
            }
        },
    };
    // libjpeg decodes the tiles of Lossless L files, which exercises predictors and restart
    // markers that LibRaw's Sony loader does not handle.
    let (libjpeg_outcome, libjpeg_report) = match libjpeg {
        Some(runner) if variant == "lossless_l" => match runner.dump(path) {
            Ok(Some(dump)) => against_libjpeg(&ours, &dump),
            Ok(None) => (Outcome::Skipped, json!({})),
            Err(e) => (Outcome::OracleError, json!({ "error": format!("{e:#}") })),
        },
        _ => (Outcome::Skipped, json!({})),
    };
    let (rawler_outcome, rawler_report) = if with_rawler {
        let (o, lm, r) = against_rawler(path, &ours, pseudo.as_ref());
        levels_mismatch |= lm;
        (o, r)
    } else {
        (Outcome::Skipped, json!({}))
    };
    let row = tally.row(variant);
    row.files += 1;
    *row.libraw.entry(libraw_outcome.name()).or_default() += 1;
    *row.libjpeg.entry(libjpeg_outcome.name()).or_default() += 1;
    *row.rawler.entry(rawler_outcome.name()).or_default() += 1;
    if levels_mismatch {
        row.levels_mismatch += 1;
    }

    let (black, white, wb) = levels_rggb(&ours);
    let mut libraw_report = libraw_report;
    libraw_report["outcome"] = json!(libraw_outcome.name());
    let mut libjpeg_report = libjpeg_report;
    libjpeg_report["outcome"] = json!(libjpeg_outcome.name());
    let mut rawler_report = rawler_report;
    rawler_report["outcome"] = json!(rawler_outcome.name());
    json!({
        "path": path.display().to_string(),
        "model": ours.meta.model,
        "variant": variant,
        "width": ours.pixel_width(),
        "height": ours.pixel_height(),
        "decode_ms": decode_ms,
        "ours": { "status": "ok", "black_rggb": black, "white_rggb": white, "wb_rggb": wb },
        "libraw": libraw_report,
        "libjpeg": libjpeg_report,
        "rawler": rawler_report,
    })
}

/// One line per file for the console.
pub fn one_line(entry: &Value) -> String {
    format!(
        "{:<44} {:<15} libraw {:<16} libjpeg {:<10} rawler {}",
        Path::new(entry["path"].as_str().unwrap_or("?"))
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        entry["variant"].as_str().unwrap_or("?"),
        entry["libraw"]["outcome"].as_str().unwrap_or("?"),
        entry["libjpeg"]["outcome"].as_str().unwrap_or("?"),
        entry["rawler"]["outcome"].as_str().unwrap_or("?"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_arrays_are_identical_and_a_difference_is_located() {
        let a = [1u16, 2, 3, 4, 5, 6];
        let same = diff(&a, &a, 3);
        assert_eq!(same.mismatched, 0);
        let mut b = a;
        b[4] = 9;
        let d = diff(&a, &b, 3);
        assert_eq!((d.mismatched, d.max_abs), (1, 4));
        assert_eq!(d.first[0]["x"], 1);
        assert_eq!(d.first[0]["y"], 1);
    }

    #[test]
    fn a_dump_of_the_wrong_length_is_never_identical() {
        let a = [7u16; 10];
        assert_eq!(diff(&a, &a[..6], 5).mismatched, 4);
        let longer = [7u16; 13];
        assert_eq!(diff(&a, &longer, 5).mismatched, 3);
        assert_eq!(diff(&a, &[], 5).mismatched, 10);
    }
}
