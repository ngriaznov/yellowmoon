//! Reference Grade, proof of concept: fit a node's corrector fields to a reference picture at run time
//! (`research/reference-grade-2026-11-23.md`, "Recommendation: fit, don't predict", Phase 1).
//!
//! A CPU-only command line tool. It changes no shipped behaviour and no kernel.
//!
//! ```text
//! export YELLOWMOON_DATA=$PWD/data
//! cargo run --release -p ym-core --example refgrade_poc -- \
//!     --photo <file.ARW> --reference <image.jpg|png> --out <folder> [--strength 1.0] [--no-hue]
//! cargo run --release -p ym-core --example refgrade_poc -- \
//!     --photos <dir|file>... --references <dir|file>... --out <folder>      # every pair, plus grid.jpg per photo
//! cargo run --release -p ym-core --example refgrade_poc -- \
//!     --selftest --photo <file.ARW> --out <folder>                          # E1: recover known grades
//! ```
//!
//! # What it does
//!
//! 1. **The photo at the new node's position.** The photo is opened the way the app opens it (Signature as the
//!    default look, every setting automatic). It is rendered on the CPU with the Output look at Neutral to
//!    `FullTarget::DngLinearAwg4`, which is C1 to C11 of Main: linear AWG4 at the position of a node appended after
//!    Main. That picture is box-filtered (in linear light) to 1600 px for the renders and 512 px for fitting, and
//!    cached under `<out>/cache`. Output sharpening, capture sharpening and the Output's own crop do not apply (a
//!    simplification, noted in the report).
//! 2. **The node's parameters (30, all identity at 0):** the luma curve (8 values: an offset and 7 slopes of 8 knots
//!    at -6, -4, -2, -1, 0, +1, +2, +4 stops from 0.18), the R, G and B curves (3 knots each at -3, 0, +3 stops), the
//!    C9 saturation, and two hue curves of 6 knots each (`hue_vs_sat`, `hue_vs_hue`). Everything but the hue curves
//!    is a real corrector setting and runs through the real CPU chain (`ColorChain` with the values as Main's grade
//!    and curves: the code of a corrector node, C9 then C9b then C11).
//! 3. **Where the hue step is.** [`Candidate::apply`]: the hue curves are applied by `ParametricLook::evaluate`
//!    (the parametric look's periodic hue curves, in the LogC4 `e1, e2` chroma plane) on the LogC4 value that the
//!    corrector returns. That is after C9, C9b and C11 and before the Output look (C10), the cine rendering (C12)
//!    and the encode (C13).
//! 4. **Objective** on display CIELAB (see the `W_*` and `*_SCALE` constants), **optimiser**: fixed passes of
//!    golden-section line searches over the parameters, in stages (tone, colour per band, saturation and hue-vs-sat,
//!    hue-vs-hue, joint polish). No random numbers; the per-pixel work is parallel but every pixel is independent and the
//!    reductions run in a fixed order, so the result does not depend on the thread count.
//!
//! References are treated as sRGB (a simplification: no ICC handling). Uniform letterbox bars are stripped first.

// A proof of concept: index loops over the three channels read better than iterator chains here.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde_json::{Value, json};
use ym_color::consts::{REC709, linalg};
use ym_color::curve::{Curve, CurveSet};
use ym_color::deltae::{Summary, d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::display::{ChainParams, ColorChain, LookSetting, OutputTarget};
use ym_color::grade::{GradeParams, logc4_at_stops};
use ym_color::look::{HuePoint, ParametricLook, ParametricLookSpec, SCHEMA_ID};
use ym_color::tone::ToneParams;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{FullRequest, FullTarget, PhotoSource};
use ym_core::session::OpenOptions;
use ym_gpu::queue::CancelToken;

type R<T> = Result<T, Box<dyn Error>>;

// ---------------------------------------------------------------------------------------------
// Constants (every one named; the objective's weights and scales are the tuning knobs)
// ---------------------------------------------------------------------------------------------

/// Long edge of the picture the fit reads, in pixels.
const FIT_LONG_EDGE_PX: usize = 512;
/// Long edge of the renders in the sheet, in pixels.
const SHOW_LONG_EDGE_PX: usize = 1600;
/// Columns and rows of the stratified sample grid: one pixel per cell, jittered by a seeded generator.
const SAMPLE_GRID: (usize, usize) = (128, 64);
/// The seed of the sample jitter and of the self-test's random grades.
const SEED: u64 = 0x5EED_2026_1124;
/// Width of a panel of `sheet.jpg`, in pixels.
const PANEL_W_PX: usize = 640;
/// Width of a cell of `grid.jpg`, in pixels.
const GRID_CELL_W_PX: usize = 360;
/// JPEG quality of the sheets.
const JPEG_QUALITY: u8 = 92;
/// Letterbox: a row or column is a bar when its standard deviation is below this (code values)...
const BAR_STD_MAX: f64 = 2.0;
/// ...and its mean is below this (code values).
const BAR_MEAN_MAX: f64 = 16.0;
/// The most of a side a bar strip may take.
const BAR_MAX_FRACTION: f64 = 0.35;

/// Luma curve knots, in stops from 0.18 (scene grey).
const LUMA_STOPS: [f64; 8] = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
/// R, G and B curve knots, in stops from 0.18.
const RGB_STOPS: [f64; 3] = [-3.0, 0.0, 3.0];
/// Hue curve knots, in degrees of the LogC4 chroma plane.
const HUE_KNOTS_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// Number of parameters: 8 luma, 9 RGB, 1 saturation, 6 hue-vs-sat, 6 hue-vs-hue.
const N_PARAMS: usize = 30;
/// Index ranges of the parameter groups.
const LUMA_RANGE: std::ops::Range<usize> = 0..8;
const RGB_RANGE: std::ops::Range<usize> = 8..17;
const SAT_INDEX: usize = 17;
const HSAT_RANGE: std::ops::Range<usize> = 18..24;
const HHUE_RANGE: std::ops::Range<usize> = 24..30;

/// Bound of the luma curve's offset of the first knot, in stops.
const BOUND_LUMA_OFFSET_STOPS: f64 = 2.0;
/// Bound of the natural log of a luma curve segment's slope (in stop space).
const BOUND_LUMA_LOG_SLOPE: f64 = 1.1;
/// Bound of an RGB curve knot's shift, in stops.
const BOUND_RGB_STOPS: f64 = 1.0;
/// Bound of the C9 saturation, in percent.
const BOUND_SATURATION_PCT: f64 = 60.0;
/// Bound of `hue_vs_sat` around 1 (the multiplier runs 0.3 to 1.7).
const BOUND_HUE_SAT: f64 = 0.7;
/// Bound of `hue_vs_hue`, in degrees.
const BOUND_HUE_DEG: f64 = 30.0;

/// Weights of the objective terms.
const W_SHAPE: f64 = 1.0;
const W_LEVEL: f64 = 0.15;
const W_AB: f64 = 1.0;
const W_SECTOR: f64 = 1.0;
const W_CHROMA: f64 = 0.5;
const W_REG: f64 = 0.03;
const W_CLIP: f64 = 3.0;
/// Scales that make the terms unitless: an error of this size counts as 1.
const L_SCALE: f64 = 5.0;
const AB_SCALE: f64 = 4.0;
const CHROMA_SCALE: f64 = 6.0;
/// A clipped share error of this size counts as 1.
const CLIP_SCALE: f64 = 0.02;
/// Clipped share the photo may exceed the reference's by before the penalty starts.
const CLIP_SLACK: f64 = 0.005;
/// A display channel at or above this encoded value counts as clipped white...
const CLIP_HI_ENCODED: f64 = 0.995;
/// ...and all channels at or below this as crushed black.
const CLIP_LO_ENCODED: f64 = 0.004;
/// Quantile levels of L*, in percent.
const QUANTILES_PCT: [f64; 19] = [
    5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
    85.0, 90.0, 95.0,
];
/// Hue sectors of CIELAB hue, 60 degrees each, the first centred on 0.
const SECTORS: usize = 6;

/// The optimiser's schedule: (parameters, half range as a fraction of the bound, golden-section iterations).
const STAGES: [(&str, std::ops::Range<usize>, f64, usize); 10] = [
    ("tone", LUMA_RANGE, 1.0, 8),
    ("tone", LUMA_RANGE, 0.4, 6),
    ("colour per band", RGB_RANGE, 1.0, 7),
    ("colour per band", RGB_RANGE, 0.4, 6),
    (
        "saturation and hue-vs-sat",
        SAT_INDEX..HSAT_RANGE.end,
        1.0,
        7,
    ),
    (
        "saturation and hue-vs-sat",
        SAT_INDEX..HSAT_RANGE.end,
        0.4,
        6,
    ),
    ("hue-vs-hue", HHUE_RANGE, 1.0, 7),
    ("hue-vs-hue", HHUE_RANGE, 0.4, 6),
    ("joint polish", 0..N_PARAMS, 0.25, 6),
    ("joint polish", 0..N_PARAMS, 0.1, 5),
];

fn bound(i: usize) -> f64 {
    match i {
        0 => BOUND_LUMA_OFFSET_STOPS,
        1..=7 => BOUND_LUMA_LOG_SLOPE,
        8..=16 => BOUND_RGB_STOPS,
        17 => BOUND_SATURATION_PCT,
        18..=23 => BOUND_HUE_SAT,
        _ => BOUND_HUE_DEG,
    }
}

fn param_name(i: usize) -> String {
    match i {
        0 => "luma.offset_stops".into(),
        1..=7 => format!("luma.log_slope[{}]", i),
        8..=16 => {
            let c = ["red", "green", "blue"][(i - 8) / 3];
            format!("{c}.shift_stops[{:+}]", RGB_STOPS[(i - 8) % 3])
        }
        17 => "saturation_pct".into(),
        18..=23 => format!("hue_vs_sat[{}deg]", HUE_KNOTS_DEG[i - 18]),
        _ => format!("hue_vs_hue[{}deg]", HUE_KNOTS_DEG[i - 24]),
    }
}

// ---------------------------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------------------------

/// SplitMix64: a seeded generator with no state beyond one word.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

// ---------------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------------

/// An interleaved image of linear RGB (sRGB primaries for display pictures, AWG4 for node pictures).
#[derive(Clone)]
struct Img {
    w: usize,
    h: usize,
    px: Vec<f32>,
}

impl Img {
    fn at(&self, x: usize, y: usize) -> [f64; 3] {
        let i = 3 * (y * self.w + x);
        [
            f64::from(self.px[i]),
            f64::from(self.px[i + 1]),
            f64::from(self.px[i + 2]),
        ]
    }
}

/// Area (box) resampling with fractional edges, in the image's own (linear) values.
fn resize_area(src: &Img, dw: usize, dh: usize) -> Img {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    // Horizontal pass, then vertical. f64 accumulation, rows in parallel (each row is independent).
    let weights = |sn: usize, dn: usize| -> Vec<Vec<(usize, f64)>> {
        let scale = sn as f64 / dn as f64;
        (0..dn)
            .map(|d| {
                let (a, b) = (d as f64 * scale, (d + 1) as f64 * scale);
                let (i0, i1) = (a.floor() as usize, (b.ceil() as usize).min(sn));
                (i0..i1)
                    .map(|i| {
                        let lo = a.max(i as f64);
                        let hi = b.min((i + 1) as f64);
                        (i, ((hi - lo) / scale).max(0.0))
                    })
                    .collect()
            })
            .collect()
    };
    let wx = weights(src.w, dw);
    let wy = weights(src.h, dh);
    let mut mid = vec![0f32; 3 * dw * src.h];
    mid.par_chunks_mut(3 * dw).enumerate().for_each(|(y, row)| {
        for (x, w) in wx.iter().enumerate() {
            let mut acc = [0f64; 3];
            for &(i, k) in w {
                let s = 3 * (y * src.w + i);
                for c in 0..3 {
                    acc[c] += k * f64::from(src.px[s + c]);
                }
            }
            for c in 0..3 {
                row[3 * x + c] = acc[c] as f32;
            }
        }
    });
    let mut out = vec![0f32; 3 * dw * dh];
    out.par_chunks_mut(3 * dw).enumerate().for_each(|(y, row)| {
        for x in 0..dw {
            let mut acc = [0f64; 3];
            for &(j, k) in &wy[y] {
                let s = 3 * (j * dw + x);
                for c in 0..3 {
                    acc[c] += k * f64::from(mid[s + c]);
                }
            }
            for c in 0..3 {
                row[3 * x + c] = acc[c] as f32;
            }
        }
    });
    Img {
        w: dw,
        h: dh,
        px: out,
    }
}

fn fit_size(w: usize, h: usize, long_edge: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= long_edge {
        return (w, h);
    }
    let s = long_edge as f64 / long as f64;
    (
        ((w as f64 * s).round() as usize).max(1),
        ((h as f64 * s).round() as usize).max(1),
    )
}

/// Rotates and flips a buffer the way an EXIF orientation says (1 to 8).
fn orient(img: &Img, exif: u16) -> Img {
    let (sw, sh) = (img.w, img.h);
    let swap = (5..=8).contains(&exif);
    let (dw, dh) = if swap { (sh, sw) } else { (sw, sh) };
    let mut out = vec![0f32; 3 * dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let (sx, sy) = match exif {
                2 => (sw - 1 - x, y),
                3 => (sw - 1 - x, sh - 1 - y),
                4 => (x, sh - 1 - y),
                5 => (y, x),
                6 => (y, sh - 1 - x),
                7 => (sw - 1 - y, sh - 1 - x),
                8 => (sw - 1 - y, x),
                _ => (x, y),
            };
            let (s, d) = (3 * (sy * sw + sx), 3 * (y * dw + x));
            out[d..d + 3].copy_from_slice(&img.px[s..s + 3]);
        }
    }
    Img {
        w: dw,
        h: dh,
        px: out,
    }
}

/// An 8-bit sRGB picture.
struct Rgb8 {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

fn load_reference_file(path: &Path) -> R<Rgb8> {
    let bytes = fs::read(path)?;
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if ext == "png" {
        let mut dec = png::Decoder::new(std::io::Cursor::new(&bytes));
        dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = dec.read_info()?;
        let mut buf = vec![0u8; reader.output_buffer_size().ok_or("png too large")?];
        let info = reader.next_frame(&mut buf)?;
        let (w, h) = (info.width as usize, info.height as usize);
        let n = match info.color_type {
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Indexed => return Err("indexed png was not expanded".into()),
        };
        return Ok(Rgb8 {
            w,
            h,
            data: expand_to_rgb(&buf[..w * h * n], n),
        });
    }
    let mut dec = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&bytes));
    let pixels = dec
        .decode()
        .map_err(|e| format!("{}: {e:?}", path.display()))?;
    let info = dec.info().ok_or("no jpeg info")?;
    let (w, h) = (usize::from(info.width), usize::from(info.height));
    let n = pixels.len() / (w * h);
    Ok(Rgb8 {
        w,
        h,
        data: expand_to_rgb(&pixels, n),
    })
}

fn expand_to_rgb(buf: &[u8], n: usize) -> Vec<u8> {
    match n {
        3 => buf.to_vec(),
        4 => buf
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        2 => buf
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        _ => buf.iter().flat_map(|&g| [g, g, g]).collect(),
    }
}

/// Strips uniform letterbox bars: edge rows or columns with a standard deviation under [`BAR_STD_MAX`] and a mean
/// under [`BAR_MEAN_MAX`] (over all their samples). Returns the picture and the bars as (top, bottom, left, right).
fn strip_bars(img: &Rgb8) -> (Rgb8, [usize; 4]) {
    let stats = |samples: &mut dyn Iterator<Item = u8>| -> (f64, f64) {
        let (mut n, mut s, mut s2) = (0f64, 0f64, 0f64);
        for v in samples {
            let v = f64::from(v);
            n += 1.0;
            s += v;
            s2 += v * v;
        }
        let mean = s / n.max(1.0);
        (mean, (s2 / n.max(1.0) - mean * mean).max(0.0).sqrt())
    };
    let is_bar = |(mean, std): (f64, f64)| std < BAR_STD_MAX && mean < BAR_MEAN_MAX;
    let (w, h) = (img.w, img.h);
    let row = |y: usize, x0: usize, x1: usize| {
        stats(&mut img.data[3 * (y * w + x0)..3 * (y * w + x1)].iter().copied())
    };
    let col = |x: usize, y0: usize, y1: usize| {
        stats(&mut (y0..y1).flat_map(|y| {
            img.data[3 * (y * w + x)..3 * (y * w + x) + 3]
                .iter()
                .copied()
        }))
    };
    let (max_v, max_h) = (
        (h as f64 * BAR_MAX_FRACTION) as usize,
        (w as f64 * BAR_MAX_FRACTION) as usize,
    );
    let (mut top, mut bottom) = (0, 0);
    while top < max_v && is_bar(row(top, 0, w)) {
        top += 1;
    }
    while bottom < max_v && is_bar(row(h - 1 - bottom, 0, w)) {
        bottom += 1;
    }
    let (mut left, mut right) = (0, 0);
    while left < max_h && is_bar(col(left, top, h - bottom)) {
        left += 1;
    }
    while right < max_h && is_bar(col(w - 1 - right, top, h - bottom)) {
        right += 1;
    }
    let (nw, nh) = (w - left - right, h - top - bottom);
    let mut data = Vec::with_capacity(3 * nw * nh);
    for y in top..h - bottom {
        data.extend_from_slice(&img.data[3 * (y * w + left)..3 * (y * w + left + nw)]);
    }
    (Rgb8 { w: nw, h: nh, data }, [top, bottom, left, right])
}

/// An 8-bit sRGB picture as linear sRGB floats.
fn rgb8_to_linear(img: &Rgb8) -> Img {
    let table: Vec<f32> = (0..256)
        .map(|v| srgb_to_linear(f64::from(v) / 255.0) as f32)
        .collect();
    Img {
        w: img.w,
        h: img.h,
        px: img.data.iter().map(|&v| table[usize::from(v)]).collect(),
    }
}

/// A linear sRGB picture as 8-bit sRGB.
fn linear_to_rgb8(img: &Img) -> Rgb8 {
    let data = img
        .px
        .iter()
        .map(|&v| (linear_to_srgb(f64::from(v)) * 255.0).round() as u8)
        .collect();
    Rgb8 {
        w: img.w,
        h: img.h,
        data,
    }
}

// ---------------------------------------------------------------------------------------------
// The photo at the node's position
// ---------------------------------------------------------------------------------------------

/// The Output the photo has by default, kept to build the chain after the node.
struct OutputStage {
    look: Arc<ym_color::look::LookTransform>,
    intensity_pct: f64,
    look_id: String,
}

/// A photo as the fit sees it.
struct PhotoNode {
    /// Linear AWG4 at the node's input, long edge [`SHOW_LONG_EDGE_PX`].
    show: Img,
    output: OutputStage,
    notes: Vec<String>,
}

fn write_cache(path: &Path, img: &Img) -> R<()> {
    let mut out = Vec::with_capacity(16 + 4 * img.px.len());
    out.extend_from_slice(&(img.w as u64).to_le_bytes());
    out.extend_from_slice(&(img.h as u64).to_le_bytes());
    for v in &img.px {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out)?;
    Ok(())
}

fn read_cache(path: &Path) -> Option<Img> {
    let b = fs::read(path).ok()?;
    let w = u64::from_le_bytes(b.get(0..8)?.try_into().ok()?) as usize;
    let h = u64::from_le_bytes(b.get(8..16)?.try_into().ok()?) as usize;
    if b.len() != 16 + 12 * w * h {
        return None;
    }
    let px = b[16..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(Img { w, h, px })
}

fn open_core(out: &Path) -> R<Core> {
    // Preferences and looks in a scratch folder: the tool must not read or write the person's.
    let mut options = CoreOptions::from_environment()?;
    options.app_root = out.join(".app-home");
    options.thumbnails = false;
    fs::create_dir_all(&options.app_root)?;
    Ok(Core::open(options)?)
}

fn load_photo(core: &Core, file: &Path, cache: &Path) -> R<PhotoNode> {
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let session = &core.session;
    let outcome = session.open(&[file.to_path_buf()], OpenOptions::default());
    let id = *outcome
        .added
        .first()
        .ok_or("the session did not take the file")?;
    if !session.wait_idle(Duration::from_secs(900)) {
        return Err("the photo did not finish developing in 15 minutes".into());
    }
    let view = session.view(id)?;
    let raw = session.raw_image(id)?;
    let src = PhotoSource::from_view(&view, raw);
    let look_id = src.settings.output().look.clone();
    let intensity_pct = src.settings.output().look_intensity;
    let resolved = core
        .looks
        .provider()
        .look(&look_id)
        .map_err(|e| format!("look {look_id:?}: {e}"))?;
    let output = OutputStage {
        look: Arc::clone(&resolved.transform),
        intensity_pct,
        look_id: format!("{look_id:?}"),
    };
    let mut notes = Vec::new();
    let cache_file = cache.join(format!("{stem}.node{SHOW_LONG_EDGE_PX}.f32"));
    if let Some(show) = read_cache(&cache_file) {
        notes.push("node picture read from the cache".into());
        return Ok(PhotoNode {
            show,
            output,
            notes,
        });
    }
    // The node's input: C1 to C11 of Main with the Output look at Neutral, which is the DNG target's linear AWG4.
    let mut node_src = src.clone();
    node_src.settings.output_mut().look = ym_looks::library::LookId::neutral();
    let cancel = CancelToken::new();
    let t = Instant::now();
    let full = core.renderer.render_full(
        &node_src,
        &FullRequest {
            target: FullTarget::DngLinearAwg4,
        },
        &cancel,
    )?;
    notes.push(format!(
        "full-resolution CPU render to linear AWG4: {:.1} s, {}x{}",
        t.elapsed().as_secs_f64(),
        full.width,
        full.height
    ));
    let raw_img = Img {
        w: full.width,
        h: full.height,
        px: full.rgb.clone(),
    };
    drop(full);
    let geometry = core.renderer.geometry(&node_src, &cancel)?;
    let oriented = orient(&raw_img, geometry.exif_orientation);
    drop(raw_img);
    let final_size = core.renderer.final_size(&node_src, &cancel)?;
    if (oriented.w, oriented.h) != final_size {
        notes.push(format!(
            "warning: node picture {}x{} differs from the final size {:?}",
            oriented.w, oriented.h, final_size
        ));
    }
    let (sw, sh) = fit_size(oriented.w, oriented.h, SHOW_LONG_EDGE_PX);
    let show = resize_area(&oriented, sw, sh);
    fs::create_dir_all(cache)?;
    write_cache(&cache_file, &show)?;
    Ok(PhotoNode {
        show,
        output,
        notes,
    })
}

// ---------------------------------------------------------------------------------------------
// The parameter family: normalised values, real units, the chain
// ---------------------------------------------------------------------------------------------

/// The parameters in real units, as written to `params.json` and `settings.json`.
#[derive(Clone, Debug)]
struct Real {
    /// (stop, input code, output stops, output code) of the 8 luma knots.
    luma: Vec<(f64, f64, f64, f64)>,
    /// Per channel: (stop, input code, output stops, output code) of 3 knots.
    rgb: [Vec<(f64, f64, f64, f64)>; 3],
    saturation_pct: f64,
    hue_vs_sat: [f64; 6],
    hue_vs_hue_deg: [f64; 6],
}

/// Normalised parameters `x` (identity at 0, each in -1 to 1) to real units, scaled by `strength`.
fn to_real(x: &[f64; N_PARAMS], strength: f64) -> Real {
    let t = |i: usize| strength * x[i] * bound(i);
    // Luma: the first knot's output offset, then the slope of each segment in stop space.
    let mut luma = Vec::new();
    let mut y_stops = LUMA_STOPS[0] + t(0);
    let mut prev_code = 0.0f64;
    for (k, &s) in LUMA_STOPS.iter().enumerate() {
        if k > 0 {
            y_stops += (s - LUMA_STOPS[k - 1]) * t(k).exp();
        }
        let code = logc4_at_stops(y_stops).clamp(prev_code, 1.0);
        prev_code = code;
        luma.push((s, logc4_at_stops(s), y_stops, code));
    }
    let mut rgb: [Vec<(f64, f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (c, curve) in rgb.iter_mut().enumerate() {
        for (k, &s) in RGB_STOPS.iter().enumerate() {
            let y = s + t(8 + 3 * c + k);
            curve.push((s, logc4_at_stops(s), y, logc4_at_stops(y).clamp(0.0, 1.0)));
        }
    }
    let mut hue_vs_sat = [1.0; 6];
    let mut hue_vs_hue_deg = [0.0; 6];
    for k in 0..6 {
        hue_vs_sat[k] = 1.0 + t(HSAT_RANGE.start + k);
        hue_vs_hue_deg[k] = t(HHUE_RANGE.start + k);
    }
    Real {
        luma,
        rgb,
        saturation_pct: t(SAT_INDEX),
        hue_vs_sat,
        hue_vs_hue_deg,
    }
}

fn curve_from_knots(knots: &[(f64, f64, f64, f64)]) -> R<Curve> {
    let mut pts = vec![(0.0, 0.0)];
    pts.extend(knots.iter().map(|k| (k.1, k.3)));
    pts.push((1.0, 1.0));
    Ok(Curve::new(&pts)?)
}

/// A candidate corrector: the real chain of a node (as Main's grade and curves), and the POC's hue step.
struct Candidate {
    /// C9, C9b and C11 of a node, through `ColorChain` with an identity matrix and tone, no look.
    node: ColorChain,
    /// The hue curves, applied to the node's output (`None` when both are neutral).
    hue: Option<ParametricLook>,
}

impl Candidate {
    fn new(real: &Real, with_hue: bool) -> R<Self> {
        let mut p = ChainParams::new(linalg::IDENTITY, ToneParams::IDENTITY);
        p.grade = GradeParams {
            saturation: real.saturation_pct,
            ..GradeParams::default()
        };
        p.curves = CurveSet {
            luma: curve_from_knots(&real.luma)?,
            r: curve_from_knots(&real.rgb[0])?,
            g: curve_from_knots(&real.rgb[1])?,
            b: curve_from_knots(&real.rgb[2])?,
        };
        p.output = OutputTarget::Srgb;
        p.dither = false;
        let node = ColorChain::new(&p)?;
        let neutral = real.hue_vs_sat.iter().all(|v| *v == 1.0)
            && real.hue_vs_hue_deg.iter().all(|v| *v == 0.0);
        let hue = if with_hue && !neutral {
            let points = |v: &[f64; 6]| -> Vec<HuePoint> {
                HUE_KNOTS_DEG
                    .iter()
                    .zip(v)
                    .map(|(&h, &value)| HuePoint { hue_deg: h, value })
                    .collect()
            };
            let spec = ParametricLookSpec {
                schema: SCHEMA_ID.to_owned(),
                name: "reference-grade-hue".to_owned(),
                input: None,
                luma_curve: Vec::new(),
                cdl: None,
                hue_vs_sat: points(&real.hue_vs_sat),
                hue_vs_hue: points(&real.hue_vs_hue_deg),
                luma_vs_sat: Vec::new(),
                red_chroma: None,
                red_density: None,
            };
            Some(ParametricLook::from_spec(&spec)?)
        } else {
            None
        };
        Ok(Self { node, hue })
    }

    /// The node's output for a pixel at its input (linear AWG4), as linear AWG4 for the Output.
    ///
    /// Order: the corrector (C9, C9b, C11, the real code), then the POC hue step on its LogC4 result, then back to
    /// linear for the Output's look (C10), the cine rendering (C12) and the encode (C13).
    fn apply(&self, lin: [f64; 3]) -> [f64; 3] {
        let v = self.node.apply_logc4_pixel(lin);
        match &self.hue {
            None => v.to_linear().0,
            Some(look) => {
                let h = look.evaluate(v.0);
                ym_color::logc::LogC4(h).to_linear().0
            }
        }
    }
}

/// The Output after the node: the look, the cine rendering and the sRGB encode.
fn make_output_chain(out: &OutputStage) -> R<ColorChain> {
    let mut p = ChainParams::new(linalg::IDENTITY, ToneParams::IDENTITY);
    p.look = Some(LookSetting::new(Arc::clone(&out.look), out.intensity_pct));
    p.output = OutputTarget::Srgb;
    p.dither = false;
    Ok(ColorChain::new(&p)?)
}

/// Display-linear sRGB of one node-input pixel (not clamped).
fn render_pixel(cand: &Candidate, output: &ColorChain, lin: [f64; 3]) -> [f64; 3] {
    output.apply_to_display_linear(cand.apply(lin))
}

// ---------------------------------------------------------------------------------------------
// Colour measurement and the objective
// ---------------------------------------------------------------------------------------------

/// One measured pixel.
#[derive(Clone, Copy, Default)]
struct Px {
    lab: [f64; 3],
    hi: bool,
    lo: bool,
}

struct LabConv {
    m: [[f64; 3]; 3],
    white: [f64; 3],
    hi: f64,
    lo: f64,
}

impl LabConv {
    fn new() -> Self {
        Self {
            m: REC709.rgb_to_xyz(),
            white: d65_white_xyz(),
            hi: srgb_to_linear(CLIP_HI_ENCODED),
            lo: srgb_to_linear(CLIP_LO_ENCODED),
        }
    }
    /// Display-linear sRGB (clamped to the display's range, as an 8-bit file holds it) to a measured pixel.
    fn px(&self, lin: [f64; 3]) -> Px {
        let c = lin.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) });
        let xyz = linalg::mul_vec(&self.m, c);
        Px {
            lab: xyz_to_lab(xyz, self.white),
            hi: c.iter().any(|&v| v >= self.hi),
            lo: c.iter().all(|&v| v <= self.lo),
        }
    }
}

/// What the objective reads from a picture.
#[derive(Clone, Debug)]
struct Stats {
    /// L* quantiles minus the median.
    shape: [f64; 19],
    median: f64,
    /// Mean (a*, b*) in the darkest, middle and lightest third of the picture's own L*.
    band_ab: [[f64; 2]; 3],
    sector_chroma: [f64; SECTORS],
    sector_share: [f64; SECTORS],
    mean_chroma: f64,
    hi_share: f64,
    lo_share: f64,
}

fn stats_of(px: &[Px]) -> Stats {
    let n = px.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| px[a].lab[0].total_cmp(&px[b].lab[0]).then(a.cmp(&b)));
    let l_at = |q_pct: f64| {
        let pos = q_pct / 100.0 * (n - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        let (a, b) = (px[idx[lo]].lab[0], px[idx[hi]].lab[0]);
        a + (b - a) * (pos - lo as f64)
    };
    let median = l_at(50.0);
    let mut shape = [0.0; 19];
    for (s, q) in shape.iter_mut().zip(QUANTILES_PCT) {
        *s = l_at(q) - median;
    }
    let mut band_ab = [[0.0; 2]; 3];
    let mut band_n = [0usize; 3];
    for (rank, &i) in idx.iter().enumerate() {
        let band = (rank * 3 / n).min(2);
        band_ab[band][0] += px[i].lab[1];
        band_ab[band][1] += px[i].lab[2];
        band_n[band] += 1;
    }
    for b in 0..3 {
        let k = band_n[b].max(1) as f64;
        band_ab[b] = [band_ab[b][0] / k, band_ab[b][1] / k];
    }
    let mut sector_c = [0.0; SECTORS];
    let mut sector_n = [0usize; SECTORS];
    let (mut total_c, mut hi, mut lo) = (0.0, 0usize, 0usize);
    for p in px {
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        total_c += c;
        let h = b.atan2(a).to_degrees();
        let s = (((h + 30.0).rem_euclid(360.0)) / 60.0) as usize % SECTORS;
        sector_c[s] += c;
        sector_n[s] += 1;
        hi += usize::from(p.hi);
        lo += usize::from(p.lo);
    }
    let mut sector_share = [0.0; SECTORS];
    for s in 0..SECTORS {
        sector_share[s] = sector_n[s] as f64 / n as f64;
        sector_c[s] = if sector_n[s] > 0 {
            sector_c[s] / sector_n[s] as f64
        } else {
            0.0
        };
    }
    Stats {
        shape,
        median,
        band_ab,
        sector_chroma: sector_c,
        sector_share,
        mean_chroma: total_c / n as f64,
        hi_share: hi as f64 / n as f64,
        lo_share: lo as f64 / n as f64,
    }
}

/// The objective's terms, unweighted (each is 0 for a perfect match; an error of one "scale" counts about 1).
#[derive(Clone, Copy, Debug, Default)]
struct Terms {
    shape: f64,
    level: f64,
    ab: f64,
    sector: f64,
    chroma: f64,
    reg: f64,
    clip: f64,
}

impl Terms {
    fn total(&self) -> f64 {
        W_SHAPE * self.shape
            + W_LEVEL * self.level
            + W_AB * self.ab
            + W_SECTOR * self.sector
            + W_CHROMA * self.chroma
            + W_REG * self.reg
            + W_CLIP * self.clip
    }
    fn to_json(self) -> Value {
        json!({
            "shape": self.shape, "level": self.level, "ab_bands": self.ab, "chroma_sectors": self.sector,
            "mean_chroma": self.chroma, "regulariser": self.reg, "clip": self.clip,
            "weighted_total": self.total(),
            "weights": { "shape": W_SHAPE, "level": W_LEVEL, "ab_bands": W_AB, "chroma_sectors": W_SECTOR,
                "mean_chroma": W_CHROMA, "regulariser": W_REG, "clip": W_CLIP },
        })
    }
}

fn terms(out: &Stats, reference: &Stats, x: &[f64; N_PARAMS]) -> Terms {
    let shape = out
        .shape
        .iter()
        .zip(&reference.shape)
        .map(|(a, b)| ((a - b) / L_SCALE).powi(2))
        .sum::<f64>()
        / 19.0;
    let level = ((out.median - reference.median) / L_SCALE).powi(2);
    let mut ab = 0.0;
    for b in 0..3 {
        let da = (out.band_ab[b][0] - reference.band_ab[b][0]) / AB_SCALE;
        let db = (out.band_ab[b][1] - reference.band_ab[b][1]) / AB_SCALE;
        ab += 0.5 * (da * da + db * db) / 3.0;
    }
    let (mut sum_w, mut sum) = (0.0, 0.0);
    for s in 0..SECTORS {
        let w = (out.sector_share[s] * reference.sector_share[s]).sqrt();
        sum_w += w;
        sum += w * ((out.sector_chroma[s] - reference.sector_chroma[s]) / CHROMA_SCALE).powi(2);
    }
    let sector = if sum_w > 0.0 { sum / sum_w } else { 0.0 };
    let chroma = ((out.mean_chroma - reference.mean_chroma) / CHROMA_SCALE).powi(2);
    let reg = x.iter().map(|v| v * v).sum::<f64>() / N_PARAMS as f64;
    let over = |o: f64, r: f64| ((o - r - CLIP_SLACK).max(0.0) / CLIP_SCALE).powi(2);
    let clip = over(out.hi_share, reference.hi_share) + over(out.lo_share, reference.lo_share);
    Terms {
        shape,
        level,
        ab,
        sector,
        chroma,
        reg,
        clip,
    }
}

// ---------------------------------------------------------------------------------------------
// Samples, evaluation, optimiser
// ---------------------------------------------------------------------------------------------

/// A fixed stratified sample of a node picture: one pixel per cell of a grid, jittered by a seeded generator.
fn stratified_sample(img: &Img, grid: (usize, usize), seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(grid.0 * grid.1);
    for gy in 0..grid.1 {
        for gx in 0..grid.0 {
            let fx = (gx as f64 + rng.unit()) / grid.0 as f64;
            let fy = (gy as f64 + rng.unit()) / grid.1 as f64;
            let x = ((fx * img.w as f64) as usize).min(img.w - 1);
            let y = ((fy * img.h as f64) as usize).min(img.h - 1);
            out.push(img.at(x, y));
        }
    }
    out
}

/// Evaluates candidates on a fixed sample.
struct Evaluator<'a> {
    output: &'a ChainParamsOut,
    lin: Vec<[f64; 3]>,
    reference: Stats,
    conv: LabConv,
    with_hue: bool,
    strength: f64,
    evals: usize,
    buf: Vec<Px>,
    eval_time: Duration,
}

type ChainParamsOut = ColorChain;

impl Evaluator<'_> {
    fn stats(&mut self, x: &[f64; N_PARAMS]) -> Stats {
        let real = to_real(x, self.strength);
        match Candidate::new(&real, self.with_hue) {
            Ok(cand) => {
                let (output, conv) = (self.output, &self.conv);
                self.buf
                    .par_chunks_mut(256)
                    .zip(self.lin.par_chunks(256))
                    .for_each(|(o, i)| {
                        for (d, s) in o.iter_mut().zip(i) {
                            *d = conv.px(render_pixel(&cand, output, *s));
                        }
                    });
                stats_of(&self.buf)
            }
            // A candidate the chain refuses is worse than anything.
            Err(_) => {
                let mut s = stats_of(&self.buf);
                s.median = 1e6;
                s
            }
        }
    }

    fn objective(&mut self, x: &[f64; N_PARAMS]) -> (f64, Terms) {
        let t0 = Instant::now();
        let s = self.stats(x);
        let t = terms(&s, &self.reference, x);
        self.evals += 1;
        self.eval_time += t0.elapsed();
        (t.total(), t)
    }
}

/// Golden-section search of one coordinate over `[lo, hi]` (clamped to -1..1): a fixed number of iterations.
fn golden(
    ev: &mut Evaluator<'_>,
    x: &mut [f64; N_PARAMS],
    i: usize,
    half: f64,
    iters: usize,
    current: f64,
) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let x0 = x[i];
    let (mut a, mut b) = ((x0 - half).max(-1.0), (x0 + half).min(1.0));
    let eval_at = |ev: &mut Evaluator<'_>, v: f64| {
        let mut t = *x;
        t[i] = v;
        ev.objective(&t).0
    };
    let (mut c, mut d) = (b - INV_PHI * (b - a), a + INV_PHI * (b - a));
    let (mut fc, mut fd) = (eval_at(ev, c), eval_at(ev, d));
    let mut best = (x0, current);
    for (v, f) in [(c, fc), (d, fd)] {
        if f < best.1 {
            best = (v, f);
        }
    }
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = eval_at(ev, c);
            if fc < best.1 {
                best = (c, fc);
            }
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = eval_at(ev, d);
            if fd < best.1 {
                best = (d, fd);
            }
        }
    }
    x[i] = best.0;
    best.1
}

struct FitResult {
    x: [f64; N_PARAMS],
    before: Terms,
    after: Terms,
    evals: usize,
    seconds: f64,
    per_eval_ms: f64,
    stage_log: Vec<Value>,
}

fn fit(ev: &mut Evaluator<'_>) -> FitResult {
    let t0 = Instant::now();
    let strength = ev.strength;
    // The fit itself runs at strength 1 (the deltas are scaled afterwards by `strength`, see `to_real`).
    ev.strength = 1.0;
    let mut x = [0.0f64; N_PARAMS];
    let (f0, before) = ev.objective(&x);
    let mut f = f0;
    let mut log = Vec::new();
    for (name, range, frac, iters) in STAGES.iter().cloned() {
        let start = f;
        for i in range {
            if ev.with_hue || !(HSAT_RANGE.contains(&i) || HHUE_RANGE.contains(&i)) {
                f = golden(ev, &mut x, i, frac, iters, f);
            }
        }
        log.push(json!({ "stage": name, "half_range": frac, "iterations": iters, "objective_before": start, "objective_after": f }));
    }
    let seconds = t0.elapsed().as_secs_f64();
    let (_, after) = ev.objective(&x);
    ev.strength = strength;
    FitResult {
        x,
        before,
        after,
        evals: ev.evals,
        seconds,
        per_eval_ms: ev.eval_time.as_secs_f64() * 1000.0 / ev.evals.max(1) as f64,
        stage_log: log,
    }
}

// ---------------------------------------------------------------------------------------------
// Rendering, Reinhard, sheets
// ---------------------------------------------------------------------------------------------

/// Renders a node picture through a candidate and the Output: linear sRGB (clamped).
fn render_image(node: &Img, cand: &Candidate, output: &ColorChain) -> Img {
    let mut px = vec![0f32; node.px.len()];
    px.par_chunks_mut(3 * node.w)
        .zip(node.px.par_chunks(3 * node.w))
        .for_each(|(o, i)| {
            for (d, s) in o.chunks_exact_mut(3).zip(i.chunks_exact(3)) {
                let v = render_pixel(
                    cand,
                    output,
                    [f64::from(s[0]), f64::from(s[1]), f64::from(s[2])],
                );
                for c in 0..3 {
                    d[c] = if v[c].is_nan() {
                        0.0
                    } else {
                        v[c].clamp(0.0, 1.0) as f32
                    };
                }
            }
        });
    Img {
        w: node.w,
        h: node.h,
        px,
    }
}

/// Reinhard et al. 2001, "Color transfer between images": mean and standard deviation transfer in l-alpha-beta,
/// applied to `source` with the statistics of `target`. A pixel map on the default render; a baseline only.
fn reinhard(source: &Img, target: &Img) -> Img {
    const RGB2LMS: [[f64; 3]; 3] = [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ];
    let s3 = 3f64.sqrt();
    let s6 = 6f64.sqrt();
    let s2 = 2f64.sqrt();
    let lms2lab = [
        [1.0 / s3, 1.0 / s3, 1.0 / s3],
        [1.0 / s6, 1.0 / s6, -2.0 / s6],
        [1.0 / s2, -1.0 / s2, 0.0],
    ];
    let lab2lms = [
        [s3 / 3.0, s6 / 6.0, s2 / 2.0],
        [s3 / 3.0, s6 / 6.0, -s2 / 2.0],
        [s3 / 3.0, -2.0 * s6 / 6.0, 0.0],
    ];
    let lms2rgb = linalg::inv(&RGB2LMS).unwrap_or(linalg::IDENTITY);
    let to_lab = |p: [f64; 3]| {
        let lms = linalg::mul_vec(&RGB2LMS, p).map(|v| v.max(1e-6).log10());
        linalg::mul_vec(&lms2lab, lms)
    };
    let moments = |img: &Img| {
        let n = (img.w * img.h) as f64;
        let (mut m, mut m2) = ([0.0; 3], [0.0; 3]);
        for p in img.px.chunks_exact(3) {
            let l = to_lab([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
            for c in 0..3 {
                m[c] += l[c];
                m2[c] += l[c] * l[c];
            }
        }
        let mean = m.map(|v| v / n);
        let std = [0, 1, 2].map(|c| (m2[c] / n - mean[c] * mean[c]).max(1e-12).sqrt());
        (mean, std)
    };
    let (ms, ss) = moments(source);
    let (mt, st) = moments(target);
    let mut px = vec![0f32; source.px.len()];
    px.par_chunks_mut(3 * source.w)
        .zip(source.px.par_chunks(3 * source.w))
        .for_each(|(o, i)| {
            for (d, s) in o.chunks_exact_mut(3).zip(i.chunks_exact(3)) {
                let mut l = to_lab([f64::from(s[0]), f64::from(s[1]), f64::from(s[2])]);
                for c in 0..3 {
                    l[c] = (l[c] - ms[c]) * st[c] / ss[c] + mt[c];
                }
                let lms = linalg::mul_vec(&lab2lms, l).map(|v| 10f64.powf(v));
                let rgb = linalg::mul_vec(&lms2rgb, lms);
                for c in 0..3 {
                    d[c] = rgb[c].clamp(0.0, 1.0) as f32;
                }
            }
        });
    Img {
        w: source.w,
        h: source.h,
        px,
    }
}

/// The 5 by 7 glyphs the sheets need (rows top to bottom, 5 bits each, the leftmost column is bit 4).
fn glyph(c: char) -> [u8; 7] {
    match c {
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'e' => [0, 0, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        'f' => [
            0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000,
        ],
        'r' => [0, 0, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        'n' => [0, 0, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'c' => [0, 0, 0b01110, 0b10001, 0b10000, 0b10001, 0b01110],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'a' => [0, 0, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        'd' => [
            0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10001, 0b01111,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'l' => [
            0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'o' => [0, 0, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'w' => [0, 0, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
        'm' => [0, 0, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101],
        'u' => [0, 0, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        't' => [
            0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110,
        ],
        'i' => [0b00100, 0, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'h' => [
            0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        'b' => [
            0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b11110,
        ],
        's' => [0, 0, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        _ => [0; 7],
    }
}

/// A canvas of 8-bit RGB.
struct Canvas {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

impl Canvas {
    fn new(w: usize, h: usize, grey: u8) -> Self {
        Self {
            w,
            h,
            data: vec![grey; 3 * w * h],
        }
    }
    /// Draws `img` scaled to fit inside the box (x, y, bw, bh), centred, nearest-neighbour on an already small picture.
    fn put_fit(&mut self, img: &Rgb8, x: usize, y: usize, bw: usize, bh: usize) {
        let s = (bw as f64 / img.w as f64).min(bh as f64 / img.h as f64);
        let (tw, th) = (
            ((img.w as f64 * s).round() as usize).max(1),
            ((img.h as f64 * s).round() as usize).max(1),
        );
        let small = if (tw, th) == (img.w, img.h) {
            None
        } else {
            let lin = rgb8_to_linear(img);
            Some(linear_to_rgb8(&resize_area(&lin, tw, th)))
        };
        let src = small.as_ref().unwrap_or(img);
        let (ox, oy) = (x + (bw - tw) / 2, y + (bh - th) / 2);
        for yy in 0..th {
            let d = 3 * ((oy + yy) * self.w + ox);
            self.data[d..d + 3 * tw].copy_from_slice(&src.data[3 * yy * tw..3 * (yy + 1) * tw]);
        }
    }
    fn text(&mut self, s: &str, x: usize, y: usize, scale: usize, grey: u8) {
        let mut cx = x;
        for ch in s.chars() {
            let g = glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits >> (4 - col) & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let (px, py) = (cx + col * scale + dx, y + row * scale + dy);
                                if px < self.w && py < self.h {
                                    let d = 3 * (py * self.w + px);
                                    self.data[d..d + 3].fill(grey);
                                }
                            }
                        }
                    }
                }
            }
            cx += 6 * scale;
        }
    }
    fn save_jpeg(&self, path: &Path) -> R<()> {
        let enc = jpeg_encoder::Encoder::new_file(path, JPEG_QUALITY)?;
        enc.encode(
            &self.data,
            self.w as u16,
            self.h as u16,
            jpeg_encoder::ColorType::Rgb,
        )?;
        Ok(())
    }
}

fn save_rgb8_jpeg(img: &Rgb8, path: &Path) -> R<()> {
    let enc = jpeg_encoder::Encoder::new_file(path, JPEG_QUALITY)?;
    enc.encode(
        &img.data,
        img.w as u16,
        img.h as u16,
        jpeg_encoder::ColorType::Rgb,
    )?;
    Ok(())
}

fn write_sheet(path: &Path, reference: &Rgb8, panels: [&Rgb8; 3], photo_aspect: f64) -> R<()> {
    const GAP: usize = 8;
    const BAND: usize = 30;
    let panel_h = (PANEL_W_PX as f64 / photo_aspect).round() as usize;
    let (w, h) = (4 * PANEL_W_PX + 5 * GAP, panel_h + BAND + 2 * GAP);
    let mut c = Canvas::new(w, h, 18);
    let labels = [
        "Reference",
        "Yellowmoon default",
        "Reference Grade",
        "Reinhard (baseline)",
    ];
    let all = [reference, panels[0], panels[1], panels[2]];
    for (i, (img, label)) in all.iter().zip(labels).enumerate() {
        let x = GAP + i * (PANEL_W_PX + GAP);
        c.text(label, x, GAP + 4, 3, 235);
        c.put_fit(img, x, GAP + BAND, PANEL_W_PX, panel_h);
    }
    c.save_jpeg(path)
}

// ---------------------------------------------------------------------------------------------
// Settings file and reports
// ---------------------------------------------------------------------------------------------

fn curve_json(knots: &[(f64, f64, f64, f64)]) -> Value {
    let mut pts = vec![json!([0.0, 0.0])];
    pts.extend(knots.iter().map(|k| json!([k.1, k.3])));
    pts.push(json!([1.0, 1.0]));
    Value::Array(pts)
}

/// The non-hue part as a settings file: a corrector (id 2) after Main with the saturation and the four curves.
fn settings_json(real: &Real) -> Value {
    json!({
        "nodes": {
            "items": [{
                "id": 2, "kind": "corrector", "name": "Reference Grade", "pos": [1, 0],
                "grade": {
                    "saturation": real.saturation_pct,
                    "curve": {
                        "luma": curve_json(&real.luma),
                        "red": curve_json(&real.rgb[0]),
                        "green": curve_json(&real.rgb[1]),
                        "blue": curve_json(&real.rgb[2]),
                    }
                }
            }],
            "links": [
                { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
                { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
            ]
        }
    })
}

fn knots_json(k: &[(f64, f64, f64, f64)]) -> Value {
    Value::Array(
        k.iter()
            .map(|(s, xi, ys, yc)| json!({ "stop": s, "input_logc4": xi, "output_stops": ys, "output_logc4": yc }))
            .collect(),
    )
}

fn real_json(real: &Real, x: &[f64; N_PARAMS], strength: f64) -> Value {
    let params: Vec<Value> = (0..N_PARAMS)
        .map(|i| json!({ "name": param_name(i), "normalised": x[i], "bound": bound(i), "moved": x[i].abs() >= 0.02 }))
        .collect();
    json!({
        "strength": strength,
        "luma_curve": knots_json(&real.luma),
        "red_curve": knots_json(&real.rgb[0]),
        "green_curve": knots_json(&real.rgb[1]),
        "blue_curve": knots_json(&real.rgb[2]),
        "saturation_pct": real.saturation_pct,
        "hue_vs_sat": HUE_KNOTS_DEG.iter().zip(real.hue_vs_sat).map(|(h, v)| json!({"hue_deg": h, "multiplier": v})).collect::<Vec<_>>(),
        "hue_vs_hue": HUE_KNOTS_DEG.iter().zip(real.hue_vs_hue_deg).map(|(h, v)| json!({"hue_deg": h, "shift_deg": v})).collect::<Vec<_>>(),
        "parameters": params,
    })
}

// ---------------------------------------------------------------------------------------------
// One pair
// ---------------------------------------------------------------------------------------------

struct Reference {
    stem: String,
    /// Stripped, 8-bit sRGB, for display.
    show8: Rgb8,
    /// Linear sRGB, fit size.
    fit: Img,
    bars: [usize; 4],
}

fn load_reference(path: &Path) -> R<Reference> {
    let raw = load_reference_file(path)?;
    let (stripped, bars) = strip_bars(&raw);
    let lin = rgb8_to_linear(&stripped);
    let (fw, fh) = fit_size(lin.w, lin.h, FIT_LONG_EDGE_PX);
    let fit = resize_area(&lin, fw, fh);
    let (sw, sh) = fit_size(lin.w, lin.h, PANEL_W_PX);
    let show8 = linear_to_rgb8(&resize_area(&lin, sw, sh));
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Reference {
        stem,
        show8,
        fit,
        bars,
    })
}

fn picture_stats(conv: &LabConv, img: &Img) -> Stats {
    let px: Vec<Px> = (0..img.w * img.h)
        .map(|i| {
            conv.px(img.px[3 * i..3 * i + 3]
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or([0.0; 3]))
        })
        .collect();
    stats_of(&px)
}

/// One photo prepared for fitting against many references.
struct PhotoFit<'a> {
    photo: &'a PhotoNode,
    output: ColorChain,
    fit_img: Img,
    lin_sample: Vec<[f64; 3]>,
    default_show: Img,
}

fn prepare_photo(photo: &PhotoNode) -> R<PhotoFit<'_>> {
    let output = make_output_chain(&photo.output)?;
    let (fw, fh) = fit_size(photo.show.w, photo.show.h, FIT_LONG_EDGE_PX);
    let fit_img = resize_area(&photo.show, fw, fh);
    let lin_sample = stratified_sample(&fit_img, SAMPLE_GRID, SEED);
    let ident = Candidate::new(&to_real(&[0.0; N_PARAMS], 1.0), false)?;
    let default_show = render_image(&photo.show, &ident, &output);
    Ok(PhotoFit {
        photo,
        output,
        fit_img,
        lin_sample,
        default_show,
    })
}

struct PairOutcome {
    fitted_show: Img,
    fit: FitResult,
    real: Real,
    x_scaled: [f64; N_PARAMS],
    wall_total: f64,
}

fn run_fit(
    pf: &PhotoFit<'_>,
    reference_fit: &Img,
    strength: f64,
    with_hue: bool,
) -> R<(FitResult, Stats)> {
    let conv = LabConv::new();
    let ref_stats = picture_stats(&conv, reference_fit);
    let mut ev = Evaluator {
        output: &pf.output,
        lin: pf.lin_sample.clone(),
        reference: ref_stats.clone(),
        conv,
        with_hue,
        strength,
        evals: 0,
        buf: vec![Px::default(); pf.lin_sample.len()],
        eval_time: Duration::ZERO,
    };
    Ok((fit(&mut ev), ref_stats))
}

fn run_pair(
    pf: &PhotoFit<'_>,
    reference: &Reference,
    strength: f64,
    with_hue: bool,
) -> R<PairOutcome> {
    let t0 = Instant::now();
    let (fit_res, _) = run_fit(pf, &reference.fit, strength, with_hue)?;
    let mut x_scaled = fit_res.x;
    for v in &mut x_scaled {
        *v *= strength;
    }
    let real = to_real(&fit_res.x, strength);
    let cand = Candidate::new(&real, with_hue)?;
    let fitted_show = render_image(&pf.photo.show, &cand, &pf.output);
    Ok(PairOutcome {
        fitted_show,
        fit: fit_res,
        real,
        x_scaled,
        wall_total: t0.elapsed().as_secs_f64(),
    })
}

fn write_pair(
    dir: &Path,
    photo_name: &str,
    pf: &PhotoFit<'_>,
    reference: &Reference,
    o: &PairOutcome,
    strength: f64,
    with_hue: bool,
) -> R<()> {
    fs::create_dir_all(dir)?;
    let default8 = linear_to_rgb8(&pf.default_show);
    let fitted8 = linear_to_rgb8(&o.fitted_show);
    let ref_show_lin = rgb8_to_linear(&reference.show8);
    let reinhard8 = linear_to_rgb8(&reinhard(&pf.default_show, &ref_show_lin));
    let aspect = pf.default_show.w as f64 / pf.default_show.h as f64;
    write_sheet(
        &dir.join("sheet.jpg"),
        &reference.show8,
        [&default8, &fitted8, &reinhard8],
        aspect,
    )?;
    save_rgb8_jpeg(&fitted8, &dir.join("fitted.jpg"))?;
    save_rgb8_jpeg(&default8, &dir.join("default.jpg"))?;
    let conv = LabConv::new();
    let fitted_fit = resize_area(&o.fitted_show, pf.fit_img.w, pf.fit_img.h);
    let ref_stats = picture_stats(&conv, &reference.fit);
    let default_fit = resize_area(&pf.default_show, pf.fit_img.w, pf.fit_img.h);
    let def_stats = picture_stats(&conv, &default_fit);
    let fit_stats = picture_stats(&conv, &fitted_fit);
    let mut reg_zero = [0.0; N_PARAMS];
    let before_full = terms(&def_stats, &ref_stats, &reg_zero);
    reg_zero = o.x_scaled.map(|v| v / strength.max(1e-9));
    let after_full = terms(&fit_stats, &ref_stats, &reg_zero);
    let report = json!({
        "photo": photo_name,
        "reference": reference.stem,
        "strength": strength,
        "hue_step": with_hue,
        "reference_bars_stripped_top_bottom_left_right": reference.bars,
        "fitted_values": real_json(&o.real, &o.fit.x, strength),
        "objective_on_fit_sample": { "before": o.fit.before.to_json(), "after": o.fit.after.to_json() },
        "objective_on_full_fit_pictures_512px": { "default": before_full.to_json(), "fitted_at_strength": after_full.to_json() },
        "picture_stats": {
            "median_L": { "reference": ref_stats.median, "default": def_stats.median, "fitted": fit_stats.median },
            "mean_chroma": { "reference": ref_stats.mean_chroma, "default": def_stats.mean_chroma, "fitted": fit_stats.mean_chroma },
            "clipped_white_share": { "reference": ref_stats.hi_share, "default": def_stats.hi_share, "fitted": fit_stats.hi_share },
            "crushed_black_share": { "reference": ref_stats.lo_share, "default": def_stats.lo_share, "fitted": fit_stats.lo_share },
        },
        "timings": {
            "samples": pf.lin_sample.len(),
            "fit_seconds": o.fit.seconds,
            "objective_evaluations": o.fit.evals,
            "ms_per_objective_evaluation": o.fit.per_eval_ms,
            "pair_total_seconds_including_renders": o.wall_total,
            "threads": rayon::current_num_threads(),
        },
        "optimiser_stages": o.fit.stage_log,
        "output_look": pf.photo.output.look_id,
        "output_look_intensity_pct": pf.photo.output.intensity_pct,
        "notes": pf.photo.notes,
        "settings_json_note": if with_hue { "settings.json holds the part that is real corrector settings; the hue curves have no settings field yet, so `ym render` with it gives the picture without the hue step." } else { "no hue step: `ym render --settings settings.json` should reproduce the fitted picture (up to output sharpening)." },
    });
    fs::write(
        dir.join("params.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    fs::write(
        dir.join("settings.json"),
        serde_json::to_string_pretty(&settings_json(&o.real))?,
    )?;
    Ok(())
}

fn write_grid(path: &Path, pf: &PhotoFit<'_>, refs: &[(&Reference, &PairOutcome)]) -> R<()> {
    const GAP: usize = 6;
    const LABEL: usize = 18;
    let cell_h = (GRID_CELL_W_PX as f64 * pf.default_show.h as f64 / pf.default_show.w as f64)
        .round() as usize;
    let thumb_h = (GRID_CELL_W_PX * 9 / 16).max(1);
    let col_h = LABEL + thumb_h + GAP + cell_h;
    let cols = refs.len() + 1;
    let mut c = Canvas::new(GAP + cols * (GRID_CELL_W_PX + GAP), col_h + 2 * GAP, 18);
    let x0 = GAP;
    c.text("0", x0, GAP + 2, 2, 235);
    c.put_fit(
        &linear_to_rgb8(&pf.default_show),
        x0,
        GAP + LABEL + thumb_h + GAP,
        GRID_CELL_W_PX,
        cell_h,
    );
    for (i, (r, o)) in refs.iter().enumerate() {
        let x = GAP + (i + 1) * (GRID_CELL_W_PX + GAP);
        c.text(&format!("{}", i + 1), x, GAP + 2, 2, 235);
        c.put_fit(&r.show8, x, GAP + LABEL, GRID_CELL_W_PX, thumb_h);
        c.put_fit(
            &linear_to_rgb8(&o.fitted_show),
            x,
            GAP + LABEL + thumb_h + GAP,
            GRID_CELL_W_PX,
            cell_h,
        );
    }
    c.save_jpeg(path)
}

// ---------------------------------------------------------------------------------------------
// E1 self-test
// ---------------------------------------------------------------------------------------------

fn selftest(
    pf: &PhotoFit<'_>,
    out: &Path,
    photo_name: &str,
    trials: usize,
    moderate: f64,
) -> R<Value> {
    let conv = LabConv::new();
    let mut rng = Rng(SEED ^ 0xE1);
    let n_px = pf.fit_img.w * pf.fit_img.h;
    let mut pooled: Vec<f64> = Vec::new();
    let mut pooled_default: Vec<f64> = Vec::new();
    let mut trial_json = Vec::new();
    for t in 0..trials {
        let mut x_true = [0.0; N_PARAMS];
        for v in &mut x_true {
            *v = moderate * (2.0 * rng.unit() - 1.0);
        }
        let truth_real = to_real(&x_true, 1.0);
        let truth_cand = Candidate::new(&truth_real, true)?;
        let truth = render_image(&pf.fit_img, &truth_cand, &pf.output);
        let t0 = Instant::now();
        let (fit_res, _) = run_fit(pf, &truth, 1.0, true)?;
        let fitted_real = to_real(&fit_res.x, 1.0);
        let fitted_cand = Candidate::new(&fitted_real, true)?;
        let fitted = render_image(&pf.fit_img, &fitted_cand, &pf.output);
        let ident = Candidate::new(&to_real(&[0.0; N_PARAMS], 1.0), false)?;
        let ungraded = render_image(&pf.fit_img, &ident, &pf.output);
        let de = |a: &Img, b: &Img| -> Vec<f64> {
            (0..n_px)
                .into_par_iter()
                .map(|i| {
                    let l = |im: &Img| {
                        let p = &im.px[3 * i..3 * i + 3];
                        conv.px([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
                            .lab
                    };
                    delta_e_2000(l(a), l(b))
                })
                .collect()
        };
        let d_fit = de(&fitted, &truth);
        let d_def = de(&ungraded, &truth);
        let s = Summary::of(&d_fit).ok_or("empty")?;
        let sd = Summary::of(&d_def).ok_or("empty")?;
        pooled.extend(&d_fit);
        pooled_default.extend(&d_def);
        // The parameter error, in normalised units, over the parameters that matter (all of them).
        let rms_param = (x_true
            .iter()
            .zip(fit_res.x)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / N_PARAMS as f64)
            .sqrt();
        let rms_true = (x_true.iter().map(|a| a * a).sum::<f64>() / N_PARAMS as f64).sqrt();
        trial_json.push(json!({
            "trial": t, "fitted_vs_truth_de2000": { "median": s.median, "p95": s.p95, "mean": s.mean },
            "ungraded_vs_truth_de2000": { "median": sd.median, "p95": sd.p95 },
            "rms_normalised_parameter_error": rms_param, "rms_normalised_true_parameters": rms_true,
            "objective_before": fit_res.before.total(), "objective_after": fit_res.after.total(),
            "fit_seconds": fit_res.seconds, "seconds_with_renders": t0.elapsed().as_secs_f64(),
        }));
        if t < 3 {
            let dir = out.join(format!("e1-trial-{t}"));
            fs::create_dir_all(&dir)?;
            let to8 = |i: &Img| linear_to_rgb8(i);
            let mut c = Canvas::new(3 * 520 + 4 * 8, 360 + 8 * 2 + 30, 18);
            for (k, (img, label)) in [
                (&ungraded, "default"),
                (&truth, "Reference"),
                (&fitted, "Reference Grade"),
            ]
            .iter()
            .enumerate()
            {
                let x = 8 + k * 528;
                c.text(label, x, 12, 3, 235);
                c.put_fit(&to8(img), x, 46, 520, 360);
            }
            c.save_jpeg(&dir.join("sheet.jpg"))?;
        }
    }
    let sp = Summary::of(&pooled).ok_or("empty")?;
    let sd = Summary::of(&pooled_default).ok_or("empty")?;
    let med_of_trial_medians = {
        let mut v: Vec<f64> = trial_json
            .iter()
            .filter_map(|t| t["fitted_vs_truth_de2000"]["median"].as_f64())
            .collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Ok(json!({
        "photo": photo_name,
        "trials": trials,
        "random_parameter_amplitude_fraction_of_bound": moderate,
        "pooled_over_all_pixels_of_all_trials": {
            "fitted_vs_truth_de2000": { "median": sp.median, "p95": sp.p95, "mean": sp.mean },
            "ungraded_vs_truth_de2000": { "median": sd.median, "p95": sd.p95, "mean": sd.mean },
        },
        "median_of_trial_medians": med_of_trial_medians,
        "per_trial": trial_json,
    }))
}

// ---------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Args {
    photos: Vec<PathBuf>,
    references: Vec<PathBuf>,
    out: PathBuf,
    strength: f64,
    no_hue: bool,
    selftest: bool,
    trials: usize,
    amplitude: f64,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        strength: 1.0,
        trials: 10,
        amplitude: 0.5,
        ..Args::default()
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--photo" | "--photos" => a.photos.push(PathBuf::from(val()?)),
            "--reference" | "--references" => a.references.push(PathBuf::from(val()?)),
            "--out" => a.out = PathBuf::from(val()?),
            "--strength" => a.strength = val()?.parse()?,
            "--trials" => a.trials = val()?.parse()?,
            "--amplitude" => a.amplitude = val()?.parse()?,
            "--no-hue" => a.no_hue = true,
            "--selftest" => a.selftest = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if a.out.as_os_str().is_empty() {
        return Err("--out <folder> is required".into());
    }
    Ok(a)
}

/// A file, or the files with these extensions in a folder, sorted by name.
fn expand(paths: &[PathBuf], exts: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|f| {
                    f.extension().is_some_and(|e| {
                        exts.contains(&e.to_string_lossy().to_lowercase().as_str())
                    })
                })
                .collect();
            files.sort();
            out.extend(files);
        } else {
            out.push(p.clone());
        }
    }
    out
}

fn main() -> R<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;
    let photos = expand(&args.photos, &["arw"]);
    if photos.is_empty() {
        return Err("no photos".into());
    }
    let core = open_core(&args.out)?;
    let cache = args.out.join("cache");
    let mut summary = Vec::new();
    for photo_path in &photos {
        let name = photo_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("== {name}");
        let t = Instant::now();
        let photo = load_photo(&core, photo_path, &cache)?;
        eprintln!(
            "   node picture {}x{} in {:.1} s {:?}",
            photo.show.w,
            photo.show.h,
            t.elapsed().as_secs_f64(),
            photo.notes
        );
        let pf = prepare_photo(&photo)?;
        let photo_dir = args.out.join(&name);
        fs::create_dir_all(&photo_dir)?;
        if args.selftest {
            let r = selftest(&pf, &photo_dir, &name, args.trials, args.amplitude)?;
            fs::write(photo_dir.join("e1.json"), serde_json::to_string_pretty(&r)?)?;
            eprintln!(
                "   E1 pooled: {}",
                r["pooled_over_all_pixels_of_all_trials"]
            );
            summary.push(r);
            continue;
        }
        let refs = expand(&args.references, &["jpg", "jpeg", "png"]);
        if refs.is_empty() {
            return Err("no references".into());
        }
        let mut loaded = Vec::new();
        for r in &refs {
            loaded.push(load_reference(r)?);
        }
        let mut outcomes = Vec::new();
        let mut index: BTreeMap<String, Value> = BTreeMap::new();
        for r in &loaded {
            let o = run_pair(&pf, r, args.strength, !args.no_hue)?;
            eprintln!(
                "   {}: fit {:.2} s, {} evals, {:.2} ms/eval, objective {:.3} -> {:.3}",
                r.stem,
                o.fit.seconds,
                o.fit.evals,
                o.fit.per_eval_ms,
                o.fit.before.total(),
                o.fit.after.total()
            );
            write_pair(
                &photo_dir.join(&r.stem),
                &name,
                &pf,
                r,
                &o,
                args.strength,
                !args.no_hue,
            )?;
            index.insert(
                r.stem.clone(),
                json!({ "fit_seconds": o.fit.seconds, "per_eval_ms": o.fit.per_eval_ms, "before": o.fit.before.total(), "after": o.fit.after.total() }),
            );
            outcomes.push(o);
        }
        let pairs: Vec<(&Reference, &PairOutcome)> = loaded.iter().zip(&outcomes).collect();
        write_grid(&photo_dir.join("grid.jpg"), &pf, &pairs)?;
        let legend: Vec<Value> = loaded
            .iter()
            .enumerate()
            .map(|(i, r)| json!({ "column": i + 1, "reference": r.stem }))
            .collect();
        fs::write(
            photo_dir.join("grid.json"),
            serde_json::to_string_pretty(
                &json!({ "column 0": "Yellowmoon default", "columns": legend, "fits": index }),
            )?,
        )?;
        summary.push(json!({ "photo": name, "fits": index }));
    }
    fs::write(
        args.out.join(if args.selftest {
            "e1-summary.json"
        } else {
            "summary.json"
        }),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(())
}
