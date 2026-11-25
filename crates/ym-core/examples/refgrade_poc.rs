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

use std::collections::{BTreeMap, BTreeSet};
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
use ym_color::grade::{E1, E2, GradeParams, logc4_at_stops};
use ym_color::look::{HuePoint, ParametricLook, ParametricLookSpec, SCHEMA_ID};
use ym_color::tone::ToneParams;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{FullRequest, FullTarget, PhotoSource};
use ym_core::session::OpenOptions;
use ym_gpu::queue::CancelToken;

#[path = "refgrade_poc/eval.rs"]
mod eval;

type R<T> = Result<T, Box<dyn Error>>;

/// What V3 needs from outside: the label folder and the skin network.
struct Ctx {
    labels: Option<PathBuf>,
    seg: Option<ym_ml::mediapipe::selfie::SelfieSegmenter>,
    network: Option<Network>,
    network5: Option<Network>,
    network6: Option<Network>,
    network7: Option<Network>,
    network8: Option<Network>,
    network9: Option<Network>,
    network10: Option<Network>,
    network11: Option<Network>,
    /// The round-12 principles switched back to their v11 behaviour (`--ablate`).
    ablate: Ablate,
}

/// Which of the round-12 principles P1 to P5 (`refgrade-notes/17-model-v12.md`) are switched back to their v11 behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Ablate {
    /// Index 0 is P1, index 4 is P5.
    off: [bool; 5],
}

impl Ablate {
    /// Parses `p1,p3`: a comma-separated list from `p1` to `p5`, each at most once.
    fn parse(list: &str) -> R<Ablate> {
        let mut a = Ablate::default();
        for part in list.split(',') {
            let part = part.trim().to_lowercase();
            let k = match part.as_str() {
                "p1" => 0,
                "p2" => 1,
                "p3" => 2,
                "p4" => 3,
                "p5" => 4,
                other => {
                    return Err(format!(
                        "--ablate takes p1 to p5 separated by commas, not {other:?}"
                    )
                    .into());
                }
            };
            if a.off[k] {
                return Err(format!("--ablate lists {part} twice").into());
            }
            a.off[k] = true;
        }
        Ok(a)
    }

    /// Whether principle `n` (1 to 5) stays on.
    fn on(&self, n: usize) -> bool {
        !self.off[n - 1]
    }

    /// Whether nothing is switched off.
    fn is_none(&self) -> bool {
        self.off.iter().all(|&o| !o)
    }

    /// The label of a run: `base` as it is, or `base-no-p2` / `base-no-p1-p2-p3-p4-p5`.
    fn label(&self, base: &str) -> String {
        let mut s = base.to_string();
        if !self.is_none() {
            s.push_str("-no");
            for (k, _) in self.off.iter().enumerate().filter(|&(_, &o)| o) {
                s.push_str(&format!("-p{}", k + 1));
            }
        }
        s
    }
}

static CTX: std::sync::OnceLock<Ctx> = std::sync::OnceLock::new();

fn ctx() -> &'static Ctx {
    CTX.get_or_init(|| Ctx {
        labels: None,
        seg: None,
        network: None,
        network5: None,
        network6: None,
        network7: None,
        network8: None,
        network9: None,
        network10: None,
        network11: None,
        ablate: Ablate::default(),
    })
}

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

/// Number of global parameters: 8 luma, 9 RGB, 1 saturation, 6 hue-vs-sat, 6 hue-vs-hue.
const N_GLOBAL: usize = 30;
/// All parameters: the global ones, then V5's local grades (3 for each of sky, vegetation, water, ground, built and skin,
/// and the subject's exposure offset).
const N_PARAMS: usize = N_GLOBAL + 19;
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
    ("joint polish", 0..N_GLOBAL, 0.25, 6),
    ("joint polish", 0..N_GLOBAL, 0.1, 5),
];

fn bound(i: usize) -> f64 {
    match i {
        0 => BOUND_LUMA_OFFSET_STOPS,
        1..=7 => BOUND_LUMA_LOG_SLOPE,
        8..=16 => BOUND_RGB_STOPS,
        17 => BOUND_SATURATION_PCT,
        18..=23 => BOUND_HUE_SAT,
        24..=29 => BOUND_HUE_DEG,
        // Local grades: exposure up to one stop, hue up to the element's cap, chroma scale in 0.5..1.6 (normalised).
        _ => match (i - N_GLOBAL) % 3 {
            _ if i == N_PARAMS - 1 => 1.0,
            1 => V5_CAP_DEG[(i - N_GLOBAL) / 3],
            _ => 1.0,
        },
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
        24..=29 => format!("hue_vs_hue[{}deg]", HUE_KNOTS_DEG[i - 24]),
        _ if i == N_PARAMS - 1 => "local.subject.exposure_stops".into(),
        _ => format!(
            "local.{}.{}",
            V5_ELEMS[(i - N_GLOBAL) / 3],
            ["exposure_stops", "hue_deg", "chroma_scale"][(i - N_GLOBAL) % 3]
        ),
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
    /// The file's stem.
    name: String,
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
            name: stem.clone(),
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
        name: stem.clone(),
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
    /// V5: the local grades, applied after the hue step through the elements' soft masks (`None`: none).
    local: Option<LocalSet>,
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
        Ok(Self {
            node,
            hue,
            local: None,
        })
    }

    /// The node's output for a pixel at its input (linear AWG4), as linear AWG4 for the Output.
    ///
    /// Order: the corrector (C9, C9b, C11, the real code), then the POC hue step on its LogC4 result, then back to
    /// linear for the Output's look (C10), the cine rendering (C12) and the encode (C13).
    /// [`Candidate::apply`] with the pixel's soft masks: the local grades (a POC-only CPU step, like the hue step) run in
    /// LogC4 after the hue step, element by element in the fixed order, each blended as `v + m * (local(v) - v)`.
    fn apply_m(&self, lin: [f64; 3], masks: Option<&[f32; 7]>) -> [f64; 3] {
        let v = self.node.apply_logc4_pixel(lin);
        let mut v = match &self.hue {
            None => v.0,
            Some(look) => look.evaluate(v.0),
        };
        if let (Some(loc), Some(m)) = (&self.local, masks) {
            for k in 0..7 {
                let mk = f64::from(m[k]);
                if mk > 0.0 && !(loc.exp[k] == 0.0 && loc.hue[k] == 0.0 && loc.chroma[k] == 1.0) {
                    let l = local_apply(v, loc.exp[k], loc.hue[k], loc.chroma[k]);
                    for c in 0..3 {
                        v[c] += mk * (l[c] - v[c]);
                    }
                }
            }
        }
        ym_color::logc::LogC4(v).to_linear().0
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

/// Display-linear sRGB of one node-input pixel (not clamped), with its soft masks for the local grades.
fn render_pixel_m(
    cand: &Candidate,
    output: &ColorChain,
    lin: [f64; 3],
    masks: Option<&[f32; 7]>,
) -> [f64; 3] {
    output.apply_to_display_linear(cand.apply_m(lin, masks))
}

// ---------------------------------------------------------------------------------------------
// Colour measurement and the objective
// ---------------------------------------------------------------------------------------------

/// One measured pixel.
#[derive(Clone, Copy, Debug, Default)]
struct Px {
    lab: [f64; 3],
    /// Oklab (a, b), for the purple tests of V11.
    ok: [f64; 2],
    hi: bool,
    lo: bool,
    /// V12: crushed black, L* under [`V12_CRUSH_L`].
    lo25: bool,
}

/// Oklab (a, b) of a linear sRGB colour (Ottosson's matrices).
fn oklab_ab(c: [f64; 3]) -> [f64; 2] {
    let l = (0.412_221_470_8 * c[0] + 0.536_332_536_3 * c[1] + 0.051_445_992_9 * c[2]).cbrt();
    let m = (0.211_903_498_2 * c[0] + 0.680_699_545_1 * c[1] + 0.107_396_956_6 * c[2]).cbrt();
    let s = (0.088_302_461_9 * c[0] + 0.281_718_837_6 * c[1] + 0.629_978_700_5 * c[2]).cbrt();
    [
        1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    ]
}

/// The hue, in degrees 0..360, of an Oklab (a, b) vector.
fn oklab_hue(ab: [f64; 2]) -> f64 {
    ab[1].atan2(ab[0]).to_degrees().rem_euclid(360.0)
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
        let lab = xyz_to_lab(xyz, self.white);
        Px {
            lab,
            lo25: lab[0] < V12_CRUSH_L,
            ok: oklab_ab(c),
            hi: c.iter().any(|&v| v >= self.hi),
            lo: c.iter().all(|&v| v <= self.lo),
        }
    }
}

/// Groups of V2 terms. A stage of the optimiser sees only the terms its parameters act on, so that a tone parameter is
/// never used to fix a colour term (which would be an arbitrary move the objective cannot see).
const MASK_TONE: u8 = 1;
const MASK_CAST: u8 = 2;
const MASK_CHROMA: u8 = 4;
const MASK_HUE: u8 = 8;
/// V3: the anchor, black floor and white ceiling, which take part in every stage.
const MASK_BASE: u8 = 16;
/// V5: the element terms (colour, then exposure and the subject).
const MASK_ELEM_COL: u8 = 32;
const MASK_ELEM_EXP: u8 = 64;
const MASK_ALL: u8 = 127;

/// Which objective the fit minimises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Objective {
    /// Round 1: the full L* quantile shape, band means of all pixels, chroma per sector weighted by share.
    V1,
    /// Round 2: what a grade does, not what the picture contains (see the `V2_*` constants).
    V2,
    /// Round 3: the hand-written, scene-aware grading model (`refgrade-notes/07-grading-model-v3.md`).
    V3,
    /// Round 5: a hand-written network sets the targets, the solver sets the settings (`refgrade-notes/09-model-v5.md`).
    V5,
    /// Round 6: network v6, a stronger look: the frame cast, the unify gate and the wall limit as network outputs.
    V6,
    /// Round 7: network v7: the cast measured off the whites and the skin, a stronger skin wall, a highlight-neutrality wall.
    V7,
    /// Round 8: network v8: the purple wall, the whites' direction, the skin drift wall, clothing as a group, the clip term
    /// against the photo's own shares.
    V8,
    /// Round 9: network v9: mean-vector hue and chroma in every wall, a dark-colours group, a memory-colour wall for sky and
    /// foliage.
    V9,
    /// Round 10: network v10: a pixel-level purple guard (the share of pixels that turn purple), `T.purple_max_share`.
    V10,
    /// Round 11: network v11: the purple band in Oklab hue, turn-based purple guards, a skin chroma wall.
    V11,
    /// Round 12: network v12: turns count wherever they end, one hue space, a content turn budget, whites by pixel, a stiffer
    /// clip term (`refgrade-notes/17-model-v12.md`).
    V12,
    /// Round 13: network v13: v12 with P1 withdrawn and a content turn budget that lets weak colours carry a cast
    /// (`refgrade-notes/19-model-v13.md`).
    V13,
}

impl Objective {
    /// Whether the objective reads a network (V5 and V6).
    fn net(self) -> bool {
        matches!(
            self,
            Objective::V5
                | Objective::V6
                | Objective::V7
                | Objective::V8
                | Objective::V9
                | Objective::V10
                | Objective::V11
                | Objective::V12
                | Objective::V13
        )
    }
}

/// V2 weights and scales. Tone: black floor, white ceiling, midtone contrast, and an anchor on the photo's own median.
const V2_W_FLOOR: f64 = 1.0;
const V2_W_CEIL: f64 = 1.0;
const V2_W_CONTRAST: f64 = 1.0;
const V2_W_ANCHOR: f64 = 4.0;
const V2_W_CAST: f64 = 1.5;
const V2_W_HUE: f64 = 1.0;
const V2_W_SECTOR_CHROMA: f64 = 1.0;
const V2_W_CHROMA_RATIO: f64 = 0.7;
/// Scales (an error of this size counts as 1).
const V2_FLOOR_L_SCALE: f64 = 4.0;
const V2_CEIL_L_SCALE: f64 = 4.0;
const V2_CONTRAST_SCALE: f64 = 0.08;
const V2_CAST_SCALE: f64 = 3.0;
const V2_HUE_SCALE_DEG: f64 = 8.0;
const V2_RATIO_SCALE: f64 = 0.25;
/// Dead bands: differences the content of two different pictures can explain cost nothing (L* units, ratio, Lab units,
/// degrees, ln ratio).
const V2_FLOOR_BAND_L: f64 = 6.0;
const V2_CEIL_BAND_L: f64 = 12.0;
const V2_CONTRAST_BAND: f64 = 0.12;
const V2_CAST_BAND_AB: f64 = 4.0;
const V2_HUE_BAND_DEG: f64 = 8.0;
const V2_RATIO_BAND: f64 = 0.45;
/// The regulariser's weight under V2 (the dead bands leave the fit free to wander, so it is stronger than V1's).
const V2_W_REG: f64 = 0.5;
/// The photo's median L* may move this far from the ungraded photo's before the anchor penalty starts.
const V2_ANCHOR_DEAD_BAND_L: f64 = 5.0;
/// Black floor and white ceiling quantiles of L*, in percent.
const V2_FLOOR_PCT: f64 = 2.0;
const V2_CEIL_PCT: f64 = 98.0;
/// Pixels with C* below this are "near neutral" and carry the tint of a lightness band.
const V2_NEUTRAL_CHROMA_MAX: f64 = 8.0;
/// A band's near-neutral set is also never smaller than this fraction of the band (its least coloured pixels), so that a
/// picture with a strong overall tint, in which nothing is under [`V2_NEUTRAL_CHROMA_MAX`], still has a cast to match.
const V2_NEUTRAL_MIN_FRACTION: f64 = 0.15;
/// A band's cast counts only if both pictures have at least this share of their pixels near neutral in it.
const V2_MIN_NEUTRAL_SHARE_PER_BAND: f64 = 0.01;
/// Pixels with C* below this have no useful hue and are left out of the per-hue measures.
const V2_HUE_MIN_CHROMA: f64 = 8.0;
/// A hue sector counts only if both pictures have at least this share of their pixels (hue-bearing) in it.
const V2_MIN_SECTOR_SHARE: f64 = 0.04;

/// What the objective reads from a picture.
#[derive(Clone, Debug, Default)]
struct Stats {
    /// L* quantiles minus the median (V1).
    shape: [f64; 19],
    median: f64,
    /// Mean (a*, b*) in the darkest, middle and lightest third of the picture's own L* (V1).
    band_ab: [[f64; 2]; 3],
    sector_chroma: [f64; SECTORS],
    sector_share: [f64; SECTORS],
    mean_chroma: f64,
    hi_share: f64,
    lo_share: f64,
    /// V2: black floor and white ceiling (L*), and the IQR of L* relative to the floor-to-ceiling spread.
    floor: f64,
    ceil: f64,
    iqr_ratio: f64,
    /// V2: mean (a*, b*) of the near-neutral pixels in each of the picture's own L* terciles, and the share of all
    /// pixels they are.
    neutral_ab: [[f64; 2]; 3],
    neutral_share: [f64; 3],
    /// V2: per hue sector, over the hue-bearing pixels: the mean hue angle (degrees), the mean chroma and the share.
    hue_angle: [f64; SECTORS],
    hue_chroma: [f64; SECTORS],
    hue_share: [f64; SECTORS],
    /// V2: the median C* of all pixels.
    median_chroma: f64,
    /// V3: the element statistics, when the picture has a scene.
    v3: Option<V3Stats>,
}

fn stats_of(px: &[Px], scene: Option<&Scene>, dnh: Option<&DnhCtx>) -> Stats {
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
    let (floor, ceil) = (l_at(V2_FLOOR_PCT), l_at(V2_CEIL_PCT));
    let iqr_ratio = (l_at(75.0) - l_at(25.0)) / (ceil - floor).max(1.0);
    let mut band_ab = [[0.0; 2]; 3];
    let mut band_n = [0usize; 3];
    let mut neutral_ab = [[0.0; 2]; 3];
    let mut neutral_n = [0usize; 3];
    let mut band_of = vec![0u8; n];
    let mut band_px: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (rank, &i) in idx.iter().enumerate() {
        let band = (rank * 3 / n).min(2);
        band_of[i] = band as u8;
        band_ab[band][0] += px[i].lab[1];
        band_ab[band][1] += px[i].lab[2];
        band_n[band] += 1;
        band_px[band].push((px[i].lab[1].hypot(px[i].lab[2]), px[i].lab[1], px[i].lab[2]));
    }
    for b in 0..3 {
        let mut c: Vec<f64> = band_px[b].iter().map(|p| p.0).collect();
        c.sort_unstable_by(f64::total_cmp);
        let floor_c = c
            .get((c.len() as f64 * V2_NEUTRAL_MIN_FRACTION) as usize)
            .copied()
            .unwrap_or(0.0);
        let limit = V2_NEUTRAL_CHROMA_MAX.max(floor_c);
        for &(chroma, a, bb) in &band_px[b] {
            if chroma <= limit {
                neutral_ab[b][0] += a;
                neutral_ab[b][1] += bb;
                neutral_n[b] += 1;
            }
        }
    }
    let mut neutral_share = [0.0; 3];
    for b in 0..3 {
        let k = band_n[b].max(1) as f64;
        band_ab[b] = [band_ab[b][0] / k, band_ab[b][1] / k];
        let m = neutral_n[b].max(1) as f64;
        neutral_ab[b] = [neutral_ab[b][0] / m, neutral_ab[b][1] / m];
        neutral_share[b] = neutral_n[b] as f64 / n as f64;
    }
    let mut sector_c = [0.0; SECTORS];
    let mut sector_n = [0usize; SECTORS];
    let (mut hue_c, mut hue_n) = ([0.0; SECTORS], [0usize; SECTORS]);
    let mut hue_vec = [[0.0; 2]; SECTORS];
    let (mut total_c, mut hi, mut lo) = (0.0, 0usize, 0usize);
    let mut chromas = Vec::with_capacity(n);
    for p in px {
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        chromas.push(c);
        total_c += c;
        let h = b.atan2(a).to_degrees();
        let s = (((h + 30.0).rem_euclid(360.0)) / 60.0) as usize % SECTORS;
        sector_c[s] += c;
        sector_n[s] += 1;
        if c >= V2_HUE_MIN_CHROMA {
            hue_c[s] += c;
            hue_n[s] += 1;
            hue_vec[s][0] += a / c;
            hue_vec[s][1] += b / c;
        }
        hi += usize::from(p.hi);
        lo += usize::from(p.lo);
    }
    chromas.sort_unstable_by(f64::total_cmp);
    let median_chroma = chromas[n / 2];
    let mut sector_share = [0.0; SECTORS];
    let (mut hue_angle, mut hue_chroma, mut hue_share) =
        ([0.0; SECTORS], [0.0; SECTORS], [0.0; SECTORS]);
    for s in 0..SECTORS {
        sector_share[s] = sector_n[s] as f64 / n as f64;
        sector_c[s] = if sector_n[s] > 0 {
            sector_c[s] / sector_n[s] as f64
        } else {
            0.0
        };
        hue_share[s] = hue_n[s] as f64 / n as f64;
        if hue_n[s] > 0 {
            hue_chroma[s] = hue_c[s] / hue_n[s] as f64;
            hue_angle[s] = hue_vec[s][1].atan2(hue_vec[s][0]).to_degrees();
        }
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
        floor,
        ceil,
        iqr_ratio,
        neutral_ab,
        neutral_share,
        hue_angle,
        hue_chroma,
        hue_share,
        median_chroma,
        v3: scene.map(|sc| v3_stats(px, sc, &band_of, dnh)),
    }
}

/// The objective's terms, unweighted (each is 0 for a perfect match; an error of one "scale" counts about 1).
#[derive(Clone, Copy, Debug)]
struct Terms {
    v2: bool,
    v3: bool,
    // V1
    shape: f64,
    level: f64,
    ab: f64,
    sector: f64,
    chroma: f64,
    // V2
    floor: f64,
    ceil: f64,
    contrast: f64,
    anchor: f64,
    cast: f64,
    hue: f64,
    sector_chroma_ratio: f64,
    chroma_ratio: f64,
    // V3 (floor, ceil, anchor and cast are shared with V2)
    elem_colour: f64,
    elem_exposure: f64,
    separation: f64,
    global_sat: f64,
    dnh: f64,
    // V5
    v5: bool,
    t5: V5T,
    reg_local: f64,
    // both
    reg: f64,
    clip: f64,
}

impl Terms {
    fn total(&self) -> f64 {
        self.total_masked(MASK_ALL)
    }
    /// The weighted total of the V2 terms in `mask` (V1 ignores it).
    fn total_masked(&self, mask: u8) -> f64 {
        let on = |m: u8| if mask & m != 0 { 1.0 } else { 0.0 };
        let common = if self.v2 || self.v3 || self.v5 {
            V2_W_REG
        } else {
            W_REG
        } * self.reg
            + V2_W_REG * self.reg_local
            + W_CLIP * self.clip;
        if self.v5 {
            let t = &self.t5;
            on(MASK_BASE) * (t.floor + t.ceil + t.anchor)
                + on(MASK_TONE) * t.contrast
                + on(MASK_CAST) * (t.tint + t.frame_cast)
                + on(MASK_CHROMA) * (t.sat + t.sec_chroma)
                + on(MASK_HUE) * t.sec_hue
                + on(MASK_ELEM_COL) * (t.el_hue + t.el_chroma)
                + on(MASK_ELEM_EXP) * (t.el_exp + t.subject)
                + t.guard_hue
                + t.guard_skin
                + t.guard_sky
                + t.guard_hl
                + t.guard_purple
                + t.guard_white_dir
                + t.guard_skin_drift
                + t.guard_mem
                + t.guard_purple_px
                + t.guard_skin_chroma
                + t.guard_turn
                + common
        } else if self.v3 {
            on(MASK_BASE) * (self.floor + self.ceil + V2_W_ANCHOR * self.anchor)
                + on(MASK_TONE) * (self.elem_exposure + self.separation)
                + on(MASK_CAST) * V3_W_CAST * self.cast
                + on(MASK_HUE) * (self.elem_colour + self.dnh)
                + on(MASK_CHROMA) * self.global_sat
                + common
        } else if self.v2 {
            on(MASK_TONE)
                * (V2_W_FLOOR * self.floor
                    + V2_W_CEIL * self.ceil
                    + V2_W_CONTRAST * self.contrast
                    + V2_W_ANCHOR * self.anchor)
                + on(MASK_CAST) * V2_W_CAST * self.cast
                + on(MASK_HUE) * V2_W_HUE * self.hue
                + on(MASK_CHROMA)
                    * (V2_W_SECTOR_CHROMA * self.sector_chroma_ratio
                        + V2_W_CHROMA_RATIO * self.chroma_ratio)
                + common
        } else {
            W_SHAPE * self.shape
                + W_LEVEL * self.level
                + W_AB * self.ab
                + W_SECTOR * self.sector
                + W_CHROMA * self.chroma
                + common
        }
    }
    fn to_json(self) -> Value {
        if self.v5 {
            let t = self.t5;
            json!({
                "objective": "v5",
                "floor": t.floor, "ceiling": t.ceil, "contrast": t.contrast, "anchor": t.anchor, "tint": t.tint, "frame_cast": t.frame_cast,
                "saturation": t.sat, "sector_hue": t.sec_hue, "sector_chroma": t.sec_chroma,
                "element_hue": t.el_hue, "element_chroma": t.el_chroma, "element_exposure": t.el_exp,
                "subject_separation": t.subject,
                "guard_hue_drift": t.guard_hue, "guard_skin": t.guard_skin, "guard_sky_chroma": t.guard_sky, "guard_highlights": t.guard_hl, "guard_purple": t.guard_purple, "guard_white_direction": t.guard_white_dir, "guard_skin_drift": t.guard_skin_drift, "guard_memory": t.guard_mem, "guard_purple_pixels": t.guard_purple_px, "guard_skin_chroma": t.guard_skin_chroma, "guard_turn": t.guard_turn,
                "regulariser_global": self.reg, "regulariser_local": self.reg_local, "clip": self.clip,
                "weighted_total": self.total(),
            })
        } else if self.v3 {
            json!({
                "objective": "v3",
                "element_colour": self.elem_colour, "neutral_cast": self.cast, "black_floor": self.floor,
                "white_ceiling": self.ceil, "element_exposure": self.elem_exposure,
                "subject_separation": self.separation, "median_anchor": self.anchor,
                "global_saturation": self.global_sat, "do_no_harm": self.dnh, "regulariser": self.reg, "clip": self.clip,
                "weighted_total": self.total(),
            })
        } else if self.v2 {
            json!({
                "objective": "v2",
                "black_floor": self.floor, "white_ceiling": self.ceil, "midtone_contrast": self.contrast,
                "median_anchor": self.anchor, "neutral_cast": self.cast, "hue_shift": self.hue,
                "hue_chroma_ratio": self.sector_chroma_ratio, "median_chroma_ratio": self.chroma_ratio,
                "regulariser": self.reg, "clip": self.clip, "weighted_total": self.total(),
                "weights": { "black_floor": V2_W_FLOOR, "white_ceiling": V2_W_CEIL, "midtone_contrast": V2_W_CONTRAST,
                    "median_anchor": V2_W_ANCHOR, "neutral_cast": V2_W_CAST, "hue_shift": V2_W_HUE,
                    "hue_chroma_ratio": V2_W_SECTOR_CHROMA, "median_chroma_ratio": V2_W_CHROMA_RATIO,
                    "regulariser": W_REG, "clip": W_CLIP },
            })
        } else {
            json!({
                "objective": "v1",
                "shape": self.shape, "level": self.level, "ab_bands": self.ab, "chroma_sectors": self.sector,
                "mean_chroma": self.chroma, "regulariser": self.reg, "clip": self.clip,
                "weighted_total": self.total(),
                "weights": { "shape": W_SHAPE, "level": W_LEVEL, "ab_bands": W_AB, "chroma_sectors": W_SECTOR,
                    "mean_chroma": W_CHROMA, "regulariser": W_REG, "clip": W_CLIP },
            })
        }
    }
}

/// A difference with a dead band: 0 inside `band`, the excess outside.
fn dead(d: f64, band: f64) -> f64 {
    (d.abs() - band).max(0.0)
}

/// Wraps an angle difference to -180..180 degrees.
fn wrap_deg(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

/// The terms of `out` against `reference`. `anchor` is the ungraded photo's median L* (V2 keeps the photo's key).
fn terms(obj: Objective, out: &Stats, reference: &Stats, p0: &Stats, x: &[f64; N_PARAMS]) -> Terms {
    let anchor = p0.median;
    let reg = x[..N_GLOBAL].iter().map(|v| v * v).sum::<f64>() / N_GLOBAL as f64;
    let over = |o: f64, r: f64| ((o - r - CLIP_SLACK).max(0.0) / CLIP_SCALE).powi(2);
    let clip = over(out.hi_share, reference.hi_share) + over(out.lo_share, reference.lo_share);
    let mut t = Terms {
        v2: obj == Objective::V2,
        v3: obj == Objective::V3,
        v5: obj.net(),
        t5: V5T::default(),
        reg_local: 0.0,
        elem_colour: 0.0,
        elem_exposure: 0.0,
        separation: 0.0,
        global_sat: 0.0,
        dnh: 0.0,
        shape: 0.0,
        level: 0.0,
        ab: 0.0,
        sector: 0.0,
        chroma: 0.0,
        floor: 0.0,
        ceil: 0.0,
        contrast: 0.0,
        anchor: 0.0,
        cast: 0.0,
        hue: 0.0,
        sector_chroma_ratio: 0.0,
        chroma_ratio: 0.0,
        reg,
        clip,
    };
    if obj == Objective::V3 {
        terms_v3(out, reference, p0, None, &mut t);
        return t;
    }
    if obj == Objective::V2 {
        t.floor = (dead(out.floor - reference.floor, V2_FLOOR_BAND_L) / V2_FLOOR_L_SCALE).powi(2);
        t.ceil = (dead(out.ceil - reference.ceil, V2_CEIL_BAND_L) / V2_CEIL_L_SCALE).powi(2);
        t.contrast = (dead(out.iqr_ratio - reference.iqr_ratio, V2_CONTRAST_BAND)
            / V2_CONTRAST_SCALE)
            .powi(2);
        t.anchor =
            (((out.median - anchor).abs() - V2_ANCHOR_DEAD_BAND_L).max(0.0) / L_SCALE).powi(2);
        let (mut sum, mut k) = (0.0, 0usize);
        for b in 0..3 {
            if out.neutral_share[b] >= V2_MIN_NEUTRAL_SHARE_PER_BAND
                && reference.neutral_share[b] >= V2_MIN_NEUTRAL_SHARE_PER_BAND
            {
                let da = out.neutral_ab[b][0] - reference.neutral_ab[b][0];
                let db = out.neutral_ab[b][1] - reference.neutral_ab[b][1];
                sum += (dead(da.hypot(db), V2_CAST_BAND_AB) / V2_CAST_SCALE).powi(2);
                k += 1;
            }
        }
        t.cast = if k > 0 { sum / k as f64 } else { 0.0 };
        let (mut sh, mut sr, mut k) = (0.0, 0.0, 0usize);
        for s in 0..SECTORS {
            if out.hue_share[s] >= V2_MIN_SECTOR_SHARE
                && reference.hue_share[s] >= V2_MIN_SECTOR_SHARE
            {
                sh += (dead(
                    wrap_deg(out.hue_angle[s] - reference.hue_angle[s]),
                    V2_HUE_BAND_DEG,
                ) / V2_HUE_SCALE_DEG)
                    .powi(2);
                sr += (dead(
                    (out.hue_chroma[s] / reference.hue_chroma[s].max(1e-6)).ln(),
                    V2_RATIO_BAND,
                ) / V2_RATIO_SCALE)
                    .powi(2);
                k += 1;
            }
        }
        if k > 0 {
            t.hue = sh / k as f64;
            t.sector_chroma_ratio = sr / k as f64;
        }
        t.chroma_ratio = (dead(
            ((out.median_chroma + 1.0) / (reference.median_chroma + 1.0)).ln(),
            V2_RATIO_BAND,
        ) / V2_RATIO_SCALE)
            .powi(2);
        return t;
    }
    t.shape = out
        .shape
        .iter()
        .zip(&reference.shape)
        .map(|(a, b)| ((a - b) / L_SCALE).powi(2))
        .sum::<f64>()
        / 19.0;
    t.level = ((out.median - reference.median) / L_SCALE).powi(2);
    for b in 0..3 {
        let da = (out.band_ab[b][0] - reference.band_ab[b][0]) / AB_SCALE;
        let db = (out.band_ab[b][1] - reference.band_ab[b][1]) / AB_SCALE;
        t.ab += 0.5 * (da * da + db * db) / 3.0;
    }
    let (mut sum_w, mut sum) = (0.0, 0.0);
    for s in 0..SECTORS {
        let w = (out.sector_share[s] * reference.sector_share[s]).sqrt();
        sum_w += w;
        sum += w * ((out.sector_chroma[s] - reference.sector_chroma[s]) / CHROMA_SCALE).powi(2);
    }
    t.sector = if sum_w > 0.0 { sum / sum_w } else { 0.0 };
    t.chroma = ((out.mean_chroma - reference.mean_chroma) / CHROMA_SCALE).powi(2);
    t
}

// ---------------------------------------------------------------------------------------------
// V3: the hand-written, scene-aware grading model (`refgrade-notes/07-grading-model-v3.md`)
// ---------------------------------------------------------------------------------------------

/// Element ids of the label step, plus [`ELEM_SKIN`], which is a subset of `person`.
const ELEM_SKY: usize = 1;
const ELEM_VEGETATION: usize = 2;
const ELEM_WATER: usize = 3;
const ELEM_GROUND: usize = 4;
const ELEM_ARCHITECTURE: usize = 5;
const ELEM_INTERIOR: usize = 6;
const ELEM_PERSON: usize = 7;
const ELEM_OTHER_THING: usize = 8;
const ELEM_SKIN: usize = 10;
const N_ELEM: usize = 11;
const ELEM_NAMES: [&str; N_ELEM] = [
    "unlabelled",
    "sky",
    "vegetation",
    "water",
    "ground",
    "architecture",
    "interior",
    "person",
    "other_thing",
    "other_stuff",
    "skin",
];

/// An element with C* at or below this has no hue (the label step's statistics use pixels above it).
const V3_HUE_MIN_CHROMA: f64 = 5.0;
/// A person (or subject) element needs this share of the frame.
const V3_PERSON_MIN_SHARE: f64 = 0.03;
/// An element is eligible with this share of the frame in both images and this many sample pixels in the photo.
const V3_ELEM_MIN_SHARE: f64 = 0.02;
const V3_ELEM_MIN_SAMPLES: usize = 150;
/// Hue reliability: an element's hue counts with `min(1, C_median / this)` in each image.
const V3_HUE_RELIABLE_CHROMA: f64 = 12.0;
/// Key mismatch: the reference is low key below this median L*, the photo high key above this.
const V3_LOW_KEY_REF_L: f64 = 25.0;
const V3_HIGH_KEY_PHOTO_L: f64 = 40.0;
/// Element-exposure and separation weight under a key mismatch.
const V3_KEY_MISMATCH_EXPOSURE_WEIGHT: f64 = 0.3;
/// The floor's move fraction, normally and under a key mismatch.
const V3_FLOOR_MOVE: f64 = 0.8;
const V3_FLOOR_MOVE_KEY_MISMATCH: f64 = 0.4;
const V3_CEIL_MOVE: f64 = 0.6;
const V3_FLOOR_CEIL_CLAMP_L: f64 = 12.0;
const V3_FLOOR_CEIL_SCALE_L: f64 = 2.0;
const V3_FLOOR_CEIL_BAND_L: f64 = 1.0;
/// Element exposure relative to the frame.
const V3_EXPOSURE_MOVE: f64 = 0.5;
const V3_EXPOSURE_CLAMP_L: f64 = 10.0;
const V3_EXPOSURE_SCALE_L: f64 = 3.0;
const V3_EXPOSURE_BAND_L: f64 = 2.0;
/// Subject separation.
const V3_SEP_MOVE: f64 = 0.6;
const V3_SEP_CLAMP_L: f64 = 12.0;
const V3_SEP_SCALE_L: f64 = 3.0;
const V3_SEP_BAND_L: f64 = 2.0;
/// Element colour: reliability clamp of the chroma ratio, scales and dead bands.
const V3_RATIO_MIN: f64 = 0.45;
const V3_RATIO_MAX: f64 = 1.6;
const V3_HUE_ERR_SCALE_DEG: f64 = 10.0;
const V3_CHROMA_ERR_SCALE: f64 = 4.0;
const V3_HUE_BAND_DEG: f64 = 3.0;
const V3_CHROMA_BAND: f64 = 1.0;
/// Skin's chroma clamp.
const V3_SKIN_C_MIN: f64 = 12.0;
const V3_SKIN_C_MAX: f64 = 35.0;
/// Global saturation.
const V3_SAT_POWER: f64 = 0.6;
const V3_SAT_MIN: f64 = 0.55;
const V3_SAT_MAX: f64 = 1.4;
const V3_SAT_SCALE: f64 = 0.05;
const V3_SAT_BAND: f64 = 0.03;
/// Round 3b do-no-harm: weight, sector threshold (share of the pixels outside the eligible elements), scales and bands.
const V3_DNH_WEIGHT: f64 = 1.2;
const V3_DNH_MIN_SHARE: f64 = 0.02;
const V3_DNH_HUE_SCALE_DEG: f64 = 8.0;
const V3_DNH_RATIO_SCALE: f64 = 0.08;
const V3_DNH_HUE_BAND_DEG: f64 = 2.0;
const V3_DNH_RATIO_BAND: f64 = 0.03;
/// Neutral cast weight (the term itself is V2's).
const V3_W_CAST: f64 = 1.0;
/// Fallback skin: person pixels with L* 25..85, C* 8..45 and hue 20..80 degrees.
const V3_FALLBACK_SKIN_L: (f64, f64) = (25.0, 85.0);
const V3_FALLBACK_SKIN_C: (f64, f64) = (8.0, 45.0);
const V3_FALLBACK_SKIN_HUE: (f64, f64) = (20.0, 80.0);
/// A person component needs this share of the frame to get its own MediaPipe crop (implementation choice).
const V3_SKIN_COMPONENT_MIN_SHARE: f64 = 0.005;
/// Margin of the MediaPipe crop around a person's bounding box (implementation choice).
const V3_SKIN_CROP_MARGIN: f64 = 1.15;
/// A subject needs this share of the frame (implementation choice; the notes leave it open).
const V3_SUBJECT_MIN_SHARE: f64 = 0.01;

/// One element's colour model: (id, w_hue, cap in degrees, w_sat, hue window, w_term).
struct ElemModel {
    id: usize,
    w_hue: f64,
    cap_deg: f64,
    w_sat: f64,
    window: Option<(f64, f64)>,
    w_term: f64,
}

const ELEM_MODELS: [ElemModel; 7] = [
    ElemModel {
        id: ELEM_SKY,
        w_hue: 0.70,
        cap_deg: 40.0,
        w_sat: 0.80,
        window: Some((175.0, 285.0)),
        w_term: 1.5,
    },
    ElemModel {
        id: ELEM_VEGETATION,
        w_hue: 0.60,
        cap_deg: 35.0,
        w_sat: 0.80,
        window: Some((70.0, 180.0)),
        w_term: 1.2,
    },
    ElemModel {
        id: ELEM_WATER,
        w_hue: 0.60,
        cap_deg: 35.0,
        w_sat: 0.70,
        window: Some((150.0, 285.0)),
        w_term: 1.0,
    },
    ElemModel {
        id: ELEM_GROUND,
        w_hue: 0.50,
        cap_deg: 30.0,
        w_sat: 0.70,
        window: None,
        w_term: 1.0,
    },
    ElemModel {
        id: ELEM_ARCHITECTURE,
        w_hue: 0.30,
        cap_deg: 25.0,
        w_sat: 0.50,
        window: None,
        w_term: 0.6,
    },
    ElemModel {
        id: ELEM_INTERIOR,
        w_hue: 0.30,
        cap_deg: 25.0,
        w_sat: 0.50,
        window: None,
        w_term: 0.6,
    },
    ElemModel {
        id: ELEM_SKIN,
        w_hue: 0.35,
        cap_deg: 15.0,
        w_sat: 0.50,
        window: Some((30.0, 70.0)),
        w_term: 2.0,
    },
];
/// Elements whose exposure relative to the frame is modelled.
const EXPOSURE_ELEMS: [usize; 5] = [
    ELEM_SKY,
    ELEM_VEGETATION,
    ELEM_WATER,
    ELEM_GROUND,
    ELEM_ARCHITECTURE,
];

/// What the labels say about the pixels of a picture: the element of each, whether it is skin, whether it is the subject.
#[derive(Clone, Debug, Default)]
struct Scene {
    elem: Vec<u8>,
    skin: Vec<bool>,
    subject: Vec<bool>,
    /// Where the skin came from.
    skin_source: &'static str,
}

/// The statistics of one element, as the label step defines them.
#[derive(Clone, Debug)]
struct V3Stats {
    share: [f64; N_ELEM],
    count: [usize; N_ELEM],
    light: [f64; N_ELEM],
    /// Circular mean hue of the pixels with C* above [`V3_HUE_MIN_CHROMA`] (degrees), `None` without such pixels.
    hue: [Option<f64>; N_ELEM],
    chroma: [f64; N_ELEM],
    subject_light: f64,
    subject_count: usize,
    background_light: f64,
    background_count: usize,
    /// Median C* of the labelled non-person pixels and their share of the frame.
    nonperson_chroma: f64,
    nonperson_share: f64,
    /// Mean (a*, b*) of the near-neutral pixels of the neutral-capable elements in each of the picture's L* terciles, and the
    /// share of all pixels they are (round 3b: sky, vegetation, water and skin never count).
    neutral_ab: [[f64; 2]; 3],
    neutral_share: [f64; 3],
    /// The do-no-harm sums per hue sector of photo0, when the evaluator gave a [`DnhCtx`].
    dnh: Option<[DnhSector; SECTORS]>,
    dnh_total: usize,
    n_px: usize,
}

/// Do-no-harm sums of one hue sector over the pixels outside every eligible element.
#[derive(Clone, Copy, Debug, Default)]
struct DnhSector {
    count: usize,
    /// Sum of the pixels' hue shifts against photo0, degrees (each wrapped to -180..180).
    shift_sum: f64,
    /// Sums of the candidate's and of photo0's C*.
    chroma_cand: f64,
    chroma_p0: f64,
}

/// What the do-no-harm term needs from photo0, fixed for a pair: its pixels, which of them are in scope (outside every
/// eligible element) and their hue sector (255 for none).
#[derive(Clone, Debug)]
struct DnhCtx {
    p0_px: Vec<Px>,
    sector: Vec<u8>,
}

fn median_of(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    v[v.len() / 2]
}

fn v3_stats(px: &[Px], scene: &Scene, band: &[u8], dnh: Option<&DnhCtx>) -> V3Stats {
    let n = px.len();
    let mut lights: Vec<Vec<f64>> = vec![Vec::new(); N_ELEM];
    let mut chromas: Vec<Vec<f64>> = vec![Vec::new(); N_ELEM];
    let mut hue_vec = [[0.0f64; 2]; N_ELEM];
    let mut hue_n = [0usize; N_ELEM];
    let (mut subj, mut bg): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut np_c: Vec<f64> = Vec::new();
    let mut capable: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, p) in px.iter().enumerate() {
        let e = usize::from(scene.elem[i]);
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        let mut add = |k: usize| {
            lights[k].push(p.lab[0]);
            chromas[k].push(c);
            if c > V3_HUE_MIN_CHROMA {
                hue_vec[k][0] += a / c;
                hue_vec[k][1] += b / c;
                hue_n[k] += 1;
            }
        };
        add(e);
        if scene.skin[i] {
            add(ELEM_SKIN);
        }
        if scene.subject[i] {
            subj.push(p.lab[0]);
        } else if e != 0 {
            bg.push(p.lab[0]);
        }
        if e != 0 && e != ELEM_PERSON {
            np_c.push(c);
        }
        // Round 3b: only these elements carry a neutral (never sky, vegetation, water or skin).
        let neutral_capable = matches!(
            e,
            0 | ELEM_GROUND | ELEM_ARCHITECTURE | ELEM_INTERIOR | ELEM_OTHER_THING | 9
        ) || (e == ELEM_PERSON && !scene.skin[i]);
        if neutral_capable {
            capable[usize::from(band[i])].push((c, a, b));
        }
    }
    let mut s = V3Stats {
        share: [0.0; N_ELEM],
        count: [0; N_ELEM],
        light: [0.0; N_ELEM],
        hue: [None; N_ELEM],
        chroma: [0.0; N_ELEM],
        subject_light: 0.0,
        subject_count: subj.len(),
        background_light: 0.0,
        background_count: bg.len(),
        nonperson_chroma: 0.0,
        nonperson_share: np_c.len() as f64 / n as f64,
        neutral_ab: [[0.0; 2]; 3],
        neutral_share: [0.0; 3],
        dnh: None,
        dnh_total: 0,
        n_px: n,
    };
    for b in 0..3 {
        let mut cs: Vec<f64> = capable[b].iter().map(|p| p.0).collect();
        cs.sort_unstable_by(f64::total_cmp);
        let floor_c = cs
            .get((cs.len() as f64 * V2_NEUTRAL_MIN_FRACTION) as usize)
            .copied()
            .unwrap_or(0.0);
        let limit = V2_NEUTRAL_CHROMA_MAX.max(floor_c);
        let (mut sa, mut sb, mut k) = (0.0, 0.0, 0usize);
        for &(c, a, bb) in &capable[b] {
            if c <= limit {
                sa += a;
                sb += bb;
                k += 1;
            }
        }
        s.neutral_ab[b] = [sa / k.max(1) as f64, sb / k.max(1) as f64];
        s.neutral_share[b] = k as f64 / n as f64;
    }
    if let Some(ctx) = dnh {
        let mut acc = [DnhSector::default(); SECTORS];
        let mut total = 0usize;
        for (i, p) in px.iter().enumerate() {
            let sec = ctx.sector[i];
            if sec == 255 {
                continue;
            }
            total += 1;
            let q = &ctx.p0_px[i];
            let (c, h) = (
                p.lab[1].hypot(p.lab[2]),
                p.lab[2].atan2(p.lab[1]).to_degrees(),
            );
            let (c0, h0) = (
                q.lab[1].hypot(q.lab[2]),
                q.lab[2].atan2(q.lab[1]).to_degrees(),
            );
            let a = &mut acc[usize::from(sec)];
            a.count += 1;
            a.shift_sum += wrap_deg(h - h0);
            a.chroma_cand += c;
            a.chroma_p0 += c0;
        }
        s.dnh = Some(acc);
        s.dnh_total = total;
    }
    for k in 0..N_ELEM {
        s.count[k] = lights[k].len();
        s.share[k] = lights[k].len() as f64 / n as f64;
        s.light[k] = median_of(&mut lights[k]);
        s.chroma[k] = median_of(&mut chromas[k]);
        if hue_n[k] > 0 {
            s.hue[k] = Some(
                hue_vec[k][1]
                    .atan2(hue_vec[k][0])
                    .to_degrees()
                    .rem_euclid(360.0),
            );
        }
    }
    s.subject_light = median_of(&mut subj);
    s.background_light = median_of(&mut bg);
    s.nonperson_chroma = median_of(&mut np_c);
    s
}

/// Connected components (4-connected) of a mask: (pixel list, bounding box x0, y0, x1, y1).
fn components(mask: &[bool], w: usize, h: usize) -> Vec<(usize, [usize; 4])> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let (mut count, mut bb) = (0usize, [w, h, 0usize, 0usize]);
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            count += 1;
            bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h && mask[ny * w + nx] && !seen[ny * w + nx] {
                    seen[ny * w + nx] = true;
                    stack.push(ny * w + nx);
                }
            }
        }
        out.push((count, bb));
    }
    out
}

/// The skin of a picture: MediaPipe Selfie Multiclass (body skin and face skin) intersected with the person element, run on
/// one square crop per person component; or, without the network, the fallback rule of the notes.
fn skin_mask(
    display: &Img,
    elem: &[u8],
    seg: Option<&ym_ml::mediapipe::selfie::SelfieSegmenter>,
    conv: &LabConv,
) -> (Vec<bool>, &'static str) {
    let (w, h) = (display.w, display.h);
    let person: Vec<bool> = elem
        .iter()
        .map(|&e| usize::from(e) == ELEM_PERSON)
        .collect();
    let share = person.iter().filter(|&&p| p).count() as f64 / (w * h) as f64;
    let mut skin = vec![false; w * h];
    if share < V3_PERSON_MIN_SHARE {
        return (skin, "none (person under 3 % of the frame)");
    }
    if let Some(seg) = seg {
        let rgb8 = linear_to_rgb8(display);
        if let Ok(src) = ym_ml::mediapipe::blazeface::Rgb8::new(w, h, &rgb8.data) {
            let mut ok = true;
            for (count, bb) in components(&person, w, h) {
                if (count as f64) < V3_SKIN_COMPONENT_MIN_SHARE * (w * h) as f64 {
                    continue;
                }
                let (bw, bh) = ((bb[2] - bb[0] + 1) as f64, (bb[3] - bb[1] + 1) as f64);
                let roi = ym_ml::mediapipe::roi::Roi {
                    center_px: [bb[0] as f64 + bw / 2.0, bb[1] as f64 + bh / 2.0],
                    side_px: bw.max(bh) * V3_SKIN_CROP_MARGIN,
                    rotation_rad: 0.0,
                };
                match seg.segment(&src, &roi, ym_ml::mediapipe::blazeface::Resample::Box) {
                    Ok(map) => {
                        for y in bb[1]..=bb[3] {
                            for x in bb[0]..=bb[2] {
                                let i = y * w + x;
                                if !person[i] {
                                    continue;
                                }
                                let [u, v] = roi.from_picture([x as f64 + 0.5, y as f64 + 0.5]);
                                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                                    continue;
                                }
                                use ym_ml::mediapipe::selfie::Class;
                                let best = [
                                    Class::Background,
                                    Class::Hair,
                                    Class::BodySkin,
                                    Class::FaceSkin,
                                    Class::Clothes,
                                    Class::Others,
                                ]
                                .into_iter()
                                .max_by(|a, b| map.prob(*a, u, v).total_cmp(&map.prob(*b, u, v)));
                                skin[i] = matches!(best, Some(Class::BodySkin | Class::FaceSkin));
                            }
                        }
                    }
                    Err(_) => ok = false,
                }
            }
            if ok {
                return (
                    skin,
                    "MediaPipe Selfie Multiclass (body skin + face skin) intersected with person",
                );
            }
        }
        skin.fill(false);
    }
    for i in 0..w * h {
        if !person[i] {
            continue;
        }
        let p = conv.px([
            f64::from(display.px[3 * i]),
            f64::from(display.px[3 * i + 1]),
            f64::from(display.px[3 * i + 2]),
        ]);
        let (l, a, b) = (p.lab[0], p.lab[1], p.lab[2]);
        let (c, hue) = (a.hypot(b), b.atan2(a).to_degrees().rem_euclid(360.0));
        skin[i] = (V3_FALLBACK_SKIN_L.0..=V3_FALLBACK_SKIN_L.1).contains(&l)
            && (V3_FALLBACK_SKIN_C.0..=V3_FALLBACK_SKIN_C.1).contains(&c)
            && (V3_FALLBACK_SKIN_HUE.0..=V3_FALLBACK_SKIN_HUE.1).contains(&hue);
    }
    (
        skin,
        "fallback rule (person pixels with L* 25..85, C* 8..45, hue 20..80)",
    )
}

/// The subject of a picture: the person element if it covers at least 3 % of the frame, else the `other_thing` pixels in the
/// central 60 % of the frame (when they cover at least [`V3_SUBJECT_MIN_SHARE`]).
fn subject_mask(elem: &[u8], w: usize, h: usize) -> Vec<bool> {
    let n = (w * h) as f64;
    let person: Vec<bool> = elem
        .iter()
        .map(|&e| usize::from(e) == ELEM_PERSON)
        .collect();
    if person.iter().filter(|&&p| p).count() as f64 / n >= V3_PERSON_MIN_SHARE {
        return person;
    }
    let central = |i: usize| {
        let (x, y) = ((i % w) as f64 / w as f64, (i / w) as f64 / h as f64);
        (0.2..0.8).contains(&x) && (0.2..0.8).contains(&y)
    };
    let thing: Vec<bool> = (0..w * h)
        .map(|i| usize::from(elem[i]) == ELEM_OTHER_THING && central(i))
        .collect();
    if thing.iter().filter(|&&p| p).count() as f64 / n >= V3_SUBJECT_MIN_SHARE {
        thing
    } else {
        vec![false; w * h]
    }
}

/// Reads an indexed PNG of element ids.
fn read_label_png(path: &Path) -> R<(usize, usize, Vec<u8>)> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(fs::read(path)?));
    dec.set_transformations(png::Transformations::IDENTITY);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or("label png too large")?];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    if info.bit_depth != png::BitDepth::Eight || info.color_type != png::ColorType::Indexed {
        return Err(format!("{}: expected an 8-bit indexed label png", path.display()).into());
    }
    buf.truncate(w * h);
    Ok((w, h, buf))
}

/// Nearest-neighbour resize of a label map, after cutting `crop` (top, bottom, left, right) off its edges.
fn resize_labels(
    src: &[u8],
    sw: usize,
    sh: usize,
    crop: [usize; 4],
    dw: usize,
    dh: usize,
) -> Vec<u8> {
    let (cw, ch) = (sw - crop[2] - crop[3], sh - crop[0] - crop[1]);
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy = crop[0] + (((y as f64 + 0.5) / dh as f64) * ch as f64) as usize;
        for x in 0..dw {
            let sx = crop[2] + (((x as f64 + 0.5) / dw as f64) * cw as f64) as usize;
            out[y * dw + x] = src[sy.min(sh - 1) * sw + sx.min(sw - 1)];
        }
    }
    out
}

/// A scene for a picture of `w` by `h` whose display version is `display`.
fn make_scene(
    elem: Vec<u8>,
    display: &Img,
    seg: Option<&ym_ml::mediapipe::selfie::SelfieSegmenter>,
    conv: &LabConv,
) -> Scene {
    let (skin, source) = skin_mask(display, &elem, seg, conv);
    let subject = subject_mask(&elem, display.w, display.h);
    Scene {
        elem,
        skin,
        subject,
        skin_source: source,
    }
}

/// The photo's scene gathered at the sample pixels.
fn gather_scene(scene: &Scene, idx: &[usize]) -> Scene {
    Scene {
        elem: idx.iter().map(|&i| scene.elem[i]).collect(),
        skin: idx.iter().map(|&i| scene.skin[i]).collect(),
        subject: idx.iter().map(|&i| scene.subject[i]).collect(),
        skin_source: scene.skin_source,
    }
}

fn wrap180(d: f64) -> f64 {
    wrap_deg(d)
}

/// Which elements are eligible in a pair: at least 2 % of the frame in both images and enough photo samples.
fn eligibility(r: &V3Stats, z: &V3Stats) -> [bool; N_ELEM] {
    let mut e = [false; N_ELEM];
    for (k, flag) in e.iter_mut().enumerate() {
        *flag = r.share[k] >= V3_ELEM_MIN_SHARE
            && z.share[k] >= V3_ELEM_MIN_SHARE
            && z.count[k] >= V3_ELEM_MIN_SAMPLES;
    }
    e
}

/// Round 3b, do-no-harm: the hue sector (of photo0) of every photo pixel outside every eligible element with C* above
/// [`V3_HUE_MIN_CHROMA`]; 255 for the others. Clothing (person minus skin) is outside, and so are unlabelled, other things,
/// and elements with no eligible counterpart.
fn dnh_sectors(p0: &[Px], scene: &Scene, elig: &[bool; N_ELEM]) -> Vec<u8> {
    p0.iter()
        .enumerate()
        .map(|(i, p)| {
            let e = usize::from(scene.elem[i]);
            let in_element = (e != ELEM_PERSON && e < ELEM_SKIN && elig[e])
                || (scene.skin[i] && elig[ELEM_SKIN]);
            let (a, b) = (p.lab[1], p.lab[2]);
            if in_element || a.hypot(b) <= V3_HUE_MIN_CHROMA {
                return 255;
            }
            let h = b.atan2(a).to_degrees();
            ((((h + 30.0).rem_euclid(360.0)) / 60.0) as usize % SECTORS) as u8
        })
        .collect()
}

/// The V3 terms (`07-grading-model-v3.md` sections 2 and 3) of `out` against the reference, with the photo's own
/// statistics `p0` as the base of every move. `diag` collects what each element's target was.
fn terms_v3(
    out: &Stats,
    reference: &Stats,
    p0: &Stats,
    diag: Option<&mut Vec<Value>>,
    t: &mut Terms,
) {
    let (Some(o), Some(r), Some(z)) = (&out.v3, &reference.v3, &p0.v3) else {
        return;
    };
    let mut diag = diag;
    let low_key_ref = reference.median < V3_LOW_KEY_REF_L;
    let high_key_photo = p0.median > V3_HIGH_KEY_PHOTO_L;
    let mismatch = low_key_ref && high_key_photo;
    let exposure_weight = if mismatch {
        V3_KEY_MISMATCH_EXPOSURE_WEIGHT
    } else {
        1.0
    };
    let floor_move = if mismatch {
        V3_FLOOR_MOVE_KEY_MISMATCH
    } else {
        V3_FLOOR_MOVE
    };
    let elig = eligibility(r, z);
    let eligible = |e: usize| elig[e];
    // 3.1 element colour
    let mut colour = 0.0;
    for m in &ELEM_MODELS {
        let e = m.id;
        if !eligible(e) {
            continue;
        }
        let weight = m.w_term * z.share[e].sqrt();
        let mut hue_term = 0.0;
        let (mut hue_target, mut dh_used) = (None, 0.0);
        if let (Some(rh), Some(zh)) = (r.hue[e], z.hue[e]) {
            let rel = (r.chroma[e] / V3_HUE_RELIABLE_CHROMA).min(1.0)
                * (z.chroma[e] / V3_HUE_RELIABLE_CHROMA).min(1.0);
            let mut w_hue = m.w_hue;
            let mut window = m.window;
            if let (ELEM_SKY, Some((lo, hi))) = (e, m.window)
                && !(lo..=hi).contains(&zh)
            {
                w_hue *= 0.5;
                window = None;
            }
            let dh = wrap180(rh - zh).clamp(-m.cap_deg, m.cap_deg);
            dh_used = dh;
            let mut target = (zh + w_hue * rel * dh).rem_euclid(360.0);
            if let Some((lo, hi)) = window {
                target = target.clamp(lo, hi);
            }
            hue_target = Some(target);
            if let Some(oh) = o.hue[e] {
                hue_term =
                    (dead(wrap180(oh - target), V3_HUE_BAND_DEG) / V3_HUE_ERR_SCALE_DEG).powi(2);
            }
        }
        let ratio = (r.chroma[e] / z.chroma[e].max(1e-6)).clamp(V3_RATIO_MIN, V3_RATIO_MAX);
        let mut c_target = z.chroma[e] * ratio.powf(m.w_sat);
        if e == ELEM_SKIN {
            c_target = c_target.clamp(V3_SKIN_C_MIN, V3_SKIN_C_MAX);
        }
        let c_term = (dead(o.chroma[e] - c_target, V3_CHROMA_BAND) / V3_CHROMA_ERR_SCALE).powi(2);
        colour += weight * (hue_term + c_term);
        if let Some(d) = diag.as_deref_mut() {
            d.push(json!({ "element": ELEM_NAMES[e], "term": "colour", "photo_area_share": z.share[e],
                "ref_hue": r.hue[e], "photo0_hue": z.hue[e], "hue_delta_used": dh_used, "hue_target": hue_target,
                "ref_chroma": r.chroma[e], "photo0_chroma": z.chroma[e], "chroma_target": c_target,
                "candidate_hue": o.hue[e], "candidate_chroma": o.chroma[e], "contribution": weight * (hue_term + c_term) }));
        }
    }
    t.elem_colour = colour;
    // 3.2 neutral cast: V2's term as it is.
    t.cast = v3_cast(o, r);
    // 3.3a black floor and white ceiling
    let p2_target = p0.floor
        + floor_move
            * (reference.floor - p0.floor).clamp(-V3_FLOOR_CEIL_CLAMP_L, V3_FLOOR_CEIL_CLAMP_L);
    let p98_target = p0.ceil
        + V3_CEIL_MOVE
            * (reference.ceil - p0.ceil).clamp(-V3_FLOOR_CEIL_CLAMP_L, V3_FLOOR_CEIL_CLAMP_L);
    t.floor = (dead(out.floor - p2_target, V3_FLOOR_CEIL_BAND_L) / V3_FLOOR_CEIL_SCALE_L).powi(2);
    t.ceil = (dead(out.ceil - p98_target, V3_FLOOR_CEIL_BAND_L) / V3_FLOOR_CEIL_SCALE_L).powi(2);
    // 3.3b element exposure relative to the frame
    let mut exposure = 0.0;
    for &e in &EXPOSURE_ELEMS {
        if !eligible(e) {
            continue;
        }
        let off_ref = r.light[e] - reference.median;
        let off_0 = z.light[e] - p0.median;
        let target = off_0
            + V3_EXPOSURE_MOVE * (off_ref - off_0).clamp(-V3_EXPOSURE_CLAMP_L, V3_EXPOSURE_CLAMP_L);
        let off = o.light[e] - out.median;
        let c = exposure_weight
            * z.share[e].sqrt()
            * (dead(off - target, V3_EXPOSURE_BAND_L) / V3_EXPOSURE_SCALE_L).powi(2);
        exposure += c;
        if let Some(d) = diag.as_deref_mut() {
            d.push(json!({ "element": ELEM_NAMES[e], "term": "exposure", "offset_ref": off_ref, "offset_photo0": off_0,
                "offset_target": target, "offset_candidate": off, "contribution": c }));
        }
    }
    t.elem_exposure = exposure;
    // 3.3c subject separation
    let mut separation = 0.0;
    if r.subject_count > 0
        && r.background_count > 0
        && z.subject_count >= V3_ELEM_MIN_SAMPLES
        && z.background_count >= V3_ELEM_MIN_SAMPLES
        && o.subject_count > 0
    {
        let sep_ref = r.subject_light - r.background_light;
        let sep_0 = z.subject_light - z.background_light;
        let target = sep_0 + V3_SEP_MOVE * (sep_ref - sep_0).clamp(-V3_SEP_CLAMP_L, V3_SEP_CLAMP_L);
        let sep = o.subject_light - o.background_light;
        separation = exposure_weight * (dead(sep - target, V3_SEP_BAND_L) / V3_SEP_SCALE_L).powi(2);
        if let Some(d) = diag.as_deref_mut() {
            d.push(json!({ "term": "separation", "sep_ref": sep_ref, "sep_photo0": sep_0, "target": target,
                "candidate": sep, "contribution": separation }));
        }
    }
    t.separation = separation;
    // 3.3d exposure anchor: V2's
    t.anchor = (dead((out.median - p0.median).abs(), V2_ANCHOR_DEAD_BAND_L) / L_SCALE).powi(2);
    // 3.4 global saturation
    let mut sat = 0.0;
    let mut ratio_star = 1.0;
    if r.nonperson_share >= V3_ELEM_MIN_SHARE
        && z.nonperson_share >= V3_ELEM_MIN_SHARE
        && z.nonperson_chroma > 1e-6
    {
        let target = ((r.nonperson_chroma / z.nonperson_chroma).powf(V3_SAT_POWER))
            .clamp(V3_SAT_MIN, V3_SAT_MAX);
        ratio_star = target;
        let ratio = o.nonperson_chroma / z.nonperson_chroma;
        sat = (dead(ratio - target, V3_SAT_BAND) / V3_SAT_SCALE).powi(2);
        if let Some(d) = diag.as_deref_mut() {
            d.push(json!({ "term": "global_saturation", "target_ratio": target, "candidate_ratio": ratio, "contribution": sat }));
        }
    }
    t.global_sat = sat;
    // 3.6 (round 3b) do-no-harm: outside the eligible elements each hue sector with at least 2 % of those pixels keeps its
    // mean hue shift at 0 and its chroma ratio at the global saturation target, against photo0.
    let mut dnh = 0.0;
    if let Some(acc) = &o.dnh {
        for (sec, a) in acc.iter().enumerate() {
            if o.dnh_total == 0
                || (a.count as f64) < V3_DNH_MIN_SHARE * o.dnh_total as f64
                || a.chroma_p0 <= 1e-6
            {
                continue;
            }
            let shift = a.shift_sum / a.count as f64;
            let ratio = a.chroma_cand / a.chroma_p0;
            let share = a.count as f64 / o.n_px as f64;
            let c = V3_DNH_WEIGHT
                * share.sqrt()
                * ((dead(shift, V3_DNH_HUE_BAND_DEG) / V3_DNH_HUE_SCALE_DEG).powi(2)
                    + (dead(ratio - ratio_star, V3_DNH_RATIO_BAND) / V3_DNH_RATIO_SCALE).powi(2));
            dnh += c;
            if let Some(d) = diag.as_deref_mut() {
                d.push(json!({ "term": "do_no_harm", "sector": sec, "pixel_share": share, "mean_hue_shift": shift,
                    "chroma_ratio": ratio, "ratio_target": ratio_star, "contribution": c }));
            }
        }
    }
    t.dnh = dnh;
    if let Some(d) = diag {
        d.push(json!({ "term": "gates", "low_key_reference": low_key_ref, "high_key_photo": high_key_photo,
            "key_mismatch": mismatch, "floor_move": floor_move, "p2_target": p2_target, "p98_target": p98_target,
            "p2_candidate": out.floor, "p98_candidate": out.ceil }));
    }
}

/// The neutral-cast term over the neutral-capable elements only (round 3b): V2's dead band and scale, and the same gate.
fn v3_cast(out: &V3Stats, reference: &V3Stats) -> f64 {
    let (mut sum, mut k) = (0.0, 0usize);
    for b in 0..3 {
        if out.neutral_share[b] >= V2_MIN_NEUTRAL_SHARE_PER_BAND
            && reference.neutral_share[b] >= V2_MIN_NEUTRAL_SHARE_PER_BAND
        {
            let da = out.neutral_ab[b][0] - reference.neutral_ab[b][0];
            let db = out.neutral_ab[b][1] - reference.neutral_ab[b][1];
            sum += (dead(da.hypot(db), V2_CAST_BAND_AB) / V2_CAST_SCALE).powi(2);
            k += 1;
        }
    }
    if k > 0 { sum / k as f64 } else { 0.0 }
}

// ---------------------------------------------------------------------------------------------
// V5: a hand-written network sets the targets (`refgrade-notes/09-model-v5.md`, `refgrade-v5.ywn`)
// ---------------------------------------------------------------------------------------------

/// The elements of V5, in the order of the local grades: `built` is architecture plus interior, `skin` is MediaPipe skin.
const V5_ELEMS: [&str; 6] = ["sky", "vegetation", "water", "ground", "built", "skin"];
/// The cap (degrees) of each element's hue move: the network's `cap` constants, which also bound the local hue rotation.
const V5_CAP_DEG: [f64; 6] = [40.0, 35.0, 35.0, 30.0, 25.0, 15.0];
/// The objective's `w_term` per element (`09` section 4.2).
const V5_W_TERM: [f64; 6] = [1.5, 1.2, 1.0, 1.0, 0.6, 2.0];
/// Code value of one stop of a local exposure offset (`09` section 5).
const V5_STOP_CODE: f64 = 0.0648;
/// Local grades: bounds of the chroma scale.
const V5_CHROMA_SCALE_MIN: f64 = 0.5;
const V5_CHROMA_SCALE_MAX: f64 = 1.6;
/// A share of the frame at which an element or a sector counts as present (the network gates take over from here).
const V5_PRESENT_SHARE: f64 = 0.02;
/// Excluded from the sector inputs: elements both pictures hold at this share.
const V5_EXCLUDE_SHARE: f64 = 0.02;
/// Objective scales, dead bands and weights (`09` section 4).
const V5_FLOOR: (f64, f64, f64) = (2.0, 1.0, 1.0);
const V5_CEIL: (f64, f64, f64) = (2.0, 1.0, 1.0);
const V5_CONTRAST: (f64, f64, f64) = (0.03, 0.01, 1.0);
const V5_ANCHOR: (f64, f64, f64) = (1.0, 5.0, 4.0);
const V5_TINT: (f64, f64) = (3.0, 1.0);
const V5_TINT_BAND_WEIGHT: [f64; 3] = [1.0, 0.7, 1.0];
const V5_SAT: (f64, f64, f64) = (0.05, 0.03, 1.0);
const V5_SECTOR_HUE: (f64, f64, f64) = (8.0, 2.0, 0.8);
const V5_SECTOR_CHROMA: (f64, f64, f64) = (0.08, 0.03, 0.8);
const V5_EL_HUE: (f64, f64) = (10.0, 3.0);
const V5_EL_CHROMA: (f64, f64) = (0.10, 0.03);
const V5_EL_EXPOSURE: (f64, f64) = (3.0, 2.0);
const V5_SUBJECT: (f64, f64, f64) = (3.0, 2.0, 1.0);
/// V7: the cast is measured on pixels between these L* values (and not skin), when they cover this share of the frame.
const V7_CAST_L: (f64, f64) = (15.0, 80.0);
const V7_CAST_MIN_SHARE: f64 = 0.10;
/// V9: a group counts in the hue walls when its mean-vector chroma in photo0 is at least this; the dark-colours group (P's
/// pixels with L* below `V9_DARK_L` and C* at least `V9_DARK_C`, not skin; it needs 2 % of the frame); the memory-colour wall's
/// weight and scale.
const V9_MIN_MV_C: f64 = 2.5;
const V9_DARK_L: f64 = 35.0;
const V9_DARK_C: f64 = 2.0;
const V9_MEM: (f64, f64) = (3.0, 3.0);
/// V8: the purple band (hue, degrees), the wall's weight and scale, the approach term's reach, scale and weight, a group counts
/// when its median C* is at least this; the whites' direction wall (weight, scale); the skin drift wall (weight, scale).
const V8_PURPLE_BAND: (f64, f64) = (285.0, 345.0);
/// V10: a pixel is newly purple at C* of at least this in the candidate, and it was not purple in photo0 (outside the band, or
/// under `V10_NP_FROM_C`). Penalty `V10_NP_WEIGHT * ((share - T.purple_max_share) / V10_NP_SCALE)^2` above the limit.
const V10_NP_C: f64 = 4.0;
/// V11: the purple band in Oklab hue, degrees.
const V11_PURPLE_BAND: (f64, f64) = (300.0, 355.0);
/// V11: the purple group wall `3 * (excess / 5)^2`, `excess` in degrees of turn beyond `T.purple_rot_deg`.
const V11_TURN: (f64, f64) = (3.0, 5.0);
/// V11: the skin chroma wall `12 * ((r - max) / 0.05)^2`, for skin of at least this share of the frame.
const V11_SKIN_CHROMA: (f64, f64) = (12.0, 0.05);
const V11_SKIN_MIN_SHARE: f64 = 0.01;
/// V12: crushed black is L* under this (it was: every channel at or under 0.004 encoded, about L* 0.4).
const V12_CRUSH_L: f64 = 2.5;
/// V12: the clip term's scale (it was [`CLIP_SCALE`], 0.02).
const V12_CLIP_SCALE: f64 = 0.01;
/// V12: the content turn budget `6 * ((share - T.turn_share) / 0.01)^2`; a pixel counts from this photo0 C*.
const V12_TURN: (f64, f64) = (6.0, 0.01);
const V12_TURN_MIN_C: f64 = 8.0;
/// V12: the whites for the chroma cap and the direction: photo0 L* at least [`V7_HL_L_MIN`] and C* under this, not skin.
const V12_WHITE_C_MAX: f64 = 8.0;
/// V12: the memory wall `6 * (excess / 2)^2`.
const V12_MEM: (f64, f64) = (6.0, 2.0);
/// V11: the purple groups: the elements 0..6, clothing, the sectors, the whites and the dark colours.
const PG_CLOTH: usize = 6;
const PG_SEC: usize = 7;
const PG_HL: usize = PG_SEC + SECTORS;
const PG_DARK: usize = PG_HL + 1;
const PG_N: usize = PG_DARK + 1;
const V10_NP_FROM_C: f64 = 2.0;
const V10_NP_WEIGHT: f64 = 6.0;
const V10_NP_SCALE: f64 = 0.01;
const V8_PURPLE_WEIGHT: f64 = 3.0;
const V8_PURPLE_SCALE_DEG: f64 = 5.0;
const V8_APPROACH_REACH_DEG: f64 = 15.0;
const V8_APPROACH_WEIGHT: f64 = 1.0;
const V8_PURPLE_MIN_C: f64 = 6.0;
const V8_WHITE_DIR: (f64, f64) = (3.0, 2.0);
const V8_SKIN_DRIFT: (f64, f64) = (12.0, 3.0);
/// V7: the skin wall's weight (was 3).
const V7_GUARD_SKIN_WEIGHT: f64 = 12.0;
/// V7 highlight-neutrality wall: the whites of the photo are pixels with L* at least this and C* under that; the wall's scale
/// and weight; the hue drift of the group counts when photo0's median C* there is at least `V7_HL_MIN_C`; and the group needs
/// this share of the frame.
const V7_HL_L_MIN: f64 = 80.0;
const V7_HL_C_MAX: f64 = 10.0;
const V7_HL_SCALE: f64 = 2.0;
const V7_HL_WEIGHT: f64 = 3.0;
const V7_HL_MIN_C: f64 = 4.0;
const V7_HL_MIN_SHARE: f64 = 0.02;
/// V6 frame cast term (`10`): scale, dead band, weight (times `G.unify`).
const V6_CAST: (f64, f64, f64) = (2.0, 1.0, 1.2);
/// Guards (`09` section 4.3): hue drift limit, its band scale and weight; the element limit is `cap + 5`.
const V5_GUARD_HUE_LIMIT_DEG: f64 = 25.0;
const V5_GUARD_HUE_EXTRA_DEG: f64 = 5.0;
const V5_GUARD_HUE_SCALE_DEG: f64 = 5.0;
const V5_GUARD_HUE_WEIGHT: f64 = 2.0;
const V5_GUARD_SECTOR_MIN_SHARE: f64 = 0.04;
const V5_GUARD_SKIN_HUE: (f64, f64) = (30.0, 70.0);
const V5_GUARD_SKIN_CHROMA: (f64, f64) = (12.0, 38.0);
const V5_GUARD_SKIN_SCALE: f64 = 3.0;
const V5_GUARD_SKIN_WEIGHT: f64 = 3.0;
const V5_GUARD_SKY_FACTOR: f64 = 1.3;
const V5_GUARD_SKY_SCALE: f64 = 2.0;
const V5_GUARD_SKY_WEIGHT: f64 = 2.0;
/// Need test (`09` section 5): hue error, log chroma error and exposure error that earn a local grade.
const V5_NEED_HUE_DEG: f64 = 6.0;
const V5_NEED_LOGC: f64 = 0.14;
const V5_NEED_EXPOSURE_L: f64 = 4.0;
/// A gate above this counts as open.
const V5_GATE_OPEN: f64 = 0.5;
/// Guided filter of the soft masks: radius as a share of the short side, and epsilon on L*/100.
const V5_GF_RADIUS: f64 = 0.02;
const V5_GF_EPS: f64 = 1e-3;
/// Two golden-section passes over the local parameters: (half range, iterations).
const V5_LOCAL_PASSES: [(f64, usize); 2] = [(1.0, 7), (0.4, 6)];
/// The polish pass over the global and the needed local parameters.
const V5_POLISH: (f64, usize) = (0.25, 6);

/// One neuron: `name = act(bias + sum w * node)`.
#[derive(Debug, Clone)]
struct Neuron {
    name: String,
    relu: bool,
    bias: f64,
    terms: Vec<(f64, String)>,
}

/// The hand-written network, expanded.
#[derive(Debug, Clone)]
struct Network {
    neurons: Vec<Neuron>,
    index: BTreeMap<String, usize>,
}

/// Every input name the network may read.
fn network_input_names() -> Vec<String> {
    let mut v = Vec::new();
    for x in ["P", "R"] {
        for n in [
            "p2",
            "p50",
            "p98",
            "iqr_rel",
            "logsat",
            "subj",
            "sep",
            "cast_a",
            "cast_b",
            "conc",
            "magenta_share",
        ] {
            v.push(format!("{x}.{n}"));
        }
        for b in ["s", "m", "h"] {
            for c in ["a", "b"] {
                v.push(format!("{x}.tint_{b}_{c}"));
            }
        }
        for e in V5_ELEMS {
            for n in ["share", "L", "h", "C", "logC"] {
                v.push(format!("{x}.{e}.{n}"));
            }
        }
        for j in 0..SECTORS {
            v.push(format!("{x}.s{j}.share"));
            v.push(format!("{x}.s{j}.logC"));
        }
    }
    for e in V5_ELEMS {
        v.push(format!("D.{e}.dh"));
    }
    for j in 0..SECTORS {
        v.push(format!("D.s{j}.dh"));
    }
    v
}

fn parse_neuron(
    line: &str,
    inputs: &BTreeSet<String>,
    defined: &BTreeMap<String, usize>,
) -> R<Neuron> {
    let (name, rest) = line
        .split_once('=')
        .ok_or_else(|| format!("no '=' in: {line}"))?;
    let name = name.trim().to_owned();
    let rest = rest.trim();
    let open = rest.find('(').ok_or_else(|| format!("{name}: no '('"))?;
    let close = rest.rfind(')').ok_or_else(|| format!("{name}: no ')'"))?;
    let act = rest[..open].trim();
    let relu = match act {
        "relu" => true,
        "lin" => false,
        other => return Err(format!("{name}: unknown activation {other}").into()),
    };
    let toks: Vec<&str> = rest[open + 1..close].split_whitespace().collect();
    let bias: f64 = toks
        .first()
        .ok_or_else(|| format!("{name}: empty"))?
        .parse()
        .map_err(|_| format!("{name}: bias is not a number: {}", toks[0]))?;
    let mut terms = Vec::new();
    let mut i = 1;
    while i < toks.len() {
        let sign = match toks[i] {
            "+" => 1.0,
            "-" => -1.0,
            other => return Err(format!("{name}: expected + or -, got {other}").into()),
        };
        let t = *toks
            .get(i + 1)
            .ok_or_else(|| format!("{name}: sign without a term"))?;
        let (w, node) = match t.split_once('*') {
            Some((w, n)) => (
                w.parse::<f64>()
                    .map_err(|_| format!("{name}: weight {w} is not a number"))?,
                n,
            ),
            None => (1.0, t),
        };
        if !defined.contains_key(node) && !inputs.contains(node) {
            return Err(format!(
                "{name}: refers to {node}, which is not an input and not defined earlier"
            )
            .into());
        }
        terms.push((sign * w, node.to_owned()));
        i += 2;
    }
    Ok(Neuron {
        name,
        relu,
        bias,
        terms,
    })
}

/// Parses and expands the network file (templates, constants, `apply`). Refuses a reference to a node not defined earlier,
/// a constant missing for a template value, and a duplicate name; there is no fallback.
fn parse_network(text: &str) -> R<Network> {
    let inputs: BTreeSet<String> = network_input_names().into_iter().collect();
    let mut consts: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut templates: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
    let mut current: Option<(String, String, Vec<String>)> = None;
    let mut lines: Vec<String> = Vec::new();
    for raw in text.lines() {
        let l = raw.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some((name, var, body)) = current.as_mut() {
            if l == "end" {
                templates.insert(name.clone(), (var.clone(), std::mem::take(body)));
                current = None;
            } else {
                body.push(l.to_owned());
            }
            continue;
        }
        if let Some(rest) = l.strip_prefix("const ") {
            let mut it = rest.split_whitespace();
            let name = it.next().ok_or("const without a name")?.to_owned();
            let kv = it
                .map(|p| {
                    p.split_once('=')
                        .map(|(k, v)| (k.to_owned(), v.to_owned()))
                        .ok_or_else(|| format!("const {name}: {p} is not k=v"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            consts.insert(name, kv);
        } else if let Some(rest) = l.strip_prefix("template ") {
            let rest = rest.trim();
            let open = rest.find('(').ok_or("template without (var)")?;
            let close = rest.find(')').ok_or("template without (var)")?;
            current = Some((
                rest[..open].trim().to_owned(),
                rest[open + 1..close].trim().to_owned(),
                Vec::new(),
            ));
        } else if let Some(rest) = l.strip_prefix("apply ") {
            let mut it = rest.split_whitespace();
            let tname = it.next().ok_or("apply without a name")?;
            let (var, body) = templates
                .get(tname)
                .ok_or_else(|| format!("apply: no template {tname}"))?;
            for value in it {
                for b in body {
                    let mut s = b.replace(&format!("{{{var}}}"), value);
                    if let Some(kv) = consts.get(value) {
                        for (k, v) in kv {
                            s = s.replace(&format!("{{{k}}}"), v);
                        }
                    }
                    if let Some(p) = s.find('{') {
                        let end = s[p..].find('}').map_or(s.len(), |e| p + e + 1);
                        return Err(format!(
                            "template {tname}, value {value}: constant {} is missing (in: {})",
                            &s[p..end],
                            b
                        )
                        .into());
                    }
                    lines.push(s);
                }
            }
        } else {
            lines.push(l.to_owned());
        }
    }
    if current.is_some() {
        return Err("a template has no `end`".into());
    }
    let mut neurons: Vec<Neuron> = Vec::new();
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for l in &lines {
        let n = parse_neuron(l, &inputs, &index)?;
        if index.contains_key(&n.name) || inputs.contains(&n.name) {
            return Err(format!("duplicate name {}", n.name).into());
        }
        index.insert(n.name.clone(), neurons.len());
        neurons.push(n);
    }
    Ok(Network { neurons, index })
}

impl Network {
    /// Evaluates every neuron in file order, in f64.
    fn eval(&self, inputs: &BTreeMap<String, f64>) -> R<Vec<f64>> {
        let mut vals: Vec<f64> = Vec::with_capacity(self.neurons.len());
        for n in &self.neurons {
            let mut s = n.bias;
            for (w, node) in &n.terms {
                let v = match self.index.get(node) {
                    Some(&i) => vals[i],
                    None => *inputs
                        .get(node)
                        .ok_or_else(|| format!("input {node} was not given"))?,
                };
                s += w * v;
            }
            vals.push(if n.relu { s.max(0.0) } else { s });
        }
        Ok(vals)
    }

    /// The expanded network, every weight with its number.
    fn dump(&self) -> String {
        let mut out = String::new();
        for n in &self.neurons {
            out.push_str(&format!(
                "{} = {}( {:+}",
                n.name,
                if n.relu { "relu" } else { "lin" },
                n.bias
            ));
            for (w, node) in &n.terms {
                out.push_str(&format!(" {:+}*{}", w, node));
            }
            out.push_str(" )\n");
        }
        out
    }
}

/// The values of the network for one pair, by name.
#[derive(Default)]
struct NetOut {
    names: Vec<String>,
    vals: Vec<f64>,
    map: BTreeMap<String, f64>,
}

impl NetOut {
    fn new(net: &Network, inputs: &BTreeMap<String, f64>) -> R<Self> {
        let vals = net.eval(inputs)?;
        let names: Vec<String> = net.neurons.iter().map(|n| n.name.clone()).collect();
        let map = names.iter().cloned().zip(vals.iter().copied()).collect();
        Ok(Self { names, vals, map })
    }
    fn get(&self, name: &str) -> f64 {
        self.map.get(name).copied().unwrap_or(0.0)
    }
    /// The outputs (`T.*`, `G.*`) and the gate neurons the notes ask for, as JSON.
    fn outputs_json(&self) -> Value {
        let mut outs = serde_json::Map::new();
        let mut gates = serde_json::Map::new();
        for (n, v) in self.names.iter().zip(&self.vals) {
            if n.starts_with("T.") || n.starts_with("G.") {
                outs.insert(n.clone(), json!(v));
            }
            if n.starts_with("G.")
                || n.ends_with(".matched")
                || n.ends_with(".reliable")
                || n == "glob.ka"
                || n == "glob.kb"
                || n.ends_with(".e_gate")
                || n == "sky.m0"
                || n == "sky.litP"
                || n == "sky.litR"
            {
                gates.insert(n.clone(), json!(v));
            }
        }
        json!({ "outputs": outs, "gates": gates })
    }
}

/// The self-check: two hand-made input vectors. Prints both and returns whether every check passed.
fn network_check(net: &Network) -> R<bool> {
    let mut p: BTreeMap<String, f64> = network_input_names()
        .into_iter()
        .map(|n| (n, 0.0))
        .collect();
    // A photo: a mid-key landscape with sky, vegetation and ground, and a little warm tint in the shadows.
    let set = |m: &mut BTreeMap<String, f64>, k: &str, v: f64| {
        m.insert(k.to_owned(), v);
    };
    for x in ["P", "R"] {
        for (k, v) in [
            ("p2", 6.0),
            ("p50", 52.0),
            ("p98", 94.0),
            ("iqr_rel", 0.42),
            ("logsat", 15f64.ln()),
            ("subj", 1.0),
            ("sep", 6.0),
            ("tint_s_a", 1.0),
            ("tint_s_b", 2.0),
            ("tint_m_a", 0.5),
            ("tint_m_b", 1.0),
            ("tint_h_a", -0.5),
            ("tint_h_b", 1.5),
        ] {
            set(&mut p, &format!("{x}.{k}"), v);
        }
        for (e, share, l, h, c) in [
            ("sky", 0.30, 74.0, 255.0, 22.0),
            ("vegetation", 0.20, 42.0, 120.0, 26.0),
            ("ground", 0.10, 45.0, 70.0, 14.0),
        ] {
            set(&mut p, &format!("{x}.{e}.share"), share);
            set(&mut p, &format!("{x}.{e}.L"), l);
            set(&mut p, &format!("{x}.{e}.h"), h);
            set(&mut p, &format!("{x}.{e}.C"), c);
            set(&mut p, &format!("{x}.{e}.logC"), f64::max(c, 1.0).ln());
        }
        for (j, share, logc) in [(0, 0.08, 2.4), (3, 0.07, 2.6)] {
            set(&mut p, &format!("{x}.s{j}.share"), share);
            set(&mut p, &format!("{x}.s{j}.logC"), logc);
        }
    }
    let v6 = net.index.contains_key("G.unify");
    let (w_sky, k_sat) = if v6 { (0.85, 0.85) } else { (0.7, 0.6) };
    let v7 = net.index.contains_key("T.hl_cmax");
    let wall_open = if v7 { 45.0 } else { 55.0 };
    let r_same = p.clone();
    let out1 = NetOut::new(net, &p)?;
    let mut ok = true;
    println!("== check 1: R identical to P");
    let expect_eq = |name: &str, want: f64, out: &NetOut, ok: &mut bool| {
        let got = out.get(name);
        let pass = (got - want).abs() < 1e-9;
        *ok &= pass;
        println!(
            "   {name:<16} = {got:>10.5}  (P's own value {want:>10.5})  {}",
            if pass { "ok" } else { "FAIL" }
        );
    };
    for k in [
        "p2", "p98", "iqr_rel", "p50", "logsat", "tint_s_a", "tint_s_b", "tint_m_a", "tint_m_b",
        "tint_h_a", "tint_h_b", "sep",
    ] {
        let tname = match k {
            "p2" | "p98" | "iqr_rel" | "p50" | "logsat" | "sep" => format!("T.{k}"),
            _ => format!("T.{k}"),
        };
        expect_eq(&tname, p[&format!("P.{k}")], &out1, &mut ok);
    }
    for e in ["sky", "vegetation", "ground"] {
        expect_eq(&format!("T.{e}.h"), p[&format!("P.{e}.h")], &out1, &mut ok);
        expect_eq(
            &format!("T.{e}.logC"),
            p[&format!("P.{e}.logC")],
            &out1,
            &mut ok,
        );
        expect_eq(
            &format!("T.{e}.off"),
            p[&format!("P.{e}.L")] - p["P.p50"],
            &out1,
            &mut ok,
        );
    }
    for j in [0, 3] {
        expect_eq(&format!("T.s{j}.dh"), 0.0, &out1, &mut ok);
        expect_eq(&format!("T.s{j}.dlogC"), 0.0, &out1, &mut ok);
    }
    println!(
        "   gates: key_mismatch {:.2}, sky.matched {:.2}, sky.hue {:.2}, veg.matched {:.2}, water.matched {:.2}, subj {:.2}, s0 {:.2}, s3 {:.2}",
        out1.get("G.key_mismatch"),
        out1.get("G.sky.matched"),
        out1.get("G.sky.hue"),
        out1.get("G.vegetation.matched"),
        out1.get("G.water.matched"),
        out1.get("G.subj"),
        out1.get("G.s0"),
        out1.get("G.s3")
    );
    // The second vector: R has a teal sky (hue 255 -> 205), a darker floor (p2 6 -> 0) and more saturation (x1.35), and a
    // brighter sky (74 -> 84).
    let mut r = r_same.clone();
    set(&mut r, "R.p2", 0.0);
    set(&mut r, "R.logsat", 15f64.ln() + 1.35f64.ln());
    set(&mut r, "R.sky.h", 205.0);
    set(&mut r, "R.sky.L", 84.0);
    let dh = ((205.0f64 - 255.0 + 180.0).rem_euclid(360.0)) - 180.0;
    set(&mut r, "D.sky.dh", dh);
    let out2 = NetOut::new(net, &r)?;
    println!(
        "== check 2: R with a teal sky (255 -> 205 deg), a darker floor (p2 6 -> 0) and 1.35x saturation, a brighter sky (74 -> 84)"
    );
    let show = |name: &str, base: f64, out: &NetOut, want: &str, pass: bool, ok: &mut bool| {
        let got = out.get(name);
        *ok &= pass;
        println!(
            "   {name:<16} = {got:>10.5}  (P {base:>10.5}, expected {want})  {}",
            if pass { "ok" } else { "FAIL" }
        );
    };
    let t = |n: &str| out2.get(n);
    show(
        "T.p2",
        6.0,
        &out2,
        "below 6, by 0.8 x 6 = 4.8",
        (t("T.p2") - (6.0 - 4.8)).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.sky.h",
        255.0,
        &out2,
        &format!(
            "255 - {w_sky} x 40 = {} (move capped at 40)",
            255.0 - w_sky * 40.0
        ),
        (t("T.sky.h") - (255.0 - w_sky * 40.0)).abs() < 1e-6,
        &mut ok,
    );
    show(
        "T.logsat",
        15f64.ln(),
        &out2,
        &format!("up by {k_sat} x ln 1.35"),
        (t("T.logsat") - (15f64.ln() + k_sat * 1.35f64.ln())).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.sky.off",
        74.0 - 52.0,
        &out2,
        "22 + 0.5 x 10 = 27",
        (t("T.sky.off") - 27.0).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.vegetation.h",
        120.0,
        &out2,
        "unchanged (no difference)",
        (t("T.vegetation.h") - 120.0).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.p98",
        94.0,
        &out2,
        "unchanged",
        (t("T.p98") - 94.0).abs() < 1e-9,
        &mut ok,
    );
    // A third vector: a low-key reference (median 20) on the same high-key photo: the key-mismatch gate must open, the floor
    // moves only 0.4 of the difference and the exposure targets and the subject separation switch off.
    let mut r3 = r_same.clone();
    set(&mut r3, "R.p50", 20.0);
    set(&mut r3, "R.p2", 0.0);
    set(&mut r3, "R.sep", 14.0);
    let out3 = NetOut::new(net, &r3)?;
    println!("== check 3: a low-key reference (median 20), p2 6 -> 0, subject separation 6 -> 14");
    show(
        "G.key_mismatch",
        0.0,
        &out3,
        "1",
        (out3.get("G.key_mismatch") - 1.0).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.p2",
        6.0,
        &out3,
        "6 - 0.4 x 6 = 3.6",
        (out3.get("T.p2") - 3.6).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.sky.off",
        22.0,
        &out3,
        "22 (no exposure move)",
        (out3.get("T.sky.off") - 22.0).abs() < 1e-9,
        &mut ok,
    );
    show(
        "T.sep",
        6.0,
        &out3,
        "6 (gate closed)",
        (out3.get("T.sep") - 6.0).abs() < 1e-9,
        &mut ok,
    );
    if v6 {
        // A fourth vector: a one-colour amber reference against a neutral photo: unify opens, the frame cast target moves up
        // by 0.8 of the (clipped) difference and the hue-drift wall opens to 55.
        let mut p4 = r_same.clone();
        set(&mut p4, "P.cast_a", 0.0);
        set(&mut p4, "P.cast_b", 1.0);
        set(&mut p4, "P.conc", 0.1);
        set(&mut p4, "R.cast_a", 6.0);
        set(&mut p4, "R.cast_b", 28.0);
        set(&mut p4, "R.conc", 0.9);
        let out4 = NetOut::new(net, &p4)?;
        println!(
            "== check 4 (v6): a one-colour amber reference (cast 6, 28; conc 0.9) against a neutral photo (cast 0, 1; conc 0.1)"
        );
        show(
            "G.unify",
            0.0,
            &out4,
            "1",
            (out4.get("G.unify") - 1.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.cast_b",
            1.0,
            &out4,
            "1 + 0.8 x 15 = 13",
            (out4.get("T.cast_b") - 13.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.cast_a",
            0.0,
            &out4,
            "0.8 x 6 = 4.8",
            (out4.get("T.cast_a") - 4.8).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.wall_deg",
            25.0,
            &out4,
            &format!("{wall_open}"),
            (out4.get("T.wall_deg") - wall_open).abs() < 1e-9,
            &mut ok,
        );
        // Identical pictures and a one-colour frame in both: unify stays shut and the wall stays at 25.
        let mut p5 = r_same.clone();
        for x in ["P", "R"] {
            set(&mut p5, &format!("{x}.cast_a"), 6.0);
            set(&mut p5, &format!("{x}.cast_b"), 28.0);
            set(&mut p5, &format!("{x}.conc"), 0.9);
        }
        let out5 = NetOut::new(net, &p5)?;
        println!("== check 5 (v6): both pictures one-colour amber (nothing to unify)");
        show(
            "G.unify",
            0.0,
            &out5,
            "0",
            out5.get("G.unify").abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.wall_deg",
            25.0,
            &out5,
            "25",
            (out5.get("T.wall_deg") - 25.0).abs() < 1e-9,
            &mut ok,
        );
        // A black studio backdrop for a sky (median L* 10) in the reference: the dark-sky rule shuts the sky's gate.
        let mut p6 = r_same.clone();
        set(&mut p6, "R.sky.L", 10.0);
        let out6 = NetOut::new(net, &p6)?;
        println!("== check 6 (v6): the reference's \"sky\" has median L* 10 (a black backdrop)");
        show(
            "G.sky.matched",
            1.0,
            &out6,
            "0 (dark-sky rule)",
            out6.get("G.sky.matched").abs() < 1e-9,
            &mut ok,
        );
    }
    if v7 {
        // A seventh vector: a reference with a highlight tint (a*, b*) = (2, 9): the most C* a white may take is 5 + 0.8 x 11.
        let mut p7 = r_same.clone();
        set(&mut p7, "R.tint_h_a", 2.0);
        set(&mut p7, "R.tint_h_b", 9.0);
        let out7 = NetOut::new(net, &p7)?;
        println!("== check 7 (v7): a reference whose highlights lean (a*, b*) = (2, 9)");
        show(
            "T.hl_cmax",
            5.0,
            &out7,
            "5 + 0.8 x 11 = 13.8",
            (out7.get("T.hl_cmax") - 13.8).abs() < 1e-9,
            &mut ok,
        );
        let mut p8 = r_same.clone();
        set(&mut p8, "R.tint_h_a", -20.0);
        set(&mut p8, "R.tint_h_b", -20.0);
        let out8 = NetOut::new(net, &p8)?;
        show(
            "T.hl_cmax",
            5.0,
            &out8,
            "14 (capped)",
            (out8.get("T.hl_cmax") - 14.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.hl_cmax",
            5.0,
            &NetOut::new(net, &r_same)?,
            "5 + 0.8 x (0.5 + 1.5) = 6.6 (the reference's own highlight tint)",
            (NetOut::new(net, &r_same)?.get("T.hl_cmax") - 6.6).abs() < 1e-9,
            &mut ok,
        );
    }
    if net.index.contains_key("T.purple_ok") {
        // The v8 vectors: purple is allowed when the reference holds a tenth of its frame in magenta, not when it holds none, and
        // the whites take 0.8 of a strong highlight tint within the largest tolerance.
        let mut p9 = r_same.clone();
        set(&mut p9, "R.magenta_share", 0.10);
        let out9 = NetOut::new(net, &p9)?;
        println!("== check 8 (v8): a reference with magenta_share 0.10");
        show(
            "T.purple_ok",
            0.0,
            &out9,
            "1",
            (out9.get("T.purple_ok") - 1.0).abs() < 1e-9,
            &mut ok,
        );
        let out10 = NetOut::new(net, &r_same)?;
        println!("== check 9 (v8): a reference with magenta_share 0.0");
        show(
            "T.purple_ok",
            0.0,
            &out10,
            "0",
            out10.get("T.purple_ok").abs() < 1e-9,
            &mut ok,
        );
        let mut p11 = r_same.clone();
        set(&mut p11, "R.tint_h_a", -8.0);
        set(&mut p11, "R.tint_h_b", 4.0);
        let out11 = NetOut::new(net, &p11)?;
        println!("== check 10 (v8): a reference with highlight tint (a*, b*) = (-8, 4)");
        show(
            "T.white_a",
            0.0,
            &out11,
            "-6.4",
            (out11.get("T.white_a") + 6.4).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.white_b",
            0.0,
            &out11,
            "3.2",
            (out11.get("T.white_b") - 3.2).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.white_tol",
            0.0,
            &out11,
            "6",
            (out11.get("T.white_tol") - 6.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.skin_drift",
            0.0,
            &out11,
            "6 (no matched skin)",
            (out11.get("T.skin_drift") - 6.0).abs() < 1e-9,
            &mut ok,
        );
    }
    if net.index.contains_key("T.purple_max_share") {
        // The v10 vectors: one pixel in a hundred may turn purple, up to a quarter where the reference holds magenta.
        let none = NetOut::new(net, &r_same)?;
        println!("== check 13 (v10): a reference with magenta_share 0");
        show(
            "T.purple_max_share",
            0.0,
            &none,
            "0.01",
            (none.get("T.purple_max_share") - 0.01).abs() < 1e-9,
            &mut ok,
        );
        let mut p14 = r_same.clone();
        set(&mut p14, "R.magenta_share", 0.10);
        let some = NetOut::new(net, &p14)?;
        println!("== check 14 (v10): a reference with magenta_share 0.10");
        show(
            "T.purple_max_share",
            0.0,
            &some,
            "0.25",
            (some.get("T.purple_max_share") - 0.25).abs() < 1e-9,
            &mut ok,
        );
    }
    if net.index.contains_key("T.purple_rot_deg") {
        // The v11 vectors: a turn of 10 degrees into the purple band, 45 where the reference holds purple; skin chroma may rise
        // 0.12 (log), 0.30 where the reference's skin is matched.
        let none = NetOut::new(net, &r_same)?;
        println!("== check 15 (v11): a reference with magenta_share 0");
        show(
            "T.purple_rot_deg",
            0.0,
            &none,
            "10",
            (none.get("T.purple_rot_deg") - 10.0).abs() < 1e-9,
            &mut ok,
        );
        let mut p16 = r_same.clone();
        set(&mut p16, "R.magenta_share", 0.10);
        let some = NetOut::new(net, &p16)?;
        println!("== check 16 (v11): a reference with magenta_share 0.10");
        show(
            "T.purple_rot_deg",
            0.0,
            &some,
            "45",
            (some.get("T.purple_rot_deg") - 45.0).abs() < 1e-9,
            &mut ok,
        );
        let mut p17 = r_same.clone();
        set(&mut p17, "P.skin.share", 0.30);
        set(&mut p17, "R.skin.share", 0.30);
        let matched = NetOut::new(net, &p17)?;
        println!("== check 17 (v11): the skin matched (both pictures hold 30 %)");
        show(
            "T.skin_dlogC_max",
            0.0,
            &matched,
            "0.30",
            (matched.get("T.skin_dlogC_max") - 0.30).abs() < 1e-9,
            &mut ok,
        );
        let mut p18 = r_same.clone();
        set(&mut p18, "P.skin.share", 0.30);
        set(&mut p18, "R.skin.share", 0.0);
        let unmatched = NetOut::new(net, &p18)?;
        println!("== check 18 (v11): the skin unmatched (R.skin.share 0)");
        show(
            "T.skin_dlogC_max",
            0.0,
            &unmatched,
            "0.12",
            (unmatched.get("T.skin_dlogC_max") - 0.12).abs() < 1e-9,
            &mut ok,
        );
    }
    if net.index.contains_key("T.turn_deg") {
        // The v12 vectors: a strong colour may turn 20 degrees, 30 when unify is open; 3 % of the frame may turn further; a
        // reference may license 2 points of clipping beyond the photo's own.
        let shut = NetOut::new(net, &r_same)?;
        println!("== check 19 (v12): unify shut");
        show(
            "T.turn_deg",
            0.0,
            &shut,
            "20",
            (shut.get("T.turn_deg") - 20.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.turn_share",
            0.0,
            &shut,
            "0.03",
            (shut.get("T.turn_share") - 0.03).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.clip_extra",
            0.0,
            &shut,
            "0.02",
            (shut.get("T.clip_extra") - 0.02).abs() < 1e-9,
            &mut ok,
        );
        let mut p20 = r_same.clone();
        set(&mut p20, "P.cast_a", 0.0);
        set(&mut p20, "P.cast_b", 1.0);
        set(&mut p20, "P.conc", 0.1);
        set(&mut p20, "R.cast_a", 6.0);
        set(&mut p20, "R.cast_b", 28.0);
        set(&mut p20, "R.conc", 0.9);
        let open = NetOut::new(net, &p20)?;
        println!(
            "== check 20 (v12): unify open (a one-colour amber reference against a neutral photo)"
        );
        show(
            "G.unify",
            0.0,
            &open,
            "1",
            (open.get("G.unify") - 1.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.turn_deg",
            0.0,
            &open,
            "30",
            (open.get("T.turn_deg") - 30.0).abs() < 1e-9,
            &mut ok,
        );
    }
    if net.index.contains_key("T.turn_cap") {
        // The v13 vectors: a weak colour may turn up to 45 degrees, whatever unify says; from C* 25 up the v12 limit holds.
        let shut = NetOut::new(net, &r_same)?;
        let mut p20 = r_same.clone();
        set(&mut p20, "P.cast_a", 0.0);
        set(&mut p20, "P.cast_b", 1.0);
        set(&mut p20, "P.conc", 0.1);
        set(&mut p20, "R.cast_a", 6.0);
        set(&mut p20, "R.cast_b", 28.0);
        set(&mut p20, "R.conc", 0.9);
        let open = NetOut::new(net, &p20)?;
        println!("== check 21 (v13): T.turn_cref and T.turn_cap, unify shut and open");
        for (out, unify) in [(&shut, 0.0), (&open, 1.0)] {
            show(
                "G.unify",
                0.0,
                out,
                if unify == 0.0 { "0" } else { "1" },
                (out.get("G.unify") - unify).abs() < 1e-9,
                &mut ok,
            );
            show(
                "T.turn_cref",
                0.0,
                out,
                "25",
                (out.get("T.turn_cref") - 25.0).abs() < 1e-9,
                &mut ok,
            );
            show(
                "T.turn_cap",
                0.0,
                out,
                "45",
                (out.get("T.turn_cap") - 45.0).abs() < 1e-9,
                &mut ok,
            );
        }
    }
    if net.index.contains_key("T.mem.sky_deg") {
        // The v9 vectors: the sky and foliage limits depend on whether the reference matches them.
        let out_m = NetOut::new(net, &r_same)?;
        let mut pu = r_same.clone();
        set(&mut pu, "R.sky.share", 0.0);
        set(&mut pu, "R.vegetation.share", 0.0);
        let out_u = NetOut::new(net, &pu)?;
        println!("== check 11 (v9): the sky unmatched (P.sky.share 0.3, R.sky.share 0)");
        show(
            "T.mem.sky_deg",
            10.0,
            &out_u,
            "10",
            (out_u.get("T.mem.sky_deg") - 10.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.mem.vegetation_deg",
            12.0,
            &out_u,
            "12",
            (out_u.get("T.mem.vegetation_deg") - 12.0).abs() < 1e-9,
            &mut ok,
        );
        println!("== check 12 (v9): the sky matched (both pictures hold 30 %)");
        show(
            "T.mem.sky_deg",
            10.0,
            &out_m,
            "45",
            (out_m.get("T.mem.sky_deg") - 45.0).abs() < 1e-9,
            &mut ok,
        );
        show(
            "T.mem.vegetation_deg",
            12.0,
            &out_m,
            "40",
            (out_m.get("T.mem.vegetation_deg") - 40.0).abs() < 1e-9,
            &mut ok,
        );
    }
    println!(
        "{}",
        if ok {
            "network check: all passed"
        } else {
            "network check: FAILED"
        }
    );
    Ok(ok)
}

// ---- V5 measurements ------------------------------------------------------------------------

/// Whether pixel `i` belongs to V5 element `k` (see [`V5_ELEMS`]).
fn in_elem(scene: &Scene, i: usize, k: usize) -> bool {
    let e = usize::from(scene.elem[i]);
    match k {
        0 => e == ELEM_SKY,
        1 => e == ELEM_VEGETATION,
        2 => e == ELEM_WATER,
        3 => e == ELEM_GROUND,
        4 => e == ELEM_ARCHITECTURE || e == ELEM_INTERIOR,
        _ => scene.skin[i],
    }
}

/// One element's statistics.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
struct ElM {
    share: f64,
    l: f64,
    h: f64,
    c: f64,
    logc: f64,
    n: usize,
    has_hue: bool,
    /// V9: the mean (a*, b*) of every member pixel: its angle is the group's hue and its length the group's chroma in the walls.
    mv_a: f64,
    mv_b: f64,
}

impl ElM {
    /// The mean-vector hue (degrees, 0..360) and chroma.
    fn mv(&self) -> (f64, f64) {
        (
            circ_mean_deg(self.mv_a, self.mv_b),
            self.mv_a.hypot(self.mv_b),
        )
    }
}

/// One hue sector's statistics.
#[derive(Clone, Debug, Default)]
struct SecM {
    share: f64,
    logc: f64,
    hue: f64,
    n: usize,
}

/// What the network and the objective read from a picture.
#[derive(Clone, Debug, Default)]
struct Meas {
    p2: f64,
    p50: f64,
    p98: f64,
    iqr_rel: f64,
    logsat: f64,
    /// Mean (a*, b*) of the near-neutral, neutral-capable pixels per L* tercile (shadows, mids, highlights).
    tint: [[f64; 2]; 3],
    el: [ElM; 6],
    subj: bool,
    sep: f64,
    sec: [SecM; SECTORS],
    hi: f64,
    lo: f64,
    /// V12: the share of the frame with L* under [`V12_CRUSH_L`].
    lo25: f64,
    /// V6: the mean a* and b* over all pixels, and `conc`: their length over the mean C* (0: colours cancel, 1: one hue).
    cast_a: f64,
    cast_b: f64,
    conc: f64,
    /// V8: the share of the frame with C* > 8 and hue 285..345 degrees.
    magenta_share: f64,
    /// V8: the clothing group, person minus skin.
    clothing: ElM,
}

fn circ_mean_deg(sx: f64, sy: f64) -> f64 {
    sy.atan2(sx).to_degrees().rem_euclid(360.0)
}

/// Measures a picture (`09` section 2). `excl` are the elements left out of the sectors (skin always is).
/// `v7_cast`: measure the frame cast only on pixels with 15 <= L* <= 80 that are not skin (all non-skin pixels if those cover
/// under 10 % of the frame), as network v7 wants; otherwise on every pixel.
fn measure(px: &[Px], scene: &Scene, excl: &[bool; 6], v7_cast: bool, v11: bool) -> Meas {
    let n = px.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| px[a].lab[0].total_cmp(&px[b].lab[0]).then(a.cmp(&b)));
    let l_at = |q: f64| {
        let pos = q / 100.0 * (n - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        let (a, b) = (px[idx[lo]].lab[0], px[idx[hi]].lab[0]);
        a + (b - a) * (pos - lo as f64)
    };
    let (p2, p50, p98) = (l_at(2.0), l_at(50.0), l_at(98.0));
    let iqr_rel = (l_at(75.0) - l_at(25.0)) / (p98 - p2).max(1.0);
    let mut band = vec![0u8; n];
    for (rank, &i) in idx.iter().enumerate() {
        band[i] = ((rank * 3 / n).min(2)) as u8;
    }
    let mut el_l: [Vec<f64>; 6] = Default::default();
    let mut el_c: [Vec<f64>; 6] = Default::default();
    let mut el_v = [[0.0f64; 2]; 6];
    let mut el_hn = [0usize; 6];
    let mut el_ab = [[0.0f64; 2]; 6];
    let (mut pool_a, mut pool_b): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut capable: [Vec<(f64, f64, f64)>; 3] = Default::default();
    let (mut subj, mut bg): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut sec_c: [Vec<f64>; SECTORS] = Default::default();
    let mut sec_v = [[0.0f64; 2]; SECTORS];
    let (mut hi, mut lo, mut lo25) = (0usize, 0usize, 0usize);
    for (i, p) in px.iter().enumerate() {
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        let e = usize::from(scene.elem[i]);
        for k in 0..6 {
            if in_elem(scene, i, k) {
                el_ab[k][0] += a;
                el_ab[k][1] += b;
                el_l[k].push(p.lab[0]);
                el_c[k].push(c);
                if c > V3_HUE_MIN_CHROMA {
                    el_v[k][0] += a / c;
                    el_v[k][1] += b / c;
                    el_hn[k] += 1;
                }
            }
        }
        if matches!(e, 2 | 3 | 4 | 5 | 6 | 8 | 9) {
            pool_a.push(c);
        }
        if e != ELEM_SKY && e != ELEM_PERSON {
            pool_b.push(c);
        }
        if matches!(e, 0 | 4 | 5 | 6 | 8 | 9) || (e == ELEM_PERSON && !scene.skin[i]) {
            capable[usize::from(band[i])].push((c, a, b));
        }
        if scene.subject[i] {
            subj.push(p.lab[0]);
        } else if e != 0 {
            bg.push(p.lab[0]);
        }
        if c > V3_HUE_MIN_CHROMA
            && !scene.skin[i]
            && !(0..5).any(|k| excl[k] && in_elem(scene, i, k))
        {
            let h = b.atan2(a).to_degrees();
            let s = ((((h + 30.0).rem_euclid(360.0)) / 60.0) as usize) % SECTORS;
            sec_c[s].push(c);
            sec_v[s][0] += a / c;
            sec_v[s][1] += b / c;
        }
        hi += usize::from(p.hi);
        lo += usize::from(p.lo);
        lo25 += usize::from(p.lo25);
    }
    let nf = n as f64;
    let cast_set: Vec<usize> = if v7_cast {
        let mid: Vec<usize> = (0..n)
            .filter(|&i| !scene.skin[i] && (V7_CAST_L.0..=V7_CAST_L.1).contains(&px[i].lab[0]))
            .collect();
        if mid.len() as f64 >= V7_CAST_MIN_SHARE * nf {
            mid
        } else {
            (0..n).filter(|&i| !scene.skin[i]).collect()
        }
    } else {
        (0..n).collect()
    };
    let cast_set = if cast_set.is_empty() {
        (0..n).collect()
    } else {
        cast_set
    };
    let (sum_a, sum_b, sum_c) = cast_set.iter().fold((0.0, 0.0, 0.0), |s, &i| {
        let p = &px[i];
        (
            s.0 + p.lab[1],
            s.1 + p.lab[2],
            s.2 + p.lab[1].hypot(p.lab[2]),
        )
    });
    let cn = cast_set.len() as f64;
    let (cast_a, cast_b) = (sum_a / cn, sum_b / cn);
    let conc = cast_a.hypot(cast_b) / (sum_c / cn).max(1.0);
    let mut m = Meas {
        cast_a,
        cast_b,
        conc,
        p2,
        p50,
        p98,
        iqr_rel,
        hi: hi as f64 / nf,
        lo: lo as f64 / nf,
        lo25: lo25 as f64 / nf,
        ..Meas::default()
    };
    let pool = if pool_a.len() as f64 / nf >= 0.01 {
        &mut pool_a
    } else {
        &mut pool_b
    };
    m.logsat = median_of(pool).max(1.0).ln();
    for b in 0..3 {
        let mut cs: Vec<f64> = capable[b].iter().map(|p| p.0).collect();
        cs.sort_unstable_by(f64::total_cmp);
        let floor_c = cs
            .get((cs.len() as f64 * V2_NEUTRAL_MIN_FRACTION) as usize)
            .copied()
            .unwrap_or(0.0);
        let limit = V2_NEUTRAL_CHROMA_MAX.max(floor_c);
        let (mut sa, mut sb, mut k) = (0.0, 0.0, 0usize);
        for &(c, a, bb) in &capable[b] {
            if c <= limit {
                sa += a;
                sb += bb;
                k += 1;
            }
        }
        if k > 0 {
            m.tint[b] = [sa / k as f64, sb / k as f64];
        }
    }
    // V8: the magenta share and the clothing group (person minus skin).
    let mut mag = 0usize;
    let (mut cl_l, mut cl_c) = (Vec::new(), Vec::new());
    let (mut cl_x, mut cl_y, mut cl_hn) = (0.0, 0.0, 0usize);
    let (mut cl_sa, mut cl_sb) = (0.0, 0.0);
    for (i, p) in px.iter().enumerate() {
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        if c > 8.0 {
            let (band, h) = if v11 {
                (V11_PURPLE_BAND, oklab_hue(p.ok))
            } else {
                (V8_PURPLE_BAND, b.atan2(a).to_degrees().rem_euclid(360.0))
            };
            if (band.0..=band.1).contains(&h) {
                mag += 1;
            }
        }
        if usize::from(scene.elem[i]) == ELEM_PERSON && !scene.skin[i] {
            cl_sa += a;
            cl_sb += b;
            cl_l.push(p.lab[0]);
            cl_c.push(c);
            if c > V3_HUE_MIN_CHROMA {
                cl_x += a / c;
                cl_y += b / c;
                cl_hn += 1;
            }
        }
    }
    m.magenta_share = mag as f64 / nf;
    if !cl_l.is_empty() {
        let c = median_of(&mut cl_c);
        m.clothing = ElM {
            share: cl_l.len() as f64 / nf,
            l: median_of(&mut cl_l),
            h: if cl_hn > 0 {
                circ_mean_deg(cl_x, cl_y)
            } else {
                0.0
            },
            c,
            logc: c.max(1.0).ln(),
            n: cl_hn,
            has_hue: cl_hn > 0,
            mv_a: cl_sa / cl_l.len() as f64,
            mv_b: cl_sb / cl_l.len() as f64,
        };
    }
    for k in 0..6 {
        let cnt = el_l[k].len();
        if cnt == 0 {
            continue;
        }
        let c = median_of(&mut el_c[k]);
        m.el[k] = ElM {
            share: cnt as f64 / nf,
            l: median_of(&mut el_l[k]),
            h: if el_hn[k] > 0 {
                circ_mean_deg(el_v[k][0], el_v[k][1])
            } else {
                0.0
            },
            c,
            logc: c.max(1.0).ln(),
            n: cnt,
            has_hue: el_hn[k] > 0,
            mv_a: el_ab[k][0] / cnt as f64,
            mv_b: el_ab[k][1] / cnt as f64,
        };
    }
    let enough = |v: &Vec<f64>| v.len() as f64 >= (0.01 * nf).max(30.0);
    m.subj = enough(&subj) && enough(&bg);
    if m.subj {
        m.sep = median_of(&mut subj) - median_of(&mut bg);
    }
    for j in 0..SECTORS {
        let cnt = sec_c[j].len();
        if cnt > 0 {
            m.sec[j] = SecM {
                share: cnt as f64 / nf,
                logc: median_of(&mut sec_c[j]).max(1.0).ln(),
                hue: circ_mean_deg(sec_v[j][0], sec_v[j][1]),
                n: cnt,
            };
        }
    }
    m
}

/// The network's input vector from the photo's and the reference's measurements.
fn network_inputs(p: &Meas, r: &Meas) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    for (x, q) in [("P", p), ("R", r)] {
        m.insert(format!("{x}.p2"), q.p2);
        m.insert(format!("{x}.p50"), q.p50);
        m.insert(format!("{x}.p98"), q.p98);
        m.insert(format!("{x}.iqr_rel"), q.iqr_rel);
        m.insert(format!("{x}.logsat"), q.logsat);
        m.insert(format!("{x}.subj"), if q.subj { 1.0 } else { 0.0 });
        m.insert(format!("{x}.sep"), q.sep);
        m.insert(format!("{x}.cast_a"), q.cast_a);
        m.insert(format!("{x}.cast_b"), q.cast_b);
        m.insert(format!("{x}.conc"), q.conc);
        m.insert(format!("{x}.magenta_share"), q.magenta_share);
        for (b, name) in ["s", "m", "h"].iter().enumerate() {
            m.insert(format!("{x}.tint_{name}_a"), q.tint[b][0]);
            m.insert(format!("{x}.tint_{name}_b"), q.tint[b][1]);
        }
        for (k, e) in V5_ELEMS.iter().enumerate() {
            let em = &q.el[k];
            m.insert(format!("{x}.{e}.share"), em.share);
            m.insert(format!("{x}.{e}.L"), em.l);
            m.insert(format!("{x}.{e}.h"), em.h);
            m.insert(format!("{x}.{e}.C"), em.c);
            m.insert(format!("{x}.{e}.logC"), em.logc);
        }
        for j in 0..SECTORS {
            m.insert(format!("{x}.s{j}.share"), q.sec[j].share);
            m.insert(format!("{x}.s{j}.logC"), q.sec[j].logc);
        }
    }
    for (k, e) in V5_ELEMS.iter().enumerate() {
        let d = if p.el[k].has_hue && r.el[k].has_hue {
            wrap_deg(r.el[k].h - p.el[k].h)
        } else {
            0.0
        };
        m.insert(format!("D.{e}.dh"), d);
    }
    for j in 0..SECTORS {
        let d = if p.sec[j].n > 0 && r.sec[j].n > 0 {
            wrap_deg(r.sec[j].hue - p.sec[j].hue)
        } else {
            0.0
        };
        m.insert(format!("D.s{j}.dh"), d);
    }
    m
}

/// Everything fixed for one pair: the photo's and the reference's measurements, the network's answer, and the photo's
/// pixels as they are ungraded.
#[derive(Default)]
struct V5Ctx {
    excl: [bool; 6],
    p: Meas,
    r: Meas,
    inputs: BTreeMap<String, f64>,
    out: NetOut,
    /// Per sample pixel: the sector of its hue in P (255: not in a sector), P's hue, and per sector P's log median C*.
    sector: Vec<u8>,
    p_hue: Vec<f64>,
    p_sec_logc: [f64; SECTORS],
    /// Whether the network is v6 (it has the frame cast and the wall limit as outputs).
    v6: bool,
    /// Whether the network is v7 (it also has `T.hl_cmax`): the cast off the whites and the skin, the stronger skin wall and
    /// the highlight wall.
    v7: bool,
    /// Whether the network is v8 (it also has `T.purple_ok`): the purple wall, the whites' direction, the skin drift wall,
    /// clothing as a group and the clip term against the photo's own shares.
    v8: bool,
    /// Whether the network is v9 (it also has `T.mem.sky_deg`): the walls read mean-vector hue and chroma.
    v9: bool,
    /// Whether the network is v10 (it also has `T.purple_max_share`): the pixel-level purple guard.
    v10: bool,
    /// Whether the network is v11 (it also has `T.purple_rot_deg`): Oklab purple band, turn-based purple guards, skin chroma wall.
    v11: bool,
    /// Whether the network is v12 (it also has `T.turn_deg`), per principle: the network is v12 and `--ablate` leaves it on.
    /// P1: turns count wherever they end. P2: the content turn budget. P3: one hue space and the stiffer memory wall.
    /// P4: whites by pixel. P5: the clip allowance, the crush test and the stiffer clip term.
    p1: bool,
    p2: bool,
    /// V13: the content turn budget counts a turn against `allowed_turn_deg(C*)` instead of `T.turn_deg`; P1 is off.
    v13: bool,
    p3: bool,
    p4: bool,
    p5: bool,
    /// V12: P's whites for the chroma cap and the direction (L* >= 80, C* < 8, not skin).
    hl4_scope: Vec<bool>,
    /// V11: per sample pixel, P's Oklab hue and CIELAB C*, and whether it is not skin.
    p_okh: Vec<f64>,
    p_cstar: Vec<f64>,
    not_skin: Vec<bool>,
    /// V11: the purple groups' pixel indices and P's mean Oklab (a, b) over each.
    pg: Vec<Vec<u32>>,
    p_pg_ab: Vec<[f64; 2]>,
    /// V10: per sample pixel, whether it may count as newly purple (not skin, and in P outside the purple band or under C* 2).
    np_scope: Vec<bool>,
    /// V9: P's mean (a*, b*) over its whites and over each sector, and the dark-colours group (P's pixels with L* < 35 and
    /// C* >= 2, not skin): its scope and P's mean (a*, b*) there.
    p_hl_a: f64,
    p_hl_b: f64,
    p_sec_ab: [[f64; 2]; SECTORS],
    dark_scope: Vec<bool>,
    p_dark_ab: [f64; 2],
    /// V7: the sample pixels that are whites in P (L* >= 80, C* < 10), their share, P's median C* there and P's mean hue there.
    hl_scope: Vec<bool>,
    p_hl_c: f64,
    p_hl_h: Option<f64>,
}

fn build_v5_ctx(
    net: &Network,
    p_px: &[Px],
    scene: &Scene,
    r_px: &[Px],
    r_scene: &Scene,
    ablate: Ablate,
    objective: Objective,
) -> R<V5Ctx> {
    let v7 = net.index.contains_key("T.hl_cmax");
    let v11 = net.index.contains_key("T.purple_rot_deg");
    let v12 = net.index.contains_key("T.turn_deg");
    // V13: P1 is withdrawn (always off, as with `--ablate p1`); P2 counts a turn against `allowed_turn_deg`, which needs
    // `T.turn_cref` and `T.turn_cap`, so a network without them is an error, not a silent v12.
    let v13 = objective == Objective::V13;
    if v13 && !(net.index.contains_key("T.turn_cap") && net.index.contains_key("T.turn_cref")) {
        return Err(
            "objective v13 needs a network with T.turn_cref and T.turn_cap (refgrade-v13.ywn)"
                .into(),
        );
    }
    let (p1, p2, p3, p4, p5) = (
        v12 && !v13 && ablate.on(1),
        v12 && ablate.on(2),
        v12 && ablate.on(3),
        v12 && ablate.on(4),
        v12 && ablate.on(5),
    );
    let p0 = measure(p_px, scene, &[false; 6], v7, v11);
    let r0 = measure(r_px, r_scene, &[false; 6], v7, v11);
    let mut excl = [false; 6];
    for (k, flag) in excl.iter_mut().enumerate().take(5) {
        *flag = p0.el[k].share >= V5_EXCLUDE_SHARE && r0.el[k].share >= V5_EXCLUDE_SHARE;
    }
    let p = measure(p_px, scene, &excl, v7, v11);
    let r = measure(r_px, r_scene, &excl, v7, v11);
    let inputs = network_inputs(&p, &r);
    let out = NetOut::new(net, &inputs)?;
    // P's sectors on the same pixels every candidate is measured on.
    let mut sector = vec![255u8; p_px.len()];
    let mut p_hue = vec![0.0; p_px.len()];
    let mut cs: [Vec<f64>; SECTORS] = Default::default();
    let mut sec_sum = [[0.0f64; 2]; SECTORS];
    let mut sec_n = [0usize; SECTORS];
    for (i, q) in p_px.iter().enumerate() {
        let (a, b) = (q.lab[1], q.lab[2]);
        let c = a.hypot(b);
        p_hue[i] = b.atan2(a).to_degrees();
        if c > V3_HUE_MIN_CHROMA
            && !scene.skin[i]
            && !(0..5).any(|k| excl[k] && in_elem(scene, i, k))
        {
            let s = ((((p_hue[i] + 30.0).rem_euclid(360.0)) / 60.0) as usize) % SECTORS;
            sector[i] = s as u8;
            cs[s].push(c);
            sec_sum[s][0] += a;
            sec_sum[s][1] += b;
            sec_n[s] += 1;
        }
    }
    let mut p_sec_logc = [0.0; SECTORS];
    for j in 0..SECTORS {
        p_sec_logc[j] = median_of(&mut cs[j]).max(1.0).ln();
    }
    let hl_scope: Vec<bool> = p_px
        .iter()
        .map(|q| q.lab[0] >= V7_HL_L_MIN && q.lab[1].hypot(q.lab[2]) < V7_HL_C_MAX)
        .collect();
    let (p_hl_c, p_hl_h, p_hl_a, p_hl_b) = hl_stats(p_px, &hl_scope);
    let mut p_sec_ab = [[0.0; 2]; SECTORS];
    for j in 0..SECTORS {
        if sec_n[j] > 0 {
            p_sec_ab[j] = [
                sec_sum[j][0] / sec_n[j] as f64,
                sec_sum[j][1] / sec_n[j] as f64,
            ];
        }
    }
    let dark_scope: Vec<bool> = p_px
        .iter()
        .enumerate()
        .map(|(i, q)| {
            !scene.skin[i] && q.lab[0] < V9_DARK_L && q.lab[1].hypot(q.lab[2]) >= V9_DARK_C
        })
        .collect();
    let p_dark_ab = mean_ab(p_px, &dark_scope);
    let p_okh: Vec<f64> = p_px.iter().map(|q| oklab_hue(q.ok)).collect();
    let p_cstar: Vec<f64> = p_px.iter().map(|q| q.lab[1].hypot(q.lab[2])).collect();
    let not_skin: Vec<bool> = (0..p_px.len()).map(|i| !scene.skin[i]).collect();
    let mut pg: Vec<Vec<u32>> = vec![Vec::new(); PG_N];
    if v11 {
        for i in 0..p_px.len() {
            for k in 0..6 {
                if in_elem(scene, i, k) {
                    pg[k].push(i as u32);
                }
            }
            if usize::from(scene.elem[i]) == ELEM_PERSON && !scene.skin[i] {
                pg[PG_CLOTH].push(i as u32);
            }
            if sector[i] != 255 {
                pg[PG_SEC + usize::from(sector[i])].push(i as u32);
            }
            if hl_scope[i] {
                pg[PG_HL].push(i as u32);
            }
            if dark_scope[i] {
                pg[PG_DARK].push(i as u32);
            }
        }
    }
    let p_pg_ab = pg_means(p_px, &pg);
    let np_scope: Vec<bool> = p_px
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let h = q.lab[2].atan2(q.lab[1]).to_degrees().rem_euclid(360.0);
            let in_band = (V8_PURPLE_BAND.0..=V8_PURPLE_BAND.1).contains(&h);
            !scene.skin[i] && (!in_band || q.lab[1].hypot(q.lab[2]) < V10_NP_FROM_C)
        })
        .collect();
    let hl4_scope: Vec<bool> = p_px
        .iter()
        .enumerate()
        .map(|(i, q)| {
            !scene.skin[i] && q.lab[0] >= V7_HL_L_MIN && q.lab[1].hypot(q.lab[2]) < V12_WHITE_C_MAX
        })
        .collect();
    Ok(V5Ctx {
        v13,
        p1,
        p2,
        p3,
        p4,
        p5,
        hl4_scope,
        v7,
        v8: net.index.contains_key("T.purple_ok"),
        v9: net.index.contains_key("T.mem.sky_deg"),
        v10: net.index.contains_key("T.purple_max_share"),
        v11,
        p_okh,
        p_cstar,
        not_skin,
        pg,
        p_pg_ab,
        np_scope,
        p_hl_a,
        p_hl_b,
        p_sec_ab,
        dark_scope,
        p_dark_ab,
        hl_scope,
        p_hl_c,
        p_hl_h,
        v6: net.index.contains_key("G.unify"),
        excl,
        p,
        r,
        inputs,
        out,
        sector,
        p_hue,
        p_sec_logc,
    })
}

/// The mean (a*, b*) over the pixels in the scope.
fn mean_ab(px: &[Px], scope: &[bool]) -> [f64; 2] {
    let (mut sa, mut sb, mut n) = (0.0, 0.0, 0usize);
    for (q, _) in px.iter().zip(scope).filter(|&(_, &s)| s) {
        sa += q.lab[1];
        sb += q.lab[2];
        n += 1;
    }
    if n == 0 {
        [0.0, 0.0]
    } else {
        [sa / n as f64, sb / n as f64]
    }
}

/// The median C* and the circular mean hue (of the pixels with C* above 5) over the group.
fn hl_stats(px: &[Px], scope: &[bool]) -> (f64, Option<f64>, f64, f64) {
    let mut cs = Vec::new();
    let (mut sx, mut sy, mut hn) = (0.0, 0.0, 0usize);
    let (mut sa, mut sb) = (0.0, 0.0);
    for (q, _) in px.iter().zip(scope).filter(|&(_, &s)| s) {
        let (a, b) = (q.lab[1], q.lab[2]);
        let c = a.hypot(b);
        cs.push(c);
        sa += a;
        sb += b;
        if c > V3_HUE_MIN_CHROMA {
            sx += a / c;
            sy += b / c;
            hn += 1;
        }
    }
    let k = cs.len().max(1) as f64;
    let med = median_of(&mut cs);
    (
        med,
        if hn > 0 {
            Some(circ_mean_deg(sx, sy))
        } else {
            None
        },
        sa / k,
        sb / k,
    )
}

/// V11: the mean Oklab (a, b) of `px` over each group's pixels.
fn pg_means(px: &[Px], pg: &[Vec<u32>]) -> Vec<[f64; 2]> {
    pg.iter()
        .map(|g| {
            if g.is_empty() {
                return [0.0, 0.0];
            }
            let (mut a, mut b) = (0.0, 0.0);
            for &i in g {
                a += px[i as usize].ok[0];
                b += px[i as usize].ok[1];
            }
            [a / g.len() as f64, b / g.len() as f64]
        })
        .collect()
}

/// V12 (P4): the mean per-pixel C* over the whites in `scope`, and the mean per-pixel distance of their (a*, b*) to the target.
/// Opposite tints do not cancel in either.
fn hl4_stats(px: &[Px], scope: &[bool], white_a: f64, white_b: f64) -> (f64, f64) {
    let (mut c, mut d, mut n) = (0.0, 0.0, 0usize);
    for (q, _) in px.iter().zip(scope).filter(|&(_, &s)| s) {
        c += q.lab[1].hypot(q.lab[2]);
        d += (q.lab[1] - white_a).hypot(q.lab[2] - white_b);
        n += 1;
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        (c / n as f64, d / n as f64)
    }
}

/// V12 (P1): whether a colour's move counts as a purple turn, wherever it ends. `p`, `c` are (C*, Oklab hue) in photo0 and in
/// the candidate. A grey that becomes purple counts; so does a strong colour that turns by at least `rot` degrees when either
/// end is in the purple band (with C* of at least [`V10_NP_C`] there, so that the band means something).
fn counts_as_purple_turn(p: (f64, f64), c: (f64, f64), rot: f64) -> bool {
    let in_band = |h: f64| (V11_PURPLE_BAND.0..=V11_PURPLE_BAND.1).contains(&h);
    if c.0 >= V10_NP_C && in_band(c.1) && p.0 < V10_NP_FROM_C {
        return true;
    }
    wrap_deg(c.1 - p.1).abs() >= rot
        && ((c.0 >= V10_NP_C && in_band(c.1)) || (p.0 >= V10_NP_C && in_band(p.1)))
}

/// V13 (P2): the Oklab hue turn, in degrees, a colour of photo0 CIELAB chroma `c_star` may make before it counts as content
/// turning: `clamp(asin(min(1, cref * sin(turn_deg) / c_star)), turn_deg, cap)`. A pale colour moved sideways reads as a
/// tint, so it may turn up to `cap`; from `c_star = cref` up it may turn only `turn_deg`. A `c_star` of 0 gives `cap`.
fn allowed_turn_deg(c_star: f64, turn_deg: f64, cref: f64, cap: f64) -> f64 {
    let ratio = if c_star > 0.0 {
        (cref * turn_deg.to_radians().sin() / c_star).min(1.0)
    } else {
        1.0
    };
    ratio
        .asin()
        .to_degrees()
        .max(turn_deg)
        .min(cap.max(turn_deg))
}

/// V12 (P2): the share of the sample's pixels that are strong colours in photo0 (C* at least [`V12_TURN_MIN_C`], not skin) and
/// whose Oklab hue turned by more than `turn_deg`. With `cref_cap` (V13: `T.turn_cref`, `T.turn_cap`) the limit is
/// [`allowed_turn_deg`] of the pixel's photo0 C* instead of `turn_deg`.
fn content_turn_share(px: &[Px], ctx: &V5Ctx, turn_deg: f64, cref_cap: Option<(f64, f64)>) -> f64 {
    if px.is_empty() {
        return 0.0;
    }
    let n = px
        .iter()
        .enumerate()
        .filter(|&(i, q)| {
            ctx.not_skin[i]
                && ctx.p_cstar[i] >= V12_TURN_MIN_C
                && wrap_deg(oklab_hue(q.ok) - ctx.p_okh[i]).abs()
                    > cref_cap.map_or(turn_deg, |(cref, cap)| {
                        allowed_turn_deg(ctx.p_cstar[i], turn_deg, cref, cap)
                    })
        })
        .count();
    n as f64 / px.len() as f64
}

/// V12 (P5): the share of a frame a grade may clip or crush: the photo's own, plus at most `extra` where the reference has more.
fn clip_allowance(photo: f64, reference: f64, extra: f64) -> f64 {
    photo.max(reference.min(photo + extra))
}

/// V11: the share of `px` that is newly purple: Oklab hue in the purple band, C* at least [`V10_NP_C`], and either a grey in
/// photo0 (C* under [`V10_NP_FROM_C`]) or a hue that turned by at least `rot` degrees; skin left out.
fn turned_purple_share(px: &[Px], ctx: &V5Ctx, rot: f64) -> f64 {
    if px.is_empty() {
        return 0.0;
    }
    if ctx.p1 {
        let n = px
            .iter()
            .enumerate()
            .filter(|&(i, q)| {
                ctx.not_skin[i]
                    && counts_as_purple_turn(
                        (ctx.p_cstar[i], ctx.p_okh[i]),
                        (q.lab[1].hypot(q.lab[2]), oklab_hue(q.ok)),
                        rot,
                    )
            })
            .count();
        return n as f64 / px.len() as f64;
    }
    let n = px
        .iter()
        .enumerate()
        .filter(|&(i, q)| {
            if !ctx.not_skin[i] || q.lab[1].hypot(q.lab[2]) < V10_NP_C {
                return false;
            }
            let h = oklab_hue(q.ok);
            (V11_PURPLE_BAND.0..=V11_PURPLE_BAND.1).contains(&h)
                && (ctx.p_cstar[i] < V10_NP_FROM_C || wrap_deg(h - ctx.p_okh[i]).abs() >= rot)
        })
        .count();
    n as f64 / px.len() as f64
}

/// V10: the share of `px` that is newly purple: hue inside the purple band and C* at least [`V10_NP_C`], on a pixel that `scope`
/// allows (not skin; in photo0 outside the band, or under C* [`V10_NP_FROM_C`]).
fn newly_purple_share(px: &[Px], scope: &[bool]) -> f64 {
    if px.is_empty() {
        return 0.0;
    }
    let n = px
        .iter()
        .zip(scope)
        .filter(|&(q, &ok)| {
            ok && q.lab[1].hypot(q.lab[2]) >= V10_NP_C && {
                let h = q.lab[2].atan2(q.lab[1]).to_degrees().rem_euclid(360.0);
                (V8_PURPLE_BAND.0..=V8_PURPLE_BAND.1).contains(&h)
            }
        })
        .count();
    n as f64 / px.len() as f64
}

/// A candidate's measurements: the picture's, and per sector the mean hue shift and log chroma change against P on P's pixels.
#[derive(Default)]
struct CandMeas {
    m: Meas,
    sec_shift: [f64; SECTORS],
    sec_dlogc: [f64; SECTORS],
    /// V7: the candidate's median C* and mean hue over P's whites.
    hl_c: f64,
    hl_h: Option<f64>,
    /// V8: the candidate's mean a* and b* over P's whites.
    hl_a: f64,
    hl_b: f64,
    /// V9: the candidate's mean (a*, b*) over each of P's sectors' pixels and over P's dark-colours pixels.
    sec_ab: [[f64; 2]; SECTORS],
    dark_ab: [f64; 2],
    /// V10: the share of the sample's pixels that are newly purple in the candidate.
    np_share: f64,
    /// V11: the candidate's mean Oklab (a, b) over each purple group's pixels.
    pg_ab: Vec<[f64; 2]>,
    /// V12: the mean per-pixel C* and the mean per-pixel distance to the whites' target over P's whites; the share of the
    /// sample whose strong colour turned further than `T.turn_deg`.
    hl4_c: f64,
    hl4_d: f64,
    turn_share: f64,
}

fn cand_meas(px: &[Px], scene: &Scene, ctx: &V5Ctx) -> CandMeas {
    let m = measure(px, scene, &ctx.excl, ctx.v7, ctx.v11);
    let (hl_c, hl_h, hl_a, hl_b) = if ctx.v7 {
        hl_stats(px, &ctx.hl_scope)
    } else {
        (0.0, None, 0.0, 0.0)
    };
    let mut shift = [0.0f64; SECTORS];
    let mut cnt = [0usize; SECTORS];
    let mut cs: [Vec<f64>; SECTORS] = Default::default();
    for (i, q) in px.iter().enumerate() {
        let s = ctx.sector[i];
        if s == 255 {
            continue;
        }
        let s = usize::from(s);
        let h = q.lab[2].atan2(q.lab[1]).to_degrees();
        shift[s] += wrap_deg(h - ctx.p_hue[i]);
        cnt[s] += 1;
        cs[s].push(q.lab[1].hypot(q.lab[2]));
    }
    let mut sec_ab = [[0.0; 2]; SECTORS];
    if ctx.v9 {
        let mut sum = [[0.0f64; 2]; SECTORS];
        for (i, q) in px.iter().enumerate() {
            let s = ctx.sector[i];
            if s != 255 {
                sum[usize::from(s)][0] += q.lab[1];
                sum[usize::from(s)][1] += q.lab[2];
            }
        }
        for j in 0..SECTORS {
            if cnt[j] > 0 {
                sec_ab[j] = [sum[j][0] / cnt[j] as f64, sum[j][1] / cnt[j] as f64];
            }
        }
    }
    let dark_ab = if ctx.v9 {
        mean_ab(px, &ctx.dark_scope)
    } else {
        [0.0, 0.0]
    };
    let mut sec_shift = [0.0; SECTORS];
    let mut sec_dlogc = [0.0; SECTORS];
    for j in 0..SECTORS {
        if cnt[j] > 0 {
            sec_shift[j] = shift[j] / cnt[j] as f64;
            sec_dlogc[j] = median_of(&mut cs[j]).max(1.0).ln() - ctx.p_sec_logc[j];
        }
    }
    let np_share = if ctx.v11 {
        turned_purple_share(px, ctx, ctx.out.get("T.purple_rot_deg"))
    } else {
        newly_purple_share(px, &ctx.np_scope)
    };
    let pg_ab = if ctx.v11 {
        pg_means(px, &ctx.pg)
    } else {
        Vec::new()
    };
    let (hl4_c, hl4_d) = if ctx.p4 {
        hl4_stats(
            px,
            &ctx.hl4_scope,
            ctx.out.get("T.white_a"),
            ctx.out.get("T.white_b"),
        )
    } else {
        (0.0, 0.0)
    };
    let turn_share = if ctx.p2 {
        content_turn_share(
            px,
            ctx,
            ctx.out.get("T.turn_deg"),
            ctx.v13
                .then(|| (ctx.out.get("T.turn_cref"), ctx.out.get("T.turn_cap"))),
        )
    } else {
        0.0
    };
    CandMeas {
        m,
        hl4_c,
        hl4_d,
        turn_share,
        np_share,
        pg_ab,
        sec_shift,
        sec_dlogc,
        hl_c,
        hl_h,
        hl_a,
        hl_b,
        sec_ab,
        dark_ab,
    }
}

/// The V5 terms, unweighted by stage.
#[derive(Clone, Copy, Debug, Default)]
struct V5T {
    floor: f64,
    ceil: f64,
    contrast: f64,
    anchor: f64,
    tint: f64,
    sat: f64,
    sec_hue: f64,
    sec_chroma: f64,
    el_hue: f64,
    el_chroma: f64,
    el_exp: f64,
    subject: f64,
    guard_hue: f64,
    guard_skin: f64,
    guard_sky: f64,
    /// V6: the frame cast term.
    frame_cast: f64,
    /// V7: the highlight-neutrality wall (chroma and hue drift of the whites).
    guard_hl: f64,
    /// V8: the purple wall, the whites' direction and the skin drift wall.
    guard_purple: f64,
    guard_white_dir: f64,
    guard_skin_drift: f64,
    /// V9: the memory-colour wall (sky and foliage).
    guard_mem: f64,
    /// V10: the pixel-level purple guard.
    guard_purple_px: f64,
    /// V11: the skin chroma wall, and the skin's log chroma change against photo0 (a diagnostic for every network objective).
    guard_skin_chroma: f64,
    /// V12: the content turn budget.
    guard_turn: f64,
    skin_dlogc: f64,
    /// The share of the sample's pixels that are newly purple (a diagnostic for every network objective).
    np_share: f64,
}

fn sq(x: f64) -> f64 {
    x * x
}

/// The walls of network v9: every group's hue and chroma are those of its mean (a*, b*) vector, which holds steady for dark,
/// low-chroma colours. Groups: the elements, clothing, the sectors of unelemented pixels, the whites and the dark colours.
fn guards_v9(t: &mut V5T, c: &CandMeas, ctx: &V5Ctx) {
    let o = &ctx.out;
    let m = &c.m;
    let wall = o.get("T.wall_deg");
    let drift = |d: f64, limit: f64| {
        if d.abs() > limit {
            V5_GUARD_HUE_WEIGHT * sq((d.abs() - limit) / V5_GUARD_HUE_SCALE_DEG)
        } else {
            0.0
        }
    };
    let vec_hc = |a: f64, b: f64| (circ_mean_deg(a, b), a.hypot(b));
    // V12 (P3): one hue space. A group's hue is that of its mean Oklab (a, b); its chroma threshold stays in CIELAB.
    let hues = |gid: usize, ph: f64, ch: f64| {
        if ctx.p3 {
            (oklab_hue(ctx.p_pg_ab[gid]), oklab_hue(c.pg_ab[gid]))
        } else {
            (ph, ch)
        }
    };
    // (photo0 hue, photo0 chroma, candidate hue) of every group that counts.
    let mut groups: Vec<(f64, f64, f64, usize)> = Vec::new();
    let n = ctx.hl_scope.len() as f64;
    for (k, e) in V5_ELEMS.iter().enumerate() {
        let (pe, me) = (&ctx.p.el[k], &m.el[k]);
        if pe.share < V5_PRESENT_SHARE || me.n == 0 {
            continue;
        }
        let ((ph, pc), (ch, _)) = (pe.mv(), me.mv());
        if pc < V9_MIN_MV_C {
            continue;
        }
        let (ph, ch) = hues(k, ph, ch);
        let limit = if o.get(&format!("G.{e}.hue")) > V5_GATE_OPEN {
            wall.max(V5_CAP_DEG[k] + V5_GUARD_HUE_EXTRA_DEG)
        } else {
            wall
        };
        t.guard_hue += drift(wrap_deg(ch - ph), limit);
        groups.push((ph, pc, ch, k));
        // The memory-colour wall: sky and foliage hold their hue within the network's limit.
        let key = match k {
            0 => Some("T.mem.sky_deg"),
            1 => Some("T.mem.vegetation_deg"),
            _ => None,
        };
        if let Some(key) = key {
            let excess = (wrap_deg(ch - ph).abs() - o.get(key)).max(0.0);
            let (w, scale) = if ctx.p3 { V12_MEM } else { V9_MEM };
            t.guard_mem += w * sq(excess / scale);
        }
    }
    let cl = (&ctx.p.clothing, &m.clothing);
    if cl.0.share >= V5_PRESENT_SHARE && cl.1.n > 0 {
        let ((ph, pc), (ch, _)) = (cl.0.mv(), cl.1.mv());
        if pc >= V9_MIN_MV_C {
            let (ph, ch) = hues(PG_CLOTH, ph, ch);
            t.guard_hue += drift(wrap_deg(ch - ph), wall);
            groups.push((ph, pc, ch, PG_CLOTH));
        }
    }
    for j in 0..SECTORS {
        if ctx.p.sec[j].share >= V5_GUARD_SECTOR_MIN_SHARE && ctx.p.sec[j].n > 0 {
            let (ph, pc) = vec_hc(ctx.p_sec_ab[j][0], ctx.p_sec_ab[j][1]);
            let (ch, _) = vec_hc(c.sec_ab[j][0], c.sec_ab[j][1]);
            if pc >= V9_MIN_MV_C {
                let (ph, ch) = hues(PG_SEC + j, ph, ch);
                t.guard_hue += drift(wrap_deg(ch - ph), wall);
                groups.push((ph, pc, ch, PG_SEC + j));
            }
        }
    }
    let n_hl = ctx.hl_scope.iter().filter(|&&s| s).count() as f64;
    if n_hl >= V7_HL_MIN_SHARE * n {
        let (ph, pc) = vec_hc(ctx.p_hl_a, ctx.p_hl_b);
        let (ch, cc) = vec_hc(c.hl_a, c.hl_b);
        // The highlight chroma cap, on the mean-vector chroma.
        // V12 (P4): the cap is on the mean per-pixel chroma of the whites.
        let cc = if ctx.p4 { c.hl4_c } else { cc };
        t.guard_hl = V7_HL_WEIGHT * sq((cc - o.get("T.hl_cmax")).max(0.0) / V7_HL_SCALE);
        if pc >= V7_HL_MIN_C {
            let (ph, ch) = hues(PG_HL, ph, ch);
            t.guard_hl += drift(wrap_deg(ch - ph), wall);
        }
        if pc >= V9_MIN_MV_C {
            groups.push((ph, pc, ch, PG_HL));
        }
        // The whites' direction.
        let d = if ctx.p4 {
            c.hl4_d
        } else {
            (c.hl_a - o.get("T.white_a")).hypot(c.hl_b - o.get("T.white_b"))
        };
        let tol = o.get("T.white_tol");
        if d > tol {
            t.guard_white_dir = V8_WHITE_DIR.0 * sq((d - tol) / V8_WHITE_DIR.1);
        }
    }
    let n_dark = ctx.dark_scope.iter().filter(|&&s| s).count() as f64;
    if n_dark >= V5_PRESENT_SHARE * n {
        let (ph, pc) = vec_hc(ctx.p_dark_ab[0], ctx.p_dark_ab[1]);
        let (ch, _) = vec_hc(c.dark_ab[0], c.dark_ab[1]);
        if pc >= V9_MIN_MV_C {
            let (ph, ch) = hues(PG_DARK, ph, ch);
            t.guard_hue += drift(wrap_deg(ch - ph), wall);
            groups.push((ph, pc, ch, PG_DARK));
        }
    }
    // The purple wall, with its approach term. V11 measures hue in Oklab and counts turns, not entries.
    let ok = 1.0 - o.get("T.purple_ok");
    let (lo, hi) = if ctx.v11 {
        V11_PURPLE_BAND
    } else {
        V8_PURPLE_BAND
    };
    let rot = if ctx.v11 {
        o.get("T.purple_rot_deg")
    } else {
        0.0
    };
    let in_band = |h: f64| (lo..=hi).contains(&h.rem_euclid(360.0));
    let dist = |h: f64| {
        if in_band(h) {
            0.0
        } else {
            wrap_deg(h - lo).abs().min(wrap_deg(h - hi).abs())
        }
    };
    for (ph, pc, ch, gid) in groups {
        if pc < V9_MIN_MV_C {
            continue;
        }
        let (ph, ch) = if ctx.v11 {
            (oklab_hue(ctx.p_pg_ab[gid]), oklab_hue(c.pg_ab[gid]))
        } else {
            (ph, ch)
        };
        if ctx.v11 {
            // V12 (P1): a turn counts wherever it ends, so a group that starts in the band counts too.
            if in_band(ch) || (ctx.p1 && in_band(ph)) {
                let excess = (wrap_deg(ch - ph).abs() - rot).max(0.0);
                t.guard_purple += V11_TURN.0 * sq(excess / V11_TURN.1);
                continue;
            }
        } else if in_band(ph) {
            continue;
        }
        if in_band(ph) {
            continue;
        }
        let ch = ch.rem_euclid(360.0);
        if in_band(ch) {
            let d_in = (ch - lo).min(hi - ch);
            t.guard_purple += V8_PURPLE_WEIGHT * sq(d_in / V8_PURPLE_SCALE_DEG) * ok;
        } else if dist(ch) < V8_APPROACH_REACH_DEG && dist(ch) < dist(ph) {
            t.guard_purple += V8_APPROACH_WEIGHT
                * sq((V8_APPROACH_REACH_DEG - dist(ch)) / V5_GUARD_HUE_SCALE_DEG)
                * ok;
        }
    }
    // Skin: the absolute wall and the drift wall, on the mean vector.
    let (pe, me) = (&ctx.p.el[5], &m.el[5]);
    if pe.share >= V5_PRESENT_SHARE && me.n > 0 {
        let ((ph, pc), (h, chroma)) = (pe.mv(), me.mv());
        let eh = (V5_GUARD_SKIN_HUE.0 - h)
            .max(h - V5_GUARD_SKIN_HUE.1)
            .max(0.0);
        let ec = (V5_GUARD_SKIN_CHROMA.0 - chroma)
            .max(chroma - V5_GUARD_SKIN_CHROMA.1)
            .max(0.0);
        t.guard_skin =
            V7_GUARD_SKIN_WEIGHT * (sq(eh / V5_GUARD_SKIN_SCALE) + sq(ec / V5_GUARD_SKIN_SCALE));
        if pc >= V9_MIN_MV_C {
            let (ph, h) = if ctx.p3 {
                (oklab_hue(ctx.p_pg_ab[5]), oklab_hue(c.pg_ab[5]))
            } else {
                (ph, h)
            };
            let excess = (wrap_deg(h - ph).abs() - o.get("T.skin_drift")).max(0.0);
            t.guard_skin_drift = V8_SKIN_DRIFT.0 * sq(excess / V8_SKIN_DRIFT.1);
        }
        if ctx.v11 && pe.share >= V11_SKIN_MIN_SHARE {
            let excess = (t.skin_dlogc - o.get("T.skin_dlogC_max")).max(0.0);
            t.guard_skin_chroma = V11_SKIN_CHROMA.0 * sq(excess / V11_SKIN_CHROMA.1);
        }
    }
    // V12 (P2): the content turn budget.
    if ctx.p2 {
        let limit = o.get("T.turn_share");
        if c.turn_share > limit {
            t.guard_turn = V12_TURN.0 * sq((c.turn_share - limit) / V12_TURN.1);
        }
    }
    // The sky chroma wall, on the mean-vector chroma.
    if ctx.p.el[0].share >= V5_PRESENT_SHARE {
        let limit = (V5_GUARD_SKY_FACTOR * ctx.p.el[0].mv().1).max(o.get("T.sky.logC").exp());
        let excess = (m.el[0].mv().1 - limit).max(0.0);
        t.guard_sky = V5_GUARD_SKY_WEIGHT * sq(excess / V5_GUARD_SKY_SCALE);
    }
}

/// The clip term of a network objective for a candidate's measurements `m`.
fn clip_term(ctx: &V5Ctx, m: &Meas) -> f64 {
    let over = |o: f64, r: f64| ((o - r - CLIP_SLACK).max(0.0) / CLIP_SCALE).powi(2);
    // V8: against the larger of the reference's and the photo's own share (a photo black in half its frame is not pushed to
    // lift it).
    let (rh, rl) = if ctx.v8 {
        (ctx.r.hi.max(ctx.p.hi), ctx.r.lo.max(ctx.p.lo))
    } else {
        (ctx.r.hi, ctx.r.lo)
    };
    let mut clip = over(m.hi, rh) + over(m.lo, rl);
    if ctx.p5 {
        // V12 (P5): the reference may license at most `T.clip_extra` more than the photo has; crushed black is L* < 2.5; stiffer.
        let extra = ctx.out.get("T.clip_extra");
        let over12 = |o: f64, a: f64| ((o - a - CLIP_SLACK).max(0.0) / V12_CLIP_SCALE).powi(2);
        clip = over12(m.hi, clip_allowance(ctx.p.hi, ctx.r.hi, extra))
            + over12(m.lo25, clip_allowance(ctx.p.lo25, ctx.r.lo25, extra));
    }
    clip
}

/// The objective of `09` section 4 for a candidate.
fn terms_v5(c: &CandMeas, ctx: &V5Ctx) -> V5T {
    let o = &ctx.out;
    let m = &c.m;
    let mut t = V5T {
        np_share: c.np_share,
        ..V5T::default()
    };
    {
        let (pe, me) = (&ctx.p.el[5], &m.el[5]);
        if pe.share >= V11_SKIN_MIN_SHARE && me.n > 0 {
            t.skin_dlogc = (me.mv().1.max(1.0) / pe.mv().1.max(1.0)).ln();
        }
    }
    let term = |err: f64, scale: f64, dead_band: f64| sq(dead(err, dead_band) / scale);
    t.floor = V5_FLOOR.2 * term(m.p2 - o.get("T.p2"), V5_FLOOR.0, V5_FLOOR.1);
    t.ceil = V5_CEIL.2 * term(m.p98 - o.get("T.p98"), V5_CEIL.0, V5_CEIL.1);
    t.contrast = V5_CONTRAST.2 * term(m.iqr_rel - o.get("T.iqr_rel"), V5_CONTRAST.0, V5_CONTRAST.1);
    t.anchor = V5_ANCHOR.2 * term(m.p50 - o.get("T.p50"), V5_ANCHOR.0, V5_ANCHOR.1);
    for (b, name) in ["s", "m", "h"].iter().enumerate() {
        for (ch, cn) in ["a", "b"].iter().enumerate() {
            t.tint += V5_TINT_BAND_WEIGHT[b]
                * term(
                    m.tint[b][ch] - o.get(&format!("T.tint_{name}_{cn}")),
                    V5_TINT.0,
                    V5_TINT.1,
                );
        }
    }
    t.sat = V5_SAT.2 * term(m.logsat - o.get("T.logsat"), V5_SAT.0, V5_SAT.1);
    for j in 0..SECTORS {
        let g = o.get(&format!("G.s{j}"));
        if g <= 0.0 {
            continue;
        }
        let w = g * ctx.p.sec[j].share.sqrt();
        t.sec_hue += V5_SECTOR_HUE.2
            * w
            * term(
                c.sec_shift[j] - o.get(&format!("T.s{j}.dh")),
                V5_SECTOR_HUE.0,
                V5_SECTOR_HUE.1,
            );
        t.sec_chroma += V5_SECTOR_CHROMA.2
            * w
            * term(
                c.sec_dlogc[j] - o.get(&format!("T.s{j}.dlogC")),
                V5_SECTOR_CHROMA.0,
                V5_SECTOR_CHROMA.1,
            );
    }
    for (k, e) in V5_ELEMS.iter().enumerate() {
        let (gm, gh) = (
            o.get(&format!("G.{e}.matched")),
            o.get(&format!("G.{e}.hue")),
        );
        let w = V5_W_TERM[k] * ctx.p.el[k].share.sqrt();
        let em = &m.el[k];
        if gh > 0.0 && em.has_hue {
            t.el_hue += w
                * gh
                * term(
                    wrap_deg(em.h - o.get(&format!("T.{e}.h"))),
                    V5_EL_HUE.0,
                    V5_EL_HUE.1,
                );
        }
        if gm > 0.0 {
            t.el_chroma += w
                * gm
                * term(
                    em.logc - o.get(&format!("T.{e}.logC")),
                    V5_EL_CHROMA.0,
                    V5_EL_CHROMA.1,
                );
            t.el_exp += w
                * gm
                * term(
                    (em.l - m.p50) - o.get(&format!("T.{e}.off")),
                    V5_EL_EXPOSURE.0,
                    V5_EL_EXPOSURE.1,
                );
        }
    }
    if ctx.v6 {
        let g = o.get("G.unify");
        if g > 0.0 {
            let e = |got: f64, name: &str| sq(dead(got - o.get(name), V6_CAST.1) / V6_CAST.0);
            t.frame_cast = V6_CAST.2 * g * (e(m.cast_a, "T.cast_a") + e(m.cast_b, "T.cast_b"));
        }
    }
    let gs = o.get("G.subj");
    if gs > 0.0 {
        t.subject = V5_SUBJECT.2 * gs * term(m.sep - o.get("T.sep"), V5_SUBJECT.0, V5_SUBJECT.1);
    }
    // Guards.
    if ctx.v9 {
        guards_v9(&mut t, c, ctx);
        if ctx.v10 {
            let limit = o.get("T.purple_max_share");
            if c.np_share > limit {
                t.guard_purple_px = V10_NP_WEIGHT * sq((c.np_share - limit) / V10_NP_SCALE);
            }
        }
        return t;
    }
    let drift = |d: f64, limit: f64| {
        if d.abs() > limit {
            V5_GUARD_HUE_WEIGHT * sq((d.abs() - limit) / V5_GUARD_HUE_SCALE_DEG)
        } else {
            0.0
        }
    };
    // The wall limit: 25 degrees in v5; in v6 the network's `T.wall_deg` (25, up to 55 when unify is open).
    let wall = if ctx.v6 {
        o.get("T.wall_deg")
    } else {
        V5_GUARD_HUE_LIMIT_DEG
    };
    for (k, e) in V5_ELEMS.iter().enumerate() {
        if ctx.p.el[k].share >= V5_PRESENT_SHARE && ctx.p.el[k].has_hue && m.el[k].has_hue {
            let limit = if o.get(&format!("G.{e}.hue")) > V5_GATE_OPEN {
                if ctx.v6 {
                    wall.max(V5_CAP_DEG[k] + V5_GUARD_HUE_EXTRA_DEG)
                } else {
                    V5_CAP_DEG[k] + V5_GUARD_HUE_EXTRA_DEG
                }
            } else {
                wall
            };
            t.guard_hue += drift(wrap_deg(m.el[k].h - ctx.p.el[k].h), limit);
        }
    }
    for j in 0..SECTORS {
        if ctx.p.sec[j].share >= V5_GUARD_SECTOR_MIN_SHARE {
            t.guard_hue += drift(c.sec_shift[j], wall);
        }
    }
    if ctx.p.el[5].share >= V5_PRESENT_SHARE && m.el[5].has_hue {
        let h = m.el[5].h;
        let eh = (V5_GUARD_SKIN_HUE.0 - h)
            .max(h - V5_GUARD_SKIN_HUE.1)
            .max(0.0);
        let ec = (V5_GUARD_SKIN_CHROMA.0 - m.el[5].c)
            .max(m.el[5].c - V5_GUARD_SKIN_CHROMA.1)
            .max(0.0);
        let w = if ctx.v7 {
            V7_GUARD_SKIN_WEIGHT
        } else {
            V5_GUARD_SKIN_WEIGHT
        };
        t.guard_skin = w * (sq(eh / V5_GUARD_SKIN_SCALE) + sq(ec / V5_GUARD_SKIN_SCALE));
    }
    if ctx.v7 {
        let n_hl = ctx.hl_scope.iter().filter(|&&s| s).count();
        if n_hl as f64 >= V7_HL_MIN_SHARE * ctx.hl_scope.len() as f64 {
            let over = (c.hl_c - o.get("T.hl_cmax")).max(0.0);
            t.guard_hl = V7_HL_WEIGHT * sq(over / V7_HL_SCALE);
            if ctx.p_hl_c >= V7_HL_MIN_C
                && let (Some(ch), Some(ph)) = (c.hl_h, ctx.p_hl_h)
            {
                t.guard_hl += drift(wrap_deg(ch - ph), wall);
            }
        }
    }
    if ctx.v8 {
        let ok = 1.0 - o.get("T.purple_ok");
        // The groups: elements (>= 2 %), clothing (>= 2 %), sectors of unelemented pixels (>= 4 %) and the whites.
        let mut groups: Vec<(f64, f64, f64)> = Vec::new(); // (photo0 hue, photo0 median C*, candidate hue)
        for k in 0..6 {
            if ctx.p.el[k].share >= V5_PRESENT_SHARE && ctx.p.el[k].has_hue && m.el[k].has_hue {
                groups.push((ctx.p.el[k].h, ctx.p.el[k].c, m.el[k].h));
            }
        }
        if ctx.p.clothing.share >= V5_PRESENT_SHARE && ctx.p.clothing.has_hue && m.clothing.has_hue
        {
            groups.push((ctx.p.clothing.h, ctx.p.clothing.c, m.clothing.h));
            // Clothing is its own group for the hue-drift wall.
            t.guard_hue += drift(wrap_deg(m.clothing.h - ctx.p.clothing.h), wall);
        }
        for j in 0..SECTORS {
            if ctx.p.sec[j].share >= V5_GUARD_SECTOR_MIN_SHARE && ctx.p.sec[j].n > 0 {
                groups.push((
                    ctx.p.sec[j].hue,
                    ctx.p_sec_logc[j].exp(),
                    (ctx.p.sec[j].hue + c.sec_shift[j]).rem_euclid(360.0),
                ));
            }
        }
        let n_hl = ctx.hl_scope.iter().filter(|&&s| s).count();
        if n_hl as f64 >= V7_HL_MIN_SHARE * ctx.hl_scope.len() as f64
            && let (Some(ph), Some(ch)) = (ctx.p_hl_h, c.hl_h)
        {
            groups.push((ph, ctx.p_hl_c, ch));
        }
        let (lo, hi) = V8_PURPLE_BAND;
        let in_band = |h: f64| (lo..=hi).contains(&h.rem_euclid(360.0));
        let dist = |h: f64| {
            if in_band(h) {
                0.0
            } else {
                wrap_deg(h - lo).abs().min(wrap_deg(h - hi).abs())
            }
        };
        for (ph, pc, ch) in groups {
            if in_band(ph) || pc < V8_PURPLE_MIN_C {
                continue;
            }
            let ch = ch.rem_euclid(360.0);
            if in_band(ch) {
                let d_in = (ch - lo).min(hi - ch);
                t.guard_purple += V8_PURPLE_WEIGHT * sq(d_in / V8_PURPLE_SCALE_DEG) * ok;
            } else if dist(ch) < V8_APPROACH_REACH_DEG && dist(ch) < dist(ph) {
                t.guard_purple += V8_APPROACH_WEIGHT
                    * sq((V8_APPROACH_REACH_DEG - dist(ch)) / V5_GUARD_HUE_SCALE_DEG)
                    * ok;
            }
        }
        // The whites' direction: their mean (a*, b*) stays within `T.white_tol` of the reference-given target.
        if n_hl as f64 >= V7_HL_MIN_SHARE * ctx.hl_scope.len() as f64 {
            let d = (c.hl_a - o.get("T.white_a")).hypot(c.hl_b - o.get("T.white_b"));
            let tol = o.get("T.white_tol");
            if d > tol {
                t.guard_white_dir = V8_WHITE_DIR.0 * sq((d - tol) / V8_WHITE_DIR.1);
            }
        }
        // The skin drift wall: the skin hue may move at most `T.skin_drift` from photo0's own.
        if ctx.p.el[5].share >= V5_PRESENT_SHARE && ctx.p.el[5].has_hue && m.el[5].has_hue {
            let excess =
                (wrap_deg(m.el[5].h - ctx.p.el[5].h).abs() - o.get("T.skin_drift")).max(0.0);
            t.guard_skin_drift = V8_SKIN_DRIFT.0 * sq(excess / V8_SKIN_DRIFT.1);
        }
    }
    if ctx.p.el[0].share >= V5_PRESENT_SHARE {
        let limit = (V5_GUARD_SKY_FACTOR * ctx.p.el[0].c).max(o.get("T.sky.logC").exp());
        let excess = (m.el[0].c - limit).max(0.0);
        t.guard_sky = V5_GUARD_SKY_WEIGHT * sq(excess / V5_GUARD_SKY_SCALE);
    }
    t
}

// ---- V5 local grades --------------------------------------------------------------------------

/// The local grades of one candidate: per element (sky, vegetation, water, ground, built, skin, subject) an exposure offset in
/// stops, a hue rotation in degrees and a chroma scale. The subject has the exposure offset only.
#[derive(Clone, Debug, Default)]
struct LocalSet {
    exp: [f64; 7],
    hue: [f64; 7],
    chroma: [f64; 7],
}

impl LocalSet {
    fn is_identity(&self) -> bool {
        (0..7).all(|k| self.exp[k] == 0.0 && self.hue[k] == 0.0 && self.chroma[k] == 1.0)
    }
}

/// The local block of the normalised parameters (`N_GLOBAL..`), scaled by `strength`: 3 per element, then the subject's exposure.
fn local_from_x(x: &[f64; N_PARAMS], strength: f64) -> LocalSet {
    let mut s = LocalSet {
        chroma: [1.0; 7],
        ..LocalSet::default()
    };
    for k in 0..6 {
        let b = N_GLOBAL + 3 * k;
        s.exp[k] = strength * x[b];
        s.hue[k] = strength * x[b + 1] * V5_CAP_DEG[k];
        let c = strength * x[b + 2];
        s.chroma[k] = if c >= 0.0 {
            V5_CHROMA_SCALE_MAX.powf(c)
        } else {
            V5_CHROMA_SCALE_MIN.powf(-c)
        };
    }
    s.exp[6] = strength * x[N_PARAMS - 1];
    s
}

/// One local grade on a LogC4 triple: exposure, then a rotation and a scale of the chroma in the `e1, e2` plane.
fn local_apply(v: [f64; 3], exp_stops: f64, hue_deg: f64, chroma: f64) -> [f64; 3] {
    let mean = (v[0] + v[1] + v[2]) / 3.0;
    let p = v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2];
    let q = v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2];
    let (s, c) = hue_deg.to_radians().sin_cos();
    let (p2, q2) = (chroma * (p * c - q * s), chroma * (p * s + q * c));
    let shift = exp_stops * V5_STOP_CODE;
    [
        mean + shift + p2 * E1[0] + q2 * E2[0],
        mean + shift + p2 * E1[1] + q2 * E2[1],
        mean + shift + p2 * E1[2] + q2 * E2[2],
    ]
}

/// Soft masks per element (sky, vegetation, water, ground, built, skin, subject) over a picture.
#[allow(dead_code)]
struct MaskSet {
    w: usize,
    h: usize,
    data: Vec<[f32; 7]>,
}

/// Box mean with edge clamping, radius `r`, by prefix sums.
fn box_mean(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f64; w * h];
    for y in 0..h {
        let mut pre = vec![0f64; w + 1];
        for x in 0..w {
            pre[x + 1] = pre[x] + f64::from(src[y * w + x]);
        }
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
            tmp[y * w + x] = (pre[b] - pre[a]) / (b - a) as f64;
        }
    }
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        let mut pre = vec![0f64; h + 1];
        for y in 0..h {
            pre[y + 1] = pre[y] + tmp[y * w + x];
        }
        for y in 0..h {
            let (a, b) = (y.saturating_sub(r), (y + r + 1).min(h));
            out[y * w + x] = ((pre[b] - pre[a]) / (b - a) as f64) as f32;
        }
    }
    out
}

/// The soft masks: each element's binary mask refined by a guided filter on P's L*/100 (radius 2 % of the short side,
/// epsilon 1e-3), clamped to 0..1. `binary` is per pixel the 7 binary masks.
fn soft_masks(binary: &[[bool; 7]], guide: &[f32], w: usize, h: usize) -> MaskSet {
    let r = ((V5_GF_RADIUS * w.min(h) as f64).round() as usize).max(1);
    let mean_i = box_mean(guide, w, h, r);
    let ii: Vec<f32> = guide.iter().map(|v| v * v).collect();
    let corr_i = box_mean(&ii, w, h, r);
    let mut data = vec![[0f32; 7]; w * h];
    for k in 0..7 {
        let p: Vec<f32> = binary
            .iter()
            .map(|b| if b[k] { 1.0 } else { 0.0 })
            .collect();
        if p.iter().all(|&v| v == 0.0) {
            continue;
        }
        let mean_p = box_mean(&p, w, h, r);
        let ip: Vec<f32> = guide.iter().zip(&p).map(|(a, b)| a * b).collect();
        let corr_ip = box_mean(&ip, w, h, r);
        let mut a = vec![0f32; w * h];
        let mut b = vec![0f32; w * h];
        for i in 0..w * h {
            let var = corr_i[i] - mean_i[i] * mean_i[i];
            let cov = corr_ip[i] - mean_i[i] * mean_p[i];
            a[i] = cov / (var + V5_GF_EPS as f32);
            b[i] = mean_p[i] - a[i] * mean_i[i];
        }
        let (ma, mb) = (box_mean(&a, w, h, r), box_mean(&b, w, h, r));
        for i in 0..w * h {
            data[i][k] = (ma[i] * guide[i] + mb[i]).clamp(0.0, 1.0);
        }
    }
    MaskSet { w, h, data }
}

/// Nearest-neighbour resize of a per-pixel array.
fn resize_nn<T: Copy>(src: &[T], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        let sy = (((y as f64 + 0.5) / dh as f64) * sh as f64) as usize;
        for x in 0..dw {
            let sx = (((x as f64 + 0.5) / dw as f64) * sw as f64) as usize;
            out.push(src[sy.min(sh - 1) * sw + sx.min(sw - 1)]);
        }
    }
    out
}

/// The 7 binary masks of a picture from its labels.
fn binary_masks(scene: &Scene) -> Vec<[bool; 7]> {
    (0..scene.elem.len())
        .map(|i| {
            let mut b = [false; 7];
            for (k, flag) in b.iter_mut().enumerate().take(6) {
                *flag = in_elem(scene, i, k);
            }
            b[6] = scene.subject[i];
            b
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Samples, evaluation, optimiser
// ---------------------------------------------------------------------------------------------

/// A fixed stratified sample of a node picture: one pixel per cell of a grid, jittered by a seeded generator.
fn stratified_sample(img: &Img, grid: (usize, usize), seed: u64) -> (Vec<[f64; 3]>, Vec<usize>) {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(grid.0 * grid.1);
    let mut idx = Vec::with_capacity(grid.0 * grid.1);
    for gy in 0..grid.1 {
        for gx in 0..grid.0 {
            let fx = (gx as f64 + rng.unit()) / grid.0 as f64;
            let fy = (gy as f64 + rng.unit()) / grid.1 as f64;
            let x = ((fx * img.w as f64) as usize).min(img.w - 1);
            let y = ((fy * img.h as f64) as usize).min(img.h - 1);
            out.push(img.at(x, y));
            idx.push(y * img.w + x);
        }
    }
    (out, idx)
}

/// Evaluates candidates on a fixed sample.
struct Evaluator<'a> {
    output: &'a ChainParamsOut,
    lin: Vec<[f64; 3]>,
    reference: Stats,
    objective: Objective,
    /// The ungraded photo's statistics (V2's anchor, V3's base of every move).
    photo0: Stats,
    /// V3: the photo's scene at the sample pixels.
    scene: Option<Scene>,
    /// V3: what the do-no-harm term needs from photo0 (set once photo0 and the reference are known).
    dnh: Option<DnhCtx>,
    /// V5: the network's answer and the fixed measurements of the pair.
    v5: Option<V5Ctx>,
    /// V5: the photo's soft masks at the sample pixels, and which elements have a mask of at least 2 % of the frame.
    masks: Option<Vec<[f32; 7]>>,
    mask_ok: [bool; 7],
    /// The V2 term groups the current stage minimises.
    mask: u8,
    conv: LabConv,
    with_hue: bool,
    strength: f64,
    evals: usize,
    buf: Vec<Px>,
    eval_time: Duration,
}

type ChainParamsOut = ColorChain;

impl Evaluator<'_> {
    /// Renders the sample pixels of a candidate into `buf`; false if the chain refused it.
    fn fill(&mut self, x: &[f64; N_PARAMS]) -> bool {
        let real = to_real(x, self.strength);
        match Candidate::new(&real, self.with_hue) {
            Ok(mut cand) => {
                if self.v5.is_some() {
                    let loc = local_from_x(x, self.strength);
                    if !loc.is_identity() {
                        cand.local = Some(loc);
                    }
                }
                let (output, conv, masks) = (self.output, &self.conv, self.masks.as_deref());
                self.buf
                    .par_chunks_mut(256)
                    .zip(self.lin.par_chunks(256))
                    .enumerate()
                    .for_each(|(ci, (o, i))| {
                        for (j, (d, s)) in o.iter_mut().zip(i).enumerate() {
                            let m = masks.map(|ms| &ms[ci * 256 + j]);
                            *d = conv.px(render_pixel_m(&cand, output, *s, m));
                        }
                    });
                true
            }
            Err(_) => false,
        }
    }

    fn stats(&mut self, x: &[f64; N_PARAMS]) -> Stats {
        if self.fill(x) {
            stats_of(&self.buf, self.scene.as_ref(), self.dnh.as_ref())
        } else {
            // A candidate the chain refuses is worse than anything.
            let mut s = stats_of(&self.buf, self.scene.as_ref(), self.dnh.as_ref());
            s.median = 1e6;
            s
        }
    }

    /// V5: the candidate's measurements and terms.
    fn v5_terms(&mut self, x: &[f64; N_PARAMS]) -> (Terms, Option<CandMeas>) {
        let ok = self.fill(x);
        let reg = x[..N_GLOBAL].iter().map(|v| v * v).sum::<f64>() / N_GLOBAL as f64;
        let reg_local =
            x[N_GLOBAL..].iter().map(|v| v * v).sum::<f64>() / (N_PARAMS - N_GLOBAL) as f64;
        let mut t = blank_terms();
        t.v5 = true;
        t.reg = reg;
        t.reg_local = reg_local;
        let (Some(ctx), Some(scene)) = (self.v5.as_ref(), self.scene.as_ref()) else {
            return (t, None);
        };
        if !ok {
            t.clip = 1e6;
            return (t, None);
        }
        let cm = cand_meas(&self.buf, scene, ctx);
        t.t5 = terms_v5(&cm, ctx);
        t.clip = clip_term(ctx, &cm.m);
        (t, Some(cm))
    }

    fn objective(&mut self, x: &[f64; N_PARAMS]) -> (f64, Terms) {
        let t0 = Instant::now();
        let t = if self.objective.net() {
            self.v5_terms(x).0
        } else {
            let s = self.stats(x);
            terms(self.objective, &s, &self.reference, &self.photo0, x)
        };
        self.evals += 1;
        self.eval_time += t0.elapsed();
        (t.total_masked(self.mask), t)
    }
}

/// A `Terms` with every term at 0.
fn blank_terms() -> Terms {
    let z = Stats::default();
    terms(Objective::V1, &z, &z, &z, &[0.0; N_PARAMS])
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
    /// V5: the network's answer, the need test, the local grades, the guards that fired, and the text of the sheet's panel.
    extra: Value,
}

fn fit(ev: &mut Evaluator<'_>) -> FitResult {
    if ev.objective.net() {
        return fit_v5(ev);
    }
    let t0 = Instant::now();
    let strength = ev.strength;
    // The fit itself runs at strength 1 (the deltas are scaled afterwards by `strength`, see `to_real`).
    ev.strength = 1.0;
    let mut x = [0.0f64; N_PARAMS];
    let (_, before) = ev.objective(&x);
    let mut f: f64;
    let mut log = Vec::new();
    for (name, range, frac, iters) in STAGES.iter().cloned() {
        ev.mask = match name {
            // V3 (notes section 4): tone sees 3.3, the colour stages 3.1 + 3.2 + 3.4, hue-vs-hue 3.1.
            "tone" if ev.objective == Objective::V3 => MASK_BASE | MASK_TONE,
            "colour per band" | "saturation and hue-vs-sat" if ev.objective == Objective::V3 => {
                MASK_BASE | MASK_HUE | MASK_CAST | MASK_CHROMA
            }
            "hue-vs-hue" if ev.objective == Objective::V3 => MASK_BASE | MASK_HUE,
            "tone" => MASK_TONE,
            "colour per band" => MASK_TONE | MASK_CAST,
            "saturation and hue-vs-sat" => MASK_TONE | MASK_CAST | MASK_CHROMA,
            _ => MASK_ALL,
        };
        f = ev.objective(&x).0;
        let start = f;
        for i in range {
            if ev.with_hue || !(HSAT_RANGE.contains(&i) || HHUE_RANGE.contains(&i)) {
                f = golden(ev, &mut x, i, frac, iters, f);
            }
        }
        log.push(json!({ "stage": name, "half_range": frac, "iterations": iters, "objective_before": start, "objective_after": f }));
    }
    let seconds = t0.elapsed().as_secs_f64();
    ev.mask = MASK_ALL;
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
        extra: Value::Null,
    }
}

/// The V5 need test (`09` section 5): which elements get a local grade after the global stages.
fn v5_need(ev: &mut Evaluator<'_>, x: &[f64; N_PARAMS]) -> ([bool; 7], Vec<Value>) {
    let (_, cm) = ev.v5_terms(x);
    let mask_ok = ev.mask_ok;
    let mut need = [false; 7];
    let mut report = Vec::new();
    let (Some(cm), Some(ctx)) = (cm, ev.v5.as_ref()) else {
        return (need, report);
    };
    let o = &ctx.out;
    for (k, e) in V5_ELEMS.iter().enumerate() {
        let matched = o.get(&format!("G.{e}.matched")) > V5_GATE_OPEN;
        let hue_on = o.get(&format!("G.{e}.hue")) > V5_GATE_OPEN;
        let exp_on = o.get(&format!("{e}.e_gate")) > V5_GATE_OPEN;
        let em = &cm.m.el[k];
        let hue_err = if hue_on && em.has_hue {
            wrap_deg(em.h - o.get(&format!("T.{e}.h"))).abs()
        } else {
            0.0
        };
        let logc_err = if matched {
            (em.logc - o.get(&format!("T.{e}.logC"))).abs()
        } else {
            0.0
        };
        let exp_err = if exp_on {
            ((em.l - cm.m.p50) - o.get(&format!("T.{e}.off"))).abs()
        } else {
            0.0
        };
        let want = matched
            && (hue_err > V5_NEED_HUE_DEG
                || logc_err > V5_NEED_LOGC
                || exp_err > V5_NEED_EXPOSURE_L);
        need[k] = want && mask_ok[k];
        report.push(json!({ "element": e, "matched": matched, "mask_at_least_2_percent": mask_ok[k], "hue_error_deg": hue_err,
            "log_chroma_error": logc_err, "exposure_error_L": exp_err, "local_grade": need[k] }));
    }
    let sep_err = if o.get("G.subj") > V5_GATE_OPEN {
        (cm.m.sep - o.get("T.sep")).abs()
    } else {
        0.0
    };
    need[6] = sep_err > V5_NEED_EXPOSURE_L && mask_ok[6];
    report.push(
        json!({ "element": "subject", "separation_error_L": sep_err, "local_grade": need[6] }),
    );
    (need, report)
}

/// The V5 fit: the global stages, the need test, the local stage, the joint polish.
fn fit_v5(ev: &mut Evaluator<'_>) -> FitResult {
    let t0 = Instant::now();
    let strength = ev.strength;
    ev.strength = 1.0;
    let mut x = [0.0f64; N_PARAMS];
    ev.mask = MASK_ALL;
    let (_, before) = ev.objective(&x);
    let mut log = Vec::new();
    let mut f: f64;
    for (name, range, frac, iters) in STAGES[..8].iter().cloned() {
        // Every stage sees the whole objective (a stage only chooses which parameters it moves), so each golden-section pass
        // can only lower it: the fit never ends above where it began.
        ev.mask = MASK_ALL;
        f = ev.objective(&x).0;
        let start = f;
        for i in range {
            if ev.with_hue || !(HSAT_RANGE.contains(&i) || HHUE_RANGE.contains(&i)) {
                f = golden(ev, &mut x, i, frac, iters, f);
            }
        }
        log.push(json!({ "stage": name, "half_range": frac, "iterations": iters, "objective_before": start, "objective_after": f }));
    }
    ev.mask = MASK_ALL;
    let (f_global, after_global) = ev.objective(&x);
    let (need, need_report) = v5_need(ev, &x);
    let mut local_idx: Vec<usize> = Vec::new();
    for (k, &n) in need.iter().enumerate() {
        if n {
            if k < 6 {
                local_idx.extend([N_GLOBAL + 3 * k, N_GLOBAL + 3 * k + 1, N_GLOBAL + 3 * k + 2]);
            } else {
                local_idx.push(N_PARAMS - 1);
            }
        }
    }
    f = f_global;
    for (pass, (frac, iters)) in V5_LOCAL_PASSES.iter().cloned().enumerate() {
        let start = f;
        for &i in &local_idx {
            f = golden(ev, &mut x, i, frac, iters, f);
        }
        log.push(json!({ "stage": "local", "pass": pass, "half_range": frac, "iterations": iters, "objective_before": start, "objective_after": f }));
    }
    let start = f;
    let polish: Vec<usize> = (0..N_GLOBAL).chain(local_idx.iter().copied()).collect();
    for i in polish {
        if ev.with_hue || !(HSAT_RANGE.contains(&i) || HHUE_RANGE.contains(&i)) {
            f = golden(ev, &mut x, i, V5_POLISH.0, V5_POLISH.1, f);
        }
    }
    log.push(json!({ "stage": "joint polish", "half_range": V5_POLISH.0, "iterations": V5_POLISH.1, "objective_before": start, "objective_after": f }));
    let seconds = t0.elapsed().as_secs_f64();
    let (_, after) = ev.objective(&x);
    let extra = v5_extra(ev, &x, &need, need_report, &after_global, &after);
    ev.strength = strength;
    FitResult {
        x,
        before,
        after,
        evals: ev.evals,
        seconds,
        per_eval_ms: ev.eval_time.as_secs_f64() * 1000.0 / ev.evals.max(1) as f64,
        stage_log: log,
        extra,
    }
}

/// What the V5 report says about a fit: the network's inputs, outputs and gates, the need test, the local grades, the guards.
fn v5_extra(
    ev: &Evaluator<'_>,
    x: &[f64; N_PARAMS],
    need: &[bool; 7],
    need_report: Vec<Value>,
    after_global: &Terms,
    after: &Terms,
) -> Value {
    let Some(ctx) = ev.v5.as_ref() else {
        return Value::Null;
    };
    let o = &ctx.out;
    let loc = local_from_x(x, 1.0);
    let names = [
        "sky",
        "vegetation",
        "water",
        "ground",
        "built",
        "skin",
        "subject",
    ];
    let local: Vec<Value> = (0..7)
        .filter(|&k| need[k])
        .map(|k| json!({ "element": names[k], "exposure_stops": loc.exp[k], "hue_deg": loc.hue[k], "chroma_scale": loc.chroma[k] }))
        .collect();
    let mut lines = vec![format!(
        "key mismatch: {}",
        if o.get("G.key_mismatch") > V5_GATE_OPEN {
            "yes"
        } else {
            "no"
        }
    )];
    let short = ["sky", "veg", "water", "ground", "built", "skin"];
    for (k, e) in V5_ELEMS.iter().enumerate() {
        if o.get(&format!("G.{e}.matched")) > V5_GATE_OPEN {
            let pm = &ctx.p.el[k];
            lines.push(format!(
                "{}: h {:.0}>{:.0} c {:.0}>{:.0} e {:+.1}",
                short[k],
                pm.h,
                o.get(&format!("T.{e}.h")).rem_euclid(360.0),
                pm.c,
                o.get(&format!("T.{e}.logC")).exp(),
                o.get(&format!("T.{e}.off")) - (pm.l - ctx.p.p50)
            ));
        }
    }
    if o.names.iter().any(|n| n == "G.unify") {
        lines.push(format!(
            "unify: {:.0}  wall {:.0} deg",
            o.get("G.unify"),
            o.get("T.wall_deg")
        ));
    }
    let open: Vec<String> = (0..SECTORS)
        .filter(|&j| o.get(&format!("G.s{j}")) > V5_GATE_OPEN)
        .map(|j| j.to_string())
        .collect();
    lines.push(format!(
        "sectors open: {}",
        if open.is_empty() {
            "none".into()
        } else {
            open.join(" ")
        }
    ));
    let mut any = false;
    for k in 0..7 {
        if need[k] {
            any = true;
            lines.push(if k == 6 {
                format!("local subject: {:+.2} st", loc.exp[k])
            } else {
                format!(
                    "local {}: {:+.2} st, {:+.0}*, x{:.2}",
                    short[k], loc.exp[k], loc.hue[k], loc.chroma[k]
                )
            });
        }
    }
    if !any {
        lines.push("local grades: none".into());
    }
    let g = |t: &Terms| {
        let t5 = t.t5;
        json!({ "guard_hue_drift": t5.guard_hue, "guard_skin": t5.guard_skin, "guard_sky_chroma": t5.guard_sky, "guard_highlights": t5.guard_hl, "guard_purple": t5.guard_purple, "guard_white_direction": t5.guard_white_dir, "guard_skin_drift": t5.guard_skin_drift, "guard_memory": t5.guard_mem, "guard_purple_pixels": t5.guard_purple_px, "guard_skin_chroma": t5.guard_skin_chroma, "guard_turn": t5.guard_turn })
    };
    json!({
        "newly_purple_share": after.t5.np_share,
        "skin_dlogC": after.t5.skin_dlogc,
        "skin_chroma_ratio": after.t5.skin_dlogc.exp(),
        "network_inputs": ctx.inputs,
        "network": o.outputs_json(),
        "excluded_from_sectors": ctx.excl,
        "need_test": need_report,
        "local_grades": local,
        "guards_after": g(after),
        "terms_after_global_stages": after_global.to_json(),
        "panel_lines": lines,
        "note": "The local grades are a POC-only CPU step (like the hue step) and cannot be written to settings.json; its settings.json holds the global grade's real corrector settings only.",
    })
}

// ---------------------------------------------------------------------------------------------
// Rendering, Reinhard, sheets
// ---------------------------------------------------------------------------------------------

/// Renders a node picture through a candidate and the Output: linear sRGB (clamped).
fn render_image(node: &Img, cand: &Candidate, output: &ColorChain) -> Img {
    render_image_m(node, cand, output, None)
}

/// [`render_image`] with soft masks (of the picture's size) for the local grades.
fn render_image_m(
    node: &Img,
    cand: &Candidate,
    output: &ColorChain,
    masks: Option<&MaskSet>,
) -> Img {
    let mut px = vec![0f32; node.px.len()];
    px.par_chunks_mut(3 * node.w)
        .zip(node.px.par_chunks(3 * node.w))
        .enumerate()
        .for_each(|(y, (o, i))| {
            for (x, (d, s)) in o.chunks_exact_mut(3).zip(i.chunks_exact(3)).enumerate() {
                let m = masks.map(|ms| &ms.data[y * ms.w + x]);
                let v = render_pixel_m(
                    cand,
                    output,
                    [f64::from(s[0]), f64::from(s[1]), f64::from(s[2])],
                    m,
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
        'g' => [0, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'j' => [0b00010, 0, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
        'k' => [
            0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010,
        ],
        'p' => [0, 0, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
        'q' => [0, 0, 0b01111, 0b10001, 0b01111, 0b00001, 0b00001],
        'v' => [0, 0, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'x' => [0, 0, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
        'y' => [0, 0, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'z' => [0, 0, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        ':' => [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
        ',' => [0, 0, 0, 0, 0b01100, 0b00100, 0b01000],
        '>' => [
            0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000,
        ],
        '*' => [0b01100, 0b10010, 0b10010, 0b01100, 0, 0, 0],
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
    /// V3: the labels at the fit picture's pixels, when a label folder is given and has this picture.
    scene: Option<Scene>,
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
    let scene = match &ctx().labels {
        Some(dir) => {
            // `<film>__<stem>.png`, the film being the folder the still lives in (symlinks followed).
            let real = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            let group = real
                .parent()
                .and_then(|d| d.file_name())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let file = dir.join("labels").join(format!("{group}__{stem}.png"));
            if file.exists() {
                let (lw, lh, labels) = read_label_png(&file)?;
                if (lw, lh) != (raw.w, raw.h) {
                    return Err(format!(
                        "{}: labels are {lw}x{lh}, the picture {}x{}",
                        file.display(),
                        raw.w,
                        raw.h
                    )
                    .into());
                }
                let elem = resize_labels(&labels, lw, lh, bars, fit.w, fit.h);
                Some(make_scene(elem, &fit, ctx().seg.as_ref(), &LabConv::new()))
            } else {
                None
            }
        }
        None => None,
    };
    Ok(Reference {
        stem,
        show8,
        fit,
        bars,
        scene,
    })
}

fn picture_stats(conv: &LabConv, img: &Img) -> Stats {
    picture_stats_scene(conv, img, None)
}

fn picture_stats_scene(conv: &LabConv, img: &Img, scene: Option<&Scene>) -> Stats {
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
    stats_of(&px, scene, None)
}

/// One photo prepared for fitting against many references.
struct PhotoFit<'a> {
    photo: &'a PhotoNode,
    output: ColorChain,
    fit_img: Img,
    lin_sample: Vec<[f64; 3]>,
    default_show: Img,
    /// V3: the photo's labels at the fit picture's pixels and gathered at the sample pixels.
    scene_fit: Option<Scene>,
    scene_samples: Option<Scene>,
    /// V5: the soft masks of the local grades at the fit and the render sizes, which elements have a mask of at least 2 % of
    /// the frame, and the pixel index of every sample.
    masks_fit: Option<MaskSet>,
    masks_show: Option<MaskSet>,
    mask_ok: [bool; 7],
    sample_idx: Vec<usize>,
}

fn prepare_photo(photo: &PhotoNode) -> R<PhotoFit<'_>> {
    prepare_photo_with(photo, ctx().labels.as_deref())
}

/// [`prepare_photo`] with the label folder given (the evaluation harness takes it per photo from the split file).
fn prepare_photo_with<'a>(photo: &'a PhotoNode, labels: Option<&Path>) -> R<PhotoFit<'a>> {
    let output = make_output_chain(&photo.output)?;
    let (fw, fh) = fit_size(photo.show.w, photo.show.h, FIT_LONG_EDGE_PX);
    let fit_img = resize_area(&photo.show, fw, fh);
    let (lin_sample, sample_idx) = stratified_sample(&fit_img, SAMPLE_GRID, SEED);
    let ident = Candidate::new(&to_real(&[0.0; N_PARAMS], 1.0), false)?;
    let default_show = render_image(&photo.show, &ident, &output);
    let (mut scene_fit, mut scene_samples) = (None, None);
    let (mut masks_fit, mut masks_show, mut mask_ok) = (None, None, [false; 7]);
    if let Some(dir) = labels {
        let file = dir
            .join("labels")
            .join(format!("photo__{}.png", photo.name));
        if file.exists() {
            let (lw, lh, labels) = read_label_png(&file)?;
            let elem = resize_labels(&labels, lw, lh, [0; 4], fit_img.w, fit_img.h);
            let display = resize_area(&default_show, fit_img.w, fit_img.h);
            let conv = LabConv::new();
            let sc = make_scene(elem, &display, ctx().seg.as_ref(), &conv);
            // V5's soft masks: the binary masks refined by a guided filter on P's L*/100, at the fit and the render sizes.
            let guide = |img: &Img| -> Vec<f32> {
                (0..img.w * img.h)
                    .map(|i| {
                        let p = &img.px[3 * i..3 * i + 3];
                        (conv
                            .px([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
                            .lab[0]
                            / 100.0) as f32
                    })
                    .collect()
            };
            let binary_fit = binary_masks(&sc);
            for (k, ok) in mask_ok.iter_mut().enumerate() {
                let n = binary_fit.iter().filter(|b| b[k]).count();
                *ok = n as f64 / binary_fit.len() as f64 >= V5_PRESENT_SHARE;
            }
            masks_fit = Some(soft_masks(
                &binary_fit,
                &guide(&display),
                fit_img.w,
                fit_img.h,
            ));
            let (sw, sh) = (photo.show.w, photo.show.h);
            let show_scene = Scene {
                elem: resize_labels(&labels, lw, lh, [0; 4], sw, sh),
                skin: resize_nn(&sc.skin, fit_img.w, fit_img.h, sw, sh),
                subject: resize_nn(&sc.subject, fit_img.w, fit_img.h, sw, sh),
                skin_source: sc.skin_source,
            };
            masks_show = Some(soft_masks(
                &binary_masks(&show_scene),
                &guide(&default_show),
                sw,
                sh,
            ));
            scene_samples = Some(gather_scene(&sc, &sample_idx));
            scene_fit = Some(sc);
        }
    }
    Ok(PhotoFit {
        photo,
        output,
        fit_img,
        lin_sample,
        default_show,
        scene_fit,
        scene_samples,
        masks_fit,
        masks_show,
        mask_ok,
        sample_idx,
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
    objective: Objective,
    ref_scene: Option<&Scene>,
) -> R<(FitResult, Stats)> {
    let conv = LabConv::new();
    let v5 = objective.net();
    let v3 = objective == Objective::V3 || v5;
    if v3 && (ref_scene.is_none() || pf.scene_samples.is_none()) {
        return Err(
            "objective v3 and v5 need labels for the photo and the reference (--labels)".into(),
        );
    }
    let ref_stats = picture_stats_scene(&conv, reference_fit, if v3 { ref_scene } else { None });
    let mut ev = Evaluator {
        output: &pf.output,
        lin: pf.lin_sample.clone(),
        reference: ref_stats.clone(),
        objective,
        photo0: ref_stats.clone(),
        scene: if v3 { pf.scene_samples.clone() } else { None },
        dnh: None,
        v5: None,
        masks: if v5 {
            pf.masks_fit
                .as_ref()
                .map(|m| pf.sample_idx.iter().map(|&i| m.data[i]).collect())
        } else {
            None
        },
        mask_ok: pf.mask_ok,
        mask: MASK_ALL,
        conv,
        with_hue,
        strength,
        evals: 0,
        buf: vec![Px::default(); pf.lin_sample.len()],
        eval_time: Duration::ZERO,
    };
    ev.photo0 = ev.stats(&[0.0; N_PARAMS]);
    if let (true, Some(sc), Some(r), Some(z)) = (
        objective == Objective::V3,
        ev.scene.as_ref(),
        &ref_stats.v3,
        &ev.photo0.v3,
    ) {
        let elig = eligibility(r, z);
        let sector = dnh_sectors(&ev.buf, sc, &elig);
        ev.dnh = Some(DnhCtx {
            p0_px: ev.buf.clone(),
            sector,
        });
    }
    if v5 {
        // V5 runs next to V6 (for the Round 5 panel) read their own network, `--network5`.
        let net = if objective == Objective::V5 && ctx().network5.is_some() {
            ctx().network5.as_ref()
        } else if objective == Objective::V6 && ctx().network6.is_some() {
            ctx().network6.as_ref()
        } else if objective == Objective::V7 && ctx().network7.is_some() {
            ctx().network7.as_ref()
        } else if objective == Objective::V8 && ctx().network8.is_some() {
            ctx().network8.as_ref()
        } else if objective == Objective::V11 && ctx().network11.is_some() {
            ctx().network11.as_ref()
        } else if objective == Objective::V10 && ctx().network10.is_some() {
            ctx().network10.as_ref()
        } else if objective == Objective::V9 && ctx().network9.is_some() {
            ctx().network9.as_ref()
        } else {
            ctx().network.as_ref()
        }
        .ok_or("objective v5 and v6 need --network <file.ywn>")?;
        let (Some(sc), Some(rs)) = (ev.scene.as_ref(), ref_scene) else {
            return Err("objective v5 needs labels".into());
        };
        let p_px = ev.buf.clone();
        let r_px = picture_px(&ev.conv, reference_fit);
        let c = build_v5_ctx(net, &p_px, sc, &r_px, rs, ctx().ablate, objective)?;
        ev.v5 = Some(c);
    }
    Ok((fit(&mut ev), ref_stats))
}

/// Every pixel of a linear sRGB picture, measured.
fn picture_px(conv: &LabConv, img: &Img) -> Vec<Px> {
    (0..img.w * img.h)
        .map(|i| {
            let p = &img.px[3 * i..3 * i + 3];
            conv.px([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        })
        .collect()
}

/// The fitted picture: the global grade, and for V5 the local grades through the soft masks, at the fit or the render size.
fn render_fitted(
    pf: &PhotoFit<'_>,
    x: &[f64; N_PARAMS],
    strength: f64,
    with_hue: bool,
    show: bool,
    v5: bool,
) -> R<Img> {
    let mut cand = Candidate::new(&to_real(x, strength), with_hue)?;
    let (node, masks) = if show {
        (&pf.photo.show, pf.masks_show.as_ref())
    } else {
        (&pf.fit_img, pf.masks_fit.as_ref())
    };
    if v5 {
        let loc = local_from_x(x, strength);
        if !loc.is_identity() {
            cand.local = Some(loc);
        }
    }
    Ok(render_image_m(
        node,
        &cand,
        &pf.output,
        if v5 { masks } else { None },
    ))
}

fn run_pair(
    pf: &PhotoFit<'_>,
    reference: &Reference,
    strength: f64,
    with_hue: bool,
    objective: Objective,
) -> R<PairOutcome> {
    let t0 = Instant::now();
    let (fit_res, _) = run_fit(
        pf,
        &reference.fit,
        strength,
        with_hue,
        objective,
        reference.scene.as_ref(),
    )?;
    let mut x_scaled = fit_res.x;
    for v in &mut x_scaled {
        *v *= strength;
    }
    let real = to_real(&fit_res.x, strength);
    let fitted_show = render_fitted(pf, &fit_res.x, strength, with_hue, true, objective.net())?;
    Ok(PairOutcome {
        fitted_show,
        fit: fit_res,
        real,
        x_scaled,
        wall_total: t0.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::too_many_arguments)]
fn write_pair(
    dir: &Path,
    photo_name: &str,
    pf: &PhotoFit<'_>,
    reference: &Reference,
    o: &PairOutcome,
    strength: f64,
    with_hue: bool,
    objective: Objective,
    round2: Option<&PairOutcome>,
    round5: Option<&PairOutcome>,
    round6: Option<&PairOutcome>,
    round7: Option<&PairOutcome>,
    round8: Option<&PairOutcome>,
    round9: Option<&PairOutcome>,
    round10: Option<&PairOutcome>,
) -> R<()> {
    fs::create_dir_all(dir)?;
    let default8 = linear_to_rgb8(&pf.default_show);
    let fitted8 = linear_to_rgb8(&o.fitted_show);
    let ref_show_lin = rgb8_to_linear(&reference.show8);
    let reinhard8 = linear_to_rgb8(&reinhard(&pf.default_show, &ref_show_lin));
    let aspect = pf.default_show.w as f64 / pf.default_show.h as f64;
    if matches!(
        objective,
        Objective::V3
            | Objective::V5
            | Objective::V6
            | Objective::V7
            | Objective::V8
            | Objective::V9
            | Objective::V10
            | Objective::V11
            | Objective::V12
            | Objective::V13
    ) && (round2.is_some()
        || round5.is_some()
        || round6.is_some()
        || round7.is_some()
        || round8.is_some()
        || round9.is_some()
        || round10.is_some())
    {
        let r2_8 = round2.map(|r| linear_to_rgb8(&r.fitted_show));
        let r5_8 = round5.map(|r| linear_to_rgb8(&r.fitted_show));
        let r6_8 = round6.map(|r| linear_to_rgb8(&r.fitted_show));
        let r7_8 = round7.map(|r| linear_to_rgb8(&r.fitted_show));
        let r8_8 = round8.map(|r| linear_to_rgb8(&r.fitted_show));
        let r9_8 = round9.map(|r| linear_to_rgb8(&r.fitted_show));
        let r10_8 = round10.map(|r| linear_to_rgb8(&r.fitted_show));
        let mut rounds: Vec<(&str, &Rgb8)> = Vec::new();
        if let Some(r) = &r2_8 {
            rounds.push(("Round 2", r));
        }
        if let Some(r) = &r5_8 {
            rounds.push(("Round 5", r));
        }
        if let Some(r) = &r6_8 {
            rounds.push(("Round 6", r));
        }
        if let Some(r) = &r7_8 {
            rounds.push(("Round 7", r));
        }
        if let Some(r) = &r8_8 {
            rounds.push(("Round 8", r));
        }
        if let Some(r) = &r9_8 {
            rounds.push(("Round 9", r));
        }
        if let Some(r) = &r10_8 {
            rounds.push(("Round 10", r));
        }
        rounds.push((
            match objective {
                Objective::V13 => "Round 13",
                Objective::V12 => "Round 12",
                Objective::V11 => "Round 11",
                Objective::V10 => "Round 10",
                Objective::V9 => "Round 9",
                Objective::V8 => "Round 8",
                Objective::V7 => "Round 7",
                Objective::V6 => "Round 6",
                Objective::V5 => "Round 5",
                _ => "Round 3",
            },
            &fitted8,
        ));
        write_sheet_v3(
            &dir.join("sheet.jpg"),
            reference,
            &default8,
            &rounds,
            pf,
            aspect,
            o.fit.extra["panel_lines"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
        )?;
    } else {
        write_sheet(
            &dir.join("sheet.jpg"),
            &reference.show8,
            [&default8, &fitted8, &reinhard8],
            aspect,
        )?;
    }
    save_rgb8_jpeg(&fitted8, &dir.join("fitted.jpg"))?;
    save_rgb8_jpeg(&default8, &dir.join("default.jpg"))?;
    let conv = LabConv::new();
    let fitted_fit = resize_area(&o.fitted_show, pf.fit_img.w, pf.fit_img.h);
    let v3 = objective == Objective::V3;
    let ref_stats = picture_stats_scene(
        &conv,
        &reference.fit,
        if v3 { reference.scene.as_ref() } else { None },
    );
    let default_fit = resize_area(&pf.default_show, pf.fit_img.w, pf.fit_img.h);
    let def_stats = picture_stats_scene(
        &conv,
        &default_fit,
        if v3 { pf.scene_fit.as_ref() } else { None },
    );
    let fit_stats = picture_stats_scene(
        &conv,
        &fitted_fit,
        if v3 { pf.scene_fit.as_ref() } else { None },
    );
    let mut reg_zero = [0.0; N_PARAMS];
    let before_full = terms(objective, &def_stats, &ref_stats, &def_stats, &reg_zero);
    reg_zero = o.x_scaled.map(|v| v / strength.max(1e-9));
    let after_full = terms(objective, &fit_stats, &ref_stats, &def_stats, &reg_zero);
    let mut v3_diag_before = Vec::new();
    let mut v3_diag_after = Vec::new();
    if v3 {
        let mut scratch = before_full;
        terms_v3(
            &def_stats,
            &ref_stats,
            &def_stats,
            Some(&mut v3_diag_before),
            &mut scratch,
        );
        let mut scratch = after_full;
        terms_v3(
            &fit_stats,
            &ref_stats,
            &def_stats,
            Some(&mut v3_diag_after),
            &mut scratch,
        );
    }
    let skin_sources = json!({
        "photo": pf.scene_fit.as_ref().map(|s| s.skin_source),
        "reference": reference.scene.as_ref().map(|s| s.skin_source),
    });
    let report = json!({
        "photo": photo_name,
        "v3_diagnostics_before": v3_diag_before,
        "v3_diagnostics_after": v3_diag_after,
        "skin_source": skin_sources,
        "round2_fit_seconds": round2.map(|r| r.fit.seconds),
        "reference": reference.stem,
        "strength": strength,
        "hue_step": with_hue,
        "objective": format!("{objective:?}"),
        "reference_bars_stripped_top_bottom_left_right": reference.bars,
        "fitted_values": real_json(&o.real, &o.fit.x, strength),
        "objective_on_fit_sample": { "before": o.fit.before.to_json(), "after": o.fit.after.to_json() },
        "objective_on_full_fit_pictures_512px": if objective.net() { Value::Null } else { json!({ "default": before_full.to_json(), "fitted_at_strength": after_full.to_json() }) },
        "v5": o.fit.extra,
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
        "settings_json_note": if objective.net() { "settings.json holds the global grade as real corrector settings (saturation and the four curves). The hue curves and the local grades are POC-only CPU steps with no settings field, so `ym render` with it gives the picture without them." } else if with_hue { "settings.json holds the part that is real corrector settings; the hue curves have no settings field yet, so `ym render` with it gives the picture without the hue step." } else { "no hue step: `ym render --settings settings.json` should reproduce the fitted picture (up to output sharpening)." },
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

/// Colours of the label overlay, by element id; skin is drawn white.
const LABEL_PALETTE: [[u8; 3]; N_ELEM] = [
    [50, 50, 50],
    [90, 150, 255],
    [60, 200, 80],
    [0, 200, 220],
    [200, 150, 80],
    [210, 90, 90],
    [200, 80, 200],
    [255, 220, 0],
    [255, 120, 0],
    [150, 150, 150],
    [255, 255, 255],
];

/// A picture with its labels mixed in half and half (skin white), at the picture's own size.
fn label_overlay(img: &Rgb8, scene: &Scene, sw: usize, sh: usize) -> Rgb8 {
    let mut data = img.data.clone();
    for y in 0..img.h {
        for x in 0..img.w {
            let (lx, ly) = ((x * sw / img.w).min(sw - 1), (y * sh / img.h).min(sh - 1));
            let i = ly * sw + lx;
            let k = if scene.skin[i] {
                ELEM_SKIN
            } else {
                usize::from(scene.elem[i]).min(ELEM_SKIN)
            };
            let d = 3 * (y * img.w + x);
            for c in 0..3 {
                data[d + c] = ((u16::from(data[d + c]) + u16::from(LABEL_PALETTE[k][c])) / 2) as u8;
            }
        }
    }
    Rgb8 {
        w: img.w,
        h: img.h,
        data,
    }
}

/// The sheet of the scene-aware rounds: Reference, Yellowmoon default, then one panel per round in `rounds` (label and
/// picture), and under them a strip with the labels the model recognised in the reference and in the photo, the text panel
/// of the last round's decisions, and the palette.
fn write_sheet_v3(
    path: &Path,
    reference: &Reference,
    default8: &Rgb8,
    rounds: &[(&str, &Rgb8)],
    pf: &PhotoFit<'_>,
    photo_aspect: f64,
    lines: Vec<String>,
) -> R<()> {
    const GAP: usize = 8;
    const BAND: usize = 30;
    const STRIP_H: usize = 150;
    let n = 2 + rounds.len();
    let panel_h = (PANEL_W_PX as f64 / photo_aspect).round() as usize;
    let (w, h) = (
        n * PANEL_W_PX + (n + 1) * GAP,
        panel_h + BAND + STRIP_H + 3 * GAP,
    );
    let mut c = Canvas::new(w, h, 18);
    let mut all: Vec<(&str, &Rgb8)> = vec![
        ("Reference", &reference.show8),
        ("Yellowmoon default", default8),
    ];
    all.extend(rounds.iter().cloned());
    for (i, (label, img)) in all.iter().enumerate() {
        let x = GAP + i * (PANEL_W_PX + GAP);
        c.text(label, x, GAP + 4, 3, 235);
        c.put_fit(img, x, GAP + BAND, PANEL_W_PX, panel_h);
    }
    let y = GAP + BAND + panel_h + GAP;
    if let Some(sc) = &reference.scene {
        let small = {
            let (tw, th) = fit_size(reference.fit.w, reference.fit.h, 320);
            linear_to_rgb8(&resize_area(&reference.fit, tw, th))
        };
        let ov = label_overlay(&small, sc, reference.fit.w, reference.fit.h);
        c.put_fit(&ov, GAP, y, PANEL_W_PX, STRIP_H);
    }
    if let Some(sc) = &pf.scene_fit {
        let small = {
            let (tw, th) = fit_size(pf.fit_img.w, pf.fit_img.h, 320);
            linear_to_rgb8(&resize_area(&pf.default_show, tw, th))
        };
        let ov = label_overlay(&small, sc, pf.fit_img.w, pf.fit_img.h);
        c.put_fit(&ov, GAP + PANEL_W_PX + GAP, y, PANEL_W_PX, STRIP_H);
    }
    // The text panel (the network's decisions and the local grades) under the first round, the palette under the last one
    // (one swatch per element in label order: sky, vegetation, water, ground, architecture, interior, person, other thing,
    // other stuff, skin).
    let x0 = GAP + 2 * (PANEL_W_PX + GAP);
    for (i, l) in lines.iter().take(8).enumerate() {
        c.text(l, x0 + 8, y + 4 + i * 17, 2, 225);
    }
    let x1 = GAP + (n - 1) * (PANEL_W_PX + GAP);
    for (k, id) in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter().enumerate() {
        for yy in 0..30 {
            for xx in 0..30 {
                let d = 3 * ((y + STRIP_H - 40 + yy) * c.w + x1 + 20 + k * 44 + xx);
                c.data[d..d + 3].copy_from_slice(&LABEL_PALETTE[*id]);
            }
        }
    }
    c.save_jpeg(path)
}

fn write_grid(
    path: &Path,
    pf: &PhotoFit<'_>,
    refs: &[(&Reference, &PairOutcome)],
    round2: &[Option<PairOutcome>],
) -> R<()> {
    const GAP: usize = 6;
    const LABEL: usize = 18;
    let cell_h = (GRID_CELL_W_PX as f64 * pf.default_show.h as f64 / pf.default_show.w as f64)
        .round() as usize;
    let thumb_h = (GRID_CELL_W_PX * 9 / 16).max(1);
    let rows = if round2.iter().any(Option::is_some) {
        2
    } else {
        1
    };
    let col_h = LABEL + thumb_h + rows * (GAP + cell_h);
    let cols = refs.len() + 1;
    let mut c = Canvas::new(GAP + cols * (GRID_CELL_W_PX + GAP), col_h + 2 * GAP, 18);
    let row_y = |r: usize| GAP + LABEL + thumb_h + GAP + r * (cell_h + GAP);
    c.text("0", GAP, GAP + 2, 2, 235);
    let default8 = linear_to_rgb8(&pf.default_show);
    for r in 0..rows {
        c.put_fit(&default8, GAP, row_y(r), GRID_CELL_W_PX, cell_h);
    }
    for (i, (r, o)) in refs.iter().enumerate() {
        let x = GAP + (i + 1) * (GRID_CELL_W_PX + GAP);
        c.text(&format!("{}", i + 1), x, GAP + 2, 2, 235);
        c.put_fit(&r.show8, x, GAP + LABEL, GRID_CELL_W_PX, thumb_h);
        // With round 2 given: its fit in the first row (Round 2), the main fit in the second (Round 3).
        if let Some(Some(o2)) = round2.get(i) {
            c.put_fit(
                &linear_to_rgb8(&o2.fitted_show),
                x,
                row_y(0),
                GRID_CELL_W_PX,
                cell_h,
            );
        }
        c.put_fit(
            &linear_to_rgb8(&o.fitted_show),
            x,
            row_y(rows - 1),
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
    objective: Objective,
    keep_key: bool,
) -> R<Value> {
    let conv = LabConv::new();
    let mut rng = Rng(SEED ^ 0xE1);
    let n_px = pf.fit_img.w * pf.fit_img.h;
    let mut pooled: Vec<f64> = Vec::new();
    let mut pooled_default: Vec<f64> = Vec::new();
    let mut trial_json = Vec::new();
    for t in 0..trials {
        let x_true = random_grade(&mut rng, moderate, keep_key, &[pf])?;
        let truth_real = to_real(&x_true, 1.0);
        let truth_cand = Candidate::new(&truth_real, true)?;
        let truth = render_image(&pf.fit_img, &truth_cand, &pf.output);
        let t0 = Instant::now();
        let (fit_res, _) = run_fit(pf, &truth, 1.0, true, objective, pf.scene_fit.as_ref())?;
        let fitted = render_fitted(pf, &fit_res.x, 1.0, true, false, objective.net())?;
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
            / N_GLOBAL as f64)
            .sqrt();
        let rms_true = (x_true.iter().map(|a| a * a).sum::<f64>() / N_GLOBAL as f64).sqrt();
        trial_json.push(json!({
            "trial": t, "fitted_vs_truth_de2000": { "median": s.median, "p95": s.p95, "mean": s.mean },
            "ungraded_vs_truth_de2000": { "median": sd.median, "p95": sd.p95 },
            "median_L": { "ungraded": picture_stats(&conv, &ungraded).median, "truth": picture_stats(&conv, &truth).median,
                "fitted": picture_stats(&conv, &fitted).median },
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
        "objective": format!("{objective:?}"),
        "grades_keep_the_photos_key": keep_key,
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

/// Median L* of a photo graded with `x` (at the fit picture's size).
fn graded_median(pf: &PhotoFit<'_>, x: &[f64; N_PARAMS]) -> R<f64> {
    let cand = Candidate::new(&to_real(x, 1.0), true)?;
    Ok(picture_stats(
        &LabConv::new(),
        &render_image(&pf.fit_img, &cand, &pf.output),
    )
    .median)
}

/// Draws a random grade of the family: each normalised parameter uniform in `+-amplitude`. With `keep_key` the draw is
/// repeated until the grade moves the median L* of every photo in `photos` by no more than the anchor's dead band
/// ([`V2_ANCHOR_DEAD_BAND_L`]): a grade that keeps the photo's key, which is what V2 can be asked to recover.
fn random_grade(
    rng: &mut Rng,
    amplitude: f64,
    keep_key: bool,
    photos: &[&PhotoFit<'_>],
) -> R<[f64; N_PARAMS]> {
    for _ in 0..400 {
        let mut x = [0.0; N_PARAMS];
        for v in &mut x[..N_GLOBAL] {
            *v = amplitude * (2.0 * rng.unit() - 1.0);
        }
        if !keep_key {
            return Ok(x);
        }
        let mut ok = true;
        for pf in photos {
            let base = graded_median(pf, &[0.0; N_PARAMS])?;
            ok &= (graded_median(pf, &x)? - base).abs() <= V2_ANCHOR_DEAD_BAND_L;
        }
        if ok {
            return Ok(x);
        }
    }
    Err("no key-preserving grade found in 400 draws".into())
}

/// CIEDE2000 between two pictures of the same size, per pixel.
fn delta_e_pixels(conv: &LabConv, a: &Img, b: &Img) -> Vec<f64> {
    (0..a.w * a.h)
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
}

/// E2, the cross-content check: the reference is photo A graded with G, the truth is photo B graded with the same G;
/// B is fitted to the reference and compared with the truth. Over every ordered pair of different photos and
/// `grades` seeded random grades G of the family.
fn e2(
    pfs: &[PhotoFit<'_>],
    names: &[String],
    grades: usize,
    amplitude: f64,
    objs: &[Objective],
    keep_key: bool,
) -> R<Value> {
    let conv = LabConv::new();
    let mut rng = Rng(SEED ^ 0xE2);
    // One set of grades for all photos: with `keep_key` each keeps every photo's key.
    let all: Vec<&PhotoFit<'_>> = pfs.iter().collect();
    let mut gs: Vec<[f64; N_PARAMS]> = Vec::new();
    for _ in 0..grades {
        gs.push(random_grade(&mut rng, amplitude, keep_key, &all)?);
    }
    let mut results = serde_json::Map::new();
    let mut summary = serde_json::Map::new();
    for &obj in objs {
        let mut pooled: Vec<f64> = Vec::new();
        let mut pooled_default: Vec<f64> = Vec::new();
        let mut fit_medians: Vec<f64> = Vec::new();
        let mut fit_secs: Vec<f64> = Vec::new();
        let mut fits = Vec::new();
        for a in 0..pfs.len() {
            for b in 0..pfs.len() {
                if a == b {
                    continue;
                }
                for (g, x) in gs.iter().enumerate() {
                    let cand = Candidate::new(&to_real(x, 1.0), true)?;
                    let reference = render_image(&pfs[a].fit_img, &cand, &pfs[a].output);
                    let truth = render_image(&pfs[b].fit_img, &cand, &pfs[b].output);
                    let (fr, _) = run_fit(
                        &pfs[b],
                        &reference,
                        1.0,
                        true,
                        obj,
                        pfs[a].scene_fit.as_ref(),
                    )?;
                    let fitted = render_fitted(&pfs[b], &fr.x, 1.0, true, false, obj.net())?;
                    let ungraded = render_image(
                        &pfs[b].fit_img,
                        &Candidate::new(&to_real(&[0.0; N_PARAMS], 1.0), false)?,
                        &pfs[b].output,
                    );
                    let d = delta_e_pixels(&conv, &fitted, &truth);
                    let dd = delta_e_pixels(&conv, &ungraded, &truth);
                    let s = Summary::of(&d).ok_or("empty")?;
                    pooled.extend(d.iter().step_by(4));
                    pooled_default.extend(dd.iter().step_by(4));
                    fit_medians.push(s.median);
                    fit_secs.push(fr.seconds);
                    fits.push(
                        json!({ "reference_photo": names[a], "target_photo": names[b], "grade": g,
                        "median": s.median, "p95": s.p95, "fit_seconds": fr.seconds, "x": fr.x.to_vec(), "x_true": x.to_vec(), "terms_after": fr.after.to_json(), "terms_before": fr.before.to_json() }),
                    );
                }
            }
        }
        let sp = Summary::of(&pooled).ok_or("empty")?;
        let sd = Summary::of(&pooled_default).ok_or("empty")?;
        fit_medians.sort_by(f64::total_cmp);
        let sum = json!({
            "fits": fits.len(),
            "fitted_vs_truth_de2000_pooled": { "median": sp.median, "p95": sp.p95, "mean": sp.mean },
            "ungraded_vs_truth_de2000_pooled": { "median": sd.median, "p95": sd.p95 },
            "median_of_fit_medians": fit_medians[fit_medians.len() / 2],
            "mean_fit_seconds": fit_secs.iter().sum::<f64>() / fit_secs.len() as f64,
        });
        summary.insert(format!("{obj:?}"), sum);
        results.insert(format!("{obj:?}"), Value::Array(fits));
    }
    Ok(
        json!({ "grades": grades, "grades_keep_every_photos_key": keep_key, "amplitude_fraction_of_bound": amplitude, "photos": names,
        "summary": summary, "fits": results }),
    )
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
    objective: Option<Objective>,
    e2: bool,
    keep_key: bool,
    labels: Option<PathBuf>,
    network: Option<PathBuf>,
    network5: Option<PathBuf>,
    network6: Option<PathBuf>,
    network7: Option<PathBuf>,
    network8: Option<PathBuf>,
    network9: Option<PathBuf>,
    network10: Option<PathBuf>,
    network11: Option<PathBuf>,
    network_check: bool,
    grades: usize,
    /// The evaluation harness (`eval.rs`): the split file, the set, the previous round for the grids, `--compare a b`.
    eval: bool,
    split: Option<PathBuf>,
    set: Option<String>,
    prev: Option<Objective>,
    compare: Option<(PathBuf, PathBuf)>,
    shard: Option<(usize, usize)>,
    no_grids: bool,
    merge: Vec<PathBuf>,
    /// Round-12 principles switched back to v11 (`--ablate p1,p3`), and the photos to keep (`--only a,b`, by file stem).
    ablate: Ablate,
    only: Vec<String>,
}

fn parse_objective(name: &str) -> R<Objective> {
    Ok(match name {
        "v1" => Objective::V1,
        "v2" => Objective::V2,
        "v3" => Objective::V3,
        "v5" => Objective::V5,
        "v6" => Objective::V6,
        "v7" => Objective::V7,
        "v8" => Objective::V8,
        "v9" => Objective::V9,
        "v10" => Objective::V10,
        "v11" => Objective::V11,
        "v12" => Objective::V12,
        "v13" => Objective::V13,
        other => return Err(format!("--objective takes v1 to v13, not {other}").into()),
    })
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        strength: 1.0,
        trials: 10,
        amplitude: 0.5,
        grades: 5,
        ..Args::default()
    };
    let mut ablate_given = false;
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
            "--e2" => a.e2 = true,
            "--labels" => a.labels = Some(PathBuf::from(val()?)),
            "--network" => a.network = Some(PathBuf::from(val()?)),
            "--network-check" => a.network_check = true,
            "--network5" => a.network5 = Some(PathBuf::from(val()?)),
            "--network6" => a.network6 = Some(PathBuf::from(val()?)),
            "--network7" => a.network7 = Some(PathBuf::from(val()?)),
            "--network8" => a.network8 = Some(PathBuf::from(val()?)),
            "--network9" => a.network9 = Some(PathBuf::from(val()?)),
            "--network10" => a.network10 = Some(PathBuf::from(val()?)),
            "--network11" => a.network11 = Some(PathBuf::from(val()?)),
            "--keep-key" => a.keep_key = true,
            "--grades" => a.grades = val()?.parse()?,
            "--objective" => a.objective = Some(parse_objective(&val()?)?),
            "--prev" => a.prev = Some(parse_objective(&val()?)?),
            "--eval" => a.eval = true,
            "--no-grids" => a.no_grids = true,
            "--shard" => {
                let v = val()?;
                let (i, n) = v.split_once('/').ok_or("--shard takes i/n")?;
                let (i, n): (usize, usize) = (i.parse()?, n.parse()?);
                if n == 0 || i >= n {
                    return Err("--shard i/n needs i < n".into());
                }
                a.shard = Some((i, n));
            }
            "--ablate" => {
                if ablate_given {
                    return Err("--ablate may be given once".into());
                }
                ablate_given = true;
                a.ablate = Ablate::parse(&val()?)?;
            }
            "--only" => a.only.extend(
                val()?
                    .split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty()),
            ),
            "--merge" => a.merge.push(PathBuf::from(val()?)),
            "--split" => a.split = Some(PathBuf::from(val()?)),
            "--set" => a.set = Some(val()?),
            "--compare" => {
                let (x, y) = (val()?, val()?);
                a.compare = Some((PathBuf::from(x), PathBuf::from(y)));
            }
            "--selftest" => a.selftest = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if !a.ablate.is_none() && !matches!(a.objective, Some(Objective::V12 | Objective::V13)) {
        return Err("--ablate needs --objective v12 or v13".into());
    }
    if a.out.as_os_str().is_empty()
        && let Some((_, b)) = &a.compare
    {
        a.out = b.clone();
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
    let network = match &args.network {
        Some(path) => {
            let net = parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            fs::write(args.out.join("expanded-network.txt"), net.dump())?;
            eprintln!(
                "network: {} neurons, expanded to expanded-network.txt",
                net.neurons.len()
            );
            Some(net)
        }
        None => None,
    };
    let network5 = match &args.network5 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let network6 = match &args.network6 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let network7 = match &args.network7 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let network8 = match &args.network8 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let network9 = match &args.network9 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let network10 = match &args.network10 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let network11 = match &args.network11 {
        Some(path) => Some(
            parse_network(&fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    if args.network_check {
        let net = network
            .as_ref()
            .ok_or("--network-check needs --network <file.ywn>")?;
        let ok = network_check(net)?;
        return if ok {
            Ok(())
        } else {
            Err("network check failed".into())
        };
    }
    if !args.merge.is_empty() {
        let name = args
            .ablate
            .label(&format!("{:?}", args.objective.unwrap_or(Objective::V2)).to_lowercase());
        return eval::merge(&args.merge, &name, &args.out);
    }
    if let Some((a, b)) = &args.compare {
        return eval::compare(a, b, &args.out);
    }
    let photos = expand(&args.photos, &["arw"]);
    if photos.is_empty() && args.split.is_none() {
        return Err("no photos".into());
    }
    // V3's inputs: the label folder, and the skin network when its weights are in the data folder.
    let seg = ym_core::data_dir::find_data_dir().ok().and_then(|d| {
        ym_ml::mediapipe::selfie::SelfieNet::load(&d.join("models").join("mediapipe"))
            .ok()
            .map(ym_ml::mediapipe::selfie::SelfieSegmenter::new)
    });
    if args.labels.is_some() {
        eprintln!(
            "skin: {}",
            if seg.is_some() {
                "MediaPipe Selfie Multiclass"
            } else {
                "fallback rule (no network weights)"
            }
        );
    }
    let _ = CTX.set(Ctx {
        labels: args.labels.clone(),
        seg,
        network,
        network5,
        network6,
        network7,
        network8,
        network9,
        network10,
        network11,
        ablate: args.ablate,
    });
    let core = open_core(&args.out)?;
    let cache = args.out.join("cache");
    let objective = args.objective.unwrap_or(Objective::V2);
    if args.eval {
        let split = match &args.split {
            Some(p) => eval::read_split(p)?,
            None => eval::builtin_split(&photos),
        };
        return eval::run_eval(&eval::EvalOpts {
            split,
            set: args.set.clone().unwrap_or_else(|| "all".into()),
            objective,
            prev: args.prev,
            references: args.references.clone(),
            out: &args.out,
            core: &core,
            cache: &cache,
            strength: args.strength,
            with_hue: !args.no_hue,
            shard: args.shard,
            no_grids: args.no_grids,
            ablate: args.ablate,
            only: args.only.clone(),
        });
    }
    if args.e2 {
        let mut nodes = Vec::new();
        for p in &photos {
            nodes.push((
                p.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                load_photo(&core, p, &cache)?,
            ));
        }
        let mut pfs = Vec::new();
        for (_, n) in &nodes {
            pfs.push(prepare_photo(n)?);
        }
        let names: Vec<String> = nodes.iter().map(|(n, _)| n.clone()).collect();
        let objs: Vec<Objective> = args
            .objective
            .map_or(vec![Objective::V1, Objective::V2], |o| vec![o]);
        let r = e2(
            &pfs,
            &names,
            args.grades,
            args.amplitude,
            &objs,
            args.keep_key,
        )?;
        fs::write(args.out.join("e2.json"), serde_json::to_string_pretty(&r)?)?;
        println!("{}", serde_json::to_string_pretty(&r["summary"])?);
        return Ok(());
    }
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
            let r = selftest(
                &pf,
                &photo_dir,
                &name,
                args.trials,
                args.amplitude,
                objective,
                args.keep_key,
            )?;
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
        let mut outcomes2: Vec<Option<PairOutcome>> = Vec::new();
        let mut outcomes5: Vec<Option<PairOutcome>> = Vec::new();
        let mut outcomes6: Vec<Option<PairOutcome>> = Vec::new();
        let mut outcomes7: Vec<Option<PairOutcome>> = Vec::new();
        let mut outcomes8: Vec<Option<PairOutcome>> = Vec::new();
        let mut outcomes9: Vec<Option<PairOutcome>> = Vec::new();
        let mut outcomes10: Vec<Option<PairOutcome>> = Vec::new();
        let mut index: BTreeMap<String, Value> = BTreeMap::new();
        for r in &loaded {
            let o = run_pair(&pf, r, args.strength, !args.no_hue, objective)?;
            // Round 2 for comparison: the V2 fit of the same pair.
            // Round 5 next to Round 6: the V5 fit of the same pair (with `--network5`).
            let o5 =
                if matches!(objective, Objective::V6 | Objective::V7) && ctx().network5.is_some() {
                    Some(run_pair(
                        &pf,
                        r,
                        args.strength,
                        !args.no_hue,
                        Objective::V5,
                    )?)
                } else {
                    None
                };
            // Round 6 next to Round 7: the V6 fit of the same pair (with `--network6`).
            let o6 =
                if matches!(objective, Objective::V7 | Objective::V8) && ctx().network6.is_some() {
                    Some(run_pair(
                        &pf,
                        r,
                        args.strength,
                        !args.no_hue,
                        Objective::V6,
                    )?)
                } else {
                    None
                };
            // Round 7 next to Round 8: the V7 fit of the same pair (with `--network7`).
            // Round 8 next to Round 9: the V8 fit of the same pair (with `--network8`).
            let o8 = if matches!(objective, Objective::V9 | Objective::V10)
                && ctx().network8.is_some()
            {
                Some(run_pair(
                    &pf,
                    r,
                    args.strength,
                    !args.no_hue,
                    Objective::V8,
                )?)
            } else {
                None
            };
            // Round 9 next to Round 10: the V9 fit of the same pair (with `--network9`).
            // Round 10 next to Round 11: the V10 fit of the same pair (with `--network10`).
            let o10 = if objective == Objective::V11 && ctx().network10.is_some() {
                Some(run_pair(
                    &pf,
                    r,
                    args.strength,
                    !args.no_hue,
                    Objective::V10,
                )?)
            } else {
                None
            };
            let o9 = if matches!(objective, Objective::V10 | Objective::V11)
                && ctx().network9.is_some()
            {
                Some(run_pair(
                    &pf,
                    r,
                    args.strength,
                    !args.no_hue,
                    Objective::V9,
                )?)
            } else {
                None
            };
            let o7 =
                if matches!(objective, Objective::V8 | Objective::V9) && ctx().network7.is_some() {
                    Some(run_pair(
                        &pf,
                        r,
                        args.strength,
                        !args.no_hue,
                        Objective::V7,
                    )?)
                } else {
                    None
                };
            let o2 = if matches!(objective, Objective::V3 | Objective::V5 | Objective::V6) {
                Some(run_pair(
                    &pf,
                    r,
                    args.strength,
                    !args.no_hue,
                    Objective::V2,
                )?)
            } else {
                None
            };
            eprintln!(
                "   {}: fit {:.2} s, {} evals, {:.2} ms/eval, objective {:.3} -> {:.3}; newly purple {} -> {:.4}; skin ratio {} -> {:.3}",
                r.stem,
                o.fit.seconds,
                o.fit.evals,
                o.fit.per_eval_ms,
                o.fit.before.total(),
                o.fit.after.total(),
                o10.as_ref()
                    .or(o9.as_ref())
                    .map_or("-".to_string(), |q| format!(
                        "{:.4}",
                        q.fit.extra["newly_purple_share"].as_f64().unwrap_or(0.0)
                    )),
                o.fit.extra["newly_purple_share"].as_f64().unwrap_or(0.0),
                o10.as_ref()
                    .or(o9.as_ref())
                    .map_or("-".to_string(), |q| format!(
                        "{:.3}",
                        q.fit.extra["skin_chroma_ratio"].as_f64().unwrap_or(1.0)
                    )),
                o.fit.extra["skin_chroma_ratio"].as_f64().unwrap_or(1.0)
            );
            write_pair(
                &photo_dir.join(&r.stem),
                &name,
                &pf,
                r,
                &o,
                args.strength,
                !args.no_hue,
                objective,
                o2.as_ref(),
                o5.as_ref(),
                o6.as_ref(),
                o7.as_ref(),
                o8.as_ref(),
                o9.as_ref(),
                o10.as_ref(),
            )?;
            outcomes2.push(o2);
            outcomes5.push(o5);
            outcomes6.push(o6);
            outcomes7.push(o7);
            outcomes8.push(o8);
            outcomes9.push(o9);
            outcomes10.push(o10);
            index.insert(
                r.stem.clone(),
                json!({ "fit_seconds": o.fit.seconds, "per_eval_ms": o.fit.per_eval_ms, "before": o.fit.before.total(), "after": o.fit.after.total() }),
            );
            outcomes.push(o);
        }
        let pairs: Vec<(&Reference, &PairOutcome)> = loaded.iter().zip(&outcomes).collect();
        let grid_rows = if outcomes10.iter().any(Option::is_some) {
            &outcomes10
        } else if outcomes9.iter().any(Option::is_some) {
            &outcomes9
        } else if outcomes8.iter().any(Option::is_some) {
            &outcomes8
        } else if outcomes7.iter().any(Option::is_some) {
            &outcomes7
        } else if outcomes6.iter().any(Option::is_some) {
            &outcomes6
        } else if outcomes5.iter().any(Option::is_some) {
            &outcomes5
        } else {
            &outcomes2
        };
        write_grid(&photo_dir.join("grid.jpg"), &pf, &pairs, grid_rows)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Oklab hue of an sRGB colour (display-linear input from encoded 0..1 values given as linear here).
    fn hue_of(lin: [f64; 3]) -> f64 {
        oklab_hue(oklab_ab(lin))
    }

    #[test]
    fn v12_magenta_turned_out_of_the_band_counts() {
        // A magenta (C* 30, Oklab 328) that turns blue (C* 30, 264): 64 degrees, out of the band. v11 did not count it.
        assert!(counts_as_purple_turn((30.0, 328.0), (30.0, 264.0), 10.0));
        // The reverse (a blue turned into the band) counts, and so does a grey that becomes purple.
        assert!(counts_as_purple_turn((30.0, 264.0), (30.0, 328.0), 10.0));
        assert!(counts_as_purple_turn((1.0, 100.0), (6.0, 330.0), 45.0));
        // A magenta that stays (turn under the limit), and a turn between two colours outside the band, do not.
        assert!(!counts_as_purple_turn((30.0, 328.0), (30.0, 335.0), 10.0));
        assert!(!counts_as_purple_turn((30.0, 100.0), (30.0, 200.0), 10.0));
        // A low-chroma pixel in the band that moves is not a turn of a colour.
        assert!(!counts_as_purple_turn((2.0, 328.0), (2.0, 264.0), 10.0));
    }

    #[test]
    fn v12_opposite_tinted_whites_do_not_cancel() {
        let conv = LabConv::new();
        // A teal white and a pink white of equal strength: their mean vector is neutral.
        let teal = conv.px([0.80, 0.95, 0.95]);
        let pink = conv.px([0.95, 0.80, 0.85]);
        let px = [teal, pink];
        let scope = [true, true];
        let (mean_a, mean_b) = (
            (px[0].lab[1] + px[1].lab[1]) / 2.0,
            (px[0].lab[2] + px[1].lab[2]) / 2.0,
        );
        let mean_vector_c = mean_a.hypot(mean_b);
        let (c, d) = hl4_stats(&px, &scope, 0.0, 0.0);
        assert!(
            mean_vector_c < 0.5 * c,
            "the mean vector cancels: {mean_vector_c} against {c}"
        );
        assert!(c > 5.0, "the per-pixel chroma sees both tints: {c}");
        assert!(
            (c - d).abs() < 1e-9,
            "the distance to a neutral target is the chroma"
        );
    }

    #[test]
    fn v12_clip_allowance() {
        // The photo holds 10 %, the reference 40 %: the grade may reach 12 %, not 40 %.
        assert!((clip_allowance(0.10, 0.40, 0.02) - 0.12).abs() < 1e-12);
        // A reference with less than the photo does not lower the photo's own share.
        assert!((clip_allowance(0.10, 0.01, 0.02) - 0.10).abs() < 1e-12);
        // A reference slightly above the photo licenses what it holds.
        assert!((clip_allowance(0.10, 0.11, 0.02) - 0.11).abs() < 1e-12);
    }

    #[test]
    fn ablate_parses_and_labels() {
        assert_eq!(Ablate::default().label("v12"), "v12");
        assert_eq!(Ablate::parse("p2").unwrap().label("v12"), "v12-no-p2");
        // The order given does not matter, and spaces and capitals are forgiven.
        assert_eq!(
            Ablate::parse("P5, p1").unwrap().label("v12"),
            "v12-no-p1-p5"
        );
        assert_eq!(
            Ablate::parse("p1,p2,p3,p4,p5").unwrap().label("v12"),
            "v12-no-p1-p2-p3-p4-p5"
        );
        let a = Ablate::parse("p3").unwrap();
        assert!(a.on(1) && a.on(2) && !a.on(3) && a.on(4) && a.on(5));
        assert!(Ablate::parse("p6").is_err());
        assert!(Ablate::parse("").is_err());
        assert!(Ablate::parse("p1,p1").is_err());
    }

    /// A context for the guard tests: v12 on with the given principles, everything else empty.
    fn ablation_ctx(ablate: Ablate, vals: &[(&str, f64)]) -> V5Ctx {
        let mut ctx = V5Ctx {
            p1: ablate.on(1),
            p2: ablate.on(2),
            p3: ablate.on(3),
            p4: ablate.on(4),
            p5: ablate.on(5),
            ..V5Ctx::default()
        };
        for (k, v) in vals {
            ctx.out.map.insert((*k).to_string(), *v);
        }
        ctx
    }

    fn off(n: usize) -> Ablate {
        Ablate::parse(&format!("p{n}")).unwrap()
    }

    #[test]
    fn ablate_p1_turn_out_of_the_band_counts_only_with_p1() {
        let conv = LabConv::new();
        // A magenta pixel in photo0 (C* 30, Oklab hue 328) that the candidate turns to a saturated blue.
        let blue = conv.px([0.1, 0.2, 0.9]);
        assert!(blue.lab[1].hypot(blue.lab[2]) > V10_NP_C);
        let mk = |a: Ablate| {
            let mut ctx = ablation_ctx(a, &[]);
            ctx.not_skin = vec![true];
            ctx.p_cstar = vec![30.0];
            ctx.p_okh = vec![328.0];
            ctx
        };
        assert_eq!(
            turned_purple_share(&[blue], &mk(Ablate::default()), 10.0),
            1.0
        );
        assert_eq!(turned_purple_share(&[blue], &mk(off(1)), 10.0), 0.0);
    }

    #[test]
    fn ablate_p2_turn_budget_only_with_p2() {
        let vals = [("T.turn_share", 0.03)];
        let c = CandMeas {
            turn_share: 0.06,
            ..CandMeas::default()
        };
        let run = |a: Ablate| {
            let ctx = ablation_ctx(a, &vals);
            let mut t = V5T::default();
            guards_v9(&mut t, &c, &ctx);
            t.guard_turn
        };
        assert!((run(Ablate::default()) - 6.0 * 9.0).abs() < 1e-9);
        assert_eq!(run(off(2)), 0.0);
    }

    #[test]
    fn v13_allowed_turn_by_chroma() {
        let near = |a: f64, b: f64| assert!((a - b).abs() < 0.01, "{a} vs {b}");
        // Unify shut: T.turn_deg 20.
        near(allowed_turn_deg(8.0, 20.0, 25.0, 45.0), 45.0);
        near(allowed_turn_deg(15.0, 20.0, 25.0, 45.0), 34.75);
        near(allowed_turn_deg(25.0, 20.0, 25.0, 45.0), 20.0);
        near(allowed_turn_deg(60.0, 20.0, 25.0, 45.0), 20.0);
        // Unify open: T.turn_deg 30.
        near(allowed_turn_deg(15.0, 30.0, 25.0, 45.0), 45.0);
        near(allowed_turn_deg(20.0, 30.0, 25.0, 45.0), 38.68);
        near(allowed_turn_deg(25.0, 30.0, 25.0, 45.0), 30.0);
        // No chroma: the cap.
        near(allowed_turn_deg(0.0, 20.0, 25.0, 45.0), 45.0);
    }

    /// A one-pixel context whose photo0 colour has C* `c` and Oklab hue `hue`, not skin.
    fn turn_ctx(v13: bool, c: f64, hue: f64) -> V5Ctx {
        let mut ctx = ablation_ctx(Ablate::default(), &[]);
        ctx.v13 = v13;
        ctx.not_skin = vec![true];
        ctx.p_cstar = vec![c];
        ctx.p_okh = vec![hue];
        ctx
    }

    /// A pixel whose Oklab hue is `turn` degrees from `hue`.
    fn turned_px(conv: &LabConv, hue: f64, turn: f64) -> Px {
        let mut q = conv.px([0.4, 0.3, 0.2]);
        let c = q.ok[0].hypot(q.ok[1]).max(0.05);
        let h = (hue + turn).to_radians();
        q.ok = [c * h.cos(), c * h.sin()];
        assert!((wrap_deg(oklab_hue(q.ok) - hue).abs() - turn).abs() < 1e-6);
        q
    }

    #[test]
    fn v13_pale_pixel_turned_35_does_not_count() {
        let conv = LabConv::new();
        let q = turned_px(&conv, 60.0, 35.0);
        let (v12, v13) = (turn_ctx(false, 12.0, 60.0), turn_ctx(true, 12.0, 60.0));
        assert_eq!(content_turn_share(&[q], &v12, 20.0, None), 1.0);
        assert_eq!(
            content_turn_share(&[q], &v13, 20.0, Some((25.0, 45.0))),
            0.0
        );
    }

    #[test]
    fn v13_strong_pixel_turned_25_counts() {
        let conv = LabConv::new();
        let q = turned_px(&conv, 60.0, 25.0);
        let (v12, v13) = (turn_ctx(false, 40.0, 60.0), turn_ctx(true, 40.0, 60.0));
        assert_eq!(content_turn_share(&[q], &v12, 20.0, None), 1.0);
        assert_eq!(
            content_turn_share(&[q], &v13, 20.0, Some((25.0, 45.0))),
            1.0
        );
    }

    #[test]
    fn ablate_p3_oklab_hue_wall_only_with_p3() {
        // The whites' CIELAB mean vector does not move, but their Oklab mean turns by 90 degrees.
        let vals = [("T.wall_deg", 20.0), ("T.hl_cmax", 20.0)];
        let mut c = CandMeas {
            hl_a: 10.0,
            hl_b: 0.0,
            pg_ab: vec![[0.1, 0.0]; PG_N],
            ..CandMeas::default()
        };
        c.pg_ab[PG_HL] = [0.0, 0.1];
        let run = |a: Ablate| {
            let mut ctx = ablation_ctx(a, &vals);
            ctx.hl_scope = vec![true; 10];
            ctx.p_hl_a = 10.0;
            ctx.p_hl_b = 0.0;
            ctx.p_pg_ab = vec![[0.1, 0.0]; PG_N];
            let mut t = V5T::default();
            guards_v9(&mut t, &c, &ctx);
            t.guard_hl
        };
        assert!(run(Ablate::default()) > 0.0);
        assert_eq!(run(off(3)), 0.0);
    }

    #[test]
    fn ablate_p4_opposite_tinted_whites_cancel_again() {
        // The mean vector of the whites is neutral, the per-pixel chroma and distance are 12.
        let vals = [("T.hl_cmax", 4.0), ("T.white_tol", 2.0)];
        let c = CandMeas {
            hl4_c: 12.0,
            hl4_d: 12.0,
            ..CandMeas::default()
        };
        let run = |a: Ablate| {
            let mut ctx = ablation_ctx(a, &vals);
            ctx.hl_scope = vec![true; 10];
            let mut t = V5T::default();
            guards_v9(&mut t, &c, &ctx);
            (t.guard_hl, t.guard_white_dir)
        };
        let (cap, dir) = run(Ablate::default());
        assert!(cap > 0.0 && dir > 0.0);
        assert_eq!(run(off(4)), (0.0, 0.0));
    }

    #[test]
    fn ablate_p5_clip_allowance_only_with_p5() {
        // The photo holds 10 % clipped white, the reference 40 %; the candidate clips 20 %.
        let mk = |a: Ablate| {
            let mut ctx = ablation_ctx(a, &[("T.clip_extra", 0.02)]);
            ctx.v8 = true;
            ctx.p.hi = 0.10;
            ctx.r.hi = 0.40;
            ctx
        };
        let m = Meas {
            hi: 0.20,
            ..Meas::default()
        };
        assert!(clip_term(&mk(Ablate::default()), &m) > 0.0);
        // v11: the larger of the two shares licenses it.
        assert_eq!(clip_term(&mk(off(5)), &m), 0.0);
        // Crushed black counts only with P5 (L* < 2.5), not under v11's test.
        let crushed = Meas {
            lo25: 0.20,
            ..Meas::default()
        };
        assert!(clip_term(&mk(Ablate::default()), &crushed) > 0.0);
        assert_eq!(clip_term(&mk(off(5)), &crushed), 0.0);
    }

    #[test]
    fn oklab_hue_of_primaries() {
        assert!((hue_of([0.0, 0.0, 1.0]) - 264.05).abs() < 0.5, "blue");
        assert!((hue_of([1.0, 0.0, 1.0]) - 328.36).abs() < 0.5, "magenta");
        assert!((hue_of([1.0, 0.0, 0.0]) - 29.23).abs() < 0.5, "red");
        assert!((hue_of([1.0, 1.0, 0.0]) - 109.77).abs() < 0.5, "yellow");
    }
}
