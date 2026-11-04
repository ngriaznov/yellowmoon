//! The spatial passes of the Film node on the CPU: scatter and halation (P2), the diffusion of the coupler
//! inhibitor (P3) and grain (P4), around F-04's baked colour passes (P1, P5). `docs/24-film.md` section 3.9.
//!
//! Written from: James, *The Theory of the Photographic Process*, 4th ed. (chapter 20 on the spread of light in
//! the emulsion and the modulation transfer function, chapter 21 on halation); Hunt, *The Reproduction of
//! Colour*, 6th ed., chapter 15 (inhibitor diffusion and adjacency effects); the Kodak and Fujifilm data sheets'
//! MTF graphs; and for the blur the sampling of a Gaussian and the bilinear resampling of a coarse grid, both
//! textbook. Grain is in [`crate::grain`]. The numbers are in `data/film/params/` (`halation.json`, `grain.json`,
//! `couplers.json`), each with its source or "chosen".
//!
//! **Sizes** are micrometres on the film. The long edge of the photo is 36 mm of film, so a size becomes
//! `size / 36,000` of the long edge and then pixels at the render's scale ([`Frame::px_per_um`]).
//!
//! **Tiles equal the whole frame, bit for bit** (`docs/17` D4 and D5). A pixel of a pass depends only on its
//! receptive field, never on the tile's origin or size:
//! * every pass reads its input at coordinates clamped into the photo frame, so the frame's edge is the same
//!   for every tile;
//! * a Gaussian of at most [`DIRECT_MAX_SIGMA_PX`] is a direct separable sum in a fixed tap order; a larger one
//!   is done on a coarse grid whose cells sit at absolute multiples of the factor `f` (a box mean of `f x f`
//!   pixels, a direct blur of the cells, a bilinear read back), so the cells never depend on the tile;
//! * the passes compute their output on the tile plus the halos of the passes after them, cut to the frame.
//!
//! **Halo.** A Gaussian of `sigma` pixels needs `ceil(6 sigma)` pixels, and three cells more when it is done on
//! the coarse grid ([`blur_halo`]). A pass's halo is the largest of its terms.
//!
//! Buffers between passes are f32; the arithmetic inside is f64, summed in a fixed order, no `fma`.

use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

use crate::Vec3;
use crate::bake::BakedFilm;
use crate::error::FilmError;
use crate::grain::{self, GrainModel, GrainParams, Layers, N_SUBLAYERS};
use crate::math::pow10;
use crate::profile::Names;

/// The long edge of the photo frame, in micrometres of film (full-frame 35 mm).
pub const FILM_LONG_EDGE_UM: f64 = 36_000.0;
/// A Gaussian narrower than this (in pixels) is a delta: its weight stays at the pixel.
pub const MIN_SIGMA_PX: f64 = 0.2;
/// A Gaussian up to this wide (in pixels) is a direct sum; a wider one is done on a coarse grid.
pub const DIRECT_MAX_SIGMA_PX: f64 = 4.0;

// ---------------------------------------------------------------------------------------------------------
// Geometry and buffers
// ---------------------------------------------------------------------------------------------------------

/// The photo frame at the render's scale, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl Frame {
    /// Pixels per micrometre of film: the long edge is 36,000 um.
    #[must_use]
    pub fn px_per_um(&self) -> f64 {
        self.width.max(self.height) as f64 / FILM_LONG_EDGE_UM
    }

    /// The whole frame as a rectangle.
    #[must_use]
    pub fn rect(&self) -> Rect {
        Rect {
            x0: 0,
            y0: 0,
            x1: self.width,
            y1: self.height,
        }
    }
}

/// A rectangle of pixels, `x1` and `y1` excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left.
    pub x0: usize,
    /// Top.
    pub y0: usize,
    /// Right, excluded.
    pub x1: usize,
    /// Bottom, excluded.
    pub y1: usize,
}

impl Rect {
    /// Width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.x1 - self.x0
    }

    /// Height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.y1 - self.y0
    }

    /// `h` pixels more on every side, cut to the frame.
    #[must_use]
    pub fn expand(&self, h: usize, frame: Frame) -> Self {
        Self {
            x0: self.x0.saturating_sub(h),
            y0: self.y0.saturating_sub(h),
            x1: (self.x1 + h).min(frame.width),
            y1: (self.y1 + h).min(frame.height),
        }
    }

    /// Whether `other` lies inside this rectangle.
    #[must_use]
    pub fn contains(&self, other: &Rect) -> bool {
        self.x0 <= other.x0 && self.y0 <= other.y0 && self.x1 >= other.x1 && self.y1 >= other.y1
    }
}

/// Three f32 channels per pixel over a rectangle of the frame.
#[derive(Debug, Clone)]
pub struct Plane {
    /// The pixels this plane holds.
    pub rect: Rect,
    /// Interleaved RGB, row by row.
    pub data: Vec<f32>,
}

impl Plane {
    /// A plane of zeros, or of NaN when `poison` is set (the poison test of `docs/17` D7).
    #[must_use]
    pub fn new(rect: Rect, poison: bool) -> Self {
        let fill = if poison { f32::NAN } else { 0.0 };
        Self {
            rect,
            data: vec![fill; rect.width() * rect.height() * 3],
        }
    }

    /// The pixel at frame coordinates `(x, y)`, which must lie in the plane.
    #[inline]
    #[must_use]
    pub fn px(&self, x: usize, y: usize) -> [f32; 3] {
        let i = ((y - self.rect.y0) * self.rect.width() + (x - self.rect.x0)) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

/// How a run executes. Neither field changes a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOptions {
    /// Threads; rows are split between them and the bytes are the same for any count.
    pub threads: usize,
    /// Every buffer starts as NaN, so a pass that reads something it did not write shows up.
    pub poison: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            poison: false,
        }
    }
}

/// Runs `f(row, row_slice)` for every row of `out`, the rows split between `threads`.
fn par_rows<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    out: &mut [T],
    row_len: usize,
    threads: usize,
    f: F,
) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let t = threads.max(1).min(rows);
    if t <= 1 {
        for (r, row) in out.chunks_mut(row_len).enumerate() {
            f(r, row);
        }
        return;
    }
    let per = rows.div_ceil(t);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * per + k, row);
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------------------------------------

/// The coarse-grid factor of a Gaussian of `sigma` pixels: 1 for a direct sum, else a power of two that leaves
/// the coarse sigma between 2.5 and 5 cells (at least 2).
fn decim_factor(sigma: f64) -> usize {
    if sigma <= DIRECT_MAX_SIGMA_PX {
        1
    } else {
        1 << ((sigma / 2.5).log2().floor().max(1.0) as u32)
    }
}

/// The pixels a Gaussian of `sigma` pixels reads beyond its output: `ceil(6 sigma)`, plus three cells on the
/// coarse grid. Zero for a delta.
#[must_use]
pub fn blur_halo(sigma_px: f64) -> usize {
    if sigma_px < MIN_SIGMA_PX {
        return 0;
    }
    let f = decim_factor(sigma_px);
    let cells = if f > 1 { 3 * f } else { 0 };
    (6.0 * sigma_px).ceil() as usize + cells
}

/// The one-sided weights `w[0..=r]` of a sampled Gaussian, normalised so that `w0 + 2 sum w = 1`.
fn kernel(sigma: f64) -> Vec<f64> {
    let r = (6.0 * sigma).ceil() as usize;
    let mut w: Vec<f64> = (0..=r)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let total: f64 = w[0] + 2.0 * w[1..].iter().sum::<f64>();
    for v in &mut w {
        *v /= total;
    }
    w
}

/// The sum of the squared weights of the whole (one-dimensional) kernel: the factor by which a blur lowers the
/// variance of white noise, per axis.
#[must_use]
pub fn kernel_power(sigma_px: f64) -> f64 {
    let w = kernel(sigma_px);
    w[0] * w[0] + 2.0 * w[1..].iter().map(|v| v * v).sum::<f64>()
}

/// A separable blur of `src` over `out`, a direct sum in a fixed tap order. Reads clamp into
/// `[0, bounds.0) x [0, bounds.1)`.
fn blur_direct(
    src: &Plane,
    out: Rect,
    bounds: (usize, usize),
    sigma: f64,
    opt: RunOptions,
) -> Plane {
    let w = kernel(sigma);
    let r = w.len() - 1;
    let (bw, bh) = bounds;
    let ty0 = out.y0.saturating_sub(r);
    let ty1 = (out.y1 + r).min(bh);
    let trect = Rect {
        x0: out.x0,
        y0: ty0,
        x1: out.x1,
        y1: ty1,
    };
    let mut tmp = Plane::new(trect, opt.poison);
    let ow = out.width();
    par_rows(&mut tmp.data, ow * 3, opt.threads, |row, dst| {
        let y = ty0 + row;
        let base = ((y - src.rect.y0) * src.rect.width()) * 3;
        for xi in 0..ow {
            let x = out.x0 + xi;
            let mut acc = [0.0_f64; 3];
            for c in 0..3 {
                acc[c] = w[0] * f64::from(src.data[base + (x - src.rect.x0) * 3 + c]);
            }
            for i in 1..=r {
                let xl = x.saturating_sub(i);
                let xr = (x + i).min(bw - 1);
                for c in 0..3 {
                    let l = f64::from(src.data[base + (xl - src.rect.x0) * 3 + c]);
                    let rr = f64::from(src.data[base + (xr - src.rect.x0) * 3 + c]);
                    acc[c] += w[i] * (l + rr);
                }
            }
            for c in 0..3 {
                dst[xi * 3 + c] = acc[c] as f32;
            }
        }
    });
    let mut res = Plane::new(out, opt.poison);
    par_rows(&mut res.data, ow * 3, opt.threads, |row, dst| {
        let y = out.y0 + row;
        let line = |yy: usize| &tmp.data[(yy - ty0) * ow * 3..(yy - ty0 + 1) * ow * 3];
        let mut acc: Vec<f64> = line(y).iter().map(|v| w[0] * f64::from(*v)).collect();
        for i in 1..=r {
            let up = line(y.saturating_sub(i));
            let down = line((y + i).min(bh - 1));
            for (k, a) in acc.iter_mut().enumerate() {
                *a += w[i] * (f64::from(up[k]) + f64::from(down[k]));
            }
        }
        for (d, a) in dst.iter_mut().zip(&acc) {
            *d = *a as f32;
        }
    });
    res
}

/// A Gaussian of `sigma` pixels over `out`, reading `src` clamped into the frame. Direct up to
/// [`DIRECT_MAX_SIGMA_PX`], else on the coarse grid. `src` must cover `out` and [`blur_halo`] pixels more.
#[must_use]
pub fn blur(src: &Plane, out: Rect, frame: Frame, sigma: f64, opt: RunOptions) -> Plane {
    let f = decim_factor(sigma);
    if f == 1 {
        return blur_direct(src, out, (frame.width, frame.height), sigma, opt);
    }
    let ff = f as f64;
    // The box of f pixels, the bilinear read back and the blur of the cells add up to the Gaussian.
    let sigma_c = (sigma * sigma - ff * ff / 4.0).max(0.0).sqrt() / ff;
    let rc = (6.0 * sigma_c).ceil() as usize;
    let (ncx, ncy) = (frame.width.div_ceil(f), frame.height.div_ceil(f));
    let centre = |x: f64| (x - 0.5 * (ff - 1.0)) / ff;
    let range = |lo: usize, hi: usize, n: usize| -> (usize, usize) {
        let a = centre(lo as f64).floor().max(0.0) as usize;
        let b = ((centre((hi - 1) as f64).floor() + 1.0).max(0.0) as usize).min(n - 1);
        (a.min(n - 1), b)
    };
    let (cx0, cx1) = range(out.x0, out.x1, ncx);
    let (cy0, cy1) = range(out.y0, out.y1, ncy);
    let need = Rect {
        x0: cx0.saturating_sub(rc),
        y0: cy0.saturating_sub(rc),
        x1: (cx1 + rc + 1).min(ncx),
        y1: (cy1 + rc + 1).min(ncy),
    };
    // The cells: the mean of the pixels of each, in row-major order.
    let mut cells = Plane::new(need, opt.poison);
    let cw = need.width();
    par_rows(&mut cells.data, cw * 3, opt.threads, |row, dst| {
        let j = need.y0 + row;
        let (py0, py1) = (j * f, ((j + 1) * f).min(frame.height));
        for ci in 0..cw {
            let i = need.x0 + ci;
            let (px0, px1) = (i * f, ((i + 1) * f).min(frame.width));
            let mut acc = [0.0_f64; 3];
            for y in py0..py1 {
                for x in px0..px1 {
                    let p = src.px(x, y);
                    for c in 0..3 {
                        acc[c] += f64::from(p[c]);
                    }
                }
            }
            let n = ((py1 - py0) * (px1 - px0)) as f64;
            for c in 0..3 {
                dst[ci * 3 + c] = (acc[c] / n) as f32;
            }
        }
    });
    let want = Rect {
        x0: cx0,
        y0: cy0,
        x1: cx1 + 1,
        y1: cy1 + 1,
    };
    let blurred = blur_direct(&cells, want, (ncx, ncy), sigma_c, opt);
    let mut res = Plane::new(out, opt.poison);
    let ow = out.width();
    par_rows(&mut res.data, ow * 3, opt.threads, |row, dst| {
        let y = out.y0 + row;
        let uy = centre(y as f64);
        let jy = uy.floor();
        let ty = uy - jy;
        let j0 = (jy.max(0.0) as usize).min(ncy - 1);
        let j1 = ((jy + 1.0).max(0.0) as usize).min(ncy - 1);
        for xi in 0..ow {
            let ux = centre((out.x0 + xi) as f64);
            let ix = ux.floor();
            let tx = ux - ix;
            let i0 = (ix.max(0.0) as usize).min(ncx - 1);
            let i1 = ((ix + 1.0).max(0.0) as usize).min(ncx - 1);
            let (a, b, c2, d) = (
                blurred.px(i0, j0),
                blurred.px(i1, j0),
                blurred.px(i0, j1),
                blurred.px(i1, j1),
            );
            for c in 0..3 {
                let top = f64::from(a[c]) * (1.0 - tx) + f64::from(b[c]) * tx;
                let bot = f64::from(c2[c]) * (1.0 - tx) + f64::from(d[c]) * tx;
                dst[xi * 3 + c] = (top * (1.0 - ty) + bot * ty) as f32;
            }
        }
    });
    res
}

/// The direct sum for any width (slow for a wide one): the reference the coarse grid is tested against.
#[must_use]
pub fn blur_reference(src: &Plane, out: Rect, frame: Frame, sigma: f64, opt: RunOptions) -> Plane {
    blur_direct(src, out, (frame.width, frame.height), sigma, opt)
}

/// One Gaussian of a point spread function: its width in pixels and its weight per channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term {
    /// Sigma in pixels.
    pub sigma_px: f64,
    /// Weight of the term in each channel.
    pub weight: Vec3,
}

/// The largest halo of a set of terms.
#[must_use]
pub fn terms_halo(terms: &[Term]) -> usize {
    terms
        .iter()
        .map(|t| blur_halo(t.sigma_px))
        .max()
        .unwrap_or(0)
}

/// `centre * src + sum_t weight_t * (gauss_t * src)` over `out`, with the centre weight `1 - sum weight` (so a
/// uniform area is unchanged). The terms narrower than [`MIN_SIGMA_PX`] stay at the pixel.
fn mix(src: &Plane, out: Rect, frame: Frame, terms: &[Term], opt: RunOptions) -> Plane {
    let mut centre = [1.0_f64; 3];
    for t in terms {
        for c in 0..3 {
            centre[c] -= t.weight[c];
        }
    }
    let mut live: Vec<&Term> = Vec::new();
    for t in terms {
        if t.sigma_px >= MIN_SIGMA_PX {
            live.push(t);
        } else {
            for c in 0..3 {
                centre[c] += t.weight[c];
            }
        }
    }
    let ow = out.width();
    let mut acc: Vec<f64> = vec![0.0; ow * out.height() * 3];
    par_rows(&mut acc, ow * 3, opt.threads, |row, dst| {
        let y = out.y0 + row;
        for xi in 0..ow {
            let p = src.px(out.x0 + xi, y);
            for c in 0..3 {
                dst[xi * 3 + c] = centre[c] * f64::from(p[c]);
            }
        }
    });
    for t in live {
        let b = blur(src, out, frame, t.sigma_px, opt);
        par_rows(&mut acc, ow * 3, opt.threads, |row, dst| {
            for xi in 0..ow {
                for c in 0..3 {
                    dst[xi * 3 + c] += t.weight[c] * f64::from(b.data[(row * ow + xi) * 3 + c]);
                }
            }
        });
    }
    let mut res = Plane::new(out, opt.poison);
    par_rows(&mut res.data, ow * 3, opt.threads, |row, dst| {
        for (d, s) in dst.iter_mut().zip(&acc[row * ow * 3..(row + 1) * ow * 3]) {
            *d = *s as f32;
        }
    });
    res
}

// ---------------------------------------------------------------------------------------------------------
// The model: our numbers, and what a profile adds
// ---------------------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct Scatter {
    sigma_um: f64,
    weight: f64,
}

#[derive(Deserialize)]
struct ClassWeight {
    strong: f64,
    weak: f64,
    none: f64,
}

#[derive(Deserialize)]
struct HalationShape {
    shape: Vec<[f64; 2]>,
    class_weight: ClassWeight,
    channel_factor: [f64; 3],
}

#[derive(Deserialize)]
struct HalationHeader {
    scatter: Scatter,
    halation: HalationShape,
    max_total_weight: f64,
}

#[derive(Deserialize)]
struct HalationFile {
    #[serde(rename = "_header")]
    header: HalationHeader,
}

#[derive(Deserialize)]
struct DiffusionPart {
    shape: Vec<[f64; 2]>,
}

#[derive(Deserialize)]
struct DiffusionFile {
    diffusion: DiffusionPart,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, name: &str) -> Result<T, FilmError> {
    let bytes = std::fs::read(path).map_err(|source| FilmError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|e| FilmError::data(name.to_string(), format!("it does not parse ({e})")))
}

fn check_shape(shape: &[[f64; 2]], name: &str) -> Result<Vec<[f64; 2]>, FilmError> {
    let total: f64 = shape.iter().map(|s| s[0]).sum();
    let bad = |s: &[f64; 2]| s[0].is_nan() || s[0] < 0.0 || s[1].is_nan() || s[1] <= 0.0;
    if shape.is_empty() || shape.iter().any(bad) || total.is_nan() || total <= 0.0 {
        return Err(FilmError::data(
            name.to_string(),
            "a shape needs positive widths and shares",
        ));
    }
    Ok(shape.iter().map(|s| [s[0] / total, s[1]]).collect())
}

/// The antihalation class of a profile (`info.antihalation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Antihalation {
    /// A backing that absorbs most of the returning light.
    Strong,
    /// A thin dye layer.
    Weak,
    /// None.
    None,
}

/// What the spatial passes read of one profile file, besides what [`crate::Profile`] holds.
#[derive(Debug, Clone)]
pub struct ProfileExtras {
    /// The antihalation class.
    pub antihalation: Antihalation,
    /// The density of each sublayer and the total, along the exposure axis.
    pub layers: Layers,
}

impl ProfileExtras {
    /// Reads `<profiles>/<id>.json` (the antihalation class and `density_curves_layers`).
    ///
    /// # Errors
    ///
    /// [`FilmError`] when the file is missing, damaged, or its layers do not have the shape 256 x 3 x 3.
    pub fn load(profiles: &Path, id: &str) -> Result<Self, FilmError> {
        let path = profiles.join(format!("{id}.json"));
        let name = format!("{id}.json");
        let v: serde_json::Value = {
            let bytes = std::fs::read(&path).map_err(|source| FilmError::Io {
                path: path.clone(),
                source,
            })?;
            serde_json::from_slice(&bytes)
                .map_err(|e| FilmError::data(name.clone(), format!("it does not parse ({e})")))?
        };
        let bad = |what: &str| FilmError::data(name.clone(), what.to_string());
        let antihalation = match v["info"]["antihalation"].as_str() {
            Some("strong") => Antihalation::Strong,
            Some("weak") => Antihalation::Weak,
            Some("none") => Antihalation::None,
            _ => return Err(bad("info.antihalation is not strong, weak or none")),
        };
        let num = |x: &serde_json::Value| x.as_f64().unwrap_or(0.0);
        let layers_v = v["data"]["density_curves_layers"]
            .as_array()
            .ok_or_else(|| bad("data.density_curves_layers is missing"))?;
        let mut sublayers = Vec::with_capacity(layers_v.len());
        let mut total = Vec::with_capacity(layers_v.len());
        for row in layers_v {
            let rows = row.as_array().filter(|r| r.len() == N_SUBLAYERS);
            let rows = rows.ok_or_else(|| bad("a density_curves_layers row is not 3 x 3"))?;
            let mut m = [[0.0; 3]; N_SUBLAYERS];
            let mut t = [0.0; 3];
            for (s, r) in rows.iter().enumerate() {
                let r = r
                    .as_array()
                    .filter(|r| r.len() == 3)
                    .ok_or_else(|| bad("a density_curves_layers row is not 3 x 3"))?;
                for c in 0..3 {
                    m[s][c] = num(&r[c]);
                    t[c] += m[s][c];
                }
            }
            sublayers.push(m);
            total.push(t);
        }
        Ok(Self {
            antihalation,
            layers: Layers { sublayers, total },
        })
    }
}

/// The halation model of one stock: the widths and the weights per channel at 100 %.
#[derive(Debug, Clone)]
pub struct HalationModel {
    scatter_sigma_um: f64,
    scatter_weight: f64,
    /// `[share, sigma_um]`, the shares summing to 1.
    shape: Vec<[f64; 2]>,
    /// The halation weight per channel at 100 %.
    weight: Vec3,
    max_total: f64,
}

impl HalationModel {
    /// The weight of the returning light per channel at Amount 100 %.
    #[must_use]
    pub fn weight(&self) -> Vec3 {
        self.weight
    }
}

/// Everything the spatial passes read of a stock: our parameter files and the profile's extras.
#[derive(Debug, Clone)]
pub struct SpatialModel {
    halation: HalationModel,
    /// `[share, sigma_um]` of the coupler inhibitor's spread.
    diffusion: Vec<[f64; 2]>,
    grain: GrainModel,
}

impl SpatialModel {
    /// Reads `<data_dir>/film/params/` and the profile of `film_id`.
    ///
    /// # Errors
    ///
    /// [`FilmError`] for a missing or damaged file or an unknown stock.
    pub fn load(data_dir: &Path, film_id: &str) -> Result<Self, FilmError> {
        let root = data_dir.join("film");
        let params = root.join("params");
        let names = Names::load(&params)?;
        let name = names
            .get(film_id)
            .ok_or_else(|| FilmError::UnknownStock(film_id.to_string()))?;
        let extras = ProfileExtras::load(&root.join("spektrafilm").join("profiles"), film_id)?;
        let h: HalationFile = read_json(&params.join("halation.json"), "params/halation.json")?;
        let h = h.header;
        let class = match extras.antihalation {
            Antihalation::Strong => h.halation.class_weight.strong,
            Antihalation::Weak => h.halation.class_weight.weak,
            Antihalation::None => h.halation.class_weight.none,
        };
        let shape = check_shape(&h.halation.shape, "params/halation.json")?;
        let weight = h.halation.channel_factor.map(|f| class * f);
        let halation = HalationModel {
            scatter_sigma_um: h.scatter.sigma_um,
            scatter_weight: h.scatter.weight,
            shape,
            weight,
            max_total: h.max_total_weight,
        };
        let d: DiffusionFile = read_json(&params.join("couplers.json"), "params/couplers.json")?;
        let diffusion = check_shape(&d.diffusion.shape, "params/couplers.json")?;
        let gp = GrainParams::load(&params)?;
        let (rms, fallback) = gp.rms_x1000(film_id, name.iso, name.kind);
        let grain = GrainModel::new(&extras.layers, &gp, rms, fallback);
        Ok(Self {
            halation,
            diffusion,
            grain,
        })
    }

    /// The halation model.
    #[must_use]
    pub fn halation(&self) -> &HalationModel {
        &self.halation
    }

    /// The calibrated grain.
    #[must_use]
    pub fn grain(&self) -> &GrainModel {
        &self.grain
    }
}

// ---------------------------------------------------------------------------------------------------------
// The node's spatial settings
// ---------------------------------------------------------------------------------------------------------

/// The settings of the spatial passes (`docs/24` section 4.5). 1.0 is 100 %.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialSettings {
    /// Halation Amount: the stock's own halation times this (0 to 3). 0 leaves only the emulsion's scatter.
    pub halation_amount: f64,
    /// Halation Size (0.5 to 2).
    pub halation_size: f64,
    /// Grain Amount: the standard deviation times this (0 to 2). 0 turns the pass off.
    pub grain_amount: f64,
    /// Grain Size: the particle's side times this (0.5 to 2).
    pub grain_size: f64,
    /// The grain's seed.
    pub grain_seed: u32,
    /// The width of the coupler inhibitor's spread, times this. 0 turns the diffusion off (the couplers act
    /// per pixel).
    pub diffusion_size: f64,
}

impl Default for SpatialSettings {
    fn default() -> Self {
        Self {
            halation_amount: 1.0,
            halation_size: 1.0,
            grain_amount: 1.0,
            grain_size: 1.0,
            grain_seed: 0,
            diffusion_size: 1.0,
        }
    }
}

impl SpatialSettings {
    /// Halation, grain and the diffusion off; the emulsion's scatter stays.
    #[must_use]
    pub fn off() -> Self {
        Self {
            halation_amount: 0.0,
            grain_amount: 0.0,
            diffusion_size: 0.0,
            ..Self::default()
        }
    }
}

/// The halos of the three spatial passes at a frame size, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Halos {
    /// P2, scatter and halation.
    pub halation: usize,
    /// P3, the coupler diffusion.
    pub diffusion: usize,
    /// P4, the grain's dye cloud.
    pub grain: usize,
}

impl Halos {
    /// The sum: how far beyond a tile the first pass reads.
    #[must_use]
    pub fn total(&self) -> usize {
        self.halation + self.diffusion + self.grain
    }
}

struct Plan {
    halation: Vec<Term>,
    diffusion: Vec<Term>,
    /// The dye cloud's sigma in pixels; the pass is on when `grain_amount > 0`.
    cloud_px: f64,
    halos: Halos,
}

/// An error of a tile run.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpatialError {
    /// The input does not cover what the passes read.
    #[error(
        "the input covers {have:?} but the tile {tile:?} needs {need:?} (the tile and a halo of the passes)"
    )]
    InputTooSmall {
        /// What was given.
        have: Rect,
        /// The tile.
        tile: Rect,
        /// What is needed.
        need: Rect,
    },
    /// The tile is empty or outside the frame.
    #[error("the tile {0:?} is empty or outside the frame")]
    BadTile(Rect),
}

// ---------------------------------------------------------------------------------------------------------
// The Film node on a tile
// ---------------------------------------------------------------------------------------------------------

/// The Film node's passes P1 to P5 on a tile of the photo frame: [`BakedFilm`] for the colour, the spatial
/// passes between.
#[derive(Debug, Clone)]
pub struct FilmFrame {
    baked: Arc<BakedFilm>,
    model: Arc<SpatialModel>,
    settings: SpatialSettings,
}

impl FilmFrame {
    /// Builds the node from the baked colour passes, the stock's spatial model and the settings.
    #[must_use]
    pub fn new(baked: BakedFilm, model: SpatialModel, settings: SpatialSettings) -> Self {
        Self {
            baked: Arc::new(baked),
            model: Arc::new(model),
            settings,
        }
    }

    /// The baked colour passes.
    #[must_use]
    pub fn baked(&self) -> &BakedFilm {
        &self.baked
    }

    /// The settings.
    #[must_use]
    pub fn settings(&self) -> &SpatialSettings {
        &self.settings
    }

    fn plan(&self, frame: Frame) -> Plan {
        let s = &self.settings;
        let px = frame.px_per_um();
        let h = &self.model.halation;
        let mut halation = Vec::new();
        let amount = s.halation_amount.max(0.0);
        let mut w = h.weight.map(|v| v * amount);
        let total = |w: Vec3| h.scatter_weight + w.iter().copied().fold(0.0, f64::max);
        let t = total(w);
        if t > h.max_total {
            let k = (h.max_total - h.scatter_weight).max(0.0)
                / w.iter().copied().fold(f64::MIN_POSITIVE, f64::max);
            w = w.map(|v| v * k);
        }
        if h.scatter_weight > 0.0 {
            halation.push(Term {
                sigma_px: h.scatter_sigma_um * px,
                weight: [h.scatter_weight; 3],
            });
        }
        if amount > 0.0 {
            for sh in &h.shape {
                halation.push(Term {
                    sigma_px: sh[1] * s.halation_size * px,
                    weight: w.map(|v| v * sh[0]),
                });
            }
        }
        let mut diffusion = Vec::new();
        if s.diffusion_size > 0.0 {
            for sh in &self.model.diffusion {
                diffusion.push(Term {
                    sigma_px: sh[1] * s.diffusion_size * px,
                    weight: [sh[0]; 3],
                });
            }
        }
        let cloud_px = self.model.grain.cloud_sigma_um * s.grain_size * px;
        let grain_on = s.grain_amount > 0.0;
        let halos = Halos {
            halation: terms_halo(&halation),
            diffusion: terms_halo(&diffusion),
            grain: if grain_on { blur_halo(cloud_px) } else { 0 },
        };
        Plan {
            halation,
            diffusion,
            cloud_px,
            halos,
        }
    }

    /// The halos of the passes at a frame size.
    #[must_use]
    pub fn halos(&self, frame: Frame) -> Halos {
        self.plan(frame).halos
    }

    /// The part of the LogC4 input a tile needs: the tile and every halo, cut to the frame.
    #[must_use]
    pub fn required_input(&self, frame: Frame, tile: Rect) -> Rect {
        tile.expand(self.halos(frame).total(), frame)
    }

    /// Runs the node on `tile` of the photo frame. `input` is LogC4 / AWG4 (f32, RGB) and covers at least
    /// [`FilmFrame::required_input`]. The result is LogC4 / AWG4 over `tile`, the same bytes the whole frame
    /// gives at those pixels, for any tile, any thread count and any history of buffers.
    ///
    /// # Errors
    ///
    /// [`SpatialError`] when the tile is empty or outside the frame, or the input is too small.
    pub fn run_tile(
        &self,
        frame: Frame,
        tile: Rect,
        input: &Plane,
        opt: RunOptions,
    ) -> Result<Plane, SpatialError> {
        if tile.width() == 0 || tile.height() == 0 || !frame.rect().contains(&tile) {
            return Err(SpatialError::BadTile(tile));
        }
        let plan = self.plan(frame);
        let h = plan.halos;
        let need = tile.expand(h.total(), frame);
        if !input.rect.contains(&need) {
            return Err(SpatialError::InputTooSmall {
                have: input.rect,
                tile,
                need,
            });
        }
        let s = &self.settings;
        let baked = &*self.baked;
        // Regions, each cut to the frame: D and the inhibitor on `r_d`, E' and the first inhibitor on `r_q`,
        // E on `r_e`.
        let r_d = tile.expand(h.grain, frame);
        let r_q = r_d.expand(h.diffusion, frame);
        let r_e = r_q.expand(h.halation, frame);

        // P1: colour in, LogC4 to the raw exposure E (linear).
        let mut e = Plane::new(r_e, opt.poison);
        let ew = r_e.width();
        par_rows(&mut e.data, ew * 3, opt.threads, |row, dst| {
            let y = r_e.y0 + row;
            for xi in 0..ew {
                let p = input.px(r_e.x0 + xi, y);
                let le = baked.exposure_of([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                for c in 0..3 {
                    dst[xi * 3 + c] = pow10(le[c]) as f32;
                }
            }
        });

        // P2: scatter and halation, E to E'.
        let e2 = if plan.halation.is_empty() {
            e
        } else {
            mix(&e, r_q, frame, &plan.halation, opt)
        };

        // P3: develop, then the coupler field.
        let log_e = |p: [f32; 3]| -> Vec3 { p.map(|v| f64::from(v).max(1e-30).log10()) };
        let diffuse = !plan.diffusion.is_empty() && baked_has_couplers(baked);
        let mut d = Plane::new(r_d, opt.poison);
        let dw = r_d.width();
        if diffuse {
            // The per-pixel solve gives the density and the inhibitor it releases; the inhibitor spreads;
            // the pixel develops once more with the spread inhibitor. On a uniform area the spread changes
            // nothing and the second development is the fixed point of the first.
            let mut q0 = Plane::new(r_q, opt.poison);
            let qw = r_q.width();
            par_rows(&mut q0.data, qw * 3, opt.threads, |row, dst| {
                let y = r_q.y0 + row;
                for xi in 0..qw {
                    let le = log_e(e2.px(r_q.x0 + xi, y));
                    let q = baked.coupler_inhibitor(baked.film_density(le));
                    for c in 0..3 {
                        dst[xi * 3 + c] = q[c] as f32;
                    }
                }
            });
            let q = mix(&q0, r_d, frame, &plan.diffusion, opt);
            par_rows(&mut d.data, dw * 3, opt.threads, |row, dst| {
                let y = r_d.y0 + row;
                for xi in 0..dw {
                    let x = r_d.x0 + xi;
                    let le = log_e(e2.px(x, y));
                    let qq = q.px(x, y);
                    let v = baked.density_with_inhibitor(le, qq.map(f64::from));
                    for c in 0..3 {
                        dst[xi * 3 + c] = v[c] as f32;
                    }
                }
            });
        } else {
            par_rows(&mut d.data, dw * 3, opt.threads, |row, dst| {
                let y = r_d.y0 + row;
                for xi in 0..dw {
                    let v = baked.film_density(log_e(e2.px(r_d.x0 + xi, y)));
                    for c in 0..3 {
                        dst[xi * 3 + c] = v[c] as f32;
                    }
                }
            });
        }
        drop(e2);

        // P4: grain.
        let d = if s.grain_amount > 0.0 {
            self.grain_pass(&d, tile, frame, &plan, opt)
        } else {
            d
        };

        // P5: colour out, density to LogC4.
        let mut out = Plane::new(tile, opt.poison);
        let tw = tile.width();
        par_rows(&mut out.data, tw * 3, opt.threads, |row, dst| {
            let y = tile.y0 + row;
            for xi in 0..tw {
                let p = d.px(tile.x0 + xi, y);
                let v = baked.colour_out(p.map(f64::from));
                for c in 0..3 {
                    dst[xi * 3 + c] = v[c] as f32;
                }
            }
        });
        Ok(out)
    }

    /// Pass P4 alone: adds the grain to a film density `d` over `tile`. `d` covers the tile and the dye cloud's
    /// halo (cut to the frame).
    #[must_use]
    pub fn apply_grain(&self, d: &Plane, tile: Rect, frame: Frame, opt: RunOptions) -> Plane {
        self.grain_pass(d, tile, frame, &self.plan(frame), opt)
    }

    fn grain_pass(
        &self,
        d: &Plane,
        tile: Rect,
        frame: Frame,
        plan: &Plan,
        opt: RunOptions,
    ) -> Plane {
        let s = &self.settings;
        let model = &self.model.grain;
        let rect = d.rect;
        let w = rect.width();
        // The standard deviation of density per um of pixel side is in `sigma`; the pixel's side in um is
        // 1 / px_per_um, and Size and Amount scale the amplitude.
        let scale = frame.px_per_um() * s.grain_amount * s.grain_size;
        let seed = s.grain_seed;
        let mut noise = Plane::new(rect, opt.poison);
        par_rows(&mut noise.data, w * 3, opt.threads, |row, dst| {
            let y = rect.y0 + row;
            for xi in 0..w {
                let x = rect.x0 + xi;
                let p = d.px(x, y);
                for c in 0..3 {
                    let dens = f64::from(p[c]);
                    let mut g = 0.0;
                    for sub in 0..N_SUBLAYERS {
                        let n = grain::normal(seed, x as u32, y as u32, c as u32, sub as u32);
                        g += model.amplitude(c, sub, dens) * n;
                    }
                    dst[xi * 3 + c] = (g * scale) as f32;
                }
            }
        });
        // The dye cloud: the blur keeps the variance (it divides by the kernel's power).
        let (cloud, norm) = if plan.cloud_px >= MIN_SIGMA_PX {
            let b = blur(&noise, tile, frame, plan.cloud_px, opt);
            (b, 1.0 / kernel_power(plan.cloud_px))
        } else {
            (noise, 1.0)
        };
        let mut out = Plane::new(tile, opt.poison);
        let tw = tile.width();
        par_rows(&mut out.data, tw * 3, opt.threads, |row, dst| {
            let y = tile.y0 + row;
            for xi in 0..tw {
                let x = tile.x0 + xi;
                let p = d.px(x, y);
                let g = cloud.px(x, y);
                for c in 0..3 {
                    dst[xi * 3 + c] = (f64::from(p[c]) + f64::from(g[c]) * norm) as f32;
                }
            }
        });
        out
    }
}

fn baked_has_couplers(b: &BakedFilm) -> bool {
    b.coupler_amount() != 0.0
}
