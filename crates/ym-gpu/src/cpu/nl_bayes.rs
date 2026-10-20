//! Methods B and C of `docs/algorithms/05-single-frame-denoise.md`: multiscale NL-Bayes on 2 × 2 patches
//! (sections B/C.1, B/C.2 and B/C.5, read as *Yellowmoon's reading* G1–G5 and BC-a to BC-f say).
//!
//! # Parameters (B/C.1)
//!
//! | | B | C |
//! |---|---|---|
//! | planes | R, G, B | Y, Cb, Cr (BT.709, [`ycbcr`]) |
//! | group size `K` | 8 | 24 |
//! | search window `S`, half-width `h` | `R + 6`, `floor(S / 2)` | the same |
//! | σ of channel 0 | `0.02 L` | `0.0125 L` |
//! | σ of channels 1 and 2 | `0.02 L` | `0.02 C` |
//! | pyramid base | 128 | 256 |
//! | intensity model `s` | `(clamp(mean(ȳ), 0.2, 0.8) / m_flat)^(−7/6)` | 1 |
//! | final-step noise gain | 2 | 1 |
//! | patch-search guide | `KR R + KG G + KB B` ([`ycbcr::KR`], ...) | Y |
//!
//! # Pipeline (B/C.2)
//!
//! [`measure`] finds the depth `n` ([`flat_regions::pyramid_depth`]), runs the detector on the coarsest level of
//! the *RGB* pyramid for both methods ([`flat_regions::detect_frame`], reading BC-b), builds the working pyramid
//! (method C converts to Y/Cb/Cr first, then downsamples) and measures one [`Profile`] per level and channel on the
//! levels as built (reading BC-c). [`denoise_with`] walks the pyramid coarse to fine: for `k = n` down to 1,
//! `D_k` = the level-`k` denoise of the current level `k`, then the current level `k − 1` is the level as built
//! plus `up(D_k − level_k)` (`level_k` as built; `up` is linear, so this is the text's
//! `level_{k−1} + up(D_k) − up(level_k)` with one upsampling instead of two); then level 0 is denoised. Method C
//! converts back to RGB; last, the blend with the input ([`ycbcr::blend`], reading G5). [`denoise`] is both
//! steps on one pyramid.
//!
//! # One level (B/C.5)
//!
//! * Reference patch corners (reading BC-d) are the even frame positions from 0 to `W − 2`, plus `W − 2` when it
//!   is odd (the same for `y`); the level's frame position is the image's placement halved per level.
//! * The search window `[ref − h, ref − h + S)` is clipped to the corners `[0, W − 2] × [0, H − 2]`; the distance
//!   is the L1 sum over the guide's 4 pixels, summed in patch order; the group is the reference, then the
//!   `K − 1` others of smallest distance (ties: smaller raster index first), in that order, or all the others
//!   when there are fewer. `K'` is the group's actual size; a group of one is left unfiltered.
//! * Pass 1, per channel: `ȳ`, `Cy = Yc Ycᵀ / (K' − 1)`, `Cn[i][j] = σ² P[i][j] · 0.6^|i−j| · s`,
//!   `Cy[i][i] = max(Cy[i][i], Cn[i][i])`, `x̂ = ȳ + (yc − Cn Cy⁻¹ yc)`. Pass 2 searches again on the basic
//!   image's guide: `Cb = Ybc Ybcᵀ / (K' − 1)` about the basic patches' own mean, `Cn` as in pass 1 (`s` from the
//!   noisy `ȳ`) times the final gain, `x̂ = ȳ + Cb (Cb + Cn)⁻¹ (y − ȳ)`. The inverses are 4 × 4 LDLᵀ; a pivot
//!   `≤ 2⁻²³` (or not a number) leaves that group of that channel unfiltered, `x̂ = y`, in either pass (reading
//!   BC-e).
//! * Aggregation (reading BC-f, G3): every estimate is added to its 4 pixels, one sum per channel and one count
//!   per pixel shared by the channels; the output is sum / count. References run in passes `P` px apart (`P`
//!   the smallest even number `≥ S + 1`, so the `S + 1` px footprints of one pass never overlap): the regular
//!   passes `(i, j)`, `i, j < P / 2`, in raster order (`j` outer), then the extra column of BC-d in passes by
//!   `j`, the extra row in passes by `i`, then the extra corner. The references of a pass run in parallel
//!   (each row of references owns the accumulator rows of its footprint); a pixel receives at most one
//!   reference per pass, its patches in slot order. The result is the same bits for any number of threads.
//!
//! # Points the reading leaves open (decided here)
//!
//! * `0.6^|i−j|` is a table of products and the intensity model's power is `libm::pow`, so the bits do not
//!   depend on the platform's maths library (as method A's cosine table).
//! * A channel whose σ is 0 is passed through at every level (the formulas give `x̂ = y` there too).
//! * Method B: a channel whose `m_flat` is not positive at a level is passed through at that level (the
//!   intensity model divides by it).
//! * A level narrower or lower than 2 px is passed through.
//! * No flat region at all, or no σ above 0: the input is returned as it is.
//!
//! # Borders and tiling
//!
//! Windows are clipped, never mirrored. The pass grid, the parity of the reference corners and the extra
//! column and row follow the frame position ([`Placement`], reading G4), but the windows are clipped to the
//! buffer and the profiles are frame-level ([`Measured`]); tiles are not exact yet.

use std::borrow::Cow;

use rayon::prelude::*;

use super::flat_regions::{self, Profile, Region};
use super::ycbcr::{self, KB, KG, KR, Planes};
use super::{Placement, RgbImage};

/// Pivot limit of the 4 × 4 LDLᵀ inversion: a pivot at or below 2⁻²³ leaves the group unfiltered.
const PIVOT_MIN: f64 = 1.0 / 8_388_608.0;

/// `0.6^|i − j|` by `|i − j|`, as products (each correctly rounded, so the same bits everywhere).
const TAPER: [f64; 4] = [1.0, 0.6, 0.6 * 0.6, 0.6 * 0.6 * 0.6];

/// A 4 × 4 matrix (patch order `(0,0), (1,0), (0,1), (1,1)`).
type Mat4 = [[f64; 4]; 4];
/// A 2 × 2 patch as a 4-vector.
type Vec4 = [f64; 4];

/// Which of the two NL-Bayes methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// Method B: R, G, B planes, 8 patches per group, the intensity noise model.
    B,
    /// Method C: Y, Cb, Cr planes (BT.709), 24 patches per group.
    C,
}

impl Variant {
    /// The group size `K`.
    pub const fn group_size(self) -> usize {
        match self {
            Variant::B => 8,
            Variant::C => 24,
        }
    }

    /// The pyramid base.
    pub const fn pyramid_base(self) -> usize {
        match self {
            Variant::B => 128,
            Variant::C => 256,
        }
    }

    /// The final-step (pass 2) noise gain.
    pub const fn final_gain(self) -> f64 {
        match self {
            Variant::B => 2.0,
            Variant::C => 1.0,
        }
    }

    /// The gains `[σ1, σ2, σ2]` on the measured noise of the three channels, from the luma and chroma strengths
    /// (0 to 100). Method B ignores `chroma`.
    pub fn sigmas(self, luma: f64, chroma: f64) -> [f64; 3] {
        match self {
            Variant::B => [0.02 * luma; 3],
            Variant::C => [0.0125 * luma, 0.02 * chroma, 0.02 * chroma],
        }
    }
}

/// The inputs of methods B and C.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NlBayesParams {
    /// Method B or C.
    pub variant: Variant,
    /// Luma strength `L`, 0 to 100 (all three channels for method B).
    pub luma: f64,
    /// Chroma strength `C`, 0 to 100 (method C only).
    pub chroma: f64,
    /// Radius `R`; the search window is `R + 6` px.
    pub radius: usize,
    /// Blend `B` with the input, 0 to 100 (0: none).
    pub blend: f64,
}

impl NlBayesParams {
    /// The search window `S = R + 6` in pixels.
    pub const fn search_window(&self) -> usize {
        self.radius + 6
    }
}

/// What a denoise measures on the whole frame before it filters, separable from the filtering (so that tiles
/// can later share one measurement).
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    /// The method the profiles were measured for (their planes: RGB for B, Y/Cb/Cr for C).
    pub variant: Variant,
    /// The pyramid depth `n` (levels 0 to `n`).
    pub depth: usize,
    /// The flat regions of the coarsest level, sorted by score; empty means the input is returned.
    pub regions: Vec<Region>,
    /// Per level (0 to `n`), per channel: the profile, or `None` when no region is usable there (that channel
    /// passes through at that level).
    pub profiles: Vec<[Option<Profile>; 3]>,
}

/// One pyramid level as built.
struct Level {
    width: usize,
    height: usize,
    place: Placement,
    planes: Planes,
}

/// The placement of the next coarser level: the frame position halved, the frame size halved up.
fn coarser(p: Placement) -> Placement {
    Placement {
        x0: p.x0 / 2,
        y0: p.y0 / 2,
        frame_width: p.frame_width.div_ceil(2),
        frame_height: p.frame_height.div_ceil(2),
    }
}

/// The working pyramid (RGB for B, Y/Cb/Cr for C) of `depth` steps.
fn pyramid(img: &RgbImage, variant: Variant, depth: usize) -> Vec<Level> {
    let planes = match variant {
        Variant::B => ycbcr::rgb_planes(img),
        Variant::C => ycbcr::to_ycbcr(img),
    };
    let mut levels = vec![Level {
        width: img.width,
        height: img.height,
        place: img.place,
        planes,
    }];
    for _ in 0..depth {
        let Some(last) = levels.last() else { break };
        let (cw, ch) = (last.width.div_ceil(2), last.height.div_ceil(2));
        let planes = std::array::from_fn(|c| {
            flat_regions::downsample(&last.planes[c], last.width, last.height).0
        });
        let place = coarser(last.place);
        levels.push(Level {
            width: cw,
            height: ch,
            place,
            planes,
        });
    }
    levels
}

/// The profiles of every level and channel of a pyramid as built.
fn profiles_of(pyr: &[Level], regions: &[Region]) -> Vec<[Option<Profile>; 3]> {
    pyr.iter()
        .map(|lv| {
            std::array::from_fn(|c| {
                flat_regions::profile(&lv.planes[c], lv.width, lv.height, regions)
            })
        })
        .collect()
}

/// Measures what [`denoise_with`] needs: the depth, the flat regions and the per-level, per-channel profiles
/// (`img` is read as the whole frame).
pub fn measure(img: &RgbImage, variant: Variant) -> Measured {
    let (depth, regions) = flat_regions::detect_frame(img, variant.pyramid_base());
    let profiles = if regions.is_empty() {
        vec![[None; 3]; depth + 1]
    } else {
        profiles_of(&pyramid(img, variant, depth), &regions)
    };
    Measured {
        variant,
        depth,
        regions,
        profiles,
    }
}

/// Method B or C on a whole frame: [`measure`], then [`denoise_with`], on one pyramid.
pub fn denoise(img: &RgbImage, p: &NlBayesParams) -> RgbImage {
    let (depth, regions) = flat_regions::detect_frame(img, p.variant.pyramid_base());
    if regions.is_empty() || !has_strength(p) {
        return img.clone();
    }
    let pyr = pyramid(img, p.variant, depth);
    let m = Measured {
        variant: p.variant,
        depth,
        profiles: profiles_of(&pyr, &regions),
        regions,
    };
    finish(img, p, &m, &pyr)
}

/// Method B or C with measurements made beforehand by [`measure`] for the same variant.
pub fn denoise_with(img: &RgbImage, p: &NlBayesParams, m: &Measured) -> RgbImage {
    debug_assert_eq!(m.variant, p.variant, "profiles measured for another method");
    if m.regions.is_empty() || !has_strength(p) {
        return img.clone();
    }
    let pyr = pyramid(img, p.variant, m.depth);
    finish(img, p, m, &pyr)
}

/// Whether any channel has a σ above 0.
fn has_strength(p: &NlBayesParams) -> bool {
    p.variant.sigmas(p.luma, p.chroma).iter().any(|&s| s > 0.0)
}

/// The coarse-to-fine walk, the conversion back to RGB and the blend.
fn finish(img: &RgbImage, p: &NlBayesParams, m: &Measured, pyr: &[Level]) -> RgbImage {
    let n = pyr.len() - 1; // `pyramid` always gives level 0
    let profiles = |k: usize| m.profiles.get(k);
    let mut cur = pyr[n].planes.clone();
    for k in (1..=n).rev() {
        let (lv, fine) = (&pyr[k], &pyr[k - 1]);
        let d = denoise_level(lv, &cur, p, profiles(k));
        cur = std::array::from_fn(|c| {
            let diff: Vec<f64> = d[c].iter().zip(&lv.planes[c]).map(|(a, b)| a - b).collect();
            let up = flat_regions::upsample(&diff, lv.width, lv.height, fine.width, fine.height);
            fine.planes[c].iter().zip(&up).map(|(a, b)| a + b).collect()
        });
    }
    let planes = denoise_level(&pyr[0], &cur, p, profiles(0));
    let out = match p.variant {
        Variant::B => ycbcr::image_from_rgb(&planes, img),
        Variant::C => ycbcr::from_ycbcr(&planes, img),
    };
    ycbcr::blend(out, img, p.blend)
}

/// A channel's noise model at one level.
#[derive(Clone, Copy, Debug)]
struct NoiseModel {
    /// `σ² P[i][j] · 0.6^|i−j|`.
    base: Mat4,
    /// `m_flat` for the intensity model (method B); `None` for method C (`s = 1`).
    m_flat: Option<f64>,
}

impl NoiseModel {
    /// The model of a channel with gain `sigma`, or `None` when the channel passes through.
    fn new(prof: &Profile, sigma: f64, variant: Variant) -> Option<NoiseModel> {
        if sigma.is_nan() || sigma <= 0.0 || prof.p.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let m_flat = match variant {
            Variant::B if prof.m_flat > 0.0 && prof.m_flat.is_finite() => Some(prof.m_flat),
            Variant::B => return None,
            Variant::C => None,
        };
        let s2 = sigma * sigma;
        let base = std::array::from_fn(|i| {
            std::array::from_fn(|j| s2 * prof.p[i][j] * TAPER[i.abs_diff(j)])
        });
        Some(NoiseModel { base, m_flat })
    }

    /// The group noise covariance `Cn` for a group of (noisy) mean `mean`, times `gain`.
    fn cn(&self, mean: &Vec4, gain: f64) -> Mat4 {
        let s = match self.m_flat {
            Some(m) => {
                let mu = (mean[0] + mean[1] + mean[2] + mean[3]) / 4.0;
                libm::pow(mu.clamp(0.2, 0.8) / m, -7.0 / 6.0)
            }
            None => 1.0,
        };
        let f = s * gain;
        self.base.map(|row| row.map(|v| v * f))
    }
}

/// The patch-search guide of a level's planes.
fn guide(variant: Variant, planes: &Planes) -> Cow<'_, [f64]> {
    match variant {
        Variant::B => {
            let [r, g, b] = planes;
            Cow::Owned(
                r.iter()
                    .zip(g)
                    .zip(b)
                    .map(|((r, g), b)| KR * r + KG * g + KB * b)
                    .collect(),
            )
        }
        Variant::C => Cow::Borrowed(&planes[0]),
    }
}

/// Denoises one level (both passes); `cur` is the level's current planes, `geom` the level as built (size and
/// place).
fn denoise_level(
    geom: &Level,
    cur: &Planes,
    p: &NlBayesParams,
    profiles: Option<&[Option<Profile>; 3]>,
) -> Planes {
    let sig = p.variant.sigmas(p.luma, p.chroma);
    let models: [Option<NoiseModel>; 3] = std::array::from_fn(|c| {
        profiles
            .and_then(|pr| pr[c].as_ref())
            .and_then(|prof| NoiseModel::new(prof, sig[c], p.variant))
    });
    if models.iter().all(Option::is_none) || geom.width < 2 || geom.height < 2 {
        return cur.clone();
    }
    let window = p.search_window();
    let ctx = LevelCtx {
        width: geom.width,
        height: geom.height,
        place: geom.place,
        noisy: cur,
        models,
        group: p.variant.group_size(),
        window,
        half: window / 2,
        period: (window + 1).next_multiple_of(2),
        gain: p.variant.final_gain(),
    };
    let basic = ctx.aggregate(&guide(p.variant, cur), None);
    ctx.aggregate(&guide(p.variant, &basic), Some(&basic))
}

/// Everything one level's passes read.
struct LevelCtx<'a> {
    width: usize,
    height: usize,
    place: Placement,
    noisy: &'a Planes,
    models: [Option<NoiseModel>; 3],
    group: usize,
    window: usize,
    half: usize,
    period: usize,
    gain: f64,
}

/// Per-task scratch, reused across references.
struct Scratch {
    /// Selection keys ([`key`]) of the nearest others so far, ascending, at most `K − 1`.
    best: Vec<u128>,
    /// The group's patch corners (raster index), the reference first.
    group: Vec<usize>,
    /// The noisy patches of one channel, by slot.
    y: Vec<Vec4>,
    /// The basic patches of one channel (pass 2), by slot.
    yb: Vec<Vec4>,
    /// The estimates of one channel, by slot.
    est: Vec<Vec4>,
}

impl Scratch {
    fn new(group: usize) -> Scratch {
        Scratch {
            best: Vec::with_capacity(group),
            group: Vec::with_capacity(group),
            y: vec![[0.0; 4]; group],
            yb: vec![[0.0; 4]; group],
            est: vec![[0.0; 4]; group],
        }
    }
}

/// The selection key of a candidate patch: the bits of its distance above its raster index. Distances are sums
/// of absolute values, so non-negative, and the bits of non-negative floats order like the numbers (a NaN last);
/// integer order is therefore distance order with ties broken by the smaller index (reading G3).
fn key(distance: f64, index: usize) -> u128 {
    (u128::from(distance.to_bits()) << 64) | index as u128
}

/// Keeps the `k` smallest keys seen so far in `best`, ascending.
fn keep_smallest(best: &mut Vec<u128>, k: usize, key: u128) {
    if best.len() < k {
        best.push(key);
    } else if let Some(worst) = best.last_mut().filter(|w| key < **w) {
        *worst = key;
    } else {
        return;
    }
    // One insertion-sort step: move the new last key down to its place.
    let mut pos = best.len() - 1;
    while pos > 0 && best[pos - 1] > key {
        best[pos] = best[pos - 1];
        pos -= 1;
    }
    best[pos] = key;
}

/// The reference corners along one axis of `n` px whose first sample sits at frame position `origin` of a frame
/// of `frame` px: the regular corners by phase (`(origin + x) mod period / 2`), and the extra corner `frame − 2`
/// when it is odd and in the buffer.
fn axis_refs(
    n: usize,
    origin: usize,
    frame: usize,
    period: usize,
) -> (Vec<Vec<usize>>, Option<usize>) {
    let mut regular = vec![Vec::new(); period / 2];
    if n >= 2 {
        for x in 0..=n - 2 {
            let f = origin + x;
            if f.is_multiple_of(2) {
                regular[(f % period) / 2].push(x);
            }
        }
    }
    let extra = frame
        .checked_sub(2)
        .filter(|f| f % 2 == 1 && *f >= origin && *f - origin + 2 <= n)
        .map(|f| f - origin);
    (regular, extra)
}

impl LevelCtx<'_> {
    /// The passes of reading BC-f, in order: (reference columns, reference rows), buffer coordinates.
    fn passes(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let (xr, xe) = axis_refs(
            self.width,
            self.place.x0,
            self.place.frame_width,
            self.period,
        );
        let (yr, ye) = axis_refs(
            self.height,
            self.place.y0,
            self.place.frame_height,
            self.period,
        );
        let mut out = Vec::new();
        for ys in &yr {
            for xs in &xr {
                out.push((xs.clone(), ys.clone()));
            }
        }
        if let Some(x) = xe {
            out.extend(yr.iter().map(|ys| (vec![x], ys.clone())));
        }
        if let Some(y) = ye {
            out.extend(xr.iter().map(|xs| (xs.clone(), vec![y])));
        }
        if let (Some(x), Some(y)) = (xe, ye) {
            out.push((vec![x], vec![y]));
        }
        out.retain(|(xs, ys)| !xs.is_empty() && !ys.is_empty());
        out
    }

    /// One full pass over the level: pass 1 when `basic` is `None`, else pass 2. Returns the aggregated planes;
    /// channels without a model are the noisy planes.
    fn aggregate(&self, guide: &[f64], basic: Option<&Planes>) -> Planes {
        let mut acc = vec![[0.0f64; 4]; self.width * self.height];
        for (xs, ys) in self.passes() {
            self.run_pass(&mut acc, &xs, &ys, guide, basic);
        }
        std::array::from_fn(|c| {
            if self.models[c].is_none() {
                return self.noisy[c].clone();
            }
            // Every pixel lies in at least one reference patch (BC-d), so the count is at least 1.
            acc.iter().map(|a| a[c] / a[3]).collect()
        })
    }

    /// The references `xs × ys` of one pass. Rows of references are `≥ S + 1` px apart, so each owns the
    /// accumulator rows from its first footprint row to the next one's, and the rows run in parallel.
    fn run_pass(
        &self,
        acc: &mut [[f64; 4]],
        xs: &[usize],
        ys: &[usize],
        guide: &[f64],
        basic: Option<&Planes>,
    ) {
        let w = self.width;
        let mut chunks = Vec::with_capacity(ys.len());
        let mut rest = acc;
        let mut row0 = 0;
        for (n, &ry) in ys.iter().enumerate() {
            let end = ys.get(n + 1).map_or(self.height, |&next| next - self.half);
            let (chunk, tail) = std::mem::take(&mut rest).split_at_mut((end - row0) * w);
            chunks.push((ry, row0, chunk));
            rest = tail;
            row0 = end;
        }
        chunks.into_par_iter().for_each_init(
            || Scratch::new(self.group),
            |s, (ry, row0, chunk)| {
                for &rx in xs {
                    self.reference(s, rx, ry, guide, basic, chunk, row0);
                }
            },
        );
    }

    /// Searches, filters and aggregates the group of the reference patch at `(rx, ry)`. `chunk` holds the
    /// accumulator rows from `row0`.
    #[allow(clippy::too_many_arguments)] // the natural argument list of one reference
    fn reference(
        &self,
        s: &mut Scratch,
        rx: usize,
        ry: usize,
        guide: &[f64],
        basic: Option<&Planes>,
        chunk: &mut [[f64; 4]],
        row0: usize,
    ) {
        let w = self.width;
        let offs = [0, 1, w, w + 1];
        let r = ry * w + rx;
        let g0 = offs.map(|o| guide[r + o]);
        let (x_lo, x_hi) = (
            rx.saturating_sub(self.half),
            (rx + self.window - self.half).min(w - 1),
        );
        let (y_lo, y_hi) = (
            ry.saturating_sub(self.half),
            (ry + self.window - self.half).min(self.height - 1),
        );
        s.best.clear();
        for cy in y_lo..y_hi {
            for i in cy * w + x_lo..cy * w + x_hi {
                if i != r {
                    let d = (guide[i] - g0[0]).abs()
                        + (guide[i + 1] - g0[1]).abs()
                        + (guide[i + w] - g0[2]).abs()
                        + (guide[i + w + 1] - g0[3]).abs();
                    keep_smallest(&mut s.best, self.group - 1, key(d, i));
                }
            }
        }
        s.group.clear();
        s.group.push(r);
        // The low 64 bits of a key are the index.
        s.group
            .extend(s.best.iter().map(|&k| (k & u128::from(u64::MAX)) as usize));

        let k = s.group.len();
        let base = row0 * w;
        for (c, model) in self.models.iter().enumerate() {
            let Some(model) = model else { continue };
            gather(&self.noisy[c], &s.group, &offs, &mut s.y[..k]);
            match basic {
                None => filter_pass1(&s.y[..k], model, &mut s.est[..k]),
                Some(b) => {
                    gather(&b[c], &s.group, &offs, &mut s.yb[..k]);
                    filter_pass2(&s.y[..k], &s.yb[..k], model, self.gain, &mut s.est[..k])
                }
            };
            for (&i, e) in s.group.iter().zip(&s.est[..k]) {
                for (&o, &v) in offs.iter().zip(e) {
                    chunk[i - base + o][c] += v;
                }
            }
        }
        for &i in &s.group {
            for o in offs {
                chunk[i - base + o][3] += 1.0;
            }
        }
    }
}

/// The 2 × 2 patches of `plane` at the corners `group` into `out`.
fn gather(plane: &[f64], group: &[usize], offs: &[usize; 4], out: &mut [Vec4]) {
    for (o, &i) in out.iter_mut().zip(group) {
        *o = offs.map(|d| plane[i + d]);
    }
}

/// The mean of a group of patches, summed in slot order.
fn mean4(y: &[Vec4]) -> Vec4 {
    let mut m = [0.0; 4];
    for p in y {
        for (a, v) in m.iter_mut().zip(p) {
            *a += v;
        }
    }
    let inv = 1.0 / y.len() as f64;
    m.map(|v| v * inv)
}

/// The upper triangle of a 4 × 4 matrix, row by row.
const UPPER: [(usize, usize); 10] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
];

/// `Σ (p − mean)(p − mean)ᵀ / (len − 1)` (`len ≥ 2`), summed in slot order; the lower triangle mirrors the
/// upper one.
fn cov4(y: &[Vec4], mean: &Vec4) -> Mat4 {
    let mut u = [0.0f64; 10];
    for p in y {
        let d: Vec4 = std::array::from_fn(|i| p[i] - mean[i]);
        for (e, &(i, j)) in u.iter_mut().zip(&UPPER) {
            *e += d[i] * d[j];
        }
    }
    let inv = 1.0 / (y.len() - 1) as f64;
    let mut c = [[0.0; 4]; 4];
    for (&v, &(i, j)) in u.iter().zip(&UPPER) {
        c[i][j] = v * inv;
        c[j][i] = v * inv;
    }
    c
}

/// `a · b`.
fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j]
        })
    })
}

/// `a · v`.
fn mat_vec(a: &Mat4, v: &Vec4) -> Vec4 {
    a.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3])
}

/// The inverse of a symmetric positive definite 4 × 4 matrix through `A = L D Lᵀ` (`L` unit lower triangular):
/// `A⁻¹ = L⁻ᵀ D⁻¹ L⁻¹`. `None` when a pivot `D[k]` is `≤ 2⁻²³` or not a number.
#[allow(clippy::needless_range_loop)] // the index loops are the LDLᵀ formulas
fn spd_inverse(a: &Mat4) -> Option<Mat4> {
    let mut l = [[0.0f64; 4]; 4];
    let mut d = [0.0f64; 4];
    for j in 0..4 {
        let mut dj = a[j][j];
        for k in 0..j {
            dj -= l[j][k] * l[j][k] * d[k];
        }
        if dj.is_nan() || dj <= PIVOT_MIN {
            return None;
        }
        d[j] = dj;
        l[j][j] = 1.0;
        for i in j + 1..4 {
            let mut v = a[i][j];
            for k in 0..j {
                v -= l[i][k] * l[j][k] * d[k];
            }
            l[i][j] = v / dj;
        }
    }
    // X = L⁻¹, unit lower triangular.
    let mut x = [[0.0f64; 4]; 4];
    for i in 0..4 {
        x[i][i] = 1.0;
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s += l[i][k] * x[k][j];
            }
            x[i][j] = -s;
        }
    }
    // A⁻¹[i][j] = Σ_{k ≥ max(i, j)} X[k][i] X[k][j] / D[k].
    let mut inv = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in i..4 {
            let mut s = 0.0;
            for k in j..4 {
                s += x[k][i] * x[k][j] / d[k];
            }
            inv[i][j] = s;
            inv[j][i] = s;
        }
    }
    Some(inv)
}

/// Pass 1 on one channel of one group: `y` the noisy patches (the reference first), estimates into `out`.
/// Returns `false` when the group is left unfiltered (`out = y`).
fn filter_pass1(y: &[Vec4], model: &NoiseModel, out: &mut [Vec4]) -> bool {
    if y.len() < 2 {
        out.copy_from_slice(y);
        return false;
    }
    let mean = mean4(y);
    let mut cy = cov4(y, &mean);
    let cn = model.cn(&mean, 1.0);
    for (i, row) in cy.iter_mut().enumerate() {
        row[i] = row[i].max(cn[i][i]);
    }
    let Some(inv) = spd_inverse(&cy) else {
        out.copy_from_slice(y);
        return false;
    };
    let a = mat_mul(&cn, &inv);
    for (o, p) in out.iter_mut().zip(y) {
        let yc: Vec4 = std::array::from_fn(|i| p[i] - mean[i]);
        let ay = mat_vec(&a, &yc);
        *o = std::array::from_fn(|i| mean[i] + (yc[i] - ay[i]));
    }
    true
}

/// Pass 2 on one channel of one group: `y` the noisy patches, `yb` the basic patches at the same corners.
/// Returns `false` when the group is left unfiltered (`out = y`).
fn filter_pass2(y: &[Vec4], yb: &[Vec4], model: &NoiseModel, gain: f64, out: &mut [Vec4]) -> bool {
    if y.len() < 2 {
        out.copy_from_slice(y);
        return false;
    }
    let mean = mean4(y);
    let cb = cov4(yb, &mean4(yb));
    let cn = model.cn(&mean, gain);
    let m: Mat4 = std::array::from_fn(|i| std::array::from_fn(|j| cb[i][j] + cn[i][j]));
    let Some(inv) = spd_inverse(&m) else {
        out.copy_from_slice(y);
        return false;
    };
    let wm = mat_mul(&cb, &inv);
    for (o, p) in out.iter_mut().zip(y) {
        let yc: Vec4 = std::array::from_fn(|i| p[i] - mean[i]);
        let wy = mat_vec(&wm, &yc);
        *o = std::array::from_fn(|i| mean[i] + wy[i]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth::{self, Rng};

    /// A smooth scene with flat areas: gradients, soft blobs and two constant rectangles, in [0.15, 0.85].
    fn smooth_scene(w: usize, h: usize) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        let blobs = [
            (0.7, 0.25, 0.10, [0.20, 0.10, -0.05]),
            (0.3, 0.75, 0.12, [-0.15, 0.05, 0.20]),
            (0.55, 0.55, 0.06, [0.10, 0.20, 0.10]),
        ];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
                let mut p = [0.35 + 0.25 * fx, 0.40 + 0.20 * fy, 0.30 + 0.15 * (fx + fy)];
                for (cx, cy, s, amp) in blobs {
                    let g = (-((fx - cx).powi(2) + (fy - cy).powi(2)) / (2.0 * s * s)).exp();
                    for c in 0..3 {
                        p[c] += amp[c] * g;
                    }
                }
                if (0.08..0.33).contains(&fx) && (0.08..0.30).contains(&fy) {
                    p = [0.50, 0.45, 0.40];
                }
                if (0.62..0.92).contains(&fx) && (0.60..0.90).contains(&fy) {
                    p = [0.30, 0.35, 0.45];
                }
                img.set_pixel(x, y, p.map(|v| v as f32));
            }
        }
        img
    }

    fn psnr(a: &RgbImage, b: &RgbImage) -> f64 {
        let mse = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.data.len() as f64;
        10.0 * (1.0 / mse).log10()
    }

    fn params(variant: Variant, luma: f64, chroma: f64, radius: usize) -> NlBayesParams {
        NlBayesParams {
            variant,
            luma,
            chroma,
            radius,
            blend: 0.0,
        }
    }

    fn noisy(img: &RgbImage, sigma: f32, seed: u64) -> RgbImage {
        let mut n = img.clone();
        synth::add_noise(&mut n.data, sigma, seed);
        n
    }

    fn naive_inverse(a: &Mat4) -> Mat4 {
        let mut m = *a;
        let mut inv: Mat4 = std::array::from_fn(|i| std::array::from_fn(|j| f64::from(i == j)));
        for col in 0..4 {
            let piv = (col..4)
                .max_by(|&p, &q| m[p][col].abs().total_cmp(&m[q][col].abs()))
                .unwrap();
            m.swap(col, piv);
            inv.swap(col, piv);
            let d = m[col][col];
            for j in 0..4 {
                m[col][j] /= d;
                inv[col][j] /= d;
            }
            for row in 0..4 {
                if row != col {
                    let f = m[row][col];
                    for j in 0..4 {
                        m[row][j] -= f * m[col][j];
                        inv[row][j] -= f * inv[col][j];
                    }
                }
            }
        }
        inv
    }

    fn model(base: Mat4, m_flat: Option<f64>) -> NoiseModel {
        NoiseModel { base, m_flat }
    }

    #[test]
    fn spd_inverse_matches_a_gauss_jordan_inverse() {
        let mut rng = Rng::new(11);
        for _ in 0..50 {
            let b: Mat4 =
                std::array::from_fn(|_| std::array::from_fn(|_| f64::from(rng.range(-1.0, 1.0))));
            // A = Bᵀ B + 0.05 I is symmetric positive definite.
            let a: Mat4 = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    (0..4).map(|k| b[k][i] * b[k][j]).sum::<f64>() + if i == j { 0.05 } else { 0.0 }
                })
            });
            let got = spd_inverse(&a).expect("positive definite");
            let want = naive_inverse(&a);
            for i in 0..4 {
                for j in 0..4 {
                    assert!((got[i][j] - want[i][j]).abs() < 1e-9 * (1.0 + want[i][j].abs()));
                }
            }
        }
        // A pivot of 2^-23 fails, twice that passes, not a number fails.
        let diag = |v: f64| -> Mat4 {
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { v } else { 0.0 }))
        };
        assert!(spd_inverse(&diag(PIVOT_MIN)).is_none());
        assert!(spd_inverse(&diag(2.0 * PIVOT_MIN)).is_some());
        assert!(spd_inverse(&diag(f64::NAN)).is_none());
    }

    #[test]
    fn selection_keeps_the_nearest_by_distance_then_raster_index() {
        let mut rng = Rng::new(9);
        for k in [0usize, 1, 7, 23, 60] {
            // Quantised distances, so that ties are common; indices unique.
            let cands: Vec<(f64, usize)> = (0..48)
                .map(|i| ((rng.next_f32() * 8.0).floor() as f64 * 0.125, 1000 - 7 * i))
                .collect();
            let mut best = Vec::new();
            for &(d, i) in &cands {
                keep_smallest(&mut best, k, key(d, i));
            }
            let got: Vec<usize> = best
                .iter()
                .map(|&b| (b & u128::from(u64::MAX)) as usize)
                .collect();
            let mut sorted = cands.clone();
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let want: Vec<usize> = sorted.iter().take(k).map(|c| c.1).collect();
            assert_eq!(got, want, "k = {k}");
        }
        // Bits order like the numbers for non-negative distances.
        assert!(key(0.0, 5) < key(f64::MIN_POSITIVE, 0));
        assert!(key(1.0, 9) < key(1.0 + f64::EPSILON, 0));
        assert!(key(f64::INFINITY, 9) < key(f64::NAN, 0));
    }

    #[test]
    fn a_singular_group_takes_the_unfiltered_path() {
        // Patches t·(1, 1, 1, 1): Cy is rank 1, and the noise diagonal (1e-9) is below it, so the clamp leaves it
        // singular and the second pivot is 0.
        let y: Vec<Vec4> = (0..8).map(|k| [0.1 * k as f64 + 0.2; 4]).collect();
        let tiny: Mat4 =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1e-9 } else { 0.0 }));
        let mut out = vec![[0.0; 4]; 8];
        assert!(!filter_pass1(&y, &model(tiny, None), &mut out));
        assert_eq!(out, y);
        // Pass 2: Cb is rank 1 and Cn tiny, so Cb + Cn is singular too.
        let mut out = vec![[0.0; 4]; 8];
        assert!(!filter_pass2(&y, &y, &model(tiny, None), 1.0, &mut out));
        assert_eq!(out, y);
        // A group of one is unfiltered.
        let mut out = vec![[0.0; 4]; 1];
        assert!(!filter_pass1(&y[..1], &model(tiny, None), &mut out));
        assert_eq!(out[0], y[0]);
    }

    #[test]
    fn pass_one_shrinks_noise_towards_the_group_mean() {
        // Independent noise of variance 1e-4 per pixel with Cn = 1e-4 I: x̂ moves part of the way to ȳ.
        let mut rng = Rng::new(4);
        let y: Vec<Vec4> = (0..24)
            .map(|_| std::array::from_fn(|_| 0.5 + 0.01 * f64::from(rng.gaussian())))
            .collect();
        let cn: Mat4 =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1e-4 } else { 0.0 }));
        let mut out = vec![[0.0; 4]; 24];
        assert!(filter_pass1(&y, &model(cn, None), &mut out));
        let mean = mean4(&y);
        let spread = |v: &[Vec4]| {
            v.iter()
                .map(|p| (0..4).map(|i| (p[i] - mean[i]).powi(2)).sum::<f64>())
                .sum::<f64>()
        };
        assert!(
            spread(&out) < 0.5 * spread(&y),
            "{} vs {}",
            spread(&out),
            spread(&y)
        );
        // The intensity model of method B: a darker group gets more noise (s > 1 below m_flat).
        let m = model(cn, Some(0.5));
        assert!(m.cn(&[0.25; 4], 1.0)[0][0] > 1e-4);
        assert!(m.cn(&[0.75; 4], 1.0)[0][0] < 1e-4);
        assert_eq!(m.cn(&[0.05; 4], 1.0), m.cn(&[0.2; 4], 1.0)); // clamped at 0.2
        assert_eq!(m.cn(&[0.5; 4], 2.0)[1][1], 2e-4);
    }

    #[test]
    fn passes_cover_every_corner_once_with_disjoint_footprints() {
        for (w, h, radius) in [
            (10usize, 9usize, 0usize),
            (11, 12, 1),
            (17, 17, 2),
            (2, 3, 0),
        ] {
            let planes: Planes = std::array::from_fn(|_| vec![0.0; w * h]);
            let window = radius + 6;
            let ctx = LevelCtx {
                width: w,
                height: h,
                place: Placement::whole(w, h),
                noisy: &planes,
                models: [None; 3],
                group: 8,
                window,
                half: window / 2,
                period: (window + 1).next_multiple_of(2),
                gain: 1.0,
            };
            let corners = |n: usize| {
                let mut v: Vec<usize> = (0..=n - 2).step_by(2).collect();
                if (n - 2) % 2 == 1 {
                    v.push(n - 2);
                }
                v
            };
            let mut want: Vec<(usize, usize)> = corners(h)
                .iter()
                .flat_map(|&y| corners(w).into_iter().map(move |x| (x, y)))
                .collect();
            let mut got = vec![];
            for (xs, ys) in ctx.passes() {
                for pair in xs.windows(2).chain(ys.windows(2)) {
                    assert!(pair[1] - pair[0] > window, "{w}x{h}: {pair:?}");
                }
                got.extend(ys.iter().flat_map(|&y| xs.iter().map(move |&x| (x, y))));
            }
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got, want, "{w}x{h}");
        }
    }

    #[test]
    fn methods_b_and_c_raise_psnr_by_at_least_3_db() {
        // 515 x 521: pyramid depth 2 for B and 1 for C, odd sizes at every level.
        let clean = smooth_scene(515, 521);
        let noisy = noisy(&clean, 10.0 / 255.0, 21);
        let before = psnr(&noisy, &clean);
        for radius in 0..=2 {
            for (variant, luma, chroma) in [(Variant::B, 50.0, 0.0), (Variant::C, 80.0, 50.0)] {
                let out = denoise(&noisy, &params(variant, luma, chroma, radius));
                let after = psnr(&out, &clean);
                println!(
                    "{variant:?} R = {radius}: PSNR {before:.2} -> {after:.2} dB (+{:.2})",
                    after - before
                );
                assert!(
                    after - before >= 3.0,
                    "{variant:?} R = {radius}: {before:.2} -> {after:.2}"
                );
            }
        }
    }

    #[test]
    fn denoise_equals_measure_then_denoise_with() {
        let img = noisy(&smooth_scene(300, 280), 0.03, 2);
        for (variant, luma, chroma) in [(Variant::B, 50.0, 0.0), (Variant::C, 80.0, 50.0)] {
            let p = params(variant, luma, chroma, 1);
            let m = measure(&img, variant);
            assert_eq!(m.profiles.len(), m.depth + 1);
            assert!(!m.regions.is_empty());
            assert_eq!(denoise(&img, &p), denoise_with(&img, &p, &m));
        }
    }

    #[test]
    fn no_flat_region_returns_the_input() {
        // A frame smaller than one detector cell (13 px) has no region.
        let img = noisy(&smooth_scene(12, 40), 0.03, 6);
        for (variant, luma, chroma) in [(Variant::B, 50.0, 0.0), (Variant::C, 80.0, 50.0)] {
            let p = params(variant, luma, chroma, 1);
            assert!(measure(&img, variant).regions.is_empty());
            assert_eq!(denoise(&img, &p), img);
        }
        // Forced: measurements without regions.
        let img = noisy(&smooth_scene(200, 150), 0.03, 6);
        let mut m = measure(&img, Variant::C);
        m.regions.clear();
        assert_eq!(
            denoise_with(&img, &params(Variant::C, 80.0, 50.0, 1), &m),
            img
        );
        // No usable profile anywhere: every channel passes through (exact for RGB planes).
        let mut m = measure(&img, Variant::B);
        m.profiles.iter_mut().for_each(|p| *p = [None; 3]);
        assert_eq!(
            denoise_with(&img, &params(Variant::B, 50.0, 0.0, 1), &m),
            img
        );
    }

    #[test]
    fn zero_strength_returns_the_input_and_full_blend_too() {
        let img = noisy(&smooth_scene(200, 150), 0.03, 8);
        assert_eq!(denoise(&img, &params(Variant::B, 0.0, 70.0, 1)), img);
        assert_eq!(denoise(&img, &params(Variant::C, 0.0, 0.0, 1)), img);
        let mut p = params(Variant::C, 80.0, 50.0, 1);
        p.blend = 100.0;
        assert_eq!(denoise(&img, &p).data, img.data);
        // C = 0 in method C leaves the chroma of the input: only Y changes.
        let out = denoise(&img, &params(Variant::C, 80.0, 0.0, 1));
        let ([_, cb0, cr0], [_, cb1, cr1]) = (ycbcr::to_ycbcr(&img), ycbcr::to_ycbcr(&out));
        let worst = cb0
            .iter()
            .chain(&cr0)
            .zip(cb1.iter().chain(&cr1))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn output_is_bit_identical_for_one_and_four_threads() {
        // Odd sizes: extra column, row and corner passes; 523 x 389 has one pyramid level for B.
        let img = noisy(&synth::natural(523, 389, 13), 0.03, 3);
        for (variant, luma, chroma) in [(Variant::B, 50.0, 0.0), (Variant::C, 80.0, 50.0)] {
            let p = params(variant, luma, chroma, 1);
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| denoise(&img, &p))
            };
            let (one, four) = (run(1), run(4));
            assert!(one != img);
            let bits = |i: &RgbImage| i.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&one), bits(&four), "{variant:?}");
        }
    }

    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn timing_per_megapixel() {
        let (w, h) = (2048usize, 1365usize);
        let img = noisy(&synth::natural(w, h, 1), 10.0 / 255.0, 2);
        let mp = (w * h) as f64 / 1e6;
        for (variant, luma, chroma) in [(Variant::B, 50.0, 0.0), (Variant::C, 80.0, 50.0)] {
            let p = params(variant, luma, chroma, 1);
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let t = std::time::Instant::now();
            let a = one.install(|| denoise(&img, &p));
            let single = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let b = denoise(&img, &p);
            let wall = t.elapsed().as_secs_f64();
            assert_eq!(a, b);
            println!(
                "{variant:?} R = 1, {w}x{h}: {:.2} CPU-s/MP (one thread), {:.2} s/MP wall on {} threads",
                single / mp,
                wall / mp,
                rayon::current_num_threads()
            );
        }
    }
}
