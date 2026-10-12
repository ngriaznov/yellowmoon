//! The raw stage group R1 to R9: from the decoded mosaic to scene-linear camera RGB balanced with
//! the initial gains `g0` (`docs/05-raw-processing.md`).
//!
//! The maths lives in `ym_gpu::cpu` (one module per stage, `cpu::chain::Chain` for the whole
//! classical chain) and `ym_noise::sigma_map`; this module decides *which* stages run on *which*
//! pixels:
//!
//! * [`FrameData`]: the photo's area copied out of the readout, plus the quantisation steps.
//! * [`RawPlan`]: everything fixed by the file and the RAW-tab switches (levels at the area's
//!   parity, the vignetting profile, the lens maps and their auto crop).
//! * **Full resolution** ([`FullChain`]): `cpu::chain::Chain` with the real settings, run whole or
//!   tile by tile (`run_tile`). Demosaic is the classical Menon; the neural switch falls back to it
//!   (see [`crate::pipeline::NeuralStatus`]).
//! * **Reduced resolution** ([`reduced_bayer`]): the fit view and the thumbnails. R1, R4 and R5 run
//!   on the full mosaic, the photosites are then binned `k x k` per colour (a box filter is the
//!   demosaic at 1/k scale: every output pixel has its own red, green and blue), the vignetting
//!   gain is applied per output pixel, the light bilateral denoise runs with the noise left after
//!   binning, and the lens maps resample the result. R7 (colour guided filter, detail) and R8
//!   (sharpening) are skipped below full resolution: both are defined in full-resolution pixels
//!   (`docs/05`: radii and PSF in pixels), and binning has already averaged the noise they cure.
//! * **Camera-processed files** (Lossless M and S, [`FrameData::reduced`]): no mosaic. R1 scales all
//!   three channels by one range, R4 balances, the light denoise, R7 and R8 follow; there is no R3,
//!   R5, R6 or R9 (the camera has already applied its own shading and lens corrections).
//!
//! All parallel loops here split by output row and never reduce across rows, so the result does not
//! depend on the number of threads.

use std::sync::Arc;

use half::f16;
use rayon::prelude::*;
use ym_gpu::cpu::chain::{Chain, Prepared, RawFrame};
use ym_gpu::cpu::demosaic::DEFAULT_METHOD;
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::detail::{self, NoiseReduction};
use ym_gpu::cpu::highlights::{self, OffsetAccumulator};
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::sharpen::{self, SharpenParams};
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{Cfa, CfaColor as GpuColor, LUMA, Mosaic, NoiseSigma, Placement, Rect, RgbImage};
use ym_gpu::queue::CancelToken;
use ym_noise::model::NoiseModel;
use ym_noise::sigma_map::{HalfPlanes, QstepMap, SigmaParams, sigma_quad};
use ym_raw::decode::RawImage;
use ym_raw::layout::{CfaColor, Layout};

use super::RenderError;
use crate::auto::{ImageArea, image_area};

/// Rows of a half-resolution band of the sigma map (one band is 2 x this many mosaic rows).
const SIGMA_BAND_HALF_ROWS: usize = 32;
/// Context rows a band of the sigma map needs on each side (the 5 x 5 box of `sigma_map`).
const SIGMA_CONTEXT_HALF_ROWS: usize = 2;
/// The smallest side a reduced image may have, in pixels. Below it the scale is halved.
const MIN_REDUCED_SIDE_PX: usize = 16;

/// The pixels a photo's raw stages work on.
#[derive(Debug)]
pub struct FrameData {
    /// The decoded photo (kept alive so that a frame that is the whole readout needs no copy).
    raw: Arc<RawImage>,
    /// The area copied out of the readout when it is not the whole readout.
    copy: Option<Vec<u16>>,
    /// The area, in readout coordinates.
    pub area: ImageArea,
    /// Width of the frame in pixels.
    pub width: usize,
    /// Height of the frame in pixels.
    pub height: usize,
    /// The colour filter array seen from the frame's first photosite (`RGGB` for camera-RGB files).
    pub cfa: Cfa,
    /// Quantisation step per photosite in raw units, `width * height` (cRAW).
    pub qstep: Option<Vec<f32>>,
    /// True for camera-processed files: `data()` holds interleaved RGB, three samples per pixel.
    pub reduced: bool,
}

impl FrameData {
    /// Cuts the photo's area out of the readout (`docs/04`: the camera's crop, else the active
    /// area, else the whole readout) and converts the quantisation steps to `f32`.
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] for an area that does not fit the data.
    pub fn new(raw: &Arc<RawImage>) -> Result<Self, RenderError> {
        let area = image_area(raw);
        let reduced = raw.is_reduced();
        let channels = raw.channels();
        let stride = raw.data.width;
        let (full_w, full_h) = (raw.pixel_width(), raw.pixel_height());
        if area.x + area.width > full_w || area.y + area.height > full_h || area.width == 0 {
            return Err(RenderError::Image(
                "The photo's image area does not fit the data in the file.".to_owned(),
            ));
        }
        let whole = area.x == 0 && area.y == 0 && area.width == full_w && area.height == full_h;
        let copy = if whole {
            None
        } else {
            let mut v = Vec::with_capacity(area.width * area.height * channels);
            for y in area.y..area.y + area.height {
                let s = y * stride + area.x * channels;
                v.extend_from_slice(&raw.data.data[s..s + area.width * channels]);
            }
            Some(v)
        };
        let cfa = match raw.layout {
            Layout::Bayer { pattern, .. } => {
                let c = |dx: usize, dy: usize| match pattern.color_at(area.x + dx, area.y + dy) {
                    CfaColor::Red => GpuColor::Red,
                    CfaColor::Green => GpuColor::Green,
                    CfaColor::Blue => GpuColor::Blue,
                };
                Cfa::new([[c(0, 0), c(1, 0)], [c(0, 1), c(1, 1)]])
            }
            Layout::Ycbcr { .. } => Cfa::RGGB,
        };
        let qstep = raw.qstep.as_ref().map(|q| {
            let mut out = vec![0.0f32; area.width * area.height];
            out.par_chunks_mut(area.width.max(1))
                .enumerate()
                .for_each(|(y, row)| {
                    let s = (area.y + y) * q.width + area.x;
                    for (d, v) in row.iter_mut().zip(&q.data[s..s + area.width]) {
                        *d = f16::to_f32(*v);
                    }
                });
            out
        });
        Ok(Self {
            raw: Arc::clone(raw),
            copy,
            area,
            width: area.width,
            height: area.height,
            cfa,
            qstep,
            reduced,
        })
    }

    /// The samples of the frame, row-major, without padding.
    #[must_use]
    pub fn data(&self) -> &[u16] {
        self.copy.as_deref().unwrap_or(&self.raw.data.data)
    }

    /// A [`RawFrame`] for `cpu::chain` (mosaics only).
    #[must_use]
    pub fn as_raw_frame(&self) -> RawFrame<'_> {
        RawFrame {
            width: self.width,
            height: self.height,
            cfa: self.cfa,
            data: self.data(),
            qstep: self.qstep.as_deref(),
        }
    }

    /// Bytes this frame holds beyond the decoded image (for the memory ledger).
    #[must_use]
    pub fn extra_bytes(&self) -> usize {
        self.copy.as_ref().map_or(0, |c| c.len() * 2)
            + self.qstep.as_ref().map_or(0, |q| q.len() * 4)
    }
}

/// The settings of the raw group that come from the photo's settings and estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawParams {
    /// The initial gains `g0` (red, green = 1, blue).
    pub g0: [f32; 3],
    /// `decode.neural_demosaic`. Only part of the key today: the neural path is not available, so
    /// the classical demosaic runs either way, but the switch invalidates the group (`docs/01`
    /// table: `decode.*`) so that a render made after the network arrives is never served stale.
    pub neural_demosaic: bool,
    /// `decode.highlight_recovery`.
    pub highlight_recovery: bool,
    /// `decode.lens_corrections`.
    pub lens_corrections: bool,
    /// `decode.chromatic_aberration`.
    pub chromatic_aberration: bool,
    /// The resolved noise reduction (R6 and R7).
    pub noise_reduction: NoiseReduction,
    /// The resolved capture sharpening (R8).
    pub sharpen: SharpenParams,
    /// ISO of the shot (sets `L_auto`).
    pub iso: f32,
    /// The noise model (the sigma map).
    pub noise: NoiseModel,
}

/// A cache key for [`RawParams`]: the same numbers, bit for bit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawParamsKey(Vec<u64>);

impl RawParams {
    /// The key that says two sets of parameters give the same pixels.
    #[must_use]
    pub fn key(&self) -> RawParamsKey {
        let f = |v: f64| v.to_bits();
        let g = |v: f32| u64::from(v.to_bits());
        let mut k = vec![
            g(self.g0[0]),
            g(self.g0[1]),
            g(self.g0[2]),
            u64::from(self.neural_demosaic),
            u64::from(self.highlight_recovery),
            u64::from(self.lens_corrections),
            u64::from(self.chromatic_aberration),
            g(self.noise_reduction.luminance),
            g(self.noise_reduction.color),
            g(self.noise_reduction.detail),
            g(self.sharpen.amount),
            g(self.sharpen.radius_px),
            g(self.iso),
        ];
        k.extend(self.noise_key_parts().map(f));
        RawParamsKey(k)
    }

    /// The numbers of the noise model that the sigma map reads.
    fn noise_key_parts(&self) -> [f64; 8] {
        let n = &self.noise;
        [
            n.a,
            n.b,
            n.delta[0],
            n.delta[1],
            n.delta[2],
            n.delta[3],
            n.sigma_row,
            n.q,
        ]
    }

    /// The key of the sigma map: the noise model and the gains (not the strengths).
    #[must_use]
    pub fn sigma_key(&self) -> Vec<u64> {
        let mut k = vec![
            u64::from(self.g0[0].to_bits()),
            u64::from(self.g0[2].to_bits()),
        ];
        k.extend(self.noise_key_parts().map(f64::to_bits));
        k
    }
}

/// Everything about a photo's raw stages that the file and the RAW-tab switches fix.
#[derive(Debug, Clone)]
pub struct RawPlan {
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Camera-processed file (no mosaic).
    pub reduced: bool,
    /// R1 levels indexed by the frame's own parity (mosaics).
    pub levels: Levels,
    /// R1 for camera-RGB files: black per channel and the one shared range.
    pub rgb_black: [f32; 3],
    /// `1 / range` shared by the three channels of a camera-RGB file.
    pub rgb_inv_range: f32,
    /// R3 profile, when the file has the tag and the switch is on.
    pub vignette: Option<VignetteProfile>,
    /// R9 geometry with its auto crop, when it moves any pixel.
    pub lens: Option<LensGeometry>,
    /// Size of the image after R9 at full resolution.
    pub out_w: usize,
    /// Height after R9 at full resolution.
    pub out_h: usize,
    /// Whether any photosite of the frame is clipped (the clip rule of R1). When none is, R5 changes
    /// nothing and is skipped.
    pub has_clipped: bool,
    /// The usable lens tags (an empty vector is an absent tag).
    pub tags: LensTags,
    /// The RAW-tab lens switches the plan was built with.
    pub switches: LensSwitches,
    /// Plain-words notes about lens tables that could not be used.
    pub notes: Vec<String>,
}

fn table(t: &Option<ym_raw::metadata::LensTable>) -> Option<&[i16]> {
    t.as_ref().map(|t| t.raw.as_slice())
}

/// Whether R1 would flag any photosite of the frame as clipped (`docs/05` R1: raw at or above
/// `white - 4 qstep`, or normalised at or above 0.995 without a `qstep` plane): the same
/// expressions as `cpu::normalize`, so the answer agrees with its mask.
fn frame_has_clipped(frame: &FrameData, levels: &Levels) -> bool {
    let (w, h) = (frame.width, frame.height);
    let inv: [f32; 4] = std::array::from_fn(|i| 1.0 / (levels.white[i] - levels.black[i]));
    let data = frame.data();
    let q = frame.qstep.as_deref();
    (0..h).into_par_iter().any(|y| {
        let row = &data[y * w..(y + 1) * w];
        let qrow = q.map(|q| &q[y * w..(y + 1) * w]);
        row.iter().enumerate().any(|(x, &raw)| {
            let p = Cfa::position(x, y);
            let v = f32::from(raw);
            match qrow {
                Some(q) => v >= levels.white[p] - normalize::QSTEP_MARGIN * q[x],
                None => (v - levels.black[p]) * inv[p] >= normalize::CLIP_FRACTION,
            }
        })
    })
}

/// Rows of a band of [`reconstruct_banded`] (even).
const R5_BAND_ROWS: usize = 128;
/// Rows of context a band of [`reconstruct_banded`] reads on each side (even; the algorithm needs
/// four: one superpixel for the 3 x 3 neighbourhood and one for its dilation).
const R5_CONTEXT_ROWS: usize = 8;

/// R5 over a whole mosaic, touching only the bands that hold a clipped photosite: the chroma
/// offsets are accumulated over those bands (in band order, so the sums do not depend on the
/// thread count), then applied to each of them. Bands with no clipped photosite within
/// [`R5_CONTEXT_ROWS`] have no clipped superpixel and no reference superpixel next to one, so
/// leaving them out gives the whole-frame result of `highlights::reconstruct`.
fn reconstruct_banded(
    m: &mut Mosaic,
    mask: &ym_gpu::cpu::ClipMask,
) -> Result<(), ym_gpu::cpu::CpuError> {
    let (w, h) = (m.width, m.height);
    let row_clipped: Vec<bool> = (0..h)
        .into_par_iter()
        .map(|y| mask.data[y * w..(y + 1) * w].iter().any(|&v| v != 0))
        .collect();
    let bands: Vec<(usize, usize)> = (0..h)
        .step_by(R5_BAND_ROWS)
        .map(|y0| (y0, (y0 + R5_BAND_ROWS).min(h)))
        .collect();
    let reach = 2 * R5_CONTEXT_ROWS;
    let active: Vec<(usize, usize)> = bands
        .into_iter()
        .filter(|&(y0, y1)| {
            row_clipped[y0.saturating_sub(reach)..(y1 + reach).min(h)]
                .iter()
                .any(|&c| c)
        })
        .collect();
    if active.is_empty() {
        return Ok(());
    }
    // The rectangle of a band with its context, even-aligned at the top.
    let sub_rect = |&(y0, y1): &(usize, usize)| {
        let sy = y0.saturating_sub(R5_CONTEXT_ROWS);
        let ey = (y1 + R5_CONTEXT_ROWS).min(h);
        Rect::new(0, sy, w, ey - sy)
    };
    let source: &Mosaic = m;
    let accs: Vec<Result<OffsetAccumulator, ym_gpu::cpu::CpuError>> = active
        .par_iter()
        .map(|band| {
            let r = sub_rect(band);
            let sub = source.extract(r);
            let sub_mask = mask.extract(r);
            let core = Rect::new(0, band.0 - r.y, w, band.1 - band.0);
            highlights::accumulate_offsets(&sub, &sub_mask, core)
        })
        .collect();
    let mut total = OffsetAccumulator::default();
    for a in accs {
        total.merge(&a?);
    }
    let offsets = total.finish();
    let source: &Mosaic = m;
    let done: Vec<Result<Mosaic, ym_gpu::cpu::CpuError>> = active
        .par_iter()
        .map(|band| {
            let r = sub_rect(band);
            let mut sub = source.extract(r);
            let sub_mask = mask.extract(r);
            highlights::apply_offsets(&mut sub, &sub_mask, &offsets)?;
            Ok(sub)
        })
        .collect();
    for (band, sub) in active.iter().zip(done) {
        let sub = sub?;
        let r = sub_rect(band);
        let core = Rect::new(0, band.0 - r.y, w, band.1 - band.0);
        m.paste(&sub, core, 0, band.0);
    }
    Ok(())
}

impl RawPlan {
    /// Builds the plan for a photo.
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] for levels without a range.
    pub fn new(raw: &RawImage, frame: &FrameData, params: &RawParams) -> Result<Self, RenderError> {
        let (w, h) = (frame.width, frame.height);
        let area = frame.area;
        let bad = |what: &str| {
            RenderError::Image(format!("The levels in the file are unusable: {what}."))
        };
        // Levels.
        let mut levels = Levels::uniform(0.0, 1.0);
        let mut rgb_black = [0.0f32; 3];
        let mut rgb_inv_range = 1.0f32;
        if frame.reduced {
            // Black and white are R, G, G, B for the camera-RGB layout. One range for all three
            // channels: the green one (`auto::Levels::new` does the same for the estimates).
            let range: Vec<f32> = (0..4).map(|p| raw.white[p] - raw.black[p]).collect();
            if range.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
                return Err(bad("white is not above black"));
            }
            let shared = range.iter().copied().fold(0.0, f32::max);
            rgb_black = [raw.black[0], raw.black[1], raw.black[3]];
            rgb_inv_range = 1.0 / shared;
        } else {
            // The file's levels are indexed by the readout's parity; the frame starts at the area.
            for p in 0..4usize {
                let src = (((p >> 1) + area.y) & 1) * 2 + (((p & 1) + area.x) & 1);
                levels.black[p] = raw.black[src];
                levels.white[p] = raw.white[src];
            }
            levels.validate().map_err(|e| bad(&e.to_string()))?;
        }
        // Lens tags, each on its own so that one bad table does not drop the others.
        let mut notes = Vec::new();
        let tags = if frame.reduced {
            LensTags::default()
        } else {
            let lens = raw.lens.as_ref().or(raw.meta.lens_corrections.as_ref());
            let mut tags = LensTags::default();
            if let Some(l) = lens {
                match LensTags::from_stored(table(&l.vignetting), None, None) {
                    Ok(t) => tags.vignetting = t.vignetting,
                    Err(e) => notes.push(format!("Vignetting table ignored: {e}.")),
                }
                match LensTags::from_stored(None, table(&l.chromatic_aberration), None) {
                    Ok(t) => tags.ca = t.ca,
                    Err(e) => notes.push(format!("Chromatic aberration table ignored: {e}.")),
                }
                match LensTags::from_stored(None, None, table(&l.distortion)) {
                    Ok(t) => tags.distortion = t.distortion,
                    Err(e) => notes.push(format!("Distortion table ignored: {e}.")),
                }
            }
            tags
        };
        let switches = LensSwitches {
            lens_corrections: params.lens_corrections,
            chromatic_aberration: params.chromatic_aberration,
        };
        let vignette = VignetteProfile::from_tags(&tags, switches)
            .map_err(|e| RenderError::Image(format!("The vignetting table is unusable: {e}.")))?;
        let geometry = LensGeometry::new(w, h, &tags, switches)
            .map_err(|e| RenderError::Image(format!("The lens tables are unusable: {e}.")))?;
        let (lens, out_w, out_h) = if geometry.is_identity() {
            (None, w, h)
        } else {
            let g = geometry.with_auto_crop();
            let (ow, oh) = g.output_size();
            (Some(g), ow, oh)
        };
        let has_clipped = !frame.reduced && frame_has_clipped(frame, &levels);
        Ok(Self {
            width: w,
            height: h,
            reduced: frame.reduced,
            has_clipped,
            levels,
            rgb_black,
            rgb_inv_range,
            vignette,
            lens,
            out_w,
            out_h,
            tags,
            switches,
            notes,
        })
    }

    /// The scale actually used for a requested reduction `k`: halved until the image is at least
    /// 16 photosites on each side.
    #[must_use]
    pub fn effective_scale(&self, k: usize) -> usize {
        let mut k = k.max(1);
        while k > 1
            && (self.out_w / k < MIN_REDUCED_SIDE_PX || self.out_h / k < MIN_REDUCED_SIDE_PX)
        {
            k /= 2;
        }
        k
    }

    /// The lens geometry for a frame binned by `k`, with the full-resolution crop scaled down.
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] when the lens tables cannot make a map for the smaller frame.
    pub fn lens_at(&self, k: usize) -> Result<Option<LensGeometry>, RenderError> {
        let Some(full) = &self.lens else {
            return Ok(None);
        };
        if k == 1 {
            return Ok(Some(full.clone()));
        }
        let (w, h) = (self.width / k, self.height / k);
        let geometry = LensGeometry::new(w, h, &self.tags, self.switches)
            .map_err(|e| RenderError::Image(format!("The lens tables are unusable: {e}.")))?;
        let c = full.crop();
        let kk = k as f64;
        let x = ((c.x as f64 / kk).round() as usize).min(w.saturating_sub(1));
        let y = ((c.y as f64 / kk).round() as usize).min(h.saturating_sub(1));
        let cw = ((c.w as f64 / kk).round() as usize).clamp(1, w - x);
        let ch = ((c.h as f64 / kk).round() as usize).clamp(1, h - y);
        Ok(Some(geometry.with_crop(Rect::new(x, y, cw, ch))))
    }

    /// The size of the image after R9 for a frame binned by `k`.
    #[must_use]
    pub fn out_size_at(&self, k: usize) -> (usize, usize) {
        match self.lens_at(k) {
            Ok(Some(g)) => g.output_size(),
            _ => (self.width / k.max(1), self.height / k.max(1)),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Sigma map
// ---------------------------------------------------------------------------------------------

/// The noise sigma of the balanced mosaic at half resolution (`docs/06` sigma map: the RMS of the
/// four CFA positions of each 2 x 2 cell, from `ym_noise::sigma_map::sigma_quad`), computed on the
/// data **before** the vignetting gain. The gain multiplies signal and noise alike, so the sigma of
/// the corrected data is this times the gain ([`gain_plane`]).
#[derive(Debug, Clone)]
pub struct SigmaPlane {
    /// Width in half-resolution cells.
    pub width: usize,
    /// Height in half-resolution cells.
    pub height: usize,
    /// `width * height` sigmas.
    pub data: Vec<f32>,
}

impl SigmaPlane {
    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// Computes the sigma plane of a mosaic frame (see [`SigmaPlane`]).
///
/// # Errors
///
/// [`RenderError::Image`] when the frame is too small or the maths refuses a size, and
/// [`RenderError::Cancelled`] when `cancel` fires.
pub fn sigma_plane(
    frame: &FrameData,
    plan: &RawPlan,
    params: &RawParams,
    cancel: &CancelToken,
) -> Result<SigmaPlane, RenderError> {
    let (hw, hh) = (frame.width / 2, frame.height / 2);
    if hw == 0 || hh == 0 {
        return Err(RenderError::Image(
            "The photo is too small to render.".to_owned(),
        ));
    }
    // R1 and R4 (no vignetting): the signal the sensor really recorded, balanced.
    let (mut m, _mask) = normalize::normalize(
        frame.data(),
        frame.width,
        frame.height,
        frame.cfa,
        &plan.levels,
        None,
    )
    .map_err(RenderError::from_cpu)?;
    ym_gpu::cpu::gains::apply_gains(&mut m, params.g0);
    cancel.check().map_err(|_| RenderError::Cancelled)?;
    let mut g0_pos = [0.0f32; 4];
    for (p, g) in g0_pos.iter_mut().enumerate() {
        *g = params.g0[frame.cfa.color_at(p & 1, p >> 1).plane()];
    }
    let sp = SigmaParams::new(&params.noise, g0_pos);
    let mut out = vec![0.0f32; hw * hh];
    let bands: Vec<(usize, usize)> = (0..hh)
        .step_by(SIGMA_BAND_HALF_ROWS)
        .map(|r0| (r0, (r0 + SIGMA_BAND_HALF_ROWS).min(hh)))
        .collect();
    let width = m.width;
    let results: Vec<Result<(), RenderError>> = out
        .par_chunks_mut(SIGMA_BAND_HALF_ROWS * hw)
        .zip(bands.par_iter())
        .map(|(dst, &(r0, r1))| {
            let c0 = r0.saturating_sub(SIGMA_CONTEXT_HALF_ROWS);
            let c1 = (r1 + SIGMA_CONTEXT_HALF_ROWS).min(hh);
            let slab_h = c1 - c0;
            let mut planes = vec![0.0f32; 4 * hw * slab_h];
            for p in 0..4usize {
                let (px, py) = (p & 1, p >> 1);
                for j in 0..slab_h {
                    let src_row = (2 * (c0 + j) + py) * width;
                    let dst_row = (p * slab_h + j) * hw;
                    for i in 0..hw {
                        planes[dst_row + i] = m.data[src_row + 2 * i + px];
                    }
                }
            }
            let mut slab_out = vec![0.0f32; hw * slab_h];
            sigma_quad(
                &sp,
                &HalfPlanes {
                    data: &planes,
                    width: hw,
                    height: slab_h,
                },
                QstepMap::None,
                None,
                &mut slab_out,
            )
            .map_err(|e| RenderError::Image(format!("The noise map could not be made: {e}.")))?;
            let first = (r0 - c0) * hw;
            dst.copy_from_slice(&slab_out[first..first + (r1 - r0) * hw]);
            Ok(())
        })
        .collect();
    for r in results {
        r?;
    }
    Ok(SigmaPlane {
        width: hw,
        height: hh,
        data: out,
    })
}

/// The vignetting gain (the multiplier R3 applies) at the centre of every pixel of a frame binned
/// by `k`, `ow x oh` of them, or `None` without a profile.
#[must_use]
pub fn gain_plane(plan: &RawPlan, k: usize, ow: usize, oh: usize) -> Option<Vec<f32>> {
    let v = plan.vignette.as_ref()?;
    let (w, h) = (plan.width as f64, plan.height as f64);
    let (cx, cy) = (w * 0.5, h * 0.5);
    let inv_hd = 1.0 / (0.5 * w.hypot(h));
    let kk = k as f64;
    let mut g = vec![1.0f32; ow * oh];
    g.par_chunks_mut(ow.max(1))
        .enumerate()
        .for_each(|(oy, row)| {
            let dy = (oy as f64 + 0.5) * kk - cy;
            for (ox, d) in row.iter_mut().enumerate() {
                let dx = (ox as f64 + 0.5) * kk - cx;
                *d = v.gain_at(dx.hypot(dy) * inv_hd) as f32;
            }
        });
    Some(g)
}

/// How much of a photosite's noise is left in the luminance of a `k x k` bin ([`bin_mosaic`]),
/// relative to one photosite: `sqrt(sum_c w_c^2 v_c)` over the luminance weights `w_c`, with the
/// variance factor `v_c` of each colour's mean: `2 / k^2` for green (the mean of `k^2 / 2`
/// photosites) and the squared sum of the lattice weights, per axis and then squared, for red and
/// blue.
#[must_use]
pub fn binning_noise_factor(k: usize) -> f64 {
    let k = k.max(2);
    let w = [f64::from(LUMA[0]), f64::from(LUMA[1]), f64::from(LUMA[2])];
    let per_axis: f64 = lattice_weights(k, 0)
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum();
    let red_blue = per_axis * per_axis;
    let green = 2.0 / (k * k) as f64;
    (w[0] * w[0] * red_blue + w[1] * w[1] * green + w[2] * w[2] * red_blue).sqrt()
}

/// The sigma of every pixel of a frame binned by `k`, including the vignetting gain.
fn binned_sigma(
    sigma: &SigmaPlane,
    gains: Option<&[f32]>,
    k: usize,
    ow: usize,
    oh: usize,
) -> Vec<f32> {
    let step = (k / 2).max(1);
    let f = binning_noise_factor(k) as f32;
    let mut out = vec![0.0f32; ow * oh];
    out.par_chunks_mut(ow.max(1))
        .enumerate()
        .for_each(|(oy, row)| {
            for (ox, d) in row.iter_mut().enumerate() {
                let mut s2 = 0.0f32;
                let mut n = 0.0f32;
                for cy in oy * step..(oy + 1) * step {
                    let cy = cy.min(sigma.height - 1);
                    for cx in ox * step..(ox + 1) * step {
                        let s = sigma.data[cy * sigma.width + cx.min(sigma.width - 1)];
                        s2 += s * s;
                        n += 1.0;
                    }
                }
                let g = gains.map_or(1.0, |g| g[oy * ow + ox]);
                *d = (s2 / n).sqrt() * f * g;
            }
        });
    out
}

/// The full-resolution sigma map for the vignetting-corrected data: each half-resolution sigma
/// times the gain at its cell's centre, copied to the four photosites of the cell.
fn full_sigma_map(sigma: &SigmaPlane, plan: &RawPlan) -> Vec<f32> {
    let (w, h) = (plan.width, plan.height);
    let gains = gain_plane(plan, 2, sigma.width, sigma.height);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            let cy = (y / 2).min(sigma.height - 1);
            for (x, d) in row.iter_mut().enumerate() {
                let cx = (x / 2).min(sigma.width - 1);
                let g = gains.as_ref().map_or(1.0, |g| g[cy * sigma.width + cx]);
                *d = sigma.data[cy * sigma.width + cx] * g;
            }
        });
    out
}

/// Luminance sigma of a balanced RGB image whose pixels each average `n` samples per channel:
/// `sqrt(sum_c (w_c g0_c)^2 var(x_c) / n)` with `x_c` the unbalanced signal (camera-processed
/// files, which have no mosaic to measure).
fn luma_sigma_rgb(img: &RgbImage, model: &NoiseModel, g0: [f32; 3], n: f64) -> Vec<f32> {
    let np = img.plane_len();
    let w = [
        f64::from(LUMA[0]) * f64::from(g0[0]),
        f64::from(LUMA[1]) * f64::from(g0[1]),
        f64::from(LUMA[2]) * f64::from(g0[2]),
    ];
    let planes = [0usize, 1, 3];
    let mut out = vec![0.0f32; np];
    out.par_chunks_mut(img.width.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            for (x, d) in row.iter_mut().enumerate() {
                let i = y * img.width + x;
                let mut v = 0.0f64;
                for c in 0..3 {
                    let signal = f64::from(img.data[c * np + i]) / f64::from(g0[c]);
                    v += w[c] * w[c] * model.variance(signal, planes[c]);
                }
                *d = (v / n).sqrt() as f32;
            }
        });
    out
}

// ---------------------------------------------------------------------------------------------
// Binning
// ---------------------------------------------------------------------------------------------

/// The integral of a triangle (height 1, half-width 2) centred at 0 from minus infinity to `u`.
fn hat_integral(u: f64) -> f64 {
    if u <= -2.0 {
        0.0
    } else if u <= 0.0 {
        (u + 2.0) * (u + 2.0) / 4.0
    } else if u <= 2.0 {
        2.0 - (2.0 - u) * (2.0 - u) / 4.0
    } else {
        2.0
    }
}

/// The weights of the sites of one colour's lattice along one axis for a block of `k` photosites:
/// the lattice has a site every two photosites, the colour's at parity `q` (centres at
/// `2j + q + 0.5` from the block start, `j` from -1 to `k / 2`), and the weight of a site is the
/// mean over the block of the triangle that interpolates the lattice linearly. The weights sum to
/// one and their centroid is the middle of the block.
fn lattice_weights(k: usize, q: usize) -> Vec<f32> {
    (-1..=(k / 2) as isize)
        .map(|j| {
            let t = 2.0 * j as f64 + q as f64 + 0.5;
            ((hat_integral(k as f64 - t) - hat_integral(-t)) / k as f64) as f32
        })
        .collect()
}

/// `i` moved by whole periods of 2 into `0..n`: the mirror that keeps the colour of a photosite.
#[inline]
fn same_parity(i: isize, n: usize) -> usize {
    let n = n as isize;
    if i < 0 {
        (i + 2).min(n - 1) as usize
    } else if i >= n {
        (i - 2).max(0) as usize
    } else {
        i as usize
    }
}

/// Bins the photosites into `k x k` blocks, one pixel per block with all three colours.
///
/// Green is the mean of the block's `k^2 / 2` green photosites (a quincunx: its middle is the
/// block's). Red and blue sit on a lattice of their own, a site every two photosites, one corner of
/// every 2 x 2 cell, so the plain mean of a block's reds is centred half a photosite up and to the
/// left of the block's middle and the mean of its blues half a photosite down and to the right
/// (RGGB). Binned that way the three planes of an edge are `0.5 / k` of a pixel apart, a red and
/// blue fringe on every sharp edge of the Fit view, a quarter of the edge's contrast at 1/2
/// resolution (found in review: a blurred step edge came out with red and blue 85 display codes
/// apart at 1/2). So red and blue are the mean over the block of the lattice **interpolated
/// linearly** (the weights of `lattice_weights`: the block's sites plus one more at each end, partly weighted,
/// mirrored at the frame's edge), which centres them: a linear ramp bins to equal planes.
///
/// The result is `width / k x height / k` pixels; a trailing partial block is dropped. `k` must be
/// even (the Bayer period).
#[must_use]
pub fn bin_mosaic(m: &Mosaic, k: usize) -> RgbImage {
    debug_assert!(k >= 2 && k.is_multiple_of(2));
    let (w, h) = (m.width, m.height);
    let (ow, oh) = (w / k, h / k);
    let mut img = RgbImage::new(ow, oh);
    if ow == 0 || oh == 0 {
        return img;
    }
    let n = ow * oh;
    let (rp, rest) = img.data.split_at_mut(n);
    let (gp, bp) = rest.split_at_mut(n);
    // The parity (column, row) of the red and the blue lattice in the 2 x 2 cell.
    let parity = |color: GpuColor| {
        (0..2usize)
            .flat_map(|y| (0..2usize).map(move |x| (x, y)))
            .find(|&(x, y)| m.cfa.color_at(x, y) == color)
            .unwrap_or((0, 0))
    };
    let lattices = [GpuColor::Red, GpuColor::Blue].map(|c| {
        let (qx, qy) = parity(c);
        (qx, qy, lattice_weights(k, qx), lattice_weights(k, qy))
    });
    let green_count = (k * k / 2) as f32;
    (
        rp.par_chunks_mut(ow),
        gp.par_chunks_mut(ow),
        bp.par_chunks_mut(ow),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(oy, (r, g, b))| {
            // Green: the mean of the block's green photosites.
            let mut sums = vec![0.0f32; ow];
            for y in oy * k..(oy + 1) * k {
                let row = m.row(y);
                let first = usize::from(m.cfa.color_at(0, y) != GpuColor::Green);
                for (ox, s) in sums.iter_mut().enumerate() {
                    let block = &row[ox * k..(ox + 1) * k];
                    *s += block[first..].iter().step_by(2).sum::<f32>();
                }
            }
            for (d, s) in g.iter_mut().zip(&sums) {
                *d = s / green_count;
            }
            // Red and blue: the interpolated lattice, the rows of the lattice one after the other.
            for (plane, (qx, qy, wx, wy)) in [r, b].into_iter().zip(&lattices) {
                plane.fill(0.0);
                let mut line = vec![0.0f32; ow];
                for (ii, wyi) in wy.iter().enumerate() {
                    let y = (oy * k) as isize + 2 * (ii as isize - 1) + *qy as isize;
                    let row = m.row(same_parity(y, h));
                    for (ox, l) in line.iter_mut().enumerate() {
                        let x0 = (ox * k) as isize + *qx as isize;
                        *l = wx
                            .iter()
                            .enumerate()
                            .map(|(jj, wxj)| wxj * row[same_parity(x0 + 2 * (jj as isize - 1), w)])
                            .sum();
                    }
                    for (p, l) in plane.iter_mut().zip(&line) {
                        *p += wyi * l;
                    }
                }
            }
        });
    img
}

// ---------------------------------------------------------------------------------------------
// The reduced-resolution path
// ---------------------------------------------------------------------------------------------

fn cancelled(cancel: &CancelToken) -> Result<(), RenderError> {
    cancel.check().map_err(|_| RenderError::Cancelled)
}

/// R1 to R9 at `1/k` of the resolution for a mosaic (see the module documentation).
///
/// `k` is 2, 4 or 8 (the caller passes [`RawPlan::effective_scale`]); `sigma` is the plane from
/// [`sigma_plane`].
///
/// # Errors
///
/// [`RenderError::Cancelled`] when `cancel` fires between stages; [`RenderError::Image`] for
/// inputs the stages refuse.
pub fn reduced_bayer(
    frame: &FrameData,
    plan: &RawPlan,
    params: &RawParams,
    sigma: &SigmaPlane,
    k: usize,
    cancel: &CancelToken,
) -> Result<RgbImage, RenderError> {
    debug_assert!(!frame.reduced && k >= 2);
    let (mut m, mask) = normalize::normalize(
        frame.data(),
        frame.width,
        frame.height,
        frame.cfa,
        &plan.levels,
        frame.qstep.as_deref(),
    )
    .map_err(RenderError::from_cpu)?;
    cancelled(cancel)?;
    ym_gpu::cpu::gains::apply_gains(&mut m, params.g0);
    if params.highlight_recovery && plan.has_clipped {
        reconstruct_banded(&mut m, &mask).map_err(RenderError::from_cpu)?;
        cancelled(cancel)?;
    }
    drop(mask);
    let mut img = bin_mosaic(&m, k);
    drop(m);
    cancelled(cancel)?;
    let (ow, oh) = (img.width, img.height);
    // Vignetting at the binned pixels.
    let gains = gain_plane(plan, k, ow, oh);
    if let Some(g) = &gains {
        let n = ow * oh;
        for plane in img.data.chunks_mut(n) {
            plane
                .par_iter_mut()
                .zip(g.par_iter())
                .for_each(|(v, g)| *v *= *g);
        }
    }
    // R6: the light bilateral denoise with the noise that binning left.
    let scale = detail::network_sigma_scale(params.noise_reduction.luminance, params.iso);
    if scale > 0.0 {
        let mut s = binned_sigma(sigma, gains.as_deref(), k, ow, oh);
        s.par_iter_mut().for_each(|v| *v *= scale);
        img =
            denoise::bilateral_opponent(&img, &NoiseSigma::Plane(&s), &BilateralParams::default());
    }
    cancelled(cancel)?;
    // R9.
    finish_lens(img, plan, k)
}

/// R9 for an image at `1/k`: the resample with the scaled auto crop, or the image itself.
fn finish_lens(img: RgbImage, plan: &RawPlan, k: usize) -> Result<RgbImage, RenderError> {
    match plan.lens_at(k)? {
        Some(geo) => {
            let (cw, ch) = geo.output_size();
            ym_gpu::cpu::resample::resample_lens(&img, &geo, Rect::whole(cw, ch))
                .map_err(RenderError::from_cpu)
        }
        None => Ok(img),
    }
}

/// R1 to R8 for a camera-processed file at `1/k` (`k = 1` is full resolution): one range for the
/// three channels, the gains, the light denoise and, at full resolution, R7 and R8.
///
/// # Errors
///
/// [`RenderError::Cancelled`] and [`RenderError::Image`] as [`reduced_bayer`].
pub fn reduced_rgb(
    frame: &FrameData,
    plan: &RawPlan,
    params: &RawParams,
    k: usize,
    cancel: &CancelToken,
) -> Result<RgbImage, RenderError> {
    debug_assert!(frame.reduced);
    let k = k.max(1);
    let (w, h) = (frame.width, frame.height);
    let (ow, oh) = (w / k, h / k);
    let mut img = RgbImage::new(ow, oh);
    let n = ow * oh;
    let data = frame.data();
    let inv = plan.rgb_inv_range;
    let black = plan.rgb_black;
    let g0 = params.g0;
    {
        let (rp, rest) = img.data.split_at_mut(n);
        let (gp, bp) = rest.split_at_mut(n);
        let area = (k * k) as f32;
        (
            rp.par_chunks_mut(ow.max(1)),
            gp.par_chunks_mut(ow.max(1)),
            bp.par_chunks_mut(ow.max(1)),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(oy, (r, g, b))| {
                for ox in 0..ow {
                    let mut s = [0.0f32; 3];
                    for y in oy * k..(oy + 1) * k {
                        let row = &data[(y * w + ox * k) * 3..(y * w + ox * k + k) * 3];
                        for px in row.chunks_exact(3) {
                            for c in 0..3 {
                                let v = (f32::from(px[c]) - black[c]) * inv;
                                s[c] += v.max(normalize::NEGATIVE_FLOOR);
                            }
                        }
                    }
                    r[ox] = s[0] / area * g0[0];
                    g[ox] = s[1] / area * g0[1];
                    b[ox] = s[2] / area * g0[2];
                }
            });
    }
    cancelled(cancel)?;
    let sigma = luma_sigma_rgb(&img, &params.noise, g0, (k * k) as f64);
    let scale = detail::network_sigma_scale(params.noise_reduction.luminance, params.iso);
    let classical = (k == 1 && params.noise_reduction.needs_classical()).then(|| img.clone());
    if scale > 0.0 {
        let s: Vec<f32> = sigma.iter().map(|v| v * scale).collect();
        img =
            denoise::bilateral_opponent(&img, &NoiseSigma::Plane(&s), &BilateralParams::default());
    }
    cancelled(cancel)?;
    if k == 1 {
        let ns = NoiseSigma::Plane(&sigma);
        if params.noise_reduction.color > 0.0 {
            img = detail::chroma_guided_filter(&img, &ns, params.noise_reduction.color);
        }
        if let Some(c) = &classical {
            img = detail::inject_detail(&img, c, params.noise_reduction.beta())
                .map_err(RenderError::from_cpu)?;
        }
        if params.sharpen.iterations() > 0 {
            img = sharpen::richardson_lucy(&img, &params.sharpen, &ns);
        }
    }
    Ok(img)
}

// ---------------------------------------------------------------------------------------------
// The full-resolution path
// ---------------------------------------------------------------------------------------------

/// The classical chain at full resolution, prepared: `cpu::chain::Chain` with the real settings
/// and the whole-frame reductions (the R5 chroma offsets) done.
#[derive(Debug)]
pub struct FullChain {
    /// The chain.
    pub chain: Chain,
    /// The whole-frame reductions.
    pub prepared: Prepared,
    /// Output width (after R9).
    pub out_w: usize,
    /// Output height (after R9).
    pub out_h: usize,
}

impl FullChain {
    /// Builds the chain for a mosaic frame and runs [`Chain::prepare`].
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] for inputs the stages refuse, [`RenderError::Cancelled`].
    pub fn new(
        frame: &FrameData,
        plan: &RawPlan,
        params: &RawParams,
        sigma: &SigmaPlane,
        cancel: &CancelToken,
    ) -> Result<Self, RenderError> {
        debug_assert!(!frame.reduced);
        let scale = detail::network_sigma_scale(params.noise_reduction.luminance, params.iso);
        let chain = Chain {
            levels: plan.levels,
            vignette: plan.vignette.clone(),
            gains: params.g0,
            highlight_recovery: params.highlight_recovery && plan.has_clipped,
            method: DEFAULT_METHOD,
            denoise: (scale > 0.0).then(BilateralParams::default),
            noise_reduction: params.noise_reduction,
            iso: params.iso,
            sigma: 0.0,
            sigma_map: Some(full_sigma_map(sigma, plan)),
            sharpen: (params.sharpen.iterations() > 0).then_some(params.sharpen),
            lens: plan.lens.clone(),
        };
        cancelled(cancel)?;
        let prepared = chain
            .prepare(&frame.as_raw_frame())
            .map_err(RenderError::from_cpu)?;
        let (out_w, out_h) = chain.output_size(frame.width, frame.height);
        Ok(Self {
            chain,
            prepared,
            out_w,
            out_h,
        })
    }

    /// Runs the part `out` (output coordinates) of the chain.
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] when `out` is outside the output.
    pub fn run_tile(&self, frame: &FrameData, out: Rect) -> Result<RgbImage, RenderError> {
        self.chain
            .run_tile(&frame.as_raw_frame(), &self.prepared, out)
            .map_err(RenderError::from_cpu)
    }

    /// Bytes the chain holds (the sigma map).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.chain.sigma_map.as_ref().map_or(0, |m| m.len() * 4)
    }
}

/// Re-exported for the placement of an image that is a whole frame.
#[must_use]
pub fn whole_placement(w: usize, h: usize) -> Placement {
    Placement::whole(w, h)
}

#[cfg(test)]
mod tests {
    use ym_gpu::cpu::synth;
    use ym_gpu::cpu::{ClipMask, Mosaic};

    use super::*;

    /// A natural-looking mosaic pushed past white in places, with the clip mask R1 would make.
    fn clipped(w: usize, h: usize, gain: f32, seed: u64) -> (Mosaic, ClipMask) {
        let img = synth::natural(w, h, seed);
        let mut m = synth::mosaic_from_rgb(&img, Cfa::RGGB);
        let mut mask = ClipMask::new(w, h);
        for (v, f) in m.data.iter_mut().zip(mask.data.iter_mut()) {
            *v *= gain;
            if *v >= 1.0 {
                *v = 1.0;
                *f = 1;
            }
        }
        (m, mask)
    }

    fn max_diff(a: &Mosaic, b: &Mosaic) -> f32 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn banded_highlight_recovery_equals_the_whole_frame_one() {
        // 5.5 bands of 128 rows; clipping in several of them, none in others.
        for (gain, seed) in [(1.9f32, 3u64), (1.3, 4), (2.6, 5)] {
            let (mut a, mask) = clipped(210, 700, gain, seed);
            let mut b = a.clone();
            assert!(mask.count() > 100, "the scene must clip");
            reconstruct_banded(&mut a, &mask).expect("banded");
            highlights::reconstruct(&mut b, &mask).expect("whole");
            assert!(
                max_diff(&a, &b) <= 1e-5,
                "gain {gain}: {}",
                max_diff(&a, &b)
            );
            assert!(
                max_diff(&a, &clipped(210, 700, gain, seed).0) > 0.01,
                "something was reconstructed"
            );
        }
    }

    #[test]
    fn banded_recovery_handles_a_blob_at_a_band_boundary_and_an_untouched_frame() {
        let (w, h) = (96usize, 520usize);
        let mut m = Mosaic::new(w, h, Cfa::RGGB);
        m.data.fill(0.3);
        let mut mask = ClipMask::new(w, h);
        // A blob straddling the boundary between the first and second band (rows 120 to 136).
        for y in 120..136 {
            for x in 30..60 {
                m.data[y * w + x] = 1.0;
                mask.data[y * w + x] = 1;
            }
        }
        let mut a = m.clone();
        let mut b = m.clone();
        reconstruct_banded(&mut a, &mask).expect("banded");
        highlights::reconstruct(&mut b, &mask).expect("whole");
        assert!(max_diff(&a, &b) <= 1e-5);
        // No clipped photosite: nothing changes, and the plan says so.
        let clear = ClipMask::new(w, h);
        let mut c = m.clone();
        reconstruct_banded(&mut c, &clear).expect("banded");
        assert_eq!(c.data, m.data);
    }

    #[test]
    fn has_clipped_agrees_with_the_mask_of_r1() {
        use ym_raw::decode::RawImage;
        let _ = std::marker::PhantomData::<RawImage>;
        // Through the real decoder path: a frame with and without a clipped photosite.
        for clip in [false, true] {
            let (w, h) = (64u32, 48u32);
            let mut data = vec![4000u16; (w * h) as usize];
            if clip {
                data[(20 * w + 31) as usize] = 16300;
            }
            let bytes = ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", w, h)
                .black([512; 4])
                .white([16300; 3])
                .build(&data);
            let raw = std::sync::Arc::new(
                ym_raw::decode::decode_bytes(&bytes, &ym_raw::decode::DecodeOptions::default())
                    .expect("decode"),
            );
            let frame = FrameData::new(&raw).expect("frame");
            let params = RawParams {
                g0: [1.0; 3],
                neural_demosaic: true,
                highlight_recovery: true,
                lens_corrections: true,
                chromatic_aberration: true,
                noise_reduction: NoiseReduction {
                    luminance: 0.0,
                    color: 0.0,
                    detail: 0.0,
                },
                sharpen: SharpenParams {
                    amount: 0.0,
                    radius_px: 0.8,
                },
                iso: 100.0,
                noise: NoiseModel::new(1e-4, 1e-6),
            };
            let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
            assert_eq!(plan.has_clipped, clip);
            let (_, mask) = normalize::normalize(
                frame.data(),
                frame.width,
                frame.height,
                frame.cfa,
                &plan.levels,
                None,
            )
            .expect("normalize");
            assert_eq!(mask.count() > 0, clip);
        }
    }

    #[test]
    fn binning_averages_each_colour_over_the_block() {
        let (w, h) = (16usize, 8usize);
        let mut m = Mosaic::new(w, h, Cfa::RGGB);
        for y in 0..h {
            for x in 0..w {
                // Red 10, greens 20 and 30, blue 40.
                m.data[y * w + x] = match (x & 1, y & 1) {
                    (0, 0) => 10.0,
                    (1, 0) => 20.0,
                    (0, 1) => 30.0,
                    _ => 40.0,
                };
            }
        }
        for k in [2usize, 4] {
            let img = bin_mosaic(&m, k);
            assert_eq!((img.width, img.height), (w / k, h / k));
            for y in 0..img.height {
                for x in 0..img.width {
                    assert_eq!(img.pixel(x, y), [10.0, 25.0, 40.0], "k = {k}");
                }
            }
        }
        // The same data with the phase shifted (GRBG) bins to the same colours.
        let mut g = Mosaic::new(w, h, Cfa::GRBG);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = match g.cfa.color_at(x, y) {
                    GpuColor::Red => 10.0,
                    GpuColor::Green => 25.0,
                    GpuColor::Blue => 40.0,
                };
            }
        }
        assert_eq!(bin_mosaic(&g, 4).pixel(1, 0), [10.0, 25.0, 40.0]);
    }

    #[test]
    fn the_lattice_weights_sum_to_one_and_are_centred_on_the_block() {
        for k in [2usize, 4, 8] {
            for q in [0usize, 1] {
                let w = lattice_weights(k, q);
                assert_eq!(w.len(), k / 2 + 2);
                let sum: f64 = w.iter().map(|v| f64::from(*v)).sum();
                assert!((sum - 1.0).abs() < 1e-6, "k {k} q {q}: {sum}");
                // The centroid of the sites, in photosites from the block start, is its middle.
                let centroid: f64 = w
                    .iter()
                    .enumerate()
                    .map(|(i, v)| f64::from(*v) * (2.0 * (i as f64 - 1.0) + q as f64 + 0.5))
                    .sum();
                assert!(
                    (centroid - k as f64 / 2.0).abs() < 1e-6,
                    "k {k} q {q}: {centroid}"
                );
            }
        }
        // The 2 x 2 block's reds: three sites, the nearest two carrying nearly everything.
        let w = lattice_weights(2, 0);
        assert!((w[0] - 0.03125).abs() < 1e-7 && (w[1] - 0.6875).abs() < 1e-7);
        assert!((w[2] - 0.28125).abs() < 1e-7);
    }

    #[test]
    fn a_ramp_bins_to_three_equal_planes_for_every_phase_and_scale() {
        // A neutral ramp along x and one along y: every colour's mean must be the ramp's value at
        // the middle of the block (the plain mean of a block's reds is half a photosite off, which
        // is a red and blue fringe on every edge).
        for cfa in Cfa::ALL {
            for k in [2usize, 4, 8] {
                let (w, h) = (16 * k, 12 * k);
                let mut m = Mosaic::new(w, h, cfa);
                for y in 0..h {
                    for x in 0..w {
                        m.data[y * w + x] = 0.001 * (x as f32 + 0.5) + 0.0007 * (y as f32 + 0.5);
                    }
                }
                let img = bin_mosaic(&m, k);
                for oy in 1..img.height - 1 {
                    for ox in 1..img.width - 1 {
                        let want =
                            0.001 * (ox * k + k / 2) as f32 + 0.0007 * (oy * k + k / 2) as f32;
                        let p = img.pixel(ox, oy);
                        for (c, v) in p.iter().enumerate() {
                            assert!(
                                (v - want).abs() < 2e-5,
                                "{cfa:?} k {k} ({ox},{oy}) channel {c}: {v} against {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The plain mean of the block's photosites of each colour (what `bin_mosaic` did before):
    /// the reference the alignment is measured against.
    fn plain_bin(m: &Mosaic, k: usize) -> RgbImage {
        let (ow, oh) = (m.width / k, m.height / k);
        let mut img = RgbImage::new(ow, oh);
        let n = ow * oh;
        for oy in 0..oh {
            for ox in 0..ow {
                let mut sum = [0.0f32; 3];
                let mut count = [0.0f32; 3];
                for y in oy * k..(oy + 1) * k {
                    for x in ox * k..(ox + 1) * k {
                        let c = m.cfa.color_at(x, y).plane();
                        sum[c] += m.data[y * m.width + x];
                        count[c] += 1.0;
                    }
                }
                for c in 0..3 {
                    img.data[c * n + oy * ow + ox] = sum[c] / count[c];
                }
            }
        }
        img
    }

    #[test]
    fn a_step_edge_bins_to_planes_that_agree_better_than_the_plain_mean_does() {
        // A neutral step between two photosites, at every phase of the block: red and blue are
        // sampled on lattices of their own, so they cannot agree with green exactly at the edge.
        // The interpolated lattices halve the worst disagreement of the plain mean (which puts the
        // red and the blue sites on opposite sides of the edge: the whole step at 1/2) and lower
        // the mean one too.
        for k in [2usize, 4, 8] {
            let (w, h) = (16 * k, 8 * k);
            let (mut worst_new, mut worst_old) = (0.0f32, 0.0f32);
            let (mut sum_new, mut sum_old) = (0.0f32, 0.0f32);
            for phase in 0..2 * k {
                let edge = 8 * k + phase;
                let mut m = Mosaic::new(w, h, Cfa::RGGB);
                for y in 0..h {
                    for x in 0..w {
                        m.data[y * w + x] = if x < edge { 0.0 } else { 1.0 };
                    }
                }
                let spread = |img: &RgbImage| {
                    (0..img.height)
                        .flat_map(|oy| (1..img.width - 1).map(move |ox| (ox, oy)))
                        .map(|(ox, oy)| {
                            let p = img.pixel(ox, oy);
                            (p[0] - p[2])
                                .abs()
                                .max((p[0] - p[1]).abs())
                                .max((p[2] - p[1]).abs())
                        })
                        .fold(0.0, f32::max)
                };
                let (new, old) = (spread(&bin_mosaic(&m, k)), spread(&plain_bin(&m, k)));
                worst_new = worst_new.max(new);
                worst_old = worst_old.max(old);
                sum_new += new;
                sum_old += old;
            }
            assert!(
                worst_new <= 0.5 * worst_old + 1e-4,
                "k {k}: worst {worst_new} against {worst_old} of the step"
            );
            assert!(sum_new < sum_old, "k {k}: mean {sum_new} against {sum_old}");
        }
    }

    #[test]
    fn the_binned_noise_factor_falls_with_the_scale() {
        // The luminance noise of a bin falls roughly as 2 / k (green's mean of k^2 / 2 photosites
        // dominates it); red and blue average over their interpolated lattices.
        let f = [2usize, 4, 8].map(binning_noise_factor);
        assert!((f[0] - 0.52).abs() < 0.01, "{f:?}");
        assert!(
            (f[0] / f[1] - 2.0).abs() < 0.15 && (f[1] / f[2] - 2.0).abs() < 0.1,
            "{f:?}"
        );
    }

    #[test]
    fn a_noisy_flat_frame_keeps_its_noise_model_at_every_scale() {
        // Measures the claim behind `binning_noise_factor`: the luminance noise of the binned
        // image of a flat noisy mosaic is the model's sigma times the factor.
        let (w, h) = (256usize, 256usize);
        let (a, b) = (3e-3f64, 2e-5f64);
        let level = 0.2f64;
        let sigma = (a * level + b).sqrt();
        let mut rng = synth::Rng::new(11);
        let mut m = Mosaic::new(w, h, Cfa::RGGB);
        for v in &mut m.data {
            *v = (level + sigma * f64::from(rng.gaussian())) as f32;
        }
        for k in [2usize, 4] {
            let img = bin_mosaic(&m, k);
            let y = img.luma_plane();
            let mean = y.iter().map(|v| f64::from(*v)).sum::<f64>() / y.len() as f64;
            let var = y
                .iter()
                .map(|v| (f64::from(*v) - mean).powi(2))
                .sum::<f64>()
                / y.len() as f64;
            let measured = var.sqrt();
            let predicted = sigma * binning_noise_factor(k);
            assert!(
                (measured / predicted - 1.0).abs() < 0.08,
                "k = {k}: measured {measured:.5}, predicted {predicted:.5}"
            );
        }
    }
}
