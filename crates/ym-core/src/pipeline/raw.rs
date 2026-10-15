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
//!   tile by tile (`run_tile`). Demosaic is the classical adaptive ratio method
//!   ([`DEFAULT_METHOD`]) unless an internal switch
//!   ([`RenderOptions::demosaic`](super::graph::RenderOptions::demosaic), set by tests and
//!   evaluation tools) names another method; the neural switch falls back to it (see
//!   [`crate::pipeline::NeuralStatus`]). The adaptive ratio demosaic reads the noise model on the
//!   16-bit working scale ([`RawParams::noise_dn`]); a model that cannot be used is read as zero
//!   noise ([`usable_noise_model`] for the whole raw group, `noise_dn` for the chain), so a photo
//!   never fails to render for it and never comes out black from a `NaN` sigma map.
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

use std::sync::{Arc, OnceLock};

use half::f16;
use rayon::prelude::*;
use ym_gpu::chain::{
    DeviceRunner, DeviceWork, GpuChain, GpuReduced, ReducedDenoise, ReducedParams, StageRange,
};
use ym_gpu::cpu::chain::{Chain, Prepared, RawFrame};
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::detail::{self, NoiseReduction};
use ym_gpu::cpu::highlights::{self, OffsetAccumulator};
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::sharpen::diffraction::{
    DiffractionParams, DiffractionPlan, F_NUMBER_MAX, F_NUMBER_MIN, SPEC_LUMA,
};
use ym_gpu::cpu::sharpen::{self, CaptureSharpen, SharpenParams};
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{
    Cfa, CfaColor as GpuColor, LUMA, Mosaic, NoiseDn, NoiseSigma, Placement, Rect, RgbImage,
};
use ym_gpu::device::{Gpu, GpuContext, GpuError};
use ym_gpu::kernels::bilateral::SigmaHalf;
use ym_gpu::kernels::lens::LensTable;
use ym_gpu::kernels::prep::{DeviceRaw, VignetteTable};
use ym_gpu::queue::{CancelToken, JobSpec};
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
    /// The controls of R8 diffraction deconvolution, `Some` only when that method was chosen
    /// ([`CaptureMethod`]) and the photo has what it needs ([`diffraction_params`]: an f-number
    /// from 1 to 50 and a usable pixel pitch). `None` runs Richardson-Lucy with
    /// [`sharpen`](Self::sharpen), which is also what a `Some` falls back to when the plan turns
    /// out inactive ([`capture_sharpen`](Self::capture_sharpen)). Internal: not part of the
    /// settings.
    pub diffraction: Option<DiffractionParams>,
}

/// Which R8 capture-sharpening method a photo is given (`docs/algorithms/03-diffraction-deconvolution.md`
/// section 1.2). An internal switch with no settings field and no UI: Richardson-Lucy is the
/// default until the owner decides (RA-H2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureMethod {
    /// Richardson-Lucy with a Gaussian PSF.
    #[default]
    RichardsonLucy,
    /// Diffraction deconvolution from the f-number and the pixel pitch; Richardson-Lucy when the
    /// photo has no usable f-number or pitch.
    Diffraction,
}

/// The method every photo gets. Changing it changes every render, so it moves only with the
/// owner's decision (RA-H2).
pub const DEFAULT_CAPTURE_METHOD: CaptureMethod = CaptureMethod::RichardsonLucy;

/// The 35 mm film diagonal in micrometres, `sqrt(36^2 + 24^2) mm` as the spec writes it
/// (`docs/algorithms/03-diffraction-deconvolution.md` section 2.2).
const DIAGONAL_35MM_UM: f64 = 43_266.6;
/// The smallest pixel pitch the stage accepts, in micrometres (spec section 2.2).
pub const PITCH_MIN_UM: f64 = 0.5;
/// The largest pixel pitch the stage accepts, in micrometres (spec section 2.2).
pub const PITCH_MAX_UM: f64 = 20.0;

/// Where the pixel pitch of a photo came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchSource {
    /// The camera registry (`sensor_width_mm` and `active_width_px`, or the stated pitch).
    Registry,
    /// The EXIF crop factor (`focal_length_35mm / focal_length_mm`) and the size of the frame.
    CropFactor,
}

/// The pixel pitch of a photo in micrometres and the image pixels per sensor pixel, with where the
/// pitch came from, or `None` when the photo has neither (spec section 2.2):
///
/// 1. the registry's [`SensorGeometry`](ym_profiles::registry::SensorGeometry) of the body: its
///    pitch, with `scale = frame_width / active_width_px` for a camera-processed reduced file
///    (Lossless M or S; the camera has downscaled the image) and 1 for every mosaic file (a crop
///    mode does not change the pitch);
/// 2. else the EXIF crop factor, `crop = focal_length_35mm / focal_length_mm`, with
///    `pitch = 43266.6 / (crop * sqrt(W^2 + H^2))` for the `W x H` frame R8 receives; that is
///    already the pitch of the delivered image, so `scale = 1` also for a reduced file;
/// 3. a result outside 0.5 to 20 micrometres counts as not available.
#[must_use]
pub fn pixel_pitch(
    meta: &ym_raw::metadata::RawMetadata,
    sensor: Option<&ym_profiles::registry::SensorGeometry>,
    frame_width: usize,
    frame_height: usize,
    reduced: bool,
) -> Option<(f64, f64, PitchSource)> {
    let usable = |pitch: f64| pitch.is_finite() && (PITCH_MIN_UM..=PITCH_MAX_UM).contains(&pitch);
    if let Some(sensor) = sensor
        && usable(sensor.pixel_pitch_um)
        && sensor.active_width_px > 0
    {
        let scale = if reduced {
            frame_width as f64 / f64::from(sensor.active_width_px)
        } else {
            1.0
        };
        return Some((sensor.pixel_pitch_um, scale, PitchSource::Registry));
    }
    let focal = meta.focal_length_mm.filter(|f| f.is_finite() && *f > 0.0)?;
    let focal_35mm = f64::from(meta.focal_length_35mm.filter(|f| *f > 0)?);
    let crop = focal_35mm / focal;
    let diagonal_px = (frame_width as f64).hypot(frame_height as f64);
    let pitch = DIAGONAL_35MM_UM / (crop * diagonal_px);
    usable(pitch).then_some((pitch, 1.0, PitchSource::CropFactor))
}

/// The controls of R8 diffraction deconvolution for a photo, or `None` when the stage is inactive
/// for it: no f-number, an f-number of 0 (manual lenses often write it), `NaN` or outside 1 to 50,
/// or no usable pixel pitch ([`pixel_pitch`]). The caller then runs Richardson-Lucy.
#[must_use]
pub fn diffraction_params(
    meta: &ym_raw::metadata::RawMetadata,
    sensor: Option<&ym_profiles::registry::SensorGeometry>,
    frame_width: usize,
    frame_height: usize,
    reduced: bool,
) -> Option<DiffractionParams> {
    let f_number = meta
        .f_number
        .filter(|f| (F_NUMBER_MIN..=F_NUMBER_MAX).contains(f))?;
    let (pitch_um, scale, _) = pixel_pitch(meta, sensor, frame_width, frame_height, reduced)?;
    Some(DiffractionParams::new(pitch_um, f_number).with_scale(scale))
}

/// Whether `model` can be used by the raw stages: its working-scale form is valid
/// ([`NoiseDn::is_valid`]: finite, not negative, a variance that stays finite up to white) and the
/// black-level offsets are finite (the sigma map subtracts them from the signal).
///
/// The estimator of `ym-noise` never makes a model that fails this; it guards against corrupt data.
#[must_use]
pub fn noise_model_is_usable(model: &NoiseModel) -> bool {
    NoiseDn::from_model(model).is_valid() && model.delta.iter().all(|d| d.is_finite())
}

/// The noise model the raw group works with for a photo: `model` when it is usable
/// ([`noise_model_is_usable`]), the zero-noise model otherwise.
///
/// A photo with a model that cannot be used is processed as if its noise were unknown, never
/// refused and never rendered black from a `NaN` sigma map (`docs/algorithms/INTEGRATION.md`
/// section 10). A usable model is returned unchanged, bit for bit.
#[must_use]
pub fn usable_noise_model(model: &NoiseModel) -> NoiseModel {
    if noise_model_is_usable(model) {
        *model
    } else {
        NoiseModel::new(0.0, 0.0)
    }
}

/// A cache key for [`RawParams`]: the same numbers, bit for bit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawParamsKey(Vec<u64>);

impl RawParams {
    /// The noise model on the 16-bit working scale that the raw-algorithm stages read (the R6
    /// adaptive ratio demosaic, R8 diffraction deconvolution): [`NoiseDn::from_model`] of
    /// [`noise`](Self::noise) when that is usable ([`NoiseDn::is_valid`]), the zero-noise model
    /// otherwise.
    ///
    /// A noise model that cannot be used (a negative or non-finite field, a variance that
    /// overflows at white) never makes a render fail: the photo is processed as if its noise were
    /// unknown, which the algorithms accept (`docs/algorithms/INTEGRATION.md` section 3). The
    /// renderer already replaces such a model by the zero model when it makes the parameters
    /// ([`usable_noise_model`]), so for those this is the model itself; a `RawParams` made by hand
    /// with an unusable model gets the same treatment here.
    #[must_use]
    pub fn noise_dn(&self) -> NoiseDn {
        NoiseDn::from_model(&self.noise).or_zero()
    }

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
        match self.diffraction {
            None => k.push(0),
            Some(d) => {
                k.push(1);
                k.extend(d.pixel_pitch_um.map(f));
                k.extend([f(d.f_number), f(d.scale), g(d.strength)]);
            }
        }
        RawParamsKey(k)
    }

    /// The R8 method to run and its controls (`None`: R8 is skipped). Diffraction deconvolution when
    /// [`diffraction`](Self::diffraction) is set and its plan is active for the gains and the noise
    /// model; otherwise Richardson-Lucy when [`sharpen`](Self::sharpen) asks for any iteration.
    #[must_use]
    pub fn capture_sharpen(&self) -> Option<CaptureSharpen> {
        if let Some(d) = self.diffraction
            && DiffractionPlan::new(&d, self.g0, &self.noise_dn(), SPEC_LUMA).is_some()
        {
            return Some(CaptureSharpen::Diffraction(d));
        }
        (self.sharpen.iterations() > 0).then_some(CaptureSharpen::RichardsonLucy(self.sharpen))
    }

    /// The same parameters with R8 capture sharpening off: neither Richardson-Lucy nor diffraction
    /// deconvolution runs. A render-time switch of the internal chain parameters (no setting, no UI):
    /// the renders of a DNG use it, because a DNG carries no sharpening (`docs/09`, "What a DNG
    /// carries"). Everything else stays as it was, in particular the adaptive ratio demosaic's own
    /// shadow step (`shadow_sharpening`), which is part of the demosaic and not of R8. The key changes
    /// ([`RawParams::key`] reads the amount and the diffraction controls), so a render with R8 off
    /// never reads a cached tile or chain made with R8 on.
    #[must_use]
    pub fn without_capture_sharpening(mut self) -> Self {
        self.sharpen.amount = 0.0;
        self.diffraction = None;
        self
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
            // The crop is the one of the whole geometry (CA included); then the lateral CA moves
            // to the mosaic (R5b), where red and blue are still where the lens put them: a
            // green-guided demosaic pulls them towards green and R9 would over-correct what is left
            // (`docs/05` R9, "Real files"). The reduced path (`lens_at(k)`) bins each colour on its
            // own, which does not do that, and keeps the CA in its R9.
            let g = geometry.with_auto_crop();
            let (ow, oh) = g.output_size();
            (Some(g.with_ca_on_mosaic()), ow, oh)
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
    gain_plane_of(plan.vignette.as_ref(), plan.width, plan.height, k, ow, oh)
}

/// [`gain_plane`] from the profile and the frame size alone (what [`FullChain`] keeps).
fn gain_plane_of(
    vignette: Option<&VignetteProfile>,
    width: usize,
    height: usize,
    k: usize,
    ow: usize,
    oh: usize,
) -> Option<Vec<f32>> {
    let v = vignette?;
    let (w, h) = (width as f64, height as f64);
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
/// times the gain at its cell's centre, copied to the four photosites of the cell. The reference the
/// lazy map of [`FullChain::cpu_chain`] and the device's expansion are tested against.
#[cfg(test)]
fn full_sigma_map(sigma: &SigmaPlane, plan: &RawPlan) -> Vec<f32> {
    let gains = gain_plane(plan, 2, sigma.width, sigma.height);
    expand_sigma_map(sigma, gains.as_deref(), plan.width, plan.height)
}

/// `full_sigma_map` (a test helper) from the cell gains: the rule the device's expansion reproduces cell for cell
/// (`ym_gpu::kernels::bilateral::SigmaHalf::at`).
fn expand_sigma_map(sigma: &SigmaPlane, gains: Option<&[f32]>, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            let cy = (y / 2).min(sigma.height - 1);
            for (x, d) in row.iter_mut().enumerate() {
                let cx = (x / 2).min(sigma.width - 1);
                let g = gains.map_or(1.0, |g| g[cy * sigma.width + cx]);
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

/// What the device's fit-view path ([`GpuReduced`]) needs for the photo at `1/k`: the numbers
/// [`reduced_bayer`] reads, as plain data (the R1 levels, the R4 gains, whether R5 runs, the
/// vignetting profile, the noise model and the gains per CFA position for the sigma plane, the noise
/// the binning leaves, the light denoise and the lens at `1/k`).
///
/// Returns `None` when the device cannot take them (a vignetting profile or a lens table with more
/// knots than the kernels hold, a binning factor they refuse): the CPU makes the group then.
///
/// # Errors
///
/// [`RenderError::Image`] when the lens tables cannot make a map for the smaller frame (the CPU
/// path's error too).
pub fn reduced_params(
    frame: &FrameData,
    plan: &RawPlan,
    params: &RawParams,
    k: usize,
) -> Result<Option<ReducedParams>, RenderError> {
    if k < 2 || !k.is_multiple_of(2) || k > 16 {
        return Ok(None);
    }
    let mut g0_pos = [0.0f32; 4];
    for (p, g) in g0_pos.iter_mut().enumerate() {
        *g = params.g0[frame.cfa.color_at(p & 1, p >> 1).plane()];
    }
    let vignette = match plan.vignette.as_ref().map(VignetteTable::from_profile) {
        Some(Ok(t)) => Some(t),
        Some(Err(_)) => return Ok(None),
        None => None,
    };
    let lens = match plan.lens_at(k)? {
        Some(geo) => match LensTable::from_geometry(&geo) {
            Ok(t) => Some(t),
            Err(_) => return Ok(None),
        },
        None => None,
    };
    let scale = detail::network_sigma_scale(params.noise_reduction.luminance, params.iso);
    Ok(Some(ReducedParams {
        k,
        levels: plan.levels,
        g0: params.g0,
        highlight_recovery: params.highlight_recovery && plan.has_clipped,
        vignette,
        noise: SigmaParams::new(&params.noise, g0_pos),
        noise_factor: binning_noise_factor(k) as f32,
        denoise: (scale > 0.0).then(|| ReducedDenoise {
            scale,
            params: BilateralParams::default(),
        }),
        lens,
    }))
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
        match params.capture_sharpen() {
            Some(CaptureSharpen::RichardsonLucy(p)) => {
                img = sharpen::richardson_lucy(&img, &p, &ns);
            }
            Some(CaptureSharpen::Diffraction(d)) => {
                // `capture_sharpen` returned it because the plan is active, so this always plans.
                let noise = params.noise_dn();
                if let Some(plan) = DiffractionPlan::new(&d, g0, &noise, SPEC_LUMA) {
                    img = plan.apply(&img);
                }
            }
            None => {}
        }
    }
    Ok(img)
}

// ---------------------------------------------------------------------------------------------
// The full-resolution path
// ---------------------------------------------------------------------------------------------

/// A half-resolution sigma plane that [`FullChain`] can keep: the renderer's cached
/// `Arc<SigmaPlane>` is shared (no copy), a plain `&SigmaPlane` (tests, tools) is copied once.
pub trait SharedSigma {
    /// The plane as a shared value.
    fn shared(self) -> Arc<SigmaPlane>;
}

impl SharedSigma for &Arc<SigmaPlane> {
    fn shared(self) -> Arc<SigmaPlane> {
        Arc::clone(self)
    }
}

impl SharedSigma for &SigmaPlane {
    fn shared(self) -> Arc<SigmaPlane> {
        Arc::new(self.clone())
    }
}

/// The classical chain at full resolution, prepared: `cpu::chain::Chain` with the real settings
/// and the whole-frame reductions (the R5 chroma offsets) done.
///
/// **The full-resolution sigma map is lazy.** The CPU stages read the noise sigma of every photosite
/// (`Chain::sigma_map`, 4 bytes a pixel: 200 MB at 50 MP), but the device expands it from the
/// half-resolution plane and the cell gains it already has ([`SigmaHalf`]), tile by tile. So the chain
/// is built without the map ([`device_chain`](Self::device_chain)) and the map is made the first
/// time a CPU stage runs ([`cpu_chain`](Self::cpu_chain), [`run_tile`](Self::run_tile)); a photo
/// whose raw group runs on the device (`StageRange::DEMOSAIC_TO_SHARPEN`) never makes it.
#[derive(Debug)]
pub struct FullChain {
    /// The chain without its full-resolution sigma map (and so with `sigma = 0` for its CPU stages).
    chain: Chain,
    /// The half-resolution noise sigma the map is made from: the renderer's cached plane, shared.
    sigma: Arc<SigmaPlane>,
    /// The vignetting profile, for the gain at the centre of each cell.
    vignette: Option<VignetteProfile>,
    /// Frame size in photosites.
    frame_w: usize,
    frame_h: usize,
    /// The gain of every cell of `sigma` (made the first time the device or the CPU map needs it).
    gains: OnceLock<Option<Vec<f32>>>,
    /// The chain with its sigma map, for the CPU stages (made on first use).
    cpu_chain: OnceLock<Chain>,
    /// The whole-frame reductions.
    pub prepared: Prepared,
    /// Output width (after R9).
    pub out_w: usize,
    /// Output height (after R9).
    pub out_h: usize,
}

impl FullChain {
    /// Builds the chain for a mosaic frame with the default R6 method ([`DEFAULT_METHOD`]) and
    /// runs [`Chain::prepare`].
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] for inputs the stages refuse, [`RenderError::Cancelled`].
    pub fn new(
        frame: &FrameData,
        plan: &RawPlan,
        params: &RawParams,
        sigma: impl SharedSigma,
        cancel: &CancelToken,
    ) -> Result<Self, RenderError> {
        Self::with_method(frame, plan, params, sigma, DEFAULT_METHOD, cancel)
    }

    /// [`FullChain::new`] with the R6 demosaic `method`: the renderer passes
    /// [`RenderOptions::demosaic`](super::graph::RenderOptions::demosaic). The chain gets the R4
    /// gains and the noise model of `params`, which the adaptive ratio demosaic reads; it undoes R4
    /// on each tile itself (`Chain::run_tile`), so every method returns the balanced image.
    ///
    /// # Errors
    ///
    /// As [`FullChain::new`]; also [`RenderError::Image`] when `method` is the adaptive ratio
    /// demosaic and the gains are unusable. A noise model that is unusable is no error: it is read
    /// as zero noise ([`RawParams::noise_dn`]).
    pub fn with_method(
        frame: &FrameData,
        plan: &RawPlan,
        params: &RawParams,
        sigma: impl SharedSigma,
        method: Method,
        cancel: &CancelToken,
    ) -> Result<Self, RenderError> {
        debug_assert!(!frame.reduced);
        let sigma = sigma.shared();
        let scale = detail::network_sigma_scale(params.noise_reduction.luminance, params.iso);
        let noise = params.noise_dn();
        if !noise_model_is_usable(&params.noise) {
            // No file name or path in the message (the log carries none at info level and above).
            tracing::warn!("the noise model of the photo is not usable; rendering with zero noise");
        }
        let chain = Chain {
            levels: plan.levels,
            vignette: plan.vignette.clone(),
            gains: params.g0,
            highlight_recovery: params.highlight_recovery && plan.has_clipped,
            method,
            denoise: (scale > 0.0).then(BilateralParams::default),
            noise_reduction: params.noise_reduction,
            iso: params.iso,
            sigma: 0.0,
            // Made on first use (`cpu_chain`): the device does not read it.
            sigma_map: None,
            noise: Some(noise),
            sharpen: params.capture_sharpen(),
            lens: plan.lens.clone(),
        };
        cancelled(cancel)?;
        let prepared = chain
            .prepare(&frame.as_raw_frame())
            .map_err(RenderError::from_cpu)?;
        let (out_w, out_h) = chain.output_size(frame.width, frame.height);
        Ok(Self {
            chain,
            sigma,
            vignette: plan.vignette.clone(),
            frame_w: plan.width,
            frame_h: plan.height,
            gains: OnceLock::new(),
            cpu_chain: OnceLock::new(),
            prepared,
            out_w,
            out_h,
        })
    }

    /// The gain of every cell of the half-resolution sigma plane, `None` without a vignetting
    /// profile (a gain of exactly 1).
    fn cell_gains(&self) -> Option<&[f32]> {
        self.gains
            .get_or_init(|| {
                gain_plane_of(
                    self.vignette.as_ref(),
                    self.frame_w,
                    self.frame_h,
                    2,
                    self.sigma.width,
                    self.sigma.height,
                )
            })
            .as_deref()
    }

    /// The half-resolution noise sigma and the cell gains, as the device's expansion takes them.
    #[must_use]
    pub fn sigma_half(&self) -> SigmaHalf<'_> {
        SigmaHalf {
            width: self.sigma.width,
            height: self.sigma.height,
            data: &self.sigma.data,
            gains: self.cell_gains(),
        }
    }

    /// The chain with its full-resolution sigma map, for the stages that run on the CPU. The map
    /// (4 bytes a pixel) is made the first time this is called and kept.
    #[must_use]
    pub fn cpu_chain(&self) -> &Chain {
        self.cpu_chain.get_or_init(|| {
            let mut chain = self.chain.clone();
            chain.sigma_map = Some(expand_sigma_map(
                &self.sigma,
                self.cell_gains(),
                self.frame_w,
                self.frame_h,
            ));
            chain
        })
    }

    /// The chain as the device walks it: without the full-resolution sigma map, which the device
    /// expands from [`sigma_half`](Self::sigma_half). A CPU stage run on it would see no noise.
    #[must_use]
    pub fn device_chain(&self) -> &Chain {
        &self.chain
    }

    /// The chain, to change a setting of it (evaluation tools: another lens geometry, a stage off).
    /// The CPU's copy with the sigma map is made again from it on the next CPU run, so a change is
    /// never hidden behind a copy made before it.
    pub fn chain_mut(&mut self) -> &mut Chain {
        self.cpu_chain = OnceLock::new();
        &mut self.chain
    }

    /// Whether the full-resolution sigma map has been made (it is when a CPU stage ran).
    #[must_use]
    pub fn sigma_map_built(&self) -> bool {
        self.cpu_chain.get().is_some()
    }

    /// Runs the part `out` (output coordinates) of the chain on the CPU.
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] when `out` is outside the output.
    pub fn run_tile(&self, frame: &FrameData, out: Rect) -> Result<RgbImage, RenderError> {
        self.cpu_chain()
            .run_tile(&frame.as_raw_frame(), &self.prepared, out)
            .map_err(RenderError::from_cpu)
    }

    /// Runs the part `out` (output coordinates) of the chain with the stages of `route.range` on
    /// the device: the CPU runs the stages before the range on the calling thread, the range goes to
    /// the device as one job (the route's priority, owner and the render's own cancel token), and the
    /// CPU runs the rest. The same contract as [`run_tile`](Self::run_tile), within the tolerances of
    /// the kernels (`docs/05`). A range that holds every stage that reads the noise sigma
    /// ([`StageRange::covers_sigma_stages`]) gets it from the half-resolution plane and the full map
    /// is not made; a shorter range leaves such a stage to the CPU, which reads the map. The range
    /// [`FRONT_RANGE`] (R1 to R9) runs every stage on the device from [`GpuRoute::resident`], the raw
    /// words uploaded once per photo.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when the render's token was set; any other error is the device's (or a
    /// CPU stage's) and the caller answers it with [`run_tile`](Self::run_tile).
    pub fn run_tile_gpu(
        &self,
        frame: &FrameData,
        out: Rect,
        route: &GpuRoute,
    ) -> Result<RgbImage, GpuError> {
        if route.range == StageRange::RAW_TO_LENS {
            // R1 to R9 from the photo's resident raw words: nothing of the frame is read on the host
            // (the CPU's chroma offsets of `prepared` go to the device as numbers).
            let resident = route.resident.as_ref().ok_or_else(|| {
                GpuError::InvalidInput(
                    "the range R1..R9 needs the photo's raw words on the device".into(),
                )
            })?;
            return GpuChain::new(&self.chain, &self.prepared, route.range)?
                .with_sigma(self.sigma_half())?
                .run_tile_front(route, resident, out, &route.token);
        }
        let raw = frame.as_raw_frame();
        if route.range.covers_sigma_stages() {
            GpuChain::new(&self.chain, &self.prepared, route.range)?
                .with_sigma(self.sigma_half())?
                .run_tile(route, &raw, out, &route.token)
        } else {
            GpuChain::new(self.cpu_chain(), &self.prepared, route.range)?.run_tile(
                route,
                &raw,
                out,
                &route.token,
            )
        }
    }

    /// Bytes the chain holds of its own: once made, the cell gains and the full-resolution sigma map.
    /// (The half-resolution plane is the renderer's sigma cache, shared, and counted there.)
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.gains
            .get()
            .and_then(Option::as_ref)
            .map_or(0, |g| g.len() * 4)
            + self
                .cpu_chain
                .get()
                .and_then(|c| c.sigma_map.as_ref())
                .map_or(0, |m| m.len() * 4)
    }
}

/// The stages the renderer runs on the device for the raw group of a full-resolution photo: the adaptive
/// ratio demosaic, the bilateral denoise, R7 and capture sharpening (`StageRange::DEMOSAIC_TO_SHARPEN`,
/// G-06). A route of this range never makes the full-resolution sigma map ([`FullChain::sigma_map_built`]).
pub const DEVICE_RANGE: StageRange = StageRange::DEMOSAIC_TO_SHARPEN;

/// The stages the renderer runs on the device when both [`GpuStages::RAW_CHAIN`] and
/// [`GpuStages::RAW_FRONT`] are on: the whole classical chain R1 to R9 from the raw words that sit on
/// the device (`StageRange::RAW_TO_LENS`, G-07). With the front off the range is [`DEVICE_RANGE`].
///
/// [`GpuStages::RAW_CHAIN`]: crate::app::gpu::GpuStages::RAW_CHAIN
/// [`GpuStages::RAW_FRONT`]: crate::app::gpu::GpuStages::RAW_FRONT
pub const FRONT_RANGE: StageRange = StageRange::RAW_TO_LENS;

/// How often a render waiting for its device job looks at its cancel token.
const DEVICE_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// How the device work of a render's tiles reaches the device: the device, the range of stages it
/// runs, and what every job of the render carries. One route per render, so all of its jobs have one
/// priority and one cancel token (cancelling the render, or navigating away from its photo, reaches
/// the queued ones; `docs/gpu-plan.md` 4.7).
#[derive(Clone)]
pub struct GpuRoute {
    /// The device.
    pub gpu: Arc<Gpu>,
    /// The stages that run on it.
    pub range: StageRange,
    /// The priority of every job of the render.
    pub spec: JobSpec,
    /// The render's cancel token, shared by its jobs.
    pub token: CancelToken,
    /// The photo's raw words on the device, for the range [`FRONT_RANGE`]; `None` for the ranges that
    /// read the host's mosaic.
    pub resident: Option<Arc<DeviceRaw>>,
}

impl std::fmt::Debug for GpuRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRoute")
            .field("range", &self.range)
            .field("spec", &self.spec)
            .field("resident", &self.resident.is_some())
            .finish_non_exhaustive()
    }
}

impl GpuRoute {
    /// Runs `job` on the device as one job of the route (its priority, the render's token) and waits
    /// for it. A render that is cancelled while its job is still queued behind others stops waiting at
    /// once: the job is abandoned, skipped when it is popped (its token is set) and its buffers go
    /// back to the pool when it is dropped. A job that is already running finishes on its own the same
    /// way; the render does not wait for it.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when the token was set; the job's own errors; the scheduler's.
    pub fn run_job<R: Send + 'static>(
        &self,
        job: impl FnOnce(&GpuContext, &CancelToken) -> Result<R, GpuError> + Send + 'static,
    ) -> Result<R, GpuError> {
        let handle =
            self.gpu
                .submit_with_token(self.spec, self.token.clone(), move |ctx, token| {
                    ctx.check_alive()?;
                    job(ctx, token)
                });
        loop {
            if let Some(done) = handle.wait_timeout(DEVICE_POLL) {
                return done?;
            }
            if self.token.is_cancelled() {
                return Err(GpuError::Cancelled);
            }
        }
    }

    /// Uploads the photo's raw words to the device (a job of the route). The caller keeps the result
    /// for the life of the photo: the photo's graph holds it, once per photo, within the renderer's
    /// two-photo LRU.
    ///
    /// # Errors
    ///
    /// As [`run_job`](Self::run_job), and [`GpuError::InvalidInput`] for a frame the upload refuses
    /// (a camera-processed file, a readout above two slabs), the pool's errors.
    pub fn upload(&self, frame: &Arc<FrameData>) -> Result<Arc<DeviceRaw>, GpuError> {
        if frame.reduced {
            return Err(GpuError::InvalidInput(
                "a camera-processed file has no mosaic to put on the device".into(),
            ));
        }
        let frame = Arc::clone(frame);
        self.run_job(move |ctx, _| {
            DeviceRaw::upload(
                ctx,
                "photo raw words",
                frame.width,
                frame.height,
                frame.cfa,
                frame.data(),
                frame.qstep.as_deref(),
            )
            .map(Arc::new)
        })
    }

    /// The fit-view group of the resident photo at `1/k`, read back as a host image.
    ///
    /// # Errors
    ///
    /// As [`run_job`](Self::run_job), and the group's own errors ([`GpuReduced::run_rgb`]).
    pub fn reduced(
        &self,
        resident: &Arc<DeviceRaw>,
        params: ReducedParams,
    ) -> Result<RgbImage, GpuError> {
        let resident = Arc::clone(resident);
        self.run_job(move |ctx, token| GpuReduced::run_rgb(ctx, &resident, &params, token))
    }
}

impl DeviceRunner for GpuRoute {
    /// Submits the work as one job of the route (its priority, the render's token) and waits.
    ///
    /// A render that is cancelled while its job is still queued behind others (navigation, a closed
    /// session) stops waiting at once: the job is abandoned, skipped when it is popped (its token is
    /// set) and its buffers go back to the pool when it is dropped. A job that is already running
    /// finishes on its own the same way; the render does not wait for it.
    fn run_device(&self, work: DeviceWork, _cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        self.run_job(move |ctx, token| work.run(ctx, token))
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

    /// The chain runs R5 on the mosaic after R4, where a clipped photosite sits at `g0` of its
    /// colour: the levels of the three colours are unequal and R5 levels each superpixel up to the
    /// largest bound. The cases above clip at 1.0 everywhere, so they do not exercise that; this
    /// one gives the mosaic the gains `[2.0, 1.0, 1.7]` the way R4 does (the case of
    /// `tiled_reconstruction_equals_whole_with_unequal_clip_levels` in ym-gpu) and asks again that
    /// the banded run is the whole-frame one.
    #[test]
    fn banded_recovery_equals_the_whole_frame_one_with_unequal_clip_levels() {
        const G0: [f32; 3] = [2.0, 1.0, 1.7];
        let (w, h) = (210usize, 700usize);
        for (gain, seed) in [(1.9f32, 3u64), (1.3, 4), (2.6, 5)] {
            let (mut a, mask) = clipped(w, h, gain, seed);
            for (i, v) in a.data.iter_mut().enumerate() {
                *v *= G0[Cfa::RGGB.color_at(i % w, i / w).plane()];
            }
            let before = a.clone();
            let mut b = a.clone();
            assert!(mask.count() > 100, "the scene must clip");
            reconstruct_banded(&mut a, &mask).expect("banded");
            highlights::reconstruct(&mut b, &mask).expect("whole");
            assert!(
                max_diff(&a, &b) <= 1e-5,
                "gain {gain}: banded differs from whole by {}",
                max_diff(&a, &b)
            );
            assert!(
                max_diff(&a, &before) > 0.01,
                "gain {gain}: something was reconstructed"
            );
            // The levelling is active: some clipped green photosite, whose own clip level is 1.0,
            // has been raised towards the largest bound, 2.0 (red's level).
            let leveled = (0..w * h).any(|i| {
                mask.data[i] != 0
                    && Cfa::RGGB.color_at(i % w, i / w) == GpuColor::Green
                    && a.data[i] > 1.5
            });
            assert!(
                leveled,
                "gain {gain}: no clipped green photosite was levelled"
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
                diffraction: None,
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
    fn the_full_chain_carries_the_noise_model_on_the_working_scale() {
        let (w, h) = (64u32, 48u32);
        let data = vec![4000u16; (w * h) as usize];
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
            g0: [2.0, 1.0, 1.6],
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
            noise: NoiseModel {
                a: 2.0e-5,
                b: 1.0e-8,
                delta: [0.0; 4],
                sigma_row: 3.0e-5,
                q: 1.5e-5,
            },
            diffraction: None,
        };
        let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        let sigma = SigmaPlane {
            width: w as usize / 2,
            height: h as usize / 2,
            data: vec![0.01; (w as usize / 2) * (h as usize / 2)],
        };
        let full = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
            .expect("full chain");
        let want = NoiseDn::from_model(&params.noise);
        assert_eq!(full.chain.noise, Some(want));
        // The numbers of docs/algorithms/INTEGRATION.md section 3.
        assert_eq!(want.g, 65535.0 * 2.0e-5);
        let r = 65535.0 * (1.0e-8 + 3.0e-5 * 3.0e-5 + 1.5e-5 * 1.5e-5 / 12.0_f64).sqrt();
        assert_eq!(want.r, r);
    }

    #[test]
    fn binning_averages_each_color_over_the_block() {
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
    // -----------------------------------------------------------------------------------------
    // R8 diffraction deconvolution: pixel pitch, gate, method choice (RA-07)
    // -----------------------------------------------------------------------------------------

    use ym_profiles::registry::{ProfileStore, SensorGeometry};

    /// The metadata of a synthetic ARW of `model` with the given f-number and focal lengths.
    fn meta_of(
        model: &str,
        f_number: Option<f64>,
        focal_mm: Option<f64>,
        focal_35mm: Option<u16>,
    ) -> ym_raw::metadata::RawMetadata {
        let data = vec![4000u16; 16 * 16];
        let bytes = ym_raw::testgen::ArwBuilder::new(model, 16, 16)
            .black([512; 4])
            .white([16300; 3])
            .build(&data);
        let mut meta =
            ym_raw::decode::decode_bytes(&bytes, &ym_raw::decode::DecodeOptions::default())
                .expect("decode")
                .meta;
        meta.f_number = f_number;
        meta.focal_length_mm = focal_mm;
        meta.focal_length_35mm = focal_35mm;
        meta
    }

    fn sensor(width_mm: f64, width_px: u32) -> SensorGeometry {
        SensorGeometry {
            sensor_width_mm: width_mm,
            active_width_px: width_px,
            pixel_pitch_um: 1000.0 * width_mm / f64::from(width_px),
        }
    }

    #[test]
    fn the_registry_gives_the_pitch_and_a_reduced_file_its_scale() {
        let meta = meta_of("ILCE-7RM4", Some(8.0), Some(50.0), Some(50));
        let a7r4 = sensor(35.7, 9504);
        // A mosaic file: the pitch of the sensor, scale 1, whatever the frame (a crop mode too).
        for (w, h) in [(9504, 6336), (6240, 4160)] {
            let (pitch, scale, source) =
                pixel_pitch(&meta, Some(&a7r4), w, h, false).expect("pitch");
            assert!((pitch - 3.7560).abs() < 1e-3, "{pitch}");
            assert_eq!((scale, source), (1.0, PitchSource::Registry));
        }
        // Lossless M (6240 x 4160 from the 9504 px sensor): the camera downscaled the image.
        let (pitch, scale, source) =
            pixel_pitch(&meta, Some(&a7r4), 6240, 4160, true).expect("pitch");
        assert!((pitch - 3.7560).abs() < 1e-3);
        assert!((scale - 6240.0 / 9504.0).abs() < 1e-12, "{scale}");
        assert_eq!(source, PitchSource::Registry);
        // The controls carry both.
        let d = diffraction_params(&meta, Some(&a7r4), 6240, 4160, true).expect("controls");
        assert_eq!(d.f_number, 8.0);
        assert!((d.scale - 6240.0 / 9504.0).abs() < 1e-12);
        assert!((d.effective_pitch_um() - 3.7560 * 9504.0 / 6240.0).abs() < 2e-3);
    }

    #[test]
    fn without_a_registry_sensor_the_crop_factor_gives_the_pitch() {
        // Full frame at 50 mm: crop 1; pitch = 43266.6 / diagonal in pixels.
        let meta = meta_of("ILCE-7RM4", Some(8.0), Some(50.0), Some(50));
        let (pitch, scale, source) = pixel_pitch(&meta, None, 9504, 6336, false).expect("pitch");
        let want = 43_266.6 / (9504.0f64).hypot(6336.0);
        assert!((pitch - want).abs() < 1e-9, "{pitch} vs {want}");
        assert!((pitch - 3.788).abs() < 0.001);
        assert_eq!((scale, source), (1.0, PitchSource::CropFactor));
        // The A7R IV's own pitch is 3.756: the formula is within 1 % here.
        assert!((pitch / 3.756 - 1.0).abs() < 0.01);
        // For 3:2 frames it equals 36000 / (crop * W).
        assert!((pitch - 36_000.0 / 9504.0).abs() < 1e-3);
        // APS-C (crop 1.5, 35 mm equivalent 75 for 50 mm) on a 6000 x 4000 frame: 4.0 um. The real
        // pitch is 3.92 um (23.5 mm over 6000 px): the nominal crop of 1.5 is really 1.53, so the
        // fallback is 2 % off, the error the spec expects of it.
        let aps = meta_of("ILCE-6400", Some(5.6), Some(50.0), Some(75));
        let (pitch, _, _) = pixel_pitch(&aps, None, 6000, 4000, false).expect("pitch");
        assert!((pitch - 4.0).abs() < 1e-3, "{pitch}");
        assert!((pitch / 3.917 - 1.0).abs() < 0.03, "{pitch}");
        // A reduced file: the formula already gives the pitch of the delivered image, scale 1.
        let (pitch, scale, source) = pixel_pitch(&meta, None, 6240, 4160, true).expect("pitch");
        assert!((pitch - 43_266.6 / (6240.0f64).hypot(4160.0)).abs() < 1e-9);
        assert_eq!((scale, source), (1.0, PitchSource::CropFactor));
    }

    #[test]
    fn a_pitch_that_is_missing_or_outside_the_sanity_range_is_not_available() {
        let meta = meta_of("ILCE-7RM4", Some(8.0), Some(50.0), Some(50));
        // No registry, no focal lengths.
        let none = meta_of("ILCE-7RM4", Some(8.0), None, None);
        assert!(pixel_pitch(&none, None, 9504, 6336, false).is_none());
        for (mm, eq) in [
            (Some(0.0), Some(50)),
            (Some(-5.0), Some(50)),
            (Some(f64::NAN), Some(50)),
            (Some(50.0), Some(0)),
            (Some(50.0), None),
            (None, Some(50)),
        ] {
            let m = meta_of("ILCE-7RM4", Some(8.0), mm, eq);
            assert!(
                pixel_pitch(&m, None, 9504, 6336, false).is_none(),
                "{mm:?} {eq:?}"
            );
        }
        // A tiny frame (100 x 60 px) would be a 371 um pitch; a huge one under 0.5 um.
        assert!(pixel_pitch(&meta, None, 100, 60, false).is_none());
        assert!(pixel_pitch(&meta, None, 200_000, 133_000, false).is_none());
        // The limits themselves are inside (0.5 and 20 um).
        let at = |pitch: f64| {
            let diag = 43_266.6 / pitch;
            let w = (diag * 0.832_050_294_337_843_8).round() as usize; // 3:2
            let h = (diag * 0.554_700_196_225_229_1).round() as usize;
            pixel_pitch(&meta, None, w, h, false)
        };
        assert!(at(0.6).is_some() && at(19.0).is_some());
        assert!(at(0.4).is_none() && at(25.0).is_none());
        // A registry sensor outside the range is ignored (the crop factor takes over).
        let silly = SensorGeometry {
            sensor_width_mm: 36.0,
            active_width_px: 100,
            pixel_pitch_um: 360.0,
        };
        let (_, _, source) = pixel_pitch(&meta, Some(&silly), 9504, 6336, false).expect("fallback");
        assert_eq!(source, PitchSource::CropFactor);
    }

    #[test]
    fn the_f_number_gate_is_one_to_fifty() {
        let a7r4 = sensor(35.7, 9504);
        let params = |f: Option<f64>| {
            let meta = meta_of("ILCE-7RM4", f, Some(50.0), Some(50));
            diffraction_params(&meta, Some(&a7r4), 9504, 6336, false)
        };
        for bad in [
            None,
            Some(0.0),
            Some(0.99),
            Some(50.01),
            Some(f64::NAN),
            Some(-4.0),
            Some(f64::INFINITY),
        ] {
            assert!(params(bad).is_none(), "{bad:?}");
        }
        for ok in [1.0, 1.4, 2.8, 8.0, 22.0, 50.0] {
            let d = params(Some(ok)).expect("active");
            assert_eq!((d.f_number, d.strength, d.scale), (ok, 1.0, 1.0));
        }
    }

    #[test]
    fn the_registry_of_the_data_folder_gives_every_listed_body_a_pitch_the_chain_accepts() {
        let store =
            ProfileStore::load(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"))
                .expect("the data folder");
        let mut with_sensor = 0;
        for b in &store.registry().bodies {
            let meta = meta_of(&b.model, Some(8.0), Some(35.0), Some(35));
            let found = store.detect(&meta.make, &meta.model, meta.model_id);
            assert_eq!(found.sensor.is_some(), b.sensor().is_some(), "{}", b.model);
            // With the sensor the registry pitch is used, without it the crop factor (35 mm lens,
            // crop 1, on a frame of the body's own pixel width and 3:2).
            let w = b.active_width_px.map_or(6000, |w| w as usize);
            let (pitch, _, source) =
                pixel_pitch(&meta, found.sensor.as_ref(), w, w * 2 / 3, false).expect("a pitch");
            assert!(
                (PITCH_MIN_UM..=PITCH_MAX_UM).contains(&pitch),
                "{}: {pitch}",
                b.model
            );
            if let Some(s) = found.sensor {
                with_sensor += 1;
                assert_eq!(source, PitchSource::Registry);
                assert_eq!(pitch, s.pixel_pitch_um);
            } else {
                assert_eq!(
                    (b.model.as_str(), source),
                    ("ILME-FX3", PitchSource::CropFactor)
                );
            }
        }
        assert_eq!(with_sensor, 27);
        // A body the registry does not list has no sensor: the crop factor (or nothing).
        let meta = meta_of("ILCE-7M6", Some(8.0), Some(50.0), Some(50));
        let found = store.detect(&meta.make, &meta.model, meta.model_id);
        assert!(found.sensor.is_none());
    }

    fn params_for_tests() -> RawParams {
        RawParams {
            g0: [2.0, 1.0, 1.6],
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
                amount: 120.0,
                radius_px: 0.8,
            },
            iso: 400.0,
            noise: NoiseModel::new(3e-5, 4e-8),
            diffraction: None,
        }
    }

    #[test]
    fn richardson_lucy_is_the_default_and_the_fallback() {
        assert_eq!(DEFAULT_CAPTURE_METHOD, CaptureMethod::RichardsonLucy);
        assert_eq!(CaptureMethod::default(), CaptureMethod::RichardsonLucy);
        let mut p = params_for_tests();
        // No diffraction controls: Richardson-Lucy with the resolved amount and radius.
        assert_eq!(
            p.capture_sharpen(),
            Some(CaptureSharpen::RichardsonLucy(p.sharpen))
        );
        // Amount 0: R8 is skipped.
        p.sharpen.amount = 0.0;
        assert_eq!(p.capture_sharpen(), None);
        // Diffraction controls with an active plan: Diffraction, whatever the RL amount.
        let d = DiffractionParams::new(3.76, 11.0);
        p.diffraction = Some(d);
        assert_eq!(p.capture_sharpen(), Some(CaptureSharpen::Diffraction(d)));
        p.sharpen.amount = 120.0;
        assert_eq!(p.capture_sharpen(), Some(CaptureSharpen::Diffraction(d)));
        // A plan that is inactive (a bad gain) falls back to Richardson-Lucy.
        p.g0 = [2.0, f32::NAN, 1.6];
        assert_eq!(
            p.capture_sharpen(),
            Some(CaptureSharpen::RichardsonLucy(p.sharpen))
        );
        // The cache key tells the two apart, and tells controls apart.
        let mut q = params_for_tests();
        let key_rl = q.key();
        q.diffraction = Some(d);
        let key_d = q.key();
        assert_ne!(key_rl, key_d);
        q.diffraction = Some(d.with_scale(0.5));
        assert_ne!(key_d, q.key());
        q.diffraction = Some(d);
        assert_eq!(key_d, q.key());
    }

    /// A frame with a step edge and some texture, exposed to half scale, as a raw file.
    fn edge_frame(w: u32, h: u32) -> (Arc<RawImage>, FrameData) {
        let mut data = vec![0u16; (w * h) as usize];
        let mut rng = synth::Rng::new(5);
        for y in 0..h {
            for x in 0..w {
                // A soft vertical edge at x = w / 2 (blurred over about 3 px) and a tilted one.
                let t = ((f64::from(x) - f64::from(w) / 2.0) / 3.0).tanh() * 0.5 + 0.5;
                let u = ((f64::from(x) + 0.6 * f64::from(y) - f64::from(w) / 3.0) / 4.0).tanh()
                    * 0.5
                    + 0.5;
                let v = 0.05 + 0.35 * t + 0.1 * u + 0.002 * f64::from(rng.gaussian());
                data[(y * w + x) as usize] = (512.0 + v.clamp(0.0, 1.0) * 15788.0) as u16;
            }
        }
        let bytes = ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", w, h)
            .black([512; 4])
            .white([16300; 3])
            .build(&data);
        let raw = Arc::new(
            ym_raw::decode::decode_bytes(&bytes, &ym_raw::decode::DecodeOptions::default())
                .expect("decode"),
        );
        let frame = FrameData::new(&raw).expect("frame");
        (raw, frame)
    }

    #[test]
    fn the_full_chain_runs_the_chosen_method_and_tiles_equal_the_whole_frame() {
        let (w, h) = (160u32, 120u32);
        let (raw, frame) = edge_frame(w, h);
        let sigma = SigmaPlane {
            width: w as usize / 2,
            height: h as usize / 2,
            data: vec![0.004; (w as usize / 2) * (h as usize / 2)],
        };
        let mut params = params_for_tests();
        params.highlight_recovery = false;
        let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        let run = |p: &RawParams| {
            let full =
                FullChain::new(&frame, &plan, p, &sigma, &CancelToken::new()).expect("chain");
            let whole = full
                .run_tile(&frame, Rect::whole(full.out_w, full.out_h))
                .expect("whole frame");
            (full, whole)
        };
        let (rl_chain, rl) = run(&params);
        assert!(matches!(
            rl_chain.chain.sharpen,
            Some(CaptureSharpen::RichardsonLucy(_))
        ));
        let mut with_d = params;
        with_d.diffraction = Some(DiffractionParams::new(3.76, 16.0));
        let (d_chain, d) = run(&with_d);
        assert!(matches!(
            d_chain.chain.sharpen,
            Some(CaptureSharpen::Diffraction(_))
        ));
        assert!(
            rl.max_abs_diff(&d) > 1e-4,
            "the two methods give different pixels"
        );
        // Without R8 the output differs from both.
        let mut off = params;
        off.sharpen.amount = 0.0;
        let (_, plain) = run(&off);
        assert!(plain.max_abs_diff(&d) > 1e-4 && plain.max_abs_diff(&rl) > 1e-4);
        // Inactive controls (f/0, a pitch of 0): Richardson-Lucy, the same bits as the default.
        for bad in [
            DiffractionParams::new(3.76, 0.0),
            DiffractionParams::new(0.0, 8.0),
        ] {
            let mut p = params;
            p.diffraction = Some(bad);
            let (chain, img) = run(&p);
            assert!(matches!(
                chain.chain.sharpen,
                Some(CaptureSharpen::RichardsonLucy(_))
            ));
            assert_eq!(img, rl);
        }
        // Tiles with the plan's halo give the whole frame (the R8 stage is exact; the rest is 1e-4).
        let tiles = ym_gpu::cpu::tiling::tile_grid(d_chain.out_w, d_chain.out_h, 64, 48, 2);
        let mut tiled = RgbImage::new(d_chain.out_w, d_chain.out_h);
        for core in tiles {
            let t = d_chain.run_tile(&frame, core).expect("tile");
            tiled.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
        }
        assert!(d.max_abs_diff(&tiled) <= 1e-4, "{}", d.max_abs_diff(&tiled));
    }

    /// RA-04: a full render with the adaptive ratio demosaic forced (what
    /// `RenderOptions::demosaic` does) is the whole-frame `demosaic_with` call.
    ///
    /// How it is compared: every stage but R1, R4 and R6 is off (no denoise, no R7 control, no R8,
    /// no lens, no vignetting, no highlight recovery), so the chain's whole-frame output must be
    /// the demosaic of the balanced mosaic that R1 and R4 make, done by hand here over the whole
    /// frame, bit for bit; and so must every tile of a grid (tile versus whole).
    #[test]
    fn a_full_render_with_the_adaptive_ratio_demosaic_is_the_whole_frame_demosaic() {
        let (w, h) = (160u32, 120u32);
        let (raw, frame) = edge_frame(w, h);
        let sigma = SigmaPlane {
            width: w as usize / 2,
            height: h as usize / 2,
            data: vec![0.004; (w as usize / 2) * (h as usize / 2)],
        };
        let models = [
            // SNR at 18 % grey above 25 dB: shadow matching off, halo 11.
            NoiseModel::new(3e-5, 4e-8),
            // 23.9 dB: shadow matching on, halo 13.
            NoiseModel {
                a: 5e-4,
                b: 4e-5,
                delta: [0.0; 4],
                sigma_row: 1e-3,
                q: 0.0,
            },
        ];
        let mut menon_pictures = Vec::new();
        for model in models {
            let mut params = params_for_tests();
            params.highlight_recovery = false;
            params.lens_corrections = false;
            params.chromatic_aberration = false;
            params.sharpen.amount = 0.0;
            params.noise = model;
            let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
            let build = |method: Method| {
                FullChain::with_method(&frame, &plan, &params, &sigma, method, &CancelToken::new())
                    .expect("chain")
            };
            let full = build(Method::AdaptiveRatio);
            let c = &full.chain;
            assert_eq!(c.method, Method::AdaptiveRatio);
            assert!(
                c.vignette.is_none()
                    && c.lens.is_none()
                    && c.denoise.is_none()
                    && c.sharpen.is_none(),
                "only R1, R4 and R6 are left in this chain: vignette {:?}, lens {:?}, denoise {:?}, sharpen {:?}",
                c.vignette.is_some(),
                c.lens.is_some(),
                c.denoise,
                c.sharpen
            );
            assert_eq!(c.gains, params.g0);
            assert_eq!(c.noise, Some(NoiseDn::from_model(&model)));
            let whole = full
                .run_tile(&frame, Rect::whole(full.out_w, full.out_h))
                .expect("whole frame");

            // By hand: R1 and R4 over the whole frame, then one whole-frame `demosaic_with`.
            let (mut m, _) = normalize::normalize(
                frame.data(),
                frame.width,
                frame.height,
                frame.cfa,
                &plan.levels,
                frame.qstep.as_deref(),
            )
            .expect("normalize");
            ym_gpu::cpu::gains::apply_gains(&mut m, params.g0);
            let inputs = ym_gpu::cpu::demosaic::DemosaicInputs {
                gains: params.g0,
                noise: NoiseDn::from_model(&model),
                ca: None,
            };
            let by_hand =
                ym_gpu::cpu::demosaic::demosaic_whole_with(&m, Method::AdaptiveRatio, &inputs);
            assert_eq!(whole, by_hand, "the chain is the whole-frame demosaic");

            // Tile versus whole: tiles of a grid that does not divide the frame.
            for (tw, th) in [(64, 48), (50, 38)] {
                let mut tiled = RgbImage::new(full.out_w, full.out_h);
                for core in ym_gpu::cpu::tiling::tile_grid(full.out_w, full.out_h, tw, th, 2) {
                    let t = full.run_tile(&frame, core).expect("tile");
                    tiled.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
                }
                assert_eq!(tiled, whole, "tiles of {tw} x {th} against the whole frame");
            }

            // The default method is the adaptive ratio demosaic (owner's decision, 2026-10-13):
            // `FullChain::new` is the forced chain, bit for bit, and Menon is another picture.
            assert_eq!(DEFAULT_METHOD, Method::AdaptiveRatio);
            let default = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
                .expect("default chain");
            assert_eq!(default.chain.method, DEFAULT_METHOD);
            let default_whole = default
                .run_tile(&frame, Rect::whole(full.out_w, full.out_h))
                .expect("default whole frame");
            assert_eq!(
                default_whole, whole,
                "the default is the adaptive ratio demosaic"
            );
            let menon = build(Method::Menon)
                .run_tile(&frame, Rect::whole(full.out_w, full.out_h))
                .expect("Menon whole frame");
            assert!(whole.max_abs_diff(&menon) > 1e-4, "the methods differ");
            menon_pictures.push(menon);
        }
        // Menon ignores the noise model, so both runs of it are the same picture.
        assert_eq!(menon_pictures.len(), models.len());
        assert_eq!(menon_pictures[0], menon_pictures[1]);
    }

    #[test]
    fn the_adaptive_ratio_chain_refuses_gains_it_cannot_divide_by() {
        let (w, h) = (64u32, 48u32);
        let (raw, frame) = edge_frame(w, h);
        let sigma = SigmaPlane {
            width: w as usize / 2,
            height: h as usize / 2,
            data: vec![0.004; (w as usize / 2) * (h as usize / 2)],
        };
        let mut params = params_for_tests();
        params.highlight_recovery = false;
        let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        params.g0 = [2.0, 1.0, 0.0];
        let refused = FullChain::with_method(
            &frame,
            &plan,
            &params,
            &sigma,
            Method::AdaptiveRatio,
            &CancelToken::new(),
        );
        assert!(matches!(refused, Err(RenderError::Image(m)) if m.contains("adaptive ratio")));
        // A method that does not divide is not asked.
        FullChain::with_method(
            &frame,
            &plan,
            &params,
            &sigma,
            Method::Menon,
            &CancelToken::new(),
        )
        .expect("Menon runs");
    }

    /// A photo whose noise model cannot be used still renders, with the default adaptive ratio
    /// demosaic: it is read as zero noise (`RawParams::noise_dn`), the same picture as a photo
    /// whose model is exactly zero.
    #[test]
    fn an_unusable_noise_model_renders_as_zero_noise_instead_of_failing() {
        let (w, h) = (96u32, 64u32);
        let (raw, frame) = edge_frame(w, h);
        let sigma = SigmaPlane {
            width: w as usize / 2,
            height: h as usize / 2,
            data: vec![0.004; (w as usize / 2) * (h as usize / 2)],
        };
        let render = |model: NoiseModel| {
            let mut params = params_for_tests();
            params.highlight_recovery = false;
            params.lens_corrections = false;
            params.chromatic_aberration = false;
            params.sharpen.amount = 0.0;
            params.noise = model;
            let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
            // The default constructor: the adaptive ratio demosaic.
            let full = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
                .expect("a photo with this noise model renders");
            assert_eq!(full.chain.method, Method::AdaptiveRatio);
            let noise = full.chain.noise.expect("the chain carries a noise model");
            assert!(noise.is_valid(), "{noise:?}");
            let img = full
                .run_tile(&frame, Rect::whole(full.out_w, full.out_h))
                .expect("whole frame");
            (params.noise_dn(), noise, img)
        };
        let zero_model = NoiseModel {
            a: 0.0,
            b: 0.0,
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 0.0,
        };
        let (zero_dn, zero_chain_noise, zero_img) = render(zero_model);
        assert_eq!(zero_dn, NoiseDn::ZERO);
        assert_eq!(zero_chain_noise, NoiseDn::ZERO);
        let nan_model = NoiseModel::new(f64::NAN, 4e-8);
        let negative_model = NoiseModel::new(-3e-5, 4e-8);
        let negative_variance = NoiseModel::new(3e-5, -1e-6);
        let overflow_model = NoiseModel::new(1e300, 1e300);
        let infinite_model = NoiseModel::new(f64::INFINITY, 4e-8);
        for (name, model) in [
            ("NaN gain", nan_model),
            ("negative gain", negative_model),
            ("negative variance", negative_variance),
            ("variance overflow", overflow_model),
            ("infinite gain", infinite_model),
        ] {
            let (dn, chain_noise, img) = render(model);
            assert_eq!(dn, NoiseDn::ZERO, "{name}");
            assert_eq!(chain_noise, NoiseDn::ZERO, "{name}");
            assert_eq!(img, zero_img, "{name}: the picture of a zero noise model");
        }
        // A usable model is not touched, and it is another picture than the zero model (the
        // noise gates of the demosaic move).
        let usable = NoiseModel::new(3e-5, 4e-8);
        let (dn, chain_noise, img) = render(usable);
        assert_eq!(dn, NoiseDn::from_model(&usable));
        assert_eq!(chain_noise, dn);
        assert!(
            img.max_abs_diff(&zero_img) > 0.0,
            "the model matters to the adaptive ratio"
        );
    }

    /// The renderer's guard for the whole raw group (`graph.rs`): a usable model is kept bit for
    /// bit, an unusable one (including a black-level offset that is not finite, which the
    /// working-scale model does not read but the sigma map does) is the zero model.
    #[test]
    fn a_usable_noise_model_is_kept_and_an_unusable_one_is_the_zero_model() {
        let zero = NoiseModel::new(0.0, 0.0);
        let full = NoiseModel {
            a: 3e-5,
            b: 4e-8,
            delta: [1e-4, -2e-4, 3e-4, 0.0],
            sigma_row: 2e-5,
            q: 6e-5,
        };
        for usable in [full, zero, NoiseModel::new(5e-4, 4e-5)] {
            assert!(noise_model_is_usable(&usable), "{usable:?}");
            assert_eq!(usable_noise_model(&usable), usable);
        }
        let mut nan_offset = full;
        nan_offset.delta[1] = f64::NAN;
        let mut infinite_offset = full;
        infinite_offset.delta[3] = f64::NEG_INFINITY;
        let mut nan_row = full;
        nan_row.sigma_row = f64::NAN;
        let mut nan_q = full;
        nan_q.q = f64::NAN;
        for unusable in [
            NoiseModel::new(f64::NAN, 4e-8),
            NoiseModel::new(f64::INFINITY, 4e-8),
            NoiseModel::new(-3e-5, 4e-8),
            NoiseModel::new(3e-5, -4e-8),
            NoiseModel::new(1e300, 1e300),
            nan_offset,
            infinite_offset,
            nan_row,
            nan_q,
        ] {
            assert!(!noise_model_is_usable(&unusable), "{unusable:?}");
            assert_eq!(usable_noise_model(&unusable), zero, "{unusable:?}");
            assert_eq!(
                NoiseDn::from_model(&usable_noise_model(&unusable)),
                NoiseDn::ZERO
            );
        }
    }

    #[test]
    fn a_reduced_camera_file_runs_the_chosen_method_too() {
        // The camera-processed path (Lossless M and S) has its own R8 call, in `reduced_rgb`.
        let (w, h) = (96u32, 64u32);
        let mut rgb = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                let t = ((f64::from(x) - 48.0) / 2.5).tanh() * 0.5 + 0.5;
                for c in 0..3 {
                    rgb.push((1024.0 + (0.05 + 0.4 * t) * (1.0 + 0.1 * c as f64) * 14000.0) as u16);
                }
            }
        }
        let bytes = ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", w, h)
            .lossless_ms(ym_raw::testgen::LosslessMs::new(
                ym_raw::layout::ChromaSubsampling::Yuv420,
                rgb,
                w,
                h,
                [1.0, 1.0, 1.0],
            ))
            .build(&[]);
        let raw = Arc::new(
            ym_raw::decode::decode_bytes(&bytes, &ym_raw::decode::DecodeOptions::default())
                .expect("decode"),
        );
        assert!(raw.is_reduced());
        let frame = FrameData::new(&raw).expect("frame");
        let mut params = params_for_tests();
        params.g0 = [1.0, 1.0, 1.0];
        let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        let run = |p: &RawParams| {
            reduced_rgb(&frame, &plan, p, 1, &CancelToken::new()).expect("reduced render")
        };
        let rl = run(&params);
        let mut with_d = params;
        with_d.diffraction = Some(DiffractionParams::new(3.76, 16.0).with_scale(0.66));
        let d = run(&with_d);
        assert!(
            rl.max_abs_diff(&d) > 1e-4,
            "the two methods give different pixels"
        );
        // Inactive controls give the Richardson-Lucy bits of the default.
        let mut inactive = params;
        inactive.diffraction = Some(DiffractionParams::new(3.76, 0.0));
        assert_eq!(run(&inactive), rl);
        // R8 off: neither.
        let mut off = params;
        off.sharpen.amount = 0.0;
        let plain = run(&off);
        assert!(plain.max_abs_diff(&d) > 1e-4 && plain.max_abs_diff(&rl) > 1e-4);
        // The diffraction result is the plan applied to the R7 output (here: plain).
        let plan_d = DiffractionPlan::new(
            &DiffractionParams::new(3.76, 16.0).with_scale(0.66),
            params.g0,
            &NoiseDn::from_model(&params.noise),
            SPEC_LUMA,
        )
        .expect("active");
        assert_eq!(plan_d.apply(&plain), d);
    }

    // ------------------------------------------------------------------------------------------
    // G-06: the lazy full-resolution sigma map and the device's expansion of the half plane
    // ------------------------------------------------------------------------------------------

    /// A frame with a step edge, the noise reduction on (so that the denoise, R7 and R8 all run), a
    /// vignetting profile (the synthetic file has none) and a half-resolution sigma that varies.
    fn full_chain_with_every_stage(w: u32, h: u32) -> (FrameData, RawPlan, SigmaPlane, RawParams) {
        let (raw, frame) = edge_frame(w, h);
        let mut params = params_for_tests();
        params.highlight_recovery = false;
        params.noise_reduction = NoiseReduction {
            luminance: 40.0,
            color: 60.0,
            detail: 70.0,
        };
        let mut plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        plan.vignette = Some(
            VignetteProfile::from_tag(&[0, -400, -1500, -3500, -6000]).expect("vignetting profile"),
        );
        let (cw, ch) = (w as usize / 2, h as usize / 2);
        let mut rng = synth::Rng::new(9);
        let sigma = SigmaPlane {
            width: cw,
            height: ch,
            data: (0..cw * ch)
                .map(|i| 0.003 + 0.006 * (i % cw) as f32 / cw as f32 + 0.001 * rng.next_f32())
                .collect(),
        };
        (frame, plan, sigma, params)
    }

    #[test]
    fn the_full_resolution_sigma_map_is_made_on_the_first_cpu_use_and_is_the_old_one() {
        // Odd sizes: the last photosite column and row read the last cell.
        let (w, h) = (161u32, 121u32);
        let (frame, plan, sigma, params) = full_chain_with_every_stage(w, h);
        let full = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
            .expect("full chain");
        // Nothing but the half plane is held, and the chain the device walks has no map.
        assert!(!full.sigma_map_built());
        assert_eq!(full.bytes(), 0);
        assert!(full.device_chain().sigma_map.is_none());
        let want = full_sigma_map(&sigma, &plan);
        // The half-resolution view the device takes is the cells and the gains of the plan.
        let half = full.sigma_half();
        assert_eq!((half.width, half.height), (sigma.width, sigma.height));
        let gains = gain_plane(&plan, 2, sigma.width, sigma.height).expect("gains");
        assert_eq!(half.gains, Some(gains.as_slice()));
        for (x, y) in [
            (0usize, 0usize),
            (3, 5),
            (160, 120),
            (159, 119),
            (160, 0),
            (0, 120),
        ] {
            assert_eq!(half.at(x, y).to_bits(), want[y * w as usize + x].to_bits());
        }
        assert!(!full.sigma_map_built(), "the half view makes no map");
        // The first CPU use makes the map, and it is the one the eager chain had, bit for bit.
        let tile = Rect::new(7, 5, 80, 70);
        let a = full.run_tile(&frame, tile).expect("cpu tile");
        assert!(full.sigma_map_built());
        let map = full.cpu_chain().sigma_map.as_ref().expect("map");
        assert_eq!(map.len(), (w * h) as usize);
        assert!(
            map.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(full.bytes(), gains.len() * 4 + map.len() * 4);
        let mut eager = full.device_chain().clone();
        eager.sigma_map = Some(want);
        let b = eager
            .run_tile(&frame.as_raw_frame(), &full.prepared, tile)
            .expect("eager tile");
        assert_eq!(
            a.data, b.data,
            "the lazy map gives the eager chain's pixels"
        );
        // The noise reduction was on: the sigma reached the stages (a chain without a map would
        // have denoised nothing, it has `sigma = 0`).
        let blind = full
            .device_chain()
            .run_tile(&frame.as_raw_frame(), &full.prepared, tile)
            .expect("blind tile");
        assert!(a.max_abs_diff(&blind) > 1e-5, "the sigma map has an effect");
        // A change through `chain_mut` drops the CPU's copy, so the next CPU run sees it.
        let mut full = full;
        full.chain_mut().noise_reduction.color = 0.0;
        assert!(!full.sigma_map_built());
        let c = full
            .run_tile(&frame, tile)
            .expect("cpu tile after a change");
        assert!(full.sigma_map_built());
        assert!(
            c.max_abs_diff(&a) > 1e-6,
            "the change reached the CPU chain"
        );
    }

    /// The device: policy `On` through the application's own opener (a device is created nowhere
    /// else, `tests/gpu_policy.rs`), or `None` without an adapter, loudly unless the environment
    /// requires one.
    fn device() -> Option<Arc<Gpu>> {
        let slot = crate::app::gpu::open_gpu(
            crate::app::gpu::GpuPolicy::On,
            crate::scheduler::MemoryBudget::detect(),
        );
        match slot.gpu() {
            Some(gpu) => Some(Arc::clone(gpu)),
            None => {
                assert!(
                    !ym_gpu::device::gpu_required_by_env(),
                    "YM_REQUIRE_GPU is set but no device opened: {:?}",
                    slot.failure()
                );
                eprintln!("no GPU adapter: the device tests of raw.rs are skipped");
                None
            }
        }
    }

    #[test]
    fn the_device_expands_the_half_resolution_sigma_like_the_cpus_full_map_cell_for_cell() {
        let Some(gpu) = device() else { return };
        for (w, h) in [(161u32, 121u32), (160, 120)] {
            let (frame, plan, sigma, params) = full_chain_with_every_stage(w, h);
            let full = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
                .expect("full chain");
            let want = full_sigma_map(&sigma, &plan);
            for rect in [
                Rect::new(0, 0, w as usize, h as usize),
                Rect::new(1, 3, 77, 61),
                Rect::new(w as usize - 5, h as usize - 3, 5, 3),
                Rect::new(40, 0, 1, h as usize),
            ] {
                let got = gpu
                    .run(JobSpec::new(ym_gpu::queue::Priority::Export), {
                        let (data, gains) = (
                            sigma.data.clone(),
                            full.sigma_half().gains.map(<[f32]>::to_vec),
                        );
                        let (cw, ch) = (sigma.width, sigma.height);
                        move |ctx, _| {
                            let half = SigmaHalf {
                                width: cw,
                                height: ch,
                                data: &data,
                                gains: gains.as_deref(),
                            };
                            ym_gpu::kernels::bilateral::expand_sigma(
                                ctx,
                                &ym_gpu::kernels::bilateral::SigmaInput::Half(&half),
                                rect,
                            )
                        }
                    })
                    .expect("device expansion");
                let mut cells = Vec::new();
                for y in rect.y..rect.y1() {
                    cells.extend_from_slice(
                        &want[y * w as usize + rect.x..y * w as usize + rect.x1()],
                    );
                }
                assert!(
                    got.iter()
                        .zip(&cells)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{w}x{h} {rect:?}: the device's sigma plane is not the CPU map's"
                );
            }
        }
    }

    #[test]
    fn a_route_of_the_raw_group_never_makes_the_full_sigma_map_and_a_shorter_one_does() {
        let Some(gpu) = device() else { return };
        let (w, h) = (200u32, 150u32);
        let (frame, plan, sigma, params) = full_chain_with_every_stage(w, h);
        let route = |range| GpuRoute {
            gpu: Arc::clone(&gpu),
            range,
            spec: JobSpec::new(ym_gpu::queue::Priority::Export),
            token: CancelToken::new(),
            resident: None,
        };
        let full = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
            .expect("full chain");
        let rects = [
            Rect::new(0, 0, 64, 64),
            Rect::new(60, 40, 70, 60),
            Rect::new(full.out_w - 64, full.out_h - 50, 64, 50),
        ];
        // R6 to R8: the device stages read the half plane, and no map is made.
        let on_device: Vec<RgbImage> = rects
            .iter()
            .map(|&r| {
                full.run_tile_gpu(&frame, r, &route(DEVICE_RANGE))
                    .expect("device tile")
            })
            .collect();
        assert!(
            !full.sigma_map_built(),
            "the raw group of the device needs no sigma map"
        );
        assert_eq!(full.bytes(), full.cell_gains().map_or(0, |g| g.len() * 4));
        // The renderer's cached plane is shared, not copied.
        let shared = Arc::new(sigma.clone());
        let again = FullChain::new(&frame, &plan, &params, &shared, &CancelToken::new())
            .expect("full chain");
        assert!(Arc::ptr_eq(&again.sigma, &shared));
        // R6 alone: the CPU denoise, R7 and R8 read the map, which is made now.
        let hybrid: Vec<RgbImage> = rects
            .iter()
            .map(|&r| {
                full.run_tile_gpu(&frame, r, &route(StageRange::DEMOSAIC))
                    .expect("hybrid tile")
            })
            .collect();
        assert!(full.sigma_map_built());
        // The stages after the demosaic agree: same device demosaic, the CPU stages against the
        // device's (the kernels' bounds, `tests/gpu_chain.rs`).
        for (a, b) in on_device.iter().zip(&hybrid) {
            assert!(a.max_abs_diff(b) <= 1.1e-5, "{}", a.max_abs_diff(b));
        }
        // And all of it is the CPU chain's picture, up to the demosaic's decisions on exact ties.
        for (a, &r) in on_device.iter().zip(&rects) {
            let cpu = full.run_tile(&frame, r).expect("cpu tile");
            let (mut worst, mut sum) = (0.0f32, 0.0f64);
            for (x, y) in a.data.iter().zip(&cpu.data) {
                let d = (x - y).abs();
                worst = worst.max(d);
                sum += f64::from(d);
            }
            let mean = sum / a.data.len() as f64;
            assert!(
                mean < 2e-6 && worst < 0.05,
                "mean {mean:e}, worst {worst:e}"
            );
        }
    }
    #[test]
    fn the_front_range_runs_every_stage_from_the_resident_words_and_needs_no_sigma_map() {
        let Some(gpu) = device() else { return };
        let (w, h) = (200u32, 150u32);
        let (frame, plan, sigma, params) = full_chain_with_every_stage(w, h);
        let frame = Arc::new(frame);
        let mut route = GpuRoute {
            gpu: Arc::clone(&gpu),
            range: FRONT_RANGE,
            spec: JobSpec::new(ym_gpu::queue::Priority::Export),
            token: CancelToken::new(),
            resident: None,
        };
        let full = FullChain::new(&frame, &plan, &params, &sigma, &CancelToken::new())
            .expect("full chain");
        let rects = [
            Rect::new(0, 0, 64, 64),
            Rect::new(60, 40, 70, 60),
            Rect::new(full.out_w - 64, full.out_h - 50, 64, 50),
        ];
        // The range reads the photo's words on the device: without them it is refused.
        let refused = full.run_tile_gpu(&frame, rects[0], &route);
        assert!(
            matches!(refused, Err(GpuError::InvalidInput(_))),
            "{refused:?}"
        );
        route.resident = Some(route.upload(&frame).expect("upload"));
        let front: Vec<RgbImage> = rects
            .iter()
            .map(|&r| full.run_tile_gpu(&frame, r, &route).expect("device tile"))
            .collect();
        assert!(
            !full.sigma_map_built(),
            "R1 to R9 on the device needs no sigma map"
        );
        // The CPU chain's picture, up to the demosaic's decisions on exact ties.
        for (a, &r) in front.iter().zip(&rects) {
            let cpu = full.run_tile(&frame, r).expect("cpu tile");
            let (mut worst, mut sum) = (0.0f32, 0.0f64);
            for (x, y) in a.data.iter().zip(&cpu.data) {
                let d = (x - y).abs();
                worst = worst.max(d);
                sum += f64::from(d);
            }
            let mean = sum / a.data.len() as f64;
            assert!(
                mean < 2e-6 && worst < 0.05,
                "mean {mean:e}, worst {worst:e}"
            );
        }
        // A camera-processed file puts nothing on the device.
        let (_, mut reduced) = edge_frame(40, 30);
        reduced.reduced = true;
        assert!(route.upload(&Arc::new(reduced)).is_err());
    }

    /// A soft ramp with an overexposed block (the words past the white level, so R1 flags them and R5
    /// has work), noise on top.
    fn clipped_frame(w: u32, h: u32) -> (Arc<RawImage>, FrameData) {
        let mut data = vec![0u16; (w * h) as usize];
        let mut rng = synth::Rng::new(11);
        for y in 0..h {
            for x in 0..w {
                let (t, u) = (f64::from(x) / f64::from(w), f64::from(y) / f64::from(h));
                let mut v = 0.08 + 0.5 * t + 0.35 * u * u + 0.004 * f64::from(rng.gaussian());
                if x > 2 * w / 3 && y > h / 3 {
                    v += 0.9;
                }
                data[(y * w + x) as usize] =
                    (512.0 + v.clamp(0.0, 1.5) * 15788.0).min(16383.0) as u16;
            }
        }
        let bytes = ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", w, h)
            .black([512; 4])
            .white([16300; 3])
            .build(&data);
        let raw = Arc::new(
            ym_raw::decode::decode_bytes(&bytes, &ym_raw::decode::DecodeOptions::default())
                .expect("decode"),
        );
        let frame = FrameData::new(&raw).expect("frame");
        (raw, frame)
    }

    #[test]
    fn the_device_group_matches_reduced_bayer() {
        let Some(gpu) = device() else { return };
        let (w, h) = (240u32, 180u32);
        let (raw, frame) = clipped_frame(w, h);
        let mut params = params_for_tests();
        params.noise_reduction = NoiseReduction {
            luminance: 40.0,
            color: 60.0,
            detail: 70.0,
        };
        let mut plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        plan.vignette = Some(
            VignetteProfile::from_tag(&[0, -400, -1500, -3500, -6000]).expect("vignetting profile"),
        );
        let tags = LensTags {
            vignetting: vec![],
            distortion: vec![0, 120, 400, 900, 1600, 2500],
            ca: vec![0, 5000, 12000, 20000, 0, -5000, -12000, -20000],
        };
        plan.switches = LensSwitches::default();
        plan.lens = Some(
            LensGeometry::new(w as usize, h as usize, &tags, plan.switches)
                .expect("lens")
                .with_auto_crop(),
        );
        plan.tags = tags;
        assert!(plan.has_clipped, "the frame has words past white");
        let cancel = CancelToken::new();
        let sigma = sigma_plane(&frame, &plan, &params, &cancel).expect("sigma");
        let frame = Arc::new(frame);
        let route = GpuRoute {
            gpu: Arc::clone(&gpu),
            range: FRONT_RANGE,
            spec: JobSpec::new(ym_gpu::queue::Priority::Viewport),
            token: CancelToken::new(),
            resident: None,
        };
        let resident = route.upload(&frame).expect("upload");
        let mut worst = 0.0f32;
        for k in [2usize, 4, 8] {
            let want =
                reduced_bayer(&frame, &plan, &params, &sigma, k, &cancel).expect("cpu group");
            let device_params = reduced_params(&frame, &plan, &params, k)
                .expect("params")
                .expect("the device takes them");
            assert!(device_params.highlight_recovery && device_params.vignette.is_some());
            assert!(device_params.denoise.is_some() && device_params.lens.is_some());
            let got = route
                .reduced(&resident, device_params)
                .expect("device group");
            assert_eq!(
                (got.width, got.height),
                (want.width, want.height),
                "k = {k}"
            );
            let d = got.max_abs_diff(&want);
            eprintln!(
                "[G-07] reduced group k = {k}: {} x {}, device against reduced_bayer {d:e}",
                got.width, got.height
            );
            worst = worst.max(d);
        }
        // R5, the gain, the denoise and R9 together. First measured 9.5e-7 (k = 2; 8.3e-7 at k = 4,
        // 6.0e-7 at k = 8); twice that, the same on every adapter (the plan's number is 1e-5).
        assert!(worst <= 1.9e-6, "{worst:e}");
        // The parameters the device refuses give no group: an odd factor, one beyond the kernels'.
        assert!(
            reduced_params(&frame, &plan, &params, 3)
                .expect("params")
                .is_none()
        );
        assert!(
            reduced_params(&frame, &plan, &params, 18)
                .expect("params")
                .is_none()
        );
    }
}

#[cfg(test)]
mod ra03b_eval;

#[cfg(test)]
mod ra07_eval;
