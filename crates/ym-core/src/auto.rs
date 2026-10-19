//! `AutoEstimates`: the illuminant, exposure, white-balance readout, noise model and the Auto
//! sharpening and noise-reduction values of one photo, computed once from the decoded raw and
//! recomputed by "Re-run" (`docs/01-architecture.md` sections Session and DevelopSettings,
//! `docs/02-colour-pipeline.md` sections Auto exposure and Illuminant estimate, `docs/05` R7 and R8,
//! `docs/06-noise.md`).
//!
//! # What is computed, in order
//!
//! 1. **The 1/8 preview** ([`build_preview`]): the image area binned 8 x 8 per colour into
//!    black-subtracted camera RGB with white at 1.0, before any white balance, and a mask of the
//!    blocks that contain a clipped photosite. Lossless M and S files (camera RGB, no mosaic) are
//!    binned the same way, normalised with one range for the three channels (their white balance
//!    was divided out again, so red and blue have lower white levels; scaling each channel by its
//!    own would put the balance back). The preview is not kept: it is 11 MB for a 61 MP frame.
//! 2. **The scene light** ([`IlluminantResult`]): the trained estimator of `ym_color::illum` for the
//!    camera profile, with the camera's as-shot white balance as its prior. When the data folder
//!    holds no model for the profile ([`IlluminantModels`]) the **fallback** is the as-shot white
//!    balance itself, converted to a camera white (`[g_G / g_R, 1, g_G / g_B]`), with class
//!    probabilities from its CCT (tungsten below 3500 K, daylight above 4500 K, a linear blend in
//!    between, nothing for fluorescent and LED, which the as-shot value cannot tell apart) and
//!    confidence 0.5; a file with neither a model nor an as-shot white balance gets the profile's
//!    5500 K daylight white with confidence 0. The result is a [`SceneLight`] and the initial
//!    white `w_scene_initial` that the raw stages balance the mosaic with (`g0 = white_scale / w`; the
//!    white scale is `[1, 1, 1]` unless the body borrows a profile it does not match, `docs/03`). A body's
//!    white scale puts its raw white on the profile's scale before the estimator, the as-shot prior and
//!    Pick white see it ([`AutoEstimates::compute_scaled`]).
//! 3. **Auto exposure** (`ym_color::exposure`): the preview, balanced with `g0` and brightened by
//!    the vignetting gain of R3 when the file has a 0x7032 table ([`Preview::metered`]; the
//!    render applies it, so the exposure is metered on the picture that is rendered), through C1
//!    to C6 at EV 0, metered with the centre weight and the highlight guard. Burst sharing
//!    (step 6) needs the other photos, so the session does it ([`burst_frame`]) and stores the
//!    shared value in [`ExposureResult::ev`] (`own_ev` keeps the photo's own).
//!
//!    3b. **The white lift** (`ym_color::exposure::white`, `docs/02` C12 step 1b): the same preview, through
//!    the all-Auto chain at the photo's exposure and C7 at its defaults (`Rolloff::default()`: the
//!    highlight range at 1, so highlights the range already takes to white need less lift), sized for the
//!    reference look (Signature) by [`estimate_white`]: the amount `A` and the anchor, two scalars per
//!    photo ([`AutoEstimates::white`]). The user's own `rolloff` never re-sizes it. Estimated at the photo's own exposure; a burst median that replaces
//!    the exposure does not re-size it (with the constant camera gain that happens only for a frame
//!    the highlight guard lowered, by at most one stop). No burst rule: the estimate is a function of
//!    the photo's own preview, so the app and `ym` give the same two numbers for any set of photos.
//! 4. **The noise model** ([`NoiseResult`]): `ym_noise::estimate` on the image area with the prior
//!    of the body and ISO (`data/cameras/noise`), pooled across the session by the session
//!    (`pool_key`). A Lossless M or S file has no mosaic to measure and gets the prior alone.
//! 5. **Sharpening Auto** (`docs/05` R8) and **noise reduction Auto** (`docs/05` R7), from
//!    `ym_gpu::cpu` so the Automatic card and the render agree by construction.
//! 6. **Kelvin and tint readouts**: the Auto readout of the all-Auto, EV 0 chain.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use ym_color::chain::{ChainError, ColorEstimates, FusedChain, ProfileTables};
use ym_color::consts::Vec3;
use ym_color::exposure::white::{WhiteEstimate, estimate_white};
use ym_color::exposure::{
    AutoExposure, BurstFrame, ExposureError, ExposureSetting, PreviewRgb, ShotSettings,
    auto_exposure,
};
use ym_color::illum::{IllumError, IlluminantEstimate, IlluminantModel, RawPreview, ShotInfo};
use ym_color::look::LookTransform;
use ym_color::spectral::{ClassProbabilities, LightClass};
use ym_color::tone::ToneParams;
use ym_color::wb::{self, LightSource, SceneLight, WbError, WbReadout, WhiteBalance};
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::lens::{LensSwitches, LensTags};
use ym_gpu::cpu::sharpen::auto_sharpen;
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_noise::estimate::{
    BatchPool, EstimateInput, EstimatorConfig, NoiseEstimate, PlaneSource, PoolKey, U16Mosaic,
    estimate as estimate_noise_model,
};
use ym_noise::model::NoiseModel;
use ym_noise::priors::{CameraContext, NoisePrior, PriorStore};
use ym_profiles::profile::Profile;
use ym_profiles::registry::ProfileStore;
use ym_raw::decode::RawImage;
use ym_raw::layout::{CfaColor, DecodeKind, Layout, Rect};

use crate::apple_auto::{AppleAuto, AppleValues, AutoSteps};
use ym_raw::metadata::RawMetadata;

use crate::settings::{DevelopSettings, NoiseReductionValue, Rolloff, SharpenValue};

/// Preview block size: one preview pixel is the mean of this many photosites per side (1/8).
pub const PREVIEW_BLOCK_PX: usize = 8;

/// Normalised level at or above which a photosite counts as clipped when the decoder gave no
/// quantisation step (`docs/05` R1).
pub const CLIP_FRACTION: f32 = 0.995;

/// Confidence given to the as-shot fallback (the camera's own white balance, no evidence of ours).
pub const FALLBACK_CONFIDENCE: f64 = 0.5;

/// CCT at and below which the fallback calls the light tungsten (kelvin).
pub const FALLBACK_TUNGSTEN_BELOW_K: f64 = 3500.0;
/// CCT at and above which the fallback calls the light daylight (kelvin).
pub const FALLBACK_DAYLIGHT_ABOVE_K: f64 = 4500.0;
/// The daylight temperature of the fallback when the file has no white balance at all (kelvin).
pub const FALLBACK_DEFAULT_K: f64 = 5500.0;

/// Base ISO assumed for Auto sharpening when the gain table does not name one.
pub const DEFAULT_BASE_ISO: u32 = 100;

/// Why the automatic estimates could not be made. A photo with this error is skipped.
#[derive(Debug, thiserror::Error)]
pub enum AutoError {
    /// The decoded image cannot be analysed (too small, or levels that make no sense).
    #[error("This photo can't be analysed ({0}). Copy it from the card again, or skip it.")]
    BadImage(String),
    /// Auto exposure failed.
    #[error("This photo can't be analysed ({0}). Skip it and open the others.")]
    Exposure(#[from] ExposureError),
    /// The colour chain could not be built.
    #[error("This photo can't be analysed ({0}). Skip it and open the others.")]
    Chain(#[from] ChainError),
    /// A white was unusable.
    #[error("This photo can't be analysed ({0}). Skip it and open the others.")]
    White(#[from] WbError),
}

// ---------------------------------------------------------------------------------------------
// Illuminant models
// ---------------------------------------------------------------------------------------------

/// The illuminant estimator models found in the data folder, by camera profile id.
///
/// A profile without a model uses the as-shot fallback (see the module documentation), so a missing
/// model is not an error: [`IlluminantModels::missing`] lists the profiles that fall back.
#[derive(Debug, Clone, Default)]
pub struct IlluminantModels {
    models: HashMap<String, Arc<IlluminantModel>>,
    missing: Vec<String>,
    failed: Vec<(String, String)>,
}

impl IlluminantModels {
    /// No models: every photo uses the fallback.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Loads `<data_root>/models/illum/<profile-id>.json` for every profile of the store. A missing
    /// file is recorded in [`IlluminantModels::missing`]; a file that cannot be read or fails its
    /// SHA-256 check is recorded in [`IlluminantModels::failed`] with the reason, and that profile
    /// also falls back.
    #[must_use]
    pub fn load(data_root: &Path, profiles: &ProfileStore) -> Self {
        let mut out = Self::default();
        for id in profiles.profile_ids() {
            let id = id.as_str();
            let path = data_root.join("models/illum").join(format!("{id}.json"));
            if !path.exists() {
                out.missing.push(id.to_owned());
                continue;
            }
            match IlluminantModel::load_for_profile(data_root, id) {
                Ok(model) => {
                    out.models.insert(id.to_owned(), Arc::new(model));
                }
                Err(e) => {
                    tracing::warn!(profile = id, error = %e, "illuminant model not usable; using the as-shot fallback");
                    out.failed.push((id.to_owned(), e.to_string()));
                }
            }
        }
        out
    }

    /// Adds or replaces the model of a profile (tests, tools).
    pub fn insert(&mut self, profile_id: &str, model: IlluminantModel) {
        self.missing.retain(|m| m != profile_id);
        self.models.insert(profile_id.to_owned(), Arc::new(model));
    }

    /// The model of a profile, if there is one.
    #[must_use]
    pub fn get(&self, profile_id: &str) -> Option<&Arc<IlluminantModel>> {
        self.models.get(profile_id)
    }

    /// Profile ids that have no model file.
    #[must_use]
    pub fn missing(&self) -> &[String] {
        &self.missing
    }

    /// Profile ids whose model file exists but could not be used, with the reason.
    #[must_use]
    pub fn failed(&self) -> &[(String, String)] {
        &self.failed
    }

    /// Number of models loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// True when no model is loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------
// Image geometry
// ---------------------------------------------------------------------------------------------

/// The part of the readout that is the photo: the camera's crop, else the active area, else the
/// whole readout (in photosites, readout coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageArea {
    /// Left edge.
    pub x: usize,
    /// Top edge.
    pub y: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

/// The photo's area inside the decoded data. Lossless M and S files (camera RGB) use the whole
/// frame.
#[must_use]
pub fn image_area(raw: &RawImage) -> ImageArea {
    let (full_w, full_h) = (raw.pixel_width(), raw.pixel_height());
    match raw.layout {
        Layout::Bayer {
            active_area, crop, ..
        } => image_area_of(full_w, full_h, active_area, crop),
        Layout::Ycbcr { .. } => ImageArea {
            x: 0,
            y: 0,
            width: full_w,
            height: full_h,
        },
    }
}

/// [`image_area`] from the numbers the probe reports: the readout size, the active area and the
/// camera's crop. A rectangle that does not fit in the readout is ignored.
#[must_use]
pub fn image_area_of(
    width: usize,
    height: usize,
    active_area: Option<Rect>,
    crop: Option<Rect>,
) -> ImageArea {
    let usable = |r: Option<Rect>| {
        r.filter(|r| r.fits_in(width as u32, height as u32))
            .map(|r| ImageArea {
                x: r.x as usize,
                y: r.y as usize,
                width: r.width as usize,
                height: r.height as usize,
            })
    };
    usable(crop).or(usable(active_area)).unwrap_or(ImageArea {
        x: 0,
        y: 0,
        width,
        height,
    })
}

/// The size of the photo in pixels as the camera oriented it, from the probe alone.
#[must_use]
pub fn meta_image_size(meta: &RawMetadata) -> (usize, usize) {
    let reduced = matches!(meta.decode, DecodeKind::LosslessMS { .. });
    let area = if reduced {
        image_area_of(meta.width as usize, meta.height as usize, None, None)
    } else {
        image_area_of(
            meta.width as usize,
            meta.height as usize,
            meta.active_area,
            meta.crop,
        )
    };
    oriented_size(area.width, area.height, meta.orientation)
}

/// Width and height of the photo as the camera oriented it (EXIF orientation 5 to 8 swap them).
#[must_use]
pub fn oriented_size(
    area_width: usize,
    area_height: usize,
    exif_orientation: u16,
) -> (usize, usize) {
    if (5..=8).contains(&exif_orientation) {
        (area_height, area_width)
    } else {
        (area_width, area_height)
    }
}

/// Maps a point of the photo as the camera oriented it (EXIF orientation applied) to the same spot
/// on the sensor, both as fractions of the image area in [0, 1]. `white_pick` is stored in the
/// first space (`DevelopSettings::white_pick`).
///
/// Orientations: 1 upright, 2 mirrored, 3 turned 180 degrees, 4 flipped, 5 transposed, 6 rotated
/// 90 degrees clockwise to display, 7 transversed, 8 rotated 90 degrees counter-clockwise. A value
/// outside 1 to 8 is treated as 1.
#[must_use]
pub fn display_to_sensor(exif_orientation: u16, [u, v]: [f64; 2]) -> [f64; 2] {
    match exif_orientation {
        2 => [1.0 - u, v],
        3 => [1.0 - u, 1.0 - v],
        4 => [u, 1.0 - v],
        5 => [v, u],
        6 => [v, 1.0 - u],
        7 => [1.0 - v, 1.0 - u],
        8 => [1.0 - v, u],
        _ => [u, v],
    }
}

// ---------------------------------------------------------------------------------------------
// The 1/8 preview
// ---------------------------------------------------------------------------------------------

/// The 1/8-resolution preview: black-subtracted camera RGB, white at 1.0, before white balance.
#[derive(Debug, Clone)]
pub struct Preview {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` values, `R G B` per pixel.
    pub rgb: Vec<f32>,
    /// One byte per pixel, 1 where a photosite of the block is clipped.
    pub clip: Vec<u8>,
    /// The R3 gain `1 / G(r)` at the centre of each pixel's block (`docs/05` R3), `None` for a
    /// file without a usable vignetting table and for Lossless M and S files, which the render
    /// does not correct either. `rgb` is **without** it: the illuminant estimator reads the
    /// preview as the sensor saw it ([`Preview::as_raw_preview`]); only the metering applies it
    /// ([`Preview::metered`]).
    pub vignette_gain: Option<Vec<f32>>,
}

impl Preview {
    /// The preview as the illuminant estimator reads it.
    #[must_use]
    pub fn as_raw_preview(&self) -> RawPreview<'_> {
        RawPreview {
            width: self.width,
            height: self.height,
            rgb: &self.rgb,
            clip_mask: Some(&self.clip),
        }
    }

    /// The RGB balanced with the initial gains `g0` (a copy), without the vignetting gain.
    #[must_use]
    pub fn balanced(&self, g0: Vec3) -> Vec<f32> {
        let g = [g0[0] as f32, g0[1] as f32, g0[2] as f32];
        self.rgb
            .chunks_exact(3)
            .flat_map(|p| [p[0] * g[0], p[1] * g[1], p[2] * g[2]])
            .collect()
    }

    /// The RGB as auto exposure meters it: the vignetting gain of each block (R3, the same for the
    /// three channels), then the initial gains `g0` (R4), in the order of the raw stages. For a
    /// file without a vignetting table it is [`Preview::balanced`].
    ///
    /// The render divides every photosite by the lens's relative illumination `G(r)`, so its
    /// corners are brighter than the 1/8 preview's. Metering the preview without the gain read
    /// lens-corrected files 0.07 to 0.12 EV darker than their render (`docs/02` Auto exposure,
    /// Evaluation); the gain at the block centre is within 1e-4 of the mean over the block.
    #[must_use]
    pub fn metered(&self, g0: Vec3) -> Vec<f32> {
        let Some(gain) = &self.vignette_gain else {
            return self.balanced(g0);
        };
        let g = [g0[0] as f32, g0[1] as f32, g0[2] as f32];
        self.rgb
            .chunks_exact(3)
            .zip(gain)
            .flat_map(|(p, &v)| [p[0] * v * g[0], p[1] * v * g[1], p[2] * v * g[2]])
            .collect()
    }
}

/// The R3 gain `1 / G(r)` at the centre of each block of a `pw` x `ph` preview of `raw`'s image
/// area, `None` when the file has no vignetting table the render would use: none stored, a table
/// the render ignores, or a camera-processed file (Lossless M and S).
///
/// The radius is the one of `VignetteProfile::radius`: from the centre of the image area to the
/// centre of the block, over half the area's diagonal. It does not depend on the
/// `decode.lens_corrections` switch: the estimates are made when the photo is opened, before any
/// setting is changed (the CLI sets its options after the open too), and describe the default
/// render, which corrects the lens.
fn vignette_gains(raw: &RawImage, area: &ImageArea, pw: usize, ph: usize) -> Option<Vec<f32>> {
    if raw.is_reduced() {
        return None;
    }
    let stored = raw
        .lens
        .as_ref()
        .or(raw.meta.lens_corrections.as_ref())?
        .vignetting
        .as_ref()?;
    let tags = LensTags::from_stored(Some(stored.raw.as_slice()), None, None).ok()?;
    let profile = VignetteProfile::from_tags(&tags, LensSwitches::default()).ok()??;
    let (w, h) = (area.width as f64, area.height as f64);
    let inv_half_diagonal = 1.0 / (0.5 * w.hypot(h));
    // The mean of the photosite centres `x + 0.5` over the block `[xs, xe)` is `(xs + xe) / 2`.
    let centre = |i: usize, n: usize, len: usize| (i * len / n + (i + 1) * len / n) as f64 * 0.5;
    let mut gains = Vec::with_capacity(pw * ph);
    for by in 0..ph {
        let dy = centre(by, ph, area.height) - 0.5 * h;
        for bx in 0..pw {
            let dx = centre(bx, pw, area.width) - 0.5 * w;
            gains.push(profile.gain_at(dx.hypot(dy) * inv_half_diagonal) as f32);
        }
    }
    Some(gains)
}

/// Per-photosite normalisation of a raw image: `(raw - black) / (white - black)` per CFA position,
/// with the range of a camera-processed file taken from its green channel.
struct Levels {
    black: [f32; 4],
    inv_range: [f32; 4],
    /// Raw value at or above which a photosite is clipped when there is no `qstep` plane.
    clip_raw: [f32; 4],
    white: [f32; 4],
}

impl Levels {
    fn new(raw: &RawImage) -> Result<Self, AutoError> {
        // Colour order of the levels differs: CFA raster order for a mosaic, R G G B for the
        // camera-RGB layout (see `ym_raw::decode::RawImage`). Both are indexed by `p` below.
        let (black, white) = (raw.black, raw.white);
        let mut range = [0.0f32; 4];
        for p in 0..4 {
            range[p] = white[p] - black[p];
            if !(range[p].is_finite() && range[p] > 0.0 && black[p].is_finite()) {
                return Err(AutoError::BadImage(format!(
                    "white level {} is not above black level {}",
                    white[p], black[p]
                )));
            }
        }
        // A mosaic saturates at the same raw level in every colour, so each position is scaled
        // by its own range. The camera-RGB layout of Lossless M and S is different: its white
        // balance was divided out again, which lowers the ceiling of red and blue by their gains
        // (`docs/04`). Scaling each channel by its own range would put the camera's balance
        // back (a neutral surface would read equal in all three), and the initial gains would
        // balance it a second time. One range for the three channels keeps the data in the raw
        // scale of a mosaic: channel `c` then clips at `1 / g_c`, as `docs/02` C1 says.
        let shared = range.iter().copied().fold(0.0, f32::max);
        let reduced = matches!(raw.layout, Layout::Ycbcr { .. });
        let inv_range = std::array::from_fn(|p| 1.0 / if reduced { shared } else { range[p] });
        let clip_raw = std::array::from_fn(|p| black[p] + CLIP_FRACTION * range[p]);
        Ok(Self {
            black,
            inv_range,
            clip_raw,
            white,
        })
    }
}

/// Bins the photo's area 8 x 8 per colour (see the module documentation).
///
/// # Errors
///
/// [`AutoError::BadImage`] for an image smaller than 2 x 2 pixels or levels with no range.
pub fn build_preview(raw: &RawImage) -> Result<Preview, AutoError> {
    let area = image_area(raw);
    if area.width < 2 || area.height < 2 {
        return Err(AutoError::BadImage("the image is too small".to_owned()));
    }
    let levels = Levels::new(raw)?;
    let pw = (area.width / PREVIEW_BLOCK_PX).max(1);
    let ph = (area.height / PREVIEW_BLOCK_PX).max(1);
    let mut rgb = vec![0.0f32; pw * ph * 3];
    let mut clip = vec![0u8; pw * ph];
    let stride = raw.data.width;
    let data = &raw.data.data;
    let qstep = raw.qstep.as_ref();
    match raw.layout {
        Layout::Bayer { pattern, .. } => {
            let color_of_pos: [usize; 4] = std::array::from_fn(|p| match pattern.0[p] {
                CfaColor::Red => 0,
                CfaColor::Green => 1,
                CfaColor::Blue => 2,
            });
            rgb.par_chunks_mut(pw * 3)
                .zip(clip.par_chunks_mut(pw))
                .enumerate()
                .for_each(|(by, (rgb_row, clip_row))| {
                    let ys = area.y + by * area.height / ph;
                    let ye = area.y + (by + 1) * area.height / ph;
                    for bx in 0..pw {
                        let xs = area.x + bx * area.width / pw;
                        let xe = area.x + (bx + 1) * area.width / pw;
                        let mut sum = [0.0f32; 3];
                        let mut count = [0u32; 3];
                        let mut clipped = false;
                        for y in ys..ye {
                            let row = &data[y * stride + xs..y * stride + xe];
                            let q_row = qstep.map(|q| &q.data[y * stride + xs..y * stride + xe]);
                            for (i, &v) in row.iter().enumerate() {
                                let x = xs + i;
                                let p = ((y & 1) << 1) | (x & 1);
                                let v = f32::from(v);
                                let c = color_of_pos[p];
                                sum[c] += (v - levels.black[p]) * levels.inv_range[p];
                                count[c] += 1;
                                clipped |= match q_row {
                                    Some(q) => v >= levels.white[p] - 4.0 * q[i].to_f32(),
                                    None => v >= levels.clip_raw[p],
                                };
                            }
                        }
                        for c in 0..3 {
                            rgb_row[3 * bx + c] = sum[c] / count[c].max(1) as f32;
                        }
                        clip_row[bx] = u8::from(clipped);
                    }
                });
        }
        Layout::Ycbcr { .. } => {
            // Camera RGB, three samples per pixel; levels are ordered R, G, G, B.
            let channel_pos = [0usize, 1, 3];
            rgb.par_chunks_mut(pw * 3)
                .zip(clip.par_chunks_mut(pw))
                .enumerate()
                .for_each(|(by, (rgb_row, clip_row))| {
                    let ys = area.y + by * area.height / ph;
                    let ye = area.y + (by + 1) * area.height / ph;
                    for bx in 0..pw {
                        let xs = area.x + bx * area.width / pw;
                        let xe = area.x + (bx + 1) * area.width / pw;
                        let mut sum = [0.0f32; 3];
                        let mut clipped = false;
                        for y in ys..ye {
                            let row = &data[y * stride + 3 * xs..y * stride + 3 * xe];
                            for px in row.chunks_exact(3) {
                                for c in 0..3 {
                                    let p = channel_pos[c];
                                    let v = f32::from(px[c]);
                                    sum[c] += (v - levels.black[p]) * levels.inv_range[p];
                                    clipped |= v >= levels.clip_raw[p];
                                }
                            }
                        }
                        let n = ((ye - ys) * (xe - xs)).max(1) as f32;
                        for c in 0..3 {
                            rgb_row[3 * bx + c] = sum[c] / n;
                        }
                        clip_row[bx] = u8::from(clipped);
                    }
                });
        }
    }
    let vignette_gain = vignette_gains(raw, &area, pw, ph);
    Ok(Preview {
        width: pw,
        height: ph,
        rgb,
        clip,
        vignette_gain,
    })
}

// ---------------------------------------------------------------------------------------------
// Illuminant
// ---------------------------------------------------------------------------------------------

/// Where the scene light came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IlluminantSource {
    /// The trained estimator for the profile; `model_version` is the model file's version.
    Model {
        /// The model's version string (goes into the export metadata).
        model_version: String,
        /// Whether the camera's as-shot white balance entered as the prior.
        prior_used: bool,
    },
    /// No model for this profile: the camera's as-shot white balance, converted (see the module
    /// documentation).
    AsShotFallback,
    /// No model and no as-shot white balance: the profile's 5500 K daylight white.
    DefaultDaylight,
}

/// The scene light and how it was found.
#[derive(Debug, Clone)]
pub struct IlluminantResult {
    /// How the light was found.
    pub source: IlluminantSource,
    /// The estimator's raw output when a model ran.
    pub estimate: Option<IlluminantEstimate>,
    /// The scene light the colour chain starts from.
    pub scene: SceneLight,
}

/// The as-shot gains of a decoded image in Sony's R, G1, G2, B order.
///
/// `RawImage::wb_as_shot` is stored **per CFA position** (`docs/04`); for the RGGB sensors of
/// every Sony body that is already R, G1, G2, B, but a mosaic with another phase (the decoder reads
/// BGGR, GRBG and GBRG too) must be read by colour or red and blue swap. G1 is the green that
/// shares a row with red. A camera-RGB layout (Lossless M and S) is stored in R, G, G, B order.
#[must_use]
pub fn as_shot_rggb(raw: &RawImage) -> Option<[f32; 4]> {
    let wb = raw.wb_as_shot?;
    let Layout::Bayer { pattern, .. } = raw.layout else {
        return Some(wb);
    };
    let at = |color: CfaColor| pattern.0.iter().position(|c| *c == color);
    let (Some(r), Some(b)) = (at(CfaColor::Red), at(CfaColor::Blue)) else {
        return Some(wb);
    };
    // The green next to red (same row) and the green next to blue (same row).
    let beside = |p: usize| (p / 2) * 2 + (1 - p % 2);
    Some([wb[r], wb[beside(r)], wb[beside(b)], wb[b]])
}

/// The camera's as-shot white as a camera white (green = 1): `[g_G / g_R, 1, g_G / g_B]` from the
/// stored levels in R, G1, G2, B order ([`as_shot_rggb`]), `None` when they are missing or not
/// usable.
#[must_use]
pub fn as_shot_white(wb_rggb: Option<[f32; 4]>) -> Option<Vec3> {
    as_shot_white_scaled(wb_rggb, wb::IDENTITY_WHITE_SCALE)
}

/// [`as_shot_white`] on the profile's scale: the body's own white times its white scale
/// (`docs/03`, White scale of stand-in bodies), green = 1. `None` when the levels or the scale are not
/// usable.
#[must_use]
pub fn as_shot_white_scaled(wb_rggb: Option<[f32; 4]>, white_scale: Vec3) -> Option<Vec3> {
    let gains = shot_gains(wb_rggb)?;
    wb::scale_white(
        [1.0 / gains[0], 1.0 / gains[1], 1.0 / gains[2]],
        white_scale,
    )
    .ok()
}

/// The camera's as-shot white balance as gains `[R/G, 1, B/G]` on the body's own raw scale, without any
/// white scale: what the camera's default rendering multiplies the raw by (a white surface records
/// `1 / gains`). `None` when the levels are missing or not usable. The Before view's camera default uses
/// it ([`crate::pipeline::color::color_plan`]).
#[must_use]
pub fn as_shot_gains(wb_rggb: Option<[f32; 4]>) -> Option<Vec3> {
    let g = shot_gains(wb_rggb)?;
    Some([g[0] / g[1], 1.0, g[2] / g[1]])
}

/// The as-shot gains `[R, G, B]` (green is the mean of the two greens), `None` when unusable.
fn shot_gains(wb_rggb: Option<[f32; 4]>) -> Option<Vec3> {
    let [r, g1, g2, b] = wb_rggb?;
    let gains = [
        f64::from(r),
        0.5 * (f64::from(g1) + f64::from(g2)),
        f64::from(b),
    ];
    gains
        .iter()
        .all(|g| g.is_finite() && *g > 0.0)
        .then_some(gains)
}

/// As-shot gains on the profile's scale: a white surface records `1 / g`, and the profile's white is that times
/// `white_scale`, so the gains are divided by it (red and blue; green is 1).
fn shot_gains_scaled(gains: Vec3, white_scale: Vec3) -> Vec3 {
    [
        gains[0] / white_scale[0],
        gains[1] / white_scale[1],
        gains[2] / white_scale[2],
    ]
}

/// Class probabilities of the as-shot fallback from the CCT alone.
fn fallback_classes(cct_k: f64) -> ClassProbabilities {
    let t = ((cct_k - FALLBACK_TUNGSTEN_BELOW_K)
        / (FALLBACK_DAYLIGHT_ABOVE_K - FALLBACK_TUNGSTEN_BELOW_K))
        .clamp(0.0, 1.0);
    // The two weights sum to 1, so this cannot fail; the daylight-only value is the safe default.
    ClassProbabilities::new(t, 1.0 - t, 0.0, 0.0)
        .unwrap_or_else(|_| ClassProbabilities::only(LightClass::Daylight))
}

fn estimate_light(
    model: Option<&IlluminantModel>,
    preview: &Preview,
    meta: &RawMetadata,
    wb_rggb: Option<[f32; 4]>,
    profile: &Profile,
    white_scale: Vec3,
) -> Result<IlluminantResult, AutoError> {
    let locus = &profile.locus;
    if let Some(model) = model {
        let shot = ShotInfo {
            as_shot_gains: shot_gains(wb_rggb).map(|g| shot_gains_scaled(g, white_scale)),
            f_number: meta.f_number,
            exposure_time_s: meta.exposure_time_s,
            iso: meta.iso.map(f64::from),
        };
        // The estimator reads the preview on the profile's scale (`docs/03`, White scale of stand-in
        // bodies): red and blue of the body's raw times the white scale. The identity reads the preview as is.
        let scaled_rgb = (white_scale != wb::IDENTITY_WHITE_SCALE).then(|| {
            let s = [
                white_scale[0] as f32,
                white_scale[1] as f32,
                white_scale[2] as f32,
            ];
            preview
                .rgb
                .chunks_exact(3)
                .flat_map(|p| [p[0] * s[0], p[1] * s[1], p[2] * s[2]])
                .collect::<Vec<f32>>()
        });
        let raw_preview = match &scaled_rgb {
            Some(rgb) => RawPreview {
                rgb,
                ..preview.as_raw_preview()
            },
            None => preview.as_raw_preview(),
        };
        match model
            .estimate(&raw_preview, &shot)
            .map_err(|e: IllumError| e.to_string())
            .and_then(|est| {
                est.scene_light(locus)
                    .map(|scene| (est, scene))
                    .map_err(|e| e.to_string())
            }) {
            Ok((estimate, scene)) => {
                return Ok(IlluminantResult {
                    source: IlluminantSource::Model {
                        model_version: model.version().to_owned(),
                        prior_used: estimate.prior_used,
                    },
                    estimate: Some(estimate),
                    scene,
                });
            }
            Err(reason) => {
                tracing::warn!(%reason, "illuminant estimate failed; using the as-shot fallback");
            }
        }
    }
    let (white, source, confidence) = match as_shot_white_scaled(wb_rggb, white_scale) {
        Some(w) => (w, IlluminantSource::AsShotFallback, FALLBACK_CONFIDENCE),
        None => (
            locus.rgb_at(FALLBACK_DEFAULT_K),
            IlluminantSource::DefaultDaylight,
            0.0,
        ),
    };
    let readout = locus
        .rgb_to_cct_duv(white)
        .map_err(|e| AutoError::White(WbError::Spectral(e)))?;
    let classes = fallback_classes(readout.cct_k);
    let scene = SceneLight::from_parts(white, classes, confidence, readout.cct_k, readout.duv)?;
    Ok(IlluminantResult {
        source,
        estimate: None,
        scene,
    })
}

// ---------------------------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------------------------

/// Where the noise model came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSource {
    /// Measured on this photo.
    Image,
    /// Measured on this photo and pooled with `files` photos of the same body, ISO, bit depth and
    /// compression in the session.
    Pooled {
        /// Photos in the group, this one included.
        files: usize,
    },
    /// Nothing could be measured (a Lossless M or S file, or a frame the estimator refused): the
    /// centre of the prior for the body and ISO.
    PriorOnly,
}

/// The noise model of a photo, in normalised units (after R1, before white balance).
#[derive(Debug, Clone)]
pub struct NoiseResult {
    /// The model: `var(mu) = a (mu - delta) + b + sigma_row^2 + q^2 / 12`.
    pub model: NoiseModel,
    /// The estimator's full result (variance points, diagnostics) when it ran on this photo; the
    /// session pools these.
    pub estimate: Option<Arc<NoiseEstimate>>,
    /// Where the model came from.
    pub source: NoiseSource,
}

/// The key under which photos pool their noise estimates (`docs/06`): model, ISO, bit depth and
/// compression.
#[must_use]
pub fn pool_key(meta: &RawMetadata) -> PoolKey {
    PoolKey {
        model: meta.model.clone(),
        iso: meta.iso.unwrap_or(0),
        bit_depth: match meta.decode {
            DecodeKind::Uncompressed12 | DecodeKind::Arw1 => 12,
            _ => 14,
        },
        compression: meta.decode.id().to_owned(),
    }
}

fn camera_context(meta: &RawMetadata, assume_iso: bool) -> CameraContext {
    CameraContext {
        model_id: meta.model_id.map(u32::from),
        model: Some(meta.model.clone()),
        iso: meta.iso.or(assume_iso.then_some(100)),
        class: None,
    }
}

/// Reads the quantisation steps of a decoded image as a rectangle source, normalised.
struct QstepSource<'a> {
    plane: &'a ym_raw::layout::Plane<half::f16>,
    origin: (usize, usize),
    width: usize,
    height: usize,
    /// `1 / (white - black)` per CFA position of the area's origin.
    scale: [f32; 4],
}

impl PlaneSource for QstepSource<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn read_rect(&self, x0: usize, y0: usize, w: usize, h: usize, out: &mut [f32]) {
        for r in 0..h {
            let src = &self.plane.data
                [(self.origin.1 + y0 + r) * self.plane.width + self.origin.0 + x0..][..w];
            for (c, (o, v)) in out[r * w..(r + 1) * w].iter_mut().zip(src).enumerate() {
                let p = (((y0 + r) & 1) << 1) | ((x0 + c) & 1);
                *o = v.to_f32() * self.scale[p];
            }
        }
    }
}

/// The prior-only model: the centre of the body's prior, quantisation of one DN.
fn prior_only_model(prior: &NoisePrior, q_norm: f64) -> Option<NoiseModel> {
    let a = prior.a_center()?;
    Some(NoiseModel::new(a, prior.b_center().unwrap_or(0.0)).with_quantisation(q_norm))
}

/// The per-photo noise estimate (`docs/06` Estimator), without pooling.
fn estimate_noise(raw: &RawImage, priors: &PriorStore) -> Result<NoiseResult, AutoError> {
    let meta = &raw.meta;
    // One DN in normalised units, as the estimator's `q_norm` (the widest level range).
    let widest = (0..4)
        .map(|p| raw.white[p] - raw.black[p])
        .fold(1.0f32, f32::max);
    let q_norm_default = 1.0 / f64::from(widest);
    let ctx = camera_context(meta, false);
    let prior = priors.prior_for(&ctx);
    let fallback = |prior: &NoisePrior, q: f64| -> Result<NoiseResult, AutoError> {
        // With no ISO there is no prior; ask for the class prior of ISO 100 rather than leave the
        // photo without a noise model.
        let prior = if prior.a_center().is_some() {
            prior.clone()
        } else {
            priors.prior_for(&camera_context(meta, true))
        };
        let model = prior_only_model(&prior, q_norm_default.max(q)).ok_or_else(|| {
            AutoError::BadImage("no noise prior is available for this camera".to_owned())
        })?;
        Ok(NoiseResult {
            model,
            estimate: None,
            source: NoiseSource::PriorOnly,
        })
    };
    if raw.is_reduced() {
        return fallback(&prior, 0.0);
    }
    let area = image_area(raw);
    let full_w = raw.data.width;
    // Black and white are per raster position of the readout; rotate them to the position of the
    // area's origin (the identity when it is even).
    let rotate = |v: [f32; 4]| -> [f32; 4] {
        std::array::from_fn(|p| v[(((area.y + (p >> 1)) & 1) << 1) | ((area.x + (p & 1)) & 1)])
    };
    let (black, white) = (rotate(raw.black), rotate(raw.white));
    let data = &raw.data.data[area.y * full_w + area.x..];
    let mosaic = match U16Mosaic::with_stride(data, area.width, area.height, full_w, black, white) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "noise estimate skipped");
            return fallback(&prior, 0.0);
        }
    };
    let scale: [f32; 4] = std::array::from_fn(|p| 1.0 / (white[p] - black[p]));
    let qstep = raw.qstep.as_ref().map(|plane| QstepSource {
        plane,
        origin: (area.x, area.y),
        width: area.width,
        height: area.height,
        scale,
    });
    let q_norm = f64::from(scale.iter().copied().fold(f32::MAX, f32::min));
    let input = EstimateInput {
        mosaic: &mosaic,
        qstep: qstep.as_ref().map(|q| q as &dyn PlaneSource),
        q_norm,
    };
    match estimate_noise_model(&input, &prior, &EstimatorConfig::default()) {
        Ok(est) => Ok(NoiseResult {
            model: est.model,
            estimate: Some(Arc::new(est)),
            source: NoiseSource::Image,
        }),
        Err(e) => {
            tracing::warn!(error = %e, "noise estimate failed; using the prior");
            fallback(&prior, q_norm)
        }
    }
}

/// The most photos whose estimates are merged into one pooled model. More files add little (the
/// statistics saturate after a handful) and every addition costs a refit.
pub const POOL_MAX_FILES: usize = 16;

/// The signal levels (normalised) at which a pooled model is compared with its members' own.
const AGREEMENT_LEVELS: [f64; 3] = [0.02, 0.1, 0.4];

/// A pooled model is used only if its variance at each of three signal levels (0.02, 0.1 and 0.4
/// of white) is within this factor of the median of the members' own variances there.
///
/// This is the last word against a pool that contradicts its members, not the thing that makes
/// pooling work. Three frames of one scene used to pool to 1.77e-4 for a true 5e-4 (own fits
/// 5.0 to 5.1e-4) because `ym_noise::BatchPool` divided the relative variance of a merged bin by
/// the number of files although the scatter of a bin about the noise line is the scene's and
/// repeats from frame to frame; that is fixed there (a merged bin is as precise as its best member
/// and no more), and the same frames now pool to their own fit with this check passing. It stays for
/// what pooling cannot know: members whose own models were adjusted after their points were
/// measured (the filtered-file factor of a group that mixes filtered and plain frames), or files
/// that share a body and an ISO but not the in-camera processing behind the noise.
pub const POOL_AGREEMENT_FACTOR: f64 = 1.6;

/// A pool for [`pool_group`] and for the session's per-group pools.
#[derive(Debug, Clone, Default)]
pub struct NoisePool {
    pool: BatchPool,
    files: usize,
    /// The median over the members of their own variance at [`AGREEMENT_LEVELS`].
    consensus: Option<[f64; 3]>,
}

impl NoisePool {
    /// Builds a pool from the estimates of a group (same [`pool_key`]), taking the first
    /// [`POOL_MAX_FILES`] in the order given.
    ///
    /// The members are added in the order given, so the result depends only on the set and order
    /// of photos, never on which one finished decoding first.
    #[must_use]
    pub fn build<'a>(
        key: &PoolKey,
        members: impl IntoIterator<Item = (&'a NoiseResult, &'a NoisePrior)>,
    ) -> Self {
        let mut pool = BatchPool::new();
        let mut variances: [Vec<f64>; 3] = Default::default();
        for (own, prior) in members.into_iter().take(POOL_MAX_FILES) {
            if let Some(est) = &own.estimate {
                pool.add(key.clone(), est, prior);
                for (slot, mu) in variances.iter_mut().zip(AGREEMENT_LEVELS) {
                    slot.push(own.model.pixel_variance(mu, 0));
                }
            }
        }
        let files = pool.files(key);
        let consensus = (files > 1).then(|| {
            variances.map(|mut v| {
                v.sort_by(f64::total_cmp);
                let n = v.len();
                if n % 2 == 1 {
                    v[n / 2]
                } else {
                    0.5 * (v[n / 2 - 1] + v[n / 2])
                }
            })
        });
        Self {
            pool,
            files,
            consensus,
        }
    }

    /// Photos in the pool.
    #[must_use]
    pub fn files(&self) -> usize {
        self.files
    }

    /// Whether a pooled model agrees with the members' own estimates (see
    /// [`POOL_AGREEMENT_FACTOR`]).
    fn agrees(&self, model: &NoiseModel) -> bool {
        let Some(consensus) = self.consensus else {
            return true;
        };
        AGREEMENT_LEVELS.iter().zip(consensus).all(|(mu, want)| {
            let got = model.pixel_variance(*mu, 0);
            let ratio = got / want;
            (1.0 / POOL_AGREEMENT_FACTOR..=POOL_AGREEMENT_FACTOR).contains(&ratio)
        })
    }

    /// The pooled result for one photo: the pool's fit with the photo's own filtered-file factor.
    /// A photo without an image-based estimate keeps what it has, and so does every photo when
    /// the pooled fit disagrees with its members ([`POOL_AGREEMENT_FACTOR`]) or the group has one
    /// file only.
    pub fn result_for(
        &mut self,
        key: &PoolKey,
        own: &NoiseResult,
        prior: &NoisePrior,
    ) -> NoiseResult {
        let Some(est) = &own.estimate else {
            return own.clone();
        };
        if self.files <= 1 {
            return own.clone();
        }
        // An estimate marked prior-only adds nothing to the pool and returns the group's fit with
        // this file's own filtered factor.
        let mut probe = (**est).clone();
        probe.points.clear();
        probe.diagnostics.confidence = ym_noise::estimate::Confidence::PriorOnly;
        match self.pool.add(key.clone(), &probe, prior) {
            Some(pooled) if self.agrees(&pooled.model) => NoiseResult {
                model: pooled.model,
                estimate: own.estimate.clone(),
                source: NoiseSource::Pooled { files: self.files },
            },
            Some(pooled) => {
                tracing::warn!(
                    pooled_a = pooled.model.a,
                    own_a = own.model.a,
                    files = self.files,
                    "the pooled noise fit disagrees with its members; keeping the file's own estimate"
                );
                own.clone()
            }
            None => own.clone(),
        }
    }
}

/// Pools the noise estimates of a group of photos (same [`pool_key`]) and returns each member's
/// pooled result, in the order given (see [`NoisePool`]).
#[must_use]
pub fn pool_group(key: &PoolKey, members: &[(&NoiseResult, &NoisePrior)]) -> Vec<NoiseResult> {
    let mut pool = NoisePool::build(key, members.iter().copied());
    members
        .iter()
        .map(|(own, prior)| pool.result_for(key, own, prior))
        .collect()
}

/// The prior for a photo, for pooling.
#[must_use]
pub fn noise_prior(priors: &PriorStore, meta: &RawMetadata) -> NoisePrior {
    priors.prior_for(&camera_context(meta, false))
}

// ---------------------------------------------------------------------------------------------
// Exposure
// ---------------------------------------------------------------------------------------------

/// The automatic exposure of a photo: the camera's own exposure (`ym_color::exposure::CAMERA_GAIN_EV`, `docs/02`
/// Auto exposure), lowered by the highlight guard when the picture would clip, and shared across a burst.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureResult {
    /// `EV_auto` in effect: the burst median once the session has shared it, else `own_ev`.
    pub ev: f64,
    /// `EV_auto` of this photo alone, after the highlight guard.
    pub own_ev: f64,
    /// The details (`ev_before_guard`, the guard's reduction, the reported key statistic and pixel counts).
    pub meter: AutoExposure,
}

/// What burst grouping needs to know about a photo (`docs/02` Auto exposure, step 6): the camera,
/// the capture time on a common clock and the exposure settings.
#[must_use]
pub fn burst_frame(meta: &RawMetadata) -> BurstFrame {
    BurstFrame {
        camera: format!("{}|{}", meta.make, meta.model),
        capture_time_s: meta.capture_time.map(|t| capture_seconds(&t)),
        settings: ShotSettings {
            shutter_s: meta.exposure_time_s.unwrap_or(0.0),
            f_number: meta.f_number.unwrap_or(0.0),
            iso: f64::from(meta.iso.unwrap_or(0)),
            exposure_compensation_ev: meta.exposure_comp_ev.unwrap_or(0.0),
        },
    }
}

/// Seconds since 1970 of a capture time, with the camera's UTC offset removed when it has one.
fn capture_seconds(t: &ym_raw::exif::CaptureTime) -> f64 {
    // Days from civil date (Howard Hinnant's algorithm).
    let (y, m, d) = (i64::from(t.year), i64::from(t.month), i64::from(t.day));
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let seconds =
        days * 86_400 + i64::from(t.hour) * 3600 + i64::from(t.minute) * 60 + i64::from(t.second)
            - i64::from(t.utc_offset_minutes.unwrap_or(0)) * 60;
    seconds as f64 + f64::from(t.subsec_ms.unwrap_or(0)) / 1000.0
}

// ---------------------------------------------------------------------------------------------
// AutoEstimates
// ---------------------------------------------------------------------------------------------

/// What the inputs of [`AutoEstimates::compute`] are besides the decoded image.
pub struct AutoInputs<'a> {
    /// The camera profile the photo uses.
    pub profile: &'a Profile,
    /// The illuminant model of that profile, if the data folder has one.
    pub illuminant_model: Option<&'a IlluminantModel>,
    /// Noise priors by body.
    pub priors: &'a PriorStore,
}

/// Per-photo automatic estimates, computed once from the decoded raw and recomputed by Re-run.
#[derive(Debug, Clone)]
pub struct AutoEstimates {
    /// The scene light and how it was found.
    pub illuminant: IlluminantResult,
    /// The colour part: scene light, initial white and `EV_auto`, as `ym_color::chain` takes it.
    pub color: ColorEstimates,
    /// The automatic exposure.
    pub exposure: ExposureResult,
    /// The white lift: the automatic amount and the anchor (`docs/02` C12 step 1b). Add the user's
    /// `white_offset_ev` with [`WhiteEstimate::lift`].
    pub white: WhiteEstimate,
    /// The noise model.
    pub noise: NoiseResult,
    /// What `sharpen: Auto` resolves to.
    pub sharpen: SharpenValue,
    /// What `noise_reduction: Auto` resolves to.
    pub noise_reduction: NoiseReductionValue,
    /// The Auto white-balance readout (Kelvin rounded to 50 K, tint) of the all-Auto chain (with Apple
    /// Auto's nudge of the white, when there is one).
    pub readout: WbReadout,
    /// What `contrast: Auto` resolves to: 0, or Apple Auto's contrast (`docs/19` section 5).
    pub contrast: f64,
    /// What `saturation: Auto` resolves to: 0, or Apple Auto's saturation.
    pub saturation: f64,
    /// What `highlights: Auto` resolves to: 0, or Apple Auto's highlights.
    pub highlights: f64,
    /// What `shadows: Auto` resolves to: 0, or Apple Auto's shadows.
    pub shadows: f64,
    /// What `black: Auto` resolves to: 0, or Apple Auto's black level.
    pub black: f64,
    /// What the shadows wheel's Auto resolves to: `[0, 0]`, or Apple Auto's shadow colour.
    pub balance_shadows: [f64; 2],
    /// What the highlights wheel's Auto resolves to: `[0, 0]`, or Apple Auto's highlight colour.
    pub balance_highlights: [f64; 2],
    /// Where the photo stands with Apple Auto (`docs/19` section 12): [`AppleAuto::Off`] for the
    /// estimates of the photo alone and wherever Apple Auto is off.
    pub apple: AppleAuto,
    /// The Apple Auto steps in effect: the photo's answer, or its burst's shared steps (`docs/19`
    /// section 5). `None` when Auto is the rule.
    pub apple_steps: Option<AutoSteps>,
}

/// Which Auto made a photo's automatic values (`docs/19` sections 8 and 9: `ym:Auto`, `auto.source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoSource {
    /// Yellowmoon's own Auto.
    Rule,
    /// Apple Auto's answer (the photo's own or its burst's).
    Apple,
}

impl AutoSource {
    /// The wire name: `rule` or `apple`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rule => "rule",
            Self::Apple => "apple",
        }
    }
}

/// The `auto` object of `src/api/types.ts` (`AutoEstimates` there) and of `ym render --json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoSummary {
    /// `EV_auto` (with Apple Auto's change, when there is one).
    pub exposure_ev: f64,
    /// The automatic white lift in stops (0 for a photo with nothing above +1 stop to lift): the
    /// Automatic card's White row. The lift in effect is this plus `white_offset_ev`.
    pub white_ev: f64,
    /// The Auto Kelvin (a multiple of 50; with Apple Auto's nudge, when there is one).
    pub kelvin: f64,
    /// The Auto tint.
    pub tint: f64,
    /// What `contrast: Auto` resolves to: 0, or Apple Auto's contrast.
    pub contrast: f64,
    /// What `saturation: Auto` resolves to: 0, or Apple Auto's saturation.
    pub saturation: f64,
    /// What `highlights: Auto` resolves to: 0, or Apple Auto's highlights.
    pub highlights: f64,
    /// What `shadows: Auto` resolves to: 0, or Apple Auto's shadows.
    pub shadows: f64,
    /// What `black: Auto` resolves to: 0, or Apple Auto's black level.
    pub black: f64,
    /// What the shadows wheel's Auto resolves to: `[0, 0]`, or Apple Auto's shadow colour.
    pub balance_shadows: [f64; 2],
    /// What the highlights wheel's Auto resolves to: `[0, 0]`, or Apple Auto's highlight colour.
    pub balance_highlights: [f64; 2],
    /// Which Auto made these values: `rule` or `apple`.
    pub source: AutoSource,
    /// Where the photo stands with Apple Auto (`docs/19` section 12).
    pub apple: AppleAuto,
    /// Apple Auto's steps in effect (the photo's answer, or its burst's shared steps), `None` when
    /// Auto is the rule. A step the model left empty is `null`: nothing was set for that control.
    pub apple_steps: Option<AutoSteps>,
    /// The most probable light class.
    pub light_class: LightClass,
    /// Confidence of the light estimate, 0 to 1.
    pub light_confidence: f64,
    /// The noise model `variance = a * signal + b`.
    pub noise: NoiseAb,
    /// What `sharpen: Auto` resolves to.
    pub sharpen: SharpenValue,
    /// What `noise_reduction: Auto` resolves to.
    pub noise_reduction: NoiseReductionValue,
}

/// The shot-noise gain and read-noise variance, in normalised units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoiseAb {
    /// Shot-noise gain.
    pub a: f64,
    /// Read-noise variance.
    pub b: f64,
}

/// The base ISO of a body for Auto sharpening: from the gain-regime table, else 100.
#[must_use]
pub fn base_iso(priors: &PriorStore, meta: &RawMetadata) -> u32 {
    meta.model_id
        .and_then(|id| priors.gain_regimes().body(u32::from(id)))
        .and_then(|b| b.base_iso)
        .unwrap_or(DEFAULT_BASE_ISO)
}

/// The Auto sharpening and noise-reduction values for a shot (`docs/05` R8 and R7).
#[must_use]
pub fn auto_detail_values(
    iso: Option<u32>,
    base_iso: u32,
    f_number: Option<f64>,
) -> (SharpenValue, NoiseReductionValue) {
    let iso_f = iso.filter(|i| *i > 0).unwrap_or(base_iso) as f32;
    let s = auto_sharpen(iso_f, base_iso as f32, f_number.map(|f| f as f32));
    let nr = NoiseReduction::auto(iso_f);
    (
        SharpenValue {
            amount: f64::from(s.amount),
            radius_px: f64::from(s.radius_px),
        },
        NoiseReductionValue {
            luminance: f64::from(nr.luminance),
            color: f64::from(nr.color),
            detail: f64::from(nr.detail),
        },
    )
}

impl AutoEstimates {
    /// Computes every estimate from the decoded image.
    ///
    /// Failures of the optional parts never fail the photo: a missing or failing illuminant model
    /// gives the as-shot fallback, an unmeasurable noise level gives the prior. What fails is an
    /// image that cannot be analysed at all.
    ///
    /// # Errors
    ///
    /// [`AutoError`] for an image that is too small or has levels with no range, and for a colour
    /// chain that cannot be built (a profile with unusable tables).
    pub fn compute(raw: &RawImage, inputs: &AutoInputs<'_>) -> Result<Self, AutoError> {
        Self::compute_scaled(raw, inputs, wb::IDENTITY_WHITE_SCALE)
    }

    /// As [`AutoEstimates::compute`] for a body with a white scale (`CameraMatch::white_scale`, `docs/03`
    /// White scale of stand-in bodies): the as-shot white, the illuminant estimator's input and the initial
    /// gains are put on the profile's scale, so a body that borrows a profile reads its whites the way the
    /// profile's curves would. [`AutoEstimates::compute`] is this with `[1, 1, 1]`, for a body whose profile
    /// was measured on it.
    ///
    /// # Errors
    ///
    /// As [`AutoEstimates::compute`]; a `white_scale` that is not positive and finite (green 1) is
    /// [`AutoError::White`].
    pub fn compute_scaled(
        raw: &RawImage,
        inputs: &AutoInputs<'_>,
        white_scale: Vec3,
    ) -> Result<Self, AutoError> {
        Self::compute_with_look(raw, inputs, white_scale, None)
    }

    /// As [`AutoEstimates::compute_scaled`], with the white lift sized for `reference_look` (the
    /// Signature look: the lift's amount is what the picture needs on top of that look, so the two do
    /// not add up twice; `None` is the identity look). The session passes the Signature look from the
    /// data folder; the lift does not follow the look a photo is later given.
    ///
    /// # Errors
    ///
    /// As [`AutoEstimates::compute_scaled`].
    pub fn compute_with_look(
        raw: &RawImage,
        inputs: &AutoInputs<'_>,
        white_scale: Vec3,
        reference_look: Option<&LookTransform>,
    ) -> Result<Self, AutoError> {
        let meta = &raw.meta;
        wb::scale_white([1.0; 3], white_scale)?;
        let preview = build_preview(raw)?;
        let illuminant = estimate_light(
            inputs.illuminant_model,
            &preview,
            meta,
            as_shot_rggb(raw),
            inputs.profile,
            white_scale,
        )?;
        let w_scene_initial = illuminant.scene.w_scene;
        let mut color = ColorEstimates {
            scene: illuminant.scene,
            w_scene_initial,
            white_scale,
            ev_auto: 0.0,
            white_shift_mired: 0.0,
            tint_shift: 0.0,
        };
        let tables = tables_of(inputs.profile);
        let metering = color.metering_chain(tables)?;
        let g0 = color.initial_gains()?;
        let balanced = preview.metered(g0);
        let preview_rgb = PreviewRgb {
            width: preview.width,
            height: preview.height,
            rgb: &balanced,
            clip_mask: Some(&preview.clip),
        };
        let meter = auto_exposure(&preview_rgb, metering.matrix())?;
        color.ev_auto = meter.ev_auto;
        let white = white_lift_estimate(&color, tables, &preview_rgb, reference_look)?;
        let noise = estimate_noise(raw, inputs.priors)?;
        let (sharpen, noise_reduction) =
            auto_detail_values(meta.iso, base_iso(inputs.priors, meta), meta.f_number);
        Ok(Self {
            illuminant,
            color,
            exposure: ExposureResult {
                ev: meter.ev_auto,
                own_ev: meter.ev_auto,
                meter,
            },
            white,
            noise,
            sharpen,
            noise_reduction,
            readout: metering.readout(),
            contrast: 0.0,
            saturation: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            black: 0.0,
            balance_shadows: [0.0, 0.0],
            balance_highlights: [0.0, 0.0],
            apple: AppleAuto::Off,
            apple_steps: None,
        })
    }

    /// The same estimates with the exposure set to a shared value (the burst median).
    #[must_use]
    pub fn with_shared_ev(&self, ev: f64) -> Self {
        let mut s = self.clone();
        s.exposure.ev = ev;
        s.color.ev_auto = ev;
        s
    }

    /// The same estimates with a pooled noise model.
    #[must_use]
    pub fn with_noise(&self, noise: NoiseResult) -> Self {
        let mut s = self.clone();
        s.noise = noise;
        s
    }

    /// The same estimates with Apple Auto's state and steps applied (`docs/19` section 5), on top of
    /// the rule's exposure as it stands (after the burst median): the exposure gains the table's EV
    /// (`exposure.ev` and `color.ev_auto` together, as [`AutoEstimates::with_shared_ev`] moves them),
    /// the white its nudges of warmth and tint (and the readout follows them), and highlights,
    /// shadows, black, contrast, saturation and the shadows and highlights wheels their values. A
    /// step the model left empty keeps the value without Apple Auto.
    /// The highlight guard is not applied again and the white lift is not sized again. `steps` of
    /// `None` (no answer, a failure, Apple Auto off) leave every value at the rule's, bit for bit; only
    /// `apple` records the state. A step out of its range counts as no answer.
    #[must_use]
    pub fn with_apple(
        &self,
        steps: Option<AutoSteps>,
        state: AppleAuto,
        tables: ProfileTables<'_>,
    ) -> Self {
        let mut s = self.clone();
        s.apple = state;
        let Some((steps, values)) = steps.and_then(|st| st.values().map(|v| (st, v))) else {
            s.apple_steps = None;
            return s;
        };
        s.apple_steps = Some(steps);
        let AppleValues {
            exposure_ev,
            highlights,
            shadows,
            black,
            contrast,
            white_shift_mired,
            tint_shift,
            saturation,
            balance_shadows,
            balance_highlights,
        } = values;
        if exposure_ev != 0.0 {
            s.exposure.ev = self.exposure.ev + exposure_ev;
            s.color.ev_auto = self.color.ev_auto + exposure_ev;
        }
        s.contrast = contrast;
        s.saturation = saturation;
        s.highlights = highlights;
        s.shadows = shadows;
        s.black = black;
        s.balance_shadows = balance_shadows;
        s.balance_highlights = balance_highlights;
        if white_shift_mired != 0.0 || tint_shift != 0.0 {
            s.color.white_shift_mired = white_shift_mired;
            s.color.tint_shift = tint_shift;
            // The metering chain is the all-Auto chain; its readout is what the White balance rows
            // show. It cannot fail for estimates whose rule chain was built (the nudge moves a white
            // inside the tables), but if it did the rule's readout is kept rather than the photo lost.
            match s.color.metering_chain(tables) {
                Ok(chain) => s.readout = chain.readout(),
                Err(e) => {
                    tracing::warn!(error = %e, "the nudged white could not be read out");
                }
            }
        }
        s
    }

    /// Which Auto made the values: Apple Auto's when steps are in effect, else the rule.
    #[must_use]
    pub fn source(&self) -> AutoSource {
        if self.apple_steps.is_some() {
            AutoSource::Apple
        } else {
            AutoSource::Rule
        }
    }

    /// The initial gains `g0` of the raw stages, `white_scale / w_scene_initial` with green = 1
    /// (`1 / w_scene_initial` for a body whose profile was measured on it).
    ///
    /// # Errors
    ///
    /// [`WbError::InvalidWhite`] for an unusable white (not possible for a computed estimate).
    pub fn initial_gains(&self) -> Result<Vec3, WbError> {
        self.color.initial_gains()
    }

    /// The fused C1 to C6 chain for a photo's settings: the exposure and white balance resolve
    /// against these estimates, `picked_white` is the sample that belongs to
    /// `settings.white_pick` ([`pick_white_sample`]).
    ///
    /// # Errors
    ///
    /// [`ChainError`] for a Pick white sample that is refused or values that cannot make a chain.
    pub fn fused_chain(
        &self,
        profile: &Profile,
        settings: &DevelopSettings,
        picked_white: Option<Vec3>,
    ) -> Result<FusedChain, ChainError> {
        self.color.chain(
            tables_of(profile),
            settings.light_source,
            settings.white_balance.to_color(),
            settings.exposure.to_color(),
            picked_white.filter(|_| settings.white_pick.is_some()),
        )
    }

    /// The summary the UI and `--json` show.
    #[must_use]
    pub fn summary(&self) -> AutoSummary {
        let classes = self.illuminant.scene.classes;
        let light_class = LightClass::ALL
            .into_iter()
            .fold((LightClass::Daylight, f64::NEG_INFINITY), |best, c| {
                let p = classes.get(c);
                if p > best.1 { (c, p) } else { best }
            })
            .0;
        AutoSummary {
            exposure_ev: self.exposure.ev,
            white_ev: self.white.amount_ev,
            kelvin: self.readout.kelvin_k,
            tint: self.readout.tint,
            contrast: self.contrast,
            saturation: self.saturation,
            highlights: self.highlights,
            shadows: self.shadows,
            black: self.black,
            balance_shadows: self.balance_shadows,
            balance_highlights: self.balance_highlights,
            source: self.source(),
            apple: self.apple.clone(),
            apple_steps: self.apple_steps,
            light_class,
            light_confidence: self.illuminant.scene.confidence,
            noise: NoiseAb {
                a: self.noise.model.a,
                b: self.noise.model.b,
            },
            sharpen: self.sharpen,
            noise_reduction: self.noise_reduction,
        }
    }
}

/// The white lift of a photo (`docs/02` C12 step 1b): its preview through the all-Auto chain at the
/// photo's exposure and C7 at its defaults, not the user's (`Rolloff::default()`: the highlight range at
/// 1 with the chain's clip level, `ClipLevel::luma`), sized for `reference_look`. The noise floor is 0,
/// so the toe is off whatever its amount: it only acts within a few noise floors of black, far from the
/// percentiles the estimate reads.
fn white_lift_estimate(
    color: &ColorEstimates,
    tables: ProfileTables<'_>,
    preview: &PreviewRgb<'_>,
    reference_look: Option<&LookTransform>,
) -> Result<WhiteEstimate, AutoError> {
    let chain = color.chain(
        tables,
        LightSource::Auto,
        WhiteBalance::Auto,
        ExposureSetting::Auto,
        None,
    )?;
    let rolloff = Rolloff::default();
    let Ok(tone) = ToneParams::new(
        rolloff.highlights,
        rolloff.low_light,
        chain.clip().luma(),
        0.0,
    ) else {
        // A clip level that is not a positive number: nothing to size a lift against.
        return Ok(WhiteEstimate::none());
    };
    Ok(estimate_white(
        preview,
        chain.matrix(),
        &tone,
        reference_look,
    )?)
}

/// The three tables of a profile that the colour chain reads.
#[must_use]
pub fn tables_of(profile: &Profile) -> ProfileTables<'_> {
    ProfileTables {
        sensor_map: &profile.sensor_map.classes,
        alexa_white: &profile.alexa_white,
        locus: &profile.locus,
    }
}

// ---------------------------------------------------------------------------------------------
// Pick white
// ---------------------------------------------------------------------------------------------

/// The Pick white sample at a point: the per-channel median of the 5 x 5 pixels around it, in
/// camera RGB balanced with the initial gains `g0` (`docs/02` C1), ready for
/// [`wb::picked_scene_white`].
///
/// `point` is `DevelopSettings::white_pick`: a fraction of the photo as the camera oriented it. The
/// pixels are demosaiced the cheap way (each channel is the mean of the photosites of that colour
/// in the 3 x 3 around the pixel), which is exact for a neutral patch. The sample is taken on the
/// raw data as it is, without the vignetting gain; a gain is the same for all three channels, so
/// it does not change the colour. `None` when the point is outside the photo or the data is not
/// finite.
#[must_use]
pub fn pick_white_sample(raw: &RawImage, point: [f64; 2], g0: Vec3) -> Option<Vec3> {
    if !point.iter().all(|v| v.is_finite()) {
        return None;
    }
    let area = image_area(raw);
    let [fx, fy] = display_to_sensor(
        raw.meta.orientation,
        [point[0].clamp(0.0, 1.0), point[1].clamp(0.0, 1.0)],
    );
    let cx = ((fx * area.width as f64) as usize).min(area.width - 1);
    let cy = ((fy * area.height as f64) as usize).min(area.height - 1);
    // A 5 x 5 window kept inside the area.
    let half = 2usize;
    let x_lo = cx
        .saturating_sub(half)
        .min(area.width.saturating_sub(2 * half + 1));
    let y_lo = cy
        .saturating_sub(half)
        .min(area.height.saturating_sub(2 * half + 1));
    let x_hi = (x_lo + 2 * half + 1).min(area.width);
    let y_hi = (y_lo + 2 * half + 1).min(area.height);
    let levels = Levels::new(raw).ok()?;
    let stride = raw.data.width;
    let at = |x: usize, y: usize, c: usize| -> f64 {
        // Normalised value of channel `c` of the photosite or pixel at area coordinates.
        let (px, py) = (area.x + x, area.y + y);
        match raw.layout {
            Layout::Bayer { .. } => {
                let p = ((py & 1) << 1) | (px & 1);
                f64::from(
                    (f32::from(raw.data.data[py * stride + px]) - levels.black[p])
                        * levels.inv_range[p],
                )
            }
            Layout::Ycbcr { .. } => {
                let p = [0usize, 1, 3][c];
                f64::from(
                    (f32::from(raw.data.data[py * stride + 3 * px + c]) - levels.black[p])
                        * levels.inv_range[p],
                )
            }
        }
    };
    let mut pixels: Vec<Vec3> = Vec::with_capacity((x_hi - x_lo) * (y_hi - y_lo));
    for y in y_lo..y_hi {
        for x in x_lo..x_hi {
            let rgb = match raw.layout {
                Layout::Bayer { pattern, .. } => {
                    let mut sum = [0.0f64; 3];
                    let mut n = [0u32; 3];
                    for yy in y.saturating_sub(1)..=(y + 1).min(area.height - 1) {
                        for xx in x.saturating_sub(1)..=(x + 1).min(area.width - 1) {
                            let c = match pattern.color_at(area.x + xx, area.y + yy) {
                                CfaColor::Red => 0,
                                CfaColor::Green => 1,
                                CfaColor::Blue => 2,
                            };
                            sum[c] += at(xx, yy, 0);
                            n[c] += 1;
                        }
                    }
                    if n.contains(&0) {
                        return None;
                    }
                    [
                        sum[0] / f64::from(n[0]),
                        sum[1] / f64::from(n[1]),
                        sum[2] / f64::from(n[2]),
                    ]
                }
                Layout::Ycbcr { .. } => [at(x, y, 0), at(x, y, 1), at(x, y, 2)],
            };
            pixels.push([rgb[0] * g0[0], rgb[1] * g0[1], rgb[2] * g0[2]]);
        }
    }
    wb::median_rgb(&pixels)
}
