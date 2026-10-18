//! The inputs and labels of the automatic settings dataset (`docs/18-auto-settings.md` sections 3 and 6).
//!
//! * [`photo_features`] is the feature vector of `ym_color::autoset` (section 3) for a photo, computed from the
//!   pieces `AutoEstimates::compute_with_look` already has. The dump (`examples/autoset_dump.rs`) and the app use
//!   it, so both compute the features the same way.
//! * [`scaled_preview`] is the same photo exposed `2^k` times more (the dataset's augmentation, section 6).
//! * [`labels`] is the exposure label `L` of section 6.1 and the quantile tables it reads, ported from
//!   `training/autoset/labels.py` (the two agree to 1e-9 on the fixtures of `tests/data/autoset/labels.json`).
//! * [`image`] has the image operations of the dump: the area resize, the central crop, the quarter turn and the
//!   luminance of the embedded JPEG.

pub mod image;
pub mod labels;

use ym_color::autoset::{FeatureInputs, Features, features};
use ym_color::chain::{ColorEstimates, ProfileTables};
use ym_color::exposure::PreviewRgb;
use ym_color::gamut::awg4_luminance;
use ym_color::illum::ShotInfo;
use ym_raw::metadata::RawMetadata;

use crate::auto::{AutoError, Preview};

/// The features of a photo (`docs/18` section 3): 39 values and the key `m` they are measured from.
///
/// `preview` is the photo's 1/8 preview ([`crate::auto::build_preview`]) and `color` its colour estimates (scene
/// light, initial gains, `EV_auto`); `tables` are the profile's tables ([`crate::auto::tables_of`]).
/// `mean_luminance` is the centre-weighted geometric mean at EV 0 that `auto_exposure` reports
/// (`AutoExposure::mean_luminance`, which `AutoEstimates` keeps in `exposure.meter`). Everything here is the
/// computation `AutoEstimates::compute_with_look` makes, so the photo's own estimates give the same numbers.
///
/// The metering chain is `color.metering_chain` (every colour setting on Auto, EV 0, and the neutral Auto white of
/// `white_preference` 0: the chain the model was trained on, whatever the setting says), the pixels are
/// `preview.metered(g0)` with `g0 = color.initial_gains()`, and the clip level is the AWG4 `Y` of the clipped
/// neutral of that chain (not the Rec.709 luma of `ClipLevel::luma`). The brightness value of the shot comes from
/// the metadata (`ShotInfo::bv`), the ISO and the flash from the metadata too.
///
/// # Errors
///
/// [`AutoError::White`] and [`AutoError::Chain`] when the colour estimates cannot make a chain, and
/// [`AutoError::Autoset`] when a feature cannot be computed (no usable pixel, a mean luminance that is not positive,
/// or a feature that is not finite).
pub fn photo_features(
    meta: &RawMetadata,
    preview: &Preview,
    color: &ColorEstimates,
    tables: ProfileTables<'_>,
    mean_luminance: f64,
) -> Result<Features, AutoError> {
    let g0 = color.initial_gains()?;
    let metering = color.metering_chain(tables)?;
    let balanced = preview.metered(g0);
    let metered = PreviewRgb {
        width: preview.width,
        height: preview.height,
        rgb: &balanced,
        clip_mask: Some(&preview.clip),
    };
    let iso = meta.iso.map(f64::from);
    let shot = ShotInfo {
        as_shot_gains: None,
        f_number: meta.f_number,
        exposure_time_s: meta.exposure_time_s,
        iso,
    };
    let inputs = FeatureInputs {
        preview: &metered,
        matrix_ev0: metering.matrix(),
        mean_luminance,
        clip_y_ev0: awg4_luminance(metering.clip().awg4),
        bv_settings: shot.bv(),
        light: &color.scene,
        iso,
        flash_fired: meta.flash.is_some_and(|flash| flash.fired),
    };
    Ok(features(&inputs)?)
}

/// The same photo exposed `2^k` times more (`docs/18` section 6, the augmentation rows).
///
/// Every value is multiplied by `2^k`. For `k > 0` a value that reaches 1.0 (the sensor's white) is set to 1.0 and
/// its pixel is marked clipped in the clip mask (the mask keeps the clips it had). For `k < 0` nothing is re-clipped
/// and the mask stays as it is. `k = 0` gives a copy. The vignetting gain is unchanged, as the spec says.
///
/// Scaling by a power of two is exact in binary floating point, so `k = -1` halves every value exactly; the
/// values are stored as `f32` in the preview, as they are there.
#[must_use]
pub fn scaled_preview(preview: &Preview, k: f64) -> Preview {
    let gain = k.exp2();
    let mut clip = preview.clip.clone();
    let mut rgb = Vec::with_capacity(preview.rgb.len());
    for (i, pixel) in preview.rgb.chunks_exact(3).enumerate() {
        let mut marked = false;
        for &value in pixel {
            let scaled = f64::from(value) * gain;
            if k > 0.0 && scaled >= 1.0 {
                rgb.push(1.0);
                marked = true;
            } else {
                // Narrowing to f32 is the preview's own storage. NaN stays NaN.
                rgb.push(scaled as f32);
            }
        }
        if marked {
            clip[i] = 1;
        }
    }
    Preview {
        width: preview.width,
        height: preview.height,
        rgb,
        clip,
        vignette_gain: preview.vignette_gain.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview(values: &[f32], clip: &[u8]) -> Preview {
        Preview {
            width: clip.len(),
            height: 1,
            rgb: values.to_vec(),
            clip: clip.to_vec(),
            vignette_gain: None,
        }
    }

    #[test]
    fn a_zero_scaling_is_a_copy() {
        let p = preview(&[0.25, 0.5, 1.0, 0.1, 0.2, 0.3], &[0, 1]);
        let q = scaled_preview(&p, 0.0);
        assert_eq!(q.rgb, p.rgb);
        assert_eq!(q.clip, p.clip);
    }

    #[test]
    fn a_negative_scaling_halves_exactly_and_keeps_the_mask() {
        let p = preview(&[0.25, 0.5, 0.75, 0.125, 0.0, 0.5], &[1, 0]);
        let q = scaled_preview(&p, -1.0);
        assert_eq!(q.rgb, vec![0.125, 0.25, 0.375, 0.0625, 0.0, 0.25]);
        assert_eq!(q.clip, vec![1, 0], "a negative scaling never marks a pixel");
    }

    #[test]
    fn a_positive_scaling_clips_at_white_and_marks_the_pixel() {
        // 0.5 x 2 = 1.0 reaches white: 1.0 and marked. 0.4 x 2 = 0.8 stays. 0.9 x 2 = 1.8 clips.
        let p = preview(&[0.5, 0.1, 0.1, 0.4, 0.3, 0.2, 0.9, 0.0, 0.0], &[0, 0, 0]);
        let q = scaled_preview(&p, 1.0);
        assert_eq!(q.rgb[..3], [1.0, 0.2, 0.2]);
        assert_eq!(q.rgb[3..6], [0.8, 0.6, 0.4]);
        assert_eq!(q.rgb[6..], [1.0, 0.0, 0.0]);
        assert_eq!(q.clip, vec![1, 0, 1]);
    }

    #[test]
    fn the_clip_mask_survives_a_positive_scaling() {
        let p = preview(&[0.01, 0.01, 0.01, 0.01, 0.01, 0.01], &[1, 0]);
        let q = scaled_preview(&p, 2.0);
        assert_eq!(q.clip, vec![1, 0]);
        assert!((q.rgb[3] - 0.04).abs() < 1e-7);
    }

    #[test]
    fn the_vignetting_gain_is_kept() {
        let mut p = preview(&[0.2, 0.2, 0.2], &[0]);
        p.vignette_gain = Some(vec![1.5]);
        let q = scaled_preview(&p, -0.5);
        assert_eq!(q.vignette_gain, Some(vec![1.5]));
    }
}
