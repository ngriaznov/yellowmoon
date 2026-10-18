//! The camera's exposure label `L` of `docs/18-auto-settings.md` section 6.1, made from a photo's embedded JPEG, and the
//! camera floor that the app applies to the learned exposure (section 4.4, the owner's decision of 2026-10-18).
//!
//! `L` is the exposure at which the camera's own JPEG puts our scene luminance at middle grey. Two users read it:
//!
//! * the dataset dump (`examples/autoset_dump.rs`) writes the two quantile tables and `L` of every row;
//! * the app, for every photo that an `autoset` model exposes: the model's exposure is never below `L`
//!   ([`camera_floor`]).
//!
//! Both call [`camera_exposure`], so they compute the same numbers. The JPEG is decoded with `zune-jpeg`; nothing here
//! reads a file: the caller passes the JPEG's bytes.

use thiserror::Error;
use ym_color::autoset::Applied;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::exposure::EV_LIMIT;
use ym_color::gamut::awg4_luminance_row;
use ym_raw::metadata::RawMetadata;

use super::image::{adobe_rgb_luminance, fit_to_preview, rgb_of, srgb_luminance, to_common};
use super::labels::{exposure_label, quantile_grid, quantile_table};
use crate::auto::Preview;

/// The embedded JPEG and the preview are brought to a common size; their aspect ratios may differ by this much
/// (relative), `docs/18` section 6.1.
pub const ASPECT_TOLERANCE: f64 = 0.02;

/// The EXIF ColorSpace (0xA001) of the camera's Adobe RGB mode, which Sony writes uncalibrated (with the interoperability
/// index R03).
pub const COLOR_SPACE_UNCALIBRATED: u16 = 0xFFFF;

/// Why a photo's camera exposure label cannot be read from its embedded JPEG.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CameraExposureError {
    /// The embedded JPEG does not decode to an RGB image (the decoder's reason).
    #[error("the embedded JPEG cannot be decoded ({0})")]
    Decode(String),
    /// The metered preview is empty, or its values do not hold `width x height` RGB pixels.
    #[error("the metered preview does not match its size")]
    PreviewSize,
}

/// The camera's exposure label and the two quantile tables it is read from (`docs/18` section 6.1).
#[derive(Debug, Clone, PartialEq)]
pub struct CameraExposure {
    /// `L` in stops: `log2(0.18 / x)`, the exposure at which the camera's JPEG puts our grey (`x` is the scene level
    /// the JPEG renders as grey). `None` when the curve does not reach middle grey within the range the table allows.
    pub level: Option<f64>,
    /// The quantiles of our scene-linear luminance at EV 0, at the common size (the dump's `g3.scene_q`), 999 points.
    pub scene_q: Vec<f64>,
    /// The quantiles of the JPEG's linear luminance, at the common size (the dump's `g3.jpeg_q`), 999 points.
    pub jpeg_q: Vec<f64>,
}

/// Whether the photo's embedded JPEG is Adobe RGB: the EXIF ColorSpace of the camera's Adobe RGB mode is
/// [`COLOR_SPACE_UNCALIBRATED`] (`docs/18` section 6.1).
#[must_use]
pub fn adobe_rgb_of(meta: &RawMetadata) -> bool {
    meta.color_space == Some(COLOR_SPACE_UNCALIBRATED)
}

/// The camera's exposure label of a photo and the quantile tables it is read from (`docs/18` section 6.1).
///
/// `jpeg` is the embedded JPEG's bytes, `adobe_rgb` its colour space ([`adobe_rgb_of`]). `preview` is the photo's 1/8
/// preview and `balanced` its metered RGB ([`Preview::metered`] at the initial gains `g0`), of `width x height x 3`
/// values; `metering_matrix` is the fused C1 to C6 matrix at EV 0 of the metering chain
/// (`ColorEstimates::metering_chain`). The steps are those of the study: the JPEG is fitted to the preview (quarter
/// turn, central crop beyond [`ASPECT_TOLERANCE`]) and both images are brought to the common size, the smaller of the
/// two in each direction; the quantiles of both luminances are read on [`quantile_grid`]; the level is
/// [`exposure_label`].
///
/// Returns `Ok(None)` when either table has no finite point (no usable pixel): there is no label.
///
/// # Errors
///
/// [`CameraExposureError::Decode`] when the JPEG is not an RGB image, [`CameraExposureError::PreviewSize`] when the
/// preview is empty or `balanced` does not hold its pixels.
pub fn camera_exposure(
    jpeg: &[u8],
    adobe_rgb: bool,
    preview: &Preview,
    balanced: &[f32],
    metering_matrix: &Mat3,
) -> Result<Option<CameraExposure>, CameraExposureError> {
    let (pw, ph) = (preview.width, preview.height);
    let pixels = pw * ph;
    if pixels == 0 || balanced.len() != 3 * pixels {
        return Err(CameraExposureError::PreviewSize);
    }
    let (jw, jh, jrgb) = decode_jpeg(jpeg)?;
    let (rgb, w, h) = fit_to_preview(&jrgb, jw, jh, (pw, ph), ASPECT_TOLERANCE);
    // The common size: the smaller of the two in each direction, the larger image area-averaged down to it.
    let (cw, ch) = (pw.min(w), ph.min(h));
    let linear = if adobe_rgb {
        adobe_rgb_luminance(&rgb)
    } else {
        srgb_luminance(&rgb)
    };
    let mut jpeg_linear = to_common(linear, w, h, cw, ch);
    // Our scene luminance: the metering chain at EV 0 on the metered preview, every pixel (no clip mask, no floor),
    // at the common size.
    let mut scene_linear = to_common(
        scene_luminance(balanced, pixels, metering_matrix),
        pw,
        ph,
        cw,
        ch,
    );
    let grid = quantile_grid();
    let scene_q = quantile_table(&mut scene_linear, &grid);
    let jpeg_q = quantile_table(&mut jpeg_linear, &grid);
    if !scene_q.iter().chain(&jpeg_q).all(|v| v.is_finite()) {
        return Ok(None);
    }
    let level = exposure_label(&scene_q, &jpeg_q);
    Ok(Some(CameraExposure {
        level,
        scene_q,
        jpeg_q,
    }))
}

/// The embedded JPEG as 8-bit RGB: width, height and interleaved pixels.
///
/// # Errors
///
/// [`CameraExposureError::Decode`] when the bytes are not a JPEG the decoder reads, or it is not an RGB image.
pub fn decode_jpeg(jpeg: &[u8]) -> Result<(usize, usize, Vec<u8>), CameraExposureError> {
    let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg));
    let pixels = decoder
        .decode()
        .map_err(|error| CameraExposureError::Decode(format!("{error:?}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| CameraExposureError::Decode("it has no header".to_owned()))?;
    let (w, h) = (usize::from(info.width), usize::from(info.height));
    if w == 0 || h == 0 || pixels.len() != w * h * 3 {
        return Err(CameraExposureError::Decode(
            "it is not an RGB image".to_owned(),
        ));
    }
    Ok((w, h, pixels))
}

/// `Y` of the AWG4 row on every pixel of the metered preview at EV 0 (`auto_exposure`'s own luminance, folded with the
/// matrix), NaN where a value is not finite. Every pixel counts, clipped ones too.
fn scene_luminance(balanced: &[f32], pixels: usize, matrix: &Mat3) -> Vec<f64> {
    let row = luminance_row(matrix);
    (0..pixels)
        .map(|i| match rgb_of(balanced, i) {
            None => f64::NAN,
            Some(c) => {
                let y = row[0] * c[0] + row[1] * c[1] + row[2] * c[2];
                if y.is_finite() { y } else { f64::NAN }
            }
        })
        .collect()
}

/// The luminance row of the metering chain: the AWG4 `Y` row applied after the matrix (`PreviewRgb::luminance_row`).
fn luminance_row(matrix: &Mat3) -> Vec3 {
    let y = awg4_luminance_row();
    std::array::from_fn(|c| y[0] * matrix[0][c] + y[1] * matrix[1][c] + y[2] * matrix[2][c])
}

/// The camera's exposure after the camera floor of `docs/18` section 6.1 (`EV_before_guard`, section 4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraFloor {
    /// The exposure before the highlight guard: `max(EV_model, L)` clamped to ±[`EV_LIMIT`] when the floor applies,
    /// else the model's own value.
    pub ev_before_guard: f64,
    /// True when `L` raised the exposure above the model's (`AutosetEstimates::camera_floor`).
    pub raised: bool,
}

/// The camera floor (`docs/18` sections 4.4 and 6.1): the learned exposure is never darker than the camera's own,
/// `EV_before_guard = max(EV_model, L)`, when the model's exposure is in use and `L` is known. The highlight guard and
/// the burst median follow it as before.
///
/// The model's value stands when `L` is unknown, and when the model's exposure is not usable: a prediction that is not
/// finite gives the rule (`ev_weight` 0 in [`Applied`]), and the rule ignores `L`. The clamp to ±[`EV_LIMIT`] of the
/// model's own value applies to the result too, so a label beyond the limit raises the exposure only to the limit.
#[must_use]
pub fn camera_floor(model: &Applied, camera_ev: Option<f64>) -> CameraFloor {
    let unchanged = CameraFloor {
        ev_before_guard: model.ev_before_guard,
        raised: false,
    };
    match camera_ev {
        Some(camera) if model.ev_weight > 0.0 && camera.is_finite() => {
            let ev = model.ev_before_guard.max(camera).clamp(-EV_LIMIT, EV_LIMIT);
            CameraFloor {
                ev_before_guard: ev,
                raised: ev > model.ev_before_guard,
            }
        }
        _ => unchanged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_color::autoset::rule_only;
    use ym_color::consts::linalg::IDENTITY;
    use ym_color::exposure::CAMERA_GAIN_EV;
    use ym_raw::testgen::tiny_jpeg;

    /// A model's exposure with the given value and weight (the contrast is not read here).
    fn model(ev: f64, weight: f64) -> Applied {
        Applied {
            ev_before_guard: ev,
            ev_weight: weight,
            contrast: 0.0,
            contrast_weight: 0.0,
        }
    }

    #[test]
    fn a_model_below_the_camera_is_raised_to_the_camera() {
        let floor = camera_floor(&model(0.2, 0.8), Some(1.1));
        assert_eq!(
            floor,
            CameraFloor {
                ev_before_guard: 1.1,
                raised: true
            }
        );
    }

    #[test]
    fn a_model_above_the_camera_keeps_its_own_exposure() {
        let floor = camera_floor(&model(1.6, 0.8), Some(1.1));
        assert_eq!(
            floor,
            CameraFloor {
                ev_before_guard: 1.6,
                raised: false
            }
        );
    }

    #[test]
    fn a_model_equal_to_the_camera_is_not_reported_as_raised() {
        let floor = camera_floor(&model(1.1, 0.8), Some(1.1));
        assert_eq!(floor.ev_before_guard, 1.1);
        assert!(!floor.raised);
    }

    #[test]
    fn an_unknown_camera_level_keeps_the_model_exposure() {
        let floor = camera_floor(&model(0.2, 0.8), None);
        assert_eq!(
            floor,
            CameraFloor {
                ev_before_guard: 0.2,
                raised: false
            }
        );
    }

    #[test]
    fn a_non_finite_camera_level_is_unknown() {
        let floor = camera_floor(&model(0.2, 0.8), Some(f64::NAN));
        assert_eq!(floor.ev_before_guard, 0.2);
        assert!(!floor.raised);
    }

    #[test]
    fn the_rule_ignores_the_camera_level() {
        let rule = rule_only();
        let floor = camera_floor(&rule, Some(2.0));
        assert_eq!(floor.ev_before_guard, rule.ev_before_guard);
        assert_eq!(rule.ev_before_guard, CAMERA_GAIN_EV);
        assert!(!floor.raised);
    }

    #[test]
    fn a_prediction_that_is_not_usable_is_the_rule_and_ignores_the_camera_level() {
        let unusable = model(CAMERA_GAIN_EV, 0.0);
        let floor = camera_floor(&unusable, Some(2.0));
        assert_eq!(floor.ev_before_guard, CAMERA_GAIN_EV);
        assert!(!floor.raised);
    }

    #[test]
    fn the_floor_is_clamped_to_the_exposure_limit() {
        let floor = camera_floor(&model(0.5, 0.8), Some(EV_LIMIT + 1.0));
        assert_eq!(floor.ev_before_guard, EV_LIMIT);
        assert!(floor.raised);
        // A model already at the limit is not raised by a label beyond it.
        let at_limit = camera_floor(&model(EV_LIMIT, 0.8), Some(EV_LIMIT + 1.0));
        assert_eq!(at_limit.ev_before_guard, EV_LIMIT);
        assert!(!at_limit.raised);
    }

    /// A preview of `width` x `height` pixels with a smooth ramp of camera values, and its balanced RGB at unit gains.
    fn preview(width: usize, height: usize) -> (Preview, Vec<f32>) {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let t = (x + 2 * y) as f32 / (width + 2 * height) as f32;
                rgb.extend_from_slice(&[0.05 + 0.6 * t, 0.1 + 0.5 * t, 0.2 + 0.3 * t]);
            }
        }
        let p = Preview {
            width,
            height,
            rgb,
            clip: vec![0; width * height],
            vignette_gain: None,
        };
        let balanced = p.balanced([1.0; 3]);
        (p, balanced)
    }

    #[test]
    fn a_synthetic_jpeg_gives_tables_of_the_common_size_and_the_label_of_its_tables() {
        let jpeg = tiny_jpeg(64, 48, 7);
        let (p, balanced) = preview(32, 24);
        let first = camera_exposure(&jpeg, false, &p, &balanced, &IDENTITY)
            .expect("a JPEG that decodes")
            .expect("finite tables");
        assert_eq!(first.scene_q.len(), 999);
        assert_eq!(first.jpeg_q.len(), 999);
        assert!(
            first
                .scene_q
                .iter()
                .chain(&first.jpeg_q)
                .all(|v| v.is_finite())
        );
        assert_eq!(first.level, exposure_label(&first.scene_q, &first.jpeg_q));
        // The same bytes give the same bits.
        let again = camera_exposure(&jpeg, false, &p, &balanced, &IDENTITY)
            .expect("a JPEG that decodes")
            .expect("finite tables");
        assert_eq!(first, again);
    }

    /// FNV-1a over the decoded bytes: the test's fingerprint of a picture.
    fn fingerprint(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    #[test]
    fn subsampled_jpegs_decode_to_the_same_bytes_on_every_machine() {
        // The camera label, and so the exposure, depends on the decoded JPEG (AGENTS.md rule 8). `zune-jpeg` picks
        // AVX2, SSE or NEON code at run time and has no switch to force its scalar code, so this pins the decoded
        // bytes of two pictures with chroma subsampling (4:2:2 as in Sony's previews, and 4:2:0), which exercise its
        // IDCT, upsampling and colour conversion. The fingerprints were taken on x86_64 with AVX2 and agree with a
        // build without its SIMD features (as do the previews of 206 local ARW files); the macOS run of this test
        // checks NEON. Fixtures: `tests/data/autoset/make_subsampled_jpegs.py`.
        for (bytes, expected) in [
            (
                &include_bytes!("../../tests/data/autoset/subsampled-422.jpg")[..],
                0xbf2a_c9c4_10ca_3180_u64,
            ),
            (
                &include_bytes!("../../tests/data/autoset/subsampled-420.jpg")[..],
                0x5873_58b3_4048_f400_u64,
            ),
        ] {
            let (w, h, rgb) = decode_jpeg(bytes).expect("the fixture decodes");
            assert_eq!((w, h), (96, 64));
            assert_eq!(fingerprint(&rgb), expected, "{:016x}", fingerprint(&rgb));
        }
    }

    #[test]
    fn bytes_that_are_not_a_jpeg_are_a_decode_error() {
        let (p, balanced) = preview(32, 24);
        let error = camera_exposure(b"not a jpeg at all", false, &p, &balanced, &IDENTITY)
            .expect_err("not a JPEG");
        assert!(matches!(error, CameraExposureError::Decode(_)), "{error:?}");
    }

    #[test]
    fn a_preview_whose_values_do_not_match_its_size_is_refused() {
        let jpeg = tiny_jpeg(64, 48, 7);
        let (p, mut balanced) = preview(32, 24);
        balanced.pop();
        assert_eq!(
            camera_exposure(&jpeg, false, &p, &balanced, &IDENTITY),
            Err(CameraExposureError::PreviewSize)
        );
    }
}
