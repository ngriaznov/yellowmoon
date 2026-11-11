//! Mask arithmetic and the mask builders of `docs/27` section 5.
//!
//! The builders work on full-resolution masks (`docs/27` section 9, decision 1) and return
//! [`BoundedMask`]s cropped to the bounding box of their non-zero pixels.

use std::f64::consts::SQRT_2;

use rayon::prelude::*;

use crate::error::{Result, RetouchError};
use crate::feather::feather;
use crate::models::{FacePart, MeshTable, Models};
use crate::types::{BoundedMask, Face, LinearAwg4Image, Mask, Rect, face_extent_px};

/// `(a * b) / 255` with integer division.
pub fn mul(a: u8, b: u8) -> u8 {
    ((u32::from(a) * u32::from(b)) / 255) as u8
}

/// `(a * (255 - b)) / 255` with integer division.
pub fn mul_inv(a: u8, b: u8) -> u8 {
    ((u32::from(a) * (255 - u32::from(b))) / 255) as u8
}

/// The larger of two bytes.
pub fn max(a: u8, b: u8) -> u8 {
    a.max(b)
}

fn zip_masks(a: &Mask, b: &Mask, op: fn(u8, u8) -> u8) -> Result<Mask> {
    a.validate()?;
    b.validate()?;
    if a.size_px() != b.size_px() {
        return Err(RetouchError::MaskSizeMismatch {
            left: a.size_px(),
            right: b.size_px(),
        });
    }
    let data = a
        .data
        .par_iter()
        .zip(b.data.par_iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    Ok(Mask {
        width_px: a.width_px,
        height_px: a.height_px,
        data,
    })
}

/// Per-pixel [`mul`] of two masks of the same size.
pub fn mul_masks(a: &Mask, b: &Mask) -> Result<Mask> {
    zip_masks(a, b, mul)
}

/// Per-pixel [`mul_inv`] of two masks of the same size.
pub fn mul_inv_masks(a: &Mask, b: &Mask) -> Result<Mask> {
    zip_masks(a, b, mul_inv)
}

/// Per-pixel [`max`] of two masks of the same size.
pub fn max_masks(a: &Mask, b: &Mask) -> Result<Mask> {
    zip_masks(a, b, max)
}

/// Runs the parser for the skin (`docs/27` 5.2): the face's skin without ears, at full size.
/// `None` when the parser finds no skin.
pub fn skin_mask(
    models: &Models,
    image: &LinearAwg4Image,
    face: &Face,
    full_size_px: (u32, u32),
) -> Result<Option<Mask>> {
    let Some(skin) = models
        .parser
        .part_mask(image, face, FacePart::SkinWithoutEars)?
    else {
        return Ok(None);
    };
    Ok(Some(skin.to_full(full_size_px)?))
}

/// Makes a soft elliptical hole in `mask` for one beauty-mark `rect` (`docs/27` 5.3).
///
/// The inner ellipse is inscribed in the box and is 0. The outer ellipse has the same centre and
/// axes times the square root of 2. Between them, with `d` the normalised elliptical distance of
/// the pixel centre (1 at the inner edge, root 2 at the outer), `pixel = floor(pixel * (d - 1) /
/// (sqrt(2) - 1))`. Pixels outside the outer ellipse, and an empty box, change nothing.
pub fn punch_hole(mask: &mut Mask, rect: &Rect) {
    if rect.width_px == 0 || rect.height_px == 0 {
        return;
    }
    let (cx, cy) = (
        f64::from(rect.x_px) + f64::from(rect.width_px) / 2.0,
        f64::from(rect.y_px) + f64::from(rect.height_px) / 2.0,
    );
    let (ax, ay) = (
        f64::from(rect.width_px) / 2.0,
        f64::from(rect.height_px) / 2.0,
    );
    let (w, h) = (i64::from(mask.width_px), i64::from(mask.height_px));
    let x0 = ((cx - ax * SQRT_2).floor() as i64).max(0);
    let x1 = ((cx + ax * SQRT_2).ceil() as i64).min(w);
    let y0 = ((cy - ay * SQRT_2).floor() as i64).max(0);
    let y1 = ((cy + ay * SQRT_2).ceil() as i64).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let (nx, ny) = ((x as f64 + 0.5 - cx) / ax, (y as f64 + 0.5 - cy) / ay);
            let d = (nx * nx + ny * ny).sqrt();
            let pixel = &mut mask.data[(y * w + x) as usize];
            if d <= 1.0 {
                *pixel = 0;
            } else if d < SQRT_2 {
                *pixel = (f64::from(*pixel) * (d - 1.0) / (SQRT_2 - 1.0)).floor() as u8;
            }
        }
    }
}

/// The blemish mask (`docs/27` 5.3): `skin` with a soft hole
/// for each beauty-mark box. No other blur or erosion.
pub fn blemish_mask(skin: &Mask, beauty_marks: &[Rect]) -> Result<Mask> {
    skin.validate()?;
    let mut mask = skin.clone();
    for rect in beauty_marks {
        punch_hole(&mut mask, rect);
    }
    Ok(mask)
}

/// The dark-circle mask (`docs/27` 5.4): `mul(under_eye, skin_without_ears)`, then
/// `mul_inv(m, feather(eyes, +0.03 * F))` when the eyes mask exists. All masks are full size;
/// `extent_px` is `F` ([`face_extent_px`]).
pub fn dark_circle_mask(
    under_eye: &Mask,
    skin_without_ears: &Mask,
    eyes: Option<&Mask>,
    extent_px: f64,
) -> Result<Mask> {
    let m = mul_masks(under_eye, skin_without_ears)?;
    match eyes {
        Some(eyes) => mul_inv_masks(&m, &feather(eyes, 0.03 * extent_px)?),
        None => Ok(m),
    }
}

/// The grading mask (`docs/27` 5.6), which is also the `edge` of 5.5: `feather(skin, -0.065 * F)`,
/// so skin grading fades out before the hairline, the brows and the jaw. `extent_px` is `F`.
pub fn grading_mask(skin: &Mask, extent_px: f64) -> Result<Mask> {
    feather(skin, -0.065 * extent_px)
}

/// The even-skin mask (`docs/27` 5.5). `uniformity` is the mesh's `InvertedSkinUniformity` mask;
/// `exclude` is an optional exclusion mask (default none). All masks are full size; `extent_px`
/// is `F`.
pub fn even_skin_mask(
    skin: &Mask,
    uniformity: &Mask,
    exclude: Option<&Mask>,
    extent_px: f64,
) -> Result<Mask> {
    let edge = grading_mask(skin, extent_px)?;
    even_skin_with_edge(skin, uniformity, exclude, extent_px, &edge)
}

/// 5.5 with the `edge` of 5.6 already built.
fn even_skin_with_edge(
    skin: &Mask,
    uniformity: &Mask,
    exclude: Option<&Mask>,
    extent_px: f64,
    edge: &Mask,
) -> Result<Mask> {
    let mut core = mul_inv_masks(skin, uniformity)?;
    if let Some(exclude) = exclude {
        core = mul_inv_masks(&core, exclude)?;
    }
    let core = feather(&core, -0.011 * extent_px)?;
    mul_masks(&core, edge)
}

/// `smoothstep(e0, e1, x)`: 0 below `e0`, 1 above `e1`, the cubic `3t^2 - 2t^3` between.
pub fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The nearest-rank percentile `p` (0..=1) of `values`, sorted in place with a total order so the
/// result never depends on the input order. 0 for an empty slice.
pub(crate) fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let rank = (p.clamp(0.0, 1.0) * values.len() as f64).ceil() as usize;
    values[rank.clamp(1, values.len()) - 1]
}

/// The gate of 5.6 for one pixel: `round(255 * bright * pale * notred)`, with `Y = (r + g + b) / 3`,
/// `sat = 1 - min / max`, `y_mouth` the mouth region's 80th percentile of `Y`.
pub fn teeth_gate(pixel: [f32; 3], y_mouth: f64) -> u8 {
    let (r, g, b) = (
        f64::from(pixel[0]).max(0.0),
        f64::from(pixel[1]).max(0.0),
        f64::from(pixel[2]).max(0.0),
    );
    let y = (r + g + b) / 3.0;
    if y_mouth.is_nan() || y_mouth <= 0.0 || !y.is_finite() {
        return 0;
    }
    let max = r.max(g).max(b);
    let sat = if max > 0.0 {
        1.0 - r.min(g).min(b) / max
    } else {
        0.0
    };
    let bright = smoothstep(0.6, 1.0, y / y_mouth);
    let pale = 1.0 - smoothstep(0.35, 0.6, sat);
    let notred = 1.0 - smoothstep(0.08, 0.2, (r - g) / y.max(1e-4));
    (255.0 * bright * pale * notred).round().clamp(0.0, 255.0) as u8
}

/// The teeth mask (`docs/27` 5.6) from the full-size mouth-interior mask and the full-resolution
/// image: `mouth = feather(mouth, -0.004 F)`; `Ymouth` is the 80th percentile of `Y` over the
/// pixels where `mouth` is not 0; `teeth = mul(mouth, gate)`, then `feather(teeth, +0.002 F)`.
pub fn teeth_mask(mouth: &Mask, image: &LinearAwg4Image, extent_px: f64) -> Result<Mask> {
    mouth.validate()?;
    image.validate()?;
    let size = (image.width_px, image.height_px);
    if mouth.size_px() != size {
        return Err(RetouchError::MaskSizeMismatch {
            left: mouth.size_px(),
            right: size,
        });
    }
    let mouth = feather(mouth, -0.004 * extent_px)?;
    let Some(bounds) = BoundedMask::from_full(&mouth) else {
        return Ok(mouth);
    };
    let w = image.width_px as usize;
    let (x0, y0) = (bounds.region.x_px as usize, bounds.region.y_px as usize);
    let (rw, rh) = (
        bounds.region.width_px as usize,
        bounds.region.height_px as usize,
    );
    let mut ys = Vec::new();
    for y in y0..y0 + rh {
        for x in x0..x0 + rw {
            if mouth.data[y * w + x] != 0 {
                let p = image.data[y * w + x];
                ys.push(
                    (f64::from(p[0]).max(0.0)
                        + f64::from(p[1]).max(0.0)
                        + f64::from(p[2]).max(0.0))
                        / 3.0,
                );
            }
        }
    }
    let y_mouth = percentile(&mut ys, 0.8);
    let mut gate = Mask::zeros(image.width_px, image.height_px)?;
    gate.data
        .par_chunks_mut(w)
        .enumerate()
        .skip(y0)
        .take(rh)
        .for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate().skip(x0).take(rw) {
                if mouth.data[y * w + x] != 0 {
                    *out = teeth_gate(image.data[y * w + x], y_mouth);
                }
            }
        });
    let teeth = mul_masks(&mouth, &gate)?;
    feather(&teeth, 0.002 * extent_px)
}

/// The masks of one face, each cropped to its bounding box; `None` when the mask is all zeros, the
/// parser found no skin, or (for the teeth) found no mouth.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FaceMasks {
    /// Where blemish removal works.
    pub blemish: Option<BoundedMask>,
    /// Where dark circles works.
    pub dark_circles: Option<BoundedMask>,
    /// Where even skin works.
    pub even_skin: Option<BoundedMask>,
    /// Where skin grading works (5.6).
    pub grading: Option<BoundedMask>,
    /// Where teeth whitening works (5.6).
    pub teeth: Option<BoundedMask>,
}

/// Builds the masks of one face with the models from the full-resolution `image`. A failing
/// beauty-mark detector is logged and ignored, and so is a parser that fails on the mouth
/// interior (teeth whitening is then off). `exclude` is the optional exclusion mask of 5.5 (full
/// size).
pub fn face_masks(
    models: &Models,
    image: &LinearAwg4Image,
    face: &Face,
    full_size_px: (u32, u32),
    exclude: Option<&Mask>,
) -> Result<FaceMasks> {
    let Some(skin) = skin_mask(models, image, face, full_size_px)? else {
        return Ok(FaceMasks::default());
    };
    let extent_px = face_extent_px(full_size_px, face);

    let marks = match &models.beauty_marks {
        None => Vec::new(),
        Some(detector) => detector.detect(image, face).unwrap_or_else(|error| {
            tracing::warn!(%error, "the beauty-mark detector failed; going on without it");
            Vec::new()
        }),
    };
    let blemish = blemish_mask(&skin, &marks)?;

    let under_eye = models
        .mesh
        .render(image, face, MeshTable::UnderEye, full_size_px)?
        .to_full(full_size_px)?;
    let eyes = match models.parser.part_mask(image, face, FacePart::Eyes)? {
        Some(eyes) => Some(eyes.to_full(full_size_px)?),
        None => None,
    };
    let dark = dark_circle_mask(&under_eye, &skin, eyes.as_ref(), extent_px)?;

    let uniformity = models
        .mesh
        .render(image, face, MeshTable::InvertedSkinUniformity, full_size_px)?
        .to_full(full_size_px)?;
    let grading = grading_mask(&skin, extent_px)?;
    let even = even_skin_with_edge(&skin, &uniformity, exclude, extent_px, &grading)?;

    let teeth = match models
        .parser
        .part_mask(image, face, FacePart::MouthInterior)
    {
        Ok(Some(mouth)) => Some(teeth_mask(&mouth.to_full(full_size_px)?, image, extent_px)?),
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, "the mouth interior failed; teeth whitening is off");
            None
        }
    };

    Ok(FaceMasks {
        blemish: BoundedMask::from_full(&blemish),
        dark_circles: BoundedMask::from_full(&dark),
        even_skin: BoundedMask::from_full(&even),
        grading: BoundedMask::from_full(&grading),
        teeth: teeth.as_ref().and_then(BoundedMask::from_full),
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::mock;

    proptest! {
        #[test]
        fn u8_arithmetic(a in any::<u8>(), b in any::<u8>()) {
            prop_assert_eq!(mul(a, b), mul(b, a));
            prop_assert!(mul(a, b) <= a.min(b));
            prop_assert_eq!(mul(a, 255), a);
            prop_assert_eq!(mul(a, 0), 0);
            prop_assert_eq!(mul_inv(a, 0), a);
            prop_assert_eq!(mul_inv(a, 255), 0);
            prop_assert!(mul_inv(a, b) <= a);
            prop_assert_eq!(max(a, b), max(b, a));
            prop_assert!(max(a, b) >= a && max(a, b) >= b);
            prop_assert_eq!(u32::from(mul(a, b)), u32::from(a) * u32::from(b) / 255);
        }
    }

    #[test]
    fn arithmetic_is_exact() {
        assert_eq!(mul(128, 128), 64); // 16384 / 255 = 64.25
        assert_eq!(mul(200, 100), 78); // 20000 / 255 = 78.4
        assert_eq!(mul(254, 254), 253);
        assert_eq!(mul_inv(200, 100), 121); // 200 * 155 / 255 = 121.57
        assert_eq!(mul_inv(255, 254), 1);
        assert_eq!(max(3, 9), 9);
        let a = Mask::new(2, 1, vec![200, 10]).unwrap();
        let b = Mask::new(2, 1, vec![100, 255]).unwrap();
        assert_eq!(mul_masks(&a, &b).unwrap().data, vec![78, 10]);
        assert_eq!(mul_inv_masks(&a, &b).unwrap().data, vec![121, 0]);
        assert_eq!(max_masks(&a, &b).unwrap().data, vec![200, 255]);
        let c = Mask::zeros(3, 1).unwrap();
        assert!(matches!(
            mul_masks(&a, &c),
            Err(RetouchError::MaskSizeMismatch { .. })
        ));
    }

    #[test]
    fn hole_ramp() {
        let mut m = Mask::new(40, 40, vec![255; 1600]).unwrap();
        let rect = Rect::new(10, 10, 20, 20);
        punch_hole(&mut m, &rect);
        // Centre (20, 20), axes 10. Inside the inscribed ellipse: 0.
        assert_eq!(m.get(20, 20), 0);
        assert_eq!(m.get(12, 20), 0);
        // Outside the outer ellipse (axes 14.14): untouched.
        assert!(
            m.get(10, 10) > 0 && m.get(10, 10) < 255,
            "the box corner is in the ramp"
        );
        assert_eq!(m.get(0, 0), 255);
        assert_eq!(m.get(5, 20), 255);
        // Between: pixel centre (31.5, 20.5): d = 1.15 on x, 0.05 on y.
        let (nx, ny) = (11.5f64 / 10.0, 0.5f64 / 10.0);
        let d = (nx * nx + ny * ny).sqrt();
        assert!(d > 1.0 && d < SQRT_2);
        let expected = (255.0 * (d - 1.0) / (SQRT_2 - 1.0)).floor() as u8;
        assert_eq!(m.get(31, 20), expected);
        // The ramp rises with the distance.
        let row: Vec<u8> = (30..35).map(|x| m.get(x, 20)).collect();
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        // An empty box and a box off the mask change nothing.
        let before = m.clone();
        punch_hole(&mut m, &Rect::new(3, 3, 0, 5));
        punch_hole(&mut m, &Rect::new(500, 500, 4, 4));
        punch_hole(&mut m, &Rect::new(-50, -50, 4, 4));
        assert_eq!(m, before);
    }

    const FULL: (u32, u32) = (200, 160);

    fn scene() -> (LinearAwg4Image, Face, Models) {
        let image = LinearAwg4Image::filled(FULL.0, FULL.1, [0.5; 3]).unwrap();
        let face = Face {
            rect: Rect::new(50, 30, 80, 100),
            scale: 0.5,
        };
        (image, face, mock::models(vec![Rect::new(85, 70, 8, 8)]))
    }

    #[test]
    fn builders_with_mocks() {
        let (image, face, models) = scene();
        let masks = face_masks(&models, &image, &face, FULL, None).unwrap();
        let blemish = masks.blemish.unwrap();
        let dark = masks.dark_circles.unwrap();
        let even = masks.even_skin.unwrap();
        for m in [&blemish, &dark, &even] {
            m.validate().unwrap();
            assert!(!m.mask.is_all_zero());
        }
        let full = blemish.to_full(FULL).unwrap();
        assert_eq!(full.get(89, 74), 0, "the beauty mark is a hole");
        assert_eq!(full.get(90, 132), 0, "nothing below the face: no neck");
        assert_eq!(full.get(90, 80), 255);
        // Dark circles sit below the eyes, never on them.
        let eyes = models
            .parser
            .part_mask(&image, &face, FacePart::Eyes)
            .unwrap()
            .unwrap()
            .to_full(FULL)
            .unwrap();
        let dark_full = dark.to_full(FULL).unwrap();
        assert!(
            dark_full
                .data
                .iter()
                .zip(&eyes.data)
                .all(|(d, e)| !(*d > 0 && *e == 255))
        );
        assert!(dark.region.y_px >= face.rect.y_px + 40);
        // Even skin is strictly inside the skin and is feathered towards the edge.
        let skin = models
            .parser
            .part_mask(&image, &face, FacePart::SkinWithoutEars)
            .unwrap()
            .unwrap()
            .to_full(FULL)
            .unwrap();
        let even_full = even.to_full(FULL).unwrap();
        assert!(even_full.data.iter().zip(&skin.data).all(|(e, s)| e <= s));
        assert!(even_full.data.iter().any(|&e| e > 0 && e < 255));
        // The grading mask is the even-skin edge: inside the skin, fading out before the border.
        let grading = masks.grading.unwrap();
        let grading_full = grading.to_full(FULL).unwrap();
        assert!(
            grading_full
                .data
                .iter()
                .zip(&skin.data)
                .all(|(g, s)| g <= s)
        );
        assert!(grading_full.data.iter().any(|&g| g > 0 && g < 255));
        assert_eq!(grading_full.get(90, 80), 255);
        assert!(
            even_full
                .data
                .iter()
                .zip(&grading_full.data)
                .all(|(e, g)| e <= g)
        );
        let teeth = masks.teeth.unwrap();
        assert!(!teeth.mask.is_all_zero());
    }

    #[test]
    fn exclusion_removes_even_skin() {
        let (image, face, models) = scene();
        let base = face_masks(&models, &image, &face, FULL, None)
            .unwrap()
            .even_skin
            .unwrap();
        let all = Mask::new(FULL.0, FULL.1, vec![255; (FULL.0 * FULL.1) as usize]).unwrap();
        let none = face_masks(&models, &image, &face, FULL, Some(&all)).unwrap();
        assert!(none.even_skin.is_none());
        assert!(base.region.width_px > 0);
    }

    #[test]
    fn missing_parts_and_failures() {
        let (image, face, mut models) = scene();
        models.beauty_marks = Some(Box::new(mock::FailingBeautyMarks));
        let masks = face_masks(&models, &image, &face, FULL, None).unwrap();
        assert!(masks.blemish.is_some());
        models.parser = Box::new(mock::NoParts);
        assert_eq!(
            face_masks(&models, &image, &face, FULL, None).unwrap(),
            FaceMasks::default()
        );
    }

    #[test]
    fn builders_do_not_depend_on_threads() {
        let (image, face, models) = scene();
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| face_masks(&models, &image, &face, FULL, None).unwrap())
        };
        assert_eq!(run(1), run(4));
    }

    /// A 400 x 320 picture with a mock mouth in the mock face.
    fn mouth_scene() -> (LinearAwg4Image, Face, Models, (u32, u32)) {
        let full = (400u32, 320u32);
        let face = Face {
            rect: Rect::new(100, 40, 200, 250),
            scale: 0.5,
        };
        let mut image = LinearAwg4Image::filled(full.0, full.1, [0.45, 0.35, 0.28]).unwrap();
        mock::paint_mouth(&mut image, &face.rect, 1.0);
        (image, face, mock::models(vec![]), full)
    }

    #[test]
    fn smoothstep_and_percentile() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
        let mut v = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&mut v, 0.8), 4.0);
        assert_eq!(percentile(&mut v, 0.0), 1.0);
        assert_eq!(percentile(&mut v, 1.0), 5.0);
        assert_eq!(percentile(&mut [], 0.5), 0.0);
        let mut one: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&mut one, 0.99), 99.0);
    }

    #[test]
    fn gate_keeps_teeth_and_drops_lips_and_gums() {
        let y_mouth = f64::from(mock::mouth_colors::TEETH.iter().sum::<f32>()) / 3.0;
        let gate = |c: [f32; 3]| teeth_gate(c, y_mouth);
        assert!(gate(mock::mouth_colors::TEETH) > 150);
        assert_eq!(gate(mock::mouth_colors::LIPS), 0);
        assert_eq!(gate(mock::mouth_colors::GAP), 0);
        assert_eq!(gate([0.0; 3]), 0);
        assert_eq!(gate([0.7; 3]), 255, "white teeth pass fully");
        assert_eq!(teeth_gate([0.7; 3], 0.0), 0);
        assert_eq!(teeth_gate([f32::NAN; 3], 0.5), 0);
    }

    #[test]
    fn teeth_mask_covers_the_teeth_only() {
        let (image, face, models, full) = mouth_scene();
        let masks = face_masks(&models, &image, &face, full, None).unwrap();
        let teeth = masks.teeth.unwrap();
        let tf = teeth.to_full(full).unwrap();
        let (cx, cy, ax, ay) = mock::mouth_ellipse(&face.rect);
        // Teeth band: inside the mask; lips ring, the dark gap and the skin: outside.
        assert!(tf.get(cx as i64, cy as i64) > 150);
        assert_eq!(tf.get((cx + 1.2 * ax) as i64, cy as i64), 0, "lips");
        assert_eq!(
            tf.get(cx as i64, (cy + 0.8 * ay) as i64),
            0,
            "the gap below the teeth"
        );
        assert_eq!(
            tf.get((cx + 0.95 * ax) as i64, cy as i64),
            0,
            "gum at the corner"
        );
        assert_eq!(tf.get(10, 10), 0);
        // The mask lies inside the mouth interior grown by the final feather.
        let mouth = models
            .parser
            .part_mask(&image, &face, FacePart::MouthInterior)
            .unwrap()
            .unwrap()
            .to_full(full)
            .unwrap();
        for (i, &t) in tf.data.iter().enumerate() {
            if t > 0 {
                let (x, y) = ((i as u32 % full.0) as i64, (i as u32 / full.0) as i64);
                let near = (-2..=2).any(|dy| (-2..=2).any(|dx| mouth.get(x + dx, y + dy) > 0));
                assert!(near, "teeth mask pixel ({x}, {y}) is outside the mouth");
            }
        }
        // Size mismatch is an error; an empty mouth gives an empty mask.
        let small = Mask::zeros(10, 10).unwrap();
        assert!(teeth_mask(&small, &image, 10.0).is_err());
        let none = Mask::zeros(full.0, full.1).unwrap();
        assert!(teeth_mask(&none, &image, 10.0).unwrap().is_all_zero());
    }

    #[test]
    fn teeth_mask_percentile_adapts_to_the_picture() {
        // The same mouth in a picture twice as bright (gain) gives the same mask.
        let (image, face, models, full) = mouth_scene();
        let base = face_masks(&models, &image, &face, full, None)
            .unwrap()
            .teeth;
        let mut bright = image.clone();
        for p in &mut bright.data {
            *p = p.map(|c| c * 2.0);
        }
        let again = face_masks(&models, &bright, &face, full, None)
            .unwrap()
            .teeth;
        assert_eq!(base, again);
    }
}
