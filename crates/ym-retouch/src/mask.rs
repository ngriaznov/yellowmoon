//! Mask arithmetic and the mask builders of `docs/27` section 5.
//!
//! The builders work on full-resolution masks (`docs/27` section 9, decision 1) and return
//! [`BoundedMask`]s cropped to the bounding box of their non-zero pixels.

use std::f64::consts::SQRT_2;

use rayon::prelude::*;

use crate::error::{Result, RetouchError};
use crate::feather::feather;
use crate::models::{FacePart, MeshTable, Models};
use crate::types::{BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, face_extent_px};

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

/// The skin of one face: the skin without ears and, when a neck parser found one, the neck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkinParts {
    /// `SkinWithoutEars`, full size.
    pub skin: Mask,
    /// `NeckSkin`, full size, when found.
    pub neck: Option<Mask>,
}

impl SkinParts {
    /// The skin mask of section 5.2: `max(skin, neck)` over the union of both regions when
    /// `include_neck` and a neck exists, else the skin alone.
    pub fn skin_mask(&self, include_neck: bool) -> Result<Mask> {
        match (&self.neck, include_neck) {
            (Some(neck), true) => max_masks(&self.skin, neck),
            _ => Ok(self.skin.clone()),
        }
    }
}

/// Runs the parsers for the skin (`docs/27` 5.2). `None` when the parser finds no skin. A neck
/// parser that fails is logged and ignored.
pub fn skin_parts(
    models: &Models,
    image: &LinearAwg4Image,
    face: &Face,
    full_size_px: (u32, u32),
) -> Result<Option<SkinParts>> {
    let Some(skin) = models
        .parser
        .part_mask(image, face, FacePart::SkinWithoutEars)?
    else {
        return Ok(None);
    };
    let skin = skin.to_full(full_size_px)?;
    let neck = match &models.neck {
        None => None,
        Some(parser) => match parser.part_mask(image, face, FacePart::NeckSkin) {
            Ok(Some(neck)) => Some(neck.to_full(full_size_px)?),
            Ok(None) => None,
            Err(error) => {
                tracing::warn!(%error, "the neck parser failed; going on without the neck");
                None
            }
        },
    };
    Ok(Some(SkinParts { skin, neck }))
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

/// The blemish mask (`docs/27` 5.3): `skin` (with the neck as the caller chose) with a soft hole
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

/// The even-skin mask (`docs/27` 5.5). `skin` carries the neck as the caller chose;
/// `uniformity` is the mesh's `InvertedSkinUniformity` mask; `exclude` is an optional exclusion
/// mask (default none). All masks are full size; `extent_px` is `F`.
pub fn even_skin_mask(
    skin: &Mask,
    uniformity: &Mask,
    exclude: Option<&Mask>,
    extent_px: f64,
) -> Result<Mask> {
    let mut core = mul_inv_masks(skin, uniformity)?;
    if let Some(exclude) = exclude {
        core = mul_inv_masks(&core, exclude)?;
    }
    let core = feather(&core, -0.011 * extent_px)?;
    let edge = feather(skin, -0.065 * extent_px)?;
    mul_masks(&core, &edge)
}

/// The three masks of one face, each cropped to its bounding box; `None` when the mask is all
/// zeros or the parser found no skin.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FaceMasks {
    /// Where blemish removal works.
    pub blemish: Option<BoundedMask>,
    /// Where dark circles works.
    pub dark_circles: Option<BoundedMask>,
    /// Where even skin works.
    pub even_skin: Option<BoundedMask>,
}

/// Builds the masks of one face with the models. A failing beauty-mark detector is logged and
/// ignored. `exclude` is the optional exclusion mask of 5.5 (full size).
pub fn face_masks(
    models: &Models,
    image: &LinearAwg4Image,
    face: &Face,
    settings: &SkinSettings,
    full_size_px: (u32, u32),
    exclude: Option<&Mask>,
) -> Result<FaceMasks> {
    let Some(parts) = skin_parts(models, image, face, full_size_px)? else {
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
    let blemish = blemish_mask(&parts.skin_mask(settings.include_neck_blemish)?, &marks)?;

    let under_eye = models
        .mesh
        .render(face, MeshTable::UnderEye, full_size_px)?
        .to_full(full_size_px)?;
    let eyes = match models.parser.part_mask(image, face, FacePart::Eyes)? {
        Some(eyes) => Some(eyes.to_full(full_size_px)?),
        None => None,
    };
    let dark = dark_circle_mask(&under_eye, &parts.skin, eyes.as_ref(), extent_px)?;

    let uniformity = models
        .mesh
        .render(face, MeshTable::InvertedSkinUniformity, full_size_px)?
        .to_full(full_size_px)?;
    let even = even_skin_mask(
        &parts.skin_mask(settings.include_neck_even_skin)?,
        &uniformity,
        exclude,
        extent_px,
    )?;

    Ok(FaceMasks {
        blemish: BoundedMask::from_full(&blemish),
        dark_circles: BoundedMask::from_full(&dark),
        even_skin: BoundedMask::from_full(&even),
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
    fn skin_union_with_neck() {
        let (image, face, models) = scene();
        let parts = skin_parts(&models, &image, &face, FULL).unwrap().unwrap();
        assert!(parts.neck.is_some());
        let without = parts.skin_mask(false).unwrap();
        let with = parts.skin_mask(true).unwrap();
        assert_eq!(without, parts.skin);
        assert!(with.data.iter().zip(&without.data).all(|(a, b)| a >= b));
        assert!(with.data.iter().zip(&without.data).any(|(a, b)| a > b));
        // Pixel just below the face rect: neck only.
        assert_eq!(without.get(90, 135), 0);
        assert_eq!(with.get(90, 135), 255);
    }

    #[test]
    fn builders_with_mocks() {
        let (image, face, models) = scene();
        let settings = SkinSettings {
            blemish: 50.0,
            dark_circles: 50.0,
            even_skin: 50.0,
            include_neck_blemish: true,
            ..Default::default()
        };
        let masks = face_masks(&models, &image, &face, &settings, FULL, None).unwrap();
        let blemish = masks.blemish.unwrap();
        let dark = masks.dark_circles.unwrap();
        let even = masks.even_skin.unwrap();
        for m in [&blemish, &dark, &even] {
            m.validate().unwrap();
            assert!(!m.mask.is_all_zero());
        }
        let full = blemish.to_full(FULL).unwrap();
        assert_eq!(full.get(89, 74), 0, "the beauty mark is a hole");
        assert_eq!(full.get(90, 132), 255, "the neck is included");
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
        // The neck flag for even skin is separate.
        let with_neck = SkinSettings {
            include_neck_even_skin: true,
            ..settings
        };
        let masks2 = face_masks(&models, &image, &face, &with_neck, FULL, None).unwrap();
        assert!(masks2.even_skin.unwrap().region.height_px >= even.region.height_px);
        assert_eq!(masks2.dark_circles, Some(dark));
    }

    #[test]
    fn exclusion_removes_even_skin() {
        let (image, face, models) = scene();
        let settings = SkinSettings::default();
        let base = face_masks(&models, &image, &face, &settings, FULL, None)
            .unwrap()
            .even_skin
            .unwrap();
        let all = Mask::new(FULL.0, FULL.1, vec![255; (FULL.0 * FULL.1) as usize]).unwrap();
        let none = face_masks(&models, &image, &face, &settings, FULL, Some(&all)).unwrap();
        assert!(none.even_skin.is_none());
        assert!(base.region.width_px > 0);
    }

    #[test]
    fn missing_parts_and_failures() {
        let (image, face, mut models) = scene();
        models.neck = None;
        models.beauty_marks = Some(Box::new(mock::FailingBeautyMarks));
        let settings = SkinSettings {
            include_neck_blemish: true,
            ..Default::default()
        };
        let masks = face_masks(&models, &image, &face, &settings, FULL, None).unwrap();
        assert_eq!(
            masks.blemish.unwrap().to_full(FULL).unwrap().get(90, 132),
            0,
            "no neck parser, no neck"
        );
        models.parser = Box::new(mock::NoParts);
        assert_eq!(
            face_masks(&models, &image, &face, &settings, FULL, None).unwrap(),
            FaceMasks::default()
        );
    }

    #[test]
    fn builders_do_not_depend_on_threads() {
        let (image, face, models) = scene();
        let settings = SkinSettings {
            include_neck_even_skin: true,
            ..Default::default()
        };
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| face_masks(&models, &image, &face, &settings, FULL, None).unwrap())
        };
        assert_eq!(run(1), run(4));
    }
}
