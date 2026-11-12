//! Show skin areas (`docs/27-skin-retouch.md` S-11c): one face's areas tinted over a finished viewport frame.
//!
//! The areas are the masks the analysis already holds (`FaceMasks`, made once per photo and key, never by this view), so
//! nothing runs but a pass over the frame's pixels on the CPU. Each pixel of the view is placed in the analysis picture
//! (view pixel, then final image pixel, then analysis pixel) and asks the face's masks for its four values:
//!
//! * the **skin**, with the mouth's inside cut out, tinted [`SKIN`] (the mask accent);
//! * **under the eyes**, tinted [`UNDER_EYES`];
//! * the **teeth**, tinted [`TEETH`] (no tint when the mouth is closed: the mask is empty);
//! * the **core** of even skin and grading, drawn as a [`CORE`] dashed outline.
//!
//! It is a view and nothing else: the settings, the grade and the cache of the photo's own frames are not touched.

use ym_retouch::SkinAreaValues;

use crate::pipeline::geometry::Affine;
use crate::pipeline::graph::FaceEntry;

/// The skin: the mask accent, `#ff6fd2`.
pub const SKIN: [u8; 3] = [0xff, 0x6f, 0xd2];
/// Under the eyes: a clear blue that reads on warm skin and beside the pink, `#3d9bff`.
pub const UNDER_EYES: [u8; 3] = [0x3d, 0x9b, 0xff];
/// The teeth: a violet, `#8a5cff`, apart from the pink of the lips and the blue.
pub const TEETH: [u8; 3] = [0x8a, 0x5c, 0xff];
/// The outline of the even-skin and grading core: a green, `#8ee56b`.
pub const CORE: [u8; 3] = [0x8e, 0xe5, 0x6b];

/// How strongly a full mask tints (0 to 1): the skin lightly so that the photo shows, the small areas more.
const SKIN_ALPHA: f32 = 0.45;
const UNDER_EYES_ALPHA: f32 = 0.6;
const TEETH_ALPHA: f32 = 0.65;
/// The dashes of the outline, in view pixels along a diagonal count.
const DASH_PX: i64 = 6;

fn blend(px: &mut [u8], tint: [u8; 3], alpha: f32) {
    if alpha <= 0.0 {
        return;
    }
    for (c, t) in px.iter_mut().zip(tint) {
        *c = (f32::from(*c) + (f32::from(t) - f32::from(*c)) * alpha).round() as u8;
    }
}

/// Tints face `number` (1-based) of `entry` over `rgba` (`w * h` pixels, 4 bytes each, row-major), whose pixel `(x, y)` is at
/// `to_final` of `(x + 0.5, y + 0.5)` in pixels of the final image of size `final_px`. Returns whether anything was drawn
/// (`false` for a number the analysis does not have).
pub(crate) fn tint_skin_areas(
    rgba: &mut [u8],
    (w, h): (usize, usize),
    to_final: &Affine,
    final_px: (usize, usize),
    entry: &FaceEntry,
    number: u32,
) -> bool {
    let Some(masks) = number
        .checked_sub(1)
        .and_then(|i| entry.retouch.masks(i as usize))
    else {
        return false;
    };
    let Some((bx0, by0, bx1, by1)) = masks.areas_bounds() else {
        return false;
    };
    let (ax, ay) = (
        f64::from(entry.analysis_px.0) / final_px.0.max(1) as f64,
        f64::from(entry.analysis_px.1) / final_px.1.max(1) as f64,
    );
    let at = |x: i64, y: i64| -> SkinAreaValues {
        let (fx, fy) = to_final.apply(x as f64 + 0.5, y as f64 + 0.5);
        masks.areas_at((fx * ax).floor() as i64, (fy * ay).floor() as i64)
    };
    // The face's rectangle in analysis pixels, grown by the outline's reach, to skip the pixels no area can be in.
    let (gx0, gy0, gx1, gy1) = (bx0 - 2, by0 - 2, bx1 + 2, by1 + 2);
    let mut drawn = false;
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = to_final.apply(x as f64 + 0.5, y as f64 + 0.5);
            let (px, py) = ((fx * ax).floor() as i64, (fy * ay).floor() as i64);
            if px < gx0 || py < gy0 || px >= gx1 || py >= gy1 {
                continue;
            }
            let v = masks.areas_at(px, py);
            let pixel = &mut rgba[(y * w + x) * 4..][..4];
            let (xi, yi) = (x as i64, y as i64);
            blend(
                &mut pixel[..3],
                SKIN,
                SKIN_ALPHA * f32::from(v.skin) / 255.0,
            );
            blend(
                &mut pixel[..3],
                UNDER_EYES,
                UNDER_EYES_ALPHA * f32::from(v.under_eyes) / 255.0,
            );
            blend(
                &mut pixel[..3],
                TEETH,
                TEETH_ALPHA * f32::from(v.teeth) / 255.0,
            );
            drawn |= v.skin > 0 || v.under_eyes > 0 || v.teeth > 0;
            // The core's edge: inside it, with a neighbour outside, on the dashes.
            if v.core >= 128
                && (xi + yi).div_euclid(DASH_PX) % 2 == 0
                && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| at(xi + dx, yi + dy).core < 128)
            {
                blend(&mut pixel[..3], CORE, 1.0);
                drawn = true;
            }
        }
    }
    drawn
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ym_retouch::mock::{eye_centers, mouth_ellipse};
    use ym_retouch::{Face, LinearAwg4Image, Rect, RetouchCache, SkinSettings, prepare_skin};

    use super::*;
    use crate::pipeline::graph::{FaceEntry, FacesKey, FoundFace};

    const SIZE: (usize, usize) = (400, 400);
    const GREY: u8 = 100;

    fn entry(face: Face) -> FaceEntry {
        let image = LinearAwg4Image::new(
            SIZE.0 as u32,
            SIZE.1 as u32,
            vec![[0.18f32; 3]; SIZE.0 * SIZE.1],
        )
        .expect("an image");
        let mut cache = RetouchCache::new(1, 0);
        prepare_skin(
            &image,
            &[(face, SkinSettings::default())],
            &ym_retouch::mock::models(Vec::new()),
            &mut cache,
        )
        .expect("masks");
        FaceEntry {
            key: FacesKey {
                text: "t".to_owned(),
                frame: "f".to_owned(),
            },
            hash: 1,
            analysis_px: (SIZE.0 as u32, SIZE.1 as u32),
            faces: vec![FoundFace { number: 1, face }],
            timings: Default::default(),
            srgb8: Arc::new(Vec::new()),
            retouch: cache,
        }
    }

    fn canvas() -> Vec<u8> {
        vec![GREY; SIZE.0 * SIZE.1 * 4]
    }

    fn px(rgba: &[u8], x: usize, y: usize) -> [u8; 3] {
        let i = (y * SIZE.0 + x) * 4;
        [rgba[i], rgba[i + 1], rgba[i + 2]]
    }

    fn face() -> Face {
        Face {
            rect: Rect::new(100, 60, 200, 260),
            scale: 0.65,
        }
    }

    fn tinted(number: u32) -> (Vec<u8>, bool) {
        let e = entry(face());
        let mut rgba = canvas();
        let drawn = tint_skin_areas(&mut rgba, SIZE, &Affine::IDENTITY, SIZE, &e, number);
        (rgba, drawn)
    }

    #[test]
    fn each_area_gets_its_own_tint() {
        let (rgba, drawn) = tinted(1);
        assert!(drawn);
        let r = face().rect;
        // The skin on the cheek: pink (red and blue up, green about where it was).
        let cheek = px(&rgba, (r.x_px + 40) as usize, (r.y_px + 150) as usize);
        assert!(
            cheek[0] > GREY + 50 && cheek[2] > GREY + 30 && cheek[1] < GREY + 20,
            "{cheek:?} is pink"
        );
        // Under the left eye: blue on top of the skin's pink (blue well above red).
        let [(lx, ly), _] = eye_centers(&r);
        let eye = px(
            &rgba,
            lx as usize,
            (ly + 0.09 * f64::from(r.height_px)) as usize,
        );
        assert!(
            eye[2] > eye[0] + 60 && eye[2] > eye[1] + 40,
            "{eye:?} is blue"
        );
        // The mouth: the skin is cut out and the teeth are violet (blue over red over green, not pink).
        let (cx, cy, _, _) = mouth_ellipse(&r);
        let mouth = px(&rgba, cx as usize, cy as usize);
        assert!(
            mouth[2] > mouth[0] + 50 && mouth[0] > mouth[1] + 15,
            "{mouth:?} is violet"
        );
        // Outside the face: untouched.
        assert_eq!(px(&rgba, 10, 10), [GREY; 3]);
        assert_eq!(px(&rgba, 380, 380), [GREY; 3]);
        // The alpha stays as it was.
        assert!(rgba.chunks_exact(4).all(|p| p[3] == GREY));
    }

    #[test]
    fn the_core_is_a_dashed_outline_on_the_gradings_edge() {
        let (rgba, _) = tinted(1);
        let e = entry(face());
        let masks = e.retouch.masks(0).expect("masks");
        let mut outline = 0;
        let mut on_core = 0;
        for y in 0..SIZE.1 {
            for x in 0..SIZE.0 {
                if px(&rgba, x, y) == CORE {
                    outline += 1;
                    on_core += usize::from(masks.areas_at(x as i64, y as i64).core >= 128);
                }
            }
        }
        assert!(outline > 100, "an outline is drawn: {outline}");
        assert_eq!(outline, on_core, "it is drawn inside the core's edge");
        // Dashed: a full outline would be about twice as long.
        let (mut inside, mut edge) = (0usize, 0usize);
        for y in 1..SIZE.1 - 1 {
            for x in 1..SIZE.0 - 1 {
                let c = |x: usize, y: usize| masks.areas_at(x as i64, y as i64).core >= 128;
                if c(x, y) && !(c(x + 1, y) && c(x - 1, y) && c(x, y + 1) && c(x, y - 1)) {
                    edge += 1;
                }
                inside += usize::from(c(x, y));
            }
        }
        assert!(
            inside > 0 && outline < edge,
            "{outline} dashes of {edge} edge pixels"
        );
    }

    #[test]
    fn a_number_the_photo_does_not_have_draws_nothing() {
        for number in [0, 2, 99] {
            let (rgba, drawn) = tinted(number);
            assert!(!drawn, "face {number}");
            assert_eq!(rgba, canvas(), "face {number}");
        }
    }

    #[test]
    fn the_view_follows_the_zoom() {
        // The same face seen at twice the size: view pixel (x, y) is final pixel (x / 2, y / 2).
        let e = entry(face());
        let mut rgba = vec![GREY; 2 * SIZE.0 * 2 * SIZE.1 * 4];
        let drawn = tint_skin_areas(
            &mut rgba,
            (2 * SIZE.0, 2 * SIZE.1),
            &Affine::scale_translate(0.5, 0.5, 0.0, 0.0),
            SIZE,
            &e,
            1,
        );
        assert!(drawn);
        let r = face().rect;
        let (cx, cy, _, _) = mouth_ellipse(&r);
        let i = ((2.0 * cy) as usize * 2 * SIZE.0 + (2.0 * cx) as usize) * 4;
        assert!(
            rgba[i + 2] > rgba[i] + 50,
            "teeth are violet at twice the size"
        );
    }
}
