//! The entry point of `docs/27` section 8: masks and blemish maps per face (cached), blemish apply
//! for every face, then dark circles and even skin for every face.
//!
//! Masks and blemish maps are built from the full-resolution image and kept in a
//! [`RetouchCache`]. A preview works on a smaller image: call [`prepare_skin`] once with the
//! full-resolution image, then [`retouch_skin`] on each working image, which only reads the cache
//! (`docs/27` section 9, decision 9). An export passes the full-resolution image to
//! [`retouch_skin`] directly and the cache fills itself.

use crate::blemish::{BlemishMap, BlemishMapCache, BlemishMapKey, apply_blemish, blemish_map};
use crate::error::{Result, RetouchError};
use crate::lowfreq::retouch_low_frequency;
use crate::mask::{FaceMasks, face_masks};
use crate::models::Models;
use crate::types::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, check_render_scale, face_size_px,
};

/// What the masks of one face depend on besides the image: the face and the two neck switches.
#[derive(Debug, Clone, PartialEq)]
struct FaceEntry {
    face: Face,
    include_neck_blemish: bool,
    include_neck_even_skin: bool,
}

impl FaceEntry {
    fn of(face: &Face, settings: &SkinSettings) -> Self {
        Self {
            face: *face,
            include_neck_blemish: settings.include_neck_blemish,
            include_neck_even_skin: settings.include_neck_even_skin,
        }
    }
}

#[derive(Debug)]
struct PreparedMasks {
    full_size_px: (u32, u32),
    entries: Vec<FaceEntry>,
    masks: Vec<FaceMasks>,
}

/// The masks per face and the blemish maps per face of one image (`docs/27` section 8).
///
/// Both depend on the image and on everything upstream of it, not on the amounts. The caller names
/// the image with two opaque keys ([`RetouchCache::set_keys`]); a different key drops everything.
/// Masks are rebuilt when the faces, the full size or a neck switch change; a blemish map is rebuilt
/// when the faces or the blemish neck switch change.
#[derive(Debug, Default)]
pub struct RetouchCache {
    image_key: u64,
    upstream_key: u64,
    prepared: Option<PreparedMasks>,
    maps: BlemishMapCache,
}

impl RetouchCache {
    /// An empty cache for the image named by `image_key` and `upstream_key`.
    pub fn new(image_key: u64, upstream_key: u64) -> Self {
        Self {
            image_key,
            upstream_key,
            ..Self::default()
        }
    }

    /// Names the image. The cache keeps its content only when both keys are unchanged.
    pub fn set_keys(&mut self, image_key: u64, upstream_key: u64) {
        if (self.image_key, self.upstream_key) != (image_key, upstream_key) {
            self.clear();
            self.image_key = image_key;
            self.upstream_key = upstream_key;
        }
    }

    /// Drops every mask and map; the keys stay.
    pub fn clear(&mut self) {
        self.prepared = None;
        self.maps.clear();
    }

    /// The number of cached blemish maps.
    pub fn map_count(&self) -> usize {
        self.maps.len()
    }

    /// True when masks are cached.
    pub fn has_masks(&self) -> bool {
        self.prepared.is_some()
    }

    /// The cached masks of face `index`, if any.
    pub fn masks(&self, index: usize) -> Option<&FaceMasks> {
        self.prepared.as_ref().and_then(|p| p.masks.get(index))
    }

    fn masks_match(&self, full_size_px: (u32, u32), entries: &[FaceEntry]) -> bool {
        self.prepared
            .as_ref()
            .is_some_and(|p| p.full_size_px == full_size_px && p.entries == entries)
    }

    /// The map key: the image, the faces and the one switch the blemish mask depends on, folded
    /// into the upstream key.
    fn map_key(&self, entries: &[FaceEntry]) -> BlemishMapKey {
        let mut upstream = self.upstream_key ^ 0xcbf2_9ce4_8422_2325;
        for e in entries {
            upstream =
                (upstream ^ u64::from(e.include_neck_blemish)).wrapping_mul(0x0100_0000_01b3);
        }
        BlemishMapKey {
            image_key: self.image_key,
            face_count: entries.len(),
            face_rects: entries.iter().map(|e| e.face.rect).collect(),
            upstream_key: upstream,
        }
    }
}

/// Builds and caches the masks and the blemish maps of `faces` from the full-resolution image, so
/// that [`retouch_skin`] can then run on previews of it. Does nothing for what is already cached.
///
/// Maps are built for every face with a blemish mask, whatever its Blemish amount, because they do
/// not depend on the amount.
pub fn prepare_skin(
    full_image: &LinearAwg4Image,
    faces: &[(Face, SkinSettings)],
    models: &Models,
    cache: &mut RetouchCache,
) -> Result<()> {
    full_image.validate()?;
    let full_size = (full_image.width_px, full_image.height_px);
    let entries = entries_of(faces);
    ensure_masks(Some(full_image), full_size, faces, &entries, models, cache)?;
    let wanted: Vec<bool> = vec![true; faces.len()];
    ensure_maps(
        Some(full_image),
        full_size,
        faces,
        &entries,
        &wanted,
        models,
        cache,
    )?;
    Ok(())
}

fn entries_of(faces: &[(Face, SkinSettings)]) -> Vec<FaceEntry> {
    faces.iter().map(|(f, s)| FaceEntry::of(f, s)).collect()
}

fn ensure_masks(
    full_image: Option<&LinearAwg4Image>,
    full_size: (u32, u32),
    faces: &[(Face, SkinSettings)],
    entries: &[FaceEntry],
    models: &Models,
    cache: &mut RetouchCache,
) -> Result<()> {
    if cache.masks_match(full_size, entries) {
        return Ok(());
    }
    let Some(image) = full_image else {
        return Err(RetouchError::NotPrepared);
    };
    let masks = faces
        .iter()
        .map(|(face, settings)| face_masks(models, image, face, settings, full_size, None))
        .collect::<Result<Vec<_>>>()?;
    cache.prepared = Some(PreparedMasks {
        full_size_px: full_size,
        entries: entries.to_vec(),
        masks,
    });
    Ok(())
}

/// The rectangle a face's blemish map covers: the face rectangle joined with the blemish mask's
/// region (the neck lies below the face), clipped to the image. `None` when there is nothing.
fn map_rect(face: &Face, masks: &FaceMasks, full_size: (u32, u32)) -> Option<Rect> {
    let bm = masks.blemish.as_ref()?;
    let r = face.rect;
    let x0 = i64::from(r.x_px).min(i64::from(bm.region.x_px)).max(0);
    let y0 = i64::from(r.y_px).min(i64::from(bm.region.y_px)).max(0);
    let x1 = r
        .right_px()
        .max(bm.region.right_px())
        .min(i64::from(full_size.0));
    let y1 = r
        .bottom_px()
        .max(bm.region.bottom_px())
        .min(i64::from(full_size.1));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    Some(Rect::new(
        x0 as i32,
        y0 as i32,
        (x1 - x0) as u32,
        (y1 - y0) as u32,
    ))
}

fn crop_image(image: &LinearAwg4Image, rect: Rect) -> Result<LinearAwg4Image> {
    let (w, h) = (rect.width_px as usize, rect.height_px as usize);
    let iw = image.width_px as usize;
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        let start = (rect.y_px as usize + y) * iw + rect.x_px as usize;
        data.extend_from_slice(&image.data[start..start + w]);
    }
    LinearAwg4Image::new(rect.width_px, rect.height_px, data)
}

/// The bounded mask over `rect` as a plain mask of the rectangle's size (0 outside the region).
fn crop_mask(bm: &BoundedMask, rect: Rect) -> Result<Mask> {
    let mut out = Mask::zeros(rect.width_px, rect.height_px)?;
    let (mw, ow) = (bm.mask.width_px as usize, rect.width_px as usize);
    let x0 = i64::from(rect.x_px).max(i64::from(bm.region.x_px));
    let x1 = rect.right_px().min(bm.region.right_px());
    let y0 = i64::from(rect.y_px).max(i64::from(bm.region.y_px));
    let y1 = rect.bottom_px().min(bm.region.bottom_px());
    if x0 >= x1 || y0 >= y1 {
        return Ok(out);
    }
    let len = (x1 - x0) as usize;
    for y in y0..y1 {
        let src = (y - i64::from(bm.region.y_px)) as usize * mw
            + (x0 - i64::from(bm.region.x_px)) as usize;
        let dst = (y - i64::from(rect.y_px)) as usize * ow + (x0 - i64::from(rect.x_px)) as usize;
        out.data[dst..dst + len].copy_from_slice(&bm.mask.data[src..src + len]);
    }
    Ok(out)
}

fn build_map(
    full_image: &LinearAwg4Image,
    face: &Face,
    rect: Rect,
    masks: &FaceMasks,
    models: &Models,
) -> Result<BlemishMap> {
    let Some(bm) = masks.blemish.as_ref() else {
        return Err(RetouchError::NotPrepared);
    };
    let crop = crop_image(full_image, rect)?;
    let mask = crop_mask(bm, rect)?;
    let full = (full_image.width_px, full_image.height_px);
    let face_px = face_size_px(full, 1.0, face)?;
    blemish_map(
        &crop,
        &mask,
        rect,
        [1.0; 3],
        face_px,
        models.proposer.as_ref(),
    )
}

/// Makes sure the maps of the faces flagged in `wanted` are cached. Without the full-resolution
/// image a missing map is an error.
fn ensure_maps(
    full_image: Option<&LinearAwg4Image>,
    full_size: (u32, u32),
    faces: &[(Face, SkinSettings)],
    entries: &[FaceEntry],
    wanted: &[bool],
    models: &Models,
    cache: &mut RetouchCache,
) -> Result<()> {
    let key = cache.map_key(entries);
    for (i, (face, _)) in faces.iter().enumerate() {
        if !wanted[i] {
            continue;
        }
        let Some(masks) = cache.prepared.as_ref().and_then(|p| p.masks.get(i)) else {
            return Err(RetouchError::NotPrepared);
        };
        let Some(rect) = map_rect(face, masks, full_size) else {
            continue;
        };
        if cache.maps.get(&key, i).is_some() {
            continue;
        }
        let Some(image) = full_image else {
            return Err(RetouchError::NotPrepared);
        };
        let masks = masks.clone();
        cache
            .maps
            .get_or_build(&key, i, || build_map(image, face, rect, &masks, models))?;
    }
    Ok(())
}

/// Retouches the faces of the working image (`docs/27` section 8) and returns a new image.
///
/// - `image`: the working image. Its size must be `full_size_px * render_scale` within 2 px.
/// - `full_size_px`: the full-resolution size; masks and blemish maps are built at that size.
/// - `faces`: each face with its settings (clamped on entry). Rectangles are in full-resolution
///   pixels.
/// - `render_scale`: working size over full size, 0.0001 to 1.
/// - `models`, `cache`: the models and the cache of this image ([`RetouchCache`]).
/// - `protect`: an optional protection mask in working pixels (the size of `image`); 255 leaves a
///   pixel out of blemish removal.
///
/// Order: output is a copy of the input; blemish removal for every face; then dark circles and
/// even skin for every face. With every amount 0 the output equals the input bit for bit.
///
/// The masks and maps are taken from the cache. When something is missing and `image` is the
/// full-resolution image, they are built from it and cached; when `image` is a preview, the call
/// fails with [`RetouchError::NotPrepared`] unless [`prepare_skin`] ran first.
pub fn retouch_skin(
    image: &LinearAwg4Image,
    full_size_px: (u32, u32),
    faces: &[(Face, SkinSettings)],
    render_scale: f32,
    models: &Models,
    cache: &mut RetouchCache,
    protect: Option<&Mask>,
) -> Result<LinearAwg4Image> {
    image.validate()?;
    let scale = check_render_scale(render_scale)?;
    check_working_size(image, full_size_px, render_scale, scale)?;
    if let Some(p) = protect {
        p.validate()?;
        if p.size_px() != (image.width_px, image.height_px) {
            return Err(RetouchError::MaskSizeMismatch {
                left: (image.width_px, image.height_px),
                right: p.size_px(),
            });
        }
    }
    let faces: Vec<(Face, SkinSettings)> = faces.iter().map(|(f, s)| (*f, s.clamped())).collect();
    let mut output = image.clone();
    let wants_blemish: Vec<bool> = faces.iter().map(|(_, s)| s.blemish > 0.0).collect();
    let wants_low = faces
        .iter()
        .any(|(_, s)| s.dark_circles > 0.0 || s.even_skin > 0.0);
    if !wants_blemish.iter().any(|&b| b) && !wants_low {
        return Ok(output);
    }

    let full_image = ((image.width_px, image.height_px) == full_size_px).then_some(image);
    let entries = entries_of(&faces);
    ensure_masks(full_image, full_size_px, &faces, &entries, models, cache)?;
    ensure_maps(
        full_image,
        full_size_px,
        &faces,
        &entries,
        &wants_blemish,
        models,
        cache,
    )?;

    let key = cache.map_key(&entries);
    for (i, (_, settings)) in faces.iter().enumerate() {
        if !wants_blemish[i] {
            continue;
        }
        if let Some(map) = cache.maps.get(&key, i) {
            apply_blemish(&mut output, &map, settings.a(), render_scale, protect)?;
        }
    }
    if wants_low {
        for (i, (face, settings)) in faces.iter().enumerate() {
            let Some(masks) = cache.masks(i) else {
                continue;
            };
            retouch_low_frequency(
                &mut output,
                full_size_px,
                face,
                settings,
                render_scale,
                masks.dark_circles.as_ref(),
                masks.even_skin.as_ref(),
            )?;
        }
    }
    Ok(output)
}

fn check_working_size(
    image: &LinearAwg4Image,
    full_size_px: (u32, u32),
    render_scale: f32,
    scale: f64,
) -> Result<()> {
    let expected = |side: u32| (f64::from(side) * scale).round().max(1.0);
    let (ew, eh) = (expected(full_size_px.0), expected(full_size_px.1));
    if full_size_px.0 == 0
        || full_size_px.1 == 0
        || (f64::from(image.width_px) - ew).abs() > 2.0
        || (f64::from(image.height_px) - eh).abs() > 2.0
    {
        return Err(RetouchError::WorkingSize {
            image: (image.width_px, image.height_px),
            full: full_size_px,
            scale: render_scale,
            expected: (ew as u32, eh as u32),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::mock::{self, MockProposer};

    fn hash(x: u32, y: u32) -> f32 {
        let mut h = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca6b);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
        (h & 0xffff) as f32 / 65535.0 - 0.5
    }

    /// A smooth skin-coloured picture with a gradient, fine noise, spots and dark patches.
    fn synthetic(w: u32, h: u32, face: &Face) -> LinearAwg4Image {
        let r = face.rect;
        let (cx, cy) = (
            f64::from(r.x_px) + f64::from(r.width_px) / 2.0,
            f64::from(r.y_px) + f64::from(r.height_px) / 2.0,
        );
        let unit = f64::from(r.width_px) / 500.0;
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (f64::from(x), f64::from(y));
                let mut luma = 0.28 + 0.1 * (fx / f64::from(w)) + 0.05 * (fy / f64::from(h));
                luma += 0.01 * f64::from(hash(x, y));
                for (sx, sy) in [(-80.0, 60.0), (60.0, -20.0), (20.0, 120.0)] {
                    let d = ((fx - cx - sx * unit).powi(2) + (fy - cy - sy * unit).powi(2)).sqrt();
                    luma *= 1.0 - 0.25 * (-(d / (9.0 * unit)).powi(2)).exp();
                }
                let d = ((fx - cx + 80.0 * unit).powi(2) + (fy - cy - 60.0 * unit - 40.0).powi(2))
                    .sqrt();
                luma *= 1.0 - 0.2 * (-(d / (40.0 * unit)).powi(2)).exp();
                let l = luma as f32;
                data.push([l * 1.0, l * 0.78, l * 0.62]);
            }
        }
        LinearAwg4Image::new(w, h, data).unwrap()
    }

    fn settings(blemish: f32, dark: f32, even: f32) -> SkinSettings {
        SkinSettings {
            blemish,
            dark_circles: dark,
            even_skin: even,
            texture: 0.0,
            include_neck_blemish: true,
            include_neck_even_skin: true,
        }
    }

    fn test_models(proposer: MockProposer) -> Models {
        Models {
            proposer: Box::new(proposer),
            ..mock::models(vec![Rect::new(560, 300, 24, 24)])
        }
    }

    fn big_face() -> Face {
        Face {
            rect: Rect::new(350, 60, 500, 600),
            scale: 600.0 / 1200.0,
        }
    }

    fn downscale2(img: &LinearAwg4Image) -> LinearAwg4Image {
        let (w, h) = (img.width_px as usize / 2, img.height_px as usize / 2);
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f64; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = img.data[(2 * y + dy) * img.width_px as usize + 2 * x + dx];
                    for c in 0..3 {
                        acc[c] += f64::from(p[c]) / 4.0;
                    }
                }
                data.push([acc[0] as f32, acc[1] as f32, acc[2] as f32]);
            }
        }
        LinearAwg4Image::new(w as u32, h as u32, data).unwrap()
    }

    fn bits(img: &LinearAwg4Image) -> Vec<u32> {
        img.data.iter().flatten().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn identity_with_every_amount_zero() {
        let face = big_face();
        let img = synthetic(1200, 900, &face);
        let models = test_models(MockProposer::scale(192, 0.9));
        let mut cache = RetouchCache::new(1, 1);
        let faces = [(face, settings(0.0, 0.0, 0.0))];
        let out = retouch_skin(&img, (1200, 900), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_eq!(bits(&out), bits(&img));
        assert!(!cache.has_masks(), "nothing to do builds nothing");
        // Out-of-range and NaN amounts clamp to 0 or 100; a negative one is 0.
        let faces = [(
            face,
            SkinSettings {
                blemish: -5.0,
                dark_circles: f32::NAN,
                ..settings(0.0, 0.0, 0.0)
            },
        )];
        let out = retouch_skin(&img, (1200, 900), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_eq!(bits(&out), bits(&img));
        // No faces.
        let out = retouch_skin(&img, (1200, 900), &[], 1.0, &models, &mut cache, None).unwrap();
        assert_eq!(out, img);
    }

    fn working_union_mask(cache: &RetouchCache, w: u32, h: u32) -> Mask {
        let mut union = Mask::zeros(w, h).unwrap();
        let m = cache.masks(0).unwrap();
        for bm in [&m.blemish, &m.dark_circles, &m.even_skin]
            .into_iter()
            .flatten()
        {
            let full = bm.to_full((w, h)).unwrap();
            for (u, v) in union.data.iter_mut().zip(&full.data) {
                *u = (*u).max(*v);
            }
        }
        union
    }

    #[test]
    fn mask_locality_through_the_whole_pipeline() {
        let face = big_face();
        let img = synthetic(1200, 900, &face);
        let faces = [(face, settings(60.0, 60.0, 60.0))];

        // Identity proposer: the blemish gain is 1 everywhere; outside every mask, no pixel moves.
        let models = test_models(MockProposer::identity(192));
        let mut cache = RetouchCache::new(1, 1);
        let out = retouch_skin(&img, (1200, 900), &faces, 1.0, &models, &mut cache, None).unwrap();
        let union = working_union_mask(&cache, 1200, 900);
        let mut changed_inside = 0;
        for (i, (a, b)) in img.data.iter().zip(&out.data).enumerate() {
            if union.data[i] == 0 {
                assert_eq!(
                    a.map(f32::to_bits),
                    b.map(f32::to_bits),
                    "pixel {i} outside every mask"
                );
            } else if a != b {
                changed_inside += 1;
            }
        }
        assert!(
            changed_inside > 1000,
            "the tools did something: {changed_inside}"
        );

        // An active proposer: the blemish map is sampled bilinearly at a coarser grid, so the
        // gain may leak into the pixels next to the mask; everything 3 px from it stays exact.
        let models = test_models(MockProposer::scale(192, 0.9));
        let mut cache = RetouchCache::new(1, 1);
        let out = retouch_skin(&img, (1200, 900), &faces, 1.0, &models, &mut cache, None).unwrap();
        let union = working_union_mask(&cache, 1200, 900);
        let near =
            |x: i64, y: i64| (-3..=3).any(|dy| (-3..=3).any(|dx| union.get(x + dx, y + dy) != 0));
        let mut moved = 0;
        for y in 0..900i64 {
            for x in 0..1200i64 {
                let i = (y * 1200 + x) as usize;
                if img.data[i] != out.data[i] {
                    moved += 1;
                    assert!(near(x, y), "pixel ({x}, {y}) moved far from every mask");
                }
            }
        }
        assert!(moved > 1000);
    }

    #[test]
    fn scale_consistency_between_export_and_preview() {
        let face = big_face();
        let full = synthetic(1200, 900, &face);
        let half = downscale2(&full);
        let models = test_models(MockProposer::scale(192, 0.9));
        let faces = [(face, settings(60.0, 60.0, 60.0))];

        let mut cache_full = RetouchCache::new(7, 0);
        let out_full = retouch_skin(
            &full,
            (1200, 900),
            &faces,
            1.0,
            &models,
            &mut cache_full,
            None,
        )
        .unwrap();

        // The preview never sees the full image except through `prepare_skin`.
        let mut cache_half = RetouchCache::new(7, 0);
        let early = retouch_skin(
            &half,
            (1200, 900),
            &faces,
            0.5,
            &models,
            &mut cache_half,
            None,
        );
        assert_eq!(early.unwrap_err(), RetouchError::NotPrepared);
        prepare_skin(&full, &faces, &models, &mut cache_half).unwrap();
        assert_eq!(cache_half.map_count(), 1);
        let out_half = retouch_skin(
            &half,
            (1200, 900),
            &faces,
            0.5,
            &models,
            &mut cache_half,
            None,
        )
        .unwrap();

        let reduced = downscale2(&out_full);
        assert_eq!((reduced.width_px, reduced.height_px), (600, 450));
        let (mut diff, mut sum) = (0.0f64, 0.0f64);
        let mut moved = 0.0f64;
        for ((a, b), o) in reduced.data.iter().zip(&out_half.data).zip(&half.data) {
            for c in 0..3 {
                diff += f64::from((a[c] - b[c]).abs());
                sum += f64::from(b[c]);
                moved += f64::from((b[c] - o[c]).abs());
            }
        }
        assert!(moved > 1.0, "the retouch changed the preview: {moved}");
        let ratio = diff / sum;
        assert!(ratio < 0.01, "mean absolute difference is {ratio}");
        // The preview path matches the effect size too: compare against the change itself.
        assert!(
            diff < 0.5 * moved,
            "difference {diff} against change {moved}"
        );
    }

    #[test]
    fn same_bytes_with_one_three_and_four_threads() {
        let face = Face {
            rect: Rect::new(175, 30, 250, 300),
            scale: 0.5,
        };
        let img = synthetic(600, 450, &face);
        let models = test_models(MockProposer::scale(192, 0.9));
        let faces = [(face, settings(60.0, 60.0, 60.0))];
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut cache = RetouchCache::new(3, 3);
                retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap()
            })
        };
        let one = run(1);
        assert_ne!(bits(&one), bits(&img));
        assert_eq!(bits(&one), bits(&run(3)));
        assert_eq!(bits(&one), bits(&run(4)));
    }

    #[test]
    fn cache_reuse_keys_and_switches() {
        let face = Face {
            rect: Rect::new(175, 30, 250, 300),
            scale: 0.5,
        };
        let img = synthetic(600, 450, &face);
        let models = test_models(MockProposer::scale(192, 0.9));
        let mut cache = RetouchCache::new(1, 1);
        let faces = [(face, settings(60.0, 0.0, 0.0))];
        let a = retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_eq!(cache.map_count(), 1);
        // A different amount reuses the map and the result differs.
        let faces2 = [(face, settings(90.0, 0.0, 0.0))];
        let b = retouch_skin(&img, (600, 450), &faces2, 1.0, &models, &mut cache, None).unwrap();
        assert_ne!(bits(&a), bits(&b));
        assert_eq!(cache.map_count(), 1);
        // The same call again is identical.
        let a2 = retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_eq!(bits(&a), bits(&a2));
        // The even-skin neck switch rebuilds masks but keeps the map; the blemish one drops it.
        let mut s = settings(60.0, 0.0, 0.0);
        s.include_neck_even_skin = false;
        let mut other = RetouchCache::new(1, 1);
        prepare_skin(&img, &faces, &models, &mut other).unwrap();
        let masks_before = other.masks(0).unwrap().clone();
        prepare_skin(&img, &[(face, s)], &models, &mut other).unwrap();
        assert_eq!(other.map_count(), 1);
        assert_ne!(other.masks(0).unwrap().even_skin, masks_before.even_skin);
        s.include_neck_blemish = false;
        prepare_skin(&img, &[(face, s)], &models, &mut other).unwrap();
        assert_eq!(other.map_count(), 1);
        assert_ne!(other.masks(0).unwrap().blemish, masks_before.blemish);
        // New keys drop everything.
        other.set_keys(2, 1);
        assert!(!other.has_masks());
        assert_eq!(other.map_count(), 0);
    }

    #[test]
    fn protection_mask_stops_blemish_removal() {
        let face = Face {
            rect: Rect::new(175, 30, 250, 300),
            scale: 0.5,
        };
        let img = synthetic(600, 450, &face);
        let models = test_models(MockProposer::scale(192, 0.9));
        let faces = [(face, settings(100.0, 0.0, 0.0))];
        let mut cache = RetouchCache::new(1, 1);
        let all = Mask::new(600, 450, vec![255; 600 * 450]).unwrap();
        let out = retouch_skin(
            &img,
            (600, 450),
            &faces,
            1.0,
            &models,
            &mut cache,
            Some(&all),
        )
        .unwrap();
        assert_eq!(bits(&out), bits(&img));
        let out = retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_ne!(bits(&out), bits(&img));
        let bad = Mask::zeros(10, 10).unwrap();
        let err = retouch_skin(
            &img,
            (600, 450),
            &faces,
            1.0,
            &models,
            &mut cache,
            Some(&bad),
        );
        assert!(matches!(err, Err(RetouchError::MaskSizeMismatch { .. })));
    }

    #[test]
    fn bad_inputs_are_errors() {
        let face = Face {
            rect: Rect::new(175, 30, 250, 300),
            scale: 0.5,
        };
        let img = synthetic(600, 450, &face);
        let models = test_models(MockProposer::identity(64));
        let mut cache = RetouchCache::default();
        let faces = [(face, settings(50.0, 50.0, 50.0))];
        for scale in [0.0, 1.5, f32::NAN] {
            let r = retouch_skin(&img, (600, 450), &faces, scale, &models, &mut cache, None);
            assert!(matches!(r, Err(RetouchError::RenderScale(_))));
        }
        let r = retouch_skin(&img, (600, 450), &faces, 0.5, &models, &mut cache, None);
        assert!(matches!(r, Err(RetouchError::WorkingSize { .. })));
        let r = retouch_skin(&img, (1200, 900), &faces, 1.0, &models, &mut cache, None);
        assert!(matches!(r, Err(RetouchError::WorkingSize { .. })));
        let broken = LinearAwg4Image {
            width_px: 5,
            height_px: 5,
            data: vec![[0.0; 3]; 3],
        };
        let r = retouch_skin(&broken, (5, 5), &faces, 1.0, &models, &mut cache, None);
        assert!(matches!(r, Err(RetouchError::Size { .. })));
    }

    #[test]
    fn failing_detector_and_missing_neck_are_survivable() {
        let face = Face {
            rect: Rect::new(175, 30, 250, 300),
            scale: 0.5,
        };
        let img = synthetic(600, 450, &face);
        let models = Models {
            beauty_marks: Some(Box::new(mock::FailingBeautyMarks)),
            neck: Some(Box::new(mock::NoParts)),
            ..test_models(MockProposer::scale(192, 0.9))
        };
        let faces = [(face, settings(60.0, 60.0, 60.0))];
        let mut cache = RetouchCache::default();
        let out = retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_ne!(bits(&out), bits(&img));
    }

    #[test]
    fn crop_mask_matches_to_full() {
        let bm = BoundedMask::new(
            Rect::new(-3, 4, 6, 5),
            Mask::new(6, 5, (1..=30).collect()).unwrap(),
        )
        .unwrap();
        let full = bm.to_full((10, 12)).unwrap();
        let rect = Rect::new(0, 2, 8, 6);
        let crop = crop_mask(&bm, rect).unwrap();
        for y in 0..6i64 {
            for x in 0..8i64 {
                assert_eq!(crop.get(x, y), full.get(x, y + 2));
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn output_is_finite_for_any_settings(
            blemish in -20.0f32..140.0,
            dark in -20.0f32..140.0,
            even in -20.0f32..140.0,
            texture in -150.0f32..150.0,
            neck in any::<bool>(),
        ) {
            let face = Face { rect: Rect::new(50, 20, 100, 120), scale: 0.6 };
            let img = synthetic(200, 160, &face);
            let models = test_models(MockProposer::scale(64, 0.9));
            let s = SkinSettings {
                blemish, dark_circles: dark, even_skin: even, texture,
                include_neck_blemish: neck, include_neck_even_skin: neck,
            };
            let mut cache = RetouchCache::default();
            let out = retouch_skin(&img, (200, 160), &[(face, s)], 1.0, &models, &mut cache, None).unwrap();
            prop_assert_eq!((out.width_px, out.height_px), (200, 160));
            prop_assert!(out.data.iter().flatten().all(|v| v.is_finite()));
            if blemish <= 0.0 && dark <= 0.0 && even <= 0.0 {
                prop_assert_eq!(bits(&out), bits(&img));
            }
        }
    }
}
