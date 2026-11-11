//! The entry point of `docs/27` section 8: masks and blemish maps per face (cached), blemish apply
//! for every face, then dark circles and even skin for every face.
//!
//! Masks and blemish maps are built from the full-resolution image and kept in a
//! [`RetouchCache`]. A preview works on a smaller image: call [`prepare_skin`] once with the
//! full-resolution image, then [`retouch_skin`] on each working image, which only reads the cache
//! (`docs/27` section 9, decision 9). An export passes the full-resolution image to
//! [`retouch_skin`] directly and the cache fills itself.

use crate::blemish::{BlemishMap, BlemishMapCache, BlemishMapKey, blemish_map};
use crate::error::{Result, RetouchError};
use std::sync::Arc;

use crate::blemish::apply_blemish_at;
use crate::lowfreq::{Placement, ToolMasks, join_rects, retouch_tones_at, tones_region};
use crate::mask::{FaceMasks, face_masks};
use crate::models::Models;
use crate::types::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, check_render_scale, face_size_px,
};

#[derive(Debug)]
struct PreparedMasks {
    full_size_px: (u32, u32),
    faces: Vec<Face>,
    masks: Vec<FaceMasks>,
}

/// The masks per face and the blemish maps per face of one image (`docs/27` section 8).
///
/// Both depend on the image and on everything upstream of it, not on the amounts. The caller names
/// the image with two opaque keys ([`RetouchCache::set_keys`]); a different key drops everything.
/// Masks and blemish maps are rebuilt when the faces or the full size change, and never for a
/// change of amounts.
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

    /// The cached map of face `index` under the current faces, if there is one.
    fn map_of(&self, index: usize) -> Option<Arc<BlemishMap>> {
        let faces: Vec<Face> = self.prepared.as_ref()?.faces.clone();
        self.maps.get(&self.map_key(&faces), index)
    }

    fn masks_match(&self, full_size_px: (u32, u32), faces: &[Face]) -> bool {
        self.prepared
            .as_ref()
            .is_some_and(|p| p.full_size_px == full_size_px && p.faces == faces)
    }

    /// The map key: the image, the faces and everything upstream.
    fn map_key(&self, faces: &[Face]) -> BlemishMapKey {
        BlemishMapKey {
            image_key: self.image_key,
            face_count: faces.len(),
            face_rects: faces.iter().map(|f| f.rect).collect(),
            upstream_key: self.upstream_key,
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
    let face_list = faces_of(faces);
    ensure_masks(Some(full_image), full_size, &face_list, models, cache)?;
    let wanted: Vec<bool> = vec![true; faces.len()];
    ensure_maps(
        Some(full_image),
        full_size,
        &face_list,
        &wanted,
        models,
        cache,
    )?;
    Ok(())
}

fn faces_of(faces: &[(Face, SkinSettings)]) -> Vec<Face> {
    faces.iter().map(|(f, _)| *f).collect()
}

fn ensure_masks(
    full_image: Option<&LinearAwg4Image>,
    full_size: (u32, u32),
    faces: &[Face],
    models: &Models,
    cache: &mut RetouchCache,
) -> Result<()> {
    if cache.masks_match(full_size, faces) {
        return Ok(());
    }
    let Some(image) = full_image else {
        return Err(RetouchError::NotPrepared);
    };
    let masks = faces
        .iter()
        .map(|face| face_masks(models, image, face, full_size, None))
        .collect::<Result<Vec<_>>>()?;
    cache.prepared = Some(PreparedMasks {
        full_size_px: full_size,
        faces: faces.to_vec(),
        masks,
    });
    Ok(())
}

/// The rectangle a face's blemish map covers: the face rectangle joined with the blemish mask's
/// region (a parser's mask may reach beyond the rectangle), clipped to the image. `None` when there is nothing.
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
    // A proposer whose constants follow the face size is made for this face.
    let own = models.proposer.for_face(face_px);
    blemish_map(
        &crop,
        &mask,
        rect,
        [1.0; 3],
        face_px,
        own.as_deref().unwrap_or(models.proposer.as_ref()),
    )
}

/// Makes sure the maps of the faces flagged in `wanted` are cached. Without the full-resolution
/// image a missing map is an error.
fn ensure_maps(
    full_image: Option<&LinearAwg4Image>,
    full_size: (u32, u32),
    faces: &[Face],
    wanted: &[bool],
    models: &Models,
    cache: &mut RetouchCache,
) -> Result<()> {
    let key = cache.map_key(faces);
    for (i, face) in faces.iter().enumerate() {
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
///   pixel out of every tool, and a value between scales every tool's weight by `1 - protect / 255`
///   (the owner, 2026-11-11: Protect covers blemish removal, dark circles, even skin, grading and
///   teeth whitening).
///
/// Order: output is a copy of the input; blemish removal for every face; then, for every face,
/// dark circles, even skin, skin grading and teeth whitening (`docs/27` 7.9). With every amount 0 the output equals the input bit for bit.
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
    let wants_low = faces.iter().any(|(_, s)| {
        s.dark_circles > 0.0 || s.even_skin > 0.0 || s.grading_active() || s.teeth_whitening > 0.0
    });
    if !wants_blemish.iter().any(|&b| b) && !wants_low {
        return Ok(output);
    }

    let full_image = ((image.width_px, image.height_px) == full_size_px).then_some(image);
    let face_list = faces_of(&faces);
    ensure_masks(full_image, full_size_px, &face_list, models, cache)?;
    ensure_maps(
        full_image,
        full_size_px,
        &face_list,
        &wants_blemish,
        models,
        cache,
    )?;

    let key = cache.map_key(&face_list);
    for (i, (_, settings)) in faces.iter().enumerate() {
        if !wants_blemish[i] {
            continue;
        }
        if let Some(map) = cache.maps.get(&key, i) {
            apply_blemish_at(
                &mut output,
                (0, 0),
                &map,
                settings.a(),
                render_scale,
                protect,
            )?;
        }
    }
    if wants_low {
        for (i, (face, settings)) in faces.iter().enumerate() {
            let Some(masks) = cache.masks(i) else {
                continue;
            };
            let tools = ToolMasks {
                dark_circles: masks.dark_circles.as_ref(),
                even_skin: masks.even_skin.as_ref(),
                grading: masks.grading.as_ref(),
                teeth: masks.teeth.as_ref(),
            };
            retouch_tones_at(
                &mut output,
                Placement {
                    origin_px: (0, 0),
                    working_px: (image.width_px, image.height_px),
                },
                full_size_px,
                face,
                settings,
                render_scale,
                &tools,
                protect,
            )?;
        }
    }
    Ok(output)
}

/// The rectangle of the working image (`working_px`) that one face's tools read and write, from the cached masks and map: the
/// blemish map's footprint (when Blemish is above 0) joined with the region of [`tones_region`]. `None` when no tool of the
/// face does anything.
fn face_region(
    index: usize,
    face: &Face,
    settings: &SkinSettings,
    working_px: (u32, u32),
    full_size_px: (u32, u32),
    render_scale: f32,
    cache: &RetouchCache,
) -> Result<Option<Rect>> {
    let Some(masks) = cache.masks(index) else {
        return Err(RetouchError::NotPrepared);
    };
    let scale = f64::from(render_scale);
    let mut out: Option<Rect> = None;
    if settings.blemish > 0.0
        && let Some(map) = cache.map_of(index)
    {
        // The map's rectangle in working pixels, with the pixel the bilinear taps and the conservative column range of
        // `apply_blemish_at` can reach.
        let r = map.rect;
        let x0 = ((f64::from(r.x_px) - 1.0) * scale).floor() as i64 - 1;
        let y0 = ((f64::from(r.y_px) - 1.0) * scale).floor() as i64 - 1;
        let x1 = ((r.right_px() as f64 + 1.0) * scale).ceil() as i64 + 1;
        let y1 = ((r.bottom_px() as f64 + 1.0) * scale).ceil() as i64 + 1;
        let (x0, y0) = (
            x0.clamp(0, i64::from(working_px.0)),
            y0.clamp(0, i64::from(working_px.1)),
        );
        let (x1, y1) = (
            x1.clamp(0, i64::from(working_px.0)),
            y1.clamp(0, i64::from(working_px.1)),
        );
        if x0 < x1 && y0 < y1 {
            out = Some(Rect::new(
                x0 as i32,
                y0 as i32,
                (x1 - x0) as u32,
                (y1 - y0) as u32,
            ));
        }
    }
    let tools = ToolMasks {
        dark_circles: masks.dark_circles.as_ref(),
        even_skin: masks.even_skin.as_ref(),
        grading: masks.grading.as_ref(),
        teeth: masks.teeth.as_ref(),
    };
    if let Some(t) = tones_region(
        working_px,
        full_size_px,
        face,
        settings,
        render_scale,
        &tools,
    )? {
        out = Some(join_rects(out, t));
    }
    Ok(out)
}

fn intersects(a: &Rect, b: &Rect) -> bool {
    i64::from(a.x_px) < b.right_px()
        && i64::from(b.x_px) < a.right_px()
        && i64::from(a.y_px) < b.bottom_px()
        && i64::from(b.y_px) < a.bottom_px()
}

fn contains(outer: &Rect, inner: &Rect) -> bool {
    i64::from(outer.x_px) <= i64::from(inner.x_px)
        && i64::from(outer.y_px) <= i64::from(inner.y_px)
        && outer.right_px() >= inner.right_px()
        && outer.bottom_px() >= inner.bottom_px()
}

/// The parts of the working image that [`retouch_skin_region`] needs to retouch `faces` (`docs/27` decision 13): for each
/// face the rectangle its tools read and write, rectangles that overlap merged into one, every origin even, and cut to the
/// working image. The parts are disjoint, in reading order. Faces whose tools do nothing give no part. The masks and maps
/// must be in the cache ([`prepare_skin`]).
///
/// # Errors
///
/// [`RetouchError::NotPrepared`] without the cached masks, and the errors of the scale.
pub fn retouch_regions(
    working_px: (u32, u32),
    full_size_px: (u32, u32),
    faces: &[(Face, SkinSettings)],
    render_scale: f32,
    cache: &RetouchCache,
) -> Result<Vec<Rect>> {
    check_render_scale(render_scale)?;
    let face_list = faces_of(faces);
    if !cache.masks_match(full_size_px, &face_list) {
        return Err(RetouchError::NotPrepared);
    }
    let mut rects = Vec::new();
    for (i, (face, settings)) in faces.iter().enumerate() {
        let settings = settings.clamped();
        if let Some(r) = face_region(
            i,
            face,
            &settings,
            working_px,
            full_size_px,
            render_scale,
            cache,
        )? {
            rects.push(r);
        }
    }
    // Even origins (the half-resolution path of the low-frequency layers works on 2 x 2 blocks of the whole image's grid),
    // then overlapping rectangles merged until none overlap.
    let align = |r: Rect| {
        let x0 = i64::from(r.x_px) & !1;
        let y0 = i64::from(r.y_px) & !1;
        Rect::new(
            x0 as i32,
            y0 as i32,
            (r.right_px() - x0) as u32,
            (r.bottom_px() - y0) as u32,
        )
    };
    let mut rects: Vec<Rect> = rects.into_iter().map(align).collect();
    loop {
        let mut merged = false;
        'search: for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                if intersects(&rects[i], &rects[j]) {
                    let joined = join_rects(Some(rects[i]), rects[j]);
                    rects[i] = joined;
                    rects.remove(j);
                    merged = true;
                    break 'search;
                }
            }
        }
        if !merged {
            break;
        }
    }
    rects.sort_by_key(|r| (r.y_px, r.x_px));
    Ok(rects)
}

/// [`retouch_skin`] on a part of the working image (`docs/27` decision 13): `part` is the rectangle of the working image
/// whose top-left pixel is `place.origin_px` (`place.working_px` is the size of the whole), and `protect` is the same
/// rectangle of the protection mask. Returns the retouched part. The pixels are those of the same rectangle of
/// [`retouch_skin`] on the whole image, bit for bit, when `part` holds every rectangle of the faces it meets (one of
/// [`retouch_regions`]).
///
/// Masks and maps come from the cache only ([`prepare_skin`]); nothing is built here and no model runs. A face whose
/// rectangle does not meet the part is left out; one that meets it without being inside it is an error.
///
/// # Errors
///
/// [`RetouchError::NotPrepared`] without the cached masks or maps; [`RetouchError::Part`] for an odd origin, a part outside
/// the working image, or a face that is cut by the part; and the errors of [`retouch_skin`].
pub fn retouch_skin_region(
    part: &LinearAwg4Image,
    place: Placement,
    full_size_px: (u32, u32),
    faces: &[(Face, SkinSettings)],
    render_scale: f32,
    cache: &RetouchCache,
    protect: Option<&Mask>,
) -> Result<LinearAwg4Image> {
    part.validate()?;
    let scale = check_render_scale(render_scale)?;
    let working = place.working_px;
    check_working_size_px(working, full_size_px, render_scale, scale)?;
    if place.origin_px.0 < 0
        || place.origin_px.1 < 0
        || place.origin_px.0 & 1 != 0
        || place.origin_px.1 & 1 != 0
    {
        return Err(RetouchError::Part {
            why: "its origin must be even and inside the picture",
        });
    }
    let bounds = Rect::new(
        place.origin_px.0,
        place.origin_px.1,
        part.width_px,
        part.height_px,
    );
    if bounds.right_px() > i64::from(working.0) || bounds.bottom_px() > i64::from(working.1) {
        return Err(RetouchError::Part {
            why: "it reaches past the picture",
        });
    }
    if let Some(p) = protect {
        p.validate()?;
        if p.size_px() != (part.width_px, part.height_px) {
            return Err(RetouchError::MaskSizeMismatch {
                left: (part.width_px, part.height_px),
                right: p.size_px(),
            });
        }
    }
    let faces: Vec<(Face, SkinSettings)> = faces.iter().map(|(f, s)| (*f, s.clamped())).collect();
    let mut output = part.clone();
    let face_list = faces_of(&faces);
    if !cache.masks_match(full_size_px, &face_list) {
        return Err(RetouchError::NotPrepared);
    }
    // The faces of this part, by their regions.
    let mut mine = Vec::new();
    for (i, (face, settings)) in faces.iter().enumerate() {
        let Some(region) = face_region(
            i,
            face,
            settings,
            working,
            full_size_px,
            render_scale,
            cache,
        )?
        else {
            continue;
        };
        if !intersects(&bounds, &region) {
            continue;
        }
        if !contains(&bounds, &region) {
            return Err(RetouchError::Part {
                why: "a face reaches past it",
            });
        }
        mine.push(i);
    }
    let key = cache.map_key(&face_list);
    for &i in &mine {
        let settings = &faces[i].1;
        if settings.blemish > 0.0 {
            let Some(map) = cache.maps.get(&key, i) else {
                if cache.masks(i).is_some_and(|m| m.blemish.is_some()) {
                    return Err(RetouchError::NotPrepared);
                }
                continue;
            };
            apply_blemish_at(
                &mut output,
                place.origin_px,
                &map,
                settings.a(),
                render_scale,
                protect,
            )?;
        }
    }
    for &i in &mine {
        let (face, settings) = &faces[i];
        let Some(masks) = cache.masks(i) else {
            continue;
        };
        let tools = ToolMasks {
            dark_circles: masks.dark_circles.as_ref(),
            even_skin: masks.even_skin.as_ref(),
            grading: masks.grading.as_ref(),
            teeth: masks.teeth.as_ref(),
        };
        retouch_tones_at(
            &mut output,
            place,
            full_size_px,
            face,
            settings,
            render_scale,
            &tools,
            protect,
        )?;
    }
    Ok(output)
}

fn check_working_size(
    image: &LinearAwg4Image,
    full_size_px: (u32, u32),
    render_scale: f32,
    scale: f64,
) -> Result<()> {
    check_working_size_px(
        (image.width_px, image.height_px),
        full_size_px,
        render_scale,
        scale,
    )
}

fn check_working_size_px(
    working_px: (u32, u32),
    full_size_px: (u32, u32),
    render_scale: f32,
    scale: f64,
) -> Result<()> {
    let expected = |side: u32| (f64::from(side) * scale).round().max(1.0);
    let (ew, eh) = (expected(full_size_px.0), expected(full_size_px.1));
    // Two pixels, and a pixel of the masks' frame at a scale above 1 (the sides of a frame are rounded in whole pixels of its own).
    let tolerance = 2.0f64.max(scale.ceil());
    if full_size_px.0 == 0
        || full_size_px.1 == 0
        || (f64::from(working_px.0) - ew).abs() > tolerance
        || (f64::from(working_px.1) - eh).abs() > tolerance
    {
        return Err(RetouchError::WorkingSize {
            image: working_px,
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
            ..SkinSettings::default()
        }
    }

    /// Every tool at once: the three of 7.1 to 7.6 at 60 plus grading and teeth.
    fn settings_all() -> SkinSettings {
        SkinSettings {
            skin_contrast: 50.0,
            shadow_desaturation: 60.0,
            shine_removal: 60.0,
            teeth_whitening: 70.0,
            ..settings(60.0, 60.0, 60.0)
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
        for bm in [
            &m.blemish,
            &m.dark_circles,
            &m.even_skin,
            &m.grading,
            &m.teeth,
        ]
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
        let faces = [(face, settings_all())];

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
        let faces = [(face, settings_all())];
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
        // Amounts (any of them) reuse masks and map.
        prepare_skin(&img, &faces, &models, &mut cache).unwrap();
        let masks_before = cache.masks(0).unwrap().clone();
        retouch_skin(
            &img,
            (600, 450),
            &[(face, settings_all())],
            1.0,
            &models,
            &mut cache,
            None,
        )
        .unwrap();
        assert_eq!(cache.map_count(), 1);
        assert_eq!(cache.masks(0), Some(&masks_before));
        assert!(masks_before.grading.is_some());
        // The synthetic skin has no teeth: the gate of 5.6 finds none.
        assert!(masks_before.teeth.is_none());
        // Another face rectangle rebuilds the masks and adds a map.
        let moved = Face {
            rect: Rect::new(180, 30, 250, 300),
            ..face
        };
        let mut other = RetouchCache::new(1, 1);
        prepare_skin(&img, &faces, &models, &mut other).unwrap();
        prepare_skin(
            &img,
            &[(moved, settings(60.0, 0.0, 0.0))],
            &models,
            &mut other,
        )
        .unwrap();
        assert_eq!(other.map_count(), 1);
        assert_ne!(other.masks(0).unwrap().blemish, masks_before.blemish);
        // New keys drop everything.
        other.set_keys(2, 1);
        assert!(!other.has_masks());
        assert_eq!(other.map_count(), 0);
    }

    #[test]
    fn protection_mask_stops_every_tool() {
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
        // Every tool at 100: a fully protected image is unchanged to the bit.
        let every = SkinSettings {
            blemish: 100.0,
            dark_circles: 100.0,
            even_skin: 100.0,
            skin_contrast: 100.0,
            shadow_desaturation: 100.0,
            shine_removal: 100.0,
            teeth_whitening: 100.0,
            ..SkinSettings::default()
        };
        let faces = [(face, every)];
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
        let free = retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_ne!(bits(&free), bits(&img));
        // Only the left of x = 400 protected: it is unchanged, the rest is retouched.
        let mut half = Mask::zeros(600, 450).unwrap();
        for y in 0..450 {
            for x in 0..400 {
                half.data[y * 600 + x] = 255;
            }
        }
        let part = retouch_skin(
            &img,
            (600, 450),
            &faces,
            1.0,
            &models,
            &mut cache,
            Some(&half),
        )
        .unwrap();
        let (mut left_changed, mut right_changed) = (0, 0);
        for y in 0..450usize {
            for x in 0..600usize {
                let i = y * 600 + x;
                if part.data[i] != img.data[i] {
                    if x < 400 {
                        left_changed += 1;
                    } else {
                        right_changed += 1;
                    }
                }
            }
        }
        assert_eq!(left_changed, 0, "protected pixels are bit-identical");
        assert!(right_changed > 100, "the unprotected part is retouched");
        // A soft protection moves the pixels less.
        let soft = Mask::new(600, 450, vec![128; 600 * 450]).unwrap();
        let mid = retouch_skin(
            &img,
            (600, 450),
            &faces,
            1.0,
            &models,
            &mut cache,
            Some(&soft),
        )
        .unwrap();
        let dist = |a: &LinearAwg4Image| -> f64 {
            a.data
                .iter()
                .zip(&img.data)
                .map(|(p, q)| (0..3).map(|c| f64::from((p[c] - q[c]).abs())).sum::<f64>())
                .sum()
        };
        assert!(dist(&mid) < 0.75 * dist(&free) && dist(&mid) > 0.0);
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

    fn upscale2(img: &LinearAwg4Image) -> LinearAwg4Image {
        let (w, h) = (img.width_px as usize * 2, img.height_px as usize * 2);
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                data.push(img.data[(y / 2) * img.width_px as usize + x / 2]);
            }
        }
        LinearAwg4Image::new(w as u32, h as u32, data).unwrap()
    }

    /// Stitches the parts [`retouch_skin_region`] gives over a copy of the working image.
    fn by_parts(
        working: &LinearAwg4Image,
        full: (u32, u32),
        faces: &[(Face, SkinSettings)],
        scale: f32,
        cache: &RetouchCache,
        protect: Option<&Mask>,
    ) -> (LinearAwg4Image, Vec<Rect>) {
        let size = (working.width_px, working.height_px);
        let parts = retouch_regions(size, full, faces, scale, cache).unwrap();
        let mut out = working.clone();
        for r in &parts {
            assert!(r.x_px % 2 == 0 && r.y_px % 2 == 0, "even origin {r:?}");
            let crop = crop_image(working, *r).unwrap();
            let prot = protect.map(|p| crop_mask_plain(p, *r));
            let done = retouch_skin_region(
                &crop,
                Placement {
                    origin_px: (r.x_px, r.y_px),
                    working_px: size,
                },
                full,
                faces,
                scale,
                cache,
                prot.as_ref(),
            )
            .unwrap();
            for y in 0..r.height_px as usize {
                let dst = (r.y_px as usize + y) * size.0 as usize + r.x_px as usize;
                let src = y * r.width_px as usize;
                out.data[dst..dst + r.width_px as usize]
                    .copy_from_slice(&done.data[src..src + r.width_px as usize]);
            }
        }
        (out, parts)
    }

    fn crop_mask_plain(m: &Mask, r: Rect) -> Mask {
        let (w, h) = (r.width_px as usize, r.height_px as usize);
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            let s = (r.y_px as usize + y) * m.width_px as usize + r.x_px as usize;
            data.extend_from_slice(&m.data[s..s + w]);
        }
        Mask::new(w as u32, h as u32, data).unwrap()
    }

    /// The tools on parts around the faces equal the tools on the whole picture, bit for bit, at the scales an export and a
    /// preview use, with several faces (two far apart, two that overlap), every tool on and a protection plane.
    #[test]
    fn parts_around_the_faces_equal_the_whole_picture_bit_for_bit() {
        let faces_at = [
            // Two faces far apart, and a pair whose regions overlap.
            Face {
                rect: Rect::new(60, 40, 220, 260),
                scale: 0.26,
            },
            Face {
                rect: Rect::new(700, 60, 200, 240),
                scale: 0.24,
            },
            Face {
                rect: Rect::new(380, 220, 180, 210),
                scale: 0.21,
            },
            Face {
                rect: Rect::new(470, 240, 170, 200),
                scale: 0.2,
            },
        ];
        let (w, h) = (1000u32, 520u32);
        let mut full = synthetic(w, h, &faces_at[0]);
        for f in &faces_at[1..] {
            let extra = synthetic(w, h, f);
            for (a, b) in full.data.iter_mut().zip(&extra.data) {
                *a = [
                    (a[0] + b[0]) * 0.5,
                    (a[1] + b[1]) * 0.5,
                    (a[2] + b[2]) * 0.5,
                ];
            }
        }
        let models = test_models(MockProposer::scale(192, 0.9));
        let mix = [
            settings_all(),
            settings(70.0, 0.0, 0.0),
            settings_all(),
            settings(0.0, 80.0, 40.0),
        ];
        let faces: Vec<(Face, SkinSettings)> = faces_at.iter().copied().zip(mix).collect();
        let mut cache = RetouchCache::new(3, 3);
        prepare_skin(&full, &faces, &models, &mut cache).unwrap();
        for (scale, working) in [
            (1.0f32, full.clone()),
            (0.5, downscale2(&full)),
            (2.0, upscale2(&full)),
        ] {
            let size = (working.width_px, working.height_px);
            let protect = Mask::new(
                size.0,
                size.1,
                (0..size.0 * size.1)
                    .map(|i| {
                        if (i % size.0 / 40 + i / size.0 / 40) % 3 == 0 {
                            255
                        } else if i % 7 == 0 {
                            128
                        } else {
                            0
                        }
                    })
                    .collect(),
            )
            .unwrap();
            for protect in [None, Some(&protect)] {
                let whole = retouch_skin(
                    &working,
                    (w, h),
                    &faces,
                    scale,
                    &models,
                    &mut cache,
                    protect,
                )
                .unwrap();
                assert_ne!(
                    bits(&whole),
                    bits(&working),
                    "the tools changed the picture at {scale}"
                );
                let (stitched, parts) = by_parts(&working, (w, h), &faces, scale, &cache, protect);
                assert!(!parts.is_empty());
                assert_eq!(
                    bits(&stitched),
                    bits(&whole),
                    "scale {scale}: parts equal the whole picture"
                );
                if scale == 1.0 {
                    // The two far faces are two parts and the overlapping pair is one; a part is smaller than the picture.
                    assert!(parts.len() >= 2 && parts.len() <= 3, "{parts:?}");
                    let area: u64 = parts
                        .iter()
                        .map(|r| u64::from(r.width_px) * u64::from(r.height_px))
                        .sum();
                    assert!(
                        area < u64::from(w * h) * 3 / 4,
                        "parts cover {area} of {}",
                        w * h
                    );
                }
            }
        }
    }

    #[test]
    fn a_part_that_cuts_a_face_or_starts_odd_is_refused() {
        let face = Face {
            rect: Rect::new(100, 60, 200, 240),
            scale: 0.4,
        };
        let full = synthetic(600, 400, &face);
        let models = test_models(MockProposer::scale(192, 0.9));
        let faces = [(face, settings_all())];
        let mut cache = RetouchCache::new(4, 4);
        prepare_skin(&full, &faces, &models, &mut cache).unwrap();
        let place = |x, y| Placement {
            origin_px: (x, y),
            working_px: (600, 400),
        };
        let crop = crop_image(&full, Rect::new(150, 100, 100, 100)).unwrap();
        let cut = retouch_skin_region(
            &crop,
            place(150, 100),
            (600, 400),
            &faces,
            1.0,
            &cache,
            None,
        );
        assert!(matches!(cut, Err(RetouchError::Part { .. })), "{cut:?}");
        let crop = crop_image(&full, Rect::new(151, 100, 100, 100)).unwrap();
        let odd = retouch_skin_region(
            &crop,
            place(151, 100),
            (600, 400),
            &faces,
            1.0,
            &cache,
            None,
        );
        assert!(matches!(odd, Err(RetouchError::Part { .. })), "{odd:?}");
        // A part with no face in it is returned as it is.
        let far = crop_image(&full, Rect::new(500, 300, 100, 100)).unwrap();
        let same =
            retouch_skin_region(&far, place(500, 300), (600, 400), &faces, 1.0, &cache, None)
                .unwrap();
        assert_eq!(bits(&same), bits(&far));
        // Nothing prepared: refused.
        let empty = RetouchCache::new(5, 5);
        let none =
            retouch_skin_region(&far, place(500, 300), (600, 400), &faces, 1.0, &empty, None);
        assert_eq!(none.unwrap_err(), RetouchError::NotPrepared);
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
        for scale in [0.0, 17.0, f32::NAN] {
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
    fn failing_detector_and_missing_mouth_are_survivable() {
        let face = Face {
            rect: Rect::new(175, 30, 250, 300),
            scale: 0.5,
        };
        let img = synthetic(600, 450, &face);
        let models = Models {
            beauty_marks: Some(Box::new(mock::FailingBeautyMarks)),
            ..test_models(MockProposer::scale(192, 0.9))
        };
        let faces = [(face, settings(60.0, 60.0, 60.0))];
        let mut cache = RetouchCache::default();
        let out = retouch_skin(&img, (600, 450), &faces, 1.0, &models, &mut cache, None).unwrap();
        assert_ne!(bits(&out), bits(&img));

        // A parser that finds no mouth: whitening does nothing, the rest goes on.
        struct NoMouth;
        impl crate::FaceParser for NoMouth {
            fn part_mask(
                &self,
                image: &LinearAwg4Image,
                face: &Face,
                part: crate::FacePart,
            ) -> Result<Option<BoundedMask>> {
                if part == crate::FacePart::MouthInterior {
                    return Err(RetouchError::Model("no mouth".into()));
                }
                mock::MockParser.part_mask(image, face, part)
            }
        }
        let models = Models {
            parser: Box::new(NoMouth),
            ..test_models(MockProposer::identity(192))
        };
        let only_teeth = SkinSettings {
            teeth_whitening: 100.0,
            ..SkinSettings::default()
        };
        let mut cache = RetouchCache::default();
        let out = retouch_skin(
            &img,
            (600, 450),
            &[(face, only_teeth)],
            1.0,
            &models,
            &mut cache,
            None,
        )
        .unwrap();
        assert_eq!(bits(&out), bits(&img));
        assert!(cache.masks(0).unwrap().teeth.is_none());
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
            contrast in -150.0f32..150.0,
            shadow in -20.0f32..140.0,
            shine in -20.0f32..140.0,
            teeth in -20.0f32..140.0,
        ) {
            let face = Face { rect: Rect::new(50, 20, 100, 120), scale: 0.6 };
            let img = synthetic(200, 160, &face);
            let models = test_models(MockProposer::scale(64, 0.9));
            let s = SkinSettings {
                blemish, dark_circles: dark, even_skin: even, texture,
                skin_contrast: contrast, shadow_desaturation: shadow,
                shine_removal: shine, teeth_whitening: teeth,
            };
            let mut cache = RetouchCache::default();
            let out = retouch_skin(&img, (200, 160), &[(face, s)], 1.0, &models, &mut cache, None).unwrap();
            prop_assert_eq!((out.width_px, out.height_px), (200, 160));
            prop_assert!(out.data.iter().flatten().all(|v| v.is_finite()));
            if blemish <= 0.0 && dark <= 0.0 && even <= 0.0 && contrast == 0.0
                && shadow <= 0.0 && shine <= 0.0 && teeth <= 0.0
            {
                prop_assert_eq!(bits(&out), bits(&img));
            }
        }
    }
}
