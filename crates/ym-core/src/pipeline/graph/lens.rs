//! The lens in the render graph (`docs/21` sections 3 and 5): L1 to L3 between the raw group (after
//! R9) and the colour group (before C1), on the CPU.
//!
//! A photo with a lens (a setting with an amount above 0, and a lens library) gets, once per photo:
//!
//! * a **plan** ([`LensCtx`]): the lens bound to the photo's frame (the auto-cropped full-resolution
//!   frame of R9), anamorphic axis, f-number, clip levels `g0` and camera, cached in the photo's
//!   graph by the text of those inputs;
//! * the **planes** ([`PlanesEntry`]): the lens plane, tail plane, source list and blur map,
//!   made from the raw group at `k_L` and kept outside the evictable caches, so that every tile and
//!   every scale of the photo shares them. Their key leaves the three amounts out: a slider move
//!   re-runs only the passes below.
//!
//! and the passes, in four places that mirror the five places the raw group leaves the graph:
//!
//! | Raw group | With a lens |
//! |---|---|
//! | [`Job::group_backing`] (fit view, thumbnails, look tiles, histogram) | [`Job::lens_group_backing`]: `ym_lens::apply` over the whole frame at 1/k, cached in `lens_groups` |
//! | [`Job::region_pixels`] (the 100 % view) | [`Job::lens_region_pixels`]: lens tiles on the 512 px grid, cached in `lens_tiles` |
//! | [`Job::region`] / [`Job::region_resident`] (exports) | [`Job::lens_region`] / [`Job::lens_region_resident`]: the raw region grown by the lens halo, then `apply`, nothing cached |
//! | [`Job::tiles_ready`] / [`Job::warm_tiles`] | [`Job::lens_tiles_ready`] / [`Job::lens_warm_tiles`] |
//!
//! Without a lens every one of them is the raw method it mirrors, and nothing here runs: no plan, no
//! planes, no counter, the bytes of a render that never heard of lenses.
//!
//! # On the device
//!
//! There are no device lens kernels yet. A raw group that lives on the device is read back
//! ([`Job::group`]) and the lens runs on the host; its output is a host image, which every consumer
//! (the sampler and the colour pass on the device, the CPU colour pass, the export tail) takes. The
//! raw chain of a full-resolution region still runs on the device when it can, with a host result.
//!
//! # Exports and determinism
//!
//! The planes of an export are made for the export, from the CPU reference raw group, and kept in
//! the request ([`Job::lens_job_planes`]); the graph's planes are the viewer's and may come from a
//! group the device made. So the bytes of an export do not depend on what was looked at before it,
//! and the app and `ym` write the same file.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ym_gpu::cpu::sampler::SamplerWarp;
use ym_gpu::cpu::{Placement, Rect, RgbImage};
use ym_lens::consts::NEWTON_STEPS;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensError, LensPlan, LensPlanes,
    PixelRect, ReshapeReason, ReshapeReport, apply, halo_px,
};
use ym_profiles::profile::Profile;
use ym_raw::makernote::SonyFocus;

use super::{Job, Pixels, Renderer, TileImage, lock};
use crate::pipeline::RenderError;
use crate::pipeline::raw::RawParamsKey;

/// The note of a render whose camera curves could not be read: the lens colours use a neutral camera.
const CAMERA_NOTE: &str =
    "The camera's colour curves could not be read, so the lens colours are neutral.";

/// The key of a reduced group after the lens: the raw group's, the lens's (`LensCtx::key`) and k.
pub(crate) type GroupKey = (RawParamsKey, String, usize);
/// The key of a 512 px tile after the lens: the raw group's, the lens's and the tile.
pub(crate) type TileKey = (RawParamsKey, String, usize, usize);

/// One photo's lens for one set of inputs (the plan, with what the cache keys need).
pub(crate) struct LensCtx {
    /// The lens bound to the photo.
    pub plan: LensPlan,
    /// Everything that decides the pixels after the lens besides the raw group: the plan's key
    /// (lens, data versions, amounts, camera, frame, axis, f-number, clip levels) and the focus
    /// point the planes were made with. In the frame and display keys, and in the keys of the lens
    /// caches.
    pub key: String,
    /// The camera's focus point as fractions of the auto-cropped frame, in the buffer's orientation.
    pub focus: Option<(f64, f64)>,
    /// L4: the anamorphic bend, the lens's distortion and the zoom, made once for the photo on the
    /// full-resolution canvas (the zoom is the same at every scale), or `None` when the plan has
    /// neither. [`Job::geometry`] puts it in the geometry.
    pub warp: Option<SamplerWarp>,
    /// Plain-words notes every request that uses the plan carries.
    notes: Vec<String>,
}

/// The lens's planes with what they were made from.
pub(crate) struct PlanesEntry {
    /// The raw group's key: the planes read its image.
    raw: RawParamsKey,
    /// [`LensPlan::planes_key`].
    planes_key: String,
    /// The bits of the focus point.
    focus: Option<(u64, u64)>,
    /// The planes.
    pub planes: Arc<LensPlanes>,
}

fn lens_error(e: LensError) -> RenderError {
    match e {
        LensError::Cancelled => RenderError::Cancelled,
        other => RenderError::Image(format!(
            "The lens effect can't be applied to this photo: {other}."
        )),
    }
}

fn focus_bits(focus: Option<(f64, f64)>) -> Option<(u64, u64)> {
    focus.map(|(x, y)| (x.to_bits(), y.to_bits()))
}

/// What the planes say about the bokeh for `plan`: a plan that does not ask for reshaping says so,
/// whatever planes made earlier for a plan that did.
fn report_of(plan: &LensPlan, planes: &LensPlanes) -> ReshapeReport {
    if plan.needs_blur_map() {
        planes.report()
    } else {
        ReshapeReport {
            reshaped: false,
            reason: Some(ReshapeReason::NotAsked),
        }
    }
}

/// The camera's focus point as fractions of the auto-cropped frame (the frame the lens works on),
/// in the buffer's orientation, or `None` when the file has none.
///
/// The maker note gives the point in the camera's frame units, in sensor orientation. R9 crops the
/// corrected image; the distortion correction moves a point by a few pixels at most, so the point is
/// taken as it stands in the corrected image (no inverse of R9's map is needed), then shifted by
/// the crop. The detector only needs the area, not the pixel. `sensor` is the size of the frame the
/// point is in, `crop` R9's crop of it (`None`: the whole frame).
fn focus_fraction(
    focus: Option<SonyFocus>,
    sensor: (usize, usize),
    crop: Option<Rect>,
) -> Option<(f64, f64)> {
    let focus = focus?;
    let (frame_w, frame_h) = (f64::from(focus.frame[0]), f64::from(focus.frame[1]));
    if frame_w <= 0.0 || frame_h <= 0.0 {
        return None;
    }
    let x = f64::from(focus.point[0]) / frame_w * sensor.0 as f64;
    let y = f64::from(focus.point[1]) / frame_h * sensor.1 as f64;
    let (x, y, w, h) = match crop {
        Some(c) => (x - c.x as f64, y - c.y as f64, c.w as f64, c.h as f64),
        None => (x, y, sensor.0 as f64, sensor.1 as f64),
    };
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    Some(((x / w).clamp(0.0, 1.0), (y / h).clamp(0.0, 1.0)))
}

impl Renderer {
    /// The camera's response on the lens data: the profile's spectral curves projected on the
    /// eight wavelengths and the 31-sample grid of the lens files. Read once per camera.
    fn lens_camera(
        &self,
        profile: &Profile,
        data_root: Option<&Path>,
    ) -> Result<CameraResponse, String> {
        let name = profile.spectral_curves.clone();
        if let Some(Ok(hit)) = lock(&self.inner.lens_cameras).get(&name) {
            return Ok(hit.clone());
        }
        let root = data_root.ok_or_else(|| "the renderer has no data folder".to_owned())?;
        let curves = profile.load_curves(root).map_err(|e| e.to_string())?;
        let made = CameraResponse::from_curves(&curves).map_err(|e| e.to_string())?;
        lock(&self.inner.lens_cameras).insert(name, Ok(made.clone()));
        Ok(made)
    }
}

impl Job<'_> {
    /// The photo's lens for this request, or `None` when there is none: all three amounts at 0
    /// (the bytes of a render without the field), or no lens library (the render says so in its
    /// notes, `LENS_UNAVAILABLE_NOTE`). Cached in the photo's graph by the text of its inputs.
    ///
    /// # Errors
    ///
    /// [`RenderError::Image`] when the plan cannot be made for this frame; the errors of
    /// [`Job::params`] and [`Job::plan`].
    pub(crate) fn lens_ctx(&mut self) -> Result<Option<Arc<LensCtx>>, RenderError> {
        if let Some(known) = &self.lens {
            return Ok(known.clone());
        }
        let made = self.make_lens_ctx()?;
        if let Some(ctx) = &made {
            for note in &ctx.notes {
                self.note(note);
            }
        }
        self.lens = Some(made.clone());
        Ok(made)
    }

    fn make_lens_ctx(&mut self) -> Result<Option<Arc<LensCtx>>, RenderError> {
        let settings = self.src.settings.lens;
        if settings.is_identity() {
            return Ok(None);
        }
        let Some(library) = self.opts.lens.clone() else {
            return Ok(None);
        };
        let params = self.params()?;
        let raw_plan = self.plan()?;
        // The anamorphic axis is the canvas's horizontal: the buffer's x unless the quarter turns
        // swap the sides, then its y.
        let base = self.base_geometry();
        let axis = if base.swaps_sides() { Axis::Y } else { Axis::X };
        let f_number = self
            .src
            .raw
            .meta
            .f_number
            .filter(|f| f.is_finite() && *f > 0.0);
        let focus = focus_fraction(
            self.src.raw.meta.focus,
            (raw_plan.width, raw_plan.height),
            raw_plan
                .lens
                .as_ref()
                .map(ym_gpu::cpu::lens::LensGeometry::crop),
        );
        let amounts = LensAmounts::from(settings);
        let frame = FrameInfo {
            width: raw_plan.out_w,
            height: raw_plan.out_h,
            axis,
            f_number,
            clip: params.g0,
        };
        let inputs = format!(
            "{amounts:?}|{frame:?}|{}|{:?}",
            self.src.profile.spectral_curves,
            focus_bits(focus)
        );
        let cached = {
            let st = lock(&self.g.state);
            st.lens_ctx
                .as_ref()
                .filter(|(text, _)| *text == inputs)
                .map(|(_, ctx)| Arc::clone(ctx))
        };
        if let Some(ctx) = cached {
            return Ok(Some(ctx));
        }
        let mut notes = Vec::new();
        let camera = match self
            .r
            .lens_camera(&self.src.profile, self.opts.data_root.as_deref())
        {
            Ok(camera) => camera,
            Err(why) => {
                tracing::warn!(error = %why, "the camera's curves could not be read for the lens; a neutral camera is used");
                notes.push(CAMERA_NOTE.to_owned());
                CameraResponse::neutral()
            }
        };
        let plan = LensPlan::new(&library, &amounts, frame, &camera).map_err(lens_error)?;
        let key = format!("{}|focus{:?}", plan.key(), focus_bits(focus));
        // L4 on the full-resolution canvas: the zoom is fitted here, once, and reduced canvases
        // reuse it (the geometry's grids take their own canvas size).
        let (canvas_w, canvas_h) = base.canvas_size(raw_plan.out_w, raw_plan.out_h);
        let l4 = plan.warp(canvas_w as f64, canvas_h as f64);
        let warp = (!l4.is_identity()).then(|| {
            SamplerWarp::new(
                l4.mu(),
                l4.beta(),
                l4.distortion_k1(),
                l4.distortion_k2(),
                l4.zoom(),
                NEWTON_STEPS as u8,
            )
        });
        let ctx = Arc::new(LensCtx {
            plan,
            key,
            focus,
            warp,
            notes,
        });
        lock(&self.g.state).lens_ctx = Some((inputs, Arc::clone(&ctx)));
        Ok(Some(ctx))
    }

    /// The lens's part of the keys of a frame and of a display frame (`None` without a lens, so
    /// their keys are the keys of a render that never heard of lenses).
    pub(crate) fn lens_key(&mut self) -> Result<Option<String>, RenderError> {
        Ok(self.lens_ctx()?.map(|ctx| ctx.key.clone()))
    }

    /// The pixels the lens reads around an output tile at full resolution: 0 without a lens. An
    /// export adds it to the halo its working set is reserved for. Makes the planes (the halo
    /// depends on the photo's blur map).
    ///
    /// # Errors
    ///
    /// The errors of [`Job::lens_planes`].
    pub(crate) fn lens_halo(&mut self) -> Result<usize, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return Ok(0);
        };
        let planes = self.lens_planes(&ctx)?;
        Ok(halo_px(&ctx.plan, &planes, 1))
    }

    /// What the finished request says about the lens: what this request's planes said, else what the
    /// photo's planes in the graph say (a frame served from the cache made none).
    pub(super) fn lens_report_for_finish(&mut self) -> Option<ReshapeReport> {
        if self.src.settings.lens.is_identity() || self.opts.lens.is_none() {
            return None;
        }
        if self.lens_report.is_some() {
            return self.lens_report;
        }
        let ctx = self.lens_ctx().ok().flatten()?;
        let planes_key = ctx.plan.planes_key();
        if let Some(planes) = &self.lens_job_planes {
            return Some(report_of(&ctx.plan, planes));
        }
        let st = lock(&self.g.state);
        let entry = st.lens_planes.as_ref()?;
        (entry.planes_key == planes_key).then(|| report_of(&ctx.plan, &entry.planes))
    }

    // -----------------------------------------------------------------------------------------
    // The planes
    // -----------------------------------------------------------------------------------------

    /// The lens's planes for the photo, made when they are not there: from the raw group at `k_L`
    /// ([`LensPlan::lens_plane_scale`]) and the camera's focus point. A viewer shares them through
    /// the graph; an export makes its own from the CPU reference ([`Job::lens_job_planes`]).
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`], [`RenderError::Image`] for a lens plane the planes refuse, and
    /// the errors of the raw group.
    pub(crate) fn lens_planes(
        &mut self,
        ctx: &Arc<LensCtx>,
    ) -> Result<Arc<LensPlanes>, RenderError> {
        if self.restartable {
            if let Some(planes) = &self.lens_job_planes
                && planes.serves(&ctx.plan)
            {
                self.lens_report = Some(report_of(&ctx.plan, planes));
                return Ok(Arc::clone(planes));
            }
        } else {
            let raw = self.params()?.key();
            let hit = {
                let st = lock(&self.g.state);
                st.lens_planes
                    .as_ref()
                    .filter(|e| {
                        e.raw == raw
                            && e.planes_key == ctx.plan.planes_key()
                            && e.focus == focus_bits(ctx.focus)
                            && e.planes.serves(&ctx.plan)
                    })
                    .map(|e| Arc::clone(&e.planes))
            };
            if let Some(planes) = hit {
                self.lens_report = Some(report_of(&ctx.plan, &planes));
                return Ok(planes);
            }
        }
        self.check()?;
        let source = self.lens_plane_source(ctx)?;
        self.check()?;
        let t = Instant::now();
        let image = ImageRef {
            width: source.width,
            height: source.height,
            x0: 0,
            y0: 0,
            frame_width: source.width,
            frame_height: source.height,
            data: &source.data,
        };
        let cancel = self.cancel.clone();
        let planes =
            LensPlanes::build(&ctx.plan, &image, ctx.focus, &move || cancel.is_cancelled())
                .map(Arc::new)
                .map_err(lens_error)?;
        self.runs.lens_planes += 1;
        self.timings.lens_ms += t.elapsed().as_secs_f64() * 1000.0;
        self.lens_report = Some(report_of(&ctx.plan, &planes));
        if self.restartable {
            self.lens_job_planes = Some(Arc::clone(&planes));
        } else {
            let entry = PlanesEntry {
                raw: self.params()?.key(),
                planes_key: ctx.plan.planes_key(),
                focus: focus_bits(ctx.focus),
                planes: Arc::clone(&planes),
            };
            lock(&self.g.state).lens_planes = Some(entry);
        }
        Ok(planes)
    }

    /// The whole frame at `k_L` as a host image, the input of [`LensPlanes::build`].
    ///
    /// A viewer takes the raw group the graph holds (a device group is read back, [`Job::group`]); a
    /// frame under 1024 px, which only tiny test frames are, has `k_L` = 1 and is assembled from the
    /// raw tiles. An export makes the image fresh on the CPU reference, whatever device the renderer
    /// has and whatever the viewer cached.
    fn lens_plane_source(&mut self, ctx: &LensCtx) -> Result<Arc<RgbImage>, RenderError> {
        let k = ctx.plan.lens_plane_scale();
        let plan = self.plan()?;
        // A camera-processed file's groups are the CPU's at any scale.
        if plan.reduced {
            return self.group(k);
        }
        let reference = self.restartable;
        if k >= 2 {
            if !reference {
                return self.group(k);
            }
            let params = self.params()?;
            let frame = self.frame()?;
            let image = self.group_on_host(&frame, &plan, &params, k)?;
            self.runs.raw_groups += 1;
            return Ok(Arc::new(image));
        }
        let whole = Rect::whole(plan.out_w, plan.out_h);
        if !reference {
            return self.region(whole, true).map(Arc::new);
        }
        let frame = self.frame()?;
        let full = self.full(false)?;
        let t = Instant::now();
        let image = full.run_tile(&frame, whole)?;
        self.runs.tiles += 1;
        self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
        Ok(Arc::new(image))
    }

    // -----------------------------------------------------------------------------------------
    // The passes
    // -----------------------------------------------------------------------------------------

    /// `ym_lens::apply` over `out` of the frame at scale `k`, with `input` placed at `(x0, y0)` of a
    /// frame of `frame` pixels. The result keeps its place in the frame.
    #[allow(clippy::too_many_arguments)]
    fn lens_apply(
        &mut self,
        ctx: &LensCtx,
        planes: &LensPlanes,
        k: usize,
        input: &RgbImage,
        at: (usize, usize),
        frame: (usize, usize),
        out: PixelRect,
    ) -> Result<RgbImage, RenderError> {
        self.check()?;
        let t = Instant::now();
        let image = ImageRef {
            width: input.width,
            height: input.height,
            x0: at.0,
            y0: at.1,
            frame_width: frame.0,
            frame_height: frame.1,
            data: &input.data,
        };
        let data = apply(&ctx.plan, planes, k, &image, out).map_err(lens_error)?;
        self.timings.lens_ms += t.elapsed().as_secs_f64() * 1000.0;
        let mut image = RgbImage::from_planar(out.w, out.h, data).map_err(RenderError::from_cpu)?;
        image.place = Placement {
            x0: out.x,
            y0: out.y,
            frame_width: frame.0,
            frame_height: frame.1,
        };
        Ok(image)
    }

    /// The raw group at 1/`k` after the lens, for a fit view, a thumbnail, a look tile or the
    /// histogram: the lens over the whole frame at that scale (`docs/21` section 3, "Regions"), cached
    /// per lens setting. Without a lens it is [`Job::group_backing`], untouched. With one it is
    /// always a host image: a group on the device is read back first.
    ///
    /// # Errors
    ///
    /// The errors of [`Job::group_backing`] and [`Job::lens_planes`], and
    /// [`RenderError::Image`] when the lens refuses the image.
    pub(crate) fn lens_group_backing(&mut self, k: usize) -> Result<super::Backing, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.group_backing(k);
        };
        let key: GroupKey = (self.params()?.key(), ctx.key.clone(), k);
        let hit = lock(&self.g.state)
            .lens_groups
            .as_mut()
            .and_then(|groups| groups.get(&key).cloned());
        if let Some(image) = hit {
            return Ok(super::Backing::Cpu(image));
        }
        self.check()?;
        let planes = self.lens_planes(&ctx)?;
        let raw = self.group(k)?;
        let (w, h) = (raw.width, raw.height);
        let image = self.lens_apply(
            &ctx,
            &planes,
            k,
            &raw,
            (0, 0),
            (w, h),
            PixelRect::new(0, 0, w, h),
        )?;
        self.runs.lens_groups += 1;
        let image = Arc::new(image);
        let (bytes, budget) = (image.data.len() * 4, self.g.lens_group_budget());
        lock(&self.g.state)
            .lens_groups
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, Arc::clone(&image), bytes);
        Ok(super::Backing::Cpu(image))
    }

    /// The lens output over `rect` (coordinates of the auto-cropped image), made from the raw
    /// region grown by the lens halo and clamped to the frame. `keep_raw` lets the raw part come
    /// from the cached raw tiles (the 100 % view) or from one uncached call (an export).
    fn lens_region_direct(
        &mut self,
        ctx: &LensCtx,
        planes: &LensPlanes,
        rect: Rect,
        keep_raw: bool,
    ) -> Result<RgbImage, RenderError> {
        let plan = self.plan()?;
        let (fw, fh) = (plan.out_w, plan.out_h);
        let out = PixelRect::new(rect.x, rect.y, rect.w, rect.h);
        let want = out.grown(halo_px(&ctx.plan, planes, 1), fw, fh);
        let raw = self.region(Rect::new(want.x, want.y, want.w, want.h), keep_raw)?;
        self.lens_apply(ctx, planes, 1, &raw, (want.x, want.y), (fw, fh), out)
    }

    /// The error of a region outside the photo (as the raw group's methods give it).
    fn check_region(&mut self, rect: Rect) -> Result<(), RenderError> {
        let plan = self.plan()?;
        if rect.is_empty() || rect.x1() > plan.out_w || rect.y1() > plan.out_h {
            return Err(RenderError::Image(
                "The region is outside the photo.".to_owned(),
            ));
        }
        Ok(())
    }

    /// One 512 px tile of the full-resolution raw group after the lens, from the cache or made.
    fn lens_tile(
        &mut self,
        ctx: &Arc<LensCtx>,
        tx: usize,
        ty: usize,
    ) -> Result<Arc<RgbImage>, RenderError> {
        let key: TileKey = (self.params()?.key(), ctx.key.clone(), tx, ty);
        let hit = lock(&self.g.state)
            .lens_tiles
            .as_mut()
            .and_then(|tiles| tiles.get(&key).cloned());
        if let Some(image) = hit {
            return Ok(image);
        }
        self.check()?;
        let planes = self.lens_planes(ctx)?;
        let plan = self.plan()?;
        let t = self.opts.tile_px.max(64);
        let rect = Rect::new(
            tx * t,
            ty * t,
            t.min(plan.out_w - tx * t),
            t.min(plan.out_h - ty * t),
        );
        let image = Arc::new(self.lens_region_direct(ctx, &planes, rect, true)?);
        self.runs.lens_tiles += 1;
        let (bytes, budget) = (image.data.len() * 4, self.g.lens_tile_budget());
        lock(&self.g.state)
            .lens_tiles
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, Arc::clone(&image), bytes);
        Ok(image)
    }

    /// The lens output over `rect`, put together from cached lens tiles.
    fn lens_region_cached(
        &mut self,
        ctx: &Arc<LensCtx>,
        rect: Rect,
    ) -> Result<RgbImage, RenderError> {
        self.check_region(rect)?;
        let plan = self.plan()?;
        let t = self.opts.tile_px.max(64);
        let mut out = RgbImage::new(rect.w, rect.h);
        out.place = Placement {
            x0: rect.x,
            y0: rect.y,
            frame_width: plan.out_w,
            frame_height: plan.out_h,
        };
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                self.check()?;
                let tile = self.lens_tile(ctx, tx, ty)?;
                let tr = Rect::new(tx * t, ty * t, tile.width, tile.height);
                let ov = tr.intersect(&rect);
                out.paste(
                    &tile,
                    Rect::new(ov.x - tr.x, ov.y - tr.y, ov.w, ov.h),
                    ov.x - rect.x,
                    ov.y - rect.y,
                );
            }
        }
        Ok(out)
    }

    /// The pixels the 100 % view samples from after the lens: the full-resolution raw group over
    /// `rect`, out of cached lens tiles, as a host image. Without a lens it is
    /// [`Job::region_pixels`].
    ///
    /// # Errors
    ///
    /// The errors of [`Job::region_pixels`] and of the lens.
    pub(crate) fn lens_region_pixels(&mut self, rect: Rect) -> Result<Pixels, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.region_pixels(rect);
        };
        self.lens_region_cached(&ctx, rect)
            .map(|image| Pixels::Host(Arc::new(image)))
    }

    /// The full-resolution raw group over `rect` after the lens as a host image: out of cached lens
    /// tiles with `keep`, else from the raw region grown by the lens halo in one uncached call (an
    /// export's tile). Without a lens it is [`Job::region`].
    ///
    /// # Errors
    ///
    /// The errors of [`Job::region`] and of the lens.
    pub fn lens_region(&mut self, rect: Rect, keep: bool) -> Result<RgbImage, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.region(rect, keep);
        };
        if keep {
            return self.lens_region_cached(&ctx, rect);
        }
        self.check_region(rect)?;
        let planes = self.lens_planes(&ctx)?;
        let image = self.lens_region_direct(&ctx, &planes, rect, false)?;
        self.runs.lens_tiles += 1;
        Ok(image)
    }

    /// [`Job::region_resident`] after the lens: a host tile (there are no device lens kernels, so the
    /// tile the raw chain made on the device is read back inside [`Job::region`]); the export's colour
    /// stage uploads it when it runs on the device. Without a lens it is [`Job::region_resident`].
    ///
    /// # Errors
    ///
    /// As [`Job::lens_region`].
    pub fn lens_region_resident(&mut self, rect: Rect) -> Result<TileImage, RenderError> {
        if self.lens_ctx()?.is_none() {
            return self.region_resident(rect);
        }
        self.lens_region(rect, false).map(TileImage::Host)
    }

    /// Whether every lens tile (every raw tile without a lens) that `rect` needs is cached.
    ///
    /// # Errors
    ///
    /// The errors of [`Job::tiles_ready`].
    pub fn lens_tiles_ready(&mut self, rect: Rect) -> Result<bool, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.tiles_ready(rect);
        };
        let raw = self.params()?.key();
        let t = self.opts.tile_px.max(64);
        let st = lock(&self.g.state);
        let Some(tiles) = st.lens_tiles.as_ref() else {
            return Ok(false);
        };
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                if !tiles.contains(&(raw.clone(), ctx.key.clone(), tx, ty)) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Makes the lens tiles that cover `rect` and keeps them (the raw tiles under them stay cached
    /// too): the work ahead of a 100 % view. Without a lens it is [`Job::warm_tiles`].
    ///
    /// # Errors
    ///
    /// The errors of [`Job::warm_tiles`] and of the lens.
    pub fn lens_warm_tiles(&mut self, rect: Rect) -> Result<(), RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.warm_tiles(rect);
        };
        self.check_region(rect)?;
        let t = self.opts.tile_px.max(64);
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                self.check()?;
                self.lens_tile(&ctx, tx, ty)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn focus(frame: [u16; 2], point: [u16; 2]) -> Option<SonyFocus> {
        Some(SonyFocus {
            frame,
            point,
            box_size: None,
        })
    }

    #[test]
    fn the_focus_point_is_a_fraction_of_the_sensor_frame_without_a_crop() {
        // 6000 x 4000 camera frame on a 6048 x 4024 sensor frame: fractions are the same either way.
        let f = focus_fraction(focus([6000, 4000], [1500, 3000]), (6048, 4024), None).unwrap();
        assert!(
            (f.0 - 0.25).abs() < 1e-12 && (f.1 - 0.75).abs() < 1e-12,
            "{f:?}"
        );
    }

    #[test]
    fn the_crop_of_r9_moves_and_scales_the_focus_point() {
        // The point at (0.5, 0.5) of a 1000 x 800 sensor frame is (500, 400); the crop starts at
        // (100, 50) and is 800 x 700 wide.
        let crop = Rect::new(100, 50, 800, 700);
        let f = focus_fraction(focus([100, 100], [50, 50]), (1000, 800), Some(crop)).unwrap();
        assert!((f.0 - 0.5).abs() < 1e-12, "{f:?}");
        assert!((f.1 - 350.0 / 700.0).abs() < 1e-12, "{f:?}");
        // A point outside the crop is held at its edge: the fractions stay inside the frame.
        let f = focus_fraction(focus([100, 100], [1, 99]), (1000, 800), Some(crop)).unwrap();
        assert_eq!(f, (0.0, 1.0));
    }

    #[test]
    fn a_file_without_a_focus_point_or_with_an_empty_frame_has_none() {
        assert_eq!(focus_fraction(None, (100, 100), None), None);
        assert_eq!(
            focus_fraction(focus([0, 0], [0, 0]), (100, 100), None),
            None
        );
        assert_eq!(
            focus_fraction(
                focus([100, 100], [5, 5]),
                (100, 100),
                Some(Rect::new(0, 0, 0, 10))
            ),
            None
        );
    }
}
