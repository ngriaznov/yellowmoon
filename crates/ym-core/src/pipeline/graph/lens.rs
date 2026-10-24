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
//! # L4, the bend
//!
//! The anamorphic bend, the lens's distortion and the zoom are geometry, not a pass: the plan's map
//! is fitted once per photo on the full-resolution canvas ([`LensCtx::warp`]) and [`Job::geometry`]
//! puts it in the [`Geometry`](crate::pipeline::geometry::Geometry) every view and export samples
//! through. The passes above work on the photo as the lens delivers it, unbent; the sampler bends it
//! (`docs/21` section 4, L4).
//!
//! # Stage caches
//!
//! `Z = L2(L1(X))` does not read Flares, and the flare terms of L3 read neither Edges nor `Z`, so the
//! two halves are cached on their own (`lens_field_*` and `lens_flare_*` in the table of the module
//! above): a Flares move runs L3's terms over the cached `Z`, an Edges move runs L1 and L2 and reuses
//! the terms, an Anamorphic move runs both. With Edges and Anamorphic at 0 `Z` is the raw group itself.
//!
//! # On the device
//!
//! With [`GpuStages::LENS`](crate::app::gpu::GpuStages) on and a view that keeps its images on the
//! device, [`lens_device`](super::lens_device) runs the stages as kernels per tile, reading the raw
//! group where it lives, and the caches hold device images (no readback, `docs/21` section 3). L1 still
//! runs on the host where it acts (its output is uploaded as the input of L2). Any other request
//! reads the raw group back ([`Job::group`]) and the lens runs on the host; its output is a host image,
//! which every consumer (the sampler and the colour pass on the device, the CPU colour pass, the export
//! tail) takes. The raw chain of a full-resolution region still runs on the device when it can, with a
//! host result.
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
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensError, LensPlan, LensPlanes,
    PixelRect, ReshapeReason, ReshapeReport, add_flares, apply, apply_field, field_halo_px,
    flare_terms, flares_halo_px, halo_px,
};
use ym_profiles::profile::Profile;
use ym_raw::makernote::SonyFocus;

use super::{Backing, Job, Pixels, Renderer, ResidentPart, TileImage, lock};
use crate::pipeline::RenderError;
use crate::pipeline::geometry::sampler_warp;
use crate::pipeline::raw::RawParamsKey;

/// The note of a render whose camera curves could not be read: the lens colours use a neutral camera.
const CAMERA_NOTE: &str =
    "The camera's colour curves could not be read, so the lens colours are neutral.";

use super::lens_device;
/// The key of a reduced group after the lens: the raw group's, the lens's (`LensCtx::key`) and k.
pub(crate) use super::lens_device::LensDevice;

pub(crate) type GroupKey = (RawParamsKey, String, usize);
/// The key of a 512 px tile after the lens: the raw group's, the lens's and the tile.
pub(crate) type TileKey = (RawParamsKey, String, usize, usize);
// The field stage's caches use the same shapes of key with the lens's `field_key` in the second place.

/// One photo's lens for one set of inputs (the plan, with what the cache keys need).
pub(crate) struct LensCtx {
    /// The lens bound to the photo.
    pub plan: LensPlan,
    /// Everything that decides the pixels after the lens besides the raw group: the plan's key
    /// (lens, data versions, amounts, camera, frame, axis, f-number, clip levels) and the focus
    /// point the planes were made with. In the frame and display keys, and in the keys of the lens
    /// caches.
    pub key: String,
    /// What `Z = L2(L1(X))` depends on besides the raw group: [`LensPlan::field_key`] (not Flares)
    /// and the focus point the planes were made with. In the keys of the field stage's caches, so a
    /// Flares move keeps `Z`.
    pub field_key: String,
    /// What the flare terms of L3 depend on besides the raw group: [`LensPlan::flares_key`] (not
    /// Edges). In the keys of the flare caches, so an Edges move keeps them.
    pub flares_key: String,
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

/// Which part of the lens a pass runs (`ym_lens::apply`, `apply_field`, `flare_terms`).
enum LensPass {
    /// L1 to L3.
    All,
    /// `Z = L2(L1(X))`.
    Field,
}

pub(super) fn lens_error(e: LensError) -> RenderError {
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
        let field_key = format!("{}|focus{:?}", plan.field_key(), focus_bits(focus));
        let flares_key = plan.flares_key();
        // L4 on the full-resolution canvas: the zoom is fitted here, once, and reduced canvases
        // reuse it (the geometry's grids take their own canvas size).
        let (canvas_w, canvas_h) = base.canvas_size(raw_plan.out_w, raw_plan.out_h);
        let l4 = plan.warp(canvas_w as f64, canvas_h as f64);
        let warp = sampler_warp(&l4);
        let ctx = Arc::new(LensCtx {
            plan,
            key,
            field_key,
            flares_key,
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

    /// One of the lens passes over `out` of the frame at scale `k`, with `input` placed at `(x0, y0)` of
    /// a frame of `frame` pixels. The result keeps its place in the frame.
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
        pass: LensPass,
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
        let data = match pass {
            LensPass::All => apply(&ctx.plan, planes, k, &image, out),
            LensPass::Field => apply_field(&ctx.plan, planes, k, &image, out),
        }
        .map_err(lens_error)?;
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

    /// The raw group at 1/`k` after the lens's field stage, `Z = L2(L1(X))`, which Flares does not touch:
    /// from the field cache, else made from `raw` (the group) and cached by the amounts it reads. With
    /// Edges and Anamorphic at 0 the stage does nothing and `Z` is the raw group itself. (The CPU lens;
    /// a field stage the device made is not read here.)
    fn lens_field_group(
        &mut self,
        ctx: &LensCtx,
        planes: &LensPlanes,
        k: usize,
        raw: &Arc<RgbImage>,
    ) -> Result<Arc<RgbImage>, RenderError> {
        if ctx.plan.field_is_identity() {
            return Ok(Arc::clone(raw));
        }
        let key: GroupKey = (self.params()?.key(), ctx.field_key.clone(), k);
        let hit = lock(&self.g.state)
            .lens_field_groups
            .as_mut()
            .and_then(|groups| groups.get(&key).cloned());
        if let Some(Backing::Cpu(z)) = hit {
            return Ok(z);
        }
        let (w, h) = (raw.width, raw.height);
        let z = self.lens_apply(
            ctx,
            planes,
            k,
            raw,
            (0, 0),
            (w, h),
            PixelRect::new(0, 0, w, h),
            LensPass::Field,
        )?;
        self.runs.lens_fields += 1;
        let z = Arc::new(z);
        let (bytes, budget) = (z.data.len() * 4, self.g.lens_field_group_budget());
        lock(&self.g.state)
            .lens_field_groups
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, Backing::Cpu(Arc::clone(&z)), bytes);
        Ok(z)
    }

    /// `(1 - a) Z + terms`: the lens's output from the field stage's picture and the flare terms of the
    /// same rectangle, placed at `place` in the frame.
    fn lens_add_flares(
        &mut self,
        ctx: &LensCtx,
        z: &RgbImage,
        terms: &[f64],
        place: Placement,
    ) -> Result<RgbImage, RenderError> {
        self.check()?;
        let t = Instant::now();
        let data = add_flares(&ctx.plan, &z.data, terms).map_err(lens_error)?;
        self.timings.lens_ms += t.elapsed().as_secs_f64() * 1000.0;
        let mut image =
            RgbImage::from_planar(z.width, z.height, data).map_err(RenderError::from_cpu)?;
        image.place = place;
        Ok(image)
    }

    /// The flare terms of the raw group at 1/`k`, `F (glow + veil + ghosts + ...)`: from the flare
    /// cache, else made from `raw` (the group) and cached by the amounts they read. They do not depend
    /// on Edges.
    fn lens_flare_group(
        &mut self,
        ctx: &LensCtx,
        planes: &LensPlanes,
        k: usize,
        raw: &RgbImage,
    ) -> Result<Arc<Vec<f64>>, RenderError> {
        let key: GroupKey = (self.params()?.key(), ctx.flares_key.clone(), k);
        let hit = lock(&self.g.state)
            .lens_flare_groups
            .as_mut()
            .and_then(|groups| groups.get(&key).cloned());
        if let Some(f) = hit {
            return Ok(f);
        }
        let (w, h) = (raw.width, raw.height);
        let f = Arc::new(self.lens_terms(
            ctx,
            planes,
            k,
            raw,
            (0, 0),
            (w, h),
            PixelRect::new(0, 0, w, h),
        )?);
        self.runs.lens_flares += 1;
        let (bytes, budget) = (f.len() * 8, self.g.lens_flare_group_budget());
        lock(&self.g.state)
            .lens_flare_groups
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, Arc::clone(&f), bytes);
        Ok(f)
    }

    /// The flare terms over `out` ([`flare_terms`]); the plan has flares, so there are some.
    #[allow(clippy::too_many_arguments)]
    fn lens_terms(
        &mut self,
        ctx: &LensCtx,
        planes: &LensPlanes,
        k: usize,
        input: &RgbImage,
        at: (usize, usize),
        frame: (usize, usize),
        out: PixelRect,
    ) -> Result<Vec<f64>, RenderError> {
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
        let terms = flare_terms(&ctx.plan, planes, k, &image, out).map_err(lens_error)?;
        self.timings.lens_ms += t.elapsed().as_secs_f64() * 1000.0;
        // Flares above 0 always add light; a plan without any fraction of light has none.
        Ok(terms.unwrap_or_else(|| vec![0.0; 3 * out.w * out.h]))
    }

    /// The raw group at 1/`k` after the lens, for a fit view, a thumbnail, a look tile or the
    /// histogram: the lens over the whole frame at that scale (`docs/21` section 3, "Regions"), cached
    /// per lens setting. Without a lens it is [`Job::group_backing`], untouched. With one it is made on
    /// the device when the lens stage is on and the caches keep their images there (and then stays on the
    /// device, no pixel read back), else on the host.
    ///
    /// The field stage (`Z = L2(L1(X))`) is cached on its own by the amounts it reads, so a Flares move
    /// runs L3 alone, and an Edges or Anamorphic move runs the field stage and L3.
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
        if let Some(backing) = hit {
            return Ok(backing);
        }
        self.check()?;
        let made = match self.lens_group_on_device(&ctx, k)? {
            Some(made) => made,
            None => Backing::Cpu(self.lens_group_cpu(&ctx, k)?),
        };
        self.runs.lens_groups += 1;
        let (bytes, budget) = (made.bytes(), self.g.lens_group_budget_for(&made));
        lock(&self.g.state)
            .lens_groups
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, made.clone(), bytes);
        Ok(made)
    }

    /// The CPU lens over the raw group at 1/`k`: the field stage (cached), the flare terms (cached) and their
    /// sum.
    fn lens_group_cpu(
        &mut self,
        ctx: &Arc<LensCtx>,
        k: usize,
    ) -> Result<Arc<RgbImage>, RenderError> {
        let planes = self.lens_planes(ctx)?;
        let raw = self.group(k)?;
        let (w, h) = (raw.width, raw.height);
        let z = self.lens_field_group(ctx, &planes, k, &raw)?;
        if ctx.plan.flares() == 0.0 {
            // No flares: the lens's output is the field stage's.
            return Ok(z);
        }
        let f = self.lens_flare_group(ctx, &planes, k, &raw)?;
        let place = Placement {
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
        };
        Ok(Arc::new(self.lens_add_flares(ctx, &z, &f, place)?))
    }

    /// The group again on the host, for [`PartSource::LensGroup`](super::PartSource) when the device
    /// cannot hand its image over.
    pub(super) fn lens_make_group_on_host(&mut self, k: usize) -> Result<RgbImage, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return Err(RenderError::Image(
                "The lens of this photo is gone.".to_owned(),
            ));
        };
        self.lens_group_cpu(&ctx, k).map(|image| (*image).clone())
    }

    /// Puts a host image in the place of the cache entry of a lens group or tile.
    pub(super) fn lens_keep_host(&mut self, source: super::PartSource, image: &Arc<RgbImage>) {
        let Ok(Some(ctx)) = self.lens_ctx() else {
            return;
        };
        let Ok(params) = self.params() else { return };
        let (raw, bytes) = (params.key(), image.data.len() * 4);
        let mut st = lock(&self.g.state);
        match source {
            super::PartSource::LensGroup(k) => {
                if let Some(groups) = st.lens_groups.as_mut() {
                    groups.insert(
                        (raw, ctx.key.clone(), k),
                        Backing::Cpu(Arc::clone(image)),
                        bytes,
                    );
                }
            }
            super::PartSource::LensTile(tx, ty) => {
                if let Some(tiles) = st.lens_tiles.as_mut() {
                    tiles.insert(
                        (raw, ctx.key.clone(), tx, ty),
                        Backing::Cpu(Arc::clone(image)),
                        bytes,
                    );
                }
            }
            _ => {}
        }
    }

    /// The lens over the raw group at 1/`k` on the device, tile by tile of 1024 px, or `None` when the CPU
    /// makes it (no device route, a device error that falls back, a lens that does nothing on the device).
    /// The field stage's tiles are cached on the device on their own.
    fn lens_group_on_device(
        &mut self,
        ctx: &Arc<LensCtx>,
        k: usize,
    ) -> Result<Option<Backing>, RenderError> {
        let Some(route) = self.lens_gpu(true)? else {
            return Ok(None);
        };
        let planes = self.lens_planes(ctx)?;
        let raw = self.group_backing(k)?;
        let (w, h) = raw.size();
        let z_key: GroupKey = (self.params()?.key(), ctx.field_key.clone(), k);
        let z_cached = if ctx.plan.field_is_identity() {
            None
        } else {
            lock(&self.g.state)
                .lens_field_groups
                .as_mut()
                .and_then(|groups| groups.get(&z_key).cloned())
        };
        let src = lens_device::RawSrc::Group(k, raw);
        let tile = lens_device::GROUP_TILE_PX;
        let mut pieces = Vec::new();
        let mut z_pieces = Vec::new();
        let mut made_z = false;
        for ty in 0..h.div_ceil(tile) {
            for tx in 0..w.div_ceil(tile) {
                self.check()?;
                let out = Rect::new(
                    tx * tile,
                    ty * tile,
                    tile.min(w - tx * tile),
                    tile.min(h - ty * tile),
                );
                let done = self.lens_device_tile(
                    &route,
                    ctx,
                    &planes,
                    k,
                    (w, h),
                    out,
                    &src,
                    z_cached.as_ref(),
                )?;
                match done {
                    Ok(t) => {
                        pieces.push(ym_gpu::kernels::copy::Piece {
                            at: (out.x, out.y),
                            image: t.out,
                        });
                        if let Some(z) = t.z {
                            made_z = true;
                            z_pieces.push(ym_gpu::kernels::copy::Piece {
                                at: (out.x, out.y),
                                image: z,
                            });
                        }
                    }
                    Err(e) => {
                        self.lens_gpu_failed(&e)?;
                        return Ok(None);
                    }
                }
            }
        }
        if made_z {
            self.runs.lens_fields += 1;
            let z = self.lens_resident(z_pieces, (w, h), &route);
            let (bytes, budget) = (z.bytes(), self.g.lens_field_group_budget_for(&z));
            lock(&self.g.state)
                .lens_field_groups
                .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
                .insert(z_key, z, bytes);
        }
        if ctx.plan.flares() > 0.0 {
            self.runs.lens_flares += 1;
        }
        Ok(Some(self.lens_resident(pieces, (w, h), &route)))
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

    /// The lens output over `rect` (coordinates of the auto-cropped image), made from the raw
    /// region grown by the lens halo and clamped to the frame, in one call (L1 to L3). `keep_raw` lets
    /// the raw part come from the cached raw tiles or from one uncached call (an export).
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
        self.lens_apply(
            ctx,
            planes,
            1,
            &raw,
            (want.x, want.y),
            (fw, fh),
            out,
            LensPass::All,
        )
    }

    /// The rectangle of the 512 px tile `(tx, ty)` of the auto-cropped image.
    fn lens_tile_rect(&mut self, tx: usize, ty: usize) -> Result<Rect, RenderError> {
        let plan = self.plan()?;
        let t = self.opts.tile_px.max(64);
        Ok(Rect::new(
            tx * t,
            ty * t,
            t.min(plan.out_w - tx * t),
            t.min(plan.out_h - ty * t),
        ))
    }

    /// One 512 px tile of the full-resolution raw group after the lens's field stage, `Z = L2(L1(X))`,
    /// from the field cache or made from the raw region grown by the field halo (the CPU lens).
    fn lens_field_tile(
        &mut self,
        ctx: &Arc<LensCtx>,
        planes: &LensPlanes,
        tx: usize,
        ty: usize,
    ) -> Result<Arc<RgbImage>, RenderError> {
        let key: TileKey = (self.params()?.key(), ctx.field_key.clone(), tx, ty);
        let hit = lock(&self.g.state)
            .lens_field_tiles
            .as_mut()
            .and_then(|tiles| tiles.get(&key).cloned());
        if let Some(Backing::Cpu(z)) = hit {
            return Ok(z);
        }
        self.check()?;
        let plan = self.plan()?;
        let (fw, fh) = (plan.out_w, plan.out_h);
        let rect = self.lens_tile_rect(tx, ty)?;
        let out = PixelRect::new(rect.x, rect.y, rect.w, rect.h);
        let want = out.grown(field_halo_px(&ctx.plan, planes, 1), fw, fh);
        let raw = self.region(Rect::new(want.x, want.y, want.w, want.h), true)?;
        let z = Arc::new(self.lens_apply(
            ctx,
            planes,
            1,
            &raw,
            (want.x, want.y),
            (fw, fh),
            out,
            LensPass::Field,
        )?);
        self.runs.lens_fields += 1;
        let (bytes, budget) = (z.data.len() * 4, self.g.lens_field_tile_budget());
        lock(&self.g.state)
            .lens_field_tiles
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, Backing::Cpu(Arc::clone(&z)), bytes);
        Ok(z)
    }

    /// The flare terms of the 512 px tile `(tx, ty)`: from the flare cache, else made from the raw
    /// region grown by the glow core's reach.
    fn lens_flare_tile(
        &mut self,
        ctx: &Arc<LensCtx>,
        planes: &LensPlanes,
        tx: usize,
        ty: usize,
    ) -> Result<Arc<Vec<f64>>, RenderError> {
        let key: TileKey = (self.params()?.key(), ctx.flares_key.clone(), tx, ty);
        let hit = lock(&self.g.state)
            .lens_flare_tiles
            .as_mut()
            .and_then(|tiles| tiles.get(&key).cloned());
        if let Some(f) = hit {
            return Ok(f);
        }
        self.check()?;
        let plan = self.plan()?;
        let (fw, fh) = (plan.out_w, plan.out_h);
        let rect = self.lens_tile_rect(tx, ty)?;
        let out = PixelRect::new(rect.x, rect.y, rect.w, rect.h);
        let want = out.grown(flares_halo_px(&ctx.plan, planes, 1), fw, fh);
        let raw = self.region(Rect::new(want.x, want.y, want.w, want.h), true)?;
        let f = Arc::new(self.lens_terms(ctx, planes, 1, &raw, (want.x, want.y), (fw, fh), out)?);
        self.runs.lens_flares += 1;
        let (bytes, budget) = (f.len() * 8, self.g.lens_flare_tile_budget());
        lock(&self.g.state)
            .lens_flare_tiles
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, Arc::clone(&f), bytes);
        Ok(f)
    }

    /// One 512 px tile of the full-resolution raw group after the lens on the CPU: the field stage's
    /// tile (cached on its own by the amounts it reads) and L3 on it.
    fn lens_tile_cpu(
        &mut self,
        ctx: &Arc<LensCtx>,
        tx: usize,
        ty: usize,
    ) -> Result<Arc<RgbImage>, RenderError> {
        self.check()?;
        let planes = self.lens_planes(ctx)?;
        let z = if ctx.plan.field_is_identity() {
            None
        } else {
            Some(self.lens_field_tile(ctx, &planes, tx, ty)?)
        };
        Ok(match z {
            // No flares: the lens's output is the field stage's.
            Some(z) if ctx.plan.flares() == 0.0 => z,
            z => {
                let f = self.lens_flare_tile(ctx, &planes, tx, ty)?;
                let plan = self.plan()?;
                let rect = self.lens_tile_rect(tx, ty)?;
                let place = Placement {
                    x0: rect.x,
                    y0: rect.y,
                    frame_width: plan.out_w,
                    frame_height: plan.out_h,
                };
                match z {
                    Some(z) => Arc::new(self.lens_add_flares(ctx, &z, &f, place)?),
                    None => {
                        // Without a field stage `Z` is the raw tile itself.
                        let raw = self.region(rect, true)?;
                        Arc::new(self.lens_add_flares(ctx, &raw, &f, place)?)
                    }
                }
            }
        })
    }

    /// The tile again on the host, for [`PartSource::LensTile`](super::PartSource) when the device
    /// cannot hand its image over.
    pub(super) fn lens_make_tile_on_host(
        &mut self,
        tx: usize,
        ty: usize,
    ) -> Result<RgbImage, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return Err(RenderError::Image(
                "The lens of this photo is gone.".to_owned(),
            ));
        };
        self.lens_tile_cpu(&ctx, tx, ty)
            .map(|image| (*image).clone())
    }

    /// The lens over the 512 px tile `(tx, ty)` on the device (the raw tiles around it gathered where they
    /// are), or `None` when the CPU makes it.
    fn lens_tile_on_device(
        &mut self,
        ctx: &Arc<LensCtx>,
        tx: usize,
        ty: usize,
    ) -> Result<Option<Backing>, RenderError> {
        let Some(route) = self.lens_gpu(true)? else {
            return Ok(None);
        };
        let planes = self.lens_planes(ctx)?;
        let plan = self.plan()?;
        let frame = (plan.out_w, plan.out_h);
        let out = self.lens_tile_rect(tx, ty)?;
        let z_key: TileKey = (self.params()?.key(), ctx.field_key.clone(), tx, ty);
        let z_cached = if ctx.plan.field_is_identity() {
            None
        } else {
            lock(&self.g.state)
                .lens_field_tiles
                .as_mut()
                .and_then(|tiles| tiles.get(&z_key).cloned())
        };
        let done = self.lens_device_tile(
            &route,
            ctx,
            &planes,
            1,
            frame,
            out,
            &lens_device::RawSrc::Tiles,
            z_cached.as_ref(),
        )?;
        match done {
            Ok(t) => {
                if let Some(z) = t.z {
                    self.runs.lens_fields += 1;
                    let z = self.lens_resident(
                        vec![ym_gpu::kernels::copy::Piece {
                            at: (0, 0),
                            image: z,
                        }],
                        (out.w, out.h),
                        &route,
                    );
                    let (bytes, budget) = (z.bytes(), self.g.lens_field_tile_budget_for(&z));
                    lock(&self.g.state)
                        .lens_field_tiles
                        .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
                        .insert(z_key, z, bytes);
                }
                if ctx.plan.flares() > 0.0 {
                    self.runs.lens_flares += 1;
                }
                Ok(Some(self.lens_resident(
                    vec![ym_gpu::kernels::copy::Piece {
                        at: (0, 0),
                        image: t.out,
                    }],
                    (out.w, out.h),
                    &route,
                )))
            }
            Err(e) => {
                self.lens_gpu_failed(&e)?;
                Ok(None)
            }
        }
    }

    /// One 512 px tile of the full-resolution raw group after the lens, from the cache or made: on the
    /// device where it can be, else on the CPU.
    fn lens_tile_backing(
        &mut self,
        ctx: &Arc<LensCtx>,
        tx: usize,
        ty: usize,
    ) -> Result<Backing, RenderError> {
        let key: TileKey = (self.params()?.key(), ctx.key.clone(), tx, ty);
        let hit = lock(&self.g.state)
            .lens_tiles
            .as_mut()
            .and_then(|tiles| tiles.get(&key).cloned());
        if let Some(backing) = hit {
            return Ok(backing);
        }
        self.check()?;
        let made = match self.lens_tile_on_device(ctx, tx, ty)? {
            Some(made) => made,
            None => Backing::Cpu(self.lens_tile_cpu(ctx, tx, ty)?),
        };
        self.runs.lens_tiles += 1;
        let (bytes, budget) = (made.bytes(), self.g.lens_tile_budget_for(&made));
        lock(&self.g.state)
            .lens_tiles
            .get_or_insert_with(|| crate::scheduler::ByteLru::new(budget))
            .insert(key, made.clone(), bytes);
        Ok(made)
    }

    /// One lens tile as a host image (a tile on the device is read back, and the cache keeps the host
    /// image from then on).
    fn lens_tile_host(
        &mut self,
        ctx: &Arc<LensCtx>,
        tx: usize,
        ty: usize,
    ) -> Result<Arc<RgbImage>, RenderError> {
        match self.lens_tile_backing(ctx, tx, ty)? {
            Backing::Cpu(image) => Ok(image),
            Backing::Gpu(resident) => self.demote(super::PartSource::LensTile(tx, ty), &resident),
        }
    }

    /// The lens output over `rect`, put together from cached lens tiles on the host.
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
                let tile = self.lens_tile_host(ctx, tx, ty)?;
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
    /// `rect`, out of cached lens tiles. When every tile is on the device they stay there and the sampler
    /// reads them in place ([`Pixels::Resident`]); else the host assembles the region. Without a lens it
    /// is [`Job::region_pixels`].
    ///
    /// # Errors
    ///
    /// The errors of [`Job::region_pixels`] and of the lens.
    pub(crate) fn lens_region_pixels(&mut self, rect: Rect) -> Result<Pixels, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.region_pixels(rect);
        };
        self.check_region(rect)?;
        let plan = self.plan()?;
        if !self.keeps_on_device() || plan.reduced {
            return self
                .lens_region_cached(&ctx, rect)
                .map(|image| Pixels::Host(Arc::new(image)));
        }
        let t = self.opts.tile_px.max(64);
        let mut parts = Vec::new();
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                self.check()?;
                match self.lens_tile_backing(&ctx, tx, ty)? {
                    Backing::Gpu(image) => parts.push(ResidentPart {
                        image,
                        at: (tx * t, ty * t),
                        source: super::PartSource::LensTile(tx, ty),
                    }),
                    // A tile the CPU made: the host assembles the region.
                    Backing::Cpu(_) => {
                        return self
                            .lens_region_cached(&ctx, rect)
                            .map(|image| Pixels::Host(Arc::new(image)));
                    }
                }
            }
        }
        Ok(Pixels::Resident { parts, rect })
    }

    /// The full-resolution raw group over `rect` after the lens as a host image: out of cached lens
    /// tiles with `keep`, else from the raw region grown by the lens halo in one uncached call (an
    /// export's tile). Without a lens it is [`Job::region`]. (The CPU lens: an export whose colour stage
    /// is on the device asks for [`Job::lens_region_resident`].)
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

    /// [`Job::region_resident`] after the lens: for an export whose colour stage is on the device. With the
    /// lens stage on the device the tile is made there from the raw region (itself left on the device where
    /// the raw chain made it) and stays there; else the CPU lens makes a host tile that the colour stage
    /// uploads. Without a lens it is [`Job::region_resident`].
    ///
    /// # Errors
    ///
    /// As [`Job::lens_region`]; an export's device error asks to start again on the CPU
    /// ([`Job::take_gpu_restart`]).
    pub fn lens_region_resident(&mut self, rect: Rect) -> Result<TileImage, RenderError> {
        let Some(ctx) = self.lens_ctx()? else {
            return self.region_resident(rect);
        };
        self.check_region(rect)?;
        if let Some(route) = self.lens_gpu(false)? {
            let planes = self.lens_planes(&ctx)?;
            let plan = self.plan()?;
            let frame = (plan.out_w, plan.out_h);
            let done = self.lens_device_tile(
                &route,
                &ctx,
                &planes,
                1,
                frame,
                rect,
                &lens_device::RawSrc::Export,
                None,
            )?;
            match done {
                Ok(t) => {
                    self.runs.lens_tiles += 1;
                    if !ctx.plan.field_is_identity() {
                        self.runs.lens_fields += 1;
                    }
                    if ctx.plan.flares() > 0.0 {
                        self.runs.lens_flares += 1;
                    }
                    return Ok(TileImage::Device(t.out));
                }
                Err(e) => self.lens_gpu_failed(&e)?,
            }
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
    /// too, on the device where the raw chain made them): the work ahead of a 100 % view. Without a lens
    /// it is [`Job::warm_tiles`].
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
                self.lens_tile_backing(&ctx, tx, ty)?;
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
