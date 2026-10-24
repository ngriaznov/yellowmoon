//! The lens on the device (`docs/21` sections 3 and 9, `docs/17-gpu.md` section 11): L2 and L3 run as
//! kernels per tile, from the raw group where it lives, and hand the colour pass a device image without
//! a readback.
//!
//! # What the device holds
//!
//! * per photo, uploaded once ([`LensDevice`]): the tail plane and the spike tables of the planes, the
//!   spread functions of the lens on the camera, the lens's rim and normal-tail tables, and the scale data
//!   (the glow core's taps, the items and their bins, `ym_lens::device::FlareScaleData`) of the plan shapes
//!   a few recent requests asked for. None of them depends on Flares or Edges: a Flares move uploads
//!   nothing, an Edges move changes a uniform.
//! * in the caches, as images on the device: the raw group at 1/k or the 512 px tiles after the lens
//!   (`lens_groups`, `lens_tiles`) and after its field stage alone (`lens_field_groups`,
//!   `lens_field_tiles`), so a Flares move runs L3 over the cached `Z` and nothing else.
//!
//! # A tile
//!
//! One job on the device thread per tile of the output (1024 px for a group, the raw tile grid for the
//! 100 % view, the export's rectangle): the raw window (the tile grown by the lens's halo, gathered from
//! the raw group's pieces), L2 into `Z`, L3 over `Z` into the output. Every pixel is a function of its
//! absolute position and of the input within the halo, so any tiling gives the same bits (D5).
//!
//! # Bokeh reshaping (L1)
//!
//! L1 has no kernel yet: where it acts the host runs `ym_lens::reshape` on the raw region, uploads its
//! output as the input of L2, and L3 still reads the raw picture. Everything after L1 stays on the device.
//!
//! # Failures
//!
//! A device error falls back to the CPU lens for the request with a note, and ends the photo's use of the
//! device lens (a shader that does not compile ends it for the process); a strict policy fails the
//! request; an export asks to start again wholly on the CPU (`Job::take_gpu_restart`).

use std::sync::{Arc, Weak};

use ym_gpu::cpu::{Placement, Rect, RgbImage};
use ym_gpu::device::{Gpu, GpuContext, GpuError};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::copy::{Piece, encode_crop, encode_gather};
use ym_gpu::kernels::lens_field::{FieldTablesBuffer, encode_field};
use ym_gpu::kernels::lens_flare::{
    FlareLensBuffer, FlarePlaneBuffer, FlareScaleBuffers, encode_flare,
};
use ym_gpu::stream::PassList;
use ym_lens::device::{FieldPass, FlareMix};
use ym_lens::{ImageRef, LensPlanes, PixelRect, reshape, reshape_halo};

use super::super::raw::GpuRoute;
use super::lens::{LensCtx, lens_error};
use super::{Backing, Job, ResidentImage, TileImage, lock};
use crate::app::gpu::GpuStages;
use crate::pipeline::RenderError;

/// Tile of the lens output on the device for a reduced group, in pixels.
pub(super) const GROUP_TILE_PX: usize = 1024;

/// Scale data kept per photo (the plan shapes of the last few requests).
const SCALES_KEPT: usize = 6;

/// What the device holds of a photo's lens, made on first use.
#[derive(Default)]
pub(crate) struct LensDevice {
    gpu: Option<Weak<Gpu>>,
    /// The lens's spread functions on the camera, with the key of the tables.
    field: Option<(String, FieldTablesBuffer)>,
    /// The tail plane and spike tables of the planes they were made from.
    flare_planes: Option<(Arc<LensPlanes>, FlarePlaneBuffer)>,
    /// The lens's rim, aperture and normal tail, with their key.
    flare_lens: Option<(String, FlareLensBuffer)>,
    /// The scale data of recent plan shapes, newest last, with their key.
    scales: Vec<(String, FlareScaleBuffers)>,
}

/// The pieces of a raw picture, on the device or the host, that a tile reads.
#[derive(Default)]
pub(super) struct RawIn {
    /// Images on the device with their place in the picture.
    pieces: Vec<Piece>,
    /// Images on the host with their place; the job uploads them.
    hosts: Vec<(Arc<RgbImage>, (usize, usize))>,
}

impl RawIn {
    /// Adds a cached image of the raw group whose first pixel is at `at` of the picture.
    pub(super) fn push_backing(&mut self, backing: &Backing, at: (usize, usize)) {
        match backing {
            Backing::Cpu(image) => self.hosts.push((Arc::clone(image), at)),
            Backing::Gpu(resident) => {
                for p in resident.pieces() {
                    self.pieces.push(Piece {
                        at: (at.0 + p.at.0, at.1 + p.at.1),
                        image: Arc::clone(&p.image),
                    });
                }
            }
        }
    }

    /// Adds a tile an export made, whose first pixel is at `at`.
    pub(super) fn push_tile(&mut self, tile: TileImage, at: (usize, usize)) {
        match tile {
            TileImage::Device(image) => self.pieces.push(Piece { at, image }),
            TileImage::Host(image) => self.hosts.push((Arc::new(image), at)),
        }
    }
}

/// Where a tile's raw pixels come from.
pub(super) enum RawSrc {
    /// The raw group at 1/k, as the cache holds it.
    Group(usize, Backing),
    /// The 512 px tiles of the full-resolution group (the 100 % view).
    Tiles,
    /// An export's region, made in one call.
    Export,
}

/// What a tile needs of the lens, with everything the job runs from.
struct TilePlan {
    /// The output rectangle (frame coordinates at the scale of the frame).
    out: Rect,
    /// The window of the raw picture the job gathers: the output grown by what L2 (without L1) and L3 read.
    want: Rect,
    /// L2: the spread functions and the pass, when the field stage acts and `Z` is not cached.
    field: Option<(FieldTablesBuffer, FieldPass)>,
    /// L3: the planes, the lens's tables, the scale data and the mix, when the flares add light.
    flare: Option<(FlarePlaneBuffer, FlareLensBuffer, FlareScaleBuffers, FlareMix)>,
    /// A cached `Z` over the output (an image whose first pixel is the output's), on the device.
    z_device: Option<Arc<GpuImage>>,
    /// A cached `Z` over the output on the host, uploaded by the job.
    z_host: Option<Arc<RgbImage>>,
    /// The output of L1 over the rectangle L2 reads (the host ran the reshaping), with its place.
    l1: Option<(Vec<f32>, Rect)>,
}

/// What a tile job gives back.
pub(super) struct TileOut {
    /// `Z` over the output, when the job made it (to cache).
    pub z: Option<Arc<GpuImage>>,
    /// The lens's output over the output rectangle.
    pub out: Arc<GpuImage>,
}

fn crop_rect(r: Rect, o: Rect) -> Rect {
    Rect::new(o.x - r.x, o.y - r.y, o.w, o.h)
}

/// Runs a tile on the device thread.
fn run_tile(ctx: &GpuContext, raw: RawIn, tile: &TilePlan) -> Result<TileOut, GpuError> {
    ctx.inject_tile_fault()?;
    let mut passes = PassList::new(ctx, "lens tile");
    // Images this job uploads: recorded passes read them when the list is submitted, so they must not go
    // back to the pool (and be written by a later allocation of the job) before then.
    let mut held: Vec<Arc<GpuImage>> = Vec::new();
    let want = tile.want;
    // The raw window, when L2 or L3 read it.
    let reads_raw = tile.flare.is_some() || (tile.field.is_some() && tile.l1.is_none());
    let input = if reads_raw {
        let mut pieces = raw.pieces;
        for (image, at) in &raw.hosts {
            let mut uploaded = GpuImage::upload_rgb(ctx, "lens raw", image)?;
            uploaded.place = Placement::whole(image.width, image.height);
            let uploaded = Arc::new(uploaded);
            held.push(Arc::clone(&uploaded));
            pieces.push(Piece {
                at: *at,
                image: uploaded,
            });
        }
        Some(encode_gather(&mut passes, &pieces, want)?)
    } else {
        None
    };
    // Z: cached, or L2 over the raw window (or over L1's output).
    let mut made_z = None;
    let (z, z_at): (Arc<GpuImage>, (usize, usize)) = match (&tile.z_device, &tile.z_host, &tile.field) {
        (Some(z), _, _) => (Arc::clone(z), (tile.out.x, tile.out.y)),
        (None, Some(host), _) => {
            let mut z = GpuImage::upload_rgb(ctx, "lens cached z", host)?;
            z.place = Placement::whole(host.width, host.height);
            let z = Arc::new(z);
            held.push(Arc::clone(&z));
            (z, (tile.out.x, tile.out.y))
        }
        (None, None, Some((tables, pass))) => {
            let z = if let Some((data, rect)) = &tile.l1 {
                let src = GpuImage::upload(
                    ctx,
                    "lens reshaped",
                    rect.w,
                    rect.h,
                    3,
                    data,
                    Placement::whole(rect.w, rect.h),
                )?;
                let src = Arc::new(src);
                held.push(Arc::clone(&src));
                encode_field(&mut passes, tables, pass, &src, (rect.x, rect.y), tile.out)?
            } else {
                let Some(input) = &input else {
                    return Err(GpuError::InvalidInput(
                        "lens tile: the field stage has no input".into(),
                    ));
                };
                encode_field(&mut passes, tables, pass, input, (want.x, want.y), tile.out)?
            };
            let z = Arc::new(z);
            made_z = Some(Arc::clone(&z));
            (z, (tile.out.x, tile.out.y))
        }
        // No field stage: Z is the raw picture itself.
        (None, None, None) => {
            let Some(input) = &input else {
                return Err(GpuError::InvalidInput("lens tile: no input".into()));
            };
            (Arc::clone(input), (want.x, want.y))
        }
    };
    let out = match &tile.flare {
        Some((planes, lens, scale, mix)) => {
            let Some(input) = &input else {
                return Err(GpuError::InvalidInput(
                    "lens tile: the flares have no input".into(),
                ));
            };
            Arc::new(encode_flare(
                &mut passes,
                planes,
                lens,
                scale,
                *mix,
                input,
                (want.x, want.y),
                &z,
                z_at,
                tile.out,
            )?)
        }
        None => {
            if z_at == (tile.out.x, tile.out.y) && z.width == tile.out.w && z.height == tile.out.h
            {
                z
            } else {
                // Z is the raw window: the output is its part.
                let part = crop_rect(Rect::new(z_at.0, z_at.1, z.width, z.height), tile.out);
                let mut img = encode_crop(&mut passes, &z, part)?;
                img.place = Placement {
                    x0: tile.out.x,
                    y0: tile.out.y,
                    frame_width: z.place.frame_width,
                    frame_height: z.place.frame_height,
                };
                Arc::new(img)
            }
        }
    };
    passes.submit()?.finish()?;
    drop(held);
    Ok(TileOut { z: made_z, out })
}

impl Job<'_> {
    /// The route to the device for the lens of this request, or `None` when the CPU runs it: there is no
    /// device, the lens stage is not on ([`GpuStages::LENS`]), a view that keeps its images on the device
    /// is not what the request makes (a request whose results would be read back anyway uses the CPU lens),
    /// a device error ended the route for this photo or the process, or the device is lost.
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] when the policy is strict and the route is gone.
    pub(super) fn lens_gpu(&mut self, results_stay: bool) -> Result<Option<GpuRoute>, RenderError> {
        let Some(gpu) = self.opts.gpu.clone() else {
            return Ok(None);
        };
        if !self.opts.gpu_stages.contains(GpuStages::LENS) {
            return Ok(None);
        }
        if results_stay && !self.keeps_on_device() {
            return Ok(None);
        }
        if self.g.lens_gpu_failed.load(std::sync::atomic::Ordering::SeqCst)
            || self
                .r
                .inner
                .lens_gpu_disabled
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            if self.opts.gpu_strict {
                return Err(RenderError::Gpu(
                    "an earlier error ended the use of the graphics card for the lens, and the strict setting does not let the CPU take over"
                        .to_owned(),
                ));
            }
            return Ok(None);
        }
        if gpu.is_lost() {
            let why = gpu
                .lost_reason()
                .unwrap_or_else(|| "the device was lost".to_owned());
            if self.opts.gpu_strict {
                return Err(RenderError::Gpu(why));
            }
            self.note(&format!(
                "The graphics card stopped working, so the CPU did: {why}."
            ));
            return Ok(None);
        }
        Ok(Some(GpuRoute {
            gpu,
            range: super::super::raw::DEVICE_RANGE,
            spec: ym_gpu::queue::JobSpec::new(self.gpu_priority).owned_by(u64::from(self.src.id.0)),
            token: self.cancel.clone(),
            resident: None,
        }))
    }

    /// A lens stage failed on the device with `error`. Cancelled stays cancelled; a shader that does not
    /// compile ends the device lens for the process; a strict policy fails the request; an export asks to
    /// start again on the CPU; any other request falls back to the CPU lens with a note, and one whose
    /// error is not about its size ends the photo's use of the device lens.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`], or [`RenderError::Gpu`] for a strict policy or an export's restart.
    pub(super) fn lens_gpu_failed(&mut self, error: &GpuError) -> Result<(), RenderError> {
        use std::sync::atomic::Ordering;
        if matches!(error, GpuError::Cancelled) {
            return Err(RenderError::Cancelled);
        }
        let reason = error.to_string();
        tracing::warn!(error = %reason, "the device failed on the lens; the CPU takes over");
        if matches!(error, GpuError::Shader(_)) {
            self.r.inner.lens_gpu_disabled.store(true, Ordering::SeqCst);
        }
        if self.opts.gpu_strict {
            return Err(RenderError::Gpu(reason));
        }
        let about_size = matches!(
            error,
            GpuError::OutOfBudget { .. } | GpuError::BufferTooLarge { .. }
        );
        if !about_size {
            self.g.lens_gpu_failed.store(true, Ordering::SeqCst);
        }
        if self.restartable {
            // One file is wholly device or wholly CPU: start again on the CPU, where the photo's route is
            // now off.
            self.g.raw_gpu_failed.store(true, Ordering::SeqCst);
            self.raw_ranges.clear();
            self.restart = Some(reason.clone());
            return Err(RenderError::Gpu(reason));
        }
        self.note(&format!(
            "The graphics card could not apply the lens, so the CPU did: {reason}."
        ));
        Ok(())
    }

    /// Takes the photo's lens device state (an export's own, a viewer's from the graph) to use it
    /// outside the lock; put it back with [`Job::put_lens_device`].
    fn take_lens_device(&mut self) -> LensDevice {
        if self.restartable {
            self.lens_job_device.take().unwrap_or_default()
        } else {
            lock(&self.g.state).lens_device.take().unwrap_or_default()
        }
    }

    fn put_lens_device(&mut self, device: LensDevice) {
        if self.restartable {
            self.lens_job_device = Some(device);
        } else {
            lock(&self.g.state).lens_device = Some(device);
        }
    }

    /// Makes the tile plan's buffers for the photo's lens at the frame `frame` pixels (the scale of the
    /// caller): uploads the ones the device does not hold yet, in jobs of the route.
    #[allow(clippy::type_complexity)]
    fn lens_device_buffers(
        &mut self,
        route: &GpuRoute,
        ctx: &LensCtx,
        planes: &Arc<LensPlanes>,
        frame: (usize, usize),
        need_field: bool,
        need_flare: bool,
    ) -> Result<
        (
            Option<(FieldTablesBuffer, FieldPass)>,
            Option<(FlarePlaneBuffer, FlareLensBuffer, FlareScaleBuffers, FlareMix)>,
        ),
        GpuError,
    > {
        let mut device = self.take_lens_device();
        let result = (|| {
            // A new device (the renderer was given one) makes everything again.
            let same = device
                .gpu
                .as_ref()
                .is_some_and(|w| Weak::ptr_eq(w, &Arc::downgrade(&route.gpu)));
            if !same {
                device = LensDevice {
                    gpu: Some(Arc::downgrade(&route.gpu)),
                    ..LensDevice::default()
                };
            }
            let field = if need_field {
                match ctx.plan.field_pass(frame.0, frame.1) {
                    Some(pass) => {
                        let data = ctx.plan.field_tables_data();
                        let tables = match (&device.field, data) {
                            (Some((key, t)), Some(d)) if *key == d.key => t.clone(),
                            (_, Some(d)) => {
                                let key = d.key.clone();
                                let t = route.run_job(move |c, _| FieldTablesBuffer::upload(c, &d))?;
                                device.field = Some((key, t.clone()));
                                t
                            }
                            (_, None) => {
                                return Err(GpuError::InvalidInput(
                                    "the lens has a field pass but no tables".into(),
                                ));
                            }
                        };
                        Some((tables, pass))
                    }
                    None => None,
                }
            } else {
                None
            };
            let flare = if need_flare {
                let shape = format!(
                    "{}|{}x{}|{:p}",
                    ctx.plan.flare_shape_key(),
                    frame.0,
                    frame.1,
                    Arc::as_ptr(planes)
                );
                match ctx.plan.flare_scale_data(planes, frame.0, frame.1) {
                    None => None,
                    Some(data) => {
                        let plane_buf = match &device.flare_planes {
                            Some((p, b)) if Arc::ptr_eq(p, planes) => b.clone(),
                            _ => {
                                let Some(d) = planes.flare_plane_data() else {
                                    return Err(GpuError::InvalidInput(
                                        "the planes hold no highlight data".into(),
                                    ));
                                };
                                let b = route.run_job(move |c, _| FlarePlaneBuffer::upload(c, &d))?;
                                device.flare_planes = Some((Arc::clone(planes), b.clone()));
                                b
                            }
                        };
                        let lens_data = ctx.plan.flare_lens_data();
                        let lens_buf = match &device.flare_lens {
                            Some((key, b)) if *key == lens_data.key => b.clone(),
                            _ => {
                                let key = lens_data.key.clone();
                                let b = route
                                    .run_job(move |c, _| FlareLensBuffer::upload(c, &lens_data))?;
                                device.flare_lens = Some((key, b.clone()));
                                b
                            }
                        };
                        let scale_buf = match device.scales.iter().position(|(k, _)| *k == shape) {
                            Some(i) => {
                                let entry = device.scales.remove(i);
                                let b = entry.1.clone();
                                device.scales.push(entry);
                                b
                            }
                            None => {
                                let b = route
                                    .run_job(move |c, _| FlareScaleBuffers::upload(c, &data))?;
                                device.scales.push((shape, b.clone()));
                                while device.scales.len() > SCALES_KEPT {
                                    device.scales.remove(0);
                                }
                                b
                            }
                        };
                        Some((plane_buf, lens_buf, scale_buf, ctx.plan.flare_mix()))
                    }
                }
            } else {
                None
            };
            Ok((field, flare))
        })();
        self.put_lens_device(device);
        result
    }

    /// The raw group's pieces for the lens, and `Z` of the tile when a cache holds it.
    fn lens_z_for(z: Option<&Backing>, out: Rect) -> (Option<Arc<GpuImage>>, Option<Arc<RgbImage>>) {
        match z {
            Some(Backing::Gpu(resident)) => {
                // A tile's own `Z` is one piece at the origin of an image of the tile's size; a group's
                // `Z` holds the tile as a piece at the tile's place in the frame.
                let at = if (resident.width, resident.height) == (out.w, out.h) {
                    (0, 0)
                } else {
                    (out.x, out.y)
                };
                let hit = resident
                    .pieces()
                    .iter()
                    .find(|p| p.at == at && p.image.width == out.w && p.image.height == out.h);
                (hit.map(|p| Arc::clone(&p.image)), None)
            }
            Some(Backing::Cpu(image)) => {
                // A host `Z` of the whole group or tile: the part over the output.
                let part = if (image.width, image.height) == (out.w, out.h) {
                    Arc::clone(image)
                } else {
                    Arc::new(image.extract(out))
                };
                (None, Some(part))
            }
            None => (None, None),
        }
    }

    /// One tile of the lens on the device. `raw` holds the pieces of the raw picture around the tile;
    /// `frame` is the frame at this scale; `z` is the cached field stage of the photo (a group's or a
    /// tile's) when there is one.
    ///
    /// Returns the tile's output and the `Z` the job made (to cache).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lens_device_tile(
        &mut self,
        route: &GpuRoute,
        ctx: &Arc<LensCtx>,
        planes: &Arc<LensPlanes>,
        k: usize,
        frame: (usize, usize),
        out: Rect,
        src: &RawSrc,
        z: Option<&Backing>,
    ) -> Result<Result<TileOut, GpuError>, RenderError> {
        let need_field = !ctx.plan.field_is_identity();
        let (z_device, z_host) = Self::lens_z_for(z, out);
        let z_cached = z_device.is_some() || z_host.is_some();
        let need_flare = ctx.plan.flares() > 0.0;
        let (field, flare) = match self.lens_device_buffers(
            route,
            ctx,
            planes,
            frame,
            need_field && !z_cached,
            need_flare,
        ) {
            Ok(b) => b,
            Err(e) => return Ok(Err(e)),
        };
        let clamp = |r: PixelRect| Rect::new(r.x, r.y, r.w, r.h);
        let grown = |px: usize| {
            clamp(PixelRect::new(out.x, out.y, out.w, out.h).grown(px, frame.0, frame.1))
        };
        let field_halo = field.as_ref().map_or(0, |(_, p)| p.halo());
        let flare_halo = flare.as_ref().map_or(0, |(_, _, s, _)| s.halo());
        // L1 where it acts: the host reshapes the raw region L2 reads and L2 reads that.
        let l1_halo = if field.is_some() {
            reshape_halo(&ctx.plan, planes, k)
        } else {
            0
        };
        let mut l1 = None;
        let mut want = grown(flare_halo.max(if l1_halo == 0 { field_halo } else { 0 }));
        if l1_halo > 0 {
            let l2_rect = grown(field_halo);
            let need = clamp(
                PixelRect::new(l2_rect.x, l2_rect.y, l2_rect.w, l2_rect.h)
                    .grown(l1_halo, frame.0, frame.1),
            );
            let (image, origin) = self.lens_host_raw(src, need)?;
            let input = ImageRef {
                width: image.width,
                height: image.height,
                x0: origin.0,
                y0: origin.1,
                frame_width: frame.0,
                frame_height: frame.1,
                data: &image.data,
            };
            let t = std::time::Instant::now();
            let data = reshape(
                &ctx.plan,
                planes,
                k,
                &input,
                PixelRect::new(l2_rect.x, l2_rect.y, l2_rect.w, l2_rect.h),
            )
            .map_err(lens_error)?;
            self.timings.lens_ms += t.elapsed().as_secs_f64() * 1000.0;
            l1 = Some((data, l2_rect));
            want = grown(flare_halo);
        }
        let reads_raw = flare.is_some() || (field.is_some() && l1.is_none());
        let raw = if reads_raw {
            self.lens_raw_in(src, want)?
        } else {
            RawIn::default()
        };
        let tile = TilePlan {
            out,
            want,
            field,
            flare,
            z_device,
            z_host,
            l1,
        };
        let t = std::time::Instant::now();
        let made = route.run_job(move |c, _| run_tile(c, raw, &tile));
        self.timings.lens_ms += t.elapsed().as_secs_f64() * 1000.0;
        Ok(made)
    }

    /// The pieces of the raw picture that cover `want`.
    fn lens_raw_in(&mut self, src: &RawSrc, want: Rect) -> Result<RawIn, RenderError> {
        let mut raw = RawIn::default();
        match src {
            RawSrc::Group(_, backing) => raw.push_backing(backing, (0, 0)),
            RawSrc::Tiles => {
                let t = self.opts.tile_px.max(64);
                for ty in want.y / t..=(want.y1() - 1) / t {
                    for tx in want.x / t..=(want.x1() - 1) / t {
                        self.check()?;
                        let backing = self.tile_backing(tx, ty, true)?;
                        raw.push_backing(&backing, (tx * t, ty * t));
                    }
                }
            }
            RawSrc::Export => {
                let tile = self.region_resident(want)?;
                raw.push_tile(tile, (want.x, want.y));
            }
        }
        Ok(raw)
    }

    /// A host image of the raw picture around `need`, with the place of its first pixel in the frame (the
    /// reshaping runs on the host).
    fn lens_host_raw(
        &mut self,
        src: &RawSrc,
        need: Rect,
    ) -> Result<(Arc<RgbImage>, (usize, usize)), RenderError> {
        match src {
            RawSrc::Group(k, _) => Ok((self.group(*k)?, (0, 0))),
            RawSrc::Tiles => Ok((Arc::new(self.region(need, true)?), (need.x, need.y))),
            RawSrc::Export => Ok((Arc::new(self.region(need, false)?), (need.x, need.y))),
        }
    }

    /// The device image of a finished tile as a cache entry.
    pub(super) fn lens_resident(
        &self,
        pieces: Vec<Piece>,
        size: (usize, usize),
        route: &GpuRoute,
    ) -> Backing {
        Backing::Gpu(Arc::new(ResidentImage::of_pieces(
            pieces, size.0, size.1, &route.gpu,
        )))
    }
}
