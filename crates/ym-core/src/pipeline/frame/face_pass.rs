//! The Face Refinement node's step on the CPU reference path (`docs/27-skin-retouch.md` section 10, S-11b, decisions 9 and 13).
//!
//! The node is a picture step of the colour chain like a Film node (`film_pass`): the chain stops at the node's input
//! (the prefix chain, no look, LogC4 / AWG4), the node's picture is made from it, and the chain blends it by the node's mask
//! (`run_film`). What differs is what makes the picture. A face is a small part of the frame and every tool changes only
//! pixels inside a **region** around it (`ym_retouch::retouch_regions`: the rectangle its tools read and write, rectangles that
//! overlap merged), so the step makes the picture over those regions only:
//!
//! 1. the regions come from the cached analysis ([`FaceEntry`]) and the frame's size, at the render's scale
//!    (`working / analysis`);
//! 2. a render that needs a pixel of a region needs the whole region ([`FaceWork::widen`] grows the area the render asks for
//!    to the regions it meets, and to the reach of the protection plane's edges);
//! 3. per region, the prefix chain's LogC4 over it is decoded to linear AWG4, `retouch_skin_region` applies the cached masks and
//!    blemish maps to it (no network runs, whatever the grade upstream), the result is encoded back, and the pixels it left as
//!    they were are marked so that the chain passes them through exactly (a bit-identical pass-through outside the face's
//!    effect, not a round trip through f32);
//! 4. the picture is those regions; the chain reads nothing there for any other pixel and passes its input through.
//!
//! An export asks for tiles; a region met by several tiles is made once ([`FaceWork::part`] keeps what it made).

use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_color::mask_raster::{Area, PlaneJob, RasterRow, RasterSteps};
use ym_gpu::cpu::sampler::{GridPositions, SamplerGrid};
use ym_retouch::{Face, LinearAwg4Image, Mask, Placement, SkinSettings, retouch_regions};

use super::RenderError;
use crate::pipeline::graph::FaceEntry;
use crate::settings::FaceRefinementNode;

/// The picture of one region: LogC4 / AWG4, interleaved f32, with `NaN` for a pixel the retouch left as it was (the chain
/// then passes its input through for it, exactly).
#[derive(Debug)]
pub(crate) struct FacePart {
    pub area: Area,
    pub data: Vec<f32>,
}

/// What the protection plane of a region is made from: the camera picture around it and where its pixels are in the photo.
pub(crate) struct ProtectInput<'a> {
    pub grid: &'a SamplerGrid,
    /// The grid pixel of the picture's first pixel.
    pub grid_off: (usize, usize),
    /// The area of the frame the picture covers.
    pub area: Area,
    /// The picture, planar camera RGB (`3 * w * h`), and the pixels outside the photo.
    pub planar: &'a [f32],
    pub outside: Option<&'a [u8]>,
    /// The Input's luma of a camera pixel (the guide of Refine edges).
    pub guide: &'a (dyn Fn([f64; 3]) -> f64 + Sync),
}

/// One Face Refinement node of a render: its faces with their settings, the regions they are retouched in, and its protection.
pub(crate) struct FaceWork {
    entry: Arc<FaceEntry>,
    faces: Vec<(Face, SkinSettings)>,
    /// The frame's size over the analysis picture's.
    scale: f32,
    /// The frame (the photo's extent at this render's scale).
    frame: Area,
    /// The regions, in absolute pixels of the frame.
    regions: Vec<Area>,
    protect: Option<RasterRow>,
    /// The reach of the protection plane's Grow, Feather and Refine edges, in pixels of the frame.
    protect_halo: usize,
    done: Mutex<Vec<Option<Arc<FacePart>>>>,
}

fn render_error(e: &impl std::fmt::Display) -> RenderError {
    RenderError::Mask(format!("The faces can't be retouched: {e}."))
}

impl FaceWork {
    /// The node's work over a frame, or `None` when there is nothing to do: no face has a region, or the regions cannot be
    /// planned (logged; the node then passes its picture through).
    pub(crate) fn build(
        entry: &Arc<FaceEntry>,
        node: &FaceRefinementNode,
        frame: Area,
        protect: Option<RasterRow>,
    ) -> Option<Self> {
        if frame.is_empty() || entry.faces.is_empty() {
            return None;
        }
        let faces = entry.with_settings(|n| node.skin_settings_for_face(n));
        let scale = frame.w as f32 / entry.analysis_px.0.max(1) as f32;
        let working = (frame.w as u32, frame.h as u32);
        let regions = match retouch_regions(
            working,
            entry.analysis_px,
            &faces,
            scale,
            &entry.retouch,
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "the face regions could not be planned; the node passes its picture through");
                return None;
            }
        };
        if regions.is_empty() {
            return None;
        }
        let regions: Vec<Area> = regions
            .iter()
            .map(|r| Area {
                x: frame.x + i64::from(r.x_px),
                y: frame.y + i64::from(r.y_px),
                w: r.width_px as usize,
                h: r.height_px as usize,
            })
            .collect();
        let u_px = frame.w.max(frame.h) as f64;
        let protect_halo = protect.as_ref().map_or(0, |r| {
            RasterSteps::of(r.grow, r.feather, r.refine, u_px).halo() + 2
        });
        let done = Mutex::new(vec![None; regions.len()]);
        Some(Self {
            entry: Arc::clone(entry),
            faces,
            scale,
            frame,
            regions,
            protect,
            protect_halo,
            done,
        })
    }

    /// The area a render needs to make `need` of this node's picture: `need` grown to hold every region it meets (with the
    /// reach of the protection plane around it). The regions are the ones `need` meets, not the ones the grown area meets: a
    /// region the grown area happens to reach is made when a render asks for its own pixels.
    pub(crate) fn widen(&self, need: Area) -> Area {
        if need.is_empty() {
            return need;
        }
        // A region an earlier tile of the same export made is kept: a later tile does not read it again.
        self.regions
            .iter()
            .enumerate()
            .filter(|(i, r)| !need.intersect(r).is_empty() && self.kept(*i).is_none())
            .fold(need, |out, (_, r)| {
                union(out, r.grown(self.protect_halo, &self.frame))
            })
    }

    /// The regions that meet `made`, by index.
    pub(crate) fn regions_in(&self, made: &Area) -> Vec<usize> {
        (0..self.regions.len())
            .filter(|&i| !self.regions[i].intersect(made).is_empty())
            .collect()
    }

    /// The area of region `ri`.
    pub(crate) fn region(&self, ri: usize) -> Area {
        self.regions[ri]
    }

    /// What region `ri` made before, if anything.
    pub(crate) fn kept(&self, ri: usize) -> Option<Arc<FacePart>> {
        self.done
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[ri]
            .clone()
    }

    /// The protection plane over region `ri` as the library takes it: 0 to 255, 255 leaves a pixel alone. `None` without
    /// protection strokes or without a photo grid to place them.
    pub(crate) fn protect_mask(&self, ri: usize, input: &ProtectInput<'_>) -> Option<Mask> {
        let row = self.protect.as_ref()?;
        let window = self.regions[ri];
        let photo = input.grid.photo_frame();
        let steps = RasterSteps::of(row.grow, row.feather, row.refine, 1.0 / photo.pixel);
        let grown = window
            .grown(steps.halo(), &self.frame)
            .intersect(&input.area);
        if !grown.contains(&window) {
            return None;
        }
        let (cw, off) = (
            input.area.w,
            (grown.x - input.area.x, grown.y - input.area.y),
        );
        let map = GridPositions::new(
            input.grid,
            (
                input.grid_off.0 + off.0 as usize,
                input.grid_off.1 + off.1 as usize,
            ),
        );
        let position = |x: usize, y: usize| {
            let (cx, cy) = (off.0 as usize + x, off.1 as usize + y);
            if input.outside.is_some_and(|o| o[cy * cw + cx] != 0) {
                None
            } else {
                Some(ym_color::mask::PhotoMap::position(&map, x, y))
            }
        };
        let guide: Option<Vec<f64>> = steps.guided.is_some().then(|| {
            let plane = cw * input.area.h;
            let mut g = Vec::with_capacity(grown.w * grown.h);
            for y in 0..grown.h {
                for x in 0..grown.w {
                    let i = (off.1 as usize + y) * cw + off.0 as usize + x;
                    g.push((input.guide)([
                        f64::from(input.planar[i]),
                        f64::from(input.planar[plane + i]),
                        f64::from(input.planar[2 * plane + i]),
                    ]));
                }
            }
            g
        });
        let codes = row.plane(&PlaneJob {
            photo,
            grown,
            frame: self.frame,
            window,
            position: &position,
            guide: guide.as_deref(),
        });
        let data: Vec<u8> = codes
            .iter()
            .map(|&v| ((u32::from(v) * 255 + 32767) / 65535) as u8)
            .collect();
        Mask::new(window.w as u32, window.h as u32, data).ok()
    }

    /// Makes region `ri`'s picture from the prefix chain's LogC4 over it (`logc4`, interleaved `w * h * 3`), with the
    /// protection plane over it, and keeps it.
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when the library refuses the region.
    pub(crate) fn part(
        &self,
        ri: usize,
        logc4: &[f32],
        protect: Option<&Mask>,
    ) -> Result<Arc<FacePart>, RenderError> {
        let started = std::time::Instant::now();
        let area = self.regions[ri];
        let (w, h) = (area.w, area.h);
        // LogC4 to linear AWG4, per pixel in f64 (the order and rounding are the same on every thread count).
        let lin: Vec<[f32; 3]> = logc4
            .par_chunks_exact(3)
            .map(|p| {
                let l = logc4_decode_rgb([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                [l[0] as f32, l[1] as f32, l[2] as f32]
            })
            .collect();
        let part = LinearAwg4Image::new(w as u32, h as u32, lin).map_err(|e| render_error(&e))?;
        let place = Placement {
            origin_px: (
                (area.x - self.frame.x) as i32,
                (area.y - self.frame.y) as i32,
            ),
            working_px: (self.frame.w as u32, self.frame.h as u32),
        };
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        let out = ym_retouch::retouch_skin_region(
            &part,
            place,
            self.entry.analysis_px,
            &self.faces,
            self.scale,
            &self.entry.retouch,
            protect,
        )
        .map_err(|e| render_error(&e))?;
        let retouch_ms = started.elapsed().as_secs_f64() * 1000.0 - decode_ms;
        let mut data = vec![0.0f32; 3 * w * h];
        data.par_chunks_exact_mut(3)
            .zip(out.data.par_iter().zip(part.data.par_iter()))
            .for_each(|(d, (new, old))| {
                if new == old {
                    // Left as it was: the chain passes its own value through.
                    d.fill(f32::NAN);
                } else {
                    let e =
                        logc4_encode_rgb([f64::from(new[0]), f64::from(new[1]), f64::from(new[2])]);
                    d[0] = e[0] as f32;
                    d[1] = e[1] as f32;
                    d[2] = e[2] as f32;
                }
            });
        tracing::debug!(
            w,
            h,
            decode_ms,
            retouch_ms,
            encode_ms = started.elapsed().as_secs_f64() * 1000.0 - decode_ms - retouch_ms,
            "a face region retouched"
        );
        let made = Arc::new(FacePart { area, data });
        self.done
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[ri] = Some(Arc::clone(&made));
        Ok(made)
    }
}

fn union(a: Area, b: Area) -> Area {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let (x, y) = (a.x.min(b.x), a.y.min(b.y));
    let (x1, y1) = (a.x1().max(b.x1()), a.y1().max(b.y1()));
    Area {
        x,
        y,
        w: (x1 - x) as usize,
        h: (y1 - y) as usize,
    }
}
