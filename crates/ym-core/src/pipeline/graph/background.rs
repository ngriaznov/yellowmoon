//! The photo's blur map, and the Background mask that reads it (`docs/23` section 4.6, `docs/algorithms/06`
//! section 5.4).
//!
//! A Background row's raster `S0` is cut from `P`, the probability that a cell of the blur map is out of focus
//! (`ym_ml::bokeh_net`: the learned model on the lens plane, or the classical detector when the model cannot run).
//! It is **the map the lens uses**: the same lens plane (the raw group at `k_L`, before any lens pass), the same
//! network, made once per photo and shared. Whichever of the lens's bokeh reshaping ([`Job::lens_blur_input`]) and a
//! Background row ([`Job::background_for`]) needs it first makes it; the other reads it. Because it is made from the
//! raw group and not from what the lens did to it, the lens sliders do not move it.
//!
//! The entry ([`BlurEntry`]) holds the map as the lens takes it (cells in the buffer's orientation, `b` and `P`)
//! and `P` turned to the **photo frame** (the photo as the camera oriented it, `docs/23` section 3.6), which is
//! what a raster row's plane covers. It lives in the photo's graph for a viewer and in the request for an export
//! (an export makes its own from the CPU reference group, as it does the lens's planes, so the bytes of an export
//! never depend on what the viewer's device made).
//!
//! Level and Softness do not touch the entry: they cut a new `S0` from the cached `P`
//! ([`ym_color::mask_raster::background_plane`]), and Feather, Grow and Refine edges only re-run the raster steps.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ym_color::mask_raster::BlurMap;
use ym_color::plan::Stop as PlanStop;
use ym_lens::{BlurMapInput, ImageRef, classical_blur_map, lens_plane_scale_for};
use ym_ml::bokeh_net::BokehNet;

use super::lens::{blur_map_input, lens_error};
use super::{Job, lock};
use crate::pipeline::RenderError;
use crate::pipeline::color::uses_background;
use crate::pipeline::geometry::oriented_to_sensor;
use crate::settings::DevelopSettings;
use crate::pipeline::raw::RawParamsKey;

/// The next key of a blur map: every map the core makes has one of its own, so a plane or a grade plan made from an
/// older map (another photo, other raw parameters) is never taken for this one's.
static NEXT_KEY: AtomicU64 = AtomicU64::new(1);

/// The photo's blur map with what it was made from.
pub(crate) struct BlurEntry {
    /// The raw group's key: the map reads its image.
    raw: RawParamsKey,
    /// Whether the learned model made it (the lens uses only that one; the classical detector's map is the
    /// Background mask's alone).
    network: bool,
    /// The map as the lens takes it: the cells in the buffer's orientation.
    input: Arc<BlurMapInput>,
    /// `P` over the photo frame.
    pub(crate) photo: Arc<BlurMap>,
}

impl BlurEntry {
    /// Bytes the entry holds (for the graph's ledger).
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + 4 * (self.input.b.len() + self.input.p.len() + self.photo.p.len())
    }
}

/// `p`, cells of a map in the sensor's orientation (`width x height`), turned to the photo as the camera oriented
/// it (EXIF orientation 1 to 8): the cells of the photo frame, `(width, height, cells)`. A turn of the cell grid, so
/// every cell keeps its value.
pub(crate) fn orient_cells(
    width: usize,
    height: usize,
    p: &[f32],
    exif_orientation: u16,
) -> (usize, usize, Vec<f32>) {
    if width == 0 || height == 0 || p.len() != width * height {
        return (
            width.max(1),
            height.max(1),
            vec![0.0; width.max(1) * height.max(1)],
        );
    }
    let swap = (5..=8).contains(&exif_orientation);
    let (w, h) = if swap {
        (height, width)
    } else {
        (width, height)
    };
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let s = oriented_to_sensor(
                exif_orientation,
                [(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64],
            );
            let sx = ((s[0] * width as f64).floor().max(0.0) as usize).min(width - 1);
            let sy = ((s[1] * height as f64).floor().max(0.0) as usize).min(height - 1);
            out[y * w + x] = p[sy * width + sx];
        }
    }
    (w, h, out)
}

impl Job<'_> {
    /// The photo's blur map for the plan that ends at `stop`, when that plan has a Background row with the eye on:
    /// made when it is not there ([`Job::blur_entry`]). `None` for a plan with none, so a photo without a
    /// Background row never makes a map.
    ///
    /// # Errors
    ///
    /// The errors of [`Job::blur_entry`].
    pub(crate) fn background_for(
        &mut self,
        settings: &DevelopSettings,
        stop: PlanStop,
    ) -> Result<Option<Arc<BlurMap>>, RenderError> {
        if !uses_background(settings, stop) {
            return Ok(None);
        }
        Ok(Some(Arc::clone(&self.blur_entry()?.photo)))
    }

    /// The map the lens takes: the learned model's, from the photo's entry when the Background mask made it, else
    /// made now on `image` (the lens plane) and kept for the mask. `None` when the model fails (the classical
    /// detector then makes the lens's map).
    ///
    /// # Errors
    ///
    /// The errors of [`Job::params`].
    pub(super) fn lens_blur_input(
        &mut self,
        model: &BokehNet,
        image: &ImageRef,
    ) -> Result<Option<Arc<BlurMapInput>>, RenderError> {
        let raw = self.params()?.key();
        if let Some(known) = self.blur_known(&raw) {
            return Ok(known.network.then(|| Arc::clone(&known.input)));
        }
        // One viewer request makes the map; a second that asked together waits and reads it.
        let graph = Arc::clone(&self.g);
        let _gate = (!self.restartable).then(|| lock(&graph.blur_gate));
        if let Some(known) = self.blur_known(&raw) {
            return Ok(known.network.then(|| Arc::clone(&known.input)));
        }
        let t = Instant::now();
        let Some(input) = blur_map_input(model, image).map(Arc::new) else {
            return Ok(None);
        };
        self.runs.blur_maps += 1;
        tracing::debug!(
            ms = t.elapsed().as_secs_f64() * 1000.0,
            "the blur map (the lens)"
        );
        let entry = self.blur_made(raw, Arc::clone(&input), true);
        self.blur_keep(entry);
        Ok(Some(input))
    }

    /// The photo's blur map: from the graph (a viewer) or the request (an export), else made on the lens plane,
    /// by the learned model and by the classical detector when the model is missing or fails.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`], the errors of the raw group, and [`RenderError::Image`] when the classical
    /// detector refuses the plane.
    pub(crate) fn blur_entry(&mut self) -> Result<Arc<BlurEntry>, RenderError> {
        let raw = self.params()?.key();
        if let Some(known) = self.blur_known(&raw) {
            return Ok(known);
        }
        // One viewer request makes the map; a second that asked together (a frame and the node thumbnails) waits and reads it.
        let graph = Arc::clone(&self.g);
        let _gate = (!self.restartable).then(|| lock(&graph.blur_gate));
        if let Some(known) = self.blur_known(&raw) {
            return Ok(known);
        }
        self.check()?;
        let plan = self.plan()?;
        let k = lens_plane_scale_for(plan.out_w, plan.out_h);
        let source = self.plane_source_at(k)?;
        self.check()?;
        let image = ImageRef {
            width: source.width,
            height: source.height,
            x0: 0,
            y0: 0,
            frame_width: source.width,
            frame_height: source.height,
            data: &source.data,
        };
        let t = Instant::now();
        let model = self.opts.bokeh_model.clone();
        let learned = model
            .as_deref()
            .and_then(|m| blur_map_input(m, &image))
            .map(Arc::new);
        self.check()?;
        let (input, network) = if let Some(input) = learned {
            (input, true)
        } else {
            let focus = self.focus_point()?;
            let cancel = self.cancel.clone();
            let made = classical_blur_map(&image, focus, &move || cancel.is_cancelled())
                .map_err(lens_error)?;
            (Arc::new(made), false)
        };
        self.runs.blur_maps += 1;
        tracing::debug!(
            ms = t.elapsed().as_secs_f64() * 1000.0,
            network,
            "the blur map (the Background mask)"
        );
        let entry = self.blur_made(raw, input, network);
        self.blur_keep(Arc::clone(&entry));
        Ok(entry)
    }

    /// The entry this request would use for the raw group `raw`, when it has one.
    fn blur_known(&self, raw: &RawParamsKey) -> Option<Arc<BlurEntry>> {
        let entry = if self.restartable {
            self.blur_job.clone()
        } else {
            lock(&self.g.state).blur.clone()
        };
        entry.filter(|e| e.raw == *raw)
    }

    /// Keeps a new entry where this request looks for it.
    fn blur_keep(&mut self, entry: Arc<BlurEntry>) {
        if self.restartable {
            self.blur_job = Some(entry);
        } else {
            lock(&self.g.state).blur = Some(entry);
        }
    }

    /// The entry for a map `input` of the lens plane: `P` turned to the photo frame, with a key of its own.
    fn blur_made(
        &self,
        raw: RawParamsKey,
        input: Arc<BlurMapInput>,
        network: bool,
    ) -> Arc<BlurEntry> {
        let (width, height, mut p) = orient_cells(
            input.width,
            input.height,
            &input.p,
            self.src.exif_orientation(),
        );
        for v in &mut p {
            *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        }
        Arc::new(BlurEntry {
            raw,
            network,
            input,
            photo: Arc::new(BlurMap {
                key: NEXT_KEY.fetch_add(1, Ordering::Relaxed),
                width,
                height,
                p,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::orient_cells;

    /// A 3 x 2 map whose cells are 0..5 row by row, turned by each EXIF orientation, against the cells written out.
    #[test]
    fn exif_orientations_turn_the_cell_grid() {
        let p: Vec<f32> = (0..6).map(|v| v as f32).collect();
        // (orientation, width, height, cells) of the photo as the camera oriented it.
        let cases: [(u16, usize, usize, [f32; 6]); 8] = [
            (1, 3, 2, [0., 1., 2., 3., 4., 5.]),
            (2, 3, 2, [2., 1., 0., 5., 4., 3.]),
            (3, 3, 2, [5., 4., 3., 2., 1., 0.]),
            (4, 3, 2, [3., 4., 5., 0., 1., 2.]),
            (5, 2, 3, [0., 3., 1., 4., 2., 5.]),
            (6, 2, 3, [3., 0., 4., 1., 5., 2.]),
            (7, 2, 3, [5., 2., 4., 1., 3., 0.]),
            (8, 2, 3, [2., 5., 1., 4., 0., 3.]),
        ];
        for (orientation, w, h, want) in cases {
            let (ow, oh, got) = orient_cells(3, 2, &p, orientation);
            assert_eq!((ow, oh), (w, h), "orientation {orientation}");
            assert_eq!(got, want, "orientation {orientation}");
        }
    }

    #[test]
    fn a_map_that_does_not_fit_its_size_is_all_zero() {
        let (w, h, got) = orient_cells(4, 4, &[1.0; 3], 1);
        assert_eq!((w, h), (4, 4));
        assert!(got.iter().all(|v| *v == 0.0));
    }
}
