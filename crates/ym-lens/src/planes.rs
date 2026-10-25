//! The amount-independent per-photo data of the lens (`docs/21` section 3, "Planes and lists, once per photo").
//!
//! L1 (the blur map and the ball list) and L3 (the tail plane and the source list) put their data here. For
//! now the planes hold only what identifies them, so the API is the final one.

use crate::error::LensError;
use crate::image::ImageRef;
use crate::plan::LensPlan;

/// Why a photo's bokeh was left as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReshapeReason {
    /// The settings did not ask for it, or (until L1 is built) it cannot run.
    NotAsked,
    /// The photo has little out-of-focus background.
    LittleBlur,
}

/// What the bokeh reshaping did for this photo (`RenderReport::lens`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReshapeReport {
    /// True when the bokeh was reshaped.
    pub reshaped: bool,
    /// Why it was not, when it was not.
    pub reason: Option<ReshapeReason>,
}

/// The data every tile of a photo shares: made once per photo, on the CPU, from the lens plane.
#[derive(Clone, Debug)]
pub struct LensPlanes {
    key: String,
    focus: Option<(f64, f64)>,
    report: ReshapeReport,
    lens_plane_size: (usize, usize),
    /// L3's tail plane, source list and kernels.
    flare: crate::l3::FlarePlanes,
    /// L1's blur map and ball list, when the plan needs them ([`LensPlan::needs_blur_map`]).
    bokeh: Option<crate::l1::BokehPlanes>,
}

impl LensPlanes {
    /// Makes the planes of a photo.
    ///
    /// `lens_plane` is the whole frame at scale [`LensPlan::lens_plane_scale`] (the raw group's image there).
    /// `focus` is the camera's focus point as fractions `(x, y)` of the frame in the buffer's orientation, when
    /// the file has one. `cancel` is polled between steps and ends the build with [`LensError::Cancelled`].
    ///
    /// # Errors
    ///
    /// [`LensError::Cancelled`], or [`LensError::Frame`] when the lens plane or the focus point does not fit
    /// the plan.
    pub fn build(
        plan: &LensPlan,
        lens_plane: &ImageRef,
        focus: Option<(f64, f64)>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Self, LensError> {
        Self::build_with_map(plan, lens_plane, focus, None, cancel)
    }

    /// [`LensPlanes::build`] with the blur map of another source (the learned model) in place of the classical
    /// detector's edge measure: see [`BlurMapInput`]. The balls, the subject anchor and the cleaning of the map
    /// are the same. With `None` it is `build`.
    ///
    /// # Errors
    ///
    /// As [`LensPlanes::build`].
    pub fn build_with_map(
        plan: &LensPlan,
        lens_plane: &ImageRef,
        focus: Option<(f64, f64)>,
        map: Option<&crate::l1::BlurMapInput>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Self, LensError> {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        if let Some((x, y)) = focus
            && !((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y))
        {
            return Err(LensError::Frame(format!(
                "the focus point ({x}, {y}) must be fractions of the frame from 0 to 1"
            )));
        }
        let planes_key = plan.planes_key();
        if plan.is_identity() {
            return Ok(LensPlanes {
                key: planes_key,
                focus,
                report: ReshapeReport {
                    reshaped: false,
                    reason: Some(ReshapeReason::NotAsked),
                },
                lens_plane_size: (0, 0),
                flare: crate::l3::FlarePlanes::empty(),
                bokeh: None,
            });
        }
        lens_plane.validate()?;
        let k = plan.lens_plane_scale();
        let f = plan.frame();
        let fits = |plane: usize, full: usize| plane * k + k > full && plane * k < full + k;
        if lens_plane.x0 != 0
            || lens_plane.y0 != 0
            || lens_plane.width != lens_plane.frame_width
            || lens_plane.height != lens_plane.frame_height
            || !fits(lens_plane.frame_width, f.width)
            || !fits(lens_plane.frame_height, f.height)
        {
            return Err(LensError::Frame(format!(
                "the lens plane must be the whole {}x{} frame reduced {k} times, found {}x{} at ({}, {}) of {}x{}",
                f.width,
                f.height,
                lens_plane.width,
                lens_plane.height,
                lens_plane.x0,
                lens_plane.y0,
                lens_plane.frame_width,
                lens_plane.frame_height
            )));
        }
        if cancel() {
            return Err(LensError::Cancelled);
        }
        // L1 detects the blur already in the photo, only when a slider asks to reshape bokeh.
        let bokeh = if plan.needs_blur_map() {
            // The balls L1 redraws carry the colour of their light, not the clip pattern's cast.
            let lit = crate::light::lit_input(plan, lens_plane);
            let plane = lit
                .as_deref()
                .map_or(*lens_plane, |data| ImageRef { data, ..*lens_plane });
            Some(crate::l1::BokehPlanes::build(
                plan, &plane, focus, map, cancel,
            )?)
        } else {
            None
        };
        let report = bokeh.as_ref().map_or(
            ReshapeReport {
                reshaped: false,
                reason: Some(ReshapeReason::NotAsked),
            },
            crate::l1::BokehPlanes::report,
        );
        if cancel() {
            return Err(LensError::Cancelled);
        }
        Ok(LensPlanes {
            key: planes_key,
            focus,
            report,
            lens_plane_size: (lens_plane.width, lens_plane.height),
            flare: crate::l3::FlarePlanes::build(plan, lens_plane, cancel)?,
            bokeh,
        })
    }

    /// What the bokeh reshaping did.
    #[must_use]
    pub fn report(&self) -> ReshapeReport {
        self.report
    }

    /// L1's data: the blur map and the ball list. `None` when the plan the planes were built for did not ask
    /// to reshape bokeh ([`LensPlan::needs_blur_map`]).
    #[must_use]
    pub fn bokeh(&self) -> Option<&crate::l1::BokehPlanes> {
        self.bokeh.as_ref()
    }

    /// True when these planes hold everything `plan` needs: the same [`LensPlan::planes_key`], and the blur map
    /// when the plan asks to reshape bokeh. The key leaves the slider amounts out, so a plan that gains
    /// Anamorphic needs planes built again; this says so.
    #[must_use]
    pub fn serves(&self, plan: &LensPlan) -> bool {
        self.key == plan.planes_key() && (!plan.needs_blur_map() || self.bokeh.is_some())
    }

    /// Bytes the planes hold (for the cache's budget).
    #[must_use]
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.key.len()
            + self.flare.bytes()
            + self.bokeh.as_ref().map_or(0, crate::l1::BokehPlanes::bytes)
    }

    /// L3's data: the tail plane, the source list and the kernels.
    pub(crate) fn flare(&self) -> &crate::l3::FlarePlanes {
        &self.flare
    }

    /// The [`LensPlan::planes_key`] these planes were built for.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The camera's focus point, as fractions of the frame.
    #[must_use]
    pub fn focus(&self) -> Option<(f64, f64)> {
        self.focus
    }

    /// The size of the lens plane the planes were built from.
    #[must_use]
    pub fn lens_plane_size(&self) -> (usize, usize) {
        self.lens_plane_size
    }
}
