//! One photo's lens: the plan (`docs/21` section 5).
//!
//! A [`LensPlan`] binds a lens, three amounts, a frame and a camera: the lens data projected on the camera's
//! channels, the derived constants (`q`, `e'`, `alpha`, ...) and the L4 map. It is cheap to clone around by
//! reference and holds no pixels.

use std::sync::Arc;

use crate::amounts::{LensAmounts, LensKind};
use crate::consts::LENS_PLANE_LONG_EDGE;
use crate::data::{AttachmentData, LensData, LensLibrary};
use crate::error::LensError;
use crate::field::FieldTables;
use crate::spectral::CameraResponse;
use crate::warp::{BendMap, LensWarp};

/// The anamorphic axis in the buffer: the photo's horizontal as the person sees it (`docs/21` section 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// The axis is the buffer's x axis.
    X,
    /// The axis is the buffer's y axis.
    Y,
}

/// The frame the lens is fitted to: the full-resolution frame after R9's auto crop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameInfo {
    /// Width in pixels at full resolution.
    pub width: usize,
    /// Height in pixels at full resolution.
    pub height: usize,
    /// The anamorphic axis in the buffer.
    pub axis: Axis,
    /// The photo's f-number from EXIF; the starburst assumes f/4 without one.
    pub f_number: Option<f64>,
    /// The clip level `g0` of each channel in the lens group's scale.
    pub clip: [f32; 3],
}

/// One photo's lens: lens data on the camera's channels, amounts, frame and axis.
#[derive(Clone, Debug)]
pub struct LensPlan {
    amounts: LensAmounts,
    frame: FrameInfo,
    lens: Arc<LensData>,
    attachment: Arc<AttachmentData>,
    camera: CameraResponse,
    /// `q`: the bokeh's height over its width, 1 to `Q_max`.
    q: f64,
    /// `e = edges / 100`.
    e: f64,
    /// `F = flares / 100`.
    flares: f64,
    /// `e' = e + (1 - e) kappa (q - 1)`: the edge amount the field optics run at.
    e_field: f64,
    /// `alpha = 1 + kappa (q - 1)`.
    alpha: f64,
    /// `a_s (q - 1)`: the side softness's amount.
    side: f64,
    /// `1 + h (q - 1)`: the factor on the axis part of the channels' centroid offsets.
    fringe: f64,
    /// The bend and distortion of L4.
    bend: BendMap,
    /// The spread functions on the camera's channels, when the field optics can act.
    field: Option<FieldTables>,
}

impl LensPlan {
    /// Builds the plan of one photo.
    ///
    /// # Errors
    ///
    /// [`LensError::MissingLens`] when the library lacks the lens, [`LensError::Amount`] for an amount that is
    /// not a number from 0 to 100, [`LensError::Frame`] for an empty frame or a bad f-number.
    pub fn new(
        lib: &LensLibrary,
        amounts: &LensAmounts,
        frame: FrameInfo,
        camera: &CameraResponse,
    ) -> Result<Self, LensError> {
        amounts.validate()?;
        if frame.width == 0 || frame.height == 0 {
            return Err(LensError::Frame("the frame is empty".into()));
        }
        if let Some(f) = frame.f_number
            && !(f.is_finite() && f > 0.0)
        {
            return Err(LensError::Frame(format!(
                "the f-number must be a positive number, found {f}"
            )));
        }
        if !frame.clip.iter().all(|c| c.is_finite()) {
            return Err(LensError::Frame("the clip levels must be finite".into()));
        }
        let lens = lib
            .get_shared(amounts.kind)
            .ok_or_else(|| LensError::MissingLens(amounts.kind.to_string()))?;
        let attachment = lib.attachment_shared();

        let q = 1.0 + (attachment.q_max - 1.0) * amounts.anamorphic / 100.0;
        let e = amounts.edges / 100.0;
        let flares = amounts.flares / 100.0;
        let extra = q - 1.0;
        let e_field = e + (1.0 - e) * attachment.field.kappa * extra;
        let bend = BendMap {
            mu: attachment.bend.mu0 * extra,
            beta: attachment.bend.beta0 * extra,
            d1: e * lens.field.distortion.k1,
            d2: e * lens.field.distortion.k2,
        };
        let field = if !amounts.is_identity() && e_field > 0.0 {
            Some(FieldTables::new(&lens, camera))
        } else {
            None
        };
        Ok(LensPlan {
            amounts: *amounts,
            frame,
            alpha: 1.0 + attachment.field.kappa * extra,
            side: attachment.field.side_softness * extra,
            fringe: 1.0 + attachment.field.fringe * extra,
            lens,
            attachment,
            camera: camera.clone(),
            q,
            e,
            flares,
            e_field,
            bend,
            field,
        })
    }

    /// True when all three amounts are 0: no lens pass runs and the render is what it is without a lens.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.amounts.is_identity()
    }

    /// The amounts the plan was built for.
    #[must_use]
    pub fn amounts(&self) -> &LensAmounts {
        &self.amounts
    }

    /// The lens.
    #[must_use]
    pub fn kind(&self) -> LensKind {
        self.amounts.kind
    }

    /// The full-resolution frame.
    #[must_use]
    pub fn frame(&self) -> &FrameInfo {
        &self.frame
    }

    /// The lens data.
    #[must_use]
    pub fn lens(&self) -> &LensData {
        &self.lens
    }

    /// The anamorphic attachment's data.
    #[must_use]
    pub fn attachment(&self) -> &AttachmentData {
        &self.attachment
    }

    /// The camera projection the plan was built with.
    #[must_use]
    pub fn camera(&self) -> &CameraResponse {
        &self.camera
    }

    /// `q = 1 + (Q_max - 1) anamorphic / 100`: the bokeh's height over its width.
    #[must_use]
    pub fn q(&self) -> f64 {
        self.q
    }

    /// The half-length of the vertical stretch (L1's first gather) per unit of the pixel's blur radius `b`:
    /// `sqrt(3 (q^2 - 1)) / 2` for a box line, times [`crate::consts::PROFILE_SCALE`] for the biweight line the
    /// gather uses, which has the same spread: a texture blurred by a disc of radius `b` ends with the spread of
    /// a texture blurred by an ellipse `q` times taller than wide. A line of half-length `(q - 1) b` would give
    /// a disc the extent of that ellipse but only about half its extra vertical spread (`docs/21` section 4, L1,
    /// step 4). It is 0 at `q = 1` and 2.29 at `q = 2`.
    #[must_use]
    pub fn stretch_per_blur(&self) -> f64 {
        (3.0 * (self.q * self.q - 1.0)).max(0.0).sqrt() / 2.0 * crate::consts::PROFILE_SCALE
    }

    /// `e = edges / 100`.
    #[must_use]
    pub fn edges(&self) -> f64 {
        self.e
    }

    /// `F = flares / 100`.
    #[must_use]
    pub fn flares(&self) -> f64 {
        self.flares
    }

    /// The edge amount `e' = e + (1 - e) kappa (q - 1)` the field optics (L2) run at.
    ///
    /// It is Edges plus the attachment's own share, so that at Edges 0 the oval falloff, the side softness and
    /// the horizontal fringe act alone (`docs/21` section 4, L2). It equals `e` at Anamorphic 0.
    #[must_use]
    pub fn field_amount(&self) -> f64 {
        self.e_field
    }

    /// `alpha = 1 + kappa (q - 1)`: the stretch of the field radius along the axis.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// `a_s (q - 1)`: the amount of the side softness `1 + a_s (q - 1) x^2`.
    #[must_use]
    pub fn side_softness(&self) -> f64 {
        self.side
    }

    /// `1 + h (q - 1)`: the factor on the axis part of the channels' centroid offsets.
    #[must_use]
    pub fn fringe_factor(&self) -> f64 {
        self.fringe
    }

    /// The fraction `a` of light the flare terms take from the picture, `a` of `docs/21` section 4, L3: the
    /// sum of the fractions in use (the streak's with its Anamorphic factor `(q - 1) / (Q_max - 1)`) times `F`,
    /// the fractions being the lens's after the floors of [`FlareStrengths`](crate::FlareStrengths).
    #[must_use]
    pub fn light_fraction(&self) -> f64 {
        let a = &self.attachment;
        let st = self.flare_strengths();
        let anamorphic = (self.q - 1.0) / (a.q_max - 1.0);
        let lens_part = st.lens_fraction();
        let attachment_part = anamorphic * st.attachment_fraction();
        self.flares * (lens_part + attachment_part)
    }

    /// True when the picture's blur map is needed: Anamorphic above 0, or Edges above 0 on a lens whose swirl
    /// or cat's-eye is not zero (the bokeh reshaping, L1).
    #[must_use]
    pub fn needs_blur_map(&self) -> bool {
        self.amounts.anamorphic > 0.0
            || (self.amounts.edges > 0.0
                && (self.lens.swirl_max() > 0.0 || self.lens.cats_eye_max() > 0.0))
    }

    /// `k_L`: the scale of the lens plane, the largest power of two, at least 2, for which the frame's long
    /// edge over `k_L` is still at least 1024 px (a 6000 px frame gets 4, an 8640 px frame 8, a frame under
    /// 2048 px 2). A frame under 1024 px long, which only tiny test frames are, gets 1.
    #[must_use]
    pub fn lens_plane_scale(&self) -> usize {
        let long = self.frame.width.max(self.frame.height);
        if long < LENS_PLANE_LONG_EDGE {
            return 1;
        }
        let mut k = 2usize;
        while long >= LENS_PLANE_LONG_EDGE.saturating_mul(k.saturating_mul(2)) {
            k *= 2;
        }
        k
    }

    /// L4 for a canvas of `canvas_width x canvas_height` pixels (the anamorphic axis is the canvas's x).
    ///
    /// Build it at the full-resolution canvas and use [`LensWarp::rescaled`] for reduced views, so that every
    /// scale shares one zoom.
    #[must_use]
    pub fn warp(&self, canvas_width: f64, canvas_height: f64) -> LensWarp {
        LensWarp::new(self.bend, canvas_width, canvas_height)
    }

    /// Everything the per-pixel passes depend on, as text with exact numbers.
    ///
    /// All-zero amounts are no lens whatever the kind, so they share one key and a change of lens at 0
    /// invalidates nothing.
    #[must_use]
    pub fn key(&self) -> String {
        if self.is_identity() {
            return "lens:none".to_owned();
        }
        let f = &self.frame;
        format!(
            "lens:{}:v{}:att{}:{:?}:{:?}:{:?}:{}:{}x{}:{:?}:fn{:?}:clip{:?}",
            self.amounts.kind,
            self.lens.info.version,
            self.attachment.version,
            self.amounts.flares,
            self.amounts.edges,
            self.amounts.anamorphic,
            self.camera.key(),
            f.width,
            f.height,
            f.axis,
            f.f_number,
            f.clip,
        )
    }

    /// What `Z = L2(L1(X))` ([`apply_field`](crate::apply_field)) depends on besides its input and the planes: the
    /// lens, its data versions, Edges and Anamorphic, the camera, the frame size and the axis. Not Flares and
    /// not the f-number (L3 alone reads them), so a Flares move keeps `Z` and runs L3 only.
    ///
    /// Like [`LensPlan::key`], all-zero amounts share one key whatever the lens.
    #[must_use]
    pub fn field_key(&self) -> String {
        if self.is_identity() {
            return "lens-field:none".to_owned();
        }
        if self.field_is_identity() {
            return "lens-field:same".to_owned();
        }
        let f = &self.frame;
        format!(
            "lens-field:{}:v{}:att{}:{:?}:{:?}:{}:{}x{}:{:?}",
            self.amounts.kind,
            self.lens.info.version,
            self.attachment.version,
            self.amounts.edges,
            self.amounts.anamorphic,
            self.camera.key(),
            f.width,
            f.height,
            f.axis,
        )
    }

    /// What the flare terms of L3 ([`flare_terms`](crate::flare_terms)) depend on besides their input and the
    /// planes: the lens, its data versions, Flares, Anamorphic, the camera, the frame, the axis, the clip
    /// levels and the f-number. Not Edges, so an Edges move keeps them.
    ///
    /// Like [`LensPlan::key`], all-zero amounts share one key whatever the lens; so do all plans with
    /// Flares 0, which add no light.
    #[must_use]
    pub fn flares_key(&self) -> String {
        if self.is_identity() || self.amounts.flares == 0.0 {
            return "lens-flares:none".to_owned();
        }
        let f = &self.frame;
        format!(
            "lens-flares:{}:v{}:att{}:{:?}:{:?}:{}:{}x{}:{:?}:fn{:?}:clip{:?}",
            self.amounts.kind,
            self.lens.info.version,
            self.attachment.version,
            self.amounts.flares,
            self.amounts.anamorphic,
            self.camera.key(),
            f.width,
            f.height,
            f.axis,
            f.f_number,
            f.clip,
        )
    }

    /// True when `Z = L2(L1(X))` is `X` itself for sure: Edges and Anamorphic are both 0, so neither the
    /// reshaping nor the field optics act (Flares alone, or no lens at all).
    #[must_use]
    pub fn field_is_identity(&self) -> bool {
        self.amounts.edges == 0.0 && self.amounts.anamorphic == 0.0
    }

    /// What [`LensPlanes`](crate::LensPlanes) depend on: the lens, its data version, the camera, the frame
    /// size, the axis and the clip levels (the restored light of L3 reads them). Not the amounts: moving a
    /// slider keeps the planes.
    #[must_use]
    pub fn planes_key(&self) -> String {
        if self.is_identity() {
            return "lens-planes:none".to_owned();
        }
        let f = &self.frame;
        format!(
            "lens-planes:{}:v{}:att{}:{}:{}x{}:{:?}:clip{:?}",
            self.amounts.kind,
            self.lens.info.version,
            self.attachment.version,
            self.camera.key(),
            f.width,
            f.height,
            f.axis,
            f.clip,
        )
    }

    /// The projected spread functions, when the field optics can act.
    pub(crate) fn field_tables(&self) -> Option<&FieldTables> {
        self.field.as_ref()
    }

    /// The field optics (L2) for the device at a frame of `width x height` pixels (the frame at the scale
    /// of the caller): the numbers its kernel takes, or `None` when L2 does nothing for this plan.
    #[must_use]
    pub fn field_pass(&self, width: usize, height: usize) -> Option<crate::device::FieldPass> {
        crate::field::Geometry::new(self, width, height, crate::field::Options::default())
            .map(|g| g.pass())
    }

    /// The lens's spread functions on the camera's channels as the device reads them, when the field
    /// optics can act for this plan (Edges or Anamorphic above 0). They depend on the lens and the
    /// camera only ([`FieldTablesData::key`](crate::device::FieldTablesData)), so one upload serves every
    /// amount.
    #[must_use]
    pub fn field_tables_data(&self) -> Option<crate::device::FieldTablesData> {
        let tables = self.field.as_ref()?;
        Some(crate::device::FieldTablesData {
            key: format!(
                "field:{}:v{}:{}",
                self.amounts.kind,
                self.lens.info.version,
                self.camera.key()
            ),
            data: tables.flatten(),
        })
    }
}
