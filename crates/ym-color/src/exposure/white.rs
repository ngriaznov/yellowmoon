//! The white lift: a per-photo gain on the tones above +1 stop over middle grey, so the brightest
//! tones of a photo can reach a real white (`docs/02-colour-pipeline.md`, C12 step 1b; decision of
//! 2026-10-14, task D-02).
//!
//! # Why
//!
//! ACES 2 SDR is a cinema rendering with a flat shoulder: 1.4 scene stops above the placed middle grey
//! lift the display by about 0.4 stop, so a photo whose brightest tones sit 2 to 3 stops over grey ends
//! at light grey (the 99.5th percentile of the display luma code is 147 to 218 on the 12 local photos
//! at +0.85 EV against 161 to 247 in their camera JPEGs). The cameras' own tone curves put such tones
//! near white. The lift is the missing white, sized from the photo's own highlights; mid grey and
//! everything below +1 stop stay where they are.
//!
//! # Alongside the highlight range
//!
//! Since pipeline 0.4.0 the chain also has C7's highlight range ([`crate::tone::HighlightRange`]): on the
//! scene side, before LogC4, it takes the camera's top stop (from one stop under the Sony clip up) onto
//! the ALEXA's, so the clipped highlights of a photo land at white. The two do different jobs and both
//! run: the range moves where the **clip** lands and leaves everything under its knee (about +2.6 stops
//! over grey at the camera's exposure) untouched; the lift is sized from the photo's **own brightest
//! tones**, wherever they are, and acts from +1 stop. The order is C7 (the range), C8 to C11, then C12
//! step 1b (the lift), step 1a (the placement) and ACES 2. The estimator sees the photo through the
//! default C7, so the tones the range already took to white need no lift and get less of one
//! ([`estimate_white`]).
//!
//! # The stage ([`WhiteLift`])
//!
//! Scene-linear, on the C11 output decoded to linear AWG4, **before the display placement and ACES 2, on
//! the ACES 2 path only** (an imported ARRI display LUT and a LogC3-to-Rec.709 look keep ARRI's own
//! highlights):
//!
//! ```text
//! Y    = 0.2126 R + 0.7152 G + 0.0722 B          the luma of the AWG4 triple ([`drive_luma`])
//! s    = log2(Y / 0.18)                         stops over middle grey
//! t    = clamp((s - 1) / span, 0, 1)            span = max(anchor - 1, 0.25)
//! gain = 2^(A * t^2 (3 - 2t))                   smoothstep
//! rgb' = gain * rgb                             all three channels, so the ratios are kept
//! ```
//!
//! `A` is the amount in stops ([`WhiteLift::amount_ev`]) and `anchor` the stops over grey at which the
//! lift is complete ([`WhiteLift::anchor_ev`]): the two scalars of a photo. Below +1 stop the gain is
//! exactly 1; between +1 stop and the anchor it grows smoothly; above the anchor every tone gets the
//! same `2^A`. The gain never falls when a channel rises, so **every output channel is non-decreasing in
//! each input channel** (a product of a channel and a non-negative non-decreasing function of the luma),
//! and the picture's colours keep their ratios. `A = 0` is the exact identity.
//!
//! The luma has the Rec.709 weights, which are all positive, and not the AWG4 `Y` row, whose blue weight
//! is slightly negative (`-0.036`): with the `Y` row a saturated blue's own output would fall when its
//! blue channel rises. A neutral value is its own luma either way (both rows sum to 1), so what the
//! estimator measures on the neutral axis is what the stage does to a neutral. It is the same luma C7's
//! highlight range reads ([`crate::tone::luma`]).
//!
//! # The estimator ([`estimate_white`])
//!
//! A pure CPU function of the 1/8 preview, like [`super::auto_exposure`]: same input, same two
//! scalars, whatever the thread count (it uses none). On the preview as the render sees it (balanced
//! with the initial gains, vignetting gain applied, through the fused matrix at the photo's exposure and
//! C7 at its default amounts, the highlight range at 1: a photo whose highlights the range takes to white
//! has its 99.5th percentile there already, so it gets less lift, or none):
//!
//! 1. `L` = stops over grey of the luma ([`drive_luma`]) of every pixel. `P` is its 99.5th percentile (the brightest
//!    half percent of the photo) and `B` the 99.5th percentile of a box-blurred copy of `L` (the
//!    **anchor**: the level of the photo's highlight areas, a few small speculars do not move it).
//! 2. Both are carried through the **reference look** (the Signature look, applied to the neutral axis;
//!    no look is the identity), because the lift acts on the C11 output and the look already lifts the
//!    shoulder (`docs/02`, "Re-tune with the Signature look").
//! 3. `d` is the display luma code (0 to 255, ACES 2 with the placement, sRGB) of the neutral value
//!    `P` would have, and `high_key = clamp((B - 1) / 1.5, 0, 1)` says how high-key the photo is (`B`
//!    before the look: the photo's own highlights, so the look's shoulder is not counted twice).
//!    The target code is `d + high_key * max(TARGET_CODE - d, 0)`.
//! 4. `A` is the smallest amount that lifts `P` to the target code (a bisection on the neutral axis),
//!    held to `0 ..= MAX_AUTO_AMOUNT_EV`. A photo whose highlights are not above +1 stop gets `A = 0`.
//!
//! The estimator never sees the user's settings: it runs once per open, with the photo's automatic
//! exposure, like the other automatic estimates, so a slider move never moves it. The user's offset
//! ([`WhiteEstimate::lift`]) is added to `A` afterwards.
//!
//! # Decisions where the spec is silent
//!
//! * The anchor is blurred with a box of radius `0.007 * width` preview pixels (8 for a 1188 pixel
//!   preview, 64 sensor pixels at the full frame), edges average the part of the window inside the
//!   frame. The research prototype used a self-guided filter of the same extent; the box is cheaper and
//!   the anchor differs by less than 0.1 stop.
//! * Clipped blocks count like any other pixel: a clipped sky is a white to be lifted.
//! * A pixel with a non-finite channel counts as the darkest value.
//! * The reference look is applied to the neutral axis only; the picture's colours do not enter the
//!   sizing (the lift is meant for white, and neutral is what white is).
//! * C7 is the one the caller passes (the default amounts, not the user's: a move of RAW › Range does
//!   not re-size the lift). The neutral axis of steps 3 and 4 has no C7: the percentiles are already
//!   past it.

use crate::consts::Vec3;
use crate::consts::linalg::Mat3;
use crate::display::{ChainParams, ColorChain, OutputTarget};
use crate::exposure::{ExposureError, LUMINANCE_FLOOR, PreviewRgb, TARGET_GREY};
use crate::logc::LogC4;
use crate::look::LookTransform;
use crate::tone::{self, ToneParams};

/// Tones up to this many stops over middle grey are untouched by the lift.
pub const KNEE_EV: f64 = 1.0;
/// The smallest span of the ramp, in stops: the anchor is never taken closer to the knee than this.
pub const MIN_SPAN_EV: f64 = 0.25;
/// The largest amount the estimator returns, in stops.
pub const MAX_AUTO_AMOUNT_EV: f64 = 2.0;
/// The largest amount of a lift after the user's offset, in stops.
pub const MAX_AMOUNT_EV: f64 = 3.0;
/// The range of the user's offset (`white_offset_ev`), in stops, either way.
pub const OFFSET_RANGE_EV: f64 = 3.0;
/// The display luma code (8-bit sRGB) the brightest tones of a fully high-key photo are taken to.
pub const TARGET_CODE: f64 = 235.0;
/// The anchor, in stops over grey, below which a photo is not high-key (no lift).
pub const HIGH_KEY_FROM_EV: f64 = 1.0;
/// The anchor, in stops over grey, from which a photo is fully high-key (the full target).
pub const HIGH_KEY_TO_EV: f64 = 2.5;
/// The percentile of the luminance (and of its blurred copy) that sizes and anchors the lift.
pub const PERCENTILE: f64 = 0.995;
/// The radius of the anchor's box blur, as a fraction of the preview width.
pub const BLUR_RADIUS_FRACTION: f64 = 0.007;

/// Why a [`WhiteLift`] was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WhiteError {
    /// The amount must be a finite number from 0 to [`MAX_AMOUNT_EV`].
    #[error("the white lift amount must be a number from 0 to {max} stops, got {0}", max = MAX_AMOUNT_EV)]
    Amount(f64),
    /// The anchor must be a finite number of stops.
    #[error("the white lift anchor must be a finite number of stops, got {0}")]
    Anchor(f64),
    /// The reference rendering of the estimator could not be built (not possible with the built-in
    /// identity matrix; kept as an error instead of a panic).
    #[error("the white lift's reference rendering failed: {0}")]
    Reference(String),
}

/// The white lift of a photo: the amount `A` and the anchor, the two scalars of the stage.
///
/// Immutable; [`WhiteLift::NONE`] (amount 0) is the exact identity, and every identity lift compares
/// equal to it (the anchor of a zero amount is dropped).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WhiteLift {
    amount_ev: f64,
    anchor_ev: f64,
}

impl WhiteLift {
    /// No lift: the exact identity.
    pub const NONE: Self = Self {
        amount_ev: 0.0,
        anchor_ev: 0.0,
    };

    /// A lift of `amount_ev` stops, complete at `anchor_ev` stops over middle grey. An amount of 0
    /// gives [`WhiteLift::NONE`].
    ///
    /// # Errors
    ///
    /// [`WhiteError::Amount`] for an amount that is not a number from 0 to [`MAX_AMOUNT_EV`];
    /// [`WhiteError::Anchor`] for an anchor that is not finite.
    pub fn new(amount_ev: f64, anchor_ev: f64) -> Result<Self, WhiteError> {
        if !amount_ev.is_finite() || !(0.0..=MAX_AMOUNT_EV).contains(&amount_ev) {
            return Err(WhiteError::Amount(amount_ev));
        }
        if !anchor_ev.is_finite() {
            return Err(WhiteError::Anchor(anchor_ev));
        }
        if amount_ev == 0.0 {
            return Ok(Self::NONE);
        }
        Ok(Self {
            amount_ev,
            anchor_ev,
        })
    }

    /// Whether the lift changes nothing.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.amount_ev == 0.0
    }

    /// The amount `A` in stops.
    #[must_use]
    pub fn amount_ev(&self) -> f64 {
        self.amount_ev
    }

    /// The anchor: the stops over middle grey at which the lift is complete.
    #[must_use]
    pub fn anchor_ev(&self) -> f64 {
        self.anchor_ev
    }

    /// The span of the ramp in stops, `max(anchor - KNEE_EV, MIN_SPAN_EV)`.
    #[must_use]
    pub fn span_ev(&self) -> f64 {
        (self.anchor_ev - KNEE_EV).max(MIN_SPAN_EV)
    }

    /// The scene-linear gain of a pixel of luma `y` ([`drive_luma`]): 1 for a luma that is not above 0,
    /// and for anything up to +1 stop over middle grey, then `2^(A * smoothstep)`.
    #[must_use]
    pub fn gain_for_luminance(&self, y: f64) -> f64 {
        if self.amount_ev == 0.0 || y.is_nan() || y <= 0.0 {
            return 1.0;
        }
        let stops = (y / TARGET_GREY).log2();
        let t = ((stops - KNEE_EV) / self.span_ev()).clamp(0.0, 1.0);
        (self.amount_ev * t * t * (3.0 - 2.0 * t)).exp2()
    }

    /// The gain of a linear AWG4 pixel (from its luma, [`drive_luma`]).
    #[must_use]
    pub fn gain(&self, awg4: Vec3) -> f64 {
        if self.amount_ev == 0.0 {
            return 1.0;
        }
        self.gain_for_luminance(drive_luma(awg4))
    }

    /// The lifted pixel: all three channels times [`WhiteLift::gain`]. The identity returns `awg4`
    /// bit for bit.
    #[must_use]
    pub fn apply(&self, awg4: Vec3) -> Vec3 {
        if self.amount_ev == 0.0 {
            return awg4;
        }
        let g = self.gain(awg4);
        [awg4[0] * g, awg4[1] * g, awg4[2] * g]
    }
}

/// The luma that drives the lift: the Rec.709 weights on the linear AWG4 triple (all positive; a
/// neutral value is its own luma). The same luma as C7's highlight range ([`tone::luma`]).
#[must_use]
pub fn drive_luma(awg4: Vec3) -> f64 {
    tone::luma(awg4)
}

/// What [`estimate_white`] found for one photo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhiteEstimate {
    /// The automatic amount `A` in stops, 0 to [`MAX_AUTO_AMOUNT_EV`].
    pub amount_ev: f64,
    /// The anchor in stops over middle grey (after the reference look): where the lift is complete.
    pub anchor_ev: f64,
    /// How high-key the photo is, 0 to 1: `clamp((B - 1) / 1.5, 0, 1)` with `B` the photo's own anchor in
    /// stops over middle grey, before the reference look.
    pub high_key: f64,
    /// The photo's 99.5th percentile of luminance in stops over middle grey, after the reference look.
    pub highlight_ev: f64,
    /// The display luma code (0 to 255) of a neutral tone at `highlight_ev` without the lift.
    pub code_before: f64,
    /// The code the lift aims at: `code_before + high_key * max(TARGET_CODE - code_before, 0)`.
    pub target_code: f64,
    /// The code of the same tone with the lift (equal to `target_code` unless the amount was held
    /// at its limit).
    pub code_after: f64,
    /// Pixels the percentiles were taken over.
    pub pixels: usize,
}

impl WhiteEstimate {
    /// No lift, for a photo with nothing to measure.
    #[must_use]
    pub fn none() -> Self {
        Self {
            amount_ev: 0.0,
            anchor_ev: HIGH_KEY_FROM_EV,
            high_key: 0.0,
            highlight_ev: f64::NEG_INFINITY,
            code_before: 0.0,
            target_code: 0.0,
            code_after: 0.0,
            pixels: 0,
        }
    }

    /// The lift to render with: the automatic amount plus the user's offset in stops (negative
    /// brings it down, an offset that takes the amount below 0 gives no lift), held to
    /// [`MAX_AMOUNT_EV`]. A non-finite offset counts as 0.
    #[must_use]
    pub fn lift(&self, offset_ev: f64) -> WhiteLift {
        let offset = if offset_ev.is_finite() {
            offset_ev
        } else {
            0.0
        };
        let amount = (self.amount_ev + offset).clamp(0.0, MAX_AMOUNT_EV);
        WhiteLift::new(amount, self.anchor_ev).unwrap_or(WhiteLift::NONE)
    }
}

/// The neutral axis of the reference rendering: how the Signature look moves a grey tone, and which
/// display code a grey tone has after the look (ACES 2 with the placement, sRGB).
struct Reference<'a> {
    look: Option<&'a LookTransform>,
    display: ColorChain,
}

impl<'a> Reference<'a> {
    fn new(look: Option<&'a LookTransform>) -> Result<Self, WhiteError> {
        let identity: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // C7 off (no highlight range, no cap, no toe): the values fed here are already past C7.
        let mut params = ChainParams::new(identity, ToneParams::IDENTITY);
        params.output = OutputTarget::Srgb;
        params.dither = false;
        let display = ColorChain::new(&params).map_err(|e| WhiteError::Reference(e.to_string()))?;
        // A look that carries its own display rendering has no neutral axis in LogC4: no look.
        let look = look.filter(|l| !l.includes_display_rendering());
        Ok(Self { look, display })
    }

    /// The luminance of the neutral tone `y` (linear AWG4 `[y, y, y]`) after the look.
    fn after_look(&self, y: f64) -> f64 {
        let Some(look) = self.look else { return y };
        if y <= 0.0 || !y.is_finite() {
            return y;
        }
        let code = LogC4([crate::logc::logc4_encode(y); 3]);
        match look.apply_logc4(code, 100.0) {
            Ok(out) => {
                let lum = drive_luma(out.to_linear().0);
                if lum.is_finite() { lum } else { y }
            }
            Err(_) => y,
        }
    }

    /// The display luma code (0 to 255) of the neutral tone `y` (after the look, before the lift).
    fn code(&self, y: f64) -> f64 {
        let y = y.max(0.0);
        let e = self.display.apply_pixel([y, y, y]);
        255.0 * (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2])
    }
}

/// `a + (b - a) * f`, the form that returns `a` exactly for equal neighbours.
fn lerp_order(a: f64, b: f64, f: f64) -> f64 {
    let v = if f >= 0.5 {
        b - (b - a) * (1.0 - f)
    } else {
        a + (b - a) * f
    };
    v.clamp(a.min(b), a.max(b))
}

/// The linearly interpolated quantile `q` of `values` (reordered). NaN for no values.
fn quantile_of(values: &mut [f32], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    let (_, at, above) = values.select_nth_unstable_by(lo, f32::total_cmp);
    let a = f64::from(*at);
    let b = if hi == lo {
        a
    } else {
        // The next order statistic is the smallest of those above.
        above
            .iter()
            .copied()
            .min_by(f32::total_cmp)
            .map_or(a, f64::from)
    };
    lerp_order(a, b, frac)
}

/// A mean over the box `[x - r, x + r] x [y - r, y + r]` clipped to the frame, by summed rows and
/// columns (f64, in a fixed order, so the result is the same on every run).
fn box_blur(values: &[f32], width: usize, height: usize, r: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0f64; width * height];
    let mut prefix = vec![0.0f64; width + 1];
    for y in 0..height {
        let row = &values[y * width..(y + 1) * width];
        for (x, &v) in row.iter().enumerate() {
            prefix[x + 1] = prefix[x] + f64::from(v);
        }
        for x in 0..width {
            let (lo, hi) = (x.saturating_sub(r), (x + r + 1).min(width));
            horizontal[y * width + x] = (prefix[hi] - prefix[lo]) / (hi - lo) as f64;
        }
    }
    let mut out = vec![0.0f32; width * height];
    let mut prefix = vec![0.0f64; height + 1];
    for x in 0..width {
        for y in 0..height {
            prefix[y + 1] = prefix[y] + horizontal[y * width + x];
        }
        for y in 0..height {
            let (lo, hi) = (y.saturating_sub(r), (y + r + 1).min(height));
            out[y * width + x] = ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32;
        }
    }
    out
}

/// The smoothstep `t^2 (3 - 2t)` of `t` held to 0 to 1.
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Sizes the white lift of a photo from its 1/8 preview (see the module documentation).
///
/// `matrix` is the fused C1 to C6 matrix **at the exposure the photo renders with** (camera RGB
/// balanced with the initial gains to linear AWG4, `2^EV` included: `FusedChain::matrix` of the
/// all-Auto chain), `preview` the preview as the render sees it (`Preview::metered`), `tone` the C7
/// parameters of the default render (the highlight range at its default amount with this chain's clip
/// level, `ClipLevel::luma`: the lift is sized for what is left after the range) and `reference` the look
/// the amount is sized for (the Signature look; `None` sizes it for the bare ACES 2 picture).
///
/// A matrix with a non-finite entry, a preview with no pixel and a reference rendering that cannot be
/// built give [`WhiteEstimate::none`] / an error as documented; every other input gives a finite
/// result.
///
/// # Errors
///
/// [`ExposureError::BufferSize`] and [`ExposureError::MaskSize`] for buffers of the wrong length.
pub fn estimate_white(
    preview: &PreviewRgb<'_>,
    matrix: &Mat3,
    tone: &ToneParams,
    reference: Option<&LookTransform>,
) -> Result<WhiteEstimate, ExposureError> {
    preview.validate()?;
    let (width, height) = (preview.width, preview.height);
    let pixels = width * height;
    if pixels == 0 || matrix.iter().flatten().any(|v| !v.is_finite()) {
        return Ok(WhiteEstimate::none());
    }
    let Ok(reference) = Reference::new(reference) else {
        return Ok(WhiteEstimate::none());
    };

    // 1. Stops over middle grey of every pixel, through the fused matrix and C7 (the highlight range).
    let floor_stops = (LUMINANCE_FLOOR / TARGET_GREY).log2();
    let tone = tone.prepare();
    let mut stops = vec![0.0f32; pixels];
    for (i, out) in stops.iter_mut().enumerate() {
        let p = &preview.rgb[3 * i..3 * i + 3];
        let rgb = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
        let value = if rgb.iter().all(|v| v.is_finite()) {
            let awg4 = tone.apply_rgb(crate::consts::linalg::mul_vec(matrix, rgb));
            let y = drive_luma(awg4);
            if y.is_finite() {
                (y.max(LUMINANCE_FLOOR) / TARGET_GREY).log2()
            } else {
                floor_stops
            }
        } else {
            floor_stops
        };
        *out = value as f32;
    }
    let radius = ((BLUR_RADIUS_FRACTION * width as f64).round() as usize).max(1);
    let mut base = box_blur(&stops, width, height, radius);
    let mut sorted = stops;
    let highlight_pre = quantile_of(&mut sorted, PERCENTILE);
    let anchor_pre = quantile_of(&mut base, PERCENTILE);

    // 2. Both through the reference look, on the neutral axis.
    let stops_after = |s: f64| (reference.after_look(TARGET_GREY * s.exp2()) / TARGET_GREY).log2();
    let highlight_ev = stops_after(highlight_pre);
    let anchor_ev = stops_after(anchor_pre);
    if !highlight_ev.is_finite() || !anchor_ev.is_finite() {
        return Ok(WhiteEstimate::none());
    }

    // 3. Where the percentile is on the display and where it should go.
    let y_p = TARGET_GREY * highlight_ev.exp2();
    let code_before = reference.code(y_p);
    // How high-key the photo is is read from its own anchor, before the look: the look lifts the
    // shoulder by itself (the Signature look by about half a stop above +1), and counting that in
    // the ramp as well would make the lift grow twice for the same photo.
    let high_key =
        ((anchor_pre - HIGH_KEY_FROM_EV) / (HIGH_KEY_TO_EV - HIGH_KEY_FROM_EV)).clamp(0.0, 1.0);
    let target_code = code_before + high_key * (TARGET_CODE - code_before).max(0.0);

    // 4. The smallest amount that gets the percentile there (it is lifted by the ramp's value at its
    // own level, 1 whenever it is above the anchor).
    let span = (anchor_ev - KNEE_EV).max(MIN_SPAN_EV);
    let ramp = smoothstep((highlight_ev - KNEE_EV) / span);
    let mut amount = 0.0;
    let mut code_after = code_before;
    if target_code > code_before + 0.5 && ramp > 1e-9 {
        let lifted = |a: f64| reference.code(y_p * (a * ramp).exp2());
        if lifted(MAX_AUTO_AMOUNT_EV) <= target_code {
            amount = MAX_AUTO_AMOUNT_EV;
        } else {
            let (mut lo, mut hi) = (0.0, MAX_AUTO_AMOUNT_EV);
            for _ in 0..48 {
                let mid = 0.5 * (lo + hi);
                if lifted(mid) < target_code {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            amount = 0.5 * (lo + hi);
        }
        code_after = lifted(amount);
    }
    Ok(WhiteEstimate {
        amount_ev: amount,
        anchor_ev,
        high_key,
        highlight_ev,
        code_before,
        target_code,
        code_after,
        pixels,
    })
}
