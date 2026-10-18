//! Output sharpening for rendered and exported pixels, specified in
//! `docs/algorithms/04-output-sharpening.md`.
//!
//! A luma unsharp mask in linear light with three threshold methods, optional halo suppression
//! and a hue-preserving ratio transfer back to RGB (`RGB_out = RGB_in * clamp(L' / L, 0, 16)`).
//! It runs on display-linear pixels at the size they are shown or written, after any Lanczos-3
//! resize and before the transfer function and the dither. There is no gamma, log or perceptual
//! transform anywhere in the stage: every threshold is an absolute linear level (white = 1.0).
//! Integration: `docs/algorithms/INTEGRATION.md` sections 1, 2, 5 and 7.
//!
//! # Use
//!
//! ```
//! use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
//!
//! // The Detail slider "Output" maps to `amount_pct`; everything else is the screen preset.
//! let sharpen = OutputSharpen::screen().with_amount_pct(100.0);
//! // A 6000 px frame shown at a 2048 px long edge.
//! let s = scale_for_edges(2048, 6000);
//! let (width, height) = (8, 6);
//! let pixels = vec![0.18_f32; 3 * width * height]; // interleaved display-linear RGB
//! let sharpened = sharpen.apply_interleaved(width, height, &pixels, s).unwrap();
//! assert_eq!(sharpened.len(), pixels.len());
//! // A tiled caller pads each tile by this many pixels on every side.
//! assert!(sharpen.halo_px(s).unwrap() <= sharpen.max_halo_px());
//! ```
//!
//! # Contract
//!
//! * **Input**: display-linear RGB, white = 1.0, `f32`, not clamped (values above 1.0 and slight
//!   negatives are handled). The stage uses no gains and no noise model. Inputs must be finite:
//!   a NaN spreads through the blurs and only its own pixel's ratio is forced to 0.
//! * **Output**: same layout, units and size. Each channel keeps its sign; the only clip is a
//!   pixel whose sharpened luma goes below 0, which becomes black. A flat image, a 1 x 1 image
//!   and an amount below 0.1 % (no stage) come back bit-identical.
//! * **Layouts**: planar ([`RgbImage`]) and interleaved `f32` give identical results. Only the
//!   luma (a per-pixel function of the three channels) and the final multiplication touch RGB,
//!   and both are computed the same way in both layouts.
//! * **Scale**: the radius is defined in rendered (full-size) pixels. A caller that shows or
//!   writes a smaller image passes `scale = out_long_edge / rendered_long_edge` ([`scale_for_edges`]);
//!   sigma and the threshold are then rescaled (spec section 2.4). Full size is 1.0. Output is
//!   never upscaled, so the scale is in (0, 1].
//!
//! # Numerics
//!
//! As in the spec (section 3.9): planes are `f32`; the kernel taps, every convolution, the
//! high-pass, the halo-window sums and the envelope formula accumulate in `f64` in the stated
//! order (taps left to right, horizontal pass first, window row by row); per-pixel formulas
//! (luma, the methods, the accumulation, the clamp, the transfer) are `f32` without fused
//! multiply-add. The luma is `f32` in a fixed operation order because the halo limiter compares
//! luma values for equality and order. The high-pass and the window sums are taken as differences
//! to the centre sample so a flat region gives exact zeros. `min` and `max` follow the C++
//! semantics of the reference (`std::min`, `std::max`: the first argument wins when a comparison
//! fails), so a NaN behaves as there.
//!
//! Where this port differs from the reference, none of it changes a value:
//!
//! * **`erf`** comes from the `libm` crate (a port of the FreeBSD msun routine), the reference
//!   uses the C library's. Both are accurate to about one unit in the last place; the kernel taps
//!   agree to within 1e-15 (the tests compare them with the vectors' `%.17g` kernels).
//! * **Loop structure.** The convolutions run tap-outermost over a row of independent `f64`
//!   accumulators. Every accumulator still sums its taps left to right from 0, so the bits are
//!   those of the pixel-at-a-time loops. Borders are mirrored once into a padded row.
//! * **Bands.** The image is processed in bands of [`BAND_ROWS`] rows plus [`SharpenPlan::halo_px`]
//!   rows of context above and below (clipped to the frame), in parallel. This is the tiling
//!   contract of spec section 5.2 (bit-exact against the whole frame, tested), so the result does
//!   not depend on the band height or on the number of threads. It bounds the working memory to a
//!   few planes of a band instead of 12 to 16 planes of the whole frame.
//! * **`scale` is an argument** of the apply functions, not a field of [`OutputSharpen`]: the
//!   settings do not know the output size.
//!
//! # Receptive field
//!
//! [`SharpenPlan::halo_px`] is the exact figure of spec section 1.4 for a plan (17 px for the
//! screen preset at full size, 9 px at a 2048 px long edge from 6000, 14 px for the print preset
//! at full size); [`OutputSharpen::max_halo_px`] is the bound of the mode (21 px screen, 41 px
//! print), valid for every scale and every stage.

use std::f64::consts::SQRT_2;
use std::ops::Range;

use rayon::prelude::*;

use super::{RgbImage, mirror};

/// Core rows of one band of the parallel run. An implementation constant (the result does not
/// depend on it, see the module documentation), chosen so a 24 MP frame gives dozens of bands
/// while the halo rows add only about a third.
pub const BAND_ROWS: usize = 128;

/// Largest amount, percent.
const AMOUNT_MAX_PCT: f32 = 1000.0;
/// Largest threshold, 8-bit levels of the normalised signal.
const THRESHOLD_MAX_LEVELS: f32 = 100.0;
/// Largest radius of an extra stage, pixels.
const EXTRA_RADIUS_MAX_PX: f32 = 15.0;
/// Screen mode: radius cap, pixels.
const SCREEN_RADIUS_CAP_PX: f32 = 2.5;
/// Print mode: print distance per pixel of radius.
const PRINT_DISTANCE_PER_PX: f32 = 2500.0;
/// Print mode: radius cap, pixels.
const PRINT_RADIUS_CAP_PX: f32 = 15.0;
/// Tap cap in screen mode.
const MAX_TAPS_SCREEN: usize = 13;
/// Tap cap in print mode.
const MAX_TAPS_PRINT: usize = 33;
/// A stage needs at least this amount (fraction, 1.0 = 100 %).
const MIN_AMOUNT: f32 = 0.001;
/// A stage needs at least this sigma.
const MIN_SIGMA: f32 = 0.01;
/// Sigma below this is not rescaled with the output size.
const SIGMA_KNEE: f32 = 0.15;
/// Halo fractions at or below this are zero.
const HALO_DEAD_ZONE: f32 = 0.001;
/// Radius of the halo-limiter window (5 x 5).
const HALO_WINDOW_RADIUS: usize = 2;
/// Samples in the halo-limiter window.
const HALO_WINDOW_SAMPLES: usize = (2 * HALO_WINDOW_RADIUS + 1) * (2 * HALO_WINDOW_RADIUS + 1);
/// Smallest count that triggers the plateau pull of the envelope: `(N + 1) >> 2`.
const HALO_RANK_MIN: i32 = ((HALO_WINDOW_SAMPLES + 1) >> 2) as i32;
/// Default tie tolerance of the halo limiter, relative to the window's centre: `2^-12`.
pub const HALO_TIE_TOLERANCE: f32 = 1.0 / 4096.0;
/// Largest tie tolerance: `2^-4`. Below 1/2 a sample within the tolerance lies within a factor of
/// two of the centre, so its difference to the centre is exact in `f32` and `f64` alike.
const HALO_TIE_TOLERANCE_MAX: f32 = 0.0625;
/// With a tie tolerance, the plateau pull fades out over the last quarter before the two sides
/// spread equally (`d_near` from `0.75 d_far` to `d_far`) instead of switching off there.
const HALO_PULL_FADE: f64 = 0.25;
/// Epsilon of the gain-map denominator (units of `d` squared).
const GAIN_EPSILON: f32 = 1e-10;
/// Floor of the luma in the transfer.
const LUMA_FLOOR: f32 = 1e-20;
/// Largest ratio of the transfer.
const RATIO_MAX: f32 = 16.0;

/// Errors of output sharpening.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OutputSharpenError {
    /// A parameter is NaN, infinite or out of range.
    #[error("output sharpening: {name} {requirement}")]
    Invalid {
        /// The parameter.
        name: &'static str,
        /// What it must be.
        requirement: &'static str,
    },
    /// A buffer's length does not match its stated dimensions.
    #[error(
        "output sharpening: the buffer holds {got} samples but {width} x {height} pixels need {expected}"
    )]
    BadLength {
        /// Samples supplied.
        got: usize,
        /// Samples required.
        expected: usize,
        /// Stated width in pixels.
        width: usize,
        /// Stated height in pixels.
        height: usize,
    },
    /// The image has no pixels.
    #[error("output sharpening: the image has no pixels")]
    Empty,
}

/// Where the output stage takes its radius from, and how long its kernels may get.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Radius is `radius_px`, capped at 2.5; kernels have at most 13 taps.
    Screen,
    /// Radius is `print_distance / 2500`, capped at 15; kernels have at most 33 taps.
    Print,
}

/// How a stage turns the high-pass `d = L - blur(L)` into a luma delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// `s = d - clamp(d, -T, T)`; the amount is divided by `1 - T`.
    HardThreshold = 0,
    /// `s = d * W1`, `W1 = blur(|d| max(|d| - T, 0)) / blur(d^2 + eps)`.
    SmoothGain = 1,
    /// `s = d * W2`, `W2` from a ramp on the gradient of the blurred luma.
    GradientRamp = 2,
}

impl Method {
    /// The method's number (0, 1 or 2), as in the spec and the vectors.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// The method with this number; anything but 0, 1 and 2 is rejected.
    pub fn from_id(id: u32) -> Result<Method, OutputSharpenError> {
        match id {
            0 => Ok(Method::HardThreshold),
            1 => Ok(Method::SmoothGain),
            2 => Ok(Method::GradientRamp),
            _ => Err(OutputSharpenError::Invalid {
                name: "method",
                requirement: "must be 0, 1 or 2",
            }),
        }
    }
}

/// One more whole-image stage (spec section 2.7). Yellowmoon v1 has none: the list is empty and
/// the machinery stays so a creative control can be added without changing the stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtraStage {
    /// The method. With [`auto_method`](Self::auto_method) set and `HardThreshold`, the rule of
    /// spec section 2.7 picks between methods 0 and 1.
    pub method: Method,
    /// Choose method 0 or 1 by the rule (only with `method == HardThreshold`).
    pub auto_method: bool,
    /// Strength in percent, 0 to 1000.
    pub amount_pct: f32,
    /// Radius in rendered pixels, 0 to 15 (the mode caps the kernel length).
    pub radius_px: f32,
    /// Threshold in 8-bit levels, 0 to 100.
    pub threshold_levels: f32,
    /// Halo suppression, 0 to 100.
    pub halo_pct: f32,
}

/// The parameters of output sharpening (spec section 2.1), without the output scale, which is
/// passed to the apply functions.
///
/// `OutputSharpen::screen()` is the preset the Detail tab's "Output" slider uses (its amount is
/// the slider); [`print`](Self::print) is the print preset. Both fill every field, so a caller
/// changes at most [`amount_pct`](Self::amount_pct) through
/// [`with_amount_pct`](Self::with_amount_pct).
#[derive(Clone, Debug, PartialEq)]
pub struct OutputSharpen {
    /// Where the radius comes from and how long the kernels may get.
    pub mode: Mode,
    /// How the high-pass is thresholded; used as given (no automatic selection).
    pub method: Method,
    /// Strength in percent, 0 to 1000. Below 0.1 % no stage is created.
    pub amount_pct: f32,
    /// Screen mode: radius in rendered pixels, 0 or more (values above 2.5 are capped; values of
    /// 0.09 or less create no stage). Ignored in print mode.
    pub radius_px: f32,
    /// Threshold in 8-bit levels of the normalised signal, 0 to 100 (absolute, linear light).
    pub threshold_levels: f32,
    /// Halo suppression: the share of the delta that goes through the limiter,
    /// `H = sqrt(halo_pct / 100)`; 0 to 100.
    pub halo_pct: f32,
    /// Print mode: `radius = min(print_distance / 2500, 15)` pixels; 0 or more. Ignored in
    /// screen mode.
    pub print_distance: f32,
    /// Weights of the luma `sqrt(wR R^2 + wG G^2 + wB B^2)`; each finite and above 0. The default
    /// is the reference's `[0.299, 0.587, 0.114]` (spec section 2.6).
    pub luma_weights: [f32; 3],
    /// More whole-image stages; empty in Yellowmoon v1.
    pub extra_stages: Vec<ExtraStage>,
    /// Tie tolerance of the halo limiter, relative to the centre of its window, 0 to 1/16 (spec
    /// section 3.5). Above 0 the limiter is continuous: a sample within `|c| * halo_tie_tolerance`
    /// of the centre `c` counts on both sides, the counts and sums change smoothly over the next
    /// tolerance, and the plateau pull fades instead of switching. The default
    /// [`HALO_TIE_TOLERANCE`] (`2^-12`) keeps the last bits of a device's `f32` colour chain from
    /// flipping the limiter where values tie in exact arithmetic; 0 gives the reference's exact
    /// comparisons and switches.
    pub halo_tie_tolerance: f32,
}

impl Default for OutputSharpen {
    /// The screen preset.
    fn default() -> OutputSharpen {
        OutputSharpen::screen()
    }
}

/// The reference luma weights.
const REFERENCE_LUMA_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];

impl OutputSharpen {
    /// The screen preset (spec section 2.5): mode Screen, method 1 (smooth gain), amount 100 %,
    /// radius 1.0 px, threshold 1 level, halo 60 %, reference luma weights.
    pub fn screen() -> OutputSharpen {
        OutputSharpen {
            mode: Mode::Screen,
            method: Method::SmoothGain,
            amount_pct: 100.0,
            radius_px: 1.0,
            threshold_levels: 1.0,
            halo_pct: 60.0,
            print_distance: 5000.0,
            luma_weights: REFERENCE_LUMA_WEIGHTS,
            extra_stages: Vec::new(),
            halo_tie_tolerance: HALO_TIE_TOLERANCE,
        }
    }

    /// The print preset (spec section 2.5): mode Print, method 1, amount 150 %, print distance
    /// 5000 (radius 2 px), threshold 1 level, halo 0, reference luma weights. Kept in the code
    /// and tested; not exposed in the app.
    pub fn print() -> OutputSharpen {
        OutputSharpen {
            mode: Mode::Print,
            amount_pct: 150.0,
            halo_pct: 0.0,
            ..OutputSharpen::screen()
        }
    }

    /// The same parameters with another amount (percent). Not validated here; [`validate`](Self::validate)
    /// and the apply functions reject values outside 0 to 1000.
    #[must_use]
    pub fn with_amount_pct(mut self, amount_pct: f32) -> OutputSharpen {
        self.amount_pct = amount_pct;
        self
    }

    /// Checks every parameter: NaN, infinite and out-of-range values are rejected.
    pub fn validate(&self) -> Result<(), OutputSharpenError> {
        fn bad(name: &'static str, requirement: &'static str) -> OutputSharpenError {
            OutputSharpenError::Invalid { name, requirement }
        }
        if !in_range(self.amount_pct, 0.0, AMOUNT_MAX_PCT) {
            return Err(bad("amount_pct", "must be a number from 0 to 1000"));
        }
        if !self.radius_px.is_finite() || self.radius_px < 0.0 {
            return Err(bad("radius_px", "must be a number of 0 or more"));
        }
        if !in_range(self.threshold_levels, 0.0, THRESHOLD_MAX_LEVELS) {
            return Err(bad("threshold_levels", "must be a number from 0 to 100"));
        }
        if !in_range(self.halo_pct, 0.0, 100.0) {
            return Err(bad("halo_pct", "must be a number from 0 to 100"));
        }
        if !self.print_distance.is_finite() || self.print_distance < 0.0 {
            return Err(bad("print_distance", "must be a number of 0 or more"));
        }
        // A channel with weight 0 is invisible to the luma, and a pixel carrying only that
        // channel would get an arbitrary ratio.
        if self
            .luma_weights
            .iter()
            .any(|w| !w.is_finite() || *w <= 0.0)
        {
            return Err(bad("luma_weights", "must all be finite and above 0"));
        }
        if !in_range(self.halo_tie_tolerance, 0.0, HALO_TIE_TOLERANCE_MAX) {
            return Err(bad(
                "halo_tie_tolerance",
                "must be a number from 0 to 0.0625",
            ));
        }
        for e in &self.extra_stages {
            if !in_range(e.amount_pct, 0.0, AMOUNT_MAX_PCT) {
                return Err(bad(
                    "extra stage amount_pct",
                    "must be a number from 0 to 1000",
                ));
            }
            if !in_range(e.radius_px, 0.0, EXTRA_RADIUS_MAX_PX) {
                return Err(bad(
                    "extra stage radius_px",
                    "must be a number from 0 to 15",
                ));
            }
            if !in_range(e.threshold_levels, 0.0, THRESHOLD_MAX_LEVELS) {
                return Err(bad(
                    "extra stage threshold_levels",
                    "must be a number from 0 to 100",
                ));
            }
            if !in_range(e.halo_pct, 0.0, 100.0) {
                return Err(bad(
                    "extra stage halo_pct",
                    "must be a number from 0 to 100",
                ));
            }
        }
        Ok(())
    }

    /// Derives the stages for an output that is `scale` times the rendered size (spec sections
    /// 2.2 to 2.4 and 2.7). `scale` must be in (0, 1]; use [`scale_for_edges`].
    pub fn plan(&self, scale: f32) -> Result<SharpenPlan, OutputSharpenError> {
        self.validate()?;
        if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
            return Err(OutputSharpenError::Invalid {
                name: "scale",
                requirement: "must be a number above 0 and at most 1",
            });
        }
        Ok(make_plan(self, scale))
    }

    /// The exact receptive field in pixels at this scale (spec section 1.4): the padding a tile
    /// needs on every side to give the whole-frame result bit for bit. 0 when no stage exists.
    pub fn halo_px(&self, scale: f32) -> Result<usize, OutputSharpenError> {
        Ok(self.plan(scale)?.halo_px())
    }

    /// The bound of the mode, valid at every scale: 21 px for screen, 41 px for print. Equals the
    /// receptive field of the widest kernels the mode allows (method 2 with halo suppression).
    pub fn max_halo_px(&self) -> usize {
        let max_taps = max_taps(self.mode);
        let r = max_taps / 2;
        // Kernel radius, gradient, window, and the halo-map kernel at its largest (sigma 2).
        2 * r + 1 + HALO_WINDOW_RADIUS + kernel_radius(2.0, max_taps)
    }

    /// Sharpens a planar image. The result is identical to [`apply_interleaved`](Self::apply_interleaved)
    /// on the same pixels.
    pub fn apply(&self, img: &RgbImage, scale: f32) -> Result<RgbImage, OutputSharpenError> {
        let mut out = img.clone();
        self.apply_in_place(&mut out, scale)?;
        Ok(out)
    }

    /// Sharpens a planar image in place.
    pub fn apply_in_place(&self, img: &mut RgbImage, scale: f32) -> Result<(), OutputSharpenError> {
        let plan = self.plan(scale)?;
        let (w, h) = (img.width, img.height);
        check_dims(w, h, img.data.len())?;
        if plan.stages.is_empty() {
            return Ok(());
        }
        let luma = {
            let [r, g, b] = img.planes();
            luma_plane(w, h, |y, row| {
                let o = y * w;
                for (x, l) in row.iter_mut().enumerate() {
                    *l = luma_of(&plan.luma_weights, r[o + x], g[o + x], b[o + x]);
                }
            })
        };
        let ratio = ratio_plane(&plan, &luma, w, h, BAND_ROWS);
        drop(luma);
        let [r, g, b] = img.planes_mut();
        for plane in [r, g, b] {
            plane
                .par_iter_mut()
                .zip(ratio.par_iter())
                .for_each(|(v, &k)| *v *= k);
        }
        Ok(())
    }

    /// Sharpens interleaved `[r, g, b, r, g, b, ...]` samples, row-major, `width * height` pixels.
    /// The result is identical to [`apply`](Self::apply) on the same pixels.
    pub fn apply_interleaved(
        &self,
        width: usize,
        height: usize,
        rgb: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, OutputSharpenError> {
        let mut out = rgb.to_vec();
        self.apply_interleaved_in_place(width, height, &mut out, scale)?;
        Ok(out)
    }

    /// Sharpens interleaved samples in place.
    pub fn apply_interleaved_in_place(
        &self,
        width: usize,
        height: usize,
        rgb: &mut [f32],
        scale: f32,
    ) -> Result<(), OutputSharpenError> {
        let plan = self.plan(scale)?;
        check_dims(width, height, rgb.len())?;
        if plan.stages.is_empty() {
            return Ok(());
        }
        let w = width;
        let luma = luma_plane(w, height, |y, row| {
            let px = &rgb[3 * w * y..3 * w * (y + 1)];
            for (l, p) in row.iter_mut().zip(px.chunks_exact(3)) {
                *l = luma_of(&plan.luma_weights, p[0], p[1], p[2]);
            }
        });
        let ratio = ratio_plane(&plan, &luma, w, height, BAND_ROWS);
        drop(luma);
        rgb.par_chunks_mut(3 * w)
            .zip(ratio.par_chunks(w))
            .for_each(|(row, rr)| {
                for (p, &k) in row.chunks_exact_mut(3).zip(rr) {
                    p[0] *= k;
                    p[1] *= k;
                    p[2] *= k;
                }
            });
        Ok(())
    }
}

/// The scale `s = out_long_edge / rendered_long_edge` of spec section 2.4: the `f64` division of
/// the two integers, rounded to `f32`. Output is never upscaled, so the result is at most 1
/// (also when `rendered_long_edge` is 0, which gives 1).
pub fn scale_for_edges(out_long_edge_px: usize, rendered_long_edge_px: usize) -> f32 {
    if rendered_long_edge_px == 0 || out_long_edge_px >= rendered_long_edge_px {
        return 1.0;
    }
    (out_long_edge_px as f64 / rendered_long_edge_px as f64) as f32
}

// ------------------------------------------------------------------------------------------------
// Parameter derivation
// ------------------------------------------------------------------------------------------------

/// One derived stage (spec sections 2.2 and 2.7), in the units of the vectors' `derived.stages`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StagePlan {
    /// The method.
    pub method: Method,
    /// Amount as a fraction (1.0 = 100 %), after the boost of an extra stage, before the
    /// method-0 compensation.
    pub amount: f32,
    /// Gaussian sigma in output pixels (after the resolution scaling).
    pub sigma: f32,
    /// Threshold in normalised linear units (white = 1.0), after the resolution scaling.
    pub threshold: f32,
    /// The halo-limited share of the amount, 0 to 1.
    pub halo: f32,
    /// This is the output stage (the others are extra stages).
    pub output: bool,
}

/// The stages and constants one run needs, derived from [`OutputSharpen`] and the output scale.
#[derive(Clone, Debug, PartialEq)]
pub struct SharpenPlan {
    stages: Vec<StagePlan>,
    luma_weights: [f32; 3],
    scale: f32,
    max_taps: usize,
    halo_enabled: bool,
    halo_tie_tolerance: f32,
}

impl SharpenPlan {
    /// The stages in ascending sigma (ties in creation order). Empty when the amount or the
    /// radius is too small to make a stage: the run is then a copy.
    pub fn stages(&self) -> &[StagePlan] {
        &self.stages
    }

    /// The output scale the plan was derived for.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The tap cap of the mode: 13 (screen) or 33 (print).
    pub fn max_taps(&self) -> usize {
        self.max_taps
    }

    /// Whether any stage sends part of its delta through the halo limiter.
    pub fn halo_enabled(&self) -> bool {
        self.halo_enabled
    }

    /// The luma weights.
    pub fn luma_weights(&self) -> [f32; 3] {
        self.luma_weights
    }

    /// The halo limiter's tie tolerance, relative to the window's centre.
    pub fn halo_tie_tolerance(&self) -> f32 {
        self.halo_tie_tolerance
    }

    /// Sigma of the halo-limiter's smoothing kernel: `2 s` (f32).
    pub fn halo_sigma(&self) -> f32 {
        self.scale + self.scale
    }

    /// The high-pass kernel of stage `stage` (an index into [`stages`](Self::stages)): the
    /// Gaussian of the stage's sigma, capped at the mode's tap count.
    ///
    /// # Panics
    ///
    /// Panics when `stage` is not an index into [`stages`](Self::stages).
    pub fn kernel(&self, stage: usize) -> Vec<f64> {
        gaussian_kernel(self.stages[stage].sigma, self.max_taps)
    }

    /// The gain-map kernel of stage `stage`, sigma `max(2 sigma, 1)`. Only methods 1 and 2 with a
    /// threshold above 0 use it.
    ///
    /// # Panics
    ///
    /// Panics when `stage` is not an index into [`stages`](Self::stages).
    pub fn gain_kernel(&self, stage: usize) -> Vec<f64> {
        gaussian_kernel(gain_sigma(self.stages[stage].sigma), self.max_taps)
    }

    /// The kernel that smooths the four halo-limiter maps (sigma `2 s`). Only used when
    /// [`halo_enabled`](Self::halo_enabled).
    pub fn halo_kernel(&self) -> Vec<f64> {
        gaussian_kernel(self.halo_sigma(), self.max_taps)
    }

    /// The receptive field in pixels (spec section 1.4): per stage the radius of its blur, plus
    /// the gain-map blur and, for method 2, the 3 x 3 gradient (both only with a threshold);
    /// the largest of those, plus the 5 x 5 window and the halo-map blur when the limiter runs.
    pub fn halo_px(&self) -> usize {
        if self.stages.is_empty() {
            return 0;
        }
        let mut chain = 0;
        for st in &self.stages {
            let mut c = kernel_radius(st.sigma, self.max_taps);
            let gain = st.method != Method::HardThreshold && st.threshold > 0.0;
            if gain {
                c += kernel_radius(gain_sigma(st.sigma), self.max_taps);
                if st.method == Method::GradientRamp {
                    c += 1;
                }
            }
            chain = chain.max(c);
        }
        if self.halo_enabled {
            chain += HALO_WINDOW_RADIUS + kernel_radius(self.halo_sigma(), self.max_taps);
        }
        chain
    }
}

/// `std::max(a, b)` of the reference: `a` unless `a < b`.
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// `std::min(a, b)` of the reference: `a` unless `b < a`.
#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// `std::max` on `f64`.
#[inline(always)]
fn fmax64(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

fn in_range(v: f32, lo: f32, hi: f32) -> bool {
    v.is_finite() && v >= lo && v <= hi
}

const fn max_taps(mode: Mode) -> usize {
    match mode {
        Mode::Screen => MAX_TAPS_SCREEN,
        Mode::Print => MAX_TAPS_PRINT,
    }
}

/// Radius (px) to Gaussian sigma before resolution scaling; 0 means "no stage".
fn sigma_from_radius(r: f32) -> f32 {
    if r >= 0.5 {
        r * 0.7 + 0.3
    } else if r > 0.09 {
        (r - 0.09).sqrt() * 0.97
    } else {
        0.0
    }
}

/// Resolution scaling for an output `s` times the rendered size, `s` in (0, 1].
fn scale_sigma(sigma: f32, s: f32) -> f32 {
    if s >= 1.0 || sigma < SIGMA_KNEE {
        return sigma;
    }
    s * (sigma - SIGMA_KNEE) + SIGMA_KNEE
}

/// Centre tap of a pixel-integrated, unit-area Gaussian without truncation: `erf(1 / (2 sqrt(2) sigma))`.
fn centre_weight(sigma: f32) -> f64 {
    libm::erf(0.5 / (f64::from(sigma) * SQRT_2))
}

/// Keeps the threshold's meaning when sigma was rescaled: `T (1 - c(s1)^2) / (1 - c(s0)^2)`.
fn normalise_threshold(t: f32, sigma0: f32, sigma1: f32) -> f32 {
    if sigma1 == sigma0 {
        return t;
    }
    let c0 = centre_weight(sigma0);
    let c1 = centre_weight(sigma1);
    (((1.0 - c1 * c1) * f64::from(t)) / (1.0 - c0 * c0)) as f32
}

/// Halo suppression 0 to 100 to the halo-limited fraction `sqrt(h / 100)`, with a dead zone.
fn halo_fraction(pct: f32) -> f32 {
    let h = fmin(fmax(pct * 0.01, 0.0), 1.0);
    let r = h.sqrt();
    if r > HALO_DEAD_ZONE { r } else { 0.0 }
}

/// Automatic method choice for an extra stage (spec section 2.7).
fn auto_method(sigma: f32, amount: f32, threshold: f32) -> Method {
    let t = fmax(threshold - 0.004, 0.1);
    let r = fmax(sigma - 0.65, 0.1);
    if r * amount * t >= 0.025 {
        Method::SmoothGain
    } else {
        Method::HardThreshold
    }
}

/// Number of taps: `((int)(5 sigma + 2.5)) | 1`, at least 3, at most `max_taps` (f32 arithmetic).
///
/// The spec's caps are 13 and 33. A cap that is not odd and at least 3 (which the stage never
/// passes) is rounded down to the odd number of at least 3, so the kernel keeps an odd tap count
/// and a centre tap instead of underflowing in [`gaussian_kernel`].
fn tap_count(sigma: f32, max_taps: usize) -> usize {
    let cap = (max_taps.max(3) - 1) / 2 * 2 + 1;
    let n0 = ((sigma * 5.0 + 2.5) as i32 | 1) as usize;
    if n0 < 3 { 3 } else { n0.min(cap) }
}

fn kernel_radius(sigma: f32, max_taps: usize) -> usize {
    tap_count(sigma, max_taps) / 2
}

/// Sigma of the gain-map smoothing kernel: `max(2 sigma, 1)`.
fn gain_sigma(sigma: f32) -> f32 {
    let s2 = sigma + sigma;
    if s2 <= 1.0 { 1.0 } else { s2 }
}

/// The pixel-integrated Gaussian taps of spec section 2.3 (`f64`, odd count, unit sum): `sigma`
/// is the standard deviation in pixels, `max_taps` the cap of the mode (13 or 33; any other value
/// is rounded down to an odd number of at least 3). Exposed so a GPU port can load the same taps.
pub fn gaussian_kernel(sigma: f32, max_taps: usize) -> Vec<f64> {
    let n = tap_count(sigma, max_taps);
    let half = ((n - 1) / 2) as f64;
    let s = f64::from(sigma) * SQRT_2;
    let mut k = vec![0.0f64; n];
    let mut sum = 0.0f64;
    for (i, tap) in k.iter_mut().enumerate() {
        let x = i as f64 - half;
        let v = libm::erf((x + 0.5) / s) - libm::erf((x - 0.5) / s);
        *tap = v;
        sum += v;
    }
    for tap in &mut k {
        *tap /= sum;
    }
    k
}

/// Parameters to stages. Assumes [`OutputSharpen::validate`] passed and `scale` is in (0, 1].
fn make_plan(p: &OutputSharpen, s: f32) -> SharpenPlan {
    let max_taps = max_taps(p.mode);
    let mut stages: Vec<StagePlan> = Vec::new();
    // Raw halo percentages, mapped to fractions below.
    let mut halo_pct: Vec<f32> = Vec::new();

    // The output stage: its method is used as given.
    let r = match p.mode {
        Mode::Print => fmin(
            p.print_distance / PRINT_DISTANCE_PER_PX,
            PRINT_RADIUS_CAP_PX,
        ),
        Mode::Screen => fmin(p.radius_px, SCREEN_RADIUS_CAP_PX),
    };
    let sigma0 = sigma_from_radius(r);
    let amount = p.amount_pct / 100.0;
    let mut have_output = false;
    if sigma0 >= MIN_SIGMA && amount >= MIN_AMOUNT {
        let sigma = scale_sigma(sigma0, s);
        stages.push(StagePlan {
            method: p.method,
            amount,
            sigma,
            threshold: normalise_threshold(p.threshold_levels / 255.0, sigma0, sigma),
            halo: 0.0,
            output: true,
        });
        halo_pct.push(p.halo_pct);
        have_output = true;
    }

    // Extra stages (none in Yellowmoon v1).
    for c in &p.extra_stages {
        let sigma0 = sigma_from_radius(c.radius_px);
        let sigma = scale_sigma(sigma0, s);
        let amount = c.amount_pct / 100.0;
        if sigma >= MIN_SIGMA && amount >= MIN_AMOUNT {
            let threshold = normalise_threshold(c.threshold_levels / 255.0, sigma0, sigma);
            let method = if c.auto_method && c.method == Method::HardThreshold {
                auto_method(sigma, amount, threshold)
            } else {
                c.method
            };
            stages.push(StagePlan {
                method,
                amount,
                sigma,
                threshold,
                halo: 0.0,
                output: false,
            });
            halo_pct.push(c.halo_pct);
        }
    }

    // Halo suppression percentages to fractions.
    let mut halo_enabled = false;
    for (st, &pct) in stages.iter_mut().zip(&halo_pct) {
        st.halo = halo_fraction(pct);
        if st.halo > 0.0 {
            halo_enabled = true;
        }
    }

    // Finer extra stages are boosted by the output stage; the halo-limited part H * A is kept.
    if have_output {
        let out = stages[0];
        for st in stages.iter_mut().skip(1) {
            if st.sigma >= out.sigma {
                continue;
            }
            let b = ((out.sigma - st.sigma) * out.amount) / out.sigma;
            let a0 = st.amount;
            let a1 = a0 * (b + 1.0);
            st.amount = a1;
            st.halo = (a0 * st.halo) / a1;
        }
    }

    // Stable sort by sigma, ascending: ties keep their creation order.
    stages.sort_by(|a, b| {
        a.sigma
            .partial_cmp(&b.sigma)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    SharpenPlan {
        stages,
        luma_weights: p.luma_weights,
        scale: s,
        max_taps,
        halo_enabled,
        halo_tie_tolerance: p.halo_tie_tolerance,
    }
}

// ------------------------------------------------------------------------------------------------
// Layout helpers
// ------------------------------------------------------------------------------------------------

fn check_dims(width: usize, height: usize, len: usize) -> Result<(), OutputSharpenError> {
    if width == 0 || height == 0 {
        return Err(OutputSharpenError::Empty);
    }
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .unwrap_or(usize::MAX);
    if len != expected {
        return Err(OutputSharpenError::BadLength {
            got: len,
            expected,
            width,
            height,
        });
    }
    Ok(())
}

/// `L = sqrt(((wR R) R + (wG G) G) + (wB B) B)`, `f32`, in exactly this order, no fused
/// multiply-add (spec section 3.1).
#[inline(always)]
fn luma_of(w: &[f32; 3], r: f32, g: f32, b: f32) -> f32 {
    (((w[0] * r) * r + (w[1] * g) * g) + (w[2] * b) * b).sqrt()
}

/// Builds the luma plane row by row (in parallel); `fill(y, row)` writes row `y`.
fn luma_plane<F>(w: usize, h: usize, fill: F) -> Vec<f32>
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let mut luma = vec![0.0f32; w * h];
    luma.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| fill(y, row));
    luma
}

// ------------------------------------------------------------------------------------------------
// The convolutions
// ------------------------------------------------------------------------------------------------

/// Copies `src` into `dst[r..r + src.len()]` and mirrors `r` samples on each side (the mirror
/// of `ym_gpu::cpu::mirror`, which also handles kernels longer than the row).
fn pad_row(dst: &mut [f32], src: &[f32], r: usize) {
    let w = src.len();
    dst[r..r + w].copy_from_slice(src);
    for i in 0..r {
        dst[r - 1 - i] = src[mirror(-1 - i as isize, w)];
        dst[r + w + i] = src[mirror((w + i) as isize, w)];
    }
}

/// Separable convolution with mirrored borders (spec section 3.6): horizontal pass over every
/// row, then vertical pass; taps summed left to right in `f64`, the intermediate plane stored in
/// `f32`.
fn blur(src: &[f32], w: usize, h: usize, k: &[f64]) -> Vec<f32> {
    let r = k.len() / 2;
    let mut tmp = vec![0.0f32; w * h];
    let mut dst = vec![0.0f32; w * h];
    let mut pad = vec![0.0f32; w + 2 * r];
    let mut acc = vec![0.0f64; w];
    for y in 0..h {
        pad_row(&mut pad, &src[y * w..(y + 1) * w], r);
        acc.fill(0.0);
        for (j, &kj) in k.iter().enumerate() {
            for (a, &v) in acc.iter_mut().zip(&pad[j..j + w]) {
                *a += kj * f64::from(v);
            }
        }
        for (o, &a) in tmp[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = a as f32;
        }
    }
    for y in 0..h {
        acc.fill(0.0);
        for (j, &kj) in k.iter().enumerate() {
            let yy = mirror(y as isize + j as isize - r as isize, h);
            for (a, &v) in acc.iter_mut().zip(&tmp[yy * w..(yy + 1) * w]) {
                *a += kj * f64::from(v);
            }
        }
        for (o, &a) in dst[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = a as f32;
        }
    }
    dst
}

/// The high-pass `h = G * L - L` in difference form (spec section 3.2): every term relative to the
/// centre sample, so a flat region gives exactly 0 at any precision.
///
/// ```text
/// t(x, y) = sum_j k_j (L(x + j - r, y) - L(x, y))
/// h(x, y) = sum_j k_j ((L(x, y + j - r) - L(x, y)) + t(x, y + j - r))
/// ```
fn highpass(l: &[f32], w: usize, h: usize, k: &[f64]) -> Vec<f32> {
    let r = k.len() / 2;
    let mut t = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];
    let mut pad = vec![0.0f32; w + 2 * r];
    let mut acc = vec![0.0f64; w];
    for y in 0..h {
        let row = &l[y * w..(y + 1) * w];
        pad_row(&mut pad, row, r);
        acc.fill(0.0);
        for (j, &kj) in k.iter().enumerate() {
            for ((a, &v), &c) in acc.iter_mut().zip(&pad[j..j + w]).zip(row) {
                *a += kj * (f64::from(v) - f64::from(c));
            }
        }
        for (o, &a) in t[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = a as f32;
        }
    }
    for y in 0..h {
        let row = &l[y * w..(y + 1) * w];
        acc.fill(0.0);
        for (j, &kj) in k.iter().enumerate() {
            let yy = mirror(y as isize + j as isize - r as isize, h);
            let lrow = &l[yy * w..(yy + 1) * w];
            let trow = &t[yy * w..(yy + 1) * w];
            for (((a, &lv), &c), &tv) in acc.iter_mut().zip(lrow).zip(row).zip(trow) {
                *a += kj * ((f64::from(lv) - f64::from(c)) + f64::from(tv));
            }
        }
        for (o, &a) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = a as f32;
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------
// Gain maps (methods 1 and 2)
// ------------------------------------------------------------------------------------------------

/// `W = blur(P) / blur(Q)` with the gain kernel: the smoothed numerator over the smoothed
/// denominator, divided in `f32`.
fn gain_ratio(p: &[f32], q: &[f32], w: usize, h: usize, k2: &[f64]) -> Vec<f32> {
    let pb = blur(p, w, h, k2);
    let qb = blur(q, w, h, k2);
    pb.iter().zip(&qb).map(|(&a, &b)| a / b).collect()
}

/// Method 1 gain map: `W1 = blur(|d| max(|d| - T, 0)) / blur(d^2 + eps)`, `T > 0`.
fn smooth_gain(d: &[f32], w: usize, h: usize, thr: f32, k2: &[f64]) -> Vec<f32> {
    let mut p = vec![0.0f32; d.len()];
    let mut q = vec![0.0f32; d.len()];
    for ((pv, qv), &v) in p.iter_mut().zip(&mut q).zip(d) {
        let a = v.abs();
        *pv = a * fmax(a - thr, 0.0);
        *qv = v * v + GAIN_EPSILON;
    }
    gain_ratio(&p, &q, w, h, k2)
}

/// Method 2 gain map: a ramp on the Scharr gradient of the blurred luma `B`,
/// `W2 = blur(d^2 w) / blur(d^2 + eps)`, `T > 0`.
fn gradient_gain(
    d: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    thr: f32,
    k: &[f64],
    k2: &[f64],
) -> Vec<f32> {
    let c = k.len() / 2;
    let left: f64 = k[..c].iter().sum();
    // g0: gradient of the blurred luma at an ideal step whose high-pass response equals T. A
    // kernel that has collapsed to one tap (left == 0) has no ramp: w = 0 (and d = 0 anyway).
    let ramp = left > 0.0;
    let g0 = if ramp {
        f64::from(thr) * ((k[c] + k[c + 1]) * 0.5) / left
    } else {
        0.0
    };
    let g0f = g0 as f32;
    let inv = if ramp { (0.25 / g0) as f32 } else { 0.0 };
    let mut p = vec![0.0f32; d.len()];
    let mut q = vec![0.0f32; d.len()];
    for y in 0..h {
        let ym = mirror(y as isize - 1, h);
        let yp = mirror(y as isize + 1, h);
        let (rm, r0, rp) = (
            &b[ym * w..(ym + 1) * w],
            &b[y * w..(y + 1) * w],
            &b[yp * w..(yp + 1) * w],
        );
        for x in 0..w {
            let xm = mirror(x as isize - 1, w);
            let xp = mirror(x as isize + 1, w);
            let (um, u0, up) = (rm[xm], rm[x], rm[xp]);
            let (cm, cp) = (r0[xm], r0[xp]);
            let (dm, d0, dp) = (rp[xm], rp[x], rp[xp]);
            let gx = (cp - cm) * 0.3125 + (((up + dp) - um) - dm) * 0.09375;
            let gy = (d0 - u0) * 0.3125 + (((dm + dp) - um) - up) * 0.09375;
            let wgt = if ramp {
                let t = inv * ((gx * gx + gy * gy).sqrt() - g0f);
                fmin(fmax(t, 0.0), 1.0)
            } else {
                0.0
            };
            let v = d[y * w + x];
            p[y * w + x] = v * v * wgt;
            q[y * w + x] = v * v + GAIN_EPSILON;
        }
    }
    gain_ratio(&p, &q, w, h, k2)
}

// ------------------------------------------------------------------------------------------------
// Halo limiter
// ------------------------------------------------------------------------------------------------

/// Width of the window (5) and radius (2) of the halo limiter.
const WINDOW_SPAN: usize = 2 * HALO_WINDOW_RADIUS + 1;

/// The tie tolerance `τ = |c| * rel` of a window centre `c` (spec section 3.5), one `f32` product as
/// on the device, and its inverse (`f64::MAX` for `τ = 0`, which makes [`soft_side`] an exact
/// comparison).
#[inline(always)]
fn tie_tolerance(c: f32, rel: f32) -> (f64, f64) {
    let t = f64::from(c.abs() * rel);
    (t, if t > 0.0 { 1.0 / t } else { f64::MAX })
}

/// How much a sample `d` from the centre counts on the low side and on the high side (spec section
/// 3.5): fully within `τ` of the centre (on both sides) and on its own side, not at all beyond `2τ`
/// on the other side, linearly between. With `τ = 0` (`inv = f64::MAX`) these are the exact
/// comparisons `d <= 0` and `d >= 0`. A NaN counts on neither side.
#[inline(always)]
fn soft_side(d: f64, inv: f64) -> (f64, f64) {
    (unit(2.0 - d * inv), unit(2.0 + d * inv))
}

/// `w` limited to `[0, 1]`, with a NaN giving 0.
#[inline(always)]
fn unit(w: f64) -> f64 {
    if w >= 1.0 {
        1.0
    } else if w > 0.0 {
        w
    } else {
        0.0
    }
}

/// The part of a difference `d` beyond the tolerance `t`: `d` shrunk towards 0 by `t`, 0 within it
/// (a NaN stays NaN and is dropped by the sums' `min` and `max`).
#[inline(always)]
fn beyond(d: f64, t: f64) -> f64 {
    let inside = if d < -t {
        -t
    } else if d > t {
        t
    } else {
        d
    };
    d - inside
}

/// The window centres of `P` adjacent pixels in `L` and `S`, their tie tolerances and the
/// tolerances' inverses.
struct Centres<const P: usize> {
    l: [f64; P],
    s: [f64; P],
    tl: [f64; P],
    ts: [f64; P],
    inv_l: [f64; P],
    inv_s: [f64; P],
}

impl<const P: usize> Centres<P> {
    /// The pixels `i..i + P` of the planes `l` and `s`, with the relative tie tolerance `tie`.
    #[inline(always)]
    fn at(l: &[f32], s: &[f32], i: usize, tie: f32) -> Centres<P> {
        let tl: [(f64, f64); P] = std::array::from_fn(|j| tie_tolerance(l[i + j], tie));
        let ts: [(f64, f64); P] = std::array::from_fn(|j| tie_tolerance(s[i + j], tie));
        Centres {
            l: std::array::from_fn(|j| f64::from(l[i + j])),
            s: std::array::from_fn(|j| f64::from(s[i + j])),
            tl: tl.map(|t| t.0),
            ts: ts.map(|t| t.0),
            inv_l: tl.map(|t| t.1),
            inv_s: ts.map(|t| t.1),
        }
    }
}

/// The counts and difference sums of the window comparisons for `P` adjacent pixels with the
/// centres `c`, in the order `[n_lo, n_hi, n_slo, n_shi, e_lo, e_hi, e_slo, e_shi]`: how much
/// each window sample counts at or below and at or above the centre ([`soft_side`]), and the sums
/// of the parts of the differences beyond the tolerance ([`beyond`]), below and above (spec
/// section 3.5). With a tolerance of 0 these are the reference's exact counts and sums.
///
/// `rows` are the offsets of the five window rows in the padded planes `lp` and `sp`, and `x` is
/// the column of the first pixel. The 25 samples of each pixel are visited window row by window
/// row, left to right, as in the reference, so every sum has the reference's bits. The pixels of
/// the group are independent lanes (the compiler keeps their accumulators in registers and
/// vectorises across them); compares become selects, so there is no branch per sample. Counts
/// are `f64` (exact) so every lane has the same width. The difference of two `f32` values is exact
/// in `f64`, and wherever it is near the tolerance it is exact in `f32` too, so the device makes
/// the same decisions from the same values.
#[inline(always)]
fn window_sums<const P: usize>(
    lp: &[f32],
    sp: &[f32],
    rows: &[usize; WINDOW_SPAN],
    x: usize,
    c: &Centres<P>,
) -> [[f64; P]; 8] {
    let mut acc = [[0.0f64; P]; 8];
    for &base in rows {
        // The samples of this window row that the P pixels read.
        let o = base + x;
        let lr = &lp[o..o + WINDOW_SPAN + P - 1];
        let sr = &sp[o..o + WINDOW_SPAN + P - 1];
        for dx in 0..WINDOW_SPAN {
            for j in 0..P {
                let lq = f64::from(lr[dx + j]);
                let sq = f64::from(sr[dx + j]);
                let dl = lq - c.l[j];
                let ds = sq - c.s[j];
                // A NaN is in no count and adds nothing (`max` and `min` drop it).
                let (lo, hi) = soft_side(dl, c.inv_l[j]);
                let (slo, shi) = soft_side(ds, c.inv_s[j]);
                let el = beyond(dl, c.tl[j]);
                let es = beyond(ds, c.ts[j]);
                acc[0][j] += lo;
                acc[1][j] += hi;
                acc[2][j] += slo;
                acc[3][j] += shi;
                acc[4][j] += el.min(0.0);
                acc[5][j] += el.max(0.0);
                acc[6][j] += es.min(0.0);
                acc[7][j] += es.max(0.0);
            }
        }
    }
    acc
}

/// The weight of the plateau pull on the side whose spread per sample is `near`, when the other
/// side's is `far` (spec section 3.5, with a tie tolerance): the reference's `far / (near + far / N)`
/// while `near <= 0.75 far`, fading linearly to 0 at `near = far`, and with `eps` in the denominator
/// so that it also fades as both spreads vanish. 0 when `far <= near` or either is NaN.
#[inline(always)]
fn pull_weight(near: f64, far: f64, eps: f64) -> f64 {
    let inv_n = 1.0f64 / HALO_WINDOW_SAMPLES as f64;
    if far > near {
        let fade = ((far - near) / (HALO_PULL_FADE * far)).min(1.0);
        fade * far / (near + far * inv_n + eps)
    } else {
        0.0
    }
}

/// The four envelope maps of one pixel from its window sums: `[v_hi, w_hi, v_lo, w_lo]` (spec
/// section 3.5), `f64` arithmetic stored as `f32`. `tl` is the centre's tie tolerance in `L` and
/// `soft` says whether a tolerance is in use; without one the plateau pull switches as in the
/// reference.
#[inline(always)]
fn envelope(ca: f64, sa: f64, sums: [f64; 8], tl: f64, soft: bool) -> [f32; 4] {
    let inv_n = 1.0f64 / HALO_WINDOW_SAMPLES as f64;
    let inv_r = 1.0f64 / HALO_WINDOW_RADIUS as f64;
    let [n_lo, n_hi, n_slo, n_shi, e_lo, e_hi, e_slo, e_shi] = sums;
    // Robust local maximum / minimum of L: 2 mean(values >= c) - c = c + 2 mean(l - c).
    let m_hi = 2.0 * e_hi / n_hi; // >= 0
    let m_lo = 2.0 * e_lo / n_lo; // <= 0
    let max_l = ca + m_hi;
    let min_l = ca + m_lo;
    let mut w_hi = fmax64(1.0 - (n_hi - n_shi).abs() * inv_r, inv_n);
    let mut v_hi = (sa + 2.0 * e_shi / n_shi) * w_hi;
    let mut w_lo = fmax64(1.0 - (n_lo - n_slo).abs() * inv_r, inv_n);
    let mut v_lo = (sa + 2.0 * e_slo / n_slo) * w_lo;
    let d_hi = m_hi / n_hi;
    let d_lo = -m_lo / n_lo;
    // On the flat side of an edge, pull the envelope to the plateau of L.
    let rank_min = f64::from(HALO_RANK_MIN);
    if soft {
        // Continuous: the count condition `n > k_min` ramps over one count, the switch on the
        // spreads fades (`pull_weight`), and `eps` is the spread of one sample `τ` beyond the tie.
        let eps = 2.0 * tl * inv_n * inv_n;
        let t_hi = (n_hi - rank_min).clamp(0.0, 1.0) * pull_weight(d_hi, d_lo, eps);
        let t_lo = (n_lo - rank_min).clamp(0.0, 1.0) * pull_weight(d_lo, d_hi, eps);
        if t_hi > 0.0 {
            v_hi += max_l * t_hi;
            w_hi += t_hi;
        }
        if t_lo > 0.0 {
            v_lo += min_l * t_lo;
            w_lo += t_lo;
        }
        return [v_hi as f32, w_hi as f32, v_lo as f32, w_lo as f32];
    }
    if n_hi > rank_min && d_hi < d_lo {
        let t = d_lo / (d_hi + d_lo * inv_n);
        v_hi += max_l * t;
        w_hi += t;
    }
    if n_lo > rank_min && d_lo < d_hi {
        let t = d_hi / (d_lo + d_hi * inv_n);
        v_lo += min_l * t;
        w_lo += t;
    }
    [v_hi as f32, w_hi as f32, v_lo as f32, w_lo as f32]
}

/// Robust local envelopes of `L` and of `S = L + Hacc` on a 5 x 5 window (spec section 3.5):
/// returns `(v_hi, w_hi, v_lo, w_lo)`. Sums are taken over differences to the centre value, so a
/// flat window gives exact zeros; the arithmetic is `f64`. `tie` is the relative tie tolerance.
fn halo_maps(l: &[f32], s: &[f32], w: usize, h: usize, tie: f32) -> [Vec<f32>; 4] {
    const R: usize = HALO_WINDOW_RADIUS;
    let stride = w + 2 * R;
    // Mirrored copies of both planes, so the 25 reads of a pixel never branch.
    let mut lp = vec![0.0f32; stride * h];
    let mut sp = vec![0.0f32; stride * h];
    for y in 0..h {
        pad_row(
            &mut lp[y * stride..(y + 1) * stride],
            &l[y * w..(y + 1) * w],
            R,
        );
        pad_row(
            &mut sp[y * stride..(y + 1) * stride],
            &s[y * w..(y + 1) * w],
            R,
        );
    }
    let mut out = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];
    let soft = tie > 0.0;
    let mut finish = |o: usize, ca: f64, sa: f64, tl: f64, sums: [f64; 8]| {
        for (plane, v) in out.iter_mut().zip(envelope(ca, sa, sums, tl, soft)) {
            plane[o] = v;
        }
    };
    for y in 0..h {
        let rows: [usize; WINDOW_SPAN] =
            std::array::from_fn(|k| mirror(y as isize + k as isize - R as isize, h) * stride);
        let mut x = 0;
        while x < w {
            let i = y * w + x;
            if w - x >= 2 {
                // Two pixels at a time: the two f64 lanes of a vector.
                let c = Centres::<2>::at(l, s, i, tie);
                let sums = window_sums(&lp, &sp, &rows, x, &c);
                for (j, ((&cl, &cs), &tl)) in c.l.iter().zip(&c.s).zip(&c.tl).enumerate() {
                    finish(i + j, cl, cs, tl, std::array::from_fn(|k| sums[k][j]));
                }
                x += 2;
            } else {
                let c = Centres::<1>::at(l, s, i, tie);
                let sums = window_sums(&lp, &sp, &rows, x, &c);
                finish(
                    i,
                    c.l[0],
                    c.s[0],
                    c.tl[0],
                    std::array::from_fn(|k| sums[k][0]),
                );
                x += 1;
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------
// The run
// ------------------------------------------------------------------------------------------------

/// The kernels of a plan: for each stage its high-pass kernel and (methods 1 and 2 with a
/// threshold) its gain kernel, plus the halo-map kernel.
struct Kernels {
    stages: Vec<(Vec<f64>, Option<Vec<f64>>)>,
    halo: Option<Vec<f64>>,
}

impl Kernels {
    fn new(plan: &SharpenPlan) -> Kernels {
        let stages = plan
            .stages
            .iter()
            .enumerate()
            .map(|(i, st)| {
                let gain = (st.method != Method::HardThreshold && st.threshold > 0.0)
                    .then(|| plan.gain_kernel(i));
                (plan.kernel(i), gain)
            })
            .collect();
        let halo = plan.halo_enabled.then(|| plan.halo_kernel());
        Kernels { stages, halo }
    }
}

/// The transfer ratio `clamp((L + D) / max(L, 1e-20), 0, 16)` of every pixel (a NaN gives 0), for
/// the whole `luma` plane, in bands of `band_rows` core rows.
fn ratio_plane(plan: &SharpenPlan, luma: &[f32], w: usize, h: usize, band_rows: usize) -> Vec<f32> {
    let kernels = Kernels::new(plan);
    let halo = plan.halo_px();
    let band_rows = band_rows.max(1);
    let mut ratio = vec![0.0f32; w * h];
    ratio
        .par_chunks_mut(band_rows * w)
        .enumerate()
        .for_each(|(bi, out)| {
            let y0 = bi * band_rows;
            let y1 = (y0 + band_rows).min(h);
            // The band: the core rows plus `halo` rows of context, clipped to the frame. A
            // band edge on the frame edge mirrors exactly like the whole frame; an interior
            // edge only corrupts rows farther than `halo` from the core.
            let g0 = y0.saturating_sub(halo);
            let g1 = (y1 + halo).min(h);
            band_ratio(
                plan,
                &kernels,
                &luma[g0 * w..g1 * w],
                w,
                g1 - g0,
                (y0 - g0)..(y1 - g0),
                out,
            );
        });
    ratio
}

/// Runs every stage on one band (`luma` is `h` rows of `w`) and writes the ratio of the rows in
/// `core` to `out`.
fn band_ratio(
    plan: &SharpenPlan,
    kernels: &Kernels,
    luma: &[f32],
    w: usize,
    h: usize,
    core: Range<usize>,
    out: &mut [f32],
) {
    let n = w * h;
    let mut free = vec![0.0f32; n]; // D: the accumulated free delta
    let mut limited = if plan.halo_enabled {
        vec![0.0f32; n] // Hacc: the accumulated halo-limited delta
    } else {
        Vec::new()
    };

    for (st, (k, gain_kernel)) in plan.stages.iter().zip(&kernels.stages) {
        let hp = highpass(luma, w, h, k);
        let d: Vec<f32> = hp.iter().map(|&v| -v).collect(); // d = L - G * L
        let gain: Option<Vec<f32>> = match (st.method, gain_kernel) {
            (Method::SmoothGain, Some(k2)) => Some(smooth_gain(&d, w, h, st.threshold, k2)),
            (Method::GradientRamp, Some(k2)) => {
                let b: Vec<f32> = luma.iter().zip(&hp).map(|(&l, &v)| l + v).collect(); // G * L
                Some(gradient_gain(&d, &b, w, h, st.threshold, k, k2))
            }
            _ => None,
        };
        drop(hp);
        let a = match st.method {
            Method::HardThreshold => st.amount / (1.0 - st.threshold),
            _ => st.amount,
        };
        let (a_free, a_halo) = if st.halo > 0.0 {
            let ah = st.halo * a;
            (a - ah, ah)
        } else {
            (a, 0.0)
        };
        let t = st.threshold;
        for i in 0..n {
            let v = d[i];
            let s = match (st.method, &gain) {
                (Method::HardThreshold, _) => v - fmin(fmax(v, -t), t),
                (_, Some(wg)) => v * fmin(fmax(wg[i], 0.0), 1.0),
                (_, None) => v, // T == 0: the gain map is 1 everywhere
            };
            free[i] += a_free * s;
            if st.halo > 0.0 {
                limited[i] += a_halo * s;
            }
        }
    }

    // The halo-map kernel exists exactly when some stage sends part of its delta to the limiter.
    if let Some(kh) = &kernels.halo {
        let s: Vec<f32> = luma.iter().zip(&limited).map(|(&l, &hv)| l + hv).collect();
        let maps = halo_maps(luma, &s, w, h, plan.halo_tie_tolerance);
        drop(s);
        let [b_v_hi, b_w_hi, b_v_lo, b_w_lo] = maps.map(|m| blur(&m, w, h, kh));
        for i in 0..n {
            let up = fmax(b_v_hi[i] / b_w_hi[i] - luma[i], 0.0);
            let lo = fmin(b_v_lo[i] / b_w_lo[i] - luma[i], 0.0);
            free[i] += fmax(lo, fmin(limited[i], up));
        }
    }

    for (row, y) in core.clone().enumerate() {
        let o = &mut out[row * w..(row + 1) * w];
        for (x, r) in o.iter_mut().enumerate() {
            let i = y * w + x;
            let l = fmax(luma[i], LUMA_FLOOR);
            let lp = luma[i] + free[i];
            let ratio = lp / l;
            // A NaN fails the comparison and gives 0.
            *r = if ratio >= 0.0 {
                fmin(ratio, RATIO_MAX)
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic plane of `f32` values: noise (optionally) on a ramp with a step.
    fn plane(w: usize, h: usize, noise: f32, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..w * h)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let (x, y) = (i % w, i / w);
                let base = 0.1 + 0.3 * (x as f32 / w as f32) + if y * 2 > h { 0.2 } else { 0.0 };
                base + ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * noise
            })
            .collect()
    }

    fn bits_equal(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The convolution as the spec writes it: one pixel at a time, taps left to right.
    fn blur_pixelwise(src: &[f32], w: usize, h: usize, k: &[f64]) -> Vec<f32> {
        let r = (k.len() / 2) as isize;
        let mut tmp = vec![0.0f32; w * h];
        let mut dst = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f64;
                for (j, &kj) in k.iter().enumerate() {
                    acc += kj * f64::from(src[y * w + mirror(x as isize + j as isize - r, w)]);
                }
                tmp[y * w + x] = acc as f32;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f64;
                for (j, &kj) in k.iter().enumerate() {
                    acc += kj * f64::from(tmp[mirror(y as isize + j as isize - r, h) * w + x]);
                }
                dst[y * w + x] = acc as f32;
            }
        }
        dst
    }

    /// The difference-form high-pass as the spec writes it (section 3.2).
    fn highpass_pixelwise(l: &[f32], w: usize, h: usize, k: &[f64]) -> Vec<f32> {
        let r = (k.len() / 2) as isize;
        let mut t = vec![0.0f32; w * h];
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = f64::from(l[y * w + x]);
                let mut acc = 0.0f64;
                for (j, &kj) in k.iter().enumerate() {
                    let q = f64::from(l[y * w + mirror(x as isize + j as isize - r, w)]);
                    acc += kj * (q - c);
                }
                t[y * w + x] = acc as f32;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let c = f64::from(l[y * w + x]);
                let mut acc = 0.0f64;
                for (j, &kj) in k.iter().enumerate() {
                    let yy = mirror(y as isize + j as isize - r, h);
                    acc += kj * ((f64::from(l[yy * w + x]) - c) + f64::from(t[yy * w + x]));
                }
                out[y * w + x] = acc as f32;
            }
        }
        out
    }

    #[test]
    fn convolutions_have_the_bits_of_the_pixelwise_loops() {
        let sizes = [(1, 1), (2, 7), (5, 3), (13, 2), (31, 17), (64, 9), (70, 70)];
        for (w, h) in sizes {
            let l = plane(w, h, 0.02, (w * 100 + h) as u64);
            for (sigma, cap) in [(0.3f32, 13), (1.0, 13), (2.05, 13), (4.1, 13), (5.9, 33)] {
                let k = gaussian_kernel(sigma, cap);
                assert!(
                    bits_equal(&blur(&l, w, h, &k), &blur_pixelwise(&l, w, h, &k)),
                    "blur {w}x{h} sigma {sigma}"
                );
                assert!(
                    bits_equal(&highpass(&l, w, h, &k), &highpass_pixelwise(&l, w, h, &k)),
                    "highpass {w}x{h} sigma {sigma}"
                );
            }
        }
        // A flat plane has an exactly zero high-pass at any kernel.
        let flat = vec![0.37f32; 40 * 30];
        let k = gaussian_kernel(2.05, 13);
        assert!(highpass(&flat, 40, 30, &k).iter().all(|&v| v == 0.0));
    }

    /// The window pass as the spec writes it: per pixel, 25 samples, branches, `f64` sums.
    fn halo_maps_pixelwise(l: &[f32], s: &[f32], w: usize, h: usize, tie: f32) -> [Vec<f32>; 4] {
        let r = 2isize;
        let mut out = [
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
        ];
        for y in 0..h {
            for x in 0..w {
                let c = l[y * w + x];
                let sc = s[y * w + x];
                let (ca, sa) = (f64::from(c), f64::from(sc));
                let (tl, ts) = (f64::from(c.abs() * tie), f64::from(sc.abs() * tie));
                // How much a difference counts below and above the centre: fully within the
                // tolerance and on its own side, linearly down to 0 at twice the tolerance on the
                // other side; exact comparisons without a tolerance.
                let weights = |d: f64, t: f64| -> (f64, f64) {
                    if t > 0.0 {
                        let inv = 1.0 / t;
                        let lo = (2.0 - d * inv).clamp(0.0, 1.0);
                        let hi = (2.0 + d * inv).clamp(0.0, 1.0);
                        (lo, hi)
                    } else {
                        (f64::from(u8::from(d <= 0.0)), f64::from(u8::from(d >= 0.0)))
                    }
                };
                // The part of a difference beyond the tolerance.
                let excess = |d: f64, t: f64| -> f64 {
                    if d > t {
                        d - t
                    } else if d < -t {
                        d + t
                    } else {
                        0.0
                    }
                };
                let (mut n_lo, mut n_hi, mut n_slo, mut n_shi) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let (mut e_lo, mut e_hi, mut e_slo, mut e_shi) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for dy in -r..=r {
                    let yy = mirror(y as isize + dy, h);
                    for dx in -r..=r {
                        let xx = mirror(x as isize + dx, w);
                        let dl = f64::from(l[yy * w + xx]) - ca;
                        let ds = f64::from(s[yy * w + xx]) - sa;
                        if !dl.is_nan() {
                            let (lo, hi) = weights(dl, tl);
                            n_lo += lo;
                            n_hi += hi;
                            let e = excess(dl, tl);
                            if e < 0.0 {
                                e_lo += e;
                            } else if e > 0.0 {
                                e_hi += e;
                            }
                        }
                        if !ds.is_nan() {
                            let (lo, hi) = weights(ds, ts);
                            n_slo += lo;
                            n_shi += hi;
                            let e = excess(ds, ts);
                            if e < 0.0 {
                                e_slo += e;
                            } else if e > 0.0 {
                                e_shi += e;
                            }
                        }
                    }
                }
                let (inv_n, inv_r) = (1.0f64 / 25.0, 1.0f64 / 2.0);
                let m_hi = 2.0 * e_hi / n_hi;
                let m_lo = 2.0 * e_lo / n_lo;
                let (max_l, min_l) = (ca + m_hi, ca + m_lo);
                let mut w_hi = fmax64(1.0 - (n_hi - n_shi).abs() * inv_r, inv_n);
                let mut v_hi = (sa + 2.0 * e_shi / n_shi) * w_hi;
                let mut w_lo = fmax64(1.0 - (n_lo - n_slo).abs() * inv_r, inv_n);
                let mut v_lo = (sa + 2.0 * e_slo / n_slo) * w_lo;
                let d_hi = m_hi / n_hi;
                let d_lo = -m_lo / n_lo;
                if tie > 0.0 {
                    // The pull fades over the last quarter before the spreads are equal, and with
                    // the spreads themselves (eps), and ramps in over one count above 6.
                    let eps = 2.0 * tl * inv_n * inv_n;
                    let pull = |near: f64, far: f64, n: f64| -> f64 {
                        if far > near {
                            let fade = ((far - near) / (0.25 * far)).min(1.0);
                            (n - 6.0).clamp(0.0, 1.0) * (fade * far / (near + far * inv_n + eps))
                        } else {
                            0.0
                        }
                    };
                    let t = pull(d_hi, d_lo, n_hi);
                    if t > 0.0 {
                        v_hi += max_l * t;
                        w_hi += t;
                    }
                    let t = pull(d_lo, d_hi, n_lo);
                    if t > 0.0 {
                        v_lo += min_l * t;
                        w_lo += t;
                    }
                } else {
                    if n_hi > 6.0 && d_hi < d_lo {
                        let t = d_lo / (d_hi + d_lo * inv_n);
                        v_hi += max_l * t;
                        w_hi += t;
                    }
                    if n_lo > 6.0 && d_lo < d_hi {
                        let t = d_hi / (d_lo + d_hi * inv_n);
                        v_lo += min_l * t;
                        w_lo += t;
                    }
                }
                let i = y * w + x;
                out[0][i] = v_hi as f32;
                out[1][i] = w_hi as f32;
                out[2][i] = v_lo as f32;
                out[3][i] = w_lo as f32;
            }
        }
        out
    }

    #[test]
    fn halo_window_pass_has_the_bits_of_the_branching_loops() {
        let sizes = [
            (1, 1),
            (2, 1),
            (3, 5),
            (7, 5),
            (1, 9),
            (16, 16),
            (33, 20),
            (130, 7),
        ];
        for (w, h) in sizes {
            for (noise, seed) in [(0.0f32, 1u64), (0.004, 2), (0.05, 3)] {
                let l = plane(w, h, noise, seed + (w * h) as u64);
                // S = L + a halo-limited delta that is zero in places (exact ties), a few ulps
                // in others (ties within the tolerance), positive and negative elsewhere.
                let s: Vec<f32> = l
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match i % 5 {
                        0 => v,
                        1 => v + 0.01,
                        2 => v - 0.02,
                        3 => v * (1.0 + 3.0 * f32::EPSILON),
                        _ => v + 0.0007 * (i % 11) as f32,
                    })
                    .collect();
                for tie in [0.0, HALO_TIE_TOLERANCE, HALO_TIE_TOLERANCE_MAX] {
                    let fast = halo_maps(&l, &s, w, h, tie);
                    let slow = halo_maps_pixelwise(&l, &s, w, h, tie);
                    for (k, (a, b)) in fast.iter().zip(&slow).enumerate() {
                        assert!(bits_equal(a, b), "{w}x{h} noise {noise} tie {tie} map {k}");
                    }
                    // And with S = L exactly (no halo-limited delta anywhere).
                    let fast = halo_maps(&l, &l, w, h, tie);
                    let slow = halo_maps_pixelwise(&l, &l, w, h, tie);
                    for (a, b) in fast.iter().zip(&slow) {
                        assert!(bits_equal(a, b), "{w}x{h} noise {noise} tie {tie}, S = L");
                    }
                }
            }
        }
    }

    fn test_plans() -> Vec<(OutputSharpen, f32)> {
        let mut v = vec![
            (OutputSharpen::screen(), 1.0),
            (OutputSharpen::print(), 1.0),
        ];
        let mut m2 = OutputSharpen::screen();
        m2.method = Method::GradientRamp;
        m2.radius_px = 2.5;
        v.push((m2, 0.6));
        let mut m0 = OutputSharpen::screen();
        m0.method = Method::HardThreshold;
        m0.halo_pct = 0.0;
        m0.threshold_levels = 3.0;
        v.push((m0, 1.0));
        let mut e = OutputSharpen::screen();
        e.extra_stages.push(ExtraStage {
            method: Method::HardThreshold,
            auto_method: true,
            amount_pct: 150.0,
            radius_px: 0.6,
            threshold_levels: 1.0,
            halo_pct: 20.0,
        });
        v.push((e, 1.0));
        v
    }

    #[test]
    fn the_band_height_does_not_change_the_result() {
        // Bands of any height (down to one row) give the whole-frame bits: the tiling contract.
        let (w, h) = (53, 47);
        let luma = plane(w, h, 0.01, 7);
        for (p, scale) in test_plans() {
            let plan = p.plan(scale).expect("plan");
            let whole = ratio_plane(&plan, &luma, w, h, h);
            for band_rows in [1, 2, 3, 5, 8, 17, 46, 47, 48, 500] {
                let banded = ratio_plane(&plan, &luma, w, h, band_rows);
                assert!(
                    bits_equal(&whole, &banded),
                    "{:?} scale {scale}: bands of {band_rows} rows differ",
                    p.method
                );
            }
        }
    }

    #[test]
    fn min_and_max_follow_the_reference() {
        // std::max(a, b) is `a < b ? b : a`, std::min(a, b) is `b < a ? b : a`: with a NaN the
        // first argument is returned.
        assert!(fmax(f32::NAN, 0.0).is_nan());
        assert_eq!(fmax(0.0, f32::NAN), 0.0);
        assert!(fmin(f32::NAN, 1.0).is_nan());
        assert_eq!(fmin(1.0, f32::NAN), 1.0);
        assert_eq!((fmax(1.0, 2.0), fmax(2.0, 1.0)), (2.0, 2.0));
        assert_eq!((fmin(1.0, 2.0), fmin(2.0, 1.0)), (1.0, 1.0));
    }

    #[test]
    fn derived_scalars_follow_the_spec() {
        // Sigma map and its step at R = 0.5.
        assert_eq!(sigma_from_radius(0.09), 0.0);
        assert!((sigma_from_radius(0.4999) - 0.621).abs() < 2e-3);
        assert!((sigma_from_radius(0.5) - 0.650).abs() < 1e-6);
        // Tap rule.
        assert_eq!(tap_count(0.86, 13), 7);
        assert_eq!(tap_count(2.05, 13), 13);
        assert_eq!(tap_count(4.1, 13), 13);
        assert_eq!(tap_count(4.1, 33), 23);
        assert_eq!(tap_count(0.01, 13), 3);
        // Gain kernel sigma: max(2 sigma, 1).
        assert_eq!(gain_sigma(0.4), 1.0);
        assert_eq!(gain_sigma(0.5), 1.0);
        assert_eq!(gain_sigma(0.9), 1.8);
        // Scaling leaves sigma below the knee and at s = 1 alone.
        assert_eq!(scale_sigma(0.1, 0.3), 0.1);
        assert_eq!(scale_sigma(1.7, 1.0), 1.7);
        assert!(scale_sigma(1.7, 0.5) < 1.7 && scale_sigma(1.7, 0.5) > 0.15);
        // Halo fraction and dead zone.
        assert!((halo_fraction(60.0) - 0.774_596_7).abs() < 1e-7);
        assert_eq!(halo_fraction(0.00005), 0.0);
        assert_eq!(halo_fraction(100.0), 1.0);
        assert_eq!(halo_fraction(250.0), 1.0);
        // The luma has the reference's operation order: ((wR R) R + (wG G) G) + (wB B) B.
        let w = REFERENCE_LUMA_WEIGHTS;
        let (r, g, b) = (0.3f32, 0.5f32, 0.7f32);
        let want = (((w[0] * r) * r + (w[1] * g) * g) + (w[2] * b) * b).sqrt();
        assert_eq!(luma_of(&w, r, g, b), want);
        assert_eq!(luma_of(&w, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn black_stays_black_and_the_ratio_is_capped() {
        // A black pixel in a bright field: the luma floor keeps the ratio finite, and the output
        // is 0 * ratio = 0.
        let mut px = vec![0.8f32; 3 * 9 * 9];
        let centre = 3 * (4 * 9 + 4);
        px[centre..centre + 3].fill(0.0);
        let out = OutputSharpen::screen()
            .with_amount_pct(1000.0)
            .apply_interleaved(9, 9, &px, 1.0)
            .expect("sharpen");
        assert_eq!(&out[centre..centre + 3], &[0.0, 0.0, 0.0]);
        assert!(out.iter().all(|v| v.is_finite()));
        // Ratios stay in [0, 16] on a hard-edged pattern at the largest amount.
        let luma = plane(40, 40, 0.5, 4)
            .iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>();
        let plan = OutputSharpen::print()
            .with_amount_pct(1000.0)
            .plan(1.0)
            .expect("plan");
        let ratio = ratio_plane(&plan, &luma, 40, 40, BAND_ROWS);
        assert!(ratio.iter().all(|r| (0.0..=RATIO_MAX).contains(r)));
    }
}
