//! The highlight range and the low-light toe (stage C7) (`docs/02-colour-pipeline.md` section C7).
//!
//! Two gains that keep the channel ratios of a linear AWG4 pixel: the **highlight range** acts on its
//! luma above a knee, the **low-light toe** on its norm `n = max(R, G, B)` near black. The range comes
//! first, then the toe acts on the gained pixel: `apply = toe(range(rgb))`. They never act on the same
//! pixels: the range is exactly 1 below +1 stop over middle grey, the toe acts within a few noise
//! floors of black.
//!
//! # The highlight range ([`HighlightRange`])
//!
//! An ALEXA (ALEV III) at EI 800 clips 7.64 stops over middle grey; a Sony body at the camera's
//! exposure clips about 3.6 stops over it. Below the Sony's clip the chain already holds what an
//! ALEXA exposed for the same grey would record, so the one difference is where the clip lands. The
//! range maps the Sony's top two stops onto the ALEXA's: nothing changes below a knee two stops under
//! the Sony clip, the Sony clip lands on a target between where it was recorded (`r = 0`) and the
//! ALEXA's EI 800 clip (`r = 1`), and nothing goes past the ALEXA's clip.
//!
//! ```text
//! Y    = 0.2126 R + 0.7152 G + 0.0722 B          luma of the linear AWG4 pixel ([`luma`])
//! s    = log2(Y / 0.18)                          stops over middle grey
//! h_S  = log2(Y_clip / 0.18)                     the Sony clip: Y_clip = ClipLevel::luma()
//! h_A  = log2((1 - 256/65535) * 800 / 4)         the ALEXA clip at EI 800 (HIGHLIGHT_CAP_STOPS)
//! s_k  = max(1, h_S - 2)                         the knee
//! w    = clamp((h_S - s_k - 0.25) / 1.75, 0, 1)  the fade at low clip levels
//! E    = w * r * max(h_A - h_S, 0)               how far the Sony clip moves, in stops
//! u    = (s - s_k) / (h_S - s_k)                 only where E > 0 (then h_S - s_k >= 0.25)
//! phi  = u^2 (u <= 1), 2u - 1 (u > 1)
//! p    = (max(R, G, B) - min(R, G, B)) / max(R, G, B)   purity of the pixel ([`purity`]), 0 for a grey
//! g    = 1 - smoothstep(0.6, 0.9, p)             the achromatic gate ([`AchromaticGate`]), 1 up to p = 0.6
//! s'   = s + g * E * phi  where s > s_k and E > 0, else s
//! s'   = min(s', h_A)                            the ALEXA's hard clip, after the gate
//! gain = 2^(s' - s); rgb' = gain * rgb
//! ```
//!
//! `r = rolloff.highlights` (0 to 1, default 0.6). `E`, `s_k` and `h_S` are computed once per chain
//! ([`ToneParams::range`]). The gain is exactly 1 for `s <= s_k` (and below the cap), and then the pixel
//! is returned unchanged, bit for bit. `s'` is continuous with a continuous slope at the knee and at
//! `u = 1`, non-decreasing in `s`, and the Sony clip lands on `min(h_S + E, h_A)`. All three channels are
//! multiplied by the same gain, so a pixel's colour ratios are kept exactly. The gate is 1 for a grey
//! pixel (`p = 0`) and for every pixel up to `p = 0.6`, so those get the expansion exactly as before,
//! bit for bit; a pixel of purity `0.9` or more gets none and keeps its scene exposure, only the
//! ALEXA's clip still acts on it. Purity does not change when a pixel is scaled, so along an exposure ramp
//! of one colour the gate is constant and the output stays non-decreasing. Across colours the gate trades
//! against the luma weights: because the weights are positive, every output channel is non-decreasing in
//! each non-negative input channel below the cap **only when the gate is 1** (a grey pixel); a channel
//! that raises the purity of a pixel can lower its gain by more than the channel rises.
//!
//! The range moves where the camera's clip lands, on the scene side before LogC4; the display transform
//! (the cine rendering, C12) reaches white by itself at +10.4 stops over grey.
//!
//! # The low-light toe
//!
//! `rolloff.low_light = r_l` (0 to 1, default 0.5). `tau = 4 r_l sigma_floor` and
//! `f(n) = n^3 / (n^2 + tau^2)` on the norm of the gained pixel, applied as the scale
//! `n^2 / (n^2 + tau^2)` to all three channels: monotonic, close to `n` for `n >> tau`, and crushing the
//! noise floor smoothly below `tau`. `r_l = 0` disables it exactly. For `n <= 0` (all channels at or
//! below zero) the scale still applies, so a pixel with `n = 0` and `tau > 0` maps to black.
//!
//! # Edge values
//!
//! A pixel with a non-finite channel passes through unchanged. `Y <= 0` gets a range gain of 1.
//! [`ToneParams::new`] refuses a clip level that is not a positive finite number; only
//! [`ToneParams::IDENTITY`], which turns the whole stage off, carries an infinite one.

use crate::consts::{REC709_LUMA_WEIGHTS, Vec3};
use crate::gamut::LinearAwg4;

/// Scene-linear middle grey, the reference of the range's stops (`s = log2(Y / 0.18)`).
pub const HIGHLIGHT_GREY: f64 = 0.18;

/// The exposure index of the ALEXA clip the range reaches at `r = 1`: 800, ARRI's base EI.
pub const ALEXA_CLIP_EI: f64 = 800.0;

/// `log2((1 - 256 / 65535) * ei / 4)`: stops over 0.18 at which an ALEV III clips (ARRI's LogC3 formula).
///
/// Grey sits at `0.01 * 400 / ei` of the sensor's range over black and the signal is linear up to the
/// 16-bit white, `1 - 256 / 65535`, so the clip is `(1 - 256 / 65535) * ei / 4` times grey: at EI 800
/// 7.638 stops, scene-linear 35.86 (the LogC3 clip code 0.9539).
#[must_use]
pub fn alexa_clip_stops(ei: f64) -> f64 {
    ((1.0 - 256.0 / 65535.0) * ei / 4.0).log2()
}

/// `h_A`: the ALEXA (ALEV III) clip at EI 800 in stops over middle grey, [`alexa_clip_stops`] of
/// [`ALEXA_CLIP_EI`]. Nothing leaves the range above it.
pub const HIGHLIGHT_CAP_STOPS: f64 = 7.638_209_540_303_716;

/// How far the knee sits under the Sony clip, in stops. One stop until pipeline 0.8.0; two since 0.8.1
/// (the owner's "gentler range", 2026-10-30: with one stop and the full range the last stop before the
/// clip was stretched about 3.5 times, which turned the texture of bright skin into a hard edge, then a
/// flat white).
pub const HIGHLIGHT_KNEE_DEPTH_STOPS: f64 = 2.0;

/// The lowest the knee goes, in stops over middle grey: mid grey and everything under +1 stop never move.
pub const HIGHLIGHT_KNEE_FLOOR_STOPS: f64 = 1.0;

/// The span from the knee to the Sony clip under which the range has faded out completely (stops). The
/// fade `w` rises from 0 at this span to 1 at [`HIGHLIGHT_KNEE_DEPTH_STOPS`].
pub const HIGHLIGHT_MIN_SPAN_STOPS: f64 = 0.25;

/// Large finite value meaning "no knee / no cap" (finite so it is safe under Metal fast-math, K8).
pub const HIGHLIGHT_OFF_STOPS: f64 = 1e30;

/// Why a [`ToneParams`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum ToneError {
    /// The clip level must be finite and greater than zero.
    #[error("clip level must be finite and greater than zero, got {0}")]
    InvalidClipLevel(f64),
    /// The noise floor must be finite and not negative.
    #[error("noise floor must be finite and not negative, got {0}")]
    InvalidNoiseFloor(f64),
    /// The achromatic gate needs finite edges with `0 <= from < to <= 3` and `to - from >= MIN_GATE_SPAN`.
    #[error(
        "achromatic gate edges must be finite with 0 <= from < to <= 3 and at least 1e-6 apart, got {from} and {to}"
    )]
    InvalidGate {
        /// The purity at which the gate starts to close.
        from: f64,
        /// The purity from which it is closed.
        to: f64,
    },
}

/// The purity at which the achromatic gate starts to close (owner decision 2026-10-29).
pub const ACHROMATIC_GATE_FROM: f64 = 0.6;

/// The purity from which the achromatic gate is closed (owner decision 2026-10-29).
pub const ACHROMATIC_GATE_TO: f64 = 0.9;

/// The smallest distance between the two edges of an [`AchromaticGate`]. The kernel multiplies by the inverse
/// of the span, which must stay finite in f32 (a span near 1e-39 overflows it, and `0 * infinity` at the `from`
/// edge is a NaN); 1e-6 is far below any gate that means something (the owner's is 0.3 wide).
pub const MIN_GATE_SPAN: f64 = 1e-6;

/// The largest purity the gate looks at: a pixel with a negative channel has `p >= 1`, and a pixel with a
/// tiny positive maximum can have an enormous one. Purity is held to this (finite for Metal's fast-math, K8).
pub const PURITY_MAX: f64 = 2.0;

/// Purity of a linear AWG4 pixel, `p = (max(R, G, B) - min(R, G, B)) / max(R, G, B)`, held to
/// [`PURITY_MAX`]: 0 for a grey pixel (`R == G == B`, exactly), 1 for a pixel whose smallest channel is 0,
/// above 1 when a channel is negative. Scaling a pixel by a positive number does not change it.
///
/// A pixel whose largest channel is not above 0 (or not a number) gets 0: its luma is not above 0
/// either, and every stage that uses the gate is the identity there.
#[must_use]
pub fn purity(rgb: Vec3) -> f64 {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    if max.is_nan() || max <= 0.0 {
        return 0.0;
    }
    ((max - min) / max).min(PURITY_MAX)
}

/// The achromatic gate: a weight `g = 1 - smoothstep(from, to, p)` of a pixel's [`purity`], 1 for a grey
/// pixel and for every purity up to `from`, 0 from `to`. The highlight range of C7 multiplies its
/// expansion in stops by it (so did the white lift, until the cine rendering replaced it), so near-neutral highlights expand as designed and
/// high-purity colours (saturated reds) stay near their scene exposure (owner decision 2026-10-29,
/// `docs/02`).
///
/// The edges are numbers to tune: [`AchromaticGate::DEFAULT`] is `0.6` and `0.9`;
/// [`AchromaticGate::OFF`] is a gate that is always 1 (the stages as they were before the gate, for tests
/// and for comparisons).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AchromaticGate {
    from: f64,
    to: f64,
}

impl Default for AchromaticGate {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl AchromaticGate {
    /// The owner's gate, [`ACHROMATIC_GATE_FROM`] to [`ACHROMATIC_GATE_TO`].
    pub const DEFAULT: Self = Self {
        from: ACHROMATIC_GATE_FROM,
        to: ACHROMATIC_GATE_TO,
    };

    /// A gate that is 1 for every pixel: its edges lie above [`PURITY_MAX`].
    pub const OFF: Self = Self {
        from: PURITY_MAX,
        to: PURITY_MAX + 1.0,
    };

    /// A gate that closes between the purities `from` and `to`.
    ///
    /// # Errors
    ///
    /// [`ToneError::InvalidGate`] unless both are finite with `0 <= from < to <= 3` and `to - from` is at
    /// least [`MIN_GATE_SPAN`].
    pub fn new(from: f64, to: f64) -> Result<Self, ToneError> {
        if from.is_finite()
            && to.is_finite()
            && 0.0 <= from
            && from < to
            && to <= PURITY_MAX + 1.0
            && to - from >= MIN_GATE_SPAN
        {
            Ok(Self { from, to })
        } else {
            Err(ToneError::InvalidGate { from, to })
        }
    }

    /// The purity at which the gate starts to close.
    #[must_use]
    pub fn from(&self) -> f64 {
        self.from
    }

    /// The purity from which the gate is closed.
    #[must_use]
    pub fn to(&self) -> f64 {
        self.to
    }

    /// `1 / (to - from)`, what the GPU kernel multiplies by.
    #[must_use]
    pub fn inv_span(&self) -> f64 {
        1.0 / (self.to - self.from)
    }

    /// Whether the gate is 1 for every pixel ([`AchromaticGate::OFF`]).
    #[must_use]
    pub fn is_off(&self) -> bool {
        self.from >= PURITY_MAX
    }

    /// The weight for a purity `p`: `1 - smoothstep(from, to, p)`. Exactly 1 for `p <= from`.
    #[must_use]
    pub fn weight_of_purity(&self, p: f64) -> f64 {
        let t = ((p - self.from) / (self.to - self.from)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// The weight of a linear AWG4 pixel, [`AchromaticGate::weight_of_purity`] of its [`purity`]. Exactly
    /// 1.0 for a grey pixel.
    #[must_use]
    pub fn weight(&self, rgb: Vec3) -> f64 {
        self.weight_of_purity(purity(rgb))
    }
}

/// The luma of a linear AWG4 triple with the Rec.709 weights ([`REC709_LUMA_WEIGHTS`]): the norm the
/// highlight range reads. The weights are all positive, so raising a channel never lowers it.
#[must_use]
pub fn luma(rgb: Vec3) -> f64 {
    REC709_LUMA_WEIGHTS[0] * rgb[0]
        + REC709_LUMA_WEIGHTS[1] * rgb[1]
        + REC709_LUMA_WEIGHTS[2] * rgb[2]
}

/// Stops over [`HIGHLIGHT_GREY`] of a positive luma, as `log2(y) - log2(0.18)` so that no luma, however
/// large, overflows on the way.
fn stops_over_grey(y: f64) -> f64 {
    y.log2() - HIGHLIGHT_GREY.log2()
}

/// The highlight range of one chain, prepared: the four numbers of the formula (module docs), in stops
/// over middle grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighlightRange {
    /// `s_k`: below it (and at it) the gain is exactly 1.
    pub knee_stops: f64,
    /// `h_S`: the Sony clip, where the clipped neutral's luma lies.
    pub clip_stops: f64,
    /// `E`: how far the Sony clip moves (it lands on `h_S + E`, never past the cap). 0 turns the
    /// expansion off and leaves only the cap.
    pub expansion_stops: f64,
    /// `h_A`: the ALEXA clip; no output luma lies above it.
    pub cap_stops: f64,
}

impl HighlightRange {
    /// No range and no cap: the identity. Knee, clip and cap are [`HIGHLIGHT_OFF_STOPS`], the expansion 0.
    pub const NONE: HighlightRange = HighlightRange {
        knee_stops: HIGHLIGHT_OFF_STOPS,
        clip_stops: HIGHLIGHT_OFF_STOPS,
        expansion_stops: 0.0,
        cap_stops: HIGHLIGHT_OFF_STOPS,
    };

    /// The range for a highlight amount `r` (clamped to `[0, 1]`, NaN counts as 0) and the luma of the
    /// clipped neutral after C1 to C6 (`ClipLevel::luma`). A clip luma that is not a positive finite
    /// number gives [`HighlightRange::NONE`].
    #[must_use]
    pub fn new(highlights: f64, clip_luma: f64) -> Self {
        if !clip_luma.is_finite() || clip_luma <= 0.0 {
            return Self::NONE;
        }
        let clip = stops_over_grey(clip_luma);
        let knee = HIGHLIGHT_KNEE_FLOOR_STOPS.max(clip - HIGHLIGHT_KNEE_DEPTH_STOPS);
        let fade = ((clip - knee - HIGHLIGHT_MIN_SPAN_STOPS)
            / (HIGHLIGHT_KNEE_DEPTH_STOPS - HIGHLIGHT_MIN_SPAN_STOPS))
            .clamp(0.0, 1.0);
        let expansion = fade * clamp_unit(highlights) * (HIGHLIGHT_CAP_STOPS - clip).max(0.0);
        Self {
            knee_stops: knee,
            clip_stops: clip,
            expansion_stops: expansion,
            cap_stops: HIGHLIGHT_CAP_STOPS,
        }
    }

    /// `s'`: where a tone `s` stops over middle grey goes for a grey pixel (gate 1): `s` itself up to the
    /// knee, the expansion above it, never above the cap. NaN stays NaN.
    #[must_use]
    pub fn stops(&self, s: f64) -> f64 {
        self.stops_gated(s, 1.0)
    }

    /// As [`HighlightRange::stops`] with the expansion weighted by the achromatic gate `weight` (0 to 1,
    /// [`AchromaticGate::weight`]): `s + weight * E * phi` above the knee, then the cap. A weight of 1 is
    /// [`HighlightRange::stops`] to the last bit.
    #[must_use]
    pub fn stops_gated(&self, s: f64, weight: f64) -> f64 {
        let mut out = s;
        if self.expansion_stops > 0.0 && s > self.knee_stops {
            // E > 0 only when the span is at least HIGHLIGHT_MIN_SPAN_STOPS, so the division is safe.
            let u = (s - self.knee_stops) / (self.clip_stops - self.knee_stops);
            let phi = if u <= 1.0 { u * u } else { 2.0 * u - 1.0 };
            out = s + (weight * self.expansion_stops) * phi;
        }
        if out > self.cap_stops {
            self.cap_stops
        } else {
            out
        }
    }

    /// The gain `2^(s' - s)` for a pixel of luma `luma` and gate 1 (a grey pixel): exactly 1.0 where
    /// `s' == s`, and 1.0 for a luma at or below zero or not finite. A coloured pixel is
    /// [`HighlightRange::gain_for`].
    #[must_use]
    pub fn gain(&self, luma: f64) -> f64 {
        self.gain_with(luma, || 1.0)
    }

    /// The gain of a linear AWG4 pixel: the luma's gain with the expansion weighted by `gate` of the
    /// pixel's purity. A grey pixel gets [`HighlightRange::gain`] of its luma, bit for bit.
    #[must_use]
    pub fn gain_for(&self, rgb: Vec3, gate: &AchromaticGate) -> f64 {
        self.gain_with(luma(rgb), || gate.weight(rgb))
    }

    /// The gain for a luma, with the gate's weight asked for only where the expansion acts (above the
    /// knee), so the purity is never computed for the pixels that are returned unchanged.
    fn gain_with(&self, luma: f64, weight: impl FnOnce() -> f64) -> f64 {
        if !luma.is_finite() || luma <= 0.0 {
            return 1.0;
        }
        let s = stops_over_grey(luma);
        let w = if self.expansion_stops > 0.0 && s > self.knee_stops {
            weight()
        } else {
            1.0
        };
        let t = self.stops_gated(s, w);
        if t == s { 1.0 } else { (t - s).exp2() }
    }

    /// Where the Sony clip lands: `min(h_S + E, h_A)` stops over middle grey.
    #[must_use]
    pub fn target_stops(&self) -> f64 {
        self.stops(self.clip_stops)
    }
}

/// Parameters of stage C7.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneParams {
    /// `rolloff.highlights`, `r`: the highlight range, clamped to `[0, 1]`. 0 keeps the clip where it
    /// was recorded, 1 takes it to the ALEXA's EI 800 clip (the default setting is 0.6).
    pub highlights: f64,
    /// `rolloff.low_light`, `r_l`, clamped to `[0, 1]`. Zero disables the toe.
    pub low_light: f64,
    /// `Y_clip`: the luma of the clipped neutral after C1 to C6 (`ClipLevel::luma`), exposure included.
    /// Infinite only in [`ToneParams::IDENTITY`].
    pub clip_level: f64,
    /// `sigma_floor`: read noise of the noise model mapped to linear AWG4 at this exposure.
    pub noise_floor: f64,
    /// The achromatic gate on the highlight range's expansion ([`AchromaticGate::DEFAULT`] unless a test
    /// or a tool sets another with [`ToneParams::with_gate`]).
    pub gate: AchromaticGate,
}

impl ToneParams {
    /// C7 off entirely (no range, no cap, no toe): for tests and tools that need the identity.
    /// `clip_level` is `f64::INFINITY` here and only here; [`ToneParams::new`] still refuses a
    /// non-finite clip level.
    pub const IDENTITY: ToneParams = ToneParams {
        highlights: 0.0,
        low_light: 0.0,
        clip_level: f64::INFINITY,
        noise_floor: 0.0,
        gate: AchromaticGate::DEFAULT,
    };

    /// Builds parameters, clamping the two amounts to `[0, 1]` (NaN counts as 0).
    ///
    /// `clip_level` is the luma of the clipped neutral (`ClipLevel::luma`).
    ///
    /// # Errors
    ///
    /// [`ToneError::InvalidClipLevel`] unless `clip_level` is finite and positive;
    /// [`ToneError::InvalidNoiseFloor`] unless `noise_floor` is finite and not negative.
    pub fn new(
        highlights: f64,
        low_light: f64,
        clip_level: f64,
        noise_floor: f64,
    ) -> Result<Self, ToneError> {
        if !clip_level.is_finite() || clip_level <= 0.0 {
            return Err(ToneError::InvalidClipLevel(clip_level));
        }
        if !noise_floor.is_finite() || noise_floor < 0.0 {
            return Err(ToneError::InvalidNoiseFloor(noise_floor));
        }
        Ok(Self {
            highlights: clamp_unit(highlights),
            low_light: clamp_unit(low_light),
            clip_level,
            noise_floor,
            gate: AchromaticGate::DEFAULT,
        })
    }

    /// These parameters with another achromatic gate ([`AchromaticGate::OFF`] is the highlight range as it
    /// was before the gate).
    #[must_use]
    pub fn with_gate(mut self, gate: AchromaticGate) -> Self {
        self.gate = gate;
        self
    }

    /// The prepared highlight range: [`HighlightRange::new`] of the highlight amount and the clip
    /// level, [`HighlightRange::NONE`] when the clip level is infinite (or otherwise unusable).
    #[must_use]
    pub fn range(&self) -> HighlightRange {
        HighlightRange::new(self.highlights, self.clip_level)
    }

    /// The highlight range's gain for a linear AWG4 triple: `range().gain_for(rgb, &gate)`, the luma's
    /// gain with the expansion weighted by the achromatic gate.
    #[must_use]
    pub fn highlight_gain(&self, rgb: Vec3) -> f64 {
        self.range().gain_for(rgb, &self.gate)
    }

    /// The toe width `tau = 4 r_l sigma_floor`.
    #[must_use]
    pub fn toe_tau(&self) -> f64 {
        4.0 * clamp_unit(self.low_light) * self.noise_floor
    }

    /// The low-light toe alone: `n^3 / (n^2 + tau^2)`, or `n` when `tau` is zero.
    #[must_use]
    pub fn toe(&self, n: f64) -> f64 {
        n * toe_scale(n, self.toe_tau())
    }

    /// Applies stage C7 to a linear AWG4 pixel: `toe(range(rgb))`.
    #[must_use]
    pub fn apply(&self, rgb: LinearAwg4) -> LinearAwg4 {
        LinearAwg4(self.apply_rgb(rgb.0))
    }

    /// Applies stage C7 to a raw linear AWG4 triple: `toe(range(rgb))`.
    #[must_use]
    pub fn apply_rgb(&self, rgb: Vec3) -> Vec3 {
        self.prepare().apply_rgb(rgb)
    }

    /// The per-chain numbers of the stage, computed once.
    pub(crate) fn prepare(&self) -> PreparedTone {
        PreparedTone {
            range: self.range(),
            gate: self.gate,
            tau: self.toe_tau(),
        }
    }
}

/// C7 with its per-chain numbers computed ([`ToneParams::prepare`]): what a chain applies per pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PreparedTone {
    range: HighlightRange,
    gate: AchromaticGate,
    tau: f64,
}

impl PreparedTone {
    /// `toe(range(rgb))`; a pixel with a non-finite channel passes through.
    pub(crate) fn apply_rgb(&self, rgb: Vec3) -> Vec3 {
        if rgb.iter().any(|c| !c.is_finite()) {
            return rgb;
        }
        let gain = self.range.gain_for(rgb, &self.gate);
        let ranged = if gain == 1.0 {
            rgb
        } else {
            rgb.map(|c| c * gain)
        };
        if self.tau == 0.0 {
            return ranged;
        }
        let n = ranged[0].max(ranged[1]).max(ranged[2]);
        let toe = toe_scale(n, self.tau);
        ranged.map(|c| c * toe)
    }
}

/// `m^2 / (m^2 + tau^2)`, the factor the toe applies to `m`, written as `1 / (1 + (tau / m)^2)` so
/// that nothing underflows to `0 / 0` when both `m` and `tau` are tiny.
fn toe_scale(m: f64, tau: f64) -> f64 {
    if tau == 0.0 {
        1.0
    } else if m == 0.0 {
        0.0
    } else {
        let ratio = tau / m;
        1.0 / (1.0 + ratio * ratio)
    }
}

fn clamp_unit(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_parameters() {
        assert!(matches!(
            ToneParams::new(0.5, 0.5, 0.0, 0.0),
            Err(ToneError::InvalidClipLevel(_))
        ));
        assert!(matches!(
            ToneParams::new(0.5, 0.5, f64::NAN, 0.0),
            Err(ToneError::InvalidClipLevel(_))
        ));
        assert!(matches!(
            ToneParams::new(0.5, 0.5, f64::INFINITY, 0.0),
            Err(ToneError::InvalidClipLevel(_))
        ));
        assert!(matches!(
            ToneParams::new(0.5, 0.5, 1.0, -1.0),
            Err(ToneError::InvalidNoiseFloor(_))
        ));
        let p = ToneParams::new(2.0, -1.0, 1.0, 0.0).unwrap();
        assert_eq!((p.highlights, p.low_light), (1.0, 0.0));
    }

    #[test]
    fn the_cap_is_the_alexa_clip_at_ei_800() {
        assert!((alexa_clip_stops(ALEXA_CLIP_EI) - HIGHLIGHT_CAP_STOPS).abs() <= 1e-15);
    }

    #[test]
    fn the_identity_has_no_range() {
        assert_eq!(ToneParams::IDENTITY.range(), HighlightRange::NONE);
        assert_eq!(ToneParams::IDENTITY.toe_tau(), 0.0);
    }
}
