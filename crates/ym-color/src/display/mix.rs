//! The mixers on the CPU reference (`docs/25-mixers.md` section 3): what a Parallel, a Layer, a Splitter and a
//! Combiner compute on one pixel, in f64, with no `fma` and a fixed order, so that the result is the same bits on every
//! machine and for any number of threads.
//!
//! The functions are pure and work on LogC4 triples (`[f64; 3]`); [`ColorChain`](super::ColorChain) calls them for the
//! plan's join steps. A value is read as [`Source`]: the LogC4 triple and, for the Input's picture, the linear values it
//! was encoded from (C7), which a Layer takes without a decode.

use crate::consts::Vec3;
use crate::logc::{logc4_decode_rgb, logc4_encode_rgb};

/// How a Layer blends an input over the ones below it (`docs/25` section 3.3). The formulas are on linear light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    /// The layer covers what is below: `b`.
    #[default]
    Normal,
    /// `a + b`.
    Add,
    /// The layer taken from what is below: `a - b`.
    Subtract,
    /// `a * b`.
    Multiply,
    /// `max(a, b)`.
    Lighten,
    /// `min(a, b)`.
    Darken,
    /// `|a - b|`.
    Difference,
}

impl BlendMode {
    /// Every mode, in the order of the Mode menu.
    pub const ALL: [BlendMode; 7] = [
        Self::Normal,
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Lighten,
        Self::Darken,
        Self::Difference,
    ];

    /// The mode on one channel: `a` is the stack below and `b` the layer, both linear. No clamp: a negative value is
    /// kept (`docs/25` section 3.3, *Negative linear values*).
    #[must_use]
    pub fn channel(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Normal => b,
            Self::Add => a + b,
            Self::Subtract => a - b,
            Self::Multiply => a * b,
            Self::Lighten => a.max(b),
            Self::Darken => a.min(b),
            Self::Difference => (a - b).abs(),
        }
    }

    /// The mode on a linear triple.
    #[must_use]
    pub fn blend(self, a: Vec3, b: Vec3) -> Vec3 {
        [
            self.channel(a[0], b[0]),
            self.channel(a[1], b[1]),
            self.channel(a[2], b[2]),
        ]
    }
}

/// What the settings give a mixer in the chain: whether it is bypassed and, for a Layer, its mode and opacity. A
/// Splitter and a Combiner have no parameters ([`JoinParams::default`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JoinParams {
    /// A bypassed Parallel outputs its base and a bypassed Layer its first input.
    pub bypass: bool,
    /// A Layer's mode.
    pub mode: BlendMode,
    /// A Layer's opacity, 0 to 1. The default (0) is not a Layer's: set it.
    pub opacity: f64,
}

impl JoinParams {
    /// A Parallel.
    #[must_use]
    pub fn parallel(bypass: bool) -> Self {
        Self {
            bypass,
            mode: BlendMode::Normal,
            opacity: 1.0,
        }
    }

    /// A Layer.
    #[must_use]
    pub fn layer(bypass: bool, mode: BlendMode, opacity: f64) -> Self {
        Self {
            bypass,
            mode,
            opacity,
        }
    }
}

/// A picture value as a join reads it.
#[derive(Debug, Clone, Copy)]
pub struct Source {
    /// The LogC4 triple.
    pub logc4: Vec3,
    /// The linear AWG4 values it was encoded from, when they are known exactly (the Input's picture).
    pub linear: Option<Vec3>,
}

impl Source {
    fn linear(&self) -> Vec3 {
        self.linear.unwrap_or_else(|| logc4_decode_rgb(self.logc4))
    }
}

/// A Parallel (`docs/25` section 3.2): `base + sum of (input - base)` per channel on LogC4 code values, in port order,
/// skipping the inputs for which `is_base` is true (an input that is the base adds nothing, whatever the floats say).
/// Nothing is clamped.
#[must_use]
pub fn parallel(base: Vec3, inputs: &[Vec3], is_base: impl Fn(usize) -> bool) -> Vec3 {
    let mut acc = base;
    for (i, v) in inputs.iter().enumerate() {
        if is_base(i) {
            continue;
        }
        acc = [
            acc[0] + (v[0] - base[0]),
            acc[1] + (v[1] - base[1]),
            acc[2] + (v[2] - base[2]),
        ];
    }
    acc
}

/// A Layer (`docs/25` section 3.3). `inputs[0]` is the opaque bottom; `alphas[i]` for `i >= 1` is the key `m` of the
/// node that feeds input `i` (`None`: the opacity alone). Layer `i` has alpha `m * opacity`. Composited from the bottom up in
/// linear light with the W3C rule for a source over an opaque backdrop, `acc + alpha * (B(acc, b) - acc)`; alpha 0 skips
/// the layer and alpha 1 takes the blend itself. When no layer shows, the bottom input is returned as it is, with no
/// decode and encode.
#[must_use]
pub fn layer(mode: BlendMode, opacity: f64, inputs: &[Source], alphas: &[Option<f64>]) -> Vec3 {
    let Some(bottom) = inputs.first() else {
        return [0.0; 3];
    };
    let mut acc: Option<Vec3> = None;
    for (i, top) in inputs.iter().enumerate().skip(1) {
        let alpha = match alphas.get(i).copied().flatten() {
            Some(m) => m * opacity,
            None => opacity,
        };
        if alpha == 0.0 {
            continue;
        }
        let below = acc.unwrap_or_else(|| bottom.linear());
        let blended = mode.blend(below, top.linear());
        acc = Some(if alpha == 1.0 {
            blended
        } else {
            [
                below[0] + alpha * (blended[0] - below[0]),
                below[1] + alpha * (blended[1] - below[1]),
                below[2] + alpha * (blended[2] - below[2]),
            ]
        });
    }
    acc.map_or(bottom.logc4, logc4_encode_rgb)
}

/// A Splitter (`docs/25` section 3.4): `(r, r, r)`, `(g, g, g)` and `(b, b, b)`.
#[must_use]
pub fn split(v: Vec3) -> [Vec3; 3] {
    [[v[0]; 3], [v[1]; 3], [v[2]; 3]]
}

/// A Combiner (`docs/25` section 3.4): the red of the first picture, the green of the second, the blue of the third.
#[must_use]
pub fn combine(red: Vec3, green: Vec3, blue: Vec3) -> Vec3 {
    [red[0], green[1], blue[2]]
}
