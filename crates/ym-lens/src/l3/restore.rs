//! The restored light `R` and the highlights `H` (`docs/21` section 4, L3).
//!
//! R5 leaves a pixel at its clip level in every channel only where every channel clipped, so the clip test
//! `f` ramps from 0 to 1 as `min_c(X_c / g0_c)` goes from 0.95 to 0.98. The restored light is `X (1 + f (2^h -
//! 1))` with the headroom `h` of 3 EV, and only the flares read it: the picture's own clipped pixels are not
//! raised. The highlights are the restored light weighted by a soft knee that is 0 at half the clip level and 1
//! at the clip level; the knee reads the brightest channel, so a coloured light keeps its colour in its ghosts.

use crate::consts::{KNEE_START, LENS_HEADROOM_EV};
use crate::l3::profile::smoothstep;
use crate::plan::LensPlan;

/// The clip test, the headroom and the knee for one scale of the picture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Restore {
    clip: [f64; 3],
    ramp: (f64, f64),
    /// `2^h - 1`.
    boost: f64,
}

impl Restore {
    /// The restoration of `plan`'s clip levels with the ramp `ramp` (`CLIP_RAMP` for the picture's own pixels,
    /// `CLIP_RAMP_PLANE` for the lens plane).
    pub(crate) fn new(plan: &LensPlan, ramp: (f64, f64)) -> Self {
        let c = plan.frame().clip;
        Restore {
            clip: [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])],
            ramp,
            boost: libm::exp2(LENS_HEADROOM_EV) - 1.0,
        }
    }

    /// The factor `1 + f (2^h - 1)` the pixel `x` is raised by.
    #[inline]
    pub(crate) fn factor(&self, x: [f64; 3]) -> f64 {
        let mut m = f64::INFINITY;
        for (v, g0) in x.iter().zip(&self.clip) {
            if *g0 > 0.0 {
                m = m.min(v / g0);
            }
        }
        if !m.is_finite() {
            return 1.0;
        }
        let f = ((m - self.ramp.0) / (self.ramp.1 - self.ramp.0)).clamp(0.0, 1.0);
        1.0 + f * self.boost
    }

    /// The soft knee of the pixel `x`: 0 at half the clip level of its brightest channel, 1 at the clip level.
    #[inline]
    pub(crate) fn knee(&self, x: [f64; 3]) -> f64 {
        let mut m = f64::NEG_INFINITY;
        for (v, g0) in x.iter().zip(&self.clip) {
            if *g0 > 0.0 {
                m = m.max(v / g0);
            }
        }
        if !m.is_finite() {
            return 0.0;
        }
        smoothstep((m - KNEE_START) / (1.0 - KNEE_START))
    }

    /// The restored light of the pixel `x`.
    #[inline]
    pub(crate) fn restored(&self, x: [f64; 3]) -> [f64; 3] {
        let g = self.factor(x);
        [x[0] * g, x[1] * g, x[2] * g]
    }

    /// The highlights of the pixel `x`: the restored light times the knee.
    #[inline]
    pub(crate) fn highlight(&self, x: [f64; 3]) -> [f64; 3] {
        let k = self.knee(x);
        let r = self.restored(x);
        [r[0] * k, r[1] * k, r[2] * k]
    }

    /// The largest channel of `h` in clip levels: how strongly the pixel counts as a highlight.
    #[inline]
    pub(crate) fn level(&self, h: [f64; 3]) -> f64 {
        let mut m = 0.0f64;
        for (v, g0) in h.iter().zip(&self.clip) {
            if *g0 > 0.0 {
                m = m.max(v / g0);
            }
        }
        m
    }
}
