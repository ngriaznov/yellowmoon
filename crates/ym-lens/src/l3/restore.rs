//! The restored light `R` and the highlights `H` (`docs/21` section 4, L3).
//!
//! R5 leaves a pixel at its clip level in every channel only where every channel clipped, so the clip test
//! `f` ramps from 0 to 1 as `min_c(X_c / g0_c)` goes from 0.95 to 0.98. The restored light is `X (1 + f (2^h -
//! 1))` with the headroom `h` of 3 EV, and only the flares read it: the picture's own clipped pixels are not
//! raised. The highlights are the restored light weighted by a soft knee that is 0 at half the clip level and 1
//! at the clip level; the knee reads the brightest channel, so a coloured light keeps its colour in its ghosts.
//!
//! A pixel R5 left at the clip pattern (every channel at its own clip level, which is a colour cast of the
//! white balance, not the light's: it happens only with highlight recovery switched off) is restored as a
//! neutral of the same total, so a clipped white light is white in the flares whatever `g0` is.

use crate::consts::{CLIP_PATTERN_SPREAD, KNEE_START, LENS_HEADROOM_EV};
use crate::l3::profile::smoothstep;
use crate::plan::LensPlan;

/// The clip test, the headroom and the knee for one scale of the picture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Restore {
    clip: [f64; 3],
    ramp: (f64, f64),
    /// `2^h - 1`.
    boost: f64,
    /// The share of the small lights' extra headroom that is kept (1: all of it, `crate::l3::lights`).
    scale: f64,
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
            scale: 1.0,
        }
    }

    /// The same restoration keeping only `scale` (0 to 1) of the light the extra headroom of a small clipped
    /// light adds beyond the plain headroom: a photo whose lights would add more than the cap of
    /// [`LIGHT_BOOST_ENERGY_CAP`](crate::consts::LIGHT_BOOST_ENERGY_CAP) has them all reduced alike.
    pub(crate) fn with_boost_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// The clip test `f` of the pixel `x`: 0 below the ramp, 1 above it, 0 when no channel has a clip level.
    #[inline]
    pub(crate) fn clip_test(&self, x: [f64; 3]) -> f64 {
        let mut m = f64::INFINITY;
        for (v, g0) in x.iter().zip(&self.clip) {
            if *g0 > 0.0 {
                m = m.min(v / g0);
            }
        }
        if !m.is_finite() {
            return 0.0;
        }
        ((m - self.ramp.0) / (self.ramp.1 - self.ramp.0)).clamp(0.0, 1.0)
    }

    /// The clip test's pixel `x` with the clip pattern's colour cast taken off: `x + w (mean(x) - x)`, `w` the
    /// clip test times how close the channels are to each other in clip levels (1 up to no spread, 0 from
    /// [`CLIP_PATTERN_SPREAD`] on). A pixel with its own colour (R5's recovered cores are neutral or the
    /// light's colour, their channels differ in clip levels) is as it is.
    #[inline]
    pub(crate) fn without_clip_cast(&self, x: [f64; 3]) -> [f64; 3] {
        let (mut hi, mut lo) = (f64::NEG_INFINITY, f64::INFINITY);
        for (v, g0) in x.iter().zip(&self.clip) {
            if *g0 > 0.0 {
                let n = v / g0;
                hi = hi.max(n);
                lo = lo.min(n);
            }
        }
        if !hi.is_finite() || hi <= 0.0 {
            return x;
        }
        let w = self.clip_test(x) * (1.0 - smoothstep((hi - lo) / hi / CLIP_PATTERN_SPREAD));
        if w <= 0.0 {
            return x;
        }
        let mean = (x[0] + x[1] + x[2]) / 3.0;
        [
            x[0] + w * (mean - x[0]),
            x[1] + w * (mean - x[1]),
            x[2] + w * (mean - x[2]),
        ]
    }

    /// The factor `1 + f (2^h - 1)` the pixel `x` is raised by.
    #[inline]
    pub(crate) fn factor(&self, x: [f64; 3]) -> f64 {
        1.0 + self.clip_test(x) * self.boost
    }

    /// The factor `1 + f (2^h + s (2^(h + extra_ev) - 2^h) - 1)`: the pixel of a light assumed `extra_ev` EV
    /// brighter than the plain headroom, of which the share `s` ([`Restore::with_boost_scale`]) is kept
    /// ([`crate::l3::lights`]).
    #[inline]
    pub(crate) fn factor_ev(&self, x: [f64; 3], extra_ev: f64) -> f64 {
        if extra_ev == 0.0 {
            return self.factor(x);
        }
        let plain = libm::exp2(LENS_HEADROOM_EV);
        let lit = libm::exp2(LENS_HEADROOM_EV + extra_ev);
        1.0 + self.clip_test(x) * (plain + self.scale * (lit - plain) - 1.0)
    }

    /// The light, per channel in clip levels averaged over the channels, that the extra headroom `extra_ev` adds
    /// to the highlights of the pixel `x` beyond the plain headroom, before any scale: `mean_c(X_c / g0_c) f
    /// knee (2^(h + extra_ev) - 2^h)`.
    pub(crate) fn extra_light(&self, x: [f64; 3], extra_ev: f64) -> f64 {
        let mut sum = 0.0;
        for (v, g0) in x.iter().zip(&self.clip) {
            if *g0 > 0.0 {
                sum += v / g0;
            }
        }
        let plain = libm::exp2(LENS_HEADROOM_EV);
        let lit = libm::exp2(LENS_HEADROOM_EV + extra_ev);
        sum / 3.0 * self.clip_test(x) * self.knee(x) * (lit - plain)
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
        let y = self.without_clip_cast(x);
        [y[0] * g, y[1] * g, y[2] * g]
    }

    /// The restored light of the pixel `x` of a light assumed `extra_ev` EV brighter than the plain headroom.
    #[inline]
    pub(crate) fn restored_ev(&self, x: [f64; 3], extra_ev: f64) -> [f64; 3] {
        let g = self.factor_ev(x, extra_ev);
        let y = self.without_clip_cast(x);
        [y[0] * g, y[1] * g, y[2] * g]
    }

    /// The highlights of the pixel `x` of a light assumed `extra_ev` EV brighter than the plain headroom.
    #[inline]
    pub(crate) fn highlight_ev(&self, x: [f64; 3], extra_ev: f64) -> [f64; 3] {
        let k = self.knee(x);
        let r = self.restored_ev(x, extra_ev);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::CLIP_RAMP;
    use crate::test_support::write_synthetic_lenses;

    fn restore_with_clip(clip: [f32; 3]) -> Restore {
        use crate::amounts::{LensAmounts, LensKind};
        use crate::plan::{Axis, FrameInfo};
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_lenses(dir.path()).unwrap();
        let lib = crate::data::LensLibrary::from_dir(dir.path()).unwrap();
        let amounts = LensAmounts {
            kind: LensKind::Classic,
            flares: 100.0,
            edges: 0.0,
            anamorphic: 0.0,
        };
        let frame = FrameInfo {
            width: 600,
            height: 400,
            axis: Axis::X,
            f_number: None,
            clip,
        };
        let plan = LensPlan::new(
            &lib,
            &amounts,
            frame,
            &crate::spectral::CameraResponse::neutral(),
        )
        .unwrap();
        Restore::new(&plan, CLIP_RAMP)
    }

    #[test]
    fn a_pixel_left_at_the_clip_pattern_is_restored_as_a_neutral_of_the_same_total() {
        // With highlight recovery off a fully clipped pixel sits at (g0_R, 1, g0_B): a magenta cast.
        let rs = restore_with_clip([2.9, 1.0, 1.5]);
        let x = [2.9, 1.0, 1.5];
        let r = rs.restored(x);
        let g = rs.factor(x);
        let mean = (2.9 + 1.0 + 1.5) / 3.0;
        for v in r {
            assert!((v - mean * g).abs() < 1e-9, "{r:?}");
        }
        // A recovered neutral core (R5 raises fully clipped photosites to the largest): as it is.
        let y = [3.1, 3.0, 3.0];
        let r = rs.restored(y);
        let g = rs.factor(y);
        for (a, b) in r.iter().zip(&y) {
            assert!((a - b * g).abs() < 1e-12, "{r:?}");
        }
        // A pixel that is not clipped at all keeps its colour and its light.
        let z = [0.3, 0.2, 0.1];
        assert_eq!(rs.restored(z), z);
        // A coloured light that clipped (the sun at sunset, red above its clip, blue well below): as it is.
        let warm = [3.2, 2.0, 0.6];
        let g = rs.factor(warm);
        let r = rs.restored(warm);
        for (a, b) in r.iter().zip(&warm) {
            assert!((a - b * g).abs() < 1e-12, "{r:?}");
        }
    }
}
