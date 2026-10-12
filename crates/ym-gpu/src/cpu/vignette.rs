//! R3 vignetting correction from Sony tag 0x7032 (`docs/05-raw-processing.md` section R3).
//!
//! The tag holds `n` signed 16-bit knots `v_i`. Knot `i` sits at the normalised radius
//! `(i + 0.5) / (n - 1)`, where `r = 1` is the half-diagonal of the active area, and describes the
//! relative illumination the lens leaves at that radius,
//! `G_i = 2^(0.5 - 2^(v_i * 2^-13 - 1))` ([`knot_attenuation`]): `v = 0` gives `G = 1`, positive
//! `v` gives `G < 1` (a darker corner), negative `v` gives `G` up to `sqrt(2)`. `G(r)` is the
//! linear interpolation of the knots, held at the end values; the correction **divides** each
//! photosite by `G(r)` (`r` measured from the frame centre to the photosite centre), i.e. it
//! brightens where the lens darkened.
//!
//! # Verified against darktable
//!
//! These conventions are those of darktable 4.6.1's "embedded metadata" lens correction (version 2
//! of its algorithm, the default). `tools/lens/` renders flat fields with `darktable-cli` and
//! measures the gain it applies (the lens module's own float output): the radial profile agrees with
//! this module to 0.008%. Readings the same renders reject (maximum deviation of the gain): the
//! first reading of the spec, multiplying by `G^2` with a cubic through knots at `i / (n - 1)`
//! (66%); the same dividing by `G^2` (17%); darktable's older algorithm 1, dividing by `G^2` with
//! the right knots (15%); the right gain with knots at `i / (n - 1)` (1.4%). A cubic instead of
//! linear interpolation cannot be told apart for vignetting (0.03%). The recorded measurement is
//! `tests/cpu_data/darktable_oracle.json`; `tests/cpu_lens.rs` checks the code against it.
//!
//! The one difference is the pixel grid: darktable counts radii from pixel *indices* (the centre
//! of the frame at index `W / 2`), this module from pixel *centres* (`x + 0.5`). It is a shift of
//! half a pixel, worth about 1e-5 in the gain.

use rayon::prelude::*;

use super::lens::{LensSwitches, LensTags};
use super::spline::LinearKnots;
use super::{CpuError, Mosaic, Placement};

/// The smallest attenuation the correction divides by (darktable clamps at the same value).
pub const MIN_ATTENUATION: f64 = 1e-4;

/// The relative illumination `G` at one raw knot value: `2^(0.5 - 2^(v * 2^-13 - 1))`.
pub fn knot_attenuation(v: i16) -> f64 {
    (0.5 - (f64::from(v) / 8192.0 - 1.0).exp2()).exp2()
}

/// The vignetting function `G(r)` built from tag 0x7032.
#[derive(Clone, Debug)]
pub struct VignetteProfile {
    attenuation: LinearKnots,
}

impl VignetteProfile {
    /// Builds the profile from the raw knots of tag 0x7032 (at least two).
    pub fn from_tag(knots: &[i16]) -> Result<VignetteProfile, CpuError> {
        if knots.len() < 2 {
            return Err(CpuError::TooFewKnots {
                tag: "0x7032 vignetting",
                got: knots.len(),
                min: 2,
            });
        }
        let ys: Vec<f64> = knots.iter().map(|&v| knot_attenuation(v)).collect();
        Ok(VignetteProfile {
            attenuation: LinearKnots::sony(&ys)?,
        })
    }

    /// The profile for a file's lens tags and the RAW-tab switches: `None` when
    /// `decode.lens_corrections` is off or the file has no 0x7032 tag ("No lens data in this file").
    pub fn from_tags(
        tags: &LensTags,
        switches: LensSwitches,
    ) -> Result<Option<VignetteProfile>, CpuError> {
        if !switches.lens_corrections || tags.vignetting.is_empty() {
            return Ok(None);
        }
        VignetteProfile::from_tag(&tags.vignetting).map(Some)
    }

    /// The relative illumination `G(r)` the lens leaves at normalised radius `r`.
    #[inline]
    pub fn attenuation_at(&self, r: f64) -> f64 {
        self.attenuation.eval(r)
    }

    /// The multiplier applied to a photosite at normalised radius `r`: `1 / G(r)`, with `G` not
    /// allowed below [`MIN_ATTENUATION`].
    #[inline]
    pub fn gain_at(&self, r: f64) -> f64 {
        1.0 / self.attenuation_at(r).max(MIN_ATTENUATION)
    }

    /// Normalised radius of photosite `(x, y)` of a buffer placed at `place`: distance from the
    /// frame centre to the photosite centre, over the half-diagonal.
    #[inline]
    pub fn radius(place: &Placement, x: usize, y: usize) -> f64 {
        let (cx, cy) = place.centre();
        let dx = (place.x0 + x) as f64 + 0.5 - cx;
        let dy = (place.y0 + y) as f64 + 0.5 - cy;
        dx.hypot(dy) / place.half_diagonal()
    }
}

/// Multiplies each photosite by `1 / G(r)`, in place. Pointwise given the placement (exact under
/// tiling).
pub fn apply_vignette(m: &mut Mosaic, profile: &VignetteProfile) {
    let place = m.place;
    let w = m.width.max(1);
    let (cx, cy) = place.centre();
    let inv_hd = 1.0 / place.half_diagonal();
    m.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let dy = (place.y0 + y) as f64 + 0.5 - cy;
        for (x, v) in row.iter_mut().enumerate() {
            let dx = (place.x0 + x) as f64 + 0.5 - cx;
            let r = dx.hypot(dy) * inv_hd;
            *v *= profile.gain_at(r) as f32;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cfa;

    #[test]
    fn knot_attenuation_matches_the_documented_formula() {
        for v in [-8192i16, -4000, -1, 0, 1, 3000, 8192, 16384] {
            let a = f64::from(v) * 2f64.powi(-13);
            let direct = 2f64.powf(0.5 - 2f64.powf(a - 1.0));
            assert!((knot_attenuation(v) - direct).abs() < 1e-12, "v={v}");
        }
        assert_eq!(knot_attenuation(0), 1.0);
        // v = 8192 (a = 1): 2^(0.5 - 1) = 1 / sqrt(2), a correction of +0.5 EV.
        assert!((knot_attenuation(8192) - 0.5f64.sqrt()).abs() < 1e-12);
        // v = 16384 (a = 2): 2^(0.5 - 2) = 2^-1.5, +1.5 EV.
        assert!((knot_attenuation(16384) - 2f64.powf(-1.5)).abs() < 1e-12);
        // Negative knots darken (G > 1), saturating at sqrt(2).
        assert!(knot_attenuation(-8192) > 1.18 && knot_attenuation(-8192) < 1.19);
        assert!(knot_attenuation(i16::MIN) < 2f64.sqrt());
    }

    #[test]
    fn knots_sit_at_half_a_step_and_are_read_linearly() {
        let knots: [i16; 4] = [0, 4096, 8192, 12288];
        let prof = VignetteProfile::from_tag(&knots).unwrap();
        // Knot i of 4 at (i + 0.5) / 3.
        for (i, &v) in knots.iter().enumerate() {
            let r = (i as f64 + 0.5) / 3.0;
            assert!(
                (prof.attenuation_at(r) - knot_attenuation(v)).abs() < 1e-12,
                "knot {i}"
            );
        }
        // Midway between knots 1 and 2 the value is the mean of the two attenuations.
        let mid = 0.5 * (knot_attenuation(4096) + knot_attenuation(8192));
        let r_mid = 0.5 * (1.5 + 2.5) / 3.0;
        assert!((prof.attenuation_at(r_mid) - mid).abs() < 1e-12);
        // Held at the first knot below it and at the last one above it.
        assert_eq!(prof.attenuation_at(0.0), 1.0);
        assert!((prof.attenuation_at(2.0) - knot_attenuation(12288)).abs() < 1e-12);
    }

    #[test]
    fn the_correction_divides_by_the_attenuation() {
        // Positive knots darken the corner (G < 1) and the correction brightens it.
        let prof = VignetteProfile::from_tag(&[0, 2000, 6000, 12000]).unwrap();
        for r in [0.0, 0.3, 0.7, 1.0] {
            let g = prof.attenuation_at(r);
            assert!(g <= 1.0);
            assert!((prof.gain_at(r) * g - 1.0).abs() < 1e-12);
            assert!(prof.gain_at(r) >= 1.0);
        }
        // Negative knots (G > 1) give a gain below one: the tag is applied as written.
        let neg = VignetteProfile::from_tag(&[0, -2000, -6000]).unwrap();
        assert!(neg.gain_at(1.0) < 1.0);
        // The largest encodable knot (a = 4) leaves 2^-7.5 of the light: a gain of about 181, finite
        // and well inside darktable's `fmaxf(1e-4, sf)` guard.
        let extreme = VignetteProfile::from_tag(&[i16::MAX, i16::MAX]).unwrap();
        assert!(extreme.attenuation_at(0.5) > MIN_ATTENUATION);
        assert!((extreme.gain_at(0.5) - 2f64.powf(7.5)).abs() / 2f64.powf(7.5) < 1e-3);
    }

    #[test]
    fn correction_flattens_a_synthetic_vignette() {
        // A lens that darkens the corner by 2^-0.6 or so; the correction must undo it exactly.
        let knots: [i16; 9] = [0, 300, 1000, 2000, 3200, 4600, 6200, 7800, 9000];
        let prof = VignetteProfile::from_tag(&knots).unwrap();
        let (w, h) = (96, 64);
        let mut m = Mosaic::new(w, h, Cfa::RGGB);
        for y in 0..h {
            for x in 0..w {
                let r = VignetteProfile::radius(&m.place, x, y);
                m.data[y * w + x] = (0.5 * prof.attenuation_at(r)) as f32;
            }
        }
        let corner_before = m.get(0, 0);
        let centre = m.get(w / 2, h / 2);
        assert!(corner_before / centre < 0.75, "synthetic vignette too weak");
        apply_vignette(&mut m, &prof);
        let lo = m.data.iter().cloned().fold(f32::MAX, f32::min);
        let hi = m.data.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (hi - lo) / 0.5 < 1e-4,
            "not flat after correction: {lo}..{hi}"
        );
        assert!((m.get(0, 0) / m.get(w / 2, h / 2) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn corner_radius_is_one_and_centre_is_zero() {
        let place = Placement::whole(100, 60);
        let r_corner = VignetteProfile::radius(&place, 99, 59);
        assert!((r_corner - 1.0).abs() < 0.02);
        assert!(VignetteProfile::radius(&place, 50, 30) < 0.02);
    }

    #[test]
    fn switches_and_missing_tags_give_no_profile() {
        let tags = LensTags {
            vignetting: vec![0, 1000, 3000],
            ..Default::default()
        };
        assert!(
            VignetteProfile::from_tags(&tags, LensSwitches::default())
                .unwrap()
                .is_some()
        );
        let off = LensSwitches {
            lens_corrections: false,
            chromatic_aberration: true,
        };
        assert!(VignetteProfile::from_tags(&tags, off).unwrap().is_none());
        assert!(
            VignetteProfile::from_tags(&LensTags::default(), LensSwitches::default())
                .unwrap()
                .is_none()
        );
        let bad = LensTags {
            vignetting: vec![5],
            ..Default::default()
        };
        assert!(VignetteProfile::from_tags(&bad, LensSwitches::default()).is_err());
        assert!(VignetteProfile::from_tag(&[0]).is_err());
    }

    #[test]
    fn tile_equals_whole() {
        let prof = VignetteProfile::from_tag(&[0, 500, 2000, 4000]).unwrap();
        let (w, h) = (40, 30);
        let mut whole = Mosaic::new(w, h, Cfa::RGGB);
        whole
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = 0.1 + (i % 17) as f32 * 0.03);
        let tile_src = whole.extract(crate::cpu::Rect::new(7, 5, 20, 17));
        apply_vignette(&mut whole, &prof);
        let mut tile = tile_src;
        apply_vignette(&mut tile, &prof);
        for y in 0..17 {
            for x in 0..20 {
                assert_eq!(tile.get(x, y), whole.get(x + 7, y + 5));
            }
        }
    }
}
