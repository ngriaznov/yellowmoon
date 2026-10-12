//! The target of the fit: what Yellowmoon shows. A DNG pixel is scene-linear AWG4 after the look and curve (C1 to C11),
//! and the picture a user sees of it is the ACES 2 output transform for sRGB (C12) and the sRGB encode (C13):
//! `ym_color::display::DisplayTransform::Aces2` with `OutputTarget::Srgb`. This module is that function for one colour,
//! plus the colour differences between display code values.

use ym_color::aces2::{Aces2Output, Aces2Transform};
use ym_color::consts::{Vec3, linalg};
use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::encode::srgb_decode;
use ym_color::gamut;

/// The sRGB code values (0 to 1, not quantised) Yellowmoon shows for a scene-linear AWG4 colour.
#[must_use]
pub fn aces2_srgb(awg4: Vec3) -> Vec3 {
    let ap0 = linalg::mul_vec(&gamut::matrices().awg4_to_ap0, awg4);
    Aces2Transform::shared(Aces2Output::Srgb).transform_ap0(ap0)
}

/// CIE L*a*b* (D65) of sRGB code values.
#[must_use]
pub fn lab_of_srgb_code(code: Vec3) -> Vec3 {
    let linear = code.map(|c| srgb_decode(c.clamp(0.0, 1.0)));
    xyz_to_lab(
        linalg::mul_vec(&gamut::matrices().rec709_to_xyz, linear),
        d65_white_xyz(),
    )
}

/// CIEDE2000 between two sets of sRGB code values.
#[must_use]
pub fn delta_e_codes(a: Vec3, b: Vec3) -> f64 {
    delta_e_2000(lab_of_srgb_code(a), lab_of_srgb_code(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_grey_and_black_and_white_land_where_aces_puts_them() {
        let grey = aces2_srgb([0.18; 3]);
        // ACES 2 puts 18 % grey at about 10 to 20 % display light, which is a code of 0.35 to 0.5
        assert!(grey.iter().all(|c| (0.3..0.6).contains(c)), "{grey:?}");
        assert!(grey[0] == grey[1] && grey[1] == grey[2] || (grey[0] - grey[2]).abs() < 1e-3);
        let black = aces2_srgb([0.0; 3]);
        assert!(black.iter().all(|c| c.abs() < 1e-6), "{black:?}");
        let white = aces2_srgb([1000.0; 3]);
        assert!(white.iter().all(|c| (c - 1.0).abs() < 1e-3), "{white:?}");
    }

    #[test]
    fn colour_differences_follow_cielab() {
        assert_eq!(delta_e_codes([0.5; 3], [0.5; 3]), 0.0);
        assert!((delta_e_codes([0.0; 3], [1.0; 3]) - 100.0).abs() < 0.5);
    }
}
