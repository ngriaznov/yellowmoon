//! Stage 9: the scan. The print's dye densities become a display colour.
//!
//! Written from CIE 15 (tristimulus values of a reflecting or transmitting object under an illuminant, the 1931
//! observer), Bradford adaptation (Lam 1985; Hunt, *The Reproduction of Colour*, chapter 14), and the paper data
//! sheets the profiles are read from.
//!
//! **Light off the print.** `R(l) = 10^-(base(l) + sum_c D_c cd_c(l))` under the viewing light (D50), projected on
//! the 1931 observer: `XYZ = sum_l R(l) ill(l) cmf(l) / sum_l ill(l) ybar(l)`. The tristimulus values are adapted
//! from the viewing white (D50) to the output's white (D65) by Bradford, and turned into linear RGB in the
//! output's primaries.
//!
//! **White point.** Paper white (the base alone, `D = 0`) is mapped to 1 in each channel (the scanner's white
//! correction), so the print has no paper tint. For a slide the white is the bare light: the base is part of the
//! picture.
//!
//! The values are not clipped here: a saturated colour can leave the output's gamut and come out negative. The
//! return stage ([`crate::chain`]) holds them to the cine rendering's range.

use ym_color::consts::linalg::{self, Mat3};
use ym_color::consts::{D65_XY, xy_to_xyz};
use ym_color::deltae::xyz_to_lab;
use ym_color::gamut::matrices;

use crate::cie::{Cie, Light};
use crate::math::pow10;
use crate::params::Output;
use crate::profile::Profile;
use crate::scene::bradford;
use crate::{N_LAMBDA, Spectrum, Vec3};

/// What the viewed medium's white is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhiteKind {
    /// A print: the bare paper base is white.
    PaperBase,
    /// A slide: the bare light is white.
    Light,
}

/// Stage 9 for one medium and one output.
#[derive(Debug, Clone)]
pub struct Scan {
    weights: [Vec3; N_LAMBDA],
    base: Spectrum,
    dye: Vec<Vec3>,
    to_rgb: Mat3,
    white_rgb: Vec3,
    rgb_to_xyz: Mat3,
    d65_to_d50: Mat3,
    d50_white: Vec3,
}

impl Scan {
    /// Builds the scan of `medium` (a paper, or a slide film) into `output`.
    #[must_use]
    pub fn new(medium: &Profile, cie: &Cie, output: Output, white: WhiteKind) -> Self {
        let ill = cie.light(Light::D50);
        let norm: f64 = (0..N_LAMBDA).map(|i| ill[i] * cie.cmf[i][1]).sum();
        let mut weights = [[0.0; 3]; N_LAMBDA];
        for i in 0..N_LAMBDA {
            for c in 0..3 {
                weights[i][c] = ill[i] * cie.cmf[i][c] / norm;
            }
        }
        let m = matrices();
        let (to_xyz, from_xyz) = match output {
            Output::Srgb => (m.rec709_to_xyz, m.xyz_to_rec709),
            Output::DisplayP3 => (m.p3_to_xyz, m.xyz_to_p3),
        };
        let d50_white = cie.xyz_of_light(&ill, true);
        let d65_white = xy_to_xyz(D65_XY);
        let to_rgb = linalg::mul(&from_xyz, &bradford(d50_white, d65_white));
        let mut scan = Self {
            weights,
            base: medium.base_density,
            dye: medium.channel_density.clone(),
            to_rgb,
            white_rgb: [1.0; 3],
            rgb_to_xyz: to_xyz,
            d65_to_d50: bradford(d65_white, d50_white),
            d50_white,
        };
        if white == WhiteKind::PaperBase {
            scan.white_rgb = scan.linear_rgb([0.0; 3]);
        } else {
            let mut w = scan.clone();
            w.base = [0.0; N_LAMBDA];
            scan.white_rgb = w.linear_rgb([0.0; 3]);
        }
        scan
    }

    /// The output RGB of the dye densities `d`, before the white correction.
    fn linear_rgb(&self, d: Vec3) -> Vec3 {
        let mut xyz = [0.0; 3];
        for i in 0..N_LAMBDA {
            let dye = &self.dye[i];
            let t = pow10(-(self.base[i] + d[0] * dye[0] + d[1] * dye[1] + d[2] * dye[2]));
            for c in 0..3 {
                xyz[c] += t * self.weights[i][c];
            }
        }
        linalg::mul_vec(&self.to_rgb, xyz)
    }

    /// The display colour of the print with dye densities `d`: linear RGB in the output's primaries, paper white
    /// at 1.
    #[must_use]
    pub fn display(&self, d: Vec3) -> Vec3 {
        let rgb = self.linear_rgb(d);
        [
            rgb[0] / self.white_rgb[0],
            rgb[1] / self.white_rgb[1],
            rgb[2] / self.white_rgb[2],
        ]
    }

    /// The luminance `Y` of a display colour in this output.
    #[must_use]
    pub fn luminance(&self, display: Vec3) -> f64 {
        self.rgb_to_xyz[1][0] * display[0] + self.rgb_to_xyz[1][1] * display[1] + self.rgb_to_xyz[1][2] * display[2]
    }

    /// CIELAB (D50) of a display colour: the output's RGB to XYZ, Bradford to D50, `Lab` against the D50 white.
    #[must_use]
    pub fn lab_d50(&self, display: Vec3) -> Vec3 {
        let xyz = linalg::mul_vec(&self.rgb_to_xyz, display);
        let xyz50 = linalg::mul_vec(&self.d65_to_d50, xyz);
        xyz_to_lab(xyz50, self.d50_white)
    }
}
