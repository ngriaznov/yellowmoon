//! Stage 1: scene light. LogC4 / AWG4 in, a chromaticity and a luminance out.
//!
//! Written from the LogC4 definition (in `ym-color`), the Bradford chromatic adaptation transform (Lam, 1985;
//! Hunt, *The Reproduction of Colour*, chapter 14; Lindbloom's published matrix) and CIE 15 (chromaticity).
//!
//! The picture is taken to be lit by the film's own reference light: a neutral in the picture is a neutral on the
//! film (`docs/24` section 2.6). So the scene's XYZ (AWG4 white is D65) is adapted by Bradford from D65 to the
//! reference light's white, and then turned into `(x, y)` and `Y`.
//!
//! **Soft gamut compression.** A camera colour may lie outside the spectral locus (negative components, or a
//! saturated primary: AWG4's red is on the locus). The chromaticity is moved along the ray from the reference
//! white: with `R` the distance from the white to the locus (the polygon of the observer's monochromatic
//! stimuli, closed by the purple line) along that ray and `r` the colour's distance, a colour with `r <= 0.85 R`
//! is untouched, and beyond the knee the distance becomes
//! `k + (L - k) tanh((r - k) / (L - k))` with `k = 0.85 R` and `L = 0.985 R`. The map is smooth, monotone, never
//! reaches the locus, and leaves ordinary colours alone. The luminance `Y` is kept.

use ym_color::consts::linalg::{self, Mat3};
use ym_color::gamut::matrices;
use ym_color::logc::logc4_decode_rgb;

use crate::cie::Cie;
use crate::{N_LAMBDA, Vec3};

/// The Bradford cone-response matrix (Lam 1985).
pub const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// The Bradford adaptation from a source white to a destination white (both XYZ).
#[must_use]
pub fn bradford(src_white: Vec3, dst_white: Vec3) -> Mat3 {
    let s = linalg::mul_vec(&BRADFORD, src_white);
    let d = linalg::mul_vec(&BRADFORD, dst_white);
    let scale = linalg::diag([d[0] / s[0], d[1] / s[1], d[2] / s[2]]);
    let inv = linalg::inv(&BRADFORD).unwrap_or(linalg::IDENTITY);
    linalg::mul(&inv, &linalg::mul(&scale, &BRADFORD))
}

/// What stage 1 gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneLight {
    /// Chromaticity x, after the soft compression.
    pub x: f64,
    /// Chromaticity y, after the soft compression.
    pub y: f64,
    /// Luminance relative to the scene white (a neutral of `g` has `Y = g`); 0 when negative.
    pub luminance: f64,
}

/// Stage 1 for one reference light.
#[derive(Debug, Clone)]
pub struct SceneStage {
    to_adapted_xyz: Mat3,
    white_xy: [f64; 2],
    locus: Vec<[f64; 2]>,
}

/// The knee of the gamut compression, as a share of the distance to the locus.
const KNEE: f64 = 0.85;
/// The distance the compression approaches, as a share of the distance to the locus.
const LIMIT: f64 = 0.985;

impl SceneStage {
    /// Builds the stage for a film whose reference light has the spectrum `reference`.
    #[must_use]
    pub fn new(cie: &Cie, reference: &crate::Spectrum) -> Self {
        let m = matrices();
        let d65 = ym_color::consts::xy_to_xyz(ym_color::consts::D65_XY);
        let ref_white = cie.xyz_of_light(reference, true);
        let adapt = bradford(d65, ref_white);
        let to_adapted_xyz = linalg::mul(&adapt, &m.awg4_to_xyz);
        let white_xy = cie.xy_of_light(reference);
        let mut locus = Vec::with_capacity(N_LAMBDA);
        for c in &cie.cmf {
            let s = c[0] + c[1] + c[2];
            if c[1] > 1e-6 && s > 0.0 {
                locus.push([c[0] / s, c[1] / s]);
            }
        }
        Self {
            to_adapted_xyz,
            white_xy,
            locus,
        }
    }

    /// The chromaticity of the reference white.
    #[must_use]
    pub fn white_xy(&self) -> [f64; 2] {
        self.white_xy
    }

    /// Runs the stage on a LogC4 / AWG4 triple.
    #[must_use]
    pub fn run(&self, logc4: Vec3) -> SceneLight {
        self.from_linear(logc4_decode_rgb(logc4))
    }

    /// Runs the stage on a linear AWG4 triple.
    #[must_use]
    pub fn from_linear(&self, awg4: Vec3) -> SceneLight {
        let xyz = linalg::mul_vec(&self.to_adapted_xyz, awg4);
        let sum = xyz[0] + xyz[1] + xyz[2];
        let luminance = if xyz[1].is_finite() { xyz[1].max(0.0) } else { 0.0 };
        if !(sum > 0.0) || !sum.is_finite() {
            return SceneLight {
                x: self.white_xy[0],
                y: self.white_xy[1],
                luminance,
            };
        }
        let p = [xyz[0] / sum, xyz[1] / sum];
        let q = self.compress(p);
        SceneLight {
            x: q[0],
            y: q[1],
            luminance,
        }
    }

    /// The soft compression of a chromaticity toward the reference white (see the module header).
    #[must_use]
    pub fn compress(&self, p: [f64; 2]) -> [f64; 2] {
        let w = self.white_xy;
        let d = [p[0] - w[0], p[1] - w[1]];
        let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if r < 1e-12 {
            return p;
        }
        let u = [d[0] / r, d[1] / r];
        let big_r = self.locus_distance(u);
        let k = KNEE * big_r;
        if r <= k {
            return p;
        }
        let l = LIMIT * big_r;
        let r2 = k + (l - k) * ((r - k) / (l - k)).tanh();
        [w[0] + u[0] * r2, w[1] + u[1] * r2]
    }

    /// The distance from the white along the unit direction `u` to the boundary of the locus polygon.
    fn locus_distance(&self, u: [f64; 2]) -> f64 {
        let w = self.white_xy;
        let n = self.locus.len();
        let mut best = f64::INFINITY;
        for k in 0..n {
            let a = self.locus[k];
            let b = self.locus[(k + 1) % n];
            // Solve w + t u = a + s (b - a), t > 0, 0 <= s <= 1.
            let e = [b[0] - a[0], b[1] - a[1]];
            let den = u[0] * e[1] - u[1] * e[0];
            if den.abs() < 1e-18 {
                continue;
            }
            let aw = [a[0] - w[0], a[1] - w[1]];
            let t = (aw[0] * e[1] - aw[1] * e[0]) / den;
            let s = (aw[0] * u[1] - aw[1] * u[0]) / den;
            if t > 0.0 && (-1e-12..=1.0 + 1e-12).contains(&s) && t < best {
                best = t;
            }
        }
        if best.is_finite() { best } else { 1.0 }
    }
}
