//! Shared constants: gamut primaries, D65 white, LogC and ACES constants (`docs/02-colour-pipeline.md` section Constants).
//!
//! Every matrix in the crate is derived in f64 from primaries and white points, never taken from a
//! rounded library table: [`Primaries::rgb_to_xyz`] is the normalised primary matrix (NPM), and
//! [`cat02_adaptation`] is von Kries adaptation in CAT02 space. The numeric LogC constants are
//! stored as literals (so the per-pixel code folds them) and checked against their defining
//! formulas by [`LogC4Constants::from_formulas`] in the tests.

pub mod linalg;

pub use linalg::{Mat3, Vec3};

/// An RGB colour space defined by CIE 1931 xy chromaticities of its primaries and white point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primaries {
    /// Red primary, CIE xy.
    pub red: [f64; 2],
    /// Green primary, CIE xy.
    pub green: [f64; 2],
    /// Blue primary, CIE xy.
    pub blue: [f64; 2],
    /// White point, CIE xy.
    pub white: [f64; 2],
}

/// CIE xy of the D65 white used by every gamut here except ACES AP0 (x 0.3127, y 0.3290).
pub const D65_XY: [f64; 2] = [0.3127, 0.3290];

/// CIE xy of the ACES white (x 0.32168, y 0.33767), the white of AP0 and of the ARRI table.
pub const ACES_WHITE_XY: [f64; 2] = [0.32168, 0.33767];

/// ARRI Wide Gamut 3 primaries with a D65 white.
pub const AWG3: Primaries = Primaries {
    red: [0.6840, 0.3130],
    green: [0.2210, 0.8480],
    blue: [0.0861, -0.1020],
    white: D65_XY,
};

/// ARRI Wide Gamut 4 primaries with a D65 white.
pub const AWG4: Primaries = Primaries {
    red: [0.7347, 0.2653],
    green: [0.1424, 0.8576],
    blue: [0.0991, -0.0308],
    white: D65_XY,
};

/// ACES AP0 primaries with the ACES white.
pub const AP0: Primaries = Primaries {
    red: [0.7347, 0.2653],
    green: [0.0, 1.0],
    blue: [0.0001, -0.0770],
    white: ACES_WHITE_XY,
};

/// ACES AP1 primaries with the ACES white.
pub const AP1: Primaries = Primaries {
    red: [0.713, 0.293],
    green: [0.165, 0.830],
    blue: [0.128, 0.044],
    white: ACES_WHITE_XY,
};

/// Rec.709 / sRGB primaries with a D65 white.
pub const REC709: Primaries = Primaries {
    red: [0.64, 0.33],
    green: [0.30, 0.60],
    blue: [0.15, 0.06],
    white: D65_XY,
};

/// Display P3 (P3-D65) primaries.
pub const DISPLAY_P3: Primaries = Primaries {
    red: [0.680, 0.320],
    green: [0.265, 0.690],
    blue: [0.150, 0.060],
    white: D65_XY,
};

/// Rec.2020 primaries with a D65 white.
pub const REC2020: Primaries = Primaries {
    red: [0.708, 0.292],
    green: [0.170, 0.797],
    blue: [0.131, 0.046],
    white: D65_XY,
};

/// Rec.709 luminance weights used by the ASC CDL saturation (`docs/02` section C10).
pub const REC709_LUMA_WEIGHTS: Vec3 = [0.2126, 0.7152, 0.0722];

/// The CAT02 cone-response matrix (CIECAM02), XYZ to RGB cone space.
pub const CAT02: Mat3 = [
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834],
];

/// XYZ of a chromaticity `xy` at luminance `Y = 1`.
#[must_use]
pub fn xy_to_xyz(xy: [f64; 2]) -> Vec3 {
    [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]]
}

/// The gamuts above under the `Primaries::NAME` spelling. They are the same values, not copies: there is
/// one table of chromaticities, in this module.
impl Primaries {
    /// [`AP0`].
    pub const AP0: Primaries = AP0;
    /// [`AP1`].
    pub const AP1: Primaries = AP1;
    /// [`REC709`].
    pub const REC709: Primaries = REC709;
    /// [`DISPLAY_P3`] (P3 primaries, D65 white).
    pub const P3_D65: Primaries = DISPLAY_P3;
    /// [`REC2020`].
    pub const REC2020: Primaries = REC2020;
}

impl Primaries {
    /// XYZ of the white point at `Y = 1`.
    #[must_use]
    pub fn white_xyz(&self) -> Vec3 {
        xy_to_xyz(self.white)
    }

    /// Normalised primary matrix: linear RGB to XYZ, with `RGB = (1,1,1)` mapping to the white at `Y = 1`.
    ///
    /// The primaries in this module are fixed and non-degenerate, so the inversion cannot fail;
    /// a degenerate custom `Primaries` returns the identity (and fails any downstream check).
    #[must_use]
    pub fn rgb_to_xyz(&self) -> Mat3 {
        let r = xy_to_xyz(self.red);
        let g = xy_to_xyz(self.green);
        let b = xy_to_xyz(self.blue);
        // Columns are the primaries' XYZ at Y = 1; solve P * S = white for the channel scales S.
        let p = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        let Some(p_inv) = linalg::inv(&p) else {
            return linalg::IDENTITY;
        };
        let s = linalg::mul_vec(&p_inv, self.white_xyz());
        [
            [p[0][0] * s[0], p[0][1] * s[1], p[0][2] * s[2]],
            [p[1][0] * s[0], p[1][1] * s[1], p[1][2] * s[2]],
            [p[2][0] * s[0], p[2][1] * s[1], p[2][2] * s[2]],
        ]
    }

    /// Inverse of [`Primaries::rgb_to_xyz`]: XYZ to linear RGB.
    #[must_use]
    pub fn xyz_to_rgb(&self) -> Mat3 {
        linalg::inv(&self.rgb_to_xyz()).unwrap_or(linalg::IDENTITY)
    }
}

/// Von Kries chromatic adaptation in CAT02 space from the source white to the destination white
/// (both XYZ). The result maps XYZ under the source white to XYZ under the destination white.
#[must_use]
pub fn cat02_adaptation(src_white_xyz: Vec3, dst_white_xyz: Vec3) -> Mat3 {
    let src = linalg::mul_vec(&CAT02, src_white_xyz);
    let dst = linalg::mul_vec(&CAT02, dst_white_xyz);
    let scale = linalg::diag([dst[0] / src[0], dst[1] / src[1], dst[2] / src[2]]);
    let cat_inv = linalg::inv(&CAT02).unwrap_or(linalg::IDENTITY);
    linalg::mul(&cat_inv, &linalg::mul(&scale, &CAT02))
}

/// The constants of ARRI LogC4 (ARRI LogC4 specification, 2022).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogC4Constants {
    /// `a = (2^18 - 16) / 117.45`.
    pub a: f64,
    /// `b = (1023 - 95) / 1023`.
    pub b: f64,
    /// `c = 95 / 1023`, the code value of linear 0.
    pub c: f64,
    /// Slope of the linear segment below the toe, `s = 7 ln2 2^(7 - 14c/b) / (a b)`.
    pub s: f64,
    /// Linear value where the log segment ends, `t = (2^(14 (-c/b) + 6) - 64) / a`.
    pub t: f64,
}

impl LogC4Constants {
    /// Evaluates the defining formulas of the LogC4 specification in f64.
    #[must_use]
    pub fn from_formulas() -> Self {
        let a: f64 = (f64::from(1_u32 << 18) - 16.0) / 117.45;
        let b: f64 = (1023.0 - 95.0) / 1023.0;
        let c: f64 = 95.0 / 1023.0;
        let s: f64 = 7.0 * std::f64::consts::LN_2 * (7.0 - 14.0 * c / b).exp2() / (a * b);
        let t: f64 = ((14.0 * (-c / b) + 6.0).exp2() - 64.0) / a;
        Self { a, b, c, s, t }
    }
}

/// LogC4 constants as literals (identical to [`LogC4Constants::from_formulas`], checked in tests).
pub const LOGC4: LogC4Constants = LogC4Constants {
    a: 2231.8263090676883,
    b: 0.9071358748778103,
    c: 0.09286412512218964,
    s: 0.1135972086105891,
    t: -0.01805699611991131,
};

/// The constants of ARRI LogC3 at exposure index 800 (ARRI LogC3 VFX document).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogC3Constants {
    /// Linear cut point between the linear segment and the log segment.
    pub cut: f64,
    /// Log segment gain inside the logarithm: `a`.
    pub a: f64,
    /// Log segment offset inside the logarithm: `b`.
    pub b: f64,
    /// Log segment scale `c`.
    pub c: f64,
    /// Log segment offset `d`.
    pub d: f64,
    /// Linear segment slope `e`.
    pub e: f64,
    /// Linear segment offset `f`.
    pub f: f64,
}

/// LogC3 constants for EI 800.
pub const LOGC3_EI800: LogC3Constants = LogC3Constants {
    cut: 0.010591,
    a: 5.555556,
    b: 0.052272,
    c: 0.247190,
    d: 0.385537,
    e: 5.367655,
    f: 0.092809,
};

/// Code value of scene-linear 0.18 grey in LogC4: `encode(0.18) = 0.2783958365...`.
pub const LOGC4_GRAY: f64 = 0.278_395_836_548_265_3;

/// One photographic stop in LogC4 code values on the log segment, `b / 14 = 0.06480` (`docs/02` section C9).
pub const LOGC4_STOP: f64 = LOGC4.b / 14.0;

/// Scene-linear middle grey, 18%.
pub const MIDDLE_GRAY: f64 = 0.18;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logc4_literals_match_the_formulas() {
        let f = LogC4Constants::from_formulas();
        for (lit, calc) in [
            (LOGC4.a, f.a),
            (LOGC4.b, f.b),
            (LOGC4.c, f.c),
            (LOGC4.s, f.s),
            (LOGC4.t, f.t),
        ] {
            assert!(
                (lit - calc).abs() <= 4.0 * f64::EPSILON * calc.abs(),
                "{lit} vs {calc}"
            );
        }
    }

    #[test]
    fn npm_maps_white_to_white() {
        for p in [AWG3, AWG4, AP0, REC709, DISPLAY_P3, REC2020] {
            let w = linalg::mul_vec(&p.rgb_to_xyz(), [1.0, 1.0, 1.0]);
            let expect = p.white_xyz();
            for k in 0..3 {
                assert!((w[k] - expect[k]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn cat02_maps_source_white_to_destination_white() {
        let src = xy_to_xyz(D65_XY);
        let dst = xy_to_xyz(ACES_WHITE_XY);
        let out = linalg::mul_vec(&cat02_adaptation(src, dst), src);
        for k in 0..3 {
            assert!((out[k] - dst[k]).abs() < 1e-14);
        }
    }
}
