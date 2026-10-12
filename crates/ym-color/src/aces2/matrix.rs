// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! Primaries and 3x3 matrix helpers for the ACES 2 port.
//!
//! The CTL reference multiplies row vectors by matrices (`mult_f3_f33(v, M)` is `v * M`) and builds
//! the RGB to XYZ matrices from chromaticities with a closed-form expression. This module keeps
//! that convention so every formula in the port can be read next to the CTL line it came from.
//!
//! NOTE: the colour-core code in `ym-color` (gamut matrices) will hold the same primaries; the
//! constants here are kept local so the ACES 2 module has no dependency on files that were written
//! concurrently. Deduplicate once both exist (the values are the published chromaticities).

/// A row vector of three values.
pub(crate) type V3 = [f64; 3];

/// A 3x3 matrix, row-major, used with row vectors (`v * M`).
pub(crate) type M33 = [[f64; 3]; 3];

/// Chromaticities of the red, green and blue primaries and the white point (CIE 1931 xy).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Primaries {
    /// Red primary xy.
    pub red: [f64; 2],
    /// Green primary xy.
    pub green: [f64; 2],
    /// Blue primary xy.
    pub blue: [f64; 2],
    /// White point xy.
    pub white: [f64; 2],
}

impl Primaries {
    /// ACES AP0 primaries (SMPTE ST 2065-1), ACES white.
    pub const AP0: Primaries = Primaries {
        red: [0.73470, 0.26530],
        green: [0.00000, 1.00000],
        blue: [0.00010, -0.07700],
        white: [0.32168, 0.33767],
    };

    /// ACES AP1 primaries, ACES white. These are also the "reach" primaries of the output transform.
    pub const AP1: Primaries = Primaries {
        red: [0.713, 0.293],
        green: [0.165, 0.830],
        blue: [0.128, 0.044],
        white: [0.32168, 0.33767],
    };

    /// CAM16 sharpened "primaries" (imaginary), used to build the cone response matrix.
    pub(crate) const CAM16: Primaries = Primaries {
        red: [0.8336, 0.1735],
        green: [2.3854, -1.4659],
        blue: [0.087, -0.125],
        white: [0.333, 0.333],
    };

    /// Rec.709 / sRGB primaries, D65 white.
    pub const REC709: Primaries = Primaries {
        red: [0.6400, 0.3300],
        green: [0.3000, 0.6000],
        blue: [0.1500, 0.0600],
        white: [0.3127, 0.3290],
    };

    /// P3 primaries with D65 white (Display P3).
    pub const P3_D65: Primaries = Primaries {
        red: [0.6800, 0.3200],
        green: [0.2650, 0.6900],
        blue: [0.1500, 0.0600],
        white: [0.3127, 0.3290],
    };

    /// Rec.2020 / Rec.2100 primaries, D65 white.
    pub const REC2020: Primaries = Primaries {
        red: [0.7080, 0.2920],
        green: [0.1700, 0.7970],
        blue: [0.1310, 0.0460],
        white: [0.3127, 0.3290],
    };

    /// RGB to XYZ matrix for white luminance `y` (row-vector convention: `xyz = rgb * M`).
    ///
    /// Same closed form as the CTL `RGBtoXYZ_f33`.
    pub(crate) fn rgb_to_xyz(&self, y: f64) -> M33 {
        let (r, g, b, w) = (self.red, self.green, self.blue, self.white);
        let x_w = w[0] * y / w[1];
        let z_w = (1.0 - w[0] - w[1]) * y / w[1];

        let d = r[0] * (b[1] - g[1]) + b[0] * (g[1] - r[1]) + g[0] * (r[1] - b[1]);

        let sr = (x_w * (b[1] - g[1]) - g[0] * (y * (b[1] - 1.0) + b[1] * (x_w + z_w))
            + b[0] * (y * (g[1] - 1.0) + g[1] * (x_w + z_w)))
            / d;
        let sg = (x_w * (r[1] - b[1]) + r[0] * (y * (b[1] - 1.0) + b[1] * (x_w + z_w))
            - b[0] * (y * (r[1] - 1.0) + r[1] * (x_w + z_w)))
            / d;
        let sb = (x_w * (g[1] - r[1]) - r[0] * (y * (g[1] - 1.0) + g[1] * (x_w + z_w))
            + g[0] * (y * (r[1] - 1.0) + r[1] * (x_w + z_w)))
            / d;

        [
            [sr * r[0], sr * r[1], sr * (1.0 - r[0] - r[1])],
            [sg * g[0], sg * g[1], sg * (1.0 - g[0] - g[1])],
            [sb * b[0], sb * b[1], sb * (1.0 - b[0] - b[1])],
        ]
    }

    /// XYZ to RGB matrix, the inverse of [`Primaries::rgb_to_xyz`].
    pub(crate) fn xyz_to_rgb(&self, y: f64) -> M33 {
        invert(&self.rgb_to_xyz(y))
    }
}

/// Row vector times matrix (CTL `mult_f3_f33`).
#[inline]
pub(crate) fn mul_vec_mat(v: V3, m: &M33) -> V3 {
    [
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    ]
}

/// Matrix times matrix (CTL `mult_f33_f33`): `a * b`.
pub(crate) fn mul_mat_mat(a: &M33, b: &M33) -> M33 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Matrix inverse by the adjugate. The matrices used here are all well conditioned.
pub(crate) fn invert(m: &M33) -> M33 {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let inv_det = 1.0 / det;
    [
        [
            c00 * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            c01 * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            c02 * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

/// The 3x3 identity.
pub(crate) const IDENTITY: M33 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rec709_matrix_matches_the_published_one() {
        // Column-vector sRGB to XYZ (D65): [0.4124 0.3576 0.1805; 0.2126 0.7152 0.0722; 0.0193 0.1192 0.9505].
        // Row-vector convention: this is the transpose.
        let m = Primaries::REC709.rgb_to_xyz(1.0);
        let expected = [
            [0.4123907993, 0.2126390059, 0.0193308187],
            [0.3575843394, 0.7151686788, 0.1191947798],
            [0.1804807884, 0.0721923154, 0.9505321522],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - expected[i][j]).abs() < 1e-6,
                    "[{i}][{j}] {} vs {}",
                    m[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        for p in [
            Primaries::AP0,
            Primaries::AP1,
            Primaries::CAM16,
            Primaries::P3_D65,
            Primaries::REC2020,
        ] {
            let m = p.rgb_to_xyz(1.0);
            let prod = mul_mat_mat(&m, &invert(&m));
            for (i, row) in prod.iter().enumerate() {
                for (j, &cell) in row.iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((cell - want).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn white_maps_to_the_white_point() {
        let m = Primaries::P3_D65.rgb_to_xyz(1.0);
        let xyz = mul_vec_mat([1.0, 1.0, 1.0], &m);
        let w = Primaries::P3_D65.white;
        assert!((xyz[1] - 1.0).abs() < 1e-12);
        assert!((xyz[0] - w[0] / w[1]).abs() < 1e-12);
        assert!((xyz[2] - (1.0 - w[0] - w[1]) / w[1]).abs() < 1e-12);
    }
}
