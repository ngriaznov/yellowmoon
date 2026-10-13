// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! Primaries and 3x3 matrix helpers for the ACES 2 port.
//!
//! The CTL reference multiplies row vectors by matrices (`mult_f3_f33(v, M)` is `v * M`) and builds
//! the RGB to XYZ matrices from chromaticities with a closed-form expression. This module keeps
//! that convention so every formula in the port can be read next to the CTL line it came from.
//!
//! The gamuts are [`crate::consts::Primaries`] (one table of chromaticities for the crate: the
//! `Primaries::AP0`, `AP1`, `REC709`, `P3_D65` and `REC2020` names are associated constants there).
//! Only the CAM16 "sharpened primaries", which exist for this model alone, are defined here.
//! [`ctl_rgb_to_xyz`] is the CTL's closed form of the RGB to XYZ matrix (row-vector convention),
//! kept apart from `Primaries::rgb_to_xyz` (column vectors, by matrix inversion) so that every
//! formula of the port reads next to the CTL line it came from.

use crate::consts::Primaries;

/// A row vector of three values.
pub(crate) type V3 = [f64; 3];

/// A 3x3 matrix, row-major, used with row vectors (`v * M`).
pub(crate) type M33 = [[f64; 3]; 3];

/// CAM16 sharpened "primaries" (imaginary), used to build the cone response matrix.
pub(crate) const CAM16: Primaries = Primaries {
    red: [0.8336, 0.1735],
    green: [2.3854, -1.4659],
    blue: [0.087, -0.125],
    white: [0.333, 0.333],
};

/// RGB to XYZ matrix of `primaries` for white luminance `y` (row-vector convention:
/// `xyz = rgb * M`).
///
/// Same closed form as the CTL `RGBtoXYZ_f33`.
pub(crate) fn ctl_rgb_to_xyz(primaries: &Primaries, y: f64) -> M33 {
    let (r, g, b, w) = (
        primaries.red,
        primaries.green,
        primaries.blue,
        primaries.white,
    );
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

/// XYZ to RGB matrix of `primaries`, the inverse of [`ctl_rgb_to_xyz`].
pub(crate) fn ctl_xyz_to_rgb(primaries: &Primaries, y: f64) -> M33 {
    invert(&ctl_rgb_to_xyz(primaries, y))
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
        let m = ctl_rgb_to_xyz(&Primaries::REC709, 1.0);
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
            CAM16,
            Primaries::P3_D65,
            Primaries::REC2020,
        ] {
            let m = ctl_rgb_to_xyz(&p, 1.0);
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
        let m = ctl_rgb_to_xyz(&Primaries::P3_D65, 1.0);
        let xyz = mul_vec_mat([1.0, 1.0, 1.0], &m);
        let w = Primaries::P3_D65.white;
        assert!((xyz[1] - 1.0).abs() < 1e-12);
        assert!((xyz[0] - w[0] / w[1]).abs() < 1e-12);
        assert!((xyz[2] - (1.0 - w[0] - w[1]) / w[1]).abs() < 1e-12);
    }
}
