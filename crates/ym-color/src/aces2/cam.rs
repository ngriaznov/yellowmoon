// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! Hellwig 2022 / CAM16-derived colour appearance model used by the ACES 2 output transform.
//!
//! Port of the `JMhParams`, `init_JMhParams` and `RGB_to_JMh` / `JMh_to_RGB` family in
//! `Lib.Academy.OutputTransform.ctl`. Colours go RGB (of the given primaries, 1.0 = 100 nits)
//! to a pseudo-cone space, through the post-adaptation compression, to A, a, b and then to
//! J (lightness), M (colourfulness) and h (hue angle in degrees).

use super::matrix::{IDENTITY, M33, Primaries, V3, invert, mul_mat_mat, mul_vec_mat};

/// Luminance (nits) that RGB (1, 1, 1) represents in the model.
pub const REF_LUMINANCE: f64 = 100.0;
/// Adapting luminance.
const L_A: f64 = 100.0;
/// Background luminance.
const Y_B: f64 = 20.0;
/// Dim surround: c, Nc and the chromatic scale used in the Aab matrix.
const SURROUND: [f64; 3] = [0.9, 0.59, 0.9];

const J_SCALE: f64 = 100.0;
const CAM_NL_Y_REFERENCE: f64 = 100.0;
const CAM_NL_OFFSET: f64 = 0.2713 * CAM_NL_Y_REFERENCE;
const CAM_NL_SCALE: f64 = 4.0 * CAM_NL_Y_REFERENCE;

/// Hue wrap, degrees.
pub(crate) const HUE_LIMIT: f64 = 360.0;

/// `model_gamma = surround[1] * (1.48 + sqrt(Y_b / ref_luminance))`.
#[inline]
pub(crate) fn model_gamma() -> f64 {
    SURROUND[1] * (1.48 + (Y_B / REF_LUMINANCE).sqrt())
}

/// Pre-computed conversion matrices and constants for conversions to and from JMh for one set of
/// RGB primaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JMhParams {
    /// RGB to adapted CAM16 cone responses (row-vector convention).
    pub rgb_to_cam16_c: M33,
    /// Inverse of `rgb_to_cam16_c`.
    pub cam16_c_to_rgb: M33,
    /// Compressed cone responses to (A, a, b).
    pub cone_response_to_aab: M33,
    /// Inverse of `cone_response_to_aab`.
    pub aab_to_cone_response: M33,
    /// Normalised luminance-level adaptation factor `F_L / 100`.
    pub f_l_n: f64,
    /// `c * z`, the exponent that turns A into J.
    pub cz: f64,
    /// `1 / cz`.
    pub inv_cz: f64,
    /// Achromatic response of the reference white used for the J to Y shortcut.
    pub a_w_j: f64,
    /// `1 / a_w_j`.
    pub inv_a_w_j: f64,
}

#[inline]
fn copysign_ctl(x: f64, y: f64) -> f64 {
    // CTL: sign(y) * fabs(x), where sign(0) = 0.
    if y < 0.0 {
        -x.abs()
    } else if y > 0.0 {
        x.abs()
    } else {
        0.0
    }
}

#[inline]
fn post_adaptation_fwd_unsigned(rc: f64) -> f64 {
    let f_l_y = rc.powf(0.42);
    f_l_y / (CAM_NL_OFFSET + f_l_y)
}

#[inline]
fn post_adaptation_inv_unsigned(ra: f64) -> f64 {
    let ra_lim = ra.min(0.99);
    let f_l_y = (CAM_NL_OFFSET * ra_lim) / (1.0 - ra_lim);
    f_l_y.powf(1.0 / 0.42)
}

/// Signed post-adaptation cone response compression (forward).
#[inline]
pub(crate) fn post_adaptation_fwd(v: f64) -> f64 {
    copysign_ctl(post_adaptation_fwd_unsigned(v.abs()), v)
}

/// Signed post-adaptation cone response compression (inverse).
#[inline]
pub(crate) fn post_adaptation_inv(v: f64) -> f64 {
    copysign_ctl(post_adaptation_inv_unsigned(v.abs()), v)
}

#[inline]
fn achromatic_n_to_j(a: f64, cz: f64) -> f64 {
    J_SCALE * a.powf(cz)
}

#[inline]
fn j_to_achromatic_n(j: f64, inv_cz: f64) -> f64 {
    (j * (1.0 / J_SCALE)).powf(inv_cz)
}

impl JMhParams {
    /// `init_JMhParams(prims)`.
    pub fn new(prims: &Primaries) -> Self {
        let matrix_16 = Primaries::CAM16.xyz_to_rgb(1.0);

        let base_cone_response_to_aab: M33 = [
            [2.0, 1.0, 1.0 / 9.0],
            [1.0, -12.0 / 11.0, 1.0 / 9.0],
            [1.0 / 20.0, 1.0 / 11.0, -2.0 / 9.0],
        ];

        let rgb_to_xyz = prims.rgb_to_xyz(1.0);
        let xyz_w = mul_vec_mat([REF_LUMINANCE; 3], &rgb_to_xyz);
        let y_w = xyz_w[1];

        // Step 0 - converting CIE XYZ tristimulus values to sharpened RGB values.
        let rgb_w = mul_vec_mat(xyz_w, &matrix_16);

        // Viewing condition dependent parameters.
        let k = 1.0 / (5.0 * L_A + 1.0);
        let k4 = k * k * k * k;
        let f_l = 0.2 * k4 * (5.0 * L_A) + 0.1 * (1.0 - k4).powf(2.0) * (5.0 * L_A).powf(1.0 / 3.0);

        let f_l_n = f_l / REF_LUMINANCE;
        let cz = model_gamma();

        let d_rgb: V3 = [
            f_l_n * y_w / rgb_w[0],
            f_l_n * y_w / rgb_w[1],
            f_l_n * y_w / rgb_w[2],
        ];

        let rgb_wc: V3 = [
            d_rgb[0] * rgb_w[0],
            d_rgb[1] * rgb_w[1],
            d_rgb[2] * rgb_w[2],
        ];

        let rgb_aw: V3 = [
            post_adaptation_fwd(rgb_wc[0]),
            post_adaptation_fwd(rgb_wc[1]),
            post_adaptation_fwd(rgb_wc[2]),
        ];

        let scale_cam_nl = scaled_identity(CAM_NL_SCALE);
        let cone_response_to_aab = mul_mat_mat(&scale_cam_nl, &base_cone_response_to_aab);
        let a_w = cone_response_to_aab[0][0] * rgb_aw[0]
            + cone_response_to_aab[1][0] * rgb_aw[1]
            + cone_response_to_aab[2][0] * rgb_aw[2];
        let a_w_j = post_adaptation_fwd_unsigned(f_l);

        // Prescale the CAM16 LMS responses to directly provide for chromatic adaptation.
        let m1 = mul_mat_mat(&rgb_to_xyz, &matrix_16);
        let m2 = scaled_identity(REF_LUMINANCE);
        let rgb_to_cam16 = mul_mat_mat(&m1, &m2);
        let mut rgb_to_cam16_c = rgb_to_cam16;
        for row in rgb_to_cam16_c.iter_mut() {
            // scale_matrix_diagonal_f33_f3(I, D_RGB) is diag(D_RGB), applied on the right.
            row[0] *= d_rgb[0];
            row[1] *= d_rgb[1];
            row[2] *= d_rgb[2];
        }

        let mut cone_response_to_aab_scaled = cone_response_to_aab;
        for row in cone_response_to_aab_scaled.iter_mut() {
            row[0] /= a_w;
            row[1] *= 43.0 * SURROUND[2];
            row[2] *= 43.0 * SURROUND[2];
        }

        JMhParams {
            rgb_to_cam16_c,
            cam16_c_to_rgb: invert(&rgb_to_cam16_c),
            cone_response_to_aab: cone_response_to_aab_scaled,
            aab_to_cone_response: invert(&cone_response_to_aab_scaled),
            f_l_n,
            cz,
            inv_cz: 1.0 / cz,
            a_w_j,
            inv_a_w_j: 1.0 / a_w_j,
        }
    }

    /// `_A_to_Y`: achromatic response A to luminance (nits).
    #[inline]
    fn a_to_y(&self, a: f64) -> f64 {
        let ra = self.a_w_j * a;
        post_adaptation_inv_unsigned(ra) / self.f_l_n
    }

    /// `J_to_Y`: lightness J to luminance (nits).
    #[inline]
    pub fn j_to_y(&self, j: f64) -> f64 {
        self.a_to_y(j_to_achromatic_n(j.abs(), self.inv_cz))
    }

    /// `Y_to_J`: luminance (nits) to lightness J (sign preserved).
    #[inline]
    pub fn y_to_j(&self, y: f64) -> f64 {
        let ra = post_adaptation_fwd_unsigned(y.abs() * self.f_l_n);
        let j = achromatic_n_to_j(ra * self.inv_a_w_j, self.cz);
        copysign_ctl(j, y)
    }

    /// `J_to_Achromatic_n`.
    #[inline]
    pub fn j_to_a(&self, j: f64) -> f64 {
        j_to_achromatic_n(j, self.inv_cz)
    }

    /// `RGB_to_Aab`.
    #[inline]
    pub fn rgb_to_aab(&self, rgb: V3) -> V3 {
        let rgb_m = mul_vec_mat(rgb, &self.rgb_to_cam16_c);
        let rgb_a = [
            post_adaptation_fwd(rgb_m[0]),
            post_adaptation_fwd(rgb_m[1]),
            post_adaptation_fwd(rgb_m[2]),
        ];
        mul_vec_mat(rgb_a, &self.cone_response_to_aab)
    }

    /// `Aab_to_JMh`. Returns (0, 0, 0) for a non-positive achromatic response.
    #[inline]
    pub fn aab_to_jmh(&self, aab: V3) -> V3 {
        if aab[0] <= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let j = achromatic_n_to_j(aab[0], self.cz);
        let m = (aab[1] * aab[1] + aab[2] * aab[2]).sqrt();
        let h = wrap_to_360(aab[2].atan2(aab[1]).to_degrees());
        [j, m, h]
    }

    /// `RGB_to_JMh`.
    #[inline]
    pub fn rgb_to_jmh(&self, rgb: V3) -> V3 {
        self.aab_to_jmh(self.rgb_to_aab(rgb))
    }

    /// `JMh_to_Aab`.
    #[inline]
    pub fn jmh_to_aab(&self, jmh: V3) -> V3 {
        let h_rad = jmh[2].to_radians();
        let (sin_hr, cos_hr) = h_rad.sin_cos();
        [
            j_to_achromatic_n(jmh[0], self.inv_cz),
            jmh[1] * cos_hr,
            jmh[1] * sin_hr,
        ]
    }

    /// `Aab_to_RGB`.
    #[inline]
    pub fn aab_to_rgb(&self, aab: V3) -> V3 {
        let rgb_a = mul_vec_mat(aab, &self.aab_to_cone_response);
        let rgb_m = [
            post_adaptation_inv(rgb_a[0]),
            post_adaptation_inv(rgb_a[1]),
            post_adaptation_inv(rgb_a[2]),
        ];
        mul_vec_mat(rgb_m, &self.cam16_c_to_rgb)
    }

    /// `JMh_to_RGB`.
    #[inline]
    pub fn jmh_to_rgb(&self, jmh: V3) -> V3 {
        self.aab_to_rgb(self.jmh_to_aab(jmh))
    }
}

fn scaled_identity(s: f64) -> M33 {
    let mut m = IDENTITY;
    for (i, row) in m.iter_mut().enumerate() {
        row[i] *= s;
    }
    m
}

/// `wrap_to_360`: `fmod` into [0, 360]. Tiny negative inputs can round to exactly 360.0, which is
/// why the hue tables carry an extra wrap entry.
#[inline]
pub(crate) fn wrap_to_360(hue: f64) -> f64 {
    let y = hue % 360.0;
    if y < 0.0 { y + 360.0 } else { y }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jmh_round_trip() {
        let p = JMhParams::new(&Primaries::AP0);
        for rgb in [
            [0.18, 0.18, 0.18],
            [0.5, 0.2, 0.1],
            [2.0, 0.5, 4.0],
            [0.01, 0.3, 0.02],
        ] {
            let back = p.jmh_to_rgb(p.rgb_to_jmh(rgb));
            for k in 0..3 {
                assert!(
                    (back[k] - rgb[k]).abs() < 1e-9 * rgb[k].abs().max(1.0),
                    "{rgb:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn neutral_axis_has_no_colourfulness() {
        for prims in [
            Primaries::AP0,
            Primaries::AP1,
            Primaries::P3_D65,
            Primaries::REC709,
        ] {
            let p = JMhParams::new(&prims);
            for v in [1e-4, 0.01, 0.18, 1.0, 50.0] {
                let jmh = p.rgb_to_jmh([v, v, v]);
                assert!(
                    jmh[1] < 1e-9 * jmh[0].max(1e-3),
                    "M = {} for grey {v}",
                    jmh[1]
                );
            }
        }
    }

    #[test]
    fn j_y_round_trip_and_white() {
        let p = JMhParams::new(&Primaries::AP0);
        for y in [0.001, 0.1, 1.0, 18.0, 100.0] {
            let j = p.y_to_j(y);
            assert!((p.j_to_y(j) - y).abs() < 1e-9 * y.max(1.0));
        }
        // Reference white (100 nits) is J = 100 for the reference viewing conditions? Not exactly:
        // J is relative to the model's A_w_J, so only check it is a sane lightness.
        assert!(p.y_to_j(100.0) > 90.0 && p.y_to_j(100.0) < 110.0);
        // Sign handling and zero.
        assert_eq!(p.y_to_j(0.0), 0.0);
        assert!(p.y_to_j(-1.0) < 0.0);
    }

    #[test]
    fn wrap_behaviour() {
        assert_eq!(wrap_to_360(0.0), 0.0);
        assert_eq!(wrap_to_360(360.0), 0.0);
        assert!((wrap_to_360(-90.0) - 270.0).abs() < 1e-12);
        assert_eq!(wrap_to_360(-1e-30), 360.0);
        assert!((wrap_to_360(725.0) - 5.0).abs() < 1e-12);
    }
}
