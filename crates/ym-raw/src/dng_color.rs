//! The DNG colour model: the camera's matrices at a colour temperature, and the as-shot white.
//!
//! Written from the DNG Specification 1.7, chapter 6 ("Mapping Camera Color Space to CIE XYZ
//! Space"). DNG writes up to two calibrations, `ColorMatrix1` and `2` for the illuminants
//! `CalibrationIlluminant1` and `2`; a matrix for another light is the mired-linear blend of the
//! two (clamped outside their range). The camera's own space is `AnalogBalance x CameraCalibration
//! x ColorMatrix x XYZ`, which is what the raw samples are in.

use crate::dng::DngInfo;

/// A 3 x 3 matrix, row by row.
pub type Mat3 = [[f64; 3]; 3];

/// CIE 1931 XYZ of the D65 white at Y = 1.
pub const XYZ_D65: [f64; 3] = [0.950_47, 1.0, 1.088_83];

/// The correlated colour temperature of the D65 white, in kelvin.
pub const D65_CCT_K: f64 = 6504.0;

/// Reshapes nine numbers (row by row) into a matrix.
pub fn to_mat(v: &[f64; 9]) -> Mat3 {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

/// `a * b`.
pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// `m * v`.
pub fn mul_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// The inverse of `m`, or `None` for a matrix that is singular or nearly so.
pub fn inverse(m: &Mat3) -> Option<Mat3> {
    let c = |r: usize, k: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
        m[r1][k1] * m[r2][k2] - m[r1][k2] * m[r2][k1]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    let scale = m.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
    if !det.is_finite() || det.abs() <= 1e-12 * scale.powi(3).max(f64::MIN_POSITIVE) {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = c(j, i) / det;
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(out)
}

/// The colour temperature, in kelvin, of an EXIF `LightSource` code as the DNG specification uses it
/// for `CalibrationIlluminant`. `None` for codes that name no temperature (unknown, other).
pub fn illuminant_cct_k(code: u16) -> Option<f64> {
    Some(match code {
        1 => 5500.0,     // daylight
        2 => 4200.0,     // fluorescent
        3 => 2850.0,     // tungsten
        4 => 5500.0,     // flash
        9 => 5500.0,     // fine weather
        10 => 6500.0,    // cloudy
        11 => 7500.0,    // shade
        12 => 6430.0,    // daylight fluorescent
        13 => 5000.0,    // day white fluorescent
        14 => 4200.0,    // cool white fluorescent
        15 => 3450.0,    // white fluorescent
        17 => 2856.0,    // standard light A
        18 => 4874.0,    // standard light B
        19 => 6774.0,    // standard light C
        20 => 5503.0,    // D55
        21 => D65_CCT_K, // D65
        22 => 7504.0,    // D75
        23 => 5003.0,    // D50
        24 => 3200.0,    // ISO studio tungsten
        _ => return None,
    })
}

/// The weight of calibration 1 for a light of `cct_k`: 1 at or below the first illuminant, 0 at or
/// above the second, linear in mired in between (DNG Specification, "Camera Calibration
/// Interpolation"). `None` when one calibration is missing or the two have no temperature.
fn weight_of_first(info: &DngInfo, cct_k: f64) -> Option<f64> {
    let t1 = illuminant_cct_k(info.calibration_illuminant[0]?)?;
    let t2 = illuminant_cct_k(info.calibration_illuminant[1]?)?;
    if (t1 - t2).abs() < 1.0 {
        return Some(1.0);
    }
    // `low` is the warmer (lower temperature) light.
    let (low, high, first_is_low) = if t1 < t2 {
        (t1, t2, true)
    } else {
        (t2, t1, false)
    };
    let w_low = if cct_k <= low {
        1.0
    } else if cct_k >= high {
        0.0
    } else {
        ((1.0 / cct_k) - (1.0 / high)) / ((1.0 / low) - (1.0 / high))
    };
    Some(if first_is_low { w_low } else { 1.0 - w_low })
}

fn blend(a: &Mat3, b: &Mat3, w_a: f64) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = w_a * a[i][j] + (1.0 - w_a) * b[i][j];
        }
    }
    out
}

/// One matrix pair blended for `cct_k`; the one that exists when only one does.
fn blended(
    first: Option<[f64; 9]>,
    second: Option<[f64; 9]>,
    info: &DngInfo,
    cct_k: f64,
) -> Option<Mat3> {
    match (first, second) {
        (Some(a), Some(b)) => match weight_of_first(info, cct_k) {
            Some(w) => Some(blend(&to_mat(&a), &to_mat(&b), w)),
            None => Some(to_mat(&a)),
        },
        (Some(a), None) => Some(to_mat(&a)),
        (None, Some(b)) => Some(to_mat(&b)),
        (None, None) => None,
    }
}

/// The matrix from CIE XYZ to the camera's own space for a light of `cct_k`:
/// `AnalogBalance x CameraCalibration x ColorMatrix` with the two calibrations blended.
/// `None` when the file has no `ColorMatrix`.
pub fn xyz_to_camera_at(info: &DngInfo, cct_k: f64) -> Option<Mat3> {
    let cm = blended(info.color_matrix[0], info.color_matrix[1], info, cct_k)?;
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let cc = blended(
        info.camera_calibration[0],
        info.camera_calibration[1],
        info,
        cct_k,
    )
    .unwrap_or(identity);
    let ab = info.analog_balance.map_or(identity, |a| {
        [[a[0], 0.0, 0.0], [0.0, a[1], 0.0], [0.0, 0.0, a[2]]]
    });
    Some(mul(&ab, &mul(&cc, &cm)))
}

/// The matrix from XYZ to the camera's own space for D65 light (the target of the mapping to the
/// A7R IV): the blend of the calibrations at 6504 K, which is `ColorMatrix` itself when one of the
/// illuminants is D65.
pub fn xyz_to_camera_d65(info: &DngInfo) -> Option<Mat3> {
    xyz_to_camera_at(info, D65_CCT_K)
}

/// CIE XYZ at Y = 1 of the chromaticity `xy`.
pub fn xy_to_xyz(xy: [f64; 2]) -> [f64; 3] {
    [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]]
}

/// The correlated colour temperature of a chromaticity, by McCamy's approximation.
pub fn cct_of_xy(xy: [f64; 2]) -> f64 {
    let n = (xy[0] - 0.3320) / (0.1858 - xy[1]);
    (449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33).clamp(1500.0, 25000.0)
}

/// The camera's response to the as-shot white, in the camera's own space (the DNG's
/// `AsShotNeutral`): the tag when the file has it, else the response to the white of
/// `AsShotWhiteXY` through the calibration blended for that white. The result is scaled so that
/// its largest value is 1. `None` when the file has neither.
pub fn as_shot_neutral(info: &DngInfo) -> Option<[f64; 3]> {
    let neutral = match (info.as_shot_neutral, info.as_shot_white_xy) {
        (Some(n), _) => n,
        (None, Some(xy)) => {
            let m = xyz_to_camera_at(info, cct_of_xy(xy))?;
            mul_vec(&m, xy_to_xyz(xy))
        }
        (None, None) => return None,
    };
    let top = neutral.iter().copied().fold(0.0, f64::max);
    (top.is_finite() && top > 0.0 && neutral.iter().all(|v| v.is_finite() && *v > 0.0))
        .then(|| neutral.map(|v| v / top))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dng::DngKind;

    fn info(cm1: Option<[f64; 9]>, cm2: Option<[f64; 9]>, ill: [Option<u16>; 2]) -> DngInfo {
        DngInfo {
            version: [1, 4, 0, 0],
            kind: DngKind::Cfa,
            big_endian: false,
            unique_camera_model: None,
            color_matrix: [cm1, cm2],
            calibration_illuminant: ill,
            camera_calibration: [None, None],
            analog_balance: None,
            as_shot_neutral: None,
            as_shot_white_xy: None,
            forward_matrix: [None, None],
            noise_profile: None,
            linearization: None,
            black_delta_h: Vec::new(),
            black_delta_v: Vec::new(),
            gain_maps: Vec::new(),
        }
    }

    const A: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    const B: [f64; 9] = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];

    #[test]
    fn the_d65_matrix_is_the_d65_calibration_when_there_is_one() {
        let i = info(Some(A), Some(B), [Some(17), Some(21)]);
        assert_eq!(xyz_to_camera_d65(&i), Some(to_mat(&B)));
        let swapped = info(Some(B), Some(A), [Some(21), Some(17)]);
        assert_eq!(xyz_to_camera_d65(&swapped), Some(to_mat(&B)));
    }

    #[test]
    fn calibrations_blend_in_mired_and_clamp_outside_their_range() {
        let i = info(Some(A), Some(B), [Some(17), Some(21)]);
        let at = |k: f64| xyz_to_camera_at(&i, k).expect("matrix")[0][0];
        assert_eq!(at(2000.0), 1.0);
        assert_eq!(at(9000.0), 2.0);
        // Halfway in mired between 2856 K and 6504 K.
        let mid = 2.0 / (1.0 / 2856.0 + 1.0 / 6504.0);
        assert!((at(mid) - 1.5).abs() < 1e-9, "{}", at(mid));
    }

    #[test]
    fn one_matrix_serves_every_light() {
        let i = info(Some(B), None, [Some(21), None]);
        assert_eq!(xyz_to_camera_at(&i, 3000.0), Some(to_mat(&B)));
        assert_eq!(
            xyz_to_camera_at(&info(None, None, [None, None]), 3000.0),
            None
        );
    }

    #[test]
    fn inverse_undoes_the_matrix() {
        let m = [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.3, 0.8]];
        let product = mul(&m, &inverse(&m).expect("invertible"));
        for (i, row) in product.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!((v - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
        assert!(inverse(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]).is_none());
    }

    #[test]
    fn the_neutral_comes_from_the_tag_or_from_the_white_point() {
        let mut i = info(Some(A), None, [Some(21), None]);
        i.as_shot_neutral = Some([0.5, 1.0, 0.25]);
        assert_eq!(as_shot_neutral(&i), Some([0.5, 1.0, 0.25]));
        i.as_shot_neutral = None;
        assert_eq!(as_shot_neutral(&i), None);
        // The D65 white through an identity matrix is XYZ itself, scaled to a largest value of 1.
        i.as_shot_white_xy = Some([0.3127, 0.3290]);
        let n = as_shot_neutral(&i).expect("neutral");
        let xyz = xy_to_xyz([0.3127, 0.3290]);
        let top = xyz.iter().copied().fold(0.0, f64::max);
        for k in 0..3 {
            assert!((n[k] - xyz[k] / top).abs() < 1e-12);
        }
    }

    #[test]
    fn mccamy_puts_d65_near_6500_kelvin() {
        let cct = cct_of_xy([0.3127, 0.3290]);
        assert!((cct - 6504.0).abs() < 60.0, "{cct}");
    }
}
