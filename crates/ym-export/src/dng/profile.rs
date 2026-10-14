//! The colour profile a Yellowmoon DNG carries (`docs/09-export.md` section DNG, Profile).
//!
//! The pixels are scene-linear AWG4 with a D65 white, so the camera "native" space of the DNG is AWG4:
//!
//! * `ColorMatrix1` is `inv(AWG4 to XYZ)`: XYZ (D65) to AWG4, with `CalibrationIlluminant1 = 21` (D65).
//! * `ForwardMatrix1` is `Bradford(D65 to D50) * (AWG4 to XYZ)`, scaled per row so that camera white
//!   (1, 1, 1) maps to the D50 white exactly, as the DNG specification requires (XYZ D50 out).
//! * `AsShotNeutral` is (1, 1, 1).
//!
//! The tone curve and the look table are inputs: `tools/dngfit` fits them to the ACES 2 display
//! transform (`docs/09`) and the product embeds the result ([`super::fitted::FittedProfile`]);
//! [`DngProfile::identity`] writes identities, for structure tests.

use std::ops::RangeInclusive;

use ym_color::consts::{D65_XY, Mat3, Vec3, linalg, xy_to_xyz};
use ym_color::gamut;

use crate::icc::{D50_PCS, bradford_adaptation};

/// `ColorMatrix1`: XYZ (D65) to linear AWG4.
#[must_use]
pub fn color_matrix_1() -> Mat3 {
    gamut::matrices().xyz_to_awg4
}

/// `ForwardMatrix1`: linear AWG4 (white balanced, which AWG4 already is) to XYZ D50.
///
/// Bradford adaptation from the D65 white to the D50 PCS white, times AWG4 to XYZ, with each row
/// scaled so that (1, 1, 1) maps to the D50 white to f64 precision.
#[must_use]
pub fn forward_matrix_1() -> Mat3 {
    let adapt = bradford_adaptation(xy_to_xyz(D65_XY), D50_PCS);
    let mut m = linalg::mul(&adapt, &gamut::matrices().awg4_to_xyz);
    let white = linalg::mul_vec(&m, [1.0, 1.0, 1.0]);
    for (row, (&w, &target)) in m.iter_mut().zip(white.iter().zip(&D50_PCS)) {
        let scale = target / w;
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
    m
}

/// Writes a matrix as nine row-major `SRATIONAL` values with a denominator of 10^7 (a rounding error of
/// 5e-8 per entry).
#[must_use]
pub fn srational_matrix(m: &Mat3) -> Vec<tiff::encoder::SRational> {
    m.iter()
        .flatten()
        .map(|&v| tiff::encoder::SRational {
            n: (v * 1e7).round() as i32,
            d: 10_000_000,
        })
        .collect()
}

/// The `ProfileLookTable`: hue/saturation/value modifiers (`docs/09`, `research/notes/dng-export.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct LookTable {
    /// `[hue divisions, saturation divisions, value divisions]`. The DNG SDK caps these at 36 x 32 x 16.
    pub dims: [u32; 3],
    /// For each entry (value-major, then hue, then saturation): hue shift in degrees, saturation
    /// scale, value scale. `dims[0] * dims[1] * dims[2] * 3` floats.
    pub data: Vec<f32>,
    /// `ProfileLookTableEncoding`: 0 linear, 1 sRGB.
    pub encoding: u32,
}

impl LookTable {
    /// The identity table: no hue shift, saturation and value scales of 1. It has two value divisions:
    /// with one division and sRGB encoding Adobe's reference renderer does not return the input
    /// (see [`LookTable::validate`]).
    #[must_use]
    pub fn identity() -> Self {
        let dims = [2, 2, 2];
        let entries = (dims[0] * dims[1] * dims[2]) as usize;
        let mut data = Vec::with_capacity(entries * 3);
        for _ in 0..entries {
            data.extend_from_slice(&[0.0, 1.0, 1.0]);
        }
        Self {
            dims,
            data,
            encoding: 1,
        }
    }

    /// Checks the table against the limits of the DNG SDK.
    ///
    /// A table with a single value division (a "2.5D" table) must use linear encoding (0): with
    /// `ProfileLookTableEncoding` 1 (sRGB) Adobe's reference renderer (`dng_validate`, DNG SDK 1.7.1)
    /// changes the brightness even of an identity table (mid grey 0.18 comes out as 0.027), so such a
    /// profile would not render as fitted.
    pub fn validate(&self) -> Result<(), String> {
        let [h, s, v] = self.dims;
        if h < 1 || s < 2 || v < 1 {
            return Err(format!(
                "the look table needs at least 1 x 2 x 1 divisions, not {h} x {s} x {v}"
            ));
        }
        if h > 36 || s > 32 || v > 16 {
            return Err(format!(
                "the look table is limited to 36 x 32 x 16 divisions, not {h} x {s} x {v}"
            ));
        }
        let expected = (h * s * v * 3) as usize;
        if self.data.len() != expected {
            return Err(format!(
                "the look table needs {expected} values, got {}",
                self.data.len()
            ));
        }
        if self.encoding > 1 {
            return Err("the look table encoding must be 0 (linear) or 1 (sRGB)".to_owned());
        }
        if v == 1 && self.encoding == 1 {
            return Err(
                "a look table with one value division must use linear encoding (0): Adobe's renderer mishandles the sRGB encoding of 2.5D tables"
                    .to_owned(),
            );
        }
        if self.data.iter().any(|v| !v.is_finite()) {
            return Err("the look table has a value that isn't a number".to_owned());
        }
        // Entries are (hue shift, saturation scale, value scale), value-major, then hue, then saturation.
        for (i, entry) in self.data.chunks_exact(3).enumerate() {
            if entry[1] < 0.0 || entry[2] < 0.0 {
                return Err(format!(
                    "look table entry {i} has a negative saturation or value scale"
                ));
            }
            // The DNG specification: "All zero input saturation entries are required to have a value
            // scale factor of 1.0" (a grey can't be re-toned by a look table; the tone curve does that).
            if i % s as usize == 0 && (entry[2] - 1.0).abs() > ZERO_SATURATION_VALUE_TOLERANCE {
                return Err(format!(
                    "look table entry {i} is at zero saturation, where the value scale must be 1.0, not {}",
                    entry[2]
                ));
            }
        }
        Ok(())
    }
}

/// How far from 1.0 the value scale of a zero-saturation look table entry may be (float rounding).
const ZERO_SATURATION_VALUE_TOLERANCE: f32 = 1e-4;

/// The `BaselineExposure` offsets a fitted profile may carry, in EV (a tone curve that covers scene values from
/// `2^-4` up to `2^12` at the most).
pub const FIT_OFFSET_RANGE_EV: RangeInclusive<f64> = -12.0..=4.0;

/// The fitted parts of the profile. The same for every photo; computed once by `tools/dngfit`.
#[derive(Debug, Clone, PartialEq)]
pub struct DngProfile {
    /// `ProfileToneCurve` as (input, output) pairs in [0, 1], increasing input.
    pub tone_curve: Vec<[f32; 2]>,
    /// The `ProfileLookTable`.
    pub look_table: LookTable,
}

impl DngProfile {
    /// An identity tone curve and look table, for use until `tools/dngfit` has fitted the real ones.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            tone_curve: vec![[0.0, 0.0], [1.0, 1.0]],
            look_table: LookTable::identity(),
        }
    }

    /// Checks the curve (2 or more points, inputs strictly increasing in [0, 1], outputs in [0, 1]) and the table.
    pub fn validate(&self) -> Result<(), String> {
        let c = &self.tone_curve;
        if c.len() < 2 {
            return Err("the tone curve needs at least two points".to_owned());
        }
        for (i, p) in c.iter().enumerate() {
            if !(0.0..=1.0).contains(&p[0]) || !(0.0..=1.0).contains(&p[1]) {
                return Err(format!("tone curve point {i} is outside [0, 1]"));
            }
            if i > 0 && p[0] <= c[i - 1][0] {
                return Err(format!("tone curve inputs must increase (point {i})"));
            }
        }
        self.look_table.validate()
    }
}

impl Default for DngProfile {
    fn default() -> Self {
        Self::identity()
    }
}

/// How `BaselineExposure` is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaselineExposure {
    /// `log2(x_white) + fit_offset_ev + placement_ev`: undoes the 1/`x_white` scaling of the stored values, plus the
    /// constant `tools/dngfit` found for the profile (`-4` for the embedded one, 0 for the identity profile), plus the
    /// display placement.
    ///
    /// The placement is what the app's ACES 2 path adds before the transform (`ym_color::display::DISPLAY_PLACEMENT_EV`,
    /// `docs/02` C12 step 1a). The embedded profile is fitted to the plain transform; Adobe's renderer scales the
    /// linear values by `2^BaselineExposure` before the profile, so adding the placement here makes the DNG as bright
    /// as the app (`2^placement_ev * x` through the plain transform is the placed transform of `x`). It is 0 for a
    /// writer that is not given the fitted profile.
    ///
    /// `x_white` is floored at [`BaselineExposure::white_floor`], so the value is never negative.
    Auto {
        /// The fitted offset in EV.
        fit_offset_ev: f64,
        /// The display placement in EV (0 for no placement).
        placement_ev: f64,
    },
    /// Write this value (EV) as it is.
    Fixed(f64),
}

impl Default for BaselineExposure {
    fn default() -> Self {
        Self::Auto {
            fit_offset_ev: 0.0,
            placement_ev: 0.0,
        }
    }
}

impl BaselineExposure {
    /// The lowest white level `x_white` the writer may use: `max(2^-fit_offset_ev, 1.0)` for [`Auto`](Self::Auto),
    /// 1.0 for [`Fixed`](Self::Fixed).
    ///
    /// Adobe's renderer does not scale by a negative `BaselineExposure`: it applies a highlight shoulder, which depends on
    /// the exposure and so on the photo, and a fitted profile cannot be made for that (`docs/09`, As built, `tools/dngfit`).
    /// With the white level floored at `2^-fit_offset_ev` the exposure is zero for every photo whose 99.99th percentile
    /// is below that (scene 16 for the embedded profile, nearly all of them) and positive for the rest. The cost is shadow
    /// precision: the stored codes are scene value / `x_white` * 65535.
    ///
    /// The placement does not move the floor: it only adds to a value that is never negative, and a floor at
    /// `2^-(fit_offset_ev + placement_ev)` would give 0.85 EV finer shadow codes at the price of the stored headroom
    /// above scene 8.9 (a user who lowers the exposure in the other editor needs it).
    #[must_use]
    pub fn white_floor(self) -> f32 {
        match self {
            Self::Auto { fit_offset_ev, .. } => (-fit_offset_ev).exp2().max(1.0) as f32,
            Self::Fixed(_) => 1.0,
        }
    }

    /// The value to write for a photo whose stored values were scaled by `1 / x_white`.
    #[must_use]
    pub fn resolve_ev(self, x_white: f32) -> f64 {
        match self {
            Self::Auto {
                fit_offset_ev,
                placement_ev,
            } => f64::from(x_white).log2() + fit_offset_ev + placement_ev,
            Self::Fixed(ev) => ev,
        }
    }
}

/// The TIFF/EXIF orientation after rotating an upright display by `user_clockwise_deg` (0, 90, 180
/// or 270), given the camera's own EXIF orientation (1 to 8). Mirrored values (2, 4, 5, 7) stay
/// mirrored. `None` for values out of range.
///
/// Each orientation is "mirror horizontally, then rotate clockwise by k quarter turns"; a further
/// clockwise rotation of the displayed image just adds quarter turns.
#[must_use]
pub fn compose_orientation(camera_exif_orientation: u16, user_clockwise_deg: u16) -> Option<u16> {
    // (quarter turns clockwise, mirrored) for EXIF 1..=8.
    const TABLE: [(u16, bool); 8] = [
        (0, false), // 1
        (0, true),  // 2
        (2, false), // 3
        (2, true),  // 4
        (3, true),  // 5
        (1, false), // 6
        (1, true),  // 7
        (3, false), // 8
    ];
    if !(1..=8).contains(&camera_exif_orientation)
        || !user_clockwise_deg.is_multiple_of(90)
        || user_clockwise_deg >= 360
    {
        return None;
    }
    let (k, mirrored) = TABLE[usize::from(camera_exif_orientation - 1)];
    let turns = (k + user_clockwise_deg / 90) % 4;
    TABLE
        .iter()
        .position(|&entry| entry == (turns, mirrored))
        .map(|i| i as u16 + 1)
}

/// The D50 white as `Vec3`, re-exported for tests.
pub const D50: Vec3 = D50_PCS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_matrix_inverts_awg4_to_xyz() {
        let cm = color_matrix_1();
        let prod = linalg::mul(&cm, &gamut::matrices().awg4_to_xyz);
        for (r, row) in prod.iter().enumerate() {
            for (c, value) in row.iter().enumerate() {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((value - want).abs() < 1e-14);
            }
        }
        // D65 white maps to camera (1, 1, 1).
        let w = linalg::mul_vec(&cm, xy_to_xyz(D65_XY));
        for v in w {
            assert!((v - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn forward_matrix_maps_camera_white_to_d50() {
        let fm = forward_matrix_1();
        let w = linalg::mul_vec(&fm, [1.0, 1.0, 1.0]);
        for k in 0..3 {
            assert!((w[k] - D50[k]).abs() < 1e-14, "{k}: {}", w[k]);
        }
        // It is the Bradford-adapted AWG4 matrix: close to the unscaled product.
        let raw = linalg::mul(
            &crate::icc::bradford_d65_to_d50(),
            &gamut::matrices().awg4_to_xyz,
        );
        for r in 0..3 {
            for c in 0..3 {
                assert!((fm[r][c] - raw[r][c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rationals_keep_seven_digits() {
        let fm = forward_matrix_1();
        let r = srational_matrix(&fm);
        assert_eq!(r.len(), 9);
        for (i, v) in fm.iter().flatten().enumerate() {
            let got = f64::from(r[i].n) / f64::from(r[i].d);
            assert!((got - v).abs() <= 5.1e-8, "{i}: {got} vs {v}");
        }
    }

    #[test]
    fn identity_profile_validates_and_bad_ones_do_not() {
        assert!(DngProfile::identity().validate().is_ok());
        let mut p = DngProfile::identity();
        p.tone_curve = vec![[0.0, 0.0]];
        assert!(p.validate().is_err());
        p.tone_curve = vec![[0.0, 0.0], [0.0, 1.0]];
        assert!(p.validate().is_err());
        p.tone_curve = vec![[0.0, 0.0], [1.0, 1.5]];
        assert!(p.validate().is_err());
        let mut p = DngProfile::identity();
        p.look_table.data.pop();
        assert!(p.validate().is_err());
        let mut p = DngProfile::identity();
        p.look_table.dims = [40, 2, 1];
        assert!(p.validate().is_err());
        let mut t = LookTable::identity();
        t.data[0] = f32::NAN;
        assert!(t.validate().is_err());
        // Zero-saturation entries (the first of each run of saturation divisions) keep the value scale at 1.
        let mut t = LookTable::identity();
        t.data[2] = 0.9; // value, hue 0, saturation 0
        assert!(t.validate().unwrap_err().contains("zero saturation"));
        let mut t = LookTable::identity();
        t.data[3 * 2 + 2] = 0.9; // value 0, hue 1, saturation 0
        assert!(t.validate().is_err());
        let mut t = LookTable::identity();
        t.data[3 * 3 + 2] = 1.3; // value 0, hue 1, saturation 1: allowed to re-tone
        t.data[3 * 3 + 1] = 0.5;
        assert!(t.validate().is_ok());
        t.data[3 * 3 + 2] = -0.1;
        assert!(t.validate().is_err(), "negative scales are meaningless");
        // One value division only with linear encoding.
        let flat = |encoding| LookTable {
            dims: [2, 2, 1],
            data: [0.0, 1.0, 1.0].repeat(4),
            encoding,
        };
        assert!(flat(1).validate().is_err());
        assert!(flat(0).validate().is_ok());
    }

    #[test]
    fn the_white_floor_cancels_a_negative_offset_and_is_at_least_one() {
        let auto = |fit_offset_ev| BaselineExposure::Auto {
            fit_offset_ev,
            placement_ev: 0.0,
        };
        assert_eq!(BaselineExposure::default().white_floor(), 1.0);
        assert_eq!(auto(-4.0).white_floor(), 16.0);
        assert_eq!(auto(-0.5).white_floor(), 0.5_f64.exp2() as f32);
        // a positive offset (a profile fitted for a bright tone curve) never lowers the floor below 1
        assert_eq!(auto(2.0).white_floor(), 1.0);
        assert_eq!(BaselineExposure::Fixed(-3.0).white_floor(), 1.0);
        // the exposure of a photo at the floor is zero, whatever the offset
        for offset in [-4.0, -2.5, -0.25, 0.0] {
            let b = auto(offset);
            assert!(b.resolve_ev(b.white_floor()).abs() < 1e-6, "{offset}");
        }
        // and above it the exposure is positive
        assert!(auto(-4.0).resolve_ev(32.0) > 0.99);
        // the placement adds to the exposure and leaves the floor alone
        let placed = BaselineExposure::Auto {
            fit_offset_ev: -4.0,
            placement_ev: 0.85,
        };
        assert_eq!(placed.white_floor(), 16.0);
        assert!((placed.resolve_ev(16.0) - 0.85).abs() < 1e-12);
        assert!((placed.resolve_ev(32.0) - 1.85).abs() < 1e-12);
    }

    #[test]
    fn baseline_exposure_follows_the_scale() {
        assert_eq!(BaselineExposure::default().resolve_ev(1.0), 0.0);
        assert!((BaselineExposure::default().resolve_ev(8.0) - 3.0).abs() < 1e-12);
        assert!(
            (BaselineExposure::Auto {
                fit_offset_ev: -0.25,
                placement_ev: 0.0
            }
            .resolve_ev(2.0)
                - 0.75)
                .abs()
                < 1e-12
        );
        assert_eq!(BaselineExposure::Fixed(1.5).resolve_ev(64.0), 1.5);
    }

    /// The EXIF orientations as pixel permutations of a small grid, written from the EXIF definition
    /// (what to do to the stored image to see it upright) and not from `compose_orientation`'s table.
    fn displayed(orientation: u16, a: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let flip_h = |a: &[Vec<u8>]| -> Vec<Vec<u8>> {
            a.iter()
                .map(|r| r.iter().rev().copied().collect())
                .collect()
        };
        let flip_v = |a: &[Vec<u8>]| -> Vec<Vec<u8>> { a.iter().rev().cloned().collect() };
        let transpose = |a: &[Vec<u8>]| -> Vec<Vec<u8>> {
            (0..a[0].len())
                .map(|c| a.iter().map(|row| row[c]).collect())
                .collect()
        };
        let rot_cw = |a: &[Vec<u8>]| flip_h(&transpose(a));
        match orientation {
            1 => a.to_vec(),
            2 => flip_h(a),
            3 => flip_v(&flip_h(a)),
            4 => flip_v(a),
            5 => transpose(a),
            6 => rot_cw(a),
            7 => flip_v(&flip_h(&transpose(a))), // transverse: mirrored along the other diagonal
            8 => flip_v(&flip_h(&rot_cw(a))),    // 270 degrees clockwise
            other => panic!("orientation {other}"),
        }
    }

    #[test]
    fn composed_orientations_show_the_camera_view_turned_by_the_user() {
        // A 3 x 2 image whose pixels are all different, so every flip and turn changes it.
        let stored: Vec<Vec<u8>> = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let rot_cw = |a: &[Vec<u8>]| -> Vec<Vec<u8>> {
            (0..a[0].len())
                .map(|c| a.iter().rev().map(|row| row[c]).collect())
                .collect()
        };
        for camera in 1..=8_u16 {
            let mut want = displayed(camera, &stored);
            for turns in 0..4_u16 {
                let tag = compose_orientation(camera, turns * 90).unwrap();
                assert_eq!(
                    displayed(tag, &stored),
                    want,
                    "camera {camera} turned {} degrees gives tag {tag}",
                    turns * 90
                );
                want = rot_cw(&want);
            }
        }
    }

    #[test]
    fn orientation_composes_quarter_turns() {
        // Camera upright: the user's rotation is the orientation.
        assert_eq!(compose_orientation(1, 0), Some(1));
        assert_eq!(compose_orientation(1, 90), Some(6));
        assert_eq!(compose_orientation(1, 180), Some(3));
        assert_eq!(compose_orientation(1, 270), Some(8));
        // Camera portrait (6) plus a quarter turn is upside down (3); plus three, upright.
        assert_eq!(compose_orientation(6, 90), Some(3));
        assert_eq!(compose_orientation(6, 270), Some(1));
        assert_eq!(compose_orientation(8, 90), Some(1));
        // Mirrored orientations stay mirrored: 2 (mirror) + 90 = 7 (mirror, then 90 CW).
        assert_eq!(compose_orientation(2, 90), Some(7));
        assert_eq!(compose_orientation(5, 90), Some(2));
        // Four quarter turns come back for every orientation.
        for o in 1..=8 {
            let mut cur = o;
            for _ in 0..4 {
                cur = compose_orientation(cur, 90).unwrap();
            }
            assert_eq!(cur, o);
        }
        assert_eq!(compose_orientation(0, 0), None);
        assert_eq!(compose_orientation(9, 0), None);
        assert_eq!(compose_orientation(1, 45), None);
        assert_eq!(compose_orientation(1, 360), None);
    }
}
