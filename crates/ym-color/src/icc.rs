//! A minimal reader of ICC matrix/TRC profiles, and the conversion of their code values to linear sRGB (`docs/34-reference-grade.md`
//! section 6.3, RG-05).
//!
//! The reference of a Reference Grade node is a picture in some RGB space; the fit measures in linear sRGB (CIELAB D65 on sRGB
//! display codes, section 5.1). This module reads what is needed to get there and nothing else:
//!
//! * the header (the `acsp` signature, an `RGB ` data space, an `XYZ ` connection space),
//! * the colorant tags `rXYZ`, `gXYZ`, `bXYZ` (already in the D50 connection space),
//! * the curves `rTRC`, `gTRC`, `bTRC` as `curv` (the identity, a gamma or a table) or `para` (types 0 to 4).
//!
//! A profile with a lookup table (`A2B0`, `A2B1`, `A2B2`), a grey, CMYK or named-colour profile, a missing or damaged tag, or a
//! curve outside these types is refused with an [`IccError`]; the caller takes the picture as sRGB and says so (section 6.3,
//! item 4). Everything is `f64` and runs in a fixed order, with no SIMD choice and no dependence on the machine, so the converted
//! picture has the same bits everywhere (`AGENTS.md` rule 8). It is our own code, written from the ICC specification (ICC.1:2010
//! and v4.3), not a colour-management crate (section 6.3, decision: our own reader).
//!
//! The conversion of a colour is: code value, to linear by the curve, to XYZ (D50) by the colorant matrix, to XYZ (D65) by the
//! Bradford adaptation, to linear sRGB by the Rec.709 matrix. The three steps after the curves are one 3 x 3 matrix
//! ([`ToLinearSrgb`]), made once. Colours outside the sRGB gamut come out outside 0 to 1; [`ToLinearSrgb::convert`] clamps them,
//! as the fit's own pictures are clamped (section 5.1).

use crate::consts::{D65_XY, Mat3, Primaries, Vec3, linalg, xy_to_xyz};
use crate::encode::srgb_decode;

/// The ICC profile connection space white, D50 (ICC.1: X 0.9642, Y 1.0, Z 0.8249).
pub const D50_PCS: Vec3 = [0.9642, 1.0, 0.8249];

/// The Bradford cone-response matrix, XYZ to cone space.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// The Bradford chromatic adaptation from `src_white` to `dst_white` (both XYZ): the matrix that maps XYZ under the source white to
/// XYZ under the destination white.
#[must_use]
pub fn bradford(src_white: Vec3, dst_white: Vec3) -> Mat3 {
    let src = linalg::mul_vec(&BRADFORD, src_white);
    let dst = linalg::mul_vec(&BRADFORD, dst_white);
    let scale = linalg::diag([dst[0] / src[0], dst[1] / src[1], dst[2] / src[2]]);
    // The Bradford matrix is fixed and invertible; the identity fallback cannot happen.
    let inverse = linalg::inv(&BRADFORD).unwrap_or(linalg::IDENTITY);
    linalg::mul(&inverse, &linalg::mul(&scale, &BRADFORD))
}

/// Why a profile was refused. Every one of them means the picture is taken as sRGB (`docs/34` section 6.3, item 4).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IccError {
    /// The data ends before a header, a tag table, or a tag.
    #[error("the profile is cut short")]
    Truncated,
    /// The header has no `acsp` signature.
    #[error("not an ICC profile")]
    NotAProfile,
    /// The data colour space is not RGB or the connection space is not XYZ (a grey, CMYK, Lab or named-colour profile).
    #[error("the profile is not an RGB matrix profile")]
    NotRgbXyz,
    /// The profile carries a lookup table (`A2B0`, `A2B1` or `A2B2`); only the matrix and the curves are read.
    #[error("the profile has a lookup table")]
    HasLookupTable,
    /// A colorant or curve tag is missing.
    #[error("the profile lacks the tag {0}")]
    MissingTag(&'static str),
    /// A colorant tag is not a well-formed `XYZ ` value, or the colorants do not make an invertible matrix.
    #[error("the colorants are damaged")]
    BadColorants,
    /// A curve is not `curv` or `para`, or its parameters are damaged or not a usable curve.
    #[error("a tone curve is not one this reader handles")]
    BadCurve,
}

/// A tone curve (`TRC`): code value in 0 to 1 to linear in 0 to 1.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve {
    /// `curv` with no entries, or `para` with no change: `y = x`.
    Identity,
    /// `curv` with one entry (a gamma, u8Fixed8) or `para` type 0: `y = x^g`.
    Gamma(f64),
    /// `curv` with a table: the entries scaled to 0 to 1, read by linear interpolation.
    Table(Vec<f64>),
    /// `para` of type 0 to 4 (`function_type`) with its parameters `g, a, b, c, d, e, f` (the ones the type has; the others 0).
    Parametric {
        /// The function type, 0 to 4.
        function_type: u16,
        /// `g, a, b, c, d, e, f` of ICC.1 (section 10.18).
        p: [f64; 7],
    },
}

impl Curve {
    /// The sRGB curve (IEC 61966-2.1).
    #[must_use]
    pub fn srgb() -> Self {
        Curve::Parametric {
            function_type: 3,
            p: [
                2.4,
                1.0 / 1.055,
                0.055 / 1.055,
                1.0 / 12.92,
                0.040_45,
                0.0,
                0.0,
            ],
        }
    }

    /// The linear value at the code value `x` (0 to 1; the input is clamped to it). Never NaN; the result is clamped to 0 to 1.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        let y = match self {
            Curve::Identity => x,
            Curve::Gamma(g) => x.powf(*g),
            Curve::Table(t) => {
                let last = t.len().saturating_sub(1);
                if t.is_empty() {
                    x
                } else if last == 0 {
                    t[0]
                } else {
                    let pos = x * last as f64;
                    let i = (pos.floor() as usize).min(last - 1);
                    let f = pos - i as f64;
                    t[i] + (t[i + 1] - t[i]) * f
                }
            }
            Curve::Parametric { function_type, p } => {
                let [g, a, b, c, d, e, f] = *p;
                // A base below zero would make `powf` NaN: the standard's functions are defined for the domain they state.
                let power = |u: f64| u.max(0.0).powf(g);
                match function_type {
                    0 => power(x),
                    1 => {
                        if a != 0.0 && x >= -b / a {
                            power(a * x + b)
                        } else {
                            0.0
                        }
                    }
                    2 => {
                        if a != 0.0 && x >= -b / a {
                            power(a * x + b) + c
                        } else {
                            c
                        }
                    }
                    3 => {
                        if x >= d {
                            power(a * x + b)
                        } else {
                            c * x
                        }
                    }
                    _ => {
                        if x >= d {
                            power(a * x + b) + e
                        } else {
                            c * x + f
                        }
                    }
                }
            }
        };
        if y.is_nan() { 0.0 } else { y.clamp(0.0, 1.0) }
    }
}

/// An RGB matrix/TRC profile: three curves and the colorant matrix to XYZ in the D50 connection space.
#[derive(Debug, Clone, PartialEq)]
pub struct RgbProfile {
    /// The red, green and blue curves.
    pub curves: [Curve; 3],
    /// The columns are `rXYZ`, `gXYZ` and `bXYZ`: linear RGB to XYZ (D50).
    pub to_xyz_d50: Mat3,
}

fn be16(d: &[u8], at: usize) -> Result<u16, IccError> {
    d.get(at..at + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .ok_or(IccError::Truncated)
}

fn be32(d: &[u8], at: usize) -> Result<u32, IccError> {
    d.get(at..at + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or(IccError::Truncated)
}

/// An s15Fixed16 number.
fn s15f16(d: &[u8], at: usize) -> Result<f64, IccError> {
    Ok(f64::from(be32(d, at)? as i32) / 65536.0)
}

/// The largest table a `curv` tag may hold: ICC allows 2^32 entries, no real profile has more than 2^16 (the bound keeps a
/// damaged file from asking for gigabytes).
const MAX_CURVE_ENTRIES: usize = 1 << 17;

impl RgbProfile {
    /// Reads `data` (a whole ICC profile) as a matrix/TRC profile.
    ///
    /// # Errors
    ///
    /// An [`IccError`] when it is not one, or damaged.
    pub fn parse(data: &[u8]) -> Result<Self, IccError> {
        if data.len() < 132 {
            return Err(IccError::Truncated);
        }
        if &data[36..40] != b"acsp" {
            return Err(IccError::NotAProfile);
        }
        if &data[16..20] != b"RGB " || &data[20..24] != b"XYZ " {
            return Err(IccError::NotRgbXyz);
        }
        let count = be32(data, 128)? as usize;
        // 12 bytes a tag; the table must fit in the data.
        if count > (data.len() - 132) / 12 {
            return Err(IccError::Truncated);
        }
        let find = |sig: &[u8; 4]| -> Result<Option<&[u8]>, IccError> {
            for i in 0..count {
                let at = 132 + 12 * i;
                if &data[at..at + 4] == sig {
                    let off = be32(data, at + 4)? as usize;
                    let len = be32(data, at + 8)? as usize;
                    let end = off.checked_add(len).ok_or(IccError::Truncated)?;
                    return data.get(off..end).map(Some).ok_or(IccError::Truncated);
                }
            }
            Ok(None)
        };
        for lut in [b"A2B0", b"A2B1", b"A2B2"] {
            if find(lut)?.is_some() {
                return Err(IccError::HasLookupTable);
            }
        }
        let tag = |sig: &'static [u8; 4]| -> Result<&[u8], IccError> {
            find(sig)?.ok_or(IccError::MissingTag(match sig {
                b"rXYZ" => "rXYZ",
                b"gXYZ" => "gXYZ",
                b"bXYZ" => "bXYZ",
                b"rTRC" => "rTRC",
                b"gTRC" => "gTRC",
                _ => "bTRC",
            }))
        };
        let xyz = |t: &[u8]| -> Result<Vec3, IccError> {
            if t.len() < 20 || &t[0..4] != b"XYZ " {
                return Err(IccError::BadColorants);
            }
            Ok([s15f16(t, 8)?, s15f16(t, 12)?, s15f16(t, 16)?])
        };
        let r = xyz(tag(b"rXYZ")?)?;
        let g = xyz(tag(b"gXYZ")?)?;
        let b = xyz(tag(b"bXYZ")?)?;
        let to_xyz_d50 = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        // Colorants of a real display are in the tenths and the matrix is well away from singular; a profile whose matrix is not
        // invertible, or whose entries are not finite or are huge, is damaged.
        if to_xyz_d50
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 8.0)
            || linalg::inv(&to_xyz_d50).is_none()
            || linalg::det(&to_xyz_d50).abs() < 1e-6
        {
            return Err(IccError::BadColorants);
        }
        let curves = [
            read_curve(tag(b"rTRC")?)?,
            read_curve(tag(b"gTRC")?)?,
            read_curve(tag(b"bTRC")?)?,
        ];
        Ok(Self { curves, to_xyz_d50 })
    }

    /// A profile of the standard primaries with one curve for the three channels: what a PNG `cICP` says (section 6.3, item 2).
    /// The colorants are the primaries' XYZ taken from their D65 white to the connection space's D50 by Bradford, the way ICC
    /// profiles store them.
    #[must_use]
    pub fn from_primaries(primaries: &Primaries, curve: Curve) -> Self {
        let adapt = bradford(xy_to_xyz(D65_XY), D50_PCS);
        Self {
            curves: [curve.clone(), curve.clone(), curve],
            to_xyz_d50: linalg::mul(&adapt, &primaries.rgb_to_xyz()),
        }
    }

    /// The conversion of this profile's code values to linear sRGB.
    #[must_use]
    pub fn to_linear_srgb(&self) -> ToLinearSrgb {
        let adapt = bradford(D50_PCS, xy_to_xyz(D65_XY));
        let to_xyz_d65 = linalg::mul(&adapt, &self.to_xyz_d50);
        let matrix = linalg::mul(&Primaries::REC709.xyz_to_rgb(), &to_xyz_d65);
        ToLinearSrgb {
            curves: self.curves.clone(),
            matrix: Some(matrix),
        }
    }
}

fn read_curve(t: &[u8]) -> Result<Curve, IccError> {
    if t.len() < 12 {
        return Err(IccError::BadCurve);
    }
    match &t[0..4] {
        b"curv" => {
            let n = be32(t, 8).map_err(|_| IccError::BadCurve)? as usize;
            match n {
                0 => Ok(Curve::Identity),
                1 => {
                    // u8Fixed8: the gamma.
                    let g = f64::from(be16(t, 12).map_err(|_| IccError::BadCurve)?) / 256.0;
                    if g.is_finite() && g > 0.0 {
                        Ok(Curve::Gamma(g))
                    } else {
                        Err(IccError::BadCurve)
                    }
                }
                n if n <= MAX_CURVE_ENTRIES => {
                    let mut table = Vec::with_capacity(n);
                    for i in 0..n {
                        let v = be16(t, 12 + 2 * i).map_err(|_| IccError::BadCurve)?;
                        table.push(f64::from(v) / 65535.0);
                    }
                    Ok(Curve::Table(table))
                }
                _ => Err(IccError::BadCurve),
            }
        }
        b"para" => {
            let function_type = be16(t, 8).map_err(|_| IccError::BadCurve)?;
            let count = match function_type {
                0 => 1,
                1 => 3,
                2 => 4,
                3 => 5,
                4 => 7,
                _ => return Err(IccError::BadCurve),
            };
            let mut p = [0.0; 7];
            for (i, slot) in p.iter_mut().enumerate().take(count) {
                *slot = s15f16(t, 12 + 4 * i).map_err(|_| IccError::BadCurve)?;
            }
            // Parameters that make no curve: a zero or negative gamma, and a break that is not in the domain.
            if p[0] <= 0.0 || p.iter().any(|v| !v.is_finite()) {
                return Err(IccError::BadCurve);
            }
            Ok(Curve::Parametric { function_type, p })
        }
        _ => Err(IccError::BadCurve),
    }
}

/// The conversion of a picture's code values to linear sRGB: three curves and, for a space other than sRGB, a 3 x 3 matrix.
///
/// For speed the curves are tabulated once per code value ([`ToLinearSrgb::table`]) and the picture is converted with
/// [`ToLinearSrgb::convert`]; both are exact functions of the profile and the code, so the picture's bits do not depend on the
/// machine.
#[derive(Debug, Clone, PartialEq)]
pub struct ToLinearSrgb {
    curves: [Curve; 3],
    /// `None` for sRGB itself: no matrix at all, so an sRGB picture's linear values are exactly the sRGB curve of its codes.
    matrix: Option<Mat3>,
}

impl ToLinearSrgb {
    /// An untagged picture, or one that says sRGB (sections 6.3 items 1 and 2): the sRGB curve and no matrix.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            curves: [Curve::srgb(), Curve::srgb(), Curve::srgb()],
            matrix: None,
        }
    }

    /// The conversion for a `cICP` picture of these code points (`[primaries, transfer, matrix, full range]`, ITU-T H.273), or `None`
    /// when they are not ones this reader handles: primaries 1 (Rec.709 / sRGB), 12 (Display P3) or 9 (Rec.2020), transfer 13 (the
    /// sRGB curve) or 1 (read as the pure gamma 2.4 that our own exports pair with it, `docs/09`), matrix coefficients 0 (RGB)
    /// and full range.
    #[must_use]
    pub fn from_cicp(code_points: [u8; 4]) -> Option<Self> {
        let [primaries, transfer, matrix, full] = code_points;
        if matrix != 0 || full != 1 {
            return None;
        }
        let primaries = match primaries {
            1 => Primaries::REC709,
            12 => Primaries::P3_D65,
            9 => Primaries::REC2020,
            _ => return None,
        };
        let curve = match transfer {
            13 => Curve::srgb(),
            1 => Curve::Gamma(2.4),
            _ => return None,
        };
        if primaries == Primaries::REC709 && transfer == 13 {
            return Some(Self::srgb());
        }
        Some(RgbProfile::from_primaries(&primaries, curve).to_linear_srgb())
    }

    /// The linear value of every code of `levels` (256 for 8 bits, 65536 for 16 bits) of channel `channel` (0 to 2), before the
    /// matrix.
    #[must_use]
    pub fn table(&self, channel: usize, levels: usize) -> Vec<f64> {
        let top = (levels - 1).max(1) as f64;
        (0..levels)
            .map(|i| {
                let x = i as f64 / top;
                // The sRGB case calls the crate's own decoder so it is the one the rest of the pipeline uses.
                if self.matrix.is_none() {
                    srgb_decode(x)
                } else {
                    self.curves[channel].eval(x)
                }
            })
            .collect()
    }

    /// Linear sRGB from the three curves' outputs: the matrix, then the clamp to 0 to 1 of the fit's pictures (section 5.1).
    #[must_use]
    pub fn convert(&self, linear: Vec3) -> Vec3 {
        let v = match &self.matrix {
            Some(m) => linalg::mul_vec(m, linear),
            None => linear,
        };
        [clamp01(v[0]), clamp01(v[1]), clamp01(v[2])]
    }

    /// The matrix from the curves' outputs to linear sRGB, or `None` for sRGB.
    #[must_use]
    pub fn matrix(&self) -> Option<&Mat3> {
        self.matrix.as_ref()
    }
}

fn clamp01(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header and a tag table with `tags`, for the reader's refusals.
    fn profile(space: &[u8; 4], pcs: &[u8; 4], tags: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut d = vec![0u8; 128];
        d[16..20].copy_from_slice(space);
        d[20..24].copy_from_slice(pcs);
        d[36..40].copy_from_slice(b"acsp");
        d.extend_from_slice(&(tags.len() as u32).to_be_bytes());
        let mut offset = 132 + 12 * tags.len();
        let mut body = Vec::new();
        for (sig, data) in tags {
            d.extend_from_slice(*sig);
            d.extend_from_slice(&(offset as u32).to_be_bytes());
            d.extend_from_slice(&(data.len() as u32).to_be_bytes());
            body.extend_from_slice(data);
            offset += data.len();
        }
        d.extend_from_slice(&body);
        d
    }

    fn xyz(v: Vec3) -> Vec<u8> {
        let mut d = b"XYZ \0\0\0\0".to_vec();
        for x in v {
            d.extend_from_slice(&((x * 65536.0).round() as i32).to_be_bytes());
        }
        d
    }

    fn curv(entries: &[u16]) -> Vec<u8> {
        let mut d = b"curv\0\0\0\0".to_vec();
        d.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for e in entries {
            d.extend_from_slice(&e.to_be_bytes());
        }
        d
    }

    fn para(function_type: u16, params: &[f64]) -> Vec<u8> {
        let mut d = b"para\0\0\0\0".to_vec();
        d.extend_from_slice(&function_type.to_be_bytes());
        d.extend_from_slice(&[0, 0]);
        for p in params {
            d.extend_from_slice(&((p * 65536.0).round() as i32).to_be_bytes());
        }
        d
    }

    fn matrix_profile(trc: &[u8]) -> Vec<u8> {
        profile(
            b"RGB ",
            b"XYZ ",
            &[
                (b"rXYZ", xyz([0.4361, 0.2225, 0.0139])),
                (b"gXYZ", xyz([0.3851, 0.7169, 0.0971])),
                (b"bXYZ", xyz([0.1431, 0.0606, 0.7141])),
                (b"rTRC", trc.to_vec()),
                (b"gTRC", trc.to_vec()),
                (b"bTRC", trc.to_vec()),
            ],
        )
    }

    #[test]
    fn a_matrix_profile_with_each_curve_kind_reads() {
        for trc in [
            curv(&[]),
            curv(&[0x0233]),
            curv(&[0, 16384, 32768, 65535]),
            para(0, &[2.2]),
            para(1, &[2.0, 0.9, 0.1]),
            para(2, &[2.0, 0.9, 0.1, 0.05]),
            para(3, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45]),
            para(
                4,
                &[
                    2.4,
                    1.0 / 1.055,
                    0.055 / 1.055,
                    1.0 / 12.92,
                    0.040_45,
                    0.0,
                    0.0,
                ],
            ),
        ] {
            let p = RgbProfile::parse(&matrix_profile(&trc)).expect("reads");
            for x in [0.0, 0.2, 0.5, 1.0] {
                let y = p.curves[0].eval(x);
                assert!((0.0..=1.0).contains(&y));
            }
        }
    }

    #[test]
    fn the_curves_are_the_ones_of_the_standard() {
        let srgb = RgbProfile::parse(&matrix_profile(&para(
            3,
            &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45],
        )))
        .expect("reads");
        for i in 0..=255 {
            let x = f64::from(i) / 255.0;
            // s15Fixed16 rounds the parameters to 1.5e-5.
            assert!(
                (srgb.curves[1].eval(x) - srgb_decode(x)).abs() < 3e-5,
                "{i}"
            );
        }
        assert_eq!(Curve::Identity.eval(0.3), 0.3);
        assert!((Curve::Gamma(2.0).eval(0.5) - 0.25).abs() < 1e-15);
        let table = Curve::Table(vec![0.0, 0.5, 1.0]);
        assert!((table.eval(0.25) - 0.25).abs() < 1e-15);
        assert!((table.eval(0.75) - 0.75).abs() < 1e-15);
        assert_eq!(table.eval(1.0), 1.0);
        // NaN and out-of-range inputs.
        assert_eq!(Curve::Gamma(2.0).eval(f64::NAN), 0.0);
        assert_eq!(Curve::Gamma(2.0).eval(7.0), 1.0);
    }

    #[test]
    fn the_refusals() {
        assert_eq!(RgbProfile::parse(&[0; 40]), Err(IccError::Truncated));
        let mut not = matrix_profile(&curv(&[]));
        not[36..40].copy_from_slice(b"zzzz");
        assert_eq!(RgbProfile::parse(&not), Err(IccError::NotAProfile));
        let mut cmyk = matrix_profile(&curv(&[]));
        cmyk[16..20].copy_from_slice(b"CMYK");
        assert_eq!(RgbProfile::parse(&cmyk), Err(IccError::NotRgbXyz));
        let mut lab = matrix_profile(&curv(&[]));
        lab[20..24].copy_from_slice(b"Lab ");
        assert_eq!(RgbProfile::parse(&lab), Err(IccError::NotRgbXyz));
        let grey = profile(b"GRAY", b"XYZ ", &[(b"kTRC", curv(&[]))]);
        assert_eq!(RgbProfile::parse(&grey), Err(IccError::NotRgbXyz));
        let missing = profile(b"RGB ", b"XYZ ", &[(b"rXYZ", xyz([0.4, 0.2, 0.0]))]);
        assert!(matches!(
            RgbProfile::parse(&missing),
            Err(IccError::MissingTag(_))
        ));
        let lut = profile(
            b"RGB ",
            b"XYZ ",
            &[
                (b"A2B0", vec![0; 32]),
                (b"rXYZ", xyz([0.4361, 0.2225, 0.0139])),
                (b"gXYZ", xyz([0.3851, 0.7169, 0.0971])),
                (b"bXYZ", xyz([0.1431, 0.0606, 0.7141])),
                (b"rTRC", curv(&[])),
                (b"gTRC", curv(&[])),
                (b"bTRC", curv(&[])),
            ],
        );
        assert_eq!(RgbProfile::parse(&lut), Err(IccError::HasLookupTable));
        // Damaged: a tag that runs past the end, a curve of an unknown type, a singular matrix, a zero gamma.
        let mut cut = matrix_profile(&curv(&[]));
        cut.truncate(cut.len() - 10);
        assert!(RgbProfile::parse(&cut).is_err());
        assert_eq!(
            RgbProfile::parse(&matrix_profile(b"zzzz\0\0\0\0\0\0\0\0")),
            Err(IccError::BadCurve)
        );
        assert_eq!(
            RgbProfile::parse(&matrix_profile(&para(9, &[2.0]))),
            Err(IccError::BadCurve)
        );
        assert_eq!(
            RgbProfile::parse(&matrix_profile(&para(0, &[0.0]))),
            Err(IccError::BadCurve)
        );
        let singular = profile(
            b"RGB ",
            b"XYZ ",
            &[
                (b"rXYZ", xyz([0.4, 0.2, 0.0])),
                (b"gXYZ", xyz([0.4, 0.2, 0.0])),
                (b"bXYZ", xyz([0.4, 0.2, 0.0])),
                (b"rTRC", curv(&[])),
                (b"gTRC", curv(&[])),
                (b"bTRC", curv(&[])),
            ],
        );
        assert_eq!(RgbProfile::parse(&singular), Err(IccError::BadColorants));
        // A tag table that claims more tags than the data holds.
        let mut many = matrix_profile(&curv(&[]));
        many[128..132].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(RgbProfile::parse(&many), Err(IccError::Truncated));
    }

    #[test]
    fn the_primaries_of_srgb_give_the_identity_matrix() {
        let m = *RgbProfile::from_primaries(&Primaries::REC709, Curve::srgb())
            .to_linear_srgb()
            .matrix()
            .expect("a matrix");
        for (r, row) in m.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-12, "{r}{c}: {v}");
            }
        }
    }

    #[test]
    fn white_stays_white_and_grey_stays_grey_whatever_the_primaries() {
        for primaries in [Primaries::P3_D65, Primaries::REC2020] {
            let t = RgbProfile::from_primaries(&primaries, Curve::srgb()).to_linear_srgb();
            for v in [0.0, 0.18, 0.5, 1.0] {
                let out = t.convert([v; 3]);
                for c in out {
                    assert!((c - v).abs() < 1e-12, "{v}: {out:?}");
                }
            }
        }
    }

    #[test]
    fn cicp_code_points() {
        assert_eq!(
            ToLinearSrgb::from_cicp([1, 13, 0, 1]),
            Some(ToLinearSrgb::srgb())
        );
        assert!(ToLinearSrgb::from_cicp([12, 13, 0, 1]).is_some());
        assert!(ToLinearSrgb::from_cicp([9, 1, 0, 1]).is_some());
        // Not handled: PQ, HLG, another matrix, limited range, unknown primaries.
        for cp in [
            [9, 16, 0, 1],
            [9, 18, 0, 1],
            [1, 13, 1, 1],
            [1, 13, 0, 0],
            [5, 13, 0, 1],
        ] {
            assert_eq!(ToLinearSrgb::from_cicp(cp), None, "{cp:?}");
        }
    }

    #[test]
    fn the_srgb_conversion_is_the_crates_own_decoder_and_clamps() {
        let t = ToLinearSrgb::srgb();
        let table = t.table(0, 256);
        assert_eq!(table.len(), 256);
        assert_eq!(table[0], 0.0);
        assert_eq!(table[255], 1.0);
        assert_eq!(table[128], srgb_decode(128.0 / 255.0));
        assert_eq!(t.convert([1.2, -0.1, f64::NAN]), [1.0, 0.0, 0.0]);
    }
}
