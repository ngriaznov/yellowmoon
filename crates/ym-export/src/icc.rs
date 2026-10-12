//! ICC v4 matrix/TRC display profiles, generated here, byte for byte deterministic.
//!
//! One profile per export colour space (`docs/09-export.md` section Colour management):
//!
//! | Space | Primaries (D65) | Tone curve (`para` parametric curve) |
//! |---|---|---|
//! | sRGB | Rec.709 | type 3, the IEC 61966-2-1 piecewise curve (gamma 2.4, offset 0.055, linear toe below 0.04045) |
//! | Display P3 | P3-D65 | the same sRGB curve |
//! | Rec.2020 | Rec.2020 | type 0, a pure gamma 2.4 (the BT.1886 inverse our encode uses) |
//!
//! The colorant tags hold the primaries' XYZ **adapted to the D50 profile connection space with
//! Bradford** (`chad` records that adaptation, so a colour-managed reader can undo it), `wtpt`
//! is the D50 PCS white as in other v4 matrix profiles, `desc` and `cprt` are `mluc` text. The header carries
//! a fixed creation date and the MD5 profile ID, so the same build always yields the same bytes.
//!
//! The Bradford adaptation here is also what the DNG writer uses for its forward matrix.

use std::sync::LazyLock;

use ym_color::consts::{D65_XY, Mat3, Vec3, linalg, xy_to_xyz};
use ym_color::gamut;

use crate::options::ColorSpace;

/// The ICC profile connection space white, D50 (ICC.1: X 0.9642, Y 1.0, Z 0.8249).
pub const D50_PCS: Vec3 = [0.9642, 1.0, 0.8249];

/// The Bradford cone-response matrix, XYZ to cone space.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Bradford chromatic adaptation from `src_white` to `dst_white` (both XYZ): the matrix that maps XYZ
/// under the source white to XYZ under the destination white.
#[must_use]
pub fn bradford_adaptation(src_white: Vec3, dst_white: Vec3) -> Mat3 {
    let src = linalg::mul_vec(&BRADFORD, src_white);
    let dst = linalg::mul_vec(&BRADFORD, dst_white);
    let scale = linalg::diag([dst[0] / src[0], dst[1] / src[1], dst[2] / src[2]]);
    // The Bradford matrix is fixed and invertible; the identity fallback is unreachable.
    let inverse = linalg::inv(&BRADFORD).unwrap_or(linalg::IDENTITY);
    linalg::mul(&inverse, &linalg::mul(&scale, &BRADFORD))
}

/// Bradford adaptation from the D65 white (x 0.3127, y 0.3290) to the ICC D50 PCS white.
#[must_use]
pub fn bradford_d65_to_d50() -> Mat3 {
    bradford_adaptation(xy_to_xyz(D65_XY), D50_PCS)
}

/// The profile for `space`: an ICC v4 matrix/TRC display profile. Generated once, then cached.
#[must_use]
pub fn profile(space: ColorSpace) -> &'static [u8] {
    static PROFILES: LazyLock<[Vec<u8>; 3]> = LazyLock::new(|| {
        [
            build_profile(ColorSpace::Srgb),
            build_profile(ColorSpace::DisplayP3),
            build_profile(ColorSpace::Rec2020),
        ]
    });
    let index = match space {
        ColorSpace::Srgb => 0,
        ColorSpace::DisplayP3 => 1,
        ColorSpace::Rec2020 => 2,
    };
    &PROFILES[index]
}

/// The profile's description text (the `desc` tag).
#[must_use]
pub fn description(space: ColorSpace) -> &'static str {
    match space {
        ColorSpace::Srgb => "Yellowmoon sRGB (IEC 61966-2-1)",
        ColorSpace::DisplayP3 => "Yellowmoon Display P3",
        ColorSpace::Rec2020 => "Yellowmoon Rec.2020 (BT.1886)",
    }
}

/// The copyright text (the `cprt` tag).
pub const COPYRIGHT: &str = "Yellowmoon contributors. Licensed under Apache-2.0.";

/// The creation date written to every profile: 2026-10-01 00:00:00. A constant, never "now".
const PROFILE_DATE: [u16; 6] = [2026, 10, 1, 0, 0, 0];

/// A signature as four bytes.
type Sig = [u8; 4];

fn s15f16(x: f64) -> [u8; 4] {
    // Values here are small (|x| < 3), far inside the i32 range.
    ((x * 65536.0).round() as i32).to_be_bytes()
}

fn pad4(data: &mut Vec<u8>) {
    while !data.len().is_multiple_of(4) {
        data.push(0);
    }
}

fn tag_xyz(xyz: Vec3) -> Vec<u8> {
    let mut d = Vec::with_capacity(20);
    d.extend_from_slice(b"XYZ ");
    d.extend_from_slice(&[0; 4]);
    for v in xyz {
        d.extend_from_slice(&s15f16(v));
    }
    d
}

fn tag_sf32(values: &[f64]) -> Vec<u8> {
    let mut d = Vec::with_capacity(8 + 4 * values.len());
    d.extend_from_slice(b"sf32");
    d.extend_from_slice(&[0; 4]);
    for &v in values {
        d.extend_from_slice(&s15f16(v));
    }
    d
}

fn tag_para(function_type: u16, params: &[f64]) -> Vec<u8> {
    let mut d = Vec::with_capacity(12 + 4 * params.len());
    d.extend_from_slice(b"para");
    d.extend_from_slice(&[0; 4]);
    d.extend_from_slice(&function_type.to_be_bytes());
    d.extend_from_slice(&[0; 2]);
    for &p in params {
        d.extend_from_slice(&s15f16(p));
    }
    d
}

fn tag_mluc(text: &str) -> Vec<u8> {
    let utf16: Vec<u8> = text.encode_utf16().flat_map(u16::to_be_bytes).collect();
    let mut d = Vec::with_capacity(28 + utf16.len());
    d.extend_from_slice(b"mluc");
    d.extend_from_slice(&[0; 4]);
    d.extend_from_slice(&1_u32.to_be_bytes()); // one record
    d.extend_from_slice(&12_u32.to_be_bytes()); // record size
    d.extend_from_slice(b"enUS");
    d.extend_from_slice(&(utf16.len() as u32).to_be_bytes());
    d.extend_from_slice(&28_u32.to_be_bytes()); // offset of the string inside the tag
    d.extend_from_slice(&utf16);
    d
}

/// The tone curve tag of `space`.
fn trc_tag(space: ColorSpace) -> Vec<u8> {
    match space {
        ColorSpace::Srgb | ColorSpace::DisplayP3 => {
            // Y = (aX + b)^g for X >= d, Y = cX below: the IEC 61966-2-1 curve.
            tag_para(3, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45])
        }
        ColorSpace::Rec2020 => tag_para(0, &[2.4]),
    }
}

/// The colorants (D50-adapted, columns of the RGB-to-XYZ matrix) and the `chad` matrix of `space`.
#[must_use]
pub fn colorants_d50(space: ColorSpace) -> ([Vec3; 3], Mat3) {
    let m = gamut::matrices();
    let to_xyz_d65 = match space {
        ColorSpace::Srgb => &m.rec709_to_xyz,
        ColorSpace::DisplayP3 => &m.p3_to_xyz,
        ColorSpace::Rec2020 => &m.rec2020_to_xyz,
    };
    let chad = bradford_d65_to_d50();
    let to_xyz_d50 = linalg::mul(&chad, to_xyz_d65);
    let column = |c: usize| [to_xyz_d50[0][c], to_xyz_d50[1][c], to_xyz_d50[2][c]];
    ([column(0), column(1), column(2)], chad)
}

fn build_profile(space: ColorSpace) -> Vec<u8> {
    let ([red, green, blue], chad) = colorants_d50(space);
    let trc = trc_tag(space);
    let chad_flat: Vec<f64> = chad.iter().flatten().copied().collect();

    // Tag data in table order. The three TRC entries share one block of data.
    let mut data: Vec<(Sig, Vec<u8>)> = vec![
        (*b"bTRC", Vec::new()), // filled with a reference to rTRC below
        (*b"bXYZ", tag_xyz(blue)),
        (*b"chad", tag_sf32(&chad_flat)),
        (*b"cprt", tag_mluc(COPYRIGHT)),
        (*b"desc", tag_mluc(description(space))),
        (*b"gTRC", Vec::new()),
        (*b"gXYZ", tag_xyz(green)),
        (*b"rTRC", trc),
        (*b"rXYZ", tag_xyz(red)),
        (*b"wtpt", tag_xyz(D50_PCS)),
    ];
    for (_, d) in &mut data {
        pad4(d);
    }

    let tag_count = data.len();
    let table_len = 4 + 12 * tag_count;
    let mut offset = 128 + table_len;
    let mut entries: Vec<(Sig, u32, u32)> = Vec::new();
    let mut body: Vec<u8> = Vec::new();
    let mut shared_trc: Option<(u32, u32)> = None;
    for (sig, d) in &data {
        let is_trc = matches!(sig, b"rTRC" | b"gTRC" | b"bTRC");
        if is_trc && d.is_empty() {
            continue; // resolved after rTRC is placed
        }
        entries.push((*sig, offset as u32, d.len() as u32));
        if sig == b"rTRC" {
            shared_trc = Some((offset as u32, d.len() as u32));
        }
        body.extend_from_slice(d);
        offset += d.len();
    }
    if let Some((off, len)) = shared_trc {
        entries.push((*b"gTRC", off, len));
        entries.push((*b"bTRC", off, len));
    }
    entries.sort_by_key(|e| e.0);

    let total = 128 + table_len + body.len();
    let mut out = Vec::with_capacity(total);
    // Header, 128 bytes.
    out.extend_from_slice(&(total as u32).to_be_bytes()); // 0 size
    out.extend_from_slice(&[0; 4]); // 4 preferred CMM: none
    out.extend_from_slice(&[0x04, 0x30, 0x00, 0x00]); // 8 version 4.3.0.0
    out.extend_from_slice(b"mntr"); // 12 device class: display
    out.extend_from_slice(b"RGB "); // 16 data colour space
    out.extend_from_slice(b"XYZ "); // 20 PCS
    for v in PROFILE_DATE {
        out.extend_from_slice(&v.to_be_bytes()); // 24 creation date
    }
    out.extend_from_slice(b"acsp"); // 36
    out.extend_from_slice(&[0; 4]); // 40 platform: none
    out.extend_from_slice(&[0; 4]); // 44 flags
    out.extend_from_slice(&[0; 4]); // 48 manufacturer
    out.extend_from_slice(&[0; 4]); // 52 model
    out.extend_from_slice(&[0; 8]); // 56 attributes
    out.extend_from_slice(&[0; 4]); // 64 rendering intent: perceptual
    for v in D50_PCS {
        out.extend_from_slice(&s15f16(v)); // 68 PCS illuminant
    }
    out.extend_from_slice(b"YMON"); // 80 creator
    out.extend_from_slice(&[0; 16]); // 84 profile ID (filled below)
    out.extend_from_slice(&[0; 28]); // 100 reserved
    debug_assert_eq!(out.len(), 128);
    // Tag table.
    out.extend_from_slice(&(tag_count as u32).to_be_bytes());
    for (sig, off, len) in &entries {
        out.extend_from_slice(sig);
        out.extend_from_slice(&off.to_be_bytes());
        out.extend_from_slice(&len.to_be_bytes());
    }
    out.extend_from_slice(&body);
    debug_assert_eq!(out.len(), total);

    // The profile ID is the MD5 of the profile with flags, intent and ID zeroed (they already are,
    // so hashing the buffer as it stands is exactly the ICC rule).
    let id = md5(&out);
    out[84..100].copy_from_slice(&id);
    out
}

/// MD5 (RFC 1321), used for the ICC profile ID.
#[must_use]
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((f64::from(i + 1)).sin().abs() * 4_294_967_296.0) as u32)
        .collect();
    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for chunk in message.chunks_exact(64) {
        let mut m = [0_u32; 16];
        for (word, bytes) in m.iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }
    let mut out = [0_u8; 16];
    for (dst, word) in out.chunks_exact_mut(4).zip(state) {
        dst.copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn md5_matches_rfc_1321_vectors() {
        assert_eq!(hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(&md5(b"a")), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(hex(&md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            hex(&md5(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            )),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    fn tags(profile: &[u8]) -> HashMap<[u8; 4], &[u8]> {
        let count = u32::from_be_bytes(profile[128..132].try_into().unwrap()) as usize;
        (0..count)
            .map(|i| {
                let e = &profile[132 + 12 * i..144 + 12 * i];
                let off = u32::from_be_bytes(e[4..8].try_into().unwrap()) as usize;
                let len = u32::from_be_bytes(e[8..12].try_into().unwrap()) as usize;
                (e[0..4].try_into().unwrap(), &profile[off..off + len])
            })
            .collect()
    }

    fn xyz(tag: &[u8]) -> Vec3 {
        assert_eq!(&tag[0..4], b"XYZ ");
        let f = |i: usize| {
            f64::from(i32::from_be_bytes(
                tag[8 + 4 * i..12 + 4 * i].try_into().unwrap(),
            )) / 65536.0
        };
        [f(0), f(1), f(2)]
    }

    #[test]
    fn header_is_a_valid_v4_display_profile() {
        for space in ColorSpace::ALL {
            let p = profile(space);
            assert_eq!(
                u32::from_be_bytes(p[0..4].try_into().unwrap()) as usize,
                p.len()
            );
            assert_eq!(p.len() % 4, 0);
            assert_eq!(&p[8..12], &[4, 0x30, 0, 0]);
            assert_eq!(&p[12..16], b"mntr");
            assert_eq!(&p[16..20], b"RGB ");
            assert_eq!(&p[20..24], b"XYZ ");
            assert_eq!(&p[36..40], b"acsp");
            assert_eq!(&p[44..48], &[0; 4], "flags");
            assert_eq!(&p[64..68], &[0; 4], "intent");
            assert!(p.len() < 1200, "a small profile: {} bytes", p.len());
            // The profile ID is the MD5 of the profile with the ID field zeroed.
            let mut zeroed = p.to_vec();
            zeroed[84..100].fill(0);
            assert_eq!(&p[84..100], &md5(&zeroed));
            // Tag offsets are 4-byte aligned and inside the profile.
            let count = u32::from_be_bytes(p[128..132].try_into().unwrap()) as usize;
            assert_eq!(count, 10);
            for i in 0..count {
                let e = &p[132 + 12 * i..144 + 12 * i];
                let off = u32::from_be_bytes(e[4..8].try_into().unwrap()) as usize;
                let len = u32::from_be_bytes(e[8..12].try_into().unwrap()) as usize;
                assert_eq!(off % 4, 0);
                assert!(off + len <= p.len());
            }
        }
    }

    #[test]
    fn profiles_are_deterministic() {
        for space in ColorSpace::ALL {
            assert_eq!(build_profile(space), build_profile(space));
            assert_eq!(build_profile(space).as_slice(), profile(space));
        }
        assert_ne!(profile(ColorSpace::Srgb), profile(ColorSpace::DisplayP3));
    }

    #[test]
    fn colorants_sum_to_d50_and_chad_maps_d65_to_d50() {
        for space in ColorSpace::ALL {
            let t = tags(profile(space));
            let (r, g, b) = (xyz(t[b"rXYZ"]), xyz(t[b"gXYZ"]), xyz(t[b"bXYZ"]));
            for k in 0..3 {
                let sum = r[k] + g[k] + b[k];
                assert!((sum - D50_PCS[k]).abs() < 4e-5, "{space:?} {k}: {sum}");
            }
            assert_eq!(
                xyz(t[b"wtpt"]),
                [0.9642, 1.0, 0.8249].map(|v| (v * 65536.0_f64).round() / 65536.0)
            );
            let chad = t[b"chad"];
            assert_eq!(&chad[0..4], b"sf32");
            assert_eq!(chad.len(), 8 + 36);
            let m: Vec<f64> = (0..9)
                .map(|i| {
                    f64::from(i32::from_be_bytes(
                        chad[8 + 4 * i..12 + 4 * i].try_into().unwrap(),
                    )) / 65536.0
                })
                .collect();
            let d65 = xy_to_xyz(D65_XY);
            for row in 0..3 {
                let v = m[3 * row] * d65[0] + m[3 * row + 1] * d65[1] + m[3 * row + 2] * d65[2];
                assert!((v - D50_PCS[row]).abs() < 5e-5, "{space:?} row {row}: {v}");
            }
        }
    }

    #[test]
    fn colorants_match_published_d50_values() {
        // sRGB: the well-known Bradford-adapted colorants of the ICC sRGB profiles.
        let t = tags(profile(ColorSpace::Srgb));
        for (tag, want) in [
            (b"rXYZ", [0.4360, 0.2225, 0.0139]),
            (b"gXYZ", [0.3851, 0.7169, 0.0971]),
            (b"bXYZ", [0.1431, 0.0606, 0.7141]),
        ] {
            let got = xyz(t[tag]);
            for k in 0..3 {
                assert!((got[k] - want[k]).abs() < 6e-4, "sRGB {tag:?} {k}: {got:?}");
            }
        }
        // Display P3: Apple's Display P3 profile.
        let t = tags(profile(ColorSpace::DisplayP3));
        for (tag, want) in [
            (b"rXYZ", [0.5151, 0.2412, -0.0011]),
            (b"gXYZ", [0.2919, 0.6922, 0.0419]),
            (b"bXYZ", [0.1571, 0.0666, 0.7841]),
        ] {
            let got = xyz(t[tag]);
            for k in 0..3 {
                assert!((got[k] - want[k]).abs() < 1e-3, "P3 {tag:?} {k}: {got:?}");
            }
        }
        // Rec.2020: the published D50 colorants of a BT.2020 v4 profile.
        let t = tags(profile(ColorSpace::Rec2020));
        for (tag, want) in [
            (b"rXYZ", [0.6734, 0.2790, -0.0019]),
            (b"gXYZ", [0.1656, 0.6753, 0.0300]),
            (b"bXYZ", [0.1251, 0.0456, 0.7969]),
        ] {
            let got = xyz(t[tag]);
            for k in 0..3 {
                assert!((got[k] - want[k]).abs() < 1e-3, "2020 {tag:?} {k}: {got:?}");
            }
        }
    }

    #[test]
    fn tone_curves_are_parametric_and_shared() {
        for space in ColorSpace::ALL {
            let t = tags(profile(space));
            assert_eq!(t[b"rTRC"], t[b"gTRC"]);
            assert_eq!(t[b"rTRC"], t[b"bTRC"]);
            let c = t[b"rTRC"];
            assert_eq!(&c[0..4], b"para");
            let function = u16::from_be_bytes([c[8], c[9]]);
            let params: Vec<f64> = c[12..]
                .chunks_exact(4)
                .map(|b| f64::from(i32::from_be_bytes(b.try_into().unwrap())) / 65536.0)
                .collect();
            match space {
                ColorSpace::Rec2020 => {
                    assert_eq!(function, 0);
                    assert_eq!(params.len(), 1);
                    assert!((params[0] - 2.4).abs() < 1e-5);
                }
                _ => {
                    assert_eq!(function, 3);
                    assert_eq!(params.len(), 5);
                    let [g, a, b, cc, d] = [params[0], params[1], params[2], params[3], params[4]];
                    assert!((g - 2.4).abs() < 1e-5);
                    assert!((a - 1.0 / 1.055).abs() < 1e-5);
                    assert!((b - 0.055 / 1.055).abs() < 1e-5);
                    assert!((cc - 1.0 / 12.92).abs() < 1e-5);
                    assert!((d - 0.04045).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn description_and_copyright_are_mluc_text() {
        let t = tags(profile(ColorSpace::DisplayP3));
        let text = |tag: &[u8]| {
            assert_eq!(&tag[0..4], b"mluc");
            let len = u32::from_be_bytes(tag[20..24].try_into().unwrap()) as usize;
            let off = u32::from_be_bytes(tag[24..28].try_into().unwrap()) as usize;
            let units: Vec<u16> = tag[off..off + len]
                .chunks_exact(2)
                .map(|b| u16::from_be_bytes([b[0], b[1]]))
                .collect();
            String::from_utf16(&units).unwrap()
        };
        assert_eq!(text(t[b"desc"]), "Yellowmoon Display P3");
        assert_eq!(text(t[b"cprt"]), COPYRIGHT);
    }

    #[test]
    fn bradford_maps_the_source_white_to_the_destination_white() {
        let m = bradford_d65_to_d50();
        let out = linalg::mul_vec(&m, xy_to_xyz(D65_XY));
        for k in 0..3 {
            assert!((out[k] - D50_PCS[k]).abs() < 1e-14);
        }
        // The textbook Bradford D65 to D50 matrix (Lindbloom, D50 = 0.96422, 1, 0.82521) agrees to 1e-3.
        let lindbloom = [
            [1.0478112, 0.0228866, -0.0501270],
            [0.0295424, 0.9904844, -0.0170491],
            [-0.0092345, 0.0150436, 0.7521316],
        ];
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m[r][c] - lindbloom[r][c]).abs() < 1.5e-3,
                    "[{r}][{c}] {}",
                    m[r][c]
                );
            }
        }
    }
}
