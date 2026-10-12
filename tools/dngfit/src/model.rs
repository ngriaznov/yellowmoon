//! A model of Adobe's reference renderer for a LinearRaw DNG with a camera profile (DNG SDK 1.7.1, `dng_render`), as far
//! as `dngfit` needs it, and the exact inverse of its profile stages.
//!
//! The pipeline for one pixel, with `x` the scene-linear AWG4 colour and `2^offset` the gain of the profile's
//! `BaselineExposure` offset (so the renderer's input is `x * 2^offset`, clipped to 0 to 1, which requires the
//! exposure to be zero or positive: see [`crate::fit`] for why a negative one is not an option):
//!
//! ```text
//! x * 2^offset --ForwardMatrix1--> XYZ D50 --> linear ProPhoto RGB --clip 0..1--> p
//!   p --> HSV (hexcone, V = max) --> ProfileLookTable (trilinear in hue, saturation and encoded value;
//!         hue shift in degrees, saturation scale, value scale) --> RGB'
//!   RGB' --> ProfileToneCurve (applied to the largest and smallest channel, the middle one interpolated between them) --> RGB''
//!   RGB'' --> XYZ D50 --> linear sRGB --clip 0..1--> sRGB encode
//! ```
//!
//! The tone curve is a spline through the profile's points that the renderer samples into a table; the model takes it
//! as a sampled function ([`ToneTable`], measured from the renderer by [`crate::fit::measure_tone`]), which absorbs
//! the spline and the table resolution.

use ym_color::consts::{Mat3, Primaries, Vec3, linalg};
use ym_color::encode::{srgb_decode, srgb_encode};
use ym_export::dng::LookTable;
use ym_export::dng::profile::forward_matrix_1;

/// ProPhoto RGB (ROMM): the working space of the renderer's look table (primaries from the DNG SDK, D50 white).
pub const PROPHOTO: Primaries = Primaries {
    red: [0.7347, 0.2653],
    green: [0.1596, 0.8404],
    blue: [0.0366, 0.0001],
    white: [0.3457, 0.3585],
};

/// The matrices between AWG4, ProPhoto and sRGB.
#[derive(Clone, Copy, Debug)]
pub struct Matrices {
    /// Scene-linear AWG4 to linear ProPhoto (`ForwardMatrix1`, then XYZ D50 to ProPhoto).
    pub awg4_to_prophoto: Mat3,
    /// Linear ProPhoto to scene-linear AWG4.
    pub prophoto_to_awg4: Mat3,
    /// Linear ProPhoto to linear sRGB (D50-adapted).
    pub prophoto_to_srgb: Mat3,
    /// Linear sRGB to linear ProPhoto.
    pub srgb_to_prophoto: Mat3,
}

impl Matrices {
    /// Builds the matrices from the primaries and `ForwardMatrix1`.
    #[must_use]
    pub fn new() -> Self {
        let pro_to_xyz = PROPHOTO.rgb_to_xyz();
        let xyz_to_pro = PROPHOTO.xyz_to_rgb();
        let awg4_to_prophoto = linalg::mul(&xyz_to_pro, &forward_matrix_1());
        // sRGB with the D50 white: the primaries' D65 matrix adapted by Bradford (the ICC convention, as the DNG SDK's).
        let srgb_to_xyz_d50 = linalg::mul(
            &ym_export::icc::bradford_d65_to_d50(),
            &ym_color::gamut::matrices().rec709_to_xyz,
        );
        let xyz_d50_to_srgb = linalg::inv(&srgb_to_xyz_d50).unwrap_or(linalg::IDENTITY);
        let prophoto_to_srgb = linalg::mul(&xyz_d50_to_srgb, &pro_to_xyz);
        Self {
            awg4_to_prophoto,
            prophoto_to_awg4: linalg::inv(&awg4_to_prophoto).unwrap_or(linalg::IDENTITY),
            prophoto_to_srgb,
            srgb_to_prophoto: linalg::inv(&prophoto_to_srgb).unwrap_or(linalg::IDENTITY),
        }
    }
}

impl Default for Matrices {
    fn default() -> Self {
        Self::new()
    }
}

/// The renderer's tone curve as a table over `[0, 1]` with linear interpolation (`samples[i]` is the output at
/// `i / (len - 1)`).
#[derive(Clone, Debug, PartialEq)]
pub struct ToneTable {
    /// The samples; at least two, non-decreasing.
    pub samples: Vec<f64>,
}

impl ToneTable {
    /// The identity.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            samples: vec![0.0, 1.0],
        }
    }

    /// The curve at `u` (clamped to 0 to 1).
    #[must_use]
    pub fn eval(&self, u: f64) -> f64 {
        let n = self.samples.len() - 1;
        let pos = u.clamp(0.0, 1.0) * n as f64;
        let i = (pos.floor() as usize).min(n - 1);
        let f = pos - i as f64;
        self.samples[i] + (self.samples[i + 1] - self.samples[i]) * f
    }

    /// The input that gives `y` (the curve is non-decreasing; flat parts give their lowest input).
    #[must_use]
    pub fn inverse(&self, y: f64) -> f64 {
        let n = self.samples.len() - 1;
        let y = y.clamp(self.samples[0], self.samples[n]);
        // first sample >= y
        let hi = self.samples.partition_point(|&s| s < y).clamp(1, n);
        let (lo_y, hi_y) = (self.samples[hi - 1], self.samples[hi]);
        let f = if hi_y > lo_y {
            (y - lo_y) / (hi_y - lo_y)
        } else {
            0.0
        };
        ((hi - 1) as f64 + f) / n as f64
    }
}

/// `DNG_RGBtoHSV`: hue in sixths (0 to 6), saturation and value in 0 to 1 for non-negative input.
#[must_use]
pub fn rgb_to_hsv(rgb: Vec3) -> Vec3 {
    let [r, g, b] = rgb;
    let v = r.max(g).max(b);
    let gap = v - r.min(g).min(b);
    if gap > 0.0 {
        let h = if r == v {
            let h = (g - b) / gap;
            if h < 0.0 { h + 6.0 } else { h }
        } else if g == v {
            2.0 + (b - r) / gap
        } else {
            4.0 + (r - g) / gap
        };
        [h, gap / v, v]
    } else {
        [0.0, 0.0, v]
    }
}

/// `DNG_HSVtoRGB`.
#[must_use]
pub fn hsv_to_rgb(hsv: Vec3) -> Vec3 {
    let [mut h, s, v] = hsv;
    if s <= 0.0 {
        return [v, v, v];
    }
    if h < 0.0 {
        h += 6.0;
    }
    if h >= 6.0 {
        h -= 6.0;
    }
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// The tone curve applied the way the renderer's `RGBTone` does: the largest and smallest channels go through
/// the curve and the middle one keeps its relative position between them.
#[must_use]
pub fn rgb_tone(rgb: Vec3, curve: &ToneTable) -> Vec3 {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| rgb[b].total_cmp(&rgb[a]));
    let (hi, mid, lo) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0; 3];
    if rgb[hi] > rgb[lo] {
        let t_hi = curve.eval(rgb[hi]);
        let t_lo = curve.eval(rgb[lo]);
        out[hi] = t_hi;
        out[lo] = t_lo;
        out[mid] = t_lo + (t_hi - t_lo) * (rgb[mid] - rgb[lo]) / (rgb[hi] - rgb[lo]);
    } else {
        let t = curve.eval(rgb[hi]);
        out = [t; 3];
    }
    out
}

/// The look table's modifiers at an HSV position, interpolated like the renderer: trilinear, hue wrapping, saturation
/// and encoded value clamped to the grid.
#[must_use]
pub fn look_at(table: &LookTable, hsv: Vec3) -> Vec3 {
    let [hd, sd, vd] = table.dims.map(|d| d as usize);
    let [h, s, v] = hsv;
    let entry = |vi: usize, hi: usize, si: usize| -> Vec3 {
        let i = ((vi * hd + hi) * sd + si) * 3;
        [
            f64::from(table.data[i]),
            f64::from(table.data[i + 1]),
            f64::from(table.data[i + 2]),
        ]
    };
    let h_scale = if hd < 2 { 0.0 } else { hd as f64 / 6.0 };
    let s_scale = (sd - 1) as f64;
    let v_scale = (vd - 1) as f64;
    let v_enc = if table.encoding == 1 && vd > 1 {
        srgb_encode(v.min(1.0))
    } else {
        v
    };
    let h_scaled = h * h_scale;
    let s_scaled = s * s_scale;
    let v_scaled = v_enc * v_scale;
    let h0 = (h_scaled as usize).min(hd.saturating_sub(1));
    let s0 = (s_scaled as usize).min(sd.saturating_sub(2));
    let v0 = if vd < 2 {
        0
    } else {
        (v_scaled as usize).min(vd - 2)
    };
    let h1 = if h0 + 1 >= hd { 0 } else { h0 + 1 };
    let (fh1, fs1) = (h_scaled - h0 as f64, s_scaled - s0 as f64);
    let fv1 = if vd < 2 { 0.0 } else { v_scaled - v0 as f64 };
    let (fh0, fs0, fv0) = (1.0 - fh1, 1.0 - fs1, 1.0 - fv1);
    let v1 = if vd < 2 { 0 } else { v0 + 1 };
    let mut out = [0.0; 3];
    for (vi, wv) in [(v0, fv0), (v1, fv1)] {
        if wv == 0.0 {
            continue;
        }
        for (hi, wh) in [(h0, fh0), (h1, fh1)] {
            for (si, ws) in [(s0, fs0), (s0 + 1, fs1)] {
                let w = wv * wh * ws;
                if w == 0.0 {
                    continue;
                }
                let e = entry(vi, hi, si);
                for k in 0..3 {
                    out[k] += w * e[k];
                }
            }
        }
    }
    out
}

/// What the look table does to a linear ProPhoto colour in 0 to 1.
#[must_use]
pub fn apply_look(table: &LookTable, rgb: Vec3) -> Vec3 {
    let [h, s, v] = rgb_to_hsv(rgb);
    let [dh, ds, dv] = look_at(table, [h, s, v]);
    let h2 = h + dh * (6.0 / 360.0);
    let s2 = (s * ds).min(1.0);
    // With an encoded table the value scale multiplies the encoded value, which is then decoded again.
    let v2 = if table.encoding == 1 && table.dims[2] > 1 {
        srgb_decode((srgb_encode(v.min(1.0)) * dv).min(1.0))
    } else {
        (v * dv).min(1.0)
    };
    hsv_to_rgb([h2, s2, v2])
}

/// The renderer for a profile: tone table and look table, with the gain of the offset.
#[derive(Clone, Debug)]
pub struct Model {
    /// The matrices.
    pub m: Matrices,
    /// The tone curve.
    pub tone: ToneTable,
    /// The look table.
    pub look: LookTable,
    /// The gain of the profile's `BaselineExposure` offset, `2^fit_offset_ev`.
    pub gain: f64,
}

impl Model {
    /// A model of the renderer for this profile.
    #[must_use]
    pub fn new(tone: ToneTable, look: LookTable, fit_offset_ev: f64) -> Self {
        Self {
            m: Matrices::new(),
            tone,
            look,
            gain: fit_offset_ev.exp2(),
        }
    }

    /// The renderer's ProPhoto colour for a scene-linear AWG4 colour (after the gain, clipped to 0 to 1).
    #[must_use]
    pub fn prophoto_of(&self, awg4: Vec3) -> Vec3 {
        let p = linalg::mul_vec(&self.m.awg4_to_prophoto, awg4.map(|c| c * self.gain));
        p.map(|c| c.clamp(0.0, 1.0))
    }

    /// The sRGB code values (0 to 1) the renderer returns for a post-gain linear ProPhoto colour.
    #[must_use]
    pub fn render_prophoto(&self, p: Vec3) -> Vec3 {
        let looked = apply_look(&self.look, p);
        let toned = rgb_tone(looked, &self.tone);
        let srgb = linalg::mul_vec(&self.m.prophoto_to_srgb, toned);
        srgb.map(|c| srgb_encode(c.clamp(0.0, 1.0)))
    }

    /// The sRGB code values for a scene-linear AWG4 colour.
    #[must_use]
    pub fn render(&self, awg4: Vec3) -> Vec3 {
        self.render_prophoto(self.prophoto_of(awg4))
    }
}

/// The linear ProPhoto colour whose tone-mapped, sRGB-converted result is the sRGB colour `code`: the exact inverse
/// of the last two stages, used to find which colour the look table has to produce. `None` when the colour is not
/// reachable (a code outside the sRGB cube is clamped first; the tone curve may be flat there).
#[must_use]
pub fn before_tone(code: Vec3, curve: &ToneTable, m: &Matrices) -> Vec3 {
    let linear = code.map(|c| srgb_decode(c.clamp(0.0, 1.0)));
    let toned = linalg::mul_vec(&m.srgb_to_prophoto, linear).map(|c| c.clamp(0.0, 1.0));
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| toned[b].total_cmp(&toned[a]));
    let (hi, mid, lo) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0; 3];
    if toned[hi] > toned[lo] {
        let u_hi = curve.inverse(toned[hi]);
        let u_lo = curve.inverse(toned[lo]);
        out[hi] = u_hi;
        out[lo] = u_lo;
        out[mid] = u_lo + (u_hi - u_lo) * (toned[mid] - toned[lo]) / (toned[hi] - toned[lo]);
    } else {
        out = [curve.inverse(toned[hi]); 3];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_round_trips_and_follows_the_sdk_conventions() {
        for rgb in [
            [0.5, 0.25, 0.1],
            [0.1, 0.5, 0.25],
            [0.25, 0.1, 0.5],
            [0.3, 0.3, 0.3],
            [0.9, 0.0, 0.0],
            [0.2, 0.7, 0.7],
        ] {
            let back = hsv_to_rgb(rgb_to_hsv(rgb));
            for k in 0..3 {
                assert!((back[k] - rgb[k]).abs() < 1e-12, "{rgb:?} -> {back:?}");
            }
        }
        // pure red is hue 0, pure green hue 2 (sixths), pure blue hue 4
        assert_eq!(rgb_to_hsv([1.0, 0.0, 0.0]), [0.0, 1.0, 1.0]);
        assert_eq!(rgb_to_hsv([0.0, 1.0, 0.0])[0], 2.0);
        assert_eq!(rgb_to_hsv([0.0, 0.0, 1.0])[0], 4.0);
        // magenta-ish red wraps to just under 6
        assert!(rgb_to_hsv([1.0, 0.0, 0.1])[0] > 5.5);
    }

    #[test]
    fn the_tone_table_interpolates_and_inverts() {
        let t = ToneTable {
            samples: vec![0.0, 0.1, 0.5, 1.0],
        };
        assert!((t.eval(1.0 / 3.0) - 0.1).abs() < 1e-12);
        assert!((t.eval(0.5) - 0.3).abs() < 1e-12);
        for y in [0.0, 0.05, 0.1, 0.3, 0.77, 1.0] {
            assert!((t.eval(t.inverse(y)) - y).abs() < 1e-12, "{y}");
        }
        assert_eq!(t.eval(-1.0), 0.0);
        assert_eq!(t.eval(2.0), 1.0);
    }

    #[test]
    fn rgb_tone_keeps_the_order_and_neutrals_pass_through_the_curve() {
        let t = ToneTable {
            samples: vec![0.0, 0.3, 0.8, 1.0],
        };
        let out = rgb_tone([0.2, 0.5, 0.8], &t);
        assert!(out[0] < out[1] && out[1] < out[2]);
        assert!((out[0] - t.eval(0.2)).abs() < 1e-12 && (out[2] - t.eval(0.8)).abs() < 1e-12);
        let n = rgb_tone([0.4; 3], &t);
        assert!(n.iter().all(|&c| (c - t.eval(0.4)).abs() < 1e-12));
        // inverse of the tone stage
        let m = Matrices::new();
        let code = [0.6, 0.3, 0.2];
        let before = before_tone(code, &t, &m);
        let toned = rgb_tone(before, &t);
        let srgb =
            linalg::mul_vec(&m.prophoto_to_srgb, toned).map(|c| srgb_encode(c.clamp(0.0, 1.0)));
        for k in 0..3 {
            assert!((srgb[k] - code[k]).abs() < 1e-6, "{srgb:?} vs {code:?}");
        }
    }

    #[test]
    fn the_identity_table_changes_nothing_and_a_node_entry_acts_at_its_node() {
        let id = LookTable::identity();
        for rgb in [
            [0.5, 0.25, 0.1],
            [0.1, 0.5, 0.25],
            [0.9, 0.9, 0.9],
            [0.02, 0.03, 0.5],
        ] {
            let out = apply_look(&id, rgb);
            for k in 0..3 {
                assert!((out[k] - rgb[k]).abs() < 1e-9, "{rgb:?} -> {out:?}");
            }
        }
        // 6 hue x 3 saturation x 3 value divisions with sRGB encoding; change the node at hue 2 (green), s = 1, v encoded 0.5
        let mut t = LookTable {
            dims: [6, 3, 3],
            data: [0.0, 1.0, 1.0].repeat(54),
            encoding: 1,
        };
        let (h, s, v) = (2usize, 2usize, 1usize);
        let i = ((v * 6 + h) * 3 + s) * 3;
        t.data[i + 1] = 0.5; // halve the saturation there
        let v_lin = srgb_decode(0.5);
        let at_node = hsv_to_rgb([2.0, 1.0, v_lin]); // pure green at encoded value 0.5
        let out = apply_look(&t, at_node);
        let hsv = rgb_to_hsv(out);
        assert!(
            (hsv[1] - 0.5).abs() < 1e-9 && (hsv[2] - v_lin).abs() < 1e-9,
            "{hsv:?}"
        );
        // far from the node (red, s = 1) nothing changes
        let red = hsv_to_rgb([0.0, 1.0, v_lin]);
        let out = apply_look(&t, red);
        for k in 0..3 {
            assert!((out[k] - red[k]).abs() < 1e-9);
        }
    }
}
