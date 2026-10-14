// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! The inverse of the ACES 2 output transform: display-linear RGB back to scene-linear ACES2065-1.
//!
//! Port of `outputTransform_inv`, `tonemap_and_compress_inv`, `chroma_compress_inv`,
//! `gamut_compress_inv`, `tonescale_inv` and the `invert` branch of `toe` in the Academy Software
//! Foundation's `aces-core` (`lib/Lib.Academy.OutputTransform.ctl`, `Lib.Academy.Tonescale.ctl`,
//! commit `069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80`, Apache-2.0). The forward half of the port is
//! in the other modules of [`crate::aces2`]; the gamut compression geometry is shared with it
//! (`Aces2Params::compress_gamut_with`), as the reference shares it through one `invert` flag.
//!
//! # What the DNG export needs it for
//!
//! A look that carries its own display rendering (LogC3 to Rec.709) or an imported ARRI display LUT
//! gives the picture as display values. A DNG holds scene-linear data that Adobe's renderer turns
//! into a picture with the profile fitted to ACES 2, so the pixels are chosen as the scene values
//! ACES 2 would turn into that picture: this inverse (`docs/09`, "What a DNG carries").
//!
//! # What it can and cannot return
//!
//! * **Exact where the forward transform is a bijection.** Forward then inverse returns the
//!   input for every scene value whose chroma is inside the reach gamut (nearly every photograph),
//!   and inverse then forward returns the display value for every display value inside the
//!   limiting gamut. The gamut compression is inverted in two passes above the analytical
//!   threshold, as the reference does ("an approximate inverse is required above the threshold");
//!   `tests/aces2_inverse.rs` measures the round trip.
//! * **Display white has no finite scene value in practice.** The tone scale reaches its roof at
//!   128 times the reference white (`r_hit`), so display values close to 1.0 invert to scene
//!   values that grow without bound. [`Aces2Transform::inverse_ap0_linear`] returns what the
//!   reference returns (finite, up to 1e3 or so); a caller that stores scene values limits them
//!   (the DNG path stops at the top of the fitted profile's range).
//! * **Clamping.** Negative and non-finite display values count as 0, values above the display's
//!   peak as the peak. A colour outside the limiting gamut cannot come out of the forward
//!   transform, so it is clamped, not gamut-mapped.

use super::Aces2Transform;
use super::cam::REF_LUMINANCE;
use super::compress::{chroma_compress_norm, reach_m_from_table};
use super::matrix::{V3, invert, mul_vec_mat};
use super::params::Aces2Params;
use super::tonescale::ToneScaleParams;

/// `tonescale_inv`: display luminance (1.0 is the 100 nit reference white) back to scene-referred
/// linear. The input is limited to what the curve can reach, `n / (u_2 n_r)`.
fn tonescale_inverse(ts: &ToneScaleParams, y: f64) -> f64 {
    let z = y.min(ts.n / (ts.u_2 * ts.n_r)).max(0.0);
    let h = (z + (z * (4.0 * ts.t_1 + z)).sqrt()) / 2.0;
    // At z = 0, h = 0 and `m_2 / h` is infinite: the scene value is 0. The denominator stays
    // positive up to the limit; the floor only absorbs rounding at the very top.
    let denominator = ((ts.m_2 / h).powf(1.0 / ts.g) - 1.0).max(f64::MIN_POSITIVE);
    ts.s_2 / denominator
}

/// `toe` with `invert = true`: the inverse of the forward toe on `0..=limit` (above `limit` the toe
/// is the identity).
fn toe_inverse(x: f64, limit: f64, k1_in: f64, k2_in: f64) -> f64 {
    if x > limit {
        return x;
    }
    let k2 = k2_in.max(0.001);
    let k1 = (k1_in * k1_in + k2 * k2).sqrt();
    let k3 = (limit + k1) / (limit + k2);
    (x * x + k1 * x) / (k3 * (x + k2))
}

impl Aces2Params {
    /// `chroma_compress_inv`: un-compresses M given the tone-mapped lightness (the J of `jmh_tc`) and
    /// the J it came from. Returns (J, M, h).
    fn chroma_compress_inverse(&self, jmh_tc: V3, j: f64) -> V3 {
        let (tonemapped_j, m_compr, h) = (jmh_tc[0], jmh_tc[1], jmh_tc[2]);
        let mut m = m_compr;
        if m_compr != 0.0 && j > 0.0 {
            let n_j = tonemapped_j / self.limit_j_max;
            let sn_j = (1.0 - n_j).max(0.0);
            let m_norm = chroma_compress_norm(h, self.chroma_compress_scale);
            let limit = n_j.powf(self.model_gamma_inv) * reach_m_from_table(h, &self.reach_m_table)
                / m_norm;

            let toe_limit = limit - 0.001;
            let toe_sn_j_sat = sn_j * self.sat;
            let toe_sqrt_n_j_sat_thr = (n_j * n_j + self.sat_thr).sqrt();
            let toe_n_j_compr = n_j * self.compr;

            m = m_compr / m_norm;
            m = toe_inverse(m, limit, toe_n_j_compr, sn_j);
            m = limit - toe_inverse(limit - m, toe_limit, toe_sn_j_sat, toe_sqrt_n_j_sat_thr);
            m *= m_norm;
            m *= (tonemapped_j / j).powf(-self.model_gamma_inv);
        }
        [j, m, h]
    }

    /// `tonemap_and_compress_inv`: the inverse tone scale on J, then the inverse chroma compression.
    fn tonemap_and_compress_inverse(&self, jmh_tc: V3) -> V3 {
        let luminance = self.input.j_to_y(jmh_tc[0]);
        let linear = tonescale_inverse(&self.tone_scale, luminance / REF_LUMINANCE);
        let j = self.input.y_to_j(linear * REF_LUMINANCE);
        self.chroma_compress_inverse(jmh_tc, j)
    }

    /// `gamut_compress_inv`: moves (J, M) back along its compression vector. Above the analytical
    /// threshold the slope gain depends on the J being solved for, so the reference inverts twice:
    /// once to estimate that J, once with it.
    fn gamut_compress_inverse(&self, jmh: V3) -> V3 {
        let (j, m, h) = (jmh[0], jmh[1], jmh[2]);
        if j <= 0.0 {
            return [0.0, 0.0, h];
        }
        if m < 0.0 || j > self.limit_j_max {
            return [j, 0.0, h];
        }
        let hdp = self.hue_dependent_gamut_params(h);
        let mut jx = j;
        if jx > hdp.analytical_threshold {
            jx = self.compress_gamut_with(jmh, jx, &hdp, true)[0];
        }
        self.compress_gamut_with(jmh, jx, &hdp, true)
    }

    /// `outputTransform_inv`: RGB of the limiting primaries, in units of 100 nits, back to linear
    /// ACES2065-1 (AP0). Not clamped: the caller keeps `rgb` inside `0..=peak / 100 nit`
    /// ([`Aces2Transform::inverse_ap0_linear`] does).
    pub fn output_transform_inv(&self, rgb: V3) -> V3 {
        let compressed = self.limit.rgb_to_jmh(rgb);
        let tonemapped = self.gamut_compress_inverse(compressed);
        let jmh = self.tonemap_and_compress_inverse(tonemapped);
        self.input.jmh_to_rgb(jmh)
    }
}

impl Aces2Transform {
    /// The scene-linear ACES2065-1 triple whose [`transform_ap0_linear`](Self::transform_ap0_linear)
    /// is `display`: display-linear RGB in the *encoding* primaries (1.0 is 100 nits) in, linear AP0
    /// out.
    ///
    /// `display` is cleaned first: a value that is not a number counts as 0, infinities and
    /// everything outside `0..=peak / 100 nit` are clamped, and for an output whose encoding
    /// primaries are not its limiting primaries (Rec.2020) the triple goes through the matrix back to
    /// the limiting primaries and is clamped again there. The result is finite. Near display white it
    /// is large (display 1.0 is scene 128, the tone scale's roof, for a neutral; see the module
    /// documentation) and can have negative components for a saturated colour: the caller limits it.
    #[must_use]
    pub fn inverse_ap0_linear(&self, display: [f64; 3]) -> [f64; 3] {
        let peak = self.peak_linear;
        let clean = |x: f64| if x.is_nan() { 0.0 } else { x.clamp(0.0, peak) };
        let display = [clean(display[0]), clean(display[1]), clean(display[2])];
        let limiting = if self.output.limiting_primaries() == self.output.encoding_primaries() {
            display
        } else {
            let m = invert(&self.limit_to_display);
            let v = mul_vec_mat(display, &m);
            [clean(v[0]), clean(v[1]), clean(v[2])]
        };
        let ap0 = self.params.output_transform_inv(limiting);
        let finite = |x: f64| if x.is_finite() { x } else { 0.0 };
        [finite(ap0[0]), finite(ap0[1]), finite(ap0[2])]
    }
}
