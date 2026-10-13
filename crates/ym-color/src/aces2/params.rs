// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! All pre-computed parameters and tables of one ACES 2 output transform (`ODTParams`).

use std::sync::{Arc, OnceLock};

use super::SDR_PEAK_LUMINANCE_NITS;
use super::cam::{JMhParams, REF_LUMINANCE, model_gamma};
use super::compress::{
    CHROMA_COMPRESS, CHROMA_COMPRESS_FACT, CHROMA_EXPAND, CHROMA_EXPAND_FACT, CHROMA_EXPAND_THR,
    FOCUS_DISTANCE, FOCUS_DISTANCE_SCALING,
};
use super::matrix::{M33, V3, ctl_rgb_to_xyz, ctl_xyz_to_rgb, mul_mat_mat, mul_vec_mat};
use super::tables::{
    determine_hue_linearity_search_range, make_reach_m_table, make_uniform_hue_gamut_table,
    make_upper_hull_gamma_table,
};
use super::tonescale::ToneScaleParams;
use crate::consts::Primaries;

/// Smallest peak luminance, in nits, the transform is defined for.
///
/// The tone scale's roof (`r_hit`, the scene value that reaches the peak) is `128 + 768 *
/// ln(peak / 100) / ln(100)`; it turns negative below 46.4 nits, where the model has no meaning.
/// 48 nits is the DCI cinema white, the lowest practical target.
pub const MIN_PEAK_LUMINANCE_NITS: f64 = 48.0;
/// Largest peak luminance, in nits (the reference and OpenColorIO both stop at 10 000).
pub const MAX_PEAK_LUMINANCE_NITS: f64 = 10_000.0;

/// Nominal number of hue samples (one per degree).
pub const TABLE_SIZE: usize = 360;
/// Index of hue 0 in every table: entry 0 is the lower wrap entry.
pub const BASE_INDEX: usize = 1;
/// Index of the first upper wrap entry (hue 360 equivalent).
pub const UPPER_WRAP_INDEX: usize = BASE_INDEX + TABLE_SIZE;
/// Entries per table: lower wrap, 360 nominal and two upper wrap entries (the second upper entry
/// covers a hue that rounds to exactly 360.0).
pub const TOTAL_TABLE_SIZE: usize = TABLE_SIZE + 3;

/// Everything the ACES 2 output transform precomputes for one peak luminance and one set of
/// limiting primaries (`ODTParams`), as plain f64 arrays a GPU port can upload.
///
/// All `Vec` tables hold [`TOTAL_TABLE_SIZE`] entries indexed as described on each field.
#[derive(Clone, Debug)]
pub struct Aces2Params {
    /// Peak luminance in nits.
    pub peak_luminance_nits: f64,
    /// Limiting primaries the output is rendered to.
    pub limiting_primaries: Primaries,
    /// JMh parameters of the input space (AP0).
    pub input: JMhParams,
    /// JMh parameters of the reach gamut (AP1 primaries).
    pub reach: JMhParams,
    /// JMh parameters of the limiting gamut.
    pub limit: JMhParams,
    /// Tone scale parameters.
    pub tone_scale: ToneScaleParams,
    /// Lightness J of the peak luminance (maximum J of the display).
    pub limit_j_max: f64,
    /// `1 / model_gamma`.
    pub model_gamma_inv: f64,
    /// Reach gamut M at `limit_j_max` per degree of hue; entry `BASE_INDEX + h` is hue `h`.
    pub reach_m_table: Vec<f64>,
    /// Chroma expansion strength.
    pub sat: f64,
    /// Chroma expansion threshold.
    pub sat_thr: f64,
    /// Chroma compression strength.
    pub compr: f64,
    /// Scale of the hue dependent M normalisation.
    pub chroma_compress_scale: f64,
    /// J of the mid grey output, the focus point of the gamut compression vectors.
    pub mid_j: f64,
    /// Distance of the gamut compression focus point.
    pub focus_dist: f64,
    /// `1 / gamma` of the lower hull approximation.
    pub lower_hull_gamma_inv: f64,
    /// Hues of the (almost uniform) hue table, degrees. Entry 0 is the last hue minus 360, entries
    /// from `UPPER_WRAP_INDEX` are the first hues plus 360.
    pub hue_table: Vec<f64>,
    /// Cusp J of the limiting gamut at each entry of `hue_table`.
    pub cusp_j_table: Vec<f64>,
    /// Cusp M (inflated by `1 + smooth_m * smooth_cusps`) at each entry of `hue_table`.
    pub cusp_m_table: Vec<f64>,
    /// `1 / gamma` of the upper hull approximation at each entry of `hue_table`.
    pub upper_hull_gamma_inv_table: Vec<f64>,
    /// Window (relative to the uniform position) the hue interval search starts from.
    pub hue_linearity_search_range: [i32; 2],

    ap0_to_ap1: M33,
    ap1_to_ap0: M33,
}

impl Aces2Params {
    /// `init_ODTParams(peakLuminance, limitingPrimaries)`. Builds every table; deterministic.
    ///
    /// The peak is limited to [`MIN_PEAK_LUMINANCE_NITS`]..=[`MAX_PEAK_LUMINANCE_NITS`] and NaN
    /// becomes the 100 nit SDR peak: outside that range the reference maths yields a negative tone
    /// scale roof (a panic in `clamp`) or never terminates (an infinite peak). The value actually
    /// used is stored in `peak_luminance_nits`. `limiting_primaries` must be a valid set of
    /// chromaticities (non-degenerate triangle, white with y > 0); nothing else is checked.
    pub fn new(peak_luminance_nits: f64, limiting_primaries: Primaries) -> Self {
        let peak_luminance_nits = if peak_luminance_nits.is_nan() {
            SDR_PEAK_LUMINANCE_NITS
        } else {
            peak_luminance_nits.clamp(MIN_PEAK_LUMINANCE_NITS, MAX_PEAK_LUMINANCE_NITS)
        };
        let input = JMhParams::new(&Primaries::AP0);
        let reach = JMhParams::new(&Primaries::AP1);
        let limit = JMhParams::new(&limiting_primaries);

        let tone_scale = ToneScaleParams::new(peak_luminance_nits);

        // Shared compression parameters.
        let limit_j_max = input.y_to_j(peak_luminance_nits);
        let model_gamma_inv = 1.0 / model_gamma();
        let reach_m_table = make_reach_m_table(&reach, limit_j_max);

        // Chroma compression parameters.
        let sat =
            (CHROMA_EXPAND - (CHROMA_EXPAND * CHROMA_EXPAND_FACT) * tone_scale.log_peak).max(0.2);
        let sat_thr = CHROMA_EXPAND_THR / peak_luminance_nits;
        let compr =
            CHROMA_COMPRESS + (CHROMA_COMPRESS * CHROMA_COMPRESS_FACT) * tone_scale.log_peak;
        let chroma_compress_scale = (0.03379 * peak_luminance_nits).powf(0.30596) - 0.45135;

        // Gamut compression parameters.
        let mid_j = input.y_to_j(tone_scale.c_t * REF_LUMINANCE);
        let focus_dist =
            FOCUS_DISTANCE + FOCUS_DISTANCE * FOCUS_DISTANCE_SCALING * tone_scale.log_peak;
        let lower_hull_gamma = 1.14 + 0.07 * tone_scale.log_peak;
        let lower_hull_gamma_inv = 1.0 / lower_hull_gamma;

        let cusps = make_uniform_hue_gamut_table(
            &reach,
            &limit,
            peak_luminance_nits,
            tone_scale.forward_limit,
            limit_j_max,
        );
        let hue_linearity_search_range = determine_hue_linearity_search_range(&cusps.hue);
        let upper_hull_gamma_inv_table = make_upper_hull_gamma_table(
            &cusps,
            peak_luminance_nits,
            limit_j_max,
            mid_j,
            focus_dist,
            lower_hull_gamma_inv,
            &limit,
        );

        let ap0_to_xyz = ctl_rgb_to_xyz(&Primaries::AP0, 1.0);
        let ap1_to_xyz = ctl_rgb_to_xyz(&Primaries::AP1, 1.0);
        let ap0_to_ap1 = mul_mat_mat(&ap0_to_xyz, &ctl_xyz_to_rgb(&Primaries::AP1, 1.0));
        let ap1_to_ap0 = mul_mat_mat(&ap1_to_xyz, &ctl_xyz_to_rgb(&Primaries::AP0, 1.0));

        Aces2Params {
            peak_luminance_nits,
            limiting_primaries,
            input,
            reach,
            limit,
            tone_scale,
            limit_j_max,
            model_gamma_inv,
            reach_m_table,
            sat,
            sat_thr,
            compr,
            chroma_compress_scale,
            mid_j,
            focus_dist,
            lower_hull_gamma_inv,
            hue_table: cusps.hue,
            cusp_j_table: cusps.cusp_j,
            cusp_m_table: cusps.cusp_m,
            upper_hull_gamma_inv_table,
            hue_linearity_search_range,
            ap0_to_ap1,
            ap1_to_ap0,
        }
    }

    /// Cached parameters for 100 nit SDR and Rec.709 or P3-D65 limiting primaries.
    ///
    /// Building the tables takes tens of milliseconds, so the two SDR variants used by the
    /// Yellowmoon outputs are built once per process.
    pub(super) fn cached_sdr(limiting: Primaries) -> Arc<Aces2Params> {
        static REC709: OnceLock<Arc<Aces2Params>> = OnceLock::new();
        static P3_D65: OnceLock<Arc<Aces2Params>> = OnceLock::new();
        let slot = if limiting == Primaries::REC709 {
            &REC709
        } else {
            debug_assert_eq!(limiting, Primaries::P3_D65);
            &P3_D65
        };
        Arc::clone(
            slot.get_or_init(|| Arc::new(Aces2Params::new(SDR_PEAK_LUMINANCE_NITS, limiting))),
        )
    }

    /// `clamp_AP0_to_AP1(aces, 0, forward_limit)`: the input is clamped in AP1 so only values the
    /// model can represent enter it. NaN becomes 0 and infinities are limited to +-1e30 (far
    /// beyond any finite value the clamp passes) so the matrix product stays finite.
    fn clamp_ap0_to_ap1(&self, aces: V3) -> V3 {
        let finite = |x: f64| {
            if x.is_nan() {
                0.0
            } else {
                x.clamp(-1e30, 1e30)
            }
        };
        let aces = [finite(aces[0]), finite(aces[1]), finite(aces[2])];
        let ap1 = mul_vec_mat(aces, &self.ap0_to_ap1);
        let limit = self.tone_scale.forward_limit;
        let clamped = [
            ap1[0].clamp(0.0, limit),
            ap1[1].clamp(0.0, limit),
            ap1[2].clamp(0.0, limit),
        ];
        mul_vec_mat(clamped, &self.ap1_to_ap0)
    }

    /// `tonemap_and_compress_fwd`: tone scale J, then compress M.
    fn tonemap_and_compress_fwd(&self, jmh: V3) -> V3 {
        let linear = self.input.j_to_y(jmh[0]) / REF_LUMINANCE;
        let tonemapped_y = self.tone_scale.forward(linear);
        let j_ts = self.input.y_to_j(tonemapped_y);
        self.chroma_compress_fwd(jmh, j_ts)
    }

    /// `outputTransform_fwd`: linear ACES2065-1 (AP0) to RGB of the limiting primaries, in units of
    /// 100 nits (1.0 = 100 nit white), before any clamp.
    pub fn output_transform_fwd(&self, aces: V3) -> V3 {
        let ap0_clamped = self.clamp_ap0_to_ap1(aces);
        let jmh = self.input.rgb_to_jmh(ap0_clamped);
        let tonemapped = self.tonemap_and_compress_fwd(jmh);
        let compressed = self.gamut_compress_fwd(tonemapped);
        self.limit.jmh_to_rgb(compressed)
    }
}
