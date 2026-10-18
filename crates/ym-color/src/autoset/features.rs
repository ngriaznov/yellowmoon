//! The feature vector of `autoset` (`docs/18-auto-settings.md` §3): 39 values, in the order of [`FEATURE_NAMES`],
//! from the metering preview and the scene light.
//!
//! For every pixel whose camera channels are finite, `Y` is the AWG4 `Y` row on the camera RGB at EV 0 (the row
//! [`auto_exposure`](crate::exposure::auto_exposure) uses) and `M` the largest AWG4 channel, both floored at
//! [`FLOOR`]. With the key `m = log2(mean_luminance / 0.18)`, the pixel gives `s = log2(Y / 0.18) - m` and
//! `t = log2(M / 0.18) - m`. Clipped pixels take part in the quantiles (they are the highlights); a pixel with a
//! non-finite channel takes part in nothing but the clip share.
//!
//! The spatial thirds use integer division. With `tw = w / 3` and `th = h / 3`: the top third is the rows
//! `0..th`, the bottom third the rows `h - th..h` (both over every column), and the central third the rows
//! `th..h - th` within the columns `tw..w - tw`. A third with no usable pixel takes the mean of all usable pixels
//! (only previews of a few pixels have one).
//!
//! The quantiles are linearly interpolated as in [`exposure::quantile`](crate::exposure), but found by selection
//! ([`select_ranks`]), never by a full sort. The budget for the whole pass is 40 ms on a 1/8 preview of a 61 MP file
//! (`docs/18` §2).

use crate::consts::linalg::{Mat3, mul_vec};
use crate::exposure::{PreviewRgb, TARGET_GREY, interpolate};
use crate::wb::SceneLight;

use super::AutosetError;

/// Number of features of the model's input (`docs/18` §3).
pub const FEATURE_COUNT: usize = 39;

/// The feature names, in the order of [`Features::values`] and of the model's input (`docs/18` §3).
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "key_bv",
    "bv_known",
    "q01",
    "q02",
    "q05",
    "q10",
    "q20",
    "q30",
    "q40",
    "q50",
    "q60",
    "q70",
    "q80",
    "q90",
    "q95",
    "q98",
    "q99",
    "q995",
    "q999",
    "mq50",
    "mq90",
    "mq98",
    "mq995",
    "mq999",
    "clip_headroom",
    "clipped_share",
    "centre_minus_key",
    "top_minus_bottom",
    "bright_share",
    "dark_share",
    "p_daylight",
    "p_tungsten",
    "p_fluorescent",
    "p_led",
    "mired",
    "duv",
    "light_confidence",
    "log_iso",
    "flash",
];

/// The luminance and channel-maximum floor before the logarithm, 1e-6 (`docs/18` §3). It is lower than the exposure
/// meter's [`LUMINANCE_FLOOR`](crate::exposure::LUMINANCE_FLOOR) of 1e-4, as the spec sets.
pub const FLOOR: f64 = 1e-6;

/// Probabilities of the quantiles of `s - m` (features 2 to 18, `q01` to `q999`).
const PROBABILITIES_S: [f64; 17] = [
    0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
    0.995, 0.999,
];

/// Probabilities of the quantiles of `t - m` (features 19 to 23, `mq50` to `mq999`).
const PROBABILITIES_T: [f64; 5] = [0.50, 0.90, 0.98, 0.995, 0.999];

/// A pixel is bright when it lies more than this many stops above the key (`bright_share`).
const BRIGHT_STOPS: f64 = 2.0;

/// A pixel is dark when it lies more than this many stops below the key (`dark_share`).
const DARK_STOPS: f64 = -3.0;

/// The inputs of [`features`]: the metering preview and the readings that go with it.
#[derive(Debug, Clone, Copy)]
pub struct FeatureInputs<'a> {
    /// The metering preview: the 1/8 preview, `g0`-balanced, with the vignetting gain applied, and its clip mask.
    pub preview: &'a PreviewRgb<'a>,
    /// The fused C1 to C6 matrix at EV 0 (`FusedChain::matrix` of a chain built at `exposure_ev = 0`), as
    /// [`auto_exposure`](crate::exposure::auto_exposure) takes it.
    pub matrix_ev0: &'a Mat3,
    /// The centre-weighted geometric mean luminance at EV 0 (`AutoExposure::mean_luminance`).
    pub mean_luminance: f64,
    /// The AWG4 `Y` of the clipped neutral at EV 0: the AWG4 luminance row on `FusedChain::clip().awg4` at EV 0.
    pub clip_y_ev0: f64,
    /// The brightness value of the shot settings (`ShotInfo::bv()`); `None` when unknown.
    pub bv_settings: Option<f64>,
    /// The scene light (`IlluminantEstimate::scene_light`): its CCT, Duv, class probabilities and confidence.
    pub light: &'a SceneLight,
    /// The ISO speed; `None` when unknown (the feature is then 0).
    pub iso: Option<f64>,
    /// Whether the EXIF Flash field says the flash fired.
    pub flash_fired: bool,
}

/// The features of one photo, and the key they were measured from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Features {
    /// The 39 values, in the order of [`FEATURE_NAMES`].
    pub values: [f64; FEATURE_COUNT],
    /// The key `m = log2(mean_luminance / 0.18)`, in stops over middle grey. The quantile features are measured from
    /// it.
    pub key: f64,
}

/// The features of a photo (`docs/18` §3).
///
/// # Errors
///
/// [`AutosetError::Preview`] for buffers of the wrong size, [`AutosetError::MeanLuminance`] and
/// [`AutosetError::ClipLevel`] for readings that are not positive and finite, [`AutosetError::NoUsablePixel`] when no
/// pixel has finite channels, and [`AutosetError::NonFinite`] when a feature still comes out not finite (for example a
/// scene light with a CCT of zero).
pub fn features(inputs: &FeatureInputs<'_>) -> Result<Features, AutosetError> {
    let preview = inputs.preview;
    preview.validate()?;
    let mean = inputs.mean_luminance;
    if !(mean.is_finite() && mean > 0.0) {
        return Err(AutosetError::MeanLuminance(mean));
    }
    let clip_y = inputs.clip_y_ev0;
    if !(clip_y.is_finite() && clip_y > 0.0) {
        return Err(AutosetError::ClipLevel(clip_y));
    }
    let key = (mean / TARGET_GREY).log2();

    let mut scan = Scan::new(preview, inputs.matrix_ev0, key);
    if scan.s.is_empty() {
        return Err(AutosetError::NoUsablePixel);
    }
    let usable = scan.s.len() as f64;
    let all_mean = scan.all.mean_or(0.0);
    let centre = scan.centre.mean_or(all_mean);
    let top = scan.top.mean_or(all_mean);
    let bottom = scan.bottom.mean_or(all_mean);
    let quantiles_s = quantiles(&mut scan.s, &PROBABILITIES_S, |s| s);
    // The logarithm is monotone, so the quantiles of `t` are the logarithms of the quantiles of `M`: selection runs
    // on the channel maxima, and only the picked neighbours take a logarithm.
    let quantiles_t = quantiles(&mut scan.maxima, &PROBABILITIES_T, |max_channel| {
        (max_channel / TARGET_GREY).log2() - key
    });

    let (key_bv, bv_known) = match inputs.bv_settings.filter(|bv| bv.is_finite()) {
        Some(bv) => (bv + key, 1.0),
        None => (0.0, 0.0),
    };
    let log_iso = match inputs.iso.filter(|iso| iso.is_finite() && *iso > 0.0) {
        Some(iso) => (iso / 100.0).log2(),
        None => 0.0,
    };
    let light = inputs.light;
    let [p_daylight, p_tungsten, p_fluorescent, p_led] = light.classes.to_array();

    let mut values = [0.0; FEATURE_COUNT];
    values[0] = key_bv;
    values[1] = bv_known;
    values[2..19].copy_from_slice(&quantiles_s);
    values[19..24].copy_from_slice(&quantiles_t);
    values[24] = (clip_y / TARGET_GREY).log2() - key;
    values[25] = scan.clipped as f64 / (preview.width * preview.height) as f64;
    values[26] = centre;
    values[27] = top - bottom;
    values[28] = scan.bright as f64 / usable;
    values[29] = scan.dark as f64 / usable;
    values[30..34].copy_from_slice(&[p_daylight, p_tungsten, p_fluorescent, p_led]);
    values[34] = 1e6 / light.cct_k / 100.0;
    values[35] = light.duv * 100.0;
    values[36] = light.confidence;
    values[37] = log_iso;
    values[38] = if inputs.flash_fired { 1.0 } else { 0.0 };

    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(AutosetError::NonFinite(FEATURE_NAMES[i]));
    }
    Ok(Features { values, key })
}

/// The part of the preview a pixel belongs to, for the thirds.
#[derive(Debug, Clone, Copy)]
enum Third {
    Top,
    Bottom,
    Centre,
}

/// The sum and count of `s - m` over one part of the preview.
#[derive(Debug, Clone, Copy, Default)]
struct Region {
    sum: f64,
    count: usize,
}

impl Region {
    fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    /// The mean, or `fallback` when no usable pixel lies in the part.
    fn mean_or(&self, fallback: f64) -> f64 {
        if self.count == 0 {
            fallback
        } else {
            self.sum / self.count as f64
        }
    }
}

/// What the single pass over the preview collects.
struct Scan {
    /// `s - m` of every usable pixel, in reading order (reordered by the quantiles).
    s: Vec<f64>,
    /// The largest AWG4 channel `M` of every usable pixel, in the same order, floored at [`FLOOR`]. The quantiles of
    /// `t` are taken on these, and their logarithms at the end.
    maxima: Vec<f64>,
    /// Usable pixels above [`BRIGHT_STOPS`] (strictly).
    bright: usize,
    /// Usable pixels below [`DARK_STOPS`] (strictly).
    dark: usize,
    /// Pixels whose clip mask is set, usable or not.
    clipped: usize,
    top: Region,
    bottom: Region,
    centre: Region,
    all: Region,
}

impl Scan {
    /// One pass over the preview: `Y` and `M` of every pixel with finite channels, the key subtracted, and the sums
    /// the spatial features need. Row by row, in a fixed order.
    fn new(preview: &PreviewRgb<'_>, matrix: &Mat3, key: f64) -> Self {
        let (width, height) = (preview.width, preview.height);
        let (tw, th) = (width / 3, height / 3);
        let luminance_row = PreviewRgb::luminance_row(matrix);
        let pixels = width * height;
        let mut scan = Scan {
            s: Vec::with_capacity(pixels),
            maxima: Vec::with_capacity(pixels),
            bright: 0,
            dark: 0,
            clipped: 0,
            top: Region::default(),
            bottom: Region::default(),
            centre: Region::default(),
            all: Region::default(),
        };
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                if preview.clip_mask.is_some_and(|mask| mask[i] != 0) {
                    scan.clipped += 1;
                }
                let px = &preview.rgb[3 * i..3 * i + 3];
                let rgb = [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
                if !rgb.iter().all(|c| c.is_finite()) {
                    continue;
                }
                // Y as `auto_exposure` computes it: the folded luminance row on the camera RGB.
                let y_lum = luminance_row[0] * rgb[0]
                    + luminance_row[1] * rgb[1]
                    + luminance_row[2] * rgb[2];
                let awg = mul_vec(matrix, rgb);
                if !(y_lum.is_finite() && awg.iter().all(|c| c.is_finite())) {
                    continue;
                }
                let s = (y_lum.max(FLOOR) / TARGET_GREY).log2() - key;
                let max_channel = awg[0].max(awg[1]).max(awg[2]).max(FLOOR);
                let third = if y < th {
                    Some(Third::Top)
                } else if y >= height - th {
                    Some(Third::Bottom)
                } else if (tw..width - tw).contains(&x) {
                    Some(Third::Centre)
                } else {
                    None
                };
                scan.push(s, max_channel, third);
            }
        }
        scan
    }

    /// Records one usable pixel: its `s - m`, its largest channel, and its place in the thirds.
    fn push(&mut self, s: f64, max_channel: f64, third: Option<Third>) {
        if s > BRIGHT_STOPS {
            self.bright += 1;
        }
        if s < DARK_STOPS {
            self.dark += 1;
        }
        self.all.add(s);
        match third {
            Some(Third::Top) => self.top.add(s),
            Some(Third::Bottom) => self.bottom.add(s),
            Some(Third::Centre) => self.centre.add(s),
            None => {}
        }
        self.s.push(s);
        self.maxima.push(max_channel);
    }
}

/// The linearly interpolated quantiles of `values` at `probs` (numpy's default, as
/// [`exposure::quantile`](crate::exposure)), by selection. `values` is reordered; NaN for no values.
///
/// `map` takes the selected values to the quantity whose quantiles are wanted. It must be non-decreasing, as the
/// logarithm is: then the order statistics of the mapped values are the mapped order statistics, and the result is
/// the sort-based quantile of the mapped list. Only the neighbours of each probability are mapped.
fn quantiles<const N: usize>(
    values: &mut [f64],
    probs: &[f64; N],
    map: impl Fn(f64) -> f64,
) -> [f64; N] {
    if values.is_empty() {
        return [f64::NAN; N];
    }
    let last = (values.len() - 1) as f64;
    // (lower rank, upper rank, fraction) of every probability.
    let positions: [(usize, usize, f64); N] = std::array::from_fn(|k| {
        let pos = probs[k].clamp(0.0, 1.0) * last;
        let lo = pos.floor() as usize;
        (lo, pos.ceil() as usize, pos - lo as f64)
    });
    let mut ranks: Vec<usize> = positions.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let mut picked = vec![0.0; ranks.len()];
    select_ranks(values, 0, &ranks, &mut picked);
    std::array::from_fn(|k| {
        let (lo, hi, frac) = positions[k];
        // Both ranks are in `ranks` by construction, so `partition_point` finds their slots exactly.
        let a = map(picked[ranks.partition_point(|&r| r < lo)]);
        let b = map(picked[ranks.partition_point(|&r| r < hi)]);
        // A monotone map keeps `a <= b`. The swap only guards `interpolate`'s clamp against an ulp-level break.
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        interpolate(a, b, frac)
    })
}

/// Writes the order statistics of a slice at `ranks` into `out` (both parallel to each other). `values` holds the
/// absolute positions `base..base + values.len()`, and `ranks` are ascending, distinct and inside that range.
///
/// Each rank is selected with `select_nth_unstable_by` on the part of the buffer that the earlier selections left
/// it in, the middle rank first: the elements below it are searched for the lower ranks, the ones above for the
/// upper ranks. The selected values are exact copies, so the result is the same as a sort's.
fn select_ranks(values: &mut [f64], base: usize, ranks: &[usize], out: &mut [f64]) {
    if ranks.is_empty() {
        return;
    }
    let mid = ranks.len() / 2;
    let (below, pivot, above) = values.select_nth_unstable_by(ranks[mid] - base, f64::total_cmp);
    out[mid] = *pivot;
    select_ranks(below, base, &ranks[..mid], &mut out[..mid]);
    select_ranks(
        above,
        ranks[mid] + 1,
        &ranks[mid + 1..],
        &mut out[mid + 1..],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exposure::quantile as sort_quantile;

    /// SplitMix64, the crate's seeded generator for tests (the same constants as the exposure tests).
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn feature_names_are_the_spec_list_in_order() {
        assert_eq!(FEATURE_NAMES.len(), 39);
        assert_eq!(FEATURE_NAMES[0], "key_bv");
        assert_eq!(FEATURE_NAMES[2], "q01");
        assert_eq!(FEATURE_NAMES[18], "q999");
        assert_eq!(FEATURE_NAMES[19], "mq50");
        assert_eq!(FEATURE_NAMES[23], "mq999");
        assert_eq!(FEATURE_NAMES[24], "clip_headroom");
        assert_eq!(FEATURE_NAMES[30], "p_daylight");
        assert_eq!(FEATURE_NAMES[33], "p_led");
        assert_eq!(FEATURE_NAMES[38], "flash");
        let mut names = FEATURE_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FEATURE_COUNT, "names are distinct");
    }

    #[test]
    fn selection_quantiles_equal_sort_quantiles_bit_for_bit() {
        let mut rng = Rng(7);
        for round in 0..200 {
            let n = 1 + (rng.next_u64() % 3000) as usize;
            // Mixed signs, duplicates, the floor value and negative zero, as luminance logs produce.
            let data: Vec<f64> = (0..n)
                .map(|_| match rng.next_u64() % 8 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => 1.0,
                    3 => -1.0,
                    _ => (rng.next_u64() % 2_000_001) as f64 / 100_000.0 - 10.0,
                })
                .collect();
            let mut sorted = data.clone();
            sorted.sort_unstable_by(f64::total_cmp);
            let probs: [f64; 5] = [0.0, 0.5, 0.999, 1.0, (round as f64 + 0.5) / 201.0];
            let mut scratch = data.clone();
            let ours = quantiles(&mut scratch, &probs, |v| v);
            for (p, got) in probs.iter().zip(ours) {
                let want = sort_quantile(&sorted, *p);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "round {round}, n {n}, p {p}: {got} vs {want}"
                );
            }
            // The fixed probabilities of the features too.
            let mut scratch = data.clone();
            let ours = quantiles(&mut scratch, &PROBABILITIES_S, |v| v);
            for (p, got) in PROBABILITIES_S.iter().zip(ours) {
                assert_eq!(got.to_bits(), sort_quantile(&sorted, *p).to_bits());
            }
            // Positive data mapped by the logarithm of the features: the quantiles of the mapped list.
            let positive: Vec<f64> = data.iter().map(|v| v.abs() + 1e-6).collect();
            let log = |v: f64| (v / 0.18).log2() - 0.25;
            let mut mapped: Vec<f64> = positive.iter().map(|v| log(*v)).collect();
            mapped.sort_unstable_by(f64::total_cmp);
            let mut scratch = positive.clone();
            let ours = quantiles(&mut scratch, &PROBABILITIES_T, log);
            for (p, got) in PROBABILITIES_T.iter().zip(ours) {
                assert_eq!(got.to_bits(), sort_quantile(&mapped, *p).to_bits());
            }
        }
    }

    #[test]
    fn selection_quantiles_of_a_ramp_are_the_interpolated_ranks() {
        // The values 0, 1, ..., 9 in reverse order: the quantile at p sits at position 9p.
        let mut ramp: Vec<f64> = (0..10).rev().map(f64::from).collect();
        let got = quantiles(&mut ramp, &[0.0, 0.25, 0.5, 0.95, 1.0], |v| v);
        for (g, want) in got.iter().zip([0.0, 2.25, 4.5, 8.55, 9.0]) {
            assert!((g - want).abs() < 1e-12, "{g} vs {want}");
        }
    }

    #[test]
    fn a_single_value_is_every_quantile() {
        let mut one = vec![-2.5];
        assert_eq!(quantiles(&mut one, &PROBABILITIES_T, |v| v), [-2.5; 5]);
        let mut none: Vec<f64> = Vec::new();
        assert!(quantiles(&mut none, &PROBABILITIES_T, |v| v)[0].is_nan());
    }
}
