//! The fit's parameters (`docs/34` section 5.2): 49 normalised numbers from -1 to 1, the identity at 0, and their map to the node's
//! fields.

use crate::grade::logc4_at_stops;

use super::objective::ELEMENT_CAP_DEG;

/// The global numbers: tone (8), colour per band (9), saturation (1), `hue_vs_sat` (6), `hue_vs_hue` (6).
pub const N_GLOBAL: usize = 30;
/// The local numbers: three for each of the six elements and the subject's exposure.
pub const N_LOCAL: usize = 19;
/// All the numbers.
pub const N_PARAMS: usize = N_GLOBAL + N_LOCAL;
/// The luma curve's knots, in stops from scene grey 0.18.
pub const LUMA_STOPS: [f64; 8] = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
/// The red, green and blue curves' knots, in stops.
pub const RGB_STOPS: [f64; 3] = [-3.0, 0.0, 3.0];
/// The hue curves' knots, in degrees.
pub const HUE_KNOTS_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
/// The index of the saturation number.
pub const SATURATION_INDEX: usize = 17;
/// The first `hue_vs_sat` number.
pub const HUE_VS_SAT_START: usize = 18;
/// The first `hue_vs_hue` number.
pub const HUE_VS_HUE_START: usize = 24;
/// The index of the subject's exposure number.
pub const SUBJECT_INDEX: usize = N_PARAMS - 1;

/// The normalised numbers.
pub type Params = [f64; N_PARAMS];

/// The first knot's offset is this many stops at `x = 1`.
const LUMA_OFFSET_STOPS: f64 = 2.0;
/// A luma segment's slope in stop space is `exp(this * x)`.
const LUMA_LOG_SLOPE: f64 = 1.1;
/// A channel knot moves this many stops at `x = 1`.
const RGB_STOPS_RANGE: f64 = 1.0;
/// Saturation, in percent, at `x = 1`.
const SATURATION_PCT: f64 = 60.0;
/// `hue_vs_sat` is `1 + this * x`.
const HUE_SAT_RANGE: f64 = 0.7;
/// `hue_vs_hue` is `this * x` degrees.
const HUE_DEG_RANGE: f64 = 30.0;
/// An element's chroma ratio is this to the `x` for `x >= 0` ...
const CHROMA_MAX: f64 = 1.6;
/// ... and this to the `-x` for `x < 0`.
const CHROMA_MIN: f64 = 0.5;

/// One element's numbers in the node's units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementNumbers {
    /// The exposure offset in stops.
    pub exposure_ev: f64,
    /// The hue turn in degrees.
    pub hue_deg: f64,
    /// The chroma ratio.
    pub chroma: f64,
}

impl ElementNumbers {
    /// The identity: 0, 0, 1.
    pub const IDENTITY: Self = Self {
        exposure_ev: 0.0,
        hue_deg: 0.0,
        chroma: 1.0,
    };
}

/// The node's fitted fields as numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Numbers {
    /// The luma curve's knots `[input code, output code]`, with the end points (0, 0) and (1, 1); `None` when its numbers are all 0
    /// (the identity curve, which the node keeps as its default).
    pub luma: Option<Vec<[f64; 2]>>,
    /// The red, green and blue curves, as `luma`.
    pub rgb: [Option<Vec<[f64; 2]>>; 3],
    /// The manual saturation, in percent.
    pub saturation_pct: f64,
    /// The six `hue_vs_sat` values at 0, 60, ... 300 degrees.
    pub hue_vs_sat: [f64; 6],
    /// The six `hue_vs_hue` values in degrees.
    pub hue_vs_hue_deg: [f64; 6],
    /// Sky, vegetation, water, ground, built, skin.
    pub elements: [ElementNumbers; 6],
    /// The subject's exposure offset in stops.
    pub subject_ev: f64,
}

fn knots(points: &[(f64, f64)]) -> Vec<[f64; 2]> {
    let mut out = vec![[0.0, 0.0]];
    out.extend(points.iter().map(|&(x, y)| [x, y]));
    out.push([1.0, 1.0]);
    out
}

/// The numbers the normalised parameters `x` stand for, as in `docs/34` section 5.2. The hue curves are always six points.
#[must_use]
pub fn to_numbers(x: &Params) -> Numbers {
    // Luma: the first knot's output offset, then the slope of each segment in stop space. A curve is increasing by construction.
    let luma = if x[..8].iter().all(|&v| v == 0.0) {
        None
    } else {
        let mut y_stops = LUMA_STOPS[0] + x[0] * LUMA_OFFSET_STOPS;
        let mut prev_code = 0.0f64;
        let mut pts = Vec::with_capacity(8);
        for (k, &s) in LUMA_STOPS.iter().enumerate() {
            if k > 0 {
                y_stops += (s - LUMA_STOPS[k - 1]) * (x[k] * LUMA_LOG_SLOPE).exp();
            }
            let code = logc4_at_stops(y_stops).clamp(prev_code, 1.0);
            prev_code = code;
            pts.push((logc4_at_stops(s), code));
        }
        Some(knots(&pts))
    };
    let mut rgb: [Option<Vec<[f64; 2]>>; 3] = [None, None, None];
    for (c, curve) in rgb.iter_mut().enumerate() {
        let xs = &x[8 + 3 * c..11 + 3 * c];
        if xs.iter().all(|&v| v == 0.0) {
            continue;
        }
        let pts: Vec<(f64, f64)> = RGB_STOPS
            .iter()
            .zip(xs)
            .map(|(&s, &v)| {
                (
                    logc4_at_stops(s),
                    logc4_at_stops(s + v * RGB_STOPS_RANGE).clamp(0.0, 1.0),
                )
            })
            .collect();
        *curve = Some(knots(&pts));
    }
    let mut hue_vs_sat = [1.0; 6];
    let mut hue_vs_hue_deg = [0.0; 6];
    for k in 0..6 {
        hue_vs_sat[k] = 1.0 + x[HUE_VS_SAT_START + k] * HUE_SAT_RANGE;
        hue_vs_hue_deg[k] = x[HUE_VS_HUE_START + k] * HUE_DEG_RANGE;
    }
    let mut elements = [ElementNumbers::IDENTITY; 6];
    for (k, e) in elements.iter_mut().enumerate() {
        let b = N_GLOBAL + 3 * k;
        let c = x[b + 2];
        *e = ElementNumbers {
            exposure_ev: x[b],
            hue_deg: x[b + 1] * ELEMENT_CAP_DEG[k],
            chroma: if c >= 0.0 {
                CHROMA_MAX.powf(c)
            } else {
                CHROMA_MIN.powf(-c)
            },
        };
    }
    Numbers {
        luma,
        rgb,
        saturation_pct: x[SATURATION_INDEX] * SATURATION_PCT,
        hue_vs_sat,
        hue_vs_hue_deg,
        elements,
        subject_ev: x[SUBJECT_INDEX],
    }
}

impl Numbers {
    /// Whether every number is the identity: the fit's result is then no change.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.luma.is_none()
            && self.rgb.iter().all(Option::is_none)
            && self.saturation_pct == 0.0
            && self.hue_vs_sat.iter().all(|&v| v == 1.0)
            && self.hue_vs_hue_deg.iter().all(|&v| v == 0.0)
            && self.elements.iter().all(|e| *e == ElementNumbers::IDENTITY)
            && self.subject_ev == 0.0
    }
}

/// The name of number `i`, for a report.
#[must_use]
pub fn param_name(i: usize) -> String {
    match i {
        0 => "luma.offset_stops".into(),
        1..=7 => format!("luma.log_slope[{i}]"),
        8..=16 => {
            let c = ["red", "green", "blue"][(i - 8) / 3];
            format!("{c}.shift_stops[{:+}]", RGB_STOPS[(i - 8) % 3])
        }
        SATURATION_INDEX => "saturation_pct".into(),
        18..=23 => format!("hue_vs_sat[{}deg]", HUE_KNOTS_DEG[i - 18]),
        24..=29 => format!("hue_vs_hue[{}deg]", HUE_KNOTS_DEG[i - 24]),
        SUBJECT_INDEX => "local.subject.exposure_stops".into(),
        _ => format!(
            "local.{}.{}",
            super::ELEMENT_NAMES[(i - N_GLOBAL) / 3],
            ["exposure_stops", "hue_deg", "chroma_scale"][(i - N_GLOBAL) % 3]
        ),
    }
}
