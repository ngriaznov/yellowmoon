//! The preferred white (`docs/18-auto-settings.md` §5): the user white that Auto white balance sets for a photo.
//!
//! The scene light (what the photo was lit by) is never changed. Only the user white, the light the chain is told to
//! neutralise, moves: daylight stays neutral, while warm light and dim scenes keep part of their warmth. The rule
//! is a starting point with four constants ([`PreferredWhite`]), to be checked by eye (AS-H2); a learned map may
//! replace it in a later version.

/// The constants of the preferred-white rule (`docs/18` §5). [`PreferredWhite::default`] is the v1 starting point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreferredWhite {
    /// `R_warm`: how far the neutralisation of a warm light is held back at full warmth, so that `D_w = 1 - R_warm`
    /// for a light `w_span_mired` warmer than the reference (0.30 in v1).
    pub r_warm: f64,
    /// `W_span` in mired: the warmth over which `R_warm` ramps in (150 in v1).
    pub w_span_mired: f64,
    /// `R_dim`: how far the neutralisation of a dim scene is held back, in full at a Bv of `b_dim - b_span` and below
    /// (0.15 in v1).
    pub r_dim: f64,
    /// `B_dim`: the key brightness value (Bv) below which the dim term starts, 2 in v1.
    pub b_dim: f64,
    /// `B_span`: the Bv range over which the dim term ramps, 4 in v1.
    pub b_span: f64,
    /// `d_min`: the lowest value of `D`, 0.5 in v1.
    pub d_min: f64,
    /// `mired_ref`: the mired of the reference white, a surface neutral at 6000 K (`1e6 / 6000`, Zhai and Luo 2017).
    pub mired_ref: f64,
}

impl Default for PreferredWhite {
    /// The constants of `docs/18` §5.
    fn default() -> Self {
        Self {
            r_warm: 0.30,
            w_span_mired: 150.0,
            r_dim: 0.15,
            b_dim: 2.0,
            b_span: 4.0,
            d_min: 0.5,
            mired_ref: 1e6 / 6000.0,
        }
    }
}

/// The smoothstep of `docs/18` §5: 0 at or below 0, 1 at or above 1, `3x² - 2x³` between.
fn smoothstep(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * (3.0 - 2.0 * x)
    }
}

/// The `(cct_k, duv)` of the user white for a scene light (`docs/18` §5).
///
/// `cct_k` and `duv` are the scene light's own readout (its `SceneLight::cct_k` and `duv`); `key_bv` the brightness of
/// the scene (`Features::values[0]` when the settings' Bv is known, `None` otherwise); `strength` the `white_preference`
/// setting, 0 to 1, which scales `1 - D` (values outside are clamped; NaN counts as 0).
///
/// The result is the input, unchanged, when `strength` is 0, when the light is no warmer than the reference, when
/// `D` is 1 (the photo is neutralised as before), and when `cct_k` is not positive and finite. Otherwise only the mired
/// moves: `mired_u = mired_ref + D' (mired_s - mired_ref)` with `D' = 1 - strength (1 - D)`. The Duv is kept as it is.
#[must_use]
pub fn preferred_white(
    cct_k: f64,
    duv: f64,
    key_bv: Option<f64>,
    strength: f64,
    p: &PreferredWhite,
) -> (f64, f64) {
    let input = (cct_k, duv);
    if strength.is_nan() || strength <= 0.0 || !(cct_k.is_finite() && cct_k > 0.0) {
        return input;
    }
    let mired_s = 1e6 / cct_k;
    let warm = mired_s - p.mired_ref;
    if warm.is_nan() || warm <= 0.0 {
        return input;
    }
    let d_warm = 1.0 - p.r_warm * smoothstep(warm / p.w_span_mired);
    let d_dim = match key_bv.filter(|bv| bv.is_finite()) {
        Some(bv) => 1.0 - p.r_dim * smoothstep((p.b_dim - bv) / p.b_span),
        None => 1.0,
    };
    let d = (d_warm * d_dim).max(p.d_min).min(1.0);
    let d_effective = 1.0 - strength.min(1.0) * (1.0 - d);
    if d_effective >= 1.0 {
        return input;
    }
    // `mired_ref + d (mired_s - mired_ref)`, written as `mired_s - (1 - d) warm`: exact when `d` is 1.
    let mired_u = mired_s - (1.0 - d_effective) * warm;
    (1e6 / mired_u, duv)
}
