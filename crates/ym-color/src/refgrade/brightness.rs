//! Match brightness: the gap between the photo's and the reference's middle brightness and the exposure shift that moves the photo
//! by a share of it (`docs/34-reference-grade.md` section 5.8, RG-16).
//!
//! The fit keeps the photo's own median (`T.p50`). With Match brightness `b` above 0 it first makes a moved photo `P'`: the photo with
//! an exposure shift `d`, found by a bisection so that the photo's *displayed* median luminance moves by `b` times the gap to the
//! reference's. The shapes here are pure and have no I/O; the render that measures a shift is the caller's (a closure).

use super::measure::{Px, median_lightness};

/// The gap `g` between the reference's and the photo's median luminance is held to this many stops either way.
pub const GAP_HOLD_EV: f64 = 1.5;
/// The exposure shifts the bisection looks in, in stops.
pub const SHIFT_RANGE_EV: (f64, f64) = (-2.0, 2.0);
/// The bisection's steps: the shift is found to `4 / 2^16` of a stop.
pub const BISECTION_STEPS: usize = 16;
/// The lowest L\* the gap reads: a black picture's median is held here, so the luminance ratio stays a number.
pub const MIN_LSTAR: f64 = 1.0;

/// The relative luminance `Y` (0 to 1) of a CIELAB lightness: the inverse of CIELAB's `L*`, with `L*` held at [`MIN_LSTAR`] or more
/// and at 100 or less.
#[must_use]
pub fn luminance_of_lstar(l: f64) -> f64 {
    const EPSILON: f64 = 216.0 / 24389.0;
    const KAPPA: f64 = 24389.0 / 27.0;
    let l = l.clamp(MIN_LSTAR, 100.0);
    let f = (l + 16.0) / 116.0;
    let cubed = f * f * f;
    if cubed > EPSILON { cubed } else { l / KAPPA }
}

/// The gap between two pictures' brightness, in stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap {
    /// `log2(Y_R / Y_P)` before the hold: positive when the reference is brighter.
    pub raw_ev: f64,
    /// The gap held to [`GAP_HOLD_EV`] either way.
    pub held_ev: f64,
    /// Whether the hold changed it.
    pub limited: bool,
}

impl Gap {
    /// The gap from the two medians' L\*.
    #[must_use]
    pub fn of_lstar(photo_l: f64, reference_l: f64) -> Self {
        let raw_ev = (luminance_of_lstar(reference_l) / luminance_of_lstar(photo_l)).log2();
        let held_ev = raw_ev.clamp(-GAP_HOLD_EV, GAP_HOLD_EV);
        Self {
            raw_ev,
            held_ev,
            limited: held_ev != raw_ev,
        }
    }

    /// The gap between the median L\* of two pictures' pixels, as the fit measures them.
    ///
    /// # Panics
    ///
    /// When a picture is empty.
    #[must_use]
    pub fn of_pictures(photo: &[Px], reference: &[Px]) -> Self {
        Self::of_lstar(median_lightness(photo), median_lightness(reference))
    }
}

/// `b` for a Match brightness in percent: 0 to 1.
#[must_use]
pub fn share_of_percent(percent: f64) -> f64 {
    (percent / 100.0).clamp(0.0, 1.0)
}

/// The move asked for, in stops: `Δ = b * g` for the held gap `g`.
#[must_use]
pub fn asked_ev(percent: f64, gap: &Gap) -> f64 {
    share_of_percent(percent) * gap.held_ev
}

/// The exposure shift, in stops, that brings the photo's displayed median luminance to `target_y`, found by a bisection of
/// [`BISECTION_STEPS`] steps in [`SHIFT_RANGE_EV`]. `median_y_at(d)` renders the photo with the shift `d` and returns its median
/// luminance (it must not fall as `d` rises). When no shift in the range reaches the target the nearer end is taken. Deterministic:
/// the same closure gives the same shift.
///
/// # Errors
///
/// The closure's own.
pub fn solve_shift<E>(
    target_y: f64,
    mut median_y_at: impl FnMut(f64) -> Result<f64, E>,
) -> Result<f64, E> {
    let (mut lo, mut hi) = SHIFT_RANGE_EV;
    let (mut lo_moved, mut hi_moved) = (false, false);
    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        if median_y_at(mid)? < target_y {
            lo = mid;
            lo_moved = true;
        } else {
            hi = mid;
            hi_moved = true;
        }
    }
    // A target no shift reaches left one end untouched: that end is the answer.
    Ok(match (lo_moved, hi_moved) {
        (true, false) => SHIFT_RANGE_EV.1,
        (false, true) => SHIFT_RANGE_EV.0,
        _ => 0.5 * (lo + hi),
    })
}

/// What Match brightness did in a fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrightnessMove {
    /// The field's value, in percent.
    pub percent: f64,
    /// The gap between the reference's and the photo's median, held to 1.5 stops (0 when Match brightness is 0: nothing is measured).
    pub gap_ev: f64,
    /// The move asked for, `b * gap_ev`.
    pub asked_ev: f64,
    /// The exposure shift the bisection found.
    pub shift_ev: f64,
    /// The move the photo's displayed median made: `log2(Y_P' / Y_P)`. It is `asked_ev` within the bisection's step, or less when no
    /// shift in range reaches it.
    pub moved_ev: f64,
    /// Whether the gap was held at 1.5 stops.
    pub limited: bool,
}

impl BrightnessMove {
    /// Match brightness at 0: nothing moved, nothing measured.
    #[must_use]
    pub fn none() -> Self {
        Self {
            percent: 0.0,
            gap_ev: 0.0,
            asked_ev: 0.0,
            shift_ev: 0.0,
            moved_ev: 0.0,
            limited: false,
        }
    }
}
