//! The labels of the `autoset` dataset (`docs/18-auto-settings.md` section 6): the quantile tables the dump writes and
//! the exposure label `L` of section 6.1, ported from `training/autoset/labels.py::exposure_fit`.
//!
//! The port is exact up to floating point: the same selection of the 0.02 to 0.98 points of the grid, the same
//! mid-tone band rule, the same cumulative maximum, and `camera_gain_ev` with its range check, its log-log
//! interpolation (stable order, the `1e-12` de-duplication that keeps the first of a run, linear extrapolation from
//! the two outermost points, `np.interp` inside) and `L = log2(0.18 / x)`. `training/autoset/dataset.py` recomputes
//! `L` from the stored tables and fails on a difference above 1e-9, so the two must agree.

use ym_color::exposure::TARGET_GREY;

/// Number of points of the quantile tables: `linspace(0.001, 0.999, 999)` (`docs/18` section 6).
pub const QUANTILE_GRID_LEN: usize = 999;

/// The first grid point of the tables (`numpy.linspace`'s start).
const GRID_START: f64 = 0.001;
/// The last grid point of the tables (`numpy.linspace`'s stop, which numpy sets exactly).
const GRID_STOP: f64 = 0.999;
/// The exposure label reads the points between these probabilities (`INNER` of `training/autoset/labels.py`).
const INNER_LOW: f64 = 0.02;
/// See [`INNER_LOW`].
const INNER_HIGH: f64 = 0.98;
/// The mid-tone band, in stops around middle grey (`BAND_STOPS`): `[0.18 * 2^-2, 0.18 * 2^+1]`.
const BAND_LOW_STOPS: f64 = -2.0;
/// See [`BAND_LOW_STOPS`].
const BAND_HIGH_STOPS: f64 = 1.0;
/// The band is used when at least this many points lie in it (`MIN_BAND_POINTS`).
const MIN_BAND_POINTS: usize = 20;
/// The camera's JPEG may be this many times above or below the grey point, and no more (`curve_inverse_at`).
const RANGE_FACTOR: f64 = 4.0;
/// Two curve points closer than this in log scene level are one point (`_log_interp`'s `1e-12`).
const DUPLICATE_LOG_GAP: f64 = 1e-12;

/// The quantile probabilities of the tables: `linspace(0.001, 0.999, 999)`, bit for bit as numpy computes it.
///
/// numpy takes `step = (stop - start) / (num - 1)`, the points `start + i * step`, and then sets the last point to
/// `stop` exactly. The dump writes its tables on these probabilities and the labels read them, so both sides use
/// this function.
#[must_use]
pub fn quantile_grid() -> Vec<f64> {
    let step = (GRID_STOP - GRID_START) / (QUANTILE_GRID_LEN - 1) as f64;
    let mut grid: Vec<f64> = (0..QUANTILE_GRID_LEN)
        .map(|i| GRID_START + i as f64 * step)
        .collect();
    grid[QUANTILE_GRID_LEN - 1] = GRID_STOP;
    grid
}

/// The linearly interpolated quantiles of `values` at the probabilities of `grid` (numpy's default method, `linear`).
///
/// Non-finite values are dropped first. The values are sorted once. Returns NaN for every probability when no finite
/// value is left. The interpolation is numpy's `_lerp`, so that a quantile that falls on a run of equal values is that
/// value exactly.
pub fn quantile_table(values: &mut Vec<f64>, grid: &[f64]) -> Vec<f64> {
    values.retain(|v| v.is_finite());
    values.sort_unstable_by(f64::total_cmp);
    grid.iter().map(|&p| quantile_sorted(values, p)).collect()
}

/// The linear quantile of an ascending slice (empty gives NaN). `q` is clamped to 0 to 1.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    lerp(sorted[lo], sorted[hi], pos - lo as f64)
}

/// numpy's `_lerp`: `a + (b - a) t`, written from `b` for `t >= 0.5`, and kept inside `[a, b]`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let v = if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    };
    v.clamp(a, b)
}

/// The exposure label `L` of a photo from its two quantile tables (`docs/18` section 6.1): `log2(0.18 / x)`, where
/// `x` is the scene level at which the camera's JPEG reaches middle grey.
///
/// `scene_q` holds the quantiles of our scene-linear luminance (AWG4 `Y` at EV 0) and `jpeg_q` those of the camera's
/// JPEG's linear luminance, both on [`quantile_grid`] and both at the common size. `None` when either table does not
/// have [`QUANTILE_GRID_LEN`] points, or when the curve does not reach middle grey within the range the table allows
/// (the `NaN` of `training/autoset/labels.py`).
#[must_use]
pub fn exposure_label(scene_q: &[f64], jpeg_q: &[f64]) -> Option<f64> {
    if scene_q.len() != QUANTILE_GRID_LEN || jpeg_q.len() != QUANTILE_GRID_LEN {
        return None;
    }
    let grid = quantile_grid();
    let inner: Vec<usize> = (0..QUANTILE_GRID_LEN)
        .filter(|&i| (INNER_LOW..=INNER_HIGH).contains(&grid[i]))
        .collect();
    let scene: Vec<f64> = inner.iter().map(|&i| scene_q[i]).collect();
    let jpeg: Vec<f64> = inner.iter().map(|&i| jpeg_q[i]).collect();

    let band_low = TARGET_GREY * BAND_LOW_STOPS.exp2();
    let band_high = TARGET_GREY * BAND_HIGH_STOPS.exp2();
    let in_band: Vec<bool> = jpeg
        .iter()
        .map(|&y| y >= band_low && y <= band_high)
        .collect();
    let use_band = in_band.iter().filter(|&&b| b).count() >= MIN_BAND_POINTS;

    let (mut used_scene, mut used_jpeg) = (Vec::new(), Vec::new());
    for i in 0..scene.len() {
        if !use_band || in_band[i] {
            used_scene.push(scene[i]);
            used_jpeg.push(jpeg[i]);
        }
    }
    let curve_scene = cumulative_max(&used_scene);
    let curve_jpeg = cumulative_max(&used_jpeg);

    let level = gain_ev(&curve_scene, &curve_jpeg)?;
    level.is_finite().then_some(level)
}

/// `np.maximum.accumulate`: the running maximum, which stays NaN once a NaN has appeared (numpy's `maximum`
/// propagates NaN).
fn cumulative_max(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut running: Option<f64> = None;
    for &v in values {
        let next = match running {
            None => v,
            Some(r) if r.is_nan() || v.is_nan() => f64::NAN,
            Some(r) => r.max(v),
        };
        out.push(next);
        running = Some(next);
    }
    out
}

/// `camera_gain_ev` of `training/autoset/metrics.py`: `log2(anchor / x)` for the scene level `x` at which the curve
/// (scene, JPEG) reaches [`TARGET_GREY`]; `None` when there is no such level (`NaN` in Python).
fn gain_ev(scene: &[f64], jpeg: &[f64]) -> Option<f64> {
    let x = scene_level_at(scene, jpeg, TARGET_GREY)?;
    (x > 0.0).then(|| (TARGET_GREY / x).log2())
}

/// `curve_inverse_at`: the scene level at which the JPEG column reaches `y`. `None` when the JPEG column has no
/// positive value, or when `y` lies more than two stops outside the positive range of the column.
fn scene_level_at(scene: &[f64], jpeg: &[f64], y: f64) -> Option<f64> {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in jpeg.iter().filter(|&&v| v > 0.0) {
        low = low.min(v);
        high = high.max(v);
    }
    if low > high || y < low / RANGE_FACTOR || y > high * RANGE_FACTOR {
        return None;
    }
    log_interp(y, jpeg, scene)
}

/// `_log_interp(x, xs, ys)` of the study: piecewise linear in log-log between the points with `xs > 0` and `ys > 0`,
/// linear extrapolation from the two outermost points, `None` where Python gives NaN.
fn log_interp(x: f64, xs: &[f64], ys: &[f64]) -> Option<f64> {
    // Stable sort by the abscissa (numpy's `argsort(kind="stable")`), keeping the pairs together.
    let mut points: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter(|(a, b)| **a > 0.0 && **b > 0.0)
        .map(|(a, b)| (a.ln(), b.ln()))
        .collect();
    if points.len() < 2 || x <= 0.0 || x.is_nan() {
        return None;
    }
    points.sort_by(|p, q| p.0.total_cmp(&q.0));
    // Keep the first of each run of abscissae closer than DUPLICATE_LOG_GAP. The gap is taken to the previous point
    // of the sorted list, as `np.diff` does, not to the last kept one.
    let mut lx = Vec::with_capacity(points.len());
    let mut ly = Vec::with_capacity(points.len());
    for (i, &(a, b)) in points.iter().enumerate() {
        if i == 0 || a - points[i - 1].0 > DUPLICATE_LOG_GAP {
            lx.push(a);
            ly.push(b);
        }
    }
    if lx.len() < 2 {
        return None;
    }
    let n = lx.len();
    let t = x.ln();
    if t <= lx[0] {
        let slope = (ly[1] - ly[0]) / (lx[1] - lx[0]);
        return Some((ly[0] + slope * (t - lx[0])).exp());
    }
    if t >= lx[n - 1] {
        let slope = (ly[n - 1] - ly[n - 2]) / (lx[n - 1] - lx[n - 2]);
        return Some((ly[n - 1] + slope * (t - lx[n - 1])).exp());
    }
    // np.interp inside the range: the segment j with lx[j] <= t < lx[j + 1].
    let j = lx.partition_point(|&v| v <= t) - 1;
    let slope = (ly[j + 1] - ly[j]) / (lx[j + 1] - lx[j]);
    Some((slope * (t - lx[j]) + ly[j]).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_has_the_numpy_end_points_and_is_ascending() {
        let grid = quantile_grid();
        assert_eq!(grid.len(), QUANTILE_GRID_LEN);
        assert_eq!(grid[0], 0.001);
        assert_eq!(grid[QUANTILE_GRID_LEN - 1], 0.999);
        assert!(grid.windows(2).all(|w| w[0] < w[1]));
        // numpy's grid has 0.02 at index 19 and 0.98 at index 979, so 961 points lie in INNER.
        assert_eq!(grid[19], 0.02);
        assert_eq!(grid[979], 0.98);
        assert_eq!(
            grid.iter()
                .filter(|&&p| (INNER_LOW..=INNER_HIGH).contains(&p))
                .count(),
            961
        );
    }

    #[test]
    fn quantiles_of_a_ramp_are_the_interpolated_ranks() {
        // The values 0 to 9 (given in reverse): the quantile at p sits at position 9 p.
        let mut values: Vec<f64> = (0..10).rev().map(f64::from).collect();
        let got = quantile_table(&mut values, &[0.0, 0.25, 0.5, 0.95, 1.0]);
        for (g, want) in got.iter().zip([0.0, 2.25, 4.5, 8.55, 9.0]) {
            assert!((g - want).abs() < 1e-12, "{g} vs {want}");
        }
    }

    #[test]
    fn quantiles_drop_non_finite_values_and_empty_input_is_nan() {
        let mut values = vec![f64::NAN, 2.0, f64::INFINITY, 4.0];
        assert_eq!(
            quantile_table(&mut values, &[0.0, 0.5, 1.0]),
            vec![2.0, 3.0, 4.0]
        );
        let mut none = vec![f64::NAN];
        assert!(quantile_table(&mut none, &[0.5])[0].is_nan());
    }

    #[test]
    fn a_curve_that_reaches_grey_gives_the_stop_count_of_its_gain() {
        // JPEG = 2 x scene exactly: the scene level for grey is 0.09, L = log2(0.18 / 0.09) = 1.
        let grid = quantile_grid();
        let scene: Vec<f64> = grid.iter().map(|p| 0.02 + 0.5 * p).collect();
        let jpeg: Vec<f64> = scene.iter().map(|v| 2.0 * v).collect();
        let level = exposure_label(&scene, &jpeg).expect("a level");
        assert!((level - 1.0).abs() < 1e-9, "{level}");
    }

    #[test]
    fn tables_of_the_wrong_length_have_no_label() {
        assert_eq!(exposure_label(&[0.1; 3], &[0.1; 3]), None);
    }
}
