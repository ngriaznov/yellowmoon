//! Robust order statistics used by the estimator. All functions reorder their input slice.

use std::cmp::Ordering;

fn cmp(a: &f32, b: &f32) -> Ordering {
    a.total_cmp(b)
}

/// Median of `v` (mean of the two middle values for an even length). `NaN`-free input expected.
/// Returns `0.0` for an empty slice.
pub fn median(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (_, &mut hi, _) = v.select_nth_unstable_by(mid, cmp);
    if n % 2 == 1 {
        hi
    } else {
        // The lower middle value is the maximum of the left partition.
        let lo = v[..mid].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        0.5 * (lo + hi)
    }
}

/// The value at quantile `q` in [0, 1] using the nearest-rank rule on the sorted order.
/// Returns `0.0` for an empty slice.
pub fn quantile(v: &mut [f32], q: f64) -> f32 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let k = ((q * n as f64).ceil() as usize).clamp(1, n) - 1;
    let (_, &mut x, _) = v.select_nth_unstable_by(k, cmp);
    x
}

/// Consistency factor turning a median absolute deviation into a Gaussian standard deviation.
pub const MAD_TO_SIGMA: f64 = 1.4826;

/// Robust variance `(1.4826 * MAD)^2`, with the MAD taken about the median of `v`.
pub fn mad_variance(v: &mut [f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = median(v);
    for x in v.iter_mut() {
        *x = (*x - m).abs();
    }
    let mad = f64::from(median(v));
    (MAD_TO_SIGMA * mad).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even_and_empty() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(median(&mut [7.0]), 7.0);
    }

    #[test]
    fn quantile_nearest_rank() {
        let mut v: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        assert_eq!(quantile(&mut v, 0.4), 4.0);
        assert_eq!(quantile(&mut v, 1.0), 10.0);
        assert_eq!(quantile(&mut v, 0.0), 1.0);
    }

    #[test]
    fn mad_variance_ignores_outliers() {
        let mut v: Vec<f32> = (0..1000).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let clean = mad_variance(&mut v.clone());
        v.extend([100.0, -80.0, 50.0]);
        let dirty = mad_variance(&mut v);
        assert!((dirty / clean - 1.0).abs() < 0.05);
    }
}
