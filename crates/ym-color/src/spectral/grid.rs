//! Uniform temperature grids and their mired-weighted interpolation (`training/sensor_map/README.md`,
//! "Reading `profile.json`").
//!
//! A grid is `{ start_k, step_k, values }`: `values[i]` belongs to `T_i = start_k + i * step_k`. A query
//! outside the table is clamped to its ends. With `T_i <= T <= T_(i+1)` the weight is taken in **mired**:
//!
//! ```text
//! a = (1/T - 1/T_i) / (1/T_(i+1) - 1/T_i)
//! ```
//!
//! and the value is either blended linearly, `(1 - a) * v_i + a * v_(i+1)` (log-chroma, CIE xy, the Duv
//! scale), or geometrically, `v_i^(1 - a) * v_(i+1)^a` per component (`w_A`, whose green component stays
//! exactly 1). This is the same arithmetic as `training/sensor_map/runtime_ref.py::Grid`.
//!
//! [`bracket_by_mired`] does the corresponding job for the class lists of the profile (sensor-map
//! matrices, fluorescent and LED `w_A`): linear in mired between the two bracketing entries, clamped
//! at the ends.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Errors from building a [`Grid`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GridError {
    /// Interpolation needs at least two table entries.
    #[error("a grid needs at least 2 values, found {0}")]
    TooShort(usize),
    /// `start_k` must be a positive finite temperature.
    #[error("start_k must be a positive finite number, found {0}")]
    BadStart(f64),
    /// `step_k` must be a positive finite temperature step.
    #[error("step_k must be a positive finite number, found {0}")]
    BadStep(f64),
    /// A table entry is NaN or infinite.
    #[error("value {index} is not finite")]
    NonFinite {
        /// Index of the offending entry.
        index: usize,
    },
}

/// A value stored in a [`Grid`]: a scalar or a small fixed-size vector, blended component-wise.
pub trait GridValue: Copy {
    /// Linear blend `(1 - a) * x + a * y`. Exactly `x` at `a == 0` and exactly `y` at `a == 1`.
    fn lerp(x: Self, y: Self, a: f64) -> Self;
    /// Geometric blend `x^(1 - a) * y^a` per component (components must be positive).
    fn geometric(x: Self, y: Self, a: f64) -> Self;
    /// Whether every component is finite.
    fn all_finite(&self) -> bool;
}

impl GridValue for f64 {
    fn lerp(x: Self, y: Self, a: f64) -> Self {
        (1.0 - a) * x + a * y
    }

    fn geometric(x: Self, y: Self, a: f64) -> Self {
        x.powf(1.0 - a) * y.powf(a)
    }

    fn all_finite(&self) -> bool {
        self.is_finite()
    }
}

impl<const N: usize> GridValue for [f64; N] {
    fn lerp(x: Self, y: Self, a: f64) -> Self {
        std::array::from_fn(|k| (1.0 - a) * x[k] + a * y[k])
    }

    fn geometric(x: Self, y: Self, a: f64) -> Self {
        std::array::from_fn(|k| x[k].powf(1.0 - a) * y[k].powf(a))
    }

    fn all_finite(&self) -> bool {
        self.iter().all(|v| v.is_finite())
    }
}

/// The wire form of a [`Grid`]: exactly the three JSON fields.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GridData<T> {
    start_k: f64,
    step_k: f64,
    values: Vec<T>,
}

#[derive(Serialize)]
struct GridDataRef<'a, T> {
    start_k: f64,
    step_k: f64,
    values: &'a [T],
}

/// A table sampled on a uniform temperature grid: `values[i]` belongs to `start_k + i * step_k`.
///
/// Built through [`Grid::new`] (or deserialisation, which calls it), so a grid always has at least two
/// finite entries and a positive start and step; the interpolation functions never fail.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid<T> {
    start_k: f64,
    step_k: f64,
    values: Vec<T>,
}

impl<T: GridValue> Grid<T> {
    /// Builds a grid.
    ///
    /// # Errors
    ///
    /// Returns a [`GridError`] for fewer than two values, a start or step that is not a positive finite
    /// number, or a non-finite value.
    pub fn new(start_k: f64, step_k: f64, values: Vec<T>) -> Result<Self, GridError> {
        if values.len() < 2 {
            return Err(GridError::TooShort(values.len()));
        }
        if !(start_k.is_finite() && start_k > 0.0) {
            return Err(GridError::BadStart(start_k));
        }
        if !(step_k.is_finite() && step_k > 0.0) {
            return Err(GridError::BadStep(step_k));
        }
        if let Some(index) = values.iter().position(|v| !v.all_finite()) {
            return Err(GridError::NonFinite { index });
        }
        Ok(Self {
            start_k,
            step_k,
            values,
        })
    }

    /// Temperature of the first entry in kelvin.
    #[must_use]
    pub fn start_k(&self) -> f64 {
        self.start_k
    }

    /// Spacing between entries in kelvin.
    #[must_use]
    pub fn step_k(&self) -> f64 {
        self.step_k
    }

    /// The table entries.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Number of entries (at least 2).
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always `false`: a grid has at least two entries. Present for the usual `len`/`is_empty` pairing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Temperature of the last entry in kelvin.
    #[must_use]
    pub fn end_k(&self) -> f64 {
        self.start_k + self.step_k * (self.values.len() - 1) as f64
    }

    /// Temperature of entry `index` in kelvin.
    #[must_use]
    pub fn temperature_k(&self, index: usize) -> f64 {
        self.start_k + index as f64 * self.step_k
    }

    /// The bracketing pair for a temperature: the lower entry index `i` (the pair is `i`, `i + 1`) and
    /// the mired weight `a` in `[0, 1]`. `t_k` is clamped to the table first (a NaN clamps to the start).
    #[must_use]
    pub fn locate(&self, t_k: f64) -> (usize, f64) {
        let t = t_k.max(self.start_k).min(self.end_k());
        let last_pair = self.values.len() - 2;
        let raw = ((t - self.start_k) / self.step_k).floor();
        // `raw` is finite and in [0, len - 1] after the clamp; the cast only drops the fraction.
        let i = (raw.max(0.0) as usize).min(last_pair);
        let t0 = self.temperature_k(i);
        let t1 = t0 + self.step_k;
        let a = (1.0 / t - 1.0 / t0) / (1.0 / t1 - 1.0 / t0);
        (i, a.clamp(0.0, 1.0))
    }

    /// Value at `t_k`, blended linearly in mired between the two bracketing entries.
    #[must_use]
    pub fn lerp_mired(&self, t_k: f64) -> T {
        let (i, a) = self.locate(t_k);
        T::lerp(self.values[i], self.values[i + 1], a)
    }

    /// Value at `t_k`, blended geometrically in mired (`v_i^(1-a) * v_(i+1)^a`).
    #[must_use]
    pub fn geometric_mired(&self, t_k: f64) -> T {
        let (i, a) = self.locate(t_k);
        T::geometric(self.values[i], self.values[i + 1], a)
    }
}

impl<'de, T: GridValue + Deserialize<'de>> Deserialize<'de> for Grid<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = GridData::<T>::deserialize(deserializer)?;
        Grid::new(raw.start_k, raw.step_k, raw.values).map_err(serde::de::Error::custom)
    }
}

impl<T: Serialize> Serialize for Grid<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        GridDataRef {
            start_k: self.start_k,
            step_k: self.step_k,
            values: &self.values,
        }
        .serialize(serializer)
    }
}

/// Brackets `mired` in a list of entries sorted by increasing mired: `(lo, hi, a)` with the value
/// `(1 - a) * entry[lo] + a * entry[hi]`.
///
/// Below the first entry (or for a NaN) the result is `(0, 0, 0)`, above the last `(n - 1, n - 1, 0)`
/// (clamped at the ends); inside, `lo` is the last entry with `mired(lo) <= mired` and `hi = lo + 1`.
/// Returns `None` for an empty list. Same rule as `runtime_ref.interp_class_list`.
pub fn bracket_by_mired<E>(
    entries: &[E],
    mired: f64,
    key: impl Fn(&E) -> f64,
) -> Option<(usize, usize, f64)> {
    let first = key(entries.first()?);
    let last_index = entries.len() - 1;
    let last = key(&entries[last_index]);
    if mired.is_nan() || mired <= first {
        return Some((0, 0, 0.0));
    }
    if mired >= last {
        return Some((last_index, last_index, 0.0));
    }
    // entries[upper - 1] <= mired < entries[upper]; both exist because first < mired < last.
    let upper = entries.partition_point(|e| key(e) <= mired);
    let lo = upper - 1;
    let (m_lo, m_hi) = (key(&entries[lo]), key(&entries[upper]));
    Some((lo, upper, (mired - m_lo) / (m_hi - m_lo)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Grid<[f64; 2]> {
        // Values equal 1e6 / T at each node, so mired interpolation is exact between nodes.
        let values = (0..5)
            .map(|i| {
                let t = 1000.0 + 100.0 * f64::from(i);
                [1e6 / t, 2.0 * 1e6 / t]
            })
            .collect();
        Grid::new(1000.0, 100.0, values).unwrap()
    }

    #[test]
    fn locate_is_exact_on_nodes_and_clamped_outside() {
        let g = ramp();
        assert_eq!(g.locate(1000.0), (0, 0.0));
        assert_eq!(g.locate(1100.0), (1, 0.0));
        // The last node is the upper end of the last pair.
        assert_eq!(g.locate(1400.0), (3, 1.0));
        assert_eq!(g.locate(10.0), (0, 0.0));
        assert_eq!(g.locate(9999.0), (3, 1.0));
        assert_eq!(g.locate(f64::NAN), (0, 0.0));
        assert_eq!(g.end_k(), 1400.0);
        assert_eq!(g.temperature_k(2), 1200.0);
    }

    #[test]
    fn mired_weight_matches_the_formula() {
        let g = ramp();
        let (i, a) = g.locate(1150.0);
        assert_eq!(i, 1);
        let expected = (1.0 / 1150.0 - 1.0 / 1100.0) / (1.0 / 1200.0 - 1.0 / 1100.0);
        assert!((a - expected).abs() < 1e-15);
        // A table linear in mired reproduces itself between nodes.
        let v = g.lerp_mired(1150.0);
        assert!((v[0] - 1e6 / 1150.0).abs() < 1e-9);
        assert!((v[1] - 2e6 / 1150.0).abs() < 1e-9);
    }

    #[test]
    fn geometric_blend_keeps_unity_exactly() {
        let g = Grid::new(
            1500.0,
            10.0,
            vec![[2.5, 1.0, 0.3], [2.4, 1.0, 0.31], [2.3, 1.0, 0.32]],
        )
        .unwrap();
        for t in [1500.0, 1503.3, 1509.9, 1510.0, 1517.0, 1520.0, 1e9] {
            assert_eq!(g.geometric_mired(t)[1], 1.0);
        }
        // At a node the geometric blend returns the node.
        assert_eq!(g.geometric_mired(1510.0), [2.4, 1.0, 0.31]);
        // Halfway in mired it is the geometric mean of the neighbours.
        let (i, a) = g.locate(1505.0);
        assert_eq!(i, 0);
        let w = g.geometric_mired(1505.0);
        assert!((w[0] - 2.5_f64.powf(1.0 - a) * 2.4_f64.powf(a)).abs() < 1e-15);
    }

    #[test]
    fn rejects_malformed_grids() {
        assert!(matches!(
            Grid::new(1500.0, 10.0, vec![1.0]),
            Err(GridError::TooShort(1))
        ));
        assert!(matches!(
            Grid::new(0.0, 10.0, vec![1.0, 2.0]),
            Err(GridError::BadStart(_))
        ));
        assert!(matches!(
            Grid::new(1500.0, 0.0, vec![1.0, 2.0]),
            Err(GridError::BadStep(_))
        ));
        assert!(matches!(
            Grid::new(1500.0, 10.0, vec![1.0, f64::NAN]),
            Err(GridError::NonFinite { index: 1 })
        ));
    }

    #[test]
    fn json_round_trip_and_unknown_field_rejection() {
        let text = r#"{"start_k": 1500, "step_k": 10, "values": [[0.5, -1.0], [0.4, -0.9]]}"#;
        let g: Grid<[f64; 2]> = serde_json::from_str(text).unwrap();
        assert_eq!(g.start_k(), 1500.0);
        assert_eq!(g.values().len(), 2);
        let back: Grid<[f64; 2]> =
            serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back, g);
        assert!(serde_json::from_str::<Grid<[f64; 2]>>(
            r#"{"start_k": 1500, "step_k": 10, "values": [[0.5, -1.0], [0.4, -0.9]], "extra": 1}"#
        )
        .is_err());
        assert!(
            serde_json::from_str::<Grid<[f64; 2]>>(
                r#"{"start_k": 1500, "step_k": 10, "values": [[0.5, -1.0]]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn bracket_by_mired_clamps_and_interpolates() {
        let ms = [100.0, 200.0, 400.0];
        let key = |m: &f64| *m;
        assert_eq!(bracket_by_mired(&[] as &[f64], 1.0, key), None);
        assert_eq!(bracket_by_mired(&ms, 50.0, key), Some((0, 0, 0.0)));
        assert_eq!(bracket_by_mired(&ms, 100.0, key), Some((0, 0, 0.0)));
        assert_eq!(bracket_by_mired(&ms, 450.0, key), Some((2, 2, 0.0)));
        assert_eq!(bracket_by_mired(&ms, 400.0, key), Some((2, 2, 0.0)));
        assert_eq!(bracket_by_mired(&ms, 150.0, key), Some((0, 1, 0.5)));
        assert_eq!(bracket_by_mired(&ms, 200.0, key), Some((1, 2, 0.0)));
        assert_eq!(bracket_by_mired(&ms, 300.0, key), Some((1, 2, 0.5)));
        assert_eq!(bracket_by_mired(&ms, f64::NAN, key), Some((0, 0, 0.0)));
        assert_eq!(bracket_by_mired(&[7.0], 9.0, key), Some((0, 0, 0.0)));
    }
}
