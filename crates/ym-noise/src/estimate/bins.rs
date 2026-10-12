//! Intensity bins: 24 bins over [0, 0.98], log-spaced below 0.05 (`docs/06-noise.md` step 3).

/// Number of intensity bins.
pub const NUM_BINS: usize = 24;

/// Upper edge of the last bin; windows brighter than this carry no usable noise statistics.
pub const TOP_EDGE: f64 = 0.98;

/// Edge between the log-spaced and the linearly spaced bins.
pub const KNEE: f64 = 0.05;

/// Upper edge of the first (catch-all, includes negative means) bin.
pub const FIRST_EDGE: f64 = 0.002;

/// Number of log-spaced bins between [`FIRST_EDGE`] and [`KNEE`].
const LOG_BINS: usize = 6;

/// The 25 edges of the 24 bins. Bin 0 is `(-inf, 0.002)`, bins 1 to 6 are log-spaced up to 0.05
/// and bins 7 to 23 are linear up to 0.98.
pub fn edges() -> [f64; NUM_BINS + 1] {
    let mut e = [0.0; NUM_BINS + 1];
    e[0] = f64::NEG_INFINITY;
    e[1] = FIRST_EDGE;
    let ratio = (KNEE / FIRST_EDGE).powf(1.0 / LOG_BINS as f64);
    for k in 1..=LOG_BINS {
        e[1 + k] = if k == LOG_BINS {
            KNEE
        } else {
            FIRST_EDGE * ratio.powi(k as i32)
        };
    }
    let linear = NUM_BINS - 1 - LOG_BINS;
    let step = (TOP_EDGE - KNEE) / linear as f64;
    for k in 1..=linear {
        e[1 + LOG_BINS + k] = if k == linear {
            TOP_EDGE
        } else {
            KNEE + step * k as f64
        };
    }
    e
}

/// Bin lookup for a mean intensity.
#[derive(Clone, Debug)]
pub struct BinLayout {
    edges: [f64; NUM_BINS + 1],
}

impl Default for BinLayout {
    fn default() -> Self {
        Self { edges: edges() }
    }
}

impl BinLayout {
    /// The bin holding `mean`, or `None` when `mean >= 0.98` (or is not finite).
    pub fn bin_of(&self, mean: f64) -> Option<usize> {
        if !mean.is_finite() || mean >= TOP_EDGE {
            return None;
        }
        // The edge list is short; a linear scan from the top beats a binary search here.
        (0..NUM_BINS).rev().find(|&k| mean >= self.edges[k])
    }

    /// Lower and upper edge of bin `k` in intensity, with the first bin starting at 0.
    pub fn range(&self, k: usize) -> (f64, f64) {
        (if k == 0 { 0.0 } else { self.edges[k] }, self.edges[k + 1])
    }

    /// Length of the union of the ranges of the bins in `bins`, as a fraction of `[0, 0.98]`.
    pub fn coverage(&self, bins: impl IntoIterator<Item = usize>) -> f64 {
        let mut seen = [false; NUM_BINS];
        for k in bins {
            if k < NUM_BINS {
                seen[k] = true;
            }
        }
        let total: f64 = (0..NUM_BINS)
            .filter(|&k| seen[k])
            .map(|k| {
                let (lo, hi) = self.range(k);
                hi - lo
            })
            .sum();
        total / TOP_EDGE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_increasing_and_span_the_range() {
        let e = edges();
        assert_eq!(e.len(), 25);
        for k in 1..NUM_BINS {
            assert!(e[k + 1] > e[k]);
        }
        assert_eq!(e[1], 0.002);
        assert!((e[7] - 0.05).abs() < 1e-12);
        assert!((e[NUM_BINS] - 0.98).abs() < 1e-12);
        // Log spacing below the knee: constant ratio.
        let r = e[3] / e[2];
        assert!((e[2] / e[1] - r).abs() < 1e-9 && (e[7] / e[6] - r).abs() < 1e-9);
        // Linear spacing above it.
        let w = e[9] - e[8];
        assert!((e[NUM_BINS] - e[NUM_BINS - 1] - w).abs() < 1e-12);
    }

    #[test]
    fn bin_lookup_covers_the_edges() {
        let l = BinLayout::default();
        assert_eq!(l.bin_of(-0.03), Some(0));
        assert_eq!(l.bin_of(0.0), Some(0));
        assert_eq!(l.bin_of(0.002), Some(1));
        assert_eq!(l.bin_of(0.0499), Some(6));
        assert_eq!(l.bin_of(0.05), Some(7));
        assert_eq!(l.bin_of(0.9799), Some(23));
        assert_eq!(l.bin_of(0.98), None);
        assert_eq!(l.bin_of(1.0), None);
        assert_eq!(l.bin_of(f64::NAN), None);
    }

    #[test]
    fn coverage_is_a_linear_fraction() {
        let l = BinLayout::default();
        assert!((l.coverage(0..NUM_BINS) - 1.0).abs() < 1e-12);
        assert_eq!(l.coverage([]), 0.0);
        // The log bins together span 0.05 of 0.98.
        assert!((l.coverage(0..7) - 0.05 / 0.98).abs() < 1e-12);
    }
}
