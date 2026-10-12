//! ARRI's 17-CCT matrix table: loading, bracketing pair and interpolation in mired, native to AWG3 (`docs/02-colour-pipeline.md` sections ARRI CCT matrices and C5-C6).
//!
//! `data/arri/Alexa_aces_matrix.txt` (ARRI via aces-dev v1.3.1, AMPAS licence) holds camera-native
//! (white-balanced) to ACES AP0 matrices at 17 colour temperatures. This module
//!
//! * parses that file ([`ArriTable::parse`]), and embeds it at compile time
//!   ([`ArriTable::builtin`]): the table is 3 KB, the runtime needs neither a data-folder lookup
//!   nor network, and the AMPAS notice travels with the binary (see `NOTICE`);
//! * derives native to AWG3 as `inv(AWG3 to AP0) * M_AP0(CCT)` with the exact f64 AWG3 to AP0
//!   matrix of [`crate::gamut`];
//! * interpolates linearly in mired (`1e6 / CCT`) between the **bracketing** pair of table entries,
//!   clamping the CCT to [`CCT_MIN_K`] to [`CCT_MAX_K`]. At the 17 table CCTs the result equals the
//!   table value exactly.
//!
//! ARRI's `v3_IDT_maker.py` instead calls `searchsorted` and pairs index `i` with `i + 1`, which
//! extrapolates from the *next* pair for a CCT between two entries. The two schemes agree exactly
//! at table CCTs and differ between them; `tests/arri.rs` implements the ARRI pairing and shows both.

use std::sync::LazyLock;

use crate::consts::{Mat3, linalg};
use crate::gamut::matrices;

/// Lowest CCT of the ARRI table in kelvin; lower inputs are clamped to it.
pub const CCT_MIN_K: f64 = 2000.0;

/// Highest CCT of the ARRI table in kelvin; higher inputs are clamped to it.
pub const CCT_MAX_K: f64 = 11000.0;

/// The embedded copy of `data/arri/Alexa_aces_matrix.txt`.
const EMBEDDED_TABLE: &str = include_str!("../../../data/arri/Alexa_aces_matrix.txt");

/// Errors from parsing an ARRI matrix table.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ArriError {
    /// A data line does not hold a CCT and nine matrix coefficients.
    #[error("line {line}: expected a CCT and 9 matrix coefficients, found {found} numbers")]
    WrongColumnCount {
        /// 1-based line number.
        line: usize,
        /// How many whitespace-separated fields the line had.
        found: usize,
    },
    /// A field is not a finite number.
    #[error("line {line}: {field:?} is not a finite number")]
    BadNumber {
        /// 1-based line number.
        line: usize,
        /// The offending field text.
        field: String,
    },
    /// CCTs must be positive and strictly increasing.
    #[error("line {line}: CCT {cct} is not greater than the previous entry")]
    NotIncreasing {
        /// 1-based line number.
        line: usize,
        /// The offending CCT.
        cct: f64,
    },
    /// Interpolation needs at least two entries.
    #[error("the table has {0} entries, at least 2 are needed")]
    TooFewEntries(usize),
    /// AWG3 to AP0 could not be inverted (cannot happen for the built-in primaries).
    #[error("AWG3 to AP0 matrix is singular")]
    SingularGamut,
}

/// ARRI's native to AP0 matrices at their table CCTs, plus the derived native to AWG3 matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct ArriTable {
    ccts_k: Vec<f64>,
    native_to_ap0: Vec<Mat3>,
    native_to_awg3: Vec<Mat3>,
}

static BUILTIN: LazyLock<ArriTable> = LazyLock::new(|| {
    // The embedded text is checked by the unit tests; a parse failure here is a build defect.
    ArriTable::parse(EMBEDDED_TABLE).expect("embedded ARRI table parses (checked by tests)")
});

impl ArriTable {
    /// The table embedded from `data/arri/Alexa_aces_matrix.txt`.
    #[must_use]
    pub fn builtin() -> &'static ArriTable {
        &BUILTIN
    }

    /// Parses the text format of `Alexa_aces_matrix.txt`: `##` comment lines, a quoted header line,
    /// then rows of `CCT RR RG RB GR GG GB BR BG BB` (row-major native to AP0).
    ///
    /// # Errors
    ///
    /// Returns an [`ArriError`] for a row with the wrong number of fields, a non-finite number,
    /// non-increasing CCTs, or fewer than two rows.
    pub fn parse(text: &str) -> Result<Self, ArriError> {
        let mut ccts_k = Vec::new();
        let mut native_to_ap0: Vec<Mat3> = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let line = index + 1;
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('"') {
                continue;
            }
            let fields: Vec<&str> = trimmed.split_ascii_whitespace().collect();
            if fields.len() != 10 {
                return Err(ArriError::WrongColumnCount {
                    line,
                    found: fields.len(),
                });
            }
            let mut nums = [0.0_f64; 10];
            for (slot, field) in nums.iter_mut().zip(&fields) {
                *slot = field
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| ArriError::BadNumber {
                        line,
                        field: (*field).to_string(),
                    })?;
            }
            let cct = nums[0];
            if cct <= 0.0 || ccts_k.last().is_some_and(|&prev| cct <= prev) {
                return Err(ArriError::NotIncreasing { line, cct });
            }
            ccts_k.push(cct);
            native_to_ap0.push([
                [nums[1], nums[2], nums[3]],
                [nums[4], nums[5], nums[6]],
                [nums[7], nums[8], nums[9]],
            ]);
        }
        if ccts_k.len() < 2 {
            return Err(ArriError::TooFewEntries(ccts_k.len()));
        }
        let ap0_to_awg3 = linalg::inv(&matrices().awg3_to_ap0).ok_or(ArriError::SingularGamut)?;
        let native_to_awg3 = native_to_ap0
            .iter()
            .map(|m| linalg::mul(&ap0_to_awg3, m))
            .collect();
        Ok(Self {
            ccts_k,
            native_to_ap0,
            native_to_awg3,
        })
    }

    /// The table CCTs in kelvin, ascending.
    #[must_use]
    pub fn ccts_k(&self) -> &[f64] {
        &self.ccts_k
    }

    /// Number of table entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ccts_k.len()
    }

    /// Whether the table has no entries (never true for a parsed table).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ccts_k.is_empty()
    }

    /// The table's native to AP0 matrix at entry `index`.
    #[must_use]
    pub fn native_to_ap0_entry(&self, index: usize) -> Option<&Mat3> {
        self.native_to_ap0.get(index)
    }

    /// The derived native to AWG3 matrix at entry `index`.
    #[must_use]
    pub fn native_to_awg3_entry(&self, index: usize) -> Option<&Mat3> {
        self.native_to_awg3.get(index)
    }

    /// The bracketing pair for a CCT: the lower entry index `i` (so the pair is `i`, `i + 1`) and
    /// the interpolation weight `w` in `[0, 1]`, linear in mired. `w == 0` exactly at entry `i`,
    /// `w == 1` exactly at entry `i + 1`. The CCT is clamped to the table range first (a NaN clamps
    /// to the lowest entry).
    #[must_use]
    pub fn bracket(&self, cct_k: f64) -> (usize, f64) {
        let first = self.ccts_k[0];
        let last = self.ccts_k[self.ccts_k.len() - 1];
        let cct = cct_k.max(first).min(last);
        // Largest i with ccts[i] <= cct, restricted to 0 ..= len - 2 so that (i, i + 1) is a pair.
        let upper = self.ccts_k.partition_point(|&c| c <= cct);
        let i = upper.saturating_sub(1).min(self.ccts_k.len() - 2);
        let mired = 1.0e6 / cct;
        let m0 = 1.0e6 / self.ccts_k[i];
        let m1 = 1.0e6 / self.ccts_k[i + 1];
        let w = if cct <= self.ccts_k[i] {
            0.0
        } else if cct >= self.ccts_k[i + 1] {
            1.0
        } else {
            ((mired - m0) / (m1 - m0)).clamp(0.0, 1.0)
        };
        (i, w)
    }

    /// Native to AP0 matrix for a CCT, interpolated linearly in mired between the bracketing pair.
    #[must_use]
    pub fn native_to_ap0(&self, cct_k: f64) -> Mat3 {
        let (i, w) = self.bracket(cct_k);
        linalg::lerp(&self.native_to_ap0[i], &self.native_to_ap0[i + 1], w)
    }

    /// Native to AWG3 matrix for a CCT (`inv(AWG3 to AP0) * M_AP0(CCT)`), interpolated linearly in
    /// mired between the bracketing pair. This is `M_arri(CCT_user)` of stage C5. Rows sum to 1.
    #[must_use]
    pub fn native_to_awg3(&self, cct_k: f64) -> Mat3 {
        let (i, w) = self.bracket(cct_k);
        linalg::lerp(&self.native_to_awg3[i], &self.native_to_awg3[i + 1], w)
    }

    /// Native to AWG4 matrix for a CCT: [`ArriTable::native_to_awg3`] followed by the exact AWG3 to AWG4 matrix.
    #[must_use]
    pub fn native_to_awg4(&self, cct_k: f64) -> Mat3 {
        linalg::mul(&matrices().awg3_to_awg4, &self.native_to_awg3(cct_k))
    }

    /// Largest deviation from 1 of any row sum of the table's native to AP0 matrices.
    #[must_use]
    pub fn max_row_sum_error_ap0(&self) -> f64 {
        Self::max_row_sum_error(&self.native_to_ap0)
    }

    /// Largest deviation from 1 of any row sum of the derived native to AWG3 matrices.
    #[must_use]
    pub fn max_row_sum_error_awg3(&self) -> f64 {
        Self::max_row_sum_error(&self.native_to_awg3)
    }

    fn max_row_sum_error(ms: &[Mat3]) -> f64 {
        ms.iter()
            .flat_map(linalg::row_sums)
            .fold(0.0_f64, |worst, s| worst.max((s - 1.0).abs()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_table_has_seventeen_entries() {
        let t = ArriTable::builtin();
        assert_eq!(t.len(), 17);
        assert_eq!(t.ccts_k()[0], 2000.0);
        assert_eq!(t.ccts_k()[16], 11000.0);
    }

    #[test]
    fn rejects_malformed_tables() {
        assert!(matches!(
            ArriTable::parse(""),
            Err(ArriError::TooFewEntries(0))
        ));
        assert!(matches!(
            ArriTable::parse("2000 1 2 3"),
            Err(ArriError::WrongColumnCount { line: 1, found: 4 })
        ));
        assert!(matches!(
            ArriTable::parse("2000 1 0 0 0 1 0 0 0 nan"),
            Err(ArriError::BadNumber { line: 1, .. })
        ));
        let dup = "2000 1 0 0 0 1 0 0 0 1\n2000 1 0 0 0 1 0 0 0 1";
        assert!(matches!(
            ArriTable::parse(dup),
            Err(ArriError::NotIncreasing { line: 2, .. })
        ));
    }

    #[test]
    fn clamps_outside_the_table() {
        let t = ArriTable::builtin();
        assert_eq!(t.native_to_awg3(500.0), t.native_to_awg3(2000.0));
        assert_eq!(t.native_to_awg3(50_000.0), t.native_to_awg3(11000.0));
        assert_eq!(t.native_to_awg3(f64::NAN), t.native_to_awg3(2000.0));
    }
}
