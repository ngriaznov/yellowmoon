//! The camera locus: camera RGB of the scene-light models against CCT, and the readout of CCT and
//! Duv from a camera RGB (`docs/02-colour-pipeline.md` section "Camera RGB <-> CCT and Duv";
//! `training/sensor_map/README.md`, "RGB -> (CCT, Duv)").
//!
//! The tables hold the camera RGB (green = 1) of blackbody light 1500 to 4000 K and CIE daylight 4000
//! to 25000 K every 10 K as log-chroma `(ln R/G, ln B/G)`, the matching CIE xy, and the CIE 1960 uv
//! distance per unit of log-chroma distance across the locus (`duv_scale`, every 100 K). The runtime
//! only interpolates them, so no CIE data is needed.
//!
//! * [`Locus::rgb_to_cct_duv`]: nearest point on each branch's chord polyline, the branch with the
//!   smaller distance wins (daylight on a tie), CCT from the mired position of the foot point, Duv =
//!   signed log-chroma distance times `duv_scale(CCT)`, **positive towards magenta** (the opposite of
//!   CIE's sign). The chord normal is the tangent rotated by -90 degrees, flipped when needed so it
//!   always points towards magenta (see `magenta_normal`: it only differs from `runtime_ref.py` on the
//!   backward chords of the D-series staircase above about 16000 K).
//! * [`Locus::cct_duv_to_rgb`]: the inverse for small Duv.
//!
//! The arithmetic follows `training/sensor_map/runtime_ref.py` step for step.

use serde::{Deserialize, Serialize};

use super::grid::Grid;
use super::{
    BLACKBODY_START_K, DAYLIGHT_START_K, DUV_SCALE_STEP_K, GRID_STEP_K, KELVIN_SPLIT_K,
    SpectralError, TABLE_END_K, check_grid_range,
};

/// The meaning of the two numbers in each locus row.
pub const LOCUS_AXES: [&str; 2] = ["ln(R/G)", "ln(B/G)"];
/// The interpolation rule recorded in `locus.interpolation`.
pub const LOCUS_INTERPOLATION: &str = "linear in mired";
/// The sign convention recorded in `locus.duv_sign`.
pub const DUV_SIGN: &str = "positive toward magenta";

/// CIE 1931 xy of the locus lights, same grids as the log-chroma tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocusXy {
    /// Blackbody, 1500 to 4000 K.
    pub blackbody: Grid<[f64; 2]>,
    /// CIE daylight, 4000 to 25000 K.
    pub daylight: Grid<[f64; 2]>,
}

/// CIE 1960 uv distance per unit of log-chroma distance across the locus, every 100 K.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuvScale {
    /// Blackbody, 1500 to 4000 K.
    pub blackbody: Grid<f64>,
    /// CIE daylight, 4000 to 25000 K.
    pub daylight: Grid<f64>,
}

/// A colour temperature with its distance from the locus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CctDuv {
    /// Correlated colour temperature in kelvin.
    pub cct_k: f64,
    /// Signed distance from the locus in CIE 1960 uv units, positive towards magenta.
    pub duv: f64,
}

/// The `locus` section of a camera profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Locus {
    /// Meaning of the two numbers in each row; must equal [`LOCUS_AXES`].
    pub axes: Vec<String>,
    /// Interpolation rule; must equal [`LOCUS_INTERPOLATION`].
    pub interpolation: String,
    /// Log-chroma `(ln R/G, ln B/G)` of blackbody light, 1500 to 4000 K, 10 K step.
    pub blackbody: Grid<[f64; 2]>,
    /// Log-chroma of CIE daylight, 4000 to 25000 K, 10 K step.
    pub daylight: Grid<[f64; 2]>,
    /// CIE xy of the same lights.
    pub xy: LocusXy,
    /// The Duv scale tables.
    pub duv_scale: DuvScale,
    /// Sign convention; must equal [`DUV_SIGN`].
    pub duv_sign: String,
}

/// Unit normal of a chord with direction `e` (increasing temperature) and length `norm`, pointing
/// towards magenta: the tangent rotated by -90 degrees, `(e_y, -e_x) / |e|`.
///
/// On a smooth locus, R/G falls and B/G rises with temperature, so that vector points towards
/// increasing `ln R/G` and `ln B/G` (less green, a magenta shift). The CIE D series as colour-science
/// builds it (M1 and M2 rounded to three decimals) is a staircase, and above about 16000 K some of its
/// chords run backwards; the rotated vector of such a chord points towards green. The magenta direction
/// is the same everywhere (perpendicular to the locus, `(1, 1)` in log-chroma space), so a normal
/// that would point away from it is flipped. For every chord with a forward direction this is
/// `runtime_ref.py`'s rule exactly.
fn magenta_normal(e: [f64; 2], norm: f64) -> [f64; 2] {
    let n = [e[1] / norm, -e[0] / norm];
    if n[0] + n[1] < 0.0 { [-n[0], -n[1]] } else { n }
}

/// The foot of the perpendicular from a point to one branch's polyline.
#[derive(Debug, Clone, Copy)]
struct Foot {
    /// Distance from the point to the polyline in log-chroma units.
    distance: f64,
    /// Mired position of the foot point.
    mired: f64,
    /// Signed distance, positive on the magenta side of the chord.
    signed: f64,
}

/// Nearest point of a branch's polyline to `p`: projects `p` on every chord, clamps to the chord and
/// keeps the chord with the smallest distance (the first on ties). Chords of zero length (the D series
/// staircase can repeat a row exactly) are skipped.
fn nearest_on_branch(grid: &Grid<[f64; 2]>, p: [f64; 2]) -> Option<Foot> {
    let v = grid.values();
    let mut best: Option<Foot> = None;
    for i in 0..v.len() - 1 {
        let (a, b) = (v[i], v[i + 1]);
        let e = [b[0] - a[0], b[1] - a[1]];
        let ee = e[0] * e[0] + e[1] * e[1];
        if ee <= 0.0 {
            continue;
        }
        let t = (((p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1]) / ee).clamp(0.0, 1.0);
        let q = [a[0] + t * e[0], a[1] + t * e[1]];
        let d = [p[0] - q[0], p[1] - q[1]];
        let distance = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if best.is_some_and(|b| distance >= b.distance) {
            continue;
        }
        let m0 = 1e6 / grid.temperature_k(i);
        let m1 = 1e6 / grid.temperature_k(i + 1);
        let n = magenta_normal(e, ee.sqrt());
        best = Some(Foot {
            distance,
            mired: m0 + t * (m1 - m0),
            signed: d[0] * n[0] + d[1] * n[1],
        });
    }
    best
}

/// Unit normal towards magenta of chord `i`; for a zero-length chord that of the nearest chord with a
/// direction. `None` when every chord is degenerate.
fn chord_normal(grid: &Grid<[f64; 2]>, i: usize) -> Option<[f64; 2]> {
    let v = grid.values();
    let chords = v.len() - 1;
    let normal_of = |j: usize| {
        let e = [v[j + 1][0] - v[j][0], v[j + 1][1] - v[j][1]];
        let norm = (e[0] * e[0] + e[1] * e[1]).sqrt();
        (norm > 0.0).then(|| magenta_normal(e, norm))
    };
    for offset in 0..chords {
        let forward = if i + offset < chords {
            normal_of(i + offset)
        } else {
            None
        };
        let backward = if offset > 0 && offset <= i {
            normal_of(i - offset)
        } else {
            None
        };
        if let Some(n) = forward.or(backward) {
            return Some(n);
        }
    }
    None
}

impl Locus {
    fn chroma_grid(&self, cct_k: f64) -> &Grid<[f64; 2]> {
        if cct_k < KELVIN_SPLIT_K {
            &self.blackbody
        } else {
            &self.daylight
        }
    }

    /// Camera log-chroma `(ln R/G, ln B/G)` of the scene-light model at `cct_k` (blackbody below 4000 K,
    /// CIE daylight from 4000 K), linear in mired, clamped to the table ends.
    #[must_use]
    pub fn chroma_at(&self, cct_k: f64) -> [f64; 2] {
        self.chroma_grid(cct_k).lerp_mired(cct_k)
    }

    /// Camera RGB (green = 1) of the scene-light model at `cct_k`.
    #[must_use]
    pub fn rgb_at(&self, cct_k: f64) -> [f64; 3] {
        let [lr, lb] = self.chroma_at(cct_k);
        [lr.exp(), 1.0, lb.exp()]
    }

    /// CIE 1931 xy of the scene-light model at `cct_k`, linear in mired.
    #[must_use]
    pub fn xy_at(&self, cct_k: f64) -> [f64; 2] {
        if cct_k < KELVIN_SPLIT_K {
            self.xy.blackbody.lerp_mired(cct_k)
        } else {
            self.xy.daylight.lerp_mired(cct_k)
        }
    }

    /// CIE 1960 `(u, v)` of the scene-light model at `cct_k`, from the stored xy.
    #[must_use]
    pub fn uv_at(&self, cct_k: f64) -> [f64; 2] {
        let [x, y] = self.xy_at(cct_k);
        let d = -2.0 * x + 12.0 * y + 3.0;
        [4.0 * x / d, 6.0 * y / d]
    }

    /// CIE 1960 uv distance per unit of log-chroma distance across the locus at `cct_k` (blackbody
    /// table below 4000 K, daylight from 4000 K), linear in mired.
    #[must_use]
    pub fn duv_scale_at(&self, cct_k: f64) -> f64 {
        if cct_k < KELVIN_SPLIT_K {
            self.duv_scale.blackbody.lerp_mired(cct_k)
        } else {
            self.duv_scale.daylight.lerp_mired(cct_k)
        }
    }

    /// Camera RGB of a light (any scale) to its CCT and Duv.
    ///
    /// 1. `p = (ln R/G, ln B/G)`.
    /// 2. For each branch (blackbody 1500 to 4000 K, daylight 4000 to 25000 K) find the nearest point
    ///    of the polyline of table entries; chords of zero length are skipped.
    /// 3. The branch with the smaller distance wins (daylight on a tie).
    /// 4. The foot point's position along its chord gives the mired coordinate, `CCT = 1e6 / mired`,
    ///    so the readout inverts the locus interpolation exactly on the chord.
    /// 5. `Duv = signed distance * duv_scale(CCT)`, positive towards magenta.
    ///
    /// # Errors
    ///
    /// [`SpectralError::InvalidRgb`] when a channel is not a positive finite number.
    pub fn rgb_to_cct_duv(&self, rgb: [f64; 3]) -> Result<CctDuv, SpectralError> {
        if !rgb.iter().all(|c| c.is_finite() && *c > 0.0) {
            return Err(SpectralError::InvalidRgb(rgb));
        }
        let p = [(rgb[0] / rgb[1]).ln(), (rgb[2] / rgb[1]).ln()];
        if !p.iter().all(|v| v.is_finite()) {
            // Ratios beyond the f64 range (1e-300 against 1e300): not a light.
            return Err(SpectralError::InvalidRgb(rgb));
        }
        let blackbody = nearest_on_branch(&self.blackbody, p);
        let daylight = nearest_on_branch(&self.daylight, p);
        let foot = match (blackbody, daylight) {
            (Some(b), Some(d)) => {
                if d.distance <= b.distance {
                    d
                } else {
                    b
                }
            }
            (Some(f), None) | (None, Some(f)) => f,
            (None, None) => return Err(SpectralError::DegenerateLocus),
        };
        let cct_k = 1e6 / foot.mired;
        Ok(CctDuv {
            cct_k,
            duv: foot.signed * self.duv_scale_at(cct_k),
        })
    }

    /// CCT and Duv (positive towards magenta) to camera RGB with green = 1: the locus point at `cct_k`
    /// moved by `duv / duv_scale` along the chord's normal. Inverts [`Locus::rgb_to_cct_duv`] for small
    /// Duv (the foot of the perpendicular has to fall inside the same chord).
    #[must_use]
    pub fn cct_duv_to_rgb(&self, cct_k: f64, duv: f64) -> [f64; 3] {
        let grid = self.chroma_grid(cct_k);
        let (i, a) = grid.locate(cct_k);
        let (p0, p1) = (grid.values()[i], grid.values()[i + 1]);
        let q = [(1.0 - a) * p0[0] + a * p1[0], (1.0 - a) * p0[1] + a * p1[1]];
        let n = chord_normal(grid, i).unwrap_or([0.0, 0.0]);
        let s = duv / self.duv_scale_at(cct_k);
        [(q[0] + s * n[0]).exp(), 1.0, (q[1] + s * n[1]).exp()]
    }

    /// Checks the structure the schema cannot express: constant strings, the grid ranges, positive
    /// scales and the direction of the locus.
    ///
    /// # Errors
    ///
    /// Returns the first problem found as a [`SpectralError::InvalidTable`].
    pub fn validate(&self) -> Result<(), SpectralError> {
        let bad = |table: &str, reason: String| SpectralError::InvalidTable {
            table: table.to_string(),
            reason,
        };
        if self.axes != LOCUS_AXES {
            return Err(bad(
                "locus.axes",
                format!("expected {LOCUS_AXES:?}, found {:?}", self.axes),
            ));
        }
        if self.interpolation != LOCUS_INTERPOLATION {
            return Err(bad(
                "locus.interpolation",
                format!(
                    "expected {LOCUS_INTERPOLATION:?}, found {:?}",
                    self.interpolation
                ),
            ));
        }
        if self.duv_sign != DUV_SIGN {
            return Err(bad(
                "locus.duv_sign",
                format!("expected {DUV_SIGN:?}, found {:?}", self.duv_sign),
            ));
        }
        check_grid_range(
            "locus.blackbody",
            &self.blackbody,
            BLACKBODY_START_K,
            KELVIN_SPLIT_K,
            GRID_STEP_K,
        )?;
        check_grid_range(
            "locus.daylight",
            &self.daylight,
            DAYLIGHT_START_K,
            TABLE_END_K,
            GRID_STEP_K,
        )?;
        check_grid_range(
            "locus.xy.blackbody",
            &self.xy.blackbody,
            BLACKBODY_START_K,
            KELVIN_SPLIT_K,
            GRID_STEP_K,
        )?;
        check_grid_range(
            "locus.xy.daylight",
            &self.xy.daylight,
            DAYLIGHT_START_K,
            TABLE_END_K,
            GRID_STEP_K,
        )?;
        check_grid_range(
            "locus.duv_scale.blackbody",
            &self.duv_scale.blackbody,
            BLACKBODY_START_K,
            KELVIN_SPLIT_K,
            DUV_SCALE_STEP_K,
        )?;
        check_grid_range(
            "locus.duv_scale.daylight",
            &self.duv_scale.daylight,
            DAYLIGHT_START_K,
            TABLE_END_K,
            DUV_SCALE_STEP_K,
        )?;
        for (name, grid) in [
            ("locus.duv_scale.blackbody", &self.duv_scale.blackbody),
            ("locus.duv_scale.daylight", &self.duv_scale.daylight),
        ] {
            if grid.values().iter().any(|&s| s <= 0.0) {
                return Err(bad(name, "the Duv scale must be positive".to_string()));
            }
        }
        // Log-chroma moves with temperature (R/G falls, B/G rises) on both branches. CIE daylight as
        // colour-science builds it (M1 and M2 rounded to three decimals) is a staircase around a smooth
        // curve, so single steps may go the wrong way by up to 2e-4 there.
        for (name, grid, slack) in [
            ("locus.blackbody", &self.blackbody, 1e-8),
            ("locus.daylight", &self.daylight, 2e-4),
        ] {
            let v = grid.values();
            let monotone = v
                .windows(2)
                .all(|w| w[1][0] - w[0][0] <= slack && w[1][1] - w[0][1] >= -slack);
            let (first, last) = (v[0], v[v.len() - 1]);
            if !monotone || !(last[0] < first[0] && last[1] > first[1]) {
                return Err(bad(
                    name,
                    format!("ln(R/G) must fall and ln(B/G) rise with temperature (slack {slack})"),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth synthetic locus: ln R/G = 1 - mired/250, ln B/G = mired^-1 stuff; exactly linear in
    /// mired so the readout is exact for points near the locus.
    fn synthetic() -> Locus {
        let bb: Vec<[f64; 2]> = (0..251)
            .map(|i| {
                let m = 1e6 / (1500.0 + 10.0 * f64::from(i));
                [0.002 * m - 0.5, -0.003 * m + 0.2]
            })
            .collect();
        let dy: Vec<[f64; 2]> = (0..2101)
            .map(|i| {
                let m = 1e6 / (4000.0 + 10.0 * f64::from(i));
                [0.002 * m - 0.5, -0.003 * m + 0.2]
            })
            .collect();
        let xy_bb = vec![[0.5, 0.4]; 251];
        let xy_dy = vec![[0.3, 0.3]; 2101];
        Locus {
            axes: LOCUS_AXES.iter().map(ToString::to_string).collect(),
            interpolation: LOCUS_INTERPOLATION.to_string(),
            blackbody: Grid::new(1500.0, 10.0, bb).unwrap(),
            daylight: Grid::new(4000.0, 10.0, dy).unwrap(),
            xy: LocusXy {
                blackbody: Grid::new(1500.0, 10.0, xy_bb).unwrap(),
                daylight: Grid::new(4000.0, 10.0, xy_dy).unwrap(),
            },
            duv_scale: DuvScale {
                blackbody: Grid::new(1500.0, 100.0, vec![0.05; 26]).unwrap(),
                daylight: Grid::new(4000.0, 100.0, vec![0.05; 211]).unwrap(),
            },
            duv_sign: DUV_SIGN.to_string(),
        }
    }

    #[test]
    fn synthetic_locus_validates_and_round_trips() {
        let locus = synthetic();
        locus.validate().unwrap();
        for t in [
            1500.0, 1800.0, 2856.0, 3999.0, 4000.0, 5500.0, 6504.0, 12000.0, 25000.0,
        ] {
            let rgb = locus.rgb_at(t);
            let back = locus.rgb_to_cct_duv(rgb).unwrap();
            assert!((back.cct_k - t).abs() < 1e-6 * t, "t={t}: {back:?}");
            assert!(back.duv.abs() < 1e-9, "t={t}: {back:?}");
        }
    }

    #[test]
    fn duv_is_positive_towards_magenta() {
        let locus = synthetic();
        // Less green, equally much red and blue: both log-chroma coordinates rise, a magenta shift.
        let mut rgb = locus.rgb_at(5000.0);
        rgb[1] = 0.97;
        let magenta = locus.rgb_to_cct_duv(rgb).unwrap();
        assert!(magenta.duv > 0.0, "{magenta:?}");
        // More green is the other way.
        rgb[1] = 1.03;
        let green = locus.rgb_to_cct_duv(rgb).unwrap();
        assert!(green.duv < 0.0, "{green:?}");
        // (CCT, Duv) round trip for small Duv.
        for duv in [-0.004, -0.001, 0.0, 0.002, 0.004] {
            for t in [2000.0, 3456.0, 4500.0, 9000.0] {
                let back = locus.rgb_to_cct_duv(locus.cct_duv_to_rgb(t, duv)).unwrap();
                assert!((back.cct_k - t).abs() < 1e-3, "{t} {duv}: {back:?}");
                assert!((back.duv - duv).abs() < 1e-9, "{t} {duv}: {back:?}");
            }
        }
    }

    #[test]
    fn daylight_wins_ties_and_zero_length_chords_are_skipped() {
        let mut locus = synthetic();
        // Repeat a daylight row exactly (a zero-length chord) and check nothing breaks.
        let mut v = locus.daylight.values().to_vec();
        v[100] = v[101];
        locus.daylight = Grid::new(4000.0, 10.0, v).unwrap();
        let rgb = locus.rgb_at(5000.0);
        assert!(locus.rgb_to_cct_duv(rgb).unwrap().cct_k.is_finite());
        // At 4000 K both branches contain the point: the daylight branch is chosen. Its CCT is
        // 1e6 / mired of the daylight start, i.e. 4000 K, not the blackbody end.
        let at_split = locus.rgb_to_cct_duv(locus.rgb_at(4000.0)).unwrap();
        assert!((at_split.cct_k - 4000.0).abs() < 1e-9);
        assert!(at_split.duv.abs() < 1e-12);
        // The inverse also copes with a degenerate chord (it borrows the neighbour's normal).
        let rgb = locus.cct_duv_to_rgb(4000.0 + 10.0 * 100.5, 0.002);
        assert!(rgb.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn backward_chords_still_point_towards_magenta() {
        // A staircase: every second chord runs backwards along the locus (a retrace).
        let mut locus = synthetic();
        let mut v = locus.daylight.values().to_vec();
        for i in (1..v.len() - 1).step_by(2) {
            // Swap two neighbouring rows: the chords around them run forward, backward, forward.
            v.swap(i, i + 1);
        }
        locus.daylight = Grid::new(4000.0, 10.0, v).unwrap();
        for t in [4500.0, 6500.0, 12000.0, 20000.0] {
            let mut rgb = locus.rgb_at(t);
            rgb[1] = 0.9;
            let magenta = locus.rgb_to_cct_duv(rgb).unwrap();
            assert!(magenta.duv > 0.0, "{t} K: {magenta:?}");
            rgb[1] = 1.1;
            let green = locus.rgb_to_cct_duv(rgb).unwrap();
            assert!(green.duv < 0.0, "{t} K: {green:?}");
            // The inverse agrees about which side is magenta.
            let rgb = locus.cct_duv_to_rgb(t, 0.003);
            assert!(
                rgb[0] > locus.rgb_at(t)[0] * 0.99 && rgb[2] > locus.rgb_at(t)[2] * 0.99,
                "{t} K: {rgb:?}"
            );
        }
    }

    #[test]
    fn rejects_invalid_rgb() {
        let locus = synthetic();
        for rgb in [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [f64::NAN, 1.0, 1.0],
            [1.0, f64::INFINITY, 1.0],
        ] {
            assert!(matches!(
                locus.rgb_to_cct_duv(rgb),
                Err(SpectralError::InvalidRgb(_))
            ));
        }
        // The scale of the RGB does not matter.
        let a = locus.rgb_to_cct_duv([0.2, 0.4, 0.3]).unwrap();
        let b = locus.rgb_to_cct_duv([2.0, 4.0, 3.0]).unwrap();
        assert!((a.cct_k - b.cct_k).abs() < 1e-9 * a.cct_k && (a.duv - b.duv).abs() < 1e-15);
    }

    #[test]
    fn table_choice_follows_the_4000_k_rule() {
        let locus = synthetic();
        assert_eq!(
            locus.chroma_at(3999.999),
            locus.blackbody.lerp_mired(3999.999)
        );
        assert_eq!(locus.chroma_at(4000.0), locus.daylight.lerp_mired(4000.0));
        // Clamped at the ends of the daylight table.
        assert_eq!(locus.chroma_at(30000.0), locus.daylight.values()[2100]);
        assert_eq!(locus.chroma_at(1000.0), locus.blackbody.values()[0]);
    }

    #[test]
    fn validate_catches_structural_errors() {
        let mut locus = synthetic();
        locus.duv_sign = "positive toward green".to_string();
        assert!(locus.validate().is_err());
        let mut locus = synthetic();
        locus.axes = vec!["ln(R/G)".to_string()];
        assert!(locus.validate().is_err());
        let mut locus = synthetic();
        locus.duv_scale.daylight = Grid::new(4000.0, 100.0, vec![0.05; 210]).unwrap();
        assert!(locus.validate().is_err());
        let mut locus = synthetic();
        let flipped: Vec<[f64; 2]> = locus.blackbody.values().iter().rev().copied().collect();
        locus.blackbody = Grid::new(1500.0, 10.0, flipped).unwrap();
        assert!(locus.validate().is_err());
    }
}
