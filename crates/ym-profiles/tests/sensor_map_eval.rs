//! M1-04 / M1-05: the Rust evaluation of the sensor map (a port of `training/sensor_map/evaluate.py` and
//! `gate.py`), run on the shipped profiles.
//!
//! * `sony_profiles.csv` "own" and "A7R IV" numbers (`docs/03`, `docs/13`): per-light 3x3 maps fitted
//!   with the **research objective** (`fit_map_weighted_rgb`, weights `Y^(-2/3)`) at each of the 9 test
//!   lights on the 190 training patches, scored on 138 colours (CES 99, ColorChecker 24, NIST VS 15) with
//!   `ym_color::deltae` after ARRI's matrix (`ym_color::arri`). Must match the published CSV within 0.02.
//! * The sensor-map gate (`docs/02` section C2): 99 CES under 9 held-out lights, the shipped tables
//!   blended by `ym_color::sensor_map` with the true light's class and mired; and, with the Lab-optimal fit
//!   done in Rust, the gate refit with LED-B3, LED-RGB1 and LED-V2 held out of the LED class.
//!
//! The CMFs, light SPDs and reflectances are CIE (CC BY-SA 4.0) and rawtoaces data and are not committed.
//! The tests read the fixtures `python3 -m training.sensor_map.export_fixtures` writes to
//! `testdata/spectral/` (git-ignored) and skip with a message when they are absent. CI runs the exporter
//! first (after `tools/fetch/spectral.sh`) and sets `YM_REQUIRE_FIXTURES=1` so a missing folder fails.
//!
//! Run with `cargo test -p ym-profiles --test sensor_map_eval -- --nocapture` to see the differences.

mod common;

use std::collections::BTreeMap;

use common::{data_root, repo_root, store};
use serde_json::Value;
use ym_color::arri::ArriTable;
use ym_color::consts::{Mat3, Vec3, cat02_adaptation, linalg};
use ym_color::deltae::{Summary, aces_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::gamut::matrices;
use ym_color::sensor_map::{MapEntry, SensorMapTables};
use ym_color::spectral::fixtures::{Fixtures, Light};
use ym_color::spectral::{ClassProbabilities, LightClass, SpectralCurves};
use ym_profiles::profile::Profile;

// ---------------------------------------------------------------------------------------------
// The evaluation (Rust port of evaluate.py / fit.py / gate.py)
// ---------------------------------------------------------------------------------------------

struct Eval {
    fx: Fixtures,
    alexa: SpectralCurves,
    arri: &'static ArriTable,
    aces_white: Vec3,
    /// CES 99 + ColorChecker 24 + NIST VS 15, the research evaluation set.
    r_test: Vec<Vec<f64>>,
}

fn peak_normalized(curves: SpectralCurves) -> SpectralCurves {
    curves
        .peak_normalized()
        .expect("curves have positive channels")
}

impl Eval {
    fn new(fx: Fixtures) -> Self {
        let alexa = SpectralCurves::from_path(
            data_root().join("alexa/alexa_alev3_estimate_380_780_5.json"),
        )
        .expect("ALEXA estimate loads");
        let mut r_test = fx.ces.spectra.clone();
        r_test.extend(fx.colorchecker.spectra.iter().cloned());
        r_test.extend(fx.nist_vs.spectra.iter().cloned());
        assert_eq!(
            (
                fx.ces.spectra.len(),
                fx.colorchecker.spectra.len(),
                fx.nist_vs.spectra.len()
            ),
            (99, 24, 15)
        );
        assert_eq!(fx.training.spectra.len(), 190);
        Self {
            alexa: peak_normalized(alexa),
            arri: ArriTable::builtin(),
            aces_white: aces_white_xyz(),
            r_test,
            fx,
        }
    }

    /// A Sony body's curves as the research scripts use them: noise below zero clipped, peak 1.
    fn sony(profile: &Profile) -> SpectralCurves {
        let curves = profile.load_curves(&data_root()).expect("curves load");
        peak_normalized(curves.clipped_at_zero())
    }

    /// White-balanced camera RGB of reflectances: `raw = (E o R)^T S`, `white = E^T S`, `raw / white`.
    fn camera_rgb_wb(curves: &SpectralCurves, e: &[f64], refl: &[Vec<f64>]) -> Vec<Vec3> {
        let white = curves.response(e).unwrap();
        refl.iter()
            .map(|r| {
                let product: Vec<f64> = e.iter().zip(r).map(|(a, b)| a * b).collect();
                let raw = curves.response(&product).unwrap();
                [raw[0] / white[0], raw[1] / white[1], raw[2] / white[2]]
            })
            .collect()
    }

    /// Luminance of the scene colorimetry CAT02-adapted to the ACES white (`ssf_core.target_XYZ`, Y row).
    fn target_y(&self, e: &[f64], refl: &[Vec<f64>]) -> Vec<f64> {
        let cmf = &self.fx.cmf;
        let norm: f64 = cmf.iter().zip(e).map(|(c, e)| c[1] * e).sum();
        let mut white = [0.0; 3];
        for (c, e) in cmf.iter().zip(e) {
            for k in 0..3 {
                white[k] += c[k] * e;
            }
        }
        let white = white.map(|v| v / norm);
        let adapt = cat02_adaptation(white, self.aces_white);
        refl.iter()
            .map(|r| {
                let mut xyz = [0.0; 3];
                for ((c, e), r) in cmf.iter().zip(e).zip(r) {
                    for k in 0..3 {
                        xyz[k] += e * r * c[k];
                    }
                }
                let xyz = xyz.map(|v| v / norm);
                linalg::mul_vec(&adapt, xyz)[1]
            })
            .collect()
    }

    /// The research objective: white-preserving 3x3 by weighted linear least squares in RGB, weights
    /// `clip(Y, 1e-3)^(-2/3)` (`fit.fit_map_weighted_rgb`, `sony_profiles.fit_map`).
    fn fit_map_weighted_rgb(
        &self,
        s_src: &SpectralCurves,
        s_dst: &SpectralCurves,
        e: &[f64],
    ) -> Mat3 {
        let train = &self.fx.training.spectra;
        let src = Self::camera_rgb_wb(s_src, e, train);
        let dst = Self::camera_rgb_wb(s_dst, e, train);
        let w: Vec<f64> = self
            .target_y(e, train)
            .iter()
            .map(|y| y.max(1e-3).powf(-2.0 / 3.0))
            .collect();
        // Normal equations of  min || w (a0 m0 + a1 m1 - b) ||  with a0 = src0 - src2, a1 = src1 - src2.
        let (mut s00, mut s01, mut s11) = (0.0, 0.0, 0.0);
        for (s, w) in src.iter().zip(&w) {
            let (a0, a1) = ((s[0] - s[2]) * w, (s[1] - s[2]) * w);
            s00 += a0 * a0;
            s01 += a0 * a1;
            s11 += a1 * a1;
        }
        let det = s00 * s11 - s01 * s01;
        let mut m = [[0.0; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            let (mut t0, mut t1) = (0.0, 0.0);
            for ((s, d), w) in src.iter().zip(&dst).zip(&w) {
                let (a0, a1) = ((s[0] - s[2]) * w, (s[1] - s[2]) * w);
                let b = (d[r] - s[2]) * w;
                t0 += a0 * b;
                t1 += a1 * b;
            }
            let (m0, m1) = ((s11 * t0 - s01 * t1) / det, (s00 * t1 - s01 * t0) / det);
            *row = [m0, m1, 1.0 - m0 - m1];
        }
        m
    }

    /// CIE Lab of an AP0 triple with the ACES white (`ssf_core.ap0_to_Lab`).
    fn ap0_to_lab(&self, ap0: Vec3) -> Vec3 {
        xyz_to_lab(
            linalg::mul_vec(&matrices().ap0_to_xyz, ap0),
            self.aces_white,
        )
    }

    /// ΔE00 per reflectance between the ALEXA rendering and the Sony rendering through `m`, both taken
    /// through ARRI's matrix at `cct_k` (`evaluate.render_error_with_matrix`).
    fn render_error(
        &self,
        s_true: &SpectralCurves,
        m: &Mat3,
        e: &[f64],
        cct_k: f64,
        refl: &[Vec<f64>],
    ) -> Vec<f64> {
        let m_arri = self.arri.native_to_ap0(cct_k);
        let alexa = Self::camera_rgb_wb(&self.alexa, e, refl);
        let sony = Self::camera_rgb_wb(s_true, e, refl);
        alexa
            .iter()
            .zip(&sony)
            .map(|(a, s)| {
                let target = self.ap0_to_lab(linalg::mul_vec(&m_arri, *a));
                let mapped = self.ap0_to_lab(linalg::mul_vec(&m_arri, linalg::mul_vec(m, *s)));
                delta_e_2000(mapped, target)
            })
            .collect()
    }

    /// `evaluate.map_error`: per research light, X's RGB through a map fitted on `s_fit` at that light.
    fn map_error(
        &self,
        s_true: &SpectralCurves,
        s_fit: &SpectralCurves,
    ) -> BTreeMap<String, Vec<f64>> {
        self.fx
            .lights_with_prefix("research:")
            .map(|l| {
                let m = self.fit_map_weighted_rgb(s_fit, &self.alexa, &l.spd);
                let errors = self.render_error(s_true, &m, &l.spd, l.cct_k, &self.r_test);
                (l.name.trim_start_matches("research:").to_string(), errors)
            })
            .collect()
    }
}

/// `evaluate.summary`: pooled median and p95, and the light with the worst median.
fn summary(res: &BTreeMap<String, Vec<f64>>) -> (f64, f64, String, f64) {
    let all: Vec<f64> = res.values().flatten().copied().collect();
    let s = Summary::of(&all).unwrap();
    let (worst, worst_median) = res
        .iter()
        .map(|(k, v)| (k.clone(), Summary::of(v).unwrap().median))
        .fold((String::new(), f64::NEG_INFINITY), |best, cur| {
            if cur.1 > best.1 { cur } else { best }
        });
    (s.median, s.p95, worst, worst_median)
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}

fn csv_rows() -> BTreeMap<String, BTreeMap<String, String>> {
    let text =
        std::fs::read_to_string(repo_root().join("research/alexa-ssf/out/sony_profiles.csv"))
            .expect("research CSV");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    lines
        .map(|l| {
            let cells: Vec<&str> = l.split(',').collect();
            (
                cells[0].to_string(),
                header
                    .iter()
                    .zip(&cells)
                    .map(|(h, c)| ((*h).to_string(), (*c).to_string()))
                    .collect(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// sony_profiles.csv
// ---------------------------------------------------------------------------------------------

#[test]
fn own_and_stand_in_numbers_reproduce_sony_profiles_csv() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let expected = fx.expected["sony_profiles"].clone();
    assert_eq!(num(&expected["tolerance"]), 0.02);
    let ev = Eval::new(fx);
    let s = store();
    let csv = csv_rows();
    let a7r4 = Eval::sony(s.default_profile());

    let mut max_csv: BTreeMap<&str, f64> = BTreeMap::new();
    let mut max_py = 0.0_f64;
    let started = std::time::Instant::now();
    for p in s.profiles() {
        let own_curves = Eval::sony(p);
        let key = format!("{} {}", "Sony", own_curves.model());
        let row = csv
            .get(&key)
            .unwrap_or_else(|| panic!("{key} is not in sony_profiles.csv"));
        let own = summary(&ev.map_error(&own_curves, &own_curves));
        let stand_in = summary(&ev.map_error(&own_curves, &a7r4));
        let mut check = |column: &'static str, ours: f64| {
            let published: f64 = row[column].parse().unwrap();
            let d = (ours - published).abs();
            assert!(
                d <= 0.02,
                "{key} {column}: ours {ours:.3}, published {published}"
            );
            let e = max_csv.entry(column).or_insert(0.0);
            *e = e.max(d);
        };
        check("own_median", own.0);
        check("own_p95", own.1);
        check("7RM4_median", stand_in.0);
        check("7RM4_p95", stand_in.1);
        check("7RM4_worst_median", stand_in.3);
        assert_eq!(stand_in.2, row["7RM4_worst_light"], "{key}: worst light");

        // The Python reference (same arithmetic, unrounded): every per-light median and p95.
        let py = &expected["bodies"][p.id.as_str()];
        for (tag, res) in [
            ("own", ev.map_error(&own_curves, &own_curves)),
            ("a7r_iv", ev.map_error(&own_curves, &a7r4)),
        ] {
            for (light, errors) in &res {
                let sm = Summary::of(errors).unwrap();
                let reference = &py[tag]["per_light"][light.as_str()];
                max_py = max_py
                    .max((sm.median - num(&reference["median"])).abs())
                    .max((sm.p95 - num(&reference["p95"])).abs());
            }
        }
        // docs/03: the A7R IV profile is its own stand-in.
        if p.id == s.registry().default_profile {
            assert_eq!((own.0, own.1), (stand_in.0, stand_in.1));
        }
    }
    eprintln!(
        "sony_profiles.csv vs Rust evaluation (11 bodies, {:.1} s):",
        started.elapsed().as_secs_f64()
    );
    for (column, d) in &max_csv {
        eprintln!("  max |difference| {column}: {d:.4}");
    }
    eprintln!(
        "  max |difference| vs the Python reference's per-light medians and p95s: {max_py:.2e}"
    );
    assert!(
        max_py <= 1e-6,
        "the Rust port drifted from the Python evaluation: {max_py:e}"
    );
}

// ---------------------------------------------------------------------------------------------
// The sensor-map gate with the shipped tables
// ---------------------------------------------------------------------------------------------

const MEDIAN_MAX: f64 = 0.5;
const P95_MAX: f64 = 1.2;

/// Rust against the Python reference where both do the same arithmetic (the shipped tables blended and
/// rendered): measured 1.2e-9, so 1e-6 leaves room for another libm and still catches any real change.
const SAME_ARITHMETIC_TOL: f64 = 1e-6;
/// Rust refit against the Python refit: two solvers on a cost that is flat to rounding noise for
/// coefficient changes of ~1e-8; measured 5.3e-9 in ΔE00.
const REFIT_TOL: f64 = 1e-5;

fn class_of(light: &Light) -> LightClass {
    match light.class.as_str() {
        "daylight" => LightClass::Daylight,
        "tungsten" => LightClass::Tungsten,
        "fluorescent" => LightClass::Fluorescent,
        "led" => LightClass::Led,
        other => panic!("unexpected class {other}"),
    }
}

struct GateRow {
    name: String,
    median: f64,
    p95: f64,
    max: f64,
}

impl GateRow {
    fn over(&self) -> bool {
        !(self.median <= MEDIAN_MAX && self.p95 <= P95_MAX)
    }
}

/// The gate for one body: each held-out light rendered through `map_for(light)`.
fn gate_rows(ev: &Eval, curves: &SpectralCurves, map_for: &dyn Fn(&Light) -> Mat3) -> Vec<GateRow> {
    ev.fx
        .lights_with_prefix("gate:")
        .map(|l| {
            let m = map_for(l);
            let errors = ev.render_error(curves, &m, &l.spd, l.cct_k, &ev.fx.ces.spectra);
            let s = Summary::of(&errors).unwrap();
            GateRow {
                name: l.name.trim_start_matches("gate:").to_string(),
                median: s.median,
                p95: s.p95,
                max: s.max,
            }
        })
        .collect()
}

/// The runtime blend with the true light known: its class with probability 1, `mired = 1e6 / CCT`.
fn blend_for(tables: &SensorMapTables, light: &Light) -> Mat3 {
    tables.blend(
        &ClassProbabilities::only(class_of(light)),
        1e6 / light.cct_k,
    )
}

#[test]
fn shipped_tables_reproduce_the_gate_numbers() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let expected = fx.expected["sensor_map_gate"].clone();
    assert_eq!(
        (num(&expected["median_max"]), num(&expected["p95_max"])),
        (MEDIAN_MAX, P95_MAX)
    );
    let ev = Eval::new(fx);
    let s = store();
    let mut max_diff = 0.0_f64;
    let mut a7r4_over = Vec::new();
    for p in s.profiles() {
        let curves = Eval::sony(p);
        let rows = gate_rows(&ev, &curves, &|l| blend_for(&p.sensor_map.classes, l));
        assert_eq!(rows.len(), 9);
        let py = &expected["profiles"][p.id.as_str()]["lights"];
        for row in &rows {
            // `shipped_fit`: the gate lights through the shipped tables (all 45 reference lights on grid).
            let reference = &py[row.name.as_str()]["shipped_fit"];
            for (what, ours, theirs) in [
                ("median", row.median, num(&reference["median"])),
                ("p95", row.p95, num(&reference["p95"])),
                ("max", row.max, num(&reference["max"])),
            ] {
                let d = (ours - theirs).abs();
                // The same arithmetic on the same inputs (measured: 1.2e-9). A looser bound than the
                // table rounding would hide a blend or interpolation change of a few hundredths of a
                // ΔE00, which is what this comparison is for.
                assert!(
                    d <= SAME_ARITHMETIC_TOL,
                    "{} {} {what}: ours {ours:.6}, reference {theirs:.6}",
                    p.id,
                    row.name
                );
                max_diff = max_diff.max(d);
            }
        }
        let over: Vec<&str> = rows
            .iter()
            .filter(|r| r.over())
            .map(|r| r.name.as_str())
            .collect();
        let worst_median = rows.iter().map(|r| r.median).fold(0.0, f64::max);
        let worst_p95 = rows.iter().map(|r| r.p95).fold(0.0, f64::max);
        eprintln!(
            "shipped tables, {:<17} worst median {worst_median:.3}, worst p95 {worst_p95:.3}, over the gate: {over:?}",
            p.id
        );
        if p.id == s.registry().default_profile {
            a7r4_over = over.iter().map(ToString::to_string).collect();
            for r in &rows {
                eprintln!(
                    "  A7R IV {:<18} median {:.3}  p95 {:.3}  max {:.3}",
                    r.name, r.median, r.p95, r.max
                );
            }
        }
    }
    eprintln!("max |difference| to the Python 'shipped_fit' numbers: {max_diff:.2e}");
    // Mykyta's decisions: the gate stays as specified (thresholds unchanged). 2026-10-12: LED-V2 is a
    // reported failure. 2026-10-25, "Move ALEXA estimate": the estimate the gate scores against carries
    // a hue push, and five more lights are over. 2026-10-26, "4.5": the push is taken from 1.5 to 4.5
    // degrees, and LED-RGB1 is over as well (docs/03, docs/STATUS.md have the numbers before and after).
    // Through the shipped tables (all 45 reference lights on grid) LED-RGB1 passes (0.362 / 1.084); the gate
    // refit, which holds it out of the LED class, leaves it over (0.555 / 1.369, below).
    assert_eq!(
        a7r4_over,
        accepted_failures()
            .into_iter()
            .filter(|l| l != "LED-RGB1")
            .collect::<Vec<_>>(),
        "the A7R IV gate through the shipped tables: the accepted failures without LED-RGB1, which the shipped fit clears"
    );
}

/// The lights of the A7R IV gate that are over, as the owner accepted them: LED-V2 (2026-10-12), with the
/// ALEXA estimate moved (2026-10-25) blackbody 2850 K, D 5250 K, D 6750 K, FL3.15 and LED-B3, and with the
/// push taken to 4.5 degrees (2026-10-26) LED-RGB1. The same list is `--expect-failing` in
/// `.github/workflows/ci.yml`.
fn accepted_failures() -> Vec<String> {
    [
        "blackbody-2850K",
        "daylight-5250K",
        "daylight-6750K",
        "FL3.15",
        "LED-B3",
        "LED-RGB1",
        "LED-V2",
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

// ---------------------------------------------------------------------------------------------
// The Lab-optimal fit in Rust (fit.fit_map_lab) and the gate refit
// ---------------------------------------------------------------------------------------------

/// Solves the small dense system `a x = b` by Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col];
        for row in col + 1..N {
            let f = a[row][col] / pivot_row[col];
            for (x, y) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *x -= f * y;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Some(x)
}

fn white_preserving(p: &[f64; 6]) -> Mat3 {
    [
        [p[0], p[1], 1.0 - p[0] - p[1]],
        [p[2], p[3], 1.0 - p[2] - p[3]],
        [p[4], p[5], 1.0 - p[4] - p[5]],
    ]
}

/// Derivative of the Lab nonlinearity `f` (`fit._lab_fprime`).
fn lab_fprime(t: f64) -> f64 {
    const EPSILON: f64 = 216.0 / 24389.0;
    const KAPPA: f64 = 24389.0 / 27.0;
    if t > EPSILON {
        t.cbrt().powi(-2) / 3.0
    } else {
        KAPPA / 116.0
    }
}

impl Eval {
    /// The shipped objective: the white-preserving 3x3 minimising the CIE Lab error after ARRI's matrix,
    /// `sum_n |Lab(A M c_s[n]) - Lab(A c_a[n])|^2` over the 190 training patches (`fit.fit_map_lab`).
    /// Levenberg-Marquardt with the analytic Jacobian, started from the research objective's solution.
    fn fit_map_lab(&self, s_src: &SpectralCurves, light: &Light) -> Mat3 {
        let train = &self.fx.training.spectra;
        let m_arri = self.arri.native_to_ap0(light.cct_k);
        let src = Self::camera_rgb_wb(s_src, &light.spd, train);
        let dst = Self::camera_rgb_wb(&self.alexa, &light.spd, train);
        let lab_target: Vec<Vec3> = dst
            .iter()
            .map(|d| self.ap0_to_lab(linalg::mul_vec(&m_arri, *d)))
            .collect();
        let c = linalg::mul(&matrices().ap0_to_xyz, &m_arri); // white-balanced ALEXA-native RGB -> XYZ
        let w = self.aces_white;

        let start = self.fit_map_weighted_rgb(s_src, &self.alexa, &light.spd);
        let mut p = [
            start[0][0],
            start[0][1],
            start[1][0],
            start[1][1],
            start[2][0],
            start[2][1],
        ];

        // Residuals (3 per patch) and Jacobian rows for a parameter vector.
        let evaluate = |p: &[f64; 6]| -> (f64, [[f64; 6]; 6], [f64; 6]) {
            let m = white_preserving(p);
            let (mut cost, mut jtj, mut jtr) = (0.0, [[0.0; 6]; 6], [0.0; 6]);
            for (s, target) in src.iter().zip(&lab_target) {
                let xyz = linalg::mul_vec(&c, linalg::mul_vec(&m, *s));
                let lab = xyz_to_lab(xyz, w);
                let r = [lab[0] - target[0], lab[1] - target[1], lab[2] - target[2]];
                cost += 0.5 * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                let fp = [
                    lab_fprime(xyz[0] / w[0]) / w[0],
                    lab_fprime(xyz[1] / w[1]) / w[1],
                    lab_fprime(xyz[2] / w[2]) / w[2],
                ];
                // dLab/dXYZ
                let d_lab = [
                    [0.0, 116.0 * fp[1], 0.0],
                    [500.0 * fp[0], -500.0 * fp[1], 0.0],
                    [0.0, 200.0 * fp[1], -200.0 * fp[2]],
                ];
                // dXYZ/dp[2r + j] = c[:, r] * (src[j] - src[2])
                let mut jac = [[0.0; 6]; 3];
                for (i, jac_row) in jac.iter_mut().enumerate() {
                    for r_idx in 0..3 {
                        for j in 0..2 {
                            let mut v = 0.0;
                            for k in 0..3 {
                                v += d_lab[i][k] * c[k][r_idx];
                            }
                            jac_row[2 * r_idx + j] = v * (s[j] - s[2]);
                        }
                    }
                }
                for i in 0..3 {
                    for a in 0..6 {
                        jtr[a] += jac[i][a] * r[i];
                        for b in 0..6 {
                            jtj[a][b] += jac[i][a] * jac[i][b];
                        }
                    }
                }
            }
            (cost, jtj, jtr)
        };

        let (mut cost, mut jtj, mut jtr) = evaluate(&p);
        let mut lambda = 1e-3;
        for _ in 0..200 {
            let mut a = jtj;
            for (i, row) in a.iter_mut().enumerate() {
                row[i] += lambda * jtj[i][i].max(1e-12);
            }
            let Some(step) = solve(a, jtr.map(|v| -v)) else {
                break;
            };
            let mut trial = p;
            for (t, s) in trial.iter_mut().zip(step) {
                *t += s;
            }
            let (new_cost, new_jtj, new_jtr) = evaluate(&trial);
            if new_cost <= cost {
                let converged = step.iter().fold(0.0_f64, |m, s| m.max(s.abs())) < 1e-14;
                (p, cost, jtj, jtr) = (trial, new_cost, new_jtj, new_jtr);
                lambda = (lambda * 0.1).max(1e-15);
                if converged {
                    break;
                }
            } else {
                lambda *= 10.0;
                if lambda > 1e12 {
                    break;
                }
            }
        }
        white_preserving(&p)
    }

    /// Fits every reference light of a class (`ref:<class>:*`, already sorted by mired), skipping `exclude`.
    fn fit_tables(&self, s_src: &SpectralCurves, exclude: &[&str]) -> SensorMapTables {
        let entries = |class: &str| -> Vec<MapEntry> {
            self.fx
                .lights_with_prefix(&format!("ref:{class}:"))
                .filter(|l| !exclude.contains(&l.name.rsplit(':').next().unwrap()))
                .map(|l| MapEntry {
                    light: l.name.rsplit(':').next().unwrap().to_string(),
                    mired: 1e6 / l.cct_k,
                    m: self.fit_map_lab(s_src, l),
                })
                .collect()
        };
        SensorMapTables {
            daylight: entries("daylight"),
            tungsten: entries("tungsten"),
            fluorescent: entries("fluorescent"),
            led: entries("led"),
        }
    }
}

#[test]
fn the_rust_lab_fit_reproduces_the_shipped_tables() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let ev = Eval::new(fx);
    let s = store();
    let mut worst = 0.0_f64;
    let started = std::time::Instant::now();
    for id in ["sony-ilce-7rm4", "sony-ilce-7sm2", "sony-dsc-rx100m4"] {
        let p = s.profile_by_name(id).unwrap();
        let fitted = ev.fit_tables(&Eval::sony(p), &[]);
        for class in LightClass::ALL {
            let (ours, shipped) = (fitted.class(class), p.sensor_map.classes.class(class));
            assert_eq!(ours.len(), shipped.len(), "{id} {class:?}");
            for (a, b) in ours.iter().zip(shipped) {
                assert_eq!(a.light, b.light, "{id} {class:?}: order of lights");
                assert!((a.mired - b.mired).abs() < 1e-6, "{id} {}: mired", a.light);
                let d = linalg::max_abs_diff(&a.m, &b.m);
                // The cost is flat to rounding noise for changes of ~1e-8 in the coefficients (README), so an
                // independent solver lands within that valley, not on the same digits.
                assert!(
                    d < 1e-6,
                    "{id} {}: the Rust fit differs from the shipped matrix by {d:e}",
                    a.light
                );
                worst = worst.max(d);
            }
        }
    }
    eprintln!(
        "Lab fit in Rust vs shipped tables (3 bodies x 45 lights, {:.1} s): max |difference| of a matrix element {worst:.2e}",
        started.elapsed().as_secs_f64()
    );
}

#[test]
fn the_gate_with_a_rust_refit_reproduces_the_gate_numbers() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let expected = fx.expected["sensor_map_gate"].clone();
    let ev = Eval::new(fx);
    let s = store();
    let held_out = ["LED-B3", "LED-RGB1", "LED-V2"];
    let started = std::time::Instant::now();
    let mut max_diff = 0.0_f64;
    let mut worst_overall = (0.0_f64, 0.0_f64);
    let mut a7r4 = Vec::new();
    for p in s.profiles() {
        let curves = Eval::sony(p);
        // Refit with the three LEDs held out of the LED class; tungsten, daylight and fluorescent keep all
        // their lights (none of the nine gate lights is one of them).
        let refit = ev.fit_tables(&curves, &held_out);
        assert_eq!(refit.led.len(), 6);
        let rows = gate_rows(&ev, &curves, &|l| blend_for(&refit, l));
        let py = &expected["profiles"][p.id.as_str()];
        for row in &rows {
            let reference = &py["lights"][row.name.as_str()]["gate"];
            for (what, ours, theirs) in [
                ("median", row.median, num(&reference["median"])),
                ("p95", row.p95, num(&reference["p95"])),
                ("max", row.max, num(&reference["max"])),
            ] {
                let d = (ours - theirs).abs();
                // An independent solver (Rust Levenberg-Marquardt against SciPy's) lands in the same
                // valley of the flat Lab cost: measured 5.3e-9.
                assert!(
                    d <= REFIT_TOL,
                    "{} {} {what}: ours {ours:.6}, Python gate {theirs:.6}",
                    p.id,
                    row.name
                );
                max_diff = max_diff.max(d);
            }
        }
        let over: Vec<String> = rows
            .iter()
            .filter(|r| r.over())
            .map(|r| r.name.clone())
            .collect();
        let python_over: Vec<String> = py["lights_over"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(over, python_over, "{}: lights over the gate", p.id);
        let worst_median = rows.iter().map(|r| r.median).fold(0.0, f64::max);
        let worst_p95 = rows.iter().map(|r| r.p95).fold(0.0, f64::max);
        assert!(
            (worst_median - num(&py["worst_median"])).abs() <= REFIT_TOL
                && (worst_p95 - num(&py["worst_p95"])).abs() <= REFIT_TOL
        );
        eprintln!(
            "gate (Rust refit), {:<17} worst median {worst_median:.3}, worst p95 {worst_p95:.3}, over: {over:?}",
            p.id
        );
        if p.id == s.registry().default_profile {
            worst_overall = (worst_median, worst_p95);
            a7r4 = rows;
        }
    }
    eprintln!(
        "gate with the Rust refit, 11 bodies x 9 lights ({:.1} s): max |difference| to gate.json {max_diff:.2e}",
        started.elapsed().as_secs_f64()
    );
    for r in &a7r4 {
        eprintln!(
            "  A7R IV {:<18} median {:.3}  p95 {:.3}{}",
            r.name,
            r.median,
            r.p95,
            if r.over() { "   <-- over" } else { "" }
        );
    }
    // The M1-05 gate for the A7R IV profile: thresholds unchanged, the failures the owner accepted
    // (LED-V2 on 2026-10-12; five more with the ALEXA estimate moved on 2026-10-25; LED-RGB1 with the
    // push taken to 4.5 degrees on 2026-10-26).
    let over: Vec<&str> = a7r4
        .iter()
        .filter(|r| r.over())
        .map(|r| r.name.as_str())
        .collect();
    let accepted = accepted_failures();
    assert_eq!(
        over,
        accepted.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert!(
        a7r4.iter()
            .filter(|r| !accepted.contains(&r.name))
            .all(|r| r.median <= MEDIAN_MAX && r.p95 <= P95_MAX)
    );
    assert!(
        worst_overall.0 > MEDIAN_MAX,
        "LED-V2's median is over the gate: {worst_overall:?}"
    );
}
