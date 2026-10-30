//! The fit (`docs/09`, Profile, "Fit procedure"): first the tone curve and `BaselineExposure` offset on the neutral
//! axis, then the look table for chroma.
//!
//! # The scale of the DNG
//!
//! Adobe's renderer clips what reaches its tone curve at 1 after the exposure gain. The cine rendering reaches display white exactly at scene 243 (+10.4 stops) and is already at
//! 94 % of display light at 16, so the curve covers scene values up to 16 (the limit of `ym_color::display::DNG_SCENE_LIMIT`): the stored values are divided by `x_white` and `BaselineExposure` is `log2(x_white)` plus the profile's
//! offset (`-log2(x_min)`, -4 for `x_min` 16), which puts the renderer's input at `x / x_min`.
//!
//! **That is only right while the exposure is zero or positive.** For a negative `BaselineExposure` the DNG SDK's
//! renderer does not scale: it applies a highlight shoulder (linear with slope `2^b` up to 0.25 of the range, then a
//! quadratic that ends at 1), so the same profile would render a photo with `x_white` 2 differently from one with
//! `x_white` 8. Measured: with `b = -3` and an identity profile, scene 0.4 renders at 0.086 of display light instead of
//! 0.050 and 0.84 at 0.654 instead of 0.105. The writer therefore has to floor `x_white` at `x_min`
//! (`max(p99.99, x_min)`, where it floors at 1.0 today), which keeps `b = log2(x_white / x_min)` non-negative for
//! every photo. A photo whose 99.99th percentile exceeds `x_min` gets a positive exposure and a wider range.

use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::Serialize;
use ym_color::consts::{Vec3, linalg};
use ym_color::deltae::percentile;
use ym_color::encode::{srgb_decode, srgb_encode};
use ym_export::dng::{DngProfile, LookTable};

use crate::adobe::Adobe;
use crate::model::{Matrices, ToneTable, before_tone, hsv_to_rgb, rgb_to_hsv};
use crate::target::{cine_srgb, delta_e_codes};

/// The renderer samples its tone curve into a table of this many intervals.
pub const TONE_GRID: usize = 4096;

/// What to fit.
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// The scene-linear AWG4 value that maps to 1.0 at the input of the tone curve (`2^-fit_offset_ev`).
    pub x_min: f64,
    /// Points of the `ProfileToneCurve`.
    pub tone_points: usize,
    /// Iterations of the knot correction.
    pub tone_iterations: usize,
    /// `[hue, saturation, value]` divisions of the look table.
    pub dims: [u32; 3],
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            x_min: 16.0,
            tone_points: 64,
            tone_iterations: 8,
            dims: [36, 32, 16],
        }
    }
}

impl FitConfig {
    /// The `BaselineExposure` offset in EV.
    #[must_use]
    pub fn offset_ev(&self) -> f64 {
        -self.x_min.log2()
    }

    /// Checks the settings before anything is rendered: a scale the profile file can hold, enough tone points, and a table
    /// of 1 to 36 hues, 2 to 32 saturations and 2 to 16 values (the DNG limits; the sRGB encoding of the fit needs two
    /// value divisions, and a saturation axis needs two).
    ///
    /// # Errors
    ///
    /// A message that says which setting is wrong.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.x_min.is_finite() && (1.0 / 16.0..=4096.0).contains(&self.x_min),
            "--x-min {} is outside 1/16 to 4096 (BaselineExposure's offset must stay between -12 and 4 EV)",
            self.x_min
        );
        ensure!(
            self.tone_points >= 4,
            "--tone-points {} is too few: the tone curve needs at least 4 points",
            self.tone_points
        );
        let [h, s, v] = self.dims;
        ensure!(
            (1..=36).contains(&h) && (2..=32).contains(&s) && (2..=16).contains(&v),
            "--dims {h},{s},{v} is outside the look table's limits: 1 to 36 hues, 2 to 32 saturations, 2 to 16 values"
        );
        Ok(())
    }
}

/// The cine rendering's neutral axis as the tone curve's target: the linear light of sRGB code `d(x)` for the grey `x`.
#[must_use]
pub fn neutral_target_linear(x: f64) -> f64 {
    srgb_decode(cine_srgb([x; 3])[1])
}

/// The curve's knot positions: 0, then knots that are geometric through the shadows (the renderer samples the curve at
/// 1/4096 steps, so nothing below that matters) and spread evenly in the part above 1/16 of the range.
#[must_use]
pub fn knot_positions(points: usize) -> Vec<f64> {
    let lo = 1.0 / TONE_GRID as f64;
    let split = 1.0 / 16.0;
    // a third of the points in the shadows, the rest over the upper 15/16 of the range
    let n_low = (points / 3).max(2);
    let n_high = points.saturating_sub(n_low).max(2);
    let mut u = vec![0.0];
    for k in 0..n_low {
        let t = k as f64 / n_low as f64;
        u.push(lo * (split / lo).powf(t));
    }
    for k in 0..n_high {
        let t = (k + 1) as f64 / n_high as f64;
        // a little denser just above the split, where the cine rendering is still bending hard
        u.push(split + (1.0 - split) * (0.5 * t + 0.5 * t * t));
    }
    if let Some(last) = u.last_mut() {
        *last = 1.0;
    }
    u
}

/// Measures the renderer's tone curve for `profile`: neutral colours at `i / TONE_GRID` of its input range with the
/// identity look, rendered by `dng_validate`, returned as linear light (the sRGB decode of the output).
pub fn measure_tone(adobe: &Adobe, tone_curve: &[[f32; 2]], x_min: f64) -> Result<ToneTable> {
    let mut profile = DngProfile::identity();
    profile.tone_curve = tone_curve.to_vec();
    let colors: Vec<[f32; 3]> = (0..=TONE_GRID)
        .map(|i| {
            let x = (i as f64 / TONE_GRID as f64 * x_min) as f32;
            [x; 3]
        })
        .collect();
    let r = adobe.render(&colors, &profile, -x_min.log2())?;
    ensure!(
        r.report.highlight_clipped_fraction == 0.0,
        "the neutral ramp was clipped by the writer's white level"
    );
    let samples = r
        .srgb
        .iter()
        .map(|o| srgb_decode(f64::from(o[1])))
        .collect();
    Ok(ToneTable { samples })
}

/// How well the neutral axis came out.
#[derive(Clone, Debug, Serialize)]
pub struct ToneReport {
    /// Knots in the curve.
    pub points: usize,
    /// Iterations run.
    pub iterations: usize,
    /// The largest ΔE00 between Adobe's neutral and the cine rendering over the grid above `floor_scene`, per iteration.
    pub max_delta_e_by_iteration: Vec<f64>,
    /// Median ΔE00 over the same grid after the last iteration.
    pub median_delta_e: f64,
    /// 99th percentile of the same.
    pub p99_delta_e: f64,
    /// Largest difference of 8-bit sRGB codes (x255) anywhere on the grid.
    pub max_code_difference_8bit: f64,
    /// Scene values below this are left out of the statistics (display code under 1/255, invisible).
    pub floor_scene: f64,
}

/// ΔE00 of Adobe's neutral against the cine rendering along the measured grid.
#[must_use]
pub fn neutral_errors(tone: &ToneTable, x_min: f64) -> Vec<(f64, f64, f64)> {
    let n = TONE_GRID;
    (0..=n)
        .map(|i| {
            let u = i as f64 / n as f64;
            let x = u * x_min;
            let want = cine_srgb([x; 3]);
            let got = srgb_encode(tone.samples[i].clamp(0.0, 1.0));
            (
                x,
                delta_e_codes(want, [got; 3]),
                (want[1] - got).abs() * 255.0,
            )
        })
        .collect()
}

/// Fits the tone curve on the neutral axis by correcting the knots until Adobe's output matches the cine rendering.
pub fn fit_tone(adobe: &Adobe, cfg: &FitConfig) -> Result<(Vec<[f32; 2]>, ToneTable, ToneReport)> {
    let u = knot_positions(cfg.tone_points);
    let target_at = |uk: f64| neutral_target_linear(uk * cfg.x_min);
    let mut y: Vec<f64> = u.iter().map(|&uk| target_at(uk)).collect();
    let to_curve = |y: &[f64]| -> Vec<[f32; 2]> {
        u.iter()
            .zip(y)
            .map(|(&a, &b)| [a as f32, b.clamp(0.0, 1.0) as f32])
            .collect()
    };
    let mut history = Vec::new();
    let mut tone;
    let floor_scene = 0.5 / 255.0 / 12.92; // below one code of sRGB 8-bit
    for _ in 0..cfg.tone_iterations {
        tone = measure_tone(adobe, &to_curve(&y), cfg.x_min)?;
        let errors = neutral_errors(&tone, cfg.x_min);
        let worst = errors
            .iter()
            .filter(|e| e.0 > floor_scene)
            .map(|e| e.1)
            .fold(0.0, f64::max);
        history.push(worst);
        for (k, &uk) in u.iter().enumerate() {
            let measured = tone.eval(uk);
            y[k] += target_at(uk) - measured;
            y[k] = y[k].clamp(0.0, 1.0);
        }
        // the curve of a renderer is read as non-decreasing
        for k in 1..y.len() {
            if y[k] < y[k - 1] {
                y[k] = y[k - 1];
            }
        }
    }
    let curve = to_curve(&y);
    tone = measure_tone(adobe, &curve, cfg.x_min)?;
    let errors = neutral_errors(&tone, cfg.x_min);
    let mut des: Vec<f64> = errors
        .iter()
        .filter(|e| e.0 > floor_scene)
        .map(|e| e.1)
        .collect();
    des.sort_by(f64::total_cmp);
    let q = |f: f64| des[((des.len() - 1) as f64 * f) as usize];
    let report = ToneReport {
        points: curve.len(),
        iterations: cfg.tone_iterations,
        max_delta_e_by_iteration: {
            let mut h = history;
            h.push(q(1.0));
            h
        },
        median_delta_e: q(0.5),
        p99_delta_e: q(0.99),
        max_code_difference_8bit: errors.iter().map(|e| e.2).fold(0.0, f64::max),
        floor_scene,
    };
    Ok((curve, tone, report))
}

/// The linear ProPhoto colour at a node of the look table (after the renderer's gain): hue `hi / H` of the circle,
/// saturation `si / (S - 1)`, value `vi / (V - 1)` in the table's encoding.
#[must_use]
pub fn node_color(dims: [u32; 3], encoding: u32, hi: u32, si: u32, vi: u32) -> Vec3 {
    let [hd, sd, vd] = dims;
    let h = 6.0 * f64::from(hi) / f64::from(hd);
    let s = f64::from(si) / f64::from(sd - 1);
    let ve = if vd > 1 {
        f64::from(vi) / f64::from(vd - 1)
    } else {
        1.0
    };
    let v = if encoding == 1 && vd > 1 {
        srgb_decode(ve)
    } else {
        ve
    };
    hsv_to_rgb([h, s, v])
}

fn wrap_sixths(d: f64) -> f64 {
    let mut d = d % 6.0;
    if d > 3.0 {
        d -= 6.0;
    } else if d <= -3.0 {
        d += 6.0;
    }
    d
}

/// What the analytic solve of the look table found.
#[derive(Clone, Debug, Serialize)]
pub struct LookReport {
    /// `[hue, saturation, value]` divisions.
    pub dims: [u32; 3],
    /// Entries in all.
    pub entries: usize,
    /// Entries computed from a target (the rest are the zero-saturation row, black, or filled from neighbours).
    pub solved: usize,
    /// Entries whose colour needs a negative AWG4 component (outside the gamut of the scene data), filled from neighbours.
    pub outside_awg4: usize,
    /// Range of the entries: hue shift in degrees.
    pub hue_shift_range_deg: [f64; 2],
    /// Range of the saturation scales.
    pub saturation_scale_range: [f64; 2],
    /// Range of the value scales.
    pub value_scale_range: [f64; 2],
}

/// The look table that turns the renderer's HSV colours into the colours the cine rendering gives, solved analytically at every
/// node with the model of the renderer ([`crate::model`]): the node's colour is a point on the grid, so the entry there
/// acts alone, and the table has to produce the colour that the tone curve then maps to the cine rendering's output. The nodes whose
/// colour lies outside the AWG4 gamut (the scene data has no such colour) are filled from their neighbours.
#[must_use]
pub fn solve_look(cfg: &FitConfig, tone: &ToneTable) -> (LookTable, LookReport) {
    let m = Matrices::new();
    let gain = cfg.offset_ev().exp2();
    let [hd, sd, vd] = cfg.dims;
    let encoding = 1;
    let total = (hd * sd * vd) as usize;
    let index = |vi: u32, hi: u32, si: u32| ((vi * hd + hi) * sd + si) as usize;
    let mut entries: Vec<Option<[f64; 3]>> = vec![None; total];
    let mut solved = 0;
    let mut outside = 0;
    for vi in 0..vd {
        for hi in 0..hd {
            for si in 0..sd {
                let i = index(vi, hi, si);
                let p = node_color(cfg.dims, encoding, hi, si, vi);
                let [h, s, v] = rgb_to_hsv(p);
                if si == 0 || v <= 0.0 {
                    entries[i] = Some([0.0, 1.0, 1.0]);
                    continue;
                }
                let x = linalg::mul_vec(&m.prophoto_to_awg4, p).map(|c| c / gain);
                if x.iter().any(|&c| c < -1e-9) {
                    outside += 1;
                    continue;
                }
                let target = cine_srgb(x.map(|c| c.max(0.0)));
                let q = before_tone(target, tone, &m);
                let [h2, s2, v2] = rgb_to_hsv(q);
                let dh = if s2 < 1e-9 {
                    0.0
                } else {
                    wrap_sixths(h2 - h) * 60.0
                };
                let ds = s2 / s;
                let dv = if v2 <= 0.0 {
                    0.0
                } else {
                    srgb_encode(v2.min(1.0)) / srgb_encode(v.min(1.0))
                };
                entries[i] = Some([dh, ds, dv]);
                solved += 1;
            }
        }
    }
    fill_from_neighbours(&mut entries, cfg.dims);
    let data: Vec<f32> = entries
        .iter()
        .flat_map(|e| e.unwrap_or([0.0, 1.0, 1.0]).map(|v| v as f32))
        .collect();
    let range = |k: usize| {
        let (lo, hi) = data
            .chunks_exact(3)
            .map(|e| f64::from(e[k]))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
        [lo, hi]
    };
    let report = LookReport {
        dims: cfg.dims,
        entries: total,
        solved,
        outside_awg4: outside,
        hue_shift_range_deg: range(0),
        saturation_scale_range: range(1),
        value_scale_range: range(2),
    };
    (
        LookTable {
            dims: cfg.dims,
            data,
            encoding,
        },
        report,
    )
}

/// Gives the unset entries the mean of their set neighbours (hue wraps), repeatedly, so a hole fills from its edges.
fn fill_from_neighbours(entries: &mut [Option<[f64; 3]>], dims: [u32; 3]) {
    let [hd, sd, vd] = dims.map(|d| d as usize);
    let at = |v: usize, h: usize, s: usize| (v * hd + h) * sd + s;
    loop {
        let mut next: Vec<(usize, [f64; 3])> = Vec::new();
        for v in 0..vd {
            for h in 0..hd {
                for s in 0..sd {
                    let i = at(v, h, s);
                    if entries[i].is_some() {
                        continue;
                    }
                    let mut neighbours = Vec::new();
                    neighbours.push(at(v, (h + 1) % hd, s));
                    neighbours.push(at(v, (h + hd - 1) % hd, s));
                    if s + 1 < sd {
                        neighbours.push(at(v, h, s + 1));
                    }
                    if s > 0 {
                        neighbours.push(at(v, h, s - 1));
                    }
                    if v + 1 < vd {
                        neighbours.push(at(v + 1, h, s));
                    }
                    if v > 0 {
                        neighbours.push(at(v - 1, h, s));
                    }
                    let set: Vec<[f64; 3]> =
                        neighbours.iter().filter_map(|&n| entries[n]).collect();
                    if !set.is_empty() {
                        let mut mean = [0.0; 3];
                        for e in &set {
                            for k in 0..3 {
                                mean[k] += e[k] / set.len() as f64;
                            }
                        }
                        next.push((i, mean));
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        for (i, e) in next {
            entries[i] = Some(e);
        }
    }
}

/// Colour differences of Adobe's rendering against the cine rendering over a set of colours.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Stats {
    /// Colours compared.
    pub count: usize,
    /// Median ΔE00.
    pub median: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Mean.
    pub mean: f64,
    /// Maximum.
    pub max: f64,
}

impl Stats {
    /// Statistics of `values` (empty gives zeros).
    #[must_use]
    pub fn of(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
                mean: 0.0,
                max: 0.0,
            };
        }
        let q = |f: f64| percentile(values, f).unwrap_or(f64::NAN);
        Self {
            count: values.len(),
            median: q(50.0),
            p95: q(95.0),
            p99: q(99.0),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            max: values.iter().copied().fold(0.0, f64::max),
        }
    }
}

/// ΔE00 between what Adobe's renderer makes of each colour (scene-linear AWG4) with `profile` and what the cine rendering makes of it.
pub fn evaluate(
    adobe: &Adobe,
    profile: &DngProfile,
    fit_offset_ev: f64,
    colors: &[[f32; 3]],
) -> Result<Vec<f64>> {
    const CHUNK: usize = 60_000;
    let mut out = Vec::with_capacity(colors.len());
    for chunk in colors.chunks(CHUNK) {
        let r = adobe.render(chunk, profile, fit_offset_ev)?;
        ensure!(
            r.report.highlight_clipped_fraction == 0.0,
            "the writer's white level clipped some of the test colours"
        );
        let des: Vec<f64> = chunk
            .par_iter()
            .zip(r.srgb.par_iter())
            .map(|(c, got)| {
                let want = cine_srgb(c.map(f64::from));
                delta_e_codes(want, got.map(f64::from))
            })
            .collect();
        out.extend(des);
    }
    Ok(out)
}

/// What fitting the look table did.
#[derive(Clone, Debug, Serialize)]
pub struct LookFitReport {
    /// The node-exact solution the least-squares fit starts from.
    pub node_exact: LookReport,
    /// The least-squares solve.
    pub least_squares: crate::lookfit::SolveReport,
}

/// Fits the look table: the node-exact solution first, then the least-squares fit to a grid of colours `density` times
/// as fine as the table's (see [`crate::lookfit`]).
#[must_use]
pub fn fit_look(
    cfg: &FitConfig,
    tone: &ToneTable,
    density: usize,
    pull: f64,
    iterations: usize,
) -> (LookTable, LookFitReport) {
    let (exact, node_report) = solve_look(cfg, tone);
    let dims = cfg.dims;
    let start: Vec<[f64; 3]> = exact
        .data
        .chunks_exact(3)
        .map(|e| [f64::from(e[0]), f64::from(e[1]), f64::from(e[2])])
        .collect();
    let sd = dims[1];
    // the value scale at zero saturation is 1 by the DNG specification
    let fixed: Vec<[bool; 3]> = (0..start.len())
        .map(|k| [false, false, (k as u32).is_multiple_of(sd)])
        .collect();
    let samples = crate::lookfit::grid_samples(dims, cfg.offset_ev().exp2(), tone, density);
    let (solution, report) =
        crate::lookfit::solve(dims, &samples, &start, &fixed, pull, iterations);
    let data: Vec<f32> = solution
        .iter()
        .flat_map(|e| {
            [
                e[0].clamp(-180.0, 180.0) as f32,
                e[1].max(0.0) as f32,
                e[2].max(0.0) as f32,
            ]
        })
        .collect();
    (
        LookTable {
            dims,
            data,
            encoding: 1,
        },
        LookFitReport {
            node_exact: node_report,
            least_squares: report,
        },
    )
}

/// A named set of test colours.
pub struct TestSet {
    /// What the set is.
    pub name: String,
    /// The colours (scene-linear AWG4).
    pub colors: Vec<[f32; 3]>,
}

/// The sets the fit is reported on: the neutral ramp, hue and chroma sweeps at four lightnesses, a seeded random set of
/// photograph-like colours, and (when the scenes exist) the chart patches of the synthetic golden scenes at eight
/// exposures. None of them is used by the fit, which works on a regular grid of the HSV space.
pub fn test_sets(scenes_dir: Option<&std::path::Path>) -> Vec<TestSet> {
    let mut sets = vec![
        TestSet {
            name: "neutral ramp (400 greys, 0.002 to 16)".to_owned(),
            colors: crate::samples::neutral_ramp(400, 0.002, 16.0),
        },
        TestSet {
            name: "hue and chroma sweep (36 hues x 10 chromas x 4 lightnesses)".to_owned(),
            colors: crate::samples::hue_chroma_sweep(36, 10, &[25.0, 45.0, 65.0, 85.0], 90.0),
        },
        TestSet {
            name: "random photograph-like colours (20000, seed 11)".to_owned(),
            colors: crate::samples::random_colors(20000, 11),
        },
    ];
    if let Some(dir) = scenes_dir
        && let Ok(colors) =
            crate::samples::golden_patch_colors(dir, &[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        && !colors.is_empty()
    {
        sets.push(TestSet {
            name: "synthetic golden scenes: chart patches at -4 to +3 EV".to_owned(),
            colors,
        });
    }
    sets
}

/// One test set's result.
#[derive(Clone, Debug, Serialize)]
pub struct SetResult {
    /// The set's name.
    pub name: String,
    /// ΔE00 of Adobe's render against the cine rendering.
    pub stats: Stats,
}

/// Evaluates `profile` on every set (renders them with `dng_validate`).
pub fn evaluate_sets(
    adobe: &Adobe,
    profile: &DngProfile,
    fit_offset_ev: f64,
    sets: &[TestSet],
) -> Result<Vec<SetResult>> {
    sets.iter()
        .map(|s| {
            Ok(SetResult {
                name: s.name.clone(),
                stats: Stats::of(&evaluate(adobe, profile, fit_offset_ev, &s.colors)?),
            })
        })
        .collect()
}

/// Rounds the table's entries to the precision the profile file keeps (hue shifts to 0.001 degrees, scales to 1e-5), so
/// that what is evaluated is what is written.
pub fn round_table(table: &mut LookTable) {
    for e in table.data.chunks_exact_mut(3) {
        e[0] = (f64::from(e[0]) * 1e3).round() as f32 / 1e3;
        e[1] = (f64::from(e[1]) * 1e5).round() as f32 / 1e5;
        e[2] = (f64::from(e[2]) * 1e5).round() as f32 / 1e5;
    }
}

/// Rounds the curve's points (to 1e-7 in the input and 1e-6 in the output).
pub fn round_curve(curve: &mut [[f32; 2]]) {
    for p in curve {
        p[0] = (f64::from(p[0]) * 1e7).round() as f32 / 1e7;
        p[1] = (f64::from(p[1]) * 1e6).round() as f32 / 1e6;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knots_start_at_zero_end_at_one_and_increase() {
        for points in [16, 64, 128] {
            let u = knot_positions(points);
            assert_eq!(u[0], 0.0);
            assert_eq!(*u.last().unwrap(), 1.0);
            assert!(
                u.windows(2).all(|w| w[1] > w[0]),
                "{points}: strictly increasing"
            );
            assert!(
                u.len() >= points - 1 && u.len() <= points + 4,
                "{} for {points}",
                u.len()
            );
            // dense in the shadows: a third of the points are below 1/16 of the range
            assert!(u.iter().filter(|&&x| x < 1.0 / 16.0 + 1e-9).count() >= points / 3);
        }
    }

    #[test]
    fn the_neutral_target_is_the_cine_axis() {
        // The cine rendering's black is 8e-14 at scene 0 (its soft floor), not exactly 0.
        assert!(neutral_target_linear(0.0) < 1e-12);
        let mid = neutral_target_linear(0.18);
        assert!(
            (mid - 0.1109).abs() < 5e-4,
            "18 % grey lands at about 11 % display light: {mid}"
        );
        assert!(neutral_target_linear(16.0) > 0.9 && neutral_target_linear(16.0) < 1.0);
    }

    #[test]
    fn node_colors_cover_the_cone_and_the_neutral_row_is_grey() {
        let dims = [36, 16, 16];
        let grey = node_color(dims, 1, 5, 0, 8);
        assert!((grey[0] - grey[1]).abs() < 1e-12 && (grey[1] - grey[2]).abs() < 1e-12);
        // encoded value 8/15 of sRGB
        assert!((grey[0] - srgb_decode(8.0 / 15.0)).abs() < 1e-12);
        let red = node_color(dims, 1, 0, 15, 15);
        assert_eq!(red, [1.0, 0.0, 0.0]);
        let green = node_color(dims, 1, 12, 15, 15);
        assert!((green[1] - 1.0).abs() < 1e-12 && green[0].abs() < 1e-12);
    }

    #[test]
    fn the_node_exact_table_is_valid_and_keeps_the_zero_saturation_row_neutral() {
        // an identity tone curve in the renderer's domain is enough to run the solve
        let cfg = FitConfig {
            dims: [12, 8, 6],
            ..FitConfig::default()
        };
        let tone = ToneTable {
            samples: (0..=TONE_GRID)
                .map(|i| neutral_target_linear(i as f64 / TONE_GRID as f64 * cfg.x_min))
                .collect(),
        };
        let (table, report) = solve_look(&cfg, &tone);
        assert_eq!(table.dims, [12, 8, 6]);
        assert_eq!(table.data.len(), 12 * 8 * 6 * 3);
        table.validate().unwrap();
        assert_eq!(report.entries, 12 * 8 * 6);
        assert!(report.solved > report.entries / 2, "{report:?}");
        for (k, e) in table.data.chunks_exact(3).enumerate() {
            if k % 8 == 0 {
                assert_eq!(e, [0.0, 1.0, 1.0], "the zero-saturation entry {k}");
            }
            assert!(e.iter().all(|v| v.is_finite()));
        }
        // the cine rendering changes chroma and the tone curve darkens: the table cannot be the identity
        assert!(
            table
                .data
                .iter()
                .zip(LookTable::identity().data.iter().cycle())
                .any(|(a, b)| (a - b).abs() > 0.01)
        );
    }

    #[test]
    fn a_fit_configuration_is_checked_before_anything_is_rendered() {
        assert!(FitConfig::default().validate().is_ok());
        let bad = |f: &dyn Fn(&mut FitConfig)| {
            let mut c = FitConfig::default();
            f(&mut c);
            c.validate().is_err()
        };
        assert!(bad(&|c| c.x_min = 0.0));
        assert!(bad(&|c| c.x_min = f64::NAN));
        assert!(bad(&|c| c.x_min = 1.0e9));
        assert!(bad(&|c| c.tone_points = 2));
        assert!(bad(&|c| c.dims = [0, 16, 16]));
        assert!(bad(&|c| c.dims = [36, 1, 16]));
        assert!(bad(&|c| c.dims = [36, 16, 1]));
        assert!(bad(&|c| c.dims = [37, 16, 16]));
        assert!(bad(&|c| c.dims = [36, 33, 16]));
        assert!(bad(&|c| c.dims = [36, 16, 17]));
    }

    #[test]
    fn rounding_keeps_a_table_valid() {
        let mut t = LookTable {
            dims: [4, 3, 2],
            data: [0.123_456_78_f32, 1.234_567_8, 0.987_654_3].repeat(24),
            encoding: 1,
        };
        for k in 0..24 {
            if k % 3 == 0 {
                t.data[k * 3 + 2] = 1.0;
            }
        }
        round_table(&mut t);
        t.validate().unwrap();
        assert_eq!(t.data[0], 0.123);
        assert_eq!(t.data[1], 1.23457);
        let mut c = vec![[0.123_456_79_f32, 0.987_654_3_f32]];
        round_curve(&mut c);
        assert_eq!(c[0], [0.123_456_8, 0.987_654]);
    }

    #[test]
    fn statistics_of_an_empty_set_are_zero() {
        assert_eq!(Stats::of(&[]).count, 0);
        let s = Stats::of(&[1.0, 2.0, 3.0, 10.0]);
        assert_eq!((s.count, s.max), (4, 10.0));
        assert!((s.median - 2.5).abs() < 1e-12);
    }
}
