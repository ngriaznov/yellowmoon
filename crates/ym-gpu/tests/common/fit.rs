//! The shared part of the fit harness (`docs/29-raw-constants.md` section 5, task RC-05): the
//! command-line overrides of a `Tuning` by field name, the data note, the measures of change and the
//! seeded noise that the three examples `fit_demosaic`, `fit_diffraction` and `fit_sharpen` use.
//!
//! Included by the examples with `#[path = "../tests/common/fit.rs"]` (as `demosaic_metrics.rs` is) and
//! by the test `tests/fit_harness.rs`. Nothing here is part of a render.
//!
//! The harness evaluates a `Tuning` and prints numbers. It never edits a constant of the code, reads
//! no environment variable for a constant and ships no data: Kodak, McMaster and PixelShift200 are
//! read from `testdata/` (git-ignored, `tools/fetch/`) for evaluation and local fitting, and no
//! picture made from them is written anywhere.

#![allow(dead_code)]

use serde_json::{Value, json};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::demosaic::adaptive_ratio;
use ym_gpu::cpu::output_sharpen;
use ym_gpu::cpu::sharpen::diffraction;

// ---------------------------------------------------------------------------------------------
// Fields of a Tuning by name
// ---------------------------------------------------------------------------------------------

/// A number a `Tuning` field can hold, set from the `f64` of the command line.
pub trait FieldValue: Copy + std::fmt::Display {
    /// The value of `v` in this type, or why it does not fit.
    fn from_f64(v: f64) -> Result<Self, String>;
}

impl FieldValue for f64 {
    fn from_f64(v: f64) -> Result<f64, String> {
        Ok(v)
    }
}

impl FieldValue for f32 {
    fn from_f64(v: f64) -> Result<f32, String> {
        if v.abs() > f64::from(f32::MAX) {
            return Err(format!("{v} does not fit an f32"));
        }
        Ok(v as f32)
    }
}

macro_rules! integer_field {
    ($($t:ty),*) => {$(
        impl FieldValue for $t {
            fn from_f64(v: f64) -> Result<$t, String> {
                if v.fract() != 0.0 || v < 0.0 || v > <$t>::MAX as f64 {
                    return Err(format!("{v} is not a whole number in 0..={}", <$t>::MAX));
                }
                Ok(v as $t)
            }
        }
    )*};
}
integer_field!(u32, usize);

/// A `Tuning` whose fields can be read and set by name (the field names of the struct, which are the
/// names of `REGISTRY`).
pub trait Fields: Copy {
    /// Every field name, in the order of the struct.
    const NAMES: &'static [&'static str];
    /// The value of the field as the decimal text it prints with (`0.7` for an `f32` that holds 0.7),
    /// so that `0.9 x` of it is the number a person would type.
    fn get(&self, name: &str) -> Option<f64>;
    /// Sets the field `name` to `v`.
    fn set(&mut self, name: &str, v: f64) -> Result<(), String>;
}

macro_rules! tuning_fields {
    ($t:ty; $($name:ident),* $(,)?) => {
        impl Fields for $t {
            const NAMES: &'static [&'static str] = &[$(stringify!($name)),*];

            fn get(&self, name: &str) -> Option<f64> {
                match name {
                    $(stringify!($name) => format!("{}", self.$name).parse().ok(),)*
                    _ => None,
                }
            }

            fn set(&mut self, name: &str, v: f64) -> Result<(), String> {
                if !v.is_finite() {
                    return Err(format!("{name}: {v} is not a finite number"));
                }
                match name {
                    $(stringify!($name) => {
                        self.$name = FieldValue::from_f64(v).map_err(|e| format!("{name}: {e}"))?;
                        Ok(())
                    })*
                    _ => Err(unknown_field(name, Self::NAMES)),
                }
            }
        }
    };
}

/// The message for a field name no `Tuning` of the package has.
pub fn unknown_field(name: &str, names: &[&str]) -> String {
    format!(
        "unknown Tuning field `{name}`; the fields of this package are: {}",
        names.join(", ")
    )
}

tuning_fields!(adaptive_ratio::Tuning;
    second_diff_weight, ca_relax_slope, ca_relax_floor_px, centre_weight_linear,
    centre_weight_cubic, chroma_floor, sat_dominance, shadow_amount, shadow_laplacian_scale,
    shadow_fade_end, shadow_fade_span, match_excess_weight, offset_snr, snr_per_slider,
    match_below_ratio);

tuning_fields!(diffraction::Tuning;
    lambda_min_nm, lambda_max_nm, weight_flat_from_nm, weight_ramp_top_nm, weight_ramp_width_nm,
    step_fraction, airy_scale, r80_energy, r80_floor, support_r80_factor, support_pitch_factor,
    angular_per_px, angular_round, kernel_energy, kernel_tail, line_angles, x80_cutoff, x80_level,
    sigma_factor, sigma_min, sigma_max, gauss_radius_per_sigma, gauss_radius_add, clip_fraction,
    clip_weight_largest, clip_weight_middle, pull_gain, pull_band, ridge, gate, gate_fade_lo,
    gate_fade_hi, weight_min, cd_min, ratio_cap, divide_guard, moment_floor, floor_dn);

tuning_fields!(output_sharpen::Tuning;
    sigma_slope, sigma_offset, sigma_knee_radius_px, sigma_sqrt_origin, sigma_sqrt_scale,
    screen_radius_cap_px, taps_per_sigma, taps_offset, taps_min, max_taps_screen, max_taps_print,
    sigma_knee, screen_amount_pct, screen_radius_px, screen_threshold_levels, screen_halo_pct,
    halo_dead_zone, halo_pull_fade, halo_tie_tolerance, gain_sigma_multiple, gain_sigma_min,
    halo_sigma_multiple, ratio_max, luma_floor, gain_epsilon, scharr_centre, scharr_corner,
    ramp_width, print_distance_per_px, print_radius_cap_px, print_amount_pct, print_distance,
    auto_t_offset, auto_t_min, auto_r_offset, auto_r_min, auto_cut);

/// Which Tuning the overrides start from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// `Tuning::ORACLE`: the constants of the research package, the floor of every fit.
    Oracle,
    /// `Tuning::DEFAULT`: what ships.
    Default,
}

/// One evaluation: a label and the fields it moves.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// The name in the output.
    pub label: String,
    /// `(field, value)`, applied in order.
    pub sets: Vec<(String, f64)>,
}

impl Evaluation {
    /// `base` with the evaluation's fields set.
    ///
    /// # Errors
    ///
    /// An unknown field name or a value the field cannot hold.
    pub fn apply<T: Fields>(&self, base: &T) -> Result<T, String> {
        let mut t = *base;
        for (name, v) in &self.sets {
            t.set(name, *v)?;
        }
        Ok(t)
    }
}

/// `name=value`.
///
/// # Errors
///
/// Text that is not `name=value` with a number.
pub fn parse_assignment(text: &str) -> Result<(String, f64), String> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| format!("`{text}` is not name=value"))?;
    let v: f64 = value
        .trim()
        .parse()
        .map_err(|_| format!("`{}` is not a number (in `{text}`)", value.trim()))?;
    Ok((name.trim().to_owned(), v))
}

/// The number `v` rounded to 12 significant digits: `1.6 * 0.9` is `1.44`, not `1.4400000000000002`.
pub fn tidy(v: f64) -> f64 {
    format!("{v:.11e}").parse().unwrap_or(v)
}

/// The arguments every fit example takes. Anything else the example reads itself.
#[derive(Clone, Debug)]
pub struct Common {
    /// `--from oracle|default` (default `oracle`).
    pub start: Start,
    /// `--set name=value` (repeatable): one evaluation, labelled `set`, with every field moved.
    pub set: Vec<(String, f64)>,
    /// `--rel name=f1,f2,...`: one evaluation per factor, the field at `f` times its starting value.
    pub rel: Vec<(String, Vec<f64>)>,
    /// `--eval "label@name=v,name=v"`: one evaluation per argument.
    pub evals: Vec<Evaluation>,
    /// `--held`: also run the held-out scenes that cost extra time.
    pub held: bool,
    /// `--list`: print the field names and their starting values, then stop.
    pub list: bool,
}

impl Default for Common {
    fn default() -> Common {
        Common {
            start: Start::Oracle,
            set: Vec::new(),
            rel: Vec::new(),
            evals: Vec::new(),
            held: false,
            list: false,
        }
    }
}

impl Common {
    /// Reads the argument at `args[*i]` if it is one of the common flags, moving `*i` past what it
    /// took. `Ok(false)` leaves it to the example.
    ///
    /// # Errors
    ///
    /// A common flag with a missing or malformed value.
    pub fn take(&mut self, args: &[String], i: &mut usize) -> Result<bool, String> {
        let flag = args[*i].as_str();
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--from" => {
                self.start = match value(i)?.as_str() {
                    "oracle" => Start::Oracle,
                    "default" => Start::Default,
                    other => return Err(format!("--from takes oracle or default, not `{other}`")),
                };
            }
            "--set" => self.set.push(parse_assignment(&value(i)?)?),
            "--rel" => {
                let text = value(i)?;
                let (name, list) = text
                    .split_once('=')
                    .ok_or_else(|| format!("--rel needs name=f1,f2 (got `{text}`)"))?;
                let factors = list
                    .split(',')
                    .map(|f| {
                        f.trim()
                            .parse::<f64>()
                            .map_err(|_| format!("`{f}` is not a number (in `{text}`)"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.rel.push((name.trim().to_owned(), factors));
            }
            "--eval" => {
                let text = value(i)?;
                let (label, list) = text
                    .split_once('@')
                    .ok_or_else(|| format!("--eval needs label@name=v,name=v (got `{text}`)"))?;
                let sets = list
                    .split(',')
                    .map(parse_assignment)
                    .collect::<Result<Vec<_>, _>>()?;
                self.evals.push(Evaluation {
                    label: label.trim().to_owned(),
                    sets,
                });
            }
            "--held" => self.held = true,
            "--list" => self.list = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The evaluations to run, in the order `--rel`, `--eval`, `--set`, each checked against the
    /// fields of `T` before anything runs. `base` is the starting Tuning (for `--rel`).
    ///
    /// # Errors
    ///
    /// An unknown field name or a value that does not fit its field.
    pub fn evaluations<T: Fields>(&self, base: &T) -> Result<Vec<(Evaluation, T)>, String> {
        let mut all = Vec::new();
        for (name, factors) in &self.rel {
            let from = base
                .get(name)
                .ok_or_else(|| unknown_field(name, T::NAMES))?;
            for f in factors {
                all.push(Evaluation {
                    label: format!("{name} x{f}"),
                    sets: vec![(name.clone(), tidy(from * f))],
                });
            }
        }
        all.extend(self.evals.iter().cloned());
        if !self.set.is_empty() {
            all.push(Evaluation {
                label: "set".to_owned(),
                sets: self.set.clone(),
            });
        }
        all.into_iter()
            .map(|e| e.apply(base).map(|t| (e, t)))
            .collect()
    }
}

/// The one JSON line of the field list of `T` at `base` (`--list`).
pub fn list_line<T: Fields>(package: &str, base: &T) -> Value {
    json!({
        "kind": "fields",
        "package": package,
        "fields": T::NAMES.iter().map(|n| (*n, base.get(n))).collect::<std::collections::BTreeMap<_, _>>(),
    })
}

/// A JSON line to stdout.
pub fn emit(v: &Value) {
    println!("{v}");
}

// ---------------------------------------------------------------------------------------------
// The data note
// ---------------------------------------------------------------------------------------------

/// What every fit report says about its data (`docs/29` F-07, G5): the sets that were used and the sets
/// that were skipped, and that nothing of them is committed.
pub fn data_note(example: &str, used: &[(String, usize)], skipped: &[(String, String)]) -> Value {
    json!({
        "kind": "data_note",
        "example": example,
        "used": used.iter().map(|(n, k)| json!({"set": n, "images": k})).collect::<Vec<_>>(),
        "skipped": skipped.iter().map(|(n, why)| json!({"set": n, "why": why})).collect::<Vec<_>>(),
        "note": "Evaluation and local fitting only. Kodak, McMaster, PixelShift200 and any photograph are \
                 read from testdata/ (git-ignored) and are never committed, and no picture made from them is \
                 written. The test images are re-mosaicked 8-bit sRGB pictures and the noise models are \
                 stand-ins: no real-sensor ground truth (docs/29 G5) unless PixelShift200 is listed as used.",
    })
}

// ---------------------------------------------------------------------------------------------
// Measures
// ---------------------------------------------------------------------------------------------

/// The sRGB code (0 to 255, not rounded) of a linear value.
pub fn oetf(v: f32) -> f64 {
    let v = f64::from(v).clamp(0.0, 1.0);
    255.0
        * if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
}

/// Mean and largest `|code difference|` between two images in 8-bit sRGB codes, `border` pixels off
/// every side, values divided by `scale` first (the audit's measure of change, `docs/29` section 2.2).
pub fn code_diff(a: &RgbImage, b: &RgbImage, border: usize, scale: f64) -> (f64, f64) {
    let (w, h) = (a.width, a.height);
    let (mut s, mut n, mut mx) = (0.0, 0u64, 0.0f64);
    for c in 0..3 {
        let (pa, pb) = (a.plane(c), b.plane(c));
        for y in border..h - border {
            for x in border..w - border {
                let d = (oetf((f64::from(pa[y * w + x]) / scale) as f32)
                    - oetf((f64::from(pb[y * w + x]) / scale) as f32))
                .abs();
                s += d;
                n += 1;
                mx = mx.max(d);
            }
        }
    }
    (s / n as f64, mx)
}

/// splitmix64 with Irwin-Hall normals (the reference's generator, as in `capture_sharpen_ab`).
pub struct Rng(pub u64);

impl Rng {
    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(11_400_714_819_323_198_485);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(13_787_848_793_156_543_929);
        z = (z ^ (z >> 27)).wrapping_mul(10_723_151_780_598_845_931);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Approximately normal (the sum of twelve uniforms, minus six).
    pub fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// The mean of `v`, `NaN` when it is empty.
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// A JSON number that is `null` for a `NaN` (serde_json refuses to write one).
pub fn num(v: f64) -> Value {
    if v.is_finite() { json!(v) } else { Value::Null }
}
