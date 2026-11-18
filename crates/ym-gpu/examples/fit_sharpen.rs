//! The fit harness of spec 04, output sharpening (`docs/29-raw-constants.md` section 2.3, task RC-05):
//! evaluates a `Tuning` of the screen preset on Kodak and prints one JSON line per evaluation.
//! Measurement only: nothing here changes a default and nothing is written to disk.
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example fit_sharpen \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' -- \
//!     [--from oracle|default] [--set name=value]... [--rel name=0.9,1.1]... \
//!     [--eval "label@name=v,name=v"]... [--held] [--edge] [--list]
//! ```
//!
//! Arguments are those of `fit_demosaic`; the fields are those of `output_sharpen::Tuning`, the preset
//! fields included (`screen_radius_px`, `screen_amount_pct`, `screen_halo_pct`, `screen_threshold_levels`).
//! The base is the starting `Tuning`.
//!
//! The scenes of section 2.2: the screen preset (method 1) on the 24 Kodak images in linear light at scale 1
//! and at scale 0.33 (a typical downsized export), 48 outputs per evaluation; the change against the base is
//! the mean and the largest `|difference|` in 8-bit sRGB codes, 4 px border off, per scale. `--held` adds the 18
//! McMaster images (held out) as `held_change_codes`.
//!
//! `--edge` adds the edge profile of spec 04 section 9 on a synthetic soft edge (ours) at scales 1.0 and 0.4:
//! the 10 to 90 % rise, the overshoot and undershoot, and the noise gain of a flat patch. PSNR is not
//! an objective of a sharpening (`docs/29` G6, F-05): these are, with the change in codes.
//!
//! `--gate9` adds the numbers of the gate itself, the edge profile of spec 04 section 9 as `cpu_output_sharpen.rs`
//! measures it (a soft slanted edge of Gaussian sigma 0.8 px, 0.1 to 0.5, 5 degrees; the noisy flat patch; the
//! hard step at amount 300 and R 1.5) at scales 1.0 and 0.4, for the screen preset, with the halo-0 and print
//! rows at full size for the record (`docs/29` RC-09).
//!
//! Kodak and McMaster are read from `testdata/demosaic` (`YM_TESTDATA` moves it), are for evaluation and
//! local fitting, and are never committed; no picture made from them is written.

use serde_json::{Value, json};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::{Method, OutputSharpen, Tuning};

#[path = "../tests/common/demosaic_metrics.rs"]
mod demosaic_metrics;
#[path = "../tests/common/fit.rs"]
mod fit;

use demosaic_metrics::{demosaic_sets_root, linearize, load_set};
use fit::{Common, Rng, Start, code_diff, data_note, emit, num};

/// The screen preset of `tuning`: `OutputSharpen::screen()` reads `Tuning::DEFAULT`, this reads the one
/// under test (the same fields, in the same way).
pub fn screen_with(t: &Tuning) -> OutputSharpen {
    OutputSharpen {
        amount_pct: t.screen_amount_pct,
        radius_px: t.screen_radius_px,
        threshold_levels: t.screen_threshold_levels,
        halo_pct: t.screen_halo_pct,
        print_distance: t.print_distance,
        halo_tie_tolerance: t.halo_tie_tolerance,
        tuning: *t,
        ..OutputSharpen::screen()
    }
}

const SCALES: [f32; 2] = [1.0, 0.33];

/// The outputs of every image at every scale, scale-major.
pub fn run(images: &[RgbImage], t: &Tuning) -> Vec<RgbImage> {
    let o = screen_with(t);
    let mut v = Vec::new();
    for s in SCALES {
        for img in images {
            v.push(o.apply(img, s).expect("sharpen"));
        }
    }
    v
}

fn change(n: usize, outs: &[RgbImage], base: &[RgbImage]) -> Value {
    let mut m = serde_json::Map::new();
    for (k, s) in SCALES.iter().enumerate() {
        let (mut sm, mut mx) = (0.0, 0.0f64);
        for j in 0..n {
            let (mean_d, max_d) = code_diff(&outs[k * n + j], &base[k * n + j], 4, 1.0);
            sm += mean_d;
            mx = mx.max(max_d);
        }
        m.insert(
            format!("{s}"),
            json!({"mean_codes": sm / n as f64, "max_codes": mx}),
        );
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------------------------
// The edge profile of spec 04 section 9
// ---------------------------------------------------------------------------------------------

const EDGE_W: usize = 192;
const EDGE_H: usize = 64;
const EDGE_AT: f64 = 96.0;
const EDGE_LO: f64 = 0.1;
const EDGE_HI: f64 = 0.5;
const EDGE_NOISE: f64 = 0.004;
const EDGE_SCALES: [f32; 2] = [1.0, 0.4];

/// A soft vertical edge (a logistic of scale 0.6 px, about 1.3 px for 10 to 90 %) with seeded Gaussian noise.
pub fn edge_image() -> RgbImage {
    let mut img = RgbImage::new(EDGE_W, EDGE_H);
    let mut rng = Rng(7);
    for c in 0..3 {
        let plane = img.plane_mut(c);
        for y in 0..EDGE_H {
            for x in 0..EDGE_W {
                let step = 1.0 / (1.0 + (-(x as f64 + 0.5 - EDGE_AT) / 0.6).exp());
                plane[y * EDGE_W + x] =
                    (EDGE_LO + (EDGE_HI - EDGE_LO) * step + EDGE_NOISE * rng.gauss()) as f32;
            }
        }
    }
    img
}

/// Rise (10 to 90 %), overshoot and undershoot (percent of the step) of the row-averaged profile of the
/// green plane, and the standard deviation of the flat patch `x` in `8..56`, all rows.
pub fn edge_measure(img: &RgbImage) -> (f64, f64, f64, f64) {
    let g = img.plane(1);
    let profile: Vec<f64> = (0..EDGE_W)
        .map(|x| {
            (0..EDGE_H)
                .map(|y| f64::from(g[y * EDGE_W + x]))
                .sum::<f64>()
                / EDGE_H as f64
        })
        .collect();
    let step = EDGE_HI - EDGE_LO;
    let cross = |level: f64| -> f64 {
        let at = (EDGE_AT as usize - 8..EDGE_AT as usize + 8)
            .find(|&x| profile[x] <= level && profile[x + 1] > level)
            .unwrap_or(EDGE_AT as usize);
        at as f64 + (level - profile[at]) / (profile[at + 1] - profile[at])
    };
    let rise = cross(EDGE_LO + 0.9 * step) - cross(EDGE_LO + 0.1 * step);
    let over = (profile.iter().copied().fold(f64::MIN, f64::max) - EDGE_HI) / step * 100.0;
    let under = (EDGE_LO - profile.iter().copied().fold(f64::MAX, f64::min)) / step * 100.0;
    let flat: Vec<f64> = (0..EDGE_H)
        .flat_map(|y| (8..56).map(move |x| (y, x)))
        .map(|(y, x)| f64::from(g[y * EDGE_W + x]))
        .collect();
    let m = flat.iter().sum::<f64>() / flat.len() as f64;
    let sd = (flat.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / flat.len() as f64).sqrt();
    (rise, over, under, sd)
}

pub fn edge_numbers(edge: &RgbImage, t: &Tuning) -> Value {
    let o = screen_with(t);
    let (_, _, _, sd_in) = edge_measure(edge);
    let mut m = serde_json::Map::new();
    for s in EDGE_SCALES {
        let out = o.apply(edge, s).expect("sharpen");
        let (rise, over, under, sd) = edge_measure(&out);
        m.insert(
            format!("{s}"),
            json!({
                "rise_10_90_px": num(rise), "overshoot_pct": num(over),
                "undershoot_pct": num(under), "noise_gain": num(sd / sd_in),
            }),
        );
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------------------------
// The gate: the edge profile of spec 04 section 9, as the tests of `cpu_output_sharpen.rs` measure it
// ---------------------------------------------------------------------------------------------

/// The slanted edge of section 9: `lo` to `hi` through a Gaussian of `sigma` px, `angle_deg` from vertical,
/// each pixel the mean of 4 x 4 samples of the analytic profile.
pub fn slanted_edge(size: usize, angle_deg: f64, sigma: f64, lo: f32, hi: f32) -> RgbImage {
    const SS: usize = 4;
    let a = angle_deg.to_radians();
    let c = size as f64 / 2.0;
    let mut img = RgbImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let px = x as f64 + (sx as f64 + 0.5) / SS as f64;
                    let py = y as f64 + (sy as f64 + 0.5) / SS as f64;
                    let d = (px - c) * a.cos() - (py - c) * a.sin();
                    acc += 0.5 * (1.0 + libm::erf(d / (sigma * std::f64::consts::SQRT_2)));
                }
            }
            let t = acc / (SS * SS) as f64;
            let v = (f64::from(lo) + f64::from(hi - lo) * t) as f32;
            img.set_pixel(x, y, [v, v, v]);
        }
    }
    img
}

/// The edge-spread function of the middle rows, binned by the distance from the edge line (0.1 px bins).
fn esf(img: &RgbImage, angle_deg: f64) -> Vec<f64> {
    let a = angle_deg.to_radians();
    let size = img.width;
    let c = size as f64 / 2.0;
    let (y0, y1) = (size * 5 / 16, size * 11 / 16);
    let mut bins: Vec<(f64, usize)> = vec![(0.0, 0); 160];
    for y in y0..y1 {
        for x in 0..size {
            let d = (x as f64 + 0.5 - c) * a.cos() - (y as f64 + 0.5 - c) * a.sin();
            let b = ((d + 8.0) / 0.1).floor();
            if (0.0..160.0).contains(&b) {
                bins[b as usize].0 += f64::from(img.pixel(x, y)[1]);
                bins[b as usize].1 += 1;
            }
        }
    }
    bins.iter()
        .filter(|(_, n)| *n > 0)
        .map(|(s, n)| s / *n as f64)
        .collect()
}

/// The 10 to 90 % rise along row `y`, linear interpolation.
fn row_rise_10_90(img: &RgbImage, y: usize, lo: f64, hi: f64) -> f64 {
    let v: Vec<f64> = (0..img.width)
        .map(|x| f64::from(img.pixel(x, y)[1]))
        .collect();
    let cross = |level: f64| {
        (0..v.len() - 1)
            .find(|&x| (v[x] - level) * (v[x + 1] - level) <= 0.0 && v[x] != v[x + 1])
            .map(|x| x as f64 + (level - v[x]) / (v[x + 1] - v[x]))
            .unwrap_or(f64::NAN)
    };
    cross(lo + 0.9 * (hi - lo)) - cross(lo + 0.1 * (hi - lo))
}

/// The reference's generator of the section 9 noise test: splitmix64 with Box-Muller.
struct BoxMuller(u64);

impl BoxMuller {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(11_400_714_819_323_198_485);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(13_787_848_793_156_543_929);
        z = (z ^ (z >> 27)).wrapping_mul(10_723_151_780_598_845_931);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = ((self.next() >> 11) as f64 + 1.0) * (1.0 / 9_007_199_254_740_992.0);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn std_dev_centre(img: &RgbImage, margin: usize) -> f64 {
    let mut vals = Vec::new();
    for y in margin..img.height - margin {
        for x in margin..img.width - margin {
            vals.push(f64::from(img.pixel(x, y)[1]));
        }
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// The profile of one parameter set at scale `s`: rise, overshoot and undershoot of the soft edge.
fn gate9_profile(p: &OutputSharpen, s: f32, edge: &RgbImage) -> Value {
    let (flo, fhi) = (0.1f64, 0.5f64);
    let out = p.apply(edge, s).expect("sharpen");
    let rise = row_rise_10_90(&out, 64, flo, fhi);
    let e = esf(&out, 5.0);
    let max = e.iter().copied().fold(f64::MIN, f64::max);
    let min = e.iter().copied().fold(f64::MAX, f64::min);
    json!({
        "rise_10_90_px": num(rise),
        "overshoot_pct": num(100.0 * (max - fhi).max(0.0) / (fhi - flo)),
        "undershoot_pct": num(100.0 * (flo - min).max(0.0) / (fhi - flo)),
    })
}

/// The edge profile of spec 04 section 9 for `t`: per scale (1.0 and 0.4) the screen preset's rise and
/// over- and undershoot, the noise gain of a flat patch (sigma 0.004; threshold 1 level and 0), and the halo
/// suppression on the hard 0.1 to 0.5 step at amount 300 and R 1.5 (the overshoot and undershoot, as a
/// fraction of the unit signal, without and with halo 100); at full size also the halo-0 and print rows.
pub fn gate9_numbers(t: &Tuning) -> Value {
    gate9_numbers_at(t, &EDGE_SCALES)
}

/// [`gate9_numbers`] at other output scales (`--gate9-scales`): the keys of the result are the scales.
pub fn gate9_numbers_at(t: &Tuning, scales: &[f32]) -> Value {
    let edge = slanted_edge(128, 5.0, 0.8, 0.1, 0.5);
    let rise_before = row_rise_10_90(&edge, 64, 0.1, 0.5);
    let mut patch = RgbImage::new(96, 96);
    for v in &mut patch.data {
        *v = 0.18;
    }
    let mut rng = BoxMuller(1234);
    for v in &mut patch.data {
        *v = (f64::from(*v) + 0.004 * rng.gauss()) as f32;
    }
    let sd_in = std_dev_centre(&patch, 24);
    let mut step = RgbImage::new(32, 8);
    for y in 0..8 {
        for x in 0..32 {
            let v = if x >= 16 { 0.5 } else { 0.1 };
            step.set_pixel(x, y, [v, v, v]);
        }
    }
    let mut m = serde_json::Map::new();
    m.insert("rise_before_px".into(), num(rise_before));
    for &s in scales {
        let screen = screen_with(t);
        let mut v = gate9_profile(&screen, s, &edge);
        for (key, threshold) in [("noise_gain", 1.0f32), ("noise_gain_threshold_0", 0.0)] {
            let mut p = screen_with(t);
            p.threshold_levels = threshold;
            let after = std_dev_centre(&p.apply(&patch, s).expect("sharpen"), 24);
            v[key] = num(after / sd_in);
        }
        let mut hp = screen_with(t);
        hp.method = Method::HardThreshold;
        hp.threshold_levels = 0.0;
        hp.amount_pct = 300.0;
        hp.radius_px = 1.5;
        for (key, halo) in [("halo0", 0.0f32), ("halo100", 100.0)] {
            hp.halo_pct = halo;
            let o = hp.apply(&step, s).expect("sharpen");
            let (mut over, mut under) = (0.0f64, 0.0f64);
            for x in 0..32 {
                let g = f64::from(o.pixel(x, 4)[1]);
                over = over.max(g - 0.5);
                under = under.max(0.1 - g);
            }
            v[format!("step_overshoot_{key}")] = num(over);
            v[format!("step_undershoot_{key}")] = num(under);
        }
        if s == 1.0 {
            let mut no_halo = screen_with(t);
            no_halo.halo_pct = 0.0;
            v["screen_halo0"] = gate9_profile(&no_halo, s, &edge);
            let print = OutputSharpen {
                tuning: *t,
                ..OutputSharpen::print()
            };
            v["print"] = gate9_profile(&print, s, &edge);
        }
        m.insert(format!("{s}"), v);
    }
    Value::Object(m)
}

fn load_images(dir: &str) -> Vec<RgbImage> {
    load_set(&demosaic_sets_root().join(dir))
        .iter()
        .map(|(_, i)| linearize(i))
        .collect()
}

fn usage(why: &str) -> ! {
    eprintln!("fit_sharpen: {why}");
    eprintln!(
        "usage: fit_sharpen [--from oracle|default] [--set name=value]... [--rel name=f1,f2]... \
         [--eval \"label@name=v,name=v\"]... [--held] [--edge] [--gate9] [--gate9-scales s1,s2,...] [--list]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut common = Common::default();
    let mut want_edge = false;
    let mut want_gate9 = false;
    let mut gate9_scales: Vec<f32> = EDGE_SCALES.to_vec();
    let mut i = 0;
    while i < args.len() {
        match common.take(&args, &mut i) {
            Ok(true) => {}
            Ok(false) => match args[i].as_str() {
                "--edge" => want_edge = true,
                "--gate9" => want_gate9 = true,
                "--gate9-scales" => {
                    i += 1;
                    let list = args
                        .get(i)
                        .unwrap_or_else(|| usage("--gate9-scales needs a list"));
                    gate9_scales = list
                        .split(',')
                        .map(|v| {
                            v.parse::<f32>()
                                .ok()
                                .filter(|s| *s > 0.0 && *s <= 1.0)
                                .unwrap_or_else(|| {
                                    usage(&format!("`{v}` is not a scale in (0, 1]"))
                                })
                        })
                        .collect();
                    want_gate9 = true;
                }
                other => usage(&format!("unknown argument `{other}`")),
            },
            Err(e) => usage(&e),
        }
        i += 1;
    }
    let start = match common.start {
        Start::Oracle => Tuning::ORACLE,
        Start::Default => Tuning::DEFAULT,
    };
    if common.list {
        emit(&fit::list_line("output_sharpen", &start));
        return;
    }
    let evals = common.evaluations(&start).unwrap_or_else(|e| usage(&e));

    let kodak = load_images("kodak");
    let mcmaster = if common.held {
        load_images("mcmaster")
    } else {
        Vec::new()
    };
    let mut used = Vec::new();
    let mut skipped = Vec::new();
    if kodak.is_empty() {
        skipped.push((
            "kodak".to_owned(),
            format!(
                "no images in {} (run tools/fetch/demosaic-sets.sh)",
                demosaic_sets_root().join("kodak").display()
            ),
        ));
    } else {
        used.push(("kodak".to_owned(), kodak.len()));
    }
    if common.held {
        if mcmaster.is_empty() {
            skipped.push((
                "mcmaster".to_owned(),
                "held-out set missing (tools/fetch/demosaic-sets.sh)".to_owned(),
            ));
        } else {
            used.push(("mcmaster (held out)".to_owned(), mcmaster.len()));
        }
    }
    if want_edge {
        used.push(("synthetic soft edge (ours, seeded)".to_owned(), 1));
    }
    if want_gate9 {
        used.push((
            "spec 04 section 9 edge profile (ours, synthetic)".to_owned(),
            1,
        ));
    }
    emit(&data_note("fit_sharpen", &used, &skipped));

    let edge = want_edge.then(edge_image);
    let base = run(&kodak, &start);
    let base_held = run(&mcmaster, &start);
    let from = if common.start == Start::Oracle {
        "oracle"
    } else {
        "default"
    };
    let line = |label: &str,
                set: Value,
                t: &Tuning,
                outs: &[RgbImage],
                outs_held: &[RgbImage],
                is_base: bool| {
        let mut v = json!({
            "kind": "eval", "example": "fit_sharpen", "label": label, "from": from, "set": set,
            "outputs": outs.len(),
        });
        if !is_base {
            v["change_codes"] = change(kodak.len(), outs, &base);
            if !outs_held.is_empty() {
                v["held_change_codes"] = change(mcmaster.len(), outs_held, &base_held);
            }
        }
        if let Some(e) = &edge {
            v["edge"] = edge_numbers(e, t);
        }
        if want_gate9 {
            v["gate9"] = gate9_numbers_at(t, &gate9_scales);
        }
        emit(&v);
    };
    line("base", json!({}), &start, &base, &base_held, true);
    for (e, t) in &evals {
        let outs = run(&kodak, t);
        let held = run(&mcmaster, t);
        let set: std::collections::BTreeMap<_, _> = e.sets.iter().cloned().collect();
        line(&e.label, json!(set), t, &outs, &held, false);
    }
}
