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
//! Kodak and McMaster are read from `testdata/demosaic` (`YM_TESTDATA` moves it), are for evaluation and
//! local fitting, and are never committed; no picture made from them is written.

use serde_json::{Value, json};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::{OutputSharpen, Tuning};

#[path = "../tests/common/demosaic_metrics.rs"]
#[allow(dead_code)]
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
         [--eval \"label@name=v,name=v\"]... [--held] [--edge] [--list]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut common = Common::default();
    let mut want_edge = false;
    let mut i = 0;
    while i < args.len() {
        match common.take(&args, &mut i) {
            Ok(true) => {}
            Ok(false) => match args[i].as_str() {
                "--edge" => want_edge = true,
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
