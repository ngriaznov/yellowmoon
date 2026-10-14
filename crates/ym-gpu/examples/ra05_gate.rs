//! RA-05 demosaic gate (`docs/algorithms/TASKS.md` RA-05, `docs/05` R6): PSNR, zipper and
//! false-colour numbers for Bilinear, Menon, LMMSE, AdaptiveRatio and LogGuided, and the R6 rule's
//! verdict. Measurement only: nothing here changes `DEFAULT_METHOD` (the owner chose `AdaptiveRatio`
//! after the blind look, RA-H4).
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example ra05_gate \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' [-- psnr|metrics|noisy|all] [--out DIR]
//! ```
//!
//! The output folder (default `testdata/ra05`, git-ignored) gets `numbers.md` (every table that is
//! printed) and `gate.json` (the same numbers). The sets: the procedural set of
//! `tests/cpu_demosaic.rs`, Kodak and McMaster from `testdata/demosaic` (`tools/fetch/demosaic-sets.sh`),
//! and, if they are ever fetchable, Gharbi hard/moire and PixelShift200 (a folder per set next to
//! kodak and mcmaster: `gharbi_hard`, `gharbi_moire`, `pixelshift200`; they are read when present).
//!
//! The protocol, the noise models and the definitions of zipper and false colour are in
//! `tests/common/ra05_metrics.rs` (shared with the unit tests in `tests/cpu_demosaic_metrics.rs`).
//!
//! * `psnr`: linear light (sRGB decoded) at signal scales 1, 0.1 and 0.01 with a 10 px border
//!   excluded, mean of per-image dB; the gamma-encoded values at scale 1 for continuity with the
//!   older numbers of `docs/05`.
//! * `metrics`: zipper rate, false-colour rate and mean chroma difference at scale 1.
//! * `noisy`: spec 01 section 6.3's noisy-input check, Poisson-Gaussian-like noise at four
//!   ISO-like levels (100, 800, 1600, 6400) with the matching noise model, PSNR after the demosaic and after the R6
//!   bilateral denoise.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::{Cfa, NoiseDn, NoiseSigma, RgbImage, synth};
use ym_noise::model::NoiseModel;

#[path = "../tests/common/ra05_metrics.rs"]
mod ra05_metrics;

use ra05_metrics::{
    Artifacts, BORDER_PX, LabImage, PRIMARY, SCALES, SENSITIVITY, Variant, artifacts,
    demosaic_sets_root, linearize, load_set, mean_finite, psnr_row, run_variant, synthetic_set,
};

/// Prints every line and keeps it for `numbers.md`.
#[derive(Default)]
struct Report {
    text: String,
}

impl Report {
    fn line(&mut self, s: impl AsRef<str>) {
        println!("{}", s.as_ref());
        self.text.push_str(s.as_ref());
        self.text.push('\n');
    }
}

/// A set of ground-truth images, in linear light unless the protocol says otherwise.
struct Set {
    name: String,
    images: Vec<(String, RgbImage)>,
    /// Per-image rows are printed for small sets.
    per_image: bool,
}

/// All columns of a PSNR run: the primary variants, then the sensitivity variants.
fn all_variants() -> Vec<Variant> {
    PRIMARY.iter().chain(SENSITIVITY.iter()).copied().collect()
}

/// `| a | b |` row.
fn md_row(cells: &[String]) -> String {
    format!("| {} |", cells.join(" | "))
}

fn md_table(report: &mut Report, header: &[String], rows: &[Vec<String>]) {
    report.line(md_row(header));
    report.line(format!("|{}|", vec!["---"; header.len()].join("|")));
    for r in rows {
        report.line(md_row(r));
    }
    report.line("");
}

fn labels(variants: &[Variant]) -> Vec<String> {
    variants.iter().map(|v| v.label.to_owned()).collect()
}

/// The mean PSNR of one set at one scale: one value per column of [`all_variants`].
struct Cell {
    set: String,
    protocol: &'static str,
    scale: f32,
    mean_db: Vec<f64>,
    /// `(image, dB per column)` of every image (printed only for the small sets).
    per_image: Vec<(String, Vec<f64>)>,
}

fn psnr_cells(report: &mut Report, set: &Set, protocol: &'static str, scales: &[f32]) -> Vec<Cell> {
    let variants = all_variants();
    let mut cells = Vec::new();
    for &scale in scales {
        let mut mean = vec![0.0; variants.len()];
        let mut per_image = Vec::new();
        for (name, img) in &set.images {
            let row = psnr_row(img, scale, &variants);
            for (m, v) in mean.iter_mut().zip(&row) {
                *m += v / set.images.len() as f64;
            }
            per_image.push((name.clone(), row));
        }
        if set.per_image {
            report.line(format!(
                "**{} ({protocol}), signal scale {scale}: per image (dB)**\n",
                set.name
            ));
            let mut header = vec!["image".to_owned()];
            header.extend(labels(&variants[..PRIMARY.len()]));
            let mut rows: Vec<Vec<String>> = per_image
                .iter()
                .map(|(n, r)| {
                    let mut line = vec![n.clone()];
                    line.extend(r[..PRIMARY.len()].iter().map(|v| format!("{v:.2}")));
                    line
                })
                .collect();
            let mut mean_line = vec!["**mean**".to_owned()];
            mean_line.extend(mean[..PRIMARY.len()].iter().map(|v| format!("**{v:.2}**")));
            rows.push(mean_line);
            md_table(report, &header, &rows);
        }
        cells.push(Cell {
            set: set.name.clone(),
            protocol,
            scale,
            mean_db: mean,
            per_image,
        });
    }
    cells
}

/// The PSNR tables of one set: one row per signal scale, primary columns, then the sensitivity
/// columns.
fn print_set_table(report: &mut Report, title: &str, cells: &[Cell]) {
    let variants = all_variants();
    report.line(format!("**{title}: mean PSNR (dB)**\n"));
    let mut header = vec!["scale".to_owned()];
    header.extend(labels(&variants[..PRIMARY.len()]));
    let rows: Vec<Vec<String>> = cells
        .iter()
        .map(|c| {
            let mut line = vec![format!("{}", c.scale)];
            line.extend(c.mean_db[..PRIMARY.len()].iter().map(|v| format!("{v:.2}")));
            line
        })
        .collect();
    md_table(report, &header, &rows);
    report.line(format!("*{title}: sensitivity to the noise model (dB)*\n"));
    let mut header = vec!["scale".to_owned()];
    header.extend(labels(&variants[PRIMARY.len()..]));
    let rows: Vec<Vec<String>> = cells
        .iter()
        .map(|c| {
            let mut line = vec![format!("{}", c.scale)];
            line.extend(c.mean_db[PRIMARY.len()..].iter().map(|v| format!("{v:.2}")));
            line
        })
        .collect();
    md_table(report, &header, &rows);
}

fn column(label: &str) -> usize {
    PRIMARY
        .iter()
        .position(|v| v.label == label)
        .expect("a primary column")
}

/// The R6 rule of `docs/05` ("the method with the best PSNR becomes the default") applied to the
/// linear-light cells: the winner of every cell and the margins, then the aggregate.
fn decide(report: &mut Report, cells: &[&Cell]) -> Value {
    let menon = column("menon");
    let ar = column("adaptive_ratio");
    let lmmse = column("lmmse");
    let lg = column("log_guided");
    // The four methods of the classical R6 rule; log_guided is a preview method.
    let rule: Vec<usize> = ["bilinear", "menon", "lmmse", "adaptive_ratio"]
        .iter()
        .map(|l| column(l))
        .collect();
    report.line("**The R6 rule per cell (best PSNR; the four R6 methods, log_guided is a preview method)**\n");
    let header: Vec<String> = [
        "set",
        "scale",
        "winner",
        "wins by (dB)",
        "adaptive_ratio - menon",
        "adaptive_ratio - lmmse",
        "log_guided - menon",
        "images: adaptive_ratio beats menon",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    let mut rows = Vec::new();
    let mut wins = vec![0usize; PRIMARY.len()];
    let mut json_cells = Vec::new();
    for c in cells {
        let mut order = rule.clone();
        order.sort_by(|&a, &b| c.mean_db[b].total_cmp(&c.mean_db[a]));
        let (best, second) = (order[0], order[1]);
        wins[best] += 1;
        let margin = c.mean_db[best] - c.mean_db[second];
        let beats = c.per_image.iter().filter(|(_, r)| r[ar] > r[menon]).count();
        rows.push(vec![
            c.set.clone(),
            format!("{}", c.scale),
            PRIMARY[best].label.to_owned(),
            format!("{margin:.2}"),
            format!("{:+.2}", c.mean_db[ar] - c.mean_db[menon]),
            format!("{:+.2}", c.mean_db[ar] - c.mean_db[lmmse]),
            format!("{:+.2}", c.mean_db[lg] - c.mean_db[menon]),
            format!("{beats} of {}", c.per_image.len()),
        ]);
        json_cells.push(json!({
            "images_adaptive_ratio_beats_menon": beats, "images": c.per_image.len(),
            "set": c.set, "scale": c.scale, "winner": PRIMARY[best].label, "margin_db": margin,
            "adaptive_ratio_minus_menon_db": c.mean_db[ar] - c.mean_db[menon],
            "adaptive_ratio_minus_lmmse_db": c.mean_db[ar] - c.mean_db[lmmse],
            "log_guided_minus_menon_db": c.mean_db[lg] - c.mean_db[menon],
        }));
    }
    md_table(report, &header, &rows);
    // Aggregate: the mean over the cells (each cell is a set mean at a scale, so every set and
    // every scale weighs the same) and the number of cells won.
    let mut mean = vec![0.0; PRIMARY.len()];
    for c in cells {
        for (m, v) in mean.iter_mut().zip(&c.mean_db) {
            *m += v / cells.len() as f64;
        }
    }
    let mut order = rule.clone();
    order.sort_by(|&a, &b| mean[b].total_cmp(&mean[a]));
    report.line(format!(
        "**Aggregate over the {} linear-light cells (sets x scales, equal weight)**\n",
        cells.len()
    ));
    let header: Vec<String> = ["method", "mean of cells (dB)", "vs menon (dB)", "cells won"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    let rows: Vec<Vec<String>> = (0..PRIMARY.len())
        .map(|i| {
            vec![
                PRIMARY[i].label.to_owned(),
                format!("{:.2}", mean[i]),
                format!("{:+.2}", mean[i] - mean[menon]),
                if rule.contains(&i) {
                    format!("{}", wins[i])
                } else {
                    "n/a".to_owned()
                },
            ]
        })
        .collect();
    md_table(report, &header, &rows);
    let winner = order[0];
    report.line(format!(
        "R6 rule: **{}** would become the default (mean of cells {:.2} dB, {:+.2} dB over {}).\n",
        PRIMARY[winner].label,
        mean[winner],
        mean[winner] - mean[order[1]],
        PRIMARY[order[1]].label
    ));
    json!({
        "cells": json_cells,
        "aggregate_mean_db": PRIMARY.iter().zip(&mean).map(|(v, m)| (v.label, *m)).collect::<std::collections::BTreeMap<_, _>>(),
        "cells_won": rule.iter().map(|&i| (PRIMARY[i].label, wins[i])).collect::<std::collections::BTreeMap<_, _>>(),
        "rule_winner": PRIMARY[winner].label,
        "rule_winner_margin_db": mean[winner] - mean[order[1]],
    })
}

fn cells_json(cells: &[Cell]) -> Value {
    let variants = all_variants();
    Value::Array(
        cells
            .iter()
            .map(|c| {
                json!({
                    "set": c.set, "protocol": c.protocol, "scale": c.scale,
                    "mean_db": variants.iter().zip(&c.mean_db).map(|(v, m)| (v.label, *m)).collect::<std::collections::BTreeMap<_, _>>(),
                    "per_image_db": c.per_image.iter().map(|(n, r)| json!({
                        "image": n,
                        "db": variants.iter().zip(r).map(|(v, m)| (v.label, *m)).collect::<std::collections::BTreeMap<_, _>>(),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------------------------
// Zipper and false colour
// ---------------------------------------------------------------------------------------------

/// Artefact numbers of every column of [`PRIMARY`] on every image of the set at scale 1.
fn metrics_for_set(report: &mut Report, set: &Set) -> Value {
    let truth_labs: Vec<LabImage> = set.images.iter().map(|(_, i)| LabImage::new(i)).collect();
    // rows[image][variant]
    let mut rows: Vec<Vec<Artifacts>> = Vec::new();
    for (i, (_, img)) in set.images.iter().enumerate() {
        let m = synth::mosaic_from_rgb(img, Cfa::RGGB);
        rows.push(
            PRIMARY
                .iter()
                .map(|v| {
                    let demo = run_variant(&m, *v, 1.0);
                    artifacts(&truth_labs[i], &LabImage::new(&demo), BORDER_PX)
                })
                .collect(),
        );
    }
    let edge_px_mean = mean_finite(rows.iter().map(|r| r[0].edge_px as f64));
    report.line(format!(
        "**{}: zipper, false colour (signal scale 1, linear light; {} images, mean of per-image rates; {:.0} edge pixels per image)**\n",
        set.name,
        set.images.len(),
        edge_px_mean
    ));
    let header: Vec<String> = [
        "method",
        "zipper (% of edge px)",
        "false colour (% of px)",
        "chroma dE00 (mean)",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    let mut out_rows = Vec::new();
    let mut json_rows = Vec::new();
    for (j, v) in PRIMARY.iter().enumerate() {
        let z = mean_finite(rows.iter().map(|r| r[j].zipper_pct));
        let f = mean_finite(rows.iter().map(|r| r[j].false_colour_pct));
        let c = mean_finite(rows.iter().map(|r| r[j].chroma_de00_mean));
        out_rows.push(vec![
            v.label.to_owned(),
            format!("{z:.2}"),
            format!("{f:.3}"),
            format!("{c:.3}"),
        ]);
        json_rows.push(json!({
            "method": v.label, "zipper_pct": z, "false_colour_pct": f, "chroma_de00_mean": c,
        }));
    }
    md_table(report, &header, &out_rows);
    let mut per_image = Vec::new();
    if set.per_image {
        report.line(format!(
            "*{}: per image (zipper % / false colour %)*\n",
            set.name
        ));
        let mut header = vec!["image".to_owned()];
        header.extend(labels(&PRIMARY));
        let table: Vec<Vec<String>> = set
            .images
            .iter()
            .zip(&rows)
            .map(|((n, _), r)| {
                let mut line = vec![n.clone()];
                line.extend(r.iter().map(|a| {
                    if a.zipper_pct.is_nan() {
                        format!("n/a / {:.2}", a.false_colour_pct)
                    } else {
                        format!("{:.1} / {:.2}", a.zipper_pct, a.false_colour_pct)
                    }
                }));
                line
            })
            .collect();
        md_table(report, &header, &table);
        for ((n, _), r) in set.images.iter().zip(&rows) {
            per_image.push(json!({
                "image": n,
                "by_method": PRIMARY.iter().zip(r).map(|(v, a)| (v.label, json!({
                    "edge_px": a.edge_px, "zipper_pct": a.zipper_pct,
                    "false_colour_pct": a.false_colour_pct, "chroma_de00_mean": a.chroma_de00_mean,
                }))).collect::<std::collections::BTreeMap<_, _>>(),
            }));
        }
    }
    json!({ "set": set.name, "images": set.images.len(), "rows": json_rows, "per_image": per_image })
}

// ---------------------------------------------------------------------------------------------
// Noisy inputs
// ---------------------------------------------------------------------------------------------

/// One ISO-like noise level: stand-ins for plausible full-frame values (as `examples/ra07_ab.rs`
/// uses), not measurements. Row noise is left out.
struct Level {
    name: &'static str,
    a: f64,
    b: f64,
    /// Seed of the noise of this level (image `k` uses `seed_base + k`); fixed per level so that
    /// adding a level does not change the numbers of the others.
    seed_base: u64,
}

const LEVELS: [Level; 4] = [
    Level {
        name: "ISO 100-like",
        a: 2.5e-5,
        b: 4e-8,
        seed_base: 1000,
    },
    Level {
        name: "ISO 800-like",
        a: 2e-4,
        b: 3e-7,
        seed_base: 2000,
    },
    // Spec 01 section 6.3 asks for ISO 100 / 1600 / 6400; the 800 level above came first and stays
    // for continuity. This one follows the same stand-in scaling (`a` proportional to the ISO, `b`
    // growing by about 1.9 per stop between the 800 and 6400 levels): not a measurement.
    Level {
        name: "ISO 1600-like",
        a: 4e-4,
        b: 6e-7,
        seed_base: 4000,
    },
    Level {
        name: "ISO 6400-like",
        a: 2e-3,
        b: 2e-6,
        seed_base: 3000,
    },
];

/// Gaussian noise of variance `a x + b` (x clamped at 0) on every photosite of `m`, seeded.
fn add_shot_and_read_noise(data: &mut [f32], a: f64, b: f64, seed: u64) {
    let mut rng = synth::Rng::new(seed);
    for v in data {
        let var = a * f64::from(v.max(0.0)) + b;
        *v += (var.sqrt() as f32) * rng.gaussian();
    }
}

fn noisy_table(report: &mut Report, sets: &[&Set]) -> Value {
    let mut json_rows = Vec::new();
    report.line(
        "**Noisy inputs (spec 01 section 6.3): linear light, signal scale 1, mean PSNR (dB) over the sets' images** \
         Noise: Gaussian with variance `a x + b` per photosite (no row noise, stand-ins for ISO 100, 800, 1600 and 6400, not measurements); \
         adaptive_ratio and log_guided get the matching model, unit gains; `+ R6` is after `denoise::bilateral_opponent` with \
         sigma at 18 % grey and strength 1.\n",
    );
    for set in sets {
        let mut header = vec!["level".to_owned(), "stage".to_owned()];
        header.extend(labels(&PRIMARY));
        let mut rows = Vec::new();
        for level in &LEVELS {
            let model = NoiseModel {
                a: level.a,
                b: level.b,
                delta: [0.0; 4],
                sigma_row: 0.0,
                q: 1.0 / 65535.0,
            };
            let noise = NoiseDn::from_model(&model);
            let sigma_grey = (level.a * 0.18 + level.b).sqrt() as f32;
            let mut raw = vec![0.0; PRIMARY.len()];
            let mut den = vec![0.0; PRIMARY.len()];
            for (k, (_, img)) in set.images.iter().enumerate() {
                let mut m = synth::mosaic_from_rgb(img, Cfa::RGGB);
                add_shot_and_read_noise(&mut m.data, level.a, level.b, level.seed_base + k as u64);
                for (j, v) in PRIMARY.iter().enumerate() {
                    let inputs = DemosaicInputs {
                        gains: [1.0; 3],
                        noise: if matches!(v.label, "adaptive_ratio" | "log_guided") {
                            noise
                        } else {
                            NoiseDn::ZERO
                        },
                        ca: None,
                    };
                    let out = demosaic::demosaic_with(&m, v.method, &inputs);
                    raw[j] += psnr(&out, img, BORDER_PX, 1.0) / set.images.len() as f64;
                    let d = denoise::bilateral_opponent(
                        &out,
                        &NoiseSigma::Scalar(sigma_grey),
                        &BilateralParams::default(),
                    );
                    den[j] += psnr(&d, img, BORDER_PX, 1.0) / set.images.len() as f64;
                }
            }
            for (stage, vals) in [("demosaic", &raw), ("+ R6 bilateral", &den)] {
                let mut line = vec![level.name.to_owned(), stage.to_owned()];
                line.extend(vals.iter().map(|v| format!("{v:.2}")));
                rows.push(line);
            }
            json_rows.push(json!({
                "set": set.name, "level": level.name, "a": level.a, "b": level.b,
                "demosaic_db": PRIMARY.iter().zip(&raw).map(|(v, m)| (v.label, *m)).collect::<std::collections::BTreeMap<_, _>>(),
                "bilateral_db": PRIMARY.iter().zip(&den).map(|(v, m)| (v.label, *m)).collect::<std::collections::BTreeMap<_, _>>(),
            }));
        }
        report.line(format!("*{}*\n", set.name));
        md_table(report, &header, &rows);
    }
    Value::Array(json_rows)
}

// ---------------------------------------------------------------------------------------------
// Sets and main
// ---------------------------------------------------------------------------------------------

/// A set of the 8-bit images of `testdata/demosaic/<dir>`: linear light, and the gamma-encoded
/// values for the continuity rows.
fn image_set(dir: &str, name: &str) -> Option<(Set, Set)> {
    let root = demosaic_sets_root();
    let images = load_set(&root.join(dir));
    if images.is_empty() {
        return None;
    }
    let linear = Set {
        name: name.to_owned(),
        images: images
            .iter()
            .map(|(n, i)| (n.clone(), linearize(i)))
            .collect(),
        per_image: false,
    };
    let gamma = Set {
        name: name.to_owned(),
        images,
        per_image: false,
    };
    Some((linear, gamma))
}

fn main() {
    let mut mode = "all".to_owned();
    let mut out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/ra05");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = PathBuf::from(args.next().expect("--out needs a folder")),
            "psnr" | "metrics" | "noisy" | "all" => mode = a,
            other => {
                eprintln!("usage: ra05_gate [psnr|metrics|noisy|all] [--out DIR] (got {other})");
                std::process::exit(2);
            }
        }
    }
    std::fs::create_dir_all(&out_dir).expect("create the output folder");
    let mut report = Report::default();
    let mut json = json!({
        "about": "RA-05 demosaic gate: measurement only. Protocol in crates/ym-gpu/tests/common/ra05_metrics.rs.",
        "border_px": BORDER_PX,
        "scales": SCALES,
        "noise_models": {
            "adaptive_ratio": "nominal low-noise model of spec 01 section 6.3 (a 2e-5, b 1e-8, q 1/65535), unit gains, fixed at every scale",
            "log_guided": "zero model (p = 1), spec 02 section 6.4",
        },
    });

    // The synthetic set is already linear light; the 8-bit sets are decoded.
    let synthetic = Set {
        name: "synthetic (cpu_demosaic.rs)".to_owned(),
        images: synthetic_set(),
        per_image: true,
    };
    let kodak = image_set("kodak", "Kodak (24)");
    let mcmaster = image_set("mcmaster", "McMaster (18)");
    let mut blocked = Vec::new();
    for (dir, name) in [
        ("gharbi_hard", "Gharbi hard"),
        ("gharbi_moire", "Gharbi moire"),
        ("pixelshift200", "PixelShift200"),
    ] {
        if load_set(&demosaic_sets_root().join(dir)).is_empty() {
            blocked.push(name);
        }
    }
    if !blocked.is_empty() {
        report.line(format!(
            "Sets not run (no images in testdata/demosaic): {}. See docs/05 R6 for why.\n",
            blocked.join(", ")
        ));
    }
    if kodak.is_none() || mcmaster.is_none() {
        report.line("Kodak or McMaster missing: run tools/fetch/demosaic-sets.sh first.\n");
    }
    let mut linear_sets: Vec<&Set> = vec![&synthetic];
    if let Some((k, _)) = &kodak {
        linear_sets.push(k);
    }
    if let Some((m, _)) = &mcmaster {
        linear_sets.push(m);
    }

    if matches!(mode.as_str(), "psnr" | "all") {
        report.line("## PSNR, linear light (sRGB decoded), 10 px border excluded\n");
        let mut all_cells: Vec<Cell> = Vec::new();
        for set in &linear_sets {
            let protocol = if set.name.starts_with("synthetic") {
                "procedural, linear"
            } else {
                "sRGB decoded, linear light"
            };
            let cells = psnr_cells(&mut report, set, protocol, &SCALES);
            print_set_table(&mut report, &set.name, &cells);
            all_cells.extend(cells);
        }
        let refs: Vec<&Cell> = all_cells.iter().collect();
        let verdict = decide(&mut report, &refs);
        json["psnr_linear"] = cells_json(&all_cells);
        json["r6_rule"] = verdict;

        // Gamma-encoded values at scale 1: the numbers of docs/05's first table.
        report.line(
            "## PSNR, gamma-encoded 8-bit values as they are (scale 1; continuity with docs/05)\n",
        );
        let mut gamma_cells = Vec::new();
        for g in [&kodak, &mcmaster].into_iter().flatten().map(|(_, g)| g) {
            let cells = psnr_cells(&mut report, g, "8-bit values as they are", &[1.0]);
            print_set_table(&mut report, &format!("{} gamma", g.name), &cells);
            gamma_cells.extend(cells);
        }
        json["psnr_gamma"] = cells_json(&gamma_cells);
        report.line(
            "docs/05 R6 reference numbers (M1-12): gamma, Kodak bilinear 29.82, Menon 38.88, LMMSE 39.16; \
             McMaster bilinear 32.31, Menon 34.21, LMMSE 35.09; linear light, Kodak, Menon 36.43 \
             (constant), LMMSE 36.67 / 36.17 / 35.61 at scales 1 / 0.1 / 0.01.\n",
        );
    }

    if matches!(mode.as_str(), "metrics" | "all") {
        report.line("## Zipper and false colour (signal scale 1, linear light)\n");
        report.line(format!(
            "Definitions (tests/common/ra05_metrics.rs): edge pixel = an 8-neighbour more than {} CIE 1976 Lab units away in the ground truth; \
             zipper = the difference to the nearest-in-colour neighbour grew by more than {} in the demosaiced image (after Lu and Tan 2003); \
             false colour = chroma difference to the truth (CIEDE2000, lightness equalised) above {}. \
             `cpu/metrics.rs` has no such metrics, they are defined in the measurement code.\n",
            ra05_metrics::EDGE_DELTA_E,
            ra05_metrics::ZIPPER_DELTA_E,
            ra05_metrics::FALSE_COLOUR_DELTA_E00
        ));
        let mut v = Vec::new();
        for set in &linear_sets {
            v.push(metrics_for_set(&mut report, set));
        }
        json["artifacts"] = Value::Array(v);
    }

    if matches!(mode.as_str(), "noisy" | "all") {
        report.line("## Noisy inputs\n");
        json["noisy"] = noisy_table(&mut report, &linear_sets);
    }
    json["sets_not_run"] = json!(blocked);

    std::fs::write(out_dir.join("numbers.md"), &report.text).expect("write numbers.md");
    std::fs::write(
        out_dir.join("gate.json"),
        serde_json::to_string_pretty(&json).expect("json") + "\n",
    )
    .expect("write gate.json");
    println!("wrote numbers.md and gate.json in {}", out_dir.display());
}
