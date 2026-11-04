#![allow(clippy::needless_range_loop)]
//! Diagnostics of F-06b: where the disagreement with the oracle sits, one stage at a time. Numbers only (black-box outputs in
//! `tests/oracle/`); nothing here reads or uses another program's code. Run with `-- --nocapture` to see them.
//!
//! * `scan_alone` / `slide_scan_alone`: our stage 9 on the oracle's own print (or film) densities, injected at its taps
//!   (`probe_*.json`), against its `rgb_out`. A regression guard (the 95th percentile of dE00 stays under 1.0).
//! * `curves_alone`: our curves on the oracle's own `log_e_film` against its `cmy_film`. A guard (0.003 density).
//! * `error_by_saturation`: the stage-2 error against the colour's distance to the spectral locus, per stock.
//! * `table_score` (`YM_TABLE=<dir>` for another upsampling table): the stage-2 RMS over the seven stocks.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use ym_color::consts::linalg;
use ym_color::deltae::{Summary, d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::gamut::matrices;
use ym_film::cie::Cie;
use ym_film::{FilmChain, FilmParams, Output, PaperChoice, Spectra, TableSpectra};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle")
}

#[derive(Deserialize)]
struct Inputs {
    patches: Vec<[f64; 3]>,
    lattice: Vec<[f64; 3]>,
    ramp: Vec<[f64; 3]>,
}
#[derive(Deserialize)]
struct Stock {
    groups: BTreeMap<String, BTreeMap<String, Vec<[f64; 3]>>>,
}
#[derive(Deserialize)]
struct Probe {
    densities: Vec<[f64; 3]>,
    taps: BTreeMap<String, Vec<[f64; 3]>>,
}

const STOCKS: [&str; 7] = [
    "kodak_portra_400",
    "kodak_ektar_100",
    "kodak_gold_200",
    "fujifilm_pro_400h",
    "kodak_vision3_500t",
    "fujifilm_velvia_100",
    "kodak_ektachrome_100",
];

fn load(id: &str) -> (Inputs, Stock) {
    let read = |name: String| std::fs::read_to_string(oracle_dir().join(name)).expect("fixture");
    (
        serde_json::from_str(&read("inputs.json".into())).expect("inputs"),
        serde_json::from_str(&read(format!("{id}.json"))).expect("stock"),
    )
}

fn probe(id: &str) -> Probe {
    let text =
        std::fs::read_to_string(oracle_dir().join(format!("probe_{id}.json"))).expect("probe");
    serde_json::from_str(&text).expect("probe json")
}

fn build_with(id: &str, table: Arc<dyn Spectra>) -> FilmChain {
    FilmChain::new(
        &data(),
        id,
        &PaperChoice::Matching,
        FilmParams {
            output: Output::Srgb,
            ..FilmParams::default()
        },
        table,
    )
    .expect("chain")
}

fn shipped_table() -> Arc<dyn Spectra> {
    Arc::new(TableSpectra::load(&data().join("film/upsampling")).expect("table"))
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn lab_of(rgb: [f64; 3]) -> [f64; 3] {
    xyz_to_lab(
        linalg::mul_vec(&matrices().rec709_to_xyz, rgb),
        d65_white_xyz(),
    )
}

/// Our scan of the densities injected into the oracle, against its `rgb_out` (paper white scaled to 1 for a print).
fn scan_scores(id: &str) -> Summary {
    let p = probe(id);
    let chain = build_with(id, shipped_table());
    let slide = chain.paper().is_none();
    let t = if slide {
        &p.taps["rgb_out_from_film"]
    } else {
        &p.taps["rgb_out_from_print"]
    };
    let white = if slide { [1.0; 3] } else { t[0] };
    let de: Vec<f64> = p
        .densities
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let ours = chain.scan_stage().display(*d);
            let theirs = [t[k][0] / white[0], t[k][1] / white[1], t[k][2] / white[2]];
            delta_e_2000(lab_of(ours), lab_of(theirs))
        })
        .collect();
    let s = Summary::of(&de).expect("de");
    println!(
        "{id} scan alone: dE00 median {:.3} p95 {:.3} max {:.3}",
        s.median, s.p95, s.max
    );
    s
}

/// Stage 9 alone, prints: a paper's dyes read as the oracle reads them (an undefined dye density holds its nearest
/// defined value; zero let a deep shadow leak red and blue and put the Kodak paper's 95th percentile at 4.4).
#[test]
fn scan_alone() {
    for id in ["kodak_portra_400", "fujifilm_pro_400h"] {
        let s = scan_scores(id);
        assert!(s.p95 < 1.0, "{id}: {s:?}");
    }
}

/// Stage 9 alone, slides.
#[test]
fn slide_scan_alone() {
    for id in ["fujifilm_velvia_100", "kodak_ektachrome_100"] {
        let s = scan_scores(id);
        assert!(s.p95 < 1.0, "{id}: {s:?}");
    }
}

/// Stage 5 alone: our curves on the oracle's own `log_e_film` give its `cmy_film`. The slides' large stage-5 differences
/// are therefore the steep reversal curves magnifying stage 2, not the curves.
#[test]
fn curves_alone() {
    for id in STOCKS {
        let (_, st) = load(id);
        let chain = build_with(id, shipped_table());
        let mut worst = 0.0_f64;
        for grp in ["patches", "lattice", "ramp"] {
            for (le, cm) in st.groups[grp]["log_e_film"]
                .iter()
                .zip(&st.groups[grp]["cmy_film"])
            {
                let o = chain.develop_stage().curves(*le);
                for c in 0..3 {
                    worst = worst.max((o[c] - cm[c]).abs());
                }
            }
        }
        println!("{id} curves alone: worst {worst:.4} density");
        assert!(worst < 0.003, "{id}: {worst}");
    }
}

/// The distance of a chromaticity from the white, as a share of the distance to the spectral locus along the same ray.
fn relative_radius(cie: &Cie, white: [f64; 2], p: [f64; 2]) -> f64 {
    let mut locus = vec![];
    for c in &cie.cmf {
        let s = c[0] + c[1] + c[2];
        if c[1] > 1e-6 && s > 0.0 {
            locus.push([c[0] / s, c[1] / s]);
        }
    }
    let d = [p[0] - white[0], p[1] - white[1]];
    let r = d[0].hypot(d[1]);
    let u = [d[0] / r, d[1] / r];
    let mut best = f64::INFINITY;
    for k in 0..locus.len() {
        let (a, b) = (locus[k], locus[(k + 1) % locus.len()]);
        let e = [b[0] - a[0], b[1] - a[1]];
        let den = u[0] * e[1] - u[1] * e[0];
        if den.abs() < 1e-18 {
            continue;
        }
        let aw = [a[0] - white[0], a[1] - white[1]];
        let t = (aw[0] * e[1] - aw[1] * e[0]) / den;
        let s = (aw[0] * u[1] - aw[1] * u[0]) / den;
        if t > 0.0 && (-1e-12..=1.0 + 1e-12).contains(&s) && t < best {
            best = t;
        }
    }
    r / best
}

/// Stage 2 (window on, relative to the grey) by the colour's distance to the locus: the error grows with saturation, which
/// is the upsampling table (the sigmoid spectra of saturated colours), not the chain.
#[test]
fn error_by_saturation() {
    let cie = Cie::load(&data().join("film/cie")).expect("cie");
    let edges = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9, 1.01];
    for id in STOCKS {
        let (inp, st) = load(id);
        let chain = build_with(id, shipped_table());
        let g = st.groups["ramp"]["log_e_film"][12];
        let go = chain
            .exposure_stage()
            .log_exposure(&chain.scene_stage().from_linear(inp.ramp[12]), 0.0);
        let mut bins: Vec<Vec<f64>> = vec![vec![]; edges.len() - 1];
        for (grp, src) in [("patches", &inp.patches), ("lattice", &inp.lattice)] {
            for (a, t) in src.iter().zip(&st.groups[grp]["log_e_film"]) {
                let l = chain.scene_stage().from_linear(*a);
                let q = chain.scene_stage().compress([l.x, l.y]);
                if (q[0] - l.x).abs() + (q[1] - l.y).abs() > 1e-9 {
                    continue;
                }
                let rr = relative_radius(&cie, chain.scene_stage().white_xy(), [l.x, l.y]);
                let o = chain.exposure_stage().log_exposure(&l, 0.0);
                let e = (0..3)
                    .map(|c| (o[c] - go[c] - (t[c] - g[c])).abs())
                    .fold(0.0, f64::max);
                for b in 0..bins.len() {
                    if rr >= edges[b] && rr < edges[b + 1] {
                        bins[b].push(e);
                    }
                }
            }
        }
        let mut line = String::new();
        for (b, v) in bins.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
            let mx = v.iter().fold(0.0_f64, |a, x| a.max(*x));
            line += &format!(
                " r{:.2}-{:.2}: n{} rms {:.3} max {:.3};",
                edges[b],
                edges[b + 1],
                v.len(),
                rms(v),
                mx
            );
        }
        println!("{id}:{line}");
    }
}

/// Stage 2 against the oracle (window on, relative to the grey) for the table in `$YM_TABLE` (default: the shipped one),
/// pooled over the seven stocks. The study that chose the luminance target of the table (`docs/24` section 3.3).
#[test]
fn table_score() {
    let dir =
        std::env::var("YM_TABLE").map_or_else(|_| data().join("film/upsampling"), PathBuf::from);
    let table: Arc<dyn Spectra> = Arc::new(TableSpectra::load(&dir).expect("table"));
    let (mut all, mut patches, mut lattice, mut line) = (vec![], vec![], vec![], String::new());
    for id in STOCKS {
        let (inp, st) = load(id);
        let chain = build_with(id, table.clone());
        let g = st.groups["ramp"]["log_e_film"][12];
        let go = chain
            .exposure_stage()
            .log_exposure(&chain.scene_stage().from_linear(inp.ramp[12]), 0.0);
        let mut d = vec![];
        for (grp, src) in [
            ("patches", &inp.patches),
            ("lattice", &inp.lattice),
            ("ramp", &inp.ramp),
        ] {
            for (a, t) in src.iter().zip(&st.groups[grp]["log_e_film"]) {
                let l = chain.scene_stage().from_linear(*a);
                let q = chain.scene_stage().compress([l.x, l.y]);
                if (q[0] - l.x).abs() + (q[1] - l.y).abs() > 1e-9 {
                    continue;
                }
                let o = chain.exposure_stage().log_exposure(&l, 0.0);
                for c in 0..3 {
                    let e = o[c] - go[c] - (t[c] - g[c]);
                    d.push(e);
                    match grp {
                        "patches" => patches.push(e),
                        "lattice" => lattice.push(e),
                        _ => {}
                    }
                }
            }
        }
        line += &format!(" {}:{:.4}", &id[..id.len().min(12)], rms(&d));
        all.extend(d);
    }
    let mx = all.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    println!(
        "TABLE {} pooled rms {:.4} max {:.3} patches {:.4} lattice {:.4} |{line}",
        dir.display(),
        rms(&all),
        mx,
        rms(&patches),
        rms(&lattice)
    );
}
