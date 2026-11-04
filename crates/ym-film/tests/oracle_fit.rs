#![allow(clippy::needless_range_loop)]
//! F-06b: the print side (stages 7 and 8) fitted to the oracle's outputs, and the held-out check.
//!
//! The enlarger is fed the oracle's own `cmy_film` (so that nothing upstream of it can leak into the fit) and compared with
//! the oracle's `cmy_print` and `log_e_print`, after the mid-grey normalisation of `tests/oracle.rs` (the two filters and
//! the exposure are solved per stock so that the grey's three print densities equal the oracle's). The nine numbers of
//! the enlarger model (`ym_film::params::EnlargerModel`: lamp temperature, the glass edge, the yellow edge and the
//! magenta band) are fitted on the ColorChecker patches and the grey ramp; the 125-point lattice, and the probe lattices
//! (`tests/oracle/probe_*.json`, densities injected at `cmy_film`), are held out.
//!
//! `YM_FIT_RUN=1 cargo test -p ym-film --release --test oracle_fit -- --ignored --nocapture fit_enlarger` fits and prints
//! the JSON of `data/film/params/fitted/enlarger.json`. `held_out` (a normal test) measures the shipped numbers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use ym_film::enlarger::Enlarger;
use ym_film::params::{BandEdge, EnlargerModel, GlassEdge, load_enlarger_parts};
use ym_film::{FilmChain, FilmParams, Output, PaperChoice, TableSpectra};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle")
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

const NEGATIVES: [&str; 5] = [
    "kodak_portra_400",
    "kodak_ektar_100",
    "kodak_gold_200",
    "fujifilm_pro_400h",
    "kodak_vision3_500t",
];

fn chain(id: &str) -> FilmChain {
    FilmChain::new(
        &data(),
        id,
        &PaperChoice::Matching,
        FilmParams {
            output: Output::Srgb,
            ..FilmParams::default()
        },
        Arc::new(TableSpectra::load(&data().join("film/upsampling")).expect("table")),
    )
    .expect("chain")
}

/// One negative's reference: the oracle's film densities in, its print numbers out.
struct Case {
    id: &'static str,
    chain: FilmChain,
    grey_film: [f64; 3],
    grey_print: [f64; 3],
    grey_log_e: [f64; 3],
    /// (film densities, cmy_print, log_e_print)
    train: Vec<([f64; 3], [f64; 3], [f64; 3])>,
    lattice: Vec<([f64; 3], [f64; 3], [f64; 3])>,
    probe: Vec<([f64; 3], [f64; 3], [f64; 3])>,
}

fn rows(st: &Stock, group: &str) -> Vec<([f64; 3], [f64; 3], [f64; 3])> {
    let g = &st.groups[group];
    (0..g["cmy_film"].len())
        .map(|k| (g["cmy_film"][k], g["cmy_print"][k], g["log_e_print"][k]))
        .collect()
}

fn cases() -> Vec<Case> {
    NEGATIVES
        .iter()
        .map(|id| {
            let st: Stock = serde_json::from_str(
                &std::fs::read_to_string(oracle_dir().join(format!("{id}.json"))).expect("fixture"),
            )
            .expect("json");
            let mut train = rows(&st, "patches");
            let ramp = rows(&st, "ramp");
            let grey = ramp[12];
            train.extend(ramp);
            let probe: Probe = serde_json::from_str(
                &std::fs::read_to_string(oracle_dir().join(format!("probe_{id}.json")))
                    .expect("probe"),
            )
            .expect("probe json");
            let probe_rows = (0..probe.densities.len())
                .map(|k| {
                    (
                        probe.densities[k],
                        probe.taps["cmy_print"][k],
                        probe.taps["log_e_print"][k],
                    )
                })
                // The densities a negative reaches: every layer between 0.25 and 2.3 (not the injected extremes).
                .filter(|(d, _, _)| {
                    d.iter().all(|v| (0.2..=2.4).contains(v)) && d.iter().any(|v| *v > 0.2)
                })
                .collect();
            Case {
                id,
                chain: chain(id),
                grey_film: grey.0,
                grey_print: grey.1,
                grey_log_e: grey.2,
                lattice: rows(&st, "lattice"),
                train,
                probe: probe_rows,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Knobs {
    y: f64,
    m: f64,
    shift: f64,
    c: f64,
}

/// Densities of the print of film densities `d`, with the knobs: (log10 of the light on the paper plus the shift, densities).
fn print(enl: &Enlarger, d: [f64; 3], k: Knobs) -> ([f64; 3], [f64; 3]) {
    let p = enl.light_on_paper(d, [k.y, k.m, k.c]);
    let log_e = p.map(|v| v.max(1e-300).log10() + k.shift);
    (log_e, enl.paper_density(log_e))
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0; 3];
    for k in 0..3 {
        let mut m = a;
        for row in 0..3 {
            m[row][k] = b[row];
        }
        out[k] = det(&m) / d;
    }
    Some(out)
}

/// Mid-grey normalisation: the knobs for which the grey's print densities equal the oracle's.
fn normalise(enl: &Enlarger, c: &Case) -> Knobs {
    normalise_with_cyan(enl, c, 0.0)
}

fn normalise_with_cyan(enl: &Enlarger, c: &Case, cyan: f64) -> Knobs {
    let mut x = [0.0, 0.0, 0.0];
    let kn = |x: [f64; 3]| Knobs {
        y: x[0],
        m: x[1],
        shift: x[2],
        c: cyan,
    };
    let resid = |x: [f64; 3]| {
        let d = print(enl, c.grey_film, kn(x)).1;
        [
            d[0] - c.grey_print[0],
            d[1] - c.grey_print[1],
            d[2] - c.grey_print[2],
        ]
    };
    for _ in 0..40 {
        let r = resid(x);
        let err = r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if err < 1e-9 {
            break;
        }
        let mut jac = [[0.0; 3]; 3];
        for k in 0..3 {
            let h = if k == 2 { 0.01 } else { 0.5 };
            let mut x2 = x;
            x2[k] += h;
            let r2 = resid(x2);
            for row in 0..3 {
                jac[row][k] = (r2[row] - r[row]) / h;
            }
        }
        let Some(step) = solve3(jac, [-r[0], -r[1], -r[2]]) else {
            break;
        };
        let len = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut t = if len > 20.0 { 20.0 / len } else { 1.0 };
        for _ in 0..20 {
            let xc = [x[0] + t * step[0], x[1] + t * step[1], x[2] + t * step[2]];
            let rc = resid(xc);
            let ec = rc.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if ec < err {
                x = xc;
                break;
            }
            t *= 0.5;
        }
    }
    kn(x)
}

type Rows = [([f64; 3], [f64; 3], [f64; 3])];

/// (sum of squared stage-8 differences, sum of squared stage-7 differences relative to the grey, count*3, max stage-8).
fn score(enl: &Enlarger, c: &Case, k: Knobs, set: &Rows) -> (f64, f64, f64, f64) {
    let (g_log, _) = print(enl, c.grey_film, k);
    let (mut s8, mut s7, mut mx) = (0.0, 0.0, 0.0_f64);
    for (d, t8, t7) in set {
        let (log_e, dens) = print(enl, *d, k);
        for ch in 0..3 {
            let e8 = dens[ch] - t8[ch];
            let e7 = (log_e[ch] - g_log[ch]) - (t7[ch] - c.grey_log_e[ch]);
            s8 += e8 * e8;
            s7 += e7 * e7;
            mx = mx.max(e8.abs());
        }
    }
    (s8, s7, 3.0 * set.len() as f64, mx)
}

fn model_of(x: &[f64; 9]) -> EnlargerModel {
    EnlargerModel {
        lamp_k: x[0],
        glass: GlassEdge {
            peak: 0.86,
            edge_nm: x[1],
            width_nm: x[2],
        },
        yellow: BandEdge {
            edge_nm: x[3],
            width_nm: x[4],
            ..BandEdge::default()
        },
        magenta: BandEdge {
            edge_nm: x[5],
            width_nm: x[6],
            edge2_nm: x[7],
            width2_nm: x[8],
        },
        cyan: BandEdge {
            edge_nm: 595.0,
            width_nm: 20.0,
            ..BandEdge::default()
        },
    }
}

const LOW: [f64; 9] = [2800.0, 600.0, 15.0, 430.0, 4.0, 430.0, 4.0, 540.0, 4.0];
const HIGH: [f64; 9] = [4200.0, 800.0, 140.0, 560.0, 45.0, 560.0, 45.0, 680.0, 45.0];

fn enlargers(cases: &[Case], x: &[f64; 9]) -> Vec<Enlarger> {
    let parts = model_of(x).parts("fit").expect("parts");
    cases
        .iter()
        .map(|c| Enlarger::new(c.chain.film(), c.chain.paper().expect("paper"), &parts))
        .collect()
}

fn cost(cases: &[Case], x: &[f64; 9]) -> f64 {
    for k in 0..9 {
        if x[k] < LOW[k] || x[k] > HIGH[k] {
            return 1e9;
        }
    }
    let enls = enlargers(cases, x);
    let mut total = 0.0;
    let mut n = 0.0;
    for (c, e) in cases.iter().zip(&enls) {
        let k = normalise(e, c);
        let (s8, s7, cnt, _) = score(e, c, k, &c.train);
        total += s8 + s7;
        n += cnt;
    }
    (total / n).sqrt()
}

/// Nelder-Mead on the scaled parameters (each divided by its range).
fn nelder_mead(
    f: &dyn Fn(&[f64; 9]) -> f64,
    start: [f64; 9],
    step: f64,
    iters: usize,
) -> ([f64; 9], f64) {
    let scale: Vec<f64> = (0..9).map(|k| HIGH[k] - LOW[k]).collect();
    let to_x = |u: &[f64; 9]| -> [f64; 9] {
        let mut x = [0.0; 9];
        for k in 0..9 {
            x[k] = u[k] * scale[k];
        }
        x
    };
    let mut u0 = [0.0; 9];
    for k in 0..9 {
        u0[k] = start[k] / scale[k];
    }
    let mut simplex: Vec<([f64; 9], f64)> = vec![(u0, f(&to_x(&u0)))];
    for k in 0..9 {
        let mut u = u0;
        u[k] += step;
        simplex.push((u, f(&to_x(&u))));
    }
    for _ in 0..iters {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("nan"));
        let best = simplex[0].1;
        let worst = simplex[9].1;
        if (worst - best).abs() < 1e-10 {
            break;
        }
        let mut cen = [0.0; 9];
        for s in &simplex[..9] {
            for k in 0..9 {
                cen[k] += s.0[k] / 9.0;
            }
        }
        let pt = |t: f64| -> [f64; 9] {
            let mut u = [0.0; 9];
            for k in 0..9 {
                u[k] = cen[k] + t * (simplex[9].0[k] - cen[k]);
            }
            u
        };
        let refl = pt(-1.0);
        let fr = f(&to_x(&refl));
        if fr < simplex[0].1 {
            let exp = pt(-2.0);
            let fe = f(&to_x(&exp));
            simplex[9] = if fe < fr { (exp, fe) } else { (refl, fr) };
        } else if fr < simplex[8].1 {
            simplex[9] = (refl, fr);
        } else {
            let con = pt(if fr < simplex[9].1 { -0.5 } else { 0.5 });
            let fc = f(&to_x(&con));
            if fc < simplex[9].1.min(fr) {
                simplex[9] = (con, fc);
            } else {
                let b = simplex[0].0;
                for s in simplex.iter_mut().skip(1) {
                    for k in 0..9 {
                        s.0[k] = b[k] + 0.5 * (s.0[k] - b[k]);
                    }
                    s.1 = f(&to_x(&s.0));
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("nan"));
    (to_x(&simplex[0].0), simplex[0].1)
}

fn report(cases: &[Case], enls: &[Enlarger], label: &str) {
    println!("{label}");
    for (c, e) in cases.iter().zip(enls) {
        let k = normalise(e, c);
        let line = |name: &str, set: &Rows| {
            let (s8, s7, n, mx) = score(e, c, k, set);
            format!(
                "{name}: cmy_print rms {:.4} max {:.3}, log_e rms {:.4}",
                (s8 / n).sqrt(),
                mx,
                (s7 / n).sqrt()
            )
        };
        println!(
            "  {:<20} y {:+6.1} m {:+6.1} | {} | {} | {}",
            c.id,
            k.y,
            k.m,
            line("train", &c.train),
            line("lattice", &c.lattice),
            line("probe", &c.probe)
        );
    }
}

#[test]
#[ignore = "the fit; run by hand"]
fn fit_enlarger() {
    let cs = cases();
    let hand = [3400.0, 700.0, 70.0, 500.0, 16.0, 490.0, 12.0, 595.0, 14.0];
    report(&cs, &enlargers(&cs, &hand), "hand-chosen");
    println!("cost at hand-chosen {:.4}", cost(&cs, &hand));
    let f = |x: &[f64; 9]| cost(&cs, x);
    let mut best = nelder_mead(&f, hand, 0.03, 1500);
    for _ in 0..4 {
        best = nelder_mead(&f, best.0, 0.01, 1500);
    }
    println!("fitted cost {:.4}: {:.3?}", best.1, best.0);
    report(&cs, &enlargers(&cs, &best.0), "fitted");
}

/// The shipped parameters, measured on the held-out sets (a regression guard, not a gate).
#[test]
#[ignore = "diagnostic"]
fn cyan_scan() {
    let cs = cases();
    let parts = load_enlarger_parts(&data().join("film/params")).expect("parts");
    for c in &cs {
        let e = Enlarger::new(c.chain.film(), c.chain.paper().expect("paper"), &parts);
        for cy in [-40.0, -20.0, 0.0, 20.0, 40.0, 80.0] {
            let k = normalise_with_cyan(&e, c, cy);
            let (s8, s7, n, mx) = score(&e, c, k, &c.train);
            println!(
                "{} cyan {cy:+5.0}: y {:+6.1} m {:+6.1} train rms {:.4} max {:.3} log_e {:.4}",
                c.id,
                k.y,
                k.m,
                (s8 / n).sqrt(),
                mx,
                (s7 / n).sqrt()
            );
        }
    }
}

#[test]
#[ignore = "diagnostic"]
fn worst_print_points() {
    let cs = cases();
    let parts = load_enlarger_parts(&data().join("film/params")).expect("parts");
    for c in cs
        .iter()
        .filter(|c| c.id == "kodak_ektar_100" || c.id == "kodak_portra_400")
    {
        let e = Enlarger::new(c.chain.film(), c.chain.paper().expect("paper"), &parts);
        let k = normalise(&e, c);
        let mut v: Vec<(f64, usize)> = vec![];
        for (i, (d, t8, _)) in c.train.iter().enumerate() {
            let dens = print(&e, *d, k).1;
            let m = (0..3)
                .map(|ch| (dens[ch] - t8[ch]).abs())
                .fold(0.0, f64::max);
            v.push((m, i));
        }
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (m, i) in v.iter().take(6) {
            let (d, t8, t7) = c.train[*i];
            let (le, dens) = print(&e, d, k);
            println!(
                "{} #{i}: film {:.3?} theirs cmy_print {:.3?} ours {:.3?} err {m:.3}; log_e theirs-grey {:.3?} ours-grey {:.3?}",
                c.id,
                d,
                t8,
                dens,
                [
                    t7[0] - c.grey_log_e[0],
                    t7[1] - c.grey_log_e[1],
                    t7[2] - c.grey_log_e[2]
                ],
                {
                    let g = print(&e, c.grey_film, k).0;
                    [le[0] - g[0], le[1] - g[1], le[2] - g[2]]
                }
            );
        }
    }
}

#[test]
fn held_out() {
    let cs = cases();
    let parts = load_enlarger_parts(&data().join("film/params")).expect("parts");
    let enls: Vec<Enlarger> = cs
        .iter()
        .map(|c| Enlarger::new(c.chain.film(), c.chain.paper().expect("paper"), &parts))
        .collect();
    report(&cs, &enls, "shipped parameters");
}
