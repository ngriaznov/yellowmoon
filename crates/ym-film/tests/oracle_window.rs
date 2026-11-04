#![allow(clippy::needless_range_loop)]
//! Diagnostic for the reading of `hanatos2025_adaptation_window_params` (F-06 item 4), not a gate.
//!
//! For real reflectances (ColorChecker, CES, NIST CQS, the 190 training spectra) lit by the film's reference light, the *true*
//! raw exposure of layer `c` is `sum refl * ill * S_c` (the profile's sensitivities, no window). The engine instead takes the
//! chromaticity of that radiance, looks the spectrum up in the upsampling table and integrates that. Each reading of the window
//! is scored by how close the second comes to the first, as the RMS over patches and layers of the difference in `log10`
//! exposure relative to the white's (the first test). The later tests score the readings against the oracle's `log_e_film`
//! (`tests/oracle/`, black-box numbers only) and print the greens stage by stage.
//! Run: `cargo test -p ym-film --release --test oracle_window -- --nocapture`.

use std::path::{Path, PathBuf};

use ym_film::cie::Cie;
use ym_film::exposure::reference_spectrum;
use ym_film::math::{erfc, falling_edge, rising_edge};
use ym_film::{N_LAMBDA, Profile, Spectra, TableSpectra, lambda_nm};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn reflectances() -> Vec<Vec<f64>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/spectral");
    let mut out = Vec::new();
    for f in [
        "reflectances_colorchecker_24.csv",
        "reflectances_ces_99.csv",
        "reflectances_nist_cqs_vs_15.csv",
        "reflectances_training_190.csv",
    ] {
        let Ok(t) = std::fs::read_to_string(dir.join(f)) else {
            continue;
        };
        for l in t.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let r: Vec<f64> = l
                .split(',')
                .skip(1)
                .map(|v| v.trim().parse().unwrap_or(0.0))
                .collect();
            if r.len() == N_LAMBDA {
                out.push(r);
            }
        }
    }
    out
}

type Window = Box<dyn Fn(f64, &[f64]) -> f64>;

fn readings() -> Vec<(&'static str, Window)> {
    vec![
        ("none", Box::new(|_, _| 1.0)),
        (
            "x1.2",
            Box::new(|l, p| rising_edge(l, p[0], p[1] * 1.2) * falling_edge(l, p[2], p[3] * 1.2)),
        ),
        (
            "x1.7",
            Box::new(|l, p| rising_edge(l, p[0], p[1] * 1.7) * falling_edge(l, p[2], p[3] * 1.7)),
        ),
        (
            "x2.5",
            Box::new(|l, p| rising_edge(l, p[0], p[1] * 2.5) * falling_edge(l, p[2], p[3] * 2.5)),
        ),
        (
            "x3",
            Box::new(|l, p| rising_edge(l, p[0], p[1] * 3.0) * falling_edge(l, p[2], p[3] * 3.0)),
        ),
        (
            "x4",
            Box::new(|l, p| rising_edge(l, p[0], p[1] * 4.0) * falling_edge(l, p[2], p[3] * 4.0)),
        ),
        (
            "ours: edges, width divides",
            Box::new(|l, p| rising_edge(l, p[0], p[1]) * falling_edge(l, p[2], p[3])),
        ),
        (
            "width is sigma (x sqrt2)",
            Box::new(|l, p| {
                let s = std::f64::consts::SQRT_2;
                rising_edge(l, p[0], p[1] * s) * falling_edge(l, p[2], p[3] * s)
            }),
        ),
        (
            "width is FWHM-ish (x 2)",
            Box::new(|l, p| rising_edge(l, p[0], p[1] * 2.0) * falling_edge(l, p[2], p[3] * 2.0)),
        ),
        (
            "gaussian cdf, sigma = width",
            Box::new(|l, p| {
                let cdf = |z: f64| 0.5 * erfc(-z / std::f64::consts::SQRT_2);
                cdf((l - p[0]) / p[1]) * cdf((p[2] - l) / p[3])
            }),
        ),
        (
            "blue edge only",
            Box::new(|l, p| rising_edge(l, p[0], p[1])),
        ),
        (
            "red edge only",
            Box::new(|l, p| falling_edge(l, p[2], p[3])),
        ),
        (
            "red edge at c+2w",
            Box::new(|l, p| rising_edge(l, p[0], p[1]) * falling_edge(l, p[2] + 2.0 * p[3], p[3])),
        ),
        (
            "blue edge at c-2w",
            Box::new(|l, p| rising_edge(l, p[0] - 2.0 * p[1], p[1]) * falling_edge(l, p[2], p[3])),
        ),
    ]
}

#[test]
fn window_readings_scored_on_real_reflectances() {
    let refl = reflectances();
    if refl.is_empty() {
        eprintln!("skipped: testdata/spectral is absent");
        return;
    }
    let cie = Cie::load(&data().join("film/cie")).expect("cie");
    let table = TableSpectra::load(&data().join("film/upsampling")).expect("table");
    let names = [
        "kodak_portra_400",
        "kodak_ektar_100",
        "kodak_gold_200",
        "fujifilm_pro_400h",
        "kodak_vision3_500t",
        "fujifilm_velvia_100",
        "kodak_ektachrome_100",
    ];
    let readings = readings();
    let mut totals = vec![0.0; readings.len()];
    for id in names {
        let p = Profile::load(&data().join("film/spektrafilm/profiles"), id).expect("profile");
        let ill = reference_spectrum(&p, &cie).expect("ill");
        let sens = p.sensitivity();
        let wp = &p.window_params;
        // Radiances and the white.
        let radiance: Vec<[f64; N_LAMBDA]> = refl
            .iter()
            .map(|r| {
                let mut s = [0.0; N_LAMBDA];
                for i in 0..N_LAMBDA {
                    s[i] = r[i] * ill[i];
                }
                s
            })
            .collect();
        let resp = |s: &[f64; N_LAMBDA], w: &dyn Fn(f64) -> f64, c: usize| -> f64 {
            (0..N_LAMBDA)
                .map(|i| s[i] * sens[i][c] * w(lambda_nm(i)))
                .sum()
        };
        let one = |_: f64| 1.0;
        let true_white: Vec<f64> = (0..3).map(|c| resp(&ill, &one, c)).collect();
        let yill = cie.xyz_of_light(&ill, false)[1];
        print!("{id:<22}");
        for (ri, (name, win)) in readings.iter().enumerate() {
            let w = |l: f64| win(l, wp);
            // The window renormalises on the reference light, as the engine does.
            let k: Vec<f64> = (0..3).map(|c| true_white[c] / resp(&ill, &w, c)).collect();
            let tw = table_white(&cie, &table, &ill, &w, &sens, &k);
            let mut sum = 0.0;
            let mut n = 0.0;
            for s in &radiance {
                let y = cie.xyz_of_light(s, false)[1];
                let xy = cie.xy_of_light(s);
                let u = table.spectrum(xy[0], xy[1]);
                for c in 0..3 {
                    let truth = (resp(s, &one, c) / y / (true_white[c] / yill)).log10();
                    let model = (resp(&u, &w, c) * k[c] / tw[c]).log10();
                    sum += (truth - model).powi(2);
                    n += 1.0;
                }
            }
            let rms = (sum / n).sqrt();
            totals[ri] += rms;
            print!(" {rms:.4}");
            let _ = name;
        }
        println!();
    }
    println!(
        "columns: {}",
        readings.iter().map(|r| r.0).collect::<Vec<_>>().join(" | ")
    );
    println!(
        "mean rms log10 error: {}",
        totals
            .iter()
            .map(|t| format!("{:.4}", t / names.len() as f64))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn table_white(
    cie: &Cie,
    table: &TableSpectra,
    ill: &[f64; N_LAMBDA],
    w: &dyn Fn(f64) -> f64,
    sens: &[[f64; 3]],
    k: &[f64],
) -> [f64; 3] {
    let xy = cie.xy_of_light(ill);
    let u = table.spectrum(xy[0], xy[1]);
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = (0..N_LAMBDA)
            .map(|i| u[i] * sens[i][c] * w(lambda_nm(i)))
            .sum::<f64>()
            * k[c];
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------
// Against the oracle's stage 2 (`log_e_film`), window on and off. Needs the fixtures of `tests/oracle/`.

mod fixtures {
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    pub struct Inputs {
        pub patches: Vec<[f64; 3]>,
        pub lattice: Vec<[f64; 3]>,
        pub ramp: Vec<[f64; 3]>,
    }

    #[derive(Deserialize)]
    pub struct Stock {
        pub groups: BTreeMap<String, BTreeMap<String, Vec<[f64; 3]>>>,
    }
}

fn load_oracle(id: &str) -> Option<(fixtures::Inputs, fixtures::Stock)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle");
    let inputs =
        serde_json::from_str(&std::fs::read_to_string(dir.join("inputs.json")).ok()?).ok()?;
    let stock =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id}.json"))).ok()?)
            .ok()?;
    Some((inputs, stock))
}

/// RMS difference in log10 exposure between our stage 2 (a given window) and the oracle's, over the patches and the lattice
/// (points the soft compression leaves alone), after removing the difference at the mid-grey of the ramp per layer.
fn stage2_rms(id: &str, window: &[f64; N_LAMBDA], theirs_key: &str) -> Option<(f64, f64)> {
    use std::sync::Arc;
    use ym_film::exposure::ExposureStage;
    use ym_film::scene::SceneStage;
    let (inputs, stock) = load_oracle(id)?;
    let cie = Cie::load(&data().join("film/cie")).ok()?;
    let table: Arc<dyn Spectra> =
        Arc::new(TableSpectra::load(&data().join("film/upsampling")).ok()?);
    let p = Profile::load(&data().join("film/spektrafilm/profiles"), id).ok()?;
    let ill = reference_spectrum(&p, &cie).ok()?;
    let scene = SceneStage::new(&cie, &ill);
    let stage = ExposureStage::with_window(&p, &cie, table, window).ok()?;
    let ours = |awg4: [f64; 3]| stage.log_exposure(&scene.from_linear(awg4), 0.0);
    let grey_ours = ours(inputs.ramp[12]);
    let grey_theirs = stock.groups["ramp"][theirs_key][12];
    let (mut sum_abs, mut sum_al, mut n) = (0.0, 0.0, 0.0);
    for (group, pts) in [("patches", &inputs.patches), ("lattice", &inputs.lattice)] {
        for (k, awg4) in pts.iter().enumerate() {
            let light = scene.from_linear(*awg4);
            let q = scene.compress([light.x, light.y]);
            if (q[0] - light.x).abs() + (q[1] - light.y).abs() > 1e-9 {
                continue;
            }
            let o = ours(*awg4);
            let t = stock.groups[group][theirs_key][k];
            for c in 0..3 {
                sum_abs += (o[c] - t[c]).powi(2);
                sum_al += ((o[c] - grey_ours[c]) - (t[c] - grey_theirs[c])).powi(2);
                n += 1.0;
            }
        }
    }
    Some(((sum_abs / n).sqrt(), (sum_al / n).sqrt()))
}

#[test]
fn window_readings_against_the_oracle_stage_2() {
    let ids = [
        "kodak_portra_400",
        "kodak_ektar_100",
        "kodak_gold_200",
        "fujifilm_pro_400h",
        "kodak_vision3_500t",
        "fujifilm_velvia_100",
        "kodak_ektachrome_100",
    ];
    let readings = readings();
    let mut sums = vec![0.0; readings.len()];
    let mut sums_off = 0.0;
    let mut count = 0.0;
    for id in ids {
        let Ok(p) = Profile::load(&data().join("film/spektrafilm/profiles"), id) else {
            return;
        };
        if load_oracle(id).is_none() {
            eprintln!("skipped: no oracle fixtures");
            return;
        }
        let none = [1.0; N_LAMBDA];
        let off = stage2_rms(id, &none, "log_e_film_nowindow").expect("fixture");
        print!(
            "{id:<22} no window vs no window: abs {:.4} aligned {:.4} |",
            off.0, off.1
        );
        sums_off += off.1;
        for (i, (_, win)) in readings.iter().enumerate() {
            let mut w = [0.0; N_LAMBDA];
            for k in 0..N_LAMBDA {
                w[k] = win(lambda_nm(k), &p.window_params);
            }
            let r = stage2_rms(id, &w, "log_e_film").expect("fixture");
            sums[i] += r.1;
            print!(" {:.4}", r.1);
        }
        println!();
        count += 1.0;
    }
    println!(
        "columns: {}",
        readings.iter().map(|r| r.0).collect::<Vec<_>>().join(" | ")
    );
    println!(
        "mean aligned rms vs oracle (window on): {}",
        sums.iter()
            .map(|t| format!("{:.4}", t / count))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "mean aligned rms, both windows off: {:.4}",
        sums_off / count
    );
}

/// The greens of the ColorChecker at stage 2, relative to mid grey (log10 exposure per layer R, G, B): the oracle, our reading
/// of the window, the corrected reading, no window, and the exposure of the real reflectance (no upsampling, no window).
#[test]
fn greens_at_stage_2() {
    use std::sync::Arc;
    use ym_film::exposure::{ExposureStage, window_of};
    use ym_film::scene::SceneStage;
    let id = "kodak_portra_400";
    let Some((inputs, stock)) = load_oracle(id) else {
        return;
    };
    let refl = reflectances();
    if refl.is_empty() {
        return;
    }
    let cie = Cie::load(&data().join("film/cie")).expect("cie");
    let table: Arc<dyn Spectra> =
        Arc::new(TableSpectra::load(&data().join("film/upsampling")).expect("table"));
    let p = Profile::load(&data().join("film/spektrafilm/profiles"), id).expect("profile");
    let ill = reference_spectrum(&p, &cie).expect("ill");
    let scene = SceneStage::new(&cie, &ill);
    let mut old = [1.0; N_LAMBDA];
    let wp = &p.window_params;
    for k in 0..N_LAMBDA {
        old[k] = rising_edge(lambda_nm(k), wp[0], wp[1]) * falling_edge(lambda_nm(k), wp[2], wp[3]);
    }
    let new = window_of(wp);
    let stage = |w: &[f64; N_LAMBDA]| {
        ExposureStage::with_window(&p, &cie, table.clone(), w).expect("stage")
    };
    let (s_old, s_new, s_none) = (stage(&old), stage(&new), stage(&[1.0; N_LAMBDA]));
    let sens = p.sensitivity();
    let truth = |r: &[f64]| -> [f64; 3] {
        // The exposure of the real spectrum lit by the reference light, relative to the white, per unit luminance.
        let mut e = [0.0; 3];
        let (mut y, mut yw) = (0.0, 0.0);
        let mut white = [0.0; 3];
        for i in 0..N_LAMBDA {
            y += r[i] * ill[i] * cie.cmf[i][1];
            yw += ill[i] * cie.cmf[i][1];
            for c in 0..3 {
                e[c] += r[i] * ill[i] * sens[i][c];
                white[c] += ill[i] * sens[i][c];
            }
        }
        [0, 1, 2].map(|c| ((e[c] / y) / (white[c] / yw)).log10())
    };
    let grey_idx = 21; // neutral 5 (18 %)
    let rel = |v: [f64; 3], g: [f64; 3]| [v[0] - g[0], v[1] - g[1], v[2] - g[2]];
    let at =
        |s: &ExposureStage, k: usize| s.log_exposure(&scene.from_linear(inputs.patches[k]), 0.0);
    let names = [
        (3, "foliage"),
        (5, "bluish green"),
        (10, "yellow green"),
        (13, "green"),
        (2, "blue sky"),
        (6, "orange"),
    ];
    println!("patch (log10 exposure relative to the 18 % grey; R G B)");
    for (k, name) in names {
        let t = rel(
            stock.groups["patches"]["log_e_film"][k],
            stock.groups["patches"]["log_e_film"][grey_idx],
        );
        let o = rel(at(&s_old, k), at(&s_old, grey_idx));
        let n = rel(at(&s_new, k), at(&s_new, grey_idx));
        let z = rel(at(&s_none, k), at(&s_none, grey_idx));
        let r = rel(truth(&refl[k]), truth(&refl[grey_idx]));
        println!(
            "{name:<13} oracle {t:+.3?}  old {o:+.3?}  corrected {n:+.3?}  no window {z:+.3?}  real {r:+.3?}"
        );
    }
}
