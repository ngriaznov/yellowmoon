#![allow(clippy::needless_range_loop)]
//! Diagnostic for the reading of `hanatos2025_adaptation_window_params` (F-06 item 4), not a gate.
//!
//! For real reflectances (ColorChecker, CES, NIST CQS, the 190 training spectra) lit by the film's reference light, the *true*
//! raw exposure of layer `c` is `sum refl * ill * S_c` (the profile's sensitivities, no window). The engine instead takes the
//! chromaticity of that radiance, looks the spectrum up in the upsampling table and integrates that. Each reading of the window
//! is scored by how close the second comes to the first, as the RMS over patches and layers of the difference in `log10`
//! exposure relative to the white's. No other program's output is used.
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
            let r: Vec<f64> = l.split(',').skip(1).map(|v| v.trim().parse().unwrap_or(0.0)).collect();
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
        ("x1.2", Box::new(|l, p| rising_edge(l, p[0], p[1] * 1.2) * falling_edge(l, p[2], p[3] * 1.2))),
        ("x1.7", Box::new(|l, p| rising_edge(l, p[0], p[1] * 1.7) * falling_edge(l, p[2], p[3] * 1.7))),
        ("x2.5", Box::new(|l, p| rising_edge(l, p[0], p[1] * 2.5) * falling_edge(l, p[2], p[3] * 2.5))),
        ("x3", Box::new(|l, p| rising_edge(l, p[0], p[1] * 3.0) * falling_edge(l, p[2], p[3] * 3.0))),
        ("x4", Box::new(|l, p| rising_edge(l, p[0], p[1] * 4.0) * falling_edge(l, p[2], p[3] * 4.0))),
        ("ours: edges, width divides", Box::new(|l, p| rising_edge(l, p[0], p[1]) * falling_edge(l, p[2], p[3]))),
        ("width is sigma (x sqrt2)", Box::new(|l, p| {
            let s = std::f64::consts::SQRT_2;
            rising_edge(l, p[0], p[1] * s) * falling_edge(l, p[2], p[3] * s)
        })),
        ("width is FWHM-ish (x 2)", Box::new(|l, p| rising_edge(l, p[0], p[1] * 2.0) * falling_edge(l, p[2], p[3] * 2.0))),
        ("gaussian cdf, sigma = width", Box::new(|l, p| {
            let cdf = |z: f64| 0.5 * erfc(-z / std::f64::consts::SQRT_2);
            cdf((l - p[0]) / p[1]) * cdf((p[2] - l) / p[3])
        })),
        ("blue edge only", Box::new(|l, p| rising_edge(l, p[0], p[1]))),
        ("red edge only", Box::new(|l, p| falling_edge(l, p[2], p[3]))),
        ("red edge at c+2w", Box::new(|l, p| rising_edge(l, p[0], p[1]) * falling_edge(l, p[2] + 2.0 * p[3], p[3]))),
        ("blue edge at c-2w", Box::new(|l, p| rising_edge(l, p[0] - 2.0 * p[1], p[1]) * falling_edge(l, p[2], p[3]))),
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
            (0..N_LAMBDA).map(|i| s[i] * sens[i][c] * w(lambda_nm(i))).sum()
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
    println!("columns: {}", readings.iter().map(|r| r.0).collect::<Vec<_>>().join(" | "));
    println!(
        "mean rms log10 error: {}",
        totals.iter().map(|t| format!("{:.4}", t / names.len() as f64)).collect::<Vec<_>>().join(" ")
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
        out[c] = (0..N_LAMBDA).map(|i| u[i] * sens[i][c] * w(lambda_nm(i))).sum::<f64>() * k[c];
    }
    out
}
