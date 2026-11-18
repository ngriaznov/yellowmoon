//! The lateral CA left after R9, with and without the CA-aware green refinement of the adaptive
//! ratio demosaic (spec 01 sections 3.6 and 11). Measurement only: nothing here changes a default.
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example ca_green_residual \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' \
//!     [-- [--size WxH] [--noise SIGMA] [--seed N] [--diagnose] [--out DIR]]
//! ```
//!
//! The scene is a grey (R = G = B) analytic field of edges at every orientation and softness of
//! about 2.4 px (10 to 90 %), so the correct image has no colour and any chroma or any displacement
//! of red and blue against green after the chain is an error. The mosaic is made *consistent with
//! the lens tags*: the site of colour `c` at source position `q` gets the scene value at the output
//! position `p` with `source_position(c, p) = q` (found by iteration), so R9 with the tags is the
//! exact inverse and what remains is the demosaic's and the resampler's own error. Several CA
//! strengths (corner shifts of red and blue of about 0.6, 1.2 and 2.4 px each) with and without
//! barrel distortion.
//!
//! Per configuration the table gives, for plain adaptive ratio (`ca = None`), adaptive ratio with
//! the CA path, Menon and, as a scale reference, Menon with R9 *without* the CA correction:
//!
//! * the displacement of red and blue against green along the radius after R9
//!   (`metrics::lateral_ca_residual`, per radial band, RMS in pixels),
//! * the colour error where the scene has edges: mean `|R - G|` and `|B - G|` (the truth has none),
//! * the PSNR of the green plane and of all three planes against the ideal scene, 16 px border off,
//! * the CPU time of the demosaic per megapixel.
//!
//! `--diagnose` also prints, per radial band, the displacement of red against green in the
//! demosaiced image *before* R9 next to what the tags say it is (`amount * rho^3` for these tables),
//! and the signed means after R9: a demosaic that interpolates red and blue as differences from
//! green pulls their edges toward green's (about a quarter to a third of the displacement here), so
//! R9, which corrects the full tag value afterwards, over-corrects, and the residual it leaves has
//! the opposite sign.
//!
//! The output folder (default `testdata/ca_green_residual`, git-ignored) gets `synthetic_seed<N>_noise<S>.md`.

use std::path::PathBuf;

use ym_gpu::cpu::NoiseDn;
use ym_gpu::cpu::demosaic::adaptive_ratio::Tuning;
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::metrics::{CaResidualOptions, lateral_ca_residual, psnr};

// The scenes, the lens tags, the mosaic and the variants are shared with the fit harness
// (`fit_demosaic`): `tests/common/ca_scenes.rs`.
#[path = "../tests/common/ca_scenes.rs"]
mod ca_scenes;

use ca_scenes::{CA_CONFIGS, Scene, ca_noise_model, ca_scene, edge_chroma, grey_of, run_variant};

fn main() {
    let mut size = (1800usize, 1200usize);
    let mut noise_sigma = 0.0f32;
    let mut seed = 2024u64;
    let mut diagnose = false;
    let mut out_dir = PathBuf::from("testdata/ca_green_residual");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                let v = &args[i + 1];
                let (a, b) = v.split_once('x').expect("WxH");
                size = (a.parse().expect("width"), b.parse().expect("height"));
                i += 1;
            }
            "--noise" => {
                noise_sigma = args[i + 1].parse().expect("sigma");
                i += 1;
            }
            "--diagnose" => diagnose = true,
            "--seed" => {
                seed = args[i + 1].parse().expect("seed");
                i += 1;
            }
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let (fw, fh) = size;
    let hd = 0.5 * (fw as f64).hypot(fh as f64);
    let mpix = (fw * fh) as f64 / 1e6;
    let model = ca_noise_model(noise_sigma);
    let noise = NoiseDn::from_model(&model);
    let scene = Scene::new(seed);
    let opts = CaResidualOptions {
        patch_px: 48,
        stride_px: 48,
        ..CaResidualOptions::default()
    };
    let mut report = String::new();
    let mut say = |s: String| {
        println!("{s}");
        report.push_str(&s);
        report.push('\n');
    };
    say(format!(
        "CA-aware green refinement, synthetic CA scene (seed {seed}), {fw} x {fh}, noise sigma {noise_sigma}, window of the residual: {} px patches, red and blue against green along the radius after R9 (RMS, px).\n",
        opts.patch_px
    ));
    for (strength, distortion) in CA_CONFIGS {
        let sc = ca_scene(&scene, size, strength, distortion, noise_sigma);
        let (geo, geo_no_ca, curves) = (&sc.geo, &sc.geo_no_ca, &sc.curves);
        let (centre, corner_px, mosaic, truth) = (sc.centre, sc.corner_px, &sc.mosaic, &sc.truth);
        let (cw, ch) = geo.output_size();
        let tuning = Tuning::DEFAULT;
        let rows = [
            (
                "adaptive ratio, plain",
                run_variant(mosaic, Method::AdaptiveRatio, None, noise, geo, &tuning),
            ),
            (
                "adaptive ratio, CA path",
                run_variant(
                    mosaic,
                    Method::AdaptiveRatio,
                    Some(curves),
                    noise,
                    geo,
                    &tuning,
                ),
            ),
            (
                "Menon",
                run_variant(mosaic, Method::Menon, None, noise, geo, &tuning),
            ),
            (
                "Menon, R9 without CA (reference)",
                run_variant(mosaic, Method::Menon, None, noise, geo_no_ca, &tuning),
            ),
        ];
        say(format!(
            "### strength {strength}, distortion {}: |R - B| at the corner {corner_px:.2} px, output {cw} x {ch}\n",
            if distortion { "1.5 % barrel" } else { "none" }
        ));
        let res: Vec<_> = rows
            .iter()
            .map(|(_, r)| lateral_ca_residual(&r.out, centre, hd, &opts))
            .collect();
        let bands = res[0].bands.len();
        let mut head = String::from("| variant |");
        let mut sep = String::from("|---|");
        for b in &res[0].bands {
            head.push_str(&format!(" R/B {:.1}-{:.1} |", b.radius_from, b.radius_to));
            sep.push_str("---|");
        }
        head.push_str(" R/B all | chroma at edges (R-G / B-G mean) | PSNR G | PSNR RGB | s/MP |");
        sep.push_str("---|---|---|---|---|");
        say(head);
        say(sep);
        if diagnose {
            // The displacement before R9 against what the tags say it is: amount * rho^3.
            let raw_img = demosaic::demosaic_with(
                mosaic,
                Method::Menon,
                &DemosaicInputs {
                    gains: [1.0; 3],
                    noise,
                    ca: None,
                },
            );
            let r0 = lateral_ca_residual(&raw_img, (fw as f64 / 2.0, fh as f64 / 2.0), hd, &opts);
            for (b, band) in r0.bands.iter().enumerate() {
                let mut sum = 0.0;
                let mut n = 0.0;
                for p in &r0.details {
                    if p.radius >= band.radius_from && p.radius < band.radius_to {
                        sum += (1.2 * strength * p.radius.powi(3)).powi(2);
                        n += 1.0;
                    }
                }
                println!(
                    "diagnose band {b}: measured red rms {:.3} (mean {:.3}), analytic {:.3}, patches {}",
                    band.red_rms_px,
                    band.red_mean_px,
                    (sum / f64::max(n, 1.0)).sqrt(),
                    band.patches
                );
            }
        }
        if diagnose {
            for ((name, _), r) in rows.iter().zip(&res) {
                let means: Vec<String> = r
                    .bands
                    .iter()
                    .map(|b| format!("{:+.3}/{:+.3}", b.red_mean_px, b.blue_mean_px))
                    .collect();
                println!(
                    "diagnose signed mean R/B after R9, {name}: {}",
                    means.join("  ")
                );
            }
        }
        let truth_g = grey_of(truth, 1);
        for ((name, run), r) in rows.iter().zip(&res) {
            let mut line = format!("| {name} |");
            for b in 0..bands {
                line.push_str(&format!(
                    " {:.3} / {:.3} |",
                    r.bands[b].red_rms_px, r.bands[b].blue_rms_px
                ));
            }
            let (chroma_r, chroma_b) = edge_chroma(&run.out, truth.plane(1));
            line.push_str(&format!(
                " {:.3} / {:.3} | {:.4} / {:.4} | {:.2} | {:.2} | {:.2} |",
                r.red_rms_px,
                r.blue_rms_px,
                chroma_r,
                chroma_b,
                psnr(&grey_of(&run.out, 1), &truth_g, 16, 1.0),
                psnr(&run.out, truth, 16, 1.0),
                run.seconds / mpix,
            ));
            say(line);
        }
        say(String::new());
    }
    std::fs::create_dir_all(&out_dir).expect("output folder");
    let name = format!("synthetic_seed{seed}_noise{noise_sigma}.md");
    std::fs::write(out_dir.join(name), report).expect("write the report");
}
