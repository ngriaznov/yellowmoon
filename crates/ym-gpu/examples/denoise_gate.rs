//! The single-frame denoise measurement (`docs/algorithms/05-single-frame-denoise.md`, *Measurement*).
//! Measurement only: nothing here changes what a render does.
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --release --example denoise_gate -- \
//!     [calibrate|compare|all] [--limit N] [--out DIR]
//! ```
//!
//! Kodak and McMaster (`testdata/demosaic`, `tools/fetch/demosaic-sets.sh`) in linear light, placed where a
//! photo at the camera's exposure sits on the raw scale (times `2^-CAMERA_GAIN_EV`: middle grey at 0.083,
//! white at 0.46 of the clip), mosaicked as RGGB, with Gaussian noise of variance `a x + b` per photosite at
//! the ISO-like levels of the demosaic gate and one more (stand-ins, not measurements), and demosaiced with the
//! adaptive ratio demosaic and the matching noise model. Each variant then runs on that image, and the PSNR
//! against the clean image is taken in the working encoding (`cpu::denoise_space`, the display-like state the
//! methods are made for, peak 1) and in linear light (peak 0.46, the image's white), 10 px border excluded,
//! mean of per-image dB.
//!
//! Variants: `none` (the demosaic alone); `current` (today's classical path at Auto: the R6 bilateral with a
//! σ map and R7 with Colour 50 and Detail 50); methods A, B and C of the spec in the working encoding.
//!
//! * `calibrate`: method A's thresholds against the noise. The oracle noise is the standard deviation of the
//!   working-encoding difference between the noisy and the clean image (Y, and Cb with Cr); `L = kL σY 22.627 /
//!   0.1` puts the hard threshold at `kL` orthonormal standard deviations (the same for C with 0.06). A sweep of
//!   `kL` and `kC` on the first `--limit` Kodak images (default 8); also appendix D's estimate against the
//!   oracle, and methods B and C at a few strengths around their "measured noise" settings.
//! * `compare`: every variant on both sets at every level: A at the calibrated `kL`, `kC` (from `calibrate` in
//!   the same run, or the defaults below), B at L 50, C at L 80 and C 50; R 1; without blend (Detail 0) and
//!   with today's Auto detail (blend 15); also R 0 and 2 for each method; seconds per megapixel.
//!
//! The output folder (default `testdata/denoise_gate`, git-ignored) gets `numbers.md` and `gate.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_gpu::cpu::bm_dct::{self, BmDctParams};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::denoise_space::{decode_image, encode_image};
use ym_gpu::cpu::detail::{self, C_AUTO, D_AUTO, NoiseReduction};
use ym_gpu::cpu::frame_denoise::{C_PER_SIGMA, L_PER_SIGMA, working_noise};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::nl_bayes::{self, NlBayesParams, Variant as NlVariant};
use ym_gpu::cpu::{Cfa, NoiseDn, NoiseSigma, RgbImage, filters, noise_level, synth, ycbcr};
use ym_noise::model::NoiseModel;

#[path = "../tests/common/demosaic_metrics.rs"]
mod demosaic_metrics;

use demosaic_metrics::{BORDER_PX, demosaic_sets_root, linearize, load_set};

/// Prints every line and keeps it for `numbers.md`.
#[derive(Default)]
struct Report {
    lines: Vec<String>,
}

impl Report {
    fn line(&mut self, s: impl Into<String>) {
        let s = s.into();
        println!("{s}");
        self.lines.push(s);
    }

    fn table(&mut self, header: &[String], rows: &[Vec<String>]) {
        self.line(format!("| {} |", header.join(" | ")));
        self.line(format!("|{}|", vec!["---"; header.len()].join("|")));
        for r in rows {
            self.line(format!("| {} |", r.join(" | ")));
        }
        self.line("");
    }
}

/// One ISO-like noise level: the demosaic gate's stand-ins (`examples/demosaic_gate.rs`) and one more at
/// 12800 by the same scaling. Not measurements; row noise left out.
struct Level {
    name: &'static str,
    a: f64,
    b: f64,
    seed_base: u64,
}

const LEVELS: [Level; 5] = [
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
    Level {
        name: "ISO 12800-like",
        a: 4e-3,
        b: 4e-6,
        seed_base: 5000,
    },
];

/// The default thresholds of method A, in orthonormal noise standard deviations, until `calibrate` runs.
const DEFAULT_KL: f64 = 2.7;
const DEFAULT_KC: f64 = 2.7;

/// One test image: the clean image on the raw scale, the noisy demosaiced one, both in the working encoding,
/// the oracle noise and the level.
struct Case {
    clean_lin: RgbImage,
    noisy_lin: RgbImage,
    clean_w: RgbImage,
    noisy_w: RgbImage,
    /// The oracle noise: the standard deviation of the working-encoding difference.
    sigma_y: f64,
    sigma_c: f64,
    /// The noise the render computes from the model (`frame_denoise::working_noise`, unit gains).
    model_y: f64,
    model_c: f64,
    sigma_map: Vec<f32>,
}

fn add_noise(data: &mut [f32], a: f64, b: f64, seed: u64) {
    let mut rng = synth::Rng::new(seed);
    for v in data {
        let var = a * f64::from(v.max(0.0)) + b;
        *v += (var.sqrt() as f32) * rng.gaussian();
    }
}

/// The standard deviation of `a − b` over the interior (border excluded), for plane lists.
fn diff_std(a: &[&[f64]], b: &[&[f64]], w: usize, h: usize) -> f64 {
    let (mut s, mut s2, mut n) = (0.0, 0.0, 0.0);
    for (pa, pb) in a.iter().zip(b) {
        for y in BORDER_PX..h - BORDER_PX {
            for x in BORDER_PX..w - BORDER_PX {
                let d = pa[y * w + x] - pb[y * w + x];
                s += d;
                s2 += d * d;
                n += 1.0;
            }
        }
    }
    let m = s / n;
    (s2 / n - m * m).max(0.0).sqrt()
}

fn make_case(img_lin: &RgbImage, level: &Level, seed: u64) -> Case {
    let gain = (-CAMERA_GAIN_EV).exp2() as f32;
    let mut clean_lin = img_lin.clone();
    clean_lin.data.iter_mut().for_each(|v| *v *= gain);
    let mut m = synth::mosaic_from_rgb(&clean_lin, Cfa::RGGB);
    add_noise(&mut m.data, level.a, level.b, seed);
    let model = NoiseModel {
        a: level.a,
        b: level.b,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: 1.0 / 65535.0,
    };
    let inputs = DemosaicInputs {
        gains: [1.0; 3],
        noise: NoiseDn::from_model(&model),
        ca: None,
    };
    let noisy_lin = demosaic::demosaic_with(&m, Method::AdaptiveRatio, &inputs);
    let clean_w = encode_image(&clean_lin, CAMERA_GAIN_EV);
    let noisy_w = encode_image(&noisy_lin, CAMERA_GAIN_EV);
    let (w, h) = (clean_w.width, clean_w.height);
    let [ny, ncb, ncr] = ycbcr::to_ycbcr(&noisy_w);
    let [cy, ccb, ccr] = ycbcr::to_ycbcr(&clean_w);
    let sigma_y = diff_std(&[&ny], &[&cy], w, h);
    let sigma_c = diff_std(&[&ncb, &ncr], &[&ccb, &ccr], w, h);
    // The σ map of today's path: the noise model at a 5 x 5 box mean of the noisy luma (the chain's map
    // comes from 5 x 5 means of the mosaic).
    let luma = noisy_lin.luma_plane();
    let (mut mean, mut tmp) = (vec![0.0; w * h], vec![0.0; w * h]);
    filters::box_mean(&luma, w, h, 2, &mut mean, &mut tmp);
    let sigma_map = mean
        .iter()
        .map(|&v| (level.a * f64::from(v.max(0.0)) + level.b).sqrt() as f32)
        .collect();
    let (model_y, model_c) = working_noise(&model, [1.0; 3]);
    Case {
        clean_lin,
        noisy_lin,
        clean_w,
        noisy_w,
        sigma_y,
        sigma_c,
        model_y,
        model_c,
        sigma_map,
    }
}

/// A variant's output, in linear light (the raw scale).
type Output = RgbImage;

fn current(case: &Case) -> Output {
    let ns = NoiseSigma::Plane(&case.sigma_map);
    let d = denoise::bilateral_opponent(&case.noisy_lin, &ns, &BilateralParams::default());
    let nr = NoiseReduction {
        luminance: 0.0,
        color: C_AUTO,
        detail: D_AUTO,
    };
    detail::apply(&d, Some(&case.noisy_lin), &nr, &ns).expect("same sizes")
}

fn method_a(case: &Case, kl: f64, kc: f64, radius: usize, blend: f64) -> Output {
    let p = BmDctParams {
        luma: (kl * case.model_y * L_PER_SIGMA).min(100.0),
        chroma: (kc * case.model_c * C_PER_SIGMA).min(100.0),
        radius,
        blend,
    };
    decode_image(&bm_dct::denoise(&case.noisy_w, &p), CAMERA_GAIN_EV)
}

fn method_bc(
    case: &Case,
    variant: NlVariant,
    luma: f64,
    chroma: f64,
    radius: usize,
    blend: f64,
) -> Output {
    let p = NlBayesParams {
        variant,
        luma,
        chroma,
        radius,
        blend,
    };
    decode_image(&nl_bayes::denoise(&case.noisy_w, &p), CAMERA_GAIN_EV)
}

/// PSNR in the working encoding (peak 1) and in linear light (peak: the clean image's white on the raw scale).
fn score(case: &Case, out: &Output) -> (f64, f64) {
    let w = encode_image(out, CAMERA_GAIN_EV);
    let peak_lin = (-CAMERA_GAIN_EV).exp2();
    (
        psnr(&w, &case.clean_w, BORDER_PX, 1.0),
        psnr(out, &case.clean_lin, BORDER_PX, peak_lin),
    )
}

/// Mean of per-image dB and the seconds per megapixel of one variant over a list of cases.
fn run(cases: &[Case], f: &dyn Fn(&Case) -> Output) -> (f64, f64, f64) {
    let (mut pw, mut pl, mut secs, mut mp) = (0.0, 0.0, 0.0, 0.0);
    for c in cases {
        let t = Instant::now();
        let out = f(c);
        secs += t.elapsed().as_secs_f64();
        mp += (c.noisy_w.width * c.noisy_w.height) as f64 / 1e6;
        let (a, b) = score(c, &out);
        pw += a;
        pl += b;
    }
    let n = cases.len() as f64;
    (pw / n, pl / n, secs / mp)
}

fn cases_for(images: &[(String, RgbImage)], level: &Level) -> Vec<Case> {
    images
        .iter()
        .enumerate()
        .map(|(k, (_, img))| make_case(img, level, level.seed_base + k as u64))
        .collect()
}

fn calibrate(report: &mut Report, kodak: &[(String, RgbImage)], limit: usize) -> (f64, f64, Value) {
    let images = &kodak[..limit.min(kodak.len())];
    report.line(format!(
        "## Calibration (first {} Kodak images; PSNR in the working encoding, dB)\n",
        images.len()
    ));
    let ks = [1.5, 2.0, 2.5, 2.7, 3.0, 3.5];
    let mut json_rows = Vec::new();
    let mut votes = Vec::new();
    for level in &LEVELS {
        let cases = cases_for(images, level);
        let sy = cases.iter().map(|c| c.sigma_y).sum::<f64>() / cases.len() as f64;
        let sc = cases.iter().map(|c| c.sigma_c).sum::<f64>() / cases.len() as f64;
        let (my, mc) = (cases[0].model_y, cases[0].model_c);
        // Appendix D's estimate against the oracle.
        let est: Vec<(f64, f64)> = cases
            .iter()
            .filter_map(|c| noise_level::estimate(&c.noisy_w).map(|e| (e.sigma_y, e.sigma_c)))
            .collect();
        let (ey, ec) = if est.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (
                est.iter().map(|e| e.0).sum::<f64>() / est.len() as f64,
                est.iter().map(|e| e.1).sum::<f64>() / est.len() as f64,
            )
        };
        report.line(format!(
            "*{}*: from the model σY {:.4} ({:.1}/255), σC {:.4}; measured after the demosaic (oracle) σY {:.4}, σC {:.4} (ratio {:.2}, {:.2}); appendix D's estimate σY {:.4}, σC {:.4} ({} of {} images found a flat region)\n",
            level.name, my, my * 255.0, mc, sy, sc, sy / my, sc / mc, ey, ec, est.len(), cases.len()
        ));
        let mut header = vec!["kL \\ kC".to_owned()];
        header.extend(ks.iter().map(|k| format!("{k}")));
        let mut rows = Vec::new();
        let mut best = (f64::NEG_INFINITY, 0.0, 0.0);
        for &kl in &ks {
            let mut row = vec![format!("{kl}")];
            for &kc in &ks {
                let (pw, _, _) = run(&cases, &|c| method_a(c, kl, kc, 1, 0.0));
                if pw > best.0 {
                    best = (pw, kl, kc);
                }
                row.push(format!("{pw:.2}"));
                json_rows.push(json!({"level": level.name, "kL": kl, "kC": kc, "psnr_w": pw}));
            }
            rows.push(row);
        }
        report.table(&header, &rows);
        report.line(format!(
            "Best: kL {} kC {} ({:.2} dB)\n",
            best.1, best.2, best.0
        ));
        votes.push((best.1, best.2));
        // B and C around their measured-noise settings.
        let mut rows = Vec::new();
        for l in [30.0, 40.0, 50.0, 60.0, 70.0] {
            let (pw, _, _) = run(&cases, &|c| method_bc(c, NlVariant::B, l, 0.0, 1, 0.0));
            rows.push(vec![format!("B L {l}"), format!("{pw:.2}")]);
            json_rows.push(json!({"level": level.name, "method": "B", "L": l, "psnr_w": pw}));
        }
        for (l, c) in [
            (60.0, 50.0),
            (80.0, 30.0),
            (80.0, 50.0),
            (80.0, 70.0),
            (100.0, 50.0),
        ] {
            let (pw, _, _) = run(&cases, &|cs| method_bc(cs, NlVariant::C, l, c, 1, 0.0));
            rows.push(vec![format!("C L {l} C {c}"), format!("{pw:.2}")]);
            json_rows
                .push(json!({"level": level.name, "method": "C", "L": l, "C": c, "psnr_w": pw}));
        }
        report.table(&["setting".to_owned(), "PSNR W (dB)".to_owned()], &rows);
    }
    // The constants: the median over the levels of the best kL and kC.
    let mut kl: Vec<f64> = votes.iter().map(|v| v.0).collect();
    let mut kc: Vec<f64> = votes.iter().map(|v| v.1).collect();
    kl.sort_by(f64::total_cmp);
    kc.sort_by(f64::total_cmp);
    let (kl, kc) = (kl[kl.len() / 2], kc[kc.len() / 2]);
    report.line(format!(
        "**Method A's thresholds: kL {kl}, kC {kc} (median of the levels' best).**\n"
    ));
    (kl, kc, Value::Array(json_rows))
}

fn compare(
    report: &mut Report,
    sets: &[(&str, Vec<(String, RgbImage)>)],
    kl: f64,
    kc: f64,
) -> Value {
    let mut json_rows = Vec::new();
    for (name, images) in sets {
        report.line(format!(
            "## {name} ({} images): PSNR in the working encoding / linear light (dB), seconds per megapixel\n",
            images.len()
        ));
        let header: Vec<String> = ["variant", "R", "blend"]
            .iter()
            .map(|s| (*s).to_owned())
            .chain(LEVELS.iter().map(|l| l.name.to_owned()))
            .chain(std::iter::once("s/MP".to_owned()))
            .collect();
        type F<'a> = Box<dyn Fn(&Case) -> Output + 'a>;
        let mut variants: Vec<(String, usize, f64, F)> = vec![
            (
                "none".into(),
                0,
                0.0,
                Box::new(|c: &Case| c.noisy_lin.clone()),
            ),
            ("current (bilateral + R7)".into(), 0, 0.0, Box::new(current)),
        ];
        for r in [0usize, 1, 2] {
            for blend in [0.0, 15.0] {
                if r != 1 && blend > 0.0 {
                    continue;
                }
                variants.push((
                    format!("A kL {kl} kC {kc}"),
                    r,
                    blend,
                    Box::new(move |c: &Case| method_a(c, kl, kc, r, blend)),
                ));
                variants.push((
                    "B L 50".into(),
                    r,
                    blend,
                    Box::new(move |c: &Case| method_bc(c, NlVariant::B, 50.0, 0.0, r, blend)),
                ));
                variants.push((
                    "C L 80 C 50".into(),
                    r,
                    blend,
                    Box::new(move |c: &Case| method_bc(c, NlVariant::C, 80.0, 50.0, r, blend)),
                ));
            }
        }
        let level_cases: Vec<Vec<Case>> = LEVELS.iter().map(|l| cases_for(images, l)).collect();
        let mut rows = Vec::new();
        for (label, r, blend, f) in &variants {
            let mut row = vec![label.clone(), r.to_string(), format!("{blend}")];
            let mut secs = 0.0;
            for (level, cases) in LEVELS.iter().zip(&level_cases) {
                let (pw, pl, s) = run(cases, f.as_ref());
                secs += s / LEVELS.len() as f64;
                row.push(format!("{pw:.2} / {pl:.2}"));
                json_rows.push(json!({"set": name, "variant": label, "R": r, "blend": blend, "level": level.name, "psnr_w": pw, "psnr_lin": pl, "s_per_mp": s}));
            }
            row.push(format!("{secs:.2}"));
            rows.push(row);
        }
        report.table(&header, &rows);
    }
    Value::Array(json_rows)
}

fn main() {
    let mut mode = "all".to_owned();
    let mut limit = 8usize;
    let mut out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/denoise_gate");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).expect("--limit N"),
            "--out" => out_dir = PathBuf::from(args.next().expect("--out DIR")),
            m => mode = m.to_owned(),
        }
    }
    let root = demosaic_sets_root();
    let load = |dir: &str| -> Vec<(String, RgbImage)> {
        load_set(&root.join(dir))
            .into_iter()
            .map(|(n, i)| (n, linearize(&i)))
            .collect()
    };
    let kodak = load("kodak");
    let mcmaster = load("mcmaster");
    assert!(
        !kodak.is_empty(),
        "no Kodak images under {} (tools/fetch/demosaic-sets.sh)",
        root.display()
    );
    let mut report = Report::default();
    report.line("# Single-frame denoise measurement\n");
    let threads = rayon::current_num_threads();
    report.line(format!(
        "Threads: {threads}. Seconds are wall time of this process on a shared machine.\n"
    ));
    let mut json_out = json!({});
    let (mut kl, mut kc) = (DEFAULT_KL, DEFAULT_KC);
    if mode == "calibrate" || mode == "all" {
        let (a, b, j) = calibrate(&mut report, &kodak, limit);
        (kl, kc) = (a, b);
        json_out["calibration"] = j;
    }
    if mode == "compare" || mode == "all" {
        let sets = vec![("Kodak", kodak), ("McMaster", mcmaster)];
        json_out["compare"] = compare(&mut report, &sets, kl, kc);
    }
    json_out["kL"] = json!(kl);
    json_out["kC"] = json!(kc);
    std::fs::create_dir_all(&out_dir).expect("output folder");
    std::fs::write(out_dir.join("numbers.md"), report.lines.join("\n")).expect("numbers.md");
    std::fs::write(
        out_dir.join("gate.json"),
        serde_json::to_string_pretty(&json_out).expect("json"),
    )
    .expect("gate.json");
    println!("wrote {}", out_dir.display());
}
