//! RA-03b: the lateral CA left after R9, with and without the CA-aware green refinement of the
//! adaptive ratio demosaic (`docs/algorithms/TASKS.md` RA-03b, spec 01 section 3.6). Measurement
//! only: nothing here changes a default.
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example ra03b_ca_residual \
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
//! The output folder (default `testdata/ra03b`, git-ignored) gets `synthetic_seed<N>_noise<S>.md`.

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use ym_gpu::cpu::demosaic::{self, CaCurves, DemosaicInputs, Method};
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::metrics::{CaResidualOptions, lateral_ca_residual, psnr};
use ym_gpu::cpu::synth::Rng;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, Placement, RgbImage, resample};
use ym_noise::model::NoiseModel;

/// An analytic grey scene: soft-edged level sets of two random sums of plane waves.
struct Scene {
    waves: Vec<[f64; 4]>,
    split: usize,
}

impl Scene {
    fn new(seed: u64) -> Scene {
        let mut rng = Rng::new(seed);
        let mut waves = Vec::new();
        // Coarse field (periods 18 to 200 px), then a finer one (periods 9 to 40 px).
        for (count, lo, hi) in [(8usize, 0.03f64, 0.35f64), (8, 0.15, 0.7)] {
            for _ in 0..count {
                let k = lo * (hi / lo).powf(f64::from(rng.range(0.0, 1.0)));
                let a = f64::from(rng.range(0.0, std::f32::consts::TAU));
                let phase = f64::from(rng.range(0.0, std::f32::consts::TAU));
                let amp = f64::from(rng.range(0.5, 1.0));
                waves.push([k * a.cos(), k * a.sin(), phase, amp]);
            }
        }
        Scene { waves, split: 8 }
    }

    /// The scene value at the continuous position `(x, y)` (pixel `i` covers `[i, i + 1)`).
    fn eval(&self, x: f64, y: f64) -> f64 {
        let level = |waves: &[[f64; 4]]| {
            let (mut g, mut gx, mut gy) = (0.0, 0.0, 0.0);
            for w in waves {
                let arg = w[0] * x + w[1] * y + w[2];
                g += w[3] * arg.cos();
                gx -= w[3] * w[0] * arg.sin();
                gy -= w[3] * w[1] * arg.sin();
            }
            // Signed distance to the level set, to first order, in pixels: a uniform edge width.
            let d = g / (gx * gx + gy * gy + 0.02f64 * 0.02).sqrt();
            1.0 / (1.0 + (-d / 0.55).exp())
        };
        let (coarse, fine) = self.waves.split_at(self.split);
        let v = 0.5 + 0.30 * (level(coarse) - 0.5) * 2.0 + 0.16 * (level(fine) - 0.5) * 2.0;
        v.clamp(0.04, 0.96)
    }
}

/// The lens tags for a corner shift of about `strength * 1.2` px for red and `strength * 1.0` px for
/// blue on a frame of half diagonal `hd`: 8 knots per colour, K - 1 growing with the square of the
/// radius; optionally a barrel correction of 1.5 % at the corner.
fn tags_for(strength: f64, hd: f64, distortion: bool) -> LensTags {
    let n = 8usize;
    let rho = |j: usize| (j as f64 + 0.5) / (n as f64 - 1.0);
    let knot = |amount_px: f64, j: usize| -> i16 {
        let k_minus_1 = amount_px / hd * rho(j) * rho(j);
        (k_minus_1 * 2f64.powi(21)).round() as i16
    };
    let mut ca: Vec<i16> = (0..n).map(|j| knot(1.2 * strength, j)).collect();
    ca.extend((0..n).map(|j| knot(-1.0 * strength, j)));
    let dist = if distortion {
        (0..n)
            .map(|j| (0.015 * rho(j) * rho(j) * 2f64.powi(14)).round() as i16)
            .collect()
    } else {
        vec![]
    };
    LensTags {
        vignetting: vec![],
        ca,
        distortion: dist,
    }
}

/// The source position `q` of colour `c` is sampled from the scene at the output position `p` with
/// `source_position(c, p) = q`: found by iterating `p <- p + (q - source_position(c, p))`.
fn mosaic_of_scene(scene: &Scene, geo: &LensGeometry, cfa: Cfa, noise_sigma: f32) -> Mosaic {
    let (w, h) = (geo.frame_width(), geo.frame_height());
    let mut m = Mosaic::new(w, h, cfa);
    m.place = Placement::whole(w, h);
    m.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let c = cfa.color_at(x, y).plane();
            let q = (x as f64 + 0.5, y as f64 + 0.5);
            let mut p = q;
            for _ in 0..8 {
                let s = geo.source_position(c, p.0, p.1);
                p = (p.0 + (q.0 - s.0), p.1 + (q.1 - s.1));
            }
            *v = scene.eval(p.0, p.1) as f32;
        }
    });
    if noise_sigma > 0.0 {
        let mut rng = Rng::new(99);
        for v in &mut m.data {
            *v += noise_sigma * rng.gaussian();
        }
    }
    m
}

/// One variant: the image after R9, the demosaic's CPU seconds.
struct Run {
    out: RgbImage,
    seconds: f64,
}

fn run_variant(
    m: &Mosaic,
    method: Method,
    ca: Option<&CaCurves>,
    noise: NoiseDn,
    geo_r9: &LensGeometry,
) -> Run {
    let inputs = DemosaicInputs {
        gains: [1.0; 3],
        noise,
        ca,
    };
    let t = Instant::now();
    let img = demosaic::demosaic_with(m, method, &inputs);
    let seconds = t.elapsed().as_secs_f64();
    let (cw, ch) = geo_r9.output_size();
    let out = resample::resample_lens(&img, geo_r9, Rect::whole(cw, ch)).expect("R9");
    Run { out, seconds }
}

/// The colour error at the edges of the scene: mean `|R - G|` and mean `|B - G|` over the pixels
/// (16 px border off) where the ideal green's gradient exceeds 0.03 per pixel.
fn edge_chroma(out: &RgbImage, truth_g: &[f32]) -> (f64, f64) {
    let (w, h) = (out.width, out.height);
    let (mut sr, mut sb, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 16..h - 16 {
        for x in 16..w - 16 {
            let gx = f64::from(truth_g[y * w + x + 1] - truth_g[y * w + x - 1]) * 0.5;
            let gy = f64::from(truth_g[(y + 1) * w + x] - truth_g[(y - 1) * w + x]) * 0.5;
            if gx.hypot(gy) < 0.03 {
                continue;
            }
            let i = y * w + x;
            let g = f64::from(out.plane(1)[i]);
            sr += (f64::from(out.plane(0)[i]) - g).abs();
            sb += (f64::from(out.plane(2)[i]) - g).abs();
            n += 1;
        }
    }
    (sr / n.max(1) as f64, sb / n.max(1) as f64)
}

/// `img` with all three planes equal to plane `c` of `src` (for the PSNR of one plane).
fn grey_of(src: &RgbImage, c: usize) -> RgbImage {
    let mut g = RgbImage::new(src.width, src.height);
    for k in 0..3 {
        g.plane_mut(k).copy_from_slice(src.plane(c));
    }
    g
}

fn main() {
    let mut size = (1800usize, 1200usize);
    let mut noise_sigma = 0.0f32;
    let mut seed = 2024u64;
    let mut diagnose = false;
    let mut out_dir = PathBuf::from("testdata/ra03b");
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
    let model = if noise_sigma > 0.0 {
        // Gaussian noise of this sigma on the normalised scale: all in the signal-independent term.
        NoiseModel {
            a: 0.0,
            b: f64::from(noise_sigma) * f64::from(noise_sigma),
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 0.0,
        }
    } else {
        NoiseModel {
            a: 2e-5,
            b: 1e-8,
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 1.0 / 65535.0,
        }
    };
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
        "RA-03b synthetic CA scene (seed {seed}), {fw} x {fh}, noise sigma {noise_sigma}, window of the residual: {} px patches, red and blue against green along the radius after R9 (RMS, px).\n",
        opts.patch_px
    ));
    for (strength, distortion) in [
        (0.5, false),
        (1.0, false),
        (2.0, false),
        (1.0, true),
        (2.0, true),
    ] {
        let tags = tags_for(strength, hd, distortion);
        let sw_ca = LensSwitches::default();
        let sw_no_ca = LensSwitches {
            lens_corrections: true,
            chromatic_aberration: false,
        };
        let geo_src = LensGeometry::new(fw, fh, &tags, sw_ca).expect("lens");
        let geo = geo_src.clone().with_auto_crop();
        let geo_no_ca = LensGeometry::new(fw, fh, &tags, sw_no_ca)
            .expect("lens")
            .with_crop(geo.crop());
        let curves = geo.ca_curves().expect("curves").expect("the lens has CA");
        let crop = geo.crop();
        let centre = (
            fw as f64 / 2.0 - crop.x as f64,
            fh as f64 / 2.0 - crop.y as f64,
        );
        let corner_px = curves.misregistration_px(hd);
        let mosaic = mosaic_of_scene(&scene, &geo_src, Cfa::RGGB, noise_sigma);
        // Ideal output: the scene at the output pixel centres of the cropped output.
        let (cw, ch) = geo.output_size();
        let mut truth = RgbImage::new(cw, ch);
        for y in 0..ch {
            for x in 0..cw {
                let v = scene.eval((crop.x + x) as f64 + 0.5, (crop.y + y) as f64 + 0.5) as f32;
                for c in 0..3 {
                    truth.plane_mut(c)[y * cw + x] = v;
                }
            }
        }
        let rows = [
            (
                "adaptive ratio, plain",
                run_variant(&mosaic, Method::AdaptiveRatio, None, noise, &geo),
            ),
            (
                "adaptive ratio, CA path",
                run_variant(&mosaic, Method::AdaptiveRatio, Some(&curves), noise, &geo),
            ),
            (
                "Menon",
                run_variant(&mosaic, Method::Menon, None, noise, &geo),
            ),
            (
                "Menon, R9 without CA (reference)",
                run_variant(&mosaic, Method::Menon, None, noise, &geo_no_ca),
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
                &mosaic,
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
        let truth_g = grey_of(&truth, 1);
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
                psnr(&run.out, &truth, 16, 1.0),
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
