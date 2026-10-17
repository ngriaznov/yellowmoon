//! A/B of the two R8 capture-sharpening methods on synthetic scenes (spec
//! `docs/algorithms/03-diffraction-deconvolution.md` section 6.3): Richardson-Lucy with the Auto
//! values against diffraction deconvolution, and, from f/11, the noise-gate variant of section 7.1
//! item 1.
//!
//! ```text
//! cargo run -p ym-gpu --example capture_sharpen_ab [-- <output-dir>]
//! ```
//!
//! The output directory (default `testdata/capture_sharpen_ab/synthetic`, git-ignored) gets `results.json`
//! (every number, plus the normalised edge profiles) and PNG strips of crops. The table is also
//! printed.
//!
//! Scenes: `synth::fine_text` (glyphs of 2 to 5 px), `synth::natural`, and a slanted edge at 5
//! degrees (0.05 to 0.5, neutral) with two flat halves. Each is blurred with the diffraction PSF of
//! the plan for a 3.76 um pitch and the f-number (the sharp scene convolved with the plan's own
//! kernel, mirror borders), then given Poisson-Gaussian noise with channel-correlated luma noise
//! (one normal deviate for the three channels of a pixel, in unbalanced units, as in the reference's
//! scenes) at three levels standing for ISO 100, 800 and 6400. The noise models are plausible
//! full-frame values, not measurements: a = 2.5e-5, 2e-4, 2e-3 and b = 4e-8, 3e-7, 2e-6 with
//! sigma_row 5e-5, 7e-5, 2e-4.
//!
//! Measures (all on luma with the spec's weights, interior only):
//!
//! * MSE and PSNR against the sharp scene (fine text, natural);
//! * slanted edge: MTF50 in cycles per pixel (edge spread function binned at 1/4 px along the edge
//!   normal, differentiated, Hann windowed, DFT), the 10 to 90 % rise distance, and the overshoot
//!   and undershoot as a percentage of the step (normalised profile above 1 and below 0 within 12 px
//!   of the edge);
//! * luma noise SD in two flat 40 x 40 patches, 48 px or more from the edge, relative to the input.
//!
//! Richardson-Lucy gets the Auto amount and radius for the ISO and f-number
//! (`sharpen::auto_sharpen`) and a per-pixel luma sigma from the same correlated-noise model.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_gpu::cpu::sharpen::diffraction::eval::{EdgeGeometry, edge_stats, patch_sd};
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA};
use ym_gpu::cpu::sharpen::{self, auto_sharpen};
use ym_gpu::cpu::{LUMA, NoiseDn, NoiseSigma, RgbImage, mirror, synth};
use ym_noise::model::NoiseModel;

const PITCH_UM: f64 = 3.76;
const G0: [f32; 3] = [2.0, 1.0, 1.5];
const LOW: f32 = 0.05;
const HIGH: f32 = 0.5;
const EDGE_SIZE: usize = 192;
const BORDER_PX: usize = 16;

/// splitmix64 with Irwin-Hall normals (the reference's generator).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(11_400_714_819_323_198_485);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(13_787_848_793_156_543_929);
        z = (z ^ (z >> 27)).wrapping_mul(10_723_151_780_598_845_931);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

#[derive(Clone, Copy)]
struct Iso {
    name: &'static str,
    iso: f32,
    model: NoiseModel,
}

fn isos() -> [Iso; 3] {
    let m = |a, b, sigma_row| NoiseModel {
        a,
        b,
        delta: [0.0; 4],
        sigma_row,
        q: 0.0,
    };
    [
        Iso {
            name: "ISO 100",
            iso: 100.0,
            model: m(2.5e-5, 4e-8, 5e-5),
        },
        Iso {
            name: "ISO 800",
            iso: 800.0,
            model: m(2e-4, 3e-7, 7e-5),
        },
        Iso {
            name: "ISO 6400",
            iso: 6400.0,
            model: m(2e-3, 2e-6, 2e-4),
        },
    ]
}

fn floor_var(m: &NoiseModel) -> f64 {
    m.b + m.sigma_row * m.sigma_row + m.q * m.q / 12.0
}

/// Poisson-Gaussian noise with one normal deviate shared by the three channels of a pixel.
fn add_correlated_noise(img: &mut RgbImage, m: &NoiseModel, seed: u64) {
    let mut rng = Rng(seed);
    let n = img.plane_len();
    let fv = floor_var(m);
    for i in 0..n {
        let z = rng.gauss();
        for (c, &gain) in G0.iter().enumerate() {
            let g = f64::from(gain);
            let u = f64::from(img.data[c * n + i]) / g;
            let sd = (m.a * u.max(0.0) + fv).sqrt();
            img.data[c * n + i] = (g * (u + sd * z)) as f32;
        }
    }
}

/// Convolution with the plan's kernel, mirror borders.
fn psf_blur(img: &RgbImage, plan: &DiffractionPlan) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let r = plan.radius() as isize;
    let kw = 2 * plan.radius() + 1;
    let mut out = img.clone();
    for c in 0..3 {
        let src = img.plane(c).to_vec();
        let dst = out.plane_mut(c);
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f64;
                for dy in -r..=r {
                    let sy = mirror(y as isize + dy, h);
                    for dx in -r..=r {
                        let k =
                            f64::from(plan.kernel()[(dy + r) as usize * kw + (dx + r) as usize]);
                        s += k * f64::from(src[sy * w + mirror(x as isize + dx, w)]);
                    }
                }
                dst[y * w + x] = s as f32;
            }
        }
    }
    out
}

/// A slanted edge (normal at 5 degrees from the x axis) between two neutral levels, 8 x 8 supersampled.
fn slanted_edge(size: usize) -> (RgbImage, (f64, f64), (f64, f64)) {
    let theta = 5.0f64.to_radians();
    let n = (theta.cos(), theta.sin());
    let c = (size as f64 / 2.0, size as f64 / 2.0);
    let mut img = RgbImage::new(size, size);
    let ss = 8;
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0f64;
            for sy in 0..ss {
                for sx in 0..ss {
                    let px = x as f64 + (f64::from(sx) + 0.5) / f64::from(ss);
                    let py = y as f64 + (f64::from(sy) + 0.5) / f64::from(ss);
                    let d = n.0 * (px - c.0) + n.1 * (py - c.1);
                    acc += if d > 0.0 {
                        f64::from(HIGH)
                    } else {
                        f64::from(LOW)
                    };
                }
            }
            let v = (acc / f64::from(ss * ss)) as f32;
            img.set_pixel(x, y, [v, v, v]);
        }
    }
    (img, n, c)
}

fn luma_spec(img: &RgbImage) -> Vec<f64> {
    let n = img.plane_len();
    (0..n)
        .map(|i| {
            f64::from(SPEC_LUMA[0]) * f64::from(img.data[i])
                + f64::from(SPEC_LUMA[1]) * f64::from(img.data[n + i])
                + f64::from(SPEC_LUMA[2]) * f64::from(img.data[2 * n + i])
        })
        .collect()
}

/// Luma noise sigma of the correlated-noise model for a luma with `weights`, per pixel of `img`.
fn luma_sigma(img: &RgbImage, m: &NoiseModel, weights: [f32; 3]) -> Vec<f32> {
    let d: f64 = (0..3)
        .map(|c| f64::from(weights[c]) * f64::from(G0[c]))
        .sum();
    let n = img.plane_len();
    let fv = floor_var(m);
    (0..n)
        .map(|i| {
            let y: f64 = (0..3)
                .map(|c| f64::from(weights[c]) * f64::from(img.data[c * n + i]))
                .sum();
            (d * m.a * y.max(0.0) + d * d * fv).sqrt() as f32
        })
        .collect()
}

/// The two flat 40 x 40 windows of the edge scene, 48 px or more from the edge.
fn flat_windows(size: usize) -> [(usize, usize, usize, usize); 2] {
    [(8, 76, 40, 40), (size - 48, 76, 40, 40)]
}

fn mse_vs(sharp: &RgbImage, out: &RgbImage) -> f64 {
    let (w, h) = (sharp.width, sharp.height);
    let n = w * h;
    let mut s = 0.0f64;
    let mut cnt = 0.0f64;
    for c in 0..3 {
        for y in BORDER_PX..h - BORDER_PX {
            for x in BORDER_PX..w - BORDER_PX {
                let i = c * n + y * w + x;
                s += (f64::from(sharp.data[i]) - f64::from(out.data[i])).powi(2);
                cnt += 1.0;
            }
        }
    }
    s / cnt
}

fn psnr(mse: f64) -> f64 {
    10.0 * (1.0 / mse).log10()
}

/// The methods compared, in order.
const METHODS: [&str; 4] = ["input", "rl_auto", "diffraction", "diffraction_gate"];

fn run_methods(
    noisy: &RgbImage,
    iso: &Iso,
    f_number: f64,
    plan: &DiffractionPlan,
) -> [RgbImage; 4] {
    let rl = auto_sharpen(iso.iso, 100.0, Some(f_number as f32));
    let sigma = luma_sigma(noisy, &iso.model, LUMA);
    let rl_out = sharpen::richardson_lucy_whole(noisy, &rl, &NoiseSigma::Plane(&sigma), 0);
    let d_out = plan.apply_whole(noisy);
    let g_out = plan.clone().with_noise_gate(true).apply_whole(noisy);
    [noisy.clone(), rl_out, d_out, g_out]
}

fn to_srgb(v: f32) -> u8 {
    let v = f64::from(v).clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

/// Writes a strip of crops side by side (nearest neighbour, `zoom` times), balanced linear shown
/// through the sRGB curve (not the product's rendering).
fn write_strip(
    path: &Path,
    images: &[&RgbImage],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    zoom: usize,
) {
    let (ow, oh) = ((w * zoom + 4) * images.len() - 4, h * zoom);
    let mut buf = vec![255u8; ow * oh * 3];
    for (k, img) in images.iter().enumerate() {
        for y in 0..h * zoom {
            for x in 0..w * zoom {
                let p = img.pixel(x0 + x / zoom, y0 + y / zoom);
                let o = (y * ow + k * (w * zoom + 4) + x) * 3;
                for c in 0..3 {
                    buf[o + c] = to_srgb(p[c]);
                }
            }
        }
    }
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), ow as u32, oh as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&buf)
        .expect("png data");
}

fn main() {
    let out_dir: PathBuf = std::env::args().nth(1).map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../testdata/capture_sharpen_ab/synthetic")
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&out_dir).expect("output directory");
    let f_numbers = [5.6, 8.0, 11.0, 16.0, 22.0];
    let text = synth::fine_text(192, 128, 2);
    let natural = synth::natural(192, 128, 7);
    let (edge, normal, centre) = slanted_edge(EDGE_SIZE);
    let geometry = EdgeGeometry { normal, centre };

    let mut records: Vec<Value> = Vec::new();
    let mut table = String::new();
    writeln!(
        table,
        "| f | ISO | method | text PSNR | natural PSNR | MTF50 c/px | rise 10-90 px | overshoot % | undershoot % | flat noise x |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    .expect("write");
    for &f in &f_numbers {
        let pp = DiffractionParams::new(PITCH_UM, f);
        for (ii, iso) in isos().iter().enumerate() {
            let model_dn = NoiseDn::from_model(&iso.model);
            let plan = DiffractionPlan::new(&pp, G0, &model_dn, SPEC_LUMA).expect("active plan");
            let seed = 1000 + 17 * ii as u64;
            let scene = |sharp: &RgbImage, extra: u64| {
                let mut blurred = psf_blur(sharp, &plan);
                add_correlated_noise(&mut blurred, &iso.model, seed + extra);
                blurred
            };
            let text_in = scene(&text, 1);
            let nat_in = scene(&natural, 2);
            let edge_in = scene(&edge, 3);
            let text_out = run_methods(&text_in, iso, f, &plan);
            let nat_out = run_methods(&nat_in, iso, f, &plan);
            let edge_out = run_methods(&edge_in, iso, f, &plan);
            let sd_in = patch_sd(&luma_spec(&edge_in), EDGE_SIZE, &flat_windows(EDGE_SIZE));
            for (mi, method) in METHODS.iter().enumerate() {
                if *method == "diffraction_gate" && plan.scaled_gate_factor() <= 1.0 {
                    continue; // identical to the specified gate
                }
                let t_mse = mse_vs(&text, &text_out[mi]);
                let n_mse = mse_vs(&natural, &nat_out[mi]);
                let luma = luma_spec(&edge_out[mi]);
                let es = edge_stats(&luma, EDGE_SIZE, EDGE_SIZE, &geometry, 32);
                let noise_x = patch_sd(&luma, EDGE_SIZE, &flat_windows(EDGE_SIZE)) / sd_in;
                writeln!(
                    table,
                    "| {f} | {} | {method} | {:.2} | {:.2} | {:.3} | {:.2} | {:.1} | {:.1} | {:.3} |",
                    iso.name,
                    psnr(t_mse),
                    psnr(n_mse),
                    es.mtf50,
                    es.rise,
                    es.overshoot_pct,
                    es.undershoot_pct,
                    noise_x
                )
                .expect("write");
                records.push(json!({
                    "f_number": f, "pitch_um": PITCH_UM, "iso": iso.name, "method": method,
                    "text_mse": t_mse, "text_psnr_db": psnr(t_mse),
                    "natural_mse": n_mse, "natural_psnr_db": psnr(n_mse),
                    "mtf50_cpp": es.mtf50, "rise_10_90_px": es.rise,
                    "overshoot_pct": es.overshoot_pct, "undershoot_pct": es.undershoot_pct,
                    "flat_noise_ratio": noise_x,
                    "esf": if ii == 0 && (f == 8.0 || f == 16.0 || f == 22.0) { json!(es.esf) } else { Value::Null },
                }));
            }
            // Crops for the owner's look: sharp, input, Richardson-Lucy, diffraction (and the gate).
            if (ii == 0 && (f == 8.0 || f == 16.0 || f == 22.0)) || (ii == 1 && f == 16.0) {
                let tag = format!("f{f}_{}", iso.name.replace(' ', "").to_lowercase());
                let faint = |imgs: &[RgbImage; 4], sharp: &RgbImage| -> Vec<RgbImage> {
                    let mut v = vec![sharp.clone()];
                    v.extend(imgs.iter().cloned());
                    v
                };
                let strip = |name: &str, imgs: Vec<RgbImage>, x0, y0, w, h| {
                    let refs: Vec<&RgbImage> = imgs.iter().collect();
                    write_strip(
                        &out_dir.join(format!("{name}_{tag}.png")),
                        &refs,
                        x0,
                        y0,
                        w,
                        h,
                        3,
                    );
                };
                strip("text", faint(&text_out, &text), 8, 8, 56, 40);
                strip("natural", faint(&nat_out, &natural), 70, 40, 56, 40);
                strip("edge", faint(&edge_out, &edge), 70, 80, 56, 40);
            }
        }
    }
    print!("{table}");
    let doc = json!({
        "description": "Capture-sharpening A/B: Richardson-Lucy Auto vs diffraction deconvolution on synthetic scenes (docs/algorithms/03 section 6.3)",
        "pitch_um": PITCH_UM,
        "g0": G0,
        "noise_models": isos().iter().map(|i| json!({"name": i.name, "a": i.model.a, "b": i.model.b, "sigma_row": i.model.sigma_row})).collect::<Vec<_>>(),
        "strip_order": ["sharp", "input", "rl_auto", "diffraction", "diffraction_gate"],
        "records": records,
    });
    std::fs::write(
        out_dir.join("results.json"),
        serde_json::to_string_pretty(&doc).expect("json"),
    )
    .expect("write results");
    std::fs::write(out_dir.join("table.md"), &table).expect("write table");
    eprintln!("wrote {}", out_dir.display());
}
