//! SCRATCH sensitivity of spec 03 constants (never committed).
use ym_gpu::cpu::rc_tune;
use ym_gpu::cpu::sharpen::diffraction::eval::{EdgeGeometry, edge_stats, patch_sd};
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA};
use ym_gpu::cpu::{NoiseDn, RgbImage, mirror, synth};
use ym_noise::model::NoiseModel;
#[allow(unused_imports)]
use ym_gpu::cpu::LUMA;
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

struct Res { text: f64, nat: f64, mtf: f64, over: f64, noise: f64 }

fn run(cfg: &[(usize, f64)], idx: u32) -> Vec<(String, Res, Vec<RgbImage>)> {
    rc_tune::reset();
    for &(i, v) in cfg { rc_tune::set(i, v); }
    let text = synth::fine_text(192, 128, 2);
    let natural = synth::natural(192, 128, 7);
    let (edge, normal, centre) = slanted_edge(EDGE_SIZE);
    let geometry = EdgeGeometry { normal, centre };
    let mut out = Vec::new();
    for &f in &[8.0f64, 16.0] {
        for (ii, iso) in isos().iter().enumerate().take(2) {
            // the geometry cache is keyed by pitch: a different pitch bit pattern per config
            let pitch = PITCH_UM + f64::from(idx) * 1e-9;
            let pp = DiffractionParams::new(pitch, f);
            let model_dn = NoiseDn::from_model(&iso.model);
            let plan = DiffractionPlan::new(&pp, G0, &model_dn, SPEC_LUMA).expect("plan");
            let seed = 1000 + 17 * ii as u64;
            let scene = |sharp: &RgbImage, extra: u64| {
                let mut b = psf_blur(sharp, &BASE.with(|b| b.borrow().get(&(f.to_bits())).cloned().unwrap()));
                add_correlated_noise(&mut b, &iso.model, seed + extra);
                b
            };
            let t_in = scene(&text, 1);
            let n_in = scene(&natural, 2);
            let e_in = scene(&edge, 3);
            let t_out = plan.apply_whole(&t_in);
            let n_out = plan.apply_whole(&n_in);
            let e_out = plan.apply_whole(&e_in);
            let sd_in = patch_sd(&luma_spec(&e_in), EDGE_SIZE, &flat_windows(EDGE_SIZE));
            let luma = luma_spec(&e_out);
            let es = edge_stats(&luma, EDGE_SIZE, EDGE_SIZE, &geometry, 32);
            let noise = patch_sd(&luma, EDGE_SIZE, &flat_windows(EDGE_SIZE)) / sd_in;
            out.push((format!("f/{f} {}", iso.name), Res { text: psnr(mse_vs(&text, &t_out)), nat: psnr(mse_vs(&natural, &n_out)), mtf: es.mtf50, over: es.overshoot_pct, noise }, vec![t_out, n_out, e_out]));
        }
    }
    out
}
thread_local! { static BASE: std::cell::RefCell<std::collections::HashMap<u64, DiffractionPlan>> = Default::default(); }

fn main() {
    // Baseline optics plans (default constants): they blur the scenes in every config.
    rc_tune::reset();
    let model = isos()[0].model;
    for &f in &[8.0f64, 16.0] {
        let p = DiffractionPlan::new(&DiffractionParams::new(PITCH_UM, f), G0, &NoiseDn::from_model(&model), SPEC_LUMA).unwrap();
        BASE.with(|b| b.borrow_mut().insert(f.to_bits(), p));
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    // args: "idx:value,idx:value" groups separated by space; "base" for none.
    let base = run(&[], 0);
    let psn = |r: &Res| format!("text {:.2} nat {:.2} mtf50 {:.3} over {:.1}% noise {:.3}", r.text, r.nat, r.mtf, r.over, r.noise);
    for (n, r, _) in &base { println!("BASE {n}: {}", psn(r)); }
    for (k, a) in args.iter().enumerate() {
        let (label, spec) = a.split_once('=').map_or((a.as_str(), a.as_str()), |_| (a.as_str(), a.as_str()));
        let _ = spec;
        let (name, cfgs) = a.split_once('@').unwrap();
        let cfg: Vec<(usize, f64)> = cfgs.split(',').map(|p| { let (i, v) = p.split_once(':').unwrap(); (i.parse().unwrap(), v.parse().unwrap()) }).collect();
        let res = run(&cfg, k as u32 + 1);
        let mut dt = 0.0; let mut dn = 0.0; let mut dm = 0.0; let mut dmax = 0.0f64; let mut dnoise = 0.0; let mut dov = 0.0;
        for ((_, r, o), (_, b, bo)) in res.iter().zip(base.iter()) {
            dt += r.text - b.text; dn += r.nat - b.nat; dm += r.mtf - b.mtf; dnoise += r.noise - b.noise; dov += r.over - b.over;
            for (x, y) in o.iter().zip(bo.iter()) {
                for (p, q) in x.data.iter().zip(y.data.iter()) {
                    let d = (f64::from(*p) - f64::from(*q)).abs() / 0.5 * 255.0; // code-ish on a 0..0.5 scale
                    dmax = dmax.max(d);
                }
            }
        }
        let n = res.len() as f64;
        let _ = label;
        println!("{name}\ttext {:+.3} dB\tnat {:+.3} dB\tmtf50 {:+.4}\tover {:+.2}%\tnoise {:+.4}\tmax|dout| {:.2} (8-bit-of-0.5)", dt / n, dn / n, dm / n, dov / n, dnoise / n, dmax);
    }
}
