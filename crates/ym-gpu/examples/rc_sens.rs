//! SCRATCH sensitivity harness (never committed): spec 01 and spec 04 constants.
use std::path::PathBuf;

use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params, ShadowMatch};
use ym_gpu::cpu::demosaic::DemosaicInputs;
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::output_sharpen::OutputSharpen;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage, rc_tune, synth};
use ym_noise::model::NoiseModel;

#[path = "../tests/common/demosaic_metrics.rs"]
#[allow(dead_code)]
mod demosaic_metrics;
use demosaic_metrics::{demosaic_sets_root, linearize, load_set, synthetic_set};

fn oetf(v: f32) -> f64 {
    let v = f64::from(v).clamp(0.0, 1.0);
    255.0 * if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// mean and max |code difference| between two images, 8-bit sRGB codes, border off.
fn code_diff(a: &RgbImage, b: &RgbImage, border: usize, scale: f64) -> (f64, f64) {
    let (w, h) = (a.width, a.height);
    let (mut s, mut n, mut mx) = (0.0, 0u64, 0.0f64);
    for c in 0..3 {
        let (pa, pb) = (a.plane(c), b.plane(c));
        for y in border..h - border {
            for x in border..w - border {
                let d = (oetf((f64::from(pa[y * w + x]) / scale) as f32) - oetf((f64::from(pb[y * w + x]) / scale) as f32)).abs();
                s += d;
                n += 1;
                mx = mx.max(d);
            }
        }
    }
    (s / n as f64, mx)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(11_400_714_819_323_198_485);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(13_787_848_793_156_543_929);
        z = (z ^ (z >> 27)).wrapping_mul(10_723_151_780_598_845_931);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| (self.next() >> 11) as f64 / 9_007_199_254_740_992.0).sum::<f64>() - 6.0
    }
}

struct Scene {
    name: String,
    truth: Option<RgbImage>, // scaled truth
    mosaic: Mosaic,
    gains: [f32; 3],
    noise: NoiseDn,
    scale: f64,
    shadow: ShadowMatch,
}

fn model(a: f64, b: f64, row: f64) -> NoiseModel {
    NoiseModel { a, b, delta: [0.0; 4], sigma_row: row, q: 1.0 / 65535.0 }
}

fn scenes() -> Vec<Scene> {
    let root = demosaic_sets_root();
    let mut sets: Vec<(String, RgbImage)> = Vec::new();
    for d in ["kodak", "mcmaster"] {
        for (n, img) in load_set(&root.join(d)) {
            sets.push((format!("{d}/{n}"), linearize(&img)));
        }
    }
    sets.extend(synthetic_set());
    let mut out = Vec::new();
    let low = model(2e-5, 1e-8, 0.0);
    let high = model(5e-4, 4e-5, 1e-3);
    for (n, t) in &sets {
        // A: clean, scale 1, low-noise model
        out.push(Scene { name: format!("A {n}"), truth: Some(t.clone()), mosaic: synth::mosaic_from_rgb(t, Cfa::RGGB), gains: [1.0; 3], noise: NoiseDn::from_model(&low), scale: 1.0, shadow: ShadowMatch::Auto });
    }
    // B: dark, noisy, high model (ISO 6400-like), every 3rd image to bound time
    for (k, (n, t)) in sets.iter().enumerate().filter(|(k, _)| k % 3 == 0) {
        let s = 0.25f32;
        let mut ts = t.clone();
        for v in ts.data.iter_mut() { *v *= s; }
        let mut m = synth::mosaic_from_rgb(&ts, Cfa::RGGB);
        let mut rng = Rng(77 + k as u64);
        for v in m.data.iter_mut() {
            let x = f64::from(*v).max(0.0);
            *v = (x + (high.a * x + high.b + high.sigma_row * high.sigma_row).sqrt() * rng.gauss()) as f32;
        }
        // The model works on the signal scale 1 = white; the scene sits at 0.25 of white.
        out.push(Scene { name: format!("B {n}"), truth: Some(ts), mosaic: m, gains: [1.0; 3], noise: NoiseDn::from_model(&high), scale: 0.25, shadow: ShadowMatch::Auto });
    }
    // C: real photo crops (no truth): A1 files, RGGB assumed, as-shot gains.
    let photos = std::env::var("YM_PHOTO").ok();
    if let Some(path) = photos {
        let raw = ym_raw::decode::decode(PathBuf::from(&path)).expect("decode");
        let (blk, wht) = (raw.black[0] as f64, raw.white[0] as f64);
        let wb = raw.wb_as_shot.unwrap_or([1024.0; 4]);
        let g = wb[1] as f64;
        let gains = [(wb[0] as f64 / g) as f32, 1.0, (wb[3] as f64 / g) as f32];
        let (pw, ph) = (raw.data.width, raw.data.height);
        for (k, (fx, fy)) in [(0.5, 0.5), (0.3, 0.7)].iter().enumerate() {
            let (w, h) = (1024usize, 768usize);
            let x0 = ((pw as f64 * fx) as usize - w / 2) & !1;
            let y0 = ((ph as f64 * fy) as usize - h / 2) & !1;
            let mut data = vec![0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let v = f64::from(raw.data.data[(y0 + y) * pw + x0 + x]);
                    data[y * w + x] = ((v - blk) / (wht - blk)) as f32;
                }
            }
            let m = Mosaic::from_data(w, h, Cfa::RGGB, data).unwrap();
            // Stand-in models: low = ISO 100-like, high = ISO 6400-like (not measurements).
            out.push(Scene { name: format!("C{k} photo low-model"), truth: None, mosaic: m.clone(), gains, noise: NoiseDn::from_model(&model(2.5e-5, 4e-8, 5e-5)), scale: 1.0, shadow: ShadowMatch::Auto });
            out.push(Scene { name: format!("C{k} photo high-model"), truth: None, mosaic: m, gains, noise: NoiseDn::from_model(&model(2e-3, 2e-6, 2e-4)), scale: 1.0, shadow: ShadowMatch::On });
        }
    }
    out
}

fn run_scene(s: &Scene) -> RgbImage {
    let inputs = DemosaicInputs { gains: s.gains, noise: s.noise, ca: None };
    let p = Params { shadow_match: s.shadow, ..Params::default() };
    adaptive_ratio::demosaic(&s.mosaic, &inputs, &p).expect("demosaic")
}

fn group(name: &str) -> &str { name.split(' ').next().unwrap_or("") }

struct Cfg { label: String, set: Vec<(usize, f64)> }

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_default();
    if mode == "demosaic" { demosaic(&args[1..]) } else { output(&args[1..]) }
}

fn parse(args: &[String]) -> Vec<Cfg> {
    args.iter().map(|a| {
        let (name, cfgs) = a.split_once('@').unwrap();
        Cfg { label: name.to_owned(), set: cfgs.split(',').map(|p| { let (i, v) = p.split_once(':').unwrap(); (i.parse().unwrap(), v.parse().unwrap()) }).collect() }
    }).collect()
}

fn demosaic(args: &[String]) {
    let sc = scenes();
    rc_tune::reset();
    let base: Vec<RgbImage> = sc.iter().map(run_scene).collect();
    let groups = ["A", "B", "C0", "C1"];
    let ps = |outs: &[RgbImage]| -> Vec<f64> {
        groups.iter().map(|g| {
            let v: Vec<f64> = sc.iter().zip(outs).filter(|(s, _)| group(&s.name) == *g && s.truth.is_some()).map(|(s, o)| psnr(o, s.truth.as_ref().unwrap(), 10, s.scale)).collect();
            if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
        }).collect()
    };
    let bp = ps(&base);
    println!("BASE psnr A {:.3} B {:.3}", bp[0], bp[1]);
    for cfg in parse(args) {
        rc_tune::reset();
        for &(i, v) in &cfg.set { rc_tune::set(i, v); }
        let outs: Vec<RgbImage> = sc.iter().map(run_scene).collect();
        let p = ps(&outs);
        let mut line = format!("{}\tdPSNR A {:+.4} B {:+.4}", cfg.label, p[0] - bp[0], p[1] - bp[1]);
        for g in ["A", "B", "C"] {
            let (mut sm, mut n, mut mx) = (0.0, 0, 0.0f64);
            for ((s, o), b) in sc.iter().zip(&outs).zip(&base) {
                if s.name.starts_with(g) {
                    let (m, x) = code_diff(o, b, 10, s.scale);
                    sm += m; n += 1; mx = mx.max(x);
                }
            }
            if n > 0 { line += &format!("\t{g}: mean|d| {:.4} max|d| {:.1}", sm / f64::from(n), mx); }
        }
        println!("{line}");
    }
    rc_tune::reset();
}

fn output(args: &[String]) {
    // display-referred images: Kodak + McMaster linear light; the stage works on linear.
    let root = demosaic_sets_root();
    let mut imgs = Vec::new();
    for d in ["kodak", "mcmaster"] {
        for (n, img) in load_set(&root.join(d)) { imgs.push((n, linearize(&img))); }
    }
    imgs.truncate(24);
    let scales = [1.0f32, 0.33];
    let run = |imgs: &[(String, RgbImage)], tweak: &dyn Fn(&mut OutputSharpen)| -> Vec<RgbImage> {
        let mut v = Vec::new();
        for s in scales {
            for (_, i) in imgs {
                let mut o = OutputSharpen::screen();
                tweak(&mut o);
                v.push(o.apply(i, s).expect("sharpen"));
            }
        }
        v
    };
    rc_tune::reset();
    let base = run(&imgs, &|_| {});
    println!("BASE ok n={}", base.len());
    let report = |label: &str, outs: &[RgbImage]| {
        let n = imgs.len();
        let mut line = label.to_owned();
        for (k, s) in scales.iter().enumerate() {
            let (mut sm, mut mx) = (0.0, 0.0f64);
            for j in 0..n {
                let (m, x) = code_diff(&outs[k * n + j], &base[k * n + j], 4, 1.0);
                sm += m; mx = mx.max(x);
            }
            line += &format!("\tscale {s}: mean|d| {:.4} max|d| {:.1}", sm / n as f64, mx);
        }
        println!("{line}");
    };
    // constants via rc, preset fields via the `p:` prefix
    for a in args {
        let (name, cfgs) = a.split_once('@').unwrap();
        rc_tune::reset();
        let mut radius = None; let mut amount = None; let mut halo = None; let mut thr = None;
        for p in cfgs.split(',') {
            let (i, v) = p.split_once(':').unwrap();
            let v: f64 = v.parse().unwrap();
            match i { "radius" => radius = Some(v as f32), "amount" => amount = Some(v as f32), "halo" => halo = Some(v as f32), "thr" => thr = Some(v as f32), _ => rc_tune::set(i.parse().unwrap(), v) }
        }
        let outs = run(&imgs, &|o| { if let Some(r) = radius { o.radius_px = r } if let Some(r) = amount { o.amount_pct = r } if let Some(r) = halo { o.halo_pct = r } if let Some(r) = thr { o.threshold_levels = r } });
        report(name, &outs);
    }
    rc_tune::reset();
}
