//! The fit harness of spec 03, diffraction deconvolution (`docs/29-raw-constants.md` section 2.3, task RC-05):
//! evaluates a `Tuning` of the deconvolution on the scenes of `capture_sharpen_ab` and prints one JSON line
//! per evaluation. Measurement only: nothing here changes a default and nothing is written to disk.
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example fit_diffraction \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' -- \
//!     [--from oracle|default] [--set name=value]... [--rel name=0.9,1.1]... \
//!     [--eval "label@name=v,name=v"]... [--held] [--list] [--weights FILE] [--kodak] [--lamps]
//! ```
//!
//! Arguments are those of `fit_demosaic` (`--from`, `--set`, `--rel`, `--eval`, `--held`, `--list`); the
//! fields are those of `sharpen::diffraction::Tuning`. The base is the starting `Tuning`.
//!
//! The scenes are `capture_sharpen_ab`'s, all ours (no dataset is read): fine text, a natural scene and a
//! slanted edge at 5 degrees (0.05 to 0.5), blurred with the Airy PSF of the **reference** optics
//! (`Tuning::ORACLE`, so that a moved constant is tested against the same input) for a 3.76 um pitch at
//! f/8 and f/16, with correlated Poisson-Gaussian noise at the ISO 100 and ISO 800 stand-in models, then
//! restored by the plan under test. Four cases form the fit set; `--held` adds the same four with a second
//! noise seed (held out), reported as `mean_delta_held`.
//!
//! Per case: the PSNR of the restored fine text and natural scene against the sharp scene, and on the edge
//! the MTF50, the 10 to 90 % rise, the overshoot and the noise of the flat halves relative to the input
//! (`docs/29` F-05: the edge numbers next to the PSNR). `mean_delta` is the mean over the cases of the
//! change against the base, and `max_dout` the largest change of any output sample on the scale of
//! an 8-bit picture of range 0.5.
//!
//! RC-11 adds three inputs. `--weights FILE` reads a spectral weight per wavelength (a JSON array of 159
//! numbers, 461 to 619 nm; D1) into the starting `Tuning`. `--kodak` adds the 24 Kodak images of
//! `testdata/demosaic/kodak` (`YM_TESTDATA` moves the root), a centre crop of 160 x 112 of each in linear
//! light, blurred with the reference optics at the four cases and restored: the mean PSNR against the
//! sharp crop, as `kodak` (not committed, nothing written). `--lamps` adds a scene of two lamps (a disc
//! of radius 6 and one of radius 10 on a dark gradient, 1.6 times white before the clip, tinted so that
//! the channels clip at different places), blurred, noised and clipped at white: the PSNR against the
//! clipped scene, as `lamps` (the scene is ours; it makes the clip penalty D7 act).

use serde_json::{Value, json};
use ym_gpu::cpu::sharpen::diffraction::eval::{EdgeGeometry, edge_stats, patch_sd};
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA, Tuning};
use ym_gpu::cpu::{NoiseDn, RgbImage, mirror, synth};
use ym_noise::model::NoiseModel;

#[path = "../tests/common/demosaic_metrics.rs"]
mod demosaic_metrics;
#[path = "../tests/common/fit.rs"]
mod fit;

use demosaic_metrics::{demosaic_sets_root, linearize, load_set};

use fit::{Common, Rng, Start, data_note, emit, mean, num};

const PITCH_UM: f64 = 3.76;
const G0: [f32; 3] = [2.0, 1.0, 1.5];
const LOW: f32 = 0.05;
const HIGH: f32 = 0.5;
const EDGE_SIZE: usize = 192;
const BORDER_PX: usize = 16;

#[derive(Clone, Copy)]
struct Iso {
    name: &'static str,
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
            model: m(2.5e-5, 4e-8, 5e-5),
        },
        Iso {
            name: "ISO 800",
            model: m(2e-4, 3e-7, 7e-5),
        },
        Iso {
            name: "ISO 6400",
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

/// One scene set of one case: the blurred, noisy inputs and what to compare them with.
pub struct Case {
    pub name: String,
    pub held: bool,
    pub f: f64,
    pub model: NoiseModel,
    pub text_in: RgbImage,
    pub nat_in: RgbImage,
    pub edge_in: RgbImage,
    pub sd_in: f64,
}

/// The measures of one restored case.
#[derive(Clone, Copy)]
pub struct Res {
    pub text: f64,
    pub nat: f64,
    pub mtf: f64,
    pub rise: f64,
    pub over: f64,
    pub noise: f64,
}

pub struct Scenes {
    pub text: RgbImage,
    pub natural: RgbImage,
    pub geometry: EdgeGeometry,
    pub cases: Vec<Case>,
}

pub fn build_scenes(held: bool) -> Scenes {
    let text = synth::fine_text(192, 128, 2);
    let natural = synth::natural(192, 128, 7);
    let (edge, normal, centre) = slanted_edge(EDGE_SIZE);
    let geometry = EdgeGeometry { normal, centre };
    let mut cases = Vec::new();
    for second_seed in [false, true] {
        if second_seed && !held {
            continue;
        }
        for &f in &[8.0f64, 16.0] {
            for (ii, iso) in isos().iter().enumerate().take(2) {
                let model_dn = NoiseDn::from_model(&iso.model);
                // The truth optics: the reference constants, so every evaluation sees the same input.
                let truth = DiffractionPlan::new(
                    &DiffractionParams::new(PITCH_UM, f).with_tuning(&Tuning::ORACLE),
                    G0,
                    &model_dn,
                    SPEC_LUMA,
                )
                .expect("active plan");
                let seed = 1000 + 17 * ii as u64 + if second_seed { 5000 } else { 0 };
                let scene = |sharp: &RgbImage, extra: u64| {
                    let mut b = psf_blur(sharp, &truth);
                    add_correlated_noise(&mut b, &iso.model, seed + extra);
                    b
                };
                let edge_in = scene(&edge, 3);
                let sd_in = patch_sd(&luma_spec(&edge_in), EDGE_SIZE, &flat_windows(EDGE_SIZE));
                cases.push(Case {
                    name: format!(
                        "f/{f} {}{}",
                        iso.name,
                        if second_seed { " seed 2" } else { "" }
                    ),
                    held: second_seed,
                    f,
                    model: iso.model,
                    text_in: scene(&text, 1),
                    nat_in: scene(&natural, 2),
                    edge_in,
                    sd_in,
                });
            }
        }
    }
    Scenes {
        text,
        natural,
        geometry,
        cases,
    }
}

/// Restores every case with `tuning`; returns the measures and the three outputs of each case.
pub fn run(sc: &Scenes, tuning: &Tuning) -> Vec<(Res, [RgbImage; 3])> {
    // One leaked copy per evaluation: `DiffractionParams` holds a `&'static Tuning`. An example run
    // makes a few dozen evaluations at most, a few hundred bytes each.
    let tuning: &'static Tuning = Box::leak(Box::new(*tuning));
    sc.cases
        .iter()
        .map(|c| {
            let params = DiffractionParams::new(PITCH_UM, c.f).with_tuning(tuning);
            let plan = DiffractionPlan::new(&params, G0, &NoiseDn::from_model(&c.model), SPEC_LUMA)
                .expect("plan");
            let t_out = plan.apply_whole(&c.text_in);
            let n_out = plan.apply_whole(&c.nat_in);
            let e_out = plan.apply_whole(&c.edge_in);
            let luma = luma_spec(&e_out);
            let es = edge_stats(&luma, EDGE_SIZE, EDGE_SIZE, &sc.geometry, 32);
            let noise = patch_sd(&luma, EDGE_SIZE, &flat_windows(EDGE_SIZE)) / c.sd_in;
            let res = Res {
                text: psnr(mse_vs(&sc.text, &t_out)),
                nat: psnr(mse_vs(&sc.natural, &n_out)),
                mtf: es.mtf50,
                rise: es.rise,
                over: es.overshoot_pct,
                noise,
            };
            (res, [t_out, n_out, e_out])
        })
        .collect()
}

/// The scene of two lamps, in the balanced units of the stage (`G0` times the raw radiance, so that the
/// raw white of channel `c` is `G0[c]`): radiance up to 1.6 of white, so the discs clip once seen
/// through the optics, and in the second lamp all three channels pass the clip threshold.
fn lamps_radiance(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let lamps = [
        (w as f64 * 0.32, h as f64 * 0.5, 6.0, [1.6f32, 1.4, 0.8]),
        (w as f64 * 0.70, h as f64 * 0.45, 10.0, [1.5, 1.6, 1.3]),
    ];
    for y in 0..h {
        for x in 0..w {
            let g = 0.03 + 0.06 * (x as f32 / w as f32) + 0.02 * (y as f32 / h as f32);
            let mut p = [g, 0.9 * g, 0.7 * g];
            let ss = 4;
            for (cx, cy, r, c) in &lamps {
                let mut cover = 0.0f32;
                for sy in 0..ss {
                    for sx in 0..ss {
                        let px = x as f64 + (f64::from(sx) + 0.5) / f64::from(ss);
                        let py = y as f64 + (f64::from(sy) + 0.5) / f64::from(ss);
                        if (px - cx).hypot(py - cy) <= *r {
                            cover += 1.0;
                        }
                    }
                }
                let cover = cover / (ss * ss) as f32;
                for k in 0..3 {
                    p[k] = p[k] * (1.0 - cover) + c[k] * cover;
                }
            }
            img.set_pixel(x, y, [p[0] * G0[0], p[1] * G0[1], p[2] * G0[2]]);
        }
    }
    img
}

/// The raw clip: channel `c` saturates at `G0[c]` in balanced units.
fn clip_white(img: &mut RgbImage) {
    let n = img.plane_len();
    for (c, white) in G0.iter().enumerate() {
        for v in &mut img.data[c * n..(c + 1) * n] {
            *v = v.min(*white);
        }
    }
}

/// One Kodak or lamps input: the sharp truth, the blurred noisy input and the case it belongs to.
pub struct Extra {
    pub set: &'static str,
    pub case: usize,
    pub sharp: RgbImage,
    pub input: RgbImage,
}

/// The Kodak crops and the lamps, blurred with the reference optics at the four cases of `sc`.
pub fn build_extras(sc: &Scenes, kodak: bool, lamps: bool) -> Vec<Extra> {
    let mut sharps: Vec<(&'static str, RgbImage, bool)> = Vec::new();
    if kodak {
        for (_, img) in load_set(&demosaic_sets_root().join("kodak")) {
            let lin = linearize(&img);
            let (w, h) = (160usize, 112usize);
            let (x0, y0) = ((lin.width - w) / 2, (lin.height - h) / 2);
            let mut c = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    c.set_pixel(x, y, lin.pixel(x0 + x, y0 + y));
                }
            }
            sharps.push(("kodak", c, false));
        }
    }
    if lamps {
        sharps.push(("lamps", lamps_radiance(160, 112), true));
    }
    let mut out = Vec::new();
    for (ci, c) in sc.cases.iter().enumerate() {
        if c.held {
            continue;
        }
        let truth = DiffractionPlan::new(
            &DiffractionParams::new(PITCH_UM, c.f).with_tuning(&Tuning::ORACLE),
            G0,
            &NoiseDn::from_model(&c.model),
            SPEC_LUMA,
        )
        .expect("active plan");
        for (k, (set, sharp, clip)) in sharps.iter().enumerate() {
            let mut b = psf_blur(sharp, &truth);
            add_correlated_noise(&mut b, &c.model, 7000 + 31 * ci as u64 + k as u64);
            let mut truth_img = sharp.clone();
            if *clip {
                clip_white(&mut b);
                clip_white(&mut truth_img);
            }
            out.push(Extra {
                set,
                case: ci,
                sharp: truth_img,
                input: b,
            });
        }
    }
    out
}

/// The mean PSNR against the sharp scene of every extra of `set`, restored with `tuning`.
pub fn run_extras(sc: &Scenes, extras: &[Extra], set: &str, tuning: &Tuning) -> Option<(f64, f64)> {
    let tuning: &'static Tuning = Box::leak(Box::new(*tuning));
    let (mut sum, mut n, mut sum_in) = (0.0f64, 0.0f64, 0.0f64);
    for e in extras.iter().filter(|e| e.set == set) {
        let c = &sc.cases[e.case];
        let params = DiffractionParams::new(PITCH_UM, c.f).with_tuning(tuning);
        let plan = DiffractionPlan::new(&params, G0, &NoiseDn::from_model(&c.model), SPEC_LUMA)
            .expect("plan");
        let mut o = plan.apply_whole(&e.input);
        if set == "lamps" {
            clip_white(&mut o);
        }
        sum += psnr(mse_vs(&e.sharp, &o));
        sum_in += psnr(mse_vs(&e.sharp, &e.input));
        n += 1.0;
    }
    (n > 0.0).then(|| (sum / n, sum_in / n))
}

fn case_json(c: &Case, r: &Res) -> Value {
    json!({
        "name": c.name, "held": c.held,
        "text_psnr_db": num(r.text), "natural_psnr_db": num(r.nat), "mtf50_cpp": num(r.mtf),
        "rise_10_90_px": num(r.rise), "overshoot_pct": num(r.over), "flat_noise_x": num(r.noise),
    })
}

/// The mean change against the base over the cases with `held`, and the largest change of any sample.
pub fn deltas(
    sc: &Scenes,
    res: &[(Res, [RgbImage; 3])],
    base: &[(Res, [RgbImage; 3])],
    held: bool,
) -> Option<Value> {
    let mut d = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut dmax = 0.0f64;
    for ((c, (r, o)), (b, bo)) in sc.cases.iter().zip(res).zip(base) {
        if c.held != held {
            continue;
        }
        d[0].push(r.text - b.text);
        d[1].push(r.nat - b.nat);
        d[2].push(r.mtf - b.mtf);
        d[3].push(r.over - b.over);
        d[4].push(r.noise - b.noise);
        d[5].push(r.rise - b.rise);
        for (x, y) in o.iter().zip(bo.iter()) {
            for (p, q) in x.data.iter().zip(y.data.iter()) {
                dmax = dmax.max((f64::from(*p) - f64::from(*q)).abs() / 0.5 * 255.0);
            }
        }
    }
    if d[0].is_empty() {
        return None;
    }
    Some(json!({
        "text_psnr_db": mean(&d[0]), "natural_psnr_db": mean(&d[1]), "mtf50_cpp": mean(&d[2]),
        "overshoot_pct": mean(&d[3]), "flat_noise_x": mean(&d[4]), "rise_10_90_px": mean(&d[5]),
        "max_dout": dmax,
    }))
}

fn usage(why: &str) -> ! {
    eprintln!("fit_diffraction: {why}");
    eprintln!(
        "usage: fit_diffraction [--from oracle|default] [--set name=value]... [--rel name=f1,f2]... \
         [--eval \"label@name=v,name=v\"]... [--held] [--list]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut common = Common::default();
    let mut extra: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match common.take(&args, &mut i) {
            Ok(true) => {}
            Ok(false) => match args[i].as_str() {
                "--kodak" | "--lamps" => extra.push(args[i].clone()),
                "--weights" => {
                    extra.push(args[i].clone());
                    i += 1;
                    extra.push(
                        args.get(i)
                            .cloned()
                            .unwrap_or_else(|| usage("--weights needs a file")),
                    );
                }
                _ => usage(&format!("unknown argument `{}`", args[i])),
            },
            Err(e) => usage(&e),
        }
        i += 1;
    }
    let start = match common.start {
        Start::Oracle => Tuning::ORACLE,
        Start::Default => Tuning::DEFAULT,
    };
    if common.list {
        emit(&fit::list_line("diffraction", &start));
        return;
    }
    let mut start = start;
    let mut want_kodak = false;
    let mut want_lamps = false;
    for (k, a) in extra.iter().enumerate() {
        match a.as_str() {
            "--kodak" => want_kodak = true,
            "--lamps" => want_lamps = true,
            "--weights" => {
                let path = extra
                    .get(k + 1)
                    .unwrap_or_else(|| usage("--weights needs a file"));
                let text = std::fs::read_to_string(path).unwrap_or_else(|e| usage(&e.to_string()));
                let v: Vec<f64> =
                    serde_json::from_str(&text).unwrap_or_else(|e| usage(&e.to_string()));
                let table: [f64; 159] = v
                    .try_into()
                    .unwrap_or_else(|_| usage("--weights: 159 numbers"));
                start.weight_table = Some(Box::leak(Box::new(table)));
            }
            _ => {}
        }
    }
    let evals = common.evaluations(&start).unwrap_or_else(|e| usage(&e));
    let sc = build_scenes(common.held);
    let extras = build_extras(&sc, want_kodak, want_lamps);
    let kodak_n = extras.iter().filter(|e| e.set == "kodak").count() / 4;
    let mut used = vec![(
        "synthetic scenes of capture_sharpen_ab (fine text, natural, slanted edge; ours, seeded)"
            .to_owned(),
        3,
    )];
    if want_kodak {
        used.push(("kodak (centre crops, evaluation only)".to_owned(), kodak_n));
    }
    if want_lamps {
        used.push(("lamps (ours, procedural)".to_owned(), 1));
    }
    emit(&data_note("fit_diffraction", &used, &[]));
    let base = run(&sc, &start);
    let from = if common.start == Start::Oracle {
        "oracle"
    } else {
        "default"
    };
    let line = |label: &str,
                set: Value,
                res: &[(Res, [RgbImage; 3])],
                is_base: bool,
                t: &Tuning| {
        let mut v = json!({
            "kind": "eval", "example": "fit_diffraction", "label": label, "from": from, "set": set,
            "cases": sc.cases.iter().zip(res).map(|(c, (r, _))| case_json(c, r)).collect::<Vec<_>>(),
        });
        for name in ["kodak", "lamps"] {
            if let Some((p, p_in)) = run_extras(&sc, &extras, name, t) {
                v[name] = json!({"psnr_db": num(p), "input_psnr_db": num(p_in)});
            }
        }
        if !is_base {
            v["mean_delta"] = deltas(&sc, res, &base, false).unwrap_or(Value::Null);
            v["max_dout"] = v["mean_delta"]["max_dout"].clone();
            if let Some(h) = deltas(&sc, res, &base, true) {
                v["mean_delta_held"] = h;
            }
        }
        emit(&v);
    };
    line("base", json!({}), &base, true, &start);
    for (e, t) in &evals {
        let res = run(&sc, t);
        let set: std::collections::BTreeMap<_, _> = e.sets.iter().cloned().collect();
        line(&e.label, json!(set), &res, false, t);
    }
}
