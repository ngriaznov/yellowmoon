//! RA-06: the log-guided fast demosaic (`Method::LogGuided`,
//! `docs/algorithms/02-log-guided-demosaic.md`).
//!
//! * `vectors_*`: the 16 vectors of `tests/cpu_data/log_guided_demosaic/` (made by the
//!   development oracle, never edited). The fixed-point form is checked against `expected` with the
//!   vector's `tolerance` (one working-scale code), the float form against `expected_float` with
//!   `tolerance_float`. The actual maxima are printed (`--nocapture`).
//! * The property tests of spec section 6.2: flat fields, tiling, CFA-phase and mirror
//!   equivariance, receptive field, scale, native passthrough, headroom, fixed against float,
//!   the exhaustive bounds of `log2_fixed` and `exp2_fixed`.
//! * Determinism (section 6.3): banded against whole, one thread against four, byte for byte.
//! * The `Method::LogGuided` dispatch: halo, noise, small buffers.

use std::path::Path;

use ym_gpu::cpu::demosaic::log_guided::{self, LogGuidedParams};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::synth::{self, Rng};
use ym_gpu::cpu::{Cfa, CfaColor, Mosaic, NoiseDn, Rect, RgbImage, mirror};
use ym_noise::model::NoiseModel;

// ---------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------

struct Vector {
    name: String,
    guide_detail: i32,
    headroom_log2: u8,
    noise_model: NoiseModel,
    derived: serde_json::Value,
    mosaic: Mosaic,
    expected: Vec<f32>,
    expected_float: Vec<f32>,
    tolerance: f64,
    tolerance_float: f64,
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

fn load_vectors() -> Vec<Vector> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/log_guided_demosaic");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).expect("vector file"))
                    .expect("vector json");
            let n = &v["params"]["noise"];
            let mut noise_model =
                NoiseModel::new(n["a"].as_f64().unwrap(), n["b"].as_f64().unwrap());
            noise_model.sigma_row = n["sigma_row"].as_f64().unwrap();
            noise_model.q = n["q"].as_f64().unwrap();
            let input = &v["input"];
            let (w, h) = (
                input["width"].as_u64().unwrap() as usize,
                input["height"].as_u64().unwrap() as usize,
            );
            let cfa = Cfa::from_name(input["cfa"].as_str().unwrap()).expect("cfa name");
            let mosaic = Mosaic::from_data(w, h, cfa, floats(&input["data"])).expect("mosaic");
            Vector {
                name: v["name"].as_str().unwrap().to_owned(),
                guide_detail: v["params"]["guide_detail"].as_i64().unwrap() as i32,
                headroom_log2: v["params"]["headroom_log2"].as_u64().unwrap() as u8,
                noise_model,
                derived: v["params"]["derived"].clone(),
                mosaic,
                expected: floats(&v["expected"]["data"]),
                expected_float: floats(&v["expected_float"]["data"]),
                tolerance: v["tolerance"].as_f64().unwrap(),
                tolerance_float: v["tolerance_float"].as_f64().unwrap(),
            }
        })
        .collect()
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "plane lengths");
    got.iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn there_are_sixteen_vectors() {
    assert_eq!(load_vectors().len(), 16);
}

#[test]
fn vectors_derived_values_match() {
    for v in load_vectors() {
        let noise = NoiseDn::from_model(&v.noise_model);
        let fixed = log_guided::derive(noise, v.guide_detail, v.headroom_log2);
        let float = log_guided::derive(noise, v.guide_detail, 0);
        let d = &v.derived;
        let name = &v.name;
        assert_eq!(fixed.headroom, d["headroom"].as_f64().unwrap(), "{name}");
        assert_eq!(fixed.p, d["p"].as_u64().unwrap() as u32, "{name} p");
        assert_eq!(
            float.p,
            d["p_float_form"].as_u64().unwrap() as u32,
            "{name} p (float form)"
        );
        assert_eq!(fixed.k, d["k"].as_i64().unwrap() as i32, "{name} k");
        let kernel: Vec<i32> = d["kernel"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(fixed.kernel.to_vec(), kernel, "{name} kernel");
        let r = d["read_noise_dn"].as_f64().unwrap();
        assert!(
            (fixed.read_noise_dn - r).abs() <= 1e-7 * r.abs().max(1.0),
            "{name}: r_dn {} against {r}",
            fixed.read_noise_dn
        );
    }
}

#[test]
fn vectors_fixed_form() {
    let mut worst = 0.0f64;
    for v in load_vectors() {
        let noise = NoiseDn::from_model(&v.noise_model);
        let out = log_guided::demosaic_fixed(&v.mosaic, noise, v.guide_detail, v.headroom_log2);
        assert_eq!((out.width, out.height), (v.mosaic.width, v.mosaic.height));
        let diff = max_abs_diff(&out.data, &v.expected);
        println!(
            "{:<22} fixed: max |diff| {diff:.3e} = {:.4} codes (tolerance {:.3e})",
            v.name,
            diff * 65535.0 / (1u32 << v.headroom_log2) as f64,
            v.tolerance
        );
        assert!(
            diff <= v.tolerance,
            "{}: fixed form differs by {diff} (tolerance {})",
            v.name,
            v.tolerance
        );
        worst = worst.max(diff);
    }
    println!("fixed form, worst over the vectors: {worst:.3e}");
}

#[test]
fn vectors_float_form() {
    let mut worst = 0.0f64;
    for v in load_vectors() {
        let noise = NoiseDn::from_model(&v.noise_model);
        let out = log_guided::demosaic_float(&v.mosaic, noise, v.guide_detail);
        let diff = max_abs_diff(&out.data, &v.expected_float);
        println!(
            "{:<22} float: max |diff| {diff:.3e} (tolerance {:.0e})",
            v.name, v.tolerance_float
        );
        assert!(
            diff <= v.tolerance_float,
            "{}: float form differs by {diff} (tolerance {})",
            v.name,
            v.tolerance_float
        );
        worst = worst.max(diff);
    }
    println!("float form, worst over the vectors: {worst:.3e}");
}

#[test]
fn vectors_through_the_method_dispatch_with_the_default_detail() {
    // `Method::LogGuided` is the float form with guide_detail 0 and the noise of the inputs; the
    // vectors with detail 0 therefore reproduce through `demosaic_with`, banded or whole.
    let vectors = load_vectors();
    let detail_zero: Vec<&Vector> = vectors.iter().filter(|v| v.guide_detail == 0).collect();
    // Thirteen of the sixteen vectors have detail 0 (the others are `detail_plus100`,
    // `detail_minus50` and `tiny_6x5`): the filter must not be vacuous.
    assert_eq!(detail_zero.len(), 13);
    for v in detail_zero {
        let inputs = DemosaicInputs {
            noise: NoiseDn::from_model(&v.noise_model),
            ..DemosaicInputs::default()
        };
        for out in [
            demosaic::demosaic_with(&v.mosaic, Method::LogGuided, &inputs),
            demosaic::demosaic_whole_with(&v.mosaic, Method::LogGuided, &inputs),
        ] {
            let diff = max_abs_diff(&out.data, &v.expected_float);
            assert!(
                diff <= v.tolerance_float,
                "{}: dispatch differs by {diff}",
                v.name
            );
        }
    }
}

#[test]
fn the_default_method_is_unchanged_and_log_guided_is_listed_once() {
    // RA-06 adds a method; it does not change the default of R6 (the owner decides, RA-H4).
    assert_eq!(demosaic::DEFAULT_METHOD, Method::Menon);
    assert_eq!(
        Method::ALL
            .iter()
            .filter(|&&m| m == Method::LogGuided)
            .count(),
        1
    );
    assert_eq!(Method::LogGuided.name(), "log_guided");
}

#[test]
fn the_method_ignores_the_gains_and_the_ca_curves_of_the_inputs() {
    // The input is the balanced mosaic: nothing is divided out, so the gains do not matter. Only
    // the read noise of the inputs is read.
    let m = mosaic_of(&synth::natural(70, 66, 2), Cfa::GBRG);
    let noise = NoiseDn::from_model(&NoiseModel::new(2e-4, 4e-8));
    let plain = DemosaicInputs {
        noise,
        ..DemosaicInputs::default()
    };
    let gained = DemosaicInputs {
        gains: [2.1, 1.0, 1.7],
        noise,
        ca: None,
    };
    for out in [
        demosaic::demosaic_with(&m, Method::LogGuided, &gained),
        demosaic::demosaic_whole_with(&m, Method::LogGuided, &gained),
    ] {
        assert!(bits_equal(
            &out,
            &demosaic::demosaic_with(&m, Method::LogGuided, &plain)
        ));
    }
}

// ---------------------------------------------------------------------------------------------
// Scenes (ports of the oracle's, so the property tests run on the oracle's own inputs)
// ---------------------------------------------------------------------------------------------

fn put(img: &mut RgbImage, x: usize, y: usize, rgb: [f64; 3]) {
    img.set_pixel(x, y, [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]);
}

fn texture(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let mut r = 0.10 + 0.25 * fx;
            let mut g = 0.12 + 0.20 * fy;
            let mut b = 0.18 + 0.10 * (1.0 - fx);
            if x as f64 + 0.6 * y as f64 > 0.9 * w as f64 {
                r *= 2.2;
                g *= 2.0;
                b *= 1.4;
            }
            if y >= h / 8 && y < h / 3 && x >= w / 2 {
                let s = if (x / 2) % 2 == 0 { 1.6 } else { 0.6 };
                r *= s;
                g *= s;
                b *= s;
            }
            let (ddx, ddy) = (
                x as f64 + 0.5 - 0.3 * w as f64,
                y as f64 + 0.5 - 0.7 * h as f64,
            );
            if ddx * ddx + ddy * ddy < 0.02 * w as f64 * h as f64 {
                (r, g, b) = (0.65, 0.20, 0.08);
            }
            put(&mut img, x, y, [r, g, b]);
        }
    }
    img
}

fn flat(w: usize, h: usize, rgb: [f64; 3]) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            put(&mut img, x, y, rgb);
        }
    }
    img
}

/// Anti-aliased two-colour step: `kind` 'h', 'v' or 'd' (diagonal).
fn edge(w: usize, h: usize, kind: char, lo: [f64; 3], hi: [f64; 3]) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut inside = 0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let px = x as f64 + (f64::from(sx) + 0.5) / 4.0;
                    let py = y as f64 + (f64::from(sy) + 0.5) / 4.0;
                    let on_hi = match kind {
                        'h' => py > 0.5 * h as f64 + 0.3,
                        'v' => px > 0.5 * w as f64 + 0.3,
                        _ => (px - 0.5 * w as f64) * 0.8 + (py - 0.5 * h as f64) > 0.0,
                    };
                    inside += i32::from(on_hi);
                }
            }
            let a = f64::from(inside) / 16.0;
            put(
                &mut img,
                x,
                y,
                [0, 1, 2].map(|c| lo[c] + a * (hi[c] - lo[c])),
            );
        }
    }
    img
}

fn zone_plate(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let (cx, cy) = (0.5 * w as f64, 0.5 * h as f64);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            let v = 0.30 + 0.22 * (std::f64::consts::PI * (dx * dx + dy * dy) / w as f64).cos();
            put(&mut img, x, y, [1.10 * v, v, 0.85 * v]);
        }
    }
    img
}

/// The value nearest to `v` on the working-scale grid of headroom `h`: `(n * H) / 65535` in f32.
fn on_grid(v: f64, headroom: f64) -> f64 {
    let n = (v * 65535.0 / headroom + 0.5).floor();
    f64::from((n * headroom / 65535.0) as f32)
}

fn mosaic_of(img: &RgbImage, cfa: Cfa) -> Mosaic {
    synth::mosaic_from_rgb(img, cfa)
}

/// Poisson-Gaussian noise of `noise` on a balanced mosaic with gains `g0`, and the R1 clamp at
/// -0.05 (unbalanced units).
fn add_noise(m: &mut Mosaic, a: f64, b: f64, g0: [f64; 3], seed: u64) {
    let mut rng = Rng::new(seed);
    for y in 0..m.height {
        for x in 0..m.width {
            let c = m.cfa.color_at(x, y).plane();
            let v = &mut m.data[y * m.width + x];
            let u = f64::from(*v) / g0[c];
            let var = a * u.max(0.0) + b;
            let un = (u + var.sqrt() * f64::from(rng.gaussian())).max(-0.05);
            *v = (un * g0[c]) as f32;
        }
    }
}

/// The two forms with one parameter set, to run the same property on both.
#[derive(Clone, Copy, Debug)]
enum Form {
    Fixed,
    Float,
}

impl Form {
    const BOTH: [Form; 2] = [Form::Fixed, Form::Float];

    fn run(self, m: &Mosaic, noise: NoiseDn, guide_detail: i32, headroom_log2: u8) -> RgbImage {
        match self {
            Form::Fixed => log_guided::demosaic_fixed(m, noise, guide_detail, headroom_log2),
            Form::Float => log_guided::demosaic_float(m, noise, guide_detail),
        }
    }
}

fn bits_equal(a: &RgbImage, b: &RgbImage) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn finite_noise(b: f64) -> NoiseDn {
    NoiseDn::from_model(&NoiseModel::new(0.0, b))
}

#[test]
fn the_texture_port_reproduces_the_oracle_inputs() {
    // The phase vectors sample the oracle's texture scene; the port gives the same photosites.
    let texture = texture(32, 32);
    let mut checked = 0;
    for v in load_vectors()
        .iter()
        .filter(|v| v.name.starts_with("phase_"))
    {
        let m = mosaic_of(&texture, v.mosaic.cfa);
        assert_eq!(m.data, v.mosaic.data, "{}", v.name);
        checked += 1;
    }
    assert_eq!(checked, 4);
}

// ---------------------------------------------------------------------------------------------
// Property tests (spec section 6.2)
// ---------------------------------------------------------------------------------------------

#[test]
fn log2_and_exp2_fixed_meet_their_exhaustive_bounds() {
    // log2_fixed (spec 3.2): -4.98 .. +2.98 codes, anchors, doubling adds 4096, steps back by at
    // most one code (23 times in the top octave).
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    let (mut drops, mut worst_drop) = (0, 0i32);
    for y in 1..=65535u32 {
        let l = i32::from(log2_guided(y));
        let e = f64::from(l) - 4096.0 * f64::from(y).log2();
        lo = lo.min(e);
        hi = hi.max(e);
        if y > 1 {
            let step = l - i32::from(log2_guided(y - 1));
            if step < 0 {
                drops += 1;
                worst_drop = worst_drop.min(step);
            }
        }
        if y <= 32767 {
            assert_eq!(
                i32::from(log2_guided(2 * y)),
                l + 4096,
                "doubling {y} adds exactly 4096"
            );
        }
    }
    println!("log2_fixed - 4096 log2 y in [{lo:.4}, {hi:.4}], {drops} single-step drops");
    assert!(lo >= -4.98 - 1e-9 && hi <= 2.98 + 1e-9, "[{lo}, {hi}]");
    assert!(worst_drop >= -1);
    assert_eq!(drops, 23, "the spec counts 23 one-code drops");
    assert_eq!(
        (log2_guided(1), log2_guided(2), log2_guided(65535)),
        (0, 4096, 65535)
    );
    assert_eq!(log_guided::log2_fixed(0), 0);

    // exp2_fixed (spec 3.4): -(1 + 1.18e-4 t) <= C - t <= 1.24e-4 t, monotone, max 65522.
    let (mut under, mut over) = (0.0f64, 0.0f64);
    let (mut prev, mut max) = (0u16, 0u16);
    for l in 0..=65535u32 {
        let t = (f64::from(l) / 4096.0).exp2();
        let c = log_guided::exp2_fixed(l as u16);
        let e = f64::from(c) - t;
        if e < 0.0 {
            under = under.max((-e - 1.0) / t);
        }
        over = over.max(e / t);
        assert!(c >= prev, "monotone at {l}");
        prev = c;
        max = max.max(c);
    }
    println!("exp2_fixed: -(1 + {under:.3e} t) <= C - t <= {over:.3e} t, max {max}");
    assert!(under <= 1.18e-4 + 1e-9 && over <= 1.24e-4 + 1e-9);
    assert_eq!(max, 65522);
    for e in 0..16u32 {
        assert_eq!(
            u32::from(log_guided::exp2_fixed((4096 * e) as u16)),
            1 << e,
            "exp2_fixed(4096 * {e})"
        );
    }
}

fn log2_guided(y: u32) -> u16 {
    log_guided::log2_fixed(y as u16)
}

#[test]
fn flat_fields_come_back_exact() {
    // Constant grey and constant colour on the code grid, all four phases, H = 1 and 4, detail
    // -50 / 0 / 100: the fixed form is bit-exact, the float form within 1e-6 relative.
    let values: [[f64; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [1.0, 1.0, 1.0],
        [0.42, 0.30, 0.18],
        [0.003, 0.9, 0.0007],
    ];
    let noise = finite_noise(4e-8);
    for cfa in Cfa::ALL {
        for headroom_log2 in [0u8, 2] {
            let headroom = f64::from(1u32 << headroom_log2);
            for detail in [-50, 0, 100] {
                for v in values {
                    let c = v.map(|x| on_grid(x, headroom));
                    let m = mosaic_of(&flat(13, 10, c), cfa);
                    let fixed = log_guided::demosaic_fixed(&m, noise, detail, headroom_log2);
                    let float = log_guided::demosaic_float(&m, noise, detail);
                    for (ch, &want) in c.iter().enumerate() {
                        for (&f, &g) in fixed.plane(ch).iter().zip(float.plane(ch)) {
                            assert_eq!(
                                f.to_bits(),
                                (want as f32).to_bits(),
                                "fixed {cfa:?} H={headroom} detail {detail} {v:?}"
                            );
                            assert!(
                                (f64::from(g) - want).abs() <= 1e-6 * want.max(1e-6),
                                "float {cfa:?} detail {detail} {v:?}: {g} against {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tiles_with_five_pixels_of_context_give_the_whole_frame_bit_for_bit() {
    let mut m = mosaic_of(&texture(41, 37), Cfa::GBRG);
    add_noise(&mut m, 2e-4, 1e-7, [2.0, 1.0, 1.6], 7);
    let noise = NoiseDn::from_model(&NoiseModel::new(2e-4, 1e-7));
    let (detail, headroom_log2) = (50, 1);
    let frame = Rect::whole(m.width, m.height);
    for form in Form::BOTH {
        let whole = form.run(&m, noise, detail, headroom_log2);
        let (tw, th, halo) = (9, 8, log_guided::HALO_PX);
        let mut worst = 0.0f32;
        for ty in (0..m.height).step_by(th) {
            for tx in (0..m.width).step_by(tw) {
                let core = Rect::new(tx, ty, tw.min(m.width - tx), th.min(m.height - ty));
                let ext = core.grow(ym_gpu::cpu::Halo::uniform(halo), frame);
                let tile = m.extract(ext);
                let res = form.run(&tile, noise, detail, headroom_log2);
                for c in 0..3 {
                    for y in core.y..core.y1() {
                        for x in core.x..core.x1() {
                            let got = res.plane(c)[(y - ext.y) * ext.w + (x - ext.x)];
                            let want = whole.plane(c)[y * m.width + x];
                            worst = worst.max((got - want).abs());
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "{form:?} tile ({tx}, {ty}) channel {c} pixel ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(worst, 0.0);
    }
}

#[test]
fn the_phase_of_a_cropped_buffer_does_not_change_the_interior() {
    let base = mosaic_of(&texture(40, 36), Cfa::RGGB);
    let noise = finite_noise(1e-7);
    for form in Form::BOTH {
        let whole = form.run(&base, noise, 30, 2);
        for (ox, oy) in [(1, 0), (0, 1), (1, 1)] {
            let sub = base.extract(Rect::new(ox, oy, base.width - ox, base.height - oy));
            assert_ne!(sub.cfa, base.cfa, "the crop changes the phase");
            let res = form.run(&sub, noise, 30, 2);
            let halo = log_guided::HALO_PX;
            for c in 0..3 {
                for y in halo..sub.height - halo {
                    for x in halo..sub.width - halo {
                        let got = res.plane(c)[y * sub.width + x];
                        let want = whole.plane(c)[(y + oy) * base.width + x + ox];
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{form:?} crop ({ox}, {oy}) channel {c} ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}

fn flip_mosaic(m: &Mosaic, fx: bool, fy: bool) -> Mosaic {
    let (w, h) = (m.width, m.height);
    let mut out = m.clone();
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (
                if fx { w - 1 - x } else { x },
                if fy { h - 1 - y } else { y },
            );
            out.data[y * w + x] = m.data[sy * w + sx];
        }
    }
    // The new (0, 0) is the old corner; the pattern repeats every two samples.
    out.cfa = m
        .cfa
        .shifted(if fx { w - 1 } else { 0 }, if fy { h - 1 } else { 0 });
    out
}

fn flip_image(img: &RgbImage, fx: bool, fy: bool) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut out = img.clone();
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (
                    if fx { w - 1 - x } else { x },
                    if fy { h - 1 - y } else { y },
                );
                out.plane_mut(c)[y * w + x] = img.plane(c)[sy * w + sx];
            }
        }
    }
    out
}

#[test]
fn flipping_the_mosaic_flips_the_result() {
    let noise = finite_noise(1e-7);
    for form in Form::BOTH {
        let mut worst = 0.0f64;
        let mut exact = true;
        for cfa in Cfa::ALL {
            let m = mosaic_of(&texture(32, 28), cfa);
            let r = form.run(&m, noise, 0, 1);
            for k in 1..4 {
                let (fx, fy) = (k & 1 != 0, k & 2 != 0);
                let back = flip_image(&form.run(&flip_mosaic(&m, fx, fy), noise, 0, 1), fx, fy);
                for (a, b) in r.data.iter().zip(&back.data) {
                    exact &= a.to_bits() == b.to_bits();
                    worst = worst
                        .max((f64::from(*a) - f64::from(*b)).abs() / f64::from(a.abs()).max(1e-3));
                }
            }
        }
        match form {
            Form::Fixed => assert!(exact, "the fixed form is bit-exact under flips"),
            Form::Float => assert!(worst <= 1e-6, "float form worst relative {worst}"),
        }
        println!("flip equivariance {form:?}: bit-exact {exact}, worst relative {worst:.2e}");
    }
}

#[test]
fn the_receptive_field_is_exactly_five_pixels() {
    let m = mosaic_of(&texture(31, 31), Cfa::RGGB);
    let noise = finite_noise(1e-8);
    for form in Form::BOTH {
        let base = form.run(&m, noise, 100, 2);
        let mut reach = 0usize;
        for (px, py) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (cx, cy) = (15 + px, 15 + py);
            let mut q = m.clone();
            let i = cy * q.width + cx;
            q.data[i] = q.data[i] * 4.0 + 0.3;
            let r = form.run(&q, noise, 100, 2);
            for c in 0..3 {
                for y in 0..m.height {
                    for x in 0..m.width {
                        let j = y * m.width + x;
                        if r.plane(c)[j].to_bits() != base.plane(c)[j].to_bits() {
                            reach = reach.max(x.abs_diff(cx).max(y.abs_diff(cy)));
                        }
                    }
                }
            }
        }
        println!("receptive-field radius ({form:?}): {reach} px");
        assert_eq!(reach, log_guided::HALO_PX, "{form:?}");
    }
    assert_eq!(Method::LogGuided.halo(), 5);
    assert_eq!(demosaic::LOG_GUIDED_HALO, 5);
}

#[test]
fn doubling_signal_and_read_noise_doubles_the_output() {
    let m = mosaic_of(&texture(32, 32), Cfa::RGGB); // maximum about 1.23
    let mut m2 = m.clone();
    for v in &mut m2.data {
        *v *= 2.0;
    }
    // r_dn 10.5 and 21: p = 5 and 10 (float), 2 and 2 at H = 2 and 4 (fixed).
    let (n1, n2) = (NoiseDn::new(0.0, 10.5), NoiseDn::new(0.0, 21.0));

    // Float form: up to rounding.
    let a = log_guided::demosaic_float(&m, n1, 0);
    let b = log_guided::demosaic_float(&m2, n2, 0);
    let worst = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (2.0 * f64::from(*x) - f64::from(*y)).abs() / f64::from(y.abs()).max(1e-3))
        .fold(0.0, f64::max);
    println!("scale x2, float form: worst relative {worst:.2e}");
    assert!(worst <= 1e-6, "{worst}");

    // Fixed form with the headroom doubled as well: exactly twice.
    let a = log_guided::demosaic_fixed(&m, n1, 0, 1);
    let b = log_guided::demosaic_fixed(&m2, n2, 0, 2);
    for (x, y) in a.data.iter().zip(&b.data) {
        assert_eq!((2.0 * x).to_bits(), y.to_bits());
    }

    // Fixed form at the same headroom: only the rounding differs, within 32 codes of the doubled
    // image.
    let a = log_guided::demosaic_fixed(&m, n1, 0, 2);
    let b = log_guided::demosaic_fixed(&m2, n2, 0, 2);
    let codes = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (2.0 * f64::from(*x) - f64::from(*y)).abs() * 65535.0 / 4.0)
        .fold(0.0, f64::max);
    println!("scale x2, fixed form, same headroom: worst {codes:.2} codes");
    assert!(codes <= 32.0, "{codes}");
}

#[test]
fn random_inputs_never_give_nan_and_native_samples_pass_through() {
    let mut rng = Rng::new(42);
    for trial in 0..24usize {
        let w = 3 + (rng.next_u64() % 20) as usize;
        let h = 1 + (rng.next_u64() % 20) as usize;
        let cfa = Cfa::ALL[trial % 4];
        let mut m = Mosaic::new(w, h, cfa);
        for v in &mut m.data {
            // [-0.05, 20], skewed to small values.
            *v = -0.05 + 20.0 * rng.next_f32() * rng.next_f32() * rng.next_f32();
        }
        if trial % 3 == 0 {
            m.data[0] = f32::NAN;
        }
        if trial % 5 == 0 {
            let last = m.data.len() - 1;
            m.data[last] = f32::INFINITY;
        }
        if trial % 7 == 0 {
            let mid = m.data.len() / 2;
            m.data[mid] = f32::NEG_INFINITY;
        }
        let detail = if trial % 2 == 1 { 100 } else { -50 };
        let headroom_log2 = (trial % 5) as u8;
        let noise = finite_noise((trial % 4) as f64 * 1e-6);
        for form in Form::BOTH {
            let out = form.run(&m, noise, detail, headroom_log2);
            assert!(
                out.data.iter().all(|v| v.is_finite()),
                "{form:?} trial {trial}: non-finite output"
            );
            for y in 0..h {
                for x in 0..w {
                    let c = cfa.color_at(x, y).plane();
                    let v = m.data[y * w + x];
                    let want = if v.is_finite() { v } else { 0.0 };
                    assert_eq!(
                        out.plane(c)[y * w + x].to_bits(),
                        want.to_bits(),
                        "{form:?} trial {trial}: native sample ({x}, {y})"
                    );
                }
            }
            if let Form::Float = form {
                // Interpolated samples are never negative in the float form either.
                for c in 0..3 {
                    for y in 0..h {
                        for x in 0..w {
                            if cfa.color_at(x, y).plane() != c {
                                assert!(out.plane(c)[y * w + x] >= 0.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn values_above_the_headroom_saturate_in_interpolated_channels_only() {
    let m = mosaic_of(&flat(12, 12, [1.6, 1.3, 0.5]), Cfa::RGGB);
    let n = 144;
    let noise = NoiseDn::ZERO;
    // H = 1: the red at a green site (x = 1, y = 0) saturates at 1.0, the native red at the red
    // site (0, 0) keeps 1.6.
    let out = log_guided::demosaic_fixed(&m, noise, 0, 0);
    assert_eq!(out.plane(0)[0], 1.6);
    assert!((out.plane(0)[1] - 1.0).abs() < 1e-7, "{}", out.plane(0)[1]);
    assert!(out.plane(0)[1] <= 1.0);
    assert_eq!(out.data.len(), 3 * n);
    // H = 2 keeps 1.6 (within two codes of the doubled scale).
    let out = log_guided::demosaic_fixed(&m, noise, 0, 1);
    assert!(
        (out.plane(0)[1] - 1.6).abs() < 2.0 / 65535.0,
        "{}",
        out.plane(0)[1]
    );
    // The float form has no headroom at all.
    let out = log_guided::demosaic_float(&m, noise, 0);
    assert!((out.plane(0)[1] - 1.6).abs() < 1e-5, "{}", out.plane(0)[1]);
}

#[test]
fn the_fixed_and_float_forms_agree_and_beat_bilinear_on_edges_and_texture() {
    let (lo, hi) = ([0.05, 0.08, 0.06], [0.60, 0.45, 0.30]);
    let scenes: [(&str, RgbImage); 5] = [
        ("edge_h", edge(40, 40, 'h', lo, hi)),
        ("edge_v", edge(40, 40, 'v', lo, hi)),
        ("edge_d", edge(40, 40, 'd', lo, hi)),
        ("zone_plate", zone_plate(40, 40)),
        ("texture", texture(40, 40)),
    ];
    let noise = finite_noise(1e-8);
    let (mut worst_max, mut worst_mean) = (0.0f64, 0.0f64);
    for (name, img) in &scenes {
        let m = mosaic_of(img, Cfa::RGGB);
        let fixed = log_guided::demosaic_fixed(&m, noise, 0, 1); // the texture reaches 1.23
        let float = log_guided::demosaic_float(&m, noise, 0);
        let diffs: Vec<f64> = fixed
            .data
            .iter()
            .zip(&float.data)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .collect();
        let max = diffs.iter().copied().fold(0.0, f64::max);
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        worst_max = worst_max.max(max);
        worst_mean = worst_mean.max(mean);
        let bilinear = demosaic::bilinear(&m);
        let (pa, pb, pl) = (
            psnr(&fixed, img, 6, 1.0),
            psnr(&float, img, 6, 1.0),
            psnr(&bilinear, img, 6, 1.0),
        );
        println!(
            "{name:<10} |fixed - float| max {max:.2e} mean {mean:.2e} | PSNR fixed {pa:.2} float {pb:.2} bilinear {pl:.2} dB"
        );
        if *name != "zone_plate" {
            assert!(pa > pl && pb > pl, "{name}: both forms beat bilinear");
        }
    }
    assert!(worst_max <= 2e-3, "max {worst_max}");
    assert!(worst_mean <= 5e-5, "mean {worst_mean}");
}

// ---------------------------------------------------------------------------------------------
// Quality gate (spec section 6.4)
// ---------------------------------------------------------------------------------------------

/// The synthetic set of `tests/cpu_demosaic.rs`.
fn synthetic_set() -> Vec<RgbImage> {
    vec![
        synth::zone_plate(192, 192),
        synth::edges(192, 192),
        synth::color_patches(192, 192),
        synth::fine_text(192, 192, 1),
        synth::natural(192, 192, 1),
        synth::natural(192, 192, 2),
        synth::natural(192, 192, 3),
    ]
}

#[test]
fn on_the_synthetic_set_it_is_within_one_db_of_menon_at_every_signal_scale() {
    // The RA-06 gate: the mean PSNR over the synthetic set, a 10 px border excluded, at signal
    // scales 1, 0.1 and 0.01 (peak scaled along), p from a zero-noise model.
    let set = synthetic_set();
    for scale in [1.0f32, 0.1, 0.01] {
        let mut mean = [0.0f64; 3]; // bilinear, menon, log-guided
        for img in &set {
            let mut truth = img.clone();
            for v in &mut truth.data {
                *v *= scale;
            }
            let m = mosaic_of(&truth, Cfa::RGGB);
            for (i, method) in [Method::Bilinear, Method::Menon, Method::LogGuided]
                .into_iter()
                .enumerate()
            {
                let out = demosaic::demosaic(&m, method);
                mean[i] += psnr(&out, &truth, 10, f64::from(scale)) / set.len() as f64;
            }
        }
        println!(
            "scale {scale}: bilinear {:.2}, menon {:.2}, log_guided {:.2} dB (log_guided - menon {:+.2})",
            mean[0],
            mean[1],
            mean[2],
            mean[2] - mean[1]
        );
        assert!(
            mean[2] >= mean[1] - 1.0,
            "scale {scale}: log_guided {:.2} is more than 1 dB below menon {:.2}",
            mean[2],
            mean[1]
        );
        assert!(
            mean[2] > mean[0] + 1.0,
            "scale {scale}: log_guided {:.2} against bilinear {:.2}",
            mean[2],
            mean[0]
        );
    }
}

#[test]
fn smooth_ramps_are_nearly_exact_in_the_interior() {
    // The check of `demosaic.rs` for every method: a smooth colour ramp comes back within 2e-3 away
    // from the border, in all four phases.
    let (w, h) = (40, 32);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.set_pixel(
                x,
                y,
                [
                    0.2 + 0.5 * x as f32 / w as f32,
                    0.3 + 0.4 * y as f32 / h as f32,
                    0.6 - 0.3 * (x + y) as f32 / (w + h) as f32,
                ],
            );
        }
    }
    for cfa in Cfa::ALL {
        let m = mosaic_of(&img, cfa);
        let out = demosaic::demosaic_whole(&m, Method::LogGuided);
        let mut worst = 0.0f32;
        for y in 6..26 {
            for x in 6..34 {
                for c in 0..3 {
                    worst = worst.max((out.pixel(x, y)[c] - img.pixel(x, y)[c]).abs());
                }
            }
        }
        println!(
            "smooth ramp {}: worst interior error {worst:.2e}",
            cfa.name()
        );
        assert!(worst < 2e-3, "{cfa:?}: {worst}");
    }
}

// ---------------------------------------------------------------------------------------------
// Determinism, tiling through the banded runner, dispatch
// ---------------------------------------------------------------------------------------------

#[test]
fn banded_and_whole_results_and_thread_counts_give_the_same_bytes() {
    let img = synth::natural(300, 261, 5);
    let noise = NoiseDn::from_model(&NoiseModel::new(2e-4, 4e-8));
    let inputs = DemosaicInputs {
        noise,
        ..DemosaicInputs::default()
    };
    for cfa in Cfa::ALL {
        let m = mosaic_of(&img, cfa);
        let whole = demosaic::demosaic_whole_with(&m, Method::LogGuided, &inputs);
        let banded = demosaic::demosaic_with(&m, Method::LogGuided, &inputs);
        assert!(bits_equal(&whole, &banded), "{cfa:?}: banded against whole");
        let mut results = Vec::new();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            results.push(pool.install(|| demosaic::demosaic_with(&m, Method::LogGuided, &inputs)));
        }
        assert!(
            bits_equal(&results[0], &results[1]),
            "{cfa:?}: 1 thread against 4"
        );
        assert!(
            bits_equal(&results[0], &whole),
            "{cfa:?}: pool against whole"
        );
        // An odd origin changes the phase and the band layout; the interior must not care.
        let sub = m.extract(Rect::new(3, 5, 281, 250));
        let again = demosaic::demosaic_with(&sub, Method::LogGuided, &inputs);
        for c in 0..3 {
            for y in 5..sub.height - 5 {
                for x in 5..sub.width - 5 {
                    assert_eq!(
                        again.plane(c)[y * sub.width + x].to_bits(),
                        whole.plane(c)[(y + 5) * m.width + x + 3].to_bits(),
                        "{cfa:?} odd-origin crop channel {c} ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[test]
fn the_method_reads_the_noise_of_the_inputs() {
    let img = synth::natural(64, 64, 3);
    let m = mosaic_of(&img, Cfa::RGGB);
    let direct = |noise| log_guided::demosaic(&m, noise);
    let via = |noise| {
        demosaic::demosaic_with(
            &m,
            Method::LogGuided,
            &DemosaicInputs {
                noise,
                ..DemosaicInputs::default()
            },
        )
    };
    let quiet = NoiseDn::ZERO;
    let loud = NoiseDn::from_model(&NoiseModel::new(0.0, 4e-6)); // r_dn about 131, p = 65
    assert!(bits_equal(&direct(quiet), &via(quiet)));
    assert!(bits_equal(&direct(loud), &via(loud)));
    assert!(
        !bits_equal(&direct(quiet), &direct(loud)),
        "the offset p must change the result"
    );
    // `demosaic(m, method)` is the zero-noise call.
    assert!(bits_equal(
        &demosaic::demosaic(&m, Method::LogGuided),
        &direct(quiet)
    ));
    // An invalid noise value reads as 0 (spec 2.2), it does not poison the frame.
    for r in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            bits_equal(&direct(NoiseDn::new(0.0, r)), &direct(quiet)),
            "r = {r}"
        );
    }
}

#[test]
fn degenerate_sizes_are_defined() {
    // Any size from 1x1 up works (the mirror wraps several times); the dispatch sends buffers
    // under 2x2 to bilinear.
    let noise = finite_noise(1e-7);
    for cfa in Cfa::ALL {
        for (w, h) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 3),
            (5, 3),
            (3, 5),
        ] {
            let mut rng = Rng::new((w * 31 + h) as u64);
            let mut m = Mosaic::new(w, h, cfa);
            for v in &mut m.data {
                *v = rng.range(0.0, 1.5);
            }
            for form in Form::BOTH {
                let out = form.run(&m, noise, 40, 1);
                assert_eq!((out.width, out.height), (w, h));
                assert!(out.data.iter().all(|v| v.is_finite()), "{form:?} {w}x{h}");
            }
            let dispatched = demosaic::demosaic(&m, Method::LogGuided);
            assert_eq!((dispatched.width, dispatched.height), (w, h));
            assert!(dispatched.data.iter().all(|v| v.is_finite()));
        }
    }
    let empty = Mosaic::new(0, 0, Cfa::RGGB);
    assert!(log_guided::demosaic(&empty, NoiseDn::ZERO).data.is_empty());
    assert!(
        log_guided::demosaic_fixed(&empty, NoiseDn::ZERO, 0, 2)
            .data
            .is_empty()
    );
}

/// A plain transcription of spec section 3.9 (the float form), written for the review of RA-06:
/// planes over the padded frame, the 5x5 box as 25 sequential additions, the site class read
/// from the CFA at each pixel. It is slow and simple on purpose. The tests compare the optimised
/// implementation (separable box, per-row passes, stride-2 site loops) with it byte for byte, on
/// inputs the vectors do not reach (a sharpened guide that undershoots, huge values, non-finite
/// samples, every phase).
fn naive_float(m: &Mosaic, noise: NoiseDn, guide_detail: i32) -> RgbImage {
    const B: usize = log_guided::HALO_PX;
    let (w, h) = (m.width, m.height);
    let d = log_guided::derive(noise, guide_detail, 0);
    let p = f64::from(d.p);
    let kf = d.kernel.map(|k| f64::from(k) / 1024.0);
    let (wp, hp) = (w + 2 * B, h + 2 * B);
    let at = |x: usize, y: usize| y * wp + x;

    // 3.1: A = 65535 max(v, 0), non-finite -> 0, mirror-padded.
    let mut a = vec![0.0f32; wp * hp];
    for y in 0..hp {
        for x in 0..wp {
            let sy = mirror(y as isize - B as isize, h);
            let sx = mirror(x as isize - B as isize, w);
            let v = m.data[sy * w + sx];
            let v = if v.is_finite() { v } else { 0.0 };
            a[at(x, y)] = (65535.0 * f64::from(v).max(0.0)) as f32;
        }
    }
    // 3.2: L = log2(p + A / 2).
    let l: Vec<f32> = a
        .iter()
        .map(|&v| (p + 0.5 * f64::from(v)).log2() as f32)
        .collect();
    // 3.3: horizontal then vertical guide filter, no rounding.
    let mut lh = vec![0.0f32; wp * hp];
    for y in 0..hp {
        for x in 2..wp - 2 {
            let mut acc = 0.0f64;
            for (i, k) in kf.iter().enumerate() {
                acc += k * f64::from(l[at(x - 2 + i, y)]);
            }
            lh[at(x, y)] = acc as f32;
        }
    }
    // 3.4: C = 2^max(Lv, 0).
    let mut c = vec![0.0f32; wp * hp];
    for y in 2..hp - 2 {
        for x in 2..wp - 2 {
            let mut acc = 0.0f64;
            for (j, k) in kf.iter().enumerate() {
                acc += k * f64::from(lh[at(x, y - 2 + j)]);
            }
            c[at(x, y)] = f64::max(acc, 0.0).exp2() as f32;
        }
    }
    // 3.5: structure measures.
    let mut dh = vec![0.0f32; wp * hp];
    let mut dv = vec![0.0f32; wp * hp];
    for y in 3..hp - 3 {
        for x in 3..wp - 3 {
            let cc = f64::from(c[at(x, y)]);
            let hh = 0.5 * (f64::from(c[at(x - 1, y)]) + f64::from(c[at(x + 1, y)]));
            let vv = 0.5 * (f64::from(c[at(x, y - 1)]) + f64::from(c[at(x, y + 1)]));
            dh[at(x, y)] = ((hh - cc).abs() + p) as f32;
            dv[at(x, y)] = ((vv - cc).abs() + p) as f32;
        }
    }
    // 3.6: 5x5 box means, 25 sequential additions.
    let mut bh = vec![0.0f32; wp * hp];
    let mut bv = vec![0.0f32; wp * hp];
    for y in B..hp - B {
        for x in B..wp - B {
            let (mut sh, mut sv) = (0.0f64, 0.0f64);
            for j in 0..5 {
                for i in 0..5 {
                    sh += f64::from(dh[at(x - 2 + i, y - 2 + j)]);
                    sv += f64::from(dv[at(x - 2 + i, y - 2 + j)]);
                }
            }
            bh[at(x, y)] = (sh / 25.0) as f32;
            bv[at(x, y)] = (sv / 25.0) as f32;
        }
    }
    // 3.7 and 3.9: reconstruction.
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    let cube = |v: f32| {
        let t = f64::from(v).max(1.0);
        t * t * t
    };
    for y in 0..h {
        for x in 0..w {
            let (xx, yy) = (x + B, y + B);
            let cc = f64::from(c[at(xx, yy)]);
            let ha = |x: usize| 0.5 * (f64::from(a[at(x - 1, yy)]) + f64::from(a[at(x + 1, yy)]));
            let va = |x: usize| 0.5 * (f64::from(a[at(x, yy - 1)]) + f64::from(a[at(x, yy + 1)]));
            let hc = |x: usize| 0.5 * (f64::from(c[at(x - 1, yy)]) + f64::from(c[at(x + 1, yy)]));
            let vc = |x: usize| 0.5 * (f64::from(c[at(x, yy - 1)]) + f64::from(c[at(x, yy + 1)]));
            let green = || {
                let wh = cube(bv[at(xx, yy)]);
                let wv = cube(bh[at(xx, yy)]);
                let num = wh * ha(xx) + wv * va(xx);
                let den = wh * hc(xx) + wv * vc(xx);
                num * (cc / den)
            };
            let diag = || (va(xx - 1) + va(xx + 1)) * (cc / (vc(xx - 1) + vc(xx + 1)));
            let ratio_h = || (cc / hc(xx)) * ha(xx);
            let ratio_v = || (cc / vc(xx)) * va(xx);
            let deq = |v: f64| (v / 65535.0) as f32;
            let native = {
                let v = m.data[y * w + x];
                if v.is_finite() { v } else { 0.0 }
            };
            let rgb = match m.cfa.color_at(x, y) {
                CfaColor::Red => [native, deq(green()), deq(diag())],
                CfaColor::Blue => [deq(diag()), deq(green()), native],
                CfaColor::Green => {
                    if m.cfa.color_at(x + 1, y) == CfaColor::Red {
                        [deq(ratio_h()), native, deq(ratio_v())]
                    } else {
                        [deq(ratio_v()), native, deq(ratio_h())]
                    }
                }
            };
            out.set_pixel(x, y, rgb);
        }
    }
    out
}

/// A plain transcription of the fixed-point form (spec sections 3.1 to 3.8), like [`naive_float`]:
/// planes over the padded frame, the clamps written out with `i64`, the 5x5 box as 25 sequential
/// additions, the site class read from the CFA at each pixel. It uses the crate's `log2_fixed`
/// and `exp2_fixed` (checked exhaustively on their own).
fn naive_fixed(m: &Mosaic, noise: NoiseDn, guide_detail: i32, headroom_log2: u8) -> RgbImage {
    const B: usize = log_guided::HALO_PX;
    let (w, h) = (m.width, m.height);
    let d = log_guided::derive(noise, guide_detail, headroom_log2);
    let p = i64::from(d.p);
    let k = d.kernel.map(i64::from);
    let (wp, hp) = (w + 2 * B, h + 2 * B);
    let at = |x: usize, y: usize| y * wp + x;

    // 3.1: A = round_half_even(v S / H) clamped to 0..=65535; non-finite -> 0.
    let mut a = vec![0i64; wp * hp];
    for y in 0..hp {
        for x in 0..wp {
            let sy = mirror(y as isize - B as isize, h);
            let sx = mirror(x as isize - B as isize, w);
            let v = f64::from(m.data[sy * w + sx]);
            let t = v * 65535.0 / d.headroom;
            a[at(x, y)] = if !v.is_finite() || t <= 0.0 {
                0
            } else if t >= 65535.0 {
                65535
            } else {
                t.round_ties_even() as i64
            };
        }
    }
    // 3.2: L = log2_fixed(p + floor((A + 1) / 2)).
    let l: Vec<i64> = a
        .iter()
        .map(|&v| i64::from(log2_guided((p + (v + 1) / 2) as u32)))
        .collect();
    // 3.3: guide filter, horizontal then vertical: clamp(floor((sum + 512) / 1024), 0, 65535).
    let filtered = |acc: i64| (acc + 512).div_euclid(1024).clamp(0, 65535);
    let mut lh = vec![0i64; wp * hp];
    for y in 0..hp {
        for x in 2..wp - 2 {
            let acc: i64 = (0..5).map(|i| k[i] * l[at(x - 2 + i, y)]).sum();
            lh[at(x, y)] = filtered(acc);
        }
    }
    // 3.4: C = exp2_fixed(Lv).
    let mut c = vec![0i64; wp * hp];
    for y in 2..hp - 2 {
        for x in 2..wp - 2 {
            let acc: i64 = (0..5).map(|j| k[j] * lh[at(x, y - 2 + j)]).sum();
            c[at(x, y)] = i64::from(log_guided::exp2_fixed(filtered(acc) as u16));
        }
    }
    let rhalf = |a: i64, b: i64| (a + b + 1).div_euclid(2);
    // 3.5: structure measures.
    let mut dh = vec![0i64; wp * hp];
    let mut dv = vec![0i64; wp * hp];
    for y in 3..hp - 3 {
        for x in 3..wp - 3 {
            let cc = c[at(x, y)];
            dh[at(x, y)] = ((rhalf(c[at(x - 1, y)], c[at(x + 1, y)]) - cc).abs() + p).min(65535);
            dv[at(x, y)] = ((rhalf(c[at(x, y - 1)], c[at(x, y + 1)]) - cc).abs() + p).min(65535);
        }
    }
    // 3.6: 5x5 box means, floor(sum / 25).
    let mut bh = vec![0i64; wp * hp];
    let mut bv = vec![0i64; wp * hp];
    for y in B..hp - B {
        for x in B..wp - B {
            let (mut sh, mut sv) = (0i64, 0i64);
            for j in 0..5 {
                for i in 0..5 {
                    sh += dh[at(x - 2 + i, y - 2 + j)];
                    sv += dv[at(x - 2 + i, y - 2 + j)];
                }
            }
            bh[at(x, y)] = sh / 25;
            bv[at(x, y)] = sv / 25;
        }
    }
    // 3.7 and 3.8: reconstruction in binary64, rounded to a code, then back to the headroom scale.
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    let code = |v: f64| -> i64 {
        if v.is_nan() || v <= 0.0 {
            0
        } else if v >= 65535.0 {
            65535
        } else {
            v.round_ties_even() as i64
        }
    };
    let cube = |v: i64| {
        let t = v.max(1) as f64;
        t * t * t
    };
    for y in 0..h {
        for x in 0..w {
            let (xx, yy) = (x + B, y + B);
            let cc = c[at(xx, yy)] as f64;
            let ha = |x: usize| rhalf(a[at(x - 1, yy)], a[at(x + 1, yy)]) as f64;
            let va = |x: usize| rhalf(a[at(x, yy - 1)], a[at(x, yy + 1)]) as f64;
            let hc = |x: usize| rhalf(c[at(x - 1, yy)], c[at(x + 1, yy)]) as f64;
            let vc = |x: usize| rhalf(c[at(x, yy - 1)], c[at(x, yy + 1)]) as f64;
            let green = || {
                let wh = cube(bv[at(xx, yy)]);
                let wv = cube(bh[at(xx, yy)]);
                let num = wh * ha(xx) + wv * va(xx);
                let den = wh * hc(xx) + wv * vc(xx);
                code(num * (cc / den))
            };
            let diag = || code((va(xx - 1) + va(xx + 1)) * (cc / (vc(xx - 1) + vc(xx + 1))));
            let ratio_h = || code((cc / hc(xx)) * ha(xx));
            let ratio_v = || code((cc / vc(xx)) * va(xx));
            let deq = |n: i64| (n as f64 * d.headroom / 65535.0) as f32;
            let native = {
                let v = m.data[y * w + x];
                if v.is_finite() { v } else { 0.0 }
            };
            let rgb = match m.cfa.color_at(x, y) {
                CfaColor::Red => [native, deq(green()), deq(diag())],
                CfaColor::Blue => [deq(diag()), deq(green()), native],
                CfaColor::Green => {
                    if m.cfa.color_at(x + 1, y) == CfaColor::Red {
                        [deq(ratio_h()), native, deq(ratio_v())]
                    } else {
                        [deq(ratio_v()), native, deq(ratio_h())]
                    }
                }
            };
            out.set_pixel(x, y, rgb);
        }
    }
    out
}

/// Inputs the vectors do not reach, for the differential tests against the plain transcriptions.
fn differential_scenes(cfa: Cfa, rng: &mut Rng) -> Vec<(&'static str, Mosaic)> {
    // Bright spikes on a black field: with a sharpened guide the outer taps undershoot below
    // one code (the lower clamp of the guide filter and of C).
    let mut spike = Mosaic::new(23, 19, cfa);
    spike.data[9 * 23 + 11] = 1.5;
    spike.data[3 * 23 + 4] = 0.8;
    // A bright block on black: three or more saturated columns push the sharpened guide above
    // the top of the 16-bit scale (the upper clamp of the guide filter, fixed form).
    let mut block = Mosaic::new(25, 21, cfa);
    for y in 7..14 {
        for x in 8..17 {
            block.data[y * 25 + x] = 4.0;
        }
    }
    // A noisy, bright texture with values above 1.0.
    let mut hdr = mosaic_of(&texture(37, 29), cfa);
    add_noise(&mut hdr, 2e-4, 1e-7, [2.0, 1.0, 1.6], 11);
    for v in hdr.data.iter_mut().step_by(7) {
        *v *= 5.0;
    }
    // Random samples, negatives, NaN and infinity, odd sizes.
    let (w, h) = (
        3 + (rng.next_u64() % 40) as usize,
        3 + (rng.next_u64() % 30) as usize,
    );
    let mut noisy = Mosaic::new(w, h, cfa);
    for v in &mut noisy.data {
        *v = -0.05 + 6.0 * rng.next_f32() * rng.next_f32() * rng.next_f32();
    }
    noisy.data[0] = f32::NAN;
    let last = noisy.data.len() - 1;
    noisy.data[last] = f32::INFINITY;
    // Values far beyond any real frame.
    let mut huge = Mosaic::new(15, 13, cfa);
    for v in &mut huge.data {
        *v = rng.next_f32() * 1.0e6;
    }
    vec![
        ("spike", spike),
        ("block", block),
        ("hdr", hdr),
        ("noisy", noisy),
        ("huge", huge),
    ]
}

/// The read-noise models the differential tests run with: none, a typical one (p = 10) and a very
/// noisy one (p = 1500).
fn differential_noise() -> [NoiseDn; 3] {
    [
        NoiseDn::ZERO,
        NoiseDn::from_model(&NoiseModel::new(2e-4, 1e-7)),
        NoiseDn::new(0.0, 3000.0),
    ]
}

#[test]
fn the_float_form_equals_a_plain_transcription_of_the_spec_byte_for_byte() {
    let mut rng = Rng::new(2026);
    let mut checked = 0;
    for cfa in Cfa::ALL {
        for (name, m) in differential_scenes(cfa, &mut rng) {
            for detail in [-50, 0, 35, 100] {
                for noise in differential_noise() {
                    let fast = log_guided::demosaic_float(&m, noise, detail);
                    let plain = naive_float(&m, noise, detail);
                    assert!(
                        bits_equal(&fast, &plain),
                        "{name} {cfa:?} detail {detail} noise {noise:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 4 * 5 * 4 * 3);
}

#[test]
fn the_fixed_form_equals_a_plain_transcription_of_the_spec_byte_for_byte() {
    let mut rng = Rng::new(2027);
    let mut checked = 0;
    for cfa in Cfa::ALL {
        for (name, m) in differential_scenes(cfa, &mut rng) {
            for detail in [-50, 0, 35, 100] {
                for headroom_log2 in [0u8, 2, 4] {
                    for noise in differential_noise() {
                        let fast = log_guided::demosaic_fixed(&m, noise, detail, headroom_log2);
                        let plain = naive_fixed(&m, noise, detail, headroom_log2);
                        assert!(
                            bits_equal(&fast, &plain),
                            "{name} {cfa:?} detail {detail} H {} noise {noise:?}",
                            1u32 << headroom_log2
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 4 * 5 * 4 * 3 * 3);
}

#[test]
fn a_mosaic_with_the_wrong_data_length_gives_a_black_image_of_its_size() {
    // Documented deviation from the oracle (which returns an empty image): the stated size, black.
    let mut m = Mosaic::new(8, 6, Cfa::GRBG);
    m.data.fill(0.5);
    m.data.truncate(7);
    for form in Form::BOTH {
        let out = form.run(&m, finite_noise(1e-8), 0, 1);
        assert_eq!((out.width, out.height), (8, 6), "{form:?}");
        assert_eq!(out.data.len(), 3 * 8 * 6, "{form:?}");
        assert!(out.data.iter().all(|&v| v == 0.0), "{form:?}");
        assert_eq!(out.place, m.place, "{form:?}");
    }
}

#[test]
fn extreme_values_and_offsets_stay_finite() {
    // Balanced mosaics above 1.0 are routine; values far beyond any real frame, and a read-noise
    // offset at its cap (p = 32767), must not produce NaN or infinity in either form, and the
    // native samples still pass through.
    let mut rng = Rng::new(9);
    for (scale, b) in [(1.0e4f32, 1e-7), (1.0e20, 1e-7), (2.0, 1.0), (1.0e4, 1.0)] {
        for cfa in [Cfa::RGGB, Cfa::GBRG] {
            let mut m = Mosaic::new(24, 20, cfa);
            for v in &mut m.data {
                *v = rng.next_f32() * scale;
            }
            for form in Form::BOTH {
                let out = form.run(&m, finite_noise(b), 0, 4);
                assert!(
                    out.data.iter().all(|v| v.is_finite()),
                    "{form:?} scale {scale} b {b}: non-finite output"
                );
                for y in 0..m.height {
                    for x in 0..m.width {
                        let c = cfa.color_at(x, y).plane();
                        assert_eq!(
                            out.plane(c)[y * m.width + x].to_bits(),
                            m.data[y * m.width + x].to_bits(),
                            "{form:?} native sample ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn zero_and_negative_fields() {
    // A field of zeros: constant guide, output 0.
    let zero = Mosaic::new(16, 16, Cfa::GRBG);
    for form in Form::BOTH {
        let out = form.run(&zero, finite_noise(1e-8), 0, 2);
        assert!(out.data.iter().all(|&v| v == 0.0), "{form:?}");
    }
    // Interpolation and the guide see max(v, 0): a negative field keeps its native samples and
    // gives 0 elsewhere.
    let mut neg = Mosaic::new(16, 16, Cfa::RGGB);
    neg.data.fill(-0.03);
    for form in Form::BOTH {
        let out = form.run(&neg, finite_noise(1e-8), 0, 2);
        for y in 0..16 {
            for x in 0..16 {
                let own = neg.cfa.color_at(x, y).plane();
                for c in 0..3 {
                    let v = out.plane(c)[y * 16 + x];
                    if c == own {
                        assert_eq!(v, -0.03, "{form:?} native");
                    } else {
                        assert_eq!(v, 0.0, "{form:?} interpolated");
                    }
                }
            }
        }
    }
}

#[test]
fn the_default_headroom_rule_reads_the_frame_maximum() {
    let mut m = Mosaic::new(8, 8, Cfa::RGGB);
    m.data.fill(0.5);
    assert_eq!(LogGuidedParams::for_mosaic(&m).headroom_log2, 0);
    m.data[10] = 3.4;
    assert_eq!(LogGuidedParams::for_mosaic(&m).headroom_log2, 2);
    m.data[11] = f32::NAN; // not a candidate
    assert_eq!(LogGuidedParams::for_mosaic(&m).headroom_log2, 2);
    m.data[12] = 40.0;
    assert_eq!(LogGuidedParams::for_mosaic(&m).headroom_log2, 4);
    let nothing = Mosaic::new(0, 0, Cfa::RGGB);
    assert_eq!(
        LogGuidedParams::for_mosaic(&nothing),
        LogGuidedParams::default()
    );
}
