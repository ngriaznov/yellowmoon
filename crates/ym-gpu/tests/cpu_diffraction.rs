//! R8 diffraction deconvolution (`docs/algorithms/03-diffraction-deconvolution.md`).
//!
//! * `vectors_*`: every vector of `tests/cpu_data/diffraction_deconvolution/` within its stated
//!   tolerance (planner vectors 1e-6, image vectors 2e-5, exact vectors bit-equal), and the
//!   planner fields (`active`, `radius`, `gauss_radius`, `halo_px` exactly).
//! * the property tests of spec section 6.2, ported from the reference's self-test: flat fields
//!   and identities exact, tiling parity at the plan's halo, square symmetries, scale behaviour,
//!   kernel sweep and the plans tabulated in section 2.7, no NaN or Inf, the luma of the output,
//!   flat-patch noise, sharpening of a blurred step and the Bessel function.
//! * determinism: the banded run equals the whole-buffer run bit for bit, with one thread and with
//!   four.
//!
//! The scenes are the ones of the reference (splitmix64 with Irwin-Hall normals), so the numbers
//! the spec quotes (error 5.30 to 3.31, rise 2.62 to 2.04 px, noise x1.019) can be compared.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_gpu::cpu::chain::{Chain, RawFrame};
use ym_gpu::cpu::demosaic::Method;
use ym_gpu::cpu::denoise::BilateralParams;
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::normalize::Levels;
use ym_gpu::cpu::sharpen::diffraction::{
    DiffractionParams, DiffractionPlan, MAX_HALO_PX, SPEC_LUMA, bessel_j1,
};
use ym_gpu::cpu::sharpen::{CaptureSharpen, SharpenParams};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::{Halo, Rect, Stage, tile_grid};
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{Cfa, NoiseDn, RgbImage, mirror};
use ym_noise::model::NoiseModel;

// ---------------------------------------------------------------------------------------------
// Scenes and helpers of the reference
// ---------------------------------------------------------------------------------------------

/// splitmix64 with Irwin-Hall normals (arithmetic only, identical on every platform).
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
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.uniform();
        }
        s - 6.0
    }
}

/// The reference's `Params` for a scene: square pixels, gains (2, 1, 1.5), about ISO 100.
#[derive(Clone, Copy, Debug)]
struct Scene {
    pitch_um: f64,
    f_number: f64,
    scale: f64,
    strength: f32,
    g0: [f32; 3],
    a: f64,
    b: f64,
    sigma_row: f64,
    q: f64,
}

fn base(pitch_um: f64, f_number: f64) -> Scene {
    Scene {
        pitch_um,
        f_number,
        scale: 1.0,
        strength: 1.0,
        g0: [2.0, 1.0, 1.5],
        a: 2.5e-5,
        b: 4e-8,
        sigma_row: 5e-5,
        q: 0.0,
    }
}

impl Scene {
    fn params(&self) -> DiffractionParams {
        DiffractionParams::new(self.pitch_um, self.f_number)
            .with_scale(self.scale)
            .with_strength(self.strength)
    }

    fn model(&self) -> NoiseModel {
        NoiseModel {
            a: self.a,
            b: self.b,
            delta: [0.0; 4],
            sigma_row: self.sigma_row,
            q: self.q,
        }
    }

    fn plan(&self) -> Option<DiffractionPlan> {
        DiffractionPlan::new(
            &self.params(),
            self.g0,
            &NoiseDn::from_model(&self.model()),
            SPEC_LUMA,
        )
    }

    fn active_plan(&self) -> DiffractionPlan {
        self.plan().expect("an active plan")
    }

    fn floor_var(&self) -> f64 {
        self.b + self.sigma_row * self.sigma_row + self.q * self.q / 12.0
    }
}

fn run(plan: &DiffractionPlan, img: &RgbImage) -> RgbImage {
    plan.apply_whole(img)
}

/// Poisson-Gaussian noise from the model, in unbalanced units, independent per channel.
fn add_noise(img: &mut RgbImage, s: &Scene, seed: u64) {
    let mut rng = Rng(seed);
    let n = img.plane_len();
    let floor_var = s.floor_var();
    for c in 0..3 {
        let g = f64::from(s.g0[c]);
        for ix in 0..n {
            let u = f64::from(img.data[c * n + ix]) / g;
            let sd = (s.a * u.max(0.0) + floor_var).sqrt();
            img.data[c * n + ix] = (g * (u + sd * rng.gauss())) as f32;
        }
    }
}

/// Blurs every channel with the plan's kernel (mirror borders): simulated diffraction. An
/// independent, direct implementation of the convolution.
fn psf_blur(img: &RgbImage, plan: &DiffractionPlan) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let r = plan.radius() as isize;
    let kw = 2 * plan.radius() + 1;
    let mut out = img.clone();
    for c in 0..3 {
        let src = img.plane(c);
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

/// A smooth edge between two balanced colours; `dist` is the signed distance in pixels.
fn fill_edge(img: &mut RgbImage, col0: [f32; 3], col1: [f32; 3], nx: f64, ny: f64, offset: f64) {
    for y in 0..img.height {
        for x in 0..img.width {
            let dist = nx * (x as f64 + 0.5) + ny * (y as f64 + 0.5) - offset;
            let t = (dist + 0.5).clamp(0.0, 1.0);
            let px: [f32; 3] = std::array::from_fn(|c| {
                ((1.0 - t) * f64::from(col0[c]) + t * f64::from(col1[c])) as f32
            });
            img.set_pixel(x, y, px);
        }
    }
}

fn flat(w: usize, h: usize, rgb: [f32; 3]) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for (c, &v) in rgb.iter().enumerate() {
        img.plane_mut(c).fill(v);
    }
    img
}

/// The reference's textured scene.
fn textured(w: usize, h: usize, seed: u64, lo: f64, hi: f64) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = Rng(seed);
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f64, y as f64);
            let base = 0.5
                * (1.0 + (0.37 * xf + 0.11 * yf * yf * 0.05).sin() * (0.23 * yf - 0.05 * xf).cos());
            let edge = if (x * 3 + y * 2) % 37 < 18 { 0.25 } else { 0.0 };
            for c in 0..3 {
                let t = (0.7 * base + edge + 0.08 * rng.uniform()).clamp(0.0, 1.0);
                img.plane_mut(c)[y * w + x] = (lo + (hi - lo) * t * (0.8 + 0.2 * c as f64)) as f32;
            }
        }
    }
    img
}

/// A flat patch whose luma noise matches the model exactly: the same normal deviate on all three
/// channels (fully correlated channels, the assumption of spec section 2.3).
fn matched_noise_patch(s: &Scene, w: usize, h: usize, level: f64, seed: u64) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = Rng(seed);
    let n = img.plane_len();
    let floor_var = s.floor_var();
    for ix in 0..n {
        let z = rng.gauss();
        for c in 0..3 {
            let g = f64::from(s.g0[c]);
            let u = level / g;
            img.data[c * n + ix] = (g * (u + (s.a * u + floor_var).sqrt() * z)) as f32;
        }
    }
    img
}

fn luma_of(img: &RgbImage, ix: usize) -> f64 {
    let n = img.plane_len();
    f64::from(SPEC_LUMA[0]) * f64::from(img.data[ix])
        + f64::from(SPEC_LUMA[1]) * f64::from(img.data[n + ix])
        + f64::from(SPEC_LUMA[2]) * f64::from(img.data[2 * n + ix])
}

fn luma_sd(img: &RgbImage) -> f64 {
    let n = img.plane_len();
    let (mut m, mut m2) = (0.0, 0.0);
    for ix in 0..n {
        let y = luma_of(img, ix);
        m += y;
        m2 += y * y;
    }
    m /= n as f64;
    (m2 / n as f64 - m * m).sqrt()
}

fn transform(img: &RgbImage, t: usize) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut out = if t == 0 {
        RgbImage::new(h, w)
    } else {
        RgbImage::new(w, h)
    };
    for y in 0..h {
        for x in 0..w {
            let px = img.pixel(x, y);
            match t {
                0 => out.set_pixel(y, x, px),
                1 => out.set_pixel(w - 1 - x, y, px),
                _ => out.set_pixel(x, h - 1 - y, px),
            }
        }
    }
    out
}

fn crop(img: &RgbImage, r: Rect) -> RgbImage {
    img.extract(r)
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/diffraction_deconvolution")
}

fn vectors() -> Vec<(String, Value)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(vector_dir())
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("vector file");
            let v: Value = serde_json::from_str(&text).expect("vector json");
            (
                p.file_stem().expect("stem").to_string_lossy().into_owned(),
                v,
            )
        })
        .collect()
}

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}"))
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

/// The plan a vector's `params` describe, or `None` when inactive.
fn plan_of(params: &Value) -> Option<DiffractionPlan> {
    let g0 = floats(&params["g0"]);
    let luma = floats(&params["luma"]);
    let n = &params["noise"];
    let model = NoiseModel {
        a: num(n, "a"),
        b: num(n, "b"),
        delta: [0.0; 4],
        sigma_row: num(n, "sigma_row"),
        q: num(n, "q"),
    };
    assert_eq!(num(params, "working_scale"), 65535.0);
    let p = DiffractionParams {
        pixel_pitch_um: [
            num(params, "pixel_pitch_x_um"),
            num(params, "pixel_pitch_y_um"),
        ],
        f_number: num(params, "f_number"),
        scale: num(params, "scale"),
        strength: num(params, "strength") as f32,
    };
    DiffractionPlan::new(
        &p,
        [g0[0], g0[1], g0[2]],
        &NoiseDn::from_model(&model),
        [luma[0], luma[1], luma[2]],
    )
}

fn check_plan_fields(name: &str, plan: &Option<DiffractionPlan>, expected: &Value, tol: f64) {
    let active = num(expected, "active") as i32;
    assert_eq!(plan.is_some(), active == 1, "{name}: active");
    let Some(plan) = plan else { return };
    assert_eq!(
        plan.radius() as f64,
        num(expected, "radius"),
        "{name}: radius"
    );
    assert_eq!(
        plan.gauss_radius() as f64,
        num(expected, "gauss_radius"),
        "{name}: gauss_radius"
    );
    assert_eq!(
        plan.halo_px() as f64,
        num(expected, "halo_px"),
        "{name}: halo_px"
    );
    for (got, key) in [
        (f64::from(plan.gauss_sigma()), "gauss_sigma"),
        (plan.sigma0_dn(), "sigma0_dn"),
        (plan.sigma1_dn(), "sigma1_dn"),
    ] {
        let d = (got - num(expected, key)).abs();
        assert!(
            d <= tol,
            "{name}: {key} got {got}, vector {} (diff {d:e})",
            num(expected, key)
        );
    }
}

#[test]
fn vectors_planner() {
    let mut count = 0;
    let mut worst = 0.0f64;
    for (name, v) in vectors() {
        if !v["input"].is_null() {
            continue;
        }
        count += 1;
        let tol = num(&v, "tolerance");
        let e = &v["expected"];
        let plan = plan_of(&v["params"]);
        check_plan_fields(&name, &plan, e, tol);
        if let Some(plan) = &plan {
            for (got, key) in [(plan.r80_um(), "r80_um"), (plan.support_um(), "support_um")] {
                let d = (got - num(e, key)).abs();
                worst = worst.max(d);
                assert!(d <= tol, "{name}: {key} got {got}, vector {}", num(e, key));
            }
            let want = floats(&e["kernel"]);
            let kw = 2 * plan.radius() + 1;
            assert_eq!(want.len(), kw * kw, "{name}: kernel size");
            let d = max_diff(plan.kernel(), &want);
            worst = worst.max(d);
            assert!(d <= tol, "{name}: kernel differs by {d:e}");
            let sum: f64 = plan.kernel().iter().map(|&k| f64::from(k)).sum();
            assert!((sum - 1.0).abs() < 1e-5, "{name}: kernel sums to {sum}");
        } else {
            assert_eq!(
                num(&v["params"], "f_number"),
                64.0,
                "{name}: only f/64 is inactive"
            );
        }
    }
    assert_eq!(count, 10, "planner vectors");
    eprintln!("planner vectors: {count}, worst difference {worst:e}");
}

#[test]
fn vectors_images() {
    let mut count = 0;
    let mut worst = 0.0f64;
    for (name, v) in vectors() {
        if v["input"].is_null() {
            continue;
        }
        count += 1;
        let tol = num(&v, "tolerance");
        let input = &v["input"];
        let (w, h) = (num(input, "width") as usize, num(input, "height") as usize);
        assert_eq!(input["layout"], "planar_rgb");
        let data = floats(&input["data"]);
        let img = RgbImage::from_planar(w, h, data.clone()).expect("input image");
        let e = &v["expected"];
        assert_eq!(
            (num(e, "width") as usize, num(e, "height") as usize),
            (w, h)
        );
        let want = floats(&e["data"]);
        let plan = plan_of(&v["params"]);
        let whole = plan
            .as_ref()
            .map_or_else(|| img.clone(), |p| p.apply_whole(&img));
        let banded = plan.as_ref().map_or_else(|| img.clone(), |p| p.apply(&img));
        assert!(
            bits_equal(&whole.data, &banded.data),
            "{name}: banded run differs from whole"
        );
        assert_eq!(whole.place, img.place);
        if tol == 0.0 {
            // Exact vectors: the stage is an identity (flat field, inactive stage).
            assert_eq!(
                num(e, "active") as i32,
                i32::from(plan.is_some()),
                "{name}: active"
            );
            assert!(
                bits_equal(&whole.data, &want),
                "{name}: not bit-equal to the vector"
            );
            assert!(
                bits_equal(&whole.data, &data),
                "{name}: not bit-equal to the input"
            );
        } else {
            check_plan_fields(&name, &plan, e, 1e-6);
            let d = max_diff(&whole.data, &want);
            worst = worst.max(d);
            assert!(d <= tol, "{name}: max abs difference {d:e} > {tol:e}");
        }
    }
    assert_eq!(count, 13, "image vectors");
    eprintln!("image vectors: {count}, worst difference {worst:e}");
}

// ---------------------------------------------------------------------------------------------
// Spec section 6.2
// ---------------------------------------------------------------------------------------------

#[test]
fn bessel_j1_agrees_with_bessels_integral_on_0_to_120() {
    // J1(x) = (1/pi) integral_0^pi cos(t - x sin t) dt. The integrand is smooth and periodic, so
    // the trapezoid rule converges geometrically: 2000 panels give machine precision for x <= 120.
    let reference = |x: f64| {
        let n = 2000;
        let mut s = 0.0;
        for i in 0..=n {
            let t = std::f64::consts::PI * f64::from(i) / f64::from(n);
            let wgt = if i == 0 || i == n { 0.5 } else { 1.0 };
            s += wgt * (t - x * t.sin()).cos();
        }
        s / f64::from(n)
    };
    let mut worst = 0.0f64;
    let mut x = 0.0f64;
    while x <= 120.0 {
        worst = worst.max((bessel_j1(x) - reference(x)).abs());
        x += 0.173;
    }
    for x in [11.999, 12.0, 12.001, 25.0, 113.9, 114.0, 120.0] {
        worst = worst.max((bessel_j1(x) - reference(x)).abs());
    }
    eprintln!("bessel J1: worst error against Bessel's integral on [0, 120]: {worst:e}");
    assert!(worst < 1e-10, "worst error {worst:e}");
}

#[test]
fn flat_fields_come_back_bit_exact() {
    // Levels 0, 0.18 grey, a colour, above white (1.9, 1.3, 1.6 and 3.5, 2.0, 2.9), slight negatives.
    let levels: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [0.18, 0.18, 0.18],
        [0.4, 0.2, 0.1],
        [1.9, 1.3, 1.6],
        [-0.02, -0.01, -0.03],
        [3.5, 2.0, 2.9],
    ];
    for f in [2.8, 11.0, 22.0] {
        let plan = base(3.76, f).active_plan();
        for lv in levels {
            let img = flat(33, 27, lv);
            assert!(
                bits_equal(&run(&plan, &img).data, &img.data),
                "f/{f} {lv:?}"
            );
            assert!(bits_equal(&plan.apply(&img).data, &img.data));
        }
    }
}

#[test]
fn strength_zero_and_an_inactive_stage_are_identities() {
    let mut s = base(3.76, 11.0);
    s.strength = 0.0;
    let mut tex = textured(40, 36, 7, 0.02, 0.9);
    add_noise(&mut tex, &s, 99);
    assert!(
        bits_equal(&run(&s.active_plan(), &tex).data, &tex.data),
        "strength 0"
    );
    // NaN strength reads as 1, not as 0: the stage then does something.
    s.strength = f32::NAN;
    assert!(
        !bits_equal(&run(&s.active_plan(), &tex).data, &tex.data),
        "NaN strength"
    );
    // Inactive: f/64, f/0.99, NaN, pitch <= 0 (no plan, so the caller leaves the image alone).
    for (pitch, f) in [
        (3.76, 64.0),
        (3.76, 0.99),
        (3.76, f64::NAN),
        (0.0, 8.0),
        (-3.0, 8.0),
    ] {
        assert!(base(pitch, f).plan().is_none(), "pitch {pitch} f/{f}");
    }
}

#[test]
fn tiles_with_the_plans_halo_equal_the_whole_frame() {
    // f/2.8, f/22 and f/45 (2.4 um), tiles in the interior and on the frame edges. Reference:
    // difference 0; contract: <= 1e-4.
    for f in [2.8, 22.0, 45.0] {
        let s = base(2.4, f);
        let plan = s.active_plan();
        let mut img = textured(96, 80, 11, 0.01, 1.2);
        add_noise(&mut img, &s, 5);
        let whole = run(&plan, &img);
        let halo = plan.halo_px();
        assert_eq!(halo, 10 + plan.radius() + plan.gauss_radius());
        let frame = Rect::whole(96, 80);
        let cores = [
            Rect::new(0, 0, 32, 24),
            Rect::new(32, 24, 32, 24),
            Rect::new(64, 56, 32, 24),
            Rect::new(17, 9, 29, 31),
            Rect::new(40, 0, 20, 80),
        ];
        let mut worst = 0.0f64;
        let mut worst_short = 0.0f64;
        for core in cores {
            for (extra, exact) in [(0usize, true), (4, false)] {
                let ctx = if exact { halo } else { halo - extra };
                let ext = core.grow(Halo::uniform(ctx), frame);
                let tile = plan.apply_whole(&crop(&img, ext));
                let rel = core.relative_to(&ext);
                let mut d = 0.0f64;
                for c in 0..3 {
                    for y in 0..core.h {
                        for x in 0..core.w {
                            let a = whole.plane(c)[(core.y + y) * 96 + core.x + x];
                            let b = tile.plane(c)[(rel.y + y) * ext.w + rel.x + x];
                            d = d.max((f64::from(a) - f64::from(b)).abs());
                        }
                    }
                }
                if exact {
                    worst = worst.max(d);
                } else {
                    worst_short = worst_short.max(d);
                }
            }
        }
        eprintln!(
            "f/{f} (halo {halo} px): tile - whole = {worst:e}; with 4 px less context {worst_short:e}"
        );
        assert!(worst <= 1e-4, "f/{f}: {worst}");
        assert_eq!(
            worst, 0.0,
            "f/{f}: at the exact halo the tile equals the whole bit for bit"
        );
    }
}

#[test]
fn transpose_and_flips_commute_with_the_stage() {
    let s = base(3.76, 16.0);
    let plan = s.active_plan();
    let mut img = textured(44, 38, 21, 0.02, 0.8);
    add_noise(&mut img, &s, 8);
    let base_out = run(&plan, &img);
    let mut worst = 0.0f64;
    for t in 0..3 {
        let out = run(&plan, &transform(&img, t));
        worst = worst.max(max_diff(&out.data, &transform(&base_out, t).data));
    }
    eprintln!("symmetries: max difference {worst:e}");
    assert!(worst < 1e-5, "{worst}");
}

#[test]
fn scaling_the_input_and_the_noise_model_scales_the_output() {
    let mut s = base(3.76, 11.0);
    s.a = 4e-4;
    s.b = 4e-6;
    s.sigma_row = 0.0;
    let mut img = textured(40, 40, 3, 0.05, 0.6);
    add_noise(&mut img, &s, 4);
    let full = run(&s.active_plan(), &img);
    // alpha = 1/2: a -> a/2, b -> b/4.
    let mut half_scene = s;
    half_scene.a *= 0.5;
    half_scene.b *= 0.25;
    let mut half = img.clone();
    half.data.iter_mut().for_each(|v| *v *= 0.5);
    let out = run(&half_scene.active_plan(), &half);
    let worst = out
        .data
        .iter()
        .zip(&full.data)
        .map(|(o, f)| (2.0 * f64::from(*o) - f64::from(*f)).abs())
        .fold(0.0, f64::max);
    eprintln!("scale 1/2: {worst:e}");
    assert!(worst < 1e-5, "{worst}");
    // alpha = 0.7: equal up to rounding.
    let mut s7 = s;
    s7.a *= 0.7;
    s7.b *= 0.49;
    let mut scaled = img.clone();
    scaled.data.iter_mut().for_each(|v| *v *= 0.7);
    let out7 = run(&s7.active_plan(), &scaled);
    let worst7 = out7
        .data
        .iter()
        .zip(&full.data)
        .map(|(o, f)| (f64::from(*o) / 0.7 - f64::from(*f)).abs())
        .fold(0.0, f64::max);
    eprintln!("scale 0.7: {worst7:e}");
    assert!(worst7 < 1e-5, "{worst7}");
}

#[test]
fn the_kernel_is_normalised_symmetric_and_the_plans_stay_in_range() {
    let mut max_halo = 0;
    for pitch in [1.0, 2.4, 3.76, 5.93, 8.4, 12.0] {
        for f in [
            1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0, 45.0, 50.0,
        ] {
            for scale in [1.0, 0.5, 0.25] {
                let mut s = base(pitch, f);
                s.scale = scale;
                let plan = s.active_plan();
                let r = plan.radius();
                let kw = 2 * r + 1;
                assert!(
                    (1..=7).contains(&r),
                    "pitch {pitch} f/{f} scale {scale}: R {r}"
                );
                assert!((0.2..=3.0).contains(&plan.gauss_sigma()));
                assert_eq!(plan.halo_px(), 10 + r + plan.gauss_radius());
                assert!(plan.halo_px() <= MAX_HALO_PX);
                assert_eq!(plan.halo_px(), s.params().halo_px(), "the scheduler's halo");
                let k = plan.kernel();
                let sum: f64 = k.iter().map(|&v| f64::from(v)).sum();
                assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
                for y in 0..kw {
                    for x in 0..kw {
                        let v = k[y * kw + x];
                        assert!(v >= 0.0);
                        assert_eq!(v, k[y * kw + (kw - 1 - x)]);
                        assert_eq!(v, k[(kw - 1 - y) * kw + x]);
                        assert_eq!(v, k[x * kw + y]);
                    }
                }
                max_halo = max_halo.max(plan.halo_px());
            }
        }
    }
    eprintln!("planner sweep: largest halo {max_halo}");
    assert!(max_halo <= MAX_HALO_PX);
}

#[test]
fn the_plans_of_the_spec_come_out_as_listed() {
    // Spec section 2.7: R exact, sigma_G +-0.006, then r_G, halo, r80 and the centre tap.
    // (pitch, f, R, sigma_G, r_G, halo, r80 um, centre tap)
    #[allow(clippy::type_complexity)]
    let table: [(f64, f64, usize, f64, usize, usize, f64, f64); 8] = [
        (3.76, 2.8, 2, 0.229, 1, 13, 1.30, 0.764),
        (3.76, 5.6, 3, 0.427, 2, 15, 2.60, 0.555),
        (3.76, 8.0, 3, 0.500, 2, 15, 3.71, 0.412),
        (3.76, 11.0, 4, 0.661, 3, 17, 5.11, 0.280),
        (3.76, 16.0, 5, 0.791, 3, 18, 7.43, 0.166),
        (3.76, 22.0, 5, 0.994, 4, 19, 10.22, 0.104),
        (3.76, 50.0, 6, 1.610, 6, 22, 23.31, 0.041),
        (5.93, 11.0, 3, 0.469, 2, 15, 5.11, 0.470),
    ];
    for (pitch, f, r, sg, rg, halo, r80, centre) in table {
        let plan = base(pitch, f).active_plan();
        let tag = format!("{pitch} um f/{f}");
        assert_eq!(plan.radius(), r, "{tag}: R");
        assert!(
            (f64::from(plan.gauss_sigma()) - sg).abs() <= 0.006,
            "{tag}: sigma_G {}",
            plan.gauss_sigma()
        );
        assert_eq!(plan.gauss_radius(), rg, "{tag}: r_G");
        assert_eq!(plan.halo_px(), halo, "{tag}: halo");
        assert!(
            (plan.r80_um() - r80).abs() <= 0.006,
            "{tag}: r80 {}",
            plan.r80_um()
        );
        let kw = 2 * r + 1;
        let tap = f64::from(plan.kernel()[r * kw + r]);
        assert!((tap - centre).abs() <= 0.0006, "{tag}: centre tap {tap}");
    }
}

#[test]
fn no_nan_or_inf_on_extreme_inputs() {
    let sizes = [(1, 1), (2, 3), (7, 5), (1, 17), (31, 2), (48, 40)];
    let mut seed = 1u64;
    for f in [1.0, 2.8, 8.0, 22.0, 50.0] {
        for &(w, h) in &sizes {
            let mut rng = Rng(seed);
            seed += 1;
            let mut img = RgbImage::new(w, h);
            for v in &mut img.data {
                let u = rng.uniform();
                *v = (if u < 0.1 {
                    -0.05 * rng.uniform()
                } else if u > 0.9 {
                    4.0 * rng.uniform()
                } else {
                    u
                }) as f32;
            }
            for variant in 0..3 {
                let mut s = base(3.76, f);
                if variant == 1 {
                    // No noise model: sigma0 sits at the 1 DN floor.
                    (s.a, s.b, s.sigma_row, s.q) = (0.0, 0.0, 0.0, 0.0);
                }
                if variant == 2 {
                    s.a = 5e-3;
                    s.b = 1e-5;
                    s.strength = 3.0;
                }
                let out = run(&s.active_plan(), &img);
                assert!(
                    out.data.iter().all(|v| v.is_finite()),
                    "f/{f} {w}x{h} variant {variant}"
                );
                assert_eq!((out.width, out.height), (w, h));
            }
        }
    }
}

#[test]
fn the_output_luma_equals_the_corrected_luma() {
    // Spec 6.2 item 8: the luma of the output (same weights) equals L_s to 2e-6 relative, and the
    // stage did change some pixels.
    let s = base(3.76, 11.0);
    let plan = s.active_plan();
    let mut img = textured(40, 32, 9, 0.0005, 1.4);
    add_noise(&mut img, &s, 3);
    let (l, y) = plan.corrected_luma(&img);
    let out = run(&plan, &img);
    let (mut worst, mut changed) = (0.0f64, 0);
    for ix in 0..img.plane_len() {
        let want = l[ix] / 65535.0;
        let got = luma_of(&out, ix);
        worst = worst.max((got - want).abs() / want.abs().max(1.0));
        if (l[ix] - f64::from(y[ix])).abs() > 1.0 {
            changed += 1;
        }
    }
    eprintln!("output luma vs L_s: max relative difference {worst:e}, {changed} pixels changed");
    assert!(worst < 2e-6, "{worst}");
    assert!(changed > 50, "only {changed} pixels changed");
}

#[test]
fn flat_patch_noise_stays_flat_at_f8() {
    // Spec 6.2 item 9: at f/8 the flat-patch noise grows by at most 5 % (reference x1.019 worst
    // of ISO 100 and 6400). Beyond about f/11 at 3.76 um it grows; reported, not gated.
    let mut worst = 0.0f64;
    for (a, b) in [(2.5e-5, 4e-8), (2e-3, 2e-6)] {
        let mut s = base(3.76, 8.0);
        s.a = a;
        s.b = b;
        let img = matched_noise_patch(&s, 64, 64, 0.15, 123);
        let out = run(&s.active_plan(), &img);
        worst = worst.max(luma_sd(&out) / luma_sd(&img));
    }
    eprintln!("f/8 model-matched flat noise: luma sd x{worst:.3}");
    assert!(worst <= 1.05, "x{worst}");
    for f in [11.0, 16.0, 22.0] {
        let s = base(3.76, f);
        let img = matched_noise_patch(&s, 64, 64, 0.15, 123);
        let out = run(&s.active_plan(), &img);
        eprintln!(
            "f/{f} model-matched flat noise: luma sd x{:.3}",
            luma_sd(&out) / luma_sd(&img)
        );
    }
}

/// 10-90 % rise distance of the mean luma profile along the middle rows of a vertical step.
fn rise_10_90(img: &RgbImage, lo: f64, hi: f64) -> f64 {
    let (w, h) = (img.width, img.height);
    let mut prof = vec![0.0f64; w];
    let rows = h / 2;
    let y0 = h / 4;
    for y in y0..y0 + rows {
        for (x, p) in prof.iter_mut().enumerate() {
            *p += luma_of(img, y * w + x) / rows as f64;
        }
    }
    let cross = |level: f64| {
        for x in 1..w {
            let (a, b) = (prof[x - 1], prof[x]);
            if a < level && b >= level {
                return (x - 1) as f64 + (level - a) / (b - a) + 0.5;
            }
        }
        0.0
    };
    cross(lo + 0.9 * (hi - lo)) - cross(lo + 0.1 * (hi - lo))
}

#[test]
fn a_blurred_step_gets_closer_to_the_sharp_step_and_steeper() {
    // Spec 6.2 item 10: f/16, 3.76 um, ISO 100 noise: squared error 5.30 -> 3.31 and rise
    // 2.62 -> 2.04 px in the reference.
    let s = base(3.76, 16.0);
    let plan = s.active_plan();
    let mut sharp = RgbImage::new(48, 32);
    fill_edge(&mut sharp, [0.05; 3], [0.6; 3], 1.0, 0.0, 24.0);
    let mut blurred = psf_blur(&sharp, &plan);
    add_noise(&mut blurred, &s, 17);
    let out = run(&plan, &blurred);
    let sse = |a: &RgbImage| -> f64 {
        a.data
            .iter()
            .zip(&sharp.data)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
            .sum()
    };
    let (mse_in, mse_out) = (sse(&blurred), sse(&out));
    // The step goes from 0.05 to 0.6 in all three channels, so its luma is the channel value.
    let (r_in, r_out) = (rise_10_90(&blurred, 0.05, 0.6), rise_10_90(&out, 0.05, 0.6));
    eprintln!(
        "blurred step: squared error {mse_in:.3} -> {mse_out:.3}, 10-90 rise {r_in:.2} -> {r_out:.2} px"
    );
    assert!(mse_out < 0.8 * mse_in, "{mse_in} -> {mse_out}");
    assert!(r_out < r_in, "{r_in} -> {r_out}");
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn banded_and_threaded_runs_are_byte_identical() {
    let s = base(3.76, 11.0);
    let plan = s.active_plan();
    // Taller than one band (128 rows) so that several bands and their halos are involved.
    let mut img = textured(70, 300, 5, 0.01, 1.1);
    add_noise(&mut img, &s, 6);
    let whole = run(&plan, &img);
    let banded = plan.apply(&img);
    assert!(
        bits_equal(&whole.data, &banded.data),
        "banded differs from whole"
    );
    let in_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| plan.apply(&img))
    };
    let one = in_pool(1);
    let four = in_pool(4);
    assert!(bits_equal(&one.data, &four.data), "1 thread vs 4 threads");
    assert!(bits_equal(&one.data, &whole.data));
    let again = plan.apply(&img);
    assert!(bits_equal(&again.data, &banded.data), "repeatable");
}

#[test]
fn banded_runs_equal_the_whole_buffer_at_the_largest_halos_and_on_odd_sizes() {
    // The 128-row bands carry the plan's halo as context. Check it where it is largest (24 px: R 7
    // and r_G 7, at 2.4 um and f/45 and at 0.8 um and f/40), where it is smallest, and on sizes that
    // put a band edge one row from the end of the frame (129, 257), that leave a short last band
    // (258, 267, 300) or are narrower than the halo (5 and 9 columns: every window is mirrored
    // more than once).
    let plans = [(2.4, 45.0), (0.8, 40.0), (3.76, 16.0), (8.4, 1.0)];
    let sizes = [
        (23usize, 300usize),
        (5, 267),
        (9, 129),
        (9, 128),
        (40, 257),
        (61, 258),
    ];
    let mut worst_halo = 0;
    for (pitch, f) in plans {
        let s = base(pitch, f);
        let plan = s.active_plan();
        worst_halo = worst_halo.max(plan.halo_px());
        for (k, &(w, h)) in sizes.iter().enumerate() {
            let mut img = textured(w, h, 40 + k as u64, 0.01, 1.2);
            add_noise(&mut img, &s, 50 + k as u64);
            let whole = run(&plan, &img);
            let banded = plan.apply(&img);
            assert!(
                bits_equal(&whole.data, &banded.data),
                "{pitch} um f/{f} (halo {}) {w}x{h}: banded differs from whole",
                plan.halo_px()
            );
        }
    }
    assert_eq!(worst_halo, 24, "the sweep includes the largest halo");
    // And with one and with four threads, on the plan with the largest halo.
    let s = base(2.4, 45.0);
    let plan = s.active_plan();
    let mut img = textured(23, 300, 77, 0.01, 1.2);
    add_noise(&mut img, &s, 78);
    let whole = run(&plan, &img);
    for threads in [1, 4] {
        let out = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| plan.apply(&img));
        assert!(bits_equal(&out.data, &whole.data), "{threads} threads");
    }
}

#[test]
fn the_planner_cache_gives_the_same_plan_to_every_thread() {
    // The planner keeps eight geometries in a global cache; the tile scheduler asks for the halo of
    // every tile from several threads. More keys than entries, asked in different orders by four
    // threads at once: every answer equals the one a single thread got.
    use rayon::prelude::*;
    let keys: Vec<(f64, f64)> = (0..24)
        .map(|i| (2.0 + 0.17 * f64::from(i), 2.8 + 1.9 * f64::from(i % 9)))
        .collect();
    let expected: Vec<(usize, usize, usize, Vec<f32>)> = keys
        .iter()
        .map(|&(p, f)| {
            let plan = base(p, f).active_plan();
            (
                plan.halo_px(),
                plan.radius(),
                plan.gauss_radius(),
                plan.kernel().to_vec(),
            )
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    pool.install(|| {
        (0..96usize).into_par_iter().for_each(|i| {
            // Different threads walk the keys from different offsets.
            let ix = (i * 7 + i / 5) % keys.len();
            let (p, f) = keys[ix];
            let params = DiffractionParams::new(p, f);
            let want = &expected[ix];
            assert_eq!(params.halo_px(), want.0, "halo of key {ix}");
            let plan = base(p, f).active_plan();
            assert_eq!(
                (plan.halo_px(), plan.radius(), plan.gauss_radius()),
                (want.0, want.1, want.2),
                "plan of key {ix}"
            );
            assert!(
                bits_equal(plan.kernel(), &want.3),
                "kernel of key {ix} differs"
            );
        });
    });
}

// ---------------------------------------------------------------------------------------------
// R8 in the chain
// ---------------------------------------------------------------------------------------------

/// The chain with every raw stage on, sharpening as given, R9 optional.
fn chain_with(sharpen: Option<CaptureSharpen>, noise: Option<NoiseDn>) -> Chain {
    Chain {
        levels: Levels {
            black: [512.0, 514.0, 513.0, 511.0],
            white: [16000.0; 4],
        },
        vignette: Some(
            VignetteProfile::from_tag(&[0, -400, -1500, -3500, -6000]).expect("vignette"),
        ),
        gains: [2.0, 1.0, 1.6],
        highlight_recovery: true,
        method: Method::Menon,
        denoise: Some(BilateralParams::default()),
        frame_denoise: None,
        noise_reduction: NoiseReduction {
            luminance: 40.0,
            color: 60.0,
            detail: 70.0,
        },
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise,
        sharpen,
        lens: None,
    }
}

fn raw_scene(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let img = synth::natural(w, h, seed);
    let m = synth::mosaic_from_rgb(&img, Cfa::RGGB);
    let mut rng = synth::Rng::new(seed + 7);
    m.data
        .iter()
        .map(|&v| {
            let x = 512.0 + (v * 1.3 + 0.004 * rng.gaussian()) * 15488.0;
            x.clamp(0.0, 16000.0) as u16
        })
        .collect()
}

fn frame<'a>(data: &'a [u16], w: usize, h: usize) -> RawFrame<'a> {
    RawFrame {
        width: w,
        height: h,
        cfa: Cfa::RGGB,
        data,
        qstep: None,
    }
}

fn model_dn() -> NoiseDn {
    NoiseDn::from_model(&NoiseModel {
        a: 2.0e-4,
        b: 4.0e-8,
        delta: [0.0; 4],
        sigma_row: 1.0e-4,
        q: 2.0e-4,
    })
}

#[test]
fn the_chain_runs_the_diffraction_stage_on_the_r7_image_with_its_gains_and_noise() {
    let (w, h) = (97, 71);
    let data = raw_scene(w, h, 5);
    let raw = frame(&data, w, h);
    let params = DiffractionParams::new(3.76, 11.0);
    let noise = model_dn();

    let before = chain_with(None, Some(noise)).run(&raw).expect("without R8");
    let chain = chain_with(Some(CaptureSharpen::Diffraction(params)), Some(noise));
    let got = chain.run(&raw).expect("with R8");
    let plan = DiffractionPlan::new(&params, chain.gains, &noise, SPEC_LUMA).expect("active");
    let want = plan.apply(&before);
    assert!(
        bits_equal(&got.data, &want.data),
        "the chain applies the plan to the R7 output"
    );
    assert!(
        max_diff(&got.data, &before.data) > 1e-3,
        "the stage changed something"
    );

    // The noise model matters (the weights come from it): zero noise and no model agree, a rich
    // one differs.
    let zero = chain_with(
        Some(CaptureSharpen::Diffraction(params)),
        Some(NoiseDn::ZERO),
    );
    let none = chain_with(Some(CaptureSharpen::Diffraction(params)), None);
    assert!(bits_equal(
        &zero.run(&raw).expect("zero").data,
        &none.run(&raw).expect("none").data
    ));
    assert!(!bits_equal(&zero.run(&raw).expect("zero").data, &got.data));
}

#[test]
fn an_inactive_diffraction_stage_leaves_the_image_alone_and_needs_no_halo() {
    let (w, h) = (64, 48);
    let data = raw_scene(w, h, 8);
    let raw = frame(&data, w, h);
    let before = chain_with(None, Some(model_dn()))
        .run(&raw)
        .expect("without R8");
    for params in [
        DiffractionParams::new(3.76, 0.0),
        DiffractionParams::new(3.76, 64.0),
        DiffractionParams::new(0.0, 8.0),
        DiffractionParams::new(3.76, f64::NAN),
    ] {
        let chain = chain_with(Some(CaptureSharpen::Diffraction(params)), Some(model_dn()));
        assert!(
            bits_equal(&chain.run(&raw).expect("run").data, &before.data),
            "{params:?}"
        );
        let stages = chain.stages();
        let Some(Stage::Sharpen(m)) = stages.iter().find(|s| matches!(s, Stage::Sharpen(_))) else {
            panic!("no sharpen stage");
        };
        assert_eq!(m.halo_px(), 0);
    }
}

#[test]
fn richardson_lucy_stays_what_the_chain_ran_before() {
    // CaptureSharpen::RichardsonLucy is the old Option<SharpenParams>: same stage, same bits as
    // the explicit call on the R7 output, and the same halo as before (the 24 px cap).
    let (w, h) = (97, 71);
    let data = raw_scene(w, h, 5);
    let raw = frame(&data, w, h);
    let p = SharpenParams {
        amount: 120.0,
        radius_px: 0.8,
    };
    let chain = chain_with(Some(p.into()), Some(model_dn()));
    let before = chain_with(None, Some(model_dn()))
        .run(&raw)
        .expect("without R8");
    let want =
        ym_gpu::cpu::sharpen::richardson_lucy(&before, &p, &ym_gpu::cpu::NoiseSigma::Scalar(0.01));
    assert!(bits_equal(&chain.run(&raw).expect("run").data, &want.data));
    let stages = chain.stages();
    let halo = stages.iter().find_map(|s| match s {
        Stage::Sharpen(m) => Some(m.halo_px()),
        _ => None,
    });
    assert_eq!(halo, Some(24));
}

#[test]
fn tiled_chains_equal_the_whole_frame_with_the_diffraction_halo() {
    let (w, h) = (150, 110);
    let data = raw_scene(w, h, 3);
    let raw = frame(&data, w, h);
    for f in [2.8, 11.0, 22.0] {
        let chain = chain_with(
            Some(CaptureSharpen::Diffraction(DiffractionParams::new(3.76, f))),
            Some(model_dn()),
        );
        let prepared = chain.prepare(&raw).expect("prepare");
        let whole = chain
            .run_tile(&raw, &prepared, Rect::whole(w, h))
            .expect("whole");
        let mut tiled = RgbImage::new(w, h);
        for core in tile_grid(w, h, 52, 46, 1) {
            let t = chain.run_tile(&raw, &prepared, core).expect("tile");
            assert_eq!((t.width, t.height), (core.w, core.h));
            tiled.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
        }
        let d = max_diff(&whole.data, &tiled.data);
        eprintln!("f/{f}: chain tiles vs whole {d:e}");
        assert!(d <= 1e-4, "f/{f}: {d}");
    }
}

#[test]
fn the_diffraction_halo_is_not_capped_at_the_richardson_lucy_limit() {
    // A lower bound of the dependency radius of the stage, found by changing one input pixel and
    // looking at where the output moves. It is not the radius itself: the outermost taps of the
    // kernel and of the Gaussians are so small that a change that far out stays below the f32
    // resolution of the output (a model-consistent scene reaches 18 px of 24 at f/45, 2.4 um).
    // The declared halo is the sum of the radii of the chain (spec section 1.5), so it is an upper
    // bound; the tile tests above show that it is enough for bit equality.
    let s = base(2.4, 45.0);
    let plan = s.active_plan();
    let halo = plan.halo_px();
    assert_eq!(halo, 24, "this plan is the largest of the sweep");
    let (w, h) = (81, 81);
    let img = textured(w, h, 4, 0.05, 0.9);
    let base_out = run(&plan, &img);
    let mut poked = img.clone();
    let (cx, cy) = (40usize, 40usize);
    for c in 0..3 {
        poked.plane_mut(c)[cy * w + cx] += 0.3;
    }
    let out = run(&plan, &poked);
    let mut reach = 0usize;
    for y in 0..h {
        for x in 0..w {
            if base_out.pixel(x, y) != out.pixel(x, y) {
                reach = reach.max(x.abs_diff(cx)).max(y.abs_diff(cy));
            }
        }
    }
    eprintln!("a changed pixel moves the output up to {reach} px away (halo {halo})");
    assert!(reach <= halo, "reach {reach} > halo {halo}");
    assert!(
        reach > 18,
        "the perturbation must reach well past the RL-style radii: {reach}"
    );
}

// ---------------------------------------------------------------------------------------------
// Spec 7.1 item 1: the noise-gate variant (an A/B candidate, not the specified behaviour)
// ---------------------------------------------------------------------------------------------

#[test]
fn the_noise_gate_variant_agrees_up_to_f8_and_holds_flat_noise_beyond() {
    // q_w from the plan: 0.07 at f/2.8, 0.40 at f/8, 0.75 at f/16, 0.94 at f/50 (3.76 um).
    for (f, q) in [(2.8, 0.07), (8.0, 0.40), (16.0, 0.75), (50.0, 0.94)] {
        let plan = base(3.76, f).active_plan();
        assert!(
            (plan.white_noise_residual() - q).abs() < 0.01,
            "f/{f}: {}",
            plan.white_noise_residual()
        );
        assert_eq!(plan.gate_factor(), 1.0, "the specified plan is not scaled");
    }
    // At f/8 and below the factor is 1 (2 q_w < 1), so the variant is the specified stage bit for bit.
    for f in [2.8, 5.6, 8.0] {
        let s = base(3.76, f);
        let spec = s.active_plan();
        let variant = s.active_plan().with_noise_gate(true);
        assert_eq!(variant.gate_factor(), 1.0, "f/{f}");
        let mut img = textured(48, 40, 3, 0.02, 0.9);
        add_noise(&mut img, &s, 4);
        assert!(
            bits_equal(&run(&spec, &img).data, &run(&variant, &img).data),
            "f/{f}"
        );
    }
    // Beyond it the gate opens wider and flat noise stays near the input (spec: x1.07 to x1.08 from
    // f/11 to f/32 against x1.21 to x1.79 and more for the specified gate).
    let mut report = Vec::new();
    for f in [11.0, 16.0, 22.0, 32.0] {
        let s = base(3.76, f);
        let spec = s.active_plan();
        let variant = s.active_plan().with_noise_gate(true);
        assert!(variant.gate_factor() > 1.0, "f/{f}");
        assert!(variant.clone().with_noise_gate(false).gate_factor() == 1.0);
        let img = matched_noise_patch(&s, 64, 64, 0.15, 123);
        let growth = |plan: &DiffractionPlan| luma_sd(&run(plan, &img)) / luma_sd(&img);
        let (gs, gv) = (growth(&spec), growth(&variant));
        report.push(format!(
            "f/{f}: specified x{gs:.3}, variant x{gv:.3} (f = {:.2})",
            variant.gate_factor()
        ));
        assert!(
            gv < gs,
            "f/{f}: the variant must grow the noise less ({gv} vs {gs})"
        );
        assert!(gv < 1.12, "f/{f}: variant growth {gv}");
    }
    eprintln!(
        "noise gate variant, flat patch with luma noise matching the model:\n  {}",
        report.join("\n  ")
    );
}
