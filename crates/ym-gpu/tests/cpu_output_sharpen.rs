//! Output sharpening, the CPU port (`docs/algorithms/04-output-sharpening.md`).
//!
//! * `vectors_*`: every vector of `tests/cpu_data/output_sharpening/` (the C++ oracle's output,
//!   never hand-edited): the derived plan (stages, kernels, halo) and the sharpened pixels within
//!   the vector's tolerance (0 means bit-exact), planar and interleaved.
//! * `property_*`: the spec's section 6.2 property tests, mirrored from the reference's self-test:
//!   flat fields, identity, tiling parity, flip and transpose equivariance, scale behaviour,
//!   robustness, hue preservation, halo suppression, plan derivation.
//! * `preset_*`: the screen and print presets (section 6.3): derived sigma, threshold, halo
//!   fraction and receptive field at full size and at a 2048 px long edge from 6000 px.
//! * `edge_profile_*`: a soft slanted edge and a noisy flat patch measured before and after.
//! * `determinism_*`: one thread and four threads give the same bytes.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::{
    ExtraStage, Method, Mode, OutputSharpen, OutputSharpenError, SharpenPlan, gaussian_kernel,
    scale_for_edges,
};

// ------------------------------------------------------------------------------------------------
// Test images (the reference's `testimg`)
// ------------------------------------------------------------------------------------------------

/// splitmix64 with Box-Muller, as the reference's generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(11400714819323198485);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(13787848793156543929);
        z = (z ^ (z >> 27)).wrapping_mul(10723151780598845931);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = ((self.next() >> 11) as f64 + 1.0) * (1.0 / 9007199254740992.0);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Irwin-Hall noise: twelve uniforms summed, minus 6 (mean 0, variance 1). Only additions, so
    /// it has the same bits on every platform, unlike `ln` and `cos` of `gauss`.
    fn gauss_portable(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

fn flat(w: usize, h: usize, r: f32, g: f32, b: f32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.set_pixel(x, y, [r, g, b]);
        }
    }
    img
}

fn add_noise(img: &mut RgbImage, sigma: f64, seed: u64) {
    let mut rng = Rng(seed);
    for v in &mut img.data {
        *v = (f64::from(*v) + sigma * rng.gauss()) as f32;
    }
}

fn random_image(w: usize, h: usize, lo: f32, hi: f32, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut img = RgbImage::new(w, h);
    for v in &mut img.data {
        *v = (f64::from(lo) + f64::from(hi - lo) * rng.uniform()) as f32;
    }
    img
}

/// Edges, a bar pattern and a diagonal line, without noise.
fn busy_scene(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut v = if x * 3 < w {
                0.08f32
            } else if x * 3 < 2 * w {
                0.42
            } else {
                0.2
            };
            if y > h / 2 && (x / 2) % 2 == 0 {
                v += 0.25;
            }
            if x + y == w / 2 || x + y == w / 2 + 1 {
                v = 0.9;
            }
            img.set_pixel(x, y, [v * 1.1, v, v * 0.7]);
        }
    }
    img
}

/// The busy test image of the reference: the scene plus Gaussian noise of sigma 0.004.
fn busy(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut img = busy_scene(w, h);
    add_noise(&mut img, 0.004, seed);
    img
}

/// The busy scene with noise of sigma 0.004 made of additions only (no `ln`, no `cos`), so every
/// platform builds the same bits. The digest test needs that: it fails on a single input bit.
fn busy_portable(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut img = busy_scene(w, h);
    let mut rng = Rng(seed);
    for v in &mut img.data {
        *v = (f64::from(*v) + 0.004 * rng.gauss_portable()) as f32;
    }
    img
}

fn crop(img: &RgbImage, x0: usize, y0: usize, w: usize, h: usize) -> RgbImage {
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.set_pixel(x, y, img.pixel(x0 + x, y0 + y));
        }
    }
    out
}

fn transpose(img: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(img.height, img.width);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set_pixel(y, x, img.pixel(x, y));
        }
    }
    out
}

fn flip_x(img: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set_pixel(img.width - 1 - x, y, img.pixel(x, y));
        }
    }
    out
}

fn interleave(img: &RgbImage) -> Vec<f32> {
    let n = img.plane_len();
    let mut out = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[3 * i + c] = img.data[c * n + i];
        }
    }
    out
}

fn max_abs_diff(a: &RgbImage, b: &RgbImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0, f64::max)
}

fn bit_equal(a: &RgbImage, b: &RgbImage) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn all_finite(img: &RgbImage) -> bool {
    img.data.iter().all(|v| v.is_finite())
}

/// The reference's seven parameter sets of its self-test, with the output scale of each.
fn param_sets() -> Vec<(OutputSharpen, f32)> {
    let mut v = Vec::new();
    let mut p = OutputSharpen::screen();
    v.push((p.clone(), 1.0)); // screen preset: method 1, halo 60
    p.method = Method::HardThreshold;
    p.radius_px = 2.5;
    p.halo_pct = 0.0;
    p.threshold_levels = 2.0;
    v.push((p.clone(), 1.0)); // method 0, largest screen radius, no halo
    p.method = Method::GradientRamp;
    p.halo_pct = 60.0;
    v.push((p.clone(), 1.0)); // method 2, largest screen radius, halo
    p.method = Method::SmoothGain;
    v.push((p.clone(), 0.37)); // downscaled export
    v.push((OutputSharpen::print(), 1.0));
    let mut q = OutputSharpen::print();
    q.print_distance = 30000.0;
    q.method = Method::GradientRamp;
    q.halo_pct = 40.0;
    v.push((q, 1.0)); // print, long kernels, method 2 with halo
    let mut e = OutputSharpen::screen();
    let mut c = ExtraStage {
        method: Method::HardThreshold,
        auto_method: true,
        amount_pct: 150.0,
        radius_px: 0.6,
        threshold_levels: 1.0,
        halo_pct: 20.0,
    };
    e.extra_stages.push(c);
    c.method = Method::GradientRamp;
    c.auto_method = false;
    c.radius_px = 2.0;
    c.halo_pct = 0.0;
    e.extra_stages.push(c);
    v.push((e, 1.0)); // three stages
    v
}

fn sharpen(p: &OutputSharpen, scale: f32, img: &RgbImage) -> RgbImage {
    p.apply(img, scale).expect("sharpen")
}

// ------------------------------------------------------------------------------------------------
// Vectors
// ------------------------------------------------------------------------------------------------

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/output_sharpening")
}

/// Every vector, sorted by file name.
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

fn f32_of(v: &Value, key: &str) -> f32 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}")) as f32
}

fn method_of(v: &Value) -> Method {
    Method::from_id(v["method_id"].as_u64().expect("method_id") as u32).expect("method")
}

/// The parameters and output scale of a vector.
fn params_of(v: &Value) -> (OutputSharpen, f32) {
    let p = &v["params"];
    let mode = match p["mode"].as_str().expect("mode") {
        "screen" => Mode::Screen,
        "print" => Mode::Print,
        other => panic!("mode {other}"),
    };
    let lw = p["luma_weights"].as_array().expect("luma weights");
    let extra_stages = p["extra_stages"]
        .as_array()
        .expect("extra stages")
        .iter()
        .map(|e| ExtraStage {
            method: method_of(e),
            auto_method: e["auto_method"].as_bool().expect("auto_method"),
            amount_pct: f32_of(e, "amount_pct"),
            radius_px: f32_of(e, "radius_px"),
            threshold_levels: f32_of(e, "threshold_levels"),
            halo_pct: f32_of(e, "halo_pct"),
        })
        .collect();
    (
        OutputSharpen {
            mode,
            method: method_of(p),
            amount_pct: f32_of(p, "amount_pct"),
            radius_px: f32_of(p, "radius_px"),
            threshold_levels: f32_of(p, "threshold_levels"),
            halo_pct: f32_of(p, "halo_pct"),
            print_distance: f32_of(p, "print_distance"),
            luma_weights: [
                lw[0].as_f64().expect("w") as f32,
                lw[1].as_f64().expect("w") as f32,
                lw[2].as_f64().expect("w") as f32,
            ],
            extra_stages,
            // The reference compares exactly; the tie tolerance is ours (spec 7.1 item 5).
            halo_tie_tolerance: 0.0,
        },
        f32_of(p, "scale"),
    )
}

fn image_of(v: &Value) -> RgbImage {
    assert_eq!(v["layout"].as_str(), Some("planar_rgb"));
    let w = v["width"].as_u64().expect("width") as usize;
    let h = v["height"].as_u64().expect("height") as usize;
    let data: Vec<f32> = v["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|x| x.as_f64().expect("sample") as f32)
        .collect();
    RgbImage::from_planar(w, h, data).expect("planar image")
}

fn kernel_diff(ours: &[f64], theirs: &Value) -> f64 {
    let theirs: Vec<f64> = theirs
        .as_array()
        .expect("kernel")
        .iter()
        .map(|x| x.as_f64().expect("tap"))
        .collect();
    assert_eq!(ours.len(), theirs.len(), "kernel length");
    ours.iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Compares a plan with the vector's `derived` block. Returns the largest kernel tap difference.
fn check_derived(name: &str, v: &Value, plan: &SharpenPlan) -> f64 {
    let d = &v["params"]["derived"];
    assert_eq!(
        plan.max_taps() as u64,
        d["max_taps"].as_u64().expect("max_taps"),
        "{name}: max taps"
    );
    assert_eq!(
        plan.halo_enabled(),
        d["halo_enabled"].as_bool().expect("halo_enabled"),
        "{name}: halo enabled"
    );
    assert_eq!(
        plan.halo_px() as u64,
        d["halo_px"].as_u64().expect("halo_px"),
        "{name}: halo px"
    );
    let mut worst = kernel_diff(&plan.halo_kernel(), &d["halo_kernel"]);
    let stages = d["stages"].as_array().expect("stages");
    assert_eq!(plan.stages().len(), stages.len(), "{name}: stage count");
    for (i, (st, ref_st)) in plan.stages().iter().zip(stages).enumerate() {
        assert_eq!(
            st.output,
            ref_st["output"].as_bool().expect("output"),
            "{name} stage {i}: output"
        );
        assert_eq!(st.method, method_of(ref_st), "{name} stage {i}: method");
        for (what, ours) in [
            ("amount", st.amount),
            ("sigma", st.sigma),
            ("threshold", st.threshold),
            ("halo", st.halo),
        ] {
            assert_eq!(
                ours.to_bits(),
                f32_of(ref_st, what).to_bits(),
                "{name} stage {i}: {what}"
            );
        }
        worst = worst.max(kernel_diff(&plan.kernel(i), &ref_st["kernel"]));
        worst = worst.max(kernel_diff(&plan.gain_kernel(i), &ref_st["gain_kernel"]));
    }
    worst
}

#[test]
fn vectors_match_the_oracle() {
    let all = vectors();
    assert_eq!(all.len(), 16, "the package has 16 vectors");
    let mut worst_kernel = 0.0f64;
    for (name, v) in &all {
        let (os, scale) = params_of(v);
        let plan = os.plan(scale).expect("plan");
        worst_kernel = worst_kernel.max(check_derived(name, v, &plan));

        let input = image_of(&v["input"]);
        let expected = image_of(&v["expected"]);
        let tol = v["tolerance"].as_f64().expect("tolerance");
        let out = os.apply(&input, scale).expect("apply");
        let err = max_abs_diff(&out, &expected);
        eprintln!("{name:<34} max |error| {err:.3e} (tolerance {tol:.0e})");
        if tol == 0.0 {
            assert!(bit_equal(&out, &expected), "{name}: must be bit-exact");
        } else {
            assert!(err <= tol, "{name}: error {err:e} above tolerance {tol:e}");
        }

        // Interleaved gives the same bits as planar.
        let inter = os
            .apply_interleaved(input.width, input.height, &interleave(&input), scale)
            .expect("apply interleaved");
        let planar_inter = interleave(&out);
        assert!(
            inter
                .iter()
                .zip(&planar_inter)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: interleaved differs from planar"
        );
    }
    eprintln!("largest kernel tap difference against the oracle: {worst_kernel:.3e}");
    assert!(worst_kernel <= 1e-15, "kernels differ by {worst_kernel:e}");
}

// ------------------------------------------------------------------------------------------------
// Properties (spec section 6.2)
// ------------------------------------------------------------------------------------------------

#[test]
fn property_flat_fields_come_back_bit_exact() {
    let colours = [
        [0.2f32, 0.18, 0.12],
        [1.7, 1.25, -0.02],
        [0.0, 0.0, 0.0],
        [3e-5, 2e-5, 1e-5],
    ];
    for (p, s) in param_sets() {
        for c in colours {
            let img = flat(23, 17, c[0], c[1], c[2]);
            assert!(
                bit_equal(&sharpen(&p, s, &img), &img),
                "flat {c:?} changed (mode {:?}, method {:?})",
                p.mode,
                p.method
            );
        }
    }
}

#[test]
fn property_amount_zero_is_the_identity_and_creates_no_stage() {
    let img = random_image(19, 13, -0.05, 2.0, 7);
    let z = OutputSharpen::screen().with_amount_pct(0.0);
    assert!(bit_equal(&sharpen(&z, 1.0, &img), &img));
    assert!(z.plan(1.0).expect("plan").stages().is_empty());
    assert_eq!(z.halo_px(1.0), Ok(0));
    // Just below 0.1 % is no stage either; 0.1 % is one.
    let tiny = OutputSharpen::screen().with_amount_pct(0.09);
    assert!(tiny.plan(1.0).expect("plan").stages().is_empty());
    assert!(bit_equal(&sharpen(&tiny, 1.0, &img), &img));
    let just = OutputSharpen::screen().with_amount_pct(0.1);
    assert_eq!(just.plan(1.0).expect("plan").stages().len(), 1);
    // A radius of 0.09 px or less makes no stage.
    let mut r = OutputSharpen::screen();
    r.radius_px = 0.09;
    assert!(r.plan(1.0).expect("plan").stages().is_empty());
    // The in-place and interleaved forms also leave the pixels alone.
    let mut inter = interleave(&img);
    z.apply_interleaved_in_place(img.width, img.height, &mut inter, 1.0)
        .expect("in place");
    assert_eq!(inter, interleave(&img));
}

/// Runs the stage on tiles (the core grown by the plan's halo, clipped to the frame) and pastes
/// the cores.
fn tiled(p: &OutputSharpen, scale: f32, img: &RgbImage, tw: usize, th: usize) -> RgbImage {
    let halo = p.halo_px(scale).expect("halo");
    let mut out = RgbImage::new(img.width, img.height);
    let mut y0 = 0;
    while y0 < img.height {
        let y1 = (y0 + th).min(img.height);
        let mut x0 = 0;
        while x0 < img.width {
            let x1 = (x0 + tw).min(img.width);
            let gx0 = x0.saturating_sub(halo);
            let gy0 = y0.saturating_sub(halo);
            let gx1 = (x1 + halo).min(img.width);
            let gy1 = (y1 + halo).min(img.height);
            let tile = crop(img, gx0, gy0, gx1 - gx0, gy1 - gy0);
            let t = sharpen(p, scale, &tile);
            for y in y0..y1 {
                for x in x0..x1 {
                    out.set_pixel(x, y, t.pixel(x - gx0, y - gy0));
                }
            }
            x0 = x1;
        }
        y0 = y1;
    }
    out
}

#[test]
fn property_tiles_of_core_plus_halo_match_the_whole_frame_bit_exactly() {
    let img = busy(71, 66, 11);
    for (i, (p, s)) in param_sets().into_iter().enumerate() {
        let whole = sharpen(&p, s, &img);
        let t = tiled(&p, s, &img, 17, 23);
        let d = max_abs_diff(&whole, &t);
        eprintln!(
            "param set {i}: halo {} px, max |tiled - whole| = {d:.3e}",
            p.halo_px(s).expect("halo")
        );
        assert!(
            bit_equal(&whole, &t),
            "param set {i}: tiled differs by {d:e}"
        );
    }
}

#[test]
fn property_tiling_with_the_mode_bound_halo_is_also_exact() {
    // A caller that pads with the mode bound (21 px screen, 41 px print) instead of the exact
    // figure gets the same pixels.
    let img = busy(70, 60, 3);
    for (p, s) in param_sets() {
        let whole = sharpen(&p, s, &img);
        let halo = p.max_halo_px();
        let (x0, y0, w, h) = (20usize, 13usize, 25usize, 19usize);
        let gx0 = x0.saturating_sub(halo);
        let gy0 = y0.saturating_sub(halo);
        let gx1 = (x0 + w + halo).min(img.width);
        let gy1 = (y0 + h + halo).min(img.height);
        let tile = sharpen(&p, s, &crop(&img, gx0, gy0, gx1 - gx0, gy1 - gy0));
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                assert_eq!(
                    whole.pixel(x, y).map(f32::to_bits),
                    tile.pixel(x - gx0, y - gy0).map(f32::to_bits),
                    "pixel ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn property_flip_and_transpose_equivariance() {
    let img = busy(37, 29, 5);
    let (mut worst_t, mut worst_f) = (0.0f64, 0.0f64);
    for (p, s) in param_sets() {
        let a = sharpen(&p, s, &img);
        worst_t = worst_t.max(max_abs_diff(
            &transpose(&a),
            &sharpen(&p, s, &transpose(&img)),
        ));
        worst_f = worst_f.max(max_abs_diff(&flip_x(&a), &sharpen(&p, s, &flip_x(&img))));
    }
    eprintln!(
        "transpose {worst_t:.3e}, flip {worst_f:.3e} (limit 1e-5; the reference gives 1.2e-7 and 2.4e-7)"
    );
    assert!(worst_t <= 1e-5 && worst_f <= 1e-5);
}

#[test]
fn property_threshold_zero_is_homogeneous_of_degree_one() {
    let img = busy(33, 27, 3);
    for (mut p, s) in param_sets() {
        p.threshold_levels = 0.0;
        for e in &mut p.extra_stages {
            e.threshold_levels = 0.0;
        }
        let a = sharpen(&p, s, &img);
        for k in [2.0f32, 0.25] {
            let mut scaled = img.clone();
            scaled.data.iter_mut().for_each(|v| *v *= k);
            let mut b = sharpen(&p, s, &scaled);
            b.data.iter_mut().for_each(|v| *v /= k);
            assert!(bit_equal(&a, &b), "f({k} x) != {k} f(x) for {:?}", p.method);
        }
    }
}

/// A clean chart like the golden set's: a white patch clipped flat at 1, a dark band under it and
/// grey patches, with horizontal and vertical edges and no noise, so whole rows and columns tie.
fn chart_scene(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if y < h / 3 {
                if x < w / 2 { 1.0f32 } else { 0.62 }
            } else if y == h / 3 {
                0.012
            } else if x < w / 3 {
                0.18
            } else {
                0.035
            };
            img.set_pixel(x, y, [v, v * 0.97, v * 0.9]);
        }
    }
    img
}

/// `img` with every sample moved by up to `ulps` units in the last place (seeded): how the
/// device's `f32` colour chain differs from the CPU's `f64` one.
fn jitter_ulps(img: &RgbImage, ulps: u64, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut out = img.clone();
    for v in &mut out.data {
        let k = (rng.next() % (2 * ulps + 1)) as i64 - ulps as i64;
        *v = f32::from_bits((i64::from(v.to_bits()) + k) as u32);
    }
    out
}

/// A checkerboard of 1 px cells: every window sample ties with a mirrored one.
fn checkerboard(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if (x + y) % 2 == 0 { 0.7f32 } else { 0.1 };
            img.set_pixel(x, y, [v, v * 0.9, v * 0.8]);
        }
    }
    img
}

#[test]
fn last_bit_changes_of_the_input_do_not_flip_the_limiter_on_ties() {
    // Spec 3.5 and 7.1 item 5: with exact comparisons, values that tie (the rows of a clean edge, a
    // clipped patch, a checkerboard) stop tying when the input moves by an ulp, as the device's f32
    // colour chain moves it, a rank count flips and the pixel moves by up to 1e-1: the golden check
    // on the device failed on that. With the tie tolerance the limiter is continuous, and the output
    // moves about as much as the input did.
    let mut worst = (0.0f64, 0.0f64);
    for (name, img) in [
        ("chart", chart_scene(96, 64)),
        ("checkerboard", checkerboard(64, 48)),
    ] {
        for (p, s) in param_sets() {
            if p.halo_pct == 0.0 && p.extra_stages.iter().all(|e| e.halo_pct == 0.0) {
                continue; // no limiter, no comparisons
            }
            let mut exact = p.clone();
            exact.halo_tie_tolerance = 0.0;
            for seed in 1..=4 {
                let moved = jitter_ulps(&img, 8, seed);
                let d_tol = max_abs_diff(&sharpen(&p, s, &img), &sharpen(&p, s, &moved));
                let d_exact = max_abs_diff(&sharpen(&exact, s, &img), &sharpen(&exact, s, &moved));
                eprintln!(
                    "{name:<12} {:?} s {s}: tolerance {d_tol:.3e}, exact {d_exact:.3e}",
                    p.method
                );
                worst = (worst.0.max(d_tol), worst.1.max(d_exact));
            }
        }
    }
    // Measured: 4.9e-6 with the tolerance, 1.5e-1 with exact comparisons.
    assert!(worst.0 <= 2e-5, "with the tie tolerance: {:e}", worst.0);
    assert!(worst.1 >= 1e-2, "exact comparisons: {:e}", worst.1);
}

#[test]
fn property_robustness() {
    for (p, s) in param_sets() {
        assert!(all_finite(&sharpen(
            &p,
            s,
            &random_image(29, 31, -0.05, 3.0, 99)
        )));
        for w in [1usize, 2, 3, 7] {
            for h in [1usize, 2, 5] {
                let img = random_image(w, h, 0.0, 1.5, (w * 10 + h) as u64);
                assert!(all_finite(&sharpen(&p, s, &img)), "{w} x {h}");
            }
        }
    }
    // A collapsed kernel with method 2 is a finite no-op.
    let mut t = OutputSharpen::screen();
    t.radius_px = 0.0927; // sigma about 0.05: the kernel collapses to one tap
    t.method = Method::GradientRamp;
    t.threshold_levels = 5.0;
    let img = busy(21, 21, 1);
    let out = sharpen(&t, 0.3, &img);
    assert!(all_finite(&out) && max_abs_diff(&out, &img) < 1e-6);
    // Validation rejects scale > 1, scale <= 0 and NaN.
    let p = OutputSharpen::screen();
    for bad in [1.5f32, 0.0, -0.1, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                p.plan(bad),
                Err(OutputSharpenError::Invalid { name: "scale", .. })
            ),
            "scale {bad}"
        );
    }
    let mut nan = OutputSharpen::screen();
    nan.threshold_levels = f32::NAN;
    assert!(matches!(
        nan.plan(1.0),
        Err(OutputSharpenError::Invalid {
            name: "threshold_levels",
            ..
        })
    ));
}

#[test]
fn property_parameter_validation() {
    let ok = OutputSharpen::screen();
    assert_eq!(ok.validate(), Ok(()));
    assert_eq!(OutputSharpen::print().validate(), Ok(()));
    type Edit = fn(&mut OutputSharpen);
    let cases: [(&str, Edit); 16] = [
        ("amount_pct", |p| p.amount_pct = f32::NAN),
        ("amount_pct", |p| p.amount_pct = -1.0),
        ("amount_pct", |p| p.amount_pct = 1000.5),
        ("radius_px", |p| p.radius_px = f32::NAN),
        ("radius_px", |p| p.radius_px = -0.5),
        ("threshold_levels", |p| p.threshold_levels = 100.5),
        ("threshold_levels", |p| p.threshold_levels = f32::INFINITY),
        ("halo_pct", |p| p.halo_pct = 101.0),
        ("halo_pct", |p| p.halo_pct = f32::NAN),
        ("print_distance", |p| p.print_distance = -1.0),
        ("print_distance", |p| p.print_distance = f32::NAN),
        ("luma_weights", |p| p.luma_weights = [0.3, 0.0, 0.1]),
        ("luma_weights", |p| p.luma_weights = [0.3, f32::NAN, 0.1]),
        ("luma_weights", |p| p.luma_weights = [0.3, 0.5, -0.1]),
        ("extra stage radius_px", |p| {
            p.extra_stages.push(ExtraStage {
                method: Method::SmoothGain,
                auto_method: false,
                amount_pct: 100.0,
                radius_px: 15.5,
                threshold_levels: 1.0,
                halo_pct: 0.0,
            })
        }),
        ("extra stage amount_pct", |p| {
            p.extra_stages.push(ExtraStage {
                method: Method::SmoothGain,
                auto_method: false,
                amount_pct: f32::NAN,
                radius_px: 1.0,
                threshold_levels: 1.0,
                halo_pct: 0.0,
            })
        }),
    ];
    for (name, edit) in cases {
        let mut p = OutputSharpen::screen();
        edit(&mut p);
        match p.validate() {
            Err(OutputSharpenError::Invalid { name: got, .. }) => assert_eq!(got, name),
            other => panic!("{name}: expected a rejection, got {other:?}"),
        }
        assert!(p.apply_interleaved(1, 1, &[0.0; 3], 1.0).is_err());
    }
    // Method numbers other than 0 to 2 are rejected.
    assert!(Method::from_id(3).is_err());
    assert_eq!(Method::from_id(2), Ok(Method::GradientRamp));
    // A radius above the screen cap is capped, not rejected; so is a long print distance.
    let mut cap = OutputSharpen::screen();
    cap.radius_px = 40.0;
    assert_eq!(cap.validate(), Ok(()));
    let a = cap.plan(1.0).expect("plan");
    let mut b = OutputSharpen::screen();
    b.radius_px = 2.5;
    assert_eq!(a.stages(), b.plan(1.0).expect("plan").stages());
}

#[test]
fn property_buffer_errors() {
    let p = OutputSharpen::screen();
    assert!(matches!(
        p.apply_interleaved(2, 2, &[0.0; 11], 1.0),
        Err(OutputSharpenError::BadLength {
            got: 11,
            expected: 12,
            ..
        })
    ));
    assert_eq!(
        p.apply_interleaved(0, 5, &[], 1.0),
        Err(OutputSharpenError::Empty)
    );
    let bad = RgbImage {
        width: 3,
        height: 2,
        place: ym_gpu::cpu::Placement::whole(3, 2),
        data: vec![0.0; 17],
    };
    assert!(matches!(
        p.apply(&bad, 1.0),
        Err(OutputSharpenError::BadLength { .. })
    ));
}

#[test]
fn property_channel_ratios_are_preserved() {
    let img = busy(31, 25, 8);
    let out = sharpen(&OutputSharpen::screen(), 1.0, &img);
    let n = img.plane_len();
    let mut changed = 0;
    for i in 0..n {
        let (r, g, b) = (
            f64::from(img.data[i]),
            f64::from(img.data[n + i]),
            f64::from(img.data[2 * n + i]),
        );
        let (r2, g2, b2) = (
            f64::from(out.data[i]),
            f64::from(out.data[n + i]),
            f64::from(out.data[2 * n + i]),
        );
        assert!((r2 * g - g2 * r).abs() <= 1e-6 * (r2 * g).abs() + 1e-12);
        assert!((b2 * g - g2 * b).abs() <= 1e-6 * (b2 * g).abs() + 1e-12);
        if (r2 - r).abs() > 1e-4 {
            changed += 1;
        }
    }
    assert!(changed > n / 20, "the image should actually be sharpened");
}

#[test]
fn property_halo_suppression_cuts_the_overshoot() {
    let (lo, hi) = (0.1f32, 0.5f32);
    let mut step = RgbImage::new(32, 8);
    for y in 0..8 {
        for x in 0..32 {
            let v = if x >= 16 { hi } else { lo };
            step.set_pixel(x, y, [v, v, v]);
        }
    }
    let mut p = OutputSharpen::screen();
    p.method = Method::HardThreshold;
    p.threshold_levels = 0.0;
    p.amount_pct = 300.0;
    p.radius_px = 1.5;
    let mut over = [0.0f64; 2];
    let mut under = [0.0f64; 2];
    for (j, halo) in [0.0f32, 100.0].into_iter().enumerate() {
        p.halo_pct = halo;
        let o = sharpen(&p, 1.0, &step);
        for x in 0..32 {
            let v = f64::from(o.pixel(x, 4)[1]);
            over[j] = over[j].max(v - 0.5);
            under[j] = under[j].max(0.1 - v);
        }
    }
    eprintln!(
        "step 0.1 to 0.5, amount 300, R 1.5: overshoot {:.4} -> {:.4}, undershoot {:.4} -> {:.4} with halo 100 (spec: 0.43 -> 0.007, 0.10 -> 0.007)",
        over[0], over[1], under[0], under[1]
    );
    assert!(over[0] > 0.05 && over[1] < 0.25 * over[0]);
    assert!(under[1] < 0.25 * under[0]);
    // The numbers of the spec (section 6.2): 0.43 -> 0.007 and 0.10 -> 0.007.
    assert!((over[0] - 0.43).abs() < 0.01 && (over[1] - 0.007).abs() < 0.002);
    assert!((under[0] - 0.10).abs() < 0.005 && (under[1] - 0.007).abs() < 0.002);
}

fn plan_of(p: &OutputSharpen, s: f32) -> SharpenPlan {
    p.plan(s).expect("plan")
}

#[test]
fn property_plan_derivation_radius_sigma_and_taps() {
    // Spec section 2.2: radius to sigma, and the tap count at the screen cap.
    let table = [
        (0.1f32, 0.097f32, 3usize),
        (0.2, 0.322, 5),
        (0.3, 0.445, 5),
        (0.5, 0.650, 5),
        (0.6, 0.720, 7),
        (0.8, 0.860, 7),
        (1.0, 1.000, 7),
        (1.5, 1.350, 9),
        (2.0, 1.700, 11),
        (2.5, 2.050, 13),
    ];
    for (r, sigma, taps) in table {
        let mut p = OutputSharpen::screen();
        p.radius_px = r;
        let plan = plan_of(&p, 1.0);
        let st = plan.stages()[0];
        assert!((st.sigma - sigma).abs() < 6e-4, "R {r}: sigma {}", st.sigma);
        assert_eq!(plan.kernel(0).len(), taps, "R {r}: taps");
    }
    // The step of the radius map at R = 0.5: just below it sigma is 0.621.
    let mut p = OutputSharpen::screen();
    p.radius_px = 0.4999;
    assert!((plan_of(&p, 1.0).stages()[0].sigma - 0.621).abs() < 2e-3);
    // Tap count: 4.1 gives 13 (screen cap) and 23 (print cap) taps.
    assert_eq!(gaussian_kernel(4.1, 13).len(), 13);
    assert_eq!(gaussian_kernel(4.1, 33).len(), 23);
    assert_eq!(gaussian_kernel(0.01, 13).len(), 3);
}

#[test]
fn property_plan_derivation_kernels() {
    for sigma in [0.2f32, 0.5, 0.86, 1.0, 1.3, 2.05, 4.1, 5.9] {
        for cap in [13usize, 33] {
            let k = gaussian_kernel(sigma, cap);
            let sum: f64 = k.iter().sum();
            assert!((sum - 1.0).abs() < 1e-15, "sigma {sigma}: sum {sum}");
            assert_eq!(k.len() % 2, 1);
            for i in 0..k.len() {
                assert!((k[i] - k[k.len() - 1 - i]).abs() < 1e-15);
            }
        }
    }
    // The example of the spec: sigma 1.
    let k = gaussian_kernel(1.0, 13);
    let expected = [
        0.005979818,
        0.060625743,
        0.241842857,
        0.383103164,
        0.241842857,
        0.060625743,
        0.005979818,
    ];
    assert_eq!(k.len(), 7);
    for (a, b) in k.iter().zip(expected) {
        assert!((a - b).abs() < 1e-8);
    }
}

#[test]
fn property_kernel_caps_outside_the_spec_are_rounded_not_fatal() {
    // `gaussian_kernel` is public and takes the cap as an argument. The spec's caps are 13 and 33;
    // a cap of 0 used to underflow (`n - 1` on a `usize`), and an even cap gave an even kernel
    // without a centre tap. Any cap now gives an odd kernel of at least 3 taps.
    for (cap, longest) in [
        (0usize, 3usize),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 5),
        (12, 11),
        (13, 13),
        (14, 13),
        (33, 33),
        (34, 33),
    ] {
        for sigma in [0.01f32, 0.5, 1.0, 2.05, 5.9, 100.0] {
            let k = gaussian_kernel(sigma, cap);
            assert_eq!(k.len() % 2, 1, "cap {cap}, sigma {sigma}: odd tap count");
            assert!((3..=longest).contains(&k.len()), "cap {cap}: {}", k.len());
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-15,
                "cap {cap}, sigma {sigma}: sum {sum}"
            );
            for i in 0..k.len() {
                assert!((k[i] - k[k.len() - 1 - i]).abs() < 1e-15);
            }
        }
        // The widest kernel (sigma 100) reaches the cap exactly.
        assert_eq!(gaussian_kernel(100.0, cap).len(), longest, "cap {cap}");
    }
    // The spec's own caps are untouched.
    assert_eq!(gaussian_kernel(100.0, 13).len(), 13);
    assert_eq!(gaussian_kernel(100.0, 33).len(), 33);
}

#[test]
fn property_plan_derivation_halo_mapping_and_dead_zone() {
    let table = [
        (10.0f32, 0.316f32),
        (20.0, 0.447),
        (40.0, 0.632),
        (60.0, 0.775),
        (80.0, 0.894),
        (100.0, 1.0),
    ];
    for (pct, h) in table {
        let mut p = OutputSharpen::screen();
        p.halo_pct = pct;
        let st = plan_of(&p, 1.0).stages()[0];
        assert!((st.halo - h).abs() < 6e-4, "halo {pct}: {}", st.halo);
    }
    let mut p = OutputSharpen::screen();
    p.halo_pct = 0.00005;
    let plan = plan_of(&p, 1.0);
    assert_eq!(plan.stages()[0].halo, 0.0);
    assert!(!plan.halo_enabled());
    p.halo_pct = 0.0;
    assert!(!plan_of(&p, 1.0).halo_enabled());
    p.halo_pct = 60.0;
    assert!(plan_of(&p, 1.0).halo_enabled());
}

#[test]
fn property_plan_derivation_auto_method() {
    // An extra stage alone (no output stage, so no boost): R 0.8 (sigma 0.86) switches from
    // method 0 to method 1 at an amount of 120 %.
    let extra = |amount: f32| ExtraStage {
        method: Method::HardThreshold,
        auto_method: true,
        amount_pct: amount,
        radius_px: 0.8,
        threshold_levels: 1.0,
        halo_pct: 0.0,
    };
    for (amount, method) in [
        (110.0f32, Method::HardThreshold),
        (119.0, Method::HardThreshold),
        (121.0, Method::SmoothGain),
        (150.0, Method::SmoothGain),
    ] {
        let mut p = OutputSharpen::screen().with_amount_pct(0.0);
        p.extra_stages.push(extra(amount));
        let plan = plan_of(&p, 1.0);
        assert_eq!(plan.stages().len(), 1);
        assert_eq!(plan.stages()[0].method, method, "amount {amount}");
        assert!(!plan.stages()[0].output);
    }
    // The rule applies to method 0 only: an explicit method is kept.
    let mut e = extra(110.0);
    e.method = Method::GradientRamp;
    let mut p = OutputSharpen::screen().with_amount_pct(0.0);
    p.extra_stages.push(e);
    assert_eq!(plan_of(&p, 1.0).stages()[0].method, Method::GradientRamp);
}

#[test]
fn property_plan_derivation_threshold_rescaling_and_halo_kernel() {
    // The table of spec section 2.4: R 1.0, threshold 1 level.
    let table = [
        // s, sigma, taps, T in levels, sigma_h, halo kernel taps
        (1.0f32, 1.000f32, 7usize, 1.000f64, 2.0f32, 13usize),
        (0.75, 0.788, 7, 0.908, 1.5, 11),
        (0.5, 0.575, 5, 0.728, 1.0, 7),
        (0.4, 0.490, 5, 0.610, 0.8, 7),
        (1.0 / 3.0, 0.433, 5, 0.510, 0.667, 5),
        (0.25, 0.363, 5, 0.360, 0.5, 5),
        (0.1, 0.235, 3, 0.077, 0.2, 3),
    ];
    for (s, sigma, taps, levels, sigma_h, halo_taps) in table {
        let plan = plan_of(&OutputSharpen::screen(), s);
        let st = plan.stages()[0];
        assert!((st.sigma - sigma).abs() < 1e-3, "s {s}: sigma {}", st.sigma);
        assert_eq!(plan.kernel(0).len(), taps, "s {s}: taps");
        let got = f64::from(st.threshold) * 255.0;
        assert!((got - levels).abs() < 1e-3, "s {s}: T {got} levels");
        assert!((plan.halo_sigma() - sigma_h).abs() < 1e-3);
        assert_eq!(plan.halo_kernel().len(), halo_taps, "s {s}: halo taps");
    }
    // The formula, independently: T0 (1 - c(sigma)^2) / (1 - c(sigma')^2), c = erf(1 / (2 sqrt 2 x)).
    let c = |x: f64| libm::erf(1.0 / (2.0 * std::f64::consts::SQRT_2 * x));
    let mut p = OutputSharpen::screen();
    p.threshold_levels = 2.0;
    let plan = plan_of(&p, 0.5);
    let sigma = f64::from(plan.stages()[0].sigma);
    let want = 2.0 / 255.0 * (1.0 - c(sigma).powi(2)) / (1.0 - c(1.0).powi(2));
    assert!((f64::from(plan.stages()[0].threshold) - want).abs() < 1e-9);
    // Below the knee (sigma' 0.097 < 0.15) neither sigma nor the threshold is rescaled.
    let mut small = OutputSharpen::screen();
    small.radius_px = 0.1;
    let a = plan_of(&small, 1.0).stages()[0];
    let b = plan_of(&small, 0.3).stages()[0];
    assert_eq!((a.sigma, a.threshold), (b.sigma, b.threshold));
}

#[test]
fn property_plan_derivation_halo_px() {
    // Spec section 1.4 and the reference's self-test.
    let mut p = OutputSharpen::screen();
    p.radius_px = 2.5;
    p.halo_pct = 0.0;
    p.method = Method::HardThreshold;
    assert_eq!(p.halo_px(1.0), Ok(6));
    p.method = Method::SmoothGain;
    assert_eq!(p.halo_px(1.0), Ok(12));
    p.method = Method::GradientRamp;
    assert_eq!(p.halo_px(1.0), Ok(13));
    p.halo_pct = 60.0;
    assert_eq!(p.halo_px(1.0), Ok(21));
    p.method = Method::SmoothGain;
    assert_eq!(p.halo_px(1.0), Ok(20));
    p.method = Method::HardThreshold;
    assert_eq!(p.halo_px(1.0), Ok(14));
    let mut q = OutputSharpen::print();
    q.print_distance = 37500.0;
    assert_eq!(q.halo_px(1.0), Ok(32));
    q.method = Method::GradientRamp;
    assert_eq!(q.halo_px(1.0), Ok(33));
    q.method = Method::SmoothGain;
    q.halo_pct = 10.0;
    assert_eq!(q.halo_px(1.0), Ok(40));
    q.method = Method::GradientRamp;
    assert_eq!(q.halo_px(1.0), Ok(41));
    // The mode bounds are 21 and 41, and no stage at any scale exceeds them.
    assert_eq!(OutputSharpen::screen().max_halo_px(), 21);
    assert_eq!(OutputSharpen::print().max_halo_px(), 41);
    for (p, _) in param_sets() {
        for s in [1.0f32, 0.9, 0.5, 0.34, 0.1, 0.01] {
            assert!(p.halo_px(s).expect("halo") <= p.max_halo_px());
        }
    }
}

#[test]
fn property_plan_derivation_boost_and_sorting() {
    let (e, _) = param_sets().pop().expect("last set");
    let plan = plan_of(&e, 1.0);
    let st = plan.stages();
    assert_eq!(st.len(), 3);
    assert!(st.windows(2).all(|w| w[0].sigma <= w[1].sigma), "sorted");
    let fine = st[0];
    assert!(!fine.output);
    let out_sigma = 1.0f32; // radius 1.0 gives sigma 1.0
    let b = ((out_sigma - fine.sigma) * 1.0) / out_sigma;
    assert!(
        (fine.amount - 1.5 * (1.0 + b)).abs() < 1e-6,
        "boosted amount"
    );
    // The halo-limited part H * A is unchanged by the boost.
    let h20 = (0.2f32).sqrt();
    assert!((fine.amount * fine.halo - 1.5 * h20).abs() < 1e-6);
    assert_eq!(fine.method, Method::HardThreshold, "auto keeps method 0");
    // Exactly one output stage; the coarser extra stage (R 2.0) is not boosted.
    assert_eq!(st.iter().filter(|s| s.output).count(), 1);
    let coarse = st[2];
    assert!(!coarse.output && coarse.amount == 1.5 && coarse.halo == 0.0);
    // Ties keep their creation order (stable sort): two equal extra stages stay in order.
    let mut p = OutputSharpen::screen().with_amount_pct(0.0);
    let mk = |amount: f32| ExtraStage {
        method: Method::HardThreshold,
        auto_method: false,
        amount_pct: amount,
        radius_px: 1.0,
        threshold_levels: 1.0,
        halo_pct: 0.0,
    };
    p.extra_stages = vec![mk(10.0), mk(20.0), mk(30.0)];
    let amounts: Vec<f32> = plan_of(&p, 1.0).stages().iter().map(|s| s.amount).collect();
    assert_eq!(amounts, vec![0.1, 0.2, 0.3]);
}

// ------------------------------------------------------------------------------------------------
// Presets (spec section 6.3)
// ------------------------------------------------------------------------------------------------

#[test]
fn preset_screen_derived_values() {
    let p = OutputSharpen::screen();
    assert_eq!(p.mode, Mode::Screen);
    assert_eq!(p.method, Method::SmoothGain);
    assert_eq!(
        (p.amount_pct, p.radius_px, p.threshold_levels, p.halo_pct),
        (100.0, 1.0, 1.0, 60.0)
    );
    assert_eq!(p.luma_weights, [0.299, 0.587, 0.114]);
    assert!(p.extra_stages.is_empty());
    assert_eq!(OutputSharpen::default(), p);

    // Full size.
    let plan = plan_of(&p, 1.0);
    let st = plan.stages()[0];
    assert_eq!(plan.stages().len(), 1);
    assert!(st.output && st.method == Method::SmoothGain);
    assert_eq!(st.amount, 1.0);
    assert_eq!(st.sigma, 1.0);
    assert_eq!(st.threshold, 1.0f32 / 255.0);
    assert!((st.halo - 0.7745967).abs() < 1e-7);
    assert_eq!(plan.kernel(0).len(), 7);
    assert_eq!(plan.gain_kernel(0).len(), 13); // sigma 2: n0 = 12 | 1 = 13
    assert_eq!(plan.halo_kernel().len(), 13);
    assert_eq!(plan.max_taps(), 13);
    // 3 (high-pass) + 6 (gain map) + 2 (window) + 6 (halo map).
    assert_eq!(plan.halo_px(), 17);

    // A 6000 px frame exported at a 2048 px long edge.
    let s = scale_for_edges(2048, 6000);
    assert_eq!(s, (2048.0f64 / 6000.0) as f32);
    let plan = plan_of(&p, s);
    let st = plan.stages()[0];
    eprintln!(
        "screen at s = {s}: sigma {}, T {} levels, halo_px {}",
        st.sigma,
        f64::from(st.threshold) * 255.0,
        plan.halo_px()
    );
    assert!((st.sigma - 0.440).abs() < 5e-4);
    assert!((f64::from(st.threshold) * 255.0 - 0.523).abs() < 5e-4);
    assert_eq!(plan.kernel(0).len(), 5);
    assert_eq!(plan.gain_kernel(0).len(), 7); // sigma max(0.88, 1) = 1
    assert_eq!(plan.halo_kernel().len(), 5); // sigma 0.683
    assert_eq!(plan.halo_px(), 2 + 3 + 2 + 2);
    assert!(plan.halo_px() <= p.max_halo_px());
}

#[test]
fn preset_print_derived_values() {
    let p = OutputSharpen::print();
    assert_eq!(p.mode, Mode::Print);
    assert_eq!(p.method, Method::SmoothGain);
    assert_eq!(
        (
            p.amount_pct,
            p.print_distance,
            p.threshold_levels,
            p.halo_pct
        ),
        (150.0, 5000.0, 1.0, 0.0)
    );
    let plan = plan_of(&p, 1.0);
    let st = plan.stages()[0];
    assert_eq!(st.amount, 1.5);
    assert!((st.sigma - 1.7).abs() < 1e-6, "R 2 px gives sigma 1.7");
    assert_eq!(st.threshold, 1.0f32 / 255.0);
    assert_eq!(st.halo, 0.0);
    assert!(!plan.halo_enabled());
    assert_eq!(plan.max_taps(), 33);
    assert_eq!(plan.kernel(0).len(), 11);
    assert_eq!(plan.gain_kernel(0).len(), 19);
    assert_eq!(plan.halo_px(), 5 + 9);

    let s = scale_for_edges(2048, 6000);
    let plan = plan_of(&p, s);
    let st = plan.stages()[0];
    // sigma = s (1.7 - 0.15) + 0.15.
    let want = f64::from(s) * 1.55 + 0.15;
    assert!((f64::from(st.sigma) - want).abs() < 1e-6);
    assert!(st.threshold < 1.0 / 255.0 && st.threshold > 0.0);
    assert_eq!(plan.kernel(0).len(), 5);
    assert_eq!(plan.gain_kernel(0).len(), 9);
    assert_eq!(plan.halo_px(), 2 + 4);
}

#[test]
fn preset_scale_for_edges() {
    assert_eq!(scale_for_edges(6000, 6000), 1.0);
    assert_eq!(scale_for_edges(7000, 6000), 1.0, "never upscaled");
    assert_eq!(scale_for_edges(100, 0), 1.0);
    assert_eq!(scale_for_edges(1000, 4000), 0.25);
    let s = scale_for_edges(2048, 6000);
    assert!((f64::from(s) - 0.341333).abs() < 1e-6);
    assert!(OutputSharpen::screen().plan(s).is_ok());
}

#[test]
fn preset_amount_changes_only_the_amount() {
    let a = OutputSharpen::screen().with_amount_pct(35.0);
    let mut b = OutputSharpen::screen();
    b.amount_pct = 35.0;
    assert_eq!(a, b);
    let plan = plan_of(&a, 1.0);
    assert_eq!(plan.stages()[0].amount, 0.35);
    // The strength grows with the amount: a step edge overshoots more.
    let mut step = RgbImage::new(24, 6);
    for y in 0..6 {
        for x in 0..24 {
            let v = if x >= 12 { 0.5 } else { 0.1 };
            step.set_pixel(x, y, [v, v, v]);
        }
    }
    let over = |pct: f32| {
        let o = sharpen(&OutputSharpen::screen().with_amount_pct(pct), 1.0, &step);
        (0..24).map(|x| o.pixel(x, 3)[1]).fold(0.0f32, f32::max) - 0.5
    };
    let (o0, o100, o200) = (over(0.0), over(100.0), over(200.0));
    assert_eq!(o0, 0.0);
    assert!(o100 > 0.0 && o200 > o100, "{o0} {o100} {o200}");
}

// ------------------------------------------------------------------------------------------------
// Edge profile and noise (spec sections 2.5 and 6.3)
// ------------------------------------------------------------------------------------------------

/// A soft slanted edge: `lo` to `hi` through a Gaussian of `sigma` px, the edge `angle_deg` from
/// vertical, each pixel the mean of 4 x 4 samples of the analytic profile.
fn slanted_edge(size: usize, angle_deg: f64, sigma: f64, lo: f32, hi: f32) -> RgbImage {
    const SS: usize = 4;
    let a = angle_deg.to_radians();
    let c = size as f64 / 2.0;
    let mut img = RgbImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let px = x as f64 + (sx as f64 + 0.5) / SS as f64;
                    let py = y as f64 + (sy as f64 + 0.5) / SS as f64;
                    let d = (px - c) * a.cos() - (py - c) * a.sin();
                    acc += 0.5 * (1.0 + libm::erf(d / (sigma * std::f64::consts::SQRT_2)));
                }
            }
            let t = acc / (SS * SS) as f64;
            let v = (f64::from(lo) + f64::from(hi - lo) * t) as f32;
            img.set_pixel(x, y, [v, v, v]);
        }
    }
    img
}

/// The edge-spread function of the middle rows of a slanted edge, binned by the distance from the
/// edge line (0.1 px bins, mean per bin): `(distance, value)` pairs.
fn esf(img: &RgbImage, angle_deg: f64) -> Vec<(f64, f64)> {
    let a = angle_deg.to_radians();
    let size = img.width;
    let c = size as f64 / 2.0;
    let (y0, y1) = (size * 5 / 16, size * 11 / 16);
    let mut bins: Vec<(f64, usize)> = vec![(0.0, 0); 160];
    for y in y0..y1 {
        for x in 0..size {
            let d = (x as f64 + 0.5 - c) * a.cos() - (y as f64 + 0.5 - c) * a.sin();
            let b = ((d + 8.0) / 0.1).floor();
            if (0.0..160.0).contains(&b) {
                let v = f64::from(img.pixel(x, y)[1]);
                bins[b as usize].0 += v;
                bins[b as usize].1 += 1;
            }
        }
    }
    bins.iter()
        .enumerate()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(i, (s, n))| (-8.0 + 0.1 * (i as f64 + 0.5), s / *n as f64))
        .collect()
}

/// The 10 % to 90 % rise distance along row `y`, in pixels, with linear interpolation between
/// the samples of the row.
fn row_rise_10_90(img: &RgbImage, y: usize, lo: f64, hi: f64) -> f64 {
    let v: Vec<f64> = (0..img.width)
        .map(|x| f64::from(img.pixel(x, y)[1]))
        .collect();
    let cross = |level: f64| {
        (0..v.len() - 1)
            .find(|&x| (v[x] - level) * (v[x + 1] - level) <= 0.0 && v[x] != v[x + 1])
            .map(|x| x as f64 + (level - v[x]) / (v[x + 1] - v[x]))
            .expect("crossing")
    };
    cross(lo + 0.9 * (hi - lo)) - cross(lo + 0.1 * (hi - lo))
}

/// Overshoot and undershoot of the profile in percent of the step.
fn overshoot_undershoot(profile: &[(f64, f64)], lo: f64, hi: f64) -> (f64, f64) {
    let max = profile.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    let min = profile.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    (
        100.0 * (max - hi).max(0.0) / (hi - lo),
        100.0 * (lo - min).max(0.0) / (hi - lo),
    )
}

#[test]
fn edge_profile_screen_preset() {
    // Spec section 2.5: a soft slanted edge (Gaussian sigma 0.8 px, 0.1 to 0.5, 5 degrees), the
    // 10-90 % rise along the middle row, over/undershoot from the binned edge-spread function.
    let (lo, hi) = (0.1f32, 0.5f32);
    let (flo, fhi) = (f64::from(lo), f64::from(hi));
    let edge = slanted_edge(128, 5.0, 0.8, lo, hi);
    let rise0 = row_rise_10_90(&edge, 64, flo, fhi);
    eprintln!("soft edge before sharpening: rise {rise0:.2} px (spec 2.51)");
    assert!((rise0 - 2.51).abs() < 0.1, "test edge rise {rise0}");

    let mut no_halo = OutputSharpen::screen();
    no_halo.halo_pct = 0.0;
    // label, parameters, scale, spec rise, spec overshoot (percent)
    let cases = [
        (
            "screen, full size",
            OutputSharpen::screen(),
            1.0f32,
            1.94,
            3.0,
        ),
        ("screen, s = 0.4", OutputSharpen::screen(), 0.4, 2.30, 0.3),
        (
            "screen, halo 0, full size",
            no_halo,
            1.0,
            f64::NAN,
            f64::NAN,
        ),
        ("print, full size", OutputSharpen::print(), 1.0, 1.13, 20.7),
    ];
    let mut measured = Vec::new();
    for (label, p, s, spec_rise, spec_over) in cases {
        let out = sharpen(&p, s, &edge);
        let rise = row_rise_10_90(&out, 64, flo, fhi);
        let (over, under) = overshoot_undershoot(&esf(&out, 5.0), flo, fhi);
        eprintln!(
            "{label:<28} rise {rise:.2} px (spec {spec_rise:.2}), overshoot {over:.1} % / undershoot {under:.1} % (spec {spec_over:.1})"
        );
        measured.push((rise, over, under));
    }
    // Rise: within 0.1 px of the spec's figures (the edge phase moves it by a few hundredths).
    assert!(
        (measured[0].0 - 1.94).abs() < 0.1,
        "screen rise {}",
        measured[0].0
    );
    assert!(
        (measured[1].0 - 2.30).abs() < 0.1,
        "s = 0.4 rise {}",
        measured[1].0
    );
    assert!(
        (measured[3].0 - 1.13).abs() < 0.1,
        "print rise {}",
        measured[3].0
    );
    // Overshoot at full size is at most 3.5 %, both sides alike; halo 0 overshoots more; the
    // downscaled output hardly overshoots (spec: 0.3 %); print overshoots about 20 %.
    assert!(
        measured[0].1 <= 3.5 && measured[0].2 <= 3.5,
        "{:?}",
        measured[0]
    );
    assert!(measured[2].1 > measured[0].1 + 1.0);
    assert!(
        measured[1].1 <= 0.6 && measured[1].2 <= 0.6,
        "{:?}",
        measured[1]
    );
    assert!((measured[3].1 - 20.7).abs() < 1.5 && (measured[3].2 - 20.5).abs() < 1.5);
}

/// Standard deviation of one channel over the middle of an image.
fn std_dev_centre(img: &RgbImage, margin: usize) -> f64 {
    let mut vals = Vec::new();
    for y in margin..img.height - margin {
        for x in margin..img.width - margin {
            vals.push(f64::from(img.pixel(x, y)[1]));
        }
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

#[test]
fn edge_profile_noise_gain_on_a_flat_patch() {
    let mut patch = flat(96, 96, 0.18, 0.18, 0.18);
    add_noise(&mut patch, 0.004, 1234);
    let before = std_dev_centre(&patch, 24);
    let mut gain = Vec::new();
    for threshold in [1.0f32, 0.0] {
        let mut p = OutputSharpen::screen();
        p.threshold_levels = threshold;
        let after = std_dev_centre(&sharpen(&p, 1.0, &patch), 24);
        gain.push(after / before);
        eprintln!(
            "noise sigma {before:.5} -> {after:.5} (x{:.3}) with threshold {threshold} level(s)",
            after / before
        );
    }
    // Spec: x1.03 with the 1-level threshold (limit 1.05), x1.38 without.
    assert!(gain[0] <= 1.05, "noise gain {}", gain[0]);
    assert!(gain[1] > gain[0] && gain[1] > 1.2);
}

// ------------------------------------------------------------------------------------------------
// Oracle digests of frames taller than one band
// ------------------------------------------------------------------------------------------------

/// FNV-1a (64 bit) over the little-endian bytes of the samples.
fn fnv1a(samples: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in samples {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The vectors are at most 48 x 48, so they never reach the 128-row bands of the parallel run.
/// These frames do: 173 x 300 is three bands (the last one 44 rows), 1 x 300 and 5 x 259 are
/// narrow ones with the widest receptive fields. The digests are those of the C++ reference's
/// output on exactly these inputs, taken when the port's output was verified bit-identical to it
/// (the reviewer's differential run). They are not the port's own output, and nothing from
/// the reference is stored here but these numbers. A change of any bit of any sample fails. The
/// inputs use no `ln` or `cos` (see [`busy_portable`]), so a platform's libm cannot move a bit.
#[test]
fn oracle_digests_of_frames_taller_than_one_band() {
    let big = busy_portable(173, 300, 21);
    let mut cases: Vec<(OutputSharpen, f32, RgbImage, u64, usize)> = param_sets()
        .into_iter()
        .zip([
            (0x30f4_ead3_e501_32f7u64, 17usize),
            (0x2dc6_240e_e16a_94df, 6),
            (0x4c1b_5b2f_d484_bd31, 21),
            (0x19ed_f5d4_175c_bb4f, 13),
            (0x7ae1_1d18_1b7c_bbce, 14),
            (0x8f00_2115_21cb_d692, 41),
            (0x6ee4_2e80_1ea3_a438, 20),
        ])
        .map(|((p, s), (digest, halo))| (p, s, big.clone(), digest, halo))
        .collect();
    // The widest receptive field of the print mode on a one-pixel-wide frame.
    let mut wide = OutputSharpen::print();
    wide.method = Method::GradientRamp;
    wide.halo_pct = 10.0;
    wide.print_distance = 37500.0;
    cases.push((
        wide,
        1.0,
        random_image(1, 300, -0.05, 3.0, 5),
        0x5b61_fb7e_ccc2_ef3f,
        41,
    ));
    // Three stages on a narrow frame of three bands at a reduced scale.
    let (three, _) = param_sets().pop().expect("last set");
    cases.push((
        three,
        0.6,
        random_image(5, 259, 0.0, 1.5, 6),
        0x510e_59a7_8763_381e,
        16,
    ));
    for (i, (p, s, img, digest, halo)) in cases.iter_mut().enumerate() {
        // The reference compares exactly; the tie tolerance is ours (spec 7.1 item 5).
        p.halo_tie_tolerance = 0.0;
        assert_eq!(p.halo_px(*s), Ok(*halo), "case {i}: halo_px");
        let out = sharpen(p, *s, img);
        assert_eq!(
            fnv1a(&out.data),
            *digest,
            "case {i} ({} x {}): the output differs from the reference's",
            img.width,
            img.height
        );
    }
}

// ------------------------------------------------------------------------------------------------
// Determinism
// ------------------------------------------------------------------------------------------------

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(f)
}

#[test]
fn determinism_one_thread_and_four_threads_give_the_same_bytes() {
    // 300 rows: three bands of 128 rows, so the parallel run really splits the frame.
    let img = busy(173, 300, 21);
    let inter = interleave(&img);
    for (i, (p, s)) in param_sets().into_iter().enumerate() {
        let one = in_pool(1, || {
            p.apply_interleaved(img.width, img.height, &inter, s)
                .expect("one thread")
        });
        let four = in_pool(4, || {
            p.apply_interleaved(img.width, img.height, &inter, s)
                .expect("four threads")
        });
        let planar = in_pool(3, || sharpen(&p, s, &img));
        assert!(
            one.iter()
                .zip(&four)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "param set {i}: 1 and 4 threads differ"
        );
        assert!(
            one.iter()
                .zip(&interleave(&planar))
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "param set {i}: planar differs from interleaved"
        );
        assert!(
            one.iter().zip(&inter).any(|(a, b)| a != b),
            "param set {i} should change the image"
        );
    }
}
