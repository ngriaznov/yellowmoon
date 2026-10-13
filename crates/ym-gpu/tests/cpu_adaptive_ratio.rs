//! The adaptive ratio demosaic (`Method::AdaptiveRatio`, RA-03,
//! `docs/algorithms/01-adaptive-ratio-demosaic.md`).
//!
//! * `vectors_match_within_their_tolerance`: every vector of
//!   `tests/cpu_data/adaptive_ratio_demosaic/` (made by the C++ oracle, never hand-edited) through
//!   the whole-piece routine and through the banded one, within the vector's tolerance (1e-5), and
//!   `derived_values_match_the_vectors` (1e-9 relative). The CA vector `ca_weighting` is the
//!   CA path's (RA-03b) and is checked in `cpu_adaptive_ratio_ca.rs`.
//! * The property tests of spec section 6.2: flat field, balanced entry, CFA-phase equivariance,
//!   tiling parity, receptive field, scale behaviour, robustness, negative residual, determinism
//!   across thread counts.
//! * Plumbing: the banded runner equals the whole-piece routine bit for bit, `demosaic_with`
//!   dispatches to it, invalid input falls back to Menon.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Derived, Error, Params, ShadowMatch};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::synth::{self, Rng};
use ym_gpu::cpu::tiling::{Halo, Rect, tile_grid};
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use ym_noise::model::NoiseModel;

const VECTOR_DIR: &str = "tests/cpu_data/adaptive_ratio_demosaic";

fn vector_files() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(VECTOR_DIR);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
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

/// One vector of the package, decoded.
struct Case {
    name: String,
    json: Value,
    mosaic: Mosaic,
    gains: [f32; 3],
    noise: NoiseDn,
    params: Params,
    expected: Vec<f32>,
    tolerance: f64,
}

impl Case {
    fn inputs(&self) -> DemosaicInputs<'static> {
        DemosaicInputs {
            gains: self.gains,
            noise: self.noise,
            ca: None,
        }
    }

    fn has_ca(&self) -> bool {
        !self.json["params"]["ca"].is_null()
    }
}

fn cases() -> Vec<Case> {
    vector_files()
        .into_iter()
        .map(|path| {
            let json: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("vector file"))
                    .expect("vector json");
            let params = &json["params"];
            let input = &json["input"];
            let (w, h) = (
                input["width"].as_u64().expect("width") as usize,
                input["height"].as_u64().expect("height") as usize,
            );
            let cfa = Cfa::from_name(input["cfa"].as_str().expect("cfa")).expect("known pattern");
            let mosaic = Mosaic::from_data(w, h, cfa, floats(&input["data"])).expect("mosaic");
            let g = params["gains"].as_array().expect("gains");
            let gains = [0, 1, 2].map(|i| g[i].as_f64().expect("gain") as f32);
            let n = &params["noise"];
            let model = NoiseModel {
                a: num(n, "a"),
                b: num(n, "b"),
                delta: [0.0; 4],
                sigma_row: num(n, "sigma_row"),
                q: num(n, "q"),
            };
            let shadow_match = match params["shadow_match"].as_str().expect("shadow_match") {
                "auto" => ShadowMatch::Auto,
                "on" => ShadowMatch::On,
                "off" => ShadowMatch::Off,
                other => panic!("unknown shadow_match {other}"),
            };
            Case {
                name: path
                    .file_stem()
                    .expect("stem")
                    .to_string_lossy()
                    .into_owned(),
                mosaic,
                gains,
                noise: NoiseDn::from_model(&model),
                params: Params {
                    detail: num(params, "detail") as f32,
                    strength: num(params, "strength") as f32,
                    shadow_match,
                    shadow_sharpening: params["shadow_sharpening"].as_bool().expect("bool"),
                },
                expected: floats(&json["expected"]["data"]),
                tolerance: num(&json, "tolerance"),
                json,
            }
        })
        .collect()
}

/// The largest of `values` (0 for none), or NaN when any of them is NaN. `f32::max` and
/// `f64::max` return the other operand for a NaN, so a plain fold over them reads a NaN output as a
/// difference of 0. The demosaic fills the planes it has not written with NaN in debug builds so
/// that a read outside a stage's region shows up in the output: the comparisons below must see it.
fn max_strict(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, |acc, v| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            acc.max(v)
        }
    })
}

#[test]
fn the_comparison_helpers_see_a_nan() {
    // The helpers every comparison below goes through must not read a NaN as "no difference".
    assert!(max_strict([0.0, f64::NAN, 1.0]).is_nan());
    assert!(max_strict([f64::NAN]).is_nan());
    assert_eq!(max_strict([0.5, 2.0, 1.0]), 2.0);
    let a = RgbImage::new(4, 4);
    let mut b = a.clone();
    assert_eq!(image_diff(&a, &b), 0.0);
    b.data[7] = f32::NAN;
    assert!(image_diff(&a, &b).is_nan());
    assert!(max_abs_diff(&b, &a.data).is_nan());
}

fn max_abs_diff(got: &RgbImage, want: &[f32]) -> f64 {
    assert_eq!(got.data.len(), want.len(), "planar sample count");
    max_strict(
        got.data
            .iter()
            .zip(want)
            .map(|(a, b)| f64::from(*a - *b).abs()),
    )
}

/// Largest absolute difference between two images of the same size; NaN when either holds a NaN
/// (unlike [`RgbImage::max_abs_diff`], which ignores NaN).
fn image_diff(a: &RgbImage, b: &RgbImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "dimensions");
    max_strict(
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| f64::from(*x - *y).abs()),
    )
}

#[test]
fn vectors_match_within_their_tolerance() {
    let mut checked = 0;
    for case in cases() {
        if case.has_ca() {
            continue; // ca_weighting: cpu_adaptive_ratio_ca.rs
        }
        let whole = adaptive_ratio::demosaic_whole(&case.mosaic, &case.inputs(), &case.params)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let banded = adaptive_ratio::demosaic(&case.mosaic, &case.inputs(), &case.params)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(whole, banded, "{}: banded differs from whole", case.name);
        let worst = max_abs_diff(&whole, &case.expected);
        println!(
            "{:<26} max |rust - oracle| = {worst:.3e} (tolerance {:.0e})",
            case.name, case.tolerance
        );
        assert!(
            worst <= case.tolerance,
            "{}: {worst} > {}",
            case.name,
            case.tolerance
        );
        checked += 1;
    }
    assert_eq!(checked, 17, "every vector except ca_weighting");
}

#[test]
fn derived_values_match_the_vectors() {
    for case in cases() {
        // The vectors carry f64 gains (1.9 is not an f32); the f64 constructor reproduces the
        // oracle's derivation exactly, the f32 one is what the demosaic runs on.
        let g = case.json["params"]["gains"].as_array().expect("gains");
        let gains64 = [0, 1, 2].map(|i| g[i].as_f64().expect("gain"));
        let d = Derived::with_f64_gains(gains64, case.noise, &case.params).expect("derived");
        let from_f32 = Derived::new(case.gains, case.noise, &case.params).expect("derived");
        let want = &case.json["params"]["derived"];
        let close = |key: &str, got: f64| {
            let w = num(want, key);
            assert!(
                (got - w).abs() <= 1e-9 * w.abs().max(1e-300),
                "{} {key}: {got} vs {w}",
                case.name
            );
        };
        close("rho_r", d.rho_r);
        close("rho_b", d.rho_b);
        close("g_dn", d.g_dn);
        close("r_dn", d.r_dn);
        close("K", d.offset1_dn);
        close("s", d.snr_target);
        close("T", d.gate_dn);
        close("k", d.offset3_dn);
        close("snr18_db", d.snr18_db);
        assert_eq!(
            d.shadow_match,
            want["shadow_match"].as_bool().expect("bool"),
            "{}",
            case.name
        );
        assert_eq!(
            d.halo_px as u64,
            want["halo"].as_u64().expect("halo"),
            "{}",
            case.name
        );
        // With f32 gains only the two rho move (by the f32 rounding of the gains, below 2e-8);
        // everything that comes from the noise model is identical.
        assert!(
            (from_f32.rho_r / d.rho_r - 1.0).abs() < 2e-8,
            "{}",
            case.name
        );
        assert!(
            (from_f32.rho_b / d.rho_b - 1.0).abs() < 2e-8,
            "{}",
            case.name
        );
        assert_eq!(
            (
                from_f32.offset1_dn,
                from_f32.gate_dn,
                from_f32.shadow_match,
                from_f32.halo_px
            ),
            (d.offset1_dn, d.gate_dn, d.shadow_match, d.halo_px),
            "{}",
            case.name
        );
    }
}

#[test]
fn the_phase_vectors_agree_after_the_shift() {
    // One RGGB frame sampled at offsets (0/1, 0/1): pixels at least 11 px from every edge agree
    // (the spec checked 3675 samples, max difference 0; the oracle ran f64 gains, we run f32
    // ones, so the bound here is the vectors' own tolerance).
    let by_name: Vec<Case> = cases()
        .into_iter()
        .filter(|c| c.name.starts_with("phase_"))
        .collect();
    assert_eq!(by_name.len(), 4);
    let offset = |name: &str| match name {
        "phase_rggb" => (0usize, 0usize),
        "phase_grbg" => (1, 0),
        "phase_gbrg" => (0, 1),
        "phase_bggr" => (1, 1),
        other => panic!("{other}"),
    };
    let outputs: Vec<(String, RgbImage)> = by_name
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                adaptive_ratio::demosaic_whole(&c.mosaic, &c.inputs(), &c.params).expect("run"),
            )
        })
        .collect();
    let (n, margin) = (40usize, 11usize);
    let mut compared = 0;
    for (a, ima) in &outputs {
        for (b, imb) in &outputs {
            let ((ax, ay), (bx, by)) = (offset(a), offset(b));
            // Frame pixel (fx, fy) is buffer pixel (fx - ox, fy - oy) of each vector.
            for fy in margin + 1..n - margin {
                for fx in margin + 1..n - margin {
                    for c in 0..3 {
                        let pa = ima.plane(c)[(fy - ay) * n + fx - ax];
                        let pb = imb.plane(c)[(fy - by) * n + fx - bx];
                        assert!(
                            (pa - pb).abs() <= 1e-5,
                            "{a} vs {b} at frame ({fx},{fy}) channel {c}: {pa} vs {pb}"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert_eq!(compared, 4 * 4 * 3 * 17 * 17);
}

// ---------------------------------------------------------------------------------------------
// Scenes and models for the property tests
// ---------------------------------------------------------------------------------------------

fn nz(a: f64, b: f64, sigma_row: f64, q: f64) -> NoiseDn {
    NoiseDn::from_model(&NoiseModel {
        a,
        b,
        delta: [0.0; 4],
        sigma_row,
        q,
    })
}

fn low_noise() -> NoiseDn {
    nz(2e-5, 1e-8, 0.0, 1.0 / 65535.0)
}

fn high_noise() -> NoiseDn {
    nz(5e-4, 4e-5, 1e-3, 0.0)
}

fn inputs(gains: [f32; 3], noise: NoiseDn) -> DemosaicInputs<'static> {
    DemosaicInputs {
        gains,
        noise,
        ca: None,
    }
}

/// A textured, correlated scene sampled through `cfa`, scaled by `level`, with optional noise.
fn scene(w: usize, h: usize, cfa: Cfa, seed: u64, level: f32, noise_sigma: f32) -> Mosaic {
    let mut m = synth::mosaic_from_rgb(&synth::natural(w, h, seed), cfa);
    for v in &mut m.data {
        *v *= level;
    }
    if noise_sigma > 0.0 {
        synth::add_noise(&mut m.data, noise_sigma, seed + 100);
        for v in &mut m.data {
            *v = v.max(-0.05);
        }
    }
    m
}

fn bits(img: &RgbImage) -> Vec<u32> {
    img.data.iter().map(|v| v.to_bits()).collect()
}

// ---------------------------------------------------------------------------------------------
// Property tests of spec 6.2
// ---------------------------------------------------------------------------------------------

#[test]
fn a_flat_field_comes_back_as_gain_times_value() {
    let (w, h) = (23, 19);
    let gain_sets = [[1.0f32, 1.0, 1.0], [2.13, 1.0, 1.61], [0.8, 1.0, 3.4]];
    let colours = [
        [0.21f32, 0.37, 0.16],
        [0.004, 0.006, 0.003],
        [1.7, 2.4, 1.1],
    ];
    let mut worst = 0.0f64;
    let mut runs = 0;
    for cfa in Cfa::ALL {
        for gains in gain_sets {
            for col in colours {
                for shadow_match in [ShadowMatch::Off, ShadowMatch::On] {
                    let mut m = Mosaic::new(w, h, cfa);
                    for y in 0..h {
                        for x in 0..w {
                            m.data[y * w + x] = col[cfa.color_at(x, y).plane()];
                        }
                    }
                    let params = Params {
                        shadow_match,
                        ..Params::default()
                    };
                    let out = adaptive_ratio::demosaic(&m, &inputs(gains, low_noise()), &params)
                        .expect("flat field");
                    for c in 0..3 {
                        let want = f64::from(col[c]) * f64::from(gains[c]);
                        for &v in out.plane(c) {
                            worst = max_strict([worst, (f64::from(v) - want).abs() / want]);
                        }
                    }
                    runs += 1;
                }
            }
        }
    }
    println!("flat field: {runs} runs, max relative error {worst:.3e}");
    assert!(worst <= 1.2e-7, "{worst}");
}

#[test]
fn a_balanced_grey_comes_back_as_the_same_grey() {
    let g0 = [2.07f32, 1.0, 1.53];
    let m = Mosaic::from_data(20, 16, Cfa::GBRG, vec![0.4; 20 * 16]).expect("mosaic");
    let out = adaptive_ratio::demosaic_balanced(&m, &inputs(g0, low_noise()), &Params::default())
        .expect("balanced");
    let worst = max_strict(out.data.iter().map(|v| f64::from((v - 0.4).abs())));
    println!("balanced grey: max |out - 0.4| = {worst:.3e}");
    assert!(worst < 1e-7, "{worst}");
}

#[test]
fn balanced_and_unbalanced_entry_agree() {
    let g0 = [2.07f32, 1.0, 1.53];
    let m = scene(40, 40, Cfa::RGGB, 5, 0.3, 0.0);
    let mut balanced = m.clone();
    for y in 0..m.height {
        for x in 0..m.width {
            balanced.data[y * m.width + x] *= g0[m.cfa.color_at(x, y).plane()];
        }
    }
    let inp = inputs(g0, low_noise());
    let a = adaptive_ratio::demosaic(&m, &inp, &Params::default()).expect("unbalanced");
    let b =
        adaptive_ratio::demosaic_balanced(&balanced, &inp, &Params::default()).expect("balanced");
    let d = image_diff(&a, &b);
    println!("demosaic(m) vs demosaic_balanced(m * g0): max |diff| {d:.3e}");
    assert!(d < 1e-5, "{d}");
}

#[test]
fn crops_at_every_phase_give_the_identical_interior() {
    // A frame cropped at offsets (0/1, 0/1) presents all four patterns; the interiors (at least
    // the halo from the crop edges) equal the whole-frame result exactly.
    let (fw, fh) = (64usize, 56usize);
    for (name, noise, scale, sigma) in [
        ("plain", low_noise(), 0.4, 0.001),
        ("shadow matching", high_noise(), 0.05, 0.003),
    ] {
        let frame = scene(fw, fh, Cfa::RGGB, 77, scale, sigma);
        let inp = inputs([1.9, 1.0, 1.4], noise);
        let params = Params::default();
        let whole = adaptive_ratio::demosaic_whole(&frame, &inp, &params).expect("whole");
        let halo = Derived::new(inp.gains, inp.noise, &params)
            .expect("derived")
            .halo_px;
        let mut worst = 0.0f64;
        for oy in 0..2 {
            for ox in 0..2 {
                let crop = frame.extract(Rect::new(ox, oy, fw - 3, fh - 3));
                let part = adaptive_ratio::demosaic_whole(&crop, &inp, &params).expect("crop");
                for c in 0..3 {
                    for y in halo..crop.height - halo {
                        for x in halo..crop.width - halo {
                            let a = part.plane(c)[y * crop.width + x];
                            let b = whole.plane(c)[(y + oy) * fw + x + ox];
                            worst = max_strict([worst, f64::from((a - b).abs())]);
                        }
                    }
                }
            }
        }
        println!("equivariance {name}: crops at (0..1, 0..1), max |diff| {worst:.3e}");
        assert_eq!(worst, 0.0, "{name}");
    }
}

#[test]
fn tiles_with_the_declared_halo_equal_the_whole_frame() {
    // Odd frame sizes and non-RGGB phases; tiles of 24 x 20 with the halo of the configuration
    // and with the declared maximum (13); the limit is 1e-4, the oracle measures 0.
    let (fw, fh) = (77usize, 61usize);
    let frame_rect = Rect::whole(fw, fh);
    for (name, cfa, noise, scale, sigma) in [
        ("plain GBRG", Cfa::GBRG, low_noise(), 0.4, 0.001),
        ("shadow matching RGGB", Cfa::RGGB, high_noise(), 0.05, 0.003),
        ("shadow matching BGGR", Cfa::BGGR, high_noise(), 0.05, 0.003),
        ("plain GRBG", Cfa::GRBG, low_noise(), 0.01, 0.0002),
    ] {
        let frame = scene(fw, fh, cfa, 1234, scale, sigma);
        let inp = inputs([2.2, 1.0, 1.3], noise);
        let params = Params::default();
        let whole = adaptive_ratio::demosaic_whole(&frame, &inp, &params).expect("whole");
        let own = Derived::new(inp.gains, inp.noise, &params)
            .expect("derived")
            .halo_px;
        for halo in [own, Method::AdaptiveRatio.halo()] {
            for tile in [(24, 20), (33, 17)] {
                let mut tiled = RgbImage::new(fw, fh);
                for core in tile_grid(fw, fh, tile.0, tile.1, 2) {
                    let ext = core.grow_aligned(Halo::uniform(halo), frame_rect, 2);
                    let res = adaptive_ratio::demosaic_whole(&frame.extract(ext), &inp, &params)
                        .expect("tile");
                    tiled.paste(&res, core.relative_to(&ext), core.x, core.y);
                }
                let d = image_diff(&whole, &tiled);
                println!("tiling {name}: halo {halo}, tiles {tile:?}: max |diff| {d:.3e}");
                assert!(d <= 1e-4, "{name} halo {halo} {tile:?}: {d}");
                assert_eq!(d, 0.0, "{name} halo {halo} {tile:?}: expected bit-exact");
            }
        }
    }
}

/// The largest Chebyshev distance from a perturbed photosite to a changed output sample, over
/// four photosites (all four sites of a 2 x 2 block) and two perturbations each.
fn measured_reach(base: &Mosaic, inp: &DemosaicInputs<'_>, params: &Params, bump: f32) -> usize {
    let (w, h) = (base.width, base.height);
    let reference = adaptive_ratio::demosaic_whole(base, inp, params).expect("base");
    let mut reach = 0;
    for (px, py) in [
        (w / 2, h / 2),
        (w / 2 + 1, h / 2),
        (w / 2, h / 2 + 1),
        (w / 2 + 1, h / 2 + 1),
    ] {
        for delta in [bump, -0.6 * bump] {
            let mut m = base.clone();
            m.data[py * w + px] += delta;
            let out = adaptive_ratio::demosaic_whole(&m, inp, params).expect("perturbed");
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        if out.plane(c)[y * w + x] != reference.plane(c)[y * w + x] {
                            reach = reach.max(x.abs_diff(px).max(y.abs_diff(py)));
                        }
                    }
                }
            }
        }
    }
    reach
}

#[test]
fn the_receptive_field_stays_within_the_declared_halo() {
    // Plain with the shadow sharpening active (dark texture), plain on a bright scene, and
    // shadow matching. The spec measures 9 px (plain, sharpening active) and 10 px (matching);
    // the per-configuration bounds are 11 and 13, and Method::halo() is the larger.
    let plain_dark = scene(61, 61, Cfa::RGGB, 900, 0.02, 0.0);
    let plain_bright = scene(61, 61, Cfa::RGGB, 901, 0.5, 0.002);
    let matching = scene(61, 61, Cfa::RGGB, 902, 0.05, 0.003);
    for (name, base, noise, params, bump, bound) in [
        (
            "plain, dark",
            &plain_dark,
            low_noise(),
            Params::default(),
            0.002,
            11,
        ),
        (
            "plain, bright",
            &plain_bright,
            low_noise(),
            Params::default(),
            0.05,
            11,
        ),
        (
            "shadow matching",
            &matching,
            high_noise(),
            Params::default(),
            0.02,
            13,
        ),
    ] {
        let inp = inputs([1.8, 1.0, 1.5], noise);
        let reach = measured_reach(base, &inp, &params, bump);
        println!("receptive field {name}: measured {reach} px, bound {bound} px");
        assert!(reach <= bound, "{name}: {reach} > {bound}");
        assert!(
            reach >= 5,
            "{name}: the perturbation test saw only {reach} px"
        );
    }
    assert_eq!(Method::AdaptiveRatio.halo(), 13);
}

#[test]
fn the_method_is_nearly_scale_invariant_when_the_noise_scales_with_it() {
    let m = scene(64, 64, Cfa::RGGB, 5, 0.5, 0.0);
    let inp = inputs([1.0; 3], low_noise());
    let base = adaptive_ratio::demosaic(&m, &inp, &Params::default()).expect("base");
    for lambda in [0.25f64, 4.0] {
        let mut ms = m.clone();
        for v in &mut ms.data {
            *v = (f64::from(*v) * lambda) as f32;
        }
        // a * lambda, b * lambda^2, q * lambda (sigma_row would scale like lambda as well).
        let scaled_noise = nz(2e-5 * lambda, 1e-8 * lambda * lambda, 0.0, lambda / 65535.0);
        let out =
            adaptive_ratio::demosaic(&ms, &inputs([1.0; 3], scaled_noise), &Params::default())
                .expect("scaled");
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for c in 0..3 {
            for y in 12..52 {
                for x in 12..52 {
                    let j = y * 64 + x;
                    let want = lambda * f64::from(base.plane(c)[j]);
                    let e = f64::from(out.plane(c)[j]) - want;
                    num += e * e;
                    den += want * want;
                }
            }
        }
        let rel = (num / den).sqrt();
        println!("scale {lambda}: relative RMS change {rel:.3e}");
        assert!(rel < 0.02, "scale {lambda}: {rel}");
    }
}

#[test]
fn random_input_gives_finite_output_for_every_size_phase_and_model() {
    let mut rng = Rng::new(42);
    let sizes = [(2, 2), (2, 5), (3, 3), (5, 2), (7, 5), (13, 11), (30, 17)];
    let mut runs = 0;
    for (w, h) in sizes {
        for cfa in Cfa::ALL {
            for cfg in 0..4 {
                let data: Vec<f32> = (0..w * h)
                    .map(|_| {
                        let u = rng.range(0.0, 1.0);
                        if cfg == 3 {
                            0.0
                        } else if u < 0.2 {
                            -0.05 * rng.range(0.0, 1.0)
                        } else if u < 0.3 {
                            4.0 * rng.range(0.0, 1.0)
                        } else {
                            rng.range(0.0, 1.0)
                        }
                    })
                    .collect();
                let m = Mosaic::from_data(w, h, cfa, data).expect("mosaic");
                let noise = match cfg {
                    1 => high_noise(),
                    2 => nz(0.2, 0.01, 0.0, 0.0),
                    _ => low_noise(),
                };
                let params = Params {
                    shadow_match: if cfg == 2 {
                        ShadowMatch::On
                    } else {
                        ShadowMatch::Auto
                    },
                    // Detail and strength at 100 floor s at 1.
                    detail: if cfg == 1 { 100.0 } else { 0.0 },
                    strength: if cfg == 1 { 100.0 } else { 0.0 },
                    shadow_sharpening: true,
                };
                let gains = [
                    0.3 + 4.0 * rng.range(0.0, 1.0),
                    1.0,
                    0.3 + 4.0 * rng.range(0.0, 1.0),
                ];
                let out = adaptive_ratio::demosaic(&m, &inputs(gains, noise), &params)
                    .unwrap_or_else(|e| panic!("{w}x{h} {} cfg {cfg}: {e}", cfa.name()));
                assert!(
                    out.data.iter().all(|v| v.is_finite()),
                    "{w}x{h} {} cfg {cfg}: not finite",
                    cfa.name()
                );
                // The whole-piece routine agrees with the banded one on every size.
                let whole = adaptive_ratio::demosaic_whole(&m, &inputs(gains, noise), &params)
                    .expect("whole");
                assert_eq!(bits(&out), bits(&whole), "{w}x{h} {}", cfa.name());
                runs += 1;
            }
        }
    }
    println!(
        "{runs} random runs: sizes 2x2 to 30x17, values in [-0.05, 4], zero and extreme noise"
    );
}

#[test]
fn invalid_input_is_rejected() {
    let good = Mosaic::from_data(2, 2, Cfa::RGGB, vec![0.1; 4]).expect("mosaic");
    let params = Params::default();
    let ok = inputs([1.0; 3], low_noise());
    assert!(adaptive_ratio::demosaic(&good, &ok, &params).is_ok());
    let one = Mosaic::from_data(1, 1, Cfa::RGGB, vec![0.5]).expect("mosaic");
    assert_eq!(
        adaptive_ratio::demosaic(&one, &ok, &params),
        Err(Error::BadSize {
            width: 1,
            height: 1
        })
    );
    let mut nan = good.clone();
    nan.data[2] = f32::NAN;
    assert_eq!(
        adaptive_ratio::demosaic(&nan, &ok, &params),
        Err(Error::BadInput)
    );
    assert!(matches!(
        adaptive_ratio::demosaic(&good, &inputs([1.0, 0.0, 1.0], low_noise()), &params),
        Err(Error::BadGains(_))
    ));
    assert!(matches!(
        adaptive_ratio::demosaic(&good, &inputs([1.0; 3], NoiseDn::new(-1.0, 1.0)), &params),
        Err(Error::BadNoise(_))
    ));
    // The infallible entry point falls back to Menon (the mosaic read as balanced).
    let mosaic = scene(30, 26, Cfa::RGGB, 3, 0.4, 0.0);
    let bad = inputs([1.0, 0.0, 1.0], low_noise());
    assert_eq!(
        demosaic::demosaic_with(&mosaic, Method::AdaptiveRatio, &bad),
        demosaic::demosaic_with(&mosaic, Method::Menon, &bad)
    );
    assert_eq!(
        demosaic::demosaic_whole_with(&mosaic, Method::AdaptiveRatio, &bad),
        demosaic::demosaic_whole_with(&mosaic, Method::Menon, &bad)
    );
}

#[test]
fn a_negative_sample_restores_its_residual_in_its_own_channel() {
    // A B site of RGGB at (3, 3) measured at -0.04 instead of 0: the own channel differs by
    // exactly gain * value, the other channels not at all.
    let g0 = [2.0f32, 1.0, 1.5];
    let mut neg = Mosaic::from_data(8, 8, Cfa::RGGB, vec![0.01; 64]).expect("mosaic");
    neg.data[3 * 8 + 3] = -0.04;
    let mut zeroed = neg.clone();
    zeroed.data[3 * 8 + 3] = 0.0;
    let inp = inputs(g0, low_noise());
    let a = adaptive_ratio::demosaic(&neg, &inp, &Params::default()).expect("negative");
    let b = adaptive_ratio::demosaic(&zeroed, &inp, &Params::default()).expect("zeroed");
    let (va, vb) = (a.plane(2)[3 * 8 + 3], b.plane(2)[3 * 8 + 3]);
    assert!(
        (f64::from(va) - (f64::from(vb) - 1.5 * 0.04)).abs() < 1e-7,
        "{va} vs {vb}"
    );
    for c in [0usize, 1] {
        assert_eq!(a.plane(c), b.plane(c), "channel {c}");
    }
    // The other pixels of the blue channel are untouched too.
    for (i, (x, y)) in a.plane(2).iter().zip(b.plane(2)).enumerate() {
        if i != 3 * 8 + 3 {
            assert_eq!(x, y, "blue at {i}");
        }
    }
}

#[test]
fn the_output_is_identical_for_every_thread_count() {
    // `--jobs 1` against `--jobs 4`: byte-identical. Tall enough for several bands, both
    // configurations, two phases.
    let run = |threads: usize, m: &Mosaic, inp: &DemosaicInputs<'_>| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| adaptive_ratio::demosaic(m, inp, &Params::default()).expect("run"))
    };
    for (cfa, noise, scale, sigma) in [
        (Cfa::RGGB, low_noise(), 0.4, 0.001),
        (Cfa::GBRG, high_noise(), 0.05, 0.003),
    ] {
        let m = scene(83, 421, cfa, 31, scale, sigma);
        let inp = inputs([2.1, 1.0, 1.6], noise);
        let one = run(1, &m, &inp);
        assert!(one.data.iter().all(|v| v.is_finite()), "{}", cfa.name());
        let four = run(4, &m, &inp);
        let eight = run(8, &m, &inp);
        assert_eq!(bits(&one), bits(&four), "{}", cfa.name());
        assert_eq!(bits(&one), bits(&eight), "{}", cfa.name());
        // And equal to the whole-piece routine.
        let whole = adaptive_ratio::demosaic_whole(&m, &inp, &Params::default()).expect("whole");
        assert_eq!(bits(&one), bits(&whole), "{}", cfa.name());
    }
}

// ---------------------------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------------------------

#[test]
fn demosaic_with_runs_the_adaptive_ratio_demosaic_on_the_balanced_mosaic() {
    let m = scene(70, 300, Cfa::GRBG, 8, 0.4, 0.001);
    let rich = inputs([2.0, 1.0, 1.6], high_noise());
    // `demosaic_with` takes the balanced mosaic (the chain's R4/R5 output, like every method) and
    // undoes R4 itself: it is `demosaic_balanced`, banded, and `demosaic_whole_with` is the
    // whole-piece one, bit for bit (INTEGRATION.md section 4).
    let direct = adaptive_ratio::demosaic_balanced(&m, &rich, &Params::default()).expect("direct");
    assert_eq!(
        demosaic::demosaic_with(&m, Method::AdaptiveRatio, &rich),
        direct
    );
    let whole =
        adaptive_ratio::demosaic_balanced_whole(&m, &rich, &Params::default()).expect("whole");
    assert_eq!(
        demosaic::demosaic_whole_with(&m, Method::AdaptiveRatio, &rich),
        whole
    );
    assert_eq!(direct, whole, "banded equals whole");
    // `demosaic(m, method)` is unit gains and zero noise; with unit gains the division is the
    // identity, so the unbalanced entry point gives the same bits.
    let unit = DemosaicInputs::default();
    assert_eq!(
        demosaic::demosaic(&m, Method::AdaptiveRatio),
        demosaic::demosaic_with(&m, Method::AdaptiveRatio, &unit)
    );
    assert_eq!(
        demosaic::demosaic(&m, Method::AdaptiveRatio),
        adaptive_ratio::demosaic(&m, &unit, &Params::default()).expect("unbalanced")
    );
    // The gains and the noise model are read (unlike the three older methods).
    assert_ne!(
        demosaic::demosaic(&m, Method::AdaptiveRatio),
        demosaic::demosaic_with(&m, Method::AdaptiveRatio, &rich)
    );
    // The output keeps the placement of a sub-buffer.
    let part = m.extract(Rect::new(6, 10, 51, 200));
    let out = demosaic::demosaic_with(&part, Method::AdaptiveRatio, &rich);
    assert_eq!(out.place, part.place);
}

/// The chain's R6 input is the R4 output, `g0 * m`: through `demosaic_with` it gives what the
/// unbalanced entry point gives for `m` itself, up to the `f32` rounding of `g0 * m` and of the
/// division back (the oracle measures 3e-8 for `demosaic_balanced`; a near-tie of a decision may
/// move a pixel a little more, so the bound here is 1e-5 of white).
#[test]
fn demosaic_with_agrees_with_the_unbalanced_entry_point_on_balanced_input() {
    for cfa in Cfa::ALL {
        let m = scene(61, 53, cfa, 21, 0.35, 0.002);
        for (g0, noise) in [
            ([1.0f32, 1.0, 1.0], low_noise()),
            ([2.0, 1.0, 1.6], low_noise()),
            ([1.9, 1.0, 1.4], high_noise()),
            ([0.8, 1.0, 3.1], high_noise()),
        ] {
            let inp = inputs(g0, noise);
            let mut balanced = m.clone();
            for (i, v) in balanced.data.iter_mut().enumerate() {
                *v *= g0[cfa.color_at(i % 61, i / 61).plane()];
            }
            let via_chain = demosaic::demosaic_with(&balanced, Method::AdaptiveRatio, &inp);
            let reference = adaptive_ratio::demosaic(&m, &inp, &Params::default()).expect("run");
            let d = via_chain.max_abs_diff(&reference);
            println!(
                "{} {g0:?}: |balanced entry - unbalanced entry| = {d:.3e}",
                cfa.name()
            );
            assert!(d < 1e-5, "{} {g0:?}: {d}", cfa.name());
        }
    }
}

#[test]
fn banded_equals_whole_for_every_phase_and_configuration() {
    // 70 x 300: three bands (128, 128, 44 rows).
    for cfa in Cfa::ALL {
        for (noise, scale, sigma, params) in [
            (low_noise(), 0.4, 0.001, Params::default()),
            (high_noise(), 0.05, 0.003, Params::default()),
            (
                low_noise(),
                0.02,
                0.0,
                Params {
                    shadow_match: ShadowMatch::On,
                    shadow_sharpening: false,
                    ..Params::default()
                },
            ),
        ] {
            let m = scene(70, 300, cfa, 12, scale, sigma);
            let inp = inputs([1.9, 1.0, 1.4], noise);
            let whole = adaptive_ratio::demosaic_whole(&m, &inp, &params).expect("whole");
            let banded = adaptive_ratio::demosaic(&m, &inp, &params).expect("banded");
            // Equal bits would also hold for two identical NaNs.
            assert!(
                whole.data.iter().all(|v| v.is_finite()),
                "{}: not finite",
                cfa.name()
            );
            assert_eq!(bits(&whole), bits(&banded), "{}", cfa.name());
        }
    }
}

#[test]
fn banded_equals_whole_around_the_band_boundaries() {
    // Bands are 128 rows. The heights below put the last band at 127, 128, 1, 2 and 3 rows (and
    // the same one band further down): a short last band has the least real context to read, and
    // its mirror at the frame edge must still give the whole-frame values. Frames 2 and 3
    // photosites wide are narrower than every halo (the mirror repeats). Narrow frames keep the
    // run short; every output must also be finite (a stage that read outside its region would
    // show NaN in a debug build).
    for height in [127usize, 128, 129, 130, 131, 255, 256, 257, 258] {
        for width in [2usize, 3, 21] {
            for cfa in Cfa::ALL {
                for (noise, scale, sigma) in
                    [(low_noise(), 0.4, 0.001), (high_noise(), 0.05, 0.003)]
                {
                    let m = scene(width, height, cfa, 7, scale, sigma);
                    let inp = inputs([2.0, 1.0, 1.5], noise);
                    let params = Params::default();
                    let whole = adaptive_ratio::demosaic_whole(&m, &inp, &params).expect("whole");
                    let banded = adaptive_ratio::demosaic(&m, &inp, &params).expect("banded");
                    let label = format!("{width}x{height} {}", cfa.name());
                    assert!(
                        whole.data.iter().all(|v| v.is_finite()),
                        "{label}: not finite"
                    );
                    assert_eq!(bits(&whole), bits(&banded), "{label}");
                }
            }
        }
    }
}

#[test]
fn a_zero_input_comes_back_as_zeros() {
    for noise in [NoiseDn::ZERO, low_noise(), high_noise()] {
        let m = Mosaic::new(17, 13, Cfa::BGGR);
        let out = adaptive_ratio::demosaic(&m, &inputs([2.0, 1.0, 1.5], noise), &Params::default())
            .expect("zero");
        assert!(out.data.iter().all(|v| *v == 0.0));
    }
}
