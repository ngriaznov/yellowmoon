//! Tiled versus whole-frame results for every CPU reference stage (`docs/01` GPU tiling,
//! `docs/05` Tiling and halos): max abs difference <= 1e-4, exactly 0 for pointwise stages.
//!
//! Each stage is run once on the whole frame and once per tile of a grid that does not divide the
//! frame evenly, with the stage's halo from `ym_gpu::cpu::tiling`, keeping the tile cores only.

use ym_gpu::cpu::demosaic::{self, Method};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::detail::{self, NoiseReduction};
use ym_gpu::cpu::gains;
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::sharpen::{self, SharpenParams};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::{Halo, Rect, Stage, required_input_rect, sharpen_halo, tile_grid};
use ym_gpu::cpu::vignette::{self, VignetteProfile};
use ym_gpu::cpu::{Cfa, ClipMask, Mosaic, NoiseSigma, RgbImage};

const W: usize = 203;
const H: usize = 157;
const TOL: f32 = 1e-4;

fn scene() -> RgbImage {
    synth::natural(W, H, 21)
}

fn rgb_tiled(
    whole_in: &RgbImage,
    halo: usize,
    tile: (usize, usize),
    stage: impl Fn(&RgbImage) -> RgbImage,
) -> RgbImage {
    let mut out = RgbImage::new(W, H);
    let frame = Rect::whole(W, H);
    for core in tile_grid(W, H, tile.0, tile.1, 1) {
        let ext = core.grow(Halo::uniform(halo), frame);
        let res = stage(&whole_in.extract(ext));
        out.paste(&res, core.relative_to(&ext), core.x, core.y);
    }
    out
}

fn mosaic_tiled<T>(
    m: &Mosaic,
    halo: usize,
    tile: (usize, usize),
    stage: impl Fn(&Mosaic) -> T,
    paste: impl Fn(&mut RgbImage, &T, Rect, usize, usize),
) -> RgbImage {
    let mut out = RgbImage::new(W, H);
    let frame = Rect::whole(W, H);
    for core in tile_grid(W, H, tile.0, tile.1, 2) {
        let ext = core.grow_aligned(Halo::uniform(halo), frame, 2);
        let res = stage(&m.extract(ext));
        paste(&mut out, &res, core.relative_to(&ext), core.x, core.y);
    }
    out
}

fn report(name: &str, diff: f32, tol: f32) {
    println!("{name:<34} max |tiled - whole| = {diff:.3e} (limit {tol:.0e})");
    assert!(diff <= tol, "{name}: {diff} > {tol}");
}

#[test]
fn pointwise_mosaic_stages_are_exact() {
    // R1, R3, R4 on a raw plane, whole versus tiles with odd origins.
    let img = scene();
    let mosaic = synth::mosaic_from_rgb(&img, Cfa::RGGB);
    let raw: Vec<u16> = mosaic
        .data
        .iter()
        .map(|v| (512.0 + v * 15000.0) as u16)
        .collect();
    let levels = Levels {
        black: [510.0, 512.0, 514.0, 516.0],
        white: [15900.0, 15950.0, 16000.0, 16050.0],
    };
    let qstep: Vec<f32> = (0..W * H).map(|i| 1.0 + (i % 5) as f32).collect();
    let (mut whole, wmask) =
        normalize::normalize(&raw, W, H, Cfa::RGGB, &levels, Some(&qstep)).unwrap();
    let profile = VignetteProfile::from_tag(&[0, -300, -1200, -3000, -6000]).unwrap();
    vignette::apply_vignette(&mut whole, &profile);
    gains::apply_gains(&mut whole, [2.1, 1.0, 1.7]);

    let mut tiled = Mosaic::new(W, H, Cfa::RGGB);
    let mut tmask = ClipMask::new(W, H);
    for core in tile_grid(W, H, 37, 29, 1) {
        let place = ym_gpu::cpu::Placement::whole(W, H).offset(core.x, core.y);
        let tile_raw: Vec<u16> = (core.y..core.y1())
            .flat_map(|y| raw[y * W + core.x..y * W + core.x1()].to_vec())
            .collect();
        let tile_q: Vec<f32> = (core.y..core.y1())
            .flat_map(|y| qstep[y * W + core.x..y * W + core.x1()].to_vec())
            .collect();
        let (mut m, mk) = normalize::normalize_tile(
            &tile_raw,
            core.w,
            core.h,
            place,
            Cfa::RGGB.shifted(core.x, core.y),
            &levels,
            Some(&tile_q),
        )
        .unwrap();
        vignette::apply_vignette(&mut m, &profile);
        gains::apply_gains(&mut m, [2.1, 1.0, 1.7]);
        tiled.paste(&m, Rect::whole(core.w, core.h), core.x, core.y);
        for y in 0..core.h {
            for x in 0..core.w {
                tmask.data[(core.y + y) * W + core.x + x] = mk.data[y * core.w + x];
            }
        }
    }
    let diff = whole
        .data
        .iter()
        .zip(&tiled.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    report("R1 + R3 + R4 (pointwise)", diff, 0.0);
    assert_eq!(wmask, tmask, "clip mask differs");
}

#[test]
fn demosaic_tiles_match_for_every_method_and_phase() {
    let img = scene();
    for cfa in Cfa::ALL {
        let m = synth::mosaic_from_rgb(&img, cfa);
        for method in Method::ALL {
            let whole = demosaic::demosaic_whole(&m, method);
            for tile in [(64, 48), (33, 21), (130, 100)] {
                let tiled = mosaic_tiled(
                    &m,
                    method.halo(),
                    tile,
                    |t| demosaic::demosaic_whole(t, method),
                    |o, r, from, x, y| o.paste(r, from, x, y),
                );
                report(
                    &format!("R6 {} {} tile {:?}", method.name(), cfa.name(), tile),
                    whole.max_abs_diff(&tiled),
                    TOL,
                );
            }
        }
    }
}

#[test]
fn bilateral_denoise_tiles_match() {
    let mut img = scene();
    synth::add_noise(&mut img.data, 0.01, 4);
    let p = BilateralParams::default();
    let sigma: Vec<f32> = (0..W * H)
        .map(|i| 0.006 + 0.00003 * (i % 101) as f32)
        .collect();
    let whole = denoise::bilateral_whole(&img, &NoiseSigma::Plane(&sigma), 0, &p);
    // Tiles get the matching part of the sigma map.
    let mut tiled = RgbImage::new(W, H);
    let frame = Rect::whole(W, H);
    for core in tile_grid(W, H, 50, 40, 1) {
        let ext = core.grow(Halo::uniform(p.radius_px), frame);
        let s: Vec<f32> = (ext.y..ext.y1())
            .flat_map(|y| sigma[y * W + ext.x..y * W + ext.x1()].to_vec())
            .collect();
        let res = denoise::bilateral_whole(&img.extract(ext), &NoiseSigma::Plane(&s), 0, &p);
        tiled.paste(&res, core.relative_to(&ext), core.x, core.y);
    }
    report("R6 bilateral denoise", whole.max_abs_diff(&tiled), TOL);
}

#[test]
fn detail_stages_tile_exactly() {
    let mut img = scene();
    synth::add_noise(&mut img.data, 0.015, 8);
    // Guided filter for several colour strengths (halo 2 * radius).
    for color in [10.0f32, 50.0, 100.0] {
        let nr = NoiseReduction {
            luminance: 30.0,
            color,
            detail: 50.0,
        };
        let halo = detail::guided_halo_px(nr.chroma_radius_px());
        let whole = detail::chroma_guided_whole(&img, &NoiseSigma::Scalar(0.015), 0, color);
        let tiled = rgb_tiled(&img, halo, (60, 45), |t| {
            detail::chroma_guided_whole(t, &NoiseSigma::Scalar(0.015), 0, color)
        });
        report(
            &format!("R7 guided filter C={color}"),
            whole.max_abs_diff(&tiled),
            TOL,
        );
    }
    // Detail injection is pointwise.
    let classical = synth::natural(W, H, 22);
    let whole = detail::inject_detail(&img, &classical, 0.3).unwrap();
    let mut tiled = RgbImage::new(W, H);
    for core in tile_grid(W, H, 41, 33, 1) {
        let res = detail::inject_detail(&img.extract(core), &classical.extract(core), 0.3).unwrap();
        tiled.paste(&res, Rect::whole(core.w, core.h), core.x, core.y);
    }
    report(
        "R7 detail re-injection (pointwise)",
        whole.max_abs_diff(&tiled),
        0.0,
    );
}

#[test]
fn richardson_lucy_tiles_match_within_the_halo_cap() {
    let img = scene();
    // The Auto settings (n = 4, sigma_max = 1.0) have an exact radius of 24 px, the cap.
    for (amount, radius) in [(120.0f32, 0.8f32), (60.0, 0.8), (30.0, 1.5)] {
        let p = SharpenParams {
            amount,
            radius_px: radius,
        };
        let halo = sharpen_halo(p.iterations(), p.sigma_max_px());
        let exact = 2 * p.iterations() * (3.0 * p.sigma_max_px()).ceil() as usize;
        let whole = sharpen::richardson_lucy_whole(&img, &p, &NoiseSigma::Scalar(0.004), 0);
        let tiled = rgb_tiled(&img, halo, (70, 50), |t| {
            sharpen::richardson_lucy_whole(t, &p, &NoiseSigma::Scalar(0.004), 0)
        });
        let diff = whole.max_abs_diff(&tiled);
        println!("RL A={amount} rho={radius}: halo {halo} (exact radius {exact})");
        report(&format!("R8 RL A={amount} rho={radius}"), diff, TOL);
    }
}

#[test]
fn richardson_lucy_beyond_the_cap_stays_close() {
    // A = 300 (10 iterations) at rho = 1.2: the exact radius (180 px) is far above the 24 px cap.
    // The scheduler's tiles then differ from the whole frame by the truncated far-field only.
    let img = scene();
    let p = SharpenParams {
        amount: 300.0,
        radius_px: 1.2,
    };
    let whole = sharpen::richardson_lucy_whole(&img, &p, &NoiseSigma::Scalar(0.004), 0);
    let tiled = rgb_tiled(&img, p.halo_px(), (70, 50), |t| {
        sharpen::richardson_lucy_whole(t, &p, &NoiseSigma::Scalar(0.004), 0)
    });
    let diff = whole.max_abs_diff(&tiled);
    println!("R8 RL A=300 rho=1.2 beyond the cap: max diff {diff:.3e}");
    assert!(diff < 2e-2, "{diff}");
}

#[test]
fn required_input_rect_covers_every_dependency() {
    // Perturb one raw photosite and check that no stage output changes outside the rectangle the
    // backwards walk reports for that output pixel's neighbourhood: here the walk for a 1x1 output
    // gives the rectangle that must contain every photosite influencing it.
    let img = scene();
    let m = synth::mosaic_from_rgb(&img, Cfa::RGGB);
    let chain = [
        Stage::Normalize,
        Stage::Gains,
        Stage::Highlights,
        Stage::Demosaic(Method::Lmmse),
        Stage::Denoise { radius_px: 2 },
        Stage::Sharpen {
            iterations: 2,
            sigma_max_px: 1.0,
        },
    ];
    let frame = Rect::whole(W, H);
    let p = SharpenParams {
        amount: 60.0,
        radius_px: 0.8,
    };
    let run = |mosaic: &Mosaic| {
        let d = demosaic::demosaic_whole(mosaic, Method::Lmmse);
        let n = denoise::bilateral_whole(
            &d,
            &NoiseSigma::Scalar(0.01),
            0,
            &BilateralParams::default(),
        );
        sharpen::richardson_lucy_whole(&n, &p, &NoiseSigma::Scalar(0.01), 0)
    };
    let base = run(&m);
    let (px, py) = (100usize, 78usize);
    let mut m2 = m.clone();
    m2.data[py * W + px] += 0.2;
    let out = run(&m2);
    for y in 0..H {
        for x in 0..W {
            if base.pixel(x, y) != out.pixel(x, y) {
                let need = required_input_rect(&chain, Rect::new(x, y, 1, 1), frame);
                assert!(
                    need.x <= px && px < need.x1() && need.y <= py && py < need.y1(),
                    "output ({x},{y}) depends on ({px},{py}) outside {need:?}"
                );
            }
        }
    }
}
