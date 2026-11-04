//! The spatial passes of F-05 (`docs/24` sections 3.9 and 8): tiles, determinism, uniform areas, preview against
//! export, grain.
#![allow(clippy::needless_range_loop)]
mod bake_common;
use std::sync::{Arc, OnceLock};

use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_color::gamut::LinearAwg4;
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_film::bake::{BakeSettings, Baker};
use ym_film::grain;
use ym_film::spatial::*;
use ym_film::{FilmChain, Output};

fn shared_baker() -> &'static Baker {
    static B: OnceLock<Baker> = OnceLock::new();
    B.get_or_init(|| baker(4))
}

fn film_with(id: &str, settings: SpatialSettings) -> FilmFrame {
    let chain: FilmChain = chain(id, Output::DisplayP3);
    let mut bs = BakeSettings::of(&chain);
    bs.t3a_size = 33;
    bs.t3b_size = 33;
    let baked = shared_baker().bake(&chain, &bs).expect("bake");
    let model = SpatialModel::load(&data(), id).expect("model");
    FilmFrame::new(baked, model, settings)
}

fn portra(settings: SpatialSettings) -> FilmFrame {
    film_with("kodak_portra_400", settings)
}

/// A photo-like test image, continuous in the frame's normalised coordinates: a gradient, a bright soft disc
/// (a highlight for the halation), a dark bar with a hard edge and a coloured patch.
fn image_at(x: f64, y: f64, aspect: f64, soft: f64) -> [f64; 3] {
    let (u, v) = (x, y);
    let mut lin = [
        0.12 + 0.25 * u,
        0.15 + 0.2 * v / aspect,
        0.1 + 0.1 * (1.0 - u),
    ];
    let r = ((u - 0.3).powi(2) + (v - 0.3 * aspect).powi(2)).sqrt();
    let disc = 1.0 / (1.0 + ((r - 0.06) / (0.012 * soft)).exp());
    for c in 0..3 {
        lin[c] += 12.0 * disc;
    }
    let w = 0.004 * soft;
    let bar = 1.0 / (1.0 + (-(u - 0.55) / w).exp()) / (1.0 + ((u - 0.7) / w).exp());
    let e = bar / (1.0 + (-(v - 0.15 * aspect) / w).exp());
    for c in 0..3 {
        lin[c] *= 1.0 - 0.9 * e;
    }
    let patch = 1.0
        / (1.0 + (((u - 0.8).abs().max((v - 0.6 * aspect).abs()) - 0.05) / (0.004 * soft)).exp());
    lin[0] += 0.5 * patch;
    lin[1] *= 1.0 - 0.6 * patch;
    logc4_encode_rgb(lin)
}

fn make_input(frame: Frame, rect: Rect) -> Plane {
    make_input_soft(frame, rect, 1.0)
}

fn make_input_soft(frame: Frame, rect: Rect, soft: f64) -> Plane {
    let mut p = Plane::new(rect, false);
    let aspect = frame.height as f64 / frame.width as f64;
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let v = image_at(
                (x as f64 + 0.5) / frame.width as f64,
                (y as f64 + 0.5) / frame.width as f64,
                aspect,
                soft,
            );
            let i = ((y - rect.y0) * rect.width() + (x - rect.x0)) * 3;
            for c in 0..3 {
                p.data[i + c] = v[c] as f32;
            }
        }
    }
    p
}

fn same_bits(a: &Plane, b: &Plane, tag: &str) {
    assert_eq!(a.rect, b.rect, "{tag}");
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{tag}: value {i}: {x} vs {y}");
    }
}

fn crop(p: &Plane, r: Rect) -> Plane {
    let mut o = Plane::new(r, false);
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let i = ((y - r.y0) * r.width() + (x - r.x0)) * 3;
            o.data[i..i + 3].copy_from_slice(&p.px(x, y));
        }
    }
    o
}

fn whole(f: &FilmFrame, frame: Frame, opt: RunOptions) -> Plane {
    let rect = frame.rect();
    f.run_tile(frame, rect, &make_input(frame, rect), opt)
        .expect("run")
}

fn run_rect(f: &FilmFrame, frame: Frame, tile: Rect, opt: RunOptions) -> Plane {
    let need = f.required_input(frame, tile);
    f.run_tile(frame, tile, &make_input(frame, need), opt)
        .expect("tile")
}

const FRAME: Frame = Frame {
    width: 1280,
    height: 853,
};

#[test]
fn halos_at_two_sizes() {
    let f = portra(SpatialSettings::default());
    for long in [2000, 8000] {
        let fr = Frame {
            width: long,
            height: long * 2 / 3,
        };
        let h = f.halos(fr);
        eprintln!(
            "halo at {long} px: halation {} px, diffusion {} px, grain {} px, total {}",
            h.halation,
            h.diffusion,
            h.grain,
            h.total()
        );
    }
    let h8 = f.halos(Frame {
        width: 8000,
        height: 5333,
    });
    assert!(h8.halation <= 330 && h8.halation >= 250, "{h8:?}");
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let f = portra(SpatialSettings::default());
    let opt = RunOptions::default();
    let all = whole(&f, FRAME, opt);
    let h = f.halos(FRAME);
    eprintln!("halos {h:?}");
    // Two tile sizes, with origins that are not multiples of anything, at the corner and in the middle.
    for (tw, th) in [(97, 61), (301, 200)] {
        for (ox, oy) in [
            (0, 0),
            (503, 311),
            (FRAME.width - tw, FRAME.height - th),
            (251, 0),
        ] {
            let tile = Rect {
                x0: ox,
                y0: oy,
                x1: ox + tw,
                y1: oy + th,
            };
            let got = run_rect(&f, FRAME, tile, opt);
            same_bits(
                &got,
                &crop(&all, tile),
                &format!("tile {tw}x{th} at {ox},{oy}"),
            );
        }
    }
}

#[test]
fn repeat_threads_and_poison_do_not_change_a_bit() {
    let f = portra(SpatialSettings::default());
    let a = whole(&f, FRAME, RunOptions::default());
    let b = whole(&f, FRAME, RunOptions::default());
    same_bits(&a, &b, "repeat");
    let c = whole(
        &f,
        FRAME,
        RunOptions {
            threads: 8,
            poison: false,
        },
    );
    same_bits(&a, &c, "8 threads");
    let d = whole(
        &f,
        FRAME,
        RunOptions {
            threads: 3,
            poison: true,
        },
    );
    same_bits(&a, &d, "poison, 3 threads");
    assert!(a.data.iter().all(|v| v.is_finite()));
    // A tile under poison too.
    let tile = Rect {
        x0: 400,
        y0: 100,
        x1: 520,
        y1: 190,
    };
    let t = run_rect(
        &f,
        FRAME,
        tile,
        RunOptions {
            threads: 2,
            poison: true,
        },
    );
    same_bits(&t, &crop(&a, tile), "poison tile");
}

#[test]
fn the_far_corner_of_an_8658_by_5772_frame() {
    let fr = Frame {
        width: 8658,
        height: 5772,
    };
    let f = portra(SpatialSettings::default());
    let opt = RunOptions {
        threads: 4,
        poison: true,
    };
    let small = Rect {
        x0: fr.width - 40,
        y0: fr.height - 33,
        x1: fr.width,
        y1: fr.height,
    };
    let big = Rect {
        x0: fr.width - 211,
        y0: fr.height - 150,
        x1: fr.width,
        y1: fr.height,
    };
    let a = run_rect(&f, fr, small, opt);
    let b = run_rect(&f, fr, big, opt);
    same_bits(&a, &crop(&b, small), "far corner");
    // Away from the edge, deep in the frame: a different tile origin, the same absolute coordinates.
    let mid = Rect {
        x0: 8000,
        y0: 5000,
        x1: 8057,
        y1: 5041,
    };
    let mid_big = Rect {
        x0: 7990,
        y0: 4990,
        x1: 8100,
        y1: 5100,
    };
    same_bits(
        &run_rect(&f, fr, mid, opt),
        &crop(&run_rect(&f, fr, mid_big, opt), mid),
        "far middle",
    );
    let h = f.halos(fr);
    eprintln!("8658 x 5772 halos {h:?}");
}

#[test]
fn a_slide_and_a_zero_amount_also_tile() {
    let s = SpatialSettings {
        halation_amount: 2.0,
        halation_size: 1.5,
        grain_amount: 1.5,
        grain_size: 2.0,
        grain_seed: 77,
        diffusion_size: 1.3,
    };
    for id in ["fujifilm_velvia_100", "kodak_vision3_500t"] {
        let f = film_with(id, s);
        let all = whole(&f, FRAME, RunOptions::default());
        let tile = Rect {
            x0: 333,
            y0: 222,
            x1: 640,
            y1: 480,
        };
        same_bits(
            &run_rect(&f, FRAME, tile, RunOptions::default()),
            &crop(&all, tile),
            id,
        );
    }
    // Everything off: the tile is the per-pixel chain (plus the emulsion's scatter).
    let f = portra(SpatialSettings::off());
    let tile = Rect {
        x0: 10,
        y0: 10,
        x1: 90,
        y1: 70,
    };
    let all = whole(&f, FRAME, RunOptions::default());
    same_bits(
        &run_rect(&f, FRAME, tile, RunOptions::default()),
        &crop(&all, tile),
        "off",
    );
}

fn uniform_frame(v: [f64; 3], frame: Frame) -> Plane {
    let mut p = Plane::new(frame.rect(), false);
    for px in p.data.chunks_exact_mut(3) {
        for c in 0..3 {
            px[c] = v[c] as f32;
        }
    }
    p
}

#[test]
fn a_uniform_frame_gives_the_per_pixel_chain() {
    let fr = Frame {
        width: 300,
        height: 200,
    };
    for id in ["kodak_portra_400", "fujifilm_velvia_100", "kodak_ektar_100"] {
        for s in [
            SpatialSettings::off(),
            SpatialSettings {
                grain_amount: 0.0,
                ..SpatialSettings::default()
            },
        ] {
            let f = film_with(id, s);
            let mut worst = 0.0_f64;
            for lin in [0.02, 0.18, 0.5, 2.0, 8.0] {
                let v = logc4_encode_rgb([lin, lin * 0.8, lin * 1.2]);
                let inp = uniform_frame(v, fr);
                let out = f
                    .run_tile(fr, fr.rect(), &inp, RunOptions::default())
                    .expect("run");
                // The reference sees the f32 input.
                let v32 = [0, 1, 2].map(|c| f64::from(inp.data[c]));
                let want = f.baked().run(v32);
                let got = out.px(150, 100);
                for c in 0..3 {
                    worst = worst.max((f64::from(got[c]) - want[c]).abs());
                }
            }
            eprintln!(
                "{id}: uniform frame against the per-pixel chain: {worst:.2e} (diffusion {})",
                s.diffusion_size
            );
            assert!(worst < 1e-6, "{id}: {worst:e}");
        }
    }
}

/// Display linear of a LogC4 pixel in the Output's primaries.
fn display_linear(p: [f32; 3], cine: &CineRendering) -> [f64; 3] {
    cine.forward(LinearAwg4(logc4_decode_rgb(p.map(f64::from))))
}

fn preview_error(s: SpatialSettings) -> (f64, f64, (usize, usize)) {
    let cine = CineRendering::new(Output::DisplayP3.target());
    let f = portra(s);
    let k = 4;
    let big = Frame {
        width: 2048,
        height: 1366,
    };
    let small = Frame {
        width: big.width / k,
        height: big.height / k,
    };
    // Soft edges (a preview is an area-mean of the photo, and the colour curves are not linear, so a hard
    // edge would measure the resampling and not the passes). The preview's input is the export's, area-mean in
    // scene linear.
    let rect = big.rect();
    let ein = make_input_soft(big, rect, 6.0);
    let mut pin = Plane::new(small.rect(), false);
    for y in 0..small.height {
        for x in 0..small.width {
            let mut acc = [0.0_f64; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let p = ein.px(x * k + dx, y * k + dy);
                    let l = logc4_decode_rgb(p.map(f64::from));
                    for c in 0..3 {
                        acc[c] += l[c] / (k * k) as f64;
                    }
                }
            }
            let v = logc4_encode_rgb(acc);
            let i = (y * small.width + x) * 3;
            for c in 0..3 {
                pin.data[i + c] = v[c] as f32;
            }
        }
    }
    let opt = RunOptions {
        threads: 4,
        poison: false,
    };
    let export = f.run_tile(big, rect, &ein, opt).expect("export");
    let preview = f.run_tile(small, small.rect(), &pin, opt).expect("preview");
    let (mut worst, mut sum, mut n, mut at) = (0.0_f64, 0.0_f64, 0.0_f64, (0, 0));
    for y in 0..small.height {
        for x in 0..small.width {
            let mut acc = [0.0_f64; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let d = display_linear(export.px(x * k + dx, y * k + dy), &cine);
                    for c in 0..3 {
                        acc[c] += d[c] / (k * k) as f64;
                    }
                }
            }
            let p = display_linear(preview.px(x, y), &cine);
            for c in 0..3 {
                let e = (p[c] - acc[c]).abs();
                if e > worst {
                    worst = e;
                    at = (x, y);
                }
                sum += e;
                n += 1.0;
            }
        }
    }
    (worst, sum / n, at)
}

#[test]
fn preview_against_the_reduced_export_for_halation_and_diffusion() {
    let none = SpatialSettings {
        grain_amount: 0.0,
        ..SpatialSettings::default()
    };
    let floor = SpatialSettings {
        diffusion_size: 0.0,
        ..SpatialSettings::off()
    };
    let (fw, fm, fat) = preview_error(floor);
    eprintln!(
        "sampling floor (no halation, no diffusion; scatter only): largest {fw:.4} at {fat:?}, mean {fm:.5}"
    );
    let (w, m, at) = preview_error(none);
    eprintln!(
        "preview k=4 against reduced export (halation + diffusion): largest {w:.4} at {at:?}, mean {m:.5} (gate 0.02, 0.002)"
    );
    assert!(w <= 0.02, "largest {w}");
    assert!(m <= 0.002, "mean {m}");
}

fn flat_density(frame: Frame, tile: Rect, d: [f32; 3]) -> Plane {
    let mut p = Plane::new(tile, false);
    for px in p.data.chunks_exact_mut(3) {
        px.copy_from_slice(&d);
    }
    let _ = frame;
    p
}

fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n)
}

#[test]
fn grain_variance_matches_the_model_and_the_reduced_export() {
    let f = portra(SpatialSettings::default());
    let model = SpatialModel::load(&data(), "kodak_portra_400").expect("model");
    let g = model.grain();
    // Calibration: the model gives the RMS granularity in the 48 um aperture at density 1.0.
    let side48 = (std::f64::consts::PI * 24.0 * 24.0_f64).sqrt();
    for c in 0..3 {
        let s = g.sigma(c, 1.0, side48);
        eprintln!(
            "Portra 400 channel {c}: sigma at D 1.0, 48 um aperture: {s:.5} (RMS {} x 1e-3, fallback {})",
            g.rms_x1000, g.fallback
        );
        assert!((s / (g.rms_x1000 / 1000.0) - 1.0).abs() < 0.01, "{s}");
    }
    // A flat density at two scales; the pass's standard deviation against the model, within 1 %.
    let d = [1.0_f32, 0.9, 0.8];
    let big = Frame {
        width: 4096,
        height: 2731,
    };
    let tile = Rect {
        x0: 1000,
        y0: 1000,
        x1: 1700,
        y1: 1700,
    };
    let need = tile.expand(f.halos(big).grain, big);
    let out = f.apply_grain(
        &flat_density(big, need, d),
        tile,
        big,
        RunOptions {
            threads: 4,
            poison: true,
        },
    );
    let side = 1.0 / big.px_per_um();
    for c in 0..3 {
        let v: Vec<f64> = out
            .data
            .chunks_exact(3)
            .map(|p| f64::from(p[c]) - f64::from(d[c]))
            .collect();
        let (m, var) = stats(&v);
        let want = g.sigma(c, f64::from(d[c]), side);
        eprintln!(
            "channel {c}: mean {m:.2e}, sigma {:.5} against the model {want:.5}",
            var.sqrt()
        );
        assert!(m.abs() < 0.01 * want, "mean {m}");
        assert!(
            (var.sqrt() / want - 1.0).abs() < 0.01,
            "sigma {} vs {want}",
            var.sqrt()
        );
    }
    // With the dye cloud wider than a pixel (a 24,000 px frame) the blur must keep the variance.
    let huge = Frame {
        width: 24000,
        height: 16000,
    };
    let tile = Rect {
        x0: 5000,
        y0: 5000,
        x1: 5800,
        y1: 5800,
    };
    let need = tile.expand(f.halos(huge).grain, huge);
    let out = f.apply_grain(
        &flat_density(huge, need, d),
        tile,
        huge,
        RunOptions::default(),
    );
    let side = 1.0 / huge.px_per_um();
    for c in 0..3 {
        let v: Vec<f64> = out
            .data
            .chunks_exact(3)
            .map(|p| f64::from(p[c]) - f64::from(d[c]))
            .collect();
        let (_, var) = stats(&v);
        let want = g.sigma(c, f64::from(d[c]), side);
        assert!(
            (var.sqrt() / want - 1.0).abs() < 0.015,
            "cloud: sigma {} vs {want}",
            var.sqrt()
        );
    }
    // Preview against the reduced export: the variance of the grain within 10 %.
    let small = Frame {
        width: big.width / 4,
        height: big.height / 4,
    };
    let t_small = Rect {
        x0: 100,
        y0: 100,
        x1: 400,
        y1: 400,
    };
    let t_big = Rect {
        x0: 400,
        y0: 400,
        x1: 1600,
        y1: 1600,
    };
    let e = f.apply_grain(
        &flat_density(big, t_big.expand(f.halos(big).grain, big), d),
        t_big,
        big,
        RunOptions::default(),
    );
    let p = f.apply_grain(
        &flat_density(small, t_small.expand(f.halos(small).grain, small), d),
        t_small,
        small,
        RunOptions::default(),
    );
    for c in 0..3 {
        let mut reduced = Vec::new();
        for y in 0..t_big.height() / 4 {
            for x in 0..t_big.width() / 4 {
                let mut a = 0.0;
                for dy in 0..4 {
                    for dx in 0..4 {
                        a += f64::from(e.px(t_big.x0 + x * 4 + dx, t_big.y0 + y * 4 + dy)[c]);
                    }
                }
                reduced.push(a / 16.0);
            }
        }
        let pv: Vec<f64> = p.data.chunks_exact(3).map(|q| f64::from(q[c])).collect();
        let (_, vr) = stats(&reduced);
        let (_, vp) = stats(&pv);
        eprintln!(
            "channel {c}: grain variance, preview {vp:.3e}, reduced export {vr:.3e}, ratio {:.4} (gate within 10 %)",
            vp / vr
        );
        assert!((vp / vr - 1.0).abs() < 0.10);
    }
}

#[test]
fn the_hash_is_exact_and_the_draws_are_independent() {
    // Integer maths: the same words on every platform.
    assert_eq!(
        grain::hash32(0, 0, 0, 0, 0, 0),
        grain::hash32(0, 0, 0, 0, 0, 0)
    );
    assert_ne!(
        grain::hash32(0, 1, 0, 0, 0, 0),
        grain::hash32(0, 0, 1, 0, 0, 0)
    );
    assert_ne!(
        grain::hash32(1, 0, 0, 0, 0, 0),
        grain::hash32(0, 0, 0, 0, 0, 0)
    );
    // Mean and variance of the draw on a flat field, per channel and sublayer, within 1 %.
    let n = 1000u32;
    for c in 0..3 {
        for s in 0..3 {
            let v: Vec<f64> = (0..n * n)
                .map(|i| grain::normal(11, i % n, i / n, c, s))
                .collect();
            let (m, var) = stats(&v);
            assert!(m.abs() < 0.01, "mean {m}");
            assert!((var - 1.0).abs() < 0.01, "variance {var}");
        }
    }
    // Neighbours, channels, sublayers and seeds are uncorrelated.
    let corr = |a: &dyn Fn(u32) -> f64, b: &dyn Fn(u32) -> f64| -> f64 {
        let m = 400_000u32;
        (0..m).map(|i| a(i) * b(i)).sum::<f64>() / f64::from(m)
    };
    let n0 = |i: u32| grain::normal(3, i % 700, i / 700, 0, 0);
    let right = |i: u32| grain::normal(3, i % 700 + 1, i / 700, 0, 0);
    let down = |i: u32| grain::normal(3, i % 700, i / 700 + 1, 0, 0);
    let ch = |i: u32| grain::normal(3, i % 700, i / 700, 1, 0);
    let sub = |i: u32| grain::normal(3, i % 700, i / 700, 0, 1);
    let seed = |i: u32| grain::normal(4, i % 700, i / 700, 0, 0);
    for (name, o) in [
        ("right", &right as &dyn Fn(u32) -> f64),
        ("down", &down),
        ("channel", &ch),
        ("sublayer", &sub),
        ("seed", &seed),
    ] {
        let r = corr(&n0, o);
        assert!(r.abs() < 0.01, "{name}: {r}");
    }
    let _ = Arc::new(0);
}

#[test]
fn the_coarse_grid_follows_the_exact_gaussian() {
    // A bright disc and an edge, blurred at widths that use the coarse grid, against the direct sum.
    let fr = Frame {
        width: 640,
        height: 480,
    };
    let rect = fr.rect();
    let mut src = Plane::new(rect, false);
    for y in 0..fr.height {
        for x in 0..fr.width {
            let r = (((x as f64 - 200.0).powi(2) + (y as f64 - 200.0).powi(2)).sqrt() - 40.0) / 3.0;
            let v = 1.0 / (1.0 + r.exp()) + if x > 400 { 0.5 } else { 0.0 };
            let i = (y * fr.width + x) * 3;
            src.data[i..i + 3].fill(v as f32);
        }
    }
    for sigma in [7.0, 12.0, 25.0, 40.0] {
        let a = blur(&src, rect, fr, sigma, RunOptions::default());
        let b = blur_reference(&src, rect, fr, sigma, RunOptions::default());
        let worst = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "coarse grid at sigma {sigma}: largest difference from the direct sum {worst:.4} (peak 1.5)"
        );
        assert!(worst < 0.02, "sigma {sigma}: {worst}");
    }
}

/// Speeds, reported with `--ignored --nocapture`.
#[test]
#[ignore = "timing report"]
fn speed_report() {
    let f = portra(SpatialSettings::default());
    let cases = [
        (
            "24 MP (6000 x 4000)",
            Frame {
                width: 6000,
                height: 4000,
            },
        ),
        (
            "2000 px preview",
            Frame {
                width: 2000,
                height: 1333,
            },
        ),
    ];
    for (name, fr) in cases {
        let input = make_input(fr, fr.rect());
        for threads in [1, 4] {
            let t = std::time::Instant::now();
            let out = f
                .run_tile(
                    fr,
                    fr.rect(),
                    &input,
                    RunOptions {
                        threads,
                        poison: false,
                    },
                )
                .expect("run");
            eprintln!(
                "{name}, {threads} thread(s): {:.2} s",
                t.elapsed().as_secs_f64()
            );
            assert!(out.data[0].is_finite());
        }
    }
    let _ = BakeSettings::of;
}

/// A dark frame with a small bright lamp in the middle: where halation shows.
fn lamp_input(fr: Frame) -> Plane {
    let mut p = Plane::new(fr.rect(), false);
    let (cx, cy) = (fr.width / 2, fr.height / 2);
    for y in 0..fr.height {
        for x in 0..fr.width {
            let lamp = x.abs_diff(cx) < 6 && y.abs_diff(cy) < 6;
            let v = logc4_encode_rgb([if lamp { 40.0 } else { 0.03 }; 3]);
            let i = (y * fr.width + x) * 3;
            for c in 0..3 {
                p.data[i + c] = v[c] as f32;
            }
        }
    }
    p
}

#[test]
fn the_passes_do_something_and_the_stock_decides_the_strength() {
    let fr = Frame {
        width: 1280,
        height: 853,
    };
    let cine = CineRendering::new(Output::DisplayP3.target());
    let base = SpatialSettings {
        grain_amount: 0.0,
        halation_amount: 0.0,
        diffusion_size: 0.0,
        ..SpatialSettings::default()
    };
    let opt = RunOptions {
        threads: 4,
        poison: false,
    };
    let inp = lamp_input(fr);
    let run = |id: &str, s: SpatialSettings| {
        film_with(id, s)
            .run_tile(fr, fr.rect(), &inp, opt)
            .expect("run")
    };
    let a = run("kodak_portra_400", base);
    let halo = run(
        "kodak_portra_400",
        SpatialSettings {
            halation_amount: 1.0,
            ..base
        },
    );
    let weak = run(
        "kodak_gold_200",
        SpatialSettings {
            halation_amount: 1.0,
            ..base
        },
    );
    let gold0 = run("kodak_gold_200", base);
    let diff = run(
        "kodak_portra_400",
        SpatialSettings {
            diffusion_size: 1.0,
            ..base
        },
    );
    let (cx, cy) = (fr.width / 2, fr.height / 2);
    let disp = |p: &Plane, x: usize, y: usize| display_linear(p.px(x, y), &cine);
    for off in [8, 12, 20, 40, 80] {
        let (u, v) = (disp(&a, cx + off, cy), disp(&halo, cx + off, cy));
        eprintln!("{off} px from the lamp: base {u:.4?}, Portra 400 halation {v:.4?}");
    }
    let near = |p: &Plane, q: &Plane, off: usize| {
        let (u, v) = (disp(p, cx + off, cy), disp(q, cx + off, cy));
        (v[0] - u[0], v[2] - u[2])
    };
    let (dr, db) = near(&a, &halo, 12);
    assert!(dr > 0.0005, "halation adds light near a lamp: {dr}");
    assert!(dr > db, "the red layer halates most: {dr} {db}");
    let (gr, _) = near(&gold0, &weak, 12);
    assert!(
        gr > dr,
        "a weak backing halates more: Gold 200 {gr}, Portra 400 {dr}"
    );
    let dd = (0..fr.width)
        .map(|x| {
            let (u, v) = (disp(&a, x, cy), disp(&diff, x, cy));
            (0..3).map(|c| (u[c] - v[c]).abs()).fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    eprintln!("diffusion changes a row by at most {dd:.4} display");
    assert!(dd > 0.0005);
}

/// A sample for the eye: one test photo through Portra 400 with halation and grain on, the whole frame beside a
/// 100 % crop. `YM_SAMPLE_OUT=<file.png> cargo test -p ym-film --release --test spatial sample -- --ignored`.
#[test]
#[ignore = "writes a picture"]
fn sample_png() {
    use ym_color::encode::OutputSpace;
    let Ok(out_path) = std::env::var("YM_SAMPLE_OUT") else {
        return;
    };
    let name = std::env::var("YM_SAMPLE_PHOTO").unwrap_or_else(|_| "A1M01739_neutral.png".into());
    let path = photo_dir().join(name);
    let dec = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(&path).expect("photo"),
    ));
    let mut rd = dec.read_info().expect("header");
    let mut buf = vec![0; rd.output_buffer_size().expect("size")];
    let info = rd.next_frame(&mut buf).expect("frame");
    let (sw, sh) = (info.width as usize, info.height as usize);
    let ch = info.color_type.samples();
    let cine_in = CineRendering::new(Output::Srgb.target());
    // Scene linear of the photo, then a bilinear 3x enlargement (a 1000 px photo has no film grain scale).
    let mut lin = vec![[0.0_f64; 3]; sw * sh];
    for (i, l) in lin.iter_mut().enumerate() {
        let mut e = [0.0; 3];
        for c in 0..3 {
            let o = (i * ch + c) * 2;
            e[c] = f64::from(u16::from_be_bytes([buf[o], buf[o + 1]])) / 65535.0;
        }
        *l = cine_in
            .inverse(e.map(ym_color::encode::srgb_decode))
            .scene
            .0;
    }
    let k = 3;
    let fr = Frame {
        width: sw * k,
        height: sh * k,
    };
    let mut inp = Plane::new(fr.rect(), false);
    for y in 0..fr.height {
        for x in 0..fr.width {
            let fx = ((x as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, (sw - 1) as f64);
            let fy = ((y as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, (sh - 1) as f64);
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
            let mut v = [0.0; 3];
            for c in 0..3 {
                let top = lin[y0 * sw + x0][c] * (1.0 - tx) + lin[y0 * sw + x1][c] * tx;
                let bot = lin[y1 * sw + x0][c] * (1.0 - tx) + lin[y1 * sw + x1][c] * tx;
                v[c] = top * (1.0 - ty) + bot * ty;
            }
            let e = logc4_encode_rgb(v);
            let i = (y * fr.width + x) * 3;
            for c in 0..3 {
                inp.data[i + c] = e[c] as f32;
            }
        }
    }
    let id = "kodak_portra_400";
    let chain = chain(id, Output::Srgb);
    let mut bs = BakeSettings::of(&chain);
    bs.t3a_size = 49;
    bs.t3b_size = 65;
    let baked = shared_baker().bake(&chain, &bs).expect("bake");
    let f = FilmFrame::new(
        baked,
        SpatialModel::load(&data(), id).expect("model"),
        SpatialSettings::default(),
    );
    let out = f
        .run_tile(
            fr,
            fr.rect(),
            &inp,
            RunOptions {
                threads: 4,
                poison: false,
            },
        )
        .expect("run");
    let cine = CineRendering::new(Output::Srgb.target());
    let to8 = |p: [f32; 3]| -> [u8; 3] {
        let d = display_linear(p, &cine);
        OutputSpace::Srgb
            .encode_clamped(d)
            .map(|v| (v * 255.0 + 0.5) as u8)
    };
    // Left: the whole frame at 1/3 (box mean in display linear). Right: a 100 % crop of the same size.
    let (w, h) = (sw, sh);
    let mut img = vec![0u8; 2 * w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0_f64; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let d = display_linear(out.px(x * k + dx, y * k + dy), &cine);
                    for c in 0..3 {
                        acc[c] += d[c] / (k * k) as f64;
                    }
                }
            }
            let e = OutputSpace::Srgb
                .encode_clamped(acc)
                .map(|v| (v * 255.0 + 0.5) as u8);
            img[(y * 2 * w + x) * 3..(y * 2 * w + x) * 3 + 3].copy_from_slice(&e);
            let (cx, cy) = (fr.width / 2 - w / 2 + x, fr.height / 2 - h / 2 + y);
            let e = to8(out.px(cx, cy));
            img[(y * 2 * w + w + x) * 3..(y * 2 * w + w + x) * 3 + 3].copy_from_slice(&e);
        }
    }
    let file = std::fs::File::create(&out_path).expect("create");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), (2 * w) as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("header")
        .write_image_data(&img)
        .expect("write");
    eprintln!("wrote {out_path}");
}
