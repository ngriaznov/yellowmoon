//! The Film node's colour passes on the device against `ym-film`'s CPU code (`docs/24-film.md` section 8, GPU parity):
//! P1 (colour in), P3's per-pixel develop with the coupler step, and P5 (colour out with the per-pixel return), each
//! fed with the CPU's own input so that one kernel's error is not another's, then the three together in LogC4.
//!
//! Gates (`docs/24` section 8): colour passes in LogC4 within 2e-4 and within one 10-bit code after C13. The inputs are
//! seeded saturated colours (the whole LogC4 cube), the pixels of the 20 test photos when `testdata/film-bake/` is
//! present, and edge cases (zeros, negative LogC4, the top of the range). The device is the lavapipe software Vulkan
//! adapter in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`).
#![allow(clippy::needless_range_loop)]

mod common;
mod film_common;

use common::{TestGpu, gpus};
use film_common::*;
use ym_film::Output;
use ym_film::bake::BakedFilm;
use ym_film::math::pow10;
use ym_film::spatial::SpatialSettings;
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::film::*;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

#[derive(Clone, Copy)]
enum Stage {
    ColourIn,
    Density,
    Inhibitor,
    ColourOut,
    /// P1, then P3, then P5 on the device, nothing read back between.
    Chain,
}

/// Runs one stage over a `w x h` planar input and returns its planar output.
fn run_stage(
    dev: &TestGpu,
    data: &FilmColourData,
    stage: Stage,
    input: &[f32],
    w: usize,
    h: usize,
) -> Vec<f32> {
    let (data, input) = (data.clone(), input.to_vec());
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let film = FilmDevice::upload(ctx, &data)?;
            let rect = FRect {
                x0: 0,
                y0: 0,
                x1: w as i32,
                y1: h as i32,
            };
            let src = FilmImage::upload(ctx, "film test input", rect, &input)?;
            let mid = FilmImage::alloc(ctx, "film test mid", rect)?;
            let dst = FilmImage::alloc(ctx, "film test out", rect)?;
            let mut passes = PassList::new(ctx, "film test");
            match stage {
                Stage::ColourIn => encode_colour_in(&mut passes, &film, (w, h), &src, &dst, rect)?,
                Stage::Density => encode_develop(
                    &mut passes,
                    &film,
                    DevelopMode::Density,
                    &src,
                    None,
                    &dst,
                    rect,
                )?,
                Stage::Inhibitor => encode_develop(
                    &mut passes,
                    &film,
                    DevelopMode::Inhibitor,
                    &src,
                    None,
                    &dst,
                    rect,
                )?,
                Stage::ColourOut => encode_colour_out(&mut passes, &film, &src, &dst, rect)?,
                Stage::Chain => {
                    let d = FilmImage::alloc(ctx, "film test d", rect)?;
                    encode_colour_in(&mut passes, &film, (w, h), &src, &mid, rect)?;
                    encode_develop(
                        &mut passes,
                        &film,
                        DevelopMode::Density,
                        &mid,
                        None,
                        &d,
                        rect,
                    )?;
                    encode_colour_out(&mut passes, &film, &d, &dst, rect)?;
                    passes.keep(d.img.buf);
                }
            }
            let read = dst.img.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            dst.img.take(&done, read)
        })
        .expect("film stage")
}

/// Every `photo_stride`-th pixel of the 20 test photos (empty when `testdata/film-bake/` is absent).
fn photo_set(photo_stride: usize) -> Vec<[f64; 3]> {
    photo_pixels(photo_stride)
        .into_iter()
        .flat_map(|(_, p)| p)
        .collect()
}

/// The photos and the synthetic colours together (the pixels the pointwise stage tests use).
fn test_pixels(photo_stride: usize) -> Vec<[f64; 3]> {
    let mut px = synthetic_set();
    px.extend(photo_set(photo_stride));
    px
}

/// Saturated colours: six hues, four saturations, eleven exposures from -6 to +4 stops, as linear AWG4 mixed from the
/// primaries.
fn synthetic_set() -> Vec<[f64; 3]> {
    let mut px: Vec<[f64; 3]> = Vec::new();
    let hues = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    for h in hues {
        for s in [0.25, 0.5, 0.75, 0.9] {
            for e in -6..=4 {
                let lum = 0.18 * 2.0_f64.powi(e);
                px.push(ym_color::logc::logc4_encode_rgb(
                    h.map(|c| lum * ((1.0 - s) + s * c * 2.0)),
                ));
            }
        }
    }
    px
}

/// The reported pixels (not gated, as the owner decided for the CPU's random colours): the whole LogC4 cube, seeded,
/// its corners and the extremes of the range, where the pre-image of the display colour is ill-conditioned (a fold, the
/// flat top of the tone curve).
fn random_pixels() -> Vec<[f64; 3]> {
    let mut rng = Rng(0xF11A);
    let mut px: Vec<[f64; 3]> = (0..4000)
        .map(|_| [rng.unit(), rng.unit(), rng.unit()])
        .collect();
    // Saturated corners and edges of the cube, darks, the linear branch (negative LogC4), the top.
    for r in [0.0, 0.5, 1.0] {
        for g in [0.0, 0.5, 1.0] {
            for b in [0.0, 0.5, 1.0] {
                px.push([r, g, b]);
            }
        }
    }
    px.extend([
        [-0.02, 0.1, 0.3],
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.3, 0.3, 0.3],
        [0.6, 0.6, 0.6],
        [0.9, 0.9, 0.9],
    ]);
    px
}

struct Stats {
    max: f64,
    p999: f64,
}

fn stats(mut d: Vec<f64>) -> Stats {
    d.sort_by(f64::total_cmp);
    Stats {
        max: *d.last().unwrap_or(&0.0),
        p999: d[(d.len() * 999 / 1000).min(d.len() - 1)],
    }
}

fn side(w: usize, n: usize) -> (usize, usize) {
    (w, n.div_ceil(w))
}

const CASES: [(&str, Output); 4] = [
    ("kodak_portra_400", Output::DisplayP3),
    ("kodak_ektar_100", Output::Srgb),
    ("fujifilm_velvia_100", Output::DisplayP3),
    ("kodak_vision3_500t", Output::Srgb),
];

fn baked(id: &str, o: Output) -> BakedFilm {
    film(id, o, SpatialSettings::default()).baked().clone()
}

#[test]
fn colour_in_matches_the_cpu() {
    let px = test_pixels(40);
    let (w, h) = side(64, px.len());
    let input = planar(&px, w, h);
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (id, o) in CASES {
            let b = baked(id, o);
            let got = run_stage(dev, &colour_data(&b), Stage::ColourIn, &input, w, h);
            let (mut rel, mut dlog) = (Vec::new(), Vec::new());
            for (i, p) in px.iter().enumerate() {
                // The device reads the f32 plane the host uploaded.
                let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
                let le = b.exposure_of(q);
                for c in 0..3 {
                    let want = f64::from(pow10(le[c]) as f32);
                    let g = f64::from(got[c * w * h + i]);
                    rel.push(((g - want) / want).abs());
                    dlog.push((g.log10() - want.log10()).abs());
                }
                let _ = p;
            }
            let (r, l) = (stats(rel), stats(dlog));
            println!(
                "P1 {id:<22} {o:?}: relative E max {:.2e} p99.9 {:.2e}; log10 E max {:.2e}",
                r.max, r.p999, l.max
            );
            assert!(r.max < 2e-4, "{id}: E relative {:.3e}", r.max);
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn develop_matches_the_cpu() {
    let px = test_pixels(40);
    let (w, h) = side(64, px.len());
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (id, o) in CASES {
            let b = baked(id, o);
            // The CPU's own exposure as the input.
            let e: Vec<[f32; 3]> = px
                .iter()
                .map(|&p| b.exposure_of(p).map(|v| pow10(v) as f32))
                .collect();
            let mut input = vec![0.0_f32; 3 * w * h];
            for i in 0..w * h {
                for c in 0..3 {
                    input[c * w * h + i] = e[i.min(e.len() - 1)][c];
                }
            }
            for (stage, name) in [(Stage::Density, "density"), (Stage::Inhibitor, "inhibitor")] {
                let got = run_stage(dev, &colour_data(&b), stage, &input, w, h);
                let mut d = Vec::new();
                for i in 0..px.len() {
                    let le = e[i].map(|v| f64::from(v).max(1e-30).log10());
                    let dens = b.film_density(le);
                    let want = match stage {
                        Stage::Inhibitor => b.coupler_inhibitor(dens),
                        _ => dens,
                    };
                    for c in 0..3 {
                        d.push((f64::from(got[c * w * h + i]) - want[c]).abs());
                    }
                }
                let s = stats(d);
                println!(
                    "P3 {name:<9} {id:<22} {o:?}: max {:.2e} p99.9 {:.2e}",
                    s.max, s.p999
                );
                assert!(s.max < 1e-4, "{id} {name}: {:.3e}", s.max);
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

/// The film densities of the test pixels (the CPU's), as the input of P5.
fn densities(b: &BakedFilm, px: &[[f64; 3]]) -> Vec<[f64; 3]> {
    px.iter()
        .map(|&p| b.film_density(b.exposure_of(p)))
        .collect()
}

fn code_diff(
    cine: &ym_color::display::cine::CineRendering,
    space: ym_color::encode::OutputSpace,
    a: [f64; 3],
    b: [f64; 3],
) -> f64 {
    let (ca, cb) = (codes10(cine, space, a), codes10(cine, space, b));
    (0..3).map(|c| (ca[c] - cb[c]).abs()).fold(0.0, f64::max)
}

/// Compares the device's LogC4 of `px` through `stage` with the CPU's and prints the numbers; returns (LogC4 max,
/// codes max).
fn compare_out(
    dev: &TestGpu,
    id: &str,
    o: Output,
    stage: Stage,
    px: &[[f64; 3]],
    label: &str,
) -> (f64, f64, f64) {
    let (w, h) = side(64, px.len());
    let b = baked(id, o);
    let (inputs, name): (Vec<[f64; 3]>, &str) = match stage {
        Stage::ColourOut => (densities(&b, px), "P5"),
        _ => (px.to_vec(), "chain"),
    };
    let input = planar(&inputs, w, h);
    let got = run_stage(dev, &colour_data(&b), stage, &input, w, h);
    let cine = ym_color::display::cine::CineRendering::new(output_target(o));
    let space = output_space(o);
    let (mut dl, mut dc) = (Vec::new(), Vec::new());
    for i in 0..px.len() {
        let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
        let want = match stage {
            Stage::ColourOut => b.colour_out(q),
            _ => b.run(q),
        };
        let g = [0, 1, 2].map(|c| f64::from(got[c * w * h + i]));
        dl.extend((0..3).map(|c| (g[c] - want[c]).abs()));
        dc.push(code_diff(&cine, space, g, want));
    }
    let over = dl.iter().filter(|&&d| d > 2e-4).count() as f64 / dl.len() as f64;
    let (l, c) = (stats(dl), stats(dc));
    println!(
        "{name} {id:<22} {o:?} {label:<8} n {:>6}: LogC4 max {:.2e} p99.9 {:.2e} over 2e-4 {:.3}%; codes max {:.3} p99.9 {:.4}",
        px.len(),
        l.max,
        l.p999,
        100.0 * over,
        c.max,
        c.p999
    );
    (l.max, c.max, over)
}

/// Whether the cine rendering can reach the print's display colour of a film density: its chroma is within the reachable
/// length of its ray (chroma fraction 1) and no display channel is held at the floor.
fn reachable(b: &BakedFilm, d: [f64; 3]) -> bool {
    let display = if b.is_slide() {
        b.scan_display(d)
    } else {
        b.scan_display(b.paper_density(b.print_log_exposure(d)))
    };
    if display
        .iter()
        .any(|&c| c.is_nan() || c <= ym_color::display::cine::D_MIN)
    {
        return false;
    }
    let cine = ym_color::display::cine::CineRendering::new(output_target(b.output()));
    let c = cine.display_coordinates(display);
    c.ca2.hypot(c.cb2) <= b.core_table().reachable_length(c)
}

/// The owner's rule for the analogous baking gate (2026-11-04): "Gate photos, report random." The photos' pixels and the
/// synthetic colours the cine rendering can reach (chroma fraction 1, no display channel held at the floor) are gated
/// strictly, with no allowance: at most 2e-4 in LogC4 (`docs/24` section 8) and one 10-bit code after C13. The
/// unreachable or floor-held synthetic colours and the random LogC4 cube are printed (largest, 99.9 %, share over 2e-4)
/// and not asserted.
fn out_test(stage: Stage) {
    let (photos, synth, random) = (photo_set(40), synthetic_set(), random_pixels());
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (id, o) in CASES {
            let b = baked(id, o);
            let (yes, no): (Vec<_>, Vec<_>) = synth
                .iter()
                .partition(|&&p| reachable(&b, b.film_density(b.exposure_of(p))));
            let (yes, no): (Vec<[f64; 3]>, Vec<[f64; 3]>) = (
                yes.into_iter().copied().collect(),
                no.into_iter().copied().collect(),
            );
            for (set, label) in [(&photos, "photos"), (&yes, "reach")] {
                if set.is_empty() {
                    continue;
                }
                let (l, c, _) = compare_out(dev, id, o, stage, set, label);
                assert!(l <= 2e-4, "{id} {label}: LogC4 {l:.3e}");
                assert!(c <= 1.0, "{id} {label}: {c:.3} codes");
            }
            for (set, label) in [(&no, "unreach"), (&random, "random")] {
                if !set.is_empty() {
                    compare_out(dev, id, o, stage, set, label);
                }
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn colour_out_matches_the_cpu() {
    out_test(Stage::ColourOut);
}

#[test]
fn the_colour_passes_together_match_the_chain() {
    out_test(Stage::Chain);
}

#[test]
#[ignore = "diagnostic: where the colour-out LogC4 error sits"]
fn diagnose_colour_out() {
    let px = test_pixels(40);
    let n_cube = 264;
    let (w, h) = side(64, px.len());
    for dev in f32_devices() {
        for (id, o) in [CASES[0], CASES[1]] {
            let b = baked(id, o);
            let dens = densities(&b, &px);
            let mut input = vec![0.0_f32; 3 * w * h];
            for i in 0..w * h {
                for c in 0..3 {
                    input[c * w * h + i] = dens[i.min(dens.len() - 1)][c] as f32;
                }
            }
            let got = run_stage(dev, &colour_data(&b), Stage::ColourOut, &input, w, h);
            let cine = ym_color::display::cine::CineRendering::new(output_target(o));
            let mut rows: Vec<(f64, usize)> = Vec::new();
            for i in 0..px.len() {
                let d32 = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
                let want = b.colour_out(d32);
                let g = [0, 1, 2].map(|c| f64::from(got[c * w * h + i]));
                rows.push((
                    (0..3).map(|c| (g[c] - want[c]).abs()).fold(0.0, f64::max),
                    i,
                ));
            }
            let over = |lo: usize, hi: usize| {
                rows.iter()
                    .filter(|r| r.1 >= lo && r.1 < hi && r.0 > 2e-4)
                    .count()
            };
            println!(
                "{id}: over 2e-4: cube {} of {n_cube}, photos {} of {}",
                over(0, n_cube),
                over(n_cube, px.len()),
                px.len() - n_cube
            );
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(e, i) in rows.iter().take(8) {
                let d32 = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
                let display = {
                    let p = b.print_log_exposure(d32);
                    let dp = b.paper_density(p);
                    if b.is_slide() {
                        b.scan_display(d32)
                    } else {
                        b.scan_display(dp)
                    }
                };
                let c = cine.display_coordinates(display);
                let reach = b.core_table().reachable_length(c);
                let rho = c.ca2.hypot(c.cb2);
                println!(
                    "  err {e:.2e} px {i} {} display {:?} y2 {:.3e} rho {rho:.3} reach {reach:.3}",
                    if i < n_cube { "cube" } else { "photo" },
                    display.map(|v| (v * 1e4).round() / 1e4),
                    c.y2
                );
            }
        }
    }
}

#[test]
#[ignore = "timing on lavapipe (rough per-pixel cost): run with --ignored --nocapture"]
fn cost_per_pixel_on_lavapipe() {
    let px = test_pixels(8);
    let (w, h) = (512, 256);
    let input = planar(&px, w, h);
    let b = baked("kodak_portra_400", Output::DisplayP3);
    let data = colour_data(&b);
    let dens = densities(&b, &px[..w * h.min(px.len() / w)]);
    let dinput = planar(&dens, w, h);
    for dev in f32_devices() {
        for (stage, name, inp) in [
            (Stage::ColourIn, "P1", &input),
            (Stage::Density, "P3 develop", &input),
            (Stage::ColourOut, "P5", &dinput),
        ] {
            let _ = run_stage(dev, &data, stage, inp, w, h);
            let t = std::time::Instant::now();
            let _ = run_stage(dev, &data, stage, inp, w, h);
            let us = t.elapsed().as_secs_f64() * 1e6 / (w * h) as f64;
            println!(
                "cost {name}: {us:.2} us a pixel on lavapipe (4 vCPUs, includes the table upload)"
            );
        }
    }
}

/// F-08c diagnostic: Rec.2020 colour out, device against CPU, in 10-bit codes after C13 (the CPU's f64 forward for both),
/// worst first, with the CPU's display colour of the print.
#[test]
#[ignore = "diagnostic: F-08c, Rec.2020 colour-out codes"]
fn diagnose_rec2020_codes() {
    let mut px = synthetic_set();
    px.extend(random_pixels());
    px.extend(photo_set(40));
    let (w, h) = side(64, px.len());
    for dev in f32_devices() {
        for id in [
            "kodak_portra_400",
            "kodak_ektar_100",
            "fujifilm_velvia_100",
            "kodak_vision3_500t",
        ] {
            let o = Output::Rec2020;
            let b = baked(id, o);
            let dens = densities(&b, &px);
            let input = planar(&dens, w, h);
            let got = run_stage(dev, &colour_data(&b), Stage::ColourOut, &input, w, h);
            let cine = ym_color::display::cine::CineRendering::new(output_target(o));
            let space = output_space(o);
            let mut rows = Vec::new();
            for i in 0..px.len() {
                let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
                let want = b.colour_out(q);
                let g = [0, 1, 2].map(|c| f64::from(got[c * w * h + i]));
                rows.push((code_diff(&cine, space, g, want) * 64.0, i, g, want, q));
            }
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            println!("{id}: worst codes of 65535:");
            for (e, i, g, want, q) in rows.iter().take(6) {
                let dp = if b.is_slide() {
                    *q
                } else {
                    b.paper_density(b.print_log_exposure(*q))
                };
                let display = b.scan_display(dp);
                println!(
                    "  {e:.1} px {i} display {display:?} dev {g:.7?} cpu {want:.7?} dens {q:.4?}",
                );
            }
        }
    }
}

/// F-08c diagnostic: a neutral wedge down to black through the whole chain, Rec.2020 against the others.
#[test]
#[ignore = "diagnostic: F-08c, neutral wedge"]
fn diagnose_wedge() {
    let px: Vec<[f64; 3]> = (0..60)
        .map(|k| {
            let lum = 0.18 * 2.0_f64.powf(-0.5 * f64::from(k));
            ym_color::logc::logc4_encode_rgb([lum; 3])
        })
        .collect();
    let (w, h) = side(64, px.len());
    for dev in f32_devices() {
        for o in [Output::Rec2020, Output::DisplayP3] {
            let b = baked("kodak_portra_400", o);
            let input = planar(&px, w, h);
            let got = run_stage(dev, &colour_data(&b), Stage::Chain, &input, w, h);
            let cine = ym_color::display::cine::CineRendering::new(output_target(o));
            let space = output_space(o);
            println!("{o:?}");
            for i in 0..px.len() {
                let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
                let want = b.run(q);
                let g = [0, 1, 2].map(|c| f64::from(got[c * w * h + i]));
                let dd = b.scan_display(
                    b.paper_density(b.print_log_exposure(b.film_density(b.exposure_of(q)))),
                );
                println!(
                    "  {i:2} codes {:7.2} display {:?} dev {:.7?} cpu {:.7?}",
                    code_diff(&cine, space, g, want) * 64.0,
                    dd,
                    g,
                    want
                );
            }
        }
    }
}

/// F-08c diagnostic: noisy near-black scene colours (some channels negative) through the whole chain.
#[test]
#[ignore = "diagnostic: F-08c, noisy blacks"]
fn diagnose_blacks() {
    let mut rng = Rng(0xB1AC);
    let px: Vec<[f64; 3]> = (0..4096)
        .map(|_| {
            let amp = 10f64.powf(-5.0 + 2.5 * rng.unit());
            ym_color::logc::logc4_encode_rgb([0, 1, 2].map(|_| amp * (2.0 * rng.unit() - 0.5)))
        })
        .collect();
    let (w, h) = side(64, px.len());
    for dev in f32_devices() {
        for o in [Output::Rec2020, Output::DisplayP3] {
            let b = baked("kodak_portra_400", o);
            let input = planar(&px, w, h);
            let got = run_stage(dev, &colour_data(&b), Stage::Chain, &input, w, h);
            let cine = ym_color::display::cine::CineRendering::new(output_target(o));
            let space = output_space(o);
            let mut rows = Vec::new();
            for i in 0..px.len() {
                let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
                let want = b.run(q);
                let g = [0, 1, 2].map(|c| f64::from(got[c * w * h + i]));
                rows.push((code_diff(&cine, space, g, want) * 64.0, i, g, want, q));
            }
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            println!("{o:?}: worst codes of 65535 (n {}):", px.len());
            for (e, i, g, want, q) in rows.iter().take(6) {
                let dd = b.scan_display(
                    b.paper_density(b.print_log_exposure(b.film_density(b.exposure_of(*q)))),
                );
                println!("  {e:.1} px {i} in {q:.5?} display {dd:?} dev {g:.7?} cpu {want:.7?}");
            }
        }
    }
}

/// F-08c diagnostic: the device's LogC4 error in units of the last place of an f32 at the CPU value, for the whole
/// chain and the colour-out pass alone, over the photos' pixels: percentiles.
#[test]
#[ignore = "diagnostic: F-08c, error in ulps"]
fn diagnose_ulp_error() {
    let px = photo_set(7);
    let (w, h) = side(64, px.len());
    for dev in f32_devices() {
        for (id, o) in [
            ("kodak_portra_400", Output::Rec2020),
            ("kodak_portra_400", Output::DisplayP3),
        ] {
            let b = baked(id, o);
            let dens = densities(&b, &px);
            let input = planar(&dens, w, h);
            let got = run_stage(dev, &colour_data(&b), Stage::ColourOut, &input, w, h);
            let scene = planar(&px, w, h);
            let got_chain = run_stage(dev, &colour_data(&b), Stage::Chain, &scene, w, h);
            for (name, g, inp) in [("colour out", &got, &input), ("chain", &got_chain, &scene)] {
                let mut errs: Vec<f64> = Vec::new();
                let mut rnd: Vec<f64> = Vec::new();
                for i in 0..px.len() {
                    let q = [0, 1, 2].map(|c| f64::from(inp[c * w * h + i]));
                    let want = if name == "chain" {
                        b.run(q)
                    } else {
                        b.colour_out(q)
                    };
                    for c in 0..3 {
                        let ulp = f64::from(f32::EPSILON) * 0.5;
                        errs.push((f64::from(g[c * w * h + i]) - want[c]).abs() / ulp);
                        rnd.push((f64::from(want[c] as f32) - want[c]).abs() / ulp);
                    }
                }
                if name == "colour out" && matches!(o, Output::Rec2020) {
                    let mut rows: Vec<(f64, usize)> = (0..px.len())
                        .map(|i| {
                            let q = [0, 1, 2].map(|c| f64::from(inp[c * w * h + i]));
                            let want = b.colour_out(q);
                            (
                                (0..3)
                                    .map(|c| (f64::from(g[c * w * h + i]) - want[c]).abs())
                                    .fold(0.0, f64::max),
                                i,
                            )
                        })
                        .collect();
                    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
                    for &(e, i) in rows.iter().step_by(rows.len() / 400).take(14) {
                        let q = [0, 1, 2].map(|c| f64::from(inp[c * w * h + i]));
                        let dp = b.paper_density(b.print_log_exposure(q));
                        println!(
                            "  err {e:.2e} dens {q:.4?} display {:.6?} dev {:.7?} cpu {:.7?}",
                            b.scan_display(dp),
                            [0, 1, 2].map(|c| g[c * w * h + i]),
                            b.colour_out(q)
                        );
                    }
                }
                errs.sort_by(f64::total_cmp);
                rnd.sort_by(f64::total_cmp);
                let pc = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p) as usize];
                println!(
                    "{id} {o:?} {name} n {}: units of 6e-8 p50 {:.2} p90 {:.2} p99 {:.2} p99.9 {:.2} max {:.2}; the f32 rounding alone p50 {:.2} p99 {:.2} max {:.2}",
                    px.len(),
                    pc(&errs, 0.5),
                    pc(&errs, 0.9),
                    pc(&errs, 0.99),
                    pc(&errs, 0.999),
                    errs[errs.len() - 1],
                    pc(&rnd, 0.5),
                    pc(&rnd, 0.99),
                    rnd[rnd.len() - 1]
                );
            }
        }
    }
}
