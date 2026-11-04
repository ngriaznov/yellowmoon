//! The Film node's spatial passes and the whole node on a tile, on the device against `ym-film` (`docs/24-film.md`
//! sections 3.9, 3.9a and 8): the Gaussians (direct and coarse grid) within 1e-4 of the pass's peak, the grain hash exact
//! and the grain values within 1e-5, the node on a tile within 2e-4 in LogC4 of `FilmFrame::run_tile`, tiles equal to the
//! whole frame bit for bit at two tile sizes (halos and grain included), the far corner of an 8,658 x 5,772 frame (D4),
//! a repeat run and a run without poison (D7).
#![allow(clippy::needless_range_loop)]

mod common;
mod film_common;

use common::{TestGpu, gpus, start_unpoisoned};
use film_common::*;
use ym_film::Output;
use ym_film::grain;
use ym_film::spatial::{self as sp, Frame, Plane, Rect, RunOptions, SpatialSettings};
use ym_gpu::device::{Gpu, StorageMode, StoragePreference};
use ym_gpu::kernels::film::*;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

fn frect(r: Rect) -> FRect {
    FRect {
        x0: r.x0 as i32,
        y0: r.y0 as i32,
        x1: r.x1 as i32,
        y1: r.y1 as i32,
    }
}

/// Interleaved RGB (`ym-film`'s `Plane`) to planar.
fn planar_of(p: &Plane) -> Vec<f32> {
    let n = p.rect.width() * p.rect.height();
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = p.data[i * 3 + c];
        }
    }
    out
}

/// Planar to interleaved.
fn interleaved(v: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[i * 3 + c] = v[c * n + i];
        }
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max)
}

// ---- the Gaussians ----------------------------------------------------------------------------------------

/// A soft edge, a disc and some texture, three channels with different amplitudes, peak 1.5.
fn blur_image(w: usize, h: usize) -> Plane {
    let rect = Rect {
        x0: 0,
        y0: 0,
        x1: w,
        y1: h,
    };
    let mut p = Plane::new(rect, false);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f64 / w as f64, y as f64 / h as f64);
            let edge = if u > 0.4 { 1.0 } else { 0.0 };
            let disc = if (u - 0.7).powi(2) + (v - 0.5).powi(2) < 0.02 {
                0.5
            } else {
                0.0
            };
            let tex = 0.05 * ((x * 7 + y * 13) % 11) as f64 / 11.0;
            for c in 0..3 {
                p.data[(y * w + x) * 3 + c] =
                    ((edge + disc) * (0.6 + 0.2 * c as f64) + tex).min(1.5) as f32;
            }
        }
    }
    p
}

#[test]
fn the_gaussians_match_the_cpu_within_1e_4_of_the_peak() {
    let (w, h) = (400, 300);
    let frame = Frame {
        width: w,
        height: h,
    };
    let src = blur_image(w, h);
    let planar_src = planar_of(&src);
    let peak = 1.5;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for sigma in [0.3, 1.5, 3.9, 4.5, 7.0, 12.0, 25.0, 40.0] {
            for out in [
                Rect {
                    x0: 90,
                    y0: 70,
                    x1: 310,
                    y1: 230,
                },
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: 120,
                    y1: 90,
                },
                Rect {
                    x0: 300,
                    y0: 200,
                    x1: 400,
                    y1: 300,
                },
            ] {
                let want = sp::blur(&src, out, frame, sigma, RunOptions::default());
                let (planar_src, o) = (planar_src.clone(), frect(out));
                let data = colour_data_dummy(dev);
                let got = dev
                    .gpu
                    .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                        let film = FilmDevice::upload(ctx, &data)?;
                        let s = FilmImage::upload(ctx, "src", FRect::frame((w, h)), &planar_src)?;
                        let d = FilmImage::alloc(ctx, "dst", o)?;
                        let mut passes = PassList::new(ctx, "blur");
                        encode_gauss(ctx, &mut passes, &film, (w, h), &s, o, sigma, &d, None)?;
                        let read = d.img.encode_read(&mut passes)?;
                        let done = passes.submit()?.finish()?;
                        d.img.take(&done, read)
                    })
                    .expect("blur");
                let diff =
                    max_diff(&interleaved(&got, out.width() * out.height()), &want.data) / peak;
                println!(
                    "blur sigma {sigma:>5} out {out:?}: {diff:.2e} of the peak (halo {})",
                    sp::blur_halo(sigma)
                );
                assert!(diff < 1e-4, "sigma {sigma}: {diff:.3e}");
                assert_eq!(blur_halo(sigma), sp::blur_halo(sigma));
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

/// Tables for a device whose kernels the test does not run (a Gaussian needs only the dummy buffer).
fn colour_data_dummy(_dev: &TestGpu) -> FilmColourData {
    static D: std::sync::OnceLock<FilmColourData> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        colour_data(
            film(
                "kodak_portra_400",
                Output::DisplayP3,
                SpatialSettings::default(),
            )
            .baked(),
        )
    })
    .clone()
}

#[test]
fn the_taps_are_the_cpu_kernel() {
    for sigma in [0.3, 1.0, 3.9, 12.0, 40.0] {
        assert!(
            (gauss_power(sigma) - sp::kernel_power(sigma)).abs() < 1e-14,
            "{sigma}"
        );
    }
}

// ---- the grain --------------------------------------------------------------------------------------------

fn grain_plan(settings: SpatialSettings, frame: Frame) -> (ym_film::spatial::FilmFrame, FilmPlan) {
    let f = film("kodak_portra_400", Output::DisplayP3, settings);
    let plan = plan_data(&f, frame);
    (f, plan)
}

#[test]
fn the_grain_hash_is_exact() {
    let frame = Frame {
        width: 8658,
        height: 5772,
    };
    let (_, plan) = grain_plan(
        SpatialSettings {
            grain_seed: 0xDEAD_BEEF,
            ..SpatialSettings::default()
        },
        frame,
    );
    let g = plan.grain.clone().expect("grain");
    assert_eq!(g.hash, grain::HASH_CONSTANTS);
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (x0, y0) in [(0, 0), (8600, 5700)] {
            for (c, s) in [(0, 0), (1, 2), (2, 1)] {
                let win = FRect {
                    x0,
                    y0,
                    x1: x0 + 58,
                    y1: y0 + 72,
                };
                let g2 = g.clone();
                let got = dev
                    .gpu
                    .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                        let d = FilmImage::alloc(ctx, "probe", win)?;
                        let mut passes = PassList::new(ctx, "probe");
                        encode_grain_probe(ctx, &mut passes, &g2, c, s, &d, win)?;
                        let read = d.img.encode_read(&mut passes)?;
                        let done = passes.submit()?.finish()?;
                        d.img.take(&done, read)
                    })
                    .expect("probe");
                let n = 58 * 72;
                for i in 0..n {
                    let (x, y) = ((x0 + (i % 58) as i32) as u32, (y0 + (i / 58) as i32) as u32);
                    for k in 0..2 {
                        let want = grain::hash32(g.seed, x, y, c, s, k);
                        assert_eq!(
                            got[k as usize * n + i].to_bits(),
                            want,
                            "hash ({x},{y}) c{c} s{s} k{k}"
                        );
                    }
                    let sum = grain::normal_sum(g.seed, x, y, c, s);
                    assert_eq!(got[2 * n + i], sum as f32, "sum ({x},{y})");
                }
            }
        }
        println!(
            "grain hash: {} words and sums, all exact",
            2 * 3 * 2 * 58 * 72
        );
    }
    assert!(seen || gpus().is_empty());
}

fn density_plane(rect: Rect) -> Plane {
    let mut p = Plane::new(rect, false);
    let (w, h) = (rect.width(), rect.height());
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                p.data[(y * w + x) * 3 + c] = (0.15
                    + 2.2
                        * ((x as f64 / w as f64 + 0.3 * c as f64 + y as f64 / h as f64 * 0.5)
                            % 1.0)) as f32;
            }
        }
    }
    p
}

#[test]
fn the_grain_values_match_the_cpu_within_1e_5() {
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        // With a dye cloud (σ = 0.67 px) and without (σ = 0.055 px at 2,000 px).
        for (frame, size) in [
            (
                Frame {
                    width: 6000,
                    height: 4000,
                },
                2.0,
            ),
            (
                Frame {
                    width: 2000,
                    height: 1333,
                },
                1.0,
            ),
        ] {
            let settings = SpatialSettings {
                grain_size: size,
                grain_seed: 77,
                ..SpatialSettings::default()
            };
            let (f, plan) = grain_plan(settings, frame);
            let (tx, ty) = (frame.width / 2, frame.height / 2);
            let tile = Rect {
                x0: tx,
                y0: ty,
                x1: tx + 120,
                y1: ty + 90,
            };
            let halo = if plan.cloud_px >= MIN_SIGMA_PX {
                blur_halo(plan.cloud_px)
            } else {
                0
            };
            let d_rect = tile.expand(halo, frame);
            let d = density_plane(d_rect);
            let want = f.apply_grain(&d, tile, frame, RunOptions::default());
            let (pd, t) = (planar_of(&d), frect(tile));
            let (plan2, dr) = (plan.clone(), frect(d_rect));
            let data = colour_data_dummy(dev);
            let got = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let film = FilmDevice::upload(ctx, &data)?;
                    let dimg = FilmImage::upload(ctx, "d", dr, &pd)?;
                    let out = FilmImage::alloc(ctx, "out", t)?;
                    let mut passes = PassList::new(ctx, "grain");
                    let g = plan2.grain.clone().expect("grain");
                    encode_grain(ctx, &mut passes, &film, &plan2, &g, &dimg, t, &out)?;
                    let read = out.img.encode_read(&mut passes)?;
                    let done = passes.submit()?.finish()?;
                    out.img.take(&done, read)
                })
                .expect("grain");
            let diff = max_diff(&interleaved(&got, tile.width() * tile.height()), &want.data);
            println!(
                "grain {}x{} size {size} cloud {:.2} px: max {diff:.2e}",
                frame.width, frame.height, plan.cloud_px
            );
            assert!(diff < 1e-5, "{diff:.3e}");
        }
    }
    assert!(seen || gpus().is_empty());
}

// ---- the node on a tile -----------------------------------------------------------------------------------

/// The node on `tile` of the device: input is the test image over the rectangle the tile needs.
fn run_tile_gpu(gpu: &Gpu, data: &FilmColourData, plan: &FilmPlan, tile: FRect) -> Vec<f32> {
    let need = required_input(plan, tile);
    let input = make_input(plan.frame, need);
    let (data, plan) = (data.clone(), plan.clone());
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let film = FilmDevice::upload(ctx, &data)?;
        let src = FilmImage::upload(ctx, "film in", need, &input)?;
        let mut passes = PassList::new(ctx, "film tile");
        let out = encode_film_tile(ctx, &mut passes, &film, &plan, &src, tile)?;
        let read = out.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        out.img.take(&done, read)
    })
    .expect("film tile")
}

fn cpu_tile(f: &ym_film::spatial::FilmFrame, frame: Frame, tile: Rect) -> Vec<f32> {
    let need = f.required_input(frame, tile);
    let input_planar = make_input((frame.width, frame.height), frect(need));
    let n = need.width() * need.height();
    let plane = Plane {
        rect: need,
        data: interleaved(&input_planar, n),
    };
    let out = f
        .run_tile(frame, tile, &plane, RunOptions::default())
        .expect("run_tile");
    planar_of(&out)
}

#[test]
fn the_node_on_a_tile_matches_run_tile() {
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        // The third frame is wide enough for the dye cloud to be wider than a pixel.
        for (id, o, frame) in [
            (
                "kodak_portra_400",
                Output::DisplayP3,
                Frame {
                    width: 1280,
                    height: 853,
                },
            ),
            (
                "fujifilm_velvia_100",
                Output::Srgb,
                Frame {
                    width: 1280,
                    height: 853,
                },
            ),
            (
                "kodak_ektar_100",
                Output::DisplayP3,
                Frame {
                    width: 1280,
                    height: 853,
                },
            ),
            (
                "kodak_portra_400",
                Output::Srgb,
                Frame {
                    width: 6000,
                    height: 4000,
                },
            ),
        ] {
            let (tx, ty) = (frame.width * 3 / 8, frame.height / 3);
            let tile = Rect {
                x0: tx,
                y0: ty,
                x1: tx + 160,
                y1: ty + 120,
            };
            // Grain on with a seed, so the whole node runs; the grain's size makes the cloud wider than a pixel.
            let settings = SpatialSettings {
                grain_size: 2.0,
                grain_seed: 5,
                ..SpatialSettings::default()
            };
            let f = film(id, o, settings);
            let want = cpu_tile(&f, frame, tile);
            let (data, plan) = (colour_data(f.baked()), plan_data(&f, frame));
            let got = run_tile_gpu(&dev.gpu, &data, &plan, frect(tile));
            let diff = max_diff(&got, &want);
            println!(
                "node {id:<22} {o:?}: LogC4 max {diff:.2e} (halos {:?})",
                halos(&plan)
            );
            assert!(diff < 2e-4, "{id}: {diff:.3e}");
        }
    }
    assert!(seen || gpus().is_empty());
}

fn assemble(
    gpu: &Gpu,
    data: &FilmColourData,
    plan: &FilmPlan,
    whole: FRect,
    size: (i32, i32),
) -> Vec<f32> {
    let (w, h) = (whole.width() as usize, whole.height() as usize);
    let mut out = vec![0.0_f32; 3 * w * h];
    let mut y = whole.y0;
    while y < whole.y1 {
        let mut x = whole.x0;
        while x < whole.x1 {
            let t = FRect {
                x0: x,
                y0: y,
                x1: (x + size.0).min(whole.x1),
                y1: (y + size.1).min(whole.y1),
            };
            let v = run_tile_gpu(gpu, data, plan, t);
            let (tw, th) = (t.width() as usize, t.height() as usize);
            for c in 0..3 {
                for r in 0..th {
                    let dst = c * w * h
                        + (t.y0 - whole.y0) as usize * w
                        + r * w
                        + (t.x0 - whole.x0) as usize;
                    out[dst..dst + tw]
                        .copy_from_slice(&v[c * tw * th + r * tw..c * tw * th + (r + 1) * tw]);
                }
            }
            x += size.0;
        }
        y += size.1;
    }
    out
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit_and_a_repeat_and_poison_change_nothing() {
    let frame = Frame {
        width: 360,
        height: 241,
    };
    let settings = SpatialSettings {
        grain_size: 2.0,
        grain_seed: 9,
        ..SpatialSettings::default()
    };
    // At 360 px the halos are small, so use a wide halation and diffusion to make them matter.
    let settings = SpatialSettings {
        halation_size: 2.0,
        diffusion_size: 2.0,
        ..settings
    };
    let f = film("kodak_portra_400", Output::DisplayP3, settings);
    let (data, plan) = (colour_data(f.baked()), plan_data(&f, frame));
    let whole = FRect::frame((frame.width, frame.height));
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let full = run_tile_gpu(&dev.gpu, &data, &plan, whole);
        println!("halos {:?}", halos(&plan));
        for size in [(97, 61), (301, 200)] {
            let tiled = assemble(&dev.gpu, &data, &plan, whole, size);
            assert!(
                tiled
                    .iter()
                    .zip(&full)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "tiles of {size:?} differ from the whole frame"
            );
        }
        let again = run_tile_gpu(&dev.gpu, &data, &plan, whole);
        assert!(
            again
                .iter()
                .zip(&full)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "a repeat differs"
        );
        if let Some(plain) = start_unpoisoned(StoragePreference::Auto) {
            let np = run_tile_gpu(&plain, &data, &plan, whole);
            assert!(
                np.iter()
                    .zip(&full)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "a run without poison differs"
            );
        }
        println!("tiles 97x61 and 301x200, a repeat and a run without poison: bit for bit");
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_far_corner_of_an_8658_by_5772_frame_is_the_same_for_any_tile() {
    let frame = Frame {
        width: 8658,
        height: 5772,
    };
    let settings = SpatialSettings {
        grain_size: 2.0,
        grain_seed: 3,
        ..SpatialSettings::default()
    };
    let f = film("kodak_portra_400", Output::DisplayP3, settings);
    let (data, plan) = (colour_data(f.baked()), plan_data(&f, frame));
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let small = FRect {
            x0: 8618,
            y0: 5739,
            x1: 8658,
            y1: 5772,
        };
        let big = FRect {
            x0: 8447,
            y0: 5622,
            x1: 8658,
            y1: 5772,
        };
        let (a, b) = (
            run_tile_gpu(&dev.gpu, &data, &plan, small),
            run_tile_gpu(&dev.gpu, &data, &plan, big),
        );
        let (sw, bw, bh) = (40usize, 211usize, 150usize);
        for c in 0..3 {
            for y in 0..33 {
                for x in 0..40 {
                    let (bx, by) = (x + (8618 - 8447) as usize, y + (5739 - 5622) as usize);
                    assert_eq!(
                        a[c * sw * 33 + y * sw + x].to_bits(),
                        b[c * bw * bh + by * bw + bx].to_bits(),
                        "({x},{y}) c{c}"
                    );
                }
            }
        }
        // A tile in the middle of the width against a larger one.
        let t1 = FRect {
            x0: 8000,
            y0: 3000,
            x1: 8060,
            y1: 3050,
        };
        let t2 = FRect {
            x0: 7900,
            y0: 2950,
            x1: 8100,
            y1: 3100,
        };
        let (a, b) = (
            run_tile_gpu(&dev.gpu, &data, &plan, t1),
            run_tile_gpu(&dev.gpu, &data, &plan, t2),
        );
        for c in 0..3 {
            for y in 0..50 {
                for x in 0..60 {
                    assert_eq!(
                        a[c * 60 * 50 + y * 60 + x].to_bits(),
                        b[c * 200 * 150 + (y + 50) * 200 + x + 100].to_bits()
                    );
                }
            }
        }
        println!(
            "far corner and x = 8000: bit for bit (halos {:?})",
            halos(&plan)
        );
    }
    assert!(seen || gpus().is_empty());
}
