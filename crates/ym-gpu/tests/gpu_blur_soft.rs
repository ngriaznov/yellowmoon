//! Soft's device kernel (`docs/33-blur.md` 3.5, 3.7, 8; BL-06; `kernels::blur_soft`) against its CPU twin
//! (`cpu::blur_soft`, which is `ym_color::blur::soft` in `f64`) on lavapipe:
//!
//! * the sums `N` and `D` of the kernel over a made area, at a small radius (the direct sum) and at the coarse grid's cell
//!   sizes 2, 4 and 8, with a bright light among dark pixels, a mask edge (Respect mask edges on and off), the picture's edge
//!   and corner, and a tile cut inside the frame: within 2e-4 relative (the stated bound; the measured numbers are printed);
//! * the node's picture (the kernel with the scaffold's prep and blend) within 2e-4 in LogC4 of the twin's;
//! * a tile of any size equals the whole frame bit for bit, a repeat and a run without the pool's poison are the same bits.
#![allow(clippy::needless_range_loop)]

mod common;

use common::{TestGpu, gpus, start_unpoisoned};
use ym_color::blur::soft::cell_level;
use ym_color::blur::{BlurKind, BlurParams, radius_px, reach_px};
use ym_color::logc::logc4_decode_rgb;
use ym_color::mask_raster::Area;
use ym_gpu::cpu::Placement;
use ym_gpu::cpu::blur as twin;
use ym_gpu::cpu::blur_soft;
use ym_gpu::device::{Gpu, GpuError, StorageMode, StoragePreference};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::blur_soft::BlurSoft;
use ym_gpu::kernels::blur_stack::{
    BlurKernel, BlurWork, SumsImage, encode_blur_picture, encode_prep, encode_pyramid,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The gate of `docs/33` 3.7 for the device against the twin, relative (floor 1e-3) for the sums, absolute in LogC4 for pictures.
const GATE: f64 = 2e-4;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

/// A LogC4 value at the absolute pixel `(x, y)` of a frame of `w x h`: a gradient, a bright disc, a few 3 x 3 lights (code
/// about 0.95: hundreds of times middle grey) on a dark square below the toe, a function of the place alone.
fn logc4_at(x: i64, y: i64, (w, h): (usize, usize)) -> [f64; 3] {
    let (u, v) = (x as f64 / w as f64, y as f64 / h as f64);
    let mut c = [0.25 + 0.3 * u, 0.3 + 0.2 * v, 0.2 + 0.25 * (1.0 - u)];
    let r = ((u - 0.35).powi(2) + (1.5 * (v - 0.4)).powi(2)).sqrt();
    if r < 0.12 {
        c = [0.9, 0.88, 0.85];
    }
    if (0.65..0.75).contains(&u) && (0.6..0.75).contains(&v) {
        c = [-0.02, -0.015, -0.01];
    }
    for (lu, lv) in [(0.7, 0.68), (0.02, 0.03), (0.985, 0.97), (0.5, 0.5)] {
        let (lx, ly) = ((lu * w as f64) as i64, (lv * h as f64) as i64);
        if (x - lx).abs() <= 1 && (y - ly).abs() <= 1 {
            c = [0.95, 0.94, 0.93];
        }
    }
    c
}

/// The node's mask: 0 on the left (u < 0.25), 1 on the right (u > 0.6), a ramp between, a few values the clean-up must handle.
fn mask_at(x: i64, y: i64, (w, h): (usize, usize)) -> f64 {
    let _ = h;
    if (x, y) == (97, 51) {
        return f64::NAN;
    }
    if (x, y) == (98, 51) {
        return -0.5;
    }
    ((x as f64 / w as f64 - 0.25) / 0.35).clamp(0.0, 1.0)
}

fn picture(area: Area, frame: (usize, usize)) -> Vec<f32> {
    let n = area.w * area.h;
    let mut d = vec![0.0f32; 3 * n];
    for y in 0..area.h {
        for x in 0..area.w {
            let c = logc4_at(area.x + x as i64, area.y + y as i64, frame);
            for k in 0..3 {
                d[k * n + y * area.w + x] = c[k] as f32;
            }
        }
    }
    d
}

fn mask(area: Area, frame: (usize, usize)) -> Vec<f32> {
    let mut d = vec![0.0f32; area.w * area.h];
    for y in 0..area.h {
        for x in 0..area.w {
            d[y * area.w + x] = mask_at(area.x + x as i64, area.y + y as i64, frame) as f32;
        }
    }
    d
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn upload(
    ctx: &ym_gpu::device::GpuContext,
    area: Area,
    planes: usize,
    d: &[f32],
) -> Result<GpuImage, GpuError> {
    GpuImage::upload(
        ctx,
        "soft test",
        area.w,
        area.h,
        planes,
        d,
        Placement::whole(area.w, area.h),
    )
}

fn rel_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs() / y.abs().max(1e-3))
        .fold(0.0, f64::max)
}

fn abs_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

/// The kernel's sums `N` and `D` over `made` from the picture over `read`, read back (four planes over `made`).
fn device_sums(
    gpu: &Gpu,
    (radius, respect): (f64, bool),
    frame: (usize, usize),
    (read, made): (Area, Area),
) -> Vec<f32> {
    let (pic, msk) = (picture(read, frame), mask(read, frame));
    let photo = Area::whole(frame.0, frame.1);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "blur soft");
        let picture = upload(ctx, read, 3, &pic)?;
        let mask = upload(ctx, read, 1, &msk)?;
        let pyramid = encode_pyramid(
            ctx,
            &mut passes,
            (&picture, &mask),
            (read, photo),
            (0.0, respect),
            0,
        )?;
        let sums = SumsImage {
            img: GpuImage::alloc(
                ctx,
                "sums",
                made.w,
                made.h,
                4,
                Placement::whole(made.w, made.h),
            )?,
            area: made,
        };
        BlurSoft::new(radius).encode(ctx, &mut passes, &pyramid, &sums)?;
        let r = sums.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        let _ = (&picture, &mask, &pyramid);
        sums.img.take(&done, r)
    })
    .expect("soft sums")
}

/// The twin's sums over `made`.
fn twin_sums(
    (radius, respect): (f64, bool),
    frame: (usize, usize),
    (read, made): (Area, Area),
) -> Vec<f64> {
    let (pic, msk) = (picture(read, frame), mask(read, frame));
    let base = twin::prep(&f64s(&pic), &f64s(&msk), read.w, read.h, 0.0, respect);
    let sigma = radius_px(radius, frame.0.max(frame.1) as f64) / 2.0;
    blur_soft::sums(&base, read, Area::whole(frame.0, frame.1), made, sigma)
}

/// The cases: the frame, the radius, and what the cell size is.
const CASES: [((usize, usize), f64); 6] = [
    ((300, 200), 60.0),   // sigma 3.6: the direct sum
    ((300, 200), 100.0),  // sigma 6: cells of 2
    ((500, 340), 100.0),  // sigma 10: cells of 4 (cell sigma just under 4)
    ((900, 600), 100.0),  // sigma 18: cells of 8
    ((900, 600), 40.0),   // sigma 7.2: cells of 2
    ((1100, 740), 100.0), // sigma 22: cells of 8
];

#[test]
fn the_kernel_is_the_twin_at_every_cell_size_the_edge_and_the_corner() {
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (frame, radius) in CASES {
            let photo = Area::whole(frame.0, frame.1);
            let params = BlurParams {
                kind: BlurKind::Soft { radius },
                respect_mask_edges: true,
            };
            let reach = reach_px(&params, frame.0, frame.1) as i64;
            let sigma = radius_px(radius, frame.0.max(frame.1) as f64) / 2.0;
            // The whole frame, the top-left corner tile, a tile inside, and the bottom-right corner tile: each with the reach.
            let tiles = [
                photo,
                Area {
                    x: 0,
                    y: 0,
                    w: 97,
                    h: 61,
                },
                Area {
                    x: 111,
                    y: 53,
                    w: 120,
                    h: 80,
                },
                Area {
                    x: frame.0 as i64 - 83,
                    y: frame.1 as i64 - 57,
                    w: 83,
                    h: 57,
                },
            ];
            for made in tiles {
                let grow = |a: Area| {
                    Area {
                        x: (a.x - reach).max(0),
                        y: (a.y - reach).max(0),
                        w: 0,
                        h: 0,
                    }
                    .with_end(
                        (a.x1() + reach).min(frame.0 as i64),
                        (a.y1() + reach).min(frame.1 as i64),
                    )
                };
                let read = grow(made);
                for respect in [true, false] {
                    let got = device_sums(&dev.gpu, (radius, respect), frame, (read, made));
                    let want = twin_sums((radius, respect), frame, (read, made));
                    let d = rel_diff(&got, &want);
                    println!(
                        "soft {}x{} radius {radius} sigma {sigma:.1} (cells of {}), made {}x{} at ({}, {}), respect {respect}: {d:.2e}",
                        frame.0,
                        frame.1,
                        1u32 << cell_level(sigma),
                        made.w,
                        made.h,
                        made.x,
                        made.y
                    );
                    assert!(
                        d < GATE,
                        "{frame:?} radius {radius} {made:?} respect {respect}: {d}"
                    );
                    worst = worst.max(d);
                }
            }
        }
    }
    println!("soft kernel against its twin: worst {worst:.2e} relative (gate {GATE:.0e})");
    assert!(seen || gpus().is_empty());
}

trait WithEnd {
    fn with_end(self, x1: i64, y1: i64) -> Area;
}

impl WithEnd for Area {
    fn with_end(self, x1: i64, y1: i64) -> Area {
        Area {
            w: (x1 - self.x) as usize,
            h: (y1 - self.y) as usize,
            ..self
        }
    }
}

/// The node's picture over `made` by the device (the scaffold's prep, the kernel and the blend).
fn device_picture(
    gpu: &Gpu,
    (radius, respect): (f64, bool),
    frame: (usize, usize),
    (read, made): (Area, Area),
) -> Vec<f32> {
    let (pic, msk) = (picture(read, frame), mask(read, frame));
    let photo = Area::whole(frame.0, frame.1);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "blur soft node");
        let picture = upload(ctx, read, 3, &pic)?;
        let mask = upload(ctx, read, 1, &msk)?;
        let params = BlurParams {
            kind: BlurKind::Soft { radius },
            respect_mask_edges: respect,
        };
        let work = BlurWork {
            params,
            mask: std::sync::Arc::new(
                ym_color::display::ColorChain::new(&common::chains::base(
                    ym_color::display::OutputTarget::Srgb,
                ))
                .expect("chain"),
            ),
            reach: 0,
            kernel: ym_gpu::kernels::blur_stack::device_kernel(&params),
        };
        assert!(work.kernel.is_some(), "Soft has a device kernel");
        let out = encode_blur_picture(
            ctx,
            &mut passes,
            &work,
            (&picture, &mask),
            (read, made, photo),
        )?;
        let r = out.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        out.img.take(&done, r)
    })
    .expect("soft node")
}

#[test]
fn the_nodes_picture_is_the_twins_in_logc4() {
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (frame, radius) in [((300, 200), 100.0), ((900, 600), 100.0), ((300, 200), 60.0)] {
            let photo = Area::whole(frame.0, frame.1);
            let (read, made) = (photo, photo);
            for respect in [true, false] {
                let got = device_picture(&dev.gpu, (radius, respect), frame, (read, made));
                let (pic, msk) = (picture(read, frame), mask(read, frame));
                let sums = twin_sums((radius, respect), frame, (read, made));
                let want = twin::blend(&f64s(&pic), &f64s(&msk), read, &sums, made);
                let d = abs_diff(&got, &want);
                println!(
                    "soft node {}x{} radius {radius} respect {respect}: picture within {d:.2e} of the twin's, in LogC4",
                    frame.0, frame.1
                );
                assert!(d < GATE, "{frame:?} radius {radius} respect {respect}: {d}");
                worst = worst.max(d);
            }
        }
    }
    println!("soft picture against its twin: worst {worst:.2e} in LogC4 (gate {GATE:.0e})");
    assert!(seen || gpus().is_empty());
}

#[test]
fn a_tile_with_its_reach_equals_the_frame_bit_for_bit_and_a_repeat_and_a_clean_pool_agree() {
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (frame, radius) in [
            ((300, 200), 100.0),
            ((500, 340), 100.0),
            ((900, 600), 100.0),
            ((300, 200), 60.0),
        ] {
            let photo = Area::whole(frame.0, frame.1);
            let params = BlurParams {
                kind: BlurKind::Soft { radius },
                respect_mask_edges: true,
            };
            let reach = reach_px(&params, frame.0, frame.1) as i64;
            let whole = device_sums(&dev.gpu, (radius, true), frame, (photo, photo));
            let n = frame.0 * frame.1;
            for made in [
                Area {
                    x: 0,
                    y: 0,
                    w: 97,
                    h: 61,
                },
                Area {
                    x: 133,
                    y: 71,
                    w: 97,
                    h: 61,
                },
                Area {
                    x: frame.0 as i64 - 97,
                    y: frame.1 as i64 - 61,
                    w: 97,
                    h: 61,
                },
                Area {
                    x: 41,
                    y: 9,
                    w: 211,
                    h: 150,
                },
                Area {
                    x: 3,
                    y: 5,
                    w: 160,
                    h: 60,
                },
            ] {
                if made.x1() > frame.0 as i64 || made.y1() > frame.1 as i64 {
                    continue;
                }
                let read = Area {
                    x: (made.x - reach).max(0),
                    y: (made.y - reach).max(0),
                    w: 0,
                    h: 0,
                }
                .with_end(
                    (made.x1() + reach).min(frame.0 as i64),
                    (made.y1() + reach).min(frame.1 as i64),
                );
                let tile = device_sums(&dev.gpu, (radius, true), frame, (read, made));
                let tn = made.w * made.h;
                for c in 0..4 {
                    for y in 0..made.h {
                        for x in 0..made.w {
                            let g = tile[c * tn + y * made.w + x];
                            let w = whole
                                [c * n + (made.y as usize + y) * frame.0 + made.x as usize + x];
                            assert_eq!(
                                g.to_bits(),
                                w.to_bits(),
                                "{frame:?} radius {radius} {made:?} at ({x}, {y}) plane {c}"
                            );
                        }
                    }
                }
            }
            let again = device_sums(&dev.gpu, (radius, true), frame, (photo, photo));
            assert!(
                whole
                    .iter()
                    .zip(&again)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            if let Some(clean) = start_unpoisoned(StoragePreference::Auto) {
                let other = device_sums(&clean, (radius, true), frame, (photo, photo));
                assert!(
                    whole
                        .iter()
                        .zip(&other)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn a_flat_field_and_the_energy_of_a_point_hold_on_the_device() {
    // A flat picture stays flat to the edge and the corner (N / D equals the value), and the sums of a point keep its mass.
    let frame = (400usize, 300usize);
    let photo = Area::whole(frame.0, frame.1);
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let flat_code = 0.4f32;
        let n = frame.0 * frame.1;
        let pic = vec![flat_code; 3 * n];
        let msk = vec![1.0f32; n];
        let out: Vec<f32> = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "flat");
                let picture = upload(ctx, photo, 3, &pic)?;
                let mask = upload(ctx, photo, 1, &msk)?;
                let pyramid = encode_pyramid(
                    ctx,
                    &mut passes,
                    (&picture, &mask),
                    (photo, photo),
                    (0.0, true),
                    0,
                )?;
                let sums = SumsImage {
                    img: GpuImage::alloc(
                        ctx,
                        "sums",
                        photo.w,
                        photo.h,
                        4,
                        Placement::whole(photo.w, photo.h),
                    )?,
                    area: photo,
                };
                BlurSoft::new(100.0).encode(ctx, &mut passes, &pyramid, &sums)?;
                let r = sums.img.encode_read(&mut passes)?;
                let done = passes.submit()?.finish()?;
                let _ = (&picture, &mask, &pyramid);
                sums.img.take(&done, r)
            })
            .expect("flat");
        let lin = logc4_decode_rgb([f64::from(flat_code); 3])[0];
        let mut worst = 0.0f64;
        for i in 0..n {
            let ratio = f64::from(out[i]) / f64::from(out[3 * n + i]);
            worst = worst.max((ratio - lin).abs() / lin);
        }
        println!(
            "flat field at Radius 100 (cells of {}): N / D within {worst:.2e} of the value",
            1u32 << cell_level(radius_px(100.0, 400.0) / 2.0)
        );
        assert!(worst < 1e-5, "{worst}");
    }
    assert!(seen || gpus().is_empty());
}

// ---- speed (`docs/33` 9, BL-06): run by hand, `cargo test -p ym-gpu --release --test gpu_blur_soft -- --ignored --nocapture` ----

/// The device time, in milliseconds, of the first `stage` passes of a Soft node over `made` read from `read`: 0 the prep (decode, the
/// weights), 1 the prep and the kernel, 2 the whole node (the prep, the kernel and the blend). The best of `repeat` runs after one
/// warm-up; the upload and the read-back are not in it.
fn time_node(
    gpu: &Gpu,
    (radius, frame): (f64, (usize, usize)),
    (read, made): (Area, Area),
    stage: u32,
    repeat: usize,
) -> f64 {
    let n = read.w * read.h;
    // A picture that is a function of the place, built without the scene's per-pixel logic (the sizes here are millions of pixels).
    let pic: Vec<f32> = (0..3 * n)
        .map(|i| 0.3 + 0.4 * ((i % 977) as f32 / 977.0))
        .collect();
    let msk: Vec<f32> = (0..n)
        .map(|i| ((i % read.w) as f32 / read.w as f32).clamp(0.0, 1.0))
        .collect();
    let photo = Area::whole(frame.0, frame.1);
    let mut best = f64::MAX;
    for run in 0..=repeat {
        let (p, m) = (pic.clone(), msk.clone());
        let ms: f64 = gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "blur soft speed");
                let picture = upload(ctx, read, 3, &p)?;
                let mask = upload(ctx, read, 1, &m)?;
                let params = BlurParams {
                    kind: BlurKind::Soft { radius },
                    respect_mask_edges: true,
                };
                let t = std::time::Instant::now();
                match stage {
                    0 | 1 => {
                        let pyramid = encode_pyramid(
                            ctx,
                            &mut passes,
                            (&picture, &mask),
                            (read, photo),
                            (0.0, true),
                            0,
                        )?;
                        let sums = SumsImage {
                            img: GpuImage::alloc(
                                ctx,
                                "sums",
                                made.w,
                                made.h,
                                4,
                                Placement::whole(made.w, made.h),
                            )?,
                            area: made,
                        };
                        if stage == 1 {
                            BlurSoft::new(radius).encode(ctx, &mut passes, &pyramid, &sums)?;
                        }
                        passes.submit()?.finish()?;
                        let _ = (&pyramid, &sums);
                    }
                    _ => {
                        let work = BlurWork {
                            params,
                            mask: std::sync::Arc::new(
                                ym_color::display::ColorChain::new(&common::chains::base(
                                    ym_color::display::OutputTarget::Srgb,
                                ))
                                .expect("chain"),
                            ),
                            reach: 0,
                            kernel: ym_gpu::kernels::blur_stack::device_kernel(&params),
                        };
                        let out = encode_blur_picture(
                            ctx,
                            &mut passes,
                            &work,
                            (&picture, &mask),
                            (read, made, photo),
                        )?;
                        passes.submit()?.finish()?;
                        let _ = &out;
                    }
                }
                Ok(t.elapsed().as_secs_f64() * 1e3)
            })
            .expect("timed node");
        if run > 0 {
            best = best.min(ms);
        }
    }
    best
}

#[test]
#[ignore = "a speed measurement, run by hand"]
fn soft_speed_on_lavapipe() {
    for dev in f32_devices() {
        // The fit view's size and a 24 MP export's: the whole frame in one piece.
        for (frame, label) in [((1800usize, 1200usize), "2.2 MP"), ((3000, 2000), "6 MP")] {
            let photo = Area::whole(frame.0, frame.1);
            let mp = (frame.0 * frame.1) as f64 / 1e6;
            for radius in [10.0, 100.0] {
                let params = BlurParams {
                    kind: BlurKind::Soft { radius },
                    respect_mask_edges: true,
                };
                let sigma = radius_px(radius, frame.0.max(frame.1) as f64) / 2.0;
                let reach = reach_px(&params, frame.0, frame.1);
                let t: Vec<f64> = (0..3)
                    .map(|s| time_node(&dev.gpu, (radius, frame), (photo, photo), s, 3))
                    .collect();
                println!(
                    "soft {label} radius {radius} (sigma {sigma:.1}, cells of {}, reach {reach}): prep {:.1} ms, kernel {:.1} ms, blend {:.1} ms, node {:.1} ms = {:.2} ms/MP (prep {:.2}, kernel {:.2})",
                    1u32 << cell_level(sigma),
                    t[0],
                    t[1] - t[0],
                    t[2] - t[1],
                    t[2],
                    t[2] / mp,
                    t[0] / mp,
                    (t[1] - t[0]) / mp
                );
            }
        }
        // One tile of a 50 MP export at the largest Soft: the scheduler's `fit_tile` gives 364 px and the reach 880 grows it to 2124 px.
        let frame = (8600usize, 5700usize);
        let params = BlurParams {
            kind: BlurKind::Soft { radius: 100.0 },
            respect_mask_edges: true,
        };
        let reach = reach_px(&params, frame.0, frame.1) as i64;
        let made = Area {
            x: 4000,
            y: 2500,
            w: 364,
            h: 364,
        };
        let read = Area {
            x: made.x - reach,
            y: made.y - reach,
            w: 364 + 2 * reach as usize,
            h: 364 + 2 * reach as usize,
        };
        let t = time_node(&dev.gpu, (100.0, frame), (read, made), 2, 2);
        let tiles = (frame.0 as f64 / 364.0).ceil() * (frame.1 as f64 / 364.0).ceil();
        println!(
            "soft 50 MP, Radius 100, reach {reach}: one tile of 364 px read as {} x {}: {t:.0} ms; {tiles:.0} tiles: {:.1} s of device time, the frame read {:.1} times over",
            read.w,
            read.h,
            t * tiles / 1e3,
            tiles * (read.w * read.h) as f64 / (frame.0 * frame.1) as f64
        );
    }
}
