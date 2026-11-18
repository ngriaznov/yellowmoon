//! Motion's device kernel (`docs/33-blur.md` 3.5, 3.7, 8; BL-08; `kernels::blur_motion`) against its CPU twin
//! (`cpu::blur_motion`, which is `ym_color::blur::motion` in `f64`) on lavapipe:
//!
//! * the sums `N` and `D` of the kernel over a made area, at lengths from under a pixel to Length 100 and at angles along both
//!   axes, the diagonal and odd ones, with a bright light among dark pixels, a mask edge (Respect mask edges on and off), the
//!   picture's edge and corner, and a tile cut inside the frame: within 2e-4 relative (the stated bound; the measured numbers
//!   are printed);
//! * the node's picture (the kernel with the scaffold's prep and blend) within 2e-4 in LogC4 of the twin's;
//! * a tile of any size equals the whole frame bit for bit, a repeat and a run without the pool's poison are the same bits.
#![allow(clippy::needless_range_loop)]

mod common;

use common::{TestGpu, gpus, start_unpoisoned};
use ym_color::blur::{BlurKind, BlurParams, length_px, reach_px};
use ym_color::logc::logc4_decode_rgb;
use ym_color::mask_raster::Area;
use ym_gpu::cpu::Placement;
use ym_gpu::cpu::blur as twin;
use ym_gpu::cpu::blur_motion;
use ym_gpu::device::{Gpu, GpuError, StorageMode, StoragePreference};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::blur_motion::BlurMotion;
use ym_gpu::kernels::blur_stack::{
    BlurKernel, BlurWork, SumsImage, encode_blur_picture, encode_pyramid,
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
        "motion test",
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


/// The Length and Angle of a case.
type Line = (f64, f64);

fn params_of((length, angle_deg): Line, respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Motion { length, angle_deg },
        respect_mask_edges: respect,
    }
}

fn grow(a: Area, reach: i64, frame: (usize, usize)) -> Area {
    let x0 = (a.x - reach).max(0);
    let y0 = (a.y - reach).max(0);
    let x1 = (a.x1() + reach).min(frame.0 as i64);
    let y1 = (a.y1() + reach).min(frame.1 as i64);
    Area {
        x: x0,
        y: y0,
        w: (x1 - x0) as usize,
        h: (y1 - y0) as usize,
    }
}

/// The kernel's sums `N` and `D` over `made` from the picture over `read`, read back (four planes over `made`).
fn device_sums(
    gpu: &Gpu,
    (line, respect): (Line, bool),
    frame: (usize, usize),
    (read, made): (Area, Area),
) -> Vec<f32> {
    let (pic, msk) = (picture(read, frame), mask(read, frame));
    let photo = Area::whole(frame.0, frame.1);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "blur motion");
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
        BlurMotion::new(line.0, line.1).encode(ctx, &mut passes, &pyramid, &sums)?;
        let r = sums.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        let _ = (&picture, &mask, &pyramid);
        sums.img.take(&done, r)
    })
    .expect("motion sums")
}

/// The twin's sums over `made`.
fn twin_sums(
    (line, respect): (Line, bool),
    frame: (usize, usize),
    (read, made): (Area, Area),
) -> Vec<f64> {
    let (pic, msk) = (picture(read, frame), mask(read, frame));
    let base = twin::prep(&f64s(&pic), &f64s(&msk), read.w, read.h, 0.0, respect);
    let l = length_px(line.0, frame.0.max(frame.1) as f64);
    blur_motion::sums(&base, read, made, l, line.1)
}

/// The cases: the frame, Length and Angle.
const CASES: [((usize, usize), Line); 7] = [
    ((300, 200), (3.0, 0.0)),    // under a pixel
    ((300, 200), (100.0, 0.0)),  // 24 px along x
    ((300, 200), (100.0, 90.0)), // along y
    ((300, 200), (60.0, 45.0)),  // the diagonal
    ((500, 340), (100.0, 137.0)),
    ((900, 600), (100.0, 20.0)), // 72 px
    ((900, 600), (35.0, -33.0)),
];

#[test]
fn the_kernel_is_the_twin_at_every_length_and_angle_the_edge_and_the_corner() {
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (frame, line) in CASES {
            let photo = Area::whole(frame.0, frame.1);
            let reach = reach_px(&params_of(line, true), frame.0, frame.1) as i64;
            let l = length_px(line.0, frame.0.max(frame.1) as f64);
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
                let read = grow(made, reach, frame);
                for respect in [true, false] {
                    let got = device_sums(&dev.gpu, (line, respect), frame, (read, made));
                    let want = twin_sums((line, respect), frame, (read, made));
                    let d = rel_diff(&got, &want);
                    println!(
                        "motion {}x{} length {} ({l:.1} px) angle {}, made {}x{} at ({}, {}), respect {respect}: {d:.2e}",
                        frame.0, frame.1, line.0, line.1, made.w, made.h, made.x, made.y
                    );
                    assert!(
                        d < GATE,
                        "{frame:?} {line:?} {made:?} respect {respect}: {d}"
                    );
                    worst = worst.max(d);
                }
            }
        }
    }
    println!("motion kernel against its twin: worst {worst:.2e} relative (gate {GATE:.0e})");
    assert!(seen || gpus().is_empty());
}

/// The node's picture over `made` by the device (the scaffold's prep, the kernel and the blend).
fn device_picture(
    gpu: &Gpu,
    (line, respect): (Line, bool),
    frame: (usize, usize),
    (read, made): (Area, Area),
) -> Vec<f32> {
    let (pic, msk) = (picture(read, frame), mask(read, frame));
    let photo = Area::whole(frame.0, frame.1);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "blur motion node");
        let picture = upload(ctx, read, 3, &pic)?;
        let mask = upload(ctx, read, 1, &msk)?;
        let params = params_of(line, respect);
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
        assert!(work.kernel.is_some(), "Motion has a device kernel");
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
    .expect("motion node")
}

#[test]
fn the_nodes_picture_is_the_twins_in_logc4() {
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (frame, line) in [
            ((300, 200), (100.0, 20.0)),
            ((900, 600), (100.0, 90.0)),
            ((300, 200), (60.0, 45.0)),
        ] {
            let photo = Area::whole(frame.0, frame.1);
            for respect in [true, false] {
                let got = device_picture(&dev.gpu, (line, respect), frame, (photo, photo));
                let (pic, msk) = (picture(photo, frame), mask(photo, frame));
                let sums = twin_sums((line, respect), frame, (photo, photo));
                let want = twin::blend(&f64s(&pic), &f64s(&msk), photo, &sums, photo);
                let d = abs_diff(&got, &want);
                println!(
                    "motion node {}x{} {line:?} respect {respect}: picture within {d:.2e} of the twin's, in LogC4",
                    frame.0, frame.1
                );
                assert!(d < GATE, "{frame:?} {line:?} respect {respect}: {d}");
                worst = worst.max(d);
            }
        }
    }
    println!("motion picture against its twin: worst {worst:.2e} in LogC4 (gate {GATE:.0e})");
    assert!(seen || gpus().is_empty());
}

#[test]
fn a_tile_with_its_reach_equals_the_frame_bit_for_bit_and_a_repeat_and_a_clean_pool_agree() {
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (frame, line) in [
            ((300, 200), (100.0, 20.0)),
            ((500, 340), (100.0, 90.0)),
            ((300, 200), (60.0, 135.0)),
        ] {
            let photo = Area::whole(frame.0, frame.1);
            let reach = reach_px(&params_of(line, true), frame.0, frame.1) as i64;
            let whole = device_sums(&dev.gpu, (line, true), frame, (photo, photo));
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
            ] {
                if made.x1() > frame.0 as i64 || made.y1() > frame.1 as i64 {
                    continue;
                }
                let read = grow(made, reach, frame);
                let tile = device_sums(&dev.gpu, (line, true), frame, (read, made));
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
                                "{frame:?} {line:?} {made:?} at ({x}, {y}) plane {c}"
                            );
                        }
                    }
                }
            }
            let again = device_sums(&dev.gpu, (line, true), frame, (photo, photo));
            assert!(
                whole
                    .iter()
                    .zip(&again)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            if let Some(clean) = start_unpoisoned(StoragePreference::Auto) {
                let other = device_sums(&clean, (line, true), frame, (photo, photo));
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
fn a_flat_field_holds_on_the_device() {
    // A flat picture stays flat to the edge and the corner: N / D equals the value.
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
                BlurMotion::new(100.0, 33.0).encode(ctx, &mut passes, &pyramid, &sums)?;
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
        println!("flat field at Length 100: N / D within {worst:.2e} of the value");
        assert!(worst < 1e-5, "{worst}");
    }
    assert!(seen || gpus().is_empty());
}

// ---- speed (`docs/33` 9, BL-08): run by hand, `cargo test -p ym-gpu --release --test gpu_blur_motion -- --ignored --nocapture` ----

/// The device time, in milliseconds, of the first `stage` passes of a Motion node over the whole frame: 0 the prep, 1 the prep and the
/// kernel, 2 the whole node. The best of `repeat` runs after one warm-up; the upload and the read-back are not in it.
fn time_node(gpu: &Gpu, (line, frame): (Line, (usize, usize)), stage: u32, repeat: usize) -> f64 {
    let photo = Area::whole(frame.0, frame.1);
    let n = frame.0 * frame.1;
    let pic: Vec<f32> = (0..3 * n)
        .map(|i| 0.3 + 0.4 * ((i % 977) as f32 / 977.0))
        .collect();
    let msk: Vec<f32> = (0..n)
        .map(|i| ((i % frame.0) as f32 / frame.0 as f32).clamp(0.0, 1.0))
        .collect();
    let mut best = f64::MAX;
    for run in 0..=repeat {
        let (p, m) = (pic.clone(), msk.clone());
        let ms: f64 = gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "blur motion speed");
                let picture = upload(ctx, photo, 3, &p)?;
                let mask = upload(ctx, photo, 1, &m)?;
                let params = params_of(line, true);
                let t = std::time::Instant::now();
                if stage <= 1 {
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
                    if stage == 1 {
                        BlurMotion::new(line.0, line.1).encode(ctx, &mut passes, &pyramid, &sums)?;
                    }
                    passes.submit()?.finish()?;
                    let _ = (&pyramid, &sums);
                } else {
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
                        (photo, photo, photo),
                    )?;
                    passes.submit()?.finish()?;
                    let _ = &out;
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
fn motion_speed_on_lavapipe() {
    for dev in f32_devices() {
        for (frame, label) in [((1800usize, 1200usize), "2.2 MP"), ((3000, 2000), "6 MP")] {
            let mp = (frame.0 * frame.1) as f64 / 1e6;
            for (length, angle) in [(10.0, 0.0), (10.0, 30.0), (100.0, 0.0), (100.0, 30.0), (100.0, 45.0)] {
                let l = length_px(length, frame.0.max(frame.1) as f64);
                let taps = ym_color::blur::motion::taps(l, angle).len();
                let t: Vec<f64> = (0..3)
                    .map(|s| time_node(&dev.gpu, ((length, angle), frame), s, 2))
                    .collect();
                println!(
                    "motion {label} length {length} ({l:.1} px, {taps} taps) angle {angle}: prep {:.1} ms, kernel {:.1} ms, blend {:.1} ms, node {:.1} ms = {:.2} ms/MP (kernel {:.2})",
                    t[0],
                    t[1] - t[0],
                    t[2] - t[1],
                    t[2],
                    t[2] / mp,
                    (t[1] - t[0]) / mp
                );
            }
        }
    }
}
