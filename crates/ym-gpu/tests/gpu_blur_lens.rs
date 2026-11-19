//! The Lens type of a Blur node on the device (`docs/33-blur.md` section 3.5, BL-07b; `kernels::blur_lens`) against its CPU
//! references: the sums against the twin (`cpu::blur_lens::sums`, f64) and the node's picture against the CPU's fast method
//! (`ym_color::blur::apply_fast`, f64), on lavapipe. Tiles of the node equal the whole made area bit for bit.
//!
//! The stated bound is 2e-4 in LogC4 for the picture (`docs/33` section 3.7) and 2e-4 relative for the sums.
#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use common::chains;
use common::{TestGpu, gpus};
use ym_color::blur::{self, BlurKind, BlurParams, Frame, radius_px};
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, OutputTarget};
use ym_color::logc::logc4_encode_rgb;
use ym_color::mask_raster::Area;
use ym_gpu::cpu::Placement;
use ym_gpu::cpu::blur as twin;
use ym_gpu::cpu::blur_lens;
use ym_gpu::device::{GpuContext, StorageMode};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::blur_lens::LensKernel;
use ym_gpu::kernels::blur_stack::{
    BlurKernel, BlurWork, SumsImage, device_kernel, encode_blur_picture, encode_pyramid,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The gate of `docs/33` section 3.7.
const GATE: f64 = 2e-4;

const FRAME: (usize, usize) = (1200, 800);

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

/// A LogC4 value at `(x, y)`: a gradient, a wide bright disc, a dark square below the toe, and four bright lights (400 x middle
/// grey) among dark pixels, a function of the place alone so that every window agrees.
fn logc4_at(x: i64, y: i64) -> [f64; 3] {
    let (u, v) = (x as f64 / FRAME.0 as f64, y as f64 / FRAME.1 as f64);
    let mut c = [0.25 + 0.3 * u, 0.3 + 0.2 * v, 0.2 + 0.25 * (1.0 - u)];
    let r = ((u - 0.35).powi(2) + (1.5 * (v - 0.4)).powi(2)).sqrt();
    if r < 0.12 {
        c = [0.9, 0.88, 0.85];
    }
    if (800..880).contains(&x) && (480..560).contains(&y) {
        c = [-0.02, -0.015, -0.01];
    }
    for (lx, ly) in [(600i64, 400i64), (640, 420), (301, 133), (1000, 90)] {
        if (x - lx).abs() < 2 && (y - ly).abs() < 2 {
            c = logc4_encode_rgb([72.0, 70.0, 66.0]);
        }
    }
    c
}

/// The mask: a ramp over the whole width (0.17 at the left edge, 1 at the right), so every window holds fractional values.
fn mask_at(x: i64, _y: i64) -> f64 {
    ((x as f64 + 200.0) / 1200.0).clamp(0.0, 1.0)
}

fn picture(area: Area) -> Vec<f32> {
    let n = area.w * area.h;
    let mut d = vec![0.0f32; 3 * n];
    for y in 0..area.h {
        for x in 0..area.w {
            let c = logc4_at(area.x + x as i64, area.y + y as i64);
            for k in 0..3 {
                d[k * n + y * area.w + x] = c[k] as f32;
            }
        }
    }
    d
}

fn mask(area: Area) -> Vec<f32> {
    let mut d = vec![0.0f32; area.w * area.h];
    for y in 0..area.h {
        for x in 0..area.w {
            d[y * area.w + x] = mask_at(area.x + x as i64, area.y + y as i64) as f32;
        }
    }
    d
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn upload(ctx: &GpuContext, area: Area, planes: usize, d: &[f32]) -> GpuImage {
    GpuImage::upload(
        ctx,
        "lens test",
        area.w,
        area.h,
        planes,
        d,
        Placement::whole(area.w, area.h),
    )
    .expect("upload")
}

fn lens(radius: f64, blades: u32, rotation_deg: f64, highlights: f64, respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Lens {
            radius,
            blades,
            rotation_deg,
            highlights,
        },
        respect_mask_edges: respect,
    }
}

/// The cases (the frame is 1200 px, so `R = 0.48 Radius` px): R = 7.2 px (the exact table), 14.4, 28.8, 48 and 43 px (rows), round
/// and polygon, Highlights, Respect on and off.
fn cases() -> Vec<(&'static str, BlurParams)> {
    vec![
        (
            "R 7.2, the exact table, round",
            lens(15.0, 0, 0.0, 0.0, true),
        ),
        ("R 14.4, hexagon", lens(30.0, 6, 15.0, 0.0, true)),
        ("R 28.8, round, respect off", lens(60.0, 0, 0.0, 0.0, false)),
        (
            "R 48, 7 blades, Highlights 60",
            lens(100.0, 7, 33.0, 60.0, true),
        ),
        (
            "R 43, round, Highlights 100, respect off",
            lens(90.0, 0, 0.0, 100.0, false),
        ),
    ]
}

fn read_of(made: Area, reach: usize) -> Area {
    let frame = Area::whole(FRAME.0, FRAME.1);
    let g = |v: i64| v - reach as i64;
    let x0 = g(made.x).max(frame.x);
    let y0 = g(made.y).max(frame.y);
    let x1 = (made.x1() + reach as i64).min(frame.x1());
    let y1 = (made.y1() + reach as i64).min(frame.y1());
    Area {
        x: x0,
        y: y0,
        w: (x1 - x0) as usize,
        h: (y1 - y0) as usize,
    }
}

fn unused_chain() -> ColorChain {
    ColorChain::new(&chains::base(OutputTarget::Srgb)).expect("chain")
}

/// The picture the node makes over `made`, read back, with the real kernel.
fn node_picture(gpu: &ym_gpu::device::Gpu, params: BlurParams, made: Area) -> Vec<f32> {
    let reach = blur::reach_px(&params, FRAME.0, FRAME.1);
    let read = read_of(made, reach);
    let (pic, msk) = (picture(read), mask(read));
    let frame = Area::whole(FRAME.0, FRAME.1);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "lens node");
        let picture = upload(ctx, read, 3, &pic);
        let mask = upload(ctx, read, 1, &msk);
        let w = BlurWork {
            params,
            mask: Arc::new(unused_chain()),
            reach,
            kernel: device_kernel(&params),
        };
        let out =
            encode_blur_picture(ctx, &mut passes, &w, (&picture, &mask), (read, made, frame))?;
        let r = out.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        out.img.take(&done, r)
    })
    .expect("node picture")
}

/// The CPU's fast method over the same read area, cut to `made`: LogC4 planes.
fn cpu_picture(params: &BlurParams, made: Area) -> Vec<f64> {
    let reach = blur::reach_px(params, FRAME.0, FRAME.1);
    let read = read_of(made, reach);
    let (pic, msk) = (picture(read), mask(read));
    let n = read.w * read.h;
    let v: Vec<Vec3> = (0..n)
        .map(|i| {
            [
                f64::from(pic[i]),
                f64::from(pic[n + i]),
                f64::from(pic[2 * n + i]),
            ]
        })
        .collect();
    let m = f64s(&msk);
    let frame = Frame {
        full_w: FRAME.0,
        full_h: FRAME.1,
        x0: read.x as usize,
        y0: read.y as usize,
    };
    let out = blur::apply_fast(&v, &m, read.w, read.h, params, &frame).expect("cpu");
    let mn = made.w * made.h;
    let mut res = vec![0.0; 3 * mn];
    for y in 0..made.h {
        for x in 0..made.w {
            let r = (made.y - read.y) as usize + y;
            let c = (made.x - read.x) as usize + x;
            let p = out[r * read.w + c];
            for k in 0..3 {
                res[k * mn + y * made.w + x] = p[k];
            }
        }
    }
    res
}

fn abs_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn the_node_picture_is_the_cpu_fast_method_within_the_gate() {
    // The made area: the wide disc's rim, the lights and the mask's ramp, a window of 300 x 200 pixels.
    let made = Area {
        x: 480,
        y: 300,
        w: 300,
        h: 200,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (name, params) in cases() {
            let got = node_picture(&dev.gpu, params, made);
            let want = cpu_picture(&params, made);
            let d = abs_diff(&got, &want);
            println!(
                "{name}: the picture is within {d:.2e} of the CPU's fast method, in LogC4 (gate {GATE:.0e})"
            );
            assert!(d < GATE, "{name}: {d}");
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_node_picture_at_the_frames_edge_and_corner_is_the_cpu_fast_method() {
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for made in [
            Area {
                x: 0,
                y: 0,
                w: 160,
                h: 120,
            },
            Area {
                x: FRAME.0 as i64 - 150,
                y: FRAME.1 as i64 - 110,
                w: 150,
                h: 110,
            },
        ] {
            for (name, params) in cases() {
                let got = node_picture(&dev.gpu, params, made);
                let want = cpu_picture(&params, made);
                let d = abs_diff(&got, &want);
                println!("edge {made:?}, {name}: {d:.2e}");
                assert!(d < GATE, "{made:?}, {name}: {d}");
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

/// The kernel alone: the sums `N` and `D` over `made` from the pyramid of the device, against the twin's from the twin's prep,
/// relative to 2e-4 (a floor of 1e-3 for values near 0).
#[test]
fn the_sums_are_the_twins() {
    let made = Area {
        x: 500,
        y: 320,
        w: 200,
        h: 140,
    };
    let frame = Area::whole(FRAME.0, FRAME.1);
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (name, params) in cases() {
            let reach = blur::reach_px(&params, FRAME.0, FRAME.1);
            let read = read_of(made, reach);
            let (pic, msk) = (picture(read), mask(read));
            let BlurKind::Lens {
                radius,
                blades,
                rotation_deg,
                highlights,
            } = params.kind
            else {
                unreachable!()
            };
            let kernel = LensKernel::of(&params).expect("lens");
            let (pic2, msk2) = (pic.clone(), msk.clone());
            let got: Vec<f32> = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let mut passes = PassList::new(ctx, "lens sums");
                    let picture = upload(ctx, read, 3, &pic2);
                    let mask = upload(ctx, read, 1, &msk2);
                    let pyr = encode_pyramid(
                        ctx,
                        &mut passes,
                        (&picture, &mask),
                        (read, frame),
                        (highlights, params.respect_mask_edges),
                        kernel.levels_for(frame),
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
                    kernel.encode(ctx, &mut passes, &pyr, &sums)?;
                    let r = sums.img.encode_read(&mut passes)?;
                    let done = passes.submit()?.finish()?;
                    sums.img.take(&done, r)
                })
                .expect("sums");
            // The twin: prep, then the kernel's f64 twin.
            let r_px = radius_px(radius, FRAME.0.max(FRAME.1) as f64);
            let base = twin::prep(
                &f64s(&pic),
                &f64s(&msk),
                read.w,
                read.h,
                highlights,
                params.respect_mask_edges,
            );
            let want = blur_lens::sums(&base, read, made, r_px, blades, rotation_deg);
            let mut worst = 0.0f64;
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max((f64::from(*g) - w).abs() / w.abs().max(1e-3));
            }
            println!("{name}: sums within {worst:.2e} (relative, floor 1e-3) of the twin's");
            assert!(worst < GATE, "{name}: {worst}");
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn tiles_of_the_node_equal_the_whole_made_area_bit_for_bit() {
    let whole = Area {
        x: 420,
        y: 260,
        w: 240,
        h: 180,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (name, params) in cases() {
            let full = node_picture(&dev.gpu, params, whole);
            let n = whole.w * whole.h;
            // 97 x 71 tiles (odd sizes: the prefix sums are integers, so the window's origin does not matter).
            let mut y = whole.y;
            while y < whole.y1() {
                let mut x = whole.x;
                while x < whole.x1() {
                    let tile = Area {
                        x,
                        y,
                        w: 97.min((whole.x1() - x) as usize),
                        h: 71.min((whole.y1() - y) as usize),
                    };
                    let got = node_picture(&dev.gpu, params, tile);
                    let tn = tile.w * tile.h;
                    for k in 0..3 {
                        for ty in 0..tile.h {
                            for tx in 0..tile.w {
                                let a = got[k * tn + ty * tile.w + tx];
                                let b = full[k * n
                                    + (tile.y - whole.y) as usize * whole.w
                                    + ty * whole.w
                                    + (tile.x - whole.x) as usize
                                    + tx];
                                assert!(
                                    a.to_bits() == b.to_bits(),
                                    "{name}: tile {tile:?} at ({tx}, {ty}) plane {k}: {a} against {b}"
                                );
                            }
                        }
                    }
                    x += 97;
                }
                y += 71;
            }
            println!("{name}: every tile of 97 x 71 is the whole area's bits");
        }
    }
    assert!(seen || gpus().is_empty());
}

/// The node's cost on this machine's device (lavapipe is a software Vulkan, so this is a CPU figure and not a Mac's), per
/// megapixel, for a whole 0.96 MP picture: upload, the prep, the kernel, the read-back and the blend, the scaffold's own cost (a
/// radius too small to blur) taken apart. Printed, not asserted (`docs/33` section 9).
#[test]
#[ignore = "speed: run by hand with --release --ignored --nocapture"]
fn the_cost_per_megapixel_is_measured() {
    let made = Area::whole(FRAME.0, FRAME.1);
    let mp = (FRAME.0 * FRAME.1) as f64 / 1e6;
    for dev in f32_devices() {
        let time = |params: BlurParams| {
            let _ = node_picture(&dev.gpu, params, made);
            let t = std::time::Instant::now();
            for _ in 0..3 {
                let _ = node_picture(&dev.gpu, params, made);
            }
            t.elapsed().as_secs_f64() / 3.0 * 1000.0 / mp
        };
        let base = time(lens(0.01, 0, 0.0, 0.0, true));
        println!("scaffold only (R 0.005 px): {base:.0} ms per MP");
        for (name, radius) in [
            ("R 4.8 px (Radius 10), the table", 10.0),
            ("R 7.9 px (Radius 16.5), the table", 16.5),
            ("R 48 px (Radius 100), rows", 100.0),
        ] {
            let round = time(lens(radius, 0, 0.0, 0.0, true));
            let hex = time(lens(radius, 6, 15.0, 60.0, true));
            println!(
                "{name}: round {round:.0} ms per MP ({:.0} over the scaffold), 6 blades and Highlights {hex:.0} ms per MP",
                round - base
            );
        }
    }
}

/// The kernel's device time per megapixel of the made area at `R` = 5, 8, 48, 120 and 240 px on a 6000 x 4000 frame and 344 px on an
/// 8600 x 5700 one (the radii the brief names and the largest; the made area is 640 x 480 in the middle, its read area grown by the reach), round and hexagon: the scan and the
/// kernel passes, the best of three runs after one warm-up, the upload not counted (`docs/33` 11 item 79).
#[test]
#[ignore = "speed: run by hand with --release --ignored --nocapture"]
fn the_kernel_cost_per_megapixel_at_the_radii_of_the_brief() {
    let mp = 640.0 * 480.0 / 1e6;
    for dev in f32_devices() {
        for r_px in [5.0f64, 8.0, 48.0, 120.0, 240.0, 344.0] {
            // The 8600 x 5700 frame of a 50 MP photo for its largest radius; the 6000 x 4000 frame else.
            let (fw, fh) = if r_px > 240.0 {
                (8600, 5700)
            } else {
                (6000, 4000)
            };
            let frame = Area::whole(fw, fh);
            let made = Area {
                x: (fw as i64 - 640) / 2,
                y: (fh as i64 - 480) / 2,
                w: 640,
                h: 480,
            };
            let radius = r_px / (0.0004 * fw as f64);
            for blades in [0u32, 6] {
                let params = lens(radius, blades, 15.0, 0.0, true);
                let reach = blur::reach_px(&params, fw, fh);
                let read = Area {
                    x: made.x - reach as i64,
                    y: made.y - reach as i64,
                    w: made.w + 2 * reach,
                    h: made.h + 2 * reach,
                };
                let n = read.w * read.h;
                let pic: Vec<f32> = (0..3 * n)
                    .map(|i| 0.3 + 0.4 * ((i % 977) as f32 / 977.0))
                    .collect();
                let msk = vec![1.0f32; n];
                let mut best = f64::MAX;
                for run in 0..4 {
                    let (p, m) = (pic.clone(), msk.clone());
                    let ms: f64 = dev
                        .gpu
                        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                            let mut passes = PassList::new(ctx, "lens speed");
                            let picture = upload(ctx, read, 3, &p);
                            let mask = upload(ctx, read, 1, &m);
                            let kernel = LensKernel::of(&params).expect("lens");
                            let pyr = encode_pyramid(
                                ctx,
                                &mut passes,
                                (&picture, &mask),
                                (read, frame),
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
                            // The prep alone first, so that its time can be taken off.
                            passes.submit()?.finish()?;
                            let mut passes = PassList::new(ctx, "lens speed kernel");
                            let t = std::time::Instant::now();
                            kernel.encode(ctx, &mut passes, &pyr, &sums)?;
                            passes.submit()?.finish()?;
                            let _ = (&picture, &mask, &pyr, &sums);
                            Ok(t.elapsed().as_secs_f64() * 1000.0)
                        })
                        .expect("speed");
                    if run > 0 {
                        best = best.min(ms);
                    }
                }
                println!(
                    "R {r_px:5.0} px, {} : kernel {best:8.1} ms for {mp:.2} MP = {:8.1} ms per MP (read area {} x {})",
                    if blades == 0 { "round" } else { "6 blades" },
                    best / mp,
                    read.w,
                    read.h
                );
            }
        }
    }
}
