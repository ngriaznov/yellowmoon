//! The Zoom type of a Blur node on the device (`docs/33-blur.md` section 3.5, BL-09; `kernels::blur_zoom`) against its CPU
//! reference: the gather over the pyramid against the twin (`ym_color::blur::zoom::gather`, f64, over the planes `twin::prep`
//! makes) and the node's picture against the CPU's fast method (`ym_color::blur::apply_fast`, f64), on lavapipe. Tiles of the node
//! equal the whole made area bit for bit.
//!
//! The stated bound is 2e-4 in LogC4 for the picture (`docs/33` section 3.7) and 2e-4 relative for the sums.
#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use common::chains;
use common::{TestGpu, gpus};
use ym_color::blur::zoom;
use ym_color::blur::{self, BlurKind, BlurParams, Frame, zoom_span};
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, OutputTarget};
use ym_color::logc::logc4_encode_rgb;
use ym_color::mask_raster::Area;
use ym_gpu::cpu::Placement;
use ym_gpu::cpu::blur as twin;
use ym_gpu::device::{GpuContext, StorageMode};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::blur_stack::{
    BlurKernel, BlurWork, SumsImage, device_kernel, encode_blur_picture, encode_pyramid,
};
use ym_gpu::kernels::blur_zoom::ZoomKernel;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The gate of `docs/33` section 3.7.
const GATE: f64 = 2e-4;

/// A big frame, so that the pyramid has several levels (the picture is a function of the place, nothing is stored for the whole).
const FRAME: (usize, usize) = (6000, 4000);

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

/// A LogC4 value at `(x, y)`: a gradient, a wide bright disc, a dark square below the toe, and bright lights (400 x middle grey)
/// among dark pixels, a function of the place alone so that every window agrees.
fn logc4_at(x: i64, y: i64) -> [f64; 3] {
    let (u, v) = (x as f64 / FRAME.0 as f64, y as f64 / FRAME.1 as f64);
    let mut c = [0.25 + 0.3 * u, 0.3 + 0.2 * v, 0.2 + 0.25 * (1.0 - u)];
    let r = ((u - 0.35).powi(2) + (1.5 * (v - 0.4)).powi(2)).sqrt();
    if r < 0.12 {
        c = [0.9, 0.88, 0.85];
    }
    if (1900..1980).contains(&x) && (1000..1060).contains(&y) {
        c = [-0.02, -0.015, -0.01];
    }
    // Texture, so that the taps of a line differ.
    let n = ((x * 7919 + y * 104_729) % 97) as f64 / 970.0;
    c = [c[0] + n, c[1] + 0.5 * n, c[2] - 0.5 * n];
    for (lx, ly) in [
        (1500i64, 1000i64),
        (1520, 1010),
        (2400, 1030),
        (301, 133),
        (5000, 3500),
    ] {
        if (x - lx).abs() < 2 && (y - ly).abs() < 2 {
            c = logc4_encode_rgb([72.0, 70.0, 66.0]);
        }
    }
    c
}

/// The mask: a ramp over the whole width, so every window holds fractional values.
fn mask_at(x: i64, _y: i64) -> f64 {
    ((x as f64 + 800.0) / 6000.0).clamp(0.0, 1.0)
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
        "zoom test",
        area.w,
        area.h,
        planes,
        d,
        Placement::whole(area.w, area.h),
    )
    .expect("upload")
}

fn zoom_node(amount: f64, center: [f64; 2], respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Zoom { amount, center },
        respect_mask_edges: respect,
    }
}

/// The cases: the top level 0 (Amount 8, the middle: the line is at most 15 px), then the levels 3, 4 and 5 of centres in the
/// frame and in its corner, Respect on and off.
fn cases() -> Vec<(&'static str, BlurParams)> {
    vec![
        (
            "level 0 (Amount 8, the middle)",
            zoom_node(8.0, [0.5, 0.5], true),
        ),
        (
            "Amount 40, centre (0.1, 0.2)",
            zoom_node(40.0, [0.1, 0.2], true),
        ),
        (
            "Amount 100, centre (0.1, 0.2), respect off",
            zoom_node(100.0, [0.1, 0.2], false),
        ),
        (
            "Amount 70, centre (0.8, 0.65)",
            zoom_node(70.0, [0.8, 0.65], true),
        ),
        (
            "Amount 100, centre in the corner (0, 0)",
            zoom_node(100.0, [0.0, 0.0], true),
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
        let mut passes = PassList::new(ctx, "zoom node");
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

/// A strip across the frame: its pixels lie 100 to 3000 px from the centres of the cases, so every level below the top is read.
const STRIP: Area = Area {
    x: 500,
    y: 980,
    w: 2700,
    h: 48,
};

#[test]
fn the_device_has_a_kernel_for_a_zoom_node() {
    for (name, params) in cases() {
        assert!(device_kernel(&params).is_some(), "{name}");
    }
}

#[test]
fn the_node_picture_is_the_cpu_fast_method_within_the_gate() {
    let block = Area {
        x: 1400,
        y: 900,
        w: 300,
        h: 200,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for made in [STRIP, block] {
            for (name, params) in cases() {
                let got = node_picture(&dev.gpu, params, made);
                let want = cpu_picture(&params, made);
                let d = abs_diff(&got, &want);
                println!(
                    "{name}, {}x{} at ({}, {}): the picture is within {d:.2e} of the CPU's fast method, in LogC4 (gate {GATE:.0e})",
                    made.w, made.h, made.x, made.y
                );
                assert!(d < GATE, "{name}: {d}");
            }
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
            Area {
                x: FRAME.0 as i64 - 200,
                y: 0,
                w: 200,
                h: 90,
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

/// The gather alone: the sums `N` and `D` over `made` from the pyramid of the device, against the twin's over the planes of
/// `twin::prep`, relative to 2e-4 (a floor of 1e-3 for values near 0).
#[test]
fn the_gather_over_the_pyramid_is_the_twins() {
    let frame = Area::whole(FRAME.0, FRAME.1);
    let made = Area {
        x: 600,
        y: 960,
        w: 2400,
        h: 40,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (name, params) in cases() {
            let reach = blur::reach_px(&params, FRAME.0, FRAME.1);
            let read = read_of(made, reach);
            let (pic, msk) = (picture(read), mask(read));
            let BlurKind::Zoom { amount, center } = params.kind else {
                unreachable!()
            };
            let kernel = ZoomKernel::of(&params).expect("zoom");
            let (pic2, msk2) = (pic.clone(), msk.clone());
            let got: Vec<f32> = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let mut passes = PassList::new(ctx, "zoom sums");
                    let picture = upload(ctx, read, 3, &pic2);
                    let mask = upload(ctx, read, 1, &msk2);
                    let pyr = encode_pyramid(
                        ctx,
                        &mut passes,
                        (&picture, &mask),
                        (read, frame),
                        (0.0, params.respect_mask_edges),
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
            // The twin: prep, then the f64 gather over the made area from the pyramid of the whole read window.
            let base = twin::prep(
                &f64s(&pic),
                &f64s(&msk),
                read.w,
                read.h,
                0.0,
                params.respect_mask_edges,
            );
            let n = read.w * read.h;
            let planes: Vec<[f64; 4]> = (0..n)
                .map(|i| [base[i], base[n + i], base[2 * n + i], base[3 * n + i]])
                .collect();
            let want = zoom::gather_region(
                &planes,
                (read.x as usize, read.y as usize, read.w, read.h),
                FRAME,
                (zoom_span(amount), center),
                (made.x as usize, made.y as usize, made.w, made.h),
            );
            let mn = made.w * made.h;
            let mut worst = 0.0f64;
            for y in 0..made.h {
                for x in 0..made.w {
                    let w = want[y * made.w + x];
                    for k in 0..4 {
                        let g = f64::from(got[k * mn + y * made.w + x]);
                        worst = worst.max((g - w[k]).abs() / w[k].abs().max(1e-3));
                    }
                }
            }
            println!(
                "{name}: top level {}, sums within {worst:.2e} (relative, floor 1e-3) of the twin's",
                kernel.levels_for(frame)
            );
            assert!(worst < GATE, "{name}: {worst}");
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn tiles_of_the_node_equal_the_whole_made_area_bit_for_bit() {
    let whole = Area {
        x: 900,
        y: 940,
        w: 600,
        h: 142,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (name, params) in cases() {
            let full = node_picture(&dev.gpu, params, whole);
            let n = whole.w * whole.h;
            // 97 x 71 tiles (odd sizes: the cells are at absolute positions whatever the tile).
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

/// A repeat run and a clean pool give the same bits (determinism, hard rule 8).
#[test]
fn a_repeat_of_the_node_is_the_same_bits() {
    let made = Area {
        x: 1000,
        y: 960,
        w: 300,
        h: 60,
    };
    for dev in f32_devices() {
        for (name, params) in cases() {
            let a = node_picture(&dev.gpu, params, made);
            let b = node_picture(&dev.gpu, params, made);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{name}"
            );
        }
    }
}

/// The node's cost on this machine's device (lavapipe is a software Vulkan, so this is a CPU figure and not a Mac's), per
/// megapixel, for a 1.0 MP made area: upload, the prep, the pyramid, the gather, the read-back and the blend, the scaffold's own
/// cost (an Amount too small to blur is not run, so Amount 0.01 stands in) taken apart. Printed, not asserted (`docs/33` section 9).
/// `--ignored`: it reads a 1.0 MP picture of the 6000 x 4000 frame for each run.
#[test]
#[ignore = "a speed measurement: run with --ignored --nocapture"]
fn the_cost_per_megapixel_is_measured() {
    let made = Area {
        x: 1500,
        y: 1000,
        w: 1000,
        h: 1000,
    };
    let mp = (made.w * made.h) as f64 / 1e6;
    for dev in f32_devices() {
        let time = |params: BlurParams| {
            let _ = node_picture(&dev.gpu, params, made);
            let t = std::time::Instant::now();
            for _ in 0..3 {
                let _ = node_picture(&dev.gpu, params, made);
            }
            t.elapsed().as_secs_f64() / 3.0 * 1000.0 / mp
        };
        let base = time(zoom_node(0.01, [0.5, 0.5], true));
        println!("scaffold only (Amount 0.01): {base:.0} ms per MP");
        for (name, params) in [
            (
                "Amount 20, centre (0.1, 0.2)",
                zoom_node(20.0, [0.1, 0.2], true),
            ),
            (
                "Amount 100, centre (0.1, 0.2)",
                zoom_node(100.0, [0.1, 0.2], true),
            ),
            (
                "Amount 100, corner (0, 0)",
                zoom_node(100.0, [0.0, 0.0], true),
            ),
        ] {
            let t = time(params);
            println!(
                "{name}: {t:.0} ms per MP ({:.0} over the scaffold)",
                t - base
            );
        }
    }
}
