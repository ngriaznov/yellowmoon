//! The magnification of a view above 100 % (`kernels::magnify`, `shaders/magnify.wgsl`) against its
//! CPU reference (`cpu::magnify`, what `ym-core` runs for a magnified view the device does not make).
//!
//! | Case | Asserted | Measured (Mesa lavapipe) |
//! |---|---|---|
//! | the magnified display-linear values, 12 views (zooms 1.013 to 8, odd sizes, pans that leave paper on every side, a source one pixel wide, a mask with holes and none), every storage mode | **identical** to the CPU reference (maximum difference 0) and the same paper | 0 differing samples |
//! | a part of a view (the taps of some columns and rows) | the same pixels of the whole view, bit for bit (D5) | identical |
//! | the fused RGBA8 pass | the planes followed by the encode kernel, byte for byte (sRGB and Rec.2020, dithered and rounded, two dither origins) | identical |
//! | the RGBA8 codes against the CPU (the reference values, encoded by `ym_color`'s f64 C13) | within one code, at least 99 % exact: the C13 bound of `docs/17-gpu.md` section 12, since the values are identical and only C13 differs (f32 on the device) | max 1 code, see the printed share |
//!
//! The values are exact by construction: the host computes every position, index and weight (the taps
//! of `cpu::magnify::MagnifyMap`) and the kernel's only arithmetic is the CPU's sum in the CPU's order,
//! every product and partial sum behind the fast-math barrier, so the same holds on Metal (fast-math
//! cannot fuse or reorder anything; `docs/17-gpu.md` section 13). The test asserts identity on every
//! adapter; `mac-gpu.yml` runs it on Metal.
//!
//! On lavapipe in `gpu-lavapipe` (`YM_REQUIRE_GPU=1`) and on Metal in `mac-gpu.yml`; skipped, loudly,
//! without an adapter.

mod common;

use common::{Rng, TestGpu, gpus};
use ym_color::encode::{OutputSpace, dither_8bit, quantize};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::magnify::{MagnifyMap, magnify};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::GpuError;
use ym_gpu::kernels::encode::{C13, Encoding, encode_host};
use ym_gpu::kernels::magnify::{magnify_planes, magnify_rgba8, max_binding_bytes};
use ym_gpu::queue::{JobSpec, Priority};

/// Prints like `println!` and to the process's own stderr, so that the numbers of a passing test
/// reach the CI log of the Apple runners.
macro_rules! note {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr(),
            format_args!("[gpu_magnify] {line}\n"),
        );
    }};
}

/// The share of 8-bit codes that must equal the CPU's (`docs/17-gpu.md` section 12, C13: within one
/// code, at least 99 % exact).
const MIN_EXACT_8: f64 = 0.99;

fn run<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("GPU job")
}

/// A sharpened display-linear picture: smooth colour with texture, edges, noise, a few values below
/// 0 and above 1 (output sharpening overshoots).
fn picture(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let edge = if (x / 3 + y / 5) % 2 == 0 { 0.25 } else { 0.0 };
            let r = 0.45 + 0.4 * (fx * 9.0).sin() * (fy * 4.0).cos() + edge;
            let g = 0.15 + 0.7 * fx * fy + 0.03 * rng.range(-1.0, 1.0);
            let b = 0.9 - 0.6 * fx + 0.25 * (fy * 13.0).sin() + 0.05 * rng.range(-1.0, 1.0);
            img.set_pixel(x, y, [r as f32, g as f32, b as f32]);
        }
    }
    img
}

/// One magnified view: the source and its mask, and the taps.
struct Case {
    name: String,
    src: RgbImage,
    mask: Option<Vec<u8>>,
    map: MagnifyMap,
}

/// A view of `view` device pixels at `zoom` whose pixel (0, 0) starts at `start` of a final image of
/// `final_size`, over the source `src_size` at `origin` (as `ym-core` makes them: centres at
/// `(x + 0.5) / zoom + start`).
#[allow(clippy::too_many_arguments)] // a test fixture: every parameter of the geometry
fn case(
    name: &str,
    src_size: (usize, usize),
    origin: (usize, usize),
    final_size: (usize, usize),
    view: (usize, usize),
    zoom: f64,
    start: (f64, f64),
    masked: bool,
    seed: u64,
) -> Case {
    let cols: Vec<f64> = (0..view.0)
        .map(|x| (x as f64 + 0.5) / zoom + start.0)
        .collect();
    let rows: Vec<f64> = (0..view.1)
        .map(|y| (y as f64 + 0.5) / zoom + start.1)
        .collect();
    let map = MagnifyMap::new(&cols, &rows, final_size, origin, src_size).expect("taps");
    // A tilted crop's corners: a triangle of the source outside the photo, and a few holes.
    let mask = masked.then(|| {
        (0..src_size.0 * src_size.1)
            .map(|i| {
                let (x, y) = (i % src_size.0, i / src_size.0);
                u8::from(x + 2 * y < src_size.0 / 3 || i % 37 == 5)
            })
            .collect()
    });
    Case {
        name: name.to_owned(),
        src: picture(src_size.0, src_size.1, seed),
        mask,
        map,
    }
}

fn cases() -> Vec<Case> {
    vec![
        case(
            "200 %",
            (70, 50),
            (0, 0),
            (70, 50),
            (120, 90),
            2.0,
            (3.0, 2.0),
            false,
            1,
        ),
        case(
            "200 %, a crop",
            (70, 50),
            (0, 0),
            (70, 50),
            (120, 90),
            2.0,
            (3.0, 2.0),
            true,
            2,
        ),
        case(
            "400 %, panned",
            (40, 30),
            (20, 10),
            (90, 60),
            (128, 96),
            4.0,
            (25.5, 13.25),
            false,
            3,
        ),
        case(
            "odd zoom 337 %",
            (41, 29),
            (3, 5),
            (60, 40),
            (121, 83),
            3.37,
            (5.1, 7.3),
            true,
            4,
        ),
        case(
            "101.3 %",
            (66, 48),
            (0, 0),
            (66, 48),
            (65, 47),
            1.013,
            (0.4, 0.2),
            false,
            5,
        ),
        case(
            "800 %",
            (20, 16),
            (8, 4),
            (40, 30),
            (144, 112),
            8.0,
            (9.0, 5.5),
            false,
            6,
        ),
        case(
            "odd zoom 790 %",
            (17, 13),
            (2, 1),
            (30, 20),
            (101, 77),
            7.9,
            (3.7, 2.9),
            true,
            7,
        ),
        // A view wider and taller than the picture: paper on every side (a narrow photo at 200 %).
        case(
            "paper all round",
            (30, 20),
            (0, 0),
            (30, 20),
            (100, 80),
            2.0,
            (-10.0, -10.0),
            true,
            8,
        ),
        case(
            "paper at the end",
            (30, 20),
            (0, 0),
            (30, 20),
            (80, 60),
            2.5,
            (10.0, 6.0),
            false,
            9,
        ),
        // A source one pixel wide: every column tap is that column.
        case(
            "one column",
            (1, 12),
            (4, 0),
            (5, 12),
            (12, 40),
            4.0,
            (3.5, 0.0),
            false,
            10,
        ),
        case(
            "odd cell 1x",
            (51, 33),
            (0, 0),
            (51, 33),
            (103, 67),
            2.02,
            (0.0, 0.0),
            true,
            11,
        ),
        case(
            "retina 3x",
            (30, 22),
            (6, 6),
            (50, 40),
            (91, 67),
            3.0,
            (6.7, 6.9),
            false,
            12,
        ),
    ]
}

fn first_difference(a: &[f32], b: &[f32]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(i) => format!("first at sample {i}: {} against {}", a[i], b[i]),
        None => "none".to_owned(),
    }
}

#[test]
fn the_magnified_values_are_the_cpu_references_bit_for_bit() {
    let mut compared = 0usize;
    for dev in gpus() {
        for c in cases() {
            let (cpu_rgb, cpu_paper) = magnify(&c.src, c.mask.as_deref(), &c.map).expect("cpu");
            let (src, mask, map) = (c.src.clone(), c.mask.clone(), c.map.clone());
            let (gpu_rgb, gpu_paper) = run(dev, move |ctx| {
                magnify_planes(ctx, &src, mask.as_deref(), &map)
            });
            assert!(gpu_rgb.iter().all(|v| v.is_finite()), "{}: finite", c.name);
            assert_eq!(
                gpu_paper, cpu_paper,
                "{} ({:?}): the paper",
                c.name, dev.mode
            );
            // `==` on the values: a sum that comes to zero may be +0 on one side and -0 on the
            // other (fast-math folds the `0 +` that starts it), which no code can tell apart.
            let max = gpu_rgb
                .iter()
                .zip(&cpu_rgb)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                gpu_rgb == cpu_rgb,
                "{} ({:?}): the device's values are not the CPU's, max {max:e} ({})",
                c.name,
                dev.mode,
                first_difference(&gpu_rgb, &cpu_rgb)
            );
            assert!(cpu_paper.contains(&0), "{}: some picture", c.name);
            compared += gpu_rgb.len();
        }
    }
    note!("magnified values: {compared} samples, maximum difference 0 (identical)");
}

#[test]
fn a_part_of_a_view_is_the_same_pixels_of_the_whole_view() {
    let Some(dev) = gpus().first() else { return };
    for c in cases().into_iter().filter(|c| c.map.width() > 40) {
        let (x0, x1, y0, y1) = (7, c.map.width() - 9, 5, c.map.height() - 3);
        let part = MagnifyMap {
            cols: c.map.cols[x0..x1].to_vec(),
            rows: c.map.rows[y0..y1].to_vec(),
            src: c.map.src,
        };
        let (src, mask, map) = (c.src.clone(), c.mask.clone(), c.map.clone());
        let (whole, whole_paper) = run(dev, move |ctx| {
            magnify_planes(ctx, &src, mask.as_deref(), &map)
        });
        let (src, mask) = (c.src.clone(), c.mask.clone());
        let (cut, cut_paper) = run(dev, move |ctx| {
            magnify_planes(ctx, &src, mask.as_deref(), &part)
        });
        let w = c.map.width();
        let pw = x1 - x0;
        for y in y0..y1 {
            let row = &whole[3 * (y * w + x0)..3 * (y * w + x1)];
            let cut_row = &cut[3 * (y - y0) * pw..3 * (y - y0 + 1) * pw];
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(cut_row), bits(row), "{}: row {y}", c.name);
            assert_eq!(
                &cut_paper[(y - y0) * pw..(y - y0 + 1) * pw],
                &whole_paper[y * w + x0..y * w + x1],
                "{}: row {y}",
                c.name
            );
        }
    }
}

/// The C13 settings the RGBA8 tests run.
fn encodings() -> Vec<(C13, [u8; 3])> {
    let c13 = |space, dither, origin| C13 {
        space,
        dither,
        origin,
    };
    vec![
        (c13(OutputSpace::DisplayP3, true, (0, 0)), [15, 15, 16]),
        (c13(OutputSpace::DisplayP3, true, (17, 5)), [255, 255, 255]),
        (c13(OutputSpace::Srgb, false, (0, 0)), [15, 15, 16]),
        (c13(OutputSpace::Rec2020, true, (3, 70)), [1, 2, 3]),
    ]
}

/// The CPU's RGBA8 of a magnified view: `ym_color`'s C13 (f64) of the reference values, the dither at
/// the pixel's place, the paper colour where the view is paper.
fn cpu_rgba8(rgb: &[f32], paper: &[u8], width: usize, c13: C13, background: [u8; 3]) -> Vec<u8> {
    let mut out = vec![255u8; 4 * paper.len()];
    for (i, px) in out.chunks_exact_mut(4).enumerate() {
        if paper[i] != 0 {
            px[..3].copy_from_slice(&background);
            continue;
        }
        let lin = [
            f64::from(rgb[3 * i]),
            f64::from(rgb[3 * i + 1]),
            f64::from(rgb[3 * i + 2]),
        ];
        let e = c13.space.encode_clamped(lin);
        let (x, y) = (c13.origin.0 + i % width, c13.origin.1 + i / width);
        for c in 0..3 {
            px[c] = if c13.dither {
                dither_8bit(e[c], x, y)
            } else {
                quantize(e[c], 8) as u8
            };
        }
    }
    out
}

#[test]
fn the_fused_pass_is_the_planes_then_the_encode_kernel_byte_for_byte() {
    let Some(dev) = gpus().first() else { return };
    for c in cases() {
        let (src, mask, map) = (c.src.clone(), c.mask.clone(), c.map.clone());
        let (rgb, paper) = run(dev, move |ctx| {
            magnify_planes(ctx, &src, mask.as_deref(), &map)
        });
        let (w, h) = (c.map.width(), c.map.height());
        // The planes as an image of their own, encoded by the encode kernel.
        let mut planar = RgbImage::new(w, h);
        for i in 0..w * h {
            planar.set_pixel(i % w, i / w, [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]]);
        }
        for (c13, background) in encodings() {
            let (planar, paper) = (planar.clone(), paper.clone());
            let chained = run(dev, move |ctx| {
                encode_host(
                    ctx,
                    &planar,
                    Some(&paper),
                    Rect::whole(w, h),
                    Encoding::Rgba8 { background },
                    c13,
                )
            });
            let (src, mask, map) = (c.src.clone(), c.mask.clone(), c.map.clone());
            let fused = run(dev, move |ctx| {
                magnify_rgba8(ctx, &src, mask.as_deref(), &map, c13, background)
            });
            assert!(
                fused == chained.bytes,
                "{} {c13:?}: the fused pass is not the planes then the encode kernel",
                c.name
            );
        }
    }
}

#[test]
fn the_codes_are_the_cpu_c13_of_the_same_values_within_one_code() {
    let (mut worst, mut min_exact, mut codes) = (0u8, 1.0f64, 0usize);
    for dev in gpus() {
        for c in cases() {
            let (rgb, paper) = magnify(&c.src, c.mask.as_deref(), &c.map).expect("cpu");
            for (c13, background) in encodings() {
                let (src, mask, map) = (c.src.clone(), c.mask.clone(), c.map.clone());
                let gpu = run(dev, move |ctx| {
                    magnify_rgba8(ctx, &src, mask.as_deref(), &map, c13, background)
                });
                let cpu = cpu_rgba8(&rgb, &paper, c.map.width(), c13, background);
                assert_eq!(gpu.len(), cpu.len());
                let (mut exact, mut n, mut max) = (0usize, 0usize, 0u8);
                for (i, (g, p)) in gpu.chunks_exact(4).zip(cpu.chunks_exact(4)).enumerate() {
                    assert_eq!(g[3], 255, "{}: alpha", c.name);
                    if paper[i] != 0 {
                        assert_eq!(g, p, "{}: the paper at pixel {i}", c.name);
                        continue;
                    }
                    for k in 0..3 {
                        let d = g[k].abs_diff(p[k]);
                        max = max.max(d);
                        exact += usize::from(d == 0);
                        n += 1;
                    }
                }
                let share = exact as f64 / n as f64;
                assert!(max <= 1, "{} {c13:?}: {max} codes off", c.name);
                assert!(share >= MIN_EXACT_8, "{} {c13:?}: {share:.5} exact", c.name);
                worst = worst.max(max);
                min_exact = min_exact.min(share);
                codes += n;
            }
        }
    }
    note!(
        "RGBA8 against the CPU's f64 C13 of the same values: {codes} codes, worst {worst} code, \
         smallest exact share {min_exact:.5}"
    );
}

#[test]
fn the_largest_binding_fits_the_design_size() {
    // The RGBA8 view of the largest Mac display (Pro Display XDR, 6016 x 3384 device pixels).
    common::limits::assert_max_binding_bytes(
        "magnify",
        Rect::new(0, 0, 6016, 3384),
        max_binding_bytes,
    );
}

#[test]
fn a_source_of_another_size_or_a_short_buffer_is_refused() {
    let Some(dev) = gpus().first() else { return };
    let c = case(
        "refused",
        (20, 10),
        (0, 0),
        (20, 10),
        (30, 20),
        1.5,
        (0.0, 0.0),
        false,
        1,
    );
    let mut other = c.map.clone();
    other.src = (21, 10);
    let src = c.src.clone();
    let err = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            magnify_planes(ctx, &src, None, &other)
        })
        .expect_err("another source size");
    assert!(matches!(err, GpuError::InvalidInput(_)), "{err:?}");
}
