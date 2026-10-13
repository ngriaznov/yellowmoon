//! Output sharpening in the writer (`docs/09` Pixel pipeline per format, `docs/algorithms/04`; the
//! owner's decision of 2026-10-13): the stage runs on the picture at the size it is written, right
//! after the Lanczos-3 resize, for JPEG, TIFF and PNG and never for a DNG.
//!
//! * no amount (0, or below 0.1 %) leaves every byte as it was,
//! * an amount gives the bytes of "the whole-frame stage, then the writer" (the strips the writer
//!   runs it in are invisible), at full size and after a resize, for linear and encoded input,
//! * cancel and progress follow the stage, a bad amount is refused,
//! * through `encode_to_writer` a soft edge gets sharper without a halo, and flat noise is left
//!   alone (spec 04 section 6.3).

mod common;

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};

use common::Rng;
use ym_color::encode::OutputSpace;
use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::request::{
    PictureStages, encode_to_vec_with, encode_to_writer, encode_to_writer_with,
};
use ym_export::resize::resize_lanczos3;
use ym_export::*;
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};

fn request<'a>(
    pixels: ImageRef<'a>,
    format: ExportFormat,
    size: Size,
    meta: &'a Metadata,
) -> ExportRequest<'a> {
    ExportRequest {
        pixels,
        format,
        color_space: ColorSpace::Srgb,
        size,
        metadata: meta,
        remove_location: false,
        dng: None,
    }
}

fn linear(w: u32, h: u32, data: &[f32]) -> ImageRef<'_> {
    ImageRef::new(w, h, data, PixelEncoding::DisplayLinear).unwrap()
}

fn formats() -> Vec<(&'static str, ExportFormat)> {
    vec![
        (
            "jpeg",
            ExportFormat::Jpeg {
                quality: JpegQuality::default(),
            },
        ),
        (
            "tiff8",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::None,
            },
        ),
        (
            "tiff16",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Zip,
            },
        ),
        (
            "png8",
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
        ),
        (
            "png16",
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
        ),
    ]
}

fn encode(req: &ExportRequest<'_>, stages: &PictureStages) -> Vec<u8> {
    encode_to_vec_with(req, &ExportHooks::none(), stages)
        .unwrap()
        .0
}

#[test]
fn no_amount_leaves_every_byte_of_every_format_as_it_was() {
    let (w, h) = (96, 70);
    let image = common::display_scene(w, h);
    let meta = common::metadata();
    for (name, format) in formats() {
        for size in [Size::Full, Size::LongEdge { px: 48 }] {
            let req = request(linear(w, h, &image), format, size, &meta);
            let (plain, summary) = ym_export::encode_to_vec(&req, &ExportHooks::none()).unwrap();
            for stages in [
                PictureStages::default(),
                PictureStages::output_sharpening(0.0),
                // Below a tenth of a percent the stage has no amount (spec 04: A < 0.001).
                PictureStages::output_sharpening(0.05),
            ] {
                let (bytes, again) =
                    encode_to_vec_with(&req, &ExportHooks::none(), &stages).unwrap();
                assert!(bytes == plain, "{name} {size:?} {stages:?}: bytes differ");
                assert_eq!(again, summary);
            }
            // A real amount changes the bytes (so the equality above says something).
            let sharp = encode(&req, &PictureStages::output_sharpening(100.0));
            assert!(sharp != plain, "{name} {size:?}: 100 % changed nothing");
        }
    }
}

/// What the writer is specified to do, spelled out with the pieces: the resize, the stage on the
/// whole frame at `s = out long edge / rendered long edge`, then the writer at full size.
fn by_hand(
    image: &ImageRef<'_>,
    format: ExportFormat,
    size: Size,
    meta: &Metadata,
    amount: f32,
) -> Vec<u8> {
    let (ow, oh) = size.output_dimensions(image.width, image.height);
    let space = OutputSpace::Srgb;
    let resized;
    let at_size = if (ow, oh) == (image.width, image.height) {
        *image
    } else {
        resized = resize_lanczos3(image, ow, oh, space, &ExportHooks::none()).unwrap();
        resized.as_ref()
    };
    // The stage reads linear light: decode an encoded input first, as the resize does.
    let lin: Vec<f32> = if at_size.encoding == PixelEncoding::DisplayEncoded {
        at_size
            .rgb
            .iter()
            .map(|&v| space.decode_channel(f64::from(v)) as f32)
            .collect()
    } else {
        at_size.rgb.to_vec()
    };
    let s = scale_for_edges(ow.max(oh) as usize, image.width.max(image.height) as usize);
    let sharp = OutputSharpen::screen()
        .with_amount_pct(amount)
        .apply_interleaved(ow as usize, oh as usize, &lin, s)
        .unwrap();
    let req = request(linear(ow, oh, &sharp), format, Size::Full, meta);
    ym_export::encode_to_vec(&req, &ExportHooks::none())
        .unwrap()
        .0
}

#[test]
fn the_writer_gives_the_bytes_of_the_whole_frame_stage_then_the_encode() {
    // 1100 rows: two strips of the writer's stage (1024 core rows each), so a seam would show.
    let (w, h) = (150, 1100);
    let image = common::display_scene(w, h);
    let meta = common::metadata();
    for amount in [100.0_f32, 35.0, 200.0] {
        for (name, format) in formats() {
            for size in [
                Size::Full,
                Size::LongEdge { px: 700 },
                Size::LongEdge { px: 90 },
            ] {
                let img = linear(w, h, &image);
                let req = request(img, format, size, &meta);
                let got = encode(&req, &PictureStages::output_sharpening(amount));
                let want = by_hand(&img, format, size, &meta, amount);
                assert!(got == want, "{name} {size:?} at {amount} %: bytes differ");
            }
        }
    }
}

#[test]
fn encoded_input_is_decoded_to_linear_light_first() {
    let (w, h) = (80, 60);
    let linear_image = common::display_scene(w, h);
    let space = OutputSpace::Srgb;
    let encoded: Vec<f32> = linear_image
        .iter()
        .map(|&v| space.encode_channel(f64::from(v)) as f32)
        .collect();
    let meta = common::metadata();
    let format = ExportFormat::Tiff {
        bits: BitDepth::Sixteen,
        compression: TiffCompression::None,
    };
    let img = ImageRef::new(w, h, &encoded, PixelEncoding::DisplayEncoded).unwrap();
    for size in [Size::Full, Size::LongEdge { px: 40 }] {
        let req = request(img, format, size, &meta);
        let got = encode(&req, &PictureStages::output_sharpening(100.0));
        let want = by_hand(&img, format, size, &meta, 100.0);
        assert!(got == want, "{size:?}: bytes differ");
    }
}

#[test]
fn a_dng_is_never_sharpened() {
    let meta = common::metadata();
    let (w, h) = (37, 21);
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(w, h);
    let profile = DngProfile::identity();
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::default(),
            profile: &profile,
        }),
    };
    let plain = ym_export::encode_to_vec(&req, &ExportHooks::none())
        .unwrap()
        .0;
    for amount in [0.0, 100.0, 200.0] {
        let bytes = encode(&req, &PictureStages::output_sharpening(amount));
        assert!(bytes == plain, "DNG changed with {amount} %");
    }
}

#[test]
fn amounts_outside_the_range_are_refused() {
    let image = common::display_scene(16, 12);
    let meta = common::metadata();
    let req = request(
        linear(16, 12, &image),
        ExportFormat::Png {
            bits: BitDepth::Eight,
        },
        Size::Full,
        &meta,
    );
    for bad in [-1.0, 200.5, 1000.0, f32::NAN, f32::INFINITY] {
        let r = encode_to_writer_with(
            &req,
            &mut Cursor::new(Vec::new()),
            &ExportHooks::none(),
            &PictureStages::output_sharpening(bad),
        );
        assert!(
            matches!(r, Err(ExportError::InvalidInput(_))),
            "{bad}: {r:?}"
        );
    }
    for good in [0.0, 5.0, 200.0] {
        assert!(
            encode_to_writer_with(
                &req,
                &mut Cursor::new(Vec::new()),
                &ExportHooks::none(),
                &PictureStages::output_sharpening(good),
            )
            .is_ok()
        );
    }
}

#[test]
fn the_stage_reports_progress_and_can_be_cancelled() {
    let (w, h) = (120, 2300); // three strips
    let image = common::display_scene(w, h);
    let meta = common::metadata();
    let req = request(
        linear(w, h, &image),
        ExportFormat::Png {
            bits: BitDepth::Eight,
        },
        Size::LongEdge { px: 1500 },
        &meta,
    );
    let stages = PictureStages::output_sharpening(100.0);
    let seen = std::sync::Mutex::new(Vec::<f32>::new());
    let progress = |f: f32| seen.lock().unwrap().push(f);
    let hooks = ExportHooks::new(None, Some(&progress));
    encode_to_writer_with(&req, &mut Cursor::new(Vec::new()), &hooks, &stages).unwrap();
    let seen = seen.into_inner().unwrap();
    assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)), "{seen:?}");
    assert!(seen.windows(2).all(|p| p[0] <= p[1]), "{seen:?}");
    assert!(seen.last().copied().unwrap_or(0.0) >= 0.99, "{seen:?}");
    // Cancelled before it starts: nothing is written.
    let stop = AtomicBool::new(false);
    let cancel = || stop.load(Ordering::SeqCst);
    let hooks = ExportHooks::new(Some(&cancel), None);
    assert!(encode_to_writer_with(&req, &mut Cursor::new(Vec::new()), &hooks, &stages).is_ok());
    stop.store(true, Ordering::SeqCst);
    let r = encode_to_writer_with(&req, &mut Cursor::new(Vec::new()), &hooks, &stages);
    assert!(matches!(r, Err(ExportError::Cancelled)), "{r:?}");
    // Cancelled inside the stage: the resize takes the first 30 % of the progress, so the first
    // report above that comes from the stage; the stage looks again before its next strip.
    let stop = AtomicBool::new(false);
    let cancel = || stop.load(Ordering::SeqCst);
    let progress = |f: f32| {
        if f > 0.31 {
            stop.store(true, Ordering::SeqCst);
        }
    };
    let hooks = ExportHooks::new(Some(&cancel), Some(&progress));
    let mut sink = Cursor::new(Vec::new());
    let r = encode_to_writer_with(&req, &mut sink, &hooks, &stages);
    assert!(matches!(r, Err(ExportError::Cancelled)), "{r:?}");
    assert!(sink.into_inner().is_empty(), "nothing was written");
}

// ---- the edge profile through the writer (spec 04 section 6.3) --------------------------------------

/// Error function (Abramowitz and Stegun 7.1.26 is too coarse for 1e-3; use the series/continued
/// fraction through `libm`-free high-order rational form by Numerical Recipes `erfc` Chebyshev).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    let erfc = ans;
    if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
}

/// A soft slanted edge (5 degrees from vertical), Gaussian sigma 0.8 px, from 0.1 to 0.5, each
/// pixel the mean of 4 x 4 samples, as grey RGB. Returns the image and the edge's angle.
fn soft_edge(w: u32, h: u32) -> Vec<f32> {
    let angle = 5.0_f64.to_radians();
    let (nx, ny) = (angle.cos(), -angle.sin());
    let (cx, cy) = (f64::from(w) / 2.0, f64::from(h) / 2.0);
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    let (px, py) = (
                        f64::from(x) + (f64::from(i) + 0.5) / 4.0,
                        f64::from(y) + (f64::from(j) + 0.5) / 4.0,
                    );
                    let d = (px - cx) * nx + (py - cy) * ny;
                    acc += 0.1 + 0.4 * 0.5 * (1.0 + erf(d / (0.8 * std::f64::consts::SQRT_2)));
                }
            }
            let g = (acc / 16.0) as f32;
            v.extend_from_slice(&[g, g, g]);
        }
    }
    v
}

/// The linear green values of a 16-bit sRGB PNG, row-major, and its size.
fn decode_png16(bytes: &[u8]) -> (usize, usize, Vec<f64>) {
    let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    let (w, h) = (info.width as usize, info.height as usize);
    let linear = (0..w * h)
        .map(|i| {
            let o = (i * 3 + 1) * 2;
            let code = f64::from(u16::from_be_bytes([buf[o], buf[o + 1]])) / 65535.0;
            OutputSpace::Srgb.decode_channel(code)
        })
        .collect();
    (w, h, linear)
}

/// The 10 to 90 % rise distance along the middle row (linear interpolation), and the overshoot and
/// undershoot in percent of the step, from the edge-spread function binned by distance to the edge.
fn edge_profile(w: usize, h: usize, g: &[f64], (lo, hi): (f64, f64)) -> (f64, f64, f64) {
    let row = &g[(h / 2) * w..(h / 2 + 1) * w];
    let crossing = |level: f64| {
        (1..w)
            .find(|&x| row[x - 1] < level && row[x] >= level)
            .map(|x| (x - 1) as f64 + (level - row[x - 1]) / (row[x] - row[x - 1]))
            .unwrap()
    };
    let step = hi - lo;
    let rise = crossing(lo + 0.9 * step) - crossing(lo + 0.1 * step);
    let angle = 5.0_f64.to_radians();
    let (nx, ny) = (angle.cos(), -angle.sin());
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut bins: Vec<(f64, usize)> = vec![(0.0, 0); 400];
    for y in 0..h {
        for x in 0..w {
            let d = (x as f64 + 0.5 - cx) * nx + (y as f64 + 0.5 - cy) * ny;
            if d.abs() < 20.0 {
                let b = ((d + 20.0) / 0.1) as usize;
                bins[b].0 += g[y * w + x];
                bins[b].1 += 1;
            }
        }
    }
    let means: Vec<f64> = bins
        .iter()
        .filter(|b| b.1 > 0)
        .map(|b| b.0 / b.1 as f64)
        .collect();
    let max = means.iter().copied().fold(f64::MIN, f64::max);
    let min = means.iter().copied().fold(f64::MAX, f64::min);
    (rise, 100.0 * (max - hi) / step, 100.0 * (lo - min) / step)
}

/// The standard deviation of the green channel of a flat patch with noise of sigma 0.004 in each
/// channel (independent, as sensor noise is) through the writer, with and without the stage: the
/// gain.
fn noise_gain(amount: f32, size: Size) -> f64 {
    let (w, h) = (300_u32, 300_u32);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..w * h * 3 {
        // Sum of 12 uniforms: a normal of sigma 1, scaled to sigma 0.004 around 0.3.
        let n: f32 = (0..12).map(|_| rng.unit()).sum::<f32>() - 6.0;
        v.push(0.3 + 0.004 * n);
    }
    let meta = common::metadata();
    let req = request(
        linear(w, h, &v),
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
        size,
        &meta,
    );
    let std = |amount: f32| {
        let bytes = encode(&req, &PictureStages::output_sharpening(amount));
        let (pw, ph, g) = decode_png16(&bytes);
        // The middle of the patch, away from the frame's mirrored border.
        let (m, n) = (pw / 4, pw - pw / 4);
        let (mh, nh) = (ph / 4, ph - ph / 4);
        let mut vals = Vec::new();
        for y in mh..nh {
            for x in m..n {
                vals.push(g[y * pw + x]);
            }
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt()
    };
    std(amount) / std(0.0)
}

#[test]
fn a_soft_edge_gets_sharper_through_the_writer_without_a_halo_and_flat_noise_is_left_alone() {
    let (w, h) = (256_u32, 256_u32);
    let edge = soft_edge(w, h);
    let meta = common::metadata();
    let format = ExportFormat::Png {
        bits: BitDepth::Sixteen,
    };
    let req = request(linear(w, h, &edge), format, Size::Full, &meta);
    let before = {
        let bytes = encode(&req, &PictureStages::output_sharpening(0.0));
        let (pw, ph, g) = decode_png16(&bytes);
        edge_profile(pw, ph, &g, (0.1, 0.5))
    };
    let after = {
        let bytes = encode(&req, &PictureStages::output_sharpening(100.0));
        let (pw, ph, g) = decode_png16(&bytes);
        edge_profile(pw, ph, &g, (0.1, 0.5))
    };
    eprintln!(
        "full size: rise {:.2} -> {:.2} px, overshoot {:.2} % undershoot {:.2} %",
        before.0, after.0, after.1, after.2
    );
    // The spec's numbers: 2.51 -> 1.94 px, overshoot at most 3.5 %.
    assert!(
        (before.0 - 2.5).abs() < 0.1,
        "unsharpened rise {}",
        before.0
    );
    assert!(
        after.0 < 2.05 && after.0 > 1.7,
        "sharpened rise {}",
        after.0
    );
    assert!(
        after.1 <= 3.5 && after.2 <= 3.5,
        "overshoot {} undershoot {}",
        after.1,
        after.2
    );
    let gain = noise_gain(100.0, Size::Full);
    eprintln!("full size: noise gain x{gain:.3}");
    assert!(gain <= 1.05, "noise gain {gain}");

    // At a 128 px long edge (half size, s = 0.5) the edge is sharper than the resize alone leaves it.
    let small = Size::LongEdge { px: 128 };
    let req = request(linear(w, h, &edge), format, small, &meta);
    let (b, a) = (
        {
            let (pw, ph, g) = decode_png16(&encode(&req, &PictureStages::output_sharpening(0.0)));
            edge_profile(pw, ph, &g, (0.1, 0.5))
        },
        {
            let (pw, ph, g) = decode_png16(&encode(&req, &PictureStages::output_sharpening(100.0)));
            edge_profile(pw, ph, &g, (0.1, 0.5))
        },
    );
    eprintln!(
        "half size: rise {:.2} -> {:.2} px, overshoot {:.2} -> {:.2} %, undershoot {:.2} -> {:.2} %",
        b.0, a.0, b.1, a.1, b.2, a.2
    );
    assert!(a.0 < b.0, "rise {} -> {}", b.0, a.0);
    // The Lanczos-3 resize overshoots by itself; the stage adds a few percent at most.
    assert!(
        a.1 <= b.1 + 5.0 && a.2 <= b.2 + 5.0,
        "overshoot {} -> {}, undershoot {} -> {}",
        b.1,
        a.1,
        b.2,
        a.2
    );
    assert!(noise_gain(100.0, small) <= 1.05);
}

#[test]
fn plain_encode_to_writer_is_the_unsharpened_encode() {
    // The long-standing entry point never sharpens, whatever the photo's settings are.
    let image = common::display_scene(64, 48);
    let meta = common::metadata();
    let req = request(
        linear(64, 48, &image),
        ExportFormat::Jpeg {
            quality: JpegQuality::default(),
        },
        Size::Full,
        &meta,
    );
    let mut a = Cursor::new(Vec::new());
    encode_to_writer(&req, &mut a, &ExportHooks::none()).unwrap();
    assert!(a.into_inner() == encode(&req, &PictureStages::default()));
}
