//! The size estimate behind the "≈ 120 MB" of the export panel (`docs/09-export.md`, section Size estimate).
//!
//! A file's size depends on the pixels as much as on the pixel count: the same 24 MP frame is 1 bit per pixel
//! as a smooth, denoised JPEG at quality 92 and 2.7 bits per pixel as a detailed one. So the estimate takes a
//! cheap measure of how much fine detail the picture carries, [`DetailStat`], computed from a small preview of
//! the render (the 240 px thumbnails the app already has), next to the export's pixel count and options.
//!
//! # The model
//!
//! For every format variant the compressed size is a power law of the detail and of the output size:
//!
//! ```text
//! ln(bits per pixel)  [JPEG]      = a + b*x + c*y + d*x*y      x = ln(g / G0)
//! ln(size / raw size) [TIFF, PNG] = a + b*x + c*y + d*x*y      y = ln(output_megapixels / 10)
//! ```
//!
//! `g` is the detail statistic and `G0` ([`TYPICAL_GRADIENT`]) the typical one, so that `a` is the (log) value for
//! a typical picture of 10 MP. There is one `(a, b, c, d)` per JPEG quality knot (the constant `JPEG_FIT`, values
//! between the knots are interpolated in the log domain) and per TIFF/PNG variant (`RATIO_FIT`). The coefficients
//! were fitted by least squares on Yellowmoon renders (`tests/calibrate_estimate.rs` writes the measurements,
//! `tests/py/fit_estimate.py` fits and prints the constants; `docs/09`), and are re-fitted the same way whenever the
//! pipeline changes how much a render is smoothed or sharpened.
//!
//! Without a preview ([`estimate_bytes`]) the typical detail is assumed.

use crate::dng;
use crate::options::{BitDepth, ExportFormat, Size, TiffCompression};

/// The long edge, in pixels, the detail statistic is measured at. Previews larger than this are box-averaged
/// down to it (by a whole factor); smaller ones are used as they are, with the scale correction of
/// [`SCALE_EXPONENT`].
pub const ANALYSIS_LONG_EDGE_PX: u32 = 240;

/// Previews with fewer pixels than this on the short side (after the reduction to the analysis size) carry no
/// usable statistic: [`DetailStat::measure_rgb8`] returns `None` for them.
pub const MIN_ANALYSED_EDGE_PX: u32 = 16;

/// How the mean gradient scales with the size of the analysed picture: `g(s) ~ s^-SCALE_EXPONENT` for the
/// long edge `s` in pixels (measured on the calibration renders at 240 and 480 px: 0.25 on the median). The
/// statistic is normalised to [`ANALYSIS_LONG_EDGE_PX`] with it, which only matters for previews a little
/// smaller than that (the whole-factor reduction of a 6000 px frame leaves 208 to 240 px).
pub const SCALE_EXPONENT: f64 = 0.25;

/// The typical detail statistic: the geometric mean over the calibration renders. Used when no preview is
/// given.
pub const TYPICAL_GRADIENT: f64 = 0.01751;

/// The statistic is clamped to this range before the model sees it, so that a flat or a very busy preview
/// (outside the calibration data) cannot send the estimate to extremes.
pub const GRADIENT_RANGE: (f64, f64) = (0.003, 0.12);

/// One fitted power law: `ln y = a + b * ln(g / G0) + c * ln(mp / 10) + d * ln(g / G0) * ln(mp / 10)`, with `g` the
/// detail statistic and `mp` the output size in megapixels.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LogFit {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl LogFit {
    const fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// `ln y`, with the detail clamped to [`GRADIENT_RANGE`] and the size to 0.1 to 400 MP.
    fn ln_eval(self, gradient: f64, megapixels: f64) -> f64 {
        let lg = (gradient.clamp(GRADIENT_RANGE.0, GRADIENT_RANGE.1) / TYPICAL_GRADIENT).ln();
        let lm = (megapixels.clamp(0.1, 400.0) / 10.0).ln();
        self.a + self.b * lg + self.c * lm + self.d * lg * lm
    }

    fn eval(self, gradient: f64, megapixels: f64) -> f64 {
        self.ln_eval(gradient, megapixels).exp()
    }
}

/// JPEG bits per pixel by quality: `(quality, fit)`. 4:4:4 chroma from quality 90, 4:2:0 below, optimised
/// tables (this crate's encoder).
const JPEG_FIT: [(f64, LogFit); 10] = [
    (50.0, LogFit::new(-1.2607, 0.8027, -0.1832, -0.0316)),
    (60.0, LogFit::new(-1.0951, 0.7718, -0.1721, -0.0422)),
    (75.0, LogFit::new(-0.7394, 0.6840, -0.1473, -0.0638)),
    (80.0, LogFit::new(-0.5607, 0.6372, -0.1371, -0.0707)),
    (85.0, LogFit::new(-0.3469, 0.5830, -0.1265, -0.0777)),
    (89.0, LogFit::new(-0.1222, 0.5243, -0.1149, -0.0835)),
    (90.0, LogFit::new(0.1980, 0.4872, -0.1059, -0.0786)),
    (92.0, LogFit::new(0.3289, 0.4644, -0.1043, -0.0765)),
    (95.0, LogFit::new(0.6990, 0.3797, -0.0908, -0.0724)),
    (100.0, LogFit::new(1.8750, 0.2032, -0.0483, -0.0395)),
];

/// Compressed size over raw size for TIFF and PNG, `[8-bit, 16-bit]`.
const RATIO_FIT: RatioFits = RatioFits {
    lzw: [
        LogFit::new(-0.9017, 0.2265, -0.0562, -0.0505),
        LogFit::new(-0.0079, 0.0866, -0.0125, -0.0162),
    ],
    zip: [
        LogFit::new(-0.9513, 0.2016, -0.0505, -0.0396),
        LogFit::new(-0.2388, 0.0551, -0.0068, -0.0094),
    ],
    png: [
        LogFit::new(-1.0001, 0.1863, -0.0425, -0.0372),
        LogFit::new(-0.2426, 0.0431, -0.0049, -0.0098),
    ],
};

struct RatioFits {
    lzw: [LogFit; 2],
    zip: [LogFit; 2],
    png: [LogFit; 2],
}

/// The largest ratio of a compressed TIFF or PNG to its raw size the model will give (LZW on noisy 16-bit samples
/// can be as large as the raw data; `docs/09`).
const MAX_RATIO: f64 = 1.25;
/// The smallest and largest JPEG bits per pixel the model will give.
const BPP_RANGE: (f64, f64) = (0.05, 24.0);

/// A cheap measure of how much fine detail and noise a picture carries: the mean absolute difference between
/// neighbouring pixels (horizontal and vertical) of its luma (Rec.709 weights on the display-encoded values, 0 to
/// 1), for the picture reduced to about [`ANALYSIS_LONG_EDGE_PX`] pixels on its long edge.
///
/// It is a number from about 0.005 (a smooth, denoised frame) to 0.06 (foliage, fine texture, visible noise).
/// Measuring it costs one pass over the preview.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DetailStat(f64);

impl DetailStat {
    /// The detail of a typical frame ([`TYPICAL_GRADIENT`]), what [`estimate_bytes`] assumes.
    pub const TYPICAL: DetailStat = DetailStat(TYPICAL_GRADIENT);

    /// A statistic from its value, `None` if it is not a finite non-negative number.
    #[must_use]
    pub fn from_gradient(gradient: f64) -> Option<Self> {
        (gradient.is_finite() && gradient >= 0.0).then_some(Self(gradient))
    }

    /// The statistic: the mean neighbour difference of the picture at [`ANALYSIS_LONG_EDGE_PX`].
    #[must_use]
    pub fn gradient(self) -> f64 {
        self.0
    }

    /// Measures an 8-bit sRGB-encoded RGB preview (`width * height * 3` bytes, interleaved), for example the
    /// app's thumbnail. `None` when the buffer does not match the size or the preview is too small to say
    /// anything (under [`MIN_ANALYSED_EDGE_PX`] pixels on the short side after the reduction).
    #[must_use]
    pub fn measure_rgb8(width: u32, height: u32, rgb: &[u8]) -> Option<Self> {
        measure(width, height, rgb, |v: u8| f64::from(v) * (1.0 / 255.0))
    }

    /// Measures a display-encoded `f32` RGB preview (values 0 to 1, interleaved), same rules as
    /// [`DetailStat::measure_rgb8`].
    #[must_use]
    pub fn measure_encoded_f32(width: u32, height: u32, rgb: &[f32]) -> Option<Self> {
        measure(width, height, rgb, |v: f32| f64::from(v))
    }
}

/// Box-averages `rgb` down by a whole factor to at most [`ANALYSIS_LONG_EDGE_PX`], takes the mean neighbour
/// difference and normalises it to the analysis size.
fn measure<T: Copy>(
    width: u32,
    height: u32,
    rgb: &[T],
    value: impl Fn(T) -> f64,
) -> Option<DetailStat> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || w.checked_mul(h)?.checked_mul(3)? != rgb.len() {
        return None;
    }
    let long = w.max(h);
    let k = long.div_ceil(ANALYSIS_LONG_EDGE_PX as usize).max(1);
    let (tw, th) = (w / k, h / k);
    if tw.min(th) < MIN_ANALYSED_EDGE_PX as usize {
        return None;
    }
    // The reduced picture, box average over k x k cells (the remainder at the right and bottom is dropped).
    let mut small = vec![0.0f64; tw * th * 3];
    let norm = 1.0 / (k * k) as f64;
    for ty in 0..th {
        let cells = &mut small[ty * tw * 3..(ty + 1) * tw * 3];
        for y in ty * k..(ty + 1) * k {
            let row = &rgb[y * w * 3..(y * w + tw * k) * 3];
            for (cell, pixels) in cells.chunks_exact_mut(3).zip(row.chunks_exact(k * 3)) {
                for p in pixels.chunks_exact(3) {
                    cell[0] += value(p[0]);
                    cell[1] += value(p[1]);
                    cell[2] += value(p[2]);
                }
            }
        }
        for v in cells {
            *v *= norm;
        }
    }
    // The luma of the reduced picture, then the mean of 0.5 * (|dx| + |dy|) over every pixel that has both a
    // right and a lower neighbour.
    let luma: Vec<f64> = small
        .chunks_exact(3)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();
    let mut total = 0.0f64;
    for y in 0..th - 1 {
        for x in 0..tw - 1 {
            let v = luma[y * tw + x];
            total += (luma[y * tw + x + 1] - v).abs() + (luma[(y + 1) * tw + x] - v).abs();
        }
    }
    let gradient = 0.5 * total / ((tw - 1) * (th - 1)) as f64;
    let analysed_long_edge = tw.max(th) as f64;
    DetailStat::from_gradient(
        gradient * (analysed_long_edge / f64::from(ANALYSIS_LONG_EDGE_PX)).powf(SCALE_EXPONENT),
    )
}

/// Linear interpolation of the fits' log value between quality knots, so that the size grows smoothly with the
/// quality. The result is the bits per pixel.
fn jpeg_bits_per_pixel(quality: f64, gradient: f64, megapixels: f64) -> f64 {
    let at_knot = |i: usize| JPEG_FIT[i].1.ln_eval(gradient, megapixels);
    // A larger quality never gives a smaller file: running maximum over the knots.
    let mut ln_bpp = [0.0f64; JPEG_FIT.len()];
    let mut running = f64::NEG_INFINITY;
    for (i, slot) in ln_bpp.iter_mut().enumerate() {
        running = running.max(at_knot(i));
        *slot = running;
    }
    let first = JPEG_FIT[0].0;
    let last = JPEG_FIT[JPEG_FIT.len() - 1].0;
    let q = quality.clamp(first, last);
    let mut value = ln_bpp[ln_bpp.len() - 1];
    for i in 0..JPEG_FIT.len() - 1 {
        let (q0, q1) = (JPEG_FIT[i].0, JPEG_FIT[i + 1].0);
        if q <= q1 {
            let t = (q - q0) / (q1 - q0);
            value = ln_bpp[i] + t * (ln_bpp[i + 1] - ln_bpp[i]);
            break;
        }
    }
    value.exp().clamp(BPP_RANGE.0, BPP_RANGE.1)
}

/// The expected file size in bytes of an export of a `width` x `height` frame for a picture of typical detail:
/// the number behind "≈ 120 MB" in the export panel when no preview is at hand. See
/// [`estimate_bytes_with_detail`].
#[must_use]
pub fn estimate_bytes(width: u32, height: u32, format: ExportFormat, size: Size) -> u64 {
    estimate_bytes_with_detail(width, height, format, size, None)
}

/// The expected file size in bytes of an export of a `width` x `height` frame (the size of the render before
/// the export's own resize), given the picture's [`DetailStat`] (`None`: a typical picture).
///
/// The metadata and profile add under 4 KB, which is included. The error against real exports is in `docs/09`
/// (section Size estimate). On the nine real renders it was fitted on, the middle estimate is within a few percent
/// of the file, and with the statistic about one JPEG file in ten at quality 80 to 92 is off by more than 30 %
/// (two frames of the same statistic can carry different fine texture) while TIFF and PNG files stay within 25 %;
/// without the statistic ([`estimate_bytes`], typical detail) about one file in three is off by more than 30 %. For
/// a picture the fit has not seen the errors are a little larger (leave-one-out figures in `docs/09`). Neither
/// sees sensor noise that the render did not remove (noise reduction off): such files come out larger than
/// estimated.
#[must_use]
pub fn estimate_bytes_with_detail(
    width: u32,
    height: u32,
    format: ExportFormat,
    size: Size,
    detail: Option<DetailStat>,
) -> u64 {
    let (w, h) = if format == ExportFormat::Dng {
        (width, height)
    } else {
        size.output_dimensions(width, height)
    };
    let pixels = f64::from(w) * f64::from(h);
    let megapixels = pixels / 1e6;
    let gradient = detail.unwrap_or(DetailStat::TYPICAL).gradient();
    let bits_index = |bits: BitDepth| usize::from(bits == BitDepth::Sixteen);
    let ratio = |fit: LogFit| fit.eval(gradient, megapixels).min(MAX_RATIO);
    let bytes = match format {
        ExportFormat::Jpeg { quality } => {
            pixels * jpeg_bits_per_pixel(f64::from(quality.get()), gradient, megapixels) / 8.0
        }
        ExportFormat::Tiff { bits, compression } => {
            let raw = pixels * 3.0 * f64::from(bits.bits()) / 8.0;
            raw * match compression {
                TiffCompression::None => 1.0,
                TiffCompression::Lzw => ratio(RATIO_FIT.lzw[bits_index(bits)]),
                TiffCompression::Zip => ratio(RATIO_FIT.zip[bits_index(bits)]),
            }
        }
        ExportFormat::Png { bits } => {
            pixels * 3.0 * f64::from(bits.bits()) / 8.0 * ratio(RATIO_FIT.png[bits_index(bits)])
        }
        ExportFormat::Dng => {
            // 16-bit raw, plus the full-size q92 JPEG preview and a 256 px thumbnail.
            pixels * 6.0
                + pixels
                    * jpeg_bits_per_pixel(f64::from(dng::PREVIEW_QUALITY), gradient, megapixels)
                    / 8.0
                + 200_000.0
        }
    };
    (bytes + 4096.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::JpegQuality;

    fn jpeg(q: u8) -> ExportFormat {
        ExportFormat::Jpeg {
            quality: JpegQuality::new(q).unwrap(),
        }
    }

    fn detail(g: f64) -> Option<DetailStat> {
        DetailStat::from_gradient(g)
    }

    #[test]
    fn estimate_is_monotone_and_sane() {
        let (w, h) = (8640, 5760);
        let mut last = 0;
        for q in 50..=100 {
            let e = estimate_bytes(w, h, jpeg(q), Size::Full);
            assert!(e >= last, "q{q}");
            last = e;
        }
        // 50 MP of typical detail at quality 92: a few tens of megabytes at most, more than at quality 75.
        let q92 = estimate_bytes(w, h, jpeg(92), Size::Full);
        assert!((5_000_000..40_000_000).contains(&q92), "{q92}");
        assert!(q92 > estimate_bytes(w, h, jpeg(75), Size::Full));
        let tiff16 = estimate_bytes(
            w,
            h,
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::None,
            },
            Size::Full,
        );
        assert!((tiff16 as i64 - 8640 * 5760 * 6).abs() < 10_000);
        let compressed = |compression| {
            estimate_bytes(
                w,
                h,
                ExportFormat::Tiff {
                    bits: BitDepth::Sixteen,
                    compression,
                },
                Size::Full,
            )
        };
        assert!(compressed(TiffCompression::Lzw) < tiff16 * 11 / 10);
        assert!(compressed(TiffCompression::Zip) < compressed(TiffCompression::Lzw));
        // A smaller long edge gives a smaller file; DNG ignores the size.
        let png = |size| {
            estimate_bytes(
                w,
                h,
                ExportFormat::Png {
                    bits: BitDepth::Eight,
                },
                size,
            )
        };
        assert!(png(Size::LongEdge { px: 2000 }) * 10 < png(Size::Full));
        let dng_full = estimate_bytes(w, h, ExportFormat::Dng, Size::Full);
        let dng_small = estimate_bytes(w, h, ExportFormat::Dng, Size::LongEdge { px: 2000 });
        assert_eq!(dng_full, dng_small);
        assert!(dng_full > 8640 * 5760 * 6);
    }

    #[test]
    fn more_detail_means_a_larger_file_in_every_format() {
        let formats = [
            jpeg(50),
            jpeg(92),
            jpeg(100),
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Lzw,
            },
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Zip,
            },
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
            ExportFormat::Dng,
        ];
        for format in formats {
            let at = |g: f64| estimate_bytes_with_detail(6000, 4000, format, Size::Full, detail(g));
            let (smooth, typical, busy) = (at(0.006), at(TYPICAL_GRADIENT), at(0.05));
            assert!(smooth < typical && typical < busy, "{format:?}");
            // No preview is the typical picture.
            assert_eq!(
                estimate_bytes(6000, 4000, format, Size::Full),
                typical,
                "{format:?}"
            );
        }
        // The spread for JPEG at quality 92 is large (the reason for the statistic): a busy frame is more than
        // twice a smooth one, not a few percent more.
        let at = |g: f64| estimate_bytes_with_detail(6000, 4000, jpeg(92), Size::Full, detail(g));
        assert!(at(0.05) > 2 * at(0.006));
    }

    #[test]
    fn the_statistic_is_clamped_so_extremes_stay_sane() {
        let at = |g: f64| estimate_bytes_with_detail(6000, 4000, jpeg(92), Size::Full, detail(g));
        assert_eq!(at(0.0), at(GRADIENT_RANGE.0));
        assert_eq!(at(1.0), at(GRADIENT_RANGE.1));
        assert_eq!(at(f64::MAX), at(GRADIENT_RANGE.1));
        // A busy frame is bounded by the bits-per-pixel cap: under 24 bits for each of the 24 MP.
        assert!(at(1.0) < 24_000_000 * 24 / 8 + 4096);
        assert!(DetailStat::from_gradient(f64::NAN).is_none());
        assert!(DetailStat::from_gradient(-0.1).is_none());
        assert!(DetailStat::from_gradient(f64::INFINITY).is_none());
        // Tiny frames and huge ones do not overflow or go negative.
        for (w, h) in [(1, 1), (16, 16), (65_535, 65_535)] {
            let e = estimate_bytes(w, h, jpeg(92), Size::Full);
            assert!(e >= 4096);
        }
    }

    #[test]
    fn jpeg_quality_between_knots_is_interpolated_in_the_log_domain() {
        let at = |q: u8| estimate_bytes(6000, 4000, jpeg(q), Size::Full) as f64;
        // Between the knots 60 and 75 the value of 65 is on the straight line of the logarithms.
        let expected = (at(60).ln() + (at(75).ln() - at(60).ln()) * (5.0 / 15.0)).exp();
        assert!((at(65) / expected - 1.0).abs() < 1e-3);
        // The knots are the fits themselves.
        for (q, fit) in JPEG_FIT {
            let bpp = fit
                .eval(TYPICAL_GRADIENT, 24.0)
                .clamp(BPP_RANGE.0, BPP_RANGE.1);
            let bytes = 6000.0 * 4000.0 * bpp / 8.0 + 4096.0;
            assert!(
                (at(q as u8) / bytes - 1.0).abs() < 1e-3,
                "q{q}: {} against {bytes}",
                at(q as u8)
            );
        }
    }

    /// A smooth gradient, 8-bit RGB.
    fn gradient_rgb8(w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let t = (x + y) as f64 / (w + h) as f64;
                v.extend_from_slice(&[(t * 200.0) as u8, (t * 180.0) as u8, (t * 160.0) as u8]);
            }
        }
        v
    }

    #[test]
    fn the_statistic_of_a_flat_picture_is_zero_and_of_a_ramp_is_its_slope() {
        let flat = vec![128u8; 240 * 160 * 3];
        assert_eq!(
            DetailStat::measure_rgb8(240, 160, &flat)
                .unwrap()
                .gradient(),
            0.0
        );
        // A horizontal ramp of 240 steps from 0 to 1 in luma: a neighbour differs by 1/239 in x, 0 in y.
        let mut ramp = Vec::new();
        for _ in 0..160 {
            for x in 0..240usize {
                let v = (x as f64 / 239.0 * 255.0).round() as u8;
                ramp.extend_from_slice(&[v, v, v]);
            }
        }
        let g = DetailStat::measure_rgb8(240, 160, &ramp)
            .unwrap()
            .gradient();
        assert!((g - 0.5 / 239.0).abs() < 2e-4, "{g}");
        // A smooth gradient has a low statistic, white noise a high one.
        let smooth = DetailStat::measure_rgb8(240, 160, &gradient_rgb8(240, 160)).unwrap();
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let noise: Vec<u8> = (0..240 * 160 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 56) as u8
            })
            .collect();
        let noisy = DetailStat::measure_rgb8(240, 160, &noise).unwrap();
        assert!(smooth.gradient() < 0.01, "{smooth:?}");
        assert!(noisy.gradient() > 0.2, "{noisy:?}");
    }

    #[test]
    fn large_previews_are_reduced_by_a_whole_factor_and_u8_and_f32_agree() {
        // A 960 x 640 picture is averaged 4 x 4 down to 240 x 160 before measuring.
        let (w, h) = (960usize, 640usize);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let rgb8: Vec<u8> = (0..w * h * 3)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let base = ((i / 3 % w) as f64 / w as f64 * 160.0) as u8;
                base.saturating_add((state >> 59) as u8)
            })
            .collect();
        let rgb32: Vec<f32> = rgb8.iter().map(|&v| f32::from(v) / 255.0).collect();
        let a = DetailStat::measure_rgb8(w as u32, h as u32, &rgb8).unwrap();
        let b = DetailStat::measure_encoded_f32(w as u32, h as u32, &rgb32).unwrap();
        assert!((a.gradient() / b.gradient() - 1.0).abs() < 1e-6);
        // The same picture reduced by hand to 240 x 160 and measured as it is gives the same number.
        let mut small = Vec::with_capacity(240 * 160 * 3);
        for ty in 0..160 {
            for tx in 0..240 {
                for c in 0..3 {
                    let mut sum = 0.0f32;
                    for y in 0..4 {
                        for x in 0..4 {
                            sum += rgb32[((ty * 4 + y) * w + tx * 4 + x) * 3 + c];
                        }
                    }
                    small.push(sum / 16.0);
                }
            }
        }
        let c = DetailStat::measure_encoded_f32(240, 160, &small).unwrap();
        assert!((a.gradient() / c.gradient() - 1.0).abs() < 1e-4);
        // Noise averaged away by the reduction: far lower than the statistic at full size would be.
        assert!(a.gradient() < 0.1);
    }

    #[test]
    fn unusable_previews_give_none() {
        assert!(DetailStat::measure_rgb8(0, 0, &[]).is_none());
        // A length that does not match the size.
        assert!(DetailStat::measure_rgb8(240, 160, &vec![0u8; 240 * 160 * 3 - 1]).is_none());
        assert!(DetailStat::measure_rgb8(240, 160, &vec![0u8; 240 * 160 * 3 + 3]).is_none());
        // Under 16 px on the short side.
        assert!(DetailStat::measure_rgb8(240, 15, &vec![0u8; 240 * 15 * 3]).is_none());
        assert!(DetailStat::measure_rgb8(15, 15, &vec![0u8; 15 * 15 * 3]).is_none());
        assert!(DetailStat::measure_encoded_f32(3, 3, &[0.0; 27]).is_none());
        // Sixteen is enough.
        assert!(DetailStat::measure_rgb8(16, 16, &vec![0u8; 16 * 16 * 3]).is_some());
        // A very wide strip whose short side is too short after the reduction.
        assert!(DetailStat::measure_rgb8(4800, 100, &vec![0u8; 4800 * 100 * 3]).is_none());
    }

    #[test]
    fn small_previews_are_corrected_to_the_analysis_size() {
        // The same scene as a 120 px and a 240 px preview. Per pixel the smaller one differs more (g ~ s^-0.25), so
        // its statistic is multiplied by (120 / 240)^0.25, which brings it back towards the 240 px figure.
        let big = gradient_rgb8(240, 160);
        let a = DetailStat::measure_rgb8(240, 160, &big).unwrap().gradient();
        let mut half = Vec::new();
        for y in 0..80 {
            for x in 0..120 {
                let p = &big[((y * 2) * 240 + x * 2) * 3..][..3];
                half.extend_from_slice(p);
            }
        }
        let b = DetailStat::measure_rgb8(120, 80, &half).unwrap().gradient();
        // On a ramp the per-pixel step doubles at half the size; the correction takes back a part of it.
        assert!(b > a);
        assert!(
            (b / a - 2.0 * (0.5f64).powf(SCALE_EXPONENT)).abs() < 0.05,
            "{a} {b}"
        );
    }
}
