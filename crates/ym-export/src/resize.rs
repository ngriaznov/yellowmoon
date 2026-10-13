//! Lanczos-3 resize in linear light (`docs/09-export.md` section Pixel pipeline per format).
//!
//! The export resizes **before** the display encode, on linear RGB, so averaging happens in light
//! and not in code values (resizing encoded values darkens fine detail and edges). The filter is
//! separable Lanczos with a = 3, sized by the scale factor when shrinking (so it also filters
//! aliasing), with taps outside the image dropped and the remaining weights renormalised to sum to 1.
//! That is the same convolution as Pillow's `LANCZOS` filter, which the tests use as an
//! independent check.
//!
//! Accumulation is in f64, so a flat field comes out bit-exact; the intermediate image is f32.
//! Every output row is computed independently, so the result does not depend on the thread count.
//! Lanczos overshoots at hard edges (about 7 % at a step); nothing is clamped here, the display
//! encode clamps to [0, 1].

use std::f64::consts::PI;

use rayon::prelude::*;
use ym_color::encode::OutputSpace;
use ym_gpu::cpu::output_sharpen::OutputSharpen;

use crate::error::ExportError;
use crate::pixels::{ExportHooks, ImageRef, LinearImage, PixelEncoding};

/// The Lanczos window half-width `a`.
pub const LANCZOS_A: f64 = 3.0;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = x * PI;
        px.sin() / px
    }
}

/// The Lanczos-3 kernel.
#[must_use]
pub fn lanczos3(x: f64) -> f64 {
    if (-LANCZOS_A..LANCZOS_A).contains(&x) {
        sinc(x) * sinc(x / LANCZOS_A)
    } else {
        0.0
    }
}

/// Per output index: the first input index and its normalised taps.
struct Taps {
    start: Vec<usize>,
    offsets: Vec<usize>,
    weights: Vec<f64>,
}

impl Taps {
    fn new(in_len: usize, out_len: usize) -> Self {
        let scale = in_len as f64 / out_len as f64;
        let filter_scale = scale.max(1.0);
        let support = LANCZOS_A * filter_scale;
        let mut start = Vec::with_capacity(out_len);
        let mut offsets = Vec::with_capacity(out_len + 1);
        let mut weights = Vec::new();
        offsets.push(0);
        for i in 0..out_len {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = (((center + support + 0.5).floor()) as usize).min(in_len);
            let first = weights.len();
            let mut sum = 0.0;
            for x in lo..hi {
                let w = lanczos3((x as f64 - center + 0.5) / filter_scale);
                weights.push(w);
                sum += w;
            }
            if sum != 0.0 {
                for w in &mut weights[first..] {
                    *w /= sum;
                }
            }
            start.push(lo);
            offsets.push(weights.len());
        }
        Self {
            start,
            offsets,
            weights,
        }
    }

    fn row(&self, i: usize) -> (usize, &[f64]) {
        (
            self.start[i],
            &self.weights[self.offsets[i]..self.offsets[i + 1]],
        )
    }
}

/// Resizes `src` to `out_width` x `out_height` with Lanczos-3 in linear light.
///
/// `src` must be display-linear, or display-encoded in `space` (it is decoded to linear light
/// on the fly). The result is display-linear. `space` is only used for the decode of encoded input.
pub fn resize_lanczos3(
    src: &ImageRef<'_>,
    out_width: u32,
    out_height: u32,
    space: OutputSpace,
    hooks: &ExportHooks<'_>,
) -> Result<LinearImage, ExportError> {
    src.validate()?;
    if src.encoding == PixelEncoding::SceneLinearAwg4 {
        return Err(ExportError::invalid(
            "scene-linear AWG4 pixels can't be resized as a display image",
        ));
    }
    if out_width == 0 || out_height == 0 {
        return Err(ExportError::invalid("the output size is zero"));
    }
    let (in_w, in_h) = (src.width as usize, src.height as usize);
    let (out_w, out_h) = (out_width as usize, out_height as usize);
    let xs = Taps::new(in_w, out_w);
    let ys = Taps::new(in_h, out_h);
    let mid_row = out_w * 3;

    // Horizontal pass: every input row to `out_w` samples.
    let mut mid = vec![0.0_f32; mid_row * in_h];
    mid.par_chunks_mut(mid_row).enumerate().for_each_init(
        || vec![0.0_f32; in_w * 3],
        |scratch, (y, dst)| {
            if y % 64 == 0 && hooks.is_cancelled() {
                return;
            }
            let row = src.row(y);
            let row: &[f32] = if src.encoding == PixelEncoding::DisplayEncoded {
                for (s, d) in row.iter().zip(scratch.iter_mut()) {
                    *d = space.decode_channel(f64::from(*s)) as f32;
                }
                scratch
            } else {
                row
            };
            for (x, out) in dst.chunks_exact_mut(3).enumerate() {
                let (first, weights) = xs.row(x);
                let mut acc = [0.0_f64; 3];
                for (k, &w) in weights.iter().enumerate() {
                    let p = &row[(first + k) * 3..(first + k) * 3 + 3];
                    acc[0] += w * f64::from(p[0]);
                    acc[1] += w * f64::from(p[1]);
                    acc[2] += w * f64::from(p[2]);
                }
                out[0] = acc[0] as f32;
                out[1] = acc[1] as f32;
                out[2] = acc[2] as f32;
            }
        },
    );
    hooks.check()?;
    hooks.report(0.5);

    // Vertical pass: combine rows of the intermediate image.
    let mut out = vec![0.0_f32; mid_row * out_h];
    out.par_chunks_mut(mid_row).enumerate().for_each_init(
        || vec![0.0_f64; mid_row],
        |acc, (i, dst)| {
            if i % 64 == 0 && hooks.is_cancelled() {
                return;
            }
            acc.fill(0.0);
            let (first, weights) = ys.row(i);
            for (k, &w) in weights.iter().enumerate() {
                let row = &mid[(first + k) * mid_row..(first + k + 1) * mid_row];
                for (a, &v) in acc.iter_mut().zip(row) {
                    *a += w * f64::from(v);
                }
            }
            for (d, &a) in dst.iter_mut().zip(acc.iter()) {
                *d = a as f32;
            }
        },
    );
    hooks.check()?;
    hooks.report(1.0);
    Ok(LinearImage {
        width: out_width,
        height: out_height,
        rgb: out,
    })
}

/// Core rows of one strip of [`sharpen_output`]: enough for the stage's own bands (128 rows each)
/// to fill every core, few enough that a cancel is noticed within a second or two and that the
/// working planes stay a few tens of megabytes at 8000 px wide.
const SHARPEN_STRIP_ROWS: usize = 1024;

/// Output sharpening of the picture at the size it is written (`docs/algorithms/04`, the Detail
/// tab's "Output" slider): the screen preset at `amount_pct`, run right after the resize.
///
/// `scale` is `out_long_edge / rendered_long_edge` ([`ym_gpu::cpu::output_sharpen::scale_for_edges`]:
/// 1 for a full-size export). `src` is display-linear, or display-encoded in `space` (decoded to
/// linear light first, as the resize does); the result is display-linear, interleaved, the size of
/// `src`. `None` when the amount creates no stage (0, or below 0.1 %): the caller then keeps
/// `src`, so the bytes are exactly those of an export without the stage.
///
/// The stage runs in strips of rows with the stage's own receptive field of context above and
/// below ([`OutputSharpen::halo_px`], the tiling contract of spec section 5.2), which gives the
/// whole-frame result bit for bit, a cancel check and a progress report per strip, and no
/// full-frame working planes.
///
/// # Errors
///
/// [`ExportError::InvalidInput`] for an amount or a scale outside the stage's range,
/// [`ExportError::Cancelled`] when the hooks say so.
pub fn sharpen_output(
    src: &ImageRef<'_>,
    amount_pct: f32,
    scale: f32,
    space: OutputSpace,
    hooks: &ExportHooks<'_>,
) -> Result<Option<LinearImage>, ExportError> {
    src.validate()?;
    if src.encoding == PixelEncoding::SceneLinearAwg4 {
        return Err(ExportError::invalid(
            "scene-linear AWG4 pixels can't be sharpened as a display image",
        ));
    }
    let sharpen = OutputSharpen::screen().with_amount_pct(amount_pct);
    let invalid =
        |e: ym_gpu::cpu::output_sharpen::OutputSharpenError| ExportError::invalid(e.to_string());
    let plan = sharpen.plan(scale).map_err(invalid)?;
    if plan.stages().is_empty() {
        return Ok(None);
    }
    let halo = plan.halo_px();
    let (w, h) = (src.width as usize, src.height as usize);
    let row_len = w * 3;
    let mut out = vec![0.0_f32; row_len * h];
    let mut tmp: Vec<f32> = Vec::new();
    let mut y0 = 0;
    while y0 < h {
        hooks.check()?;
        let y1 = (y0 + SHARPEN_STRIP_ROWS).min(h);
        let (top, bottom) = (y0.saturating_sub(halo), (y1 + halo).min(h));
        tmp.clear();
        for y in top..bottom {
            let row = src.row(y);
            if src.encoding == PixelEncoding::DisplayEncoded {
                tmp.extend(
                    row.iter()
                        .map(|&v| space.decode_channel(f64::from(v)) as f32),
                );
            } else {
                tmp.extend_from_slice(row);
            }
        }
        sharpen
            .apply_interleaved_in_place(w, bottom - top, &mut tmp, scale)
            .map_err(invalid)?;
        out[y0 * row_len..y1 * row_len]
            .copy_from_slice(&tmp[(y0 - top) * row_len..(y1 - top) * row_len]);
        y0 = y1;
        hooks.report(y0 as f32 / h as f32);
    }
    hooks.check()?;
    Ok(Some(LinearImage {
        width: src.width,
        height: src.height,
        rgb: out,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear<'a>(w: u32, h: u32, data: &'a [f32]) -> ImageRef<'a> {
        ImageRef::new(w, h, data, PixelEncoding::DisplayLinear).unwrap()
    }

    #[test]
    fn kernel_values() {
        assert_eq!(lanczos3(0.0), 1.0);
        for n in [-2.0, -1.0, 1.0, 2.0] {
            assert!(lanczos3(n).abs() < 1e-15, "zero at integers: {n}");
        }
        assert_eq!(lanczos3(3.0), 0.0);
        assert_eq!(lanczos3(-3.5), 0.0);
        assert!((lanczos3(0.5) - 0.607_927_101_854_026_6).abs() < 1e-12);
        assert!(lanczos3(1.5) < 0.0 && lanczos3(2.5) > 0.0);
    }

    #[test]
    fn weights_sum_to_one_everywhere() {
        for (i, o) in [(100, 37), (64, 64), (1000, 999), (10, 3), (7, 70)] {
            let t = Taps::new(i, o);
            for x in 0..o {
                let (first, w) = t.row(x);
                let sum: f64 = w.iter().sum();
                assert!((sum - 1.0).abs() < 1e-14, "{i}->{o} at {x}: {sum}");
                assert!(first + w.len() <= i);
            }
        }
    }

    #[test]
    fn flat_field_is_exact() {
        let (w, h) = (97_u32, 61_u32);
        for value in [0.0_f32, 0.18, 1.0, 0.123_456_79, 7.5] {
            let data = vec![value; (w * h * 3) as usize];
            for (ow, oh) in [(40, 25), (97, 61), (300, 200), (1, 1), (50, 61)] {
                let out = resize_lanczos3(
                    &linear(w, h, &data),
                    ow,
                    oh,
                    OutputSpace::Srgb,
                    &ExportHooks::none(),
                )
                .unwrap();
                assert_eq!(out.width, ow);
                assert_eq!(out.height, oh);
                assert!(
                    out.rgb.iter().all(|&v| v == value),
                    "flat {value} -> {ow}x{oh} not exact"
                );
            }
        }
    }

    #[test]
    fn identity_size_keeps_the_pixels_within_rounding() {
        let (w, h) = (16_u32, 12_u32);
        let data: Vec<f32> = (0..w * h * 3)
            .map(|i| ((i * 31) % 97) as f32 / 97.0)
            .collect();
        let out = resize_lanczos3(
            &linear(w, h, &data),
            w,
            h,
            OutputSpace::Srgb,
            &ExportHooks::none(),
        )
        .unwrap();
        for (a, b) in out.rgb.iter().zip(&data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn mean_is_preserved_and_channels_stay_separate() {
        let (w, h) = (64_u32, 48_u32);
        let mut data = vec![0.0_f32; (w * h * 3) as usize];
        for p in data.chunks_exact_mut(3) {
            p[0] = 1.0; // red only
        }
        let out = resize_lanczos3(
            &linear(w, h, &data),
            21,
            16,
            OutputSpace::Srgb,
            &ExportHooks::none(),
        )
        .unwrap();
        for p in out.rgb.chunks_exact(3) {
            assert_eq!(p[0], 1.0);
            assert_eq!(p[1], 0.0);
            assert_eq!(p[2], 0.0);
        }
    }

    #[test]
    fn step_edge_overshoot_is_bounded() {
        // A hard 0 -> 1 edge shrunk 4x: Lanczos-3 rings, the overshoot must stay under 10 %.
        let (w, h) = (256_u32, 8_u32);
        let mut data = vec![0.0_f32; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 128..w as usize {
                let i = (y * w as usize + x) * 3;
                data[i..i + 3].fill(1.0);
            }
        }
        let out = resize_lanczos3(
            &linear(w, h, &data),
            64,
            2,
            OutputSpace::Srgb,
            &ExportHooks::none(),
        )
        .unwrap();
        let (lo, hi) = out
            .rgb
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(hi > 1.0, "a Lanczos filter overshoots at an edge: {hi}");
        assert!(hi <= 1.10 && lo >= -0.10, "overshoot {hi} undershoot {lo}");
    }

    /// A zone plate has every spatial frequency up to the Nyquist limit; after a 2x or 3x shrink the
    /// frequencies above the new Nyquist limit alias, but the output must not ring beyond the input range.
    #[test]
    fn zone_plate_ringing_is_bounded() {
        let n = 384_usize;
        let mut data = vec![0.0_f32; n * n * 3];
        let c = n as f64 / 2.0;
        for y in 0..n {
            for x in 0..n {
                let r2 = (x as f64 - c).powi(2) + (y as f64 - c).powi(2);
                // Local frequency grows linearly with radius; it reaches 0.5 cycle/pixel at the corner.
                let k = PI / (2.0 * c * std::f64::consts::SQRT_2);
                let v = 0.5 + 0.5 * (k * r2).cos();
                let i = (y * n + x) * 3;
                data[i..i + 3].fill(v as f32);
            }
        }
        for factor in [2_usize, 3, 5] {
            let o = n / factor;
            let out = resize_lanczos3(
                &linear(n as u32, n as u32, &data),
                o as u32,
                o as u32,
                OutputSpace::Srgb,
                &ExportHooks::none(),
            )
            .unwrap();
            let (lo, hi) = out
                .rgb
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            // The input spans [0, 1]; a Lanczos-3 shrink stays within a 10 % ringing margin of that.
            assert!(lo >= -0.10 && hi <= 1.10, "x{factor}: [{lo}, {hi}]");
            // And a low-frequency centre stays near the input (the plate's centre is flat white).
            let mid = out.rgb[((o / 2) * o + o / 2) * 3];
            assert!(mid > 0.9, "x{factor}: centre {mid}");
        }
    }

    #[test]
    fn encoded_input_is_resized_in_linear_light() {
        // Black and white stripes, encoded: the average must be mid grey in LIGHT (0.5 linear is
        // sRGB code ~0.735), not 0.5 in code values.
        let (w, h) = (64_u32, 4_u32);
        let mut data = vec![0.0_f32; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in (0..w as usize).step_by(2) {
                let i = (y * w as usize + x) * 3;
                data[i..i + 3].fill(1.0);
            }
        }
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayEncoded).unwrap();
        let out = resize_lanczos3(&img, 4, 1, OutputSpace::Srgb, &ExportHooks::none()).unwrap();
        let mid = out.rgb[3];
        assert!((mid - 0.5).abs() < 0.02, "linear light average {mid}");
    }

    #[test]
    fn scene_linear_input_is_refused_and_cancel_works() {
        let data = vec![0.0_f32; 4 * 4 * 3];
        let awg = ImageRef::new(4, 4, &data, PixelEncoding::SceneLinearAwg4).unwrap();
        assert!(resize_lanczos3(&awg, 2, 2, OutputSpace::Srgb, &ExportHooks::none()).is_err());
        let always = || true;
        let hooks = ExportHooks::new(Some(&always), None);
        let big = vec![0.5_f32; 200 * 200 * 3];
        let r = resize_lanczos3(&linear(200, 200, &big), 50, 50, OutputSpace::Srgb, &hooks);
        assert!(matches!(r, Err(ExportError::Cancelled)));
    }

    proptest::proptest! {
        /// For any pair of sizes every output pixel's weights add up to one and stay inside the input.
        #[test]
        fn weights_always_normalise(input in 1_usize..400, output in 1_usize..400) {
            let t = Taps::new(input, output);
            for x in 0..output {
                let (first, w) = t.row(x);
                let sum: f64 = w.iter().sum();
                proptest::prop_assert!((sum - 1.0).abs() < 1e-13, "{input}->{output} at {x}: {sum}");
                proptest::prop_assert!(!w.is_empty() && first + w.len() <= input);
            }
        }
    }
}
