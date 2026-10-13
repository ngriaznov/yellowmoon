//! Measurement code of the RA-05 demosaic gate (`docs/algorithms/TASKS.md` RA-05, `docs/05` R6),
//! shared by `examples/ra05_gate.rs` (the tables) and `tests/cpu_demosaic_metrics.rs` (the unit
//! tests of the metrics). Included with `#[path]`, so every user sees the same code.
//!
//! # Protocol
//!
//! * **PSNR** follows `docs/05`: a ground-truth RGB image in linear light is mosaicked as RGGB
//!   (no optical low-pass), demosaiced, and compared with [`ym_gpu::cpu::metrics::psnr`] over the
//!   interior (a 10 px border excluded), peak = the signal scale. The sets are 8-bit sRGB images,
//!   so "linear light" means the sRGB electro-optical transfer function applied first
//!   ([`linearize`]); [`Protocol::Gamma`] keeps the 8-bit values as they are (the older numbers of
//!   `docs/05`). Signal scales 1, 0.1 and 0.01 multiply the linear image (and the peak), as
//!   `docs/05` did for LMMSE. The mean is the mean of the per-image dB values.
//! * **Noise model.** The adaptive ratio demosaic and the log-guided demosaic read a noise model.
//!   The gate uses the nominal low-noise model of spec 01 section 6.3 (`a = 2e-5`, `b = 1e-8`,
//!   `q = 1/65535`, unit gains) for the adaptive ratio demosaic, kept fixed whatever the signal
//!   scale (a darker image shot at the same ISO), and the zero model for the log-guided demosaic
//!   (`p = 1`, spec 02 section 6.4). Three sensitivity variants say how much that choice matters.
//! * **Zipper and false colour.** `cpu/metrics.rs` has PSNR, Lab, CIEDE2000 and
//!   [`chroma_delta_e00`](ym_gpu::cpu::metrics::chroma_delta_e00) but no zipper or false-colour
//!   metric (the spec text that says so is wrong), so they are defined here: see [`artifacts`].

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::metrics::{delta_e_2000, lab_from_camera_rgb, psnr};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use ym_noise::model::NoiseModel;

/// Border excluded from every metric, in pixels (`docs/05`: 10 px).
pub const BORDER_PX: usize = 10;

/// Signal scales of the PSNR tables: the linear image times each of these (`docs/05`).
pub const SCALES: [f32; 3] = [1.0, 0.1, 0.01];

/// An edge pixel of the zipper metric: some 8-neighbour differs from it by more than this CIE
/// 1976 colour difference in the ground truth.
pub const EDGE_DELTA_E: f64 = 10.0;

/// A zipper pixel: the colour difference to its nearest-in-colour neighbour grew by more than this
/// (CIE 1976 units, about one just-noticeable difference) in the demosaiced image.
pub const ZIPPER_DELTA_E: f64 = 2.3;

/// A false-colour pixel: the chroma difference to the ground truth (CIEDE2000, lightness equalised,
/// `docs/05` R5's measure) is above this. The same 5 as the R5 gate's "chroma within dE00 5".
pub const FALSE_COLOUR_DELTA_E00: f64 = 5.0;

// ---------------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------------

/// How the 8-bit sRGB test images become ground truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    /// The sRGB electro-optical transfer function is applied: scene-linear light, which is what
    /// the pipeline demosaics (`docs/05` R6 "in linear light").
    Linear,
    /// The 8-bit values divided by 255, as they are (the gamma-encoded numbers of `docs/05`'s first
    /// table).
    Gamma,
}

/// The sRGB electro-optical transfer function (encoded value in `[0, 1]` to linear light).
pub fn srgb_to_linear(v: f32) -> f32 {
    let v = f64::from(v);
    let lin = if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    };
    lin as f32
}

/// `img` in linear light (sRGB decoded).
pub fn linearize(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = srgb_to_linear(*v);
    }
    out
}

/// `img` times `scale`.
pub fn scaled(img: &RgbImage, scale: f32) -> RgbImage {
    let mut out = img.clone();
    for v in &mut out.data {
        *v *= scale;
    }
    out
}

/// The procedural set of `tests/cpu_demosaic.rs` (`synthetic_set` there), image for image.
pub fn synthetic_set() -> Vec<(String, RgbImage)> {
    vec![
        ("zone plate".to_owned(), synth::zone_plate(192, 192)),
        ("edges".to_owned(), synth::edges(192, 192)),
        ("colour patches".to_owned(), synth::color_patches(192, 192)),
        ("fine text".to_owned(), synth::fine_text(192, 192, 1)),
        ("natural 1".to_owned(), synth::natural(192, 192, 1)),
        ("natural 2".to_owned(), synth::natural(192, 192, 2)),
        ("natural 3".to_owned(), synth::natural(192, 192, 3)),
    ]
}

/// `testdata/demosaic` (override the root with `YM_TESTDATA`).
pub fn demosaic_sets_root() -> PathBuf {
    std::env::var_os("YM_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"))
        .join("demosaic")
}

/// An 8-bit PNG as an image with values in `[0, 1]` (as they are, not decoded).
pub fn load_png(path: &Path) -> RgbImage {
    let file = std::fs::File::open(path).expect("open png");
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .expect("png info");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let channels = info.color_type.samples();
    let rgb: Vec<f32> = buf[..info.width as usize * info.height as usize * channels]
        .chunks(channels)
        .flat_map(|p| {
            if channels >= 3 {
                [p[0], p[1], p[2]]
            } else {
                [p[0], p[0], p[0]]
            }
        })
        .map(|v| f32::from(v) / 255.0)
        .collect();
    RgbImage::from_interleaved(info.width as usize, info.height as usize, &rgb).expect("image")
}

/// An 8-bit RGB TIFF as an image with values in `[0, 1]`.
pub fn load_tiff(path: &Path) -> RgbImage {
    let file = std::fs::File::open(path).expect("open tiff");
    let mut decoder =
        tiff::decoder::Decoder::new(std::io::BufReader::new(file)).expect("tiff decoder");
    let (w, h) = decoder.dimensions().expect("tiff dims");
    let data = match decoder.read_image().expect("tiff data") {
        tiff::decoder::DecodingResult::U8(v) => v,
        other => panic!("unexpected tiff sample type {other:?}"),
    };
    let rgb: Vec<f32> = data.iter().map(|&v| f32::from(v) / 255.0).collect();
    RgbImage::from_interleaved(w as usize, h as usize, &rgb).expect("image")
}

/// Every PNG and TIFF of `dir`, sorted by file name (an empty list when the folder is missing).
pub fn load_set(dir: &Path) -> Vec<(String, RgbImage)> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().to_string();
            match p.extension()?.to_str()? {
                "png" => Some((name, load_png(&p))),
                "tif" | "tiff" => Some((name, load_tiff(&p))),
                _ => None,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Methods and their inputs
// ---------------------------------------------------------------------------------------------

/// Which noise model a method is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    /// Zero model (`demosaic()`'s default): the adaptive ratio demosaic gets K = T = 1, the
    /// log-guided demosaic p = 1.
    Zero,
    /// The nominal low-noise model of spec 01 section 6.3, the same at every signal scale.
    Nominal,
    /// The nominal model scaled with the signal (`a·λ`, `b·λ²`, `q·λ`, spec 01 section 6.2's scale
    /// test): the method sees the same relative noise at every scale.
    NominalScaled,
}

/// One column of the tables: a method and the noise model it is run with.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    /// Column label.
    pub label: &'static str,
    /// The demosaic method.
    pub method: Method,
    /// Its noise model.
    pub noise: Noise,
}

/// The columns of the gate: the four methods the rule compares, and the log-guided preview method.
pub const PRIMARY: [Variant; 5] = [
    Variant {
        label: "bilinear",
        method: Method::Bilinear,
        noise: Noise::Zero,
    },
    Variant {
        label: "menon",
        method: Method::Menon,
        noise: Noise::Zero,
    },
    Variant {
        label: "lmmse",
        method: Method::Lmmse,
        noise: Noise::Zero,
    },
    Variant {
        label: "adaptive_ratio",
        method: Method::AdaptiveRatio,
        noise: Noise::Nominal,
    },
    Variant {
        label: "log_guided",
        method: Method::LogGuided,
        noise: Noise::Zero,
    },
];

/// How much the noise model matters: the same two methods under the other models.
pub const SENSITIVITY: [Variant; 3] = [
    Variant {
        label: "ar zero model",
        method: Method::AdaptiveRatio,
        noise: Noise::Zero,
    },
    Variant {
        label: "ar scaled model",
        method: Method::AdaptiveRatio,
        noise: Noise::NominalScaled,
    },
    Variant {
        label: "lg nominal",
        method: Method::LogGuided,
        noise: Noise::Nominal,
    },
];

/// The nominal low-noise model (`a = 2e-5`, `b = 1e-8`, `q = 1/65535`, no row noise) scaled by
/// `lambda` as spec 01 section 6.2 scales it (`a·λ`, `b·λ²`, `q·λ`).
pub fn nominal_model(lambda: f64) -> NoiseModel {
    NoiseModel {
        a: 2e-5 * lambda,
        b: 1e-8 * lambda * lambda,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: lambda / 65535.0,
    }
}

/// The inputs `noise` stands for at signal scale `scale`: unit gains, no CA.
pub fn inputs(noise: Noise, scale: f32) -> DemosaicInputs<'static> {
    let noise = match noise {
        Noise::Zero => NoiseDn::ZERO,
        Noise::Nominal => NoiseDn::from_model(&nominal_model(1.0)),
        Noise::NominalScaled => NoiseDn::from_model(&nominal_model(f64::from(scale))),
    };
    DemosaicInputs {
        gains: [1.0; 3],
        noise,
        ca: None,
    }
}

/// Demosaics `m` (made at signal scale `scale`) the way `variant` says.
pub fn run_variant(m: &Mosaic, variant: Variant, scale: f32) -> RgbImage {
    demosaic::demosaic_with(m, variant.method, &inputs(variant.noise, scale))
}

/// PSNR (dB) of each of `variants` on one linear-light ground-truth image at `scale`: the image is
/// scaled, mosaicked as RGGB, demosaiced, and compared over the interior with the peak scaled
/// along.
pub fn psnr_row(truth_linear: &RgbImage, scale: f32, variants: &[Variant]) -> Vec<f64> {
    let truth = scaled(truth_linear, scale);
    let m = synth::mosaic_from_rgb(&truth, Cfa::RGGB);
    variants
        .iter()
        .map(|v| {
            psnr(
                &run_variant(&m, *v, scale),
                &truth,
                BORDER_PX,
                f64::from(scale),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Zipper and false colour
// ---------------------------------------------------------------------------------------------

/// An image in CIE Lab (D65), through the stand-in matrix of `cpu/metrics.rs` (the sets are sRGB
/// images, so Rec. 709 primaries are exactly right for them). Values are clamped to `[0, 1]`
/// first, as `psnr` does.
pub struct LabImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major Lab triplets.
    pub lab: Vec<[f64; 3]>,
}

impl LabImage {
    /// Converts `img` pixel by pixel.
    pub fn new(img: &RgbImage) -> LabImage {
        let (r, g, b) = (img.plane(0), img.plane(1), img.plane(2));
        let lab = r
            .iter()
            .zip(g)
            .zip(b)
            .map(|((r, g), b)| {
                lab_from_camera_rgb([r, g, b].map(|v| f64::from(*v).clamp(0.0, 1.0)))
            })
            .collect();
        LabImage {
            width: img.width,
            height: img.height,
            lab,
        }
    }
}

/// CIE 1976 colour difference.
fn delta_e76(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// CIEDE2000 with the lightness forced equal, so only chroma and hue count:
/// `chroma_delta_e00` of `cpu/metrics.rs` on Lab values that are already computed.
fn chroma_de00(a: [f64; 3], b: [f64; 3]) -> f64 {
    let l = 0.5 * (a[0] + b[0]);
    delta_e_2000([l, a[1], a[2]], [l, b[1], b[2]])
}

/// The artefact numbers of one demosaiced image against its ground truth.
#[derive(Clone, Copy, Debug)]
pub struct Artifacts {
    /// Edge pixels of the ground truth in the interior (the denominator of [`Self::zipper_pct`]).
    pub edge_px: usize,
    /// Zipper rate, percent of edge pixels; `NaN` when the image has no edge pixel.
    pub zipper_pct: f64,
    /// False-colour rate, percent of interior pixels.
    pub false_colour_pct: f64,
    /// Mean chroma difference to the ground truth (CIEDE2000, lightness equalised) over the
    /// interior.
    pub chroma_de00_mean: f64,
}

/// Zipper and false-colour numbers of `demo` against `truth` over the interior (`border` pixels
/// excluded on every side).
///
/// * **Zipper** (after Lu and Tan, "Color filter array demosaicking: new method and performance
///   measures", IEEE TIP 2003): for every *edge pixel* of the ground truth (one with an
///   8-neighbour more than [`EDGE_DELTA_E`] away in CIE 1976 Lab), find the neighbour nearest to
///   it in colour in the truth; the pixel is a zipper pixel when the colour difference to that same
///   neighbour grew by more than [`ZIPPER_DELTA_E`] in `demo`. The rate is the percentage of edge
///   pixels. A method that interpolates along an edge keeps the difference; one that interpolates
///   across it makes neighbours of one colour disagree, the alternating "zipper".
/// * **False colour**: the chroma difference to the ground truth (CIEDE2000 with the lightness
///   equalised, the measure of the R5 gate) above [`FALSE_COLOUR_DELTA_E00`]; the rate is the
///   percentage of interior pixels, and the mean difference is reported too.
///
/// The ground truth alone decides which pixels are edge pixels and which neighbour is nearest, so
/// the denominators are the same for every method.
pub fn artifacts(truth: &LabImage, demo: &LabImage, border: usize) -> Artifacts {
    assert_eq!(
        (truth.width, truth.height),
        (demo.width, demo.height),
        "artifacts: dimension mismatch"
    );
    let (w, h) = (truth.width, truth.height);
    let border = border.max(1);
    const NEIGHBOURS: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    let (mut edge, mut zipper, mut false_colour, mut n) = (0usize, 0usize, 0usize, 0usize);
    let mut chroma_sum = 0.0f64;
    for y in border..h.saturating_sub(border) {
        for x in border..w.saturating_sub(border) {
            let p = y * w + x;
            let (mut nearest, mut nearest_d, mut farthest_d) = (p, f64::INFINITY, 0.0f64);
            for (dx, dy) in NEIGHBOURS {
                let q = (y as isize + dy) as usize * w + (x as isize + dx) as usize;
                let d = delta_e76(truth.lab[p], truth.lab[q]);
                farthest_d = farthest_d.max(d);
                if d < nearest_d {
                    (nearest, nearest_d) = (q, d);
                }
            }
            if farthest_d > EDGE_DELTA_E {
                edge += 1;
                let grown = delta_e76(demo.lab[p], demo.lab[nearest]) - nearest_d;
                if grown > ZIPPER_DELTA_E {
                    zipper += 1;
                }
            }
            let de = chroma_de00(truth.lab[p], demo.lab[p]);
            chroma_sum += de;
            if de > FALSE_COLOUR_DELTA_E00 {
                false_colour += 1;
            }
            n += 1;
        }
    }
    Artifacts {
        edge_px: edge,
        zipper_pct: if edge == 0 {
            f64::NAN
        } else {
            100.0 * zipper as f64 / edge as f64
        },
        false_colour_pct: if n == 0 {
            f64::NAN
        } else {
            100.0 * false_colour as f64 / n as f64
        },
        chroma_de00_mean: if n == 0 {
            f64::NAN
        } else {
            chroma_sum / n as f64
        },
    }
}

/// [`Artifacts`] of each of `variants` on one linear-light ground-truth image at signal scale 1.
pub fn artifacts_row(truth_linear: &RgbImage, variants: &[Variant]) -> Vec<Artifacts> {
    let m = synth::mosaic_from_rgb(truth_linear, Cfa::RGGB);
    let truth_lab = LabImage::new(truth_linear);
    variants
        .iter()
        .map(|v| {
            let demo = run_variant(&m, *v, 1.0);
            artifacts(&truth_lab, &LabImage::new(&demo), BORDER_PX)
        })
        .collect()
}

/// Mean of the finite values of `values` (`NaN` when there are none).
pub fn mean_finite(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for v in values {
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_decode_has_the_right_anchors() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        // 8-bit code 128 is 0.2158 in linear light, code 188 (0.7373) is 0.5 (sRGB, IEC 61966-2-1).
        assert!((srgb_to_linear(128.0 / 255.0) - 0.21586).abs() < 1e-4);
        assert!((srgb_to_linear(0.735_357) - 0.5).abs() < 1e-4);
        // The toe is linear: 12.92 times the encoded value.
        assert!((srgb_to_linear(0.02) - 0.02 / 12.92).abs() < 1e-9);
    }

    #[test]
    fn identical_images_have_no_artifacts() {
        let img = synth::edges(64, 64);
        let lab = LabImage::new(&img);
        let a = artifacts(&lab, &lab, BORDER_PX);
        assert!(a.edge_px > 0, "the edges image has edge pixels");
        assert_eq!(a.zipper_pct, 0.0);
        assert_eq!(a.false_colour_pct, 0.0);
        assert_eq!(a.chroma_de00_mean, 0.0);
    }

    #[test]
    fn a_flat_image_has_no_edge_pixels() {
        let mut img = RgbImage::new(40, 40);
        img.data.fill(0.4);
        let lab = LabImage::new(&img);
        let a = artifacts(&lab, &lab, BORDER_PX);
        assert_eq!(a.edge_px, 0);
        assert!(a.zipper_pct.is_nan(), "no edge pixels, no rate");
        assert_eq!(a.false_colour_pct, 0.0);
    }

    /// A vertical edge between a red and a blue half; the "demosaiced" image gets a one-pixel
    /// staggered edge (every other row shifted by one column), the textbook zipper.
    #[test]
    fn a_staggered_edge_is_a_zipper_and_a_clean_edge_is_not() {
        let (w, h) = (48, 48);
        let build = |stagger: bool| {
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let edge_x = if stagger && y % 2 == 1 { 25 } else { 24 };
                    let c = if x < edge_x {
                        [0.8, 0.1, 0.1]
                    } else {
                        [0.1, 0.1, 0.8]
                    };
                    img.set_pixel(x, y, c);
                }
            }
            img
        };
        let truth = LabImage::new(&build(false));
        let clean = artifacts(&truth, &truth, BORDER_PX);
        let zip = artifacts(&truth, &LabImage::new(&build(true)), BORDER_PX);
        assert_eq!(clean.zipper_pct, 0.0);
        // Column 24 of the odd rows holds the red of the staggered edge where the truth has blue:
        // its nearest-in-colour truth neighbour (the blue to its right) is now far away. Half of
        // the rows are affected and the edge pixels of the truth are columns 23 and 24.
        assert!(
            zip.zipper_pct > 20.0,
            "staggered edge: zipper {:.1} %",
            zip.zipper_pct
        );
        assert!(zip.false_colour_pct > 0.0);
    }

    #[test]
    fn false_colour_ignores_a_pure_lightness_change() {
        let mut a = RgbImage::new(40, 40);
        a.data.fill(0.3);
        let mut b = a.clone();
        for v in &mut b.data {
            *v *= 1.5; // grey stays grey: no chroma change
        }
        let r = artifacts(&LabImage::new(&a), &LabImage::new(&b), BORDER_PX);
        // The stand-in matrix is neutral only to its printed digits: a residue of a few 1e-6.
        assert!(r.chroma_de00_mean < 1e-3, "grey to grey: {r:?}");
        assert_eq!(r.false_colour_pct, 0.0);
    }

    /// The cached-Lab form must agree with `cpu/metrics.rs`'s `chroma_delta_e00`.
    #[test]
    fn chroma_de00_matches_the_metrics_module() {
        let mut rng = synth::Rng::new(5);
        for _ in 0..200 {
            let a: [f64; 3] = std::array::from_fn(|_| f64::from(rng.next_f32()));
            let b: [f64; 3] = std::array::from_fn(|_| f64::from(rng.next_f32()));
            let want = ym_gpu::cpu::metrics::chroma_delta_e00(a, b);
            let got = chroma_de00(lab_from_camera_rgb(a), lab_from_camera_rgb(b));
            assert!((got - want).abs() < 1e-9, "{a:?} {b:?}: {got} vs {want}");
        }
    }

    #[test]
    fn nominal_model_scales_like_the_spec() {
        let one = nominal_model(1.0);
        let tenth = nominal_model(0.1);
        assert_eq!(one.a, 2e-5);
        assert_eq!(one.b, 1e-8);
        assert!((one.q - 1.0 / 65535.0).abs() < 1e-15);
        assert!((tenth.a - 2e-6).abs() < 1e-18);
        assert!((tenth.b - 1e-10).abs() < 1e-22);
        assert!((tenth.q - 0.1 / 65535.0).abs() < 1e-16);
    }

    #[test]
    fn mean_finite_skips_nan() {
        assert_eq!(mean_finite([1.0, f64::NAN, 3.0]), 2.0);
        assert!(mean_finite([f64::NAN]).is_nan());
    }
}
