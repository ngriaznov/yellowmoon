//! R6's single-frame denoise by one of the methods of `docs/algorithms/05-single-frame-denoise.md`, in place of
//! the bilateral denoise and R7 (*Integration* there).
//!
//! The stage takes the demosaiced, balanced linear camera RGB of R6, puts it in the working encoding
//! ([`denoise_space`](super::denoise_space)), runs the method and decodes the result. The controls of the
//! Detail tab keep their meaning relative to Auto, so that the settings, the Automatic card and the sliders are
//! the same for every method: with `r = L / L_auto(ISO)` and `c = C / 50` (the Auto values of `docs/05` R7),
//!
//! * method A: `L_A = r · kL · σY · 22.627 / 0.1` and `C_A = c · kC · σC · 22.627 / 0.06`, where `σY` and `σC`
//!   are the photo's luma and chroma noise in the working encoding at middle grey, from its noise model and
//!   gains ([`working_noise`]); `kL` and `kC` put the hard thresholds at that many standard deviations of the
//!   noise (calibrated by `examples/denoise_gate.rs`);
//! * method B: `L_B = 30 r` and method C: `L_C = 80 r`, `C_C = 90 c`, the settings measured best by the same
//!   tool (the text's "measured noise" settings are B's 50 and C's 80 and 50);
//! * every method: the blend of the spec is today's detail re-injection weight, `B = 100 · 0.15 (D / 50)^2`
//!   (15 % at Auto), and the radius `R` is 1.
//!
//! Receptive field: method A's 23 px ([`bm_dct::HALO_PX`]); methods B and C read the whole frame (a pyramid and a
//! measurement of the whole image), so a tile asks for the whole frame ([`FrameDenoise::halo_px`] is `None`).

use super::RgbImage;
use super::bm_dct::{self, BmDctParams};
use super::denoise_space::{WORKING_GAIN_EV, decode_image, encode_image, srgb_encode};
use super::detail::{C_AUTO, NoiseReduction, l_auto};
use super::nl_bayes::{self, NlBayesParams, Variant};
use super::ycbcr::{CB_DIV, CR_DIV, KB, KG, KR};
use ym_noise::model::NoiseModel;

/// The denoise of R6 (an internal switch of the renderer, like the demosaic method; `docs/05` R6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DenoiseMethod {
    /// Today's light bilateral denoise and R7 (colour guided filter, detail re-injection).
    #[default]
    Bilateral,
    /// Method A: block matching and 3D DCT hard thresholding.
    BmDct,
    /// Method B: multiscale NL-Bayes in RGB with the intensity noise model.
    NlBayesRgb,
    /// Method C: multiscale NL-Bayes in Y/Cb/Cr.
    NlBayesYcc,
}

impl DenoiseMethod {
    /// Every method.
    pub const ALL: [DenoiseMethod; 4] = [
        DenoiseMethod::Bilateral,
        DenoiseMethod::BmDct,
        DenoiseMethod::NlBayesRgb,
        DenoiseMethod::NlBayesYcc,
    ];

    /// The name the switches use.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DenoiseMethod::Bilateral => "bilateral",
            DenoiseMethod::BmDct => "bm_dct",
            DenoiseMethod::NlBayesRgb => "nl_bayes_rgb",
            DenoiseMethod::NlBayesYcc => "nl_bayes_ycc",
        }
    }

    /// The method of a [`name`](Self::name).
    #[must_use]
    pub fn from_name(name: &str) -> Option<DenoiseMethod> {
        DenoiseMethod::ALL.into_iter().find(|m| m.name() == name)
    }

    /// Whether the method reads the whole frame (a tile asks for all of it).
    #[must_use]
    pub const fn needs_whole_frame(self) -> bool {
        matches!(self, DenoiseMethod::NlBayesRgb | DenoiseMethod::NlBayesYcc)
    }
}

/// Method A's luma threshold in standard deviations of the photo's luma noise (`kL`), measured by
/// `examples/denoise_gate.rs` (the median of the five ISO-like levels' best on Kodak; within 0.12 dB of the best
/// at every level). The noise is the model's per photosite; the demosaic correlates it, which the constant absorbs.
pub const BM_DCT_KL: f64 = 3.5;
/// Method A's chroma threshold in standard deviations of the photo's chroma noise (`kC`), measured the same way
/// (the levels' best are 16, 8, 8, 6 and 6 from ISO 100 to 12800; 8 is within 0.2 dB of each).
pub const BM_DCT_KC: f64 = 8.0;
/// Method B's `L` at Auto: the measured best (30), below the text's "measured noise" 50, because method B's
/// measurement reads high on detail (reading BC-i).
pub const NL_BAYES_RGB_L: f64 = 30.0;
/// Method C's `L` at Auto (the measured best, the text's 80).
pub const NL_BAYES_YCC_L: f64 = 80.0;
/// Method C's `C` at Auto (the measured best).
pub const NL_BAYES_YCC_C: f64 = 90.0;
/// `L` of method A per standard deviation of luma noise: `tau = 0.1 L` on the scale `8^1.5` times the
/// orthonormal one.
pub const L_PER_SIGMA: f64 = 22.627_416_997_969_52 / 0.1;
/// `C` of method A per standard deviation of chroma noise (`tau = 0.06 C`).
pub const C_PER_SIGMA: f64 = 22.627_416_997_969_52 / 0.06;
/// The radius `R` of every method.
pub const RADIUS: usize = 1;
/// Middle grey on the raw scale at the camera's exposure (`0.18 / 2^CAMERA_GAIN_EV`).
fn grey_raw() -> f64 {
    0.18 / WORKING_GAIN_EV.exp2()
}

/// The photo's luma and chroma noise in the working encoding at middle grey, `(σY, σC)`, from its noise model
/// (variance `a x + b` on the normalised raw scale) and the R4 gains `g0`: a balanced channel at grey has the
/// variance `g a x + g² b`, the encoding's slope at grey scales it, and Y, Cb and Cr weigh the channels (taken
/// as independent; the demosaic correlates them, which the calibration of `kL` and `kC` absorbs).
#[must_use]
pub fn working_noise(model: &NoiseModel, g0: [f32; 3]) -> (f64, f64) {
    let x = grey_raw();
    let g = WORKING_GAIN_EV.exp2();
    // d/dx srgb(g x) at grey, numerically (the curve is smooth there).
    let h = 1e-6;
    let slope = (srgb_encode(g * (x + h)) - srgb_encode(g * (x - h))) / (2.0 * h);
    let var: Vec<f64> = g0
        .iter()
        .map(|&gc| {
            let gc = f64::from(gc);
            (gc * model.a * x + gc * gc * model.b).max(0.0) * slope * slope
        })
        .collect();
    let weigh =
        |w: [f64; 3]| (w[0] * w[0] * var[0] + w[1] * w[1] * var[1] + w[2] * w[2] * var[2]).sqrt();
    let sy = weigh([KR, KG, KB]);
    let sb = weigh([-KR / CB_DIV, -KG / CB_DIV, (1.0 - KB) / CB_DIV]);
    let sr = weigh([(1.0 - KR) / CR_DIV, -KG / CR_DIV, -KB / CR_DIV]);
    (sy, ((sb * sb + sr * sr) / 2.0).sqrt())
}

/// The resolved stage: the method and the spec's strength inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDenoise {
    /// The method (never [`DenoiseMethod::Bilateral`], which is the chain's own stages).
    pub method: DenoiseMethod,
    /// `L`.
    pub luma: f64,
    /// `C`.
    pub chroma: f64,
    /// `R`.
    pub radius: usize,
    /// `B` (percent).
    pub blend: f64,
}

impl FrameDenoise {
    /// The stage for `method` and the photo's controls, ISO, noise model and gains (module documentation);
    /// `None` for [`DenoiseMethod::Bilateral`].
    #[must_use]
    pub fn resolve(
        method: DenoiseMethod,
        nr: &NoiseReduction,
        iso: f32,
        model: &NoiseModel,
        g0: [f32; 3],
    ) -> Option<FrameDenoise> {
        let r = f64::from(nr.luminance / l_auto(iso));
        let c = f64::from(nr.color / C_AUTO);
        let (luma, chroma) = match method {
            DenoiseMethod::Bilateral => return None,
            DenoiseMethod::BmDct => {
                let (sy, sc) = working_noise(model, g0);
                (
                    r * BM_DCT_KL * sy * L_PER_SIGMA,
                    c * BM_DCT_KC * sc * C_PER_SIGMA,
                )
            }
            DenoiseMethod::NlBayesRgb => (NL_BAYES_RGB_L * r, 0.0),
            DenoiseMethod::NlBayesYcc => (NL_BAYES_YCC_L * r, NL_BAYES_YCC_C * c),
        };
        Some(FrameDenoise {
            method,
            luma: luma.clamp(0.0, 100.0),
            chroma: chroma.clamp(0.0, 100.0),
            radius: RADIUS,
            blend: f64::from(nr.beta()) * 100.0,
        })
    }

    /// Context on each side that a tile needs, `None` for the whole frame.
    #[must_use]
    pub const fn halo_px(&self) -> Option<usize> {
        match self.method {
            DenoiseMethod::BmDct => Some(bm_dct::HALO_PX),
            _ => None,
        }
    }

    /// Denoises `img` (balanced linear camera RGB; a window of the frame with [`halo_px`](Self::halo_px) of
    /// context, or the whole frame).
    #[must_use]
    pub fn apply(&self, img: &RgbImage) -> RgbImage {
        let w = encode_image(img, WORKING_GAIN_EV);
        let out = match self.method {
            DenoiseMethod::Bilateral => return img.clone(),
            DenoiseMethod::BmDct => bm_dct::denoise(
                &w,
                &BmDctParams {
                    luma: self.luma,
                    chroma: self.chroma,
                    radius: self.radius,
                    blend: self.blend,
                },
            ),
            DenoiseMethod::NlBayesRgb | DenoiseMethod::NlBayesYcc => nl_bayes::denoise(
                &w,
                &NlBayesParams {
                    variant: if self.method == DenoiseMethod::NlBayesRgb {
                        Variant::B
                    } else {
                        Variant::C
                    },
                    luma: self.luma,
                    chroma: self.chroma,
                    radius: self.radius,
                    blend: self.blend,
                },
            ),
        };
        decode_image(&out, WORKING_GAIN_EV)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth;

    fn auto(iso: f32) -> NoiseReduction {
        NoiseReduction::auto(iso)
    }

    #[test]
    fn names_round_trip_and_only_b_and_c_read_the_whole_frame() {
        for m in DenoiseMethod::ALL {
            assert_eq!(DenoiseMethod::from_name(m.name()), Some(m));
        }
        assert_eq!(DenoiseMethod::from_name("median"), None);
        assert_eq!(DenoiseMethod::default(), DenoiseMethod::Bilateral);
        assert!(!DenoiseMethod::BmDct.needs_whole_frame());
        assert!(DenoiseMethod::NlBayesRgb.needs_whole_frame());
        assert!(DenoiseMethod::NlBayesYcc.needs_whole_frame());
    }

    #[test]
    fn at_auto_each_method_gets_its_own_strengths_and_the_detail_blend() {
        let model = NoiseModel::new(4e-4, 6e-7);
        let g0 = [2.0, 1.0, 1.5];
        assert_eq!(
            FrameDenoise::resolve(DenoiseMethod::Bilateral, &auto(1600.0), 1600.0, &model, g0),
            None
        );
        let b = FrameDenoise::resolve(DenoiseMethod::NlBayesRgb, &auto(1600.0), 1600.0, &model, g0)
            .expect("B");
        assert_eq!((b.luma, b.chroma, b.radius), (NL_BAYES_RGB_L, 0.0, RADIUS));
        // Detail 50 is today's re-injection weight 0.15: a blend of 15 %.
        assert!((b.blend - 15.0).abs() < 1e-5);
        let c = FrameDenoise::resolve(DenoiseMethod::NlBayesYcc, &auto(1600.0), 1600.0, &model, g0)
            .expect("C");
        assert_eq!((c.luma, c.chroma), (NL_BAYES_YCC_L, NL_BAYES_YCC_C));
        let a = FrameDenoise::resolve(DenoiseMethod::BmDct, &auto(1600.0), 1600.0, &model, g0)
            .expect("A");
        let (sy, sc) = working_noise(&model, g0);
        assert!((a.luma - BM_DCT_KL * sy * L_PER_SIGMA).abs() < 1e-9);
        assert!((a.chroma - BM_DCT_KC * sc * C_PER_SIGMA).abs() < 1e-9);
    }

    #[test]
    fn the_controls_scale_each_method_relative_to_auto() {
        let model = NoiseModel::new(4e-4, 6e-7);
        let g0 = [1.0; 3];
        let at = |l: f32, c: f32, d: f32, m| {
            let nr = NoiseReduction {
                luminance: l,
                color: c,
                detail: d,
            };
            FrameDenoise::resolve(m, &nr, 1600.0, &model, g0).expect("method")
        };
        // ISO 1600: L_auto 50. Twice the Auto luminance doubles B's gain; Colour 0 turns C's chroma off.
        let auto_l = l_auto(1600.0);
        assert_eq!(
            at(2.0 * auto_l, 50.0, 50.0, DenoiseMethod::NlBayesRgb).luma,
            2.0 * NL_BAYES_RGB_L
        );
        assert_eq!(at(auto_l, 0.0, 50.0, DenoiseMethod::NlBayesYcc).chroma, 0.0);
        assert_eq!(at(0.0, 50.0, 50.0, DenoiseMethod::BmDct).luma, 0.0);
        // Detail 0 is no blend, 100 the 60 % of today's rule.
        assert_eq!(at(auto_l, 50.0, 0.0, DenoiseMethod::BmDct).blend, 0.0);
        assert!((at(auto_l, 50.0, 100.0, DenoiseMethod::BmDct).blend - 60.0).abs() < 1e-4);
        // Strengths stay in the spec's range (C's 80 twice over is 160).
        assert_eq!(
            at(100.0, 100.0, 50.0, DenoiseMethod::NlBayesYcc).luma,
            100.0
        );
    }

    #[test]
    fn the_working_noise_follows_the_model_and_the_gains() {
        let low = working_noise(&NoiseModel::new(1e-4, 1e-7), [1.0; 3]);
        let high = working_noise(&NoiseModel::new(4e-4, 4e-7), [1.0; 3]);
        // Four times the variance is twice the deviation.
        assert!((high.0 / low.0 - 2.0).abs() < 1e-9);
        assert!((high.1 / low.1 - 2.0).abs() < 1e-9);
        // Gains above 1 on red and blue add noise to Y and more to the chroma.
        let gained = working_noise(&NoiseModel::new(1e-4, 1e-7), [2.0, 1.0, 1.6]);
        assert!(gained.0 > low.0 && gained.1 / low.1 > gained.0 / low.0);
        assert_eq!(
            working_noise(&NoiseModel::new(0.0, 0.0), [1.0; 3]),
            (0.0, 0.0)
        );
    }

    #[test]
    fn method_a_removes_noise_from_a_flat_patch_in_linear_light() {
        let (w, h) = (96, 64);
        let grey = 0.18 / WORKING_GAIN_EV.exp2();
        let mut img = RgbImage::new(w, h);
        img.data.iter_mut().for_each(|v| *v = grey as f32);
        let clean = img.clone();
        let model = NoiseModel::new(4e-4, 6e-7);
        let sigma = (model.a * grey + model.b).sqrt() as f32;
        let mut rng = synth::Rng::new(3);
        img.data
            .iter_mut()
            .for_each(|v| *v += sigma * rng.gaussian());
        let fd = FrameDenoise::resolve(
            DenoiseMethod::BmDct,
            &auto(1600.0),
            1600.0,
            &model,
            [1.0; 3],
        )
        .expect("A");
        let out = fd.apply(&img);
        assert_eq!((out.width, out.height, out.place), (w, h, img.place));
        let rms = |a: &RgbImage| {
            let n = a.data.len() as f64;
            (a.data
                .iter()
                .zip(&clean.data)
                .map(|(x, c)| f64::from(x - c).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
        };
        assert!(
            rms(&out) < 0.5 * rms(&img),
            "{} against {}",
            rms(&out),
            rms(&img)
        );
    }
}
