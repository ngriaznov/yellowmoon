//! The classical raw-processing chain R1 to R9 on the CPU, whole frame or one tile at a time.
//!
//! This is the CPU fallback path (`docs/05-raw-processing.md` section R6 Classical, "CPU-only
//! fallback export"), the reference the GPU chain is checked against, and the harness for the
//! tiled-versus-whole tests: [`Chain::run_tile`] walks the stages backwards from the wanted output
//! rectangle, cuts the needed rectangle out of the raw frame, runs the stages forward cropping to
//! what the next stage needs, and returns the core. A whole-frame run is one tile.
//!
//! Stage order: R1 normalise, R3 vignetting, R4 gains, R5 highlights (with chroma offsets
//! computed once per frame by [`Chain::prepare`]), R6 demosaic then the light bilateral denoise,
//! R7 colour guided filter and detail re-injection, R8 capture sharpening (Richardson-Lucy or
//! diffraction deconvolution, [`CaptureSharpen`]), R9 lens resample. The
//! neural path replaces R6 and the luminance part of R7 (`ym-ml`); everything else is shared.
//!
//! In the classical path the "network output" of R7 is the denoised image and `Y_classical` is
//! the plain demosaic before the denoise, so the Detail control puts back a fraction of what the
//! denoise removed. The Luminance control scales the denoise strength with the same
//! `(L / L_auto)^1.5` rule that scales the network's σ.

use super::demosaic::{self, CaCurves, DemosaicInputs, Method};
use super::denoise::{self, BilateralParams};
use super::detail::{self, NoiseReduction};
use super::gains;
use super::highlights::{self, ChromaOffsets};
use super::lens::LensGeometry;
use super::normalize::{self, Levels};
use super::sharpen::diffraction::{DiffractionPlan, SPEC_LUMA};
use super::sharpen::{self, CaptureSharpen};
use super::tiling::{Rect, Stage};
use super::vignette::{self, VignetteProfile};
use super::{Cfa, ClipMask, CpuError, Mosaic, NoiseDn, NoiseSigma, Placement, RgbImage};

/// A raw frame (the active area) as the decoder delivers it.
#[derive(Clone, Copy, Debug)]
pub struct RawFrame<'a> {
    /// Width in photosites.
    pub width: usize,
    /// Height in photosites.
    pub height: usize,
    /// The pattern at the first photosite.
    pub cfa: Cfa,
    /// Raw values, row-major.
    pub data: &'a [u16],
    /// Quantisation step per photosite in raw units (cRAW), if any.
    pub qstep: Option<&'a [f32]>,
}

/// The per-frame settings of the classical chain.
#[derive(Clone, Debug)]
pub struct Chain {
    /// R1 black and white levels per CFA position.
    pub levels: Levels,
    /// R3 vignetting profile (`None`: skipped, as with `decode.lens_corrections = false`).
    pub vignette: Option<VignetteProfile>,
    /// R4 gains `g0` (red, green, blue).
    pub gains: [f32; 3],
    /// R5 on or off (`decode.highlight_recovery`).
    pub highlight_recovery: bool,
    /// R6 classical method. [`Method::AdaptiveRatio`] reads [`gains`](Self::gains) and
    /// [`noise`](Self::noise): R6 gets the balanced mosaic of R4/R5 and the method undoes R4
    /// itself (`docs/algorithms/INTEGRATION.md` section 4), so the output is balanced for every
    /// method.
    pub method: Method,
    /// R6 bilateral denoise (`None`: skipped).
    pub denoise: Option<BilateralParams>,
    /// R7 resolved controls.
    pub noise_reduction: NoiseReduction,
    /// ISO of the shot (sets `L_auto` for the Luminance rule).
    pub iso: f32,
    /// Estimated luminance noise σ, in balanced linear units, scalar.
    pub sigma: f32,
    /// Optional σ map for the whole frame (`width * height`), used instead of `sigma` when present.
    pub sigma_map: Option<Vec<f32>>,
    /// The noise model of the shot on the 16-bit working scale
    /// ([`NoiseDn::from_model`], `docs/algorithms/INTEGRATION.md` section 3), `None` when it is not
    /// known (read as zero noise). It is for the methods of the raw-algorithms package that derive
    /// their constants from the noise: the R6 adaptive ratio demosaic (and the log-guided one,
    /// which takes its read noise from it) and R8 diffraction deconvolution.
    pub noise: Option<NoiseDn>,
    /// R8 capture sharpening: the method and its controls (`None`: skipped). A diffraction stage
    /// that is inactive for these controls and gains (no usable pitch or f-number) leaves the image
    /// unchanged; the caller that wants Richardson-Lucy in that case chooses it when it builds the
    /// chain (`docs/algorithms/03-diffraction-deconvolution.md` section 1.2).
    pub sharpen: Option<CaptureSharpen>,
    /// R9 geometry (`None`: no resample; the output is the whole frame).
    pub lens: Option<LensGeometry>,
}

/// Whole-frame quantities computed once before any tile runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prepared {
    /// R5 global chroma offsets (zero when R5 is off).
    pub offsets: ChromaOffsets,
}

impl Chain {
    /// The stages of this chain, in processing order, for halo bookkeeping.
    pub fn stages(&self) -> Vec<Stage<'_>> {
        let mut s = vec![Stage::Normalize];
        if self.vignette.is_some() {
            s.push(Stage::Vignette);
        }
        s.push(Stage::Gains);
        if self.highlight_recovery {
            s.push(Stage::Highlights);
        }
        s.push(Stage::Demosaic(self.method));
        if let Some(d) = &self.denoise {
            s.push(Stage::Denoise {
                radius_px: d.radius_px,
            });
        }
        // The classical image the detail term needs is kept alongside, so no extra halo.
        s.push(Stage::Detail {
            chroma_radius_px: self.noise_reduction.chroma_radius_px(),
            needs_classical: false,
        });
        if let Some(method) = self.sharpen {
            s.push(Stage::Sharpen(method));
        }
        if let Some(l) = &self.lens {
            s.push(Stage::Lens(l));
        }
        s
    }

    /// Size of the output image: the lens crop, or the frame without a lens stage.
    pub fn output_size(&self, frame_w: usize, frame_h: usize) -> (usize, usize) {
        self.lens
            .as_ref()
            .map_or((frame_w, frame_h), LensGeometry::output_size)
    }

    /// The rectangle of the raw frame (even-aligned) that [`run_tile`](Self::run_tile) reads for `out`.
    pub fn required_raw_rect(&self, frame_w: usize, frame_h: usize, out: Rect) -> Rect {
        let stages = self.stages();
        super::tiling::required_input_rect(&stages, out, Rect::whole(frame_w, frame_h))
    }

    /// What R6 gets besides the mosaic: the R4 gains, the noise model (zero when it is not known)
    /// and the lateral CA curves `ca` ([`ca_curves`](Self::ca_curves)).
    fn demosaic_inputs<'a>(&self, ca: Option<&'a CaCurves>) -> DemosaicInputs<'a> {
        DemosaicInputs {
            gains: self.gains,
            noise: self.noise.unwrap_or(NoiseDn::ZERO),
            ca,
        }
    }

    /// The lateral CA curves for the CA-aware green refinement of the adaptive ratio demosaic
    /// (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 3.6, RA-03b): the R9 CA of
    /// [`lens`](Self::lens), sampled by [`LensGeometry::ca_curves`]. `None` (the plain path) for
    /// every other method, for a chain without a lens stage, for a lens without CA (no 0x7035
    /// table, or `decode.chromatic_aberration` off, which `LensGeometry::new` already applied) and
    /// for a distortion table the curves cannot be sampled from. The curves belong to the frame, not
    /// the tile, so every tile of a frame sees the same ones.
    fn ca_curves(&self) -> Option<CaCurves> {
        if self.method != Method::AdaptiveRatio {
            return None;
        }
        self.lens.as_ref()?.ca_curves().ok().flatten()
    }

    /// R1 for `rect` of the frame, then R3 and R4 (all pointwise).
    fn mosaic_of(
        &self,
        raw: &RawFrame<'_>,
        rect: Rect,
        with_gains: bool,
    ) -> Result<(Mosaic, ClipMask), CpuError> {
        if raw.data.len() != raw.width * raw.height {
            return Err(CpuError::BadLength {
                got: raw.data.len(),
                expected: raw.width * raw.height,
                width: raw.width,
                height: raw.height,
            });
        }
        let mut tile = Vec::with_capacity(rect.area());
        let mut qtile = raw.qstep.map(|_| Vec::with_capacity(rect.area()));
        for y in rect.y..rect.y1() {
            tile.extend_from_slice(&raw.data[y * raw.width + rect.x..y * raw.width + rect.x1()]);
            if let (Some(q), Some(qt)) = (raw.qstep, qtile.as_mut()) {
                qt.extend_from_slice(&q[y * raw.width + rect.x..y * raw.width + rect.x1()]);
            }
        }
        let place = Placement::whole(raw.width, raw.height).offset(rect.x, rect.y);
        let (mut m, mask) = normalize::normalize_tile(
            &tile,
            rect.w,
            rect.h,
            place,
            raw.cfa.shifted(rect.x, rect.y),
            &self.levels,
            qtile.as_deref(),
        )?;
        if let Some(v) = &self.vignette {
            vignette::apply_vignette(&mut m, v);
        }
        if with_gains {
            gains::apply_gains(&mut m, self.gains);
        }
        Ok((m, mask))
    }

    /// Checks that the raw frame and the per-pixel inputs of this chain agree in size, so that no
    /// stage indexes past the end of a buffer.
    fn validate(&self, raw: &RawFrame<'_>) -> Result<(), CpuError> {
        let n = raw.width * raw.height;
        let bad = |got: usize| CpuError::BadLength {
            got,
            expected: n,
            width: raw.width,
            height: raw.height,
        };
        if raw.data.len() != n {
            return Err(bad(raw.data.len()));
        }
        if let Some(q) = raw.qstep
            && q.len() != n
        {
            return Err(bad(q.len()));
        }
        if let Some(m) = &self.sigma_map
            && m.len() != n
        {
            return Err(bad(m.len()));
        }
        if let Some(l) = &self.lens
            && (l.frame_width(), l.frame_height()) != (raw.width, raw.height)
        {
            return Err(CpuError::Mismatch(format!(
                "lens geometry is for a {}x{} frame, the raw frame is {}x{}",
                l.frame_width(),
                l.frame_height(),
                raw.width,
                raw.height
            )));
        }
        // The adaptive ratio demosaic divides by the gains and derives its constants from the
        // noise model. `demosaic_with` cannot fail (it would quietly run Menon instead), so a
        // chain that asks for it checks the numbers here, once, and says what is wrong.
        if self.method == Method::AdaptiveRatio
            && let Err(e) = self.demosaic_inputs(None).validate()
        {
            return Err(CpuError::Mismatch(format!(
                "the adaptive ratio demosaic cannot run: {e}"
            )));
        }
        self.levels.validate()
    }

    /// Computes the whole-frame reductions (the R5 chroma offsets).
    pub fn prepare(&self, raw: &RawFrame<'_>) -> Result<Prepared, CpuError> {
        self.validate(raw)?;
        if !self.highlight_recovery {
            return Ok(Prepared::default());
        }
        let (m, mask) = self.mosaic_of(raw, Rect::whole(raw.width, raw.height), true)?;
        Ok(Prepared {
            offsets: highlights::chroma_offsets(&m, &mask)?,
        })
    }

    /// Runs the whole frame.
    pub fn run(&self, raw: &RawFrame<'_>) -> Result<RgbImage, CpuError> {
        let prepared = self.prepare(raw)?;
        let (w, h) = self.output_size(raw.width, raw.height);
        self.run_tile(raw, &prepared, Rect::whole(w, h))
    }

    /// σ for the tile `rect`: the scalar, or the matching part of the σ map, scaled by `scale`.
    fn sigma_tile(&self, raw_width: usize, rect: Rect, scale: f32) -> (Option<Vec<f32>>, f32) {
        match &self.sigma_map {
            Some(map) => {
                let mut v = Vec::with_capacity(rect.area());
                for y in rect.y..rect.y1() {
                    v.extend(
                        map[y * raw_width + rect.x..y * raw_width + rect.x1()]
                            .iter()
                            .map(|s| s * scale),
                    );
                }
                (Some(v), 0.0)
            }
            None => (None, self.sigma * scale),
        }
    }

    /// Runs the part of the output covered by `out` (output coordinates: those of the lens crop,
    /// or of the frame when there is no lens stage) and returns exactly that rectangle.
    pub fn run_tile(
        &self,
        raw: &RawFrame<'_>,
        prepared: &Prepared,
        out: Rect,
    ) -> Result<RgbImage, CpuError> {
        self.validate(raw)?;
        let frame = Rect::whole(raw.width, raw.height);
        let (ow, oh) = self.output_size(raw.width, raw.height);
        if out.x1() > ow || out.y1() > oh {
            return Err(CpuError::RectOutside {
                rect: out,
                width: ow,
                height: oh,
            });
        }
        let stages = self.stages();
        // need[i] is the rectangle stage i must receive; need[stages.len()] is the output.
        let mut need = vec![out; stages.len() + 1];
        for i in (0..stages.len()).rev() {
            need[i] = stages[i].input_rect(need[i + 1], frame);
        }

        let mut cur = need[0];
        let (mut mosaic, mut mask) = self.mosaic_of(raw, cur, false)?; // R1 (+ R3 when the profile is set)
        let mut rgb: Option<RgbImage> = None;
        let mut classical: Option<RgbImage> = None;

        let take = |rgb: &mut Option<RgbImage>| -> Result<RgbImage, CpuError> {
            rgb.take()
                .ok_or_else(|| CpuError::Mismatch("a stage ran before the demosaic".into()))
        };
        for (i, stage) in stages.iter().enumerate() {
            match stage {
                Stage::Normalize | Stage::Vignette => {} // done in mosaic_of
                Stage::Gains => gains::apply_gains(&mut mosaic, self.gains),
                Stage::Highlights => {
                    highlights::apply_offsets(&mut mosaic, &mask, &prepared.offsets)?
                }
                Stage::Demosaic(method) => {
                    // R6 hands the balanced mosaic of R4/R5 to every method; the adaptive ratio
                    // demosaic undoes R4 inside `demosaic_with` (INTEGRATION.md section 4). The
                    // gains and the noise are per frame, never per tile, so every tile of a
                    // frame sees the same inputs. `Y_classical` of R7 is a copy of this output.
                    let ca = self.ca_curves();
                    rgb = Some(demosaic::demosaic_with(
                        &mosaic,
                        *method,
                        &self.demosaic_inputs(ca.as_ref()),
                    ));
                }
                Stage::Denoise { .. } => {
                    let img = take(&mut rgb)?;
                    let scale =
                        detail::network_sigma_scale(self.noise_reduction.luminance, self.iso);
                    let (plane, scalar) = self.sigma_tile(raw.width, cur, scale);
                    let ns = plane
                        .as_deref()
                        .map_or(NoiseSigma::Scalar(scalar), NoiseSigma::Plane);
                    if self.noise_reduction.needs_classical() {
                        classical = Some(img.clone());
                    }
                    let params = self.denoise.unwrap_or_default();
                    rgb = Some(denoise::bilateral_opponent(&img, &ns, &params));
                }
                Stage::Detail { .. } => {
                    let img = take(&mut rgb)?;
                    let (plane, scalar) = self.sigma_tile(raw.width, cur, 1.0);
                    let ns = plane
                        .as_deref()
                        .map_or(NoiseSigma::Scalar(scalar), NoiseSigma::Plane);
                    let mut colored = if self.noise_reduction.color > 0.0 {
                        detail::chroma_guided_filter(&img, &ns, self.noise_reduction.color)
                    } else {
                        img
                    };
                    if let Some(cl) = classical.as_ref() {
                        colored = detail::inject_detail(&colored, cl, self.noise_reduction.beta())?;
                    }
                    rgb = Some(colored);
                }
                Stage::Sharpen(CaptureSharpen::RichardsonLucy(params)) => {
                    let img = take(&mut rgb)?;
                    let (plane, scalar) = self.sigma_tile(raw.width, cur, 1.0);
                    let ns = plane
                        .as_deref()
                        .map_or(NoiseSigma::Scalar(scalar), NoiseSigma::Plane);
                    rgb = Some(sharpen::richardson_lucy(&img, params, &ns));
                }
                Stage::Sharpen(CaptureSharpen::Diffraction(params)) => {
                    let img = take(&mut rgb)?;
                    let noise = self.noise.unwrap_or(NoiseDn::ZERO);
                    // The plan comes from the controls, the gains and the noise model, none of which
                    // depends on the tile, so every tile of a frame gets the same plan. Inactive
                    // (a bad gain): the stage is an identity.
                    rgb = Some(
                        match DiffractionPlan::new(params, self.gains, &noise, SPEC_LUMA) {
                            Some(plan) => plan.apply(&img),
                            None => img,
                        },
                    );
                }
                Stage::Lens(geo) => {
                    let img = take(&mut rgb)?;
                    return super::resample::resample_lens(&img, geo, out);
                }
            }
            // Crop to what the next stage needs (its input rectangle).
            let next = need[i + 1];
            if next != cur {
                let r = next.relative_to(&cur);
                match rgb.as_mut() {
                    None => {
                        mosaic = mosaic.extract(r);
                        mask = mask.extract(r);
                    }
                    Some(img) => {
                        *img = img.extract(r);
                        if let Some(cl) = classical.as_mut() {
                            *cl = cl.extract(r);
                        }
                    }
                }
                cur = next;
            }
        }
        debug_assert_eq!(cur, out);
        rgb.ok_or_else(|| CpuError::Mismatch("chain produced no image".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::sharpen::SharpenParams;
    use crate::cpu::synth;
    use crate::cpu::tiling::tile_grid;

    fn test_chain(w: usize, h: usize, with_lens: bool) -> Chain {
        let tags = crate::cpu::lens::LensTags {
            vignetting: vec![0, -400, -1500, -3500, -6000],
            distortion: vec![0, 30, 100, 220, 400, 650],
            ca: vec![0, 100, 400, 900, 0, -100, -400, -900],
        };
        let lens = if with_lens {
            Some(
                LensGeometry::new(w, h, &tags, crate::cpu::lens::LensSwitches::default())
                    .expect("lens")
                    .with_auto_crop(),
            )
        } else {
            None
        };
        Chain {
            levels: Levels {
                black: [512.0, 514.0, 513.0, 511.0],
                white: [16000.0; 4],
            },
            vignette: Some(VignetteProfile::from_tag(&tags.vignetting).expect("vignette")),
            gains: [2.0, 1.0, 1.6],
            highlight_recovery: true,
            method: Method::Lmmse,
            denoise: Some(BilateralParams::default()),
            noise_reduction: NoiseReduction {
                luminance: 40.0,
                color: 60.0,
                detail: 70.0,
            },
            iso: 800.0,
            sigma: 0.01,
            sigma_map: None,
            noise: None,
            sharpen: Some(
                SharpenParams {
                    amount: 90.0,
                    radius_px: 0.8,
                }
                .into(),
            ),
            lens,
        }
    }

    fn raw_of(w: usize, h: usize, seed: u64) -> Vec<u16> {
        // A natural-looking scene exposed so that some highlights clip, plus noise.
        let img = synth::natural(w, h, seed);
        let m = synth::mosaic_from_rgb(&img, Cfa::RGGB);
        let mut rng = synth::Rng::new(seed + 7);
        m.data
            .iter()
            .map(|&v| {
                let x = 512.0 + (v * 1.3 + 0.004 * rng.gaussian()) * 15488.0;
                x.clamp(0.0, 16000.0) as u16
            })
            .collect()
    }

    #[test]
    fn tiles_reproduce_the_whole_frame_with_every_stage_enabled() {
        let (w, h) = (150, 110);
        let data = raw_of(w, h, 3);
        let raw = RawFrame {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data: &data,
            qstep: None,
        };
        for with_lens in [false, true] {
            let chain = test_chain(w, h, with_lens);
            let prepared = chain.prepare(&raw).expect("prepare");
            let (ow, oh) = chain.output_size(w, h);
            let whole = chain
                .run_tile(&raw, &prepared, Rect::whole(ow, oh))
                .expect("whole");
            let mut tiled = RgbImage::new(ow, oh);
            for core in tile_grid(ow, oh, 52, 46, 1) {
                let t = chain.run_tile(&raw, &prepared, core).expect("tile");
                assert_eq!((t.width, t.height), (core.w, core.h));
                tiled.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
            }
            let diff = whole.max_abs_diff(&tiled);
            assert!(
                diff <= 1e-4,
                "lens={with_lens}: tiled chain differs from whole by {diff}"
            );
        }
    }

    #[test]
    fn disabled_stages_are_skipped() {
        let (w, h) = (64, 48);
        let data = raw_of(w, h, 4);
        let raw = RawFrame {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data: &data,
            qstep: None,
        };
        let mut chain = test_chain(w, h, false);
        chain.vignette = None;
        chain.highlight_recovery = false;
        chain.denoise = None;
        chain.sharpen = None;
        chain.noise_reduction = NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        };
        let out = chain.run(&raw).expect("run");
        // With nothing but normalise, gains and demosaic the result equals those steps by hand.
        let (mut m, _) =
            normalize::normalize(&data, w, h, Cfa::RGGB, &chain.levels, None).expect("normalize");
        gains::apply_gains(&mut m, chain.gains);
        let by_hand = demosaic::demosaic(&m, Method::Lmmse);
        assert!(out.max_abs_diff(&by_hand) < 1e-6);
    }

    #[test]
    fn rejects_inputs_of_the_wrong_size_instead_of_panicking() {
        let (w, h) = (32, 32);
        let data = raw_of(w, h, 6);
        let short_q = vec![1.0f32; 10];
        let frame = |qstep| RawFrame {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data: &data,
            qstep,
        };
        let chain = test_chain(w, h, false);
        assert!(chain.run(&frame(Some(&short_q))).is_err(), "short qstep");
        let mut with_map = test_chain(w, h, false);
        with_map.sigma_map = Some(vec![0.01; 7]);
        assert!(with_map.run(&frame(None)).is_err(), "short sigma map");
        assert!(with_map.prepare(&frame(None)).is_err());
        let wrong_lens = test_chain(64, 48, true);
        assert!(
            wrong_lens.run(&frame(None)).is_err(),
            "lens for another frame size"
        );
        let mut bad_levels = test_chain(w, h, false);
        bad_levels.levels = Levels::uniform(100.0, 50.0);
        assert!(bad_levels.run(&frame(None)).is_err());
    }

    #[test]
    fn rejects_a_tile_outside_the_output() {
        let (w, h) = (32, 32);
        let data = raw_of(w, h, 5);
        let raw = RawFrame {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data: &data,
            qstep: None,
        };
        let chain = test_chain(w, h, false);
        let prepared = chain.prepare(&raw).expect("prepare");
        assert!(
            chain
                .run_tile(&raw, &prepared, Rect::new(20, 0, 20, 10))
                .is_err()
        );
    }
}
