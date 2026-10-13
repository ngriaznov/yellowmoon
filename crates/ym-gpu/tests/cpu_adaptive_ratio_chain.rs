//! RA-04: the adaptive ratio demosaic inside the CPU chain (`Chain::run_tile`, stage R6).
//!
//! R6 gets the balanced mosaic of R4/R5. For `Method::AdaptiveRatio` the chain passes its gains
//! `g0` and its noise model to `demosaic_with`, which divides every photosite by `g0` of its
//! colour (the exact inverse of R4) before the three passes (`docs/algorithms/INTEGRATION.md`
//! section 4). These tests check what that means for a whole render:
//!
//! * **Same stages, tile versus whole.** With every stage but R1, R4 and R6 off, the chain's
//!   output is `demosaic_with` on the normalised, balanced whole mosaic, bit for bit, and so is
//!   every tile of a grid that does not divide the frame (the halo is 13, the spec's maximum), for
//!   all four CFA phases, unequal gains and both a low and a high noise model (shadow matching off
//!   and on).
//! * **Every stage on.** Tiles equal the whole frame within 1e-4 with vignetting, highlight
//!   recovery, denoise, R7, R8 and the lens resample, for all four phases.
//! * **`Y_classical`.** R7's classical image is the R6 output of the same method.
//! * **Determinism.** One thread and four threads give the same bytes.
//! * **Checks.** A chain that asks for the method with unusable gains or noise is refused; the
//!   other methods keep their behaviour with the same numbers.

use ym_gpu::cpu::chain::{Chain, RawFrame};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::detail::{self, NoiseReduction};
use ym_gpu::cpu::gains;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::{Rect, tile_grid};
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{Cfa, CpuError, NoiseDn, NoiseSigma, RgbImage};

/// A low-noise model (SNR at 18 % grey above 25 dB: shadow matching off, halo 11).
const LOW_NOISE: NoiseDn = NoiseDn::new(1.311, 6.56);
/// A high-noise model (SNR at 18 % grey 23.9 dB: shadow matching on, halo 13).
const HIGH_NOISE: NoiseDn = NoiseDn::new(32.77, 419.6);
/// Gains of a daylight-like white balance with a stronger blue than red.
const G0: [f32; 3] = [2.0, 1.0, 1.6];

fn levels() -> Levels {
    Levels {
        black: [512.0, 514.0, 513.0, 511.0],
        white: [16000.0; 4],
    }
}

/// A natural-looking scene sampled through `cfa`, exposed so that a part of it clips (`exposure`
/// above 1) or stays dark (below 1), with noise.
fn raw_of(w: usize, h: usize, cfa: Cfa, seed: u64, exposure: f32) -> Vec<u16> {
    let img = synth::natural(w, h, seed);
    let m = synth::mosaic_from_rgb(&img, cfa);
    let mut rng = synth::Rng::new(seed + 7);
    m.data
        .iter()
        .map(|&v| {
            let x = 512.0 + (v * exposure + 0.004 * rng.gaussian()) * 15488.0;
            x.clamp(0.0, 16000.0) as u16
        })
        .collect()
}

fn frame(data: &[u16], w: usize, h: usize, cfa: Cfa) -> RawFrame<'_> {
    RawFrame {
        width: w,
        height: h,
        cfa,
        data,
        qstep: None,
    }
}

/// A chain with R1, R4 and R6 only: no vignetting, no highlight recovery, no denoise, no R7 control
/// and no R8, no lens stage.
fn minimal_chain(method: Method, gains: [f32; 3], noise: Option<NoiseDn>) -> Chain {
    Chain {
        levels: levels(),
        vignette: None,
        gains,
        highlight_recovery: false,
        method,
        denoise: None,
        noise_reduction: NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        },
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise,
        sharpen: None,
        lens: None,
    }
}

/// The chain's input by hand: R1 and R4 over the whole frame.
fn balanced_mosaic(chain: &Chain, raw: &RawFrame<'_>) -> ym_gpu::cpu::Mosaic {
    let (mut m, _) = normalize::normalize(
        raw.data,
        raw.width,
        raw.height,
        raw.cfa,
        &chain.levels,
        None,
    )
    .expect("normalize");
    gains::apply_gains(&mut m, chain.gains);
    m
}

fn inputs(chain: &Chain) -> DemosaicInputs<'static> {
    DemosaicInputs {
        gains: chain.gains,
        noise: chain.noise.unwrap_or(NoiseDn::ZERO),
        ca: None,
    }
}

fn run_tiled(chain: &Chain, raw: &RawFrame<'_>, tile: (usize, usize)) -> RgbImage {
    let prepared = chain.prepare(raw).expect("prepare");
    let (ow, oh) = chain.output_size(raw.width, raw.height);
    let mut out = RgbImage::new(ow, oh);
    for core in tile_grid(ow, oh, tile.0, tile.1, 1) {
        let t = chain.run_tile(raw, &prepared, core).expect("tile");
        assert_eq!((t.width, t.height), (core.w, core.h));
        out.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
    }
    out
}

#[test]
fn a_render_with_only_r6_is_the_whole_frame_demosaic_and_so_is_every_tile() {
    // R6 only, so the chain output is directly comparable to one `demosaic_with` call: the same
    // R1 and R4 by hand, then the whole-frame demosaic. 150 x 110 with tiles of 52 x 46 and
    // 31 x 29 (odd origins, edge tiles of other sizes, seams everywhere).
    let (w, h) = (150, 110);
    for cfa in Cfa::ALL {
        let data = raw_of(w, h, cfa, 3, 1.0);
        let raw = frame(&data, w, h, cfa);
        for (name, noise) in [("low", LOW_NOISE), ("high", HIGH_NOISE)] {
            let chain = minimal_chain(Method::AdaptiveRatio, G0, Some(noise));
            let whole_demosaic = demosaic::demosaic_whole_with(
                &balanced_mosaic(&chain, &raw),
                Method::AdaptiveRatio,
                &inputs(&chain),
            );
            let banded_demosaic = demosaic::demosaic_with(
                &balanced_mosaic(&chain, &raw),
                Method::AdaptiveRatio,
                &inputs(&chain),
            );
            assert_eq!(banded_demosaic, whole_demosaic, "banded equals whole");

            let run = chain.run(&raw).expect("run");
            assert_eq!(
                run,
                whole_demosaic,
                "{} {name}: the chain is the demosaic",
                cfa.name()
            );
            for tile in [(52, 46), (31, 29)] {
                let tiled = run_tiled(&chain, &raw, tile);
                assert_eq!(
                    tiled,
                    whole_demosaic,
                    "{} {name} tiles {tile:?}: tiled differs from the whole-frame demosaic",
                    cfa.name()
                );
            }
        }
    }
}

#[test]
fn the_gains_and_the_noise_model_reach_the_demosaic() {
    // The method reads both: another gain set or another noise model gives another picture, and
    // the output stays balanced (white comes out as g0).
    let (w, h) = (96, 80);
    let data = raw_of(w, h, Cfa::GBRG, 5, 1.0);
    let raw = frame(&data, w, h, Cfa::GBRG);
    let reference = minimal_chain(Method::AdaptiveRatio, G0, Some(HIGH_NOISE))
        .run(&raw)
        .expect("reference");
    let no_noise = minimal_chain(Method::AdaptiveRatio, G0, None)
        .run(&raw)
        .expect("no noise");
    let other_noise = minimal_chain(Method::AdaptiveRatio, G0, Some(LOW_NOISE))
        .run(&raw)
        .expect("low noise");
    let other_gains = minimal_chain(Method::AdaptiveRatio, [1.5, 1.0, 1.2], Some(HIGH_NOISE))
        .run(&raw)
        .expect("other gains");
    assert_ne!(
        reference, no_noise,
        "a missing noise model reads as zero noise"
    );
    assert_ne!(reference, other_noise);
    assert_ne!(reference, other_gains);
    // `None` is the zero model, nothing else.
    let zero = minimal_chain(Method::AdaptiveRatio, G0, Some(NoiseDn::ZERO))
        .run(&raw)
        .expect("zero noise");
    assert_eq!(no_noise, zero);

    // A flat grey scene: the balanced mosaic is `g0 * grey` per colour, and the output of the
    // chain is balanced: every pixel `g0 * grey`.
    let grey: Vec<u16> = vec![512 + 4000; w * h];
    let mut flat_chain = minimal_chain(Method::AdaptiveRatio, G0, Some(HIGH_NOISE));
    flat_chain.levels = Levels::uniform(512.0, 16000.0);
    let flat = flat_chain
        .run(&frame(&grey, w, h, Cfa::GBRG))
        .expect("flat");
    let level = 4000.0 / 15488.0;
    for (c, gain) in G0.iter().enumerate() {
        let want = level * gain;
        let worst = flat
            .plane(c)
            .iter()
            .map(|v| (v - want).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "plane {c}: {worst} from {want}");
    }
}

/// The chain of the whole classical stack: vignetting, highlight recovery, denoise, R7, R8 and a
/// lens crop.
fn full_chain(w: usize, h: usize, method: Method, noise: Option<NoiseDn>, lens: bool) -> Chain {
    let tags = LensTags {
        vignetting: vec![0, -400, -1500, -3500, -6000],
        distortion: vec![0, 30, 100, 220, 400, 650],
        ca: vec![0, 100, 400, 900, 0, -100, -400, -900],
    };
    Chain {
        levels: levels(),
        vignette: Some(VignetteProfile::from_tag(&tags.vignetting).expect("vignette")),
        gains: G0,
        highlight_recovery: true,
        method,
        denoise: Some(BilateralParams::default()),
        noise_reduction: NoiseReduction {
            luminance: 40.0,
            color: 60.0,
            detail: 70.0,
        },
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise,
        sharpen: Some(
            SharpenParams {
                amount: 90.0,
                radius_px: 0.8,
            }
            .into(),
        ),
        lens: lens.then(|| {
            LensGeometry::new(w, h, &tags, LensSwitches::default())
                .expect("lens")
                .with_auto_crop()
        }),
    }
}

#[test]
fn tiles_reproduce_the_whole_frame_with_every_stage_enabled() {
    // The scene is exposed 1.3 stops up so that highlights clip (R5 runs on a balanced mosaic that
    // R6 then un-balances again) and has noise (shadow matching with the high model).
    let (w, h) = (150, 110);
    for cfa in Cfa::ALL {
        let data = raw_of(w, h, cfa, 9, 1.3);
        let raw = frame(&data, w, h, cfa);
        for (name, noise) in [("low", LOW_NOISE), ("high", HIGH_NOISE)] {
            for lens in [false, true] {
                let chain = full_chain(w, h, Method::AdaptiveRatio, Some(noise), lens);
                let prepared = chain.prepare(&raw).expect("prepare");
                assert!(
                    prepared.offsets != Default::default(),
                    "the scene must clip so that R5 does something"
                );
                let (ow, oh) = chain.output_size(w, h);
                let whole = chain
                    .run_tile(&raw, &prepared, Rect::whole(ow, oh))
                    .expect("whole");
                let tiled = run_tiled(&chain, &raw, (52, 46));
                let diff = whole.max_abs_diff(&tiled);
                assert!(
                    diff <= 1e-4,
                    "{} {name} lens={lens}: tiled chain differs from whole by {diff}",
                    cfa.name()
                );
                assert!(whole.data.iter().all(|v| v.is_finite()));
            }
        }
    }
}

/// R7 by hand around a whole-frame R6: denoise, colour guided filter and detail re-injection of
/// `chain` over the whole frame, with `classical` as the image the detail term puts back.
fn r7_by_hand(chain: &Chain, r6: &RgbImage, classical: Option<&RgbImage>) -> RgbImage {
    let scale = detail::network_sigma_scale(chain.noise_reduction.luminance, chain.iso);
    let denoised = denoise::bilateral_opponent(
        r6,
        &NoiseSigma::Scalar(chain.sigma * scale),
        &chain.denoise.unwrap_or_default(),
    );
    let ns = NoiseSigma::Scalar(chain.sigma);
    let colored = detail::chroma_guided_filter(&denoised, &ns, chain.noise_reduction.color);
    match classical {
        Some(c) => {
            detail::inject_detail(&colored, c, chain.noise_reduction.beta()).expect("detail")
        }
        None => colored,
    }
}

#[test]
fn y_classical_of_r7_is_the_r6_output_of_the_same_method() {
    let (w, h) = (110, 90);
    let data = raw_of(w, h, Cfa::GRBG, 4, 1.0);
    let raw = frame(&data, w, h, Cfa::GRBG);
    let mut chain = minimal_chain(Method::AdaptiveRatio, G0, Some(HIGH_NOISE));
    chain.denoise = Some(BilateralParams::default());
    chain.noise_reduction = NoiseReduction {
        luminance: 40.0,
        color: 60.0,
        detail: 70.0,
    };
    assert!(chain.noise_reduction.needs_classical());

    let m = balanced_mosaic(&chain, &raw);
    let r6 = demosaic::demosaic_whole_with(&m, Method::AdaptiveRatio, &inputs(&chain));
    let expected = r7_by_hand(&chain, &r6, Some(&r6));
    let out = chain.run(&raw).expect("run");
    assert_eq!(out, expected, "R7's classical image is the R6 output");

    // The detail term really carries the classical image: a Menon `Y_classical` under an adaptive
    // ratio R6 would be another picture.
    let menon = demosaic::demosaic_whole_with(&m, Method::Menon, &inputs(&chain));
    let wrong = r7_by_hand(&chain, &r6, Some(&menon));
    assert_ne!(out, wrong);

    // And tile by tile (the classical image is cropped with the denoised one).
    let tiled = run_tiled(&chain, &raw, (40, 33));
    let diff = out.max_abs_diff(&tiled);
    assert!(diff <= 1e-4, "tiled R7 with the adaptive ratio R6: {diff}");
}

#[test]
fn one_thread_and_four_give_the_same_bytes() {
    let (w, h) = (150, 110);
    let data = raw_of(w, h, Cfa::BGGR, 6, 1.3);
    let raw = frame(&data, w, h, Cfa::BGGR);
    let chain = full_chain(w, h, Method::AdaptiveRatio, Some(HIGH_NOISE), true);
    let render = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                let whole = chain.run(&raw).expect("whole");
                let tiled = run_tiled(&chain, &raw, (52, 46));
                (whole, tiled)
            })
    };
    let (one_whole, one_tiled) = render(1);
    let (four_whole, four_tiled) = render(4);
    assert_eq!(
        one_whole, four_whole,
        "whole frame: --jobs 1 against --jobs 4"
    );
    assert_eq!(one_tiled, four_tiled, "tiles: --jobs 1 against --jobs 4");
}

#[test]
fn a_chain_with_unusable_gains_or_noise_for_this_method_is_refused() {
    let (w, h) = (32, 32);
    let data = raw_of(w, h, Cfa::RGGB, 1, 1.0);
    let raw = frame(&data, w, h, Cfa::RGGB);
    for gains in [[0.0, 1.0, 1.6], [2.0, 1.0, f32::NAN], [-1.0, 1.0, 1.0]] {
        let chain = minimal_chain(Method::AdaptiveRatio, gains, Some(LOW_NOISE));
        let err = chain.run(&raw).expect_err("bad gains");
        assert!(
            matches!(&err, CpuError::Mismatch(m) if m.contains("adaptive ratio")),
            "{err}"
        );
        assert!(chain.prepare(&raw).is_err());
        // The same numbers are not this chain's problem for a method that ignores them.
        if gains.iter().all(|g| g.is_finite()) {
            minimal_chain(Method::Menon, gains, Some(LOW_NOISE))
                .run(&raw)
                .expect("Menon does not read the gains");
        }
    }
    for noise in [
        NoiseDn::new(-1.0, 1.0),
        NoiseDn::new(1.0, f64::NAN),
        NoiseDn::new(f64::INFINITY, 1.0),
    ] {
        let chain = minimal_chain(Method::AdaptiveRatio, G0, Some(noise));
        assert!(chain.run(&raw).is_err(), "{noise:?}");
        // Every other method keeps its behaviour with the same noise field.
        for method in [Method::Bilinear, Method::Menon, Method::Lmmse] {
            let with = minimal_chain(method, G0, Some(noise))
                .run(&raw)
                .expect("run");
            let without = minimal_chain(method, G0, None).run(&raw).expect("run");
            assert_eq!(with, without, "{method:?} ignores the noise");
        }
    }
    // A usable chain runs.
    minimal_chain(Method::AdaptiveRatio, G0, Some(LOW_NOISE))
        .run(&raw)
        .expect("a valid chain");
    minimal_chain(Method::AdaptiveRatio, G0, None)
        .run(&raw)
        .expect("a missing noise model is zero noise");
}

#[test]
fn the_other_methods_are_not_changed_by_the_gains_of_the_chain() {
    // The default method (Menon), Bilinear and LMMSE take the balanced mosaic and ignore the
    // inputs, so the chain gives what `demosaic(m, method)` gives, as before RA-04.
    let (w, h) = (90, 70);
    let data = raw_of(w, h, Cfa::RGGB, 8, 1.0);
    let raw = frame(&data, w, h, Cfa::RGGB);
    for method in [Method::Bilinear, Method::Menon, Method::Lmmse] {
        let chain = minimal_chain(method, G0, Some(HIGH_NOISE));
        let by_hand = demosaic::demosaic(&balanced_mosaic(&chain, &raw), method);
        assert_eq!(chain.run(&raw).expect("run"), by_hand, "{method:?}");
    }
}
