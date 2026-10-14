//! G-03: the device chain (`ym_gpu::chain`) against the CPU chain (`cpu::chain::Chain`), range R6.
//!
//! `GpuChain` steps the very `Chain` and `Prepared` the CPU uses; the CPU runs R1 to R5b, the
//! adaptive ratio demosaic runs on the device (including the R4 undo, the CA path and the shadow
//! colour matching), the CPU continues with the denoise, R7, R8 and R9. What is asserted, and what
//! was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value, the bound is twice that or
//! the documented bound of the kernel, `tests/gpu_adaptive_ratio.rs`):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | R6..R6 against the CPU chain: 4 CFA phases x odd sizes x lens on/off x low/high noise (shadow matching off/on), whole frame and frame-edge tiles | over 1e-4: [`FRACTION`], over 2e-5: [`FINE_FRACTION`], largest [`BOUND`] | see the printout of `r6_matches_the_cpu_chain` |
//! | tiles of the chain (two tile sizes) | bit-identical to the whole GPU chain | 0 differing samples |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing samples |
//! | the split rule: halves of a core, down to a few hundred pixels | bit-identical to the unsplit run | 0 differing samples |
//! | a lens whose corner core is above [`MAX_PHYS_AREA`] | split into more than one rectangle, every binding under 128 MiB, bit-identical to the same run with the limit raised | see the printout |
//! | a cancelled token | `Cancelled`, the pool's in-use bytes back at the baseline | |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); the Metal runners run the same file (`mac-gpu.yml`).

mod common;

use std::sync::Arc;

use common::exact::{assert_bit_equal, assert_poison_neutral, assert_tiles_equal_whole};
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use ym_gpu::chain::{
    DemosaicWork, DeviceRunner, DeviceWork, GpuChain, MAX_PHYS_AREA, RawStage, StageRange,
    split_core, split_core_above,
};
use ym_gpu::cpu::chain::{Chain, Prepared, RawFrame};
use ym_gpu::cpu::demosaic::adaptive_ratio::Params;
use ym_gpu::cpu::demosaic::{CaCurves, DemosaicInputs, Method};
use ym_gpu::cpu::denoise::BilateralParams;
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::Levels;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use ym_gpu::device::{GpuContext, GpuError, StorageMode};
use ym_gpu::kernels::adaptive_ratio::{self, Balance, PreparedDemosaic};
use ym_gpu::kernels::bilateral::SigmaHalf;
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

/// The device with `f32` storage (the automatic one, since the device foundation).
fn f32_device_or_skip() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

// ---------------------------------------------------------------------------------------------
// Scenes and chains
// ---------------------------------------------------------------------------------------------

/// A raw frame: noisy natural content exposed so that highlights clip, 16-bit words around a black
/// level of about 100. The fine quantisation matters for the comparison with the CPU: the decisions
/// of the demosaic compare sums of absolute differences, and sums of integers tie exactly (and then
/// flip between `f32` and `f64`) the more often the coarser the steps are (`tests/gpu_adaptive_ratio.rs`).
struct Scene {
    width: usize,
    height: usize,
    cfa: Cfa,
    data: Vec<u16>,
}

impl Scene {
    fn new(width: usize, height: usize, cfa: Cfa, seed: u64) -> Scene {
        let img = synth::natural(width, height, seed);
        let m = synth::mosaic_from_rgb(&img, cfa);
        let mut rng = synth::Rng::new(seed + 7);
        let data = m
            .data
            .iter()
            .map(|&v| {
                let x = 100.0 + (v * 1.3 + 0.004 * rng.gaussian()) * 65435.0;
                x.clamp(0.0, 65535.0) as u16
            })
            .collect();
        Scene {
            width,
            height,
            cfa,
            data,
        }
    }

    fn raw(&self) -> RawFrame<'_> {
        RawFrame {
            width: self.width,
            height: self.height,
            cfa: self.cfa,
            data: &self.data,
            qstep: None,
        }
    }
}

/// How the lens of a test chain is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lens {
    /// No lens at all. The balanced mosaic is then the raw words on a fixed grid (`(raw - black) *
    /// gain`), and a comparison of the demosaic's sums of absolute differences ties exactly in
    /// places, so the device and the CPU take different but equally plausible decisions at a few
    /// pixels (`tie-prone`, see the module notes and `tests/gpu_adaptive_ratio.rs`).
    Bare,
    /// R3 vignetting only: the divisions by the gain take the values off the grid, so ties are rare
    /// (`tie-free`).
    Vignette,
    /// Vignetting, distortion and a CA table; the CA is corrected on the mosaic (R5b), as the render
    /// does. Tie-free.
    Full,
}

impl Lens {
    fn tie_prone(self) -> bool {
        self == Lens::Bare
    }
}

/// The tags of a mild lens (the unit tests of `cpu::chain` use the same).
fn mild_tags() -> LensTags {
    LensTags {
        vignetting: vec![0, -400, -1500, -3500, -6000],
        distortion: vec![0, 30, 100, 220, 400, 650],
        ca: vec![0, 100, 400, 900, 0, -100, -400, -900],
    }
}

fn low_noise() -> NoiseDn {
    NoiseDn { g: 1.3, r: 2.0 }
}

fn high_noise() -> NoiseDn {
    NoiseDn { g: 40.0, r: 90.0 }
}

/// A chain with every stage on: vignetting, adaptive ratio (the noise model decides whether shadow
/// colour matching runs), bilateral denoise, R7, Richardson-Lucy and the lens.
fn chain_of(scene: &Scene, lens: Lens, noise: NoiseDn) -> Chain {
    let (w, h) = (scene.width, scene.height);
    let tags = mild_tags();
    let geometry = (lens == Lens::Full).then(|| {
        LensGeometry::new(w, h, &tags, LensSwitches::default())
            .expect("lens")
            .with_ca_on_mosaic()
            .with_auto_crop()
    });
    Chain {
        levels: Levels {
            black: [100.0, 102.0, 101.0, 99.0],
            white: [65535.0; 4],
        },
        vignette: (lens != Lens::Bare)
            .then(|| VignetteProfile::from_tag(&tags.vignetting).expect("vignette")),
        gains: [2.0, 1.0, 1.6],
        highlight_recovery: true,
        method: Method::AdaptiveRatio,
        denoise: Some(BilateralParams::default()),
        noise_reduction: NoiseReduction {
            luminance: 40.0,
            color: 60.0,
            detail: 70.0,
        },
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise: Some(noise),
        sharpen: Some(
            SharpenParams {
                amount: 90.0,
                radius_px: 0.8,
            }
            .into(),
        ),
        lens: geometry,
    }
}

/// A chain that does little besides the demosaic and the lens: the fast one for the large frames.
fn light_chain_of(scene: &Scene, tags: &LensTags, noise: NoiseDn) -> Chain {
    let mut chain = chain_of(scene, Lens::Bare, noise);
    chain.vignette = None;
    chain.highlight_recovery = false;
    chain.denoise = None;
    chain.noise_reduction = NoiseReduction {
        luminance: 0.0,
        color: 0.0,
        detail: 0.0,
    };
    chain.sharpen = None;
    chain.lens = Some(
        LensGeometry::new(
            scene.width,
            scene.height,
            tags,
            LensSwitches {
                lens_corrections: true,
                chromatic_aberration: false,
            },
        )
        .expect("lens")
        .with_auto_crop(),
    );
    chain
}

// ---------------------------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------------------------

/// The half-resolution noise sigma of a photo and the gain at the centre of each cell, owned (what
/// `ym-core` builds from `SigmaPlane` and `gain_plane`).
struct HalfSigma {
    width: usize,
    height: usize,
    data: Vec<f32>,
    gains: Option<Vec<f32>>,
}

impl HalfSigma {
    /// A smooth plane of sigmas around `level` with some texture, and (with `vignette`) a gain that
    /// grows from 1 at the centre to about 1.8 at the corners.
    fn new(frame_w: usize, frame_h: usize, level: f32, vignette: bool, seed: u64) -> HalfSigma {
        let (width, height) = (frame_w / 2, frame_h / 2);
        let mut rng = synth::Rng::new(seed);
        let data = (0..width * height)
            .map(|i| level * (0.6 + 0.8 * ((i % width) as f32 / width as f32)) + 0.0005 * rng.next_f32())
            .collect();
        let gains = vignette.then(|| {
            (0..width * height)
                .map(|i| {
                    let (x, y) = ((i % width) as f32 / width as f32 - 0.5, (i / width) as f32 / height as f32 - 0.5);
                    1.0 + 1.6 * (x * x + y * y)
                })
                .collect()
        });
        HalfSigma {
            width,
            height,
            data,
            gains,
        }
    }

    fn view(&self) -> SigmaHalf<'_> {
        SigmaHalf {
            width: self.width,
            height: self.height,
            data: &self.data,
            gains: self.gains.as_deref(),
        }
    }

    /// The full-resolution map of the CPU chain by the rule of `ym-core`'s `full_sigma_map`.
    fn full_map(&self, w: usize, h: usize) -> Vec<f32> {
        let half = self.view();
        let mut map = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                map.push(half.at(x, y));
            }
        }
        map
    }
}

/// Everything one run needs, owned so that it can move to the device thread.
#[derive(Clone)]
struct Setup {
    scene: Arc<Scene>,
    /// The CPU chain.
    chain: Arc<Chain>,
    /// The chain the device walks: the CPU chain, or (with a half-resolution sigma) the same chain
    /// without its full-resolution sigma map.
    gpu_chain: Arc<Chain>,
    prepared: Prepared,
    range: StageRange,
    half: Option<Arc<HalfSigma>>,
}

impl Setup {
    fn new(scene: Scene, chain: Chain) -> Setup {
        let prepared = chain.prepare(&scene.raw()).expect("prepare");
        let chain = Arc::new(chain);
        Setup {
            scene: Arc::new(scene),
            gpu_chain: Arc::clone(&chain),
            chain,
            prepared,
            range: StageRange::DEMOSAIC,
            half: None,
        }
    }

    /// The same, run on the range R6 to R8.
    fn raw_group(mut self) -> Setup {
        self.range = StageRange::DEMOSAIC_TO_SHARPEN;
        self
    }

    /// The chain gets its noise sigma from `half`: the CPU chain through the expanded map, the device
    /// chain through the half-resolution plane (and has no map at all).
    fn with_half(scene: Scene, mut chain: Chain, half: HalfSigma) -> Setup {
        chain.sigma_map = Some(half.full_map(scene.width, scene.height));
        let mut setup = Setup::new(scene, chain).raw_group();
        let mut gpu_chain = (*setup.chain).clone();
        gpu_chain.sigma_map = None;
        setup.gpu_chain = Arc::new(gpu_chain);
        setup.half = Some(Arc::new(half));
        setup
    }

    fn output_size(&self) -> (usize, usize) {
        self.chain.output_size(self.scene.width, self.scene.height)
    }

    fn cpu(&self, out: Rect) -> RgbImage {
        self.chain
            .run_tile(&self.scene.raw(), &self.prepared, out)
            .expect("the CPU chain")
    }

    /// The device chain for `out`, run on the device thread (the CPU stages run there too, which
    /// is fine for a test).
    fn gpu(&self, dev: &TestGpu, out: Rect) -> RgbImage {
        self.try_gpu(dev, out).expect("the device chain")
    }

    fn try_gpu(&self, dev: &TestGpu, out: Rect) -> Result<RgbImage, GpuError> {
        let s = self.clone();
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                s.device_chain()?.run_tile(ctx, &s.scene.raw(), out, cancel)
            })
    }

    fn device_chain(&self) -> Result<GpuChain<'_>, GpuError> {
        let g = GpuChain::new(&self.gpu_chain, &self.prepared, self.range)?;
        match &self.half {
            Some(h) => g.with_sigma(h.view()),
            None => Ok(g),
        }
    }
}

/// How two images differ.
#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    n: usize,
    max_abs: f64,
    over: usize,
    fine: usize,
}

fn compare(a: &RgbImage, b: &RgbImage) -> Stats {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut s = Stats {
        n: a.data.len(),
        ..Stats::default()
    };
    for (&x, &y) in a.data.iter().zip(&b.data) {
        assert!(x.is_finite() && y.is_finite(), "a sample is not finite");
        let d = f64::from((x - y).abs());
        s.max_abs = s.max_abs.max(d);
        s.over += usize::from(d > 1e-4);
        s.fine += usize::from(d > 2e-5);
    }
    s
}

/// What a comparison may show. Tie-free content (vignetted, so off the raw grid) agrees to the
/// kernel's own bounds with a handful of exceptions; tie-prone content (the bare grid) differs at the
/// pixels where the CPU and the device take different, equally plausible decisions. Each limit is
/// twice the first value measured on lavapipe (Mesa 25.2, LLVM 20.1), never loosened.
#[derive(Clone, Copy)]
struct Limits {
    /// The fraction of samples over 1e-4 (the spec's measure of a flip).
    over: f64,
    /// The fraction over 2e-5, where a systematic error would show first.
    fine: f64,
    /// The largest difference anywhere.
    largest: f64,
}

/// The hand-off of tie-free content, all cases together. Measured: 1.72e-5 over 1e-4 (17 samples of
/// 987 888, all in one case), 3.0e-5 over 2e-5, largest 7.8e-3.
const HANDOFF_TIE_FREE: Limits = Limits {
    over: 4e-5,
    fine: 7e-5,
    largest: 1.6e-2,
};

/// The hand-off of tie-prone content, all cases together. Measured: 1.22e-3, 1.43e-3, largest 0.34.
const HANDOFF_TIE_PRONE: Limits = Limits {
    over: 2.5e-3,
    fine: 3e-3,
    largest: 0.7,
};

/// The whole R6..R6 chain on tie-free content (the later stages carry a flipped pixel to its
/// neighbours and amplify it, Richardson-Lucy most). Measured: 1.68e-4, 8.9e-4, largest 3.0e-3.
const CHAIN_TIE_FREE: Limits = Limits {
    over: 3.4e-4,
    fine: 1.8e-3,
    largest: 6.1e-3,
};

/// The same on tie-prone content. Measured: 1.44e-2, 3.4e-2, largest 0.11.
const CHAIN_TIE_PRONE: Limits = Limits {
    over: 2.9e-2,
    fine: 6.9e-2,
    largest: 0.23,
};

/// A 1500 x 1400 corner tile of a strong barrel lens, demosaic and lens only, tie-prone content,
/// against the CPU chain. Measured: 293 of 6 300 000 over 1e-4 (4.7e-5), 740 over 2e-5, largest 2.6e-3.
const STRONG_LENS: Limits = Limits {
    over: 1e-4,
    fine: 2.4e-4,
    largest: 5.3e-3,
};

/// The 1024 x 1024 corner tile of a 50 MP frame (8658 x 5772) behind a strong barrel lens, demosaic
/// and lens only, tie-prone content, against the CPU chain (`#[ignore]`d, run under `--release`).
/// First measured: 29 of 3 145 728 over 1e-4 (9.2e-6), 45 over 2e-5 (1.4e-5), largest 8.0e-3; the
/// limits are twice that. (It is not [`STRONG_LENS`]: that one is calibrated on a 1500 x 1400 tile.)
const FIFTY_MP_CORNER: Limits = Limits {
    over: 1.9e-5,
    fine: 2.9e-5,
    largest: 1.6e-2,
};

/// One hand-off case (a whole frame of a few thousand samples): a plumbing bug (a wrong core, a
/// wrong placement, the unbalanced mosaic) moves nearly every sample, a tie flip a few percent at
/// most (measured: 3.7 % in the worst tie-prone case).
const ONE_CASE: Limits = Limits {
    over: 0.08,
    fine: 0.08,
    largest: 1.0,
};

/// One case of the whole chain, a frame or a tile of a few thousand samples: the later stages
/// spread a flip to its neighbours, and a small tile can sit on a cluster of them (measured, worst
/// tie-prone case: 4.2 % over 1e-4 of a whole frame, 19 % over 2e-5 of one 40 x 31 tile). A plumbing
/// bug moves nearly all samples.
const ONE_CHAIN_CASE: Limits = Limits {
    over: 0.4,
    fine: 0.4,
    largest: 1.0,
};

fn assert_within(label: &str, s: &Stats, limits: Limits) {
    assert!(
        s.max_abs <= limits.largest,
        "{label}: largest difference {} ({} of {} samples over 1e-4)",
        s.max_abs,
        s.over,
        s.n
    );
    assert!(
        (s.over as f64) <= limits.over * s.n as f64,
        "{label}: {} of {} samples over 1e-4",
        s.over,
        s.n
    );
    assert!(
        (s.fine as f64) <= limits.fine * s.n as f64,
        "{label}: {} of {} samples over 2e-5",
        s.fine,
        s.n
    );
}

impl Stats {
    fn add(&mut self, other: &Stats) {
        self.n += other.n;
        self.over += other.over;
        self.fine += other.fine;
        self.max_abs = self.max_abs.max(other.max_abs);
    }

    fn print(&self, what: &str) {
        eprintln!(
            "{what}: {} samples, {} over 1e-4 ({:.2e}), {} over 2e-5, largest {:e}",
            self.n,
            self.over,
            self.over as f64 / self.n.max(1) as f64,
            self.fine,
            self.max_abs
        );
    }
}

/// Every case of the comparison sweep: four patterns, two odd sizes, three lens kinds, two noise
/// models (the quiet one runs without shadow colour matching, the noisy one with it).
fn sweep() -> Vec<(Cfa, usize, usize, Lens, &'static str, NoiseDn, u64)> {
    let mut cases = Vec::new();
    let mut seed = 300;
    for cfa in Cfa::ALL {
        for (w, h) in [(133usize, 97usize), (96, 80)] {
            for lens in [Lens::Bare, Lens::Vignette, Lens::Full] {
                for (name, noise) in [("low", low_noise()), ("high", high_noise())] {
                    seed += 1;
                    cases.push((cfa, w, h, lens, name, noise, seed));
                }
            }
        }
    }
    cases
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// The demosaic alone, at the hand-off: the balanced mosaic the stepper gives R6, demosaiced by the
/// CPU method and by the device, compared before any later stage can amplify a difference.
#[test]
fn the_handoff_matches_the_cpu_demosaic_before_the_later_stages() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (mut free, mut prone) = (Stats::default(), Stats::default());
    for (cfa, w, h, lens, nname, noise, seed) in sweep() {
        let scene = Scene::new(w, h, cfa, seed);
        let setup = Setup::new(scene, chain_of(&Scene::new(w, h, cfa, seed), lens, noise));
        let (ow, oh) = setup.output_size();
        let mut run = setup
            .chain
            .begin_tile(&setup.scene.raw(), &setup.prepared, Rect::whole(ow, oh))
            .expect("tile");
        let first = run
            .stages()
            .iter()
            .position(|st| RawStage::of(st) == RawStage::Demosaic)
            .expect("demosaic");
        run.run_until(first).expect("stages before R6");
        let (mosaic, core) = run.take_demosaic_input().expect("input");
        let ca = setup.chain.ca_curves();
        let inputs = setup.chain.demosaic_inputs(ca.as_ref());
        let cpu = ym_gpu::cpu::demosaic::demosaic_with(&mosaic, Method::AdaptiveRatio, &inputs)
            .extract(core);
        let frame =
            PreparedDemosaic::new(Balance::Balanced, &inputs, &Params::default()).expect("frame");
        let work = DeviceWork::Demosaic(DemosaicWork::new(mosaic, core, frame).expect("work"));
        let gpu = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                work.run(ctx, cancel)
            })
            .expect("device");
        let s = compare(&gpu, &cpu);
        let label = format!(
            "hand-off {} {w}x{h} lens {lens:?} noise {nname}",
            cfa.name()
        );
        s.print(&label);
        // One case never loses more than a few percent of its samples.
        assert_within(&label, &s, ONE_CASE);
        if lens.tie_prone() {
            prone.add(&s);
        } else {
            free.add(&s);
        }
    }
    free.print("hand-off, tie-free content");
    prone.print("hand-off, tie-prone content");
    assert_within("hand-off, tie-free content", &free, HANDOFF_TIE_FREE);
    assert_within("hand-off, tie-prone content", &prone, HANDOFF_TIE_PRONE);
}

#[test]
fn r6_matches_the_cpu_chain() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (mut free, mut prone) = (Stats::default(), Stats::default());
    for (cfa, w, h, lens, nname, noise, seed) in sweep() {
        let setup = Setup::new(
            Scene::new(w, h, cfa, seed),
            chain_of(&Scene::new(w, h, cfa, seed), lens, noise),
        );
        let (ow, oh) = setup.output_size();
        // The whole output and tiles on the frame edges and in the middle.
        for out in [
            Rect::whole(ow, oh),
            Rect::new(0, 0, 40.min(ow), 31.min(oh)),
            Rect::new(
                ow.saturating_sub(37),
                oh.saturating_sub(29),
                37.min(ow),
                29.min(oh),
            ),
            Rect::new(ow / 3, oh / 3, 33.min(ow - ow / 3), 21.min(oh - oh / 3)),
        ] {
            let s = compare(&setup.gpu(dev, out), &setup.cpu(out));
            let label = format!("{} {w}x{h} lens {lens:?} noise {nname} {out:?}", cfa.name());
            // A tile of a few thousand samples: a plumbing bug moves nearly all of them.
            assert_within(&label, &s, ONE_CHAIN_CASE);
            if lens.tie_prone() {
                prone.add(&s);
            } else {
                free.add(&s);
            }
        }
    }
    free.print("R6..R6 chain, tie-free content");
    prone.print("R6..R6 chain, tie-prone content");
    assert_within("R6..R6 chain, tie-free content", &free, CHAIN_TIE_FREE);
    assert_within("R6..R6 chain, tie-prone content", &prone, CHAIN_TIE_PRONE);
}

#[test]
fn the_shadow_matching_switch_follows_the_noise_model_on_both_sides() {
    // The device decides shadow colour matching from the same `Derived` the CPU does, so a noisy
    // chain exercises the 12-plane path and a quiet one the 5-plane path; both must agree with the
    // CPU.
    let Some(_dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(120, 90, Cfa::RGGB, 9);
    for (noise, matching) in [(low_noise(), false), (high_noise(), true)] {
        let chain = chain_of(&scene, Lens::Bare, noise);
        let inputs = chain.demosaic_inputs(None);
        let frame =
            PreparedDemosaic::new(Balance::Balanced, &inputs, &Params::default()).expect("frame");
        assert_eq!(frame.shadow_match(), matching, "{noise:?}");
        assert_eq!(frame.halo_px(), if matching { 13 } else { 11 });
    }
}

#[test]
fn tiles_of_the_chain_equal_the_whole_chain_bit_for_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 500;
    for (cfa, lens, noise) in [
        (Cfa::RGGB, Lens::Full, high_noise()),
        (Cfa::GRBG, Lens::Bare, low_noise()),
        (Cfa::BGGR, Lens::Full, low_noise()),
    ] {
        seed += 1;
        let (w, h) = (171usize, 133usize);
        let scene = Scene::new(w, h, cfa, seed);
        let chain = chain_of(&scene, lens, noise);
        let setup = Setup::new(scene, chain);
        let (ow, oh) = setup.output_size();
        assert_tiles_equal_whole(
            &format!("{} lens {}", cfa.name(), lens == Lens::Full),
            ow,
            oh,
            3,
            &[64, 100],
            |rect| setup.gpu(dev, rect).data,
        );
    }
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let Some(_) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 77);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::new(scene, chain);
    let (ow, oh) = setup.output_size();
    let out = Rect::new(7, 5, ow - 9, oh - 8);
    let got = assert_poison_neutral("the device chain", |gpu| {
        let s = setup.clone();
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?.run_tile(
                ctx,
                &s.scene.raw(),
                out,
                cancel,
            )
        })
        .expect("chain")
        .data
    });
    assert!(got.is_some(), "a plain device is available");
}

#[test]
fn the_chain_refuses_what_the_device_does_not_run() {
    let scene = Scene::new(64, 48, Cfa::RGGB, 1);
    let mut chain = chain_of(&scene, Lens::Bare, low_noise());
    let prepared = chain.prepare(&scene.raw()).expect("prepare");
    for range in [
        StageRange {
            first: RawStage::Demosaic,
            last: RawStage::Detail,
        },
        StageRange {
            first: RawStage::Demosaic,
            last: RawStage::Lens,
        },
        StageRange {
            first: RawStage::Normalize,
            last: RawStage::Lens,
        },
        StageRange {
            first: RawStage::Denoise,
            last: RawStage::Denoise,
        },
    ] {
        let err = GpuChain::new(&chain, &prepared, range).expect_err("a range not run yet");
        assert!(matches!(err, GpuError::InvalidInput(_)), "{err}");
    }
    assert_eq!(StageRange::DEMOSAIC.label(), "R6");
    assert_eq!(StageRange::DEMOSAIC_TO_SHARPEN.label(), "R6..R8");
    assert!(StageRange::DEMOSAIC.contains(RawStage::Demosaic));
    assert!(!StageRange::DEMOSAIC.contains(RawStage::Denoise));
    for stage in [RawStage::Denoise, RawStage::Detail, RawStage::Sharpen] {
        assert!(StageRange::DEMOSAIC_TO_SHARPEN.contains(stage));
    }
    assert!(!StageRange::DEMOSAIC_TO_SHARPEN.contains(RawStage::Lens));
    assert!(StageRange::DEMOSAIC_TO_SHARPEN.covers_sigma_stages());
    assert!(!StageRange::DEMOSAIC.covers_sigma_stages());
    // A half-resolution sigma only replaces the chain's map when no CPU stage reads the sigma.
    let half = HalfSigma::new(64, 48, 0.01, false, 1);
    let r6 = GpuChain::new(&chain, &prepared, StageRange::DEMOSAIC).expect("R6");
    assert!(matches!(
        r6.with_sigma(half.view()),
        Err(GpuError::InvalidInput(_))
    ));
    let r8 = GpuChain::new(&chain, &prepared, StageRange::DEMOSAIC_TO_SHARPEN).expect("R6..R8");
    assert!(r8.with_sigma(half.view()).is_ok());
    // Every other R6 method runs on the CPU only.
    for method in [
        Method::Menon,
        Method::Lmmse,
        Method::Bilinear,
        Method::LogGuided,
    ] {
        chain.method = method;
        let err = GpuChain::new(&chain, &prepared, StageRange::DEMOSAIC).expect_err("method");
        assert!(
            matches!(err, GpuError::InvalidInput(_)),
            "{method:?}: {err}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The split rule
// ---------------------------------------------------------------------------------------------

#[test]
fn the_split_rule_tiles_the_core_exactly_and_depends_on_the_rectangle_alone() {
    for core in [
        Rect::new(0, 0, 3000, 2000),
        Rect::new(13, 7, 1581, 1581),
        Rect::new(5, 0, 60, 90_000),
        Rect::new(100, 100, 7000, 11),
        Rect::new(0, 0, 1024, 1024),
    ] {
        let parts = split_core(core);
        // Deterministic: the same list every time.
        assert_eq!(parts, split_core(core));
        // The parts tile the core exactly once.
        let area: usize = parts.iter().map(Rect::area).sum();
        assert_eq!(area, core.area(), "{core:?}");
        for (i, a) in parts.iter().enumerate() {
            assert!(core.contains(a), "{a:?} is outside {core:?}");
            for b in &parts[i + 1..] {
                assert!(a.intersect(b).is_empty(), "{a:?} overlaps {b:?}");
            }
            // Every part fits the limit (or is a single pixel).
            assert!(
                adaptive_ratio::padded_pixels(*a) <= MAX_PHYS_AREA,
                "{a:?} has {} padded pixels",
                adaptive_ratio::padded_pixels(*a)
            );
        }
    }
    // A 1024 px export tile with its halos is far below the limit: one part.
    assert_eq!(split_core(Rect::new(0, 0, 1150, 1150)).len(), 1);
    // 3000 x 2000 needs at least two.
    assert!(split_core(Rect::new(0, 0, 3000, 2000)).len() >= 3);
    // Cuts are at even offsets of the core.
    for p in split_core(Rect::new(0, 0, 3000, 2000)) {
        assert_eq!((p.x % 2, p.y % 2), (0, 0), "{p:?}");
    }
}

#[test]
fn the_kernel_binding_fits_the_design_size_at_the_largest_rectangle() {
    // K4: the largest core the split rule leaves, in a few shapes (a square, a strip).
    for rect in [
        Rect::new(0, 0, 1556, 1556),
        Rect::new(0, 0, 40, 38_000),
        Rect::new(0, 0, 38_000, 40),
    ] {
        assert!(
            adaptive_ratio::padded_pixels(rect) <= MAX_PHYS_AREA,
            "{rect:?}"
        );
        assert_max_binding_bytes("adaptive ratio", rect, adaptive_ratio::max_binding_bytes);
    }
    // The rule keeps every part under the limit, so every part's binding fits.
    for part in split_core(Rect::new(0, 0, 9000, 6000)) {
        common::limits::assert_fits_design_binding(
            &format!("part {part:?}"),
            adaptive_ratio::max_binding_bytes(part),
        );
    }
}

/// A runner that records how many rectangles each piece of work is cut into, then runs it.
struct Recording<'a> {
    ctx: &'a GpuContext,
    parts: std::cell::RefCell<Vec<usize>>,
    limit: Option<u64>,
}

impl DeviceRunner for Recording<'_> {
    fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        let DeviceWork::Demosaic(w) = work else {
            panic!("this runner is only given the demosaic range");
        };
        let parts = match self.limit {
            Some(l) => split_core_above(w.core(), l).len(),
            None => w.parts().len(),
        };
        self.parts.borrow_mut().push(parts);
        match self.limit {
            Some(l) => w.run_with_limit(self.ctx, cancel, l),
            None => DeviceWork::Demosaic(w).run(self.ctx, cancel),
        }
    }
}

#[test]
fn splitting_a_core_changes_no_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    // Odd sizes, all phases, with CA: the halves of a core are cut at the limit given here (a few
    // thousand padded pixels), so the run is cut many times.
    let mut seed = 700;
    for cfa in Cfa::ALL {
        for (w, h, noise) in [(173usize, 131usize, high_noise()), (96, 77, low_noise())] {
            seed += 1;
            let scene = Scene::new(w, h, cfa, seed);
            let chain = chain_of(&scene, Lens::Full, noise);
            let setup = Setup::new(scene, chain);
            let (ow, oh) = setup.output_size();
            let out = Rect::new(3, 2, ow - 5, oh - 4);
            let s = setup.clone();
            let (whole, cut, counts) = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                    let g = GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?;
                    let plain = Recording {
                        ctx,
                        parts: Default::default(),
                        limit: None,
                    };
                    let whole = g.run_tile(&plain, &s.scene.raw(), out, cancel)?;
                    let cutter = Recording {
                        ctx,
                        parts: Default::default(),
                        limit: Some(4_500),
                    };
                    let cut = g.run_tile(&cutter, &s.scene.raw(), out, cancel)?;
                    Ok((whole, cut, cutter.parts.take()))
                })
                .expect("chains");
            assert!(counts.iter().all(|&n| n >= 6), "cut into {counts:?}");
            assert_bit_equal(
                &format!(
                    "{} {w}x{h}: cut at 4500 padded pixels against whole",
                    cfa.name()
                ),
                &cut.data,
                &whole.data,
            );
        }
    }
}

#[test]
fn a_corner_tile_of_a_strong_lens_is_split_and_equals_the_unsplit_run() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    // A barrel lens strong enough that the source of a tile in the corner of the output is much
    // bigger than the tile: the core of the demosaic exceeds MAX_PHYS_AREA although the output tile
    // is only 1200 x 1200. 3000 x 2100 photosites.
    let (w, h) = (3000usize, 2100usize);
    let tags = LensTags {
        vignetting: vec![],
        distortion: vec![0, 600, 1800, 3600, 6000, 9000, 12_500, 16_000],
        ca: vec![],
    };
    let scene = Scene::new(w, h, Cfa::RGGB, 55);
    let chain = light_chain_of(&scene, &tags, low_noise());
    let setup = Setup::new(scene, chain);
    let (ow, oh) = setup.output_size();
    let out = Rect::new(ow - 1500, oh - 1400, 1500, 1400);
    let s = setup.clone();
    let (split, unsplit, counts, binding) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let g = GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?;
            let real = Recording {
                ctx,
                parts: Default::default(),
                limit: None,
            };
            let split = g.run_tile(&real, &s.scene.raw(), out, cancel)?;
            let raised = Recording {
                ctx,
                parts: Default::default(),
                limit: Some(u64::MAX),
            };
            let unsplit = g.run_tile(&raised, &s.scene.raw(), out, cancel)?;
            // The largest binding any part of the real run needed.
            let mut run = s
                .chain
                .begin_tile(&s.scene.raw(), &s.prepared, out)
                .expect("tile");
            let first = run
                .stages()
                .iter()
                .position(|st| RawStage::of(st) == RawStage::Demosaic)
                .expect("demosaic");
            run.run_until(first).expect("stages before R6");
            let core = run.demosaic_input().expect("input").core;
            let binding = split_core(core)
                .into_iter()
                .map(adaptive_ratio::max_binding_bytes)
                .max()
                .unwrap_or(0);
            Ok((
                split,
                unsplit,
                (real.parts.take(), raised.parts.take()),
                (binding, core),
            ))
        })
        .expect("chains");
    let (real_parts, raised_parts) = counts;
    let (binding_bytes, core) = binding;
    eprintln!(
        "strong lens: output tile {out:?}, demosaic core {}x{} = {} padded pixels (limit {MAX_PHYS_AREA}), \
         cut into {real_parts:?} rectangles, largest binding {:.1} MiB",
        core.w,
        core.h,
        adaptive_ratio::padded_pixels(core),
        binding_bytes as f64 / f64::from(1u32 << 20)
    );
    assert!(
        adaptive_ratio::padded_pixels(core) > MAX_PHYS_AREA,
        "the scene must exceed the limit"
    );
    assert!(real_parts.iter().all(|&n| n > 1), "{real_parts:?}");
    assert_eq!(raised_parts, vec![1]);
    common::limits::assert_fits_design_binding("the largest part", binding_bytes);
    assert_bit_equal("split against unsplit", &split.data, &unsplit.data);
    // And both against the CPU chain (the kernel's own tolerance).
    let s = compare(&split, &setup.cpu(out));
    s.print("strong lens corner against the CPU");
    assert_within("strong lens corner against the CPU", &s, STRONG_LENS);
}

/// The 50 MP frame size (8658 x 5772) through a corner tile: the split rule on the size the plan
/// names. Slow on lavapipe; `cargo test -p ym-gpu --release --test gpu_chain -- --ignored`.
#[test]
#[ignore = "50 MP frame: run with --release -- --ignored"]
fn a_50_mp_frame_corner_tile_is_split_within_the_design_binding() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (w, h) = (8658usize, 5772usize);
    let tags = LensTags {
        vignetting: vec![],
        distortion: vec![0, 600, 1800, 3600, 6000, 9000, 12_500, 16_000],
        ca: vec![],
    };
    let scene = Scene::new(w, h, Cfa::RGGB, 56);
    let chain = light_chain_of(&scene, &tags, low_noise());
    let setup = Setup::new(scene, chain);
    let (ow, oh) = setup.output_size();
    let out = Rect::new(ow - 1024, oh - 1024, 1024, 1024);
    // The demosaic core of this tile is above the limit, is cut, and every part fits the binding.
    let mut run = setup
        .chain
        .begin_tile(&setup.scene.raw(), &setup.prepared, out)
        .expect("tile");
    let first = run
        .stages()
        .iter()
        .position(|st| RawStage::of(st) == RawStage::Demosaic)
        .expect("demosaic");
    run.run_until(first).expect("stages before R6");
    let core = run.demosaic_input().expect("input").core;
    let parts = split_core(core);
    let binding = parts
        .iter()
        .map(|&r| adaptive_ratio::max_binding_bytes(r))
        .max()
        .unwrap_or(0);
    eprintln!(
        "50 MP corner tile {out:?}: demosaic core {}x{} = {} padded pixels (limit {MAX_PHYS_AREA}), \
         cut into {} rectangles, largest binding {:.1} MiB",
        core.w,
        core.h,
        adaptive_ratio::padded_pixels(core),
        parts.len(),
        binding as f64 / f64::from(1u32 << 20)
    );
    assert!(adaptive_ratio::padded_pixels(core) > MAX_PHYS_AREA);
    assert!(parts.len() > 1);
    common::limits::assert_fits_design_binding("the largest part", binding);
    drop(run);
    let gpu = setup.gpu(dev, out);
    let s = compare(&gpu, &setup.cpu(out));
    s.print("50 MP corner tile against the CPU");
    assert_within("50 MP corner tile against the CPU", &s, FIFTY_MP_CORNER);
}

#[test]
fn a_cancelled_chain_stops_and_leaves_the_pool_at_its_baseline() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 3);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::new(scene, chain);
    let (ow, oh) = setup.output_size();
    let s = setup.clone();
    let (before, after_ok, after_cancel, cancelled_run, err) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let before = ctx.pool().stats().in_use_bytes;
            let g = GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?;
            let ok = g.run_tile(
                ctx,
                &s.scene.raw(),
                Rect::whole(ow, oh),
                &CancelToken::new(),
            )?;
            drop(ok);
            let after_ok = ctx.pool().stats().in_use_bytes;
            // A token that is already set stops the tile before any work.
            let token = CancelToken::new();
            token.cancel();
            let counters = ctx.counters();
            let err = g.run_tile(ctx, &s.scene.raw(), Rect::whole(ow, oh), &token);
            let ran = ctx.counters().since(counters);
            // A token set while the device work is about to run: the runner sees it.
            struct CancelsFirst<'a>(&'a GpuContext, CancelToken);
            impl DeviceRunner for CancelsFirst<'_> {
                fn run_device(
                    &self,
                    work: DeviceWork,
                    cancel: &CancelToken,
                ) -> Result<RgbImage, GpuError> {
                    self.1.cancel();
                    work.run(self.0, cancel)
                }
            }
            let token = CancelToken::new();
            let again = g.run_tile(
                &CancelsFirst(ctx, token.clone()),
                &s.scene.raw(),
                Rect::whole(ow, oh),
                &token,
            );
            let after_cancel = ctx.pool().stats().in_use_bytes;
            Ok((
                before,
                after_ok,
                after_cancel,
                ran,
                (err.map(|_| ()), again.map(|_| ())),
            ))
        })
        .expect("job");
    assert_eq!(after_ok, before, "a finished tile releases its buffers");
    assert_eq!(
        after_cancel, before,
        "cancelled tiles release their buffers"
    );
    assert!(matches!(err.0, Err(GpuError::Cancelled)), "{:?}", err.0);
    assert!(matches!(err.1, Err(GpuError::Cancelled)), "{:?}", err.1);
    assert_eq!(
        cancelled_run.dispatches, 0,
        "no device work for a set token"
    );
    assert_eq!(cancelled_run.submissions, 0);
}

#[test]
fn a_mosaic_below_two_by_two_is_the_cpu_stage_and_the_smallest_the_device_takes_is_two_by_two() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    for (w, h) in [(1usize, 1usize), (3, 1), (1, 5), (2, 2), (3, 3), (2, 7)] {
        let scene = Scene::new(w.max(2), h.max(2), Cfa::RGGB, 5);
        let mut data = scene.data.clone();
        data.truncate(w * h);
        let tiny = Scene {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data,
        };
        let mut chain = chain_of(&tiny, Lens::Bare, low_noise());
        chain.denoise = None;
        chain.highlight_recovery = false;
        chain.sharpen = None;
        chain.noise_reduction = NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        };
        let setup = Setup::new(tiny, chain);
        let out = Rect::whole(w, h);
        let cpu = setup.cpu(out);
        let gpu = setup.gpu(dev, out);
        if w < 2 || h < 2 {
            // The CPU demosaics these (bilinear): the very same stage runs.
            assert_bit_equal(&format!("{w}x{h}"), &gpu.data, &cpu.data);
        } else {
            // On the device: the kernel's rounding (`f32` against `f64`) only.
            let s = compare(&gpu, &cpu);
            assert!(s.max_abs <= 1e-5, "{w}x{h}: {}", s.max_abs);
        }
    }
}

/// Small frames through the whole chain (denoise, R7, Richardson-Lucy and the lens all run, with
/// halos far larger than the frame): the device must neither panic nor refuse them, every sample
/// must be finite, and the device must agree with the CPU chain, the kernel's mirror at the edge of
/// the rectangle included. Measured on lavapipe: 41 of the 44 cases agree to 6e-6 or better (a
/// tie-prone bare-grid 23 x 11 has 54 samples up to 4.5e-3). The other two are decisions on exact
/// ties, which the `f32` kernel and the `f64` reference settle differently: a 2 x 2 mosaic (the
/// mirror makes both neighbours of every site the same sample, so every direction test ties;
/// `tests/gpu_adaptive_ratio.rs` measures it on white noise) up to 9.1e-4, and a 3 x 3 clipped
/// plateau with two samples one ulp apart (equal costs everywhere) up to 0.65 on every sample of that
/// frame. So the bound is on the whole set: at most 2 % of the 9984 samples over 1e-4 (measured
/// 0.93 %) and a largest difference below 1.3 (measured 0.65), which a plumbing error of the
/// small-frame paths (a wrong mirror, core or placement moves most frames, not two) would exceed.
#[test]
fn small_frames_through_the_whole_chain_match_the_cpu() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 900;
    let mut all = Stats::default();
    for cfa in [Cfa::RGGB, Cfa::GBRG] {
        for (w, h) in [
            (2usize, 2usize),
            (2, 3),
            (3, 2),
            (3, 3),
            (4, 4),
            (5, 7),
            (9, 6),
            (2, 40),
            (40, 2),
            (17, 17),
            (23, 11),
        ] {
            for lens in [Lens::Bare, Lens::Vignette] {
                seed += 1;
                let scene = Scene::new(w, h, cfa, seed);
                let chain = chain_of(&scene, lens, high_noise());
                let setup = Setup::new(scene, chain);
                let (ow, oh) = setup.output_size();
                // `compare` also asserts that every sample of both is finite.
                let s = compare(
                    &setup.gpu(dev, Rect::whole(ow, oh)),
                    &setup.cpu(Rect::whole(ow, oh)),
                );
                s.print(&format!("{} {w}x{h} lens {lens:?}", cfa.name()));
                all.add(&s);
            }
        }
    }
    all.print("small frames, whole chain");
    assert_within(
        "small frames, whole chain",
        &all,
        Limits {
            over: 0.02,
            fine: 0.03,
            largest: 1.3,
        },
    );
}

#[test]
fn the_work_refuses_a_bad_mosaic_before_it_reaches_the_device() {
    let scene = Scene::new(40, 30, Cfa::RGGB, 8);
    let chain = chain_of(&scene, Lens::Bare, low_noise());
    let inputs = DemosaicInputs {
        gains: chain.gains,
        noise: chain.noise_dn(),
        ca: None::<&CaCurves>,
    };
    let frame = PreparedDemosaic::new(Balance::Balanced, &inputs, &Params::default()).expect("ok");
    let mut m = Mosaic::new(40, 30, Cfa::RGGB);
    m.data.fill(0.2);
    DemosaicWork::new(m.clone(), Rect::new(0, 0, 40, 30), frame.clone()).expect("fine");
    m.data[100] = f32::NAN;
    let err = DemosaicWork::new(m.clone(), Rect::new(0, 0, 40, 30), frame.clone())
        .expect_err("a NaN photosite");
    assert!(matches!(err, GpuError::InvalidInput(_)), "{err}");
    m.data[100] = 0.2;
    let err = DemosaicWork::new(m, Rect::new(30, 0, 20, 30), frame).expect_err("outside");
    assert!(matches!(err, GpuError::InvalidInput(_)), "{err}");
}
