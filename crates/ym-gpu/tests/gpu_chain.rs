//! The device chain (`ym_gpu::chain`) against the CPU chain (`cpu::chain::Chain`), range R6.
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

/// Prints like `eprintln!` and writes the same line to the process's own stderr as well (the harness captures `eprintln!`,
/// not this), so that the numbers of a passing test reach the CI log of the Apple runners, from which a hardware column
/// is made. No annotation: the budget of those is spent by the older notes.
macro_rules! note {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        eprintln!("{line}");
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr(),
            format_args!("[{}] {line}\n", file!().rsplit('/').next().unwrap_or("")),
        );
    }};
}
#[path = "denoise_common/mod.rs"]
mod denoise_common;

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
use ym_gpu::kernels::prep::DeviceRaw;
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
    /// The quantisation steps of a lossy file (one per photosite), or none.
    qstep: Option<Vec<f32>>,
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
            qstep: None,
        }
    }

    /// The same scene as a lossy file: a step per photosite (f16 values, as the decoder keeps them).
    fn with_steps(mut self, seed: u64) -> Scene {
        let mut rng = synth::Rng::new(seed);
        let table = [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        self.qstep = Some(
            (0..self.data.len())
                .map(|_| table[(rng.next_u64() % table.len() as u64) as usize])
                .collect(),
        );
        self
    }

    fn raw(&self) -> RawFrame<'_> {
        RawFrame {
            width: self.width,
            height: self.height,
            cfa: self.cfa,
            data: &self.data,
            qstep: self.qstep.as_deref(),
        }
    }

    /// The photo's raw words resident on the device (what `ym-core` uploads once per photo).
    fn upload(&self, ctx: &GpuContext) -> Result<Arc<DeviceRaw>, GpuError> {
        Ok(Arc::new(DeviceRaw::upload(
            ctx,
            "chain test photo",
            self.width,
            self.height,
            self.cfa,
            &self.data,
            self.qstep.as_deref(),
        )?))
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
            .map(|i| {
                level * (0.6 + 0.8 * ((i % width) as f32 / width as f32)) + 0.0005 * rng.next_f32()
            })
            .collect();
        let gains = vignette.then(|| {
            (0..width * height)
                .map(|i| {
                    let (x, y) = (
                        (i % width) as f32 / width as f32 - 0.5,
                        (i / width) as f32 / height as f32 - 0.5,
                    );
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

    /// The same, run on the range R1 to R9 from the resident raw words.
    fn front(mut self) -> Setup {
        self.range = StageRange::RAW_TO_LENS;
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
                let g = s.device_chain()?;
                if s.range == StageRange::RAW_TO_LENS {
                    let raw = s.scene.upload(ctx)?;
                    g.run_tile_front(ctx, &raw, out, cancel)
                } else {
                    g.run_tile(ctx, &s.scene.raw(), out, cancel)
                }
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
        note!(
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
        let work = DeviceWork::Demosaic(Box::new(
            DemosaicWork::new(mosaic, core, frame).expect("work"),
        ));
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
            last: RawStage::Sharpen,
        },
        StageRange {
            first: RawStage::Highlights,
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
    // The whole chain from the raw words is a range of its own (it reads the resident words,
    // so `run_tile` refuses it and `run_tile_front` runs it).
    assert_eq!(StageRange::RAW_TO_LENS.label(), "R1..R9");
    assert!(GpuChain::new(&chain, &prepared, StageRange::RAW_TO_LENS).is_ok());
    for stage in [
        RawStage::Normalize,
        RawStage::Highlights,
        RawStage::Demosaic,
        RawStage::Lens,
    ] {
        assert!(StageRange::RAW_TO_LENS.contains(stage));
    }
    assert!(StageRange::RAW_TO_LENS.covers_sigma_stages());
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

/// The 50 MP frame size (8658 x 5772) through a corner tile: the split rule on the size
/// `docs/17-gpu.md` names. Slow on lavapipe;
/// `cargo test -p ym-gpu --release --test gpu_chain -- --ignored`.
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
            qstep: None,
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

// ---------------------------------------------------------------------------------------------
// The range R6 to R8
// ---------------------------------------------------------------------------------------------

/// The whole R6..R8 chain on tie-free content against the CPU chain, all cases together (the three
/// sigma sources). First measured on lavapipe: 214 of 1 228 536 samples over 1e-4 (1.74e-4), 1115
/// over 2e-5, largest 2.87e-3 (R6..R6: 206, and 3.0e-3: the stages after the demosaic add nothing a
/// tie flip of the demosaic does not already show; `the_stages_after_the_demosaic_...` isolates them).
const RAW_GROUP_TIE_FREE: Limits = Limits {
    over: 3.5e-4,
    fine: 1.8e-3,
    largest: 5.8e-3,
};

/// The same on tie-prone content. First measured: 9205 of 638 232 over 1e-4 (1.44e-2), 21 839 over
/// 2e-5 (3.42e-2), largest 0.11.
const RAW_GROUP_TIE_PRONE: Limits = Limits {
    over: 2.9e-2,
    fine: 6.9e-2,
    largest: 0.23,
};

/// The stages after the demosaic on the very planes the device demosaic made: R6..R8 against R6..R6
/// (the CPU runs the same stages after the same device demosaic), so a tie flip of the demosaic is
/// in both and only the denoise, R7 and R8 kernels differ. Largest difference over all 96 cases:
/// first measured 5.2e-6 (mean of the case maxima 1.2e-6; the stage kernels alone are at 4e-7 to
/// 7e-7), the bound is twice that.
const POST_STAGES_LARGEST: f64 = 1.1e-5;

/// Tiles of a chain whose Richardson-Lucy runs past the halo cap (n = 10, radius 1.8: exact radius
/// 140 px against 24), against the whole frame: twice the first measured value on the device
/// (1.09e-5; the CPU's own bands part from its whole-frame run by 1.07e-5 there).
const BEYOND_CAP_CHAIN: f64 = 2.2e-5;

/// A chain variant of the R6..R8 sweep: the sigma source and the stage options.
#[derive(Clone, Copy, Debug)]
enum Variant {
    /// The chain's scalar sigma.
    Scalar,
    /// A half-resolution plane without gains (no vignetting profile).
    Half,
    /// A half-resolution plane with the gain of every cell.
    HalfWithGains,
}

fn setup_for(scene: Scene, lens: Lens, noise: NoiseDn, variant: Variant) -> Setup {
    let chain = chain_of(&scene, lens, noise);
    let (w, h) = (scene.width, scene.height);
    match variant {
        Variant::Scalar => Setup::new(scene, chain).raw_group(),
        Variant::Half => Setup::with_half(scene, chain, HalfSigma::new(w, h, 0.01, false, 5)),
        Variant::HalfWithGains => {
            Setup::with_half(scene, chain, HalfSigma::new(w, h, 0.008, true, 6))
        }
    }
}

#[test]
fn r6_to_r8_matches_the_cpu_chain() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (mut free, mut prone) = (Stats::default(), Stats::default());
    for (n, (cfa, w, h, lens, nname, noise, seed)) in sweep().into_iter().enumerate() {
        let variant = [Variant::Scalar, Variant::Half, Variant::HalfWithGains][n % 3];
        let setup = setup_for(Scene::new(w, h, cfa, seed), lens, noise, variant);
        let (ow, oh) = setup.output_size();
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
            let label = format!(
                "{} {w}x{h} lens {lens:?} noise {nname} {variant:?} {out:?}",
                cfa.name()
            );
            assert_within(&label, &s, ONE_CHAIN_CASE);
            if lens.tie_prone() {
                prone.add(&s);
            } else {
                free.add(&s);
            }
        }
    }
    free.print("R6..R8 chain, tie-free content");
    prone.print("R6..R8 chain, tie-prone content");
    assert_within("R6..R8 chain, tie-free content", &free, RAW_GROUP_TIE_FREE);
    assert_within(
        "R6..R8 chain, tie-prone content",
        &prone,
        RAW_GROUP_TIE_PRONE,
    );
}

#[test]
fn the_stages_after_the_demosaic_match_the_cpu_stages_on_the_same_planes() {
    // R6..R6 and R6..R8 share the device demosaic bit for bit; what differs is who runs the denoise,
    // R7 and R8. So this difference is the error of the denoise, R7 and R8 kernels in the chain,
    // without the tie flips of the demosaic that the comparison with the CPU chain carries.
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut worst = 0.0f64;
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (k, (cfa, w, h, lens, nname, noise, seed)) in sweep().into_iter().enumerate() {
        let variant = [Variant::Scalar, Variant::Half, Variant::HalfWithGains][k % 3];
        let raw8 = setup_for(Scene::new(w, h, cfa, seed), lens, noise, variant);
        // The same chain on R6..R6: the CPU chain needs the sigma map, which `with_half` made.
        let mut raw6 = raw8.clone();
        raw6.range = StageRange::DEMOSAIC;
        raw6.gpu_chain = Arc::clone(&raw6.chain);
        raw6.half = None;
        let (ow, oh) = raw8.output_size();
        for out in [
            Rect::whole(ow, oh),
            Rect::new(ow / 3, oh / 3, 33.min(ow - ow / 3), 21.min(oh - oh / 3)),
        ] {
            let a = raw8.gpu(dev, out);
            let b = raw6.gpu(dev, out);
            let s = compare(&a, &b);
            sum += s.max_abs;
            n += 1;
            worst = worst.max(s.max_abs);
            eprintln!(
                "post stages {} {w}x{h} lens {lens:?} noise {nname} {variant:?}: largest {:.3e}, {} over 2e-5",
                cfa.name(),
                s.max_abs,
                s.fine
            );
        }
    }
    note!(
        "[R6..R8] stages after the demosaic, R6..R8 against R6..R6: worst {worst:.3e}, mean of the maxima {:.3e}",
        sum / n as f64
    );
    assert!(worst <= POST_STAGES_LARGEST, "{worst:e}");
}

#[test]
fn tiles_of_the_raw_group_equal_the_whole_chain_bit_for_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 1500;
    for (cfa, lens, noise, variant) in [
        (Cfa::RGGB, Lens::Full, high_noise(), Variant::HalfWithGains),
        (Cfa::GRBG, Lens::Bare, low_noise(), Variant::Scalar),
        (Cfa::BGGR, Lens::Full, low_noise(), Variant::Half),
    ] {
        seed += 1;
        let (w, h) = (171usize, 133usize);
        let setup = setup_for(Scene::new(w, h, cfa, seed), lens, noise, variant);
        let (ow, oh) = setup.output_size();
        assert_tiles_equal_whole(
            &format!("{} lens {:?} {variant:?}", cfa.name(), lens),
            ow,
            oh,
            3,
            &[64, 100],
            |rect| setup.gpu(dev, rect).data,
        );
    }
}

#[test]
fn the_three_sigma_sources_give_the_same_pixels() {
    // The chain's scalar, a full-resolution map holding the same value and a half-resolution plane
    // holding it: one pixel each way, bit for bit.
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (w, h) = (130usize, 90usize);
    let value = 0.0123f32;
    let scene = || Scene::new(w, h, Cfa::RGGB, 41);
    let mut chain = chain_of(&scene(), Lens::Vignette, high_noise());
    chain.sigma = value;
    let scalar = Setup::new(scene(), chain.clone()).raw_group();
    let mut mapped_chain = chain.clone();
    mapped_chain.sigma = 0.0;
    mapped_chain.sigma_map = Some(vec![value; w * h]);
    let mapped = Setup::new(scene(), mapped_chain.clone()).raw_group();
    let half = HalfSigma {
        width: w / 2,
        height: h / 2,
        data: vec![value; (w / 2) * (h / 2)],
        gains: None,
    };
    let halved = Setup::with_half(scene(), mapped_chain, half);
    let (ow, oh) = scalar.output_size();
    let out = Rect::new(5, 4, ow - 11, oh - 9);
    let a = scalar.gpu(dev, out);
    let b = mapped.gpu(dev, out);
    let c = halved.gpu(dev, out);
    assert_bit_equal("scalar against a map", &a.data, &b.data);
    assert_bit_equal("scalar against a half-resolution plane", &a.data, &c.data);
    // And the three equal the CPU chain (the scalar's) within the chain's own bounds.
    let s = compare(&a, &scalar.cpu(out));
    s.print("scalar sigma against the CPU");
    assert!(s.max_abs < 0.05, "{}", s.max_abs);
    // The half-resolution plane never builds the full map: the device chain has none.
    assert!(halved.gpu_chain.sigma_map.is_none());
    assert!(halved.chain.sigma_map.is_some());
}

/// Chains that skip stages or keep them as identities: every combination of the switches the settings
/// have, so that the bookkeeping of the rectangles a skipped stage leaves behind (the next stage's
/// core is named relative to the image it really gets) is exercised.
/// A named change to a chain.
type ChainTweak = (&'static str, Box<dyn Fn(&mut Chain)>);

fn option_chains() -> Vec<ChainTweak> {
    vec![
        ("all on", Box::new(|_| {})),
        ("no denoise", Box::new(|c| c.denoise = None)),
        (
            "luminance 0 (no denoise, no classical)",
            Box::new(|c| {
                c.denoise = None;
                c.noise_reduction.luminance = 0.0;
            }),
        ),
        (
            "no colour filter",
            Box::new(|c| c.noise_reduction.color = 0.0),
        ),
        (
            "no detail re-injection",
            Box::new(|c| c.noise_reduction.detail = 0.0),
        ),
        (
            "no colour filter, no detail",
            Box::new(|c| {
                c.noise_reduction.color = 0.0;
                c.noise_reduction.detail = 0.0;
            }),
        ),
        ("no sharpening", Box::new(|c| c.sharpen = None)),
        (
            "sharpening with no iteration",
            Box::new(|c| {
                c.sharpen = Some(
                    SharpenParams {
                        amount: 10.0,
                        radius_px: 0.8,
                    }
                    .into(),
                )
            }),
        ),
        (
            "colour only",
            Box::new(|c| {
                c.denoise = None;
                c.noise_reduction = NoiseReduction {
                    luminance: 0.0,
                    color: 80.0,
                    detail: 0.0,
                };
                c.sharpen = None;
            }),
        ),
        (
            "ten iterations, wide PSF",
            Box::new(|c| {
                c.sharpen = Some(
                    SharpenParams {
                        amount: 300.0,
                        radius_px: 1.8,
                    }
                    .into(),
                )
            }),
        ),
        (
            "strong colour and detail",
            Box::new(|c| {
                c.noise_reduction = NoiseReduction {
                    luminance: 90.0,
                    color: 100.0,
                    detail: 100.0,
                };
            }),
        ),
        (
            "nothing after the demosaic",
            Box::new(|c| {
                c.denoise = None;
                c.noise_reduction = NoiseReduction {
                    luminance: 0.0,
                    color: 0.0,
                    detail: 0.0,
                };
                c.sharpen = None;
            }),
        ),
    ]
}

#[test]
fn every_option_of_the_stages_after_the_demosaic_matches_the_cpu_and_the_tiles() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 1700;
    let mut worst_post = 0.0f64;
    for (name, tweak) in option_chains() {
        for lens in [Lens::Vignette, Lens::Full] {
            seed += 1;
            let (w, h) = (141usize, 103usize);
            let scene = Scene::new(w, h, Cfa::RGGB, seed);
            let mut chain = chain_of(&scene, lens, low_noise());
            tweak(&mut chain);
            let setup = Setup::with_half(scene, chain, HalfSigma::new(w, h, 0.01, true, seed));
            let (ow, oh) = setup.output_size();
            let out = Rect::new(7, 5, ow - 15, oh - 11);
            // R6..R8 against R6..R6: the same device demosaic, the CPU against the device after it.
            let mut raw6 = setup.clone();
            raw6.range = StageRange::DEMOSAIC;
            raw6.gpu_chain = Arc::clone(&raw6.chain);
            raw6.half = None;
            let a = setup.gpu(dev, out);
            let b = raw6.gpu(dev, out);
            let s = compare(&a, &b);
            eprintln!(
                "option '{name}' lens {lens:?}: against R6..R6 {:.3e}",
                s.max_abs
            );
            worst_post = worst_post.max(s.max_abs);
            assert!(
                s.max_abs <= POST_STAGES_LARGEST,
                "'{name}' {lens:?}: {}",
                s.max_abs
            );
            if name.starts_with("ten iterations") {
                // Beyond the halo cap (exact radius 2 * 10 * 7 = 140 px against 24): a tile differs
                // from the whole frame, on the CPU's bands as on the device (`docs/05`: about 2e-7 on
                // photographic content; this scene is noisy and clipped, with n = 10).
                let tiles_of = |render: &dyn Fn(Rect) -> RgbImage| {
                    let whole = render(Rect::whole(ow, oh));
                    let mut tiled = whole.clone();
                    for core in ym_gpu::cpu::tiling::tile_grid(ow, oh, 48, 48, 2) {
                        let tile = render(core);
                        tiled.paste(&tile, Rect::whole(core.w, core.h), core.x, core.y);
                    }
                    compare(&tiled, &whole).max_abs
                };
                let on_device = tiles_of(&|r| setup.gpu(dev, r));
                let on_cpu = tiles_of(&|r| setup.cpu(r));
                note!(
                    "[R6..R8] beyond the halo cap ({lens:?}): tiles against whole, device {on_device:.3e}, CPU {on_cpu:.3e}"
                );
                assert!(on_device <= BEYOND_CAP_CHAIN, "{on_device:e}");
            } else {
                // Tiles equal the whole, whatever the stages do.
                assert_tiles_equal_whole(
                    &format!("'{name}' {lens:?}"),
                    ow,
                    oh,
                    3,
                    &[48, 80],
                    |rect| setup.gpu(dev, rect).data,
                );
            }
        }
    }
    note!("[R6..R8] stage options against R6..R6: worst {worst_post:.3e}");
}

#[test]
fn the_raw_group_is_poison_neutral_and_thread_count_independent() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 77);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 3));
    let (ow, oh) = setup.output_size();
    let out = Rect::new(7, 5, ow - 9, oh - 8);
    let got = assert_poison_neutral("the device chain, R6..R8", |gpu| {
        let s = setup.clone();
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            s.device_chain()?.run_tile(ctx, &s.scene.raw(), out, cancel)
        })
        .expect("chain")
        .data
    });
    assert!(got.is_some(), "a plain device is available");
    let _ = dev;
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    denoise_common::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut h = denoise_common::FNV_START;
        for (cfa, lens, noise, variant, seed) in [
            (
                Cfa::RGGB,
                Lens::Full,
                high_noise(),
                Variant::HalfWithGains,
                1801u64,
            ),
            (Cfa::GBRG, Lens::Bare, low_noise(), Variant::Scalar, 1802),
        ] {
            let setup = setup_for(Scene::new(120, 90, cfa, seed), lens, noise, variant);
            let (ow, oh) = setup.output_size();
            h = denoise_common::fnv(h, &setup.gpu(dev, Rect::new(3, 2, ow - 6, oh - 5)).data);
        }
        h
    });
}

/// A runner for `RawWork` that cuts the demosaic core at a chosen limit and counts the rectangles.
struct RawRecording<'a> {
    ctx: &'a GpuContext,
    parts: std::cell::RefCell<Vec<usize>>,
    limit: u64,
}

impl DeviceRunner for RawRecording<'_> {
    fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        let DeviceWork::Raw(w) = work else {
            panic!("this runner is only given the R6..R8 range");
        };
        self.parts.borrow_mut().push(w.demosaic_parts(self.limit));
        w.run_with_limit(self.ctx, cancel, self.limit)
    }
}

#[test]
fn splitting_the_demosaic_core_of_the_raw_group_changes_no_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    // A core cut into many rectangles is put together on the host and uploaded once more; the stages
    // after the demosaic then run on one image. The result is the unsplit run's, bit for bit.
    let mut seed = 1900;
    for cfa in [Cfa::RGGB, Cfa::BGGR] {
        for (w, h, noise) in [(173usize, 131usize, high_noise()), (96, 77, low_noise())] {
            seed += 1;
            let setup = setup_for(
                Scene::new(w, h, cfa, seed),
                Lens::Full,
                noise,
                Variant::HalfWithGains,
            );
            let (ow, oh) = setup.output_size();
            let out = Rect::new(3, 2, ow - 5, oh - 4);
            let s = setup.clone();
            let (whole, cut, counts) = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                    let g = s.device_chain()?;
                    let plain = RawRecording {
                        ctx,
                        parts: Default::default(),
                        limit: MAX_PHYS_AREA,
                    };
                    let whole = g.run_tile(&plain, &s.scene.raw(), out, cancel)?;
                    let cutter = RawRecording {
                        ctx,
                        parts: Default::default(),
                        limit: 4_500,
                    };
                    let cut = g.run_tile(&cutter, &s.scene.raw(), out, cancel)?;
                    Ok((whole, cut, (plain.parts.take(), cutter.parts.take())))
                })
                .expect("chains");
            assert_eq!(counts.0, vec![1]);
            assert!(counts.1.iter().all(|&n| n >= 6), "cut into {:?}", counts.1);
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
fn the_raw_group_with_diffraction_deconvolution_matches_the_cpu() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    use ym_gpu::cpu::sharpen::diffraction::DiffractionParams;
    let (w, h) = (130usize, 100usize);
    let scene = Scene::new(w, h, Cfa::RGGB, 61);
    let mut chain = chain_of(&scene, Lens::Vignette, high_noise());
    // f/16 on a 3.76 um pixel: the plan is active; Richardson-Lucy is off.
    chain.sharpen = Some(DiffractionParams::new(3.76, 16.0).into());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(w, h, 0.01, true, 9));
    let (ow, oh) = setup.output_size();
    let out = Rect::new(5, 4, ow - 11, oh - 9);
    let mut raw6 = setup.clone();
    raw6.range = StageRange::DEMOSAIC;
    raw6.gpu_chain = Arc::clone(&raw6.chain);
    raw6.half = None;
    let a = setup.gpu(dev, out);
    let b = raw6.gpu(dev, out);
    let s = compare(&a, &b);
    s.print("diffraction in the raw group against the CPU after the same demosaic");
    // First measured on lavapipe: 1.2e-4 at one sample of 32,487 (13 over 2e-5); the bound is twice
    // that. (The kernel's own parity is 5e-6 on random frames, `tests/gpu_diffraction.rs`; inside the
    // chain the input is the device's R7 output, whose ulp-level differences the deconvolution's gate
    // and envelope decisions can amplify at a pixel.)
    assert!(s.max_abs <= 2.4e-4, "{}", s.max_abs);
    assert!(compare(&a, &setup.cpu(out)).max_abs.is_finite());
    // The deconvolution changed the picture (the plan is active).
    let mut off = setup.clone();
    let mut no_sharpen = (*setup.chain).clone();
    no_sharpen.sharpen = None;
    off.chain = Arc::new(no_sharpen);
    let mut gpu_off = (*setup.gpu_chain).clone();
    gpu_off.sharpen = None;
    off.gpu_chain = Arc::new(gpu_off);
    let c = off.gpu(dev, out);
    assert!(
        compare(&a, &c).max_abs > 1e-4,
        "the diffraction stage did nothing"
    );
    // Tiles equal the whole (the plan's halo is exact).
    assert_tiles_equal_whole("diffraction in the raw group", ow, oh, 3, &[64], |rect| {
        setup.gpu(dev, rect).data
    });
}

/// Small frames through the whole R6..R8 chain: halos far larger than the frame, mosaics below 2 x 2
/// (the CPU demosaics them, the stages after run on the device), every sample finite. R6..R8 against
/// R6..R6 on the same device demosaic: only the denoise, R7 and R8 kernels differ.
#[test]
fn small_frames_through_the_raw_group_match_the_cpu_stages() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 2000;
    let mut worst = 0.0f64;
    for cfa in [Cfa::RGGB, Cfa::GBRG] {
        for (w, h) in [
            (1usize, 1usize),
            (3, 1),
            (1, 5),
            (2, 2),
            (2, 3),
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
                let setup = Setup::new(scene, chain).raw_group();
                let mut raw6 = setup.clone();
                raw6.range = StageRange::DEMOSAIC;
                let (ow, oh) = setup.output_size();
                let a = setup.gpu(dev, Rect::whole(ow, oh));
                let b = raw6.gpu(dev, Rect::whole(ow, oh));
                let s = compare(&a, &b);
                worst = worst.max(s.max_abs);
                assert!(
                    s.max_abs <= 1e-4,
                    "{} {w}x{h} lens {lens:?}: {}",
                    cfa.name(),
                    s.max_abs
                );
            }
        }
    }
    note!("[R6..R8] small frames, R6..R8 against R6..R6: worst {worst:.3e}");
}

#[test]
fn a_cancelled_raw_group_stops_and_leaves_the_pool_at_its_baseline() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 3);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 4));
    let (ow, oh) = setup.output_size();
    let s = setup.clone();
    let (before, after_ok, after_cancel, ran, err) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let before = ctx.pool().stats().in_use_bytes;
            let g = s.device_chain()?;
            let ok = g.run_tile(
                ctx,
                &s.scene.raw(),
                Rect::whole(ow, oh),
                &CancelToken::new(),
            )?;
            drop(ok);
            let after_ok = ctx.pool().stats().in_use_bytes;
            let token = CancelToken::new();
            token.cancel();
            let counters = ctx.counters();
            let err = g.run_tile(ctx, &s.scene.raw(), Rect::whole(ow, oh), &token);
            let ran = ctx.counters().since(counters);
            // A token set as the work is about to run: the job sees it before it records anything.
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
    assert_eq!((ran.dispatches, ran.submissions), (0, 0));
}

#[test]
fn the_raw_group_records_one_submission_and_one_readback_per_tile() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 5);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 4));
    let (ow, oh) = setup.output_size();
    let s = setup.clone();
    let (raw8, raw6) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let g = s.device_chain()?;
            let before = ctx.counters();
            g.run_tile(ctx, &s.scene.raw(), Rect::whole(ow, oh), cancel)?;
            let raw8 = ctx.counters().since(before);
            let g6 = GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?;
            let before = ctx.counters();
            g6.run_tile(ctx, &s.scene.raw(), Rect::whole(ow, oh), cancel)?;
            Ok((raw8, ctx.counters().since(before)))
        })
        .expect("job");
    eprintln!("[R6..R8] one tile: R6..R8 {raw8:?}, R6 {raw6:?}");
    assert_eq!((raw8.submissions, raw8.readbacks), (1, 1));
    assert_eq!((raw6.submissions, raw6.readbacks), (1, 1));
    assert!(raw8.dispatches > raw6.dispatches + 10);
}

/// Review of the R6..R8 kernels: a tile whose demosaic core is cut (a corner tile of a strong lens)
/// is still one tile for the `tile@N` fault of the fault-injection feature. The raw group made the
/// hook fire twice for such a tile (once for the group, once inside the cut demosaic), so `tile@2`
/// failed the first cut tile and `tile@N` named other tiles than it says.
#[cfg(feature = "fault-injection")]
#[test]
fn a_cut_demosaic_core_is_still_one_tile_for_the_tile_fault() {
    use ym_gpu::device::{Gpu, GpuConfig, GpuFault};

    let Some(_dev) = f32_device_or_skip() else {
        return;
    };
    // The cut is forced by a small limit: the runner below hands the work to `run_with_limit`.
    const LIMIT: u64 = 4_000;
    struct Cutting<'a>(&'a GpuContext);
    impl DeviceRunner for Cutting<'_> {
        fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
            match work {
                DeviceWork::Raw(w) => {
                    assert!(
                        w.demosaic_parts(LIMIT) > 1,
                        "the test needs a demosaic core that is cut"
                    );
                    w.run_with_limit(self.0, cancel, LIMIT)
                }
                other => other.run(self.0, cancel),
            }
        }
    }
    let gpu = Gpu::start(GpuConfig {
        fault: Some(GpuFault::FailTile(2)),
        ..GpuConfig::default()
    })
    .expect("a device with the tile fault armed");
    let scene = Scene::new(150, 110, Cfa::RGGB, 12);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 4));
    let (ow, oh) = setup.output_size();
    let out = Rect::new(7, 5, ow - 9, oh - 8);
    let s = setup.clone();
    let (first, second, third) = gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let g = s.device_chain()?;
            let run = || g.run_tile(&Cutting(ctx), &s.scene.raw(), out, cancel);
            Ok((run(), run(), run()))
        })
        .expect("the job itself runs");
    assert!(first.is_ok(), "tile 1 is not the faulty one: {first:?}");
    assert!(
        matches!(second, Err(GpuError::Readback(_))),
        "tile 2 carries the fault: {second:?}"
    );
    assert!(third.is_ok(), "the fault fires once: {third:?}");
}

// ---------------------------------------------------------------------------------------------
// The whole chain R1..R9 from the resident raw words
// ---------------------------------------------------------------------------------------------
//
// `GpuChain::run_tile_front`: R1, R3, R4 in one pass, R5 with the CPU's chroma offsets, R5b, the
// demosaic, the stages after it and the lens resample are one device job. The CPU chain is the
// reference; a flip of a demosaic decision on a tie is the dominant difference (as in R6..R8), so the
// limits are fractions of samples.
//
// | Case | Asserted | Measured |
// |---|---|---|
// | R1..R9 against the CPU chain, 4 phases x 2 odd sizes x 3 lens kinds x 2 noise models x 4 rectangles | per case [`ONE_CHAIN_CASE`], all cases together [`FRONT_TIE_FREE`] / [`FRONT_TIE_PRONE`] | see the printout |
// | R1..R9 against R6..R8 behind the CPU's front (the same demosaic code, the front differs by its rounding) | largest [`FRONT_AGAINST_HANDOFF`] on tie-free content | see the printout |
// | tiles of two sizes | bit-identical to the whole | 0 differing |
// | a poisoned device and a plain one | bit-identical, finite | 0 differing |
// | a demosaic core cut at a few thousand padded pixels | bit-identical to the unsplit run | 0 differing |
// | `FrontShape::plan` | the CPU's `required_raw_rect` | equal |
// | a lossy file (steps) | within the same limits | see the printout |

/// All tie-free cases together (vignetted content, so off the raw grid). First measured on lavapipe
/// (Mesa 25.2, LLVM 20.1): 214 of 1 228 536 samples over 1e-4 (1.74e-4), 1113 over 2e-5 (9.1e-4),
/// largest 2.9e-3: the R6..R8 numbers to the sample (the front's mosaic is the CPU's to the bit, so the
/// demosaic takes the same decisions; with the vignetting gain formed in f32 the first version had 219,
/// 1467 and 2.9e-3). The limits are twice that.
const FRONT_TIE_FREE: Limits = Limits {
    over: 3.5e-4,
    fine: 1.8e-3,
    largest: 5.8e-3,
};

/// All tie-prone cases together (the bare grid). First measured: 9204 of 638 232 over 1e-4 (1.44e-2),
/// 21 839 over 2e-5 (3.42e-2), largest 0.11: the R6..R8 numbers, the front being bit-identical to the
/// CPU's on content without a vignetting gain.
const FRONT_TIE_PRONE: Limits = Limits {
    over: 2.9e-2,
    fine: 6.9e-2,
    largest: 0.23,
};

/// The 1024 x 1024 corner tile of a 50 MP frame (8658 x 5772) behind a strong barrel lens with CA on
/// the mosaic, a vignetting profile and R5, the whole chain R1..R9 on the device against the CPU chain
/// (`#[ignore]`d). First measured: 467 of 3 145 728 over 1e-4 (1.48e-4), 7955 over 2e-5 (2.53e-3),
/// largest 4.3e-3; the limits are twice that.
const FIFTY_MP_CORNER_FRONT: Limits = Limits {
    over: 3.0e-4,
    fine: 5.1e-3,
    largest: 8.7e-3,
};

/// R1..R9 against R6..R8 behind the CPU's front, largest difference over the tie-free cases. First
/// measured 1.43e-6 (R5b and R9 formed in f32 against the CPU's f64 positions; no sample flips a
/// decision: the first version, with the vignetting gain in f32, had 2.7e-4 at one sample); twice that.
const FRONT_AGAINST_HANDOFF: f64 = 2.9e-6;

/// The R6..R8 setup of a case and the R1..R9 setup of the same case.
fn front_pair(scene: Scene, lens: Lens, noise: NoiseDn, variant: Variant) -> (Setup, Setup) {
    let handoff = setup_for(scene, lens, noise, variant);
    let front = handoff.clone().front();
    (handoff, front)
}

#[test]
fn r1_to_r9_matches_the_cpu_chain() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (mut free, mut prone) = (Stats::default(), Stats::default());
    let mut against_handoff = 0.0f64;
    for (n, (cfa, w, h, lens, nname, noise, seed)) in sweep().into_iter().enumerate() {
        let variant = [Variant::Scalar, Variant::Half, Variant::HalfWithGains][n % 3];
        let (handoff, front) = front_pair(Scene::new(w, h, cfa, seed), lens, noise, variant);
        let (ow, oh) = front.output_size();
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
            let got = front.gpu(dev, out);
            let s = compare(&got, &front.cpu(out));
            let label = format!(
                "{} {w}x{h} lens {lens:?} noise {nname} {variant:?} {out:?}",
                cfa.name()
            );
            assert_within(&label, &s, ONE_CHAIN_CASE);
            if lens.tie_prone() {
                prone.add(&s);
            } else {
                free.add(&s);
                // The same demosaic code on the CPU's front: only the front's rounding and R9's differ.
                let h = compare(&got, &handoff.gpu(dev, out));
                against_handoff = against_handoff.max(h.max_abs);
            }
        }
    }
    free.print("R1..R9 chain, tie-free content");
    prone.print("R1..R9 chain, tie-prone content");
    note!(
        "[raw front] R1..R9 against R6..R8 behind the CPU's front (tie-free): largest {against_handoff:e}"
    );
    assert_within("R1..R9 chain, tie-free content", &free, FRONT_TIE_FREE);
    assert_within("R1..R9 chain, tie-prone content", &prone, FRONT_TIE_PRONE);
    assert!(
        against_handoff <= FRONT_AGAINST_HANDOFF,
        "{against_handoff:e}"
    );
}

#[test]
fn tiles_of_the_front_chain_equal_the_whole_chain_bit_for_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 2500;
    for (cfa, lens, noise, variant) in [
        (Cfa::RGGB, Lens::Full, high_noise(), Variant::HalfWithGains),
        (Cfa::GRBG, Lens::Bare, low_noise(), Variant::Scalar),
        (Cfa::BGGR, Lens::Full, low_noise(), Variant::Half),
        (Cfa::GBRG, Lens::Vignette, high_noise(), Variant::Scalar),
    ] {
        seed += 1;
        let (w, h) = (171usize, 133usize);
        let (_, setup) = front_pair(Scene::new(w, h, cfa, seed), lens, noise, variant);
        let (ow, oh) = setup.output_size();
        assert_tiles_equal_whole(
            &format!("R1..R9 {} lens {:?} {variant:?}", cfa.name(), lens),
            ow,
            oh,
            3,
            &[64, 100],
            |rect| setup.gpu(dev, rect).data,
        );
    }
}

#[test]
fn the_front_chain_of_a_lossy_file_matches_the_cpu_and_the_tiles() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (mut free, mut prone) = (Stats::default(), Stats::default());
    for (n, (cfa, lens)) in [
        (Cfa::RGGB, Lens::Full),
        (Cfa::GBRG, Lens::Vignette),
        (Cfa::BGGR, Lens::Bare),
    ]
    .into_iter()
    .enumerate()
    {
        let (w, h) = (133usize, 97usize);
        let scene = Scene::new(w, h, cfa, 2600 + n as u64).with_steps(9 + n as u64);
        let (_, setup) = front_pair(scene, lens, high_noise(), Variant::HalfWithGains);
        let (ow, oh) = setup.output_size();
        let out = Rect::new(3, 2, ow - 7, oh - 5);
        let s = compare(&setup.gpu(dev, out), &setup.cpu(out));
        assert_within(
            &format!("{} lossy {lens:?}", cfa.name()),
            &s,
            ONE_CHAIN_CASE,
        );
        if lens.tie_prone() {
            prone.add(&s);
        } else {
            free.add(&s);
        }
        assert_tiles_equal_whole(
            &format!("lossy {} {lens:?}", cfa.name()),
            ow,
            oh,
            3,
            &[48],
            |rect| setup.gpu(dev, rect).data,
        );
    }
    free.print("R1..R9 lossy, tie-free content");
    prone.print("R1..R9 lossy, tie-prone content");
    assert_within("R1..R9 lossy, tie-free", &free, FRONT_TIE_FREE);
    assert_within("R1..R9 lossy, tie-prone", &prone, FRONT_TIE_PRONE);
}

#[test]
fn the_front_chain_is_poison_neutral() {
    let Some(_dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 78);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 3)).front();
    let (ow, oh) = setup.output_size();
    let out = Rect::new(7, 5, ow - 9, oh - 8);
    let got = assert_poison_neutral("the device chain, R1..R9", |gpu| {
        let s = setup.clone();
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let raw = s.scene.upload(ctx)?;
            s.device_chain()?.run_tile_front(ctx, &raw, out, cancel)
        })
        .expect("chain")
        .data
    });
    assert!(got.is_some(), "a plain device is available");
}

#[test]
fn the_fronts_rectangles_are_the_cpu_steppers() {
    // `FrontShape::plan` walks the stages backwards with the halos of `Stage::input_rect`, as the CPU
    // stepper does: its first rectangle is the raw rectangle `required_raw_rect` names.
    use ym_gpu::chain::FrontShape;
    for lens in [Lens::Bare, Lens::Vignette, Lens::Full] {
        for highlight_recovery in [false, true] {
            for (w, h) in [(171usize, 133usize), (40, 30), (9, 7)] {
                let scene = Scene::new(w, h, Cfa::RGGB, 5);
                let mut chain = chain_of(&scene, lens, high_noise());
                chain.highlight_recovery = highlight_recovery;
                let frame = Rect::whole(w, h);
                let (ow, oh) = chain.output_size(w, h);
                let stages = chain.stages();
                let d = stages
                    .iter()
                    .position(|s| matches!(s, ym_gpu::cpu::tiling::Stage::Demosaic(_)))
                    .expect("demosaic");
                let shape = FrontShape {
                    highlights: highlight_recovery,
                    ca_halo_px: stages[..d].iter().find_map(|s| match s {
                        ym_gpu::cpu::tiling::Stage::MosaicCa { halo_px } => Some(*halo_px),
                        _ => None,
                    }),
                    frame,
                };
                for out in [
                    Rect::whole(ow, oh),
                    Rect::new(0, 0, 5.min(ow), 3.min(oh)),
                    Rect::new(ow - 4.min(ow), oh - 3.min(oh), 4.min(ow), 3.min(oh)),
                    Rect::new(ow / 2, oh / 2, 1, 1),
                ] {
                    let mut needs = vec![out; stages.len() + 1];
                    for i in (0..stages.len()).rev() {
                        needs[i] = stages[i].input_rect(needs[i + 1], frame);
                    }
                    let plan = shape.plan(needs[d]);
                    assert_eq!(plan.target, needs[d]);
                    assert_eq!(
                        plan.prep,
                        chain.required_raw_rect(w, h, out),
                        "{lens:?} recovery {highlight_recovery} {w}x{h} {out:?}"
                    );
                    // The rectangles nest: what R5b reads is what R5 writes, which R5 reads in `prep`.
                    if let Some((input, core)) = plan.mosaic_ca {
                        assert_eq!(
                            Rect::new(input.x + core.x, input.y + core.y, core.w, core.h),
                            needs[d]
                        );
                    }
                    if let Some(core) = plan.highlights {
                        let inner = plan.mosaic_ca.map_or(needs[d], |(input, _)| input);
                        assert_eq!(
                            Rect::new(plan.prep.x + core.x, plan.prep.y + core.y, core.w, core.h),
                            inner
                        );
                    }
                }
            }
        }
    }
}

/// A runner that cuts the demosaic core of `RawWork` at a chosen limit (the front re-runs its
/// front for each part) and counts the rectangles.
#[test]
fn splitting_the_demosaic_core_of_the_front_chain_changes_no_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 2700;
    for cfa in [Cfa::RGGB, Cfa::BGGR] {
        for (w, h, noise, lens) in [
            (173usize, 131usize, high_noise(), Lens::Full),
            (96, 77, low_noise(), Lens::Vignette),
        ] {
            seed += 1;
            let setup = setup_for(
                Scene::new(w, h, cfa, seed),
                lens,
                noise,
                Variant::HalfWithGains,
            )
            .front();
            let (ow, oh) = setup.output_size();
            let out = Rect::new(3, 2, ow - 5, oh - 4);
            let s = setup.clone();
            let (whole, cut, counts) = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                    let g = s.device_chain()?;
                    let raw = s.scene.upload(ctx)?;
                    let plain = RawRecording {
                        ctx,
                        parts: Default::default(),
                        limit: MAX_PHYS_AREA,
                    };
                    let whole = g.run_tile_front(&plain, &raw, out, cancel)?;
                    let cutter = RawRecording {
                        ctx,
                        parts: Default::default(),
                        limit: 4_500,
                    };
                    let cut = g.run_tile_front(&cutter, &raw, out, cancel)?;
                    Ok((whole, cut, (plain.parts.take(), cutter.parts.take())))
                })
                .expect("chains");
            assert_eq!(counts.0, vec![1]);
            assert!(counts.1.iter().all(|&n| n >= 6), "cut into {:?}", counts.1);
            assert_bit_equal(
                &format!(
                    "{} {w}x{h} lens {lens:?}: R1..R9 cut at 4500 padded pixels against whole",
                    cfa.name()
                ),
                &cut.data,
                &whole.data,
            );
        }
    }
}

#[test]
fn the_front_chain_refuses_what_the_device_does_not_run() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (w, h) = (64usize, 48usize);
    let scene = Scene::new(w, h, Cfa::RGGB, 8);
    let chain = chain_of(&scene, Lens::Full, low_noise());
    let setup = Setup::new(scene, chain).front();
    let (ow, oh) = setup.output_size();
    let s = setup.clone();
    let errors = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let raw = s.scene.upload(ctx)?;
            let g = s.device_chain()?;
            let mut errors = Vec::new();
            // A tile outside the output, and an empty one.
            errors.push(g.run_tile_front(ctx, &raw, Rect::new(0, 0, ow + 1, oh), cancel));
            errors.push(g.run_tile_front(ctx, &raw, Rect::new(0, 0, 0, 3), cancel));
            // Another range, and `run_tile` on this range.
            let other = GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?;
            errors.push(other.run_tile_front(ctx, &raw, Rect::whole(ow, oh), cancel));
            errors.push(g.run_tile(ctx, &s.scene.raw(), Rect::whole(ow, oh), cancel));
            // A resident frame of another size than the lens geometry's.
            let small = Scene::new(w - 2, h - 2, Cfa::RGGB, 8);
            let raw_small = small.upload(ctx)?;
            errors.push(g.run_tile_front(ctx, &raw_small, Rect::whole(ow, oh), cancel));
            Ok(errors)
        })
        .expect("job");
    assert_eq!(errors.len(), 5);
    for (i, e) in errors.iter().enumerate() {
        assert!(
            matches!(e, Err(GpuError::InvalidInput(_))),
            "case {i}: {:?}",
            e.as_ref().map(|_| ())
        );
    }
}

#[test]
fn small_frames_through_the_front_chain_match_the_cpu_chain() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 2800;
    let mut worst = 0.0f64;
    for cfa in [Cfa::RGGB, Cfa::GBRG] {
        for (w, h) in [
            (2usize, 2usize),
            (2, 3),
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
                let setup = Setup::new(scene, chain).front();
                let (ow, oh) = setup.output_size();
                let got = setup.gpu(dev, Rect::whole(ow, oh));
                let want = setup.cpu(Rect::whole(ow, oh));
                // Frames of a few photosites are all halo: every stage sees the edge rules. Equal to
                // the CPU stages' to the demosaic's own flips on these tiny, tie-rich frames.
                let s = compare(&got, &want);
                worst = worst.max(s.max_abs);
                assert!(s.max_abs.is_finite(), "{} {w}x{h} {lens:?}", cfa.name());
                // The device chain from the CPU's front (R6..R8) on the same frame: equal to it up
                // to the front's rounding.
                let hand = {
                    let mut hs = setup.clone();
                    hs.range = StageRange::DEMOSAIC_TO_SHARPEN;
                    hs.gpu(dev, Rect::whole(ow, oh))
                };
                let d = compare(&got, &hand).max_abs;
                assert!(
                    d <= 1e-3,
                    "{} {w}x{h} {lens:?}: front against hand-off {d:e}",
                    cfa.name()
                );
            }
        }
    }
    note!("[raw front] small frames, R1..R9 against the CPU chain: worst {worst:.3e}");
}

#[test]
fn a_cancelled_front_chain_stops_and_leaves_the_pool_at_its_baseline() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 3);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 4)).front();
    let (ow, oh) = setup.output_size();
    let s = setup.clone();
    let (before, after_ok, after_cancel, ran, err) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let raw = s.scene.upload(ctx)?;
            let before = ctx.pool().stats().in_use_bytes;
            let g = s.device_chain()?;
            let ok = g.run_tile_front(ctx, &raw, Rect::whole(ow, oh), &CancelToken::new())?;
            drop(ok);
            let after_ok = ctx.pool().stats().in_use_bytes;
            let token = CancelToken::new();
            token.cancel();
            let counters = ctx.counters();
            let err = g.run_tile_front(ctx, &raw, Rect::whole(ow, oh), &token);
            let ran = ctx.counters().since(counters);
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
            let again = g.run_tile_front(
                &CancelsFirst(ctx, token.clone()),
                &raw,
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
    assert_eq!((ran.dispatches, ran.submissions), (0, 0));
}

#[test]
fn the_front_chain_records_one_submission_and_one_readback_per_tile() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(150, 110, Cfa::RGGB, 5);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 4)).front();
    let (ow, oh) = setup.output_size();
    let s = setup.clone();
    let front = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let raw = s.scene.upload(ctx)?;
            let g = s.device_chain()?;
            let before = ctx.counters();
            g.run_tile_front(ctx, &raw, Rect::whole(ow, oh), cancel)?;
            Ok(ctx.counters().since(before))
        })
        .expect("job");
    eprintln!("[raw front] one tile, R1..R9: {front:?}");
    assert_eq!((front.submissions, front.readbacks), (1, 1));
}

/// A tile of the front chain is one tile for the `tile@N` fault of the fault-injection feature.
#[cfg(feature = "fault-injection")]
#[test]
fn a_front_tile_is_one_tile_for_the_tile_fault() {
    use ym_gpu::device::{Gpu, GpuConfig, GpuFault};

    let Some(_dev) = f32_device_or_skip() else {
        return;
    };
    let gpu = Gpu::start(GpuConfig {
        fault: Some(GpuFault::FailTile(2)),
        ..GpuConfig::default()
    })
    .expect("a device with the tile fault armed");
    let scene = Scene::new(150, 110, Cfa::RGGB, 12);
    let chain = chain_of(&scene, Lens::Full, high_noise());
    let setup = Setup::with_half(scene, chain, HalfSigma::new(150, 110, 0.01, true, 4)).front();
    let (ow, oh) = setup.output_size();
    let out = Rect::new(7, 5, ow - 9, oh - 8);
    let s = setup.clone();
    let (first, second, third) = gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let raw = s.scene.upload(ctx)?;
            let g = s.device_chain()?;
            let run = || g.run_tile_front(ctx, &raw, out, cancel);
            Ok((run(), run(), run()))
        })
        .expect("the job itself runs");
    assert!(first.is_ok(), "tile 1 is not the faulty one: {first:?}");
    assert!(
        matches!(second, Err(GpuError::Readback(_))),
        "tile 2 carries the fault: {second:?}"
    );
    assert!(third.is_ok(), "the fault fires once: {third:?}");
}

/// The corner tile of a 50 MP frame (8658 x 5772) behind a strong barrel lens, from the resident raw
/// words: R1, R3, R4, R5 with the CPU's chroma offsets, R5b, the demosaic (its core cut into parts
/// above the limit, each part re-running the front) and R9 on the device, against the CPU chain.
/// Slow on lavapipe; `cargo test -p ym-gpu --release --test gpu_chain -- --ignored`.
#[test]
#[ignore = "50 MP frame: run with --release -- --ignored"]
fn a_50_mp_frame_corner_tile_runs_the_whole_front_chain_and_is_split() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let (w, h) = (8658usize, 5772usize);
    let tags = LensTags {
        vignetting: vec![0, -400, -1500, -3500, -6000],
        distortion: vec![0, 600, 1800, 3600, 6000, 9000, 12_500, 16_000],
        ca: vec![0, 100, 400, 900, 0, -100, -400, -900],
    };
    let scene = Scene::new(w, h, Cfa::RGGB, 57);
    let mut chain = light_chain_of(&scene, &tags, low_noise());
    chain.vignette = Some(VignetteProfile::from_tag(&tags.vignetting).expect("vignette"));
    chain.highlight_recovery = true;
    chain.lens = Some(
        LensGeometry::new(w, h, &tags, LensSwitches::default())
            .expect("lens")
            .with_ca_on_mosaic()
            .with_auto_crop(),
    );
    let setup = Setup::new(scene, chain).front();
    let (ow, oh) = setup.output_size();
    let out = Rect::new(ow - 1024, oh - 1024, 1024, 1024);
    let s = setup.clone();
    let (gpu, parts) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
            let g = s.device_chain()?;
            let raw = s.scene.upload(ctx)?;
            let rec = RawRecording {
                ctx,
                parts: Default::default(),
                limit: MAX_PHYS_AREA,
            };
            let img = g.run_tile_front(&rec, &raw, out, cancel)?;
            Ok((img, rec.parts.take()))
        })
        .expect("the tile");
    eprintln!("50 MP corner tile {out:?}: the demosaic core is cut into {parts:?} rectangles");
    assert!(parts[0] > 1, "the corner tile of a strong lens is cut");
    let s = compare(&gpu, &setup.cpu(out));
    s.print("50 MP corner tile, R1..R9, against the CPU");
    assert_within("50 MP corner tile, R1..R9", &s, FIFTY_MP_CORNER_FRONT);
}

/// Review of the raw front: a front with neither R5 nor R5b (highlight recovery off, a lens without
/// a CA table) cut into parts. The parts of a cut core start at any photosite, so a part's mosaic
/// can start at an odd one; with no stage to align it the front must still leave exactly the
/// rectangle the demosaic reads. Every part, and the whole, must give the same bits.
#[test]
fn splitting_the_front_chain_without_highlights_and_mosaic_ca_changes_no_bit() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut seed = 2900;
    for cfa in [Cfa::RGGB, Cfa::GRBG] {
        for lens in [Lens::Bare, Lens::Vignette] {
            for (w, h) in [(173usize, 131usize), (96, 77)] {
                seed += 1;
                let scene = Scene::new(w, h, cfa, seed);
                let mut chain = chain_of(&scene, lens, high_noise());
                chain.highlight_recovery = false;
                let setup = Setup::new(scene, chain).front();
                let (ow, oh) = setup.output_size();
                // Odd and even tile origins both, so that some part starts at an odd photosite.
                for out in [
                    Rect::new(3, 2, ow - 5, oh - 4),
                    Rect::new(2, 3, ow - 6, oh - 5),
                ] {
                    let s = setup.clone();
                    let (whole, cut, counts) = dev
                        .gpu
                        .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                            let g = s.device_chain()?;
                            let raw = s.scene.upload(ctx)?;
                            let plain = RawRecording {
                                ctx,
                                parts: Default::default(),
                                limit: MAX_PHYS_AREA,
                            };
                            let whole = g.run_tile_front(&plain, &raw, out, cancel)?;
                            let cutter = RawRecording {
                                ctx,
                                parts: Default::default(),
                                limit: 4_500,
                            };
                            let cut = g.run_tile_front(&cutter, &raw, out, cancel)?;
                            Ok((whole, cut, (plain.parts.take(), cutter.parts.take())))
                        })
                        .unwrap_or_else(|e| {
                            panic!(
                                "{} {w}x{h} lens {lens:?} out {out:?}: no R5, no R5b, cut at 4500: {e:?}",
                                cfa.name()
                            )
                        });
                    assert_eq!(counts.0, vec![1]);
                    assert!(counts.1.iter().all(|&n| n >= 4), "cut into {:?}", counts.1);
                    assert_bit_equal(
                        &format!(
                            "{} {w}x{h} lens {lens:?} out {out:?}: no R5/R5b, cut against whole",
                            cfa.name()
                        ),
                        &cut.data,
                        &whole.data,
                    );
                }
            }
        }
    }
}

/// Review of the raw front: `LP_NUM_THREADS=1` against the default thread count of the software
/// adapter, bit-identical, for the whole chain from the resident words (R1, R3, R4, R5 with the
/// CPU's offsets, R5b, the demosaic, R6 to R8 and R9) — the thread-count check of
/// `thread_count_does_not_change_a_bit` covers R6..R8 only.
#[test]
fn the_front_chain_is_thread_count_independent() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    denoise_common::assert_thread_count_independent(
        "the_front_chain_is_thread_count_independent",
        || {
            let mut h = denoise_common::FNV_START;
            for (cfa, lens, noise, variant, seed) in [
                (
                    Cfa::RGGB,
                    Lens::Full,
                    high_noise(),
                    Variant::HalfWithGains,
                    2901u64,
                ),
                (
                    Cfa::GBRG,
                    Lens::Vignette,
                    low_noise(),
                    Variant::Scalar,
                    2902,
                ),
            ] {
                let setup = setup_for(Scene::new(120, 90, cfa, seed), lens, noise, variant).front();
                let (ow, oh) = setup.output_size();
                h = denoise_common::fnv(h, &setup.gpu(dev, Rect::new(3, 2, ow - 6, oh - 5)).data);
            }
            h
        },
    );
}

/// Review of the raw front: a differential fuzz of the whole front. Random frames (odd and even,
/// tiny to a hundred photosites), phases, lens kinds (none, vignetting, vignetting + distortion +
/// CA on the mosaic), highlight recovery on and off, lossy and lossless words, and random tiles of
/// the output. Two oracles per case: the tile equals the same rectangle of the whole front bit for
/// bit (D5), and the front equals the device chain behind the CPU's own front (R6..R8) up to the
/// front's rounding, which isolates R1 to R5b and R9 from the demosaic's decisions.
#[test]
fn a_fuzz_of_the_front_chain_keeps_tiles_exact_and_the_front_on_the_cpus_values() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut rng = synth::Rng::new(0xF0_07);
    let mut pick = |n: u64| (rng.next_u64() % n) as usize;
    let mut worst_hand = 0.0f64;
    let mut worst_case = String::new();
    let mut cases = 0usize;
    // `YM_FUZZ_CASES=600` runs a longer sweep by hand.
    let total: u64 = std::env::var("YM_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    for case in 0..total {
        let cfa = Cfa::ALL[pick(4)];
        let (w, h) = (6 + pick(90), 6 + pick(90));
        let lens = [Lens::Bare, Lens::Vignette, Lens::Full][pick(3)];
        let recovery = pick(3) != 0;
        let lossy = pick(3) == 0;
        let noise = if pick(2) == 0 {
            low_noise()
        } else {
            high_noise()
        };
        let mut scene = Scene::new(w, h, cfa, 7000 + case);
        if lossy {
            scene = scene.with_steps(8000 + case);
        }
        let mut chain = chain_of(&scene, lens, noise);
        chain.highlight_recovery = recovery;
        let setup = Setup::new(scene, chain).front();
        let (ow, oh) = setup.output_size();
        if ow < 2 || oh < 2 {
            continue;
        }
        let x = pick(ow as u64 - 1);
        let y = pick(oh as u64 - 1);
        let out = Rect::new(x, y, 1 + pick((ow - x) as u64), 1 + pick((oh - y) as u64));
        let label = format!(
            "case {case}: {} {w}x{h} {lens:?} recovery {recovery} lossy {lossy} tile {out:?}",
            cfa.name()
        );
        let s = setup.clone();
        let (tile, whole, hand) = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, cancel| {
                let g = s.device_chain()?;
                let raw = s.scene.upload(ctx)?;
                let tile = g.run_tile_front(ctx, &raw, out, cancel)?;
                let whole = g.run_tile_front(ctx, &raw, Rect::whole(ow, oh), cancel)?;
                let hand_chain =
                    GpuChain::new(&s.gpu_chain, &s.prepared, StageRange::DEMOSAIC_TO_SHARPEN)?;
                let hand = hand_chain.run_tile(ctx, &s.scene.raw(), out, cancel)?;
                Ok((tile, whole, hand))
            })
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_bit_equal(
            &format!("{label}: tile against the whole front"),
            &tile.data,
            &whole.extract(out).data,
        );
        let d = compare(&tile, &hand).max_abs;
        assert!(d.is_finite(), "{label}");
        if d > worst_hand {
            worst_hand = d;
            worst_case = label.clone();
        }
        assert!(
            d <= 5e-3,
            "{label}: the front differs from the CPU front by {d:e}"
        );
        cases += 1;
    }
    note!(
        "[review] front fuzz: {cases} cases, worst front against the CPU front {worst_hand:e} ({worst_case})"
    );
    assert!(
        cases as u64 * 6 >= total * 5,
        "only {cases} of {total} cases ran"
    );
}
