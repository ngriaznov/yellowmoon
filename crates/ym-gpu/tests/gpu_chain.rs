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
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

/// The fraction of samples allowed to differ from the CPU chain by more than 1e-4 (the kernel's
/// own bound for synthetic content).
const FRACTION: f64 = 1e-4;
/// The fraction over 2e-5, where a systematic error would show first.
const FINE_FRACTION: f64 = 1e-4;
/// The largest difference allowed anywhere: a flipped decision moves one pixel by the difference of
/// two plausible estimates (`tests/gpu_adaptive_ratio.rs`).
const BOUND: f64 = 5e-2;

/// The device with `f32` storage (the automatic one, since the device foundation).
fn f32_device_or_skip() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

// ---------------------------------------------------------------------------------------------
// Scenes and chains
// ---------------------------------------------------------------------------------------------

/// A raw frame: noisy natural content exposed so that highlights clip, 14-bit words around a black
/// level of 512.
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
                let x = 512.0 + (v * 1.3 + 0.004 * rng.gaussian()) * 15488.0;
                x.clamp(0.0, 16000.0) as u16
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lens {
    None,
    /// Vignetting, distortion and a CA table; the CA is corrected on the mosaic (R5b), as the render
    /// does.
    Full,
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
            black: [512.0, 514.0, 513.0, 511.0],
            white: [16000.0; 4],
        },
        vignette: (lens == Lens::Full)
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
    let mut chain = chain_of(scene, Lens::None, noise);
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

/// Everything one run needs, owned so that it can move to the device thread.
#[derive(Clone)]
struct Setup {
    scene: Arc<Scene>,
    chain: Arc<Chain>,
    prepared: Prepared,
}

impl Setup {
    fn new(scene: Scene, chain: Chain) -> Setup {
        let prepared = chain.prepare(&scene.raw()).expect("prepare");
        Setup {
            scene: Arc::new(scene),
            chain: Arc::new(chain),
            prepared,
        }
    }

    fn output_size(&self) -> (usize, usize) {
        self.chain
            .output_size(self.scene.width, self.scene.height)
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
                GpuChain::new(&s.chain, &s.prepared, StageRange::DEMOSAIC)?.run_tile(
                    ctx,
                    &s.scene.raw(),
                    out,
                    cancel,
                )
            })
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

fn assert_close(label: &str, s: &Stats) {
    assert!(s.max_abs <= BOUND, "{label}: largest difference {}", s.max_abs);
    assert!(
        (s.over as f64) <= FRACTION * s.n as f64,
        "{label}: {} of {} samples over 1e-4",
        s.over,
        s.n
    );
    assert!(
        (s.fine as f64) <= FINE_FRACTION * s.n as f64,
        "{label}: {} of {} samples over 2e-5",
        s.fine,
        s.n
    );
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn r6_matches_the_cpu_chain() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let mut total = Stats::default();
    let mut seed = 300;
    for cfa in Cfa::ALL {
        for (w, h) in [(133usize, 97usize), (96, 80)] {
            for lens in [Lens::None, Lens::Full] {
                for (nname, noise) in [("low", low_noise()), ("high", high_noise())] {
                    seed += 1;
                    let setup = Setup::new(
                        Scene::new(w, h, cfa, seed),
                        chain_of(&Scene::new(w, h, cfa, seed), lens, noise),
                    );
                    let (ow, oh) = setup.output_size();
                    // The whole output and tiles on the frame edges and in the middle.
                    for out in [
                        Rect::whole(ow, oh),
                        Rect::new(0, 0, 40.min(ow), 31.min(oh)),
                        Rect::new(ow.saturating_sub(37), oh.saturating_sub(29), 37.min(ow), 29.min(oh)),
                        Rect::new(ow / 3, oh / 3, 33.min(ow - ow / 3), 21.min(oh - oh / 3)),
                    ] {
                        let s = compare(&setup.gpu(dev, out), &setup.cpu(out));
                        let label = format!(
                            "{} {w}x{h} lens {} noise {nname} {out:?}",
                            cfa.name(),
                            lens == Lens::Full
                        );
                        assert_close(&label, &s);
                        total.n += s.n;
                        total.over += s.over;
                        total.fine += s.fine;
                        total.max_abs = total.max_abs.max(s.max_abs);
                    }
                }
            }
        }
    }
    eprintln!(
        "R6..R6 against the CPU chain: {} samples, {} over 1e-4, {} over 2e-5, largest {:e}",
        total.n, total.over, total.fine, total.max_abs
    );
}

#[test]
fn the_shadow_matching_switch_follows_the_noise_model_on_both_sides() {
    // The device decides shadow colour matching from the same `Derived` the CPU does, so a noisy
    // chain exercises the 12-plane path and a quiet one the 5-plane path; both must agree with the
    // CPU.
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    let scene = Scene::new(120, 90, Cfa::RGGB, 9);
    for (noise, matching) in [(low_noise(), false), (high_noise(), true)] {
        let chain = chain_of(&scene, Lens::None, noise);
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
        (Cfa::GRBG, Lens::None, low_noise()),
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
    let mut chain = chain_of(&scene, Lens::None, low_noise());
    let prepared = chain.prepare(&scene.raw()).expect("prepare");
    for range in [
        StageRange {
            first: RawStage::Demosaic,
            last: RawStage::Sharpen,
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
    assert!(StageRange::DEMOSAIC.contains(RawStage::Demosaic));
    assert!(!StageRange::DEMOSAIC.contains(RawStage::Denoise));
    // Every other R6 method runs on the CPU only.
    for method in [Method::Menon, Method::Lmmse, Method::Bilinear, Method::LogGuided] {
        chain.method = method;
        let err = GpuChain::new(&chain, &prepared, StageRange::DEMOSAIC).expect_err("method");
        assert!(matches!(err, GpuError::InvalidInput(_)), "{method:?}: {err}");
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
        let DeviceWork::Demosaic(w) = work;
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
                &format!("{} {w}x{h}: cut at 4500 padded pixels against whole", cfa.name()),
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
            let mut run = s.chain.begin_tile(&s.scene.raw(), &s.prepared, out).expect("tile");
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
            Ok((split, unsplit, (real.parts.take(), raised.parts.take()), (binding, core)))
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
    assert_close("strong lens corner against the CPU", &s);
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
    let gpu = setup.gpu(dev, out);
    let s = compare(&gpu, &setup.cpu(out));
    assert_close("50 MP corner tile", &s);
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
            let ok = g.run_tile(ctx, &s.scene.raw(), Rect::whole(ow, oh), &CancelToken::new())?;
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
    assert_eq!(after_cancel, before, "cancelled tiles release their buffers");
    assert!(matches!(err.0, Err(GpuError::Cancelled)), "{:?}", err.0);
    assert!(matches!(err.1, Err(GpuError::Cancelled)), "{:?}", err.1);
    assert_eq!(cancelled_run.dispatches, 0, "no device work for a set token");
    assert_eq!(cancelled_run.submissions, 0);
}

#[test]
fn a_mosaic_below_two_by_two_is_the_cpu_stage_and_equals_the_cpu_chain() {
    let Some(dev) = f32_device_or_skip() else {
        return;
    };
    for (w, h) in [(1usize, 1usize), (2, 2), (3, 1), (1, 5)] {
        let scene = Scene::new(w.max(2), h.max(2), Cfa::RGGB, 5);
        let mut data = scene.data.clone();
        data.truncate(w * h);
        let tiny = Scene {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data,
        };
        let mut chain = chain_of(&tiny, Lens::None, low_noise());
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
        assert_bit_equal(&format!("{w}x{h}"), &gpu.data, &cpu.data);
    }
}

#[test]
fn the_work_refuses_a_bad_mosaic_before_it_reaches_the_device() {
    let scene = Scene::new(40, 30, Cfa::RGGB, 8);
    let chain = chain_of(&scene, Lens::None, low_noise());
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
