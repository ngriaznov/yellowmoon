//! Shared helpers for the GPU tests: one device per storage mode for the whole test binary, skipping
//! with a clear message when there is no adapter (unless `YM_REQUIRE_GPU=1`, which makes that a
//! failure: the `gpu-lavapipe` CI job sets it), and a seeded random source.
//!
//! **Every test device runs in poison mode** (`GpuConfig::poison`): the pool fills each buffer with
//! NaN before handing it out, so a kernel that reads what it did not write fails its test instead of
//! passing by luck on a fresh, zeroed buffer (`docs/17-gpu.md` D7). [`start_unpoisoned`] gives a
//! device without it, for the tests that compare the two.
//!
//! Further helpers: [`exact`] (tile = whole frame, bit comparisons) and [`limits`] (the 128 MiB
//! design binding).
//!
//! The skip notice goes straight to the process's stderr, not through `eprintln!`: the test harness
//! captures `eprintln!` output of passing tests, so a skipped (and therefore vacuously "passing")
//! GPU test would otherwise say nothing at all in a default `cargo test` run.

#![allow(dead_code)]

use std::io::Write as _;
use std::sync::OnceLock;

use ym_gpu::device::{Gpu, GpuConfig, StorageMode, StoragePreference, gpu_required_by_env};

/// A test device and the mode it uses.
pub struct TestGpu {
    pub gpu: Gpu,
    pub mode: StorageMode,
}

fn start(storage: StoragePreference) -> Option<Gpu> {
    start_with(storage, true)
}

/// A device in `storage` mode without poison mode, or `None` when there is no adapter. The tests
/// that need the default device use [`gpu`]; this one is for comparing a run with and without poison.
pub fn start_unpoisoned(storage: StoragePreference) -> Option<Gpu> {
    start_with(storage, false)
}

fn start_with(storage: StoragePreference, poison: bool) -> Option<Gpu> {
    let config = GpuConfig {
        storage,
        poison,
        ..GpuConfig::default()
    };
    match Gpu::start(config) {
        Ok(g) => Some(g),
        Err(e) => {
            if gpu_required_by_env() && storage == StoragePreference::Auto {
                panic!("YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}");
            }
            None
        }
    }
}

/// Tells the person running the tests (and the CI log) that the GPU tests did not run. Written to
/// the real stderr so the harness's output capture cannot swallow it; on GitHub Actions it is also
/// an annotation, so the skip shows on the run's summary page.
pub fn skip_notice(reason: &str) {
    let mut err = std::io::stderr();
    let _ = writeln!(
        err,
        "\n********************************************************************\n\
         * GPU TESTS SKIPPED: {reason}\n\
         * (their loops are empty, so they pass without testing anything;\n\
         *  YM_REQUIRE_GPU=1 turns this skip into a failure)\n\
         ********************************************************************\n"
    );
    if std::env::var_os("GITHUB_ACTIONS").is_some() {
        let _ = writeln!(err, "::warning title=GPU tests skipped::{reason}");
    }
}

/// The devices available here: the automatic one first, then the other storage modes the adapter
/// supports (each is a separate device so every mode is exercised).
pub fn gpus() -> &'static [TestGpu] {
    static GPUS: OnceLock<Vec<TestGpu>> = OnceLock::new();
    GPUS.get_or_init(|| {
        let mut out = Vec::new();
        let Some(auto) = start(StoragePreference::Auto) else {
            skip_notice("no GPU adapter (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)");
            return out;
        };
        let auto_mode = auto.info().storage_mode;
        eprintln!("GPU tests on:\n{}", auto.info());
        out.push(TestGpu {
            gpu: auto,
            mode: auto_mode,
        });
        for mode in [
            StorageMode::F32,
            StorageMode::F16Packed,
            StorageMode::F16Native,
        ] {
            if mode == auto_mode {
                continue;
            }
            if let Some(gpu) = start(StoragePreference::Force(mode)) {
                out.push(TestGpu { gpu, mode });
            }
        }
        out
    })
}

/// The automatic device, or `None` (and a message) when there is no adapter.
pub fn gpu() -> Option<&'static TestGpu> {
    gpus().first()
}

/// SplitMix64: a small seeded generator, so the tests are deterministic without extra crates.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Rounds to the nearest f16 value (what the GPU's f16 storage holds), as f32.
pub fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

pub mod chains;
pub mod exact;
pub mod lens;
pub mod limits;
