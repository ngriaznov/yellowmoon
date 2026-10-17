//! Helpers shared by the tests of the R6 to R8 kernels (`gpu_denoise.rs`, `gpu_detail.rs`,
//! `gpu_rl.rs`, `gpu_chain.rs`): scenes, NaN-proof error statistics, and the check that the number
//! of threads of the software adapter changes no bit.
//!
//! A directory module (`tests/denoise_common/mod.rs`), so cargo does not build it as a test binary
//! of its own; the files include it with `#[path = "denoise_common/mod.rs"] mod denoise_common;`.

#![allow(dead_code)]

use std::process::Command;

use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Placement, RgbImage};

/// Error of one comparison, NaN-proof: a sample that is NaN on one side only is an infinite error and
/// counts in every "over" figure (`fold(0.0, f32::max)` would skip it).
#[derive(Clone, Copy, Debug)]
pub struct Err {
    /// Largest `|a - b|`.
    pub max: f32,
    /// Index of the largest.
    pub at: usize,
    /// Mean `|a - b|`.
    pub mean: f64,
    /// Samples compared.
    pub n: usize,
}

impl Err {
    /// How many samples differ by more than `bound` (a NaN on one side counts).
    pub fn over(a: &[f32], b: &[f32], bound: f32) -> usize {
        a.iter()
            .zip(b)
            .filter(|&(&x, &y)| diff(x, y).is_nan() || diff(x, y) > bound)
            .count()
    }
}

/// `|a - b|`, infinity for a NaN on one side only, 0 for NaN on both sides or equal infinities.
pub fn diff(a: f32, b: f32) -> f32 {
    if a == b || (a.is_nan() && b.is_nan()) {
        0.0
    } else if a.is_nan() || b.is_nan() {
        f32::INFINITY
    } else {
        (a - b).abs()
    }
}

/// The error of `a` against `b`.
pub fn compare(a: &[f32], b: &[f32]) -> Err {
    assert_eq!(a.len(), b.len(), "lengths differ");
    let mut e = Err {
        max: 0.0,
        at: 0,
        mean: 0.0,
        n: a.len(),
    };
    let mut sum = 0.0f64;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = diff(x, y);
        if d.is_nan() || d > e.max {
            e.max = d;
            e.at = i;
        }
        sum += f64::from(d);
    }
    e.mean = sum / a.len().max(1) as f64;
    e
}

/// A natural-looking scene with Gaussian noise of `sigma` on every channel, `w x h`, placed as the
/// whole frame.
pub fn noisy_scene(w: usize, h: usize, seed: u64, sigma: f32) -> RgbImage {
    let mut img = synth::natural(w, h, seed);
    synth::add_noise(&mut img.data, sigma, seed.wrapping_add(1000));
    img
}

/// A natural scene with strong noise on the red and blue channels only (the case the colour control is
/// for) and a trace on green, `w x h`.
pub fn chroma_noisy_scene(w: usize, h: usize, seed: u64, sigma: f32) -> RgbImage {
    let mut img = synth::natural(w, h, seed);
    let n = img.plane_len();
    let mut rng = synth::Rng::new(seed.wrapping_add(500));
    for i in 0..n {
        img.data[i] += sigma * rng.gaussian();
        img.data[n + i] += 0.2 * sigma * rng.gaussian();
        img.data[2 * n + i] += sigma * rng.gaussian();
    }
    img
}

/// A bright, nearly flat scene (a sky: level `level`, a faint gradient, Gaussian noise `sigma` on every
/// channel, extra noise on red and blue): the case where the variance of the guided filter is a
/// difference of two numbers four orders of magnitude larger than it.
pub fn bright_flat_scene(w: usize, h: usize, seed: u64, level: f32, sigma: f32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let n = img.plane_len();
    let mut rng = synth::Rng::new(seed);
    for y in 0..h {
        for x in 0..w {
            let g = 0.02 * (x as f32 / w.max(1) as f32 - 0.5) + 0.01 * (y as f32 / h.max(1) as f32);
            let i = y * w + x;
            img.data[i] = (level + g) * 1.15 + sigma * rng.gaussian() * 1.4;
            img.data[n + i] = level + g + sigma * rng.gaussian();
            img.data[2 * n + i] = (level + g) * 0.85 + sigma * rng.gaussian() * 1.4;
        }
    }
    img
}

/// Like [`noisy_scene`] but cut out of a larger frame at `(x0, y0)`, so that the position-dependent
/// stages (R8's PSF) see an origin that is not the frame's.
pub fn placed(mut img: RgbImage, x0: usize, y0: usize, frame_w: usize, frame_h: usize) -> RgbImage {
    img.place = Placement {
        x0,
        y0,
        frame_width: frame_w,
        frame_height: frame_h,
    };
    img
}

/// A sigma plane that varies smoothly with a few exact zeros and a hot patch: the shapes the
/// bilateral filter and the damping of R8 have to follow, `w * h` values.
pub fn sigma_plane(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut rng = synth::Rng::new(seed);
    let mut p = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let base = 0.004 + 0.012 * (x as f32 / w.max(1) as f32) + 0.004 * rng.next_f32();
            p[y * w + x] = if (x / 8 + y / 8) % 11 == 3 {
                0.0
            } else if x > w * 3 / 4 && y > h * 3 / 4 {
                base * 6.0
            } else {
                base
            };
        }
    }
    p
}

/// A line for the log and, on GitHub Actions, an annotation on the run's summary page: the numbers a
/// Metal runner measured (the hardware columns of the bounds are 2x the first measurement, which the
/// first Apple run supplies).
pub fn ci_note(what: &str, text: &str) {
    use std::io::Write as _;
    let mut err = std::io::stderr();
    let _ = writeln!(err, "[R6..R8] {text}");
    if std::env::var_os("GITHUB_ACTIONS").is_some() {
        let _ = writeln!(err, "::notice title=R6..R8 {what}::{text}");
    }
}

/// FNV-1a over the bits of `data`, folded into `h`.
pub fn fnv(mut h: u64, data: &[f32]) -> u64 {
    for v in data {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

/// FNV-1a offset basis.
pub const FNV_START: u64 = 0xCBF2_9CE4_8422_2325;

/// The check that the software adapter's thread count changes no bit (`LP_NUM_THREADS=1` against the
/// default, `docs/17-gpu.md` section 12): `compute` runs the cases and returns a hash of every bit
/// they produce. The test calls this with its own name; the function runs `compute` here, then runs the same test in a child
/// process with `LP_NUM_THREADS=1` (the variable is read when the adapter starts, so it cannot be set
/// in a running process) and requires the same hash. In the child it prints the hash and returns.
/// Does nothing when there is no adapter (the caller returns before).
pub fn assert_thread_count_independent(test_name: &str, compute: impl Fn() -> u64) {
    const CHILD: &str = "YM_THREAD_CHECK_CHILD";
    let here = compute();
    if std::env::var_os(CHILD).is_some() {
        println!("THREAD_CHECK_HASH={here:016x}");
        return;
    }
    let exe = std::env::current_exe().expect("the test binary's path");
    let out = Command::new(exe)
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env("LP_NUM_THREADS", "1")
        .output()
        .expect("the child run starts");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the child run failed:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The harness prints `test <name> ... ` before the test's own output, so the hash is not at the
    // start of a line.
    let there = text
        .split_once("THREAD_CHECK_HASH=")
        .and_then(|(_, rest)| rest.get(..16))
        .unwrap_or_else(|| panic!("the child printed no hash:\n{text}"));
    assert_eq!(
        format!("{here:016x}"),
        there,
        "LP_NUM_THREADS=1 gave other bits than the default thread count"
    );
}
