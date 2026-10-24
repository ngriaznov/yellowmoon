//! Shared setup for the lens kernels' tests (`gpu_lens_field.rs`, `gpu_lens_flare.rs`,
//! `gpu_lens_reshape.rs`): the shipped lens data, a real camera's projection, plans for a frame and
//! deterministic images, so the CPU references in `ym_lens` and the kernels read the same inputs.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use ym_gpu::device::{Gpu, GpuConfig, StoragePreference, gpu_required_by_env};
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes,
};

/// The repository's `data` folder.
pub fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// The shipped lenses, loaded once.
pub fn library() -> &'static LensLibrary {
    static LIB: OnceLock<LensLibrary> = OnceLock::new();
    LIB.get_or_init(|| LensLibrary::load(&data_root()).expect("the shipped lens data load"))
}

/// The A7R IV's projection: its channels differ, so the lateral colour shows.
pub fn camera() -> &'static CameraResponse {
    static CAM: OnceLock<CameraResponse> = OnceLock::new();
    CAM.get_or_init(|| {
        let path = data_root().join("cameras/sony/Sony_ILCE-7RM4_380_780_5.json");
        let curves = ym_color::spectral::SpectralCurves::from_path(path).expect("curves");
        CameraResponse::from_curves(&curves).expect("projection")
    })
}

/// A plan for a frame of `w x h` pixels. The clip levels are far above any value of the test images, so
/// no pixel is a source of light and the planes of L3 are empty: the tests of L1 and L2 see no flares.
pub fn plan(amounts: LensAmounts, w: usize, h: usize, axis: Axis) -> LensPlan {
    let frame = FrameInfo {
        width: w,
        height: h,
        axis,
        f_number: Some(5.6),
        clip: [1.0e6; 3],
    };
    LensPlan::new(library(), &amounts, frame, camera()).expect("plan")
}

/// Amounts of a lens.
pub fn amounts(kind: LensKind, flares: f64, edges: f64, anamorphic: f64) -> LensAmounts {
    LensAmounts {
        kind,
        flares,
        edges,
        anamorphic,
    }
}

/// A deterministic image with texture at every scale: planar RGB values in 0.05 to 1.
pub fn noise(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..3 * w * h)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.05 + 0.95 * ((state >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

/// The value of channel `c` of the pixel `(x, y)` of an endless deterministic texture: a hash of the
/// position, so a region far from the origin needs no image of the whole frame.
pub fn texture(c: usize, x: usize, y: usize) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (c as u64 + 1).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    0.05 + 0.95 * ((h >> 40) as f32 / (1u64 << 24) as f32)
}

/// The planes of a plan, made from a lens plane of noise (no out-of-focus background and no light, so
/// the reshaping and the flares have nothing to do and the planes only identify themselves). The lens
/// plane is the frame at the plan's `lens_plane_scale`.
pub fn planes_of(plan: &LensPlan) -> LensPlanes {
    let k = plan.lens_plane_scale();
    let f = plan.frame();
    let (pw, ph) = (f.width.div_ceil(k), f.height.div_ceil(k));
    let data = noise(pw, ph, 99);
    let plane = ImageRef {
        width: pw,
        height: ph,
        x0: 0,
        y0: 0,
        frame_width: pw,
        frame_height: ph,
        data: &data,
    };
    LensPlanes::build(plan, &plane, None, &|| false).expect("planes")
}

/// `rect` of a planar image of `width x height` pixels, as a planar image of its own.
pub fn crop(
    data: &[f32],
    width: usize,
    height: usize,
    rect: ym_gpu::cpu::tiling::Rect,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * rect.w * rect.h);
    for c in 0..3 {
        for y in rect.y..rect.y1() {
            let row = c * width * height + y * width + rect.x;
            out.extend_from_slice(&data[row..row + rect.w]);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// Scenes with lights, for the highlight optics

/// A light: its centre as fractions of the frame, its radius in half-diagonals and its colour (1.0 is the
/// clip level of the plans these tests make).
pub type Light = (f64, f64, f64, [f32; 3]);

/// Lights of every size and colour: big and small, clipped in every channel and in one only.
pub fn lights() -> Vec<Light> {
    vec![
        (0.22, 0.30, 0.012, [1.0, 1.0, 1.0]),
        (0.71, 0.24, 0.020, [1.0, 0.8, 0.5]),
        (0.58, 0.78, 0.006, [1.0, 1.0, 1.0]),
        (0.35, 0.62, 0.0025, [1.0, 1.0, 1.0]),
        (0.86, 0.66, 0.0015, [1.0, 0.9, 1.0]),
        (0.10, 0.90, 0.004, [1.0, 1.0, 1.0]),
        (0.93, 0.07, 0.003, [1.0, 1.0, 1.0]),
        (0.50, 0.50, 0.002, [1.0, 1.0, 1.0]),
    ]
}

/// The value of channel `c` at the full-resolution pixel `(x, y)` of a `w x h` frame: a textured
/// background below the flares' knee (at most 0.4 of the clip level) and the lights.
pub fn lit(c: usize, x: usize, y: usize, w: usize, h: usize, lights: &[Light]) -> f32 {
    let hd = 0.5 * ((w * w + h * h) as f64).sqrt();
    let mut v = 0.4 * texture(c, x, y);
    for (fx, fy, r, colour) in lights {
        let (dx, dy) = (
            x as f64 + 0.5 - fx * w as f64,
            y as f64 + 0.5 - fy * h as f64,
        );
        if (dx * dx + dy * dy).sqrt() <= (r * hd).max(0.8) {
            v = v.max(colour[c]);
        }
    }
    v
}

/// The whole frame of [`lit`], planar.
pub fn lit_image(w: usize, h: usize, lights: &[Light]) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * w * h);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                out.push(lit(c, x, y, w, h, lights));
            }
        }
    }
    out
}

/// The planes of a plan made from the lens plane of a lit scene (the scene sampled at the centres of the
/// plane's `k x k` blocks), so the source list holds the lights.
pub fn lit_planes(plan: &LensPlan, lights: &[Light]) -> LensPlanes {
    let k = plan.lens_plane_scale();
    let f = plan.frame();
    let (pw, ph) = (f.width.div_ceil(k), f.height.div_ceil(k));
    let mut data = Vec::with_capacity(3 * pw * ph);
    for c in 0..3 {
        for y in 0..ph {
            for x in 0..pw {
                let (sx, sy) = (
                    (x * k + k / 2).min(f.width - 1),
                    (y * k + k / 2).min(f.height - 1),
                );
                data.push(lit(c, sx, sy, f.width, f.height, lights));
            }
        }
    }
    let plane = ImageRef {
        width: pw,
        height: ph,
        x0: 0,
        y0: 0,
        frame_width: pw,
        frame_height: ph,
        data: &data,
    };
    LensPlanes::build(plan, &plane, None, &|| false).expect("planes")
}

/// A plan with the clip level 1.0 in every channel (the lit scenes' lights are clipped).
pub fn plan_clipped(amounts: LensAmounts, w: usize, h: usize, axis: Axis) -> LensPlan {
    let frame = FrameInfo {
        width: w,
        height: h,
        axis,
        f_number: Some(8.0),
        clip: [1.0; 3],
    };
    LensPlan::new(library(), &amounts, frame, camera()).expect("plan")
}

/// Serialises the lens tests of one binary: they share one software device whose jobs are slow, and a job
/// that waits behind the others' would pass the device's 30 s hang limit on a loaded machine.
pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// How long a job of the lens kernels may take on the test devices. The kernels are heavy for a software
/// rasteriser (hundreds of taps or items a pixel), and a development machine under load needs minutes for
/// a megapixel; the device's own limit is 30 s.
const JOB_LIMIT: std::time::Duration = std::time::Duration::from_secs(900);

fn start_with(poison: bool) -> Option<Gpu> {
    let config = GpuConfig {
        storage: StoragePreference::Auto,
        poison,
        readback_timeout: JOB_LIMIT,
        ..GpuConfig::default()
    };
    match Gpu::start(config) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            None
        }
    }
}

/// The test device of the lens kernels: poison mode, and a long job limit ([`JOB_LIMIT`]). `None` (and a
/// notice) when there is no adapter.
pub fn gpu() -> Option<&'static super::TestGpu> {
    static GPU: OnceLock<Option<super::TestGpu>> = OnceLock::new();
    GPU.get_or_init(|| {
        let Some(gpu) = start_with(true) else {
            super::skip_notice(
                "no GPU adapter (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)",
            );
            return None;
        };
        println!("GPU tests on:\n{}", gpu.info());
        let mode = gpu.info().storage_mode;
        Some(super::TestGpu { gpu, mode })
    })
    .as_ref()
}

/// A device without poison mode and with the same job limit.
pub fn plain_gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| start_with(false)).as_ref()
}

/// D7: runs `run` on the poisoned device (every buffer starts as NaN) and on a plain one (every new
/// buffer starts as zeros), requires the same bits and no NaN or infinity. Skips (returning `None`)
/// without an adapter.
pub fn assert_poison_neutral(what: &str, run: impl Fn(&Gpu) -> Vec<f32>) -> Option<Vec<f32>> {
    let poisoned = gpu()?;
    let plain = plain_gpu()?;
    let a = run(&poisoned.gpu);
    let b = run(plain);
    super::exact::assert_bit_equal(
        &format!("{what}: poisoned device against plain device"),
        &a,
        &b,
    );
    super::exact::assert_all_finite(&format!("{what} (poisoned device)"), &a);
    Some(a)
}
