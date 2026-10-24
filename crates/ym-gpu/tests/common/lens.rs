//! Shared setup for the lens kernels' tests (`gpu_lens_field.rs`, `gpu_lens_flare.rs`,
//! `gpu_lens_reshape.rs`): the shipped lens data, a real camera's projection, plans for a frame and
//! deterministic images, so the CPU references in `ym_lens` and the kernels read the same inputs.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

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
pub fn crop(data: &[f32], width: usize, height: usize, rect: ym_gpu::cpu::tiling::Rect) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * rect.w * rect.h);
    for c in 0..3 {
        for y in rect.y..rect.y1() {
            let row = c * width * height + y * width + rect.x;
            out.extend_from_slice(&data[row..row + rect.w]);
        }
    }
    out
}
