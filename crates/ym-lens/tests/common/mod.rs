//! Helpers shared by the integration tests: the synthetic lens folder, the real camera, images and plans.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use ym_color::spectral::SpectralCurves;
use ym_lens::test_support::write_synthetic_lenses;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, halo_px,
};

/// The repository root.
pub fn repo_root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// The synthetic lens library, written and loaded once.
pub fn library() -> &'static LensLibrary {
    static LIB: OnceLock<LensLibrary> = OnceLock::new();
    LIB.get_or_init(|| {
        let dir = tempfile::tempdir().expect("temp dir");
        write_synthetic_lenses(dir.path()).expect("write the synthetic lenses");
        LensLibrary::from_dir(dir.path()).expect("the synthetic lenses load")
    })
}

/// The path of a Sony camera's spectral curves in the repository.
pub fn curves_path(file: &str) -> PathBuf {
    repo_root().join("data/cameras/sony").join(file)
}

/// The A7R IV's projection.
pub fn camera() -> &'static CameraResponse {
    static CAM: OnceLock<CameraResponse> = OnceLock::new();
    CAM.get_or_init(|| {
        let curves = SpectralCurves::from_path(curves_path("Sony_ILCE-7RM4_380_780_5.json"))
            .expect("the A7R IV curves");
        CameraResponse::from_curves(&curves).expect("projection")
    })
}

/// A frame of `w x h` full-resolution pixels.
pub fn frame(w: usize, h: usize, axis: Axis) -> FrameInfo {
    FrameInfo {
        width: w,
        height: h,
        axis,
        f_number: None,
        clip: [1.0; 3],
    }
}

/// A plan of the synthetic library on the A7R IV.
pub fn plan(
    kind: LensKind,
    flares: f64,
    edges: f64,
    anamorphic: f64,
    w: usize,
    h: usize,
    axis: Axis,
) -> LensPlan {
    let amounts = LensAmounts {
        kind,
        flares,
        edges,
        anamorphic,
    };
    LensPlan::new(library(), &amounts, frame(w, h, axis), camera()).expect("plan")
}

/// A deterministic textured planar image with values from 0.05 to 1.
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

/// The whole frame as an image.
pub fn whole(w: usize, h: usize, data: &[f32]) -> ImageRef<'_> {
    ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data,
    }
}

/// A flat plane the size of the plan's lens plane, to build planes from.
pub fn planes_for(plan: &LensPlan) -> LensPlanes {
    let k = plan.lens_plane_scale();
    let f = plan.frame();
    let (w, h) = (f.width.div_ceil(k), f.height.div_ceil(k));
    let data = vec![0.5f32; 3 * w * h];
    LensPlanes::build(plan, &whole(w, h, &data), None, &|| false).expect("planes")
}

/// Cuts the rectangle `rect` out of a full-frame planar image, as an owned buffer.
pub fn crop(data: &[f32], w: usize, h: usize, rect: PixelRect) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * rect.w * rect.h);
    for c in 0..3 {
        for y in rect.y..rect.y + rect.h {
            let row = c * w * h + y * w + rect.x;
            out.extend_from_slice(&data[row..row + rect.w]);
        }
    }
    out
}

/// Runs [`apply`] on `out` of a full-frame image at scale `k`, giving it exactly the input the pass needs: `out`
/// grown by [`halo_px`] and clamped to the frame.
pub fn apply_tile(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    data: &[f32],
    w: usize,
    h: usize,
    out: PixelRect,
) -> Vec<f32> {
    let halo = halo_px(plan, planes, k);
    let need = out.grown(halo, w, h);
    let buf = crop(data, w, h, need);
    let input = ImageRef {
        width: need.w,
        height: need.h,
        x0: need.x,
        y0: need.y,
        frame_width: w,
        frame_height: h,
        data: &buf,
    };
    apply(plan, planes, k, &input, out).expect("apply")
}

/// Splits a `w x h` frame into `nx x ny` tiles whose interior boundaries are not multiples of anything.
pub fn tiles(w: usize, h: usize, nx: usize, ny: usize) -> Vec<PixelRect> {
    let bounds = |len: usize, n: usize| -> Vec<usize> {
        (0..=n)
            .map(|i| {
                if i == 0 || i == n {
                    i * len / n
                } else {
                    i * len / n + 3 * (i % 2) + 1
                }
            })
            .collect()
    };
    let (bx, by) = (bounds(w, nx), bounds(h, ny));
    let mut out = Vec::new();
    for ty in 0..ny {
        for tx in 0..nx {
            out.push(PixelRect::new(
                bx[tx],
                by[ty],
                bx[tx + 1] - bx[tx],
                by[ty + 1] - by[ty],
            ));
        }
    }
    out
}

/// A pixel of the position-hashed texture: 0.05 to 1, the same wherever it is asked for.
pub fn pattern(x: usize, y: usize, c: usize) -> f32 {
    let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (c as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    0.05 + 0.95 * ((z >> 40) as f32 / (1u64 << 24) as f32)
}

/// The textured image over `rect` of a frame, planar.
pub fn region(rect: PixelRect) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * rect.w * rect.h);
    for c in 0..3 {
        for y in rect.y..rect.y + rect.h {
            for x in rect.x..rect.x + rect.w {
                out.push(pattern(x, y, c));
            }
        }
    }
    out
}

/// [`apply`] on `out` of a `fw x fh` frame at scale `k` whose content is the texture, given exactly the input the
/// pass needs.
pub fn apply_textured(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    fw: usize,
    fh: usize,
    out: PixelRect,
) -> Vec<f32> {
    let halo = halo_px(plan, planes, k);
    let need = out.grown(halo, fw, fh);
    let buf = region(need);
    let input = ImageRef {
        width: need.w,
        height: need.h,
        x0: need.x,
        y0: need.y,
        frame_width: fw,
        frame_height: fh,
        data: &buf,
    };
    apply(plan, planes, k, &input, out).expect("apply")
}
