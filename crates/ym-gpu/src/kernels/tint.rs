//! The mask view's overlay on the device (`docs/23-nodes.md` section 5.8): a display-linear picture tinted
//! with the mask accent where a node's mask `m` is 1, in place, between output sharpening and C13.
//!
//! `out = display + w m (accent - display)`, `w` = 0.55 (`ym_color::mask::OVERLAY_WEIGHT`), the accent in the
//! output's linear primaries (`ym_color::mask::accent_display_linear`). The pixel's `m` is plane 0 of another
//! image of the same size (the colour pass of the mask chain writes `m` in all three of its planes, as grey
//! code values, [`NodeStop::Mask`](super::color::NodeStop::Mask)). The CPU reference is
//! [`crate::cpu::tint`], which does the arithmetic of `ym_color::mask::overlay_pixel` in `f64`; the kernel
//! does it in `f32`, so the two differ by rounding only (`tests/gpu_tint.rs`: the largest difference over
//! pictures that reach 8 is a few `1e-6`).
//!
//! The kernel writes only the window it is given and only planes 0 to 2, so the letterbox plane of the picture
//! stays as it was. Nothing is read back; [`tint_host`] is the thin wrapper the tests use.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::RgbImage;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The uniform block of `tint.wgsl` (`TintU`), 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TintU {
    /// The picture: row stride, plane length, column and row of the window's first pixel.
    img: [u32; 4],
    /// The mask image: row stride, plane length, column and row of the window's first pixel.
    mask: [u32; 4],
    /// Window width and height.
    size: [u32; 4],
    /// The accent (three linear values) and the weight.
    accent: [f32; 4],
}

/// The largest storage binding (bytes) a tint over `window` needs: the picture, four planes at most.
#[must_use]
pub fn max_binding_bytes(window: Rect) -> u64 {
    GpuImage::buffer_bytes(window.x1(), window.y1(), 4)
}

/// Records the tint of `window` of `picture` (three planes, or four with the letterbox mask) with the mask in
/// plane 0 of `mask`, whose pixel (`window.x`, `window.y`) is the mask of the picture's pixel of the same
/// place (the two images are cut from the same frame). `accent` is in the picture's linear primaries and
/// `weight` the overlay's weight.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the window is empty or outside either image, or the picture has fewer than
/// three planes; shader errors.
pub fn encode_tint(
    passes: &mut PassList<'_>,
    picture: &GpuImage,
    window: Rect,
    mask: &GpuImage,
    accent: [f32; 3],
    weight: f32,
) -> Result<(), GpuError> {
    if picture.planes < 3 {
        return Err(GpuError::InvalidInput(format!(
            "the overlay tints three planes, the picture has {}",
            picture.planes
        )));
    }
    if window.is_empty()
        || window.x1() > picture.width
        || window.y1() > picture.height
        || window.x1() > mask.width
        || window.y1() > mask.height
    {
        return Err(GpuError::InvalidInput(format!(
            "the window {window:?} is empty or outside the {} x {} picture or the {} x {} mask",
            picture.width, picture.height, mask.width, mask.height
        )));
    }
    let u = TintU {
        img: [
            picture.stride as u32,
            picture.plane_len() as u32,
            window.x as u32,
            window.y as u32,
        ],
        mask: [
            mask.stride as u32,
            mask.plane_len() as u32,
            window.x as u32,
            window.y as u32,
        ],
        size: [window.w as u32, window.h as u32, 0, 0],
        accent: [accent[0], accent[1], accent[2], weight],
    };
    let ub = passes.uniform("tint uniform", &u)?;
    passes.dispatch(
        KernelId::Tint,
        "mask overlay",
        &[
            (0, ub.binding()),
            (1, picture.buf.binding()),
            (2, mask.buf.binding()),
        ],
        (
            (window.w as u32).div_ceil(GROUP),
            (window.h as u32).div_ceil(GROUP),
            1,
        ),
    )
}

/// Uploads `image` and a mask plane (`m`, one value a pixel), tints the whole picture and reads it back. The
/// thin wrapper the tests use.
///
/// # Errors
///
/// The errors of [`encode_tint`] and of the upload and readback; [`GpuError::InvalidInput`] when `m` does not
/// have one value for each pixel.
pub fn tint_host(
    ctx: &GpuContext,
    image: &RgbImage,
    m: &[f32],
    accent: [f32; 3],
    weight: f32,
) -> Result<RgbImage, GpuError> {
    let (w, h) = (image.width, image.height);
    if m.len() != w * h {
        return Err(GpuError::InvalidInput(format!(
            "{} mask values for a {w} x {h} picture",
            m.len()
        )));
    }
    let picture = GpuImage::upload_rgb(ctx, "tint picture", image)?;
    let mask = GpuImage::upload(
        ctx,
        "tint mask",
        w,
        h,
        1,
        m,
        crate::cpu::Placement::whole(w, h),
    )?;
    let mut passes = PassList::new(ctx, "mask overlay");
    encode_tint(
        &mut passes,
        &picture,
        Rect::whole(w, h),
        &mask,
        accent,
        weight,
    )?;
    let read = picture.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = picture.take(&done, read)?;
    Ok(RgbImage {
        width: w,
        height: h,
        data,
        place: image.place,
    })
}

/// The WGSL of the tint kernel.
pub(super) fn body() -> String {
    let mut s = String::from(include_str!("../shaders/tint.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ tint_main(gid); }}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    #[test]
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        let source = shader_source(StorageMode::F32, KernelId::Tint);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("tu"))
            .expect("the uniform tu");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("tu is not a struct");
        };
        let fields = [
            ("img", offset_of!(TintU, img)),
            ("mask", offset_of!(TintU, mask)),
            ("size", offset_of!(TintU, size)),
            ("accent", offset_of!(TintU, accent)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<TintU>());
        assert_eq!(size_of::<TintU>(), 64);
    }
}
