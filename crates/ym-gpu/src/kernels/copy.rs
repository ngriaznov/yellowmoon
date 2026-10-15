//! Copying a rectangle of one device image into another (`docs/gpu-plan.md` G-10): the crop of a
//! tile whose last stages were identities, the assembly of the parts of a demosaic core that was
//! cut ([`crate::chain::MAX_PHYS_AREA`]), and the pieces of a window across tiles. The kernel moves
//! samples and computes on none, so the result has the bits of the source whatever the compiler
//! does to arithmetic (fast-math included); the CPU reference is a copy of the rows
//! ([`crate::cpu::RgbImage::extract`] and `paste`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::Placement;
use crate::cpu::tiling::Rect;
use crate::device::GpuError;
use crate::image::GpuImage;
use crate::stream::PassList;

/// The uniform block of `copy.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyU {
    /// Source row stride, plane length, column and row of the rectangle's first pixel.
    src: [u32; 4],
    /// Destination row stride, plane length, column and row of its first pixel.
    dst: [u32; 4],
    /// Width, height, planes.
    size: [u32; 4],
}

/// The largest storage binding (bytes) a copy needs: the larger of the two images.
#[must_use]
pub fn max_binding_bytes(src: Rect, dst: Rect, planes: usize) -> u64 {
    GpuImage::buffer_bytes(src.w, src.h, planes).max(GpuImage::buffer_bytes(dst.w, dst.h, planes))
}

/// Records the copy of the rectangle `rect` of `src` to `at` (column, row) of `dst`. Both images
/// have the same number of planes.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for images with different plane counts, an empty rectangle or one
/// outside `src`, or a destination that does not hold it at `at`; shader errors.
pub fn encode_copy_rect(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    rect: Rect,
    dst: &GpuImage,
    at: (usize, usize),
) -> Result<(), GpuError> {
    if src.planes != dst.planes {
        return Err(GpuError::InvalidInput(format!(
            "copy: {} planes into {}",
            src.planes, dst.planes
        )));
    }
    if rect.is_empty() || rect.x1() > src.width || rect.y1() > src.height {
        return Err(GpuError::InvalidInput(format!(
            "copy: the rectangle {rect:?} is empty or outside the {} x {} source",
            src.width, src.height
        )));
    }
    if at.0 + rect.w > dst.width || at.1 + rect.h > dst.height {
        return Err(GpuError::InvalidInput(format!(
            "copy: {} x {} pixels at {at:?} do not fit the {} x {} destination",
            rect.w, rect.h, dst.width, dst.height
        )));
    }
    let u = CopyU {
        src: [
            src.stride as u32,
            src.plane_len() as u32,
            rect.x as u32,
            rect.y as u32,
        ],
        dst: [
            dst.stride as u32,
            dst.plane_len() as u32,
            at.0 as u32,
            at.1 as u32,
        ],
        size: [rect.w as u32, rect.h as u32, src.planes as u32, 0],
    };
    let ub = passes.uniform("copy uniform", &u)?;
    passes.dispatch(
        KernelId::Copy,
        "copy rect",
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, dst.buf.binding()),
        ],
        (
            (rect.w as u32).div_ceil(GROUP),
            (rect.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(())
}

/// Records the copy of `rect` of `src` into a new image of its own, which keeps the placement of the
/// rectangle in the frame the source was cut from.
///
/// # Errors
///
/// As [`encode_copy_rect`], plus the pool's errors.
pub fn encode_crop(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    rect: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    let dst = GpuImage::alloc(
        ctx,
        "cropped image",
        rect.w,
        rect.h,
        src.planes,
        src.place.offset(rect.x, rect.y),
    )?;
    encode_copy_rect(passes, src, rect, &dst, (0, 0))?;
    Ok(dst)
}

/// A device image that lies at `at` (column, row) of a larger picture: a band of a reduced group, a
/// tile of the full-resolution group.
#[derive(Debug, Clone)]
pub struct Piece {
    /// Column and row of the image's first pixel in the picture.
    pub at: (usize, usize),
    /// The pixels.
    pub image: Arc<GpuImage>,
}

impl Piece {
    /// The rectangle the piece covers in the picture.
    #[must_use]
    pub fn rect(&self) -> Rect {
        Rect::new(self.at.0, self.at.1, self.image.width, self.image.height)
    }
}

/// Records the window `window` of the picture that `pieces` make (they do not overlap) as one image of
/// its own, the pixel (0, 0) of which is the window's first. A window that is exactly one piece is
/// that piece, with no copy. This is what the sampler reads when the raw group stays on the device:
/// the same samples as a host image cut to the window and uploaded, since the kernel only moves them.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty window, pieces with different plane counts, or pieces that
/// leave part of the window uncovered; the pool's errors.
pub fn encode_gather(
    passes: &mut PassList<'_>,
    pieces: &[Piece],
    window: Rect,
) -> Result<Arc<GpuImage>, GpuError> {
    let Some(first) = pieces.first() else {
        return Err(GpuError::InvalidInput(
            "a window needs at least one piece".into(),
        ));
    };
    let planes = first.image.planes;
    if window.is_empty() || pieces.iter().any(|p| p.image.planes != planes) {
        return Err(GpuError::InvalidInput(format!(
            "the window {window:?} is empty or the pieces differ in planes"
        )));
    }
    let parts: Vec<Rect> = pieces
        .iter()
        .map(|p| p.rect().intersect(&window))
        .filter(|r| !r.is_empty())
        .collect();
    let covered: usize = parts.iter().map(|r| r.w * r.h).sum();
    if covered != window.w * window.h {
        return Err(GpuError::InvalidInput(format!(
            "the pieces cover {covered} of the {} pixels of the window {window:?}",
            window.w * window.h
        )));
    }
    // Pieces that overlap could add up to the window's size and still leave part of it uncovered
    // (an image made by the pool is not cleared): the sum above is a proof only for pieces that do not.
    for (i, a) in parts.iter().enumerate() {
        if parts[i + 1..].iter().any(|b| !a.intersect(b).is_empty()) {
            return Err(GpuError::InvalidInput(format!(
                "the pieces overlap inside the window {window:?}"
            )));
        }
    }
    if let [only] = pieces
        && only.rect() == window
    {
        return Ok(Arc::clone(&only.image));
    }
    let ctx = passes.context();
    let dst = GpuImage::alloc(
        ctx,
        "gathered window",
        window.w,
        window.h,
        planes,
        Placement::whole(window.w, window.h),
    )?;
    for p in pieces {
        let part = p.rect().intersect(&window);
        if part.is_empty() {
            continue;
        }
        let from = Rect::new(part.x - p.at.0, part.y - p.at.1, part.w, part.h);
        encode_copy_rect(
            passes,
            &p.image,
            from,
            &dst,
            (part.x - window.x, part.y - window.y),
        )?;
    }
    Ok(Arc::new(dst))
}

/// The WGSL of the copy kernel.
pub(super) fn body() -> String {
    let mut s = String::from(include_str!("../shaders/copy.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ copy_main(gid); }}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;
    use crate::kernels::prep::tests::uniform_layout;

    #[test]
    fn the_uniform_block_has_the_layout_of_its_rust_struct() {
        let (fields, span) = uniform_layout(KernelId::Copy, "pu");
        let want = [
            ("src", offset_of!(CopyU, src)),
            ("dst", offset_of!(CopyU, dst)),
            ("size", offset_of!(CopyU, size)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<CopyU>());
    }
}
