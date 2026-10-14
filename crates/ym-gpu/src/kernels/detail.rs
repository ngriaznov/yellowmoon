//! R7 detail re-injection on the GPU (G-06): `Y += beta * (Y_classical - Y_net)`.
//!
//! The WGSL port (`shaders/detail.wgsl`) of [`crate::cpu::detail::inject_detail`], specified in
//! `docs/05` section R7. In the classical path `Y_net` is the denoised image and `Y_classical` the
//! demosaic before the denoise, so the stage puts back a fraction `beta = 0.15 (D / 50)^2` of what the
//! denoise removed, the same amount on the three channels (`R - Y` and `B - Y` are unchanged).
//!
//! # The pre-denoise copy
//!
//! The CPU keeps a copy of the whole demosaiced image for this stage (`needs_classical`). The stage
//! reads nothing but that copy's luma, so the device keeps one plane: [`encode_classical_luma`] forms
//! it from the image that goes into the denoise, for the pixels the denoise outputs, and
//! [`encode_inject_detail`] reads it at the position of each pixel of the image it corrects. Its
//! value is the one the CPU forms from its copy (the same expression on the same stored planes).
//!
//! The host wrapper [`inject_detail`] is the parity tests' entry point: it takes the denoised image
//! and the classical image, as the CPU function does.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::tiling::Rect;
use crate::cpu::{LUMA, Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The uniform block of `detail.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DtU {
    /// Input width, height, row stride, plane length.
    dims: [u32; 4],
    /// Core `x`, `y`, width and height in the input.
    core: [u32; 4],
    /// Output row stride, output plane length, luma plane row stride.
    dst: [u32; 4],
    /// The input's origin in the luma plane.
    org: [u32; 4],
    /// `beta`.
    k: [f32; 4],
    /// The luma weights.
    luma: [f32; 4],
}

/// The largest storage binding (bytes) a detail stage needs for an input rectangle of `rect` pixels:
/// the three planes of the input.
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    GpuImage::buffer_bytes(rect.w, rect.h, 3)
}

/// Records the luma plane of the `core` rectangle of `input` (the image that goes into the denoise):
/// `Y_classical`, as a new one-plane image of the core's size.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not three planes or a core that is empty or outside
/// it; the pool's errors; shader errors.
pub fn encode_classical_luma(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    check_input(input, core, "classical luma")?;
    let out = GpuImage::alloc(
        ctx,
        "classical luma",
        core.w,
        core.h,
        1,
        Placement::whole(core.w, core.h),
    )?;
    let u = uniform(input, core, out.stride, 0, (0, 0), 0.0, out.stride);
    let ub = passes.uniform("detail uniform", &u)?;
    passes.dispatch(
        KernelId::DetailLuma,
        "classical luma",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, out.buf.binding()),
        ],
        groups(core),
    )?;
    Ok(out)
}

/// Records `Y += beta * (Y_classical - Y)` for the `core` rectangle of `input` and returns the core as
/// a new three-plane image with the input's placement moved to the core. `classical` is the plane of
/// [`encode_classical_luma`] and `classical_origin` the position of the input's pixel `(0, 0)` in it.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not three planes, a core that is empty or outside
/// it, a classical plane that does not cover the core, or a `beta` that is not finite; the pool's
/// errors; shader errors.
pub fn encode_inject_detail(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
    classical: &GpuImage,
    classical_origin: (usize, usize),
    beta: f32,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    check_input(input, core, "detail injection")?;
    if !beta.is_finite() {
        return Err(GpuError::InvalidInput(format!(
            "detail injection: beta is {beta}"
        )));
    }
    if classical.planes != 1
        || classical_origin.0 + core.x1() > classical.width
        || classical_origin.1 + core.y1() > classical.height
    {
        return Err(GpuError::InvalidInput(format!(
            "detail injection: the {} x {} classical luma at {classical_origin:?} does not cover the core {core:?}",
            classical.width, classical.height
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "detail rgb",
        core.w,
        core.h,
        3,
        input.place.offset(core.x, core.y),
    )?;
    let u = uniform(
        input,
        core,
        out.stride,
        out.plane_len(),
        classical_origin,
        beta,
        classical.stride,
    );
    let ub = passes.uniform("detail uniform", &u)?;
    passes.dispatch(
        KernelId::Detail,
        "detail injection",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, classical.buf.binding()),
            (3, out.buf.binding()),
        ],
        groups(core),
    )?;
    Ok(out)
}

fn check_input(input: &GpuImage, core: Rect, what: &str) -> Result<(), GpuError> {
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "{what}: the input has {} planes, not 3",
            input.planes
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "{what}: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    Ok(())
}

fn groups(core: Rect) -> (u32, u32, u32) {
    (
        (core.w as u32).div_ceil(GROUP),
        (core.h as u32).div_ceil(GROUP),
        1,
    )
}

fn uniform(
    input: &GpuImage,
    core: Rect,
    out_stride: usize,
    out_plane: usize,
    origin: (usize, usize),
    beta: f32,
    luma_stride: usize,
) -> DtU {
    DtU {
        dims: [
            input.width as u32,
            input.height as u32,
            input.stride as u32,
            input.plane_len() as u32,
        ],
        core: [core.x as u32, core.y as u32, core.w as u32, core.h as u32],
        dst: [out_stride as u32, out_plane as u32, luma_stride as u32, 0],
        org: [origin.0 as u32, origin.1 as u32, 0, 0],
        k: [beta, 0.0, 0.0, 0.0],
        luma: [LUMA[0], LUMA[1], LUMA[2], 0.0],
    }
}

/// The detail re-injection on the host: `net` is the image to correct, `classical` the image before
/// the denoise (both `w x h`, the same rectangle), the result the core of `net` with the delta added.
/// `beta` of 0 returns the core of `net`, as the CPU's function returns its input. The GPU
/// counterpart of [`inject_detail`](crate::cpu::detail::inject_detail) when `core` is the whole image.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the two images differ in size, plus the errors of the entry
/// points and the readback's.
pub fn inject_detail(
    ctx: &GpuContext,
    net: &RgbImage,
    classical: &RgbImage,
    core: Rect,
    beta: f32,
) -> Result<RgbImage, GpuError> {
    if (net.width, net.height) != (classical.width, classical.height) {
        return Err(GpuError::InvalidInput(format!(
            "detail re-injection needs equal sizes, got {}x{} and {}x{}",
            net.width, net.height, classical.width, classical.height
        )));
    }
    if beta == 0.0 {
        if core.is_empty() || core.x1() > net.width || core.y1() > net.height {
            return Err(GpuError::InvalidInput(format!(
                "detail injection: the core {core:?} is empty or outside the {} x {} input",
                net.width, net.height
            )));
        }
        return Ok(net.extract(core));
    }
    let input = GpuImage::upload_rgb(ctx, "detail input", net)?;
    let before = GpuImage::upload_rgb(ctx, "classical input", classical)?;
    let mut passes = PassList::new(ctx, "detail injection");
    let whole = Rect::whole(net.width, net.height);
    let luma = encode_classical_luma(&mut passes, &before, whole)?;
    let out = encode_inject_detail(&mut passes, &input, core, &luma, (0, 0), beta)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    Ok(RgbImage {
        width: out.width,
        height: out.height,
        place: out.place,
        data,
    })
}

/// The WGSL of a detail pass: the shader file plus the `main` that runs the pass.
pub(super) fn body(id: KernelId) -> String {
    let function = match id {
        KernelId::DetailLuma => "detail_luma",
        KernelId::Detail => "detail_inject",
        _ => return String::new(),
    };
    let mut s = String::from(include_str!("../shaders/detail.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {function}(gid); }}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    fn module_of(id: KernelId) -> (naga::Module, naga::valid::ModuleInfo) {
        let source = shader_source(StorageMode::F32, id);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{id:?}: {}", e.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{id:?}: {e:?}"));
        (module, info)
    }

    #[test]
    fn each_pass_binds_exactly_the_buffers_the_host_binds() {
        let table: [(KernelId, &[u32]); 2] = [
            (KernelId::DetailLuma, &[0, 1, 2]),
            (KernelId::Detail, &[0, 1, 2, 3]),
        ];
        for (id, slots) in table {
            let (module, info) = module_of(id);
            assert_eq!(module.entry_points.len(), 1, "{id:?}");
            let used = info.get_entry_point(0);
            let mut bindings = BTreeSet::new();
            for (handle, var) in module.global_variables.iter() {
                if let Some(b) = &var.binding
                    && !used[handle].is_empty()
                {
                    bindings.insert(b.binding);
                }
            }
            assert_eq!(
                bindings,
                slots.iter().copied().collect::<BTreeSet<_>>(),
                "{id:?}"
            );
        }
    }

    #[test]
    fn the_luma_and_the_delta_are_hidden_from_the_optimiser() {
        // Metal's fast-math compiler would contract the luma's products and the delta into fused
        // multiply-adds (17 % of the samples an ulp off in a C model); the barriers keep the CPU's
        // bits. Lavapipe cannot show it, so the text pins it.
        let src = include_str!("../shaders/detail.wgsl");
        for needle in [
            "return bitcast<f32>(bitcast<u32>(x) ^ dt.org.w);",
            "return opaque(opaque(r + g) + b);",
            "let d = opaque(dt.k.x * opaque(",
        ] {
            assert!(src.contains(needle), "detail.wgsl lost `{needle}`");
        }
        // The uniform's last word of `org` is the runtime zero the host writes.
        let host = include_str!("detail.rs");
        assert!(host.contains("org: [origin.0 as u32, origin.1 as u32, 0, 0],"));
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_dt_u() {
        let (module, _) = module_of(KernelId::Detail);
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("dt"))
            .expect("the uniform `dt`");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`dt` is not a struct");
        };
        let want = [
            ("dims", offset_of!(DtU, dims)),
            ("core", offset_of!(DtU, core)),
            ("dst", offset_of!(DtU, dst)),
            ("org", offset_of!(DtU, org)),
            ("k", offset_of!(DtU, k)),
            ("luma", offset_of!(DtU, luma)),
        ];
        assert_eq!(members.len(), want.len());
        for (m, (name, offset)) in members.iter().zip(want) {
            assert_eq!(m.name.as_deref(), Some(name));
            assert_eq!(m.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<DtU>());
    }
}
