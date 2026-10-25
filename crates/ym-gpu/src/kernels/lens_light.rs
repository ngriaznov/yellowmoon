//! The input of L1 and L2 with the colour of the light it holds, on the device (`docs/21` section 4, *As built
//! (the light's colour)*).
//!
//! The CPU reference is `ym_lens::lit_input`: every pixel goes through `Restore::light_colour` (the clip
//! pattern's cast taken off, then each saturated channel raised to the mean of the channels that still
//! measure), which `lens_light_colour.wgsl` computes in `f32`. A function of one pixel, so a tile and the whole
//! frame give the same bits. The restored light of L3 (`lens_flare.wgsl`) includes the same function.

use bytemuck::{Pod, Zeroable};
pub use ym_lens::LightPass;
use ym_lens::consts::{CLIP_PATTERN_SPREAD, LIGHT_SATURATION_RAMP};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::Placement;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The uniform block of `lens_light.wgsl` (`LightU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightU {
    /// `g0` of each channel, 0.
    clip: [f32; 4],
    /// The clip test's ramp: from, to, 0, 0.
    ramp: [f32; 4],
    /// Input row stride, plane length, 0, 0.
    src: [u32; 4],
    /// Output row stride, plane length, width, height.
    dst: [u32; 4],
}

/// The largest storage binding (bytes) the pass needs for an image of `width x height` pixels.
#[must_use]
pub fn max_binding_bytes(width: usize, height: usize) -> u64 {
    GpuImage::buffer_bytes(width, height, 3)
}

/// Records the pass over `input` (three planes): a new image of the same size and place with every pixel's
/// light in its own colour.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an image that is not three planes; the pool's errors; shader errors.
pub fn encode_light(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    pass: LightPass,
) -> Result<GpuImage, GpuError> {
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "lens light: {} planes, not 3",
            input.planes
        )));
    }
    let ctx = passes.context();
    let dst = GpuImage::alloc(
        ctx,
        "lens lit input",
        input.width,
        input.height,
        3,
        input.place,
    )?;
    let u = LightU {
        clip: [pass.clip[0], pass.clip[1], pass.clip[2], 0.0],
        ramp: [pass.ramp.0, pass.ramp.1, 0.0, 0.0],
        src: [input.stride as u32, input.plane_len() as u32, 0, 0],
        dst: [
            dst.stride as u32,
            dst.plane_len() as u32,
            input.width as u32,
            input.height as u32,
        ],
    };
    let ub = passes.uniform("lens light uniform", &u)?;
    passes.dispatch(
        KernelId::LensLight,
        "lens light",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, dst.buf.binding()),
        ],
        (
            (input.width as u32).div_ceil(GROUP),
            (input.height as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(dst)
}

/// The pass on the host's behalf: uploads `input` (three planes, `width x height`), runs it and reads the
/// planes back. The GPU counterpart of `ym_lens::lit_input`; for tests.
///
/// # Errors
///
/// As [`encode_light`], plus the upload's and the readback's.
pub fn light(
    ctx: &GpuContext,
    input: &[f32],
    width: usize,
    height: usize,
    pass: LightPass,
) -> Result<Vec<f32>, GpuError> {
    let src = GpuImage::upload(
        ctx,
        "lens light input",
        width,
        height,
        3,
        input,
        Placement::whole(width, height),
    )?;
    let mut passes = PassList::new(ctx, "lens light");
    let img = encode_light(&mut passes, &src, pass)?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    img.take(&done, read)
}

/// The constants the colour function of `lens_light_colour.wgsl` reads, from the CPU reference's.
pub(super) fn constants() -> String {
    format!(
        "const LL_CLIP_PATTERN: f32 = {:?};\n\
         const LL_SAT_FROM: f32 = {:?};\n\
         const LL_SAT_TO: f32 = {:?};\n",
        CLIP_PATTERN_SPREAD as f32, LIGHT_SATURATION_RAMP.0 as f32, LIGHT_SATURATION_RAMP.1 as f32,
    )
}

/// The WGSL of the shared colour function: its constants, then its text.
pub(super) fn colour_function() -> String {
    let mut s = constants();
    s.push_str(include_str!("../shaders/lens_light_colour.wgsl"));
    s
}

/// The WGSL of the pass.
pub(super) fn body() -> String {
    let mut s = colour_function();
    s.push_str(include_str!("../shaders/lens_light.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ ll_main(gid); }}\n"
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
        let (fields, span) = uniform_layout(KernelId::LensLight, "pu");
        let want = [
            ("clip", offset_of!(LightU, clip)),
            ("ramp", offset_of!(LightU, ramp)),
            ("src", offset_of!(LightU, src)),
            ("dst", offset_of!(LightU, dst)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<LightU>());
    }
}
