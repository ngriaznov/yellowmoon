//! The noise sigma plane of a photo at half resolution, on the device (`docs/06-noise.md`).
//!
//! The CPU reference is `ym-core`'s `sigma_plane`: R1 and R4 on the mosaic (no vignetting), the four
//! CFA positions of each 2 x 2 cell smoothed by a 5 x 5 box and put through the noise model
//! (`ym_noise::sigma_map::sigma_quad`, with no step plane). The device reads the resident raw words
//! and makes the plane in one pass, one thread per cell, each cell reading its 5 x 5 neighbourhood of
//! cells (100 raw words) and normalising them where it reads them, so the pass needs no scratch plane
//! and no band, and a cell never depends on a tile edge. The sums keep the CPU's order, each operation
//! hidden from fast-math. The 177 ms the CPU plane costs on a 24 MP photo is the fit view's critical
//! path; this replaces it there (`GpuReduced`).

use bytemuck::{Pod, Zeroable};
use ym_noise::sigma_map::{BOX_RADIUS, SigmaParams};

use super::prep::{DeviceRaw, FrameU};
use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::normalize::{Levels, NEGATIVE_FLOOR};
use crate::cpu::{Cfa, Placement};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The uniform block of `sigma.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SigmaU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    fr: FrameU,
    /// Cell columns, cell rows, output row stride.
    cells: [u32; 4],
    /// R4 gain per CFA position of the frame.
    g0: [f32; 4],
    /// `1 / g_total` per position (1 where `g_total` is not positive), as the CPU forms it.
    inv_g: [f32; 4],
    /// `max(g_total, 0)` per position.
    g_tot: [f32; 4],
    /// The model's `delta` per position.
    delta: [f32; 4],
    /// `a`, `b + sigma_row^2`, `q^2 / 12`, the floor of R1.
    k: [f32; 4],
    /// The box filter's `1 / 5` and the quadrature mean's `1 / 4`.
    norm: [f32; 4],
}

/// The largest storage binding (bytes) the pass needs for a frame of `width x height` photosites: the
/// output plane (the raw slabs are bounded by the design binding themselves).
#[must_use]
pub fn max_binding_bytes(width: usize, height: usize) -> u64 {
    GpuImage::buffer_bytes(width / 2, height / 2, 1)
}

/// Records the sigma plane of the photo (`width / 2 x height / 2` cells, one plane).
///
/// `params` are the model and the R4 gains **per CFA position of the frame** (`SigmaParams::new(&model,
/// g0_pos)` with `g0_pos[p] = g0[colour at (p & 1, p >> 1)]`, as `ym-core` builds them); `levels` are
/// the R1 levels at the frame's parity.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a frame below 2 x 2 or levels without a range; the pool's errors;
/// shader errors.
pub fn encode_sigma_plane(
    passes: &mut PassList<'_>,
    raw: &DeviceRaw,
    levels: &Levels,
    params: &SigmaParams,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    let (hw, hh) = (raw.width() / 2, raw.height() / 2);
    if hw == 0 || hh == 0 {
        return Err(GpuError::InvalidInput(
            "sigma plane: the photo is too small".into(),
        ));
    }
    levels
        .validate()
        .map_err(|e| GpuError::InvalidInput(format!("sigma plane: {e}")))?;
    let out = GpuImage::alloc(ctx, "sigma plane", hw, hh, 1, Placement::whole(hw, hh))?;
    let g_total = params.g0;
    let u = SigmaU {
        zero: [0; 4],
        fr: FrameU::new(raw, levels),
        cells: [hw as u32, hh as u32, out.stride as u32, 0],
        g0: params.g0,
        inv_g: g_total.map(|g| if g > 0.0 { 1.0 / g } else { 1.0 }),
        g_tot: g_total.map(|g| g.max(0.0)),
        delta: params.delta,
        k: [
            params.a,
            params.k0,
            params.q_default * params.q_default * (1.0 / 12.0),
            NEGATIVE_FLOOR,
        ],
        norm: [1.0 / ((2 * BOX_RADIUS + 1) as f32), 0.25, 0.0, 0.0],
    };
    let bindings = raw.bindings();
    let ub = passes.uniform("sigma uniform", &u)?;
    passes.dispatch(
        KernelId::Sigma,
        "sigma plane",
        &[
            (0, ub.binding()),
            (1, bindings[0].clone()),
            (2, bindings[1].clone()),
            (5, out.buf.binding()),
        ],
        ((hw as u32).div_ceil(GROUP), (hh as u32).div_ceil(GROUP), 1),
    )?;
    Ok(out)
}

/// The sigma plane on the host: records [`encode_sigma_plane`] and reads the plane back
/// (`width / 2 x height / 2`, dense). For the parity tests.
///
/// # Errors
///
/// As [`encode_sigma_plane`], plus the readback's.
pub fn sigma_plane(
    ctx: &GpuContext,
    raw: &DeviceRaw,
    levels: &Levels,
    params: &SigmaParams,
) -> Result<Vec<f32>, GpuError> {
    let mut passes = PassList::new(ctx, "sigma plane");
    let plane = encode_sigma_plane(&mut passes, raw, levels, params)?;
    let read = plane.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    plane.take(&done, read)
}

/// The per-position R4 gains of a frame whose first photosite has the pattern `cfa`.
#[must_use]
pub fn gains_by_position(cfa: Cfa, g0: [f32; 3]) -> [f32; 4] {
    std::array::from_fn(|p| g0[cfa.color_at(p & 1, p >> 1).plane()])
}

/// The WGSL of the sigma kernel.
pub(super) fn body() -> String {
    let mut s = String::new();
    s.push_str(include_str!("../shaders/front_opaque.wgsl"));
    s.push_str(include_str!("../shaders/front_raw.wgsl"));
    s.push_str(include_str!("../shaders/sigma.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ sigma_main(gid); }}\n"
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
        let (fields, span) = uniform_layout(KernelId::Sigma, "pu");
        let want = [
            ("zero", offset_of!(SigmaU, zero)),
            ("fr", offset_of!(SigmaU, fr)),
            ("cells", offset_of!(SigmaU, cells)),
            ("g0", offset_of!(SigmaU, g0)),
            ("inv_g", offset_of!(SigmaU, inv_g)),
            ("g_tot", offset_of!(SigmaU, g_tot)),
            ("delta", offset_of!(SigmaU, delta)),
            ("k", offset_of!(SigmaU, k)),
            ("norm", offset_of!(SigmaU, norm)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<SigmaU>());
    }
}
