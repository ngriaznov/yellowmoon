//! R7 colour noise reduction on the GPU (G-06): the guided filter of the two chroma planes.
//!
//! The WGSL port (`shaders/guided.wgsl`) of [`crate::cpu::detail::chroma_guided_whole`], specified in
//! `docs/05` section R7: a guided filter (He et al.) of `R - Y` and `B - Y`, guided by `Y`, with the
//! box radius `round(2 + 6 C / 100)` (2 to 8) and a per-pixel `eps = max(sigma^2 (C / 50)^2, 1e-10)`.
//! [`encode_guided`] is the device-in, device-out entry point; [`chroma_guided`] is the host wrapper
//! the parity tests use.
//!
//! # What runs where
//!
//! Four dispatches over the input rectangle (the CPU's `guided`, in its order): horizontal window
//! means of six quantities (`Y`, `Y^2`, `Cr`, `Y Cr`, `Cb`, `Y Cb`), vertical means and the
//! coefficients `a` and `b` of both chroma planes, horizontal means of the four coefficient planes,
//! vertical means and the output of the core. The intermediate planes are two scratch buffers of
//! `6 + 4` planes of the input's size.
//!
//! A box mean is the mean of the samples that exist (clipped at the input's edge, not mirrored) and
//! every mean is a fixed-length direct sum of at most `2 r + 1 <= 17` samples in `f32`, from the low
//! edge up (D4): nothing runs along a whole row, so a pixel depends on its window and not on where the
//! rectangle starts. The receptive field is `2 r` (two means in series), so a tile whose rectangle
//! carries `2 r` real pixels around its core gives the bits of the whole-frame run (D5).
//!
//! # Precision
//!
//! The CPU's sums are `f64` rounded to `f32` per stored plane; the kernel's are `f32`. The variance
//! `box(Y^2) - mean_Y^2` cancels, so the comparison is made at the output: see the module notes of
//! `shaders/guided.wgsl` and the measured value in `tests/gpu_detail.rs`.
//!
//! # Memory
//!
//! For an input rectangle of `n` pixels: the input's 3 planes, 10 scratch planes in two bindings (6
//! planes and 4 planes), the sigma plane and the core's 3 output planes: at most 20 f32 per pixel
//! ([`BYTES_PER_PIXEL`] bytes), and no binding above 6 planes ([`max_binding_bytes`]).

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::detail::NoiseReduction;
use crate::cpu::tiling::Rect;
use crate::cpu::{LUMA, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::bilateral::{SigmaInput, encode_sigma};
use crate::stream::PassList;

/// The largest box radius the kernel takes: the radius of `C = 100` (`round(2 + 6 C / 100)`).
pub const MAX_RADIUS_PX: usize = 8;

/// Device bytes one input pixel needs at most: 3 input planes, 10 scratch planes, 1 sigma plane, 3
/// output planes, all `f32`.
pub const BYTES_PER_PIXEL: u64 = 4 * (3 + 10 + 1 + 3);

/// The uniform block of `guided.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GdU {
    /// Input width, height, row stride, plane length.
    dims: [u32; 4],
    /// Core `x`, `y`, width and height in the input.
    core: [u32; 4],
    /// Output row stride, output plane length, sigma row stride, radius.
    dst: [u32; 4],
    /// The input's origin in the sigma plane, scratch plane length.
    sig: [u32; 4],
    /// `(C / 50)^2`.
    k: [f32; 4],
    /// The luma weights.
    luma: [f32; 4],
}

/// The box radius of a colour strength `C` (0 to 100): `round(2 + 6 C / 100)`, the CPU's own rule.
#[must_use]
pub fn radius_px(color: f32) -> usize {
    NoiseReduction {
        luminance: 0.0,
        color,
        detail: 0.0,
    }
    .chroma_radius_px()
}

/// The largest storage binding (bytes) a guided stage needs for an input rectangle of `rect` pixels:
/// the six-plane scratch (the input's three planes, the four-plane scratch and the output are
/// smaller). A test checks it against the 128 MiB design binding at
/// [`crate::chain::MAX_PHYS_AREA`] (`docs/gpu-plan.md` K4).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    (6 * rect.w * rect.h) as u64 * 4
}

/// Records the R7 colour guided filter of the `core` rectangle of `input` (device in, device out) and
/// returns the core as a new three-plane image with the input's placement moved to the core.
///
/// `input` is the compute rectangle (windows are clipped at its edge, so a core away from the frame's
/// edge needs `2 r` real pixels around it, r the box radius [`radius_px`] of `color`); `sigma` is the
/// tile's sigma plane ([`encode_sigma`]) and `sigma_origin` the position of the input's pixel `(0, 0)`
/// in it. The chroma planes are the only thing the stage changes: the luma of the output is the
/// luma of the input up to rounding.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a colour strength that is not above 0 (the stage is a no-op: do not
/// run it) or not finite, a radius above [`MAX_RADIUS_PX`], an input that is not three planes, a core
/// that is empty or outside the input, or a sigma plane that does not cover the input; the pool's
/// errors; shader errors.
pub fn encode_guided(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
    sigma: &GpuImage,
    sigma_origin: (usize, usize),
    color: f32,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if !color.is_finite() || color <= 0.0 {
        return Err(GpuError::InvalidInput(format!(
            "the colour guided filter needs a strength above 0, got {color} (0 is a no-op: skip the stage)"
        )));
    }
    let r = radius_px(color);
    if r > MAX_RADIUS_PX {
        return Err(GpuError::InvalidInput(format!(
            "the colour guided filter of strength {color} has a box radius of {r} px, the kernel takes at most {MAX_RADIUS_PX}"
        )));
    }
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "colour guided filter: the input has {} planes, not 3",
            input.planes
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "colour guided filter: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    // The coefficients are formed over the whole input, so the sigma plane must cover all of it.
    if sigma.planes != 1
        || sigma_origin.0 + input.width > sigma.width
        || sigma_origin.1 + input.height > sigma.height
    {
        return Err(GpuError::InvalidInput(format!(
            "colour guided filter: the {} x {} sigma plane at {sigma_origin:?} does not cover the {} x {} input",
            sigma.width, sigma.height, input.width, input.height
        )));
    }
    let plane = input.width * input.height;
    if 6 * plane as u64 > u64::from(u32::MAX) {
        return Err(GpuError::InvalidInput(format!(
            "colour guided filter: {plane} pixels do not fit a 32-bit index"
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "guided rgb",
        core.w,
        core.h,
        3,
        input.place.offset(core.x, core.y),
    )?;
    let s1 = passes.scratch("guided means", 6 * plane as u64 * 4)?;
    let s2 = passes.scratch("guided coefficients", 4 * plane as u64 * 4)?;
    let k = (color / 50.0).powi(2);
    let u = GdU {
        dims: [
            input.width as u32,
            input.height as u32,
            input.stride as u32,
            input.plane_len() as u32,
        ],
        core: [core.x as u32, core.y as u32, core.w as u32, core.h as u32],
        dst: [
            out.stride as u32,
            out.plane_len() as u32,
            sigma.stride as u32,
            r as u32,
        ],
        sig: [
            sigma_origin.0 as u32,
            sigma_origin.1 as u32,
            plane as u32,
            0,
        ],
        k: [k, 0.0, 0.0, 0.0],
        luma: [LUMA[0], LUMA[1], LUMA[2], 0.0],
    };
    let ub = passes.uniform("guided uniform", &u)?;
    let whole = (
        (input.width as u32).div_ceil(GROUP),
        (input.height as u32).div_ceil(GROUP),
        1,
    );
    passes.dispatch(
        KernelId::Guided,
        "guided rows",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, s1.binding()),
        ],
        whole,
    )?;
    passes.dispatch(
        KernelId::GuidedCols,
        "guided columns",
        &[
            (0, ub.binding()),
            (2, s1.binding()),
            (3, s2.binding()),
            (4, sigma.buf.binding()),
        ],
        whole,
    )?;
    passes.dispatch(
        KernelId::GuidedSmoothRows,
        "guided smooth rows",
        &[(0, ub.binding()), (2, s1.binding()), (3, s2.binding())],
        whole,
    )?;
    passes.dispatch(
        KernelId::GuidedOutput,
        "guided output",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, s1.binding()),
            (5, out.buf.binding()),
        ],
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// The colour guided filter on the host: uploads `img`, expands `sigma` for its rectangle, runs
/// [`encode_guided`] on `core` and reads the core back. `color` of 0 or less returns the core of the
/// input unchanged, as the CPU's stage does. The GPU counterpart of
/// [`chroma_guided_whole`](crate::cpu::detail::chroma_guided_whole) when `core` is the whole image.
///
/// # Errors
///
/// As [`encode_guided`] and [`encode_sigma`], plus the readback's.
pub fn chroma_guided(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
    sigma: &SigmaInput<'_>,
    color: f32,
) -> Result<RgbImage, GpuError> {
    if color <= 0.0 {
        if core.is_empty() || core.x1() > img.width || core.y1() > img.height {
            return Err(GpuError::InvalidInput(format!(
                "colour guided filter: the core {core:?} is empty or outside the {} x {} input",
                img.width, img.height
            )));
        }
        return Ok(img.extract(core));
    }
    let input = GpuImage::upload_rgb(ctx, "guided input", img)?;
    let mut passes = PassList::new(ctx, "colour guided filter");
    let rect = Rect::new(img.place.x0, img.place.y0, img.width, img.height);
    let plane = encode_sigma(&mut passes, sigma, rect)?;
    let out = encode_guided(&mut passes, &input, core, &plane, (0, 0), color)?;
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

/// The WGSL of a guided pass: the shader file plus the `main` that runs the pass.
pub(super) fn body(id: KernelId) -> String {
    let function = match id {
        KernelId::Guided => "gd_rows",
        KernelId::GuidedCols => "gd_cols",
        KernelId::GuidedSmoothRows => "gd_smooth_rows",
        KernelId::GuidedOutput => "gd_out",
        _ => return String::new(),
    };
    let mut s = String::from(include_str!("../shaders/guided.wgsl"));
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
        // The auto-generated layout holds what the entry point touches; a host that binds another set
        // fails only on a GPU, so pin it offline.
        let table: [(KernelId, &[u32]); 4] = [
            (KernelId::Guided, &[0, 1, 2]),
            (KernelId::GuidedCols, &[0, 2, 3, 4]),
            (KernelId::GuidedSmoothRows, &[0, 2, 3]),
            (KernelId::GuidedOutput, &[0, 1, 2, 5]),
        ];
        for (id, slots) in table {
            let (module, info) = module_of(id);
            assert_eq!(module.entry_points.len(), 1, "{id:?}");
            assert_eq!(module.entry_points[0].workgroup_size, [GROUP, GROUP, 1]);
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
    fn the_uniform_block_has_the_layout_of_gd_u() {
        let (module, _) = module_of(KernelId::Guided);
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("gp"))
            .expect("the uniform `gp`");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`gp` is not a struct");
        };
        let want = [
            ("dims", offset_of!(GdU, dims)),
            ("core", offset_of!(GdU, core)),
            ("dst", offset_of!(GdU, dst)),
            ("sig", offset_of!(GdU, sig)),
            ("k", offset_of!(GdU, k)),
            ("luma", offset_of!(GdU, luma)),
        ];
        assert_eq!(members.len(), want.len());
        for (m, (name, offset)) in members.iter().zip(want) {
            assert_eq!(m.name.as_deref(), Some(name));
            assert_eq!(m.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<GdU>());
    }

    #[test]
    fn the_sums_and_quotients_are_hidden_from_the_optimiser() {
        // Metal compiles with fast-math: the exact window sums, the refined quotients and the products
        // that are subtracted from numbers of their own size must stay behind `opaque` (a C model of
        // this text under clang -ffast-math gives the strict bits; without the barriers 22 % of the
        // outputs and, at exact ties, 1 in 700 means are an ulp off). Lavapipe does not re-associate,
        // so only the text can pin it.
        let src = include_str!("../shaders/guided.wgsl");
        for needle in [
            "return bitcast<f32>(bitcast<u32>(x) ^ gp.sig.w);",
            "let ea = opaque(a - ad);",
            "let eb = opaque(b - bb);",
            "let rem2 = opaque(opaque(rem - p2.x) - p2.y);",
            "let c = opaque(c1 + opaque(rem2 / n));",
            "let inv = opaque(div_cr(1.0,",
            "let a_cr = opaque(opaque(m[3] - opaque(m[0] * m[2])) * inv);",
            "let fcr = opaque(opaque(m[0] * yl) + m[1]);",
            "opaque(yl * yl)",
            "div_cr(opaque(opaque(yl - opaque(gp.luma.x * red)) - opaque(gp.luma.z * blue)), gp.luma.y)",
        ] {
            assert!(src.contains(needle), "guided.wgsl lost `{needle}`");
        }
        assert!(!src.contains("fma("), "no fused multiply-add on purpose");
    }

    #[test]
    fn the_radius_rule_is_the_cpus() {
        assert_eq!(radius_px(0.0), 2);
        assert_eq!(radius_px(25.0), 4);
        assert_eq!(radius_px(50.0), 5);
        assert_eq!(radius_px(100.0), MAX_RADIUS_PX);
    }
}
