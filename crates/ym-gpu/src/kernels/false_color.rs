//! R6's false-colour suppression after the single-frame denoise on the GPU: R7's colour guided filter where the
//! luma alternates (`docs/algorithms/05-single-frame-denoise.md`, *Integration*).
//!
//! The WGSL port (`shaders/false_color.wgsl`) of [`crate::cpu::false_color::suppress`]: the colour guided filter
//! `G` of R7 ([`encode_guided`], the CPU's bits), a gate `w` that opens where the luma of the working encoding
//! alternates (stripes of 2 to 3 px, where the demosaic leaves false colour and method A keeps it), and
//! `out = x + w (G - x)`. [`encode_false_color`] is the device-in, device-out entry point the device chain records
//! after method A (`chain.rs`, `PostStep::FalseColor`); [`false_color`] and [`false_color_weights`] are the host
//! wrappers the parity tests use.
//!
//! # Dispatches
//!
//! [`DISPATCHES`] = 8: the guided filter's four ([`encode_guided`]), then the gate's four: the luma of the
//! working encoding over the buffer, the alternation with its row sums, the column sums with the root (the
//! energy), and over the core the 5 x 5 maximum, the smoothstep and the blend. The gate reads the luma with
//! indices clamped to the buffer, as the CPU does, so with the stage's halo ([`halo_px`]) a tile equals the whole
//! frame bit for bit.
//!
//! # Precision against the CPU
//!
//! The guided filter gives the CPU's bits (`tests/gpu_detail.rs`). The gate is `f32` against the CPU's `f64`
//! (the luma, the alternation's products, the box sums), with no decision that jumps: `w` is continuous in the
//! energy and the blend in `w`, so the output differs by rounding alone (measured in `tests/gpu_false_color.rs`).
//!
//! # Fast-math (Metal)
//!
//! Every product, sum and difference is behind the OR barrier (`bm_op`), the quotients are correctly rounded
//! (`bm_div`), the root is correctly rounded in integer arithmetic (`fc_sqrt`) and the encoding's power is the
//! kernel's own (`bm_pow`, `shaders/working_encoding.wgsl`, shared with method A); the maxima and clamps are
//! exact. In a clang model of naga's MSL (`docs/17-gpu.md` section 13), the four passes replayed over striped
//! frames give the strict bits under fast-math with contraction and approximate reciprocals in all 1.18 million
//! outputs, and 26 % differ without the barriers.
//!
//! # Memory
//!
//! The guided filter's 10 scratch planes and the gate's 2 (dense planes of the buffer), the guided output and the
//! output (the core): the largest binding is the guided filter's six-plane scratch ([`max_binding_bytes`]).

use bytemuck::{Pod, Zeroable};

use super::bm_dct;
use super::guided::{self, encode_guided};
use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::false_color::{BOX_RADIUS_PX, DILATE_RADIUS_PX, T0, T1};
use crate::cpu::tiling::Rect;
use crate::cpu::{RgbImage, false_color as cpu_false_color};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::bilateral::{SigmaInput, encode_sigma};
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The dispatches [`encode_false_color`] records: the guided filter's four and the gate's four.
pub const DISPATCHES: usize = 8;

/// The context the stage needs on each side at Colour `color` (`cpu::false_color::halo_px`): the guided filter's
/// and the gate's.
#[must_use]
pub fn halo_px(color: f32) -> usize {
    cpu_false_color::halo_px(color)
}

/// The largest storage binding (bytes) the stage needs for a buffer of `rect` pixels: the guided filter's
/// six-plane scratch (the gate's two planes, the input and the outputs are smaller). A test checks it against the
/// 128 MiB design binding at [`crate::chain::MAX_PHYS_AREA`] (`docs/17-gpu.md` K4).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    guided::max_binding_bytes(rect).max(GpuImage::buffer_bytes(rect.w, rect.h, 3))
}

/// The uniform block of `false_color.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FcU {
    /// Buffer width, height, scratch plane length (`width * height`), the barrier word (0).
    dims: [u32; 4],
    /// Input row stride, input plane length, output row stride, output plane length.
    io: [u32; 4],
    /// Core `x`, `y`, width and height in the buffer.
    core: [u32; 4],
    /// Guided output row stride, its plane length, mode (0 blend, 1 weights), 0.
    gd: [u32; 4],
}

/// Checks the input and the core.
fn check(input: &GpuImage, core: Rect) -> Result<(), GpuError> {
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "false-colour suppression: the input has {} planes, not 3",
            input.planes
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "false-colour suppression: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    let n = input.width * input.height;
    if 2 * n as u64 > u64::from(u32::MAX) {
        return Err(GpuError::InvalidInput(format!(
            "false-colour suppression: {n} pixels do not fit a 32-bit index"
        )));
    }
    Ok(())
}

/// Records the gate over `input` and its last pass over `core` into `out`: the blend with the guided output
/// `guided` (`Some((buffer, row stride, plane length))`), or the weights (`None`, `out` one plane).
fn encode_gate(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
    out: &GpuImage,
    guided: Option<(&PooledBuffer, usize, usize)>,
) -> Result<(), GpuError> {
    let n = input.width * input.height;
    let scratch = passes.scratch("false colour scratch", 2 * n as u64 * 4)?;
    let (gd_stride, gd_plane) = guided.map_or((0, 0), |(_, s, p)| (s, p));
    let u = FcU {
        dims: [
            input.width as u32,
            input.height as u32,
            n as u32,
            // The barrier word: +0.0, so that OR-ing it changes no bit.
            0,
        ],
        io: [
            input.stride as u32,
            input.plane_len() as u32,
            out.stride as u32,
            out.plane_len() as u32,
        ],
        core: [core.x as u32, core.y as u32, core.w as u32, core.h as u32],
        gd: [
            gd_stride as u32,
            gd_plane as u32,
            u32::from(guided.is_none()),
            0,
        ],
    };
    let ub = passes.uniform("false colour uniform", &u)?;
    let whole = (
        (input.width as u32).div_ceil(GROUP),
        (input.height as u32).div_ceil(GROUP),
        1,
    );
    for (id, label) in [
        (KernelId::FalseColorLuma, "false colour luma"),
        (KernelId::FalseColorRows, "false colour rows"),
        (KernelId::FalseColorEnergy, "false colour energy"),
    ] {
        let bindings: &[(u32, wgpu::BindingResource<'_>)] = if id == KernelId::FalseColorLuma {
            &[
                (0, ub.binding()),
                (1, input.buf.binding()),
                (2, scratch.binding()),
            ]
        } else {
            &[(0, ub.binding()), (2, scratch.binding())]
        };
        passes.dispatch(id, label, bindings, whole)?;
    }
    // In the weights mode the guided output is not read: the input stands in for its binding.
    let gd: &PooledBuffer = guided.map_or(&input.buf, |(b, ..)| b);
    passes.dispatch(
        KernelId::FalseColorBlend,
        "false colour blend",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, scratch.binding()),
            (3, gd.binding()),
            (4, out.buf.binding()),
        ],
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(())
}

/// Records R6's false-colour suppression of the `core` rectangle of `input` (device in, device out) and returns
/// the core as a new three-plane image with the input's placement moved to the core: the device form of
/// [`crate::cpu::false_color::suppress`] followed by the crop to `core`.
///
/// `input` is the buffer the stage reads (its edges clamp the gate and clip the guided filter's boxes, as on the
/// CPU, so a core away from the frame's edge needs [`halo_px`] real pixels around it); `sigma` is the tile's sigma
/// plane ([`encode_sigma`]) and `sigma_origin` the position of the input's pixel `(0, 0)` in it, as for
/// [`encode_guided`].
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a colour strength that is not above 0 (the stage is a no-op: do not run it), an
/// input that is not three planes, a core that is empty or outside the input, and whatever [`encode_guided`]
/// refuses; the pool's errors; shader errors.
pub fn encode_false_color(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
    sigma: &GpuImage,
    sigma_origin: (usize, usize),
    color: f32,
) -> Result<GpuImage, GpuError> {
    check(input, core)?;
    let g = encode_guided(passes, input, core, sigma, sigma_origin, color)?;
    let (g_stride, g_plane) = (g.stride, g.plane_len());
    // The guided output is read by the recorded blend: it lives as long as the list.
    let g_buf = passes.keep(g.buf);
    let out = GpuImage::alloc(
        passes.context(),
        "false colour rgb",
        core.w,
        core.h,
        3,
        input.place.offset(core.x, core.y),
    )?;
    encode_gate(passes, input, core, &out, Some((&g_buf, g_stride, g_plane)))?;
    Ok(out)
}

/// Records the gate alone: the weight of the guided filter at every pixel of `core` (one plane, `[0, 1]`), the
/// device form of [`crate::cpu::false_color::weights`]. For the tests.
///
/// # Errors
///
/// As [`encode_false_color`], without the guided filter's.
pub fn encode_false_color_weights(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
) -> Result<GpuImage, GpuError> {
    check(input, core)?;
    let out = GpuImage::alloc(
        passes.context(),
        "false colour weights",
        core.w,
        core.h,
        1,
        input.place.offset(core.x, core.y),
    )?;
    encode_gate(passes, input, core, &out, None)?;
    Ok(out)
}

/// The stage on the host: uploads `img`, expands `sigma` for its rectangle in the frame (`img.place`), runs
/// [`encode_false_color`] on `core` and reads the core back. `color` of 0 or less returns the core of the input
/// unchanged, as the CPU's stage does.
///
/// # Errors
///
/// As [`encode_false_color`] and [`encode_sigma`], plus the readback's.
pub fn false_color(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
    sigma: &SigmaInput<'_>,
    color: f32,
) -> Result<RgbImage, GpuError> {
    if color <= 0.0 {
        if core.is_empty() || core.x1() > img.width || core.y1() > img.height {
            return Err(GpuError::InvalidInput(format!(
                "false-colour suppression: the core {core:?} is empty or outside the {} x {} input",
                img.width, img.height
            )));
        }
        return Ok(img.extract(core));
    }
    let input = GpuImage::upload_rgb(ctx, "false colour input", img)?;
    let mut passes = PassList::new(ctx, "false colour");
    let rect = Rect::new(img.place.x0, img.place.y0, img.width, img.height);
    let plane = encode_sigma(&mut passes, sigma, rect)?;
    let out = encode_false_color(&mut passes, &input, core, &plane, (0, 0), color)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    drop((input, plane));
    Ok(RgbImage {
        width: out.width,
        height: out.height,
        place: out.place,
        data,
    })
}

/// The gate on the host: the weights of the `core` rectangle of `img`, row-major (`core.w * core.h`).
///
/// # Errors
///
/// As [`encode_false_color_weights`], plus the upload's and the readback's.
pub fn false_color_weights(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
) -> Result<Vec<f32>, GpuError> {
    let input = GpuImage::upload_rgb(ctx, "false colour input", img)?;
    let mut passes = PassList::new(ctx, "false colour weights");
    let out = encode_false_color_weights(&mut passes, &input, core)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    drop(input);
    Ok(data)
}

/// The constants of the gate, written from the CPU reference (`docs/17-gpu.md` K7).
fn constants() -> String {
    use std::fmt::Write as _;
    let mut s = bm_dct::constants();
    let box_side = 2 * BOX_RADIUS_PX + 1;
    let _ = writeln!(s, "const FC_BOX_R: i32 = {BOX_RADIUS_PX};");
    let _ = writeln!(s, "const FC_DILATE_R: i32 = {DILATE_RADIUS_PX};");
    let _ = writeln!(
        s,
        "const FC_BOX_AREA: f32 = {:?};",
        (box_side * box_side) as f32
    );
    let _ = writeln!(s, "const FC_T0: f32 = {:?};", T0 as f32);
    let _ = writeln!(s, "const FC_SPAN: f32 = {:?};", (T1 - T0) as f32);
    s
}

/// The WGSL of a false-colour kernel: the constants, the shader file, the working encoding and the `main`.
pub(super) fn body(id: KernelId) -> String {
    let function = match id {
        KernelId::FalseColorLuma => "fc_luma",
        KernelId::FalseColorRows => "fc_rows",
        KernelId::FalseColorEnergy => "fc_energy",
        KernelId::FalseColorBlend => "fc_blend",
        _ => return String::new(),
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/false_color.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/working_encoding.wgsl"));
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

    const IDS: [KernelId; 4] = [
        KernelId::FalseColorLuma,
        KernelId::FalseColorRows,
        KernelId::FalseColorEnergy,
        KernelId::FalseColorBlend,
    ];

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
        let table: [(KernelId, &[u32]); 4] = [
            (KernelId::FalseColorLuma, &[0, 1, 2]),
            (KernelId::FalseColorRows, &[0, 2]),
            (KernelId::FalseColorEnergy, &[0, 2]),
            (KernelId::FalseColorBlend, &[0, 1, 2, 3, 4]),
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
    fn the_uniform_block_has_the_layout_of_fc_u() {
        for id in IDS {
            let (module, _) = module_of(id);
            let var = module
                .global_variables
                .iter()
                .map(|(_, v)| v)
                .find(|v| v.name.as_deref() == Some("fp"))
                .expect("the uniform `fp`");
            let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
                panic!("`fp` is not a struct");
            };
            let want = [
                ("dims", offset_of!(FcU, dims)),
                ("io", offset_of!(FcU, io)),
                ("core", offset_of!(FcU, core)),
                ("gd", offset_of!(FcU, gd)),
            ];
            assert_eq!(members.len(), want.len(), "{id:?}");
            for (m, (name, offset)) in members.iter().zip(want) {
                assert_eq!(m.name.as_deref(), Some(name));
                assert_eq!(m.offset as usize, offset, "{id:?}: offset of {name}");
            }
            assert_eq!(*span as usize, size_of::<FcU>(), "{id:?}");
        }
    }

    #[test]
    fn the_constants_are_the_cpus() {
        let src = shader_source(StorageMode::F32, KernelId::FalseColorBlend);
        let constant = |name: &str| -> String {
            let start = src
                .find(&format!("const {name}: "))
                .unwrap_or_else(|| panic!("const {name}"));
            let text = &src[start..];
            text[text.find("= ").expect("=") + 2..text.find(';').expect(";")].to_string()
        };
        assert_eq!(constant("FC_BOX_R"), BOX_RADIUS_PX.to_string());
        assert_eq!(constant("FC_DILATE_R"), DILATE_RADIUS_PX.to_string());
        assert_eq!(constant("FC_BOX_AREA").parse::<f32>().expect("f32"), 25.0);
        assert_eq!(constant("FC_T0").parse::<f32>().expect("f32"), T0 as f32);
        assert_eq!(
            constant("FC_SPAN").parse::<f32>().expect("f32"),
            (T1 - T0) as f32
        );
        // The working encoding's constants are method A's.
        assert!(src.contains("const BM_GAIN: f32 = "));
    }

    #[test]
    fn the_sums_and_quotients_are_hidden_from_the_optimiser() {
        let src = include_str!("../shaders/false_color.wgsl");
        for needle in [
            "return bitcast<f32>(bitcast<u32>(x) | fp.dims.w);",
            "let sx = max(-bm_op(bm_op(c - l) * bm_op(r - c)), 0.0);",
            "return bm_op(sx + sy);",
            "acc = bm_op(acc + fc_alternation(fc_cx(i32(gid.x) + d), gid.y));",
            "fc_scr[gid.y * fp.dims.x + gid.x] = fc_sqrt(bm_div(acc, FC_BOX_AREA));",
            "let t = clamp(bm_div(bm_op(m - FC_T0), FC_SPAN), 0.0, 1.0);",
            "return bm_op(bm_op(t * t) * bm_op(3.0 - bm_op(2.0 * t)));",
            "v = bm_op(xv + bm_op(w * bm_op(fc_gd[c * fp.gd.y + g] - xv)));",
            "bm_op(bm_op(bm_op(BM_KR * er) + bm_op(BM_KG * eg)) + bm_op(BM_KB * eb));",
        ] {
            assert!(src.contains(needle), "false_color.wgsl lost `{needle}`");
        }
        assert!(!src.contains("fma("));
        // The device's sqrt only in the root's fallback.
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect();
        assert_eq!(code.matches(" sqrt(").count(), 1);
    }

    #[test]
    fn the_halo_and_the_largest_binding() {
        assert_eq!(halo_px(60.0), cpu_false_color::halo_px(60.0));
        assert!(halo_px(60.0) >= cpu_false_color::GATE_HALO_PX);
        let r = Rect::new(0, 0, 1000, 700);
        assert_eq!(max_binding_bytes(r), guided::max_binding_bytes(r));
    }
}
