//! R6's coarse colour pass after method A on the GPU (`docs/algorithms/05-single-frame-denoise.md`, *Coarse
//! colour*).
//!
//! The WGSL port (`shaders/coarse_chroma.wgsl`) of [`CoarseChroma::apply`]: the mean Y, Cb and Cr of the working
//! encoding over cells of [`CELL_PX`] x [`CELL_PX`] pixels anchored to the frame, a bilateral filter of the cells'
//! colour within the photo's chroma noise (`h = strength σC(cell)`, method A's `σC` from the shared
//! `working_encoding.wgsl`), and the corrections interpolated back to the pixels as a change of Cb and Cr that
//! leaves Y as it is. [`encode_coarse_chroma`] is the device-in, device-out entry point the device chain records
//! after the false-colour suppression (`chain.rs`, `PostStep::CoarseChroma`); [`coarse_chroma`] is the host
//! wrapper the parity tests use.
//!
//! # Dispatches
//!
//! [`DISPATCHES`] = 3: the cells' means and widths over the buffer's cells, the corrections over the same cells, and the
//! interpolation and change over the core. The cells and the interpolation are positions of the frame and the
//! reads are cut at the buffer's cells as on the CPU, so with the stage's halo ([`CoarseChroma::halo_px`]) a
//! tile equals the whole frame bit for bit.
//!
//! # Precision against the CPU
//!
//! `f32` against the CPU's `f64` (the encoding, the means, `σC`, the weights and their sums, the
//! interpolation), with no decision that jumps: everything is continuous in the input, so the output differs by
//! rounding alone (measured in `tests/gpu_coarse_chroma.rs`).
//!
//! # Fast-math (Metal)
//!
//! Every product, sum and difference is behind the OR barrier (`bm_op`), the quotients are correctly rounded
//! (`bm_div`), the encoding's power and the weight's exponential are the kernel's own (`bm_pow`, `bm_exp2`, the
//! exponent formed as an exact pair), and `σC`'s root is correctly rounded in integers (`bm_sqrt`), all from
//! `shaders/working_encoding.wgsl`, shared with method A. The positions are integers and quarters, exact in
//! `f32`; floors, clamps and comparisons are exact.
//!
//! # Memory
//!
//! Six planes of the buffer's cells (one sixteenth of a plane each: the means, the weight's width and the
//! corrections), the input and the output (the core): the largest binding is a three-plane image
//! ([`max_binding_bytes`]).

use bytemuck::{Pod, Zeroable};

use super::bm_dct;
use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::RgbImage;
use crate::cpu::coarse_chroma::{CELL_PX, CoarseChroma};
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The dispatches [`encode_coarse_chroma`] records: the cells, the corrections and the apply pass.
pub const DISPATCHES: usize = 3;

/// The largest storage binding (bytes) the stage needs for a buffer of `rect` pixels: its three planes (the six
/// cell planes and the output are smaller). A test checks it against the 128 MiB design binding at
/// [`crate::chain::MAX_PHYS_AREA`] (`docs/17-gpu.md` K4).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    GpuImage::buffer_bytes(rect.w, rect.h, 3)
}

/// The uniform block of `coarse_chroma.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CcU {
    /// Buffer width, height, cell plane length (`cols * rows`), the barrier word (0).
    dims: [u32; 4],
    /// Input row stride, input plane length, output row stride, output plane length.
    io: [u32; 4],
    /// Core `x`, `y`, width and height in the buffer.
    core: [u32; 4],
    /// Frame `x` and `y` of the buffer's pixel `(0, 0)`, the buffer's cell columns and rows.
    grid: [u32; 4],
    /// Frame column and row of the buffer's first cell, the window radius in cells, 0.
    cell: [u32; 4],
    /// `a`, `b`, the encoding's gain and the strength.
    lvl: [f32; 4],
    /// The gains of R, G and B, 0.
    gains: [f32; 4],
}

/// The buffer's cells: frame column and row of the first, columns and rows (`cpu::coarse_chroma` `Grid::new`).
fn cells(input: &GpuImage) -> (usize, usize, usize, usize) {
    let (x0, y0) = (input.place.x0, input.place.y0);
    let (cx0, cy0) = (x0 / CELL_PX, y0 / CELL_PX);
    let cols = (x0 + input.width - 1) / CELL_PX - cx0 + 1;
    let rows = (y0 + input.height - 1) / CELL_PX - cy0 + 1;
    (cx0, cy0, cols, rows)
}

/// The uniform's `lvl` and `gains` of the stage, `None` when a number is not finite as `f32`.
fn noise_uniform(stage: &CoarseChroma) -> Option<([f32; 4], [f32; 4])> {
    let l = &stage.noise;
    let lvl = [
        l.a as f32,
        l.b as f32,
        l.encoding_gain as f32,
        stage.strength as f32,
    ];
    let gains = [l.gains[0] as f32, l.gains[1] as f32, l.gains[2] as f32, 0.0];
    lvl.iter()
        .chain(&gains)
        .all(|v| v.is_finite())
        .then_some((lvl, gains))
}

/// Records the coarse colour pass of the `core` rectangle of `input` (device in, device out) and returns the
/// core as a new three-plane image with the input's placement moved to the core: the device form of
/// [`CoarseChroma::apply`] followed by the crop to `core`.
///
/// `input` is the buffer the stage reads, placed in its frame (`input.place`): its edges cut the cells and the
/// window as on the CPU, so a core away from the frame's edge needs [`CoarseChroma::halo_px`] real pixels
/// around it.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not three planes or does not lie inside its frame, a core
/// that is empty or outside the input, a stage whose noise or strength is not finite as `f32`, and a buffer
/// too large for 32-bit indices; the pool's errors; shader errors.
pub fn encode_coarse_chroma(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    core: Rect,
    stage: &CoarseChroma,
) -> Result<GpuImage, GpuError> {
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "coarse colour: the input has {} planes, not 3",
            input.planes
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "coarse colour: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    let place = input.place;
    if place.x0 + input.width > place.frame_width || place.y0 + input.height > place.frame_height {
        return Err(GpuError::InvalidInput(format!(
            "coarse colour: the {} x {} input at {place:?} does not lie inside its frame",
            input.width, input.height
        )));
    }
    let Some((lvl, gains)) = noise_uniform(stage) else {
        return Err(GpuError::InvalidInput(format!(
            "coarse colour: the noise and the strength must be finite, got {stage:?}"
        )));
    };
    let n = input.width * input.height;
    if 3 * n as u64 > u64::from(u32::MAX) {
        return Err(GpuError::InvalidInput(format!(
            "coarse colour: {n} pixels do not fit a 32-bit index"
        )));
    }
    let (cx0, cy0, cols, rows) = cells(input);
    let cell_plane = cols * rows;
    let scratch = passes.scratch("coarse colour cells", 6 * cell_plane as u64 * 4)?;
    let out = GpuImage::alloc(
        passes.context(),
        "coarse colour rgb",
        core.w,
        core.h,
        3,
        place.offset(core.x, core.y),
    )?;
    let u = CcU {
        dims: [
            input.width as u32,
            input.height as u32,
            cell_plane as u32,
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
        grid: [place.x0 as u32, place.y0 as u32, cols as u32, rows as u32],
        cell: [cx0 as u32, cy0 as u32, stage.radius as u32, 0],
        lvl,
        gains,
    };
    let ub = passes.uniform("coarse colour uniform", &u)?;
    let over_cells = (
        (cols as u32).div_ceil(GROUP),
        (rows as u32).div_ceil(GROUP),
        1,
    );
    passes.dispatch(
        KernelId::CoarseChromaCells,
        "coarse colour cells",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, scratch.binding()),
        ],
        over_cells,
    )?;
    passes.dispatch(
        KernelId::CoarseChromaCorrect,
        "coarse colour corrections",
        &[(0, ub.binding()), (2, scratch.binding())],
        over_cells,
    )?;
    passes.dispatch(
        KernelId::CoarseChromaApply,
        "coarse colour apply",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, scratch.binding()),
            (3, out.buf.binding()),
        ],
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// The stage on the host: uploads `img` (placed in its frame by `img.place`), runs [`encode_coarse_chroma`] on
/// `core` and reads the core back.
///
/// # Errors
///
/// As [`encode_coarse_chroma`], plus the upload's and the readback's.
pub fn coarse_chroma(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
    stage: &CoarseChroma,
) -> Result<RgbImage, GpuError> {
    let input = GpuImage::upload_rgb(ctx, "coarse colour input", img)?;
    let mut passes = PassList::new(ctx, "coarse colour");
    let out = encode_coarse_chroma(&mut passes, &input, core, stage)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    drop(input);
    Ok(RgbImage {
        width: out.width,
        height: out.height,
        place: out.place,
        data,
    })
}

/// The constants of the stage, written from the CPU reference (`docs/17-gpu.md` K7): method A's, which the
/// working encoding and the chroma noise read, then the cell's.
fn constants() -> String {
    use std::fmt::Write as _;
    let mut s = bm_dct::constants();
    let log2e = std::f64::consts::LOG2_E;
    let hi = log2e as f32;
    let _ = writeln!(s, "const CC_CELL: u32 = {CELL_PX}u;");
    let _ = writeln!(
        s,
        "const CC_CENTRE: f32 = {:?};",
        (CELL_PX as f64 - 1.0) as f32 / 2.0
    );
    let _ = writeln!(s, "const CC_INV_CELL: f32 = {:?};", 1.0 / CELL_PX as f32);
    let _ = writeln!(s, "const CC_LOG2E_HI: f32 = {hi:?};");
    let _ = writeln!(
        s,
        "const CC_LOG2E_LO: f32 = {:?};",
        (log2e - f64::from(hi)) as f32
    );
    s
}

/// The WGSL of a coarse colour kernel: the constants, the shader file, the working encoding and the `main`.
pub(super) fn body(id: KernelId) -> String {
    let function = match id {
        KernelId::CoarseChromaCells => "cc_cells",
        KernelId::CoarseChromaCorrect => "cc_correct",
        KernelId::CoarseChromaApply => "cc_apply",
        _ => return String::new(),
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/coarse_chroma.wgsl"));
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

    const IDS: [KernelId; 3] = [
        KernelId::CoarseChromaCells,
        KernelId::CoarseChromaCorrect,
        KernelId::CoarseChromaApply,
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
        let table: [(KernelId, &[u32]); 3] = [
            (KernelId::CoarseChromaCells, &[0, 1, 2]),
            (KernelId::CoarseChromaCorrect, &[0, 2]),
            (KernelId::CoarseChromaApply, &[0, 1, 2, 3]),
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
    fn the_uniform_block_has_the_layout_of_cc_u() {
        for id in IDS {
            let (module, _) = module_of(id);
            let var = module
                .global_variables
                .iter()
                .map(|(_, v)| v)
                .find(|v| v.name.as_deref() == Some("cp"))
                .expect("the uniform `cp`");
            let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
                panic!("`cp` is not a struct");
            };
            let want = [
                ("dims", offset_of!(CcU, dims)),
                ("io", offset_of!(CcU, io)),
                ("core", offset_of!(CcU, core)),
                ("grid", offset_of!(CcU, grid)),
                ("cell", offset_of!(CcU, cell)),
                ("lvl", offset_of!(CcU, lvl)),
                ("gains", offset_of!(CcU, gains)),
            ];
            assert_eq!(members.len(), want.len(), "{id:?}");
            for (m, (name, offset)) in members.iter().zip(want) {
                assert_eq!(m.name.as_deref(), Some(name));
                assert_eq!(m.offset as usize, offset, "{id:?}: offset of {name}");
            }
            assert_eq!(*span as usize, size_of::<CcU>(), "{id:?}");
        }
    }

    #[test]
    fn the_constants_are_the_cpus() {
        let src = shader_source(StorageMode::F32, KernelId::CoarseChromaApply);
        let constant = |name: &str| -> String {
            let start = src
                .find(&format!("const {name}: "))
                .unwrap_or_else(|| panic!("const {name}"));
            let text = &src[start..];
            text[text.find("= ").expect("=") + 2..text.find(';').expect(";")].to_string()
        };
        assert_eq!(constant("CC_CELL"), format!("{CELL_PX}u"));
        assert_eq!(constant("CC_CENTRE").parse::<f32>().expect("f32"), 1.5);
        assert_eq!(constant("CC_INV_CELL").parse::<f32>().expect("f32"), 0.25);
        let hi = constant("CC_LOG2E_HI").parse::<f32>().expect("f32");
        let lo = constant("CC_LOG2E_LO").parse::<f32>().expect("f32");
        assert_eq!(hi, std::f64::consts::LOG2_E as f32);
        assert!((f64::from(hi) + f64::from(lo) - std::f64::consts::LOG2_E).abs() < 1e-15);
        // The working encoding's and the chroma noise's constants are method A's.
        assert!(src.contains("const BM_GAIN: f32 = "));
        assert!(src.contains("const BM_WCB2: vec3<f32> = "));
    }

    #[test]
    fn the_sums_and_quotients_are_hidden_from_the_optimiser() {
        let src = include_str!("../shaders/coarse_chroma.wgsl");
        for needle in [
            "return bitcast<f32>(bitcast<u32>(x) | cp.dims.w);",
            "return bm_exp2(p.x, bm_op(p.y + bm_op(x * CC_LOG2E_LO)));",
            "sb = bm_op(sb + ycc.y);",
            "let m = vec3<f32>(bm_div(sy, count), bm_div(sb, count), bm_div(sr, count));",
            "let h = bm_tau_of(v, cp.lvl.w);",
            "let inv = bm_div(1.0, bm_op(bm_op(2.0 * h) * h));",
            "let d2 = bm_op(bm_op(eb * eb) + bm_op(er * er));",
            "let wt = cc_exp(bm_op(-d2 * max(own_inv, j_inv)));",
            "sr = bm_op(sr + bm_op(wt * er));",
            "db = bm_div(sb, sw);",
            "let t = bm_op(bm_op(f32(p) - CC_CENTRE) * CC_INV_CELL);",
            "return bm_op(a + bm_op(bm_op(b - a) * t));",
            "let de_g = bm_div(-bm_op(bm_op(BM_KR * de_r) + bm_op(BM_KB * de_b)), BM_KG);",
            "cc_out[c * cp.io.w + o] = bm_decode(bm_op(e + de[c]));",
        ] {
            assert!(src.contains(needle), "coarse_chroma.wgsl lost `{needle}`");
        }
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect();
        // No fused multiply-add, no device exponential, power or root.
        for banned in ["fma(", " exp(", " exp2(", " pow(", " sqrt("] {
            assert!(!code.contains(banned), "coarse_chroma.wgsl uses `{banned}`");
        }
    }

    #[test]
    fn the_largest_binding_is_the_three_planes() {
        let r = Rect::new(0, 0, 1000, 700);
        assert_eq!(max_binding_bytes(r), GpuImage::buffer_bytes(1000, 700, 3));
        // Six cell planes are three eighths of a plane.
        assert!(6 * (250 * 175) * 4 < GpuImage::buffer_bytes(1000, 700, 1));
    }
}
