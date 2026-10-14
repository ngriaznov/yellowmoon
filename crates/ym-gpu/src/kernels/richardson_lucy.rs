//! R8 Richardson-Lucy capture sharpening on the GPU (G-06).
//!
//! The WGSL port (`shaders/richardson_lucy.wgsl`) of [`crate::cpu::sharpen::richardson_lucy_whole`],
//! specified in `docs/05` section R8: Richardson-Lucy deconvolution of the linear luma with a Gaussian
//! PSF of `sigma(r) = rho (1 + r^2 / 4)` (`r` the distance from the frame centre over the
//! half-diagonal), damped against the noise sigma, `n = round(A / 30)` iterations (0 to 10), the result
//! added to the three channels. [`RlPlan`] is the plan a photo's controls give (the kernel bank, the
//! radius, the iterations), [`encode_richardson_lucy`] the device-in, device-out entry point and
//! [`richardson_lucy`] the host wrapper of the parity tests.
//!
//! # What runs where
//!
//! The CPU makes, with the reference's own formulas, the 33-level kernel bank (normalised Gaussians
//! for `sigma` from `rho` to `1.25 rho`, truncated at `ceil(3 sigma_max)`, at most 10 px), the column
//! and row terms of the squared radius from the frame position (in `f64`, so the PSF of a pixel is a
//! function of its place in the frame and not of the rectangle: D4), and the uniform. The device runs
//! `2 + 4 n` dispatches: the level of every pixel, then per iteration the two halves of the forward
//! blur with the damped ratio and the two halves of the transposed blur with the update, then the
//! output. The planes are `f32`, the sums run in the CPU's order and are `f32` where the CPU's are.
//!
//! The halo is the plan's (`2 n ceil(3 sigma_max)`, capped at 24 px, [`RlPlan::halo_px`]): a tile
//! whose rectangle carries it equals the whole frame bit for bit while the exact radius is within the
//! cap (always for the Auto values), and by about 2e-7 beyond it, as the CPU's bands do.
//!
//! # Memory
//!
//! The working buffer holds 5 planes (`d`, `u`, the level, `q`, the blur's middle) of the input
//! rectangle: 20 bytes per pixel in one binding ([`max_binding_bytes`]), beside the input's three
//! planes and the core's three.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::sharpen::{DAMPING_SIGMAS, SharpenParams};
use crate::cpu::tiling::Rect;
use crate::cpu::{LUMA, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::bilateral::{SigmaInput, encode_sigma};
use crate::stream::PassList;

/// Tabulated kernels between `sigma_min` and `sigma_max` (the CPU's `KERNEL_LEVELS`).
const KERNEL_LEVELS: usize = 33;
/// The lowest value of an estimate or blurred estimate in the ratio (the CPU's `FLOOR`).
const FLOOR: f32 = 1e-4;
/// Planes of the working buffer.
const WORK_PLANES: u64 = 5;

/// The uniform block of `richardson_lucy.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RlU {
    /// Input width, height, row stride, plane length.
    dims: [u32; 4],
    /// Core `x`, `y`, width and height in the input.
    core: [u32; 4],
    /// Output row stride, output plane length, sigma row stride, working plane length.
    dst: [u32; 4],
    /// Kernel radius, taps, iterations, source plane of the row blur.
    cfg: [u32; 4],
    /// The input's origin in the sigma plane.
    sig: [u32; 4],
    /// `rho`, the lowest tabulated sigma, `1 / step`, the damping factor.
    k: [f32; 4],
    /// The luma weights and the floor of the ratio.
    luma: [f32; 4],
}

/// The plan of Richardson-Lucy for one set of controls: what the CPU's `richardson_lucy_whole` derives
/// from its `SharpenParams` before it touches a pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct RlPlan {
    iterations: usize,
    rho: f32,
    lo: f32,
    inv_step: f32,
    radius: usize,
    halo: usize,
    /// `KERNEL_LEVELS` rows of `2 radius + 1` taps.
    bank: Vec<f32>,
}

impl RlPlan {
    /// The plan of `params`, or `None` for zero iterations (the stage is then the identity and runs
    /// nothing).
    #[must_use]
    pub fn new(params: &SharpenParams) -> Option<RlPlan> {
        let iterations = params.iterations();
        if iterations == 0 {
            return None;
        }
        let rho = params.effective_radius_px();
        let hi = params.sigma_max_px();
        let radius = (3.0 * hi).ceil() as usize;
        // The CPU's `KernelBank::new(lo = rho, hi, radius)`, with its f32 expressions.
        let lo = rho;
        let width = 2 * radius + 1;
        let step = if hi > lo {
            (hi - lo) / (KERNEL_LEVELS - 1) as f32
        } else {
            1.0
        };
        let mut bank = vec![0.0f32; KERNEL_LEVELS * width];
        for l in 0..KERNEL_LEVELS {
            let sigma = lo + step * l as f32;
            let row = &mut bank[l * width..(l + 1) * width];
            for (k, t) in row.iter_mut().enumerate() {
                let d = k as f32 - radius as f32;
                *t = (-0.5 * d * d / (sigma * sigma)).exp();
            }
            let s: f32 = row.iter().sum();
            row.iter_mut().for_each(|t| *t /= s);
        }
        Some(RlPlan {
            iterations,
            rho,
            lo,
            inv_step: if hi > lo { 1.0 / step } else { 0.0 },
            radius,
            halo: params.halo_px(),
            bank,
        })
    }

    /// Iterations, 1 to 10.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The kernel radius in pixels, `ceil(3 sigma_max)`.
    #[must_use]
    pub fn radius_px(&self) -> usize {
        self.radius
    }

    /// The halo the scheduler gives the stage (`2 n ceil(3 sigma_max)`, at most 24 px).
    #[must_use]
    pub fn halo_px(&self) -> usize {
        self.halo
    }

    /// Dispatches the stage records: the level, `4 n` blurs, the output.
    #[must_use]
    pub fn dispatches(&self) -> usize {
        2 + 4 * self.iterations
    }
}

/// The largest storage binding (bytes) a Richardson-Lucy stage needs for an input rectangle of
/// `rect` pixels: the five-plane working buffer (the input's three planes and the core's three are
/// smaller). A test checks it against the 128 MiB design binding at
/// [`crate::chain::MAX_PHYS_AREA`] (`docs/gpu-plan.md` K4).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    WORK_PLANES * (rect.w * rect.h) as u64 * 4
}

/// The column terms and then the row terms of the squared radius of every pixel of an `input`
/// rectangle: `r^2 = ex[x] + ey[y]`, with `r` the distance of the pixel centre from the frame centre
/// over the half-diagonal, formed in `f64` from the frame position like the CPU's.
fn axis_terms(input: &GpuImage) -> Vec<f32> {
    let place = input.place;
    let (cx, cy) = place.centre();
    let inv_hd2 = 1.0 / (place.half_diagonal() * place.half_diagonal());
    let mut out = Vec::with_capacity(input.width + input.height);
    for x in 0..input.width {
        let dx = (place.x0 + x) as f64 + 0.5 - cx;
        out.push((dx * dx * inv_hd2) as f32);
    }
    for y in 0..input.height {
        let dy = (place.y0 + y) as f64 + 0.5 - cy;
        out.push((dy * dy * inv_hd2) as f32);
    }
    out
}

/// Records Richardson-Lucy for the `core` rectangle of `input` (device in, device out) and returns the
/// core as a new three-plane image whose placement is the input's moved to the core.
///
/// `input` is the compute rectangle (its edge mirrors, so a core away from the frame's edge needs
/// `plan.halo_px()` real pixels around it); its placement decides the PSF of each pixel. `sigma` is
/// the tile's sigma plane ([`encode_sigma`], the noise sigma without the network scale) and
/// `sigma_origin` the position of the input's pixel `(0, 0)` in it.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not three planes, a core that is empty or outside
/// the input, or a sigma plane that does not cover the input; the pool's errors; shader errors.
pub fn encode_richardson_lucy(
    passes: &mut PassList<'_>,
    plan: &RlPlan,
    input: &GpuImage,
    core: Rect,
    sigma: &GpuImage,
    sigma_origin: (usize, usize),
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if input.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "Richardson-Lucy: the input has {} planes, not 3",
            input.planes
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "Richardson-Lucy: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    if sigma.planes != 1
        || sigma_origin.0 + input.width > sigma.width
        || sigma_origin.1 + input.height > sigma.height
    {
        return Err(GpuError::InvalidInput(format!(
            "Richardson-Lucy: the {} x {} sigma plane at {sigma_origin:?} does not cover the {} x {} input",
            sigma.width, sigma.height, input.width, input.height
        )));
    }
    let plane = input.width * input.height;
    if WORK_PLANES * plane as u64 > u64::from(u32::MAX) {
        return Err(GpuError::InvalidInput(format!(
            "Richardson-Lucy: {plane} pixels do not fit a 32-bit index"
        )));
    }
    let out = GpuImage::alloc(
        ctx,
        "richardson-lucy rgb",
        core.w,
        core.h,
        3,
        input.place.offset(core.x, core.y),
    )?;
    let work = passes.scratch("richardson-lucy planes", WORK_PLANES * plane as u64 * 4)?;
    let bank = passes.keep(storage_buffer(
        ctx,
        "richardson-lucy bank",
        bytemuck::cast_slice(&plan.bank),
    )?);
    let axes = passes.keep(storage_buffer(
        ctx,
        "richardson-lucy axes",
        bytemuck::cast_slice(&axis_terms(input)),
    )?);
    let base = RlU {
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
            plane as u32,
        ],
        cfg: [
            plan.radius as u32,
            (2 * plan.radius + 1) as u32,
            plan.iterations as u32,
            0,
        ],
        sig: [sigma_origin.0 as u32, sigma_origin.1 as u32, 0, 0],
        k: [plan.rho, plan.lo, plan.inv_step, DAMPING_SIGMAS],
        luma: [LUMA[0], LUMA[1], LUMA[2], FLOOR],
    };
    let whole = (
        (input.width as u32).div_ceil(GROUP),
        (input.height as u32).div_ceil(GROUP),
        1,
    );
    let mut run = |id: KernelId, label: &str, source: u32, groups| -> Result<(), GpuError> {
        let mut u = base;
        u.cfg[3] = source;
        let ub = passes.uniform("richardson-lucy uniform", &u)?;
        passes.dispatch(
            id,
            label,
            &[
                (0, ub.binding()),
                (1, bank.binding()),
                (2, axes.binding()),
                (3, input.buf.binding()),
                (4, sigma.buf.binding()),
                (5, work.binding()),
                (6, out.buf.binding()),
            ]
            .into_iter()
            .filter(|(slot, _)| slots_of(id).contains(slot))
            .collect::<Vec<_>>(),
            groups,
        )
    };
    run(KernelId::RichardsonLucy, "rl level", 0, whole)?;
    for _ in 0..plan.iterations {
        run(KernelId::RlBlurRows, "rl rows u", 1, whole)?;
        run(KernelId::RlBlurColsRatio, "rl columns ratio", 0, whole)?;
        run(KernelId::RlBlurRows, "rl rows q", 3, whole)?;
        run(KernelId::RlBlurColsUpdate, "rl columns update", 0, whole)?;
    }
    run(
        KernelId::RlOutput,
        "rl output",
        0,
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// The bindings each pass of the shader touches (the auto-generated layout holds exactly these).
fn slots_of(id: KernelId) -> &'static [u32] {
    match id {
        KernelId::RichardsonLucy => &[0, 2, 3, 5],
        KernelId::RlBlurRows | KernelId::RlBlurColsUpdate => &[0, 1, 5],
        KernelId::RlBlurColsRatio => &[0, 1, 4, 5],
        KernelId::RlOutput => &[0, 3, 5, 6],
        _ => &[],
    }
}

/// Richardson-Lucy on the host: uploads `img`, expands `sigma` for its rectangle in the frame
/// (`img.place`), runs the stage on `core` and reads the core back. Zero iterations return the core of
/// the input, bit for bit. The GPU counterpart of
/// [`richardson_lucy_whole`](crate::cpu::sharpen::richardson_lucy_whole) when `core` is the whole
/// image (`sigma` is the noise sigma of the CPU function, indexed like the image).
///
/// # Errors
///
/// As [`encode_richardson_lucy`] and [`encode_sigma`], plus the readback's.
pub fn richardson_lucy(
    ctx: &GpuContext,
    img: &RgbImage,
    core: Rect,
    sigma: &SigmaInput<'_>,
    params: &SharpenParams,
) -> Result<RgbImage, GpuError> {
    let Some(plan) = RlPlan::new(params) else {
        // No iteration: the identity, nothing to run.
        if core.is_empty() || core.x1() > img.width || core.y1() > img.height {
            return Err(GpuError::InvalidInput(format!(
                "Richardson-Lucy: the core {core:?} is empty or outside the {} x {} input",
                img.width, img.height
            )));
        }
        return Ok(img.extract(core));
    };
    let input = GpuImage::upload_rgb(ctx, "richardson-lucy input", img)?;
    let mut passes = PassList::new(ctx, "richardson-lucy");
    let rect = Rect::new(img.place.x0, img.place.y0, img.width, img.height);
    let plane = encode_sigma(&mut passes, sigma, rect)?;
    let out = encode_richardson_lucy(&mut passes, &plan, &input, core, &plane, (0, 0))?;
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

/// The WGSL of a Richardson-Lucy pass: the shader file plus the `main` that runs the pass.
pub(super) fn body(id: KernelId) -> String {
    let function = match id {
        KernelId::RichardsonLucy => "rl_prepare",
        KernelId::RlBlurRows => "rl_blur_rows",
        KernelId::RlBlurColsRatio => "rl_blur_cols_ratio",
        KernelId::RlBlurColsUpdate => "rl_blur_cols_update",
        KernelId::RlOutput => "rl_output",
        _ => return String::new(),
    };
    let mut s = String::from(include_str!("../shaders/richardson_lucy.wgsl"));
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

    const PASSES: [KernelId; 5] = [
        KernelId::RichardsonLucy,
        KernelId::RlBlurRows,
        KernelId::RlBlurColsRatio,
        KernelId::RlBlurColsUpdate,
        KernelId::RlOutput,
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
        for id in PASSES {
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
                slots_of(id).iter().copied().collect::<BTreeSet<_>>(),
                "{id:?}"
            );
        }
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_rl_u() {
        let (module, _) = module_of(KernelId::RichardsonLucy);
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("rp"))
            .expect("the uniform `rp`");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`rp` is not a struct");
        };
        let want = [
            ("dims", offset_of!(RlU, dims)),
            ("core", offset_of!(RlU, core)),
            ("dst", offset_of!(RlU, dst)),
            ("cfg", offset_of!(RlU, cfg)),
            ("sig", offset_of!(RlU, sig)),
            ("k", offset_of!(RlU, k)),
            ("luma", offset_of!(RlU, luma)),
        ];
        assert_eq!(members.len(), want.len());
        for (m, (name, offset)) in members.iter().zip(want) {
            assert_eq!(m.name.as_deref(), Some(name));
            assert_eq!(m.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<RlU>());
    }

    #[test]
    fn the_plan_follows_the_cpu_rules() {
        let p = |amount, radius_px| SharpenParams { amount, radius_px };
        assert!(RlPlan::new(&p(0.0, 0.8)).is_none());
        assert!(RlPlan::new(&p(14.0, 0.8)).is_none());
        let auto = RlPlan::new(&p(120.0, 0.8)).expect("plan");
        assert_eq!(auto.iterations(), 4);
        // sigma_max 1.0, radius 3, halo 2 * 4 * 3 = 24 (the cap is reached exactly).
        assert_eq!((auto.radius_px(), auto.halo_px()), (3, 24));
        assert_eq!(auto.dispatches(), 18);
        let big = RlPlan::new(&p(300.0, 2.5)).expect("plan");
        assert_eq!((big.iterations(), big.radius_px(), big.halo_px()), (10, 10, 24));
        assert_eq!(big.bank.len(), KERNEL_LEVELS * 21);
        // Every row of the bank is a normalised kernel.
        for row in big.bank.chunks(21) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        // A radius outside the range is clamped like the CPU's.
        assert_eq!(RlPlan::new(&p(90.0, 100.0)).expect("plan").radius_px(), 8);
        assert_eq!(RlPlan::new(&p(90.0, f32::NAN)).expect("plan").radius_px(), 3);
    }
}
