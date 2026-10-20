//! Pipeline creation and dispatch for the WGSL kernels in `src/shaders/` (`docs/05-raw-processing.md`,
//! `docs/02-colour-pipeline.md`).
//!
//! * [`spike`]: the M0-07 chain, R1 normalise and R4 gains, bilinear demosaic, a fixed 3x3 matrix and
//!   the sRGB transfer, tiled with halos, GPU against the CPU reference within 1e-3.
//! * [`adaptive_ratio`]: the adaptive ratio demosaic (R6's classical method: green with the
//!   noise-regularised ratio, green refinement with the optional CA weighting, chroma, shadow
//!   sharpening and shadow colour matching), eleven pipelines of `adaptive_ratio.wgsl`, f32 planes in
//!   every storage mode, GPU against `cpu::demosaic::adaptive_ratio`.
//! * [`color`]: the whole per-pixel colour chain C1 to C13 as one compute pass (fused matrix, C7
//!   highlight range and toe, C8 LogC4, C9 grade, C10 CDL and tetrahedral LUT, C11 curves, C12 ACES 2
//!   or an ARRI display LUT, C13 encode and dither), mirroring `ym_color::display::ColorChain` stage
//!   by stage.
//! * [`diffraction`]: the R8 diffraction deconvolution (nine passes of `diffraction.wgsl`,
//!   f32 planes in every storage mode, the planner stays on the CPU), GPU against
//!   `cpu::sharpen::diffraction`.
//! * [`log_guided`]: the log-guided fast demosaic (all-float form, four dispatches), GPU
//!   against `cpu::demosaic::log_guided::demosaic_float`.
//! * [`output_sharpen`]: the output sharpening stage (luma unsharp mask with three threshold
//!   methods and the halo limiter), eight passes of `output_sharpen.wgsl`, f32 planes in every
//!   storage mode, GPU against `cpu::output_sharpen`.
//!
//! WGSL sources are generated at start-up: the storage mode of the device decides how f16 image data
//! is declared and accessed ([`StorageMode`]), and the constants (LogC, gamut matrices, grade
//! constants, table offsets) are written from `ym_color::consts` so no number is typed twice.
//! [`shader_source`] returns the final text for offline validation.
//!
//! # Kernels registered ahead of their implementation
//!
//! The device foundation registers a [`KernelId`] for every kernel family still to be written, each
//! with a stub WGSL module ([`KernelId::is_placeholder`]) and an empty Rust module below, so the
//! ids, the warm-up sets ([`KernelSet`]) and the offline checks (`tests/shaders_naga.rs`) exist
//! before the kernels do. The task that writes a kernel fills its module, replaces the stub in
//! `shader.rs` with its own `.wgsl` and removes the id from [`KernelId::placeholder_owner`].

pub mod adaptive_ratio;
pub mod bilateral;
pub mod bin;
pub mod color;
pub mod copy;
pub mod detail;
pub mod diffraction;
pub mod encode;
pub mod guided;
pub mod highlights;
pub mod lens;
pub mod log_guided;
pub mod magnify;
pub mod mosaic_ca;
pub mod output_sharpen;
pub mod prep;
pub mod raw;
pub mod reduce;
pub mod resize;
pub mod richardson_lucy;
pub mod sampler;
pub mod sigma;
pub mod spike;

mod aces;
mod shader;
pub(crate) mod util;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::device::{GpuContext, GpuError, StorageMode};

pub use shader::{KernelId, KernelSet, shader_source};
pub use util::PixelData;

/// Compiled compute pipelines, created on first use. Owned by the device thread.
pub struct Kernels {
    mode: StorageMode,
    pipelines: RefCell<HashMap<KernelId, wgpu::ComputePipeline>>,
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels")
            .field("mode", &self.mode)
            .field("compiled", &self.pipelines.borrow().len())
            .finish()
    }
}

impl Kernels {
    /// Kernels for a device using `mode`.
    pub(crate) fn new(mode: StorageMode) -> Self {
        Self {
            mode,
            pipelines: RefCell::new(HashMap::new()),
        }
    }

    /// The pipeline of `id`, compiling it the first time.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] with the validation message when the shader does not compile.
    pub fn pipeline(
        &self,
        ctx: &GpuContext,
        id: KernelId,
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        if let Some(p) = self.pipelines.borrow().get(&id) {
            return Ok(p.clone());
        }
        ctx.fault().on_pipeline(id.name())?;
        let source = shader_source(self.mode, id);
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(id.name()),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(id.name()),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: ctx.pipeline_cache(),
            });
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(GpuError::Shader(format!("{}: {err}", id.name())));
        }
        self.pipelines.borrow_mut().insert(id, pipeline.clone());
        Ok(pipeline)
    }
}
