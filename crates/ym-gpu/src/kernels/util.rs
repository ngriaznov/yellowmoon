//! Upload, dispatch and readback helpers shared by the kernels.

use bytemuck::Pod;

use crate::device::{GpuContext, GpuError, StorageMode};
use crate::pool::PooledBuffer;

/// Threads per workgroup side; every kernel uses `@workgroup_size(16, 16)`.
pub(crate) const GROUP: u32 = 16;

/// Usage of a storage buffer the CPU fills and a kernel reads.
pub(crate) const STORAGE_IN: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
/// Usage of a storage buffer a kernel writes and the CPU reads back.
pub(crate) const STORAGE_OUT: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
/// Usage of a scratch storage buffer only kernels touch.
pub(crate) const STORAGE_SCRATCH: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
/// Usage of a uniform buffer.
pub(crate) const UNIFORM: wgpu::BufferUsages =
    wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);
/// Usage of a read-back staging buffer.
pub(crate) const STAGING: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

/// Rounds up to a multiple of two (row strides are even so f16 pairs share a word).
pub(crate) const fn even(n: usize) -> usize {
    n.div_ceil(2) * 2
}

/// Bytes of `samples` image samples in `mode`, rounded up to 4.
pub(crate) fn sample_bytes(mode: StorageMode, samples: usize) -> u64 {
    (samples as u64 * mode.bytes_per_sample()).div_ceil(4) * 4
}

/// Creates a pooled uniform buffer holding `value`.
pub(crate) fn uniform_buffer<T: Pod>(
    ctx: &GpuContext,
    label: &str,
    value: &T,
) -> Result<PooledBuffer, GpuError> {
    let bytes = bytemuck::bytes_of(value);
    let buf = ctx.pool().acquire(label, bytes.len() as u64, UNIFORM)?;
    ctx.queue.write_buffer(&buf, 0, bytes);
    ctx.counters_shared().count_upload(bytes.len() as u64);
    Ok(buf)
}

/// Creates a pooled storage buffer holding `data` (padded to 4 bytes with zeros).
pub(crate) fn storage_buffer(
    ctx: &GpuContext,
    label: &str,
    data: &[u8],
) -> Result<PooledBuffer, GpuError> {
    let len = (data.len() as u64).div_ceil(4).max(1) * 4;
    let buf = ctx.pool().acquire(label, len, STORAGE_IN)?;
    if data.len().is_multiple_of(4) {
        ctx.queue.write_buffer(&buf, 0, data);
    } else {
        let mut padded = data.to_vec();
        padded.resize(len as usize, 0);
        ctx.queue.write_buffer(&buf, 0, &padded);
    }
    ctx.counters_shared().count_upload(data.len() as u64);
    Ok(buf)
}

/// Converts f32 samples to the bytes of an image buffer in `mode`: f32 as is, f16 rounded to
/// nearest even (the conversion the GPU's own f16 stores use), padded to an even sample count.
pub(crate) fn image_bytes(mode: StorageMode, samples: &[f32]) -> Vec<u8> {
    match mode {
        StorageMode::F32 => bytemuck::cast_slice(samples).to_vec(),
        StorageMode::F16Native | StorageMode::F16Packed => {
            let mut out = Vec::with_capacity(samples.len() * 2 + 2);
            for &v in samples {
                let h = half::f16::from_f32(v.clamp(-65504.0, 65504.0));
                out.extend_from_slice(&h.to_bits().to_le_bytes());
            }
            if !samples.len().is_multiple_of(2) {
                out.extend_from_slice(&[0, 0]);
            }
            out
        }
    }
}

/// Decodes the bytes of an image buffer in `mode` back to f32.
pub(crate) fn image_samples(mode: StorageMode, bytes: &[u8]) -> Vec<f32> {
    match mode {
        StorageMode::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StorageMode::F16Native | StorageMode::F16Packed => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
    }
}

/// Number of workgroups to cover `pairs` thread columns by `rows` rows.
pub(crate) fn groups(stride: usize, rows: usize) -> (u32, u32) {
    let gx = (stride / 2).div_ceil(GROUP as usize) as u32;
    let gy = rows.div_ceil(GROUP as usize) as u32;
    (gx.max(1), gy.max(1))
}

/// A readback in flight: the copy has been submitted, the data is not yet mapped.
pub(crate) struct Readback {
    staging: PooledBuffer,
    bytes: u64,
    submission: wgpu::SubmissionIndex,
}

impl Readback {
    /// Records nothing: wraps a submission after `copy_buffer_to_buffer` into `staging`.
    pub(crate) fn new(
        staging: PooledBuffer,
        bytes: u64,
        submission: wgpu::SubmissionIndex,
    ) -> Self {
        Self {
            staging,
            bytes,
            submission,
        }
    }

    /// Waits for the submission (at most the readback timeout), maps the staging buffer and returns
    /// a copy of its first `bytes`.
    pub(crate) fn finish(self, ctx: &GpuContext) -> Result<Vec<u8>, GpuError> {
        let mut parts = crate::stream::map_readbacks(
            ctx,
            self.submission,
            &[(&self.staging, self.bytes)],
        )?;
        Ok(parts.pop().unwrap_or_default())
    }
}

/// Copies the first `bytes` of `src` into a fresh staging buffer, submits `encoder` (which already
/// holds the compute passes) and returns the pending readback.
pub(crate) fn submit_with_readback(
    ctx: &GpuContext,
    mut encoder: wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    bytes: u64,
) -> Result<Readback, GpuError> {
    let staging = ctx.pool().acquire("readback", bytes, STAGING)?;
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    let submission = ctx.submit([encoder.finish()])?;
    Ok(Readback::new(staging, bytes, submission))
}

/// How a kernel laid out its output in device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutKind {
    /// Three f32 planes of `stride * height` words.
    F32Planar,
    /// One RGBA8 word per pixel.
    Rgba8,
    /// Three u16 planes, two samples per word.
    U16Planar,
}

impl OutKind {
    /// The `kind` value written into the kernels' uniforms.
    pub(crate) fn code(self) -> u32 {
        match self {
            OutKind::F32Planar => 0,
            OutKind::Rgba8 => 1,
            OutKind::U16Planar => 2,
        }
    }

    /// Bytes of device output for a `stride x height` output (stride even).
    pub(crate) fn device_bytes(self, stride: usize, height: usize) -> u64 {
        let plane = (stride * height) as u64;
        match self {
            OutKind::F32Planar => plane * 12,
            OutKind::Rgba8 => plane * 4,
            OutKind::U16Planar => plane * 6,
        }
    }
}

/// Decoded pixels of a finished tile or image, dense (the row padding is removed).
#[derive(Clone, Debug, PartialEq)]
pub enum PixelData {
    /// Planar f32, `3 * width * height` samples.
    F32Planar(Vec<f32>),
    /// Interleaved RGBA8, `4 * width * height` bytes.
    Rgba8(Vec<u8>),
    /// Planar u16, `3 * width * height` samples.
    U16Planar(Vec<u16>),
}

/// Strips the row padding of a device output of `kind`.
pub(crate) fn unpack(
    kind: OutKind,
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> PixelData {
    let plane = stride * height;
    match kind {
        OutKind::F32Planar => {
            let all = image_samples(StorageMode::F32, bytes);
            let mut v = Vec::with_capacity(3 * width * height);
            for c in 0..3 {
                for y in 0..height {
                    let s = c * plane + y * stride;
                    v.extend_from_slice(&all[s..s + width]);
                }
            }
            PixelData::F32Planar(v)
        }
        OutKind::Rgba8 => {
            let mut v = Vec::with_capacity(4 * width * height);
            for y in 0..height {
                let s = y * stride * 4;
                v.extend_from_slice(&bytes[s..s + width * 4]);
            }
            PixelData::Rgba8(v)
        }
        OutKind::U16Planar => {
            let all: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let mut v = Vec::with_capacity(3 * width * height);
            for c in 0..3 {
                for y in 0..height {
                    let s = c * plane + y * stride;
                    v.extend_from_slice(&all[s..s + width]);
                }
            }
            PixelData::U16Planar(v)
        }
    }
}

/// Binds `entries` (binding index, resource) to the layout of `pipeline`'s group 0.
pub(crate) fn bind(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    entries: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}
