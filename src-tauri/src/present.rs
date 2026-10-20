//! The native viewport's two ways of drawing a frame into the window's surface (`crate::native`,
//! `docs/01` §Native viewport, `docs/17-gpu.md` section 16): the shaders and the pipelines, apart
//! from AppKit so that a test can draw with them on any device (`tests/present.rs` includes this
//! file by path and holds both to the frame's bytes).
//!
//! * **Host frames** are uploaded to an [`FRAME_FORMAT`] texture and read with `textureLoad`.
//! * **Device frames** (the core's RGBA8 buffer, kept on the device) are bound as a read-only storage
//!   buffer and read with `unpack4x8unorm`, one `u32` per pixel with R in the low byte.
//!
//! Both draw one full-screen triangle that writes the frame texel for texel into a
//! [`SURFACE_FORMAT`] target the size of the frame. Both read a code `c` as `c / 255` and the 8-bit
//! target stores it as `c` again, so the window shows the frame's codes whichever way it came.

/// The texture format of the surface: what the core's 8-bit frames are drawn into. A non-sRGB
/// format, so that the shader's output is written as it is (the frame is already encoded).
pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The texture a host frame is uploaded to: the bytes of the core's frame, not decoded.
pub const FRAME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One triangle that covers the whole target, shared by both pipelines.
const VERTEX: &str = r"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(corners[i], 0.0, 1.0);
}
";

/// Draws an uploaded host frame texel for texel (the frame and the target have the same size).
const HOST_FRAGMENT: &str = r"
@group(0) @binding(0) var frame: texture_2d<f32>;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(frame));
    let at = clamp(vec2<i32>(pos.xy), vec2<i32>(0, 0), size - vec2<i32>(1, 1));
    return textureLoad(frame, at, 0);
}
";

/// Draws a frame the core's device kept texel for texel, from the view job's RGBA8 buffer: one
/// `u32` per pixel, row-major, R in the low byte (`ym_core::pipeline::DeviceFrame`).
/// `unpack4x8unorm` gives `c / 255` per channel, as the host pipeline's texture read does.
const DEVICE_FRAGMENT: &str = r"
struct Size {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> frame: array<u32>;
@group(0) @binding(1) var<uniform> size: Size;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x), size.width - 1u);
    let y = min(u32(pos.y), size.height - 1u);
    return unpack4x8unorm(frame[y * size.width + x]);
}
";

/// The bytes of the device pipeline's `Size` uniform (16 bytes) for a frame of `width x height`.
#[must_use]
pub fn size_uniform(width: u32, height: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..4].copy_from_slice(&width.to_le_bytes());
    bytes[4..8].copy_from_slice(&height.to_le_bytes());
    bytes
}

/// The bind group layout of the host pipeline: the uploaded frame.
pub fn host_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frame"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    })
}

/// The bind group layout of the device pipeline: the frame's buffer and its size.
pub fn device_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("device frame"),
        entries: &[
            buffer(0, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer(1, wgpu::BufferBindingType::Uniform),
        ],
    })
}

/// The host pipeline (`layout` from [`host_layout`]).
pub fn host_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    pipeline(device, "frame", HOST_FRAGMENT, layout)
}

/// The device pipeline (`layout` from [`device_layout`]).
pub fn device_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    pipeline(device, "device frame", DEVICE_FRAGMENT, layout)
}

/// A full-screen-triangle pipeline with `fragment` and `layout`, drawing into [`SURFACE_FORMAT`].
fn pipeline(
    device: &wgpu::Device,
    label: &str,
    fragment: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{VERTEX}{fragment}").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SURFACE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Records the draw of the frame bound by `bind_group` with `pipeline` into `target` (the whole
/// target, cleared first).
pub fn encode_draw(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("frame"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}
