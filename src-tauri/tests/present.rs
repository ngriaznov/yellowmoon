//! The native viewport's two ways of drawing a frame (`src/present.rs`, used by `src/native.rs` on
//! macOS; `docs/17-gpu.md` section 16): an uploaded host frame through `textureLoad`, and a frame
//! the core's device kept, read from its RGBA8 storage buffer with `unpack4x8unorm`. Both must put
//! the frame's 8-bit codes into the window's `Bgra8Unorm` surface unchanged, so the picture on screen
//! is the same whichever way a frame came (and the same as before the viewport shared the core's
//! device). Drawn here into an offscreen target of the surface's format and read back.
//!
//! Runs on any adapter (Mesa lavapipe on Linux, Metal in `mac-gpu.yml`'s app job). Skipped, loudly,
//! when no adapter opens, unless `YM_REQUIRE_GPU=1`.

#[path = "../src/present.rs"]
mod present;

use std::io::Write as _;
use std::sync::Arc;

use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TEST SKIPPED in present: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// A frame of `width x height` RGBA8 whose rows hold every code in every channel, alpha included,
/// in different orders (row-major, as the core writes it).
fn frame(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((4 * width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let v = (x + 37 * y) % 256;
            rgba.extend_from_slice(&[
                v as u8,
                (255 - v) as u8,
                ((v * 7 + y * 31) % 256) as u8,
                if y % 2 == 0 { 255 } else { (v ^ 0x5a) as u8 },
            ]);
        }
    }
    rgba
}

/// The target a pipeline draws into: the surface's format, the frame's size.
fn target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: present::SURFACE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Reads a `Bgra8Unorm` target back as RGBA8, row-major, without the row padding of the copy.
fn read_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let row = (4 * width).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map the readback"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the readback finishes");
    let mapped = slice.get_mapped_range().expect("the mapped readback");
    let mut rgba = Vec::with_capacity((4 * width * height) as usize);
    for y in 0..height {
        let start = (y * row) as usize;
        for px in mapped[start..start + (4 * width) as usize].chunks_exact(4) {
            // The surface is BGRA.
            rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
        }
    }
    rgba
}

/// The host path: the frame uploaded to a texture, drawn with the host pipeline.
fn draw_host(device: &wgpu::Device, queue: &wgpu::Queue, rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: present::FRAME_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        size,
    );
    let layout = present::host_layout(device);
    let pipeline = present::host_pipeline(device, &layout);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frame"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    draw(device, queue, &pipeline, &bind_group, w, h)
}

/// The zero-copy path: the frame in a storage buffer (as the view job leaves it), drawn with the
/// device pipeline.
fn draw_device(device: &wgpu::Device, queue: &wgpu::Queue, rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let frame = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("device frame"),
        size: rgba.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&frame, 0, rgba);
    let size = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("device frame size"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&size, 0, &present::size_uniform(w, h));
    let layout = present::device_layout(device);
    let pipeline = present::device_pipeline(device, &layout);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("device frame"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: size.as_entire_binding(),
            },
        ],
    });
    draw(device, queue, &pipeline, &bind_group, w, h)
}

fn draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let target = target(device, w, h);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    present::encode_draw(&mut encoder, &view, pipeline, bind_group);
    queue.submit([encoder.finish()]);
    read_rgba(device, queue, &target, w, h)
}

#[test]
fn both_ways_of_drawing_put_the_frame_codes_on_the_surface_unchanged() {
    let Some(gpu) = gpu() else { return };
    let setup = gpu.wgpu_setup();
    let (device, queue) = (&setup.device, &setup.queue);
    // Odd sizes: a row of the buffer is not a multiple of any alignment, and the texture upload's
    // rows are not either.
    for (w, h) in [(300, 5), (257, 3), (1, 1), (64, 2)] {
        let rgba = frame(w, h);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let host = draw_host(device, queue, &rgba, w, h);
        let zero_copy = draw_device(device, queue, &rgba, w, h);
        let error = pollster::block_on(scope.pop());
        assert!(error.is_none(), "{w} x {h}: {error:?}");
        let first = |a: &[u8]| a.iter().zip(&rgba).position(|(x, y)| x != y);
        assert!(
            host == rgba,
            "{w} x {h}: the host pipeline changed a code (first at byte {:?})",
            first(&host)
        );
        assert!(
            zero_copy == rgba,
            "{w} x {h}: the zero-copy pipeline changed a code (first at byte {:?})",
            first(&zero_copy)
        );
    }
    let _ = writeln!(
        std::io::stderr(),
        "present: both pipelines wrote every code unchanged on {}",
        gpu.info().name
    );
}
