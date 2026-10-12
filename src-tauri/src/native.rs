//! The native viewport for macOS (`docs/01` §Native viewport, M0-06, M3-04).
//!
//! A plain `NSView` is put **behind** the transparent webview, inside the window's content view,
//! and a wgpu surface (a `CAMetalLayer`, made by wgpu for that view) is drawn into it. The webview
//! keeps the photo cell transparent (`PhotoCell` with `native`), so the photo shows through where
//! the cell is; everything else of the interface is the webview's. The view follows the rectangle
//! the webview reports with `viewport_set_rect` (CSS pixels of the window, origin at the top left;
//! AppKit's origin is at the bottom left, which this module converts).
//!
//! The core renders each frame (`ym-core`, RGBA8 in Display P3, the size of the cell in device
//! pixels). This module only puts it on screen: the frame is uploaded to a texture and drawn with a
//! full-screen triangle into the surface, so the pixels go from the core's memory to the layer and
//! never cross the bridge.
//!
//! **Colour.** The surface is configured with `SurfaceColorSpace::DisplayP3` when the surface
//! offers it for `Bgra8Unorm`. wgpu then sets the layer's colour space to Display P3 and the
//! values are taken as sRGB-encoded P3, which is what the core writes; macOS converts to whatever
//! the display is (P3 or sRGB), also when the window moves between displays. Without that colour
//! space the sink refuses to start and the fallback viewport is used, so colour is never shown wrong
//! silently.
//!
//! **Threads.** AppKit objects are only touched on the main thread (`run_on_main_thread`), as
//! raw pointers that this module owns; wgpu objects are `Send + Sync` and are used from the
//! viewport worker.
//!
//! This code cannot be run on Linux. It type-checks against the macOS target (`cargo check
//! --target aarch64-apple-darwin -p yellowmoon-app`), and the checks that need a Mac are the
//! "Native viewport" list of `docs/11-ui.md` (HUMAN M0-H4).

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSView, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use raw_window_handle::{
    AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use tauri::{AppHandle, Manager, Runtime};
use ym_core::pipeline::RenderedFrame;

use crate::viewport::FrameSink;
use crate::wire::{ViewportBackendKind, ViewportRect};

/// How long creating the view on the main thread may take before we give up and use the fallback.
const MAIN_THREAD_TIMEOUT: Duration = Duration::from_secs(5);

/// The texture format of the surface: what the core's 8-bit frames are drawn into. A non-sRGB
/// format, so that the shader's output is written as it is (the frame is already encoded).
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The texture the frame is uploaded to: the bytes of the core's frame, not decoded.
const FRAME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Draws the frame texel for texel (the frame and the surface have the same size).
const SHADER: &str = r"
@group(0) @binding(0) var frame: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // One triangle that covers the whole target.
    let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(corners[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(frame));
    let at = clamp(vec2<i32>(pos.xy), vec2<i32>(0, 0), size - vec2<i32>(1, 1));
    return textureLoad(frame, at, 0);
}
";

/// A pointer to an AppKit object, to be used on the main thread only.
#[derive(Clone, Copy)]
struct MainThreadPtr(usize);

// SAFETY: the pointer is only turned back into a reference inside `run_on_main_thread` closures
// (see `with_views`), and the objects it points to outlive the sink: the content view belongs to
// the window, and our view is retained by it (and by the +1 `into_raw` below).
unsafe impl Send for MainThreadPtr {}
unsafe impl Sync for MainThreadPtr {}

/// Runs `f` on the main thread and waits for its result.
fn on_main<R: Runtime, T: Send + 'static>(
    app: &AppHandle<R>,
    f: impl FnOnce(MainThreadMarker) -> T + Send + 'static,
) -> Result<T, String> {
    if let Some(mtm) = MainThreadMarker::new() {
        return Ok(f(mtm));
    }
    let (tx, rx) = mpsc::channel();
    app.run_on_main_thread(move || {
        // `run_on_main_thread` only calls this closure on the main thread.
        if let Some(mtm) = MainThreadMarker::new() {
            let _ = tx.send(f(mtm));
        }
    })
    .map_err(|e| e.to_string())?;
    rx.recv_timeout(MAIN_THREAD_TIMEOUT)
        .map_err(|_| "the main thread did not answer".to_owned())
}

/// What drawing needs, behind one lock.
struct Draw {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The uploaded frame, with its size.
    frame: Option<(wgpu::Texture, u32, u32)>,
    /// The size the surface is configured for.
    configured: (u32, u32),
    // Last, so that the surface is dropped before the instance.
    _instance: wgpu::Instance,
}

/// How often a frame asks the surface for a texture before it gives up.
const ACQUIRE_ATTEMPTS: u32 = 3;

/// The native viewport (see the module documentation).
pub struct NativeSink<R: Runtime> {
    app: AppHandle<R>,
    content: MainThreadPtr,
    view: MainThreadPtr,
    draw: Mutex<Draw>,
    /// The view is hidden: nothing has been drawn yet, or [`FrameSink::clear`] took the picture
    /// away. The first frame after that shows it again (the webview reports the cell only when it
    /// moves or changes size, which a photo that follows a broken one does not).
    hidden: AtomicBool,
}

impl<R: Runtime> NativeSink<R> {
    /// Puts the view behind the webview of the window `label` and opens the surface.
    ///
    /// # Errors
    ///
    /// A plain reason when the window, the view, the surface or the Display P3 colour space are not
    /// available; the caller then uses the fallback viewport.
    pub fn create(app: &AppHandle<R>, label: &str) -> Result<Self, String> {
        let window = app
            .get_webview_window(label)
            .ok_or_else(|| format!("there is no window called {label}"))?;
        let content = window.ns_view().map_err(|e| e.to_string())? as usize;

        // The view and the surface are made on the main thread; the rest of wgpu's start-up is not
        // tied to it.
        let (instance, surface, view) = on_main(app, move |mtm| {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            // SAFETY: `content` is the window's content `NSView` (`WebviewWindow::ns_view`), valid
            // while the window is, and this runs on the main thread.
            let content: &NSView = unsafe { &*(content as *const NSView) };
            let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1.0, 1.0));
            let view = NSView::initWithFrame(mtm.alloc::<NSView>(), frame);
            view.setWantsLayer(true);
            // Nothing to show until the first frame has been drawn.
            view.setHidden(true);
            // Below every sibling: behind the webview.
            content.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, None);
            let handle = AppKitWindowHandle::new(NonNull::from(&*view).cast::<c_void>());
            // SAFETY: the handle is the view just made; the view outlives the surface because the
            // pointer is leaked below and the content view retains it too.
            let surface = unsafe {
                instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(RawDisplayHandle::AppKit(AppKitDisplayHandle::new())),
                    raw_window_handle: RawWindowHandle::AppKit(handle),
                })
            }
            .map_err(|e| e.to_string())?;
            let view = Retained::into_raw(view) as usize;
            Ok::<_, String>((instance, surface, view))
        })??;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..wgpu::RequestAdapterOptions::default()
        }))
        .map_err(|e| format!("no graphics adapter can draw to the window: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yellowmoon viewport"),
            ..wgpu::DeviceDescriptor::default()
        }))
        .map_err(|e| format!("the graphics device could not be opened: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        if !caps.formats.contains(&SURFACE_FORMAT) {
            return Err("the window's surface cannot take 8-bit BGRA".to_owned());
        }
        if !caps
            .color_spaces(SURFACE_FORMAT)
            .contains(wgpu::SurfaceColorSpaces::DISPLAY_P3)
        {
            return Err("the window's surface has no Display P3 colour space".to_owned());
        }
        let mut config = surface
            .get_default_config(&adapter, 1, 1)
            .ok_or_else(|| "the surface does not match the adapter".to_owned())?;
        config.format = SURFACE_FORMAT;
        config.color_space = wgpu::SurfaceColorSpace::DisplayP3;
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        config.present_mode = wgpu::PresentMode::Fifo;
        config.desired_maximum_frame_latency = 1;
        surface.configure(&device, &config);

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frame"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frame"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("frame"),
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
        });

        Ok(Self {
            app: app.clone(),
            content: MainThreadPtr(content),
            view: MainThreadPtr(view),
            draw: Mutex::new(Draw {
                surface,
                device,
                queue,
                config,
                pipeline,
                layout,
                frame: None,
                configured: (1, 1),
                _instance: instance,
            }),
            hidden: AtomicBool::new(true),
        })
    }

    /// Runs `f` with the content view and our view on the main thread.
    fn with_views(&self, f: impl FnOnce(&NSView, &NSView) + Send + 'static) {
        let (content, view) = (self.content, self.view);
        let result = self.app.run_on_main_thread(move || {
            // SAFETY: both pointers are live `NSView`s (see `MainThreadPtr`) and this is the main
            // thread.
            let (content, view) =
                unsafe { (&*(content.0 as *const NSView), &*(view.0 as *const NSView)) };
            f(content, view);
        });
        if let Err(e) = result {
            tracing::warn!(error = %e, "the viewport view was not updated");
        }
    }
}

/// The frame of the view for a rectangle of the window's web content, in the content view's own
/// coordinates (AppKit's origin is at the bottom left unless the view is flipped).
fn frame_for(content: &NSView, rect: &ViewportRect) -> NSRect {
    let height = content.bounds().size.height;
    let y = if content.isFlipped() {
        rect.y.0
    } else {
        height - rect.y.0 - rect.height.0
    };
    NSRect::new(
        NSPoint::new(rect.x.0, y),
        NSSize::new(rect.width.0.max(1.0), rect.height.0.max(1.0)),
    )
}

impl Draw {
    fn configure(&mut self, width: u32, height: u32) {
        if self.configured != (width, height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.configured = (width, height);
        }
    }

    /// Uploads the frame to a texture of its size (made again when the size changes).
    fn upload(&mut self, frame: &RenderedFrame, width: u32, height: u32) -> wgpu::TextureView {
        let fits = matches!(&self.frame, Some((_, w, h)) if (*w, *h) == (width, height));
        if !fits {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.frame = Some((texture, width, height));
        }
        let (texture, _, _) = self
            .frame
            .as_ref()
            .expect("the frame texture was just made");
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn draw(&mut self, frame: &RenderedFrame) -> Result<(), String> {
        let width = u32::try_from(frame.width).map_err(|_| "the frame is too wide")?;
        let height = u32::try_from(frame.height).map_err(|_| "the frame is too tall")?;
        if width == 0 || height == 0 || frame.rgba.len() != 4 * frame.width * frame.height {
            return Err("the frame has the wrong size".to_owned());
        }
        self.configure(width, height);
        let view = self.upload(frame, width, height);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        // Nothing asks for this frame again, so a texture that is not there yet is waited for
        // (a few milliseconds) and a surface that no longer matches the layer is configured
        // again, instead of dropping the frame and leaving the view blank.
        let mut attempt = 0;
        let target = loop {
            attempt += 1;
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => break t,
                // Minimised or behind another window: the layer keeps what it shows, and the
                // next frame will do.
                wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
                wgpu::CurrentSurfaceTexture::Outdated => {
                    self.configured = (0, 0);
                    self.configure(width, height);
                }
                wgpu::CurrentSurfaceTexture::Timeout => {
                    std::thread::sleep(Duration::from_millis(4));
                }
                wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Validation => {
                    return Err("the window's surface was lost".to_owned());
                }
            }
            if attempt >= ACQUIRE_ATTEMPTS {
                return Err("the window did not take the picture. Try again".to_owned());
            }
        };
        let target_view = target
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(target);
        Ok(())
    }
}

impl<R: Runtime> FrameSink for NativeSink<R> {
    fn kind(&self) -> ViewportBackendKind {
        ViewportBackendKind::Native
    }

    fn set_rect(&self, rect: &ViewportRect) {
        let rect = *rect;
        // Only the place: whether the view shows is `present` and `clear`'s business, so a cell
        // that is reported while there is no picture does not bring a stale one back.
        self.with_views(move |content, view| view.setFrame(frame_for(content, &rect)));
    }

    fn present(&self, frame: &RenderedFrame, _seq: u32) -> Result<(), String> {
        self.draw
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .draw(frame)?;
        // Drawn first, shown second: the view never appears with whatever it held before.
        if self.hidden.swap(false, Ordering::SeqCst) {
            self.with_views(|_content, view| view.setHidden(false));
        }
        Ok(())
    }

    fn clear(&self, _seq: u32) {
        if !self.hidden.swap(true, Ordering::SeqCst) {
            self.with_views(|_content, view| view.setHidden(true));
        }
    }
}
