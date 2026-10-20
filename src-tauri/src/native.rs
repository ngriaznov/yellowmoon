//! The native viewport for macOS (`docs/01` §Native viewport, M0-06, M3-04).
//!
//! A plain `NSView` is put **behind** the transparent webview, inside the window's content view,
//! and a wgpu surface (a `CAMetalLayer`, made by wgpu for that view) is drawn into it. The webview
//! keeps the photo cell transparent (`PhotoCell` with `native`), so the photo shows through where
//! the cell is; everything else of the interface is the webview's. The view follows the rectangle
//! the webview reports with `viewport_set_rect` (CSS pixels of the window, origin at the top left;
//! AppKit's origin is at the bottom left, which this module converts).
//!
//! The page lets the view show through: the cell is transparent (`data-native`), and while the
//! Photos screen shows, so are the page's root, its body and the screen (`data-native-viewport`
//! on the root element, `src/styles/screens.css`). Behind the page the window is paper
//! ([`PAPER_SRGB`], dark since the interface is), so the page looks the same wherever it draws nothing.
//!
//! **One device.** The core renders each frame (`ym-core`, RGBA8 in Display P3, the size of the
//! cell in device pixels). When the core has a graphics device ([`ym_gpu::device::Gpu`], opened
//! once per process by `ym_core::app::gpu::open_gpu`), the surface is made from that device's
//! instance and configured with that device, and the sink asks for frames that stay on it
//! ([`FrameSink::takes_device_frames`]): the view job's RGBA8 buffer ([`DeviceFrame`]) is bound
//! as a read-only storage buffer of a full-screen triangle that writes it texel for texel into the
//! surface's texture, on the core's queue, so the pass runs after the job that wrote the buffer
//! (submission order) and the frame never goes through the processor ("zero-copy"), at every zoom
//! (a view magnified above 100 % is magnified on the device since 2026-10-20). A frame the CPU made
//! (no usable device for a request, a device error, a stage that is off) arrives as host bytes and is
//! uploaded to a texture and drawn the same way, as before. Without a device in the core
//! (policy `off`, no adapter, a software adapter under `auto`, a device that is lost) the sink
//! opens a device of its own and draws host frames only, as it always did; a core device that is
//! lost later is replaced by one of the sink's own at the next frame. The log says which at start:
//! "native viewport: zero-copy on the core's device" or "native viewport: own device, host frames".
//! The viewport's own work on the core's device runs inside error scopes, so a mistake of the
//! viewport is its own error and never taints the device the renderer uses.
//!
//! **Colour.** The surface is configured with `SurfaceColorSpace::DisplayP3` when the surface
//! offers it for `Bgra8Unorm`. wgpu then sets the layer's colour space to Display P3 and the
//! values are taken as sRGB-encoded P3, which is what the core writes; macOS converts to whatever
//! the display is (P3 or sRGB), also when the window moves between displays. Without that colour
//! space the sink refuses to start and the fallback viewport is used, so colour is never shown wrong
//! silently. Both pipelines write the frame's 8-bit codes unchanged (a code `c` is read as
//! `c / 255` and written back to the 8-bit target as `c`), so the screen shows the same pixels
//! whichever way the frame came.
//!
//! **Threads.** AppKit objects are only touched on the main thread (`run_on_main_thread`), as
//! raw pointers that this module owns; wgpu objects are `Send + Sync` and are used from the
//! viewport worker. The frame being drawn is borrowed for the whole of [`FrameSink::present`], so
//! the buffer it holds is not given back to the device's pool before the pass that reads it is
//! submitted; a job that gets the buffer later is submitted after it, and the queue keeps that order.
//!
//! This code cannot be run on Linux. It type-checks against the macOS target (`cargo check
//! --target aarch64-apple-darwin -p yellowmoon-app`; from Linux the build scripts need a stand-in
//! C compiler, `docs/13-testing.md` §Checking the macOS code on Linux), and the checks that need a
//! Mac are the "Native viewport" list of `docs/11-ui.md` (HUMAN M0-H4) and the `viewport` job of
//! `.github/workflows/build.yml`.
//!
//! **Timings.** [`FrameSink::present`] times the upload of a host frame, the wait for a drawable and
//! the pass (recording, submitting and presenting), and counts the frames drawn each way (`perf`,
//! with `YM_PERF=1` and in development builds): a wait near one refresh says the worker sits at the
//! display after every render.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSColor, NSView, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use raw_window_handle::{
    AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use tauri::{AppHandle, Manager, Runtime};
use ym_core::pipeline::{DeviceFrame, RenderedFrame};
use ym_gpu::device::Gpu;

use crate::perf::{self, Drawn};
use crate::present::{self, FRAME_FORMAT, SURFACE_FORMAT};
use crate::viewport::FrameSink;
use crate::wire::{ViewportBackendKind, ViewportRect};

/// The window's background behind the webview, in sRGB: `--paper` of the dark theme in
/// `src/styles/tokens.css` (#0f0f10, the interface's ground since the dark glass interface of
/// 2026-10-16). The page is transparent over the photo cell while the Photos screen shows, so this
/// is what the cell shows before the first frame and after [`FrameSink::clear`], and what the page
/// shows wherever it draws nothing. Tauri's transparent window starts with a clear background,
/// which would show the desktop there instead; `backgroundColor` of the window in
/// `tauri.conf.json` is the same colour, so that the window is not clear before this runs either.
/// The same colour as 8-bit codes, [`crate::api::PAPER_RGB8`], is the letterbox of the core's
/// frames; `src/styles/native-viewport.node.test.ts` holds all of them to `--paper`.
const PAPER_SRGB: [f64; 3] = [
    0.058823529411764705,
    0.058823529411764705,
    0.06274509803921569,
];

/// How long creating the view on the main thread may take before we give up and use the fallback.
const MAIN_THREAD_TIMEOUT: Duration = Duration::from_secs(5);

/// What [`FrameSink::present`] says when the sink has no surface any more (it was lost with the
/// core's device and none of its own could be made).
const NO_SURFACE: &str =
    "The photo can't be shown: the window's drawing surface is gone. Restart Yellowmoon.";

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

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs `f` with the validation, internal and out-of-memory errors of `device` captured on this
/// thread, and turns one into an error of `f`. On the core's device an error nobody captures
/// taints the device, which ends the renderer's use of it for the process (`ym_gpu::device`); a
/// mistake of the viewport must stay the viewport's.
fn captured<T>(device: &wgpu::Device, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = f();
    // The innermost scope first.
    let invalid = pollster::block_on(validation.pop());
    let broken = pollster::block_on(internal.pop());
    let out_of_memory = pollster::block_on(memory.pop());
    let value = result?;
    match invalid.or(broken).or(out_of_memory) {
        Some(e) => Err(format!(
            "the graphics device refused the viewport's work: {e}"
        )),
        None => Ok(value),
    }
}

/// Makes the view, puts it behind every sibling in the window's content view (behind the webview)
/// and paints the window paper. Main thread. Returns the view, retained once more by us.
fn add_view(mtm: MainThreadMarker, content: MainThreadPtr) -> MainThreadPtr {
    // SAFETY: `content` is the window's content `NSView` (`WebviewWindow::ns_view`), valid
    // while the window is, and this runs on the main thread.
    let content: &NSView = unsafe { &*(content.0 as *const NSView) };
    if let Some(window) = content.window() {
        let [r, g, b] = PAPER_SRGB;
        // The smoke run's colour test (`YM_SMOKE_WINDOW=red`): a red photo cell means the page
        // lets the window through but the view is not there.
        #[cfg(debug_assertions)]
        let [r, g, b] = if std::env::var("YM_SMOKE_WINDOW").is_ok_and(|v| v == "red") {
            [1.0, 0.0, 0.0]
        } else {
            [r, g, b]
        };
        window.setBackgroundColor(Some(&NSColor::colorWithSRGBRed_green_blue_alpha(
            r, g, b, 1.0,
        )));
        window.setOpaque(true);
    }
    let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1.0, 1.0));
    let view = NSView::initWithFrame(mtm.alloc::<NSView>(), frame);
    view.setWantsLayer(true);
    // Nothing to show until the first frame has been drawn.
    view.setHidden(true);
    // Below every sibling: behind the webview.
    content.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, None);
    // Leaked on purpose: the view lives as long as the window (the content view retains it too).
    MainThreadPtr(Retained::into_raw(view) as usize)
}

/// Makes the surface of `view` from `instance`, on the main thread (AppKit's). Whatever drew into
/// the view before (`old`, a [`Draw`] that is being replaced) is dropped there first and the layers
/// its surface added are taken off the view, so the new surface's layer is the view's only one.
fn surface_for<R: Runtime>(
    app: &AppHandle<R>,
    view: MainThreadPtr,
    instance: wgpu::Instance,
    old: Option<Draw>,
) -> Result<(wgpu::Instance, wgpu::Surface<'static>), String> {
    on_main(app, move |_mtm| {
        // SAFETY: `view` is our live `NSView` (`MainThreadPtr`) and this is the main thread.
        let view: &NSView = unsafe { &*(view.0 as *const NSView) };
        drop(old);
        remove_layers(view);
        let handle = AppKitWindowHandle::new(NonNull::from(view).cast::<c_void>());
        // SAFETY: the handle is our view; the view outlives the surface because it is never
        // released (`add_view`) and the content view retains it too.
        let surface = unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(RawDisplayHandle::AppKit(AppKitDisplayHandle::new())),
                raw_window_handle: RawWindowHandle::AppKit(handle),
            })
        }
        .map_err(|e| e.to_string())?;
        Ok::<_, String>((instance, surface))
    })?
}

/// Takes every sublayer off the view's layer: the `CAMetalLayer` a surface added (wgpu adds one per
/// surface and the view's layer keeps it after the surface is gone). Main thread.
fn remove_layers(view: &NSView) {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    // SAFETY: `layer` and `setSublayers:` exist on `NSView` and `CALayer`; nil is a valid
    // argument of `setSublayers:` (no sublayers); this runs on the main thread.
    unsafe {
        let layer: Option<Retained<AnyObject>> = msg_send![view, layer];
        if let Some(layer) = layer {
            let _: () = msg_send![&*layer, setSublayers: None::<&AnyObject>];
        }
    }
}

/// The pipeline that draws an uploaded host frame.
struct HostPipe {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The uploaded frame, with its size.
    frame: Option<(wgpu::Texture, u32, u32)>,
}

/// The pipeline that draws a frame straight from the buffer the core's device wrote it into.
struct DevicePipe {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The frame's size for the shader (`Size`, 16 bytes), and the size it holds now.
    size: wgpu::Buffer,
    written: (u32, u32),
    /// The core's device: a frame is drawn from its buffer only when it lives there.
    gpu: Weak<Gpu>,
}

impl HostPipe {
    fn new(device: &wgpu::Device) -> Self {
        let layout = present::host_layout(device);
        Self {
            pipeline: present::host_pipeline(device, &layout),
            layout,
            frame: None,
        }
    }

    /// Uploads the frame to a texture of its size (made again when the size changes).
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = match self.frame.take() {
            Some((texture, w, h)) if (w, h) == (width, height) => texture,
            _ => device.create_texture(&wgpu::TextureDescriptor {
                label: Some("frame"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }),
        };
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
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.frame = Some((texture, width, height));
        view
    }
}

impl DevicePipe {
    fn new(device: &wgpu::Device, gpu: Weak<Gpu>) -> Self {
        let layout = present::device_layout(device);
        let size = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("device frame size"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline: present::device_pipeline(device, &layout),
            layout,
            size,
            written: (0, 0),
            gpu,
        }
    }

    /// The bind group of `frame`, with its size written for the shader when it changed.
    fn bind(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &DeviceFrame,
        width: u32,
        height: u32,
    ) -> wgpu::BindGroup {
        if self.written != (width, height) {
            queue.write_buffer(&self.size, 0, &present::size_uniform(width, height));
            self.written = (width, height);
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("device frame"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.buffer().binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.size.as_entire_binding(),
                },
            ],
        })
    }
}

/// Where the pixels of a frame are, for [`Draw::draw`].
enum Pixels<'a> {
    /// Host bytes, to upload.
    Host(&'a [u8]),
    /// The core's buffer, to draw from.
    Device(&'a DeviceFrame),
}

/// What drawing needs, behind one lock.
struct Draw {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    host: HostPipe,
    /// The pipeline of frames that stay on the core's device; `None` on a device of the sink's own.
    zero_copy: Option<DevicePipe>,
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
    /// `None` once the surface is gone for good (see [`NO_SURFACE`]).
    draw: Mutex<Option<Draw>>,
    /// The view is hidden: nothing has been drawn yet, or [`FrameSink::clear`] took the picture
    /// away. The first frame after that shows it again (the webview reports the cell only when it
    /// moves or changes size, which a photo that follows a broken one does not).
    hidden: AtomicBool,
}

impl<R: Runtime> NativeSink<R> {
    /// Puts the view behind the webview of the window `label` and opens the surface: on `gpu`, the
    /// core's device, when it is given and usable (and can draw to the window), else on a device of
    /// the sink's own.
    ///
    /// # Errors
    ///
    /// A plain reason when the window, the view, the surface or the Display P3 colour space are not
    /// available; the caller then uses the fallback viewport.
    pub fn create(app: &AppHandle<R>, label: &str, gpu: Option<&Arc<Gpu>>) -> Result<Self, String> {
        let window = app
            .get_webview_window(label)
            .ok_or_else(|| format!("there is no window called {label}"))?;
        let content = MainThreadPtr(window.ns_view().map_err(|e| e.to_string())? as usize);
        let view = on_main(app, move |mtm| add_view(mtm, content))?;

        // A device the renderer no longer trusts is not shared either.
        let shared = gpu.filter(|g| !g.is_lost());
        let draw = match shared.map(|g| Draw::shared(app, view, g)) {
            Some(Ok(draw)) => draw,
            Some(Err(reason)) => {
                tracing::warn!(
                    %reason,
                    "native viewport: the core's graphics device cannot draw to the window"
                );
                Draw::own(app, view, None)?
            }
            None => Draw::own(app, view, None)?,
        };
        if draw.zero_copy.is_some() {
            tracing::info!("native viewport: zero-copy on the core's device");
        } else {
            tracing::info!("native viewport: own device, host frames");
        }
        Ok(Self {
            app: app.clone(),
            content,
            view,
            draw: Mutex::new(Some(draw)),
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
    /// The surface on the core's device: made from the device's instance, configured with the
    /// device, drawing frames that stay on it as well as host frames.
    fn shared<R: Runtime>(
        app: &AppHandle<R>,
        view: MainThreadPtr,
        gpu: &Arc<Gpu>,
    ) -> Result<Self, String> {
        let setup = gpu.wgpu_setup();
        let (instance, surface) = surface_for(app, view, setup.instance, None)?;
        if !setup.adapter.is_surface_supported(&surface) {
            return Err("the core's graphics adapter cannot draw to the window".to_owned());
        }
        Self::new(
            surface,
            instance,
            &setup.adapter,
            setup.device,
            setup.queue,
            Some(Arc::downgrade(gpu)),
        )
    }

    /// The surface on a device of the sink's own (Metal, high performance), drawing host frames:
    /// the viewport as it was before it shared the core's device. `old` is what drew into the view
    /// until now, replaced.
    fn own<R: Runtime>(
        app: &AppHandle<R>,
        view: MainThreadPtr,
        old: Option<Draw>,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let (instance, surface) = surface_for(app, view, instance, old)?;
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
        Self::new(surface, instance, &adapter, device, queue, None)
    }

    /// Checks the surface's format and colour space, configures it and makes the pipelines (the
    /// zero-copy one when `gpu`, the core's device, is given).
    fn new(
        surface: wgpu::Surface<'static>,
        instance: wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        gpu: Option<Weak<Gpu>>,
    ) -> Result<Self, String> {
        let caps = surface.get_capabilities(adapter);
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
            .get_default_config(adapter, 1, 1)
            .ok_or_else(|| "the surface does not match the adapter".to_owned())?;
        config.format = SURFACE_FORMAT;
        config.color_space = wgpu::SurfaceColorSpace::DisplayP3;
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        config.present_mode = wgpu::PresentMode::Fifo;
        config.desired_maximum_frame_latency = 1;
        let (host, zero_copy) = captured(&device, || {
            surface.configure(&device, &config);
            Ok((
                HostPipe::new(&device),
                gpu.map(|gpu| DevicePipe::new(&device, gpu)),
            ))
        })?;
        Ok(Self {
            surface,
            device,
            queue,
            config,
            host,
            zero_copy,
            configured: (1, 1),
            _instance: instance,
        })
    }

    /// Whether this draws on the core's device and that device is lost: the sink then needs a
    /// device of its own. (A device that is only tainted no longer computes for the renderer, but it
    /// still draws.)
    fn device_lost(&self) -> bool {
        self.zero_copy
            .as_ref()
            .and_then(|p| p.gpu.upgrade())
            .is_some_and(|g| g.health().lost())
    }

    /// Whether frames that stay on the core's device can be drawn: the sink shares the device, and
    /// the renderer still uses it.
    fn draws_device_frames(&self) -> bool {
        self.zero_copy
            .as_ref()
            .and_then(|p| p.gpu.upgrade())
            .is_some_and(|g| !g.health().is_unusable())
    }

    fn configure(&mut self, width: u32, height: u32) {
        if self.configured != (width, height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.configured = (width, height);
        }
    }

    /// Draws `frame`. `None` when the window is hidden and nothing was drawn.
    fn draw(&mut self, frame: &RenderedFrame) -> Result<Option<Drawn>, String> {
        let width = u32::try_from(frame.width).map_err(|_| "the frame is too wide")?;
        let height = u32::try_from(frame.height).map_err(|_| "the frame is too tall")?;
        let n = 4 * frame.width * frame.height;
        if width == 0 || height == 0 {
            return Err("the frame has the wrong size".to_owned());
        }
        let readback: Vec<u8>;
        let pixels = match &frame.on_device {
            Some(on_device) => {
                if (on_device.width(), on_device.height()) != (frame.width, frame.height) {
                    return Err("the frame has the wrong size".to_owned());
                }
                match &self.zero_copy {
                    Some(pipe) if on_device.is_on(&pipe.gpu) => Pixels::Device(on_device),
                    // A frame on a device this sink does not draw with (it stopped sharing the
                    // core's): read back, then drawn as a host frame.
                    _ => {
                        readback = on_device.read().map_err(|e| {
                            format!(
                                "the graphics card did not hand over the frame ({e}). Move the \
                                 photo to draw it again"
                            )
                        })?;
                        Pixels::Host(&readback)
                    }
                }
            }
            None => Pixels::Host(&frame.rgba),
        };
        if matches!(pixels, Pixels::Host(rgba) if rgba.len() != n) {
            return Err("the frame has the wrong size".to_owned());
        }
        let device = self.device.clone();
        captured(&device, || self.record(&pixels, width, height))
    }

    /// The work of [`Draw::draw`] on the device: the frame's bind group (an upload first for a host
    /// frame), a drawable, the pass, the present.
    fn record(
        &mut self,
        pixels: &Pixels<'_>,
        width: u32,
        height: u32,
    ) -> Result<Option<Drawn>, String> {
        self.configure(width, height);
        let (bind_group, drawn) = match pixels {
            Pixels::Host(rgba) => {
                // Timed for `perf` (with `YM_PERF`, and in development builds): the upload is the
                // copy of the frame into the texture, and the wait for a drawable below is the wait
                // for the display.
                let upload_timer = perf::timer(perf::Timed::Upload);
                let view = self
                    .host
                    .upload(&self.device, &self.queue, rgba, width, height);
                drop(upload_timer);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("frame"),
                    layout: &self.host.layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                });
                (bind_group, Drawn::Uploaded)
            }
            Pixels::Device(frame) => {
                let Some(pipe) = self.zero_copy.as_mut() else {
                    return Err(
                        "the frame is on a graphics device the viewport does not use".to_owned(),
                    );
                };
                // No upload: the buffer the view job wrote is the pass's source.
                let bind_group = pipe.bind(&self.device, &self.queue, frame, width, height);
                (bind_group, Drawn::ZeroCopy)
            }
        };
        // Nothing asks for this frame again, so a texture that is not there yet is waited for
        // (a few milliseconds) and a surface that no longer matches the layer is configured
        // again, instead of dropping the frame and leaving the view blank.
        let acquire_timer = perf::timer(perf::Timed::Acquire);
        let mut attempt = 0;
        let target = loop {
            attempt += 1;
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => break t,
                // Minimised or behind another window: the layer keeps what it shows, and the
                // next frame will do.
                wgpu::CurrentSurfaceTexture::Occluded => return Ok(None),
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
        drop(acquire_timer);
        let pass_timer = perf::timer(perf::Timed::Pass);
        let pipeline = match drawn {
            Drawn::ZeroCopy => match &self.zero_copy {
                Some(pipe) => &pipe.pipeline,
                None => &self.host.pipeline,
            },
            Drawn::Uploaded => &self.host.pipeline,
        };
        let target_view = target
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        present::encode_draw(&mut encoder, &target_view, pipeline, &bind_group);
        // On the core's device this is the core's queue: the pass runs after the view job that
        // wrote the buffer, which was submitted before the frame was handed over.
        self.queue.submit([encoder.finish()]);
        self.queue.present(target);
        drop(pass_timer);
        Ok(Some(drawn))
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
        let drawn = {
            let mut slot = lock(&self.draw);
            // The core's device was lost: from now on the sink draws with a device of its own, and
            // the worker asks for host frames (`takes_device_frames`).
            if slot.as_ref().is_some_and(Draw::device_lost) {
                tracing::warn!(
                    "native viewport: the core's graphics device was lost; own device, host frames"
                );
                let old = slot.take();
                *slot = Some(Draw::own(&self.app, self.view, old)?);
            }
            slot.as_mut()
                .ok_or_else(|| NO_SURFACE.to_owned())?
                .draw(frame)?
        };
        if let Some(how) = drawn {
            perf::viewport().drawn(how);
        }
        // Drawn first, shown second: the view never appears with whatever it held before.
        if self.hidden.swap(false, Ordering::SeqCst) {
            tracing::info!(
                width = frame.width,
                height = frame.height,
                zero_copy = drawn == Some(Drawn::ZeroCopy),
                "the native viewport shows a picture"
            );
            self.with_views(|content, view| {
                view.setHidden(false);
                #[cfg(debug_assertions)]
                smoke::describe(content, view);
                #[cfg(not(debug_assertions))]
                let _ = content;
            });
        }
        Ok(())
    }

    fn clear(&self, _seq: u32) {
        if !self.hidden.swap(true, Ordering::SeqCst) {
            self.with_views(|_content, view| view.setHidden(true));
        }
    }

    fn takes_device_frames(&self) -> bool {
        lock(&self.draw)
            .as_ref()
            .is_some_and(Draw::draws_device_frames)
    }
}

/// What the smoke run (`YM_SMOKE`, development builds) logs about the view once it shows a picture:
/// where it is, whether it is in the window's content view, its layers, and every sibling view.
#[cfg(debug_assertions)]
mod smoke {
    use objc2::msg_send;
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_app_kit::NSView;
    use objc2_foundation::NSRect;

    fn class_of(o: &AnyObject) -> String {
        o.class().name().to_string_lossy().into_owned()
    }

    fn rect(r: NSRect) -> String {
        format!(
            "({:.0},{:.0} {:.0}x{:.0})",
            r.origin.x, r.origin.y, r.size.width, r.size.height
        )
    }

    /// Runs on the main thread.
    pub(super) fn describe(content: &NSView, view: &NSView) {
        if std::env::var_os("YM_SMOKE").is_none() {
            return;
        }
        let window = view.window();
        let content_of_window = window.as_ref().and_then(|w| w.contentView());
        // SAFETY: `superview` only reads the view's parent; this runs on the main thread.
        let in_content = unsafe { view.superview() }.is_some_and(|s| std::ptr::eq(&*s, content));
        let content_is_windows = content_of_window
            .as_ref()
            .is_some_and(|c| std::ptr::eq(&**c, content));
        eprintln!(
            "smoke: view frame {} hidden {} in window {} superview is content {} content is the window's {} content {} {}",
            rect(view.frame()),
            view.isHidden(),
            window.is_some(),
            in_content,
            content_is_windows,
            class_of(content),
            rect(content.frame()),
        );
        if let Some(w) = &window {
            eprintln!(
                "smoke: window opaque {} frame {}",
                w.isOpaque(),
                rect(w.frame())
            );
        }
        // SAFETY: plain property reads of AppKit and Core Animation objects on the main thread;
        // every message below exists on the receiver's class (`NSView`, `NSArray`, `CALayer`).
        unsafe {
            let layer: Option<Retained<AnyObject>> = msg_send![view, layer];
            match layer {
                Some(layer) => {
                    let subs: Option<Retained<AnyObject>> = msg_send![&*layer, sublayers];
                    let n: usize = subs.as_ref().map_or(0, |s| msg_send![&**s, count]);
                    let frame: NSRect = msg_send![&*layer, frame];
                    eprintln!(
                        "smoke: view layer {} frame {} sublayers {n}",
                        class_of(&layer),
                        rect(frame)
                    );
                    if let Some(subs) = subs {
                        for i in 0..n {
                            let o: Retained<AnyObject> = msg_send![&*subs, objectAtIndex: i];
                            let f: NSRect = msg_send![&*o, frame];
                            let hidden: bool = msg_send![&*o, isHidden];
                            let opacity: f32 = msg_send![&*o, opacity];
                            eprintln!(
                                "smoke:   sublayer {i} {} frame {} hidden {hidden} opacity {opacity}",
                                class_of(&o),
                                rect(f)
                            );
                        }
                    }
                }
                None => eprintln!("smoke: view has no layer"),
            }
            let subviews: Retained<AnyObject> = msg_send![content, subviews];
            let n: usize = msg_send![&*subviews, count];
            for i in 0..n {
                let o: Retained<AnyObject> = msg_send![&*subviews, objectAtIndex: i];
                let f: NSRect = msg_send![&*o, frame];
                let hidden: bool = msg_send![&*o, isHidden];
                let opaque: bool = msg_send![&*o, isOpaque];
                let ours = std::ptr::eq(
                    std::ptr::from_ref::<AnyObject>(&o).cast::<u8>(),
                    std::ptr::from_ref::<NSView>(view).cast::<u8>(),
                );
                eprintln!(
                    "smoke:   subview {i} {} frame {} hidden {hidden} opaque {opaque}{}",
                    class_of(&o),
                    rect(f),
                    if ours { " (the viewport)" } else { "" }
                );
            }
        }
    }
}
