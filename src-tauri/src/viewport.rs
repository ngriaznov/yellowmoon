//! The viewport: what draws the photo in the photo cell (`docs/01` §Native viewport, `docs/16`).
//!
//! The webview tells the shell where the cell is (`viewport_set_rect`) and what the pointer does
//! over it (`viewport_input`, `viewport_set_mode`, `set_before`). [`Viewport`] keeps the core's
//! [`ViewportState`] for the current photo, asks the core for a frame whenever the state, the
//! photo or its settings change, and hands the frame to a [`FrameSink`]:
//!
//! * the **native** sink (macOS, `crate::native`): a wgpu surface on a Metal layer behind the
//!   transparent webview. The frame goes from the core's memory to the layer; it never crosses the
//!   bridge.
//! * the **fallback** sink ([`FallbackSink`], everywhere): the frame is encoded as a PNG with the
//!   Display P3 profile, kept in a [`FrameStore`] and served to the webview at
//!   `ym-frame://localhost/frame-<seq>.png`; the `viewport_frame` event says when a new one is
//!   there. The same store and scheme serve the filmstrip's thumbnails
//!   (`ym-frame://localhost/thumb-<photo>-<version>.png`) and the Look tab's tiles
//!   (`ym-frame://localhost/look-<photo>-<version>.png`), whichever sink draws the photo. This breaks the rule "full images never cross the bridge" (`docs/01`), and is meant for
//!   development builds on other systems and as the M0-06 fallback candidate (`docs/16`).
//!
//! # Threading
//!
//! Commands only record what the webview asked for (a list of [`Op`]s) and wake the worker thread.
//! The worker owns the rendering: it applies the recorded operations to the state (they need the
//! size of the final image), asks the renderer for the frame at viewport priority, presents it, and
//! asks again when the frame was made from a coarser image than the zoom needs
//! (`FrameOptions::coarse_only`: the tiles are computed at tile priority, then a sharp frame
//! follows). A newer request cancels the render in flight, so dragging a slider or a photo never
//! queues frames behind each other.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ym_core::app::Core;
use ym_core::pipeline::{
    FrameOptions, PhotoSource, RenderError, RenderedFrame, ViewportInput as CoreInput,
    ViewportMode as CoreMode, ViewportRect as CoreRect, ViewportState,
};
use ym_core::session::{PhotoId as CorePhotoId, SessionError};
use ym_gpu::queue::{CancelToken, JobError};

use crate::events::{AppEvent, Emitter};
use crate::wire::{
    AppError, ErrorScope, PhotoId, ViewportBackendKind, ViewportFrame, ViewportInput, ViewportMode,
    ViewportRect, convert,
};

/// The URI scheme the fallback viewport serves frames on.
pub const FRAME_SCHEME: &str = "ym-frame";

/// The smallest side, in CSS pixels, of a photo cell that counts as the webview having reported
/// one: the first frame waits for it.
const MIN_CELL_PX: f64 = 2.0;

/// How long a reader waits for the worker to apply what the webview asked for before it uses the
/// state as it is.
const SYNC_TIMEOUT: Duration = Duration::from_millis(1500);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Frame sinks
// ---------------------------------------------------------------------------------------------

/// What shows a finished frame.
pub trait FrameSink: Send + Sync + 'static {
    /// Which backend this is.
    fn kind(&self) -> ViewportBackendKind;

    /// The photo cell moved or changed size; `rect` is in CSS pixels of the window. Called as soon
    /// as the webview reports it, before the frame for the new size is ready.
    fn set_rect(&self, rect: &ViewportRect);

    /// Shows a frame (RGBA8, Display P3, the size of the cell in device pixels).
    ///
    /// # Errors
    ///
    /// A plain-words reason when the frame could not be shown.
    fn present(&self, frame: &RenderedFrame, seq: u32) -> Result<(), String>;

    /// Shows nothing (no photo, a photo that can't be drawn, the session closed). `seq` counts on
    /// from the last frame, so a listener that orders by it sees the picture go.
    fn clear(&self, seq: u32);
}

/// The last few frames of the fallback viewport, the latest thumbnail of every photo and the look
/// tiles of the Look tab, as PNG files, served to the webview by the `ym-frame` scheme.
#[derive(Debug, Default)]
pub struct FrameStore {
    frames: Mutex<VecDeque<(u32, Arc<Vec<u8>>)>>,
    thumbnails: Mutex<Thumbnails>,
    look_tiles: Mutex<LookTiles>,
}

/// A PNG file, shared with the responses that serve it.
type Png = Arc<Vec<u8>>;

/// The thumbnails the store holds.
#[derive(Debug, Default)]
struct Thumbnails {
    /// Counts up with every thumbnail put, so that a new picture is a new address.
    version: u32,
    /// Per photo, the newest versions (see [`THUMBNAIL_VERSIONS_KEPT`]), oldest first.
    by_photo: BTreeMap<u32, Vec<(u32, Png)>>,
}

/// The look tiles the store holds: a photo through one look, at most 240 px.
#[derive(Debug)]
struct LookTiles {
    /// Counts up with every tile put, so that a new picture is a new address.
    version: u32,
    /// Per photo and look, the newest versions (see [`THUMBNAIL_VERSIONS_KEPT`]), oldest first.
    by_look: BTreeMap<(u32, String), Vec<(u32, Png)>>,
    /// The bytes of the PNGs in `by_look`.
    bytes: usize,
    /// The most bytes kept (see [`LOOK_TILE_BUDGET_BYTES`]).
    budget: usize,
    /// Counts up whenever a photo's tiles are put or served.
    clock: u64,
    /// When each photo's tiles were last put or served, by `clock`.
    last_used: BTreeMap<u32, u64>,
}

impl Default for LookTiles {
    fn default() -> Self {
        Self::with_budget(LOOK_TILE_BUDGET_BYTES)
    }
}

impl LookTiles {
    fn with_budget(budget: usize) -> Self {
        Self {
            version: 0,
            by_look: BTreeMap::new(),
            bytes: 0,
            budget,
            clock: 0,
            last_used: BTreeMap::new(),
        }
    }

    fn touch(&mut self, photo: u32) {
        self.clock += 1;
        self.last_used.insert(photo, self.clock);
    }

    /// Forgets the tiles of the photos used least recently until the budget holds. The photo that
    /// was just given a tile stays, even if its tiles alone pass the budget.
    fn enforce_budget(&mut self, newest: u32) {
        while self.bytes > self.budget {
            let Some(victim) = self
                .last_used
                .iter()
                .filter(|(photo, _)| **photo != newest)
                .min_by_key(|(_, used)| **used)
                .map(|(photo, _)| *photo)
            else {
                break;
            };
            self.forget_photo(victim);
        }
    }

    fn forget_photo(&mut self, photo: u32) {
        let keys: Vec<(u32, String)> = self
            .by_look
            .range((photo, String::new())..)
            .take_while(|((p, _), _)| *p == photo)
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys {
            if let Some(versions) = self.by_look.remove(&key) {
                self.bytes -= versions.iter().map(|(_, png)| png.len()).sum::<usize>();
            }
        }
        self.last_used.remove(&photo);
    }

    /// Forgets every tile; the version counter goes on, so no old address is handed out again.
    fn clear(&mut self) {
        self.by_look.clear();
        self.last_used.clear();
        self.bytes = 0;
    }
}

/// How many bytes of look tiles the store keeps: about 4,500 tiles of 57 KB (a 240 px PNG), which is
/// every tile of a 450-photo session with ten looks on show. Past it the photos used least recently
/// lose their tiles. (With ARRI looks imported and the whole grid scrolled, each photo asks for a
/// tile per look, 5 to 11 MB a photo, and a session of hundreds of photos would otherwise keep
/// gigabytes until it closes.) A photo whose tiles are gone answers 404 for the addresses it was
/// given; the webview asks for new ones when the photo's settings change.
pub const LOOK_TILE_BUDGET_BYTES: usize = 256 << 20;

/// How many frames the store keeps: the webview may still be loading the previous one.
const FRAMES_KEPT: usize = 3;

/// How many versions of one photo's thumbnail (or of one look tile of a photo) the store keeps: the
/// picture the webview is still showing may be asked for again (a layout that remakes the `<img>`)
/// a moment after a newer one was announced.
const THUMBNAIL_VERSIONS_KEPT: usize = 2;

impl FrameStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty store that keeps at most `bytes` of look tiles (the tests' way to reach the limit
    /// with a few bytes; the application uses [`LOOK_TILE_BUDGET_BYTES`]).
    #[cfg(test)]
    #[must_use]
    pub fn with_look_tile_budget(bytes: usize) -> Self {
        Self {
            look_tiles: Mutex::new(LookTiles::with_budget(bytes)),
            ..Self::default()
        }
    }

    /// Keeps the PNG of frame `seq`.
    pub fn put(&self, seq: u32, png: Vec<u8>) {
        let mut frames = lock(&self.frames);
        frames.push_back((seq, Arc::new(png)));
        while frames.len() > FRAMES_KEPT {
            frames.pop_front();
        }
    }

    /// The PNG of frame `seq`, if it is still kept.
    #[must_use]
    pub fn get(&self, seq: u32) -> Option<Arc<Vec<u8>>> {
        lock(&self.frames)
            .iter()
            .find(|(s, _)| *s == seq)
            .map(|(_, png)| Arc::clone(png))
    }

    /// The newest frame.
    #[cfg(test)]
    #[must_use]
    pub fn latest(&self) -> Option<(u32, Arc<Vec<u8>>)> {
        lock(&self.frames).back().cloned()
    }

    /// Forgets every frame.
    pub fn clear(&self) {
        lock(&self.frames).clear();
    }

    /// Keeps the thumbnail of a photo (a PNG of at most 240 px) and returns the address it is
    /// served at. Every call gives a new address, so a refreshed thumbnail never meets a cached
    /// older picture.
    pub fn put_thumbnail(&self, photo: u32, png: Vec<u8>) -> String {
        let mut thumbs = lock(&self.thumbnails);
        thumbs.version = thumbs.version.wrapping_add(1);
        let version = thumbs.version;
        let versions = thumbs.by_photo.entry(photo).or_default();
        versions.push((version, Arc::new(png)));
        while versions.len() > THUMBNAIL_VERSIONS_KEPT {
            versions.remove(0);
        }
        thumbnail_url(photo, version)
    }

    /// The address of the newest thumbnail of every photo that has one, in photo order.
    #[must_use]
    pub fn thumbnail_urls(&self) -> Vec<(u32, String)> {
        lock(&self.thumbnails)
            .by_photo
            .iter()
            .filter_map(|(photo, versions)| {
                versions
                    .last()
                    .map(|(version, _)| (*photo, thumbnail_url(*photo, *version)))
            })
            .collect()
    }

    /// Keeps the tile of a photo through one look (a PNG of at most 240 px) and returns the
    /// address it is served at. Every call gives a new address, so a tile drawn again after the
    /// photo's settings changed never meets a cached older picture. Only the newest few tiles of
    /// each (photo, look) are kept, the photos used least recently lose all theirs when the tiles
    /// together pass [`LOOK_TILE_BUDGET_BYTES`], and none survive [`FrameStore::clear_thumbnails`].
    pub fn put_look_tile(&self, photo: u32, look: &str, png: Vec<u8>) -> String {
        let mut tiles = lock(&self.look_tiles);
        tiles.version = tiles.version.wrapping_add(1);
        let version = tiles.version;
        let added = png.len();
        let mut dropped = 0;
        let versions = tiles.by_look.entry((photo, look.to_owned())).or_default();
        versions.push((version, Arc::new(png)));
        while versions.len() > THUMBNAIL_VERSIONS_KEPT {
            dropped += versions.remove(0).1.len();
        }
        tiles.bytes = tiles.bytes + added - dropped;
        tiles.touch(photo);
        tiles.enforce_budget(photo);
        look_tile_url(photo, version)
    }

    /// Forgets every thumbnail and look tile (the session closed). Addresses that were handed out
    /// answer 404 from now on; new ones count on from where the old ones stopped.
    pub fn clear_thumbnails(&self) {
        lock(&self.thumbnails).by_photo.clear();
        lock(&self.look_tiles).clear();
    }

    /// The look tile at `photo` and `version`, if it is still kept. Serving a tile counts as using
    /// its photo.
    fn look_tile(&self, photo: u32, version: u32) -> Option<Arc<Vec<u8>>> {
        let mut tiles = lock(&self.look_tiles);
        let png = tiles
            .by_look
            .range((photo, String::new())..)
            .take_while(|((p, _), _)| *p == photo)
            .flat_map(|(_, versions)| versions)
            .find(|(v, _)| *v == version)
            .map(|(_, png)| Arc::clone(png))?;
        tiles.touch(photo);
        Some(png)
    }

    /// The thumbnail at `photo` and `version`, if it is still kept.
    fn thumbnail(&self, photo: u32, version: u32) -> Option<Arc<Vec<u8>>> {
        lock(&self.thumbnails)
            .by_photo
            .get(&photo)?
            .iter()
            .find(|(v, _)| *v == version)
            .map(|(_, png)| Arc::clone(png))
    }

    /// The response for a request to `ym-frame://localhost/frame-<seq>.png`,
    /// `…/thumb-<photo>-<version>.png` or `…/look-<photo>-<version>.png`: the file, or why not.
    #[must_use]
    pub fn respond(&self, path: &str) -> (u16, &'static str, Cow<'static, [u8]>) {
        let name = path
            .trim_start_matches('/')
            .strip_suffix(".png")
            .unwrap_or("");
        let png = if let Some(seq) = name.strip_prefix("frame-") {
            seq.parse::<u32>().ok().and_then(|s| self.get(s))
        } else if let Some(rest) = name.strip_prefix("thumb-") {
            rest.split_once('-').and_then(|(photo, version)| {
                self.thumbnail(photo.parse().ok()?, version.parse().ok()?)
            })
        } else if let Some(rest) = name.strip_prefix("look-") {
            rest.split_once('-').and_then(|(photo, version)| {
                self.look_tile(photo.parse().ok()?, version.parse().ok()?)
            })
        } else {
            None
        };
        match png {
            Some(png) => (200, "image/png", Cow::Owned(png.as_ref().clone())),
            None => (404, "text/plain", Cow::Borrowed(b"No such frame.")),
        }
    }
}

/// The address of frame `seq` in the webview.
#[must_use]
pub fn frame_url(seq: u32) -> String {
    scheme_url(&format!("frame-{seq}.png"))
}

/// The address of version `version` of a photo's thumbnail in the webview.
#[must_use]
pub fn thumbnail_url(photo: u32, version: u32) -> String {
    scheme_url(&format!("thumb-{photo}-{version}.png"))
}

/// The address of version `version` of a look tile of a photo in the webview.
#[must_use]
pub fn look_tile_url(photo: u32, version: u32) -> String {
    scheme_url(&format!("look-{photo}-{version}.png"))
}

fn scheme_url(file: &str) -> String {
    if cfg!(windows) {
        format!("http://{FRAME_SCHEME}.localhost/{file}")
    } else {
        format!("{FRAME_SCHEME}://localhost/{file}")
    }
}

/// Encodes a frame as an RGB PNG that carries the Display P3 profile, so the webview shows the
/// colours the core made (an untagged PNG would be read as sRGB). The fast setting: a frame is
/// replaced by the next one within a second.
///
/// # Errors
///
/// A plain-words reason when the encoder refuses the size.
pub fn encode_frame_png(frame: &RenderedFrame) -> Result<Vec<u8>, String> {
    let (w, h) = (frame.width, frame.height);
    if w == 0 || h == 0 || frame.rgba.len() != 4 * w * h {
        return Err("The photo could not be drawn: the frame has the wrong size.".to_owned());
    }
    let mut rgb = Vec::with_capacity(3 * w * h);
    for px in frame.rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    let width = u32::try_from(w).map_err(|_| "The frame is too wide.".to_owned())?;
    let height = u32::try_from(h).map_err(|_| "The frame is too tall.".to_owned())?;
    let mut info = png::Info::with_size(width, height);
    info.color_type = png::ColorType::Rgb;
    info.bit_depth = png::BitDepth::Eight;
    info.icc_profile = Some(Cow::Borrowed(ym_export::icc::profile(
        ym_export::ColorSpace::DisplayP3,
    )));
    let mut out = Vec::with_capacity(rgb.len() / 3);
    let mut encoder = png::Encoder::with_info(&mut out, info).map_err(|e| e.to_string())?;
    encoder.set_compression(png::Compression::Fast);
    encoder.set_filter(png::Filter::Sub);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(&rgb).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(out)
}

/// The fallback viewport: frames become PNGs in a [`FrameStore`] and a `viewport_frame` event.
pub struct FallbackSink {
    store: Arc<FrameStore>,
    emitter: Arc<dyn Emitter>,
}

impl FallbackSink {
    /// A sink over `store` that announces frames through `emitter`.
    #[must_use]
    pub fn new(store: Arc<FrameStore>, emitter: Arc<dyn Emitter>) -> Self {
        Self { store, emitter }
    }
}

impl FrameSink for FallbackSink {
    fn kind(&self) -> ViewportBackendKind {
        ViewportBackendKind::Fallback
    }

    fn set_rect(&self, _rect: &ViewportRect) {
        // The webview lays the image over the cell itself.
    }

    fn present(&self, frame: &RenderedFrame, seq: u32) -> Result<(), String> {
        let png = encode_frame_png(frame)?;
        self.store.put(seq, png);
        self.emitter.emit(AppEvent::ViewportFrame(ViewportFrame {
            seq,
            width: u32::try_from(frame.width).unwrap_or(u32::MAX),
            height: u32::try_from(frame.height).unwrap_or(u32::MAX),
            sharp: frame.sharp,
            url: frame_url(seq),
        }));
        Ok(())
    }

    fn clear(&self, seq: u32) {
        self.store.clear();
        // The webview keeps showing the last image until it is told otherwise: an empty address
        // and a size of 0 x 0 mean "no picture".
        self.emitter.emit(AppEvent::ViewportFrame(ViewportFrame {
            seq,
            width: 0,
            height: 0,
            sharp: true,
            url: String::new(),
        }));
    }
}

// ---------------------------------------------------------------------------------------------
// The controller
// ---------------------------------------------------------------------------------------------

/// Something the webview asked for that changes the viewport state. Applied by the worker, in
/// order, because most of them need the size of the final image.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Rect(CoreRect),
    Input(CoreInput),
    Mode(CoreMode),
    Before(bool),
}

struct VState {
    vp: ViewportState,
    /// The webview has reported the photo cell with a usable size. Until it has there is nothing
    /// to draw into: the state starts with a 1 x 1 cell, and a frame of that size is no picture.
    has_rect: bool,
    photo: Option<CorePhotoId>,
    ops: Vec<Op>,
    /// The worker has taken `ops` and not yet applied them.
    applying: bool,
    /// A new frame is wanted.
    dirty: bool,
    /// The worker is busy (rendering, or about to).
    busy: bool,
    /// The photo was not ready when the frame was asked for: render when it is.
    awaiting: bool,
    /// Tiles have just been computed: the next frame asks for sharp pixels.
    want_sharp: bool,
    /// The render in flight, so that a newer request can stop it.
    in_flight: Option<CancelToken>,
    seq: u32,
    frames: u64,
    /// The last error shown, so that a photo that fails to render does not repeat itself.
    last_error: Option<(Option<CorePhotoId>, String)>,
    /// Counts the photos put on screen. What the webview did to one photo (a zoom, a pan, Before)
    /// is not done to the next one when a round was already under way for the first.
    epoch: u64,
    /// The sink shows nothing: no clear is sent twice in a row.
    cleared: bool,
}

/// The state shared between the controller, the worker and the hooks the core's events call.
pub struct Shared {
    state: Mutex<VState>,
    wake: Condvar,
    settled: Condvar,
    idle: Condvar,
    emitter: Arc<dyn Emitter>,
    sink: Arc<dyn FrameSink>,
    core: OnceLock<Weak<Core>>,
    shutdown: AtomicBool,
}

impl Shared {
    fn request(&self, st: &mut VState) {
        st.dirty = true;
        st.want_sharp = false;
        if let Some(token) = &st.in_flight {
            token.cancel();
        }
        self.wake.notify_all();
    }

    /// Something about the photo changed (settings, look, display rendering): draw it again.
    pub fn invalidate(&self) {
        let mut st = lock(&self.state);
        self.request(&mut st);
    }

    /// A photo finished developing, or failed to (then there is nothing to draw and the cell has
    /// to say so): if it is the one on screen and was waiting, look at it again.
    pub fn photo_ready(&self, id: CorePhotoId) {
        let mut st = lock(&self.state);
        if st.photo == Some(id) && st.awaiting {
            st.awaiting = false;
            self.request(&mut st);
        }
    }

    /// The estimates of a photo changed (a burst median moved): draw it again if it is on screen.
    pub fn estimates_changed(&self, id: CorePhotoId) {
        let mut st = lock(&self.state);
        if st.photo == Some(id) {
            self.request(&mut st);
        }
    }

    fn core(&self) -> Option<Arc<Core>> {
        self.core.get().and_then(Weak::upgrade)
    }

    fn report(&self, scope: ErrorScope, photo: Option<CorePhotoId>, message: String) {
        {
            let mut st = lock(&self.state);
            if st.last_error.as_ref() == Some(&(photo, message.clone())) {
                return;
            }
            st.last_error = Some((photo, message.clone()));
        }
        self.emitter
            .emit(AppEvent::Error(AppError::new(scope, message)));
    }

    /// Blocks until there is something to do; `None` when shutting down.
    fn next_work(&self) -> Option<Work> {
        let mut st = lock(&self.state);
        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return None;
            }
            // Nothing is drawn, and nothing applied, before the webview has said where the cell is.
            // Operations parked for a photo that is still developing (`awaiting`) wait for its
            // `photo_ready`; taking them again at once would spin.
            if st.has_rect && (st.dirty || (!st.ops.is_empty() && !st.awaiting)) {
                st.dirty = false;
                st.busy = true;
                st.applying = !st.ops.is_empty();
                let ops = std::mem::take(&mut st.ops);
                let coarse_only = !std::mem::take(&mut st.want_sharp);
                return Some(Work {
                    photo: st.photo,
                    epoch: st.epoch,
                    ops,
                    coarse_only,
                });
            }
            st.busy = false;
            self.idle.notify_all();
            st = self.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Applies operations to the state without the image (nothing is on screen).
    fn apply_blind(&self, ops: Vec<Op>, epoch: u64) {
        let mut st = lock(&self.state);
        let ops = for_photo(ops, st.epoch == epoch);
        for op in ops {
            match op {
                Op::Rect(r) => st.vp.rect = r,
                Op::Mode(m) => {
                    st.vp.mode = m;
                    st.vp.centre = [0.5, 0.5];
                }
                Op::Before(b) => st.vp.before = b,
                Op::Input(_) => {}
            }
        }
        st.applying = false;
        self.settled.notify_all();
    }

    /// Puts operations back in front of the newer ones, to be applied at the next round.
    fn requeue(&self, ops: Vec<Op>, epoch: u64) {
        let mut st = lock(&self.state);
        let ops = for_photo(ops, st.epoch == epoch);
        if !ops.is_empty() {
            st.ops.splice(0..0, ops);
        }
        st.applying = false;
        self.settled.notify_all();
    }

    /// The photo is still developing: keeps the operations for it and waits for `photo_ready`.
    fn park(&self, core: &Core, ops: Vec<Op>, photo: CorePhotoId, epoch: u64) {
        self.park_state(ops, photo, epoch);
        // The photo may have been developed (or given up) between the check and the flag, and
        // its event been lost to the flag being down: ask once more.
        if !matches!(
            PhotoSource::from_session(&core.session, photo),
            Err(SessionError::NotReady(_))
        ) {
            self.photo_ready(photo);
        }
    }

    /// The state half of [`Shared::park`].
    fn park_state(&self, ops: Vec<Op>, photo: CorePhotoId, epoch: u64) {
        let mut st = lock(&self.state);
        let ops = for_photo(ops, st.epoch == epoch);
        if !ops.is_empty() {
            st.ops.splice(0..0, ops);
        }
        st.applying = false;
        // Only for the photo that is on screen: a newer request is already on its way when
        // another one is.
        if st.photo == Some(photo) {
            st.awaiting = true;
        }
        self.settled.notify_all();
    }

    /// Tells the sink that nothing is shown (not twice in a row).
    fn clear_sink(&self) {
        let seq = {
            let mut st = lock(&self.state);
            if st.cleared {
                return;
            }
            st.cleared = true;
            st.seq = st.seq.wrapping_add(1);
            st.seq
        };
        self.sink.clear(seq);
    }

    fn present(&self, frame: &RenderedFrame, photo: CorePhotoId) {
        let seq = {
            let mut st = lock(&self.state);
            st.seq = st.seq.wrapping_add(1);
            st.cleared = false;
            st.seq
        };
        match self.sink.present(frame, seq) {
            Ok(()) => {
                lock(&self.state).frames += 1;
                if let Ok(stats) = convert(&frame.report.stats()) {
                    self.emitter.emit(AppEvent::RenderStats(stats));
                }
            }
            Err(message) => self.report(ErrorScope::Viewport, Some(photo), message),
        }
    }

    /// One round of the worker: apply what was asked for, render, present.
    fn round(&self, work: Work) {
        let Work {
            photo,
            epoch,
            ops,
            coarse_only,
        } = work;
        let Some(core) = self.core() else {
            self.apply_blind(ops, epoch);
            return;
        };
        let Some(photo) = photo else {
            self.apply_blind(ops, epoch);
            self.clear_sink();
            return;
        };
        let src = match PhotoSource::from_session(&core.session, photo) {
            Ok(src) => src,
            Err(SessionError::NotReady(_)) => {
                // Draw it when it is developed (`photo_ready`).
                self.park(&core, ops, photo, epoch);
                return;
            }
            Err(SessionError::UnknownPhoto(_)) => {
                self.apply_blind(ops, epoch);
                self.clear_sink();
                return;
            }
            Err(SessionError::Skipped(_)) => {
                // The filmstrip and the cell say why; there is nothing to draw.
                self.apply_blind(ops, epoch);
                self.clear_sink();
                return;
            }
            Err(e) => {
                self.apply_blind(ops, epoch);
                self.report(ErrorScope::Decode, Some(photo), e.to_string());
                return;
            }
        };
        let cancel = CancelToken::new();
        {
            let mut st = lock(&self.state);
            st.in_flight = Some(cancel.clone());
            if st.dirty || !st.ops.is_empty() {
                // Already out of date: skip to the newer request.
                cancel.cancel();
            }
        }
        let image = match core.renderer.final_size(&src, &cancel) {
            Ok(image) => image,
            Err(RenderError::Cancelled) => {
                self.requeue(ops, epoch);
                return;
            }
            Err(e) => {
                self.apply_blind(ops, epoch);
                self.report(ErrorScope::Render, Some(photo), e.to_string());
                return;
            }
        };
        let vp = {
            let mut st = lock(&self.state);
            let ops = for_photo(ops, st.epoch == epoch);
            for op in ops {
                match op {
                    Op::Rect(r) => st.vp.set_rect(r, image),
                    Op::Mode(m) => st.vp.set_mode(m, image),
                    Op::Input(i) => {
                        st.vp.input(&i, image);
                    }
                    Op::Before(b) => st.vp.before = b,
                }
            }
            st.applying = false;
            self.settled.notify_all();
            st.vp.clone()
        };
        let options = FrameOptions {
            coarse_only,
            no_cache: false,
        };
        let handle = core
            .renderer
            .spawn_viewport_frame(src.clone(), vp.clone(), options);
        {
            let mut st = lock(&self.state);
            st.in_flight = Some(handle.token().clone());
            if st.dirty || !st.ops.is_empty() {
                handle.cancel();
            }
        }
        let frame = match handle.wait() {
            Ok(Ok(frame)) => frame,
            Ok(Err(RenderError::Cancelled)) | Err(JobError::Cancelled | JobError::Shutdown) => {
                return;
            }
            Ok(Err(e)) => {
                self.report(ErrorScope::Render, Some(photo), e.to_string());
                return;
            }
            Err(e) => {
                tracing::error!(error = %e, "the viewport render failed");
                self.report(
                    ErrorScope::Render,
                    Some(photo),
                    "The photo could not be drawn because of an unexpected problem. Try again."
                        .to_owned(),
                );
                return;
            }
        };
        lock(&self.state).last_error = None;
        self.present(&frame, photo);
        if !frame.sharp {
            // The zoom needs full-resolution tiles that were not ready: compute them at tile
            // priority, then draw the sharp frame.
            let tiles = core.renderer.spawn_tiles(src, vp);
            {
                let mut st = lock(&self.state);
                st.in_flight = Some(tiles.token().clone());
                if st.dirty || !st.ops.is_empty() {
                    tiles.cancel();
                }
            }
            if let Ok(Ok(_)) = tiles.wait() {
                let mut st = lock(&self.state);
                if !st.dirty && st.ops.is_empty() {
                    st.dirty = true;
                    st.want_sharp = true;
                }
            }
        }
    }
}

struct Work {
    photo: Option<CorePhotoId>,
    epoch: u64,
    ops: Vec<Op>,
    coarse_only: bool,
}

/// What of `ops` still applies once the photo on screen has (or has not) changed: the cell's
/// rectangle belongs to the window and always stays, the rest belonged to the photo it was
/// recorded for.
fn for_photo(ops: Vec<Op>, same_photo: bool) -> Vec<Op> {
    if same_photo {
        ops
    } else {
        ops.into_iter()
            .filter(|op| matches!(op, Op::Rect(_)))
            .collect()
    }
}

/// The viewport controller (see the module documentation).
pub struct Viewport {
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Viewport {
    /// A controller that presents its frames to `sink` and reports through `emitter`. It draws
    /// nothing until [`Viewport::attach`] gives it the core.
    #[must_use]
    pub fn new(emitter: Arc<dyn Emitter>, sink: Arc<dyn FrameSink>) -> Self {
        let state = VState {
            vp: ViewportState::new(CoreRect::cell(1.0, 1.0, 1.0)),
            has_rect: false,
            photo: None,
            ops: Vec::new(),
            applying: false,
            dirty: false,
            busy: false,
            awaiting: false,
            want_sharp: false,
            in_flight: None,
            seq: 0,
            frames: 0,
            last_error: None,
            epoch: 0,
            // Nothing has been shown yet.
            cleared: true,
        };
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(state),
                wake: Condvar::new(),
                settled: Condvar::new(),
                idle: Condvar::new(),
                emitter,
                sink,
                core: OnceLock::new(),
                shutdown: AtomicBool::new(false),
            }),
            worker: Mutex::new(None),
        }
    }

    /// The part of the controller the core's event callbacks may call.
    #[must_use]
    pub fn shared(&self) -> Arc<Shared> {
        Arc::clone(&self.shared)
    }

    /// Which backend shows the frames.
    #[must_use]
    pub fn kind(&self) -> ViewportBackendKind {
        self.shared.sink.kind()
    }

    /// Gives the controller the core and starts its worker thread.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the thread cannot start.
    pub fn attach(&self, core: &Arc<Core>) -> Result<(), AppError> {
        let _ = self.shared.core.set(Arc::downgrade(core));
        let shared = Arc::clone(&self.shared);
        let handle = std::thread::Builder::new()
            .name("ym-viewport".to_owned())
            .spawn(move || {
                while let Some(work) = shared.next_work() {
                    // A panic in a render must not end the worker: the next request may work.
                    let round = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        shared.round(work);
                    }));
                    if round.is_err() {
                        tracing::error!("the viewport worker caught a panic");
                        let mut st = lock(&shared.state);
                        st.applying = false;
                        shared.settled.notify_all();
                    }
                }
            })
            .map_err(AppError::internal)?;
        *lock(&self.worker) = Some(handle);
        Ok(())
    }

    /// The photo on screen changed (or there is none): the view goes back to Fit.
    pub fn set_photo(&self, photo: Option<CorePhotoId>) {
        let mut st = lock(&self.shared.state);
        if st.photo == photo {
            return;
        }
        st.photo = photo;
        st.epoch += 1;
        st.awaiting = false;
        st.vp.mode = CoreMode::Fit;
        st.vp.centre = [0.5, 0.5];
        st.vp.before = false;
        st.last_error = None;
        // What the webview did to the photo that was on screen is not done to this one; the cell
        // (the window's own) stays.
        let kept = for_photo(std::mem::take(&mut st.ops), false);
        st.ops = kept;
        let none = photo.is_none();
        self.shared.request(&mut st);
        drop(st);
        if none {
            self.shared.clear_sink();
        }
    }

    /// `viewport_set_rect`.
    pub fn set_rect(&self, rect: &ViewportRect) -> Result<(), AppError> {
        let core_rect: CoreRect = convert(rect)?;
        self.shared.sink.set_rect(rect);
        let mut st = lock(&self.shared.state);
        // A cell with no area (a hidden screen, a layout that has not run) is not the rectangle
        // the first frame waits for.
        if rect.width.0 >= MIN_CELL_PX && rect.height.0 >= MIN_CELL_PX {
            st.has_rect = true;
        }
        st.ops.push(Op::Rect(core_rect));
        self.shared.request(&mut st);
        Ok(())
    }

    /// `viewport_input` (everything but Pick white, which [`crate::api::AppApi`] handles).
    pub fn input(&self, event: &ViewportInput) -> Result<(), AppError> {
        let core_event: CoreInput = convert(event)?;
        let mut st = lock(&self.shared.state);
        // Several pans in a row are one pan: the worker is usually busy while the pointer moves.
        if let (CoreInput::Pan { dx, dy }, Some(Op::Input(CoreInput::Pan { dx: px, dy: py }))) =
            (core_event, st.ops.last_mut())
        {
            *px += dx;
            *py += dy;
        } else {
            st.ops.push(Op::Input(core_event));
        }
        self.shared.request(&mut st);
        Ok(())
    }

    /// `viewport_set_mode`.
    pub fn set_mode(&self, mode: &ViewportMode) -> Result<(), AppError> {
        let core_mode: CoreMode = convert(mode)?;
        let mut st = lock(&self.shared.state);
        st.ops.push(Op::Mode(core_mode));
        self.shared.request(&mut st);
        Ok(())
    }

    /// `set_before`.
    pub fn set_before(&self, before: bool) {
        let mut st = lock(&self.shared.state);
        st.ops.push(Op::Before(before));
        self.shared.request(&mut st);
    }

    /// Draw again (settings changed).
    pub fn invalidate(&self) {
        self.shared.invalidate();
    }

    /// The photo on screen and the state as the webview left it, after the worker has applied
    /// what is pending (so a click is converted with the zoom the person saw).
    #[must_use]
    pub fn snapshot_synced(&self) -> (Option<CorePhotoId>, ViewportState) {
        let deadline = Instant::now() + SYNC_TIMEOUT;
        let mut st = lock(&self.shared.state);
        // Operations parked for a photo that is still developing are not coming.
        while st.has_rect
            && ((!st.ops.is_empty() && !st.awaiting) || st.applying)
            && Instant::now() < deadline
        {
            let left = deadline.saturating_duration_since(Instant::now());
            st = self
                .shared
                .settled
                .wait_timeout(st, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        (st.photo, st.vp.clone())
    }

    /// Waits until nothing is pending and no render runs; false if `timeout` ran out first.
    #[cfg(test)]
    #[must_use]
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.shared.state);
        // Before the first rectangle nothing runs, so there is nothing to wait for.
        while st.has_rect && (st.dirty || st.busy || (!st.ops.is_empty() && !st.awaiting)) {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            st = self
                .shared
                .idle
                .wait_timeout(st, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }

    /// How many frames have been presented.
    #[cfg(test)]
    #[must_use]
    pub fn frames_presented(&self) -> u64 {
        lock(&self.shared.state).frames
    }
}

impl Drop for Viewport {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        if let Some(token) = &lock(&self.shared.state).in_flight {
            token.cancel();
        }
        self.shared.wake.notify_all();
        if let Some(handle) = lock(&self.worker).take() {
            let _ = handle.join();
        }
    }
}

/// Converts the photo id of the wire to the core's and back.
#[must_use]
pub fn core_id(id: PhotoId) -> CorePhotoId {
    CorePhotoId(id.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: usize, height: usize) -> RenderedFrame {
        RenderedFrame {
            width,
            height,
            rgba: (0..width * height)
                .flat_map(|i| [(i % 251) as u8, (i % 13) as u8 * 19, 7, 255])
                .collect(),
            sharp: true,
            report: ym_core::pipeline::RenderReport {
                ms: 1.0,
                path: ym_core::pipeline::RenderPath::Classical,
                space: ym_export::ColorSpace::DisplayP3,
                device: ym_core::pipeline::ComputeDevice::Cpu,
                scale: 1,
                runs: ym_core::pipeline::StageRuns::default(),
                timings: ym_core::pipeline::StageTimings::default(),
                notes: Vec::new(),
                zoom: None,
            },
        }
    }

    #[test]
    fn a_frame_becomes_a_png_with_the_display_p3_profile() {
        let png = encode_frame_png(&frame(37, 21)).expect("encode");
        let decoder = png::Decoder::new(std::io::Cursor::new(png));
        let mut reader = decoder.read_info().expect("a PNG");
        let icc = reader.info().icc_profile.clone();
        assert_eq!(
            icc.as_deref(),
            Some(ym_export::icc::profile(ym_export::ColorSpace::DisplayP3))
        );
        let mut buf = vec![0; reader.output_buffer_size().expect("size")];
        let info = reader.next_frame(&mut buf).expect("pixels");
        assert_eq!((info.width, info.height), (37, 21));
        assert_eq!(info.color_type, png::ColorType::Rgb);
        let original = frame(37, 21);
        for (rgb, rgba) in buf.chunks_exact(3).zip(original.rgba.chunks_exact(4)) {
            assert_eq!(rgb, &rgba[..3]);
        }
    }

    #[test]
    fn a_frame_of_the_wrong_size_is_refused() {
        let mut bad = frame(4, 4);
        bad.rgba.truncate(10);
        assert!(encode_frame_png(&bad).is_err());
        assert!(encode_frame_png(&frame(0, 0)).is_err());
    }

    #[test]
    fn the_store_keeps_the_last_frames_and_serves_them_by_address() {
        let store = FrameStore::new();
        assert!(store.latest().is_none());
        for seq in 1..=5 {
            store.put(seq, vec![seq as u8; 4]);
        }
        assert!(store.get(1).is_none() && store.get(2).is_none());
        assert_eq!(store.get(5).as_deref(), Some(&vec![5u8; 4]));
        assert_eq!(store.latest().map(|(s, _)| s), Some(5));

        let (status, mime, body) = store.respond("/frame-4.png");
        assert_eq!((status, mime), (200, "image/png"));
        assert_eq!(&*body, &[4u8; 4]);
        assert_eq!(store.respond("/frame-1.png").0, 404);
        assert_eq!(store.respond("/etc/passwd").0, 404);
        assert_eq!(store.respond("/frame-x.png").0, 404);
        store.clear();
        assert!(store.latest().is_none());
    }

    #[test]
    fn look_tiles_are_kept_per_photo_and_look_and_served_by_address() {
        let store = FrameStore::new();
        let path = |url: &str| url[url.rfind('/').expect("a path")..].to_owned();
        let status = |url: &str| store.respond(&path(url)).0;
        let neutral = store.put_look_tile(3, "builtin:neutral", vec![1]);
        let warm = store.put_look_tile(3, "user:warm", vec![2]);
        let other_photo = store.put_look_tile(4, "builtin:neutral", vec![3]);
        assert!(neutral.contains("look-3-1.png"), "{neutral}");
        for (url, byte) in [(&neutral, 1u8), (&warm, 2), (&other_photo, 3)] {
            let (code, mime, body) = store.respond(&path(url));
            assert_eq!((code, mime), (200, "image/png"), "{url}");
            assert_eq!(&*body, &[byte]);
        }
        // The newest two tiles of one photo through one look stay; the others are not touched.
        let second = store.put_look_tile(3, "builtin:neutral", vec![4]);
        assert_eq!(status(&neutral), 200);
        let third = store.put_look_tile(3, "builtin:neutral", vec![5]);
        assert_eq!(status(&neutral), 404, "the oldest of three is dropped");
        assert_eq!((status(&second), status(&third)), (200, 200));
        assert_eq!((status(&warm), status(&other_photo)), (200, 200));
        // A tile is not a thumbnail or a frame, and a version of one photo is not another's.
        assert_eq!(store.respond("/thumb-3-4.png").0, 404);
        assert_eq!(store.respond("/frame-4.png").0, 404);
        assert_eq!(store.respond("/look-4-5.png").0, 404);
        for bad in [
            "/look-3.png",
            "/look-x-1.png",
            "/look-3-x.png",
            "/look--.png",
        ] {
            assert_eq!(store.respond(bad).0, 404, "{bad}");
        }
        // Closing the session drops them all; the next address does not repeat an old one.
        store.clear_thumbnails();
        assert_eq!(status(&third), 404);
        assert_eq!(status(&warm), 404);
        let fresh = store.put_look_tile(3, "user:warm", vec![6]);
        assert_ne!(fresh, warm);
        assert_eq!(status(&fresh), 200);
    }

    #[test]
    fn look_tiles_are_bounded_by_bytes_and_the_photos_least_recently_used_go_first() {
        // A session of hundreds of photos with the Look tab open would otherwise keep every tile
        // of every photo it ever showed until the session closed.
        let store = FrameStore::with_look_tile_budget(600);
        let path = |url: &str| url[url.rfind('/').expect("a path")..].to_owned();
        let status = |url: &str| store.respond(&path(url)).0;
        let mut urls = Vec::new();
        for photo in 1..=4 {
            urls.push(store.put_look_tile(photo, "builtin:neutral", vec![photo as u8; 100]));
            urls.push(store.put_look_tile(photo, "user:warm", vec![photo as u8; 25]));
        }
        // 4 x 125 bytes fit in 600: nothing is dropped. Loading the tiles of photo 2, then 3, 4
        // and 1 (the webview loads a photo's tiles when it is shown) makes 2 the least recently
        // used.
        for photo in [2usize, 3, 4, 1] {
            for url in &urls[2 * (photo - 1)..2 * photo] {
                assert_eq!(status(url), 200, "photo {photo}");
            }
        }

        // Photo 5 takes the store to 650 bytes: photo 2 goes, whole, and that is enough.
        let five = store.put_look_tile(5, "builtin:neutral", vec![5; 150]);
        assert_eq!(status(&urls[2]), 404, "photo 2's neutral tile");
        assert_eq!(status(&urls[3]), 404, "photo 2's warm tile");
        for kept in [0, 1, 4, 5, 6, 7] {
            assert_eq!(status(&urls[kept]), 200, "{kept}");
        }
        assert_eq!(status(&five), 200);

        // The photo that was just given a tile is kept even when its tiles alone pass the budget.
        let big = store.put_look_tile(9, "builtin:neutral", vec![9; 1000]);
        assert_eq!(status(&big), 200);
        for u in urls.iter().chain([&five]) {
            assert_eq!(status(u), 404, "everything older went: {u}");
        }
    }

    #[test]
    fn a_look_tile_that_was_replaced_stops_counting_against_the_budget() {
        // Only the newest two tiles of a photo through one look are kept, so asking again for the
        // same tile does not grow the store: 3 x 300 bytes put, 600 held.
        let store = FrameStore::with_look_tile_budget(700);
        let path = |url: &str| url[url.rfind('/').expect("a path")..].to_owned();
        let status = |url: &str| store.respond(&path(url)).0;
        let first = store.put_look_tile(1, "builtin:neutral", vec![1; 300]);
        store.put_look_tile(1, "builtin:neutral", vec![1; 300]);
        let third = store.put_look_tile(1, "builtin:neutral", vec![1; 300]);
        assert_eq!(status(&first), 404, "the oldest of three is dropped");
        // 600 held + 50 = 650: within 700, so photo 1 stays. (Counting the dropped 300 as well
        // would make 950 and send photo 1 away.)
        let other = store.put_look_tile(2, "builtin:neutral", vec![2; 50]);
        assert_eq!((status(&third), status(&other)), (200, 200));
    }

    fn controller() -> (Viewport, Arc<crate::events::Recorder>) {
        let events = Arc::new(crate::events::Recorder::new());
        let emitter: Arc<dyn Emitter> = events.clone();
        let sink = Arc::new(FallbackSink::new(
            Arc::new(FrameStore::new()),
            Arc::clone(&emitter),
        ));
        (Viewport::new(emitter, sink), events)
    }

    fn cell(width: f64, height: f64) -> ViewportRect {
        use crate::wire::Num;
        ViewportRect {
            x: Num(0.0),
            y: Num(0.0),
            width: Num(width),
            height: Num(height),
            scale: Num(1.0),
        }
    }

    /// `next_work` on another thread, so that a test can tell "returns at once" from "waits".
    fn take_work_on_a_thread(shared: &Arc<Shared>) -> std::sync::mpsc::Receiver<Option<Vec<Op>>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let shared = Arc::clone(shared);
        std::thread::spawn(move || {
            let _ = tx.send(shared.next_work().map(|w| w.ops));
        });
        rx
    }

    #[test]
    fn operations_parked_for_a_developing_photo_wait_instead_of_spinning() {
        let (vp, _events) = controller();
        let shared = vp.shared();
        let photo = CorePhotoId(1);
        vp.set_photo(Some(photo));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        // A round finds the photo still developing and parks what it was given.
        let work = shared.next_work().expect("the first round");
        let epoch = work.epoch;
        assert_eq!(work.ops.len(), 1);
        shared.park_state(work.ops, photo, epoch);

        // Nothing is due: the worker waits, it does not take the same operations again.
        let rx = take_work_on_a_thread(&shared);
        assert!(
            rx.recv_timeout(Duration::from_millis(400)).is_err(),
            "the worker took parked operations again at once (a busy loop)"
        );
        // The photo is developed: the operations come back with the draw request.
        shared.photo_ready(photo);
        let ops = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the worker wakes")
            .expect("work");
        assert_eq!(ops.len(), 1, "the parked rectangle is still there");
        assert!(matches!(ops[0], Op::Rect(_)));
    }

    #[test]
    fn a_new_cell_wakes_a_parked_worker_once_and_it_parks_again() {
        let (vp, _events) = controller();
        let shared = vp.shared();
        let photo = CorePhotoId(1);
        vp.set_photo(Some(photo));
        // Nothing runs before the first rectangle (`nothing_is_drawn_before...` in the API tests).
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        let work = shared.next_work().expect("round");
        shared.park_state(work.ops, photo, work.epoch);
        // The window is resized while the photo develops: one more round, then quiet again. The
        // parked rectangle comes back in front of the new one.
        vp.set_rect(&cell(120.0, 90.0)).expect("rect");
        let work = shared.next_work().expect("a round for the new cell");
        assert_eq!(work.ops.len(), 2);
        shared.park_state(work.ops, photo, work.epoch);
        let rx = take_work_on_a_thread(&shared);
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err());
        shared.photo_ready(photo);
        assert!(rx.recv_timeout(Duration::from_secs(5)).is_ok());
    }

    #[test]
    fn what_was_done_to_one_photo_is_not_done_to_the_next() {
        let (vp, _events) = controller();
        let shared = vp.shared();
        vp.set_photo(Some(CorePhotoId(1)));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        vp.input(&ViewportInput::ToggleZoom {
            x: crate::wire::Num(5.0),
            y: crate::wire::Num(5.0),
        })
        .expect("input");
        vp.set_before(true);
        // The next photo is chosen while those are still waiting: only the cell survives.
        vp.set_photo(Some(CorePhotoId(2)));
        let work = shared.next_work().expect("round");
        assert_eq!(work.photo, Some(CorePhotoId(2)));
        assert_eq!(work.ops.len(), 1, "{:?}", work.ops);
        assert!(matches!(work.ops[0], Op::Rect(_)));

        // A round that was already under way for photo 2 when photo 3 was chosen gives back the
        // operations it had taken: the cell goes on, the zoom does not.
        let stale_epoch = work.epoch;
        vp.set_photo(Some(CorePhotoId(3)));
        shared.requeue(
            vec![
                Op::Rect(CoreRect::cell(100.0, 80.0, 1.0)),
                Op::Mode(CoreMode::Percent { value: 100.0 }),
                Op::Before(true),
            ],
            stale_epoch,
        );
        let work = shared.next_work().expect("round");
        assert_eq!(work.ops.len(), 1, "{:?}", work.ops);
        assert!(matches!(work.ops[0], Op::Rect(_)));
    }

    #[test]
    fn the_sink_is_told_when_the_picture_goes_and_only_once() {
        let (vp, events) = controller();
        let shared = vp.shared();
        let clears = || {
            events
                .events()
                .into_iter()
                .filter_map(|e| match e {
                    AppEvent::ViewportFrame(f) if f.url.is_empty() => Some(f),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        // Nothing was shown, so nothing is taken away.
        shared.clear_sink();
        assert!(clears().is_empty());
        shared.present(&frame(8, 6), CorePhotoId(1));
        shared.clear_sink();
        shared.clear_sink();
        let gone = clears();
        assert_eq!(gone.len(), 1, "{gone:?}");
        assert_eq!((gone[0].width, gone[0].height), (0, 0));
        let shown = events
            .events()
            .into_iter()
            .find_map(|e| match e {
                AppEvent::ViewportFrame(f) if !f.url.is_empty() => Some(f),
                _ => None,
            })
            .expect("the frame");
        assert!(gone[0].seq > shown.seq, "the clear comes after the frame");
        // A new picture, and gone again.
        shared.present(&frame(8, 6), CorePhotoId(1));
        shared.clear_sink();
        assert_eq!(clears().len(), 2);
    }

    #[test]
    fn a_photo_that_finishes_just_before_the_worker_parks_is_still_drawn() {
        use crate::testing::{Harness, WAIT};
        let h = Harness::new();
        h.open(1);
        h.develop();
        let photo = CorePhotoId(h.first().0);
        let vp = h.api.viewport();
        // The cell has been reported (nothing is drawn before it is), and the photo drawn into it.
        h.rect(200.0, 150.0);
        h.frame_of_size(200, 150);
        assert!(vp.wait_idle(WAIT));
        let shared = vp.shared();
        let epoch = lock(&shared.state).epoch;
        let before = vp.frames_presented();
        // The worker found the photo still developing; by the time it parks the operations, the
        // event that would have woken it has gone by. Parking looks again.
        shared.park(
            h.api.core(),
            vec![Op::Rect(CoreRect::cell(64.0, 48.0, 1.0))],
            photo,
            epoch,
        );
        h.frame_of_size(64, 48);
        assert!(vp.frames_presented() > before);
        assert!(!lock(&shared.state).awaiting);
    }

    #[test]
    fn frame_addresses_use_the_scheme() {
        let url = frame_url(12);
        assert!(
            url.contains("ym-frame") && url.ends_with("/frame-12.png"),
            "{url}"
        );
    }
}
