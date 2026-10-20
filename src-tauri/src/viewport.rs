//! The viewport: what draws the photo in the photo cell (`docs/01` §Native viewport, `docs/16`).
//!
//! The webview tells the shell where the cell is (`viewport_set_rect`) and what the pointer does
//! over it (`viewport_input`, `viewport_set_mode`, `set_before`). [`Viewport`] keeps the core's
//! [`ViewportState`] for the current photo, asks the core for a frame whenever the state, the
//! photo or its settings change, and hands the frame to a [`FrameSink`]:
//!
//! * the **native** sink (macOS, `crate::native`): a wgpu surface on a Metal layer behind the
//!   transparent webview, drawn with the core's own graphics device when there is one. The worker
//!   then asks for frames that stay on the device ([`FrameSink::takes_device_frames`],
//!   `FrameOutput::Device`), and the sink draws them from there: the frame never crosses to the
//!   processor, let alone the bridge.
//! * the **fallback** sink ([`FallbackSink`], everywhere): the frame is encoded as a PNG with the
//!   Display P3 profile, kept in a [`FrameStore`] and served to the webview at
//!   `ym-frame://localhost/<launch>/frame-<seq>.png`; the `viewport_frame` event says when a new one
//!   is there. The same store and scheme serve the filmstrip's thumbnails
//!   (`ym-frame://localhost/<launch>/thumb-<photo>-<version>.png`) and the Look tab's tiles
//!   (`ym-frame://localhost/<launch>/look-<photo>-<version>.png`), whichever sink draws the photo.
//!   `<launch>` is made from the start time of the app, so an address from an earlier launch is
//!   never one of this launch's (see [`FrameStore`]). This breaks the rule "full images never cross
//!   the bridge" (`docs/01`), and is meant for development builds on other systems and as the
//!   M0-06 fallback candidate (`docs/16`).
//!
//! # Threading
//!
//! Commands only record what the webview asked for (a list of [`Op`]s) and wake the worker thread.
//! The worker owns the rendering: it applies the recorded operations to the state (they need the
//! size of the final image), asks the renderer for the frame at viewport priority, presents it, and
//! asks again when the frame was made from a coarser image than the zoom needs
//! (`FrameOptions::coarse_only`: the tiles are computed at tile priority, then a sharp frame
//! follows).
//!
//! **Latest wins, but a frame in the making is shown.** A request marks the state dirty and wakes
//! the worker; it does not stop the frame being drawn (a [`Flight::Frame`]). That frame finishes
//! and is presented, and the next round starts at once from the newest state, so dragging a slider
//! or a photo never queues frames behind each other *and* the picture follows the hand while the
//! hand moves. (Every request used to cancel the render in flight, so once a frame took longer than
//! the gap between two inputs, which is a few milliseconds for a wheel and 16 for a slider, no
//! frame was ever finished until the hand stopped.) Only work nobody is waiting for gives way: the
//! pass that computes sharp tiles ([`Flight::Tiles`]), and, for a request that makes the frame
//! being drawn wrong rather than old (another photo, a cell of another size: [`Shared::request_now`]),
//! the frame itself.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ym_core::app::Core;
use ym_core::pipeline::{
    FrameOptions, FrameOutput, PhotoSource, RenderError, RenderedFrame, ViewportInput as CoreInput,
    ViewportMode as CoreMode, ViewportRect as CoreRect, ViewportState,
};
use ym_core::session::{PhotoId as CorePhotoId, SessionError};
use ym_gpu::queue::{CancelToken, JobError};

use crate::events::{AppEvent, Emitter};
use crate::perf;
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

    /// Shows a frame (RGBA8, Display P3, the size of the cell in device pixels): its host bytes, or
    /// the frame the core kept on the device ([`RenderedFrame::on_device`]) when the sink asked for
    /// those.
    ///
    /// # Errors
    ///
    /// A plain-words reason when the frame could not be shown.
    fn present(&self, frame: &RenderedFrame, seq: u32) -> Result<(), String>;

    /// Whether the sink draws frames that stay on the core's graphics device: the worker then asks
    /// for them (`FrameOutput::Device`), and a frame the device makes is never read back. Only the
    /// native viewport on the core's device does; asked before every frame, so a sink that stops
    /// sharing the device (it was lost) gets host frames from the next one on.
    fn takes_device_frames(&self) -> bool {
        false
    }

    /// Shows nothing (no photo, a photo that can't be drawn, the session closed). `seq` counts on
    /// from the last frame, so a listener that orders by it sees the picture go.
    fn clear(&self, seq: u32);
}

/// The last few frames of the fallback viewport, the latest thumbnail of every photo and the look
/// tiles of the Look tab, as PNG files, served to the webview by the `ym-frame` scheme.
///
/// **Addresses.** Every address starts with the store's launch id (`ym-frame://localhost/<launch>/…`),
/// made from the time the store was made, which is the time the app started. The version counters
/// of thumbnails and tiles start at 1 in every launch, so without it the first thumbnail of
/// photo 3 would have the same address in two launches, and a webview that kept the answer of the
/// first (the thumbnails and tiles are served as immutable, see [`Cache`]) would show it for the
/// second. A request whose launch id is not this store's is a 404.
#[derive(Debug)]
pub struct FrameStore {
    launch: String,
    frames: Mutex<VecDeque<(u32, Arc<Vec<u8>>)>>,
    thumbnails: Mutex<Thumbnails>,
    look_tiles: Mutex<LookTiles>,
}

impl Default for FrameStore {
    fn default() -> Self {
        Self::with_launch_id(new_launch_id())
    }
}

/// An id for this launch of the app: the time, in milliseconds since 1970, in hexadecimal, a
/// hyphen and how many stores this process made before (two stores of one process, as in the
/// tests, never share one). It is not a secret; it only has to differ from the ids of earlier
/// launches.
fn new_launch_id() -> String {
    static MADE: AtomicU32 = AtomicU32::new(0);
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    format!("{millis:x}-{:x}", MADE.fetch_add(1, Ordering::Relaxed))
}

/// How long the webview may keep an answer of the `ym-frame` scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cache {
    /// Not at all: a frame of the fallback viewport is replaced by the next one, and an address
    /// that answered 404 may answer 200 later (a tile that was not made yet).
    Never,
    /// For good: a thumbnail or a tile. Its address holds the launch id and a version that is
    /// never handed out twice, so the same address is always the same picture.
    Forever,
}

impl Cache {
    /// The `Cache-Control` header for the answer.
    #[must_use]
    pub fn header(self) -> &'static str {
        match self {
            Self::Never => "no-store",
            Self::Forever => "private, max-age=31536000, immutable",
        }
    }
}

/// What the `ym-frame` scheme answers for an address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Served {
    /// The HTTP status: 200, or 404 for an address that is not (or no longer) the store's.
    pub status: u16,
    /// The content type.
    pub mime: &'static str,
    /// The PNG, or a short plain text.
    pub body: Cow<'static, [u8]>,
    /// How long the webview may keep it.
    pub cache: Cache,
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
    /// The thumbnails of the photos the last removal took out of the filmstrip, kept for the Undo of that removal
    /// ([`FrameStore::set_aside_photos`]) and nowhere else: they are not served.
    set_aside: BTreeMap<u32, Vec<(u32, Png)>>,
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

    /// An empty store with this launch id (the tests' way to make two launches).
    #[must_use]
    pub fn with_launch_id(launch: String) -> Self {
        Self {
            launch,
            frames: Mutex::new(VecDeque::new()),
            thumbnails: Mutex::new(Thumbnails::default()),
            look_tiles: Mutex::new(LookTiles::default()),
        }
    }

    /// The id of this launch, the first segment of every address.
    #[cfg(test)]
    #[must_use]
    pub fn launch_id(&self) -> &str {
        &self.launch
    }

    /// The address of frame `seq` in the webview.
    #[must_use]
    pub fn frame_url(&self, seq: u32) -> String {
        scheme_url(&self.launch, &format!("frame-{seq}.png"))
    }

    /// The address of version `version` of a photo's thumbnail in the webview.
    #[must_use]
    pub fn thumbnail_url(&self, photo: u32, version: u32) -> String {
        scheme_url(&self.launch, &format!("thumb-{photo}-{version}.png"))
    }

    /// The address of version `version` of a look tile of a photo in the webview.
    #[must_use]
    pub fn look_tile_url(&self, photo: u32, version: u32) -> String {
        scheme_url(&self.launch, &format!("look-{photo}-{version}.png"))
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
        self.thumbnail_url(photo, version)
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
                    .map(|(version, _)| (*photo, self.thumbnail_url(*photo, *version)))
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
        self.look_tile_url(photo, version)
    }

    /// Forgets every thumbnail and look tile (the session closed). Addresses that were handed out
    /// answer 404 from now on; new ones count on from where the old ones stopped.
    pub fn clear_thumbnails(&self) {
        {
            let mut thumbs = lock(&self.thumbnails);
            thumbs.by_photo.clear();
            thumbs.set_aside.clear();
        }
        lock(&self.look_tiles).clear();
    }

    /// Photos were taken out of the filmstrip (Remove from filmstrip): their look tiles are forgotten, and their
    /// thumbnails are set aside instead of served, so that Undo can bring them back at once
    /// ([`FrameStore::bring_back_photos`]). What an earlier removal set aside is dropped: it can no longer be undone.
    /// Addresses handed out for these photos answer 404 meanwhile.
    pub fn set_aside_photos(&self, photos: &[u32]) {
        let mut thumbs = lock(&self.thumbnails);
        thumbs.set_aside.clear();
        let mut tiles = lock(&self.look_tiles);
        for photo in photos {
            if let Some(versions) = thumbs.by_photo.remove(photo) {
                thumbs.set_aside.insert(*photo, versions);
            }
            tiles.forget_photo(*photo);
        }
    }

    /// The removal was undone: the thumbnails set aside for these photos are served again, at the addresses they had.
    /// Returns the address of the newest thumbnail of each photo that had one, in photo order.
    pub fn bring_back_photos(&self, photos: &[u32]) -> Vec<(u32, String)> {
        let mut thumbs = lock(&self.thumbnails);
        let mut back = Vec::new();
        for photo in photos {
            if let Some(versions) = thumbs.set_aside.remove(photo) {
                if let Some((version, _)) = versions.last() {
                    back.push((*photo, self.thumbnail_url(*photo, *version)));
                }
                thumbs.by_photo.insert(*photo, versions);
            }
        }
        back.sort_by_key(|(photo, _)| *photo);
        back
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

    /// What the scheme answers for the path of a request,
    /// `/<launch>/frame-<seq>.png`, `/<launch>/thumb-<photo>-<version>.png` or
    /// `/<launch>/look-<photo>-<version>.png`: the file, or a 404. Only a thumbnail or a tile that
    /// is there is [`Cache::Forever`]; frames and every 404 are [`Cache::Never`].
    #[must_use]
    pub fn serve(&self, path: &str) -> Served {
        match self.find(path) {
            Some((png, cache)) => Served {
                status: 200,
                mime: "image/png",
                body: Cow::Owned(png.as_ref().clone()),
                cache,
            },
            None => Served {
                status: 404,
                mime: "text/plain",
                body: Cow::Borrowed(b"No such frame."),
                cache: Cache::Never,
            },
        }
    }

    /// [`FrameStore::serve`] as the tuple the tests compare.
    #[cfg(test)]
    #[must_use]
    pub fn respond(&self, path: &str) -> (u16, &'static str, Cow<'static, [u8]>) {
        let served = self.serve(path);
        (served.status, served.mime, served.body)
    }

    fn find(&self, path: &str) -> Option<(Png, Cache)> {
        let (launch, file) = path.trim_start_matches('/').split_once('/')?;
        if launch != self.launch {
            return None;
        }
        let name = file.strip_suffix(".png")?;
        if let Some(seq) = name.strip_prefix("frame-") {
            Some((self.get(seq.parse().ok()?)?, Cache::Never))
        } else if let Some(rest) = name.strip_prefix("thumb-") {
            let (photo, version) = rest.split_once('-')?;
            let png = self.thumbnail(photo.parse().ok()?, version.parse().ok()?)?;
            Some((png, Cache::Forever))
        } else if let Some(rest) = name.strip_prefix("look-") {
            let (photo, version) = rest.split_once('-')?;
            let png = self.look_tile(photo.parse().ok()?, version.parse().ok()?)?;
            Some((png, Cache::Forever))
        } else {
            None
        }
    }

    /// Holds the look tiles' lock until the value is dropped (a test of a handler that must not
    /// make the thread that called it wait for it).
    #[cfg(test)]
    pub(crate) fn hold_look_tiles(&self) -> impl Drop + '_ {
        lock(&self.look_tiles)
    }
}

/// The address of `file` of the launch `launch` in the webview (`http://ym-frame.localhost/…` on
/// Windows, where the webview serves a custom scheme as a host of `http`).
fn scheme_url(launch: &str, file: &str) -> String {
    if cfg!(windows) {
        format!("http://{FRAME_SCHEME}.localhost/{launch}/{file}")
    } else {
        format!("{FRAME_SCHEME}://localhost/{launch}/{file}")
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
            url: self.store.frame_url(seq),
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

/// What the render in flight is, which decides whether a request stops it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flight {
    /// A frame that is going to be shown. [`Shared::request`] lets it finish.
    Frame,
    /// The pass that makes the sharp tiles of a zoom: nobody waits for it, so [`Shared::request`]
    /// stops it.
    Tiles,
}

/// The cell as the webview last reported it: what a frame depends on (its place does not matter).
#[derive(Debug, Clone, Copy, PartialEq)]
struct CellSize {
    width: f64,
    height: f64,
    scale: f64,
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
    /// The render in flight and what it is, so that a request can stop it when it should. It is
    /// left in place when the render ends (cancelling a finished token does nothing).
    in_flight: Option<(CancelToken, Flight)>,
    /// A request that makes whatever is being drawn wrong was made ([`Shared::request_now`]) since
    /// the worker took its work: a round that has not begun drawing gives up. Cleared when the
    /// worker takes work.
    urgent: bool,
    /// The size of the cell at the last `viewport_set_rect`, to tell a resize from a move.
    cell: Option<CellSize>,
    /// When the oldest request that no round has taken yet was made (for the input-to-present time
    /// of `perf`).
    oldest_request: Option<Instant>,
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
    /// A new frame is wanted: an input, a settings patch, a look, a cell that moved or kept its
    /// size, Before, a mode. The frame being drawn finishes and is shown and the next round starts
    /// at once from the newest state; only the pass that makes sharp tiles is stopped, because the
    /// round that follows it makes a coarse frame first.
    fn request(&self, st: &mut VState) {
        self.mark_dirty(st);
        if let Some((token, Flight::Tiles)) = &st.in_flight {
            token.cancel();
        }
        self.wake.notify_all();
    }

    /// A new frame is wanted and the one being drawn is wrong, not just old: another photo (or
    /// none), a cell of another size. It is stopped, and so is a round that has not begun drawing.
    fn request_now(&self, st: &mut VState) {
        self.mark_dirty(st);
        st.urgent = true;
        if let Some((token, _)) = &st.in_flight {
            token.cancel();
        }
        self.wake.notify_all();
    }

    fn mark_dirty(&self, st: &mut VState) {
        st.dirty = true;
        st.want_sharp = false;
        st.oldest_request.get_or_insert_with(Instant::now);
        perf::viewport().request();
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
        tracing::warn!(?scope, photo = ?photo, %message, "the viewport reports a problem");
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
                st.urgent = false;
                st.busy = true;
                st.applying = !st.ops.is_empty();
                let ops = std::mem::take(&mut st.ops);
                let coarse_only = !std::mem::take(&mut st.want_sharp);
                return Some(Work {
                    photo: st.photo,
                    epoch: st.epoch,
                    ops,
                    coarse_only,
                    requested_at: st.oldest_request.take(),
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

    fn present(&self, frame: &RenderedFrame, photo: CorePhotoId, requested_at: Option<Instant>) {
        let seq = {
            let mut st = lock(&self.state);
            st.seq = st.seq.wrapping_add(1);
            st.cleared = false;
            st.seq
        };
        match self.sink.present(frame, seq) {
            Ok(()) => {
                lock(&self.state).frames += 1;
                perf::viewport().present(requested_at.map(|at| at.elapsed()));
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
            requested_at,
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
        perf::viewport().round();
        let cancel = CancelToken::new();
        {
            let mut st = lock(&self.state);
            st.in_flight = Some((cancel.clone(), Flight::Frame));
            if st.urgent {
                // What this round would draw is already wrong (another photo, a cell of another
                // size): do not start it. (A request that only makes it old does not stop it.)
                cancel.cancel();
            }
        }
        let image = match core.renderer.final_size(&src, &cancel) {
            Ok(image) => image,
            Err(RenderError::Cancelled) => {
                perf::viewport().cancelled();
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
        let output = if self.sink.takes_device_frames() {
            FrameOutput::Device
        } else {
            FrameOutput::Host
        };
        let handle =
            core.renderer
                .spawn_viewport_frame_to(src.clone(), vp.clone(), options, output);
        {
            let mut st = lock(&self.state);
            st.in_flight = Some((handle.token().clone(), Flight::Frame));
            if st.urgent {
                handle.cancel();
            }
        }
        let frame = match handle.wait() {
            Ok(Ok(frame)) => frame,
            Ok(Err(RenderError::Cancelled)) | Err(JobError::Cancelled | JobError::Shutdown) => {
                perf::viewport().cancelled();
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
        self.present(&frame, photo, requested_at);
        if !frame.sharp {
            // The zoom needs full-resolution tiles that were not ready: compute them at tile
            // priority, then draw the sharp frame. Nobody waits for this pass, so a request that
            // is already there (or arrives) stops it: the round that answers it draws a coarse
            // frame first.
            let tiles = core.renderer.spawn_tiles(src, vp);
            {
                let mut st = lock(&self.state);
                st.in_flight = Some((tiles.token().clone(), Flight::Tiles));
                if st.dirty || !st.ops.is_empty() {
                    tiles.cancel();
                }
            }
            match tiles.wait() {
                Ok(Ok(_)) => {
                    let mut st = lock(&self.state);
                    if !st.dirty && st.ops.is_empty() {
                        st.dirty = true;
                        st.want_sharp = true;
                    }
                }
                Ok(Err(RenderError::Cancelled)) | Err(JobError::Cancelled | JobError::Shutdown) => {
                    perf::viewport().tiles_cancelled();
                }
                _ => {}
            }
        }
    }
}

struct Work {
    photo: Option<CorePhotoId>,
    epoch: u64,
    ops: Vec<Op>,
    coarse_only: bool,
    /// When the oldest request this round answers was made, if one did (a sharp follow-up has none).
    requested_at: Option<Instant>,
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
            urgent: false,
            cell: None,
            oldest_request: None,
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
        // What is being drawn is the old photo: stop it.
        self.shared.request_now(&mut st);
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
            if !st.has_rect {
                tracing::info!(
                    width = rect.width.0,
                    height = rect.height.0,
                    scale = rect.scale.0,
                    "the webview placed the photo cell"
                );
            }
            st.has_rect = true;
        }
        st.ops.push(Op::Rect(core_rect));
        let size = CellSize {
            width: rect.width.0,
            height: rect.height.0,
            scale: rect.scale.0,
        };
        if st.cell.replace(size) == Some(size) {
            // The cell moved (the layout shifted) and kept its size: the frame being drawn is
            // still the right one.
            self.shared.request(&mut st);
        } else {
            // A frame of the old size would be shown, stretched, in the new cell.
            self.shared.request_now(&mut st);
        }
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
        if let Some((token, _)) = &lock(&self.shared.state).in_flight {
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

    /// The path of an address of the scheme: `/<launch>/<file>`.
    pub(crate) fn path_of_url(url: &str) -> &str {
        assert!(url.contains("ym-frame"), "{url}");
        let host = url.find("localhost/").expect("a localhost address");
        &url[host + "localhost".len()..]
    }

    fn frame(width: usize, height: usize) -> RenderedFrame {
        RenderedFrame {
            width,
            height,
            rgba: (0..width * height)
                .flat_map(|i| [(i % 251) as u8, (i % 13) as u8 * 19, 7, 255])
                .collect(),
            on_device: None,
            sharp: true,
            report: ym_core::pipeline::RenderReport {
                ms: 1.0,
                path: ym_core::pipeline::RenderPath::Classical,
                space: ym_export::ColorSpace::DisplayP3,
                device: ym_core::pipeline::ComputeDevice::Cpu,
                ranges: Vec::new(),
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

        let at = |file: &str| format!("/{}/{file}", store.launch_id());
        let (status, mime, body) = store.respond(&at("frame-4.png"));
        assert_eq!((status, mime), (200, "image/png"));
        assert_eq!(&*body, &[4u8; 4]);
        assert_eq!(store.respond(&at("frame-1.png")).0, 404);
        assert_eq!(store.respond("/etc/passwd").0, 404);
        assert_eq!(store.respond(&at("frame-x.png")).0, 404);
        // The address a frame is announced at is the one that serves it.
        assert_eq!(path_of_url(&store.frame_url(4)), at("frame-4.png"));
        store.clear();
        assert!(store.latest().is_none());
    }

    #[test]
    fn look_tiles_are_kept_per_photo_and_look_and_served_by_address() {
        let store = FrameStore::new();
        let path = |url: &str| path_of_url(url).to_owned();
        let status = |url: &str| store.respond(&path(url)).0;
        let at = |file: &str| format!("/{}/{file}", store.launch_id());
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
        assert_eq!(store.respond(&at("thumb-3-4.png")).0, 404);
        assert_eq!(store.respond(&at("frame-4.png")).0, 404);
        assert_eq!(store.respond(&at("look-4-5.png")).0, 404);
        for bad in ["look-3.png", "look-x-1.png", "look-3-x.png", "look--.png"] {
            assert_eq!(store.respond(&at(bad)).0, 404, "{bad}");
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
    fn the_thumbnails_of_removed_photos_are_set_aside_for_undo_and_the_next_removal_drops_them() {
        let store = FrameStore::new();
        let path = |url: &str| path_of_url(url).to_owned();
        let status = |url: &str| store.respond(&path(url)).0;
        let one = store.put_thumbnail(1, vec![1]);
        let two = store.put_thumbnail(2, vec![2]);
        let three = store.put_thumbnail(3, vec![3]);
        let tile = store.put_look_tile(2, "builtin:neutral", vec![9]);
        // Photos 1 and 2 leave: their pictures are not served, the look tile is gone, photo 3 is as it was.
        store.set_aside_photos(&[1, 2]);
        assert_eq!((status(&one), status(&two), status(&tile)), (404, 404, 404));
        assert_eq!(status(&three), 200);
        assert_eq!(store.thumbnail_urls(), vec![(3, three.clone())]);
        // Undo: the same addresses serve the same pictures again, and the answer lists them in photo order.
        assert_eq!(
            store.bring_back_photos(&[2, 1, 7]),
            vec![(1, one.clone()), (2, two.clone())]
        );
        assert_eq!((status(&one), status(&two)), (200, 200));
        assert_eq!(&*store.respond(&path(&two)).2, &[2]);
        // Nothing is left to bring back, and a second removal replaces what the first set aside.
        assert!(store.bring_back_photos(&[1, 2]).is_empty());
        store.set_aside_photos(&[1]);
        store.set_aside_photos(&[3]);
        assert!(
            store.bring_back_photos(&[1]).is_empty(),
            "the first removal can no longer be undone"
        );
        assert_eq!(store.bring_back_photos(&[3]), vec![(3, three)]);
        // The end of the session drops what was set aside too.
        store.set_aside_photos(&[2]);
        store.clear_thumbnails();
        assert!(store.bring_back_photos(&[2]).is_empty());
        assert_eq!(status(&one), 404);
    }

    #[test]
    fn look_tiles_are_bounded_by_bytes_and_the_photos_least_recently_used_go_first() {
        // A session of hundreds of photos with the Look tab open would otherwise keep every tile
        // of every photo it ever showed until the session closed.
        let store = FrameStore::with_look_tile_budget(600);
        let path = |url: &str| path_of_url(url).to_owned();
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
        let path = |url: &str| path_of_url(url).to_owned();
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
        shared.present(&frame(8, 6), CorePhotoId(1), None);
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
        shared.present(&frame(8, 6), CorePhotoId(1), None);
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
    fn frame_addresses_use_the_scheme_and_the_launch_id() {
        let store = FrameStore::with_launch_id("18f3a".to_owned());
        let url = store.frame_url(12);
        assert!(
            url.contains("ym-frame") && url.ends_with("/18f3a/frame-12.png"),
            "{url}"
        );
        assert!(store.thumbnail_url(3, 4).ends_with("/18f3a/thumb-3-4.png"));
        assert!(store.look_tile_url(3, 4).ends_with("/18f3a/look-3-4.png"));
        assert_eq!(path_of_url(&url), "/18f3a/frame-12.png");
    }

    #[test]
    fn every_store_has_a_launch_id_of_its_own() {
        let ids: Vec<String> = (0..50)
            .map(|_| FrameStore::new().launch_id().to_owned())
            .collect();
        for id in &ids {
            assert!(!id.is_empty() && !id.contains('/'), "{id:?}");
        }
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
        // The start time comes first: hexadecimal milliseconds since 1970 (after November 2023).
        let (time, _) = ids[0].split_once('-').expect("a time and a count");
        let millis = u128::from_str_radix(time, 16).expect("hex time");
        assert!(millis > 1_700_000_000_000, "{}", ids[0]);
    }

    #[test]
    fn an_address_of_an_earlier_launch_is_not_served_and_the_same_picture_is_not_the_same_address()
    {
        // Version counters start at 1 in every launch: photo 3's first thumbnail is
        // `thumb-3-1.png` in both. The launch id keeps a webview that cached the first from being
        // handed it for the second.
        let before = FrameStore::with_launch_id("aaa".to_owned());
        let after = FrameStore::with_launch_id("bbb".to_owned());
        let old = before.put_thumbnail(3, vec![1]);
        let new = after.put_thumbnail(3, vec![2]);
        let old_look = before.put_look_tile(3, "builtin:neutral", vec![1]);
        assert!(old.ends_with("thumb-3-1.png") && new.ends_with("thumb-3-1.png"));
        assert_ne!(old, new);
        assert_ne!(path_of_url(&old), path_of_url(&new));

        assert_eq!(after.respond(path_of_url(&new)).0, 200);
        for (name, url) in [("thumbnail", &old), ("look tile", &old_look)] {
            let served = after.serve(path_of_url(url));
            assert_eq!(
                served.status, 404,
                "{name}: an address of the launch before"
            );
            assert_eq!(served.cache, Cache::Never, "{name}");
        }
        // A path with no launch id, or with the id in the wrong place, is not an address either.
        for bad in [
            "/thumb-3-1.png",
            "thumb-3-1.png",
            "/bbb",
            "/bbb/",
            "//thumb-3-1.png",
            "/bbb/x/thumb-3-1.png",
        ] {
            assert_eq!(after.serve(bad).status, 404, "{bad}");
        }
    }

    #[test]
    fn only_thumbnails_and_tiles_that_exist_are_cached_for_good() {
        let store = FrameStore::new();
        store.put(7, vec![7]);
        let thumb = store.put_thumbnail(1, vec![1]);
        let tile = store.put_look_tile(1, "builtin:neutral", vec![2]);
        let at = |file: &str| format!("/{}/{file}", store.launch_id());

        for url in [&thumb, &tile] {
            let served = store.serve(path_of_url(url));
            assert_eq!(
                (served.status, served.cache),
                (200, Cache::Forever),
                "{url}"
            );
            assert_eq!(
                served.cache.header(),
                "private, max-age=31536000, immutable"
            );
        }
        // A frame is replaced by the next one; a 404 may become a 200 (a tile not made yet).
        let frame = store.serve(&at("frame-7.png"));
        assert_eq!((frame.status, frame.cache), (200, Cache::Never));
        assert_eq!(frame.cache.header(), "no-store");
        for gone in [
            "frame-8.png",
            "thumb-1-9.png",
            "thumb-2-1.png",
            "look-1-9.png",
            "look-x.png",
        ] {
            let served = store.serve(&at(gone));
            assert_eq!((served.status, served.cache), (404, Cache::Never), "{gone}");
        }
        // The session closed: the addresses answer 404 now, and not for good.
        store.clear_thumbnails();
        let served = store.serve(path_of_url(&thumb));
        assert_eq!((served.status, served.cache), (404, Cache::Never));
    }

    // ----- Which render a request stops (`Flight`) -------------------------------------------

    /// Puts a render of this kind in flight, as the worker does, and returns its token.
    fn fly(shared: &Shared, flight: Flight) -> CancelToken {
        let token = CancelToken::new();
        lock(&shared.state).in_flight = Some((token.clone(), flight));
        token
    }

    #[test]
    fn a_request_lets_the_frame_being_drawn_finish_and_stops_the_pass_that_makes_sharp_tiles() {
        let (vp, _events) = controller();
        let shared = vp.shared();
        vp.set_photo(Some(CorePhotoId(1)));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        let pan = || {
            vp.input(&ViewportInput::Pan {
                dx: crate::wire::Num(3.0),
                dy: crate::wire::Num(0.0),
            })
            .expect("input");
        };
        type Request<'a> = (&'static str, Box<dyn Fn() + 'a>);
        let requests: Vec<Request<'_>> = vec![
            ("an input", Box::new(pan)),
            ("a settings patch", Box::new(|| vp.invalidate())),
            (
                "a mode",
                Box::new(|| vp.set_mode(&ViewportMode::Fit).expect("mode")),
            ),
            ("Before", Box::new(|| vp.set_before(true))),
            // The cell moved and kept its size.
            (
                "a cell that only moved",
                Box::new(|| {
                    let mut moved = cell(100.0, 80.0);
                    moved.x = crate::wire::Num(40.0);
                    vp.set_rect(&moved).expect("rect");
                }),
            ),
            (
                "estimates",
                Box::new(|| shared.estimates_changed(CorePhotoId(1))),
            ),
        ];
        for (what, request) in &requests {
            let frame = fly(&shared, Flight::Frame);
            request();
            assert!(
                !frame.is_cancelled(),
                "{what} stopped the frame being drawn"
            );
            assert!(
                lock(&shared.state).dirty,
                "{what} did not ask for a new frame"
            );

            let tiles = fly(&shared, Flight::Tiles);
            request();
            assert!(
                tiles.is_cancelled(),
                "{what} left the sharp-tile pass running"
            );
        }
        // The pass that was asked for is not asked for again until the new round has run.
        assert!(!lock(&shared.state).want_sharp);
    }

    #[test]
    fn a_request_that_makes_the_frame_wrong_stops_it() {
        let (vp, _events) = controller();
        let shared = vp.shared();
        vp.set_photo(Some(CorePhotoId(1)));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");

        // Another photo.
        for flight in [Flight::Frame, Flight::Tiles] {
            let token = fly(&shared, flight);
            vp.set_photo(Some(CorePhotoId(2)));
            assert!(token.is_cancelled(), "{flight:?} of the photo before");
            vp.set_photo(Some(CorePhotoId(1)));
        }
        // No photo (the session closed).
        let token = fly(&shared, Flight::Frame);
        vp.set_photo(None);
        assert!(token.is_cancelled(), "the session closed");
        vp.set_photo(Some(CorePhotoId(1)));

        // A cell of another size, in either side or the scale; the same size again does not.
        for resized in [cell(120.0, 80.0), cell(120.0, 90.0)] {
            let token = fly(&shared, Flight::Frame);
            vp.set_rect(&resized).expect("rect");
            assert!(token.is_cancelled(), "a frame of the old size is no use");
        }
        let token = fly(&shared, Flight::Frame);
        vp.set_rect(&cell(120.0, 90.0)).expect("rect");
        assert!(!token.is_cancelled(), "the same size again");
        let mut sharper = cell(120.0, 90.0);
        sharper.scale = crate::wire::Num(2.0);
        vp.set_rect(&sharper).expect("rect");
        assert!(token.is_cancelled(), "the same size at another scale");
    }

    #[test]
    fn a_round_taken_before_a_request_that_makes_its_frame_wrong_gives_up_and_a_later_one_does_not()
    {
        let (vp, _events) = controller();
        let shared = vp.shared();
        vp.set_photo(Some(CorePhotoId(1)));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        assert!(lock(&shared.state).urgent);
        // The worker takes the work (a round for photo 1), and then the photo changes before the
        // round has registered what it draws: it must not draw photo 1.
        let first = shared.next_work().expect("round");
        assert_eq!(first.photo, Some(CorePhotoId(1)));
        assert!(
            !lock(&shared.state).urgent,
            "taking work answers what was asked"
        );
        vp.set_photo(Some(CorePhotoId(2)));
        assert!(
            lock(&shared.state).urgent,
            "the round for photo 1 is told to give up"
        );
        // The next round is for photo 2 and starts with the flag down.
        let second = shared.next_work().expect("round");
        assert_eq!(second.photo, Some(CorePhotoId(2)));
        assert!(!lock(&shared.state).urgent);
        // An ordinary request does not raise it.
        vp.invalidate();
        assert!(!lock(&shared.state).urgent);
    }

    #[test]
    fn the_time_from_a_request_to_the_frame_is_kept_for_the_round_that_answers_it() {
        let (vp, _events) = controller();
        let shared = vp.shared();
        vp.set_photo(Some(CorePhotoId(1)));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        std::thread::sleep(Duration::from_millis(5));
        let first = shared.next_work().expect("round");
        let waited = first
            .requested_at
            .expect("a request is behind the round")
            .elapsed();
        assert!(waited >= Duration::from_millis(5), "{waited:?}");
        // Requests made while that round runs belong to the next one, and the oldest counts.
        vp.invalidate();
        std::thread::sleep(Duration::from_millis(2));
        vp.invalidate();
        let second = shared.next_work().expect("round");
        let at = second.requested_at.expect("requests are behind the round");
        assert!(
            at > first.requested_at.expect("first") && at.elapsed() >= Duration::from_millis(2)
        );
        // A round that nobody asked for (the sharp follow-up) has no time to report.
        {
            let mut st = lock(&shared.state);
            st.dirty = true;
            st.want_sharp = true;
        }
        assert!(shared.next_work().expect("round").requested_at.is_none());
    }

    #[test]
    fn a_reader_waits_for_what_is_pending_no_longer_than_its_cap() {
        // No worker: the rectangle is recorded and never applied, as when a round never ends.
        let (vp, _events) = controller();
        vp.set_photo(Some(CorePhotoId(1)));
        vp.set_rect(&cell(100.0, 80.0)).expect("rect");
        let started = Instant::now();
        let (photo, _) = vp.snapshot_synced();
        let waited = started.elapsed();
        assert_eq!(photo, Some(CorePhotoId(1)));
        assert!(
            waited >= SYNC_TIMEOUT - Duration::from_millis(50),
            "{waited:?}"
        );
        assert!(waited < SYNC_TIMEOUT + Duration::from_secs(1), "{waited:?}");
    }

    #[test]
    fn frames_reach_the_screen_while_the_settings_change_faster_than_a_frame_takes() {
        use crate::testing::{Harness, WAIT};
        let h = Harness::new();
        h.open(1);
        h.develop();
        let vp = h.api.viewport();
        h.rect(200.0, 150.0);
        h.frame_of_size(200, 150);
        assert!(vp.wait_idle(WAIT));
        let before = vp.frames_presented();

        // A slider patch arrives every 2 ms, which is faster than a frame of the CPU renderer.
        // Every request used to stop the frame being drawn, so none was finished until the patches
        // stopped. It goes on past 200 until a second frame is out, so that a slow machine that
        // needs more than 400 ms for a frame is not mistaken for starvation.
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut calls = 0;
        while (calls < 200 || vp.frames_presented() < before + 2) && Instant::now() < deadline {
            vp.invalidate();
            calls += 1;
            std::thread::sleep(Duration::from_millis(2));
        }
        let shown = vp.frames_presented() - before;
        assert!(
            shown > 1,
            "{shown} frames presented while {calls} requests arrived 2 ms apart"
        );

        // And the picture of the last request is the one that stays.
        assert!(vp.wait_idle(WAIT));
        let settled = vp.frames_presented();
        vp.invalidate();
        assert!(vp.wait_idle(WAIT));
        assert!(vp.frames_presented() > settled);
    }

    #[test]
    fn a_reader_asking_during_a_stream_of_input_is_answered_within_its_cap() {
        use crate::testing::{Harness, WAIT};
        let h = Harness::new();
        h.open(1);
        h.develop();
        let vp = h.api.viewport();
        // A cell smaller than the photo, so that 100 % is a real zoom and there is something to pan.
        h.rect(100.0, 100.0);
        h.frame_of_size(100, 100);
        assert!(vp.wait_idle(WAIT));
        h.api
            .viewport_input(&ViewportInput::ToggleZoom {
                x: crate::wire::Num(50.0),
                y: crate::wire::Num(50.0),
            })
            .expect("zoom in");
        assert!(vp.wait_idle(WAIT));

        // Pans arrive as fast as a hand makes them while a readout (a click for Pick white, the
        // Option key's pixel) asks where the view is. It gets the state after the next round
        // applied what was pending, which is at most one frame away, and never waits past its cap.
        let api = Arc::clone(&h.api);
        let stop = Arc::new(AtomicBool::new(false));
        let feeder = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut i = 0_u32;
                while !stop.load(Ordering::SeqCst) {
                    let dx = if i.is_multiple_of(2) { 2.0 } else { -2.0 };
                    api.viewport_input(&ViewportInput::Pan {
                        dx: crate::wire::Num(dx),
                        dy: crate::wire::Num(0.0),
                    })
                    .expect("pan");
                    i += 1;
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };
        let mut slowest = Duration::ZERO;
        for _ in 0..10 {
            let asked = Instant::now();
            let (photo, state) = vp.snapshot_synced();
            slowest = slowest.max(asked.elapsed());
            assert!(photo.is_some());
            assert!(matches!(state.mode, CoreMode::Percent { .. }));
            std::thread::sleep(Duration::from_millis(20));
        }
        stop.store(true, Ordering::SeqCst);
        feeder.join().expect("the feeder");
        assert!(
            slowest < SYNC_TIMEOUT + Duration::from_secs(1),
            "a reader waited {slowest:?} (the cap is {SYNC_TIMEOUT:?})"
        );
        assert!(vp.wait_idle(WAIT));
    }
}
