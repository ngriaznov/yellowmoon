//! Fresh thumbnails after a look (or a crop, or a turn) changes.
//!
//! The session renders each photo's thumbnail once, when it has developed it (`thumbnail_ready`).
//! The filmstrip should follow the look the person applies (`docs/01` Preview strategy:
//! "rendered once after open and on look changes"), so [`Thumbnails::refresh`] queues a new one.
//! Requests are collected for a moment first, so applying a look to a hundred photos or dragging
//! the Intensity slider asks for one round, not a hundred, and the rendering runs at thumbnail
//! priority, behind whatever the viewport needs.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ym_core::app::Core;
use ym_core::pipeline::PhotoSource;
use ym_core::session::PhotoId as CorePhotoId;

use crate::events::{AppEvent, Emitter};
use crate::viewport::FrameStore;
use crate::wire::{PhotoId, ThumbnailReady};

/// How long requests are collected before the thumbnails are rendered.
const SETTLE: Duration = Duration::from_millis(300);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct Pending {
    ids: BTreeSet<u32>,
    due: Option<Instant>,
    /// A round is being rendered.
    busy: bool,
}

struct Shared {
    pending: Mutex<Pending>,
    wake: Condvar,
    idle: Condvar,
    emitter: Arc<dyn Emitter>,
    /// Where the pictures are kept for the webview to load (`ym-frame://…/thumb-…`).
    frames: Arc<FrameStore>,
    core: OnceLock<Weak<Core>>,
    shutdown: AtomicBool,
}

/// Re-renders thumbnails in the background (see the module documentation).
pub struct Thumbnails {
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Thumbnails {
    /// A refresher that keeps the thumbnails it renders in `frames` and announces their addresses
    /// through `emitter`. It does nothing until [`Thumbnails::attach`] gives it the core.
    #[must_use]
    pub fn new(emitter: Arc<dyn Emitter>, frames: Arc<FrameStore>) -> Self {
        Self {
            shared: Arc::new(Shared {
                pending: Mutex::new(Pending::default()),
                wake: Condvar::new(),
                idle: Condvar::new(),
                emitter,
                frames,
                core: OnceLock::new(),
                shutdown: AtomicBool::new(false),
            }),
            worker: Mutex::new(None),
        }
    }

    /// Gives the refresher the core and starts its thread.
    ///
    /// # Errors
    ///
    /// The error of starting the thread.
    pub fn attach(&self, core: &Arc<Core>) -> std::io::Result<()> {
        let _ = self.shared.core.set(Arc::downgrade(core));
        let shared = Arc::clone(&self.shared);
        let handle = std::thread::Builder::new()
            .name("ym-thumbnails".to_owned())
            .spawn(move || {
                while let Some(ids) = shared.next_round() {
                    shared.render(&ids);
                    lock(&shared.pending).busy = false;
                    shared.idle.notify_all();
                }
            })?;
        *lock(&self.worker) = Some(handle);
        Ok(())
    }

    /// Asks for new thumbnails of these photos.
    pub fn refresh(&self, ids: impl IntoIterator<Item = CorePhotoId>) {
        let mut pending = lock(&self.shared.pending);
        let before = pending.ids.len();
        pending.ids.extend(ids.into_iter().map(|id| id.0));
        if pending.ids.len() != before || pending.due.is_none() {
            pending.due = Some(Instant::now() + SETTLE);
        }
        self.shared.wake.notify_all();
    }

    /// Forgets the requests for these photos (they left the filmstrip).
    pub fn forget(&self, ids: &[CorePhotoId]) {
        let mut pending = lock(&self.shared.pending);
        for id in ids {
            pending.ids.remove(&id.0);
        }
    }

    /// Forgets what was asked for (the session closed).
    pub fn clear(&self) {
        let mut pending = lock(&self.shared.pending);
        pending.ids.clear();
        pending.due = None;
    }

    /// Waits until nothing is asked for and nothing renders; false if `timeout` ran out first.
    #[cfg(test)]
    #[must_use]
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut pending = lock(&self.shared.pending);
        while !pending.ids.is_empty() || pending.busy {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            pending = self
                .shared
                .idle
                .wait_timeout(pending, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }
}

impl Shared {
    /// Waits for requests to settle and returns them.
    fn next_round(&self) -> Option<Vec<u32>> {
        let mut pending = lock(&self.pending);
        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return None;
            }
            match pending.due {
                Some(due) if !pending.ids.is_empty() => {
                    let now = Instant::now();
                    if now >= due {
                        pending.due = None;
                        pending.busy = true;
                        return Some(std::mem::take(&mut pending.ids).into_iter().collect());
                    }
                    pending = self
                        .wake
                        .wait_timeout(pending, due - now)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
                _ => {
                    pending.due = None;
                    self.idle.notify_all();
                    pending = self
                        .wake
                        .wait(pending)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
    }

    fn render(&self, ids: &[u32]) {
        let Some(core) = self.core.get().and_then(Weak::upgrade) else {
            return;
        };
        for &id in ids {
            if self.shutdown.load(Ordering::SeqCst) {
                return;
            }
            // A photo that is not ready, was skipped or is gone has no thumbnail to refresh.
            let Ok(src) = PhotoSource::from_session(&core.session, CorePhotoId(id)) else {
                continue;
            };
            match core.renderer.spawn_thumbnail(src).wait() {
                Ok(Ok(png)) => {
                    // The session closed while this one was rendering: nobody wants the picture.
                    if core.session.view(CorePhotoId(id)).is_err() {
                        continue;
                    }
                    let url = self.frames.put_thumbnail(id, png);
                    self.emitter.emit(AppEvent::ThumbnailReady(ThumbnailReady {
                        id: PhotoId(id),
                        url,
                    }));
                }
                Ok(Err(e)) => tracing::debug!(photo = id, error = %e, "thumbnail not refreshed"),
                Err(e) => tracing::debug!(photo = id, error = %e, "thumbnail job did not run"),
            }
        }
    }
}

impl Drop for Thumbnails {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.wake.notify_all();
        if let Some(handle) = lock(&self.worker).take() {
            let _ = handle.join();
        }
    }
}
