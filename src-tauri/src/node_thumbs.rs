//! Node thumbnails for the node view (`docs/23` sections 5.7 and 9).
//!
//! `node_thumbnails(photo_id)` answers at once and queues the photo here. A thread of its own asks the
//! core for the thumbnails of every node of the photo ([`ym_core::pipeline::Renderer::spawn_node_thumbnails`]:
//! at thumbnail priority, behind whatever the viewport needs, and cancelled when the person moves to
//! another photo), keeps the pictures in the [`FrameStore`] and announces the ones that changed with
//! `node_thumbnails_ready`.
//!
//! The core keeps the pictures in memory under a key that holds the plan up to the node, so a photo
//! asked about again after a change late in its graph makes pictures only for the nodes the change
//! reaches. What the webview has been told is what the store holds: a picture whose version the store
//! has is not announced again, one it dropped (the store is bounded, and a removed photo's pictures
//! go) is announced again at the same address. Requests for a photo that is being rendered are
//! collected into one more round after it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;

use ym_core::app::Core;
use ym_core::pipeline::PhotoSource;
use ym_core::session::PhotoId as CorePhotoId;

use crate::events::{AppEvent, Emitter};
use crate::viewport::FrameStore;
use crate::wire::{NodeThumbnail, NodeThumbnailsReady, PhotoId};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct Pending {
    /// The photos asked about, oldest request first, each once.
    photos: VecDeque<u32>,
    /// A round is being rendered.
    busy: bool,
}

struct Shared {
    pending: Mutex<Pending>,
    wake: Condvar,
    idle: Condvar,
    emitter: Arc<dyn Emitter>,
    frames: Arc<FrameStore>,
    core: OnceLock<Weak<Core>>,
    shutdown: AtomicBool,
}

/// Makes the node thumbnails of the photos the webview asks about (see the module documentation).
pub struct NodeThumbnailer {
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl NodeThumbnailer {
    /// A maker that keeps the pictures it gets in `frames` and announces their addresses through
    /// `emitter`. It does nothing until [`NodeThumbnailer::attach`] gives it the core.
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

    /// Gives the maker the core and starts its thread.
    ///
    /// # Errors
    ///
    /// The error of starting the thread.
    pub fn attach(&self, core: &Arc<Core>) -> std::io::Result<()> {
        let _ = self.shared.core.set(Arc::downgrade(core));
        let shared = Arc::clone(&self.shared);
        let handle = std::thread::Builder::new()
            .name("ym-node-thumbnails".to_owned())
            .spawn(move || {
                while let Some(photo) = shared.next() {
                    shared.render(photo);
                    lock(&shared.pending).busy = false;
                    shared.idle.notify_all();
                }
            })?;
        *lock(&self.worker) = Some(handle);
        Ok(())
    }

    /// Asks for the thumbnails of a photo's nodes. A photo already waiting is not queued twice.
    pub fn request(&self, photo: CorePhotoId) {
        let mut pending = lock(&self.shared.pending);
        if !pending.photos.contains(&photo.0) {
            pending.photos.push_back(photo.0);
        }
        self.shared.wake.notify_all();
    }

    /// Forgets the requests for these photos (they left the filmstrip).
    pub fn forget(&self, photos: &[CorePhotoId]) {
        lock(&self.shared.pending)
            .photos
            .retain(|p| !photos.iter().any(|gone| gone.0 == *p));
    }

    /// Forgets what was asked for (the session closed).
    pub fn clear(&self) {
        lock(&self.shared.pending).photos.clear();
    }

    /// Waits until nothing is asked for and nothing renders; false if `timeout` ran out first.
    #[cfg(test)]
    #[must_use]
    pub fn wait_idle(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut pending = lock(&self.shared.pending);
        while !pending.photos.is_empty() || pending.busy {
            let now = std::time::Instant::now();
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
    /// Waits for a request and takes it.
    fn next(&self) -> Option<u32> {
        let mut pending = lock(&self.pending);
        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return None;
            }
            if let Some(photo) = pending.photos.pop_front() {
                pending.busy = true;
                return Some(photo);
            }
            self.idle.notify_all();
            pending = self
                .wake
                .wait(pending)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn render(&self, id: u32) {
        let Some(core) = self.core.get().and_then(Weak::upgrade) else {
            return;
        };
        // A photo that is not ready, was skipped or is gone has no thumbnails to make.
        let Ok(src) = PhotoSource::from_session(&core.session, CorePhotoId(id)) else {
            return;
        };
        let made = match core.renderer.spawn_node_thumbnails(src).wait() {
            Ok(Ok(made)) => made,
            // The person moved on to another photo (the job was cancelled), or the photo cannot be
            // drawn: the webview asks again when it shows the photo.
            Ok(Err(e)) => {
                tracing::debug!(photo = id, error = %e, "node thumbnails not made");
                return;
            }
            Err(e) => {
                tracing::debug!(photo = id, error = %e, "node thumbnails job did not run");
                return;
            }
        };
        // The session closed while they rendered: nobody wants the pictures, and the store was just
        // emptied.
        if core.session.view(CorePhotoId(id)).is_err() || self.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let mut changed = Vec::new();
        for thumbnail in made {
            if self
                .frames
                .has_node_thumbnail(id, thumbnail.node, thumbnail.version)
            {
                continue;
            }
            let url = self.frames.put_node_thumbnail(
                id,
                thumbnail.node,
                thumbnail.version,
                thumbnail.png,
            );
            changed.push(NodeThumbnail {
                node: thumbnail.node,
                url,
            });
        }
        if !changed.is_empty() {
            self.emitter
                .emit(AppEvent::NodeThumbnailsReady(NodeThumbnailsReady {
                    photo_id: PhotoId(id),
                    thumbnails: changed,
                }));
        }
    }
}

impl Drop for NodeThumbnailer {
    fn drop(&mut self) {
        // Under the lock the worker checks the flag with, so that it cannot miss the wake-up.
        {
            let _pending = lock(&self.shared.pending);
            self.shared.shutdown.store(true, Ordering::SeqCst);
        }
        self.shared.wake.notify_all();
        if let Some(handle) = lock(&self.worker).take() {
            let _ = handle.join();
        }
    }
}
