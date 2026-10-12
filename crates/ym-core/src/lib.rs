//! Yellowmoon core: the one place where photos are opened, developed and exported.
//!
//! `ym-core` owns the in-memory [`session`] (no catalogue, no sidecars), the `DevelopSettings`
//! that every UI control and CLI flag maps to, the per-photo automatic estimates, the render
//! graph with its stage caches, the job scheduler and the export queue. It depends on all other
//! `ym-*` library crates; only `src-tauri` and `ym-cli` depend on it.
//!
//! Built so far: [`settings`] (`DevelopSettings`, its patch, `ResolvedSettings`), [`auto`] (the
//! per-photo automatic estimates), [`camera`] (camera match, STAND-IN and REDUCED, the camera
//! card), [`session`] (opening files, the decode pool, progress events, the mosaic cache and the
//! settings commands), [`pipeline`] (the render graph: `Renderer`, the stage caches, geometry, the
//! viewport state and frames, thumbnails, look tiles, histogram, Pick white sampling, full-size
//! renders) and [`scheduler`] (priorities, cancellation, the memory budget). The export queue and
//! preferences are the next modules.
//!
//! Specs: `docs/01-architecture.md` (core concepts, scheduling, persistence, finding `data/`),
//! `docs/02-colour-pipeline.md` and `docs/05-raw-processing.md` (stage order), `docs/09-export.md`
//! (export queue).

#![warn(missing_docs)]

pub mod auto;
pub mod camera;
pub mod data_dir;
pub mod export_queue;
pub mod pipeline;
pub mod preferences;
pub mod scheduler;
pub mod session;
pub mod settings;
pub mod version;
