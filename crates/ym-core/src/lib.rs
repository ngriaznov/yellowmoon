//! Yellowmoon core: the one place where photos are opened, developed and exported.
//!
//! `ym-core` owns the in-memory [`session`] (no catalogue, no sidecars), the `DevelopSettings`
//! that every UI control and CLI flag maps to, the per-photo automatic estimates, the render
//! graph with its stage caches, the job scheduler and the export queue. It depends on all other
//! `ym-*` library crates; only `src-tauri` and `ym-cli` depend on it.
//!
//! The modules: [`settings`] (`DevelopSettings`, its patch, `ResolvedSettings`), [`auto`] (the
//! per-photo automatic estimates), [`apple_auto`] (Apple Auto's advisor, steps and table,
//! `docs/19`), [`camera`] (camera match, STAND-IN and REDUCED, the camera
//! card), [`session`] (opening files, the decode pool, progress events, the mosaic cache and the
//! settings commands), [`pipeline`] (the render graph: `Renderer`, the stage caches, geometry, the
//! viewport state and frames, thumbnails, look tiles, histogram, Pick white sampling, full-size
//! renders), [`scheduler`] (priorities, cancellation, the memory budget), [`preferences`]
//! (`preferences.json`), [`looks`] (the look library with the default look, the display rendering
//! and `looks_changed`), [`export`] (the one export function the app and the CLI share),
//! [`export_queue`] (jobs, snapshots, progress, cancel, the updater's busy flag), [`cli_preview`]
//! (the `ym render …` command of a job) and [`app`] (all of them wired together, one method per IPC
//! command).
//!
//! Specs: `docs/01-architecture.md` (core concepts, scheduling, persistence, finding `data/`),
//! `docs/02-colour-pipeline.md` and `docs/05-raw-processing.md` (stage order), `docs/09-export.md`
//! (export queue).

#![warn(missing_docs)]

pub mod app;
pub mod apple_auto;
pub mod auto;
pub mod camera;
pub mod cli_preview;
pub mod data_dir;
pub mod export;
pub mod export_queue;
pub mod looks;
pub mod pipeline;
pub mod preferences;
pub mod scheduler;
pub mod session;
pub mod settings;
pub mod version;
