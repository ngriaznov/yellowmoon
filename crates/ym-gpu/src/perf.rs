//! Where the host time of a device job goes: the spans behind the `ym::perf` frame lines.
//!
//! Off by default: with the switch off, [`span`] is one relaxed atomic load and returns `None`, no clock
//! is read. The application turns it on with [`set_enabled`] when `YM_PERF` asks for it (`src-tauri`
//! `perf.rs`, `docs/11-ui.md` Measuring). A span adds its time to the **calling thread's** [`Split`]; the
//! thread that runs a job ([`begin`], [`take`]) and the thread that renders it (the render thread, which
//! [`Split::merge`]s the device thread's split back in) each have their own, so frames that run at the
//! same time do not mix their numbers.
//!
//! The spans measure host wall-clock time. The commands of a device job are recorded on the host and run
//! later, in one submission, so the GPU's own time is in [`Stage::Wait`] (and, per kernel, in the pass
//! times of the timestamp queries when the device has them: [`crate::device::Gpu::take_pass_times`]).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);

/// Turns the spans on or off for the whole process.
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether the spans are on.
#[must_use]
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// What a span measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Preparing a view for a pass on the render thread: the raw group's window, the grid
    /// (`prepare_view`), for the picture and for the wider picture of the raster masks.
    Prepare,
    /// Making the device input of a pass on the render thread (a host window copied out of the raw
    /// group; zero for a resident raw group).
    DeviceInput,
    /// Recording the sampler of the picture the colour pass reads (and its upload when the raw group
    /// is on the host), on the device thread.
    InputImage,
    /// The same for the wider picture the raster masks look at.
    WideImage,
    /// A raster row's photo-frame plane `S0`: made or fetched from the process cache, on the CPU.
    Source,
    /// Putting an `S0` plane on the device (zero when the device's copy is kept).
    SourceUpload,
    /// The Input's luma for Refine edges (the guide): recording the colour kernel's stop 13.
    GuideRecord,
    /// Recording the raster steps (resample, Grow, Feather, Refine edges, quantise) of every row,
    /// without the source and the guide.
    RasterRecord,
    /// Preparing the colour chain on the device (tables, node records) and recording its passes.
    ColourRecord,
    /// Recording the tail (output sharpening, tint, C13, the magnifier).
    TailRecord,
    /// Finishing the command buffer and submitting it.
    Submit,
    /// Waiting for the submission: the GPU's time for the whole job, and the readback.
    Wait,
    /// The raster planes on the CPU reference path (source, resample, Grow, Feather, guide).
    CpuRaster,
    /// The Film nodes' plain data on the host, per frame: the engine's film frames, the prefix chains, the passes' plan,
    /// the scalars and the table keys (`FilmRender::build` and `device_stack`). Not part of any other stage.
    FilmBuild,
    /// A Film node's prefix chain on the device: preparing it and recording its pass (inside `colour_record`).
    FilmPrefix,
    /// A Film node's tables on the device: found in the device's cache, or converted and uploaded on a miss
    /// (inside `colour_record`).
    FilmTables,
    /// Recording a Film node's own passes (colour in, halation, coupler diffusion, grain, return; inside
    /// `colour_record`).
    FilmPasses,
}

impl Stage {
    /// Every stage, in the order the log prints them.
    pub const ALL: [Stage; 17] = [
        Stage::Prepare,
        Stage::DeviceInput,
        Stage::InputImage,
        Stage::WideImage,
        Stage::Source,
        Stage::SourceUpload,
        Stage::GuideRecord,
        Stage::RasterRecord,
        Stage::ColourRecord,
        Stage::TailRecord,
        Stage::Submit,
        Stage::Wait,
        Stage::CpuRaster,
        Stage::FilmBuild,
        Stage::FilmPrefix,
        Stage::FilmTables,
        Stage::FilmPasses,
    ];

    /// The name in the log.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Stage::Prepare => "prepare",
            Stage::DeviceInput => "device_input",
            Stage::InputImage => "input_image",
            Stage::WideImage => "wide_image",
            Stage::Source => "mask_source",
            Stage::SourceUpload => "source_upload",
            Stage::GuideRecord => "guide_record",
            Stage::RasterRecord => "raster_record",
            Stage::ColourRecord => "colour_record",
            Stage::TailRecord => "tail_record",
            Stage::Submit => "submit",
            Stage::Wait => "wait",
            Stage::CpuRaster => "cpu_raster",
            Stage::FilmBuild => "film_build",
            Stage::FilmPrefix => "film_prefix",
            Stage::FilmTables => "film_tables",
            Stage::FilmPasses => "film_passes",
        }
    }
}

const N: usize = Stage::ALL.len();

/// The time and the count of every [`Stage`] since the split was started.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Split {
    ms: [f64; N],
    count: [u32; N],
    raster_rows: u32,
    raster_size: (u32, u32),
    film_nodes: u32,
    film_size: (u32, u32),
}

impl Split {
    /// Milliseconds spent in `stage`.
    #[must_use]
    pub fn ms(&self, stage: Stage) -> f64 {
        self.ms[stage as usize]
    }

    /// How many spans of `stage` ran.
    #[must_use]
    pub fn count(&self, stage: Stage) -> u32 {
        self.count[stage as usize]
    }

    /// Raster rows whose planes were made, and the size in pixels of the window the last of them was made
    /// over (the picture grown by the masks' receptive field): `(rows, width, height)`.
    #[must_use]
    pub fn raster(&self) -> (u32, u32, u32) {
        (self.raster_rows, self.raster_size.0, self.raster_size.1)
    }

    /// Film nodes whose pictures were made, and the size in pixels of the largest picture the first of them read (the view
    /// grown by every Film halo): `(nodes, width, height)`.
    #[must_use]
    pub fn film(&self) -> (u32, u32, u32) {
        (self.film_nodes, self.film_size.0, self.film_size.1)
    }

    /// Adds `other`'s times and counts to this split.
    pub fn merge(&mut self, other: &Split) {
        for i in 0..N {
            self.ms[i] += other.ms[i];
            self.count[i] += other.count[i];
        }
        self.raster_rows += other.raster_rows;
        if other.raster_rows > 0 {
            self.raster_size = other.raster_size;
        }
        self.film_nodes += other.film_nodes;
        if other.film_nodes > 0 {
            self.film_size = other.film_size;
        }
    }
}

thread_local! {
    static SPLIT: RefCell<Split> = const { RefCell::new(Split { ms: [0.0; N], count: [0; N], raster_rows: 0, raster_size: (0, 0), film_nodes: 0, film_size: (0, 0) }) };
}

/// Starts the calling thread's split again from zero.
pub fn begin() {
    SPLIT.with(|s| *s.borrow_mut() = Split::default());
}

/// Takes the calling thread's split and starts it again from zero.
#[must_use]
pub fn take() -> Split {
    SPLIT.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// Adds a split made on another thread to the calling thread's.
pub fn add(other: &Split) {
    SPLIT.with(|s| s.borrow_mut().merge(other));
}

/// Adds `ms` milliseconds to `stage` of the calling thread's split.
pub fn record(stage: Stage, ms: f64) {
    SPLIT.with(|s| {
        let mut s = s.borrow_mut();
        s.ms[stage as usize] += ms;
        s.count[stage as usize] += 1;
    });
}

/// Notes that the planes of `rows` raster rows are made over a window of `width x height` pixels.
pub fn note_raster(rows: usize, width: usize, height: usize) {
    if enabled() {
        SPLIT.with(|s| {
            let mut s = s.borrow_mut();
            s.raster_rows += rows as u32;
            s.raster_size = (width as u32, height as u32);
        });
    }
}

/// Notes that `nodes` Film nodes made their pictures, the first from a window of `width x height` pixels (the view grown by
/// the Film halos).
pub fn note_film(nodes: usize, width: usize, height: usize) {
    if enabled() {
        SPLIT.with(|s| {
            let mut s = s.borrow_mut();
            s.film_nodes += nodes as u32;
            s.film_size = (width as u32, height as u32);
        });
    }
}

/// A running span: its time goes to the calling thread's split when it is dropped.
#[derive(Debug)]
pub struct Span {
    stage: Stage,
    start: Instant,
    skip_ms: f64,
}

impl Span {
    /// Leaves `ms` milliseconds out of the span's time: the time of spans recorded inside it.
    pub fn subtract(&mut self, ms: f64) {
        self.skip_ms += ms;
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let ms = self.start.elapsed().as_secs_f64() * 1000.0 - self.skip_ms;
        record(self.stage, ms.max(0.0));
    }
}

/// Starts a span of `stage`, or `None` (reading no clock) when the spans are off. Bind the result:
/// `let _span = perf::span(Stage::Wait);`.
#[must_use]
pub fn span(stage: Stage) -> Option<Span> {
    enabled().then(|| Span {
        stage,
        start: Instant::now(),
        skip_ms: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_adds_to_the_split_of_its_thread_only_when_on() {
        begin();
        set_enabled(false);
        drop(span(Stage::Wait));
        assert_eq!(take().count(Stage::Wait), 0);
        set_enabled(true);
        drop(span(Stage::Wait));
        drop(span(Stage::Wait));
        let here = take();
        assert_eq!(here.count(Stage::Wait), 2);
        let there = std::thread::spawn(|| {
            drop(span(Stage::Submit));
            take()
        })
        .join()
        .expect("thread");
        assert_eq!(there.count(Stage::Submit), 1);
        assert_eq!(take().count(Stage::Submit), 0);
        add(&there);
        assert_eq!(take().count(Stage::Submit), 1);
        set_enabled(false);
    }
}
