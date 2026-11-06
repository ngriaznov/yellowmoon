//! The `ym::perf` line of a viewport frame: where the time of one frame went.
//!
//! Off unless the shell turns it on with [`set_enabled`] (it does when `YM_PERF` is on, `src-tauri`
//! `perf.rs`; `docs/11-ui.md` Measuring). Off, the cost is one atomic load per frame. Nothing leaves the
//! machine: the lines go to the local log (hard rule 7).
//!
//! One frame writes one line, `frame`, and, when the device has timestamp queries, one line, `gpu`:
//!
//! ```text
//! frame 1730x1155 gpu zoom=33% scale=4 total=96.4 cache=miss rasters=yes mask_view=none |
//!   plans=0.2 sample=0.0 raw=0.0 pass=94.1 output=0.0 |
//!   prepare=0.4x2 device_input=0.1x2 input_image=0.3x1 wide_image=0.3x1 mask_source=0.0x1
//!   source_upload=0.0x1 guide_record=0.2x1 raster_record=1.9x1 colour_record=0.6x1 tail_record=0.4x1
//!   submit=0.8x1 wait=88.0x1 cpu_raster=0.0x0 film_build=0.0x0 film_prefix=0.0x0 film_tables=0.0x0 film_passes=0.0x0
//!   raster_rows=1 raster_window=1730x1155 film_nodes=0 film_window=0x0 |
//!   color_plans=0 grade_plans=1 color_passes=1 output_passes=1 | uploads=2/0.0MiB readbacks=0/0.0MiB
//!   submissions=1 dispatches=211 plane_builds=1 plane_hits=0 pipelines=0/0.0ms precise=0
//! gpu passes (ms x count): mask raster fold 41.2x62, color chain 20.3x2, ...
//! ```
//!
//! Times are milliseconds of host wall-clock. `plans`, `sample`, `raw`, `pass` and `output` are the
//! request's [`StageTimings`](crate::pipeline::StageTimings); the stages after them are
//! [`ym_gpu::perf::Stage`] and `NxM` reads "N ms in M spans". The commands of a device job are recorded
//! on the host and run later in one submission, so the GPU's time is in `wait` (and per kernel in the
//! `gpu` line): `*_record` is the host recording the commands, not the GPU running them.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use ym_gpu::device::Gpu;
use ym_gpu::perf::{self as spans, Stage};
use ym_gpu::stream::CounterSnapshot;

use crate::pipeline::{RenderError, RenderedFrame};

/// The target of the lines.
const TARGET: &str = "ym::perf";

/// Turns the frame lines (and the spans behind them) on or off for the whole process.
pub fn set_enabled(on: bool) {
    spans::set_enabled(on);
}

/// Whether the frame lines are on.
#[must_use]
pub fn enabled() -> bool {
    spans::enabled()
}

/// What a frame is measured against: when it started, and what the device had done before it.
pub(crate) struct FrameProbe {
    started: Instant,
    gpu: Option<Arc<Gpu>>,
    before: CounterSnapshot,
    rasters: bool,
    mask_view: &'static str,
    zoom: f64,
}

impl FrameProbe {
    /// Starts measuring a frame, or `None` when the lines are off.
    pub(crate) fn start(
        gpu: Option<Arc<Gpu>>,
        rasters: bool,
        mask_view: &'static str,
        zoom: f64,
    ) -> Option<Self> {
        if !enabled() {
            return None;
        }
        spans::begin();
        let before = gpu.as_ref().map(|g| g.counters()).unwrap_or_default();
        if let Some(g) = &gpu {
            // Pass times of earlier jobs are not this frame's.
            let _ = g.take_pass_times();
        }
        Some(Self {
            started: Instant::now(),
            gpu,
            before,
            rasters,
            mask_view,
            zoom,
        })
    }

    /// Writes the lines of the finished frame.
    pub(crate) fn finish(self, result: &Result<RenderedFrame, RenderError>) {
        let total = self.started.elapsed().as_secs_f64() * 1000.0;
        let split = spans::take();
        let Ok(frame) = result else {
            tracing::info!(target: TARGET, "frame failed after {total:.1} ms");
            return;
        };
        let r = &frame.report;
        let t = &r.timings;
        let mut line = format!(
            "frame {}x{} {} zoom={:.0}% scale={} total={total:.1} cache={} rasters={} mask_view={} route={} | plans={:.1} sample={:.1} raw={:.1} pass={:.1} output={:.1} |",
            frame.width,
            frame.height,
            if r.device == crate::pipeline::ComputeDevice::Gpu {
                "gpu"
            } else {
                "cpu"
            },
            self.zoom * 100.0,
            r.scale,
            if r.runs.level_hits > 0 {
                "level"
            } else if r.runs.frame_hits > 0 {
                "hit"
            } else {
                "miss"
            },
            if self.rasters { "yes" } else { "no" },
            self.mask_view,
            match (
                &r.cpu_reason,
                r.device == crate::pipeline::ComputeDevice::Gpu
            ) {
                (_, true) => "gpu".to_owned(),
                (Some(why), false) => format!("cpu({why})"),
                (None, false) => "cpu(no device, or a stage is off)".to_owned(),
            },
            t.plans_ms,
            t.sample_ms,
            t.raw_ms,
            t.pass_ms,
            t.output_ms,
        );
        for stage in Stage::ALL {
            let _ = write!(
                line,
                " {}={:.1}x{}",
                stage.name(),
                split.ms(stage),
                split.count(stage)
            );
        }
        let (rows, rw, rh) = split.raster();
        let _ = write!(line, " raster_rows={rows} raster_window={rw}x{rh}");
        let (nodes, fw, fh) = split.film();
        let _ = write!(line, " film_nodes={nodes} film_window={fw}x{fh}");
        let _ = write!(
            line,
            " | color_plans={} grade_plans={} color_passes={} output_passes={} display_hits={} level_hits={}",
            r.runs.color_plans,
            r.runs.grade_plans,
            r.runs.color_passes,
            r.runs.output_passes,
            r.runs.display_hits,
            r.runs.level_hits
        );
        if let Some(gpu) = &self.gpu {
            let d = gpu.counters().since(self.before);
            let mib = |b: u64| b as f64 / 1_048_576.0;
            let _ = write!(
                line,
                " | uploads={}/{:.1}MiB readbacks={}/{:.1}MiB submissions={} dispatches={} plane_builds={} plane_hits={} pipelines={}/{:.1}ms precise={}",
                d.uploads,
                mib(d.uploaded_bytes),
                d.readbacks,
                mib(d.readback_bytes),
                d.submissions,
                d.dispatches,
                d.plane_builds,
                d.plane_hits,
                d.pipelines,
                d.pipeline_micros as f64 / 1000.0,
                d.precise_dispatches
            );
            let passes = gpu.take_pass_times();
            tracing::info!(target: TARGET, "{line}");
            if !passes.is_empty() {
                let mut rows: Vec<_> = passes.into_iter().collect();
                rows.sort_by(|a, b| b.1.total_ms.total_cmp(&a.1.total_ms));
                let mut gpu_line = String::from("gpu passes (ms x count):");
                let mut sum = 0.0;
                for (label, p) in &rows {
                    sum += p.total_ms;
                    let _ = write!(gpu_line, " {label} {:.2}x{},", p.total_ms, p.count);
                }
                let _ = write!(gpu_line, " sum {sum:.2}");
                tracing::info!(target: TARGET, "{gpu_line}");
            }
        } else {
            tracing::info!(target: TARGET, "{line}");
        }
    }
}
