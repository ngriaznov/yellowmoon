//! Command streams: how work gets onto the device and results back (`docs/17-gpu.md` section 3).
//!
//! * [`PassList`] records a chain of compute passes into one command encoder, owns the temporaries
//!   and uniforms of the chain until the work has finished, and [`submit`](PassList::submit)s them
//!   as one queue submission. Every kernel module gets an `encode_<stage>` entry point that takes a
//!   `&mut PassList` and device images and returns a device image; the host wrapper that tests and
//!   the first wiring call is a thin layer on it (upload, `encode_`, submit, read).
//! * [`Pending`] is a submission in flight: [`finish`](Pending::finish) waits for it (never longer
//!   than [`GpuConfig::readback_timeout`](crate::device::GpuConfig::readback_timeout)), maps the
//!   readbacks the list asked for and returns them as [`Completed`], together with the optional GPU
//!   [`PassTimings`].
//! * [`upload_bytes`] and [`read_bytes`] move large host arrays in chunks (at most [`CHUNK_BYTES`]
//!   each, well within the 128 MiB of one storage binding), so a 61 MP raw plane never needs one
//!   giant staging copy.
//! * [`Counters`] count uploads, readbacks, submissions and dispatches, so a test can state what a
//!   region of code cost ("an unchanged chain uploads nothing", "at most three submissions per
//!   tile").
//!
//! # Lifetimes the list takes care of
//!
//! `queue.write_buffer` and `queue.submit` are ordered on one timeline: every write queued before a
//! submission lands before that submission's commands. A buffer that goes back to the pool in the
//! middle of recording and is handed out again would therefore have its *later* contents (a second
//! uniform block, a second stage's input) visible to the *earlier* pass. [`PassList::scratch`],
//! [`PassList::uniform`] and [`PassList::keep`] hold their buffers until the submission has been
//! waited for, so a buffer is never reused inside the list that is still recording with it.
//!
//! # Determinism
//!
//! Nothing here depends on timing or on what ran before, apart from the pool's recycling (which the
//! poison mode of `crate::pool` makes visible).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytemuck::Pod;

use crate::device::{DeviceHealth, GpuContext, GpuError};
use crate::kernels::KernelId;
use crate::kernels::util::{STAGING, STORAGE_SCRATCH, UNIFORM, bind};
use crate::pool::PooledBuffer;

/// The most workgroups one dispatch may use per dimension (the WebGPU default limit, K4).
pub const MAX_GROUPS_PER_DIMENSION: u32 = 65_535;

/// Bytes moved per queue write or per readback chunk: 64 MiB, half of one 128 MiB storage binding
/// (`docs/17-gpu.md` section 4), so the largest plane on the device (a 61 MP raw plane, 115 MiB)
/// goes in two chunks.
pub const CHUNK_BYTES: u64 = 64 << 20;

/// Passes of one list that can be timed; later passes run untimed.
const MAX_TIMED_PASSES: u32 = 128;

// ---------------------------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------------------------

/// What the device has been asked to do, counted since it opened. Read with
/// [`GpuContext::counters`] or [`crate::device::Gpu::counters`]; subtract two snapshots to measure a
/// region ([`CounterSnapshot::since`]).
#[derive(Debug, Default)]
pub struct Counters {
    uploads: AtomicU64,
    uploaded_bytes: AtomicU64,
    readbacks: AtomicU64,
    readback_bytes: AtomicU64,
    submissions: AtomicU64,
    dispatches: AtomicU64,
    /// GPU time per pass label, from the timestamp queries of the lists that were timed (empty unless the
    /// device was opened with [`GpuConfig::timestamps`](crate::device::GpuConfig::timestamps)).
    passes: Mutex<BTreeMap<String, PassTotal>>,
}

/// The GPU time one pass label has taken in total since the device opened (or since the log was last taken).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassTotal {
    /// How many times a pass of this label ran.
    pub count: u64,
    /// The sum of their GPU times in milliseconds (timestamp queries; on a shared-memory GPU the clock of the
    /// device, not of the host).
    pub total_ms: f64,
    /// The longest single pass in milliseconds.
    pub max_ms: f64,
}

impl Counters {
    /// Adds the timed passes of one finished list to the per-label totals.
    pub(crate) fn record_passes(&self, timings: &PassTimings) {
        if timings.passes.is_empty() {
            return;
        }
        let mut log = self
            .passes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for p in &timings.passes {
            let t = log.entry(p.label.clone()).or_default();
            t.count += 1;
            t.total_ms += p.ms;
            t.max_ms = t.max_ms.max(p.ms);
        }
    }

    /// Takes the per-label totals and starts them again from zero.
    pub fn take_passes(&self) -> BTreeMap<String, PassTotal> {
        std::mem::take(
            &mut *self
                .passes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// One host-to-device upload of `bytes` bytes.
    pub(crate) fn count_upload(&self, bytes: u64) {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        self.uploaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// One device-to-host readback of `bytes` bytes.
    pub(crate) fn count_readback(&self, bytes: u64) {
        self.readbacks.fetch_add(1, Ordering::Relaxed);
        self.readback_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// One queue submission.
    pub(crate) fn count_submission(&self) {
        self.submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// `n` compute dispatches recorded.
    pub(crate) fn count_dispatches(&self, n: u64) {
        self.dispatches.fetch_add(n, Ordering::Relaxed);
    }

    /// The current values.
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            uploads: self.uploads.load(Ordering::Relaxed),
            uploaded_bytes: self.uploaded_bytes.load(Ordering::Relaxed),
            readbacks: self.readbacks.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
            submissions: self.submissions.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
        }
    }
}

/// A reading of the [`Counters`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Host-to-device uploads (queue writes the crate's helpers made, uniforms included).
    pub uploads: u64,
    /// Bytes of those uploads.
    pub uploaded_bytes: u64,
    /// Device-to-host readbacks.
    pub readbacks: u64,
    /// Bytes of those readbacks.
    pub readback_bytes: u64,
    /// Queue submissions made through [`GpuContext::submit`].
    pub submissions: u64,
    /// Compute dispatches recorded by [`PassList::dispatch`].
    pub dispatches: u64,
}

impl CounterSnapshot {
    /// What happened between `earlier` and this reading (saturating, field by field).
    #[must_use]
    pub fn since(self, earlier: CounterSnapshot) -> CounterSnapshot {
        CounterSnapshot {
            uploads: self.uploads.saturating_sub(earlier.uploads),
            uploaded_bytes: self.uploaded_bytes.saturating_sub(earlier.uploaded_bytes),
            readbacks: self.readbacks.saturating_sub(earlier.readbacks),
            readback_bytes: self.readback_bytes.saturating_sub(earlier.readback_bytes),
            submissions: self.submissions.saturating_sub(earlier.submissions),
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Waiting and mapping
// ---------------------------------------------------------------------------------------------

/// Waits for `submission` (or for the queue to drain when `None`) for at most `timeout`. A timeout
/// taints the device: a job that has not finished in that time is a hung GPU, and whatever it was
/// computing must not be used (`docs/17-gpu.md` section 9).
fn wait_submission(
    device: &wgpu::Device,
    health: &DeviceHealth,
    timeout: Duration,
    submission: Option<wgpu::SubmissionIndex>,
) -> Result<(), GpuError> {
    match device.poll(wgpu::PollType::Wait {
        submission_index: submission,
        timeout: Some(timeout),
    }) {
        Ok(_) => health.check(),
        Err(wgpu::PollError::Timeout) => {
            let why = format!(
                "the GPU did not finish within {} s; the device is treated as hung",
                timeout.as_secs_f64()
            );
            health.mark_tainted(why.clone());
            Err(GpuError::DeviceLost(why))
        }
        Err(e) => {
            // A wait that fails because the device was destroyed or reset a moment ago is a lost
            // device, not a readback problem: the lost callback is delivered by the next poll, and the
            // caller's fallback policy (rebuild the device, or go to the CPU) depends on which it is.
            let _ = device.poll(wgpu::PollType::Poll);
            health.check()?;
            Err(GpuError::Readback(e.to_string()))
        }
    }
}

/// Waits for `submission`, then returns the first `bytes` of each staging buffer, in order.
///
/// All the buffers are mapped before the wait so that one poll finishes them all, and every buffer
/// that mapped is unmapped again before it can go back to the pool, whatever happened to the others.
/// After a timeout the device is tainted and the pool refuses further buffers, so a map request that
/// is still pending then is never reused.
pub(crate) fn map_readbacks(
    ctx: &GpuContext,
    submission: wgpu::SubmissionIndex,
    staging: &[(&PooledBuffer, u64)],
) -> Result<Vec<Vec<u8>>, GpuError> {
    map_readbacks_on(
        &ctx.device,
        ctx.health(),
        ctx.counters_shared(),
        ctx.readback_timeout(),
        submission,
        staging,
    )
}

fn map_readbacks_on(
    device: &wgpu::Device,
    health: &DeviceHealth,
    counters: &Counters,
    timeout: Duration,
    submission: wgpu::SubmissionIndex,
    staging: &[(&PooledBuffer, u64)],
) -> Result<Vec<Vec<u8>>, GpuError> {
    let mut receivers = Vec::with_capacity(staging.len());
    for (buffer, bytes) in staging {
        let (tx, rx) = mpsc::channel();
        buffer
            .slice(0..*bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        receivers.push(rx);
    }
    wait_submission(device, health, timeout, Some(submission))?;
    let mut out = Vec::with_capacity(staging.len());
    let mut first_error: Option<GpuError> = None;
    for ((buffer, bytes), rx) in staging.iter().zip(&receivers) {
        // The map callbacks run inside the poll above; if one needed a further poll, do it.
        let mapped = match rx.try_recv() {
            Ok(r) => Some(r),
            Err(_) => match wait_submission(device, health, timeout, None) {
                Ok(()) => rx.recv().ok(),
                Err(e) => {
                    first_error.get_or_insert(e);
                    None
                }
            },
        };
        match mapped {
            Some(Ok(())) => {
                match buffer.slice(0..*bytes).get_mapped_range() {
                    Ok(view) => {
                        out.push(view.to_vec());
                        counters.count_readback(*bytes);
                    }
                    Err(e) => {
                        first_error.get_or_insert(GpuError::Readback(e.to_string()));
                    }
                }
                // Never park a mapped buffer in the pool.
                buffer.unmap();
            }
            Some(Err(e)) => {
                first_error.get_or_insert(GpuError::Readback(e.to_string()));
            }
            None => {
                first_error
                    .get_or_insert(GpuError::Readback("the map callback was lost".to_string()));
            }
        }
    }
    match first_error {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

// ---------------------------------------------------------------------------------------------
// Chunked upload and readback
// ---------------------------------------------------------------------------------------------

/// Writes `data` to `dst` at byte `offset` in chunks of at most [`CHUNK_BYTES`]. See
/// [`upload_bytes_in_chunks`].
///
/// # Errors
///
/// As [`upload_bytes_in_chunks`].
pub fn upload_bytes(
    ctx: &GpuContext,
    dst: &wgpu::Buffer,
    offset: u64,
    data: &[u8],
) -> Result<(), GpuError> {
    upload_bytes_in_chunks(ctx, dst, offset, data, CHUNK_BYTES)
}

/// Writes `data` to `dst` at byte `offset` in chunks of at most `chunk_bytes` (rounded down to a
/// multiple of 4, at least 4). A length that is not a multiple of 4 is padded with zero bytes up to
/// the next multiple (the destination must have room for the padding). Between chunks the queue is
/// flushed, so the staging memory of an earlier chunk is released while the next one is copied;
/// that flush is a submission and is counted as one.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when `offset` is not a multiple of 4 or the data (with its padding)
/// does not fit in `dst`; [`GpuError::DeviceLost`] when the device is lost.
pub fn upload_bytes_in_chunks(
    ctx: &GpuContext,
    dst: &wgpu::Buffer,
    offset: u64,
    data: &[u8],
    chunk_bytes: u64,
) -> Result<(), GpuError> {
    ctx.health().check()?;
    let padded = (data.len() as u64).div_ceil(4) * 4;
    if !offset.is_multiple_of(4) || offset + padded > dst.size() {
        return Err(GpuError::InvalidInput(format!(
            "an upload of {} bytes at offset {offset} does not fit a {} byte buffer (offsets are multiples of 4)",
            data.len(),
            dst.size()
        )));
    }
    let chunk = (chunk_bytes / 4 * 4).max(4) as usize;
    let mut done = 0usize;
    while done < data.len() {
        let end = (done + chunk).min(data.len());
        let part = &data[done..end];
        if part.len().is_multiple_of(4) {
            ctx.queue.write_buffer(dst, offset + done as u64, part);
        } else {
            // Only the last chunk can be unaligned: pad it with zeros.
            let mut tail = part.to_vec();
            tail.resize(part.len().div_ceil(4) * 4, 0);
            ctx.queue.write_buffer(dst, offset + done as u64, &tail);
        }
        done = end;
        if done < data.len() {
            // Flush the queued write so its staging copy is freed before the next chunk is made.
            ctx.submit([])?;
        }
    }
    ctx.counters_shared().count_upload(data.len() as u64);
    Ok(())
}

/// Reads `len` bytes of `src` from byte `offset`, in chunks of at most [`CHUNK_BYTES`]. See
/// [`read_bytes_in_chunks`].
///
/// # Errors
///
/// As [`read_bytes_in_chunks`].
pub fn read_bytes(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    offset: u64,
    len: u64,
) -> Result<Vec<u8>, GpuError> {
    read_bytes_in_chunks(ctx, src, offset, len, CHUNK_BYTES)
}

/// Reads `len` bytes of `src` from byte `offset`, one staging buffer of at most `chunk_bytes`
/// (rounded down to a multiple of 4, at least 4) at a time: copy, submit, wait, map. `src` must
/// have `COPY_SRC`; `offset` is a multiple of 4 and the range, rounded up to 4 bytes, lies inside
/// the buffer.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a bad range; [`GpuError::DeviceLost`] when the device is lost or
/// a chunk does not finish within the readback timeout; [`GpuError::Readback`] when a mapping fails.
pub fn read_bytes_in_chunks(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    offset: u64,
    len: u64,
    chunk_bytes: u64,
) -> Result<Vec<u8>, GpuError> {
    ctx.health().check()?;
    let padded = len.div_ceil(4) * 4;
    if !offset.is_multiple_of(4) || offset + padded > src.size() {
        return Err(GpuError::InvalidInput(format!(
            "a readback of {len} bytes at offset {offset} does not fit a {} byte buffer (offsets are multiples of 4)",
            src.size()
        )));
    }
    let chunk = chunk_bytes / 4 * 4;
    let chunk = chunk.max(4);
    let mut out = Vec::with_capacity(len as usize);
    let mut done = 0u64;
    while done < len {
        let n = (len - done).min(chunk);
        let n4 = n.div_ceil(4) * 4;
        let staging = ctx.pool().acquire("readback", n4, STAGING)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(src, offset + done, &staging, 0, n4);
        let submission = ctx.submit([encoder.finish()])?;
        let mut parts = map_readbacks(ctx, submission, &[(&staging, n4)])?;
        let bytes = parts.pop().unwrap_or_default();
        out.extend_from_slice(&bytes[..n as usize]);
        done += n;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Pass lists
// ---------------------------------------------------------------------------------------------

/// The identity of a readback asked for in a [`PassList`]; give it to [`Completed::read`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadId(usize);

/// GPU time of one pass, from timestamp queries.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTiming {
    /// The label the pass was recorded with.
    pub label: String,
    /// Milliseconds between the pass's first and last timestamp.
    pub ms: f64,
}

/// GPU times of the timed passes of one list, in recording order (empty unless the device was
/// opened with [`GpuConfig::timestamps`](crate::device::GpuConfig::timestamps) and the adapter has
/// `TIMESTAMP_QUERY`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassTimings {
    /// One entry per timed pass.
    pub passes: Vec<PassTiming>,
}

impl PassTimings {
    /// The sum of the passes, in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.passes.iter().map(|p| p.ms).sum()
    }
}

struct ReadSlot {
    staging: PooledBuffer,
    bytes: u64,
    requested: u64,
}

struct TimestampState {
    queries: wgpu::QuerySet,
    labels: Vec<String>,
}

/// A chain of compute passes recorded into one command encoder, with the buffers it needs kept
/// alive until the submission has been waited for (see the module notes). Build it with
/// [`PassList::new`], record with [`dispatch`](PassList::dispatch) (and, for custom commands,
/// [`encoder`](PassList::encoder)), ask for readbacks with [`read`](PassList::read), then
/// [`submit`](PassList::submit).
///
/// Dropping a list without submitting it discards the recorded commands and returns its buffers to
/// the pool.
pub struct PassList<'a> {
    ctx: &'a GpuContext,
    encoder: wgpu::CommandEncoder,
    keep: Vec<Arc<PooledBuffer>>,
    reads: Vec<ReadSlot>,
    dispatches: u32,
    timestamps: Option<TimestampState>,
}

impl std::fmt::Debug for PassList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassList")
            .field("dispatches", &self.dispatches)
            .field("kept_buffers", &self.keep.len())
            .field("reads", &self.reads.len())
            .finish()
    }
}

impl<'a> PassList<'a> {
    /// An empty list on `ctx`'s queue. `label` names the command buffer in debug tools.
    pub fn new(ctx: &'a GpuContext, label: &str) -> PassList<'a> {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let timestamps = ctx.timestamps_enabled().then(|| TimestampState {
            queries: ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some(label),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * MAX_TIMED_PASSES,
            }),
            labels: Vec::new(),
        });
        PassList {
            ctx,
            encoder,
            keep: Vec::new(),
            reads: Vec::new(),
            dispatches: 0,
            timestamps,
        }
    }

    /// The device context the list records for.
    pub fn context(&self) -> &'a GpuContext {
        self.ctx
    }

    /// The command encoder, for commands [`dispatch`](PassList::dispatch) does not cover (a buffer
    /// copy, a clear). Commands run in the order they are recorded.
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }

    /// Dispatches recorded so far.
    pub fn dispatch_count(&self) -> u32 {
        self.dispatches
    }

    /// Keeps `buffer` alive (and out of the pool) until the work has finished, and returns a handle
    /// the caller can bind from.
    pub fn keep(&mut self, buffer: PooledBuffer) -> Arc<PooledBuffer> {
        let shared = Arc::new(buffer);
        self.keep.push(Arc::clone(&shared));
        shared
    }

    /// Keeps a buffer somebody else owns (a resident table, a plane uploaded once per photo) alive until
    /// the work has finished, so it cannot go back to the pool and be written by a later job while this
    /// list's passes still read it.
    pub fn hold(&mut self, buffer: Arc<PooledBuffer>) {
        self.keep.push(buffer);
    }

    /// A scratch storage buffer of `bytes` bytes that only kernels touch, kept alive by the list.
    /// Its content is whatever the pool had (NaN in poison mode): a kernel must write before it reads.
    ///
    /// # Errors
    ///
    /// [`GpuError::OutOfBudget`], [`GpuError::BufferTooLarge`] or [`GpuError::DeviceLost`] from the
    /// pool.
    pub fn scratch(&mut self, label: &str, bytes: u64) -> Result<Arc<PooledBuffer>, GpuError> {
        let buffer = self
            .ctx
            .pool()
            .acquire(label, bytes.max(4), STORAGE_SCRATCH)?;
        Ok(self.keep(buffer))
    }

    /// A uniform buffer holding `value`, kept alive by the list. Each pass that needs different
    /// parameters gets its own: a queue write cannot be undone by a later one inside one submission.
    ///
    /// # Errors
    ///
    /// As [`PassList::scratch`].
    pub fn uniform<T: Pod>(
        &mut self,
        label: &str,
        value: &T,
    ) -> Result<Arc<PooledBuffer>, GpuError> {
        let bytes = bytemuck::bytes_of(value);
        let buffer = self
            .ctx
            .pool()
            .acquire(label, bytes.len() as u64, UNIFORM)?;
        self.ctx.queue.write_buffer(&buffer, 0, bytes);
        self.ctx.counters_shared().count_upload(bytes.len() as u64);
        Ok(self.keep(buffer))
    }

    /// Records one compute pass: `kernel`'s pipeline with `entries` bound to group 0 and
    /// `groups` workgroups. A dimension of 0 records nothing.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when a dimension exceeds [`MAX_GROUPS_PER_DIMENSION`];
    /// [`GpuError::Shader`] when the pipeline does not compile.
    pub fn dispatch(
        &mut self,
        kernel: KernelId,
        label: &str,
        entries: &[(u32, wgpu::BindingResource<'_>)],
        groups: (u32, u32, u32),
    ) -> Result<(), GpuError> {
        if [groups.0, groups.1, groups.2]
            .iter()
            .any(|&g| g > MAX_GROUPS_PER_DIMENSION)
        {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {groups:?} workgroups exceed {MAX_GROUPS_PER_DIMENSION} in a dimension"
            )));
        }
        if groups.0 == 0 || groups.1 == 0 || groups.2 == 0 {
            return Ok(());
        }
        let pipeline = self.ctx.kernels().pipeline(self.ctx, kernel)?;
        let bind_group = bind(self.ctx, &pipeline, label, entries);
        let timestamp_writes = self.timestamps.as_mut().and_then(|t| {
            let index = t.labels.len() as u32;
            (index < MAX_TIMED_PASSES).then(|| {
                t.labels.push(label.to_string());
                wgpu::ComputePassTimestampWrites {
                    query_set: &t.queries,
                    beginning_of_pass_write_index: Some(2 * index),
                    end_of_pass_write_index: Some(2 * index + 1),
                }
            })
        });
        let mut pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes,
            });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        drop(pass);
        self.dispatches += 1;
        Ok(())
    }

    /// Asks for the first `bytes` of `src` (from byte `offset`) to be copied to a staging buffer
    /// after the passes recorded so far, and returns the id to read it with once the list has been
    /// submitted and finished. `src` needs `COPY_SRC`.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a range outside `src`; the pool's errors for the staging buffer.
    pub fn read(
        &mut self,
        src: &wgpu::Buffer,
        offset: u64,
        bytes: u64,
    ) -> Result<ReadId, GpuError> {
        let n4 = bytes.div_ceil(4) * 4;
        if !offset.is_multiple_of(4) || offset + n4 > src.size() {
            return Err(GpuError::InvalidInput(format!(
                "a readback of {bytes} bytes at offset {offset} does not fit a {} byte buffer",
                src.size()
            )));
        }
        let staging = self.ctx.pool().acquire("readback", n4, STAGING)?;
        self.encoder
            .copy_buffer_to_buffer(src, offset, &staging, 0, n4);
        self.reads.push(ReadSlot {
            staging,
            bytes: n4,
            requested: bytes,
        });
        Ok(ReadId(self.reads.len() - 1))
    }

    /// Finishes recording and submits the list as one queue submission.
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] when the device is lost or tainted; the pool's errors for the
    /// timestamp buffers.
    pub fn submit(mut self) -> Result<Pending, GpuError> {
        let ctx = self.ctx;
        let timing = match self.timestamps.take() {
            Some(t) if !t.labels.is_empty() => {
                let count = 2 * t.labels.len() as u32;
                let bytes = u64::from(count) * 8;
                let resolve = ctx.pool().acquire(
                    "timestamps",
                    bytes,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                )?;
                let staging = ctx.pool().acquire("timestamps readback", bytes, STAGING)?;
                self.encoder
                    .resolve_query_set(&t.queries, 0..count, &resolve, 0);
                self.encoder
                    .copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
                self.keep.push(Arc::new(resolve));
                Some(TimingReadback {
                    staging,
                    bytes,
                    labels: t.labels,
                    period_ns: f64::from(ctx.queue.get_timestamp_period()),
                })
            }
            _ => None,
        };
        let dispatches = self.dispatches;
        ctx.counters_shared()
            .count_dispatches(u64::from(dispatches));
        let submission = ctx.submit([self.encoder.finish()])?;
        Ok(Pending {
            device: ctx.device.clone(),
            health: ctx.health_arc(),
            counters: Arc::clone(ctx.counters_shared()),
            timeout: ctx.readback_timeout(),
            submission,
            _keep: self.keep,
            reads: self.reads,
            timing,
            dispatches,
        })
    }
}

struct TimingReadback {
    staging: PooledBuffer,
    bytes: u64,
    labels: Vec<String>,
    period_ns: f64,
}

/// A submission in flight. [`finish`](Pending::finish) waits for it; dropping it without waiting is
/// allowed (the queue keeps ordering, and the kept buffers go back to the pool when the value is
/// dropped).
pub struct Pending {
    device: wgpu::Device,
    health: Arc<DeviceHealth>,
    counters: Arc<Counters>,
    timeout: Duration,
    submission: wgpu::SubmissionIndex,
    _keep: Vec<Arc<PooledBuffer>>,
    reads: Vec<ReadSlot>,
    timing: Option<TimingReadback>,
    dispatches: u32,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("dispatches", &self.dispatches)
            .field("reads", &self.reads.len())
            .finish()
    }
}

/// What a finished [`Pending`] hands back.
#[derive(Debug, Default)]
pub struct Completed {
    reads: Vec<Vec<u8>>,
    /// GPU times of the timed passes, when the device records them.
    pub timings: Option<PassTimings>,
    /// Dispatches the list recorded.
    pub dispatches: u32,
}

impl Completed {
    /// The bytes of readback `id`.
    ///
    /// # Panics
    ///
    /// When `id` does not belong to this submission (a programming error).
    pub fn read(&self, id: ReadId) -> &[u8] {
        &self.reads[id.0]
    }

    /// Takes the bytes of readback `id` out (a second take returns an empty vector).
    ///
    /// # Panics
    ///
    /// When `id` does not belong to this submission (a programming error).
    pub fn take(&mut self, id: ReadId) -> Vec<u8> {
        std::mem::take(&mut self.reads[id.0])
    }
}

impl Pending {
    /// The queue submission index.
    pub fn submission(&self) -> &wgpu::SubmissionIndex {
        &self.submission
    }

    /// Dispatches the list recorded.
    pub fn dispatches(&self) -> u32 {
        self.dispatches
    }

    /// Waits for the work (at most the readback timeout; a timeout taints the device), maps the
    /// readbacks and decodes the timestamps.
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] when the device was lost, was tainted by an error while the work ran,
    /// or the wait timed out; [`GpuError::Readback`] when a mapping failed.
    pub fn finish(self) -> Result<Completed, GpuError> {
        let mut staging: Vec<(&PooledBuffer, u64)> =
            self.reads.iter().map(|r| (&r.staging, r.bytes)).collect();
        if let Some(t) = &self.timing {
            staging.push((&t.staging, t.bytes));
        }
        let mut parts = if staging.is_empty() {
            wait_submission(
                &self.device,
                &self.health,
                self.timeout,
                Some(self.submission.clone()),
            )?;
            Vec::new()
        } else {
            map_readbacks_on(
                &self.device,
                &self.health,
                &self.counters,
                self.timeout,
                self.submission.clone(),
                &staging,
            )?
        };
        // An error that surfaced while the work ran taints the device: its results are void.
        self.health.check()?;
        let timings = self.timing.as_ref().map(|t| {
            let raw = parts.pop().unwrap_or_default();
            let ticks: Vec<u64> = raw
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            PassTimings {
                passes: t
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| PassTiming {
                        label: label.clone(),
                        ms: ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * t.period_ns
                            / 1.0e6,
                    })
                    .collect(),
            }
        });
        if let Some(t) = &timings {
            self.counters.record_passes(t);
        }
        let reads = parts
            .into_iter()
            .zip(&self.reads)
            .map(|(mut bytes, slot)| {
                bytes.truncate(slot.requested as usize);
                bytes
            })
            .collect();
        Ok(Completed {
            reads,
            timings,
            dispatches: self.dispatches,
        })
    }

    /// Waits for the work without reading anything back. Same errors as [`Pending::finish`].
    ///
    /// # Errors
    ///
    /// See [`Pending::finish`].
    pub fn wait(self) -> Result<(), GpuError> {
        self.finish().map(|_| ())
    }
}
