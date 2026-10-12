//! Buffer pool with byte accounting, and the GPU memory budget (`docs/01-architecture.md` section
//! GPU tiling and Scheduling and threads).
//!
//! # What "peak GPU memory" means here
//!
//! wgpu offers no portable way to ask how much memory the device holds for us, so every buffer the
//! kernels use (images, staging, uniforms, tables) is created through [`BufferPool::acquire`] and
//! the pool keeps a ledger:
//!
//! * `allocated_bytes`: the sum of the sizes of all live wgpu buffers, whether in use or parked in
//!   the pool's free lists. This is the footprint that counts against the budget and is what the
//!   M0-07 gate ("under 1.5 GiB for 50 MP") measures: [`PoolStats::peak_allocated_bytes`].
//! * `in_use_bytes`: the part of it currently held by callers.
//!
//! Sizes are the sizes of the buffers wgpu was asked for (rounded up to a size class, see
//! [`size_class`], so a freed buffer can serve the next, slightly different, tile). Driver
//! overhead (alignment, suballocation slack) is not included; on backends where wgpu can report its
//! allocator ([`wgpu::Device::generate_allocator_report`]) the tests print that figure next to the
//! ledger as a cross-check. The ledger counts buffers only because the kernels use no textures.
//!
//! # Budget
//!
//! The budget is 4 GiB by default (`docs/01`), scaled down to the adapter: wgpu cannot tell us the
//! VRAM size, but the largest buffer an adapter accepts is bounded by its memory, so the default is
//! `min(4 GiB, 2 * max_buffer_size)` ([`MemoryBudget::for_limits`]). An allocation that would
//! exceed the budget first frees the pool's unused buffers, and fails with
//! [`GpuError::OutOfBudget`] if that is not enough. The tile scheduler sizes its tiles from the
//! budget ([`crate::tiles::tile_size_for_budget`]) so this does not normally happen.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::device::GpuError;

/// The GPU memory budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Bytes the pool may have allocated at once.
    pub bytes: u64,
}

impl MemoryBudget {
    /// The default budget of `docs/01`: 4 GiB.
    pub const DEFAULT_BYTES: u64 = 4 << 30;
    /// The smallest budget ever used (below this the tile pipeline cannot run).
    pub const MIN_BYTES: u64 = 64 << 20;

    /// The budget for a device with `limits`: `requested` as given (at least [`Self::MIN_BYTES`]),
    /// or by default `min(4 GiB, 2 * max_buffer_size)`.
    pub fn for_limits(requested: Option<u64>, limits: &wgpu::Limits) -> Self {
        let bytes = match requested {
            Some(b) => b,
            None => Self::DEFAULT_BYTES.min(limits.max_buffer_size.saturating_mul(2)),
        };
        Self {
            bytes: bytes.max(Self::MIN_BYTES),
        }
    }
}

/// Usage counters of a [`BufferPool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Bytes of all live buffers (in use plus free-listed).
    pub allocated_bytes: u64,
    /// Bytes currently handed out.
    pub in_use_bytes: u64,
    /// The largest `allocated_bytes` seen since creation or the last [`BufferPool::reset_peaks`].
    pub peak_allocated_bytes: u64,
    /// The largest `in_use_bytes` seen since creation or the last [`BufferPool::reset_peaks`].
    pub peak_in_use_bytes: u64,
    /// Live buffers.
    pub live_buffers: u64,
    /// Buffers created from the device since creation.
    pub buffers_created: u64,
    /// Acquisitions served from the free lists.
    pub reuses: u64,
}

/// Rounds a request up to its size class: at least 256 bytes, then the next multiple of a quarter
/// of the enclosing power of two (so at most 25 % of slack, and a handful of classes per octave).
pub fn size_class(size: u64) -> u64 {
    let s = size.max(256);
    let base = 1u64 << (63 - s.leading_zeros());
    let step = (base / 4).max(256);
    s.div_ceil(step) * step
}

/// The pure bookkeeping of the pool, separate from wgpu so it can be tested without a device.
#[derive(Debug)]
pub(crate) struct Ledger {
    budget: u64,
    stats: PoolStats,
}

impl Ledger {
    pub(crate) fn new(budget: u64) -> Self {
        Self {
            budget,
            stats: PoolStats::default(),
        }
    }

    pub(crate) fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Whether `class` more bytes fit in the budget.
    pub(crate) fn fits(&self, class: u64) -> bool {
        self.stats.allocated_bytes.saturating_add(class) <= self.budget
    }

    /// A new buffer of `class` bytes was created and handed out.
    pub(crate) fn created(&mut self, class: u64) {
        self.stats.allocated_bytes += class;
        self.stats.live_buffers += 1;
        self.stats.buffers_created += 1;
        self.handed_out(class);
    }

    /// A free-listed buffer of `class` bytes was handed out again.
    pub(crate) fn reused(&mut self, class: u64) {
        self.stats.reuses += 1;
        self.handed_out(class);
    }

    fn handed_out(&mut self, class: u64) {
        self.stats.in_use_bytes += class;
        self.stats.peak_allocated_bytes = self
            .stats
            .peak_allocated_bytes
            .max(self.stats.allocated_bytes);
        self.stats.peak_in_use_bytes = self.stats.peak_in_use_bytes.max(self.stats.in_use_bytes);
    }

    /// A buffer of `class` bytes went back to the free list.
    pub(crate) fn returned(&mut self, class: u64) {
        self.stats.in_use_bytes = self.stats.in_use_bytes.saturating_sub(class);
    }

    /// A free-listed buffer of `class` bytes was destroyed.
    pub(crate) fn destroyed(&mut self, class: u64) {
        self.stats.allocated_bytes = self.stats.allocated_bytes.saturating_sub(class);
        self.stats.live_buffers = self.stats.live_buffers.saturating_sub(1);
    }

    pub(crate) fn reset_peaks(&mut self) {
        self.stats.peak_allocated_bytes = self.stats.allocated_bytes;
        self.stats.peak_in_use_bytes = self.stats.in_use_bytes;
    }
}

struct PoolState {
    free: HashMap<(u64, u32), Vec<wgpu::Buffer>>,
    ledger: Ledger,
}

struct PoolShared {
    device: wgpu::Device,
    max_buffer_size: u64,
    max_storage_binding: u64,
    budget: u64,
    state: Mutex<PoolState>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A pool of wgpu buffers keyed by size class and usage. Cloning shares the pool.
#[derive(Clone)]
pub struct BufferPool {
    shared: Arc<PoolShared>,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("budget", &self.shared.budget)
            .field("stats", &self.stats())
            .finish()
    }
}

impl BufferPool {
    /// A pool on `device` with a budget of `budget_bytes`; no buffer may exceed `max_buffer_size`
    /// and no storage buffer `max_storage_binding` (the limit of one binding).
    pub fn new(
        device: wgpu::Device,
        budget_bytes: u64,
        max_buffer_size: u64,
        max_storage_binding: u64,
    ) -> Self {
        Self {
            shared: Arc::new(PoolShared {
                device,
                max_buffer_size,
                max_storage_binding,
                budget: budget_bytes,
                state: Mutex::new(PoolState {
                    free: HashMap::new(),
                    ledger: Ledger::new(budget_bytes),
                }),
            }),
        }
    }

    /// The budget in bytes.
    pub fn budget(&self) -> u64 {
        self.shared.budget
    }

    /// A snapshot of the counters.
    pub fn stats(&self) -> PoolStats {
        lock(&self.shared.state).ledger.stats()
    }

    /// Restarts the peak counters from the current levels (use before the region to measure).
    pub fn reset_peaks(&self) {
        lock(&self.shared.state).ledger.reset_peaks();
    }

    /// Destroys every free-listed buffer and returns the bytes released.
    pub fn trim(&self) -> u64 {
        let mut st = lock(&self.shared.state);
        trim_locked(&mut st, u64::MAX)
    }

    /// Hands out a buffer of at least `size` bytes with `usage`, from the free list when one of the
    /// right class exists.
    ///
    /// # Errors
    ///
    /// [`GpuError::BufferTooLarge`] when the class exceeds the adapter's `max_buffer_size` or a
    /// storage buffer exceeds its `max_storage_buffer_binding_size`;
    /// [`GpuError::OutOfBudget`] when the budget cannot hold it even after freeing unused buffers.
    pub fn acquire(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<PooledBuffer, GpuError> {
        // A request just below the adapter's maximum must not be refused because the size class
        // rounds past it: near the limit the buffer gets no rounding slack.
        let max = self.shared.max_buffer_size & !3;
        let mut class = size_class(size);
        if class > max {
            class = size.div_ceil(4) * 4;
        }
        if class > max {
            return Err(GpuError::BufferTooLarge {
                requested: class,
                limit: self.shared.max_buffer_size,
            });
        }
        // A storage buffer is bound whole (`PooledBuffer::binding`), so it must fit one binding.
        if usage.contains(wgpu::BufferUsages::STORAGE) && size > self.shared.max_storage_binding {
            return Err(GpuError::BufferTooLarge {
                requested: size,
                limit: self.shared.max_storage_binding,
            });
        }
        let key = (class, usage.bits());
        let mut st = lock(&self.shared.state);
        if let Some(buffer) = st.free.get_mut(&key).and_then(Vec::pop) {
            st.ledger.reused(class);
            return Ok(self.wrap(buffer, size, class, usage));
        }
        if !st.ledger.fits(class) {
            let need = st
                .ledger
                .stats()
                .allocated_bytes
                .saturating_add(class)
                .saturating_sub(self.shared.budget);
            trim_locked(&mut st, need);
            if !st.ledger.fits(class) {
                return Err(GpuError::OutOfBudget {
                    requested: class,
                    allocated: st.ledger.stats().allocated_bytes,
                    budget: self.shared.budget,
                });
            }
        }
        let buffer = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: class,
            usage,
            mapped_at_creation: false,
        });
        st.ledger.created(class);
        Ok(self.wrap(buffer, size, class, usage))
    }

    fn wrap(
        &self,
        buffer: wgpu::Buffer,
        size: u64,
        class: u64,
        usage: wgpu::BufferUsages,
    ) -> PooledBuffer {
        PooledBuffer {
            buffer: Some(buffer),
            size,
            class,
            usage,
            pool: Arc::clone(&self.shared),
        }
    }
}

/// Destroys free-listed buffers, largest class first, until at least `need` bytes are released
/// (or none are left). Returns the bytes released.
fn trim_locked(st: &mut PoolState, need: u64) -> u64 {
    let mut released = 0;
    let mut keys: Vec<(u64, u32)> = st.free.keys().copied().collect();
    keys.sort_unstable_by_key(|k| std::cmp::Reverse(k.0));
    for key in keys {
        if released >= need {
            break;
        }
        if let Some(list) = st.free.get_mut(&key) {
            while released < need {
                let Some(buffer) = list.pop() else { break };
                buffer.destroy();
                st.ledger.destroyed(key.0);
                released += key.0;
            }
        }
    }
    st.free.retain(|_, v| !v.is_empty());
    released
}

/// A buffer on loan from a [`BufferPool`]; dropping it returns it to the pool.
pub struct PooledBuffer {
    buffer: Option<wgpu::Buffer>,
    size: u64,
    class: u64,
    usage: wgpu::BufferUsages,
    pool: Arc<PoolShared>,
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("size", &self.size)
            .field("class", &self.class)
            .field("usage", &self.usage)
            .finish()
    }
}

impl PooledBuffer {
    /// The size that was asked for (the buffer itself is `class_size()` bytes).
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The real size of the wgpu buffer.
    pub fn class_size(&self) -> u64 {
        self.class
    }

    /// The requested range (rounded up to 4 bytes) as a bind-group binding.
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        let bytes = (self.size.div_ceil(4) * 4).clamp(4, self.class);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self,
            offset: 0,
            size: NonZeroU64::new(bytes),
        })
    }
}

impl Deref for PooledBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        // The buffer is only taken in `drop`.
        self.buffer.as_ref().unwrap_or_else(|| unreachable_taken())
    }
}

#[cold]
fn unreachable_taken() -> ! {
    // Invariant: `buffer` is `Some` until `Drop` runs, and nothing can call `deref` after that.
    unreachable!("pooled buffer used after drop")
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut st = lock(&self.pool.state);
            st.ledger.returned(self.class);
            st.free
                .entry((self.class, self.usage.bits()))
                .or_default()
                .push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_round_up_by_at_most_a_quarter() {
        assert_eq!(size_class(0), 256);
        assert_eq!(size_class(1), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 512);
        for size in [
            300u64,
            1000,
            4097,
            1 << 20,
            (1 << 20) + 1,
            50_000_000,
            123_456_789,
        ] {
            let c = size_class(size);
            assert!(c >= size, "{size} -> {c}");
            assert!(c as f64 <= size as f64 * 1.25 + 256.0, "{size} -> {c}");
            assert_eq!(size_class(c), c, "classes are fixed points");
            assert_eq!(c % 256, 0);
        }
        // Monotone.
        let mut prev = 0;
        for size in (1..5000u64).map(|i| i * 997) {
            let c = size_class(size);
            assert!(c >= prev);
            prev = c;
        }
    }

    #[test]
    fn the_ledger_tracks_allocation_use_and_peaks() {
        let mut l = Ledger::new(1000);
        assert!(l.fits(1000) && !l.fits(1001));
        l.created(400);
        l.created(300);
        assert_eq!(l.stats().allocated_bytes, 700);
        assert_eq!(l.stats().in_use_bytes, 700);
        l.returned(400);
        assert_eq!(l.stats().in_use_bytes, 300);
        assert_eq!(l.stats().allocated_bytes, 700, "parked buffers still count");
        assert!(!l.fits(400));
        l.reused(400);
        assert_eq!(l.stats().reuses, 1);
        assert_eq!(l.stats().peak_in_use_bytes, 700);
        l.returned(400);
        l.returned(300);
        l.destroyed(400);
        assert_eq!(l.stats().allocated_bytes, 300);
        assert_eq!(l.stats().peak_allocated_bytes, 700);
        l.reset_peaks();
        assert_eq!(l.stats().peak_allocated_bytes, 300);
        assert_eq!(l.stats().live_buffers, 1);
    }

    #[test]
    fn the_default_budget_scales_down_to_the_adapter() {
        let big = wgpu::Limits {
            max_buffer_size: 16 << 30,
            ..Default::default()
        };
        assert_eq!(MemoryBudget::for_limits(None, &big).bytes, 4 << 30);
        let small = wgpu::Limits {
            max_buffer_size: 256 << 20,
            ..Default::default()
        };
        assert_eq!(MemoryBudget::for_limits(None, &small).bytes, 512 << 20);
        assert_eq!(
            MemoryBudget::for_limits(Some(1 << 30), &small).bytes,
            1 << 30,
            "an explicit budget is used as given"
        );
        assert_eq!(
            MemoryBudget::for_limits(Some(1), &small).bytes,
            MemoryBudget::MIN_BYTES
        );
    }
}
