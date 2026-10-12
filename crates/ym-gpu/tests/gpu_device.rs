//! Device ownership, the pool's byte accounting and the priority queue on a real device.
//!
//! The queue's ordering, cancellation and shutdown rules are unit-tested without a GPU in
//! `src/queue.rs`; here the same rules are checked with the real context on the device thread, and
//! the pool ledger and budget are exercised against real wgpu buffers.

mod common;

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{gpu, gpus};
use ym_gpu::device::{
    Gpu, GpuConfig, GpuError, StorageMode, StoragePreference, default_pipeline_cache_dir,
};
use ym_gpu::pool::{BufferPool, MemoryBudget, size_class};
use ym_gpu::queue::{JobError, JobSpec, Priority};

fn storage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
}

#[test]
fn the_device_reports_what_doctor_prints() {
    let Some(dev) = gpu() else { return };
    let info = dev.gpu.info();
    let text = info.to_string();
    eprintln!("{text}");
    assert!(!info.name.is_empty() && !info.backend.is_empty());
    assert!(text.contains(&info.name) && text.contains("Memory budget"));
    assert!(info.limits.max_buffer_size >= 256 << 20);
    assert!(info.memory_budget_bytes >= MemoryBudget::MIN_BYTES);
    // Auto picks the best mode the device supports.
    let expected = if info.shader_f16 {
        StorageMode::F16Native
    } else if info.f16_packing {
        StorageMode::F16Packed
    } else {
        StorageMode::F32
    };
    assert_eq!(info.storage_mode, expected);
    // The context lives on the device thread, not on the caller's.
    let caller = std::thread::current().id();
    let on_gpu_thread = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let _ = ctx.info();
            Ok(std::thread::current().id())
        })
        .expect("job");
    assert_ne!(on_gpu_thread, caller);
}

#[test]
fn a_forced_storage_mode_is_honoured_or_refused() {
    let Some(auto) = gpu() else { return };
    let f32_dev = Gpu::start(GpuConfig {
        storage: StoragePreference::Force(StorageMode::F32),
        ..GpuConfig::default()
    })
    .expect("f32 storage is always available");
    assert_eq!(f32_dev.info().storage_mode, StorageMode::F32);
    if !auto.gpu.info().shader_f16 {
        let refused = Gpu::start(GpuConfig {
            storage: StoragePreference::Force(StorageMode::F16Native),
            ..GpuConfig::default()
        });
        assert!(
            matches!(refused, Err(GpuError::RequestDevice(_))),
            "{refused:?}"
        );
    }
}

#[test]
fn the_pool_reuses_buffers_and_tracks_the_peak() {
    let Some(dev) = gpu() else { return };
    let (stats_a, stats_b, stats_c) = dev
        .gpu
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            let pool = ctx.pool();
            pool.trim();
            pool.reset_peaks();
            let base = pool.stats();
            let a = pool.acquire("a", 10_000_000, storage())?;
            let b = pool.acquire("b", 20_000_000, storage())?;
            let busy = pool.stats();
            assert_eq!(
                busy.in_use_bytes - base.in_use_bytes,
                a.class_size() + b.class_size()
            );
            assert!(a.class_size() >= 10_000_000 && a.class_size() == size_class(10_000_000));
            drop(a);
            // Same class again: served from the free list, no new buffer.
            let a2 = pool.acquire("a2", 10_000_001, storage())?;
            let reused = pool.stats();
            assert_eq!(reused.buffers_created, busy.buffers_created);
            assert_eq!(reused.reuses, busy.reuses + 1);
            drop((a2, b));
            let idle = pool.stats();
            assert_eq!(idle.in_use_bytes, base.in_use_bytes);
            assert!(
                idle.allocated_bytes > base.allocated_bytes,
                "parked buffers still count"
            );
            let released = pool.trim();
            Ok((busy, reused, (idle, released, pool.stats())))
        })
        .expect("pool job");
    let (idle, released, after) = stats_c;
    assert!(stats_a.peak_allocated_bytes >= stats_a.allocated_bytes);
    assert!(stats_b.peak_in_use_bytes >= stats_a.in_use_bytes);
    assert!(released > 0 && after.allocated_bytes == idle.allocated_bytes - released);
}

#[test]
fn the_budget_is_enforced_after_freeing_what_is_unused() {
    if gpu().is_none() {
        return;
    }
    let small = Gpu::start(GpuConfig {
        memory_budget_bytes: Some(MemoryBudget::MIN_BYTES),
        ..GpuConfig::default()
    })
    .expect("device");
    small
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            let pool = ctx.pool();
            assert_eq!(pool.budget(), MemoryBudget::MIN_BYTES);
            let held = pool.acquire("big", 40 << 20, storage())?;
            // 40 + 40 MiB does not fit in 64 MiB while the first is held.
            let err = pool.acquire("too big", 40 << 20, storage()).expect_err("over budget");
            assert!(matches!(err, GpuError::OutOfBudget { budget, .. } if budget == MemoryBudget::MIN_BYTES), "{err:?}");
            drop(held);
            // The freed buffer is parked; a different class still fits after the pool trims it.
            let other = pool.acquire("other", 50 << 20, storage())?;
            assert!(pool.stats().allocated_bytes <= MemoryBudget::MIN_BYTES);
            drop(other);
            // A storage buffer wider than one binding is refused before wgpu sees it.
            let binding = ctx.info().limits.max_storage_buffer_binding_size;
            let err = pool
                .acquire("wide", binding + 4, storage())
                .expect_err("wider than one binding");
            assert!(matches!(err, GpuError::BufferTooLarge { limit, .. } if limit == binding), "{err:?}");
            // A buffer beyond the adapter's maximum is refused outright.
            let limit = ctx.info().limits.max_buffer_size;
            let err = pool.acquire("huge", limit + 1, storage()).expect_err("too large");
            assert!(matches!(err, GpuError::BufferTooLarge { .. }), "{err:?}");
            Ok(())
        })
        .expect("budget job");
}

#[test]
fn a_request_just_below_the_adapter_maximum_is_not_refused_by_the_size_class_rounding() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            // A pool with a small maximum on the real device: 19 MB rounds up to a 21 MB size class,
            // which is above the 20 MB maximum, but the buffer itself fits.
            let pool = BufferPool::new(ctx.device.clone(), 1 << 30, 20_000_000, 20_000_000);
            assert!(
                size_class(19_000_000) > 20_000_000,
                "the premise of the test"
            );
            let near = pool.acquire("near the limit", 19_000_000, storage())?;
            assert_eq!(
                near.class_size(),
                19_000_000,
                "no rounding slack near the maximum"
            );
            drop(near);
            // The maximum itself is a limit, not a suggestion.
            let err = pool
                .acquire("over the maximum", 20_000_004, storage())
                .expect_err("larger than the adapter allows");
            assert!(matches!(err, GpuError::BufferTooLarge { .. }), "{err:?}");
            Ok(())
        })
        .expect("pool job");
}

#[test]
fn jobs_run_by_priority_on_the_real_device_and_navigation_cancels_viewport_jobs() {
    let Some(dev) = gpu() else { return };
    let (open_tx, open_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    // Occupy the device thread.
    let gate = dev
        .gpu
        .submit(JobSpec::new(Priority::Viewport), move |_, _| {
            let _ = started_tx.send(());
            let _ = open_rx.recv();
        });
    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("gate running");
    let mk = |name: &'static str, p: Priority, owner: u64| {
        let order = Arc::clone(&order);
        dev.gpu
            .submit(JobSpec::new(p).owned_by(owner), move |ctx, _| {
                // Real work: allocate and free a buffer through the pool.
                let _b = ctx.pool().acquire(name, 1024, storage());
                order.lock().expect("order").push(name);
            })
    };
    let export = mk("export", Priority::Export, 1);
    let thumb = mk("thumbnail", Priority::Thumbnail, 1);
    let tile = mk("tile", Priority::TileNeural, 1);
    let view = mk("viewport", Priority::Viewport, 1);
    let other_view = mk("other photo", Priority::Viewport, 2);
    // The user moves from photo 1 to photo 2: photo 1's viewport and tile jobs go, its export stays.
    assert_eq!(dev.gpu.scheduler().cancel_navigation(Some(1)), 2);
    assert_eq!(
        view.wait_timeout(Duration::from_secs(10)),
        Some(Err(JobError::Cancelled))
    );
    assert_eq!(
        tile.wait_timeout(Duration::from_secs(10)),
        Some(Err(JobError::Cancelled))
    );
    open_tx.send(()).expect("open");
    gate.wait().expect("gate");
    for h in [export, thumb, other_view] {
        h.wait().expect("job");
    }
    assert_eq!(
        *order.lock().expect("order"),
        ["other photo", "thumbnail", "export"]
    );
}

#[test]
fn a_gpu_job_that_runs_another_gpu_job_and_waits_gets_an_error_instead_of_freezing_the_device() {
    let Some(dev) = gpu() else { return };
    let gpu: &'static Gpu = &dev.gpu;
    let outer = gpu.submit(JobSpec::new(Priority::Viewport), move |_, _| {
        // `Gpu::run` queues the job and waits: on the device thread nothing could ever run it.
        gpu.run(JobSpec::new(Priority::Export), |_, _| Ok(()))
    });
    let result = outer
        .wait_timeout(Duration::from_secs(30))
        .expect("the device thread was not frozen by the nested wait");
    assert!(
        matches!(&result, Ok(Err(GpuError::InvalidInput(m))) if m.contains("GPU thread")),
        "{result:?}"
    );
    // The device still works.
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        ctx.wait_idle();
        Ok(())
    })
    .expect("device still alive");
}

#[test]
fn the_handles_an_integrator_shares_between_threads_are_send_and_sync() {
    fn shareable<T: Send + Sync>() {}
    fn movable<T: Send>() {}
    shareable::<Gpu>();
    shareable::<BufferPool>();
    shareable::<ym_gpu::kernels::color::PreparedChain>();
    shareable::<ym_gpu::queue::CancelToken>();
    movable::<ym_gpu::queue::JobHandle<Vec<u8>>>();
}

#[test]
fn shutting_down_a_real_device_thread_does_not_hang() {
    if gpu().is_none() {
        return;
    }
    let (tx, rx) = mpsc::channel::<()>();
    let t = std::thread::spawn(move || {
        let g = Gpu::start(GpuConfig::default()).expect("device");
        let pending: Vec<_> = (0..20)
            .map(|_| g.submit(JobSpec::new(Priority::Export), |ctx, _| ctx.wait_idle()))
            .collect();
        g.shutdown();
        for h in pending {
            // Each job either ran before the shutdown or was refused by it; none hangs.
            let r = h.wait_timeout(Duration::from_secs(30)).expect("resolved");
            assert!(r.is_ok() || r == Err(JobError::Shutdown));
        }
        let late = g.submit(JobSpec::new(Priority::Viewport), |_, _| ());
        assert_eq!(
            late.wait_timeout(Duration::from_secs(10)),
            Some(Err(JobError::Shutdown))
        );
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(120))
        .expect("shutdown finished");
    t.join().expect("thread");
}

#[test]
fn the_pipeline_cache_hook_writes_a_file_when_the_adapter_supports_one() {
    if gpu().is_none() {
        return;
    }
    let dir = tempfile_dir("ymgpu-cache");
    let g = Gpu::start(GpuConfig {
        pipeline_cache_dir: Some(dir.clone()),
        ..GpuConfig::default()
    })
    .expect("device");
    let on = g.info().pipeline_cache;
    let wrote = g
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            // Compile something so the cache has content, then persist it.
            ctx.kernels()
                .pipeline(ctx, ym_gpu::kernels::KernelId::SpikeMatrix)?;
            Ok(ctx.save_pipeline_cache())
        })
        .expect("cache job");
    eprintln!("pipeline cache on this adapter: {on}; wrote a file: {wrote}");
    assert_eq!(on, wrote);
    if wrote {
        let files: Vec<_> = std::fs::read_dir(&dir).expect("dir").flatten().collect();
        assert_eq!(files.len(), 1);
        // A second start loads it without complaint.
        drop(g);
        let g2 = Gpu::start(GpuConfig {
            pipeline_cache_dir: Some(dir.clone()),
            ..GpuConfig::default()
        })
        .expect("device again");
        assert!(g2.info().pipeline_cache);
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        default_pipeline_cache_dir().is_none_or(|p| p.to_string_lossy().contains("app.yellowmoon"))
    );
}

/// The contract of `docs/13` (GPU tests without a GPU): with no adapter the GPU tests say so loudly
/// and pass; with `YM_REQUIRE_GPU=1` they fail. One GPU test of this very binary is run in a child
/// process on a backend that cannot exist on this OS (DirectX 12 on Linux and macOS), under the
/// harness's default output capture, so the notice has to survive the capture to be seen.
#[cfg(not(windows))]
#[test]
fn a_missing_adapter_skips_loudly_or_fails_when_required() {
    let run = |require: bool| {
        let mut cmd = std::process::Command::new(std::env::current_exe().expect("test binary"));
        cmd.args(["--exact", "the_device_reports_what_doctor_prints"])
            .env("WGPU_BACKEND", "dx12");
        if require {
            cmd.env("YM_REQUIRE_GPU", "1");
        } else {
            cmd.env_remove("YM_REQUIRE_GPU");
        }
        cmd.output().expect("child test process")
    };
    let skipped = run(false);
    let stderr = String::from_utf8_lossy(&skipped.stderr);
    assert!(
        skipped.status.success(),
        "without an adapter the tests skip and pass:\n{stderr}"
    );
    assert!(
        stderr.contains("GPU TESTS SKIPPED"),
        "the skip must be announced even though the test passed and its output is captured:\n{stderr}"
    );
    let required = run(true);
    // The harness prints a failed test's captured output (the panic message) on stdout.
    let output = format!(
        "{}{}",
        String::from_utf8_lossy(&required.stdout),
        String::from_utf8_lossy(&required.stderr)
    );
    assert!(
        !required.status.success(),
        "YM_REQUIRE_GPU=1 without an adapter must fail:\n{output}"
    );
    assert!(
        output.contains("YM_REQUIRE_GPU is set but no GPU adapter could be opened"),
        "{output}"
    );
}

fn tempfile_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("temp dir");
    p
}

#[test]
fn warming_up_compiles_every_kernel_in_every_storage_mode() {
    for dev in gpus() {
        let compiled = dev
            .gpu
            .run(JobSpec::new(Priority::Export), |ctx, _| {
                ctx.warm_up()?;
                // Compiled once: asking again hands back the same pipelines.
                ctx.warm_up()?;
                Ok(format!("{:?}", ctx.kernels()))
            })
            .expect("every kernel compiles");
        assert!(
            compiled.contains(&format!(
                "compiled: {}",
                ym_gpu::kernels::KernelId::ALL.len()
            )),
            "{:?}: {compiled}",
            dev.mode
        );
    }
}

#[test]
fn every_available_storage_mode_opens_a_device() {
    let modes: Vec<_> = gpus().iter().map(|d| d.mode).collect();
    eprintln!("storage modes available here: {modes:?}");
    assert!(modes.is_empty() || modes.contains(&StorageMode::F32));
}
