//! Device ownership, the pool's byte accounting and the priority queue on a real device; the
//! foundation of the GPU wave (`docs/gpu-plan.md` G-02): f32 storage by default, poison mode, the
//! lost and tainted device, shared cancel tokens, kernel sets and the burn-compatible request.
//!
//! The queue's ordering, cancellation and shutdown rules are unit-tested without a GPU in
//! `src/queue.rs`; here the same rules are checked with the real context on the device thread, and
//! the pool ledger and budget are exercised against real wgpu buffers. Every device these tests use
//! runs in poison mode (see `common`), except where a test says otherwise.
//!
//! Run the feature-gated tests with `cargo test -p ym-gpu --features fault-injection,ml --test gpu_device`.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{gpu, gpus, start_unpoisoned};
use ym_gpu::device::{
    Gpu, GpuConfig, GpuError, StorageMode, StoragePreference, default_pipeline_cache_dir,
};
use ym_gpu::kernels::{KernelId, KernelSet};
use ym_gpu::pool::{BufferPool, MemoryBudget, POISON_WORD, size_class};
use ym_gpu::queue::{CancelToken, JobError, JobSpec, Priority};
use ym_gpu::stream::{PassList, read_bytes};

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
    // Auto is f32 on every device, also where the adapter offers native f16 (STATUS question 13):
    // image planes are f32 in memory.
    assert_eq!(info.storage_mode, StorageMode::F32);
    assert_eq!(StoragePreference::default(), StoragePreference::Auto);
    assert!(text.contains("storage f32"), "{text}");
    // The test devices run in poison mode; the unified-memory flag follows the device type.
    assert!(info.poison);
    assert_eq!(info.unified_memory, info.device_type == "IntegratedGpu");
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
    // The f16 modes are still reachable on request, on the adapters that can do them.
    if auto.gpu.info().f16_packing {
        let packed = Gpu::start(GpuConfig {
            storage: StoragePreference::Force(StorageMode::F16Packed),
            ..GpuConfig::default()
        })
        .expect("packed f16 storage");
        assert_eq!(packed.info().storage_mode, StorageMode::F16Packed);
    }
    if auto.gpu.info().shader_f16 {
        let native = Gpu::start(GpuConfig {
            storage: StoragePreference::Force(StorageMode::F16Native),
            ..GpuConfig::default()
        })
        .expect("native f16 storage");
        assert_eq!(native.info().storage_mode, StorageMode::F16Native);
    }
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
        // Placeholders (kernels registered ahead of their implementation) are not compiled by the
        // warm-up; the real ones all are.
        let real = KernelId::ALL.iter().filter(|k| !k.is_placeholder()).count();
        assert!(
            compiled.contains(&format!("compiled: {real}")),
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

// ---------------------------------------------------------------------------------------------
// G-02: poison mode
// ---------------------------------------------------------------------------------------------

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn the_poison_word_is_a_nan_as_one_f32_and_as_two_f16() {
    assert!(f32::from_bits(POISON_WORD).is_nan());
    assert!(half::f16::from_bits((POISON_WORD & 0xffff) as u16).is_nan());
    assert!(half::f16::from_bits((POISON_WORD >> 16) as u16).is_nan());
}

#[test]
fn poison_mode_fills_new_and_recycled_buffers_and_adds_copy_dst_when_missing() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            assert!(ctx.poison() && ctx.pool().is_poisoned());
            // A scratch-style usage without COPY_DST: poison adds it (and the pool keys on it).
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let a = ctx.pool().acquire("poison a", 100_000, usage)?;
            let class = a.class_size();
            let got = words(&read_bytes(ctx, &a, 0, class)?);
            assert_eq!(got.len() as u64 * 4, class);
            assert!(
                got.iter().all(|&w| w == POISON_WORD),
                "a new buffer is poisoned over its whole size class"
            );
            // Real data in, buffer back, same class out again: the old data must be gone.
            ctx.queue.write_buffer(&a, 0, &vec![0x11u8; 4096]);
            drop(a);
            let reuses = ctx.pool().stats().reuses;
            let b = ctx.pool().acquire("poison b", 100_001, usage)?;
            assert_eq!(b.class_size(), class);
            assert_eq!(
                ctx.pool().stats().reuses,
                reuses + 1,
                "served from the free list"
            );
            let got = words(&read_bytes(ctx, &b, 0, class)?);
            assert!(
                got.iter().all(|&w| w == POISON_WORD),
                "a recycled buffer is poisoned again, not handed out with the last job's data"
            );
            Ok(())
        })
        .expect("poison job");
}

#[test]
fn without_poison_a_recycled_buffer_keeps_what_the_last_job_left() {
    // The reason poison mode exists: on a plain device the stale data is there to be read by mistake.
    let Some(plain) = start_unpoisoned(StoragePreference::Auto) else {
        return;
    };
    plain
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            assert!(!ctx.poison() && !ctx.pool().is_poisoned());
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let a = ctx.pool().acquire("stale a", 8192, usage)?;
            ctx.queue.write_buffer(&a, 0, &vec![0x5Au8; 8192]);
            let class = a.class_size();
            drop(a);
            let b = ctx.pool().acquire("stale b", 8192, usage)?;
            assert_eq!(b.class_size(), class);
            let got = read_bytes(ctx, &b, 0, 8192)?;
            assert!(
                got.iter().all(|&v| v == 0x5A),
                "recycled buffers come back uncleared (this is what a stale read would see)"
            );
            Ok(())
        })
        .expect("plain job");
}

// ---------------------------------------------------------------------------------------------
// G-02: lost and tainted devices
// ---------------------------------------------------------------------------------------------

#[test]
fn destroying_the_device_flips_is_lost_and_a_later_job_reports_device_lost() {
    if gpu().is_none() {
        return;
    }
    let g = Gpu::start(GpuConfig::default()).expect("device");
    assert!(!g.is_lost() && g.lost_reason().is_none());
    g.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        ctx.device.destroy();
        Ok(())
    })
    .expect("the destroying job itself runs");
    assert!(g.is_lost(), "destroy() flips is_lost");
    let reason = g.lost_reason().expect("a reason");
    assert!(reason.contains("lost"), "{reason}");

    // A later `run` reports it before the job body starts...
    let ran = Arc::new(Mutex::new(false));
    let ran2 = Arc::clone(&ran);
    let err = g
        .run(JobSpec::new(Priority::Viewport), move |_, _| {
            *ran2.lock().expect("flag") = true;
            Ok(())
        })
        .expect_err("a job on a lost device");
    assert!(matches!(err, GpuError::DeviceLost(_)), "{err:?}");
    assert!(!*ran.lock().expect("flag"), "the job body must not run");
    // ...and the building blocks a raw `submit` job uses report it too.
    let alive = g
        .submit(JobSpec::new(Priority::Viewport), |ctx, _| ctx.check_alive())
        .wait()
        .expect("job");
    assert!(matches!(alive, Err(GpuError::DeviceLost(_))), "{alive:?}");
    let alloc = g
        .submit(JobSpec::new(Priority::Viewport), |ctx, _| {
            ctx.pool().acquire("x", 1024, storage()).map(|_| ())
        })
        .wait()
        .expect("job");
    assert!(matches!(alloc, Err(GpuError::DeviceLost(_))), "{alloc:?}");
}

#[test]
fn a_device_destroyed_in_the_middle_of_a_job_fails_its_transfers_and_never_hangs() {
    if gpu().is_none() {
        return;
    }
    let g = Gpu::start(GpuConfig::default()).expect("device");
    let (tx, rx) = mpsc::channel();
    // Run on a helper thread with a deadline, so a hang is a failure and not a stuck test run.
    let outcome = {
        let g = Arc::new(g);
        let g2 = Arc::clone(&g);
        std::thread::spawn(move || {
            let r = g2.run(JobSpec::new(Priority::Export), |ctx, _| {
                let base = ctx.pool().stats().in_use_bytes;
                let buf = ctx.pool().acquire("doomed", 4096, storage())?;
                ctx.device.destroy();
                // Neither the upload, the readback nor a pass list may panic or wait for ever.
                let up = ym_gpu::stream::upload_bytes(ctx, &buf, 0, &[1u8; 64]);
                let down = read_bytes(ctx, &buf, 0, 64).map(|_| ());
                let mut passes = PassList::new(ctx, "doomed");
                passes.dispatch(
                    KernelId::Sampler,
                    "doomed",
                    &[(0, buf.binding())],
                    (1, 1, 1),
                )?;
                let id = passes.read(&buf, 0, 16);
                let list = passes.submit().and_then(|p| p.finish()).map(|_| ());
                drop(buf);
                // Every buffer the failed transfers and the failed list took is back in the pool.
                let held = ctx.pool().stats().in_use_bytes - base;
                Ok((up.is_ok(), down, id.is_ok(), list, held))
            });
            let _ = tx.send((r, g2.is_lost()));
        });
        rx.recv_timeout(Duration::from_secs(120))
            .expect("a destroyed device must fail fast, not hang")
    };
    let (result, lost) = outcome;
    let (up_ok, down, id_ok, list, held) = result.expect("the job itself returns");
    assert_eq!(held, 0, "the failed transfers leaked pool buffers");
    eprintln!(
        "after destroy: upload ok {up_ok}, readback {down:?}, read request ok {id_ok}, list {list:?}"
    );
    assert!(lost, "the destroyed device reports itself lost");
    // After destroy() nothing may succeed silently, and the failure is classed as what it is, a lost
    // device (which tells the caller to rebuild or leave the GPU), not as a readback problem.
    assert!(
        matches!(down, Err(GpuError::DeviceLost(_))),
        "the readback of a destroyed device: {down:?}"
    );
    assert!(
        matches!(list, Err(GpuError::DeviceLost(_))),
        "the pass list of a destroyed device: {list:?}"
    );
}

#[test]
fn an_error_nobody_captured_taints_the_device_and_a_captured_one_does_not() {
    if gpu().is_none() {
        return;
    }
    let g = Gpu::start(GpuConfig::default()).expect("device");
    let bad = || wgpu::BufferDescriptor {
        label: Some("bad"),
        size: 64,
        // MAP_READ may only be combined with COPY_DST: wgpu rejects this.
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    };
    // Inside an error scope the error is the caller's: the device stays healthy.
    let captured = g
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _b = ctx.device.create_buffer(&bad());
            Ok(pollster::block_on(scope.pop()).is_some())
        })
        .expect("scoped job");
    assert!(captured, "the scope captured the validation error");
    assert!(!g.is_lost(), "a captured error does not taint the device");

    // Outside one it taints the device instead of panicking the device thread.
    g.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let _b = ctx.device.create_buffer(&bad());
        ctx.wait_idle();
        Ok(())
    })
    .expect("the job survives the error");
    assert!(g.is_lost(), "an uncaptured error taints the device");
    let reason = g.lost_reason().expect("reason");
    assert!(reason.contains("uncaptured"), "{reason}");
    let err = g
        .run(JobSpec::new(Priority::Viewport), |_, _| Ok(()))
        .expect_err("tainted");
    assert!(matches!(err, GpuError::DeviceLost(_)), "{err:?}");
}

#[test]
fn a_readback_that_does_not_finish_in_time_taints_the_device() {
    let Some(dev) = gpu() else { return };
    if !dev.gpu.info().is_software {
        // Only a CPU rasteriser is reliably slow enough to time out; a real GPU would finish.
        eprintln!("skipped: the readback timeout is exercised on software adapters only");
        return;
    }
    let g = Gpu::start(GpuConfig {
        readback_timeout: Duration::from_millis(50),
        ..GpuConfig::default()
    })
    .expect("device");
    let err = g
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("slow"),
                    source: wgpu::ShaderSource::Wgsl(
                        "@group(0) @binding(0) var<storage, read_write> d: array<f32>;\n\
                         @compute @workgroup_size(1)\n\
                         fn main() {\n\
                         \tvar x = d[0];\n\
                         \tfor (var i = 0u; i < 200000000u; i++) { x = x * 1.0000001 + 0.5; }\n\
                         \td[0] = x;\n}\n"
                            .into(),
                    ),
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("slow"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            let buf = ctx.pool().acquire(
                "slow",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )?;
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("slow"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.binding(),
                }],
            });
            // The slow pass and the readback of its result in one submission, waited for with the
            // 50 ms timeout. (On lavapipe a *second* `queue.submit` blocks behind a running one,
            // so the wait has to be the first thing that can block.)
            let mut passes = PassList::new(ctx, "slow");
            {
                let mut pass = passes
                    .encoder()
                    .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            passes.read(&buf, 0, 4)?;
            passes.submit()?.finish().map(|_| ())
        })
        .expect_err("the readback must time out");
    assert!(
        matches!(&err, GpuError::DeviceLost(m) if m.contains("did not finish")),
        "{err:?}"
    );
    assert!(g.is_lost(), "a timed-out readback taints the device");
}

/// `YM_GPU_POISON` decides `GpuConfig::default().poison`: the probe test below prints it, and the
/// parent runs the probe in child processes with the variable set and unset (the environment of a
/// running test process must not be edited).
#[test]
#[ignore = "run by poison_follows_the_environment_variable in a child process"]
fn poison_env_probe() {
    println!("POISON_DEFAULT={}", GpuConfig::default().poison);
}

#[test]
fn poison_follows_the_environment_variable() {
    let probe = |value: Option<&str>| {
        let mut cmd = std::process::Command::new(std::env::current_exe().expect("test binary"));
        cmd.args(["--exact", "poison_env_probe", "--ignored", "--nocapture"]);
        match value {
            Some(v) => cmd.env("YM_GPU_POISON", v),
            None => cmd.env_remove("YM_GPU_POISON"),
        };
        let out = cmd.output().expect("child test process");
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.status.success(), "{text}");
        text.contains("POISON_DEFAULT=true")
    };
    assert!(!probe(None), "off by default");
    assert!(!probe(Some("0")) && !probe(Some("")));
    assert!(probe(Some("1")), "YM_GPU_POISON=1 turns it on");
}

// ---------------------------------------------------------------------------------------------
// G-02: shared cancel tokens
// ---------------------------------------------------------------------------------------------

/// Occupies the device thread until the returned sender is used.
fn occupy(g: &Gpu) -> (mpsc::Sender<()>, ym_gpu::queue::JobHandle<()>) {
    let (open_tx, open_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let gate = g.submit(JobSpec::new(Priority::Viewport), move |_, _| {
        let _ = started_tx.send(());
        let _ = open_rx.recv();
    });
    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("gate running");
    (open_tx, gate)
}

#[test]
fn a_job_submitted_with_a_shared_token_is_skipped_when_the_token_is_set() {
    let Some(dev) = gpu() else { return };
    let (open, gate) = occupy(&dev.gpu);
    let token = CancelToken::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let mk = |token: CancelToken, p: Priority| {
        let ran = Arc::clone(&ran);
        dev.gpu
            .submit_with_token(JobSpec::new(p), token, move |_, _| {
                ran.fetch_add(1, Ordering::SeqCst);
            })
    };
    let a = mk(token.clone(), Priority::Export);
    let b = mk(token.clone(), Priority::Thumbnail);
    let other = mk(CancelToken::new(), Priority::Export);
    // One handle's cancel() is the group's: they share the flag.
    a.cancel();
    assert!(token.is_cancelled() && b.token().is_cancelled());
    open.send(()).expect("open");
    gate.wait().expect("gate");
    assert_eq!(a.wait(), Err(JobError::Cancelled));
    assert_eq!(b.wait(), Err(JobError::Cancelled));
    other.wait().expect("a job with its own token still runs");
    assert_eq!(ran.load(Ordering::SeqCst), 1, "only the unrelated job ran");
}

#[test]
fn a_running_job_sees_the_shared_token_and_navigation_sets_it() {
    let Some(dev) = gpu() else { return };
    // A running job sees the flag through the token it is given, which is the shared one.
    let token = CancelToken::new();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let seen = dev.gpu.submit_with_token(
        JobSpec::new(Priority::Export),
        token.clone(),
        move |_, t: &CancelToken| {
            let _ = started_tx.send(());
            let _ = go_rx.recv();
            t.is_cancelled()
        },
    );
    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("running");
    token.cancel();
    go_tx.send(()).expect("go");
    assert!(
        seen.wait().expect("job"),
        "the running job saw the shared token set"
    );

    // A viewport render's jobs share one token; navigating away from its photo sets it, so the
    // jobs of that render that are still queued are skipped as well as the one that runs.
    let (open, gate) = occupy(&dev.gpu);
    let render = CancelToken::new();
    let first = dev.gpu.submit_with_token(
        JobSpec::new(Priority::Viewport).owned_by(9),
        render.clone(),
        |_, _| (),
    );
    let sibling = dev.gpu.submit_with_token(
        JobSpec::new(Priority::Thumbnail).owned_by(9),
        render.clone(),
        |_, _| (),
    );
    // Only the viewport job belongs to photo 9 and to a priority that navigation cancels; the
    // thumbnail job is skipped because it shares that job's token.
    assert_eq!(dev.gpu.scheduler().cancel_navigation(Some(9)), 1);
    assert!(render.is_cancelled());
    assert_eq!(
        first.wait_timeout(Duration::from_secs(10)),
        Some(Err(JobError::Cancelled))
    );
    open.send(()).expect("open");
    gate.wait().expect("gate");
    assert_eq!(
        sibling.wait(),
        Err(JobError::Cancelled),
        "the sibling shares the token"
    );
}

#[test]
fn run_with_token_reports_cancelled_before_the_job_starts() {
    let Some(dev) = gpu() else { return };
    let token = CancelToken::new();
    token.cancel();
    let err = dev
        .gpu
        .run_with_token(JobSpec::new(Priority::Export), token, |_, _| Ok(()))
        .expect_err("cancelled");
    assert!(matches!(err, GpuError::Cancelled), "{err:?}");
    let ok = dev
        .gpu
        .run_with_token(
            JobSpec::new(Priority::Export),
            CancelToken::new(),
            |_, t| {
                assert!(!t.is_cancelled());
                Ok(7)
            },
        )
        .expect("fresh token");
    assert_eq!(ok, 7);
}

// ---------------------------------------------------------------------------------------------
// G-02: kernel sets, shared device handles, the request
// ---------------------------------------------------------------------------------------------

/// The number the `Debug` output of the kernels reports as compiled.
fn compiled(ctx: &ym_gpu::device::GpuContext) -> usize {
    let text = format!("{:?}", ctx.kernels());
    text.split("compiled: ")
        .nth(1)
        .and_then(|t| t.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|n| n.parse().ok())
        .expect("compiled count")
}

#[test]
fn warm_up_sets_compile_their_real_kernels_and_a_placeholder_compiles_on_request() {
    if gpu().is_none() {
        return;
    }
    let g = Gpu::start(GpuConfig::default()).expect("device");
    g.run(JobSpec::new(Priority::Export), |ctx, _| {
        let real = |set: KernelSet| set.kernels().filter(|k| !k.is_placeholder()).count();
        assert_eq!(compiled(ctx), 0, "pipelines are lazy");
        ctx.warm_up_set(KernelSet::View)?;
        assert_eq!(compiled(ctx), real(KernelSet::View));
        ctx.warm_up_set(KernelSet::Tail)?;
        assert_eq!(
            real(KernelSet::Tail),
            0,
            "the resize is still a placeholder"
        );
        assert_eq!(compiled(ctx), real(KernelSet::View));
        ctx.warm_up_set(KernelSet::Raw)?;
        assert_eq!(compiled(ctx), real(KernelSet::View) + real(KernelSet::Raw));
        ctx.warm_up()?;
        assert_eq!(compiled(ctx), real(KernelSet::All));
        // The placeholders are valid pipelines: the plan's later tasks drop into ids that compile.
        for id in KernelId::ALL.into_iter().filter(|k| k.is_placeholder()) {
            ctx.kernels().pipeline(ctx, id)?;
        }
        assert_eq!(compiled(ctx), KernelId::ALL.len());
        Ok(())
    })
    .expect("warm-up job");
}

#[test]
fn the_handle_hands_out_the_wgpu_objects_it_opened() {
    let Some(dev) = gpu() else { return };
    let info = dev.gpu.info();
    let setup = dev.gpu.wgpu_setup();
    assert_eq!(format!("{:?}", setup.backend), info.backend);
    assert_eq!(setup.adapter.get_info().name, info.name);
    assert_eq!(
        setup.device.limits().max_buffer_size,
        info.limits.max_buffer_size
    );
    // Clones of the device thread's own objects: the context keeps the instance and the adapter too.
    let (name, same_limits) = dev
        .gpu
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            Ok((
                ctx.adapter().get_info().name,
                ctx.adapter().limits() == ctx.adapter().limits(),
            ))
        })
        .expect("job");
    assert_eq!(name, info.name);
    assert!(same_limits);
    // Another thread can use the queue it was handed (what burn does with its own copy).
    let t = std::thread::spawn(move || {
        setup.queue.submit([]);
    });
    t.join().expect("queue used from another thread");
}

#[cfg(not(feature = "ml"))]
#[test]
fn the_default_build_requests_no_optional_device_features() {
    // Auto is f32, no pipeline cache is asked for and timestamps are off: the device is the plainest
    // one the adapter can give. The `ml` feature is what widens the request.
    let Some(dev) = gpu() else { return };
    let features = dev.gpu.wgpu_setup().device.features();
    assert!(features.is_empty(), "{features:?}");
}

#[cfg(feature = "ml")]
#[test]
fn the_ml_feature_requests_what_cubecl_requests() {
    if gpu().is_none() {
        return;
    }
    let g = Gpu::start(GpuConfig::default()).expect("device");
    let setup = g.wgpu_setup();
    let want = setup.adapter.features()
        - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
        - wgpu::Features::all_experimental_mask();
    eprintln!(
        "ml device features: {:?}\nadapter features withheld (mappable primary buffers / experimental): {:?}",
        setup.device.features(),
        setup.adapter.features()
            & (wgpu::Features::MAPPABLE_PRIMARY_BUFFERS | wgpu::Features::all_experimental_mask())
    );
    assert_eq!(setup.device.features(), want);
    assert_eq!(setup.device.limits(), setup.adapter.limits());
    // The device still works for our own kernels.
    g.run(JobSpec::new(Priority::Export), |ctx, _| {
        ctx.warm_up_set(KernelSet::View)
    })
    .expect("kernels compile on the ml device");
}

// ---------------------------------------------------------------------------------------------
// G-02 review: the existing kernels read nothing they did not write, NaN-proof
// ---------------------------------------------------------------------------------------------

/// Acceptance (2) of G-02 says the existing kernel tests pass under poison, but several of their
/// error measures skip a NaN (`fold(0.0, f64::max)`, `ratio > worst`, `f32::max`), so a kernel that
/// read a poisoned buffer could still pass them. This module is the NaN-proof version: every existing
/// host wrapper runs on the poisoned test device and on a plain one (fresh buffers are zeros there),
/// whole frame and tiled (so recycled buffers are exercised), and the two must give the same bits and
/// no NaN or infinity anywhere. A stale read cannot satisfy that.
mod existing_kernels {
    use super::*;

    use common::chains::{self, camera_image};
    use common::exact::assert_poison_neutral;
    use ym_color::display::ColorChain;
    use ym_gpu::cpu::chain::RawFrame;
    use ym_gpu::cpu::demosaic::adaptive_ratio::{Params, ShadowMatch};
    use ym_gpu::cpu::demosaic::{CaCurves, DemosaicInputs, Method, RadialCurve};
    use ym_gpu::cpu::normalize::Levels;
    use ym_gpu::cpu::output_sharpen::OutputSharpen;
    use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA};
    use ym_gpu::cpu::tiling::Stage;
    use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage, synth};
    use ym_gpu::kernels::PixelData;
    use ym_gpu::kernels::adaptive_ratio::{self as gpu_ar, Balance};
    use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
    use ym_gpu::kernels::diffraction::PreparedDiffraction;
    use ym_gpu::kernels::log_guided as gpu_lg;
    use ym_gpu::kernels::output_sharpen::PreparedSharpen;
    use ym_gpu::kernels::raw::{RawParams, run_raw_to_display};
    use ym_gpu::tiles::TilePlan;
    use ym_noise::model::NoiseModel;

    /// Runs `run` on the poisoned device and a plain one; same bits, all finite, and not trivially
    /// empty or zero (a skipped kernel would compare nothing).
    fn neutral(what: &str, run: impl Fn(&Gpu) -> Vec<f32>) {
        let Some(out) = assert_poison_neutral(what, run) else {
            return;
        };
        assert!(
            out.len() > 100 && out.iter().any(|&v| v != 0.0),
            "{what}: the output is empty or all zero, so nothing was compared"
        );
    }

    fn flatten(img: ColorImage) -> Vec<f32> {
        match img.data {
            PixelData::F32Planar(v) => v,
            PixelData::Rgba8(v) => v.into_iter().map(f32::from).collect(),
            PixelData::U16Planar(v) => v.into_iter().map(f32::from).collect(),
        }
    }

    fn model(a: f64, b: f64, sigma_row: f64, q: f64) -> NoiseDn {
        NoiseDn::from_model(&NoiseModel {
            a,
            b,
            delta: [0.0; 4],
            sigma_row,
            q,
        })
    }

    /// A seeded unbalanced mosaic of a natural-looking scene with noise, values in -0.05..=peak.
    fn mosaic(w: usize, h: usize, cfa: Cfa, seed: u64, peak: f32, sigma: f32) -> Mosaic {
        let mut m = synth::mosaic_from_rgb(&synth::natural(w, h, seed), cfa);
        for v in &mut m.data {
            *v *= peak;
        }
        synth::add_noise(&mut m.data, sigma, seed ^ 0x5eed);
        for v in &mut m.data {
            *v = v.max(-0.05);
        }
        m
    }

    #[derive(Clone)]
    struct ArCase {
        m: Mosaic,
        balance: Balance,
        noise: NoiseDn,
        ca: Option<CaCurves>,
        params: Params,
    }

    #[test]
    fn the_adaptive_ratio_kernel_reads_nothing_it_did_not_write() {
        let low = model(2e-5, 1e-8, 0.0, 1.0 / 65535.0);
        let high = model(5e-4, 4e-5, 1e-3, 0.0);
        let curve =
            |a: f64, b: f64| RadialCurve::from_knots(&[(0.0, a), (200.0, b)]).expect("curve");
        let ca = CaCurves::new(19.0, 17.0, curve(0.0, 0.6), curve(0.0, -0.5)).expect("curves");
        let d = Params::default();
        let cases = vec![
            ArCase {
                m: mosaic(45, 37, Cfa::RGGB, 1, 1.3, 0.005),
                balance: Balance::Unbalanced,
                noise: low,
                ca: None,
                params: d,
            },
            // The dark noisy scene shadow matching is for, balanced input.
            ArCase {
                m: mosaic(40, 34, Cfa::GBRG, 2, 0.12, 0.012),
                balance: Balance::Balanced,
                noise: high,
                ca: None,
                params: Params {
                    shadow_match: ShadowMatch::On,
                    ..d
                },
            },
            // The lateral CA path.
            ArCase {
                m: mosaic(38, 42, Cfa::BGGR, 3, 1.3, 0.005),
                balance: Balance::Unbalanced,
                noise: low,
                ca: Some(ca),
                params: d,
            },
            ArCase {
                m: mosaic(43, 35, Cfa::GRBG, 4, 0.12, 0.012),
                balance: Balance::Unbalanced,
                noise: high,
                ca: None,
                params: Params {
                    shadow_sharpening: false,
                    ..d
                },
            },
        ];
        neutral("adaptive ratio", |g| {
            let cases = cases.clone();
            g.run(JobSpec::new(Priority::Export), move |ctx, token| {
                let mut out = Vec::new();
                for c in &cases {
                    let inputs = DemosaicInputs {
                        gains: [1.9, 1.0, 1.4],
                        noise: c.noise,
                        ca: c.ca.as_ref(),
                    };
                    out.extend(gpu_ar::demosaic(ctx, &c.m, c.balance, &inputs, &c.params)?.data);
                    // Tiles recycle the buffers of the tile before them.
                    out.extend(
                        gpu_ar::demosaic_tiled(
                            ctx, &c.m, c.balance, &inputs, &c.params, 16, token,
                        )?
                        .data,
                    );
                }
                Ok(out)
            })
            .expect("adaptive ratio job")
        });
    }

    #[test]
    fn the_log_guided_kernel_reads_nothing_it_did_not_write() {
        let noise = model(2e-5, 1e-8, 0.0, 1.0 / 65535.0);
        let cases: Vec<(Mosaic, i32)> = [
            (Cfa::RGGB, 37, 29, 0),
            (Cfa::GBRG, 34, 40, 100),
            (Cfa::BGGR, 31, 33, -50),
            (Cfa::GRBG, 40, 36, 0),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(cfa, w, h, detail))| (mosaic(w, h, cfa, 10 + i as u64, 1.3, 0.004), detail))
        .collect();
        neutral("log-guided demosaic", |g| {
            let cases = cases.clone();
            g.run(JobSpec::new(Priority::Export), move |ctx, token| {
                let mut out = Vec::new();
                for (m, detail) in &cases {
                    out.extend(gpu_lg::demosaic(ctx, m, noise, *detail)?.data);
                    out.extend(gpu_lg::demosaic_tiled(ctx, m, noise, *detail, 16, token)?.data);
                }
                Ok(out)
            })
            .expect("log-guided job")
        });
    }

    /// A natural scene with the gains applied and a trace of noise, as the sharpening stages see it.
    fn balanced_scene(w: usize, h: usize, seed: u64, g0: [f32; 3]) -> RgbImage {
        let mut img = synth::natural(w, h, seed);
        let n = img.plane_len();
        for (ch, gain) in g0.iter().enumerate() {
            for v in &mut img.data[ch * n..(ch + 1) * n] {
                *v *= 0.75 * gain;
            }
        }
        synth::add_noise(&mut img.data, 0.003, seed ^ 0x6e6f_6973);
        img
    }

    #[test]
    fn the_diffraction_kernel_reads_nothing_it_did_not_write() {
        let g0 = [2.0f32, 1.0, 1.5];
        let plan = DiffractionPlan::new(
            &DiffractionParams::new(3.76, 11.0)
                .with_scale(1.0)
                .with_strength(1.0),
            g0,
            &model(2.5e-5, 4e-8, 5e-5, 0.0),
            SPEC_LUMA,
        )
        .expect("an active plan");
        let images = [
            balanced_scene(61, 47, 21, g0),
            balanced_scene(40, 55, 22, g0),
        ];
        neutral("diffraction", |g| {
            let (plan, images) = (plan.clone(), images.clone());
            g.run(JobSpec::new(Priority::Export), move |ctx, token| {
                let prepared = PreparedDiffraction::new(ctx, &plan)?;
                let mut out = Vec::new();
                for img in &images {
                    out.extend(prepared.run(ctx, img)?.data);
                    out.extend(prepared.run_tiled(ctx, img, 24, token)?.data);
                }
                Ok(out)
            })
            .expect("diffraction job")
        });
    }

    #[test]
    fn the_output_sharpening_kernel_reads_nothing_it_did_not_write() {
        let images = [
            balanced_scene(70, 60, 31, [1.0; 3]),
            balanced_scene(53, 41, 32, [1.0; 3]),
        ];
        let sets: Vec<(OutputSharpen, f32)> = vec![
            (OutputSharpen::screen(), 1.0),
            (OutputSharpen::screen(), 0.6),
            (OutputSharpen::print(), 1.0),
        ];
        neutral("output sharpening", |g| {
            let (images, sets) = (images.clone(), sets.clone());
            g.run(JobSpec::new(Priority::Export), move |ctx, token| {
                let mut out = Vec::new();
                for (params, scale) in &sets {
                    let prepared = PreparedSharpen::new(ctx, params, *scale)?;
                    for img in &images {
                        out.extend(prepared.run(ctx, img)?.data);
                        let plan = prepared.tile_plan(img.width, img.height, 32);
                        prepared.run_tiled(ctx, img, &plan, 2, token, &mut |_, tile| {
                            out.extend(flatten(tile));
                            Ok(())
                        })?;
                    }
                }
                Ok(out)
            })
            .expect("output sharpening job")
        });
    }

    #[test]
    fn the_color_kernel_reads_nothing_it_did_not_write() {
        let img = camera_image(23, 17, 0xBEEF);
        let cases: Vec<(String, ColorChain)> = chains::cases()
            .into_iter()
            .map(|(name, p)| (name, ColorChain::new(&p).expect("chain")))
            .collect();
        assert!(cases.len() > 10);
        neutral("color chain", |g| {
            let (img, cases) = (img.clone(), cases.clone());
            g.run(JobSpec::new(Priority::Export), move |ctx, _| {
                let mut out = Vec::new();
                for (_, chain) in &cases {
                    let prepared = PreparedChain::new(ctx, chain)?;
                    for output in [
                        ColorOutput::Stage(Stop::C6),
                        ColorOutput::Stage(Stop::C11),
                        ColorOutput::Stage(Stop::C12),
                        ColorOutput::Stage(Stop::C13),
                        ColorOutput::Stage(Stop::Before),
                        ColorOutput::Rgba8 { dither: true },
                        ColorOutput::U16 { bits: 10 },
                    ] {
                        out.extend(flatten(prepared.render(ctx, &img, (3, 5), output)?));
                    }
                }
                Ok(out)
            })
            .expect("colour job")
        });
    }

    #[test]
    fn the_raw_front_end_to_display_pipeline_reads_nothing_it_did_not_write() {
        use common::Rng;
        let (w, h) = (120usize, 77usize);
        let mut rng = Rng(11);
        let data: Vec<u16> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64 / w as f64, (i / w) as f64 / h as f64);
                let v = 0.4 + 0.3 * (x * 6.0).sin() * (y * 4.0).cos() + 0.03 * (rng.unit() - 0.5);
                (512.0 + v.clamp(-0.01, 1.3) * 15488.0).max(0.0) as u16
            })
            .collect();
        let params = RawParams {
            cfa: Cfa::GBRG,
            levels: Levels {
                black: [512.0, 514.0, 513.0, 511.0],
                white: [16000.0, 16010.0, 15990.0, 16005.0],
            },
            gains: [1.9, 1.0, 1.5],
        };
        let mut cp = ym_color::display::ChainParams::new(chains::fused_matrix(), chains::tone());
        cp.grade = chains::grade_params();
        cp.curves = chains::curves();
        let chain = ColorChain::new(&cp).expect("chain");
        let stages = [
            Stage::Normalize,
            Stage::Gains,
            Stage::Demosaic(Method::Bilinear),
        ];
        let plan = TilePlan::new(w, h, 50, &stages);
        neutral("raw to display", |g| {
            let (data, chain, plan) = (data.clone(), chain.clone(), plan.clone());
            let params = params.clone();
            g.run(JobSpec::new(Priority::Export), move |ctx, token| {
                let raw = RawFrame {
                    width: w,
                    height: h,
                    cfa: params.cfa,
                    data: &data,
                    qstep: None,
                };
                let prepared = PreparedChain::new(ctx, &chain)?;
                let mut out = vec![f32::NAN; 3 * w * h];
                run_raw_to_display(
                    ctx,
                    &raw,
                    &params,
                    &prepared,
                    &plan,
                    ColorOutput::Stage(Stop::C13),
                    2,
                    token,
                    &mut |t, img| {
                        let tile = flatten(img);
                        let c = t.core;
                        for ch in 0..3 {
                            for y in 0..c.h {
                                let dst = ch * w * h + (c.y + y) * w + c.x;
                                out[dst..dst + c.w]
                                    .copy_from_slice(&tile[ch * c.w * c.h + y * c.w..][..c.w]);
                            }
                        }
                        Ok(())
                    },
                )?;
                Ok(out)
            })
            .expect("pipeline job")
        });
    }
}

// ---------------------------------------------------------------------------------------------
// G-02: fault injection (feature `fault-injection`)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "fault-injection")]
mod faults {
    use super::*;
    use ym_gpu::device::GpuFault;

    fn armed(fault: GpuFault) -> Option<Gpu> {
        gpu()?;
        Some(
            Gpu::start(GpuConfig {
                fault: Some(fault),
                ..GpuConfig::default()
            })
            .expect("device"),
        )
    }

    #[test]
    fn an_injected_out_of_budget_fails_every_allocation_until_cleared() {
        let Some(g) = armed(GpuFault::OutOfBudget) else {
            return;
        };
        let r = g
            .run(JobSpec::new(Priority::Export), |ctx, _| {
                ctx.pool().acquire("x", 1024, storage()).map(|_| ())
            })
            .expect_err("injected");
        assert!(matches!(r, GpuError::OutOfBudget { .. }), "{r:?}");
        // Sticky: a retry fails too, so "retry once, then the CPU" sees the failure twice.
        let r = g
            .run(JobSpec::new(Priority::Export), |ctx, _| {
                ctx.pool().acquire("x", 1024, storage()).map(|_| ())
            })
            .expect_err("still armed");
        assert!(matches!(r, GpuError::OutOfBudget { .. }), "{r:?}");
        g.set_fault(None);
        g.run(JobSpec::new(Priority::Export), |ctx, _| {
            ctx.pool().acquire("x", 1024, storage()).map(|_| ())
        })
        .expect("cleared");
        assert!(!g.is_lost(), "an allocation failure is not a lost device");
    }

    #[test]
    fn an_injected_shader_fault_fails_pipeline_compilation() {
        let Some(g) = armed(GpuFault::Shader) else {
            return;
        };
        let r = g
            .run(JobSpec::new(Priority::Export), |ctx, _| {
                ctx.kernels()
                    .pipeline(ctx, KernelId::SpikeMatrix)
                    .map(|_| ())
            })
            .expect_err("injected");
        assert!(
            matches!(&r, GpuError::Shader(m) if m.contains("injected")),
            "{r:?}"
        );
        g.set_fault(None);
        g.run(JobSpec::new(Priority::Export), |ctx, _| {
            ctx.kernels()
                .pipeline(ctx, KernelId::SpikeMatrix)
                .map(|_| ())
        })
        .expect("cleared");
    }

    #[test]
    fn an_injected_lost_device_reports_device_lost_to_the_next_job() {
        let Some(g) = armed(GpuFault::Lost) else {
            return;
        };
        let r = g
            .run(JobSpec::new(Priority::Export), |_, _| Ok(()))
            .expect_err("lost at the start of the job");
        assert!(matches!(r, GpuError::DeviceLost(_)), "{r:?}");
        assert!(g.is_lost());
    }

    /// `YM_GPU_FAULT` arms a fault when the device opens (read in a child process, as above).
    #[test]
    #[ignore = "run by the_environment_variable_arms_a_fault in a child process"]
    fn fault_env_probe() {
        if gpu().is_none() {
            println!("FAULT_PROBE=no-gpu");
            return;
        }
        let g = Gpu::start(GpuConfig::default()).expect("device");
        let hits: Vec<bool> = (0..3)
            .map(|_| {
                g.run(JobSpec::new(Priority::Export), |ctx, _| {
                    Ok(ctx.inject_tile_fault().is_err())
                })
                .expect("job")
            })
            .collect();
        println!("FAULT_PROBE={hits:?}");
    }

    #[test]
    fn the_environment_variable_arms_a_fault() {
        if gpu().is_none() {
            return;
        }
        let run = |value: Option<&str>| {
            let mut cmd = std::process::Command::new(std::env::current_exe().expect("test binary"));
            cmd.args([
                "--exact",
                "faults::fault_env_probe",
                "--ignored",
                "--nocapture",
            ]);
            match value {
                Some(v) => cmd.env("YM_GPU_FAULT", v),
                None => cmd.env_remove("YM_GPU_FAULT"),
            };
            let out = cmd.output().expect("child test process");
            assert!(out.status.success());
            String::from_utf8_lossy(&out.stdout).to_string()
        };
        assert!(run(None).contains("FAULT_PROBE=[false, false, false]"));
        assert!(run(Some("tile@2")).contains("FAULT_PROBE=[false, true, false]"));
        // A value that is not a fault is ignored with a warning, not a failure.
        assert!(run(Some("boom")).contains("FAULT_PROBE=[false, false, false]"));
    }

    #[test]
    fn an_injected_cancel_sets_the_token_of_the_nth_job_only() {
        let Some(g) = armed(GpuFault::CancelAtJob(2)) else {
            return;
        };
        let seen: Vec<bool> = (0..3)
            .map(|_| {
                g.run(JobSpec::new(Priority::Export), |_, t| Ok(t.is_cancelled()))
                    .expect("job")
            })
            .collect();
        assert_eq!(seen, [false, true, false]);
    }

    #[test]
    fn an_injected_tile_fault_fails_the_nth_call_of_the_hook() {
        let Some(g) = armed(GpuFault::FailTile(3)) else {
            return;
        };
        let results: Vec<bool> = (0..4)
            .map(|_| {
                g.run(JobSpec::new(Priority::Export), |ctx, _| {
                    Ok(ctx.inject_tile_fault().is_err())
                })
                .expect("job")
            })
            .collect();
        assert_eq!(results, [false, false, true, false]);
    }
}
