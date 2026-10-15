//! G-09: the device path is deterministic (`docs/gpu-plan.md` sections 4.5 D7 and D8, and the determinism list of
//! section 7). Same machine, same adapter and driver: the same input, settings and version give the same bytes,
//! whatever else the device did before, beside or between the jobs of an export. Mesa lavapipe in CI, Metal on the
//! Apple runners (`.github/workflows/mac-gpu.yml`); the device is opened in poison mode, so a kernel that reads
//! what an earlier job left behind fails the comparison instead of passing by luck.
//!
//! * **Twice in one process, and in two.** The same export (JPEG, 8 and 16 bit PNG, 16 bit TIFF, DNG) three times
//!   through one context, once more through a context on a device of its own, and in two child processes of this
//!   test binary, one of them with `LP_NUM_THREADS=1` (Mesa's rasteriser thread count; no effect on Metal).
//! * **A, then B, then A.** The pool, the table cache and the resident raw words of photo A must not show in what
//!   photo A renders after photo B did (frames and exports).
//! * **Cancel at tile k, then render again.** An export cancelled part way leaves nothing resident and no file, and
//!   the next export of the photo is the uninterrupted one.
//! * **Viewport jobs between the tiles of an export** do not change a byte of it, and the frames they render are the
//!   frames rendered alone.
//! * **1, 2 and 4 exports in flight** through one context give the same bytes as one.
//!
//! `--jobs 1` against `--jobs 4` of the binary is `crates/ym-cli/tests/parity_gpu.rs`; the CPU path's thread counts
//! are `determinism.rs`. Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::*;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{
    ExportContext, ExportControl, ExportParams, ExportPhoto, ExportReport, export_photo,
};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::{Crop, CropAspect, NormRect, SettingsPatch};
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists, Size, TiffCompression};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

const MODEL: &str = "ILCE-7RM4";

/// Set in the child processes of [`two_processes_and_a_rasteriser_thread_count_give_the_same_bytes`]: the file the
/// child writes its digests to.
const CHILD_OUT: &str = "YM_GPU_DETERMINISM_CHILD_OUT";

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in gpu_determinism: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn renderer(gpu: &Arc<Gpu>, resident_caches: bool) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::ALL,
            resident_caches,
            // 256 px tiles: a 600 x 400 photo is six tiles in two rows, so an export has several jobs to cancel
            // between and to interleave with.
            full_tile_px: 256,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn context(home: &Path, gpu: &Arc<Gpu>) -> ExportContext {
    context_with(home, gpu, true)
}

/// A context whose renderer keeps its view caches on the device (`resident_caches`, the default) or on the host.
fn context_with(home: &Path, gpu: &Arc<Gpu>, resident_caches: bool) -> ExportContext {
    ExportContext::new(renderer(gpu, resident_caches)).with_home(Some(home.to_path_buf()))
}

/// A synthetic photo of `w` x `h` developed in a session of its own.
fn photo(w: u32, h: u32, level: f64, seed: u64) -> Developed {
    let white = light("sony-ilce-7rm4", 4500.0);
    let scene = Scene::new(w, h, white, Layout::Patchwork, level)
        .noise(2e-4, 2e-6)
        .seed(seed);
    Developed::new(&scene, MODEL)
}

fn export_photo_of(d: &Developed) -> ExportPhoto {
    ExportPhoto::from_session(&d.session, d.id).expect("export photo")
}

/// The formats checked, with the options that make each export distinct enough to be worth comparing.
fn formats(out: &Path) -> Vec<(&'static str, ExportOptions)> {
    let base = |format| ExportOptions {
        format,
        destination: Some(out.to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    vec![
        ("JPEG", base(FormatKind::Jpeg)),
        ("PNG 8", {
            let mut o = base(FormatKind::Png);
            o.bit_depth = BitDepth::Eight;
            o
        }),
        ("PNG 16", {
            let mut o = base(FormatKind::Png);
            o.bit_depth = BitDepth::Sixteen;
            o
        }),
        ("TIFF 16", {
            let mut o = base(FormatKind::Tiff);
            o.bit_depth = BitDepth::Sixteen;
            o.tiff_compression = TiffCompression::Lzw;
            o
        }),
        ("PNG 16, long edge 331", {
            let mut o = base(FormatKind::Png);
            o.bit_depth = BitDepth::Sixteen;
            o.size = Size::LongEdge { px: 331 };
            o
        }),
        ("DNG", base(FormatKind::Dng)),
    ]
}

fn export(ctx: &ExportContext, photo: &ExportPhoto, opts: ExportOptions) -> ExportReport {
    export_photo(ctx, photo, &ExportParams::new(opts), &ExportControl::none()).expect("export")
}

fn bytes_of(ctx: &ExportContext, photo: &ExportPhoto, opts: ExportOptions) -> Vec<u8> {
    let r = export(ctx, photo, opts);
    assert!(
        r.device == ComputeDevice::Gpu && !r.notes.iter().any(|n| n.contains("graphics card")),
        "the export did not run on the device: {:?} {:?}",
        r.device,
        r.notes
    );
    std::fs::read(&r.path).expect("read the export")
}

/// FNV-1a over the bytes, 64 bit: enough to tell two files apart in a child's report.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The bytes in use in the device's pool, the colour tables (resident on purpose) let go.
fn pool_in_use(gpu: &Gpu) -> u64 {
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        ym_gpu::kernels::color::release_tables(ctx);
        ctx.pool().trim();
        Ok(ctx.pool().stats().in_use_bytes)
    })
    .expect("pool stats")
}

/// The picture of a photo in a few views, as RGBA8 bytes.
fn frames(r: &Renderer, src: &PhotoSource) -> Vec<Vec<u8>> {
    let cancel = CancelToken::new();
    let image = r.final_size(src, &cancel).expect("size");
    let mut out = Vec::new();
    for (w, h, percent) in [(300.0, 200.0, None), (240.0, 180.0, Some(100.0))] {
        let mut st = ViewportState::new(ViewportRect::cell(w, h, 1.0));
        if let Some(value) = percent {
            st.set_mode(ViewportMode::Percent { value }, image);
        }
        out.push(
            r.viewport_frame(
                src,
                &st,
                &FrameOptions {
                    no_cache: true,
                    ..FrameOptions::default()
                },
                &cancel,
            )
            .expect("frame")
            .rgba,
        );
    }
    out.push(r.thumbnail_png(src, &cancel).expect("thumbnail"));
    out
}

#[test]
fn the_same_export_is_the_same_bytes_every_time_and_on_a_device_of_its_own() {
    let Some(gpu) = gpu() else { return };
    let d = photo(600, 400, 0.5, 11);
    let p = export_photo_of(&d);
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), &gpu);
    let baseline = pool_in_use(&gpu);
    let fresh_device = Arc::new(Gpu::start(GpuConfig::default()).expect("a second device"));
    let own = context(home.path(), &fresh_device);
    for (label, opts) in formats(home.path()) {
        let first = bytes_of(&ctx, &p, opts.clone());
        for time in 2..=4 {
            let again = bytes_of(&ctx, &p, opts.clone());
            assert!(
                first == again,
                "{label}: export {time} differs from the first"
            );
        }
        let elsewhere = bytes_of(&own, &p, opts);
        assert!(
            first == elsewhere,
            "{label}: a context on a device of its own differs"
        );
    }
    assert_eq!(
        pool_in_use(&gpu),
        baseline,
        "the exports handed everything back"
    );
}

#[test]
fn photo_a_after_photo_b_is_photo_a_before_it() {
    let Some(gpu) = gpu() else { return };
    // The views of a photo keep their caches on the device (`resident_caches`) or read them back: the pool's
    // history differs, the bytes must not.
    for resident in [true, false] {
        a_then_b_then_a(&gpu, resident);
    }
}

fn a_then_b_then_a(gpu: &Arc<Gpu>, resident: bool) {
    // B differs in size, content, brightness, orientation of the work (a straightened crop, no sharpening) and
    // therefore in every buffer size the pool hands out; A is rendered, B is rendered, A is rendered again.
    let a = photo(600, 400, 0.5, 11);
    let b = photo(520, 360, 1.1, 29);
    b.session
        .update_settings(
            b.id,
            &SettingsPatch {
                output_sharpening: Some(0.0),
                crop: Some(Some(Crop {
                    rect: NormRect {
                        x: 0.05,
                        y: 0.05,
                        width: 0.9,
                        height: 0.9,
                    },
                    angle_deg: 4.0,
                    aspect: CropAspect::Free,
                })),
                ..SettingsPatch::default()
            },
        )
        .expect("settings of B");
    let (pa, pb) = (export_photo_of(&a), export_photo_of(&b));
    let home = tempfile::tempdir().expect("home");
    let ctx = context_with(home.path(), gpu, resident);
    let baseline = pool_in_use(gpu);
    let (src_a, src_b) = (a.source(), b.source());
    let r = ctx.renderer();
    let frames_a1 = frames(r, &src_a);
    let mut exports_a1 = Vec::new();
    for (_, opts) in formats(home.path()) {
        exports_a1.push(bytes_of(&ctx, &pa, opts));
    }
    // B, in every way A was looked at.
    let frames_b = frames(r, &src_b);
    for (_, opts) in formats(home.path()) {
        let _ = bytes_of(&ctx, &pb, opts);
    }
    let frames_a2 = frames(r, &src_a);
    assert!(
        frames_a1 == frames_a2,
        "a frame of A after B differs from the frame before (resident caches {resident})"
    );
    assert!(
        frames_a1 != frames_b,
        "the two photos must not render alike, or the test says nothing"
    );
    for ((label, opts), before) in formats(home.path()).into_iter().zip(&exports_a1) {
        let after = bytes_of(&ctx, &pa, opts);
        assert!(
            *before == after,
            "{label}: A after B differs from A before (resident caches {resident})"
        );
    }
    // A viewed photo keeps its group and tiles on the device until the renderer forgets it; forgetting hands
    // everything back, an export never kept anything.
    r.forget(a.id);
    r.forget(b.id);
    assert_eq!(
        pool_in_use(gpu),
        baseline,
        "resident caches {resident}: forgetting both photos hands everything back"
    );
}

#[test]
fn a_cancel_part_way_leaves_nothing_and_the_next_export_is_the_uninterrupted_one() {
    let Some(gpu) = gpu() else { return };
    let d = photo(600, 400, 0.5, 11);
    let p = export_photo_of(&d);
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), &gpu);
    let baseline = pool_in_use(&gpu);
    for (label, opts) in formats(home.path()) {
        let reference = bytes_of(&ctx, &p, opts.clone());
        // Cancel when the progress reaches 10, 40 and 70 %: one tile, three tiles, five tiles of six, more or less.
        for share in [0.1_f32, 0.4, 0.7] {
            let out = tempfile::tempdir().expect("out");
            let mut o = opts.clone();
            o.destination = Some(out.path().to_path_buf());
            let token = CancelToken::new();
            let cancel_at = |f: f32| {
                if f >= share {
                    token.cancel();
                }
            };
            let err = export_photo(
                &ctx,
                &p,
                &ExportParams::new(o),
                &ExportControl {
                    cancel: token.clone(),
                    progress: Some(&cancel_at),
                },
            );
            match err {
                Err(e) => assert!(e.is_cancelled(), "{label} at {share}: {e}"),
                // A very fast export can finish before the callback sees the share; then it must be the
                // reference.
                Ok(r) => assert!(
                    std::fs::read(&r.path).expect("read") == reference,
                    "{label} at {share}: an export that was not cancelled differs"
                ),
            }
            assert!(
                names_in(out.path()).len() <= 1,
                "{label} at {share}: {:?}",
                names_in(out.path())
            );
            assert_eq!(
                pool_in_use(&gpu),
                baseline,
                "{label} at {share}: a cancelled export hands the device buffers back"
            );
            assert_eq!(ctx.renderer().ledger().used(), 0, "{label} at {share}");
            let again = bytes_of(&ctx, &p, opts.clone());
            assert!(
                again == reference,
                "{label} after a cancel at {share}: the next export differs from the uninterrupted one"
            );
        }
    }
}

#[test]
fn viewport_jobs_between_the_tiles_of_an_export_change_nothing() {
    let Some(gpu) = gpu() else { return };
    let d = photo(600, 400, 0.5, 11);
    let p = export_photo_of(&d);
    let src = d.source();
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), &gpu);
    let alone = frames(ctx.renderer(), &src);
    for (label, opts) in formats(home.path()) {
        let reference = bytes_of(&ctx, &p, opts.clone());
        // Every progress report of the export renders the views of the same photo on the same device, at
        // viewport priority, from inside the export: between its tiles and bands.
        let injected: Mutex<Vec<Vec<Vec<u8>>>> = Mutex::new(Vec::new());
        let calls = AtomicUsize::new(0);
        let between = |_: f32| {
            if calls.fetch_add(1, Ordering::SeqCst) < 3 {
                let f = frames(ctx.renderer(), &src);
                injected.lock().expect("lock").push(f);
            }
        };
        let out = tempfile::tempdir().expect("out");
        let mut o = opts.clone();
        o.destination = Some(out.path().to_path_buf());
        let r = export_photo(
            &ctx,
            &p,
            &ExportParams::new(o),
            &ExportControl {
                cancel: CancelToken::new(),
                progress: Some(&between),
            },
        )
        .expect("export");
        assert_eq!(r.device, ComputeDevice::Gpu, "{label}");
        assert!(
            std::fs::read(&r.path).expect("read") == reference,
            "{label}: viewport jobs between the tiles changed the export"
        );
        let injected = injected.into_inner().expect("lock");
        assert!(!injected.is_empty(), "{label}: no progress was reported");
        for f in &injected {
            assert!(
                *f == alone,
                "{label}: a frame rendered during the export differs from the frame rendered alone"
            );
        }
    }
}

#[test]
fn one_two_and_four_exports_in_flight_give_the_bytes_of_one() {
    let Some(gpu) = gpu() else { return };
    let d = photo(600, 400, 0.5, 11);
    let p = export_photo_of(&d);
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), &gpu);
    let baseline = pool_in_use(&gpu);
    for (label, opts) in formats(home.path()) {
        let reference = bytes_of(&ctx, &p, opts.clone());
        for in_flight in [1_usize, 2, 4] {
            let dirs: Vec<tempfile::TempDir> = (0..in_flight)
                .map(|_| tempfile::tempdir().expect("out"))
                .collect();
            let reports: Vec<ExportReport> = std::thread::scope(|scope| {
                let handles: Vec<_> = dirs
                    .iter()
                    .map(|dir| {
                        let mut o = opts.clone();
                        o.destination = Some(dir.path().to_path_buf());
                        let (ctx, p) = (&ctx, &p);
                        scope.spawn(move || export(ctx, p, o))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("export thread"))
                    .collect()
            });
            for (i, r) in reports.iter().enumerate() {
                assert_eq!(r.device, ComputeDevice::Gpu, "{label} x{in_flight}");
                assert!(
                    std::fs::read(&r.path).expect("read") == reference,
                    "{label}: export {i} of {in_flight} in flight differs from the single export"
                );
            }
        }
    }
    assert_eq!(pool_in_use(&gpu), baseline);
    assert_eq!(ctx.renderer().ledger().used(), 0);
}

/// What one process makes of the photo, as `label=digest` lines.
fn digests() -> Option<String> {
    let gpu = gpu()?;
    let d = photo(600, 400, 0.5, 11);
    let p = export_photo_of(&d);
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), &gpu);
    let mut text = String::new();
    for (label, opts) in formats(home.path()) {
        text.push_str(&format!(
            "{label}={:016x}\n",
            fnv(&bytes_of(&ctx, &p, opts))
        ));
    }
    Some(text)
}

/// Not a test of its own: the body of the child processes of the next test. Without [`CHILD_OUT`] it does nothing.
#[test]
fn child_process_writes_its_digests() {
    let Some(path) = std::env::var_os(CHILD_OUT).map(PathBuf::from) else {
        return;
    };
    let text = digests().unwrap_or_else(|| "no adapter\n".to_owned());
    std::fs::write(path, text).expect("write the digests");
}

#[test]
fn two_processes_and_a_rasteriser_thread_count_give_the_same_bytes() {
    let Some(here) = digests() else { return };
    let exe = std::env::current_exe().expect("this test binary");
    let dir = tempfile::tempdir().expect("dir");
    // (name, LP_NUM_THREADS): the default, and one rasteriser thread (Mesa; Metal ignores it).
    for (name, threads) in [("default", None), ("one rasteriser thread", Some("1"))] {
        let out = dir.path().join("digests.txt");
        let _ = std::fs::remove_file(&out);
        let mut cmd = std::process::Command::new(&exe);
        cmd.args([
            "--exact",
            "child_process_writes_its_digests",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(CHILD_OUT, &out);
        if let Some(n) = threads {
            cmd.env("LP_NUM_THREADS", n);
        }
        let status = cmd.status().expect("run the child");
        assert!(status.success(), "{name}: the child failed: {status}");
        let there = std::fs::read_to_string(&out).expect("the child's digests");
        assert_eq!(
            here, there,
            "{name}: a second process gave other bytes than this one"
        );
    }
}
