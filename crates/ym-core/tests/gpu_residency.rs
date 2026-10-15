//! G-10: device residency through the renderer (Mesa lavapipe in CI).
//!
//! The caches of the raw group keep what the device made where it is ([`Backing`], `docs/gpu-plan.md`
//! G-10): the reduced group of a fit view as bands of rows, the tiles of the 100 % view as images, and
//! the sampler reads them in place. What is checked here:
//!
//! * **Same bytes.** A frame made from a resident group or resident tiles equals the frame made with
//!   the caches on the host (`RenderOptions::resident_caches = false`: the group is read back and
//!   uploaded again for the sampler) bit for bit: fit, 100 %, a pan over cached tiles, a straightened
//!   crop with paper in its corners.
//! * **No round trip.** A resident fit frame reads back its frame and nothing else; the host path
//!   reads the group back too.
//! * **Ledger.** The pool's ledger holds the caches while they are cached; navigating away lets go of
//!   the tiles, forgetting the photo lets go of all of it, and an export hands everything back.
//! * **Prefetch.** The raw words of a neighbour are uploaded at Thumbnail priority while fewer than
//!   two photos are resident, and the frame of that photo then uploads nothing.
//! * **Fall back** (`--features fault-injection`): a device that fails under a frame whose group is
//!   resident gives the CPU's frame bit for bit with the note, the group read back or made again on
//!   the CPU, and the next frame, on the recovered device, equals the resident one.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, PhotoSource, RawPrefetch, RenderOptions, Renderer,
    RenderedFrame, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::session::PhotoId;
use ym_export::{ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

const MODEL: &str = "ILCE-7RM4";

/// A device of this test's own, in poison mode (every pool buffer starts as NaN), so that the counters
/// and the ledger are not moved by tests that run beside it. `None` (with a notice) when no adapter
/// opens.
fn private_device() -> Option<Arc<Gpu>> {
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
                "\n* GPU TESTS SKIPPED in gpu_residency: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// The bytes in use in the device's pool.
fn pool_in_use(gpu: &Gpu) -> u64 {
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        Ok(ctx.pool().stats().in_use_bytes)
    })
    .expect("pool stats")
}

fn renderer(gpu: Option<&Arc<Gpu>>, resident_caches: bool) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: gpu.cloned(),
            resident_caches,
            // 256 px tiles: a 900 x 600 photo is twelve tiles, so a 100 % view gathers its window out
            // of several images.
            tile_px: 256,
            full_tile_px: 256,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// A photo of 900 x 600 with edges, flat patches and noise, as the session develops it.
fn photo(seed: u64) -> (Developed, PhotoSource) {
    let white = light("sony-ilce-7rm4", 4500.0);
    let scene = Scene::new(900, 600, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(seed);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    (d, src)
}

/// Every stage but `without`.
fn stages_without(without: GpuStages) -> GpuStages {
    [
        GpuStages::RAW_CHAIN,
        GpuStages::SAMPLER,
        GpuStages::COLOR,
        GpuStages::OUTPUT_SHARPEN,
        GpuStages::ENCODE,
        GpuStages::EXPORT_TAIL,
        GpuStages::RAW_FRONT,
    ]
    .into_iter()
    .filter(|s| *s != without)
    .fold(GpuStages::NONE, GpuStages::union)
}

fn cell(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn at_100(w: f64, h: f64, image: (usize, usize)) -> ViewportState {
    let mut st = cell(w, h);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    st
}

/// A frame made without the frame cache.
fn fresh(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    r.viewport_frame(src, st, &options, &CancelToken::new())
        .expect("frame")
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut ym_core::settings::DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn same_frame(a: &RenderedFrame, b: &RenderedFrame, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: size");
    let differing = a.rgba.iter().zip(&b.rgba).filter(|(x, y)| x != y).count();
    assert_eq!(differing, 0, "{what}: {differing} bytes differ");
}

#[test]
fn a_fit_frame_from_a_resident_group_is_the_host_round_trip_bit_for_bit() {
    let Some(gpu) = private_device() else { return };
    let (_d, src) = photo(1);
    let (host, resident) = (renderer(Some(&gpu), false), renderer(Some(&gpu), true));
    let variants: Vec<(&str, PhotoSource)> = vec![
        ("plain", src.clone()),
        (
            "graded and straightened",
            edit(&src, |s| {
                s.contrast = 30.0;
                s.saturation = 20.0;
                s.crop = Some(ym_core::settings::Crop {
                    rect: ym_core::settings::NormRect {
                        x: 0.05,
                        y: 0.05,
                        width: 0.9,
                        height: 0.9,
                    },
                    angle_deg: 4.0,
                    aspect: ym_core::settings::CropAspect::Free,
                });
            }),
        ),
    ];
    let st = cell(300.0, 200.0);
    for (name, v) in variants {
        let a = fresh(&host, &v, &st);
        let b = fresh(&resident, &v, &st);
        assert_eq!(b.report.device, ComputeDevice::Gpu, "{name}: {:?}", b.report.notes);
        assert!(b.report.scale >= 2, "{name}: the fit view reads a reduced group");
        same_frame(&a, &b, name);
    }
}

#[test]
fn a_resident_fit_frame_reads_back_the_frame_and_nothing_else() {
    let Some(gpu) = private_device() else { return };
    let (_d, src) = photo(2);
    let st = cell(300.0, 200.0);
    let counted = |resident: bool| {
        let r = renderer(Some(&gpu), resident);
        let before = gpu.counters();
        let frame = fresh(&r, &src, &st);
        let after = gpu.counters();
        assert_eq!(frame.report.device, ComputeDevice::Gpu, "{:?}", frame.report.notes);
        (
            after.readbacks - before.readbacks,
            after.submissions - before.submissions,
            frame,
        )
    };
    let (host_reads, host_submits, host_frame) = counted(false);
    let (resident_reads, resident_submits, resident_frame) = counted(true);
    eprintln!(
        "[G-10] fit frame, first of a photo: host caches {host_reads} readbacks and {host_submits} submissions; resident {resident_reads} readbacks and {resident_submits} submissions"
    );
    assert_eq!(resident_reads, 1, "the frame's own readback is the only one");
    assert!(
        host_reads > resident_reads,
        "the host path reads the group back too ({host_reads} readbacks)"
    );
    same_frame(&host_frame, &resident_frame, "fit");
    // The group stayed on the device: the second frame of another grade reads back the frame only.
    let r = renderer(Some(&gpu), true);
    let _ = fresh(&r, &src, &st);
    let graded = edit(&src, |s| s.contrast = 25.0);
    let before = gpu.counters();
    let again = fresh(&r, &graded, &st);
    let after = gpu.counters();
    assert_eq!(again.report.runs.raw_groups, 0, "the group is cached");
    assert_eq!(after.readbacks - before.readbacks, 1);
}

#[test]
fn a_100_percent_frame_from_resident_tiles_is_the_host_round_trip_bit_for_bit() {
    let Some(gpu) = private_device() else { return };
    let (_d, src) = photo(3);
    let (host, resident) = (renderer(Some(&gpu), false), renderer(Some(&gpu), true));
    let image = host
        .final_size(&src, &CancelToken::new())
        .expect("final size");
    // A window that crosses tile edges both ways, one inside a single tile, one cut by the edge of the
    // photo, and the straightened crop with paper in its corners.
    let cases: Vec<(&str, PhotoSource, ViewportState)> = vec![
        ("across tiles", src.clone(), at_100(300.0, 200.0, image)),
        ("a single tile", src.clone(), at_100(120.0, 90.0, image)),
        ("whole photo window", src.clone(), at_100(900.0, 600.0, image)),
        (
            "straightened",
            edit(&src, |s| {
                s.crop = Some(ym_core::settings::Crop {
                    rect: ym_core::settings::NormRect {
                        x: 0.05,
                        y: 0.05,
                        width: 0.9,
                        height: 0.9,
                    },
                    angle_deg: 4.0,
                    aspect: ym_core::settings::CropAspect::Free,
                });
            }),
            at_100(300.0, 200.0, image),
        ),
    ];
    for (name, v, st) in cases {
        let a = fresh(&host, &v, &st);
        let before = gpu.counters();
        let b = fresh(&resident, &v, &st);
        let after = gpu.counters();
        assert_eq!(b.report.device, ComputeDevice::Gpu, "{name}: {:?}", b.report.notes);
        assert_eq!(b.report.scale, 1, "{name}: full resolution");
        same_frame(&a, &b, name);
        eprintln!(
            "[G-10] 100 % {name}: resident frame {} readbacks, {} submissions",
            after.readbacks - before.readbacks,
            after.submissions - before.submissions
        );
    }
}

#[test]
fn a_pan_over_cached_resident_tiles_makes_no_tile_and_reads_back_the_frame_only() {
    let Some(gpu) = private_device() else { return };
    let (_d, src) = photo(4);
    let r = renderer(Some(&gpu), true);
    let image = r.final_size(&src, &CancelToken::new()).expect("final size");
    let st = at_100(300.0, 200.0, image);
    let first = fresh(&r, &src, &st);
    assert!(first.report.runs.tiles > 0, "the first frame makes tiles");
    assert_eq!(first.report.device, ComputeDevice::Gpu);
    // The same window under another grade: every tile is cached, on the device.
    let graded = edit(&src, |s| s.brightness = 12.0);
    let before = gpu.counters();
    let again = fresh(&r, &graded, &st);
    let after = gpu.counters();
    assert_eq!(again.report.runs.tiles, 0, "the tiles are cached");
    assert_eq!(after.readbacks - before.readbacks, 1, "the frame only");
    assert_eq!(after.uploads, before.uploads, "nothing goes up for the sampler");
    // The reference: the same two frames with the caches on the host.
    let host = renderer(Some(&gpu), false);
    let _ = fresh(&host, &src, &st);
    same_frame(&fresh(&host, &graded, &st), &again, "graded, cached tiles");
}

#[test]
fn the_ledger_holds_the_caches_and_gives_them_back_with_the_tiles_and_the_photo() {
    let Some(gpu) = private_device() else { return };
    let (_d, src) = photo(5);
    let (_d2, mut other) = photo(6);
    other.id = PhotoId(src.id.0 + 1);
    let r = renderer(Some(&gpu), true);
    let image = r.final_size(&src, &CancelToken::new()).expect("final size");
    // The colour tables and the kernels are the device's, not a photo's: warm them with the other
    // photo's first frame, then let that photo go.
    let _ = fresh(&r, &other, &cell(300.0, 200.0));
    r.forget(other.id);
    let baseline = pool_in_use(&gpu);

    r.navigate(src.id);
    let _ = fresh(&r, &src, &cell(300.0, 200.0));
    let with_group = pool_in_use(&gpu);
    assert!(with_group > baseline, "the words and the group are held");
    let _ = fresh(&r, &src, &at_100(300.0, 200.0, image));
    let with_tiles = pool_in_use(&gpu);
    assert!(with_tiles > with_group, "the tiles are held too");
    eprintln!(
        "[G-10] ledger: baseline {baseline} B, photo with its group {with_group} B, with tiles {with_tiles} B"
    );

    // Navigating away lets go of the tiles (the full-resolution intermediates), keeps the group and
    // the words (the photo left is the one before the current one).
    r.navigate(other.id);
    let after_navigation = pool_in_use(&gpu);
    eprintln!("[G-10] ledger after navigating away: {after_navigation} B");
    assert!(
        after_navigation < with_tiles,
        "the tiles of the photo left are let go of ({after_navigation} B, with tiles {with_tiles} B)"
    );
    assert!(
        after_navigation <= with_group,
        "no more than the words and the group stay ({after_navigation} B, with the group {with_group} B)"
    );
    // Forgetting the photo gives back everything: words, groups, tiles.
    r.forget(src.id);
    let end = pool_in_use(&gpu);
    assert_eq!(end, baseline, "the ledger is back at its baseline");
}

#[test]
fn a_neighbour_is_prefetched_while_fewer_than_two_photos_are_resident_and_then_costs_no_upload() {
    let Some(gpu) = private_device() else { return };
    let (_a, a) = photo(7);
    let (_b, mut b) = photo(8);
    let (_c, mut c) = photo(9);
    b.id = PhotoId(a.id.0 + 1);
    c.id = PhotoId(a.id.0 + 2);
    let r = renderer(Some(&gpu), true);
    r.navigate(a.id);
    let _ = fresh(&r, &a, &cell(300.0, 200.0));
    assert_eq!(r.resident_photos(), 1);

    let token = CancelToken::new();
    let before = gpu.counters();
    assert_eq!(r.prefetch_raw(&b, &token).expect("prefetch"), RawPrefetch::Uploaded);
    assert!(gpu.counters().uploads > before.uploads, "the words went up");
    assert_eq!(r.resident_photos(), 2);
    assert_eq!(
        r.prefetch_raw(&b, &token).expect("prefetch"),
        RawPrefetch::Resident,
        "once"
    );
    // Two are resident: a prefetch makes no room, and the photo being looked at keeps its words.
    assert_eq!(r.prefetch_raw(&c, &token).expect("prefetch"), RawPrefetch::Skipped);
    assert_eq!(r.resident_photos(), 2);

    // The prefetch as a queued job, at Thumbnail priority, for a photo that is resident already.
    let handle = r.spawn_raw_prefetch(b.clone());
    assert_eq!(
        handle.wait().expect("job").expect("prefetch"),
        RawPrefetch::Resident
    );

    // Moving to the neighbour: its first frame uploads nothing.
    r.navigate(b.id);
    let frame = fresh(&r, &b, &cell(300.0, 200.0));
    assert_eq!(frame.report.runs.raw_uploads, 0, "the words were there");
    assert_eq!(frame.report.device, ComputeDevice::Gpu);
}

#[test]
fn a_prefetch_does_nothing_without_a_device_or_for_a_stage_that_is_off() {
    let (_d, src) = photo(10);
    let token = CancelToken::new();
    assert_eq!(
        renderer(None, true).prefetch_raw(&src, &token).expect("prefetch"),
        RawPrefetch::Skipped
    );
    let Some(gpu) = private_device() else { return };
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(&gpu)),
            gpu_stages: stages_without(GpuStages::RAW_FRONT),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    assert_eq!(r.prefetch_raw(&src, &token).expect("prefetch"), RawPrefetch::Skipped);
    assert_eq!(r.resident_photos(), 0);
}

/// The tiles of an export never come to the host: the readbacks of an export with the raw front on
/// the device are the encoded bands, and the tiles that the front behind the CPU would have read back
/// are not among them. The ledger is back where it was when the export is done (the photo is not the
/// one being looked at, so its words went too).
#[test]
fn an_export_reads_back_the_bands_and_hands_the_ledger_back() {
    let Some(gpu) = private_device() else { return };
    let (d, _src) = photo(40);
    let exported = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    let out = tempfile::tempdir().expect("out");
    let home = tempfile::tempdir().expect("home");
    let make = |stages: GpuStages| {
        let r = Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                gpu: Some(Arc::clone(&gpu)),
                gpu_stages: stages,
                full_tile_px: 256,
                ..RenderOptions::default()
            },
        )
        .expect("render thread");
        ExportContext::new(r).with_home(Some(home.path().to_path_buf()))
    };
    let run = |ctx: &ExportContext| {
        let opts = ExportOptions {
            format: FormatKind::Jpeg,
            destination: Some(out.path().to_path_buf()),
            if_exists: IfExists::Replace,
            ..ExportOptions::default()
        };
        let before = gpu.counters();
        let report = export_photo(ctx, &exported, &ExportParams::new(opts), &ExportControl::none())
            .expect("export");
        (gpu.counters().since(before), report)
    };
    let front = make(GpuStages::ALL);
    let _warm = run(&front);
    let baseline = pool_in_use(&gpu);
    let (with_front, report) = run(&front);
    assert_eq!(report.device, ComputeDevice::Gpu, "{:?}", report.notes);
    assert_eq!(pool_in_use(&gpu), baseline, "the ledger is back at its baseline");
    let behind = make(stages_without(GpuStages::RAW_FRONT));
    let (without_front, _) = run(&behind);
    let tiles = 900usize.div_ceil(256) * 600usize.div_ceil(256);
    eprintln!(
        "[G-10] export of 900 x 600 in {tiles} tiles: raw front on the device {} readbacks, {} submissions, {} dispatches; front behind the CPU {} readbacks, {} submissions",
        with_front.readbacks,
        with_front.submissions,
        with_front.dispatches,
        without_front.readbacks,
        without_front.submissions
    );
    assert!(
        without_front.readbacks >= with_front.readbacks + tiles as u64,
        "the tiles of the chain behind the CPU's front are read back ({} against {} readbacks)",
        without_front.readbacks,
        with_front.readbacks
    );
}

/// Without the sampler on the device there is nothing to read the caches in place: the caches are
/// host images whatever `resident_caches` says, and the frames are the same as ever.
#[test]
fn caches_stay_on_the_host_when_the_sampler_is_not_on_the_device() {
    let Some(gpu) = private_device() else { return };
    let (_d, src) = photo(11);
    let make = |resident_caches| {
        Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                gpu: Some(Arc::clone(&gpu)),
                gpu_stages: stages_without(GpuStages::SAMPLER),
                resident_caches,
                ..RenderOptions::default()
            },
        )
        .expect("render thread")
    };
    let st = cell(300.0, 200.0);
    let before = gpu.counters();
    let on = fresh(&make(true), &src, &st);
    assert!(
        gpu.counters().readbacks - before.readbacks > 1,
        "the group was read back for the host's cache"
    );
    same_frame(&fresh(&make(false), &src, &st), &on, "caches asked for on the device");
}

#[cfg(feature = "fault-injection")]
mod faults {
    use super::*;
    use ym_gpu::device::GpuFault;

    /// A frame whose group is resident, a device that then fails (`fault`), and what must follow: the
    /// CPU's frame for the same photo bit for bit with the note, then, on the recovered device, the
    /// frame the resident group gave.
    fn the_cpu_takes_over_from_a_resident_group(fault: GpuFault, recovers: bool) {
        let Some(gpu) = private_device() else { return };
        let (_d, src) = photo(20);
        let st = cell(300.0, 200.0);
        let r = renderer(Some(&gpu), true);
        let graded = edit(&src, |s| s.contrast = 30.0);
        let resident_frame = fresh(&r, &graded, &st);
        assert_eq!(resident_frame.report.device, ComputeDevice::Gpu);
        // The group is resident now; another grade hits it.
        let other = edit(&src, |s| s.contrast = 31.0);
        let _ = fresh(&r, &src, &st);
        gpu.set_fault(Some(fault));
        let failed = fresh(&r, &other, &st);
        assert_eq!(failed.report.device, ComputeDevice::Cpu, "{:?}", failed.report.notes);
        assert!(
            failed
                .report
                .notes
                .iter()
                .any(|n| n.contains("graphics card")),
            "{:?}",
            failed.report.notes
        );
        let cpu = renderer(None, true);
        same_frame(&fresh(&cpu, &other, &st), &failed, "the CPU's frame");
        if recovers {
            gpu.set_fault(None);
            // The demoted group is a host image now; the device uploads it for the sampler and the
            // frame is what the resident group gave.
            let again = fresh(&r, &graded, &st);
            same_frame(&resident_frame, &again, "the recovered device");
        }
    }

    #[test]
    fn a_lost_device_leaves_the_cpu_to_make_the_frame_and_the_group_again() {
        the_cpu_takes_over_from_a_resident_group(GpuFault::Lost, false);
    }

    #[test]
    fn a_device_that_is_out_of_memory_hands_the_group_over_to_the_cpu_and_recovers() {
        the_cpu_takes_over_from_a_resident_group(GpuFault::OutOfBudget, true);
    }
}
