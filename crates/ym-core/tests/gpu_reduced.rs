//! The raw front and the fit-view path in the render path (Mesa lavapipe in CI) against the CPU
//! reference.
//!
//! * The fit view and the thumbnails (`Job::group(k)`): with [`GpuStages::RAW_FRONT`] the reduced
//!   raw group (R1, R4, R5, the binning, the vignetting gain, the sigma, the light denoise and R9 at
//!   1/k) is made on the device from the photo's raw words, which are uploaded once per photo
//!   (`StageRuns::raw_uploads`) and kept for the two most recent photos, released on navigation and
//!   on close. The frames agree with the CPU's within the 8-bit codes the colour pass already
//!   gives (`render_gpu.rs`); the group alone is compared with `reduced_bayer` in `ym-gpu`'s
//!   `tests/gpu_reduced.rs`.
//! * The full-resolution tiles (the 100 % view, the export): with the raw front on, the device runs
//!   R1 to R9 (`R1..R9`) from the resident words; with it off, R6 to R8 behind the CPU's front.
//! * A device error in the group falls back to the CPU for it, with the note.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` (the `gpu-lavapipe` job) turns
//! that into a failure.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource,
    RenderOptions, RenderedFrame, Renderer, ViewportRect, ViewportState,
};
use ym_core::session::PhotoId;
use ym_core::settings::{AutoOr, NoiseReductionValue};
use ym_gpu::cpu::demosaic::Method;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

/// The stages of the view (sampler, colour, output sharpening, encode): what a renderer that keeps the
/// raw stages on the CPU runs on the device.
const VIEW: GpuStages = GpuStages::SAMPLER
    .union(GpuStages::COLOR)
    .union(GpuStages::OUTPUT_SHARPEN)
    .union(GpuStages::ENCODE);

fn gpu() -> Option<Arc<Gpu>> {
    // Poison mode: every pool buffer starts as NaN, so a kernel that reads what it did not write
    // fails the comparison with the CPU instead of passing by luck.
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
                "\n* GPU TESTS SKIPPED in gpu_reduced: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn renderer(gpu: Option<&Arc<Gpu>>, stages: GpuStages) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: gpu.map(Arc::clone),
            gpu_stages: stages,
            demosaic: Method::AdaptiveRatio,
            full_tile_px: 512,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// A patchwork brighter than the sensor holds (the brightest patches clip, so R5 has work), with noise.
fn clipped_scene(w: u32, h: u32, seed: u64) -> Scene {
    let white = light("sony-ilce-7rm4", 4500.0);
    Scene::new(w, h, white, Layout::Patchwork, 1.25)
        .noise(2e-4, 2e-6)
        .seed(seed)
}

fn source(scene: &Scene) -> PhotoSource {
    Developed::new(scene, MODEL).source()
}

fn fit_frame(r: &Renderer, src: &PhotoSource, w: f64, h: f64) -> RenderedFrame {
    let st = ViewportState::new(ViewportRect::cell(w, h, 1.0));
    r.viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("fit frame")
}

/// Whether the frame's report says the graphics card did not do something (the only notes this file
/// cares about; the report also carries the note about the missing trained demosaic model).
fn mentions_the_card(f: &RenderedFrame) -> bool {
    f.report.notes.iter().any(|n| n.contains("graphics card"))
}

/// The largest difference in 8-bit codes and the mean of two frames.
fn codes(a: &RenderedFrame, b: &RenderedFrame) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (mut worst, mut sum) = (0u8, 0u64);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = x.abs_diff(*y);
        worst = worst.max(d);
        sum += u64::from(d);
    }
    (worst, sum as f64 / a.rgba.len() as f64)
}

fn render_full(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("full render")
}

#[test]
fn the_fit_view_group_runs_on_the_device_from_words_uploaded_once() {
    let Some(gpu) = gpu() else { return };
    let src = source(&clipped_scene(1300, 900, 3));
    let cpu = renderer(None, GpuStages::NONE);
    // The reference for the group: the view stages on the device, the raw stages on the CPU. The
    // frames differ by the group alone.
    let view_only = renderer(Some(&gpu), VIEW);
    let all = renderer(Some(&gpu), GpuStages::ALL);

    let a = fit_frame(&view_only, &src, 300.0, 200.0);
    assert_eq!(
        (a.report.runs.device_groups, a.report.runs.raw_uploads),
        (0, 0)
    );
    assert_eq!(a.report.runs.raw_groups, 1);
    let b = fit_frame(&all, &src, 300.0, 200.0);
    assert_eq!(b.report.device, ComputeDevice::Gpu, "{:?}", b.report.notes);
    assert!(!mentions_the_card(&b), "{:?}", b.report.notes);
    assert_eq!(b.report.scale, a.report.scale);
    assert!(b.report.scale >= 2, "a fit view is a reduced group");
    assert_eq!(b.report.runs.raw_groups, 1);
    assert_eq!(
        b.report.runs.device_groups, 1,
        "the group ran on the device"
    );
    assert_eq!(b.report.runs.raw_uploads, 1, "the words were uploaded");
    // No host sigma plane: the device made its own.
    assert_eq!(b.report.runs.sigma_planes, 0);
    assert_eq!(a.report.runs.sigma_planes, 1);
    let (worst, mean) = codes(&a, &b);
    eprintln!("[fit view] device group against CPU group: worst {worst} codes, mean {mean:e}");
    // Twice the first measured values (1 code, mean 1.7e-5): the same on every adapter.
    assert!(worst <= 2 && mean <= 3.4e-5, "{worst} {mean}");
    // And the whole renderer against the CPU one.
    let c = fit_frame(&cpu, &src, 300.0, 200.0);
    let (worst, mean) = codes(&c, &b);
    eprintln!("[fit view] device against CPU: worst {worst} codes, mean {mean:e}");
    // Twice the first measured values (1 code, mean 6.3e-5).
    assert!(worst <= 2 && mean <= 1.3e-4, "{worst} {mean}");

    // A second size: a new group (another k), the same words.
    let b2 = fit_frame(&all, &src, 600.0, 400.0);
    assert_eq!(b2.report.runs.device_groups, 1);
    assert_eq!(b2.report.runs.raw_uploads, 0, "the words are resident");
    // A thumbnail and the 100 % tiles use the same words too.
    let before = all.counters(src.id);
    let _ = all
        .thumbnail_png(&src, &CancelToken::new())
        .expect("thumbnail");
    let full = render_full(&all, &src);
    let after = all.counters(src.id);
    assert_eq!(after.raw_uploads, before.raw_uploads, "no second upload");
    assert_eq!(after.raw_uploads, 1, "one upload for the whole photo");
    assert!(full.report.runs.tiles >= 6);
    assert_eq!(
        full.report.runs.raw_uploads, 0,
        "the export used the resident words"
    );
    // An export lets go of the photo's words when it is done ...
    assert_eq!(all.resident_photos(), 0);
    // ... unless the photo is the one being looked at.
    all.navigate(src.id);
    let again = render_full(&all, &src);
    assert_eq!(
        again.report.runs.raw_uploads, 1,
        "uploaded again for this export"
    );
    assert_eq!(all.resident_photos(), 1);
}

#[test]
fn the_tiles_run_r1_to_r9_with_the_front_and_r6_to_r8_without_it() {
    let Some(gpu) = gpu() else { return };
    let src = source(&clipped_scene(1300, 900, 5));
    let cpu = render_full(&renderer(None, GpuStages::NONE), &src);
    let device_of = |img: &FullImage| -> Vec<(String, ComputeDevice)> {
        img.report
            .ranges
            .iter()
            .map(|r| (r.range.label(), r.device))
            .collect()
    };
    // The raw chain alone: R6 to R8 on the device behind the CPU's front, the host mosaic read.
    let behind = render_full(&renderer(Some(&gpu), GpuStages::RAW_CHAIN), &src);
    assert_eq!(
        device_of(&behind),
        vec![("R6..R8".to_owned(), ComputeDevice::Gpu)]
    );
    assert_eq!(behind.report.runs.raw_uploads, 0, "no words on the device");
    // With the front: everything on the device, one upload for the six tiles.
    let front = render_full(
        &renderer(Some(&gpu), GpuStages::RAW_CHAIN.union(GpuStages::RAW_FRONT)),
        &src,
    );
    assert_eq!(
        device_of(&front),
        vec![("R1..R9".to_owned(), ComputeDevice::Gpu)],
        "{:?}",
        front.report.notes
    );
    assert_eq!(front.report.runs.raw_uploads, 1);
    assert!(front.report.runs.tiles >= 6);
    // Both are the CPU render's picture, up to the decisions of the demosaic and of method A on exact
    // ties; the front changes little against the CPU's own front.
    let stat = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        let (mut worst, mut over, mut sum) = (0.0f64, 0usize, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            assert!(x.is_finite() && y.is_finite());
            let d = f64::from((x - y).abs());
            worst = worst.max(d);
            over += usize::from(d > 1e-3);
            sum += d;
        }
        (worst, over as f64 / a.len() as f64, sum / a.len() as f64)
    };
    let (w1, o1, m1) = stat(&cpu.rgb, &behind.rgb);
    let (w2, o2, m2) = stat(&cpu.rgb, &front.rgb);
    let (w3, o3, m3) = stat(&behind.rgb, &front.rgb);
    eprintln!("full render, R6..R8 against CPU: worst {w1:e}, over 1e-3 {o1:e}, mean {m1:e}");
    eprintln!("full render, R1..R9 against CPU: worst {w2:e}, over 1e-3 {o2:e}, mean {m2:e}");
    eprintln!("full render, R1..R9 against R6..R8: worst {w3:e}, over 1e-3 {o3:e}, mean {m3:e}");
    // Twice the first measured values (R6..R8 behind the CPU's front: worst 1.5e-6, mean 3.4e-8;
    // R1..R9: worst 1.7e-6, mean 3.8e-8; R1..R9 against R6..R8: worst 8.3e-7, mean 2.1e-8; none over
    // 1e-3). With the vignetting gain formed in f32 the front differed from the CPU's by 4 ulp
    // everywhere and a few samples of the noise flipped a tie of the demosaic (worst 2.4e-3).
    // Since R6's denoise is method A (`docs/algorithms/05`, 2026-10-20) the device and the CPU part
    // where a coefficient lies within rounding of method A's threshold (f32 against f64): twice the
    // measured values, R6..R8 against the CPU worst 6.8e-4, mean 1.5e-7, R1..R9 against the CPU worst
    // 6.8e-4, mean 1.5e-7, none over 1e-3; R1..R9 against R6..R8 is unchanged (7.7e-7, 2.9e-8).
    assert!(w2 < 1.4e-3 && o2 <= 1e-5 && m2 < 3.1e-7, "{w2} {o2} {m2}");
    assert!(w3 < 1.7e-6 && o3 <= 1e-5 && m3 < 4.3e-8, "{w3} {o3} {m3}");
    assert!(w1 < 1.4e-3 && o1 <= 1e-5 && m1 < 3e-7, "{w1} {o1} {m1}");
}

/// The lens switches of the RAW tab cut the front down to what the device makes bit for bit.
#[test]
fn the_front_is_the_cpus_to_the_bit_where_it_has_no_lens_to_correct() {
    let Some(gpu) = gpu() else { return };
    let base = source(&clipped_scene(1300, 900, 5));
    let edit = |lens: bool, ca: bool| {
        let mut st = base.settings.clone();
        st.decode.lens_corrections = lens;
        st.decode.chromatic_aberration = ca;
        base.with_settings(st)
    };
    let cpu = renderer(None, GpuStages::NONE);
    let front = renderer(Some(&gpu), GpuStages::RAW_CHAIN.union(GpuStages::RAW_FRONT));
    let behind = renderer(Some(&gpu), GpuStages::RAW_CHAIN);
    let stat = |a: &[f32], b: &[f32]| {
        let (mut worst, mut over) = (0.0f64, 0usize);
        for (&x, &y) in a.iter().zip(b) {
            let d = f64::from((x - y).abs());
            worst = worst.max(d);
            over += usize::from(d > 1e-3);
        }
        (worst, over)
    };
    let mut table = Vec::new();
    for (lens, ca) in [(false, false), (true, false), (true, true)] {
        let src = edit(lens, ca);
        let c = render_full(&cpu, &src);
        let r6 = render_full(&behind, &src);
        let r1 = render_full(&front, &src);
        let (w6, o6) = stat(&c.rgb, &r6.rgb);
        let (w1, o1) = stat(&c.rgb, &r1.rgb);
        table.push(format!(
            "lens {lens} CA {ca}: R6..R8 against CPU worst {w6:e} ({o6} over 1e-3), R1..R9 worst {w1:e} ({o1} over 1e-3)"
        ));
        if !lens {
            // Without a lens table the front is R1, R4 and R5 only: the CPU's bits, so the demosaic
            // takes the CPU's decisions and the picture is the R6..R8 route's.
            assert_eq!(r1.rgb, r6.rgb, "lens off: the front changes nothing");
        }
        // With one it adds no flip of the demosaic to the CPU front's own (the vignetting gain is
        // the CPU's f32 to the bit, R5b and R9 move values by 1e-6 and no decision).
        assert!(
            o1 <= o6 + 2,
            "lens {lens} CA {ca}: {o1} samples over 1e-3 against {o6}"
        );
    }
    for line in &table {
        eprintln!("{line}");
    }
}

#[test]
fn two_photos_stay_on_the_device_and_the_third_pushes_the_oldest_out() {
    let Some(gpu) = gpu() else { return };
    // Three photos in one session, so that their ids are one session's.
    let dir = tempfile::tempdir().expect("temp dir");
    let scenes: Vec<Scene> = (0..3).map(|i| clipped_scene(640, 480, 10 + i)).collect();
    for (i, s) in scenes.iter().enumerate() {
        s.write(dir.path(), &format!("DSC0000{}.ARW", i + 1), MODEL);
    }
    let session = ym_core::session::Session::new(base_env());
    let out = session.open(
        &[dir.path().to_path_buf()],
        ym_core::session::OpenOptions::default(),
    );
    assert_eq!(out.added.len(), 3);
    settle(&session);
    let sources: Vec<PhotoSource> = out
        .added
        .iter()
        .map(|id| {
            PhotoSource::from_view(
                &session.view(*id).expect("view"),
                session.raw_image(*id).expect("raw image"),
            )
        })
        .collect();
    let ids: Vec<PhotoId> = sources.iter().map(|s| s.id).collect();
    let r = renderer(Some(&gpu), GpuStages::ALL);
    // Each use is a group that is not cached yet (another noise reduction is another key), so each
    // one needs the photo's words on the device.
    let count = [
        std::cell::Cell::new(0usize),
        std::cell::Cell::new(0),
        std::cell::Cell::new(0),
    ];
    let use_photo = |i: usize| {
        let n = count[i].get();
        count[i].set(n + 1);
        let mut st = sources[i].settings.clone();
        st.noise_reduction = AutoOr::Manual(NoiseReductionValue {
            luminance: 20.0 + 5.0 * n as f64,
            color: 50.0,
            detail: 50.0,
        });
        let f = fit_frame(&r, &sources[i].with_settings(st), 256.0, 192.0);
        assert_eq!(f.report.runs.device_groups, 1, "{:?}", f.report.notes);
        f
    };
    let uploads = |i: usize| r.counters(ids[i]).raw_uploads;
    use_photo(0);
    use_photo(1);
    assert_eq!(r.resident_photos(), 2);
    assert_eq!((uploads(0), uploads(1)), (1, 1));
    // The third one needs a place: the oldest (the first) goes.
    use_photo(2);
    assert_eq!(r.resident_photos(), 2, "never more than two");
    assert_eq!(uploads(2), 1);
    // Another group of the third one: its words are there.
    let again = use_photo(2);
    assert_eq!(again.report.runs.raw_uploads, 0);
    assert_eq!(uploads(2), 1, "no second upload");
    // The first one is uploaded again when it is used; the oldest of the other two goes.
    use_photo(0);
    assert_eq!(uploads(0), 2, "the first was released and is back");
    assert_eq!(r.resident_photos(), 2);
    // Navigation: the photo looked at and the one before it are the two that stay, whatever the
    // thumbnails of the folder touch.
    r.navigate(ids[0]);
    r.navigate(ids[1]);
    assert!(r.resident_photos() <= 2);
    use_photo(1);
    use_photo(0);
    assert_eq!(r.resident_photos(), 2);
    let (u0, u1) = (uploads(0), uploads(1));
    // A thumbnail-like use of the third photo is made from words that are let go of at once.
    use_photo(2);
    assert_eq!(r.resident_photos(), 2, "the two looked at stay");
    assert_eq!(uploads(2), 2, "uploaded for that group");
    // And the two stayed: using them again uploads nothing.
    let f = use_photo(1);
    assert_eq!(f.report.runs.raw_uploads, 0);
    use_photo(0);
    assert_eq!((uploads(0), uploads(1)), (u0, u1));
    // Closing the session lets everything go.
    r.retire_through(*ids.iter().max().expect("ids"));
    assert_eq!(r.resident_photos(), 0);
}

#[test]
fn a_raw_front_that_is_off_keeps_the_group_on_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let src = source(&clipped_scene(900, 600, 7));
    let no_front = renderer(
        Some(&gpu),
        VIEW.union(GpuStages::RAW_CHAIN)
            .union(GpuStages::EXPORT_TAIL),
    );
    let f = fit_frame(&no_front, &src, 300.0, 200.0);
    assert_eq!(f.report.runs.device_groups, 0);
    assert_eq!(f.report.runs.raw_uploads, 0);
    assert_eq!(f.report.runs.sigma_planes, 1, "the host's sigma plane");
    assert!(!mentions_the_card(&f), "{:?}", f.report.notes);
}

#[cfg(feature = "fault-injection")]
mod faults {
    use super::*;
    use ym_gpu::device::GpuFault;

    #[test]
    fn a_device_error_in_the_group_falls_back_to_the_cpu_with_the_note() {
        let src = source(&clipped_scene(900, 600, 9));
        let cpu = renderer(None, GpuStages::NONE);
        let want = fit_frame(&cpu, &src, 300.0, 200.0);
        for (name, fault) in [
            ("out of budget", GpuFault::OutOfBudget),
            ("shader", GpuFault::Shader),
        ] {
            let Some(gpu) = gpu() else { return };
            let dev = renderer(Some(&gpu), GpuStages::ALL);
            gpu.set_fault(Some(fault));
            let got = fit_frame(&dev, &src, 300.0, 200.0);
            gpu.set_fault(None);
            assert_eq!(got.report.runs.device_groups, 0, "{name}");
            assert_eq!(got.rgba, want.rgba, "{name}: the CPU frame, bit for bit");
            assert!(
                got.report
                    .notes
                    .iter()
                    .any(|n| n.starts_with("The graphics card could not render this frame")),
                "{name}: {:?}",
                got.report.notes
            );
        }
    }

    #[test]
    fn a_strict_policy_fails_the_request_instead() {
        let Some(gpu) = gpu() else { return };
        let src = source(&clipped_scene(900, 600, 9));
        let strict = Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                gpu: Some(Arc::clone(&gpu)),
                gpu_stages: GpuStages::ALL,
                gpu_strict: true,
                ..RenderOptions::default()
            },
        )
        .expect("render thread");
        gpu.set_fault(Some(GpuFault::Shader));
        let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
        let result =
            strict.viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new());
        gpu.set_fault(None);
        assert!(
            matches!(result, Err(ym_core::pipeline::RenderError::Gpu(_))),
            "{:?}",
            result.map(|f| f.report.notes)
        );
    }
}
