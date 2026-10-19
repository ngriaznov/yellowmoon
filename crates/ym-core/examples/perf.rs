//! Timings of the render pipeline on a synthetic 24 MP frame (`docs/01` Preview strategy, `docs/05`
//! Performance targets).
//!
//! ```text
//! CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false \
//! CARGO_PROFILE_DEV_OVERFLOW_CHECKS=false \
//!   cargo run -p ym-core --example perf -- [--mp 24] [--gpu] [--keep <folder>] [--demosaic <method>]
//! ```
//!
//! `--demosaic <method>` sets the R6 method of the full-resolution renders (the 100 % view, the
//! full render; the fit view and the thumbnails bin the mosaic and do not demosaic) by its name
//! (`adaptive_ratio`, `menon`, ...). The default is the renderer's default, the adaptive ratio
//! demosaic; running the example once with `--demosaic menon` gives the numbers from before the
//! default changed (`docs/05` section R6, owner's decision of 2026-10-13).
//!
//! Writes one uncompressed 14-bit ARW (a gradient with noise, lens tables, ISO 400) into a
//! temporary folder (`--keep <folder>` writes it there instead and leaves it, for timing `ym
//! render` on the same file), develops it, and times: the fit preview (cold and warm), a colour-only edit,
//! a white-balance edit, the 100 % view, a thumbnail and the full-resolution render. `--gpu` also
//! runs the colour pass on the first wgpu adapter (Mesa lavapipe here: its numbers say nothing
//! about Metal).

use std::sync::Arc;
use std::time::{Duration, Instant};

use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions, Renderer,
    ViewportMode, ViewportRect, ViewportState,
};
use ym_core::session::{OpenOptions, Session, SessionEnv};
use ym_core::settings::{AutoOr, WhiteBalanceValue};
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::device::{Gpu, GpuConfig};
use ym_gpu::queue::CancelToken;
use ym_raw::testgen::{ArwBuilder, LensTables, gradient_mosaic};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mp: f64 = args
        .iter()
        .position(|a| a == "--mp")
        .map_or(24.0, |i| args[i + 1].parse().expect("megapixels"));
    let use_gpu = args.iter().any(|a| a == "--gpu");
    let demosaic = args
        .iter()
        .position(|a| a == "--demosaic")
        .map_or(DEFAULT_METHOD, |i| {
            let name = args.get(i + 1).expect("a demosaic method name");
            Method::from_name(name).unwrap_or_else(|| {
                panic!(
                    "{name:?} is not a demosaic method (one of {:?})",
                    Method::ALL.map(Method::name)
                )
            })
        });
    println!(
        "R6 demosaic method of the full-resolution renders: {}",
        demosaic.name()
    );
    let (w, h) = {
        let w = ((mp * 1e6 * 1.5).sqrt() / 2.0).round() as u32 * 2;
        (w, w * 2 / 3 / 2 * 2)
    };
    let temp = tempfile::tempdir().expect("temp dir");
    let dir_path = args.iter().position(|a| a == "--keep").map_or_else(
        || temp.path().to_path_buf(),
        |i| {
            let folder = std::path::PathBuf::from(args.get(i + 1).expect("a folder after --keep"));
            std::fs::create_dir_all(&folder).expect("create the --keep folder");
            folder
        },
    );
    let t = Instant::now();
    let mosaic = gradient_mosaic(w, h, 512, 15500, 7);
    let path = dir_path.join("DSC00001.ARW");
    ArwBuilder::new("ILCE-7M3", w, h)
        .black([512; 4])
        .white([16300; 3])
        .wb(Some([2100, 1024, 1024, 1500]))
        .iso(400)
        .lens_tables(LensTables::default())
        .write_to(&path, &mosaic)
        .expect("write");
    println!(
        "synthetic {w} x {h} ({:.1} MP) written in {:.0} ms",
        f64::from(w) * f64::from(h) / 1e6,
        ms(t.elapsed())
    );
    drop(mosaic);

    let session = Session::new(SessionEnv::from_data_dir().expect("data folder"));
    let t = Instant::now();
    let out = session.open(std::slice::from_ref(&dir_path), OpenOptions::default());
    assert!(session.wait_idle(Duration::from_secs(600)));
    println!(
        "open and develop (decode, estimates): {:.0} ms",
        ms(t.elapsed())
    );
    let id = out.added[0];
    let t = Instant::now();
    let view = session.view(id).expect("view");
    let raw = session.raw_image(id).expect("raw");
    println!("view and cached mosaic: {:.1} ms", ms(t.elapsed()));
    let src = PhotoSource::from_view(&view, raw);

    let gpu = use_gpu.then(|| {
        Arc::new(Gpu::start(GpuConfig::default()).expect("a GPU adapter (WGPU_BACKEND=vulkan)"))
    });
    if let Some(g) = &gpu {
        println!("{}", g.info());
    }
    for (label, gpu) in [("CPU", None), ("GPU", gpu)] {
        if label == "GPU" && gpu.is_none() {
            continue;
        }
        println!("\n== colour pass on the {label} ==");
        let r = Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                gpu,
                demosaic,
                ..RenderOptions::default()
            },
        )
        .expect("render thread");
        let cancel = CancelToken::new();
        let frame = |s: &PhotoSource, st: &ViewportState| {
            let t = Instant::now();
            let f = r
                .viewport_frame(
                    s,
                    st,
                    &FrameOptions {
                        no_cache: true,
                        ..FrameOptions::default()
                    },
                    &cancel,
                )
                .expect("frame");
            (ms(t.elapsed()), f)
        };
        for (name, cw, ch) in [
            ("1440 x 960 px cell", 1440.0, 960.0),
            ("2880 x 1920 px cell (retina)", 2880.0, 1920.0),
        ] {
            let st = ViewportState::new(ViewportRect::cell(cw, ch, 1.0));
            r.forget(src.id);
            let (cold, f) = frame(&src, &st);
            println!(
                "fit preview, {name}: cold {cold:.0} ms (1/{} resolution, {} device)",
                f.report.scale,
                if f.report.device == ym_core::pipeline::ComputeDevice::Gpu {
                    "GPU"
                } else {
                    "CPU"
                }
            );
            println!("    {:?}", f.report.timings);
            let (warm, f) = frame(&src, &st);
            println!("  warm (same settings, frame cache off): {warm:.0} ms");
            println!("    {:?}", f.report.timings);
            // Colour-only edits: each iteration moves one grade control.
            let mut worst: f64 = 0.0;
            let mut best = f64::MAX;
            for i in 1..=5 {
                let mut s = src.settings.clone();
                s.contrast = ym_core::settings::PercentSetting::manual_value(5.0 * f64::from(i));
                let (t, f) = frame(&src.with_settings(s), &st);
                assert_eq!(f.report.runs.raw_work(), 0);
                worst = worst.max(t);
                best = best.min(t);
            }
            println!(
                "  colour-only edit (grade group only): best {best:.0} ms, worst {worst:.0} ms"
            );
            let mut s = src.settings.clone();
            s.white_balance = AutoOr::Manual(WhiteBalanceValue {
                kelvin: 3800.0,
                tint: 5.0,
            });
            let (t, _) = frame(&src.with_settings(s), &st);
            println!("  white-balance edit (colour and grade groups): {t:.0} ms");
        }
        if label == "CPU" {
            output_sharpening_rows(&r, &src, &cancel);
        }
        // 100 % view of a 1440 x 960 cell.
        let image = r.final_size(&src, &cancel).expect("size");
        let mut st = ViewportState::new(ViewportRect::cell(1440.0, 960.0, 1.0));
        st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
        r.forget(src.id);
        let (cold, f) = frame(&src, &st);
        println!(
            "100 % view, 1440 x 960: cold {cold:.0} ms ({} tiles)",
            f.report.runs.tiles
        );
        let (warm, _) = frame(&src, &st);
        println!("  warm (tiles cached): {warm:.0} ms");
        // Thumbnail.
        r.forget(src.id);
        let t = Instant::now();
        let png = r.thumbnail_png(&src, &cancel).expect("thumbnail");
        println!(
            "thumbnail, cold: {:.0} ms ({} bytes)",
            ms(t.elapsed()),
            png.len()
        );
        let t = Instant::now();
        let _ = r.thumbnail_png(&src, &cancel).expect("thumbnail");
        println!("thumbnail, warm raw group: {:.0} ms", ms(t.elapsed()));
        if label == "CPU" {
            // Full resolution.
            r.forget(src.id);
            let t = Instant::now();
            let full = r
                .render_full(
                    &src,
                    &FullRequest {
                        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                    },
                    &cancel,
                )
                .expect("full render");
            println!(
                "full-resolution render {} x {} (classical, display-linear f32): {:.2} s, ledger peak {:.0} MiB",
                full.width,
                full.height,
                t.elapsed().as_secs_f64(),
                r.ledger().peak() as f64 / 1048576.0
            );
        }
    }
}

/// Output sharpening, the last stage (`docs/01` Render graph, `docs/algorithms/04`): what moving the
/// Output slider costs on a viewport frame whose display-linear picture is cached (the stage, then
/// C13 and the letterbox), for a 2 MP fit frame and for a 512 px tile of the 100 % view, against
/// the same frame with the stage off. The default amount is 100, so every other number of this
/// example includes the stage; this section is the only one that separates it.
fn output_sharpening_rows(r: &Renderer, src: &PhotoSource, cancel: &CancelToken) {
    println!("\n-- output sharpening (CPU; the stage and C13 on a cached display-linear frame) --");
    let with_amount = |pct: f64| {
        let mut s = src.settings.clone();
        s.output_sharpening = pct;
        src.with_settings(s)
    };
    let cached = |s: &PhotoSource, st: &ViewportState| {
        let t = Instant::now();
        let f = r
            .viewport_frame(s, st, &FrameOptions::default(), cancel)
            .expect("frame");
        (ms(t.elapsed()), f)
    };
    let image = r.final_size(src, cancel).expect("size");
    let fit = ViewportState::new(ViewportRect::cell(1730.0, 1155.0, 1.0));
    let mut tile = ViewportState::new(ViewportRect::cell(512.0, 512.0, 1.0));
    tile.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    for (name, st) in [
        ("2 MP fit frame, 1730 x 1155", &fit),
        ("512 px tile at 100 %", &tile),
    ] {
        let (cold, f) = cached(&with_amount(100.0), st);
        println!(
            "{name}: first frame {cold:.0} ms (1/{} resolution, frame {} x {})",
            f.report.scale, f.width, f.height
        );
        // Moving the slider: the display-linear frame is served from its cache.
        let mut times = Vec::new();
        let mut stage = Vec::new();
        for pct in [95.0, 90.0, 85.0, 80.0, 75.0] {
            let (t, f) = cached(&with_amount(pct), st);
            assert_eq!(
                (f.report.runs.display_hits, f.report.runs.output_passes),
                (1, 1)
            );
            assert_eq!(f.report.runs.raw_work() + f.report.runs.color_passes, 0);
            times.push(t);
            stage.push(f.report.timings.output_ms);
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        println!(
            "  slider move: {:.0} ms per frame ({:.0} ms in the stage and C13), {:.0} to {:.0} ms",
            mean(&times),
            mean(&stage),
            times.iter().copied().fold(f64::MAX, f64::min),
            times.iter().copied().fold(0.0, f64::max)
        );
        let (off, f) = cached(&with_amount(0.0), st);
        println!(
            "  the same frame with the stage off (the plain road: sample and colour pass): {off:.0} ms, colour pass {:.0} ms",
            f.report.timings.pass_ms
        );
    }
}
