//! Timings of the render pipeline on a synthetic 24 MP frame (`docs/01` Preview strategy, `docs/05`
//! Performance targets).
//!
//! ```text
//! CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false \
//! CARGO_PROFILE_DEV_OVERFLOW_CHECKS=false \
//!   cargo run -p ym-core --example perf -- [--mp 24] [--gpu] [--keep]
//! ```
//!
//! Writes one uncompressed 14-bit ARW (a gradient with noise, lens tables, ISO 400) into a
//! temporary folder, develops it, and times: the fit preview (cold and warm), a colour-only edit,
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
    let (w, h) = {
        let w = ((mp * 1e6 * 1.5).sqrt() / 2.0).round() as u32 * 2;
        (w, w * 2 / 3 / 2 * 2)
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let t = Instant::now();
    let mosaic = gradient_mosaic(w, h, 512, 15500, 7);
    let path = dir.path().join("DSC00001.ARW");
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
    let out = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
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
                s.contrast = 5.0 * f64::from(i);
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
