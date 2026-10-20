//! Renders a photo to PNG files and prints the timings of each preview (a developer tool, also the
//! harness behind the performance numbers in `docs/01`).
//!
//! ```text
//! cargo run -p ym-core --example render_demo -- <file.ARW> <out-prefix> [width height scale]
//! ```
//!
//! Writes `<prefix>-fit.png` (the viewport at Fit), `<prefix>-100.png` (the middle of the photo
//! at 100 %), `<prefix>-thumb.png`, and prints the stage runs and milliseconds of each.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, PhotoSource, RenderOptions, Renderer, ViewportInput,
    ViewportMode, ViewportRect, ViewportState,
};
use ym_core::session::{OpenOptions, Session, SessionEnv};
use ym_gpu::queue::CancelToken;

fn write_png(path: &str, w: usize, h: usize, rgba: &[u8]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(rgba).expect("png data");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let file = PathBuf::from(args.next().expect("a .ARW file"));
    let prefix = args.next().expect("an output prefix");
    let w: f64 = args.next().map_or(1440.0, |v| v.parse().expect("width"));
    let h: f64 = args.next().map_or(960.0, |v| v.parse().expect("height"));
    let scale: f64 = args.next().map_or(1.0, |v| v.parse().expect("scale"));

    let env = SessionEnv::from_data_dir().expect("the data folder");
    let session = Session::new(env);
    let t = Instant::now();
    let out = session.open(&[file], OpenOptions::default());
    assert_eq!(out.added.len(), 1, "one ARW file");
    assert!(session.wait_idle(Duration::from_secs(300)));
    let id = out.added[0];
    println!("opened and developed in {:?}", t.elapsed());
    let view = session.view(id).expect("view");
    println!(
        "camera {:?}, auto: {:?}",
        view.camera.info(),
        view.estimates.summary()
    );
    let t = Instant::now();
    let raw = session.raw_image(id).expect("raw");
    println!("decode {:?}", t.elapsed());
    let src = PhotoSource::from_view(&view, Arc::clone(&raw));

    let renderer = Renderer::new(
        Arc::new(ym_core::pipeline::NeutralOnly),
        RenderOptions::default(),
    )
    .expect("render thread");
    let cancel = CancelToken::new();
    let mut state = ViewportState::new(ViewportRect::cell(w / scale, h / scale, scale));

    let t = Instant::now();
    let f = renderer
        .viewport_frame(&src, &state, &FrameOptions::default(), &cancel)
        .expect("fit frame");
    println!(
        "fit frame {}x{} in {:?}: {:?}",
        f.width,
        f.height,
        t.elapsed(),
        f.report
    );
    write_png(&format!("{prefix}-fit.png"), f.width, f.height, &f.rgba);

    // A colour-only edit: another look intensity changes only the grade group.
    let mut edited = src.clone();
    edited.settings.contrast = ym_core::settings::PercentSetting::manual_value(25.0);
    let t = Instant::now();
    let f2 = renderer
        .viewport_frame(&edited, &state, &FrameOptions::default(), &cancel)
        .expect("edited frame");
    println!("colour-only edit {:?}: {:?}", t.elapsed(), f2.report.runs);

    // 100 %.
    let image = renderer.final_size(&src, &cancel).expect("size");
    state.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    if let Ok(c) = std::env::var("CENTRE") {
        let v: Vec<f64> = c.split(',').map(|v| v.parse().expect("number")).collect();
        state.centre = [v[0], v[1]];
        state.set_rect(state.rect, image);
    }
    let t = Instant::now();
    let f = renderer
        .viewport_frame(&src, &state, &FrameOptions::default(), &cancel)
        .expect("100% frame");
    println!("100% frame in {:?}: {:?}", t.elapsed(), f.report);
    write_png(&format!("{prefix}-100.png"), f.width, f.height, &f.rgba);
    let _ = ViewportInput::Pan { dx: 0.0, dy: 0.0 };

    let t = Instant::now();
    let png = renderer.thumbnail_png(&src, &cancel).expect("thumbnail");
    println!("thumbnail {} bytes in {:?}", png.len(), t.elapsed());
    std::fs::write(format!("{prefix}-thumb.png"), png).expect("write thumbnail");

    if std::env::var_os("FULL").is_some() {
        let t = Instant::now();
        let full = renderer
            .render_full(
                &src,
                &FullRequest {
                    target: FullTarget::DisplayLinear(ym_color::display::OutputTarget::Srgb),
                },
                &cancel,
            )
            .expect("full render");
        println!(
            "full render {}x{} in {:?}: {:?}",
            full.width,
            full.height,
            t.elapsed(),
            full.report
        );
    }
}
