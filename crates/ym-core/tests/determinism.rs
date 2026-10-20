//! The same photo, settings and version give the same pixels whatever the thread count
//! (`AGENTS.md` hard rule 8). Every render below runs on a rayon pool of 1, 2 and 4 threads with
//! the colour pass on the same number of threads, on a photo with noise, clipping, a vignetting
//! table and an odd orientation, and the results are compared bit for bit.
mod common;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, PhotoSource, RenderOptions, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::{Crop, CropAspect, NormRect};
use ym_gpu::queue::CancelToken;
use ym_raw::testgen::LensTables;

const MODEL: &str = "ILCE-7RM4";

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    // Bright enough that the patchwork's brightest patches clip, with real noise.
    let scene = Scene::new(900, 600, white, Layout::Patchwork, 1.4)
        .noise(4e-4, 3e-6)
        .seed(9);
    let dir = tempfile::tempdir().expect("dir");
    let mut vign = vec![16i16];
    vign.extend((0..16).map(|i| (i * i * 9) as i16));
    scene
        .builder(MODEL)
        .orientation(6)
        .iso(1600)
        .lens_tables(LensTables {
            vignetting: Some(vign),
            chromatic_aberration: Some({
                let mut v = vec![16i16];
                v.extend((0..16).map(|i| (i - 8) * 6));
                v
            }),
            distortion: Some({
                let mut v = vec![16i16];
                v.extend((0..16).map(|i| 10 + 12 * i));
                v
            }),
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    Developed::open(dir)
}

fn renderer(threads: usize) -> Renderer {
    Renderer::new(
        std::sync::Arc::new(ym_core::pipeline::NeutralOnly),
        RenderOptions {
            threads,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// Everything one run produces, to compare between runs.
#[derive(PartialEq, Debug)]
struct Outputs {
    fit: Vec<u8>,
    quarter: Vec<u8>,
    tiles: Vec<u8>,
    thumbnail: Vec<u8>,
    histogram: Vec<u64>,
    full: Vec<u32>,
    tilted: Vec<u8>,
}

fn run(src: &PhotoSource, threads: usize) -> Outputs {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");
    pool.install(|| {
        let r = renderer(threads);
        let cancel = CancelToken::new();
        let image = r.final_size(src, &cancel).expect("size");
        let cell = |w, h| ViewportState::new(ViewportRect::cell(w, h, 1.0));
        let frame = |s: &PhotoSource, st: &ViewportState| {
            r.viewport_frame(s, st, &FrameOptions::default(), &cancel)
                .expect("frame")
                .rgba
        };
        let fit = frame(src, &cell(300.0, 450.0));
        let quarter = frame(src, &cell(120.0, 180.0));
        let mut z = cell(300.0, 300.0);
        z.set_mode(ViewportMode::Percent { value: 100.0 }, image);
        let tiles = frame(src, &z);
        let thumbnail = r.thumbnail_png(src, &cancel).expect("thumbnail");
        let h = r.histogram(src, &cancel).expect("histogram");
        let histogram = h
            .red
            .iter()
            .chain(&h.green)
            .chain(&h.blue)
            .map(|v| v.to_bits())
            .collect();
        let full = r
            .render_full(
                src,
                &FullRequest {
                    target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                },
                &cancel,
            )
            .expect("full")
            .rgb
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut s = src.settings.clone();
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.1,
                y: 0.1,
                width: 0.8,
                height: 0.8,
            },
            angle_deg: 7.5,
            aspect: CropAspect::Free,
        });
        s.contrast = ym_core::settings::PercentSetting::manual_value(20.0);
        let tilted = frame(&src.with_settings(s), &cell(300.0, 450.0));
        Outputs {
            fit,
            quarter,
            tiles,
            thumbnail,
            histogram,
            full,
            tilted,
        }
    })
}

#[test]
fn renders_are_identical_for_one_two_and_four_threads() {
    let d = photo();
    let src = d.source();
    let one = run(&src, 1);
    for threads in [2usize, 4] {
        let other = run(&src, threads);
        assert!(
            one.fit == other.fit,
            "fit view differs with {threads} threads"
        );
        assert!(
            one.quarter == other.quarter,
            "quarter view differs with {threads} threads"
        );
        assert!(
            one.tiles == other.tiles,
            "100 % tiles differ with {threads} threads"
        );
        assert!(
            one.thumbnail == other.thumbnail,
            "thumbnail differs with {threads} threads"
        );
        assert!(
            one.histogram == other.histogram,
            "histogram differs with {threads} threads"
        );
        assert!(
            one.full == other.full,
            "full render differs with {threads} threads"
        );
        assert!(
            one.tilted == other.tilted,
            "straightened view differs with {threads} threads"
        );
    }
    // And a second run on the same pool size is the same too (nothing depends on timing).
    let again = run(&src, 1);
    assert!(one == again, "two runs on one thread differ");
}

#[test]
fn a_cached_frame_is_the_frame_that_was_computed() {
    let d = photo();
    let src = d.source();
    let r = renderer(0);
    let cancel = CancelToken::new();
    let st = ViewportState::new(ViewportRect::cell(300.0, 450.0, 1.0));
    let a = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("a");
    let b = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("b");
    assert_eq!(b.report.runs.frame_hits, 1);
    assert_eq!(a.rgba, b.rgba);
    let c = r
        .viewport_frame(
            &src,
            &st,
            &FrameOptions {
                no_cache: true,
                ..FrameOptions::default()
            },
            &cancel,
        )
        .expect("c");
    assert_eq!(a.rgba, c.rgba, "recomputing gives the cached bytes");
}
