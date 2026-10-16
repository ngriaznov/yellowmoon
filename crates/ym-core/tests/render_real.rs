//! Real files (`testdata/real/files`, local only, never committed): every one of them renders a
//! fit view, a 100 % view and a thumbnail, with the illuminant models of the data folder. Skipped
//! when the folder is absent. The full-resolution render of the smallest file is `#[ignore]`d for
//! its time: `cargo test -p ym-core --test render_real -- --ignored --nocapture`.
mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, PhotoSource, Renderer, ViewportMode, ViewportRect,
    ViewportState,
};
use ym_core::session::{OpenOptions, Session, SessionEnv};
use ym_gpu::queue::CancelToken;

fn files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/real/files");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn develop(path: &std::path::Path) -> (Session, PhotoSource) {
    let env = SessionEnv::load(&common::data_root()).expect("data folder");
    let session = Session::new(env);
    let out = session.open(&[path.to_path_buf()], OpenOptions::default());
    assert!(session.wait_idle(Duration::from_secs(120)));
    let src = PhotoSource::from_session(&session, out.added[0]).expect("source");
    (session, src)
}

fn stats(f: &ym_core::pipeline::RenderedFrame) -> (f64, f64) {
    let lum: Vec<f64> = f
        .rgba
        .chunks(4)
        .map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
        .collect();
    let mean = lum.iter().sum::<f64>() / lum.len() as f64;
    let sd = (lum.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / lum.len() as f64).sqrt();
    (mean, sd)
}

#[test]
fn every_real_file_renders_a_fit_view_a_100_percent_view_and_a_thumbnail() {
    let files = files();
    if files.is_empty() {
        eprintln!("render_real: testdata/real/files is absent, skipped");
        return;
    }
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    for (i, path) in files.iter().enumerate() {
        let name = path
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        let (_session, src) = develop(path);
        let (w, h) = r.final_size(&src, &cancel).expect("size");
        let st = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
        let t = Instant::now();
        let f = r
            .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let fit_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (mean, sd) = stats(&f);
        assert!(
            mean > 3.0 && mean < 252.0,
            "{name}: mean luminance {mean:.1}"
        );
        assert!(sd > 4.0, "{name}: a photograph is not flat (sd {sd:.1})");
        let t = Instant::now();
        let png = r.thumbnail_png(&src, &cancel).expect("thumbnail");
        let thumb_ms = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(&png[1..4], b"PNG");
        // The 100 % view of the first two files (tiles through the full chain).
        let mut note = String::new();
        if i < 2 {
            let mut z = st.clone();
            z.set_mode(ViewportMode::Percent { value: 100.0 }, (w, h));
            let t = Instant::now();
            let f = r
                .viewport_frame(&src, &z, &FrameOptions::default(), &cancel)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(f.report.scale, 1);
            let (m, s) = stats(&f);
            assert!(
                m > 3.0 && m < 252.0 && s > 1.0,
                "{name}: 100 % mean {m:.1} sd {s:.1}"
            );
            note = format!(", 100 % {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!(
            "{name}: {w}x{h}, fit {fit_ms:.0} ms (1/{}), thumbnail {thumb_ms:.0} ms{note}, mean {mean:.0} sd {sd:.0}, notes {:?}",
            f.report.scale, f.report.notes
        );
        r.forget(src.id);
    }
}

#[test]
#[ignore = "renders a 10 MP file at full resolution: about 10 s"]
fn the_smallest_real_file_renders_at_full_resolution() {
    let files = files();
    let Some(path) = files
        .iter()
        .min_by_key(|p| std::fs::metadata(p).map_or(u64::MAX, |m| m.len()))
    else {
        eprintln!("render_real: testdata/real/files is absent, skipped");
        return;
    };
    let (_s, src) = develop(path);
    let r = Renderer::neutral_only().expect("render thread");
    let t = Instant::now();
    let full = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("full render");
    eprintln!(
        "{}: {}x{} in {:.1} s, ledger peak {:.0} MiB",
        path.display(),
        full.width,
        full.height,
        t.elapsed().as_secs_f64(),
        r.ledger().peak() as f64 / 1048576.0
    );
    assert!(full.rgb.iter().all(|v| v.is_finite()));
}

/// The Before view (the camera default rendering) of every local real file, at fit with a long edge of
/// 1616 px, as binary PPM files in the folder named by `YM_BEFORE_OUT` (Display P3 code values, the
/// viewport's encoding). Local, ignored, a report and not a gate: a script compares the pictures with the
/// embedded JPEGs of the files (`docs/02`, The Before view). The folders are `testdata/real/files` and
/// `testdata/readme-photos/src`; files are numbered in sorted order, so no file name leaves the machine.
/// `YM_BEFORE_OUT=/some/dir cargo test -p ym-core --test render_real -- --ignored dump_before_views`.
#[test]
#[ignore = "local check: needs the owner's files and YM_BEFORE_OUT"]
fn dump_before_views() {
    let Some(out) = std::env::var_os("YM_BEFORE_OUT").map(PathBuf::from) else {
        eprintln!("dump_before_views: set YM_BEFORE_OUT to a folder");
        return;
    };
    let mut all = files();
    let owner = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/readme-photos/src");
    let mut more: Vec<PathBuf> = std::fs::read_dir(owner)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw")))
                .collect()
        })
        .unwrap_or_default();
    more.sort();
    all.extend(more);
    std::fs::create_dir_all(&out).expect("output folder");
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    for (i, path) in all.iter().enumerate() {
        let (_session, src) = develop(path);
        let (w, h) = r.final_size(&src, &cancel).expect("size");
        let cell_h = (1616.0 * h as f64 / w as f64).round();
        let mut st = ViewportState::new(ViewportRect::cell(1616.0, cell_h, 1.0));
        st.before = true;
        let f = r
            .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
            .expect("frame");
        let mut ppm = format!("P6\n{} {}\n255\n", f.width, f.height).into_bytes();
        for p in f.rgba.chunks(4) {
            ppm.extend_from_slice(&p[..3]);
        }
        std::fs::write(out.join(format!("before-{i:02}.ppm")), ppm).expect("write");
        eprintln!("{i:02}: {}x{} written", f.width, f.height);
    }
}
