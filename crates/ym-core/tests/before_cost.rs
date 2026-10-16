//! What it costs to flip to the Before view: the time to its first frame and the memory it adds.
//!
//! The Before view leaves out the lens vignetting correction (R3, `docs/02` *The Before view*), so its raw
//! front is its own: its own plan, its own prepared chain and its own cache entries, keyed apart from the
//! final picture's (`docs/01`, the cache table). This local check runs one scenario per process, so that the
//! resident size is clean, on one of the local photos (`testdata/real/files`, `testdata/readme-photos/src`;
//! never committed, skipped when absent) and prints one line of aggregate numbers.
//!
//! ```text
//! YM_COST_PHOTO=ILCE-7M3 YM_COST_BACKEND=cpu YM_COST_ZOOM=fit \
//!   cargo test -p ym-core --test before_cost -- --ignored --nocapture
//! ```
//!
//! `YM_COST_PHOTO` is a body name (the file stem); `YM_COST_BACKEND` is `cpu` or `gpu`; `YM_COST_ZOOM` is `fit`
//! or `100`. Steps: the final picture cold, the final picture again, Before for the first time, Before again,
//! the final picture once more. Not a gate.
mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ym_core::pipeline::{
    FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportMode, ViewportRect,
    ViewportState,
};
use ym_core::session::{OpenOptions, Session, SessionEnv};
use ym_gpu::device::{Gpu, GpuConfig};
use ym_gpu::queue::CancelToken;

fn find(stem: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    ["real/files", "readme-photos/src"]
        .iter()
        .map(|d| root.join(d).join(format!("{stem}.ARW")))
        .find(|p| p.exists())
}

/// The resident set and its peak, in MiB, from the kernel's own account of this process.
fn rss_mib() -> (f64, f64) {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    (kb("VmRSS:"), kb("VmHWM:"))
}

fn develop(path: &std::path::Path) -> (Session, PhotoSource) {
    let env = SessionEnv::load(&common::data_root()).expect("data folder");
    let session = Session::new(env);
    let out = session.open(&[path.to_path_buf()], OpenOptions::default());
    assert!(session.wait_idle(Duration::from_secs(120)));
    let src = PhotoSource::from_session(&session, out.added[0]).expect("source");
    (session, src)
}

#[test]
#[ignore = "local measurement: needs a local photo and YM_COST_PHOTO"]
fn flipping_to_before_costs() {
    let Some(stem) = std::env::var("YM_COST_PHOTO").ok() else {
        eprintln!("before_cost: set YM_COST_PHOTO (a body name), YM_COST_BACKEND and YM_COST_ZOOM");
        return;
    };
    let Some(path) = find(&stem) else {
        eprintln!("before_cost: {stem} is not among the local photos, skipped");
        return;
    };
    let backend = std::env::var("YM_COST_BACKEND").unwrap_or_else(|_| "cpu".into());
    let zoom = std::env::var("YM_COST_ZOOM").unwrap_or_else(|_| "fit".into());
    let gpu = (backend == "gpu").then(|| {
        Arc::new(Gpu::start(GpuConfig::default()).expect("a graphics device for the gpu backend"))
    });
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let (_session, src) = develop(&path);
    let cancel = CancelToken::new();
    let (w, h) = r.final_size(&src, &cancel).expect("size");
    let mut st = ViewportState::new(ViewportRect::cell(1616.0, 1010.0, 1.0));
    if zoom == "100" {
        st.set_mode(ViewportMode::Percent { value: 100.0 }, (w, h));
    }
    let baseline = rss_mib().0;
    let mut line =
        format!("{stem} {w}x{h} {backend} {zoom}: rss before any render {baseline:.0} MiB;");
    let steps = [
        ("final", false),
        ("final again", false),
        ("Before", true),
        ("Before again", true),
        ("final once more", false),
    ];
    for (name, before) in steps {
        let mut s = st.clone();
        s.before = before;
        let t = Instant::now();
        let f = r
            .viewport_frame(&src, &s, &FrameOptions::default(), &cancel)
            .expect("frame");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let g = r.graph(src.id);
        let (rss, peak) = rss_mib();
        line.push_str(&format!(
            " | {name}: {ms:.0} ms, scale 1/{}, caches {:.0} MiB, ledger {:.0} MiB, rss {rss:.0} (peak {peak:.0}) MiB, runs {:?}",
            f.report.scale,
            g.bytes() as f64 / 1048576.0,
            r.ledger().used() as f64 / 1048576.0,
            f.report.runs,
        ));
    }
    eprintln!("COST {line}");
}
