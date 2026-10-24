mod common;
use std::sync::Arc;
use common::*;
use ym_core::pipeline::{FrameOptions, NeutralOnly, RenderOptions, Renderer, ViewportRect, ViewportState};
use ym_core::settings::{LensKind, LensSettings};
use ym_gpu::queue::CancelToken;
use ym_lens::LensLibrary;

#[test]
fn probe() {
    let lib = Arc::new(LensLibrary::load(&data_root()).unwrap());
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions { lens: Some(lib), data_root: Some(data_root()), ..RenderOptions::default() }).unwrap();
    let w: u32 = std::env::var("W").ok().and_then(|v| v.parse().ok()).unwrap_or(800);
    let h: u32 = std::env::var("H").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let scene = Scene::new(w, h, light("sony-ilce-7rm4", 5500.0), Layout::Patchwork, 1.6).noise(2e-4, 1e-6);
    let d = Developed::new(&scene, "ILCE-7RM4");
    let src = d.source();
    let st = ViewportState::new(ViewportRect::cell(f64::from(w) / 2.0, f64::from(h) / 2.0, 1.0));
    let t = std::time::Instant::now();
    let f = r.viewport_frame(&src, &st, &FrameOptions { no_cache: true, ..Default::default() }, &CancelToken::new()).unwrap();
    eprintln!("no lens: {:?} total {:?}", f.report.timings, t.elapsed());
    for (name, lens) in [
        ("flares", LensSettings { kind: LensKind::Vintage, flares: 60.0, edges: 0.0, anamorphic: 0.0 }),
        ("edges", LensSettings { kind: LensKind::Vintage, flares: 0.0, edges: 50.0, anamorphic: 0.0 }),
        ("anam", LensSettings { kind: LensKind::Vintage, flares: 0.0, edges: 0.0, anamorphic: 50.0 }),
        ("all", LensSettings { kind: LensKind::Vintage, flares: 60.0, edges: 50.0, anamorphic: 50.0 }),
        ("all2", LensSettings { kind: LensKind::Vintage, flares: 61.0, edges: 50.0, anamorphic: 50.0 }),
    ] {
        let mut s = src.settings.clone();
        s.lens = lens;
        let s2 = src.with_settings(s);
        let t = std::time::Instant::now();
        let f = r.viewport_frame(&s2, &st, &FrameOptions { no_cache: true, ..Default::default() }, &CancelToken::new()).unwrap();
        eprintln!("{name}: {:?} runs {:?} total {:?}", f.report.timings, f.report.runs, t.elapsed());
    }
}
