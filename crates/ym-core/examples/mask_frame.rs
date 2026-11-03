//! Where a masked viewer frame spends its time (`docs/23` section 5.9, the node view's 100 ms target).
//!
//! ```text
//! YELLOWMOON_DATA=$PWD/data cargo run --release -p ym-core --example mask_frame -- \
//!     <file.ARW> [--gpu] [--cases a,b,c,d,e,e2] [--frames 5] [--zoom fit|100|both] [--mask-view] [--no-kept-planes]
//! ```
//!
//! Opens one real photo, then renders viewer frames of a 1600 x 1060 px cell the way the node view does
//! while a slider of node 2 moves (`trim.exposure_ev` changes between frames: a new grade, the same raster
//! rows) or while a zoomed view is panned (a few hundred pixels between frames, the grade fixed). The
//! `ym::perf` lines of the core (`ym_core::perf`) are printed as they come, between `CASE` markers, and a
//! summary of each case ends with the median of the frames' wall time. Nothing is saved.
//!
//! Cases: `a` no masks; `b` one Radial; `c` one Brush (Feather 0.3, Refine edges 0.5); `d` Background;
//! `e` 8 nodes with a Radial and a Brush each; `e2` 8 nodes with a Radial and a Linear each (shapes only).
//! `--mask-view` adds Overlay and Mask only frames for `b` and `c`. `--gpu` runs the colour pass on the
//! first wgpu adapter (`WGPU_BACKEND=vulkan` for Mesa lavapipe: its numbers say nothing about Metal).

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    FrameOptions, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions, Renderer,
    ViewportInput, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::session::{OpenOptions, Session, SessionEnv};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig};
use ym_gpu::queue::CancelToken;
use ym_ml::bokeh_net::BokehNet;

fn link(from: Value, to: Value) -> Value {
    json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } })
}

fn radial(cx: f64, cy: f64) -> Value {
    json!({ "kind": "radial", "mode": "add", "center": [cx, cy], "radii": [0.3, 0.2], "feather": 0.4 })
}

fn linear() -> Value {
    json!({ "kind": "linear", "mode": "add" })
}

fn brush(feather: f64, refine: f64, shift: f64) -> Value {
    json!({
        "kind": "brush", "mode": "add", "feather": feather, "refine": refine,
        "strokes": [
            { "size": 0.08, "hardness": 0.5, "erase": false,
              "points": [[0.2 + shift, 0.3], [0.4 + shift, 0.5], [0.6 + shift, 0.45], [0.75 + shift, 0.7]] },
            { "size": 0.05, "hardness": 0.8, "erase": false,
              "points": [[0.3 + shift, 0.75], [0.5 + shift, 0.8]] }
        ]
    })
}

fn background() -> Value {
    json!({ "kind": "background", "mode": "add" })
}

/// Main, then correctors 2 ..= `nodes`, in a line. Node 2 gets `ev` as its trim; each corrector `n` gets
/// `rows(n)`.
fn graph(nodes: u32, ev: f64, rows: &dyn Fn(u32) -> Vec<Value>) -> DevelopSettings {
    let mut items = vec![json!({ "id": 1 })];
    let mut ids = vec![1u32];
    for id in 2..=nodes {
        let mut n = json!({
            "id": id, "kind": "corrector", "pos": [f64::from(id), 0.0],
            "trim": { "exposure_ev": if id == 2 { ev } else { 0.3 } },
        });
        let r = rows(id);
        if !r.is_empty() {
            n["masks"] = json!({ "invert": false, "items": r });
        }
        items.push(n);
        ids.push(id);
    }
    let mut links = Vec::new();
    let mut from = json!("input");
    for id in ids {
        links.push(link(from, json!(id)));
        from = json!(id);
    }
    links.push(link(from, json!("output")));
    let doc = json!({ "nodes": { "items": items, "links": links } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct Run<'a> {
    r: &'a Renderer,
    src: &'a PhotoSource,
    cancel: CancelToken,
    frames: usize,
    cell: (f64, f64),
}

impl Run<'_> {
    fn state(&self, zoom100: bool, mask: Option<MaskStyle>) -> ViewportState {
        let mut st = ViewportState::new(ViewportRect::cell(self.cell.0, self.cell.1, 1.0));
        if zoom100 {
            let image = self.r.final_size(self.src, &self.cancel).expect("size");
            st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
        }
        st.mask_view = mask.map(|style| MaskView {
            node: 2,
            style,
            row: None,
        });
        st
    }

    fn frame(&self, s: DevelopSettings, st: &ViewportState) -> f64 {
        let t = Instant::now();
        self.r
            .viewport_frame(
                &self.src.with_settings(s),
                st,
                &FrameOptions::default(),
                &self.cancel,
            )
            .expect("frame");
        t.elapsed().as_secs_f64() * 1000.0
    }

    /// `slider`: the grade of node 2 changes between frames; else the view is panned and the grade stays.
    fn case(
        &self,
        name: &str,
        nodes: u32,
        rows: &dyn Fn(u32) -> Vec<Value>,
        zoom100: bool,
        mask: Option<MaskStyle>,
        slider: bool,
    ) {
        println!(
            "CASE {name} nodes={nodes} view={} mask_view={} change={}",
            if zoom100 { "100%" } else { "fit" },
            match mask {
                None => "off",
                Some(MaskStyle::Overlay) => "overlay",
                Some(MaskStyle::MaskOnly) => "mask_only",
            },
            if slider { "slider" } else { "pan" }
        );
        let image = self.r.final_size(self.src, &self.cancel).expect("size");
        let mut st = self.state(zoom100, mask);
        // Two frames to make everything that is made once (the raw group, the tiles, the source planes,
        // the pipelines), then the measured ones.
        let mut walls = Vec::new();
        // The views of the frames, and for a zoomed view the full-resolution tiles they show, made before
        // the clock runs: the app makes them at a lower priority and draws the frame again, so a drag does
        // not pay for them (this example is about what a frame costs once the tiles are there).
        let mut states = vec![st.clone()];
        for i in 1..self.frames + 2 {
            if !slider {
                // A drag of 300 px, then 5 px back, alternately.
                let dx = if i % 2 == 1 { -300.0 } else { 5.0 };
                st.input(&ViewportInput::Pan { dx, dy: -40.0 }, image);
            }
            states.push(st.clone());
        }
        if zoom100 {
            for s in &states {
                self.r
                    .prepare_tiles(self.src, s, &self.cancel)
                    .expect("tiles");
            }
        }
        for (i, st) in states.iter().enumerate() {
            let ev = if slider { 0.5 + 0.05 * i as f64 } else { 0.5 };
            println!("FRAME {i}");
            let ms = self.frame(graph(nodes, ev, rows), st);
            if i >= 2 {
                walls.push(ms);
            }
        }
        println!(
            "SUMMARY {name} wall_median_ms={:.1} min={:.1} max={:.1}",
            median(&mut walls.clone()),
            walls.iter().copied().fold(f64::MAX, f64::min),
            walls.iter().copied().fold(0.0, f64::max)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = std::path::PathBuf::from(args.first().expect("an ARW file"));
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let frames: usize = value("--frames").map_or(5, |v| v.parse().expect("frames"));
    let cases = value("--cases").unwrap_or_else(|| "a,b,c,d,e,e2".to_owned());
    let zooms = value("--zoom").unwrap_or_else(|| "both".to_owned());
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stdout)
        .try_init();
    ym_core::perf::set_enabled(true);

    let root = std::env::var_os("YELLOWMOON_DATA").map_or_else(
        || std::path::PathBuf::from("data"),
        std::path::PathBuf::from,
    );
    let env = SessionEnv::load(&root).expect("the data folder (YELLOWMOON_DATA)");
    let session = Session::new(env);
    let out = session.open(std::slice::from_ref(&path), OpenOptions::default());
    assert!(session.wait_idle(Duration::from_secs(600)));
    let id = out.added[0];
    let view = session.view(id).expect("view");
    let raw = session.raw_image(id).expect("raw");
    let src = PhotoSource::from_view(&view, raw);

    let gpu = flag("--gpu").then(|| {
        Arc::new(
            Gpu::start(GpuConfig {
                timestamps: true,
                ..GpuConfig::default()
            })
            .expect("a GPU adapter (WGPU_BACKEND=vulkan)"),
        )
    });
    if let Some(g) = &gpu {
        println!("device: {}", g.info());
        // `--no-kept-planes`: the device keeps no raster plane between frames (the speed work's "before").
        if flag("--no-kept-planes") {
            g.set_planes_kept(false);
            println!("kept raster planes: off");
        }
    }
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            gpu_stages: GpuStages::SIGNED_OFF,
            bokeh_model: BokehNet::load_bundled(&root).ok().map(Arc::new),
            data_root: Some(root),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let run = Run {
        r: &r,
        src: &src,
        cancel: CancelToken::new(),
        frames,
        cell: (1600.0, 1060.0),
    };
    let image = r.final_size(&src, &run.cancel).expect("size");
    println!("photo {} x {} px, cell 1600 x 1060", image.0, image.1);

    let none = |_: u32| Vec::new();
    let one_radial = |n: u32| {
        if n == 2 {
            vec![radial(0.5, 0.5)]
        } else {
            vec![]
        }
    };
    let one_brush = |n: u32| {
        if n == 2 {
            vec![brush(0.3, 0.5, 0.0)]
        } else {
            vec![]
        }
    };
    let one_bg = |n: u32| if n == 2 { vec![background()] } else { vec![] };
    let eight = |n: u32| {
        let s = f64::from(n) * 0.02 - 0.1;
        vec![radial(0.3 + s, 0.5), brush(0.3, 0.5, s)]
    };
    let eight_shapes = |n: u32| vec![radial(0.3 + f64::from(n) * 0.05, 0.5), linear()];
    let fit = zooms != "100";
    let hundred = zooms != "fit";
    for c in cases.split(',') {
        let (nodes, rows): (u32, &dyn Fn(u32) -> Vec<Value>) = match c {
            "a" => (2, &none),
            "b" => (2, &one_radial),
            "c" => (2, &one_brush),
            "d" => (2, &one_bg),
            "e" => (8, &eight),
            "e2" => (8, &eight_shapes),
            other => panic!("no case {other}"),
        };
        if fit {
            run.case(&format!("{c}-fit-slider"), nodes, rows, false, None, true);
        }
        if hundred {
            run.case(&format!("{c}-100-pan"), nodes, rows, true, None, false);
            run.case(&format!("{c}-100-slider"), nodes, rows, true, None, true);
        }
        if flag("--mask-view") && (c == "b" || c == "c") {
            for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
                run.case(
                    &format!("{c}-fit-slider-{style:?}"),
                    nodes,
                    rows,
                    false,
                    Some(style),
                    true,
                );
            }
        }
    }
}
