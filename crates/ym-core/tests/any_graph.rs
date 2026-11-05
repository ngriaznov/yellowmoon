//! Any graph the app can build is made on the device (G-11; `docs/23-nodes.md` section 5.12, `docs/17-gpu.md` section 18).
//!
//! A seeded generator builds random graphs: one to five nodes after Main, each a kind drawn from `NodeKind::ALL`, with
//! mask rows of kinds drawn from `MaskKind::ALL`, in a random order, with key links, Bypass and Strength 0. Nothing in
//! the generator names a kind except the two places marked below (`NOT_GENERATED_*` for kinds that cannot be generated
//! cheaply, and the `match`es that give each kind its settings, which do not compile for a kind they miss). Each graph is
//! viewed at Fit and at 100 %, with Show mask on one row, and:
//!
//! * the frame is the device's, not refused, with no "graphics card" note (`views_on_cpu` is false);
//! * it matches the CPU's within the Film bounds (an 8-bit frame: 2 codes, and few pixels beyond one);
//! * for a subset of graphs the export (PNG 16, JPEG) is the device's, matches the CPU's within one 10-bit code and is the
//!   same bytes at another tile size.
//!
//! The run asserts that every generated kind appeared. The seeds are a fixed set plus one rotating seed, from
//! `YM_GRAPH_SEED`, else `GITHUB_RUN_NUMBER`; each graph is printed with its seed and index, and a failing one with its
//! settings as JSON. `YM_GRAPH_SEED=<n> cargo test -p ym-core --test any_graph` replays a rotating seed.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, MaskKind, NodeKind};
use ym_core::vision::{FakeVision, VisionMasks};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;
use ym_raw::testgen::SplitMix64;

const MODEL: &str = "ILCE-7RM4";
/// One 10-bit code after C13, in the codes of a 16-bit file.
const TEN_BIT_CODE: u32 = 64;

/// THE ONE PLACE where node kinds are left out of the generator, each with its reason. Everything else in
/// `NodeKind::ALL` is generated.
const NOT_GENERATED_NODES: &[(NodeKind, &str)] = &[
    (NodeKind::Input, "fixed: every graph has exactly one"),
    (NodeKind::Output, "fixed: every graph has exactly one"),
];

/// THE ONE PLACE where mask kinds are left out of the generator, each with its reason.
const NOT_GENERATED_MASKS: &[(MaskKind, &str)] = &[(
    MaskKind::Object,
    "needs the SAM 2.1 weights and a minutes-long mask pass per graph; covered by mask_object and the device suites",
)];

/// Fixed seeds of the gate. Each makes `GRAPHS_PER_SEED` graphs.
const FIXED_SEEDS: [u64; 4] = [11, 23, 37, 59];
const GRAPHS_PER_SEED: usize = 3;
/// Graphs of the rotating seed.
const ROTATING_GRAPHS: usize = 3;

/// The log is the process's: the tests that look at it run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu() -> Option<Arc<Gpu>> {
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
                "\n* GPU TESTS SKIPPED in any_graph: {e}\n"
            );
            None
        }
    }
}

fn renderer(gpu: Option<Arc<Gpu>>, tile: usize) -> Renderer {
    renderer_with(gpu, tile, RenderOptions::default().view_band_bytes)
}

fn renderer_with(gpu: Option<Arc<Gpu>>, tile: usize, view_band_bytes: u64) -> Renderer {
    let stages = if gpu.is_some() {
        GpuStages::ALL
    } else {
        GpuStages::NONE
    };
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            gpu_stages: stages,
            full_tile_px: tile,
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            data_root: Some(data_root()),
            view_band_bytes,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

// ---- the generator -------------------------------------------------------------------------------------------

fn generated_nodes() -> Vec<NodeKind> {
    NodeKind::ALL
        .into_iter()
        .filter(|k| !NOT_GENERATED_NODES.iter().any(|(x, _)| x == k))
        .collect()
}

fn generated_masks() -> Vec<MaskKind> {
    MaskKind::ALL
        .into_iter()
        .filter(|k| !NOT_GENERATED_MASKS.iter().any(|(x, _)| x == k))
        .collect()
}

fn row_json(kind: MaskKind, rng: &mut SplitMix64) -> Value {
    let f =
        |rng: &mut SplitMix64, lo: f64, hi: f64| lo + (hi - lo) * (rng.below(1000) as f64 / 999.0);
    let mut r = match kind {
        MaskKind::Brush => json!({ "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false,
            "points": [[f(rng, 0.2, 0.5), f(rng, 0.3, 0.6)], [f(rng, 0.5, 0.8), f(rng, 0.4, 0.7)]] }],
            "feather": f(rng, 0.0, 0.6), "refine": f(rng, 0.0, 1.0) }),
        MaskKind::Background => json!({ "feather": f(rng, 0.0, 0.6), "refine": f(rng, 0.0, 1.0) }),
        MaskKind::Subject | MaskKind::People => {
            json!({ "grow": f(rng, -0.3, 0.3), "refine": f(rng, 0.0, 1.0) })
        }
        MaskKind::Object => json!({}),
        MaskKind::Linear => {
            json!({ "center": [0.5, 0.5], "angle_deg": f(rng, -40.0, 40.0), "feather": 0.3 })
        }
        MaskKind::Radial => {
            json!({ "center": [0.5, 0.5], "radii": [0.3, 0.2], "angle_deg": 0.0, "feather": 0.3 })
        }
        MaskKind::ColorRange => json!({ "center": [0.05, 0.0], "width": 0.2, "softness": 0.5 }),
        MaskKind::BrightnessRange => json!({ "low": 0.3, "high": 0.7, "softness": 0.4 }),
    };
    r["kind"] = json!(kind.name());
    r["mode"] = json!(if rng.below(5) == 0 { "subtract" } else { "add" });
    r
}

/// A generated graph: its settings, its description for the log, the kinds in it, and where Show mask looks.
struct Graph {
    seed: u64,
    index: usize,
    settings: DevelopSettings,
    json: Value,
    nodes: Vec<NodeKind>,
    masks: Vec<MaskKind>,
    masked: Option<(u32, usize)>,
    /// Whether the CPU's frame is a reference (Subject and People masks are made by a background worker: no).
    compare: bool,
}

/// `forced_node` and `forced_mask` are the kinds the graph must hold, cycled by the caller so a run covers every kind.
fn generate(seed: u64, index: usize, forced_node: NodeKind, forced_mask: MaskKind) -> Graph {
    let mut rng = SplitMix64(seed.wrapping_mul(1_000_003).wrapping_add(index as u64));
    let pool = generated_nodes();
    let mask_pool = generated_masks();
    let count = 1 + rng.below(5) as usize;
    let mut kinds: Vec<NodeKind> = (0..count)
        .map(|_| pool[rng.below(pool.len() as u64) as usize])
        .collect();
    let slot = rng.below(count as u64) as usize;
    kinds[slot] = forced_node;
    // The forced mask goes on a random node; others get zero to two rows of random kinds.
    let forced_at = rng.below(count as u64) as usize;
    let mut rows: Vec<Vec<MaskKind>> = Vec::new();
    for i in 0..count {
        let n = rng.below(3) as usize;
        let mut r: Vec<MaskKind> = (0..n)
            .map(|_| mask_pool[rng.below(mask_pool.len() as u64) as usize])
            .collect();
        if i == forced_at {
            r.push(forced_mask);
        }
        rows.push(r);
    }
    let mut items = vec![json!({ "id": 1 })];
    let mut links = vec![
        json!({ "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } }),
    ];
    let mut prev = json!(1);
    let mut used_masks = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        let id = 2 + i as u32;
        let mut n = match kind {
            NodeKind::Film => json!({ "id": id, "kind": "film", "pos": [id, 0.0],
                "film": { "exposure_ev": 0.3 }, "grain": { "amount": 100, "seed": id },
                "halation": { "amount": 200, "size": 120 }, "strength": 85 }),
            NodeKind::Corrector => {
                json!({ "id": id, "kind": "corrector", "pos": [id, 0.0], "trim": { "exposure_ev": 0.8 } })
            }
            NodeKind::Input | NodeKind::Output => {
                unreachable!("not generated: {NOT_GENERATED_NODES:?}")
            }
        };
        if !rows[i].is_empty() {
            let items_json: Vec<Value> = rows[i].iter().map(|k| row_json(*k, &mut rng)).collect();
            used_masks.extend(rows[i].iter().copied());
            n["masks"] = json!({ "invert": false, "items": items_json });
        }
        match rng.below(8) {
            0 => n["bypass"] = json!(true),
            1 if *kind == NodeKind::Film => n["strength"] = json!(0),
            _ => {}
        }
        items.push(n);
        links.push(json!({ "from": { "node": prev, "port": "picture" }, "to": { "node": id, "port": "picture" } }));
        prev = json!(id);
    }
    links.push(json!({ "from": { "node": prev, "port": "picture" }, "to": { "node": "output", "port": "picture" } }));
    // Key links: a node's key output into a later node's key input, at most one into each.
    for j in 1..count {
        if rng.below(4) == 0 {
            let from = 2 + rng.below(j as u64) as u32;
            links.push(json!({ "from": { "node": from, "port": "key" }, "to": { "node": 2 + j as u32, "port": "key" } }));
        }
    }
    let doc = json!({ "nodes": { "items": items, "links": links } });
    let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let masked = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.is_empty())
        .map(|(i, r)| (2 + i as u32, rng.below(r.len() as u64) as usize))
        .next();
    let compare = !used_masks
        .iter()
        .any(|k| matches!(k, MaskKind::Subject | MaskKind::People));
    Graph {
        seed,
        index,
        settings,
        json: doc,
        nodes: kinds,
        masks: used_masks,
        masked,
        compare,
    }
}

fn view(
    zoom_100: bool,
    before: bool,
    mask: Option<(u32, MaskStyle, Option<u32>)>,
) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(1332.0, 888.0, 1.0));
    if zoom_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    st.before = before;
    st.mask_view = mask.map(|(node, style, row)| MaskView { node, style, row });
    st
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(
        src,
        st,
        &FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        },
        &CancelToken::new(),
    )
    .expect("frame")
}

fn frame_diff(a: &RenderedFrame, b: &RenderedFrame) -> (u8, usize) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (mut max, mut over) = (0u8, 0usize);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        over += usize::from(d > 1);
    }
    (max, over)
}

fn export(
    d: &Developed,
    settings: &DevelopSettings,
    r: Renderer,
    out: &Path,
    format: FormatKind,
) -> (ComputeDevice, Vec<String>, Vec<u8>) {
    let ctx = ExportContext::new(r).with_home(Some(out.to_path_buf()));
    let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    photo.source = photo.source.with_settings(settings.clone());
    let opts = ExportOptions {
        format,
        bit_depth: if format == FormatKind::Png {
            BitDepth::Sixteen
        } else {
            BitDepth::Eight
        },
        color_space: ColorSpace::Srgb,
        destination: Some(out.to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    let report = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(opts),
        &ExportControl::none(),
    )
    .expect("export");
    let bytes = std::fs::read(&report.path).expect("read the export");
    (report.device, report.notes.clone(), bytes)
}

/// The lines of the log that say the device refused or failed something.
fn refusals(log: &str) -> Vec<String> {
    log.lines()
        .filter(|l| {
            l.contains("refused by an input check")
                || l.contains("failed on the device")
                || l.contains("failed in the export tail")
                || l.contains("the device failed")
        })
        .map(str::to_owned)
        .collect()
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[any graph] {text}");
}

fn rotating_seed() -> Option<u64> {
    ["YM_GRAPH_SEED", "GITHUB_RUN_NUMBER"]
        .iter()
        .find_map(|k| std::env::var(k).ok()?.trim().parse().ok())
}

/// The graphs of a run: the fixed seeds, then the rotating one. The forced kinds cycle through the generated lists, so a
/// run of enough graphs holds every kind whatever the seeds.
fn graphs() -> Vec<Graph> {
    let nodes = generated_nodes();
    let masks = generated_masks();
    let mut plan: Vec<(u64, usize)> = Vec::new();
    for seed in FIXED_SEEDS {
        plan.extend((0..GRAPHS_PER_SEED).map(|i| (seed, i)));
    }
    if let Some(seed) = rotating_seed() {
        plan.extend((0..ROTATING_GRAPHS).map(|i| (seed, i)));
    }
    plan.into_iter()
        .enumerate()
        .map(|(n, (seed, i))| generate(seed, i, nodes[n % nodes.len()], masks[n % masks.len()]))
        .collect()
}

#[test]
fn the_generator_draws_from_the_kind_lists() {
    // Every kind is generated or excluded with a reason; nothing is silently missing.
    for k in NodeKind::ALL {
        assert!(
            generated_nodes().contains(&k) || NOT_GENERATED_NODES.iter().any(|(x, _)| *x == k),
            "{k:?}"
        );
    }
    for k in MaskKind::ALL {
        assert!(
            generated_masks().contains(&k) || NOT_GENERATED_MASKS.iter().any(|(x, _)| *x == k),
            "{k:?}"
        );
    }
    // And a run holds every generated kind.
    let all = graphs();
    let seen_nodes: BTreeSet<_> = all
        .iter()
        .flat_map(|g| g.nodes.iter().map(|k| k.name()))
        .collect();
    let seen_masks: BTreeSet<_> = all
        .iter()
        .flat_map(|g| g.masks.iter().map(|k| k.name()))
        .collect();
    for k in generated_nodes() {
        assert!(
            seen_nodes.contains(k.name()),
            "node kind {k:?} never generated"
        );
    }
    for k in generated_masks() {
        assert!(
            seen_masks.contains(k.name()),
            "mask kind {k:?} never generated"
        );
    }
}

#[test]
fn no_generated_graph_is_refused_by_the_device() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let cpu = renderer(None, 256);
    let started = std::time::Instant::now();
    let all = graphs();
    let mut problems: Vec<String> = Vec::new();
    let (mut worst_view, mut worst_export, mut exports) = ((0u8, 0usize), 0u32, 0usize);
    let (mut n_film, mut n_masked, mut n_keys) = (0usize, 0usize, 0usize);
    for g in &all {
        let name = format!("seed {} graph {}", g.seed, g.index);
        say(&format!(
            "{name}: nodes {:?}, masks {:?}, show mask {:?}, compare {}",
            g.nodes.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masks.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masked,
            g.compare
        ));
        n_film += usize::from(g.nodes.contains(&NodeKind::Film));
        n_masked += usize::from(!g.masks.is_empty());
        n_keys += g.json["nodes"]["links"]
            .as_array()
            .map_or(0, |l| l.iter().filter(|x| x["to"]["port"] == "key").count());
        let before = problems.len();
        let device = renderer(Some(Arc::clone(&gpu)), 256);
        let src = d.source().with_settings(g.settings.clone());
        let log_before = captured_log().len();
        let mut states: Vec<(String, ViewportState)> = Vec::new();
        for zoom_100 in [false, true] {
            states.push((format!("100 % {zoom_100}"), view(zoom_100, false, None)));
            if let Some((node, row)) = g.masked {
                states.push((
                    format!("Show mask row {row}, 100 % {zoom_100}"),
                    view(
                        zoom_100,
                        false,
                        Some((node, MaskStyle::Overlay, Some(row as u32))),
                    ),
                ));
            }
        }
        // The first frame of a Background row makes its blur map; the frames compared are the ones after it.
        let _ = frame(&device, &src, &states[0].1);
        for (what, st) in &states {
            let at = format!("{name}: {what}");
            let got = frame(&device, &src, st);
            if got.report.device != ComputeDevice::Gpu || got.report.device_refused {
                problems.push(format!(
                    "{at}: the frame is {:?}, refused {}",
                    got.report.device, got.report.device_refused
                ));
            }
            if device.views_on_cpu(src.id) {
                problems.push(format!("{at}: views_on_cpu"));
            }
            for n in got
                .report
                .notes
                .iter()
                .filter(|n| n.contains("graphics card"))
            {
                problems.push(format!("{at}: note {n}"));
            }
            if g.compare {
                let _ = frame(&cpu, &src, st);
                let want = frame(&cpu, &src, st);
                let (max, over) = frame_diff(&got, &want);
                worst_view = (worst_view.0.max(max), worst_view.1.max(over));
                if max > 2 || over >= got.rgba.len() / 2000 {
                    problems.push(format!(
                        "{at}: against the CPU {max} codes, {over} values over 1"
                    ));
                }
            }
        }
        if g.compare && g.index == 0 {
            exports += 1;
            let home = tempfile::tempdir().expect("dir");
            for format in [FormatKind::Png, FormatKind::Jpeg] {
                let (dev_on, notes, dev) = export(
                    &d,
                    &g.settings,
                    renderer(Some(Arc::clone(&gpu)), 256),
                    home.path(),
                    format,
                );
                if dev_on != ComputeDevice::Gpu || notes.iter().any(|n| n.contains("graphics card"))
                {
                    problems.push(format!(
                        "{name}: the {format:?} export is {dev_on:?}, notes {notes:?}"
                    ));
                }
                if format == FormatKind::Png {
                    let (_, _, cpu_bytes) =
                        export(&d, &g.settings, renderer(None, 256), home.path(), format);
                    let max = decode_png16_bytes(&dev)
                        .iter()
                        .zip(decode_png16_bytes(&cpu_bytes))
                        .map(|(a, b)| a.abs_diff(b))
                        .max()
                        .unwrap_or(0);
                    worst_export = worst_export.max(max);
                    if max > TEN_BIT_CODE {
                        problems.push(format!(
                            "{name}: the PNG export differs from the CPU's by {max} of 65535"
                        ));
                    }
                }
                let (_, _, other) = export(
                    &d,
                    &g.settings,
                    renderer(Some(Arc::clone(&gpu)), 97),
                    home.path(),
                    format,
                );
                if other != dev {
                    problems.push(format!(
                        "{name}: the {format:?} export at tile 97 is other bytes"
                    ));
                }
            }
        }
        let log = captured_log();
        for line in refusals(&log[log_before.min(log.len())..]) {
            problems.push(format!("{name}: log: {line}"));
        }
        if problems.len() > before {
            say(&format!(
                "{name} FAILED; settings to replay (or YM_GRAPH_SEED={}): {}",
                g.seed, g.json
            ));
        }
    }
    say(&format!(
        "{} graphs ({n_film} with a Film node, {n_masked} with masks, {n_keys} key links), {exports} with exports; worst view {} codes, {} values over 1; worst PNG {worst_export} of 65535; {:?}",
        all.len(),
        worst_view.0,
        worst_view.1,
        started.elapsed()
    ));
    assert!(problems.is_empty(), "{problems:#?}");
}

fn decode_png16_bytes(bytes: &[u8]) -> Vec<u32> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
        .collect()
}

/// Regression (G-11): a view at scale 1 (zoom 50 to 150 %) of a photo with raster masks was made on the CPU when the wider
/// picture the masks read did not fit one binding (the owner's 12 CPU frames of 198, 410 to 840 ms). The planes of such a view
/// are made in tiles (`plane_tiles`), so the route is the device's whatever the binding size, down to the size where the
/// frame itself cannot be held: 1 MiB and up for a small photo, 4 MiB and up for a 1332 x 888 view of a 4000 px one. Below that
/// the CPU may make the frame, never quietly: it says why, in the report and in the note.
#[test]
fn views_at_scale_one_with_raster_masks_stay_on_the_device_at_any_binding_size() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let mut problems = Vec::new();
    // (photo, the cell, the bindings in KiB, the smallest binding the device must make the frame at)
    type Case = ((u32, u32), (f64, f64), &'static [u64], u64);
    let cases: [Case; 2] = [
        (
            (640, 420),
            (300.0, 200.0),
            &[65_536, 4096, 1024, 512, 256, 128, 64],
            1024,
        ),
        ((4000, 2668), (1332.0, 888.0), &[65_536, 4096], 4096),
    ];
    for ((pw, ph), (cw, ch), sizes, device_from) in cases {
        let scene = Scene::new(pw, ph, white, Layout::Patchwork, 0.5)
            .noise(2e-4, 2e-6)
            .seed(5);
        let d = Developed::new(&scene, MODEL);
        let mut rng = SplitMix64(9);
        let rows = |kinds: &[MaskKind], rng: &mut SplitMix64| -> Vec<Value> {
            kinds.iter().map(|k| row_json(*k, rng)).collect()
        };
        for (what, kinds) in [
            ("brush", vec![MaskKind::Brush]),
            (
                "brush and background",
                vec![MaskKind::Brush, MaskKind::Background],
            ),
        ] {
            let doc = json!({ "nodes": { "items": [
                { "id": 1 },
                { "id": 2, "kind": "corrector", "pos": [2, 0.0], "trim": { "exposure_ev": 0.8 },
                  "masks": { "invert": false, "items": rows(&kinds, &mut rng) } },
            ], "links": [
                { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
                { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
            ] } });
            let settings = serde_json::from_value::<DevelopSettings>(doc)
                .expect("a document")
                .validated()
                .expect("a valid graph")
                .0;
            let src = d.source().with_settings(settings);
            for &kib in sizes {
                let device = renderer_with(Some(Arc::clone(&gpu)), 256, kib * 1024);
                let _ = frame(&device, &src, &view(false, false, None));
                for pct in [50.0, 75.0, 100.0, 147.0] {
                    let mut st = ViewportState::new(ViewportRect::cell(cw, ch, 1.0));
                    st.mode = ViewportMode::Percent { value: pct };
                    let got = frame(&device, &src, &st);
                    let on_cpu = got.report.device != ComputeDevice::Gpu;
                    let reasoned = got.report.cpu_reason.is_some()
                        && got.report.device_refused
                        && got.report.notes.iter().any(|n| n.contains("graphics card"));
                    if on_cpu && (kib >= device_from || !reasoned) {
                        problems.push(format!(
                            "{pw} px, {what}, binding {kib} KiB, zoom {pct} %: {:?}, reason {:?}, notes {:?}",
                            got.report.device, got.report.cpu_reason, got.report.notes
                        ));
                    }
                    if !on_cpu && device.views_on_cpu(src.id) {
                        problems.push(format!(
                            "{pw} px, {what}, binding {kib} KiB, zoom {pct} %: views_on_cpu"
                        ));
                    }
                }
            }
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}
