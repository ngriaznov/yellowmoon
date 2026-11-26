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
//! The four kinds of `docs/25` (Parallel, Layer, Splitter, Combiner) are drawn by a second generator, `generate_joined`:
//! mixers of one to three inputs from earlier pictures, Splitters whose outputs feed later inputs, Combiners with linked and
//! unlinked inputs. Its graphs are part of `graphs()` (X-05), so every device assertion above and below holds for them: the
//! frame is the device's, not refused, within the Film bounds of the CPU's, the exports are the device's within one 10-bit code
//! and the same bytes at another tile size, at any binding size. Tests of their own run them on the CPU (tiles, threads,
//! determinism) and export every one of them on the device (`graphs_with_joins_are_exported_on_the_device`).
//!
//! **Film nodes in series (the owner, 2026-11-12: "Follow your recommendation on limits.").** Until 2026-11-21 a graph with two or more
//! Film nodes in any chain (`film_chain`) was compared with the CPU in neither its view nor its export, because each Film pass
//! amplifies the `f32` rounding of the one before it and a pixel far outside the gamut is ill-conditioned for any `f32` pipeline, the
//! CPU's included (seeds 539 and 202). **Since 2026-11-21 neither the view nor the export has a Film-chain exception:** both are
//! judged by the unstable-pixel rule below, like every graph's (`seed_539_graph_0_export_is_judged_by_the_unstable_pixel_rule`,
//! `the_unstable_view_rule_exempts_only_the_cpus_own_unstable_values`). `film_chain` is printed with each graph and decides nothing.
//!
//! **Unstable pixels (the owner, 2026-11-18: "Skip unstable pixels"; made general on 2026-11-21: "extend the rule so there are no
//! exceptions for specific cases").** The bound of one 10-bit code (64 of 65535) between the device's export and the CPU's holds on every
//! value except those where the CPU's own export is unstable: when an export is over the bound anywhere, the CPU renders the graph
//! again eight times with every continuous setting of every node (each real number of the settings document, found through the
//! settings schema, not through a list of kinds) moved at once by plus or minus 1e-5 of its own range with seeded signs, and a value
//! that any of those renders moves by more than the bound is unstable (`judge_export`). Those values are printed, not enforced. A
//! graph within the bound is never nudged. Three safeguards: when nothing could be nudged nothing is exempt; more than
//! `max(60, 0.01 % of the values)` exempt values fail; and the device's value at an unstable index must lie within the CPU's nudged
//! range widened by the bound. The same rule judges the 8-bit view (`judge_view`): a view within its bound (2 codes, fewer than a
//! 2000th of the values over 1) costs nothing; otherwise the CPU renders the view state eight times with the same nudges, a value its
//! view moves by more than 1 code is unstable and is left out of both bounds, and the same three safeguards hold (the cap on exempt
//! values, the range widened by 2 codes). (The cases that led to it: seed 23 graph 0 on Metal, a one-code rounding of a Depth plane next to a thin
//! dark line that the Output's sharpening amplifies, `a_one_code_change_of_a_depth_plane_moves_seed_23_only_through_output_sharpening`;
//! and seed 297 graph 0, two chained Colour Uniform nodes whose second has no mask row,
//! `the_unstable_pixel_rule_sees_the_sensitivity_of_a_setting_that_is_not_a_mask_row`.)
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
    ComputeDevice, FrameOptions, FullRequest, FullTarget, MaskStyle, MaskView, NeutralOnly,
    PhotoSource, RenderOptions, RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::pipeline::{DepthModel, FaceModel, has_depth_row};
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
    (NodeKind::ReferenceGrade, REFERENCE_NOT_DRAWN),
    (
        NodeKind::Parallel,
        "docs/25: generated by `generate_joined`, whose graphs `graphs()` adds",
    ),
    (
        NodeKind::Layer,
        "docs/25: generated by `generate_joined`, whose graphs `graphs()` adds",
    ),
    (
        NodeKind::Splitter,
        "docs/25: generated by `generate_joined`, whose graphs `graphs()` adds",
    ),
    (
        NodeKind::Combiner,
        "docs/25: generated by `generate_joined`, whose graphs `graphs()` adds",
    ),
];

const REFERENCE_NOT_DRAWN: &str = "docs/34: generated by `generate_reference`, whose graphs are `reference_graphs()`, not part of `graphs()`: the device runs them since RG-08 (the node's own device tests below take `reference_graphs()`, the viewer's and the export's)";

/// The kinds `generate_joined` draws besides correctors and Film nodes.
const JOINED_NODES: [NodeKind; 4] = [
    NodeKind::Parallel,
    NodeKind::Layer,
    NodeKind::Splitter,
    NodeKind::Combiner,
];

/// The types of a Blur node, as the generator draws them.
const BLUR_TYPES: [&str; 4] = ["soft", "lens", "motion", "zoom"];
/// From this `index` on, `generate` gives a Blur node the type `BLUR_TYPES[index - BLUR_FORCED_FROM]` (`blur_graphs`).
const BLUR_FORCED_FROM: usize = 40;

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
            // Depth is drawn like the other kinds: a fake model makes its map without the weights (D-04).
            depth_model: Some(DepthModel::fake_default()),
            // So are the faces (S-11b): a fake finder gives two faces without the weights.
            face_model: Some(FaceModel::fake_default()),
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

/// Asks for every view of the states once (a Show mask view of a node the Output does not need asks for its own map), then
/// waits for the depth worker of the photo to end: the frames that follow have the photo's depth map on every renderer.
fn settle_depth(r: &Renderer, src: &PhotoSource, states: &[(String, ViewportState)]) {
    if !has_depth_row(&src.settings) {
        return;
    }
    for (_, st) in states {
        let _ = frame(r, src, st);
    }
    let end = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while r.depth_worker_alive(src.id) {
        assert!(
            std::time::Instant::now() < end,
            "the depth worker did not end"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Waits for the face worker of the photo (a viewer never waits for the faces: the node passes its picture through until they are
/// found), after the renderer is told the photo is the one looked at and every view has been asked for once.
fn settle_faces(r: &Renderer, src: &PhotoSource, states: &[(String, ViewportState)]) {
    if ym_core::pipeline::color::live_face_nodes(&src.settings, ym_color::plan::Stop::Output)
        .is_empty()
    {
        return;
    }
    r.navigate(src.id);
    for (_, st) in states {
        let _ = frame(r, src, st);
    }
    let end = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while r.faces_worker_alive(src.id) || !r.faces_ready(src) {
        assert!(
            std::time::Instant::now() < end,
            "the face worker did not end"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn row_json(kind: MaskKind, rng: &mut SplitMix64) -> Value {
    let f =
        |rng: &mut SplitMix64, lo: f64, hi: f64| lo + (hi - lo) * (rng.below(1000) as f64 / 999.0);
    let mut r = match kind {
        MaskKind::Brush => json!({ "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false,
            "points": [[f(rng, 0.2, 0.5), f(rng, 0.3, 0.6)], [f(rng, 0.5, 0.8), f(rng, 0.4, 0.7)]] }],
            "feather": f(rng, 0.0, 0.6), "refine": f(rng, 0.0, 1.0) }),
        MaskKind::Background => json!({ "feather": f(rng, 0.0, 0.6), "refine": f(rng, 0.0, 1.0) }),
        MaskKind::Depth => json!({ "far": f(rng, 0.2, 1.0),
            "feather": f(rng, 0.0, 0.6), "refine": f(rng, 0.0, 1.0) }),
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
    /// Picture links that do not come from the node before: the branches of the graph.
    branches: usize,
    /// Pairs of live Film nodes neither of which is an ancestor of the other (parallel branches).
    parallel_films: usize,
    /// Whether the CPU's frame is a reference (Subject and People masks are made by a background worker: no).
    compare: bool,
    /// Whether a live Face Refinement node is in the graph (not bypassed, read by the Output): its face step runs on the device
    /// since S-13 (`docs/27`), so the frame is the device's, not refused, and agrees with the CPU's within the view bound.
    face_live: bool,
    /// The most Film nodes on any path from the Input to the Output (see [`film_chain`]). Printed information only: since 2026-11-21
    /// it decides nothing, neither for a view nor for an export (a Face Refinement node is not counted in it, as before).
    film_chain: usize,
}

impl Graph {
    /// Whether the graph has a Blur node that is not bypassed, has an amount, and that the Output reads (or a key reads)
    /// (`ColorChain::has_blur`). A node of a type with no device kernel yet (`blur_refused`) makes the frames and exports the
    /// CPU's, with a reason; one of a type whose task has landed runs on the device.
    fn blur_live(&self) -> bool {
        let Ok(plan) =
            ym_core::pipeline::color::node_plan(&self.settings.nodes, ym_color::plan::Stop::Output)
        else {
            return false;
        };
        let ids = plan.node_ids();
        self.settings
            .nodes
            .blurs
            .iter()
            .any(|(id, b)| ids.contains(id) && !b.is_inert())
    }

    /// Whether a live Blur node of the graph has a type the device does not run yet (`device_kernel` is `None`: Motion until
    /// BL-08; Soft, Lens and Zoom run since BL-06, BL-07 and BL-09): the plan is refused and the frame is the CPU's.
    fn blur_refused(&self) -> bool {
        let Ok(plan) =
            ym_core::pipeline::color::node_plan(&self.settings.nodes, ym_color::plan::Stop::Output)
        else {
            return false;
        };
        let ids = plan.node_ids();
        self.settings.nodes.blurs.iter().any(|(id, b)| {
            ids.contains(id)
                && !b.is_inert()
                && ym_gpu::kernels::blur_stack::device_kernel(
                    &ym_core::pipeline::color::blur_params_of(b),
                )
                .is_none()
        })
    }

    /// Whether the graph has a Colour Uniform node that is not bypassed and does something (a target and an amount): the
    /// graphs `uniform_graphs` collects. The device runs them since CU-05, so every device sweep takes them like any other.
    fn uniform_live(&self) -> bool {
        self.json["nodes"]["items"].as_array().is_some_and(|items| {
            items.iter().any(|n| {
                n["kind"] == "color_uniform"
                    && n["bypass"] != json!(true)
                    && !n["target"].is_null()
                    && n["compress"]
                        .as_object()
                        .is_some_and(|c| c.values().any(|v| v.as_f64().is_some_and(|v| v > 0.0)))
            })
        })
    }
}

/// The most Film nodes in series: the largest number of Film nodes on any one path from the Input to the Output, through
/// picture and key links alike (a key is made from the picture before the node, so a Film node behind a key is behind its
/// Film nodes). Every Film node counts, bypassed or at Strength 0 too. A mixer's branches are parallel: a path takes one
/// branch at a time, so a Film node on one branch is not in series with a Film node on another. A node nothing live reads
/// is on no path.
fn film_chain(doc: &Value) -> usize {
    use std::collections::HashMap;
    let id = |v: &Value| v.to_string();
    let films: BTreeSet<String> = doc["nodes"]["items"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|n| n["kind"] == "film")
        .map(|n| id(&n["id"]))
        .collect();
    let mut reads: HashMap<String, Vec<String>> = HashMap::new();
    for l in doc["nodes"]["links"].as_array().into_iter().flatten() {
        reads
            .entry(id(&l["to"]["node"]))
            .or_default()
            .push(id(&l["from"]["node"]));
    }
    // (The graph is acyclic: the settings were validated.)
    fn depth(
        node: &str,
        films: &BTreeSet<String>,
        reads: &HashMap<String, Vec<String>>,
        memo: &mut HashMap<String, usize>,
    ) -> usize {
        if let Some(&d) = memo.get(node) {
            return d;
        }
        let behind = reads
            .get(node)
            .into_iter()
            .flatten()
            .map(|a| depth(a, films, reads, memo))
            .max()
            .unwrap_or(0);
        let d = behind + usize::from(films.contains(node));
        memo.insert(node.to_owned(), d);
        d
    }
    depth("\"output\"", &films, &reads, &mut HashMap::new())
}

/// Mask Intersect (`docs/23` section 4.2): whether node `id` of graph (`seed`, `index`) combines its rows as an intersection,
/// about one node in three. It has a generator of its own, so that **no other draw moves** and every graph made before this
/// one is the same graph (the documents of the regression seeds have no `combine` and stay as they are).
fn combines(seed: u64, index: usize, id: u32) -> bool {
    let mut rng = SplitMix64(
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((index as u64) << 20)
            .wrapping_add(u64::from(id) ^ 0xC0B1),
    );
    rng.below(3) == 0
}

/// The corrector's trim and grade of a Colour Uniform node (`docs/32` section 12a), drawn for about one node in two. It has a
/// generator of its own, so that **no other draw moves** (the regression seeds' documents have neither field and stay as they
/// are). The draw is the corrector's own kinds of setting: a trim, a few grade sliders, a wheel, a warp pin and a curve point.
fn grade_of(seed: u64, index: usize, id: u32) -> Option<(Value, Value)> {
    let mut rng = SplitMix64(
        seed.wrapping_mul(0xD1B5_4A32_D192_ED03)
            .wrapping_add((index as u64) << 22)
            .wrapping_add(u64::from(id) ^ 0x6A4D),
    );
    if rng.below(2) == 0 {
        return None;
    }
    let signed = |rng: &mut SplitMix64, span: u64| rng.below(2 * span + 1) as f64 - span as f64;
    let trim = json!({ "exposure_ev": signed(&mut rng, 100) / 100.0,
        "temperature": signed(&mut rng, 30), "tint": signed(&mut rng, 20) });
    let mut grade = json!({
        "contrast": { "kind": "manual", "value": signed(&mut rng, 30) },
        "brightness": signed(&mut rng, 20),
        "highlights": { "kind": "manual", "value": signed(&mut rng, 30) },
        "shadows": { "kind": "manual", "value": signed(&mut rng, 30) },
        "saturation": { "kind": "manual", "value": signed(&mut rng, 30) },
        "color_balance": { "midtones": [signed(&mut rng, 20) / 100.0, signed(&mut rng, 20) / 100.0] } });
    if rng.below(2) == 0 {
        let (a, b) = (signed(&mut rng, 10) / 100.0, signed(&mut rng, 10) / 100.0);
        grade["color_warp"] =
            json!({ "pins": [ { "source": [a, b], "target": [b, a], "range": 0.2 } ] });
    }
    if rng.below(2) == 0 {
        let mid = 0.5 + signed(&mut rng, 10) / 100.0;
        grade["curve"] = json!({ "luma": [[0, 0], [0.5, mid], [1, 1]] });
    }
    Some((trim, grade))
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
    let mut used_masks = Vec::new();
    // Where each node's picture comes from: the one before it (a line), or, one time in three, Main, the Input or any
    // earlier node (a branch, which makes the nodes in between live only by a key link, if at all). `None` is the Input.
    let mut source: Vec<Option<usize>> = Vec::new();
    for i in 0..count {
        source.push(match rng.below(3) {
            0 if i > 0 => match rng.below(i as u64 + 1) as usize {
                n if n == i => None,
                n => Some(n),
            },
            _ => i.checked_sub(1),
        });
    }
    for (i, kind) in kinds.iter().enumerate() {
        let id = 2 + i as u32;
        let mut n = match kind {
            NodeKind::Film => json!({ "id": id, "kind": "film", "pos": [id, 0.0],
                "film": { "exposure_ev": 0.3 }, "grain": { "amount": 100, "seed": id },
                "halation": { "amount": 200, "size": 120 }, "strength": 85 }),
            NodeKind::Corrector => {
                json!({ "id": id, "kind": "corrector", "pos": [id, 0.0], "trim": { "exposure_ev": 0.8 } })
            }
            // Strong amounts, so the node changes the faces of the fake finder (S-11b).
            NodeKind::FaceRefinement => {
                json!({ "id": id, "kind": "face_refinement", "pos": [id, 0.0],
                "all": { "blemish": 30, "dark_circles": 80, "even_skin": 80, "skin_contrast": 30 } })
            }
            // A target and amounts that move colours (`docs/32`); the masks and Bypass are the common ones below.
            NodeKind::ColorUniform => {
                let hue = rng.below(360);
                let saturation = 4 + rng.below(30);
                let lightness = rng.below(100) as f64 / 100.0 - 0.5;
                let amount = |rng: &mut SplitMix64| 30 + rng.below(71);
                let (h, sa, l) = (amount(&mut rng), amount(&mut rng), amount(&mut rng));
                json!({ "id": id, "kind": "color_uniform", "pos": [id, 0.0],
                    "target": { "hue_deg": hue, "saturation": saturation, "lightness_ev": lightness },
                    "compress": { "hue": h, "saturation": sa, "lightness": l } })
            }
            // A Blur node (`docs/33`): one type, drawn (or forced by an `index` from `BLUR_FORCED_FROM`), with small amounts (the
            // picture is 640 px wide, and the CPU runs the exact kernels), and the switch drawn, three in four on.
            NodeKind::Blur => {
                let ty = if index >= BLUR_FORCED_FROM {
                    BLUR_TYPES[(index - BLUR_FORCED_FROM) % BLUR_TYPES.len()]
                } else {
                    BLUR_TYPES[rng.below(BLUR_TYPES.len() as u64) as usize]
                };
                let blades = [0u64, 5, 6, 8][rng.below(4) as usize];
                json!({ "id": id, "kind": "blur", "pos": [id, 0.0], "type": ty,
                    "soft": { "radius": 10 + rng.below(30) },
                    "lens": { "radius": 10 + rng.below(30), "blades": blades,
                        "rotation_deg": rng.below(360), "highlights": rng.below(100) },
                    "motion": { "length": 10 + rng.below(30), "angle_deg": rng.below(180) },
                    "zoom": { "amount": 20 + rng.below(60),
                        "center": [rng.below(100) as f64 / 100.0, rng.below(100) as f64 / 100.0] },
                    "respect_mask_edges": rng.below(4) != 0 })
            }
            // Excluded by `NOT_GENERATED_NODES`: `generate_joined` makes the four kinds of `docs/25`.
            NodeKind::Input
            | NodeKind::Output
            | NodeKind::Parallel
            | NodeKind::Layer
            | NodeKind::Splitter
            | NodeKind::Combiner
            | NodeKind::ReferenceGrade => {
                unreachable!("not generated: {NOT_GENERATED_NODES:?}")
            }
        };
        if !rows[i].is_empty() {
            let items_json: Vec<Value> = rows[i].iter().map(|k| row_json(*k, &mut rng)).collect();
            used_masks.extend(rows[i].iter().copied());
            n["masks"] = json!({ "invert": false, "items": items_json });
            if combines(seed, index, id) {
                n["masks"]["combine"] = json!("intersect");
            }
        }
        if *kind == NodeKind::ColorUniform
            && let Some((trim, grade)) = grade_of(seed, index, id)
        {
            n["trim"] = trim;
            n["grade"] = grade;
        }
        match rng.below(8) {
            0 => n["bypass"] = json!(true),
            1 if *kind == NodeKind::Film => n["strength"] = json!(0),
            _ => {}
        }
        items.push(n);
        // Node `i` is item `i + 1` (Main is first); its picture comes from `source[i]`, a node's id being `2 + index`.
        let from = match source[i] {
            // (The first node after Main reads Main; a later node that reads "nothing" reads the Input.)
            None if i == 0 => json!(1),
            None => json!("input"),
            Some(n) => json!(2 + n as u32),
        };
        links.push(json!({ "from": { "node": from, "port": "picture" }, "to": { "node": id, "port": "picture" } }));
    }
    let last = json!(2 + (count - 1) as u32);
    links.push(json!({ "from": { "node": last, "port": "picture" }, "to": { "node": "output", "port": "picture" } }));
    // Key links: a node's key output into a later node's key input, at most one into each.
    let mut keyed = vec![false; count];
    let mut key_from: Vec<Option<usize>> = vec![None; count];
    for j in 1..count {
        if rng.below(4) == 0 {
            let from = rng.below(j as u64) as usize;
            key_from[j] = Some(from);
            keyed[j] = true;
            links.push(json!({ "from": { "node": 2 + from as u32, "port": "key" }, "to": { "node": 2 + j as u32, "port": "key" } }));
        }
    }
    // A node nothing live reads (a branch's far side) joins the graph by a key link into a later live node that has none:
    // the joins the graph allows.
    loop {
        let mut live = vec![false; count];
        let mut stack = vec![count - 1];
        while let Some(n) = stack.pop() {
            if std::mem::replace(&mut live[n], true) {
                continue;
            }
            stack.extend(source[n]);
            stack.extend(key_from[n]);
        }
        let Some(dead) = (0..count).find(|&n| !live[n]) else {
            break;
        };
        let Some(into) = (dead + 1..count).find(|&j| live[j] && !keyed[j]) else {
            break;
        };
        // (A dead node whose own sources are dead stays dead until they are joined; the loop takes them one at a time.)
        keyed[into] = true;
        key_from[into] = Some(dead);
        links.push(json!({ "from": { "node": 2 + dead as u32, "port": "key" }, "to": { "node": 2 + into as u32, "port": "key" } }));
    }
    let branches = (0..count)
        .filter(|&i| source[i] != i.checked_sub(1))
        .count();
    // The ancestors of each node through its picture and key links, to count the Film nodes on parallel branches.
    let ancestors = |n: usize| {
        let mut seen = vec![false; count];
        let mut stack: Vec<usize> = source[n].into_iter().chain(key_from[n]).collect();
        while let Some(a) = stack.pop() {
            if !std::mem::replace(&mut seen[a], true) {
                stack.extend(source[a].into_iter().chain(key_from[a]));
            }
        }
        seen
    };
    let mut live = vec![false; count];
    let mut stack = vec![count - 1];
    while let Some(n) = stack.pop() {
        if !std::mem::replace(&mut live[n], true) {
            stack.extend(source[n].into_iter().chain(key_from[n]));
        }
    }
    let films: Vec<usize> = (0..count)
        .filter(|&i| live[i] && kinds[i] == NodeKind::Film)
        .collect();
    let parallel_films = films
        .iter()
        .flat_map(|&a| films.iter().filter(move |&&b| a < b).map(move |&b| (a, b)))
        .filter(|&(a, b)| !ancestors(b)[a])
        .count();
    // Main is the first item; node `i` is item `i + 1`.
    let face_live = (0..count).any(|i| {
        live[i] && kinds[i] == NodeKind::FaceRefinement && items[i + 1]["bypass"] != json!(true)
    });
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
    let chain = film_chain(&doc);
    Graph {
        seed,
        index,
        settings,
        json: doc,
        nodes: kinds,
        masks: used_masks,
        branches,
        parallel_films,
        masked,
        compare,
        face_live,
        film_chain: chain,
    }
}

// ---- graphs with joins (docs/25, CPU only until X-05) ---------------------------------------------------------

/// A port of a node that carries a picture: where a later node may read from.
#[derive(Clone)]
struct Out {
    node: Value,
    port: &'static str,
}

/// A graph with the four kinds of `docs/25`: `count` nodes after Main drawn from correctors, Film nodes and the four kinds,
/// each reading earlier pictures (any output of any earlier node, the Input's picture too). `forced` is a kind the graph must
/// hold, cycled by the caller. The Output reads the last picture made.
fn generate_joined(seed: u64, index: usize, forced: NodeKind) -> Graph {
    let mut rng = SplitMix64(
        seed.wrapping_mul(1_000_003)
            .wrapping_add(index as u64)
            .wrapping_add(0x9E37),
    );
    let pool = [
        NodeKind::Corrector,
        NodeKind::Corrector,
        NodeKind::Film,
        NodeKind::Parallel,
        NodeKind::Layer,
        NodeKind::Splitter,
        NodeKind::Combiner,
    ];
    let count = 3 + rng.below(4) as usize;
    let mut kinds: Vec<NodeKind> = (0..count)
        .map(|_| pool[rng.below(pool.len() as u64) as usize])
        .collect();
    let slot = rng.below(count as u64) as usize;
    kinds[slot] = forced;
    let mut items = vec![json!({ "id": 1 })];
    let mut links = vec![
        json!({ "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } }),
    ];
    let link = |from: &Out, to: u32, port: String| json!({ "from": { "node": from.node, "port": from.port }, "to": { "node": to, "port": port } });
    // What a later node may read: the Input, Main, and every output made so far.
    let mut values = vec![
        Out {
            node: json!("input"),
            port: "picture",
        },
        Out {
            node: json!(1),
            port: "picture",
        },
    ];
    let mut used_masks = Vec::new();
    let mask_kinds = [
        MaskKind::Linear,
        MaskKind::Radial,
        MaskKind::ColorRange,
        MaskKind::BrightnessRange,
    ];
    for (i, kind) in kinds.iter().enumerate() {
        let id = 2 + i as u32;
        let pos = json!([id, 0.0]);
        let pick = |rng: &mut SplitMix64, values: &[Out]| {
            values[rng.below(values.len() as u64) as usize].clone()
        };
        let mut node = match kind {
            NodeKind::Corrector | NodeKind::Film => {
                let mut n = if *kind == NodeKind::Film {
                    json!({ "id": id, "kind": "film", "pos": pos, "film": { "exposure_ev": 0.3 },
                        "grain": { "amount": 100, "seed": id }, "halation": { "amount": 200, "size": 120 }, "strength": 85 })
                } else {
                    let ev = (rng.below(21) as f64 - 10.0) / 10.0;
                    json!({ "id": id, "kind": "corrector", "pos": pos, "trim": { "exposure_ev": ev } })
                };
                if rng.below(2) == 0 {
                    let k = mask_kinds[rng.below(mask_kinds.len() as u64) as usize];
                    used_masks.push(k);
                    n["masks"] = json!({ "invert": false, "items": [row_json(k, &mut rng)] });
                    if combines(seed, index, id) {
                        n["masks"]["combine"] = json!("intersect");
                    }
                }
                if rng.below(8) == 0 {
                    n["bypass"] = json!(true);
                }
                let from = pick(&mut rng, &values);
                links.push(link(&from, id, "picture".to_owned()));
                values.push(Out {
                    node: json!(id),
                    port: "picture",
                });
                n
            }
            NodeKind::Parallel | NodeKind::Layer => {
                let mut n = if *kind == NodeKind::Layer {
                    let modes = [
                        "normal",
                        "add",
                        "subtract",
                        "multiply",
                        "lighten",
                        "darken",
                        "difference",
                    ];
                    let opacity = [1.0, 0.5, 0.25, 0.0][rng.below(4) as usize];
                    json!({ "id": id, "kind": "layer", "pos": pos,
                        "mode": modes[rng.below(modes.len() as u64) as usize], "opacity": opacity })
                } else {
                    json!({ "id": id, "kind": "parallel", "pos": pos })
                };
                if rng.below(8) == 0 {
                    n["bypass"] = json!(true);
                }
                for k in 1..=(1 + rng.below(3)) {
                    let from = pick(&mut rng, &values);
                    links.push(link(&from, id, format!("picture.{k}")));
                }
                values.push(Out {
                    node: json!(id),
                    port: "picture",
                });
                n
            }
            NodeKind::Splitter => {
                let from = pick(&mut rng, &values);
                links.push(link(&from, id, "picture".to_owned()));
                for port in ["red", "green", "blue"] {
                    values.push(Out {
                        node: json!(id),
                        port,
                    });
                }
                json!({ "id": id, "kind": "splitter", "pos": pos })
            }
            NodeKind::Combiner => {
                for port in ["red", "green", "blue"] {
                    // One input in five is left unlinked: it reads the Input's picture.
                    if rng.below(5) != 0 {
                        let from = pick(&mut rng, &values);
                        links.push(link(&from, id, port.to_owned()));
                    }
                }
                values.push(Out {
                    node: json!(id),
                    port: "picture",
                });
                json!({ "id": id, "kind": "combiner", "pos": pos })
            }
            NodeKind::Input
            | NodeKind::Output
            | NodeKind::FaceRefinement
            | NodeKind::ColorUniform
            | NodeKind::Blur
            | NodeKind::ReferenceGrade => {
                unreachable!("not in the joined pool")
            }
        };
        if !node["masks"].is_null() {
            node["masks"]["invert"] = json!(false);
        }
        items.push(node);
    }
    let last = values.last().cloned().expect("a value");
    links.push(json!({ "from": { "node": last.node, "port": last.port }, "to": { "node": "output", "port": "picture" } }));
    let doc = json!({ "nodes": { "items": items, "links": links } });
    let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let chain = film_chain(&doc);
    Graph {
        seed,
        index,
        settings,
        json: doc,
        nodes: kinds,
        masks: used_masks,
        masked: None,
        branches: 0,
        parallel_films: 0,
        compare: true,
        face_live: false,
        film_chain: chain,
    }
}

/// The graphs with joins of a run: each seed makes two, and the forced kinds cycle through the four, so every kind appears.
fn joined_graphs() -> Vec<Graph> {
    let mut all = Vec::new();
    let mut n = 0;
    for seed in FIXED_SEEDS.into_iter().chain(rotating_seed()) {
        for i in 0..2 {
            all.push(generate_joined(
                seed,
                i,
                JOINED_NODES[n % JOINED_NODES.len()],
            ));
            n += 1;
        }
    }
    all
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
    st.mask_view = mask.map(|(node, style, row)| MaskView {
        node,
        style,
        row,
        depth: false,
        face: None,
    });
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

/// How a device renderer's frame of a Face Refinement graph differs from the CPU renderer's.
struct FaceDiff {
    /// The largest difference of a value, in codes of 255, and the count of values over 1.
    max: u8,
    over: usize,
    /// Where the largest one is: the frame's pixel and channel (0 to 3), and the same place in the final image.
    at: (usize, usize, usize),
    at_final: (f64, f64),
    /// Whether that place is inside a face box, or inside the box grown by half its size on every side.
    in_box: bool,
    in_grown: bool,
    /// The largest difference inside the boxes and the count of values over 1 inside and outside them.
    max_in_boxes: u8,
    over_in: usize,
    over_out: usize,
}

impl FaceDiff {
    fn text(&self) -> String {
        format!(
            "max {} codes at ({}, {}) channel {} = final ({:.1}, {:.1}), in a face box {}, in a box grown by half {}; \
             {} values over 1 ({} inside the boxes, {} outside); largest inside the boxes {}",
            self.max,
            self.at.0,
            self.at.1,
            self.at.2,
            self.at_final.0,
            self.at_final.1,
            self.in_box,
            self.in_grown,
            self.over,
            self.over_in,
            self.over_out,
            self.max_in_boxes
        )
    }
}

/// Measures `a` against `b` (two frames of the state `st`), and says where the faces are in them: `boxes` are shares of the final
/// image of size `image`.
fn face_frame_diff(
    a: &RenderedFrame,
    b: &RenderedFrame,
    st: &ViewportState,
    image: &(usize, usize),
    boxes: &[ym_core::pipeline::FaceBox],
) -> FaceDiff {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let to_final = st.view_to_final(*image);
    let (iw, ih) = (image.0 as f64, image.1 as f64);
    let inside = |fx: f64, fy: f64, grow: f64| {
        boxes.iter().any(|f| {
            let (w, h) = (f.width * iw, f.height * ih);
            fx >= f.x * iw - w * grow
                && fx < f.x * iw + w * (1.0 + grow)
                && fy >= f.y * ih - h * grow
                && fy < f.y * ih + h * (1.0 + grow)
        })
    };
    let mut r = FaceDiff {
        max: 0,
        over: 0,
        at: (0, 0, 0),
        at_final: (0.0, 0.0),
        in_box: false,
        in_grown: false,
        max_in_boxes: 0,
        over_in: 0,
        over_out: 0,
    };
    for (i, (x, y)) in a.rgba.iter().zip(&b.rgba).enumerate() {
        let d = x.abs_diff(*y);
        let (px, c) = (i / 4, i % 4);
        let (vx, vy) = (px % a.width, px / a.width);
        let (fx, fy) = to_final.apply(vx as f64 + 0.5, vy as f64 + 0.5);
        let grown = inside(fx, fy, 0.5);
        let boxed = inside(fx, fy, 0.0);
        if d > r.max {
            r.max = d;
            r.at = (vx, vy, c);
            r.at_final = (fx, fy);
            r.in_box = boxed;
            r.in_grown = grown;
        }
        if boxed {
            r.max_in_boxes = r.max_in_boxes.max(d);
        }
        if d > 1 {
            r.over += 1;
            if boxed {
                r.over_in += 1;
            } else {
                r.over_out += 1;
            }
        }
    }
    r
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
/// Two Film nodes on parallel branches that join by a key link (G-11 part 3, item 2): Main -> Film 2 -> corrector 4, Input
/// -> Film 3, and Film 3's key into corrector 4, with a Brush row on the Film node 2 and one on the corrector. Each Film node
/// reads only its own ancestors' pictures, so neither is made over what the other reads.
fn parallel_branches(seed: u64) -> Graph {
    let mut rng = SplitMix64(seed);
    let rows = |rng: &mut SplitMix64| json!({ "invert": false, "items": [row_json(MaskKind::Brush, rng)] });
    let film = |id: u32, rng: &mut SplitMix64| {
        json!({ "id": id, "kind": "film", "pos": [id, 0.0], "film": { "exposure_ev": 0.3 },
            "grain": { "amount": 100, "seed": id }, "halation": { "amount": 200, "size": 120 },
            "strength": 85, "masks": rows(rng) })
    };
    let mut film3 = film(3, &mut rng);
    if let Some(o) = film3.as_object_mut() {
        o.remove("masks");
    }
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        film(2, &mut rng),
        film3,
        { "id": 4, "kind": "corrector", "pos": [4, 0.0], "trim": { "exposure_ev": 0.8 }, "masks": rows(&mut rng) } ],
      "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 4, "port": "picture" } },
        { "from": { "node": 3, "port": "key" }, "to": { "node": 4, "port": "key" } },
        { "from": { "node": 4, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let chain = film_chain(&doc);
    Graph {
        seed,
        index: 100,
        settings,
        json: doc,
        nodes: vec![NodeKind::Film, NodeKind::Film, NodeKind::Corrector],
        masks: vec![MaskKind::Brush, MaskKind::Brush],
        branches: 2,
        parallel_films: 1,
        masked: Some((4, 0)),
        compare: true,
        face_live: false,
        film_chain: chain,
    }
}

/// S-12: a live Face Refinement node with no mask row (the drawn graphs often put a Background, depth or radial row on it, which can
/// leave the faces alone), at strong amounts, with a corrector before it and a Film node after it: the node changes the faces of the
/// fake finder, so the comparison with the CPU is made on a frame the node really retouched.
fn face_line(seed: u64) -> Graph {
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [2, 0.0], "trim": { "exposure_ev": 0.3 } },
        { "id": 3, "kind": "face_refinement", "pos": [3, 0.0], "same_for_all": true,
          "all": { "blemish": 80, "dark_circles": 80, "even_skin": 80, "skin_contrast": 30 } },
        { "id": 4, "kind": "film", "pos": [4, 0.0], "film": { "exposure_ev": 0.3 },
          "grain": { "amount": 100, "seed": 4 }, "halation": { "amount": 200, "size": 120 }, "strength": 85 } ],
      "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": 4, "port": "picture" } },
        { "from": { "node": 4, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let chain = film_chain(&doc);
    Graph {
        seed,
        index: 101,
        settings,
        json: doc,
        nodes: vec![
            NodeKind::Corrector,
            NodeKind::FaceRefinement,
            NodeKind::Film,
        ],
        masks: Vec::new(),
        branches: 0,
        parallel_films: 0,
        masked: None,
        compare: true,
        face_live: true,
        film_chain: chain,
    }
}

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
    let mut all: Vec<Graph> = plan
        .into_iter()
        .enumerate()
        .map(|(n, (seed, i))| generate(seed, i, nodes[n % nodes.len()], masks[n % masks.len()]))
        .collect();
    // Parallel Film branches are rare in the draw; one is always there.
    all.push(parallel_branches(FIXED_SEEDS[0]));
    // A live Face Refinement graph the node really changes (S-12).
    all.push(face_line(FIXED_SEEDS[0]));
    // The four kinds of docs/25 (X-05): every device assertion holds for them too.
    all.extend(joined_graphs());
    all
}

/// The graphs the device makes: `graphs()` without the plans the device refuses (a live Blur node of a type with no device kernel
/// yet; a Soft node runs on the device since BL-06, so these sweeps hold it).
fn device_graphs() -> Vec<Graph> {
    graphs().into_iter().filter(|g| !g.blur_refused()).collect()
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
    // The kinds of `docs/25` are drawn by the other generator, which makes each of them in every run.
    let joined = joined_graphs();
    let seen: BTreeSet<_> = joined
        .iter()
        .flat_map(|g| g.nodes.iter().map(|k| k.name()))
        .collect();
    for k in JOINED_NODES {
        assert!(seen.contains(k.name()), "node kind {k:?} never generated");
        assert!(
            NOT_GENERATED_NODES.iter().any(|(x, _)| *x == k),
            "{k:?} is generated twice"
        );
    }
}

#[test]
fn the_generator_draws_both_combine_rules() {
    let (mut intersect, mut add) = (0usize, 0usize);
    for g in graphs() {
        let items = g.json["nodes"]["items"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for item in items {
            match item["masks"]["combine"].as_str() {
                Some("intersect") => intersect += 1,
                None if item["masks"]["items"].is_array() => add += 1,
                None => {}
                other => panic!("a combine of {other:?}"),
            }
        }
    }
    eprintln!("nodes with masks: {intersect} intersect, {add} add");
    assert!(intersect > 2 && add > 2, "{intersect} intersect, {add} add");
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
    let all = device_graphs();
    let mut problems: Vec<String> = Vec::new();
    let (mut worst_view, mut worst_export, mut exports) = ((0u8, 0usize), 0u32, 0usize);
    let (mut n_film, mut n_masked, mut n_keys) = (0usize, 0usize, 0usize);
    let (mut n_branched, mut n_parallel) = (0usize, 0usize);
    // S-12: the live Face Refinement graphs that were compared with the CPU, and those whose faces the node really changed.
    let (mut n_face_compared, mut n_face_changed) = (0usize, 0usize);
    // Graphs whose views are compared with the CPU's.
    let mut n_compared = 0usize;
    // The views over the bound that the rule excused as the CPU's own unstable values, and how many CPU renders that took.
    let mut view_judged = 0usize;
    for g in &all {
        let name = format!("seed {} graph {}", g.seed, g.index);
        say(&format!(
            "{name}: nodes {:?}, masks {:?}, show mask {:?}, compare {}, Film nodes in series {}, branches {}, parallel Film pairs {}",
            g.nodes.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masks.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masked,
            g.compare,
            g.film_chain,
            g.branches,
            g.parallel_films
        ));
        // The view and the export of every graph with a CPU reference are judged, Film nodes in series or not (`judge_view`,
        // `judge_export`; the owner, 2026-11-21).
        n_compared += usize::from(g.compare);
        n_film += usize::from(g.nodes.contains(&NodeKind::Film));
        n_branched += usize::from(g.branches > 0);
        n_parallel += g.parallel_films;
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
        // The first frame of a Background row makes its blur map; the frames compared are the ones after it. A Depth row's map
        // is made by the background worker: every view is asked for once and the worker is waited for, on each renderer.
        let _ = frame(&device, &src, &states[0].1);
        settle_depth(&device, &src, &states);
        if g.compare && has_depth_row(&src.settings) {
            settle_depth(&cpu, &src, &states);
        }
        if g.face_live {
            // A live Face Refinement node (`docs/27` S-13): the frame is the device's like any other (the node's picture is made by
            // the face kernels, `kernels::face`), not refused and not on the CPU, and the pixels agree with the CPU renderer's within
            // the view bound (the faces, masks and maps are the CPU reference's on both: the analysis is made without a device).
            settle_faces(&device, &src, &states);
            settle_faces(&cpu, &src, &states);
            let face_image = cpu
                .final_size(&src, &CancelToken::new())
                .expect("the final size");
            let face_boxes = cpu.face_state(&src).faces;
            // The same graph with its Face Refinement nodes bypassed: what the node changes is the difference.
            let without_nodes = {
                let mut doc = g.json.clone();
                for item in doc["nodes"]["items"].as_array_mut().expect("items") {
                    if item["kind"] == "face_refinement" {
                        item["bypass"] = json!(true);
                    }
                }
                serde_json::from_value::<DevelopSettings>(doc)
                    .expect("a document")
                    .validated()
                    .expect("a valid graph")
                    .0
            };
            let src_without = d.source().with_settings(without_nodes);
            let mut changed_in_boxes = false;
            n_face_compared += usize::from(g.compare);
            for (what, st) in &states {
                let at = format!("{name}: {what}");
                let got = frame(&device, &src, st);
                if got.report.device != ComputeDevice::Gpu
                    || got.report.device_refused
                    || got.report.cpu_reason.is_some()
                {
                    problems.push(format!(
                        "{at}: a Face Refinement frame is {:?}, refused {}, reason {:?}",
                        got.report.device, got.report.device_refused, got.report.cpu_reason
                    ));
                }
                if device.views_on_cpu(src.id) {
                    problems.push(format!(
                        "{at}: views_on_cpu is true for a Face Refinement frame"
                    ));
                }
                if g.compare {
                    let want = frame(&cpu, &src, st);
                    // Every pair is measured and written to the log, passing or not, so that a run on a device gives the numbers.
                    //
                    // The bound is the view bound, no wider, and the same bound is held inside the face regions on their own: the node
                    // must not make the device's small input difference larger.
                    let report = face_frame_diff(&got, &want, st, &face_image, &face_boxes);
                    say(&format!("{at}: against the CPU: {}", report.text()));
                    let plain = frame(&cpu, &src_without, st);
                    let moved = face_frame_diff(&want, &plain, st, &face_image, &face_boxes);
                    say(&format!(
                        "{at}: the node changes the CPU frame by up to {} codes inside the {} face box(es), {} values over 1 there",
                        moved.max_in_boxes,
                        face_boxes.len(),
                        moved.over_in
                    ));
                    changed_in_boxes |= moved.max_in_boxes > 0;
                    // And the device's own frame is not the device's frame without the node.
                    let device_plain = frame(&device, &src_without, st);
                    let device_moved =
                        face_frame_diff(&got, &device_plain, st, &face_image, &face_boxes);
                    if g.compare && moved.max_in_boxes > 0 && device_moved.max_in_boxes == 0 {
                        problems.push(format!("{at}: the node changes nothing on the device"));
                    }
                    if report.max > 2
                        || report.over >= got.rgba.len() / 2000
                        || report.max_in_boxes > 2
                    {
                        problems.push(format!("{at}: against the CPU: {}", report.text()));
                    }
                }
            }
            n_face_changed += usize::from(changed_in_boxes);
            continue;
        }
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
                let judged = judge_view(&cpu, &src, st, &got, &want, &at, &mut view_judged);
                if let Some(f) = judged.failure {
                    // Where the largest difference is, as for a graph with faces (there are no boxes to say).
                    let image = cpu
                        .final_size(&src, &CancelToken::new())
                        .expect("the final size");
                    let at_max = face_frame_diff(&got, &want, st, &image, &[]);
                    problems.push(format!(
                        "{f} (against the CPU {max} codes, {over} values over 1: {})",
                        at_max.text()
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
                    // The export is judged for every graph, Film nodes in series or not: the unstable-pixel rule (`judge_export`)
                    // is what excuses a value the CPU's own export cannot hold still, the same for every kind of node.
                    worst_export = worst_export.max(max);
                    if max > TEN_BIT_CODE {
                        let judged = judge_export(
                            &d,
                            &g.settings,
                            &decode_png16_bytes(&dev),
                            &decode_png16_bytes(&cpu_bytes),
                            home.path(),
                            &name,
                        );
                        if let Some(f) = judged.failure {
                            problems.push(format!(
                                "{f} (the PNG export differs from the CPU's by {max} of 65535)"
                            ));
                        }
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
        "{n_compared} graphs compared with the CPU in their views and exports, {view_judged} of their views over the bound and judged by the unstable-pixel rule"
    ));
    say(&format!(
        "{} graphs ({n_film} with a Film node, {n_masked} with masks, {n_keys} key links, {n_branched} with picture branches, {n_parallel} parallel Film pairs), {exports} with exports; worst view {} codes, {} values over 1; worst PNG {worst_export} of 65535; {:?}",
        all.len(),
        worst_view.0,
        worst_view.1,
        started.elapsed()
    ));
    say(&format!(
        "{n_face_compared} live Face Refinement graphs compared with the CPU, {n_face_changed} of them with faces the node really changed"
    ));
    // S-12: a live Face Refinement graph that was generated but never compared, or whose node changed nothing, would pass every
    // check above without testing the node. The fixed seeds hold some of each (the forced kinds cycle through the node list).
    assert!(
        n_face_compared > 0,
        "the fixed seeds hold no live Face Refinement graph that is compared with the CPU"
    );
    assert!(
        n_face_changed > 0,
        "no live Face Refinement graph had its faces changed by the node: the generated graphs do not test it"
    );
    assert!(problems.is_empty(), "{problems:#?}");
}

/// Seed 539, graph 0 of the rotating seed (Mac GPU run 43111535011, main e6d5d71): Main, a corrector with two Depth rows and
/// +0.8 EV, a corrector with a Linear mask and +0.8 EV, and two Film nodes (film exposure +0.3, grain 100, halation 200 at
/// size 120, Strength 85), the second linked to the first by a key. On Metal its PNG export was 222 of 65535 from the CPU's
/// (the bound is one 10-bit code, 64), on lavapipe 180 (169 with output sharpening off).
///
/// **It is not a device error.** Every stage is within its own bound: the colour pass
/// on the CPU's own scene values of the worst pixels is 1e-7 from the CPU (gate 1e-5), and with one Film node after the
/// correctors the picture is 8e-5 in display light and 3e-5 in AWG4 from the CPU. The second Film node leaves a few pixels
/// (3 values over 64 codes of 806,400, all at the saturated patches of the chart's left edge) whose scene colour is
/// AWG4 (3.0, 0.1, -6.0) to (3.9, -0.6, -10.0) after 1.6 EV, grain and halation: far outside the gamut, with a display
/// blue of 1e-7 to 1e-6. There the picture is ill-conditioned for any `f32` LogC4 between the nodes, the CPU's too: on the
/// CPU alone, a change of 1e-5 EV in the first Film node's exposure (7e-6 relative) moves the display light of those
/// pixels by 1 % (1e-4 EV: 5 %), and the device's roundings differ from the CPU's by 6e-6 in AWG4. Output sharpening
/// spreads such a pixel over its neighbours. Until 2026-11-21 this case was "reported, not enforced" by the Film-chain rule of
/// 2026-11-12; the unstable-pixel rule is general now (the owner, 2026-11-21), and the export is judged by it like any other: the
/// CPU's own export moves by more than the bound at those values when the settings are nudged, so they are exempt and printed.
const SEED_539_GRAPH_0: &str = r#"{"nodes":{"items":[{"id":1},{"id":2,"kind":"corrector","masks":{"invert":false,"items":[{"far":0.23923923923923926,"feather":0.2912912912912913,"kind":"depth","mode":"add","refine":0.3303303303303303},{"far":0.5027027027027027,"feather":0.4678678678678678,"kind":"depth","mode":"add","refine":0.0970970970970971}]},"pos":[2,0.0],"trim":{"exposure_ev":0.8}},{"id":3,"kind":"corrector","masks":{"invert":false,"items":[{"angle_deg":-27.58758758758759,"center":[0.5,0.5],"feather":0.3,"kind":"linear","mode":"add"}]},"pos":[3,0.0],"trim":{"exposure_ev":0.8}},{"film":{"exposure_ev":0.3},"grain":{"amount":100,"seed":4},"halation":{"amount":200,"size":120},"id":4,"kind":"film","pos":[4,0.0],"strength":85},{"film":{"exposure_ev":0.3},"grain":{"amount":100,"seed":5},"halation":{"amount":200,"size":120},"id":5,"kind":"film","pos":[5,0.0],"strength":85}],"links":[{"from":{"node":"input","port":"picture"},"to":{"node":1,"port":"picture"}},{"from":{"node":1,"port":"picture"},"to":{"node":2,"port":"picture"}},{"from":{"node":2,"port":"picture"},"to":{"node":3,"port":"picture"}},{"from":{"node":3,"port":"picture"},"to":{"node":4,"port":"picture"}},{"from":{"node":4,"port":"picture"},"to":{"node":5,"port":"picture"}},{"from":{"node":5,"port":"picture"},"to":{"node":"output","port":"picture"}},{"from":{"node":4,"port":"key"},"to":{"node":5,"port":"key"}}]}}"#;

#[test]
fn seed_539_graph_0_export_is_judged_by_the_unstable_pixel_rule() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let settings = serde_json::from_str::<DevelopSettings>(SEED_539_GRAPH_0).expect("settings");
    let home = tempfile::tempdir().expect("dir");
    let (dev_on, notes, dev) = export(
        &d,
        &settings,
        renderer(Some(Arc::clone(&gpu)), 256),
        home.path(),
        FormatKind::Png,
    );
    assert!(
        dev_on == ComputeDevice::Gpu && !notes.iter().any(|n| n.contains("graphics card")),
        "the export is {dev_on:?}, notes {notes:?}"
    );
    let (_, _, cpu_bytes) = export(
        &d,
        &settings,
        renderer(None, 256),
        home.path(),
        FormatKind::Png,
    );
    let (dev, cpu) = (decode_png16_bytes(&dev), decode_png16_bytes(&cpu_bytes));
    let max = dev
        .iter()
        .zip(&cpu)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    say(&format!(
        "seed 539 graph 0: PNG export {max} of 65535 from the CPU's, {} values",
        dev.len()
    ));
    let verdict = judge_export(&d, &settings, &dev, &cpu, home.path(), "seed 539 graph 0");
    assert!(
        verdict.failure.is_none(),
        "seed 539 graph 0: the PNG export fails the bound: {}",
        verdict.failure.unwrap_or_default()
    );
    // What the Mac run saw (222 of 65535 at a few values, on lavapipe 180 on the pictures before RC-13), with a CPU render for the
    // device: the three values the device differs most at, moved to 180 from the CPU's (at most 65535).
    let mut order: Vec<usize> = (0..dev.len()).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(dev[i].abs_diff(cpu[i])));
    let mut standin = cpu.clone();
    for &i in &order[..3] {
        standin[i] = if dev[i] >= cpu[i] {
            (cpu[i] + 180).min(65535)
        } else {
            cpu[i].saturating_sub(180)
        };
    }
    let verdict = judge_export(
        &d,
        &settings,
        &standin,
        &cpu,
        home.path(),
        "seed 539 graph 0, 180",
    );
    assert!(
        verdict.failure.is_none(),
        "seed 539 graph 0 at 180 codes: {}",
        verdict.failure.unwrap_or_default()
    );
}

/// The bindings of the size sweep, in KiB: from 1 MiB up (`docs/23` section 5.12, *Limits*). The reference is the frame at
/// the default binding, which holds a whole view.
const SWEEP_KIB: [u64; 2] = [1024, 4096];

/// G-11, band selection on the scheduler: for every generated graph and every view, at bindings from 1 MiB up, the route is
/// the device's with no size refusal. A frame that fits the device's memory is cut into bands or tiles by the areas the
/// scheduler infers, until every binding fits; the CPU is only for a device error. The bands and tiles each view was cut
/// into are printed (`StageRuns::color_passes`, one per band or tile), and the frames are within the view bound (2 codes, a few values over 1) of
/// the same view at the default binding.
#[test]
fn every_generated_graph_stays_on_the_device_at_any_binding_size() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let started = std::time::Instant::now();
    let mut problems: Vec<String> = Vec::new();
    let (mut frames, mut cut_frames, mut most_passes, mut floors) = (0usize, 0usize, 0u64, 0usize);
    let mut worst = (0u8, 0usize);
    // `YM_SWEEP_ONLY=<seed>:<index>` replays one graph.
    let only = std::env::var("YM_SWEEP_ONLY").ok();
    for g in &device_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        if only
            .as_deref()
            .is_some_and(|o| o != format!("{}:{}", g.seed, g.index))
        {
            continue;
        }
        let src = d.source().with_settings(g.settings.clone());
        // A live Face Refinement node is here too since S-13: its face step is the device's, and the faces are found first.
        // Fit, 100 % and 147 % (scale 1, the owner's frames), and the mask overlay at 100 %.
        let mut states: Vec<(String, ViewportState)> = vec![
            ("fit".to_owned(), view(false, false, None)),
            ("100 %".to_owned(), view(true, false, None)),
        ];
        let mut st = view(false, false, None);
        st.mode = ViewportMode::Percent { value: 147.0 };
        states.push(("147 %".to_owned(), st));
        if let Some((node, row)) = g.masked {
            states.push((
                format!("Show mask row {row}, 100 %"),
                view(
                    true,
                    false,
                    Some((node, MaskStyle::Overlay, Some(row as u32))),
                ),
            ));
        }
        let reference = renderer(Some(Arc::clone(&gpu)), 256);
        let _ = frame(&reference, &src, &states[0].1);
        // A viewer never waits for a Depth row's map (the row selects nothing until the worker has made it, as in the app), so
        // every renderer compared is let finish its map first: else a frame drawn before the map and one drawn after differ.
        settle_depth(&reference, &src, &states);
        if g.face_live {
            settle_faces(&reference, &src, &states);
        }
        let wants: Vec<RenderedFrame> = states
            .iter()
            .map(|(_, st)| frame(&reference, &src, st))
            .collect();
        for kib in SWEEP_KIB {
            let device = renderer_with(Some(Arc::clone(&gpu)), 256, kib * 1024);
            let _ = frame(&device, &src, &states[0].1);
            settle_depth(&device, &src, &states);
            if g.face_live {
                settle_faces(&device, &src, &states);
            }
            let mut line = Vec::new();
            for ((what, st), want) in states.iter().zip(&wants) {
                let at = format!("{name}, binding {kib} KiB: {what}");
                let got = frame(&device, &src, st);
                frames += 1;
                let passes = got.report.runs.color_passes;
                cut_frames += usize::from(passes > 1);
                most_passes = most_passes.max(passes);
                line.push(format!("{what} {passes}"));
                // The one resource limit, a floor: the planner found no cut. The pieces of a view are the piece grown by the summed
                // reach of the spatial steps (Film, Face Refinement and Blur alike) and the raster rows before them, so a binding
                // under the smallest piece's pictures leaves no cut of at most 256 pieces that fits, and the CPU makes the frame
                // (`RenderReport::cut_refused`: `band_split` ended with no shape, and no device pass ran). The report says it,
                // not the graph's node kinds. A size refusal after the planner accepted a cut is a miscount of the planner and
                // always a failure (CI run 44869390323), whatever the nodes.
                let floor = got.report.device != ComputeDevice::Gpu
                    && got.report.cut_refused
                    && got.report.device_refused
                    && got.report.cpu_reason.as_deref().is_some_and(|r| {
                        r.contains("byte buffer is larger than the adapter's limit")
                    })
                    && got.report.notes.iter().any(|n| n.contains("graphics card"));
                if floor {
                    floors += 1;
                    line.push(format!(
                        "{what} CPU, the reach fills the binding: {:?}",
                        got.report.cpu_reason
                    ));
                    continue;
                }
                if got.report.device != ComputeDevice::Gpu
                    || got.report.device_refused
                    || got.report.cpu_reason.is_some()
                {
                    problems.push(format!(
                        "{at}: the frame is {:?}, refused {}, reason {:?}, notes {:?}",
                        got.report.device,
                        got.report.device_refused,
                        got.report.cpu_reason,
                        got.report.notes
                    ));
                    continue;
                }
                if device.views_on_cpu(src.id) {
                    problems.push(format!("{at}: views_on_cpu"));
                }
                // (Subject and People masks are made by a background worker: such a graph's frames are not compared.)
                if g.compare {
                    let (max, over) = frame_diff(&got, want);
                    worst = (worst.0.max(max), worst.1.max(over));
                    if max > 2 || over >= got.rgba.len() / 2000 {
                        problems.push(format!(
                            "{at}: {max} codes from the default binding's frame, {over} values over 1"
                        ));
                    }
                }
            }
            say(&format!(
                "{name}, binding {kib} KiB: passes per view [{}]",
                line.join(", ")
            ));
        }
    }
    say(&format!(
        "binding sweep: {frames} frames at {SWEEP_KIB:?} KiB, {cut_frames} cut into bands or tiles (most passes in one frame: {most_passes}), {floors} on the CPU with the reach filling the binding; worst against the default binding {} codes, {} values over 1; {:?}",
        worst.0,
        worst.1,
        started.elapsed()
    ));
    assert!(problems.is_empty(), "{problems:#?}");
}

/// The ordering behind the binding sweep's Depth failures on a fast device (CI run 42374515284, Metal). A viewer never waits for
/// a Depth row's map: the row selects nothing until the worker has made it. A frame drawn before the map and one drawn after
/// it differ over most of the frame, so a comparison of renderers is only meaningful once each has its map. The model here is
/// held shut until the "before" frames are drawn, which is the order a device that renders a whole sweep inside the debounce
/// produces; [`settle_depth`] is what the sweeps do to get the "after" order on every renderer.
#[test]
fn a_depth_frame_drawn_before_the_map_differs_from_the_frame_after_it() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let Some(g) = graphs().into_iter().find(|g| {
        g.masks.contains(&MaskKind::Depth)
            && g.masked
                .is_some_and(|(_, row)| g.masks[row] == MaskKind::Depth)
    }) else {
        panic!("the generator made no graph with a Show mask on a Depth row");
    };
    let open = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&open);
    let model = DepthModel::fake(
        "gated",
        Arc::new(move |w, h, rgb, cancelled| {
            while !gate.load(Ordering::SeqCst) {
                if cancelled() {
                    return Err("cancelled".to_owned());
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let (nw, nh) = ym_ml::depth::capped_network_size(w, h);
            let (cw, ch) = (nw / ym_ml::depth::PATCH, nh / ym_ml::depth::PATCH);
            let mut v = Vec::with_capacity(cw * ch);
            for cy in 0..ch {
                for cx in 0..cw {
                    let x = (((cx as f64 + 0.5) / cw as f64) * w as f64) as usize;
                    let y = (((cy as f64 + 0.5) / ch as f64) * h as f64) as usize;
                    let p = &rgb[3 * (y.min(h - 1) * w + x.min(w - 1))..][..3];
                    v.push(f32::from(p[1]) / 255.0);
                }
            }
            Ok((cw, ch, v))
        }),
    );
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: None,
            gpu_stages: GpuStages::NONE,
            full_tile_px: 256,
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            depth_model: Some(model),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let src = d.source().with_settings(g.settings.clone());
    let (node, row) = g.masked.expect("a Show mask");
    let st = view(
        true,
        false,
        Some((node, MaskStyle::Overlay, Some(row as u32))),
    );
    let states = vec![("Show mask".to_owned(), st.clone())];
    let before = frame(&r, &src, &st);
    assert!(
        r.depth_worker_alive(src.id),
        "the worker was asked for the map"
    );
    open.store(true, Ordering::SeqCst);
    settle_depth(&r, &src, &states);
    let after = frame(&r, &src, &st);
    let (max, over) = frame_diff(&before, &after);
    assert!(
        over >= after.rgba.len() / 2000,
        "a frame before the map and one after it must differ over the frame ({max} codes, {over} values over 1)"
    );
    // After the map, the frame is stable.
    let again = frame(&r, &src, &st);
    assert_eq!(frame_diff(&after, &again), (0, 0));
}

/// The CPU frames of `a_before_view_makes_no_planes_and_is_not_cut_by_their_reach` (Fit, and 147 %), found before the Before
/// view stopped reading the spatial steps of its chain. The 147 % frame was found again when the viewport began to snap the
/// origin of a pan to a whole device pixel (G-13, `ViewportState::view_to_final`): the same view, up to half a device pixel
/// further on. Fit is unchanged. The 147 % frame was found again at `PIPELINE_VERSION` 0.13.1 (RC-13: the demosaic's A6 `0.9` and
/// A7 `3.5`; the Fit frame reads no full-size demosaic and is unchanged). With A6 and A7 put back to `0.03` and `0.75` and nothing
/// else changed, the old hash `0x94ff_67c7_c4e9_7b96` comes back, so the constants and nothing else moved it.
const EXPECTED_BEFORE_HASHES: [u64; 2] = [0x8d45_9d4b_fe28_cfc1, 0x973d_65e3_a3fb_95f5];

/// G-11 part 3, item 3: the Before view runs no node, so it has no spatial step: its chain reads nothing beyond the view. A
/// graph with raster rows and Film nodes makes no raster planes for it and is not cut by their reach (one piece at the
/// default binding), and its pixels are the pixels it had when it made them and ignored them.
#[test]
fn a_before_view_makes_no_planes_and_is_not_cut_by_their_reach() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let g = parallel_branches(1);
    let src = d.source().with_settings(g.settings.clone());
    let states = || {
        let mut fit = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
        fit.before = true;
        let mut zoomed = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
        zoomed.mode = ViewportMode::Percent { value: 147.0 };
        zoomed.before = true;
        [fit, zoomed]
    };
    let cpu = renderer(None, 256);
    let hashes: Vec<u64> = states()
        .iter()
        .map(|st| fnv(&frame(&cpu, &src, st).rgba))
        .collect();
    say(&format!("Before CPU frame hashes: {hashes:016x?}"));
    assert_eq!(hashes, EXPECTED_BEFORE_HASHES);
    let Some(gpu) = gpu() else { return };
    // The default binding.
    for st in states() {
        // A fresh renderer, and the first frame: the planes are counted when they are made, not when they are served.
        let device = renderer(Some(Arc::clone(&gpu)), 256);
        let before = gpu.counters();
        let got = frame(&device, &src, &st);
        let used = gpu.counters().since(before);
        say(&format!(
            "Before device frame hash {:016x}, plane builds {}, pieces {}",
            fnv(&got.rgba),
            used.plane_builds,
            got.report.runs.color_passes
        ));
        assert_eq!(got.report.device, ComputeDevice::Gpu);
        assert_eq!(used.plane_builds, 0, "a Before view makes no raster planes");
        assert_eq!(got.report.runs.color_passes, 1, "one piece");
        let (max, over) = frame_diff(&got, &frame(&cpu, &src, &st));
        assert!(
            max <= 2 && over < got.rgba.len() / 2000,
            "{max} codes, {over} over 1"
        );
    }
}

/// FNV-1a over a frame's bytes.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// G-11 part 3, item 2: giving each Film node only its ancestors' pictures changes the areas, never a pixel. The frames of
/// the two-branch graph on the CPU reference (Fit and 100 %) have the hashes they had when a Film node read every earlier
/// picture; the device's frames are within the view bound of them.
#[test]
fn parallel_film_branches_keep_their_pixels() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let g = parallel_branches(1);
    let src = d.source().with_settings(g.settings.clone());
    let cpu = renderer(None, 256);
    let states = || {
        let mut zoomed = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
        zoomed.mode = ViewportMode::Percent { value: 147.0 };
        [
            ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0)),
            zoomed,
        ]
    };
    let hashes: Vec<u64> = states()
        .iter()
        .map(|st| fnv(&frame(&cpu, &src, st).rgba))
        .collect();
    say(&format!("two-branch CPU frame hashes: {hashes:016x?}"));
    assert_eq!(hashes, EXPECTED_BRANCH_HASHES);
    if let Some(gpu) = gpu() {
        let device = renderer(Some(gpu), 256);
        for st in states() {
            let _ = frame(&device, &src, &st);
            let got = frame(&device, &src, &st);
            say(&format!(
                "two-branch device frame hash {:016x}",
                fnv(&got.rgba)
            ));
            assert_eq!(got.report.device, ComputeDevice::Gpu);
            let (max, over) = frame_diff(&got, &frame(&cpu, &src, &st));
            assert!(
                max <= 2 && over < got.rgba.len() / 2000,
                "{max} codes, {over} over 1"
            );
        }
    }
}

/// The CPU frames of `parallel_film_branches_keep_their_pixels`, found with every Film node reading every earlier picture.
/// The zoomed frame was found again when the viewport began to snap the origin of a pan to a whole device pixel (G-13); the
/// Fit frame is unchanged. It was found again at `PIPELINE_VERSION` 0.13.1 (RC-13: A6 `0.9` and A7 `3.5` of the demosaic); with A6 and
/// A7 put back to `0.03` and `0.75` and nothing else changed, the old hash `0x3e9e_d1c8_69ff_71fd` comes back, so the constants and
/// nothing else moved it.
const EXPECTED_BRANCH_HASHES: [u64; 2] = [0x9c79_44fc_79cf_83bd, 0x6ac4_1b30_96d7_4bd7];

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

/// The settings schema: the range and default of every real number of a settings document, read by [`nudge_leaves`].
const SETTINGS_SCHEMA: &str = include_str!("../schema/settings.schema.json");
/// How far each continuous setting is nudged to see whether a value of the CPU's own export is stable: this fraction of the
/// setting's own range (1e-5 absolute where the schema gives no range).
const NUDGE: f64 = 1e-5;
/// How many renders nudge every continuous setting at once, each by plus or minus its step with signs from [`NUDGE_SEED`].
const COMBINED_NUDGES: usize = 8;
/// The seed of the signs of the combined nudges: fixed, so a verdict is the same on every run.
const NUDGE_SEED: u64 = 0x756e_7374_6162_6c65;
/// The most unstable values a thin edge can account for: more than this (or 0.01 % of the export's values, if larger) fails.
const MOST_UNSTABLE: usize = 60;

/// The verdict of [`judge_export`].
struct Judged {
    /// Why the export fails the bound, `None` when it holds.
    failure: Option<String>,
}

/// One continuous setting of a settings document: where it is, its value, how far it is nudged and the range it is clamped to.
#[derive(Clone, Debug)]
struct Leaf {
    /// The JSON pointer of the number.
    at: String,
    /// Its value (the schema's default when the document leaves it out).
    value: f64,
    /// The size of a nudge.
    step: f64,
    /// The range of the setting (infinite where the schema gives none).
    lo: f64,
    hi: f64,
}

impl Leaf {
    /// The value moved by `sign` steps, kept in the setting's range.
    fn nudged(&self, sign: f64) -> f64 {
        (self.value + sign * self.step).clamp(self.lo, self.hi)
    }
}

/// The schema a `$ref` names (the schema itself without one).
fn resolve<'a>(root: &'a Value, schema: &'a Value) -> &'a Value {
    static NULL: Value = Value::Null;
    let mut s = schema;
    for _ in 0..8 {
        let Some(r) = s.get("$ref").and_then(Value::as_str) else {
            break;
        };
        s = root.pointer(r.trim_start_matches('#')).unwrap_or(&NULL);
    }
    s
}

/// Whether `v` is a value of the `oneOf` variant `variant`: its type, its constants (`kind`, `id`) and its required properties agree.
fn variant_fits(root: &Value, variant: &Value, v: &Value) -> bool {
    let s = resolve(root, variant);
    match s.get("type").and_then(Value::as_str) {
        Some("null") => v.is_null(),
        Some("array") => v.is_array(),
        Some("number" | "integer") => v.is_number(),
        Some("string") => v.is_string(),
        Some("boolean") => v.is_boolean(),
        Some("object") => {
            let required = s.get("required").and_then(Value::as_array);
            let props = s.get("properties").and_then(Value::as_object);
            v.is_object()
                && required.is_none_or(|r| {
                    r.iter()
                        .all(|k| k.as_str().is_some_and(|k| v.get(k).is_some()))
                })
                && props.is_none_or(|p| {
                    p.iter()
                        .all(|(k, ps)| match (resolve(root, ps).get("const"), v.get(k)) {
                            (Some(c), Some(x)) => c == x,
                            _ => true,
                        })
                })
        }
        _ => true,
    }
}

/// The real numbers of `value` under `schema` (and the ones the document leaves out where the schema has a number for them).
fn schema_leaves(
    root: &Value,
    schema: &Value,
    value: Option<&Value>,
    at: &str,
    out: &mut Vec<Leaf>,
) {
    let s = resolve(root, schema);
    if let Some(variants) = s
        .get("oneOf")
        .or_else(|| s.get("anyOf"))
        .and_then(Value::as_array)
    {
        if let Some(v) = value
            && let Some(variant) = variants.iter().find(|x| variant_fits(root, x, v))
        {
            schema_leaves(root, variant, Some(v), at, out);
        }
        return;
    }
    match s.get("type").and_then(Value::as_str) {
        Some("number") => {
            let value = match value {
                Some(v) => v.as_f64(),
                None => s.get("default").and_then(Value::as_f64),
            };
            let Some(value) = value else { return };
            let lo = s.get("minimum").and_then(Value::as_f64);
            let hi = s.get("maximum").and_then(Value::as_f64);
            let step = match (lo, hi) {
                (Some(lo), Some(hi)) if hi > lo => NUDGE * (hi - lo),
                _ => NUDGE,
            };
            out.push(Leaf {
                at: at.to_owned(),
                value,
                step,
                lo: lo.unwrap_or(f64::NEG_INFINITY),
                hi: hi.unwrap_or(f64::INFINITY),
            });
        }
        Some("object") => {
            if let Some(props) = s.get("properties").and_then(Value::as_object) {
                for (k, ps) in props {
                    schema_leaves(
                        root,
                        ps,
                        value.and_then(|v| v.get(k)),
                        &format!("{at}/{k}"),
                        out,
                    );
                }
            }
        }
        Some("array") => {
            if let Some(items) = value.and_then(Value::as_array) {
                let prefix = s.get("prefixItems").and_then(Value::as_array);
                for (i, x) in items.iter().enumerate() {
                    let item = prefix
                        .and_then(|p| p.get(i))
                        .or_else(|| s.get("items").filter(|x| x.is_object()));
                    if let Some(item) = item {
                        schema_leaves(root, item, Some(x), &format!("{at}/{i}"), out);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Every continuous setting of a settings document, whatever node it belongs to: the numbers the schema types as `number` (those the
/// document gives and those it leaves at a numeric default), then any other real number the document holds (1e-5 absolute). Integers,
/// enums, booleans, seeds and ids are not among them. Nothing here names a node kind or a setting.
fn nudge_leaves(doc: &Value) -> Vec<Leaf> {
    let mut out = Vec::new();
    if let Ok(root) = serde_json::from_str::<Value>(SETTINGS_SCHEMA) {
        schema_leaves(&root, &root, Some(doc), "", &mut out);
    }
    fn others(v: &Value, at: String, known: &BTreeSet<String>, out: &mut Vec<Leaf>) {
        match v {
            Value::Number(n) if n.is_f64() && !known.contains(&at) => out.push(Leaf {
                value: n.as_f64().unwrap_or_default(),
                at,
                step: NUDGE,
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            }),
            Value::Array(a) => a
                .iter()
                .enumerate()
                .for_each(|(i, x)| others(x, format!("{at}/{i}"), known, out)),
            Value::Object(o) => o
                .iter()
                .for_each(|(k, x)| others(x, format!("{at}/{k}"), known, out)),
            _ => {}
        }
    }
    let known: BTreeSet<String> = out.iter().map(|l| l.at.clone()).collect();
    others(doc, String::new(), &known, &mut out);
    out
}

/// Sets the number at a JSON pointer, making the objects on the way when the document leaves them out.
fn set_number(doc: &mut Value, at: &str, to: f64) {
    let mut cur = doc;
    for tok in at.split('/').skip(1) {
        cur = match cur {
            Value::Array(a) => match tok.parse::<usize>().ok().and_then(|i| a.get_mut(i)) {
                Some(x) => x,
                None => return,
            },
            other => {
                if !other.is_object() {
                    *other = json!({});
                }
                match other.as_object_mut() {
                    Some(o) => o.entry(tok.to_owned()).or_insert(Value::Null),
                    None => return,
                }
            }
        };
    }
    *cur = json!(to);
}

/// The CPU's PNG export of a settings document, as 16-bit values (`None` when the document is not a valid graph or the size differs).
fn cpu_png_of(d: &Developed, doc: &Value, home: &Path, len: usize) -> Option<Vec<u32>> {
    let valid = serde_json::from_value::<DevelopSettings>(doc.clone())
        .ok()?
        .validated()
        .ok()?;
    let png = export(d, &valid.0, renderer(None, 256), home, FormatKind::Png).2;
    let values = decode_png16_bytes(&png);
    (values.len() == len).then_some(values)
}

/// The device's PNG export against the CPU's under the 10-bit bound (`TEN_BIT_CODE`), with the owner's rule of 2026-11-18, "Skip
/// unstable pixels", made general on 2026-11-21 ("extend the rule so there are no exceptions for specific cases"): the bound holds on
/// every value except those where the CPU's own export is unstable, and those are printed.
///
/// * Within the bound everywhere: nothing more is done (no nudging, no time).
/// * Else the CPU renders the settings again with the continuous settings of **every node** (see [`nudge_leaves`]: each real number
///   of the document, the Input's and Main's included, each by [`NUDGE`] of its own range) moved: [`COMBINED_NUDGES`] renders that
///   move all of them at once by plus or minus a step with seeded signs. A value is **unstable** when any of those renders moves the CPU's own value there by more than the
///   bound. The bound is enforced on every other value; each unstable value over the bound is printed with its index, the CPU's
///   value, the device's, and the CPU's own range.
/// * The safeguards that keep this from hiding a bug: (a) when nothing could be nudged, nothing is exempt; (b) more than
///   `max(60, 0.01 % of the values)` exempt values fail, too many for a thin edge; (c) the device's value at an unstable index
///   must lie within the CPU's nudged range widened by the bound.
fn judge_export(
    d: &Developed,
    settings: &DevelopSettings,
    dev: &[u32],
    cpu: &[u32],
    home: &Path,
    name: &str,
) -> Judged {
    let diff = |a: u32, b: u32| a.abs_diff(b);
    let over: Vec<usize> = (0..dev.len())
        .filter(|&i| diff(dev[i], cpu[i]) > TEN_BIT_CODE)
        .collect();
    if over.is_empty() {
        return Judged { failure: None };
    }
    let fail = |text: String| Judged {
        failure: Some(format!("{name}: {text}")),
    };
    let Ok(doc) = serde_json::to_value(settings) else {
        return fail(format!(
            "{} values over {TEN_BIT_CODE}; the settings cannot be written as JSON to nudge them",
            over.len()
        ));
    };
    judge_export_with(d, &doc, &nudge_leaves(&doc), dev, cpu, home, name)
}

/// [`judge_export`] with the settings to nudge given: the document and its continuous settings (a test gives none to see the
/// safeguard that nothing nudged means nothing exempt).
fn judge_export_with(
    d: &Developed,
    doc: &Value,
    leaves: &[Leaf],
    dev: &[u32],
    cpu: &[u32],
    home: &Path,
    name: &str,
) -> Judged {
    let diff = |a: u32, b: u32| a.abs_diff(b);
    let over: Vec<usize> = (0..dev.len())
        .filter(|&i| diff(dev[i], cpu[i]) > TEN_BIT_CODE)
        .collect();
    if over.is_empty() {
        return Judged { failure: None };
    }
    let fail = |text: String| Judged {
        failure: Some(format!("{name}: {text}")),
    };
    // The CPU's own range at every value, and how far a nudge moved it.
    let (mut lo, mut hi) = (cpu.to_vec(), cpu.to_vec());
    let mut renders = 0usize;
    let mut take = |doc: &Value| {
        let Some(moved) = cpu_png_of(d, doc, home, cpu.len()) else {
            return;
        };
        renders += 1;
        for (i, m) in moved.iter().enumerate() {
            lo[i] = lo[i].min(*m);
            hi[i] = hi[i].max(*m);
        }
    };
    // All the settings at once, with seeded signs.
    for nudged in nudged_documents(doc, leaves) {
        take(&nudged);
    }
    if leaves.is_empty() || renders == 0 {
        return fail(format!(
            "{} values over {TEN_BIT_CODE} (largest {}) and nothing could be nudged, so nothing is exempt",
            over.len(),
            over.iter()
                .map(|&i| diff(dev[i], cpu[i]))
                .max()
                .unwrap_or(0)
        ));
    }
    let moved = |i: usize| (hi[i] - cpu[i]).max(cpu[i] - lo[i]);
    let unstable = (0..dev.len()).filter(|&i| moved(i) > TEN_BIT_CODE).count();
    let mut problems = Vec::new();
    let stable_over: Vec<usize> = over
        .iter()
        .copied()
        .filter(|&i| moved(i) <= TEN_BIT_CODE)
        .collect();
    if !stable_over.is_empty() {
        let worst = stable_over
            .iter()
            .map(|&i| (diff(dev[i], cpu[i]), i))
            .max()
            .unwrap_or((0, 0));
        problems.push(format!(
            "{} stable values differ from the CPU's by more than {TEN_BIT_CODE} (largest {} at value {})",
            stable_over.len(),
            worst.0,
            worst.1
        ));
    }
    let exempt = over.len() - stable_over.len();
    let most = MOST_UNSTABLE.max(dev.len() / 10_000);
    if exempt > most {
        problems.push(format!(
            "{exempt} values over {TEN_BIT_CODE} would be exempt as unstable (more than {most}): too many for a thin edge"
        ));
    }
    for &i in over.iter().filter(|&&i| moved(i) > TEN_BIT_CODE) {
        say(&format!(
            "{name}: unstable value {i} (not enforced): CPU {}, device {}, the CPU's own range with its settings moved by {NUDGE:e} of their ranges: {} to {}",
            cpu[i], dev[i], lo[i], hi[i]
        ));
        if dev[i] + TEN_BIT_CODE < lo[i] || dev[i] > hi[i] + TEN_BIT_CODE {
            problems.push(format!(
                "the device's value {} at unstable value {i} is outside the CPU's own range {} to {} widened by {TEN_BIT_CODE}",
                dev[i], lo[i], hi[i]
            ));
        }
    }
    say(&format!(
        "{name}: {} values over {TEN_BIT_CODE}, {} of them exempt as unstable ({unstable} unstable in all, {} settings nudged, {renders} CPU renders)",
        over.len(),
        exempt,
        leaves.len()
    ));
    Judged {
        failure: (!problems.is_empty()).then(|| format!("{name}: {}", problems.join("; "))),
    }
}

/// The settings document with every continuous setting moved at once by plus or minus a step, [`COMBINED_NUDGES`] times, the signs
/// drawn from [`NUDGE_SEED`]: the documents the CPU renders again to see what is unstable.
fn nudged_documents(doc: &Value, leaves: &[Leaf]) -> Vec<Value> {
    let mut rng = SplitMix64(NUDGE_SEED);
    (0..COMBINED_NUDGES)
        .map(|_| {
            let mut nudged = doc.clone();
            for leaf in leaves {
                let sign = if rng.below(2) == 0 { 1.0 } else { -1.0 };
                set_number(&mut nudged, &leaf.at, leaf.nudged(sign));
            }
            nudged
        })
        .collect()
}

/// The CPU's own range of every value of a view (codes of 255, alpha included) over the nudged settings of `leaves`, and how many
/// renders there were. The CPU renders the same view state of the same photo with the settings changed, as the app does when a
/// control moves (`PhotoSource::with_settings`).
fn view_ranges(
    cpu: &Renderer,
    src: &PhotoSource,
    st: &ViewportState,
    doc: &Value,
    leaves: &[Leaf],
    base: &RenderedFrame,
) -> (Vec<u8>, Vec<u8>, usize) {
    let (mut lo, mut hi) = (base.rgba.clone(), base.rgba.clone());
    let mut renders = 0usize;
    if leaves.is_empty() {
        return (lo, hi, renders);
    }
    for nudged in nudged_documents(doc, leaves) {
        let Some(valid) = serde_json::from_value::<DevelopSettings>(nudged)
            .ok()
            .and_then(|s| s.validated().ok())
        else {
            continue;
        };
        let moved = frame(cpu, &src.with_settings(valid.0), st);
        if moved.rgba.len() != base.rgba.len() {
            continue;
        }
        renders += 1;
        for (i, m) in moved.rgba.iter().enumerate() {
            lo[i] = lo[i].min(*m);
            hi[i] = hi[i].max(*m);
        }
    }
    (lo, hi, renders)
}

/// The device's 8-bit view against the CPU's under the view bound (at most 2 codes, and fewer than a 2000th of the values over 1),
/// with the unstable-pixel rule of the export applied to the view (the owner, 2026-11-21: "extend the rule so there are no exceptions
/// for specific cases").
///
/// * Within the bound: nothing more is done.
/// * Else the CPU renders the same view state [`COMBINED_NUDGES`] times with the continuous settings of every node ([`nudge_leaves`])
///   nudged at once with seeded signs, and a value that any of them moves by more than 1 code is **unstable**. Unstable values are
///   left out of both the bound on the largest difference and the count of values over 1; each unstable value the device differs at
///   by more than 1 code is printed.
/// * Safeguards: when nothing could be nudged nothing is exempt; more than `max(60, 0.01 % of the values)` exempt values fail; and the
///   device's value at an exempt index must lie within the CPU's nudged range widened by 2 codes.
fn judge_view(
    cpu: &Renderer,
    src: &PhotoSource,
    st: &ViewportState,
    dev: &RenderedFrame,
    want: &RenderedFrame,
    name: &str,
    judged_views: &mut usize,
) -> Judged {
    let (max, over) = frame_diff(dev, want);
    if max <= 2 && over < dev.rgba.len() / 2000 {
        return Judged { failure: None };
    }
    *judged_views += 1;
    let Ok(doc) = serde_json::to_value(&src.settings) else {
        return Judged {
            failure: Some(format!(
                "{name}: against the CPU {max} codes, {over} values over 1; the settings cannot be written as JSON to nudge them"
            )),
        };
    };
    let leaves = nudge_leaves(&doc);
    let (lo, hi, renders) = view_ranges(cpu, src, st, &doc, &leaves, want);
    judge_view_with(dev, want, &lo, &hi, renders, name)
}

/// [`judge_view`] with the CPU's nudged range given (`renders` is how many renders made it; none means nothing nudged).
fn judge_view_with(
    dev: &RenderedFrame,
    want: &RenderedFrame,
    lo: &[u8],
    hi: &[u8],
    renders: usize,
    name: &str,
) -> Judged {
    let n = dev.rgba.len();
    let diff = |i: usize| dev.rgba[i].abs_diff(want.rgba[i]);
    let unstable = |i: usize| (hi[i] - want.rgba[i]).max(want.rgba[i] - lo[i]) > 1;
    if renders == 0 {
        let (max, over) = frame_diff(dev, want);
        return Judged {
            failure: Some(format!(
                "{name}: against the CPU {max} codes, {over} values over 1, and nothing could be nudged, so nothing is exempt"
            )),
        };
    }
    let (mut max, mut over, mut exempt) = (0u8, 0usize, Vec::new());
    for i in 0..n {
        if diff(i) > 1 && unstable(i) {
            exempt.push(i);
        } else if !unstable(i) {
            max = max.max(diff(i));
            over += usize::from(diff(i) > 1);
        }
    }
    let mut problems = Vec::new();
    if max > 2 || over >= n / 2000 {
        problems.push(format!(
            "the values the CPU holds still differ by {max} codes, {over} over 1 (bound 2 codes, fewer than {} over 1)",
            n / 2000
        ));
    }
    let most = MOST_UNSTABLE.max(n / 10_000);
    if exempt.len() > most {
        problems.push(format!(
            "{} values over 1 would be exempt as unstable (more than {most}): too many for a thin edge",
            exempt.len()
        ));
    }
    for &i in &exempt {
        say(&format!(
            "{name}: unstable value {i} (not enforced): CPU {}, device {}, the CPU's own range {} to {}",
            want.rgba[i], dev.rgba[i], lo[i], hi[i]
        ));
        if u16::from(dev.rgba[i]) + 2 < u16::from(lo[i]) || dev.rgba[i] > hi[i].saturating_add(2) {
            problems.push(format!(
                "the device's value {} at unstable value {i} is outside the CPU's own range {} to {} widened by 2",
                dev.rgba[i], lo[i], hi[i]
            ));
        }
    }
    say(&format!(
        "{name}: view against the CPU: {} values over 1 excused as unstable, {renders} CPU renders",
        exempt.len()
    ));
    Judged {
        failure: (!problems.is_empty()).then(|| format!("{name}: {}", problems.join("; "))),
    }
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
    // The fast default: the 640 px photo at every size and zoom, and one case of the 4000 px photo (two raster rows, at the
    // smallest binding that holds the frame, at 100 %), which is the owner's frame and the one the Metal watchdog lost.
    let cases: [Case; 2] = [
        (
            (640, 420),
            (300.0, 200.0),
            &[65_536, 4096, 1024, 512, 256, 128, 64],
            1024,
        ),
        ((4000, 2668), (1332.0, 888.0), &[4096], 4096),
    ];
    scale_one_cases(
        &gpu,
        &cases,
        &[50.0, 75.0, 100.0, 147.0],
        &|pw, what, zoom| pw < 1000 || (what == "brush and background" && zoom == 100.0),
    );
}

/// The rest of the scale-1 binding test (`#[ignore]`d: on lavapipe the 4000 px photo takes minutes): the 1332 x 888 view of the
/// 4000 px photo with every kind of row, at every binding from 4 MiB (16 MiB with a Film node) and every zoom. Run it with
/// `cargo test -p ym-core --release --test any_graph -- --ignored scale_one_4000`. The Mac CI's ignored step
/// (`.github/workflows/mac-gpu.yml`) names its tests by filter and does not name this one, so it runs by hand.
#[test]
#[ignore = "minutes on lavapipe; run by hand or on the Mac"]
fn views_at_scale_one_with_raster_masks_stay_on_the_device_scale_one_4000_px() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let cases: [Case; 1] = [((4000, 2668), (1332.0, 888.0), &[65_536, 16_384, 4096], 4096)];
    scale_one_cases(&gpu, &cases, &[50.0, 75.0, 100.0, 147.0], &|_, _, _| true);
}

/// (photo, the cell, the bindings in KiB, the smallest binding the device must make the frame at)
type Case = ((u32, u32), (f64, f64), &'static [u64], u64);

/// The body of the scale-1 binding tests: every case, kind of row, binding and zoom for which `wanted(photo width, kind, zoom)`
/// holds must be made by the device, or by the CPU with its reason below `device_from`.
fn scale_one_cases(
    gpu: &Arc<Gpu>,
    cases: &[Case],
    zooms: &[f64],
    wanted: &dyn Fn(u32, &str, f64) -> bool,
) {
    let white = light("sony-ilce-7rm4", 4800.0);
    let mut problems = Vec::new();
    for &((pw, ph), (cw, ch), sizes, device_from) in cases {
        let scene = Scene::new(pw, ph, white, Layout::Patchwork, 0.5)
            .noise(2e-4, 2e-6)
            .seed(5);
        let d = Developed::new(&scene, MODEL);
        let mut rng = SplitMix64(9);
        let rows = |kinds: &[MaskKind], rng: &mut SplitMix64| -> Vec<Value> {
            kinds.iter().map(|k| row_json(*k, rng)).collect()
        };
        for (what, kinds, film) in [
            ("brush", vec![MaskKind::Brush], false),
            (
                "brush and background",
                vec![MaskKind::Brush, MaskKind::Background],
                false,
            ),
            ("brush and a Film node", vec![MaskKind::Brush], true),
        ] {
            let mut items = vec![
                json!({ "id": 1 }),
                json!({ "id": 2, "kind": "corrector", "pos": [2, 0.0], "trim": { "exposure_ev": 0.8 },
                  "masks": { "invert": false, "items": rows(&kinds, &mut rng) } }),
            ];
            let mut links = vec![
                json!({ "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } }),
                json!({ "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } }),
            ];
            let mut last = json!(2);
            if film {
                items.push(
                    json!({ "id": 3, "kind": "film", "pos": [3, 0.0], "film": { "exposure_ev": 0.3 } }),
                );
                links.push(json!({ "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } }));
                last = json!(3);
            }
            links.push(json!({ "from": { "node": last, "port": "picture" }, "to": { "node": "output", "port": "picture" } }));
            let doc = json!({ "nodes": { "items": items, "links": links } });
            let settings = serde_json::from_value::<DevelopSettings>(doc)
                .expect("a document")
                .validated()
                .expect("a valid graph")
                .0;
            let src = d.source().with_settings(settings);
            for &kib in sizes {
                // The wider picture of a Film node is the tile grown by its halos on every side (about 300 px here): a tile
                // needs a binding that holds more than that, 16 MiB for the 1332 x 888 view.
                if film && pw > 1000 && kib < 16_384 {
                    continue;
                }
                let device = renderer_with(Some(Arc::clone(gpu)), 256, kib * 1024);
                let _ = frame(&device, &src, &view(false, false, None));
                for &pct in zooms {
                    if !wanted(pw, what, pct) {
                        continue;
                    }
                    let mut st = ViewportState::new(ViewportRect::cell(cw, ch, 1.0));
                    st.mode = ViewportMode::Percent { value: pct };
                    let got = frame(&device, &src, &st);
                    let on_cpu = got.report.device != ComputeDevice::Gpu;
                    // The same frame in the biggest binding: the tiles are the view in one piece (the stage of a tiled Film
                    // view runs on the CPU, so within the CPU's bound and not bit for bit).
                    if film && !on_cpu && kib < sizes[0] {
                        let reference = renderer_with(Some(Arc::clone(gpu)), 256, sizes[0] * 1024);
                        let want = frame(&reference, &src, &st);
                        let (max, over) = frame_diff(&got, &want);
                        if max > 2 || over >= got.rgba.len() / 2000 {
                            problems.push(format!(
                                "{pw} px, {what}, binding {kib} KiB, zoom {pct} %: {max} codes from the big binding's frame, {over} values over 1"
                            ));
                        }
                    }
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

/// A list of passes is cut into submissions of bounded work (G-11, `docs/17` K5): the owner's CI on Metal lost the device in
/// the 4000 px case of the scale-1 test, where one command buffer held the sampler, the planes of two raster rows with Refine
/// edges and the colour pass over a wide picture. Lavapipe has no watchdog, so this counts what a submission holds. With the
/// budget lowered a frame is several submissions, none of more than one dispatch holds more than the budget, and the pixels
/// are the pixels of the frame made with the default budget.
#[test]
fn a_frame_is_submitted_in_pieces_of_bounded_work_with_the_same_pixels() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [2, 0.0], "trim": { "exposure_ev": 0.8 },
          "masks": { "invert": false, "items": [
            { "kind": "brush", "mode": "add", "feather": 0.2, "refine": 0.7,
              "strokes": [{ "erase": false, "hardness": 0.5, "size": 0.3,
                "points": [[0.3, 0.3], [0.6, 0.6]] }] },
            { "kind": "background", "mode": "add", "feather": 0.1, "refine": 0.5 } ] } },
        { "id": 3, "kind": "film", "pos": [3, 0.0], "film": { "exposure_ev": 0.3 } } ],
      "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let src = d.source().with_settings(settings);
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    st.mode = ViewportMode::Percent { value: 100.0 };
    const BUDGET: u64 = 1 << 20;
    gpu.set_work_budget(BUDGET);
    let before = gpu.counters();
    let cut = frame(&renderer(Some(Arc::clone(&gpu)), 256), &src, &st);
    let after = gpu.counters().since(before);
    gpu.set_work_budget(0);
    // (The device's maximum is its own for good: the frame with the default budget is made after the readings.)
    let plain = frame(&renderer(Some(Arc::clone(&gpu)), 256), &src, &st);
    assert_eq!(cut.report.device, ComputeDevice::Gpu);
    assert!(
        after.submissions > 1,
        "the frame was {} submission(s) with a budget of {BUDGET}",
        after.submissions
    );
    assert!(
        after.max_split_work <= BUDGET,
        "a submission of several dispatches held {} invocations, over the budget of {BUDGET}",
        after.max_split_work
    );
    assert!(plain.rgba == cut.rgba, "the pixels changed with the cut");
}

/// Regression (G-11): the device's row bands of a magnified view with a Film node (halation), before it a corrector with a
/// feathered Brush, equal the view made in one piece. The last rows of the last band were wrong (up to 60 codes).
#[test]
fn device_bands_of_a_magnified_film_view_equal_the_view_in_one_piece() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let brush = json!({ "kind": "brush", "mode": "add", "feather": 0.051, "refine": 0.0,
        "strokes": [{ "erase": false, "hardness": 0.5, "size": 0.2,
            "points": [[0.4967, 0.3357], [0.6183, 0.5886]] }] });
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [2, 0.0], "trim": { "exposure_ev": 0.8 },
          "masks": { "invert": false, "items": [brush] } },
        { "id": 3, "kind": "film", "pos": [3, 0.0], "film": { "exposure_ev": 0.3 },
          "halation": { "amount": 50.0 }, "grain": { "amount": 0.0 } } ],
      "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let src = d.source().with_settings(settings);
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    st.mode = ViewportMode::Percent { value: 147.0 };
    let whole = frame(
        &renderer_with(Some(Arc::clone(&gpu)), 256, 64 << 20),
        &src,
        &st,
    );
    let bands = frame(
        &renderer_with(Some(Arc::clone(&gpu)), 256, 1 << 20),
        &src,
        &st,
    );
    assert_eq!(
        bands.report.device,
        ComputeDevice::Gpu,
        "{:?}",
        bands.report.cpu_reason
    );
    // A band is a view of its own and every device stage is tile-exact: the pixels are the view in one piece, bit for bit.
    let (max, over) = frame_diff(&bands, &whole);
    assert!(
        bands.rgba == whole.rgba,
        "the bands differ from the view in one piece by {max} codes, {over} values over 1"
    );
    // The same graph exported (PNG 16): the tiles of 97, 256 and 1024 px are the same bytes.
    let home = tempfile::tempdir().expect("dir");
    let settings = src.settings.clone();
    let mut exports = Vec::new();
    for tile in [256, 97, 1024] {
        let (device, notes, bytes) = export(
            &d,
            &settings,
            renderer(Some(Arc::clone(&gpu)), tile),
            home.path(),
            FormatKind::Png,
        );
        assert_eq!(device, ComputeDevice::Gpu, "tile {tile}: {notes:?}");
        exports.push(bytes);
    }
    assert!(
        exports[0] == exports[1] && exports[0] == exports[2],
        "the PNG 16 export is other bytes at another tile size"
    );
}

// ---- graphs with joins on the CPU (docs/25 task X-04) ---------------------------------------------------------

/// A renderer without a device: `tile` px tiles for the full render and `threads` for the colour pass.
fn cpu_renderer(tile: usize, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            threads,
            full_tile_px: tile,
            gpu: None,
            gpu_stages: GpuStages::NONE,
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// [`cpu_renderer`] with a fake scene model, so that the element grades of a Reference Grade node have their masks (RG-04).
fn cpu_scene_renderer(tile: usize, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            threads,
            full_tile_px: tile,
            gpu: None,
            gpu_stages: GpuStages::NONE,
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            scene_model: Some(ym_core::pipeline::SceneModel::fake_default()),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// Waits for the scene worker of the photo (a viewer never waits for the scene: the element step does nothing until it is made),
/// after the renderer is told the photo is the one looked at and a frame has asked for it.
fn settle_scene(r: &Renderer, src: &PhotoSource, st: &ViewportState) {
    if ym_core::pipeline::color::live_element_nodes(&src.settings, ym_color::plan::Stop::Output)
        .is_empty()
    {
        return;
    }
    r.navigate(src.id);
    let _ = frame(r, src, st);
    let end = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while r.scene_worker_alive(src.id) || !r.scene_ready(src) {
        assert!(
            std::time::Instant::now() < end,
            "the scene worker did not end"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn full_bits(r: &Renderer, src: &PhotoSource, threads: usize) -> Vec<u32> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");
    pool.install(|| {
        r.render_full(
            src,
            &FullRequest {
                target: FullTarget::DisplayLinear(ym_color::display::OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("render")
        .rgb
        .iter()
        .map(|v| v.to_bits())
        .collect()
    })
}

/// Random graphs with the four kinds render on the CPU, the same bits whatever the tile size and the thread count, the same
/// bits again on a repeat, and a view that asks for the mask of a mixer shows the photo.
#[test]
fn graphs_with_joins_are_the_same_bits_whatever_the_tiles_and_the_threads() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let plain = d.source();
    let plain_bits = full_bits(&cpu_renderer(256, 1), &plain, 1);
    let configs = [(97usize, 1usize), (160, 3), (256, 4), (1024, 3)];
    let renderers: Vec<Renderer> = configs.iter().map(|&(t, n)| cpu_renderer(t, n)).collect();
    let mut changed = 0;
    for g in joined_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        say(&format!(
            "{name}: nodes {:?}, masks {:?}",
            g.nodes.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masks.iter().map(|k| k.name()).collect::<Vec<_>>(),
        ));
        let src = d.source().with_settings(g.settings.clone());
        let first = full_bits(&renderers[0], &src, configs[0].1);
        for (r, &(tile, threads)) in renderers.iter().zip(&configs).skip(1) {
            assert!(
                full_bits(r, &src, threads) == first,
                "{name}: tile {tile} on {threads} threads is other bits; settings: {}",
                g.json
            );
        }
        assert!(
            full_bits(&renderers[0], &src, configs[0].1) == first,
            "{name}: a repeat is other bits"
        );
        // The views: the same frame on one and four threads, and Show mask on a mixer is the photo's frame.
        let st = view(false, false, None);
        let a = frame(&renderers[0], &src, &st).rgba;
        assert!(
            a == frame(&renderers[2], &src, &st).rgba,
            "{name}: the Fit frame depends on the threads"
        );
        if let Some(join) = g.nodes.iter().position(|k| JOINED_NODES.contains(k)) {
            let masked = view(
                false,
                false,
                Some((2 + join as u32, MaskStyle::Overlay, None)),
            );
            assert!(
                a == frame(&renderers[0], &src, &masked).rgba,
                "{name}: Show mask on a node with no mask changes the photo"
            );
        }
        changed += usize::from(first != plain_bits);
    }
    assert!(changed > 0, "no graph with joins changed a pixel");
}

/// Every graph with the four kinds is exported on the device (PNG 16 and JPEG): the device makes it, the PNG is within one
/// 10-bit code of the CPU's, a repeat and another tile size give the same bytes.
#[test]
fn graphs_with_joins_are_exported_on_the_device() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let (mut worst, mut changed) = (0u32, 0usize);
    let plain_bytes = {
        let home = tempfile::tempdir().expect("dir");
        export(
            &d,
            &d.source().settings,
            renderer(None, 256),
            home.path(),
            FormatKind::Png,
        )
        .2
    };
    for g in &joined_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        let home = tempfile::tempdir().expect("dir");
        for format in [FormatKind::Png, FormatKind::Jpeg] {
            let (on, notes, dev) = export(
                &d,
                &g.settings,
                renderer(Some(Arc::clone(&gpu)), 256),
                home.path(),
                format,
            );
            assert_eq!(
                on,
                ComputeDevice::Gpu,
                "{name}: the {format:?} export; {notes:?}"
            );
            assert!(
                !notes.iter().any(|n| n.contains("graphics card")),
                "{name}: {notes:?}"
            );
            let (_, _, again) = export(
                &d,
                &g.settings,
                renderer(Some(Arc::clone(&gpu)), 256),
                home.path(),
                format,
            );
            assert!(
                again == dev,
                "{name}: a repeat {format:?} export is other bytes"
            );
            let (_, _, other) = export(
                &d,
                &g.settings,
                renderer(Some(Arc::clone(&gpu)), 97),
                home.path(),
                format,
            );
            assert!(
                other == dev,
                "{name}: the {format:?} export at tile 97 is other bytes"
            );
            if format == FormatKind::Png {
                let (_, _, want) =
                    export(&d, &g.settings, renderer(None, 256), home.path(), format);
                let max = decode_png16_bytes(&dev)
                    .iter()
                    .zip(decode_png16_bytes(&want))
                    .map(|(a, b)| a.abs_diff(b))
                    .max()
                    .unwrap_or(0);
                worst = worst.max(max);
                if max > TEN_BIT_CODE {
                    let judged = judge_export(
                        &d,
                        &g.settings,
                        &decode_png16_bytes(&dev),
                        &decode_png16_bytes(&want),
                        home.path(),
                        &name,
                    );
                    assert!(
                        judged.failure.is_none(),
                        "{}; the PNG export is {max} of 65535 from the CPU's; settings: {}",
                        judged.failure.unwrap_or_default(),
                        g.json
                    );
                }
                changed += usize::from(dev != plain_bytes);
            }
        }
    }
    say(&format!(
        "{} graphs with joins exported on the device; worst PNG {worst} of 65535 from the CPU's",
        joined_graphs().len()
    ));
    assert!(changed > 0, "no graph with joins changed a pixel");
}

// ---- the Colour Uniform node (docs/32, CU-04) -----------------------------------------------------------------

/// Graphs with a live Colour Uniform node: every graph of `graphs()` that has one, and six more that force the kind with each
/// of the first mask kinds.
fn uniform_graphs() -> Vec<Graph> {
    let masks = generated_masks();
    let mut all: Vec<Graph> = graphs().into_iter().filter(Graph::uniform_live).collect();
    all.extend((0..6).map(|i| {
        generate(
            FIXED_SEEDS[i % FIXED_SEEDS.len()],
            20 + i,
            NodeKind::ColorUniform,
            masks[i % masks.len()],
        )
    }));
    all.retain(|g| g.compare);
    all
}

/// The kind is drawn by the generator and appears in a run (the gate of `docs/32` section 8: "the run asserts that the kind appeared").
#[test]
fn the_generator_draws_the_colour_uniform_node_and_some_graph_has_it_live() {
    assert!(generated_nodes().contains(&NodeKind::ColorUniform));
    let all = graphs();
    assert!(
        all.iter()
            .any(|g| g.nodes.contains(&NodeKind::ColorUniform)),
        "the kind never appeared"
    );
    assert!(
        uniform_graphs().iter().any(Graph::uniform_live),
        "no live Colour Uniform node in a run"
    );
}

/// A Colour Uniform node's trim and grade (`docs/32` section 12a) are drawn and appear in a run, on a node that does something,
/// and `judge_export`'s nudges reach them through the schema: no line of the rule names the kind or the fields.
#[test]
fn a_graded_colour_uniform_node_appears_and_its_settings_are_nudged() {
    let graded: Vec<Graph> = graphs()
        .into_iter()
        .filter(|g| {
            g.json["nodes"]["items"].as_array().is_some_and(|items| {
                items.iter().any(|n| {
                    n["kind"] == "color_uniform"
                        && n.get("grade").is_some()
                        && n.get("trim").is_some()
                })
            })
        })
        .collect();
    assert!(!graded.is_empty(), "no graded Colour Uniform node in a run");
    let g = &graded[0];
    let at = g.json["nodes"]["items"]
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .position(|n| n["kind"] == "color_uniform" && n.get("grade").is_some())
        })
        .expect("a graded node");
    let leaves = nudge_leaves(&g.json);
    for field in ["trim/exposure_ev", "grade/brightness"] {
        let want = format!("/nodes/items/{at}/{field}");
        assert!(
            leaves.iter().any(|l| l.at == want),
            "{want} is not nudged: {:?}",
            leaves.iter().map(|l| &l.at).collect::<Vec<_>>()
        );
    }
}

/// Random graphs with a Colour Uniform node render on the CPU, the same bits whatever the tile size (97, 160, 256 px) and the
/// thread count (1, 3, 4), the same bits again on a repeat, and the node changes a pixel in some graph.
#[test]
fn colour_uniform_graphs_are_the_same_bits_whatever_the_tiles_and_the_threads() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let plain = d.source();
    let plain_bits = full_bits(&cpu_renderer(256, 1), &plain, 1);
    let configs = [(97usize, 1usize), (160, 3), (256, 4)];
    let renderers: Vec<Renderer> = configs.iter().map(|&(t, n)| cpu_renderer(t, n)).collect();
    let mut changed = 0;
    for g in uniform_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        say(&format!(
            "{name}: nodes {:?}, masks {:?}",
            g.nodes.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masks.iter().map(|k| k.name()).collect::<Vec<_>>(),
        ));
        let src = d.source().with_settings(g.settings.clone());
        let first = full_bits(&renderers[0], &src, configs[0].1);
        for (r, &(tile, threads)) in renderers.iter().zip(&configs).skip(1) {
            assert!(
                full_bits(r, &src, threads) == first,
                "{name}: tile {tile} on {threads} threads is other bits; settings: {}",
                g.json
            );
        }
        assert!(
            full_bits(&renderers[0], &src, configs[0].1) == first,
            "{name}: a repeat is other bits"
        );
        let st = view(false, false, None);
        assert!(
            frame(&renderers[0], &src, &st).rgba == frame(&renderers[2], &src, &st).rgba,
            "{name}: the Fit frame depends on the threads"
        );
        changed += usize::from(first != plain_bits);
    }
    assert!(
        changed > 0,
        "no graph with a Colour Uniform node changed a pixel"
    );
}

/// Since CU-05 the device runs a plan with a live Colour Uniform node: it is not refused, the frame has no CPU reason, and its
/// pixels are within the CPU renderer's bounds (a code, as every graph of the sweeps), with the node's pull in them.
#[test]
fn colour_uniform_graphs_run_on_the_device_and_match_the_cpu() {
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
    let device = renderer(Some(gpu), 256);
    let doc = json!({ "nodes": { "items": [ { "id": 1 },
        { "id": 2, "kind": "color_uniform", "pos": [2.0, 0.0],
          "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": 0.0 },
          "compress": { "hue": 80, "saturation": 80, "lightness": 40 } } ],
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let src = d.source().with_settings(settings);
    let st = view(false, false, None);
    let got = frame(&device, &src, &st);
    assert!(
        !got.report.device_refused && got.report.cpu_reason.is_none(),
        "device {:?}, refused {}, reason {:?}",
        got.report.device,
        got.report.device_refused,
        got.report.cpu_reason
    );
    // The frame is the CPU's colour with the node in it: within a code of the CPU renderer's (the raw stages of the device
    // renderer differ by rounding), and not the picture without the node.
    let want = frame(&cpu, &src, &st);
    let (worst, over) = frame_diff(&got, &want);
    assert!(
        worst <= 2 && over < got.rgba.len() / 100,
        "{worst} codes, {over} over 1"
    );
    assert!(
        frame_diff(&got, &frame(&cpu, &d.source(), &st)).0 > 2,
        "the frame has no node in it"
    );
}

/// Graphs with a live Blur node: every graph of `graphs()` that has one, and sixteen more that force the node, four with each of the
/// types, with the mask kinds in turn (those the CPU makes the same twice, and a node that is live).
fn blur_graphs() -> Vec<Graph> {
    let masks = generated_masks();
    let mut all: Vec<Graph> = graphs().into_iter().filter(Graph::blur_live).collect();
    all.extend((0..16).map(|i| {
        generate(
            FIXED_SEEDS[i % FIXED_SEEDS.len()],
            BLUR_FORCED_FROM + i,
            NodeKind::Blur,
            masks[i % masks.len()],
        )
    }));
    all.retain(|g| g.compare && g.blur_live());
    all
}

/// The kind is drawn by the generator and appears in a run (the gate of `docs/33` section 8: "the run asserts that the kind
/// appeared"), live and with each type.
#[test]
fn the_generator_draws_the_blur_node_with_each_type_and_some_graph_has_it_live() {
    assert!(generated_nodes().contains(&NodeKind::Blur));
    assert!(
        graphs().iter().any(|g| g.nodes.contains(&NodeKind::Blur)),
        "the kind never appeared"
    );
    let all = blur_graphs();
    assert!(
        all.iter().any(Graph::blur_live),
        "no live Blur node in a run"
    );
    for ty in ["soft", "lens", "motion", "zoom"] {
        assert!(
            all.iter().any(|g| g.blur_live()
                && g.json["nodes"]["items"]
                    .as_array()
                    .is_some_and(|items| items.iter().any(|n| n["kind"] == "blur"
                        && n["type"] == ty
                        && n["bypass"] != json!(true)))),
            "no live Blur node of type {ty}"
        );
    }
}

/// Random graphs with a Blur node render on the CPU, the same bits whatever the tile size (97, 160, 256 px) and the thread count
/// (1, 3, 4), the same bits again on a repeat, and the node changes a pixel in some graph.
#[test]
fn blur_graphs_are_the_same_bits_whatever_the_tiles_and_the_threads() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let configs = [(97usize, 1usize), (160, 3), (256, 4)];
    let renderers: Vec<Renderer> = configs.iter().map(|&(t, n)| cpu_renderer(t, n)).collect();
    let mut changed = 0;
    for g in blur_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        say(&format!(
            "{name}: nodes {:?}, masks {:?}, live {}",
            g.nodes.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masks.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.blur_live(),
        ));
        let src = d.source().with_settings(g.settings.clone());
        let first = full_bits(&renderers[0], &src, configs[0].1);
        for (r, &(tile, threads)) in renderers.iter().zip(&configs).skip(1) {
            assert!(
                full_bits(r, &src, threads) == first,
                "{name}: tile {tile} on {threads} threads is other bits; settings: {}",
                g.json
            );
        }
        assert!(
            full_bits(&renderers[0], &src, configs[0].1) == first,
            "{name}: a repeat is other bits"
        );
        let st = view(false, false, None);
        assert!(
            frame(&renderers[0], &src, &st).rgba == frame(&renderers[2], &src, &st).rgba,
            "{name}: the Fit frame depends on the threads"
        );
        // The same graph with every Blur node bypassed is another picture where the node is live.
        if g.blur_live() {
            let mut doc = g.json.clone();
            for item in doc["nodes"]["items"].as_array_mut().expect("items") {
                if item["kind"] == "blur" {
                    item["bypass"] = json!(true);
                }
            }
            let off = serde_json::from_value::<DevelopSettings>(doc)
                .expect("a document")
                .validated()
                .expect("a valid graph")
                .0;
            let off_bits = full_bits(&renderers[0], &d.source().with_settings(off), 1);
            changed += usize::from(first != off_bits);
        }
    }
    assert!(changed > 0, "no graph with a Blur node changed a pixel");
}

/// A plan with a live Blur node of a type with no device kernel yet is refused by the device and made on the CPU, with the reason in
/// the frame's report; one whose live nodes are all of types that run on the device (Soft, BL-06; Lens, BL-07; Zoom, BL-09) is not refused, has no CPU reason
/// and is within the CPU renderer's bounds. The PNG export is, either way, within one 10-bit code of the CPU's (the unstable-pixel
/// rule of `judge_export` applies as for every graph).
#[test]
fn blur_graphs_are_refused_by_the_device_or_run_on_it_and_match_the_cpu() {
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
    let live: Vec<Graph> = blur_graphs().into_iter().filter(Graph::blur_live).collect();
    assert!(!live.is_empty());
    let mut problems: Vec<String> = Vec::new();
    for (n, g) in live.iter().enumerate() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        let src = d.source().with_settings(g.settings.clone());
        let device = renderer(Some(Arc::clone(&gpu)), 256);
        let st = view(false, false, None);
        let _ = frame(&device, &src, &st);
        settle_depth(&device, &src, &[("fit".to_owned(), st.clone())]);
        if has_depth_row(&src.settings) {
            settle_depth(&cpu, &src, &[("fit".to_owned(), st.clone())]);
        }
        let got = device
            .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame");
        if g.blur_refused() {
            if !got.report.device_refused
                || !got
                    .report
                    .cpu_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("Blur node"))
            {
                problems.push(format!(
                    "{name}: not refused with its reason: refused {}, reason {:?}",
                    got.report.device_refused, got.report.cpu_reason
                ));
            }
        } else if got.report.device_refused || got.report.cpu_reason.is_some() {
            problems.push(format!(
                "{name}: the device runs its Blur nodes but refused: refused {}, reason {:?}",
                got.report.device_refused, got.report.cpu_reason
            ));
        }
        let want = frame(&cpu, &src, &st);
        let (worst, over) = frame_diff(&got, &want);
        if worst > 2 || over >= got.rgba.len() / 100 {
            problems.push(format!(
                "{name}: against the CPU {worst} codes, {over} over 1"
            ));
        }
        // The export of a few: the CPU's colour pass, within the bound of the CPU's own.
        if n < 4 {
            let home = tempfile::tempdir().expect("dir");
            let (_, _, dev) = export(
                &d,
                &g.settings,
                renderer(Some(Arc::clone(&gpu)), 256),
                home.path(),
                FormatKind::Png,
            );
            let (_, _, cpu_bytes) = export(
                &d,
                &g.settings,
                renderer(None, 256),
                home.path(),
                FormatKind::Png,
            );
            let (a, b) = (decode_png16_bytes(&dev), decode_png16_bytes(&cpu_bytes));
            let max = a
                .iter()
                .zip(&b)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            if max > TEN_BIT_CODE {
                let judged = judge_export(&d, &g.settings, &a, &b, home.path(), &name);
                if let Some(f) = judged.failure {
                    problems.push(format!(
                        "{f} (the PNG export differs from the CPU's by {max} of 65535)"
                    ));
                }
            }
        }
        if !problems.is_empty() {
            say(&format!("{name} FAILED; settings to replay: {}", g.json));
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

/// Seed 23, graph 0 of `generate` (the Mac CI run of `colour-uniform`, `no_generated_graph_is_refused_by_the_device`: "seed 23 graph 0:
/// the PNG export differs from the CPU's by 134 of 65535", the bound being one 10-bit code, 64). Since CU-03 the forced kind of that
/// slot is the Colour Uniform node, which `main` never drew there: Main and a Colour Uniform node (target hue 188, saturation 18,
/// +0.38 EV; compress 71, 67, 74) with two mask rows, a Colour range and a Depth row that subtracts. (`generate_joined` also makes a
/// "seed 23 graph 0", a Splitter and Layer graph with no Colour Uniform node; the log line that names the failure is the one
/// printed with "show mask", `FAILED; settings to replay` gives the settings.)
/// `seed_23_graph_0_export_matches_the_cpu_and_prints_its_stages` is pinned to the same bound, and prints the export difference of
/// the graph, of Main alone, of the graph with no, one or the other mask row, and of each axis of the node alone, so that the stage where the difference starts can be read on any
/// device. Run it on Metal with the Mac GPU workflow's diagnostic dispatch (`tests` = `-p ym-core --test any_graph`, `filter` = `seed_23`).
const SEED_23_GRAPH_0: &str = r#"{"nodes":{"items":[{"id":1},{"compress":{"hue":71,"lightness":74,"saturation":67},"id":2,"kind":"color_uniform","masks":{"invert":false,"items":[{"center":[0.05,0.0],"kind":"color_range","mode":"add","softness":0.5,"width":0.2},{"far":0.794994994994995,"feather":0.29009009009009007,"kind":"depth","mode":"subtract","refine":0.26226226226226224}]},"pos":[2,0.0],"target":{"hue_deg":188,"lightness_ev":0.38,"saturation":18}}],"links":[{"from":{"node":"input","port":"picture"},"to":{"node":1,"port":"picture"}},{"from":{"node":1,"port":"picture"},"to":{"node":2,"port":"picture"}},{"from":{"node":2,"port":"picture"},"to":{"node":"output","port":"picture"}}]}}"#;

/// One value of a PNG export: the difference, its index, the CPU's value and the device's.
type Worst = (u32, usize, u32, u32);
/// A change to a Colour Uniform node.
type Variant = Box<dyn Fn(&mut Value)>;

/// The PNG exports of a settings document on the device and on the CPU: the largest difference of a value, the count of values
/// over the 10-bit bound, and the six worst as (difference, value index, CPU's value, device's value). With `judge`, the verdict of
/// [`judge_export`] on the two (its failure, `None` when the bound holds or the values over it are the CPU's own unstable ones);
/// without it the numbers are only reported.
fn export_difference(
    d: &Developed,
    gpu: &Arc<Gpu>,
    doc: &str,
    home: &Path,
    judge: bool,
) -> (u32, usize, Vec<Worst>, Option<String>) {
    let settings = serde_json::from_str::<DevelopSettings>(doc)
        .expect("settings")
        .validated()
        .expect("a valid graph")
        .0;
    let (dev_on, notes, dev) = export(
        d,
        &settings,
        renderer(Some(Arc::clone(gpu)), 256),
        home,
        FormatKind::Png,
    );
    assert!(
        dev_on == ComputeDevice::Gpu && !notes.iter().any(|n| n.contains("graphics card")),
        "the export is {dev_on:?}, notes {notes:?}"
    );
    let (_, _, cpu) = export(d, &settings, renderer(None, 256), home, FormatKind::Png);
    let (dev, cpu) = (decode_png16_bytes(&dev), decode_png16_bytes(&cpu));
    let failure = if judge {
        judge_export(d, &settings, &dev, &cpu, home, "the export").failure
    } else {
        None
    };
    let mut worst: Vec<Worst> = dev
        .iter()
        .zip(&cpu)
        .enumerate()
        .map(|(i, (a, b))| (a.abs_diff(*b), i, *b, *a))
        .collect();
    let over = worst.iter().filter(|w| w.0 > TEN_BIT_CODE).count();
    worst.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    worst.truncate(6);
    (worst.first().map_or(0, |w| w.0), over, worst, failure)
}

#[test]
fn seed_23_graph_0_export_matches_the_cpu_and_prints_its_stages() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let home = tempfile::tempdir().expect("dir");
    let full: Value = serde_json::from_str(SEED_23_GRAPH_0).expect("json");
    // The graph as it is, then the same graph with its node bypassed (Main alone), with no mask rows, and with one row each.
    let variants: [(&str, Variant); 8] = [
        (
            "Main alone (node bypassed)",
            Box::new(|n| n["bypass"] = json!(true)),
        ),
        (
            "no mask rows",
            Box::new(|n| n["masks"] = json!({ "invert": false, "items": [] })),
        ),
        (
            "Colour range row only",
            Box::new(|n| {
                if let Some(rows) = n["masks"]["items"].as_array_mut() {
                    rows.truncate(1);
                }
            }),
        ),
        (
            "Depth row only",
            Box::new(|n| {
                if let Some(rows) = n["masks"]["items"].as_array_mut() {
                    rows.remove(0);
                }
            }),
        ),
        (
            "no mask rows, Hue only",
            Box::new(|n| {
                n["masks"] = json!({ "invert": false, "items": [] });
                n["compress"] = json!({ "hue": 71, "saturation": 0, "lightness": 0 });
            }),
        ),
        (
            "no mask rows, Saturation only",
            Box::new(|n| {
                n["masks"] = json!({ "invert": false, "items": [] });
                n["compress"] = json!({ "hue": 0, "saturation": 67, "lightness": 0 });
            }),
        ),
        (
            "no mask rows, Lightness only",
            Box::new(|n| {
                n["masks"] = json!({ "invert": false, "items": [] });
                n["compress"] = json!({ "hue": 0, "saturation": 0, "lightness": 74 });
            }),
        ),
        ("the graph", Box::new(|_| {})),
    ];
    let mut failure: Option<String> = None;
    for (what, change) in &variants {
        let mut doc = full.clone();
        if let Some(items) = doc["nodes"]["items"].as_array_mut() {
            for n in items.iter_mut().filter(|n| n["kind"] == "color_uniform") {
                change(n);
            }
        }
        let last = *what == "the graph";
        let (max, over, worst, verdict) =
            export_difference(&d, &gpu, &doc.to_string(), home.path(), last);
        say(&format!(
            "seed 23 graph 0, {what}: PNG export {max} of 65535 from the CPU's, {over} values over {TEN_BIT_CODE}; worst (difference, value index, CPU, device) {worst:?}"
        ));
        if last {
            failure = verdict;
        }
    }
    // The same graph at view scale, Fit and 100 %, as `no_generated_graph_is_refused_by_the_device` compares it (codes of 255).
    {
        let settings = serde_json::from_value::<DevelopSettings>(full.clone())
            .expect("settings")
            .validated()
            .expect("a valid graph")
            .0;
        let src = d.source().with_settings(settings);
        let (device, cpu) = (renderer(Some(Arc::clone(&gpu)), 256), renderer(None, 256));
        let states: Vec<(String, ViewportState)> = [false, true]
            .into_iter()
            .map(|z| (format!("100 % {z}"), view(z, false, None)))
            .collect();
        let _ = frame(&device, &src, &states[0].1);
        settle_depth(&device, &src, &states);
        settle_depth(&cpu, &src, &states);
        for (what, st) in &states {
            let (max, over) = frame_diff(&frame(&device, &src, st), &frame(&cpu, &src, st));
            say(&format!(
                "seed 23 graph 0, view {what}: against the CPU {max} codes, {over} values over 1"
            ));
        }
    }
    // The conditioning of the graph: the CPU's own export of the graph against the CPU's export with Main's exposure moved by 1e-5 EV
    // (7e-6 relative on every scene value, the size of an `f32` rounding). A graph whose picture moves by more than the bound for
    // that is ill-conditioned for any `f32` pipeline (as seed 539 was), and a device difference of that size is not a device error.
    {
        let cpu_png = |doc: &Value| {
            let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
                .expect("settings")
                .validated()
                .expect("a valid graph")
                .0;
            decode_png16_bytes(
                &export(
                    &d,
                    &settings,
                    renderer(None, 256),
                    home.path(),
                    FormatKind::Png,
                )
                .2,
            )
        };
        let mut nudged = full.clone();
        nudged["nodes"]["items"][0]["trim"] = json!({ "exposure_ev": 1e-5 });
        let (a, b) = (cpu_png(&full), cpu_png(&nudged));
        let moved = a
            .iter()
            .zip(&b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        say(&format!(
            "seed 23 graph 0, conditioning: the CPU's export moves by {moved} of 65535 for 1e-5 EV on Main"
        ));
    }
    // The bound of 64 holds on every value but the CPU's own unstable ones (the owner, 2026-11-18, "Skip unstable pixels"; `judge_export`).
    assert!(
        failure.is_none(),
        "seed 23 graph 0: the PNG export fails the bound: {}",
        failure.unwrap_or_default()
    );
}

/// Why seed 23 graph 0 differs on Metal (two pixels on a one-pixel dark line of the 640 x 420 export), on the CPU alone. On the pictures
/// of `PIPELINE_VERSION` 0.13.0 and before they were (277, 314) and (278, 314), value indices 603711 to 603716: the CPU's values 191 and
/// 5379 of 65535 between neighbours of 22,000 and 40,000. RC-13 (A6 `0.9` and A7 `3.5` of the demosaic) changed the pictures, and the
/// line is where it was: the pixels are now (276, 314) and (277, 314), value indices 603708 to 603713, the CPU's values 1573 and 373
/// (the neighbours 22,761 and 40,253). Their Depth plane codes on the old pictures (42269 and 42271) did not change when the far limit
/// moved by 5e-6, and their exports did; so what moves them is a one-code change of the plane at pixels around them, which the
/// device's `f32` and the CPU's `f64` give differently at about 0.35 % of the pixels (`a_depth_row_plane_on_the_device_is_within_two_codes_of_the_cpu_through_every_step`
/// counts them in the plane test). This test makes the same one-code changes on the CPU alone: it moves the Depth row's far limit by
/// 5e-6 (0.3 of a code of the plane, which changes the codes of 13,600 pixels by one) and exports twice.
///
/// **The direction matters on the new pictures.** On the old ones +5e-6 moved the line by 142. On the new ones +5e-6 moves nothing there
/// (the largest change is 9, at a pixel elsewhere; the step of the plane at the line is not crossed), and -5e-6 crosses it: 70 at the
/// line's first pixel, 6 values over 64 (value indices 603708 to 603713), the same from -1e-6 to -20e-6 (69 to 71) and 168 at -30e-6. So
/// the pinned case is the far limit at -5e-6, the same pixels' instability seen from the other side of one code step. Measured with
/// the Output's sharpening at 0, 50, 100 and 200 on the new pictures, the first pixel is 15192, 12786, 1573 and 0 (nudged: 15192,
/// 12786, 1642 and 0).
///
/// * With the Output's sharpening at its default (100), the export moves by 70 of 65535 at those two pixels, more than the bound of the
///   device against the CPU (64): the **graph** is ill-conditioned for a one-code change of a mask next to the line, which no `f32`
///   pipeline can promise to leave unchanged. The sharpened luma of the first pixel at 100 is a few hundred codes above the 0 below
///   which a pixel is black (`docs/algorithms/04` section 3.7), where the hue-preserving ratio `L' / L` is steep (at 200 it is below
///   0 and the pixel is black, the same in both). (The cause inside the sharpener's halo limiter was not narrowed further.)
/// * With the sharpening at 0 the same change moves no value by more than 1 (7 at 50): the Depth path and the Colour Uniform node are
///   well conditioned (a code of the plane is a code of 65535 of the mask, which blends the node's change linearly), so nothing in them
///   is a decision that rounding can flip.
///
/// The test asserts both numbers on the CPU (no device is needed) and prints the same for seed 539's two Depth rows.
#[test]
fn a_one_code_change_of_a_depth_plane_moves_seed_23_only_through_output_sharpening() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let scene = Scene::new(
        640,
        420,
        light("sony-ilce-7rm4", 4800.0),
        Layout::Patchwork,
        0.5,
    )
    .noise(2e-4, 2e-6)
    .seed(5);
    let d = Developed::new(&scene, MODEL);
    let home = tempfile::tempdir().expect("dir");
    let cpu_png = |doc: &Value| {
        let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
            .expect("settings")
            .validated()
            .expect("a valid graph")
            .0;
        decode_png16_bytes(
            &export(
                &d,
                &settings,
                renderer(None, 256),
                home.path(),
                FormatKind::Png,
            )
            .2,
        )
    };
    // The largest change of a value between two exports, with its index.
    let moved = |a: &[u32], b: &[u32]| {
        a.iter()
            .zip(b)
            .enumerate()
            .map(|(i, (x, y))| (x.abs_diff(*y), i))
            .max()
            .unwrap_or((0, 0))
    };
    let with_sharpening = |doc: &Value, pct: u32| {
        let mut doc = doc.clone();
        if let Some(items) = doc["nodes"]["items"].as_array_mut() {
            items.push(json!({ "id": "output", "output_sharpening": pct }));
        }
        doc
    };
    let full: Value = serde_json::from_str(SEED_23_GRAPH_0).expect("json");
    let mut nudged = full.clone();
    nudged["nodes"]["items"][1]["masks"]["items"][1]["far"] = json!(0.794994994994995 - 5e-6);
    let (sharp, sharp_at) = moved(&cpu_png(&full), &cpu_png(&nudged));
    let (flat, flat_at) = moved(
        &cpu_png(&with_sharpening(&full, 0)),
        &cpu_png(&with_sharpening(&nudged, 0)),
    );
    say(&format!(
        "seed 23 graph 0, the CPU alone, the Depth row's far moved by -5e-6: the export moves by {sharp} of 65535 (value {sharp_at}) with the Output's sharpening at 100, by {flat} (value {flat_at}) with it at 0"
    ));
    assert!(
        flat <= 2,
        "with no output sharpening a one-code change of the plane moves the export by {flat} of 65535"
    );
    assert!(
        sharp > TEN_BIT_CODE,
        "the graph is no longer ill-conditioned for a one-code change of the plane ({sharp} of 65535): the pinned case needs a new look"
    );
    // On the dark line of the pixels (276, 314) and (277, 314), as before: the same instability, not another one.
    assert!(
        (603708..=603713).contains(&sharp_at),
        "the unstable value moved to {sharp_at}: look again before pinning it"
    );
    // Seed 539 (two Depth rows, two Film nodes in series): the same change of each row's far limit, reported.
    let doc: Value = serde_json::from_str(SEED_539_GRAPH_0).expect("json");
    let base = cpu_png(&doc);
    for (row, far) in [(0usize, 0.23923923923923926), (1, 0.5027027027027027)] {
        let mut nudged = doc.clone();
        nudged["nodes"]["items"][1]["masks"]["items"][row]["far"] = json!(far + 5e-6);
        let (m, at) = moved(&base, &cpu_png(&nudged));
        say(&format!(
            "seed 539 graph 0, the CPU alone, Depth row {row}'s far moved by 5e-6: the export moves by {m} of 65535 (value {at})"
        ));
    }
}

/// The rule of `judge_export` on the CPU alone, with a CPU render standing for the device: seed 23 graph 0 with the Depth row's far
/// limit moved by -5e-6 is a "device" that differs from the CPU by 69 or 70 at the two pixels of the dark line (on the pictures before
/// RC-13: moved by +5e-6, 142 at the two pixels the Mac run failed on), and those are the
/// CPU's own unstable values, so the rule passes it and prints them. The same "device" with one stable value moved by 500 fails
/// (the bound is enforced there); with an unstable value moved far outside the CPU's own range it fails (safeguard c); with nothing to
/// nudge it fails at a change of 100 at one value (safeguard a).
#[test]
fn the_unstable_pixel_rule_exempts_only_the_cpus_own_unstable_values() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let scene = Scene::new(
        640,
        420,
        light("sony-ilce-7rm4", 4800.0),
        Layout::Patchwork,
        0.5,
    )
    .noise(2e-4, 2e-6)
    .seed(5);
    let d = Developed::new(&scene, MODEL);
    let home = tempfile::tempdir().expect("dir");
    let png = |doc: &Value| {
        let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
            .expect("settings")
            .validated()
            .expect("a valid graph")
            .0;
        (
            decode_png16_bytes(
                &export(
                    &d,
                    &settings,
                    renderer(None, 256),
                    home.path(),
                    FormatKind::Png,
                )
                .2,
            ),
            settings,
        )
    };
    let full: Value = serde_json::from_str(SEED_23_GRAPH_0).expect("json");
    let (cpu, settings) = png(&full);
    let mut nudged = full.clone();
    nudged["nodes"]["items"][1]["masks"]["items"][1]["far"] = json!(0.794994994994995 - 5e-6);
    let (dev, _) = png(&nudged);
    // The second of the two pixels of the dark line, (277, 314), red (the first, 603708, is (276, 314)).
    let at = 603711;
    assert!(
        dev[at].abs_diff(cpu[at]) > TEN_BIT_CODE,
        "the stand-in differs there"
    );
    let verdict = judge_export(&d, &settings, &dev, &cpu, home.path(), "stand-in");
    assert!(
        verdict.failure.is_none(),
        "the CPU's own unstable values pass: {:?}",
        verdict.failure
    );
    // A stable value moved by 500 (value 0 is a bright corner no mask setting moves) and an unstable value moved far outside the CPU's
    // own range both fail, each for its own reason.
    let mut bad = dev.clone();
    bad[0] += 500;
    bad[at] += 3000;
    let verdict = judge_export(&d, &settings, &bad, &cpu, home.path(), "stable and range");
    let text = verdict.failure.unwrap_or_default();
    assert!(
        text.contains("stable values differ") && text.contains("outside the CPU's own range"),
        "{text}"
    );
    // Nothing to nudge: nothing is exempt, whatever the values.
    let mut bad = dev.clone();
    bad[at] += 100;
    let verdict = judge_export_with(&d, &full, &[], &bad, &cpu, home.path(), "nothing nudged");
    assert!(
        verdict
            .failure
            .as_deref()
            .is_some_and(|f| f.contains("nothing could be nudged")),
        "{:?}",
        verdict.failure
    );
}

/// The other "seed 23 graph 0" of the log, from `generate_joined`: Main, a Splitter, a bypassed Parallel, a Layer (difference, opacity 0),
/// a bypassed corrector and a Layer (subtract, opacity 0) that is the Output. No Film node and no Colour Uniform node. It is not the
/// graph the Mac run failed on (that is `SEED_23_GRAPH_0`), and is pinned so that the two cannot be taken for each other again.
const SEED_23_GRAPH_0_JOINED: &str = r#"{"nodes":{"items":[{"id":1},{"id":2,"kind":"splitter","pos":[2,0.0]},{"bypass":true,"id":3,"kind":"parallel","pos":[3,0.0]},{"id":4,"kind":"layer","mode":"difference","opacity":0.0,"pos":[4,0.0]},{"bypass":true,"id":5,"kind":"corrector","pos":[5,0.0],"trim":{"exposure_ev":-0.8}},{"id":6,"kind":"layer","mode":"subtract","opacity":0.0,"pos":[6,0.0]}],"links":[{"from":{"node":"input","port":"picture"},"to":{"node":1,"port":"picture"}},{"from":{"node":"input","port":"picture"},"to":{"node":2,"port":"picture"}},{"from":{"node":"input","port":"picture"},"to":{"node":3,"port":"picture.1"}},{"from":{"node":1,"port":"picture"},"to":{"node":3,"port":"picture.2"}},{"from":{"node":1,"port":"picture"},"to":{"node":3,"port":"picture.3"}},{"from":{"node":2,"port":"green"},"to":{"node":4,"port":"picture.1"}},{"from":{"node":1,"port":"picture"},"to":{"node":4,"port":"picture.2"}},{"from":{"node":4,"port":"picture"},"to":{"node":5,"port":"picture"}},{"from":{"node":2,"port":"green"},"to":{"node":6,"port":"picture.1"}},{"from":{"node":"input","port":"picture"},"to":{"node":6,"port":"picture.2"}},{"from":{"node":1,"port":"picture"},"to":{"node":6,"port":"picture.3"}},{"from":{"node":6,"port":"picture"},"to":{"node":"output","port":"picture"}}]}}"#;

#[test]
fn seed_23_graph_0_joined_export_matches_the_cpu() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let home = tempfile::tempdir().expect("dir");
    let (max, over, worst, failure) =
        export_difference(&d, &gpu, SEED_23_GRAPH_0_JOINED, home.path(), true);
    say(&format!(
        "seed 23 graph 0 (joined): PNG export {max} of 65535 from the CPU's, {over} values over {TEN_BIT_CODE}; worst (difference, value index, CPU, device) {worst:?}"
    ));
    assert!(
        failure.is_none(),
        "seed 23 graph 0 (joined): the PNG export differs from the CPU's by {max} of 65535 (bound {TEN_BIT_CODE}): {}",
        failure.unwrap_or_default()
    );
}

/// Plan-fix: the owner's sequence of 0.2.4-beta.3 on the device. A Colour Uniform node with a target and a Colour range row, a frame, the
/// node's Show mask view, then the node is deleted with the page's patch (`remove` and the links), a frame, and an edit of Main, a frame.
/// With the view of the deleted node kept (the page did not clear it), the frame must still be made (the view of a node the photo no longer
/// has is the picture's own view), and the edit of Main must change it; with the view cleared the same holds.
#[test]
fn a_deleted_colour_uniform_node_leaves_the_device_frames_working() {
    use ym_core::settings::SettingsPatch;
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let patch = |v: Value| -> SettingsPatch { serde_json::from_value(v).expect("patch") };
    let link = |from: Value, to: Value| json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } });
    for keep_view in [false, true] {
        let device = renderer(Some(Arc::clone(&gpu)), 256);
        let mut settings = d.source().settings.clone();
        let build = json!({
            "nodes": { "2": { "kind": "color_uniform", "name": "Uniform", "pos": [1.0, 0.0],
                "target": { "hue_deg": 30.0, "saturation": 30.0, "lightness_ev": 0.0 },
                "compress": { "hue": 50, "saturation": 50, "lightness": 0 },
                "masks": { "invert": false, "items": [ { "kind": "color_range", "mode": "add", "invert": false,
                    "enabled": true, "opacity": 1, "center": [0.0, 0.0], "width": 0.5, "softness": 0.5 } ] } } },
            "links": [link(json!("input"), json!(1)), link(json!(1), json!(2)), link(json!(2), json!("output"))],
        });
        settings
            .apply_patch(&patch(build), None)
            .expect("add the node");
        let shown = Some((2, MaskStyle::Overlay, Some(0)));
        let src = d.source().with_settings(settings.clone());
        let live = frame(&device, &src, &view(false, false, shown));
        assert!(
            !live.report.device_refused,
            "the node's frame is the device's"
        );
        settings
            .apply_patch(
                &patch(json!({ "remove": [2],
                    "links": [link(json!("input"), json!(1)), link(json!(1), json!("output"))] })),
                None,
            )
            .expect("delete the node");
        let st = if keep_view {
            view(false, false, shown)
        } else {
            view(false, false, None)
        };
        let deleted = frame(&device, &d.source().with_settings(settings.clone()), &st);
        settings
            .apply_patch(
                &patch(json!({ "nodes": { "1": { "trim": { "exposure_ev": 1.0 } } } })),
                None,
            )
            .expect("edit Main");
        let edited = frame(&device, &d.source().with_settings(settings), &st);
        assert!(
            frame_diff(&deleted, &edited).0 > 2,
            "keep_view {keep_view}: the edit of Main changes the frame"
        );
    }
}

/// Plan-fix (seed 296 graph 0 on Metal): the Show mask view of a Film-like node whose key is linked from another node. The chain's plan
/// stops at the node's mask (only that node runs, for its key), and the render once built the node's picture from a `Stop::Picture` plan
/// that runs the node its key comes from, which the chain has no parameters for: "the plan runs node 2, which the chain has no parameters for", on every frame of the view.
#[test]
fn the_mask_view_of_a_film_node_with_a_linked_key_is_made_on_the_device() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let device = renderer(Some(gpu), 256);
    let cpu = renderer(None, 256);
    let doc = json!({ "nodes": { "items": [ { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [2.0, 0.0], "trim": { "exposure_ev": 0.8 },
          "masks": { "invert": false, "items": [ { "kind": "radial", "mode": "add", "invert": false, "enabled": true, "opacity": 1, "center": [0.4, 0.5], "radii": [0.3, 0.2], "angle_deg": 0.0, "feather": 0.3 } ] } },
        { "id": 3, "kind": "film", "pos": [3.0, 0.0], "film": { "exposure_ev": 0.3 },
          "grain": { "amount": 100, "seed": 3 }, "halation": { "amount": 200, "size": 120 }, "strength": 85,
          "masks": { "invert": false, "items": [ { "kind": "radial", "mode": "add", "invert": false, "enabled": true,
              "opacity": 1, "center": [0.5, 0.5], "radii": [0.3, 0.2], "angle_deg": 0.0, "feather": 0.3 } ] } } ],
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 3, "port": "picture" } },
            { "from": { "node": 2, "port": "key" }, "to": { "node": 3, "port": "key" } },
            { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let src = d.source().with_settings(settings);
    let st = view(false, false, Some((3, MaskStyle::Overlay, Some(0))));
    let got = frame(&device, &src, &st);
    assert!(
        !got.report.device_refused && got.report.cpu_reason.is_none(),
        "device refused {}, reason {:?}",
        got.report.device_refused,
        got.report.cpu_reason
    );
    let (worst, over) = frame_diff(&got, &frame(&cpu, &src, &st));
    assert!(
        worst <= 2 && over < got.rgba.len() / 100,
        "{worst} codes, {over} over 1"
    );
}

const SEED_297_GRAPH_0: &str = r#"{"nodes":{"items":[{"id":1},{"id":2,"kind":"blur","lens":{"blades":6,"highlights":90,"radius":38,"rotation_deg":81},"masks":{"invert":false,"items":[{"center":[0.05,0.0],"kind":"color_range","mode":"add","softness":0.5,"width":0.2},{"far":0.5723723723723724,"feather":0.2948948948948949,"kind":"depth","mode":"add","refine":0.8378378378378378}]},"motion":{"angle_deg":66,"length":38},"pos":[2,0.0],"respect_mask_edges":true,"soft":{"radius":35},"type":"motion","zoom":{"amount":57,"center":[0.08,0.39]}},{"all":{"blemish":30,"dark_circles":80,"even_skin":80,"skin_contrast":30},"id":3,"kind":"face_refinement","pos":[3,0.0]},{"compress":{"hue":67,"lightness":96,"saturation":100},"id":4,"kind":"color_uniform","masks":{"invert":false,"items":[{"angle_deg":0.0,"center":[0.5,0.5],"feather":0.3,"kind":"radial","mode":"add","radii":[0.3,0.2]}]},"pos":[4,0.0],"target":{"hue_deg":43,"lightness_ev":-0.09000000000000002,"saturation":31}},{"compress":{"hue":57,"lightness":64,"saturation":84},"id":5,"kind":"color_uniform","pos":[5,0.0],"target":{"hue_deg":314,"lightness_ev":0.39,"saturation":28}}],"links":[{"from":{"node":"input","port":"picture"},"to":{"node":1,"port":"picture"}},{"from":{"node":1,"port":"picture"},"to":{"node":2,"port":"picture"}},{"from":{"node":2,"port":"picture"},"to":{"node":3,"port":"picture"}},{"from":{"node":"input","port":"picture"},"to":{"node":4,"port":"picture"}},{"from":{"node":4,"port":"picture"},"to":{"node":5,"port":"picture"}},{"from":{"node":5,"port":"picture"},"to":{"node":"output","port":"picture"}},{"from":{"node":2,"port":"key"},"to":{"node":4,"port":"key"}},{"from":{"node":3,"port":"key"},"to":{"node":5,"port":"key"}}]}}"#;

/// The rule of `judge_export` for a sensitivity that comes from no mask row (the owner, 2026-11-21: "extend the rule so there are no
/// exceptions for specific cases"). Seed 297, graph 0 of `generate` (CI run 44814457514 on Metal): Main, a Blur node and a Face
/// Refinement node that the picture does not use, and two Colour Uniform nodes in a row, the second with no mask row. Two pixels
/// (value indices 204639 to 204644, the red, green and blue of (213, 162) and of its neighbour) are hypersensitive to the second
/// node's settings: on the CPU alone one unit of its saturation compress moves them by about 2,000 codes, and the device differs
/// from the CPU there by 72 on Metal (52 on lavapipe), over the bound of 64. The rule that nudged only mask rows called those values
/// stable and failed the run. Every continuous setting is nudged now, all together with seeded signs, and the CPU's own export there
/// moves by more than 64.
///
/// A CPU render stands for the device. Moved by +72 at the six values, it passes (they are unstable and the "device" is in the CPU's
/// range); nudging only the settings of mask rows, as the old rule did, it fails; a value moved by 3,000 (outside the range) and a
/// stable value moved by 500 each fail for their own reasons.
#[test]
fn the_unstable_pixel_rule_sees_the_sensitivity_of_a_setting_that_is_not_a_mask_row() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let scene = Scene::new(
        640,
        420,
        light("sony-ilce-7rm4", 4800.0),
        Layout::Patchwork,
        0.5,
    )
    .noise(2e-4, 2e-6)
    .seed(5);
    let d = Developed::new(&scene, MODEL);
    let home = tempfile::tempdir().expect("dir");
    let settings = serde_json::from_str::<DevelopSettings>(SEED_297_GRAPH_0)
        .expect("settings")
        .validated()
        .expect("a valid graph")
        .0;
    let doc = serde_json::to_value(&settings).expect("json");
    let leaves = nudge_leaves(&doc);
    // The settings are found by their schema, not by a list: the Colour Uniform compress, the Input's, Main's (left out of the
    // document, at the schema's default) and the Blur node's are there; integers, enums, booleans, seeds and ids are not.
    for at in [
        "/nodes/items/5/compress/saturation",
        "/nodes/items/5/target/hue_deg",
        "/nodes/items/4/masks/items/0/radii/0",
        "/nodes/items/0/rolloff/highlights",
        "/nodes/items/1/trim/exposure_ev",
        "/nodes/items/2/soft/radius",
    ] {
        assert!(leaves.iter().any(|l| l.at == at), "{at} is not nudged");
    }
    for l in &leaves {
        let last = l.at.rsplit('/').next().unwrap_or_default();
        assert!(
            ![
                "id", "kind", "bypass", "seed", "type", "mode", "blades", "invert"
            ]
            .contains(&last),
            "{} is not a continuous setting",
            l.at
        );
    }
    let cpu = cpu_png_of(&d, &doc, home.path(), usize::MAX).unwrap_or_else(|| {
        decode_png16_bytes(
            &export(
                &d,
                &settings,
                renderer(None, 256),
                home.path(),
                FormatKind::Png,
            )
            .2,
        )
    });
    let hyper = 204_639..=204_644usize;
    let mut dev = cpu.clone();
    for i in hyper.clone() {
        dev[i] += 72;
    }
    let started = std::time::Instant::now();
    let verdict = judge_export(&d, &settings, &dev, &cpu, home.path(), "seed 297 graph 0");
    say(&format!(
        "seed 297 graph 0, the rule on the CPU alone: {:?} for {} settings",
        started.elapsed(),
        leaves.len()
    ));
    assert!(
        verdict.failure.is_none(),
        "the CPU's own unstable values pass: {:?}",
        verdict.failure
    );
    // The rule of 2026-11-18: each continuous number of each mask row alone, by plus and minus 1e-5. None moves these values by more
    // than the bound, so they were stable and the export failed.
    let mut widest = 0u32;
    for l in leaves.iter().filter(|l| l.at.contains("/masks/items/")) {
        for sign in [1.0, -1.0] {
            let mut nudged = doc.clone();
            set_number(&mut nudged, &l.at, l.value + sign * NUDGE);
            let moved = cpu_png_of(&d, &nudged, home.path(), cpu.len()).unwrap_or_default();
            for i in hyper.clone() {
                widest = widest.max(moved.get(i).map_or(0, |m| m.abs_diff(cpu[i])));
            }
        }
    }
    say(&format!(
        "seed 297 graph 0, mask rows one at a time by 1e-5: the six values move by {widest} at most"
    ));
    assert!(
        widest <= TEN_BIT_CODE,
        "a mask row moves them by {widest}: the case no longer shows what the old rule missed"
    );
    // A real device error: far outside the CPU's range at an unstable value, and 500 at a stable one (value 0 is a bright corner).
    let mut bad = dev.clone();
    bad[204_641] += 3000;
    bad[0] += 500;
    let text = judge_export(&d, &settings, &bad, &cpu, home.path(), "errors")
        .failure
        .unwrap_or_default();
    assert!(
        text.contains("outside the CPU's own range") && text.contains("stable values differ"),
        "{text}"
    );
}

/// The rule of `judge_view` on the CPU alone, with a CPU frame standing for the device (the owner, 2026-11-21). The graphs are the
/// pinned ones with the most ill-conditioned pixels (seed 539: two Film nodes in series, the case the Film-chain exception of
/// 2026-11-12 was made for; seed 297 and seed 23: Colour Uniform). The first whose CPU view moves by 3 codes or more at some value
/// under the nudges is the case: a "device" that differs from the CPU there by 3 codes passes (the value is unstable, and the device
/// is in the CPU's range); the same size of difference at a stable value fails (the bound of 2 codes is enforced there); an unstable
/// value moved far outside the CPU's range fails (safeguard c); with nothing nudged nothing is exempt (safeguard a).
#[test]
fn the_unstable_view_rule_exempts_only_the_cpus_own_unstable_values() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let scene = Scene::new(
        640,
        420,
        light("sony-ilce-7rm4", 4800.0),
        Layout::Patchwork,
        0.5,
    )
    .noise(2e-4, 2e-6)
    .seed(5);
    let d = Developed::new(&scene, MODEL);
    let cpu = renderer(None, 256);
    let mut found = false;
    for (what, json) in [
        ("seed 539 graph 0", SEED_539_GRAPH_0),
        ("seed 297 graph 0", SEED_297_GRAPH_0),
        ("seed 23 graph 0", SEED_23_GRAPH_0),
    ] {
        let settings = serde_json::from_str::<DevelopSettings>(json)
            .expect("settings")
            .validated()
            .expect("a valid graph")
            .0;
        let src = d.source().with_settings(settings);
        let doc = serde_json::to_value(&src.settings).expect("json");
        let leaves = nudge_leaves(&doc);
        let st = view(false, false, None);
        let _ = frame(&cpu, &src, &st);
        settle_depth(&cpu, &src, &[("fit".to_owned(), st.clone())]);
        let base = frame(&cpu, &src, &st);
        let started = std::time::Instant::now();
        let (lo, hi, renders) = view_ranges(&cpu, &src, &st, &doc, &leaves, &base);
        let moved = |i: usize| (hi[i] - base.rgba[i]).max(base.rgba[i] - lo[i]);
        let most = (0..base.rgba.len()).map(moved).max().unwrap_or(0);
        let unstable = (0..base.rgba.len()).filter(|&i| moved(i) > 1).count();
        say(&format!(
            "{what}: the CPU's view moves by {most} codes at most under the nudges, {unstable} unstable values, {renders} renders in {:?}",
            started.elapsed()
        ));
        let Some(at) = (0..base.rgba.len())
            .filter(|&i| moved(i) >= 3 && base.rgba[i] >= 8 && base.rgba[i] <= 247)
            .max_by_key(|&i| moved(i))
        else {
            continue;
        };
        found = true;
        let with = |i: usize, to: u8| {
            let mut dev = base.clone();
            dev.rgba[i] = to;
            dev
        };
        // An unstable value 3 codes from the CPU's, inside its range (the end of the range it moves to, or 3 on that side).
        let up = hi[at] - base.rgba[at] >= base.rgba[at] - lo[at];
        let inside = if up {
            base.rgba[at] + 3.min(hi[at] - base.rgba[at])
        } else {
            base.rgba[at] - 3.min(base.rgba[at] - lo[at])
        };
        assert!(
            base.rgba[at].abs_diff(inside) > 1,
            "{what}: a difference over 1"
        );
        let verdict = judge_view_with(&with(at, inside), &base, &lo, &hi, renders, what);
        assert!(
            verdict.failure.is_none(),
            "{what}: the CPU's own unstable value passes: {:?}",
            verdict.failure
        );
        // The same 3 codes at a stable value (the first with no movement at all and room to move).
        let stable = (0..base.rgba.len())
            .find(|&i| moved(i) == 0 && base.rgba[i] >= 8 && base.rgba[i] <= 247)
            .expect("a stable value");
        let text = judge_view_with(
            &with(stable, base.rgba[stable] + 3),
            &base,
            &lo,
            &hi,
            renders,
            what,
        )
        .failure
        .unwrap_or_default();
        assert!(text.contains("differ by 3 codes"), "{what}: {text}");
        // An unstable value far outside the CPU's range.
        let far = if up {
            hi[at].saturating_add(40)
        } else {
            lo[at].saturating_sub(40)
        };
        let text = judge_view_with(&with(at, far), &base, &lo, &hi, renders, what)
            .failure
            .unwrap_or_default();
        assert!(
            text.contains("outside the CPU's own range"),
            "{what}: {text}"
        );
        // Nothing nudged: nothing is exempt.
        let text = judge_view_with(&with(at, inside), &base, &lo, &hi, 0, what)
            .failure
            .unwrap_or_default();
        assert!(text.contains("nothing could be nudged"), "{what}: {text}");
        break;
    }
    assert!(
        found,
        "no pinned graph has a view value that moves by 3 codes"
    );
}

// ---- Reference Grade graphs (docs/34, RG-02): the node runs on the CPU ----------------------------------------

/// A Reference Grade graph: Main, perhaps a corrector, then a Reference Grade node with fitted-style values (a luma curve and a
/// blue curve a little off the identity, a saturation, two hue curves with 3 to 6 points, the six elements and the subject within
/// their caps, a Strength), sometimes a mask row and a Bypass. It has a generator of its own, so that **no other draw moves**
/// (`generated_nodes` does not hold the kind, `REFERENCE_NOT_DRAWN`). Hue points sit at distinct hues, 60 degrees apart or more.
fn generate_reference(seed: u64, index: usize) -> Graph {
    let mut rng = SplitMix64(
        seed.wrapping_mul(0xA24B_AED4_963E_E407)
            .wrapping_add((index as u64) << 16)
            .wrapping_add(0x5EF0),
    );
    let signed = |rng: &mut SplitMix64, span: u64| rng.below(2 * span + 1) as f64 - span as f64;
    let with_corrector = rng.below(2) == 0;
    let id = if with_corrector { 3 } else { 2 };
    let curve_points = |rng: &mut SplitMix64| {
        let d1 = signed(rng, 40) / 1000.0;
        let d2 = signed(rng, 40) / 1000.0;
        json!([
            [0, 0],
            [0.2136, 0.2136 + d1],
            [0.2784, 0.2784],
            [0.408, 0.408 + d2],
            [1, 1]
        ])
    };
    let hue_points = |rng: &mut SplitMix64, neutral: f64, span: u64, scale: f64| {
        let n = 3 + rng.below(4) as usize;
        let start = rng.below(40) as f64;
        let step = 360.0 / n as f64;
        (0..n)
            .map(|i| {
                let v = neutral + signed(rng, span) * scale;
                json!({ "hue_deg": start + step * i as f64, "value": v })
            })
            .collect::<Vec<Value>>()
    };
    let caps = [40, 35, 35, 30, 25, 15];
    let names = ["sky", "vegetation", "water", "ground", "built", "skin"];
    let mut elements = serde_json::Map::new();
    for (name, cap) in names.iter().zip(caps) {
        elements.insert(
            (*name).to_owned(),
            json!({ "exposure_ev": signed(&mut rng, 30) / 100.0, "hue_deg": signed(&mut rng, cap),
                    "chroma": 0.7 + rng.below(80) as f64 / 100.0 }),
        );
    }
    elements.insert(
        "subject".to_owned(),
        json!({ "exposure_ev": signed(&mut rng, 30) / 100.0 }),
    );
    let strength = [100, 100, 60, 30][rng.below(4) as usize];
    let mut node = json!({ "id": id, "kind": "reference_grade", "pos": [id, 0.0],
        "grade": { "saturation": { "kind": "manual", "value": signed(&mut rng, 25) },
                   "curve": { "luma": curve_points(&mut rng), "blue": curve_points(&mut rng) } },
        "hue_curves": { "hue_vs_hue": hue_points(&mut rng, 0.0, 15, 1.0),
                        "hue_vs_sat": hue_points(&mut rng, 1.0, 30, 0.01) },
        "elements": elements, "strength": strength });
    let mut mask_kinds = Vec::new();
    if rng.below(3) == 0 {
        node["masks"] =
            json!({ "invert": false, "items": [ row_json(MaskKind::Radial, &mut rng) ] });
        mask_kinds.push(MaskKind::Radial);
    }
    if rng.below(8) == 0 {
        node["bypass"] = json!(true);
    }
    let mut items = vec![json!({ "id": 1 })];
    let mut kinds = Vec::new();
    let mut links = vec![
        json!({ "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } }),
    ];
    let mut last = 1;
    if with_corrector {
        items.push(json!({ "id": 2, "kind": "corrector", "pos": [2, 0.0], "trim": { "exposure_ev": 0.4 } }));
        kinds.push(NodeKind::Corrector);
        links.push(json!({ "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } }));
        last = 2;
    }
    items.push(node);
    kinds.push(NodeKind::ReferenceGrade);
    links.push(json!({ "from": { "node": last, "port": "picture" }, "to": { "node": id, "port": "picture" } }));
    links.push(json!({ "from": { "node": id, "port": "picture" }, "to": { "node": "output", "port": "picture" } }));
    let doc = json!({ "nodes": { "items": items, "links": links } });
    let settings = serde_json::from_value::<DevelopSettings>(doc.clone())
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    Graph {
        seed,
        index,
        settings,
        json: doc,
        nodes: kinds,
        masks: mask_kinds,
        masked: None,
        branches: 0,
        parallel_films: 0,
        compare: true,
        face_live: false,
        film_chain: 0,
    }
}

/// The Reference Grade graphs of a run: three for each fixed seed.
fn reference_graphs() -> Vec<Graph> {
    FIXED_SEEDS
        .into_iter()
        .flat_map(|seed| (0..3).map(move |i| generate_reference(seed, i)))
        .collect()
}

impl Graph {
    /// Whether the graph has a Reference Grade node that is not bypassed and has a live element grade (what needs the scene).
    fn reference_live_elements(&self) -> bool {
        self.settings
            .nodes
            .references
            .values()
            .any(|r| !r.bypass && !r.effective().elements.is_default())
    }

    /// Whether the graph has a Reference Grade node that is not bypassed and has a live hue curve or element grade (what runs the hue and element kernels).
    fn reference_live(&self) -> bool {
        self.settings
            .nodes
            .references
            .values()
            .any(|r| !r.bypass && !r.effective().hue_and_elements_are_identity())
    }
}

/// The kind is drawn by its own generator, every graph of a run holds it, and a graph has a live hue or element step.
#[test]
fn the_generator_draws_the_reference_grade_node_and_some_graph_has_it_live() {
    assert!(!generated_nodes().contains(&NodeKind::ReferenceGrade));
    let all = reference_graphs();
    assert_eq!(all.len(), 12);
    assert!(
        all.iter()
            .all(|g| g.nodes.contains(&NodeKind::ReferenceGrade)),
        "the kind is in every Reference Grade graph"
    );
    assert!(
        all.iter().any(Graph::reference_live),
        "no live node in a run"
    );
    assert!(
        all.iter()
            .any(|g| !g.settings.nodes.references.values().all(|r| r.bypass)),
        "every node is bypassed"
    );
    // The draws of the other generator are untouched: it never draws the kind.
    assert!(
        !graphs()
            .iter()
            .any(|g| g.nodes.contains(&NodeKind::ReferenceGrade))
    );
}

/// Random graphs with a Reference Grade node render on the CPU, the same bits whatever the tile size (97, 160, 256 px) and the
/// thread count (1, 3, 4), the same bits again on a repeat, and the node changes a pixel in some graph (the hue curves do; the
/// element planes are RG-04's, so the element grades have no mask here and change none).
#[test]
fn reference_grade_graphs_are_the_same_bits_whatever_the_tiles_and_the_threads() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let plain = d.source();
    let plain_bits = full_bits(&cpu_renderer(256, 1), &plain, 1);
    let mut elements_changed = 0;
    let configs = [(97usize, 1usize), (160, 3), (256, 4)];
    let renderers: Vec<Renderer> = configs
        .iter()
        .map(|&(t, n)| cpu_scene_renderer(t, n))
        .collect();
    let mut changed = 0;
    for g in reference_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        say(&format!(
            "{name}: nodes {:?}, masks {:?}, live {}",
            g.nodes.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.masks.iter().map(|k| k.name()).collect::<Vec<_>>(),
            g.reference_live(),
        ));
        let src = d.source().with_settings(g.settings.clone());
        let first = full_bits(&renderers[0], &src, configs[0].1);
        for (r, &(tile, threads)) in renderers.iter().zip(&configs).skip(1) {
            assert!(
                full_bits(r, &src, threads) == first,
                "{name}: tile {tile} on {threads} threads is other bits; settings: {}",
                g.json
            );
        }
        assert!(
            full_bits(&renderers[0], &src, configs[0].1) == first,
            "{name}: a repeat is other bits"
        );
        let st = view(false, false, None);
        for r in [&renderers[0], &renderers[2]] {
            settle_scene(r, &src, &st);
        }
        assert!(
            frame(&renderers[0], &src, &st).rgba == frame(&renderers[2], &src, &st).rgba,
            "{name}: the Fit frame depends on the threads"
        );
        changed += usize::from(first != plain_bits);
        // The same graph with its element grades taken out: where a node has a live element grade the masks change a pixel.
        if g.reference_live_elements() {
            let mut doc = g.json.clone();
            for item in doc["nodes"]["items"].as_array_mut().expect("items") {
                if item["kind"] == "reference_grade" {
                    item.as_object_mut().expect("a node").remove("elements");
                }
            }
            let bare = serde_json::from_value::<DevelopSettings>(doc)
                .expect("a document")
                .validated()
                .expect("a valid graph")
                .0;
            let bare_bits = full_bits(&renderers[0], &d.source().with_settings(bare), 1);
            elements_changed += usize::from(first != bare_bits);
        }
    }
    assert!(
        changed > 0,
        "no graph with a Reference Grade node changed a pixel"
    );
    assert!(
        elements_changed > 0,
        "no graph's element grades changed a pixel"
    );
}

/// A Reference Grade node at its identity changes no pixel: a default node, a bypassed node, a node at Strength 0 and one whose
/// fitted fields are all at the identity render the bits of the photo with no node.
#[test]
fn a_reference_grade_node_at_its_identity_changes_no_pixel() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(320, 210, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(7);
    let d = Developed::new(&scene, MODEL);
    let cpu = cpu_scene_renderer(160, 2);
    let plain_bits = full_bits(&cpu, &d.source(), 2);
    let node = |body: Value| {
        let mut n = json!({ "id": 2, "kind": "reference_grade", "pos": [2, 0.0] });
        n.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        let doc = json!({ "nodes": { "items": [ { "id": 1 }, n ], "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
        serde_json::from_value::<DevelopSettings>(doc)
            .expect("a document")
            .validated()
            .expect("a valid graph")
            .0
    };
    let fitted = json!({
        "grade": { "saturation": { "kind": "manual", "value": -20 },
                   "curve": { "luma": [[0, 0], [0.3, 0.33], [1, 1]] } },
        "hue_curves": { "hue_vs_hue": [ { "hue_deg": 0, "value": 20 } ],
                        "hue_vs_sat": [ { "hue_deg": 90, "value": 1.5 } ] },
        "elements": { "sky": { "hue_deg": 10, "chroma": 1.2 } } });
    let mut bypassed = fitted.clone();
    bypassed["bypass"] = json!(true);
    let mut strength_zero = fitted.clone();
    strength_zero["strength"] = json!(0);
    for (name, body) in [
        ("default", json!({})),
        ("bypassed", bypassed),
        ("strength 0", strength_zero),
    ] {
        let src = d.source().with_settings(node(body));
        assert!(
            full_bits(&cpu, &src, 2) == plain_bits,
            "{name}: the node changed a bit"
        );
    }
    // The fitted node does change the picture.
    let src = d.source().with_settings(node(fitted));
    assert!(full_bits(&cpu, &src, 2) != plain_bits);
}

/// Since RG-08 the device runs a plan with a live hue curve or element grade (`docs/34` section 7.2): it is not refused, the frame
/// has no CPU reason, and its pixels are within the CPU renderer's bounds (a code, as every graph of the sweeps), with the node in
/// them. Strength 0, a default node and a node with a corrector's steps alone run on the device as they did.
#[test]
fn a_live_reference_grade_node_runs_on_the_device_and_matches_the_cpu() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let cpu = cpu_scene_renderer(256, 2);
    let device = device_scene_renderer(Arc::clone(&gpu), 256);
    let make = |body: Value| {
        let mut n = json!({ "id": 2, "kind": "reference_grade", "pos": [2, 0.0] });
        n.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        let doc = json!({ "nodes": { "items": [ { "id": 1 }, n ], "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
        serde_json::from_value::<DevelopSettings>(doc)
            .expect("a document")
            .validated()
            .expect("a valid graph")
            .0
    };
    let st = view(false, false, None);
    let plain = frame(&cpu, &d.source(), &st);
    for (what, body, moves) in [
        (
            "hue curves",
            json!({ "hue_curves": {
                "hue_vs_hue": [ { "hue_deg": 30, "value": 25 } ], "hue_vs_sat": [ { "hue_deg": 200, "value": 1.6 } ] } }),
            true,
        ),
        (
            "element grades",
            json!({ "elements": { "sky": { "hue_deg": 25, "chroma": 1.5 }, "built": { "exposure_ev": 0.6 },
                                  "subject": { "exposure_ev": -0.5 } } }),
            true,
        ),
        (
            "Strength 0",
            json!({ "strength": 0,
                "hue_curves": { "hue_vs_hue": [ { "hue_deg": 30, "value": 25 } ] },
                "elements": { "sky": { "hue_deg": 10 } } }),
            false,
        ),
        ("a default node", json!({}), false),
        (
            "a corrector's steps alone",
            json!({ "trim": { "exposure_ev": 0.4 }, "grade": { "contrast": 20 } }),
            true,
        ),
    ] {
        let src = d.source().with_settings(make(body));
        settle_scene(&cpu, &src, &st);
        settle_scene(&device, &src, &st);
        let got = frame(&device, &src, &st);
        assert!(
            !got.report.device_refused && got.report.cpu_reason.is_none(),
            "{what}: device {:?}, refused {}, reason {:?}",
            got.report.device,
            got.report.device_refused,
            got.report.cpu_reason
        );
        assert_eq!(got.report.device, ComputeDevice::Gpu, "{what}");
        let want = frame(&cpu, &src, &st);
        let (worst, over) = frame_diff(&got, &want);
        assert!(
            worst <= 2 && over < got.rgba.len() / 100,
            "{what}: {worst} codes, {over} over 1"
        );
        let (moved, _) = frame_diff(&got, &plain);
        assert_eq!(
            moved > 2,
            moves,
            "{what}: the node's effect on the frame ({moved} codes)"
        );
    }
}

/// A [`renderer`] on the device with a fake scene model, so that the element grades of a Reference Grade node have their masks.
fn device_scene_renderer(gpu: Arc<Gpu>, tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
            full_tile_px: tile,
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            scene_model: Some(ym_core::pipeline::SceneModel::fake_default()),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// Every graph of `reference_graphs()` (hue curves and element grades, with the mask kinds in turn) is exported on the device (PNG
/// 16): the device makes it, the PNG is within one 10-bit code of the CPU's, a repeat and another tile size give the same bytes, and
/// the viewer's frame is the CPU's within the sweeps' bounds.
#[test]
fn reference_grade_graphs_are_exported_on_the_device_and_match_the_cpu() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let cpu_view = cpu_scene_renderer(256, 2);
    let device_view = device_scene_renderer(Arc::clone(&gpu), 256);
    let st = view(false, false, None);
    let (mut worst, mut changed, mut worst_view) = (0u32, 0usize, 0u8);
    let plain_bytes = {
        let home = tempfile::tempdir().expect("dir");
        export(
            &d,
            &d.source().settings,
            cpu_scene_renderer(256, 2),
            home.path(),
            FormatKind::Png,
        )
        .2
    };
    for g in &reference_graphs() {
        let name = format!("seed {} graph {}", g.seed, g.index);
        let src = d.source().with_settings(g.settings.clone());
        // The viewer's frame.
        settle_scene(&cpu_view, &src, &st);
        settle_scene(&device_view, &src, &st);
        let got = frame(&device_view, &src, &st);
        assert!(
            !got.report.device_refused && got.report.cpu_reason.is_none(),
            "{name}: the frame was not made on the device: {:?}",
            got.report.cpu_reason
        );
        let (v, over) = frame_diff(&got, &frame(&cpu_view, &src, &st));
        worst_view = worst_view.max(v);
        assert!(
            v <= 2 && over < got.rgba.len() / 100,
            "{name}: the frame is {v} codes from the CPU's, {over} over 1; settings: {}",
            g.json
        );
        // The export.
        let home = tempfile::tempdir().expect("dir");
        let device = |tile| device_scene_renderer(Arc::clone(&gpu), tile);
        let (on, notes, dev) = export(&d, &g.settings, device(256), home.path(), FormatKind::Png);
        assert_eq!(on, ComputeDevice::Gpu, "{name}: the export; {notes:?}");
        assert!(
            !notes.iter().any(|n| n.contains("graphics card")),
            "{name}: {notes:?}"
        );
        let (_, _, again) = export(&d, &g.settings, device(256), home.path(), FormatKind::Png);
        assert!(again == dev, "{name}: a repeat export is other bytes");
        let (_, _, other) = export(&d, &g.settings, device(97), home.path(), FormatKind::Png);
        assert!(other == dev, "{name}: the export at tile 97 is other bytes");
        let (_, _, want) = export(
            &d,
            &g.settings,
            cpu_scene_renderer(256, 2),
            home.path(),
            FormatKind::Png,
        );
        let max = decode_png16_bytes(&dev)
            .iter()
            .zip(decode_png16_bytes(&want))
            .map(|(a, b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        worst = worst.max(max);
        if max > TEN_BIT_CODE {
            let judged = judge_export(
                &d,
                &g.settings,
                &decode_png16_bytes(&dev),
                &decode_png16_bytes(&want),
                home.path(),
                &name,
            );
            assert!(
                judged.failure.is_none(),
                "{}; the PNG export is {max} of 65535 from the CPU's; settings: {}",
                judged.failure.unwrap_or_default(),
                g.json
            );
        }
        changed += usize::from(dev != plain_bytes);
    }
    say(&format!(
        "{} Reference Grade graphs exported on the device; worst PNG {worst} of 65535 from the CPU's, worst frame {worst_view} codes",
        reference_graphs().len()
    ));
    assert!(changed > 0, "no Reference Grade graph changed a pixel");
}
