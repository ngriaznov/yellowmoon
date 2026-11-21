//! The Colour range row that the first pick of a Colour Uniform node adds (`docs/32` section 2.3): how much of the photo it selects.
//!
//! The row is centred on the picked colour (the mean of a 7 x 7 square at the node's input, as the page asks for it) and has a width.
//! These tests make the node's mask in the Mask only view and report its mean `m` and the share of the photo's pixels with `m > 0.5`
//! (`docs/32` section 10 has the table). The synthetic test is part of the suite; the survey of real photos is `#[ignore]`d (it
//! decodes 24 to 60 megapixel files): `YM_PICK_SURVEY=1 cargo test --release -p ym-core --test uniform_first_pick -- --ignored --nocapture`.
mod common;

use std::path::Path;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_core::pipeline::{
    ChromaTap, FrameOptions, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions,
    Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, NodeId};
use ym_gpu::queue::CancelToken;

/// The paper around a photo in a Mask only frame.
const PAPER: [u8; 3] = [40, 60, 90];
const CELL: f64 = 300.0;
/// The page's square (`UNIFORM_PICK_RADIUS_PX`).
const PICK_RADIUS_PX: usize = 3;
/// The width the first pick's row had up to 0.2.4-beta.4 (the Colour range's own default; the slider reads 39).
const OLD_WIDTH: f64 = 0.05;
/// The width the first pick's row has now: `FIRST_PICK_WIDTH` of `src/state/graphOps.ts` (`docs/32` section 2.3).
const FIRST_PICK_WIDTH: f64 = 0.02;

/// Input, Main, a Colour Uniform node (2) with this Colour range row (none: no mask rows), Output.
fn graph(row: Option<(f64, f64, f64)>) -> DevelopSettings {
    let masks = match row {
        Some((a, b, width)) => json!({ "invert": false, "items": [
            { "kind": "color_range", "mode": "add", "center": [a, b], "width": width, "softness": 0.5 } ] }),
        None => json!({ "invert": false, "items": [] }),
    };
    let doc: Value = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "color_uniform", "pos": [1.0, 0.0], "masks": masks },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

/// A renderer that paints the paper around the photo in [`PAPER`], so that the frame's photo pixels can be told from it.
fn renderer() -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            data_root: Some(data_root()),
            background: PAPER,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// Where a share `(fx, fy)` of a photo of this aspect (width over height) is in the cell, fitted and centred.
fn cell_point(aspect: f64, fx: f64, fy: f64) -> (f64, f64) {
    let (w, h) = if aspect >= 1.0 {
        (CELL, CELL / aspect)
    } else {
        (CELL * aspect, CELL)
    };
    ((CELL - w) / 2.0 + fx * w, (CELL - h) / 2.0 + fy * h)
}

/// The chroma at the node's input of a 7 x 7 square at `css`, as Pick colour of a Colour Uniform node reads it.
fn picked(r: &Renderer, src: &PhotoSource, css: (f64, f64)) -> (f64, f64) {
    let cell = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    let s = src.with_settings(graph(None));
    let c = r
        .sample_chroma_square(
            &s,
            Some(NodeId::Corrector(2)),
            ChromaTap::NodeInput,
            PICK_RADIUS_PX,
            &cell,
            css,
            &CancelToken::new(),
        )
        .expect("a sample")
        .expect("inside the photo");
    (c.a, c.b)
}

/// What a row at `centre` and `width` selects in the Mask only frame of the node: the mean of `m` and the share of the photo's pixels
/// with `m > 0.5`, and the share of the 21 x 21 pixels around the pick (`css`) with `m > 0.5`: how much of the picked place the row keeps.
struct Selected {
    mean: f64,
    share: f64,
    kept: f64,
}

fn selected(
    r: &Renderer,
    src: &PhotoSource,
    centre: (f64, f64),
    width: f64,
    css: (f64, f64),
) -> Selected {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = Some(MaskView {
        node: 2,
        style: MaskStyle::MaskOnly,
        row: None,
        depth: false,
        face: None,
    });
    let s = src.with_settings(graph(Some((centre.0, centre.1, width))));
    let f = r
        .viewport_frame(&s, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("a frame");
    let v: Vec<f64> = f
        .rgba
        .chunks(4)
        .filter(|p| p[..3] != PAPER)
        .map(|p| f64::from(p[0]) / 255.0)
        .collect();
    let n = v.len().max(1) as f64;
    let (cx, cy) = (css.0 as usize, css.1 as usize);
    let mut near = Vec::new();
    for y in cy.saturating_sub(10)..=(cy + 10).min(f.height - 1) {
        for x in cx.saturating_sub(10)..=(cx + 10).min(f.width - 1) {
            near.push(f64::from(f.rgba[(y * f.width + x) * 4]) / 255.0);
        }
    }
    Selected {
        mean: v.iter().sum::<f64>() / n,
        share: v.iter().filter(|m| **m > 0.5).count() as f64 / n,
        kept: near.iter().filter(|m| **m > 0.5).count() as f64 / near.len().max(1) as f64,
    }
}

/// The slider's number for a stored width (`docs/algorithms/06` section 4.3: `0.005 + 0.295 (x / 100)^2`).
fn slider_of(width: f64) -> f64 {
    100.0 * ((width - 0.005) / 0.295).max(0.0).sqrt()
}

/// The widths the survey tries, in stored units.
const WIDTHS: [f64; 7] = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, OLD_WIDTH];

/// Patches of `Layout::Patchwork` that are coloured: 3 and 23 are `[1.0, 0.55, 0.40]` (orange), 8 green, 13 blue, 18 yellow; the other 19 of the
/// 24 are neutral at different levels. A pick on a coloured patch is a pick of "something clearly different from the rest".
fn patchwork() -> Developed {
    let scene = Scene::new(
        600,
        400,
        light("sony-ilce-7rm4", 5500.0),
        Layout::Patchwork,
        0.4,
    );
    Developed::new(&scene, "ILCE-7RM4")
}

/// The patch `k` of the 6 x 4 grid, as a pick point in the cell (the photo is 3:2).
fn patch_point(k: usize) -> (f64, f64) {
    cell_point(
        1.5,
        ((k % 6) as f64 + 0.5) / 6.0,
        ((k / 6) as f64 + 0.5) / 4.0,
    )
}

#[test]
fn the_first_picks_row_selects_the_picked_colour_and_not_the_whole_frame() {
    let r = renderer();
    let d = patchwork();
    let src = d.source();
    // The coloured patches are 3 and 23 (orange), 8 (green), 13 (blue) and 18 (yellow): the picked colour is 2/24 or 1/24 of the frame.
    for (k, patches) in [(3usize, 2.0), (8, 1.0), (13, 1.0), (18, 1.0), (23, 2.0)] {
        let own = patches / 24.0;
        let centre = picked(&r, &src, patch_point(k));
        let new = selected(&r, &src, centre, FIRST_PICK_WIDTH, patch_point(k));
        // The row has the patches of its own colour and next to nothing else, and it has the picked place whole.
        assert!(
            (new.share - own).abs() < 0.02 && (new.mean - own).abs() < 0.02,
            "patch {k}: the row selects {} (mean {}) of the frame, the colour is {own}",
            new.share,
            new.mean
        );
        assert!(
            new.kept > 0.99,
            "patch {k}: the row keeps {} of the pick",
            new.kept
        );
    }
    // The row of beta 4 (the Colour range's own width) reached into the neutral patches on the orange pick: 17 % of the frame for an 8 % colour.
    let orange = picked(&r, &src, patch_point(3));
    let old = selected(&r, &src, orange, OLD_WIDTH, patch_point(3));
    assert!(old.share > 0.12, "the old row selects {}", old.share);
}

/// The survey of `docs/32` section 10: synthetic patches, then the real photos of `testdata/` where they are at hand.
#[test]
#[ignore = "a survey that decodes real photos; prints the table of docs/32 section 10"]
fn survey_of_the_first_picks_row() {
    if std::env::var_os("YM_PICK_SURVEY").is_none() {
        return;
    }
    let r = renderer();
    let print = |label: &str, src: &PhotoSource, css: (f64, f64)| {
        let c = picked(&r, src, css);
        let mut line = format!(
            "{label}: chroma ({:+.4}, {:+.4}) |c| {:.4};",
            c.0,
            c.1,
            c.0.hypot(c.1)
        );
        for w in WIDTHS {
            let t = selected(&r, src, c, w, css);
            line.push_str(&format!(
                " w{w:.3}(s{:.0}) mean {:.3} share {:.3} kept {:.2};",
                slider_of(w),
                t.mean,
                t.share,
                t.kept
            ));
        }
        println!("{line}");
    };
    let d = patchwork();
    for (k, name) in [
        (3, "orange patch"),
        (8, "green patch"),
        (13, "blue patch"),
        (18, "yellow patch"),
    ] {
        print(&format!("synthetic {name}"), &d.source(), patch_point(k));
    }
    let root = data_root().join("../testdata");
    // (file under testdata/, aspect, [(label, fx, fy)]).
    let photos: [(&str, f64, &[(&str, f64, f64)]); 4] = [
        (
            "readme-photos/src/A1M03159.ARW",
            2.0 / 3.0,
            &[
                ("warm wall (skin-like)", 0.22, 0.30),
                ("sky", 0.45, 0.07),
                ("foliage", 0.40, 0.38),
                ("white monument", 0.52, 0.58),
            ],
        ),
        (
            "readme-photos/src/A1M01454.ARW",
            1.5,
            &[
                ("sky", 0.30, 0.10),
                ("field", 0.50, 0.85),
                ("trees", 0.30, 0.45),
            ],
        ),
        (
            "readme-photos/src/A1M01749.ARW",
            1.5,
            &[
                ("ochre wall", 0.21, 0.43),
                ("terracotta roof", 0.52, 0.44),
                ("grass", 0.50, 0.88),
            ],
        ),
        (
            "readme-photos/src/A1M01739.ARW",
            1.5,
            &[
                ("sky", 0.50, 0.08),
                ("plough", 0.40, 0.65),
                ("grass", 0.50, 0.90),
            ],
        ),
    ];
    for (file, aspect, points) in photos {
        let path = root.join(file);
        if !path.exists() {
            println!("{file}: absent, skipped");
            continue;
        }
        let dir = tempfile::tempdir().expect("temp dir");
        link(&path, &dir.path().join("DSC00001.ARW"));
        let d = Developed::open(dir);
        for (label, fx, fy) in points {
            print(
                &format!("{file} {label}"),
                &d.source(),
                cell_point(aspect, *fx, *fy),
            );
        }
    }
}

fn link(from: &Path, to: &Path) {
    std::os::unix::fs::symlink(from.canonicalize().expect("the file"), to).expect("a link");
}
