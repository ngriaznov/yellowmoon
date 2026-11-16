//! L1 at Anamorphic: a defocused light keeps a soft oval outline wherever it stands (`docs/21` section 4, L1, *As built
//! (lights the frame cuts)*).
//!
//! A light at the centre, on a border, in a corner or across a seam of tiles must come out an oval twice as tall as
//! wide, with ends that taper and nothing of it beyond them. A light that is not found as a ball is not stretched as
//! one: the vertical gather of the ground smears it into a pillar with straight sides (a disc blurred along a line is a
//! stadium), which is the hard straight edge the owner saw on the lights the frame cuts (2026-11-16).
//!
//! The scenes are synthetic: a textured ground blurred by the lights' own radius, with soft-rimmed lights on it, some
//! clipped, with a glow. Each light is judged on the light it adds (the picture with it minus the same picture
//! without it, with the same blur map in both, so the ground cancels), by [`Verdict`].

#![allow(clippy::needless_range_loop)]

mod common;
mod scenes;

use std::sync::OnceLock;

use common::{camera, frame, repo_root, whole};
use scenes::{disc_blur, texture};
use ym_lens::{
    Axis, BlurMapInput, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan, LensPlanes,
    reshape, reshape_halo,
};

const W: usize = 1000;
const H: usize = 667;
/// The stretch of Anamorphic 100: the oval is this many times taller than wide.
const Q: f64 = 2.0;

/// The shipped lens data (the synthetic library has no swirl or cat's-eye in its Default).
fn shipped() -> &'static LensLibrary {
    static LIB: OnceLock<LensLibrary> = OnceLock::new();
    LIB.get_or_init(|| {
        LensLibrary::load(&repo_root().join("data")).expect("the shipped lens data load")
    })
}

/// A plan at Anamorphic `amount`: the synthetic library's lens, or the shipped one.
fn plan_of(kind: LensKind, amount: f64, use_shipped: bool) -> LensPlan {
    if !use_shipped {
        return common::plan(kind, 0.0, 0.0, amount, W, H, Axis::X);
    }
    let amounts = LensAmounts {
        kind,
        flares: 0.0,
        edges: 0.0,
        anamorphic: amount,
    };
    LensPlan::new(shipped(), &amounts, frame(W, H, Axis::X), camera()).expect("plan")
}

/// A defocused light: centre and radius in pixels, rim width over radius, the level it would have (above 1 it is
/// clipped at 1: a flat plateau) and the amplitude of the glow around it over its level.
#[derive(Clone, Copy, Debug)]
struct Light {
    cx: f64,
    cy: f64,
    r: f64,
    soft: f64,
    level: f64,
    glow: f64,
    /// The disc's reach along x and y over `r`: 1 and 1 for a disc, more for a tube.
    ex: f64,
    ey: f64,
}

/// The textured ground blurred by `blur` pixels, without lights.
fn ground(blur: f64, seed: u64) -> Vec<f32> {
    ground_cell(blur, seed, 30.0)
}

/// The same with cells of `cell` px: a busy ground has small ones.
fn ground_cell(blur: f64, seed: u64, cell: f64) -> Vec<f32> {
    let tex = texture(W, H, cell, seed, 0, W);
    let mut out = Vec::with_capacity(3 * W * H);
    for c in 0..3 {
        let b = disc_blur(tex.plane(c), W, H, blur);
        out.extend(b.iter().map(|v| 0.02 + 0.10 * v));
    }
    out
}

/// The ground with the lights drawn on it (additive, clipped at 1 in the light's own value).
fn with_lights(ground: &[f32], lights: &[Light]) -> Vec<f32> {
    let n = W * H;
    let mut data = ground.to_vec();
    for (k, l) in lights.iter().enumerate() {
        let tint = [
            1.0,
            0.85 - 0.05 * (k % 3) as f64,
            0.6 + 0.05 * (k % 4) as f64,
        ];
        let reach = l.r * 4.0 + 6.0;
        let (x0, x1) = (
            (l.cx - reach).floor().max(0.0) as usize,
            ((l.cx + reach).ceil() as usize).min(W),
        );
        let (y0, y1) = (
            (l.cy - reach).floor().max(0.0) as usize,
            ((l.cy + reach).ceil() as usize).min(H),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f64 + 0.5 - l.cx) / l.ex).hypot((y as f64 + 0.5 - l.cy) / l.ey);
                let width = (l.soft * l.r).max(1.0);
                let t = ((l.r + 0.5 * width - d) / width).clamp(0.0, 1.0);
                let a = t * t * (3.0 - 2.0 * t);
                let core = (l.level * a).min(1.0);
                let glow = if d > l.r {
                    l.glow * l.level.min(1.0) * (-(d - l.r) / (0.5 * l.r)).exp()
                } else {
                    0.0
                };
                for c in 0..3 {
                    data[c * n + y * W + x] += ((core + glow) * tint[c]) as f32;
                }
            }
        }
    }
    data
}

/// A blur map of the whole frame: out of focus everywhere at `radius_px`.
fn blur_map(radius_px: f64) -> BlurMapInput {
    blur_map_bar(radius_px, None)
}

/// ... with a sharp bar (a subject's edge) over the columns `bar`: the cells there are in focus.
fn blur_map_bar(radius_px: f64, bar: Option<(usize, usize)>) -> BlurMapInput {
    let hd = 0.5 * ((W * W + H * H) as f64).sqrt();
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    BlurMapInput {
        width: cw,
        height: ch,
        plane_width: W,
        plane_height: H,
        b: (0..cw * ch)
            .map(|i| match bar {
                Some((a, b)) if (2 * (i % cw)..2 * (i % cw) + 2).any(|x| x >= a && x < b) => {
                    (1.0 / hd) as f32
                }
                _ => (radius_px / hd) as f32,
            })
            .collect(),
        p: (0..cw * ch)
            .map(|i| match bar {
                Some((a, b)) if (2 * (i % cw)..2 * (i % cw) + 2).any(|x| x >= a && x < b) => 0.02,
                _ => 0.95,
            })
            .collect(),
    }
}

/// L1 over the tiles `nx` by `ny` of the frame (1 by 1 is the whole frame), assembled.
fn reshaped(p: &LensPlan, planes: &LensPlanes, data: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * W * H];
    let halo = reshape_halo(p, planes, 1);
    for rect in common::tiles(W, H, nx, ny) {
        let need = rect.grown(halo, W, H);
        let buf = common::crop(data, W, H, need);
        let input = ImageRef {
            width: need.w,
            height: need.h,
            x0: need.x,
            y0: need.y,
            frame_width: W,
            frame_height: H,
            data: &buf,
        };
        let a = reshape(p, planes, 1, &input, rect).expect("reshape");
        for c in 0..3 {
            for y in 0..rect.h {
                let dst = c * W * H + (rect.y + y) * W + rect.x;
                let src = c * rect.w * rect.h + y * rect.w;
                out[dst..dst + rect.w].copy_from_slice(&a[src..src + rect.w]);
            }
        }
    }
    out
}

/// One scene rendered at Anamorphic 100 with the lights and without them.
struct Rendered {
    /// The ground without the lights, the picture with them, and what L1 makes of the picture with them.
    ground: Vec<f32>,
    lit: Vec<f32>,
    out_lit: Vec<f32>,
    /// What L1 makes of the ground without the lights (the same blur map).
    out_gnd: Vec<f32>,
    /// The number of balls the detector found with the lights.
    balls: usize,
}

/// How a scene is rendered.
#[derive(Clone, Copy)]
struct Setup {
    kind: LensKind,
    /// The shipped lens data (with its swirl and cat's-eye) rather than the synthetic library.
    shipped: bool,
    /// The tiles across and down.
    tiles: (usize, usize),
    /// The classical detector's own blur map rather than a learned one.
    classic: bool,
    /// The ground's cell in px (30; small is a busy ground).
    cell: f64,
    /// Anamorphic, 0 to 100.
    amount: f64,
    /// Columns of a sharp dark bar over the lights (a subject's edge), in focus on the map.
    bar: Option<(usize, usize)>,
}

impl Setup {
    const WHOLE: Setup = Setup {
        kind: LensKind::Default,
        shipped: false,
        tiles: (1, 1),
        classic: false,
        cell: 30.0,
        amount: 100.0,
        bar: None,
    };
}

/// Renders `lights` on a ground blurred by `blur` pixels.
fn render(setup: Setup, blur: f64, lights: &[Light]) -> Rendered {
    let p = plan_of(setup.kind, setup.amount, setup.shipped);
    let mut ground = ground_cell(blur, 5, setup.cell);
    let mut lit = with_lights(&ground, lights);
    if let Some((a, b)) = setup.bar {
        // A sharp dark bar over both, with fine texture so that it reads as in focus.
        for c in 0..3 {
            for y in 0..H {
                for x in a..b {
                    let v = 0.02 + 0.03 * (((x * 7 + y * 13 + c) % 5) as f32 / 4.0);
                    ground[c * W * H + y * W + x] = v;
                    lit[c * W * H + y * W + x] = v;
                }
            }
        }
    }
    let map = blur_map_bar(blur, setup.bar);
    let build = |data: &[f32]| {
        if setup.classic {
            LensPlanes::build(&p, &whole(W, H, data), None, &|| false).expect("planes")
        } else {
            LensPlanes::build_with_map(&p, &whole(W, H, data), None, Some(&map), &|| false)
                .expect("planes")
        }
    };
    let (pl, pg) = (build(&lit), build(&ground));
    let (nx, ny) = setup.tiles;
    Rendered {
        out_lit: reshaped(&p, &pl, &lit, nx, ny),
        out_gnd: reshaped(&p, &pg, &ground, nx, ny),
        balls: pl.bokeh().map_or(0, |b| b.balls().len()),
        lit,
        ground,
    }
}

/// What one light's outline measures: see [`judge`].
#[derive(Clone, Copy, Debug)]
struct Verdict {
    /// The width of the light (pixels above half its level) at 0.8 of the oval's half height from the centre, over its
    /// width at the centre, at each tip inside the frame, the larger. An ellipse is 0.6 there, a pillar with straight
    /// sides 1.0.
    taper: f64,
    /// The light at 1.35 times the oval's half height from the centre over its level, the larger of the tips inside
    /// the frame: nothing beyond the oval's end (an oval's soft rim leaves under 0.05), a pillar's tail is a third of
    /// its level and more.
    beyond: f64,
    /// The light added in all over the light in the picture: twice, as the oval is twice as tall.
    energy: f64,
}

/// Judges the light `l` in the render: the light is the output with the lights over the output without them, in green,
/// and the oval's end is `Q r` from its centre along y. A tip is judged when it is at least 4 px inside the frame.
fn judge(r: &Rendered, l: &Light) -> Verdict {
    let n = W * H;
    let light = |x: usize, y: usize| {
        f64::from(r.out_lit[n + y * W + x]) - f64::from(r.out_gnd[n + y * W + x])
    };
    // The light's level in the output: the median over the inner half of its disc that lies inside the frame.
    let mut inner: Vec<f64> = Vec::new();
    let (ix0, ix1) = (
        (l.cx - 0.5 * l.r).floor().max(0.0) as usize,
        ((l.cx + 0.5 * l.r).ceil() as usize).min(W),
    );
    let (iy0, iy1) = (
        (l.cy - 0.5 * l.r).floor().max(0.0) as usize,
        ((l.cy + 0.5 * l.r).ceil() as usize).min(H),
    );
    for y in iy0..iy1 {
        for x in ix0..ix1 {
            let d = ((x as f64 + 0.5 - l.cx).powi(2) + (y as f64 + 0.5 - l.cy).powi(2)).sqrt();
            if d < 0.5 * l.r {
                inner.push(light(x, y));
            }
        }
    }
    inner.sort_by(f64::total_cmp);
    let level = inner[inner.len() / 2];
    let half = 0.5 * level;
    let (x0, x1) = (
        (l.cx - 2.2 * l.r).floor().max(0.0) as usize,
        ((l.cx + 2.2 * l.r).ceil() as usize).min(W),
    );
    let width = |y: usize| (x0..x1).filter(|&x| light(x, y) >= half).count() as f64;
    let centre_row = (l.cy.floor().max(0.0) as usize).min(H - 1);
    let w0 = width(centre_row).max(1.0);
    let (mut taper, mut beyond) = (0.0f64, 0.0f64);
    let a = Q * l.r;
    for s in [-1.0, 1.0] {
        let tip = l.cy + s * 0.8 * a;
        let past = l.cy + s * 1.35 * a;
        if tip < 4.0 || tip > H as f64 - 4.0 || past < 0.0 || past > H as f64 - 1.0 {
            continue;
        }
        taper = taper.max(width(tip as usize) / w0);
        let tail = (x0..x1)
            .map(|x| light(x, past as usize))
            .fold(0.0f64, f64::max);
        beyond = beyond.max(tail / level);
    }
    // The energy over the light's box, before (the picture with the lights over the ground) and after.
    let (y0, y1) = (
        (l.cy - 2.0 * a - 8.0).floor().max(0.0) as usize,
        ((l.cy + 2.0 * a + 8.0).ceil() as usize).min(H),
    );
    let (mut before, mut after) = (0.0f64, 0.0f64);
    for y in y0..y1 {
        for x in x0..x1 {
            before += f64::from(r.lit[n + y * W + x]) - f64::from(r.ground[n + y * W + x]);
            after += light(x, y);
        }
    }
    Verdict {
        taper,
        beyond,
        energy: after / before.max(1e-9),
    }
}

/// Writes the box of `data` (planar f32, little endian) as `<dir>/<name>_<w>x<h>.f32`.
fn dump(dir: &str, name: &str, data: &[f32], (x0, x1, y0, y1): (usize, usize, usize, usize)) {
    let (w, h) = (x1 - x0, y1 - y0);
    let mut bytes = Vec::with_capacity(12 * w * h);
    for c in 0..3 {
        for y in y0..y1 {
            for x in x0..x1 {
                bytes.extend_from_slice(&data[c * W * H + y * W + x].to_le_bytes());
            }
        }
    }
    std::fs::create_dir_all(dir).expect("dir");
    std::fs::write(format!("{dir}/{name}_{w}x{h}.f32"), bytes).expect("write");
}

/// The places of the lights of a scene: the centre (on the seams of 2 x 2 tiles), one more on the vertical seam, one on
/// the horizontal seam, the middles of the four borders and the four corners, each light centred on the border or the
/// corner, so that the frame cuts it in half or in four.
const PLACES: [(&str, f64, f64); 11] = [
    ("centre", 500.0, 333.0),
    ("seam_x", 500.0, 130.0),
    ("seam_y", 760.0, 333.0),
    ("top", 250.0, 0.0),
    ("bottom", 250.0, 667.0),
    ("left", 0.0, 333.0),
    ("right", 1000.0, 333.0),
    ("corner_tl", 0.0, 0.0),
    ("corner_tr", 1000.0, 0.0),
    ("corner_bl", 0.0, 667.0),
    ("corner_br", 1000.0, 667.0),
];

/// A light at every place, `r` px across, with a rim `soft` radii wide.
fn scene(r: f64, soft: f64, level: f64, glow: f64) -> Vec<(&'static str, Light)> {
    PLACES
        .iter()
        .map(|&(name, cx, cy)| {
            let l = Light {
                cx,
                cy,
                r,
                soft,
                level,
                glow,
                ex: 1.0,
                ey: 1.0,
            };
            (name, l)
        })
        .collect()
}

/// The thresholds. An oval's ends taper to 0.6 at 0.8 of its half height and stay under 0.8 with a soft rim; a pillar is
/// 1.0. Beyond the end an oval has under 0.05 of its level (its rim), a pillar a third and more. The light's energy
/// doubles, as the oval's area does (within a quarter: the soft rim and the frame's cut move it a little).
const TAPER_MAX: f64 = 0.8;
const BEYOND_MAX: f64 = 0.1;
const ENERGY_RANGE: (f64, f64) = (1.5, 2.5);
/// ... and in a corner, where the frame cuts three quarters of the oval and the take-out's disc is cut with it, a little
/// less than that (measured 1.37 to 1.41).
const ENERGY_CORNER_MIN: f64 = 1.25;

/// Renders the scene and judges every light, printing the numbers with `YM_EDGES_REPORT` and dumping crops of the
/// lights (before and after, planar f32) into the folder `YM_EDGES_DUMP` names. Returns the lights that fail.
fn failures(setup: Setup, set: &[(&'static str, Light)], what: &str) -> Vec<String> {
    let blur = set[0].1.r;
    let lights: Vec<Light> = set.iter().map(|(_, l)| *l).collect();
    let rendered = render(setup, blur, &lights);
    let report = std::env::var("YM_EDGES_REPORT").is_ok();
    let dump_dir = std::env::var("YM_EDGES_DUMP").ok();
    let mut failed = Vec::new();
    for (name, l) in set {
        let v = judge(&rendered, l);
        if report {
            println!(
                "{what} {:?} r{} lvl{} {name}: taper {:.2} beyond {:.2} energy {:.2} (balls {})",
                setup.kind, l.r, l.level, v.taper, v.beyond, v.energy, rendered.balls
            );
        }
        if let Some(dir) = &dump_dir {
            let reach = 2.0 * Q * l.r + 12.0;
            let bx = (
                (l.cx - reach).floor().max(0.0) as usize,
                ((l.cx + reach).ceil() as usize).min(W),
                (l.cy - reach).floor().max(0.0) as usize,
                ((l.cy + reach).ceil() as usize).min(H),
            );
            let tag = format!("{what}_{:?}_r{}_l{}_{name}", setup.kind, l.r, l.level);
            dump(dir, &format!("{tag}_in"), &rendered.lit, bx);
            dump(dir, &format!("{tag}_out"), &rendered.out_lit, bx);
        }
        if v.taper > TAPER_MAX
            || v.beyond > BEYOND_MAX
            || !(if name.starts_with("corner") {
                ENERGY_CORNER_MIN
            } else {
                ENERGY_RANGE.0
            }..=ENERGY_RANGE.1)
                .contains(&v.energy)
        {
            failed.push(format!("{what} r{} lvl{} {name}: {v:?}", l.r, l.level));
        }
    }
    failed
}

#[test]
fn soft_clipped_and_big_lights_everywhere_come_out_ovals_with_no_hard_edge() {
    let mut failed = Vec::new();
    for (r, soft, level, glow) in [
        (8.0, 0.4, 0.8, 0.0),
        (14.0, 0.4, 0.8, 0.0),
        (14.0, 0.3, 8.0, 0.05),
        (20.0, 0.4, 8.0, 0.05),
    ] {
        failed.extend(failures(
            Setup::WHOLE,
            &scene(r, soft, level, glow),
            "whole",
        ));
    }
    assert!(failed.is_empty(), "cut outlines: {failed:#?}");
}

#[test]
fn tiles_cut_through_the_lights_leave_no_seam_and_the_same_bits() {
    let set = scene(14.0, 0.3, 8.0, 0.05);
    let lights: Vec<Light> = set.iter().map(|(_, l)| *l).collect();
    let whole = render(Setup::WHOLE, 14.0, &lights);
    for tiles in [(2, 2), (3, 3)] {
        let setup = Setup {
            tiles,
            ..Setup::WHOLE
        };
        let tiled = render(setup, 14.0, &lights);
        assert!(
            tiled
                .out_lit
                .iter()
                .zip(&whole.out_lit)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{tiles:?} tiles differ from the whole frame"
        );
        let failed = failures(setup, &set, "tiles");
        assert!(failed.is_empty(), "{tiles:?}: {failed:#?}");
    }
}

#[test]
fn the_classical_detector_finds_the_lights_the_frame_cuts_too() {
    // No learned map: the detector's own edge readings are the blur evidence.
    let set: Vec<_> = scene(14.0, 0.3, 8.0, 0.05)
        .into_iter()
        .filter(|(n, _)| matches!(*n, "centre" | "top" | "left" | "right" | "bottom"))
        .collect();
    let setup = Setup {
        classic: true,
        ..Setup::WHOLE
    };
    let failed = failures(setup, &set, "classic");
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn nothing_changes_at_anamorphic_0() {
    let set = scene(14.0, 0.3, 8.0, 0.05);
    let mut lights: Vec<Light> = set.iter().map(|(_, l)| *l).collect();
    lights.extend(centre_scene(14.0, 0.3, 8.0, 0.05).iter().map(|(_, l)| *l));
    let g = ground(14.0, 5);
    let lit = with_lights(&g, &lights);
    let p = plan_of(LensKind::Default, 0.0, false);
    let map = blur_map(14.0);
    let planes = LensPlanes::build_with_map(&p, &whole(W, H, &lit), None, Some(&map), &|| false)
        .expect("planes");
    let out = reshaped(&p, &planes, &lit, 1, 1);
    assert!(
        out.iter()
            .zip(&lit)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Anamorphic 0 changed the picture"
    );
}

/// What a light's outline measures when its neighbours may overlap it: see [`judge_columns`].
#[derive(Clone, Copy, Debug)]
struct Columns {
    /// The height of the light (rows above half its level) 0.8 of its half width from its middle column, over its
    /// height there, the smaller of the two sides: an ellipse is 0.6, a pillar 1.0.
    ends: f64,
    /// The light 1.35 half heights above and below the light, over its level, in the light's columns.
    beyond: f64,
}

/// Judges the light by its columns: the height of the oval along the middle column and along columns 0.8 of the way
/// to its sides, and what lies beyond its ends. `q` is the stretch (1 + Anamorphic / 100).
fn judge_columns(r: &Rendered, l: &Light, q: f64) -> Columns {
    let n = W * H;
    let light = |x: usize, y: usize| {
        f64::from(r.out_lit[n + y * W + x]) - f64::from(r.out_gnd[n + y * W + x])
    };
    let (rx, ry) = (l.r * l.ex, l.r * l.ey);
    let a = q * ry;
    let mut inner: Vec<f64> = Vec::new();
    for y in (l.cy - 0.7 * ry).max(0.0) as usize..((l.cy + 0.7 * ry) as usize).min(H) {
        for x in (l.cx - 0.7 * rx).max(0.0) as usize..((l.cx + 0.7 * rx) as usize).min(W) {
            inner.push(light(x, y));
        }
    }
    inner.sort_by(f64::total_cmp);
    let level = inner[inner.len() * 3 / 4].max(1e-6);
    let half = 0.5 * level;
    let (y0, y1) = (
        (l.cy - 2.5 * a).max(0.0) as usize,
        ((l.cy + 2.5 * a) as usize).min(H),
    );
    let height = |x: f64| -> f64 {
        let x = (x.max(0.0) as usize).min(W - 1);
        (y0..y1).filter(|&y| light(x, y) >= half).count() as f64
    };
    let h0 = height(l.cx).max(1.0);
    let ends = height(l.cx + 0.8 * rx).min(height(l.cx - 0.8 * rx)) / h0;
    let mut beyond = 0.0f64;
    for s in [-1.0, 1.0] {
        let past = l.cy + s * 1.35 * a;
        if past < 0.0 || past > H as f64 - 1.0 {
            continue;
        }
        for x in (l.cx - rx).max(0.0) as usize..((l.cx + rx) as usize).min(W) {
            beyond = beyond.max(light(x, past as usize) / level);
        }
    }
    Columns { ends, beyond }
}

/// A scene's group of lights, named, placed at `(ox, oy)`.
fn group(
    name: &str,
    ox: f64,
    oy: f64,
    lights: &[(f64, f64, f64, f64)],
    base: Light,
) -> Vec<(String, Light)> {
    lights
        .iter()
        .enumerate()
        .map(|(i, &(dx, dy, ex, ey))| {
            (
                format!("{name}#{i}"),
                Light {
                    cx: ox + dx,
                    cy: oy + dy,
                    ex,
                    ey,
                    ..base
                },
            )
        })
        .collect()
}

/// The centre scenes: pairs, rows and stacks of overlapping lights, tubes, a light behind a sharp bar.
fn centre_scene(r: f64, soft: f64, level: f64, glow: f64) -> Vec<(String, Light)> {
    let base = Light {
        cx: 0.0,
        cy: 0.0,
        r,
        soft,
        level,
        glow,
        ex: 1.0,
        ey: 1.0,
    };
    let mut v = Vec::new();
    let xs = [150.0, 400.0, 650.0, 880.0];
    let ys = [110.0, 333.0, 555.0];
    let mut k = 0;
    let mut place = |v: &mut Vec<(String, Light)>, name: &str, ls: &[(f64, f64, f64, f64)]| {
        let (ox, oy) = (xs[k % 4], ys[k / 4]);
        k += 1;
        v.extend(group(name, ox, oy, ls, base));
    };
    for ov in [0.1, 0.3, 0.6] {
        let d = 2.0 * r - ov * r;
        place(
            &mut v,
            &format!("pair{ov}"),
            &[(-0.5 * d, 0.0, 1.0, 1.0), (0.5 * d, 0.0, 1.0, 1.0)],
        );
        place(
            &mut v,
            &format!("row{ov}"),
            &[
                (-2.0 * d, 0.0, 1.0, 1.0),
                (-d, 0.0, 1.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (d, 0.0, 1.0, 1.0),
            ],
        );
    }
    for e in [1.6, 2.0, 2.5] {
        place(&mut v, &format!("tubex{e}"), &[(0.0, 0.0, e, 1.0)]);
    }
    for e in [1.7, 2.5] {
        place(&mut v, &format!("tubey{e}"), &[(0.0, 0.0, 1.0, e)]);
    }
    v
}

/// The share of the box of a group of lights that its light fills (the pixels above half its level over the box that
/// holds them), in the render: an ellipse is 0.785, two discs that overlap less, a rounded rectangle (a stadium, what
/// the gather makes of a light it smears) 0.88 and more.
fn group_fill(r: &Rendered, lights: &[Light], q: f64, after: bool) -> f64 {
    let n = W * H;
    let light = |x: usize, y: usize| {
        if after {
            f64::from(r.out_lit[n + y * W + x]) - f64::from(r.out_gnd[n + y * W + x])
        } else {
            f64::from(r.lit[n + y * W + x]) - f64::from(r.ground[n + y * W + x])
        }
    };
    let reach_x = lights.iter().map(|l| l.r * l.ex).fold(0.0, f64::max);
    let reach_y = lights.iter().map(|l| l.r * l.ey * q).fold(0.0, f64::max);
    let x0 = lights
        .iter()
        .map(|l| l.cx - l.r * l.ex)
        .fold(f64::MAX, f64::min)
        - 1.8 * reach_x;
    let x1 = lights
        .iter()
        .map(|l| l.cx + l.r * l.ex)
        .fold(f64::MIN, f64::max)
        + 1.8 * reach_x;
    let y0 = lights.iter().map(|l| l.cy).fold(f64::MAX, f64::min) - 1.8 * reach_y;
    let y1 = lights.iter().map(|l| l.cy).fold(f64::MIN, f64::max) + 1.8 * reach_y;
    let (x0, x1) = (x0.max(0.0) as usize, (x1 as usize).min(W));
    let (y0, y1) = (y0.max(0.0) as usize, (y1 as usize).min(H));
    let mut v: Vec<f64> = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            v.push(light(x, y));
        }
    }
    let mut sorted = v.clone();
    sorted.sort_by(f64::total_cmp);
    let half = 0.5 * sorted[sorted.len() - 1 - sorted.len() / 100];
    let (mut count, mut bx0, mut bx1, mut by0, mut by1) =
        (0usize, usize::MAX, 0usize, usize::MAX, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            if light(x, y) >= half {
                count += 1;
                bx0 = bx0.min(x);
                bx1 = bx1.max(x);
                by0 = by0.min(y);
                by1 = by1.max(y);
            }
        }
    }
    if count == 0 {
        return 0.0;
    }
    count as f64 / (((bx1 - bx0 + 1) * (by1 - by0 + 1)) as f64)
}

/// The darkest light of a group's box (the output over the output without the lights) over its brightest: a light the
/// gather smears next to a take-out or a held place can leave a dark band, which the ground never had.
fn group_dark(r: &Rendered, lights: &[Light], q: f64) -> f64 {
    let n = W * H;
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    let reach = lights
        .iter()
        .map(|l| l.r * l.ex.max(l.ey * q))
        .fold(0.0, f64::max);
    let x0 = lights.iter().map(|l| l.cx).fold(f64::MAX, f64::min) - 2.0 * reach;
    let x1 = lights.iter().map(|l| l.cx).fold(f64::MIN, f64::max) + 2.0 * reach;
    let y0 = lights.iter().map(|l| l.cy).fold(f64::MAX, f64::min) - 2.0 * reach;
    let y1 = lights.iter().map(|l| l.cy).fold(f64::MIN, f64::max) + 2.0 * reach;
    for y in y0.max(0.0) as usize..(y1 as usize).min(H) {
        for x in x0.max(0.0) as usize..(x1 as usize).min(W) {
            let v = f64::from(r.out_lit[n + y * W + x]) - f64::from(r.out_gnd[n + y * W + x]);
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    lo / hi.max(1e-9)
}

/// The lights of a centre scene by group (the name before the `#`).
fn groups(set: &[(String, Light)]) -> Vec<(String, Vec<Light>)> {
    let mut out: Vec<(String, Vec<Light>)> = Vec::new();
    for (n, l) in set {
        let g = n.split('#').next().unwrap_or("").to_string();
        match out.last_mut() {
            Some((name, ls)) if *name == g => ls.push(*l),
            _ => out.push((g, vec![*l])),
        }
    }
    out
}

/// How much the light's fill (see [`group_fill`]) may change: a stretched oval and an unstretched light have the same,
/// a smeared one rises by 0.06 to 0.07 (measured on the tubes before the lights were held).
const FILL_CHANGE_MAX: f64 = 0.03;
/// ... and the darkest light of the group may not be darker than this over its brightest.
const DARK_MIN: f64 = -0.05;

/// Renders a centre scene at Anamorphic `amount` and returns the groups that are cut (fill or dark band).
fn centre_failures(setup: Setup, set: &[(String, Light)], only: &str, what: &str) -> Vec<String> {
    let blur = set[0].1.r;
    let lights: Vec<Light> = set.iter().map(|(_, l)| *l).collect();
    let q = 1.0 + setup.amount / 100.0;
    let rendered = render(setup, blur, &lights);
    let report = std::env::var("YM_EDGES_REPORT").is_ok();
    let mut failed = Vec::new();
    for (g, ls) in groups(set) {
        if !g.starts_with(only) {
            continue;
        }
        let (fin, fout) = (
            group_fill(&rendered, &ls, q, false),
            group_fill(&rendered, &ls, q, true),
        );
        let dark = group_dark(&rendered, &ls, q);
        if report {
            println!(
                "{what} A{} {g}: fill {fin:.2} -> {fout:.2}, dark {dark:.3}",
                setup.amount
            );
        }
        if (fout - fin).abs() > FILL_CHANGE_MAX || dark < DARK_MIN {
            failed.push(format!(
                "{what} A{} {g}: fill {fin:.2} -> {fout:.2}, dark {dark:.3}",
                setup.amount
            ));
        }
    }
    failed
}

#[test]
fn tubes_and_lights_that_are_no_balls_are_not_smeared_into_pillars() {
    // Lights that fail the ball rule (a tube 1.6 to 2.5 times as long as wide, along or across the stretch) are held:
    // as round or as long as they were, never a rounded rectangle or a ring with dark bands.
    let mut failed = Vec::new();
    for amount in [50.0, 100.0] {
        for (r, soft, level, glow) in [(14.0, 0.3, 8.0, 0.05), (14.0, 0.4, 0.8, 0.0)] {
            let setup = Setup {
                amount,
                ..Setup::WHOLE
            };
            failed.extend(centre_failures(
                setup,
                &centre_scene(r, soft, level, glow),
                "tube",
                &format!("lvl{level}"),
            ));
        }
    }
    assert!(failed.is_empty(), "smeared: {failed:#?}");
}

#[test]
fn a_light_half_behind_a_sharp_edge_and_a_light_on_a_busy_ground_are_not_smeared() {
    let base = Light {
        cx: 500.0,
        cy: 333.0,
        r: 14.0,
        soft: 0.3,
        level: 8.0,
        glow: 0.05,
        ex: 1.0,
        ey: 1.0,
    };
    let mut failed = Vec::new();
    for amount in [50.0, 100.0] {
        // The bar hides the right half of the light and is in focus on the map.
        let bar = Setup {
            amount,
            bar: Some((500, 540)),
            ..Setup::WHOLE
        };
        let one = vec![("light#0".to_string(), base)];
        failed.extend(centre_failures(bar, &one, "light", "bar"));
        // A busy ground (cells of 8 px) behind a tube and behind a disc.
        let busy = Setup {
            amount,
            cell: 8.0,
            ..Setup::WHOLE
        };
        let mut set = group("tubex", 300.0, 333.0, &[(0.0, 0.0, 2.0, 1.0)], base);
        set.extend(group("disc", 700.0, 333.0, &[(0.0, 0.0, 1.0, 1.0)], base));
        failed.extend(centre_failures(busy, &set, "tube", "busy"));
    }
    assert!(failed.is_empty(), "smeared: {failed:#?}");
}

#[test]
#[ignore = "a probe: prints the numbers of the centre scenes"]
fn centre_probe() {
    let dump_dir = std::env::var("YM_EDGES_DUMP").ok();
    for amount in [50.0, 100.0] {
        let q = 1.0 + amount / 100.0;
        for (r, soft, level, glow) in [(14.0, 0.3, 8.0, 0.05), (14.0, 0.4, 0.8, 0.0)] {
            let set = centre_scene(r, soft, level, glow);
            let lights: Vec<Light> = set.iter().map(|(_, l)| *l).collect();
            let setup = Setup {
                amount,
                ..Setup::WHOLE
            };
            let rendered = render(setup, r, &lights);
            println!(
                "A{amount} lvl{level}: {} lights, {} balls",
                lights.len(),
                rendered.balls
            );
            let mut names: Vec<String> = set
                .iter()
                .map(|(n, _)| n.split('#').next().unwrap_or("").to_string())
                .collect();
            names.dedup();
            for g in &names {
                let ls: Vec<Light> = set
                    .iter()
                    .filter(|(n, _)| n.split('#').next() == Some(g))
                    .map(|(_, l)| *l)
                    .collect();
                println!(
                    "  {g}: fill {:.2} -> {:.2}",
                    group_fill(&rendered, &ls, q, false),
                    group_fill(&rendered, &ls, q, true)
                );
            }
            for (name, l) in &set {
                let c = judge_columns(&rendered, l, q);
                println!("  {name}: ends {:.2} beyond {:.2}", c.ends, c.beyond);
                if let Some(dir) = &dump_dir {
                    let reach = 3.0 * q * r * l.ey.max(l.ex) + 12.0;
                    let bx = (
                        (l.cx - reach).max(0.0) as usize,
                        ((l.cx + reach) as usize).min(W),
                        (l.cy - reach).max(0.0) as usize,
                        ((l.cy + reach) as usize).min(H),
                    );
                    let tag = format!("a{amount}_l{level}_{}", name.replace('#', "_"));
                    dump(dir, &format!("{tag}_in"), &rendered.lit, bx);
                    dump(dir, &format!("{tag}_out"), &rendered.out_lit, bx);
                }
            }
        }
    }
}
