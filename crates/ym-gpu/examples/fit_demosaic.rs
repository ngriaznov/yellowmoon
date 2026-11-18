//! The fit harness of spec 01, the adaptive ratio demosaic (`docs/29-raw-constants.md` section 2.3, task
//! RC-05): evaluates a `Tuning` of the demosaic on the scenes of the audit and prints one JSON line per
//! evaluation. Measurement only: nothing here changes a default and nothing is written to disk.
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example fit_demosaic \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' -- \
//!     [--from oracle|default] [--set name=value]... [--rel name=0.9,1.1]... \
//!     [--eval "label@name=v,name=v"]... [--held] [--photo FILE.ARW]... \
//!     [--artifacts] [--edge] [--ca | --ca-only] [--ca-size WxH] [--list]
//! ```
//!
//! With no evaluation argument only the base line is printed. The base is the starting `Tuning`
//! (`--from oracle`, the default: `Tuning::ORACLE`, the constants of the research package; or
//! `--from default`, what ships), and every other line carries its change against it. A field is named as
//! in `adaptive_ratio::Tuning` (`--list` prints them): `--set sat_dominance=0.8`,
//! `--rel chroma_floor=0.9,1.1` (the field at 0.9 x and 1.1 x its value, one line each).
//!
//! The scenes are those of section 2.2 ("How the numbers were made"):
//!
//! * **A, clean:** the procedural set of `tests/cpu_demosaic.rs` (7), Kodak (24) and McMaster (18) in
//!   linear light at signal scale 1, unit gains, the nominal low-noise model of spec 01 section 6.3, shadow
//!   matching `Auto`. PSNR against the ground truth, 10 px border off. Fit set: procedural and Kodak;
//!   held-out: McMaster.
//! * **B, dark and noisy:** every third image of that list at 0.25 of white with Poisson-Gaussian noise
//!   (`a = 5e-4`, `b = 4e-5`, row noise 1e-3), the matching model passed. Seeded (`77 + index`). Same split;
//!   `--held` adds **B2**, the same images with a second noise seed (`1077 + index`), all held out.
//! * **C, real mosaics:** two crops of each `--photo` file (an ARW; it stays on this machine), as-shot
//!   gains, RGGB assumed at the readout origin, two stand-in noise models. No ground truth: only the change
//!   against the base is measured.
//! * **P, PixelShift200:** the images of `testdata/demosaic/pixelshift200` when that folder exists, as A
//!   (the one real-sensor ground truth). When it does not, the data note says the set was skipped.
//! * `--ca`: the five CA scenes of `ca_green_residual` (`tests/common/ca_scenes.rs`, ours, 600 x 400 by
//!   default), the residual of red and blue against green after R9 and the PSNR of the CA path; `--ca-only`
//!   runs these scenes and no other.
//! * `--artifacts`: zipper rate, false-colour rate and chroma difference on the A scenes (the gate's
//!   definitions, `tests/common/demosaic_metrics.rs`).
//! * `--edge`: a slanted edge at 0.25 of white with the noise of B through the demosaic: MTF50, rise,
//!   overshoot and the noise of the flat halves (the objective of the shadow sharpening, `docs/29` F-05).
//! * `--skip-a`: leave the A scenes (and the extra cells) out, so that a grid over the shadow constants costs
//!   the B scenes and the edge only (RC-08). The A keys of the line are then `null`.
//! * `--edge-seeds N`: the edge numbers are the mean over `N` noise seeds of the slanted edge (default 1, the
//!   seed of RC-05 to RC-07); a smoother objective for the shadow sharpening grid.
//! * `--noisy [on|auto|off]`: the noisy table of `demosaic_gate` (the four ISO-like levels on the procedural
//!   set, Kodak and McMaster, signal scale 1, the gate's noise and seeds), PSNR after the demosaic and after
//!   the R6 bilateral, with the shadow colour matching forced `on` (the objective of A11, so that it acts at
//!   every level), left to `auto` (what the gate does) or forced `off` (default `on`).
//! * `--ladder`: shadow colour matching on and off over a ladder of eight noise models from 40 dB to 15 dB at
//!   18 % grey, on the same images (A14's crossover).
//! * `--cells`: the nine cells of `demosaic_gate` (the A images at signal scales 1, 0.1 and 0.01, nominal
//!   model fixed): the set means, `mean9` (the gate's aggregate), `mean_fit` and `mean_held` (RC-07).
//!
//! The change against the base is the audit's measure: the mean and the largest `|difference|` in 8-bit
//! sRGB codes. The data lives in `testdata/` (`YM_TESTDATA` moves it), is for evaluation and local fitting,
//! and is never committed.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_gpu::cpu::demosaic::DemosaicInputs;
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params, ShadowMatch, Tuning};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::sharpen::diffraction::SPEC_LUMA;
use ym_gpu::cpu::sharpen::diffraction::eval::{EdgeGeometry, edge_stats, patch_sd};
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, NoiseSigma, RgbImage, synth};
use ym_noise::model::NoiseModel;

#[path = "../tests/common/ca_scenes.rs"]
mod ca_scenes;
#[path = "../tests/common/demosaic_metrics.rs"]
mod demosaic_metrics;
#[path = "../tests/common/fit.rs"]
mod fit;

use demosaic_metrics::{
    BORDER_PX, LabImage, artifacts, demosaic_sets_root, linearize, load_set, mean_finite,
    synthetic_set,
};
use fit::{Common, Fields, Rng, Start, code_diff, data_note, emit, mean, num};

/// What a scene is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// Clean, scale 1, nominal model.
    A,
    /// Dark and noisy, noise seed 1.
    B,
    /// The same images with the second noise seed (held out).
    B2,
    /// A real mosaic (no ground truth).
    C,
    /// PixelShift200 ground truth.
    P,
    /// The clean A images at signal scale 0.1 (the nine cells of the gate; only with `--cells`).
    Cell01,
    /// The clean A images at signal scale 0.01 (the nine cells of the gate; only with `--cells`).
    Cell001,
}

impl Group {
    pub fn key(self) -> &'static str {
        match self {
            Group::A => "A",
            Group::B => "B",
            Group::B2 => "B2",
            Group::C => "C",
            Group::P => "P",
            Group::Cell01 => "cell_0.1",
            Group::Cell001 => "cell_0.01",
        }
    }
}

/// The source of a scene: the fit set is the procedural set and Kodak, McMaster is held out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Synthetic,
    Kodak,
    McMaster,
    PixelShift,
    Photo,
}

impl Source {
    pub fn held_out(self) -> bool {
        matches!(self, Source::McMaster | Source::PixelShift)
    }
}

pub struct Scene {
    pub group: Group,
    pub source: Source,
    /// The truth, scaled like the mosaic (`None` for a real mosaic).
    pub truth: Option<RgbImage>,
    pub mosaic: Mosaic,
    pub gains: [f32; 3],
    pub noise: NoiseDn,
    /// The signal scale of the scene (white = 1 / scale).
    pub scale: f64,
    pub shadow: ShadowMatch,
}

pub fn model(a: f64, b: f64, row: f64) -> NoiseModel {
    NoiseModel {
        a,
        b,
        delta: [0.0; 4],
        sigma_row: row,
        q: 1.0 / 65535.0,
    }
}

/// Poisson-Gaussian noise of `m` on every photosite of the mosaic, seeded, in the order of the data.
pub fn add_noise(mosaic: &mut Mosaic, m: &NoiseModel, seed: u64) {
    let mut rng = Rng(seed);
    for v in mosaic.data.iter_mut() {
        let x = f64::from(*v).max(0.0);
        *v = (x + (m.a * x + m.b + m.sigma_row * m.sigma_row).sqrt() * rng.gauss()) as f32;
    }
}

/// A clean scene: the mosaic of `truth` at signal scale 1, unit gains, the nominal low-noise model of
/// spec 01 section 6.3 (`a = 2e-5`, `b = 1e-8`), shadow matching `Auto`.
pub fn clean_scene(group: Group, source: Source, truth: &RgbImage) -> Scene {
    Scene {
        group,
        source,
        truth: Some(truth.clone()),
        mosaic: synth::mosaic_from_rgb(truth, Cfa::RGGB),
        gains: [1.0; 3],
        noise: NoiseDn::from_model(&model(2e-5, 1e-8, 0.0)),
        scale: 1.0,
        shadow: ShadowMatch::Auto,
    }
}

/// A dark, noisy scene: `truth` at 0.25 of white with Poisson-Gaussian noise (`a = 5e-4`, `b = 4e-5`,
/// row noise 1e-3, the "high" model of spec 01 section 2.2) from `seed`, the matching model passed.
pub fn dark_noisy_scene(group: Group, source: Source, truth: &RgbImage, seed: u64) -> Scene {
    let high = model(5e-4, 4e-5, 1e-3);
    let mut ts = truth.clone();
    for v in ts.data.iter_mut() {
        *v *= 0.25;
    }
    let mut m = synth::mosaic_from_rgb(&ts, Cfa::RGGB);
    add_noise(&mut m, &high, seed);
    Scene {
        group,
        source,
        truth: Some(ts),
        mosaic: m,
        gains: [1.0; 3],
        noise: NoiseDn::from_model(&high),
        scale: 0.25,
        shadow: ShadowMatch::Auto,
    }
}

pub struct Loaded {
    pub scenes: Vec<Scene>,
    pub used: Vec<(String, usize)>,
    pub skipped: Vec<(String, String)>,
}

/// Every scene of the run. `second_seed` adds B2, `photos` the C scenes.
pub fn load_scenes(
    root: &Path,
    second_seed: bool,
    photos: &[PathBuf],
    cells: bool,
    skip_a: bool,
) -> Loaded {
    let mut sets: Vec<(String, Source, RgbImage)> = Vec::new();
    let mut used = Vec::new();
    let mut skipped = Vec::new();
    for (dir, source) in [("kodak", Source::Kodak), ("mcmaster", Source::McMaster)] {
        let images = load_set(&root.join(dir));
        if images.is_empty() {
            skipped.push((
                dir.to_owned(),
                format!(
                    "no images in {} (run tools/fetch/demosaic-sets.sh)",
                    root.join(dir).display()
                ),
            ));
        } else {
            used.push((dir.to_owned(), images.len()));
            for (n, img) in images {
                sets.push((format!("{dir}/{n}"), source, linearize(&img)));
            }
        }
    }
    let synthetic = synthetic_set();
    used.push(("synthetic (cpu_demosaic.rs)".to_owned(), synthetic.len()));
    for (n, img) in synthetic {
        sets.push((n, Source::Synthetic, img));
    }
    let mut scenes = Vec::new();
    for (_, source, t) in sets.iter().filter(|_| !skip_a) {
        scenes.push(clean_scene(Group::A, *source, t));
    }
    // The other two signal scales of the nine cells of `demosaic_gate`: the same images scaled by 0.1 and
    // 0.01, the nominal model unchanged (it is fixed at every scale in the gate).
    if cells && !skip_a {
        for (_, source, t) in &sets {
            for (group, scale) in [(Group::Cell01, 0.1f32), (Group::Cell001, 0.01f32)] {
                let mut sc = clean_scene(group, *source, &demosaic_metrics::scaled(t, scale));
                sc.scale = f64::from(scale);
                scenes.push(sc);
            }
        }
    }
    // B: every third image, at 0.25 of white, ISO 6400-like.
    for (k, (_, source, t)) in sets.iter().enumerate().filter(|(k, _)| k % 3 == 0) {
        let seeds = [(Group::B, 77u64), (Group::B2, 1077u64)];
        for (group, base) in seeds {
            if group == Group::B2 && !second_seed {
                continue;
            }
            scenes.push(dark_noisy_scene(group, *source, t, base + k as u64));
        }
    }
    // P: PixelShift200, when it is there.
    let ps = load_set(&root.join("pixelshift200"));
    if ps.is_empty() {
        skipped.push((
            "pixelshift200".to_owned(),
            format!(
                "not present at {} (the one real-sensor ground truth; docs/29 G5)",
                root.join("pixelshift200").display()
            ),
        ));
    } else {
        used.push(("pixelshift200".to_owned(), ps.len()));
        for (_, img) in ps {
            let t = linearize(&img);
            scenes.push(clean_scene(Group::P, Source::PixelShift, &t));
        }
    }
    // C: crops of real files.
    for photo in photos {
        match load_photo_crops(photo) {
            Ok(mut crops) => {
                used.push((
                    format!(
                        "photo {}",
                        photo
                            .file_name()
                            .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
                    ),
                    crops.len(),
                ));
                scenes.append(&mut crops);
            }
            Err(e) => skipped.push((format!("photo {}", photo.display()), e)),
        }
    }
    Loaded {
        scenes,
        used,
        skipped,
    }
}

/// Two 1024 x 768 crops of a real ARW with two stand-in noise models each (ISO 100-like and ISO
/// 6400-like; stand-ins, not measurements of the body).
fn load_photo_crops(path: &PathBuf) -> Result<Vec<Scene>, String> {
    let raw = ym_raw::decode::decode(path).map_err(|e| format!("cannot decode: {e}"))?;
    let (blk, wht) = (f64::from(raw.black[0]), f64::from(raw.white[0]));
    let wb = raw.wb_as_shot.unwrap_or([1024.0; 4]);
    let g = f64::from(wb[1]);
    let gains = [
        (f64::from(wb[0]) / g) as f32,
        1.0,
        (f64::from(wb[3]) / g) as f32,
    ];
    let (pw, ph) = (raw.data.width, raw.data.height);
    let mut out = Vec::new();
    for (fx, fy) in [(0.5, 0.5), (0.3, 0.7)] {
        let (w, h) = (1024usize, 768usize);
        if pw < w + 2 || ph < h + 2 {
            return Err(format!("{pw} x {ph} is too small for a {w} x {h} crop"));
        }
        let x0 = ((pw as f64 * fx) as usize - w / 2) & !1;
        let y0 = ((ph as f64 * fy) as usize - h / 2) & !1;
        let mut data = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = f64::from(raw.data.data[(y0 + y) * pw + x0 + x]);
                data[y * w + x] = ((v - blk) / (wht - blk)) as f32;
            }
        }
        let m = Mosaic::from_data(w, h, Cfa::RGGB, data).map_err(|e| format!("{e:?}"))?;
        out.push(Scene {
            group: Group::C,
            source: Source::Photo,
            truth: None,
            mosaic: m.clone(),
            gains,
            noise: NoiseDn::from_model(&model(2.5e-5, 4e-8, 5e-5)),
            scale: 1.0,
            shadow: ShadowMatch::Auto,
        });
        out.push(Scene {
            group: Group::C,
            source: Source::Photo,
            truth: None,
            mosaic: m,
            gains,
            noise: NoiseDn::from_model(&model(2e-3, 2e-6, 2e-4)),
            scale: 1.0,
            shadow: ShadowMatch::On,
        });
    }
    Ok(out)
}

pub fn run_scene(s: &Scene, tuning: &Tuning) -> RgbImage {
    let inputs = DemosaicInputs {
        gains: s.gains,
        noise: s.noise,
        ca: None,
    };
    let p = Params {
        shadow_match: s.shadow,
        tuning: *tuning,
        ..Params::default()
    };
    adaptive_ratio::demosaic(&s.mosaic, &inputs, &p).expect("demosaic")
}

/// The groups and splits whose mean PSNR is reported: `(key, group, split)` with the split `None` for
/// all, `Some(false)` for the fit sets and `Some(true)` for the held-out sets.
const PSNR_KEYS: [(&str, Group, Option<bool>); 8] = [
    ("A", Group::A, None),
    ("A_fit", Group::A, Some(false)),
    ("A_held", Group::A, Some(true)),
    ("B", Group::B, None),
    ("B_fit", Group::B, Some(false)),
    ("B_held", Group::B, Some(true)),
    ("B2", Group::B2, None),
    ("P", Group::P, None),
];

pub fn psnr_means(scenes: &[Scene], outs: &[RgbImage]) -> Vec<(&'static str, f64)> {
    PSNR_KEYS
        .iter()
        .map(|&(key, group, split)| {
            let v: Vec<f64> = scenes
                .iter()
                .zip(outs)
                .filter(|(s, _)| {
                    s.group == group && split.is_none_or(|held| s.source.held_out() == held)
                })
                .filter_map(|(s, o)| s.truth.as_ref().map(|t| psnr(o, t, 10, s.scale)))
                .collect();
            (key, mean(&v))
        })
        .collect()
}

/// Zipper rate, false-colour rate and mean chroma difference of the A scenes (the gate's definitions).
fn artifact_means(scenes: &[Scene], outs: &[RgbImage], truth_labs: &[Option<LabImage>]) -> Value {
    let rows: Vec<_> = scenes
        .iter()
        .zip(outs)
        .zip(truth_labs)
        .filter_map(|((s, o), t)| {
            let t = t.as_ref()?;
            (s.group == Group::A).then(|| artifacts(t, &LabImage::new(o), BORDER_PX))
        })
        .collect();
    json!({
        "images": rows.len(),
        "zipper_pct": num(mean_finite(rows.iter().map(|r| r.zipper_pct))),
        "false_colour_pct": num(mean_finite(rows.iter().map(|r| r.false_colour_pct))),
        "chroma_de00_mean": num(mean_finite(rows.iter().map(|r| r.chroma_de00_mean))),
    })
}

const EDGE_SIZE: usize = 192;
const EDGE_LOW: f32 = 0.05;
const EDGE_HIGH: f32 = 0.25;

/// A slanted edge (normal at 5 degrees from the x axis) between two neutral levels, 8 x 8 supersampled.
fn slanted_edge(size: usize) -> (RgbImage, (f64, f64), (f64, f64)) {
    let theta = 5.0f64.to_radians();
    let n = (theta.cos(), theta.sin());
    let c = (size as f64 / 2.0, size as f64 / 2.0);
    let mut img = RgbImage::new(size, size);
    let ss = 8;
    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.0f64;
            for sy in 0..ss {
                for sx in 0..ss {
                    let px = x as f64 + (f64::from(sx) + 0.5) / f64::from(ss);
                    let py = y as f64 + (f64::from(sy) + 0.5) / f64::from(ss);
                    let d = n.0 * (px - c.0) + n.1 * (py - c.1);
                    acc += f64::from(if d > 0.0 { EDGE_HIGH } else { EDGE_LOW });
                }
            }
            let v = (acc / f64::from(ss * ss)) as f32;
            img.set_pixel(x, y, [v, v, v]);
        }
    }
    (img, n, c)
}

fn luma_spec(img: &RgbImage) -> Vec<f64> {
    let n = img.plane_len();
    (0..n)
        .map(|i| {
            f64::from(SPEC_LUMA[0]) * f64::from(img.data[i])
                + f64::from(SPEC_LUMA[1]) * f64::from(img.data[n + i])
                + f64::from(SPEC_LUMA[2]) * f64::from(img.data[2 * n + i])
        })
        .collect()
}

/// The two flat 40 x 40 windows of the edge scene, 48 px or more from the edge.
fn flat_windows(size: usize) -> [(usize, usize, usize, usize); 2] {
    [(8, 76, 40, 40), (size - 48, 76, 40, 40)]
}

pub struct EdgeScene {
    pub scene: Scene,
    pub geometry: EdgeGeometry,
}

/// The slanted edge with the noise of seed index `k` (`k = 0` is the seed of RC-05 to RC-07).
pub fn edge_scene(k: u64) -> EdgeScene {
    let (img, normal, centre) = slanted_edge(EDGE_SIZE);
    let high = model(5e-4, 4e-5, 1e-3);
    let mut m = synth::mosaic_from_rgb(&img, Cfa::RGGB);
    add_noise(&mut m, &high, 4242 + 1000 * k);
    EdgeScene {
        scene: Scene {
            group: Group::B,
            source: Source::Synthetic,
            truth: Some(img),
            mosaic: m,
            gains: [1.0; 3],
            noise: NoiseDn::from_model(&high),
            scale: 1.0,
            shadow: ShadowMatch::Auto,
        },
        geometry: EdgeGeometry { normal, centre },
    }
}

/// The edge numbers, the mean over the seeds of `edges` (one seed: the numbers of that scene).
pub fn edge_numbers(edges: &[EdgeScene], tuning: &Tuning) -> Value {
    let mut acc = [0.0f64; 5];
    for e in edges {
        let out = run_scene(&e.scene, tuning);
        let luma = luma_spec(&out);
        let es = edge_stats(&luma, EDGE_SIZE, EDGE_SIZE, &e.geometry, 32);
        let flat_sd = patch_sd(&luma, EDGE_SIZE, &flat_windows(EDGE_SIZE));
        for (a, v) in acc.iter_mut().zip([
            es.mtf50,
            es.rise,
            es.overshoot_pct,
            es.undershoot_pct,
            flat_sd,
        ]) {
            *a += v;
        }
    }
    let n = edges.len() as f64;
    json!({
        "mtf50_cpp": num(acc[0] / n), "rise_10_90_px": num(acc[1] / n),
        "overshoot_pct": num(acc[2] / n), "undershoot_pct": num(acc[3] / n),
        "flat_sd": num(acc[4] / n),
        "levels": [EDGE_LOW, EDGE_HIGH],
        "seeds": edges.len(),
    })
}

fn ca_numbers(scenes: &[ca_scenes::CaScene], noise: NoiseDn, tuning: &Tuning) -> Value {
    Value::Array(
        scenes
            .iter()
            .map(|s| {
                let (r, out) = ca_scenes::ca_path_result(s, noise, tuning);
                let psnr_g = psnr(
                    &ca_scenes::grey_of(&out, 1),
                    &ca_scenes::grey_of(&s.truth, 1),
                    16,
                    1.0,
                );
                json!({
                    "strength": s.strength, "barrel": s.distortion,
                    "red_rms_px": num(r.red_rms_px), "blue_rms_px": num(r.blue_rms_px),
                    "psnr_rgb_db": num(r.psnr_rgb_db), "psnr_g_db": num(psnr_g),
                })
            })
            .collect(),
    )
}

/// `(mean, max)` change of `outs` against `base` per group, in codes.
fn change(scenes: &[Scene], outs: &[RgbImage], base: &[RgbImage]) -> Value {
    let mut m = serde_json::Map::new();
    for group in [Group::A, Group::B, Group::B2, Group::C, Group::P] {
        let (mut sm, mut n, mut mx) = (0.0, 0, 0.0f64);
        for ((s, o), b) in scenes.iter().zip(outs).zip(base) {
            if s.group == group {
                let (mean_d, max_d) = code_diff(o, b, 10, s.scale);
                sm += mean_d;
                n += 1;
                mx = mx.max(max_d);
            }
        }
        if n > 0 {
            m.insert(
                group.key().to_owned(),
                json!({"mean_codes": sm / f64::from(n), "max_codes": mx}),
            );
        }
    }
    Value::Object(m)
}

/// The nine cells of `demosaic_gate` (procedural, Kodak, McMaster at scales 1, 0.1 and 0.01): the mean PSNR
/// of each set at each scale and the mean of the cells (equal weight), all, fit sets only and held out only.
fn cells_numbers(scenes: &[Scene], outs: &[RgbImage]) -> Value {
    let sets = [
        ("synthetic", Source::Synthetic),
        ("kodak", Source::Kodak),
        ("mcmaster", Source::McMaster),
    ];
    let scales = [
        ("1", Group::A),
        ("0.1", Group::Cell01),
        ("0.01", Group::Cell001),
    ];
    let mut cells = serde_json::Map::new();
    let (mut all, mut fit_c, mut held_c) = (Vec::new(), Vec::new(), Vec::new());
    for (sname, source) in sets {
        for (cname, group) in scales {
            let v: Vec<f64> = scenes
                .iter()
                .zip(outs)
                .filter(|(s, _)| s.group == group && s.source == source)
                .filter_map(|(s, o)| s.truth.as_ref().map(|t| psnr(o, t, 10, s.scale)))
                .collect();
            if v.is_empty() {
                continue;
            }
            let m = mean(&v);
            cells.insert(format!("{sname}@{cname}"), num(m));
            all.push(m);
            if source.held_out() {
                held_c.push(m);
            } else {
                fit_c.push(m);
            }
        }
    }
    json!({
        "cells": cells,
        "mean9": num(mean(&all)), "mean_fit": num(mean(&fit_c)), "mean_held": num(mean(&held_c)),
        "n": all.len(),
    })
}

// ---------------------------------------------------------------------------------------------
// The noisy table and the ladder (RC-08)
// ---------------------------------------------------------------------------------------------

/// One set of ground-truth images for the noisy table: the procedural set, Kodak, McMaster, in the order
/// of `demosaic_gate` (the noise of image `k` of a set is seeded by the level and `k`).
pub struct TruthSet {
    pub name: &'static str,
    pub source: Source,
    pub images: Vec<RgbImage>,
}

pub fn truth_sets(root: &Path) -> Vec<TruthSet> {
    let mut out = vec![TruthSet {
        name: "synthetic",
        source: Source::Synthetic,
        images: synthetic_set().into_iter().map(|(_, i)| i).collect(),
    }];
    for (dir, name, source) in [
        ("kodak", "kodak", Source::Kodak),
        ("mcmaster", "mcmaster", Source::McMaster),
    ] {
        let images = load_set(&root.join(dir));
        if !images.is_empty() {
            out.push(TruthSet {
                name,
                source,
                images: images.iter().map(|(_, i)| linearize(i)).collect(),
            });
        }
    }
    out
}

/// A noise level: `name`, `a`, `b` of the variance `a x + b`, and the seed base of the level.
pub struct Level {
    pub name: String,
    pub a: f64,
    pub b: f64,
    pub seed_base: u64,
}

/// The four ISO-like levels of `demosaic_gate` (same values and seeds).
pub fn gate_levels() -> Vec<Level> {
    [
        ("ISO 100-like", 2.5e-5, 4e-8, 1000),
        ("ISO 800-like", 2e-4, 3e-7, 2000),
        ("ISO 1600-like", 4e-4, 6e-7, 4000),
        ("ISO 6400-like", 2e-3, 2e-6, 3000),
    ]
    .into_iter()
    .map(|(n, a, b, s)| Level {
        name: n.to_owned(),
        a,
        b,
        seed_base: s,
    })
    .collect()
}

/// The ladder: eight noise models with the SNR at 18 % grey from 40 dB to 15 dB in equal steps, `b = 1.5e-3 a`
/// (the ratio of the gate's ISO 800 and 1600 levels), `a` from `var18 = 0.18^2 / 10^(snr/10) = a (0.18 + b/a)`.
pub fn ladder_levels() -> Vec<(f64, Level)> {
    (0..8)
        .map(|k| {
            let snr_db = 40.0 - 25.0 * k as f64 / 7.0;
            let var18 = 0.18f64 * 0.18 / 10f64.powf(snr_db / 10.0);
            let a = var18 / (0.18 + 1.5e-3);
            (
                snr_db,
                Level {
                    name: format!("{snr_db:.1} dB"),
                    a,
                    b: 1.5e-3 * a,
                    seed_base: 5000 + 1000 * k,
                },
            )
        })
        .collect()
}

/// Gaussian noise of variance `a x + b` (x clamped at 0) on every photosite, as `demosaic_gate` does.
fn add_gate_noise(data: &mut [f32], a: f64, b: f64, seed: u64) {
    let mut rng = synth::Rng::new(seed);
    for v in data {
        let var = a * f64::from(v.max(0.0)) + b;
        *v += (var.sqrt() as f32) * rng.gaussian();
    }
}

/// The mean PSNR over the images of each set at each level: `(demosaic, after R6)` per `[set][level]`.
pub fn noisy_cells(
    sets: &[TruthSet],
    levels: &[Level],
    tuning: &Tuning,
    shadow: ShadowMatch,
) -> Vec<Vec<(f64, f64)>> {
    sets.iter()
        .map(|set| {
            levels
                .iter()
                .map(|level| {
                    let model = NoiseModel {
                        a: level.a,
                        b: level.b,
                        delta: [0.0; 4],
                        sigma_row: 0.0,
                        q: 1.0 / 65535.0,
                    };
                    let noise = NoiseDn::from_model(&model);
                    let sigma_grey = (level.a * 0.18 + level.b).sqrt() as f32;
                    let (mut raw, mut den) = (0.0f64, 0.0f64);
                    for (k, img) in set.images.iter().enumerate() {
                        let mut m = synth::mosaic_from_rgb(img, Cfa::RGGB);
                        add_gate_noise(&mut m.data, level.a, level.b, level.seed_base + k as u64);
                        let inputs = DemosaicInputs {
                            gains: [1.0; 3],
                            noise,
                            ca: None,
                        };
                        let p = Params {
                            shadow_match: shadow,
                            tuning: *tuning,
                            ..Params::default()
                        };
                        let out = adaptive_ratio::demosaic(&m, &inputs, &p).expect("demosaic");
                        raw += psnr(&out, img, BORDER_PX, 1.0);
                        let d = denoise::bilateral_opponent(
                            &out,
                            &NoiseSigma::Scalar(sigma_grey),
                            &BilateralParams::default(),
                        );
                        den += psnr(&d, img, BORDER_PX, 1.0);
                    }
                    let n = set.images.len() as f64;
                    (raw / n, den / n)
                })
                .collect()
        })
        .collect()
}

/// The JSON of a noisy table: per level and set the two stages, and the means over the cells (equal weight):
/// all, fit sets (procedural and Kodak) and held out (McMaster), for each stage and for both.
pub fn noisy_json(sets: &[TruthSet], levels: &[Level], cells: &[Vec<(f64, f64)>]) -> Value {
    let mut by_level = serde_json::Map::new();
    let (mut all, mut fit_c, mut held_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut all_r, mut fit_r, mut held_r) = (Vec::new(), Vec::new(), Vec::new());
    let (mut all_d, mut fit_d, mut held_d) = (Vec::new(), Vec::new(), Vec::new());
    for (li, level) in levels.iter().enumerate() {
        let mut one = serde_json::Map::new();
        for (si, set) in sets.iter().enumerate() {
            let (d, r) = cells[si][li];
            one.insert(
                set.name.to_owned(),
                json!({"demosaic_db": num(d), "r6_db": num(r)}),
            );
            for (v, bucket) in [(d, 0usize), (r, 1)] {
                let held = set.source.held_out();
                let (a, f, h) = match bucket {
                    0 => (&mut all_d, &mut fit_d, &mut held_d),
                    _ => (&mut all_r, &mut fit_r, &mut held_r),
                };
                a.push(v);
                if held {
                    h.push(v);
                } else {
                    f.push(v);
                }
                all.push(v);
                if held {
                    held_c.push(v);
                } else {
                    fit_c.push(v);
                }
            }
        }
        by_level.insert(level.name.clone(), Value::Object(one));
    }
    json!({
        "levels": by_level,
        "mean": num(mean(&all)), "mean_fit": num(mean(&fit_c)), "mean_held": num(mean(&held_c)),
        "mean_demosaic": num(mean(&all_d)), "mean_fit_demosaic": num(mean(&fit_d)),
        "mean_held_demosaic": num(mean(&held_d)),
        "mean_r6": num(mean(&all_r)), "mean_fit_r6": num(mean(&fit_r)),
        "mean_held_r6": num(mean(&held_r)),
    })
}

/// The ladder at `tuning`: the PSNR with shadow matching on and off at each rung, fit and held-out.
pub fn ladder_json(sets: &[TruthSet], tuning: &Tuning) -> Value {
    let rungs = ladder_levels();
    let levels: Vec<Level> = rungs
        .iter()
        .map(|(_, l)| Level {
            name: l.name.clone(),
            a: l.a,
            b: l.b,
            seed_base: l.seed_base,
        })
        .collect();
    let on = noisy_cells(sets, &levels, tuning, ShadowMatch::On);
    let off = noisy_cells(sets, &levels, tuning, ShadowMatch::Off);
    let rows: Vec<Value> = rungs
        .iter()
        .enumerate()
        .map(|(li, (snr_db, l))| {
            let pick = |cells: &[Vec<(f64, f64)>], held: bool, stage: usize| -> f64 {
                let v: Vec<f64> = sets
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.source.held_out() == held)
                    .map(|(si, _)| {
                        let (d, r) = cells[si][li];
                        if stage == 0 { d } else { r }
                    })
                    .collect();
                mean(&v)
            };
            let mut row = json!({"snr18_db": num(*snr_db), "a": l.a, "b": l.b});
            for (key, held) in [("fit", false), ("held", true)] {
                for (stage, stage_key) in [(0usize, "demosaic"), (1, "r6")] {
                    let (o, f) = (pick(&on, held, stage), pick(&off, held, stage));
                    row[format!("{key}_{stage_key}")] =
                        json!({"on_db": num(o), "off_db": num(f), "gain_db": num(o - f)});
                }
            }
            row
        })
        .collect();
    json!({"rungs": rows, "sets": sets.iter().map(|s| (s.name, s.images.len())).collect::<std::collections::BTreeMap<_, _>>()})
}

fn usage(why: &str) -> ! {
    eprintln!("fit_demosaic: {why}");
    eprintln!(
        "usage: fit_demosaic [--from oracle|default] [--set name=value]... [--rel name=f1,f2]... \
         [--eval \"label@name=v,name=v\"]... [--held] [--photo FILE.ARW]... [--artifacts] [--edge] \
         [--cells] [--skip-a] [--edge-seeds N] [--noisy [on|auto|off]] [--ladder] \
         [--ca | --ca-only [--ca-size WxH]] [--list]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut common = Common::default();
    let mut photos: Vec<PathBuf> = Vec::new();
    let (mut want_artifacts, mut want_edge, mut want_ca, mut ca_only) =
        (false, false, false, false);
    let mut want_cells = false;
    let (mut skip_a, mut want_ladder) = (false, false);
    let mut edge_seeds = 1u64;
    let mut noisy: Option<ShadowMatch> = None;
    let mut ca_size = (600usize, 400usize);
    let mut i = 0;
    while i < args.len() {
        match common.take(&args, &mut i) {
            Ok(true) => {}
            Ok(false) => match args[i].as_str() {
                "--photo" => {
                    i += 1;
                    photos.push(PathBuf::from(
                        args.get(i).unwrap_or_else(|| usage("--photo needs a file")),
                    ));
                }
                "--artifacts" => want_artifacts = true,
                "--edge" => want_edge = true,
                "--cells" => want_cells = true,
                "--skip-a" => skip_a = true,
                "--ladder" => want_ladder = true,
                "--edge-seeds" => {
                    i += 1;
                    edge_seeds = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .filter(|n| *n >= 1)
                        .unwrap_or_else(|| usage("--edge-seeds needs a count of 1 or more"));
                }
                "--noisy" => {
                    // An optional mode: `on` (default), `auto` or `off`.
                    let mode = match args.get(i + 1).map(String::as_str) {
                        Some("on") => Some(ShadowMatch::On),
                        Some("auto") => Some(ShadowMatch::Auto),
                        Some("off") => Some(ShadowMatch::Off),
                        _ => None,
                    };
                    if mode.is_some() {
                        i += 1;
                    }
                    noisy = Some(mode.unwrap_or(ShadowMatch::On));
                }
                "--ca" => want_ca = true,
                "--ca-only" => {
                    want_ca = true;
                    ca_only = true;
                }
                "--ca-size" => {
                    i += 1;
                    let v = args.get(i).unwrap_or_else(|| usage("--ca-size needs WxH"));
                    let (a, b) = v
                        .split_once('x')
                        .unwrap_or_else(|| usage("--ca-size needs WxH"));
                    ca_size = (
                        a.parse().unwrap_or_else(|_| usage("bad width")),
                        b.parse().unwrap_or_else(|_| usage("bad height")),
                    );
                }
                other => usage(&format!("unknown argument `{other}`")),
            },
            Err(e) => usage(&e),
        }
        i += 1;
    }
    let start = match common.start {
        Start::Oracle => Tuning::ORACLE,
        Start::Default => Tuning::DEFAULT,
    };
    if common.list {
        emit(&fit::list_line("adaptive_ratio", &start));
        return;
    }
    let evals = common.evaluations(&start).unwrap_or_else(|e| usage(&e));

    let loaded = if ca_only {
        Loaded {
            scenes: Vec::new(),
            used: Vec::new(),
            skipped: Vec::new(),
        }
    } else {
        load_scenes(
            &demosaic_sets_root(),
            common.held,
            &photos,
            want_cells,
            skip_a,
        )
    };
    let ca_set = want_ca.then(|| ca_scenes::ca_scene_set(ca_size, 0.0, 2024));
    let ca_noise = NoiseDn::from_model(&ca_scenes::ca_noise_model(0.0));
    let edge: Option<Vec<EdgeScene>> = want_edge.then(|| (0..edge_seeds).map(edge_scene).collect());
    let truth = (noisy.is_some() || want_ladder).then(|| truth_sets(&demosaic_sets_root()));
    let gate_lv = gate_levels();
    let mut used = loaded.used.clone();
    if want_ca {
        used.push((
            format!("CA scenes {}x{} (ours, seed 2024)", ca_size.0, ca_size.1),
            5,
        ));
    }
    emit(&data_note("fit_demosaic", &used, &loaded.skipped));

    let scenes = &loaded.scenes;
    let truth_labs: Vec<Option<LabImage>> = if want_artifacts {
        scenes
            .iter()
            .map(|s| {
                (s.group == Group::A)
                    .then(|| s.truth.as_ref().map(LabImage::new))
                    .flatten()
            })
            .collect()
    } else {
        Vec::new()
    };
    let base_outs: Vec<RgbImage> = scenes.iter().map(|s| run_scene(s, &start)).collect();
    let base_psnr = psnr_means(scenes, &base_outs);

    let report = |label: &str,
                  sets: &[(String, f64)],
                  t: &Tuning,
                  outs: &[RgbImage],
                  is_base: bool| {
        let p = psnr_means(scenes, outs);
        let mut line = json!({
            "kind": "eval", "example": "fit_demosaic", "label": label,
            "from": if common.start == Start::Oracle { "oracle" } else { "default" },
            "set": sets.iter().map(|(n, v)| (n.clone(), *v)).collect::<std::collections::BTreeMap<_, _>>(),
            "psnr_db": p.iter().map(|(k, v)| ((*k).to_owned(), num(*v))).collect::<std::collections::BTreeMap<_, _>>(),
        });
        if !is_base {
            line["d_psnr_db"] = json!(
                p.iter()
                    .zip(&base_psnr)
                    .map(|((k, v), (_, b))| ((*k).to_owned(), num(v - b)))
                    .collect::<std::collections::BTreeMap<_, _>>()
            );
            line["change_codes"] = change(scenes, outs, &base_outs);
        }
        if want_cells {
            let mut c = cells_numbers(scenes, outs);
            if !is_base {
                let b = cells_numbers(scenes, &base_outs);
                for k in ["mean9", "mean_fit", "mean_held"] {
                    if let (Some(x), Some(y)) = (c[k].as_f64(), b[k].as_f64()) {
                        c[format!("d_{k}")] = num(x - y);
                    }
                }
            }
            line["cells"] = c;
        }
        if want_artifacts {
            line["artifacts"] = artifact_means(scenes, outs, &truth_labs);
        }
        if let Some(e) = &edge {
            line["edge"] = edge_numbers(e, t);
        }
        if let Some(c) = &ca_set {
            line["ca"] = ca_numbers(c, ca_noise, t);
        }
        if let (Some(mode), Some(sets)) = (noisy, &truth) {
            let cells = noisy_cells(sets, &gate_lv, t, mode);
            line["noisy"] = noisy_json(sets, &gate_lv, &cells);
            line["noisy"]["shadow"] = json!(format!("{mode:?}"));
        }
        if let (true, Some(sets)) = (want_ladder, &truth) {
            line["ladder"] = ladder_json(sets, t);
        }
        emit(&line);
    };
    report("base", &[], &start, &base_outs, true);
    for (e, t) in &evals {
        let outs: Vec<RgbImage> = scenes.iter().map(|s| run_scene(s, t)).collect();
        report(&e.label, &e.sets, t, &outs, false);
    }
}

#[allow(dead_code)]
fn field_names() -> &'static [&'static str] {
    <Tuning as Fields>::NAMES
}
