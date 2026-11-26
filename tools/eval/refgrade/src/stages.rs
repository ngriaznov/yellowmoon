//! The stage comparison of RG-10 (`docs/34` section 16.10): the same 49 numbers of a (photo, reference) pair, rendered by the proof of
//! concept and by the product, compared stage by stage.
//!
//! The proof of concept's side is dumped by a scratch build of its tool (display-linear `f32` pictures of the parts of its solution at
//! the analysis size, and of the reference). The product's side renders the same numbers on every pixel of the analysis picture through
//! the core's own chain. The parts are the node's parameter groups, so a difference that first shows in a group names the stage:
//! the identity (the photo's picture, the Output look, the display rendering and the encode), the tone and colour curves, the
//! saturation, the hue curves, the hue-versus-hue curve alone (the hue step and its luma restore), the element grades alone (the
//! masks), and everything. The scene is the proof of concept's (variant A1), so it is not what differs.

#![allow(clippy::needless_range_loop, clippy::single_range_in_vec_init)]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use serde_json::{Value, json};
use ym_color::display::ColorChain;
use ym_color::mask::NodeMasks;
use ym_color::refgrade::N_PARAMS;
use ym_color::refgrade::measure::{LabConv, Px};
use ym_color::refgrade::params::{Params, to_numbers};
use ym_core::pipeline::color::reference_node_params;
use ym_core::pipeline::eval_scene::{EvalLabels, PhotoScene};
use ym_core::pipeline::fit::FitPhoto;
use ym_core::reference_fit::apply_numbers;
use ym_core::settings::ReferenceGradeNode;
use ym_gpu::queue::CancelToken;

use crate::measure::{Pic, SceneRef, sample_ab, transfers};
use crate::run::{
    NODE, RunOpts, add_node_patch, open_core, open_photo, read_label_png, reference_labels,
};
use crate::split::{read_split, reference_files};

/// What a stage comparison reads.
pub struct StageOpts {
    /// The run's options (split, references, folder of the references' labels, data, threads, `out`).
    pub run: RunOpts,
    /// The proof of concept's dumps (`<photo>__<reference>__<part>.f32` and `__meta.json`).
    pub poc_dir: PathBuf,
    /// The pairs, `photo:reference`.
    pub pairs: Vec<String>,
    /// The proof of concept's records of the pairs (`T1` to `T3`), for the recorded transfer.
    pub solutions: PathBuf,
}

/// The parts of the numbers, by name (the proof of concept's dump uses the same names).
pub const PARTS: [(&str, &[std::ops::Range<usize>]); 9] = [
    ("ident", &[]),
    ("tone", &[0..8]),
    ("tone_colour", &[0..17]),
    ("saturation", &[0..18]),
    ("hue_sat", &[0..24]),
    ("hue", &[0..30]),
    ("hue_only", &[24..30]),
    ("locals_only", &[30..N_PARAMS]),
    ("full", &[0..N_PARAMS]),
];

fn srgb_decode(v: f64) -> f64 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The product's display codes of the numbers `x` on every pixel of the analysis picture.
fn render_codes(
    photo: &FitPhoto,
    base: &ReferenceGradeNode,
    x: &Params,
    bypass: bool,
) -> Result<Vec<[f64; 3]>> {
    let mut node = base.clone();
    node.bypass = bypass;
    node.strength = 100.0;
    apply_numbers(&mut node, &to_numbers(x));
    let params = reference_node_params(&node, NodeMasks::default()).map_err(|e| anyhow!("{e}"))?;
    let mut chain: ColorChain = photo.chain.clone();
    chain
        .replace_node(photo.node, &params)
        .map_err(|e| anyhow!("{e}"))?;
    let masks = &*photo.masks;
    Ok((0..photo.sample.len())
        .into_par_iter()
        .map(|i| {
            let (px, py) = photo.sample[i];
            let spot = photo.spots[i].with_raster(masks, px, py);
            chain
                .codes_from_node(&photo.states[i], Some(spot), photo.node)
                .unwrap_or([0.0; 3])
        })
        .collect())
}

/// Differences of two pictures of one size: code values (0 to 255 units) and CIELAB (CIE76), maximum, mean, 99th percentile, and
/// the mean signed difference in L\*, a\*, b\* (product minus proof of concept).
fn diff(
    product_codes: &[[f64; 3]],
    poc_linear: &[f32],
    (w, h): (usize, usize),
    conv: &LabConv,
) -> Value {
    let n = product_codes.len();
    let (mut cmax, mut csum) = (0.0f64, 0.0f64);
    let mut de: Vec<f64> = Vec::with_capacity(n);
    let mut sl = [0.0f64; 3];
    for (i, c) in product_codes.iter().enumerate() {
        let lin = [
            f64::from(poc_linear[3 * i]),
            f64::from(poc_linear[3 * i + 1]),
            f64::from(poc_linear[3 * i + 2]),
        ];
        for k in 0..3 {
            let d = (c[k].clamp(0.0, 1.0) - srgb_encode(lin[k])).abs() * 255.0;
            cmax = cmax.max(d);
            csum += d;
        }
        let a = conv.px_of_codes(*c).lab;
        let b = conv.px(lin).lab;
        for k in 0..3 {
            sl[k] += a[k] - b[k];
        }
        de.push(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt());
    }
    let nf = n as f64;
    de.sort_unstable_by(f64::total_cmp);
    // The same on block means of 16 by 16 pixels, in linear light: what is left when the pixel-to-pixel part (the noise and the
    // aliasing of a picture sampled at a point, against one reduced by an area average) is averaged away.
    let bs = 16usize;
    let (mut bsum, mut bmax, mut bn) = (0.0f64, 0.0f64, 0usize);
    let mut bl = [0.0f64; 3];
    for by in 0..h / bs {
        for bx in 0..w / bs {
            let (mut a, mut b) = ([0.0f64; 3], [0.0f64; 3]);
            for y in by * bs..(by + 1) * bs {
                for x in bx * bs..(bx + 1) * bs {
                    let i = y * w + x;
                    for k in 0..3 {
                        a[k] += srgb_decode(product_codes[i][k].clamp(0.0, 1.0));
                        b[k] += f64::from(poc_linear[3 * i + k]);
                    }
                }
            }
            let c = (bs * bs) as f64;
            let la = conv.px(a.map(|v| v / c)).lab;
            let lb = conv.px(b.map(|v| v / c)).lab;
            let d = ((la[0] - lb[0]).powi(2) + (la[1] - lb[1]).powi(2) + (la[2] - lb[2]).powi(2))
                .sqrt();
            bsum += d;
            bmax = bmax.max(d);
            bn += 1;
            for k in 0..3 {
                bl[k] += la[k] - lb[k];
            }
        }
    }
    let bnf = bn.max(1) as f64;
    json!({
        "block16_de76_mean": bsum / bnf, "block16_de76_max": bmax,
        "block16_mean_signed_lab": [bl[0] / bnf, bl[1] / bnf, bl[2] / bnf],
        "code_max": cmax, "code_mean": csum / (3.0 * nf),
        "de76_max": de.last().copied().unwrap_or(0.0), "de76_mean": de.iter().sum::<f64>() / nf,
        "de76_p99": de[((n as f64 * 0.99) as usize).min(n - 1)],
        "mean_signed_lab": [sl[0] / nf, sl[1] / nf, sl[2] / nf],
    })
}

/// The mean (a\*, b\*) and the share of the frame of the non-skin pixels with L\* 15 to 80: the cast of T3.
fn cast(px: &[Px], skin: &[bool]) -> Value {
    let (mut a, mut b, mut c) = (0.0, 0.0, 0usize);
    for (q, &s) in px.iter().zip(skin) {
        if !s && (15.0..=80.0).contains(&q.lab[0]) {
            a += q.lab[1];
            b += q.lab[2];
            c += 1;
        }
    }
    let n = c.max(1) as f64;
    json!({ "a": a / n, "b": b / n, "share": c as f64 / px.len().max(1) as f64 })
}

fn px_of_linear(conv: &LabConv, lin: &[f32]) -> Vec<Px> {
    lin.chunks_exact(3)
        .map(|c| conv.px([f64::from(c[0]), f64::from(c[1]), f64::from(c[2])]))
        .collect()
}

/// Runs the comparison and writes `stages.json` into `opts.run.out`.
///
/// # Errors
///
/// Inputs that cannot be read, a core that will not open, a render that fails.
pub fn stages(opts: &StageOpts) -> Result<()> {
    fs::create_dir_all(&opts.run.out)?;
    let split = read_split(&opts.run.split)?;
    let refs = reference_files(&opts.run.references)?;
    let sols: Vec<Value> = serde_json::from_str(&fs::read_to_string(&opts.solutions)?)?;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.run.threads.max(1))
        .build_global();
    let app = tempfile::tempdir()?;
    let core = open_core(&opts.run, app.path())?;
    let cancel = CancelToken::new();
    let conv = LabConv::new();
    let mut out: Vec<Value> = Vec::new();
    for pair in &opts.pairs {
        let (pname, rname) = pair
            .split_once(':')
            .ok_or_else(|| anyhow!("--pair takes photo:reference"))?;
        let entry = split
            .iter()
            .find(|e| e.name() == pname)
            .ok_or_else(|| anyhow!("{pname}: not in the split"))?;
        let still = refs
            .iter()
            .find(|p| p.file_stem().is_some_and(|s| s.to_string_lossy() == rname))
            .ok_or_else(|| anyhow!("{rname}: no such reference"))?;
        let tag = format!("{pname}__{rname}");
        let meta: Value = serde_json::from_str(&fs::read_to_string(
            opts.poc_dir.join(format!("{tag}__meta.json")),
        )?)?;
        let x_poc: Vec<f64> = meta["x"]
            .as_array()
            .ok_or_else(|| anyhow!("meta has no x"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        eprintln!("== {tag}");
        let id = open_photo(&core, &entry.raw)?;
        let one = (|| -> Result<Value> {
            let now = core.session.view(id)?.settings;
            fs::write(
                opts.run.out.join(format!("{tag}__settings_before.json")),
                serde_json::to_string_pretty(&now)?,
            )?;
            core.update_settings(id, &add_node_patch(&now)?)?;
            let file = entry
                .label_file()
                .filter(|f| f.is_file())
                .ok_or_else(|| anyhow!("{pname}: no label map"))?;
            let (width, height, elements) = read_label_png(&file)?;
            core.renderer
                .set_eval_photo_scene(Some(PhotoScene::Labels(EvalLabels {
                    width,
                    height,
                    elements,
                    crop: [0; 4],
                })));
            let dir = opts
                .run
                .reference_labels
                .as_deref()
                .ok_or_else(|| anyhow!("--reference-labels is needed"))?;
            core.renderer
                .set_eval_reference_labels(Some(reference_labels(dir, still)?));
            core.reference_choose_file(id, NODE, still, Some(rname))
                .map_err(|e| anyhow!("{e}"))?;
            let eval = core
                .reference_eval_photo(id, NODE, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            let refm = core
                .reference_measured(id, NODE, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            let base: ReferenceGradeNode = core
                .session
                .view(id)?
                .settings
                .nodes
                .reference_grade(NODE)
                .ok_or_else(|| anyhow!("the node is gone"))?
                .clone();
            let poc_size = (
                meta["photo_size"][0].as_u64().unwrap_or(0) as usize,
                meta["photo_size"][1].as_u64().unwrap_or(0) as usize,
            );
            let same = poc_size == (eval.width, eval.height);
            let mut parts = serde_json::Map::new();
            let mut product_px: std::collections::BTreeMap<&str, Vec<Px>> = Default::default();
            let mut poc_px: std::collections::BTreeMap<&str, Vec<Px>> = Default::default();
            for (name, ranges) in PARTS {
                let mut x: Params = [0.0; N_PARAMS];
                for r in ranges {
                    x[r.clone()].copy_from_slice(&x_poc[r.clone()]);
                }
                let codes = render_codes(&eval, &base, &x, false)?;
                let lin = read_f32(&opts.poc_dir.join(format!("{tag}__{name}.f32")))?;
                // The product's picture as display-linear f32, the proof of concept's format, for a closer look outside the tool.
                let mut bytes = Vec::with_capacity(codes.len() * 12);
                for c in &codes {
                    for v in c {
                        bytes.extend_from_slice(
                            &(srgb_decode(v.clamp(0.0, 1.0)) as f32).to_le_bytes(),
                        );
                    }
                }
                fs::write(
                    opts.run.out.join(format!("{tag}__{name}.product.f32")),
                    bytes,
                )?;
                if same {
                    parts.insert(
                        name.into(),
                        diff(&codes, &lin, (eval.width, eval.height), &conv),
                    );
                }
                product_px.insert(name, codes.iter().map(|&c| conv.px_of_codes(c)).collect());
                poc_px.insert(name, px_of_linear(&conv, &lin));
            }
            // The node at zero against the node bypassed: is the node at its identity the identity?
            let mean_lin = |codes: &[[f64; 3]]| -> Value {
                let n = codes.len() as f64;
                let mut m = [0.0f64; 3];
                for c in codes {
                    for k in 0..3 {
                        m[k] += srgb_decode(c[k].clamp(0.0, 1.0)) / n;
                    }
                }
                json!(m)
            };
            let zero: Params = [0.0; N_PARAMS];
            let identity_check = json!({
                "node_at_zero": mean_lin(&render_codes(&eval, &base, &zero, false)?),
                "node_bypassed": mean_lin(&render_codes(&eval, &base, &zero, true)?),
            });
            // The real pipeline's picture of the photo with the node in its graph (as the viewer shows it), at the same state.
            let src = ym_core::pipeline::PhotoSource::from_session(&core.session, id)
                .map_err(|e| anyhow!("{e}"))?;
            let real = core
                .renderer
                .advisor_picture(&src, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            let mut rm = [0.0f64; 3];
            for c in real.rgb.chunks_exact(3) {
                for k in 0..3 {
                    rm[k] += srgb_decode(f64::from(c[k]) / 255.0) / (real.rgb.len() / 3) as f64;
                }
            }
            let identity_check = {
                let mut v = identity_check;
                v["real_pipeline_with_node"] = json!(rm);
                v["real_size"] = json!([real.width, real.height]);
                v
            };
            // The reference's pixels: the proof of concept's loader against the product's.
            let ref_lin = read_f32(&opts.poc_dir.join(format!("{tag}__reference.f32")))?;
            let ref_size = (
                meta["reference_size"][0].as_u64().unwrap_or(0) as usize,
                meta["reference_size"][1].as_u64().unwrap_or(0) as usize,
            );
            let poc_ref = px_of_linear(&conv, &ref_lin);
            let ref_same = ref_size == (refm.width, refm.height);
            let ref_diff = if ref_same {
                let (mut de, mut mx) = (0.0f64, 0.0f64);
                let mut sl = [0.0f64; 3];
                for (a, b) in refm.px.iter().zip(&poc_ref) {
                    let d = ((a.lab[0] - b.lab[0]).powi(2)
                        + (a.lab[1] - b.lab[1]).powi(2)
                        + (a.lab[2] - b.lab[2]).powi(2))
                    .sqrt();
                    de += d;
                    mx = mx.max(d);
                    for k in 0..3 {
                        sl[k] += a.lab[k] - b.lab[k];
                    }
                }
                let n = refm.px.len() as f64;
                json!({ "de76_mean": de / n, "de76_max": mx, "mean_signed_lab": [sl[0]/n, sl[1]/n, sl[2]/n] })
            } else {
                Value::Null
            };
            // The transfer of the proof of concept's own pictures under the product's measures, and of the product's pictures.
            let skin = &eval.scene.skin;
            let sc = SceneRef {
                class: &eval.scene.class,
                skin,
            };
            let refscene = SceneRef {
                class: &refm.scene.class,
                skin: &refm.scene.skin,
            };
            let t_of = |p0: &[Px], fit: &[Px], rp: &[Px]| -> Value {
                let p0p = Pic {
                    px: p0,
                    scene: Some(sc),
                };
                let fitp = Pic {
                    px: fit,
                    scene: Some(sc),
                };
                let refp = Pic {
                    px: rp,
                    scene: Some(refscene),
                };
                json!(transfers(&p0p, &fitp, &refp, &sample_ab(p0)))
            };
            let mut t = serde_json::Map::new();
            if same && ref_same {
                t.insert(
                    "poc_pictures_poc_reference".into(),
                    t_of(&poc_px["ident"], &poc_px["full"], &poc_ref),
                );
                t.insert(
                    "poc_pictures_product_reference".into(),
                    t_of(&poc_px["ident"], &poc_px["full"], &refm.px),
                );
                t.insert(
                    "product_pictures_product_reference".into(),
                    t_of(&product_px["ident"], &product_px["full"], &refm.px),
                );
                t.insert(
                    "product_pictures_poc_reference".into(),
                    t_of(&product_px["ident"], &product_px["full"], &poc_ref),
                );
            }
            let recorded = sols
                .iter()
                .find(|r| r["photo"] == pname && r["reference"] == rname)
                .map(|r| json!([r["T1"], r["T2"], r["T3"]]));
            let mut casts = serde_json::Map::new();
            if same {
                for k in ["ident", "full"] {
                    casts.insert(format!("poc_{k}"), cast(&poc_px[k], skin));
                    casts.insert(format!("product_{k}"), cast(&product_px[k], skin));
                }
            }
            casts.insert(
                "poc_reference".into(),
                cast(
                    &poc_ref,
                    &if ref_same {
                        refm.scene.skin.clone()
                    } else {
                        vec![false; poc_ref.len()]
                    },
                ),
            );
            casts.insert("product_reference".into(), cast(&refm.px, &refm.scene.skin));
            Ok(json!({
                "pair": tag, "analysis_size_product": [eval.width, eval.height], "analysis_size_poc": [poc_size.0, poc_size.1],
                "reference_size_product": [refm.width, refm.height], "reference_size_poc": [ref_size.0, ref_size.1],
                "poc_bars_cut_by_poc": meta["bars"],
                "parts": parts, "identity_check": identity_check, "reference": ref_diff, "transfer": t, "poc_recorded_t": recorded, "casts": casts,
            }))
        })();
        core.renderer.set_eval_photo_scene(None);
        core.renderer.set_eval_reference_labels(None);
        core.reference_clear(id, NODE);
        core.remove_photos(&[id]);
        let v = one?;
        eprintln!("{}", serde_json::to_string(&v["parts"])?);
        out.push(v);
    }
    fs::write(
        opts.run.out.join("stages.json"),
        serde_json::to_string_pretty(&out)?,
    )?;
    Ok(())
}
