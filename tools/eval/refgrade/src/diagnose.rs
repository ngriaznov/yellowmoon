//! The diagnosis of a fit that differs from round 15's (`docs/34` section 16.10): the proof of concept's solution of a (photo,
//! reference) pair and the product's, both scored under **the product's objective**, term by term.
//!
//! It answers: given the proof of concept's numbers, does the product's objective score them lower than its own fit's? If so the
//! difference is the **search** (the product's search stopped at a worse place of its own objective). If the product's objective scores
//! the proof of concept's numbers *higher*, the search found the better place of this objective, and the difference is in the
//! **objective** (or in what it measures: the render, the sample, the measures), not in the search.
//!
//! The input is a JSON array of the proof of concept's records of the pairs, each with `photo`, `reference`, `x` (its 49 numbers) and
//! `objective` (its own total). The scene is the proof of concept's (variant A1), so the scene is not what differs.

use std::fs;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use ym_color::refgrade::N_PARAMS;
use ym_color::refgrade::measure::Px;
use ym_color::refgrade::network::Network;
use ym_color::refgrade::objective::Fixed;
use ym_color::refgrade::params::Params;
use ym_color::refgrade::search::{CandidateRenderer, FitSetup, score};
use ym_core::pipeline::PhotoSource;
use ym_core::pipeline::eval_scene::{EvalLabels, PhotoScene};
use ym_core::reference_fit::ChainCandidates;
use ym_core::session::PhotoId;
use ym_core::settings::ReferenceGradeNode;
use ym_gpu::queue::CancelToken;

use crate::measure::{Pic, SceneRef, sample_ab, transfers};
use crate::run::{NODE, add_node_patch, open_core, open_photo, read_label_png, reference_labels};
use crate::run::{RunOpts, selected};
use crate::split::reference_files;

/// What a diagnosis reads.
pub struct DiagOpts {
    /// The run's options (the split, the references, the data folder, the folder of the references' label maps); `only` names the photos.
    pub run: RunOpts,
    /// The proof of concept's records of the pairs.
    pub solutions: PathBuf,
}

fn numbers(v: &Value) -> Result<Params> {
    let a = v
        .as_array()
        .ok_or_else(|| anyhow!("the record has no `x`"))?;
    if a.len() != N_PARAMS {
        return Err(anyhow!("`x` has {} numbers, not {N_PARAMS}", a.len()));
    }
    let mut x = [0.0; N_PARAMS];
    for (d, s) in x.iter_mut().zip(a) {
        *d = s.as_f64().ok_or_else(|| anyhow!("`x` has a non-number"))?;
    }
    Ok(x)
}

fn terms_json(t: &ym_color::refgrade::objective::Terms) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in t.named() {
        m.insert(k.into(), json!(v));
    }
    Value::Object(m)
}

/// Runs the diagnosis and writes `diagnose.json` into `opts.run.out`.
///
/// # Errors
///
/// An input that cannot be read, a core that will not open, a fit that fails.
pub fn diagnose(opts: &DiagOpts) -> Result<()> {
    let list: Vec<Value> = serde_json::from_str(&fs::read_to_string(&opts.solutions)?)?;
    fs::create_dir_all(&opts.run.out)?;
    let refs = reference_files(&opts.run.references)?;
    let photos = selected(&opts.run)?;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.run.threads.max(1))
        .build_global();
    let app = tempfile::tempdir()?;
    let core = open_core(&opts.run, app.path())?;
    let net = Network::shipped().map_err(|e| anyhow!("{e}"))?;
    let cancel = CancelToken::new();
    let mut out: Vec<Value> = Vec::new();
    for entry in &photos {
        let name = entry.name();
        let mine: Vec<&Value> = list
            .iter()
            .filter(|r| r["photo"].as_str() == Some(name.as_str()))
            .collect();
        if mine.is_empty() {
            continue;
        }
        eprintln!("== {name}");
        let id: PhotoId = open_photo(&core, &entry.raw)?;
        let result = (|| -> Result<()> {
            let now = core.session.view(id)?.settings;
            core.update_settings(id, &add_node_patch(&now)?)?;
            let file = entry
                .label_file()
                .filter(|f| f.is_file())
                .ok_or_else(|| anyhow!("{name}: no label map"))?;
            let (width, height, elements) = read_label_png(&file)?;
            core.renderer
                .set_eval_photo_scene(Some(PhotoScene::Labels(EvalLabels {
                    width,
                    height,
                    elements,
                    crop: [0; 4],
                })));
            for rec in mine {
                let rname = rec["reference"].as_str().unwrap_or("");
                let still = refs
                    .iter()
                    .find(|p| p.file_stem().is_some_and(|s| s.to_string_lossy() == rname))
                    .ok_or_else(|| anyhow!("{rname}: no such reference"))?;
                let dir = opts
                    .run
                    .reference_labels
                    .as_deref()
                    .ok_or_else(|| anyhow!("--reference-labels is needed"))?;
                core.renderer
                    .set_eval_reference_labels(Some(reference_labels(dir, still)?));
                core.reference_choose_file(id, NODE, still, Some(rname))
                    .map_err(|e| anyhow!("{e}"))?;
                // The product's own fit on the proof of concept's scene.
                let fit = core
                    .reference_fit(id, NODE, &cancel, &mut |_| {})
                    .map_err(|e| anyhow!("{e}"))?;
                core.update_settings(id, &fit.undo)?;
                // The objective as the fit built it: the sample, the photo's pixels, the reference's.
                let src =
                    PhotoSource::from_session(&core.session, id).map_err(|e| anyhow!("{e}"))?;
                let photo = core
                    .renderer
                    .reference_fit_photo(&src, NODE, &cancel)
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
                let mut cands = ChainCandidates::new(&photo, &base);
                let mut p_px = vec![Px::default(); cands.sample_len()];
                cands
                    .render(&[0.0; N_PARAMS], &mut p_px)
                    .map_err(|_| anyhow!("the photo could not be rendered"))?;
                let fixed = Fixed::new(
                    &net,
                    &p_px,
                    &photo.scene,
                    &refm.px,
                    &refm.scene,
                    base.skin_protection,
                )
                .map_err(|e| anyhow!("{e}"))?;
                let setup = FitSetup {
                    fixed: &fixed,
                    scene: &photo.scene,
                    mask_ok: photo.mask_ok,
                };
                let x_poc = numbers(&rec["x"])?;
                let x_prod = fit.result.x;
                let (o_poc, t_poc) =
                    score(&setup, &mut cands, &x_poc).map_err(|e| anyhow!("{e}"))?;
                let (o_prod, t_prod) =
                    score(&setup, &mut cands, &x_prod).map_err(|e| anyhow!("{e}"))?;
                let (o_zero, _) =
                    score(&setup, &mut cands, &[0.0; N_PARAMS]).map_err(|e| anyhow!("{e}"))?;
                // The same numbers rendered on every pixel and measured as the tool measures a fit: do they give the transfer the proof of
                // concept recorded for them (T1 to T3 of the same pair in its records)?
                let eval = core
                    .reference_eval_photo(id, NODE, &cancel)
                    .map_err(|e| anyhow!("{e}"))?;
                let mut ecand = ChainCandidates::new(&eval, &base);
                let n = ecand.sample_len();
                let render = |c: &mut ChainCandidates<'_>, x: &Params| -> Result<Vec<Px>> {
                    let mut px = vec![Px::default(); n];
                    c.render(x, &mut px)
                        .map_err(|_| anyhow!("render refused"))?;
                    Ok(px)
                };
                let p0 = render(&mut ecand, &[0.0; N_PARAMS])?;
                let (px_poc, px_prod) = (render(&mut ecand, &x_poc)?, render(&mut ecand, &x_prod)?);
                let p0_sample = sample_ab(&p0);
                let sc = SceneRef {
                    class: &eval.scene.class,
                    skin: &eval.scene.skin,
                };
                let p0p = Pic {
                    px: &p0,
                    scene: Some(sc),
                };
                let refp = Pic {
                    px: &refm.px,
                    scene: Some(SceneRef {
                        class: &refm.scene.class,
                        skin: &refm.scene.skin,
                    }),
                };
                let t_of = |px: &[Px]| {
                    let f = Pic {
                        px,
                        scene: p0p.scene,
                    };
                    transfers(&p0p, &f, &refp, &p0_sample)
                };
                let (tr_poc, tr_prod) = (t_of(&px_poc), t_of(&px_prod));
                let dist = x_poc
                    .iter()
                    .zip(&x_prod)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                eprintln!(
                    "   {rname}: product objective at the proof of concept's numbers {o_poc:.3}, at its own {o_prod:.3} (fit said {:.3}), identity {o_zero:.3}, own objective of the proof of concept {:.3}, start {:?}",
                    fit.result.total,
                    rec["objective"].as_f64().unwrap_or(f64::NAN),
                    fit.result.start,
                );
                out.push(json!({
                    "photo": name, "reference": rname,
                    "product_objective_at_poc_x": o_poc,
                    "product_objective_at_product_x": o_prod,
                    "product_fit_total": fit.result.total,
                    "product_objective_at_identity": o_zero,
                    "poc_own_objective": rec["objective"],
                    "poc_search_start": rec["search_start"],
                    "product_search_start": format!("{:?}", fit.result.start),
                    "distance_between_numbers": dist,
                    "terms_at_poc_x": terms_json(&t_poc),
                    "terms_at_product_x": terms_json(&t_prod),
                    "t_at_poc_x_by_product_render": tr_poc, "t_at_product_x": tr_prod,
                    "t_poc_recorded": [rec["T1"].clone(), rec["T2"].clone(), rec["T3"].clone()],
                    "x_poc": x_poc.to_vec(), "x_product": x_prod.to_vec(),
                }));
                core.reference_clear(id, NODE);
            }
            Ok(())
        })();
        core.renderer.set_eval_photo_scene(None);
        core.renderer.set_eval_reference_labels(None);
        core.remove_photos(&[id]);
        result?;
    }
    fs::write(
        opts.run.out.join("diagnose.json"),
        serde_json::to_string_pretty(&out)?,
    )?;
    Ok(())
}
