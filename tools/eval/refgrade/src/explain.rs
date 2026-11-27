//! `explain`: why a fit does harm (RG-13, the diagnosis stage; `docs/34` section 10.1).
//!
//! It re-runs the product's fit of chosen (photo, reference) pairs exactly as `run` does (same core, same node, same reference loader, so
//! the same numbers: a record of `run` and the pair's `fit` here agree to the bit) and writes, per pair, one JSON file with:
//!
//! * the fitted numbers, grouped as `docs/34` section 5.2 groups them, in the node's units too;
//! * the objective's terms at the solution, at the identity and at each of the three searches' results;
//! * the statistics behind **H2** (skin) and **H4** (whites) for the photo, the fit and the reference, measured on every pixel of the
//!   analysis picture as the harms are, and the same groups as the objective sees them on the fit's own 8192-pixel sample;
//! * which skin mask H2 reads and which the objective's skin walls read, and their overlap;
//! * **counterfactuals**: each group of numbers put back to the identity (`zero`), and each group alone (`only`); for each the harm values,
//!   the transfer T1 to T3 and the objective's terms, so the file says which group does the harm and whether the objective sees it;
//! * optionally (`--repolish`) the full objective after the harmful group is zeroed **and the other numbers searched again**, which says
//!   whether the objective prefers a harmless grade the search did not reach.
//!
//! Two more modes: `--facts-only` writes the photo-level facts of every dev photo (skin and whites as H2, H4 and the walls see them at the
//! identity), and `--whatif-from DIR --variants ...` searches the numbers of earlier `explain` files again under a **changed objective**
//! (the product's objective with a term taken out or a wall added, built here from the product's own [`Terms`] and the candidate's pixels;
//! the product's objective, search and walls are not touched). The what-ifs are measurements for a principle's proposal, not the principle.
//!
//! It only reads: nothing in `run`, the objective or the search is changed, and the photos are the evaluation set's (`set` = `dev`), never
//! the owner's: a pair that names another photo is refused.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value, json};
use ym_color::refgrade::measure::{PixelScene, Px, class};
use ym_color::refgrade::network::Network;
use ym_color::refgrade::objective::{Fixed, Terms, candidate_measure};
use ym_color::refgrade::params::{Numbers, Params, param_name, to_numbers};
use ym_color::refgrade::search::{CandidateRenderer, FitSetup, Start, score};
use ym_color::refgrade::{ELEMENT_NAMES, N_PARAMS};
use ym_core::pipeline::PhotoSource;
use ym_core::pipeline::fit::FitPhoto;
use ym_core::reference_fit::ChainCandidates;
use ym_core::session::PhotoId;
use ym_core::settings::ReferenceGradeNode;
use ym_gpu::queue::CancelToken;

use crate::measure::{HARMS, Pic, SceneRef, harms, oklab_hue, sample_ab, transfers, wrap_deg};
use crate::run::{NODE, RunOpts, add_node_patch, open_core, open_photo, selected};
use crate::split::{SplitEntry, reference_files};

/// What `explain` reads.
pub struct ExplainOpts {
    /// The run's options: the split, the references, the output folder, the shard (over the photos), the threads, the data folder.
    pub run: RunOpts,
    /// The pairs file: one `photo:reference` per line (blank lines and `#` comments are skipped).
    pub pairs: PathBuf,
    /// Also search again with each harm-clearing group at identity (about three minutes more per pair).
    pub repolish: bool,
    /// What-if searches (diagnostic only, the product's objective is not touched): the folder of an earlier `explain`'s files, whose
    /// fitted numbers are taken instead of fitting again, and the variants of the objective to search under (`no_skin_abs`,
    /// `skin_relative`, `whites_licence`). One `<photo>__<reference>.whatif.json` per pair.
    pub whatif: Option<(PathBuf, Vec<String>)>,
    /// Write only the photo-level facts of every dev photo (`facts/<photo>.json`: skin and whites as the harms and the walls see them
    /// at the identity), with no reference and no fit; the pairs file is not read.
    pub facts_only: bool,
}

/// The pairs of a pairs file, in file order.
///
/// # Errors
///
/// A line that is not `photo:reference`.
pub fn parse_pairs(text: &str) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (p, r) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("pairs line {}: expected photo:reference", n + 1))?;
        let (p, r) = (p.trim(), r.trim());
        if p.is_empty() || r.is_empty() {
            bail!("pairs line {}: expected photo:reference", n + 1);
        }
        out.push((p.to_string(), r.to_string()));
    }
    Ok(out)
}

/// A group of the fit's numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    /// The group's name in the file.
    pub name: &'static str,
    /// The indices of its numbers.
    pub idx: Vec<usize>,
}

/// The groups the counterfactuals put back to the identity: the spec's groups (`docs/34` section 5.2), the channel curves by channel and
/// by knot (shadows at -3 stops, mids at 0, highlights at +3: a white-balance move is the mid knots), each element's grade and the skin's
/// three numbers one by one.
#[must_use]
pub fn groups() -> Vec<Group> {
    let g = |name, idx: Vec<usize>| Group { name, idx };
    let mut v = vec![
        g("global_all", (0..30).collect()),
        g("local_all", (30..N_PARAMS).collect()),
        g("tone", (0..8).collect()),
        g("channels", (8..17).collect()),
        g("red", (8..11).collect()),
        g("green", (11..14).collect()),
        g("blue", (14..17).collect()),
        g("channel_low", vec![8, 11, 14]),
        g("channel_mid", vec![9, 12, 15]),
        g("channel_high", vec![10, 13, 16]),
        g("saturation", vec![17]),
        g("hue_vs_sat", (18..24).collect()),
        g("hue_vs_hue", (24..30).collect()),
        g("hue_curves", (18..30).collect()),
    ];
    let names = ["sky", "vegetation", "water", "ground", "built", "skin"];
    debug_assert_eq!(names, ELEMENT_NAMES);
    for (k, n) in names.into_iter().enumerate() {
        v.push(g(n, (30 + 3 * k..33 + 3 * k).collect()));
    }
    v.push(g("subject", vec![N_PARAMS - 1]));
    v.push(g("skin_exposure", vec![45]));
    v.push(g("skin_hue", vec![46]));
    v.push(g("skin_chroma", vec![47]));
    v
}

/// `x` with the group's numbers at 0.
#[must_use]
pub fn zeroed(x: &Params, g: &Group) -> Params {
    let mut y = *x;
    for &i in &g.idx {
        y[i] = 0.0;
    }
    y
}

/// `x` with every number outside the group at 0.
#[must_use]
pub fn only(x: &Params, g: &Group) -> Params {
    let mut y = [0.0; N_PARAMS];
    for &i in &g.idx {
        y[i] = x[i];
    }
    y
}

/// One group of pixels, measured as H2 and H4 measure it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stat {
    /// The number of pixels.
    pub n: usize,
    /// Their share of the picture.
    pub share: f64,
    /// The mean L\*.
    pub l: f64,
    /// The mean a\*.
    pub a: f64,
    /// The mean b\*.
    pub b: f64,
    /// The mean per-pixel C\*.
    pub c_mean: f64,
    /// The mean-vector C\* (the length of the mean (a\*, b\*)).
    pub c_vec: f64,
    /// The Oklab hue of the mean Oklab (a, b) vector (what H2 and the walls turn on).
    pub hue_ok: f64,
    /// The CIELAB hue of the mean (a\*, b\*) vector.
    pub hue_lab: f64,
}

impl Stat {
    fn json(&self) -> Value {
        json!({
            "n": self.n, "share": self.share, "L": self.l, "a": self.a, "b": self.b,
            "C_mean": self.c_mean, "C_vec": self.c_vec, "hue_oklab": self.hue_ok, "hue_lab": self.hue_lab,
        })
    }
}

/// The pixels `mask` selects (same index in `px`), or `None` when it selects none.
#[must_use]
pub fn stat(px: &[Px], mask: &[bool]) -> Option<Stat> {
    let (mut n, mut l, mut a, mut b, mut c, mut oa, mut ob) =
        (0usize, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (q, _) in px.iter().zip(mask).filter(|&(_, &m)| m) {
        n += 1;
        l += q.lab[0];
        a += q.lab[1];
        b += q.lab[2];
        c += q.lab[1].hypot(q.lab[2]);
        oa += q.ok[0];
        ob += q.ok[1];
    }
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let (a, b) = (a / nf, b / nf);
    Some(Stat {
        n,
        share: nf / px.len().max(1) as f64,
        l: l / nf,
        a,
        b,
        c_mean: c / nf,
        c_vec: a.hypot(b),
        hue_ok: oklab_hue([oa / nf, ob / nf]),
        hue_lab: b.atan2(a).to_degrees().rem_euclid(360.0),
    })
}

/// H2's two parts for a fit against the photo: the Oklab mean-vector hue move in degrees (only when the photo's mean-vector C\* is at
/// least 4) and the mean-vector chroma ratio, as `harms` computes them.
#[must_use]
pub fn skin_move(photo: &Stat, fit: &Stat) -> (Option<f64>, f64) {
    let dh = (photo.c_vec >= 4.0).then(|| wrap_deg(fit.hue_ok - photo.hue_ok));
    (dh, fit.c_vec / photo.c_vec.max(1e-9))
}

fn terms_json(t: &Terms) -> Value {
    let mut m = Map::new();
    for (k, v) in t.named() {
        m.insert(k.into(), json!(v));
    }
    Value::Object(m)
}

fn opt_stat(s: &Option<Stat>) -> Value {
    s.as_ref().map_or(Value::Null, Stat::json)
}

/// The node's numbers as JSON.
fn numbers_json(n: &Numbers) -> Value {
    json!({
        "luma": n.luma, "red": n.rgb[0], "green": n.rgb[1], "blue": n.rgb[2],
        "saturation_pct": n.saturation_pct, "hue_vs_sat": n.hue_vs_sat, "hue_vs_hue_deg": n.hue_vs_hue_deg,
        "elements": ELEMENT_NAMES.iter().zip(&n.elements).map(|(name, e)| json!({
            "element": name, "exposure_ev": e.exposure_ev, "hue_deg": e.hue_deg, "chroma": e.chroma,
        })).collect::<Vec<_>>(),
        "subject_ev": n.subject_ev,
    })
}

/// Everything one pair's evaluations share.
struct Ctx<'a> {
    setup: FitSetup<'a>,
    cands: ChainCandidates<'a>,
    ecand: ChainCandidates<'a>,
    eval: &'a FitPhoto,
    p0: Vec<Px>,
    p0_sample: Vec<[f64; 2]>,
    refpx: &'a [Px],
    refscene: &'a PixelScene,
    skin: Vec<bool>,
    whites: Vec<bool>,
    ref_whites: Vec<bool>,
    ref_c: f64,
    p0_skin: Option<Stat>,
    p0_whites: Option<Stat>,
}

impl Ctx<'_> {
    fn scene_ref(&self) -> SceneRef<'_> {
        SceneRef {
            class: &self.eval.scene.class,
            skin: &self.eval.scene.skin,
        }
    }

    /// The numbers `x`: the objective's terms on the sample, and the harms, the transfers and the skin and whites statistics on the
    /// full picture.
    fn at(&mut self, name: &str, kind: &str, x: &Params) -> Result<(Value, f64, Vec<Px>)> {
        let (obj, t) = score(&self.setup, &mut self.cands, x).map_err(|e| anyhow!("{e}"))?;
        let mut px = vec![Px::default(); self.p0.len()];
        self.ecand
            .render(x, &mut px)
            .map_err(|_| anyhow!("render refused"))?;
        let p0p = Pic {
            px: &self.p0,
            scene: Some(self.scene_ref()),
        };
        let fitp = Pic {
            px: &px,
            scene: p0p.scene,
        };
        let refp = Pic {
            px: self.refpx,
            scene: Some(SceneRef {
                class: &self.refscene.class,
                skin: &self.refscene.skin,
            }),
        };
        let h = harms(&p0p, &fitp, &refp);
        let tr = transfers(&p0p, &fitp, &refp, &self.p0_sample);
        let skin = stat(&px, &self.skin);
        let whites = stat(&px, &self.whites);
        let mut v = json!({
            "name": name, "kind": kind,
            "nonzero": x.iter().filter(|&&v| v != 0.0).count(),
            "objective": obj, "look_only": t.look_only_total(), "walls_and_clip": t.walls_and_clip(),
            "terms": terms_json(&t),
            "flags": HARMS.iter().zip(&h).filter(|(_, h)| h.1).map(|(id, _)| *id).collect::<Vec<_>>(),
            "H2": { "value": h[1].0, "flag": h[1].1 },
            "H4": { "value": h[3].0, "flag": h[3].1 },
            "T1": tr[0], "T2": tr[1], "T3": tr[2],
        });
        if let (Some(a), Some(b)) = (&self.p0_skin, &skin) {
            let (dh, ratio) = skin_move(a, b);
            v["skin"] = json!({ "dh_deg": dh, "chroma_ratio": ratio, "stat": b.json() });
        }
        if let Some(w) = &whites {
            v["whites"] = json!({ "stat": w.json(), "limit": 8.0f64.max(self.ref_c + 4.0) });
        }
        Ok((v, obj, px))
    }
}

/// Golden section of one coordinate, the search's own step (`search.rs`), here on the tool's scoring.
fn golden(
    f: &mut dyn FnMut(&Params) -> Result<f64>,
    x: &mut Params,
    i: usize,
    (half, iters): (f64, usize),
    current: f64,
) -> Result<f64> {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let x0 = x[i];
    let (mut a, mut b) = ((x0 - half).max(-1.0), (x0 + half).min(1.0));
    let mut at = |x: &Params, v: f64| -> Result<f64> {
        let mut t = *x;
        t[i] = v;
        f(&t)
    };
    let (mut c, mut d) = (b - INV_PHI * (b - a), a + INV_PHI * (b - a));
    let (mut fc, mut fd) = (at(x, c)?, at(x, d)?);
    let mut best = (x0, current);
    for (v, fv) in [(c, fc), (d, fd)] {
        if fv < best.1 {
            best = (v, fv);
        }
    }
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = at(x, c)?;
            if fc < best.1 {
                best = (c, fc);
            }
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = at(x, d)?;
            if fd < best.1 {
                best = (d, fd);
            }
        }
    }
    x[i] = best.0;
    Ok(best.1)
}

/// The passes of a repolish: the search's local passes and its polish pass.
const REPOLISH_PASSES: [(f64, usize); 3] = [(1.0, 7), (0.4, 6), (0.25, 6)];

fn repolish(ctx: &mut Ctx<'_>, start: &Params, g: &Group) -> Result<Params> {
    let mut x = *start;
    let free: Vec<usize> = (0..N_PARAMS).filter(|i| !g.idx.contains(i)).collect();
    let mut f = |x: &Params| -> Result<f64> {
        Ok(score(&ctx.setup, &mut ctx.cands, x)
            .map_err(|e| anyhow!("{e}"))?
            .0)
    };
    let mut cur = f(&x)?;
    for pass in REPOLISH_PASSES {
        for &i in &free {
            cur = golden(&mut f, &mut x, i, pass, cur)?;
        }
    }
    Ok(x)
}

fn masks_json(ctx: &Ctx<'_>, sample: &FitPhoto, fixed: &Fixed, skin_protection: bool) -> Value {
    let w = ctx.eval.width;
    let n_s = sample.sample.len();
    let s_skin = sample.scene.skin.iter().filter(|&&s| s).count();
    let both = sample
        .sample
        .iter()
        .zip(&sample.scene.skin)
        .filter(|&(&(x, y), &s)| s && ctx.eval.scene.skin.get(y * w + x).copied().unwrap_or(false))
        .count();
    let e_skin = ctx.skin.iter().filter(|&&s| s).count();
    let person = ctx
        .eval
        .scene
        .class
        .iter()
        .filter(|&&c| c == class::PERSON)
        .count();
    let person_skin = ctx
        .eval
        .scene
        .class
        .iter()
        .zip(&ctx.skin)
        .filter(|&(&c, &s)| c == class::PERSON && s)
        .count();
    let n_e = ctx.p0.len();
    let pe = &fixed.p.el[5];
    let present = pe.share >= ym_color::refgrade::measure::PRESENT_SHARE;
    let (_, pc) = pe.mv();
    json!({
        "h2_mask": "the scene's skin plane at every pixel of the analysis picture (Pic::skin of the photo's own scene)",
        "guard_mask": "the same scene's skin plane gathered at the fit's 8192-pixel sample (Fixed.p.el[5], `in_element(i, 5)`)",
        "analysis_size": [ctx.eval.width, ctx.eval.height],
        "h2_share": e_skin as f64 / n_e as f64,
        "h2_pixels": e_skin,
        "guard_share_on_sample": s_skin as f64 / n_s as f64,
        "guard_sample_pixels": s_skin,
        "sample_skin_inside_h2_mask": both,
        "overlap_of_sample_skin_in_h2_mask": if s_skin == 0 { Value::Null } else { json!(both as f64 / s_skin as f64) },
        "person_share": person as f64 / n_e as f64,
        "skin_share_of_person": if person == 0 { Value::Null } else { json!(person_skin as f64 / person as f64) },
        "skin_protection": skin_protection,
        "guard_present_gate_2pct": present,
        "guard_mean_vector_c": pc,
        "guard_drift_wall_applies": present && skin_protection && pc >= 2.5,
        "guard_absolute_and_chroma_walls_apply": present && skin_protection,
        "h2_tests_it": e_skin as f64 / n_e as f64 >= 0.01,
    })
}

/// What the objective's whites walls see on the fit's sample: whether the whites group exists (L\* at least 80 and C\* under 10 in
/// at least 2 % of the sample, the walls' gate), the cap's pixels (not skin, C\* under 8) and their share.
fn whites_wall_json(p_px: &[Px], scene: &PixelScene, f: &Fixed) -> Value {
    let n = p_px.len() as f64;
    let hl = p_px
        .iter()
        .filter(|q| q.lab[0] >= 80.0 && q.chroma < 10.0)
        .count();
    let cap = p_px
        .iter()
        .zip(&scene.skin)
        .filter(|&(q, &s)| !s && q.lab[0] >= 80.0 && q.chroma < 8.0)
        .count();
    let any8 = p_px
        .iter()
        .filter(|q| q.lab[0] >= 80.0 && q.chroma < 8.0)
        .count();
    json!({
        "group_share_c10": hl as f64 / n, "cap_pixels_share_nonskin_c8": cap as f64 / n, "h4_definition_share_on_sample": any8 as f64 / n,
        "walls_apply": hl as f64 >= 0.02 * n,
        "hl_cmax": f.t.hl_cmax, "white_tol": f.t.white_tol, "white_target": f.t.white,
    })
}

fn targets_json(f: &Fixed) -> Value {
    let t = &f.t;
    json!({
        "wall_deg": t.wall_deg, "hl_cmax": t.hl_cmax, "white": t.white, "white_tol": t.white_tol,
        "skin_drift": t.skin_drift, "skin_dlogc_max": t.skin_dlogc_max,
        "skin_matched": t.matched[5], "skin_hue_gate": t.hue_gate[5], "skin_target_h": t.el_h[5], "skin_target_logc": t.el_logc[5],
        "unify": t.unify, "cast": t.cast, "tint": t.tint, "logsat": t.logsat,
        "turn_deg": t.turn_deg, "turn_share": t.turn_share,
        "purple_ok": t.purple_ok,
        "matched": t.matched, "hue_gate": t.hue_gate,
        "photo_cast": [f.p.cast_a, f.p.cast_b], "ref_cast": [f.r.cast_a, f.r.cast_b],
        "photo_tint": f.p.tint, "ref_tint": f.r.tint,
        "photo_skin_share_sample": f.p.el[5].share, "ref_skin_share": f.r.el[5].share,
        "photo_skin_mv_lab": { "hue": f.p.el[5].mv().0, "chroma": f.p.el[5].mv().1 },
        "ref_skin_mv_lab": { "hue": f.r.el[5].mv().0, "chroma": f.r.el[5].mv().1 },
    })
}

fn groups_json(x: &Params) -> Value {
    let mut m = Map::new();
    for g in groups() {
        let vals: Vec<Value> = g
            .idx
            .iter()
            .map(|&i| json!({ "i": i, "name": param_name(i), "x": x[i] }))
            .collect();
        m.insert(
            g.name.into(),
            json!({ "nonzero": g.idx.iter().any(|&i| x[i] != 0.0), "numbers": vals }),
        );
    }
    Value::Object(m)
}

/// Runs the explanation and writes one JSON per pair into `opts.run.out`.
///
/// # Errors
///
/// An input that cannot be read, a photo outside the dev set, a core that will not open.
pub fn explain(opts: &ExplainOpts) -> Result<()> {
    if opts.facts_only {
        return facts(opts);
    }
    let pairs = parse_pairs(&fs::read_to_string(&opts.pairs)?)?;
    fs::create_dir_all(&opts.run.out)?;
    let refs = reference_files(&opts.run.references)?;
    let mut run = opts.run.clone();
    run.set = "dev".into();
    run.shard = (0, 1);
    run.only = pairs.iter().map(|p| p.0.clone()).collect();
    run.only.sort();
    run.only.dedup();
    let all: Vec<SplitEntry> = selected(&run)?;
    // Only the dev set: a pair that names another photo is refused, not run.
    for (p, _) in &pairs {
        if !all.iter().any(|e| e.name() == *p) {
            bail!("{p}: not a photo of the dev set");
        }
    }
    if let Some(e) = all.iter().find(|e| e.set != "dev") {
        bail!("{}: not in the dev set", e.name());
    }
    let (si, sn) = opts.run.shard;
    let mine: Vec<&SplitEntry> = all
        .iter()
        .enumerate()
        .filter(|(k, _)| k % sn == si)
        .map(|(_, e)| e)
        .collect();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.run.threads.max(1))
        .build_global();
    let app = tempfile::tempdir()?;
    let core = open_core(&opts.run, app.path())?;
    let net = Network::shipped().map_err(|e| anyhow!("{e}"))?;
    let cancel = CancelToken::new();
    let suffix = if opts.whatif.is_some() {
        "whatif.json"
    } else {
        "json"
    };
    let out_file = |p: &str, r: &str| opts.run.out.join(format!("{p}__{r}.{suffix}"));
    for entry in mine {
        let name = entry.name();
        let todo: Vec<&str> = pairs
            .iter()
            .filter(|(p, r)| *p == name && !out_file(p, r).is_file())
            .map(|(_, r)| r.as_str())
            .collect();
        if todo.is_empty() {
            continue;
        }
        eprintln!("== {name}");
        let id: PhotoId = open_photo(&core, &entry.raw)?;
        let result = (|| -> Result<()> {
            let now = core.session.view(id)?.settings;
            core.update_settings(id, &add_node_patch(&now)?)?;
            let eval = core
                .reference_eval_photo(id, NODE, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            let base: ReferenceGradeNode = core
                .session
                .view(id)?
                .settings
                .nodes
                .reference_grade(NODE)
                .ok_or_else(|| anyhow!("the node is gone"))?
                .clone();
            let src = PhotoSource::from_session(&core.session, id).map_err(|e| anyhow!("{e}"))?;
            let sample = core
                .renderer
                .reference_fit_photo(&src, NODE, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            for rname in todo {
                let still = refs
                    .iter()
                    .find(|p| p.file_stem().is_some_and(|s| s.to_string_lossy() == rname))
                    .ok_or_else(|| anyhow!("{rname}: no such reference"))?;
                let t0 = Instant::now();
                let saved = match &opts.whatif {
                    Some((dir, _)) => {
                        Some(saved_numbers(&dir.join(format!("{name}__{rname}.json")))?)
                    }
                    None => None,
                };
                let rec = one_pair(
                    &core, id, entry, rname, still, &eval, &sample, &base, &net, &cancel, opts,
                    saved,
                );
                core.reference_clear(id, NODE);
                match rec {
                    Ok(v) => {
                        let path = out_file(&name, rname);
                        fs::write(&path, serde_json::to_string_pretty(&v)?)?;
                        eprintln!("   {rname}: {:.0} s", t0.elapsed().as_secs_f64());
                    }
                    Err(e) => eprintln!("   {rname}: FAILED: {e:#}"),
                }
            }
            Ok(())
        })();
        core.remove_photos(&[id]);
        result?;
    }
    Ok(())
}

/// The cost of the objective's absolute skin wall (`docs/34` section 5.4) for a skin mean vector of CIELAB hue `hue_deg` and chroma
/// `chroma`: `12 * ((eh / 3)^2 + (ec / 3)^2)`, `eh` the hue's distance outside 30 to 70 degrees and `ec` the chroma's outside 12 to 38.
/// The same numbers as the objective's (its constants are private), so a photo's own skin outside the band shows its cost at the identity.
#[must_use]
pub fn skin_absolute_wall(hue_deg: f64, chroma: f64) -> f64 {
    let eh = (30.0 - hue_deg).max(hue_deg - 70.0).max(0.0);
    let ec = (12.0 - chroma).max(chroma - 38.0).max(0.0);
    12.0 * ((eh / 3.0).powi(2) + (ec / 3.0).powi(2))
}

/// The photo-level facts of one photo at the identity: what H2 and H4 measure, and what the walls can see of it.
fn facts(opts: &ExplainOpts) -> Result<()> {
    let dir = opts.run.out.join("facts");
    fs::create_dir_all(&dir)?;
    let mut run = opts.run.clone();
    run.set = "dev".into();
    run.shard = (0, 1);
    run.only = Vec::new();
    let all: Vec<SplitEntry> = selected(&run)?;
    let (si, sn) = opts.run.shard;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.run.threads.max(1))
        .build_global();
    let app = tempfile::tempdir()?;
    let core = open_core(&opts.run, app.path())?;
    let cancel = CancelToken::new();
    for (k, entry) in all.iter().enumerate() {
        let name = entry.name();
        let file = dir.join(format!("{name}.json"));
        if k % sn != si || file.is_file() || entry.set != "dev" {
            continue;
        }
        eprintln!("== {name}");
        let id = open_photo(&core, &entry.raw)?;
        let result = (|| -> Result<()> {
            let now = core.session.view(id)?.settings;
            core.update_settings(id, &add_node_patch(&now)?)?;
            let eval = core
                .reference_eval_photo(id, NODE, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            let base: ReferenceGradeNode = core
                .session
                .view(id)?
                .settings
                .nodes
                .reference_grade(NODE)
                .ok_or_else(|| anyhow!("the node is gone"))?
                .clone();
            let src = PhotoSource::from_session(&core.session, id).map_err(|e| anyhow!("{e}"))?;
            let sample = core
                .renderer
                .reference_fit_photo(&src, NODE, &cancel)
                .map_err(|e| anyhow!("{e}"))?;
            let mut ec = ChainCandidates::new(&eval, &base);
            let mut p0 = vec![Px::default(); ec.sample_len()];
            ec.render(&[0.0; N_PARAMS], &mut p0)
                .map_err(|_| anyhow!("the photo could not be rendered"))?;
            let mut sc = ChainCandidates::new(&sample, &base);
            let mut ps = vec![Px::default(); sc.sample_len()];
            sc.render(&[0.0; N_PARAMS], &mut ps)
                .map_err(|_| anyhow!("the photo could not be rendered"))?;
            let skin = stat(&p0, &eval.scene.skin);
            let skin_sample = stat(&ps, &sample.scene.skin);
            let whites: Vec<bool> = p0
                .iter()
                .map(|q| q.lab[0] >= 80.0 && q.chroma < 8.0)
                .collect();
            let n = ps.len() as f64;
            let hl10 = ps
                .iter()
                .filter(|q| q.lab[0] >= 80.0 && q.chroma < 10.0)
                .count();
            let cap = ps
                .iter()
                .zip(&sample.scene.skin)
                .filter(|&(q, &s)| !s && q.lab[0] >= 80.0 && q.chroma < 8.0)
                .count();
            let skin_light = {
                let nsk = eval.scene.skin.iter().filter(|&&s| s).count();
                let l80 = p0
                    .iter()
                    .zip(&eval.scene.skin)
                    .filter(|&(q, &s)| s && q.lab[0] >= 80.0)
                    .count();
                if nsk == 0 {
                    Value::Null
                } else {
                    json!(l80 as f64 / nsk as f64)
                }
            };
            let person = eval
                .scene
                .class
                .iter()
                .filter(|&&c| c == class::PERSON)
                .count();
            let abs_cost = skin_sample
                .as_ref()
                .filter(|s| s.share >= 0.02)
                .map(|s| skin_absolute_wall(s.hue_lab, s.c_vec));
            let v = json!({
                "photo": name, "tags": entry.tags, "primary_tag": entry.primary_tag,
                "analysis_size": [eval.width, eval.height],
                "skin": opt_stat(&skin), "skin_on_sample": opt_stat(&skin_sample),
                "skin_share_of_frame": skin.as_ref().map(|s| s.share),
                "skin_share_of_person": if person == 0 { Value::Null } else { json!(skin.as_ref().map_or(0, |s| s.n) as f64 / person as f64) },
                "skin_with_L_ge_80": skin_light,
                "skin_absolute_wall_at_identity": abs_cost,
                "whites_h4": opt_stat(&stat(&p0, &whites)),
                "whites_sample_share_c10": hl10 as f64 / n,
                "whites_cap_pixels_share_nonskin_c8_sample": cap as f64 / n,
                "whites_walls_apply": hl10 as f64 >= 0.02 * n,
            });
            fs::write(&file, serde_json::to_string_pretty(&v)?)?;
            Ok(())
        })();
        core.remove_photos(&[id]);
        result?;
    }
    Ok(())
}

/// The fitted numbers an earlier `explain` saved.
fn saved_numbers(path: &Path) -> Result<Params> {
    let v: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let a = v["x"]
        .as_array()
        .ok_or_else(|| anyhow!("{}: no `x`", path.display()))?;
    if a.len() != N_PARAMS {
        bail!("{}: `x` has {} numbers", path.display(), a.len());
    }
    let mut x = [0.0; N_PARAMS];
    for (d, s) in x.iter_mut().zip(a) {
        *d = s
            .as_f64()
            .ok_or_else(|| anyhow!("{}: a number is not a number", path.display()))?;
    }
    Ok(x)
}

/// What the what-if walls read of the photo (on the fit's sample, like the objective) and of the reference.
struct Extra {
    skin_idx: Vec<usize>,
    skin_share: f64,
    p_skin_c: f64,
    p_skin_hue: f64,
    white_idx: Vec<usize>,
    white_share: f64,
    p_white_c: f64,
    ref_white_c: Option<f64>,
}

impl Extra {
    fn new(p_px: &[Px], scene: &PixelScene, ref_whites_c: Option<f64>) -> Self {
        let n = p_px.len() as f64;
        let skin_idx: Vec<usize> = (0..p_px.len()).filter(|&i| scene.skin[i]).collect();
        let (c, h) = skin_mv(p_px, &skin_idx);
        let white_idx: Vec<usize> = (0..p_px.len())
            .filter(|&i| !scene.skin[i] && p_px[i].lab[0] >= 80.0 && p_px[i].chroma < 8.0)
            .collect();
        let p_white_c = mean_c(p_px, &white_idx);
        Self {
            skin_share: skin_idx.len() as f64 / n,
            skin_idx,
            p_skin_c: c,
            p_skin_hue: h,
            white_share: white_idx.len() as f64 / n,
            white_idx,
            p_white_c,
            ref_white_c: ref_whites_c,
        }
    }
}

/// The CIELAB mean-vector chroma and the Oklab mean-vector hue of the pixels at `idx`.
fn skin_mv(px: &[Px], idx: &[usize]) -> (f64, f64) {
    if idx.is_empty() {
        return (0.0, 0.0);
    }
    let n = idx.len() as f64;
    let (mut a, mut b, mut oa, mut ob) = (0.0, 0.0, 0.0, 0.0);
    for &i in idx {
        a += px[i].lab[1];
        b += px[i].lab[2];
        oa += px[i].ok[0];
        ob += px[i].ok[1];
    }
    ((a / n).hypot(b / n), oklab_hue([oa / n, ob / n]))
}

fn mean_c(px: &[Px], idx: &[usize]) -> f64 {
    if idx.is_empty() {
        0.0
    } else {
        idx.iter().map(|&i| px[i].chroma).sum::<f64>() / idx.len() as f64
    }
}

/// The skin walls of the `skin_relative` variant: relative to the photo's own skin, two-sided in chroma, gated at the harm's own 1 %.
/// Chroma ratio inside 0.65 to 1.2 (the harm lines are 0.55 and 1.25), hue move inside 8 degrees (the harm line is 10).
fn skin_relative_wall(ex: &Extra, px: &[Px]) -> f64 {
    if ex.skin_share < 0.01 || ex.skin_idx.is_empty() {
        return 0.0;
    }
    let (c, h) = skin_mv(px, &ex.skin_idx);
    let r = (c.max(1.0) / ex.p_skin_c.max(1.0)).ln();
    let mut w = 12.0 * ((r - 1.2f64.ln()).max(0.0) / 0.05).powi(2);
    w += 12.0 * ((-r - (1.0f64 / 0.65).ln()).max(0.0) / 0.05).powi(2);
    if ex.p_skin_c >= 4.0 {
        w += 12.0 * ((wrap_deg(h - ex.p_skin_hue).abs() - 8.0).max(0.0) / 3.0).powi(2);
    }
    w
}

/// The whites licence of the `whites_licence` variant: the photo's non-skin whites may carry the reference's own whites' chroma, at most 3
/// points above the photo's, and with no whites in the reference no new chroma at all; `allow = max(p, min(r, p + 3)) + 1`.
fn whites_licence_wall(ex: &Extra, px: &[Px]) -> f64 {
    if ex.white_share < 0.005 {
        return 0.0;
    }
    let p = ex.p_white_c;
    let r = ex.ref_white_c.unwrap_or(p);
    let allow = p.max(r.min(p + 3.0)) + 1.0;
    3.0 * ((mean_c(px, &ex.white_idx) - allow).max(0.0) / 2.0).powi(2)
}

/// The full objective of `x` under a what-if variant: the product's own objective with the named change (terms taken out, or a wall
/// added). `base` is the product's objective itself.
fn variant_objective(
    ctx: &mut Ctx<'_>,
    ex: &Extra,
    variant: &str,
    x: &Params,
    buf: &mut [Px],
) -> Result<f64> {
    let (obj, t) = score(&ctx.setup, &mut ctx.cands, x).map_err(|e| anyhow!("{e}"))?;
    if variant == "base" || t.clip >= 1e5 {
        return Ok(obj);
    }
    ctx.cands
        .render(x, buf)
        .map_err(|_| anyhow!("render refused"))?;
    Ok(match variant {
        "no_skin_abs" => obj - t.guard_skin,
        "skin_relative" => {
            obj - t.guard_skin - t.guard_skin_drift - t.guard_skin_chroma
                + skin_relative_wall(ex, buf)
        }
        "whites_licence" => obj + whites_licence_wall(ex, buf),
        "both" => {
            obj - t.guard_skin - t.guard_skin_drift - t.guard_skin_chroma
                + skin_relative_wall(ex, buf)
                + whites_licence_wall(ex, buf)
        }
        other => bail!("unknown what-if variant {other}"),
    })
}

/// Searches `x0` again under the variant: the repolish passes over every number, the search's own golden step.
fn search_variant(
    ctx: &mut Ctx<'_>,
    ex: &Extra,
    variant: &str,
    x0: &Params,
    buf: &mut [Px],
) -> Result<Params> {
    let mut x = *x0;
    let mut f = |x: &Params| variant_objective(ctx, ex, variant, x, buf);
    let mut cur = f(&x)?;
    for pass in REPOLISH_PASSES {
        for i in 0..N_PARAMS {
            cur = golden(&mut f, &mut x, i, pass, cur)?;
        }
    }
    Ok(x)
}

fn whatif_pair(
    ctx: &mut Ctx<'_>,
    scene: &PixelScene,
    p_px: &[Px],
    entry: &SplitEntry,
    rname: &str,
    x: &Params,
    variants: &[String],
) -> Result<Value> {
    let ref_c = stat(ctx.refpx, &ctx.ref_whites)
        .filter(|s| s.share >= 0.005)
        .map(|s| s.c_mean);
    let ex = Extra::new(p_px, scene, ref_c);
    let mut buf = vec![Px::default(); ctx.cands.sample_len()];
    let (fit_v, fit_obj, _) = ctx.at("fit", "fit", x)?;
    let mut out = Vec::new();
    for v in variants {
        let t = Instant::now();
        let before = variant_objective(ctx, &ex, v, x, &mut buf)?;
        let y = search_variant(ctx, &ex, v, x, &mut buf)?;
        let after = variant_objective(ctx, &ex, v, &y, &mut buf)?;
        let (mut rec, obj_y, _) = ctx.at(v, "whatif", &y)?;
        rec["variant_objective_at_fit"] = json!(before);
        rec["variant_objective_after"] = json!(after);
        rec["product_objective_after"] = json!(obj_y);
        rec["product_objective_minus_fit"] = json!(obj_y - fit_obj);
        rec["changed"] = json!(
            (0..N_PARAMS)
                .filter(|&i| (y[i] - x[i]).abs() > 1e-12)
                .count()
        );
        rec["seconds"] = json!(t.elapsed().as_secs_f64());
        out.push(rec);
    }
    Ok(json!({
        "photo": entry.name(), "reference": rname, "tags": entry.tags,
        "at_fit": fit_v, "whatif": out,
    }))
}

#[allow(clippy::too_many_arguments)]
fn one_pair(
    core: &ym_core::app::Core,
    id: PhotoId,
    entry: &SplitEntry,
    rname: &str,
    still: &Path,
    eval: &FitPhoto,
    sample: &FitPhoto,
    base: &ReferenceGradeNode,
    net: &Network,
    cancel: &CancelToken,
    opts: &ExplainOpts,
    saved: Option<Params>,
) -> Result<Value> {
    core.reference_choose_file(id, NODE, still, Some(rname))
        .map_err(|e| anyhow!("{e}"))?;
    let t = Instant::now();
    let fit = match saved {
        Some(_) => None,
        None => Some(
            core.reference_fit(id, NODE, cancel, &mut |_| {})
                .map_err(|e| anyhow!("{e}"))?,
        ),
    };
    let seconds = t.elapsed().as_secs_f64();
    if let Some(f) = &fit {
        core.update_settings(id, &f.undo)?;
    }
    let refm = core
        .reference_measured(id, NODE, cancel)
        .map_err(|e| anyhow!("{e}"))?;
    // The fit's own objective, as it built it (the sample, the photo's pixels, the reference's).
    let mut cands = ChainCandidates::new(sample, base);
    let mut p_px = vec![Px::default(); cands.sample_len()];
    cands
        .render(&[0.0; N_PARAMS], &mut p_px)
        .map_err(|_| anyhow!("the photo could not be rendered"))?;
    let fixed = Fixed::new(
        net,
        &p_px,
        &sample.scene,
        &refm.px,
        &refm.scene,
        base.skin_protection,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let setup = FitSetup {
        fixed: &fixed,
        scene: &sample.scene,
        mask_ok: sample.mask_ok,
    };
    // The measures' side: every pixel.
    let mut ecand = ChainCandidates::new(eval, base);
    let mut p0 = vec![Px::default(); ecand.sample_len()];
    ecand
        .render(&[0.0; N_PARAMS], &mut p0)
        .map_err(|_| anyhow!("the photo could not be rendered"))?;
    let p0_sample = sample_ab(&p0);
    let skin = eval.scene.skin.clone();
    let whites: Vec<bool> = p0
        .iter()
        .map(|q| q.lab[0] >= 80.0 && q.lab[1].hypot(q.lab[2]) < 8.0)
        .collect();
    let ref_whites: Vec<bool> = refm
        .px
        .iter()
        .map(|q| q.lab[0] >= 80.0 && q.lab[1].hypot(q.lab[2]) < 8.0)
        .collect();
    let ref_whites_stat = stat(&refm.px, &ref_whites);
    let ref_c = ref_whites_stat.as_ref().map_or(0.0, |s| s.c_mean);
    let p0_skin = stat(&p0, &skin);
    let p0_whites = stat(&p0, &whites);
    let ref_skin = stat(&refm.px, &refm.scene.skin);
    let mut ctx = Ctx {
        setup,
        cands,
        ecand,
        eval,
        p0,
        p0_sample,
        refpx: &refm.px,
        refscene: &refm.scene,
        skin,
        whites,
        ref_whites,
        ref_c,
        p0_skin,
        p0_whites,
    };
    if let (Some(x), Some((_, variants))) = (saved, &opts.whatif) {
        return whatif_pair(&mut ctx, &sample.scene, &p_px, entry, rname, &x, variants);
    }
    let fit = fit.ok_or_else(|| anyhow!("no fit"))?;
    let x = fit.result.x;
    let (fit_v, fit_obj, fit_px) = ctx.at("fit", "fit", &x)?;
    let (ident_v, _, _) = ctx.at("identity", "identity", &[0.0; N_PARAMS])?;
    // The fit's three searches: the objective prefers one of them, and the other may be harmless.
    let mut searches = Vec::new();
    for (k, s) in fit.result.searches.iter().enumerate() {
        let label = ["neutral", "look_only", "from_look"][k];
        let (mut v, _, _) = ctx.at(label, "search", &s.x)?;
        v["need"] = json!(s.need.local);
        v["x"] = json!(s.x.to_vec());
        searches.push(v);
    }
    // The counterfactuals.
    let mut cf = Vec::new();
    let mut zero_obj: Vec<(Group, f64, bool)> = Vec::new();
    let flagged_h2 = fit_v["H2"]["flag"].as_bool().unwrap_or(false);
    let flagged_h4 = fit_v["H4"]["flag"].as_bool().unwrap_or(false);
    for g in groups() {
        if !g.idx.iter().any(|&i| x[i] != 0.0) {
            continue;
        }
        let (mut v, o, _) = ctx.at(g.name, "zero", &zeroed(&x, &g))?;
        v["clears_h2"] = json!(flagged_h2 && !v["H2"]["flag"].as_bool().unwrap_or(false));
        v["clears_h4"] = json!(flagged_h4 && !v["H4"]["flag"].as_bool().unwrap_or(false));
        v["objective_minus_fit"] = json!(o - fit_obj);
        let clears = (!flagged_h2 || v["clears_h2"] == json!(true))
            && (!flagged_h4 || v["clears_h4"] == json!(true))
            && (flagged_h2 || flagged_h4);
        zero_obj.push((g.clone(), o, clears));
        cf.push(v);
        let (v, _, _) = ctx.at(g.name, "only", &only(&x, &g))?;
        cf.push(v);
    }
    // The search again with a harm-clearing group at identity (the two cheapest of the spec's groups).
    let mut re = Vec::new();
    if opts.repolish {
        let coarse = [
            "tone",
            "channels",
            "saturation",
            "hue_vs_sat",
            "hue_vs_hue",
            "sky",
            "vegetation",
            "water",
            "ground",
            "built",
            "skin",
            "subject",
        ];
        let mut cand: Vec<&(Group, f64, bool)> = zero_obj
            .iter()
            .filter(|(g, _, clears)| *clears && coarse.contains(&g.name))
            .collect();
        cand.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (g, _, _) in cand.into_iter().take(2) {
            let y = repolish(&mut ctx, &zeroed(&x, g), g)?;
            let (mut v, o, _) = ctx.at(g.name, "repolish", &y)?;
            v["objective_minus_fit"] = json!(o - fit_obj);
            v["numbers_changed"] = json!(
                (0..N_PARAMS)
                    .filter(|&i| (y[i] - x[i]).abs() > 1e-12)
                    .map(|i| json!({ "i": i, "name": param_name(i), "from": x[i], "to": y[i] }))
                    .collect::<Vec<_>>()
            );
            re.push(v);
        }
    }
    // The statistics.
    let fit_skin = stat(&fit_px, &ctx.skin);
    let fit_whites = stat(&fit_px, &ctx.whites);
    let skin_in_whites = {
        let n_skin = ctx.skin.iter().filter(|&&s| s).count();
        let both = ctx
            .skin
            .iter()
            .zip(&ctx.whites)
            .filter(|&(&s, &w)| s && w)
            .count();
        let light = ctx
            .p0
            .iter()
            .zip(&ctx.skin)
            .filter(|&(q, &s)| s && q.lab[0] >= 80.0)
            .count();
        json!({
            "skin_pixels": n_skin,
            "share_of_skin_inside_whites_mask": if n_skin == 0 { Value::Null } else { json!(both as f64 / n_skin as f64) },
            "share_of_skin_with_L_ge_80": if n_skin == 0 { Value::Null } else { json!(light as f64 / n_skin as f64) },
            "share_of_whites_that_is_skin": ctx.p0_whites.as_ref().map(|w| {
                let b = ctx.skin.iter().zip(&ctx.whites).filter(|&(&s, &w)| s && w).count();
                b as f64 / w.n as f64
            }),
        })
    };
    let (dh, ratio) = match (&ctx.p0_skin, &fit_skin) {
        (Some(a), Some(b)) => skin_move(a, b),
        _ => (None, f64::NAN),
    };
    // Does the fit move the photo's skin toward the typical skin hue (the walls' 30 to 70 degree band, centre 50)?
    let toward_typical = match (&ctx.p0_skin, &fit_skin) {
        (Some(a), Some(b)) => {
            let d = |h: f64| wrap_deg(h - 50.0).abs();
            Some(d(b.hue_lab) < d(a.hue_lab))
        }
        _ => None,
    };
    let ref_whites_n = ctx.ref_whites.iter().filter(|&&w| w).count();
    let stats = json!({
        "skin": {
            "photo": opt_stat(&ctx.p0_skin), "fit": opt_stat(&fit_skin), "reference": opt_stat(&ref_skin),
            "dh_deg": dh, "chroma_ratio": if ratio.is_nan() { Value::Null } else { json!(ratio) },
            "fit_hue_lab_closer_to_50deg": toward_typical,
            "skin_vs_whites": skin_in_whites,
        },
        "whites": {
            "photo": opt_stat(&ctx.p0_whites), "fit": opt_stat(&fit_whites),
            "reference_own_whites": opt_stat(&ref_whites_stat),
            "reference_whites_share": ref_whites_n as f64 / refm.px.len() as f64,
            "h4_limit": 8.0f64.max(ctx.ref_c + 4.0),
            "photo_whites_share_ge_1pct": ctx.p0_whites.as_ref().is_some_and(|w| w.share >= 0.01),
        },
    });
    let rc = &fit.result;
    let whites_walls = whites_wall_json(&p_px, &sample.scene, &fixed);
    let cm = candidate_measure(&fit_sample_px(&mut ctx, &x)?, &sample.scene, &fixed);
    Ok(json!({
        "photo": entry.name(), "reference": rname, "tags": entry.tags, "primary_tag": entry.primary_tag,
        "fit": {
            "seconds": seconds, "objective": rc.total, "rescored": fit_obj, "rescore_delta": fit_obj - rc.total,
            "identity_objective": rc.identity_total, "neutral_total": rc.neutral_total, "look_total": rc.look_total,
            "start": match rc.start { Start::Neutral => "neutral", Start::Look => "look" },
            "identity": rc.identity, "evals": rc.evals, "need": rc.need.local,
        },
        "x": x.to_vec(),
        "groups": groups_json(&x),
        "numbers": numbers_json(&to_numbers(&x)),
        "targets": targets_json(&fixed),
        "masks": masks_json(&ctx, sample, &fixed, base.skin_protection),
        "whites_walls": whites_walls,
        "stats": stats,
        "at_fit": fit_v,
        "at_identity": ident_v,
        "searches": searches,
        "sample_measure": {
            "photo_skin": { "share": fixed.p.el[5].share, "n": fixed.p.el[5].n, "mv_lab": [fixed.p.el[5].mv().0, fixed.p.el[5].mv().1] },
            "fit_skin": { "share": cm.m.el[5].share, "n": cm.m.el[5].n, "mv_lab": [cm.m.el[5].mv().0, cm.m.el[5].mv().1] },
        },
        "counterfactuals": cf,
        "repolish": re,
    }))
}

fn fit_sample_px(ctx: &mut Ctx<'_>, x: &Params) -> Result<Vec<Px>> {
    let mut px = vec![Px::default(); ctx.cands.sample_len()];
    ctx.cands
        .render(x, &mut px)
        .map_err(|_| anyhow!("render refused"))?;
    Ok(px)
}

/// The photos and references of a set of pair files, for a summary of coverage (a test helper and a log line).
#[must_use]
pub fn pair_index(pairs: &[(String, String)]) -> BTreeMap<String, Vec<String>> {
    let mut m: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (p, r) in pairs {
        m.entry(p.clone()).or_default().push(r.clone());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(l: f64, a: f64, b: f64) -> Px {
        // Oklab (a, b) taken as a scaled copy of CIELAB: the tests only need a consistent hue.
        Px::new([l, a, b], [a / 100.0, b / 100.0], false, false)
    }

    #[test]
    fn pairs_are_read_in_order_and_bad_lines_are_refused() {
        let p = parse_pairs("# c\n\nphoto-a:ref_1\n photo-b : ref_2 \n").unwrap();
        assert_eq!(
            p,
            vec![
                ("photo-a".into(), "ref_1".into()),
                ("photo-b".into(), "ref_2".into())
            ]
        );
        assert!(parse_pairs("no-colon").is_err());
        assert!(parse_pairs(":ref").is_err());
        assert_eq!(pair_index(&p)["photo-a"], vec!["ref_1".to_string()]);
    }

    #[test]
    fn groups_cover_the_numbers_and_zero_and_only_are_complements() {
        let gs = groups();
        let by = |n: &str| gs.iter().find(|g| g.name == n).unwrap().clone();
        let mut all: Vec<usize> = by("global_all").idx;
        all.extend(by("local_all").idx);
        assert_eq!(all, (0..N_PARAMS).collect::<Vec<_>>());
        // The element groups are the spec's: three numbers each from index 30, the subject last.
        assert_eq!(by("skin").idx, vec![45, 46, 47]);
        assert_eq!(by("subject").idx, vec![48]);
        assert_eq!(by("channel_mid").idx, vec![9, 12, 15]);
        let mut x = [0.0; N_PARAMS];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f64 + 1.0) / 100.0;
        }
        let g = by("skin");
        let (z, o) = (zeroed(&x, &g), only(&x, &g));
        for i in 0..N_PARAMS {
            if g.idx.contains(&i) {
                assert_eq!((z[i], o[i]), (0.0, x[i]));
            } else {
                assert_eq!((z[i], o[i]), (x[i], 0.0));
            }
        }
    }

    #[test]
    fn the_skin_move_is_the_harms_own_hue_and_chroma_ratio() {
        let p = vec![
            px(60.0, 18.0, 16.0),
            px(62.0, 20.0, 14.0),
            px(10.0, 0.0, 0.0),
        ];
        let f = vec![
            px(60.0, 12.0, 22.0),
            px(62.0, 14.0, 20.0),
            px(10.0, 0.0, 0.0),
        ];
        let mask = [true, true, false];
        let (a, b) = (stat(&p, &mask).unwrap(), stat(&f, &mask).unwrap());
        assert_eq!(a.n, 2);
        assert!((a.share - 2.0 / 3.0).abs() < 1e-12);
        let (dh, ratio) = skin_move(&a, &b);
        assert!(dh.unwrap() > 10.0, "a turn of about 20 degrees");
        assert!((ratio - b.c_vec / a.c_vec).abs() < 1e-12);
        // The same numbers through the evaluation's own `harms`: the flag agrees.
        let scene = SceneRef {
            class: &[0, 0, 0],
            skin: &mask,
        };
        let (pp, fp) = (
            Pic {
                px: &p,
                scene: Some(scene),
            },
            Pic {
                px: &f,
                scene: Some(scene),
            },
        );
        let h = harms(&pp, &fp, &pp);
        assert_eq!(
            h[1].0.map(|v| (v - dh.unwrap().abs()).abs() < 1e-9),
            Some(true)
        );
        assert!(h[1].1);
        // Below a mean-vector C* of 4 the hue is not tested.
        let grey = vec![px(60.0, 1.0, 1.0), px(60.0, 1.0, 1.0)];
        let g = stat(&grey, &[true, true]).unwrap();
        assert_eq!(skin_move(&g, &g).0, None);
        assert!(stat(&p, &[false; 3]).is_none());
    }

    #[test]
    fn the_absolute_skin_wall_is_zero_in_the_band_and_quadratic_outside() {
        assert_eq!(skin_absolute_wall(50.0, 20.0), 0.0);
        assert_eq!(skin_absolute_wall(30.0, 12.0), 0.0);
        // 3 degrees outside the hue band and 3 outside the chroma band: 12 * (1 + 1).
        assert!((skin_absolute_wall(27.0, 9.0) - 24.0).abs() < 1e-9);
        assert!((skin_absolute_wall(73.0, 41.0) - 24.0).abs() < 1e-9);
    }

    fn extra_for(p_px: &[Px], skin: &[bool], ref_whites_c: Option<f64>) -> Extra {
        let mut scene = PixelScene::blank(p_px.len());
        scene.skin = skin.to_vec();
        Extra::new(p_px, &scene, ref_whites_c)
    }

    #[test]
    fn the_relative_skin_wall_is_two_sided_and_gated_at_one_percent() {
        // Ten pixels, three of them skin (30 %): chroma 20, hue near 50 degrees.
        let skin_px = |c: f64, h: f64| {
            let (a, b) = (c * h.to_radians().cos(), c * h.to_radians().sin());
            px(60.0, a, b)
        };
        let p: Vec<Px> = (0..10)
            .map(|i| {
                if i < 3 {
                    skin_px(20.0, 50.0)
                } else {
                    px(50.0, 0.0, 0.0)
                }
            })
            .collect();
        let mask: Vec<bool> = (0..10).map(|i| i < 3).collect();
        let ex = extra_for(&p, &mask, None);
        assert_eq!(
            skin_relative_wall(&ex, &p),
            0.0,
            "the photo itself costs nothing"
        );
        let mut halved = p.clone();
        let mut doubled = p.clone();
        let mut turned = p.clone();
        for i in 0..3 {
            halved[i] = skin_px(10.0, 50.0);
            doubled[i] = skin_px(40.0, 50.0);
            turned[i] = skin_px(20.0, 80.0);
        }
        // Halved chroma (below the 0.65 line), doubled chroma (above 1.2) and a 30 degree turn (above 8) all cost; both directions.
        assert!(skin_relative_wall(&ex, &halved) > 0.0);
        assert!(skin_relative_wall(&ex, &doubled) > 0.0);
        assert!(skin_relative_wall(&ex, &turned) > 0.0);
        // Skin under 1 % of the sample is not guarded (H2 would test it only from 1 % of the frame; the gate is the harm's own).
        let tiny = extra_for(&p, &[false; 10], None);
        assert_eq!(skin_relative_wall(&tiny, &halved), 0.0);
    }

    #[test]
    fn the_whites_licence_follows_the_reference_whites_and_gives_none_without_them() {
        // Four of ten pixels are non-skin whites with C* 2.
        let p: Vec<Px> = (0..10)
            .map(|i| {
                if i < 4 {
                    px(90.0, 2.0, 0.0)
                } else {
                    px(40.0, 10.0, 10.0)
                }
            })
            .collect();
        let tinted = |c: f64| -> Vec<Px> {
            (0..10)
                .map(|i| {
                    if i < 4 {
                        px(90.0, c, 0.0)
                    } else {
                        px(40.0, 10.0, 10.0)
                    }
                })
                .collect()
        };
        let none = extra_for(&p, &[false; 10], None);
        // No whites in the reference: the photo's own chroma plus the 1 point of slack.
        assert_eq!(whites_licence_wall(&none, &tinted(3.0)), 0.0);
        assert!(whites_licence_wall(&none, &tinted(6.0)) > 0.0);
        // A reference whose whites have C* 6 licenses up to 6 (and never more than 3 above the photo's 2) plus the slack.
        let some = extra_for(&p, &[false; 10], Some(6.0));
        assert_eq!(whites_licence_wall(&some, &tinted(5.5)), 0.0);
        assert!(whites_licence_wall(&some, &tinted(8.0)) > 0.0);
        let huge = extra_for(&p, &[false; 10], Some(20.0));
        assert!(
            whites_licence_wall(&huge, &tinted(9.0)) > 0.0,
            "capped at the photo's plus 3"
        );
    }

    #[test]
    fn golden_finds_the_minimum_of_one_coordinate_and_never_raises_the_start() {
        let mut f = |x: &Params| -> Result<f64> { Ok((x[3] - 0.3).powi(2)) };
        let mut x = [0.0; N_PARAMS];
        let start = f(&x).unwrap();
        let end = golden(&mut f, &mut x, 3, (1.0, 12), start).unwrap();
        assert!(end <= start && (x[3] - 0.3).abs() < 0.02, "{x:?} {end}");
        // A flat objective keeps the start (a tie keeps the start).
        let mut flat = |_: &Params| -> Result<f64> { Ok(1.0) };
        let mut y = [0.0; N_PARAMS];
        golden(&mut flat, &mut y, 5, (1.0, 6), 1.0).unwrap();
        assert_eq!(y, [0.0; N_PARAMS]);
    }
}
