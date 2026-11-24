//! The evaluation harness (`refgrade-notes/16-eval-harness.md`): harm checks H1 to H6 and look-transfer metrics T1 to T3 for
//! every (photo, reference) fit, measured on the full fit-size renders of photo0 and the fit, not on the fit sample. They are
//! checks, not objective terms; their definitions and thresholds deliberately differ from the walls.

use super::*;

/// One photo of the split.
pub struct SplitEntry {
    pub raw: PathBuf,
    pub set: String,
    pub tags: Vec<String>,
    /// The first content tag, for the per-tag tables.
    pub primary_tag: String,
    /// The label folder to pass as `--labels` for this photo, when the split file gives one.
    pub labels: Option<PathBuf>,
}

/// Reads `split.json`: an array, or an object with a `photos` array, of entries with `raw` (or `path`), `set` and `tags`.
pub fn read_split(path: &Path) -> R<Vec<SplitEntry>> {
    let v: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let arr = v
        .get("items")
        .or_else(|| v.get("photos"))
        .and_then(Value::as_array)
        .or_else(|| v.as_array())
        .ok_or("split.json: expected an array of photos")?;
    let mut out = Vec::new();
    for e in arr {
        let raw = e
            .get("raw")
            .or_else(|| e.get("path"))
            .and_then(Value::as_str)
            .ok_or("split.json: an entry has no `raw` path")?;
        let set = e
            .get("set")
            .and_then(Value::as_str)
            .unwrap_or("dev")
            .to_string();
        let tags: Vec<String> = e
            .get("tags")
            .and_then(Value::as_array)
            .map(|t| {
                t.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let primary_tag = e
            .get("primary_tag")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| tags.first().cloned())
            .unwrap_or_else(|| "untagged".into());
        let labels = e.get("labels").and_then(Value::as_str).map(PathBuf::from);
        out.push(SplitEntry {
            raw: PathBuf::from(raw),
            set,
            tags,
            primary_tag,
            labels,
        });
    }
    Ok(out)
}

/// The split used until the data agent's file exists: the owner's `A1M*` photos are test, the rest dev.
pub fn builtin_split(photos: &[PathBuf]) -> Vec<SplitEntry> {
    photos
        .iter()
        .map(|p| {
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            SplitEntry {
                raw: p.clone(),
                set: if stem.starts_with("A1M") {
                    "test"
                } else {
                    "dev"
                }
                .into(),
                tags: Vec::new(),
                primary_tag: "untagged".into(),
                labels: None,
            }
        })
        .collect()
}

const HARMS: [&str; 6] = ["H1", "H2", "H3", "H4", "H5", "H6"];
const HARM_NAMES: [&str; 6] = [
    "purple turn",
    "skin colour",
    "memory colour",
    "tinted whites",
    "clipping",
    "colour reversal",
];
/// The Oklab purple band of the checks.
const BAND: (f64, f64) = (300.0, 355.0);
/// The least photo0 mean-vector C* at which a hue shift is tested (H2, H3).
const HUE_MIN_CHROMA: f64 = 4.0;
const SW_SAMPLES: usize = 4096;
const SW_DIRECTIONS: usize = 32;
const SW_SEED: u64 = 0x5EED_2026_1124;

fn cstar(q: &Px) -> f64 {
    q.lab[1].hypot(q.lab[2])
}

fn in_band(h: f64) -> bool {
    (BAND.0..=BAND.1).contains(&h)
}

/// One picture, measured, with its labels (when it has them).
struct Pic<'a> {
    px: Vec<Px>,
    scene: Option<&'a Scene>,
}

impl Pic<'_> {
    fn skin(&self, i: usize) -> bool {
        self.scene.is_some_and(|s| s.skin[i])
    }
    fn elem(&self, i: usize) -> usize {
        self.scene.map_or(0, |s| usize::from(s.elem[i]))
    }
    fn share(&self, f: impl Fn(usize, &Px) -> bool) -> f64 {
        let n = self.px.iter().enumerate().filter(|&(i, q)| f(i, q)).count();
        n as f64 / self.px.len().max(1) as f64
    }
    /// The Oklab mean-vector hue and the CIELAB mean-vector chroma over the pixels `f` allows, with their count.
    fn mean_vec(&self, f: impl Fn(usize, &Px) -> bool) -> Option<(f64, f64, usize)> {
        let (mut oa, mut ob, mut la, mut lb, mut n) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for (i, q) in self.px.iter().enumerate() {
            if f(i, q) {
                oa += q.ok[0];
                ob += q.ok[1];
                la += q.lab[1];
                lb += q.lab[2];
                n += 1;
            }
        }
        if n == 0 {
            return None;
        }
        let nf = n as f64;
        Some((oklab_hue([oa / nf, ob / nf]), (la / nf).hypot(lb / nf), n))
    }
}

/// The harm values and flags of one fit: `(value, flag)` for H1 to H6, `None` where the check does not apply.
fn harms(p0: &Pic<'_>, fit: &Pic<'_>, refp: &Pic<'_>) -> [(Option<f64>, bool); 6] {
    let n = fit.px.len().max(1) as f64;
    let mut out = [(None, false); 6];
    // H1: purple turn.
    let h1 = fit.share(|i, q| {
        !p0.skin(i) && cstar(q) >= 5.0 && in_band(oklab_hue(q.ok)) && {
            let p = &p0.px[i];
            cstar(p) < 2.0 || wrap_deg(oklab_hue(q.ok) - oklab_hue(p.ok)).abs() >= 15.0
        }
    });
    let ref_purple = refp.share(|_, q| cstar(q) > 8.0 && in_band(oklab_hue(q.ok)));
    out[0] = (Some(h1), h1 > 0.02 && ref_purple < 0.05);
    // H2: skin colour.
    let skin_n = p0.share(|i, _| p0.skin(i));
    if skin_n >= 0.01
        && let (Some(a), Some(b)) = (
            p0.mean_vec(|i, _| p0.skin(i)),
            fit.mean_vec(|i, _| p0.skin(i)),
        )
    {
        let ratio = b.1 / a.1.max(1e-9);
        // Hue is only meaningful on a coloured area: below a mean-vector C* of 4 the hue shift is not tested.
        let dh = (a.1 >= HUE_MIN_CHROMA).then(|| wrap_deg(b.0 - a.0).abs());
        out[1] = (
            dh,
            dh.is_some_and(|d| d > 10.0) || !(0.55..=1.25).contains(&ratio),
        );
    }
    // H3: memory colour (the larger hue move of sky and vegetation).
    let mut h3: Option<f64> = None;
    for e in [ELEM_SKY, ELEM_VEGETATION] {
        let ps = p0.share(|i, _| p0.elem(i) == e);
        let rs = refp.share(|i, _| refp.elem(i) == e);
        if ps >= 0.02
            && rs < 0.02
            && let (Some(a), Some(b)) = (
                p0.mean_vec(|i, _| p0.elem(i) == e),
                fit.mean_vec(|i, _| p0.elem(i) == e),
            )
        {
            // Not sky when dark (median L* under 20); no hue test on a near-grey area.
            let mut ls: Vec<f64> = (0..p0.px.len())
                .filter(|&i| p0.elem(i) == e)
                .map(|i| p0.px[i].lab[0])
                .collect();
            if e == ELEM_SKY && median_of(&mut ls) < 20.0 {
                continue;
            }
            if a.1 >= HUE_MIN_CHROMA {
                let dh = wrap_deg(b.0 - a.0).abs();
                h3 = Some(h3.map_or(dh, |v| v.max(dh)));
            }
        }
    }
    out[2] = (h3, h3.is_some_and(|v| v > 15.0));
    // H4: tinted whites.
    let white = |_: usize, q: &Px| q.lab[0] >= 80.0 && cstar(q) < 8.0;
    let white_share = p0.share(white);
    if white_share >= 0.01 {
        let mean_c = |pic: &Pic<'_>, scope: &dyn Fn(usize) -> bool| {
            let (mut s, mut k) = (0.0, 0usize);
            for (i, q) in pic.px.iter().enumerate() {
                if scope(i) {
                    s += cstar(q);
                    k += 1;
                }
            }
            if k == 0 { 0.0 } else { s / k as f64 }
        };
        let fit_c = mean_c(fit, &|i| white(i, &p0.px[i]));
        let ref_c = mean_c(refp, &|i| white(i, &refp.px[i]));
        out[3] = (Some(fit_c), fit_c > 8.0f64.max(ref_c + 4.0));
    }
    // H5: clipping.
    let clip = |pic: &Pic<'_>| {
        (
            pic.share(|_, q| q.lab[0] > 98.0),
            pic.share(|_, q| q.lab[0] < 2.0),
        )
    };
    let (f, p, r) = (clip(fit), clip(p0), clip(refp));
    let ex_hi = f.0 - p.0.max(r.0);
    let ex_lo = f.1 - p.1.max(r.1);
    let ex = ex_hi.max(ex_lo);
    out[4] = (Some(ex), ex > 0.03);
    // H6: colour reversal.
    let h6 = fit.share(|i, q| {
        let p = &p0.px[i];
        !p0.skin(i) && cstar(p) >= 10.0 && wrap_deg(oklab_hue(q.ok) - oklab_hue(p.ok)).abs() > 60.0
    });
    out[5] = (Some(h6), h6 > 0.03);
    let _ = n;
    out
}

/// `count` seeded samples of (a*, b*) of a picture.
fn sample_ab(px: &[Px]) -> Vec<[f64; 2]> {
    let mut rng = Rng(SW_SEED);
    (0..SW_SAMPLES)
        .map(|_| {
            let q = &px[(rng.next_u64() % px.len() as u64) as usize];
            [q.lab[1], q.lab[2]]
        })
        .collect()
}

/// The sliced Wasserstein-1 distance of two equal-size samples over fixed directions.
fn sliced_w1(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let mut total = 0.0;
    for k in 0..SW_DIRECTIONS {
        let th = std::f64::consts::PI * k as f64 / SW_DIRECTIONS as f64;
        let (c, s) = (th.cos(), th.sin());
        let mut pa: Vec<f64> = a.iter().map(|p| p[0] * c + p[1] * s).collect();
        let mut pb: Vec<f64> = b.iter().map(|p| p[0] * c + p[1] * s).collect();
        pa.sort_unstable_by(f64::total_cmp);
        pb.sort_unstable_by(f64::total_cmp);
        total += pa.iter().zip(&pb).map(|(x, y)| (x - y).abs()).sum::<f64>() / pa.len() as f64;
    }
    total / SW_DIRECTIONS as f64
}

/// The key-preserving tone shape of L*: `(p98 - p2, (p50 - p2) / (p98 - p2))`.
fn tone_shape(px: &[Px]) -> [f64; 2] {
    let mut l: Vec<f64> = px.iter().map(|q| q.lab[0]).collect();
    l.sort_unstable_by(f64::total_cmp);
    let at = |q: f64| l[((l.len() - 1) as f64 * q).round() as usize];
    let (p2, p50, p98) = (at(0.02), at(0.5), at(0.98));
    let span = (p98 - p2).max(1e-9);
    [span, (p50 - p2) / span]
}

fn tone_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).abs() / 50.0 + (a[1] - b[1]).abs()
}

/// The cast: the mean (a*, b*) of the non-skin pixels with 15 <= L* <= 80.
fn cast(pic: &Pic<'_>) -> [f64; 2] {
    let (mut a, mut b, mut n) = (0.0, 0.0, 0usize);
    for (i, q) in pic.px.iter().enumerate() {
        if !pic.skin(i) && (15.0..=80.0).contains(&q.lab[0]) {
            a += q.lab[1];
            b += q.lab[2];
            n += 1;
        }
    }
    if n == 0 {
        [0.0, 0.0]
    } else {
        [a / n as f64, b / n as f64]
    }
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The three transfer metrics; `None` where the denominator is too small (under 1 for T1 and T3, under 0.01 for T2).
fn transfers(
    p0: &Pic<'_>,
    fit: &Pic<'_>,
    refp: &Pic<'_>,
    p0_sample: &[[f64; 2]],
) -> [Option<f64>; 3] {
    let ref_sample = sample_ab(&refp.px);
    let fit_sample = sample_ab(&fit.px);
    let d0 = sliced_w1(p0_sample, &ref_sample);
    let t1 = (d0 >= 1.0).then(|| 1.0 - sliced_w1(&fit_sample, &ref_sample) / d0);
    let rt = tone_shape(&refp.px);
    let e0 = tone_dist(tone_shape(&p0.px), rt);
    let t2 = (e0 >= 0.01).then(|| 1.0 - tone_dist(tone_shape(&fit.px), rt) / e0);
    let rc = cast(refp);
    let c0 = dist2(cast(p0), rc);
    let t3 = (c0 >= 1.0).then(|| 1.0 - dist2(cast(fit), rc) / c0);
    [t1, t2, t3]
}

/// Mean and median of the present values.
fn mean_median(v: &[f64]) -> Option<(f64, f64)> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_unstable_by(f64::total_cmp);
    let med = if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    };
    Some((s.iter().sum::<f64>() / s.len() as f64, med))
}

/// The per-fit records of a set summarised: flag shares, photo shares, transfer means.
fn summarise(recs: &[&Value]) -> Value {
    let flag = |r: &Value, h: &str| r["harms"][h]["flag"].as_bool().unwrap_or(false);
    let any = |r: &Value| HARMS.iter().any(|h| flag(r, h));
    let n = recs.len().max(1) as f64;
    let mut photos: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for r in recs {
        photos
            .entry(r["photo"].as_str().unwrap_or(""))
            .or_default()
            .push(r);
    }
    let np = photos.len().max(1) as f64;
    let mut fit_share = serde_json::Map::new();
    let mut photo_share = serde_json::Map::new();
    for h in HARMS {
        fit_share.insert(
            h.into(),
            json!(recs.iter().filter(|r| flag(r, h)).count() as f64 / n),
        );
        photo_share.insert(
            h.into(),
            json!(
                photos
                    .values()
                    .filter(|v| v.iter().any(|r| flag(r, h)))
                    .count() as f64
                    / np
            ),
        );
    }
    fit_share.insert(
        "any".into(),
        json!(recs.iter().filter(|r| any(r)).count() as f64 / n),
    );
    photo_share.insert(
        "any".into(),
        json!(photos.values().filter(|v| v.iter().any(|r| any(r))).count() as f64 / np),
    );
    let mut transfer = serde_json::Map::new();
    for t in ["T1", "T2", "T3"] {
        let v: Vec<f64> = recs.iter().filter_map(|r| r[t].as_f64()).collect();
        transfer.insert(
            t.into(),
            mean_median(&v).map_or(
                Value::Null,
                |(m, d)| json!({ "mean": m, "median": d, "n": v.len() }),
            ),
        );
    }
    let secs: Vec<f64> = recs
        .iter()
        .filter_map(|r| r["fit_seconds"].as_f64())
        .collect();
    json!({
        "fits": recs.len(),
        "photos": photos.len(),
        "fit_flag_share": fit_share,
        "photo_flag_share": photo_share,
        "transfer": transfer,
        "mean_fit_seconds": mean_median(&secs).map(|m| m.0),
    })
}

fn pct(v: &Value) -> String {
    v.as_f64()
        .map_or("-".into(), |x| format!("{:.1}%", 100.0 * x))
}

fn tcell(v: &Value) -> String {
    if v.is_null() {
        "-".into()
    } else {
        format!(
            "{:.3} / {:.3}",
            v["mean"].as_f64().unwrap_or(f64::NAN),
            v["median"].as_f64().unwrap_or(f64::NAN)
        )
    }
}

/// The markdown of a summary.
fn summary_md(objective: &str, sets: &BTreeMap<String, Value>, per_ref: &Value) -> String {
    let mut s = format!("# Evaluation: {objective}\n\n");
    s.push_str("Harm flags and transfer metrics per set, on the full fit-size renders (`16-eval-harness.md`).\n\n");
    for (set, v) in sets {
        s.push_str(&format!(
            "## Set `{set}`: {} fits, {} photos\n\n",
            v["fits"], v["photos"]
        ));
        s.push_str("| Harm | Fits flagged | Photos with a flagged fit |\n|---|---|---|\n");
        for (i, h) in HARMS.iter().enumerate() {
            s.push_str(&format!(
                "| {h} {} | {} | {} |\n",
                HARM_NAMES[i],
                pct(&v["fit_flag_share"][h]),
                pct(&v["photo_flag_share"][h])
            ));
        }
        s.push_str(&format!(
            "| any | {} | {} |\n\n",
            pct(&v["fit_flag_share"]["any"]),
            pct(&v["photo_flag_share"]["any"])
        ));
        s.push_str("| Transfer | mean / median | fits counted |\n|---|---|---|\n");
        for (t, name) in [("T1", "colour"), ("T2", "tone"), ("T3", "cast")] {
            s.push_str(&format!(
                "| {t} {name} | {} | {} |\n",
                tcell(&v["transfer"][t]),
                v["transfer"][t]["n"]
            ));
        }
        s.push_str(&format!(
            "\nMean fit time: {:.2} s\n\n",
            v["mean_fit_seconds"].as_f64().unwrap_or(0.0)
        ));
    }
    s.push_str("## Per reference (all sets)\n\n| Reference | H1 | H2 | H3 | H4 | H5 | H6 | any | mean T1 |\n|---|---|---|---|---|---|---|---|---|\n");
    if let Some(m) = per_ref.as_object() {
        for (r, v) in m {
            s.push_str(&format!("| {r} "));
            for h in HARMS {
                s.push_str(&format!("| {} ", pct(&v["fit_flag_share"][h])));
            }
            s.push_str(&format!(
                "| {} | {} |\n",
                pct(&v["fit_flag_share"]["any"]),
                v["transfer"]["T1"]["mean"]
                    .as_f64()
                    .map_or("-".into(), |x| format!("{x:.3}"))
            ));
        }
    }
    s
}

/// A seeded shuffle (Fisher-Yates) of the indices `0..n`.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// The options of an evaluation run.
pub struct EvalOpts<'a> {
    pub split: Vec<SplitEntry>,
    pub set: String,
    pub objective: Objective,
    pub prev: Option<Objective>,
    pub references: Vec<PathBuf>,
    pub out: &'a Path,
    pub core: &'a Core,
    pub cache: &'a Path,
    pub strength: f64,
    pub with_hue: bool,
    /// `(index, count)`: this process takes the photos whose position in the selected list is `index` modulo `count`.
    pub shard: Option<(usize, usize)>,
    /// No grids: fit and measure only.
    pub no_grids: bool,
}

/// Runs the harness: every photo of the chosen set(s) against every reference.
pub fn run_eval(o: &EvalOpts<'_>) -> R<()> {
    let name = format!("{:?}", o.objective).to_lowercase();
    let dir = o.out.join(format!("eval-{name}"));
    fs::create_dir_all(&dir)?;
    let refs: Vec<Reference> = expand(&o.references, &["jpg", "jpeg", "png"])
        .iter()
        .map(|p| load_reference(p))
        .collect::<R<_>>()?;
    if refs.is_empty() {
        return Err("no references".into());
    }
    let selected: Vec<&SplitEntry> = o
        .split
        .iter()
        .filter(|e| o.set == "all" || e.set == o.set)
        .collect();
    // The test photos whose grids go to the owner: a seeded sample of 12.
    let test: Vec<usize> = selected
        .iter()
        .enumerate()
        .filter(|(_, e)| e.set == "test")
        .map(|(i, _)| i)
        .collect();
    let sample: BTreeSet<usize> = shuffled(test.len(), 20_261_124)
        .into_iter()
        .take(12)
        .map(|k| test[k])
        .collect();
    let conv = LabConv::new();
    let net = o.objective.net();
    let mut records: Vec<Value> = Vec::new();
    for (idx, e) in selected.iter().enumerate() {
        if let Some((i, n)) = o.shard
            && idx % n != i
        {
            continue;
        }
        let pname = e
            .raw
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let want_grid = !o.no_grids && (e.set == "dev" || sample.contains(&idx));
        eprintln!("== {pname} ({})", e.set);
        let photo = load_photo(o.core, &e.raw, o.cache)?;
        let labels = e.labels.as_deref().or(ctx().labels.as_deref());
        let pf = prepare_photo_with(&photo, labels)?;
        let p0_img = render_fitted(&pf, &[0.0; N_PARAMS], 1.0, o.with_hue, false, false)?;
        let p0_px = picture_px(&conv, &p0_img);
        let p0_scene = pf.scene_fit.as_ref();
        let p0 = Pic {
            px: p0_px,
            scene: p0_scene,
        };
        let p0_sample = sample_ab(&p0.px);
        let mut outcomes: Vec<PairOutcome> = Vec::new();
        let mut prevs: Vec<Option<PairOutcome>> = Vec::new();
        for r in &refs {
            let (x, secs) = if want_grid {
                let out = run_pair(&pf, r, o.strength, o.with_hue, o.objective)?;
                let (x, secs) = (out.fit.x, out.fit.seconds);
                outcomes.push(out);
                prevs.push(match o.prev {
                    Some(p) => Some(run_pair(&pf, r, o.strength, o.with_hue, p)?),
                    None => None,
                });
                (x, secs)
            } else {
                let (f, _) = run_fit(
                    &pf,
                    &r.fit,
                    o.strength,
                    o.with_hue,
                    o.objective,
                    r.scene.as_ref(),
                )?;
                (f.x, f.seconds)
            };
            let fit_img = render_fitted(&pf, &x, o.strength, o.with_hue, false, net)?;
            let fit = Pic {
                px: picture_px(&conv, &fit_img),
                scene: p0_scene,
            };
            let refp = Pic {
                px: picture_px(&conv, &r.fit),
                scene: r.scene.as_ref(),
            };
            let h = harms(&p0, &fit, &refp);
            let t = transfers(&p0, &fit, &refp, &p0_sample);
            let mut harm_json = serde_json::Map::new();
            for (k, id) in HARMS.iter().enumerate() {
                harm_json.insert((*id).into(), json!({ "value": h[k].0, "flag": h[k].1 }));
            }
            let rec = json!({
                "photo": pname, "set": e.set, "tags": e.tags, "primary_tag": e.primary_tag, "reference": r.stem,
                "fit_seconds": secs,
                "harms": harm_json,
                "any_harm": h.iter().any(|x| x.1),
                "T1": t[0], "T2": t[1], "T3": t[2],
            });
            eprintln!(
                "   {}: any harm {}, T1 {}, T2 {}, T3 {}",
                r.stem,
                h.iter().any(|x| x.1),
                t[0].map_or("-".into(), |v| format!("{v:.2}")),
                t[1].map_or("-".into(), |v| format!("{v:.2}")),
                t[2].map_or("-".into(), |v| format!("{v:.2}")),
            );
            records.push(rec);
        }
        if want_grid {
            let sub = if e.set == "dev" {
                "dev"
            } else {
                "owner-sample"
            };
            let gdir = dir.join(sub).join(&pname);
            fs::create_dir_all(&gdir)?;
            let pairs: Vec<(&Reference, &PairOutcome)> = refs.iter().zip(&outcomes).collect();
            write_grid(&gdir.join("grid.jpg"), &pf, &pairs, &prevs)?;
        }
    }
    write_summaries(&dir, &name, &mut records)
}

/// `--merge <run>...`: joins the `eval-<objective>/metrics.json` of several shard runs and writes the summaries to `out`.
pub fn merge(runs: &[PathBuf], objective: &str, out: &Path) -> R<()> {
    let mut records: Vec<Value> = Vec::new();
    for r in runs {
        let v: Value = serde_json::from_str(&fs::read_to_string(
            r.join(format!("eval-{objective}")).join("metrics.json"),
        )?)?;
        records.extend(v.as_array().cloned().unwrap_or_default());
    }
    let dir = out.join(format!("eval-{objective}"));
    fs::create_dir_all(&dir)?;
    write_summaries(&dir, objective, &mut records)
}

/// Sorts the records (so shards and one process agree) and writes `metrics.json`, `summary.json` and `summary.md`.
fn write_summaries(dir: &Path, name: &str, records: &mut [Value]) -> R<()> {
    let key = |r: &Value| {
        (
            r["set"].as_str().unwrap_or("").to_string(),
            r["photo"].as_str().unwrap_or("").to_string(),
            r["reference"].as_str().unwrap_or("").to_string(),
        )
    };
    records.sort_by_key(key);
    let mut sets: BTreeMap<String, Value> = BTreeMap::new();
    let mut by_tag: BTreeMap<String, Value> = BTreeMap::new();
    for s in ["dev", "test"] {
        let recs: Vec<&Value> = records.iter().filter(|r| r["set"] == s).collect();
        if recs.is_empty() {
            continue;
        }
        sets.insert(s.into(), summarise(&recs));
        let tags: BTreeSet<&str> = recs
            .iter()
            .map(|r| r["primary_tag"].as_str().unwrap_or("untagged"))
            .collect();
        let mut m = serde_json::Map::new();
        for t in tags {
            let tr: Vec<&Value> = recs
                .iter()
                .copied()
                .filter(|r| r["primary_tag"] == t)
                .collect();
            m.insert(t.into(), summarise(&tr));
        }
        by_tag.insert(s.into(), Value::Object(m));
    }
    let refs: BTreeSet<&str> = records
        .iter()
        .map(|r| r["reference"].as_str().unwrap_or(""))
        .collect();
    let mut per_ref = serde_json::Map::new();
    let mut per_ref_dev = serde_json::Map::new();
    for r in refs {
        let all: Vec<&Value> = records.iter().filter(|x| x["reference"] == r).collect();
        per_ref.insert(r.to_string(), summarise(&all));
        let dev: Vec<&Value> = all.iter().copied().filter(|x| x["set"] == "dev").collect();
        if !dev.is_empty() {
            per_ref_dev.insert(r.to_string(), summarise(&dev));
        }
    }
    // The 10 worst dev fits: most flagged harms, then the least colour transfer.
    let flags = |r: &Value| {
        HARMS
            .iter()
            .filter(|h| r["harms"][**h]["flag"] == true)
            .count()
    };
    let mut dev: Vec<&Value> = records.iter().filter(|r| r["set"] == "dev").collect();
    dev.sort_by(|a, b| {
        flags(b).cmp(&flags(a)).then(
            a["T1"]
                .as_f64()
                .unwrap_or(1.0)
                .total_cmp(&b["T1"].as_f64().unwrap_or(1.0)),
        )
    });
    let worst: Vec<Value> = dev.iter().take(10).map(|r| (*r).clone()).collect();
    let per_ref = Value::Object(per_ref);
    fs::write(
        dir.join("metrics.json"),
        serde_json::to_string_pretty(&*records)?,
    )?;
    fs::write(
        dir.join("summary.json"),
        serde_json::to_string_pretty(&json!({
            "objective": name, "sets": sets, "by_primary_tag": by_tag,
            "per_reference": per_ref, "per_reference_dev": per_ref_dev, "worst_dev": worst,
        }))?,
    )?;
    let mut md = summary_md(name, &sets, &per_ref);
    md.push_str("\n## Dev: by primary tag\n\n");
    md.push_str(&tag_table(by_tag.get("dev")));
    md.push_str("\n## Test: by primary tag (aggregates only)\n\n");
    md.push_str(&tag_table(by_tag.get("test")));
    md.push_str("\n## Dev: per reference\n\n");
    md.push_str(&ref_table(&Value::Object(per_ref_dev)));
    md.push_str(
        "\n## Dev: the 10 worst fits\n\n| Photo | Reference | Harms | T1 |\n|---|---|---|---|\n",
    );
    for r in &worst {
        let hs: Vec<String> = HARMS
            .iter()
            .filter(|h| r["harms"][**h]["flag"] == true)
            .map(|h| {
                format!(
                    "{h} {}",
                    r["harms"][*h]["value"]
                        .as_f64()
                        .map_or("-".into(), |v| format!("{v:.3}"))
                )
            })
            .collect();
        md.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            r["photo"].as_str().unwrap_or(""),
            r["reference"].as_str().unwrap_or(""),
            hs.join(", "),
            r["T1"].as_f64().map_or("-".into(), |v| format!("{v:.2}"))
        ));
    }
    fs::write(dir.join("summary.md"), md)?;
    Ok(())
}

fn tag_table(m: Option<&Value>) -> String {
    let mut s = String::from(
        "| Tag | Photos | Fits | H1 | H2 | H3 | H4 | H5 | H6 | any | T1 | T2 | T3 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    if let Some(o) = m.and_then(Value::as_object) {
        for (t, v) in o {
            s.push_str(&format!("| {t} | {} | {} ", v["photos"], v["fits"]));
            for h in HARMS {
                s.push_str(&format!("| {} ", pct(&v["fit_flag_share"][h])));
            }
            s.push_str(&format!("| {} ", pct(&v["fit_flag_share"]["any"])));
            for t in ["T1", "T2", "T3"] {
                s.push_str(&format!(
                    "| {} ",
                    v["transfer"][t]["mean"]
                        .as_f64()
                        .map_or("-".into(), |x| format!("{x:.3}"))
                ));
            }
            s.push_str("|\n");
        }
    }
    s
}

fn ref_table(m: &Value) -> String {
    let mut s = String::from(
        "| Reference | H1 | H2 | H3 | H4 | H5 | H6 | any | T1 |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    if let Some(o) = m.as_object() {
        for (r, v) in o {
            s.push_str(&format!("| {r} "));
            for h in HARMS {
                s.push_str(&format!("| {} ", pct(&v["fit_flag_share"][h])));
            }
            s.push_str(&format!(
                "| {} | {} |\n",
                pct(&v["fit_flag_share"]["any"]),
                v["transfer"]["T1"]["mean"]
                    .as_f64()
                    .map_or("-".into(), |x| format!("{x:.3}"))
            ));
        }
    }
    s
}

/// `--compare a b`: a side-by-side summary of two evaluation folders, written to `<out>/compare.md`.
pub fn compare(a: &Path, b: &Path, out: &Path) -> R<()> {
    let read = |p: &Path| -> R<Value> {
        Ok(serde_json::from_str(&fs::read_to_string(
            p.join("summary.json"),
        )?)?)
    };
    let (ja, jb) = (read(a)?, read(b)?);
    let label = |p: &Path| {
        p.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let (la, lb) = (label(a), label(b));
    let mut s = format!("# Compare: {la} and {lb}\n\n");
    for set in ["dev", "test"] {
        let (sa, sb) = (&ja["sets"][set], &jb["sets"][set]);
        if sa.is_null() || sb.is_null() {
            continue;
        }
        s.push_str(&format!(
            "## Set `{set}`\n\n| | {la} | {lb} |\n|---|---|---|\n"
        ));
        for (i, h) in HARMS.iter().enumerate() {
            s.push_str(&format!(
                "| {h} {} (fits flagged) | {} | {} |\n",
                HARM_NAMES[i],
                pct(&sa["fit_flag_share"][h]),
                pct(&sb["fit_flag_share"][h])
            ));
        }
        s.push_str(&format!(
            "| any harm (fits) | {} | {} |\n| any harm (photos) | {} | {} |\n",
            pct(&sa["fit_flag_share"]["any"]),
            pct(&sb["fit_flag_share"]["any"]),
            pct(&sa["photo_flag_share"]["any"]),
            pct(&sb["photo_flag_share"]["any"]),
        ));
        for t in ["T1", "T2", "T3"] {
            s.push_str(&format!(
                "| {t} mean / median | {} | {} |\n",
                tcell(&sa["transfer"][t]),
                tcell(&sb["transfer"][t])
            ));
        }
        s.push('\n');
    }
    fs::create_dir_all(out)?;
    fs::write(out.join("compare.md"), s)?;
    Ok(())
}
