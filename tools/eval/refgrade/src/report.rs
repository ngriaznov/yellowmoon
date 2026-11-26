//! The per-fit records of a run, their summaries (`summary.json`, `summary.md`) and the merge of shards.
//!
//! A record is the proof of concept's: `photo`, `set`, `tags`, `primary_tag`, `reference`, `fit_seconds`, `harms` (`H1` to `H6`, each
//! `{value, flag}`), `any_harm`, `T1`, `T2`, `T3` and, for the fit's winning start, `search_start`. Records are plain json values so a run
//! of the product and a file of round 15 read the same way.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::measure::{HARM_NAMES, HARMS};

/// Whether a record flags harm `h` (`"H1"` to `"H6"`).
#[must_use]
pub fn flag(r: &Value, h: &str) -> bool {
    r["harms"][h]["flag"].as_bool().unwrap_or(false)
}

/// Whether a record flags any harm.
#[must_use]
pub fn any_flag(r: &Value) -> bool {
    HARMS.iter().any(|h| flag(r, h))
}

/// Mean and median of the present values.
#[must_use]
pub fn mean_median(v: &[f64]) -> Option<(f64, f64)> {
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
#[must_use]
pub fn summarise(recs: &[&Value]) -> Value {
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
        json!(recs.iter().filter(|r| any_flag(r)).count() as f64 / n),
    );
    photo_share.insert(
        "any".into(),
        json!(
            photos
                .values()
                .filter(|v| v.iter().any(|r| any_flag(r)))
                .count() as f64
                / np
        ),
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

fn set_md(sets: &BTreeMap<String, Value>, per_ref: &Value, name: &str) -> String {
    let mut s = format!("# Evaluation: {name}\n\n");
    s.push_str("Harm flags and transfer metrics per set, on the full fit-size renders (`docs/34` section 10.1).\n\n");
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
    s.push_str("## Per reference (all sets)\n\n");
    s.push_str(&ref_table(per_ref));
    s
}

/// A table of the harm shares and the mean T1 per tag.
#[must_use]
pub fn tag_table(m: Option<&Value>) -> String {
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

/// Sorts the records by (set, photo, reference), so shards and one process agree.
pub fn sort_records(records: &mut [Value]) {
    records.sort_by_key(|r| {
        (
            r["set"].as_str().unwrap_or("").to_string(),
            r["photo"].as_str().unwrap_or("").to_string(),
            r["reference"].as_str().unwrap_or("").to_string(),
        )
    });
}

/// Writes `metrics.json`, `summary.json` and `summary.md` into `dir` from the records (sorted in place).
///
/// # Errors
///
/// Any failure to write.
pub fn write_summaries(dir: &Path, name: &str, records: &mut [Value]) -> Result<()> {
    sort_records(records);
    fs::create_dir_all(dir)?;
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
    let flags = |r: &Value| HARMS.iter().filter(|h| flag(r, h)).count();
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
    let mut md = set_md(&sets, &per_ref, name);
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
            .filter(|h| flag(r, h))
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

/// Reads the records of a run folder: `metrics.json` when it has one, else the lines of `records.jsonl` (a shard still running or
/// killed), or `dir` itself when it is a `metrics.json` file.
///
/// # Errors
///
/// A missing or unreadable file.
pub fn read_records(path: &Path) -> Result<Vec<Value>> {
    if path.is_file() {
        let v: Value = serde_json::from_str(&fs::read_to_string(path)?)
            .with_context(|| format!("{}", path.display()))?;
        return match v {
            Value::Array(a) => Ok(a),
            _ => bail!("{}: expected an array of fit records", path.display()),
        };
    }
    let json = path.join("metrics.json");
    if json.is_file() {
        return read_records(&json);
    }
    let lines = path.join("records.jsonl");
    if lines.is_file() {
        return read_jsonl(&lines);
    }
    bail!("{}: no metrics.json or records.jsonl", path.display())
}

/// The records of a `records.jsonl` file (one json object per line; a last line cut short by a kill is ignored).
///
/// # Errors
///
/// A missing or unreadable file.
pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let text = fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect())
}

/// Joins the records of several shard folders, drops a fit that two shards both made (the first wins) and writes the summaries
/// into `out`.
///
/// # Errors
///
/// A shard that cannot be read, or a failure to write.
pub fn merge(runs: &[PathBuf], name: &str, out: &Path) -> Result<usize> {
    let mut records: Vec<Value> = Vec::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for r in runs {
        for v in read_records(r)? {
            let key = (
                v["photo"].as_str().unwrap_or("").to_string(),
                v["reference"].as_str().unwrap_or("").to_string(),
            );
            if seen.insert(key) {
                records.push(v);
            }
        }
    }
    let n = records.len();
    write_summaries(out, name, &mut records)?;
    Ok(n)
}

/// The element-mask check of several shards (`masks.jsonl`: one object per photo, from `run::mask_agreement`) summarised: the share
/// of equal pixels per photo and over all photos (the pixel-weighted share and the mean of the photos), and the elements' agreement
/// and overlap. Writes `masks.json` and `masks.md` into `out`.
///
/// # Errors
///
/// Any failure to read or write.
pub fn write_mask_summary(runs: &[PathBuf], out: &Path) -> Result<Option<usize>> {
    let mut all: Vec<Value> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for r in runs {
        let f = r.join("masks.jsonl");
        if f.is_file() {
            for v in read_jsonl(&f)? {
                if seen.insert(v["photo"].as_str().unwrap_or("").to_string()) {
                    all.push(v);
                }
            }
        }
    }
    if all.is_empty() {
        return Ok(None);
    }
    all.sort_by_key(|v| v["photo"].as_str().unwrap_or("").to_string());
    let px = |v: &Value| -> f64 {
        v["size"][0].as_f64().unwrap_or(0.0) * v["size"][1].as_f64().unwrap_or(0.0)
    };
    let total: f64 = all.iter().map(px).sum();
    let weighted = |f: &dyn Fn(&Value) -> f64| -> f64 {
        all.iter().map(|v| f(v) * px(v)).sum::<f64>() / total.max(1.0)
    };
    let mean = |f: &dyn Fn(&Value) -> Option<f64>| -> Option<f64> {
        let v: Vec<f64> = all.iter().filter_map(f).collect();
        mean_median(&v).map(|m| m.0)
    };
    let equal = |v: &Value| v["equal"].as_f64().unwrap_or(0.0);
    let n99 = all.iter().filter(|v| equal(v) >= 0.99).count();
    let worst = all.iter().map(equal).fold(f64::INFINITY, f64::min);
    let mut md = String::from("# Element masks against the label maps\n\n");
    md.push_str(&format!(
        "{} photos. The element map (the product's class plane at the analysis picture against the label map resized to it, nearest) is equal on **{:.2} %** of the pixels overall (pixel-weighted), {:.2} % on average per photo, worst photo {:.2} %; {} of {} photos are at 99 % or more.\n\n",
        all.len(),
        100.0 * weighted(&equal),
        100.0 * mean(&|v| Some(equal(v))).unwrap_or(0.0),
        100.0 * worst,
        n99,
        all.len()
    ));
    let mut json_el = serde_json::Map::new();
    md.push_str("| Element | Agreement of membership (all pixels) | Mean IoU (photos with the element) | Share of the picture: product | labels |\n|---|---|---|---|---|\n");
    for (name, _) in crate::run::MASK_ELEMENTS {
        let agree = weighted(&|v| v["elements"][name]["agree"].as_f64().unwrap_or(1.0));
        let iou = mean(&|v| v["elements"][name]["iou"].as_f64());
        let ps = weighted(&|v| v["elements"][name]["product_share"].as_f64().unwrap_or(0.0));
        let ls = weighted(&|v| v["elements"][name]["labels_share"].as_f64().unwrap_or(0.0));
        md.push_str(&format!(
            "| {name} | {:.2} % | {} | {:.1} % | {:.1} % |\n",
            100.0 * agree,
            iou.map_or("-".into(), |x| format!("{x:.3}")),
            100.0 * ps,
            100.0 * ls
        ));
        json_el.insert(
            name.into(),
            json!({ "agree": agree, "mean_iou": iou, "product_share": ps, "labels_share": ls }),
        );
    }
    md.push_str("\n## Per photo\n\n| Photo | Size | Equal |\n|---|---|---|\n");
    let mut by_equal: Vec<&Value> = all.iter().collect();
    by_equal.sort_by(|a, b| equal(a).total_cmp(&equal(b)));
    for v in &by_equal {
        md.push_str(&format!(
            "| {} | {}x{} | {:.2} % |\n",
            v["photo"].as_str().unwrap_or(""),
            v["size"][0],
            v["size"][1],
            100.0 * equal(v)
        ));
    }
    fs::create_dir_all(out)?;
    fs::write(out.join("masks.md"), md)?;
    fs::write(
        out.join("masks.json"),
        serde_json::to_string_pretty(&json!({
            "photos": all.len(), "equal_weighted": weighted(&equal),
            "equal_mean_of_photos": mean(&|v| Some(equal(v))), "equal_worst": worst,
            "photos_at_99": n99, "elements": json_el, "per_photo": all,
        }))?,
    )?;
    Ok(Some(all.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(photo: &str, reference: &str, set: &str, harm: Option<&str>, t1: Option<f64>) -> Value {
        let mut harms = serde_json::Map::new();
        for h in HARMS {
            harms.insert(h.into(), json!({ "value": 0.0, "flag": harm == Some(h) }));
        }
        json!({
            "photo": photo, "set": set, "tags": ["x"], "primary_tag": "people", "reference": reference,
            "fit_seconds": 10.0, "harms": harms, "any_harm": harm.is_some(),
            "T1": t1, "T2": 0.1, "T3": null,
        })
    }

    #[test]
    fn mean_and_median() {
        assert_eq!(mean_median(&[]), None);
        assert_eq!(mean_median(&[3.0, 1.0, 2.0]), Some((2.0, 2.0)));
        assert_eq!(mean_median(&[4.0, 1.0, 2.0, 3.0]), Some((2.5, 2.5)));
    }

    #[test]
    fn the_summary_counts_fits_photos_and_transfers() {
        let rs = [
            rec("a", "r1", "dev", Some("H1"), Some(0.5)),
            rec("a", "r2", "dev", None, Some(0.3)),
            rec("b", "r1", "dev", None, None),
            rec("b", "r2", "dev", Some("H4"), Some(0.1)),
        ];
        let refs: Vec<&Value> = rs.iter().collect();
        let s = summarise(&refs);
        assert_eq!(s["fits"], 4);
        assert_eq!(s["photos"], 2);
        assert_eq!(s["fit_flag_share"]["H1"], 0.25);
        assert_eq!(s["fit_flag_share"]["any"], 0.5);
        assert_eq!(s["photo_flag_share"]["any"], 1.0);
        assert_eq!(s["photo_flag_share"]["H4"], 0.5);
        assert!((s["transfer"]["T1"]["mean"].as_f64().unwrap_or(f64::NAN) - 0.3).abs() < 1e-12);
        assert_eq!(s["transfer"]["T1"]["n"], 3);
        assert!(s["transfer"]["T3"].is_null());
        assert_eq!(s["mean_fit_seconds"], 10.0);
    }

    #[test]
    fn shards_merge_in_one_order_and_a_repeated_fit_counts_once() {
        let dir = tempfile::tempdir().expect("dir");
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        fs::create_dir_all(&a).expect("a");
        fs::create_dir_all(&b).expect("b");
        let r1 = rec("p2", "r1", "dev", None, Some(0.2));
        let r2 = rec("p1", "r1", "dev", Some("H5"), Some(0.4));
        fs::write(a.join("records.jsonl"), format!("{r1}\n{{\"cut")).expect("a");
        fs::write(b.join("records.jsonl"), format!("{r2}\n{r1}\n")).expect("b");
        let out = dir.path().join("out");
        let n = merge(&[a.clone(), b.clone()], "x", &out).expect("merge");
        assert_eq!(n, 2);
        let merged = read_records(&out).expect("records");
        assert_eq!(merged[0]["photo"], "p1");
        assert_eq!(merged[1]["photo"], "p2");
        // The other order of the shards gives the same files.
        let out2 = dir.path().join("out2");
        merge(&[b, a], "x", &out2).expect("merge");
        assert_eq!(
            fs::read_to_string(out.join("metrics.json")).expect("m1"),
            fs::read_to_string(out2.join("metrics.json")).expect("m2")
        );
        let md = fs::read_to_string(out.join("summary.md")).expect("md");
        assert!(md.contains("Set `dev`: 2 fits, 2 photos"));
    }
}
