//! `compare`: a new run against a baseline under the owner's three acceptance rules (`docs/34` section 10.1), the all-fits reading beside
//! the third rule, the per-tag aggregates, the agreement of the transfer numbers fit by fit and the fits that differ most.
//!
//! The runs are paired by (photo, reference); a fit one run has and the other has not is counted and left out. The baseline may be a
//! file of the proof of concept's: the records have the same shape.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Value, json};

use crate::measure::{HARM_NAMES, HARMS};
use crate::report::{any_flag, flag, mean_median};

/// A rise of a harm's share of fits that the second rule allows, in share of fits (1 point).
pub const RULE2_LIMIT: f64 = 0.01;
/// A fall of a mean transfer that the third rule allows.
pub const RULE3_LIMIT: f64 = 0.01;
/// The transfer difference within which two fits count as agreeing.
pub const AGREE_WITHIN: f64 = 0.02;
/// The tolerance of the comparisons against the limits (float noise only).
const EPS: f64 = 1e-9;

/// A pair of records of one (photo, reference).
#[derive(Debug, Clone, Copy)]
struct Pair<'a> {
    base: &'a Value,
    new: &'a Value,
}

impl Pair<'_> {
    fn photo(&self) -> &str {
        self.new["photo"].as_str().unwrap_or("")
    }
    fn reference(&self) -> &str {
        self.new["reference"].as_str().unwrap_or("")
    }
    /// A transfer value of both runs, when both have one.
    fn both(&self, t: &str) -> Option<(f64, f64)> {
        Some((self.base[t].as_f64()?, self.new[t].as_f64()?))
    }
}

/// How the comparison came out.
#[derive(Debug, Clone)]
pub struct Verdict {
    /// Rule 1 as the spec states it: the share of fits with any harm falls.
    pub rule1_falls: bool,
    /// Rule 1 as RG-10 reads it against round 15: the share does not rise.
    pub rule1_not_rise: bool,
    /// Rule 2: no harm's share rises by more than 1 point.
    pub rule2: bool,
    /// Rule 3: T1, T2 and T3 on the baseline's harm-free fits each fall by at most 0.01.
    pub rule3: bool,
    /// The markdown of the whole comparison.
    pub markdown: String,
    /// The same numbers as json.
    pub json: Value,
}

impl Verdict {
    /// Whether the three lines pass with rule 1 read as "does not rise" (RG-10's acceptance).
    #[must_use]
    pub fn passes_rg10(&self) -> bool {
        self.rule1_not_rise && self.rule2 && self.rule3
    }
}

fn pct(x: f64) -> String {
    format!("{:.1}%", 100.0 * x)
}

/// Which run of a pair a measure reads.
type Pick = for<'a, 'b> fn(&'b Pair<'a>) -> &'a Value;

fn share(pairs: &[Pair<'_>], f: impl Fn(&Value) -> bool, which: Pick) -> f64 {
    pairs.iter().filter(|p| f(which(p))).count() as f64 / pairs.len().max(1) as f64
}

fn base_of<'a>(p: &Pair<'a>) -> &'a Value {
    p.base
}

fn new_of<'a>(p: &Pair<'a>) -> &'a Value {
    p.new
}

/// Mean over the pairs with both values of transfer `t`: (baseline mean, new mean, pairs counted).
fn paired_mean(pairs: &[&Pair<'_>], t: &str) -> Option<(f64, f64, usize)> {
    let v: Vec<(f64, f64)> = pairs.iter().filter_map(|p| p.both(t)).collect();
    if v.is_empty() {
        return None;
    }
    let n = v.len() as f64;
    Some((
        v.iter().map(|x| x.0).sum::<f64>() / n,
        v.iter().map(|x| x.1).sum::<f64>() / n,
        v.len(),
    ))
}

/// The transfer difference of a pair that decides which fits differ most: the larger of |T1| and |T3| differences (a missing value
/// counts as 0).
fn spread(p: &Pair<'_>) -> f64 {
    let d = |t: &str| p.both(t).map_or(0.0, |(b, n)| (n - b).abs());
    d("T1").max(d("T3"))
}

fn flags_text(r: &Value) -> String {
    let v: Vec<&str> = HARMS.iter().copied().filter(|h| flag(r, h)).collect();
    if v.is_empty() {
        "-".into()
    } else {
        v.join(" ")
    }
}

fn opt(r: &Value, t: &str) -> String {
    r[t].as_f64().map_or("-".into(), |v| format!("{v:.3}"))
}

/// Compares `new` against `base`. `set` keeps only the fits of that set (`None`: all of them); the names label the columns.
#[must_use]
pub fn compare(
    base: &[Value],
    new: &[Value],
    set: Option<&str>,
    base_name: &str,
    new_name: &str,
) -> Verdict {
    let keep = |r: &&Value| set.is_none_or(|s| r["set"].as_str() == Some(s));
    let key = |r: &Value| {
        (
            r["photo"].as_str().unwrap_or("").to_string(),
            r["reference"].as_str().unwrap_or("").to_string(),
        )
    };
    let base_map: BTreeMap<(String, String), &Value> =
        base.iter().filter(keep).map(|r| (key(r), r)).collect();
    let new_map: BTreeMap<(String, String), &Value> =
        new.iter().filter(keep).map(|r| (key(r), r)).collect();
    let pairs: Vec<Pair<'_>> = new_map
        .iter()
        .filter_map(|(k, n)| base_map.get(k).map(|b| Pair { base: b, new: n }))
        .collect();
    let only_new = new_map.len() - pairs.len();
    let only_base = base_map.len() - pairs.len();
    let photos: std::collections::BTreeSet<&str> = pairs.iter().map(Pair::photo).collect();

    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Compare: {new_name} against {base_name}\n\nSet: {}. {} fits paired ({} photos); {only_base} only in {base_name}, {only_new} only in {new_name}.\n",
        set.unwrap_or("all"),
        pairs.len(),
        photos.len(),
    );

    // Rule 1 and 2.
    let any_b = share(&pairs, any_flag, base_of);
    let any_n = share(&pairs, any_flag, new_of);
    let rule1_falls = any_n < any_b - EPS;
    let rule1_not_rise = any_n <= any_b + EPS;
    let mut rule2 = true;
    let _ = writeln!(
        md,
        "## The three lines\n\n| Line | {base_name} | {new_name} | Change | Limit | Result |\n|---|---|---|---|---|---|"
    );
    let _ = writeln!(
        md,
        "| 1. any harm, share of fits | {} | {} | {:+.1} points | falls (RG-10: does not rise) | {} (strict) / {} (RG-10) |",
        pct(any_b),
        pct(any_n),
        100.0 * (any_n - any_b),
        if rule1_falls { "PASS" } else { "FAIL" },
        if rule1_not_rise { "PASS" } else { "FAIL" },
    );
    let mut harm_rows = Vec::new();
    for (i, h) in HARMS.iter().enumerate() {
        let (b, nn) = (
            share(&pairs, |r| flag(r, h), base_of),
            share(&pairs, |r| flag(r, h), new_of),
        );
        let ok = nn - b <= RULE2_LIMIT + EPS;
        rule2 &= ok;
        let _ = writeln!(
            md,
            "| 2. {h} {} | {} | {} | {:+.1} points | at most +1.0 | {} |",
            HARM_NAMES[i],
            pct(b),
            pct(nn),
            100.0 * (nn - b),
            if ok { "PASS" } else { "FAIL" },
        );
        harm_rows.push(json!({ "harm": h, "base": b, "new": nn, "ok": ok }));
    }
    // Rule 3, on the fits the baseline left harm-free.
    let clean: Vec<&Pair<'_>> = pairs.iter().filter(|p| !any_flag(p.base)).collect();
    let all: Vec<&Pair<'_>> = pairs.iter().collect();
    let mut rule3 = true;
    let mut t_rows = Vec::new();
    for t in ["T1", "T2", "T3"] {
        let m = paired_mean(&clean, t);
        let ok = m.is_none_or(|(b, nn, _)| nn - b >= -RULE3_LIMIT - EPS);
        rule3 &= ok;
        let _ = writeln!(
            md,
            "| 3. {t} mean on the baseline's harm-free fits ({} fits) | {} | {} | {} | at least -0.010 | {} |",
            m.map_or(0, |x| x.2),
            m.map_or("-".into(), |x| format!("{:.3}", x.0)),
            m.map_or("-".into(), |x| format!("{:.3}", x.1)),
            m.map_or("-".into(), |x| format!("{:+.3}", x.1 - x.0)),
            if ok { "PASS" } else { "FAIL" },
        );
        t_rows.push(json!({ "metric": t, "base": m.map(|x| x.0), "new": m.map(|x| x.1), "fits": m.map(|x| x.2), "ok": ok }));
    }
    let _ = writeln!(
        md,
        "\nResult: rule 1 {} (strict) / {} (does not rise), rule 2 {}, rule 3 {}. **{}**\n",
        if rule1_falls { "passes" } else { "fails" },
        if rule1_not_rise { "passes" } else { "fails" },
        if rule2 { "passes" } else { "fails" },
        if rule3 { "passes" } else { "fails" },
        if rule1_not_rise && rule2 && rule3 {
            "Accepted against RG-10's reading"
        } else {
            "Not accepted: report, do not loosen"
        }
    );

    // The all-fits reading.
    let _ = writeln!(
        md,
        "## Transfer: the all-fits reading beside the third rule\n\nMeans over the fits with a value in both runs.\n\n| Metric | Fits | {base_name} | {new_name} | Change | Harm-free fits of the baseline | {base_name} | {new_name} | Change |\n|---|---|---|---|---|---|---|---|---|"
    );
    for t in ["T1", "T2", "T3"] {
        let a = paired_mean(&all, t);
        let c = paired_mean(&clean, t);
        let _ = writeln!(
            md,
            "| {t} | {} | {} | {} |",
            a.map_or(0, |x| x.2),
            a.map_or("- | - | -".into(), |(b, nn, _)| format!(
                "{b:.3} | {nn:.3} | {:+.3}",
                nn - b
            )),
            c.map_or("0 | - | - | -".into(), |(b, nn, k)| format!(
                "{k} | {b:.3} | {nn:.3} | {:+.3}",
                nn - b
            )),
        );
    }
    let medians = |t: &str, which: Pick| -> Option<(f64, f64)> {
        let v: Vec<f64> = pairs.iter().filter_map(|p| which(p)[t].as_f64()).collect();
        mean_median(&v)
    };
    let _ = writeln!(
        md,
        "\nMean / median of each run's own values (every fit it has a value for):\n"
    );
    let _ = writeln!(md, "| Metric | {base_name} | {new_name} |\n|---|---|---|");
    for t in ["T1", "T2", "T3"] {
        let f = |m: Option<(f64, f64)>| m.map_or("-".into(), |(a, b)| format!("{a:.3} / {b:.3}"));
        let _ = writeln!(
            md,
            "| {t} | {} | {} |",
            f(medians(t, base_of)),
            f(medians(t, new_of))
        );
    }

    // Per-harm shares side by side, and the flips.
    let _ = writeln!(
        md,
        "\n## Harm shares side by side\n\n| Harm | Fits flagged {base_name} | {new_name} | Photos flagged {base_name} | {new_name} | New in {new_name} | Gone from {base_name} |\n|---|---|---|---|---|---|---|"
    );
    let photo_share = |h: Option<&str>, which: Pick| -> f64 {
        let mut by: BTreeMap<&str, bool> = BTreeMap::new();
        for p in &pairs {
            let f = h.map_or_else(|| any_flag(which(p)), |h| flag(which(p), h));
            *by.entry(p.photo()).or_default() |= f;
        }
        by.values().filter(|v| **v).count() as f64 / by.len().max(1) as f64
    };
    let mut rows: Vec<(String, Option<&str>)> = HARMS
        .iter()
        .enumerate()
        .map(|(i, h)| (format!("{h} {}", HARM_NAMES[i]), Some(*h)))
        .collect();
    rows.push(("any".into(), None));
    for (label, h) in rows {
        let f = |r: &Value| h.map_or_else(|| any_flag(r), |h| flag(r, h));
        let new_only = pairs.iter().filter(|p| f(p.new) && !f(p.base)).count();
        let gone = pairs.iter().filter(|p| !f(p.new) && f(p.base)).count();
        let _ = writeln!(
            md,
            "| {label} | {} | {} | {} | {} | {new_only} | {gone} |",
            pct(share(&pairs, f, base_of)),
            pct(share(&pairs, f, new_of)),
            pct(photo_share(h, base_of)),
            pct(photo_share(h, new_of)),
        );
    }

    // Agreement of the transfer numbers fit by fit.
    let _ = writeln!(
        md,
        "\n## Agreement fit by fit\n\nThe share of fits whose transfer is within {AGREE_WITHIN} of the baseline's (fits with a value in both).\n"
    );
    let agree = |ts: &[&str]| -> (usize, usize) {
        let mut tot = 0usize;
        let mut ok = 0usize;
        for p in &pairs {
            let ds: Vec<Option<(f64, f64)>> = ts.iter().map(|t| p.both(t)).collect();
            if ds.iter().any(Option::is_none) {
                continue;
            }
            tot += 1;
            if ds
                .iter()
                .flatten()
                .all(|(b, nn)| (nn - b).abs() <= AGREE_WITHIN)
            {
                ok += 1;
            }
        }
        (ok, tot)
    };
    let mut agree_json = serde_json::Map::new();
    let _ = writeln!(
        md,
        "| Metric | Within {AGREE_WITHIN} | Fits |\n|---|---|---|"
    );
    for (label, ts) in [
        ("T1", vec!["T1"]),
        ("T2", vec!["T2"]),
        ("T3", vec!["T3"]),
        ("T1 and T3", vec!["T1", "T3"]),
    ] {
        let (ok, tot) = agree(&ts);
        let _ = writeln!(
            md,
            "| {label} | {} | {tot} |",
            pct(ok as f64 / tot.max(1) as f64)
        );
        agree_json.insert(label.into(), json!({ "within": ok, "fits": tot }));
    }
    let same_flags = pairs
        .iter()
        .filter(|p| HARMS.iter().all(|h| flag(p.base, h) == flag(p.new, h)))
        .count();
    let _ = writeln!(
        md,
        "\nThe same set of harm flags in both runs: {} of {} fits.\n",
        same_flags,
        pairs.len()
    );

    // The fits that differ most.
    let mut ranked: Vec<&Pair<'_>> = pairs.iter().collect();
    ranked.sort_by(|a, b| {
        spread(b)
            .total_cmp(&spread(a))
            .then_with(|| (a.photo(), a.reference()).cmp(&(b.photo(), b.reference())))
    });
    let _ = writeln!(
        md,
        "## The 10 fits that differ most\n\nRanked by the larger of the T1 and T3 differences.\n\n| Photo | Reference | T1 {base_name} | {new_name} | T3 {base_name} | {new_name} | Harms {base_name} | {new_name} | Start {base_name} | {new_name} |\n|---|---|---|---|---|---|---|---|---|---|"
    );
    let mut top = Vec::new();
    for p in ranked.iter().take(10) {
        let st = |r: &Value| r["search_start"].as_str().unwrap_or("-").to_string();
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            p.photo(),
            p.reference(),
            opt(p.base, "T1"),
            opt(p.new, "T1"),
            opt(p.base, "T3"),
            opt(p.new, "T3"),
            flags_text(p.base),
            flags_text(p.new),
            st(p.base),
            st(p.new),
        );
        top.push(json!({
            "photo": p.photo(), "reference": p.reference(), "spread": spread(p),
            "T1": [p.base["T1"], p.new["T1"]], "T3": [p.base["T3"], p.new["T3"]],
        }));
    }

    // Per tag.
    let _ = writeln!(
        md,
        "\n## By primary tag\n\nAny-harm share of fits, and the transfer means on the baseline's harm-free fits.\n\n| Tag | Photos | Fits | Any harm {base_name} | {new_name} | T1 {base_name} | {new_name} | T2 {base_name} | {new_name} | T3 {base_name} | {new_name} |\n|---|---|---|---|---|---|---|---|---|---|---|"
    );
    let mut tags: BTreeMap<&str, Vec<&Pair<'_>>> = BTreeMap::new();
    for p in &pairs {
        tags.entry(p.new["primary_tag"].as_str().unwrap_or("untagged"))
            .or_default()
            .push(p);
    }
    for (tag, ps) in &tags {
        let nt = ps.len().max(1) as f64;
        let ab = ps.iter().filter(|p| any_flag(p.base)).count() as f64 / nt;
        let an = ps.iter().filter(|p| any_flag(p.new)).count() as f64 / nt;
        let cl: Vec<&Pair<'_>> = ps.iter().copied().filter(|p| !any_flag(p.base)).collect();
        let np = ps
            .iter()
            .map(|p| p.photo())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let cell = |t: &str| {
            paired_mean(&cl, t).map_or("- | -".into(), |(b, nn, _)| format!("{b:.3} | {nn:.3}"))
        };
        let _ = writeln!(
            md,
            "| {tag} | {np} | {} | {} | {} | {} | {} |",
            ps.len(),
            format_args!("{} | {}", pct(ab), pct(an)),
            cell("T1"),
            cell("T2"),
            cell("T3"),
        );
    }

    // Time.
    let secs = |which: Pick| -> Option<f64> {
        let v: Vec<f64> = pairs
            .iter()
            .filter_map(|p| which(p)["fit_seconds"].as_f64())
            .collect();
        mean_median(&v).map(|m| m.0)
    };
    let _ = writeln!(
        md,
        "\n## Fit time\n\nMean seconds per fit: {base_name} {}, {new_name} {}. (Different machines and thread counts: a reading, not a comparison.)",
        secs(base_of).map_or("-".into(), |v| format!("{v:.1}")),
        secs(new_of).map_or("-".into(), |v| format!("{v:.1}")),
    );

    let json = json!({
        "set": set, "paired": pairs.len(), "only_base": only_base, "only_new": only_new,
        "rule1": { "base": any_b, "new": any_n, "falls": rule1_falls, "does_not_rise": rule1_not_rise },
        "rule2": { "harms": harm_rows, "ok": rule2 },
        "rule3": { "metrics": t_rows, "ok": rule3 },
        "agreement": agree_json,
        "differ_most": top,
    });
    Verdict {
        rule1_falls,
        rule1_not_rise,
        rule2,
        rule3,
        markdown: md,
        json,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(photo: &str, reference: &str, harm: Option<&str>, t: [Option<f64>; 3]) -> Value {
        let mut harms = serde_json::Map::new();
        for h in HARMS {
            harms.insert(h.into(), json!({ "value": 0.0, "flag": harm == Some(h) }));
        }
        json!({
            "photo": photo, "reference": reference, "set": "dev", "primary_tag": "people",
            "harms": harms, "any_harm": harm.is_some(), "T1": t[0], "T2": t[1], "T3": t[2],
            "fit_seconds": 1.0,
        })
    }

    /// 100 fits: the baseline has harm on the first 10 of them.
    fn runs(
        new_harm: impl Fn(usize) -> Option<&'static str>,
        new_t1: impl Fn(usize, f64) -> f64,
    ) -> (Vec<Value>, Vec<Value>) {
        let mut base = Vec::new();
        let mut new = Vec::new();
        for i in 0..100 {
            let (p, r) = (format!("p{}", i / 10), format!("r{}", i % 10));
            let t1 = 0.3 + 0.002 * i as f64;
            base.push(rec(
                &p,
                &r,
                (i < 10).then_some("H2"),
                [Some(t1), Some(0.2), Some(0.4)],
            ));
            new.push(rec(
                &p,
                &r,
                new_harm(i),
                [Some(new_t1(i, t1)), Some(0.2), Some(0.4)],
            ));
        }
        (base, new)
    }

    #[test]
    fn an_identical_run_passes_the_not_rise_reading_but_not_the_strict_one() {
        let (b, n) = runs(|i| (i < 10).then_some("H2"), |_, t| t);
        let v = compare(&b, &n, Some("dev"), "base", "new");
        assert!(v.rule1_not_rise && !v.rule1_falls && v.rule2 && v.rule3);
        assert!(v.passes_rg10());
        assert!(v.markdown.contains("100 fits paired"));
        assert!(v.markdown.contains("Within 0.02"));
    }

    #[test]
    fn rule_one_falls_when_harm_goes_away_and_rule_two_allows_one_point() {
        // Harm on 10 baseline fits goes away; 1 new H4 (1 point) appears: allowed. 2 new H4 (2 points): not.
        let (b, n) = runs(|i| (i == 50).then_some("H4"), |_, t| t);
        let v = compare(&b, &n, None, "base", "new");
        assert!(v.rule1_falls && v.rule1_not_rise && v.rule2 && v.rule3);
        let (b, n) = runs(|i| (i == 50 || i == 51).then_some("H4"), |_, t| t);
        let v = compare(&b, &n, None, "base", "new");
        assert!(v.rule1_falls && !v.rule2 && !v.passes_rg10());
    }

    #[test]
    fn rule_one_fails_when_the_share_of_fits_with_harm_rises() {
        let (b, n) = runs(|i| (i < 15).then_some("H2"), |_, t| t);
        let v = compare(&b, &n, None, "base", "new");
        assert!(!v.rule1_not_rise && !v.rule1_falls);
        assert!(!v.passes_rg10());
    }

    #[test]
    fn rule_three_reads_only_the_fits_the_baseline_left_harm_free() {
        // The new run loses 0.3 of T1 on the 10 fits where the baseline had harm: not a loss.
        let (b, n) = runs(|_| None, |i, t| if i < 10 { t - 0.3 } else { t });
        let v = compare(&b, &n, None, "base", "new");
        assert!(v.rule3, "{}", v.markdown);
        // The all-fits reading still shows it.
        assert!(v.markdown.contains("-0.030"), "{}", v.markdown);
        // A loss of 0.02 on the 90 clean fits fails.
        let (b, n) = runs(|_| None, |i, t| if i >= 10 { t - 0.02 } else { t });
        let v = compare(&b, &n, None, "base", "new");
        assert!(!v.rule3);
        // A loss of exactly 0.01 passes.
        let (b, n) = runs(|_| None, |i, t| if i >= 10 { t - 0.01 } else { t });
        assert!(compare(&b, &n, None, "base", "new").rule3);
    }

    #[test]
    fn a_missing_value_in_one_run_leaves_the_fit_out_of_that_mean_only() {
        let mut b = vec![rec("p", "r1", None, [Some(0.5), Some(0.1), Some(0.2)])];
        let mut n = vec![rec("p", "r1", None, [None, Some(0.1), Some(0.2)])];
        b.push(rec("p", "r2", None, [Some(0.4), Some(0.1), Some(0.2)]));
        n.push(rec("p", "r2", None, [Some(0.4), Some(0.1), Some(0.2)]));
        let v = compare(&b, &n, None, "base", "new");
        assert!(v.rule3);
        assert_eq!(v.json["rule3"]["metrics"][0]["fits"], 1);
    }

    #[test]
    fn unmatched_fits_are_counted_and_the_set_filter_applies() {
        let mut b = vec![rec("p", "r1", None, [Some(0.5), None, None])];
        let n = vec![
            rec("p", "r1", None, [Some(0.5), None, None]),
            rec("p", "r2", None, [Some(0.5), None, None]),
        ];
        b.push({
            let mut r = rec("q", "r1", None, [Some(0.5), None, None]);
            r["set"] = json!("test");
            r
        });
        let v = compare(&b, &n, Some("dev"), "base", "new");
        assert_eq!(v.json["paired"], 1);
        assert_eq!(v.json["only_new"], 1);
        assert_eq!(v.json["only_base"], 0);
    }

    #[test]
    fn the_fits_that_differ_most_come_first() {
        let (b, n) = runs(
            |i| (i < 10).then_some("H2"),
            |i, t| if i == 77 { t - 0.5 } else { t },
        );
        let v = compare(&b, &n, None, "base", "new");
        assert_eq!(v.json["differ_most"][0]["photo"], "p7");
        assert_eq!(v.json["differ_most"][0]["reference"], "r7");
        assert!((v.json["differ_most"][0]["spread"].as_f64().unwrap_or(0.0) - 0.5).abs() < 1e-12);
        assert_eq!(v.json["agreement"]["T1"]["within"], 99);
    }
}
