"""Writes ``data/models/illum/MODEL_CARD.md`` from the model files and the reports in ``data/models/illum/metrics/``.

    python3 -m training.illum.model_card [--date 2026-10-12]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from training.common import paths
from training.illum import model as md
from training.illum import spectra as sp
from training.sensor_map.registry_data import PROFILES

MODELS = paths.REPO_ROOT / "data" / "models" / "illum"
STAND_IN = "sony-ilce-7rm4"


def _f(x, nd=2):
    return f"{x:.{nd}f}"


def _row(s, nd=2):
    return f"{_f(s['median'], nd)} / {_f(s['p90'], nd)} / {_f(s['max'], 1)}"


def build(date: str, models_dir: Path = MODELS) -> str:
    reps = {}
    models = {}
    for p in PROFILES:
        mp = models_dir / f"{p.id}.json"
        rp = models_dir / "metrics" / f"{p.id}.json"
        if mp.exists() and rp.exists():
            models[p.id] = md.IllumModel.from_json(json.loads(mp.read_text()))
            reps[p.id] = json.loads(rp.read_text())
    if not models:
        raise SystemExit("no models and reports found; run training.illum.run_all first")
    any_m = next(iter(models.values()))
    meta = any_m.meta
    out = []
    w = out.append
    w("# Model card: `illum`, the illuminant estimator\n")
    w(f"Version {any_m.version}, {date}. Trained with `training/illum/` at git base `{meta.get('git_base', 'unknown')[:12]}` "
      f"(working tree; training sources SHA-256 `{meta.get('training_code_sha256', 'unknown')[:16]}`). "
      "Runs in `ym_color::illum` (`crates/ym-color/src/illum.rs`), once per photo, on the CPU.\n")
    w("## Purpose\n")
    w("Estimates the colour of the scene light from the 1/8-resolution raw preview: `w_scene` (camera RGB of the light, green = 1, "
      "in the camera's raw scale), probabilities over six classes (daylight, shade/overcast, tungsten/halogen, fluorescent, LED, "
      "mixed) and a confidence (the probability that the angular error is under 2 degrees). The result drives the residual white "
      "balance (C1), the class-blended sensor map (C2) and the ALEXA white (C3) of `docs/02`. One model per camera profile.\n")
    w("## Files\n")
    w("| Profile | File | Parameters | SHA-256 of the tensors |")
    w("|---|---|---|---|")
    for pid, m in models.items():
        w(f"| `{pid}` | `{pid}.json` | {m.param_count()} | `{m.payload_sha256()}` |")
    w("\nFormat: JSON with the tensors as base64 of little-endian `f32`; `sha256` covers the tensors in `tensor_order`; "
      "the Rust loader checks it. Total under 50 k parameters per model (`docs/07`).\n")
    w("## Architecture\n")
    h = any_m.spec
    w(f"1. **Histograms.** Log-chroma `u = ln(G/R)`, `v = ln(G/B)` of every unmasked pixel (clipped pixels masked), "
      f"{h.n_bins} x {h.n_bins} bins of {h.bin_size} log-chroma units (the window is centred on the camera's light locus, 1500 to 25000 K), "
      "bilinear splatting, weight sqrt(Y) with Y = 0.25 R + 0.5 G + 0.25 B. Two channels: the pixels, and the local absolute deviation "
      "(the edges, as in FFCC). Previews are subsampled with the stride isqrt(pixels / 16384): one under 65 536 pixels is histogrammed whole, the 1080 x 720 preview of a 50 MP frame uses every 6th pixel in both directions (21 600 pixels); the 96 x 64 training scenes use all of theirs).")
    w("2. **Filter** (Barron and Tsai, FFCC, CVPR 2017; own implementation): `Z = sum_c X_c (*) F_c + B`, a circular convolution "
      "evaluated with FFTs, 2 x 64 x 64 filter weights and a 64 x 64 bias. Softmax over the bins, then the mean and covariance of the "
      "softmax map (plus a learned variance floor) are the histogram-only estimate.")
    w("3. **As-shot prior.** The camera's own white balance (tag 0x7313) as a Gaussian `N(a, 0.05^2 I)` on `(u, v)`, multiplied into "
      "the estimate (precision-weighted mean). Without the tag the estimate is the histogram's alone.")
    w("4. **Class head.** 25 features (histogram moments of both channels relative to the estimate, the estimate, the as-shot value, "
      "`Bv = log2(N^2/t) - log2(ISO/100)`, the fit's log-determinant and peak) go through standardisation, one `tanh` layer of 24 units "
      "and a softmax (temperature calibrated on validation scenes). Trained also on copies without the as-shot value and without `Bv`.")
    w("5. **Confidence.** A logistic regression (28 weights and a bias) on the class-head features and three more (the Mahalanobis distance between "
      "the histogram estimate and the as-shot value, the log variances of the fused and of the histogram-only Gaussian), fitted on the "
      "validation scenes, which have held-out lights and patches, with and without the as-shot value: the probability that the angular "
      "error of `w_scene` is under 2 degrees. (A confidence read from the Gaussian alone was tried first: the as-shot prior caps the "
      "variance, so it carried almost no information, AUC 0.57.)\n")
    w("## Training data and licences\n")
    w("Synthetic only (no photographs, no measurements of our own; `AGENTS.md` V1 data rule). Scenes are rendered through the profile's "
      "published spectral curves (Weta Digital via rawtoaces-data, Apache-2.0) in their raw channel balance.\n")
    w("| Input | Licence | Use |")
    w("|---|---|---|")
    w("| rawtoaces-data 190 training reflectances | Apache-2.0 | 132 patches train, 20 validate, 38 test (fixed split); training also uses smooth random spectra and convex mixes of the patches |")
    w("| CIE illuminants (D series, FL1-FL12, FL3.1-FL3.15, LED-B1..V2) via colour-science 0.4.7 | CC BY-SA 4.0 (CIE) | training, validation and test lights, split by light; used at training time only, only fitted weights ship. **HUMAN M1-H4 (CIE-derived data) is still open.** |")
    w("| Generated near-white LED and fluorescent spectra, blackbody and D-series formulas | ours | training, validation and test (disjoint seed streams) |")
    w("| CIE 2017 CES 99, ColorChecker (Ohta), NIST CQS VS | CC BY-SA 4.0 / public data | **evaluation only**, never trained on |")
    w("| Cube++ (CC BY 4.0) | CC BY 4.0 | allowed for this model but **not used**: zenodo.org is unreachable from the build environment (`tools/fetch/cubepp.sh`, `training/illum/eval_cubepp.py` are ready) |")
    w("\nSplits are by light **and** by reflectance: `train`, `val` and `test` never share a light (held-out blackbody and D-series "
      "temperature bands, CIE fluorescent and LED spectra, generated spectra from disjoint seed streams; the nine sensor-map gate "
      "lights are all in `test`) or a reflectance patch (`training/illum/test_illum.py` checks both).\n")
    w("Scene model: 96 x 64 preview with 6 to 40 soft-edged cells, per-cell brightness, shading, texture, optional highlights "
      "and grey patches, dominant-colour scenes, mixed lighting (two lights with a smooth spatial weight; the truth is the "
      "brightness-weighted mean light); Poisson-Gaussian noise of nominal sensor parameters at an ISO tied to `Bv`, binned 8 x 8; "
      "per-channel clipping. **The as-shot white balance is simulated** (50 % good: a Gaussian error of 0.04 in log-chroma; 27 % off: 0.10; "
      "8 % bad: 0.25; 11 % an unrelated preset; 4 % missing; a median error of about 1.6 degrees against the truth); that distribution is an "
      "assumption, not a measurement. Training draws reflectances from a broader mix (50 % smooth random spectra, 20 % single patches, "
      "30 % convex mixes, all with a smooth perturbation) than validation and test (35 / 30 / 35, a milder perturbation), so that the "
      "held-out scenes measure generalisation to patches the model has not seen; with the narrower training mix the held-out median was "
      "2.0 degrees histogram-only, with unmixed patches 4 degrees: a model memorises its 132 patches.\n")
    tr_ = any_m.meta.get("recipe", {})
    w(f"Recipe: {tr_.get('n_train', '?')} training scenes per profile, {tr_.get('steps', '?')} Adam steps (batch {tr_.get('batch', '?')}), "
      "losses = softmax cross-entropy + Gaussian negative log-likelihood + L1-like error for the histogram-only and the fused estimate, "
      "total-variation and L2 regularisers on filter and bias (as FFCC).\n")

    w("## Metrics (held-out synthetic scenes)\n")
    w("Angular error in degrees between `w_scene` and the true camera white of the light. Gate (`docs/02`, `docs/13`): median <= 2 degrees on "
      "held-out synthetic scenes. `test` = held-out lights and held-out reflectances; the gate figure is the estimator as shipped, "
      "with the as-shot prior; histogram-only is reported because it is what a file without the white-balance tag gets.\n")
    w("| Profile | test, with prior: median / p90 / worst | test, histogram only: median / p90 / worst | CES 99 (eval only), with prior | gate (median <= 2) |")
    w("|---|---|---|---|---|")
    for pid, r in reps.items():
        g = r["gates"]
        w(f"| `{pid}` | {_row(r['test']['with_prior']['all'])} | {_row(r['test']['histogram_only']['all'])} | "
          f"{_row(r['ces']['with_prior']['all'])} | {'pass' if g['pass_with_prior'] else '**FAIL**'} (hist. only: {'pass' if g['pass_histogram_only'] else '**FAIL**'}) |")
    if STAND_IN in reps:
        r = reps[STAND_IN]
        w("\n### A7R IV in detail\n")
        w("**Against simpler estimators** (test set, same scenes; median / p90 / worst):\n")
        w("| Estimator | median / p90 / worst |")
        w("|---|---|")
        b = r["baselines_test"]
        for name, label in (("grey_world", "grey world"), ("white_patch", "white patch (99th percentile)"),
                            ("shades_of_grey_p6", "shades of grey (p = 6)"), ("grey_edge", "grey edge"),
                            ("as_shot_only", "the simulated as-shot white balance alone"),
                            ("grey_world_plus_as_shot_prior", "grey world + the same as-shot prior")):
            w(f"| {label} | {_row(b[name])} |")
        w(f"| **FFCC filter, histogram only** | {_row(r['test']['histogram_only']['all'])} |")
        w(f"| **FFCC filter + as-shot prior (shipped)** | {_row(r['test']['with_prior']['all'])} |")
        w("\n**By class** (test, with prior | histogram only; median / p90):\n")
        w("| Class | with prior | histogram only |")
        w("|---|---|---|")
        for c in md.CLASSES:
            a, h_ = r["test"]["with_prior"]["by_class"][c], r["test"]["histogram_only"]["by_class"][c]
            w(f"| {c} | {_f(a['median'])} / {_f(a['p90'])} | {_f(h_['median'])} / {_f(h_['p90'])} |")
        w("\n**By light family** (test, histogram only; median / p90, n):\n")
        w("| Family | median / p90 | n |")
        w("|---|---|---|")
        for fam, s in sorted(r["test"]["histogram_only"]["by_family"].items(), key=lambda kv: -kv[1]["n"]):
            if s["n"] >= 25:
                w(f"| {fam} | {_f(s['median'])} / {_f(s['p90'])} | {s['n']} |")
        w("\n**By ISO** (test, histogram only; median / p90):\n")
        w("| ISO | median / p90 |")
        w("|---|---|")
        for k, s in r["test"].get("by_iso_histogram_only", {}).items():
            label = {"iso_100_800": "100 to 800", "iso_800_6400": "800 to 6400", "iso_6400_up": "6400 and up"}.get(k, k)
            w(f"| {label} | {_f(s['median'])} / {_f(s['p90'])} |")
        cm = r["test"]["with_prior"]["classes"]
        w(f"\n**Classes** (test, with prior): accuracy {_f(100 * cm['accuracy'], 1)} % over the six classes, "
          f"{_f(100 * cm['accuracy_4class_no_mixed'], 1)} % over the four profile classes (daylight and shade merged, mixed excluded), "
          f"negative log-likelihood {_f(cm['nll'], 3)}. Rows true, columns predicted ({', '.join(md.CLASSES)}):\n")
        w("```")
        for row in cm["confusion"]:
            w(" ".join(f"{v:5d}" for v in row))
        w("```")
        cal = r["test"]["with_prior"]["calibration"]
        cal_h = r["test"]["histogram_only"]["calibration"]
        w(f"\n**Confidence** (test, with prior): mean {_f(cal['mean_confidence'], 3)}, fraction of scenes under 2 degrees {_f(cal['fraction_within_2deg'], 3)}, "
          f"Brier {_f(cal['brier'], 3)} (the constant base rate gives {_f(cal['brier_base_rate'], 3)}), AUC {_f(cal['auc'], 3)}; histogram only: "
          f"Brier {_f(cal_h['brier'], 3)} (base rate {_f(cal_h['brier_base_rate'], 3)}), AUC {_f(cal_h['auc'], 3)}. Quintiles of confidence "
          "against the fraction of scenes under 2 degrees (with prior):\n")
        w("| mean confidence | fraction under 2 degrees | n |")
        w("|---|---|---|")
        for q in cal["bins"]:
            w(f"| {_f(q['mean_confidence'], 3)} | {_f(q['fraction_within_2deg'], 3)} | {q['n']} |")
        if "cross_camera" in r:
            w("\n**Stand-in sensitivity.** The A7R IV model on scenes rendered through the other bodies' curves (the situation of the A1 "
              "and every body without a profile, and of a body whose sensor differs from its profile): median / p90 / worst.\n")
            w("| Body | with prior | histogram only |")
            w("|---|---|---|")
            for pid, s in r["cross_camera"].items():
                w(f"| `{pid}` | {_row(s['with_prior'])} | {_row(s['histogram_only'])} |")

    w("\n## The sensor-map gate with estimated lights\n")
    w("`training/illum/gate_estimated.py` re-runs the M1-05 gate (99 CES, nine held-out lights, tables refitted without LED-B3, LED-RGB1 "
      "and LED-V2) with the light taken from the estimator on synthetic scenes lit by each gate light (A7R IV: 200 scenes per light): "
      "the camera is balanced with the estimated white, the map is `M(probs, mired)` with the six classes folded into the four and the "
      "mired of the CCT read from the profile's locus, ARRI's matrix is taken at that CCT; the reference is the ALEXA rendering under the "
      "true light. dE00 median / p95 pooled over scenes and patches, against the known-light gate. Gate: median <= 0.5, p95 <= 1.2.\n")
    w("Three modes split the error by what the estimate is allowed to change: **full** (everything from the estimate: the gate as worded), "
      "**map only** (the camera is balanced with the true white; the estimated classes, mired and CCT choose the map and ARRI's matrix: "
      "what the estimator costs the sensor map) and **white only** (the estimated white balances the camera; the true class and CCT choose "
      "the map: what the white-balance error costs).\n")
    if STAND_IN in reps:
        gt = reps[STAND_IN]["sensor_map_gate_estimated"]
        for key, label in (("full/with_prior", "Full, with the as-shot prior (as shipped)"), ("full/histogram_only", "Full, histogram only"),
                           ("map_only/with_prior", "Map only, with the as-shot prior"), ("white_only/with_prior", "White only, with the as-shot prior")):
            if key not in gt:
                continue
            r = gt[key]
            w(f"\n**{label}**, A7R IV (angular error median / p90 of the estimate, dE00 median / p95 estimated, dE00 median / p95 known light):\n")
            w("| Light | class | angular | estimated | known light | |")
            w("|---|---|---|---|---|---|")
            for name, L in r["lights"].items():
                e, k, a = L["estimated"], L["known_light"], L["angular_error_deg"]
                bad = e["median"] > 0.5 or e["p95"] > 1.2
                w(f"| {name} | {L['class']} | {_f(a['median'])} / {_f(a['p90'])} | {_f(e['median'], 3)} / {_f(e['p95'], 3)} | "
                  f"{_f(k['median'], 3)} / {_f(k['p95'], 3)} | {'over' if bad else ''} |")
            w(f"\nWorst median {_f(r['worst_median'], 3)}, worst p95 {_f(r['worst_p95'], 3)}; lights over the gate: "
              f"{', '.join(r['lights_over']) if r['lights_over'] else 'none'}.\n")
    w("\nAll profiles (with the as-shot prior), worst median / worst p95 over the nine lights and the lights over the gate. The known-light "
      "column is the M1-05 gate with the same tables.\n")
    w("| Profile | full | map only | known light |")
    w("|---|---|---|---|")
    for pid, r in reps.items():
        g = r["sensor_map_gate_estimated"]
        f_, m_ = g.get("full/with_prior", g.get("with_prior")), g.get("map_only/with_prior")
        km = max(L["known_light"]["median"] for L in f_["lights"].values())
        kp = max(L["known_light"]["p95"] for L in f_["lights"].values())
        mo = f"{_f(m_['worst_median'], 3)} / {_f(m_['worst_p95'], 3)} ({len(m_['lights_over'])} lights over)" if m_ else "-"
        w(f"| `{pid}` | {_f(f_['worst_median'], 3)} / {_f(f_['worst_p95'], 3)} ({len(f_['lights_over'])} over) | {mo} | {_f(km, 3)} / {_f(kp, 3)} |")
    if STAND_IN in reps and "map_only/with_prior" in reps[STAND_IN]["sensor_map_gate_estimated"]:
        gt = reps[STAND_IN]["sensor_map_gate_estimated"]
        full, mo = gt["full/with_prior"], gt["map_only/with_prior"]
        per_deg = [(L["estimated"]["median"] - L["known_light"]["median"]) / L["angular_error_deg"]["median"] for L in full["lights"].values()]
        dm = [L["estimated"]["median"] - L["known_light"]["median"] for L in mo["lights"].values()]
        dp = [L["estimated"]["p95"] - L["known_light"]["p95"] for L in mo["lights"].values()]
        margins = [1.2 - L["known_light"]["p95"] for L in mo["lights"].values() if L["known_light"]["p95"] < 1.2]
        p95_over = [n for n, L in mo["lights"].items() if L["estimated"]["p95"] > 1.2]
        med_over = [n for n, L in mo["lights"].items() if L["estimated"]["median"] > 0.5]
        w("\n**Reading (A7R IV).** Estimated end to end, the gate fails for every light: the white-balance error of the estimate "
          "(a median of about 1 to 1.5 degrees per light, 4 to 8 degrees at p90) shows as a colour cast and costs "
          f"{_f(min(per_deg))} to {_f(max(per_deg))} dE00 in the median per degree of median angular error, an order of magnitude more than "
          "the sensor map's own error; a median of 0.5 dE00 would need the white to be right to about 0.3 degrees, far below what illuminant "
          "estimators reach on photographs (typically 1.5 to 3 degrees). The estimator's classes and mired alone (map only, white known) "
          f"cost {_f(min(dm))} to {_f(max(dm))} dE00 in the median and {_f(min(dp))} to {_f(max(dp))} in the p95: the median stays under 0.5 "
          f"for every light that passed with the light known except {', '.join(n for n in med_over) or 'none'}, and the p95 limit of 1.2, which the "
          f"known-light fit meets with margins of {_f(min(margins))} to {_f(max(margins))}, is exceeded on {len(p95_over)} of the nine "
          "(LED-V2 is the accepted M1-05 failure either way).\n")
    w("\n## Failure cases and limits\n")
    w("* **Mixed lighting** has no single illuminant; the truth here is the brightness-weighted mean and the error is the largest of all classes.")
    w("* **Dominant-colour scenes** (a frame filled with one hue) shift the histogram and the estimate; the as-shot prior limits the damage when the camera's own estimate is sane.")
    if STAND_IN in reps and "preset" in reps[STAND_IN]["test"]["with_prior"]["by_as_shot"]:
        pw = reps[STAND_IN]["test"]["with_prior"]["by_as_shot"]["preset"]
        ph = reps[STAND_IN]["test"]["histogram_only"]["by_as_shot"]["preset"]
        preset = (f" (A7R IV, test scenes with such a preset: median {_f(pw['median'])} / p90 {_f(pw['p90'], 1)} degrees with the prior, "
                  f"{_f(ph['median'])} / {_f(ph['p90'], 1)} without it; `training/illum/robust_prior_study.py` shows what a two-component prior would change)")
    else:
        preset = ""
    w("* **A wrong as-shot white balance** (a preset that does not match the light, 11 % of the synthetic scenes) pulls the fused estimate: "
      f"the prior is a Gaussian with sigma 0.05 as specified, not a robust one{preset}. Without the tag the histogram alone gives the histogram-only figures.")
    w("* **The histogram-only median is close to the 2-degree line** for some bodies (see the table; on freshly drawn test scenes the A7R II gave 1.98 and 2.08, and the validation scenes of the A7R IV 2.18): the figure that passes with margin is the one with the as-shot prior, which is how the estimator ships, and it rests on the assumed as-shot error model.")
    w("* **The brightness prior is five stops low for sunlit scenes.** The synthetic `Bv = log2(N^2/t) - log2(ISO/100)` runs from -2 to 13 (daylight mean 9.3); real sunlit photographs read 14 to 15 (four of the nine real files in `testdata/real/files`). The class head extrapolates in the right direction there (more daylight), but it never saw such values.")
    w("* **Daylight and shade/overcast** overlap in temperature and in brightness; the class head separates them only statistically. Both fold into the daylight class of the sensor map, so a confusion between them costs no colour accuracy.")
    w("* **Fluorescent versus LED** is the weakest class pair (see the confusion matrix); spiky spectra are separated mostly by their distance from the locus.")
    w("* **Real photographs were never seen.** The scenes are synthetic, the class and brightness priors are assumptions, and the as-shot error model is assumed. The Cube++ gate (<= 3 degrees) and the NUS-8 Sony report are blocked here (no access to zenodo.org or the NUS-8 host) and have not been run.")
    w("* **Stand-in profiles.** A body that shares a stand-in profile with a different sensor sees raw colours the model was not trained on; the cross-camera table above bounds the cost.")
    w("* **Off-locus lights** (coloured stage lighting, sodium or mercury lamps) are outside the training distribution (generated lights are within 0.012 of the locus).")
    w("* **The 4000 K junction** of the profile locus makes CCT and Duv unreliable between about 3800 and 4300 K (`docs/03`); the estimator's `w_scene` does not depend on it, but the CCT that is read from it does.\n")
    w("## Provenance\n")
    w(f"* Date {date}; model version {any_m.version}; training code at git base `{meta.get('git_base', 'unknown')}` plus the working tree "
      f"(`training_code_sha256` {meta.get('training_code_sha256', 'unknown')}).")
    w("* Reproduce: `pip install -r training/requirements.txt`, `tools/fetch/spectral.sh`, `python3 -m training.illum.run_all`, "
      "`python3 -m training.illum.model_card`, `python3 -m training.illum.export_vectors`.")
    w("* Rust parity: `crates/ym-color/tests/illum.rs` against `tests/data/illum/` (1e-9).")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    out = args.out or args.models_dir / "MODEL_CARD.md"
    out.write_text(build(args.date, args.models_dir))
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(args.models_dir.glob("sony-*.json"))]
    (args.models_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
