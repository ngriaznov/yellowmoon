"""Writes the model card of ``autoset`` from the metrics document of ``train.py`` and from the model file.

    python3 -m training.autoset.model_card --metrics metrics.json --model data/models/autoset/autoset.json \\
        [--date 2026-10-18] [--out data/models/autoset/MODEL_CARD.md]

The default output is ``data/models/autoset/MODEL_CARD.md``; nothing is written there unless this command is run without ``--out``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path
from typing import Any

from training.autoset import model as md
from training.common import paths

DEFAULT_OUT = paths.REPO_ROOT / "data" / "models" / "autoset" / "MODEL_CARD.md"


def _num(x: Any, nd: int = 2) -> str:
    if x is None:
        return "-"
    return f"{float(x):.{nd}f}"


def _pct(x: Any) -> str:
    return "-" if x is None else f"{100.0 * float(x):.1f} %"


def _stats_row(label: str, s: dict, *, unit: str = "EV", tol_key: str = "within_0p5") -> str:
    if not s.get("n"):
        return f"| {label} | 0 | - | - | - | - |"
    return (f"| {label} | {s['n']} | {_num(s['mae'])} | {_num(s['median_abs'])} | {_num(s['p90_abs'])} | "
            f"{_pct(s.get(tol_key))} |")


def build(metrics: dict, model_doc: dict | None, *, date: str, git_sha: str) -> str:
    """The card as Markdown (British spelling, as the rest of the documentation)."""
    ev_doc = metrics.get("evaluation") or {}
    data = metrics.get("data") or {}
    config = metrics.get("config") or {}
    sources = metrics.get("sources") or []
    final = metrics.get("final_model") or {}
    model = final.get("model") or {}
    version = (model_doc or {}).get("model_version", model.get("model_version", "unknown"))
    sha = (model_doc or {}).get("sha256", model.get("sha256", "not written"))
    members = int(config.get("members", md.N_MEMBERS))
    per_member = _params_per_member(model_doc) if model_doc else model.get("parameters_per_member")
    out: list[str] = []
    w = out.append

    w("# Model card: `autoset`, exposure and contrast\n")
    w(f"Version {version}, {date}. Trained with `training/autoset/` at git commit `{git_sha[:12]}`. Runs in "
      "`ym_color::autoset` (`crates/ym-color/src/autoset.rs`, AS-03), once per photo, on the CPU, in float64.\n")
    w("## Purpose\n")
    w("Decides where the key of a photo sits and how much contrast it gets, from the photo's brightness statistics "
      "(`docs/18` section 3). The exposure head predicts the key target (where the key should sit after exposure, in stops over "
      "middle grey); the file's exposure is that target minus the key, so a photo shot a stop under or over comes out the same "
      "(full rescue). The exposure applies in full; its weight is reported as a confidence only. The contrast head predicts the "
      "C9 contrast, shrunk towards 0 when the ensemble is unsure (section 4.4). Settings are chosen per photo; nothing is stored "
      "per photo.\n")
    w("## Files\n")
    w("| File | Model version | Members | Parameters per member | SHA-256 of the member tensors |")
    w("|---|---|---|---|---|")
    w(f"| `autoset.json` | {version} | {members} | {per_member if per_member is not None else '-'} | `{sha}` |\n")
    w("Format: `docs/18` section 4.3 (JSON, tensors as base64 little-endian float32, checksum over the member tensors in "
      "order). Inputs are standardised with the stored mean and standard deviation; the input statistics and the constants are "
      "not covered by the checksum.\n")
    w("## Architecture\n")
    w(f"{members} multilayer perceptrons, each 39 to 32 to 32 to 4 with `tanh` hidden units and a linear output (key target, key "
      "log-variance, contrast, contrast log-variance). The combination is the mean of the means; the variance is the mean of the "
      "member variances plus the population variance of the means. Prior sigmas 0.75 (key target) and 10 (contrast); contrast "
      "limit 30. Weights `w = s0^2 / (s0^2 + variance)`, applied as in `docs/18` section 4.4.\n")
    w("## Training data and licences\n")
    w("Dataset rows (`docs/18` section 6): one line per file with its 39 features, the label tables and the augmentations (scaled "
      "by 2^k, k in -2, -1, -0.5, +0.5, +1, +2; the file's exposure label moved by -k and its contrast label). Every file and augmentation of a scene is in the "
      "same split.\n")
    licences = sorted({s["licence"] for s in sources}) if sources else []
    if licences and all(lic == "CC0-1.0" for lic in licences):
        w("**Licence of the weights:** Apache-2.0, as the rest of Yellowmoon: every training file is CC0 (`docs/14` rule 5).\n")
    elif licences:
        w(f"**Licence of the weights:** check `docs/14` rule 5 for the training licences {', '.join(licences)}; weights trained on "
          "CC BY-SA data are CC BY-SA 4.0 with their own `LICENSE` in this folder.\n")
    if sources:
        w("| Source | Licence | Files | Scenes | Bodies |")
        w("|---|---|---|---|---|")
        for s in sources:
            w(f"| {s['source']} | {s['licence']} | {s['files']} | {s['scenes']} | {len(s.get('bodies', []))} |")
        w("")
    else:
        w("No source summary in the metrics file.\n")
    w("## Labels\n")
    w("* **Exposure (`docs/18` 6.1).** The G3 fit of the auto-exposure study (`tools/eval/metrics.py`): the scene level at which the "
      "camera's JPEG reaches mid grey, by quantile matching over the 0.02 to 0.98 quantiles, restricted to the mid-tone band of "
      "JPEG luminance 0.045 to 0.36 (-2 to +1 stop around grey; unrestricted when fewer than 20 points remain). Variance 0.05 EV "
      "squared, times 4 for DRO, HDR or a non-Standard creative style, plus the fit's residual.\n"
      "* **Contrast (`docs/18` 6.2).** The contrast whose display-luma p90 - p10 and p75 - p25 spreads best match the camera's, "
      "linearly interpolated between the grid of -40 to +40 in steps of 10, clamped to +-40. Variance from the residual, floor 4.\n"
      "* **Key target** = exposure label + key.\n")
    w(f"Files with an exposure label: {data.get('files_with_key_label', '-')} of {data.get('files', '-')}; with a contrast label: "
      f"{data.get('files_with_contrast_label', '-')}; normal scenes: {data.get('files_normal', '-')}; flagged (DRO, HDR or style): "
      f"{data.get('files_flagged', '-')}.\n")
    w("## Training\n")
    w(f"Full-batch Adam, learning rate {config.get('lr', '-')}, {config.get('steps', '-')} steps at most, L2 weight decay "
      f"{config.get('weight_decay', '-')}, early stopping on the held-out 15 % of scenes (40 evaluations without a new best). "
      f"Members are seeded {config.get('member_seed_base', '-')} and up; the scene split uses seed {config.get('seed', '-')}.\n")
    w("## Metrics\n")
    w("EV error is `EV_model - L` in EV, where `EV_model = key_target - key`. Gates are `docs/18` section 8. The constant rule is "
      "`EV = 1.11`, the camera's gain.\n")
    if ev_doc:
        ev = ev_doc.get("ev", {})
        w("| Rows | n | MAE | median | p90 | within 0.5 EV |")
        w("|---|---|---|---|---|---|")
        w(_stats_row("held-out files, model", ev.get("heldout_files", {})))
        w(_stats_row("normal files, model (gate 3)", ev.get("normal_files", {})))
        w(_stats_row("held-out files, constant 1.11", ev.get("constant_heldout_files", {})))
        w(_stats_row("normal files, constant 1.11", ev.get("constant_normal_files", {})))
        aug = ev_doc.get("augmentation", {})
        w("\n| Augmentations | n | MAE | median | p90 | share |")
        w("|---|---|---|---|---|---|")
        resc = aug.get("rescue_vs_label", {})
        w(_stats_row("rescue: |EV - label| within 0.3 EV (gate 4)", resc, tol_key="within_0p3"))
        eq = aug.get("equivariance", {})
        w(_stats_row("equivariance: |EV(aug) - (EV(base) - k)| within 0.05 (gate 2)", eq, tol_key="within_0p05"))
        con = ev_doc.get("contrast", {})
        w("\n| Contrast | n | MAE | median | p90 | within 5 units |")
        w("|---|---|---|---|---|---|")
        for label, key in (("model, clamped to +-30", "model_vs_label"), ("constant 0 (the rule before autoset)", "constant_zero_vs_label")):
            s = con.get(key, {})
            if s.get("n"):
                w(f"| {label} | {s['n']} | {_num(s['mae'], 1)} | {_num(s['median_abs'], 1)} | {_num(s['p90_abs'], 1)} | "
                  f"{_pct(s.get('within_5'))} |")
        cal = ev_doc.get("calibration", {})
        if cal:
            w("\nCalibration (share of labels within the predictive spread): " + ", ".join(
                f"{k.replace('_', ' ')} {_pct(v)}" for k, v in sorted(cal.items())) + ".\n")
    else:
        w("No evaluation in the metrics file.\n")
    w("## Failure cases\n")
    failures = ev_doc.get("failures") or []
    if failures:
        w("The ten largest EV errors on held-out files (file names as in the dataset):\n")
        w("| File | Body | Label EV | Model EV | Error EV | Flagged | Normal |")
        w("|---|---|---|---|---|---|---|")
        for f in failures:
            w(f"| {f['file']} | {f.get('body') or '-'} | {_num(f['level_ev'])} | {_num(f['ev_model'])} | "
              f"{_num(f['error_ev'])} | {'yes' if f.get('flagged') else 'no'} | {'yes' if f.get('normal') else 'no'} |")
        w("")
    else:
        w("None recorded.\n")
    w("## Limits\n")
    w("* The exposure label is the camera's own tone placement, so the model learns the camera's choice where its JPEG is "
      "tone-mapped (DRO, HDR, creative styles); those labels carry double variance.\n"
      "* The contrast label partly measures the camera's tone curve, which is why its head is bounded at +-30 and shrunk.\n"
      "* Nothing here was judged by eye. The blind A/B against Lightroom Auto and the camera JPEG (`docs/18` gate 5, AS-H1) decides "
      "whether the model is better than the rule before it.\n")
    return "\n".join(out) + "\n"


def _params_per_member(model_doc: dict) -> int | None:
    """Weights and biases of one member, counted from the file's shapes."""
    try:
        return sum(math.prod(layer[name]["shape"]) for layer in model_doc["members"][0]["layers"] for name in ("w", "b"))
    except (KeyError, IndexError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--model", type=Path, help="the model file (for the version, the checksum and the parameter count)")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--git-sha", default=paths.git_head_sha())
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    model_doc = json.loads(args.model.read_text(encoding="utf-8")) if args.model else None
    text = build(metrics, model_doc, date=args.date, git_sha=args.git_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
