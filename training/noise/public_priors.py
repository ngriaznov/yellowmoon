"""Fit per-body noise priors from public files (``docs/06-noise.md`` section "Priors from public files").

The prior-free estimator (``crates/ym-noise/examples/estimate_arw.rs``) runs over every CC0 and
free-licence ARW of the corpus. This script keeps the estimates whose smooth-region bins cover at
least 60% of the intensity range and fits, per body and gain regime:

* ``ln a = alpha ln(ISO) + beta`` where the body has files at 5 or more ISOs in the regime;
* ``b = exp(beta_b) ISO^gamma`` where it has files at 3 or more ISOs (``gamma`` limited to [0, 2]:
  read noise referred to the output grows between not at all and with the square of the gain; a
  constant per regime is ``gamma = 0``);
* nothing otherwise: the body falls back to the class prior.

ISO is the gain ISO: an ISO below the body's base ISO uses the base gain (``docs/06`` ISO quirks).
The regime of a file is decided by the body's switch ISO in ``data/cameras/noise/gain-regimes.json``
(the table is our own and to be confirmed on samples; data overrides it, so the regimes written to
the body file carry the ISO ranges they were fitted on).

Output: ``data/cameras/noise/<model-id>-<model>.json`` per qualifying body (statistics derived from
the files, never the files; ``provenance`` lists them) and a report with residual plots.

Usage, from the repository root::

    cargo build --release -p ym-noise --example estimate_arw
    python3 -m training.noise.public_priors --corpus testdata/samples --plots testdata/out/noise
    python3 -m training.noise.public_priors --estimates estimates.jsonl --dry-run

``--estimates`` reads the JSON lines of ``estimate_arw`` instead of running it (the tests use it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from training.common import jsonfmt, paths

NOISE_DIR = paths.CAMERAS_DIR / "noise"
GAIN_REGIMES_JSON = NOISE_DIR / "gain-regimes.json"
SCHEMA_VERSION = 1
DEFAULT_BASE_ISO = 100

MIN_COVERAGE = 0.60       # share of [0, 0.98] the smooth bins must cover (docs/06)
MIN_ISOS_FOR_A = 5        # distinct ISOs in a regime for the log a fit (docs/06)
MIN_ISOS_FOR_B = 3        # distinct ISOs in a regime for the b fit (docs/06)
SIGMA_LOG_A = 0.25        # prior sigma of ln a (docs/06)
SIGMA_REL_B = 0.40        # relative prior sigma of b (docs/06)
ALPHA_RANGE = (0.6, 1.4)  # a is proportional to ISO to first order; outside this the fit is not used
GAMMA_RANGE = (0.0, 2.0)  # exponent of ISO in b


# --------------------------------------------------------------------------------------------
# Fitting


@dataclass
class LogLinearFit:
    """``y = slope * x + intercept`` with ``x = ln ISO`` and ``y`` a log parameter."""

    slope: float
    intercept: float
    rms: float
    n_isos: int
    n_files: int


def fit_loglinear(x: np.ndarray, y: np.ndarray, slope_bounds: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """Least squares line through ``(x, y)``, Huber-weighted against single bad files.

    ``slope_bounds`` clamps the slope and refits the intercept. Returns ``(slope, intercept, rms)``
    where ``rms`` is the root-mean-square residual of the points.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(y)
    slope, intercept = 0.0, float(np.mean(y))
    for _ in range(20):
        sw = w.sum()
        xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
        sxx = (w * (x - xm) ** 2).sum()
        slope = (w * (x - xm) * (y - ym)).sum() / sxx if sxx > 1e-12 else 0.0
        if slope_bounds is not None:
            slope = min(max(slope, slope_bounds[0]), slope_bounds[1])
        intercept = ym - slope * xm
        r = y - (slope * x + intercept)
        s = max(1.4826 * float(np.median(np.abs(r))), 0.02)
        new_w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12))
        if np.allclose(new_w, w, atol=1e-6):
            break
        w = new_w
    r = y - (slope * x + intercept)
    return float(slope), float(intercept), float(math.sqrt(float(np.mean(r ** 2))))


def gain_iso(iso: float, base_iso: int | None) -> float:
    """Extended low ISOs use the gain of the base ISO."""
    return max(float(iso), float(base_iso or DEFAULT_BASE_ISO))


def regime_of(gain: float, switch_iso: int | None) -> int:
    """0 below the conversion-gain switch ISO, 1 at or above it (0 when the body has no switch)."""
    return 1 if switch_iso is not None and gain >= switch_iso else 0


def load_gain_regimes(path: Path = GAIN_REGIMES_JSON) -> dict[int, dict[str, Any]]:
    """``gain-regimes.json`` rows by model ID (empty when the file is missing)."""
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    return {int(b["model_id"]): b for b in doc["bodies"]}


def load_registry_names() -> dict[int, str]:
    """Display names (``Sony A1``) by model ID from the camera registry, if present."""
    if not paths.REGISTRY_JSON.is_file():
        return {}
    doc = json.loads(paths.REGISTRY_JSON.read_text())
    return {int(b["model_id"]): b["body"] for b in doc.get("bodies", [])}


def file_slug(model: str) -> str:
    """``ILCE-1`` -> ``ilce-1`` (the part of the file name after the model ID)."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def body_file_name(model_id: int, model: str) -> str:
    """``384-ilce-1.json``."""
    return f"{model_id}-{file_slug(model)}.json"


# --------------------------------------------------------------------------------------------
# Selecting estimates


def usable(est: dict[str, Any], min_coverage: float = MIN_COVERAGE) -> tuple[bool, str]:
    """Whether a per-file estimate may enter the fits, and why not."""
    if est.get("status") != "ok":
        return False, f"decode or estimate failed: {est.get('error', 'unknown')}"
    if est.get("model_id") is None or est.get("iso") in (None, 0):
        return False, "no model ID or ISO"
    if est.get("confidence") == "PriorOnly":
        return False, "nothing usable in the image"
    if est.get("coverage", 0.0) < min_coverage:
        return False, f"smooth bins cover {100 * est.get('coverage', 0.0):.0f}% of the intensity range (< {100 * min_coverage:.0f}%)"
    if est.get("filtered", {}).get("flagged"):
        return False, "filtered raw (noise is spatially correlated)"
    if not (est["fitted"]["a"] > 0):
        return False, "no positive shot-noise gain"
    return True, ""


def split_estimates(estimates: Iterable[dict[str, Any]], min_coverage: float = MIN_COVERAGE) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    """Group the usable estimates by body ``(model_id, model)``; return the excluded ones with reasons."""
    kept: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for est in estimates:
        ok, why = usable(est, min_coverage)
        if ok:
            kept[(int(est["model_id"]), str(est["model"]))].append(est)
        else:
            excluded.append({"file": est.get("file"), "model": est.get("model"), "iso": est.get("iso"), "reason": why})
    return kept, excluded


def fit_body(model_id: int, model: str, files: list[dict[str, Any]], regimes_table: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Fit one body. Returns ``{"regimes": [...], "files": [...], "notes": [...]}``.

    ``regimes`` holds only regimes with at least one fit, in the layout of the body file.
    """
    row = regimes_table.get(model_id, {})
    switch, base = row.get("switch_iso"), row.get("base_iso")
    base_iso = int(base) if base else DEFAULT_BASE_ISO
    by_regime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    listing = []
    for est in sorted(files, key=lambda e: (e["iso"], e["file"])):
        g = gain_iso(est["iso"], base_iso)
        r = regime_of(g, switch)
        by_regime[r].append({**est, "_gain": g})
        listing.append({"file": est["file"], "iso": est["iso"], "gain_iso": g, "regime": r,
                        "a": est["fitted"]["a"], "b": est["fitted"]["b"], "coverage": est["coverage"]})
    regimes: list[dict[str, Any]] = []
    notes: list[str] = []
    for r in sorted(by_regime):
        group = by_regime[r]
        isos = sorted({e["_gain"] for e in group})
        x = np.log([e["_gain"] for e in group])
        entry: dict[str, Any] = {
            "name": ("low-gain" if r == 0 else "high-gain") if switch is not None else "all",
            "iso_min": base_iso if r == 0 else int(switch),
            "iso_max": (int(switch) - 1) if (switch is not None and r == 0) else None,
            "a": None,
            "b": None,
        }
        if len(isos) >= MIN_ISOS_FOR_A:
            alpha, beta, rms = fit_loglinear(x, np.log([e["fitted"]["a"] for e in group]))
            if ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]:
                entry["a"] = {"alpha": round(alpha, 6), "beta": round(beta, 6), "sigma_log": SIGMA_LOG_A,
                              "n_isos": len(isos), "n_files": len(group), "rms_log_residual": round(rms, 6)}
            else:
                notes.append(f"regime {r}: slope of ln a against ln ISO is {alpha:.2f}, outside {ALPHA_RANGE}; no fit of a written")
        pos = [e for e in group if e["fitted"]["b"] > 0]
        b_isos = sorted({e["_gain"] for e in pos})
        if len(b_isos) >= MIN_ISOS_FOR_B:
            gamma, beta_b, rms = fit_loglinear(np.log([e["_gain"] for e in pos]), np.log([e["fitted"]["b"] for e in pos]), GAMMA_RANGE)
            entry["b"] = {"gamma": round(gamma, 6), "beta": round(beta_b, 6), "sigma_rel": SIGMA_REL_B,
                          "n_isos": len(b_isos), "n_files": len(pos), "rms_log_residual": round(rms, 6)}
        if entry["a"] or entry["b"]:
            regimes.append(entry)
        else:
            notes.append(f"regime {r}: {len(isos)} ISOs, not enough for a fit (a needs {MIN_ISOS_FOR_A}, b needs {MIN_ISOS_FOR_B})")
    return {"regimes": regimes, "files": listing, "notes": notes}


def body_document(model_id: int, model: str, body_name: str, fit: dict[str, Any], excluded: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    """The JSON document of ``data/cameras/noise/<model-id>-<model>.json``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model": model,
        "body": body_name,
        "regimes": fit["regimes"],
        "provenance": {
            "kind": "public-data prior (M2-01)",
            "method": "prior-free ym-noise estimate per file (crates/ym-noise/examples/estimate_arw.rs); "
                      "ln a = alpha ln ISO + beta per gain regime from 5 or more ISOs, b = exp(beta) ISO^gamma "
                      "from 3 or more; files kept when the smooth bins cover at least 60% of the intensity range "
                      "and the raw is not flagged as filtered",
            "script": "training/noise/public_priors.py",
            **source,
            "files": fit["files"],
            "excluded": [e for e in excluded if e.get("model") == model],
            "notes": fit["notes"],
        },
    }


# --------------------------------------------------------------------------------------------
# Running the estimator


def run_estimator(corpus: Path, binary: Path | None) -> list[dict[str, Any]]:
    """Run ``estimate_arw`` over ``corpus`` and parse its JSON lines."""
    if binary is not None and binary.is_file():
        cmd = [str(binary), str(corpus)]
    else:
        cmd = ["cargo", "run", "--release", "-q", "-p", "ym-noise", "--example", "estimate_arw", "--", str(corpus)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=paths.REPO_ROOT).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def sha256_by_name(corpus: Path) -> dict[str, str]:
    """SHA-256 of every ARW under ``corpus`` by file name (names are unique in the fetched corpus)."""
    out = {}
    for p in sorted(corpus.rglob("*")):
        if p.suffix.lower() == ".arw" and p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------
# Plots


def plot_body(model: str, fit: dict[str, Any], out_dir: Path) -> Path | None:
    """Residual plot of one body: ln a and ln b against ln ISO with the fitted lines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    files = fit["files"]
    for r_entry in fit["regimes"]:
        lo, hi = r_entry["iso_min"], r_entry["iso_max"] or 1e9
        sel = [f for f in files if lo <= f["gain_iso"] <= hi]
        if not sel:
            continue
        x = np.array([f["gain_iso"] for f in sel])
        for col, key, fitkey in ((0, "a", "a"), (1, "b", "b")):
            spec = r_entry[fitkey]
            if spec is None:
                continue
            ys = np.array([f[key] for f in sel])
            ok = ys > 0
            if fitkey == "a":
                line = spec["alpha"] * np.log(x) + spec["beta"]
            else:
                line = spec["gamma"] * np.log(x) + spec["beta"]
            axes[0][col].scatter(x[ok], ys[ok], s=14)
            xs = np.sort(x)
            axes[0][col].plot(xs, np.exp((spec["alpha"] if fitkey == "a" else spec["gamma"]) * np.log(xs) + spec["beta"]))
            axes[1][col].scatter(x[ok], np.log(ys[ok]) - line[ok], s=14)
            axes[1][col].axhline(0, lw=0.8)
    for col, name in ((0, "a"), (1, "b")):
        axes[0][col].set_xscale("log")
        axes[0][col].set_yscale("log")
        axes[0][col].set_ylabel(name)
        axes[1][col].set_xscale("log")
        axes[1][col].set_ylabel(f"ln {name} residual")
        axes[1][col].set_xlabel("gain ISO")
    fig.suptitle(f"{model}: per-file estimates and fits")
    path = out_dir / f"{file_slug(model)}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------
# Main


def build(estimates: list[dict[str, Any]], regimes_table: dict[int, dict[str, Any]], names: dict[int, str], source: dict[str, Any], min_coverage: float = MIN_COVERAGE) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """All body documents by file name, and a report dict."""
    kept, excluded = split_estimates(estimates, min_coverage)
    docs: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"bodies": [], "without_enough_data": [], "excluded": excluded, "files_total": len(estimates)}
    for (model_id, model), files in sorted(kept.items()):
        fit = fit_body(model_id, model, files, regimes_table)
        info = {"model_id": model_id, "model": model, "files": len(files), "isos": len({f["iso"] for f in files}), "regimes": fit["regimes"], "notes": fit["notes"], "fit": fit}
        if fit["regimes"]:
            docs[body_file_name(model_id, model)] = body_document(model_id, model, names.get(model_id, model), fit, excluded, source)
            report["bodies"].append(info)
        else:
            report["without_enough_data"].append(info)
    return docs, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, help="folder of ARW files (CC0 and free-licence tiers); runs the estimator over it")
    ap.add_argument("--estimates", type=Path, help="JSON lines produced by estimate_arw (skips running it)")
    ap.add_argument("--binary", type=Path, help="prebuilt estimate_arw binary (default: cargo run --release)")
    ap.add_argument("--out", type=Path, default=NOISE_DIR, help="where the body files are written (default: data/cameras/noise)")
    ap.add_argument("--plots", type=Path, help="write a residual plot per qualifying body here (testdata/out/noise is git-ignored)")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    ap.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    args = ap.parse_args(argv)

    if args.estimates:
        estimates = read_jsonl(args.estimates)
        source = {"corpus": "estimates read from " + args.estimates.name}
    elif args.corpus:
        estimates = run_estimator(args.corpus, args.binary)
        source = {"corpus": str(args.corpus), "corpus_sha256_by_file": sha256_by_name(args.corpus)}
    else:
        ap.error("give --corpus or --estimates")
    docs, report = build(estimates, load_gain_regimes(), load_registry_names(), source, args.min_coverage)

    print(f"{report['files_total']} files, {len(report['excluded'])} excluded")
    for info in report["bodies"]:
        print(f"\n{info['model']} ({info['model_id']}): {info['files']} files at {info['isos']} ISOs")
        for r in info["regimes"]:
            a, b = r["a"], r["b"]
            print(f"  {r['name']:9} ISO {r['iso_min']}-{r['iso_max'] or 'up'}: "
                  + (f"a: alpha {a['alpha']:.3f} beta {a['beta']:.3f} (rms {a['rms_log_residual']:.3f}, {a['n_isos']} ISOs)  " if a else "a: no fit  ")
                  + (f"b: gamma {b['gamma']:.2f} beta {b['beta']:.3f} (rms {b['rms_log_residual']:.3f}, {b['n_isos']} ISOs)" if b else "b: no fit"))
        for n in info["notes"]:
            print("  note:", n)
        if args.plots:
            p = plot_body(info["model"], info["fit"], args.plots)
            if p:
                print("  plot:", p)
    if report["without_enough_data"]:
        print("\nBodies without enough data for any fit (they use the class prior):")
        for info in report["without_enough_data"]:
            print(f"  {info['model']} ({info['model_id']}): {info['files']} files at {info['isos']} ISOs")
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, doc in docs.items():
            (args.out / name).write_text(jsonfmt.dumps(doc))
        print(f"\nwrote {len(docs)} body file(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
