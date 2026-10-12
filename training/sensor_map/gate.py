"""Sensor-map gate (docs/02 §C2 "Gate", docs/13, milestone M1-05), Python side.

99 CIE 2017 CES reflectances (colour-science's CFI 2017 dataset, CC BY-SA 4.0, test only)
under 9 held-out lights. For each profile:

1. Refit the tables with LED-B3, LED-RGB1 and LED-V2 removed from the LED class (the other
   classes keep all their lights; none of the nine gate lights is in a fit grid).
2. For each gate light, take the map the runtime would use **with the true light known**:
   ``M = M_class(mired)`` where ``class`` is the light's own class with probability 1 and
   ``mired = 1e6 / cct_k`` with ``cct_k`` the true CCT. The CCT is the generating temperature
   for the blackbody and D-series lights (2850, 5250, 6750 K) and the Ohno (2013) CCT of the
   SPD for the fluorescent and LED lights. ``M_class`` is linear in mired between the two
   bracketing fitted lights of that class and clamped at its ends
   (:func:`training.sensor_map.fit.blend_map`). This isolates the table blend; the illuminant
   estimator's errors are the separate M1-06 gate ("the sensor-map gate still passes with
   estimated lights").
3. Render the 99 reflectances through the Sony curves (white-balanced to the light), map them
   with ``M``, and compare with the same reflectances rendered through ``S_A`` directly (white
   balanced to the light). Both go through ARRI's matrix at ``cct_k``. ΔE00, ACES white.

Gate: per light median <= 0.5 and p95 <= 1.2 for the A7R IV profile. The other ten profiles
are reported.

Extra columns for context (not gated): ``identity`` (no mapping, M = I) and ``fit_at_light``
(the 3x3 fitted at the gate light itself, the lower bound a better blend could reach), and the
same blend from the shipped fit (all reference lights, which puts LED-B3, LED-RGB1 and LED-V2
exactly on grid).

    python3 -m training.sensor_map.gate [--profile sony-ilce-7rm4] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.common import paths
from training.sensor_map import evaluate as ev
from training.sensor_map import fit
from training.sensor_map import lights as lt
from training.sensor_map.registry_data import PROFILES

MEDIAN_MAX = 0.5
P95_MAX = 1.2
GATE_PROFILE = "sony-ilce-7rm4"


def _stats(d: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(np.max(d))}


def run_profile(profile_id: str, ctx: ev.EvalContext | None = None) -> dict:
    """Gate numbers for one profile: {light: {...}} plus pass/fail."""
    ctx = ev.context() if ctx is None else ctx
    prof = next(p for p in PROFILES if p.id == profile_id)
    S_P = fit.sony_curves()[prof.curves_name]
    gate_tables = fit.fit_tables(S_P, exclude=lt.GATE_LED)
    shipped_tables = fit.fit_tables(S_P)

    per_light = {}
    for L in lt.gate_lights():
        probs = {L.klass: 1.0}
        M = fit.blend_map(gate_tables, probs, L.mired)
        row = {
            "class": L.klass,
            "cct_k": L.cct_k,
            "mired": L.mired,
            "gate": _stats(ev.render_error_with_matrix(S_P, M, L.spd, L.cct_k, ctx.r_ces, ctx)),
            "identity": _stats(ev.render_error_with_matrix(S_P, np.eye(3), L.spd, L.cct_k, ctx.r_ces, ctx)),
            "fit_at_light": _stats(ev.render_error_with_matrix(
                S_P, fit.fit_map(S_P, ctx.alexa, L.spd, ctx.r_train), L.spd, L.cct_k, ctx.r_ces, ctx)),
            "shipped_fit": _stats(ev.render_error_with_matrix(
                S_P, fit.blend_map(shipped_tables, probs, L.mired), L.spd, L.cct_k, ctx.r_ces, ctx)),
        }
        per_light[L.name] = row
    worst_median = max(r["gate"]["median"] for r in per_light.values())
    worst_p95 = max(r["gate"]["p95"] for r in per_light.values())
    return {
        "profile": profile_id,
        "lights": per_light,
        "worst_median": worst_median,
        "worst_p95": worst_p95,
        "pass": bool(worst_median <= MEDIAN_MAX and worst_p95 <= P95_MAX),
    }


def format_table(res: dict) -> str:
    lines = [f"{res['profile']}  (gate: median <= {MEDIAN_MAX}, p95 <= {P95_MAX})",
             f"  {'light':<18} {'class':<12} {'cct_k':>8} | {'gate med':>8} {'gate p95':>8} | "
             f"{'ident med':>9} {'ident p95':>9} | {'atlight med':>11} {'atlight p95':>11} | "
             f"{'shipped med':>11} {'shipped p95':>11}"]
    for name, r in res["lights"].items():
        flag = "" if (r["gate"]["median"] <= MEDIAN_MAX and r["gate"]["p95"] <= P95_MAX) else "  <-- over"
        lines.append(
            f"  {name:<18} {r['class']:<12} {r['cct_k']:8.1f} | {r['gate']['median']:8.3f} {r['gate']['p95']:8.3f} | "
            f"{r['identity']['median']:9.3f} {r['identity']['p95']:9.3f} | "
            f"{r['fit_at_light']['median']:11.3f} {r['fit_at_light']['p95']:11.3f} | "
            f"{r['shipped_fit']['median']:11.3f} {r['shipped_fit']['p95']:11.3f}{flag}")
    lines.append(f"  worst median {res['worst_median']:.3f}, worst p95 {res['worst_p95']:.3f}: "
                 f"{'PASS' if res['pass'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="only this profile id (default: all 11)")
    ap.add_argument("--json", type=Path, default=paths.OUT / "sensor_map" / "gate.json", help="write results here")
    args = ap.parse_args(argv)
    paths.require_sources()

    ctx = ev.context()
    ids = [args.profile] if args.profile else [p.id for p in PROFILES]
    results = {}
    for pid in ids:
        results[pid] = run_profile(pid, ctx)
        print(format_table(results[pid]), flush=True)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=1) + "\n")
    print("wrote", args.json)

    print("\nsummary (worst light per profile)")
    for pid, r in results.items():
        print(f"  {pid:<18} worst median {r['worst_median']:.3f}  worst p95 {r['worst_p95']:.3f}  "
              f"{'pass' if r['pass'] else 'over threshold'}")
    if GATE_PROFILE in results:
        ok = results[GATE_PROFILE]["pass"]
        print(f"\nGATE ({GATE_PROFILE}): {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
