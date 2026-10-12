"""The sensor-map gate with estimated lights (M1-06 "the sensor-map gate still passes with estimated lights").

``training/sensor_map/gate.py`` renders the 99 CIE CES under the nine held-out gate lights with the light **known**: the
true class (probability 1) and the true mired pick the map, the true white balances the camera, ARRI's matrix is taken
at the true CCT. Here the light comes from the illuminant estimator instead. For each gate light we render synthetic
scenes lit by that light (held-out reflectances, :mod:`training.illum.scenes`), run the model, and take from its output what the
chain would use (``docs/02`` C1-C6, Auto white balance):

* the camera RGB is balanced with the **estimated** white ``w_est`` (a white-balance error shows as a cast);
* the map is ``M(probs, mired)`` with the six estimator classes folded into the profile's four
  (``ym_color::wb::EstimatorClasses::to_profile_classes``: shade is daylight, ``mixed`` goes to tungsten below 4000 K and to
  daylight from 4000 K) and the mired of the CCT that the profile's locus reads from ``w_est``;
* ARRI's matrix is taken at that estimated CCT;
* the reference is the ALEXA rendering under the true light, balanced and mapped by the true CCT, as in the gate.

ΔE00 is pooled over all scenes and the 99 reflectances of a light; median and p95 are compared with the known-light gate
(same tables, the gate refit without LED-B3, LED-RGB1 and LED-V2).

    python3 -m training.illum.gate_estimated --profile sony-ilce-7rm4 --model data/models/illum/sony-ilce-7rm4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.alexa import ssf_core as s
from training.common import paths
from training.illum import dataset as dsm
from training.illum import evaluate as ev
from training.illum import model as md
from training.illum import spectra as sp
from training.sensor_map import evaluate as sev
from training.sensor_map import fit
from training.sensor_map import gate as sgate
from training.sensor_map import lights as lt
from training.sensor_map import runtime_ref as rr
from training.sensor_map.registry_data import PROFILES

SPLIT_CCT_K = 4000.0
CLASS_OF_PROFILE = ("daylight", "tungsten", "fluorescent", "led")


def fold_classes(probs6: np.ndarray, cct_k: np.ndarray) -> np.ndarray:
    """The six estimator classes as the profile's four (daylight, tungsten, fluorescent, led), ``(..., 4)``."""
    p = np.asarray(probs6, dtype=np.float64)
    mixed_to_tungsten = np.asarray(cct_k) < SPLIT_CCT_K
    out = np.stack([p[..., 0] + p[..., 1], p[..., 2], p[..., 3], p[..., 4]], axis=-1)
    out[..., 1] += np.where(mixed_to_tungsten, p[..., 5], 0.0)
    out[..., 0] += np.where(mixed_to_tungsten, 0.0, p[..., 5])
    return out / out.sum(axis=-1, keepdims=True)


def estimated_light(res: dict, prof: rr.Profile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(w_est (n, 3), cct_k (n,), probs4 (n, 4))`` from model outputs."""
    uv = res["uv"]
    w = np.stack([np.exp(-uv[:, 0]), np.ones(len(uv)), np.exp(-uv[:, 1])], axis=-1)
    cct = np.array([rr.rgb_to_cct_duv(prof, x)[0] for x in w])
    return w, cct, fold_classes(res["probs"], cct)


MODES = ("full", "map_only", "white_only")


class Prepared:
    """Scenes and renderings of one profile that do not depend on the model's output (built once per profile)."""

    def __init__(self, profile_id: str, model: md.IllumModel, n_scenes: int, refl: str, tables):
        self.profile_id, self.n_scenes, self.refl = profile_id, n_scenes, refl
        self.ctx = sev.context()
        prof_spec = next(p for p in PROFILES if p.id == profile_id)
        self.S_fit = fit.sony_curves()[prof_spec.curves_name]   # peak-normalised: scale-free, as the maps were fitted
        self.cam = sp.camera(profile_id)                         # raw scale: what the estimator sees
        self.prof = rr.Profile.load(paths.PROFILES_DIR / profile_id / "profile.json")
        self.tables = fit.fit_tables(self.S_fit, exclude=lt.GATE_LED) if tables is None else tables
        self.lights = []
        for gl in sp.gate_lights():
            ds = dsm.build(profile_id, "test", n_scenes, model.spec, refl=refl, light_name=gl.name, procs=1, cache=False, verbose=False)
            true_cct = gl.cct_k if gl.cct_k is not None else s.illuminant_cct(gl.spd)
            m_true = s.interpolate_matrix(self.ctx.arri_ccts, self.ctx.arri_mats, true_cct)
            self.lights.append({
                "gl": gl, "ds": ds, "true_cct": float(true_cct),
                "raw": (gl.spd[:, None] * self.ctx.r_ces).T @ self.cam.curves,   # (99, 3) raw CES responses
                "w_true_raw": gl.spd @ self.cam.curves,
                "target": s.ap0_to_Lab(s.camera_rgb_wb(self.ctx.alexa, gl.spd, self.ctx.r_ces) @ m_true.T),
                "known": sgate_known(gl, self.S_fit, self.tables, self.ctx),
                "true_class": {"tungsten_halogen": "tungsten", "daylight": "daylight", "fluorescent": "fluorescent", "led": "led"}[gl.klass],
            })


def evaluate(prep: Prepared, model: md.IllumModel, use_prior: bool = True, mode: str = "full", results=None) -> dict:
    """Gate numbers with estimated lights for one profile, mode and prior setting."""
    assert mode in MODES
    ctx, tables = prep.ctx, prep.tables
    out = {}
    for L in prep.lights:
        gl, ds = L["gl"], L["ds"]
        res = ev.run(model, ds, use_prior=use_prior, use_bv=True) if results is None else results[(gl.name, use_prior)]
        w_est, cct, probs4 = estimated_light(res, prep.prof)
        err = ev.angular(res, ds)
        de = []
        for i in range(prep.n_scenes):
            w_bal = L["w_true_raw"] / L["w_true_raw"][1] if mode == "map_only" else w_est[i]
            balanced = L["raw"] / (w_bal * L["w_true_raw"][1])  # a white with green 1; the true white has green w_true_raw[1]
            if mode == "white_only":
                probs, mired_i, cct_i = {L["true_class"]: 1.0}, 1e6 / L["true_cct"], L["true_cct"]
            else:
                probs = {k: float(p) for k, p in zip(CLASS_OF_PROFILE, probs4[i])}
                mired_i, cct_i = 1e6 / cct[i], cct[i]
            M = fit.blend_map(tables, probs, mired_i)
            M_arri = s.interpolate_matrix(ctx.arri_ccts, ctx.arri_mats, cct_i)
            mapped = (balanced @ M.T) @ M_arri.T
            de.append(s.delta_E00(s.ap0_to_Lab(mapped), L["target"]))
        de = np.concatenate(de)
        out[gl.name] = {
            "class": gl.klass, "cct_k": L["true_cct"],
            "angular_error_deg": ev.summarize(err),
            "estimated": {"median": float(np.median(de)), "p95": float(np.percentile(de, 95)), "max": float(de.max())},
            "known_light": L["known"],
            "mean_class_probability_true_class": float(np.mean(res["probs"][np.arange(prep.n_scenes), sp.CLASS_INDEX[gl.klass]])),
        }
    worst_med = max(v["estimated"]["median"] for v in out.values())
    worst_p95 = max(v["estimated"]["p95"] for v in out.values())
    over = [k for k, v in out.items() if v["estimated"]["median"] > sgate.MEDIAN_MAX or v["estimated"]["p95"] > sgate.P95_MAX]
    return {"profile": prep.profile_id, "mode": mode, "use_prior": use_prior, "refl": prep.refl, "n_scenes_per_light": prep.n_scenes,
            "lights": out, "worst_median": worst_med, "worst_p95": worst_p95, "lights_over": over}


def run(profile_id: str, model: md.IllumModel, n_scenes: int = 200, use_prior: bool = True, refl: str = "default",
        tables=None, mode: str = "full") -> dict:
    """:func:`evaluate` for one setting (see :func:`run_modes` for all).

    ``mode`` splits the error by what the estimate is allowed to change:

    * ``"full"``: everything comes from the estimate (the gate as worded: white balance, classes and mired, ARRI CCT);
    * ``"map_only"``: the camera is balanced with the **true** white; the estimated classes, mired and CCT choose the map and
      ARRI's matrix (what the estimator costs the sensor map);
    * ``"white_only"``: the estimated white balances the camera; the true class, mired and CCT choose the map and the matrix
      (what the white-balance error costs).
    """
    return evaluate(Prepared(profile_id, model, n_scenes, refl, tables), model, use_prior, mode)


def run_modes(profile_id: str, model: md.IllumModel, n_scenes: int = 200, refl: str = "default", tables=None) -> dict:
    """All three modes with and without the as-shot prior: ``{"full/with_prior": ..., "map_only/histogram_only": ...}``."""
    prep = Prepared(profile_id, model, n_scenes, refl, tables)
    cache = {(L["gl"].name, up): ev.run(model, L["ds"], use_prior=up, use_bv=True) for L in prep.lights for up in (True, False)}
    return {f"{mode}/{'with_prior' if up else 'histogram_only'}": evaluate(prep, model, up, mode, cache)
            for mode in MODES for up in (True, False)}


def sgate_known(gl: sp.Light, S_fit: np.ndarray, tables, ctx) -> dict:
    """The known-light gate numbers of ``training/sensor_map/gate.py`` for the same light and tables."""
    klass = {"tungsten_halogen": "tungsten", "daylight": "daylight", "fluorescent": "fluorescent", "led": "led"}[gl.klass]
    cct = gl.cct_k if gl.cct_k is not None else s.illuminant_cct(gl.spd)
    M = fit.blend_map(tables, {klass: 1.0}, 1e6 / cct)
    d = sev.render_error_with_matrix(S_fit, M, gl.spd, cct, ctx.r_ces, ctx)
    return {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def format_table(res: dict) -> str:
    lines = [f"{res['profile']}  sensor-map gate with estimated lights, mode {res.get('mode', 'full')} ({'with' if res['use_prior'] else 'without'} as-shot prior, "
             f"{res['n_scenes_per_light']} scenes per light; DE00 median / p95 over scenes x 99 CES)",
             f"  {'light':<18} {'angular med/p90':>16} | {'estimated':>14} | {'known light':>14} | {'delta median':>12}"]
    for name, r in res["lights"].items():
        a, e, k = r["angular_error_deg"], r["estimated"], r["known_light"]
        lines.append(f"  {name:<18} {a['median']:6.2f}/{a['p90']:<8.2f} | {e['median']:6.3f}/{e['p95']:<7.3f} | "
                     f"{k['median']:6.3f}/{k['p95']:<7.3f} | {e['median'] - k['median']:+12.3f}")
    over = f"; over the gate: {', '.join(res['lights_over'])}" if res["lights_over"] else ""
    lines.append(f"  worst median {res['worst_median']:.3f}, worst p95 {res['worst_p95']:.3f}{over}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="sony-ilce-7rm4")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    model = md.IllumModel.from_json(json.loads((args.model or paths.REPO_ROOT / "data" / "models" / "illum" / f"{args.profile}.json").read_text()))
    results = run_modes(args.profile, model, args.n)
    for r in results.values():
        print(format_table(r), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
