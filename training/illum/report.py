"""The full report of one trained model: gates, baselines, per-class and per-light errors, calibration, cross-camera.

    python3 -m training.illum.report --profile sony-ilce-7rm4 [--model data/models/illum/sony-ilce-7rm4.json]

Sets (all synthetic, rendered through the profile's own curves; ``docs/13``):

* ``test``: held-out lights **and** held-out reflectances (the 38 test patches of the rawtoaces 190, smooth spectra from the
  test stream); the gate set (median <= 2 degrees);
* ``ces``: the 99 CIE CES (evaluation only) under the test lights;
* ``gate_lights``: scenes lit by each of the nine sensor-map gate lights;
* ``val``: the validation set the temperature and the confidence were calibrated on (held-out lights and patches too);
* for the A7R IV model only, ``cross_camera``: the same model on scenes rendered through the other ten bodies' curves (the
  stand-in situation of the A1 and every body without a profile).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from training.common import paths
from training.illum import baselines as bl
from training.illum import dataset as dsm
from training.illum import evaluate as ev
from training.illum import gate_estimated as ge
from training.illum import model as md
from training.illum import spectra as sp

#: Set YM_ILLUM_FAST=1 for a quick smoke run of the whole pipeline (small sets, not for numbers).
FAST = bool(os.environ.get("YM_ILLUM_FAST"))
N_TEST, N_CES, N_VAL, N_CROSS = (300, 150, 200, 40) if FAST else (3000, 1500, 2000, 400)
N_GATE = 30 if FAST else 200
STAND_IN = "sony-ilce-7rm4"


def baseline_errors(ds: dsm.Dataset) -> dict:
    w_true = bl.w_from_uv(ds.truth_uv)
    out = {}
    for name, f in bl.BASELINES.items():
        w = np.stack([f(ds.imgs[i], ds.masks[i]) for i in range(len(ds))])
        e = bl.angular_error_deg(w, w_true)
        out[name] = ev.summarize(e[np.isfinite(e)])
    ok = ~np.isnan(ds.as_shot_uv[:, 0])
    out["as_shot_only"] = ev.summarize(bl.angular_error_uv_deg(ds.as_shot_uv[ok], ds.truth_uv[ok]))
    return out


def grey_world_plus_prior(ds_val: dsm.Dataset, ds_test: dsm.Dataset) -> dict:
    """Grey world fused with the as-shot Gaussian (sigma 0.05) like the model, with the grey-world sigma picked on val."""
    def uv_gw(ds):
        w = np.stack([bl.grey_world(ds.imgs[i], ds.masks[i]) for i in range(len(ds))])
        return bl.uv_from_w(w)

    def fused(ds, uv, sg):
        a = ds.as_shot_uv
        ok = ~np.isnan(a[:, 0]) & np.isfinite(uv[:, 0])
        out = uv.copy()
        out[ok] = bl.fuse_gaussian(uv[ok], sg**2, a[ok], md.PRIOR_SIGMA**2)
        return out

    uv_v, uv_t = uv_gw(ds_val), uv_gw(ds_test)
    best = min((0.04, 0.06, 0.08, 0.12, 0.2, 0.3), key=lambda sg: np.nanmedian(bl.angular_error_uv_deg(fused(ds_val, uv_v, sg), ds_val.truth_uv)))
    e = bl.angular_error_uv_deg(fused(ds_test, uv_t, best), ds_test.truth_uv)
    s = ev.summarize(e[np.isfinite(e)])
    s["grey_world_sigma"] = best
    return s


def by_iso(model: md.IllumModel, ds: dsm.Dataset) -> dict:
    res = ev.run(model, ds, use_prior=False, use_bv=True)
    e = ev.angular(res, ds)
    out = {}
    for name, lo, hi in (("iso_100_800", 0, 800), ("iso_800_6400", 800, 6400), ("iso_6400_up", 6400, 1e9)):
        m = (ds.iso >= lo) & (ds.iso < hi) if lo else (ds.iso < hi)
        if m.any():
            out[name] = ev.summarize(e[m])
    return out


def build_report(profile_id: str, model: md.IllumModel, cross_camera: bool | None = None, log=print) -> dict:
    spec = model.spec
    ds_te = dsm.build(profile_id, "test", N_TEST, spec, keep_images=True, procs=2)
    ds_ces = dsm.build(profile_id, "test", N_CES, spec, refl="ces", keep_images=False, procs=2, stream=1)
    ds_va = dsm.build(profile_id, "val", N_VAL, spec, keep_images=True, procs=2)
    rep = {"profile_id": profile_id, "model_version": model.version, "model_sha256": model.payload_sha256(),
           "param_count": model.param_count(), "sets": {"test": N_TEST, "ces": N_CES, "val": N_VAL}}
    rep["test"] = ev.full_report(model, ds_te)
    rep["ces"] = ev.full_report(model, ds_ces)
    rep["val"] = ev.full_report(model, ds_va)
    rep["test"]["by_iso_histogram_only"] = by_iso(model, ds_te)
    rep["baselines_test"] = baseline_errors(ds_te)
    rep["baselines_test"]["grey_world_plus_as_shot_prior"] = grey_world_plus_prior(ds_va, ds_te)
    rep["sensor_map_gate_estimated"] = ge.run_modes(profile_id, model, N_GATE)
    rep["gate_median_deg"] = ev.GATE_MEDIAN_DEG
    rep["gates"] = {
        "test_median_with_prior": rep["test"]["with_prior"]["all"]["median"],
        "test_median_histogram_only": rep["test"]["histogram_only"]["all"]["median"],
        "pass_with_prior": bool(rep["test"]["with_prior"]["all"]["median"] <= ev.GATE_MEDIAN_DEG),
        "pass_histogram_only": bool(rep["test"]["histogram_only"]["all"]["median"] <= ev.GATE_MEDIAN_DEG),
    }
    if cross_camera is None:
        cross_camera = profile_id == STAND_IN
    if cross_camera:
        cc = {}
        for pid in sp.profile_ids():
            if pid == profile_id:
                continue
            # the other body's scenes, histogrammed in this model's window
            d = dsm.build(pid, "test", N_CROSS, spec, procs=2, stream=2, verbose=False)
            cc[pid] = {"with_prior": ev.summarize(ev.angular(ev.run(model, d, True, True), d)),
                       "histogram_only": ev.summarize(ev.angular(ev.run(model, d, False, True), d))}
        rep["cross_camera"] = cc
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=STAND_IN)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gate-n", type=int, default=N_GATE, help="scenes per gate light for --gate-only")
    ap.add_argument("--gate-only", action="store_true", help="recompute only the sensor-map gate with estimated lights in an existing report")
    args = ap.parse_args(argv)
    mp = args.model or paths.REPO_ROOT / "data" / "models" / "illum" / f"{args.profile}.json"
    model = md.IllumModel.from_json(json.loads(mp.read_text()))
    out = args.out or paths.REPO_ROOT / "data" / "models" / "illum" / "metrics" / f"{args.profile}.json"
    if args.gate_only:
        rep = json.loads(out.read_text())
        assert rep["model_sha256"] == model.payload_sha256(), "the report belongs to another model"
        rep["sensor_map_gate_estimated"] = ge.run_modes(args.profile, model, args.gate_n)
        out.write_text(json.dumps(rep, indent=1) + "\n")
        print(f"{args.profile}: gate with estimated lights updated in {out}")
        return 0
    rep = build_report(args.profile, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    g = rep["gates"]
    print(f"{args.profile}: test median {g['test_median_with_prior']:.2f} deg with the as-shot prior "
          f"({'pass' if g['pass_with_prior'] else 'FAIL'}), {g['test_median_histogram_only']:.2f} histogram only "
          f"({'pass' if g['pass_histogram_only'] else 'FAIL'}); wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
