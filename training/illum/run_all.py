"""Train, ship and report the illuminant models of all profiles.

    python3 -m training.illum.run_all [--profiles sony-ilce-7rm4 ...] [--steps 1500] [--n-train 20000]

For each profile: scenes (cached under ``testdata/out/illum/cache``, deleted after the profile unless ``--keep-cache``),
filter and head training (:mod:`training.illum.train`), the model file ``data/models/illum/<profile-id>.json`` and the report
``data/models/illum/metrics/<profile-id>.json`` (:mod:`training.illum.report`). Then ``python3 -m training.illum.model_card``
writes ``MODEL_CARD.md`` from the reports.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from training.common import jsonfmt, paths
from training.illum import dataset as dsm
from training.illum import model as md
from training.illum import report as rp
from training.illum import scenes as sc
from training.illum import spectra as sp
from training.illum import train as tr

MODELS = paths.REPO_ROOT / "data" / "models" / "illum"
SOURCES = ("spectra.py", "scenes.py", "features.py", "model.py", "dataset.py", "train.py", "evaluate.py", "baselines.py")

#: The training recipe of the shipped models (see MODEL_CARD.md for what was tried).
RECIPE = dict(steps=1500, lam_tv_f=2e-2, lam_tv_b=2e-2, lam_l2_f=1e-3, lam_l2_b=1e-3, eval_every=500)


def code_sha256() -> str:
    """Hash of the training sources that define the data and the model (stamped into every model file)."""
    h = hashlib.sha256()
    for name in SOURCES:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def run_profile(profile_id: str, cfg: tr.Config, n_train: int, n_val: int, keep_cache: bool, log=print,
                models_dir: Path = MODELS) -> None:
    t0 = time.time()
    model, info = tr.train_profile(profile_id, cfg, n_train, n_val, procs=2, log=log)
    model.meta.update({"training_code_sha256": code_sha256(), "git_base": paths.git_head_sha(),
                       "generator_version": sc.GENERATOR_VERSION, "recipe": cfg.__dict__ | {"n_train": n_train, "n_val": n_val},
                       "history": info["history"][-1:]})
    model = model.rounded()
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{profile_id}.json").write_text(jsonfmt.dumps(model.to_json()))
    log(f"  wrote {models_dir / (profile_id + '.json')} ({model.param_count()} parameters) after {time.time() - t0:.0f} s")
    rep = rp.build_report(profile_id, model, log=log)
    out = models_dir / "metrics" / f"{profile_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(rep, indent=1) + "\n")
    g = rep["gates"]
    log(f"  {profile_id}: test median {g['test_median_with_prior']:.2f} deg with the as-shot prior, "
        f"{g['test_median_histogram_only']:.2f} histogram only; {time.time() - t0:.0f} s in all")
    if not keep_cache:
        shutil.rmtree(dsm.CACHE, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=RECIPE["steps"])
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=MODELS, help="where to write the models and reports (default data/models/illum)")
    args = ap.parse_args(argv)
    cfg = replace(tr.Config(), **{**RECIPE, "steps": args.steps})
    for pid in args.profiles or sp.profile_ids():
        print(f"== {pid}", flush=True)
        run_profile(pid, cfg, args.n_train, args.n_val, args.keep_cache, log=lambda *a: print(*a, flush=True),
                    models_dir=args.models_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
