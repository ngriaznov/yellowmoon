"""Fit the white scale of every registry body, write ``white_scale.json``, report both references.

    python3 -m training.white_scale.fit --write        # fit from the owner's photos, write white_scale.json
    python3 -m training.white_scale.fit --check        # recompute, compare with the committed white_scale.json
    python3 -m training.white_scale.fit --report [--local testdata/real/files]

``white_scale.json`` is what ``training.sensor_map.build_registry`` puts into ``data/cameras/registry.json``
(``white_scale`` and ``white_scale_source`` of each body). The rules (docs/03, "White scale of stand-in bodies"):

* a body with a profile of its own (``exact``) and a body that shares its profile's sensor (``sensor_family``)
  has the scale ``[1, 1, 1]``;
* a stand-in body (``default``) with files in ``--photos`` gets reference (a): the scale that puts the presets of
  its files on the locus of its profile (``reference.fit_presets``); all files of a body must carry the same
  presets (they are a property of the body and its firmware, not of the shot), and the test is exact;
* a stand-in body without files whose published colour matrix equals that of a fitted body (largest
  per-coefficient difference after row normalisation at most 0.02, the limit of the sibling check) takes that
  body's scale and says so;
* any other stand-in body has ``[1, 1, 1]`` and says that nothing was measured.

Reference (b), the published colour matrix at D65, is computed for every body with files and recorded next to the
presets' residuals, but does not set the scale (docs/03 says why). The presets are read from the files with our own
reader (``sr2.py``); the matrices are numbers in ``colour_matrices.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.common import jsonfmt, paths
from training.sensor_map import registry_data as rd
from training.white_scale import reference as ref
from training.white_scale import sr2

RESULT_JSON = Path(__file__).resolve().parent / "white_scale.json"
PHOTOS_DIR = paths.TESTDATA / "readme-photos" / "src"
LOCAL_DIR = paths.TESTDATA / "real" / "files"
#: Model ID -> the stand-in body whose scale it takes when its published matrix equals that body's.
SIBLING_OF = {400: 384}
#: The files of a fitted body that live in the owner's photo folder are found by their Model string.
DECIMALS = 4
#: Daylight presets are labelled by the maker as a typical temperature (docs/03); the check window of the task.
DAYLIGHT_WINDOW_K = (5000.0, 5600.0)
MEAN_ANGLE_LIMIT_DEG = 2.0


def _round3(s) -> list[float]:
    return [round(float(s[0]), DECIMALS), 1.0, round(float(s[2]), DECIMALS)]


OWNER_SHA256 = paths.REPO_ROOT / "tools" / "fetch" / "owner-photos.sha256"


def check_owner_files(folder: Path) -> None:
    """Refuses a file named in ``tools/fetch/owner-photos.sha256`` whose SHA-256 differs: the registry names the
    owner's photos as the source of the A1's scale, so the files read must be those."""
    import hashlib

    if not OWNER_SHA256.is_file():
        return
    pinned = {}
    for line in OWNER_SHA256.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name and "/" not in name:
            pinned[name] = digest
    for f in sorted(folder.glob("*.ARW")):
        want = pinned.get(f.name)
        if want is not None and hashlib.sha256(f.read_bytes()).hexdigest() != want:
            raise SystemExit(f"{f}: SHA-256 differs from tools/fetch/owner-photos.sha256 (run tools/fetch/owner-photos.sh)")


def photo_levels(folder: Path) -> dict[str, dict[str, dict[str, tuple[int, ...]]]]:
    """``{model string: {file name: levels}}`` for the ARW files of a folder."""
    out: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
    for f in sorted(folder.glob("*.ARW")) + sorted(folder.glob("*.arw")):
        try:
            levels = sr2.read_levels(f)
            model = sr2.read_model(f)
        except (sr2.Sr2Error, OSError, ValueError) as e:  # a file that is not a usable ARW is skipped, with a note
            print(f"skipping {f.name}: {e}", file=sys.stderr)
            continue
        out.setdefault(model, {})[f.name] = levels
    return out


def _same_presets(files: dict[str, dict[str, tuple[int, ...]]]) -> bool:
    keys = list(sr2.KELVIN_PRESETS) + list(sr2.NAMED_PRESETS)
    first = next(iter(files.values()))
    return all(all(levels.get(k) == first.get(k) for k in keys) for levels in files.values())


def _body_fit(body: rd.Body, files: dict[str, dict[str, tuple[int, ...]]], matrices) -> dict:
    """Reference (a) fitted on the files of a stand-in body, with reference (b) beside it."""
    if not _same_presets(files):
        raise SystemExit(f"{body.model}: the files carry different white-balance presets; a body has one set")
    profile = ref.load_profile(body.profile)
    levels = next(iter(files.values()))
    scale = ref.fit_presets(profile, levels)
    scale = tuple(_round3(scale))
    fit = ref.judge(profile, levels, scale)
    m_scale = ref.matrix_scale(profile, matrices[body.model_id])
    m_fit = ref.judge(profile, levels, m_scale)
    return {
        "white_scale": list(scale),
        "source": {
            "kind": "presets",
            "files": sorted(files),
            "residual": {
                "mean_preset_angle_before_deg": round(fit.mean_angle_before_deg, 2),
                "mean_preset_angle_after_deg": round(fit.mean_angle_after_deg, 2),
                "daylight_k_before": round(fit.daylight_k_before),
                "daylight_k_after": round(fit.daylight_k_after),
                "matrix_scale": _round3(m_scale),
                "matrix_mean_preset_angle_deg": round(m_fit.mean_angle_after_deg, 2),
                "matrix_daylight_k": round(m_fit.daylight_k_after),
            },
            "note": (
                "Fitted so the five Kelvin presets of the body's own files (2500, 3200, 4500, 6000, 8500 K) read at their "
                f"labels on the {rd.PROFILE_BY_ID[body.profile].display_name} locus (least squares in log chroma). "
                + _matrix_sentence(fit, m_fit)
            ),
        },
    }


def _matrix_sentence(fit: ref.Fit, m_fit: ref.Fit) -> str:
    """What the published colour matrix at D65 would have given on the same files, and whether it meets the limits."""
    meets = m_fit.mean_angle_after_deg <= MEAN_ANGLE_LIMIT_DEG and DAYLIGHT_WINDOW_K[0] <= m_fit.daylight_k_after <= DAYLIGHT_WINDOW_K[1]
    return (
        "The scale the body's published colour matrix at D65 gives (matrix_scale) leaves a mean preset angle of "
        f"{m_fit.mean_angle_after_deg:.2f} degrees and reads the Daylight preset at {m_fit.daylight_k_after:.0f} K, "
        + ("within" if meets else "outside")
        + f" the limits of docs/03 (at most {MEAN_ANGLE_LIMIT_DEG:g} degrees, Daylight {DAYLIGHT_WINDOW_K[0]:.0f} to "
        f"{DAYLIGHT_WINDOW_K[1]:.0f} K), so the presets set the scale."
    )


def fit_all(photos: Path) -> dict:
    """The contents of ``white_scale.json``."""
    matrices = ref.load_matrices()
    check_owner_files(photos)
    levels_by_model = photo_levels(photos)
    bodies: dict[str, dict] = {}
    fitted: dict[int, dict] = {}
    for b in rd.BODIES:
        if b.match == "default" and b.model in levels_by_model:
            fitted[b.model_id] = _body_fit(b, levels_by_model[b.model], matrices)
    for b in rd.BODIES:
        if b.match == "exact":
            entry = {
                "white_scale": [1.0, 1.0, 1.0],
                "source": {"kind": "own_curves", "files": [], "note": "The profile was measured on this body (curves of the registry's profile); nothing to correct."},
            }
        elif b.match == "sensor_family":
            entry = {
                "white_scale": [1.0, 1.0, 1.0],
                "source": {"kind": "same_sensor", "files": [], "note": "Same sensor as the measured body of its profile (the colour-matrix check of docs/03); no scale of its own."},
            }
        elif b.model_id in fitted:
            entry = fitted[b.model_id]
        elif b.model_id in SIBLING_OF and SIBLING_OF[b.model_id] in fitted:
            base = rd.BODY_BY_ID[SIBLING_OF[b.model_id]]
            diff = ref.matrix_difference(matrices[b.model_id], matrices[base.model_id])
            if diff <= rd.SIBLING_LIMIT:
                entry = {
                    "white_scale": list(fitted[base.model_id]["white_scale"]),
                    "source": {
                        "kind": "sibling",
                        "files": [],
                        "note": (
                            f"Takes the scale of the {base.body}: the published colour matrices of the two bodies agree "
                            f"(largest per-coefficient difference {diff:.4f}, limit {rd.SIBLING_LIMIT}). No file of this body was "
                            "available, so the presets were not read."
                        ),
                    },
                }
            else:
                entry = _none(b)
        else:
            entry = _none(b)
        entry["model"] = b.model
        bodies[str(b.model_id)] = entry
    return {
        "schema_version": 1,
        "description": (
            "White scale per registry body, generated by python3 -m training.white_scale.fit --write from the white-balance "
            "presets stored in the owner's photos (tools/fetch/owner-photos.sh) and the published colour matrices in "
            "colour_matrices.json. build_registry copies it into data/cameras/registry.json."
        ),
        "bodies": bodies,
    }


def _none(b: rd.Body) -> dict:
    return {
        "white_scale": [1.0, 1.0, 1.0],
        "source": {
            "kind": "none",
            "files": [],
            "note": "No file with white-balance presets of this body or of a body with the same colour matrix was available; its whites are not corrected.",
        },
    }


# ---------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------
def report(photos: Path, local: Path | None) -> None:
    matrices = ref.load_matrices()
    reg = json.loads(paths.REGISTRY_JSON.read_text())
    by_model = {b["model"]: b for b in reg["bodies"]}
    folders = [("owner", photos)] + ([("local", local)] if local and local.is_dir() else [])
    rows = []
    for tag, folder in folders:
        for model, files in photo_levels(folder).items():
            reg_body = by_model.get(model)
            profile_id = reg_body["profile"] if reg_body else rd.DEFAULT_PROFILE
            levels = next(iter(files.values()))
            profile = ref.load_profile(profile_id)
            standins = [profile_id] if profile_id == rd.DEFAULT_PROFILE else [profile_id, rd.DEFAULT_PROFILE]
            for pid in standins:
                p = ref.load_profile(pid)
                s_a = ref.fit_presets(p, levels)
                fa = ref.judge(p, levels, s_a)
                row = {"tag": tag, "model": model, "files": sorted(files), "profile": pid, "listed": reg_body is not None, "a": fa}
                if reg_body is not None:
                    s_b = ref.matrix_scale(p, matrices[reg_body["model_id"]])
                    row["b"] = ref.judge(p, levels, s_b)
                rows.append(row)
    print(f"{'body':11s} {'src':5s} {'profile':10s} {'files':>5s} | {'presets scale':>15s} {'angle':>11s} {'Day K':>11s} | {'matrix scale':>15s} {'angle':>5s} {'Day K':>5s}")
    for r in rows:
        a, b = r["a"], r.get("b")
        line = (
            f"{r['model']:11s} {r['tag']:5s} {r['profile'][5:]:10s} {len(r['files']):5d} | "
            f"[{a.scale[0]:.3f}, {a.scale[2]:.3f}] {a.mean_angle_before_deg:5.2f}->{a.mean_angle_after_deg:4.2f} "
            f"{a.daylight_k_before:5.0f}->{a.daylight_k_after:5.0f} | "
        )
        if b is not None:
            line += f"[{b.scale[0]:.3f}, {b.scale[2]:.3f}] {b.mean_angle_after_deg:5.2f} {b.daylight_k_after:5.0f}"
        else:
            line += "(body not in the registry)"
        print(line + ("" if r["listed"] else ""))
    print("\nthe published matrix against the curves of the measured bodies (as if each were a stand-in on the A7R IV profile):")
    s_prof = ref.load_profile(rd.DEFAULT_PROFILE)
    print(f"{'body':12s} {'curves s':>16s} {'matrix s':>16s} {'angle':>6s}")
    for b in rd.BODIES:
        if b.match != "exact" or b.profile == rd.DEFAULT_PROFILE:
            continue
        p = ref.load_profile(b.profile)
        truth = ref.locus_white(s_prof, ref.D65_K) / ref.locus_white(p, ref.D65_K)
        m = ref.matrix_scale(s_prof, matrices[b.model_id])
        w = ref.locus_white(p, ref.D65_K)
        print(f"{b.model:12s} [{truth[0]:.3f}, {truth[2]:.3f}] [{m[0]:.3f}, {m[2]:.3f}] {ref.angle_deg(ref.scaled(w, truth), ref.scaled(w, m)):6.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", type=Path, default=PHOTOS_DIR, help="folder with the owner's ARW files")
    ap.add_argument("--local", type=Path, default=LOCAL_DIR, help="folder with local evaluation files (report only)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if not any(args.photos.glob("*.ARW")):
        print(f"cannot run: no ARW files in {args.photos} (tools/fetch/owner-photos.sh)", file=sys.stderr)
        return 2
    if args.report:
        report(args.photos, args.local)
        return 0
    result = fit_all(args.photos)
    text = jsonfmt.dumps(result)
    if args.write:
        RESULT_JSON.write_text(text)
        print("wrote", RESULT_JSON.relative_to(paths.REPO_ROOT))
        return 0
    committed = json.loads(RESULT_JSON.read_text())
    if committed != json.loads(text):
        print("white_scale.json differs from a fresh fit (run --write and review)", file=sys.stderr)
        return 1
    print("white_scale.json matches a fresh fit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
