"""Write ``data/cameras/profiles/<profile-id>/profile.json`` for the 11 measured Sony bodies.

    python3 -m training.sensor_map.build_profiles               # write all 11
    python3 -m training.sensor_map.build_profiles --check       # regenerate in memory, compare
    python3 -m training.sensor_map.build_profiles --git-sha SHA # stamp this commit in provenance

The output is a deterministic function of the fetched inputs, the scripts and colour-science
0.4.7: no timestamps, tables rounded to 1e-9, matrices to 1e-12 (rows sum to 1 exactly up to
float rounding). ``provenance.fit_script`` names the commit the working tree is based on;
``provenance.fit_sources_sha256`` is a hash of the script sources themselves, which is what
identifies the code that produced the numbers when the tree is dirty. ``provenance.sensor_map_objective``
and ``provenance.fit_solver`` record the fit objective (Lab-optimal white-preserving 3x3, decision
2026-10-12) and the scipy/numpy versions that solved it. ``--check`` ignores the commit in
``fit_script`` and the versions in ``fit_solver``.

Curve scales (docs/02 C1; ``CURVE_SCALES``, recorded in ``provenance.light_conventions``): the camera ``locus``
and ``duv_scale`` use the curves in the **published raw scale** (``fit.sony_curves_raw``: clipped at 0, not
normalised per channel), because the scene white at runtime comes from raw data; the ``sensor_map`` and the
``alexa_white`` tables are scale-free and keep the peak-normalised path they were fitted with, bit for bit.

Format: ``crates/ym-profiles/schema/profile.schema.json``; rules: ``training/sensor_map/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import colour
import numpy as np

from training.common import jsonfmt, paths
from training.sensor_map import fit
from training.sensor_map import lights as lt
from training.sensor_map import tables as tb
from training.sensor_map.registry_data import PROFILES, SIBLING_CHECK, Profile, siblings_of

PROFILE_VERSION = "1.0.0"
M_DECIMALS = 12
FIT_SOURCES = (
    "training/alexa/ssf_core.py",
    "training/sensor_map/fit.py",
    "training/sensor_map/lights.py",
    "training/sensor_map/tables.py",
    "training/sensor_map/registry_data.py",
    "training/sensor_map/build_profiles.py",
)
#: Which curve scale each table of the profile was computed in (recorded in ``provenance.light_conventions``;
#: docs/02 C1). The locus and Duv scale are compared with RAW data, so they use the published scale; the
#: maps and ``w_A`` divide every response by the response to the white, so the scale cancels.
CURVE_SCALES = (
    "camera RGB scale: locus (log-chroma) and duv_scale tables are computed from the camera's spectral curves in the "
    "published raw scale (clipped at 0, NOT normalised per channel; one common factor per camera, green peak 1.0, "
    "red and blue peaks as published) so a locus point is the raw white of the light and compares directly with "
    "as-shot gains, Pick white and the illuminant estimator; sensor_map matrices are computed from the same curves "
    "peak-normalised per channel (scale-free: every response is divided by the response to the white of the same curves); "
    "alexa_white tables are from the ALEXA estimate, peak-normalised by construction (only ratios enter the chain)"
)
CIE_NOTE = (
    "CIE SPDs (D series, F1-F12, LED-B1..B5, LED-BH1, LED-RGB1, LED-V1, LED-V2; CC BY-SA 4.0, "
    "via colour-science) and the CIE 1931 observer were used at fit time only; this file holds "
    "fitted coefficients and tables derived from them, not the SPDs. Pending HUMAN M1-H4 "
    "(docs/14): confirm that CIE-derived coefficient tables may ship."
)


# ----------------------------------------------------------------------------
# Provenance helpers
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_sources_sha256() -> str:
    h = hashlib.sha256()
    for rel in FIT_SOURCES:
        h.update(rel.encode() + b"\0")
        h.update((paths.REPO_ROOT / rel).read_bytes() + b"\0")
    return h.hexdigest()


def pinned_commit(var: str) -> str:
    """A pin from ``tools/fetch/spectral.sh`` (``RTA_SHA`` or ``ACES_SHA``)."""
    text = (paths.REPO_ROOT / "tools" / "fetch" / "spectral.sh").read_text()
    m = re.search(rf'^{var}="([0-9a-f]{{40}})"', text, re.M)
    if not m:
        raise SystemExit(f"cannot find {var} in tools/fetch/spectral.sh")
    return m.group(1)


# ----------------------------------------------------------------------------
# Profile assembly
# ----------------------------------------------------------------------------
def _grid_json(grid: dict) -> dict:
    return {
        "start_k": grid["start_k"],
        "step_k": grid["step_k"],
        "values": tb.rounded(grid["values"]),
    }


def _matrix_json(m: np.ndarray) -> list[list[float]]:
    """Rows sum to 1: first two entries rounded to 12 decimals, the third is the remainder."""
    rows = []
    for r in range(3):
        a, b = round(float(m[r, 0]), M_DECIMALS), round(float(m[r, 1]), M_DECIMALS)
        c = round(1.0 - a - b, M_DECIMALS)
        rows.append([a, b, c])
    return rows


def _sensor_map_json(tables: dict[str, list[fit.MapEntry]]) -> dict:
    return {
        "method": "per-light-3x3",
        "target": "alexa/target.json",
        "fit": {
            "white_preserving": True,
            "objective": "cie-lab-after-arri-matrix",
            "training_reflectances": 190,
        },
        "classes": {
            klass: [
                {"light": e.light, "mired": round(e.mired, tb.DECIMALS), "m": _matrix_json(e.m)}
                for e in entries
            ]
            for klass, entries in tables.items()
        },
    }


def _alexa_white_json(S_A: np.ndarray) -> dict:
    w = tb.w_a_tables(S_A)
    out: dict = {
        "interpolation": "bb/daylight: geometric in mired; fluorescent/led: linear in mired",
        "blackbody": _grid_json(w["blackbody"]),
        "daylight": _grid_json(w["daylight"]),
    }
    for klass in ("fluorescent", "led"):
        out[klass] = [
            {"light": e["light"], "mired": round(e["mired"], tb.DECIMALS), "w": tb.rounded(e["w"])}
            for e in tb.class_w_a(S_A, klass)
        ]
    return out


def _locus_json(S_P: np.ndarray) -> dict:
    lc = tb.locus_tables(S_P)
    ds = tb.duv_scale_tables(S_P)
    return {
        "axes": ["ln(R/G)", "ln(B/G)"],
        "interpolation": "linear in mired",
        "blackbody": _grid_json(lc["chroma"]["blackbody"]),
        "daylight": _grid_json(lc["chroma"]["daylight"]),
        "xy": {
            "blackbody": _grid_json(lc["xy"]["blackbody"]),
            "daylight": _grid_json(lc["xy"]["daylight"]),
        },
        "duv_scale": {"blackbody": _grid_json(ds["blackbody"]), "daylight": _grid_json(ds["daylight"])},
        "duv_sign": "positive toward magenta",
    }


def _checks_json(prof: Profile) -> list[dict]:
    return [
        {
            "pairing": f"{b.model} ({b.model_id}) uses {prof.id}",
            "check": "Adobe DNG Converter ColorMatrix of the sibling within 0.02 per coefficient "
                     f"of {prof.model} after normalisation (docs/03)",
            "result": SIBLING_CHECK,
        }
        for b in siblings_of(prof.id)
    ]


def build_profile(prof: Profile, git_sha: str) -> dict:
    S_P = fit.sony_curves()[prof.curves_name]  # peak-normalised: scale-free, for the maps
    S_raw = fit.sony_curves_raw()[prof.curves_name]  # published raw scale: for the locus and the Duv scale
    S_A = fit.alexa_curves()
    tables = fit.fit_tables(S_P, S_A)
    curves_path = paths.SONY_CURVES_DIR / prof.curves_file
    refs = lt.reference_lights()
    return {
        "id": prof.id,
        "version": PROFILE_VERSION,
        "display_name": prof.display_name,
        "spectral_curves": prof.spectral_curves,
        "sensor_map": _sensor_map_json(tables),
        "locus": _locus_json(S_raw),
        "alexa_white": _alexa_white_json(S_A),
        "decode": {"quirks": []},
        "provenance": {
            "curves": "Weta Digital via rawtoaces-data, Apache-2.0",
            "curves_source": {
                "repository": "https://github.com/AcademySoftwareFoundation/rawtoaces-data",
                "commit": pinned_commit("RTA_SHA"),
                "file": f"data/camera/{prof.curves_file}",
                "sha256": sha256_file(curves_path),
            },
            "target": {
                "file": "alexa/alexa_alev3_estimate_380_780_5.json",
                "sha256": sha256_file(paths.ALEXA_CURVES_JSON),
                "note": "ALEV III spectral sensitivity ESTIMATE (not a measurement), data/alexa/target.json",
            },
            "fit_script": f"training/sensor_map/fit.py@{git_sha}",
            "fit_sources_sha256": fit_sources_sha256(),
            "sensor_map_objective": fit.SHIPPED_OBJECTIVE_TEXT,
            "fit_solver": fit.solver_description(),
            "colour_science": colour.__version__,
            "training_reflectances": "rawtoaces 190-patch set (Apache-2.0), colour.characterisation.aces_it."
                                     "read_training_data_rawtoaces_v1, 380-780 nm at 5 nm",
            "reference_lights": {k: [L.name for L in v] for k, v in refs.items()},
            "duv_scale_method": "at every 100 K: median over green excess/deficit tilts of +-5 % and +-10 % "
                                "(training/sensor_map/lights.py tint_spd) of the locus light, of |CIE 1960 uv offset| / "
                                "|log-chroma offset|, both perpendicular to the locus (tangent from lights 3 % either side); "
                                "log-chroma of the camera curves in the published raw scale (see light_conventions)",
            "light_conventions": "blackbody below 4000 K and CIE D series from 4000 K (rawtoaces); mired = 1e6/CCT "
                                 "with CCT the generating temperature for blackbody and D series and the Ohno (2013) "
                                 "CCT of the SPD for F and LED lights; SPDs 380-780 nm at 5 nm, peak-normalised; " + CURVE_SCALES,
            "cie_data": CIE_NOTE,
            "checks": _checks_json(prof),
        },
    }


def render(profile: dict) -> str:
    return jsonfmt.dumps(profile)


def profile_path(prof: Profile) -> Path:
    return paths.PROFILES_DIR / prof.id / "profile.json"


def _close(a, b, tol: float = 5e-9) -> bool:
    """Structural equality with a float tolerance (BLAS differences between machines)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol
    return a == b


def _strip_sha(profile: dict) -> dict:
    """Drop what ``--check`` ignores: the stamped commit and the numpy/scipy versions in ``fit_solver``.

    The library versions are recorded for provenance only; a different (compatible) numpy or scipy
    is covered by the numeric tolerance of the comparison, not by a string match.
    """
    p = json.loads(json.dumps(profile))
    p["provenance"]["fit_script"] = "training/sensor_map/fit.py"
    p["provenance"]["fit_solver"] = ""
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--git-sha", default=None, help="commit to stamp (default: git rev-parse HEAD)")
    ap.add_argument("--check", action="store_true", help="regenerate in memory and compare with the files")
    ap.add_argument("--profile", help="only this profile id")
    args = ap.parse_args(argv)
    paths.require_sources()
    sha = args.git_sha or paths.git_head_sha()

    todo = [p for p in PROFILES if not args.profile or p.id == args.profile]
    bad = 0
    for prof in todo:
        built = build_profile(prof, sha)
        path = profile_path(prof)
        if args.check:
            if not path.exists():
                print(f"MISSING {path}")
                bad += 1
                continue
            on_disk = json.loads(path.read_text())
            same = _close(_strip_sha(on_disk), _strip_sha(json.loads(render(built))))
            print(f"{prof.id}: {'up to date' if same else 'DIFFERS'}")
            bad += 0 if same else 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            text = render(built)
            path.write_text(text)
            print(f"wrote {path.relative_to(paths.REPO_ROOT)} ({len(text) / 1024:.0f} kB)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
