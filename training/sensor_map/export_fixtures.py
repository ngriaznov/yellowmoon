"""Export what a Rust port of ``evaluate.py`` / ``gate.py`` needs, to ``testdata/spectral/`` (git-ignored).

    python3 -m training.sensor_map.export_fixtures

CIE data (the observer, the illuminant and LED SPDs, the 99 CES reflectances) is CC BY-SA 4.0 and
must not be committed (``docs/14``). This script generates it from colour-science at test time into
``testdata/spectral/`` (git-ignored); a Rust test that needs it should skip with a clear message
when the folder is missing, and CI runs this script first. Nothing here is shipped.

Files (all text; wavelengths 380-780 nm at 5 nm, 81 samples; one row per spectrum):

``cmf_1931_2deg.csv``              wavelength_nm,x,y,z
``lights.csv``                     name,class,cct_k,<81 values>: the 9 research test lights
                                    ("research:*"), the 9 gate lights ("gate:*") and the 45
                                    reference lights ("ref:<class>:*"); peak-normalised, CCT as used
                                    for the mired axis and the ARRI matrix lookup
``reflectances_training_190.csv``   name,<81>: rawtoaces 190 patches (Apache-2.0)
``reflectances_ces_99.csv``         CIE 2017 CES (CC BY-SA 4.0)
``reflectances_colorchecker_24.csv``  ColorChecker (Ohta)
``reflectances_nist_cqs_vs_15.csv`` NIST CQS 9.0 VS
``constants.json``                  AP0 -> XYZ matrix, ACES white xy, CAT02 matrix, test-light order, the
                                    research evaluation set composition (CES 99 + Ohta 24 + VS 15 = 138)
``delta_e2000_vectors.json``        Lab pairs and colour-science CIEDE2000 values (to test a Rust port)
``expected.json``                   numbers from the Python reference: sony_profiles (own / A7R IV, per
                                    light) for the 11 bodies, and the sensor-map gate for the 11 profiles
``MANIFEST.json``                   SHA-256 of every file

The Camera curves, the ALEXA curves and ARRI's matrices are already in ``data/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import colour
import numpy as np

from training.alexa import ssf_core as s
from training.common import paths
from training.sensor_map import evaluate as ev
from training.sensor_map import fit, gate
from training.sensor_map import lights as lt
from training.sensor_map.registry_data import PROFILES

SEED = 20261012


def _write_rows(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([repr(float(x)) if isinstance(x, (float, np.floating)) else x for x in r])


def _spectra_header() -> list[str]:
    return [str(int(w)) for w in s.WL]


def export_cmf(out: Path) -> None:
    _write_rows(out / "cmf_1931_2deg.csv", ["wavelength_nm", "x", "y", "z"],
                [[int(w), *s.CMF[i]] for i, w in enumerate(s.WL)])


def export_lights(out: Path) -> None:
    rows = []
    for name, E in ev.context().illums.items():
        rows.append([f"research:{name}", "test", s.illuminant_cct(E), *E])
    for L in lt.gate_lights():
        rows.append([f"gate:{L.name}", L.klass, L.cct_k, *L.spd])
    for klass, lights in lt.reference_lights().items():
        for L in lights:
            rows.append([f"ref:{klass}:{L.name}", klass, L.cct_k, *L.spd])
    _write_rows(out / "lights.csv", ["name", "class", "cct_k", *_spectra_header()], rows)


def export_reflectances(out: Path) -> None:
    sets = s.test_reflectances()
    R_train = fit.training_reflectances()
    named = {
        "reflectances_training_190.csv": ("train", R_train),
        "reflectances_ces_99.csv": ("ces", sets["CIE 2017 CES (99)"]),
        "reflectances_colorchecker_24.csv": ("cc", sets["ColorChecker (Ohta, 24)"]),
        "reflectances_nist_cqs_vs_15.csv": ("vs", sets["NIST CQS VS (15)"]),
    }
    for fname, (tag, R) in named.items():
        _write_rows(out / fname, ["name", *_spectra_header()], [[f"{tag}{i + 1}", *R[:, i]] for i in range(R.shape[1])])


def export_constants(out: Path) -> None:
    ctx = ev.context()
    doc = {
        "ap0_to_xyz": s.AP0_TO_XYZ.tolist(),
        "aces_white_xy": [float(v) for v in s.ACES_WHITE_XY],
        "cat02_matrix": colour.adaptation.CHROMATIC_ADAPTATION_TRANSFORMS["CAT02"].tolist(),
        "research_test_lights": list(ctx.illums),
        "research_evaluation_set": {"CES 99": 99, "ColorChecker (Ohta) 24": 24, "NIST CQS VS 15": 15, "total": 138},
        "gate_lights": [L.name for L in lt.gate_lights()],
        "conventions": {
            "camera_rgb_wb": "raw = (E o R)^T S; white = E^T S; rgb = raw / white",
            "target_XYZ": "XYZ = (E o R)^T CMF / (CMF_y . E); CAT02 von Kries from the light's white to the ACES white",
            "fit_weights": "clip(Y_target, 1e-3)^(-2/3), Y from target_XYZ; the research objective of "
                           "sony_profiles (evaluate.py) and the start of the shipped Lab fit",
            "fit_objective": "shipped (gate.py, build_profiles): the white-preserving 3x3 minimising "
                             "sum_n |Lab(M_arri(cct) M c_s[n]) - Lab(M_arri(cct) c_a[n])|^2 over the 190 training "
                             "patches (fit.fit_map_lab, decision 2026-10-12)",
            "lab": "CIE Lab with the ACES white (xy 0.32168, 0.33767) on AP0 -> XYZ",
            "arri_matrix": "linear in mired between bracketing table entries at the light's CCT; native -> AP0",
        },
    }
    (out / "constants.json").write_text(json.dumps(doc, indent=1) + "\n")


def export_de2000(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    L = rng.uniform(5, 98, (200, 2))
    ab = rng.uniform(-90, 90, (200, 4))
    lab1 = np.stack([L[:, 0], ab[:, 0], ab[:, 1]], axis=1)
    lab2 = np.stack([L[:, 1], ab[:, 2], ab[:, 3]], axis=1)
    # near pairs too: ΔE00 is most delicate for small differences
    lab3 = lab1 + rng.normal(0, 1.0, lab1.shape)
    pairs1 = np.concatenate([lab1, lab1])
    pairs2 = np.concatenate([lab2, lab3])
    de = colour.difference.delta_E_CIE2000(pairs1, pairs2)
    doc = {"lab1": pairs1.tolist(), "lab2": pairs2.tolist(), "delta_e2000": [float(x) for x in de]}
    (out / "delta_e2000_vectors.json").write_text(json.dumps(doc) + "\n")


def export_expected(out: Path) -> dict:
    ctx = ev.context()
    cams = s.load_cameras()
    ref = cams[ev.REF_CAMERA]
    bodies = {}
    for prof in PROFILES:
        S = cams[prof.curves_name]
        entry = {}
        for tag, S_fit in (("own", S), ("a7r_iv", ref)):
            res = ev.map_error(S, S_fit, ctx)
            med, p95, worst, worst_med = ev.summary(res)
            entry[tag] = {
                "median": med, "p95": p95, "worst_light": worst, "worst_light_median": worst_med,
                "per_light": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in res.items()},
            }
        bodies[prof.id] = entry
    gate_doc = {p.id: gate.run_profile(p.id, ctx) for p in PROFILES}
    doc = {
        "sony_profiles": {
            "tolerance": 0.02,
            "note": "median / p95 of ΔE00 over 138 colours and 9 lights; map fitted per light on the 190 patches; "
                    "equals research/alexa-ssf/out/sony_profiles.csv after rounding to 3 decimals",
            "bodies": bodies,
        },
        "sensor_map_gate": {"median_max": gate.MEDIAN_MAX, "p95_max": gate.P95_MAX, "profiles": gate_doc},
    }
    (out / "expected.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def write_manifest(out: Path) -> None:
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    doc = {
        "generator": "python3 -m training.sensor_map.export_fixtures",
        "colour_science": colour.__version__,
        "licence_note": "Contains CIE data (CC BY-SA 4.0): generated for tests, never commit this folder.",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (out / "MANIFEST.json").write_text(json.dumps(doc, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=paths.FIXTURES)
    args = ap.parse_args(argv)
    paths.require_sources()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    export_cmf(out)
    export_lights(out)
    export_reflectances(out)
    export_constants(out)
    export_de2000(out)
    export_expected(out)
    write_manifest(out)
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {len(list(out.iterdir()))} files, {total / 1024:.0f} kB, to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
