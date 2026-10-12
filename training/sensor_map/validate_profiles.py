"""Validate the shipped camera data: profiles, registry and target.

    python3 -m training.sensor_map.validate_profiles

Needs only the committed files (no fetched sources, no colour-science), so it can run anywhere:

* every ``data/cameras/profiles/*/profile.json`` against ``profile.schema.json`` (JSON Schema
  2020-12), the registry against ``registry.schema.json``, ``data/alexa/target.json`` against
  ``target.schema.json``;
* **white preservation**: every sensor-map matrix satisfies ``M [1,1,1]^T = [1,1,1]^T`` within 1e-12;
* mired strictly increasing inside every class list (sensor map, fluorescent and LED ``w_A``),
  and the light names match ``provenance.reference_lights``;
* grids: lengths agree with the ranges (blackbody ``w_A`` 1500-25000 K, daylight ``w_A`` 4000-25000 K,
  locus 1500-4000 K and 4000-25000 K, 10 K step), green exactly 1 in every ``w_A``, finite values,
  log-chroma monotone along each branch, ``w_A`` positive, xy plausible;
* the curves file exists and matches ``provenance.curves_source.sha256``; the target file matches
  ``provenance.target.sha256``;
* **locus scale** (docs/02 C1): the blackbody locus equals the camera white of Planck's law integrated against
  the shipped curve file as published (clipped at 0, not normalised per channel) to 1e-6 at 1500-3900 K, so a
  locus built from peak-normalised curves (off by x1.6 in red, x1.2 in blue) fails;
* registry: every model ID once, one ``exact`` body per profile, ``via`` and ``default`` rules,
  ``model_strings`` agree with ``bodies``, and the model-ID table equals the table in
  ``docs/03-camera-profiles.md`` (when that file is present).

Exit status 0 when everything passes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from training.common import paths

WHITE_TOL = 1e-12
LOCUS_SCALE_TOL = 1e-6  # |ln(R/G) or ln(B/G) of the table - the same from the published curves|
LOCUS_SCALE_CHECK_K = (1500, 2000, 2500, 3200, 3900)
PLANCK_C2 = 0.014388  # m K, colour-science's value (the fit scripts' convention)
GRID_SPECS = {
    # (table path in the profile) -> (start_k, end_k)
    ("alexa_white", "blackbody"): (1500, 25000),
    ("alexa_white", "daylight"): (4000, 25000),
    ("locus", "blackbody"): (1500, 4000),
    ("locus", "daylight"): (4000, 25000),
    ("locus", "xy", "blackbody"): (1500, 4000),
    ("locus", "xy", "daylight"): (4000, 25000),
    ("locus", "duv_scale", "blackbody"): (1500, 4000),
    ("locus", "duv_scale", "daylight"): (4000, 25000),
}
GRID_STEP = {("locus", "duv_scale", "blackbody"): 100, ("locus", "duv_scale", "daylight"): 100}
CLASSES = ("daylight", "tungsten", "fluorescent", "led")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checked = 0

    def check(self, ok: bool, msg: str) -> None:
        self.checked += 1
        if not ok:
            self.errors.append(msg)

    def schema(self, instance, schema_path: Path, label: str) -> None:
        validator = Draft202012Validator(json.loads(schema_path.read_text()))
        errs = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
        self.checked += 1
        for e in errs[:10]:
            self.errors.append(f"{label}: schema: /{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}")
        if len(errs) > 10:
            self.errors.append(f"{label}: schema: ... {len(errs) - 10} more")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(d: dict, path: tuple[str, ...]):
    for k in path:
        d = d[k]
    return d


def planck_380_780(t_k: float) -> np.ndarray:
    """Planck's law on the 380-780 nm / 5 nm grid of the curve files (peak-normalisation is irrelevant here)."""
    lam = (380.0 + 5.0 * np.arange(81)) * 1e-9
    return 1.0 / (lam**5 * np.expm1(PLANCK_C2 / (lam * t_k)))


def published_curves(path: Path) -> np.ndarray:
    """A curve file as published: (81, 3), clipped at 0, **not** normalised (``fit.load_curves_raw`` without the import)."""
    doc = json.loads(path.read_text())
    data = {float(k): v for k, v in doc["spectral_data"]["data"]["main"].items()}
    return np.clip(np.array([data[380.0 + 5.0 * i] for i in range(81)], dtype=float), 0.0, None)


def check_locus_scale(rep: Report, doc: dict, label: str, curves: Path) -> None:
    """The profile's blackbody locus is the raw camera white of Planck's law through the published curves."""
    S = published_curves(curves)
    grid = doc["locus"]["blackbody"]
    worst = 0.0
    for t in LOCUS_SCALE_CHECK_K:
        i = (t - grid["start_k"]) // grid["step_k"]
        r = planck_380_780(float(t)) @ S
        want = np.log([r[0] / r[1], r[2] / r[1]])
        worst = max(worst, float(np.abs(np.array(grid["values"][i]) - want).max()))
    rep.check(worst <= LOCUS_SCALE_TOL,
              f"{label}: locus.blackbody differs from the camera white of the published curves by {worst:.2e} "
              f"(> {LOCUS_SCALE_TOL:g}): built from normalised curves? (docs/02 C1)")


def check_profile(rep: Report, path: Path, doc: dict) -> None:
    label = path.parent.name
    rep.schema(doc, paths.PROFILE_SCHEMA, label)
    if rep.errors and any(e.startswith(f"{label}: schema") for e in rep.errors):
        return  # structure is wrong; the numeric checks below would only raise noise
    rep.check(doc["id"] == label, f"{label}: id {doc['id']!r} does not match the folder name")

    # white preservation and mired order of the sensor map
    classes = doc["sensor_map"]["classes"]
    for klass in CLASSES:
        entries = classes[klass]
        mireds = [e["mired"] for e in entries]
        rep.check(all(b > a for a, b in zip(mireds, mireds[1:])), f"{label}: sensor_map.{klass}: mired not strictly increasing")
        rep.check([e["light"] for e in entries] == doc["provenance"]["reference_lights"][klass],
                  f"{label}: sensor_map.{klass}: lights differ from provenance.reference_lights")
        for e in entries:
            m = np.array(e["m"], dtype=float)
            err = float(np.abs(m @ np.ones(3) - 1.0).max())
            rep.check(err <= WHITE_TOL, f"{label}: sensor_map.{klass}.{e['light']}: |M.[1,1,1] - [1,1,1]| = {err:.2e} > {WHITE_TOL:g}")
            rep.check(bool(np.isfinite(m).all()), f"{label}: sensor_map.{klass}.{e['light']}: non-finite entry")
    # mired coordinates match the generating temperatures for the formula lights
    for klass, prefix in (("tungsten", "blackbody-"), ("daylight", "daylight-")):
        for e in classes[klass]:
            t = float(e["light"].removeprefix(prefix).removesuffix("K"))
            rep.check(abs(e["mired"] - 1e6 / t) < 1e-6, f"{label}: sensor_map.{klass}.{e['light']}: mired {e['mired']} != 1e6/{t}")

    # class lists of w_A
    for klass in ("fluorescent", "led"):
        entries = doc["alexa_white"][klass]
        mireds = [e["mired"] for e in entries]
        rep.check(all(b > a for a, b in zip(mireds, mireds[1:])), f"{label}: alexa_white.{klass}: mired not strictly increasing")
        rep.check([e["light"] for e in entries] == doc["provenance"]["reference_lights"][klass],
                  f"{label}: alexa_white.{klass}: lights differ from provenance.reference_lights")
        rep.check(mireds == [e["mired"] for e in classes[klass]], f"{label}: alexa_white.{klass} mired differs from sensor_map.{klass}")

    # grids
    for gpath, (lo, hi) in GRID_SPECS.items():
        g = _get(doc, gpath)
        name = ".".join(gpath)
        step = GRID_STEP.get(gpath, 10)
        rep.check(g["start_k"] == lo and g["step_k"] == step, f"{label}: {name}: start_k/step_k {g['start_k']}/{g['step_k']}, expected {lo}/{step}")
        n_expected = (hi - lo) // step + 1
        rep.check(len(g["values"]) == n_expected, f"{label}: {name}: {len(g['values'])} rows, expected {n_expected}")
        v = np.array(g["values"], dtype=float)
        rep.check(bool(np.isfinite(v).all()), f"{label}: {name}: non-finite value")
        if gpath[1] == "duv_scale":
            rep.check(bool(((v > 0.01) & (v < 1.0)).all()), f"{label}: {name}: implausible Duv scale")
        if gpath[0] == "alexa_white":
            rep.check(bool((v[:, 1] == 1.0).all()), f"{label}: {name}: green is not exactly 1")
            rep.check(bool((v > 0).all()), f"{label}: {name}: non-positive w_A")
    for klass in ("fluorescent", "led"):
        for e in doc["alexa_white"][klass]:
            rep.check(e["w"][1] == 1.0, f"{label}: alexa_white.{klass}.{e['light']}: green is not 1")

    # locus: log-chroma moves monotonically with temperature (R/G falls, B/G rises) on both branches
    for branch in ("blackbody", "daylight"):
        v = np.array(doc["locus"][branch]["values"], dtype=float)
        d = np.diff(v, axis=0)
        # The CIE D series as colour-science builds it rounds M1 and M2 to three decimals, so the
        # daylight branch is a staircase of ~1e-4 around a smooth curve; above ~18000 K that is
        # larger than the 10 K step and single steps can go the wrong way by up to ~1e-4.
        slack = 2e-4 if branch == "daylight" else 1e-8
        rep.check(bool((d[:, 0] <= slack).all() and (d[:, 1] >= -slack).all()),
                  f"{label}: locus.{branch}: ln(R/G) not non-increasing or ln(B/G) not non-decreasing (slack {slack:g})")
        rep.check(bool(v[-1, 0] < v[0, 0] and v[-1, 1] > v[0, 1]), f"{label}: locus.{branch}: wrong overall direction")
        xy = np.array(doc["locus"]["xy"][branch]["values"], dtype=float)
        rep.check(bool(((xy > 0.15) & (xy < 0.75)).all()), f"{label}: locus.xy.{branch}: implausible chromaticity")

    # files
    curves = paths.CAMERAS_DIR / doc["spectral_curves"]
    rep.check(curves.exists(), f"{label}: spectral_curves {doc['spectral_curves']} not found")
    if curves.exists():
        rep.check(sha256_file(curves) == doc["provenance"]["curves_source"]["sha256"],
                  f"{label}: curves SHA-256 differs from provenance.curves_source.sha256")
        check_locus_scale(rep, doc, label, curves)
    if paths.ALEXA_CURVES_JSON.exists():
        rep.check(sha256_file(paths.ALEXA_CURVES_JSON) == doc["provenance"]["target"]["sha256"],
                  f"{label}: target curves SHA-256 differs from provenance.target.sha256")
    rep.check(doc["decode"]["quirks"] == [], f"{label}: decode.quirks should be empty in V1")


DOC_ROW = re.compile(r"^\|(?P<bodies>[^|]+)\|\s*(?P<profile>sony-[a-z0-9-]+)\s*\|\s*(?P<match>exact|same sensor|default)\s*\|\s*$")


def docs_table() -> dict[int, tuple[str, str]]:
    """model ID -> (profile, match) parsed from docs/03's 'Which profile each camera gets' table."""
    doc = paths.REPO_ROOT / "docs" / "03-camera-profiles.md"
    if not doc.exists():
        return {}
    out = {}
    for line in doc.read_text().splitlines():
        m = DOC_ROW.match(line)
        if not m:
            continue
        match = {"same sensor": "sensor_family"}.get(m["match"], m["match"])
        for mid in re.findall(r"\((\d+)\)", m["bodies"]):
            out[int(mid)] = (m["profile"], match)
    return out


def check_registry(rep: Report, reg: dict, profile_ids: set[str]) -> None:
    rep.schema(reg, paths.REGISTRY_SCHEMA, "registry")
    if any(e.startswith("registry: schema") for e in rep.errors):
        return
    ids = [b["model_id"] for b in reg["bodies"]]
    models = [b["model"] for b in reg["bodies"]]
    rep.check(len(ids) == len(set(ids)), "registry: duplicate model_id")
    rep.check(len(models) == len(set(models)), "registry: duplicate model string")
    listed = {p["id"] for p in reg["profiles"]}
    rep.check(listed == profile_ids, f"registry: profiles {sorted(listed)} differ from the files {sorted(profile_ids)}")
    rep.check(reg["default_profile"] in listed, "registry: default_profile is not a listed profile")
    for p in reg["profiles"]:
        rep.check((paths.CAMERAS_DIR / p["path"]).exists(), f"registry: {p['path']} not found")
    by_profile: dict[str, list[dict]] = {pid: [] for pid in listed}
    for b in reg["bodies"]:
        rep.check(b["profile"] in listed, f"registry: {b['model']}: unknown profile {b['profile']}")
        if b["match"] == "default":
            rep.check(b["profile"] == reg["default_profile"], f"registry: {b['model']}: default body must use default_profile")
        if b["match"] == "exact":
            by_profile.setdefault(b["profile"], []).append(b)
    for pid, exact in by_profile.items():
        rep.check(len(exact) == 1, f"registry: profile {pid} has {len(exact)} exact bodies, expected 1")
    exact_body = {b["profile"]: b["body"] for b in reg["bodies"] if b["match"] == "exact"}
    for b in reg["bodies"]:
        if b["match"] == "sensor_family":
            rep.check(b["via"] == exact_body.get(b["profile"]), f"registry: {b['model']}: via {b['via']!r} is not the measured body of {b['profile']}")
    by_model = {b["model"]: b["model_id"] for b in reg["bodies"]}
    rep.check(reg["model_strings"] == by_model, "registry: model_strings disagree with bodies")
    # the table in docs/03
    table = docs_table()
    if table:
        ours = {b["model_id"]: (b["profile"], b["match"]) for b in reg["bodies"]}
        rep.check(ours == table, f"registry: differs from docs/03: only in registry {sorted(set(ours.items()) - set(table.items()))}, "
                                 f"only in docs {sorted(set(table.items()) - set(ours.items()))}")
        rep.check(len(table) >= 25, f"docs/03 table parse found only {len(table)} model IDs")


def check_target(rep: Report) -> None:
    doc = json.loads(paths.ALEXA_TARGET_JSON.read_text())
    rep.schema(doc, paths.TARGET_SCHEMA, "target")
    for rel in ("spectral_curves", "ensemble_csv", "variants_csv"):
        rep.check((paths.DATA / doc[rel]).exists(), f"target: {doc[rel]} not found")
    for fname, digest in doc["provenance"]["sha256"].items():
        f = paths.ALEXA_DIR / fname
        rep.check(f.exists() and sha256_file(f) == digest, f"target: SHA-256 of {fname} differs")


def main() -> int:
    rep = Report()
    profile_files = sorted(paths.PROFILES_DIR.glob("*/profile.json"))
    rep.check(len(profile_files) == 11, f"expected 11 profile files, found {len(profile_files)}")
    ids = set()
    for pf in profile_files:
        doc = json.loads(pf.read_text())
        ids.add(pf.parent.name)
        check_profile(rep, pf, doc)
    check_registry(rep, json.loads(paths.REGISTRY_JSON.read_text()), ids)
    check_target(rep)

    print(f"{len(profile_files)} profiles, registry and target: {rep.checked} checks")
    if rep.errors:
        for e in rep.errors:
            print("FAIL", e)
        print(f"{len(rep.errors)} problem(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
