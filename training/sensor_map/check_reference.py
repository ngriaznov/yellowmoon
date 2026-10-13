"""Check that ``crates/ym-color/tests/data/spectral_reference.json`` agrees with the shipped profile.

    python3 -m training.sensor_map.check_reference

Needs only committed files (no fetched sources, no colour-science). It verifies

* the file was generated from the current ``profile.json`` (SHA-256 of its content without the ``provenance``
  block, recorded in ``sources``; the bytes of the file are not pinned);
* every ``interpolated`` / ``cct_duv`` / ``sensor_map_blend`` field is reproduced by
  ``runtime_ref`` to 1e-9 (the bar the Rust port has to meet);
* grid-point ``exact`` values agree with the table to ``on_grid_abs`` (1e-6), and off-grid ``exact``
  values to the stated per-table tolerance.

Run ``python3 -m training.sensor_map.spectral_reference`` to regenerate after the profile changes.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from training.common import content_digest, paths
from training.sensor_map import runtime_ref as rr


def main() -> int:
    ref = json.loads(paths.SPECTRAL_REFERENCE_JSON.read_text())
    prof_path = paths.REPO_ROOT / ref["sources"]["profile"]
    problems: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        if not ok:
            problems.append(msg)

    def close(a, b, tol: float) -> bool:
        return bool(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) <= tol)

    check(content_digest.digest(json.loads(prof_path.read_text()), skip_top_level=content_digest.PROFILE_SKIPS)
          == ref["sources"]["profile_content_sha256"],
          "reference is stale: the content of profile.json (provenance aside) changed since spectral_reference.json was generated")
    check(hashlib.sha256(paths.ALEXA_CURVES_JSON.read_bytes()).hexdigest() == ref["sources"]["alexa_curves_sha256"],
          "reference is stale: the ALEXA curves changed")
    profile = rr.Profile.load(prof_path)
    # the camera locus in the file is computed from the shipped curve file in its published raw scale (docs/02 C1)
    curves_path = paths.CAMERAS_DIR / json.loads(prof_path.read_text())["spectral_curves"]
    check(hashlib.sha256(curves_path.read_bytes()).hexdigest() == ref["sources"]["camera_curves_sha256"],
          "reference is stale: the camera curves changed")
    check("raw scale" in ref["sources"]["camera_curve_scale"],
          "reference was generated without the raw-scale camera curves (docs/02 C1)")
    tol = ref["tolerances"]
    on_grid, interp_tol, off = tol["on_grid_abs"], tol["interpolated_abs"], tol["off_grid_vs_exact_abs"]

    def w_tol(kind: str, t: float) -> float:
        if kind == "blackbody":
            return off["blackbody_w_a_above_2000k"] if t >= 2000 else off["blackbody_w_a"]
        return off["daylight_w_a"]

    def c_tol(kind: str, t: float) -> float:
        if kind == "blackbody":
            return off["blackbody_locus_above_2000k"] if t >= 2000 else off["blackbody_locus"]
        return off["daylight_locus"]

    for section in ("grid_points", "off_grid"):
        for kind, cases in ref["alexa_w_a"][section].items():
            for c in cases:
                t = c["cct_k"]
                got = rr.w_a(profile, t, kind)
                check(close(got, c["interpolated"], interp_tol), f"w_A {kind} {t}: interpolated differs")
                bound = on_grid if section == "grid_points" else w_tol(kind, t)
                check(close(got, c["exact"], bound), f"w_A {kind} {t}: |table - exact| > {bound:g} ({section})")
        for kind, cases in ref["camera_locus"][section].items():
            grid = profile.chroma_blackbody if kind == "blackbody" else profile.chroma_daylight
            xyg = profile.xy_blackbody if kind == "blackbody" else profile.xy_daylight
            for c in cases:
                t = c["cct_k"]
                check(close(grid.lerp_mired(t), c["ln_chroma_interpolated"], interp_tol), f"locus {kind} {t}: interpolated differs")
                check(close(xyg.lerp_mired(t), c["xy_interpolated"], interp_tol), f"xy {kind} {t}: interpolated differs")
                bound = on_grid if section == "grid_points" else c_tol(kind, t)
                check(close(grid.lerp_mired(t), c["ln_chroma_exact"], bound), f"locus {kind} {t}: |table - exact| > {bound:g}")
                check(close(xyg.lerp_mired(t), c["xy_exact"], max(bound, 1e-5)), f"xy {kind} {t}: |table - exact| too large")

    for c in ref["cct_duv"]["cases"]:
        cct, duv = rr.rgb_to_cct_duv(profile, c["camera_rgb"])
        check(abs(cct - c["cct_k"]) <= 1e-6 and abs(duv - c["duv"]) <= 1e-9, f"cct/duv case {c['name']!r} differs")
    for c in ref["cct_duv"]["inverse_cases"]:
        rgb = rr.cct_duv_to_rgb(profile, c["cct_k"], c["duv"])
        check(close(rgb, c["camera_rgb"], 1e-9), f"inverse case {c['cct_k']} K / {c['duv']}: rgb differs")
        cct, duv = rr.rgb_to_cct_duv(profile, rgb)
        check(abs(cct - c["roundtrip_cct_k"]) <= 1e-6 and abs(duv - c["roundtrip_duv"]) <= 1e-9, f"round trip {c['cct_k']} K differs")
    for c in ref["sensor_map_blend"]:
        m = rr.sensor_map(profile, c["probs"], c["mired"])
        check(close(m, c["m"], 1e-12), f"blend {c['probs']} @ {c['mired']}: M differs")
        check(close(m.sum(axis=1), 1.0, 1e-12), f"blend {c['probs']} @ {c['mired']}: not white preserving")
    for c in ref["class_w_a"]:
        w = rr.interp_class_list(profile.raw["alexa_white"][c["class"]], c["mired"], "w")
        check(close(w, c["w"], 1e-12), f"class w_A {c['class']} @ {c['mired']} differs")

    print(f"spectral_reference.json vs {ref['sources']['profile']}: {checks} checks")
    for p in problems:
        print("FAIL", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
