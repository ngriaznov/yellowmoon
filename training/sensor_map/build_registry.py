"""Write ``data/cameras/registry.json`` from the table in ``docs/03`` (``registry_data.py``).

    python3 -m training.sensor_map.build_registry

Layout (schema: ``crates/ym-profiles/schema/registry.schema.json``):

``bodies``           one entry per Sony model ID listed in docs/03 (exact bodies, same-sensor
                     siblings and the named default bodies):
                     ``{model_id, model, body, profile, match, via, notes, white_scale, white_scale_source}``
                     (the white scale of ``docs/03``, "White scale of stand-in bodies", from
                     ``training/white_scale/white_scale.json``) and, where Sony publishes them,
                     ``{sensor_width_mm, active_width_px, sensor_source_kind, sensor_source}`` (the pixel
                     pitch of ``docs/algorithms/03-diffraction-deconvolution.md`` section 2.2). A body with
                     no published pixel width has no sensor fields and uses the EXIF crop-factor fallback.
``model_strings``    Model string (IFD0 0x0110) -> model ID, for files whose maker note has no usable
                     0xB001. Model strings are the ones the cameras write for each ID.
``default_profile``  the stand-in for every other Sony file (A7R IV).
``profiles``         profile id -> path relative to ``data/cameras/``.

Lookup (docs/03): exact model ID -> sensor family -> default. A model ID not in ``bodies``
gets ``default_profile`` with ``match = "default"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.common import jsonfmt, paths
from training.sensor_map import registry_data as rd

REGISTRY_VERSION = "1.2.0"
WHITE_SCALE_JSON = Path(__file__).resolve().parents[1] / "white_scale" / "white_scale.json"


def _check_text(model_id: int) -> str:
    diff = rd.SIBLING_MAX_ABS_DIFF[model_id]
    verdict = "pass" if diff <= rd.SIBLING_LIMIT else "fail"
    return (f"ColorMatrix sibling check ({verdict}): max |diff| {diff:.4f} per coefficient against the profile body, "
            f"limit {rd.SIBLING_LIMIT}, published colour matrices")


def _notes(b: rd.Body) -> str:
    prof = rd.PROFILE_BY_ID[b.profile]
    if b.match == "exact":
        return f"Measured by Weta (rawtoaces-data): {prof.spectral_curves}."
    if b.match == "sensor_family":
        return (f"Same sensor as {prof.body}; shares its profile. {_check_text(b.model_id)} (docs/03).")
    note = (f"No measured profile; uses the {prof.body} stand-in (docs/03). "
            "A Sony file is never refused for lack of a profile.")
    if b.model_id in rd.REJECTED_SIBLINGS:
        note += (f" Considered as a same-sensor sibling of the {prof.body}, but it did not pass: "
                 f"{_check_text(b.model_id)}.")
    return note


def _sensor_fields(b: rd.Body) -> dict:
    s = rd.SENSORS.get(b.model_id)
    if s is None:
        return {}
    return {
        "sensor_width_mm": s.width_mm,
        "active_width_px": s.width_px,
        "sensor_source_kind": s.kind,
        "sensor_source": s.source,
    }


def _white_scale_fields(b: rd.Body, table: dict) -> dict:
    """``white_scale`` and ``white_scale_source`` of a body from ``training/white_scale/white_scale.json``."""
    entry = table["bodies"][str(b.model_id)]
    source = dict(entry["source"])
    return {"white_scale": entry["white_scale"], "white_scale_source": source}


def build() -> dict:
    table = json.loads(WHITE_SCALE_JSON.read_text())
    bodies = []
    for b in rd.BODIES:
        bodies.append({
            "model_id": b.model_id,
            "model": b.model,
            "body": b.body,
            "profile": b.profile,
            "match": b.match,
            "via": rd.body_name_of_profile(b.profile) if b.match == "sensor_family" else None,
            "notes": _notes(b),
            **_white_scale_fields(b, table),
            **_sensor_fields(b),
        })
    model_strings = {b.model: b.model_id for b in sorted(rd.BODIES, key=lambda x: x.model)}
    return {
        "version": REGISTRY_VERSION,
        "default_profile": rd.DEFAULT_PROFILE,
        "sources": {
            "model_ids": rd.MODEL_ID_SOURCE,
            "table": "docs/03-camera-profiles.md, 'Which profile each camera gets'",
            "sensors": "Sony specification pages (sensor size in mm; the width in pixels of the largest 3:2 still "
                       f"image, full frame): {rd.SENSOR_READ_NOTE}. Used for the pixel pitch of docs/algorithms/03 "
                       "section 2.2; sensor_source_kind is 'maker' when the excerpt came from a Sony page and "
                       "'restated' when a retailer or reference site restated the Sony specification.",
        },
        "profiles": [
            {"id": p.id, "path": f"profiles/{p.id}/profile.json"} for p in rd.PROFILES
        ],
        "bodies": bodies,
        "model_strings": model_strings,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    text = jsonfmt.dumps(build())
    paths.REGISTRY_JSON.write_text(text)
    print("wrote", paths.REGISTRY_JSON.relative_to(paths.REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
