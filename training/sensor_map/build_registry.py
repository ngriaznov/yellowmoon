"""Write ``data/cameras/registry.json`` from the table in ``docs/03`` (``registry_data.py``).

    python3 -m training.sensor_map.build_registry

Layout (schema: ``crates/ym-profiles/schema/registry.schema.json``):

``bodies``           one entry per Sony model ID listed in docs/03 (exact bodies, same-sensor
                     siblings and the named default bodies):
                     ``{model_id, model, body, profile, match, via, notes}``
``model_strings``    Model string (IFD0 0x0110) -> model ID, for files whose maker note has no usable
                     0xB001. Model strings are those ExifTool's Sony tables use for each ID.
``default_profile``  the stand-in for every other Sony file (A7R IV).
``profiles``         profile id -> path relative to ``data/cameras/``.

Lookup (docs/03): exact model ID -> sensor family -> default. A model ID not in ``bodies``
gets ``default_profile`` with ``match = "default"``.
"""

from __future__ import annotations

import argparse
import sys

from training.common import jsonfmt, paths
from training.sensor_map import registry_data as rd

REGISTRY_VERSION = "1.0.0"


def _notes(b: rd.Body) -> str:
    prof = rd.PROFILE_BY_ID[b.profile]
    if b.match == "exact":
        return f"Measured by Weta (rawtoaces-data): {prof.spectral_curves}."
    if b.match == "sensor_family":
        return (f"Same sensor as {prof.body}; shares its profile. ColorMatrix sibling check: "
                f"{rd.SIBLING_CHECK} (docs/03).")
    return (f"No measured profile; uses the {prof.body} stand-in (docs/03). "
            "A Sony file is never refused for lack of a profile.")


def build() -> dict:
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
        })
    model_strings = {b.model: b.model_id for b in sorted(rd.BODIES, key=lambda x: x.model)}
    return {
        "version": REGISTRY_VERSION,
        "default_profile": rd.DEFAULT_PROFILE,
        "sources": {
            "model_ids": rd.EXIFTOOL_SOURCE,
            "table": "docs/03-camera-profiles.md, 'Which profile each camera gets'",
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
