#!/usr/bin/env python3
"""Checks that the app bundle carries the licence texts its notices promise (the part of THIRD-PARTY.md that no generator can do).

    python3 tools/licences/check_bundle.py [REPO]

* ``src-tauri/tauri.conf.json`` ``bundle.resources`` lists ``NOTICE``, ``LICENSE`` and the generated ``THIRD-PARTY.md``
  (tools/licences/generate.py; the file is written by ``release.yml`` and by the CI job ``licences``).
* ``NOTICE`` reproduces the BSD 3-Clause conditions of the ported Menon demosaic (crates/ym-gpu/src/cpu/demosaic.rs), which a
  binary distribution must carry (clause 2), and not only the copyright line.
* ``NOTICE`` carries the MIT notice of glasscn (src/components/glass/), whose source is part of the interface.

Prints one line per problem and exits 1 when there is one. The bundle's other resources (data/...) are checked by tauri-build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Resources that must be in the bundle: published name inside the bundle -> why.
REQUIRED_RESOURCES = {
    "NOTICE": "attributions (About points to it)",
    "LICENSE": "the Apache-2.0 text of Yellowmoon itself",
    "THIRD-PARTY.md": "the licence texts of the Rust crates, JavaScript packages and fonts (NOTICE item 9, About)",
}

#: Phrases a complete BSD 3-Clause text has, and what NOTICE must carry for colour-demosaicing (item 8) and for glasscn (item 11).
NOTICE_PHRASES = (
    ("Colour Developers", "the copyright holder of the ported Menon demosaic"),
    ("Redistributions in binary form must reproduce", "the BSD 3-Clause conditions (clause 2) for the Menon demosaic"),
    ("Neither the name of", "the BSD 3-Clause non-endorsement clause"),
    ("THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS", "the BSD 3-Clause disclaimer"),
    ("Tim Mikeladze", "the copyright holder of the glasscn components (MIT), whose notice the licence requires in every copy"),
    ("The above copyright notice and this permission notice shall be included in all", "the MIT conditions of the glasscn components"),
)


def problems(root: Path) -> list[str]:
    """Everything that is missing, one sentence each."""
    found: list[str] = []
    config = root / "src-tauri" / "tauri.conf.json"
    try:
        resources = json.loads(config.read_text(encoding="utf-8")).get("bundle", {}).get("resources", {})
    except (OSError, json.JSONDecodeError) as error:
        return [f"{config} cannot be read: {error}"]
    published = set(resources.values()) if isinstance(resources, dict) else set(resources)
    for name, why in REQUIRED_RESOURCES.items():
        if name not in published:
            found.append(f"src-tauri/tauri.conf.json bundle.resources has no {name}: {why}")
    notice = root / "NOTICE"
    try:
        text = notice.read_text(encoding="utf-8")
    except OSError as error:
        return found + [f"{notice} cannot be read: {error}"]
    for phrase, why in NOTICE_PHRASES:
        if phrase not in text:
            found.append(f'NOTICE does not contain "{phrase}": {why}')
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[0]).resolve() if argv else REPO
    found = problems(root)
    for line in found:
        print(f"error: {line}")
    if not found:
        print("bundle check: NOTICE, LICENSE and THIRD-PARTY.md are bundled; NOTICE carries the BSD conditions")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
