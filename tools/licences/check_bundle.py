#!/usr/bin/env python3
"""Checks that the app bundle carries the licence texts its notices promise (the part of THIRD-PARTY.md that no generator can do).

    python3 tools/licences/check_bundle.py [--require-weights] [REPO]

* ``src-tauri/tauri.conf.json`` ``bundle.resources`` lists ``NOTICE``, ``LICENSE`` and the generated ``THIRD-PARTY.md``
  (tools/licences/generate.py; the file is written by ``release.yml`` and by the CI job ``licences``).
* ``NOTICE`` reproduces the BSD 3-Clause conditions of the ported Menon demosaic (crates/ym-gpu/src/cpu/demosaic.rs), which a
  binary distribution must carry (clause 2), and not only the copyright line.
* ``NOTICE`` carries the MIT notice of glasscn (src/components/glass/), whose source is part of the interface.
* ``bundle.resources`` carries ``data/models``, and the SAM 2.1 small weights' folder (``data/models/sam2/``, N-16, ``docs/12``,
  ``docs/14``) has its model card with the Apache-2.0 terms, a ``SHA256SUMS`` line for the weights, a ``.gitignore`` that keeps the
  weights out of git, and ``NOTICE`` carries Meta's notice with the link to the repository.
* With ``--require-weights`` (``release.yml`` and ``build.yml``, after ``tools/fetch/sam2.sh``) the weights file must also be there
  and its SHA-256 must be the one in ``SHA256SUMS``; a placeholder line in ``SHA256SUMS`` is an error then and only a warning
  otherwise (the committed tree has no weights, so the CI job ``licences`` runs without the flag).
* ``bundle.resources`` carries ``data/film`` and ``NOTICE`` has the items for the spektrafilm profiles (Andrea Volpato, CC BY-SA 4.0,
  ``docs/24`` section 6) and the CIE tables. ``data/film/spektrafilm/`` and ``data/film/cie/`` each have ``LICENSE`` and
  ``ATTRIBUTION.md``, and a ``SHA256SUMS`` that lists every other file (``LICENSE`` included), with hashes that match, and no
  line for a file that is gone. Removing the credit ends the CC BY-SA grant (section 6(a)), so this is a release stopper.

Prints one line per problem and exits 1 when there is one. The bundle's other resources (data/...) are checked by tauri-build.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Resources that must be in the bundle: published name inside the bundle -> why.
REQUIRED_RESOURCES = {
    "NOTICE": "attributions (About points to it)",
    "LICENSE": "the Apache-2.0 text of Yellowmoon itself",
    "THIRD-PARTY.md": "the licence texts of the Rust crates, JavaScript packages and fonts (NOTICE item 8, About)",
    "data/film": "the film profiles and CIE tables with their CC BY-SA 4.0 licences (docs/24 section 6)",
    "data/models": "the models, among them the SAM 2.1 small weights that tools/fetch/sam2.sh puts in data/models/sam2/ (docs/12)",
}

#: The SAM 2.1 small weights (N-16): the folder, the file, and the pin a SHA256SUMS line has before the models workflow has run.
SAM2_DIR = Path("data") / "models" / "sam2"
SAM2_WEIGHTS = "sam2.1_hiera_small.f16.safetensors"
SAM2_PLACEHOLDER = "TBD-after-first-run"

#: Phrases a complete BSD 3-Clause text has, and what NOTICE must carry for colour-demosaicing (item 7) and for glasscn (item 9).
NOTICE_PHRASES = (
    ("Colour Developers", "the copyright holder of the ported Menon demosaic"),
    ("Redistributions in binary form must reproduce", "the BSD 3-Clause conditions (clause 2) for the Menon demosaic"),
    ("Neither the name of", "the BSD 3-Clause non-endorsement clause"),
    ("THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS", "the BSD 3-Clause disclaimer"),
    ("Tim Mikeladze", "the copyright holder of the glasscn components (MIT), whose notice the licence requires in every copy"),
    ("The above copyright notice and this permission notice shall be included in all", "the MIT conditions of the glasscn components"),
    ("Meta Platforms, Inc. and affiliates", "the copyright holder of the SAM 2.1 small weights (Apache-2.0), whose notice the licence requires in every copy"),
    ("github.com/facebookresearch/sam2", "the link to the SAM 2.1 weights' source (Apache-2.0 section 4)"),
    ("Andrea Volpato", "the author of the spektrafilm profiles (CC BY-SA 4.0), who must be named in every copy"),
    ("github.com/andreavolpato/spektrafilm", "the canonical source of the spektrafilm profiles, which the licence requires"),
    ("data/film/spektrafilm/LICENSE", "the pointer to the spektrafilm licence text"),
    ("International Commission on Illumination", "the publisher of the CIE tables (CC BY-SA 4.0)"),
    ("data/film/cie/LICENSE", "the pointer to the CIE tables' licence text"),
)

#: The two data folders under data/film/ that carry third-party CC BY-SA 4.0 data, and the files each must have.
FILM_DIRS = (Path("data") / "film" / "spektrafilm", Path("data") / "film" / "cie")
FILM_REQUIRED = ("LICENSE", "ATTRIBUTION.md", "SHA256SUMS")
#: Files of those folders that SHA256SUMS does not list: itself, and the attribution text we write.
FILM_UNHASHED = ("SHA256SUMS", "ATTRIBUTION.md")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sam2_pin(root: Path) -> str | None:
    """The first field of the weights' line in ``data/models/sam2/SHA256SUMS``, or None when there is no such line."""
    try:
        text = (root / SAM2_DIR / "SHA256SUMS").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == SAM2_WEIGHTS:
            return fields[0]
    return None


def warnings(root: Path) -> list[str]:
    """What is not an error yet: the weights' pin is still the placeholder written before the models workflow ran."""
    pin = sam2_pin(root)
    if pin is not None and not re.fullmatch(r"[0-9a-f]{64}", pin):
        return [
            f"{SAM2_DIR}/SHA256SUMS still holds '{pin}' for {SAM2_WEIGHTS}: run the models workflow and copy its hashes "
            "(a build with --require-weights stops here)"
        ]
    return []


def sam2_problems(root: Path, require_weights: bool) -> list[str]:
    """The SAM 2.1 small weights' folder: what is committed, and with ``require_weights`` the fetched file as well."""
    found: list[str] = []
    folder = root / SAM2_DIR
    card = folder / "MODEL_CARD.md"
    try:
        card_text = card.read_text(encoding="utf-8")
    except OSError as error:
        found.append(f"{card} cannot be read: {error}")
    else:
        for phrase, why in (("Apache-2.0", "the licence of the weights"), ("github.com/facebookresearch/sam2", "the source of the weights")):
            if phrase not in card_text:
                found.append(f'{SAM2_DIR}/MODEL_CARD.md does not contain "{phrase}": {why}')
    ignore = folder / ".gitignore"
    try:
        ignored = ignore.read_text(encoding="utf-8")
    except OSError as error:
        found.append(f"{ignore} cannot be read: {error} (the weights must never be committed)")
    else:
        if "safetensors" not in ignored:
            found.append(f"{SAM2_DIR}/.gitignore does not ignore *.safetensors: the weights are too big for git")
    pin = sam2_pin(root)
    if pin is None:
        found.append(f"{SAM2_DIR}/SHA256SUMS has no line for {SAM2_WEIGHTS}")
    if not require_weights:
        return found
    weights = folder / SAM2_WEIGHTS
    if pin is not None and not re.fullmatch(r"[0-9a-f]{64}", pin):
        found.append(f"{SAM2_DIR}/SHA256SUMS holds '{pin}' for {SAM2_WEIGHTS}, not a SHA-256: copy the hashes of the models workflow's SHA256SUMS asset")
    elif not weights.is_file():
        found.append(f"{weights} is missing: run tools/fetch/sam2.sh before the build, or the app would have no Object masks")
    elif pin is not None and sha256_file(weights) != pin:
        found.append(f"{weights} does not have the SHA-256 in {SAM2_DIR}/SHA256SUMS: fetch it again with tools/fetch/sam2.sh")
    return found


def film_problems(root: Path) -> list[str]:
    """``data/film/spektrafilm/`` and ``data/film/cie/``: LICENSE, ATTRIBUTION.md, and a SHA256SUMS that matches every file."""
    found: list[str] = []
    for rel in FILM_DIRS:
        folder = root / rel
        name = rel.as_posix()
        missing = [f for f in FILM_REQUIRED if not (folder / f).is_file()]
        found += [f"{name}/{f} is missing: the CC BY-SA 4.0 grant needs the licence, the credit and the hashes beside the data" for f in missing]
        if "SHA256SUMS" in missing:
            continue
        listed: dict[str, str] = {}
        for line in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]):
                listed[fields[1].strip().lstrip("*")] = fields[0]
            elif line.strip():
                found.append(f"{name}/SHA256SUMS has a line that is not '<sha-256>  <path>': {line.strip()[:60]}")
        present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        for path in sorted(present - set(FILM_UNHASHED) - set(listed)):
            found.append(f"{name}/{path} is not listed in {name}/SHA256SUMS: add it with tools/film/ or remove the file")
        for path in sorted(set(listed) - present):
            found.append(f"{name}/SHA256SUMS lists {path}, which is missing")
        for path in sorted(set(listed) & present):
            if sha256_file(folder / path) != listed[path]:
                found.append(f"{name}/{path} does not match its line in {name}/SHA256SUMS: the data was changed (it must stay unmodified)")
    return found


def problems(root: Path, require_weights: bool = False) -> list[str]:
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
    return found + sam2_problems(root, require_weights) + film_problems(root)


def main(argv: list[str]) -> int:
    require_weights = "--require-weights" in argv
    paths = [arg for arg in argv if not arg.startswith("--")]
    unknown = [arg for arg in argv if arg.startswith("--") and arg != "--require-weights"]
    if unknown:
        print(f"error: unknown option {unknown[0]}; usage: check_bundle.py [--require-weights] [REPO]")
        return 2
    root = Path(paths[0]).resolve() if paths else REPO
    found = problems(root, require_weights)
    for line in found:
        print(f"error: {line}")
    for line in warnings(root):
        if not require_weights:
            print(f"warning: {line}")
    if not found:
        weights = "; the SAM 2.1 weights are in place and match SHA256SUMS" if require_weights else ""
        print(f"bundle check: NOTICE, LICENSE, THIRD-PARTY.md, data/models and data/film are bundled; NOTICE carries the BSD, MIT, Meta, spektrafilm and CIE notices; the film data folders match their sums{weights}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
