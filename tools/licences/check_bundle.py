#!/usr/bin/env python3
"""Checks that the app bundle carries the licence texts its notices promise (the part of THIRD-PARTY.md that no generator can do).

    python3 tools/licences/check_bundle.py [--require-weights [--depth-pending]] [--require-mediapipe] [REPO]

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
* The same for the Depth Anything V2-Small weights (``data/models/depth/``, D-02, ``docs/07``, ``docs/14``): the model card with the
  Apache-2.0 terms and the link to the authors' repository, a ``.gitignore`` for the weights, a ``SHA256SUMS`` line, and in ``NOTICE``
  item 13. With ``--require-weights`` the depth weights are required like SAM's, except with ``--depth-pending``, which ``build.yml``
  and ``release.yml`` pass only while ``data/models/depth/SHA256SUMS`` still holds the placeholder (the ``depth`` job of
  ``models.yml`` has not run yet); a build with the flag has no Depth masks and says so.
* The MediaPipe face networks (``data/models/mediapipe/``, S-08, ``docs/27``, ``docs/07``, ``docs/14``; Google, Apache-2.0): the model
  card with the Apache-2.0 terms and the link to Google's model bucket, a ``.gitignore`` for the weights, a ``SHA256SUMS`` line for
  each of the three files (real hashes from the start: the conversion is deterministic), and in ``NOTICE`` item 14. The files are
  required only with ``--require-mediapipe``, which ``build.yml`` and ``release.yml`` pass once the release ``models-mediapipe-faces-1``
  exists (the ``mediapipe`` job of ``models.yml`` has run); a build without the flag has no face finder and says so.
* The fixtures of the Film node's oracle (``crates/ym-film/tests/oracle/``, F-06) are checked like the data folders below: they are
  numbers computed from the CC BY-SA profiles, so they need ``LICENSE``, ``ATTRIBUTION.md`` and ``SHA256SUMS`` too.
* ``bundle.resources`` carries ``data/film`` and ``NOTICE`` has the items for the spektrafilm profiles (Andrea Volpato, CC BY-SA 4.0,
  ``docs/24`` section 6) and the CIE tables, whose item 12 reads "and a table computed from it". ``data/film/spektrafilm/``, ``data/film/cie/`` and
  ``data/film/upsampling/`` (our table, computed from the CIE observer, CC BY-SA 4.0) each have ``LICENSE`` and
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
    "data/models": "the models, among them the SAM 2.1 small weights that tools/fetch/sam2.sh puts in data/models/sam2/ the depth weights that tools/fetch/depth.sh puts in data/models/depth/ and the MediaPipe face networks that tools/fetch/mediapipe.sh puts in data/models/mediapipe/ (docs/12)",
}

#: The SAM 2.1 small weights (N-16): the folder, the file, and the pin a SHA256SUMS line has before the models workflow has run.
SAM2_DIR = Path("data") / "models" / "sam2"
SAM2_WEIGHTS = "sam2.1_hiera_small.f16.safetensors"
SAM2_PLACEHOLDER = "TBD-after-first-run"
#: The Depth Anything V2-Small weights (D-02): the same layout, fetched by tools/fetch/depth.sh.
DEPTH_DIR = Path("data") / "models" / "depth"
DEPTH_WEIGHTS = "depth_anything_v2_vits.f16.safetensors"

#: The MediaPipe face networks (S-08): the folder and the three files fetched by tools/fetch/mediapipe.sh.
MEDIAPIPE_DIR = Path("data") / "models" / "mediapipe"
MEDIAPIPE_WEIGHTS = (
    "blaze_face_full_range.f16.safetensors",
    "face_mesh_v2.f16.safetensors",
    "selfie_multiclass_256.f32.safetensors",
)

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
    ("Depth Anything V2-Small", "the name of the depth model (NOTICE item 13), Apache-2.0 section 4"),
    ("Lihe Yang", "the authors of the Depth Anything V2-Small weights (Apache-2.0), whose notice the licence requires in every copy"),
    ("github.com/DepthAnything/Depth-Anything-V2", "the link to the depth weights' source (Apache-2.0 section 4)"),
    ("MediaPipe Authors", "the copyright holder of the MediaPipe face networks (Apache-2.0), whose notice the licence requires in every copy"),
    ("github.com/google-ai-edge/mediapipe", "the link to the MediaPipe networks' source (Apache-2.0 section 4)"),
    ("Andrea Volpato", "the author of the spektrafilm profiles (CC BY-SA 4.0), who must be named in every copy"),
    ("github.com/andreavolpato/spektrafilm", "the canonical source of the spektrafilm profiles, which the licence requires"),
    ("data/film/spektrafilm/LICENSE", "the pointer to the spektrafilm licence text"),
    ("International Commission on Illumination", "the publisher of the CIE tables (CC BY-SA 4.0)"),
    ("data/film/cie/LICENSE", "the pointer to the CIE tables' licence text"),
    ("and a table computed from it", "the credit for the upsampling table computed from the CIE observer (CC BY-SA 4.0 ShareAlike)"),
    ("data/film/upsampling/LICENSE", "the pointer to the upsampling table's licence text"),
)

#: The data folders under data/film/ that carry CC BY-SA 4.0 data (the profiles and the CIE tables are third-party, the upsampling
#: table is ours, computed from the CIE observer and so shared under the same licence), and the files each must have.
#: ``crates/ym-film/tests/oracle/`` holds the oracle's fixtures, numbers computed by spektrafilm from the CC BY-SA profiles (F-06),
#: so they are Adapted Material and carry the same three files.
FILM_DIRS = (
    Path("data") / "film" / "spektrafilm",
    Path("data") / "film" / "cie",
    Path("data") / "film" / "upsampling",
    Path("crates") / "ym-film" / "tests" / "oracle",
)
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


def model_pin(root: Path, folder: Path, weights_name: str) -> str | None:
    """The first field of the weights' line in ``<folder>/SHA256SUMS``, or None when there is no such line."""
    try:
        text = (root / folder / "SHA256SUMS").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == weights_name:
            return fields[0]
    return None


def sam2_pin(root: Path) -> str | None:
    """The first field of the weights' line in ``data/models/sam2/SHA256SUMS``, or None when there is no such line."""
    return model_pin(root, SAM2_DIR, SAM2_WEIGHTS)


def depth_pin(root: Path) -> str | None:
    """The first field of the weights' line in ``data/models/depth/SHA256SUMS``, or None when there is no such line."""
    return model_pin(root, DEPTH_DIR, DEPTH_WEIGHTS)


def warnings(root: Path) -> list[str]:
    """What is not an error yet: a weights' pin is still the placeholder written before the models workflow ran."""
    found = []
    for folder, name, pin in ((SAM2_DIR, SAM2_WEIGHTS, sam2_pin(root)), (DEPTH_DIR, DEPTH_WEIGHTS, depth_pin(root))):
        if pin is not None and not re.fullmatch(r"[0-9a-f]{64}", pin):
            found.append(
                f"{folder}/SHA256SUMS still holds '{pin}' for {name}: run the models workflow and copy its hashes "
                "(a build with --require-weights stops here)"
            )
    return found


def model_problems(root: Path, folder: Path, weights_name: str, source: str, script: str, what: str, require_weights: bool) -> list[str]:
    """One model's weights folder: what is committed, and with ``require_weights`` the fetched file as well.

    ``source`` is the link the model card must carry, ``script`` the fetch script named in the hint, ``what`` what the app would lack.
    """
    found: list[str] = []
    path = root / folder
    card = path / "MODEL_CARD.md"
    try:
        card_text = card.read_text(encoding="utf-8")
    except OSError as error:
        found.append(f"{card} cannot be read: {error}")
    else:
        for phrase, why in (("Apache-2.0", "the licence of the weights"), (source, "the source of the weights")):
            if phrase not in card_text:
                found.append(f'{folder}/MODEL_CARD.md does not contain "{phrase}": {why}')
    ignore = path / ".gitignore"
    try:
        ignored = ignore.read_text(encoding="utf-8")
    except OSError as error:
        found.append(f"{ignore} cannot be read: {error} (the weights must never be committed)")
    else:
        if "safetensors" not in ignored:
            found.append(f"{folder}/.gitignore does not ignore *.safetensors: the weights are too big for git")
    pin = model_pin(root, folder, weights_name)
    if pin is None:
        found.append(f"{folder}/SHA256SUMS has no line for {weights_name}")
    if not require_weights:
        return found
    weights = path / weights_name
    if pin is not None and not re.fullmatch(r"[0-9a-f]{64}", pin):
        found.append(f"{folder}/SHA256SUMS holds '{pin}' for {weights_name}, not a SHA-256: copy the hashes of the models workflow's SHA256SUMS asset")
    elif not weights.is_file():
        found.append(f"{weights} is missing: run {script} before the build, or the app would have no {what}")
    elif pin is not None and sha256_file(weights) != pin:
        found.append(f"{weights} does not have the SHA-256 in {folder}/SHA256SUMS: fetch it again with {script}")
    return found


def sam2_problems(root: Path, require_weights: bool) -> list[str]:
    """The SAM 2.1 small weights' folder: what is committed, and with ``require_weights`` the fetched file as well."""
    return model_problems(root, SAM2_DIR, SAM2_WEIGHTS, "github.com/facebookresearch/sam2", "tools/fetch/sam2.sh", "Object masks", require_weights)


def depth_problems(root: Path, require_weights: bool) -> list[str]:
    """The Depth Anything V2-Small weights' folder; the same rules as SAM's."""
    return model_problems(root, DEPTH_DIR, DEPTH_WEIGHTS, "github.com/DepthAnything/Depth-Anything-V2", "tools/fetch/depth.sh", "Depth masks", require_weights)


def mediapipe_problems(root: Path, require_files: bool) -> list[str]:
    """The MediaPipe face networks' folder: what is committed (card, ``.gitignore``, a SHA-256 line per file), and with ``require_files``
    the three fetched files as well. One folder, three files: each is checked like a single model's weights."""
    found: list[str] = []
    for index, name in enumerate(MEDIAPIPE_WEIGHTS):
        # The card and the .gitignore are the folder's: report them once.
        problems_of_file = model_problems(
            root, MEDIAPIPE_DIR, name, "storage.googleapis.com/mediapipe-models", "tools/fetch/mediapipe.sh", "face finder for the Skin node", require_files
        )
        found += problems_of_file if index == 0 else [p for p in problems_of_file if "MODEL_CARD" not in p and ".gitignore" not in p]
    return found


def film_problems(root: Path) -> list[str]:
    """The film data folders (``FILM_DIRS``): LICENSE, ATTRIBUTION.md, and a SHA256SUMS that matches every file."""
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


def problems(root: Path, require_weights: bool = False, depth_pending: bool = False, require_mediapipe: bool = False) -> list[str]:
    """Everything that is missing, one sentence each.

    With ``depth_pending`` the depth weights are not required even under ``require_weights`` (their pin is still the placeholder).
    The MediaPipe files are required only with ``require_mediapipe`` (the release exists).
    """
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
    return (
        found
        + sam2_problems(root, require_weights)
        + depth_problems(root, require_weights and not depth_pending)
        + mediapipe_problems(root, require_mediapipe)
        + film_problems(root)
    )


def main(argv: list[str]) -> int:
    require_weights = "--require-weights" in argv
    depth_pending = "--depth-pending" in argv
    require_mediapipe = "--require-mediapipe" in argv
    paths = [arg for arg in argv if not arg.startswith("--")]
    unknown = [arg for arg in argv if arg.startswith("--") and arg not in ("--require-weights", "--depth-pending", "--require-mediapipe")]
    if unknown or (depth_pending and not require_weights):
        print(f"error: unknown option {unknown[0] if unknown else '--depth-pending without --require-weights'}; usage: check_bundle.py [--require-weights [--depth-pending]] [--require-mediapipe] [REPO]")
        return 2
    root = Path(paths[0]).resolve() if paths else REPO
    found = problems(root, require_weights, depth_pending, require_mediapipe)
    for line in found:
        print(f"error: {line}")
    for line in warnings(root):
        if not require_weights:
            print(f"warning: {line}")
    if not found:
        weights = ""
        if require_weights:
            weights = "; the SAM 2.1 weights are in place and match SHA256SUMS"
            weights += "; the depth weights are NOT checked (--depth-pending: their pin is the placeholder)" if depth_pending else "; the depth weights are in place and match SHA256SUMS"
        if require_mediapipe:
            weights += "; the MediaPipe networks are in place and match SHA256SUMS"
        print(f"bundle check: NOTICE, LICENSE, THIRD-PARTY.md, data/models and data/film are bundled; NOTICE carries the BSD, MIT, Meta, Depth Anything, MediaPipe, spektrafilm and CIE notices; the film data folders match their sums{weights}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
