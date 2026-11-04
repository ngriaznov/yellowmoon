#!/usr/bin/env python3
"""The two policy checks of docs/24 section 6.3, the author's requests that we treat as binding.

    python3 tools/licences/check_film_policy.py [REPO]

* **No training on the film data.** No text file under ``training/`` mentions ``data/film/``. The author asks that commercial AI
  models are not trained on the profiles, on a table baked from them or on their output; none of our models is.
* **No product branding.** No file under ``src/`` (the interface), ``src-tauri/src/`` or ``crates/ym-cli/src/`` mentions
  "spektrafilm", in any case, except the credit lines and About in the interface (``INTERFACE_ALLOWED``). The Film node, its
  settings, commands and menus are called Film; the name appears only as factual credit.

Prints one line per problem and exits 1 when there is one.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The folder that must not read the film data, and the text that gives a read away.
TRAINING_DIR = "training"
FILM_DATA = "data/film/"

#: Folders whose files must not carry the name, and the files of ``src/`` that may (the credit lines, About and their tests).
BRANDING_DIRS = ("src", "src-tauri/src", "crates/ym-cli/src")
INTERFACE_ALLOWED = ("src/components/About*", "src/**/About*", "src/**/filmCredit*", "src/**/FilmCredit*")
NAME = "spektrafilm"

SKIPPED_DIRS = {"__pycache__", "node_modules", ".git", "target"}


def text_files(folder: Path):
    """Yield (path, text) for the readable text files under ``folder``; binary files and caches are skipped."""
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or SKIPPED_DIRS & set(path.relative_to(folder).parts):
            continue
        try:
            yield path, path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue


def allowed(rel: str) -> bool:
    """True for the interface files that may carry the credit line."""
    return rel.startswith("src/") and any(fnmatch.fnmatch(rel, pattern) for pattern in INTERFACE_ALLOWED)


def problems(root: Path) -> list[str]:
    """Everything that breaks the two rules, one sentence each."""
    found: list[str] = []
    for path, text in text_files(root / TRAINING_DIR):
        if FILM_DATA in text:
            found.append(f"{path.relative_to(root).as_posix()} mentions {FILM_DATA}: no model of ours is trained on the film profiles, a table baked from them or a Film render")
    for folder in BRANDING_DIRS:
        for path, text in text_files(root / folder):
            rel = path.relative_to(root).as_posix()
            if NAME in text.lower() and not allowed(rel):
                found.append(f'{rel} contains "{NAME}": the name is factual credit only (About and the credit lines), never in a label, setting or command')
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[0]).resolve() if argv else REPO
    found = problems(root)
    for line in found:
        print(f"error: {line}")
    if not found:
        print("film policy: nothing under training/ mentions data/film/; the name is only in the credit lines and About")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
