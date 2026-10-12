"""Repository and test-data locations shared by the training scripts.

Inputs that are fetched, never committed, live under ``testdata/`` (git-ignored) and are
filled by ``tools/fetch/spectral.sh``. Set ``YM_TESTDATA`` to use another root.
Generated, non-shipped outputs (validation CSVs, fixtures) go under ``testdata/out`` and
``testdata/spectral``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTDATA = Path(os.environ.get("YM_TESTDATA", REPO_ROOT / "testdata"))
SOURCES = TESTDATA / "sources"

# Fetched sources (tools/fetch/spectral.sh).
RAWTOACES_DATA = SOURCES / "rawtoaces-data"
ACES_DEV = SOURCES / "aces-dev"
CAMERA_CURVES_DIR = RAWTOACES_DATA / "data" / "camera"
TRAINING_REFLECTANCES_JSON = RAWTOACES_DATA / "data" / "training" / "training_spectral.json"
ARRI_MATRIX_DIR = (
    ACES_DEV / "transforms" / "ctl" / "idt" / "vendorSupplied" / "arri" / "alexa" / "AlexaParameters-2013-Nov-13"
)

# Shipped data (committed).
DATA = REPO_ROOT / "data"
ALEXA_DIR = DATA / "alexa"
ALEXA_CURVES_JSON = ALEXA_DIR / "alexa_alev3_estimate_380_780_5.json"
CAMERAS_DIR = DATA / "cameras"
SONY_CURVES_DIR = CAMERAS_DIR / "sony"
PROFILES_DIR = CAMERAS_DIR / "profiles"
REGISTRY_JSON = CAMERAS_DIR / "registry.json"
ALEXA_TARGET_JSON = ALEXA_DIR / "target.json"
PROFILE_SCHEMA = REPO_ROOT / "crates" / "ym-profiles" / "schema" / "profile.schema.json"
REGISTRY_SCHEMA = REPO_ROOT / "crates" / "ym-profiles" / "schema" / "registry.schema.json"
TARGET_SCHEMA = REPO_ROOT / "crates" / "ym-profiles" / "schema" / "target.schema.json"
SPECTRAL_REFERENCE_JSON = REPO_ROOT / "crates" / "ym-color" / "tests" / "data" / "spectral_reference.json"

# Generated, git-ignored outputs.
OUT = TESTDATA / "out"
FIXTURES = TESTDATA / "spectral"


def require_sources() -> None:
    """Exit with a clear message when ``tools/fetch/spectral.sh`` has not been run."""
    for p in (CAMERA_CURVES_DIR, TRAINING_REFLECTANCES_JSON, ARRI_MATRIX_DIR):
        if not p.exists():
            raise SystemExit(f"missing {p}\nrun tools/fetch/spectral.sh from the repository root first")


def git_head_sha() -> str:
    """Commit the working tree is based on, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
