"""Fit the white scale of every registry body, write ``white_scale.json``, report both references.

    python3 -m training.white_scale.fit --write        # fit from the pinned files, write white_scale.json
    python3 -m training.white_scale.fit --check        # recompute, compare with the committed white_scale.json
    python3 -m training.white_scale.fit --report       # every file found, grouped by body (pinned or not)

``white_scale.json`` is what ``training.sensor_map.build_registry`` puts into ``data/cameras/registry.json``
(``white_scale`` and ``white_scale_source`` of each body). The rules (docs/03, "White scale of stand-in bodies"):

* a body with a profile of its own (``exact``) and a body that shares its profile's sensor (``sensor_family``)
  has the scale ``[1, 1, 1]``;
* a stand-in body (``default``) with pinned files gets reference (a): the scale that puts the Kelvin presets of its
  files on the locus of its profile (``reference.fit_presets``), one fit per file. Its files must agree: every file's
  scale within ``AGREE_LIMIT`` (5 %) of the median in red and in blue. When they do not, one obvious outlier may be
  left out if the body has three files or more and the rest agree; otherwise the body is not fitted. The scale is
  the median of the files kept (rounded to four decimals);
* the scale is accepted when, for every file kept, the mean angle between the Kelvin presets and the locus at their
  labels is at most ``MEAN_ANGLE_LIMIT_DEG`` and the Daylight preset reads inside ``DAYLIGHT_WINDOW_K``;
* a stand-in body without files whose published colour matrix equals that of a fitted body (largest
  per-coefficient difference after row normalisation at most 0.02, the limit of the sibling check) takes that
  body's scale and says so;
* any other stand-in body has ``[1, 1, 1]`` and says that nothing was measured.

Reference (b), the published colour matrix at D65, is computed for every body that has one and recorded next to the
presets' residuals, but does not set the scale (docs/03 says why). The presets are read from the files with our own
reader (``sr2.py``); the matrices are numbers in ``colour_matrices.json``.

**Which files.** A file counts when it is one of the pinned files, found by its content in the folders given
(``--photos``, default ``DEFAULT_FOLDERS``), whatever its name or folder: the owner's A1 photos
(``tools/fetch/owner-photos.sha256``), the local-eval files (git blob ids of ``tools/fetch/real_eval.py``) and the
Play Raw files of ``tools/fetch/white-scale-photos.json``. Each of those has a fetch script, so every shipped scale
can be reproduced. ``--write`` and ``--check`` refuse to run when a pinned file is missing; ``--report`` reads every
ARW of the folders, pinned or not, and says which are which.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from training.common import jsonfmt, paths
from training.sensor_map import registry_data as rd
from training.white_scale import reference as ref
from training.white_scale import sr2

RESULT_JSON = Path(__file__).resolve().parent / "white_scale.json"
FETCH_DIR = paths.REPO_ROOT / "tools" / "fetch"
OWNER_SHA256 = FETCH_DIR / "owner-photos.sha256"
PLAY_RAW_TABLE = FETCH_DIR / "white-scale-photos.json"
REAL_EVAL_PY = FETCH_DIR / "real_eval.py"

#: The sets a file can come from (``white_scale_source.files[].set``) and the script that fetches each.
OWNER_PHOTOS, REAL_EVAL, PLAY_RAW = "owner-photos", "real-eval", "play-raw"
SETS = {
    OWNER_PHOTOS: "tools/fetch/owner-photos.sh",
    REAL_EVAL: "tools/fetch/real-eval.sh",
    PLAY_RAW: "tools/fetch/white-scale-photos.sh",
}
#: Where the fetch scripts put the files (the fit finds them by content, so any folder with the files works).
DEFAULT_FOLDERS = (
    paths.TESTDATA / "readme-photos" / "src",
    paths.TESTDATA / "real" / "files",
    paths.TESTDATA / "white-scale" / "play-raw",
)
#: Model ID -> the stand-in body whose scale it takes when its published matrix equals that body's.
SIBLING_OF = {400: 384}
DECIMALS = 4
#: Daylight presets are labelled by the maker as a typical temperature (docs/03); the check window of the task.
DAYLIGHT_WINDOW_K = (5000.0, 5600.0)
MEAN_ANGLE_LIMIT_DEG = 2.0
#: Files of one body agree when each file's scale is within this fraction of the median in red and in blue: the unit-to-unit
#: spread of the presets that the files of bodies with curves of their own show (A7 III 4.7 % and 3.5 %, A6400 4.1 % and
#: 3.1 %, A7 IV 0.7 % and 3.7 %, A7C II 2.6 % and 3.6 %; docs/03). 5 % in red or blue is 7 to 9 mired in the Kelvin reading.
AGREE_LIMIT = 0.05
#: Files a body needs. One file cannot show whether its presets are consistent (docs/03 says what that costs).
MIN_FILES = 1
#: Files from which one outlier may be left out.
OUTLIER_MIN_FILES = 3


# ---------------------------------------------------------------------------------------------------
# Files: pins and discovery
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pin:
    """One file a scale may be fitted on, as the repo's fetch scripts pin it."""

    set: str
    name: str
    sha256: str = ""  # empty for the local-eval set, which pins git blob ids
    blob: str = ""
    model: str = ""  # the Model string it must carry (play-raw table), or empty


@dataclass(frozen=True)
class Found:
    """A pinned file found on disk."""

    pin: Pin
    path: Path
    sha256: str


def owner_pins() -> list[Pin]:
    """The A1 photos of ``tools/fetch/owner-photos.sha256`` (top-level ARW files only)."""
    out = []
    for line in OWNER_SHA256.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name and "/" not in name and name.lower().endswith(".arw"):
            out.append(Pin(OWNER_PHOTOS, name, sha256=digest, model="ILCE-1"))
    return out


def real_eval_pins() -> list[Pin]:
    """The local-eval files: git blob ids and link names from ``tools/fetch/real_eval.py`` (loaded, not copied)."""
    spec = importlib.util.spec_from_file_location("_ym_real_eval", REAL_EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [Pin(REAL_EVAL, f["name"], blob=f["blob"]) for source in module.SOURCES for f in source["files"]]


def play_raw_pins() -> list[Pin]:
    """The Play Raw files of ``tools/fetch/white-scale-photos.json``."""
    if not PLAY_RAW_TABLE.is_file():
        return []
    table = json.loads(PLAY_RAW_TABLE.read_text())
    return [Pin(PLAY_RAW, f["name"], sha256=f["sha256"], model=f["model"]) for f in table["files"]]


def all_pins() -> list[Pin]:
    return owner_pins() + real_eval_pins() + play_raw_pins()


def digests(path: Path) -> tuple[str, str]:
    """``(sha256, git blob id)`` of a file, in one pass."""
    size = path.stat().st_size
    sha, blob = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)
            blob.update(chunk)
    return sha.hexdigest(), blob.hexdigest()


def arw_files(folders) -> list[Path]:
    """The ARW files directly in the folders (links followed), each real file once."""
    seen: dict[Path, Path] = {}
    for folder in folders:
        folder = Path(folder)
        if folder.is_dir():
            for f in sorted(list(folder.glob("*.ARW")) + list(folder.glob("*.arw"))):
                seen.setdefault(f.resolve(), f)
    return list(seen.values())


@dataclass
class Scan:
    """The ARW files of some folders: which are pinned files, which are not."""

    found: list[Found] = field(default_factory=list)
    unpinned: list[Path] = field(default_factory=list)
    wrong: list[Path] = field(default_factory=list)  # carries a pinned name, not the pinned content

    def missing(self) -> list[Pin]:
        """Pinned files a fit needs and the folders do not have: every owner and Play Raw pin, and the local-eval
        files of a body the registry lists as a stand-in."""
        have = {(f.pin.set, f.pin.name) for f in self.found}
        stand_ins = {b.model for b in rd.BODIES if b.match == "default"}
        out = []
        for p in all_pins():
            needed = p.set != REAL_EVAL or p.name.removesuffix(".ARW").split("_")[0] in stand_ins
            if needed and (p.set, p.name) not in have:
                out.append(p)
        return out


def scan(folders) -> Scan:
    """Finds the pinned files in the folders by their content (SHA-256, or the git blob id for the local-eval set)."""
    pins = all_pins()
    by_sha = {p.sha256: p for p in pins if p.sha256}
    by_blob = {p.blob: p for p in pins if p.blob}
    names = {p.name for p in pins}
    out, taken = Scan(), set()
    for path in arw_files(folders):
        sha, blob = digests(path)
        pin = by_sha.get(sha) or by_blob.get(blob)
        if pin is None:
            (out.wrong if path.name in names else out.unpinned).append(path)
        elif (pin.set, pin.name) not in taken:
            taken.add((pin.set, pin.name))
            out.found.append(Found(pin, path, sha))
    return out


# ---------------------------------------------------------------------------------------------------
# Consistency of a body's files
# ---------------------------------------------------------------------------------------------------
def deviations(scales: np.ndarray) -> np.ndarray:
    """Each file's largest relative distance from the median scale, in red or blue (``scales`` is n x 3)."""
    m = np.median(scales, axis=0)
    return np.maximum(np.abs(scales[:, 0] / m[0] - 1.0), np.abs(scales[:, 2] / m[2] - 1.0))


@dataclass(frozen=True)
class Selection:
    """Which files of a body are kept, and why (``single``, ``agree``, ``outlier``; ``disagree`` or ``few``: none)."""

    kept: tuple[int, ...]
    excluded: tuple[int, ...]
    verdict: str


def choose(scales: np.ndarray) -> Selection:
    """The consistency rule of docs/03: files agree within ``AGREE_LIMIT``; failing that, one outlier may go when
    the body has ``OUTLIER_MIN_FILES`` or more and the rest agree; otherwise no file is kept."""
    n = len(scales)
    everyone = tuple(range(n))
    if n < MIN_FILES:
        return Selection((), everyone, "few")
    if n == 1:
        return Selection(everyone, (), "single")
    dev = deviations(scales)
    if (dev <= AGREE_LIMIT).all():
        return Selection(everyone, (), "agree")
    if n >= OUTLIER_MIN_FILES:
        worst = int(np.argmax(dev))
        rest = tuple(i for i in everyone if i != worst)
        if (deviations(scales[list(rest)]) <= AGREE_LIMIT).all():
            return Selection(rest, (worst,), "outlier")
    return Selection((), everyone, "disagree")


# ---------------------------------------------------------------------------------------------------
# One body
# ---------------------------------------------------------------------------------------------------
@dataclass
class FileFit:
    found: Found
    levels: dict
    scale: tuple[float, float, float]


@dataclass
class BodyResult:
    """The fit of one stand-in body: its files, the selection, the scale and how it does."""

    body: rd.Body
    profile: object
    files: list[FileFit]
    selection: Selection
    scale: tuple[float, float, float] | None = None
    fits: list[ref.Fit] = field(default_factory=list)  # one per file kept
    matrix_scale: tuple[float, float, float] | None = None
    matrix_fits: list[ref.Fit] = field(default_factory=list)
    problem: str = ""  # why the body is not fitted, empty when it is

    @property
    def kept(self) -> list[FileFit]:
        return [self.files[i] for i in self.selection.kept]

    @property
    def excluded(self) -> list[FileFit]:
        return [self.files[i] for i in self.selection.excluded]

    def spread_percent(self) -> tuple[float, float]:
        """The largest distance of a kept file's scale from the median, red and blue, in percent."""
        s = np.array([f.scale for f in self.kept])
        m = np.median(s, axis=0)
        return (
            round(100.0 * float(np.max(np.abs(s[:, 0] / m[0] - 1.0))), 2),
            round(100.0 * float(np.max(np.abs(s[:, 2] / m[2] - 1.0))), 2),
        )


def _round(s) -> tuple[float, float, float]:
    return (round(float(s[0]), DECIMALS), 1.0, round(float(s[2]), DECIMALS))


def within_limits(fit: ref.Fit) -> bool:
    return fit.mean_angle_after_deg <= MEAN_ANGLE_LIMIT_DEG and DAYLIGHT_WINDOW_K[0] <= fit.daylight_k_after <= DAYLIGHT_WINDOW_K[1]


def evaluate_body(body: rd.Body, found: list[Found], matrix=None) -> BodyResult:
    """Fit a stand-in body on the files found for it; ``matrix`` is its published D65 matrix (XYZ to camera), if any."""
    return evaluate_levels(body, [(f, sr2.read_levels(f.path)) for f in sorted(found, key=lambda x: x.pin.name)], matrix)


def evaluate_levels(body: rd.Body, items: list[tuple[Found, dict]], matrix=None) -> BodyResult:
    """The same on files already read: ``(file, white-balance levels)`` pairs."""
    profile = ref.load_profile(body.profile)
    files = []
    for f, levels in items:
        try:
            scale = ref.fit_presets(profile, levels)
        except ValueError as e:
            raise SystemExit(f"{f.pin.name or f.path.name}: {e}") from None
        if "daylight" not in levels:
            raise SystemExit(f"{f.pin.name or f.path.name}: no Daylight preset")
        files.append(FileFit(f, levels, scale))
    sel = choose(np.array([f.scale for f in files]))
    result = BodyResult(body, profile, files, sel)
    if sel.verdict == "few":
        result.problem = f"only {len(files)} file(s), at least {MIN_FILES} needed"
        return result
    if not sel.kept:
        dev = deviations(np.array([f.scale for f in files]))
        result.problem = (
            f"the presets of its {len(files)} files differ by up to {100 * dev.max():.1f} % in red or blue from their median "
            f"(limit {100 * AGREE_LIMIT:g} %, one outlier allowed from {OUTLIER_MIN_FILES} files)"
        )
        return result
    scale = _round(np.median(np.array([f.scale for f in result.kept]), axis=0))
    result.scale = scale
    result.fits = [ref.judge(profile, f.levels, scale) for f in result.kept]
    bad = [fit for fit in result.fits if not within_limits(fit)]
    if bad:
        worst = max(bad, key=lambda x: x.mean_angle_after_deg)
        result.problem = (
            f"with the scale {list(scale)} the mean preset angle is {worst.mean_angle_after_deg:.2f} degrees and the Daylight preset "
            f"reads {worst.daylight_k_after:.0f} K (limits: at most {MEAN_ANGLE_LIMIT_DEG:g} degrees, {DAYLIGHT_WINDOW_K[0]:.0f} to "
            f"{DAYLIGHT_WINDOW_K[1]:.0f} K)"
        )
        return result
    if matrix is not None:
        result.matrix_scale = tuple(float(x) for x in ref.matrix_scale(profile, matrix))
        result.matrix_fits = [ref.judge(profile, f.levels, result.matrix_scale) for f in result.kept]
    return result


def _mean(values) -> float:
    return float(np.mean(list(values)))


def _file_clause(r: BodyResult) -> str:
    n = len(r.kept)
    if n == 1:
        return "One file: it cannot show whether the body's presets are the same from unit to unit."
    sr, sb = r.spread_percent()
    if max(sr, sb) == 0.0:
        return f"{n} files with the same presets."
    return f"{n} files, each within {max(sr, sb):.2f} % of the median scale in red and blue (limit {100 * AGREE_LIMIT:g} %)."


def _excluded_clause(r: BodyResult) -> str:
    if not r.excluded:
        return ""
    med = np.median(np.array([f.scale for f in r.kept]), axis=0)
    parts = []
    for f in r.excluded:
        dev = 100 * max(abs(f.scale[0] / med[0] - 1.0), abs(f.scale[2] / med[2] - 1.0))
        parts.append(f"{f.found.pin.name} ({f.found.pin.set}; scale [{f.scale[0]:.4f}, {f.scale[2]:.4f}], {dev:.1f} % from the others)")
    return " Left out as an outlier: " + "; ".join(parts) + "."


def _matrix_clause(r: BodyResult) -> str:
    if r.matrix_scale is None:
        return " No published colour matrix of this body was available, so the matrix reference was not computed."
    angle = _mean(f.mean_angle_after_deg for f in r.matrix_fits)
    day = _mean(f.daylight_k_after for f in r.matrix_fits)
    meets = angle <= MEAN_ANGLE_LIMIT_DEG and DAYLIGHT_WINDOW_K[0] <= day <= DAYLIGHT_WINDOW_K[1]
    limits = (
        f"the limits of docs/03 (at most {MEAN_ANGLE_LIMIT_DEG:g} degrees, Daylight {DAYLIGHT_WINDOW_K[0]:.0f} to {DAYLIGHT_WINDOW_K[1]:.0f} K)"
    )
    return (
        " The scale the body's published colour matrix at D65 gives (matrix_scale) leaves a mean preset angle of "
        f"{angle:.2f} degrees and reads the Daylight preset at {day:.0f} K, "
        + (f"within {limits}; the presets still set the scale." if meets else f"outside {limits}, so the presets set the scale.")
    )


def source_entry(r: BodyResult) -> dict:
    """The registry's ``white_scale`` and ``white_scale_source`` for a fitted body."""
    residual = {
        "mean_preset_angle_before_deg": round(_mean(f.mean_angle_before_deg for f in r.fits), 2),
        "mean_preset_angle_after_deg": round(_mean(f.mean_angle_after_deg for f in r.fits), 2),
        "daylight_k_before": round(_mean(f.daylight_k_before for f in r.fits)),
        "daylight_k_after": round(_mean(f.daylight_k_after for f in r.fits)),
        "file_spread_percent": list(r.spread_percent()),
    }
    if r.matrix_scale is not None:
        residual.update(
            {
                "matrix_scale": list(_round(r.matrix_scale)),
                "matrix_mean_preset_angle_deg": round(_mean(f.mean_angle_after_deg for f in r.matrix_fits), 2),
                "matrix_daylight_k": round(_mean(f.daylight_k_after for f in r.matrix_fits)),
            }
        )
    return {
        "white_scale": list(r.scale),
        "source": {
            "kind": "presets",
            "files": [{"name": f.found.pin.name, "sha256": f.found.sha256, "set": f.found.pin.set} for f in r.kept],
            "residual": residual,
            "note": (
                "Fitted so the five Kelvin presets of the body's own files (2500, 3200, 4500, 6000, 8500 K) read at their "
                f"labels on the {rd.PROFILE_BY_ID[r.body.profile].display_name} locus (least squares in log chroma), one fit per "
                "file and the median of the files kept. "
                + _file_clause(r)
                + _excluded_clause(r)
                + _matrix_clause(r)
            ),
        },
    }


def none_entry(note: str | None = None) -> dict:
    return {
        "white_scale": [1.0, 1.0, 1.0],
        "source": {
            "kind": "none",
            "files": [],
            "note": note
            or "No file with white-balance presets of this body or of a body with the same colour matrix was available; its whites are not corrected.",
        },
    }


# ---------------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------------
def fit_all(folders=DEFAULT_FOLDERS) -> dict:
    """The contents of ``white_scale.json``."""
    matrices = ref.load_matrices()
    s = scan(folders)
    missing = s.missing()
    if missing:
        raise SystemExit(
            "cannot run: pinned files are missing from the folders given:\n"
            + "\n".join(f"  {p.name} ({p.set}: run {SETS[p.set]})" for p in missing)
        )
    by_model: dict[str, list[Found]] = {}
    stand_ins = {b.model for b in rd.BODIES if b.match == "default"}
    for f in s.found:
        model = sr2.read_model(f.path)
        if f.pin.model and f.pin.model != model:
            raise SystemExit(f"{f.pin.name}: the pin says {f.pin.model}, the file says {model}")
        if f.pin.set != REAL_EVAL and model not in stand_ins:
            raise SystemExit(f"{f.pin.name}: pinned file of {model}, which is not a stand-in body of the registry")
        if model in stand_ins:  # local-eval files of measured bodies (A7 III, A7 IV) and of bodies without an entry are not used
            by_model.setdefault(model, []).append(f)
    results = {
        b.model_id: evaluate_body(b, by_model[b.model], matrices.get(b.model_id))
        for b in rd.BODIES
        if b.match == "default" and b.model in by_model
    }
    fitted = {mid: source_entry(r) for mid, r in results.items() if r.scale is not None and not r.problem}
    bodies: dict[str, dict] = {}
    for b in rd.BODIES:
        if b.match == "exact":
            entry = {
                "white_scale": [1.0, 1.0, 1.0],
                "source": {"kind": "own_curves", "files": [], "note": "The profile was measured on this body (curves of the registry's profile); nothing to correct."},
            }
        elif b.match == "sensor_family":
            entry = {
                "white_scale": [1.0, 1.0, 1.0],
                "source": {"kind": "same_sensor", "files": [], "note": "Same sensor as the measured body of its profile (the colour-matrix check of docs/03); no scale of its own."},
            }
        elif b.model_id in fitted:
            entry = fitted[b.model_id]
        elif b.model_id in SIBLING_OF and SIBLING_OF[b.model_id] in fitted:
            base = rd.BODY_BY_ID[SIBLING_OF[b.model_id]]
            diff = ref.matrix_difference(matrices[b.model_id], matrices[base.model_id])
            if diff <= rd.SIBLING_LIMIT:
                entry = {
                    "white_scale": list(fitted[base.model_id]["white_scale"]),
                    "source": {
                        "kind": "sibling",
                        "files": [],
                        "note": (
                            f"Takes the scale of the {base.body}: the published colour matrices of the two bodies agree "
                            f"(largest per-coefficient difference {diff:.4f}, limit {rd.SIBLING_LIMIT}). No file of this body was "
                            "available, so the presets were not read."
                        ),
                    },
                }
            else:
                entry = none_entry()
        elif b.model_id in results and results[b.model_id].problem:
            entry = none_entry(
                "Files of this body were read but the presets were not accepted: " + results[b.model_id].problem + "; its whites are not corrected."
            )
        else:
            entry = none_entry()
        entry["model"] = b.model
        bodies[str(b.model_id)] = entry
    return {
        "schema_version": 2,
        "description": (
            "White scale per registry body, generated by python3 -m training.white_scale.fit --write from the white-balance "
            "presets stored in pinned files (tools/fetch/owner-photos.sh, real-eval.sh, white-scale-photos.sh) and the published "
            "colour matrices in colour_matrices.json. build_registry copies it into data/cameras/registry.json."
        ),
        "bodies": bodies,
    }


# ---------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------
def published_matrices_by_model() -> dict[str, np.ndarray]:
    """The published D65 matrices of ``data/cameras/color-matrices.json`` by Model string (every body that has one)."""
    doc = json.loads((paths.CAMERAS_DIR / "color-matrices.json").read_text())
    return {b["model"]: np.array(b["xyz_d65_to_camera"], dtype=float) for b in doc["bodies"]}


def report_rows(folders) -> list[tuple[str, str, bool, int, int, BodyResult]]:
    """``(Model string, profile id, listed in the registry, files, pinned files, fit)`` over every ARW of the folders,
    pinned or not. A body that is not a stand-in is shown on its own profile and on the default one."""
    matrices = published_matrices_by_model()
    reg = json.loads(paths.REGISTRY_JSON.read_text())
    by_model = {b["model"]: b for b in reg["bodies"]}
    pinned = {f.path.resolve(): f for f in scan(folders).found}
    groups: dict[str, list[Found]] = {}
    for path in arw_files(folders):
        try:
            model = sr2.read_model(path)
            levels = sr2.read_levels(path)
        except (sr2.Sr2Error, OSError, ValueError) as e:
            print(f"skipping {path.name}: {e}", file=sys.stderr)
            continue
        if "daylight" not in levels or sum(k in levels for k in sr2.KELVIN_PRESETS) < 3:
            print(f"skipping {path.name}: no white-balance presets", file=sys.stderr)
            continue
        groups.setdefault(model, []).append(pinned.get(path.resolve()) or Found(Pin("", path.name), path, ""))
    rows = []
    for model, files in sorted(groups.items()):
        listed = by_model.get(model)
        own = listed["profile"] if listed else rd.DEFAULT_PROFILE
        for pid in dict.fromkeys([own, rd.DEFAULT_PROFILE]):
            stub = rd.Body(listed["model_id"] if listed else 0, model, listed["body"] if listed else f"Sony {model}", pid, "default")
            rows.append((model, pid, listed is not None, len(files), sum(1 for f in files if f.pin.set), evaluate_body(stub, files, matrices.get(model))))
    return rows


def report(folders) -> None:
    print(
        f"{'body':12s} {'profile':9s} {'n':>3s} {'pin':>3s} | {'scale r, b':>13s} {'spread r/b %':>12s} {'angle':>10s} {'Daylight K':>12s} {'limits':6s} {'files':8s} | "
        f"{'matrix scale':>13s} {'angle':>5s} {'Day K':>5s}"
    )
    for model, pid, listed, n, n_pinned, r in report_rows(folders):
        scales = np.array([f.scale for f in r.files])
        scale = r.scale or _round(np.median(scales, axis=0))
        shown = r.fits or [ref.judge(r.profile, f.levels, scale) for f in r.files]
        med = np.median(scales, axis=0)
        spread = f"{100 * np.max(np.abs(scales[:, 0] / med[0] - 1)):4.1f}/{100 * np.max(np.abs(scales[:, 2] / med[2] - 1)):4.1f}"
        ok = all(within_limits(f) for f in shown)
        line = (
            f"{model:12s} {pid[5:]:9s} {n:3d} {n_pinned:3d} | [{scale[0]:.3f},{scale[2]:.3f}] {spread:>12s} "
            f"{_mean(f.mean_angle_before_deg for f in shown):4.2f}->{_mean(f.mean_angle_after_deg for f in shown):4.2f} "
            f"{_mean(f.daylight_k_before for f in shown):5.0f}->{_mean(f.daylight_k_after for f in shown):5.0f} "
            f"{'ok' if ok else 'MISS':6s} {r.selection.verdict:8s} | "
        )
        if r.matrix_scale is not None:
            line += f"[{r.matrix_scale[0]:.3f},{r.matrix_scale[2]:.3f}] {_mean(f.mean_angle_after_deg for f in r.matrix_fits):5.2f} {_mean(f.daylight_k_after for f in r.matrix_fits):5.0f}"
        elif not listed:
            line += "(body not in the registry)"
        print(line)
        if r.selection.verdict == "outlier":
            for f in r.excluded:
                print(f"      left out: {f.found.pin.name or f.found.path.name} (scale [{f.scale[0]:.4f}, {f.scale[2]:.4f}])")
    print("\nthe published matrix against the curves of the measured bodies (as if each were a stand-in on the A7R IV profile):")
    matrices = ref.load_matrices()
    s_prof = ref.load_profile(rd.DEFAULT_PROFILE)
    print(f"{'body':12s} {'curves s':>16s} {'matrix s':>16s} {'angle':>6s}")
    for b in rd.BODIES:
        if b.match != "exact" or b.profile == rd.DEFAULT_PROFILE:
            continue
        p = ref.load_profile(b.profile)
        truth = ref.locus_white(s_prof, ref.D65_K) / ref.locus_white(p, ref.D65_K)
        m = ref.matrix_scale(s_prof, matrices[b.model_id])
        w = ref.locus_white(p, ref.D65_K)
        print(f"{b.model:12s} [{truth[0]:.3f}, {truth[2]:.3f}] [{m[0]:.3f}, {m[2]:.3f}] {ref.angle_deg(ref.scaled(w, truth), ref.scaled(w, m)):6.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--photos",
        type=Path,
        action="append",
        help="folder with ARW files (repeatable; the files are found by content). Default: the folders the fetch scripts fill",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    folders = args.photos or list(DEFAULT_FOLDERS)
    if not arw_files(folders):
        print(f"cannot run: no ARW files in {', '.join(str(f) for f in folders)} (tools/fetch/owner-photos.sh, real-eval.sh, white-scale-photos.sh)", file=sys.stderr)
        return 2
    if args.report:
        report(folders)
        return 0
    result = fit_all(folders)
    text = jsonfmt.dumps(result)
    if args.write:
        RESULT_JSON.write_text(text)
        print("wrote", RESULT_JSON.relative_to(paths.REPO_ROOT))
        return 0
    committed = json.loads(RESULT_JSON.read_text())
    if committed != json.loads(text):
        print("white_scale.json differs from a fresh fit (run --write and review)", file=sys.stderr)
        return 1
    print("white_scale.json matches a fresh fit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
