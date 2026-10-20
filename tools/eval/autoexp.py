#!/usr/bin/env python3
"""Auto-exposure evaluation: what the M1-08 gate measures, and how far our exposure is from the camera's.

    python3 -I tools/eval/autoexp.py testdata/real/files --label real
    python3 -I tools/eval/autoexp.py testdata/synthetic/arw --label synthetic --one-per-scene
    python3 -I tools/eval/autoexp.py testdata/synthetic/arw --label synthetic --one-per-scene --gate   # the M1-08 gate, with an exit code

Since 2026-10-14 `EV_auto` is the camera's own exposure (`ym_color::exposure::CAMERA_GAIN_EV`, +1.11 EV, the median camera
gain of the 12 local files) lowered by the highlight guard when the picture would clip: no metering of our own moves it
(`docs/02` Auto exposure). G3 below is therefore `1.11 - camera gain` per file: on the 12 local files it measures how far each
camera is from the median of the cameras (in-sample: the constant was fitted on these files, so a pass says little about
other cameras), and on the synthetic tier, whose JPEG engine is made with the same gain, it is zero by construction
(`training/synth/preview.py`). The synthetic checks that still mean something are the highlight guard (it lowers `EV_auto` only
on scenes that would clip, by the documented amount: the replica and `ym` agree) and the exposure compensation (a constant
exposure leaves the compensation, which the raw holds, in the picture in full); see `summary.synthetic_checks`.

For every ARW in the folder it runs the real `ym` (default: the dev build `target/debug/ym`, `cargo build -p ym-cli`;
a release build is not needed), takes the camera's embedded JPEG and computes, per photo:

* **G1** the gate as `ym_color::exposure` implements it: the median of our scene-linear luminance at `EV_auto` against
  the median linear luminance of the JPEG (sRGB or Adobe RGB decoded);
* **G2** the same with our *display-referred* render (sRGB 16-bit TIFF through ACES 2): what a user sees;
* **G3** the DRT-neutral comparison: the camera's own scene-referred gain, read from its own raw-to-JPEG tone curve
  (quantile matching of the JPEG against our scene-linear image at the JPEG's spatial scale, no registration), against our `EV_auto`;
* the placement of mid grey by our display transform (ACES 2, from the render and from OpenColorIO) and what G2 would
  read for other placements ("ARRI-style" is a parameter here: no ARRI file is read);
* the camera's own exposure decision from the EXIF triple (EV100, program mode, flash), a nominal ISO 2720 / 12232
  figure, and which statistic of the raw scene the camera's meter seems to hold constant;
* sweeps: a constant exposure offset, the metering target, other meters, and keeping part of the camera's placement.

`ym render --format dng --exposure 0` is the scene-linear picture (C1 to C11, linear AWG4, before the display transform;
the EV scale is exact because C7 is the identity below the ALEXA's clip with the highlight range at 0, `docs/02`).
`ym render --format tiff` is the display render. Both are made with the Neutral look, the highlights as recorded and no
white lift (`--look builtin:neutral --rolloff 0,0.5 --white-offset -3`: the highlight range at 0, the low-light toe at
its default, the lift off): the study measures the exposure, and the default look (Signature) and the automatic white
lift (both since pipeline 0.3.0) and the default highlight range (1, since pipeline 0.4.0) change the contrast and the
highlights on purpose, and the DNG carries all three, which would move the quantile match of G3. Every call passes
`--apple-auto off`: the study measures Yellowmoon's own Auto, the same on every machine, and Apple's answers never tune
anything of ours (`docs/19` section 7). The photos are never modified: renders go to `--out`, big files are deleted after reduction unless
`--keep-renders`, and a small cache (`<out>/cache/`) lets `--reuse` redo the arithmetic without rendering again.

`--gate` turns the run into the M1-08 gate (docs/13, docs/15; Mykyta's decision of 2026-10-13): the metric is G3, a file passes
when |G3| <= `--tolerance-ev` (0.5), and the gate is met when at least `--min-fraction` (0.8) of the files pass. A file that
failed to analyse, or whose camera gain could not be read, counts as a miss and is listed. The outliers are printed and written
to `report.md` and `results.json` (`summary.gate`) for the M1-H5 review, whether or not the gate is met.

Needs numpy, Pillow and tifffile (`pip install numpy pillow tifffile`); OpenColorIO (`PyOpenColorIO`) and `exiftool` are
optional. Nothing is written to the repository: the default `--out` is under `testdata/eval/` (git-ignored).
Exit code 0 when every file was analysed (and, with `--gate`, the gate is met), 1 when some failed, 2 for a usage error,
3 when `--gate` is given and fewer than `--min-fraction` of the files are within the tolerance (this wins over 1).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging

import metrics as m
import numpy as np
import study

logging.getLogger("tifffile").setLevel(logging.ERROR)  # a synthetic ARW has pages tifffile cannot read; the extras are optional
REPO = Path(__file__).resolve().parents[2]
DEFAULT_YM = REPO / "target" / "debug" / "ym"
#: The fitted DNG profile that `ym` embeds (its tone curve and the offset that `BaselineExposure` carries).
FITTED_PROFILE = REPO / "data" / "dng" / "profile.json"
DNG_BLOCK = 8  # the 1/8 preview's binning
#: Version of the reduction a cache file holds (bump it when `reduce_photo` stores something else): an older cache
#: is not reused, so `--reuse` cannot mix numbers from two versions of the arithmetic.
CACHE_SCHEMA = 3
# The raw level at which the synthetic JPEG (training/synth/preview.py `tone`: gain 2.133, soft shoulder, sRGB, mild S-curve)
# reads 0.18 in linear light, as a gain: 1.0654 EV (the nominal gain log2(2.133) is 1.093; the S-curve darkens grey). The gain was
# calibrated so that this estimator reads the tier's median at `ym_color::exposure::CAMERA_GAIN_EV` (+1.11 EV, the camera gain
# `EV_auto` is, docs/02, 2026-10-14): it reads this engine +0.039 EV above its own gain (std 0.03): median 1.104 EV over the 37 scenes.
SYNTH_GAIN_EV = 1.0654
#: `ym_color::display::DISPLAY_PLACEMENT_EV`, read from the source so that the two cannot drift apart: the ACES 2 path brightens
#: the scene by this many stops before the transform and the DNG's `BaselineExposure` carries it.
PLACEMENT_SOURCE = REPO / "crates" / "ym-color" / "src" / "display" / "transform.rs"
VARIANT_ORDER = ("uncompressed14", "packed12", "lossless-l", "craw", "lossless-ms")
AE_PROGRAMS = {2, 3, 4, 5, 6, 7, 8}  # EXIF ExposureProgram: the camera metered (not manual, not undefined)
#: The M1-08 gate (docs/13 gate table, docs/15, decision 2026-10-13): `EV_auto` within this many EV of the camera engine's gain (G3) ...
GATE_TOLERANCE_EV = 0.5
#: ... for at least this fraction of the files.
GATE_MIN_FRACTION = 0.8
#: Exit code of `--gate` when the gate is not met (1 stays "some file failed to analyse", 2 "usage").
EXIT_GATE_NOT_MET = 3


#: The three `ym` calls made for each photo (what a report built from the cache shows instead of recorded commands).
COMMAND_TEMPLATES = [
    "ym info <file> --json --quiet --apple-auto off",
    "ym render <file> --json --quiet --overwrite --out <work>/renders --format dng --name {name}_eval_dng_ev0 --look builtin:neutral --rolloff 0,0.5 --white-offset -3 --apple-auto off --exposure 0",
    "ym render <file> --json --quiet --overwrite --out <work>/renders --format tiff --name {name}_eval_tiff --bits 16 --compression none --colour-space srgb --look builtin:neutral --rolloff 0,0.5 --white-offset -3 --apple-auto off",
]


class StudyError(Exception):
    """A file that cannot be analysed (the message says why and goes into the report)."""


# --------------------------------------------------------------------------------------------------------------
# Finding files
# --------------------------------------------------------------------------------------------------------------


def find_files(inputs: list[Path], one_per_scene: bool, filter_text: str | None) -> list[Path]:
    """The ARW files named or found (one level) in `inputs`, sorted by name; `one_per_scene` keeps one file per
    synthetic scene (the generator writes one mosaic in several raw variants), preferring an exact variant."""
    files: list[Path] = []
    for p in inputs:
        p = p.absolute()  # a name that starts with "-" can then not be read as an option by `ym` or `exiftool`
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() == ".arw" and not f.name.startswith("."))
        elif p.is_file():
            files.append(p)
        else:
            raise StudyError(f"{p} does not exist")
    if filter_text:
        files = [f for f in files if filter_text.lower() in f.name.lower()]
    if one_per_scene:
        best: dict[str, tuple[int, Path]] = {}
        for f in files:
            scene, variant = scene_and_variant(f)
            rank = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)
            if scene not in best or rank < best[scene][0]:
                best[scene] = (rank, f)
        files = sorted(v[1] for v in best.values())
    return files


def scene_and_variant(path: Path) -> tuple[str, str]:
    """`(scene id, variant)` of a synthetic file name `<scene>__<variant>.ARW` (the stem and "" for any other name)."""
    stem = path.stem
    if "__" in stem and (path.parent.parent / "scenes" / f"{stem.rsplit('__', 1)[0]}.json").exists():
        scene, variant = stem.rsplit("__", 1)
        return scene, variant
    return stem, ""


def synthetic_sidecar(path: Path) -> dict[str, Any] | None:
    """The ground-truth sidecar of a synthetic ARW, when there is one."""
    scene, variant = scene_and_variant(path)
    side = path.parent.parent / "scenes" / f"{scene}.json"
    if variant and side.exists():
        return json.loads(side.read_text())
    return None


# --------------------------------------------------------------------------------------------------------------
# Running ym
# --------------------------------------------------------------------------------------------------------------


class Ym:
    """The `ym` binary, run with an empty home folder (no imported looks, default look and display rendering) so that
    nothing of the user's setup reaches the numbers; the commands are recorded for the report."""

    def __init__(self, binary: Path, home: Path, display: str):
        self.binary = binary
        self.home = home
        self.display = display
        self.commands: list[str] = []
        self.env = dict(os.environ, YELLOWMOON_HOME=str(home))
        self._stamp = self._file_stamp()

    def _file_stamp(self) -> tuple[int, int, int] | None:
        """Identity of the binary on disk (inode, size, modification time), `None` when it cannot be read."""
        try:
            st = self.binary.stat()
        except OSError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    def run(self, args: list[str], timeout_s: float = 1800, record: bool = True) -> subprocess.CompletedProcess[str]:
        # `ym info` and the renders of one photo must come from one binary: `target/debug/ym` is rewritten by every
        # `cargo build`, and a study takes minutes. A rebuild in between once gave a render whose automatic exposure
        # (+0.7229 EV) was not the one `ym info` had reported (+0.6784 EV) for the same file, which `check_render_exposure`
        # caught. A binary that changed is now named as the cause.
        if self._file_stamp() != self._stamp:
            raise StudyError(f"{self.binary} was rebuilt or replaced while the study was running; run the study again (or point --ym at a copy)")
        cmd = [str(self.binary), *args]
        if record:
            self.commands.append(" ".join(cmd))
        try:
            return subprocess.run(cmd, capture_output=True, text=True, env=self.env, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired as err:
            raise StudyError(f"`ym {args[0]}` took longer than {timeout_s:.0f} s") from err

    def version(self) -> str:
        return self.run(["--version"], record=False).stdout.strip()

    def info(self, path: Path) -> dict[str, Any]:
        proc = self.run(["info", str(path), "--json", "--quiet", "--apple-auto", "off"])
        doc = _last_json(proc.stdout)
        if doc is None or doc.get("status") != "ok":
            raise StudyError(f"ym info failed: {_why(doc, proc)}")
        return doc

    def render(self, path: Path, out: Path, fmt: str, exposure: str | None) -> tuple[Path, dict[str, Any]]:
        args = ["render", str(path), "--json", "--quiet", "--overwrite", "--out", str(out), "--format", fmt, "--name", "{name}_eval_" + (fmt if exposure is None else f"{fmt}_ev{exposure}")]
        if fmt == "tiff":
            args += ["--bits", "16", "--compression", "none", "--colour-space", "srgb"]
        # The study is about the exposure: the Neutral look, the highlights as recorded and no white lift (the default
        # look, the default highlight range and the automatic lift change the picture on purpose, and the DNG carries
        # all three).
        args += ["--look", "builtin:neutral", "--rolloff", "0,0.5", "--white-offset", "-3"]
        # Yellowmoon's own Auto: Apple's answers would differ between Macs and never tune anything of ours (docs/19).
        args += ["--apple-auto", "off"]
        if exposure is not None:
            args += ["--exposure", exposure]
        if self.display != "aces2":
            args += ["--display", self.display]
        proc = self.run(args)
        doc = _last_json(proc.stdout)
        if doc is None or doc.get("status") != "ok" or not doc.get("output"):
            raise StudyError(f"ym render ({fmt}) failed: {_why(doc, proc)}")
        return Path(doc["output"]), doc


def check_render_exposure(doc: dict[str, Any], expected_ev: float, what: str) -> None:
    """Refuses a render whose settings report another exposure than the study assumes (the DNG at 0, the TIFF at the
    `EV_auto` of `ym info`): every number built on the picture would silently describe another exposure. A render
    document without the `settings.exposure` field (an older `ym`) is not checked."""
    exposure = (doc.get("settings") or {}).get("exposure") or {}
    ev = exposure.get("ev")
    if isinstance(ev, (int, float)) and not abs(float(ev) - expected_ev) <= 1e-6:
        raise StudyError(f"the {what} render used exposure {float(ev):+.6f} EV, the study needs {expected_ev:+.6f} EV")


def _last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # `ym info --json` prints one indented object
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _why(doc: dict[str, Any] | None, proc: subprocess.CompletedProcess[str]) -> str:
    if doc and doc.get("error"):
        return str(doc["error"].get("message", doc["error"]))
    return (proc.stderr.strip().splitlines() or ["no output"])[-1]


# --------------------------------------------------------------------------------------------------------------
# Reading the pictures
# --------------------------------------------------------------------------------------------------------------


def _tifffile():
    try:
        import tifffile
    except ImportError as err:  # pragma: no cover - environment
        raise StudyError("tifffile is needed to read the renders: pip install tifffile") from err
    return tifffile


def read_linear_dng(path: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    """`(stored uint16 (H, W, 3), BaselineExposure in EV, tags)` of the DNG `ym render --format dng` writes.

    Without a fitted profile the scene-linear value is `stored / 65535 * 2^BaselineExposure` (`ym-export` `dng.rs`,
    "Pixel scaling": the baseline exposure undoes the scaling by `x_white`). With one (`tags["fitted_profile"]`, which is
    what `ym` writes since the DNG got its Adobe profile) `BaselineExposure` is `log2(x_white)` plus the profile's fit
    offset plus the display placement, so the caller must take `fitted_profile_offset_ev(tags["tone_curve"])` and
    `display_placement_ev()` off it first.
    """
    tifffile = _tifffile()
    with tifffile.TiffFile(str(path)) as tf:
        top = tf.pages[0]
        tag = top.tags.get("BaselineExposure")
        if tag is None:
            raise StudyError("the DNG has no BaselineExposure")
        num, den = tag.value
        look_dims = top.tags.get("ProfileLookTableDims")
        dims = tuple(int(v) for v in look_dims.value) if look_dims is not None else None
        curve_tag = top.tags.get("ProfileToneCurve")
        tone_curve = tuple(float(v) for v in curve_tag.value) if curve_tag is not None else ()
        raw = None
        for sub in top.pages:
            if int(sub.photometric) == 34892:  # LinearRaw
                raw = sub
                break
        if raw is None:
            raise StudyError("the DNG has no LinearRaw image")
        data = raw.asarray()
    if data.dtype != np.uint16 or data.ndim != 3 or data.shape[2] != 3:
        raise StudyError(f"unexpected DNG pixels {data.dtype} {data.shape}")
    return data, num / den, {"look_table_dims": dims, "fitted_profile": dims not in (None, (2, 2, 2)), "tone_curve": tone_curve}


def fitted_profile_offset_ev(tone_curve: Sequence[float], profile_path: Path = FITTED_PROFILE) -> float:
    """The `fit_offset_ev` of the fitted DNG profile that a `ym` DNG carries, vouched for by its tone curve.

    The offset is not written into the DNG, only into `data/dng/profile.json`, which `ym` embeds when it is built. The
    DNG's `ProfileToneCurve` (flattened `x, y` pairs, float32) must be that file's curve, otherwise the binary was built
    from another profile and its `BaselineExposure` cannot be undone: the study refuses rather than guess.
    """
    try:
        profile = json.loads(profile_path.read_text())
        curve = np.asarray(profile["tone_curve"], dtype=np.float64).ravel()
        offset = float(profile["fit_offset_ev"])
    except (OSError, ValueError, KeyError) as err:
        raise StudyError(f"the DNG carries a fitted profile and {profile_path} cannot be read to undo its offset: {err}") from err
    dng_curve = np.asarray(tone_curve, dtype=np.float64)
    if dng_curve.shape != curve.shape or not np.allclose(dng_curve, curve, rtol=1e-5, atol=1e-9) or not math.isfinite(offset):
        raise StudyError(f"the DNG's profile is not {profile_path} (build `ym` from this checkout): the fit offset in its BaselineExposure cannot be undone")
    return offset


def display_placement_ev(source: Path = PLACEMENT_SOURCE) -> float:
    """`DISPLAY_PLACEMENT_EV` of `ym-color` (`pub const DISPLAY_PLACEMENT_EV: f64 = 0.85;`), read from its source.

    A DNG with the fitted profile carries it in `BaselineExposure` (`log2(x_white) + fit_offset_ev + placement`), so the study
    has to take it off to undo the `x_white` scaling.
    """
    import re

    try:
        text = source.read_text()
    except OSError as err:
        raise StudyError(f"cannot read {source} to find the display placement: {err}") from err
    found = re.search(r"pub const DISPLAY_PLACEMENT_EV: f64 = (-?[0-9.]+);", text)
    if found is None:
        raise StudyError(f"{source} has no `pub const DISPLAY_PLACEMENT_EV: f64 = ...;`: the DNG's BaselineExposure cannot be undone")
    return float(found.group(1))


def read_tiff16(path: Path) -> np.ndarray:
    """The 16-bit RGB picture of an uncompressed TIFF export."""
    data = _tifffile().imread(str(path))
    if data.dtype != np.uint16 or data.ndim != 3 or data.shape[2] < 3:
        raise StudyError(f"unexpected TIFF pixels {data.dtype} {data.shape}")
    return np.ascontiguousarray(data[:, :, :3])


def decode_preview(path: Path, info: dict[str, Any]) -> np.ndarray:
    """The camera's embedded JPEG (the one `ym info` reports: offset and length in the file) as 8-bit RGB."""
    from PIL import Image

    pv = info.get("preview")
    if not pv:
        raise StudyError("the file has no embedded preview")
    with path.open("rb") as fh:
        fh.seek(pv["offset"])
        blob = fh.read(pv["length"])
    img = Image.open(io.BytesIO(blob))
    img.load()
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def read_exif_extras(path: Path) -> dict[str, Any]:
    """Program, metering, flash, exposure bias and the JPEG colour space flag from the ARW's own Exif block, plus (when
    `exiftool` is installed) the maker-note settings that change the camera's rendering. All best effort."""
    out: dict[str, Any] = {}
    try:
        tifffile = _tifffile()
        with tifffile.TiffFile(str(path)) as tf:
            tag = tf.pages[0].tags.get("ExifTag")
            exif = tag.value if tag is not None else {}
            out["exposure_program"] = exif.get("ExposureProgram")
            out["metering_mode"] = exif.get("MeteringMode")
            out["flash"] = exif.get("Flash")
            out["color_space_tag"] = exif.get("ColorSpace")
            bias = exif.get("BrightnessValue")
            if isinstance(bias, tuple) and bias[1]:
                out["brightness_value"] = bias[0] / bias[1]
            interop = exif.get("InteroperabilityTag")
            if isinstance(interop, int) and interop > 0:
                out["interop_index"] = _read_interop_index(tf, interop)
    except Exception as err:  # noqa: BLE001 - descriptive extras only
        out["exif_error"] = str(err)
    exiftool = shutil.which("exiftool")
    if exiftool:
        proc = subprocess.run([exiftool, "-j", "-DynamicRangeOptimizer", "-CreativeStyle", "-HDR", "-SceneMode", str(path)], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            try:
                row = json.loads(proc.stdout)[0]
                out["dro"] = row.get("DynamicRangeOptimizer")
                out["creative_style"] = row.get("CreativeStyle")
                out["hdr"] = row.get("HDR")
            except (json.JSONDecodeError, IndexError):
                pass
    return out


def _read_interop_index(tf: Any, offset: int) -> str | None:
    """`InteroperabilityIndex` (`R98` sRGB, `R03` Adobe RGB) from the Interoperability IFD at `offset`."""
    fh = tf.filehandle
    pos = fh.tell()
    try:
        fh.seek(offset)
        end = "<" if tf.byteorder == "<" else ">"
        count = int.from_bytes(fh.read(2), "little" if end == "<" else "big")
        for _ in range(min(count, 16)):
            entry = fh.read(12)
            tag = int.from_bytes(entry[:2], "little" if end == "<" else "big")
            if tag == 1:
                return entry[8:11].decode("ascii", "replace").strip("\x00")
    finally:
        fh.seek(pos)
    return None


# --------------------------------------------------------------------------------------------------------------
# Reduction: the pictures to the small arrays the study needs
# --------------------------------------------------------------------------------------------------------------


def _scene_luminance(raw16: np.ndarray, baseline_ev: float) -> np.ndarray:
    """AWG4 luminance of the DNG's scene-linear pixels at EV 0 as float32 (a chunked pass)."""
    coeff = (m.awg4_y_row() * (2.0**baseline_ev / 65535.0)).astype(np.float32)
    y = np.empty(raw16.shape[:2], dtype=np.float32)
    for r0 in range(0, raw16.shape[0], 256):
        y[r0 : r0 + 256] = raw16[r0 : r0 + 256].astype(np.float32) @ coeff
    return y


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel()))
    rb = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def scale_match_size(src_shape: tuple[int, ...], ref_shape: tuple[int, ...]) -> tuple[int, int]:
    """`(width, height)` to area-resample a picture of `src_shape` (`(H, W, ...)`) to so that it has the spatial scale of
    a picture of `ref_shape`: the reference's size, turned to the source's orientation when one is landscape and the
    other portrait, and never larger than the source (a bigger reference would only be interpolated)."""
    sh, sw = int(src_shape[0]), int(src_shape[1])
    rh, rw = int(ref_shape[0]), int(ref_shape[1])
    if (sw >= sh) != (rw >= rh):
        rh, rw = rw, rh
    return max(1, min(rw, sw)), max(1, min(rh, sh))


def registration_check(y0: np.ndarray, j_y: np.ndarray, size: int = 64) -> tuple[float, int]:
    """How well the JPEG and our scene picture show the same thing: the best Spearman rank correlation of their
    (log) luminances at 64 x 64 over the eight turns and flips of the JPEG, and which turn it was (0 is as stored).
    The study itself does not need registration (quantile matching); this flags a mismatching preview."""
    a = np.log(np.maximum(m.resize_area(y0, size, size), 1e-4))
    best, which = -2.0, 0
    base = np.log(np.maximum(m.resize_area(j_y, size, size), 1e-4))
    for k in range(8):
        t = np.rot90(base, k % 4)
        if k >= 4:
            t = t[:, ::-1]
        c = _rank_correlation(a, t)
        if c > best:
            best, which = c, k
    return best, which


def reduce_photo(ym: Ym, path: Path, info: dict[str, Any], exif: dict[str, Any], work: Path, keep_renders: bool, sidecar: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Renders the photo twice and reduces everything the arithmetic needs; returns `(reduction, extras)`."""
    extras: dict[str, Any] = {"ym_version": ym.version()}
    rgb8 = decode_preview(path, info)
    colour_space = "adobergb" if exif.get("interop_index") == "R03" else "srgb"
    j_y = m.jpeg_luminance(rgb8, colour_space)
    extras["jpeg_size"] = [int(rgb8.shape[1]), int(rgb8.shape[0])]
    extras["jpeg_colour_space"] = colour_space
    extras["jpeg_clipped_fraction"] = float((rgb8.max(axis=2) >= 250).mean())
    del rgb8
    j_q = np.quantile(j_y, study.QUANTILE_GRID)
    j_median = float(np.median(j_y))

    out_dir = work / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dng_path, dng_doc = ym.render(path, out_dir, "dng", "0")
    check_render_exposure(dng_doc, 0.0, "scene-linear DNG")
    raw16, baseline_ev, dng_tags = read_linear_dng(dng_path)
    extras["dng_baseline_ev"] = baseline_ev
    extras["dng_fitted_profile"] = dng_tags["fitted_profile"]
    # The stored values are x / x_white; BaselineExposure is log2(x_white), plus the fit offset and the display placement
    # of a fitted profile.
    fit_offset_ev = fitted_profile_offset_ev(dng_tags["tone_curve"]) if dng_tags["fitted_profile"] else 0.0
    placement_ev = display_placement_ev() if dng_tags["fitted_profile"] else 0.0
    extras["dng_fit_offset_ev"] = fit_offset_ev
    extras["dng_placement_ev"] = placement_ev
    x_white_ev = baseline_ev - fit_offset_ev - placement_ev
    y0 = _scene_luminance(raw16, x_white_ev)
    extras["dng_size"] = [int(raw16.shape[1]), int(raw16.shape[0])]
    extras["dng_clipped_fraction"] = float((raw16.max(axis=2) == 65535).mean())
    del raw16
    y0_ds = m.box_mean(y0, DNG_BLOCK)
    # The camera curve is read by quantile matching against the JPEG, so our luminance is put at the JPEG's scale too.
    y0_q = np.quantile(m.resize_area(y0, *scale_match_size(y0.shape, j_y.shape)), study.QUANTILE_GRID)
    extras["scene_median_full_ev0"] = float(np.median(y0))
    truth_grey = None
    if sidecar is not None:
        truth_grey = _grey_patch_y0(sidecar, y0)
    corr, turn = registration_check(y0, j_y)
    extras["registration_corr"] = corr
    extras["registration_turn"] = turn
    del y0
    if not keep_renders:
        dng_path.unlink(missing_ok=True)

    tiff_path, doc = ym.render(path, out_dir, "tiff", None)
    ev_auto = (info.get("auto") or {}).get("exposure_ev")
    if isinstance(ev_auto, (int, float)):
        check_render_exposure(doc, float(ev_auto), "display")
    rgb16 = read_tiff16(tiff_path)
    disp = m.display_luminance_u16(rgb16)
    del rgb16
    disp_ds = m.box_mean(disp, DNG_BLOCK)
    disp_median = float(np.median(disp))
    extras["render_size"] = [int(disp.shape[1]), int(disp.shape[0])]
    extras["render_seconds"] = time.time() - t0
    extras["render_notes"] = doc.get("notes", [])
    del disp
    if not keep_renders:
        tiff_path.unlink(missing_ok=True)

    red = {"y0_ds": y0_ds, "y0_q": y0_q, "disp_ds": disp_ds, "disp_median": disp_median, "j_q": j_q, "j_median": j_median}
    if truth_grey is not None:
        red["truth_grey_y0"] = truth_grey
    return red, extras


def _grey_patch_y0(sidecar: dict[str, Any], y0: np.ndarray) -> float | None:
    """The mean of our scene-linear luminance (EV 0) over the interior of the 18 % grey patch of a synthetic scene."""
    for probe in sidecar.get("truth", {}).get("probes", []):
        if probe.get("name") == "neutral-0.18":
            x, y, w, h = probe.get("interior") or probe["rect"]
            patch = y0[y : y + h, x : x + w]
            return float(patch.mean()) if patch.size else None
    return None


def save_cache(path: Path, red: dict[str, Any], doc: dict[str, Any]) -> None:
    arrays = {k: np.asarray(v) for k, v in red.items()}
    arrays["_doc"] = np.frombuffer(json.dumps(doc).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load_cache(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("rb") as fh, np.load(fh) as z:
        doc = json.loads(bytes(z["_doc"]).decode())
        red = {k: (z[k] if z[k].ndim else float(z[k])) for k in z.files if k != "_doc"}
    return red, doc


# --------------------------------------------------------------------------------------------------------------
# One photo
# --------------------------------------------------------------------------------------------------------------


def analyse(ym: Ym | None, path: Path, work: Path, reuse: bool, keep_renders: bool, display: str = "aces2") -> dict[str, Any]:
    """Everything the study says about one photo (a dict that goes into the JSON; keys starting with `_` do not).

    `display` is the `ym --display` of the run: a cache made with another display rendering, another schema or another
    file (size or modification time), and one that cannot be read, is not reused.
    """
    cache = work / "cache" / f"{path.stem}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    key = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "schema": CACHE_SCHEMA, "display": display}
    if reuse and cache.exists():
        try:
            red, doc = load_cache(cache)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            red, doc = {}, {}
        if doc.get("key") == key:
            info, exif, extras = doc["info"], doc["exif"], doc["extras"]
            return _entry(path, info, exif, extras, red)
    if ym is None:
        raise StudyError("no cache for this file and no ym to render with")
    info = ym.info(path)
    exif = read_exif_extras(path)
    red, extras = reduce_photo(ym, path, info, exif, work, keep_renders, synthetic_sidecar(path))
    save_cache(cache, red, {"key": key, "info": info, "exif": exif, "extras": extras})
    return _entry(path, info, exif, extras, red)


def _entry(path: Path, info: dict[str, Any], exif: dict[str, Any], extras: dict[str, Any], red: dict[str, Any]) -> dict[str, Any]:
    auto = info.get("auto")
    if not auto:
        raise StudyError("ym gave no automatic estimates for this file")
    ex = info["exposure"]
    ev_auto = float(auto["exposure_ev"])
    ec = float(ex.get("exposure_comp_ev") or 0.0)
    e: dict[str, Any] = {"file": path.name, "path": str(path)}
    e["body"] = info["camera"]["body"]
    e["profile_match"] = info["camera"]["match"]
    e["iso"] = ex.get("iso")
    e["shutter_s"] = ex.get("shutter_s")
    e["f_number"] = ex.get("aperture")
    e["ev100"] = _ev100(ex)
    e["exposure_program"] = exif.get("exposure_program")
    e["flash_fired"] = bool((exif.get("flash") or 0) & 1)
    e["camera_metered"] = (exif.get("exposure_program") in AE_PROGRAMS) and not e["flash_fired"]
    e["dro"] = exif.get("dro")
    e["jpeg_colour_space"] = extras.get("jpeg_colour_space")
    e.update(study.file_metrics(red, ev_auto, ec))
    study.attach_curves(e, red)
    e["extras"] = extras

    # The replica of the algorithm against ym's own value (the 1/8 preview of the raw mosaic against the DNG's 1/8 blocks).
    rep = m.meter(red["y0_ds"], ec)
    e["ev_replica"] = rep.ev_auto
    e["ev_replica_minus_ym"] = rep.ev_auto - ev_auto
    e["guard_reduction_replica"] = rep.guard_reduction_ev
    e["key_statistic"] = rep.statistic
    # What the previous rule (metering to 0.18, half the compensation) would have chosen: reported for the what-if studies only.
    e["ev_previous_rule"] = m.meter(red["y0_ds"], ec, None, m.PREVIOUS_RULE).ev_auto

    # The camera's nominal exposure (ISO 2720 meter, ISO 12232 saturation speed): a reference line only.
    nominal = math.log2(m.MID_GREY / m.nominal_grey_fraction())
    e["nominal_camera_gain_ev"] = nominal
    e["camera_gain_minus_nominal_ev"] = e["camera_gain_ev"] - nominal
    e["ev_auto_minus_nominal_ev"] = ev_auto - nominal

    # Synthetic scenes know the truth: the exposure that puts the 18 % grey patch at 0.18, and the camera gain the
    # synthetic JPEG was made with.
    if "truth_grey_y0" in red:
        truth = math.log2(m.MID_GREY / float(red["truth_grey_y0"]))
        e["truth_ev"] = truth
        e["ev_auto_minus_truth_ev"] = ev_auto - truth
        e["synthetic_jpeg_gain_ev"] = SYNTH_GAIN_EV
        e["camera_gain_minus_synthetic_ev"] = e["camera_gain_ev"] - SYNTH_GAIN_EV
    return e


def _ev100(ex: dict[str, Any]) -> float | None:
    t, n, iso = ex.get("shutter_s"), ex.get("aperture"), ex.get("iso")
    if not (t and n and iso):
        return None
    return math.log2(n * n / t) - math.log2(iso / 100.0)


# --------------------------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------------------------


def _clean(o: Any) -> Any:
    """JSON-ready: numpy scalars to floats, NaN to null, private keys dropped."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summarise(entries: list[dict[str, Any]], tolerance_ev: float) -> dict[str, Any]:
    """Tallies and sweeps over the analysed photos."""
    s: dict[str, Any] = {"files": len(entries), "tolerance_ev": tolerance_ev}
    s["metrics"] = study.metric_table(entries, tolerance_ev)
    s["best_offset_ev"] = {k: study.best_offset(entries, k) for k in ("g1", "g2", "g3")}
    s["offset_sweep"] = study.offset_sweep(entries, [x / 4 for x in range(-4, 9)], tolerance_ev)
    s["placement_sweep"] = study.placement_sweep(entries, tolerance_ev=tolerance_ev)
    aces = m.aces2_neutral_display_y([m.MID_GREY])
    s["aces2_grey_display"] = None if aces is None else float(aces[0])
    placement = display_placement_ev()
    s["display_placement_ev"] = placement
    placed = m.aces2_neutral_display_y([m.MID_GREY * 2.0**placement])
    s["aces2_grey_display_placed"] = None if placed is None else float(placed[0])
    s["display_of_grey_median"] = float(np.median([e["display_of_grey"] for e in entries]))
    s["tolerance_sweep"] = [{"tolerance_ev": t, **{k: dict(zip(("within", "evaluated"), m.pass_rate([e[k] for e in entries], t))) for k in ("g1_gate_ev", "g2_display_ev", "g3_scene_ev")}} for t in (0.25, 0.5, 0.75, 1.0)]
    s["target_sweep"] = study.target_sweep(entries, (0.10, 0.14, 0.18, 0.22, 0.26, 0.32), tolerance_ev)
    gains = [e["camera_gain_ev"] for e in entries if math.isfinite(e["camera_gain_ev"])]
    anchor = float(np.median(gains)) if gains else 0.0
    s["camera_gain_anchor_ev"] = anchor
    s["camera_gain_spread"] = m.summary(gains)
    s["shrink_sweep"] = study.shrink_sweep(entries, anchor, (1.0, 0.75, 0.5, 0.25, 0.0), tolerance_ev)
    nominal = math.log2(m.MID_GREY / m.nominal_grey_fraction())
    s["shrink_sweep_nominal"] = study.shrink_sweep(entries, nominal, (1.0, 0.75, 0.5, 0.25, 0.0), tolerance_ev)
    s["statistic_sweep"] = study.statistic_sweep(entries, tolerance_ev)
    metered = [e for e in entries if e["camera_metered"]]
    s["camera_metered_files"] = [e["file"] for e in metered]
    s["meter_emulation"] = study.meter_emulation(metered) if len(metered) > 1 else []
    s["key_correlation"] = key_correlation(entries)
    s["contrast_fit"] = study.fit_contrast(entries)
    s["contrast_sweep"] = study.contrast_sweep(entries, (1.0, 1.2, 1.3, 1.4, 1.5, 1.6), tolerance_ev)
    s["replica_minus_ym"] = m.summary([e["ev_replica_minus_ym"] for e in entries])
    s["guard_check"] = guard_check(entries)
    s["camera_gain_ev"] = m.CAMERA_GAIN_EV
    s["ev_auto"] = m.summary([e["ev_auto"] for e in entries])
    truth = [e["ev_auto_minus_truth_ev"] for e in entries if "ev_auto_minus_truth_ev" in e]
    if truth:
        s["ev_auto_minus_truth"] = study.tally(truth, tolerance_ev)
        s["camera_gain_minus_synthetic"] = m.summary([e["camera_gain_minus_synthetic_ev"] for e in entries if "camera_gain_minus_synthetic_ev" in e])
    return s


def guard_check(entries: list[dict[str, Any]], tolerance_ev: float = 0.03) -> dict[str, Any]:
    """The highlight guard, the one part of `EV_auto` that still looks at the scene (`docs/02` step 5).

    With the camera's gain as the exposure the only thing that can move `EV_auto` is the guard, and it may only lower it, by at
    most one stop. This checks `ym` against the replica of the guard on the same scenes: `ym` lowered the exposure exactly on the
    files the replica says would clip (more than 0.5 % of the pixels above 8.0 after exposure), by the same amount to
    `tolerance_ev`, never raised it and never by more than the documented stop.
    """
    gain = m.CAMERA_GAIN_EV
    lowered_ym = [e["file"] for e in entries if gain - e["ev_auto"] > 1e-9]
    lowered_replica = [e["file"] for e in entries if e["guard_reduction_replica"] > 1e-9]
    diffs = [abs(e["ev_auto"] - e["ev_replica"]) for e in entries]
    reductions = [gain - e["ev_auto"] for e in entries]
    return {
        "files": len(entries),
        "camera_gain_ev": gain,
        "lowered_by_ym": lowered_ym,
        "lowered_by_replica": lowered_replica,
        "max_abs_replica_minus_ym_ev": max(diffs) if diffs else 0.0,
        "largest_reduction_ev": max(reductions) if reductions else 0.0,
        "never_raised": all(r >= -1e-9 for r in reductions),
        "at_most_one_stop": all(r <= 1.0 + 1e-9 for r in reductions),
        "agrees_with_replica": bool(diffs) and max(diffs) <= tolerance_ev and set(lowered_ym) == set(lowered_replica),
    }


def evaluate_gate(entries: list[dict[str, Any]], failures: Sequence[tuple[str, str]] = (), tolerance_ev: float = GATE_TOLERANCE_EV, min_fraction: float = GATE_MIN_FRACTION) -> dict[str, Any]:
    """The M1-08 gate on G3 (`g3_scene_ev`, our `EV_auto` minus the camera engine's gain, in EV).

    A file passes when |G3| <= `tolerance_ev` (the bound itself passes). The gate is met when at least `min_fraction` of the
    files pass, and it needs at least one file. The denominator is every file that was tried: a file that failed to analyse
    (`failures`) or whose G3 is not a finite number counts as a miss and is listed under `unusable`, so a run cannot reach 80 %
    by losing the files it does badly on. `outliers` are the files outside the tolerance, worst first, with what the M1-H5
    review needs to tell a deliberately dark or bright scene from a wrong exposure (the camera's own key, its DRO setting).
    """
    total = len(entries) + len(failures)
    needed = math.ceil(min_fraction * total - 1e-9)  # the smallest pass count whose share is at least `min_fraction`
    within = 0
    outliers: list[dict[str, Any]] = []
    unusable: list[dict[str, Any]] = []
    for e in entries:
        g3 = e.get("g3_scene_ev")
        if g3 is None or not math.isfinite(g3):
            unusable.append({"file": e["file"], "why": "no camera gain could be read from its JPEG"})
        elif abs(g3) <= tolerance_ev:
            within += 1
        else:
            outliers.append(
                {
                    "file": e["file"],
                    "body": e.get("body"),
                    "iso": e.get("iso"),
                    "dro": e.get("dro"),
                    "jpeg_median": e.get("jpeg_median"),
                    "ev_auto": e["ev_auto"],
                    "camera_gain_ev": e.get("camera_gain_ev"),
                    "g3_ev": float(g3),
                    "ours": "brighter" if g3 > 0 else "darker",
                }
            )
    unusable += [{"file": name, "why": why} for name, why in failures]
    outliers.sort(key=lambda o: -abs(o["g3_ev"]))
    return {
        "metric": "g3_scene_ev",
        "tolerance_ev": tolerance_ev,
        "min_fraction": min_fraction,
        "files": total,
        "within": within,
        "needed": needed,
        "fraction": within / total if total else 0.0,
        "passed": total > 0 and within >= needed,
        "outliers": outliers,
        "unusable": unusable,
    }


def gate_lines(gate: dict[str, Any]) -> list[str]:
    """The gate as plain text lines: the statement, the verdict, then the outliers for the M1-H5 review."""
    tol, frac = gate["tolerance_ev"], gate["min_fraction"]
    verdict = "MET" if gate["passed"] else "NOT MET"
    L = [
        f"M1-08 gate (G3: EV_auto minus the camera engine's gain): within +-{tol:g} EV for at least {frac * 100:g} % of the files",
        f"  {gate['within']} of {gate['files']} files within ({gate['fraction'] * 100:.1f} %; {gate['needed']} needed): {verdict}",
    ]
    if gate["outliers"]:
        L.append("  outliers for the M1-H5 review, worst first (G3, EV_auto, camera gain, JPEG median = the key the camera rendered, DRO):")
        for o in gate["outliers"]:
            L.append(f"    {o['file']}: G3 {_f(o['g3_ev'], 2, True)} (we are {o['ours']}), EV_auto {_f(o['ev_auto'], 2, True)}, camera gain {_f(o['camera_gain_ev'], 2, True)}, JPEG median {_f(o['jpeg_median'], 3)}, DRO {o['dro'] or '-'}, {o['body']}, ISO {o['iso']}")
    else:
        L.append("  no outliers")
    for u in gate["unusable"]:
        L.append(f"  counted as a miss: {u['file']}: {u['why']}")
    return L


def gate_markdown(gate: dict[str, Any]) -> list[str]:
    """The `## Gate` section of `report.md`."""
    tol, frac = gate["tolerance_ev"], gate["min_fraction"]
    L = ["## Gate (M1-08, metric G3)", ""]
    L += [(f"The gate (`docs/13`, `docs/15`; decision of 2026-10-13): our `EV_auto` is within +-{tol:g} EV of the camera engine's own gain (G3) for at least "
           f"{frac * 100:g} % of the files. A file that failed to analyse, or whose camera gain could not be read, counts as a miss."), ""]
    L += [f"**{'Met' if gate['passed'] else 'Not met'}: {gate['within']} of {gate['files']} files within ({gate['fraction'] * 100:.1f} %, {gate['needed']} needed).**", ""]
    if gate["outliers"]:
        L += ["Outliers for the M1-H5 review, worst first. The JPEG cannot tell a deliberately dark or bright scene from a wrong exposure; the key the camera rendered (JPEG median) and its DRO setting are given to help:", "",
              "| file | body | ISO | DRO | JPEG median | EV_auto | camera gain | G3 | we are |", "|---|---|---|---|---|---|---|---|---|"]
        for o in gate["outliers"]:
            L.append(f"| {o['file']} | {o['body']} | {o['iso']} | {o['dro'] or '-'} | {_f(o['jpeg_median'], 3)} | {_f(o['ev_auto'], 2, True)} | {_f(o['camera_gain_ev'], 2, True)} | {_f(o['g3_ev'], 2, True)} | {o['ours']} |")
        L.append("")
    else:
        L += ["No outliers.", ""]
    if gate["unusable"]:
        L += ["Counted as misses:", ""] + [f"- {u['file']}: {u['why']}" for u in gate["unusable"]] + [""]
    return L


def key_correlation(entries: list[dict[str, Any]]) -> dict[str, float]:
    """Pearson correlation over photos of each error with the scene's key as the camera rendered it (`log2` of the JPEG
    median): a metric that tracks the key is measuring the scene, not the exposure algorithm. Needs three photos."""
    out: dict[str, float] = {}
    if len(entries) < 3:
        return out
    with np.errstate(divide="ignore"):
        key = np.log2([e["jpeg_median"] for e in entries])
    for k in ("g1_gate_ev", "g2_display_ev", "g3_scene_ev", "ev_auto"):
        v = np.array([e[k] for e in entries], dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(key)
        out[k] = float(np.corrcoef(key[ok], v[ok])[0, 1]) if ok.sum() >= 3 and v[ok].std() > 0 else math.nan
        out[k + "_slope"] = float(np.polyfit(key[ok], v[ok], 1)[0]) if ok.sum() >= 3 and v[ok].std() > 0 else math.nan
    # The camera's own gain against the raw scene's key as shot (log2 of our scene-linear median at EV 0): a gain that
    # does not follow the key is a camera that applies one fixed gain to whatever its AE exposed.
    with np.errstate(divide="ignore"):
        raw_key = np.log2([e["scene_median_ev0"] for e in entries])
    gain = np.array([e["camera_gain_ev"] for e in entries], dtype=np.float64)
    ok = np.isfinite(gain) & np.isfinite(raw_key)
    out["camera_gain_vs_raw_key"] = float(np.corrcoef(raw_key[ok], gain[ok])[0, 1]) if ok.sum() >= 3 and gain[ok].std() > 0 and raw_key[ok].std() > 0 else math.nan
    return out


def _f(v: Any, nd: int = 2, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def markdown(meta: dict[str, Any], entries: list[dict[str, Any]], failures: list[tuple[str, str]], s: dict[str, Any]) -> str:
    """The report as Markdown tables."""
    L: list[str] = []
    tol = s["tolerance_ev"]
    L += [f"# Auto-exposure study: {meta['label']}", ""]
    L += [f"`{meta['ym_version']}`; {len(entries)} photos analysed, {len(failures)} failed; tolerance +-{tol} EV.", ""]
    L += ["Render commands (one `ym` process per photo, empty home folder):", ""]
    L += ["```", *meta["command_examples"], "```", ""]
    if s.get("gate"):
        L += gate_markdown(s["gate"])
    L += ["## Per photo", ""]
    L += ["| file | body | prog | DRO | ISO | t | N | EV100 | EV_auto | replica | scene med | JPEG med | G1 gate | G2 display | cam gain | G3 scene | D(0.18) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        t = e["shutter_s"]
        L.append(
            f"| {e['file']} | {e['body']} | {e['exposure_program'] if e['exposure_program'] is not None else '-'}{'F' if e['flash_fired'] else ''} | {e['dro'] or '-'} | {e['iso']} | "
            f"{(f'1/{round(1 / t)}') if t and t < 0.5 else _f(t, 1)} | {_f(e['f_number'], 1)} | {_f(e['ev100'], 1)} | {_f(e['ev_auto'], 2, True)} | "
            f"{_f(e['ev_replica_minus_ym'], 2, True)} | {_f(e['scene_median_at_auto'], 3)} | {_f(e['jpeg_median'], 3)} | {_f(e['g1_gate_ev'], 2, True)} | "
            f"{_f(e['g2_display_ev'], 2, True)} | {_f(e['camera_gain_ev'], 2, True)} | {_f(e['g3_scene_ev'], 2, True)} | {_f(e['display_of_grey'], 3)} |"
        )
    L += ["", ("Columns: `replica` is the Python replica of the algorithm minus `ym`'s `EV_auto`; `scene med` our scene-linear median at `EV_auto`; "
          "`G1` = log2(scene med / JPEG med); `G2` = log2(display median / JPEG med); `cam gain` the camera engine's gain from its tone curve at the "
          "0.18 anchor; `G3` = `EV_auto` - cam gain; `D(0.18)` our display luminance of scene 0.18 in this render. Prog: EXIF ExposureProgram (F = flash fired). DRO: the camera's Dynamic Range Optimizer setting (exiftool), a local tone mapping that the camera-curve estimate reads as part of the camera's rendering."), ""]
    L += ["## The three comparisons", ""]
    L += [f"| metric | within +-{tol:.1f} EV | median | mean | std | min | max |", "|---|---|---|---|---|---|---|"]
    names = {"g1_gate_ev": "G1 gate (scene-linear median vs JPEG median)", "g2_display_ev": "G2 display-referred median vs JPEG median", "g3_scene_ev": "G3 EV_auto vs camera gain from its own tone curve"}
    for k, label in names.items():
        t = s["metrics"][k]
        L.append(f"| {label} | {int(t['within'])} of {int(t['evaluated'])} | {_f(t['median'], 2, True)} | {_f(t['mean'], 2, True)} | {_f(t['std'], 2)} | {_f(t['min'], 2, True)} | {_f(t['max'], 2, True)} |")
    L += ["", "Pass counts at other tolerances: " + "; ".join(f"+-{r['tolerance_ev']:g} EV: G1 {r['g1_gate_ev']['within']}/{r['g1_gate_ev']['evaluated']}, G2 {r['g2_display_ev']['within']}/{r['g2_display_ev']['evaluated']}, G3 {r['g3_scene_ev']['within']}/{r['g3_scene_ev']['evaluated']}" for r in s["tolerance_sweep"]) + "."]
    bo = s["best_offset_ev"]
    L += ["", f"A constant exposure offset that centres each metric (median error to zero): G1 {_f(bo['g1'], 2, True)} EV, G2 {_f(bo['g2'], 2, True)} EV, G3 {_f(bo['g3'], 2, True)} EV.", ""]
    kc = s.get("key_correlation") or {}
    if kc:
        L += ["Photos by the key of the camera's own rendering (JPEG median, darkest first):", "", "| file | JPEG median | scene contrast (EV) | scene range (EV) | EV_auto | G1 | G2 | G3 |", "|---|---|---|---|---|---|---|---|"]
        for e in sorted(entries, key=lambda e: e["jpeg_median"]):
            L.append(f"| {e['file']} | {_f(e['jpeg_median'], 3)} | {_f(e['scene_contrast_ev'], 2)} | {_f(e['scene_range_ev'], 1)} | {_f(e['ev_auto'], 2, True)} | {_f(e['g1_gate_ev'], 2, True)} | {_f(e['g2_display_ev'], 2, True)} | {_f(e['g3_scene_ev'], 2, True)} |")
        names = {"g1_gate_ev": "G1", "g2_display_ev": "G2", "g3_scene_ev": "G3", "ev_auto": "EV_auto"}
        L += ["", "Correlation with log2(JPEG median) over photos (slope in EV per EV of key): " + ", ".join(f"{n} {_f(kc[k], 2, True)} ({_f(kc[k + '_slope'], 2, True)})" for k, n in names.items() if k in kc) + ".",
              f"The camera gain against the raw scene's key as shot (log2 of our scene-linear median at EV 0): correlation {_f(kc.get('camera_gain_vs_raw_key'), 2, True)}.", ""]
    cf = s["contrast_fit"]
    slopes = [e["camera_contrast_slope"] for e in entries if math.isfinite(e["camera_contrast_slope"])]
    L += ["## A closed-form stand-in for G3", "", ("The camera's tone curve taken as a power `gamma` about 0.18 turns the JPEG median `J` into the scene level `0.18 (J / 0.18)^(1/gamma)`; `gamma = 1` is G1. "
          f"Least squares against the quantile-matched G3: gamma {_f(cf['gamma'], 2)}, residual RMS {_f(cf['rms_ev'], 2)} EV (G1 alone: {_f(cf['g1_rms_ev'], 2)} EV). "
          f"The camera curves' log-log slopes at the anchor were {_f(min(slopes, default=math.nan), 2)} to {_f(max(slopes, default=math.nan), 2)}."), "",
          "| gamma | median | std | within | max abs diff from G3 |", "|---|---|---|---|---|"]
    for r in s["contrast_sweep"]:
        L.append(f"| {_f(r['gamma'], 2)} | {_f(r['median'], 2, True)} | {_f(r['std'], 2)} | {r['within']}/{r['evaluated']} | {_f(r['max_abs_diff_from_g3'], 2)} |")
    L += [""]
    L += ["## Offset sweep (every photo moved by the same offset)", "", "| offset EV | G1 median | G1 within | G2 median | G2 within | G3 median | G3 within |", "|---|---|---|---|---|---|---|"]
    for r in s["offset_sweep"]:
        L.append(f"| {_f(r['offset_ev'], 2, True)} | {_f(r['g1_median'], 2, True)} | {r['g1_within']}/{r['g1_evaluated']} | {_f(r['g2_median'], 2, True)} | {r['g2_within']}/{r['g2_evaluated']} | {_f(r['g3_median'], 2, True)} | {r['g3_within']}/{r['g3_evaluated']} |")
    L += ["", "## Display placement of mid grey", ""]
    aces = s["aces2_grey_display"]
    L += [(f"ACES 2 (OpenColorIO neutral axis, without the display placement of +{_f(s['display_placement_ev'], 2)} EV that `ym` applies before it): scene 0.18 -> display-linear {_f(aces, 4)}; "
           f"with the placement {_f(s['aces2_grey_display_placed'], 4)}; in our renders (median over photos) {_f(s['display_of_grey_median'], 4)}. "
          "Other placements scale the display median by `g / D(0.18)`. The 0.12 to 0.14 values stand for an ARRI-style placement (assumed, not measured: no ARRI LUT is read); "
          "0.18 is the ISO 12232 convention that camera JPEGs follow."), ""]
    L += ["| mid grey at | G2 median | G2 within |", "|---|---|---|"]
    for r in s["placement_sweep"]:
        L.append(f"| {_f(r['placement'], 2)} | {_f(r['median_ev'], 2, True)} | {r['within']}/{r['evaluated']} |")
    L += ["", "## Algorithm variants", "", "Target (the spec's 0.18):", "", "| target | G1 med | G1 in | G2 med | G2 in | G3 med | G3 in |", "|---|---|---|---|---|---|---|"]
    for r in s["target_sweep"]:
        L.append(f"| {_f(r['target'], 2)} | {_f(r['g1_median'], 2, True)} | {r['g1_within']}/{r['g1_evaluated']} | {_f(r['g2_median'], 2, True)} | {r['g2_within']}/{r['g2_evaluated']} | {_f(r['g3_median'], 2, True)} | {r['g3_within']}/{r['g3_evaluated']} |")
    gs = s["camera_gain_spread"]
    L += ["", (f"Camera engine gain over photos: median {_f(gs['median'], 2, True)} EV, std {_f(gs['std'], 2)}, min {_f(gs['min'], 2, True)}, max {_f(gs['max'], 2, True)} "
          f"(nominal ISO 2720 / 12232 figure {_f(math.log2(m.MID_GREY / m.nominal_grey_fraction()), 2, True)})."), ""]
    for key, label, anchor in (("shrink_sweep", "Keeping part of the camera's own placement, anchored at the median camera gain", s["camera_gain_anchor_ev"]), ("shrink_sweep_nominal", "The same, anchored at the nominal figure", math.log2(m.MID_GREY / m.nominal_grey_fraction()))):
        L += [f"{label} (`ev = anchor + w (EV_auto - anchor)`, anchor {_f(anchor, 2, True)} EV; w = 1 is the spec, 0 a fixed gain):", "", "| w | G1 med | G1 in | G2 med | G2 in | G3 med | G3 std | G3 in |", "|---|---|---|---|---|---|---|---|"]
        for r in s[key]:
            L.append(f"| {_f(r['weight'], 2)} | {_f(r['g1_median'], 2, True)} | {r['g1_within']}/{r['g1_evaluated']} | {_f(r['g2_median'], 2, True)} | {r['g2_within']}/{r['g2_evaluated']} | {_f(r['g3_median'], 2, True)} | {_f(r['g3_std'], 2)} | {r['g3_within']}/{r['g3_evaluated']} |")
        L.append("")
    L += ["Other meters (each normalised to 0.18 after the same 1 %/99 % band):", "", "| statistic | weights | G1 med | G1 in | G2 med | G2 in | G3 med | G3 std | G3 in |", "|---|---|---|---|---|---|---|---|---|"]
    for r in s["statistic_sweep"]:
        L.append(f"| {r['statistic']} | {'flat' if r['flat_weights'] else 'centre'} | {_f(r['g1_median'], 2, True)} | {r['g1_within']}/{r['g1_evaluated']} | {_f(r['g2_median'], 2, True)} | {r['g2_within']}/{r['g2_evaluated']} | {_f(r['g3_median'], 2, True)} | {_f(r['g3_std'], 2)} | {r['g3_within']}/{r['g3_evaluated']} |")
    L += ["", "## What the camera's meter holds constant", ""]
    if s["meter_emulation"]:
        L += [(f"Photos the camera metered (program mode, no flash): {', '.join(s['camera_metered_files'])}. For each statistic `S` of the EV 0 scene luminance, `log2(0.18 / S) + EC` "
              "should be one number if the camera exposes `S` at a fixed fraction of the white level; its spread in stops is below (smallest first)."), "", "| statistic | mean | std | files |", "|---|---|---|---|"]
        for r in s["meter_emulation"]:
            L.append(f"| {r['statistic']} | {_f(r['mean_ev'], 2, True)} | {_f(r['std_ev'], 2)} | {r['files']} |")
    else:
        L.append("Fewer than two photos were metered by the camera (needs the ExposureProgram of real files).")
    L += ["", "## Validation of the study itself", ""]
    rm = s["replica_minus_ym"]
    L += [(f"Python replica of the algorithm minus `ym`'s `EV_auto`: median {_f(rm['median'], 3, True)}, std {_f(rm['std'], 3)}, min {_f(rm['min'], 3, True)}, max {_f(rm['max'], 3, True)} EV "
          "(the replica meters the DNG's 1/8 blocks, ym meters the 1/8 mosaic preview with its clip mask)."), ""]
    gc = s["guard_check"]
    L += [(f"Highlight guard (the only part of `EV_auto` that looks at the scene; the exposure is the camera gain {_f(gc['camera_gain_ev'], 2, True)} EV otherwise): `ym` lowered the exposure on "
          f"{len(gc['lowered_by_ym'])} of {gc['files']} files, the replica says {len(gc['lowered_by_replica'])} would clip; largest reduction {_f(gc['largest_reduction_ev'], 2)} EV, "
          f"never raised: {gc['never_raised']}, at most one stop: {gc['at_most_one_stop']}, agrees with the replica: {gc['agrees_with_replica']}."), ""]
    if "ev_auto_minus_truth" in s:
        t = s["ev_auto_minus_truth"]
        c = s["camera_gain_minus_synthetic"]
        L += [(f"Synthetic scenes (ground truth: the exposure that puts the 18 % grey patch at 0.18): `EV_auto` minus truth median {_f(t['median'], 2, True)} EV, std {_f(t['std'], 2)}, "
              f"{int(t['within'])} of {int(t['evaluated'])} within +-{tol} EV. The estimator of the camera gain against the gain the synthetic JPEG was made with "
              f"({_f(SYNTH_GAIN_EV, 3, True)} EV): median {_f(c['median'], 2, True)}, std {_f(c['std'], 2)}."), ""]
    if failures:
        L += ["## Failures", ""]
        L += [f"- {name}: {why}" for name, why in failures]
    return "\n".join(L) + "\n"


def write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    cols = ["file", "body", "profile_match", "iso", "shutter_s", "f_number", "ev100", "exposure_program", "flash_fired", "camera_metered", "dro", "jpeg_colour_space",
            "ec_ev", "ev_auto", "ev_replica", "ev_replica_minus_ym", "scene_median_at_auto", "jpeg_median", "display_median", "g1_gate_ev", "g2_display_ev",
            "camera_gain_ev", "g3_scene_ev", "camera_gain_at_median_ev", "camera_contrast_slope", "display_of_grey", "display_curve_slope", "scene_contrast_ev", "scene_range_ev",
            "camera_gain_minus_nominal_ev", "ev_auto_minus_nominal_ev", "truth_ev", "ev_auto_minus_truth_ev", "camera_gain_minus_synthetic_ev"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in entries:
            w.writerow(["" if (e.get(c) is None or (isinstance(e.get(c), float) and not math.isfinite(e[c]))) else e.get(c) for c in cols])


def print_table(entries: list[dict[str, Any]], s: dict[str, Any]) -> None:
    print(f"{'file':<46} {'EV_auto':>7} {'G1':>6} {'G2':>6} {'camgain':>7} {'G3':>6}")
    for e in entries:
        print(f"{e['file'][:46]:<46} {e['ev_auto']:>+7.2f} {e['g1_gate_ev']:>+6.2f} {e['g2_display_ev']:>+6.2f} {e['camera_gain_ev']:>+7.2f} {e['g3_scene_ev']:>+6.2f}")
    for k, name in (("g1_gate_ev", "G1"), ("g2_display_ev", "G2"), ("g3_scene_ev", "G3")):
        t = s["metrics"][k]
        print(f"{name}: {int(t['within'])} of {int(t['evaluated'])} within +-{s['tolerance_ev']} EV, median {t['median']:+.2f}, std {t['std']:.2f}")


# --------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path, help="folders or ARW files")
    ap.add_argument("--ym", type=Path, default=DEFAULT_YM, help="the ym binary (default: the dev build)")
    ap.add_argument("--label", default=None, help="name of the run (default: the first folder's name)")
    ap.add_argument("--out", type=Path, default=None, help="work and report folder (default: testdata/eval/<label>)")
    ap.add_argument("--one-per-scene", action="store_true", help="synthetic sets: one raw variant per scene")
    ap.add_argument("--filter", default=None, help="only files whose name contains this text")
    ap.add_argument("--reuse", action="store_true", help="use the cache of an earlier run where it matches the file")
    ap.add_argument("--keep-renders", action="store_true", help="keep the DNG and TIFF renders (they are big)")
    ap.add_argument("--display", default="aces2", help="ym --display (aces2, or arri with an imported display LUT and --ym-home)")
    ap.add_argument("--ym-home", type=Path, default=None, help="YELLOWMOON_HOME for ym (default: an empty folder; needed for --display arri)")
    ap.add_argument("--tolerance-ev", "--tolerance", dest="tolerance", type=float, default=GATE_TOLERANCE_EV, help="the tolerance in EV: for the tallies, and for --gate the bound of |G3| (default %(default)s)")
    ap.add_argument("--gate", action="store_true", help="run as the M1-08 gate: exit 3 when fewer than --min-fraction of the files have |G3| <= --tolerance-ev; print the outliers")
    ap.add_argument("--min-fraction", type=float, default=GATE_MIN_FRACTION, help="--gate: the share of files that must be within the tolerance, in (0, 1] (default %(default)s)")
    ap.add_argument("--json", type=Path, default=None, help="where to write the JSON (default: <out>/results.json)")
    args = ap.parse_args(argv)
    if not args.inputs:
        ap.error("give a folder or files")
    if not (math.isfinite(args.tolerance) and args.tolerance > 0):
        ap.error("--tolerance-ev must be a positive number of EV")
    if not (math.isfinite(args.min_fraction) and 0 < args.min_fraction <= 1):
        ap.error("--min-fraction must be in (0, 1]")
    try:
        files = find_files(args.inputs, args.one_per_scene, args.filter)
    except StudyError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if not files:
        print("error: no ARW files found", file=sys.stderr)
        return 2
    label = args.label or args.inputs[0].resolve().name
    work = (args.out or (REPO / "testdata" / "eval" / label)).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # The empty home folder of the run is removed afterwards (an imported `--ym-home` is the user's and is left alone).
    with tempfile.TemporaryDirectory(prefix="ym-home-", dir=work) as scratch_home:
        ym: Ym | None = None
        if args.ym.exists():
            home = args.ym_home or Path(scratch_home)
            home.mkdir(parents=True, exist_ok=True)
            ym = Ym(args.ym.resolve(), home, args.display)
        elif not args.reuse:
            print(f"error: {args.ym} does not exist; build it with `cargo build -p ym-cli`", file=sys.stderr)
            return 2
        return _run(args, files, work, label, ym)


def _run(args: argparse.Namespace, files: list[Path], work: Path, label: str, ym: Ym | None) -> int:
    """Analyses every file, then writes the JSON, CSV and Markdown reports into `work` (exit code as `main`)."""
    entries: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        try:
            entries.append(analyse(ym, f, work, args.reuse, args.keep_renders, args.display))
            print(f"[{i}/{len(files)}] {f.name}: {time.time() - t0:.0f} s", file=sys.stderr)
        except (StudyError, OSError, ValueError) as err:
            failures.append((f.name, str(err)))
            print(f"[{i}/{len(files)}] {f.name}: FAILED: {err}", file=sys.stderr)
    if not entries:
        print("error: nothing could be analysed", file=sys.stderr)
        return 1
    summary = summarise(entries, args.tolerance)
    gate = evaluate_gate(entries, failures, args.tolerance, args.min_fraction) if args.gate else None
    if gate:
        summary["gate"] = gate
    meta = {
        "label": label,
        "ym": str(args.ym),
        "ym_version": ", ".join(sorted({e["extras"].get("ym_version", "?") for e in entries})),
        "tolerance_ev": args.tolerance,
        "display": args.display,
        "command_examples": (ym.commands[:3] if ym and ym.commands else COMMAND_TEMPLATES),
        "inputs": [str(p) for p in args.inputs],
    }
    doc = _clean({"meta": meta, "files": entries, "failures": [{"file": a, "error": b} for a, b in failures], "summary": summary})
    (args.json or work / "results.json").write_text(json.dumps(doc, indent=1))
    write_csv(work / "results.csv", entries)
    (work / "report.md").write_text(markdown(meta, entries, failures, summary))
    print_table(entries, summary)
    if gate:
        print("\n" + "\n".join(gate_lines(gate)))
    print(f"\nreport: {work / 'report.md'}\njson:   {args.json or work / 'results.json'}", file=sys.stderr)
    if gate and not gate["passed"]:
        return EXIT_GATE_NOT_MET
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
