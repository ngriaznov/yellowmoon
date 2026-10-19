#!/usr/bin/env python3
"""The blind look for the R6 demosaic default.

Renders photos with two R6 demosaic methods (the current default and a candidate), hides which is which
behind randomised A/B labels, and builds a page to compare them. Nothing here changes a default: the owner
decides after looking (`docs/05` R6).

    # 1. a `ym` with the test-only switch (YM_FORCE_DEMOSAIC), see README.md
    cargo build -p ym-cli --features force-demosaic

    # 2. render, crop, randomise, write the page and the key
    python3 -I tools/eval/demosaic/blind_look.py render --ym target/debug/ym --out testdata/demosaic_gate

    # 3. open testdata/demosaic_gate/index.html, pick, "Download results", then
    python3 -I tools/eval/demosaic/blind_look.py score --out testdata/demosaic_gate --picks ra05-picks.json

What `render` writes under `--out` (git-ignored, never committed):

    index.html            only the A/B labels, no method names
    key.json              which method is A and B for every file, file hashes, crop boxes
    images/<set>/<name>/  A.jpg, B.jpg (full frame), <kind>_A.png, <kind>_B.png (1:1 crops)

* **Sets.** `real` = the local real ARW files (`testdata/real/files`, local evaluation only, never
  committed, never redistributed); `synthetic` = the 61 files of the synthetic golden set that the golden
  runner renders with default settings (`testdata/synthetic/arw`, the `default` set of
  `tools/golden/manifest.json`).
* **Renders.** One `ym render` per photo and method, defaults as the app has them (an empty
  `YELLOWMOON_HOME`: the Neutral look, ACES 2, automatic development with Yellowmoon's own Auto, `--apple-auto off`,
  so that a Mac with Apple Intelligence makes the same pictures, `docs/19`), the classical demosaic forced
  (`--no-neural`), PNG with 8 bits, the method picked by `YM_FORCE_DEMOSAIC`. The two renders of a photo
  differ in the R6 method only; the exposure and colour estimates do not read the demosaic.
* **Full frames** are re-encoded as JPEG (quality 92, no chroma subsampling) from the PNG; **crops** are cut
  from the PNG, so they are lossless.
* **Randomisation.** A seeded, balanced shuffle over the files (half of them have the baseline as A, half
  the candidate), one draw per file: the full frame and its four crops share it. `key.json` holds the
  result; the page does not.
* **Crops.** Four 1:1 windows per photo (edges, fine texture, saturated colour, shadows), chosen on the
  mean of the two renders so that the choice does not favour either, from cell statistics of gradient,
  Laplacian and saturation (`pick_windows`). They are disjoint, even-aligned (the same CFA phase) and stay
  clear of the frame border.

Needs numpy and Pillow. The tests (no `ym`, no ARW files needed):
`python3 -I -m unittest discover -s tools/eval/demosaic -p 'test_*.py'`.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SEED = "ra05-blind-v1"
BASELINE = "menon"
CANDIDATE = "adaptive_ratio"
#: The method names of `ym_gpu::cpu::demosaic::Method::name`.
METHODS = ("bilinear", "menon", "lmmse", "adaptive_ratio", "log_guided")
KINDS = ("edges", "texture", "colour", "shadows")
KIND_TITLES = {
    "edges": "Edges",
    "texture": "Fine texture",
    "colour": "Saturated colour",
    "shadows": "Shadows",
}
#: Cell size in pixels of the window search (the windows are multiples of it, so they start on even pixels).
CELL = 32
#: Rows per strip of the feature pass (a multiple of CELL): keeps the float temporaries small at 61 MP.
STRIP = 256
#: Largest crop in pixels.
CROP_W, CROP_H = 480, 320
#: Rec. 709 luma weights, for the display-referred value of the 8-bit images.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
RENDER_TIMEOUT_S = 3600


# ---------------------------------------------------------------------------------------------
# Randomised labels
# ---------------------------------------------------------------------------------------------


def assign_labels(items: list[str], seed: str, baseline: str, candidate: str) -> dict[str, dict[str, str]]:
    """Which method is `A` and which is `B` for every item.

    A balanced coin: the items are put in sorted order, half of them (rounded down) get the baseline as
    `A`, the rest the candidate, and the list is shuffled with a generator seeded from `seed`. The result
    depends on the item names and the seed only, never on the order they are given in.
    """
    if baseline == candidate:
        raise ValueError("the baseline and the candidate must differ")
    ordered = sorted(set(items))
    flips = [0] * (len(ordered) // 2) + [1] * (len(ordered) - len(ordered) // 2)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))
    rng.shuffle(flips)
    return {
        item: ({"A": baseline, "B": candidate} if flip == 0 else {"A": candidate, "B": baseline})
        for item, flip in zip(ordered, flips)
    }


# ---------------------------------------------------------------------------------------------
# Crop windows
# ---------------------------------------------------------------------------------------------


def cell_features(rgb: np.ndarray, scale: float = 1.0) -> dict[str, np.ndarray]:
    """Per-cell sums (cells of CELL x CELL pixels) of the maps the window search reads.

    `rgb` is (H, W, 3), any numeric dtype, times `scale` gives values in [0, 1] (display-referred 8-bit
    images). The frame is cut to whole cells. Maps, per pixel: `g` gradient magnitude of the luma
    (central differences), `lap` absolute 4-neighbour Laplacian of the luma, `strong` 1 where `g > 0.1`,
    `sat` HSV saturation, `lum` luma, `cedge` saturation times the gradient magnitudes of R-G and B-G
    (saturated colour boundaries, where false colour shows). Strips of STRIP rows keep the temporaries
    small.
    """
    h, w = rgb.shape[:2]
    ch, cw = h // CELL, w // CELL
    names = ("g", "lap", "strong", "sat", "lum", "cedge")
    out = {k: np.zeros((ch, cw), dtype=np.float64) for k in names}
    if ch == 0 or cw == 0:
        return out
    width = cw * CELL
    for y0 in range(0, ch * CELL, STRIP):
        y1 = min(y0 + STRIP, ch * CELL)
        a, b = max(y0 - 1, 0), min(y1 + 1, h)
        blk = rgb[a:b, :width].astype(np.float32) * np.float32(scale)
        lum = blk @ LUMA
        p = np.pad(lum, 1, mode="edge")
        gx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
        gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
        g = np.hypot(gx, gy)
        lap = np.abs(p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * lum)
        mx, mn = blk.max(axis=2), blk.min(axis=2)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        chroma_grad = 0.0
        for diff in (blk[..., 0] - blk[..., 1], blk[..., 2] - blk[..., 1]):
            q = np.pad(diff, 1, mode="edge")
            chroma_grad = chroma_grad + np.hypot((q[1:-1, 2:] - q[1:-1, :-2]) * 0.5, (q[2:, 1:-1] - q[:-2, 1:-1]) * 0.5)
        maps = {
            "g": g,
            "lap": lap,
            "strong": (g > 0.1).astype(np.float32),
            "sat": sat,
            "lum": lum,
            "cedge": sat * chroma_grad,
        }
        off = y0 - a
        rows = y1 - y0
        for name, arr in maps.items():
            core = arr[off : off + rows]
            out[name][y0 // CELL : y1 // CELL] = core.reshape(rows // CELL, CELL, cw, CELL).sum(axis=(1, 3))
    return out


def _window_means(cells: np.ndarray, wh: int, ww: int) -> np.ndarray:
    """Mean per pixel of every window of `wh` x `ww` cells (indexed by its top-left cell)."""
    integral = np.pad(cells, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    sums = integral[wh:, ww:] - integral[:-wh, ww:] - integral[wh:, :-ww] + integral[:-wh, :-ww]
    return sums / float(wh * ww * CELL * CELL)


def pick_windows(cells: dict[str, np.ndarray], wh: int, ww: int, margin: int = 1) -> dict[str, tuple[int, int]]:
    """The top-left cell `(cy, cx)` of the crop window of every kind in KINDS.

    Windows are `wh` x `ww` cells, disjoint, and at least `margin` cells from the frame border (when the
    frame is too small for that, the margin and the disjointness are given up, in that order).

    * **edges**: the most strong-gradient pixels, among windows where they are at most 30 % of the pixels
      (a window full of strong gradient is texture or noise) and the exposure is neither dark nor clipped;
    * **texture**: the most Laplacian energy among windows with few strong gradients (at most 12 %): fine
      detail, not a single edge;
    * **colour**: the most saturated colour boundary energy (`cedge`);
    * **shadows**: the most Laplacian energy among windows with a mean luma of 0.02 to 0.22, else the
      darkest window.

    The scores are means over the window of maps computed on the mean of the two renders, so the choice
    does not depend on which method is A.
    """
    ch, cw = cells["g"].shape
    if ch < wh or cw < ww:
        raise ValueError(f"a window of {wh} x {ww} cells does not fit a frame of {ch} x {cw} cells")
    mean = {k: _window_means(v, wh, ww) for k, v in cells.items()}
    shape = mean["g"].shape
    yy, xx = np.indices(shape)
    exposed = (mean["lum"] >= 0.06) & (mean["lum"] <= 0.94)
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "edges": (mean["strong"], exposed & (mean["strong"] <= 0.30)),
        "texture": (mean["lap"], exposed & (mean["strong"] <= 0.12)),
        "colour": (mean["cedge"], exposed),
        "shadows": (mean["lap"], (mean["lum"] >= 0.02) & (mean["lum"] <= 0.22)),
    }
    picked: dict[str, tuple[int, int]] = {}
    for kind in KINDS:
        score, allowed = candidates[kind]
        for use_margin, use_disjoint in ((True, True), (False, True), (False, False)):
            ok = allowed.copy()
            if use_margin:
                ok &= (yy >= margin) & (xx >= margin) & (yy <= ch - wh - margin) & (xx <= cw - ww - margin)
            if use_disjoint:
                for cy0, cx0 in picked.values():
                    ok &= ~((np.abs(yy - cy0) < wh) & (np.abs(xx - cx0) < ww))
            if ok.any():
                break
        else:
            # Nothing satisfies the constraint of this kind: the window of the best raw score that is
            # still disjoint, else the best raw score.
            ok = np.ones(shape, dtype=bool)
            for cy0, cx0 in picked.values():
                ok &= ~((np.abs(yy - cy0) < wh) & (np.abs(xx - cx0) < ww))
            if not ok.any():
                ok = np.ones(shape, dtype=bool)
        if kind == "shadows" and not allowed.any():
            score = -mean["lum"]
        masked = np.where(ok, score, -np.inf)
        flat = int(np.argmax(masked))  # first maximum in row-major order: deterministic ties
        picked[kind] = (flat // shape[1], flat % shape[1])
    return picked


def window_size_cells(width: int, height: int) -> tuple[int, int] | None:
    """The crop window `(wh, ww)` in cells for a frame, or `None` when the frame is too small for crops.

    CROP_W x CROP_H pixels, but never more than a third of each side; at least 2 x 2 cells.
    """
    ww = min(CROP_W, (width // 3) // CELL * CELL) // CELL
    wh = min(CROP_H, (height // 3) // CELL * CELL) // CELL
    if ww < 2 or wh < 2:
        return None
    return wh, ww


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ym_has_switch(ym: Path) -> bool:
    """Whether this `ym` was built with the test-only `force-demosaic` feature (it contains the name of
    the environment variable it reads)."""
    needle = b"YM_FORCE_DEMOSAIC"
    with ym.open("rb") as f:
        tail = b""
        while chunk := f.read(1 << 22):
            if needle in tail + chunk:
                return True
            tail = chunk[-len(needle) :]
    return False


def render_one(ym: Path, src: Path, method: str, out_dir: Path, home: Path) -> tuple[Path, float]:
    """One `ym render` of `src` with the R6 method forced; returns the PNG path and the wall seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, YELLOWMOON_HOME=str(home), YM_FORCE_DEMOSAIC=method)
    cmd = [
        str(ym), "render", "--json", "--quiet", "--overwrite", "--no-neural", "--apple-auto", "off",
        "--format", "png", "--bits", "8", "--colour-space", "srgb",
        "--name", "{name}", "--out", str(out_dir), str(src),
    ]  # fmt: skip
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=RENDER_TIMEOUT_S, check=False)
    elapsed = time.monotonic() - t0
    last = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.strip()), "")
    try:
        doc = json.loads(last)
    except json.JSONDecodeError:
        doc = {}
    if proc.returncode != 0 or doc.get("status") != "ok" or not doc.get("output"):
        why = (doc.get("error") or {}).get("message") if isinstance(doc.get("error"), dict) else None
        raise RuntimeError(f"ym render {src.name} ({method}) failed with exit {proc.returncode}: {why or proc.stderr.strip()[:300]}")
    return Path(doc["output"]), elapsed


def save_jpeg(arr: np.ndarray, path: Path, icc: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"quality": 92, "subsampling": 0}
    if icc:
        kwargs["icc_profile"] = icc
    Image.fromarray(arr).save(path, "JPEG", **kwargs)


def build_item(
    item: str,
    src: Path,
    labels: dict[str, str],
    baseline: str,
    candidate: str,
    ym: Path,
    out: Path,
    work: Path,
    keep_renders: bool,
    render: Callable[[Path, Path, str, Path, Path], tuple[Path, float]] = render_one,
) -> dict[str, Any]:
    """Renders `src` with both methods and writes the full frames and the crops of one item. Returns its
    entry of `key.json`."""
    name = item.split("/", 1)[1]
    home = work / "home"
    pngs: dict[str, Path] = {}
    seconds: dict[str, float] = {}
    for method in (baseline, candidate):
        pngs[method], seconds[method] = render(ym, src, method, work / "renders" / item / method, home)
    imgs: dict[str, np.ndarray] = {}
    icc: bytes | None = None
    for method, path in pngs.items():
        with Image.open(path) as im:
            icc = icc or im.info.get("icc_profile")
            imgs[method] = np.asarray(im.convert("RGB"))
    if imgs[baseline].shape != imgs[candidate].shape:
        raise RuntimeError(f"{name}: the two renders differ in size: {imgs[baseline].shape} and {imgs[candidate].shape}")
    h, w = imgs[baseline].shape[:2]
    identical = bool(np.array_equal(imgs[baseline], imgs[candidate]))
    diff = np.abs(imgs[baseline].astype(np.int16) - imgs[candidate].astype(np.int16))
    stats = {
        "identical": identical,
        "differing_samples_pct": float(100.0 * np.count_nonzero(diff) / diff.size),
        "max_abs_diff_codes": int(diff.max()),
        "mean_abs_diff_codes": float(diff.mean()),
    }
    del diff
    folder = out / "images" / item
    files: dict[str, dict[str, str]] = {}
    for label in ("A", "B"):
        path = folder / f"{label}.jpg"
        save_jpeg(imgs[labels[label]], path, icc)
        files[f"full_{label}"] = {"path": path.relative_to(out).as_posix(), "sha256": sha256_file(path)}
    crops: dict[str, dict[str, Any]] = {}
    win = window_size_cells(w, h)
    if win is not None:
        wh, ww = win
        # The window search runs on the sum of the two renders (uint16), scaled to [0, 1].
        total = imgs[baseline].astype(np.uint16) + imgs[candidate].astype(np.uint16)
        cells = cell_features(total, 1.0 / 510.0)
        del total
        for kind, (cy, cx) in pick_windows(cells, wh, ww).items():
            x0, y0 = cx * CELL, cy * CELL
            box = (x0, y0, x0 + ww * CELL, y0 + wh * CELL)
            entry: dict[str, Any] = {"box_xyxy": list(box)}
            for label in ("A", "B"):
                crop = imgs[labels[label]][box[1] : box[3], box[0] : box[2]]
                path = folder / f"{kind}_{label}.png"
                Image.fromarray(np.ascontiguousarray(crop)).save(path, "PNG", compress_level=6)
                entry[label] = path.relative_to(out).as_posix()
            crops[kind] = entry
    if keep_renders:
        for method, path in pngs.items():
            keep = out / "renders" / item
            keep.mkdir(parents=True, exist_ok=True)
            (keep / f"{method}.png").write_bytes(path.read_bytes())
    for path in pngs.values():
        path.unlink(missing_ok=True)
    return {
        "name": name,
        "source_file": src.name,
        "width": w,
        "height": h,
        "A": labels["A"],
        "B": labels["B"],
        "render_seconds": seconds,
        "difference": stats,
        "files": files,
        "crops": crops,
    }


# ---------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------

PAGE_CSS = """
:root { color-scheme: dark; --bg: #1c1c1e; --fg: #e8e8ea; --dim: #9a9aa0; --line: #38383c; --acc: #e9c46a; }
* { box-sizing: border-box; }
body { margin: 0; padding: 0 16px 80px; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); padding: 10px 0; }
h1 { font-size: 18px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 28px 0 6px; } h3 { font-size: 14px; margin: 14px 0 4px; color: var(--dim); font-weight: 600; }
p, li { max-width: 70ch; } .dim { color: var(--dim); }
nav a { color: var(--acc); margin-right: 14px; }
section { border-top: 1px solid var(--line); padding-top: 4px; }
summary { cursor: pointer; margin-top: 20px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; align-items: start; }
.pair figure { margin: 0; } .pair img { display: block; max-width: 100%; height: auto; background: #000; }
figcaption { font-weight: 600; padding: 2px 0; }
.crops { display: flex; flex-wrap: wrap; gap: 16px 24px; }
.crop { overflow-x: auto; }
.crop .pair { grid-template-columns: auto auto; width: max-content; }
.crop img { image-rendering: pixelated; max-width: none; }
.flip figure + figure { display: none; } .flip.showB figure:first-child { display: none; } .flip.showB figure + figure { display: block; }
.picks label { margin-right: 12px; white-space: nowrap; }
button { font: inherit; background: #2c2c2e; color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.sticky-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
@media (max-width: 760px) { .pair { grid-template-columns: 1fr; } }
"""

PAGE_JS = """
const STORE = 'ra05-picks-v1:' + document.body.dataset.run;
let picks = {};
try { picks = JSON.parse(localStorage.getItem(STORE) || '{}'); } catch (e) { picks = {}; }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(picks)); } catch (e) {} };
const count = () => { document.getElementById('done').textContent = Object.keys(picks).length + ' of ' + document.querySelectorAll('.picks').length + ' answered'; };
document.querySelectorAll('.picks').forEach(el => {
  const id = el.dataset.id;
  el.querySelectorAll('input').forEach(inp => {
    if (picks[id] === inp.value) inp.checked = true;
    inp.addEventListener('change', () => { picks[id] = inp.value; save(); count(); });
  });
});
document.querySelectorAll('.flipbtn').forEach(b => b.addEventListener('click', () => {
  b.closest('.pairbox').querySelector('.pair').classList.toggle('flip');
}));
document.querySelectorAll('.pair').forEach(p => p.addEventListener('click', () => { if (p.classList.contains('flip')) p.classList.toggle('showB'); }));
const out = () => JSON.stringify({ version: 1, picks: picks }, null, 1);
document.getElementById('copy').addEventListener('click', () => { navigator.clipboard.writeText(out()); });
document.getElementById('download').addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out()], { type: 'application/json' }));
  a.download = 'ra05-picks.json'; a.click();
});
document.getElementById('clear').addEventListener('click', () => { if (confirm('Clear all answers?')) { picks = {}; save(); location.reload(); } });
count();
"""


def _picks(pick_id: str) -> str:
    return (
        f'<div class="picks" data-id="{html.escape(pick_id, quote=True)}">'
        + "".join(
            f'<label><input type="radio" name="{html.escape(pick_id, quote=True)}" value="{v}"> {t}</label>'
            for v, t in (("A", "A is better"), ("B", "B is better"), ("same", "No difference"))
        )
        + "</div>"
    )


def _pair(a: str, b: str, alt: str) -> str:
    return (
        '<div class="pair">'
        f'<figure><img src="{html.escape(a, quote=True)}" alt="{html.escape(alt)} A" loading="lazy"><figcaption>A</figcaption></figure>'
        f'<figure><img src="{html.escape(b, quote=True)}" alt="{html.escape(alt)} B" loading="lazy"><figcaption>B</figcaption></figure>'
        "</div>"
    )


def _flip_pair(a: str, b: str, alt: str) -> str:
    """A crop pair: the same markup as `_pair`, so that flip mode can show one figure at a time."""
    return (
        '<div class="pair">'
        f'<figure><img src="{html.escape(a, quote=True)}" alt="{html.escape(alt)} A" loading="lazy"><figcaption>A</figcaption></figure>'
        f'<figure><img src="{html.escape(b, quote=True)}" alt="{html.escape(alt)} B" loading="lazy"><figcaption>B</figcaption></figure>'
        "</div>"
    )


def run_id(seed: str, baseline: str, candidate: str, items: list[str]) -> str:
    """A short fingerprint of a run (seed, methods, file names), so that answers kept in the browser for
    one run are never read back for another. A hash, so the page reveals neither the methods nor the labels."""
    text = "|".join([seed, baseline, candidate, *sorted(items)])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def index_html(entries: dict[str, dict[str, Any]], run: str = "test") -> str:
    """The page. It shows file names and A/B labels only; no method name appears in it. `run` is the
    fingerprint of the run (`run_id`) that keys the answers kept in the browser."""
    # A file whose two renderings are identical holds demosaiced pixels already (a Lossless M or S
    # file), or is flat: there is nothing to compare, so the page leaves it out (key.json lists it).
    shown = {k: e for k, e in entries.items() if not e.get("difference", {}).get("identical")}
    left_out = len(entries) - len(shown)
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for item, entry in sorted(shown.items()):
        groups.setdefault(item.split("/", 1)[0], []).append((item, entry))
    body: list[str] = []
    nav = " ".join(f'<a href="#set-{html.escape(s)}">{html.escape(s)} ({len(v)})</a>' for s, v in groups.items())
    for set_name, items in groups.items():
        # The real photos are open; the synthetic scenes (55 files) are folded away until asked for.
        opened = " open" if set_name == "real" else ""
        body.append(
            f'<details id="set-{html.escape(set_name)}"{opened}><summary><h2 style="display:inline">{html.escape(set_name)} files ({len(items)})</h2></summary>'
        )
        for item, entry in items:
            anchor = html.escape(item.replace("/", "-"), quote=True)
            sec = [f'<section id="{anchor}"><h2>{html.escape(entry["name"])} <span class="dim">{entry["width"]} x {entry["height"]}</span></h2>']
            sec.append('<div class="pairbox">')
            sec.append(_pair(entry["files"]["full_A"]["path"], entry["files"]["full_B"]["path"], entry["name"]))
            sec.append(_picks(f"{item}/full"))
            sec.append("</div>")
            if entry["crops"]:
                sec.append('<h3>1:1 crops (open at 100 % browser zoom)</h3><div class="crops">')
                for kind in KINDS:
                    crop = entry["crops"].get(kind)
                    if not crop:
                        continue
                    sec.append(f'<div class="crop pairbox"><h3>{KIND_TITLES[kind]} <button class="flipbtn" type="button">Flip mode</button></h3>')
                    sec.append(_flip_pair(crop["A"], crop["B"], f'{entry["name"]} {kind}'))
                    sec.append(_picks(f"{item}/{kind}"))
                    sec.append("</div>")
                sec.append("</div>")
            sec.append("</section>")
            body.append("".join(sec))
        body.append("</details>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Demosaic blind look</title>"
        f"<style>{PAGE_CSS}</style></head><body data-run=\"{html.escape(run, quote=True)}\">"
        "<header><h1>Demosaic blind look</h1>"
        '<div class="sticky-actions"><nav>' + nav + "</nav>"
        '<span id="done" class="dim"></span>'
        '<button id="download" type="button">Download results</button>'
        '<button id="copy" type="button">Copy results</button>'
        '<button id="clear" type="button">Clear</button></div></header>'
        "<p>Every photo is shown twice, rendered two ways, labelled A and B at random. For each pair, say which "
        "looks better, or that you see no difference. Judge edges, fine detail, colour at saturated edges and "
        "noise in the shadows; the crops are at 1:1 (keep the browser at 100 %). Flip mode shows one image at a "
        "time at the same place: click the image to switch. Answers are kept in this browser until you clear "
        "them. Nothing is sent anywhere.</p>"
        + (f'<p class="dim">{left_out} files are left out: both renderings are identical.</p>' if left_out else "")
        + "".join(body)
        + f"<script>{PAGE_JS}</script></body></html>"
    )


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def score(key: dict[str, Any], picks: dict[str, str]) -> dict[str, Any]:
    """Tallies the picks by method: for every set and kind (and in total) how often the baseline won,
    the candidate won, or the answer was "no difference"."""
    baseline, candidate = key["baseline"], key["candidate"]
    table: dict[str, dict[str, int]] = {}

    def bump(bucket: str, outcome: str) -> None:
        row = table.setdefault(bucket, {"baseline": 0, "candidate": 0, "same": 0})
        row[outcome] += 1

    unknown: list[str] = []
    for pick_id, choice in sorted(picks.items()):
        item, _, part = pick_id.rpartition("/")
        labels = key["items"].get(item)
        if labels is None or choice not in ("A", "B", "same"):
            unknown.append(pick_id)
            continue
        outcome = "same" if choice == "same" else ("baseline" if labels[choice] == baseline else "candidate")
        set_name = item.split("/", 1)[0]
        for bucket in ("all", f"set:{set_name}", f"kind:{part}", f"set:{set_name}/kind:{part}"):
            bump(bucket, outcome)
    for row in table.values():
        row["sign_test_p"] = sign_test_p(row["candidate"], row["baseline"])  # type: ignore[assignment]
    return {"baseline": baseline, "candidate": candidate, "tallies": table, "unknown_picks": unknown}


def format_score(result: dict[str, Any]) -> str:
    lines = [f"baseline {result['baseline']} against candidate {result['candidate']}", ""]
    lines.append(f"{'bucket':<28} {'baseline':>9} {'candidate':>10} {'same':>6} {'sign test p':>12}")
    for bucket, row in sorted(result["tallies"].items()):
        lines.append(f"{bucket:<28} {row['baseline']:>9} {row['candidate']:>10} {row['same']:>6} {row['sign_test_p']:>12.4f}")
    if result["unknown_picks"]:
        lines.append("")
        lines.append(f"ignored {len(result['unknown_picks'])} answers that match no item of the key: {result['unknown_picks'][:5]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def find_sources(real: Path, synthetic: Path, only: list[str] | None) -> dict[str, Path]:
    """`{item: file}` for both sets, sorted. `item` is `real/<stem>` or `synthetic/<stem>`."""
    items: dict[str, Path] = {}
    for set_name, folder in (("real", real), ("synthetic", synthetic)):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() == ".arw" and path.is_file():
                items[f"{set_name}/{path.stem}"] = path
    if only:
        items = {k: v for k, v in items.items() if any(o in k for o in only)}
    return items


def cmd_render(args: argparse.Namespace) -> int:
    ym = Path(args.ym).resolve()
    if not ym.is_file():
        print(f"error: {ym} does not exist: build it with `cargo build -p ym-cli --features force-demosaic`", file=sys.stderr)
        return 2
    if not ym_has_switch(ym):
        print(
            f"error: {ym} was built without the test-only force-demosaic feature, so it ignores YM_FORCE_DEMOSAIC and "
            "both renders would be the default. Build it with `cargo build -p ym-cli --features force-demosaic`.",
            file=sys.stderr,
        )
        return 2
    if args.baseline not in METHODS or args.candidate not in METHODS:
        print(f"error: methods are {METHODS}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = find_sources(Path(args.real), Path(args.synthetic), args.only)
    if not sources:
        print("error: no ARW files found", file=sys.stderr)
        return 2
    labels = assign_labels(list(sources), args.seed, args.baseline, args.candidate)
    key_path = out / "key.json"
    key: dict[str, Any] = {}
    if args.reuse and key_path.is_file():
        key = json.loads(key_path.read_text())
        if (key.get("baseline"), key.get("candidate"), key.get("seed")) != (args.baseline, args.candidate, args.seed):
            print("error: --reuse: key.json was made with other methods or another seed", file=sys.stderr)
            return 2
    # `key.json` keeps the labels under `items` and everything else about a file under `details`.
    entries: dict[str, dict[str, Any]] = {
        item: {**labels_of, **key.get("details", {}).get(item, {})} for item, labels_of in key.get("items", {}).items()
    }
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ra05-") as tmp:
        work = Path(tmp)
        for n, (item, src) in enumerate(sources.items(), 1):
            if args.reuse and item in entries and all((out / f["path"]).is_file() for f in entries[item]["files"].values()):
                print(f"[{n}/{len(sources)}] {item}: kept", flush=True)
                continue
            t0 = time.monotonic()
            try:
                entries[item] = build_item(item, src, labels[item], args.baseline, args.candidate, ym, out, work, args.keep_renders)
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as err:
                failures.append(f"{item}: {err}")
                print(f"[{n}/{len(sources)}] {item}: FAILED {err}", file=sys.stderr, flush=True)
                continue
            d = entries[item]["difference"]
            print(
                f"[{n}/{len(sources)}] {item}: {time.monotonic() - t0:.0f} s, {d['differing_samples_pct']:.2f} % of samples differ"
                f" (max {d['max_abs_diff_codes']} codes){' IDENTICAL' if d['identical'] else ''}",
                flush=True,
            )
            # Write as we go, so a long run that stops keeps what it has.
            _write_outputs(out, args, ym, entries)
    _write_outputs(out, args, ym, entries)
    identical = [k for k, e in entries.items() if e["difference"]["identical"]]
    print(f"{len(entries)} files, {len(identical)} identical pairs, {len(failures)} failed; page: {out / 'index.html'}")
    if entries and len(identical) == len(entries):
        print("error: every pair is identical: the switch did not change the method", file=sys.stderr)
        return 1
    return 1 if failures else 0


def _write_outputs(out: Path, args: argparse.Namespace, ym: Path, entries: dict[str, dict[str, Any]]) -> None:
    labels = {item: {"A": e["A"], "B": e["B"]} for item, e in entries.items()}
    key = {
        "about": "Which R6 demosaic method is A and which is B for every file of the blind look (the R6 demosaic default, docs/05). "
        "Do not open this before you have answered.",
        "seed": args.seed,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "ym": {"path": str(ym), "sha256": sha256_file(ym), "force_demosaic_switch": True},
        "items": labels,
        "details": {item: {k: v for k, v in e.items() if k not in ("A", "B")} for item, e in entries.items()},
    }
    # `items` carries the labels in the shape `score` reads; `details` the rest.
    (out / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    run = run_id(args.seed, args.baseline, args.candidate, list(entries))
    (out / "index.html").write_text(index_html(entries, run))


def cmd_score(args: argparse.Namespace) -> int:
    key = json.loads((Path(args.out) / "key.json").read_text())
    picks_doc = json.loads(Path(args.picks).read_text())
    picks = picks_doc.get("picks", picks_doc)
    result = score(key, picks)
    print(format_score(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render both methods, crop, randomise, write the page and the key")
    r.add_argument("--ym", default="target/debug/ym", help="a `ym` built with --features force-demosaic")
    r.add_argument("--out", default="testdata/demosaic_gate")
    r.add_argument("--real", default="testdata/real/files")
    r.add_argument("--synthetic", default="testdata/synthetic/arw")
    r.add_argument("--baseline", default=BASELINE, help=f"the current default (default {BASELINE})")
    r.add_argument("--candidate", default=CANDIDATE, help=f"the candidate (default {CANDIDATE})")
    r.add_argument("--seed", default=SEED)
    r.add_argument("--only", nargs="*", help="only items whose name contains one of these words (for example ILCE-7M3 chart__D65)")
    r.add_argument("--reuse", action="store_true", help="keep the items already in key.json and render the rest")
    r.add_argument("--keep-renders", action="store_true", help="keep the full-size PNG of every render under <out>/renders (large)")
    r.set_defaults(fn=cmd_render)
    s = sub.add_parser("score", help="tally the answers downloaded from the page against key.json")
    s.add_argument("--out", default="testdata/demosaic_gate")
    s.add_argument("--picks", required=True, help="the ra05-picks.json the page downloads")
    s.add_argument("--json", help="also write the tallies here")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
