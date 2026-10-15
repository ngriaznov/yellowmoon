#!/usr/bin/env python3
"""Compares darktable's embedded-metadata lens correction with our model (docs/05 section Verifying
lens corrections; gates in docs/13: vignetting gain within 0.5%, corner displacement within 0.2 px).

Inputs are the directories written by tools/lens/darktable_oracle.sh (`lens_in.pfm.gz` and
`lens_out.pfm.gz`: the float image going into and coming out of darktable's lens module) and the lens
tags of the same file as JSON: {"vignetting": [...], "distortion": [...], "ca": [...]} (raw int16
knots; the frame size is taken from the images). Each comparison must be rendered with only the
correction it measures (`darktable_oracle.sh --lens-flags vignetting|distortion|ca`): darktable zooms
the corrected image, so the corrections interact in a combined render.

  compare_with_darktable.py vignette   --dir OUT/flat   --tags tags.json
  compare_with_darktable.py distortion --dir OUT/vlines --tags tags.json [--ignore-ca]
  compare_with_darktable.py distortion --dir OUT/hlines --tags tags.json [--ignore-ca] --horizontal
  compare_with_darktable.py ca         --dir OUT/vlines --tags tags.json
  compare_with_darktable.py selftest

`vignette` needs a flat-field ARW. `distortion` needs an ARW with dark lines on a light field, one
line every 100 px (`lensgen`: vlines.ARW, hlines.ARW with --horizontal). `ca` needs the vertical
lines with a large CA (`lensgen OUT 8`). All print the numbers and exit 1 when a gate fails.

How the geometry is compared: the sub-pixel centre of every line is measured in darktable's output
(per colour plane) and compared with where our model puts it. darktable does not crop the corrected
image, it zooms it up so that no black corner is left, by the factor `lens_model.darktable_zoom`
(measured to agree with the rule to 1e-6); `--horizontal` analyses the transposed image.
`selftest` checks this script itself on synthetic data (a matching "darktable" must pass, one that
differs must fail) and needs neither darktable nor files.

Dependencies: numpy, scipy. The scripts run as `python3 -I`, which does not see pip's per-user site: install them as root
(pip3 install numpy scipy), from apt (python3-numpy python3-scipy) or in a venv that comes first on PATH.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lens_io  # noqa: E402
import lens_model as lm  # noqa: E402
import record_oracle as ro  # noqa: E402

GAIN_TOL = 0.005        # docs/13: vignetting gain within 0.5%
DISPLACEMENT_TOL = 0.2  # docs/13: corner displacement within 0.2 px
CA_TOL = 0.2            # px, the same gate for the red and blue planes


def radius_map(h, w):
    """Normalised radius of every pixel in darktable's convention (pixel indices from (W/2, H/2))."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(xx - w / 2.0, yy - h / 2.0) / math.hypot(w / 2.0, h / 2.0)


# --------------------------------------------------------------------------- vignetting

def measure_gain_profile(lin_in, lin_out, bins=40, r_max=0.95):
    """Median of out / in per radial bin (mean over the colour planes first)."""
    h, w = lin_in.shape[:2]
    ratio = lin_out.mean(axis=2) / np.maximum(lin_in.mean(axis=2), 1e-6)
    r = radius_map(h, w)
    edges = np.linspace(0.0, r_max, bins + 1)
    centres, gains = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi)
        if sel.any():
            centres.append(0.5 * (lo + hi))
            gains.append(float(np.median(ratio[sel])))
    return np.array(centres), np.array(gains)


def compare_vignette(lin_in, lin_out, vignetting_knots):
    r, measured = measure_gain_profile(lin_in, lin_out)
    expected = lm.vignette_gain(vignetting_knots, r)
    dev = np.abs(measured / expected - 1.0)
    return dict(max_rel_dev=float(dev.max()), worst_r=float(r[dev.argmax()]), passed=bool(dev.max() <= GAIN_TOL))


# --------------------------------------------------------------------------- geometry

def residuals(lin_out, distortion, ca, use_distortion, use_ca, channels, horizontal=False):
    """Measured minus modelled line position (px) for every matched (row, line, plane) of the lens
    output, with the radius (in the output) of each. `horizontal` analyses the transposed image."""
    img = lin_out.transpose(1, 0, 2) if horizontal else lin_out
    h, w = img.shape[:2]
    rows = list(range(60, h, 160))
    zoom, samples, _ = ro.line_samples(img, {"distortion": distortion, "ca": ca}, use_distortion, use_ca, channels, rows)
    if len(samples) < 20:
        return dict(error=f"only {len(samples)} matched line positions")
    dev = np.array([[a - b for a, b in zip(s[2], s[3])] for s in samples])
    r = np.array([math.hypot(s[3][1 if len(s[3]) > 1 else 0] - w / 2.0, s[0] - h / 2.0) / math.hypot(w / 2.0, h / 2.0) for s in samples])
    return dict(zoom=zoom, samples=len(samples), dev=dev, r=r)


def summarise(res, tol):
    if "error" in res:
        return dict(error=res["error"], passed=False)
    dev, r = res["dev"], res["r"]
    corner = r > 0.8
    worst_corner = float(np.abs(dev[corner]).max()) if corner.any() else float("nan")
    return dict(darktable_zoom_rule=res["zoom"], samples=res["samples"],
                rms_px=np.sqrt(np.mean(dev ** 2, axis=0)).tolist(),
                max_px=np.abs(dev).max(axis=0).tolist(), max_corner_px=worst_corner,
                passed=bool(np.abs(dev).max() <= tol))


def compare_distortion(lin_out, distortion, ca, horizontal=False):
    res = residuals(lin_out, distortion, ca, True, bool(len(ca)), [1], horizontal)
    return summarise(res, DISPLACEMENT_TOL)


def compare_ca(lin_out, ca):
    res = residuals(lin_out, [], ca, False, True, [0, 1, 2])
    return summarise(res, CA_TOL)


# --------------------------------------------------------------------------- self test

def synth_lens_output(w, h, distortion, ca, use_distortion, use_ca, line_sigma=1.6):
    """What the lens module would output for vertical lines (source positions 49.5 + 100 k) under the
    darktable model of lens_model: output(p) = lines(source(p)), per colour plane, zoom included."""
    n = len(distortion) if len(distortion) else len(ca) // 2
    kx = lm.knots_x(n)
    ones = np.ones(n)
    f = np.asarray(distortion, dtype=np.float64) * 2.0 ** -14 + 1.0 if use_distortion else ones
    kr = np.asarray(ca[:n], dtype=np.float64) * 2.0 ** -21 + 1.0 if use_ca and len(ca) else ones
    kb = np.asarray(ca[n:], dtype=np.float64) * 2.0 ** -21 + 1.0 if use_ca and len(ca) else ones
    zoom = lm.darktable_zoom(w, h, distortion, ca, use_distortion, use_ca)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, hd = w / 2.0, h / 2.0, math.hypot(w / 2.0, h / 2.0)
    r = np.hypot(xx - cx, yy - cy) / hd
    out = np.empty((h, w, 3))
    for c, k in enumerate((kr, ones, kb)):
        cor = np.interp(r / zoom, kx, f * k) / zoom
        sx = cx + (xx - cx) * cor
        u = np.mod(sx - ro.LINE_X0, ro.LINE_PITCH)
        d = np.minimum(u, ro.LINE_PITCH - u)
        out[..., c] = 0.5 * (1.0 - 0.9 * np.exp(-0.5 * (d / line_sigma) ** 2))
    return out


def selftest():
    ok = True
    w, h = 900, 600
    knots = [0, 300, 1000, 2000, 3200, 4600, 6200, 7800]
    # Vignetting: the "darktable" gain equals the model -> pass; 1% off -> fail.
    r = radius_map(h, w)
    flat = np.full((h, w, 3), 0.4)
    gain = lm.vignette_gain(knots, r)[..., None]
    for label, factor, want in (("exact", 1.0, True), ("1% off", 1.01, False)):
        res = compare_vignette(flat, flat * gain * (1 + (factor - 1) * r[..., None]), knots)
        print(f"vignette {label}: max deviation {res['max_rel_dev']:.4%} at r={res['worst_r']:.2f} -> {'pass' if res['passed'] else 'fail'}")
        ok &= res["passed"] == want
    # Distortion and CA: a "darktable" that follows the model passes, one with a different table fails.
    dist = [int(1200 * ((i + 0.5) / 8) ** 2) for i in range(9)]
    ca = [int(2400 * ((i + 0.5) / 8) ** 2) for i in range(9)] + [-int(2400 * ((i + 0.5) / 8) ** 2) for i in range(9)]
    for label, table, want in (("exact", dist, True), ("50% stronger", [int(d * 1.5) for d in dist], False)):
        out = synth_lens_output(w, h, table, [], True, False)
        res = compare_distortion(out, dist, [])
        print(f"distortion {label}: {res.get('samples')} lines, corner {res.get('max_corner_px', float('nan')):.3f} px, "
              f"max {max(res.get('max_px', [float('nan')])):.3f} px -> {'pass' if res['passed'] else 'fail'}")
        ok &= res["passed"] == want
    for label, table, want in (("exact", ca, True), ("50% stronger", [int(c * 1.5) for c in ca], False)):
        out = synth_lens_output(w, h, [], table, False, True)
        res = compare_ca(out, ca)
        print(f"ca {label}: {res.get('samples')} lines, max {max(res.get('max_px', [float('nan')])):.3f} px -> {'pass' if res['passed'] else 'fail'}")
        ok &= res["passed"] == want
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("vignette", "distortion", "ca"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True, help="darktable_oracle.sh output directory of the file")
        p.add_argument("--tags", required=True, help="JSON with vignetting / distortion / ca knot lists")
        if name == "distortion":
            p.add_argument("--ignore-ca", action="store_true", help="the render has no CA correction")
            p.add_argument("--horizontal", action="store_true", help="the file has horizontal lines")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    tags = json.load(open(a.tags))
    lin_in, lin_out = lens_io.load_pair(a.dir)
    if a.cmd == "vignette":
        res = compare_vignette(lin_in, lin_out, tags["vignetting"])
        print(f"vignetting: max |darktable / ours - 1| = {res['max_rel_dev']:.4%} at r = {res['worst_r']:.2f} "
              f"(gate {GAIN_TOL:.1%}) -> {'pass' if res['passed'] else 'FAIL'}")
    elif a.cmd == "distortion":
        res = compare_distortion(lin_out, tags.get("distortion", []), [] if a.ignore_ca else tags.get("ca", []), a.horizontal)
        print(json.dumps(res, indent=1))
        print(f"distortion ({'horizontal' if a.horizontal else 'vertical'} lines): displacement gate {DISPLACEMENT_TOL} px -> {'pass' if res['passed'] else 'FAIL'}")
    else:
        res = compare_ca(lin_out, tags["ca"])
        print(json.dumps(res, indent=1))
        print(f"lateral CA: line deviation gate {CA_TOL} px -> {'pass' if res['passed'] else 'FAIL'}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
