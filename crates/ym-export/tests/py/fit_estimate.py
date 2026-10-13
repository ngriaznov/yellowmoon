"""Fits the size-estimate model of `crates/ym-export/src/estimate.rs` from the CSVs of `tests/calibrate_estimate.rs`.

    python3 -I crates/ym-export/tests/py/fit_estimate.py MEASURE.csv DETAIL.csv

`MEASURE.csv` is written by the ignored test `measure_renders` (one row per picture, export size and format variant),
`DETAIL.csv` by `measure_library_detail` (the detail statistic of each picture). Both read the 16-bit PNG renders of
`$YM_ESTIMATE_RENDERS`, which `ym render --format png --bits 16` writes. See `docs/09-export.md`, Size estimate.

For every format variant it fits   ln y = a + b*x + c*y + d*x*y   by least squares, with x = ln(g / G0), g the detail
statistic, y = ln(output megapixels / 10) and y the JPEG bits per pixel or the TIFF / PNG size over the raw size. It prints
the error of the fit (median of estimate / actual and the 90th percentile of |estimate / actual - 1|, over all pictures and
over the real ones, as "median p90") and the same with each picture left out in turn, then the Rust constants to paste into `estimate.rs`.
Pictures whose name starts with `chart` are the synthetic scenes; the others are real renders. G0 is the geometric mean of
the statistic over the real renders (`TYPICAL_GRADIENT`). Needs numpy.
"""

import csv
import math
import sys

import numpy as np

JPEG_QUALITIES = (50, 60, 75, 80, 85, 89, 90, 92, 95, 100)
RATIO_VARIANTS = [(fmt, bits) for fmt in ("tiff_lzw", "tiff_zip", "png") for bits in (8, 16)]


def load(measure_path, detail_path):
    detail = {r["image"]: float(r["detail_u8"]) for r in csv.DictReader(open(detail_path))}
    records = {}
    for r in csv.DictReader(open(measure_path)):
        key = (r["image"], int(r["out_width"]), int(r["out_height"]), r["format"], int(r["param"]))
        records[key] = dict(
            image=r["image"],
            ow=int(r["out_width"]),
            oh=int(r["out_height"]),
            bytes=int(r["bytes"]),
            g=detail[r["image"]],
        )
    return detail, records


def main():
    detail, records = load(sys.argv[1], sys.argv[2])
    images = sorted(detail)
    real = [i for i in images if not i.startswith("chart")]
    g0 = math.exp(np.mean([math.log(detail[i]) for i in real]))
    print(f"{len(images)} pictures ({len(real)} real), G0 = {g0:.5f}")

    def features(v):
        x = math.log(v["g"] / g0)
        y = math.log(v["ow"] * v["oh"] / 1e6 / 10)
        return [1.0, x, y, x * y]

    def target(fmt, param, v):
        pixels = v["ow"] * v["oh"]
        if fmt == "jpeg":
            return math.log(v["bytes"] * 8 / pixels)
        return math.log(v["bytes"] / (pixels * 3 * param / 8))

    def rows(fmt, param):
        return [(k[0], v) for k, v in records.items() if k[3] == fmt and k[4] == param]

    def solve(data, fmt, param):
        x = np.array([features(v) for _, v in data])
        y = np.array([target(fmt, param, v) for _, v in data])
        return np.linalg.lstsq(x, y, rcond=None)[0]

    def ratio(beta, fmt, param, v):
        return math.exp(float(np.array(features(v)) @ beta) - target(fmt, param, v))

    def summary(values):
        values = np.array(values)
        return f"{np.median(values):.2f} {np.quantile(np.abs(values - 1), 0.9):.2f}"

    fits = {}
    print("format  param   in-sample (all pictures / real ones)   |   leave-one-out (all / real)")
    for fmt, param in [("jpeg", q) for q in JPEG_QUALITIES] + RATIO_VARIANTS:
        data = rows(fmt, param)
        beta = solve(data, fmt, param)
        fits[(fmt, param)] = beta
        inside = [(i, ratio(beta, fmt, param, v)) for i, v in data]
        left_out = []
        for held in images:
            b = solve([(i, v) for i, v in data if i != held], fmt, param)
            left_out += [(i, ratio(b, fmt, param, v)) for i, v in data if i == held]
        both = lambda pairs: summary([r for _, r in pairs]) + " / " + summary([r for i, r in pairs if i in real])
        print(f"{fmt:<9}{param:>5}   {both(inside)}   |   {both(left_out)}")

    def fit_text(key):
        a, b, c, d = fits[key]
        return f"LogFit::new({a:.4f}, {b:.4f}, {c:.4f}, {d:.4f})"

    print("\nTYPICAL_GRADIENT =", f"{g0:.5f}")
    print("const JPEG_FIT: [(f64, LogFit); 10] = [")
    for q in JPEG_QUALITIES:
        print(f"    ({q}.0, {fit_text(('jpeg', q))}),")
    print("];")
    print("const RATIO_FIT: RatioFits = RatioFits {")
    for name, fmt in (("lzw", "tiff_lzw"), ("zip", "tiff_zip"), ("png", "png")):
        print(f"    {name}: [{fit_text((fmt, 8))}, {fit_text((fmt, 16))}],")
    print("};")


if __name__ == "__main__":
    main()
