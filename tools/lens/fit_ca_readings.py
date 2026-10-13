#!/usr/bin/env python3
"""Fits readings of the Sony lateral-CA table (0x7035) to the shifts measured on real mosaics.

Input: `testdata/lens-eval/raw_ca_<stem>.json` written by `real_eval.py raw-ca` (one signed radial
shift of red and of blue against green per block of the mosaic, with its standard error).  Local
evaluation only; see `docs/05-raw-processing.md` section R9, "Real-file check".

    python3 -I tools/lens/fit_ca_readings.py [--dist-alpha A] [stem ...]

For every file and every reading it prints the information-weighted rms residual (px) of
(measured - predicted) over red and blue blocks, the best amplitude `alpha` of the table, and the
best (alpha, knot-radius scale kappa).  A reading explains a body when its residual is close to the
measurement noise (0.03 to 0.1 px) with `alpha = 1`.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_eval as re_  # noqa: E402

OUT = re_.OUT
SIGMA0 = 0.05  # px, floor on the per-block uncertainty


@dataclass(frozen=True)
class Reading:
    table: str = "raw"  # "raw" (the raw sub-IFD, 128-step CA) or "mn" (the maker note's copy)
    offset: float = 0.5  # knot i at (i + offset) / denom
    denom: str = "n-1"  # "n-1" or "n"
    k_at_source: bool = False  # K taken at the source radius instead of the output radius
    kappa: float = 1.0  # knot radii scaled by kappa (half-diagonal normalisation)
    dist_alpha: float = 1.0  # scale of the distortion table used for r_out <-> r_src
    swap: bool = False  # red uses the blue table and vice versa
    sign: float = 1.0
    ca_div_n: bool = False  # for tests of the MN table: first n knots of each colour

    def label(self) -> str:
        bits = [self.table, f"off{self.offset:g}", self.denom, "Ksrc" if self.k_at_source else "Kout"]
        if self.kappa != 1.0:
            bits.append(f"kappa{self.kappa:g}")
        if self.swap:
            bits.append("swap")
        if self.sign != 1.0:
            bits.append("flip")
        return " ".join(bits)


def tables_of(rec: dict, reading: Reading):
    t = rec["tables"]
    n_raw, ca_raw = re_.split_stored(t["raw"]["ca"])
    n_d, dist_raw = re_.split_stored(t["raw"]["distortion"])
    if reading.table == "raw":
        return ca_raw, dist_raw
    n = n_raw // 2  # knots per colour of the recorded image
    mn_ca = t["mn"]["ca"]
    if not mn_ca:
        return None, None
    per = len(mn_ca) // 2
    ca = list(mn_ca[:n]) + list(mn_ca[per : per + n])
    mn_d = t["mn"]["distortion"]
    dist = list(mn_d[:n_d]) if mn_d else dist_raw
    return ca, dist


def predict(rec: dict, chan: str, xy: np.ndarray, reading: Reading) -> np.ndarray | None:
    ca, dist = tables_of(rec, reading)
    if not ca:
        return None
    w, h = rec["frame"]
    hd = rec["hd"]
    n = len(ca) // 2
    denom = (n - 1) if reading.denom == "n-1" else n
    cx, cy = w / 2.0, h / 2.0
    r_px = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    rho = r_px / hd
    kw = dict(offset=reading.offset, denom=denom, kscale=reading.kappa)
    if len(dist):
        grid = np.linspace(0.0, 1.5, 3001)
        f = re_.distortion_scale(dist, grid, reading.dist_alpha, **kw)
        r_out = np.interp(rho, grid * f, grid)
    else:
        r_out = rho
    at = rho if reading.k_at_source else r_out
    kr, kb = re_.ca_k(ca, at, 1.0, **kw)
    if reading.swap:
        kr, kb = kb, kr
    k = kr if chan == "red" else kb
    return reading.sign * rho * hd * (k - 1.0)


def gather(rec: dict, chan: str):
    b = np.array(rec["blocks"][chan], dtype=np.float64)
    if not len(b):
        return None
    return b[:, :2], b[:, 2], np.sqrt(b[:, 3] ** 2 + SIGMA0**2)


def score(rec: dict, reading: Reading, chans=("red", "blue")):
    """(rms with alpha = 1, best alpha, rms at best alpha) over the given colours, weighted."""
    num = den = sse1 = wsum = 0.0
    sse_terms = []
    for chan in chans:
        g = gather(rec, chan)
        if g is None:
            continue
        xy, m, se = g
        p = predict(rec, chan, xy, reading)
        if p is None:
            return None
        wgt = 1.0 / se**2
        sse1 += float((wgt * (m - p) ** 2).sum())
        wsum += float(wgt.sum())
        num += float((wgt * m * p).sum())
        den += float((wgt * p * p).sum())
        sse_terms.append((wgt, m, p))
    if wsum == 0 or den == 0:
        return None
    alpha = num / den
    sse_a = sum(float((w_ * (m - alpha * p) ** 2).sum()) for w_, m, p in sse_terms)
    return math.sqrt(sse1 / wsum), alpha, math.sqrt(sse_a / wsum)


def best_kappa(rec: dict, reading: Reading, kappas=np.linspace(0.7, 1.6, 37)):
    best = None
    for k in kappas:
        sc = score(rec, replace(reading, kappa=float(k)))
        if sc is None:
            return None
        if best is None or sc[2] < best[3]:
            best = (float(k), sc[1], sc[0], sc[2])
    return best


def main():
    args = sys.argv[1:]
    dist_alpha = 1.0
    if "--dist-alpha" in args:
        i = args.index("--dist-alpha")
        dist_alpha = float(args[i + 1])
        del args[i : i + 2]
    stems = args or sorted(p.stem[len("raw_ca_") :] for p in OUT.glob("raw_ca_*.json"))
    base = Reading(dist_alpha=dist_alpha)
    readings = [
        base,
        replace(base, offset=0.0),
        replace(base, denom="n"),
        replace(base, k_at_source=True),
        replace(base, swap=True),
        replace(base, sign=-1.0),
        replace(base, table="mn"),
    ]
    for stem in stems:
        path = OUT / f"raw_ca_{stem}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        nb = {c: len(rec["blocks"][c]) for c in ("red", "blue")}
        print(f"== {stem}  frame {rec['frame']}  blocks {nb}")
        for r in readings:
            sc = score(rec, r)
            if sc is None:
                continue
            per = {c: score(rec, r, (c,)) for c in ("red", "blue")}
            print(
                f"  {r.label():34s} rms(a=1) {sc[0]:.3f}  best a {sc[1]:+.2f} -> {sc[2]:.3f}"
                f"   red a {per['red'][1]:+.2f} blue a {per['blue'][1]:+.2f}"
            )
        bk = best_kappa(rec, base)
        if bk:
            print(f"  best kappa {bk[0]:.2f} (alpha {bk[1]:+.2f}) rms {bk[2]:.3f} -> {bk[3]:.3f}")
        bk = best_kappa(rec, replace(base, table="mn"))
        if bk:
            print(f"  MN table: best kappa {bk[0]:.2f} (alpha {bk[1]:+.2f}) rms {bk[2]:.3f} -> {bk[3]:.3f}")


if __name__ == "__main__":
    main()
