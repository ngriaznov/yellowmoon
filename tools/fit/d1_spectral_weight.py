#!/usr/bin/env python3
"""D1 of docs/29-raw-constants.md (task RC-11): the spectral weight of the polychromatic Airy PSF of spec 03,
derived from the published curves of the eleven Sony bodies (data/cameras/sony, rawtoaces-data, Apache-2.0)
and CIE illuminants D55 and D50 (data/film/cie), with the luma weights of the stage (0.299, 0.587, 0.114) on
balanced channels. numpy only; no image, no network.

    tools/fit/d1_spectral_weight.py [--write weights.json]

Prints the weights at a few wavelengths (today's ramp, the mean over the bodies and its spread), the L1
distance between the radial PSFs of the ramp and of the derived weights (2-D mass normalised to 1; the PSF
in units of the radial step does not depend on the f-number), and, with `--write`, the mean weight per
wavelength 461 to 619 nm as a JSON array that `capture_sharpen_ab --weights` and `fit_diffraction --weights`
read. The weights are normalised to a mean of 1 from 540 nm up."""
import glob, json, os, sys
import numpy as np
from math import pi
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
LUMA = np.array([0.299, 0.587, 0.114])
lam = np.arange(461, 620)

def read_illum(name):
    a = np.loadtxt(f"{ROOT}/data/film/cie/{name}", delimiter=",")
    return np.interp(lam, a[:, 0], a[:, 1])
ILL = {"D55": read_illum("CIE_illum_D55.csv"), "D50": read_illum("CIE_std_illum_D50.csv"), "E": np.ones(len(lam))}

def curves(path):
    d = json.load(open(path))["spectral_data"]["data"]["main"]
    ks = sorted(int(k) for k in d)
    arr = np.array([d[str(k)] for k in ks])  # (n,3)
    return np.stack([np.interp(lam, ks, arr[:, c]) for c in range(3)], axis=1)  # (159,3)

files = sorted(glob.glob(f"{ROOT}/data/cameras/sony/Sony_*_380_780_5.json"))
C = {f.split("Sony_")[1].split("_380")[0]: curves(f) for f in files}

def weight(S, E, balanced=True):
    resp = S * E[:, None]
    if balanced:
        resp = resp / resp.sum(axis=0, keepdims=True)  # each channel sees the light with equal total
    w = resp @ LUMA
    return w / w[lam >= 540].mean()

def ramp(t, top=620.0, width=80.0, flat=540):
    return np.where(lam >= flat, 1.0, (top - lam) / width)


TAU = (np.arange(400) + 0.5) * pi / 400

def j1(x):
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    for i in range(0, len(x), 64):
        xs = x[i:i + 64, None]
        out[i:i + 64] = np.cos(TAU[None, :] - xs * np.sin(TAU[None, :])).mean(axis=1)
    return out

_T = {}

def tmat(N, step, nsamp=500):
    key = (N, step)
    if key not in _T:
        k = np.arange(1, nsamp)
        T = np.zeros((len(lam), nsamp))
        for i, l in enumerate(lam):
            x = pi * k * step / (l * 1e-3 * N)
            t = j1(x) / x
            T[i, 1:] = 4 * t * t
        _T[key] = T
    return _T[key]

def radial(w, N, step, nsamp=500):
    """P[k] of the planner (spec 3.2), k = 0..nsamp-1"""
    P = w @ tmat(N, step, nsamp)
    P[0] = w.sum()
    return P

def l1(P1, P2, step):
    """L1 distance between two radial PSFs, each normalised to unit 2-D mass (0 to 2): sum_k k|p1-p2| / sum_k k p"""
    k = np.arange(len(P1)); k = np.where(k == 0, 1 / 6, k).astype(float)
    a = P1 / (k * P1).sum(); b = P2 / (k * P2).sum()
    return float((k * np.abs(a - b)).sum())


def main():
    pitch = 3.76
    step = min(0.25 * pitch, 0.25 * 8 * 0.42 / pi)
    idx = [lam.tolist().index(v) for v in (461, 490, 520, 540, 560, 580, 600, 619)]
    W = np.array([weight(S, ILL["D55"], True) for S in C.values()])
    print("wavelength nm        ", [461, 490, 520, 540, 560, 580, 600, 619])
    print("today's ramp         ", [round(float(ramp(0)[i]), 3) for i in idx])
    print("mean of 11 bodies    ", [round(float(W.mean(0)[i]), 3) for i in idx])
    print("lowest body          ", [round(float(W.min(0)[i]), 3) for i in idx])
    print("highest body         ", [round(float(W.max(0)[i]), 3) for i in idx])
    for ill in ("D55", "D50", "E"):
        for bal in (True, False):
            w = np.array([weight(S, ILL[ill], bal) for S in C.values()]).mean(0)
            d = l1(radial(ramp(0), 8, step), radial(w, 8, step), step)
            print(f"L1 distance of the radial PSFs, ramp against the mean weights ({ill}, {'balanced' if bal else 'raw'}): {d:.4f}")
    per = [l1(radial(W.mean(0), 8, step), radial(w, 8, step), step) for w in W]
    print(f"largest distance of one body's PSF to the mean's: {max(per):.4f}")
    print("flat weight against the ramp: %.4f" % l1(radial(ramp(0), 8, step), radial(np.ones(len(lam)), 8, step), step))
    if "--write" in sys.argv:
        path = sys.argv[sys.argv.index("--write") + 1]
        json.dump([float(x) for x in W.mean(0)], open(path, "w"))
        print("wrote", path)

if __name__ == "__main__":
    main()
