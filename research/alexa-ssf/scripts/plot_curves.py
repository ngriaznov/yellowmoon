"""Plot the ALEXA estimate (with ensemble band) against Sony ILCE-7RM4 and ARRI D21."""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssf_core as sc  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3

rows = list(csv.DictReader(open(os.path.join(sc.ROOT, "out", "alexa_alev3_estimate_380_780_5.csv"))))
wl = np.array([float(r["wavelength_nm"]) for r in rows])
est = {c: np.array([float(r[c]) for r in rows]) for c in "RGB"}
lo = {c: np.array([float(r[f"{c}_min"]) for r in rows]) for c in "RGB"}
hi = {c: np.array([float(r[f"{c}_max"]) for r in rows]) for c in "RGB"}
cams = sc.load_cameras()
sony, d21 = cams["Sony ILCE-7RM4"], cams["ARRI D21"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
names = {"R": "Red channel", "G": "Green channel", "B": "Blue channel"}
for i, (ax, c) in enumerate(zip(axes, "RGB")):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.fill_between(wl, lo[c], hi[c], color=S1, alpha=0.18, linewidth=0,
                    label="Estimate range (24 variants)")
    ax.plot(wl, d21[:, i], color=S3, lw=2, ls=(0, (1.5, 1.5)), label="ARRI D-21 (measured, 2008 sensor)")
    ax.plot(wl, sony[:, i], color=S2, lw=2, ls=(0, (5, 2)), label="Sony A7R IV (measured)")
    ax.plot(wl, est[c], color=S1, lw=2, label="ALEXA ALEV III (estimated)")
    ax.set_title(names[c], color=INK, fontsize=11, loc="left")
    ax.set_xlim(380, 780)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.axvspan(660, 780, color=INK2, alpha=0.05, linewidth=0)
axes[0].set_ylabel("Relative sensitivity (peak = 1)")
axes[0].text(668, 0.93, "weakly\nconstrained", color=INK2, fontsize=8.5, va="top")
handles, labels = axes[0].get_legend_handles_labels()
order = [3, 0, 2, 1]
fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="upper center",
           ncol=4, frameon=False, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.985))
fig.suptitle("ALEXA spectral sensitivity estimated from ARRI's published colour matrices",
             color=INK, fontsize=12.5, x=0.01, ha="left", y=1.04)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(sc.ROOT, "out", "alexa_alev3_estimate.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
