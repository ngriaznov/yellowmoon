"""Log-chroma histograms of a preview: the input of the FFCC-style filter.

Reference implementation of ``docs/02`` section Illuminant estimate, step 1. ``ym_color::illum`` is a port of
this file; the two agree to 1e-9 on the committed test vectors (``crates/ym-color/tests/data/illum/``).

Definitions (``img`` is the black-subtracted camera RGB of the 1/8-resolution preview, white level 1.0, before
any white balance; ``mask`` is non-zero where the pixel is clipped):

* Channel 0 (*intensity*): every unmasked pixel whose three channels are all above ``min_intensity`` has the
  log-chroma ``u = ln(G/R)``, ``v = ln(G/B)`` and the weight ``sqrt(Y)``, ``Y = 0.25 R + 0.5 G + 0.25 B``.
* Channel 1 (*edge*, optional): the masked local absolute deviation of each colour channel (the mean absolute
  difference to the unmasked pixels among the 8 neighbours; none outside the image) is an image of its own;
  its log-chroma and ``sqrt(Y)`` weight are histogrammed like channel 0 (all three edge values above
  ``min_edge``). Pixels with a non-finite channel are never counted.
* A pixel is splatted bilinearly onto the ``n_bins x n_bins`` grid whose bin ``(i, j)`` sits at
  ``u = lo_u + i * bin_size``, ``v = lo_v + j * bin_size``. A pixel with ``0 <= (u - lo_u) / bin_size < n_bins - 1``
  (the same for ``v``) is counted, anything else is dropped (and reported in ``in_range``).
* Each channel is divided by the sum of its weights, so it sums to 1 (an empty channel stays 0).
* Large previews are subsampled: only the pixels ``(y, x)`` with ``y`` and ``x`` multiples of
  ``stride = max(1, isqrt(width * height // max_samples))`` are histogrammed (edges are still taken between neighbouring
  pixels of the full preview). With ``max_samples = 16384`` a 1080 x 720 preview (the 1/8-resolution preview of a 50 MP frame)
  uses every 6th pixel in both directions, 21 600 pixels; the 96 x 64 training scenes use all of theirs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt

import numpy as np

Y_WEIGHTS = (0.25, 0.5, 0.25)
CHANNELS = ("intensity", "edge")


@dataclass(frozen=True)
class HistSpec:
    """Geometry and thresholds of the histogram of one model."""

    n_bins: int = 64
    bin_size: float = 0.0625
    lo_u: float = -1.5
    lo_v: float = -1.1
    min_intensity: float = 1.0 / 1024.0
    min_edge: float = 1.0 / 4096.0
    channels: tuple[str, ...] = ("intensity", "edge")
    max_samples: int = 16384

    def to_json(self) -> dict:
        return {"n_bins": self.n_bins, "bin_size": self.bin_size, "lo_u": self.lo_u, "lo_v": self.lo_v,
                "min_intensity": self.min_intensity, "min_edge": self.min_edge, "channels": list(self.channels),
                "max_samples": self.max_samples,
                "weight": "sqrt_y", "y_weights": list(Y_WEIGHTS)}

    @staticmethod
    def from_json(d: dict) -> "HistSpec":
        return HistSpec(int(d["n_bins"]), float(d["bin_size"]), float(d["lo_u"]), float(d["lo_v"]),
                        float(d["min_intensity"]), float(d["min_edge"]), tuple(d["channels"]),
                        int(d.get("max_samples", 16384)))

    def uv_of_bin(self, i, j):
        return self.lo_u + np.asarray(i) * self.bin_size, self.lo_v + np.asarray(j) * self.bin_size


@dataclass
class HistStats:
    """Side outputs of the histogram step that the class head and the fallbacks read."""

    valid_fraction: np.ndarray          # per channel: weight-carrying pixels / all pixels
    in_range: np.ndarray                # per channel: counted weight / weight of valid pixels (1 when none)
    mass: np.ndarray                    # per channel: counted weight before normalisation
    n_pixels: int = 0
    extra: dict = field(default_factory=dict)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return Y_WEIGHTS[0] * rgb[..., 0] + Y_WEIGHTS[1] * rgb[..., 1] + Y_WEIGHTS[2] * rgb[..., 2]


def edge_image(img: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masked local absolute deviation: ``(edge (H, W, 3), has_edge (H, W) bool)``.

    ``valid`` is True for unmasked pixels. For a valid pixel the edge value of a channel is the mean of
    ``|I(q) - I(p)|`` over its valid 8-neighbours ``q`` inside the image; a pixel with no valid neighbour (or a
    masked pixel) has no edge.
    """
    h, w, _ = img.shape
    pv = np.pad(valid, 1, constant_values=False)
    pi = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="constant")
    acc = np.zeros_like(img, dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nv = pv[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            ni = pi[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            acc += np.where(nv[..., None], np.abs(ni - img), 0.0)
            cnt += nv
    has = valid & (cnt > 0)
    edge = np.where(has[..., None], acc / np.maximum(cnt, 1.0)[..., None], 0.0)
    return edge, has


def _channel_hist(rgb: np.ndarray, ok: np.ndarray, floor: float, spec: HistSpec) -> tuple[np.ndarray, float, float, float]:
    """Histogram of one channel image: (hist (n, n), valid weight pixels, counted weight, valid weight)."""
    n = spec.n_bins
    sel = ok & np.isfinite(rgb).all(axis=-1) & (rgb > floor).all(axis=-1)
    px = rgb[sel]
    hist = np.zeros((n, n))
    if px.shape[0] == 0:
        return hist, 0.0, 0.0, 0.0
    u = np.log(px[:, 1] / px[:, 0])
    v = np.log(px[:, 1] / px[:, 2])
    wgt = np.sqrt(luminance(px))
    fu = (u - spec.lo_u) / spec.bin_size
    fv = (v - spec.lo_v) / spec.bin_size
    inside = (fu >= 0.0) & (fu < n - 1) & (fv >= 0.0) & (fv < n - 1)
    total = float(wgt.sum())
    fu, fv, wgt = fu[inside], fv[inside], wgt[inside]
    iu = np.floor(fu).astype(np.int64)
    iv = np.floor(fv).astype(np.int64)
    tu, tv = fu - iu, fv - iv
    flat = hist.reshape(-1)
    for di, dj, wt in ((0, 0, (1 - tu) * (1 - tv)), (1, 0, tu * (1 - tv)), (0, 1, (1 - tu) * tv), (1, 1, tu * tv)):
        np.add.at(flat, (iu + di) * n + (iv + dj), wgt * wt)
    return hist, float(sel.sum()), float(wgt.sum()), total


def sample_stride(height: int, width: int, max_samples: int) -> int:
    """Pixel step of the subsampling grid (see the module docstring)."""
    return max(1, isqrt((height * width) // max_samples))


def histograms(img: np.ndarray, mask: np.ndarray | None, spec: HistSpec) -> tuple[np.ndarray, HistStats]:
    """The normalised histograms ``(n_channels, n, n)`` of a preview and their statistics."""
    img = np.asarray(img, dtype=np.float64)
    h, w, _ = img.shape
    valid = np.ones((h, w), dtype=bool) if mask is None else (np.asarray(mask) == 0)
    stride = sample_stride(h, w, spec.max_samples)
    out = np.zeros((len(spec.channels), spec.n_bins, spec.n_bins))
    vf = np.zeros(len(spec.channels))
    inr = np.ones(len(spec.channels))
    mass = np.zeros(len(spec.channels))
    n_sampled = len(range(0, h, stride)) * len(range(0, w, stride))
    for c, name in enumerate(spec.channels):
        if name == "intensity":
            hist, nsel, counted, total = _channel_hist(img[::stride, ::stride], valid[::stride, ::stride], spec.min_intensity, spec)
        elif name == "edge":
            edge, has = edge_image(img, valid)
            hist, nsel, counted, total = _channel_hist(edge[::stride, ::stride], has[::stride, ::stride], spec.min_edge, spec)
        else:
            raise ValueError(f"unknown histogram channel {name!r}")
        vf[c] = nsel / n_sampled if n_sampled else 0.0
        inr[c] = counted / total if total > 0 else 1.0
        mass[c] = counted
        out[c] = hist / counted if counted > 0 else hist
    return out, HistStats(vf, inr, mass, h * w)
