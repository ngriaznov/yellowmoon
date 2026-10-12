"""Reference implementation of the profile runtime (what ``ym-color`` / ``ym-profiles`` must do).

Pure numpy, working on a loaded ``profile.json``. It is the specification the Rust port is
tested against (``crates/ym-color/tests/data/spectral_reference.json`` is generated with it) and
documents every interpolation rule. Nothing here needs CIE data.

Conventions
-----------
* Grid tables ``{"start_k", "step_k", "values"}``; entry ``i`` belongs to ``T_i = start_k + i * step_k``.
  A query ``T`` outside the table is clamped to its ends.
* **Interpolation variable: mired.** For ``T_i <= T <= T_(i+1)`` the weight is
  ``a = (1/T - 1/T_i) / (1/T_(i+1) - 1/T_i)`` and:

  - ``w_A`` (alexa_white blackbody / daylight): **geometric**, ``w = w_i^(1-a) * w_(i+1)^a`` per
    channel (that is linear in mired on ln w); green stays exactly 1.
  - camera locus log-chroma and CIE xy: ``(1-a) * v_i + a * v_(i+1)`` (linear in mired).
  - class lists (sensor-map matrices, fluorescent / LED ``w_A``): linear in mired between the
    two bracketing entries, clamped at the ends; ``w`` and ``M`` are interpolated linearly.

* **Which table** for a CCT: blackbody below 4000 K, CIE daylight from 4000 K (the rawtoaces rule).
* **RGB -> (CCT, Duv):** :func:`rgb_to_cct_duv`. **(CCT, Duv) -> RGB:** :func:`cct_duv_to_rgb`.
  Duv is signed, positive toward magenta, in CIE 1960 uv units (the ``duv_scale`` table, looked up
  at the foot point's CCT, converts the log-chroma distance). That is the opposite sign of CIE's
  own Duv convention (positive above the locus, toward green).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KELVIN_SPLIT = 4000.0  # blackbody below, CIE daylight from here (docs/02)


@dataclass(frozen=True)
class Grid:
    """A uniform temperature grid table."""

    start_k: float
    step_k: float
    values: np.ndarray  # (n, k)

    @classmethod
    def from_json(cls, d: dict) -> "Grid":
        return cls(float(d["start_k"]), float(d["step_k"]), np.asarray(d["values"], dtype=float))

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def temps(self) -> np.ndarray:
        return self.start_k + self.step_k * np.arange(self.n)

    @property
    def end_k(self) -> float:
        return self.start_k + self.step_k * (self.n - 1)

    def locate(self, t_k: float) -> tuple[int, float]:
        """(index i, mired weight a) with ``T_i <= T <= T_(i+1)``; ``t_k`` is clamped to the table."""
        t = min(max(float(t_k), self.start_k), self.end_k)
        i = int(np.floor((t - self.start_k) / self.step_k))
        i = min(max(i, 0), self.n - 2)
        t0 = self.start_k + i * self.step_k
        t1 = t0 + self.step_k
        a = (1.0 / t - 1.0 / t0) / (1.0 / t1 - 1.0 / t0)
        return i, min(max(a, 0.0), 1.0)

    def lerp_mired(self, t_k: float) -> np.ndarray:
        i, a = self.locate(t_k)
        return (1 - a) * self.values[i] + a * self.values[i + 1]

    def geom_mired(self, t_k: float) -> np.ndarray:
        i, a = self.locate(t_k)
        return self.values[i] ** (1 - a) * self.values[i + 1] ** a


@dataclass(frozen=True)
class Profile:
    """The parts of ``profile.json`` the runtime uses."""

    raw: dict
    w_a_blackbody: Grid
    w_a_daylight: Grid
    chroma_blackbody: Grid
    chroma_daylight: Grid
    xy_blackbody: Grid
    xy_daylight: Grid
    duv_scale_blackbody: Grid
    duv_scale_daylight: Grid

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        aw, lo = d["alexa_white"], d["locus"]
        return cls(
            raw=d,
            w_a_blackbody=Grid.from_json(aw["blackbody"]),
            w_a_daylight=Grid.from_json(aw["daylight"]),
            chroma_blackbody=Grid.from_json(lo["blackbody"]),
            chroma_daylight=Grid.from_json(lo["daylight"]),
            xy_blackbody=Grid.from_json(lo["xy"]["blackbody"]),
            xy_daylight=Grid.from_json(lo["xy"]["daylight"]),
            duv_scale_blackbody=Grid.from_json(lo["duv_scale"]["blackbody"]),
            duv_scale_daylight=Grid.from_json(lo["duv_scale"]["daylight"]),
        )


# ----------------------------------------------------------------------------
# w_A
# ----------------------------------------------------------------------------
def w_a(profile: Profile, t_k: float, kind: str | None = None) -> np.ndarray:
    """``w_A`` ([r, 1, b]) of the scene light at ``t_k``: blackbody below 4000 K, daylight from 4000 K.

    ``kind`` forces "blackbody" or "daylight" (used for the table-edge tests).
    """
    kind = kind or ("blackbody" if t_k < KELVIN_SPLIT else "daylight")
    grid = profile.w_a_blackbody if kind == "blackbody" else profile.w_a_daylight
    return grid.geom_mired(t_k)


def interp_class_list(entries: list[dict], mired: float, key: str) -> np.ndarray:
    """Linear in mired between the bracketing entries of a class list, clamped at the ends."""
    ms = [e["mired"] for e in entries]
    if mired <= ms[0]:
        return np.asarray(entries[0][key], dtype=float)
    if mired >= ms[-1]:
        return np.asarray(entries[-1][key], dtype=float)
    i = int(np.searchsorted(ms, mired, side="right"))
    a = (mired - ms[i - 1]) / (ms[i] - ms[i - 1])
    return (1 - a) * np.asarray(entries[i - 1][key], dtype=float) + a * np.asarray(entries[i][key], dtype=float)


def sensor_map(profile: Profile, probs: dict[str, float], mired: float) -> np.ndarray:
    """``M_map = sum_k p_k M_k(mired)`` from the profile's class tables."""
    classes = profile.raw["sensor_map"]["classes"]
    return sum(p * interp_class_list(classes[k], mired, "m") for k, p in probs.items() if p != 0.0)


# ----------------------------------------------------------------------------
# Locus, CCT and Duv
# ----------------------------------------------------------------------------
def locus_chroma(profile: Profile, t_k: float) -> np.ndarray:
    """Camera log-chroma ``(ln R/G, ln B/G)`` of the scene light at ``t_k`` (linear in mired)."""
    grid = profile.chroma_blackbody if t_k < KELVIN_SPLIT else profile.chroma_daylight
    return grid.lerp_mired(t_k)


def _nearest_on_branch(grid: Grid, p: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Nearest point of one branch's polyline to ``p``.

    Returns (distance, mired of the foot point, signed distance toward magenta, unit normal).
    Segments are straight chords between consecutive table entries; the first segment wins ties.
    """
    P = grid.values
    e = P[1:] - P[:-1]
    ee = np.einsum("ij,ij->i", e, e)
    ok = ee > 0.0  # the staircase of the D series can repeat a table row exactly: skip such chords
    t = np.zeros(len(e))
    t[ok] = np.clip(np.einsum("ij,ij->i", p - P[:-1], e)[ok] / ee[ok], 0.0, 1.0)
    Q = P[:-1] + t[:, None] * e
    dist = np.where(ok, np.linalg.norm(p - Q, axis=1), np.inf)
    i = int(np.argmin(dist))  # first minimum
    mired0 = 1e6 / (grid.start_k + i * grid.step_k)
    mired1 = 1e6 / (grid.start_k + (i + 1) * grid.step_k)
    mired = mired0 + t[i] * (mired1 - mired0)
    n_hat = np.array([e[i, 1], -e[i, 0]]) / np.sqrt(ee[i])  # tangent rotated by -90 degrees: toward magenta
    return float(dist[i]), float(mired), float(np.dot(p - Q[i], n_hat)), n_hat


def duv_scale_at(profile: Profile, t_k: float) -> float:
    """CIE-uv-per-log-chroma scale at ``t_k`` (blackbody table below 4000 K, daylight from 4000 K)."""
    grid = profile.duv_scale_blackbody if t_k < KELVIN_SPLIT else profile.duv_scale_daylight
    return float(grid.lerp_mired(t_k))


def rgb_to_cct_duv(profile: Profile, rgb) -> tuple[float, float]:
    """Camera RGB of the light (any scale) -> (CCT in K, Duv, positive toward magenta).

    1. ``p = (ln R/G, ln B/G)``.
    2. For each branch (blackbody 1500-4000 K, daylight 4000-25000 K) find the nearest point on
       the polyline of table entries (projection on each chord, clamped to the chord; chords of
       zero length are skipped).
    3. Take the branch with the smaller distance (daylight on a tie). The foot point's position
       along its chord gives the mired coordinate, ``CCT = 1e6 / mired`` (linear in mired between
       the two table entries, so it inverts :func:`locus_chroma` exactly on the chord).
    4. ``Duv = signed_distance * duv_scale(CCT)``; the sign is positive toward magenta: the chord's
       tangent (increasing temperature) rotated by -90 degrees, ``n = (t_y, -t_x)``.
    """
    rgb = np.asarray(rgb, dtype=float)
    p = np.log(np.array([rgb[0] / rgb[1], rgb[2] / rgb[1]]))
    d_bb, m_bb, s_bb, _ = _nearest_on_branch(profile.chroma_blackbody, p)
    d_dy, m_dy, s_dy, _ = _nearest_on_branch(profile.chroma_daylight, p)
    if d_dy <= d_bb:
        mired, signed = m_dy, s_dy
    else:
        mired, signed = m_bb, s_bb
    cct = 1e6 / mired
    return cct, signed * duv_scale_at(profile, cct)


def cct_duv_to_rgb(profile: Profile, cct_k: float, duv: float) -> np.ndarray:
    """(CCT, Duv positive toward magenta) -> camera RGB with green = 1. Inverse of :func:`rgb_to_cct_duv`
    for points whose foot of perpendicular lies inside the chord (small Duv)."""
    grid = profile.chroma_blackbody if cct_k < KELVIN_SPLIT else profile.chroma_daylight
    i, a = grid.locate(cct_k)
    P0, P1 = grid.values[i], grid.values[i + 1]
    e = P1 - P0
    q = (1 - a) * P0 + a * P1
    n_hat = np.array([e[1], -e[0]]) / np.linalg.norm(e)
    p = q + (duv / duv_scale_at(profile, cct_k)) * n_hat
    return np.array([np.exp(p[0]), 1.0, np.exp(p[1])])


def cie_uv_of_locus(profile: Profile, t_k: float) -> np.ndarray:
    """CIE 1960 (u, v) of the scene light at ``t_k`` from the stored xy table (linear in mired)."""
    grid = profile.xy_blackbody if t_k < KELVIN_SPLIT else profile.xy_daylight
    x, y = grid.lerp_mired(t_k)
    d = -2 * x + 12 * y + 3
    return np.array([4 * x / d, 6 * y / d])
