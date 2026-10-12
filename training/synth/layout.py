"""The scene layout: a patch chart, gradients, a neutral ramp, fine detail, a blown highlight and a dark shadow.

The layout is defined on a virtual 1248 x 832 canvas (3:2, width a multiple of 32 so cRAW can store it) and scaled
by an integer or fractional ``scale`` (1 for the default scene, 5 for the 6240 x 4160 full-size scene). Positions
and sizes scale; the *frequency content of the fine detail does not* (a zone plate reaches 0.5 cycles per pixel at
its edge at every scale), so the full-size scene is the same composition at five times the pixel count.

What is where (virtual pixels)::

    x:   24 ......... 784  812 ............ 1220
    y:   chart 19 x 10       neutral ramp (24 steps)         24 .. 108
    24   (190 patches,       gradients: neutral / hue /      120 .. 212
         40 px cells)        saturation
    424                      exposure ladder 6 x 8           224 .. 424
    448  zone plate | edge | chroma edge | gratings | lamp | coloured lamp
    600  zone plate | 2D gradient | shadow zone (-5 EV) | deep black | highlight roll-off ramps

The scene is built from a reflectance :class:`~training.synth.spectra.Bank`. A :class:`Scene` knows, for every
pixel, which bank spectra are mixed and how much light falls on it, but not the light or the camera: pass the
``(K, 3)`` table of camera (or ALEXA) responses of the bank to :meth:`Scene.render_refl` to get an image. Because
camera RGB is linear in the reflectance and in the SPD, a blend of two spectra is the blend of their responses,
and a mixture of two lights is the mixture of two renders, so one geometry serves every light, body and the
ALEXA reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from training.synth import spectra as sp

VIRTUAL_W, VIRTUAL_H = 1248, 832
#: Optical low-pass plus lens blur: a Gaussian PSF of this sigma in pixels (unscaled), applied to the scene.
PSF_SIGMA_PX = 0.45
CHART_COLS, CHART_ROWS = 19, 10
LADDER_EV = (-5, -4, -3, -2, -1, 0, 1, 2)
NEUTRAL_RAMP = (
    (0.95, 0.90, 0.80, 0.70, 0.59, 0.50, 0.40, 0.36, 0.28, 0.23, 0.18, 0.14),
    (0.11, 0.09, 0.07, 0.055, 0.045, 0.036, 0.03, 0.024, 0.019, 0.015, 0.011, 0.006),
)
WHITE_RHO = 0.90
GREY_RHO = 0.18
SHADOW_EV = -5.0
MIX_START, MIX_END = 0.30, 0.70  # the smoothstep of the mixed-light scene, as fractions of the width


def smoothstep(t: np.ndarray) -> np.ndarray:
    """0 below 0, 1 above 1, 3t^2 - 2t^3 between."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Probe:
    """A place where the ground truth is a single known colour: its reflectance mixture, its light and its rect."""

    name: str
    group: str  # chart | neutral | ladder | shadow | gradient | rolloff | highlight | black
    rect: tuple[int, int, int, int]  # x, y, w, h in pixels (the whole patch)
    interior: tuple[int, int, int, int]  # the part to average (far from edges and blur)
    mixture: list[tuple[int, float]]  # (bank index, weight): reflectance = sum w_i R_i
    m: float = 1.0  # irradiance relative to the scene's nominal light at the interior
    role: str = ""  # "grey_reference" | "white_reference" | ""
    info: dict = field(default_factory=dict)


def _inset(rect: tuple[int, int, int, int], frac: float = 0.25) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    dx, dy = max(int(round(w * frac)), 0), max(int(round(h * frac)), 0)
    return (x + dx, y + dy, max(w - 2 * dx, 1), max(h - 2 * dy, 1))


class Scene:
    """Geometry, probes and features of one scene at one scale."""

    def __init__(self, name: str, scale: float, patches: list[tuple[str, np.ndarray]], *, evaluation: bool = False):
        self.name = name
        self.scale = float(scale)
        self.evaluation = evaluation
        self.W = int(round(VIRTUAL_W * scale))
        self.H = int(round(VIRTUAL_H * scale))
        self.bank = sp.Bank()
        self.probes: list[Probe] = []
        self.features: list[dict] = []
        self._ops: list = []  # painters, in order
        self._irr_ops: list = []
        self._build(patches)

    # ------------------------------------------------------------------ geometry helpers
    def px(self, v: float) -> int:
        return int(round(v * self.scale))

    def rect(self, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        x0, y0 = self.px(x), self.px(y)
        x1, y1 = self.px(x + w), self.px(y + h)
        return (x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))

    # ------------------------------------------------------------------ build
    def _build(self, patches: list[tuple[str, np.ndarray]]) -> None:
        bank = self.bank
        surround = bank.flat(GREY_RHO)
        self._surround = surround
        chart_idx = [bank.add(n, spec, "rawtoaces training" if n.startswith("rta") else "evaluation") for n, spec in patches]
        hue = sp.pick_hue_patches(patches, 8)
        self.hue_names = [patches[i][0] for i in hue]
        hue_idx = [chart_idx[i] for i in hue]
        white = bank.flat(WHITE_RHO)
        one = bank.flat(1.0)
        black = bank.flat(0.006)

        # ---- A: the patch chart (19 x 10) ------------------------------------------------------
        for k, bi in enumerate(chart_idx[: CHART_COLS * CHART_ROWS]):
            col, row = k % CHART_COLS, k // CHART_COLS
            r = self.rect(24 + 40 * col + 2, 24 + 40 * row + 2, 36, 36)
            self._patch(f"{bank.names[bi]}", "chart", r, [(bi, 1.0)])
        self.features.append({"kind": "chart", "rect": list(self.rect(24, 24, 760, 400)),
                              "cols": CHART_COLS, "rows": min(CHART_ROWS, -(-len(chart_idx) // CHART_COLS)),
                              "patches": len(chart_idx[: CHART_COLS * CHART_ROWS]),
                              "note": "patch k at column k % 19, row k // 19; each patch 36 x 36 virtual px"})

        # ---- B1: the neutral ramp (2 x 12) -----------------------------------------------------
        for row, rhos in enumerate(NEUTRAL_RAMP):
            for col, rho in enumerate(rhos):
                bi = bank.flat(rho)
                r = self.rect(812 + 34 * col + 1, 24 + 44 * row, 32, 40)
                role = "grey_reference" if rho == GREY_RHO else "white_reference" if rho == WHITE_RHO else ""
                self._patch(f"neutral-{rho:g}", "neutral", r, [(bi, 1.0)], role=role)

        # ---- B2: gradients -----------------------------------------------------------------------
        g1 = self.rect(812, 120, 408, 28)
        self._gradient_scalar("gradient-neutral", g1, one, lo=0.005, hi=0.95)
        g2 = self.rect(812, 152, 408, 28)
        self._gradient_stops("gradient-hue", g2, hue_idx + [hue_idx[0]])
        g3 = self.rect(812, 184, 408, 28)
        self._gradient_stops("gradient-saturation", g3, [bank.flat(GREY_RHO), hue_idx[1]])

        # ---- B3: exposure ladder (6 reflectances x 8 exposures) -----------------------------------
        ladder_rows = [("grey18", bank.flat(GREY_RHO))] + [(f"hue{k}", hue_idx[k]) for k in (0, 1, 3, 5, 6)]
        for row, (lname, bi) in enumerate(ladder_rows):
            for col, ev in enumerate(LADDER_EV):
                r = self.rect(812 + 51 * col + 2, 224 + 33 * row + 2, 47, 29)
                self._patch(f"ladder-{lname}-ev{ev:+d}", "ladder", r, [(bi, 1.0)], m=2.0**ev, own_irr=True)

        # ---- C1: zone plate --------------------------------------------------------------------
        zp = self.rect(24, 456, 340, 340)
        self._zone_plate("zone-plate", zp, bank.flat(0.04), white)
        # ---- C2: slanted edges -----------------------------------------------------------------
        self._slanted_edge("edge-neutral", self.rect(388, 448, 130, 130), bank.flat(0.05), white, 5.0)
        chroma_lo, chroma_hi = self._isoluminant_pair(hue_idx[0], hue_idx[3])
        self._slanted_edge("edge-chroma", self.rect(530, 448, 130, 130), chroma_lo, chroma_hi, 5.0)
        # ---- C3: gratings: 5 periods, vertical and horizontal ------------------------------------
        for k, period in enumerate((2, 3, 4, 6, 8)):
            self._grating(f"grating-v{period}", self.rect(672 + 60 * k, 448, 56, 62), bank.flat(0.05), white, period, "v")
            self._grating(f"grating-h{period}", self.rect(672 + 60 * k, 516, 56, 62), bank.flat(0.05), white, period, "h")
        # ---- C4: blown highlights (lamps) ----------------------------------------------------------
        self._lamp("lamp-white", self.rect(988, 448, 120, 120), white, m=40.0)
        self._lamp("lamp-colour", self.rect(1116, 448, 110, 120), hue_idx[1], m=8.0)

        # ---- D1: 2D gradient plate -----------------------------------------------------------------
        self._gradient_plate("gradient-2d", self.rect(388, 600, 160, 160),
                             [hue_idx[0], hue_idx[3], hue_idx[5], white])
        # ---- D2: dark shadow zone with a 4 x 4 sub-chart ---------------------------------------------
        zone = self.rect(572, 600, 200, 200)
        self._irr_ops.append(("feather_box", zone, 2.0**SHADOW_EV, self.px(16)))
        self.features.append({"kind": "shadow-zone", "rect": list(zone), "irradiance": 2.0**SHADOW_EV,
                              "ev": SHADOW_EV, "feather_px": self.px(16)})
        shadow_spectra = [bank.flat(GREY_RHO), white] + hue_idx + [black] + [bank.flat(r) for r in (0.59, 0.36, 0.09, 0.036, 0.14)]
        for k, bi in enumerate(shadow_spectra[:16]):
            col, row = k % 4, k // 4
            r = self.rect(592 + 40 * col + 2, 620 + 40 * row + 2, 36, 36)
            self._patch(f"shadow-{bank.names[bi]}", "shadow", r, [(bi, 1.0)], m=2.0**SHADOW_EV)
        # ---- D3: deep black ------------------------------------------------------------------------
        self._patch("black", "black", self.rect(790, 600, 100, 100), [(black, 1.0)])
        # ---- D4: highlight roll-off ramps (exposure 1 -> 64, six stops) --------------------------------
        for k, (rname, bi) in enumerate((("white", white), ("colour", hue_idx[1]), ("grey18", bank.flat(GREY_RHO)))):
            self._rolloff(f"rolloff-{rname}", self.rect(900, 600 + 50 * k, 326, 40), bi, ev_lo=0.0, ev_hi=6.0)

    # ------------------------------------------------------------------ probes
    def _patch(self, name, group, rect, mixture, m=1.0, role="", own_irr=False):
        """A flat patch. ``m`` is the irradiance at its interior; ``own_irr`` makes the patch set it (the ladder),
        otherwise another operation does (the shadow zone) or ``m`` is 1."""
        probe = Probe(name, group, rect, _inset(rect), mixture, m=m, role=role)
        self.probes.append(probe)
        self._ops.append(("fill", rect, mixture))
        if own_irr:
            self._irr_ops.append(("box", rect, m))
        return probe

    def _gradient_scalar(self, name, rect, bank_idx, lo, hi):
        x, y, w, h = rect
        self._ops.append(("grad_scalar", rect, bank_idx, lo, hi))
        self.features.append({"kind": "gradient", "name": name, "rect": list(rect), "axis": "x", "scalar": True,
                              "reflectance_lo": lo, "reflectance_hi": hi, "spacing": "log"})
        for k, t in enumerate(np.linspace(0.04, 0.96, 12)):
            px = x + int(round(t * (w - 1)))
            rho = float(lo * (hi / lo) ** ((px - x + 0.5) / w))
            r = (px - 3, y, 7, h)
            self.probes.append(Probe(f"{name}-{k:02d}", "gradient", r, (px - 2, y + h // 4, 5, max(h // 2, 1)),
                                     [(bank_idx, rho)], info={"t": (px - x + 0.5) / w}))

    def _gradient_stops(self, name, rect, stops):
        x, y, w, h = rect
        self._ops.append(("grad_stops", rect, list(stops)))
        self.features.append({"kind": "gradient", "name": name, "rect": list(rect), "axis": "x", "scalar": False,
                              "stops": [self.bank.names[i] for i in stops]})
        n = len(stops) - 1
        for k, t in enumerate(np.linspace(0.03, 0.97, 16)):
            px = x + int(round(t * (w - 1)))
            u = (px - x + 0.5) / w * n
            i = min(int(u), n - 1)
            f = u - i
            r = (px - 3, y, 7, h)
            self.probes.append(Probe(f"{name}-{k:02d}", "gradient", r, (px - 2, y + h // 4, 5, max(h // 2, 1)),
                                     [(stops[i], 1.0 - f), (stops[i + 1], f)], info={"t": (px - x + 0.5) / w}))

    def _gradient_plate(self, name, rect, corners):
        x, y, w, h = rect
        tl, tr, bl, br = corners
        self._ops.append(("plate", rect, (tl, tr, bl, br)))
        self.features.append({"kind": "gradient-2d", "name": name, "rect": list(rect),
                              "corners": {"tl": self.bank.names[tl], "tr": self.bank.names[tr],
                                          "bl": self.bank.names[bl], "br": self.bank.names[br]}})
        for j, v in enumerate((0.1, 0.5, 0.9)):
            for i, u in enumerate((0.1, 0.5, 0.9)):
                px, py = x + int(round(u * (w - 1))), y + int(round(v * (h - 1)))
                uu, vv = (px - x + 0.5) / w, (py - y + 0.5) / h
                mix = [(tl, (1 - uu) * (1 - vv)), (tr, uu * (1 - vv)), (bl, (1 - uu) * vv), (br, uu * vv)]
                r = (px - 3, py - 3, 7, 7)
                self.probes.append(Probe(f"{name}-{j}{i}", "gradient", r, (px - 2, py - 2, 5, 5), mix,
                                         info={"u": uu, "v": vv}))

    def _zone_plate(self, name, rect, lo, hi):
        x, y, w, h = rect
        radius = 0.5 * min(w, h)
        self._ops.append(("zone_plate", rect, lo, hi, radius))
        self.features.append({"kind": "zone-plate", "name": name, "rect": list(rect), "centre": [x + w / 2, y + h / 2],
                              "radius_px": radius, "max_frequency_cycles_per_px": 0.5,
                              "reflectance_lo": 0.04, "reflectance_hi": WHITE_RHO,
                              "formula": "w = 0.5 + 0.5 cos(pi * 0.5 * r^2 / radius_px); frequency 0.5 * r / radius cycles/px; "
                                         "the corners alias on purpose",
                              "supersampling": 4})

    def _slanted_edge(self, name, rect, lo, hi, angle_deg):
        self._ops.append(("edge", rect, lo, hi, angle_deg))
        self.features.append({"kind": "slanted-edge", "name": name, "rect": list(rect), "angle_deg": angle_deg,
                              "dark": self.bank.names[lo], "light": self.bank.names[hi], "supersampling": 8,
                              "note": "edge through the rect centre, tilted by angle_deg from vertical; dark on the left"})

    def _grating(self, name, rect, lo, hi, period, orient):
        self._ops.append(("grating", rect, lo, hi, period, orient))
        self.features.append({"kind": "grating", "name": name, "rect": list(rect), "period_px": period,
                              "orientation": "vertical lines" if orient == "v" else "horizontal lines",
                              "dark": self.bank.names[lo], "light": self.bank.names[hi]})

    def _lamp(self, name, rect, bi, m):
        x, y, w, h = rect
        feather = self.px(20)
        self._ops.append(("fill", rect, [(bi, 1.0)]))
        self._irr_ops.append(("bump", rect, m, feather))
        core = (x + feather, y + feather, max(w - 2 * feather, 1), max(h - 2 * feather, 1))
        self.probes.append(Probe(name, "highlight", rect, _inset(core), [(bi, 1.0)], m=m))
        self.features.append({"kind": "highlight", "name": name, "rect": list(rect), "irradiance": m,
                              "ev_over_nominal": float(np.log2(m)), "feather_px": feather})

    def _rolloff(self, name, rect, bi, ev_lo, ev_hi):
        x, y, w, h = rect
        self._ops.append(("fill", rect, [(bi, 1.0)]))
        self._irr_ops.append(("ramp", rect, ev_lo, ev_hi))
        self.features.append({"kind": "rolloff-ramp", "name": name, "rect": list(rect), "ev_lo": ev_lo, "ev_hi": ev_hi,
                              "reflectance": self.bank.names[bi],
                              "note": "irradiance 2^ev rising linearly in ev from left to right"})
        for k, t in enumerate(np.linspace(0.04, 0.96, 12)):
            px = x + int(round(t * (w - 1)))
            ev = ev_lo + (ev_hi - ev_lo) * (px - x + 0.5) / w
            r = (px - 3, y, 7, h)
            self.probes.append(Probe(f"{name}-{k:02d}", "rolloff", r, (px - 2, y + h // 4, 5, max(h // 2, 1)),
                                     [(bi, 1.0)], m=float(2.0**ev), info={"ev": float(ev)}))

    def _isoluminant_pair(self, a: int, b: int) -> tuple[int, int]:
        """Two bank entries of equal CIE Y under D65: ``a`` and ``b`` scaled to the same luminance (the darker one
        is kept, the brighter is scaled down)."""
        from training.alexa import ssf_core as s
        from training.sensor_map import lights as lt

        e = lt.daylight_light(6500.0).spd
        spec = self.bank.spectra
        y = [float(s.CMF[:, 1] @ (e * spec[i]) / (s.CMF[:, 1] @ e)) for i in (a, b)]
        target = min(y)
        out = []
        for i, yi in zip((a, b), y):
            f = target / yi
            name = f"{self.bank.names[i]}x{f:.3f}"
            out.append(self.bank.add(name, spec[i] * f, f"{self.bank.sources[i]}, scaled to equal Y"))
        return out[0], out[1]

    # ------------------------------------------------------------------ irradiance
    def irradiance(self) -> np.ndarray:
        """(H, W) float32 irradiance relative to the nominal light (1 almost everywhere)."""
        irr = np.ones((self.H, self.W), dtype=np.float32)
        for op in self._irr_ops:
            kind = op[0]
            if kind == "box":
                _, (x, y, w, h), m = op
                irr[y : y + h, x : x + w] = m
            elif kind == "feather_box":
                _, (x, y, w, h), m, f = op
                yy = np.arange(y, y + h)[:, None]
                xx = np.arange(x, x + w)[None, :]
                d = np.minimum(np.minimum(xx - x, x + w - 1 - xx), np.minimum(yy - y, y + h - 1 - yy)) + 0.5
                t = smoothstep(d / max(f, 1))
                irr[y : y + h, x : x + w] = (1.0 - t) + t * m
            elif kind == "bump":
                _, (x, y, w, h), m, f = op
                yy = np.arange(y, y + h)[:, None]
                xx = np.arange(x, x + w)[None, :]
                d = np.minimum(np.minimum(xx - x, x + w - 1 - xx), np.minimum(yy - y, y + h - 1 - yy)) + 0.5
                t = smoothstep(d / max(f, 1))
                irr[y : y + h, x : x + w] = 1.0 + t * (m - 1.0)
            elif kind == "ramp":
                _, (x, y, w, h), ev_lo, ev_hi = op
                ev = ev_lo + (ev_hi - ev_lo) * (np.arange(w) + 0.5) / w
                irr[y : y + h, x : x + w] = (2.0**ev)[None, :]
        return irr

    # ------------------------------------------------------------------ render
    def render_refl(self, cam: np.ndarray) -> np.ndarray:
        """(H, W, 3) float32: the response of every pixel's reflectance under unit irradiance.

        ``cam`` is the ``(K, 3)`` response table of the bank under one light (camera or ALEXA)."""
        cam = np.asarray(cam, dtype=np.float64)
        out = np.empty((self.H, self.W, 3), dtype=np.float32)
        out[:] = cam[self._surround].astype(np.float32)
        for op in self._ops:
            kind = op[0]
            if kind == "fill":
                _, (x, y, w, h), mixture = op
                v = sum(wt * cam[i] for i, wt in mixture)
                out[y : y + h, x : x + w] = v.astype(np.float32)
            elif kind == "grad_scalar":
                _, (x, y, w, h), bi, lo, hi = op
                t = (np.arange(w) + 0.5) / w
                rho = lo * (hi / lo) ** t
                out[y : y + h, x : x + w] = (rho[None, :, None] * cam[bi][None, None, :]).astype(np.float32)
            elif kind == "grad_stops":
                _, (x, y, w, h), stops = op
                n = len(stops) - 1
                u = (np.arange(w) + 0.5) / w * n
                i = np.minimum(u.astype(int), n - 1)
                f = (u - i)[:, None]
                col = (1 - f) * cam[np.array(stops)[i]] + f * cam[np.array(stops)[i + 1]]
                out[y : y + h, x : x + w] = col[None, :, :].astype(np.float32)
            elif kind == "plate":
                _, (x, y, w, h), (tl, tr, bl, br) = op
                u = ((np.arange(w) + 0.5) / w)[None, :, None]
                v = ((np.arange(h) + 0.5) / h)[:, None, None]
                col = (1 - v) * ((1 - u) * cam[tl] + u * cam[tr]) + v * ((1 - u) * cam[bl] + u * cam[br])
                out[y : y + h, x : x + w] = col.astype(np.float32)
            else:  # supersampled blends of two spectra
                self._paint_blend(out, cam, op)
        return out

    def _paint_blend(self, out, cam, op) -> None:
        kind, (x, y, w, h) = op[0], op[1]
        lo, hi = op[2], op[3]
        if kind == "zone_plate":
            radius = op[4]
            cx, cy = x + w / 2.0, y + h / 2.0
            ss = 4

            def fn(X, Y):
                r2 = (X - cx) ** 2 + (Y - cy) ** 2
                return 0.5 + 0.5 * np.cos(np.pi * 0.5 * r2 / radius)

        elif kind == "edge":
            theta = np.radians(op[4])
            cx, cy = x + w / 2.0, y + h / 2.0
            ss = 8

            def fn(X, Y):
                d = (X - cx) * np.cos(theta) - (Y - cy) * np.sin(theta)
                return (d > 0).astype(np.float64)

        elif kind == "grating":
            period, orient = op[4], op[5]
            ss = 8

            def fn(X, Y):
                c = X if orient == "v" else Y
                return ((np.floor(c / (period / 2.0)) % 2) == 1).astype(np.float64)

        else:  # pragma: no cover
            raise ValueError(kind)
        wmap = supersampled(fn, x, y, w, h, ss)
        a = cam[lo].astype(np.float32)
        b = cam[hi].astype(np.float32)
        out[y : y + h, x : x + w] = a + (b - a) * wmap[:, :, None]

    # ------------------------------------------------------------------ the mixed-light weight
    def mix_weight(self) -> np.ndarray:
        """(W,) weight of the right-hand light of the mixed scene (0 left, 1 right, a smoothstep in between)."""
        return smoothstep(((np.arange(self.W) + 0.5) / self.W - MIX_START) / (MIX_END - MIX_START)).astype(np.float64)


def supersampled(fn, x: int, y: int, w: int, h: int, ss: int, band_rows: int = 128) -> np.ndarray:
    """Pixel average of ``fn(X, Y)`` over ``ss * ss`` sub-samples per pixel (a box aperture), (h, w) float32.

    ``X`` and ``Y`` are absolute positions in pixels; a pixel ``(i, j)`` covers ``[i, i+1) x [j, j+1)``."""
    out = np.empty((h, w), dtype=np.float32)
    sub = (np.arange(ss) + 0.5) / ss
    xs = (x + np.arange(w)[:, None] + sub[None, :]).reshape(-1)  # (w*ss,)
    for r0 in range(0, h, band_rows):
        r1 = min(r0 + band_rows, h)
        ys = (y + np.arange(r0, r1)[:, None] + sub[None, :]).reshape(-1)
        X, Y = np.meshgrid(xs, ys)
        v = fn(X, Y).reshape(r1 - r0, ss, w, ss)
        out[r0:r1] = v.mean(axis=(1, 3))
    return out


def apply_psf(canvas: np.ndarray, sigma_px: float = PSF_SIGMA_PX) -> np.ndarray:
    """Gaussian optical PSF applied to each channel of an (H, W, 3) image, in place-safe (returns a new array)."""
    out = np.empty_like(canvas, dtype=np.float32)
    for c in range(3):
        out[..., c] = ndimage.gaussian_filter(canvas[..., c].astype(np.float32), sigma_px, mode="nearest", truncate=3.0)
    return out
