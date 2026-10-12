#!/usr/bin/env python3
"""Generate ACES 2.0 output-transform test vectors with OpenColorIO.

Writes one JSON file per Yellowmoon output (docs/02-colour-pipeline.md, C12) to
crates/ym-color/tests/data/aces2/. Each file holds a deterministic set of linear
ACES2065-1 (AP0) inputs and the expected display-encoded RGB computed by OpenColorIO's
built-in ACES 2.0 output transforms (float32 CPU renderer, no optimisation), plus the
expected display-linear RGB before the display encoding.

OCIO transform for each output
------------------------------
All four outputs are built from two OCIO built-ins, the same two the studio config uses:

  1. "ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-<limit>_2.0"
     (ACES 2.0 output transform, 100 nit SDR, <limit> = REC709 or P3-D65, result in CIE XYZ D65)
  2. a "DISPLAY - CIE-XYZ-D65_to_<encoding>" built-in (XYZ -> encoding primaries + transfer)

  viewport     P3-D65 limited    DisplayP3 (sRGB transfer)     studio config: "Display P3 - Display" /
                                                                "ACES 2.0 - SDR 100 nits (P3 D65)"
  srgb         Rec.709 limited   sRGB                          studio config: "sRGB - Display" /
                                                                "ACES 2.0 - SDR 100 nits (Rec.709)"
  display_p3   P3-D65 limited    DisplayP3 (sRGB transfer)     same transform as viewport
  rec2020      P3-D65 limited    Rec.2020, BT.1886 gamma 2.4   no studio-config display/view pair exists
                                                                for this combination, so it is built from
                                                                the two built-ins above (REC.1886-REC.2020)

For the three outputs that have a studio-config pair the script checks that the built-in chain
and the config's DisplayViewTransform agree exactly, and records that in the JSON.

Usage
-----
  python3 tools/vectors/aces2_vectors.py                  # (re)write the committed JSON files
  python3 tools/vectors/aces2_vectors.py --check          # fail if the committed files are stale
                                                          # (inputs and values compared with a float32
                                                          # tolerance: numpy and OCIO use CPU-specific SIMD)
  python3 tools/vectors/aces2_vectors.py --sweep 1000000 --sweep-out /tmp/aces2_sweep
        # dev only: large random sweep as raw little-endian float32 files, one per output,
        # for the ignored Rust test crates/ym-color/tests/aces2_sweep.rs

Besides the four SDR files the script writes hdr_p3.json: the same OpenColorIO transform at other peak
luminances (48, 203, 1000, 4000 and 10000 nits, P3-D65 limited), for the peak-dependent maths of
ym_color::aces2::Aces2Params. Those vectors are not a Yellowmoon output (V1 renders 100 nit SDR only).

The output is deterministic for a given OpenColorIO, numpy and CPU family; `--check` tolerates
float32-level differences between machines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "crates" / "ym-color" / "tests" / "data" / "aces2"
SCHEMA = "yellowmoon/aces2-vectors/1"
STUDIO_CONFIG = "ocio://studio-config-latest"
MIN_OCIO = (2, 4)

# ---------------------------------------------------------------------------------------
# Colour spaces (chromaticities)
# ---------------------------------------------------------------------------------------

D65 = (0.3127, 0.3290)
ACES_WHITE = (0.32168, 0.33767)
GAMUTS = {
    "AP0": ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770), ACES_WHITE),
    "AP1": ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), ACES_WHITE),
    "Rec709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), D65),
    "P3-D65": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),
    "Rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), D65),
    "AWG4": ((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308), D65),
}

BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)


def xy_to_xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgb_to_xyz_matrix(gamut):
    """Column-vector RGB -> XYZ matrix (white = Y 1) from primaries and white."""
    r, g, b, w = gamut
    prim = np.array([xy_to_xyz(r), xy_to_xyz(g), xy_to_xyz(b)]).T
    s = np.linalg.solve(prim, xy_to_xyz(w))
    return prim * s


def cat_bradford(src_white, dst_white):
    src = BRADFORD @ xy_to_xyz(src_white)
    dst = BRADFORD @ xy_to_xyz(dst_white)
    return np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD


def xyy_to_ap0(x, y, big_y):
    """CIE xyY (D65 based, as every display and camera gamut here) to linear AP0.

    Points outside the spectral locus or outside AP0 give negative components on purpose.
    """
    xyz = xy_to_xyz((x, y)) * big_y
    m = np.linalg.inv(rgb_to_xyz_matrix(GAMUTS["AP0"])) @ cat_bradford(D65, ACES_WHITE)
    return m @ xyz


def to_ap0_matrix(name):
    """Linear RGB of gamut `name` -> linear AP0 (Bradford-adapted to the ACES white)."""
    g = GAMUTS[name]
    m = rgb_to_xyz_matrix(g)
    if g[3] != ACES_WHITE:
        m = cat_bradford(g[3], ACES_WHITE) @ m
    return np.linalg.inv(rgb_to_xyz_matrix(GAMUTS["AP0"])) @ m


def xyz_to_rgb_matrix(name):
    return np.linalg.inv(rgb_to_xyz_matrix(GAMUTS[name]))


# ---------------------------------------------------------------------------------------
# The four outputs
# ---------------------------------------------------------------------------------------

OUTPUTS = {
    "viewport": {
        "description": "Viewport: ACES 2.0 SDR 100 nit, P3-D65 limited, Display P3 primaries, sRGB transfer",
        "limit": "SDR-100nit-P3-D65_2.0",
        "limiting_primaries": "P3-D65",
        "display": "DISPLAY - CIE-XYZ-D65_to_DisplayP3",
        "encoding_primaries": "P3-D65",
        "transfer": "sRGB piecewise (IEC 61966-2-1)",
        "studio": ("Display P3 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"),
        "seed": 20261012,
    },
    "srgb": {
        "description": "sRGB export: ACES 2.0 SDR 100 nit, Rec.709 limited, Rec.709 primaries, sRGB transfer",
        "limit": "SDR-100nit-REC709_2.0",
        "limiting_primaries": "Rec709",
        "display": "DISPLAY - CIE-XYZ-D65_to_sRGB",
        "encoding_primaries": "Rec709",
        "transfer": "sRGB piecewise (IEC 61966-2-1)",
        "studio": ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"),
        "seed": 20261013,
    },
    "display_p3": {
        "description": "Display P3 export: ACES 2.0 SDR 100 nit, P3-D65 limited, Display P3 primaries, sRGB transfer",
        "limit": "SDR-100nit-P3-D65_2.0",
        "limiting_primaries": "P3-D65",
        "display": "DISPLAY - CIE-XYZ-D65_to_DisplayP3",
        "encoding_primaries": "P3-D65",
        "transfer": "sRGB piecewise (IEC 61966-2-1)",
        "studio": ("Display P3 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"),
        "seed": 20261014,
    },
    "rec2020": {
        "description": "Rec.2020 export: ACES 2.0 SDR 100 nit, P3-D65 limited, Rec.2020 primaries, BT.1886 gamma 2.4",
        "limit": "SDR-100nit-P3-D65_2.0",
        "limiting_primaries": "P3-D65",
        "display": "DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.2020",
        "encoding_primaries": "Rec2020",
        "transfer": "BT.1886 gamma 2.4 (L = V^2.4, Lw = 1, Lb = 0)",
        "studio": None,
        "seed": 20261015,
    },
}
ORDER = ["viewport", "srgb", "display_p3", "rec2020"]


def builtin(style):
    t = ocio.BuiltinTransform()
    t.setStyle(style)
    return t


def view_style(limit):
    return f"ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - {limit}"


def encoded_chain(spec):
    g = ocio.GroupTransform()
    g.appendTransform(builtin(view_style(spec["limit"])))
    g.appendTransform(builtin(spec["display"]))
    return g


def linear_chain(spec):
    """View built-in followed by XYZ-D65 -> encoding primaries (no transfer function)."""
    g = ocio.GroupTransform()
    g.appendTransform(builtin(view_style(spec["limit"])))
    m = np.eye(4)
    m[:3, :3] = xyz_to_rgb_matrix(spec["encoding_primaries"])
    mt = ocio.MatrixTransform()
    mt.setMatrix(m.flatten().tolist())
    g.appendTransform(mt)
    return g


def cpu_processor(cfg, transform):
    proc = cfg.getProcessor(transform)
    return proc.getOptimizedCPUProcessor(ocio.BIT_DEPTH_F32, ocio.BIT_DEPTH_F32, ocio.OPTIMIZATION_NONE)


def apply(cpu, rgb):
    out = np.ascontiguousarray(rgb, dtype=np.float32).copy()
    cpu.applyRGB(out)
    return out


# ---------------------------------------------------------------------------------------
# Input sets
# ---------------------------------------------------------------------------------------

GREY = 0.18


def structured_inputs():
    """Inputs shared by all four outputs: (group name, Nx3 float64 array)."""
    groups = []

    stops = np.arange(-8.0, 12.0 + 1e-9, 0.5)
    neutrals = np.repeat((GREY * 2.0**stops)[:, None], 3, axis=1)
    groups.append(("neutral_ramp_-8_to_+12_stops", neutrals))

    near_black = np.repeat(np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])[:, None], 3, axis=1)
    groups.append(("near_black_neutrals", near_black))

    # Saturated primaries and secondaries of the display and camera gamuts at several exposures,
    # fully saturated and half way to the grey of the same luminance.
    corners = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=float
    )
    exposures = [-5.0, -2.0, 0.0, 2.0, 5.0]
    rows = []
    for name in ["Rec709", "P3-D65", "Rec2020", "AP1", "AWG4"]:
        m = to_ap0_matrix(name)
        for ev in exposures:
            scale = GREY * 2.0**ev
            for c in corners:
                rgb = c * scale
                # grey of equal Y in the gamut: 1/3 (works for the equal-energy distribution used here)
                grey = np.full(3, c.mean() * scale)
                for sat in (1.0, 0.5):
                    rows.append(m @ (grey + sat * (rgb - grey)))
    groups.append(("gamut_primaries_secondaries_x_exposure_x_saturation", np.array(rows)))

    # A chromaticity wheel around D65: 24 hues x 2 radii (in xy, the larger one is outside the
    # spectral locus for most hues) x 3 exposures. Covers every hue sector of the gamut mapper.
    rows = []
    for ev in (-3.0, 1.0, 4.0):
        for radius in (0.10, 0.30):
            for k in range(24):
                th = 2.0 * np.pi * k / 24.0
                rows.append(xyy_to_ap0(D65[0] + radius * np.cos(th), D65[1] + radius * np.sin(th), GREY * 2.0**ev))
    groups.append(("chromaticity_wheel_xy_24_hues_x_2_radii_x_3_exposures", np.array(rows)))

    # AP0 primaries / secondaries themselves (outside the spectral locus), several exposures
    rows = [c * GREY * 2.0**ev for ev in (-4.0, 0.0, 4.0) for c in corners]
    groups.append(("ap0_primaries_secondaries", np.array(rows)))

    # Out-of-gamut and negative edge cases
    edge = np.array(
        [
            [-0.01, 0.18, 0.18],
            [0.18, -0.01, 0.18],
            [0.18, 0.18, -0.01],
            [-0.05, -0.05, 0.5],
            [0.9, -0.1, -0.1],
            [-1e-4, 1e-4, 0.0],
            [-0.5, -0.5, -0.5],
            [1000.0, 1000.0, 1000.0],
            [5000.0, 0.0, 0.0],
            [0.0, 5000.0, 0.0],
            [0.0, 0.0, 5000.0],
            [65504.0, 65504.0, 65504.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [16.0, 16.0, 16.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    groups.append(("edge_cases", edge))
    return groups


def random_inputs(seed, scale=1.0):
    """Seeded random inputs (per output): counts scale with `scale`."""
    rng = np.random.default_rng(seed)
    groups = []

    def count(n):
        return max(1, int(round(n * scale)))

    # in-gamut (AP1) colours: random barycentric mix of the AP1 primaries, random exposure
    n = count(80)
    w = rng.dirichlet([0.4, 0.4, 0.4], size=n)
    ev = rng.uniform(-8.0, 10.0, size=n)
    rgb_ap1 = w * (GREY * 2.0**ev)[:, None] * 3.0
    groups.append(("random_in_gamut_ap1", rgb_ap1 @ to_ap0_matrix("AP1").T))

    # AWG4 gamut, as the real pipeline delivers it: random chroma incl. outside AP1
    n = count(80)
    w = rng.dirichlet([0.5, 0.5, 0.5], size=n)
    ev = rng.uniform(-7.0, 9.0, size=n)
    rgb = w * (GREY * 2.0**ev)[:, None] * 3.0
    groups.append(("random_awg4", rgb @ to_ap0_matrix("AWG4").T))

    # out-of-gamut AP0: independent channels, some negative
    n = count(80)
    mag = GREY * 2.0 ** rng.uniform(-6.0, 6.0, size=n)
    rgb = rng.uniform(-0.25, 1.5, size=(n, 3)) * mag[:, None]
    groups.append(("random_out_of_gamut_ap0", rgb))

    # random chromaticities: polar around D65 out to well beyond the spectral locus, random exposure
    n = count(60)
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = 0.5 * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    ev = rng.uniform(-8.0, 8.0, size=n)
    rows = [
        xyy_to_ap0(D65[0] + r * np.cos(t), D65[1] + r * np.sin(t), GREY * 2.0**e)
        for r, t, e in zip(radius, th, ev)
    ]
    groups.append(("random_chromaticity_polar_xy", np.array(rows)))

    # near black with random chroma, a few slightly negative
    n = count(40)
    rgb = 10.0 ** rng.uniform(-6.0, -2.0, size=(n, 3)) * rng.choice([1.0, 1.0, -0.2], size=(n, 3))
    groups.append(("random_near_black", rgb))

    # very bright, above the AP1 clamp (forward limit 1024) for some channels
    n = count(30)
    rgb = 10.0 ** rng.uniform(1.0, 4.5, size=(n, 3)) * rng.choice([1.0, 0.05], size=(n, 3))
    groups.append(("random_very_bright", rgb))
    return groups


HDR_PEAKS = [48, 203, 1000, 4000, 10000]
# OpenColorIO has built-ins for these peaks (P3-D65 limited); the generic chain is cross-checked against them.
HDR_BUILTINS = {1000: "HDR-1000nit-P3-D65_2.0", 4000: "HDR-4000nit-P3-D65_2.0"}
HDR_SEED = 20261016


def hdr_chain(peak):
    """AP0 -> AP1 clamp (as the built-ins do) -> ACES 2 fixed function (P3-D65 limited) -> clamp to the peak.

    Output: P3-D65 linear light, 1.0 = 100 nits, so the maximum is peak / 100.
    """
    ap0_to_ap1 = np.linalg.inv(rgb_to_xyz_matrix(GAMUTS["AP1"])) @ rgb_to_xyz_matrix(GAMUTS["AP0"])
    m4 = np.eye(4)
    m4[:3, :3] = ap0_to_ap1
    g = ocio.GroupTransform()
    to_ap1 = ocio.MatrixTransform()
    to_ap1.setMatrix(m4.flatten().tolist())
    g.appendTransform(to_ap1)
    upper = 8.0 * (128.0 + 768.0 * (np.log(peak / 100.0) / np.log(10000.0 / 100.0)))
    clamp = ocio.RangeTransform()
    clamp.setMinInValue(0.0)
    clamp.setMaxInValue(float(upper))
    clamp.setMinOutValue(0.0)
    clamp.setMaxOutValue(float(upper))
    g.appendTransform(clamp)
    from_ap1 = ocio.MatrixTransform()
    from_ap1.setMatrix(np.linalg.inv(m4).flatten().tolist())
    g.appendTransform(from_ap1)
    red, green, blue, white = GAMUTS["P3-D65"]
    g.appendTransform(
        ocio.FixedFunctionTransform(
            ocio.FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20,
            [float(peak), red[0], red[1], green[0], green[1], blue[0], blue[1], white[0], white[1]],
        )
    )
    peak_clamp = ocio.RangeTransform()
    peak_clamp.setMinInValue(0.0)
    peak_clamp.setMaxInValue(peak / 100.0)
    peak_clamp.setMinOutValue(0.0)
    peak_clamp.setMaxOutValue(peak / 100.0)
    g.appendTransform(peak_clamp)
    return g


def hdr_builtin_chain(style_suffix):
    """The OpenColorIO built-in for a peak, followed by XYZ-D65 -> P3-D65 (no transfer function)."""
    g = ocio.GroupTransform()
    g.appendTransform(builtin(f"ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - {style_suffix}"))
    m = np.eye(4)
    m[:3, :3] = xyz_to_rgb_matrix("P3-D65")
    mt = ocio.MatrixTransform()
    mt.setMatrix(m.flatten().tolist())
    g.appendTransform(mt)
    return g


def hdr_inputs(peak, rng):
    """A compact input set per peak: neutrals, gamut corners, a chromaticity wheel and seeded random colours."""
    rows = []
    # neutral ramp, one stop apart, up to the scene value where the roof sits
    top_ev = np.log2(8.0 * (128.0 + 768.0 * (np.log(peak / 100.0) / np.log(100.0))) / GREY) if peak > 47 else 12
    for ev in np.arange(-8.0, min(top_ev, 14.0) + 1e-9, 1.0):
        rows.append(np.full(3, GREY * 2.0**ev))
    corners = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=float)
    for name in ("P3-D65", "Rec2020", "AP1"):
        m = to_ap0_matrix(name)
        for ev in (-3.0, 1.0, 4.0, 7.0):
            for c in corners:
                grey = np.full(3, c.mean() * GREY * 2.0**ev)
                rows.append(m @ (grey + 0.8 * (c * GREY * 2.0**ev - grey)))
    for ev in (-2.0, 3.0, 6.0):
        for k in range(24):
            th = 2.0 * np.pi * k / 24.0
            rows.append(xyy_to_ap0(D65[0] + 0.25 * np.cos(th), D65[1] + 0.25 * np.sin(th), GREY * 2.0**ev))
    w = rng.dirichlet([0.5, 0.5, 0.5], size=24)
    ev = rng.uniform(-6.0, 9.0, size=24)
    rows.extend(list((w * (GREY * 2.0**ev)[:, None] * 3.0) @ to_ap0_matrix("AP1").T))
    rows.extend(list(rng.uniform(-0.2, 1.5, size=(12, 3)) * (GREY * 2.0 ** rng.uniform(-4.0, 8.0, size=(12, 1)))))
    return np.asarray(rows, dtype=np.float64).astype(np.float32)


def build_hdr(base_cfg):
    rng = np.random.default_rng(HDR_SEED)
    peaks = []
    for peak in HDR_PEAKS:
        inputs = hdr_inputs(peak, rng)
        expected = apply(cpu_processor(base_cfg, hdr_chain(peak)), inputs)
        entry = {"peak_luminance_nits": peak, "input_ap0": f32_json(inputs), "expected_limiting_linear": f32_json(expected)}
        suffix = HDR_BUILTINS.get(peak)
        if suffix is not None:
            built = apply(cpu_processor(base_cfg, hdr_builtin_chain(suffix)), inputs)
            diff = float(np.abs(built - expected).max()) / (peak / 100.0)
            entry["builtin"] = {"style": f"ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - {suffix}", "max_relative_difference_to_chain": diff}
            if diff > 1e-5:
                print(f"warning: hdr {peak}: built-in differs from the generic chain by {diff}", file=sys.stderr)
        peaks.append(entry)
    return {
        "schema": "yellowmoon/aces2-hdr-vectors/1",
        "description": "ACES 2.0 output transform at other peak luminances: P3-D65 limited, linear light, 1.0 = 100 nits",
        "generator": {"script": "tools/vectors/aces2_vectors.py", "note": "inputs are float32 values; expected values are OpenColorIO float32 CPU output"},
        "ocio": {
            "version": ocio.__version__,
            "chain": "matrix AP0->AP1, range [0, 8 r_hit], matrix AP1->AP0, FixedFunction ACES_OUTPUT_TRANSFORM_20 (peak, P3-D65 primaries), range [0, peak/100]",
            "optimization": "none",
            "precision": "float32",
        },
        "input_space": "ACES2065-1 (linear, AP0 primaries, ACES white 0.32168 0.33767)",
        "limiting_primaries": "P3-D65",
        "gate": {
            "metric": "max over channels of |ours - expected| / (peak / 100)",
            "unit": "fraction of the display peak",
            "max": 1e-4,
        },
        "peaks": peaks,
    }


def f32_json(a):
    """Nested list of float32 values as shortest round-trip decimal strings turned back to numbers."""
    return [[float(np.format_float_positional(np.float32(v), unique=True, trim="-")) for v in row] for row in a]


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------


def check_version():
    ver = tuple(int(p) for p in ocio.__version__.split(".")[:2])
    if ver < MIN_OCIO:
        sys.exit(f"OpenColorIO {ocio.__version__} is too old, need >= {MIN_OCIO[0]}.{MIN_OCIO[1]}")


def build_output(name, studio_cfg, base_cfg):
    spec = OUTPUTS[name]
    groups = structured_inputs() + random_inputs(spec["seed"])
    parts = []
    group_meta = []
    start = 0
    for gname, arr in groups:
        arr32 = np.asarray(arr, dtype=np.float64).astype(np.float32)
        parts.append(arr32)
        group_meta.append({"name": gname, "start": start, "count": int(arr32.shape[0])})
        start += arr32.shape[0]
    inputs = np.concatenate(parts, axis=0)

    enc_cpu = cpu_processor(base_cfg, encoded_chain(spec))
    lin_cpu = cpu_processor(base_cfg, linear_chain(spec))
    expected = apply(enc_cpu, inputs)
    expected_lin = apply(lin_cpu, inputs)

    studio_info = None
    if spec["studio"] is not None:
        display, view = spec["studio"]
        dv = ocio.DisplayViewTransform(src="ACES2065-1", display=display, view=view)
        studio_cpu = cpu_processor(studio_cfg, dv)
        studio_out = apply(studio_cpu, inputs)
        diff = float(np.abs(studio_out - expected).max())
        studio_info = {
            "config": STUDIO_CONFIG,
            "config_name": studio_cfg.getName(),
            "display": display,
            "view": view,
            "max_abs_difference_to_builtin_chain": diff,
        }
        if diff != 0.0:
            print(f"warning: {name}: studio config differs from built-in chain by {diff}", file=sys.stderr)

    doc = {
        "schema": SCHEMA,
        "output": name,
        "description": spec["description"],
        "generator": {
            "script": "tools/vectors/aces2_vectors.py",
            "note": "inputs are float32 values; expected values are OpenColorIO float32 CPU output",
        },
        "ocio": {
            "version": ocio.__version__,
            "view_builtin": view_style(spec["limit"]),
            "display_builtin": spec["display"],
            "optimization": "none",
            "precision": "float32",
            "studio_config_pair": studio_info,
            "studio_config_pair_note": (
                "the studio config has no display with Rec.2020 primaries and a BT.1886 gamma (its Rec.2020 "
                "views sit under the HDR Rec.2100 displays), so this output is built from the two "
                "built-ins, exactly as the config builds its other display/view pairs"
                if spec["studio"] is None
                else "the built-in chain equals the studio config's DisplayViewTransform"
            ),
        },
        "input_space": "ACES2065-1 (linear, AP0 primaries, ACES white 0.32168 0.33767)",
        "peak_luminance_nits": 100,
        "limiting_primaries": spec["limiting_primaries"],
        "encoding_primaries": spec["encoding_primaries"],
        "transfer": spec["transfer"],
        "gate": {
            "metric": "max over channels of |ours - expected| * 1023",
            "unit": "10-bit code values",
            "max": 1.0,
        },
        "count": int(inputs.shape[0]),
        "groups": group_meta,
        "input_ap0": f32_json(inputs),
        "expected_encoded": f32_json(expected),
        "expected_display_linear": f32_json(expected_lin),
    }
    return doc


def dump(doc):
    """Compact JSON, one triplet per line inside the arrays (diff-friendly)."""
    head = {k: v for k, v in doc.items() if k not in ("input_ap0", "expected_encoded", "expected_display_linear")}
    text = json.dumps(head, indent=2)
    assert text.endswith("}")
    body = []
    for key in ("input_ap0", "expected_encoded", "expected_display_linear"):
        rows = ",\n".join("    " + json.dumps(r, separators=(",", ":")) for r in doc[key])
        body.append(f'  "{key}": [\n{rows}\n  ]')
    return text[:-1].rstrip() + ",\n" + ",\n".join(body) + "\n}\n"


def dump_hdr(doc):
    """Compact JSON for the HDR file: one triplet per line inside the arrays of each peak."""
    head = {k: v for k, v in doc.items() if k != "peaks"}
    text = json.dumps(head, indent=2)
    assert text.endswith("}")
    peaks = []
    for entry in doc["peaks"]:
        small = {k: v for k, v in entry.items() if k not in ("input_ap0", "expected_limiting_linear")}
        # Fields, re-indented to sit inside the "peaks" array, then the two arrays one triplet per line.
        fields = textwrap.indent(json.dumps(small, indent=2)[1:-2].rstrip(), "    ")
        arrays = []
        for key in ("input_ap0", "expected_limiting_linear"):
            rows = ",\n".join("        " + json.dumps(r, separators=(",", ":")) for r in entry[key])
            arrays.append(f'      "{key}": [\n{rows}\n      ]')
        peaks.append("    {" + fields + ",\n" + ",\n".join(arrays) + "\n    }")
    return text[:-1].rstrip() + ',\n  "peaks": [\n' + ",\n".join(peaks) + "\n  ]\n}\n"


def same_up_to_float32_noise(old, new):
    """True if two parsed JSON documents are equal, numbers within float32-level noise.

    numpy and OpenColorIO pick CPU-specific SIMD code (sin/cos, pow), so another machine can differ in the
    last bit or two of a float32 value; strings, integers, keys and array shapes must match exactly.
    """
    if isinstance(old, dict):
        return isinstance(new, dict) and old.keys() == new.keys() and all(same_up_to_float32_noise(old[k], new[k]) for k in old)
    if isinstance(old, list):
        return isinstance(new, list) and len(old) == len(new) and all(same_up_to_float32_noise(a, b) for a, b in zip(old, new))
    if isinstance(old, float) or isinstance(new, float):
        if isinstance(old, bool) or isinstance(new, bool) or not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            return False
        return math.isclose(old, new, rel_tol=2e-5, abs_tol=2e-6)
    return old == new


def run_sweep(n, out_dir, base_cfg):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        spec = OUTPUTS[name]
        scale = n / 390.0
        groups = random_inputs(spec["seed"] + 1000, scale=scale)
        inputs = np.concatenate([np.asarray(a).astype(np.float32) for _, a in groups], axis=0)
        enc = apply(cpu_processor(base_cfg, encoded_chain(spec)), inputs)
        lin = apply(cpu_processor(base_cfg, linear_chain(spec)), inputs)
        for tag, arr in (("in", inputs), ("enc", enc), ("lin", lin)):
            (out_dir / f"{name}.{tag}.f32").write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        print(f"{name}: {inputs.shape[0]} samples -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="fail if committed files differ from regenerated ones")
    ap.add_argument("--sweep", type=int, default=0, help="dev: write a random sweep of about N samples per output")
    ap.add_argument("--sweep-out", type=Path, default=Path("aces2_sweep"))
    args = ap.parse_args()

    check_version()
    studio_cfg = ocio.Config.CreateFromFile(STUDIO_CONFIG)
    base_cfg = ocio.Config()  # built-ins need no config

    if args.sweep:
        run_sweep(args.sweep, args.sweep_out, base_cfg)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    files = [(f"{name}.json", build_output(name, studio_cfg, base_cfg), dump) for name in ORDER]
    files.append(("hdr_p3.json", build_hdr(base_cfg), dump_hdr))
    for filename, doc, dumper in files:
        text = dumper(doc)
        path = args.out_dir / filename
        if args.check:
            # Parse both sides: the files must be valid JSON and equal up to float32-level noise.
            if not path.exists() or not same_up_to_float32_noise(json.loads(path.read_text()), json.loads(text)):
                stale.append(path.name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path} ({len(text) // 1024} KiB)")
    if args.check:
        if stale:
            sys.exit("stale vector files: " + ", ".join(stale))
        print("vector files are up to date")


if __name__ == "__main__":
    main()
