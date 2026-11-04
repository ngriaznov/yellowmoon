#!/usr/bin/env python3
"""Runs spektrafilm as a black box on the oracle inputs and writes the fixtures (docs/24 section 8, task F-06).

    python run_oracle.py --inputs crates/ym-film/tests/oracle/inputs.json --out crates/ym-film/tests/oracle

Run it with the Python of a scratch virtualenv in which spektrafilm is installed from its pinned commit (see README.md).
spektrafilm is a test oracle only: it is never a dependency of a shipped crate and none of its code is in this repository. This
driver uses only its public interface (``init_params``, ``simulate``, the parameter names and the tap names ``log_e_film``,
``cmy_film``, ``log_e_print``, ``cmy_print``, ``rgb_out``).

Per-pixel mode: grain, DIR couplers, halation, glare, blurs and every spatial or random effect are off, auto exposure is off, the
input gamut compression and the output gamut and lightness compression are off. The adaptation window is on (and a second run
with it off gives ``log_e_film_nowindow``). Input: linear ARRI Wide Gamut 4. Output: linear sRGB primaries, no encoding.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from spektrafilm import init_params, simulate

#: Film, and the paper it is printed on (None: a slide, scanned directly). Vision3 500T is printed on the still paper the engine
#: uses in place of the cine print film, so that both sides use the same paper.
STOCKS = {
    "kodak_portra_400": "kodak_portra_endura",
    "kodak_ektar_100": "kodak_portra_endura",
    "kodak_gold_200": "kodak_portra_endura",
    "fujifilm_pro_400h": "fujifilm_crystal_archive_typeii",
    "kodak_vision3_500t": "kodak_portra_endura",
    "fujifilm_velvia_100": None,
    "kodak_ektachrome_100": None,
}


def set_path(root, path: str, value):
    """``root.a.b = value``, rebuilding the frozen dataclasses on the way."""
    def rec(obj, parts):
        if len(parts) == 1:
            try:
                setattr(obj, parts[0], value)
                return obj
            except dataclasses.FrozenInstanceError:
                return dataclasses.replace(obj, **{parts[0]: value})
        child = rec(getattr(obj, parts[0]), parts[1:])
        try:
            setattr(obj, parts[0], child)
            return obj
        except dataclasses.FrozenInstanceError:
            return dataclasses.replace(obj, **{parts[0]: child})
    return rec(root, path.split("."))


def make_params(film: str, paper: str | None, collect: str | None, window: bool = True, inject: str | None = None):
    """Parameters for the per-pixel mode with one tap collected."""
    params = init_params(film, paper or "kodak_portra_endura")
    settings = [
        ("io.input_color_space", "ARRI Wide Gamut 4"),
        ("io.input_cctf_decoding", False),
        ("io.output_color_space", "sRGB"),
        ("io.output_cctf_encoding", False),
        ("io.input_gamut_compress.active", False),
        ("io.output_gamut_compress.algorithm", "off"),
        ("io.output_gamut_compress.lightness_compression", None),
        ("camera.auto_exposure", False),
        ("film_render.grain.active", False),
        ("film_render.dir_couplers.active", False),
        ("film_render.glare.active", False),
        ("print_render.glare.active", False),
        ("film_render.halation.halation_amount", 0.0),
        ("debug.deactivate_spatial_effects", True),
        ("debug.deactivate_stochastic_effects", True),
        ("settings.apply_hanatos2025_adaptation_window", window),
        ("taps.collect", collect),
        ("taps.inject", inject),
        # A slide is scanned directly; a negative is printed and the print is scanned.
        ("io.scan_film", paper is None),
    ]
    for path, value in settings:
        params = set_path(params, path, value)
    return params


def run(image: np.ndarray, film: str, paper: str | None, collect: str, **kw) -> np.ndarray:
    out = np.asarray(simulate(image, make_params(film, paper, collect, **kw)), dtype=np.float64)
    return out.reshape(-1, 3)


#: Density levels of the probes (``--probes``): film and print densities are put in at the taps ``cmy_film`` and ``cmy_print``
#: (the same ``taps.inject`` mechanism as the paper white) so that one stage of the oracle is read on its own. The lattice is
#: all triples of these levels.
PROBE_LEVELS = [0.0, 0.25, 0.5, 0.8, 1.1, 1.4, 1.8, 2.3]


def probe_lattice() -> np.ndarray:
    """All triples of ``PROBE_LEVELS``, plus the 21-step neutral ramp from 0 to 2.5 (equal densities)."""
    grid = np.array([[a, b, c] for a in PROBE_LEVELS for b in PROBE_LEVELS for c in PROBE_LEVELS])
    ramp = np.repeat(np.linspace(0.0, 2.5, 21)[:, None], 3, axis=1)
    return np.concatenate([grid, ramp], axis=0)


def write_probes(out: Path) -> None:
    """Writes ``probe_<stock>.json``: each stock's stage read on its own at injected densities.

    * a negative film: ``cmy_film`` injected, read at ``log_e_print`` and ``cmy_print`` (the enlarger and the paper's curves),
      and ``rgb_out`` (everything after the film);
    * a paper (named by its film): ``cmy_print`` injected, read at ``rgb_out`` (the scan alone);
    * a slide: ``cmy_film`` injected, read at ``rgb_out`` (the slide's scan alone).
    """
    dens = probe_lattice().reshape(-1, 1, 3)
    for film, paper in STOCKS.items():
        record = {"film": film, "paper": paper, "densities": np.round(dens.reshape(-1, 3), 6).tolist(), "taps": {}}
        if paper:
            record["taps"]["log_e_print"] = run(dens, film, paper, "log_e_print", inject="cmy_film")
            record["taps"]["cmy_print"] = run(dens, film, paper, "cmy_print", inject="cmy_film")
            record["taps"]["rgb_out_from_film"] = run(dens, film, paper, "rgb_out", inject="cmy_film")
            record["taps"]["rgb_out_from_print"] = run(dens, film, paper, "rgb_out", inject="cmy_print")
        else:
            record["taps"]["rgb_out_from_film"] = run(dens, film, paper, "rgb_out", inject="cmy_film")
        record["taps"] = {k: np.round(v, 6).tolist() for k, v in record["taps"].items()}
        (out / f"probe_{film}.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
        print("wrote probe", film)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="comma separated film ids")
    parser.add_argument("--probes", action="store_true", help="write probe_<stock>.json (injected densities) instead of the fixtures")
    args = parser.parse_args()
    if args.probes:
        write_probes(args.out)
        return
    inputs = json.loads(args.inputs.read_text())
    groups = [(name, np.array(inputs[name], dtype=np.float64)) for name in ("patches", "lattice", "ramp")]
    pixels = np.concatenate([g for _, g in groups], axis=0)
    image = pixels.reshape(-1, 1, 3)
    wanted = args.only.split(",") if args.only else list(STOCKS)
    for film in wanted:
        paper = STOCKS[film]
        taps = {"log_e_film": run(image, film, paper, "log_e_film"), "cmy_film": run(image, film, paper, "cmy_film"),
                "log_e_film_nowindow": run(image, film, paper, "log_e_film", window=False)}
        if paper:
            taps["log_e_print"] = run(image, film, paper, "log_e_print")
            taps["cmy_print"] = run(image, film, paper, "cmy_print")
        taps["rgb_out"] = run(image, film, paper, "rgb_out")
        record = {"film": film, "paper": paper, "groups": {}}
        start = 0
        for name, g in groups:
            record["groups"][name] = {tap: np.round(v[start:start + len(g)], 6).tolist() for tap, v in taps.items()}
            start += len(g)
        if paper:
            # The paper's own white: the colour of a print with no density (its base alone), to scale rgb_out to paper white = 1.
            zero = np.zeros((1, 1, 3))
            record["paper_white_rgb"] = np.round(run(zero, film, paper, "rgb_out", inject="cmy_print")[0], 6).tolist()
        record["inputs_sha256"] = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
        (args.out / f"{film}.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
        print("wrote", film)


if __name__ == "__main__":
    main()
