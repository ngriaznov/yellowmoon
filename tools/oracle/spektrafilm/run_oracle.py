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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="comma separated film ids")
    args = parser.parse_args()
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
