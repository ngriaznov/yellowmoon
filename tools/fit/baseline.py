#!/usr/bin/env python3
"""Formats the baseline of docs/29-raw-constants.md (task RC-06) from the raw outputs of tools/fit/baseline.sh.

Standard library only. `baseline.sh OUTDIR` runs the examples and leaves their raw output in OUTDIR; this
script reads it and prints the markdown tables of docs/29 section 2.2a. It does no image processing and
invents nothing: a number that is missing from the raw output is printed as "not run" with the reason.

    python3 -I tools/fit/baseline.py OUTDIR > OUTDIR/baseline.md

Files read from OUTDIR: demosaic_gate/gate.json, fit_demosaic.jsonl, capture_sharpen_ab/results.json,
fit_diffraction.jsonl, fit_sharpen.jsonl and spec04_edge.txt. The output is a function of those files, so the
same files give the same bytes.
"""

import json
import os
import re
import sys

METHODS = ("adaptive_ratio", "menon", "lmmse")
HELD_SETS = ("McMaster (18)",)  # held out of the fits (docs/29 section 2.3, protocol rule 2)
SET_ORDER = ("synthetic (cpu_demosaic.rs)", "Kodak (24)", "McMaster (18)")


def read_lines(path):
    """The JSON lines of an example: (data_note, base evaluation line)."""
    note, base = None, None
    with open(path) as f:
        for text in f:
            text = text.strip()
            if not text.startswith("{"):
                continue
            line = json.loads(text)
            if line.get("kind") == "data_note":
                note = line
            elif line.get("kind") == "eval" and line.get("label") == "base":
                base = line
    if note is None or base is None:
        raise SystemExit(f"{path}: no data note or no base line")
    return note, base


def split_tag(name):
    return "held out" if name in HELD_SETS else "fit"


def short(name):
    return name.split(" (")[0] if name.startswith(("Kodak", "McMaster")) else "synthetic"


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f2(v):
    return f"{v:.2f}"


def nine_cells(gate):
    cells = {(c["set"], round(c["scale"], 2)): c for c in gate["r6_rule"]["cells"]}
    rows = []
    for p in gate["psnr_linear"]:
        scale = round(p["scale"], 2)
        m = p["mean_db"]
        win = cells[(p["set"], scale)]["winner"]
        rows.append([short(p["set"]), split_tag(p["set"]), f"{scale:g}"] + [f2(m[k]) for k in METHODS] + [win])
    agg = gate["r6_rule"]["aggregate_mean_db"]
    won = gate["r6_rule"]["cells_won"]
    rows.append(["**mean of the 9 cells**", "", ""] + [f"**{agg[k]:.3f}**" for k in METHODS]
                + [f"cells won {won['adaptive_ratio']} / {won['menon']} / {won['lmmse']}"])
    return md(rows, ["set", "split", "signal scale", "adaptive ratio (dB)", "Menon (dB)", "LMMSE (dB)", "best"])


def artifacts(gate):
    rows = []
    for a in gate["artifacts"]:
        by = {r["method"]: r for r in a["rows"]}
        rows.append([short(a["set"]), split_tag(a["set"]), a["images"]]
                    + [f"{by[k]['zipper_pct']:.2f}" for k in METHODS]
                    + [f"{by[k]['false_colour_pct']:.3f}" for k in METHODS]
                    + [f"{by['adaptive_ratio']['chroma_de00_mean']:.3f}"])
    return md(rows, ["set", "split", "images", "zipper AR (% of edge px)", "Menon", "LMMSE",
                     "false colour AR (% of px)", "Menon", "LMMSE", "chroma dE00 AR"])


def noisy(gate):
    rows = []
    for n in gate["noisy"]:
        d, b = n["demosaic_db"], n["bilateral_db"]
        rows.append([short(n["set"]), split_tag(n["set"]), n["level"]]
                    + [f"{f2(d[k])} / {f2(b[k])}" for k in METHODS])
    return md(rows, ["set", "split", "level", "adaptive ratio: demosaic / + R6 (dB)", "Menon", "LMMSE"])


def groups(base):
    p, c = base["psnr_db"], base.get("artifacts")
    rows = [
        ["A (procedural 7, Kodak 24, McMaster 18)", "all 49", f"{p['A']:.3f}"],
        ["A, fit set (procedural and Kodak)", "fit", f"{p['A_fit']:.3f}"],
        ["A, held out (McMaster)", "held out", f"{p['A_held']:.3f}"],
        ["B (dark and noisy, 17 images, noise seed 1)", "all", f"{p['B']:.3f}"],
        ["B, fit images", "fit", f"{p['B_fit']:.3f}"],
        ["B, held-out images (McMaster)", "held out", f"{p['B_held']:.3f}"],
        ["B2 (the same 17 images, second noise seed)", "held out", f"{p['B2']:.3f}"],
    ]
    if p.get("P") is not None:
        rows.append(["P (PixelShift200, real sensor)", "evaluation", f"{p['P']:.3f}"])
    else:
        rows.append(["P (PixelShift200, real sensor)", "-",
                     "not run: PixelShift200 absent; evaluation data, never committed"])
    out = [md(rows, ["group", "split", "mean PSNR (dB)"])]
    if c:
        out.append(f"\nArtifacts, the {c['images']} images of group A: zipper {c['zipper_pct']:.3f} % of edge pixels, "
                   f"false colour {c['false_colour_pct']:.3f} % of pixels, chroma dE00 {c['chroma_de00_mean']:.3f}.")
    e = base.get("edge")
    if e:
        out.append(f"\nSlanted edge through the demosaic (levels {e['levels'][0]:.2f} and {e['levels'][1]:.2f} of white): "
                   f"MTF50 {e['mtf50_cpp']:.4f} cycles/px, rise {e['rise_10_90_px']:.4f} px, overshoot "
                   f"{e['overshoot_pct']:.3f} %, undershoot {e['undershoot_pct']:.3f} %, flat-patch sd {e['flat_sd']:.5f}.")
    return "\n".join(out)


def ca(base):
    names = {(False, 0.5): "strength 0.5", (False, 1.0): "strength 1", (False, 2.0): "strength 2",
             (True, 1.0): "strength 1, barrel", (True, 2.0): "strength 2, barrel"}
    rows = [[names[(s["barrel"], s["strength"])], f"{s['red_rms_px']:.3f}", f"{s['blue_rms_px']:.3f}",
             f"{s['psnr_rgb_db']:.2f}"] for s in base["ca"]]
    return md(rows, ["scene (600 x 400)", "red residual RMS (px)", "blue residual RMS (px)", "PSNR, all planes (dB)"])


def spec03_table(results):
    rows = []
    for r in results["records"]:
        method = {"rl_auto": "Richardson-Lucy Auto", "diffraction": "diffraction",
                  "diffraction_gate": "diffraction, noise gate"}.get(r["method"], r["method"])
        rows.append([f"{r['f_number']:g}", r["iso"], method, f2(r["text_psnr_db"]), f2(r["natural_psnr_db"]),
                     f"{r['mtf50_cpp']:.3f}", f2(r["rise_10_90_px"]), f"{r['overshoot_pct']:.1f}",
                     f"{r['undershoot_pct']:.1f}", f"{r['flat_noise_ratio']:.3f}"])
    return md(rows, ["f", "ISO", "method", "text PSNR (dB)", "natural PSNR (dB)", "MTF50 (c/px)", "rise (px)",
                     "overshoot (%)", "undershoot (%)", "flat noise (x)"])


def spec03_cases(base):
    rows = []
    for c in base["cases"]:
        rows.append([c["name"], "held out" if c["held"] else "fit", f2(c["text_psnr_db"]), f2(c["natural_psnr_db"]),
                     f"{c['mtf50_cpp']:.4f}", f"{c['rise_10_90_px']:.3f}", f"{c['overshoot_pct']:.2f}",
                     f"{c['flat_noise_x']:.4f}"])
    return md(rows, ["case", "split", "text PSNR (dB)", "natural PSNR (dB)", "MTF50 (c/px)", "rise (px)",
                     "overshoot (%)", "flat noise (x)"])


def spec04_spec_edge(text):
    """The numbers the spec 04 section 9 edge-profile tests print (cpu_output_sharpen.rs)."""
    before = re.search(r"soft edge before sharpening: rise ([\d.]+) px", text)
    rows = []
    for m in re.finditer(r"^(?:test \w+ \.\.\. )?(screen, full size|screen, s = 0\.4|screen, halo 0, full size|print, full size)"
                         r"\s+rise ([\d.]+) px \(spec [\w.]+\), overshoot ([\d.]+) % / undershoot ([\d.]+) %", text, re.M):
        rows.append([m.group(1), before.group(1), m.group(2), f"{m.group(3)} / {m.group(4)}"])
    gains = re.findall(r"\(x([\d.]+)\) with threshold (\d) level", text)
    if not before or len(rows) != 4 or len(gains) != 2:
        raise SystemExit("spec04_edge.txt: the edge-profile output was not found")
    out = md(rows, ["case", "rise before (px)", "rise after (px)", "overshoot / undershoot (%)"])
    out += ("\n\nNoise gain on a flat patch (sigma 0.004): x" + gains[0][0] + " with the 1-level threshold, x"
            + gains[1][0] + " with threshold 0.")
    return out


def spec04_harness_edge(base):
    rows = []
    for scale in sorted(base["edge"], key=lambda k: -float(k)):
        e = base["edge"][scale]
        rows.append([scale, f"{e['rise_10_90_px']:.4f}", f"{e['overshoot_pct']:.3f}", f"{e['undershoot_pct']:.3f}",
                     f"{e['noise_gain']:.4f}"])
    return md(rows, ["scale", "rise 10-90 (px)", "overshoot (%)", "undershoot (%)", "noise gain (x)"])


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    d = argv[1]
    gate = json.load(open(os.path.join(d, "demosaic_gate", "gate.json")))
    ab = json.load(open(os.path.join(d, "capture_sharpen_ab", "results.json")))
    note, dem = read_lines(os.path.join(d, "fit_demosaic.jsonl"))
    _, dif = read_lines(os.path.join(d, "fit_diffraction.jsonl"))
    _, shp = read_lines(os.path.join(d, "fit_sharpen.jsonl"))
    edge_text = open(os.path.join(d, "spec04_edge.txt")).read()
    not_run = list(gate.get("sets_not_run", []))
    if any(s["set"] == "pixelshift200" for s in note["skipped"]) and "PixelShift200" not in not_run:
        not_run.append("PixelShift200")
    skipped = ", ".join(not_run) if not_run else "none"
    used = ", ".join(f"{u['set']} ({u['images']})" for u in note["used"])

    out = []
    out.append("Every number below was printed by `tools/fit/baseline.sh` at `Tuning::ORACLE` (`Tuning::DEFAULT` equal to it). "
               "Data: " + used + ". Not run (absent on this machine): " + skipped + "; the real-sensor group P is therefore not run: "
               "PixelShift200 absent; evaluation data, never committed. Everything here is evaluation and local fitting data, "
               "re-mosaicked 8-bit sRGB pictures and stand-in noise models, no real-sensor ground truth.")
    out.append("#### Spec 01 - adaptive ratio demosaic\n\n**The nine cells** (`demosaic_gate`: mean PSNR in linear light, 10 px border off, "
               "nominal model of spec 01 section 6.3). The R6 rule counts the best of four methods per cell.\n\n" + nine_cells(gate))
    out.append("**Zipper and false colour** (`demosaic_gate`, signal scale 1, linear light).\n\n" + artifacts(gate))
    out.append("**The noisy table** (`demosaic_gate`, Gaussian noise `a x + b`, signal scale 1, matching model for the adaptive ratio; "
               "each cell is demosaic / after the R6 bilateral denoise).\n\n" + noisy(gate))
    out.append("**The groups of the fit harness** (`fit_demosaic --held --artifacts --edge --ca`).\n\n" + groups(dem))
    out.append("**The CA path** (`ca_green_residual`'s scenes through `fit_demosaic --ca`; the residual red and blue displacement against "
               "green after R9, and the PSNR of all three planes). These scenes are ours and generated, so none is held out.\n\n" + ca(dem))
    out.append("#### Spec 03 - diffraction deconvolution\n\n**Section 6.4** (`capture_sharpen_ab`: synthetic scenes at a 3.76 um pitch; "
               "PSNR against the sharp scene, slanted-edge MTF50, 10 to 90 % rise, overshoot, flat-patch luma noise relative to the input). "
               "The scenes are ours; no part of it is held out.\n\n" + spec03_table(ab))
    out.append("**The cases of the fit harness** (`fit_diffraction --held`: the four cases of section 2.2 and the same four with a second "
               "noise seed, held out).\n\n" + spec03_cases(dif))
    out.append("#### Spec 04 - output sharpening\n\n**The edge profile of section 9** (the tests `edge_profile_screen_preset` and "
               "`edge_profile_noise_gain_on_a_flat_patch` of `cpu_output_sharpen.rs`, a soft slanted edge of Gaussian sigma 0.8 px, 0.1 to 0.5).\n\n"
               + spec04_spec_edge(edge_text))
    out.append("**The edge profile of the fit harness** (`fit_sharpen --held --edge`: a seeded logistic edge with noise, screen preset; ours, no part held out). "
               f"The change measures (the {shp['outputs']} outputs of Kodak at scales 1 and 0.33) are 0 at the base by definition; a later "
               "evaluation adds McMaster as the held-out set with `--held`.\n\n" + spec04_harness_edge(shp))
    print("\n\n".join(out))


if __name__ == "__main__":
    main(sys.argv)
