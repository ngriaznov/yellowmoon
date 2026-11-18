#!/usr/bin/env python3
"""The search driver of the fit harness (docs/29-raw-constants.md section 2.3, task RC-05).

Standard library only. It does no image processing: it asks the examples `fit_demosaic`,
`fit_diffraction` and `fit_sharpen` (crates/ym-gpu/examples/) to evaluate `Tuning` overrides, reads the
JSON line each evaluation prints and picks the best by an objective. Deterministic: the grid is fixed by
its arguments, coordinate descent visits its candidates in a fixed order and breaks ties toward the
earlier one, and the examples are deterministic, so the same command gives the same output.

    tools/fit/search.py grid    fit_demosaic --objective psnr_db.A --axis chroma_floor=0.01:0.12:0.01
    tools/fit/search.py descent fit_demosaic --objective psnr_db.A --axis chroma_floor=0.03:0.005 \\
                                --axis sat_dominance=0.75:0.05 --bounds chroma_floor=0:0.2
    tools/fit/search.py table   demosaic < lines.jsonl      # the audit's table format

The objective is a dotted path into the line of an evaluation (`psnr_db.A`, `d_psnr_db.B`,
`edge.mtf50_cpp`, `mean_delta.text_psnr_db`); `--minimize` flips the direction. `--constraint
PATH<=VALUE` (or `>=`) rejects an evaluation that breaks it (the failed-gate rule, docs/29 section 2.3:
a value that raises the noise gain or the overshoot is not taken). A value is changed only if it beats the
start by `--min-gain` (docs/29 section 2.3 rule 3: 0.01 dB for a PSNR objective).

Everything after `--` goes to the example unchanged (`--held`, `--photo FILE.ARW`, `--artifacts`,
`--ca`, ...). The data is read from testdata/ (`YM_TESTDATA` moves it), is for evaluation and local fitting,
and is never committed; every example prints a data note first and this driver passes it on.

Set `FIT_BIN` to a built example binary to skip cargo (`FIT_BIN=target/debug/examples/fit_demosaic-<hash>`).
"""

import argparse
import json
import math
import os
import subprocess
import sys

EXAMPLES = ("fit_demosaic", "fit_diffraction", "fit_sharpen")
# Evaluations per process: the scenes are loaded once per process, the command line stays short.
CHUNK = 24


def tidy(v):
    """12 significant digits, as the examples do (1.6 * 0.9 is 1.44)."""
    return float(f"{v:.11e}")


def fmt(v):
    """The shortest text that reads back as v (a value on a command line)."""
    return repr(float(v))


def command(example, extra):
    binary = os.environ.get("FIT_BIN")
    if binary:
        return [binary] + extra
    return [
        "cargo", "run", "-q", "-p", "ym-gpu", "--example", example,
        "--config", "profile.dev.package.ym-gpu.opt-level=3", "--",
    ] + extra


def run_example(example, evaluations, passthrough, start, echo):
    """Runs the example once for `evaluations` (a list of (label, {field: value})).

    Returns (data_note, base_line, {label: line}). Raises on a failure of the example.
    """
    args = ["--from", start]
    for label, sets in evaluations:
        args += ["--eval", label + "@" + ",".join(f"{k}={fmt(v)}" for k, v in sets.items())]
    args += passthrough
    proc = subprocess.run(command(example, args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{example} failed with status {proc.returncode}")
    note, base, lines = None, None, {}
    for text in proc.stdout.splitlines():
        text = text.strip()
        if not text.startswith("{"):
            continue
        line = json.loads(text)
        if echo:
            print(text)
        if line.get("kind") == "data_note":
            note = line
        elif line.get("kind") == "eval":
            if line["label"] == "base" and base is None and not line.get("set"):
                base = line
            else:
                lines[line["label"]] = line
    return note, base, lines


def lookup(line, path):
    """The number at the dotted `path` of an evaluation line, or None."""
    cur = line
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def parse_constraint(text):
    for op in ("<=", ">="):
        if op in text:
            path, value = text.split(op, 1)
            return path.strip(), op, float(value)
    raise SystemExit(f"constraint `{text}` needs <= or >=")


def satisfies(line, constraints):
    for path, op, bound in constraints:
        v = lookup(line, path)
        if v is None:
            return False
        if op == "<=" and v > bound:
            return False
        if op == ">=" and v < bound:
            return False
    return True


def score(line, objective, minimize, constraints):
    """Larger is better; None when the line is unusable or breaks a constraint."""
    v = lookup(line, objective)
    if v is None or not satisfies(line, constraints):
        return None
    return -v if minimize else v


def axis_values(text):
    """`name=start:stop:step` (inclusive) or `name=v1,v2,...`."""
    name, spec = text.split("=", 1)
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        if step <= 0 or stop < start:
            raise SystemExit(f"axis `{text}`: need step > 0 and stop >= start")
        n = int(math.floor((stop - start) / step + 1e-9)) + 1
        return name, [tidy(start + i * step) for i in range(n)]
    return name, [float(x) for x in spec.split(",")]


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def evaluate_all(example, candidates, passthrough, start, echo):
    """Runs every candidate (a dict field -> value); returns (note, base, [line per candidate])."""
    evaluations = [(f"c{i}", c) for i, c in enumerate(candidates)]
    note, base, lines = None, None, [None] * len(candidates)
    for batch in chunks(list(enumerate(evaluations)), CHUNK):
        n, b, got = run_example(example, [e for _, e in batch], passthrough, start, echo)
        note, base = n, base or b
        for i, (label, _) in batch:
            lines[i] = got.get(label)
    return note, base, lines


def cmd_grid(a):
    names, values = [], []
    for text in a.axis:
        name, vs = axis_values(text)
        names.append(name)
        values.append(vs)
    candidates = [{}]
    for name, vs in zip(names, values):
        candidates = [dict(c, **{name: v}) for c in candidates for v in vs]
    print(f"# grid of {len(candidates)} evaluations over {', '.join(names)}", file=sys.stderr)
    note, base, lines = evaluate_all(a.example, candidates, a.passthrough, a.start, a.echo)
    constraints = [parse_constraint(c) for c in a.constraint]
    start_score = score(base, a.objective, a.minimize, constraints) if base else None
    best = None
    for cand, line in zip(candidates, lines):
        s = score(line, a.objective, a.minimize, constraints) if line else None
        if s is not None and (best is None or s > best[0]):
            best = (s, cand, line)
    report(a, note, base, best, start_score)


def cmd_descent(a):
    constraints = [parse_constraint(c) for c in a.constraint]
    bounds = {}
    for text in a.bounds:
        name, spec = text.split("=", 1)
        lo, hi = (float(x) for x in spec.split(":"))
        bounds[name] = (lo, hi)
    current, steps = {}, {}
    for text in a.axis:
        name, spec = text.split("=", 1)
        v, step = (float(x) for x in spec.split(":"))
        current[name], steps[name] = v, step
    note, base, lines = evaluate_all(a.example, [dict(current)], a.passthrough, a.start, a.echo)
    cur_line = lines[0]
    cur = score(cur_line, a.objective, a.minimize, constraints) if cur_line else None
    if cur is None:
        raise SystemExit("the starting point is unusable or breaks a constraint")
    start_score = cur
    evaluations = 1
    for sweep in range(a.max_sweeps):
        moved = False
        for name in current:
            step = steps[name]
            lo, hi = bounds.get(name, (-math.inf, math.inf))
            tries = [tidy(current[name] + k * step) for k in (-2, -1, 1, 2)]
            tries = [t for t in tries if lo <= t <= hi]
            cands = [dict(current, **{name: t}) for t in tries]
            if not cands:
                continue
            _, _, got = evaluate_all(a.example, cands, a.passthrough, a.start, a.echo)
            evaluations += len(cands)
            best = None
            for cand, line in zip(cands, got):
                s = score(line, a.objective, a.minimize, constraints) if line else None
                if s is not None and s > cur + a.min_gain and (best is None or s > best[0]):
                    best = (s, cand, line)
            if best:
                cur, current, cur_line, moved = best[0], best[1], best[2], True
                print(f"# sweep {sweep} {name} -> {current[name]} objective {lookup(cur_line, a.objective)}", file=sys.stderr)
        if not moved:
            # Halve the steps; stop when every step is under the minimum.
            steps = {k: v / 2 for k, v in steps.items()}
            if all(v < a.min_step for v in steps.values()):
                break
    print(f"# {evaluations} evaluations", file=sys.stderr)
    report(a, note, base, (cur, current, cur_line), start_score)


def report(a, note, base, best, start_score):
    if note:
        print(json.dumps(note, sort_keys=True))
    if best is None:
        print(json.dumps({"kind": "search", "example": a.example, "result": "no usable evaluation"}))
        return
    s, cand, line = best
    sign = -1 if a.minimize else 1
    gain = None if start_score is None else sign * (s - start_score)
    print(json.dumps({
        "kind": "search", "example": a.example, "objective": a.objective,
        "direction": "minimize" if a.minimize else "maximize",
        "best": cand, "best_value": lookup(line, a.objective),
        "start_value": lookup(base, a.objective) if base else None,
        "gain_over_start": gain, "min_gain": a.min_gain,
        "taken": gain is not None and gain > a.min_gain,
        "line": line,
    }, sort_keys=True))


# ---------------------------------------------------------------------------------------------
# table: the audit's text format, for comparing a run with docs/29 section 2.2
# ---------------------------------------------------------------------------------------------

def table_demosaic(lines):
    for line in lines:
        if line["label"] == "base":
            p = line["psnr_db"]
            print(f"BASE psnr A {p['A']:.3f} B {p['B']:.3f}")
            continue
        d, c = line["d_psnr_db"], line["change_codes"]
        text = f"{line['label']}\tdPSNR A {d['A']:+.4f} B {d['B']:+.4f}"
        for g in ("A", "B", "C"):
            if g in c:
                text += f"\t{g}: mean|d| {c[g]['mean_codes']:.4f} max|d| {c[g]['max_codes']:.1f}"
        print(text)


def table_diffraction(lines):
    for line in lines:
        if line["label"] == "base":
            for case in line["cases"]:
                print(f"BASE {case['name']}: text {case['text_psnr_db']:.2f} nat {case['natural_psnr_db']:.2f} "
                      f"mtf50 {case['mtf50_cpp']:.3f} over {case['overshoot_pct']:.1f}% noise {case['flat_noise_x']:.3f}")
            continue
        m = line["mean_delta"]
        print(f"{line['label']}\ttext {m['text_psnr_db']:+.3f} dB\tnat {m['natural_psnr_db']:+.3f} dB\t"
              f"mtf50 {m['mtf50_cpp']:+.4f}\tover {m['overshoot_pct']:+.2f}%\tnoise {m['flat_noise_x']:+.4f}\t"
              f"max|dout| {line['max_dout']:.2f} (8-bit-of-0.5)")


def table_sharpen(lines):
    for line in lines:
        if line["label"] == "base":
            print(f"BASE ok n={line['outputs']}")
            continue
        text = line["label"]
        for scale in line["change_codes"]:
            c = line["change_codes"][scale]
            text += f"\tscale {scale}: mean|d| {c['mean_codes']:.4f} max|d| {c['max_codes']:.1f}"
        print(text)


def cmd_table(a):
    lines = []
    for text in sys.stdin:
        text = text.strip()
        if text.startswith("{"):
            line = json.loads(text)
            if line.get("kind") == "eval":
                lines.append(line)
    {"demosaic": table_demosaic, "diffraction": table_diffraction, "sharpen": table_sharpen}[a.kind](lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("grid", cmd_grid), ("descent", cmd_descent)):
        s = sub.add_parser(name, help=f"{name} search over Tuning fields")
        s.add_argument("example", choices=EXAMPLES)
        s.add_argument("--objective", required=True, help="dotted path into an evaluation line")
        s.add_argument("--minimize", action="store_true")
        s.add_argument("--constraint", action="append", default=[], help="PATH<=VALUE or PATH>=VALUE")
        s.add_argument("--axis", action="append", required=True,
                       help="grid: name=start:stop:step or name=v1,v2; descent: name=start:step")
        s.add_argument("--from", dest="start", choices=("oracle", "default"), default="oracle")
        s.add_argument("--echo", action="store_true", help="print every JSON line the example prints")
        s.add_argument("--min-gain", type=float, default=0.01,
                       help="the objective must beat the start by this much (default 0.01)")
        if name == "descent":
            s.add_argument("--bounds", action="append", default=[], help="name=lo:hi")
            s.add_argument("--min-step", type=float, default=1e-3)
            s.add_argument("--max-sweeps", type=int, default=8)
        s.add_argument("passthrough", nargs="*", help="after --: arguments for the example")
        s.set_defaults(fn=fn)
    t = sub.add_parser("table", help="print JSON lines from stdin in the audit's table format")
    t.add_argument("kind", choices=("demosaic", "diffraction", "sharpen"))
    t.set_defaults(fn=cmd_table)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
