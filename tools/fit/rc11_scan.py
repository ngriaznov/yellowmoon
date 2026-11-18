"""RC-11 scan driver (docs/29 RC-11): evaluates Tuning overrides on the section 6.4 gate (capture_sharpen_ab), the eight cases of
fit_diffraction (4 fit + 4 held), the Kodak crops and the lamps scene, and judges each against the base."""
import json, os, subprocess, sys
# Usage: tools/fit/rc11_scan.py NAME v1,v2,...   (one Tuning field of diffraction, a grid of values).
# Needs the release binaries of the two examples (cargo build --release -p ym-gpu --example fit_diffraction
# --example capture_sharpen_ab) in FIT_BIN_DIR (default target/release/examples), the Kodak set under
# YM_TESTDATA/demosaic/kodak, and writes a scratch folder next to the binaries. Evaluation only.
D = os.environ.get("FIT_BIN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release", "examples"))
GKEYS = ["text_psnr_db", "natural_psnr_db", "mtf50_cpp", "rise_10_90_px", "overshoot_pct", "undershoot_pct", "flat_noise_ratio"]
DIR = {"text_psnr_db": 1, "natural_psnr_db": 1, "mtf50_cpp": 1, "rise_10_90_px": -1, "overshoot_pct": -1, "undershoot_pct": -1, "flat_noise_ratio": -1}
CKEYS = {"text_psnr_db": 1, "natural_psnr_db": 1, "mtf50_cpp": 1, "rise_10_90_px": -1, "overshoot_pct": -1, "flat_noise_x": -1}

def gate(sets, frm="oracle"):
    args = [D + "/capture_sharpen_ab", os.environ.get("TMPDIR", "/tmp") + "/rc11_csab", "--jsonl", "--from", frm]
    for k, v in sets.items(): args += ["--set", f"{k}={v!r}"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    recs = json.loads(out.strip().splitlines()[-1])["records"]
    return {(r["f_number"], r["iso"], r["method"]): r for r in recs}

def cases(list_of_sets, frm="oracle"):
    """one fd process for all `list_of_sets`; returns [(cases_by_name, kodak, lamps)]"""
    args = [D + "/fit_diffraction", "--from", frm, "--held", "--kodak", "--lamps"]
    for i, sets in enumerate(list_of_sets):
        args += ["--eval", f"e{i}@" + ",".join(f"{k}={v!r}" for k, v in sets.items())] if sets else ["--eval", f"e{i}@sigma_min=0.2"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    res = []
    for line in out.splitlines():
        if not line.startswith("{"): continue
        d = json.loads(line)
        if d.get("kind") != "eval" or d["label"] == "base": continue
        res.append(({c["name"]: c for c in d["cases"]}, d["kodak"]["psnr_db"], d["lamps"]["psnr_db"]))
    return res

def evaluate(list_of_sets, frm="oracle"):
    cs = cases(list_of_sets, frm)
    return [(gate(s, frm), c[0], c[1], c[2]) for s, c in zip(list_of_sets, cs)]

def judge(base, new, tol=1e-9):
    """returns dict: worse/better counts on each group and the mean changes"""
    out = {}
    g_rows = []
    for k in sorted(base[0]):
        if k[2] not in ("diffraction", "diffraction_gate") or k not in new[0]: continue
        for m in GKEYS: g_rows.append((k, m, (new[0][k][m] - base[0][k][m]) * DIR[m]))
    c_rows = {"fit": [], "held": []}
    for name in base[1]:
        grp = "held" if "seed 2" in name else "fit"
        for m in CKEYS: c_rows[grp].append((name, m, (new[1][name][m] - base[1][name][m]) * CKEYS[m]))
    out["gate_worse"] = sum(1 for r in g_rows if r[2] < -tol)
    out["gate_better"] = sum(1 for r in g_rows if r[2] > tol)
    out["fit_worse"] = sum(1 for r in c_rows["fit"] if r[2] < -tol)
    out["fit_better"] = sum(1 for r in c_rows["fit"] if r[2] > tol)
    out["held_worse"] = sum(1 for r in c_rows["held"] if r[2] < -tol)
    out["held_better"] = sum(1 for r in c_rows["held"] if r[2] > tol)
    out["kodak"] = new[2] - base[2]
    out["lamps"] = new[3] - base[3]
    def mean(rows, m):
        sub = [r[2] for r in rows if r[1] == m]
        return sum(sub) / len(sub)
    out["g"] = {m: mean(g_rows, m) for m in GKEYS}
    out["gate_worst"] = {m: min(r[2] for r in g_rows if r[1] == m) for m in GKEYS}
    out["fit"] = {m: mean(c_rows["fit"], m) for m in CKEYS}
    return out

def dominates(j, tol_db=0.0):
    return j["gate_worse"] == 0 and j["fit_worse"] == 0 and j["held_worse"] == 0 and j["kodak"] >= -tol_db and j["lamps"] >= -tol_db

def line(label, j):
    return (f"{label:34s} gate +{j['gate_better']:3d}/-{j['gate_worse']:3d} fit +{j['fit_better']}/-{j['fit_worse']} held +{j['held_better']}/-{j['held_worse']} "
            f"kodak {j['kodak']:+.4f} lamps {j['lamps']:+.4f} | gate mean: tP {j['g']['text_psnr_db']:+.3f} nP {j['g']['natural_psnr_db']:+.3f} "
            f"MTF {j['g']['mtf50_cpp']:+.4f} ov {j['g']['overshoot_pct']:+.3f} nz {j['g']['flat_noise_ratio']:+.4f}" + ("  DOMINATES" if dominates(j) else ""))

def scan(name, values, extra=None, frm="oracle", base_sets=None):
    base_sets = base_sets or {}
    pts = [dict(base_sets, **(extra or {}), **{name: v}) for v in values]
    allr = evaluate([dict(base_sets)] + pts, frm)
    base, res = allr[0], allr[1:]
    out = []
    for v, r in zip(values, res):
        j = judge(base, r)
        out.append((v, j))
        print(line(f"{name}={v}", j), flush=True)
    return out

if __name__ == "__main__":
    name = sys.argv[1]
    values = [float(x) for x in sys.argv[2].split(",")]
    scan(name, values)
