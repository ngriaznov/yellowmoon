#!/usr/bin/env python3
"""Re-derives the two fixed-point cubics of the log-guided demosaic (docs/algorithms/02 sections 3.2
and 3.4; docs/29-raw-constants.md, F-01 and RC-01).

    python3 -I tools/fit/poly_fit.py            # about a minute; prints the result, exit 0 when it reproduces the stored triples

Method (numpy only, no randomness, no network, nothing written):

1. A continuous minimax cubic (Lawson's iteration on a grid of 8001 points) gives the centre of the
   search for each function.
2. The exact integer arithmetic of `log2_fixed` and `exp2_fixed` (crates/ym-gpu/src/cpu/demosaic/
   log_guided.rs) is emulated over EVERY input (65535 values of y, 65536 values of L).
3. The integer triples around the centre are searched; the best five by the objective are printed.

The objectives (docs/29 section 2.3):

* log2: minimise the largest |L - 4096 log2 y| over y = 1 .. 65535, in codes (4096 codes per stop),
  subject to: L(1) = 0, L(2) = 4096, L(65535) = 65535; at most 23 step-backs (L(y) < L(y - 1); the
  count of the triple it replaces); no step-back larger than one code. Ties: fewer step-backs, then
  the smaller triple. The anchors of the cubic itself, p(0) = 0 and p(1) = 1, are A - B + C = 32768.
* exp2: minimise the smallest k with -(1 + k t) <= C - t <= k t, t = 2^(L/4096), over L = 0 .. 65535,
  subject to: C(4096 e) = 2^e exactly; C monotone non-decreasing; C <= 65535 (it fits in 16 bits).
  Ties: the smaller triple. The constant term of the cubic is 1.

The search boxes are printed. A candidate outside a box is not claimed to be worse; the claim is
"nothing in the box is better", which crates/ym-gpu/tests/cpu_log_guided.rs repeats on a smaller
box around the stored triple.
"""

import sys
import time

import numpy as np

# The triples the fixed-point reference ships (log_guided.rs) and the ones it replaced.
STORED_LOG2 = (46653, 19114, 5229)  # over 32768
STORED_EXP2 = (45559, 14919, 5047)  # over 65536
ORACLE_LOG2 = (46597, 19037, 5208)
ORACLE_EXP2 = (45607, 14741, 5185)

LOG2_MAX_STEP_BACKS = 23
LOG2_MAX_STEP_BACK_SIZE = 1
LOG2_BOX = (120, 60)  # +- on B and on C (A follows from A - B + C = 32768)
EXP2_POLISH_BOX = 4  # +- on each of A, B, C around the end of the climb

# ---------------------------------------------------------------------------------------------
# log2_fixed, exact integer emulation over every input
# ---------------------------------------------------------------------------------------------

Y = np.arange(1, 65536, dtype=np.int64)
E = np.floor(np.log2(Y)).astype(np.int64)
E = np.where((1 << (E + 1)) <= Y, E + 1, E)  # guard against float rounding at powers of two
E = np.where((1 << E) > Y, E - 1, E)
N = Y << (15 - E)
M = 2 * N - 65536
M2 = (M * M) >> 16
M3 = (M2 * M) >> 16
LOG2_TRUE = 4096.0 * np.log2(Y.astype(np.float64))


def log2_codes(a, b, c):
    p = ((M * a) >> 16) - ((M2 * b) >> 16) + ((M3 * c) >> 16)
    return 4096 * E + (p >> 3)


def log2_eval(a, b, c):
    """(max |error| in codes, lowest error, highest error, step-backs, largest step-back, anchors ok)."""
    code = log2_codes(a, b, c)
    err = code - LOG2_TRUE
    steps = np.diff(code)
    anchors = code[0] == 0 and code[1] == 4096 and code[-1] == 65535
    return (
        max(-err.min(), err.max()),
        err.min(),
        err.max(),
        int((steps < 0).sum()),
        int(-steps.min()) if steps.min() < 0 else 0,
        bool(anchors),
    )


# ---------------------------------------------------------------------------------------------
# exp2_fixed, exact integer emulation over every input
# ---------------------------------------------------------------------------------------------

LG = np.arange(0, 65536, dtype=np.int64)
EE = LG >> 12
FF = (LG & 4095) << 4
F2 = (FF * FF) >> 16
F3 = (F2 * FF) >> 16
EXP2_TRUE = 2.0 ** (LG / 4096.0)
SCALE = 1 << EE
POWERS = 1 << np.arange(16)


def exp2_codes(a, b, c):
    q = ((a * FF) >> 16) + ((b * F2) >> 16) + ((c * F3) >> 16)
    return SCALE + ((SCALE * q) >> 16)


def exp2_eval(a, b, c):
    """(k, upper k, lower k, largest C, monotone, exact at 2^e, fits in 16 bits)."""
    code = exp2_codes(a, b, c)
    upper = ((code - EXP2_TRUE) / EXP2_TRUE).max()
    lower = ((EXP2_TRUE - code - 1.0) / EXP2_TRUE).max()
    return (
        max(upper, lower),
        upper,
        lower,
        int(code.max()),
        bool(np.all(np.diff(code) >= 0)),
        bool(np.all(code[::4096] == POWERS)),
        bool(code.max() <= 65535),
    )


def exp2_ok(r):
    return r[4] and r[5] and r[6]


# ---------------------------------------------------------------------------------------------
# Continuous minimax by Lawson's iteration (the centres of the searches)
# ---------------------------------------------------------------------------------------------


def lawson(basis, target, iterations=400):
    w = np.ones(len(target)) / len(target)
    for _ in range(iterations):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(basis * sw[:, None], target * sw, rcond=None)
        r = np.abs(basis @ coef - target)
        w = w * r
        w /= w.sum()
    return coef, float(np.abs(basis @ coef - target).max())


def log2_seed():
    """p(f) = a f + b f^2 + c f^3 close to log2(1 + f), p(0) = 0, p(1) = 1 (a + b + c = 1)."""
    f = np.linspace(0.0, 1.0, 8001)
    basis = np.stack([f * f - f, f**3 - f], 1)
    (b, c), err = lawson(basis, np.log2(1 + f) - f)
    return (1.0 - b - c, b, c), err


def exp2_seed():
    """q(f) = a f + b f^2 + c f^3 close to 2^f - 1 in RELATIVE error, constant term 1, sum free."""
    f = np.linspace(0.0, 1.0, 8001)
    t = 2.0**f
    basis = np.stack([f / t, f**2 / t, f**3 / t], 1)
    coef, err = lawson(basis, 1.0 - 1.0 / t)
    return tuple(coef), err


# ---------------------------------------------------------------------------------------------
# The searches
# ---------------------------------------------------------------------------------------------


def search_log2():
    (a, b, c), seed_err = log2_seed()
    bc, cc = int(round(-b * 32768)), int(round(c * 32768))
    db, dc = LOG2_BOX
    rows = []
    for bi in range(bc - db, bc + db + 1):
        for ci in range(cc - dc, cc + dc + 1):
            ai = 32768 + bi - ci
            r = log2_eval(ai, bi, ci)
            feasible = (
                r[5] and r[3] <= LOG2_MAX_STEP_BACKS and r[4] <= LOG2_MAX_STEP_BACK_SIZE
            )
            if feasible:
                rows.append((r[0], r[3], (ai, bi, ci), r))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    box = f"B in [{bc - db}, {bc + db}], C in [{cc - dc}, {cc + dc}], A = 32768 + B - C"
    return rows, box, (a, b, c), seed_err


def climb_exp2(start):
    steps = [
        (da, db, dc)
        for da in (-8, -4, -2, -1, 0, 1, 2, 4, 8)
        for db in (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
        for dc in (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
        if (da, db, dc) != (0, 0, 0)
    ]
    cur = start
    best = exp2_eval(*cur)[0]
    while True:
        move = None
        for d in steps:
            cand = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            r = exp2_eval(*cand)
            if exp2_ok(r) and r[0] < best - 1e-12:
                best, move = r[0], cand
        if move is None:
            return cur
        cur = move


def search_exp2():
    coef, seed_err = exp2_seed()
    start = tuple(int(round(v * 65536)) for v in coef)
    end = climb_exp2(start)
    d = EXP2_POLISH_BOX
    rows = []
    for ai in range(end[0] - d, end[0] + d + 1):
        for bi in range(end[1] - d, end[1] + d + 1):
            for ci in range(end[2] - d, end[2] + d + 1):
                r = exp2_eval(ai, bi, ci)
                if exp2_ok(r):
                    rows.append((r[0], (ai, bi, ci), r))
    rows.sort(key=lambda x: (x[0], x[1]))
    box = (
        f"hill climb from {start} (integer steps up to 8, 16 and 16), then every triple within "
        f"+-{d} of its end {end}"
    )
    return rows, box, coef, seed_err


def main():
    t0 = time.time()
    ok = True

    print("log2_fixed: P/32768 ~ log2(1 + f), L = 4096 e + (P >> 3), every y in 1 .. 65535")
    print("objective: minimise max |L - 4096 log2 y| (codes); anchors L(1) = 0, L(2) = 4096,")
    print(f"  L(65535) = 65535; step-backs <= {LOG2_MAX_STEP_BACKS}, none larger than "
          f"{LOG2_MAX_STEP_BACK_SIZE} code; ties: fewer step-backs, smaller triple")
    rows, box, seed, seed_err = search_log2()
    print(f"continuous minimax (Lawson): a, b, c = {seed[0]:.6f} {seed[1]:.6f} {seed[2]:.6f}, "
          f"max error {seed_err:.3e} stop = {seed_err * 4096:.3f} codes")
    print(f"search box: {box} ({len(rows)} feasible candidates)")
    o = log2_eval(*ORACLE_LOG2)
    print(f"replaced triple {ORACLE_LOG2}: max |err| {o[0]:.3f} codes [{o[1]:.3f}, {o[2]:.3f}], "
          f"{o[3]} step-backs, largest {o[4]}")
    print("best five:")
    for err, drops, trip, r in rows[:5]:
        print(f"  {trip}  max |err| {err:.3f} codes  [{r[1]:.3f}, {r[2]:.3f}]  "
              f"{drops} step-backs  largest {r[4]}")
    same = rows[0][2] == STORED_LOG2
    print(f"stored triple {STORED_LOG2}: {'reproduced' if same else 'NOT reproduced'}")
    ok &= same

    print()
    print("exp2_fixed: Q/65536 ~ 2^f - 1, C = 2^e + (2^e Q >> 16), every L in 0 .. 65535")
    print("objective: minimise k with -(1 + k t) <= C - t <= k t, t = 2^(L/4096); C(4096 e) = 2^e")
    print("  exactly; C monotone non-decreasing; C <= 65535; ties: smaller triple")
    rows, box, seed, seed_err = search_exp2()
    print(f"continuous relative minimax (Lawson, constant term 1): a, b, c = "
          f"{seed[0]:.6f} {seed[1]:.6f} {seed[2]:.6f}, max relative error {seed_err:.3e}")
    print(f"search box: {box} ({len(rows)} feasible candidates)")
    o = exp2_eval(*ORACLE_EXP2)
    print(f"replaced triple {ORACLE_EXP2}: k {o[0]:.3e} (upper {o[1]:.3e}, lower {o[2]:.3e}), "
          f"max C {o[3]}")
    print("best five:")
    for k, trip, r in rows[:5]:
        print(f"  {trip}  k {k:.3e}  (upper {r[1]:.3e}, lower {r[2]:.3e})  max C {r[3]}  "
              f"monotone {r[4]}  exact at 2^e {r[5]}")
    same = rows[0][1] == STORED_EXP2
    print(f"stored triple {STORED_EXP2}: {'reproduced' if same else 'NOT reproduced'}")
    ok &= same

    print(f"\n{time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
