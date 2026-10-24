"""What the bokeh programme may spend on Modal, and the ledger that keeps count.

The owner's decision (2026-10-24): "Yes, use Modal for the bokeh map, cap $10". About $11 is left of the account's hard
$30 limit (`docs/STATUS.md`). The whole programme, every entrypoint and every run together, stays at or under
`PROGRAMME_CAP` by this module's own estimate: Modal's published prices (`training.crop.budget`), the time each call
takes, and a 25 % margin. Modal's own Workspace budget (Usage & Billing) is the hard cap above all this.

Rules, all enforced here and tested (`tests/test_budget.py`):

1. **Refuse to start.** Every entrypoint first plans its worst case: every call it may make, at its time limit, at
   Modal's rates, times the margin (`Plan.worst`). It starts only if the worst case is at most its `--budget` and the
   programme's spend so far plus that worst case is at most `PROGRAMME_CAP`.
2. **Reserve, then settle.** Starting writes a reservation (the worst case) into a ledger file on the volume. As calls
   finish, their actual cost (seconds x rate x margin) is added to the reservation's `actual`; at the end the entry is
   settled at its actual cost. A run that dies leaves its reservation open: it counts in full until the run's deadline
   (its time limit) has passed, and then at what was actually recorded plus a bound on what could still have been
   running (for a training run, the worker's own heartbeat file instead). A restarted orchestrator with the same run
   id carries on the same entry; it does not start from zero (the crop pilot's lesson, 2026-10-23: a restart began a
   new count and $5 were spent twice).
3. **Work is counted once.** Each unit of work (a mask archive, a chunk of tiles) has an item key; `charge_once`
   charges an item a single time, however often a restarted run collects its result, and the work itself skips what
   the volume already holds.
4. **Stop cleanly.** `Run.check` raises `BudgetExceeded` when the entry's actual spend passes the budget; the
   orchestrators call it after every call and cancel what is running.

    python -m training.bokeh.budget plan --entry train --minutes 100     # prints the worst case, spends nothing
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from training.crop.budget import CPU_CORE_PER_S, L4_PER_S, MARGIN, MEM_GIB_PER_S

#: The owner's cap for the whole bokeh programme, in dollars.
PROGRAMME_CAP = 10.0
#: Modal charges a non-preemptible container more; the orchestrators run with `nonpreemptible=True` (they must not be
#: restarted half way) and this covers the difference (an assumption, on the safe side; the crop pilot used it too).
NONPREEMPTIBLE = 3.0


@dataclass(frozen=True)
class Fn:
    """One kind of Modal call: its resources and its time limit."""

    cpu: float
    memory_mib: int
    timeout_s: int
    gpu: bool = False
    nonpreemptible: bool = False

    @property
    def per_second(self) -> float:
        """Dollars per second of one container (before the margin)."""
        r = (L4_PER_S if self.gpu else 0.0) + self.cpu * CPU_CORE_PER_S + self.memory_mib / 1024 * MEM_GIB_PER_S
        return r * (NONPREEMPTIBLE if self.nonpreemptible else 1.0)

    def cost(self, seconds: float) -> float:
        """What `seconds` of this call cost, with the margin."""
        return seconds * self.per_second * MARGIN

    @property
    def worst(self) -> float:
        """One call that runs to its time limit."""
        return self.cost(self.timeout_s)


#: The calls of the programme. `modal_app.py` builds its function decorators from these, so a plan is what runs.
PROBE = Fn(cpu=2, memory_mib=8192, timeout_s=600)
INDEX = Fn(cpu=2, memory_mib=8192, timeout_s=1200)
MASKS = Fn(cpu=2, memory_mib=4096, timeout_s=600)
TILES = Fn(cpu=2, memory_mib=8192, timeout_s=600)
PREVIEW = Fn(cpu=4, memory_mib=16384, timeout_s=900)
RESULTS = Fn(cpu=1, memory_mib=2048, timeout_s=300)
#: Setup (image start, copying the tiles from the volume), then training for `minutes`, then the final evaluation,
#: the export and the previews: `TRAIN_OVERHEAD_MIN` minutes on top of the training time are inside the time limit.
TRAIN_OVERHEAD_MIN = 18
#: A run may state a smaller overhead (`--overhead-min`; a smoke run scores a handful of samples). Measured in bokeh-2: the
#: tiles were copied in 10 s and the final evaluation of 576 samples took 75 s. Never below this, so that the call's time
#: limit leaves room for the setup, the evaluation and the export after the training.
MIN_OVERHEAD_MIN = 6
ORCH_OVERHEAD_MIN = 7
#: Photos per tile-building call and the number of mask archives (Open Images splits them by the id's first hex digit).
TILE_CHUNK = 1000
MASK_ZIPS = 16
#: Calls running at once (Modal's Starter plan allows more; this keeps the tail of a failed run short).
CONCURRENCY = 16


def train_fn(minutes: float, overhead_min: float = TRAIN_OVERHEAD_MIN) -> Fn:
    """The training call for `minutes` of training plus `overhead_min`: one L4, four cores, 16 GiB."""
    if overhead_min < MIN_OVERHEAD_MIN:
        raise ValueError(f"an overhead of {overhead_min} minutes is below the {MIN_OVERHEAD_MIN} the setup and the final evaluation need")
    return Fn(cpu=4, memory_mib=16384, gpu=True, timeout_s=int((minutes + overhead_min) * 60))


def orchestrator(timeout_s: int) -> Fn:
    """The small non-preemptible function that runs an entrypoint."""
    return Fn(cpu=0.25, memory_mib=1024, timeout_s=timeout_s, nonpreemptible=True)


@dataclass
class Plan:
    """The calls an entrypoint may make: `(what, number of calls, Fn)`. `worst` is all of them at their time limit."""

    entry: str
    items: list[tuple[str, int, Fn]]

    @property
    def worst(self) -> float:
        return sum(n * fn.worst for _, n, fn in self.items)

    @property
    def inflight(self) -> float:
        """The most that can be running at once, at the limit: a bound on what a dead run can still be spending."""
        biggest = max(fn.worst for _, _, fn in self.items)
        return min(self.worst, CONCURRENCY * biggest)

    def table(self) -> str:
        rows = [f"{self.entry}: worst case ${self.worst:.2f} (every call at its time limit, margin {MARGIN})"]
        for what, n, fn in self.items:
            rows.append(f"  {what:<28} {n:>3} x {fn.timeout_s / 60:>5.0f} min  ${fn.worst:.3f} each")
        return "\n".join(rows)


def plan_check() -> Plan:
    """The dry check: one small CPU call and the orchestrator that waits for it."""
    return Plan("check", [("probe", 1, PROBE), ("orchestrator", 1, orchestrator(PROBE.timeout_s + 300))])


def plan_prepare(photos: int) -> Plan:
    """Index the photos and the masks, fetch the mask archives, build the tiles of `photos` photos."""
    chunks = max(1, math.ceil(photos / TILE_CHUNK))
    return Plan("prepare", [
        ("index", 1, INDEX),
        ("mask archives", MASK_ZIPS, MASKS),
        ("tile chunks", chunks, TILES),
        ("orchestrator", 1, orchestrator(90 * 60)),
    ])


def plan_preview() -> Plan:
    return Plan("preview", [("preview", 1, PREVIEW), ("orchestrator", 1, orchestrator(PREVIEW.timeout_s + 300))])


def plan_train(minutes: float, overhead_min: float = TRAIN_OVERHEAD_MIN) -> Plan:
    return Plan("train", [
        ("training (one L4)", 1, train_fn(minutes, overhead_min)),
        ("orchestrator", 1, orchestrator(int((minutes + overhead_min + ORCH_OVERHEAD_MIN) * 60))),
    ])


def plan_results() -> Plan:
    return Plan("results", [("results", 1, RESULTS)])


def max_train_minutes(budget: float) -> float:
    """The longest `minutes` whose `plan_train` worst case fits in `budget`."""
    lo, hi = 0.0, 24 * 60.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if plan_train(mid).worst <= budget else (lo, mid)
    return lo


class BudgetExceeded(RuntimeError):
    """A plan does not fit its budget or the programme's cap, or a run's estimated spend passed its budget."""


class Ledger:
    """The programme's spend, one entry per run, in a JSON file on the volume (see the module docstring)."""

    def __init__(self, path: Path, cap: float = PROGRAMME_CAP, clock=time.time):
        self.path = Path(path)
        self.cap = cap
        self.clock = clock
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())["entries"]

    def save(self) -> None:
        """Writes the file whole through a temporary file, so a crash leaves the old or the new one, never half."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cap": self.cap, "entries": self.entries}, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def heartbeat_path(self, key: str) -> Path:
        return self.path.with_name(f"{self.path.stem}.{key}.hb.json")

    def entry_cost(self, e: dict, key: str = "", now: float | None = None) -> float:
        """What one entry counts for now (the module docstring's rule 2)."""
        now = self.clock() if now is None else now
        if e["state"] == "settled":
            return e["actual"]
        if now < e["deadline"]:
            return max(e["reserved"], e["actual"])
        hb = self.heartbeat_path(key)
        if key and hb.exists():
            seen = json.loads(hb.read_text())
            return min(e["reserved"], max(e["actual"], seen["cost"]) + e["slack"])
        return min(e["reserved"], e["actual"] + e["inflight"])

    def spent(self, now: float | None = None) -> float:
        """Everything the programme has spent or may still be spending, in dollars (an estimate, margin included)."""
        return sum(self.entry_cost(e, k, now) for k, e in self.entries.items())

    def reserve(self, key: str, plan: Plan, budget: float, deadline_s: float, slack: float = 0.0) -> dict:
        """Starts (or, with a key already open, carries on) a run. Raises `BudgetExceeded` when `plan.worst` is above
        `budget`, or the programme's spend plus the part of the plan not yet reserved is above the cap. `deadline_s` is
        the run's time limit in seconds from now. `slack` is the cost of the time a dead run's last heartbeat may lag."""
        now = self.clock()
        if key in self.entries:
            if self.entries[key]["state"] == "open":
                self.entries[key]["deadline"] = now + deadline_s  # a restarted orchestrator has a new lifetime
                return self.entries[key]
            raise ValueError(f"the run id {key!r} was used and settled; use another")
        if budget > self.cap:
            raise BudgetExceeded(f"--budget ${budget:.2f} is above the programme's cap ${self.cap:.2f}")
        if plan.worst > budget:
            raise BudgetExceeded(
                f"{plan.entry} could cost up to ${plan.worst:.2f} (every call at its time limit), more than the budget "
                f"${budget:.2f}\n{plan.table()}")
        spent = self.spent(now)
        if spent + plan.worst > self.cap:
            raise BudgetExceeded(
                f"{plan.entry} could cost up to ${plan.worst:.2f}; the programme has spent or reserved ${spent:.2f} "
                f"of ${self.cap:.2f}")
        self.entries[key] = {
            "entry": plan.entry, "state": "open", "reserved": round(plan.worst, 6), "actual": 0.0,
            "budget": budget, "started": now, "deadline": now + deadline_s, "inflight": round(plan.inflight, 6),
            "slack": round(slack, 6), "items": [], "done": [],
        }
        self.save()
        return self.entries[key]

    def record_small(self, key: str, plan: Plan, budget: float, seconds: float, fn: Fn) -> float:
        """For the one-cent entrypoints (`results`): refuses when the plan's worst case is above `budget`, then writes an
        entry settled at the cost of `seconds`. It does not reserve against the programme's cap, so that reading the results
        is never refused because the cap has been reached."""
        if plan.worst > budget:
            raise BudgetExceeded(f"{plan.entry} could cost up to ${plan.worst:.3f}, more than the budget ${budget:.3f}")
        cost = fn.cost(seconds)
        self.entries[key] = {"entry": plan.entry, "state": "settled", "reserved": round(plan.worst, 6), "actual": round(cost, 6),
                             "budget": budget, "started": self.clock(), "deadline": self.clock(), "inflight": 0.0,
                             "slack": 0.0, "items": [[plan.entry, round(cost, 5)]], "done": []}
        self.save()
        return cost

    def charge(self, key: str, what: str, seconds: float, fn: Fn) -> float:
        """Adds `seconds` of a call of kind `fn` (with the margin) to the run's actual spend; returns the cost."""
        e = self.entries[key]
        cost = fn.cost(seconds)
        e["actual"] = round(e["actual"] + cost, 6)
        e["items"].append([what, round(cost, 5)])
        return cost

    def charge_once(self, key: str, item: str, seconds: float, fn: Fn) -> float:
        """`charge`, but a single time per `item` key: a restarted run that collects the same result again pays 0."""
        e = self.entries[key]
        if item in e["done"]:
            return 0.0
        e["done"].append(item)
        return self.charge(key, item, seconds, fn)

    def check(self, key: str, stage: str) -> None:
        """Saves, and raises `BudgetExceeded` when the run's actual spend is past its budget."""
        self.save()
        e = self.entries[key]
        print(f"{stage}: spent ${e['actual']:.3f} of ${e['budget']:.2f} (estimate); programme "
              f"${self.spent():.2f} of ${self.cap:.2f}", flush=True)
        if e["actual"] > e["budget"]:
            raise BudgetExceeded(f"estimated spend ${e['actual']:.2f} passed the budget ${e['budget']:.2f} at {stage}")

    def expire_soon(self, key: str, grace_s: float = 120.0) -> None:
        """For a run that failed: its reservation stops counting in full after `grace_s` (then the rule for a dead run
        applies: what was recorded plus a bound on what can still be running) instead of at the run's whole time limit."""
        e = self.entries[key]
        if e["state"] == "open":
            e["deadline"] = min(e["deadline"], self.clock() + grace_s)
            self.save()

    def absorb_heartbeat(self, key: str, fn: Fn, extra_s: float = 300.0) -> float:
        """For a restarted training run: charges, once, the time its earlier attempt's last heartbeat recorded (plus
        `extra_s` for the start and what ran since the beat), then removes the file. Returns the cost."""
        hb = self.heartbeat_path(key)
        if not hb.exists():
            return 0.0
        seconds = json.loads(hb.read_text())["seconds"] + extra_s
        cost = self.charge_once(key, f"previous-attempt-{int(seconds)}", seconds, fn)
        hb.unlink()
        return cost

    def settle(self, key: str) -> float:
        """Closes the run at its actual cost; returns it."""
        e = self.entries[key]
        e["state"] = "settled"
        self.save()
        return e["actual"]

    def write_heartbeat(self, key: str, seconds: float, fn: Fn) -> None:
        """For a worker: records how long it has run (the cost so far, with the margin), so that a run that dies is
        settled at that instead of at the full reservation. Written by the worker, a file of its own."""
        hb = self.heartbeat_path(key)
        hb.parent.mkdir(parents=True, exist_ok=True)
        tmp = hb.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seconds": seconds, "cost": fn.cost(seconds)}))
        os.replace(tmp, hb)

    def report(self) -> dict:
        return {"cap": self.cap, "spent_or_reserved": round(self.spent(), 3),
                "entries": {k: {"entry": e["entry"], "state": e["state"], "reserved": e["reserved"],
                                "actual": round(e["actual"], 4)} for k, e in self.entries.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prints a plan's worst case. Spends nothing.")
    ap.add_argument("cmd", choices=["plan"])
    ap.add_argument("--entry", choices=["check", "prepare", "preview", "train", "results"], required=True)
    ap.add_argument("--photos", type=int, default=20000)
    ap.add_argument("--minutes", type=float, default=100.0)
    ap.add_argument("--overhead-min", type=float, default=TRAIN_OVERHEAD_MIN)
    args = ap.parse_args()
    plan = {"check": plan_check, "prepare": lambda: plan_prepare(args.photos), "preview": plan_preview,
            "train": lambda: plan_train(args.minutes, args.overhead_min), "results": plan_results}[args.entry]()
    print(plan.table())
    if args.entry == "train":
        print(f"the longest training that fits $5.00: {max_train_minutes(5.0):.0f} minutes")


if __name__ == "__main__":
    main()
