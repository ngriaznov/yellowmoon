"""What the crop pilot may spend on Modal. The owner's limit (2026-10-23): the account holds $30, the Starter plan's
monthly credits, and nothing past it without their permission. The pilot keeps its own estimate of what it has
spent, from Modal's published prices and how long each call took, and labels only as many photos as the budget
leaves room for once the training is paid for. Modal's dashboard is the real bill; this keeps the run well inside it.
"""

import json

#: Modal's prices on 2026-10-23 (modal.com/pricing), in dollars per second.
L4_PER_S = 0.000222
CPU_CORE_PER_S = 0.0000131
MEM_GIB_PER_S = 0.00000222
#: Every estimate is multiplied by this: cold starts, image pulls, rounding, idle containers before they scale down.
MARGIN = 1.25


def rate(gpu: bool, cpu: float, memory_gib: float) -> float:
    """Dollars per second of one container with an L4 (or none), `cpu` physical cores and `memory_gib` of memory."""
    return (L4_PER_S if gpu else 0.0) + cpu * CPU_CORE_PER_S + memory_gib * MEM_GIB_PER_S


class Ledger:
    """The run's estimated spend, item by item."""

    def __init__(self, budget: float):
        self.budget = budget
        self.spent = 0.0
        self.items: list[tuple[str, float]] = []

    def charge(self, what: str, seconds: float, per_second: float) -> float:
        """Adds `seconds` of a container at `per_second` (with the margin); returns the cost."""
        cost = seconds * per_second * MARGIN
        self.spent += cost
        self.items.append((what, round(cost, 4)))
        return cost

    def left(self) -> float:
        return self.budget - self.spent

    def report(self) -> dict:
        return {"budget": self.budget, "spent_estimate": round(self.spent, 2), "items": self.items}

    def save(self, path) -> None:
        """Writes the spend so far, so that a restarted run (Modal retries a preempted input) carries it on."""
        path.write_text(json.dumps({"spent": self.spent, "items": self.items}))

    @classmethod
    def resume(cls, budget: float, path) -> "Ledger":
        """A ledger that starts from the spend a previous attempt of the same run saved at `path`, if any."""
        ledger = cls(budget)
        if path.exists():
            saved = json.loads(path.read_text())
            ledger.spent = float(saved["spent"])
            ledger.items = [tuple(i) for i in saved["items"]]
        return ledger


def photos_to_label(left: float, reserve: float, seconds_per_photo: float, per_second: float, wanted: int,
                    gpus: int, max_hours: float, chunk: int) -> int:
    """How many more photos to label, in whole chunks: no more than `wanted`, no more than `left - reserve` dollars
    pay for, and no more than `gpus` labellers finish in `max_hours`."""
    if seconds_per_photo <= 0:
        raise ValueError("seconds_per_photo must be positive")
    by_money = (left - reserve) / (seconds_per_photo * per_second * MARGIN)
    by_time = max_hours * 3600 * gpus / seconds_per_photo
    n = int(max(0.0, min(wanted, by_money, by_time)))
    return n if n == wanted else n // chunk * chunk


def training_reserve(runs: int, max_minutes: float, per_second: float, setup_minutes: float = 10) -> float:
    """The most the training runs can cost: each stops at `max_minutes`, plus `setup_minutes` to start and copy."""
    return runs * (max_minutes + setup_minutes) * 60 * per_second * MARGIN

