"""The budget rules: refusal, persistence across restarts, work counted once, the programme's cap."""

import json

import pytest

from training.bokeh import budget as B
from training.bokeh.budget import BudgetExceeded, Ledger


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_a_plan_is_the_sum_of_its_calls_at_their_time_limit_with_the_margin():
    plan = B.plan_prepare(20000)
    by_hand = sum(n * fn.timeout_s * fn.per_second * B.MARGIN for _, n, fn in plan.items)
    assert plan.worst == pytest.approx(by_hand)
    assert [n for what, n, _ in plan.items if what == "tile chunks"] == [20]
    # the L4 call: 4 cores, 16 GiB, one L4 for (minutes + overhead)
    fn = B.train_fn(100)
    assert fn.timeout_s == (100 + B.TRAIN_OVERHEAD_MIN) * 60
    assert fn.per_second == pytest.approx(0.000222 + 4 * 0.0000131 + 16 * 0.00000222)


def test_the_default_plans_fit_their_default_budgets_and_the_programme():
    assert B.plan_check().worst < 0.5
    assert B.plan_prepare(20000).worst < 1.5
    assert B.plan_preview().worst < 0.3
    assert B.plan_train(100).worst < 5.0
    assert B.plan_results().worst < 0.05
    assert B.plan_check().worst + B.plan_prepare(20000).worst + B.plan_preview().worst + B.plan_train(100).worst < B.PROGRAMME_CAP / 2


def test_max_train_minutes_is_the_longest_that_fits():
    m = B.max_train_minutes(5.0)
    assert B.plan_train(m).worst <= 5.0 < B.plan_train(m + 1).worst


def test_refuses_to_start_when_the_worst_case_is_above_the_budget(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", clock=Clock())
    with pytest.raises(BudgetExceeded, match="more than the budget"):
        ledger.reserve("a", B.plan_train(100), budget=2.0, deadline_s=7200)
    assert not (tmp_path / "ledger.json").exists() and ledger.entries == {}


def test_refuses_a_budget_above_the_cap_and_a_plan_that_passes_the_cap(tmp_path):
    clock = Clock()
    ledger = Ledger(tmp_path / "ledger.json", cap=10.0, clock=clock)
    with pytest.raises(BudgetExceeded, match="cap"):
        ledger.reserve("a", B.plan_check(), budget=11.0, deadline_s=900)
    for i in range(2):  # two training runs of about $3.9 each: reserved in full
        ledger.reserve(f"t{i}", B.plan_train(140), budget=5.0, deadline_s=9000)
    assert ledger.spent() == pytest.approx(2 * B.plan_train(140).worst)
    with pytest.raises(BudgetExceeded, match="spent or reserved"):
        ledger.reserve("t2", B.plan_train(140), budget=5.0, deadline_s=9000)


def test_the_count_survives_a_restart_and_a_restarted_run_carries_on_its_entry(tmp_path):
    clock = Clock()
    path = tmp_path / "ledger.json"
    first = Ledger(path, clock=clock)
    plan = B.plan_prepare(3000)
    first.reserve("prep", plan, budget=1.5, deadline_s=5400)
    first.charge_once("prep", "index", 100, B.INDEX)
    first.charge_once("prep", "tiles-0", 200, B.TILES)
    first.save()
    # the orchestrator is restarted: a new process reads the file and reserves the same run id
    second = Ledger(path, clock=clock)
    assert second.entries["prep"]["actual"] == pytest.approx(first.entries["prep"]["actual"])
    again = second.reserve("prep", plan, budget=1.5, deadline_s=5400)
    assert again["actual"] == pytest.approx(first.entries["prep"]["actual"])
    # one entry only: the spend is not counted twice
    assert second.spent() == pytest.approx(plan.worst)
    assert len(second.entries) == 1


def test_work_is_charged_once(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", clock=Clock())
    ledger.reserve("p", B.plan_prepare(2000), budget=1.5, deadline_s=5400)
    c1 = ledger.charge_once("p", "tiles-0", 100, B.TILES)
    c2 = ledger.charge_once("p", "tiles-0", 100, B.TILES)  # the restarted run collects the same chunk again
    assert c1 == pytest.approx(B.TILES.cost(100)) and c2 == 0.0
    assert ledger.entries["p"]["actual"] == pytest.approx(c1)


def test_a_dead_run_counts_in_full_until_its_deadline_then_at_what_was_recorded_plus_what_may_still_run(tmp_path):
    clock = Clock()
    ledger = Ledger(tmp_path / "ledger.json", clock=clock)
    plan = B.plan_prepare(20000)
    ledger.reserve("dead", plan, budget=1.5, deadline_s=5400)
    ledger.charge_once("dead", "index", 300, B.INDEX)
    assert ledger.spent() == pytest.approx(plan.worst)
    clock.t += 5400 + 1
    after = ledger.spent()
    assert after == pytest.approx(min(plan.worst, B.INDEX.cost(300) + plan.inflight))
    assert after <= plan.worst + 1e-5


def test_a_dead_training_run_is_settled_by_its_heartbeat(tmp_path):
    clock = Clock()
    ledger = Ledger(tmp_path / "ledger.json", clock=clock)
    plan = B.plan_train(100)
    fn = plan.items[0][2]
    ledger.reserve("t", plan, budget=5.0, deadline_s=plan.items[1][2].timeout_s, slack=fn.cost(600))
    # the worker, in another process, beats three times (50 minutes) and dies
    worker = Ledger(tmp_path / "ledger.json", clock=clock)
    worker.write_heartbeat("t", 50 * 60, fn)
    clock.t += plan.items[1][2].timeout_s + 1
    got = Ledger(tmp_path / "ledger.json", clock=clock).spent()
    assert got == pytest.approx(fn.cost(50 * 60) + fn.cost(600))
    assert got < plan.worst / 2


def test_settling_counts_the_actual_cost_and_a_settled_id_cannot_be_reused(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", clock=Clock())
    plan = B.plan_check()
    ledger.reserve("c", plan, budget=0.5, deadline_s=900)
    ledger.charge("c", "probe", 120, B.PROBE)
    cost = ledger.settle("c")
    assert cost == pytest.approx(B.PROBE.cost(120)) and ledger.spent() == pytest.approx(cost)
    with pytest.raises(ValueError, match="settled"):
        ledger.reserve("c", plan, budget=0.5, deadline_s=900)


def test_check_stops_a_run_whose_estimate_passed_its_budget(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", clock=Clock())
    ledger.reserve("t", B.plan_prepare(1000), budget=1.5, deadline_s=5400)
    ledger.charge("t", "tiles", 3600 * 20, B.TILES)  # 20 hours of one call: far more than the plan allowed
    with pytest.raises(BudgetExceeded, match="passed the budget"):
        ledger.check("t", "tiles")


def test_the_file_is_written_whole(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", clock=Clock())
    ledger.reserve("c", B.plan_check(), budget=0.5, deadline_s=900)
    assert json.loads((tmp_path / "ledger.json").read_text())["entries"]["c"]["state"] == "open"
    assert not list(tmp_path.glob("*.tmp"))


def test_the_results_entry_is_refused_over_its_budget_and_never_by_the_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", cap=0.001, clock=Clock())
    with pytest.raises(BudgetExceeded):
        ledger.record_small("r", B.plan_results(), budget=0.001, seconds=30, fn=B.RESULTS)
    cost = ledger.record_small("r", B.plan_results(), budget=0.05, seconds=30, fn=B.RESULTS)
    assert cost < 0.01 and ledger.entries["r"]["state"] == "settled"


def test_modal_app_builds_its_functions_from_the_plans():
    pytest.importorskip("modal")
    from training.bokeh import modal_app

    assert modal_app.TRAIN_STATIC.timeout_s >= B.train_fn(100).timeout_s
    for name in ("check", "prepare", "preview", "train", "results", "fetch", "plan"):
        assert callable(getattr(modal_app, name))


def test_a_failed_run_lapses_soon_instead_of_at_its_whole_time_limit(tmp_path):
    clock = Clock()
    ledger = Ledger(tmp_path / "ledger.json", clock=clock)
    plan = B.plan_prepare(20000)
    ledger.reserve("p", plan, budget=1.5, deadline_s=5400)
    ledger.charge_once("p", "index", 300, B.INDEX)
    ledger.expire_soon("p", grace_s=120)
    assert ledger.spent() == pytest.approx(plan.worst)
    clock.t += 121
    assert ledger.spent() == pytest.approx(min(plan.worst, B.INDEX.cost(300) + plan.inflight))
    # a second reserve of the same id (a restart) gives it a new lifetime
    ledger.reserve("p", plan, budget=1.5, deadline_s=5400)
    assert ledger.spent() == pytest.approx(plan.worst)


def test_a_restarted_training_run_is_charged_for_the_time_its_dead_attempt_trained(tmp_path):
    clock = Clock()
    path = tmp_path / "ledger.json"
    plan = B.plan_train(100)
    fn = plan.items[0][2]
    first = Ledger(path, clock=clock)
    first.reserve("t", plan, budget=5.0, deadline_s=plan.items[1][2].timeout_s)
    first.write_heartbeat("t", 40 * 60, fn)  # the worker trained 40 minutes, then everything died
    second = Ledger(path, clock=clock)  # the lead starts the run again with the same id
    second.reserve("t", plan, budget=5.0, deadline_s=plan.items[1][2].timeout_s)
    assert second.absorb_heartbeat("t", fn) == pytest.approx(fn.cost(40 * 60 + 300))
    assert second.absorb_heartbeat("t", fn) == 0.0  # once
    second.charge("t", "training", 60 * 60, fn)  # the new attempt trains the remaining hour
    assert second.settle("t") == pytest.approx(fn.cost(40 * 60 + 300) + fn.cost(3600))


def test_the_gpu_smoke_and_the_full_run_fit_the_budgets_the_workflow_gives_them():
    # .github/workflows/bokeh-train.yml: ENTRY=train, ARGS="--budget 0.8 --minutes 8 --steps 1000 --run bokeh-smoke", then
    # ARGS="--budget 5 --minutes 100 --run bokeh-2"
    assert B.plan_train(8).worst <= 0.8
    assert B.plan_train(100).worst <= 5.0
    assert B.plan_train(8).worst + B.plan_train(100).worst < B.PROGRAMME_CAP / 2
