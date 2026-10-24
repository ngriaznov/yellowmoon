"""The Modal worker's behaviour when a run fails (no Modal, no network: the function is called as plain Python)."""

import json

import pytest

pytest.importorskip("modal")

from training.bokeh import budget as B  # noqa: E402
from training.bokeh import modal_app, train  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*executing locally")


class FakeVolume:
    def __init__(self):
        self.commits = 0

    def reload(self):
        pass

    def commit(self):
        self.commits += 1


def _run_train_run(tmp_path, monkeypatch, fake_train):
    volume = FakeVolume()
    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app, "volume", volume)
    monkeypatch.setattr(modal_app.shutil, "copytree", lambda *a, **k: None)  # the tiles are not needed by the fake
    monkeypatch.setattr(train, "train", fake_train)
    return volume, modal_app.train_run.local  # Function.local runs the function in this process


def test_a_run_that_diverges_keeps_its_log_and_reaches_the_orchestrator_as_a_plain_runtime_error(tmp_path, monkeypatch):
    def fake_train(tiles, out, config, heartbeat=None, log=print):
        log(json.dumps({"step": 25, "loss": 0.93}))
        log(json.dumps({"step": 3, "skipped": "non-finite loss", "where": ["synthesis lin"]}))
        raise train.TrainingDiverged("21 steps in a row were skipped for a non-finite value")

    volume, raw = _run_train_run(tmp_path, monkeypatch, fake_train)
    with pytest.raises(RuntimeError, match="^training diverged: 21 steps in a row") as err:
        raw("train-1", "bokeh-x", {"max_minutes": 5.0, "steps": 100})
    assert type(err.value) is RuntimeError  # not the subclass: the orchestrator's image has no torch to unpickle it with
    log = (tmp_path / "runs" / "bokeh-x" / "log.txt").read_text().splitlines()
    assert json.loads(log[0])["loss"] == 0.93 and "skipped" in json.loads(log[1])  # written and closed
    assert volume.commits >= 1  # and committed to the volume, so that the fetch step finds it


def test_any_other_failure_keeps_the_log_too_and_stays_the_error_that_was_raised(tmp_path, monkeypatch):
    def fake_train(tiles, out, config, heartbeat=None, log=print):
        log("before the crash")
        raise ValueError("boom")

    volume, raw = _run_train_run(tmp_path, monkeypatch, fake_train)
    with pytest.raises(ValueError, match="boom"):
        raw("train-2", "bokeh-y", {"max_minutes": 5.0})
    assert (tmp_path / "runs" / "bokeh-y" / "log.txt").read_text() == "before the crash\n" and volume.commits >= 1


class FakeCall:
    def __init__(self, error=None, result=None):
        self.error, self.result, self.cancelled = error, result, False

    def get(self, timeout=None):
        if self.error:
            raise self.error
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled = True


class FakeTrainRun:
    def __init__(self, call):
        self.call = call

    def spawn(self, *args):
        return self.call


def test_the_ledger_settles_at_the_actual_cost_when_the_training_call_fails(tmp_path, monkeypatch):
    call = FakeCall(error=RuntimeError("training diverged: 21 steps in a row were skipped for a non-finite value"))
    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app, "volume", FakeVolume())
    monkeypatch.setattr(modal_app, "train_run", FakeTrainRun(call))
    with pytest.raises(RuntimeError, match="training stopped: RuntimeError: training diverged: 21 steps in a row"):
        modal_app.run_train.local("train-3", "bokeh-z", 8.0, 0.8, {"steps": 1000})
    assert call.cancelled
    entry = B.Ledger(tmp_path / "ledger.json").entries["train-3"]
    plan = B.plan_train(8.0)
    assert entry["state"] == "settled"  # not left open until its deadline
    assert entry["actual"] < 0.05 * plan.worst  # the call lived for a moment, so it cost a moment: not the reservation
    assert entry["reserved"] == pytest.approx(plan.worst)


def test_the_ledger_settles_even_when_the_budget_check_fails_after_a_failed_call(tmp_path, monkeypatch):
    call = FakeCall(error=RuntimeError("boom"))
    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app, "volume", FakeVolume())
    monkeypatch.setattr(modal_app, "train_run", FakeTrainRun(call))
    monkeypatch.setattr(B.Ledger, "check", lambda self, key, stage: (_ for _ in ()).throw(B.BudgetExceeded("over")))
    with pytest.raises(B.BudgetExceeded):
        modal_app.run_train.local("train-4", "bokeh-w", 8.0, 0.8, {})
    assert B.Ledger(tmp_path / "ledger.json").entries["train-4"]["state"] == "settled"
