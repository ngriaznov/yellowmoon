"""The bokeh map's training on Modal (`research/bokeh-map-2026-10-24.md`; the owner, 2026-10-24: "Yes, use Modal for the
bokeh map, cap $10").

Needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment (never in the repository). Modal's client speaks gRPC,
which the agents' cloud proxy does not carry, so it runs from GitHub Actions (`.github/workflows/bokeh-train.yml`,
started by a push to the branch `bokeh-train-run`) or from a computer with direct internet. From the repository root:

    modal run -m training.bokeh.modal_app::check   --budget 0.5     # volumes and photos reachable? times a few photos
    modal run -m training.bokeh.modal_app::prepare --budget 1.5     # index masks, fetch the archives, build the tiles
    modal run -m training.bokeh.modal_app::preview --budget 0.3     # sheets of synthesised inputs and labels
    modal run -m training.bokeh.modal_app::train   --budget 0.8 --minutes 8 --steps 1000 --run bokeh-smoke   # the GPU smoke
    modal run -m training.bokeh.modal_app::train   --budget 5 --minutes 100 --run bokeh-2
    modal run -m training.bokeh.modal_app::results                  # metrics of every run, the ledger
    modal run -m training.bokeh.modal_app::fetch                    # runs, previews, credits, ledger into training/bokeh/runs/
    python -m training.bokeh.budget plan --entry train --minutes 100   # a plan's worst case, no Modal, no spend

**Budget** (`budget.py`). Every entrypoint takes `--budget`, plans its worst case (every call at its time limit, Modal's
prices, a 25 % margin), and refuses to start when that is above `--budget` or when the programme's spend so far plus it
would pass $10. The spend is kept in `ledger.json` on the volume across restarts: a reservation when the run starts,
the actual cost of each finished call added, a settlement at the end; a run that dies stays reserved until its deadline.
The orchestrators are non-preemptible (a restarted orchestrator is what doubled the crop pilot's spend) and stop the
run, cancelling what is in flight, when the estimate passes the budget. Modal's Workspace budget is the hard cap above all
this.

**Data.** The photos are the ones the crop pipeline already put on the volume `yellowmoon-crop`, read in place (that
volume is mounted read-only); the masks come from Open Images' official archives into the volume `yellowmoon-bokeh`
(`prepare.py`, `sources.py`). There is no `synthesize` step: the synthetic scenes are rendered on the training GPU, batch
by batch (`synth.py`; a pre-rendered set would be hundreds of gigabytes), and `preview` shows what they look like.

Volume `yellowmoon-bokeh`: `ledger.json`, `index/photos.json`, `masks/`, `tiles/` (shards, `credits.tsv.gz`,
`tiles.json`), `previews/`, `runs/<run>/` (`model.pt`, `bokehnet.safetensors`, `parity.json`, `config.json`,
`metrics.json`, `log.txt`, `samples/`, `ckpt.pt` while training), `check.json`.
"""

import contextlib
import json
import shutil
import time
from pathlib import Path

import modal

from training.bokeh import budget as B
from training.bokeh.budget import BudgetExceeded, Ledger, plan_check, plan_prepare, plan_preview, plan_results, plan_train

ROOT = Path("/vol")
CROP = Path("/crop")
TRAIN_STATIC_MINUTES = B.max_train_minutes(5.0)
#: The training function's own time limit, a backstop: the loop stops itself at `minutes`, and the orchestrator stops
#: the call at the run's own limit (`B.train_fn(minutes).timeout_s`), which is what the plan's worst case counts.
TRAIN_STATIC = B.train_fn(TRAIN_STATIC_MINUTES)

app = modal.App("yellowmoon-bokeh")
volume = modal.Volume.from_name("yellowmoon-bokeh", create_if_missing=True)
# The crop pipeline's volume: it must exist already (never created here) and is mounted read-only.
crop_volume = modal.Volume.from_name("yellowmoon-crop").read_only()
base = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6", "pillow==12.2.0", "scipy==1.15.3")
cpu_image = base.add_local_python_source("training")
torch_image = base.pip_install("torch==2.6.0").add_local_python_source("training")


def spec(fn: B.Fn) -> dict:
    return {"cpu": fn.cpu, "memory": fn.memory_mib, "timeout": fn.timeout_s}


VOLS = {"/vol": volume, "/crop": crop_volume}


@app.function(image=cpu_image, volumes=VOLS, **spec(B.PROBE))
def probe() -> dict:
    """The dry check: can the crop volume's photos be read and filtered, are the Open Images files reachable."""
    from training.bokeh import prepare, sources

    t0 = time.time()
    out = prepare.probe(CROP)
    import urllib.request

    reach = {}
    for name, url in (("segmentation_csv", sources.SEG_URL.format(subset="train")),
                      ("mask_archive_0", sources.MASKS_URL.format(subset="train", c="0"))):
        try:
            req = urllib.request.Request(url, headers={**sources.AGENT, "Range": "bytes=0-99"})
            with urllib.request.urlopen(req, timeout=60) as r:
                reach[name] = r.status
        except Exception as e:  # noqa: BLE001 - reported
            reach[name] = f"{type(e).__name__}: {e}"
    out["open_images"] = reach
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "check.json").write_text(json.dumps(out, indent=1))
    volume.commit()
    print(json.dumps(out, indent=1), flush=True)
    return out


@app.function(image=cpu_image, volumes=VOLS, **spec(B.INDEX))
def index_photos(photos: int, extra: int, seed: int) -> dict:
    """Lists the photos and reads their instance masks' names out of Open Images' segmentation CSV (streamed)."""
    from training.bokeh import prepare, sources

    volume.reload()
    summary = prepare.build_index(CROP, ROOT, photos, extra, seed, lambda: sources.stream_lines(sources.SEG_URL.format(subset="train")))
    volume.commit()
    return summary


@app.function(image=cpu_image, volumes=VOLS, **spec(B.MASKS), max_containers=B.CONCURRENCY)
def mask_archive(c: str) -> dict:
    """The masks of the indexed photos from archive `train-masks-<c>.zip` (385 MB, downloaded and deleted)."""
    from training.bokeh import prepare

    t0 = time.time()
    volume.reload()
    n = prepare.mask_archive(ROOT, c)
    volume.commit()
    return {"c": c, "masks": n, "seconds": time.time() - t0}


@app.function(image=cpu_image, volumes=VOLS, **spec(B.TILES), max_containers=B.CONCURRENCY)
def tile_chunk(k: int) -> dict:
    """Chunk `k` of the photos (1,000) as tiles."""
    from training.bokeh import prepare

    t0 = time.time()
    volume.reload()
    res = prepare.tile_chunk(ROOT, CROP, k, workers=2)
    volume.commit()
    res["call_seconds"] = time.time() - t0
    return res


@app.function(image=torch_image, volumes=VOLS, **spec(B.PREVIEW))
def preview_sheets() -> dict:
    from training.bokeh import previews

    volume.reload()
    paths = previews.render(ROOT / "tiles", ROOT / "previews", sheets=4, rows=6, seed=100, device="cpu")
    volume.commit()
    return {"sheets": [p.name for p in paths]}


@app.function(image=torch_image, gpu="L4", volumes=VOLS, cpu=B.train_fn(1).cpu, memory=B.train_fn(1).memory_mib,
              timeout=TRAIN_STATIC.timeout_s)
def train_run(run_id: str, run: str, config: dict) -> dict:
    """Trains one model on a local copy of the tiles; writes `runs/<run>/` (module docstring). Continues from the
    checkpoint of an earlier attempt of the same run."""
    import torch

    from training.bokeh import export, train
    from training.bokeh.model import BokehNet

    t_call = time.time()
    volume.reload()
    out = ROOT / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        return {**json.loads((out / "metrics.json").read_text()), "call_seconds": time.time() - t_call, "reused": True}
    local = Path("/tmp/tiles")
    shutil.copytree(ROOT / "tiles", local, ignore=shutil.ignore_patterns("reports", "*.tsv.gz"), dirs_exist_ok=True)
    setup = time.time() - t_call
    print(f"tiles copied in {setup:.0f} s", flush=True)
    ledger = Ledger(ROOT / "ledger.json")
    log_file = (out / "log.txt").open("a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    def heartbeat(trained_seconds: float) -> None:
        ledger.write_heartbeat(run_id, setup + trained_seconds + 60, B.train_fn(config["max_minutes"]))
        volume.commit()

    def keep_log() -> None:
        """For a run that failed or diverged: its log must reach the volume (the error raised stays the one raised)."""
        log_file.close()
        with contextlib.suppress(Exception):
            volume.commit()

    try:
        metrics = train.train(local, out, {**config, "device": "cuda", "workers": 3}, heartbeat=heartbeat, log=log)
        model = BokehNet()
        model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=False)["state_dict"])
        report = export.write(model, out / "bokehnet.safetensors", out / "parity.json", source=run)
        log(json.dumps({"export": report}))
    except train.TrainingDiverged as e:
        keep_log()
        # The orchestrator's image has no torch to rebuild a subclass with; a plain RuntimeError carries the message across.
        raise RuntimeError(str(e)) from None
    except BaseException:
        keep_log()
        raise
    log_file.close()
    (out / "ckpt.pt").unlink(missing_ok=True)
    volume.commit()
    return {**{k: v for k, v in metrics.items() if k != "history"}, "export": report, "call_seconds": time.time() - t_call}


# --- orchestrators ------------------------------------------------------------------------------------------------


class Run:
    """One entrypoint's entry in the ledger: reserve at the start, charge each finished call, settle at the end."""

    def __init__(self, run_id: str, plan: B.Plan, budget: float, deadline_s: float):
        volume.reload()
        self.t0 = time.time()
        self.id, self.plan = run_id, plan
        self.ledger = Ledger(ROOT / "ledger.json")
        print(plan.table(), flush=True)
        if run_id in self.ledger.entries:
            print(f"resumed: {self.ledger.entries[run_id]['actual']:.2f} already spent by an earlier attempt (estimate)", flush=True)
        self.ledger.reserve(run_id, plan, budget, deadline_s)
        self.commit()

    def commit(self) -> None:
        self.ledger.save()
        volume.commit()

    def charge(self, item: str, seconds: float, fn: B.Fn, once: bool = True) -> None:
        (self.ledger.charge_once if once else self.ledger.charge)(*((self.id, item, seconds, fn)))

    def check(self, stage: str) -> None:
        try:
            self.ledger.check(self.id, stage)
        finally:
            volume.commit()

    @contextlib.contextmanager
    def guard(self):
        """On any failure, lets the reservation lapse soon (`Ledger.expire_soon`) and passes the error on."""
        try:
            yield
        except BaseException:
            self.ledger.expire_soon(self.id)
            self.commit()
            raise

    def done(self) -> float:
        """Charges the orchestrator's own time, settles the entry, returns its cost."""
        orch_fn = next((f for what, _, f in self.plan.items if what == "orchestrator"), None)
        if orch_fn is not None:
            self.ledger.charge(self.id, "orchestrator", time.time() - self.t0 + 30, orch_fn)
        cost = self.ledger.settle(self.id)
        volume.commit()
        print(f"settled at ${cost:.3f}; programme ${self.ledger.spent():.2f} of ${self.ledger.cap:.2f}", flush=True)
        return cost


def orch(plan: B.Plan) -> dict:
    """Decorator arguments of an orchestrator: the plan's own `orchestrator` item."""
    fn = next(f for what, _, f in plan.items if what == "orchestrator")
    return {**spec(fn), "nonpreemptible": True}


@app.function(image=cpu_image, volumes=VOLS, **orch(plan_check()))
def run_check(run_id: str, budget: float) -> dict:
    run = Run(run_id, plan_check(), budget, plan_check().items[-1][2].timeout_s)
    with run.guard():
        t = time.time()
        out = probe.remote()
        run.charge("probe", time.time() - t + 20, B.PROBE)
        run.check("probe")
    run.done()
    return out


@app.function(image=cpu_image, volumes=VOLS, **orch(plan_prepare(20000)))
def run_prepare(run_id: str, photos: int, extra: int, seed: int, budget: float) -> dict:
    """Index, mask archives, tile chunks, credits. Refuses when the worst case for the photos the volume holds (or
    `photos`) is above `budget`; skips what the volume already has and charges every call once."""
    from training.bokeh import prepare

    volume.reload()
    known = prepare.known_summary(ROOT, photos, extra, seed)
    n_plan = known["photos"] if known else (photos or 20000) + extra
    plan = plan_prepare(n_plan)
    run = Run(run_id, plan, budget, plan.items[-1][2].timeout_s)
    with run.guard():
        summary = known
        if summary is None:  # the index is made once; a rerun with the same parameters reads it
            t = time.time()
            summary = index_photos.remote(photos, extra, seed)
            run.charge("index", time.time() - t + 20, B.INDEX)
            run.check("index")
        print(json.dumps(summary), flush=True)
        if summary["chunks"] > plan_prepare(n_plan).items[2][1]:
            raise BudgetExceeded(f"{summary['chunks']} chunks are more than the plan counted ({plan.items[2][1]})")

        volume.reload()
        todo = [c for c in prepare.HEX if summary["masks_per_archive"][c] and not (ROOT / "masks" / f"{c}.json").exists()]
        print(f"mask archives to fetch: {len(todo)}", flush=True)
        t = time.time()
        for got in mask_archive.map(todo, order_outputs=False, return_exceptions=True):
            if isinstance(got, Exception):
                run.charge(f"masks-failed-{time.time():.0f}", B.MASKS.timeout_s, B.MASKS, once=False)
                print(f"a mask archive failed: {type(got).__name__}: {got}", flush=True)
            else:
                run.charge(f"masks-{got['c']}", got["seconds"] + 30, B.MASKS)
                print(f"masks {got['c']}: {got['masks']}", flush=True)
            run.check("masks")
        volume.reload()
        chunks = [k for k in range(summary["chunks"]) if not (ROOT / "tiles" / f"shard-{k:05d}.json").exists()]
        print(f"tile chunks to build: {len(chunks)} of {summary['chunks']}", flush=True)
        for got in tile_chunk.map(chunks, order_outputs=False, return_exceptions=True):
            if isinstance(got, Exception):
                run.charge(f"tiles-failed-{time.time():.0f}", B.TILES.timeout_s, B.TILES, once=False)
                print(f"a tile chunk failed: {type(got).__name__}: {got}", flush=True)
            else:
                run.charge(f"tiles-{got['k']}", got["call_seconds"] + 30, B.TILES)
                print(json.dumps(got), flush=True)
            run.check("tiles")
        volume.reload()
        missing = [k for k in range(summary["chunks"]) if not (ROOT / "tiles" / f"shard-{k:05d}.json").exists()]
        final = prepare.finalize(ROOT)
        volume.commit()
    cost = run.done()
    return {"tiles": final, "chunks_missing": missing, "spend": cost}


@app.function(image=cpu_image, volumes=VOLS, **orch(plan_preview()))
def run_preview(run_id: str, budget: float) -> dict:
    run = Run(run_id, plan_preview(), budget, plan_preview().items[-1][2].timeout_s)
    with run.guard():
        t = time.time()
        out = preview_sheets.remote()
        run.charge("preview", time.time() - t + 20, B.PREVIEW)
        run.check("preview")
    run.done()
    return out


@app.function(image=cpu_image, volumes=VOLS, **orch(plan_train(B.max_train_minutes(5.0))))
def run_train(run_id: str, run_name: str, minutes: float, budget: float, config: dict) -> dict:
    """Trains for `minutes` (plus setup and the final evaluation, `B.TRAIN_OVERHEAD_MIN`). Refuses when the plan's worst
    case is above `budget` or the programme's cap; stops the call when its time is up; a rerun with the same
    `run_id` and `run_name` carries on from the checkpoint and the same ledger entry."""
    if minutes > TRAIN_STATIC_MINUTES:
        raise BudgetExceeded(f"{minutes} minutes is above the {TRAIN_STATIC_MINUTES:.0f} the orchestrator's time limit allows; "
                             "raise TRAIN_STATIC_MINUTES's budget in modal_app.py (with the owner's agreement)")
    plan = plan_train(minutes)
    fn = plan.items[0][2]
    resumed = run_id in Ledger(ROOT / "ledger.json").entries
    run = Run(run_id, plan, budget, plan.items[-1][2].timeout_s)
    if resumed:
        print(f"the earlier attempt's heartbeat is charged: ${run.ledger.absorb_heartbeat(run_id, fn):.2f}", flush=True)
        run.commit()
    t = time.time()
    call = train_run.spawn(run_id, run_name, {**config, "max_minutes": minutes})
    try:
        result = call.get(timeout=fn.timeout_s)
    except Exception as e:  # noqa: BLE001 - whatever stopped the call, it is stopped and counted at its full time
        call.cancel(terminate_containers=True)
        run.charge("training (stopped)", time.time() - t + 60, fn, once=False)
        try:
            run.check("training failed")
        finally:
            run.done()  # settled at what the call actually cost (a diverged run stops early), not at the reservation
        raise RuntimeError(f"training stopped: {type(e).__name__}: {e}") from e
    run.charge("training", time.time() - t + 60, fn, once=False)
    try:
        run.check("training")
    finally:
        run.done()
    return result


@app.function(image=cpu_image, volumes={"/vol": volume}, **spec(B.RESULTS))
def results_fn(run_id: str, budget: float) -> dict:
    """Every run's metrics (without the per-step history) and the ledger. Refuses when its own worst case is above
    `budget` (it costs a cent)."""
    volume.reload()
    ledger = Ledger(ROOT / "ledger.json")
    runs = {}
    for m in sorted((ROOT / "runs").glob("*/metrics.json")):
        runs[m.parent.name] = {k: v for k, v in json.loads(m.read_text()).items() if k != "history"}
    ledger.record_small(run_id, plan_results(), budget, 30, B.RESULTS)
    volume.commit()
    out = {"runs": runs, "ledger": ledger.report(), "check": json.loads((ROOT / "check.json").read_text()) if (ROOT / "check.json").exists() else None}
    print(json.dumps(out, indent=1), flush=True)
    return out


# --- local entrypoints ----------------------------------------------------------------------------------------------


def _id(entry: str, run_id: str) -> str:
    return run_id or f"{entry}-{int(time.time())}"


@app.local_entrypoint()
def check(budget: float = 0.5, run_id: str = "") -> None:
    """The dry check: volumes and photos reachable, a few photos filtered and timed. Spends a few cents."""
    print(json.dumps(run_check.remote(_id("check", run_id), budget), indent=1))


@app.local_entrypoint()
def prepare(budget: float = 1.5, photos: int = 0, extra: int = 0, seed: int = 1, run_id: str = "") -> None:
    """Builds the tiles from the crop volume's photos (`photos` 0 = all of them; `extra` more from the mirror)."""
    print(json.dumps(run_prepare.remote(_id("prepare", run_id), photos, extra, seed, budget), indent=1))


@app.local_entrypoint()
def preview(budget: float = 0.3, run_id: str = "") -> None:
    """Sheets of synthesised inputs and labels from the tiles (CPU)."""
    print(json.dumps(run_preview.remote(_id("preview", run_id), budget), indent=1))


@app.local_entrypoint()
def train(budget: float = 5.0, run: str = "bokeh-2", minutes: float = 100.0, steps: int = 0, batch: int = 24,
          sample: int = 512, lr: float = 2e-3, seed: int = 1, amp: bool = True, run_id: str = "") -> None:
    """Trains one model on one L4 for `minutes` of training (the worst case counts the whole call), or for `steps` steps
    when that comes first (`--steps 1000 --minutes 8 --budget 0.8 --run bokeh-smoke` is the GPU smoke: the full
    configuration for 1000 steps, about 6 minutes, to see a finite, falling loss before the long run). A name that was
    used before carries on from that run's checkpoint: a run that diverged needs a new one. The run stops by itself
    when its loss turns NaN or stays high (`train.py`, "Guards"). `--no-amp` trains in float32 instead of bf16 (slower;
    for the case that the smoke shows skipped steps with a non-finite gradient)."""
    config = {"batch": batch, "sample": sample, "lr": lr, "seed": seed, "amp": amp, **({"steps": steps} if steps else {})}
    print(json.dumps(run_train.remote(_id("train", run_id), run, minutes, budget, config), indent=1))


@app.local_entrypoint()
def results(budget: float = 0.05, run_id: str = "") -> None:
    """Every run's metrics and the ledger. Spends about a cent."""
    results_fn.remote(_id("results", run_id), budget)


@app.local_entrypoint()
def fetch(dest: str = "training/bokeh/runs", budget: float = 0.0) -> None:
    """Copies the small files of the volume (runs' weights, metrics, samples and logs, previews, credits, the ledger and
    the check) into `dest`. No Modal compute, no cost; `budget` is accepted for uniformity."""
    out = Path(dest)
    wanted = []
    for top in ("runs", "previews", "tiles"):
        try:
            entries = volume.listdir(top, recursive=True)
        except Exception as e:  # noqa: BLE001 - nothing there yet
            print(f"{top}: {type(e).__name__}: {e}")
            continue
        for e in entries:
            rel = Path(e.path.lstrip("/"))
            rel = rel if rel.parts[:1] == (top,) else Path(top) / rel
            name = rel.name
            if top == "runs" and (name in ("model.pt", "config.json", "metrics.json", "log.txt", "parity.json", "bokehnet.safetensors") or rel.suffix == ".png"):
                wanted.append(rel)
            elif top == "previews" and rel.suffix == ".png":
                wanted.append(rel)
            elif top == "tiles" and name in ("credits.tsv.gz", "tiles.json"):
                wanted.append(rel)
    wanted += [Path("ledger.json"), Path("check.json")]
    for rel in wanted:
        local = out / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            with local.open("wb") as fh:
                for part in volume.read_file(rel.as_posix()):
                    fh.write(part)
            print(f"{rel} -> {local}")
        except Exception as e:  # noqa: BLE001 - a missing file is reported, the rest still copied
            local.unlink(missing_ok=True)
            print(f"{rel}: {type(e).__name__}: {e}")


@app.local_entrypoint()
def plan(entry: str = "train", minutes: float = 100.0, photos: int = 20000) -> None:
    """Prints a plan's worst case. Spends nothing."""
    p = {"check": plan_check, "prepare": lambda: plan_prepare(photos), "preview": plan_preview,
         "train": lambda: plan_train(minutes), "results": plan_results}[entry]()
    print(p.table())
