"""The crop pilot on Modal (`research/crop-models-2026-10-23.md`, owner's decision of 2026-10-23): Open Images
photos, FrameLite's crops as labels, and the student trained at several training-set sizes (a learning curve).

Needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment (never in the repository). Modal's client speaks
gRPC, which the agents' cloud proxy does not carry, so it runs from GitHub Actions (`.github/workflows/crop-pilot.yml`,
started by a push to the branch `crop-pilot-run`) or from a computer with direct internet. From the repository root:

    modal run -m training.crop.modal_app::pilot --n 20000 --budget 20   # select, download, label, build, train
    modal run -m training.crop.modal_app::results                       # the runs' metrics
    modal run -m training.crop.modal_app::fetch                         # runs and labels into training/crop/runs/

**Budget.** The owner's limit is $30 on the Modal account, never more without their permission. `--budget` is what one
run may spend by its own estimate (`budget.py`: Modal's prices, how long each call took, a 25 % margin). It labels a
first chunk on one GPU, measures the seconds per photo, and labels only as many photos as the budget leaves room for
once the training (each run stopped at `--max-minutes`) is paid for; it stops if the estimate passes the budget. Run
it without `--detach`, so the app stops when the client does.

Everything lands in the Modal volume `yellowmoon-crop`: `lists/` (Open Images' image lists), `data/<name>/`
(`images/`, `manifest.json` with each photo's author and licence, `labels/` per chunk, `labels.json`, `arrays/`),
`models/framelite/` (the teacher's pinned weights), `runs/<run>/` (`model.pt`, `config.json`, `metrics.json`).
Every stage skips what is already there, so a failed run is resumed by running it again.
"""

import json
import shutil
import time
from pathlib import Path

import modal

from training.crop.budget import Ledger, photos_to_label, rate, training_reserve
from training.crop.teacher import FRAMELITE_COMMIT, FRAMELITE_REPO

GPU = "L4"
#: GPUs labelling at once (the Starter plan allows 10).
MAX_GPUS = 6
#: Photos per labelling call.
CHUNK = 250
#: The fewest labelled photos worth training on; with fewer, the run stops before training.
MIN_PHOTOS = 3000
ROOT = Path("/vol")

#: Each function's resources, and what a second of it costs (`budget.rate`).
PREPARE = {"cpu": 2.0, "memory": 8192}
FETCH = {"cpu": 1.0, "memory": 4096}
LABEL = {"cpu": 2.0, "memory": 8192}
BUILD = {"cpu": 4.0, "memory": 16384}
TRAIN = {"cpu": 4.0, "memory": 16384}
ORCHESTRATE = {"cpu": 0.25, "memory": 1024}
PRE_INDEX = {"cpu": 2.0, "memory": 8192}
PRE_CHUNK = {"cpu": 2.0, "memory": 4096}
PRETRAIN = {"cpu": 4.0, "memory": 32768}
#: Photos per pretraining download call, and how many of those calls run at once.
PRE_CHUNK_PHOTOS = 2000
PRE_CONTAINERS = 16
#: The longest one pretraining download call may take (its timeout, and its worst case in the budget).
PRE_CHUNK_SECONDS = 900


def _rate(resources: dict, gpu: bool = False) -> float:
    return rate(gpu, resources["cpu"], resources["memory"] / 1024)


app = modal.App("yellowmoon-crop")
volume = modal.Volume.from_name("yellowmoon-crop", create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6", "pillow==12.2.0")
cpu_image = base.add_local_python_source("training")
gpu_image = (
    base.apt_install("git")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "transformers==5.16.1", "tokenizers==0.23.1", "peft==0.20.0",
        "accelerate==1.14.0", "safetensors==0.8.0", "huggingface-hub==1.29.0", "einops==0.8.2",
    )
    .run_commands(f"git clone {FRAMELITE_REPO} /opt/FrameLite && git -C /opt/FrameLite checkout {FRAMELITE_COMMIT}")
    .env({"FRAMELITE_SRC": "/opt/FrameLite", "HF_HOME": "/vol/hf"})
    .add_local_python_source("training")
)


@app.function(image=cpu_image, volumes={"/vol": volume}, **PREPARE, timeout=2 * 3600)
def prepare(name: str, subset: str, n: int, seed: int) -> list[str]:
    """Selects and downloads the photos; returns the ids that arrived."""
    from training.crop import openimages

    out = ROOT / "data" / name
    manifest = out / "manifest.json"
    if manifest.exists():
        return [r["id"] for r in json.loads(manifest.read_text())]
    listing = openimages.read_list(subset, ROOT / "lists" / f"{subset}-images.csv")
    got = openimages.download(openimages.select(openimages.rows(listing), n, seed), out, workers=64)
    manifest.write_text(json.dumps(got, indent=1, ensure_ascii=False))
    volume.commit()
    return [r["id"] for r in got]


@app.function(image=gpu_image, volumes={"/vol": volume}, **FETCH, timeout=3600)
def fetch_teacher() -> None:
    """FrameLite's adapter and base model at the pinned revisions, once."""
    from training.crop.teacher import download_weights

    weights = ROOT / "models" / "framelite"
    if not (weights / "framelite.safetensors").exists():
        download_weights(weights)
        volume.commit()


@app.cls(image=gpu_image, gpu=GPU, volumes={"/vol": volume}, **LABEL, timeout=3600, max_containers=MAX_GPUS)
class Labeller:
    @modal.enter()
    def load(self) -> None:
        from training.crop.teacher import Teacher

        volume.reload()
        self.teacher = Teacher(ROOT / "models" / "framelite", "cuda", "bfloat16")

    @modal.method()
    def label(self, name: str, chunk: int, ids: list[str]) -> dict:
        """FrameLite's crops for one chunk, written to `labels/<chunk>.json`. Returns the photos, how many crops are
        valid, the teacher's own seconds summed over the photos, and this call's seconds (both for the budget)."""
        t0 = time.time()
        # Other containers' chunks are only visible after a reload: without it, a warm container labels again what
        # another one already paid for (it did on 2026-10-23, after the orchestrator was restarted).
        volume.reload()
        path = ROOT / "data" / name / "labels" / f"{chunk:05d}.json"
        if path.exists():
            labels = json.loads(path.read_text())
        else:
            images = ROOT / "data" / name / "images"
            labels = {i: self.teacher.label(images / f"{i}.jpg") for i in ids}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(labels))
            volume.commit()
        return {
            "photos": len(labels),
            "valid": sum(1 for v in labels.values() if v["crop"]),
            "teacher_seconds": sum(v.get("seconds") or 0.0 for v in labels.values()),
            "call_seconds": time.time() - t0,
        }


@app.function(image=cpu_image, volumes={"/vol": volume}, **BUILD, timeout=2 * 3600)
def build_arrays(name: str) -> int:
    """Merges the chunks' labels and builds the training arrays."""
    from training.crop import data

    volume.reload()
    folder = ROOT / "data" / name
    labels = {}
    for f in sorted((folder / "labels").glob("*.json")):
        labels.update(json.loads(f.read_text()))
    (folder / "labels.json").write_text(json.dumps(labels))
    n = data.build(folder / "images", labels, folder / "arrays")
    volume.commit()
    return n


@app.function(image=gpu_image, gpu=GPU, volumes={"/vol": volume}, **TRAIN, timeout=90 * 60)
def train_student(name: str, run: str, config: dict) -> dict:
    """Trains one student on a local copy of the arrays; writes `runs/<run>/`."""
    from training.crop.train import train

    t0 = time.time()
    volume.reload()
    out = ROOT / "runs" / run
    if (out / "metrics.json").exists():
        return {**json.loads((out / "metrics.json").read_text()), "call_seconds": time.time() - t0}
    local = Path("/tmp/arrays")
    shutil.copytree(ROOT / "data" / name / "arrays", local, dirs_exist_ok=True)
    metrics = train(local, out, config)
    volume.commit()
    return {**metrics, "call_seconds": time.time() - t0}


@app.function(image=cpu_image, volumes={"/vol": volume}, **PRE_INDEX, timeout=3600)
def pre_index(pre: str, n: int, seed: int, crop_set: str) -> int:
    """Selects the pretraining photos (none of the crop set's) and reads their boxes (`boxes.index`); returns the
    number of download chunks."""
    from training.crop import boxes

    volume.reload()
    exclude = {r["id"] for r in json.loads((ROOT / "data" / crop_set / "manifest.json").read_text())}
    chunks = boxes.index("train", n, seed, PRE_CHUNK_PHOTOS, exclude, ROOT / "lists", ROOT / "pre" / pre)
    volume.commit()
    return chunks


@app.function(image=cpu_image, volumes={"/vol": volume}, **PRE_CHUNK, timeout=PRE_CHUNK_SECONDS,
              max_containers=PRE_CONTAINERS)
def pre_chunk(pre: str, k: int) -> dict:
    """Downloads one chunk of pretraining photos and writes its arrays (`boxes.build_chunk`)."""
    from training.crop import boxes

    t0 = time.time()
    volume.reload()
    n = boxes.build_chunk(ROOT / "pre" / pre, k, workers=48)
    volume.commit()
    return {"photos": n, "call_seconds": time.time() - t0}


@app.function(image=gpu_image, gpu=GPU, volumes={"/vol": volume}, **PRETRAIN, timeout=60 * 60)
def pretrain_backbone(pre: str, run: str, config: dict) -> dict:
    """Pretrains the student's backbone on the boxes (`pretrain.train`); writes `runs/<run>/backbone.pt`."""
    from training.crop import pretrain

    t0 = time.time()
    volume.reload()
    out = ROOT / "runs" / run
    if not (out / "metrics.json").exists():
        pretrain.train(ROOT / "pre" / pre, out, config)
        volume.commit()
    return {**json.loads((out / "metrics.json").read_text()), "call_seconds": time.time() - t0}


@app.function(image=cpu_image, volumes={"/vol": volume})
def results() -> dict:
    """Every run's metrics, without the per-epoch history."""
    volume.reload()
    out = {}
    for m in sorted((ROOT / "runs").glob("*/metrics.json")):
        out[m.parent.name] = {k: v for k, v in json.loads(m.read_text()).items() if k != "history"}
    print(json.dumps(out, indent=1))
    return out


class BudgetExceeded(RuntimeError):
    """The run's estimated spend passed its budget."""


@app.function(image=gpu_image, volumes={"/vol": volume}, **ORCHESTRATE, timeout=12 * 3600)
def run_pilot(n: int, seed: int, subset: str, curve: str, epochs: int, budget: float, max_minutes: float,
              max_label_hours: float) -> dict:
    """The whole pilot: `curve` lists the training-set sizes to train at (0 = every labelled photo). Returns each
    run's metrics without the per-epoch history, the photos labelled and the spend estimate."""
    from training.crop.train import DEFAULTS

    t_start = time.time()
    name = f"{subset}-{n}-s{seed}"
    # Modal retries an input whose container is preempted; the spend so far is kept on the volume so that the retry
    # counts it instead of starting again from zero (on 2026-10-23 a restart did, and the $20 applied twice).
    ledger_path = ROOT / "data" / name / "ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    volume.reload()
    ledger = Ledger.resume(budget, ledger_path)
    if ledger.spent:
        print(f"resumed: ${ledger.spent:.2f} already spent by an earlier attempt (estimate)", flush=True)
    sizes = [int(s) for s in curve.split(",")]
    reserve = training_reserve(len(sizes), max_minutes, _rate(TRAIN, gpu=True)) + 1800 * _rate(BUILD)
    print(f"budget ${budget:.2f}; training and arrays reserved ${reserve:.2f}", flush=True)

    def check(stage: str) -> None:
        ledger.save(ledger_path)
        volume.commit()
        print(f"{stage}: spent ${ledger.spent:.2f} of ${budget:.2f} (estimate)", flush=True)
        if ledger.spent > budget:
            raise BudgetExceeded(f"estimated spend ${ledger.spent:.2f} passed the budget ${budget:.2f} at {stage}")

    t = time.time()
    ids = prepare.remote(name, subset, n, seed)
    ledger.charge("prepare", time.time() - t, _rate(PREPARE))
    print(f"{len(ids)} photos in data/{name}", flush=True)
    t = time.time()
    fetch_teacher.remote()
    ledger.charge("teacher weights", time.time() - t, _rate(FETCH))
    check("download")

    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    label_rate = _rate(LABEL, gpu=True)
    t = time.time()
    first = Labeller().label.remote(name, 0, chunks[0])
    first_seconds = time.time() - t
    ledger.charge("labelling chunk 1 (one GPU, with its start)", first_seconds, label_rate)
    # The slower of the call's own time (start and model load included) and the teacher's measured time, per photo:
    # a resumed chunk returns at once but still carries the teacher's seconds.
    per_photo = max(first_seconds, first["teacher_seconds"] * 1.1) / max(1, first["photos"])
    print(f"FrameLite: {per_photo:.2f} s a photo on the {GPU}; {first['valid']}/{first['photos']} valid", flush=True)
    check("first chunk")
    if first["valid"] < first["photos"] // 2:
        raise RuntimeError(f"only {first['valid']} of {first['photos']} crops are valid; stopping before spending more")
    rest = sum(len(c) for c in chunks[1:])
    more = photos_to_label(ledger.left(), reserve, per_photo, label_rate, rest, MAX_GPUS, max_label_hours, CHUNK)
    todo = chunks[1:1 + more // CHUNK] if more < rest else chunks[1:]
    print(f"labelling {sum(len(c) for c in todo)} more photos of {rest} (budget and time allow {more})", flush=True)
    if first["valid"] + sum(len(c) for c in todo) < MIN_PHOTOS:
        raise BudgetExceeded(f"the budget pays for fewer than {MIN_PHOTOS} labelled photos; stopping before training")

    ledger.charge(f"labeller starts ({MAX_GPUS} containers)", MAX_GPUS * 120, label_rate)
    valid, labelled = first["valid"], first["photos"]
    calls = Labeller().label.map([name] * len(todo), range(1, 1 + len(todo)), todo)
    for k, got in enumerate(calls):
        valid += got["valid"]
        labelled += got["photos"]
        ledger.charge(f"labelling chunk {k + 2}", got["call_seconds"], label_rate)
        print(f"labelled chunk {k + 2}/{1 + len(todo)}: {valid} valid crops so far", flush=True)
        ledger.save(ledger_path)
        volume.commit()
        if ledger.spent + reserve > budget * 1.05:
            raise BudgetExceeded(f"labelling ran over its share: ${ledger.spent:.2f} spent, ${reserve:.2f} reserved")

    t = time.time()
    in_arrays = build_arrays.remote(name)
    ledger.charge("arrays", time.time() - t, _rate(BUILD))
    print(f"{in_arrays} photos in the training arrays", flush=True)
    check("arrays")
    runs = []
    for size in sizes:
        if size and size >= in_arrays:
            continue
        config = {**DEFAULTS, "seed": seed, "epochs": epochs, "n_train": size or None, "device": "cuda",
                  "max_minutes": max_minutes}
        runs.append((name, f"{name}-n{size or 'all'}-e{epochs}", config))
    out = {}
    for (_, run, _), metrics in zip(runs, train_student.starmap(runs)):
        out[run] = {k: v for k, v in metrics.items() if k != "history"}
        ledger.charge(f"training {run}", (metrics["minutes"] + 10) * 60, _rate(TRAIN, gpu=True))
        print(run, json.dumps(out[run]), flush=True)
    ledger.charge("this orchestrator", time.time() - t_start, _rate(ORCHESTRATE))
    report = {"runs": out, "photos": {"labelled": labelled, "valid": valid, "in_arrays": in_arrays},
              "seconds_per_photo": round(per_photo, 3), "spend": ledger.report()}
    print(json.dumps(report["spend"]), flush=True)
    return report


@app.function(image=gpu_image, volumes={"/vol": volume}, **ORCHESTRATE, timeout=4 * 3600)
def run_training(name: str, curve: str, epochs: int, budget: float, max_minutes: float, seed: int) -> dict:
    """Builds the arrays from the labels already on the volume and trains the student at each size of `curve`
    (0 = every labelled photo); labels nothing. Refuses to start when the most the runs can cost (each stopped at
    `max_minutes`, the function's own timeout behind it) passes `budget`."""
    from training.crop.train import DEFAULTS

    sizes = [int(s) for s in curve.split(",")]
    worst = training_reserve(len(sizes), max_minutes, _rate(TRAIN, gpu=True)) + 1800 * _rate(BUILD) * 1.25
    print(f"at most ${worst:.2f} for {len(sizes)} run(s) of up to {max_minutes:.0f} minutes; budget ${budget:.2f}",
          flush=True)
    if worst > budget:
        raise BudgetExceeded(f"the training could cost up to ${worst:.2f}, more than the budget ${budget:.2f}")
    in_arrays = build_arrays.remote(name)
    print(f"{in_arrays} labelled photos in the training arrays", flush=True)
    runs = []
    for size in sizes:
        if size and size >= in_arrays:
            continue
        config = {**DEFAULTS, "seed": seed, "epochs": epochs, "n_train": size or None, "device": "cuda",
                  "max_minutes": max_minutes}
        runs.append((name, f"{name}-n{size or 'all'}-e{epochs}", config))
    out = {}
    for (_, run, _), metrics in zip(runs, train_student.starmap(runs)):
        out[run] = {k: v for k, v in metrics.items() if k != "history"}
        print(run, json.dumps(out[run]), flush=True)
    return {"runs": out, "photos_in_arrays": in_arrays}


#: Modal charges a non-preemptible container more; this covers it (an assumption, on the safe side).
NONPREEMPTIBLE = 3.0


@app.function(image=gpu_image, volumes={"/vol": volume}, **ORCHESTRATE, timeout=5 * 3600, nonpreemptible=True)
def run_improve(budget: float, crop_set: str, n_pre: int, seed: int, pre_epochs: int, pre_minutes: float,
                epochs: int, ft_minutes: float) -> dict:
    """Stage A of improving the student, on the labels already paid for (owner, 2026-10-23: "We can use remaining 13$
    if this improves our model"): a learning curve from scratch, the `bins` head, and the backbone pretrained on Open
    Images' boxes (`pretrain.py`) under both heads and at three sizes. Labels nothing. Refuses to start when the most
    it can cost, every call at its time limit, passes `budget`; stops and cancels what is running if its own count
    of the spend does. It is not preemptible, so it is not restarted half way (the pilot's was, 2026-10-23)."""
    from training.crop.budget import MARGIN
    from training.crop.train import DEFAULTS

    t_start = time.time()
    pre = f"oi-{n_pre}-s{seed}"
    ledger_path = ROOT / "pre" / f"ledger-{pre}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    volume.reload()
    ledger = Ledger.resume(budget, ledger_path)
    backbone = f"pre-{pre}-e{pre_epochs}"
    init = str(ROOT / "runs" / backbone / "backbone.pt")
    train_rate = _rate(TRAIN, gpu=True)

    def config(head: str, size: int, pretrained: bool) -> dict:
        return {**DEFAULTS, "seed": 1, "epochs": epochs, "n_train": size or None, "device": "cuda",
                "max_minutes": ft_minutes, "head": head, "init": init if pretrained else None,
                "lr": 1e-3 if pretrained else DEFAULTS["lr"]}

    def runs(pretrained: bool, plan) -> list[tuple[str, str, dict]]:
        kind = "pre" if pretrained else "scratch"
        return [(crop_set, f"ft-{kind}-{head}-n{size or 'all'}", config(head, size, pretrained)) for head, size in plan]

    scratch = runs(False, [("box", 2700), ("box", 5400), ("bins", 0)])
    pretrained = runs(True, [("box", 2700), ("box", 5400), ("box", 0), ("bins", 0)])
    chunks_most = -(-n_pre // PRE_CHUNK_PHOTOS)
    worst = MARGIN * (
        3600 * _rate(PRE_INDEX)
        + chunks_most * PRE_CHUNK_SECONDS * _rate(PRE_CHUNK)
        + (pre_minutes * 1.1 + 10) * 60 * _rate(PRETRAIN, gpu=True)
        + len(scratch + pretrained) * (ft_minutes * 1.1 + 5) * 60 * train_rate
        + 5 * 3600 * _rate(ORCHESTRATE) * NONPREEMPTIBLE
    )
    print(f"at most ${worst:.2f} (every call at its limit); budget ${budget:.2f}", flush=True)
    if ledger.spent:
        print(f"resumed: ${ledger.spent:.2f} already spent by an earlier attempt (estimate)", flush=True)
    if ledger.spent + worst > budget:
        raise BudgetExceeded(f"stage A could cost up to ${worst:.2f}, more than the budget ${budget:.2f} allows")

    out: dict[str, dict] = {}

    def check(stage: str) -> None:
        ledger.save(ledger_path)
        volume.commit()
        print(f"{stage}: spent ${ledger.spent:.2f} of ${budget:.2f} (estimate)", flush=True)
        if ledger.spent > budget:
            raise BudgetExceeded(f"estimated spend ${ledger.spent:.2f} passed the budget ${budget:.2f} at {stage}")

    def record(run: str, m) -> None:
        if isinstance(m, Exception):
            ledger.charge(f"training {run} (failed)", (ft_minutes * 1.1 + 5) * 60, train_rate)
            out[run] = {"error": f"{type(m).__name__}: {m}"}
        else:
            # A minute more than the call took, for the container's start before the call's own clock.
            ledger.charge(f"training {run}", m["call_seconds"] + 60, train_rate)
            out[run] = {k: v for k, v in m.items() if k not in ("history", "call_seconds")}
        print(run, json.dumps(out[run]), flush=True)
        check(run)

    # The runs from scratch need nothing new, so they train while the pretraining photos download.
    pending = [(args[1], train_student.spawn(*args)) for args in scratch]
    try:
        try:
            t = time.time()
            chunks = pre_index.remote(pre, n_pre, seed, crop_set)
            ledger.charge("pretraining index", time.time() - t, _rate(PRE_INDEX))
            check("index")
            photos, failed = 0, 0
            for got in pre_chunk.map([pre] * chunks, range(chunks), return_exceptions=True):
                if isinstance(got, Exception):
                    # A chunk that failed is left out; it is counted as if it ran to its time limit.
                    failed += 1
                    ledger.charge("pretraining photos (failed chunk)", PRE_CHUNK_SECONDS, _rate(PRE_CHUNK))
                    continue
                photos += got["photos"]
                ledger.charge("pretraining photos", got["call_seconds"] + 30, _rate(PRE_CHUNK))
            print(f"{photos} pretraining photos in {chunks - failed} chunks ({failed} failed)", flush=True)
            check("pretraining photos")
            pre_config = {"seed": seed, "epochs": pre_epochs, "batch": 128, "lr": 2e-3, "weight_decay": 0.05,
                          "device": "cuda", "max_minutes": pre_minutes}
            t = time.time()
            m = pretrain_backbone.remote(pre, backbone, pre_config)
            ledger.charge(f"pretraining {backbone}", time.time() - t + 60, _rate(PRETRAIN, gpu=True))
            out[backbone] = {k: v for k, v in m.items() if k not in ("history", "call_seconds")}
            out[backbone]["last_epoch"] = m["history"][-1] if m["history"] else None
            print(backbone, json.dumps(out[backbone]), flush=True)
            check("pretraining")
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - without a backbone the runs from scratch are still collected
            out[backbone] = {"error": f"{type(e).__name__}: {e}"}
            print(backbone, json.dumps(out[backbone]), flush=True)
            pretrained = []
        for args, m in zip(pretrained, train_student.starmap(pretrained, return_exceptions=True)):
            record(args[1], m)
        while pending:
            run, call = pending.pop(0)
            try:
                m = call.get()
            except Exception as e:  # noqa: BLE001 - one failed run is reported, the others still collected
                m = e
            record(run, m)
        ledger.charge("this orchestrator", time.time() - t_start, _rate(ORCHESTRATE) * NONPREEMPTIBLE)
        check("done")
    finally:
        for _, call in pending:
            call.cancel()
    return {"runs": out, "spend": ledger.report()}


@app.local_entrypoint()
def improve(budget: float = 6.5, crop_set: str = "train-20000-s1", n_pre: int = 96000, seed: int = 2,
            pre_epochs: int = 12, pre_minutes: float = 22.0, epochs: int = 40, ft_minutes: float = 12.0) -> None:
    """Runs `run_improve` remotely (stage A: no labelling) and prints its result."""
    print(json.dumps(run_improve.remote(budget, crop_set, n_pre, seed, pre_epochs, pre_minutes, epochs, ft_minutes),
                     indent=1))


@app.local_entrypoint()
def train_labelled(name: str = "train-20000-s1", curve: str = "0", epochs: int = 40, budget: float = 3.0,
                   max_minutes: float = 30.0, seed: int = 1) -> None:
    """Runs `run_training` remotely: the student on the labels already paid for, nothing labelled."""
    print(json.dumps(run_training.remote(name, curve, epochs, budget, max_minutes, seed), indent=1))


@app.local_entrypoint()
def pilot(n: int = 20000, seed: int = 1, subset: str = "train", curve: str = "5000,10000,0", epochs: int = 40,
          budget: float = 20.0, max_minutes: float = 40.0, max_label_hours: float = 3.5) -> None:
    """Runs `run_pilot` remotely and prints its result."""
    print(json.dumps(run_pilot.remote(n, seed, subset, curve, epochs, budget, max_minutes, max_label_hours), indent=1))


@app.local_entrypoint()
def fetch(dest: str = "training/crop/runs", name: str = "train-20000-s1", pre: str = "oi-96000-s2") -> None:
    """Copies every run's `model.pt`, `config.json` and `metrics.json`, the photo set's `labels.json` and
    `manifest.json`, and the pretraining photos' `manifest.json` (`pre/manifest.json`, for the model card's
    credits), from the volume into `dest`."""
    out = Path(dest)
    try:
        entries = volume.listdir("runs", recursive=True)
    except Exception as e:  # noqa: BLE001 - no runs yet (the pilot stopped early): fetch the rest
        print(f"runs: {type(e).__name__}: {e}")
        entries = []
    wanted = []
    for e in entries:
        rel = Path(e.path.lstrip("/"))
        rel = Path(*rel.parts[1:]) if rel.parts[:1] == ("runs",) else rel  # listdir may or may not keep "runs/"
        if rel.name in ("model.pt", "config.json", "metrics.json"):
            wanted.append((f"runs/{rel.as_posix()}", out / rel))
    wanted += [(f"data/{name}/{f}", out / "data" / f) for f in ("labels.json", "manifest.json")]
    wanted.append((f"pre/{pre}/manifest.json", out / "pre" / "manifest.json"))
    for remote, local in wanted:
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            with local.open("wb") as fh:
                for part in volume.read_file(remote):
                    fh.write(part)
            print(f"{remote} -> {local}")
        except Exception as e:  # noqa: BLE001 - a missing file is reported, the rest still copied
            local.unlink(missing_ok=True)
            print(f"{remote}: {type(e).__name__}: {e}")
