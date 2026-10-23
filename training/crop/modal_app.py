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


@app.function(image=gpu_image, gpu=GPU, volumes={"/vol": volume}, **TRAIN, timeout=3 * 3600)
def train_student(name: str, run: str, config: dict) -> dict:
    """Trains one student on a local copy of the arrays; writes `runs/<run>/`."""
    from training.crop.train import train

    volume.reload()
    out = ROOT / "runs" / run
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    local = Path("/tmp/arrays")
    shutil.copytree(ROOT / "data" / name / "arrays", local, dirs_exist_ok=True)
    metrics = train(local, out, config)
    volume.commit()
    return metrics


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
    ledger = Ledger(budget)
    sizes = [int(s) for s in curve.split(",")]
    reserve = training_reserve(len(sizes), max_minutes, _rate(TRAIN, gpu=True)) + 1800 * _rate(BUILD)
    print(f"budget ${budget:.2f}; training and arrays reserved ${reserve:.2f}", flush=True)

    def check(stage: str) -> None:
        print(f"{stage}: spent ${ledger.spent:.2f} of ${budget:.2f} (estimate)", flush=True)
        if ledger.spent > budget:
            raise BudgetExceeded(f"estimated spend ${ledger.spent:.2f} passed the budget ${budget:.2f} at {stage}")

    name = f"{subset}-{n}-s{seed}"
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


@app.local_entrypoint()
def pilot(n: int = 20000, seed: int = 1, subset: str = "train", curve: str = "5000,10000,0", epochs: int = 40,
          budget: float = 20.0, max_minutes: float = 40.0, max_label_hours: float = 3.5) -> None:
    """Runs `run_pilot` remotely and prints its result."""
    print(json.dumps(run_pilot.remote(n, seed, subset, curve, epochs, budget, max_minutes, max_label_hours), indent=1))


@app.local_entrypoint()
def fetch(dest: str = "training/crop/runs", name: str = "train-20000-s1") -> None:
    """Copies every run's `model.pt`, `config.json` and `metrics.json`, and the photo set's `labels.json` and
    `manifest.json`, from the volume into `dest`."""
    out = Path(dest)
    try:
        entries = volume.listdir("runs", recursive=True)
    except Exception as e:  # noqa: BLE001 - no runs yet (the pilot stopped early): fetch the rest
        print(f"runs: {type(e).__name__}: {e}")
        entries = []
    wanted = [(e.path, out / Path(e.path).relative_to("runs")) for e in entries
              if Path(e.path).name in ("model.pt", "config.json", "metrics.json")]
    wanted += [(f"data/{name}/{f}", out / "data" / f) for f in ("labels.json", "manifest.json")]
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
