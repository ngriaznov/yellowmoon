"""The crop pilot on Modal (`research/crop-models-2026-10-23.md`, owner's decision of 2026-10-23): Open Images
photos, FrameLite's crops as labels, and the student trained at several training-set sizes (a learning curve).

Needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment (never in the repository). From the repository
root:

    modal run --detach -m training.crop.modal_app::pilot --n 20000  # select, download, label, build, train
    modal run -m training.crop.modal_app::results                   # the runs' metrics
    modal volume get yellowmoon-crop runs/<run>/model.pt testdata/crop/runs/<run>/

Everything lands in the Modal volume `yellowmoon-crop`: `lists/` (Open Images' image lists), `data/<name>/`
(`images/`, `manifest.json` with each photo's author and licence, `labels/` per chunk, `labels.json`, `arrays/`),
`models/framelite/` (the teacher's pinned weights), `runs/<run>/` (`model.pt`, `config.json`, `metrics.json`).
Every stage skips what is already there, so a failed run is resumed by running it again. The pilot's steps run in
one remote function, so with `--detach` it carries on when the local client disconnects.
"""

import json
import shutil
from pathlib import Path

import modal

from training.crop.teacher import FRAMELITE_COMMIT, FRAMELITE_REPO

GPU = "L4"
#: GPUs labelling at once.
MAX_GPUS = 4
#: Photos per labelling call.
CHUNK = 250
ROOT = Path("/vol")

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


@app.function(image=cpu_image, volumes={"/vol": volume}, cpu=8, memory=16384, timeout=4 * 3600)
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


@app.function(image=gpu_image, volumes={"/vol": volume}, timeout=3600)
def fetch_teacher() -> None:
    """FrameLite's adapter and base model at the pinned revisions, once."""
    from training.crop.teacher import download_weights

    weights = ROOT / "models" / "framelite"
    if not (weights / "framelite.safetensors").exists():
        download_weights(weights)
        volume.commit()


@app.cls(image=gpu_image, gpu=GPU, volumes={"/vol": volume}, timeout=3600, max_containers=MAX_GPUS)
class Labeller:
    @modal.enter()
    def load(self) -> None:
        from training.crop.teacher import Teacher

        volume.reload()
        self.teacher = Teacher(ROOT / "models" / "framelite", "cuda", "bfloat16")

    @modal.method()
    def label(self, name: str, chunk: int, ids: list[str]) -> int:
        """FrameLite's crops for one chunk, written to `labels/<chunk>.json`; returns how many are valid."""
        path = ROOT / "data" / name / "labels" / f"{chunk:05d}.json"
        if path.exists():
            return sum(1 for v in json.loads(path.read_text()).values() if v["crop"])
        images = ROOT / "data" / name / "images"
        labels = {i: self.teacher.label(images / f"{i}.jpg") for i in ids}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels))
        volume.commit()
        return sum(1 for v in labels.values() if v["crop"])


@app.function(image=cpu_image, volumes={"/vol": volume}, cpu=8, memory=16384, timeout=2 * 3600)
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


@app.function(image=gpu_image, gpu=GPU, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=4 * 3600)
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


@app.function(image=gpu_image, volumes={"/vol": volume}, timeout=24 * 3600)
def run_pilot(n: int, seed: int, subset: str, curve: str, epochs: int) -> dict:
    """The whole pilot: `curve` lists the training-set sizes to train at (0 = every labelled photo); returns each
    run's metrics without the per-epoch history."""
    from training.crop.train import DEFAULTS

    name = f"{subset}-{n}-s{seed}"
    ids = prepare.remote(name, subset, n, seed)
    print(f"{len(ids)} photos in data/{name}", flush=True)
    fetch_teacher.remote()
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    valid = 0
    for k, got in enumerate(Labeller().label.map([name] * len(chunks), range(len(chunks)), chunks)):
        valid += got
        print(f"labelled chunk {k + 1}/{len(chunks)}: {valid} valid crops so far", flush=True)
    print(f"{build_arrays.remote(name)} photos in the training arrays", flush=True)
    runs = []
    for size in (int(s) for s in curve.split(",")):
        config = {**DEFAULTS, "seed": seed, "epochs": epochs, "n_train": size or None, "device": "cuda"}
        runs.append((name, f"{name}-n{size or 'all'}-e{epochs}", config))
    out = {}
    for (_, run, _), metrics in zip(runs, train_student.starmap(runs)):
        out[run] = {k: v for k, v in metrics.items() if k != "history"}
        print(run, json.dumps(out[run]), flush=True)
    return out


@app.local_entrypoint()
def pilot(n: int = 20000, seed: int = 1, subset: str = "train", curve: str = "5000,10000,0", epochs: int = 40) -> None:
    """Runs `run_pilot` remotely and prints its result."""
    print(json.dumps(run_pilot.remote(n, seed, subset, curve, epochs), indent=1))
