"""Pretraining for the crop student's backbone on Open Images' own boxes (`boxes.py`), so that it knows what is in a
photo before it learns to crop it from FrameLite's few thousand labels. The backbone (`model.CropNet.features`)
learns to predict where people, faces, animals, vehicles, buildings and objects in general are (`boxes.CHANNELS`
maps) and which classes are present, through two small heads that are dropped afterwards; `train.py --init` then
starts the crop student from it.

    python3 -m training.crop.pretrain --root testdata/crop/pre --out testdata/crop/runs/pre --epochs 2 --batch 8 \
        --device cpu
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.crop import boxes
from training.crop.model import STAGES, CropNet, parameters


class PretrainNet(nn.Module):
    """The crop student's backbone with two pretraining heads: the `CHANNELS` maps at stride 16 (the stride-32
    features upsampled onto the stride-16 ones) and the classes present (pooled stride-32 features)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.net = CropNet()
        c16, c32 = STAGES[-2][1], STAGES[-1][1]
        self.lat16 = nn.Conv2d(c16, 64, 1)
        self.lat32 = nn.Conv2d(c32, 64, 1)
        self.fuse = nn.Sequential(
            nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, len(boxes.CHANNELS), 1),
        )
        self.classes = nn.Linear(c32, n_classes)
        with torch.no_grad():
            self.fuse[-1].bias.fill_(-2.0)
            self.classes.bias.fill_(-5.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f16, f32 = self.net.taps(x)
        up = F.interpolate(self.lat32(f32), size=f16.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(self.lat16(f16) + up), self.classes(f32.mean(dim=(2, 3)))


def load(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Every chunk's arrays under `root/arrays`, concatenated in memory (images, maps, class bits, ids)."""
    dirs = sorted(d for d in (root / "arrays").iterdir() if d.is_dir() and not d.name.endswith(".part"))
    sizes = [len(json.loads((d / "ids.json").read_text())) for d in dirs]
    n = sum(sizes)
    first = dirs[sizes.index(max(sizes))]
    shapes = {k: np.load(first / f"{k}.npy", mmap_mode="r").shape[1:] for k in ("images", "maps", "classes")}
    arrays = {k: np.empty((n, *shapes[k]), np.uint8) for k in shapes}
    ids: list[str] = []
    at = 0
    for d, m in zip(dirs, sizes):
        for k in arrays:
            arrays[k][at:at + m] = np.load(d / f"{k}.npy", mmap_mode="r")
        ids += json.loads((d / "ids.json").read_text())
        at += m
    return arrays["images"], arrays["maps"], arrays["classes"], ids


def batch(pix, maps, bits, idx, n_classes: int, rng: np.random.Generator | None, device):
    """Images in [0, 1], map targets in [0, 1] and class targets for `idx`; with `rng`, flipped and jittered."""
    idx = np.sort(idx)
    x = torch.from_numpy(np.ascontiguousarray(pix[idx])).to(device).permute(0, 3, 1, 2).float() / 255
    m = torch.from_numpy(maps[idx]).to(device).float() / 255
    c = torch.from_numpy(np.unpackbits(bits[idx], axis=1)[:, :n_classes]).to(device).float()
    if rng is not None:
        k = x.shape[0]
        flip = torch.from_numpy(rng.random(k) < 0.5).to(device)
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        m = torch.where(flip[:, None, None, None], m.flip(3), m)
        gain = torch.from_numpy(rng.uniform(0.8, 1.2, k).astype(np.float32)).to(device)[:, None, None, None]
        contrast = torch.from_numpy(rng.uniform(0.8, 1.2, k).astype(np.float32)).to(device)[:, None, None, None]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * contrast + mean * gain).clamp(0, 1)
    return x, m, c


def losses(maps_logit, class_logit, m, c) -> tuple[torch.Tensor, torch.Tensor]:
    """The maps' cross-entropy (mean over cells) and the classes' (summed over classes, a tenth of it)."""
    dense = F.binary_cross_entropy_with_logits(maps_logit.float(), m)
    classes = F.binary_cross_entropy_with_logits(class_logit.float(), c, reduction="sum") / c.shape[0] * 0.1
    return dense, classes


def train(root: Path, out: Path, config: dict) -> dict:
    """Pretrains the backbone on `root`'s chunks; writes `backbone.pt` (the CropNet weights of the epoch with the
    lowest validation loss), `config.json` and `metrics.json` into `out` and returns the metrics."""
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config["device"])
    pix, maps, bits, ids = load(root)
    n_classes = len(json.loads((root / "classes.json").read_text()))
    val = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) / 2**32 < 0.02 for k in ids])
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    model = PretrainNet(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    steps = max(1, len(tr) // config["batch"])
    plan = {"epochs": config["epochs"], "total": steps * config["epochs"]}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / steps) * 0.5 * (1 + math.cos(math.pi * min(s, plan["total"]) / plan["total"])),
    )
    use_amp = device.type == "cuda"
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], math.inf
    t0 = time.time()
    limit = config.get("max_minutes")
    for epoch in range(config["epochs"]):
        if epoch >= plan["epochs"] or (limit and time.time() - t0 > limit * 60 * 1.1):
            break
        order = rng.permutation(tr)
        seen = []
        for s in range(steps):
            x, m, c = batch(pix, maps, bits, order[s * config["batch"]:(s + 1) * config["batch"]], n_classes, rng,
                            device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                ml, cl = model(x)
            dense, classes = losses(ml, cl, m, c)
            loss = dense + classes
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen.append((float(dense), float(classes)))
        v = evaluate(model, pix, maps, bits, va, n_classes, device)
        history.append({"epoch": epoch + 1, "dense": round(float(np.mean([a for a, _ in seen])), 5),
                        "classes": round(float(np.mean([b for _, b in seen])), 5), **v})
        print(json.dumps(history[-1]), flush=True)
        score = v["val_dense"] + v["val_classes"] if len(va) else -epoch
        if score < best:
            best = score
            torch.save(model.net.state_dict(), out / "backbone.pt")
        if epoch == 0 and limit:
            fit = max(1, int(limit * 60 / (time.time() - t0)))
            if fit < plan["epochs"]:
                plan.update(epochs=fit, total=steps * fit)
                print(f"max_minutes {limit}: {fit} epochs fit", flush=True)
    metrics = {
        "photos": {"train": int(len(tr)), "val": int(len(va))},
        "parameters": parameters(model.net),
        "epochs_run": len(history),
        "minutes": round((time.time() - t0) / 60, 1),
        "history": history,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def evaluate(model, pix, maps, bits, idx, n_classes: int, device, size: int = 256) -> dict:
    """Validation losses, and how often the most likely class is one the photo has (`val_top1`)."""
    if not len(idx):
        return {}
    model.eval()
    dense, classes, hits = [], [], 0
    with torch.no_grad():
        for s in range(0, len(idx), size):
            x, m, c = batch(pix, maps, bits, idx[s:s + size], n_classes, None, device)
            ml, cl = model(x)
            d, k = losses(ml, cl, m, c)
            dense.append(float(d) * len(x))
            classes.append(float(k) * len(x))
            hits += int(c.gather(1, cl.argmax(1, keepdim=True)).sum())
    model.train()
    n = len(idx)
    return {"val_dense": round(sum(dense) / n, 5), "val_classes": round(sum(classes) / n, 5),
            "val_top1": round(hits / n, 4)}


DEFAULTS = {"seed": 1, "epochs": 12, "batch": 128, "lr": 2e-3, "weight_decay": 0.05, "device": "cuda",
            "max_minutes": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else float, default=v)
    args = ap.parse_args()
    m = train(args.root, args.out, {k: getattr(args, k) for k in DEFAULTS})
    print(json.dumps({k: v for k, v in m.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
