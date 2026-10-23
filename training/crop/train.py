"""Trains the crop student on the arrays of `data.py`: for the `box` head L1 on the crop's edges plus (1 - IoU), for
the `bins` head the cross-entropy of each number's bins (`model.py`); horizontal flips and brightness and contrast
jitter. `init` starts the student's features from a pretrained backbone (`pretrain.py`) instead of from scratch.
Reports the validation IoU with the teacher next to two baselines (the whole photo, and the training set's mean crop)
so that a student that learned only the average crop shows as such.

    python3 -m training.crop.train --arrays testdata/crop/smoke/arrays --out testdata/crop/runs/smoke \
        --epochs 2 --batch 8 --device cpu
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from training.crop.model import CropNet, bin_target, box_from_numbers, iou, numbers_from_box, parameters


def split(ids: list[str], val_share: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Train and validation indices, by a hash of each photo's id (stable whatever the set's size)."""
    val = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) / 2**32 < val_share for k in ids])
    return np.flatnonzero(~val), np.flatnonzero(val)


def batch(pix: np.ndarray, target: np.ndarray, idx: np.ndarray, rng: np.random.Generator | None, device):
    """Images in [0, 1] (N, 3, S, S) and targets for `idx`; with `rng`, augmented (flip, brightness, contrast)."""
    x = torch.from_numpy(np.ascontiguousarray(pix[np.sort(idx)])).to(device).permute(0, 3, 1, 2).float() / 255
    t = torch.from_numpy(target[np.sort(idx)]).to(device)
    if rng is not None:
        n = x.shape[0]
        flip = torch.from_numpy(rng.random(n) < 0.5).to(device)
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        flipped = torch.stack([1 - t[:, 2], t[:, 1], 1 - t[:, 0], t[:, 3]], dim=1)
        t = torch.where(flip[:, None], flipped, t)
        gain = torch.from_numpy(rng.uniform(0.8, 1.2, n).astype(np.float32)).to(device)[:, None, None, None]
        contrast = torch.from_numpy(rng.uniform(0.8, 1.2, n).astype(np.float32)).to(device)[:, None, None, None]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * contrast + mean * gain).clamp(0, 1)
    return x, t


def evaluate(model, pix, target, idx, device, size: int = 256) -> np.ndarray:
    """The student's crops for `idx`, in order."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), size):
            x, _ = batch(pix, target, idx[s:s + size], None, device)
            out.append(model(x).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)


def crop_loss(model: CropNet, raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss of the head's raw output against the teacher's boxes (module docstring)."""
    if model.kind == "box":
        p = box_from_numbers(torch.sigmoid(raw))
        return (p - target).abs().mean() + (1 - iou(p, target)).mean()
    soft = bin_target(numbers_from_box(target))
    return -(soft * torch.log_softmax(raw, dim=-1)).sum(-1).mean()


def load_backbone(model: CropNet, path: Path) -> int:
    """Copies the `features.*` weights of a pretrained backbone into `model`; returns how many tensors it copied."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    feats = {k: v for k, v in state.items() if k.startswith("features.")}
    if not feats:
        raise ValueError(f"{path} holds no backbone weights")
    missing = [k for k in model.state_dict() if k.startswith("features.") and k not in feats]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} backbone tensors, e.g. {missing[0]}")
    model.load_state_dict(feats, strict=False)
    return len(feats)


def train(arrays: Path, out: Path, config: dict) -> dict:
    """Trains one student; writes `model.pt`, `config.json` and `metrics.json` into `out` and returns the metrics."""
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config["device"])
    pix = np.load(arrays / "images.npy", mmap_mode="r")
    target = np.load(arrays / "target.npy")
    ids = json.loads((arrays / "ids.json").read_text())
    tr, va = split(ids)
    if config.get("n_train"):
        tr = np.sort(np.random.default_rng(config["seed"]).permutation(tr)[: config["n_train"]])
    model = CropNet(head=config["head"]).to(device)
    if config.get("init"):
        load_backbone(model, Path(config["init"]))
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    steps_per_epoch = max(1, len(tr) // config["batch"])
    # `plan` changes once, after the first epoch, if `max_minutes` leaves room for fewer epochs: the cosine then ends
    # at the last epoch that fits rather than being cut off at a high learning rate.
    plan = {"epochs": config["epochs"], "total": steps_per_epoch * config["epochs"]}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / steps_per_epoch)
        * 0.5 * (1 + math.cos(math.pi * min(s, plan["total"]) / plan["total"])),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out.mkdir(parents=True, exist_ok=True)
    mean_crop = target[tr].mean(axis=0)
    val_t = torch.from_numpy(target[va])
    baselines = {
        "whole": float(iou(torch.tensor([[0.0, 0.0, 1.0, 1.0]]).expand(len(va), 4), val_t).mean()) if len(va) else None,
        "mean_crop": float(iou(torch.from_numpy(mean_crop)[None].expand(len(va), 4), val_t).mean()) if len(va) else None,
    }
    history, best = [], -1.0
    t0 = time.time()
    limit = config.get("max_minutes")
    for epoch in range(config["epochs"]):
        if epoch >= plan["epochs"] or (limit and time.time() - t0 > limit * 60 * 1.1):
            break
        order = rng.permutation(tr)
        losses = []
        for s in range(steps_per_epoch):
            x, t = batch(pix, target, order[s * config["batch"]:(s + 1) * config["batch"]], rng, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                a = model.raw(x)
            loss = crop_loss(model, a.float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
        pred = evaluate(model, pix, target, va, device)
        val_iou = float(iou(torch.from_numpy(pred), val_t).mean()) if len(va) else float("nan")
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 5), "val_iou": round(val_iou, 4)})
        print(json.dumps(history[-1]), flush=True)
        if val_iou > best or not len(va):
            best = val_iou
            torch.save({"state_dict": model.state_dict(), "config": config}, out / "model.pt")
        if epoch == 0 and limit:
            fit = max(1, int(limit * 60 / (time.time() - t0)))
            if fit < plan["epochs"]:
                plan.update(epochs=fit, total=steps_per_epoch * fit)
                print(f"max_minutes {limit}: {fit} epochs fit", flush=True)
    metrics = {
        "photos": {"train": int(len(tr)), "val": int(len(va))},
        "parameters": parameters(model),
        "best_val_iou": round(best, 4),
        "baselines_val_iou": {k: (round(v, 4) if v is not None else None) for k, v in baselines.items()},
        "mean_crop": [round(float(v), 4) for v in mean_crop],
        "epochs_run": len(history),
        "minutes": round((time.time() - t0) / 60, 1),
        "history": history,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


DEFAULTS = {
    "seed": 1, "epochs": 40, "batch": 128, "lr": 2e-3, "weight_decay": 0.05, "n_train": None, "device": "cuda",
    "max_minutes": None, "head": "box", "init": None,
}
#: The command-line type of the options whose default is None.
TYPES = {"n_train": int, "max_minutes": float, "init": str}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else TYPES[k], default=v)
    args = ap.parse_args()
    config = {k: getattr(args, k) for k in DEFAULTS}
    m = train(args.arrays, args.out, config)
    print(json.dumps({k: v for k, v in m.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
