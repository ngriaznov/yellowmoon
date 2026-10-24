"""Optional, local, evaluation only: how the bokeh network (and the classical baseline) do on real photographs with
human labels of blur.

**RESEARCH-ONLY DATA. EVALUATION ONLY.** CUHK's blur detection set, DUT-DBD and RTF carry no licence that allows
training or redistribution (`research/bokeh-map-2026-10-24.md` sections 3 and 7). The owner allows them for *local
evaluation* (2026-10-24) and nothing else:

- they are never trained on, never uploaded to the Modal volume (nothing under `training/bokeh/` copies them there),
  never committed (the folder is `testdata/bokeh-eval/`, git-ignored; `testdata/` is), never redistributed;
- no number from them gates anything but the owner's own judgement, and they are not credited in a model card as
  training data.

This script is not part of the pipeline: training and `modal_app.py` never import it.

    python -m training.bokeh.eval_real --model <run>/model.pt --download cuhk            # fetch CUHK, evaluate
    python -m training.bokeh.eval_real --model <run>/model.pt --dut-dir <DUT-DBD folder> # your own copy
    python -m training.bokeh.eval_real --model <run>/model.pt --rtf-dir <RTF folder>

**CUHK** (Shi, Xu, Jia, CVPR 2014; 1,000 photos from the internet, human masks): downloaded from the authors' page
(`BlurDatasetImage.zip` 122 MB, `BlurDatasetGT.zip`) into `testdata/bokeh-eval/cuhk/`; the SHA-256 of both archives is
written on the first download and checked afterwards. Only the 704 `out_of_focus*` photos are used (the other 296 show
partial motion blur, which this network does not model). A mask's polarity is not documented, so it is read from the data:
the side of the masks with the larger mean gradient energy is "in focus"; the script prints which that was.
**DUT-DBD** (Zhao et al., CVPR 2018) and **RTF** (Park et al., CVPR 2017) are not downloaded: their hosts are Google Drive,
Baidu and personal pages that change. Give a folder you have fetched yourself: `--dut-dir` with the photos and the masks in
two folders (any names; the one with `.png` files holds the masks, matched to the photos by file name stem), `--rtf-dir`
with the photos and their ground-truth blur maps in two folders the same way. The RTF reader assumes the map holds the
blur radius in pixels of the photo as an 8-bit or float image (`--rtf-scale` divides it); that encoding is unverified
here (no copy of RTF was available), so check the numbers against the dataset's own readme before trusting them.

**Numbers.** For CUHK and DUT: the in-focus / out-of-focus agreement of the prediction `b >= 2 * b_min` with the mask (pixel
accuracy, and precision, recall and F1 of the out-of-focus class), ungated and with the confidence gate (`c >= 0.5`; a
pixel the network is unsure about counts as in focus, the spec's precision-over-recall rule), for the network and for
the classical baseline. For RTF: the mean absolute error of `b` against the ground-truth radius over the pixels with
blur (`b_gt >= b_min`), in half-diagonal units. Photos are run at their own size (long edges of 500 to 640 px, so `b_min`
is 1.3 to 1.6 px there: smaller discs than the network was trained on; `--long-edge 1024` enlarges them first, which
softens them), padded to a multiple of 32.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.bokeh import classical
from training.bokeh.encode import B_MIN, encode_t, radius_and_confidence, srgb_to_linear
from training.bokeh.model import BokehNet

CUHK = {
    "BlurDatasetImage.zip": "https://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/data/BlurDatasetImage.zip",
    "BlurDatasetGT.zip": "https://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/data/BlurDatasetGT.zip",
}
GATE = 0.5


def download_cuhk(root: Path) -> Path:
    """The CUHK archives into `root/cuhk` (and unpacked), with their SHA-256 recorded and checked."""
    folder = root / "cuhk"
    folder.mkdir(parents=True, exist_ok=True)
    sums = folder / "SHA256SUMS"
    known = dict(line.split()[::-1] for line in sums.read_text().splitlines()) if sums.exists() else {}
    for name, url in CUHK.items():
        path = folder / name
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "yellowmoon-bokeh-eval/0.1"})
            with urllib.request.urlopen(req, timeout=600) as r, path.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in known and known[name] != digest:
            raise SystemExit(f"{path} does not match the SHA-256 recorded on first download")
        known[name] = digest
        with zipfile.ZipFile(path) as z:
            z.extractall(folder)
    sums.write_text("".join(f"{d}  {n}\n" for n, d in sorted(known.items())))
    return folder


def load_cuhk(folder: Path, limit: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, rgb uint8, mask bool where 255) of the out-of-focus photos."""
    out = []
    for p in sorted((folder / "image").glob("out_of_focus*.jpg"))[:limit]:
        gt = folder / "gt" / f"{p.stem}.png"
        if gt.exists():
            out.append((p.stem, np.asarray(Image.open(p).convert("RGB")), np.asarray(Image.open(gt).convert("L")) > 127))
    return out


def pairs_from(folder: Path, limit: int) -> list[tuple[str, Path, Path]]:
    """Photos and masks (or maps) in the two sub-folders of `folder`, matched by file name stem."""
    subs = [d for d in sorted(folder.iterdir()) if d.is_dir()] or [folder]
    photos = {p.stem: p for d in subs for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    maps = {p.stem: p for d in subs for p in d.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff", ".npy")}
    return [(k, photos[k], maps[k]) for k in sorted(photos) if k in maps][:limit]


def predict(model: BokehNet, rgb: np.ndarray, long_edge: int = 0) -> dict:
    """The network's and the baseline's answers for a photo, as maps at the photo's own size (after the optional
    resize): `b` and confidence for each."""
    img = Image.fromarray(rgb)
    if long_edge and max(img.size) != long_edge:
        s = long_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    lin = srgb_to_linear(torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255)
    e = encode_t(lin)
    ph, pw = -h % 32, -w % 32
    with torch.no_grad():
        out = model(F.pad(e, (0, pw, 0, ph), mode="replicate"))
    r, _, conf = radius_and_confidence(out)
    hd = float(np.hypot(w, h) / 2)
    up = lambda t: F.interpolate(t.unsqueeze(1), scale_factor=2, mode="bilinear", align_corners=False)[0, 0, :h, :w].numpy()  # noqa: E731
    base_r, base_c = classical.estimate(classical.gray_from_encoded(e[0].double().numpy()))
    return {"model": (up(r) / hd, up(conf)), "classical": (base_r / hd, base_c), "size": (w, h), "e": e[0].numpy()}


def gradient_energy(gray: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    g = classical.gradient_magnitude(gray) ** 2
    return float(g[mask].sum()), int(mask.sum())


class Seg:
    """Pixel counts of the in-focus / out-of-focus agreement."""

    def __init__(self):
        self.c = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)

    def add(self, pred_blur: np.ndarray, gt_blur: np.ndarray) -> None:
        self.c["tp"] += int((pred_blur & gt_blur).sum())
        self.c["fp"] += int((pred_blur & ~gt_blur).sum())
        self.c["fn"] += int((~pred_blur & gt_blur).sum())
        self.c["tn"] += int((~pred_blur & ~gt_blur).sum())

    def result(self) -> dict:
        c = self.c
        n = sum(c.values())
        p, r = c["tp"] / max(1, c["tp"] + c["fp"]), c["tp"] / max(1, c["tp"] + c["fn"])
        return {"accuracy": round((c["tp"] + c["tn"]) / max(1, n), 4), "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / max(1e-9, p + r), 4)}


def evaluate_segmentation(model: BokehNet, items, long_edge: int) -> dict:
    """`items`: (name, rgb, mask) with the mask's polarity unknown; it is read from the data (module docstring)."""
    preds, white, black = [], [0.0, 0], [0.0, 0]
    for name, rgb, mask in items:
        got = predict(model, rgb, long_edge)
        if got["size"] != (rgb.shape[1], rgb.shape[0]):  # resized: bring the mask along
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(got["size"], Image.NEAREST)) > 127
        gray = np.asarray(Image.fromarray(rgb).convert("L").resize(got["size"]), np.float64) / 255
        for acc, m in ((white, mask), (black, ~mask)):
            g, n = gradient_energy(gray, m)
            acc[0] += g
            acc[1] += n
        preds.append((got, mask))
    white_sharper = white[0] / max(1, white[1]) > black[0] / max(1, black[1])
    segs = {k: Seg() for k in ("model", "model_gated", "classical", "classical_gated")}
    for got, mask in preds:
        gt_blur = ~mask if white_sharper else mask
        for key in ("model", "classical"):
            b, c = got[key]
            blur = b >= 2 * B_MIN
            segs[key].add(blur, gt_blur)
            segs[f"{key}_gated"].add(blur & (c >= GATE), gt_blur)
    return {"images": len(preds), "mask_polarity": "white = in focus" if white_sharper else "white = out of focus",
            **{k: v.result() for k, v in segs.items()}}


def load_rtf_map(path: Path, scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64) / scale
    return np.asarray(Image.open(path), np.float64) / scale


def evaluate_rtf(model: BokehNet, folder: Path, limit: int, long_edge: int, scale: float) -> dict:
    err = {"model": [0.0, 0], "classical": [0.0, 0]}
    n = 0
    for name, photo, gt in pairs_from(folder, limit):
        rgb = np.asarray(Image.open(photo).convert("RGB"))
        got = predict(model, rgb, long_edge)
        r_gt = load_rtf_map(gt, scale)
        if r_gt.ndim == 3:
            r_gt = r_gt[..., 0]
        w, h = got["size"]
        if r_gt.shape != (h, w):
            r_gt = np.asarray(Image.fromarray(r_gt.astype(np.float32)).resize((w, h), Image.NEAREST), np.float64)
        b_gt = r_gt / float(np.hypot(w, h) / 2)
        blurred = b_gt >= B_MIN
        for key in err:
            err[key][0] += float(np.abs(got[key][0] - b_gt)[blurred].sum())
            err[key][1] += int(blurred.sum())
        n += 1
    return {"images": n, **{f"{k}_mae_b": round(v[0] / max(1, v[1]), 5) for k, v in err.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="a run's model.pt")
    ap.add_argument("--data", type=Path, default=Path("testdata/bokeh-eval"))
    ap.add_argument("--download", choices=["cuhk"], help="fetch this set into --data (the other two are given by folder)")
    ap.add_argument("--dut-dir", type=Path)
    ap.add_argument("--rtf-dir", type=Path)
    ap.add_argument("--rtf-scale", type=float, default=1.0)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--long-edge", type=int, default=0, help="enlarge or reduce photos to this long edge first (0 = native)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    model = BokehNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    report: dict = {"note": "research-only data, local evaluation only; never trained on, never uploaded, never committed"}
    if args.download == "cuhk":
        download_cuhk(args.data)
    if (args.data / "cuhk" / "image").exists():
        report["cuhk"] = evaluate_segmentation(model, load_cuhk(args.data / "cuhk", args.max_images), args.long_edge)
    if args.dut_dir:
        items = [(k, np.asarray(Image.open(p).convert("RGB")), np.asarray(Image.open(m).convert("L")) > 127)
                 for k, p, m in pairs_from(args.dut_dir, args.max_images)]
        report["dut_dbd"] = evaluate_segmentation(model, items, args.long_edge)
    if args.rtf_dir:
        report["rtf"] = evaluate_rtf(model, args.rtf_dir, args.max_images, args.long_edge, args.rtf_scale)
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text)


if __name__ == "__main__":
    main()
