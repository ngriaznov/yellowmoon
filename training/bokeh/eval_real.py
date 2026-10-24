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
    python -m training.bokeh.eval_real --model <run>/model.pt --tta --max-images 704     # all of CUHK, with the mirrored pass too
    python -m training.bokeh.eval_real --model <run>/model.pt --dut-dir <DUT-DBD folder> # your own copy
    python -m training.bokeh.eval_real --model <run>/model.pt --rtf-dir <RTF folder>

**CUHK** (Shi, Xu, Jia, CVPR 2014; 1,000 photos from the internet, human masks): downloaded from the authors' page
(`BlurDatasetImage.zip` 122 MB, `BlurDatasetGT.zip`) into `testdata/bokeh-eval/cuhk/`; the SHA-256 of both archives is
written on the first download and checked afterwards; the archives are deleted once the 704 `out_of_focus*` photos and their masks
are extracted (the other 296 show partial motion blur, which this network does not model; 42 of the 704 have the extension `.JPG`). A mask's polarity is not documented, so it is read from the data:
the side of the masks with the larger mean gradient energy is "in focus"; the script prints which that was.
**DUT-DBD** (Zhao et al., CVPR 2018) and **RTF** (Park et al., CVPR 2017) are not downloaded: their hosts are Google Drive,
Baidu and personal pages that change. Give a folder you have fetched yourself: `--dut-dir` with the photos and the masks in
two folders (any names; the one with `.png` files holds the masks, matched to the photos by file name stem), `--rtf-dir`
with the photos and their ground-truth blur maps in two folders the same way. The RTF reader assumes the map holds the
blur radius in pixels of the photo as an 8-bit or float image (`--rtf-scale` divides it); that encoding is unverified
here (no copy of RTF was available), so check the numbers against the dataset's own readme before trusting them.

**Numbers.** For CUHK and DUT: the in-focus / out-of-focus agreement of a decision rule with the mask: pixel accuracy, and
precision, recall and F1 of the out-of-focus class over all pixels, and the F1 of each photo (its mean, standard deviation and
quartiles over the photos that have out-of-focus pixels: the spread says how much a rule depends on the photo). The rules, for the
network: `raw` (`b >= 2 * b_min`), `gate c>=0.5` (the confidence gate of bokeh-2's report: `c` says how sure the *radius* is,
so it is low inside strong blur and the gate throws good blur away), `P>=t` (the runtime's decision, `encode.detection_probability`:
the probability, from the Laplace output of the pixel itself, that `b >= 2 * b_min`; t of 0.5, 0.7 and 0.9) and `quantile`
(`b * exp(-1.61 beta) >= 2 * b_min`, the lower 10 % quantile: the same as `P>=0.9`, written out). With `--tta` the network
also runs on the mirrored photo and the two outputs are averaged (the log radius; the scale is the mean scale plus half the
disagreement), and `tta ...` rules are added. For the classical baseline (`classical.py`, a stripped-down Zhuo and Sim, not
the spec's detector in Rust): `raw` and `gate c>=0.5`. For RTF: the mean absolute error of `b` against the ground-truth radius
over the pixels with blur (`b_gt >= b_min`), in half-diagonal units. Photos are run at their own size (long edges of 500 to 640 px,
so `b_min` is 1.3 to 1.6 px there: smaller discs than the network was trained on; `--long-edge 1024` enlarges them first, which
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
from training.bokeh.encode import B_DECIDE, B_MIN, CONF_BETA0, detection_probability, encode_t, srgb_to_linear
from training.bokeh.model import BokehNet

CUHK = {
    "BlurDatasetImage.zip": "https://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/data/BlurDatasetImage.zip",
    "BlurDatasetGT.zip": "https://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/data/BlurDatasetGT.zip",
}
GATE = 0.5
#: The decision thresholds on the probability that `b >= 2 * b_min` that are reported.
P_THRESHOLDS = (0.5, 0.7, 0.9)


def download_cuhk(root: Path) -> Path:
    """The CUHK archives into `root/cuhk`: SHA-256 recorded on the first download and checked afterwards, only the
    `out_of_focus*` photos and masks unpacked, the archives deleted (disk). Nothing is fetched when the photos are there."""
    folder = root / "cuhk"
    folder.mkdir(parents=True, exist_ok=True)
    if len(list((folder / "image").glob("out_of_focus*"))) >= 704 and len(list((folder / "gt").glob("out_of_focus*"))) >= 704:
        return folder
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
            for member in z.namelist():
                if "out_of_focus" in member:
                    z.extract(member, folder)
        path.unlink()
    sums.write_text("".join(f"{d}  {n}\n" for n, d in sorted(known.items())))
    return folder


def load_cuhk(folder: Path, limit: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, rgb uint8, mask bool where 255) of the out-of-focus photos (`.jpg` and `.JPG`)."""
    out = []
    photos = sorted(p for p in (folder / "image").iterdir() if p.suffix.lower() == ".jpg" and p.name.startswith("out_of_focus"))
    for p in photos[:limit]:
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


def run_network(model: BokehNet, e: torch.Tensor, mirrored: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`mu` and `s` (half resolution, cropped to the photo) of an encoded photo (1, 3, h, w); `mirrored` runs it flipped left
    to right and flips the answer back."""
    h, w = e.shape[-2:]
    x = F.pad(e, (0, -w % 32, 0, -h % 32), mode="replicate")
    with torch.no_grad():
        out = model(x.flip(3) if mirrored else x)
    out = (out.flip(3) if mirrored else out)[0, :, : (h + 1) // 2, : (w + 1) // 2]
    return out[0].numpy().copy(), out[1].clamp(-6, 3).numpy().copy()


def combine_mirrored(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The two outputs of the plain and the mirrored pass as one: the mean log radius, and a scale that is the mean scale plus
    half the disagreement of the two radii (where they differ the answer is less sure)."""
    mu = 0.5 * (a[0] + b[0])
    beta = 0.5 * (np.exp(a[1]) + np.exp(b[1])) + 0.5 * np.abs(a[0] - b[0])
    return mu, np.log(beta)


def predict(model: BokehNet, rgb: np.ndarray, long_edge: int = 0, tta: bool = False) -> dict:
    """The network's and the baseline's answers for a photo at the photo's own size (after the optional resize): the network's
    `(mu, s)` bilinearly doubled (plain and, with `tta`, mirrored and combined) and the baseline's `b` and confidence."""
    img = Image.fromarray(rgb)
    if long_edge and max(img.size) != long_edge:
        s = long_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    e = encode_t(srgb_to_linear(torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255))
    hd = float(np.hypot(w, h) / 2)
    up = lambda a: F.interpolate(torch.from_numpy(a)[None, None], scale_factor=2, mode="bilinear", align_corners=False)[0, 0, :h, :w].numpy()  # noqa: E731
    plain = run_network(model, e)
    nets = {"": plain}
    if tta:
        nets["tta "] = combine_mirrored(plain, run_network(model, e, mirrored=True))
    base_r, base_c = classical.estimate(classical.gray_from_encoded(e[0].double().numpy()))
    return {"nets": {k: (up(mu), up(s)) for k, (mu, s) in nets.items()}, "hd": hd, "classical": (base_r / hd, base_c), "size": (w, h)}


def rules(got: dict) -> dict[str, np.ndarray]:
    """Every decision rule's out-of-focus map (bool, the photo's size) for one photo (module docstring)."""
    hd = got["hd"]
    out = {}
    for tag, (mu, s) in got["nets"].items():
        beta = np.exp(s)
        b = np.exp(mu) / hd
        out[f"{tag}raw"] = b >= B_DECIDE
        out[f"{tag}gate c>={GATE}"] = (b >= B_DECIDE) & (np.exp(-beta / CONF_BETA0) >= GATE)
        p = detection_probability(torch.from_numpy(mu), torch.from_numpy(s), hd).numpy()
        for t in P_THRESHOLDS:
            out[f"{tag}P>={t}"] = p >= t
        out[f"{tag}quantile"] = (mu - 1.61 * beta) >= np.log(B_DECIDE * hd)
    b, c = got["classical"]
    out["classical raw"] = b >= B_DECIDE
    out[f"classical gate c>={GATE}"] = (b >= B_DECIDE) & (c >= GATE)
    return out


def gradient_energy(gray: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    g = classical.gradient_magnitude(gray) ** 2
    return float(g[mask].sum()), int(mask.sum())


class Seg:
    """Pixel counts of the in-focus / out-of-focus agreement, overall and per photo."""

    def __init__(self):
        self.c = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
        self.per: list[tuple[int, int, int, int]] = []

    def add(self, pred_blur: np.ndarray, gt_blur: np.ndarray) -> None:
        tp, fp = int((pred_blur & gt_blur).sum()), int((pred_blur & ~gt_blur).sum())
        fn, tn = int((~pred_blur & gt_blur).sum()), int((~pred_blur & ~gt_blur).sum())
        for k, v in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
            self.c[k] += v
        self.per.append((tp, fp, fn, tn))

    def result(self) -> dict:
        c = self.c
        n = sum(c.values())
        p, r = c["tp"] / max(1, c["tp"] + c["fp"]), c["tp"] / max(1, c["tp"] + c["fn"])
        per = np.array(self.per, float)
        has = per[:, 0] + per[:, 2] > 0.01 * per.sum(1)  # photos with out-of-focus pixels to find
        f1s = (2 * per[:, 0] / np.maximum(1, 2 * per[:, 0] + per[:, 1] + per[:, 2]))[has]
        spread = {"f1_photo_mean": float(f1s.mean()), "f1_photo_std": float(f1s.std()), "f1_photo_q25": float(np.percentile(f1s, 25)),
                  "f1_photo_median": float(np.median(f1s)), "f1_photo_q75": float(np.percentile(f1s, 75)), "photos": int(has.sum())} if has.any() else {}
        return {"accuracy": round((c["tp"] + c["tn"]) / max(1, n), 4), "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / max(1e-9, p + r), 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in spread.items()}}


def evaluate_segmentation(model: BokehNet, items, long_edge: int, tta: bool = False) -> dict:
    """`items`: (name, rgb, mask) with the mask's polarity unknown; it is read from the data (module docstring)."""
    preds, white, black = [], [0.0, 0], [0.0, 0]
    for name, rgb, mask in items:
        got = predict(model, rgb, long_edge, tta)
        if got["size"] != (rgb.shape[1], rgb.shape[0]):  # resized: bring the mask along
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(got["size"], Image.NEAREST)) > 127
        gray = np.asarray(Image.fromarray(rgb).convert("L").resize(got["size"]), np.float64) / 255
        for acc, m in ((white, mask), (black, ~mask)):
            g, n = gradient_energy(gray, m)
            acc[0] += g
            acc[1] += n
        preds.append((rules(got), mask))
    white_sharper = white[0] / max(1, white[1]) > black[0] / max(1, black[1])
    segs: dict[str, Seg] = {}
    for maps, mask in preds:
        gt_blur = ~mask if white_sharper else mask
        for name, pred in maps.items():
            segs.setdefault(name, Seg()).add(pred, gt_blur)
    return {"images": len(preds), "mask_polarity": "white = in focus" if white_sharper else "white = out of focus",
            "out_of_focus_share": round(float(np.mean([(~m if white_sharper else m).mean() for _, m in preds])), 4),
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
        mu, _ = got["nets"][""]
        for key, b_hat in (("model", np.exp(mu) / got["hd"]), ("classical", got["classical"][0])):
            err[key][0] += float(np.abs(b_hat - b_gt)[blurred].sum())
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
    ap.add_argument("--tta", action="store_true", help="also run the mirrored photo and report the combined rules (twice the time)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    model = BokehNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    report: dict = {"note": "research-only data, local evaluation only; never trained on, never uploaded, never committed"}
    if args.download == "cuhk":
        download_cuhk(args.data)
    if (args.data / "cuhk" / "image").exists():
        report["cuhk"] = evaluate_segmentation(model, load_cuhk(args.data / "cuhk", args.max_images), args.long_edge, args.tta)
    if args.dut_dir:
        items = [(k, np.asarray(Image.open(p).convert("RGB")), np.asarray(Image.open(m).convert("L")) > 127)
                 for k, p, m in pairs_from(args.dut_dir, args.max_images)]
        report["dut_dbd"] = evaluate_segmentation(model, items, args.long_edge, args.tta)
    if args.rtf_dir:
        report["rtf"] = evaluate_rtf(model, args.rtf_dir, args.max_images, args.long_edge, args.rtf_scale)
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text)


if __name__ == "__main__":
    main()
