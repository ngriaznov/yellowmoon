"""Preview sheets of the synthesis: one row per sample with the network's input (the encoded plane), a display version
of the same plane, the label (`b = r / half-diagonal` as colour, dark where the weight is low) and the loss weight (grey; cyan
on thin structures, magenta on featureless layers). For a human to check what the network is trained on; `modal_app.py`'s
`preview` and `smoke.py` write them. The default is the v3 recipe (`SynthConfig.v3()`); pass `SynthConfig()` for bokeh-2's.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.bokeh import sources
from training.bokeh.synth import SynthConfig, TileBatch, preview_grid, synthesize


def render(tiles: Path, out: Path, sheets: int = 3, rows: int = 6, seed: int = 100, device: str = "cpu",
           cfg: SynthConfig | None = None, log=print) -> list[Path]:
    """Writes `sheets` PNGs of `rows` samples each into `out`; returns their paths."""
    ts = sources.TileSet(tiles)
    pool = ts.train or list(range(len(ts)))
    with_c = [i for i in pool if ts.rows[i][1]["cands"]]
    cfg = cfg or SynthConfig.v3()
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(sheets):
        gen = torch.Generator(device=device).manual_seed(seed + k)
        rng = np.random.default_rng(seed + k)
        idx = [int(i) for i in rng.choice(with_c, min(2, rows))] if with_c else []
        idx += [int(i) for i in rng.choice(pool, rows - len(idx))]
        got = [ts.get(i) for i in idx]
        tb = TileBatch(torch.from_numpy(np.stack([g[0] for g in got])), torch.from_numpy(np.stack([g[1] for g in got])),
                       torch.from_numpy(np.stack([g[2] for g in got])), torch.tensor([g[3] for g in got], dtype=torch.float32))
        t0 = time.time()
        sy = synthesize(tb.to(device), cfg, gen)
        log(f"sheet {k}: {time.time() - t0:.1f} s; modes {sy.info['mode'].tolist()} night {sy.info['night'].tolist()}")
        path = out / f"synth-sheet-{k}.png"
        Image.fromarray(preview_grid(sy).numpy()).save(path)
        paths.append(path)
    return paths
