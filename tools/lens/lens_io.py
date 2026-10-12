"""Reading the images written by darktable_oracle.sh: gzipped PFM files (float32 RGB, linear)."""
import gzip

import numpy as np


def read_pfm(path):
    """Reads a (gzipped) colour PFM into a float64 array `h x w x 3`, top row first."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        header = f.readline().strip()
        if header not in (b"PF", b"Pf"):
            raise ValueError(f"{path}: not a PFM file")
        w, h = (int(v) for v in f.readline().split())
        scale = float(f.readline())
        channels = 3 if header == b"PF" else 1
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, channels)
    data = data[::-1].astype(np.float64)  # PFM stores the bottom row first
    return data if channels == 3 else np.repeat(data, 3, axis=2)


def load_pair(directory):
    """(lens input, lens output) of one oracle render directory, linear camera RGB."""
    from pathlib import Path
    d = Path(directory)
    return read_pfm(d / "lens_in.pfm.gz"), read_pfm(d / "lens_out.pfm.gz")
