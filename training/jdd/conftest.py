"""pytest setup: make ``training.jdd`` importable when pytest is run as ``pytest training/jdd``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The tests use tiny tensors; OpenMP busy-waiting with several threads is far slower than one thread on a
# shared or loaded machine (CI runners included).
import torch  # noqa: E402

torch.set_num_threads(1)
