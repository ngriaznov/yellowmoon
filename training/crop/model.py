"""The crop student: a small convolutional network trained from scratch (no pretrained weights, `docs/14` rule 5)
that maps a letterboxed photo to one crop, as fractions of the photo.

Inverted-residual stages (MobileNetV2-style) take the 256 x 256 square to an 8 x 8 grid; the head keeps that grid
(where things are matters for a crop) and answers four numbers p in [0, 1] that always make a valid box inside the
photo: x0 = p0, x1 = x0 + (1 - x0) p2, and the same for y with p1 and p3. The `box` head regresses them (a sigmoid);
the `bins` head classifies each into `BINS` bins and answers the most likely bin, refined by its neighbours, so that
when the photo could take a firm crop or none it picks one instead of averaging the two.
"""

import torch
from torch import nn

#: (expansion, output channels, blocks, stride) per stage, after a stride-2 stem of 24 channels.
STAGES = ((1, 24, 1, 1), (4, 32, 2, 2), (4, 48, 3, 2), (4, 96, 3, 2), (4, 160, 2, 2))
HEAD_CHANNELS = 32
HIDDEN = 256
#: Bins of the `bins` head per number, their centres spread evenly over [0, 1] ends included.
BINS = 41


class InvertedResidual(nn.Module):
    """1x1 expand, 3x3 depthwise (stride s), 1x1 project; a residual when the shape is kept."""

    def __init__(self, cin: int, cout: int, stride: int, expand: int):
        super().__init__()
        mid = cin * expand
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()]
        layers += [
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout),
        ]
        self.body = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return x + y if self.residual else y


class CropNet(nn.Module):
    """Letterboxed RGB in [0, 1] (N, 3, 256, 256) -> crop (N, 4) as `[x0, y0, x1, y1]` fractions of the photo."""

    def __init__(self, size: int = 256, head: str = "box"):
        super().__init__()
        if head not in ("box", "bins"):
            raise ValueError(f"unknown head {head!r}")
        self.kind = head
        layers: list[nn.Module] = [nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU()]
        cin = 24
        #: Index in `features` of the last layer of each stage (stride 2, 4, 8, 16, 32 after the stem's 2).
        self.stage_ends: list[int] = []
        for expand, cout, blocks, stride in STAGES:
            for b in range(blocks):
                layers.append(InvertedResidual(cin, cout, stride if b == 0 else 1, expand))
                cin = cout
            self.stage_ends.append(len(layers) - 1)
        layers += [nn.Conv2d(cin, HEAD_CHANNELS, 1, bias=False), nn.BatchNorm2d(HEAD_CHANNELS), nn.SiLU()]
        self.features = nn.Sequential(*layers)
        grid = size // 32
        outputs = 4 if head == "box" else 4 * BINS
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(HEAD_CHANNELS * grid * grid, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, outputs)
        )
        last = self.head[-1]
        nn.init.normal_(last.weight, std=1e-3)
        with torch.no_grad():
            if head == "box":
                # Start from a large crop (5 % in from each edge), near the teacher's typical answer, not from the
                # small centred box a zero output would give.
                last.bias.copy_(torch.tensor([-2.944, -2.944, 2.89, 2.89]))
            else:
                last.bias.zero_()

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """The head's output: (N, 4) before the sigmoid for `box`, (N, 4, BINS) logits for `bins`."""
        a = self.head(self.features((x - 0.5) / 0.25))
        return a if self.kind == "box" else a.view(-1, 4, BINS)

    def numbers(self, x: torch.Tensor) -> torch.Tensor:
        """The four numbers p in [0, 1] (module docstring)."""
        a = self.raw(x)
        return torch.sigmoid(a) if self.kind == "box" else decode_bins(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return box_from_numbers(self.numbers(x))

    def taps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The features at stride 16 and at stride 32 (before the head's 1x1), for the pretraining heads."""
        y = (x - 0.5) / 0.25
        out = []
        for i, layer in enumerate(self.features):
            y = layer(y)
            if i in (self.stage_ends[-2], self.stage_ends[-1]):
                out.append(y)
                if len(out) == 2:
                    return out[0], out[1]
        raise AssertionError("unreachable: the stages end before the head")


def box_from_numbers(p: torch.Tensor) -> torch.Tensor:
    """(N, 4) numbers in [0, 1] -> `[x0, y0, x1, y1]` boxes (module docstring)."""
    x0, y0 = p[:, 0], p[:, 1]
    x1 = x0 + (1 - x0) * p[:, 2]
    y1 = y0 + (1 - y0) * p[:, 3]
    return torch.stack([x0, y0, x1, y1], dim=1)


def numbers_from_box(b: torch.Tensor) -> torch.Tensor:
    """The inverse of `box_from_numbers`, clipped to [0, 1]."""
    x0, y0 = b[:, 0].clamp(0, 1), b[:, 1].clamp(0, 1)
    p2 = ((b[:, 2] - x0) / (1 - x0).clamp(min=1e-6)).clamp(0, 1)
    p3 = ((b[:, 3] - y0) / (1 - y0).clamp(min=1e-6)).clamp(0, 1)
    return torch.stack([x0, y0, p2, p3], dim=1)


def bin_target(p: torch.Tensor) -> torch.Tensor:
    """(N, 4) numbers -> (N, 4, BINS) soft targets: each number split between its two nearest bin centres."""
    pos = p.clamp(0, 1) * (BINS - 1)
    lo = pos.floor().clamp(max=BINS - 2)
    w = pos - lo
    t = torch.zeros(*p.shape, BINS, device=p.device, dtype=p.dtype)
    t.scatter_(-1, lo.long()[..., None], (1 - w)[..., None])
    t.scatter_add_(-1, (lo.long() + 1)[..., None], w[..., None])
    return t


def decode_bins(logits: torch.Tensor) -> torch.Tensor:
    """(N, 4, BINS) logits -> (N, 4) numbers: the most likely bin, refined by the probabilities of it and its two
    neighbours (their weighted mean), so the answer is not quantised to the bins yet never averages two far modes."""
    prob = torch.softmax(logits.float(), dim=-1)
    k = prob.argmax(dim=-1, keepdim=True)
    idx = torch.cat([k - 1, k, k + 1], dim=-1).clamp(0, BINS - 1)
    near = prob.gather(-1, idx)
    # At an end, the clamped neighbour repeats the end bin; count it once.
    near = torch.where(torch.cat([k == 0, torch.zeros_like(k, dtype=torch.bool), k == BINS - 1], -1), 0, near)
    centres = idx.to(prob.dtype) / (BINS - 1)
    return (near * centres).sum(-1) / near.sum(-1).clamp(min=1e-9)


def parameters(model: nn.Module) -> int:
    """The number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Intersection over union of matching rows of two (N, 4) box tensors."""
    ix = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter).clamp(min=1e-9)
