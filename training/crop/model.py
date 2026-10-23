"""The crop student: a small convolutional network trained from scratch (no pretrained weights, `docs/14` rule 5)
that maps a letterboxed photo to one crop, as fractions of the photo.

Inverted-residual stages (MobileNetV2-style) take the 256 x 256 square to an 8 x 8 grid; the head keeps that grid
(where things are matters for a crop) and regresses the crop's edges. The parametrisation keeps every answer a
valid box inside the photo: x0 = s(a0), x1 = x0 + (1 - x0) s(a2), and the same for y.
"""

import torch
from torch import nn

#: (expansion, output channels, blocks, stride) per stage, after a stride-2 stem of 24 channels.
STAGES = ((1, 24, 1, 1), (4, 32, 2, 2), (4, 48, 3, 2), (4, 96, 3, 2), (4, 160, 2, 2))
HEAD_CHANNELS = 32
HIDDEN = 256


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

    def __init__(self, size: int = 256):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU()]
        cin = 24
        for expand, cout, blocks, stride in STAGES:
            for b in range(blocks):
                layers.append(InvertedResidual(cin, cout, stride if b == 0 else 1, expand))
                cin = cout
        layers += [nn.Conv2d(cin, HEAD_CHANNELS, 1, bias=False), nn.BatchNorm2d(HEAD_CHANNELS), nn.SiLU()]
        self.features = nn.Sequential(*layers)
        grid = size // 32
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(HEAD_CHANNELS * grid * grid, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 4)
        )
        # Start from a large crop (5 % in from each edge), near the teacher's typical answer, not from the small
        # centred box a zero output would give.
        last = self.head[-1]
        nn.init.normal_(last.weight, std=1e-3)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([-2.944, -2.944, 2.89, 2.89]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.head(self.features((x - 0.5) / 0.25))
        s = torch.sigmoid(a)
        x0, y0 = s[:, 0], s[:, 1]
        x1 = x0 + (1 - x0) * s[:, 2]
        y1 = y0 + (1 - y0) * s[:, 3]
        return torch.stack([x0, y0, x1, y1], dim=1)


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
