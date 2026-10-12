"""The `jdd` U-Net exactly as in ``docs/07-ml-models.md`` (section "jdd architecture").

Input ``x`` has shape ``[B, 7, h, w]`` at half resolution, in the physical units the runtime
(``crates/ym-ml``) feeds:

====  ==========================================================================================
chan  content
====  ==========================================================================================
0..3  packed Bayer planes R, G1, G2, B (RGGB), balanced with ``g0`` (white = 1.0 before gains)
4     sigma map: noise standard deviation in the same balanced units (RMS over the 4 sites)
5     quantisation-step map in the same balanced units (RMS over the 4 sites)
6     clip mask (1.0 where the photosite is clipped, else 0.0)
====  ==========================================================================================

Fixed input scaling happens *inside* the network (buffer ``in_scale``): planes by 0.25 (so typical
values sit near 0..1 and highlights above 1 survive), sigma and qstep by 16 (their typical values
are 1e-3..1e-1). The runtime therefore never has to know the scales. The output is linear RGB
at full resolution, ``[B, 3, 2h, 2w]``, in the same balanced units as the planes.

Layout (3 resolution levels, channels 32 / 64 / 96, two 3x3 convolutions per level, SiLU):

* ``enc0`` (half res) -> ``down1`` (3x3 stride-2 conv) -> ``enc1`` (quarter) -> ``down2`` ->
  ``bott`` (eighth);
* decoder: ``up1`` = 1x1 conv to ``4*C`` channels + pixel shuffle, concat the skip, ``dec1``;
  ``up0``, concat, ``dec0`` (no ConvTranspose and no Resize anywhere; ``up_kernel=3`` is an option);
* ``head``: 1x1 conv to 12 channels at half res, *added* to a fixed bilinear demosaic of the
  input (a 4 -> 12 convolution on the packed planes, see :func:`bilinear_weights`), then
  pixel shuffle x2 -> 3 channels at full resolution (residual learning).

Input height and width (half-res) must be multiples of 4 (two stride-2 stages).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 7
OUT_CHANNELS = 3
PLANE_SCALE = 0.25
SIGMA_SCALE = 16.0
QSTEP_SCALE = 16.0
LEVEL_STRIDE = 4  # input height/width multiple required by the two stride-2 stages
ARCH_NAME = "ym-jdd"
ARCH_VERSION = 1


@dataclass(frozen=True)
class JddConfig:
    """Architecture hyper-parameters (all of them are stored in the safetensors metadata)."""

    channels: tuple[int, int, int] = (32, 64, 96)
    convs_per_level: int = 2
    up_kernel: int = 1  # kernel of the conv in front of each pixel shuffle (1 or 3)
    head_kernel: int = 1  # kernel of the 12-channel output conv (1 or 3)
    plane_scale: float = PLANE_SCALE
    sigma_scale: float = SIGMA_SCALE
    qstep_scale: float = QSTEP_SCALE
    head_init_scale: float = 0.1  # the head starts near zero: the net starts as bilinear demosaic

    def to_json(self) -> str:
        """Canonical JSON (sorted keys) used in the safetensors metadata and the model card."""
        d = dataclasses.asdict(self)
        d["channels"] = list(self.channels)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> JddConfig:
        """Inverse of :meth:`to_json`."""
        d = json.loads(text)
        d["channels"] = tuple(d["channels"])
        return cls(**d)


def bilinear_weights() -> torch.Tensor:
    """Fixed bilinear demosaic as a ``[12, 4, 3, 3]`` convolution kernel on the packed planes.

    Input channels are the RGGB planes ``R, G1, G2, B`` (``R[i,j] = M[2i,2j]``,
    ``G1 = M[2i,2j+1]``, ``G2 = M[2i+1,2j]``, ``B = M[2i+1,2j+1]``). Output channel
    ``4*c + 2*a + b`` is colour ``c`` (0 R, 1 G, 2 B) at full-res pixel ``(2i+a, 2j+b)``, i.e. the
    channel order a PixelShuffle(2) turns into a full-resolution image. Correlation convention:
    ``out[i,j] = sum w[dy+1, dx+1] * in[i+dy, j+dx]``.
    """
    w = torch.zeros(12, 4, 3, 3, dtype=torch.float64)
    R, G1, G2, B = 0, 1, 2, 3

    def put(c: int, a: int, b: int, plane: int, taps: list[tuple[int, int]]) -> None:
        for dy, dx in taps:
            w[4 * c + 2 * a + b, plane, dy + 1, dx + 1] += 1.0 / len(taps)

    # (0,0): R site
    put(0, 0, 0, R, [(0, 0)])
    put(1, 0, 0, G1, [(0, -1), (0, 0)])  # left, right
    put(1, 0, 0, G2, [(-1, 0), (0, 0)])  # up, down
    w[4 * 1 + 0, :, :, :] *= 0.5  # average of the four neighbours: halve each pair
    put(2, 0, 0, B, [(-1, -1), (-1, 0), (0, -1), (0, 0)])
    # (0,1): G1 site
    put(1, 0, 1, G1, [(0, 0)])
    put(0, 0, 1, R, [(0, 0), (0, 1)])
    put(2, 0, 1, B, [(-1, 0), (0, 0)])
    # (1,0): G2 site
    put(1, 1, 0, G2, [(0, 0)])
    put(0, 1, 0, R, [(0, 0), (1, 0)])
    put(2, 1, 0, B, [(0, -1), (0, 0)])
    # (1,1): B site
    put(2, 1, 1, B, [(0, 0)])
    put(1, 1, 1, G2, [(0, 0), (0, 1)])  # left, right
    put(1, 1, 1, G1, [(0, 0), (1, 0)])  # up, down
    w[4 * 1 + 3, :, :, :] *= 0.5
    put(0, 1, 1, R, [(0, 0), (0, 1), (1, 0), (1, 1)])
    return w.to(torch.float32)


class Block(nn.Module):
    """``convs_per_level`` 3x3 convolutions, each followed by SiLU."""

    def __init__(self, cin: int, cout: int, n: int = 2) -> None:
        super().__init__()
        self.n = n
        for i in range(n):
            setattr(self, f"conv{i + 1}", nn.Conv2d(cin if i == 0 else cout, cout, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for i in range(self.n):
            x = F.silu(getattr(self, f"conv{i + 1}")(x))
        return x


class JddNet(nn.Module):
    """Joint demosaic + denoise U-Net. See the module docstring for the contract."""

    def __init__(self, cfg: JddConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or JddConfig()
        self.cfg = cfg
        c0, c1, c2 = cfg.channels
        n = cfg.convs_per_level
        uk = cfg.up_kernel
        scale = torch.tensor(
            [cfg.plane_scale] * 4 + [cfg.sigma_scale, cfg.qstep_scale, 1.0], dtype=torch.float32
        ).view(1, IN_CHANNELS, 1, 1)
        self.register_buffer("in_scale", scale)
        self.enc0 = Block(IN_CHANNELS, c0, n)
        self.down1 = nn.Conv2d(c0, c1, 3, 2, 1)
        self.enc1 = Block(c1, c1, n)
        self.down2 = nn.Conv2d(c1, c2, 3, 2, 1)
        self.bott = Block(c2, c2, n)
        self.up1 = nn.Conv2d(c2, c1 * 4, uk, 1, uk // 2)
        self.dec1 = Block(2 * c1, c1, n)
        self.up0 = nn.Conv2d(c1, c0 * 4, uk, 1, uk // 2)
        self.dec0 = Block(2 * c0, c0, n)
        self.head = nn.Conv2d(c0, 12, cfg.head_kernel, 1, cfg.head_kernel // 2)
        self.bilinear = nn.Conv2d(4, 12, 3, 1, 1, bias=False)
        with torch.no_grad():
            self.bilinear.weight.copy_(bilinear_weights())
            self.head.weight.mul_(cfg.head_init_scale)
            self.head.bias.zero_()
        self.bilinear.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B,7,h,w]`` physical units -> ``[B,3,2h,2w]`` linear RGB (balanced units)."""
        x = x * self.in_scale
        planes = x[:, 0:4]
        e0 = self.enc0(x)
        e1 = self.enc1(F.silu(self.down1(e0)))
        b = self.bott(F.silu(self.down2(e1)))
        u1 = F.silu(F.pixel_shuffle(self.up1(b), 2))
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        u0 = F.silu(F.pixel_shuffle(self.up0(d1), 2))
        d0 = self.dec0(torch.cat([u0, e0], dim=1))
        y = self.head(d0) + self.bilinear(planes)
        return F.pixel_shuffle(y, 2) * (1.0 / self.cfg.plane_scale)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that are trained (the bilinear kernel is fixed)."""
        return [p for p in self.parameters() if p.requires_grad]


def check_input_shape(h: int, w: int) -> None:
    """Raise ``ValueError`` unless a half-res ``h x w`` input is valid for the network."""
    if h % LEVEL_STRIDE or w % LEVEL_STRIDE or h <= 0 or w <= 0:
        raise ValueError(f"half-res input must be a positive multiple of {LEVEL_STRIDE}, got {h}x{w}")


def count_parameters(model: JddNet) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.trainable_parameters())


def count_flops(model: JddNet, h: int, w: int) -> dict[str, float]:
    """Multiply-add FLOPs (2 per MAC) of the convolutions for one ``h x w`` half-res input.

    Returns ``total``, ``network`` (without the fixed bilinear conv), ``per_output_pixel`` (network
    FLOPs divided by the ``4*h*w`` full-resolution output pixels, the quantity the docs/07 budget of
    250 kFLOP refers to) and ``per_output_pixel_with_bilinear``. SiLU, concat, add and pixel shuffle
    are not counted (they are a few percent at most).
    """
    check_input_shape(h, w)
    counts: dict[str, float] = {}
    hooks = []

    def make_hook(name: str):
        def hook(mod: nn.Conv2d, _inp, out: torch.Tensor) -> None:
            macs = out.shape[1] * out.shape[2] * out.shape[3] * mod.in_channels // mod.groups
            macs *= mod.kernel_size[0] * mod.kernel_size[1]
            counts[name] = 2.0 * macs

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, h, w))
    model.train(was_training)
    for hk in hooks:
        hk.remove()
    total = sum(counts.values())
    net = total - counts.get("bilinear", 0.0)
    out_px = 4.0 * h * w
    return {
        "total": total,
        "network": net,
        "per_output_pixel": net / out_px,
        "per_output_pixel_with_bilinear": total / out_px,
    }


def named_weight_tensors(model: JddNet) -> dict[str, torch.Tensor]:
    """State dict as contiguous CPU float32 tensors (what the safetensors file stores)."""
    return {k: v.detach().cpu().contiguous().to(torch.float32) for k, v in model.state_dict().items()}


@dataclass
class ModelInfo:
    """Facts about a model that go into the safetensors metadata, the model card and the logs."""

    params: int
    rf_half_px: int
    rf_full_px: int
    halo_half_px: int
    halo_full_px: int
    flops_per_output_pixel: float
    extra: dict[str, str] = field(default_factory=dict)
