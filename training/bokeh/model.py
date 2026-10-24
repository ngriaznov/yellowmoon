"""BokehNet: the bokeh map network (`docs/21` section 7). Trained from scratch (no pretrained weights, `docs/14`
rule 5); every operation has an exact, easy Rust counterpart so that `ym_ml` can run it in f32 on the CPU.

    input   (N, 3, H, W)  the encoded lens plane (`encode.py`), H and W divisible by 32; `forward` subtracts
            `INPUT_MEAN` and divides by `INPUT_STD`
    output  (N, 2, H/2, W/2)  channel 0 `mu = log r` (r the disc radius in pixels), channel 1 `s = log beta`
            (beta the Laplace scale of the error of mu)

Encoder: a MobileNetV2-style stack (stride-2 3x3 stem, inverted residual blocks with ReLU6) to stride 32. Global
context: the stride-32 features are averaged over the whole plane, passed through a 1x1 convolution and ReLU6 and
*added* to every cell, so a textureless region knows about the rest of the picture. Decoder: U-Net steps from stride 32
to stride 2: bilinear upsampling by 2 (`align_corners=False`: output pixel `i` reads input position `(i + 0.5) / 2 -
0.5`, clamped to the edge, which is what PyTorch's `F.interpolate(mode="bilinear", align_corners=False)` computes),
concatenation with a 1x1-projected skip of the encoder, then 1x1 convolution, 3x3 depthwise convolution and 1x1
convolution, each followed by ReLU6. No transposed convolutions, no resize but the one bilinear step.

Only these operations: convolution (1x1, 3x3, 3x3 depthwise; zero padding 1; stride 1 or 2), batch norm (folded into
the convolution at export), ReLU6, bilinear 2x upsample, add, concat, global average pooling.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from training.bokeh.encode import INPUT_MEAN, INPUT_STD


@dataclass(frozen=True)
class Arch:
    """The architecture; `export.py` writes it into the weights file's header, and the Rust runtime builds from it."""

    stem: int = 16
    #: (expansion, output channels, blocks, stride) per stage; the stem is stride 2, so the stages end at strides
    #: 4, 8, 16, 16, 32 and 32.
    stages: tuple = ((2, 24, 2, 2), (3, 32, 2, 2), (3, 64, 3, 2), (3, 96, 2, 1), (4, 224, 2, 2), (4, 416, 1, 1))
    #: Indices of the stages whose output is a skip (strides 4, 8, 16, 32); the stem's output is the stride-2 skip and
    #: the last tap is the bottleneck.
    taps: tuple = (0, 1, 3, 5)
    #: Decoder widths at strides 16, 8, 4, 2.
    decoder: tuple = (64, 40, 24, 16)
    out_channels: int = 2


ARCH = Arch()


def conv_bn(cin: int, cout: int, k: int = 1, stride: int = 1, groups: int = 1, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    """1x1 expand (absent when the expansion is 1), 3x3 depthwise (stride s), 1x1 project; a residual add when the
    shape is kept. ReLU6 after the first two, none after the projection."""

    def __init__(self, cin: int, cout: int, stride: int, expand: int):
        super().__init__()
        mid = cin * expand
        layers: list[nn.Module] = []
        if expand != 1:
            layers.append(conv_bn(cin, mid, 1))
        layers.append(conv_bn(mid, mid, 3, stride, groups=mid))
        layers.append(conv_bn(mid, cout, 1, act=False))
        self.body = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return x + y if self.residual else y


class DecoderStep(nn.Module):
    """upsample the coarse input by 2, concatenate the projected skip, then 1x1, depthwise 3x3, 1x1."""

    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.skip = conv_bn(skip, cout, 1)
        self.mix = conv_bn(cin + cout, cout, 1)
        self.dw = conv_bn(cout, cout, 3, groups=cout)
        self.pw = conv_bn(cout, cout, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.pw(self.dw(self.mix(torch.cat([x, self.skip(skip)], 1))))


class BokehNet(nn.Module):
    def __init__(self, arch: Arch = ARCH):
        super().__init__()
        self.arch = arch
        self.stem = conv_bn(3, arch.stem, 3, 2)
        stages, cin, self.skip_channels = [], arch.stem, [arch.stem]
        for i, (expand, cout, blocks, stride) in enumerate(arch.stages):
            seq = []
            for b in range(blocks):
                seq.append(InvertedResidual(cin, cout, stride if b == 0 else 1, expand))
                cin = cout
            stages.append(nn.Sequential(*seq))
            if i in arch.taps:
                self.skip_channels.append(cout)
        self.stages = nn.ModuleList(stages)
        self.context = conv_bn(cin, cin, 1)
        steps, prev = [], cin
        for width, skip in zip(arch.decoder, reversed(self.skip_channels[:-1])):
            steps.append(DecoderStep(prev, skip, width))
            prev = width
        self.steps = nn.ModuleList(steps)
        self.head = nn.Conv2d(prev, arch.out_channels, 1)
        # Start from "about 1.5 px, about 70 % relative error": mu = log 1.5, s = log 0.7.
        nn.init.normal_(self.head.weight, std=1e-3)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.405, -0.357]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.head:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():
            if isinstance(m, InvertedResidual) and m.residual:
                nn.init.zeros_(m.body[-1][1].weight)  # a residual block starts as the identity

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """`e` is the encoded plane (`encode.encode_t`), (N, 3, H, W) with H and W divisible by 32."""
        if e.shape[-1] % 32 or e.shape[-2] % 32:
            raise ValueError(f"the input's size must be divisible by 32, got {tuple(e.shape[-2:])}")
        x = self.stem((e - INPUT_MEAN) / INPUT_STD)
        skips = [x]
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.arch.taps:
                skips.append(x)
        x = x + self.context(x.mean((2, 3), keepdim=True))
        for step, skip in zip(self.steps, reversed(skips[:-1])):
            x = step(x, skip)
        return self.head(x)


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macs_per_pixel(model: BokehNet, size: tuple[int, int] = (256, 256)) -> float:
    """Multiply-accumulates of one forward pass per *input* pixel (convolutions and the head; the mean, the adds and the
    bilinear steps are not counted: they are a few percent of it)."""
    total = [0]

    def hook(mod: nn.Conv2d, _inp, out):
        total[0] += out.shape[2] * out.shape[3] * mod.out_channels * (mod.in_channels // mod.groups) * (
            mod.kernel_size[0] * mod.kernel_size[1])

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, *size))
    model.train(was)
    for h in handles:
        h.remove()
    return total[0] / (size[0] * size[1])
