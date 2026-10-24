"""Synthetic defocus with a known blur, rendered on the training device from licence-clean sharp photos.

Lens character (`docs/21`) reshapes the blur a photo already has, so the network learns the blur radius `r` of every
pixel (a disc radius in pixels of the 1024-px-long frame; `b = r / half-diagonal`) and how sure it is. This module makes
the training pairs. It follows SYNDOF's layered thin-lens recipe (Lee et al., CVPR 2019, section 3), adapted to disc
kernels, linear light and bright lights (`research/bokeh-map-2026-10-24.md` section 5.2), and runs on the GPU in
PyTorch (FFT convolution per layer) because a pre-rendered set would be hundreds of gigabytes.

One batch, step by step (`synthesize`):

1. **Sources.** `TileBatch`: sharp 8-bit tiles of Open Images photos (the sharpness filter ran when the tiles were
   built, `sources.py`), each with an instance-id map (Open Images' CC BY 4.0 masks) and the ids of the instances that
   may be cut out. Decoded from sRGB to linear light. A random crop and a horizontal flip per tile.
2. **Highlights.** In 60 % of the tiles the pixels near clipping (max channel from 0.55 to 0.97, a wide soft knee) are
   raised 1.4 to 16 times, so a JPEG's clipped lights are far above clip, as real lights are.
3. **Layers.** The first tile of a sample is the background layer (reflect-padded by the kernel margin). 1 to 4 sprites
   follow: an instance cut out of *another* tile of the batch with its mask (so no hole is left behind it), scaled
   0.55 to 1.15 times, turned by up to 0.25 rad, flipped, set down at a random place, its mask eroded by a pixel in
   70 % of the cases and feathered. 0 to 2 **emitter groups** add small bright lights (3 to 40 dots or tiny patches,
   warm, neutral, cool and a few saturated colours) whose flux is chosen so that their out-of-focus ball peaks at
   0.7 to 45 times the scene's median, with alpha 0: they add light and are hidden by whatever lies in front.
4. **Radii.** Each layer gets a signed defocus `t` (positive behind the focus plane, negative in front; the disc radius
   is `|t|` pixels, 0 for the layer in focus). Four modes per sample: all in focus (30 %: a plain tile or a collage), a
   sharp subject with a blurred background and layers on either side (35 %), a sharp background with blurred
   foreground (10 %), and random radii (25 %). Radii are log-uniform from 1 to `r_max_px` (up to `b_cap` and beyond,
   for coverage). In a quarter of the batches the background is a tilted plane: its radius ramps across the frame,
   rendered as four radius bins blended by a partition of unity. Layers are drawn back to front in order of `t`.
5. **Kernels.** From the library (`kernels.py`), one lens per picture in 70 % of the samples (every layer and every bin of the
   ramp shares the kernel's family, as one iris would), one draw per layer otherwise: disc, soft rim, bright and dark rim, onion rings,
   5 to 9 blades, cat's eye, Gaussian. The kernel's label radius is the disc radius with the same second moment.
6. **Render.** Every layer's premultiplied colour and alpha are convolved with its kernel by FFT (on a canvas with the
   kernel's size as margin, cropped after) and composited back to front: `out = out (1 - a~) + c~` with the *blurred*
   alpha `a~` (SYNDOF; a sharp subject keeps its crisp edge, the blurred layer in front lets the layer behind show
   through). 25 % of the samples are "night": the scene is darkened 5 to 32 times against its lights.
7. **Label.** A layer's *share* of a pixel is its blurred alpha times the visibility left by the layers in front
   (the shares sum to 1). An emitter's share is its part of the pixel's light. The label of a pixel is the radius of
   the layer with the largest share, and its weight is lowered where no layer dominates (the band around a blurred
   edge, `0.25 + 0.75 smoothstep(0.5, 0.9, share)`), so the loss does not punish a network for the ambiguity of an
   edge. In SYNDOF's words the label is the blur at the layer's own position, so depth edges are not smeared.
8. **Sensor.** A fixed mild blur in 40 % of the samples (the lens plane is a binned raw, not a razor-sharp JPEG), the
   exposure that puts the median at `mu` of the clip level (0.03 to 0.3, night 0.004 to 0.05), a colour cast of up to
   0.4 stops on red and blue, Poisson-Gaussian noise (shot term 5e-6 to 1.3e-3 of the clip level, read noise 3 to 12
   times that) and the clip at 1, then white-balance-like gains on red and blue in half of the samples (so clip
   levels differ per channel, as on a balanced raw plane). The result is the lens-plane stand-in; `encode.py` turns it
   into the network's input.

The label is the *added* blur. A source's own blur is the main label noise (a Flickr photo with natural bokeh gets
label 0 there); the tile filter and the Laplace loss limit it (README). Everything is a function of the batch, the
config and the generator, so a seed repeats it exactly on one device.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from training.bokeh import structures
from training.bokeh.encode import R_FLOOR_PX, linear_to_srgb, lower_median_t, luma_t, srgb_to_linear
from training.bokeh.kernels import KERNEL_R, conv_fft, make_kernels, sample_params

#: Layer classes (`render_layers(layer_class=...)`): what a layer is, for its label weight.
NORMAL, FLAT, THIN = 0, 1, 2
#: Pixels farther than this from the edge of every layer are "far from any evidence" inside a featureless layer.
OBS_REACH = 8
#: The weight of a pixel far from any edge inside a featureless layer, and the floor of a thin structure's own pixels.
FLAT_INTERIOR_WEIGHT = 0.1
THIN_WEIGHT_FLOOR = 0.9


@dataclass
class SynthConfig:
    """Sizes and ranges. The defaults are the real training setup; the tests shrink the sizes."""

    #: Side of the (square) sample cut from a tile and rendered.
    sample: int = 512
    #: Margin of the canvas = half size of the kernels.
    kernel_r: int = KERNEL_R
    #: The largest disc radius drawn, pixels (b = 0.05 on a 3:2 frame of 1024 px is 31 px).
    r_max_px: float = 34.0
    r_min_px: float = 1.0
    #: Surface layers per batch (the background and 1 to 4 sprites).
    layers: tuple[int, ...] = (2, 3, 4, 5)
    mode_probs: tuple[float, ...] = (0.30, 0.35, 0.10, 0.25)
    p_ramp_batch: float = 0.25
    p_night: float = 0.20
    emitter_groups: tuple[float, ...] = (0.35, 0.40, 0.25)  # probabilities of 0, 1, 2 groups
    max_lamps: int = 40
    #: Samples rendered per FFT call: bounds the memory of the spectra.
    fft_chunk: int = 8
    #: The ramp's number of radius bins.
    bins: int = 4
    # --- v3. The defaults below are the bokeh-2 recipe, bit for bit (nothing extra is drawn from the generator); `v3()` turns
    # --- on the thin structures, the featureless layers, the depth stack mode and the layer count that goes with them.
    #: Probabilities of 0, 1, 2 thin-structure layers in a batch (`structures.thin_layer`); one entry = none.
    thin_layers: tuple[float, ...] = (1.0,)
    #: Share of the samples that have a thin structure in a layer that does, and the chance that a layer's structures are
    #: blurred (a defocused wire in front) instead of in focus.
    p_thin_sample: float = 0.55
    p_thin_blurred: float = 0.2
    #: Probabilities of 0, 1 featureless layers (`structures.flat_layer`) in a batch; one entry = none.
    flat_layers: tuple[float, ...] = (1.0,)
    p_flat_sample: float = 0.6
    #: Chance that a featureless layer is in focus, and that a sample's whole background is featureless.
    p_flat_sharp: float = 0.55
    p_flat_background: float = 0.0
    #: Probabilities of the layer counts in `layers` (None: all alike).
    layers_probs: tuple[float, ...] | None = None
    #: Skip the FFT of a layer whose radius is under half a pixel in every sample of the batch (an exact identity).
    skip_identity: bool = False

    @classmethod
    def v3(cls, **kw) -> "SynthConfig":
        """The v3 recipe (module docstring, "v3"): the bokeh-2 settings plus thin structures, featureless layers, a fifth
        mode (a stack of surfaces whose blur is spread evenly between 1.5 px and the background's) and up to 6 surfaces."""
        base = dict(layers=(2, 3, 4, 5, 6), layers_probs=(0.10, 0.25, 0.30, 0.20, 0.15), mode_probs=(0.22, 0.25, 0.08, 0.15, 0.30),
                    thin_layers=(0.40, 0.45, 0.15), flat_layers=(0.60, 0.40), p_flat_background=0.07, skip_identity=True)
        return cls(**{**base, **kw})

    def __post_init__(self):
        # Radii are drawn log-uniform: a lower bound of 0 is log(0) = -inf, which turns into NaN in every sample.
        if not 0 < self.r_min_px <= self.r_max_px:
            raise ValueError(f"need 0 < r_min_px <= r_max_px, got r_min_px={self.r_min_px}, r_max_px={self.r_max_px}")

    @property
    def canvas(self) -> int:
        return self.sample + 2 * self.kernel_r


@dataclass
class TileBatch:
    """What the loader gives: `rgb` (B, T, T, 3) uint8 sRGB tiles, `ids` (B, T, T) uint8 instance ids (0 = none),
    `cands` (B, C) int the ids that may be cut out (0 pads), `hd` (B,) the half-diagonal in pixels of the frame each
    tile was cut from."""

    rgb: torch.Tensor
    ids: torch.Tensor
    cands: torch.Tensor
    hd: torch.Tensor

    def to(self, device) -> "TileBatch":
        return TileBatch(self.rgb.to(device), self.ids.to(device), self.cands.to(device), self.hd.to(device))


@dataclass
class Synth:
    """A rendered batch. `lin` (B, 3, S, S) is the lens-plane stand-in (linear, clipped at 1 before the white-balance
    gains); `r` (B, 1, S, S) the disc radius in pixels (0 = in focus); `w` the loss weights in [0.25, 1]; `hd` (B,)."""

    lin: torch.Tensor
    r: torch.Tensor
    w: torch.Tensor
    hd: torch.Tensor
    info: dict = field(default_factory=dict)
    #: (B, 1, S, S) uint8, v3 only: the class of the layer that dominates each pixel (`NORMAL`, `FLAT`, `THIN`).
    cls: torch.Tensor | None = None

    @property
    def logr(self) -> torch.Tensor:
        """The training target: `log(max(r, R_FLOOR_PX))`."""
        return torch.log(self.r.clamp(min=R_FLOOR_PX))


# --- random helpers: every draw comes from `gen`, in a fixed order ------------------------------------------------


def _u(gen, lo: float, hi: float, *shape, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(*shape, generator=gen, device=device)


def _lu(gen, lo, hi, *shape, device) -> torch.Tensor:
    """Log-uniform in [lo, hi]; `lo` and `hi` may be tensors."""
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device)
    u = torch.rand(*shape, generator=gen, device=device)
    return torch.exp(torch.log(lo_t) + (torch.log(hi_t) - torch.log(lo_t)) * u)


def _bern(gen, p: float, *shape, device) -> torch.Tensor:
    return torch.rand(*shape, generator=gen, device=device) < p


def _scalar(gen, lo: float, hi: float) -> float:
    """One uniform draw as a Python float (a device sync on a GPU generator; used a few times per batch)."""
    return float(_u(gen, lo, hi, 1, device=gen.device))


def _smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def gauss_kernel1d(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x * x) / (2 * max(sigma, 1e-3) ** 2))
    return k / k.sum()


def gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (N, C, H, W), replicate edges."""
    if sigma <= 0:
        return x
    k = gauss_kernel1d(sigma, x.device)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def blur_per_sample(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian blur of (B, 3, H, W) with one sigma per sample (0 = none), a 7 x 7 kernel."""
    b, c, h, w = x.shape
    ax = torch.arange(-3, 4, dtype=torch.float32, device=x.device)
    g = torch.exp(-(ax.view(1, -1) ** 2) / (2 * sigma.clamp(min=1e-3).view(-1, 1) ** 2))
    g = g / g.sum(1, keepdim=True)
    delta = torch.zeros_like(g)
    delta[:, 3] = 1.0
    g = torch.where((sigma > 0).view(-1, 1), g, delta)
    k2 = (g.view(b, 1, 7, 1) * g.view(b, 1, 1, 7)).repeat_interleave(c, 0)
    y = F.conv2d(F.pad(x.reshape(1, b * c, h, w), (3, 3, 3, 3), mode="replicate"), k2, groups=b * c)
    return y.reshape(b, c, h, w)


# --- the pieces -------------------------------------------------------------------------------------------------


def _sprite(lin, ids, cands, canvas_m, gen, cfg: SynthConfig, active: torch.Tensor):
    """One sprite layer for every sample: premultiplied colour (B, 3, C, C) and alpha (B, 1, C, C). The instance is
    cut from another tile of the batch (`lin`, `ids`, `cands`), moved, scaled and turned with `affine_grid`."""
    dev = lin.device
    b, _, s, _ = lin.shape
    m, cv = canvas_m, cfg.canvas
    has = (cands > 0).any(1).float()
    wmat = (has + 1e-3).expand(b, b).clone()
    wmat[torch.arange(b), torch.arange(b)] *= 0.02
    donor = torch.multinomial(wmat, 1, generator=gen).squeeze(1)
    ncand = (cands[donor] > 0).sum(1).clamp(min=1)
    pick = torch.floor(torch.rand(b, generator=gen, device=dev) * ncand).long().clamp(max=cands.shape[1] - 1)
    cid = cands[donor, pick]
    alpha_src = (ids[donor] == cid.view(b, 1, 1).to(ids.dtype)).float().unsqueeze(1) * (cid > 0).view(b, 1, 1, 1)
    rgb_src = lin[donor]
    alpha_cv = F.pad(alpha_src, (m, m, m, m))
    rgb_cv = F.pad(rgb_src, (m, m, m, m), mode="replicate")

    lin1 = (torch.arange(cv, dtype=torch.float32, device=dev) + 0.5) / cv * 2 - 1
    gy, gx = torch.meshgrid(lin1, lin1, indexing="ij")
    mass = alpha_cv.sum((1, 2, 3)).clamp(min=1e-6)
    gcx = (alpha_cv[:, 0] * gx).sum((1, 2)) / mass
    gcy = (alpha_cv[:, 0] * gy).sum((1, 2)) / mass
    scale = _lu(gen, 0.55, 1.15, b, device=dev)
    rot = _u(gen, -0.25, 0.25, b, device=dev)
    flip = torch.where(_bern(gen, 0.5, b, device=dev), -1.0, 1.0)
    tx, ty = _u(gen, -0.55, 0.55, b, device=dev), _u(gen, -0.55, 0.55, b, device=dev)
    cos, sin = torch.cos(rot) / scale, torch.sin(rot) / scale
    a = torch.stack([torch.stack([flip * cos, sin], 1), torch.stack([-flip * sin, cos], 1)], 1)  # (B, 2, 2)
    off = torch.stack([gcx, gcy], 1) - torch.einsum("bij,bj->bi", a, torch.stack([tx, ty], 1))
    theta = torch.cat([a, off.unsqueeze(2)], 2)
    grid = F.affine_grid(theta, (b, 1, cv, cv), align_corners=False)
    alpha = F.grid_sample(alpha_cv, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    rgb = F.grid_sample(rgb_cv, grid, mode="bilinear", padding_mode="border", align_corners=False)
    erode = _bern(gen, 0.7, b, device=dev).view(b, 1, 1, 1)
    alpha = torch.where(erode, -F.max_pool2d(-alpha, 3, 1, 1), alpha)
    alpha = gauss_blur(alpha, _scalar(gen, 0.5, 0.9)).clamp(0, 1)
    ok = (alpha.sum((1, 2, 3)) >= 0.002 * s * s) & active
    alpha = alpha * ok.view(b, 1, 1, 1)
    return rgb * alpha, alpha


def _emitters(median_day: torch.Tensor, radius: torch.Tensor, gen, cfg: SynthConfig, active: torch.Tensor):
    """One emitter group for every sample: light (B, 3, C, C), no alpha. `radius` (B,) is the group's disc radius in
    pixels, so that the flux gives a ball of peak `median_day * 2^U(-0.5, 5.5)` (flux = max(pi r^2, 2) * peak)."""
    dev = median_day.device
    b, cv, n = median_day.shape[0], cfg.canvas, cfg.max_lamps
    count = torch.randint(3, n + 1, (b,), generator=gen, device=dev)
    on = (torch.arange(n, device=dev).view(1, n) < count.view(b, 1)) & active.view(b, 1)
    pos = torch.rand(b, n, 2, generator=gen, device=dev) * (cv - 1)
    peak = median_day.view(b, 1) * torch.exp2(_u(gen, -0.5, 5.5, b, n, device=dev))
    flux = torch.clamp(math.pi * radius.view(b, 1) ** 2, min=2.0) * peak * on
    palette = torch.tensor([[1.0, 0.72, 0.42], [1.0, 0.85, 0.65], [1.0, 1.0, 1.0], [0.75, 0.85, 1.0],
                            [1.0, 0.25, 0.2], [0.3, 1.0, 0.4], [0.3, 0.45, 1.0]], device=dev)
    pw = torch.tensor([0.28, 0.22, 0.22, 0.13, 0.05, 0.05, 0.05], device=dev)
    tint = palette[torch.multinomial(pw.repeat(b * n, 1), 1, generator=gen).view(b, n)]  # (B, n, 3)
    x0, y0 = pos[..., 0].floor(), pos[..., 1].floor()
    fx, fy = pos[..., 0] - x0, pos[..., 1] - y0
    light = torch.zeros(b, 3, cv * cv, device=dev)
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = (x0 + dx).clamp(0, cv - 1), (y0 + dy).clamp(0, cv - 1)
        idx = (yi * cv + xi).long().unsqueeze(1).expand(b, 3, n)
        light.scatter_add_(2, idx, (flux * wgt).unsqueeze(1) * tint.permute(0, 2, 1))
    light = light.view(b, 3, cv, cv)
    # Dots and tiny patches: a Gaussian of 0.5 to 1.2 px, or a 2 to 4 px box.
    if _scalar(gen, 0.0, 1.0) < 0.7:
        light = gauss_blur(light, _scalar(gen, 0.5, 1.2))
    else:
        k = int(_scalar(gen, 2.0, 4.999))
        light = F.avg_pool2d(F.pad(light, (k // 2, k - 1 - k // 2) * 2), k, 1)
    return light


def _signed_radii(mode, t_bg, lf, ng, night, gen, cfg: SynthConfig, dev):
    """The signed defocus of the sprites (B, lf) and the emitter groups (B, ng). See the module docstring, step 4."""
    b = mode.shape[0]
    lo, hi = cfg.r_min_px, cfg.r_max_px
    mag = _lu(gen, 1.2, hi, b, lf + ng, device=dev)
    mag3 = _lu(gen, lo, hi, b, lf + ng, device=dev)
    behind = _bern(gen, 0.5, b, lf + ng, device=dev)
    frac = _u(gen, 0.15, 1.0, b, lf + ng, device=dev)
    zero3 = _bern(gen, 0.25, b, lf + ng, device=dev)
    sign3 = torch.where(_bern(gen, 0.5, b, lf + ng, device=dev), 1.0, -1.0)
    front = _bern(gen, 0.8, b, lf + ng, device=dev)
    t = torch.zeros(b, lf + ng, device=dev)
    m = mode.view(b, 1)
    # mode 1: the first sprite is the subject (in focus); others behind it (between it and the background) or in front
    t1 = torch.where(behind, t_bg.view(b, 1) * frac, -mag)
    if lf:
        t1[:, 0] = 0.0
    t = torch.where(m == 1, t1, t)
    t = torch.where(m == 2, torch.where(front, -mag, torch.zeros_like(mag)), t)
    t = torch.where(m == 3, torch.where(zero3, torch.zeros_like(mag3), sign3 * mag3), t)
    if len(cfg.mode_probs) > 4 and lf:
        # mode 4 (v3), a stack of surfaces: the sprites' radii are spread evenly (one in each of `lf` slices of the log
        # radius) between 1.5 px and the background's, so every depth in between is drawn; a few are in front; the first
        # is the subject, in focus, half of the time.
        strata = (torch.argsort(torch.rand(b, lf, generator=gen, device=dev), 1).float() + torch.rand(b, lf, generator=gen, device=dev)) / lf
        lo_l, hi_l = math.log(1.5), torch.log(t_bg.clamp(min=1.6)).view(b, 1)
        r4 = torch.exp(lo_l + (hi_l - lo_l) * strata)
        t4 = torch.where(_bern(gen, 0.15, b, lf, device=dev), -r4, r4)
        t4[:, 0] = torch.where(_bern(gen, 0.5, b, device=dev), torch.zeros(b, device=dev), t4[:, 0])
        t = torch.cat([torch.where(m == 4, t4, t[:, :lf]), t[:, lf:]], 1)
    # emitters: lights are mostly far behind (night: balls you can see)
    if ng:
        e = t[:, lf:]
        bg = t_bg.view(b, 1)
        e1 = torch.where(front[:, lf:], torch.minimum(bg * _u(gen, 0.7, 1.3, b, ng, device=dev), bg), -mag[:, lf:])
        e = torch.where((m == 1) | (m == 4), e1, e)
        night_r = torch.where(_bern(gen, 0.8, b, ng, device=dev), _lu(gen, 3.0, hi, b, ng, device=dev),
                              -_lu(gen, 3.0, hi, b, ng, device=dev))
        e = torch.where(night.view(b, 1), night_r, e)
        t = torch.cat([t[:, :lf], e], 1)
    return torch.minimum(t, t_bg.view(b, 1))


def _reach(x: torch.Tensor, r: int) -> torch.Tensor:
    """Max filter of (B, 1, H, W) over a (2r+1) x (2r+1) square (two one-dimensional passes)."""
    return F.max_pool2d(F.max_pool2d(x, (2 * r + 1, 1), 1, (r, 0)), (1, 2 * r + 1), 1, (0, r))


def render_layers(bg: torch.Tensor, bin_w: torch.Tensor, bg_r: torch.Tensor, field_bg: torch.Tensor,
                  sprites: list, t_sprites: torch.Tensor, emitters: list, t_emitters: torch.Tensor,
                  gen: torch.Generator, margin: int, fft_chunk: int = 8,
                  kind_weights=None, layer_class: torch.Tensor | None = None, skip_identity: bool = False,
                  return_class: bool = False):
    """The deterministic core: blurs every layer with its own kernel and composites back to front, and labels.

    `bg` (B, 3, C, C) the background colour and `bin_w` (B, nb, C, C) its partition of unity over `nb` radius bins
    (one bin with weight 1 when the radius is constant) of radii `bg_r` (B, nb); `field_bg` (B, C, C) is the
    background's radius at every pixel (the label). `sprites` is a list of (premultiplied colour, alpha) and
    `emitters` a list of light images; `t_sprites` (B, lf) and `t_emitters` (B, ng) are their signed defocus
    (`|t|` pixels of disc radius, larger = further back). Returns the picture (B, 3, C, C), the disc radius in pixels
    of every pixel (B, 1, C, C) and the loss weight. `kind_weights` picks the kernel kinds (default: the library's).

    v3: `layer_class` (B, 1 + len(sprites) + len(emitters)) holds `NORMAL`, `FLAT` or `THIN` for the background, the
    sprites and the emitters, in that order. Where a featureless layer dominates a pixel that is farther than `OBS_REACH`
    px from the edge of every layer (its blurred alpha strictly between 0.02 and 0.98), the radius cannot be read from the
    picture and the weight is scaled by `FLAT_INTERIOR_WEIGHT` (rising to 1 within reach of an edge); where a thin structure
    dominates (share at least a half) the weight is at least `THIN_WEIGHT_FLOOR`: a one-pixel line covers a pixel by less
    than 0.9, and its own label is the one that must be learnt. `skip_identity` leaves out the FFT of every layer whose
    radius is under half a pixel in all samples (the kernel is then an exact delta). `return_class` adds the class of the
    dominating layer (B, 1, C, C) as a fourth value."""
    dev = bg.device
    b, _, cv, _ = bg.shape
    nb, lf, ng = bin_w.shape[1], len(sprites), len(emitters)
    n_layers = 1 + lf + ng
    t_bg = bg_r[:, 0]
    t_all = torch.cat([t_bg.view(b, 1), t_sprites.view(b, lf), t_emitters.view(b, ng)], 1)  # (B, n_layers)
    is_em = torch.zeros(b, n_layers, dtype=torch.bool, device=dev)
    is_em[:, 1 + lf:] = True
    radii = t_all.abs()
    parts, part_r = [], []
    for jj in range(nb):
        parts.append(torch.cat([bg * bin_w[:, jj:jj + 1], bin_w[:, jj:jj + 1]], 1))
        part_r.append(bg_r[:, jj])
    for k, (c_k, a_k) in enumerate(sprites):
        parts.append(torch.cat([c_k, a_k], 1))
        part_r.append(radii[:, 1 + k])
    for g, light in enumerate(emitters):
        parts.append(torch.cat([light, torch.zeros(b, 1, cv, cv, device=dev)], 1))
        part_r.append(radii[:, 1 + lf + g])
    stack = torch.stack(parts, 1)  # (B, L_x, 4, C, C)
    rad_x = torch.stack(part_r, 1).clamp(max=float(margin - 3))
    lx = stack.shape[1]
    params = sample_params(b * lx, gen, dev) if kind_weights is None else sample_params(b * lx, gen, dev, kind_weights)
    # One lens per picture: the bins of the background always share a kernel, and in 70 % of the samples so do all the layers
    # (the same iris, as in a real photo); in the others every layer draws its own, for diversity.
    same = _bern(gen, 0.7, b, 1, device=dev)
    src = torch.where(same, torch.zeros(b, lx, dtype=torch.long, device=dev), torch.arange(lx, device=dev).repeat(b, 1))
    src[:, :nb] = 0
    params = {k: v.view(b, lx).gather(1, src).reshape(-1) for k, v in params.items()}
    kern = make_kernels(rad_x.reshape(-1), params, margin).view(b, lx, 1, 2 * margin + 1, 2 * margin + 1)
    if skip_identity:
        todo = (rad_x >= 0.5).any(0).nonzero().flatten().tolist()
        blurred = stack.clone()
        if todo:
            blurred[:, todo] = torch.cat([conv_fft(stack[i:i + fft_chunk, todo], kern[i:i + fft_chunk, todo]) for i in range(0, b, fft_chunk)], 0)
    else:
        blurred = torch.cat([conv_fft(stack[i:i + fft_chunk], kern[i:i + fft_chunk]) for i in range(0, b, fft_chunk)], 0)
    bg_all = blurred[:, :nb].sum(1)
    layers = torch.cat([bg_all.unsqueeze(1), blurred[:, nb:]], 1)  # (B, n_layers, 4, C, C)
    col, alp = layers[:, :, :3], layers[:, :, 3:].clamp(0, 1)

    # draw back to front: the larger the signed radius, the further back; the background first
    key = t_all.clone()
    key[:, 0] = float("inf")
    order = torch.sort(key, dim=1, descending=True, stable=True).indices
    ar = torch.arange(b, device=dev)
    out = torch.zeros(b, 3, cv, cv, device=dev)
    for p in range(n_layers):
        sel = order[:, p]
        out = out * (1 - alp[ar, sel]) + col[ar, sel]

    # labels: shares, front to back
    y_out = luma_t(out).clamp(min=1e-6)
    vis = torch.ones(b, 1, cv, cv, device=dev)
    shares, rad_maps, em_flags = [], [], []
    for p in range(n_layers - 1, -1, -1):
        sel = order[:, p]
        a_p, c_p = alp[ar, sel], col[ar, sel]
        em = is_em[ar, sel].view(b, 1, 1, 1)
        share = torch.where(em, (luma_t(c_p) * vis / y_out).clamp(0, 1), a_p * vis)
        rmap = torch.where(sel.view(b, 1, 1) == 0, field_bg, radii[ar, sel].view(b, 1, 1).expand(b, cv, cv))
        shares.append(share)
        rad_maps.append(rmap.unsqueeze(1))
        em_flags.append(em.expand(b, 1, cv, cv))
        vis = vis * (1 - a_p)
    share = torch.cat(shares, 1)  # (B, n_layers, C, C), front first
    rmap = torch.cat(rad_maps, 1)
    emf = torch.cat(em_flags, 1)
    em_total = (share * emf).sum(1, keepdim=True).clamp(max=1.0)
    share = torch.where(emf, share, share * (1 - em_total))
    best, which = share.max(1, keepdim=True)
    label = rmap.gather(1, which)
    weight = 0.25 + 0.75 * _smoothstep(0.5, 0.9, best)
    if layer_class is None:
        return (out, label, weight, torch.zeros_like(label, dtype=torch.uint8)) if return_class else (out, label, weight)
    cls_front = torch.stack([layer_class[ar, order[:, p]] for p in range(n_layers - 1, -1, -1)], 1)  # (B, n_layers), as `share`
    dom = cls_front.view(b, n_layers, 1, 1).expand(b, n_layers, cv, cv).gather(1, which)
    edge_zone = ((alp[:, 1:] > 0.02) & (alp[:, 1:] < 0.98)).any(1).float() if n_layers > 1 else torch.zeros_like(best)
    seen = _reach(edge_zone, OBS_REACH)
    weight = torch.where(dom == FLAT, weight * (FLAT_INTERIOR_WEIGHT + (1 - FLAT_INTERIOR_WEIGHT) * seen), weight)
    weight = torch.where((dom == THIN) & (best >= 0.5), weight.clamp(min=THIN_WEIGHT_FLOOR), weight)
    return (out, label, weight, dom.to(torch.uint8)) if return_class else (out, label, weight)


def synthesize(tb: TileBatch, cfg: SynthConfig, gen: torch.Generator) -> Synth:
    """Renders one batch (the module docstring). `tb` is on the device of `gen`."""
    dev = tb.rgb.device
    b, tile = tb.rgb.shape[0], tb.rgb.shape[1]
    s, m, cv = cfg.sample, cfg.kernel_r, cfg.canvas
    assert tile >= s and tb.rgb.shape[2] >= s, "tiles are smaller than the sample"

    # 1. sources: random crop and flip, linear light
    off = torch.randint(0, tile - s + 1, (b, 2), generator=gen, device=dev).tolist()
    flip = _bern(gen, 0.5, b, device=dev)
    rgb = torch.stack([tb.rgb[i, off[i][0]:off[i][0] + s, off[i][1]:off[i][1] + s] for i in range(b)])
    ids = torch.stack([tb.ids[i, off[i][0]:off[i][0] + s, off[i][1]:off[i][1] + s] for i in range(b)])
    rgb = torch.where(flip.view(b, 1, 1, 1), rgb.flip(2), rgb)
    ids = torch.where(flip.view(b, 1, 1), ids.flip(2), ids)
    lin = srgb_to_linear(rgb.permute(0, 3, 1, 2).float() / 255)

    # 2. highlights
    thr = _u(gen, 0.85, 0.97, b, device=dev).view(b, 1, 1, 1)
    gain = torch.exp2(_u(gen, 0.5, 4.0, b, device=dev)).view(b, 1, 1, 1)
    gain = torch.where(_bern(gen, 0.6, b, device=dev).view(b, 1, 1, 1), gain, torch.ones_like(gain))
    lin = lin * (1 + _smoothstep(thr - 0.3, thr, lin.amax(1, keepdim=True)) * (gain - 1))
    median_day = lower_median_t(luma_t(lin))

    # 3-4. modes, layer counts, radii
    mode = torch.multinomial(torch.tensor(cfg.mode_probs, dtype=torch.float32, device=dev).repeat(b, 1), 1, generator=gen).squeeze(1)
    if cfg.layers_probs is None:
        lf = int(cfg.layers[int(torch.randint(0, len(cfg.layers), (1,), generator=gen, device=dev))]) - 1
    else:
        lf = int(cfg.layers[int(torch.multinomial(torch.tensor(cfg.layers_probs, dtype=torch.float32, device=dev), 1, generator=gen))]) - 1
    ng = int(torch.multinomial(torch.tensor(cfg.emitter_groups, dtype=torch.float32, device=dev), 1, generator=gen))
    n_flat = int(torch.multinomial(torch.tensor(cfg.flat_layers, dtype=torch.float32, device=dev), 1, generator=gen)) if len(cfg.flat_layers) > 1 else 0
    n_thin = int(torch.multinomial(torch.tensor(cfg.thin_layers, dtype=torch.float32, device=dev), 1, generator=gen)) if len(cfg.thin_layers) > 1 else 0
    ramp_batch = bool(_bern(gen, cfg.p_ramp_batch, 1, device=dev))
    night = _bern(gen, cfg.p_night, b, device=dev) & (ng > 0)
    lo, hi = cfg.r_min_px, cfg.r_max_px
    t_bg = torch.zeros(b, device=dev)
    t_bg = torch.where(mode == 1, torch.where(_bern(gen, 0.92, b, device=dev), _lu(gen, 1.2, hi, b, device=dev), t_bg), t_bg)
    t_bg = torch.where(mode == 3, torch.where(_bern(gen, 0.8, b, device=dev), _lu(gen, lo, hi, b, device=dev), t_bg), t_bg)
    if len(cfg.mode_probs) > 4:
        t_bg = torch.where(mode == 4, torch.where(_bern(gen, 0.95, b, device=dev), _lu(gen, min(4.0, hi), hi, b, device=dev), t_bg), t_bg)
    t_other = _signed_radii(mode, t_bg, lf, ng, night, gen, cfg, dev)
    plain = (mode == 0) & _bern(gen, 0.5, b, device=dev)  # a plain, unmodified tile: no sprites
    t_flat = torch.zeros(b, n_flat, device=dev)
    t_thin = torch.zeros(b, n_thin, device=dev)
    if n_flat:
        sharp = _bern(gen, cfg.p_flat_sharp, b, n_flat, device=dev)
        t_flat = torch.where(sharp, t_flat, _lu(gen, 1.2, hi, b, n_flat, device=dev) * torch.where(_bern(gen, 0.85, b, n_flat, device=dev), 1.0, -1.0))
        t_flat = torch.minimum(t_flat, t_bg.view(b, 1))
    for j in range(n_thin):
        if bool(_bern(gen, cfg.p_thin_blurred, 1, device=dev)):  # this layer's structures are out of focus (mostly in front)
            mag = _lu(gen, 1.2, hi, b, device=dev) * torch.where(_bern(gen, 0.7, b, device=dev), -1.0, 1.0)
            t_thin[:, j] = torch.minimum(mag, t_bg)
    flat_bg = (_bern(gen, cfg.p_flat_background, b, device=dev) & ~plain) if cfg.p_flat_background > 0 else torch.zeros(b, dtype=torch.bool, device=dev)

    # ramp: the background's radius varies across the frame
    nb = cfg.bins if ramp_batch else 1
    use_ramp = (t_bg > 0) & ramp_batch
    ax = (torch.arange(cv, dtype=torch.float32, device=dev) + 0.5) / cv * 2 - 1
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    phi = torch.where(_bern(gen, 0.7, b, device=dev), math.pi / 2 + 0.5 * torch.randn(b, generator=gen, device=dev),
                      _u(gen, 0.0, 2 * math.pi, b, device=dev))
    xi = (torch.cos(phi).view(b, 1, 1) * gx + torch.sin(phi).view(b, 1, 1) * gy) / (
        torch.abs(torch.cos(phi)) + torch.abs(torch.sin(phi))).view(b, 1, 1)  # in [-1, 1]
    r_lo = torch.where(_bern(gen, 0.5, b, device=dev), torch.zeros(b, device=dev), _lu(gen, 1.0, hi * 0.6, b, device=dev))
    r_hi = torch.maximum(_lu(gen, 1.5, hi, b, device=dev), r_lo + 1.0).clamp(max=hi)
    r_lo = torch.where(use_ramp, r_lo, t_bg)
    r_hi = torch.where(use_ramp, r_hi, t_bg)
    field_bg = r_lo.view(b, 1, 1) + (r_hi - r_lo).view(b, 1, 1) * (xi + 1) / 2  # (B, C, C)

    # layer images: background (bins), then the featureless layers (behind sprites of equal radius), the sprites, the thin
    # structures (in front), the emitters
    canvas_bg = F.pad(lin, (m, m, m, m), mode="reflect")
    if cfg.p_flat_background > 0:
        canvas_bg = torch.where(flat_bg.view(b, 1, 1, 1), structures.flat_background(lin, gen, cv), canvas_bg)
    night_gain = torch.where(night, torch.exp2(-_u(gen, 2.5, 5.0, b, device=dev)), torch.ones(b, device=dev))
    ng_ = night_gain.view(b, 1, 1, 1)
    surf_c, surf_a, surf_t = [canvas_bg * ng_], [torch.ones(b, 1, cv, cv, device=dev)], [t_bg]
    classes = [torch.where(flat_bg, FLAT, NORMAL)]
    for k in range(n_flat):
        c_k, a_k = structures.flat_layer(lin, _bern(gen, cfg.p_flat_sample, b, device=dev) & ~plain, gen, cv, m)
        surf_c.append(c_k * ng_)
        surf_a.append(a_k)
        surf_t.append(t_flat[:, k])
        classes.append(torch.full((b,), FLAT, device=dev))
    for k in range(lf):
        c_k, a_k = _sprite(lin, ids, tb.cands, m, gen, cfg, ~plain)
        surf_c.append(c_k * ng_)
        surf_a.append(a_k)
        surf_t.append(t_other[:, k])
        classes.append(torch.full((b,), NORMAL, device=dev))
    thin_types = None
    for k in range(n_thin):
        thin_types = torch.multinomial(torch.tensor(structures.TYPE_PROBS, device=dev).repeat(b, 1), 1, generator=gen).squeeze(1)
        c_k, a_k = structures.thin_layer(lin, thin_types, _bern(gen, cfg.p_thin_sample, b, device=dev) & ~plain, gen, cv, m)
        surf_c.append(c_k * ng_)
        surf_a.append(a_k)
        surf_t.append(t_thin[:, k])
        classes.append(torch.full((b,), THIN, device=dev))
    em_c, em_t = [], []
    for g in range(ng):
        em_t.append(t_other[:, lf + g])
        em_c.append(_emitters(median_day, t_other[:, lf + g].abs(), gen, cfg, torch.ones(b, dtype=torch.bool, device=dev)))
        classes.append(torch.full((b,), NORMAL, device=dev))
    n_layers = 1 + n_flat + lf + n_thin + ng
    n_sprites = n_flat + lf + n_thin
    if nb > 1:
        j = torch.arange(nb, device=dev, dtype=torch.float32).view(1, nb) / (nb - 1)
        bg_r = r_lo.view(b, 1) + (r_hi - r_lo).view(b, 1) * j
        u = ((field_bg - r_lo.view(b, 1, 1)) / (r_hi - r_lo).clamp(min=1e-3).view(b, 1, 1)).clamp(0, 1) * (nb - 1)
        bin_w = torch.stack([(1 - (u - jj).abs()).clamp(0, 1) for jj in range(nb)], 1)  # (B, nb, C, C)
        first = torch.cat([torch.ones(b, 1, cv, cv, device=dev), torch.zeros(b, nb - 1, cv, cv, device=dev)], 1)
        bin_w = torch.where(use_ramp.view(b, 1, 1, 1), bin_w, first)
        bg_r = torch.where(use_ramp.view(b, 1), bg_r, t_bg.view(b, 1).expand(b, nb))
    else:
        bg_r = t_bg.view(b, 1)
        bin_w = torch.ones(b, 1, cv, cv, device=dev)
    t_all = torch.stack(surf_t + em_t, 1)  # (B, n_layers)

    # 5-7. kernels, FFT convolution, back-to-front composite, labels
    v3 = n_flat or n_thin or cfg.p_flat_background > 0
    got = render_layers(
        surf_c[0], bin_w, bg_r, field_bg, list(zip(surf_c[1:], surf_a[1:])), t_all[:, 1:1 + n_sprites], em_c, t_all[:, 1 + n_sprites:],
        gen, m, cfg.fft_chunk, layer_class=torch.stack(classes, 1) if v3 else None, skip_identity=cfg.skip_identity, return_class=True)
    crop = (slice(None), slice(None), slice(m, m + s), slice(m, m + s))
    out, label_r, weight, dom = (g[crop] for g in got)

    # 8. sensor
    blur_sigma = torch.where(_bern(gen, 0.4, b, device=dev), _u(gen, 0.3, 0.7, b, device=dev), torch.zeros(b, device=dev))
    out = blur_per_sample(out, blur_sigma)
    mu = torch.where(night, _lu(gen, 0.004, 0.05, b, device=dev), _lu(gen, 0.03, 0.3, b, device=dev))
    m_out = lower_median_t(luma_t(out)).clamp(min=1e-12)
    x = out * (mu / m_out).view(b, 1, 1, 1)
    cast = torch.exp2(_u(gen, -0.4, 0.4, b, 2, device=dev))
    x = x * torch.stack([cast[:, 0], torch.ones(b, device=dev), cast[:, 1]], 1).view(b, 3, 1, 1)
    shot = torch.pow(10.0, _u(gen, -5.3, -2.9, b, device=dev))
    read = shot * _u(gen, 3.0, 12.0, b, device=dev)
    sd = torch.sqrt(shot.view(b, 1, 1, 1) * x.clamp(min=0) + (read ** 2).view(b, 1, 1, 1))
    x = (x + torch.randn(x.shape, generator=gen, device=dev) * sd).clamp(0, 1)
    wb = _bern(gen, 0.5, b, device=dev)
    wb_gain = torch.stack([_u(gen, 1.0, 2.2, b, device=dev), torch.ones(b, device=dev), _u(gen, 1.0, 2.0, b, device=dev)], 1)
    x = x * torch.where(wb.view(b, 1), wb_gain, torch.ones_like(wb_gain)).view(b, 3, 1, 1)

    info = {"mode": mode, "night": night, "plain": plain, "layers": n_layers, "bins": nb, "emitter_groups": ng,
            "flat_layers": n_flat, "thin_layers": n_thin, "flat_background": flat_bg}
    return Synth(lin=x, r=label_r, w=weight, hd=tb.hd.to(dev), info=info, cls=dom)


# --- previews ---------------------------------------------------------------------------------------------------


def colorize(values: torch.Tensor, vmax: float) -> torch.Tensor:
    """(N, 1, H, W) -> (N, 3, H, W) in [0, 1]: a dark-blue to red ramp, 0 to `vmax`. A NaN counts as 0: it must never
    reach the integer index below (on a GPU that is a device-side assert, which is how the first Modal run died)."""
    values = torch.nan_to_num(values, nan=0.0, posinf=vmax, neginf=0.0)
    anchors = torch.tensor([[0.05, 0.05, 0.25], [0.1, 0.35, 0.85], [0.1, 0.8, 0.8], [0.4, 0.9, 0.3],
                            [0.95, 0.85, 0.1], [0.9, 0.2, 0.1]], device=values.device)
    pos = (values / vmax).clamp(0, 1) * (anchors.shape[0] - 1)
    lo = pos.floor().clamp(max=anchors.shape[0] - 2).long()
    f = (pos - lo).squeeze(1).unsqueeze(-1)
    col = anchors[lo.squeeze(1)] * (1 - f) + anchors[lo.squeeze(1) + 1] * f
    return col.permute(0, 3, 1, 2)


def preview_grid(sy: Synth, b_max: float = 0.05) -> torch.Tensor:
    """(H_total, W_total, 3) uint8: one row per sample, four panels (the input as the network sees it, a display version
    of the same plane, the label `b = r / hd` as colour, 0 to `b_max`, and the loss weight as grey, tinted cyan where a thin
    structure dominates and magenta where a featureless layer does)."""
    from training.bokeh.encode import MID, encode_t

    e = encode_t(sy.lin)
    y = luma_t(sy.lin)
    m = lower_median_t(y).clamp(min=1e-12).view(-1, 1, 1, 1)
    disp = linear_to_srgb(sy.lin * (MID / m) * 1.4)
    b = sy.r / sy.hd.view(-1, 1, 1, 1)
    lab = colorize(b, b_max) * (0.35 + 0.65 * sy.w)
    wmap = sy.w.expand(-1, 3, -1, -1).clone()
    if sy.cls is not None:
        thin, flat = (sy.cls == THIN).float(), (sy.cls == FLAT).float()
        wmap[:, 0:1] = wmap[:, 0:1] * (1 - thin) + 0.2 * thin + 0.9 * flat * sy.w
        wmap[:, 1:2] = wmap[:, 1:2] * (1 - flat) + 0.9 * thin * sy.w + 0.1 * flat
        wmap[:, 2:3] = wmap[:, 2:3] * (1 - flat) + 0.9 * thin * sy.w + 0.9 * flat * sy.w
    panels = torch.cat([e.clamp(0, 1), disp, lab, wmap.clamp(0, 1)], 3)  # side by side
    return (torch.cat(list(panels), 1).permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu()
