"""Procedural layers of the v3 synthesis: thin in-focus structures and featureless patches.

`bokeh-2` never saw a twig, a hair or a wire in focus in front of a blurred background (the sprites are Open Images
instances, mostly chunky objects), so such structures were absorbed into the blur around them; and it never saw a sky
or a wall that is flat, so a featureless region was always "blurred wash". This module draws both, as layers that the
synthesis (`synth.py`) composites like its sprites. Everything is a function of the generator, in a fixed order, and runs
on the training device in PyTorch (no loops over pixels; one scatter and a few small convolutions per layer).

**Thin structures** (`thin_layer`): five kinds, each a set of curves with a width along the curve:

- `twig`: a main branch from a frame edge, 4 side twigs and 4 sub-twigs on those, tapering;
- `hair`: 16 to 40 near-parallel strands, 1 to 1.6 px wide, wavy, in a clump;
- `wire`: 1 to 3 nearly straight, nearly parallel lines across the frame, 1 to 3 px, with a sag;
- `grass`: 24 to 48 tapered blades from one edge, curved;
- `bar`: 1 to 6 straight bars of 3 to 11 px (a chair leg, a pole, a railing), some tapered.

A curve is a polyline of `N_VERT` vertices (a random walk of its heading, a bend, a width at each vertex). It is drawn by
putting points every fraction of a pixel along it, splatting them bilinearly into one of `len(WIDTHS)` canvases by the
local width, and convolving each canvas with an antialiased disc of that width: the coverage of a line sums to its width
across the line, so a 1 px line at a half-pixel offset is two pixels at one half, as a real photo's is. The colour is the
hue of a random place in another tile, moved in brightness until it contrasts with the background where the structure
lies (an invisible structure would be a label nobody can read).

**Featureless layers** (`flat_layer`): a sky (a gradient under a wavy horizon, with faint large clouds), a wall or
sheet (1 to 3 ellipses of flat colour or a gentle gradient), or the whole background of a sample (`flat_background`).
They have sharp antialiased edges like any object, so a blurred one has a blurred edge, and nothing inside. Inside, far
from every edge, the radius cannot be read from the picture, so the loss weight there is low (`synth.render_layers`).
"""

import math

import torch
import torch.nn.functional as F

from training.bokeh.encode import luma_t, lower_median_t

TYPES = ("twig", "hair", "wire", "grass", "bar")
#: How often each kind is drawn.
TYPE_PROBS = (0.26, 0.20, 0.14, 0.16, 0.24)
#: Curve slots per layer and vertices per curve.
K_CURVES = 48
N_VERT = 12
#: Full line widths of the canvases (pixels). A point goes to the class nearest in log width.
WIDTHS = (1.0, 1.7, 2.6, 4.0, 6.0, 9.0)
#: Points along a curve per pixel of the sample's side (so the spacing stays under a pixel for the longest curve).
POINTS_PER_SIDE = 2.2


def _u(gen, lo, hi, *shape, device):
    return lo + (hi - lo) * torch.rand(*shape, generator=gen, device=device)


def _bern(gen, p, *shape, device):
    return torch.rand(*shape, generator=gen, device=device) < p


def _randint(gen, lo, hi, *shape, device):
    """Integers in [lo, hi)."""
    return torch.randint(lo, hi, shape, generator=gen, device=device)


# --- curves ---------------------------------------------------------------------------------------------------------


def _walk(p0, theta0, length, sigma, bend, hw0, hw1, gen):
    """Vertices (n, k, N_VERT, 2), headings (n, k, N_VERT) and half widths (n, k, N_VERT) of curves that start at `p0`
    (n, k, 2) with heading `theta0` (n, k), `length` (n, k) pixels, a heading that random-walks with step `sigma` (n, k)
    and bends by `bend` radians over its length; the half width goes from `hw0` to `hw1` (n, k)."""
    n, k = theta0.shape
    dev = theta0.device
    t = torch.linspace(0.0, 1.0, N_VERT, device=dev).view(1, 1, N_VERT)
    steps = torch.randn(n, k, N_VERT, generator=gen, device=dev) * sigma.unsqueeze(-1)
    steps[..., 0] = 0.0
    head = theta0.unsqueeze(-1) + bend.unsqueeze(-1) * t + torch.cumsum(steps, -1)
    seg = (length / (N_VERT - 1)).unsqueeze(-1).unsqueeze(-1)
    d = torch.stack([torch.cos(head), torch.sin(head)], -1) * seg
    pos = p0.unsqueeze(2) + torch.cumsum(torch.cat([torch.zeros_like(d[:, :, :1]), d[:, :, :-1]], 2), 2)
    hw = hw0.unsqueeze(-1) + (hw1 - hw0).unsqueeze(-1) * t
    return pos, head, hw


def _edge_start(side, along, s, dev):
    """A start point just outside the frame edge `side` (0 left, 1 right, 2 top, 3 bottom) at fraction/pixel `along`
    (n,), with the heading that points into the frame."""
    out = _stack_by_side(side, [
        torch.stack([-10 + 0 * along, along], -1), torch.stack([s + 10 + 0 * along, along], -1),
        torch.stack([along, -10 + 0 * along], -1), torch.stack([along, s + 10 + 0 * along], -1)])
    theta = torch.tensor([0.0, math.pi, math.pi / 2, -math.pi / 2], device=dev)[side]
    return out, theta


def _stack_by_side(side, options):
    """Picks, per sample, the `options[side]` row (each option (n, 2))."""
    stacked = torch.stack(options, 1)  # (n, 4, 2)
    return stacked[torch.arange(side.shape[0], device=side.device), side]


def _gather_vertex(x, pid, j):
    """x (n, kp, N, ...) at parent `pid` (n, kc) and vertex `j` (n, kc) -> (n, kc, ...)."""
    ar = torch.arange(x.shape[0], device=x.device).view(-1, 1)
    return x[ar, pid, j]


def _children(parent, pid, j, gen, scale_len=(0.25, 0.6)):
    """Side curves on `parent` = (pos, head, hw, length). `pid`, `j` (n, kc) pick the parent curve and vertex."""
    pos, head, hw, length = parent
    n, kc = pid.shape
    dev = pos.device
    p0 = _gather_vertex(pos, pid, j)
    th = _gather_vertex(head, pid, j) + torch.where(_bern(gen, 0.5, n, kc, device=dev), 1.0, -1.0) * _u(gen, 0.45, 1.1, n, kc, device=dev)
    ln = length[torch.arange(n, device=dev).view(-1, 1), pid] * _u(gen, *scale_len, n, kc, device=dev)
    hw0 = (_gather_vertex(hw, pid, j) * _u(gen, 0.55, 0.9, n, kc, device=dev)).clamp(min=0.45)
    out = _walk(p0, th, ln, _u(gen, 0.04, 0.18, n, kc, device=dev), _u(gen, -0.6, 0.6, n, kc, device=dev),
                hw0, torch.full_like(hw0, 0.3), gen)
    return out + (ln,)


def _twig(n, s, gen, dev):
    side = _randint(gen, 0, 4, n, device=dev)
    start, th0 = _edge_start(side, _u(gen, 0.1, 0.9, n, device=dev) * s, s, dev)
    length = (_u(gen, 0.45, 1.0, n, device=dev) * s).view(n, 1)
    hw0 = _u(gen, 1.0, 3.5, n, 1, device=dev)
    main = _walk(start.view(n, 1, 2), (th0 + _u(gen, -0.6, 0.6, n, device=dev)).view(n, 1), length,
                 _u(gen, 0.05, 0.16, n, 1, device=dev), _u(gen, -0.8, 0.8, n, 1, device=dev), hw0,
                 _u(gen, 0.35, 0.7, n, 1, device=dev), gen)
    main = main + (length,)
    l1 = _children(main, torch.zeros(n, 4, dtype=torch.long, device=dev), _randint(gen, 2, N_VERT - 2, n, 4, device=dev), gen)
    l2 = _children(l1, _randint(gen, 0, 4, n, 4, device=dev), _randint(gen, 2, N_VERT - 3, n, 4, device=dev), gen)
    n_side = _randint(gen, 2, 5, n, device=dev)
    alive1 = torch.arange(4, device=dev).view(1, 4) < n_side.view(n, 1)
    alive2 = _bern(gen, 0.6, n, 4, device=dev)
    pos = torch.cat([main[0], l1[0], l2[0]], 1)
    hw = torch.cat([main[2], l1[2], l2[2]], 1)
    alive = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=dev), alive1, alive2], 1)
    return pos, hw, alive


def _hair(n, s, gen, dev):
    kk = 40
    cnt = _randint(gen, 16, kk + 1, n, device=dev)
    theta_b = torch.where(_bern(gen, 0.6, n, device=dev), math.pi / 2 + _u(gen, -1.1, 1.1, n, device=dev), _u(gen, 0.0, 2 * math.pi, n, device=dev))
    d = torch.stack([torch.cos(theta_b), torch.sin(theta_b)], -1)
    perp = torch.stack([-torch.sin(theta_b), torch.cos(theta_b)], -1)
    center = _u(gen, 0.1, 0.9, n, 2, device=dev) * s
    span = _u(gen, 0.08, 0.45, n, device=dev) * s
    t = _u(gen, -0.5, 0.5, n, kk, device=dev)
    back = _u(gen, 0.0, 0.25, n, kk, device=dev) * s
    p0 = center.unsqueeze(1) + t.unsqueeze(-1) * span.view(n, 1, 1) * perp.unsqueeze(1) - back.unsqueeze(-1) * d.unsqueeze(1)
    th = theta_b.view(n, 1) + torch.randn(n, kk, generator=gen, device=dev) * 0.05
    length = _u(gen, 0.35, 0.95, n, kk, device=dev) * s
    bend = _u(gen, -1.0, 1.0, n, 1, device=dev) * _u(gen, 0.5, 1.3, n, kk, device=dev)
    hw0 = _u(gen, 0.45, 0.8, n, kk, device=dev)
    pos, _, hw = _walk(p0, th, length, _u(gen, 0.015, 0.06, n, 1, device=dev).expand(n, kk), bend, hw0, hw0 * _u(gen, 0.5, 0.9, n, kk, device=dev), gen)
    return pos, hw, torch.arange(kk, device=dev).view(1, kk) < cnt.view(n, 1)


def _wire(n, s, gen, dev):
    kk = 3
    cnt = _randint(gen, 1, kk + 1, n, device=dev)
    horizontal = _bern(gen, 0.5, n, device=dev)
    flip = _bern(gen, 0.5, n, device=dev)
    base = torch.where(horizontal, torch.where(flip, math.pi, 0.0), torch.where(flip, -math.pi / 2, math.pi / 2))
    theta = base + _u(gen, -0.5, 0.5, n, device=dev)
    d = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    perp = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
    q = _u(gen, 0.2, 0.8, n, 2, device=dev) * s
    gap = torch.cumsum(_u(gen, 12.0, 45.0, n, kk, device=dev), 1) - _u(gen, 12.0, 45.0, n, 1, device=dev)
    p0 = q.unsqueeze(1) - 0.9 * s * d.unsqueeze(1) + gap.unsqueeze(-1) * perp.unsqueeze(1)
    th = theta.view(n, 1) + _u(gen, -0.03, 0.03, n, kk, device=dev)
    hw0 = _u(gen, 0.5, 1.5, n, kk, device=dev)
    pos, _, hw = _walk(p0, th, torch.full((n, kk), 1.8 * s, device=dev), torch.full((n, kk), 0.003, device=dev),
                       _u(gen, -0.35, 0.35, n, 1, device=dev).expand(n, kk), hw0, hw0, gen)
    return pos, hw, torch.arange(kk, device=dev).view(1, kk) < cnt.view(n, 1)


def _grass(n, s, gen, dev):
    kk = 48
    cnt = _randint(gen, 24, kk + 1, n, device=dev)
    side = torch.where(_bern(gen, 0.8, n, device=dev), 3, _randint(gen, 0, 3, n, device=dev))
    x = _u(gen, -0.05, 1.05, n, kk, device=dev) * s
    out = _u(gen, 0.0, 15.0, n, kk, device=dev)
    opts = torch.stack([torch.stack([-out, x], -1), torch.stack([s + out, x], -1), torch.stack([x, -out], -1),
                        torch.stack([x, s + out], -1)], 2)  # (n, kk, 4, 2): the start on each side
    p0 = opts[torch.arange(n, device=dev).view(n, 1), torch.arange(kk, device=dev).view(1, kk), side.view(n, 1)]
    th0 = torch.tensor([0.0, math.pi, math.pi / 2, -math.pi / 2], device=dev)[side]
    th = th0.view(n, 1) + torch.randn(n, kk, generator=gen, device=dev) * 0.35
    bend = torch.where(_bern(gen, 0.5, n, kk, device=dev), 1.0, -1.0) * _u(gen, 0.0, 1.0, n, kk, device=dev)
    hw0 = _u(gen, 1.0, 2.2, n, kk, device=dev)
    pos, _, hw = _walk(p0, th, _u(gen, 0.12, 0.5, n, kk, device=dev) * s, torch.full((n, kk), 0.03, device=dev), bend, hw0,
                       torch.full_like(hw0, 0.25), gen)
    return pos, hw, torch.arange(kk, device=dev).view(1, kk) < cnt.view(n, 1)


def _bar(n, s, gen, dev):
    kk = 6
    railing = _bern(gen, 0.25, n, device=dev)
    cnt = torch.where(railing, _randint(gen, 3, kk + 1, n, device=dev), _randint(gen, 1, 4, n, device=dev))
    theta = torch.where(_bern(gen, 0.6, n, device=dev), math.pi / 2 + _u(gen, -0.35, 0.35, n, device=dev), _u(gen, 0.0, math.pi, n, device=dev))
    d = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    perp = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
    c = _u(gen, 0.15, 0.85, n, 2, device=dev) * s
    sp = _u(gen, 14.0, 60.0, n, 1, device=dev)
    k_idx = torch.arange(kk, device=dev).view(1, kk) - 0.5 * (cnt.view(n, 1) - 1)
    length = _u(gen, 0.35, 1.4, n, kk, device=dev) * s
    p0 = c.unsqueeze(1) + (k_idx * sp).unsqueeze(-1) * perp.unsqueeze(1) - (length * _u(gen, 0.2, 0.8, n, kk, device=dev)).unsqueeze(-1) * d.unsqueeze(1)
    hw0 = _u(gen, 1.5, 5.5, n, 1, device=dev) * _u(gen, 0.9, 1.1, n, kk, device=dev)
    hw1 = hw0 * torch.where(_bern(gen, 0.3, n, 1, device=dev), _u(gen, 0.5, 0.8, n, 1, device=dev), torch.ones(n, 1, device=dev))
    zero = torch.zeros(n, kk, device=dev)
    pos, _, hw = _walk(p0, theta.view(n, 1).expand(n, kk), length, zero, zero, hw0, hw1, gen)
    return pos, hw, torch.arange(kk, device=dev).view(1, kk) < cnt.view(n, 1)


_GENERATORS = {"twig": _twig, "hair": _hair, "wire": _wire, "grass": _grass, "bar": _bar}


def make_curves(types: torch.Tensor, s: int, gen, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Curves of one thin-structure layer for every sample: `types` (B,) indexes `TYPES`. Returns vertices
    (B, K_CURVES, N_VERT, 2) in the sample's pixel coordinates (they may lie outside the frame), half widths
    (B, K_CURVES, N_VERT) and which slots are used (B, K_CURVES)."""
    b = types.shape[0]
    verts = torch.zeros(b, K_CURVES, N_VERT, 2, device=dev)
    hws = torch.zeros(b, K_CURVES, N_VERT, device=dev)
    alive = torch.zeros(b, K_CURVES, dtype=torch.bool, device=dev)
    for ti, name in enumerate(TYPES):
        idx = (types == ti).nonzero().squeeze(1)
        if idx.numel():
            pos, hw, al = _GENERATORS[name](idx.numel(), float(s), gen, dev)
            k = pos.shape[1]
            verts[idx, :k] = pos
            hws[idx, :k] = hw
            alive[idx, :k] = al
    return verts, hws, alive


# --- drawing ----------------------------------------------------------------------------------------------------------


def _disc_kernel(width: float, device) -> torch.Tensor:
    """An antialiased disc of diameter `width`, as a (1, 1, K, K) convolution kernel (value 1 inside, a one-pixel ramp)."""
    r = int(math.ceil(width / 2 + 1.0))
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    d = torch.sqrt(ax.view(1, -1) ** 2 + ax.view(-1, 1) ** 2)
    return (width / 2 + 0.5 - d).clamp(0, 1).view(1, 1, 2 * r + 1, 2 * r + 1)


def rasterize(verts: torch.Tensor, hw: torch.Tensor, alive: torch.Tensor, canvas: int, margin: int, points: int) -> torch.Tensor:
    """The coverage (B, 1, canvas, canvas), in [0, 1], of the curves (module docstring). `verts` are in the sample's
    pixel coordinates; `margin` moves them onto the canvas."""
    b, k, n, _ = verts.shape
    dev = verts.device
    nc = len(WIDTHS)
    f = torch.linspace(0.0, n - 1.0, points, device=dev)
    i0 = f.floor().clamp(max=n - 2).long()
    fr = (f - i0).view(1, 1, -1, 1)
    pts = verts[:, :, i0] + (verts[:, :, i0 + 1] - verts[:, :, i0]) * fr + margin
    wid = 2.0 * (hw[:, :, i0] + (hw[:, :, i0 + 1] - hw[:, :, i0]) * fr[..., 0])
    length = torch.linalg.norm(verts[:, :, 1:] - verts[:, :, :-1], dim=-1).sum(-1)
    ds = length / (points - 1)
    wc = torch.tensor(WIDTHS, device=dev)
    kernels = [_disc_kernel(w, dev) for w in WIDTHS]
    # a point's weight makes a line of the class's width cover that much across (the kernel's area over the width is the
    # coverage a unit of line density gives; the profile is a rounded box, clamped at one)
    per_len = wc / torch.stack([k.sum() for k in kernels])
    cls = torch.argmin((torch.log(wid.clamp(min=0.3)).unsqueeze(-1) - torch.log(wc)).abs(), -1)  # (B, K, M)
    wt = ds.unsqueeze(-1) * per_len[cls] * alive.unsqueeze(-1)
    x, y = pts[..., 0] - 0.5, pts[..., 1] - 0.5
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    base = (torch.arange(b, device=dev).view(b, 1, 1) * nc + cls) * (canvas * canvas)
    acc = torch.zeros(b * nc * canvas * canvas, device=dev)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        inside = (xi >= 0) & (xi < canvas) & (yi >= 0) & (yi < canvas)
        idx = base + yi.clamp(0, canvas - 1).long() * canvas + xi.clamp(0, canvas - 1).long()
        acc.index_add_(0, idx.flatten(), (wt * w * inside).flatten())
    acc = acc.view(b, nc, canvas, canvas)
    used = torch.bincount(cls[wt > 0].flatten(), minlength=nc).tolist()
    out = torch.zeros(b, 1, canvas, canvas, device=dev)
    for c in range(nc):
        if used[c]:
            ker = kernels[c]
            out = torch.maximum(out, F.conv2d(acc[:, c:c + 1], ker, padding=ker.shape[-1] // 2).clamp(0, 1))
    return out


def _pick_colour(lin: torch.Tensor, gen, centre: torch.Tensor, canvas_bg_lum_floor: float = 1e-4) -> torch.Tensor:
    """A colour (B, 3) for a thin structure: the hue of a random place in another tile of the batch, moved in brightness
    until it is dark (6 to 30 %) or bright (2.5 to 6 times) against the local background around `centre` (B, 2) in
    pixels. Contrast is what makes the structure's label readable."""
    b, _, s, _ = lin.shape
    dev = lin.device
    near = F.avg_pool2d(lin, 9, 1, 4)
    wide = F.avg_pool2d(lin, 25, 1, 12)
    donor = (torch.arange(b, device=dev) + torch.randint(1, max(b, 2), (b,), generator=gen, device=dev)) % b
    yy = torch.randint(0, s, (b,), generator=gen, device=dev)
    xx = torch.randint(0, s, (b,), generator=gen, device=dev)
    col = near[donor, :, yy, xx]  # (B, 3)
    cy = centre[:, 1].round().long().clamp(0, s - 1)
    cx = centre[:, 0].round().long().clamp(0, s - 1)
    bg = wide[torch.arange(b, device=dev), :, cy, cx]
    l_bg = luma_t(bg.view(b, 3, 1, 1)).view(b).clamp(min=canvas_bg_lum_floor)
    l_c = luma_t(col.view(b, 3, 1, 1)).view(b).clamp(min=canvas_bg_lum_floor)
    med = lower_median_t(luma_t(lin)).clamp(min=canvas_bg_lum_floor)
    ratio = l_c / l_bg
    dark = _bern(gen, 0.75, b, device=dev) & (l_bg > 0.3 * med)
    factor = torch.where(dark, _u(gen, 0.06, 0.30, b, device=dev), _u(gen, 2.5, 6.0, b, device=dev))
    fixed = col / l_c.view(b, 1) * (l_bg * factor).view(b, 1)
    low_contrast = (ratio > 0.45) & (ratio < 2.2)
    return torch.where(low_contrast.view(b, 1), fixed, col)


def thin_layer(lin: torch.Tensor, types: torch.Tensor, active: torch.Tensor, gen, canvas: int, margin: int):
    """One thin-structure layer for every sample: premultiplied colour (B, 3, canvas, canvas) and alpha
    (B, 1, canvas, canvas). `lin` (B, 3, S, S) is the sample's sharp picture (colours and background brightness come
    from it), `types` (B,) the kinds, `active` (B,) which samples have one."""
    b, _, s, _ = lin.shape
    dev = lin.device
    verts, hw, alive = make_curves(types, s, gen, dev)
    alive = alive & active.view(b, 1)
    alpha = rasterize(verts, hw, alive, canvas, margin, max(64, int(POINTS_PER_SIDE * s))) * active.view(b, 1, 1, 1)
    w = alive.unsqueeze(-1).float()
    centre = (verts.mean(2) * w).sum(1) / w.sum(1).clamp(min=1.0)  # (B, 2)
    col = _pick_colour(lin, gen, centre.clamp(0, s - 1))
    return col.view(b, 3, 1, 1) * alpha, alpha


# --- featureless layers -----------------------------------------------------------------------------------------------

#: Hues (unit luminance after normalising) of skies and walls: blue sky, pale cloudy, grey, warm wall, green, dusk.
_PALETTE = ((0.55, 0.70, 1.00), (0.85, 0.90, 1.00), (1.00, 1.00, 1.00), (1.00, 0.88, 0.70), (0.70, 0.90, 0.60), (1.00, 0.75, 0.55))


def _smooth_field(b: int, cv: int, cells: int, gen, dev) -> torch.Tensor:
    """(B, 1, cv, cv) smooth noise of unit-ish scale: a `cells` x `cells` random grid, bicubic."""
    g = torch.randn(b, 1, cells, cells, generator=gen, device=dev)
    return F.interpolate(g, size=(cv, cv), mode="bicubic", align_corners=False)


def _flat_colours(lin: torch.Tensor, gen, cv: int, sky: bool):
    """Two colours (B, 3) and a gradient (B, 1, cv, cv) in [0, 1] between them, brightness tied to the tile's median."""
    b = lin.shape[0]
    dev = lin.device
    med = lower_median_t(luma_t(lin)).clamp(min=1e-4)
    pal = torch.tensor(_PALETTE, device=dev)
    pal = pal / luma_t(pal.view(-1, 3, 1, 1)).view(-1, 1)
    pick = lambda: pal[torch.randint(0, pal.shape[0], (b,), generator=gen, device=dev)]  # noqa: E731
    lum0 = med * torch.exp2(_u(gen, -1.5, 2.8, b, device=dev))
    lum1 = lum0 * torch.exp2(_u(gen, -0.7, 0.7, b, device=dev) * (1.0 if sky else 0.6))
    c0, c1 = pick() * lum0.view(b, 1), pick() * lum1.view(b, 1)
    c1 = torch.where(_bern(gen, 0.5, b, 1, device=dev), c0 * (lum1 / lum0).view(b, 1), c1)  # half of them keep their hue
    ax = (torch.arange(cv, dtype=torch.float32, device=dev) + 0.5) / cv * 2 - 1
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    phi = torch.where(_bern(gen, 0.7, b, device=dev), math.pi / 2 + _u(gen, -0.3, 0.3, b, device=dev), _u(gen, 0.0, 2 * math.pi, b, device=dev))
    g = 0.5 + 0.5 * (torch.cos(phi).view(b, 1, 1) * gx + torch.sin(phi).view(b, 1, 1) * gy) / (
        torch.abs(torch.cos(phi)) + torch.abs(torch.sin(phi))).view(b, 1, 1)
    radial = _bern(gen, 0.2, b, device=dev).view(b, 1, 1)
    g = torch.where(radial, (gx * gx + gy * gy).sqrt().clamp(max=1.0), g)
    return c0, c1, g.unsqueeze(1)


def _flat_content(lin: torch.Tensor, gen, cv: int, sky: bool) -> torch.Tensor:
    """(B, 3, cv, cv) colour of a featureless layer: a gradient between two colours, plus faint large clouds."""
    b = lin.shape[0]
    c0, c1, g = _flat_colours(lin, gen, cv, sky)
    col = c0.view(b, 3, 1, 1) * (1 - g) + c1.view(b, 3, 1, 1) * g
    amp = _u(gen, 0.0, 0.06 if sky else 0.025, b, device=lin.device).view(b, 1, 1, 1)
    return col * (1 + amp * _smooth_field(b, cv, 5, gen, lin.device).clamp(-2, 2))


def _sky_alpha(b: int, cv: int, margin: int, s: int, gen, dev) -> torch.Tensor:
    """(B, 1, cv, cv): 1 above a wavy horizon (antialiased), the sky of a picture (or a wall at the side)."""
    ax = torch.arange(cv, dtype=torch.float32, device=dev) + 0.5 - margin  # sample pixel coordinates
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    y0 = _u(gen, 0.2, 0.7, b, device=dev) * s
    horizon = y0.view(b, 1, 1)
    for _ in range(3):
        amp = _u(gen, 0.0, 0.08, b, device=dev) * s
        freq = _u(gen, 0.5, 3.0, b, device=dev)
        ph = _u(gen, 0.0, 2 * math.pi, b, device=dev)
        horizon = horizon + amp.view(b, 1, 1) * torch.sin(2 * math.pi * freq.view(b, 1, 1) * gx / s + ph.view(b, 1, 1))
    rough_amp = _u(gen, 0.0, 6.0, b, device=dev)
    rough_f = _u(gen, 8.0, 40.0, b, device=dev)
    horizon = horizon + rough_amp.view(b, 1, 1) * torch.sin(2 * math.pi * rough_f.view(b, 1, 1) * gx / s + _u(gen, 0.0, 6.28, b, device=dev).view(b, 1, 1))
    alpha = (0.5 + (horizon - gy)).clamp(0, 1).unsqueeze(1)
    turn = _randint(gen, 0, 4, b, device=dev)  # mostly the sky is on top; a wall at the side otherwise
    turn = torch.where(_bern(gen, 0.7, b, device=dev), torch.zeros_like(turn), turn)
    out = alpha
    for k in (1, 2, 3):
        sel = (turn == k).view(b, 1, 1, 1)
        out = torch.where(sel, torch.rot90(alpha, k, (2, 3)), out)
    return out


def _ellipse_alpha(b: int, cv: int, margin: int, s: int, gen, dev) -> torch.Tensor:
    """(B, 1, cv, cv): the union of 1 to 3 antialiased ellipses of 0.12 to 0.5 of the frame."""
    ax = torch.arange(cv, dtype=torch.float32, device=dev) + 0.5 - margin
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    out = torch.zeros(b, 1, cv, cv, device=dev)
    n_e = _randint(gen, 1, 4, b, device=dev)
    for i in range(3):
        cx, cy = _u(gen, 0.1, 0.9, b, device=dev) * s, _u(gen, 0.1, 0.9, b, device=dev) * s
        a, bb = _u(gen, 0.12, 0.5, b, device=dev) * s, _u(gen, 0.12, 0.5, b, device=dev) * s
        th = _u(gen, 0.0, math.pi, b, device=dev)
        c, sn = torch.cos(th).view(b, 1, 1), torch.sin(th).view(b, 1, 1)
        dx, dy = gx - cx.view(b, 1, 1), gy - cy.view(b, 1, 1)
        u, v = (c * dx + sn * dy) / a.view(b, 1, 1), (-sn * dx + c * dy) / bb.view(b, 1, 1)
        rho = torch.sqrt(u * u + v * v)
        al = (0.5 + (1 - rho) * torch.minimum(a, bb).view(b, 1, 1)).clamp(0, 1).unsqueeze(1)
        out = torch.maximum(out, al * (i < n_e).view(b, 1, 1, 1))
    return out


def flat_layer(lin: torch.Tensor, active: torch.Tensor, gen, canvas: int, margin: int):
    """One featureless layer for every sample (module docstring): premultiplied colour (B, 3, canvas, canvas) and alpha
    (B, 1, canvas, canvas)."""
    b, _, s, _ = lin.shape
    dev = lin.device
    sky = _bern(gen, 0.6, b, device=dev)
    alpha = torch.where(sky.view(b, 1, 1, 1), _sky_alpha(b, canvas, margin, s, gen, dev), _ellipse_alpha(b, canvas, margin, s, gen, dev))
    alpha = alpha * active.view(b, 1, 1, 1)
    col = torch.where(sky.view(b, 1, 1, 1), _flat_content(lin, gen, canvas, True), _flat_content(lin, gen, canvas, False))
    return col * alpha, alpha


def flat_background(lin: torch.Tensor, gen, canvas: int) -> torch.Tensor:
    """(B, 3, canvas, canvas): a featureless picture for the background of a sample (a sky, a wall, a sheet)."""
    b = lin.shape[0]
    sky = _bern(gen, 0.6, b, device=lin.device).view(b, 1, 1, 1)
    return torch.where(sky, _flat_content(lin, gen, canvas, True), _flat_content(lin, gen, canvas, False))
