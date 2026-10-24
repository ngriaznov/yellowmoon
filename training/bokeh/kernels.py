"""The kernel library: the shapes an out-of-focus point of light takes in real lenses, rendered on the pixel grid.

Lens character reshapes the bokeh a photo already has, so the network must not learn one shape. Eight kinds:
`disc` (uniform, antialiased rim), `soft` (soft rim, 5 to 18 % of the radius), `bright` (bright rim, spherical
aberration), `dark` (dark rim: brightest in the middle), `onion` (concentric ripples), `poly` (5 to 9 blades, corners
partly rounded), `cats` (cat's eye: the disc cut by a second, shifted disc) and `gauss` (a Gaussian, for the smallest
radii and soft lenses). A slight ellipticity (up to 8 %) is mixed in everywhere.

**The label radius.** Whatever the shape, a kernel's radius `r` is the radius of the uniform disc with the same
second moment about its centroid: `r = sqrt(2 * sum(k * ((x - cx)^2 + (y - cy)^2)))`. For a uniform disc that is its
radius (second moment `r^2 / 2`); for a polygon, a cat's eye or a ringed disc it is the size a viewer reads and the
size the spec's `b` stands for. `make_kernels` scales each shape until the measured radius is the wanted one
(three fixed-point rounds) and moves its centroid to the grid's centre, so a blurred picture is not shifted.
Kernels sum to 1. A radius under 0.5 px is an exact one-pixel delta (no blur).
"""

import math

import torch

KINDS = ("disc", "soft", "bright", "dark", "onion", "poly", "cats", "gauss")
#: How often each kind is drawn (the disc and the polygons are what most lenses give).
KIND_WEIGHTS = (0.22, 0.13, 0.10, 0.07, 0.07, 0.22, 0.10, 0.09)
#: Half size of the kernel grid: kernels are `(2 * KERNEL_R + 1)` pixels square. Radii above about 36 px do not fit.
KERNEL_R = 44
DELTA_BELOW_PX = 0.5


def _rand(gen: torch.Generator | None, n: int, device, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)


def sample_params(n: int, gen: torch.Generator | None, device, weights=KIND_WEIGHTS) -> dict[str, torch.Tensor]:
    """Random shape parameters for `n` kernels (each a tensor of shape (n,))."""
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    kind = torch.multinomial(w.repeat(n, 1), 1, generator=gen).squeeze(1)
    return {
        "kind": kind,
        "rim": _rand(gen, n, device, 0.05, 0.18),
        "bright": _rand(gen, n, device, 0.3, 1.2),
        "dark": _rand(gen, n, device, 0.3, 0.75),
        "rings": torch.floor(_rand(gen, n, device, 2.0, 5.999)),
        "ring_amp": _rand(gen, n, device, 0.08, 0.3),
        "blades": torch.floor(_rand(gen, n, device, 5.0, 9.999)),
        "rot": _rand(gen, n, device, 0.0, 2 * math.pi),
        "round": _rand(gen, n, device, 0.0, 0.5),
        "cats": _rand(gen, n, device, 0.25, 0.7),
        "cats_rot": _rand(gen, n, device, 0.0, 2 * math.pi),
        "ellip": _rand(gen, n, device, 0.0, 0.08),
    }


def _smoothstep(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _coverage(rho: torch.Tensor, shift: torch.Tensor, p: dict[str, torch.Tensor], radius_cells: int) -> torch.Tensor:
    """(N, K, K) unnormalised kernels of nominal radius `rho` (N,) centred at `shift` (N, 2) (x, y in pixels). Each
    kind is rendered only for the kernels that have it."""
    n = rho.shape[0]
    out = torch.zeros(n, 2 * radius_cells + 1, 2 * radius_cells + 1, device=rho.device)
    for ki, name in enumerate(KINDS):
        idx = (p["kind"] == ki).nonzero().squeeze(1)
        if idx.numel():
            sub = {k: v[idx] for k, v in p.items()}
            out[idx] = _shape(name, rho[idx], shift[idx], sub, radius_cells)
    return out


def _shape(name: str, rho: torch.Tensor, shift: torch.Tensor, p: dict[str, torch.Tensor], radius_cells: int):
    """(M, K, K) coverage of one kind of shape for M kernels."""
    n = rho.shape[0]
    ax = torch.arange(-radius_cells, radius_cells + 1, dtype=torch.float32, device=rho.device)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    c = lambda t: t.view(n, 1, 1)  # noqa: E731
    x = gx[None] - c(shift[:, 0])
    y = gy[None] - c(shift[:, 1])
    e = c(p["ellip"])
    xe, ye = x / (1 + e), y * (1 + e)
    r = c(rho)
    d = torch.sqrt(xe * xe + ye * ye)
    disc = (0.5 - (d - r)).clamp(0, 1)
    dn = (d / r.clamp(min=1e-3)).clamp(max=1.2)
    if name == "disc":
        return disc
    if name == "soft":
        return 1 - _smoothstep(r * (1 - c(p["rim"])), r * (1 + c(p["rim"])), d)
    if name == "bright":
        return disc * (1 + c(p["bright"]) * dn**6)
    if name == "dark":
        return disc * (1 - c(p["dark"]) * dn**2)
    if name == "onion":
        return disc * (1 + c(p["ring_amp"]) * torch.cos(2 * math.pi * c(p["rings"]) * dn))
    if name == "poly":
        nb = c(p["blades"])
        apothem = r * torch.cos(math.pi / nb)
        sd_poly = torch.full_like(d, -1e9)
        for k in range(9):
            th = c(p["rot"]) + 2 * math.pi * k / nb
            edge = xe * torch.cos(th) + ye * torch.sin(th) - apothem
            sd_poly = torch.where(k < nb, torch.maximum(sd_poly, edge), sd_poly)
        equal_area = r * torch.sqrt(nb * torch.sin(2 * math.pi / nb) / (2 * math.pi))
        sd = (1 - c(p["round"])) * sd_poly + c(p["round"]) * (d - equal_area)
        return (0.5 - sd).clamp(0, 1)
    if name == "cats":
        ox, oy = c(p["cats"]) * r * torch.cos(c(p["cats_rot"])), c(p["cats"]) * r * torch.sin(c(p["cats_rot"]))
        d2 = torch.sqrt((xe - ox) ** 2 + (ye - oy) ** 2)
        return (0.5 - torch.maximum(d - r, d2 - r)).clamp(0, 1)
    sigma = (r / 2).clamp(min=0.3)  # gauss
    return torch.exp(-(d * d) / (2 * sigma * sigma)) * (d <= radius_cells).float()


def moments(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For normalised kernels (N, K, K): the centroid (cx, cy) in pixels from the grid's centre, and the label radius
    `sqrt(2 * second moment about the centroid)`."""
    radius_cells = (k.shape[-1] - 1) // 2
    ax = torch.arange(-radius_cells, radius_cells + 1, dtype=k.dtype, device=k.device)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    cx, cy = (k * gx).sum((1, 2)), (k * gy).sum((1, 2))
    m2 = (k * ((gx[None] - cx.view(-1, 1, 1)) ** 2 + (gy[None] - cy.view(-1, 1, 1)) ** 2)).sum((1, 2))
    return cx, cy, torch.sqrt(2 * m2)


def make_kernels(radius: torch.Tensor, p: dict[str, torch.Tensor], radius_cells: int = KERNEL_R) -> torch.Tensor:
    """(N, K, K) kernels, each summing to 1, of label radius `radius` (N,) in pixels (the module docstring)."""
    n = radius.shape[0]
    radius = radius.float()
    rho = radius.clamp(min=0.6)
    shift = torch.zeros(n, 2, device=radius.device)
    for _ in range(3):
        cov = _coverage(rho, shift, p, radius_cells)
        k = cov / cov.sum((1, 2), keepdim=True).clamp(min=1e-9)
        cx, cy, r_eq = moments(k)
        rho = (rho * radius.clamp(min=0.6) / r_eq.clamp(min=0.2)).clamp(max=radius_cells - 2)
        shift = shift + torch.stack([-cx, -cy], 1)
    cov = _coverage(rho, shift, p, radius_cells)
    k = cov / cov.sum((1, 2), keepdim=True).clamp(min=1e-9)
    delta = torch.zeros_like(k)
    delta[:, radius_cells, radius_cells] = 1.0
    return torch.where((radius < DELTA_BELOW_PX).view(n, 1, 1), delta, k)


def conv_fft(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Circular convolution of images (..., C, H, W) with one kernel per leading index, (..., 1, K, K) or
    (..., K, K) broadcastable over the channels, the kernel's centre at the origin. Pad the picture by
    `KERNEL_R` on every side and crop after, so the wrap-around never reaches the part that is kept."""
    h, w = img.shape[-2:]
    kk = kernel.shape[-1]
    r = (kk - 1) // 2
    kpad = torch.zeros(*kernel.shape[:-2], h, w, dtype=kernel.dtype, device=kernel.device)
    kpad[..., :kk, :kk] = kernel
    kpad = torch.roll(kpad, shifts=(-r, -r), dims=(-2, -1))
    return torch.fft.irfft2(torch.fft.rfft2(img) * torch.fft.rfft2(kpad), s=(h, w))
