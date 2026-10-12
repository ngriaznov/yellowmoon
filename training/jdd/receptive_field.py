"""Receptive field and tile halo of a `jdd` network, computed from the real module graph.

The graph is traced with ``torch.fx`` (so it is the graph the code actually executes, not a
hand-written copy), and the *exact* set of input positions that influence one output pixel is
propagated backwards, operation by operation, along one spatial axis:

* ``Conv2d`` (kernel k, stride s, padding p, dilation d): output ``i`` reads ``i*s - p + d*t``;
* ``pixel_shuffle`` (factor r): output ``i`` reads input ``i // r``;
* concat, add, multiply, SiLU, channel slicing: position-wise.

All kernels are square, so the 2-D footprint is the product of the 1-D footprints. Because strided
convolutions make the footprint depend on the output position modulo the network stride, the
analysis is repeated for every phase and the worst case is reported. The test-suite checks the result
against a brute-force gradient probe of the real network (``tests/test_receptive_field.py``).

Terminology (docs/07): the receptive field ``rf_half_px`` is in half-resolution input pixels; in
full-resolution pixels it is twice that. The halo (the margin tiles overlap by, then crop) is half the
full-resolution receptive field rounded up to a multiple of 16 full-resolution pixels, and it is
additionally forced to cover the worst-case one-sided reach of any phase.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

HALO_MULTIPLE_FULL_PX = 16
RF_LIMIT_FULL_PX = 128


@dataclass(frozen=True)
class ReceptiveField:
    """Result of :func:`analyse`. All ``*_half_px`` values are half-resolution pixels."""

    rf_half_px: int  # width of the widest footprint over all phases
    rf_full_px: int  # the same in full-resolution pixels (2x)
    reach_half_px: int  # worst one-sided reach (max over phases of max(left, right))
    halo_half_px: int  # tile halo, half-resolution pixels (multiple of 8)
    halo_full_px: int  # tile halo, full-resolution pixels (multiple of 16)
    spec_halo_full_px: int  # docs/07 formula alone: ceil(rf_full / 2 / 16) * 16
    per_phase: tuple[tuple[int, int], ...]  # (left, right) reach per half-res phase
    stride: int  # network stride (phase period, half-res pixels)


def _pointwise_targets() -> set:
    return {
        F.silu,
        torch.sigmoid,
        F.relu,
        F.gelu,
        torch.relu,
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        torch.add,
        torch.mul,
    }


def _conv_inputs(mod: nn.Conv2d, out_idx: set[int]) -> set[int]:
    s = mod.stride[0]
    p = mod.padding[0]
    d = mod.dilation[0]
    k = mod.kernel_size[0]
    return {i * s - p + d * t for i in out_idx for t in range(k)}


def footprint_1d(model: nn.Module, out_index_full: int) -> set[int]:
    """Half-res input indices that influence full-res output index ``out_index_full`` (one axis).

    The axis is treated as infinite (no border), so zero padding never truncates the footprint.
    Raises ``NotImplementedError`` for an operation the analysis does not understand, so a
    new layer type can never be silently mis-counted.
    """
    gm = fx.symbolic_trace(model)
    mods = dict(gm.named_modules())
    demand: dict[fx.Node, set[int]] = {}
    result: set[int] = set()
    pointwise = _pointwise_targets()
    nodes = list(gm.graph.nodes)
    out_node = next(n for n in nodes if n.op == "output")
    (final,) = out_node.args
    demand[final] = {out_index_full}
    for node in reversed(nodes):
        if node.op == "output":
            continue
        idx = demand.get(node)
        if node.op == "placeholder":
            if idx:
                result |= idx
            continue
        if node.op == "get_attr":
            continue
        if idx is None:
            continue  # not on a path to the output (e.g. unused branch)

        def push(src, indices: set[int]) -> None:
            if isinstance(src, fx.Node):
                demand.setdefault(src, set()).update(indices)

        if node.op == "call_module":
            mod = mods[node.target]
            if isinstance(mod, nn.Conv2d):
                if mod.kernel_size[0] != mod.kernel_size[1] or mod.stride[0] != mod.stride[1]:
                    raise NotImplementedError("non-square conv")
                push(node.args[0], _conv_inputs(mod, idx))
            elif isinstance(mod, nn.PixelShuffle):
                r = mod.upscale_factor
                push(node.args[0], {i // r for i in idx})
            elif isinstance(mod, (nn.SiLU, nn.ReLU, nn.Identity, nn.Sigmoid)):
                push(node.args[0], idx)
            else:
                raise NotImplementedError(f"module {type(mod).__name__}")
        elif node.op == "call_function":
            tgt = node.target
            if tgt is F.pixel_shuffle:
                r = node.args[1]
                push(node.args[0], {i // r for i in idx})
            elif tgt is torch.cat:
                for src in node.args[0]:
                    push(src, idx)
            elif tgt in pointwise:
                for src in node.args:
                    push(src, idx)
            elif tgt is operator.getitem:
                spec = node.args[1]
                spec = spec if isinstance(spec, tuple) else (spec,)
                # Only batch/channel indexing is position-wise; spatial indexing is unsupported.
                for extra in spec[2:]:
                    if extra != slice(None):
                        raise NotImplementedError("spatial indexing")
                push(node.args[0], idx)
            else:
                raise NotImplementedError(f"function {tgt}")
        elif node.op == "call_method":
            if node.target in ("mul", "add", "sub", "div", "silu", "contiguous", "float"):
                push(node.args[0], idx)
                for a in node.args[1:]:
                    push(a, idx)
            else:
                raise NotImplementedError(f"method {node.target}")
        else:
            raise NotImplementedError(f"node op {node.op}")
    return result


def network_stride(model: nn.Module) -> int:
    """Product of all conv strides on the deepest path (the phase period in half-res px)."""
    stride = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.stride[0] > 1:
            stride *= m.stride[0]
    return stride


def analyse(model: nn.Module, out_scale: int = 2) -> ReceptiveField:
    """Compute receptive field and halo of ``model`` (a network built like ``JddNet``)."""
    stride = network_stride(model)
    base = 64 * stride  # far from zero so no index is clipped by an implicit border
    per_phase: list[tuple[int, int]] = []
    widths: list[int] = []
    for phase in range(stride):
        half_idx = base + phase
        fp = footprint_1d(model, half_idx * out_scale)  # first full-res pixel of the half-res cell
        lo, hi = min(fp), max(fp)
        # A half-res cell also owns the other output pixel(s) of the shuffle; they share the footprint.
        for sub in range(1, out_scale):
            fp2 = footprint_1d(model, half_idx * out_scale + sub)
            lo, hi = min(lo, min(fp2)), max(hi, max(fp2))
        per_phase.append((half_idx - lo, hi - half_idx))
        widths.append(hi - lo + 1)
    rf_half = max(widths)
    rf_full = out_scale * rf_half
    reach = max(max(lr) for lr in per_phase)
    spec_halo_full = -(-(rf_full // 2) // HALO_MULTIPLE_FULL_PX) * HALO_MULTIPLE_FULL_PX
    need_full = out_scale * reach
    halo_full = max(spec_halo_full, -(-need_full // HALO_MULTIPLE_FULL_PX) * HALO_MULTIPLE_FULL_PX)
    return ReceptiveField(
        rf_half_px=rf_half,
        rf_full_px=rf_full,
        reach_half_px=reach,
        halo_half_px=halo_full // out_scale,
        halo_full_px=halo_full,
        spec_halo_full_px=spec_halo_full,
        per_phase=tuple(per_phase),
        stride=stride,
    )


def assert_within_limit(rf: ReceptiveField) -> None:
    """docs/07: the receptive field must stay at or below 128 full-resolution pixels."""
    if rf.rf_full_px > RF_LIMIT_FULL_PX:
        raise AssertionError(f"receptive field {rf.rf_full_px} full-res px exceeds {RF_LIMIT_FULL_PX}")


def probe_footprint(model: nn.Module, out_y_full: int, out_x_full: int, size_half: int) -> tuple[set[int], set[int]]:
    """Brute-force check: gradient support of one output pixel over a ``size_half`` square input.

    Runs the real network in float64 with a random input, back-propagates the sum of the three
    colour outputs at one full-resolution pixel and returns the sets of rows and columns (half-res)
    where any input channel received a non-zero gradient. Used by the tests to validate
    :func:`footprint_1d`.
    """
    m = model.double().eval()
    x = torch.randn(1, 7, size_half, size_half, dtype=torch.float64, requires_grad=True)
    y = m(x)
    y[0, :, out_y_full, out_x_full].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1))
    ys = set(torch.nonzero(g.sum(dim=1) > 0).flatten().tolist())
    xs = set(torch.nonzero(g.sum(dim=0) > 0).flatten().tolist())
    return ys, xs


def describe(rf: ReceptiveField) -> str:
    """One-paragraph human-readable summary (used by the model card and the CLI)."""
    return (
        f"receptive field {rf.rf_half_px} half-res px = {rf.rf_full_px} full-res px "
        f"(limit {RF_LIMIT_FULL_PX}); worst one-sided reach {rf.reach_half_px} half-res px; "
        f"tile halo {rf.halo_full_px} full-res px = {rf.halo_half_px} half-res px "
        f"(docs/07 formula alone: {rf.spec_halo_full_px})"
    )


if __name__ == "__main__":  # python3 -m training.jdd.receptive_field
    from .model import JddNet, count_flops, count_parameters

    net = JddNet()
    result = analyse(net)
    print(describe(result))
    print("per-phase (left, right):", result.per_phase)
    print("params:", count_parameters(net))
    fl = count_flops(net, 64, 64)
    print(f"FLOP per output pixel (network): {fl['per_output_pixel'] / 1e3:.1f} k (budget 250 k)")
