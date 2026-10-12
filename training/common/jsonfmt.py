"""Deterministic, compact JSON writer for the generated data files.

``json.dumps(indent=2)`` puts every number of a 2351-row table on its own line. These files
are committed and reviewed, so we want them small and stable:

* dicts are written one key per line, in insertion order (callers control the order);
* a list of numbers is written on one line when short, otherwise wrapped ``per_line`` items
  to a line;
* a list of short number lists (a 3x3 matrix, a table of ``[a, b]`` rows) with at most
  ``MATRIX_MAX_ROWS`` rows is written on one line when it fits ``LINE_WIDTH`` and one row per
  line otherwise; longer tables get ``ROWS_PER_LINE`` rows to a line;
* floats use ``repr`` (shortest string that round-trips), so values rounded by the caller
  keep their short form and unrounded values keep every bit.

The output is a pure function of the input object: no timestamps, no platform-dependent
formatting. NaN and infinity are rejected.
"""

from __future__ import annotations

import math
from typing import Any

MATRIX_MAX_ROWS = 4
NUMBERS_PER_LINE = 8
ROWS_PER_LINE = 4
LINE_WIDTH = 100


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num(x: Any) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    f = float(x)
    if not math.isfinite(f):
        raise ValueError(f"non-finite number in JSON output: {x!r}")
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f)}.0"
    return repr(f)


def _scalar(x: Any) -> str:
    import json

    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if _is_number(x):
        return _num(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    raise TypeError(f"unsupported scalar {type(x)}")


def _inline(x: Any) -> str:
    """One-line rendering of a number list or a list of number lists."""
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_inline(v) for v in x) + "]"
    return _scalar(x)


def _is_number_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_number(v) for v in x)


def _is_row_table(x: Any) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(_is_number_list(r) for r in x)
    )


def _wrap(items: list[str], per_line: int, pad: str) -> str:
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(pad + ", ".join(items[i:i + per_line]))
    return "[\n" + ",\n".join(lines) + "\n" + pad[:-2] + "]"


def _render(x: Any, level: int) -> str:
    pad = "  " * (level + 1)
    if isinstance(x, dict):
        if not x:
            return "{}"
        parts = []
        for k, v in x.items():
            parts.append(f"{pad}{_scalar(str(k))}: {_render(v, level + 1)}")
        return "{\n" + ",\n".join(parts) + "\n" + "  " * level + "}"
    if isinstance(x, (list, tuple)):
        if not x:
            return "[]"
        if _is_number_list(x):
            if len(x) <= NUMBERS_PER_LINE:
                return _inline(x)
            return _wrap([_num(v) for v in x], NUMBERS_PER_LINE, pad)
        if _is_row_table(x):
            one = _inline(x)
            if len(x) <= MATRIX_MAX_ROWS:
                if len(one) <= LINE_WIDTH:
                    return one
                return "[\n" + ",\n".join(pad + _inline(r) for r in x) + "\n" + "  " * level + "]"
            return _wrap([_inline(r) for r in x], ROWS_PER_LINE, pad)
        parts = [f"{pad}{_render(v, level + 1)}" for v in x]
        return "[\n" + ",\n".join(parts) + "\n" + "  " * level + "]"
    return _scalar(x)


def dumps(obj: Any) -> str:
    """Render ``obj`` as deterministic compact JSON, with a trailing newline."""
    return _render(obj, 0) + "\n"
