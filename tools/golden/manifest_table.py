#!/usr/bin/env python3
"""Prints the golden-set table of docs/15 M0-09 (file, body, light, ISO, licence tier) as Markdown, from tools/golden/manifest.json.

    python3 -I tools/golden/manifest_table.py                # the default set (what CI renders: `golden check`)
    python3 -I tools/golden/manifest_table.py --set full     # another set: default, full, eval, or all
    python3 -I tools/golden/manifest_table.py --summary      # one row per body and variant instead of one per file

The "licence tier" of a synthetic file is "ours (generated)": the files are written by tools/golden `synth` from our own scenes and
need no download (docs/validation-data.md). The CC0 and free-licence tiers (tools/fetch/golden.sh) have no entries yet; the table says
so instead of leaving them out. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
TIER_TEXT = {"synthetic": "ours (generated)", "synthetic-eval": "ours (generated, CIE data: local only)"}


def select(manifest: Dict[str, Any], set_name: str) -> List[Dict[str, Any]]:
    """The entries of one set (``all`` = every entry), in manifest order."""
    return [e for e in manifest["entries"] if set_name == "all" or e["set"] == set_name]


def per_file(entries: List[Dict[str, Any]]) -> str:
    """One row per file."""
    lines = ["| File | Body | Light | ISO | Variant | Licence tier |", "|---|---|---|---|---|---|"]
    for e in entries:
        name = e["file"].rsplit("/", 1)[-1]
        variant = f"{e['variant']}, {e['bits']}-bit"
        lines.append(f"| {name} | {e['body']} | {e['light']} | {e['iso']} | {variant} | {TIER_TEXT.get(e['tier'], e['tier'])} |")
    return "\n".join(lines)


def summary(entries: List[Dict[str, Any]]) -> str:
    """One row per body: files per variant, lights and ISOs."""
    variants = sorted({e["variant"] for e in entries})
    by_body: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_body[e["body"]].append(e)
    lines = ["| Body | " + " | ".join(variants) + " | Lights | ISO |", "|---|" + "---|" * (len(variants) + 2)]
    for body in sorted(by_body, key=lambda b: (-len(by_body[b]), b)):
        rows = by_body[body]
        counts = Counter(r["variant"] for r in rows)
        lights = ", ".join(sorted({r["light"] for r in rows}))
        isos = ", ".join(str(i) for i in sorted({r["iso"] for r in rows}))
        lines.append(f"| {body} | " + " | ".join(str(counts.get(v, 0)) for v in variants) + f" | {lights} | {isos} |")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=str(HERE / "manifest.json"))
    parser.add_argument("--set", default="default", choices=["default", "full", "eval", "all"])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = select(manifest, args.set)
    if not entries:
        print(f"error: the set {args.set!r} has no entries", file=sys.stderr)
        return 1
    print(summary(entries) if args.summary else per_file(entries))
    print(f"\n{len(entries)} files in the {args.set} set. Tiers without entries yet: CC0 corpus, free licences (tools/fetch/golden.sh), Play Raw.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
