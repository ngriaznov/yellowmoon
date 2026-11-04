#!/usr/bin/env python3
"""Token-shingle and structural provenance check (docs/24 section 3.1 rule 5, section 8 *Provenance*).

    python3 -I tools/provenance/shingles.py --ours crates/ym-film/src \
        --other spektrafilm=PATH[,PATH...] --other spektrafilm-rs=PATH --other vkdt-filmsim=PATH \
        [--k 8] [--out provenance.md]

Compares every source file of ``--ours`` (Rust) with every source file of each ``--other`` tree (Python, Rust, GLSL or C)
and writes a report. The program only counts: it tokenises each file, normalises it and compares sets of hashes. It never
prints a line of the other project's source, only normalised token sequences with file names and line numbers, and the
clones stay outside the repository.

Normalisation: comments and whitespace are dropped; every identifier that is not a keyword becomes ``ID``; every number
``NUM``; every string ``STR``. Keywords, operators and brackets stay. A *shingle* is ``k`` consecutive tokens (8 by default).

* **Token overlap** of a file pair: the Jaccard index of the two files' shingle sets, and the number of shared shingles.
* **Sequences**: runs of consecutive shared shingles, merged to one sequence each; the longest are listed. A run is
  *trivial* when its tokens use three or fewer different kinds of token, or when its shingles also occur in three or more
  files of our own tree (boilerplate such as ``for ID in NUM .. ID {``), or when it is shorter than 24 tokens.
* **Structure**: the same comparison on the *skeleton* of each file: only keywords and brackets, with ``k`` = 16, which
  catches code that was renamed and re-valued but kept its shape. A baseline is reported with it: the same comparison
  between our tree and an unrelated Rust tree (``--baseline``), so a reader can tell chance from copying.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Words that stay as they are, per family of languages. Everything else that looks like a name becomes ID.
KEYWORDS = {
    "rust": set(
        "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut "
        "pub ref return self Self static struct super trait true type unsafe use where while".split()
    ),
    "python": set(
        "and as assert async await break class continue def del elif else except False finally for from global if import in is "
        "lambda None nonlocal not or pass raise return True try while with yield".split()
    ),
    "c": set(
        "break case const continue default do else enum for if in inout out return struct switch uniform void while "
        "float int uint bool vec2 vec3 vec4 mat3 mat4 layout".split()
    ),
}

SUFFIX_FAMILY = {".rs": "rust", ".py": "python", ".glsl": "c", ".comp": "c", ".c": "c", ".h": "c", ".frag": "c", ".vert": "c"}

TOKEN_PATTERN = r"""
    (?P<ws>\s+)
  | (?P<lcomment>LCOMMENT[^\n]*)
  | (?P<bcomment>/\*.*?\*/)
  | (?P<tqstring>(?:[rRbBfF]{0,2})(?:\"\"\".*?\"\"\"|'''.*?'''))
  | (?P<rawstring>r\#*".*?"\#*)
  | (?P<string>(?:[bBfF]?)"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
  | (?P<number>0[xX][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?(?:f32|f64|u8|u16|u32|u64|usize|i32|i64|[fF])?|\.\d+(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>->|=>|::|//|\.\.=|\.\.|<<=|>>=|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=|\*\*|<<|>>|[-+*/%=<>!&|^~?:;,.(){}\[\]@])
  | (?P<other>.)
    """
#: Line comments start with '#' in Python and with '//' elsewhere (a '#' in Rust or C is an attribute or a directive).
TOKEN_RE = {
    "python": re.compile(TOKEN_PATTERN.replace("LCOMMENT", r"\#"), re.VERBOSE | re.DOTALL),
    "rust": re.compile(TOKEN_PATTERN.replace("LCOMMENT", "//"), re.VERBOSE | re.DOTALL),
    "c": re.compile(TOKEN_PATTERN.replace("LCOMMENT", "//"), re.VERBOSE | re.DOTALL),
}


def tokenize(text: str, family: str) -> list[tuple[str, int]]:
    """Normalised tokens of a source text, each with its 1-based line number."""
    keywords = KEYWORDS[family]
    out: list[tuple[str, int]] = []
    line = 1
    for match in TOKEN_RE[family].finditer(text):
        kind = match.lastgroup
        value = match.group()
        start_line = line
        line += value.count("\n")
        if kind in ("ws", "bcomment"):
            continue
        if kind == "lcomment":
            continue
        if kind in ("string", "tqstring", "rawstring"):
            if family == "python" and kind == "tqstring":
                continue  # docstrings are prose
            out.append(("STR", start_line))
        elif kind == "number":
            out.append(("NUM", start_line))
        elif kind == "name":
            out.append((value if value in keywords else "ID", start_line))
        else:
            out.append((value, start_line))
    return out


def skeleton(tokens: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Only keywords and brackets: the shape of the code with every name, value and operator gone."""
    shape = {"(", ")", "{", "}", "[", "]", ";"}
    keep = set().union(*KEYWORDS.values())
    return [(t, n) for t, n in tokens if t in shape or t in keep]


def shingles(tokens: list[tuple[str, int]], k: int) -> list[int]:
    """The hash of each window of ``k`` tokens, in order (the list's length is ``len(tokens) - k + 1``)."""
    texts = [t for t, _ in tokens]
    return [
        int.from_bytes(hashlib.blake2b(" ".join(texts[i : i + k]).encode(), digest_size=8).digest(), "big")
        for i in range(max(0, len(texts) - k + 1))
    ]


def source_files(roots: list[Path]) -> dict[str, Path]:
    """Source files under the roots, by a display name (root name and relative path)."""
    found: dict[str, Path] = {}
    for root in roots:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            if path.suffix in SUFFIX_FAMILY and ".git" not in path.parts and "target" not in path.parts:
                base = root.parent if root.is_file() else root
                found[f"{root.name}/{path.relative_to(base)}" if root.is_dir() else path.name] = path
    return found


class Unit:
    """One file, tokenised once."""

    def __init__(self, name: str, path: Path, k: int, k_shape: int):
        self.name = name
        self.family = SUFFIX_FAMILY[path.suffix]
        self.tokens = tokenize(path.read_text(encoding="utf-8", errors="replace"), self.family)
        self.hashes = shingles(self.tokens, k)
        shape = skeleton(self.tokens)
        self.shape_tokens = shape
        self.shape_hashes = shingles(shape, k_shape)
        self.set = set(self.hashes)
        self.shape_set = set(self.shape_hashes)


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard index of two sets (0 when both are empty)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def sequences(ours: Unit, theirs: Unit, k: int, common: Counter, min_tokens: int = 24):
    """Runs of consecutive shared shingles of ``ours`` (merged), with the other file's line of the first match."""
    shared = ours.set & theirs.set
    first_line: dict[int, int] = {}
    for i, h in enumerate(theirs.hashes):
        if h in shared and h not in first_line:
            first_line[h] = theirs.tokens[i][1]
    runs = []
    i = 0
    n = len(ours.hashes)
    while i < n:
        if ours.hashes[i] not in shared:
            i += 1
            continue
        j = i
        while j + 1 < n and ours.hashes[j + 1] in shared:
            j += 1
        tokens = [t for t, _ in ours.tokens[i : j + k]]
        distinct = len(set(tokens))
        boiler = all(common[ours.hashes[x]] >= 3 for x in range(i, j + 1))
        trivial = len(tokens) < min_tokens or distinct <= 3 or boiler
        runs.append(
            {
                "tokens": tokens,
                "ours_lines": (ours.tokens[i][1], ours.tokens[min(j + k - 1, len(ours.tokens) - 1)][1]),
                "theirs_line": first_line.get(ours.hashes[i], 0),
                "trivial": trivial,
                "distinct": distinct,
            }
        )
        i = j + 1
    return runs


def compare(ours: dict[str, Unit], theirs: dict[str, Unit], k: int):
    """For every file pair with a shared shingle: the Jaccard index and the shared count, and the shape's."""
    common: Counter = Counter()
    for unit in ours.values():
        for h in unit.set:
            common[h] += 1
    pairs = []
    index: dict[int, list[str]] = defaultdict(list)
    for name, unit in theirs.items():
        for h in unit.set:
            index[h].append(name)
    shape_index: dict[int, list[str]] = defaultdict(list)
    for name, unit in theirs.items():
        for h in unit.shape_set:
            shape_index[h].append(name)
    for oname, o in ours.items():
        candidates = {n for h in o.set for n in index.get(h, ())} | {n for h in o.shape_set for n in shape_index.get(h, ())}
        for tname in candidates:
            t = theirs[tname]
            shared = o.set & t.set
            shared_shape = o.shape_set & t.shape_set
            if not shared and not shared_shape:
                continue
            pairs.append(
                {
                    "ours": oname,
                    "theirs": tname,
                    "jaccard": jaccard(o.set, t.set),
                    "shared": len(shared),
                    "shape_jaccard": jaccard(o.shape_set, t.shape_set),
                    "shape_shared": len(shared_shape),
                    "runs": sequences(o, t, k, common) if shared else [],
                }
            )
    return pairs


def corpus_overlap(ours: dict[str, Unit], theirs: dict[str, Unit]) -> dict[str, float]:
    """Whole-tree figures: the share of our shingles that occur anywhere in theirs, for tokens and for the shape."""
    mine = set().union(*(u.set for u in ours.values())) if ours else set()
    other = set().union(*(u.set for u in theirs.values())) if theirs else set()
    mine_shape = set().union(*(u.shape_set for u in ours.values())) if ours else set()
    other_shape = set().union(*(u.shape_set for u in theirs.values())) if theirs else set()
    return {
        "ours_shingles": len(mine),
        "token_share": len(mine & other) / len(mine) if mine else 0.0,
        "shape_share": len(mine_shape & other_shape) / len(mine_shape) if mine_shape else 0.0,
        "jaccard": jaccard(mine, other),
    }


def render(results, commits: dict[str, str], k: int, k_shape: int, baseline, top: int) -> str:
    """The Markdown report."""
    lines = ["# Provenance check: crates/ym-film/src against spektrafilm, spektrafilm-rs and vkdt filmsim", ""]
    lines.append(f"Token shingles of {k} (identifiers, numbers and strings normalised; comments dropped); skeleton shingles of {k_shape} (keywords and brackets only).")
    lines.append("Commits: " + "; ".join(f"{name} `{sha}`" for name, sha in commits.items()) + ".")
    lines.append("")
    lines.append("| Tree | our shingles | share found in the tree | skeleton share found | tree-wide Jaccard | file pairs with a shared shingle | best file-pair Jaccard | non-trivial sequences, 24+ tokens | 48+ tokens | 96+ tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    def row(label, corpus, pairs):
        best = max((p["jaccard"] for p in pairs), default=0.0)
        counts = [sum(1 for p in pairs for r in p["runs"] if not r["trivial"] and len(r["tokens"]) >= n) for n in (24, 48, 96)]
        return (
            f"| {label} | {corpus['ours_shingles']} | {corpus['token_share']:.4f} | {corpus['shape_share']:.4f} | {corpus['jaccard']:.5f} | "
            f"{len(pairs)} | {best:.4f} | {counts[0]} | {counts[1]} | {counts[2]} |"
        )

    for name, (corpus, pairs) in results.items():
        lines.append(row(name, corpus, pairs))
    if baseline:
        name, corpus, pairs = baseline
        lines.append(row(f"baseline: {name} (unrelated, the chance level)", corpus, pairs))
    lines.append("")
    for name, (_, pairs) in results.items():
        lines.append(f"## {name}")
        lines.append("")
        ranked = sorted(pairs, key=lambda p: (-p["jaccard"], -p["shape_jaccard"]))[:top]
        lines.append("Top file pairs by token Jaccard (ours / theirs / Jaccard / shared shingles / skeleton Jaccard / skeleton shared):")
        lines.append("")
        lines.append("| ours | theirs | Jaccard | shared | skeleton Jaccard | skeleton shared |")
        lines.append("|---|---|---|---|---|---|")
        for p in ranked:
            lines.append(f"| {p['ours']} | {p['theirs']} | {p['jaccard']:.4f} | {p['shared']} | {p['shape_jaccard']:.4f} | {p['shape_shared']} |")
        lines.append("")
        best_by_place: dict[tuple, tuple] = {}
        for p in pairs:
            for r in p["runs"]:
                key = (p["ours"], r["ours_lines"])
                if key not in best_by_place or len(r["tokens"]) > len(best_by_place[key][1]["tokens"]):
                    best_by_place[key] = (p, r)
        runs = sorted(best_by_place.values(), key=lambda pr: (pr[1]["trivial"], -len(pr[1]["tokens"])))[:top]
        lines.append("Longest shared sequences (normalised tokens; non-trivial first):")
        lines.append("")
        if not runs:
            lines.append("None.")
        for p, r in runs:
            flag = "trivial" if r["trivial"] else "NON-TRIVIAL"
            text = " ".join(r["tokens"][:60]) + (" ..." if len(r["tokens"]) > 60 else "")
            lines.append(
                f"* {flag}, {len(r['tokens'])} tokens, {p['ours']} lines {r['ours_lines'][0]}-{r['ours_lines'][1]} / {p['theirs']} line {r['theirs_line']}: `{text}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def load(roots: list[Path], k: int, k_shape: int) -> dict[str, Unit]:
    """Tokenise every source file under the roots."""
    return {name: Unit(name, path, k, k_shape) for name, path in source_files(roots).items()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", type=Path, nargs="+", required=True)
    parser.add_argument("--other", action="append", required=True, help="NAME=PATH[,PATH...] (repeat)")
    parser.add_argument("--commit", action="append", default=[], help="NAME=SHA, recorded in the report")
    parser.add_argument("--baseline", help="NAME=PATH: an unrelated tree for the chance level")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--k-shape", type=int, default=16)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    ours = load(args.ours, args.k, args.k_shape)
    if not ours:
        print("error: no source files under --ours", file=sys.stderr)
        return 2
    results = {}
    for spec in args.other:
        name, _, paths = spec.partition("=")
        theirs = load([Path(p) for p in paths.split(",")], args.k, args.k_shape)
        if not theirs:
            print(f"error: no source files in {name}: {paths}", file=sys.stderr)
            return 2
        results[name] = (corpus_overlap(ours, theirs), compare(ours, theirs, args.k))
    baseline = None
    if args.baseline:
        name, _, path = args.baseline.partition("=")
        base = load([Path(path)], args.k, args.k_shape)
        baseline = (name, corpus_overlap(ours, base), compare(ours, base, args.k))
    commits = dict(c.split("=", 1) for c in args.commit)
    report = render(results, commits, args.k, args.k_shape, baseline, args.top)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
