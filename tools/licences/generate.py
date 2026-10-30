#!/usr/bin/env python3
"""Writes THIRD-PARTY.md: the licence texts of everything that is compiled or bundled into Yellowmoon and is not ours.

    python3 tools/licences/generate.py                  # generate THIRD-PARTY.md at the repository root, then verify it
    python3 tools/licences/generate.py --out FILE
    python3 tools/licences/generate.py --verify FILE    # only check an existing file (a development stub fails)

NOTICE item 8 and the About dialog promise this file; the app bundle ships it next to NOTICE and LICENSE
(src-tauri/tauri.conf.json, ``bundle.resources``), and ``release.yml`` and the CI job ``licences`` run this script, so the
promise cannot rot. Sources, all read from the files cargo and npm already have:

* Rust crates: ``cargo about generate --format json`` (cargo-about, config tools/licences/about.toml) for the two programs
  that ship, the app (``src-tauri``) and the ``ym`` command (``crates/ym-cli``), for the macOS targets, normal
  dependencies only. The texts are the LICENSE files in the crates' own sources, so the copyright lines are right.
  Our own crates (no registry source) are not listed: LICENSE covers them.
* JavaScript packages: ``license-checker --production`` (the tool and the allow list of the CI check ``frontend`` ), the
  LICENSE files in each package folder (run ``npm ci`` first), and the build tools whose output is in the app
  (``BUILD_TOOL_OUTPUT``: Tailwind CSS).
* Fonts: the OFL text of Figtree (``src/styles/fonts/OFL.txt``), whose woff2 files Vite copies into the app.

Everything is sorted and carries no date, so the output depends only on Cargo.lock, package-lock.json and the tool
versions. Exit status: 0 written and verified, 1 a tool failed or the document is incomplete, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: The two programs that ship (docs/12): the app, and the sidecar that becomes Contents/MacOS/ym and the ``ym`` link.
MANIFESTS = ("src-tauri/Cargo.toml", "crates/ym-cli/Cargo.toml")
LICENSE_CHECKER = "license-checker@25.0.1"
FONT_DIR = REPO / "src" / "styles" / "fonts"

#: The file a development build gets (tools/dev/prepare-sidecar.sh copies it), so that tauri-build finds the bundle resource
#: before a release exists. Its first line is the marker, and ``verify`` refuses any file that has it.
STUB_FILE = HERE / "THIRD-PARTY.stub.md"
STUB_MARKER = "<!-- ym-third-party-stub -->"
STUB = STUB_FILE.read_text(encoding="utf-8")

# What a complete document must name. The crates are the ones every build links; ``react`` is the UI.
REQUIRED_CRATES = ("serde", "tauri", "wgpu", "jpeg-encoder")
REQUIRED_NPM = ("react", "react-dom")
MIN_CRATES = 200
MIN_NPM = 5


class GenerateError(Exception):
    """A tool failed or a package has no licence text. The message says which and what to do."""


@dataclass(frozen=True, order=True)
class Package:
    """A Rust crate as it is listed."""

    name: str
    version: str
    repository: str | None = field(default=None, compare=False)


@dataclass
class Group:
    """One licence text and the crates that carry it."""

    license_id: str
    text: str
    packages: list[Package]


@dataclass
class NpmPackage:
    """A JavaScript package with the text(s) of its LICENSE file(s)."""

    name: str
    version: str
    licenses: str
    repository: str | None
    texts: list[str]
    #: Set when the package folder has no LICENSE file and the texts come from somewhere else (see ``rust_counterpart``, ``HAND_ADDED_TEXTS``).
    note: str = ""


@dataclass
class Font:
    """A font family that is bundled, with its licence."""

    family: str
    files: list[str]
    text: str


def normalize(text: str) -> str:
    """Line endings and trailing blanks do not make two licence texts different."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip("\n")


def collect_rust(reports: list[dict]) -> list[Group]:
    """Merges the reports of ``cargo about generate --format json``: one group per distinct licence text."""
    groups: dict[tuple[str, str], dict[tuple[str, str], Package]] = defaultdict(dict)
    for report in reports:
        for lic in report.get("licenses", []):
            text = normalize(lic["text"])
            for used in lic.get("used_by", []):
                crate = used["crate"]
                if not crate.get("source"):
                    continue  # a path crate: ours
                package = Package(crate["name"], crate["version"], crate.get("repository"))
                groups[(lic["id"], text)][(package.name, package.version)] = package
    return [Group(lid, text, [pkgs[key] for key in sorted(pkgs)]) for (lid, text), pkgs in sorted(groups.items())]


def collect_notices(reports: list[dict]) -> dict[tuple[str, str], str]:
    """NOTICE files that crates ship (Apache-2.0 section 4(d) asks for them to travel with the binary)."""
    notices: dict[tuple[str, str], str] = {}
    for report in reports:
        for entry in report.get("crates", []):
            package = entry["package"]
            if not package.get("source") or not package.get("manifest_path"):
                continue
            folder = Path(package["manifest_path"]).parent
            for path in sorted(folder.glob("NOTICE*")):
                if path.is_file():
                    notices[(package["name"], package["version"])] = normalize(path.read_text(encoding="utf-8", errors="replace"))
                    break
    return notices


_LICENCE_FILE = re.compile(r"^(licen[cs]e|copying|unlicense)([-_. ].*)?$", re.IGNORECASE)


def rust_counterpart(npm_name: str) -> str | None:
    """The Rust crate of the same project for an npm package that ships no LICENSE file.

    The Tauri plugins publish only an SPDX tag file (``LICENSE.spdx``) to npm; their repository (tauri-apps/plugins-workspace)
    holds one MIT and one Apache-2.0 text for the JavaScript and the Rust side alike, and the Rust crate has them.
    """
    scope, _, name = npm_name.partition("/")
    if scope == "@tauri-apps" and name.startswith("plugin-"):
        return "tauri-plugin-" + name.removeprefix("plugin-")
    return None


#: npm packages whose tarball carries no LICENSE file although package.json declares the licence. The text is the LICENSE file
#: of the package's own repository, saved in ``tools/licences/texts/`` (the file name is the package name plus ``.LICENSE``).
#: Each entry is explicit on purpose: a new package without a licence file stays an error until somebody reads its licence.
HAND_ADDED_TEXTS = {
    "react-remove-scroll-bar": "https://github.com/theKashey/react-remove-scroll-bar",
}
TEXTS_DIR = HERE / "texts"


def hand_added_text(npm_name: str) -> tuple[str, str] | None:
    """The saved licence text of an npm package that ships none, and the note that says where it comes from."""
    repository = HAND_ADDED_TEXTS.get(npm_name)
    if repository is None:
        return None
    text = normalize((TEXTS_DIR / f"{npm_name}.LICENSE").read_text(encoding="utf-8"))
    note = (
        "The npm package carries no LICENSE file. This is the LICENSE file of its repository "
        f"({repository}), which publishes the package under the same licence."
    )
    return text, note


def collect_npm(checker: dict, rust: list[Group] | None = None) -> list[NpmPackage]:
    """Reads the output of ``license-checker --json``: every LICENSE file in each package folder.

    ``rust`` supplies the texts for packages without such a file (``rust_counterpart``); without a way to get a text the
    function raises, so that a gap is never silent.
    """
    packages = []
    for key in sorted(checker):
        info = checker[key]
        name, _, version = key.rpartition("@")
        folder = Path(info["path"])
        texts: list[str] = []
        for path in sorted(folder.iterdir()) if folder.is_dir() else []:
            if not path.is_file() or not _LICENCE_FILE.match(path.name) or path.suffix.lower() == ".spdx":
                continue
            text = normalize(path.read_text(encoding="utf-8", errors="replace"))
            if text and text not in texts:
                texts.append(text)
        note = ""
        saved = hand_added_text(name) if not texts else None
        if saved:
            texts, note = [saved[0]], saved[1]
        if not texts:
            crate = rust_counterpart(name)
            borrowed = [g.text for g in rust or [] if any(p.name == crate for p in g.packages)] if crate else []
            if not borrowed:
                raise GenerateError(f"{key} has no LICENSE file in {folder}: run `npm ci` first, or add its text by hand")
            texts = borrowed
            note = (
                f"The npm package carries no LICENSE file, only an SPDX tag. These are the texts of the Rust crate {crate}, "
                "published from the same repository under the same licences."
            )
        repository = info.get("repository")
        packages.append(NpmPackage(name, version, str(info.get("licenses", "")), repository, texts, note))
    return packages


#: Build tools whose output is part of the app although the tool is not: ``license-checker --production`` leaves them out, and their
#: licence asks for its notice in copies and substantial portions. Tailwind CSS compiles into the app's stylesheet (its preflight and
#: the utility classes we use), and the minified stylesheet carries no banner, so the notice travels here. Each entry is explicit.
BUILD_TOOL_OUTPUT = {
    "tailwindcss": "A build tool. Its preflight stylesheet and the utility classes it generates are part of the app's CSS.",
}


def collect_build_tool_output(modules: Path | None = None) -> list[NpmPackage]:
    """The ``BUILD_TOOL_OUTPUT`` packages, read from ``node_modules`` like the others (run ``npm ci`` first)."""
    modules = modules or REPO / "node_modules"
    found = []
    for name, note in sorted(BUILD_TOOL_OUTPUT.items()):
        folder = modules / name
        try:
            manifest = json.loads((folder / "package.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GenerateError(f"{name} is not installed in {modules}: run `npm ci` first ({error})") from error
        texts = []
        for path in sorted(folder.iterdir()):
            if path.is_file() and _LICENCE_FILE.match(path.name) and path.suffix.lower() != ".spdx":
                text = normalize(path.read_text(encoding="utf-8", errors="replace"))
                if text and text not in texts:
                    texts.append(text)
        if not texts:
            raise GenerateError(f"{name} has no LICENSE file in {folder}")
        repository = manifest.get("repository")
        url = repository.get("url") if isinstance(repository, dict) else repository
        found.append(NpmPackage(name, str(manifest.get("version", "")), str(manifest.get("license", "")), url, texts, note))
    return found


def collect_fonts() -> list[Font]:
    """The bundled font families and their OFL text."""
    ofl = FONT_DIR / "OFL.txt"
    files = sorted(p.name for p in FONT_DIR.glob("*.woff2"))
    if not ofl.is_file() or not files:
        raise GenerateError(f"{FONT_DIR} has no OFL.txt or no woff2 files")
    return [Font("Figtree", files, normalize(ofl.read_text(encoding="utf-8")))]


def fence(text: str) -> tuple[str, str]:
    """A code fence longer than any run of backticks inside the text."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    mark = "`" * max(3, longest + 1)
    return f"{mark}text", mark


def render(rust: list[Group], npm: list[NpmPackage], fonts: list[Font], notices: dict[tuple[str, str], str], tool: str) -> str:
    """The Markdown document."""
    unique = {(p.name, p.version) for g in rust for p in g.packages}
    out: list[str] = []
    add = out.append
    add("# Third-party licences\n")
    add(
        "Yellowmoon is licensed under the Apache License 2.0 (see `LICENSE`). It is built from the packages and the font "
        "below, each used under the licence whose text follows its name. The attributions for data, algorithms and "
        "ported code are in `NOTICE`.\n"
    )
    add(
        "This file is generated (`tools/licences/generate.py`) from `Cargo.lock` and `package-lock.json` with "
        f"{tool} and {LICENSE_CHECKER}. Do not edit it.\n"
    )
    add(f"Rust crates: {len(unique)} · JavaScript packages: {len(npm)} · Fonts: {len(fonts)}\n")

    add("## Rust crates\n")
    add(
        "Compiled into the app and into the `ym` command, for macOS. Crates that share a licence text are listed "
        "together under it.\n"
    )
    for group in rust:
        add(f"### {group.license_id}\n")
        for package in group.packages:
            link = f" ({package.repository})" if package.repository else ""
            add(f"- {package.name} {package.version}{link}")
        add("")
        opening, closing = fence(group.text)
        add(opening)
        add(group.text)
        add(closing + "\n")
    if notices:
        add("### Notices that crates ship\n")
        for (name, version), text in sorted(notices.items()):
            add(f"#### {name} {version}\n")
            opening, closing = fence(text)
            add(opening)
            add(text)
            add(closing + "\n")

    add("## JavaScript packages\n")
    add("Bundled into the app's interface.\n")
    for package in npm:
        add(f"### {package.name} {package.version}\n")
        add(f"Licence: {package.licenses}" + (f" · {package.repository}" if package.repository else "") + "\n")
        if package.note:
            add(package.note + "\n")
        for text in package.texts:
            opening, closing = fence(text)
            add(opening)
            add(text)
            add(closing + "\n")

    add("## Fonts\n")
    for font in fonts:
        add(f"### {font.family}\n")
        add(f"Files: {', '.join(font.files)}. Licence: SIL Open Font License 1.1.\n")
        opening, closing = fence(font.text)
        add(opening)
        add(font.text)
        add(closing + "\n")
    return "\n".join(out).rstrip("\n") + "\n"


def verify(text: str) -> list[str]:
    """What is wrong with a document, one sentence each; an empty list means it can ship."""
    if STUB_MARKER in text:
        return ["this is the development stub (prepare-sidecar.sh writes it); run tools/licences/generate.py"]
    problems = []
    counts = re.search(r"^Rust crates: (\d+) · JavaScript packages: (\d+) · Fonts: (\d+)$", text, re.MULTILINE)
    if not counts:
        return ["the summary line `Rust crates: N · JavaScript packages: M · Fonts: K` is missing: the file is empty or not ours"]
    crates, npm, fonts = (int(n) for n in counts.groups())
    if crates < MIN_CRATES:
        problems.append(f"only {crates} Rust crates are listed (at least {MIN_CRATES} are linked)")
    if npm < MIN_NPM:
        problems.append(f"only {npm} JavaScript packages are listed (at least {MIN_NPM} are bundled)")
    if fonts < 1 or "SIL OPEN FONT LICENSE Version 1.1" not in text:
        problems.append("the OFL text of the bundled font (Figtree) is missing")
    for name in REQUIRED_CRATES:
        if not re.search(rf"^- {re.escape(name)} \d", text, re.MULTILINE):
            problems.append(f"the Rust crate {name} is not listed")
    for name in REQUIRED_NPM:
        if not re.search(rf"^### {re.escape(name)} \d", text, re.MULTILINE):
            problems.append(f"the JavaScript package {name} is not listed")
    if "Redistributions in binary form must reproduce" not in text:
        problems.append("no BSD licence text is present (brotli and subtle, BSD-3-Clause, are linked)")
    return problems


def run(command: list[str], *, cwd: Path = REPO) -> str:
    """Runs a tool and returns its standard output."""
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise GenerateError(f"{command[0]} is not installed ({error})") from error
    if done.returncode != 0:
        raise GenerateError(f"`{' '.join(command)}` failed:\n{done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


def run_cargo_about(cargo_about: str, manifest: str, scratch: Path, *, offline: bool) -> dict:
    """``cargo about generate --format json`` for one manifest."""
    out = scratch / (Path(manifest).parent.name + ".json")
    command = [cargo_about, "generate", "--format", "json", "--locked", "--fail", "-c", str(HERE / "about.toml"), "-m", manifest, "-o", str(out)]
    if offline:
        command.append("--offline")
    run(command)
    return json.loads(out.read_text(encoding="utf-8"))


def fetch_sources() -> None:
    """Downloads every crate of Cargo.lock, so that cargo-about reads their LICENSE files offline.

    Offline matters twice: ``cargo metadata`` needs the manifest of every package, whatever the platform, and online
    cargo-about asks a web service for crates without a LICENSE file, which would make the output depend on that service.
    """
    run(["cargo", "fetch", "--locked"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(REPO / "THIRD-PARTY.md"), help="where to write (default: THIRD-PARTY.md at the repository root)")
    parser.add_argument("--verify", metavar="FILE", help="only verify FILE")
    parser.add_argument("--cargo-about", default="cargo-about", help="the cargo-about executable (0.9 or newer)")
    parser.add_argument("--no-fetch", action="store_true", help="do not run `cargo fetch` first (the sources are already downloaded)")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify(Path(args.verify).read_text(encoding="utf-8"))
        for problem in problems:
            print(f"error: {args.verify}: {problem}", file=sys.stderr)
        print(f"{args.verify}: {'incomplete' if problems else 'complete'}")
        return 1 if problems else 0

    try:
        tool = run([args.cargo_about, "--version"]).strip()
        if not args.no_fetch:
            fetch_sources()
        with tempfile.TemporaryDirectory() as scratch:
            reports = [run_cargo_about(args.cargo_about, manifest, Path(scratch), offline=True) for manifest in MANIFESTS]
        checker = json.loads(
            run_to_file(["npx", "--yes", LICENSE_CHECKER, "--production", "--excludePrivatePackages", "--json"])
        )
        rust = collect_rust(reports)
        npm = sorted(collect_npm(checker, rust) + collect_build_tool_output(), key=lambda p: (p.name, p.version))
        document = render(rust, npm, collect_fonts(), collect_notices(reports), tool)
    except GenerateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    problems = verify(document)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.write_text(document, encoding="utf-8")
    counts = re.search(r"^Rust crates: .*$", document, re.MULTILINE)
    print(f"wrote {out} ({len(document) // 1024} KiB): {counts.group(0) if counts else ''}")
    return 0


def run_to_file(command: list[str]) -> str:
    """``license-checker --json`` prints the JSON to standard output; this keeps it apart from npx's own messages."""
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "npm.json"
        run(command + ["--out", str(target)])
        return target.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
