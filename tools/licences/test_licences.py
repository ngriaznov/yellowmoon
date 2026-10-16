"""Tests of the THIRD-PARTY generator and the bundle check (stdlib only:
``python3 -I -m unittest discover -s tools/licences``).

Nothing here runs cargo-about, npm or cargo: the functions under test take the JSON those tools print, and the real
run is the CI job ``licences`` (``python3 tools/licences/generate.py``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


generate = load("generate")
check_bundle = load("check_bundle")

MIT = "Copyright (c) 2020 Someone\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"
BSD3 = (
    "Copyright (c) 2016 Dropbox, Inc.\nAll rights reserved.\n\nRedistribution and use in source and binary forms,"
    "\nRedistributions in binary form must reproduce the above copyright notice, this list of conditions\n"
)
APACHE = "                                 Apache License\n                           Version 2.0, January 2004\n"
BSD_NOTICE = (
    "Copyright 2015 Colour Developers\nRedistributions in binary form must reproduce the above copyright notice\n"
    "Neither the name of the copyright holder nor the names of its contributors\n"
    "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"\n"
    "Copyright (c) 2026 Tim Mikeladze\n"
    "The above copyright notice and this permission notice shall be included in all\n"
)
OFL = "Copyright 2022 The Figtree Project Authors\n\nSIL OPEN FONT LICENSE Version 1.1 - 26 February 2007\n"


def crate(name: str, version: str = "1.0.0", source: str | None = "registry+https://github.com/rust-lang/crates.io-index",
          repository: str | None = None) -> dict:
    return {"name": name, "version": version, "source": source, "repository": repository, "manifest_path": "/nowhere/Cargo.toml"}


def report(*entries: tuple[str, str, list[dict]]) -> dict:
    """What `cargo about generate --format json` prints: licences with the crates that use each text."""
    return {
        "licenses": [
            {"id": lid, "name": lid, "text": text, "used_by": [{"crate": c, "path": None} for c in crates]}
            for lid, text, crates in entries
        ],
        "crates": [],
    }


class Rust(unittest.TestCase):
    def test_the_same_text_from_two_reports_is_one_group_with_the_union_of_crates(self):
        app = report(("MIT", MIT, [crate("serde"), crate("anyhow")]))
        cli = report(("MIT", MIT, [crate("serde"), crate("clap", "4.6.7")]))
        groups = generate.collect_rust([app, cli])
        self.assertEqual(len(groups), 1)
        self.assertEqual([(p.name, p.version) for p in groups[0].packages], [("anyhow", "1.0.0"), ("clap", "4.6.7"), ("serde", "1.0.0")])

    def test_different_texts_of_one_licence_stay_apart_and_trailing_blanks_do_not_matter(self):
        a = report(("MIT", MIT, [crate("a")]), ("MIT", MIT.replace("Someone", "Else"), [crate("b")]))
        b = report(("MIT", MIT.replace("\n", "  \r\n") + "\n\n", [crate("c")]))
        groups = generate.collect_rust([a, b])
        self.assertEqual(sorted(len(g.packages) for g in groups), [1, 2])

    def test_our_own_crates_are_not_listed(self):
        groups = generate.collect_rust([report(("Apache-2.0", APACHE, [crate("ym-core", source=None), crate("rayon")]))])
        self.assertEqual([p.name for g in groups for p in g.packages], ["rayon"])

    def test_notice_files_of_the_crates_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "with").mkdir()
            (root / "with" / "Cargo.toml").write_text("")
            (root / "with" / "NOTICE").write_text("Attribution line\n")
            (root / "without").mkdir()
            (root / "without" / "Cargo.toml").write_text("")
            rep = {
                "licenses": [],
                "crates": [
                    {"package": {"name": "with", "version": "1.0.0", "source": "registry+x", "manifest_path": str(root / "with" / "Cargo.toml")}},
                    {"package": {"name": "without", "version": "1.0.0", "source": "registry+x", "manifest_path": str(root / "without" / "Cargo.toml")}},
                ],
            }
            notices = generate.collect_notices([rep])
        self.assertEqual(notices, {("with", "1.0.0"): "Attribution line"})


class Npm(unittest.TestCase):
    def test_every_licence_file_of_the_package_is_read_and_the_spdx_tag_file_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "node_modules" / "@scope" / "api"
            pkg.mkdir(parents=True)
            (pkg / "LICENSE_MIT").write_text(MIT)
            (pkg / "LICENSE_APACHE-2.0").write_text(APACHE)
            (pkg / "LICENSE.spdx").write_text("SPDX-License-Identifier: MIT OR Apache-2.0\n")
            (pkg / "README.md").write_text("not a licence")
            checker = {"@scope/api@2.1.0": {"licenses": "Apache-2.0 OR MIT", "repository": "https://github.com/x/y", "path": str(pkg)}}
            packages = generate.collect_npm(checker)
        self.assertEqual(len(packages), 1)
        self.assertEqual((packages[0].name, packages[0].version, packages[0].licenses), ("@scope/api", "2.1.0", "Apache-2.0 OR MIT"))
        self.assertEqual(sorted(packages[0].texts), sorted([generate.normalize(APACHE), generate.normalize(MIT)]))

    def test_a_tauri_plugin_without_licence_files_gets_the_texts_of_its_rust_crate(self):
        rust = generate.collect_rust([report(("MIT", MIT, [crate("tauri-plugin-dialog", "2.8.0")]), ("Apache-2.0", APACHE, [crate("tauri-plugin-dialog", "2.8.0"), crate("other")]), ("ISC", "ISC text", [crate("unrelated")]))])
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "plugin-dialog"
            pkg.mkdir()
            (pkg / "LICENSE.spdx").write_text("SPDX-License-Identifier: MIT\n")
            checker = {"@tauri-apps/plugin-dialog@2.8.2": {"licenses": "MIT OR Apache-2.0", "path": str(pkg)}}
            packages = generate.collect_npm(checker, rust)
        self.assertEqual(sorted(packages[0].texts), sorted([generate.normalize(APACHE), generate.normalize(MIT)]))
        self.assertIn("tauri-plugin-dialog", packages[0].note)

    def test_the_counterpart_rule_is_narrow(self):
        self.assertEqual(generate.rust_counterpart("@tauri-apps/plugin-updater"), "tauri-plugin-updater")
        self.assertIsNone(generate.rust_counterpart("react"))
        self.assertIsNone(generate.rust_counterpart("@tauri-apps/api"))

    def test_a_package_that_ships_no_licence_file_gets_its_saved_text_and_a_note_that_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            checker = {"react-remove-scroll-bar@2.3.8": {"licenses": "MIT", "path": tmp}}
            packages = generate.collect_npm(checker)
        self.assertEqual(len(packages[0].texts), 1)
        self.assertIn("Permission is hereby granted", packages[0].texts[0])
        self.assertIn("Anton Korzunov", packages[0].texts[0])
        self.assertIn("carries no LICENSE file", packages[0].note)

    def test_the_saved_texts_are_the_only_ones_and_each_names_a_file_that_exists(self):
        for name in generate.HAND_ADDED_TEXTS:
            self.assertTrue((generate.TEXTS_DIR / f"{name}.LICENSE").is_file(), name)
        self.assertEqual(sorted(p.name for p in generate.TEXTS_DIR.iterdir()), sorted(f"{n}.LICENSE" for n in generate.HAND_ADDED_TEXTS))
        self.assertIsNone(generate.hand_added_text("react"))

    def test_a_package_without_a_licence_file_is_an_error_not_a_silent_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            checker = {"bare@1.0.0": {"licenses": "MIT", "path": tmp}}
            with self.assertRaisesRegex(generate.GenerateError, "bare@1.0.0"):
                generate.collect_npm(checker)


class BuildToolOutput(unittest.TestCase):
    def test_tailwind_is_listed_with_its_licence_and_a_note_about_why(self):
        packages = generate.collect_build_tool_output()
        self.assertEqual([p.name for p in packages], ["tailwindcss"])
        self.assertIn("Tailwind Labs", packages[0].texts[0])
        self.assertIn("Permission is hereby granted", packages[0].texts[0])
        self.assertIn("stylesheet", packages[0].note)
        self.assertEqual(packages[0].licenses, "MIT")

    def test_a_missing_package_is_an_error_not_a_silent_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(generate.GenerateError, "tailwindcss is not installed"):
                generate.collect_build_tool_output(Path(tmp))

    def test_a_package_without_a_licence_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "tailwindcss"
            folder.mkdir()
            (folder / "package.json").write_text(json.dumps({"version": "4.0.0", "license": "MIT"}))
            with self.assertRaisesRegex(generate.GenerateError, "no LICENSE file"):
                generate.collect_build_tool_output(Path(tmp))


class Rendering(unittest.TestCase):
    def sample(self) -> str:
        rust = generate.collect_rust(
            [report(("MIT", MIT, [crate("serde", repository="https://github.com/serde-rs/serde")]), ("BSD-3-Clause", BSD3, [crate("brotli", "9.0.0")]))]
        )
        npm = [generate.NpmPackage("react", "19.3.0", "MIT", "https://github.com/facebook/react", [MIT.strip()])]
        return generate.render(rust, npm, [generate.Font("Figtree", ["figtree-latin-wght-normal.woff2"], OFL.strip())], {}, "cargo-about 0.9.2")

    def test_the_document_names_every_package_and_carries_every_text(self):
        text = self.sample()
        for needle in ("## Rust crates", "## JavaScript packages", "## Fonts", "serde 1.0.0", "brotli 9.0.0", "react 19.3.0",
                       "https://github.com/serde-rs/serde", "Redistributions in binary form must reproduce", "SIL OPEN FONT LICENSE Version 1.1",
                       "Permission is hereby granted", "cargo-about 0.9.2"):
            self.assertIn(needle, text)

    def test_the_output_is_deterministic(self):
        self.assertEqual(self.sample(), self.sample())

    def test_a_text_with_backticks_does_not_close_its_fence(self):
        rust = generate.collect_rust([report(("MIT", MIT + "\n````\nodd\n", [crate("a")]))])
        text = generate.render(rust, [], [], {}, "cargo-about test")
        self.assertIn("`````", text)

    def test_notices_are_listed_under_their_crate(self):
        rust = generate.collect_rust([report(("Apache-2.0", APACHE, [crate("a")]))])
        text = generate.render(rust, [], [], {("a", "1.0.0"): "Attribution line"}, "cargo-about test")
        self.assertIn("Attribution line", text)


class Verification(unittest.TestCase):
    def good(self) -> str:
        rust_entries = [("MIT", MIT.replace("Someone", f"Person {i}"), [crate(f"crate{i}")]) for i in range(300)]
        rust_entries += [("MIT", MIT, [crate(n) for n in ("serde", "tauri", "wgpu", "jpeg-encoder")]), ("BSD-3-Clause", BSD3, [crate("brotli")])]
        rust = generate.collect_rust([report(*rust_entries)])
        npm = [generate.NpmPackage(n, "1.0.0", "MIT", None, [MIT.strip()]) for n in ("react", "react-dom", "scheduler", "@tauri-apps/api", "a", "b")]
        return generate.render(rust, npm, [generate.Font("Figtree", ["f.woff2"], OFL.strip())], {}, "cargo-about test")

    def test_a_complete_document_passes(self):
        self.assertEqual(generate.verify(self.good()), [])

    def test_the_stub_file_starts_with_the_marker_verify_looks_for(self):
        self.assertTrue((HERE / "THIRD-PARTY.stub.md").read_text().startswith(generate.STUB_MARKER))

    def test_the_development_stub_is_refused(self):
        self.assertTrue(generate.verify(generate.STUB))
        self.assertTrue(any("stub" in p for p in generate.verify(generate.STUB)))

    def test_an_empty_or_short_document_is_refused(self):
        self.assertTrue(generate.verify(""))
        self.assertTrue(generate.verify("# Third-party licences\n"))

    def test_each_missing_part_is_named(self):
        full = self.good()
        for needle, expect in (("serde 1.0.0", "serde"), ("react 1.0.0", "react"), ("SIL OPEN FONT LICENSE", "OFL")):
            problems = generate.verify(full.replace(needle, "x"))
            self.assertTrue(any(expect in p for p in problems), (needle, problems))


class Config(unittest.TestCase):
    def test_cargo_about_accepts_what_cargo_deny_allows(self):
        about = tomllib.loads((HERE / "about.toml").read_text())
        deny = tomllib.loads((REPO / "deny.toml").read_text())
        self.assertEqual(sorted(about["accepted"]), sorted(deny["licenses"]["allow"]))

    def test_only_the_macos_targets_are_scanned(self):
        about = tomllib.loads((HERE / "about.toml").read_text())
        self.assertEqual(sorted(about["targets"]), ["aarch64-apple-darwin", "x86_64-apple-darwin"])


class Bundle(unittest.TestCase):
    def tree(self, root: Path, *, resources: dict | None = None, notice: str | None = None, ofl: bool = True) -> None:
        (root / "src-tauri").mkdir(parents=True, exist_ok=True)
        resources = {"../NOTICE": "NOTICE", "../LICENSE": "LICENSE", "../THIRD-PARTY.md": "THIRD-PARTY.md"} if resources is None else resources
        (root / "src-tauri" / "tauri.conf.json").write_text(json.dumps({"bundle": {"resources": resources}}))
        (root / "NOTICE").write_text(notice if notice is not None else BSD_NOTICE)
        (root / "LICENSE").write_text("Apache License")

    def test_a_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp))
            self.assertEqual(check_bundle.problems(Path(tmp)), [])

    def test_third_party_must_be_in_the_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp), resources={"../NOTICE": "NOTICE", "../LICENSE": "LICENSE"})
            problems = check_bundle.problems(Path(tmp))
        self.assertTrue(any("THIRD-PARTY.md" in p for p in problems), problems)

    def test_the_bsd_conditions_must_be_in_the_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp), notice="BSD 3-Clause, Copyright 2015 Colour Developers.\n")
            problems = check_bundle.problems(Path(tmp))
        self.assertTrue(any("Redistributions in binary form" in p for p in problems), problems)

    def test_the_glasscn_notice_must_be_in_the_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tree(Path(tmp), notice=BSD_NOTICE.replace("Tim Mikeladze", "Someone"))
            problems = check_bundle.problems(Path(tmp))
        self.assertTrue(any("Tim Mikeladze" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
