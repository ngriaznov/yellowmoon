# tools/looks

Developer and HUMAN tooling for the look library (`docs/08-looks.md`). Python 3 standard library only.

## `fingerprint.py` (HUMAN M3-H2)

Teaches the app to recognise ARRI's five free look packages exactly: the three 3D-LUT packages (`.cube`) and the two Look Files packages (`.alf4` for LogC4, `.aml` for LogC3; 0.1.3). Run it once on this Mac after downloading the packages by hand from ARRI's Look Files page:

```
python3 -I tools/looks/fingerprint.py ~/Downloads/<package>.zip [<more zips>…]
```

For every look file in a zip (`.cube`, `.alf4` or `.aml`; a package holds one kind) it records the path inside the zip, the size, the LUT size (`LUT_3D_SIZE` of a `.cube`, the LUT size field in the header of a look file) and the SHA-256 in `data/looks/arri-packages.json`, under the matching package's `fingerprints`, with a short `observed` summary (zip name, counts, folder names). It never reads LUT data into its output, and it ignores `__MACOSX`, `._` files, `.xml` (ALF-1) and PDFs like the importer does. `.alf4c` is never part of a package, so the tool skips it (the importer reads it as one of the person's own look files). A zip that holds more than one kind of look file (`.cube`, `.alf4`, `.aml`) is refused: fingerprint one package at a time. It reads only the header of a look file for the LUT size and checks no checksum: reading the look is the app's job (`ym_color::alf`). The package is recognised from the zip name and contents using the patterns already in the file; if that is not unique, add `--package <id>` (for example `look-library-logc4`).

Other options: `--dry-run` shows what would be written, `--packages <file>` works on another copy, `--reformat` rewrites the list in the tool's layout.

Then read the `note:` lines it prints (differences from the patterns), and check `git diff data/looks/arri-packages.json`: the zip names under `observed` tell you how to tighten the `zip_name` patterns. Commit the file. It contains names, sizes and hashes only.

Tests: `python3 -I tools/looks/test_fingerprint.py` (synthetic zips only).
