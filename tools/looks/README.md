# tools/looks

Developer and HUMAN tooling for the look library (`docs/08-looks.md`). Python 3 standard library only.

## `fingerprint.py` (HUMAN M3-H2)

Teaches the app to recognise ARRI's five free look packages exactly. Run it once on this Mac after downloading the packages by hand from ARRI's Look Files page:

```
python3 -I tools/looks/fingerprint.py ~/Downloads/<package>.zip [<more zips>…]
```

For every `.cube` file in a zip it records the path inside the zip, the size, the LUT size (`LUT_3D_SIZE`) and the SHA-256 in `data/looks/arri-packages.json`, under the matching package's `fingerprints`, with a short `observed` summary (zip name, counts, folder names). It never reads LUT data into its output, and it ignores `__MACOSX`, `._` files, `.alf4`, `.alf2` and PDFs like the importer does. The package is recognised from the zip name and contents using the patterns already in the file; if that is not unique, add `--package <id>` (for example `look-library-logc4`).

Other options: `--dry-run` shows what would be written, `--packages <file>` works on another copy, `--reformat` rewrites the list in the tool's layout.

Then read the `note:` lines it prints (differences from the patterns), and check `git diff data/looks/arri-packages.json`: the zip names under `observed` tell you how to tighten the `zip_name` patterns. Commit the file. It contains names, sizes and hashes only.

Tests: `python3 -I tools/looks/test_fingerprint.py` (synthetic zips only).
