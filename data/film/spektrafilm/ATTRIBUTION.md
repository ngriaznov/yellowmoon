# spektrafilm profiles

```
spektrafilm by Andrea Volpato
https://github.com/andreavolpato/spektrafilm
Licensed CC BY-SA 4.0 (the text is in LICENSE, in this folder).

Taken from: commit 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc
of the main branch, 2026-06-07 ("fix: vlog midgray exposure"). Profile version 0.3.2.
Files: 28 profiles in profiles/, not modified. SHA-256 in SHA256SUMS.
Modified by: nobody.
The profiles were made by the author from Kodak and Fujifilm data sheets and other
published material. The original data belong to their holders.
Yellowmoon is not affiliated with Andrea Volpato, Kodak or Fujifilm.
```

- Source: `src/spektrafilm/data/profiles/*.json` at that commit. `LICENSE` is `SPEKTRAFILM_LICENSE.txt` at the same commit, byte for byte.
- The files are plain data, read at run time by the Film node. They are never compiled into the program, packed, encrypted or changed.
- Each profile also carries its own notice in `metadata.license` and `metadata.copyright`.
- If a profile is ever modified, the change is described in `CHANGELOG.txt` beside this file, with who made it, and this file says so.
- Written by `tools/film/update-profiles.sh`; do not edit by hand.
