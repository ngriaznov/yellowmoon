# Test photo sources (A1 and siblings)

Which photos the golden set and the evaluations may use, where they come from and how each source may be used. The golden set comes from public sources; nobody has to supply photos. Re-check the licence when you fetch a file, and record the licence and attribution of every file in `tools/fetch/manifest.json`. The scripts and the rules they enforce are described in `tools/fetch/README.md`.

Openly licensed A1 photos are scarce, so the set mixes three tiers. The A1 II (ILCE-1M2) uses the same 50 MP sensor generation as the A1, and the A7R IV, A7R V and A7CR use the A7R IV profile that the A1 borrows, so their files are useful stand-ins for scene variety.

## Tier 1 · CC0, fetched automatically, usable anywhere (CI, public bucket)

| Source | What | How |
|---|---|---|
| RawDB, Sony ILCE-1 | 3 files, ISO 500, 14-bit: Compressed, Lossless Compressed, Uncompressed (same scene) | JSON API `https://rawdb.dnglab.org/api/sets/Sony/ILCE-1` lists `path`, `size`, `sha256`, `license` (CC0 1.0) per file |
| RawDB, Sony ILCE-1M2 | 9 files, ISO 100, 14-bit: Uncompressed, Lossless (L/M/S), Lossy, each full-frame and APS-C crop | `https://rawdb.dnglab.org/api/sets/Sony/ILCE-1M2` |
| RawDB, other bodies | the same JSON API per model (`/api/sets/Sony/<model>`) for every body in `docs/03-camera-profiles.md` | use for decoder coverage and the public golden tier |
| raw.pixls.us | `https://raw.pixls.us/data/Sony/ILCE-1/` and `/ILCE-1M2/`, plus other Sony bodies | new uploads are CC0; some older files are CC BY-NC-SA: exclude those. The site disallows some automated fetchers, so fetch politely (single connection, identify the user agent) or list the files by hand once and pin their hashes |

These are mostly test scenes at one ISO: good for decoding and regression, thin for colour and light variety.

## Tier 2 · Freely licensed (mostly CC BY-SA or CC0), usable in CI with attribution

| Source | What | How |
|---|---|---|
| discuss.pixls.us, Play Raw category | real-world raw files shared for editing; the category rule is that "raw files must be freely licensed", most under Creative Commons | search the forum for ILCE-1, ILCE-1M2, ILCE-7RM4, ILCE-7RM5 and ILCE-7CR posts (Discourse search API, e.g. `/search.json?q=ILCE-1 category:play-raw`); read each post's licence; keep only CC0, CC BY and CC BY-SA files; store the author, post URL and licence with each file |

The aim is 50 files covering daylight, shade, tungsten, LED, mixed light, night, skin, foliage and high ISO. If automated access to the forum is refused, list the candidate posts in `docs/STATUS.md` and ask the maintainer to download them by hand.

CC BY-SA files may sit in a public test bucket with attribution. They never ship in the app.

## Tier 3 · Copyrighted, local evaluation only

| Source | What | Terms |
|---|---|---|
| Photography Life, "Sony a1 Sample Images & RAWs" (https://photographylife.com/sony-a1-sample-images-raws) | 22 A1 raw files, various ISOs and light | the author keeps copyright; personal use only; no republishing; downloads work only from the page itself |
| Imaging Resource, Sony A1 gallery and studio files (https://www.imaging-resource.com/news/2021/02/26/sony-alpha-1-gallery-images-real-world-sample-images-from-sonys-flagship) | real-world gallery and studio test scenes | copyrighted; personal evaluation |

Use these only on a development machine for visual review and the blind A/B (`docs/13-testing.md`). Never commit them, upload them, cache them in CI or put them in a test bucket. Download them by hand from the pages; don't scrape the galleries or script around download restrictions.

## Local evaluation · real files with unclear or restricted licences (not a tier of the golden set)

Next to the three tiers there is a fourth source that is **not** golden-set data: `tools/fetch/real-eval.sh` fetches nine real Sony files (DSLR-A300, DSLR-A500, ILCE-5000, ILCE-6300 cRAW, ILCE-7M2 uncompressed 14-bit, ILCE-7M3, ILCE-7M4, NEX-5N, NEX-7) from three public GitHub repositories with plain `git`, each at a pinned commit and each file at a pinned blob id, into `testdata/real/`. It exists because a decoder or a colour chain that has never seen a real file is a guess, and the CC0 hosts above are not always reachable from a build environment.

None of the three repositories gives a licence for these files that this project can rely on: the README of one says "You are free to use these media files however you wish" with no author, one records CC BY-NC-SA 4.0, and one says "public domain" with no provenance next to an AGPL `LICENSE`. The reasons are in `docs/validation-data.md` and `tools/fetch/README.md`. So the files follow the rules of tier 3: a development machine only, never committed, never uploaded, never cached or run in CI (the script refuses with exit code 2 when `CI` or `GITHUB_ACTIONS` is `true`), no picture made from them in the repository. Unlike tier 3 they need no download by hand.

They are used for smoke tests that skip without them, for the auto-exposure study (`tools/eval`) and for looking at real photographs. **No gate and no golden test depends on them**, and a result from them is reported as "local, not corpus data". If the origin of a file is confirmed to be CC0 (the immich repository has a public-domain statement whose origin could be checked), the maintainer can propose it for tier 1 under the usual rules of the manifest, which takes CC0 only.

## The owner's own photos (test data, CI and published images allowed)

The owner's own Sony photos are kept in the repository: the raw files live on the `readme-photos` branch in Git LFS (never merged; about 45 MB each), with the owner's reference exports in `ref-export/`. They may be used as test and evaluation data, in CI, and in images the repository and its README publish. They are never bundled with the app, and no credit is given for them.

`tools/fetch/owner-photos.sh` fetches them into `testdata/readme-photos/src/` and checks their SHA-256 (`tools/fetch/owner-photos.sha256`). There are three Sony A1 photos (A1M01454, A1M01739, A1M01749), daylight landscapes. They are not taken for V1 (they existed already), so the V1 data rule holds. They are not a golden set on their own: three files, one body, one kind of light.

## No test shoots in V1

V1 does not use controlled captures. Where a gate would have used one, it uses this instead:

| Would have needed | V1 replacement |
|---|---|
| grey-card shots under known lights | synthetic scenes, Cube++ (CC BY 4.0) converted per profile, NUS-8 Sony subset (evaluation only) |
| flat fields and lens tests | darktable-cli as an oracle, Adobe DNG Converter opcodes, line and flatness checks on public files |
| dark and flat calibration frames | per-image estimation with priors fitted from public files; noise-on-noise tests; A1 studio ISO series (evaluation tier) |
| pixel-shift captures | PixelShift200 (CC BY 4.0) |
| hand-set exposures | the camera's embedded JPEG as the exposure reference |

The captures come back in V2 (`docs/15-milestones.md`, V2 backlog).
