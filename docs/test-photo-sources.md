# Test photo sources (A1 and siblings)

The golden set comes from public sources; Mykyta doesn't need to supply photos. This lists what was found on 11 October 2026 and how each source may be used. Re-check licences when you fetch; record the licence and attribution of every file in `tools/fetch/manifest.json`.

Openly licensed A1 photos are scarce, so the set mixes three tiers. The A1 II (ILCE-1M2) uses the same 50 MP sensor generation as the A1, and the A7R IV, A7R V and A7CR use the A7R IV profile that the A1 borrows, so their files are useful stand-ins for scene variety.

## Tier 1 · CC0, fetched automatically, usable anywhere (CI, public bucket)

| Source | What | How |
|---|---|---|
| RawDB, Sony ILCE-1 | 3 files, ISO 500, 14-bit: Compressed, Lossless Compressed, Uncompressed (same scene) | JSON API `https://rawdb.dnglab.org/api/sets/Sony/ILCE-1` lists `path`, `size`, `sha256`, `license` (CC0 1.0) per file |
| RawDB, Sony ILCE-1M2 | 9 files, ISO 100, 14-bit: Uncompressed, Lossless (L/M/S), Lossy, each full-frame and APS-C crop | `https://rawdb.dnglab.org/api/sets/Sony/ILCE-1M2` |
| RawDB, other bodies | the same JSON API per model (`/api/sets/Sony/<model>`) for every body in `docs/03` | use for decoder coverage and the public golden tier |
| raw.pixls.us | `https://raw.pixls.us/data/Sony/ILCE-1/` and `/ILCE-1M2/`, plus other Sony bodies | new uploads are CC0; some older files are CC BY-NC-SA: exclude those. The site disallows some automated fetchers, so fetch politely (single connection, identify the user agent) or list the files by hand once and pin their hashes |

These are mostly test scenes at one ISO: good for decoding and regression, thin for colour and light variety.

## Tier 2 · Freely licensed (mostly CC BY-SA or CC0), usable in CI with attribution

| Source | What | How |
|---|---|---|
| discuss.pixls.us, Play Raw category | real-world raw files shared for editing; the category rule is that "raw files must be freely licensed", most under Creative Commons | search the forum for ILCE-1, ILCE-1M2, ILCE-7RM4, ILCE-7RM5 and ILCE-7CR posts (Discourse search API, e.g. `/search.json?q=ILCE-1 category:play-raw`); read each post's licence; keep only CC0, CC BY and CC BY-SA files; store the author, post URL and licence with each file |

Aim for 50 files covering daylight, shade, tungsten, LED, mixed light, night, skin, foliage and high ISO. If automated access to the forum is refused, list the candidate posts in `docs/STATUS.md` and ask Mykyta to download them by hand.

CC BY-SA files may sit in a public test bucket with attribution. They never ship in the app.

## Tier 3 · Copyrighted, local evaluation only

| Source | What | Terms |
|---|---|---|
| Photography Life, "Sony a1 Sample Images & RAWs" (https://photographylife.com/sony-a1-sample-images-raws) | 22 A1 raw files, various ISOs and light | the author keeps copyright; personal use only; no republishing; downloads work only from the page itself |
| Imaging Resource, Sony A1 gallery and studio files (https://www.imaging-resource.com/news/2021/02/26/sony-alpha-1-gallery-images-real-world-sample-images-from-sonys-flagship) | real-world gallery and studio test scenes | copyrighted; personal evaluation |

Use these only on a development machine for visual review and the blind A/B (`docs/13`). Never commit them, upload them, cache them in CI or put them in a test bucket. Download them by hand from the pages; don't scrape the galleries or script around download restrictions.

## No test shoots in V1

V1 doesn't use controlled captures. Where a gate would have used one, it uses instead:

| Would have needed | V1 replacement |
|---|---|
| grey-card shots under known lights | synthetic scenes, Cube++ (CC BY 4.0) converted per profile, NUS-8 Sony subset (evaluation only) |
| flat fields and lens tests | darktable-cli as an oracle, Adobe DNG Converter opcodes, line and flatness checks on public files |
| dark and flat calibration frames | per-image estimation with priors fitted from public files; noise-on-noise tests; A1 studio ISO series (evaluation tier) |
| pixel-shift captures | PixelShift200 (CC BY 4.0) |
| hand-set exposures | the camera's embedded JPEG as the exposure reference |

The captures come back in V2 (`docs/15`, V2 backlog).
