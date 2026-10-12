# tools/fetch

Scripts that download test data into `testdata/` (git-ignored) and check it against SHA-256 hashes. Raw files, training data and other large test data are never committed (`AGENTS.md`). The spec is `docs/13-testing.md` (section Test data) and `docs/test-photo-sources.md`; the plan for every gate's data is `docs/validation-data.md`.

| File | What |
|---|---|
| `samples.sh` | The CC0 Sony ARW corpus from RawDB and raw.pixls.us, into `testdata/arw/<model>/` |
| `golden.sh` | The golden photo set: tier 1 (CC0) and tier 2 (free licence, with attribution) from the manifest; tier 3 is a local-only list |
| `manifest.json` | The body list, the source definitions, and (once refreshed) one entry per file with size, SHA-256 and licence |
| `ymfetch.py` | The implementation, Python 3.8+ standard library only. The two `.sh` files are thin wrappers that run it with `python3 -I` |
| `test_ymfetch.py` | Unit tests against a local HTTP server; no network. `python3 -I -m unittest discover -s tools/fetch -p 'test_*.py'` |
| `spectral.sh` | Not part of this tool: fetches the spectral inputs (rawtoaces-data, aces-dev) for the colour scripts |

## Status

**The manifest ships with an empty file list.** `rawdb.dnglab.org`, `raw.pixls.us` and `discuss.pixls.us` were unreachable from the environment that wrote these scripts. From there:

```
$ curl -sS -I https://rawdb.dnglab.org/api/sets/Sony/ILCE-7RM4
curl: (56) CONNECT tunnel failed, response 403
```

So the scripts were tested against a local mock server (answers shaped like the layout in `docs/test-photo-sources.md`), not against the real hosts. Until the manifest is filled, `samples.sh` and `golden.sh` stop with exit code 3 and say so; with the network refused they exit 4:

```
$ tools/fetch/samples.sh --refresh --body ILCE-7RM4
RawDB ILCE-7RM4
error: cannot reach rawdb.dnglab.org: Tunnel connection failed: 403 Forbidden
  The network, or a proxy in front of it, refuses this host. ...
```

### First run on a machine with open internet

1. `tools/fetch/samples.sh --refresh --dry-run` shows what the RawDB API returns without writing anything. The raw answers are also saved under `testdata/.cache/api/`.
2. If it fails with "has no download URL", look at the saved answer and set `sources.rawdb.file_url_template` in `manifest.json` (placeholders `{root}`, `{path}`, `{make}`). The code assumes the answer lists `path`, `size`, `sha256` and `license` per file (`docs/test-photo-sources.md`) and takes the download URL from a `url`-like field, an absolute `/path`, or the template. **That layout and the URL rule are unverified against the live site.** `--refresh` asks for one byte of the first file before it writes the manifest, so a wrong URL rule is caught there.
3. `tools/fetch/samples.sh --refresh` writes the manifest. Review the diff and commit it. Refresh refuses to change the hash of a file that is already pinned unless you pass `--accept-changed`.
4. `tools/fetch/samples.sh --refresh --source pixls` lists raw.pixls.us files under `pending`. The directory listing does not state a licence and some old files are CC BY-NC-SA, so nothing in `pending` is ever downloaded by a plain run. Check each file's licence on the site, then `tools/fetch/samples.sh --promote pixls:ILCE-7RM4/<name> --license "CC0 1.0" --checked-by "<who, where>"`. That downloads the file once, pins its hash and records who checked.
5. `tools/fetch/samples.sh --coverage` prints the body x variant table (Markdown) for `docs/STATUS.md`. The variant columns are hints guessed from file names and API text, not authoritative; `yellowmoon info` is.

## Usage

```
tools/fetch/samples.sh                      download and verify the corpus listed in the manifest
tools/fetch/samples.sh --verify             check what is on disk (no network); exit 1 on any difference
tools/fetch/samples.sh --body ILCE-7RM4     one camera model (repeatable)
tools/fetch/samples.sh --per-body 2 --max-file-mb 80     a small subset, for quick runs
tools/fetch/samples.sh --dry-run            show the plan
tools/fetch/samples.sh --list               one line per entry (id, size, licence, tier, variant hint)
tools/fetch/samples.sh --coverage           body x variant table
tools/fetch/samples.sh --refresh [--source rawdb|pixls|all] [--group required|sibling|recent|legacy] [--body M]
tools/fetch/samples.sh --promote ID --license CC0-1.0 --checked-by "..."

tools/fetch/golden.sh                       tier 1 and tier 2 (writes testdata/golden/ATTRIBUTION.md)
tools/fetch/golden.sh --tier 1              CC0 only
tools/fetch/golden.sh --verify
tools/fetch/golden.sh --tier3               print the local-only evaluation list
tools/fetch/golden.sh --index-tier3         record size and SHA-256 of the tier 3 files you downloaded by hand
tools/fetch/golden.sh --candidates [MODEL...]     search the Play Raw forum for UNREVIEWED tier 2 candidates
tools/fetch/golden.sh --add-tier2 URL --body ILCE-1 --license "CC BY-SA 4.0" --author NAME --source-url POST_URL [--title T] [--name FILE]
```

Common options: `--manifest`, `--dest` (default `testdata/`, or `$YM_TESTDATA`), `--delay` (seconds between requests, default 1), `--timeout`, `--retries`, `--allow-empty`.

Exit codes: 0 ok, 1 verification or licence failure, 2 usage error, 3 the manifest is empty or no entry matches, 4 the network is unreachable or refused.

### In CI

```yaml
- uses: actions/cache@v4
  with: { path: testdata, key: testdata-${{ hashFiles('tools/fetch/manifest.json') }} }
- run: tools/fetch/samples.sh        # and tools/fetch/golden.sh for the golden job
```

A cache hit makes the run a hash check only. The manifest has no timestamp, so the cache key only changes when the file list changes.

## Layout of `testdata/`

```
testdata/arw/<model>/<name>.ARW             corpus and golden tier 1 (CC0)
testdata/golden/tier2/<model>/<name>.ARW    golden tier 2 (free licence, attribution in ATTRIBUTION.md)
testdata/golden/tier3/<provider>/...        copyrighted, put there by hand, local only
testdata/.cache/api/                        raw API answers saved by --refresh
testdata/sources/                           spectral.sh checkouts
```

`<name>` is the file's path inside the model's folder on the server with `/` replaced by `__`, other characters outside `A-Za-z0-9._+()-` replaced by `_`. This keeps names flat and unique, and means a manifest can never write outside `testdata/`.

## Manifest

```
schema_version  1
policy          the accepted licences (mirrors the constants in ymfetch.py, which enforces them)
sources         rawdb, pixls, playraw: base URLs, the hosts downloads and redirects may use, notes
bodies          model (RawDB/EXIF string, e.g. ILCE-7RM4), name, model_id (docs/03, null if unknown), group, note
files           one entry per downloadable file (below)
pending         raw.pixls.us candidates whose licence has not been checked
excluded        files seen by --refresh and rejected, with the reason (audit trail)
golden          tier1_bodies: bodies whose corpus files are also tagged golden; tier3: the local-only list
```

A `files` entry: `id` (`rawdb:ILCE-1/<name>`), `source`, `body`, `name`, `source_path`, `url`, `size`, `sha256`, `license` (SPDX-style), `license_source` (`rawdb-api`, `manual`, `forum-post`), `tier` (1 or 2), `sets` (`corpus`, `golden`), `variant_hint`, and for tier 2 an `attribution` object (`author`, `source_url`, `title`). The bodies are the 19 required by `docs/15` M0-09 (group `required`), the same-sensor siblings from `docs/03` (`sibling`), RX1R III and RX10 V (`recent`), and older bodies for the 12-bit and ARW 1.0 to 2.x paths (`legacy`).

## Rules the code enforces

- **Licence, per file:** the corpus and golden tier 1 take `CC0-1.0` only. Tier 2 takes CC0, CC BY 2.0 to 4.0 and CC BY-SA 2.0 to 4.0, and needs an author and a source URL. CC BY-NC, ND, unknown and unverified files are never downloaded, and a manifest that lists one fails validation before any request. Licence strings from the sites are normalised ("CC0 1.0" to `CC0-1.0`); a Creative Commons string without a version counts as unknown on purpose.
- **Polite:** one connection at a time, a descriptive User-Agent (`yellowmoon-fetch/<version> (+https://github.com/ngriaznov/yellowmoon; ...)`), a pause between requests, back-off on HTTP 429 and 5xx (honouring `Retry-After`), resumable downloads (`*.part` plus a `Range` request). raw.pixls.us disallows some automated fetchers: if it answers 403 the script says to download the file once in a browser and pin it by hand.
- **Safe:** https only (plain http only to loopback, for the tests); every download and every redirect must stay on a host named in `manifest.sources`; file names are validated; a file whose size or SHA-256 differs from the manifest is deleted, never kept.
- **Tier 3 is never fetched.** `golden.sh` has no code path that downloads it. The two providers (Photography Life, Imaging Resource) forbid republishing and allow downloads only from their own pages, so the list in the manifest holds page URLs and terms, nothing that could be requested. Download those files by hand into `testdata/golden/tier3/<provider>/`; never commit, upload or cache them in CI. `golden.sh --index-tier3` writes `testdata/golden/tier3/INDEX.json` (size and SHA-256) for local tools.

## Tier 2: finding Play Raw files

`golden.sh --candidates ILCE-1 ILCE-7RM5` queries the discuss.pixls.us search API (Play Raw category) and saves posts that have ARW attachments to `testdata/golden/tier2-candidates.json` with the licence words found in the post. These are candidates, marked `unreviewed`. A person reads each post's licence, then pins the file with `golden.sh --add-tier2 ...`, which downloads it, records the hash and attribution, and only accepts CC0, CC BY or CC BY-SA. If the forum refuses automated access, list the candidate posts in `docs/STATUS.md` and ask Mykyta to download them by hand.
