# tools/fetch/autoset

The CC0 Sony ARW files for the autoset training set (`docs/18-auto-settings.md` section 6). Files are streamed: each one is downloaded, checked, run through the dump program, and deleted, so the disk never holds more than one raw file. Raw files are never committed; the manifest (`manifest.json`) is the only thing in the repository, and it has no timestamps.

```
tools/fetch/autoset.sh refresh [--source pixls|rawdb|all] [--dry-run] [--accept-changed]
tools/fetch/autoset.sh list [--body M]...
tools/fetch/autoset.sh build --out testdata/autoset/rows.jsonl [--limit N] [--body M]... [--id ID]...
                             [--keep] [--dump CMD] [--retry-failed] [--dest DIR]
tools/fetch/autoset.sh verify-rows testdata/autoset/rows.jsonl
```

Python 3.8+ standard library only, run with `python3 -I`. The tests need no network:

```
python3 -I -m unittest discover -s tools/fetch/autoset -p 'test_*.py'
```

## Sources and licence

- **raw.pixls.us**: the `getrepository.php?set=all` listing. Kept: make Sony, licence `publicdomain/zero/1.0` (CC0 1.0), extension ARW. SR2, SRF and ARQ are excluded.
- **RawDB** (`rawdb.dnglab.org`): the paged set index with `maker=Sony` (the client also filters by maker), then one request per Sony set. Kept: file licence exactly `CC0 1.0`, extension arw (any category).

Both sources list the same files, so entries are deduplicated by SHA-256. One entry per file: the pixls entry is kept when both have it, the other source's id goes under `also`, and the exact RawDB size fills in the pixls entry's missing size. The manifest's `licence` is `CC0-1.0`; the tool refuses anything else.

## The manifest

Each file entry, one per line in `manifest.json`, sorted by (body, source, name):

| Field | Meaning |
|---|---|
| `id` | `pixls:<body>/<name>` or `rawdb:<body>/<path in the set>`. When two different files share a name, the id gets ` #<first 12 hex of SHA-256>` |
| `source` | `raw.pixls.us` or `rawdb.dnglab.org` (the value written into each row) |
| `body`, `name`, `mode` | camera model, file name, and the source's mode text (pixls: the listing's mode; RawDB: the file's tags) |
| `url` | percent-encoded download URL |
| `sha256` | lower-case SHA-256 (verified after every download) |
| `size` | exact byte count (RawDB), or `null` |
| `size_approx` | the listing's text, for example `10.55MB` (pixls); used for the disk check only |
| `licence` | `CC0-1.0` |
| `also` | ids of the same file in the other source (or in the same source under another name) |

`refresh` refuses to change the SHA-256 of an id already in the manifest (the upstream file changed) unless `--accept-changed`. It rewrites the manifest atomically; `--dry-run` prints the counts and writes nothing. `--source pixls` (or `rawdb`) rebuilds only that source's entries and keeps the other source's.

## build

For each entry, in manifest order (`--body`, `--id` filter it):

1. Skip it if the rows file already has its `aug_k = 0` row (resumable), or if it failed earlier and `--retry-failed` is not given.
2. Check free space: at least 2 × the file size + 1 GiB must be free in `<dest>/autoset/tmp`, else stop with exit 5 before downloading.
3. Download to `<sha256>.ARW.part` (1 MiB chunks, hashed while written), three tries with backoff on HTTP 429/5xx and on dropped transfers. A RawDB download follows its 303 to object storage. Rename to `<sha256>.ARW` only if the SHA-256 (and the exact size, where known) matches.
4. Run the dump program: `<dump> --source <source> --licence CC0-1.0 <file>`. Its stderr is shown; a non-zero exit is a `dump` failure.
5. Validate every row: JSON; `sha256` equal to the manifest's; `aug_k` a finite number; `features` 39 finite numbers; `licence` CC0-1.0; at least one `aug_k = 0` row. Then set `file` (the manifest name), `source`, `licence`, `id`, and `body` if the dump wrote null. Nothing is appended unless every row passes.
6. Append all rows in one write, then flush and fsync.
7. Delete the raw file and any `.part` (also on failure), unless `--keep`.

A file-level failure (download, verify, dump, rows) is written to `failures.jsonl` as `{"id", "sha256", "stage", "error"}` and the run goes on; the record is removed when the file later succeeds. A host that cannot be reached at all stops the run with exit 4 and records nothing. Ctrl-C removes the `.part` and the raw file in progress and exits 130; rows already written are kept.

Paths:

- Rows: `--out` (default `<dest>/autoset/rows.jsonl`). `failures.jsonl` is next to it.
- Raw files: `<dest>/autoset/tmp/`. `<dest>` is `--dest`, else `$YM_TESTDATA`, else `testdata/`.

The dump program is `target/release/examples/autoset_dump` by default. If it does not exist, `build` runs `cargo build -p ym-core --release --example autoset_dump` first. `--dump CMD` (split with shlex) or `$YM_AUTOSET_DUMP` overrides it.

`build` prints a summary (done, skipped, failed, rows written). Exit 0 if nothing failed.

## verify-rows

Checks a rows file against the manifest without the network: every row's `sha256` must be in the manifest and its features must be well formed. It prints the rows per body and source, and the files with rows against the manifest's entries. Exit 1 on any error.

## Environment

- `HTTPS_PROXY` is honoured (urllib). `SSL_CERT_FILE` replaces the system CA bundle; verification is never switched off.
- The User-Agent is `yellowmoon-autoset-fetch (+https://github.com/ngriaznov/yellowmoon)`.
- URLs must be https, to the two source hosts or the RawDB storage domain (`your-objectstorage.com`). Plain http is accepted for loopback only, for the tests.

## Exit codes

0 ok; 1 failures (or a manifest that breaks the rules); 2 usage error or a missing tool; 3 nothing to do (empty manifest, or nothing matched); 4 network unreachable or refused; 5 not enough disk; 130 interrupted.
