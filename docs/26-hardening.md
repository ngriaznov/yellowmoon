# 26 · Hardening (M4-02)

This document is the audit and the plan for task **M4-02 · Hardening** of `docs/15-milestones.md`: *8 GB Macs, Intel Macs, 2000-file folders, corrupt files, disk-full during export. Done when: each case handled with a clear message.*

Date of the audit: 2026-11-11. Branch `m4-02-hardening`, made from `main` at e4ae7bb. The audit changed no code. It reads the code and the specs, and it measures what a Linux machine can measure. Each case has the same four parts: **Today** (what the code does now, with `file:line`), **Gaps**, **Plan** (the fix, the exact user-facing message, the test) and what only a Mac can check. The last sections split the work into tasks H-01 to H-18 and list the checks that need a Mac ([HUMAN], M0-H4 style).

## How the numbers were made

Every number below is cited from a spec or was measured for this audit. Nothing is estimated unless a sentence says so.

- **Machine.** Linux x86_64 VM, 4 vCPU, 15 GB RAM, no GPU (`ym doctor`: "none, CPU fallback"), so every render is on the CPU path. This is not a Mac. Times and peak memory are for this machine only.
- **Binary.** `cargo build --release -p ym-cli` at e4ae7bb (`ym 0.2.3`). Synthetic ARW files from `crates/ym-raw/examples/gen_arw.rs` (uncompressed 14-bit, model ILCE-7RM4/7RM5) and `gen_corpus.rs` (48 files, every raw variant). Peak memory is `ru_maxrss` of the child process (a 10-line Python wrapper, `resource.getrusage`). The data folder is `YELLOWMOON_DATA=<repo>/data`; the SAM and depth weights are not in this checkout, so the model costs below are cited from the specs.
- **Full disk.** A 3 MiB `tmpfs` mounted over the destination (`mount -t tmpfs -o size=3m`). **Unreadable files and folders.** `setpriv --bounding-set=-dac_override,-dac_read_search`, because the sandbox runs as root, which otherwise reads everything.
- **Front end.** The production build (`vite build`) served statically, driven by headless Chrome (Playwright 1.56 library, `/usr/bin/google-chrome`, software rendering, 1440 × 900 at device scale 2) against the mock API with `?mockPhotos=N`. The harness is a copy of `e2e/perf/latency.perf.ts` for other values of N (60 presses of the right arrow 100 ms apart, time from the key event to two frames after the caption changes). Chrome here is slower than the machine behind the figures in `docs/11` ("Large folders": median 24 ms at 500 photos), so compare the rows of one table with each other, not with those figures.
- **Not measured and said so:** anything on a Mac (Metal, WKWebView, unified memory, APFS, Intel hardware), real photographs (none are in the sandbox: `testdata/` does not exist), the GPU path, SAM and depth memory.

## Findings at a glance

| Case | What already works | The biggest gaps |
|---|---|---|
| 1 Corrupt files | `ym-raw` has no `unwrap`, `expect` or `panic` outside tests; a damaged file is `Skipped` with a reason and never stops a batch; 192 damaged files (4 kinds of damage × 48 variants) gave 0 panics and 0 crashes | An unreadable folder reads "No ARW files"; a zero-byte file reads "This isn't an ARW file"; messages carry jargon ("Huffman table") and nested brackets; a photo whose file vanishes after it was Ready stays Ready; no committed damaged-file matrix across the app, the core and `ym` |
| 2 Disk full | Temp file then rename; the partial file is removed (measured for JPEG, PNG, TIFF, DNG); a failed Replace keeps the old file; one photo's failure never stops the batch | A full disk (or a read-only or vanished destination) does not stop the job: **every remaining photo is decoded and rendered before it fails** (10 photos: 87 s with or without space); N identical error lines; the header says "Exported" when nothing was saved; no free-space warning; stale `.yellowmoon-*.tmp` files are never cleaned |
| 3 2000 files | Listing, probe and develop have no O(n²) cost that shows at 2000 tiny files (2000 photos listed, probed, developed and failed in 2.9 s, 146 MiB); thumbnails cross IPC as addresses | The front end draws every photo: key latency rises with the count (median 49 ms at 100 photos, 86 ms at 500, 254 ms at 2000 in the same Chrome); 8,088 DOM nodes on the Start list and 16,063 in the running Export panel at 2000; the shell's filmstrip thumbnails have no byte budget; `Session::open` canonicalises every path while it holds the session lock; no progress or cancel while probing |
| 4 8 GB Macs | Budget rule for unified memory is unit-tested at 8 GB; a full render that does not fit is refused with a plain message (`RenderError::Memory`); caches are byte-bounded | The ledger counts a third to a half of what a 61 MP export really uses (formula 1.1 GiB, measured peak 2.4 GiB on the CPU path); the develop pool runs up to 6 jobs whatever the RAM; models stay loaded for ever; no memory-pressure handling; no way to run the whole pipeline "as an 8 GB Mac"; STATUS question 4 is still open |
| 5 Intel Macs | The x86_64 slice is built (release, build.yml) and the code has runtime AVX2 checks with scalar fallbacks; Apple Auto reports `intel_mac`; f16 is not needed (storage is f32) | Nothing ever **runs** the x86_64 slice: every macOS runner is Apple silicon and `ym --version` runs the arm64 slice; no Intel CI job; the GPU budget of a discrete AMD GPU is `min(4 GiB, 2 × max_buffer_size)` whatever its video memory; Vision masks on Intel are unproven |

---

## 1. Corrupt files

### 1.1 Today

**What a file can be and how it fails** (`docs/04-raw-decoding.md` section Errors; `crates/ym-raw/src/error.rs`):

- `NotArw` ("This isn't an ARW file. Yellowmoon opens Sony ARW files only."), `Arq`, `Unsupported`, `Corrupt { detail }` and `Io` are typed errors; the `Display` text is the sentence the person sees (`error.rs:28` for `Corrupt`, `error.rs:36` for `Io`).
- A read past the end of the file is `Corrupt` ("the file ends before its data does", `error.rs:62-68`).
- The files are read with positional reads (`FileSource`, `crates/ym-raw/src/tiff.rs:255`, `read_exact_at`), never memory-mapped, so a file that is truncated while open gives an error and not a `SIGBUS`.
- Limits (`docs/04:42`, enforced in `layout.rs:13-16`, `tiff.rs:23` and `tiff.rs:32`): at most 120 MP and 65,535 px a side, 8 MiB a value, 64 MiB of reads per probe; raw data must lie inside the file. All are `Corrupt`.
- **Panic scan.** The non-test code of `ym-raw` has no `unwrap()`, `expect(`, `panic!`, `unreachable!` or `todo!` (scan of every file before its first `cfg(test)`). The same scan over `ym-core`, `ym-export`, `ym-cli`, `src-tauri`, `ym-looks` and `ym-profiles` finds seven sites, all invariants with a comment (`ym-export/src/naming.rs:347`, `jpeg/interleave.rs:586`, `dng.rs:475`, `dng/fitted.rs:220`, `ym-cli/src/info.rs:179`, `ym-profiles/src/color_matrices.rs:98` for an embedded table) or a test-only switch (`ym-core/src/pipeline/graph.rs:321`, behind the `force-demosaic` feature). None is reachable from file contents. Slice indexing was not scanned.

**Where a bad file surfaces:**

| Place | What happens | Source |
|---|---|---|
| Open / import (app and `ym`) | `Session::open` probes every file in parallel (`session.rs:2243-2246`). A failed probe makes a photo that is `Skipped` with the error text; it is in the filmstrip and counted. Other files go on. | `docs/01:143`, `session.rs:2210` |
| Develop (decode, estimates, thumbnail) | One job per photo (`session.rs:3902`); a decode error ends that photo in `Skipped` (`session.rs:3929-3932`). A panic in a job is caught (`session.rs:3833`) and reported as "Yellowmoon hit a problem it didn't expect with this file. Skip it and open the others." (`session.rs:3856`). | |
| Thumbnail strip | A skipped photo has the tag CAN'T OPEN (`src/components/Filmstrip.tsx:244`, `Tag.tsx:53`). | |
| Viewer | A skipped photo shows "This file can't be opened." and the reason (`src/screens/PhotoView.tsx:65`). The viewport draws nothing (`src-tauri/src/viewport.rs:1275-1279`). | |
| Start screen list | The row is SKIPPED with the reason (`DevelopList`, `src/components/ExportQueueList.tsx`). | |
| Node thumbnails, tile refresh | Failures are logged at debug level only (`node_thumbs.rs:172`, `thumbs.rs:202`); the old picture stays. | |
| Masks (SAM, depth, Vision), Auto (Apple, CLI agents) | They work on a photo that is already developed, so a corrupt file never reaches them. A request that fails (a file that vanished) fails that photo's request only; the rule is used (`docs/01:188`). | |
| Export, single and batch | `load_raw` decodes again if the mosaic is not cached (`export_queue.rs:752`); a failure is that photo's `ExportError::Decode` (`export_queue.rs:904`); the job goes on to the next photo (`export_queue.rs:975-1004`). | |
| `ym render` | Each bad input is reported on stderr, counted, and the others are rendered (`render.rs:430-440`, `render.rs:300`). Exit 1 if any failed, 5 if no ARW is left (`render.rs:952-962`). | `docs/10` |
| `ym info` | Prints the error of the first bad file and exits 1. | |

**Measured with the real `ym` (CPU path), 12 hand-made files** (a 1200 × 800 uncompressed file, then damaged):

| File | Result | Message (shortened) |
|---|---|---|
| zero bytes | exit 1 | "This isn't an ARW file." |
| text renamed `.ARW`, JPEG renamed `.ARW` | exit 1 | "This isn't an ARW file." |
| first 8 or 100 bytes only | exit 1 | "…damaged or incomplete (the first directory lies outside the file)" / "(a value lies beyond the end of the file)" |
| cut at 50 %, or last 1,000 bytes missing | exit 1 | "…(the raw data runs past the end of the file (the file is cut short))" |
| 4 bytes of the TIFF header overwritten | exit 1 | "…(the first directory lies outside the file)" |
| 4,000 random bytes inside the pixel data | **exit 0, a JPEG of garbage pixels** | none (an uncompressed file has no checksum) |
| dangling symlink named `x.ARW` | exit 1 | "There is no file or folder with this name." |
| folder named `x.ARW` given as an input | walked as a folder; nothing found | none |
| mode 000 file (`setpriv`) | exit 1 | "Yellowmoon couldn't read this file (Permission denied (os error 13)). Check that it still exists and that you can open it." |
| mode 000 **folder** | exit 5 | **"There are no files to render. Give ARW files, or folders that hold them."** |

A batch of eight inputs (a good file, the file with garbage pixels, five unreadable or damaged files and a folder named `x.ARW`) exported two photos in 2.7 s, printed five error lines and "5 photos failed.", exit 1.

**Damage sweep.** The 48 files of `gen_corpus` (arw1, packed 12-bit, uncompressed, cRAW, lossless L with every predictor and tile shape, lossless M/S) were each damaged four ways (300 random bytes at 60 % of the file, 2,000 zero bytes at 50 %, one flipped bit at a random place, 8 random bytes in the first 600 bytes): **192 files, 0 panics, 0 signals, 0 hangs.** 155 exported (the damage hit pixel values that decode), 35 failed with `Corrupt` (21 "lossless JPEG: the entropy-coded data ends before the image does", 11 "the ARW 1 data holds a sample outside the 12-bit range", 2 "…fewer restart intervals…", 1 "…a code that is not in the Huffman table"), 2 were "not an ARW file".

**Tests and fixtures that exist:**

- `crates/ym-raw/tests/decode_hostile.rs:255`: about 3,600 seeded attacks on one small file of every decodable variant (random damage, size tags set to 0, 1, 31, 4096, 65535, 2³² − 1, 120 MP claims behind a 15 kB file, truncation); no panic, no attempt over 20 s, peak live heap at most 1.5 GiB (`docs/04:188`). It passes in 0.35 s in release.
- Seven `cargo fuzz` targets in `crates/ym-raw/fuzz/fuzz_targets` (`tiff_walker`, `sr2`, `probe`, `craw`, `ljpeg`, `ycbcr`, `decode_any`), run nightly by the `fuzz` job of `ci.yml` (`docs/13`). The corpora are git-ignored and made by `gen_seeds`.
- `crates/ym-raw/tests/decode_roundtrip.rs`, `container.rs`, `proptest_probe.rs`: damaged and truncated files are `Corrupt`.
- `crates/ym-core/tests/session.rs:88-215`: a folder with six good files, an ARW 6.0 file, a text file and a truncated file; the reasons are checked; nothing blocks.
- `crates/ym-cli/tests/info.rs:287,330` and `render.rs:203`: `not_arw`, `corrupt` and `arq` messages in `--json`.
- `e2e/export.spec.ts:231`: a failed file and a skipped file are listed with reasons (mock).
- **No committed or fetched set of damaged real files** (`tools/fetch` has none: a search for "corrupt", "damaged" and "truncat" finds nothing). The damage is made in memory by `testgen`.

### 1.2 Gaps

| # | Gap | Evidence |
|---|---|---|
| G1-1 | **An unreadable folder looks empty.** `walk()` returns silently when `read_dir` fails, in the session and in the CLI. The person is told "No ARW files in what you dropped." / "There are no files to render." | `session.rs:943-946`, `inputs.rs:85-88`; measured above (exit 5). Also happens for a folder on a drive that is gone, and for macOS privacy blocks on Desktop, Documents and removable volumes. |
| G1-2 | **A zero-byte file says "This isn't an ARW file."** The usual cause is an interrupted copy; the advice should be to copy again. | measured above |
| G1-3 | **Messages carry jargon and raw system text:** "Huffman table", "restart intervals", "entropy-coded data", "the first directory lies outside the file", nested brackets ("(…(the file is cut short))", `metadata.rs:881`) and "(No such file or directory (os error 2))", "(Permission denied (os error 13))". | measured above; `error.rs:28,36` print `detail` and the OS text in the sentence |
| G1-4 | **A photo that was Ready and whose file then vanishes stays Ready.** The viewport reports the read error in the error bar, the filmstrip still shows a normal thumbnail, and every render asks again. A file that is replaced by a longer or shorter one is not noticed either (no size or time is kept). | `viewport.rs:1283` (reports, once per message, `viewport.rs:1106`); `session.rs:4131` (decodes again with no check) |
| G1-5 | A permission block by macOS (an `EPERM`, "Operation not permitted", for a file on a removable volume or in a protected folder) gets the generic Io sentence, which tells the person to check that the file exists. | `error.rs:36` |
| G1-6 | **Bit flips in pixel data are accepted** (measured above, and 155 of 192 in the sweep). This cannot be detected: the format has no checksum. | |
| G1-7 | **No damaged-file matrix across the layers.** The hostile test stops at `decode_bytes`. Nothing checks, on one fixed set, that the app's `open_paths`, the export queue and `ym render` each keep going, print a catalogue message and never panic. | tests listed above |
| G1-8 | **A hung read cannot be stopped.** A file on a sleeping network share blocks `open` or `pread` in the kernel. `Session::open` probes with rayon and returns only when all probes return, so `open_paths` never answers; a develop job blocks one of the `min(cores − 2, 6)` pool threads, and `close()` does not wait for the pool (`docs/01:151`). Not reproducible on this machine; reasoned from the code. | `session.rs:2243`, `docs/01:151` |

### 1.3 Plan

**P1-1 (G1-1) Say that a folder could not be read.** `Collected` and `OpenOutcome` get `unreadable: Vec<PathBuf>`; the CLI's `Expanded` gets the same as problems of kind `unreadable`.

- Message, one folder: "Yellowmoon can't read the folder {name}. Check that it is still there and that you may open it, then try again." On macOS add the way out when the error is `PermissionDenied`: "Yellowmoon can't read the folder {name}. Allow access in System Settings, under Privacy & Security › Files and Folders (or Full Disk Access), then try again."
- With several unreadable folders the first is named and the count added ("…and 2 more folders.").
- `NO_ARW_MESSAGE` is used only when nothing was unreadable.
- Tests: unit tests of `collect` and `inputs::expand` with a folder made unreadable (Unix `chmod 000`; skip when the test runs as root, the way `tests/` of `ym-looks` skips); an `ym-cli` test that checks the exit code is 1 and that stderr names the folder; Vitest for the Start screen with the mock (`mockUnreadable=Name`).

**P1-2 and P1-3 (G1-2, G1-3, G1-5) One catalogue of file messages.** The sentence for a damaged file no longer prints `detail`. `detail` stays in the error, in `--json` (`error.detail`), in the log and with `-v`. A closed set of sentences:

| Case | Message |
|---|---|
| empty file (length 0) | "This file is empty. Copy it from the card again, or skip it." |
| cut short (every `UnexpectedEof`, raw data beyond the end of the file) | "This file is incomplete: it ends before its picture does. Copy it from the card again, or skip it." |
| any other `Corrupt` | "This file is damaged. Copy it from the card again, or skip it." |
| `Io` `NotFound` | "Yellowmoon can't find this file any more. It may have been moved, renamed or its drive disconnected." |
| `Io` `PermissionDenied` (EACCES) | "Yellowmoon isn't allowed to open this file. Check its permissions." |
| `Io` `PermissionDenied` (EPERM) on macOS | "macOS didn't let Yellowmoon open this file. Allow access in System Settings, under Privacy & Security." |
| other `Io` | "Yellowmoon couldn't read this file. Check that the drive is connected, then try again." |

Fix: `RawError::Corrupt` keeps its `detail`; the `Display` chooses one of three sentences from two constants (`EMPTY` and `CUT_SHORT` details, used by the length-0 check at the start of `FileSource` and by the `UnexpectedEof` mapping and the "runs past the end of the file" check). This leaves the 85 `RawError::corrupt(..)` call sites unchanged. The `Io` sentence is chosen from `io::ErrorKind` and `raw_os_error`. `kind()` stays as it is (`corrupt`, `io`, `not_arw`), so `--json` consumers do not break.

Test: `crates/ym-raw/tests/messages.rs` runs the hostile generator and the 192-file sweep of this audit through `decode_bytes` and asserts that every `Display` string is one of the catalogue sentences (a closed list, so a new one needs a decision) and contains none of "((", "os error", "Huffman", "entropy", "restart". Update `docs/04` Errors with the table, and the snapshots of `ym-cli`.

**P1-4 (G1-4) Notice a file that went away or changed.** `Photo` keeps `(len, modified)` from the probe. `Session::raw_image`, `develop` and the export's `load_raw` compare them before they decode (one `stat`); a mosaic already in the cache is still served.

- Gone: the photo becomes `Skipped` with the `NotFound` sentence above, and a `photo_progress` error event is sent, so the filmstrip shows CAN'T OPEN and the cell shows the reason.
- Changed: "This file changed after it was opened. Remove it from the filmstrip and open it again." The photo is `Skipped` the same way.
- An export that meets either states the same sentence for that photo and goes on.
- Tests: `crates/ym-core/tests/session.rs` (delete, truncate and rewrite a file after `settle`; assert the status, the event and that a photo whose mosaic is cached still renders until it is evicted); an `export_queue.rs` case (a file deleted between queueing and its turn: the other photos are written); the mock gets `mockVanish=DSC04207` for a Playwright case.

**P1-5 (G1-6)** Say it in the user guide (M4-03) and in `docs/04`: a raw file has no checksum, so damage inside pixel values is not detected; `ym` and the app render what is there. No code.

**P1-6 (G1-7) The damage matrix.** Add `ym_raw::testgen::damage` (feature `testgen`): `Damage::{Empty, Truncate(frac), Header(bytes), Garbage(offset_frac, len), Zeros(offset_frac, len), BitFlip(seed), Text, Jpeg}` applied to the bytes of a valid synthetic file of each variant. Three tests use it:

1. `ym-core/tests/damaged_files.rs`: variants × damages in one folder; `Session::open` + `settle`; every damaged file is `Skipped` with a catalogue sentence or, for damage in pixel values, `Ready`; the good files are `Ready` and unchanged; no panic message in the captured log.
2. `ym-cli/tests/damaged_files.rs`: the same folder through `ym render`; exit 1; the good files are in the output; stderr has one catalogue line per bad file; `--json` has `error.kind` in `{not_arw, corrupt, io}`.
3. `src-tauri/src/api/tests.rs`: `open_paths` on the folder, then `export_start` for all photos; the job summary counts the failures and the good files are written.

The sweep of this audit (192 files) is the starting content. Permission and symlink cases are Unix-only tests that skip as root.

**P1-7 (G1-8) A slow-media watchdog** is task H-13. It needs a Mac and a real network share to tune (see the Mac list).

### 1.4 Only a Mac can check

- The EPERM sentence on a real macOS privacy block (an SD card in a card reader, `~/Desktop`, `~/Documents`).
- A file on a disconnected external drive and on a slept SMB share: the sentences, and that the app stays usable (G1-8).

---

## 2. Disk full during export

### 2.1 Today

**The write path** (`docs/09:125-128`; `crates/ym-export/src/naming.rs`, `request.rs`):

1. `ExportDestination::resolve` runs **before** the photo is rendered (`ym-core/src/export.rs:1177-1178`): it creates the folder with `create_dir_all` and creates and removes a probe file (`naming.rs:193-203`). A failure is `ExportError::Io { path: <folder> }`.
2. The picture is rendered. On the CPU path the whole picture is rendered before the writer starts; on the device path the writer pulls rows while the tail renders them (`docs/09:325`).
3. `write_atomically` creates `.yellowmoon-<pid>-<n>.tmp` in the destination folder (`naming.rs:186`, `317`), writes through a 1 MiB `BufWriter`, flushes, `fsync`s, then renames it or hard-links it (`naming.rs:358-427`, `471-490`). `Drop` removes the temporary file on every way out, a panic included (`naming.rs:430-437`).
4. A failed Replace leaves the old file. Skip and Keep both never overwrite (`naming.rs:380-420`).
5. Encoder I/O errors (JPEG, PNG, TIFF, and the TIFF encoder inside the DNG writer) become `ExportError::Write { format, source }`; a DNG says "DNG" (`ym-export/src/error.rs:32-37`, `75-86`, `99-129`).

**Messages today.** `Io`: "Couldn't write to {path}: {os text}. Check that the folder exists, is writable and has free space." (`error.rs:20`). `Write`: "Couldn't write the PNG: No space left on device (os error 28). Check that the folder exists, is writable and has free space." The same sentence covers a full disk, a missing folder and a read-only folder. When `flush` or `fsync` fails the path in the sentence is the **hidden temporary file** (`naming.rs:371-376`, `487`), for example `.yellowmoon-4123-0.tmp`.

**What the export queue does after a failure.** It records the failure and goes on with the next photo (`export_queue.rs:975-1004`). The decode of the next photo is started before the current one is finished (`spawn_load`, `export_queue.rs:825`, `857`) and happens **before** the destination of the photo being checked (`export.rs:1178`).

**Measured on this machine, `ym render --cpu`, ten 3000 × 2000 photos, destination a 3 MiB `tmpfs`:**

| Format | Space | Result | Time | Left in the folder |
|---|---|---|---|---|
| JPEG q92 | 279,326 bytes each, 2.7 MB in all, fits | 10 written, exit 0 | 87.4 s | the 10 files (the control run) |
| PNG 16-bit | each file is larger than the disk | 10 failed, exit 1 | **86.7 s** | nothing |
| TIFF 16-bit | the same | 10 failed, exit 1 | **84.4 s** | nothing |
| DNG (2 photos) | the same | 2 failed, exit 1 | 34.3 s | nothing |

Every failing photo cost the full render (about 8.7 s of CPU for a 6 MP photo) and then failed with the same sentence ("Couldn't write the PNG: No space left on device (os error 28). Check…"), ten times, and the run ended with "10 photos failed." The partial files are removed. Nothing tells the person that the next photo will fail too.

**Other destinations, measured:**

- **Read-only folder** (`chmod 555`): two photos failed in 0.2 s with "Couldn't write to {folder}: Permission denied (os error 13). Check that…" once per photo. They are fast because the check comes before the render, but the next photo's raw is still decoded first (see above): for 2000 lossless-L files that is the 0.95 CPU seconds per 50 MP of `docs/04:166-175` for nothing.
- **Folder removed during the run** (`rm -rf` after the first file): photo 2, whose render had already finished, failed with "Couldn't write to {folder}: No such file or directory (os error 2)". Photos 3 and 4 re-created the folder (`create_dir_all` at `naming.rs:194`) and were written. The job ended "Exported 3 photos … 1 photo failed." So a vanished folder is silently re-made, and the render of the photo that met it is wasted. On macOS an unmounted `/Volumes/Name` cannot be re-made (`/Volumes` is root-owned), so a drive pulled during a batch will behave like the read-only case, with a "Permission denied" sentence: **not measured, Mac-only**.

**What the person sees.**

- *App.* The Export panel lists one row per file with the state FAILED and the reason as its second line (`ExportQueueList.tsx:45-52`, `src/export/run.ts:150-155`). When the run ends the title is "Not exported" if nothing was saved (`run.ts:117-121`), and the line reads "N photos could not be saved." (`run.ts:134`). **The header cell reads "Exported" for any finished run, even one that saved nothing** (`run.ts:158-161`, used at `Photos.tsx:211`). If the panel is closed there is no other signal: the export runs in the background and no notice or sound is raised when it ends (`store.ts`: `exportStopped` exists only for Cancel). The mock's example reason is "The disk is full." (`e2e/export.spec.ts:231`); the real reason is the long sentence above.
- *`ym`.* One stderr line per failed photo, a final "N photos failed.", exit code 1, which is also the code when only some failed (`render.rs:952-962`). `--json` gives `error.kind = "write"` for every write failure (`ym-core/src/export.rs:188`), so a script cannot tell a full disk from a read-only folder.

**Tests that exist.** `crates/ym-export/tests/disk_full.rs`: an in-memory sink that fails with ENOSPC after N bytes, for PNG, TIFF, JPEG and DNG, asserting the sentence. `ym-export/src/error.rs` tests: the advice is in the text. `crates/ym-core/tests/export_queue.rs`: queue behaviour. **There is no test of the queue or of `ym` with a failing disk**, and none of a destination that vanishes.

**The app's own writes.**

| Write | Behaviour | Source |
|---|---|---|
| Looks import (`looks/` under Application Support) | Staged files, then renamed; a failure rolls back new files and removes temporaries; the manifest is written last, atomically (`atomic_write`: temp, `fsync`, rename, `fsync` of the folder). Good. The sentence is "Could not read or write {path}: {os text}" with **no next step** and "Could not" against the UI's contraction style. | `ym-looks/src/import.rs:869-897`, `198`; `manifest.rs:410` |
| Preferences | `atomic_write`; "Couldn't save your preferences to {path}: {os text}. Check that the folder is writable and has free space." Raw OS text. | `preferences.rs:259`, `656-658` |
| Application log | Rotating file, 5 MB, three kept; write errors are not shown. | `src-tauri/src/logging.rs` |
| GPU pipeline cache (`~/Library/Caches/app.yellowmoon`) | Written only when the adapter has `PIPELINE_CACHE` (Vulkan today, so not on Metal); temp then rename, temp removed on error, errors ignored. | `ym-gpu/src/device.rs:1030-1046` |
| Depth and SAM caches | None on disk: "in memory only" (`docs/23:481,992`). | |
| Updater | The Tauri plugin downloads and installs; a quiet failure is silent and the update is offered again; "Update failed" with the plugin's text on hover. Not checked here. | `src/updater.ts:126-127` |
| Orphans | A crash or power loss leaves `.yellowmoon-<pid>-<n>.tmp` in the export folder and `.<name>.tmp-<pid>-<n>` in `looks/`. **Nothing removes them** (no code names them outside the writers). | `naming.rs:186`, `manifest.rs:347` |

### 2.2 Gaps

| # | Gap |
|---|---|
| G2-1 | **A destination-wide failure (disk full, read-only, folder gone) does not stop the job.** Each remaining photo is decoded and rendered before it fails. Measured: 10 photos, 87 s of work, 0 files. For 2000 photos that is the whole export. |
| G2-2 | **The raw of the next photo is decoded before the destination is checked**, so even a dead destination costs a decode per photo. |
| G2-3 | **The three causes share one sentence** and it prints "(os error 28)". "Check that the folder exists, is writable and has free space" asks three questions and answers none. |
| G2-4 | **N identical rows and N identical stderr lines**; the header says "Exported" when nothing was saved; nothing is raised when a background export ends badly. |
| G2-5 | No warning before the export when the estimated size does not fit in the free space. The estimate exists (`ExportEstimate.bytes`, the button shows "≈ 630 MB"); free space is never read (no `statvfs` or `available_space` in the repository outside the `fs4` lock in `ym-looks`). |
| G2-6 | `flush` and `fsync` errors name the hidden temporary file. |
| G2-7 | A destination that disappears between the check and the write wastes the render and is silently re-made next time (a folder the person deleted is re-created). |
| G2-8 | Orphaned `.tmp` files are never cleaned, in the export folder or in `looks/`. |
| G2-9 | The looks-import sentence has no next step; the preferences sentence prints raw OS text. |
| G2-10 | `error.kind` is `write` for all causes, and exit code 1 is the same for "one file was bad" and "the disk is full". |
| G2-11 | There is no test with a failing disk below the encoder level (queue, CLI, app API). |

### 2.3 Plan

**P2-1 (G2-3, G2-10) Classify write failures.** In `ym-export`, `ExportError::destination_problem(&self) -> Option<DestinationProblem>` looks at the `io::Error` of `Io` and `Write` (`ErrorKind::StorageFull`, `QuotaExceeded`, `ReadOnlyFilesystem`, `NotFound`, `PermissionDenied`, and `raw_os_error` 28, 69, 30, 2, 13, 1, 19, 6): `DiskFull`, `ReadOnly`, `Gone`, `Permission`. `ym-core`'s `ExportError::kind()` returns `disk_full`, `destination` (read-only, gone, permission) or `write` (the rest). The sentences name the folder by its last component in the app, by its full path in `ym`:

| Cause | Message (app) | `ym` (stderr) |
|---|---|---|
| Disk full | "The disk is full, so the export stopped. Free some space, then export the photos that are left." | "Stopped: the disk holding {path} is full. Free some space, then run the command again with --if-exists skip." |
| Read-only | "{Folder} is read-only. Choose another folder." | "{path} is read-only. Choose another folder with --out." |
| Folder gone | "Yellowmoon can't reach {folder}. If it is on a drive, check that the drive is connected, then export again." | "{path} is not there any more. Check the path or the drive, then run the command again." |
| No permission | "Yellowmoon isn't allowed to save to {folder}. Choose another folder, or allow access in System Settings, under Privacy & Security." | "Yellowmoon isn't allowed to write to {path}. Choose another folder with --out." |

The generic `Io`/`Write` sentence loses the question list: "Couldn't write the {PNG}. Check that the folder is still there and that you can save to it." The OS text moves to `detail` (JSON and log). The path in a `flush`/`fsync` failure is the destination folder, not the temporary name (G2-6).

Tests: unit tests for each `io::Error` → class and message; extend `tests/disk_full.rs` to assert the new sentences; a snapshot test of `ym render --json` for each class.

**P2-2 (G2-1, G2-2, G2-4) Stop the job on a destination-wide failure.** In `run_job` (`export_queue.rs:975`), when the error is `DiskFull`, `ReadOnly`, `Gone` or `Permission`, the job stops: no further `spawn_load`, no more renders. The job summary gets `stopped: Option<String>` (the message) and the state `Finished`; the failing photo is `failed`; the others are `not_tried`.

- Contract: `ExportProgress` gets `stopped: bool` on the event of the failing photo; `FileDone` and `JobSummary` get `not_tried: usize` (additive, so `bindings.rs` and `src/api/contract.ts` change together).
- Order: **the destination of photo i is resolved before photo i is decoded**, by moving the `ExportDestination::resolve` and a free-space read ahead of `spawn_load` for that photo (G2-2). The name is known from the metadata (`export_queue.rs:855-865`). The decode of the next photo is not started once the job has stopped.
- Panel: one banner above the list, in the panel's error style, with the message of P2-1; the rows after the failing one read "Not exported" (state `skipped` with its own note: "Not exported, the export stopped."); the title is "Not exported" when nothing was saved and "Stopped" when some were; the blurb reads "{k} photos saved to {folder}. The export stopped: {message}". The header cell reads "Export stopped" (and "Not exported" when `saved = 0`).
- `ym`: one error line (the message of P2-1), then "Stopped after {k} of {n}. {n − k} not exported." Exit code stays **1** ("some files failed"); `--json` gets `{"event":"stopped","kind":"disk_full","exported":k,"not_tried":m}`. (Question 1 below asks whether the owner wants a code of its own.)
- A failure that is about one photo (a decode error, a look that changed, a mask this computer cannot make) still does not stop the job.
- Tests: `export_queue.rs` with a destination that becomes read-only after the first photo (Unix; skipped as root) and a fault hook (`fault-injection` feature, `YM_WRITE_FAULT=enospc@<bytes>` in `PendingFile`, like `YM_GPU_FAULT`) so that the test runs anywhere: assert that photos after the failing one are neither decoded nor rendered (the session's `raw_decodes` counter, `session.rs`, and the renderer's counters), that no temporary file is left, that the summary and events are as above. An `ym-cli` test for the stderr lines, the JSON event and the exit code. A Playwright case with the mock (`mockExportFails=DSC04204:disk_full` marking a stop) for the banner, the "Not exported" rows, the title and the header cell. A Vitest case for `applyProgress` and `cellLabel`.
- CI: a Linux job `disk-full` mounts a 3 MiB `tmpfs` (`sudo mount -t tmpfs`), runs `ym render` of five synthetic photos to PNG and asserts exit 1, the stderr, an empty folder and a wall time under one photo's render (the measured waste is the thing to catch).

**P2-3 (G2-5) Warn before, never block.** The estimate's own error is documented: 90 % of the real renders within 35 % at q92 (`docs/09:149-160`), and with noise reduction off the files are 1.1 to 2.7 times bigger. A hard refusal on the estimate would refuse exports that fit, so the check warns and the stop of P2-2 is the exact one. When `free < 1.35 × estimate` (the documented 90th-percentile error) the panel shows under the button: "This export needs about {size} and {free} is free in {folder}. It may stop part way." The same sentence goes to stderr in `ym` before the run. The free space comes from `fs4::available_space` (`fs4` is already in the tree for `ym-looks`; licence MIT or Apache-2.0, `docs/14`), read when the options or the destination change and again when the job starts. Tests: a unit test for the threshold; a Playwright case (`mockFreeBytes=`); an `ym-cli` test on the `tmpfs` job.

**P2-4 (G2-7, G2-8) Temp-file hygiene.** At the start of a job the destination folder is scanned for files named exactly `.yellowmoon-<digits>-<digits>.tmp` older than 24 hours and they are removed (a crash leaves them, `Drop` never ran). The same for `.<name>.tmp-<digits>-<digits>` in `looks/` when the library loads. A folder the person deleted during the run is **not** silently re-made after the first photo: `create_dir_all` is used for the first photo of a job only, the later photos use `ensure_dir_exists` (fail with the "Folder gone" sentence). Tests: files with `File::set_modified` 25 hours ago are removed, those 23 hours old are not; a name that does not match is untouched; the second photo meets a removed folder and the job stops with "Folder gone".

**P2-5 (G2-9) The app's own writes.** `ImportError::Io` → "Yellowmoon couldn't save the looks to {folder}. Check that you can save there and that the disk has free space, then import again. Nothing was changed." (`import.rs:198`, the rollback already makes the last sentence true); the preferences sentence loses the OS text and reads "Yellowmoon couldn't save your preferences. Check that the disk has free space and that you can save to {folder}." Tests: the existing import tests with a read-only `looks/` and the preferences test at `preferences.rs:1342` assert the new text; the full-disk case runs in the `disk-full` CI job (import a three-look folder into a 64 KiB tmpfs).

### 2.4 Only a Mac can check

- APFS behaviour: ENOSPC may come at `write`, at `fsync` or at `rename`; the temp file must be gone in each case (the code removes it in each; the sentence for `fsync` must name the folder).
- An external drive pulled during a batch: the sentence (P2-1 "Folder gone" or "No permission") and that the run stops at once.
- Quota (EDQUOT, "Disk quota exceeded") on a network home folder.
- The updater on a nearly full disk (the plugin's text, and that `Update failed` is shown).

---

## 3. Folders of 2000 files

### 3.1 Today

**Opening** (`docs/01:135-142`):

- A folder is walked one level deep (`session.rs:943`); `*.ARW` in any case; dot files ignored. The walk uses `read_dir` + `symlink_metadata` + `is_file` per entry.
- `Session::open` canonicalises every path, **while it holds the session's state lock** (`session.rs:2224-2236`), sorts the files, probes them in parallel with rayon (`session.rs:2243-2246`), registers them all, and only then returns. The probe reads at most 64 MiB per file (`tiff.rs:32`); `ym info` reads a 1925 KB file with 123 `pread` calls (`strace -c`: probe, decode and estimates together).
- The app's command is `async` and runs on a blocking thread (`src-tauri/src/commands.rs:74-79`), so the window does not freeze. The store sets `opening` (`store.ts:1232`), which only disables the Open buttons (`Start.tsx:102-105`, `Photos.tsx:135`). **There is no "Reading 2000 files…" text, progress or cancel while the probes run.**
- Then one background job per photo (`session.rs:3787` `enqueue`; `take_job`, `session.rs:3874`: the current photo's develop first, then the oldest) on `min(cores − 2, 6)` threads, at least 1 (`session.rs:753-756`). A job decodes, computes the estimates and renders a 240 px thumbnail (`session.rs:3902-4090`).
- The mosaic cache holds 3 decoded mosaics (`MOSAIC_CACHE_LEN`, `session.rs:134`); a mosaic is 122 MB at 61 MP (`docs/01:164`). A thumbnail job drops its mosaic when it is done if the photo is outside the window.
- Two estimates depend on all photos: the burst median exposure and the pooled noise model. `refresh_effective` runs under the lock after every photo that finishes and walks all photos that have estimates (`session.rs:1392-1460`: clones the burst frames, groups, shares). That is O(n) per photo and O(n²) in all.
- Cancelling: `close()` bumps an epoch; queued jobs are dropped and running ones stop at their next stage; thumbnails check the epoch every 25 ms (`pipeline/thumbs.rs:1177-1191`). Opening a second folder does not cancel the first; it adds photos.

**Thumbnails and IPC** (`docs/01:868-891`):

- `thumbnail_ready` carries an address, not bytes. The shell keeps the PNGs (`src-tauri/src/viewport.rs:209-217`): the newest two versions of each photo (`THUMBNAIL_VERSIONS_KEPT = 2`, `viewport.rs:396`) and **no byte budget**. Look tiles have 256 MiB (`viewport.rs:388`) and node thumbnails 64 MiB (`viewport.rs:380`).
- The encoder is 8-bit RGB, 240 px on the long edge (`pipeline/thumbs.rs:130-138`, `graph.rs:383`), so a thumbnail cannot be larger than 240 × 240 × 3 = 172.8 KB plus a few hundred bytes of PNG framing (115.2 KB for 3:2). That is a **bound computed from the encoder, not a measurement of real photos** (there are none here). `docs/01:870` says "about 250 KB", which an 8-bit 240 px picture cannot reach; the number needs re-measuring (H-12). At the bound, 2000 photos × 2 versions hold at most 691 MB in the shell; at 115.2 KB and one version, 230 MB.
- `photo_progress` is coalesced to 30 Hz, latest per photo (`docs/01:887-891`). The 500-file test counts 3,500 progress events, 500 thumbnails and about 5,100 to 5,600 commands (`docs/11:767`); at 2000 that is about four times as many by proportion (not measured).
- The thumbnail `<img>` has no `loading="lazy"` and no `decoding="async"` (`Filmstrip.tsx:213`).

**The front end** (`docs/11:450,762-765`): "The filmstrip draws every photo (500 buttons is fine)". `Filmstrip.tsx:262-302` maps all photos to buttons. The Start list and the Export queue also draw every row (`ExportQueueList.tsx`, `DevelopList`). A status change copies the photo list: `withStatus` does `some()` then `map()` over all photos (`store.ts:1006-1020`), a thumbnail copies the thumbnail map (`store.ts:1028-1036`); both run in batches (`later`, `store.ts:837-852`).

**Export of 2000** (`export_queue.rs`): `ExportQueue::start` snapshots every photo's settings into `Job.items` (memory proportional to the count, each snapshot a few settings and an `Arc` of the estimates; not measured). Only 16 job summaries are kept (`KEPT_SUMMARIES`, `export_queue.rs:351`). Files are written one at a time; the next photo's raw is decoded while the current one renders. `applyProgress` copies all rows on every event (`run.ts:47-80`).

**Measured (this machine).**

*Core and `ym`, 2000 synthetic 192 × 128 files (54 KB each, 110 MB in all), CPU path:*

| Run | Wall | Peak RSS |
|---|---|---|
| `ym render` of 1 file to a read-only folder (open, develop, fail) | 0.2 s | 52 MiB |
| 99 files, same | 0.2 s | 56 MiB |
| 2000 files with 2000 different capture times | 2.8 s | 146 MiB |
| 2000 files with the same capture time (one burst of 2000) | 2.5 s | 146 MiB |
| 2000 files, JPEG export to a writable folder | 91.5 s (89 s of it exporting, 45 ms a photo) | 154 MiB |
| 6 × 61 MP files to a read-only folder (2 decode threads here) | 1.5 s | 632 MiB |

So listing, probing, developing and the per-photo shared estimates show **no quadratic cost at 2000 tiny files, even for one burst of 2000**, and the session costs about 47 KiB a photo in `ym` ((146 − 52) MiB / 2000, errors included). These files are tiny: the decode and render cost is not representative, and a Mac with real 61 MP files develops far slower per photo. What the table shows is the bookkeeping.

*Web view, production build, mock API, Chrome headless (rows of one table are comparable; see "How the numbers were made"):*

| Photos | Open files → "Developing N" list | → filmstrip | DOM nodes (Start list) | DOM nodes (filmstrip screen) | Key latency while all are developed (median / p95 / max) | Long tasks (count, total) |
|---|---|---|---|---|---|---|
| 100 | 117 ms | 452 ms | 489 | 368 | 49 / 66 / 69 ms | 1, 0.1 s |
| 500 | 213 ms | 625 ms | 2,089 | 766 | 86 / 115 / 122 ms | 63, 4.6 s |
| 2000 | 312 ms | 1,010 ms | 8,088 | 2,267 | **254 / 359 / 382 ms** | 63, 14.2 s |

Key latency while the photos were still developing (mock speed 10): 500 photos 81 / 123 / 143 ms, 2000 photos 227 / 296 / 329 ms. The cost per key rises with the photo count at about 0.1 ms a photo. The Export panel with 2000 picked photos has 6,131 DOM nodes before the run starts and 16,063 while it runs (2,000 queue rows); with 500 photos 1,631 and 4,063. "Pick all" took 233 ms for 2000 (53 ms for 500). The mock's own export took 169 s for 2000 (85 ms a photo) and 27.5 s for 500 (55 ms a photo): the event handling grows with the row count. `applyProgress` alone, timed in Node on a copy of `run.ts`: 1.1 s for the 40,000 events of a 2000-photo run (28 µs each, 80 million row copies; 0.12 s for 500).

The page's JS heap was 22 to 36 MB at 2000 (Chrome `performance.memory`); the decoded thumbnails of WKWebView are not measured.

### 3.2 Gaps

| # | Gap |
|---|---|
| G3-1 | **Every key press re-renders every thumbnail.** Latency is 254 ms median at 2000 with all photos developed, 2.9 times the 500-photo figure of the same run, and the M3-03 target is 100 ms. The Start list and the queue list draw every row too (8,088 and 16,063 nodes). |
| G3-2 | **The shell's filmstrip thumbnails have no byte budget** (bound 691 MB at 2000 with two versions, 230 MB with one at 3:2) and the webview holds an `<img>` for each with no lazy decode. |
| G3-3 | `Session::open` holds the session lock while it canonicalises every path (`session.rs:2224-2236`): on a slow volume, other commands wait. |
| G3-4 | No progress and no cancel while 2000 files are probed; the Open buttons just go grey. |
| G3-5 | A hung file blocks `open_paths` (see G1-8). |
| G3-6 | `docs/01:870` quotes a thumbnail size that the encoder cannot produce. |
| G3-7 | `applyProgress`, `withStatus` and the thumbnail map are O(n) per event. Measurable but small (1.1 s in all for the largest run); worth fixing together with G3-1, not alone. |
| G3-8 | No test or CI measurement at 2000 for the core, the shell or the web view; the 500-file jobs have fixed 500 (`src-tauri/src/api/tests.rs:4251`, `e2e/perf/latency.perf.ts`). |
| G3-9 | `refresh_effective` is O(n) per finished photo under the session lock. Harmless at 2000 tiny files; with real photos it is invoked rarely (one per decode). Listed for completeness, no change planned. |

### 3.3 Plan

**P3-1 (G3-1, G3-7, G3-2) Draw only what is on screen.** The filmstrip, the Start list and the queue list use a windowed list: only the rows inside the viewport plus a margin are in the DOM. The filmstrip scrolls sideways; its width is `n × cell` so the scroll bar is true. Row components are memoised by value (`memo` with a comparison of `id`, `status`, `src`, `picked`, `nodes`), so a key press re-renders two thumbnails and the caption. `<img loading="lazy" decoding="async">`. `applyProgress` and `withStatus` change one row in place (a copy-on-write array of rows with a version counter) instead of copying all.

- The component is our own small one (`src/components/windowed.tsx`, no new dependency; `react-window` and `@tanstack/virtual` are MIT, but the lists are simple). Keyboard focus, `aria-label`s, `aria-current`, the reveal-on-select scroll and the context menu must keep working: the focused or current row is always rendered.
- Test and gate: `e2e/perf/latency.perf.ts` takes `YM_PERF_PHOTOS` (default 500). A new CI step (weekly job `latency-500` becomes `latency`) runs 500 and 2000 in one run and asserts **p95 at 2000 ≤ 1.25 × p95 at 500 measured in the same run** (a ratio, so a slow runner does not matter), plus the 150 ms bound at 500 that exists. Vitest: a `Filmstrip` with 2000 photos mounts fewer than 100 buttons; arrow keys, P and the context menu still work on a photo that was off screen. Playwright: `e2e/photos.spec.ts` with `mockPhotos=2000` counts DOM nodes below a fixed number (for example 1,500).

**P3-2 (G3-2, G3-6) A budget for thumbnails.** First re-measure: write the size of the thumbnail PNG of real photographs and of the synthetic set (`ym-core` test with `--nocapture`, owner photos in CI) and correct `docs/01:870`. Then add a byte budget to the shell's thumbnail store (like the look tiles': oldest-used photos lose theirs when the budget is passed, the current photo never) and make a lost thumbnail come back: the webview asks `thumbnails()` for the addresses it has and the core re-renders the missing ones at `Priority::Thumbnail` when they come on screen (the windowed list of P3-1 tells which). The budget value is chosen from the measurement. If the measurement shows typical thumbnails near 60 KB, the store may stay without a budget and only the count of versions per photo drops to 1 for photos not on screen. Test: `src-tauri` unit tests with a small budget (put, serve, evict, re-render request); the 2000-file Rust test of P3-3 asserts the store stays below the budget.

**P3-3 (G3-3, G3-4, G3-8) The open path.**

1. Canonicalise outside the lock, in the parallel map, then take the lock once to compare (the result is the same).
2. `Session::open` takes an optional `OpenProgress` sink (`probe` done/total, counted per 50 files). The app sends it as `open_progress { done, total }`, coalesced like `photo_progress`; the Start and Photos screens show "Reading {done} of {total} files…" in place of silence, with a Cancel button that calls `cancel_open` (an epoch bump, as `close()`: the probes still running finish their file and are dropped). `ym` prints the same line on stderr when a probe takes longer than 1 second.
3. A new ignored test `crates/ym-core/tests/folders_2000.rs` (and `src-tauri` `two_thousand_files_leave_the_commands_responsive`, the existing test with the count read from `YM_PERF_PHOTOS`) writes 2000 synthetic files with 2000 capture times and with one capture time, and reports: open time, time to all Ready, the slowest command, peak RSS (`getrusage`), the shell's thumbnail bytes. Asserts: slowest command under the bound of the 500 test; RSS under a stated number chosen after the first run. It joins the weekly job.
- Messages: "Reading {done} of {total} files…" and, on cancel, the files read so far stay: "Stopped. {n} files were opened."

**P3-4 (G3-5) Slow media** is H-13 (Mac).

### 3.4 Only a Mac can check

- WKWebView with 2000 thumbnails: decoded-image memory, scroll and key latency (`docs/11` "Large folders" says the native side is Mac work, M3-H4).
- Probe and develop of 2000 real files from an SD card reader, an external SSD and an SMB share: the time to the first thumbnail, the time to all Ready, that Cancel works.
- Memory of the shell with 2000 thumbnails (Activity Monitor).

---

## 4. 8 GB Macs

### 4.1 Today

**Budgets** (`docs/17:89`, `crates/ym-core/src/scheduler.rs`):

- Default CPU budget 6 GiB and GPU 4 GiB (`scheduler.rs:32-36`); `MemoryBudget::detect` lowers the CPU budget to half the RAM (`scheduler.rs:109-120`, `4 GiB` on 8 GB).
- On a device of type `IntegratedGpu` (every Apple-silicon Mac) the GPU budget is `clamp(RAM/4, 512 MiB, 4 GiB)` and the CPU budget `min(cpu, 0.6 × RAM − GPU)` (`scheduler.rs:139-155`, `ym-gpu/src/pool.rs:93`). For 8 GB: **GPU 2 GiB, CPU 2.8 GiB**. `Core` applies it when the device opens and opens again if the GPU number changed (`app/gpu.rs:810-840`). On a machine with no usable GPU the CPU budget stays at half the RAM, 4 GiB. A discrete GPU keeps `min(4 GiB, 2 × max_buffer_size)` whatever its memory (`ym-gpu/src/pool.rs:102-111`).
- STATUS question 4 ("Acceptable on 8 GB Macs?") is still open (`docs/STATUS.md`, Questions for the owner).
- `ym doctor` prints the budget ("Memory budget CPU 6.0 GiB, GPU 4.0 GiB" on this machine).

**What the budgets cover.**

- *Mosaics.* Three decoded mosaics (`session.rs:134`): 366 MB at 61 MP.
- *Render caches.* Byte-bounded LRUs per photo graph: frames and display frames `cpu/32`, reduced groups `cpu/6` (`graph.rs:864`, `906`, `928`, `996`); with resident caches `min(cpu/6, gpu/8)`. Up to `max_photos = 3` graphs are kept (`scheduler.rs:67`), so the sum is up to three times the per-graph figures (reading of the code; not measured).
- *Levels.* Up to `gpu_bytes / 8` per photo, two per photo (`docs/17:304`).
- *The full-resolution render* reserves its working set first in the `MemoryLedger` and is refused if it does not fit (`pipeline/frame.rs:4224-4227`, `export_tail.rs:524`): `output (12 B/px) + working (tile + halo)² × 144 B + frame extras + sigma map (4 B/px)`. A refusal is `RenderError::Memory`: "There is not enough memory to render this photo at full size. Close other apps or export a smaller size." (`pipeline.rs:110-113`). Test: `crates/ym-core/tests/render_api.rs:515-545` sets the CPU budget to 1 MiB and asserts the refusal and a ledger back to 0.
- *GPU.* The pool ledger is the authority; `OutOfBudget` or `BufferTooLarge` trims the pool, retries once, then the CPU renders that photo (`docs/17:91`, `101`). Fault tests: `crates/ym-core/tests/gpu_faults.rs` (`YM_GPU_FAULT=oom`).
- *Models.* SAM 2.1 small: 77 MB file, **154 MB of f32 weights resident**, about 360 MB while encoding (`docs/23:1440`); depth model: 49.6 MB file, 24.8 M parameters (about 99 MB as f32, by the same expansion; not measured here). Both are read whole (`std::fs::read`, `ym-ml/src/sam2/mod.rs:727`, `depth/mod.rs:313`) and expanded to f32, so the load peak is the file plus the weights. They load on first use and **stay loaded for the life of the process** (`ModelState::Ready`, `pipeline/object.rs:58-64`). SAM features are about 16 MB a photo and the photo-frame rasters 4 to 8 MB each, in memory for the three kept photos (`docs/23:481`).
- *Film node.* About 12 B/px for each of E, E′, Q, D and the picture over the grown tile; a 1,024 px tile at 8,000 px reads 2,170 px square (`docs/24:611`), about 280 MB per tile by that count; "the unified-memory budget with the Film images" is listed as a Mac check there.
- *Develop threads.* `min(cores − 2, 6)` (`session.rs:753`): six on an M1 or an M2 with 8 cores, whatever the RAM. A job holds a mosaic and the estimates' scratch.

**Measured (this machine, CPU path; peak RSS = `ru_maxrss`).**

| Run | Wall | Peak RSS |
|---|---|---|
| `ym info` of one 61 MP uncompressed file (probe, decode, estimates) | 0.5 s | 212 MiB |
| `ym render` of 6 × 61 MP files, develop (2 threads) + 3 mosaics + the export's prefetch, failing at the destination | 1.5 s | 632 MiB |
| `ym render --format jpeg` of one 61 MP file (9504 × 6336) | 108 s | **2,483 MiB** |
| `ym render --format dng` of one 61 MP file | 159 s | **2,522 MiB** |

*The ledger's number for the same export*, from the formula at `frame.rs:4224` with tile 1024 and the smallest halo (160 px): 722.6 MB (output) + 240.9 MB (sigma map) + 201.9 MB (working) = **1.17 GB, about 1,110 MiB** (more with a lens halo). The process peaked at 2.2 times that. The rest is mosaics, the reduced group, the estimates, the encoder and the allocator. The ledger therefore tells the truth about the part it counts and about under half the process.

For an 8 GB Mac the CPU budget is 2.8 GiB and the process peaked at 2.4 GiB on the CPU path of this machine, with nothing else running; the app also holds the three mosaics, the render caches of three photos, the thumbnails of the folder, the webview and, on the GPU path, a 2 GiB pool. Whether that fits next to macOS and the person's other apps is the question STATUS 4 asks and is **not answerable on Linux**.

**Allocation failure.** A refused allocation is a Rust abort (`handle_alloc_error`): no unwinding, no message, `catch_unwind` cannot catch it. The only `try_reserve` is the ledger's own (`scheduler.rs:214`, which is a counter, not an allocation). On macOS the practical failure is not `malloc` returning null; it is heavy compression and swap, then the process being killed by the system under memory pressure. The code has no reaction to memory pressure: a search for `memorypressure`, `jetsam` and `os_proc_available_memory` finds nothing (`docs/17:259` lists "unified-memory pressure on an 8 GB Mac" as a Mac check).

**Simulating 8 GB.** There is no way to run the pipeline as an 8 GB machine. `MemoryBudget::for_ram(Some(8 GiB))` and the fake GPU probe test the *rule* (`scheduler.rs:1276-1330`, `app/gpu.rs:1153-1190`: two opens, one device alive at a time, the numbers 2 GiB / 2.8 GiB); `render_api.rs:515` tests a refusal with a 1 MiB budget. Nothing runs a 61 MP export under the 8 GB budget and checks the real peak.

### 4.2 Gaps

| # | Gap |
|---|---|
| G4-1 | The ledger counts 1.1 GiB of a 61 MP export that peaks at 2.4 GiB (CPU path). The budget cannot protect what it does not count: mosaics, the reduced group, caches and the develop pool are outside it. |
| G4-2 | The develop pool is `min(cores − 2, 6)` whatever the RAM. At 61 MP each job holds a mosaic (122 MB) plus scratch (the single-file figure above is 212 MiB). On an 8-core 8 GB Mac that is 1.3 GiB transient (212 MiB × 6; an estimate from the single-file figure, not measured) on top of everything else. |
| G4-3 | Models are never unloaded: SAM 154 MB plus depth about 99 MB resident, plus 200 MB while the encoder runs, even after the person left the node view. |
| G4-4 | No reaction to memory pressure on macOS. |
| G4-5 | No way to run the app or `ym` "as an 8 GB machine", and no test of the 61 MP export, the DNG, the Film node or the two models together under that budget. |
| G4-6 | The three-graph cache sum is not bounded by one number; it was not measured. |
| G4-7 | A refused allocation is an abort. `RenderError::Memory` covers only the one full-render reservation. |
| G4-8 | STATUS 4 is open, so the 8 GB numbers are provisional. |

### 4.3 Plan

**P4-1 (G4-5) A RAM override for tests and diagnosis.** `YELLOWMOON_RAM_GB=<number>` is read by `total_ram_bytes()` (`scheduler.rs:165`) and so by `MemoryBudget::detect`, the unified-memory rule and `GpuOpener`'s `ram_bytes` (`app/gpu.rs:730`); `ym doctor` prints "RAM 8 GiB (set by YELLOWMOON_RAM_GB)" when it is set. It is a development switch like `YELLOWMOON_GPU` and is documented in `docs/17` section on environment variables, not in the user guide. Tests: a unit test of the parse (rejects 0, text, over 1024); an `ym-cli` test that `ym doctor` shows 8 GiB and the budget line of the rule (CPU 4.0 GiB without a GPU; with the fake GPU probe 2.0 and 2.8).

**P4-2 (G4-1, G4-5) A peak-memory gate.** A CI job `memory-8gb` (Linux, release): `YELLOWMOON_RAM_GB=8`, a synthetic 61 MP and a 50 MP file, `ym render` to JPEG, TIFF 16-bit and DNG, with the Python wrapper of this audit (`resource.getrusage`) or `prlimit --as`. The gate **records** the peak RSS and fails if it is above a bound chosen from the first run on the CI machine (the audit's figure on a 4-core VM is 2,483 MiB for JPEG and 2,522 MiB for DNG; the bound is those plus the machine's noise, not a guess). It also asserts that the ledger's `peak()` (printed today only by `ym doctor --bench` as `host_ledger_peak_bytes`, `crates/ym-cli/src/doctor/bench.rs:446`; `ym render -v` prints it too after this task) is within a factor of two of the RSS, which fails when somebody adds a large buffer outside the ledger. The ledger is then corrected (P4-3).

**P4-3 (G4-1, G4-2) Count what is held.** The ledger reserves the mosaic of each running develop job and each decoded mosaic in the cache (`MOSAIC_CACHE_LEN × 2 B × w × h`) and the reduced group; the `RenderError::Memory` check in `frame.rs:4226` then sees them. The develop pool size becomes `min(cores − 2, 6, max(1, cpu_budget / job_bytes))` where `job_bytes` is the 61 MP figure (`2 × w × h + scratch`), so 8 GB gives three threads and 16 GB six. A job that cannot reserve waits (it does not fail). Tests: `default_decode_threads_for(cores, ram)` table test; `render_api.rs`-style test with a budget of 1.5 × one mosaic: two develop jobs never run together; the peak gate above.

**P4-4 (G4-3) Unload idle models.** `ObjectModel` and `DepthModel` move from `Ready` to `Untried` when they have not been used for five minutes or when asked to trim; the next use reloads in about half a second (SAM: "the weights load in 0.5 s", `docs/23:1440`). `trim_caches()` on `Core` is the one entry that P4-5 calls. Test: unit tests with the fake model (`YELLOWMOON_TEST_DEPTH=fake` style) and a clock hook: after the idle time `sam()` loads again and the count of loads is 2; a use during the idle time resets it.

**P4-5 (G4-4) Memory pressure (macOS).** A thread polls `sysctl kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical) every two seconds (`libc`, MIT or Apache-2.0; no Swift needed). Warning: drop the mosaics of the neighbours, the finished-frame caches of photos not on screen, and the models (P4-4). Critical: also trim the GPU pool, stop prefetch and thumbnail jobs until the level falls, and show once, in the error bar's notice style: "Your Mac is short of memory, so Yellowmoon paused background work. Close other apps to speed it up again." The sentence goes away when the level is normal. Tests: the poll reads a function pointer (a test injects levels): warning drops exactly the listed caches; critical pauses the jobs; the notice appears once. **The level source is Mac-only.**

**P4-6 (G4-7) Refusals that can happen.** The big allocations that depend on a file's claimed size (the mosaic plane, the M/S RGB plane) use `Vec::try_reserve_exact` and return `RawError::Corrupt` (detail "too large") instead of aborting. This protects Linux and tests (`ulimit -v`) more than macOS; it is cheap, and the hostile test already bounds the sizes. It is listed so the owner can drop it: **the real 8 GB protection is P4-3 to P4-5, not this.**

**P4-7 (G4-8)** Answer STATUS question 4 from the Mac check MC-3 below: the peak footprint of the app (Activity Monitor "Memory" and `footprint`) at 61 MP with the Film node and an Object mask, on an 8 GB Mac.

### 4.4 Only a Mac can check

See MC-3 to MC-5 in the last section. In short: the peak footprint of the app on an 8 GB Apple-silicon Mac for the 61 MP photo with Look, Film node, Object and Depth masks, an export of 50 and 61 MP in each format, a folder of 100 61 MP files developing, and the macOS memory-pressure state during them.

---

## 5. Intel Macs

### 5.1 Today

**The build.**

- The release builds `ym` for `aarch64-apple-darwin` and `x86_64-apple-darwin`, merges them with `lipo`, and builds the universal app with `tauri build --target universal-apple-darwin`; `MACOSX_DEPLOYMENT_TARGET` is 14.0 (`docs/12:Release pipeline` steps 4 to 6, `.github/workflows/release.yml:248-265`, `267-286`). `build.yml` does the same by hand (`build.yml:166-191`).
- The updater manifest gives the same tarball for `darwin-aarch64` and `darwin-x86_64` (`release.yml:324-337`).
- **The x86_64 slice is never run.** "Check the bundle" runs `lipo -info` (it prints, it does not assert) and `"$APP/Contents/MacOS/ym" --version` (`release.yml:298-305`) on an Apple-silicon runner, which picks the arm64 slice. Every macOS runner in the repository is Apple silicon: `macos-latest`, `macos-14`, `macos-15`, `macos-26`, `xcode-27` (`ci.yml:79,816`, `mac-gpu.yml:65`, `apple-bridge.yml:229`, `build.yml:28,226`). There is no `arch -x86_64` and no Intel runner anywhere. The Linux x86_64 jobs run the same Rust code for the CPU parts (and `expected.json` of the goldens was written on Linux x86_64, `ci.yml:366`), which is not the same binary, libm or ABI.

**Code that depends on the architecture.**

| Part | What it does on x86_64 | Source |
|---|---|---|
| CPU ML runtime | AVX2 + FMA kernels chosen at run time with `is_x86_feature_detected!`, scalar otherwise | `ym-ml/src/cpu_runtime.rs:233-271,358-395`, `sam2/kernels.rs:235-275` |
| Lens SIMD | x86_64 and aarch64 paths, scalar for the rest | `ym-lens/src/simd.rs:8,65,122` |
| Apple Auto | Swift bridge only for `aarch64` with the macOS 27 SDK; Intel reports `intel_mac` ("this Mac has an Intel processor") | `ym-apple/build.rs`, `ym-apple/src/lib.rs:63-68`, `apple_auto.rs:367,408` |
| Subject and People masks (Vision) | The Vision bridge is compiled for **both** architectures; a probe asks once whether the request runs and hides the kinds that do not (`docs/23:989`: "The Intel slice of the universal binary may fail the probe (then the kinds are hidden there)") | `ym-apple/build.rs`, `ym-core/src/vision.rs:8-14,259-263` |
| Claude Code and Antigravity | Programs found on the machine; independent of the architecture | `docs/22` |
| GPU | wgpu on Metal; `HighPerformance` power preference (`device.rs:263,547`); no required features; storage is f32 whatever the adapter (`device.rs:597`); K1 to K3: baseline WGSL, 8 storage buffers, 16 KiB workgroup memory (`docs/17:62-66`); `request_device` failure or no adapter gives the CPU and a reason | `ym-gpu/src/device.rs:539-720` |
| Budget | An Intel iGPU is `IntegratedGpu` and gets the unified rule (`RAM/4`); an AMD discrete GPU is `DiscreteGpu` and gets `min(4 GiB, 2 × max_buffer_size)` | `pool.rs:93-111`, `device.rs:709-714` |
| ML demosaic | Not shipped yet (the classical demosaic runs everywhere; "No trained demosaic model is installed yet") | `ym doctor` |

**How an Intel Mac is told.**

- *Apple Auto:* the Preferences row for it is shown only "while available" (`docs/11:396`); the owner decided that the dropdown and its prompt are hidden when no provider is available (`docs/STATUS.md`, Auto providers, decision 1). So an Intel Mac with no Claude Code or Antigravity sees nothing, and one with them sees Default Auto and those programs. `ym doctor` and `ym info` say "this Mac has an Intel processor" (`docs/19:224`).
- *Vision masks:* hidden kinds (`src/state/store.ts` `maskKinds`), with a sentence for a refused request ("Node 2 has a Subject mask, and this computer cannot make one…", `docs/10:245`).
- *No usable GPU:* the CPU renders and the photo cell shows "Rendering on the CPU".
- *macOS older than 14:* the cask has `depends_on macos: ">= :sonoma"`; the bundle's `minimumSystemVersion` is 14.0 (`docs/12`).

**Tests.** `ym-apple/src/lib.rs:240-270` checks that Intel reports `IntelMac` *in a build for Intel*, which no job produces. The unified-budget tests run with a fake probe (`app/gpu.rs:1153`).

### 5.2 Gaps

| # | Gap |
|---|---|
| G5-1 | **The x86_64 slice has never been started in CI.** A slice that does not launch (a missing dynamic library, a Swift runtime path, an illegal instruction) would be found by the first person with an Intel Mac, after a release. The `lipo -info` line prints but does not assert both architectures. |
| G5-2 | No native Intel run of `ym render` on a golden file, so nothing shows that the CPU path gives the pictures of the Linux references within the golden tolerance (ΔE00 0.5, `tools/golden/README.md`). |
| G5-3 | The GPU budget of a **discrete AMD GPU** ignores its video memory; the budget of an **Intel iGPU** follows the unified rule though its Metal working set may be smaller than `RAM/4`. Neither was measured; wgpu does not expose Metal's `recommendedMaxWorkingSetSize`. A pool that thinks it has more than the GPU can hold fails in the driver, which taints the device, the device is reopened at most twice and then the CPU renders (`docs/17:101`): correct but slow, and it costs the person time before it works. |
| G5-4 | Subject and People masks on Intel are unproven: the probe is designed to fail soft, nothing has run it. |
| G5-5 | Rosetta: if the person starts the x86_64 slice on Apple silicon (for example a terminal opened in Rosetta), `is_x86_feature_detected!("avx2")` is false on Rosetta releases that do not translate AVX2 (reported to be before macOS 15; verify when the task starts) and the scalar path runs; this is a valid path, untested. |
| G5-6 | The list of supported Intel Macs and the expected speed are not written down (no measurement; M4-03 user guide). |

### 5.3 Plan

**P5-1 (G5-1, G5-5) Assert and run the x86_64 slice.**

1. In `release.yml` "Check the bundle" and in `build.yml`: `lipo -archs` of `Contents/MacOS/Yellowmoon` and of `Contents/MacOS/ym` must be exactly `x86_64 arm64` (the order `lipo` prints), else fail.
2. Run the x86_64 slice: `arch -x86_64 "$APP/Contents/MacOS/ym" --version` and `ym doctor`. Rosetta is installed on the runner with `softwareupdate --install-rosetta --agree-to-license` if `arch -x86_64 /usr/bin/true` fails. This starts the slice, resolves its dynamic libraries and the Swift runtime path, and runs the scalar kernels (Rosetta on macOS 14 is reported not to translate AVX2; verify) — a different path from a real Intel Mac, so it is a start test, not a speed or colour test.
3. A job `intel-smoke` on a native Intel runner (`macos-15-intel`, a hosted x86_64 runner; check that the label exists when the task starts; if it does not, keep step 2 only): `cargo build -p ym-cli --release` natively, `ym --version`, `ym doctor` (the report must say "Apple Intelligence: not available (this Mac has an Intel processor)" and list the masks), then `golden check --set synthetic --filter ILCE-7M4 --filter ILCE-1` (a few files) against the Linux references: pass within the golden rule (identical hash, or ΔE00 within 0.5). The job is on `schedule` and `workflow_dispatch` and on pushes that touch `ym-ml`, `ym-lens`, `ym-apple`, `ym-gpu` or `release.yml`.
- Messages: none for the person; the failure text is the CI log.

**P5-2 (G5-3) Make the GPU budget a measured value.** Read the adapter's real limit on macOS through the Metal API (`MTLDevice.recommendedMaxWorkingSetSize`) in the Swift bridge crate (`ym-apple` already links Swift and is compiled for both architectures) and give it to `GpuOpener`: GPU budget = `min(rule, 0.75 × recommendedMaxWorkingSetSize)`. `ym doctor` prints it ("GPU memory limit from Metal: 1.5 GiB"). If the call fails the rule stays as it is. The factor 0.75 is a starting value to be set after MC-6; it is not measured. Tests: the rule is a pure function with a table test; the Swift call is checked on a Mac runner (`ym doctor` shows the number; an Apple-silicon value is above `RAM/4`).
- If the owner prefers no Swift code for this, the fallback is a conservative cap for `DiscreteGpu` and for an integrated GPU on an x86_64 build: 2 GiB. That is a guess until MC-6 and says so in the code comment.

**P5-3 (G5-4, G5-6)** The Vision probe's behaviour on Intel is checked by MC-7. The user guide (M4-03) lists the supported Intel Macs (those macOS 14 runs on: confirm against Apple's list at the time) and says which features need Apple silicon: Apple Intelligence Auto (also macOS 27 and Apple Intelligence on). No code.

### 5.4 Only a Mac can check

MC-6 to MC-9 below: a real Intel Mac with an iGPU, one with an AMD GPU, and macOS 14 on one of them.

---

## Tasks

Each task is one branch (`m4-02-h-07-...` style), one pull request, with the usual gate (fmt, clippy, tests, `npm test`, `npm run build`). They are independent unless a line says otherwise, and in a sensible order: wording first (small, no contract change), then the export stop (the biggest user-facing gap), then the 2000-file front end, then memory, then Intel.

**H-01 · File-message catalogue** (P1-2, P1-3, P1-5). `RawError::Corrupt` prints one of three sentences; `Io` sentences by kind; `detail` in `--json`, log and `-v`. *Done when:* `crates/ym-raw/tests/messages.rs` passes over the hostile generator and the 192-file sweep (every message is in the closed list, none contains "((", "os error", "Huffman", "entropy", "restart"); `docs/04` Errors lists the table; `ym-cli` snapshots updated; the zero-byte file reads "This file is empty. Copy it from the card again, or skip it."

*Done (2026-11-11).* As built: every detail that means the file is shorter than its directories say (raw data, a tile, a value, a directory or the first directory beyond the end of the file) is `CUT_SHORT`; `crates/ym-raw/src/error.rs` holds the detail constants `EMPTY` and `CUT_SHORT`, the seven `MESSAGE_*` sentences and `RawError::detail()`; `Corrupt` and `Io` print the catalogue sentences. Two changes from the plan: (1) the length-0 check is in `Tiff::open` (`tiff.rs`), not in `FileSource`, because `FileSource::open` returns `io::Result` and `decode_bytes` and `probe_bytes` need the same answer; before this an empty file read "This isn't an ARW file". (2) `ym info` given a folder now says `not_arw` ("This isn't an ARW file.") instead of an `io` error whose text named the folder, because the `Io` sentence is chosen from the kind alone. `detail` goes to `ym render --json` and `ym info --json` as `error.detail` (only when there is one, so the shape of other errors is unchanged), to stderr with `-v`, and to the log (`tracing::warn!` in `session.rs` for a failed probe and a failed decode, and in `export_queue.rs`). A failure that the export queue meets while decoding has no `error.detail` in `--json` (the queue passes a string up); its detail is in the log only. The shell and the frontend show the sentence unchanged (the reason string of `Skipped`); none of them hard-coded the old text. Tests: `ym-raw/tests/messages.rs` (5 tests; the sweep has 1,576 files: the foreign files, and for each of the 10 variants 157 damaged copies by truncation, header and body overwrites, bit flips and patched size tags; all three sentences are reached; the same files from disk give the same sentences), unit tests in `error.rs` and `tiff.rs`, `container.rs` and `ym-core/tests/session.rs` updated. The EPERM sentence on a real macOS block stays under MC-1.

**H-02 · Unreadable folders are said** (P1-1). `OpenOutcome.unreadable`, the CLI problem kind `unreadable`, the sentences. *Done when:* a folder with mode 000 gives "Yellowmoon can't read the folder {name}…" in the app (Vitest + Rust API test) and in `ym` (exit 1, stderr names it; skipped as root); `NO_ARW_MESSAGE` only for a folder that was read and held no ARW.

*Done (2026-11-11).* *As built (H-02).* `ym-core` `session.rs`: `Collected` and `OpenOutcome` get `unreadable: Vec<PathBuf>` and `unreadable_privacy_block` (the first folder's error was `EPERM`; one flag, because only the first folder's cause is said); `walk` records a folder whose listing fails (a subfolder too, with `recursive`) instead of returning silently; `pub fn unreadable_message(folders, privacy)` makes the sentences (one, "…and 1 more folder.", "…and N more folders."), `pub fn is_privacy_block(&io::Error)`; `OpenOutcome::message()` now returns `Option<String>`: the unreadable sentence, else `NO_ARW_MESSAGE` (the macOS wording only on macOS, `cfg!(target_os = "macos")`). The listing goes through a `Lister` seam (`collect_with`, `expand_with`), so the logic is tested as root too. `ym-cli`: `inputs::expand` gives a `Problem` of kind `unreadable` per folder; `render` exits 1 (not 5) when nothing is left and a folder was unreadable, without the "No ARW files were found" line; `--json` carries the problem. `src-tauri/src/api.rs`: `open_paths` returns the message as an `open` error (like "No ARW files"); the missing-file check no longer wins over it. **Changes from the plan:** (1) no wire type changed: the app gets the sentence as the `message` of the `open` error, so `bindings.ts` is untouched; the store (`store.ts`, `isUnreadableMessage` in `src/api/openMessages.ts`) shows it in the drop zone on Start and in the notice bar on Photos, for `open_paths` and `open_dialog`. (2) A drop that holds unreadable folders next to files that do open is not said (the files open; `OpenOutcome.unreadable` has the folder, the shell does not forward it); that is the one thing left for a later task. (3) `ym info` takes files only (a folder is `not_arw`), so only `ym render` has the new kind. Mock: `?mockUnreadable=Name`. Tests: `ym-core` 3 unit tests (the seam, the subfolder with `recursive`, the sentences) and 1 integration test (`chmod 000`); `ym-cli` 1 unit test (seam) and 1 integration test (exit 1, stderr, `--json`); `src-tauri` 1 API test (`chmod 000`); Vitest 2 for Start, 1 for the store, 1 for the mock option; Playwright 1 (`e2e/start.spec.ts`). The three `chmod` tests skip when the user can read a mode-000 folder (root); they were run as an ordinary user by hand for `ym render` (exit 1, both outputs). `yellowmoon-app` does not build on the Linux machine (gdk-sys needs GTK), so its change was read, not compiled.

**H-03 · The damaged-file matrix** (P1-6). `ym_raw::testgen::damage`; the three tests (core, CLI, app API). *Done when:* they pass; the 192-file sweep of this audit is a named case; permission and symlink cases skip when root; the good files of every folder are written and unchanged.

**H-04 · A file that goes away or changes** (P1-4). Stamp at probe; check before each decode; `Skipped` + event; export sentence. *Done when:* the `session.rs`, `export_queue.rs` and Playwright cases pass; a Ready photo whose mosaic is cached still renders; the sentences are exactly those in P1-2 and P1-4.

**H-05 · Classify write failures** (P2-1, P2-5). `destination_problem`, `kind` values `disk_full` and `destination`, the messages, the folder (not the temp name) in `flush`/`fsync` errors, the sentences of the looks import and the preferences. *Done when:* the unit tests for 28, 69, 30, 2, 13, 1 pass; `tests/disk_full.rs` asserts the new sentences for JPEG, PNG, TIFF and DNG; `ym render --json` snapshots show the kinds; no message contains "os error".

**H-06 · The export stops when the destination is gone, full or read-only** (P2-2). Resolve the destination before decoding; stop the job; `stopped`/`not_tried` in the contract; panel banner, rows, title, header cell; `ym` lines and JSON event. Needs H-05. *Done when:* the `export_queue.rs` test (fault hook) shows no decode or render after the failing photo; the `ym-cli` test passes; the Playwright and Vitest cases pass; the `disk-full` CI job on the 3 MiB tmpfs ends in under the time of one photo's render with exit 1 and an empty folder (the audit's baseline is 87 s for ten photos).

**H-07 · A warning before a full disk** (P2-3). `fs4::available_space`; the 1.35 × rule from the estimate's documented 90th-percentile error. *Done when:* the panel and `ym` show "This export needs about {size} and {free} is free in {folder}. It may stop part way." exactly when `free < 1.35 × estimate`, never block the export, and the unit, Playwright and CLI tests pass.

**H-08 · Temp-file hygiene** (P2-4). Stale `.yellowmoon-*.tmp` and `looks/` temporaries older than 24 h removed at job start and at library load; a deleted folder is not re-made after the first photo. *Done when:* the 25 h / 23 h tests and the "folder gone" test pass.

**H-09 · The front end at 2000 photos** (P3-1). Windowed filmstrip, Start list and queue list; memoised rows; lazy thumbnails; in-place row updates. *Done when:* Vitest, the keyboard and menu cases, the DOM-node case (`mockPhotos=2000` under 1,500 nodes) and `e2e/perf/latency.perf.ts` with `YM_PERF_PHOTOS=2000` pass; p95 at 2000 is at most 1.25 × p95 at 500 in the same run (audit baseline: 359 ms against 115 ms, a ratio of 3.1); the weekly job runs both.

*As built (H-09).* `src/components/windowed.tsx` is our own hook, `useWindowedRange` (no new dependency; no windowing library was needed). A list of 100 or more equal items draws the items in view and eight more on each side; the others are stood in for by space (the first drawn thumbnail's left margin and the last one's right margin in the strip, the `ul`'s padding in the two columns), so there is no wrapper element and the scroll bar is true. The item size is measured from the first two drawn items (their distance, gap included) and the origin from the first one, so the CSS stays the one place that says how big a row is; a hidden strip is not measured. Shorter lists are drawn whole, with no inline style: the 12-photo screenshots of `design/screenshots-c2/` are unchanged (all 688 Playwright tests pass, the screenshot comparisons among them). The strip's thumbnail is a memoised `Thumb` (by `id`, `name`, `src`, `status`, `picked`, `included`, `nodes` and its own flags), with stable callbacks, `loading="lazy" decoding="async"`. A photo on screen that is not drawn is brought into view by arithmetic (`scrollToIndex`) and then focused or revealed as before; while it is out of the window the first drawn thumbnail is the strip's tab stop. `ExportQueueList` and `DevelopList` have memoised rows. State: `applyProgress` copies the array of rows once and a row only when it changes, `queueItems` and the Start screen's items reuse the item of an unchanged row, `withStatus` and `withApplePending` find the photo once and copy the array once (the plan's "version counter" was not needed: React compares the rows by identity and by value). Not done, on purpose: the thumbnail map is still copied per `thumbnail_ready` (2,000 copies of 2,000 keys in all, not measurable), and a failed export's long note can make a queue row taller than the measured pitch (the spacers are then off by a few pixels, never the rows). **For M4-04 (accessibility):** a photo outside the window is not in the DOM, so VoiceOver cannot reach it by moving through the list until it is scrolled near; the items carry no `aria-setsize` or `aria-posinset`, so a screen reader cannot announce "photo 812 of 2000". Add both, and check keyboard and VoiceOver reading of a 2000-photo folder, in M4-04.

Measured on the shared sandbox (Chromium 153 headless, production build, mock speed 10, 80 keys 100 ms apart, `YM_PERF_PHOTOS=100,500,2000 e2e/perf/run.sh --trace off`; the absolute figures of this machine are slower than the audit's, the ratio is the point):

| Photos | Before: median / p95 / max | After, run 1: median / p95 / max | After, run 2: median / p95 / max |
|---|---|---|---|
| 100 | 52.9 / 76.8 / 109.6 ms | 59.2 / 113.6 / 131.6 ms | 39.0 / 59.4 / 68.7 ms |
| 500 | 110.7 / 226.7 / 346.8 ms | 57.4 / 105.8 / 166.0 ms | 39.2 / 60.5 / 76.7 ms |
| 2000 | 235.6 / 424.3 / 614.6 ms | 54.4 / 102.8 / 160.8 ms | 41.1 / 59.2 / 66.9 ms |

The ratio p95 at 2000 over p95 at 500 went from 1.87 on this machine (the audit's 3.1) to 0.97 and 0.98. Long tasks at 2000: 130 (18.7 s in all) before, 2 to 10 (0.1 to 0.8 s) after. DOM elements at `mockPhotos=2000` (`e2e/photos.spec.ts`): 166 on the Start list, 292 on the Photos screen, 197 with the export running (the audit: 8,088, 2,267 and 16,063). The two "after" runs were taken some time apart on the same shared machine; the spread between them is the machine's.


**H-10 · The open path at 2000 files** (P3-3). Canonicalise outside the lock; `open_progress`; Cancel; the ignored 2000-file tests in `ym-core` and `src-tauri`. *Done when:* the lock is not held while paths are canonicalised (a test holds a slow-path hook and calls `set_current`), the Start and Photos screens show "Reading {done} of {total} files…" with a working Cancel, and the 2000-file tests print open time, time to all Ready, slowest command and peak RSS and join the weekly job.

**H-11 · Thumbnail size and budget** (P3-2). Re-measure the PNG sizes; correct `docs/01:870`; add the byte budget and the re-render of evicted thumbnails if the measurement needs it. Needs H-09 for the "on screen" signal. *Done when:* `docs/01` states the measured size; the shell's thumbnail store stays under its budget in the 2000-file test; an evicted thumbnail returns when its row scrolls into view.

**H-12 · A RAM override and the 8 GB gate** (P4-1, P4-2). `YELLOWMOON_RAM_GB`; `ym doctor` line; the `memory-8gb` CI job with the peak recorded and bounded. *Done when:* the override parses and is shown by `ym doctor`; the job records peaks for 50 and 61 MP in JPEG, TIFF 16-bit and DNG and fails above the bound (set from its first run, written in the job).

**H-13 · Count what is held** (P4-3). The ledger reserves mosaics and the reduced group; the develop pool size depends on the budget. Needs H-12. *Done when:* the table test of the pool size passes (8 GB gives three threads, 16 GB six); two develop jobs never run together under a budget of 1.5 mosaics; the ledger `peak()` is within a factor of two of the RSS in the H-12 job.

**H-14 · Unload models, react to memory pressure** (P4-4, P4-5). `trim_caches()`; idle unload; the macOS pressure poll with an injected level in tests; the one-time notice. *Done when:* the unit tests (idle unload and reload, warning and critical actions, the notice shown once) pass; the Mac checks MC-4 and MC-5 are listed in STATUS.

**H-15 · Slow and hung media** (P1-7, P3-4). A watchdog around the probe (each probe on its own detached thread with a time limit; a late answer is dropped) and a limit on a develop job's read; the sentence: "This file didn't answer. Check the drive or the network connection, then open it again." The limit is chosen from MC-2. *Done when:* a test with a probe that sleeps past the limit marks the photo Skipped with that sentence and `open_paths` returns; the thread count after 100 such files is bounded and stated.

**H-16 · Run the x86_64 slice** (P5-1). `lipo -archs` asserts; `arch -x86_64 ym --version` and `ym doctor` in the release and build workflows; the `intel-smoke` job. *Done when:* the release workflow fails if either architecture is missing or if the x86_64 `ym` does not start; the `intel-smoke` job (or, without a native runner, the Rosetta step) is green and its `ym doctor` output is attached to the run.

**H-17 · The GPU budget from Metal** (P5-2). Needs a Mac for the proof. *Done when:* `ym doctor` prints the Metal working-set limit on a Mac runner, the budget is `min(rule, factor × limit)`, the table test of the pure function passes, and MC-6 sets the factor.

**H-18 · Close the books** (P1-5, P5-3, P4-7). Update `docs/04` (damage not detected), `docs/09` (stop rule, free space, temp hygiene), `docs/10` (stopped event, new kinds), `docs/11` (large folders at 2000, copy table), `docs/12` (Intel checks in the release), `docs/17` (RAM override, Metal limit), `docs/13` (the new jobs), `docs/STATUS.md` (this task, numbers, MC list under "Waiting on the owner", STATUS question 4 with the measured footprint). *Done when:* each doc says what the code now does, and the STATUS entry lists the gate numbers and the open Mac checks.

### Suggested order

H-01, H-02, H-03 (messages and tests), H-05, H-06, H-07, H-08 (disk full), H-09, H-10, H-11 (2000 files), H-12, H-13, H-14 (memory), H-16, H-17 (Intel), H-15 (needs MC-2), H-18 last. H-01 to H-03 and H-09 can start at once and in parallel.

---

## [HUMAN] Mac checks (M0-H4 style)

None of these can be done on Linux. Each one has a place in `docs/STATUS.md` under "Waiting on the owner" when its task lands.

| ID | Check | For |
|---|---|---|
| MC-1 | A file on a disconnected external drive, a slept SMB share and an SD card reader: the messages of H-01/H-02/H-04, that the app stays usable, and what the EPERM privacy block says | H-01, H-02, H-04 |
| MC-2 | Open a folder on a slept SMB share: how long a read hangs, whether `open_paths` returns, whether Cancel works; choose the limit of H-15 | H-15 |
| MC-3 | Peak footprint (Activity Monitor, `footprint`) on an **8 GB Apple-silicon Mac**: a 61 MP photo with the Look tab, a Film node, an Object mask and a Depth mask; export of 50 and 61 MP as JPEG, TIFF 16-bit and DNG; a folder of 100 61 MP files developing. Answers STATUS question 4. | H-12, H-13 |
| MC-4 | macOS memory-pressure level during MC-3 (`sysctl kern.memorystatus_vm_pressure_level`), and that the notice of H-14 appears and goes | H-14 |
| MC-5 | WKWebView with 2000 thumbnails: memory, scroll and key latency; the native viewport with 2000 photos developing | H-09, H-11 |
| MC-6 | A real **Intel Mac with an iGPU** and one **with an AMD GPU**: `ym doctor` (adapter type, limits, budget), a 61 MP export on the GPU and with `YELLOWMOON_GPU=off`, any "device lost" in the log; set the factor of H-17 | H-17 |
| MC-7 | Subject and People masks on an Intel Mac (does the Vision probe say yes, are the kinds shown or hidden correctly) | P5-3 |
| MC-8 | The x86_64 slice on **macOS 14** (Intel and Rosetta): the app starts, opens a file, exports; the update from an arm64-built version works | H-16 |
| MC-9 | Speed on the oldest Intel Mac the owner has: time to open, a slider move, a 24 MP export (no target exists; this sets the sentence in the user guide) | H-18 |
| MC-10 | APFS: a full disk during a JPEG, TIFF, PNG and DNG export (a small disk image mounted with `hdiutil`); the temporary file is gone and the sentence names the folder | H-05, H-06 |
| MC-11 | An external drive pulled during a batch: the run stops at once with the "Folder gone" or "No permission" sentence and nothing is written to `/Volumes` | H-06, H-08 |

## Questions for the owner

To be copied into `docs/STATUS.md` when the first task lands (this audit did not edit it).

1. **Exit code for a stopped export.** The plan keeps exit code 1 and adds a `stopped` event in `--json` (H-06). Do you want a code of its own (for example 6) so a script can tell "the disk is full" from "one file was bad"? Changing it edits the list in `docs/10` and in `ym --help`.
2. **A warning, not a block, for free space (H-07).** The size estimate is wrong by up to 35 % for one file in ten, and by more with noise reduction off, so the plan only warns and lets the exact stop do the work. Agreed?
3. **STATUS question 4** (the 8 GB budget) stays open until MC-3. The audit's numbers for it: 2 GiB GPU and 2.8 GiB CPU on 8 GB; a 61 MP export peaked at 2.4 GiB on the CPU path of a 15 GB Linux machine with nothing else running.
4. **Drop the detail from the file sentences (H-01).** Today "This file is damaged or incomplete (lossless JPEG: the entropy-coded data ends before the image does)" tells a technician what failed. The plan shows a short plain sentence and keeps the detail in `--json`, the log and `-v`. This follows the copy rules (no jargon), and costs the person nothing they can act on. Agreed?
5. **Swift for the Metal limit (H-17).** The Metal working-set limit is only available through Metal. `ym-apple` already holds Swift for both architectures. If you prefer no new Swift, the fallback is a fixed 2 GiB cap for AMD and Intel GPUs, which is a guess until MC-6.

## Appendix · How to repeat the measurements

```bash
# build (release); the example writes synthetic ARW files of any size
cargo build --release -p ym-cli
cargo build --release -p ym-raw --features testgen --example gen_arw --example gen_corpus
export YELLOWMOON_DATA=$PWD/data YELLOWMOON_HOME=$(mktemp -d)

# 61 MP file, peak memory (child ru_maxrss), CPU path
target/release/examples/gen_arw ILCE-7RM5 9504 6336 /tmp/mp61/DSC00001.ARW iso=400 wb=2043,1024,1024,1755
python3 -I peak.py target/release/ym render --cpu --apple-auto off --format jpeg --out /tmp/out /tmp/mp61/DSC00001.ARW
#   peak.py: subprocess.run(argv); print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024, "MiB")

# full disk
mkdir /tmp/small && sudo mount -t tmpfs -o size=3m tmpfs /tmp/small
target/release/ym render --cpu --apple-auto off --format png --out /tmp/small /tmp/ten      # ten 3000x2000 files

# unreadable file or folder as root
setpriv --bounding-set=-dac_override,-dac_read_search target/release/ym render ... 

# 2000 files: copy one gen_arw file 2000 times, rewriting the three capture-time strings
# ("YYYY:MM:DD HH:MM:SS" at bytes 206, 634 and 654 of a gen_arw file) for 2000 different times

# front end: build, serve, drive
npx vite build --outDir /tmp/dist --emptyOutDir && (cd /tmp/dist && python3 -m http.server 4180 &)
node ui_n.mjs 2000 100      # WAIT=1 waits until every photo is developed; copy of e2e/perf/latency.perf.ts for any N
node ui_export.mjs 2000     # pick all, open Export, start, wait for "Exported"
```
