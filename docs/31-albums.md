# 31 · Albums

An album is the filmstrip saved to a file the person names: which photos, in which order, and each one's settings. **File › Save** writes it and **File › Open** opens it again, so the person can stop and continue later. There is no album browser, no library and no panel: only the File menu (and a drop on the window).

**Status: specified (P-11, 2026-11-16).** Nothing here is built. The mock-ups are `design/presets/09` to `13` and wait for the owner (P-H1). It belongs to milestone P (`docs/15`), after phase 1 of presets (`docs/30`).

**The owner's request, 2026-11-16, verbatim:** *"in the presets work add album (strip with applied settings) save and load so I could continue to work with albums later. Just the File -> Save and File -> Open menus are good"*.

**This changes hard rule 4 and principle 3, at the owner's request.**

- `AGENTS.md` hard rule 4 now reads: "**Nothing is saved per photo by itself.** No catalogue, sidecars, database or edit history. The only things that persist on their own are imported looks, presets and a few preferences (see `docs/01-architecture.md` §Persistence). An album is written only when the person chooses File › Save, to a file they name (`docs/31-albums.md`)."
- `docs/00` principle 3 gains "unless you save the strip as an album".
- What stays true: Yellowmoon keeps **no list of albums, no recent albums, no autosave and no copy** in its application folder. An album is a document that the person saves where they choose; if they never choose Save, nothing is written. Originals are never touched (rule 5).

**In short.**

- `<name>.ymalbum`: JSON with the photos in strip order (paths, a fingerprint, the whole settings graph as stored, picked or not), the current photo, the versions that wrote it, and the Auto providers' answers the photos use (§2).
- **Opening an album replaces the session**, after asking when the open album has unsaved changes. Photos are found by the relative path, then the absolute path; a photo found by neither stays in the strip as a failed photo (§4).
- **⌘S Save, ⇧⌘S Save As…** (§5). The write is atomic. Save Preset moves to ⌥⌘S (§5.3).
- An open album shows its name and the macOS edited dot; closing or quitting with changes asks "Save changes to the album Lisbon?" (§6).
- `ym render album.ymalbum` renders every photo with its stored settings; `ym info album.ymalbum` lists them (§8).

## 1. What it is, and what it is not

| | A preset (`docs/30`) | An album |
|---|---|---|
| Is | a recipe for any photo | these photos, as they are |
| Holds | the groups the person ticked | everything of each photo's settings, and which photos |
| Brush strokes, Object clicks, crop, white pick | not saved (or, for the crop and the pick, never) | all kept |
| Persists | on its own, in `presets/` | only when the person chooses Save, in a file they name |
| Opened with | the Presets card | File › Open |

An album is **not** a catalogue: it has no index, nothing watches it, nothing outside the file refers to it. It is **not** an edit history: it holds the settings as they are, not how they were made (no undo steps, no list of presets that were applied).

## 2. The file

`<name>.ymalbum`, UTF-8 JSON, written compactly with sorted keys and one newline at the end (the same photos and settings are the same bytes, rule 8). Not signed, not encrypted.

```json
{ "yellowmoon_album": 1,
  "written_by": { "version": "0.2.4", "pipeline": "0.11.1" },
  "current": 3,
  "photos": [
    { "path": "../Lisbon/DSC04201.ARW", "absolute": "/Users/anna/Pictures/Lisbon/DSC04201.ARW",
      "fingerprint": { "size": 52428800, "head_sha256": "…", "tail_sha256": "…" },
      "picked": false,
      "settings": { "nodes": { "items": [ … ], "links": [ … ] } },
      "auto": { "provider": "claude", "model": "claude-sonnet-5-5", … } },
    … ] }
```

- **`yellowmoon_album`** is the format version, 1.
- **`written_by`** is the Yellowmoon version and `PIPELINE_VERSION` (`ym_core::version`) that wrote the file. It is for diagnosis: opening an album never refuses or warns because the pipeline differs (the file holds settings, not renders; a newer pipeline may draw the photo a little differently).
- **`photos`** is the strip in order. Each photo has:
  - **`path`**: the path of the photo **relative to the folder of the album file** (`../Lisbon/DSC04201.ARW`), or `null` when none exists (another volume). **`absolute`**: the absolute path. Both are UTF-8; a photo whose path is not valid UTF-8 refuses the save with "Yellowmoon can't save the path of DSC04201.ARW in an album. Rename the file or its folder, then try again."
  - **`fingerprint`**: `size` in bytes, `head_sha256` and `tail_sha256`: the SHA-256 of the first and the last 64 KiB of the file (the whole file when it is smaller; the two may overlap). Reading 128 KiB a photo keeps saving 500 photos fast (about 64 MB). It says whether *this file* is the one the album was made from, not whether it is byte-identical.
  - **`settings`**: **the photo's whole settings graph as stored**, node-first (`docs/23` §3), **unpinned: Auto stays Auto**. The crop, the orientation, the Pick white point, Object clicks, brush strokes and every node are kept, because this is the photo itself and not a recipe. It is `DevelopSettings` as the session holds it (`to_value`), the form `--settings` and the schema read, including `look` as an id.
  - **`picked`**: whether the photo is picked.
  - **`auto`** (only where a provider answered): **the Auto provider's answer the photo uses**: Apple Intelligence, Claude Code or Antigravity. It is the same JSON as `ym:AutoAdvisor` (`docs/09`, `docs/22` §14.2): `provider`, `model`, the tool's or system's version, `table` and `prompt`, the `steps`, the plan's words. **This is the album's one deliberate copy of a provider's answer.** Without it, opening the album would ask a cloud provider again and could give another picture (answers differ from run to run, `docs/22`). With it the estimates come out as they did. **Re-run still asks the chosen provider again.** A photo graded by Default Auto (the rule) has no `auto`.
- **`current`**: the index of the current photo, or `null`.

**Not in the file:** renders, masks' rasters, SAM features, depth and Vision maps, the estimates (worked out again on opening), the undo history (rule 4: no edit history), the photos' pixels, EXIF or GPS (which live in the ARW), and anything from Application Support (preferences, looks, presets: a look is an id, and a missing one is handled as Paste handles it).

**Limits.** A file above 512 MiB is refused: "This file is too large to be an album." At most 100,000 photos (`docs/26`'s windowed lists hold 2,000 comfortably; the limit is a sanity bound).

**Reading** (`Album::from_json`): a `yellowmoon_album` above 1, an unknown top-level field, or settings with a field this version does not know are refused with **"This album was made by a newer version of Yellowmoon. Update to open it."** A file that is not JSON or lacks `yellowmoon_album` or `photos` is "This file isn't a Yellowmoon album." A flat settings document of Yellowmoon 0.1 in it is the same sentence (no album was ever written in that form). **An older album always reads** (missing fields take their defaults). Out-of-range numbers are clamped like any settings document. `crates/ym-core/schema/album.schema.json` describes version 1 (P-12 writes it, tested like `settings.schema.json`).

## 3. Privacy

- The file holds **absolute paths**, which may contain the person's user name (`/Users/anna/…`), the names of their folders and files, **the names they gave their nodes**, and, where a provider answered, **its plan**, which describes the scene in words (`docs/22`). It is the person's own file in a place they chose; Yellowmoon never uploads or reads it except on File › Open, a drop or `ym`. The Save panel stays quiet about this; the user guide says it in one line ("An album holds the paths of your photos and the settings you made. Share it only with someone you would share those with.").
- **GPS and the capture time are never in it** (they live in the ARW).
- No network (rule 7); an album does not make `ym` or the app connect anywhere.

## 4. Opening an album

**File › Open Files… (⌘O)** takes `.ymalbum` as well as ARW files; a drop of an album on Start or Photos opens it; `album_open(path)` is the command (§7).

1. **It replaces the session**, as opening a document does: `Session::close` and then the album's photos, with the session's own `OpenOptions`. If the **open album has unsaved changes** (§6) the person is asked first: "Save changes to the album Lisbon?" (Save, Don't Save, Cancel). A session with no album asks nothing, as today.
2. **Finding each photo.** The relative `path` against the album's folder first, then `absolute`. The first that is a file is the photo. A photo found by **neither** is a **failed photo** in the strip: the tile says CAN'T OPEN and the cell says **"Not found: /Users/anna/Pictures/Lisbon/DSC04206.ARW"** (the album's `absolute` path). **Its entry is kept** with its settings, picked state and place, **so that saving again does not drop it**; the person can remove it from the filmstrip like any photo, which is a change to the album. A file that exists and cannot be read is the failed photo of `docs/04`/`docs/26` as always.
3. **The fingerprint.** After the photo is found, its size and head and tail hashes are compared with the album's. **A fingerprint that differs** gives the photo a notice, **"This file has changed since the album was saved."** (the photo's note in the cell and the notice bar, like the other file notices), and **its settings still apply**. A matching fingerprint says nothing. The check runs in the background and never stalls the strip.
4. **Settings.** Each found photo gets its stored settings as soon as it is developed (`Session::replace_settings`: clamped like a patch, the last word; the CLI's `--settings` uses the same call, rule 6). A stored look that is not in the library is handled **as Paste and presets handle it**: the look field stays as stored, the tile and the chip say "File missing or changed. Import again." (`docs/08` *Missing looks*), and the photo is drawn as a photo with an unusable look is today; an export of it is refused with the look error. Nothing is replaced silently.
5. **Provider answers.** A photo with an `auto` object gets that answer as its Auto provider state; nothing is asked. **Opening an album never asks a provider**: a photo without an `auto` object is graded by Default Auto's rule until Re-run, whatever provider the preferences choose. (Photos added to the album later are graded as any new photo is.)
6. **Order, picks and the current photo** are restored. A photo listed twice (the same file) is one photo.
7. **Refusals.** "This album was made by a newer version of Yellowmoon. Update to open it." / "This file isn't a Yellowmoon album." / "This file is too large to be an album." / "Yellowmoon couldn't read this file. Check that it is still there, then try again." The session is **not** replaced when the album is refused.
8. **One album with photos is refused:** choosing an album together with ARW files in File › Open Files… (or dropping both) opens nothing and says **"Open an album on its own, or open photos. Yellowmoon can't open both in one go."** Two albums at once: "Open one album at a time." Open Folder… (⇧⌘O) is as it was: it opens ARW files and never an album.
9. **Photos added while an album is open become part of the album** (Add photos, ⌘O on Photos, a drop): they join the end of the strip (path order within the call), and the album is edited. An album opened from a drop of photos plus ARW files is not possible (point 8).

**The file dialog.** `open_dialog("files")` filters with **one filter, "Photos and albums", for `arw`, `ARW` and `ymalbum`**. rfd 0.16 on macOS (`tauri-plugin-dialog` 2.8) takes the union of all the filters' extensions into one allowed list, so both kinds show together, and the existing filter "Sony ARW" is replaced, not added to (a second filter would behave the same). P-13 checks it on a Mac (P-H4); if the panel cannot show both, the fallback is two filters (Photos, Albums) in the same call.

## 5. Saving

1. **File › Save (⌘S).** The first time (no album path yet: a session opened from photos, or a new session) it asks where to save with the system's save panel: **the name defaults to the folder's name** (`Lisbon.ymalbum`; the folder of the first photo, or the album's current name) and **the panel starts in the first photo's folder**. After that Save **writes in place**, with no question. It is off when no photo is open and while a dialog is open.
2. **File › Save As… (⇧⌘S).** Always asks, starting in the album's folder with its name. The album then **is** the new file (its name, its path and the relative paths of the photos are made from the new folder).
3. **Keys.** ⌘S and ⇧⌘S are native menu key equivalents, like ⌘E and ⌘W: neither types in a text box. **This takes ⇧⌘S from Save Preset (`docs/30`), which moves to ⌥⌘S** (not free of ⌥ typing "ß", but with ⌘ it is a command; `docs/30` §9.8 has the reasons). The menu test counts ten key equivalents.
4. **The write is atomic**: a temporary file in the album's folder, `fsync`, rename, `fsync` of the folder (`ym_looks::manifest::atomic_write`). **It never touches an original** (rule 5); it writes only the album file, and only where the person chose. A write that fails changes nothing and says "Yellowmoon couldn't save the album to {folder}. Check that you can save there and that the disk has free space, then try again. Nothing was changed."
5. **What is written is the session as it is when Save runs.** Settings changes still on their way (a slider drag, a patch in flight) are waited for first (as `remove_photos` waits for them). Photos still developing are saved with their settings (they are in memory). Failed photos are saved as they are (§4.2).
6. After a successful Save the album is **not edited** (§6). The notice bar says "Saved the album Lisbon." Nothing about the album is stored in Application Support: **no recent-albums list, no remembered path, no autosave.**

## 6. The edited state, close and quit

- **While an album is open**, the folder cell shows **its name in place of the folder's** ("Lisbon", for `Lisbon.ymalbum`) and its sub-line keeps the date and the count ("2026-10-04 · 12 PHOTOS"), except that **while the album is edited it reads "12 PHOTOS · EDITED"** (the date goes while it is edited, because the cell is 300 px wide with 84 px kept for the traffic lights, and the longer line wrapped; it comes back after Save). The window's title is the album's name ("Lisbon"; with no album it is "Yellowmoon").
- **Changes.** A change to the album is: a setting of any photo (a slider, a look, a paste, a preset, Re-run, Auto crop, a node patch), a pick or an unpick, a photo added or removed, and Restore after Remove. (The strip has no reordering today; if it gets one, a move is a change too.) Undo does not clear the state: once something changed the album stays edited until it is saved.
- **Edited** is shown by **macOS's edited dot in the window's close button** (`NSWindow.isDocumentEdited`) and the title **"Lisbon — Edited"** (the Window menu, Mission Control and VoiceOver read the title; the hidden title bar does not draw it), and by "EDITED" in the folder cell's sub-line (above).
- **Close Session (⌘W), closing the window and Quit** with an edited album ask **"Save changes to the album Lisbon?"** with **Save**, **Don't Save** and **Cancel** (a macOS alert; Save is the default). Save writes (asking where first if the album has no file yet) and then goes on; Don't Save goes on; Cancel stays. If Save fails the alert does not close the session: the error bar says why. **The question comes before the export question** ("An export is still running…", `docs/11` *Session*); with both, the person answers the album first. Opening another album or other photos with File › Open while the album is edited asks the same question.
- **Without an album** (a session opened from photos and not saved) nothing changes from today: closing asks nothing, as before.
- The core keeps the state (`Session::album_state() -> Option<AlbumState { path, name, edited }>`): a counter bumped by every change above and the counter at the last save. So the shell can ask on a window close with the webview gone, and `ym` and the app agree. The event `album_changed` (no payload) tells the page and the shell, which set the title and the dot (`set_album_window(name, edited)`, `src-tauri/src/native.rs`, a main-thread call).

## 7. The core and the shell

**`ym_core::album`** (P-12; `crates/ym-core/src/album.rs`):

| Item | What it is |
|---|---|
| `Album { written_by, current, photos }`, `AlbumPhoto { path, absolute, fingerprint, picked, settings: DevelopSettings, auto: Option<AutoAnswer> }` | The file's content; `settings` is the whole graph |
| `Fingerprint { size, head_sha256, tail_sha256 }`, `Fingerprint::of(path)` | §2; reads at most 128 KiB |
| `Album::capture(&Session, album_path)` | From the session as it is: the strip in order, picks, current, each photo's settings (`Session::settings(id)`, Auto kept, no pinning), the provider answers; failed entries as they were |
| `Album::to_json()`, `Album::from_json(&str)`, `AlbumError::{NotAnAlbum, TooNew, TooLarge, Settings, Io, PathNotUtf8}` | §2, with the sentences of §2 and §4 as `Display` |
| `write_album(path, &Album)` | Atomic (§5.4) |
| `Session::open_album(&Album, album_dir, OpenOptions) -> AlbumOutcome` | §4: finds the photos, opens them, applies settings, picks, current and answers, keeps missing entries; `AlbumOutcome { photos: Vec<(PhotoId, AlbumPhotoState)>, album: AlbumState }` with `AlbumPhotoState::{Found, Changed, Missing { absolute }}` |
| `Session::album_state()`, `Session::set_album(path)` | §6 |

It reuses `Session::open` (the same decoder and caches), `Session::replace_settings` (the same call as `--settings`), `Session::set_picked`, `set_current`, `paste`'s clamping, `LookService::check_usable` for looks, and the provider's answer type of `ym:AutoAdvisor`.

**As built (P-12, 2026-11-16).** `crates/ym-core/src/album.rs` (`Album`, `AlbumPhoto`, `Fingerprint`, `AutoAnswer`, `AlbumState`, `AlbumOutcome`, `AlbumPhotoState`, `AlbumError`, `write_album`, the sentences), `Session::open_album`, `capture_photos`, `album_state`, `set_album`, `edit_count`, `on_album_changed` (`session.rs`), `Core::album_open`, `album_save`, `album_state`, `on_album_changed` (error scope `album`), `crates/ym-core/schema/album.schema.json`. Tests: `tests/album.rs` (21). The decisions the build made are items 13 to 24 of §12. The shell, the menu and the page (P-13) are not built. **P-14 (2026-11-17):** `ym render <album>` and `ym info <album>` (`crates/ym-cli/src/render.rs`, `info.rs`; `tests/album.rs`, and the byte-for-byte case in `parity.rs`); `Session::ask_auto_again` is what `--auto-provider` does to an album's frozen answers. `ym info <album>` lists the album and then the usual block of each photo found, without the album's stored answers.

**IPC (P-13).** Commands follow `docs/01` (scope `album`):

| Command | Purpose |
|---|---|
| `album_open(path)` → `SessionState` | §4. `SessionState` gains `album: { name, path, edited } \| null` and each failed photo says why ("Not found: …") |
| `album_save(as_new)` → `{ path, name } \| null` | §5. The shell opens the save panel when `as_new` or the album has no path; `null` when the person cancels |
| `album_state()` → `{ name, path, edited } \| null` | The page's mirror (after `album_changed`) |
| Event `album_changed` | The state changed (an edit, a save, an open, a close) |
| `menu_action` / `set_menu_state` | gain `save`, `save_as` (off without a photo or while a dialog is open) |
| `open_dialog("files")` | §4: the filter; an answer for an album is `opened.album` |

The shell's close handling (`src-tauri/src/app.rs`, `RunEvent::ExitRequested` and `CloseRequested`) asks the album question before `ask_to_quit` and holds the window back while it asks (the same way the export question does); the dialog is `tauri-plugin-dialog`'s message dialog with three custom buttons.

**As built (P-13, shell, 2026-11-16).** `src-tauri/src/album.rs` (the save panel and its defaults, the question "Save changes to the album Lisbon?" with Save, Don't Save and Cancel, the window's title "Lisbon — Edited" and `set_album_window`), `AppApi::album_open`, `album_save`, `album_state`, the commands `album_open`, `album_save`, `album_state`, the event `album_changed`, `menu.rs` (File › Save ⌘S, Save As… ⇧⌘S, ten key equivalents), `native::set_document_edited` (the edited dot, `NSWindow.isDocumentEdited`, set on the main thread without waiting). **The shell asks the question** for the window's close button, Quit and a File › Open Files… that picks an album (`app.rs`: the album first, then the export question; `open_dialog`); **the page asks it** for Close Session and a dropped album (the webview, `docs/30` §17). `open_paths` routes a lone `.ymalbum` to `album_open` and refuses an album with other paths or a second album. Not carried by the wire: the "This file has changed since the album was saved." notice of §4.3 (the core's `AlbumPhotoState::Changed` has no field in `OpenSummary`; it needs one when the page shows it). Checks that need a Mac (P-H4): the edited dot, the three-button alert and its default, the panel's filter showing photos and albums together.

## 8. The command line (P-14)

**`docs/10` gets the rows in P-14, with the code** (`crates/ym-cli/tests/docs.rs` fails on a flag `ym` does not have); this section is their spec until then.

- **`ym render <album.ymalbum> [options]`** renders every photo of the album **in album order** with its stored settings: the album's settings replace the photo's defaults exactly as `--settings` does (`Session::replace_settings`), then each `--preset` (`docs/30` §11), then the flags, **a flag always having the last word**. An album given with a `--settings` file is a usage error (exit 2: "The album already has the settings of its photos."). An album given together with photos or folders is a usage error ("Give an album on its own, or photos."). Output names, formats and `--out` are as for photos.
- **Photos not found** are per-photo errors of `error.kind` `not_found` with the sentence "Not found: <path>"; the others go on; the exit code is 1 (some failed), or 5 when none was found.
- **Provider answers.** The album's stored answers are used and no provider is asked, **unless `--auto-provider` (or `--apple-auto`) is given, which asks that provider anew for the run** (explicit, as everywhere in `docs/10`). Without `auto` a photo is graded by Default Auto.
- **Parity (rule 6).** The exports are **byte-identical to the app's export of the same album** with the same options: `crates/ym-cli/tests/parity.rs` opens the album through `Core` and through `ym`, exports both and compares the bytes. The same sentences for a missing photo, a changed fingerprint (a note in `--json`: `notes: ["This file has changed since the album was saved."]`) and a missing look (`error.kind` `look`, exit 3 when every failure is a missing look).
- **`ym info <album.ymalbum>`** lists the album: its photos in order (name, path, picked, found or not, fingerprint matched or not) and the versions that wrote it; `--json` gives the same. The estimates and the Auto state are the ones of `ym info <photo>` for each found photo, with the album's answers.
- A file that is not an album, is too new or too large exits 2 with its sentence. The CLI never writes an album (saving is the app's).

## 9. Tests

- **P-12 (core).** `crates/ym-core/tests/album.rs`: save and open a session of ten photos (order, picks, current, every setting equal: a manual white balance, a crop, a white pick, an Object mask's clicks, a Brush mask's strokes, a graph with nodes, Auto kept as Auto, a missing look), **the settings are stored unpinned** (no Auto becomes a number), the provider answer round trip (a fake provider: nothing is asked on opening, Re-run asks), the relative path wins, the absolute path is the fallback, **a missing photo is a kept failed entry that saving again writes back**, a fingerprint change notices and applies the settings, a photo listed twice, a refused album (newer, not an album, too large, a path not UTF-8) leaves the session as it was, **the edited state** (every change of §6 bumps it, a save clears it, undo does not), the file bytes are the same for the same session, `YELLOWMOON_HOME` holds no album, recent list or autosave after a full cycle (the `tests/core.rs` tree comparison). The schema against what the core writes and against refusals. The wire types' field names against `src/api/types.ts`.
- **P-13 (app).** `src-tauri/src/api/tests.rs` (`album_open`, `album_save` in place and as new, the event, the close question's order with an export running), `menu.rs` (Save, Save As…, the keys, ten equivalents, the state items), `src/state/store.album.test.ts` (the edited state follows every change, the question, the notices), `src/screens/Header.test.tsx` (the folder cell with an album's name and "EDITED"), `e2e/album.spec.ts` (save, change, close asks, open again, a missing photo, a drop, a mixed choice refused), the screenshot states of `design/presets/09` to `13` in `design/screenshots-c2/`.
- **P-14 (CLI).** `crates/ym-cli/tests/album.rs` (render in order, flags and presets over stored settings, `--settings` refused, a missing photo, exit codes, `info`), `parity.rs` (byte for byte with the app), `docs.rs` with the new rows.

## 10. Tasks

**P-11 · Spec, rule change, mock-ups** (done 2026-11-16, with P-01). `docs/31`, `AGENTS.md` rule 4, `docs/00`, `docs/01` *Persistence*, `docs/11` (a pointer), `docs/15`, `docs/16`, `docs/30` (the relation and the key), `docs/STATUS.md`, `design/presets/09` to `13`.

**P-12 · The core** (P-11) — `ym_core::album` (§7): the file and schema, `Fingerprint`, `capture`, `open_album`, the edited state, `Session` changes (kept failed entries, the album's answers), the tests of §9. *Done when:* they pass, `cargo clippy` and `cargo deny check` are clean, and an album opened from a saved session renders the same bytes.

**P-13 · The app** (P-12; P-H1 for the look) — the commands and the event, the menu (File › Save, Save As…, Open Files… taking albums, ⌘S, ⇧⌘S; Save Preset moved to ⌥⌘S in `menu.rs`), the folder cell, the title and the edited dot, the close and quit question, the drop, the notices, the dialog filter. *Done when:* the tests of §9 pass and the five states of `design/presets/09` to `13` are matched by pictures taken from the app (looked at by the owner before they are committed).

**P-14 · The command line** (P-12) — `ym render <album>` and `ym info <album>`, the rows of `docs/10`, the parity test. *Done when:* the CLI tests pass and `docs.rs` reads the new rows.

**[HUMAN] P-H4 · Try it** on a Mac (P-13, P-14): save a strip, quit, open it, change and close (the question), move a folder and see the missing photo, open an album and a photo together, drop an album, and check that the file dialog shows ARW files and albums together.

## 11. Later (not designed; each a task when the owner says so)

- Recent albums (Open Recent), autosave, an album browser, **Locate…** for a moved folder (relinking the missing photos), merging two albums, saving an album from a few picked photos (a smaller album), and an album that holds a preset list.

## 12. Decisions made while writing the spec

The simplest reading that meets the request was taken wherever the brief was silent.

1. **One filter, "Photos and albums"**, for `arw`, `ARW` and `ymalbum` in File › Open Files…; rfd unions filters on macOS, so two filters would show the same. An album chosen with photos is refused with a plain sentence (§4.8).
2. **⌘S Save, ⇧⌘S Save As…, ⌥⌘S Save Preset.** ⇧⌘S was `docs/30`'s Save Preset key; the albums take the macOS keys for Save and Save As, and Save Preset moves to ⌥⌘S (a command key that does not type "ß" because ⌘ is held). Ten native key equivalents.
3. **The core keeps the edited state** (a change counter), not the webview, so that the shell can ask when the window closes and `ym` and the app agree. Undo does not clear it.
4. **A Save with no album file yet** (a session opened from photos) is the first Save and asks where; the session then is an album. Nothing is saved by itself.
5. **The file holds both paths, the fingerprint (size and two 64 KiB hashes), the unpinned settings, the picks, the current photo, the versions and the providers' answers.** The answers are the one deliberate copy of a provider's output; Re-run asks again.
6. **Opening an album never asks a provider.** A photo without a stored answer is graded by Default Auto's rule until Re-run.
7. **A missing photo is a kept failed entry**, so that saving again does not drop it; removing it from the strip is a change.
8. **A changed fingerprint gives a notice and the settings still apply.**
9. **A newer file is refused** with the presets' sentence adapted ("…Update to open it."); an older pipeline version is not a reason to refuse or warn.
10. **The privacy is said here and in the user guide**, not in the Save panel (the owner's brief: keep the UI quiet).
11. **No recent albums, no autosave, no browser, no Locate…, no merge** (§11).
12. **`ym render <album>` refuses `--settings` and mixed inputs**; `--auto-provider` given explicitly asks that provider anew.
13. **`Session::open_album(&Album, album_path, options)` takes the album file's path**, not its folder (`album_dir` is the folder: a folder given instead is used as that folder), because the open album's name and file are part of the state it sets. It replaces the session (it calls `Session::close`); `Core::album_open(path, options)` reads and checks the file first (`Album::read_file`), so a refused album leaves the session as it was, then closes the session with the renderer (`close_session`) and opens.
14. **The stored settings reach each photo before its development is queued** (`Session::open_inner`'s `on_new`), so no job and no thumbnail ever see the defaults, and a Pick white point takes its sample when the photo is developed (`develop`, which does what `commit_settings` does for a pick set by hand). `docs/31` §4.4's "as soon as it is developed" is this, a step earlier.
15. **The fingerprints are compared in parallel inside `open_album`**, before it returns (128 KiB a photo; 500 photos are about 64 MB), and not in the background: the shell calls `open_album` off the UI thread, and the answer already says for each photo `Found`, `Changed` or `Missing { absolute }`. A file that cannot be read for its fingerprint counts as `Found` (it is a failed photo with its own reason); a failed entry saved again takes the fingerprint the album gave it (`Fingerprint::unknown()`, size 0 and empty hashes, when it never had one).
16. **A relative path is resolved lexically** (`resolve_relative`: `Albums/../Shoot/DSC1.ARW` is `Shoot/DSC1.ARW`), so a photo found by its relative path has the path it would have if it had been opened by hand, and saving again writes the same paths. A photo that is neither at its relative nor at its absolute path is a `Photo` with the status `Skipped` and the reason "Not found: <absolute>", with its stored settings, pick, answer and place, and it is not in the set of known files.
17. **`AutoAnswer` is the JSON of `ym:AutoAdvisor` as a struct** (`AutoAnswer::from_applied` parses `AppleAuto::advisor_json`, `to_applied` gives `AppleAuto::Answered` back). Only answers are stored: a failure and a photo graded by Default Auto have no `auto`. The photo's chosen provider's answer is stored, else any answer it keeps (a photo opened with a Claude answer and saved while Apple Intelligence is chosen keeps it). An answer is put in the slot of its own provider; it is used while that provider is the chosen one.
18. **Nothing asks a provider about an album's photo** (`Photo::auto_frozen`): it is set by `open_album` for every photo of the album, it keeps the photo out of the Auto worker's list, out of "waiting for an answer" and out of the picture rendering, and **Re-run and new instructions for the model clear it** (they ask again, as §4.5 says). A photo added to the album later is a new photo and is asked as any is.
19. **The edited state** is `State::edits` (bumped by every committed change of a photo's settings, a pick or an unpick, `pick_all` and `clear_picks` when they change something, photos added, removed or put back, and Re-run) and the value at the last save. A call that changes nothing (the same pick twice, a patch that leaves the settings as they are) does not count; an undo is a patch like any other and counts. `Session::album_state()` is `Some` only once an album is open or the session was saved (`set_album`), so a session of photos that was never saved asks nothing when it is closed, as before.
20. **`Session::on_album_changed(f)`** is the core's `album_changed`: it runs, outside every lock, when the album **becomes** edited (once, until the next save), when it is saved (`set_album`), opened and closed. It is a callback on the session, not a `SessionEvent`, because it is not tied to a photo or an epoch; the shell turns it into the event.
21. **`Album::capture` takes the strip from `Session::capture_photos`**, which reads `Photo::settings` directly (so a photo that is still developing, or failed, is saved with its settings) and never pins: Auto stays Auto. A path that is not UTF-8 refuses the whole save (`AlbumError::PathNotUtf8`) before anything is written.
22. **Reading is strict about names and lenient about content:** a field this version does not know (top level, a photo, `auto`, the settings) is `TooNew`; a missing `fingerprint`, `picked` or `written_by` takes its default; a missing `settings` or `absolute` is not an album; `current` that is not an index of the strip is none. A photo's settings are read over Neutral, and every look in the file is kept as an id.
23. **The file's keys are sorted at every level** with the core's own writer (`write_canonical`, shared with presets), so the same session is the same bytes.
24. **The wire:** `AlbumState { path, name, edited }` (`name` is the file's name without its extension) is the interface of `src/api/types.ts` (`AlbumState`), and the error scope `album` is in `ErrorScope`. The schema's `$ref: "settings.schema.json"` resolves like the preset schema's (`docs/30` §17, item 27).
25. **The webview's readings (P-13, 2026-11-16).**
    - **The album's state is `SessionState.album`** (`{ name, path, edited }`, absent for none); it is read again on `album_changed` with `album_state`. `open_dialog`'s answer needs no `opened.album`: the page reads the session.
    - **The question is asked in the webview** for Close Session (the menu item arrives as a `menu_action`, so the shell cannot ask first) and for a dropped album over an edited one; **the shell asks** for the system's close button, Quit and File › Open Files… (it holds the dialog). The webview's dialog has Save (default), Don't Save, Cancel.
    - **A dropped album with ARW files, or two albums,** is refused by the webview with the sentences of §4.8; `open_dialog` refuses the same in the core.
    - **The sub-line** reads "12 photos · edited" in the source text (drawn in capitals by CSS, like the rest of the line).
    - **An album opened from Start** shows the usual list of developing photos, and Open photos goes on, as for any open.

**As built (export after opening, 2026-11-17).** `State::estimates_final` (`session.rs`) made an export wait for the chosen provider's answer for every photo of it and of its burst, including photos frozen by an album (`Photo::auto_frozen`), which nothing asks; a frozen photo with no stored answer (a failure is not stored, or the album was saved without a provider) therefore kept the export at 0 % for ever. **The rule:** a frozen photo is settled for the provider. It keeps its stored answer if it has one, else Default Auto's grade, exactly what the viewport shows, so `ym render <album>` gives the same bytes. Re-run, new instructions and `ask_auto_again` unfreeze a photo, and the export then waits for its answer as before. A normal session has no frozen photos and is unchanged. Tests: `tests/album.rs` (`an_opened_album_*`). **Checked 2026-11-17:** an album saved after every photo was answered loses no answer (`an_album_saved_after_every_answer_exports_when_opened_in_a_working_session`, `a_real_looking_answer_survives_the_album_unchanged`): the same Core saves, closes as the app does, opens, and every photo has its answer, nothing is asked, and the export is byte-identical to the one before saving. An answer is not stored when the provider failed for that photo (a refusal, a timeout) or when the album was saved before it arrived; that photo has none after opening, which is the case above.

**As built (frozen photo, no answer, 2026-11-17).** `refresh_effective` (`session.rs`) serves a frozen photo (`Photo::auto_frozen`) that has no result for the chosen provider as `AppleAuto::Off` with provider Yellowmoon, not `Pending`: nobody asks, so the Automatic card shows no "looking at this photo" line, and `apple_pending` never counted it. Pixels are unchanged (Default Auto's grade, as before). Re-run unfreezes it and it is `Pending` while asked. Test: `tests/album.rs` (`an_album_photo_with_no_answer_is_not_pending_*`).
