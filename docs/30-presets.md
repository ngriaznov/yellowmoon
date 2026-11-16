# 30 · Presets

A preset is a saved recipe: a name and some of a photo's settings. Choosing it on another photo sets those settings and nothing else. Presets are about no photo, so they are the one thing besides imported looks and preferences that Yellowmoon keeps (hard rule 4 of `AGENTS.md`; `docs/01` *Persistence*).

**Status: specified (P-01, 2026-11-16).** Nothing here is built. The mock-ups are in `design/presets/` and wait for the owner (P-H1). This file is the spec of milestone P in `docs/15`.

**The owner's decision, 2026-11-16.** The lead proposed presets and the owner answered: **"Go with your recommendations on presets"**. That settles four things:

1. **Presets persist**, in a new `presets/` folder. Hard rule 4 now reads "imported looks, presets and a few preferences".
2. **They live on a Presets card on the Look tab**, between the Automatic card and the Look card. *(Superseded the same day by the owner's review of the mock-ups: **"Presets should be at the end after tone on Look panel. In fact, remove tone section from Look and replace with Presets"**. The Look tab's cards become **Automatic · Look · Presets**, and the Tone card goes: §9.10.)*
3. **A preset holds the eight groups the person ticks**, with the groups they changed ticked for them.
4. **Phase 1 comes first** (P-01 to P-06, including New Preset from Exported Photo). Phases 2 and 3 (§16) are listed and not designed.
5. **Presets replace the Tone card** (the owner, 2026-11-16, after seeing the mock-ups; the words are in point 2). The Presets card is the last card of the Look tab; Temperature, Contrast and Saturation stay where they already live (§9.10).

(On 2026-11-01 `design/nodes/mockup-notes.md` said "Presets are not planned until the features settle". The features have settled; this file replaces that line.)

**In short.**

- A preset is `presets/<safe name>.ympreset`: JSON with a name, the groups it holds, and a partial settings document in the node-first form of `ym render --settings` and `ym:Settings` (`docs/23` §3).
- Eight groups: Look, Lens, Light, Colour, Detail, Curve, RAW and Nodes (§2). The Save dialog ticks the groups in which the photo differs from its default settings.
- **Auto stays Auto.** Applying a preset never gives a photo another photo's exposure or white balance, unless the person set one by hand.
- Applying goes through the core's Paste path: one call for many photos, one undo step per photo, the same `PasteReport`, the same bytes in the app and in `ym` (rule 6).
- `ym presets …` and `ym render --preset` (§11). New Preset from Exported Photo reads the settings back out of an export's XMP (§12).

## 1. What it is, and what it is not

| | Copy settings and Paste (`docs/11`) | A preset |
|---|---|---|
| Lives | in the page's memory, for the running app | in the `presets/` folder, until the person deletes it |
| Holds | the whole graph but the Input's crop, orientation and white pick | the groups the person ticked |
| Applies | everything it holds, and replaces the target's graph | only its groups; the photo's other settings stay |
| Brush strokes | copied | **not saved** (§3) |
| Named | no | yes, and shared as a file |

**Presets and albums.** A preset is a recipe for any photo; an album (`docs/31-albums.md`) is these photos as they are, with their own settings, saved to a file the person names with File › Save. A preset never holds a photo's crop, white pick or brush strokes; an album holds all of them. Applying a preset to the photos of an open album is one more change to the album, and saving the album writes the result (the preset is not remembered in it: the album holds settings, not how they were made). A preset is a recipe, not a photo's edit: it holds no file name, path, camera, date or location and does not know which photo it came from. It is not a catalogue and not a sidecar: the photo's settings are still kept nowhere (`docs/00` *Out of scope*).

## 2. The groups

Group ids are identifiers (American spelling); the UI labels are British. The order of the table is the order in the dialog, in the file and in the CLI's help.

| Group id | UI label | Fields | Node |
|---|---|---|---|
| `look` | Look | `look`, `look_intensity` | Output |
| `lens` | Lens | `lens` | Input |
| `light` | Light | `exposure`; `contrast`, `brightness`, `highlights`, `shadows`, `white`, `black` | Input; Main's grade |
| `color` | Colour | `light_source`, `white_balance`; `color_balance`, `saturation`, `color_warp` | Input; Main's grade |
| `detail` | Detail | `sharpen`, `noise_reduction`; `output_sharpening` | Input; Output |
| `curve` | Curve | `curve` | Main's grade |
| `raw` | RAW | `decode`, `rolloff` | Input |
| `nodes` | Nodes | every node other than the Input, Main and the Output (correctors, Film nodes, the mixers of `docs/25`, and Face Refinement once it is merged), the links, and Main's `name`, `bypass`, `trim` and `masks` | the graph |

- Every field of `docs/01` *DevelopSettings* is in exactly one group, except the three fields that §3 leaves out. **`pos`** (where a node's card sits on the canvas) of the Input, Main and the Output goes with `nodes`.
- A group is saved whole: a Light preset holds all seven Light fields, whether or not the photo changed each. Applying a Light preset to a photo therefore sets all seven (Auto where the preset says Auto).
- **A corrector's `grade` belongs to the groups above only in Main.** The grades of the other correctors are part of `nodes`.
- A new field added to the settings later joins one group in the same pull request (a test fails on a field that is in none: §14).

## 3. What a preset never holds

- **The Input's `crop`, `orientation` and `white_pick`**, as Copy settings (`COPY_LEAVES_OUT`). They are the photo's own geometry and pixels.
- **Object mask clicks**: the row arrives empty (no `points`, no `box`), as with Paste (`docs/23` §3.8; `MaskRow::emptied`).
- **Brush strokes.** A preset leaves a Brush row's strokes out: the row arrives with no strokes and selects nothing. **This differs from Copy settings, which copies strokes.** The reason: strokes are drawn on one photo's content, and a mask drawn there means nothing on another photo, and can be many thousands of points in a shared file. The row keeps its feather, Add or Subtract and its place among the rows, so the person can paint it again.
- **Subject, People, Background, Depth, Linear, Radial and the range rows are kept.** The Find kinds (Subject, People, Background, Depth) are worked out again on each photo, as with Paste (`docs/23` §4.1). Where a Mac cannot make a kind, the row says so as it does for any photo (`docs/23` §3.10).
- **Nothing about a photo**: no file name, path, camera, date, location, no Auto estimate and no `AutoAdvisor` answer.

## 4. Saving, applying, matching and stacking

### 4.1 Saving

A preset is made from **a photo's settings** (the photo on screen, or "Update from This Photo"), from **a settings file** (`ym presets save --from settings.json`) or from **an exported photo** (§12). All three give a resolved settings document; the same function turns it into a preset: `Preset::from_photo` (§7).

1. Take the Copy settings patch of the source (`SettingsPatch::copied_from`), which already leaves out the three fields above, keeps **Auto as Auto**, and writes **a white balance that came from a picked white as the manual Kelvin and tint it resolved to** (the same rule, the same function: a point of that photo's pixels means nothing on another, the light it found does).
2. Empty the Object rows and the Brush rows (§3).
3. Cut the result to the chosen groups (§2), written over the default graph: a field of an unticked group is left out of the file.
4. Record the look's id and name when the `look` group is chosen (§5).

**Pre-ticking.** The Save dialog ticks each group in which the photo's settings differ from **the photo's default settings: the default graph with the effective default look** (`LookService::effective_default`). The comparison is per group, over the group's fields (§2), on the stored settings: Auto equals Auto, so a photo that Apple Auto graded but the person never touched ticks nothing, and a picked white (saved as a manual white balance) ticks Colour. `changed_groups` (§7) does it; the page never compares settings itself. The person can tick and untick anything; **Save needs a name and at least one group.**

### 4.2 Applying

`Preset::patch_for(target, look_usable)` makes the patch for one photo: a node-level `SettingsPatch` (`docs/23` §3.7) that holds the preset's fields of its groups, built with `SettingsPatch::of_fields`, and:

- With **`nodes`** the patch holds every other node whole, `links` (the preset's), and the Input's, Main's and the Output's `pos`, and `remove` lists the target's other nodes that the preset does not have: **the preset's nodes replace the photo's other nodes**, as Paste does (`SettingsPatch::pasted_onto`).
- **Without `nodes`** the photo's nodes and links stay as they are. A preset of Light and Colour on a photo with three correctors changes Main's grade and the Input, and nothing else.
- Never touched, whatever the file says: the Input's `crop`, `orientation` and `white_pick` (the same guard as a paste).
- **A missing look.** When the `look` group's id is not in the library, or its file is gone or changed (`LookService::check_usable` fails), the look fields are left out of the patch, **the rest applies**, and the report says which look: "Warm evening uses the look Film warm, which isn't imported. Everything else was applied." Looks are matched by id; a look of the same name from another library is another look.
- **Auto stays Auto.** A preset that says Auto for exposure sets the photo's exposure to Auto. It does not clear a Pick white point (a paste does not either: the pick stays the photo's own).

`Core::preset_apply(name, photo_ids)` loads the preset, drops the look when it is missing, and calls **`Session::paste_settings`** once with the patch: the call that every Paste goes through. That gives, with no new code on the way:

- one call for many photos, under one lock, atomic (a patch that cannot be applied to any photo changes none);
- one undo step per photo, as the `before` and `after` patches of `PasteReport` (`PastedPhoto`);
- photos that are still developing or failed are skipped, and the report names them;
- the same bytes in the app and in `ym` (rule 6): `ym render --preset` builds the same patch with the same function and applies it with `DevelopSettings::apply_patch`, which is what `paste_settings` uses.

The answer is a `PresetApplyReport { pasted, skipped, missing_look }`: the `PasteReport` fields and the name of the missing look, or none.

### 4.3 Matching

A tile is **ringed** when applying the preset to the photo would change nothing: `Preset::matches(settings, look_usable)` makes the patch, applies it to a copy of the photo's settings and compares the copy with the original. **Nothing is stored**: the core answers it with `presets_list(photo_id)`, and the page asks again after the photo's settings change (the same moment it refreshes the look chip). A preset with a missing look matches when everything else matches. The ring is the look tiles' own (`aria-pressed`, the tab accent): "this is what the photo has now".

### 4.4 Stacking

**Presets stack.** Each application is a patch on the settings the earlier ones left. Where two presets share a group, the later one wins, because it sets the same fields last. A preset of Detail applied after a preset of Look and Colour keeps both. In the app each click is one step of undo; on the command line `--preset a --preset b` applies `a` then `b` (§11).

## 5. The file

`presets/<safe name>.ympreset`, UTF-8 JSON, written compactly with sorted keys and one newline at the end (a preset saved twice from the same settings is the same bytes, rule 8):

```json
{ "yellowmoon_preset": 1, "name": "Warm evening", "groups": ["look", "light", "color"],
  "look": { "id": "user:film-warm", "name": "Film warm" },
  "settings": { "nodes": { "items": [ … ] } } }
```

- **`yellowmoon_preset`** is the format version, 1.
- **`name`** is the name as typed: trimmed, 1 to 64 characters.
- **`groups`** lists the groups the preset holds, in the order of §2, without repeats, at least one.
- **`look`** is present only with the `look` group. It repeats the look's name so that a missing look can be named. The id is the settings' own `look` (a `LookId`, for example `builtin:signature`, `user:film-warm`).
- **`settings`** is the settings document of `docs/23` §3 holding only the fields of the listed groups, with the nodes they live in (the Input, Main, the Output, and with `nodes` the others and `links`). It is read over the default graph, so a field the file does not have takes its default.

**Reading** (`Preset::from_json`):

- **A newer version is refused.** A `yellowmoon_preset` above 1, an unknown top-level field, or settings with a field this version does not know (`SettingsError::Parse`) is refused with: **"This preset was made by a newer version of Yellowmoon. Update to use it."** Nothing is guessed from a file made by a newer version.
- **An older preset always reads**: missing fields take their defaults (the same rule as every settings document).
- A file that is not JSON, has no `yellowmoon_preset` or no `settings`, or has a name or a group that is not valid: "This file isn't a Yellowmoon preset."
- Fields of `settings` outside the listed groups (a hand-written file) are ignored when applying, and left out when the preset is saved again.
- Out-of-range numbers are clamped like any settings document. A file above 4 MiB is refused: "This file is too large to be a preset."
- A flat document of Yellowmoon 0.1 inside `settings` is `SettingsError::OldForm` (`docs/23` §3.12), shown as "This file isn't a Yellowmoon preset." (no preset was ever saved in the old form).

**JSON Schema.** `crates/ym-core/schema/preset.schema.json` (P-02 writes it; it is tested like `settings.schema.json`: against what the core writes, against documents it refuses). It is:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://yellowmoon.app/schema/preset.schema.json",
  "title": "Yellowmoon preset",
  "type": "object",
  "additionalProperties": false,
  "required": ["yellowmoon_preset", "name", "groups", "settings"],
  "properties": {
    "yellowmoon_preset": { "const": 1 },
    "name": { "type": "string", "minLength": 1, "maxLength": 64 },
    "groups": {
      "type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": true,
      "items": { "enum": ["look", "lens", "light", "color", "detail", "curve", "raw", "nodes"] }
    },
    "look": {
      "type": "object", "additionalProperties": false, "required": ["id", "name"],
      "properties": { "id": { "type": "string", "minLength": 1 }, "name": { "type": "string" } }
    },
    "settings": { "$ref": "settings.schema.json" }
  },
  "if": { "properties": { "groups": { "contains": { "const": "look" } } }, "required": ["groups"] },
  "then": { "required": ["look"] },
  "else": { "not": { "required": ["look"] } }
}
```

The schema describes version 1. A reader of version 1 refuses a larger number before it looks at anything else, which the schema's `const` cannot express as a friendly sentence.

## 6. The store

`presets/` sits **beside `looks/`** in the application folder (`ym_looks::library::default_root()`: `~/Library/Application Support/app.yellowmoon/presets/`; `YELLOWMOON_HOME` replaces the root). `PresetStore` (`ym-core`, `presets/store.rs`) owns it; `PresetService` (§7) wraps it as `LookService` wraps the library.

- **Writes are atomic**: a temporary file in the same folder, `fsync`, rename, `fsync` of the folder (`ym_looks::manifest::atomic_write` does it for the manifest; the store uses it). They run **under `presets/.lock`**, the same advisory file lock as the look library's (`ym_looks::lock::LibraryLock`, opened on the presets folder; the app and `ym` may write at once). A read never takes the lock: a file is whole or absent.
- **Names** are 1 to 64 characters after trimming, with no control characters, and **unique ignoring case** (`to_lowercase`). "Warm evening" and "warm evening" are one name. The name in the file is the truth; the file name is a label.
- **The file name** is the name made safe for every file system: `/ \ : * ? " < > |` and control characters become `-`, runs of spaces and dashes are kept to one, leading and trailing spaces and dots are cut, a name that is empty after that or is `.`/`..` becomes `preset`, and the stem is cut to 200 bytes at a character boundary. Two names that make the same file name (`a/b`, `a:b`, or names that differ only in case on a case-insensitive disk) get ` 2`, ` 3` … appended to the stem of the newer one.
- **The list is sorted by name, ignoring case**, then by file name.
- **A file that cannot be read is listed as damaged, with the reason**, so that it can be deleted. It never crashes the list, never stops the others, and is never rewritten. Reasons are the closed list of §5: "This file isn't a Yellowmoon preset.", "This preset was made by a newer version of Yellowmoon. Update to use it.", "This file is too large to be a preset.", "Yellowmoon couldn't read this file." (an I/O error). Two files with the same name (one copied in by hand): the first by file name is used and the other is listed as damaged with "Another preset has this name."
- **Rename** writes the new file and then removes the old one under one lock; **Replace** writes over the file of the same name (an atomic rename); **Delete** moves the file to the Trash (the `trash` crate through `ym_looks::import::trash::move_to_trash`, which the looks wizard already uses). A machine without a Trash gives the error of `TrashError`; the file stays.
- **Reload.** `list()` re-reads the folder every time (a preset made by `ym` shows up): the folder is small, and a listing reads at most the files it finds. `on_change` runs after any change the service made, outside every lock, and is `presets_changed`.
- Nothing else is written: no index, no cache, no thumbnails (they are memory only, §9.6). A full open, edit and export cycle writes no file unless a preset is saved, renamed, replaced, imported or deleted (the `tests/core.rs` tree comparison, §14).

## 7. The core's API (P-02)

New module **`ym_core::presets`** (`crates/ym-core/src/presets.rs` and `presets/`). It reuses `ym_core::settings` for every setting and adds no field to `DevelopSettings`.

| Item | What it is |
|---|---|
| `PresetGroup` | The eight groups: `id()`, `label()`, `ALL` in the order of §2, `fields()` (the `(NodeId, SettingsField)` list of the group), `serde` as the ids |
| `PresetLook { id: LookId, name: String }` | The `look` object of the file |
| `Preset { name, groups, look, settings }` | `settings` is the whole `DevelopSettings` read over the defaults; the groups say which fields count. `groups` is a sorted set |
| `Preset::from_photo(name, &ResolvedSettings, groups, look_name)` | §4.1: `SettingsPatch::copied_from`, emptied Object and Brush rows (`MaskRow::emptied_for_preset`: `emptied` plus no strokes), cut to the groups |
| `Preset::from_exported(name, ExportedSettings, groups)` | §12 |
| `Preset::to_json()`, `Preset::from_json(&str)` | The file (§5); `PresetError::{NotAPreset, TooNew, TooLarge, Settings, InvalidName, NoGroups}` with the sentences of §5 and `Display` |
| `Preset::patch_for(&self, target: &DevelopSettings, look_usable: bool) -> PresetPatch` | §4.2: `PresetPatch { patch: SettingsPatch, missing_look: Option<String> }` |
| `Preset::matches(&self, target: &DevelopSettings, look_usable: bool) -> bool` | §4.3 |
| `changed_groups(&ResolvedSettings, defaults: &DevelopSettings) -> Vec<PresetGroup>` | The pre-ticked groups (§4.1) |
| `PresetStore` | `open(root)`, `list() -> PresetListing { presets, damaged }`, `get(name)`, `save(&Preset, replace: bool)`, `rename(old, new)`, `remove(name)`, `import(path, replace) -> ImportOutcome`, `export(name, dest)`, `resolve(query_or_path) -> Result<Preset, PresetError>` (§11), `path_of(name)`, `safe_file_name(name)` |
| `PresetService` | The store behind the lock plus `LookService` for the look checks (`check_usable`, `list` for the look's name) and `on_change(f)`: `list(photo_id)` (adds `matches`), `save`, `update`, `rename`, `remove`, `import`, `apply` and `thumbnail_settings` |
| `PresetEntry`, `DamagedPreset`, `PresetApplyReport`, `ImportOutcome` | Wire types (§8); a test compares their field names with `src/api/types.ts` as the other wire types are |
| `Core::preset_*` | One method per IPC command (§8); `Core::open` builds the service (it writes nothing) and wires `on_change` |

`AutoFields` (§12) is added next to `AutoFlags` in `ym_core::settings`, and `ym_export::xmp::read_ym` reads a file's XMP.

Where the code is, by name: the pasting is `Session::paste_settings`, the patch builders are `SettingsPatch::of_fields` and `SettingsPatch::copied_from`, the fields left out are `COPY_LEAVES_OUT`, the field names are `SettingsField`, the lock is `LibraryLock`, the Trash is `move_to_trash`, the atomic write is `ym_looks::manifest::atomic_write`, and the preferences' own atomic write is the model for the failure sentence (§9.9).

## 8. IPC (P-04, P-05, P-06)

Commands follow `docs/01` *IPC*: snake_case, errors are `AppError { scope, message }` with scope `presets`. The wire types are in `src/api/types.ts` and generated by `specta` as the others.

| Command | Purpose |
|---|---|
| `presets_list(photo_id)` → `{ presets: PresetEntry[], damaged: DamagedPreset[], folder }` | Reloads the folder first. `PresetEntry` is `{ name, groups: PresetGroup[], look: { id, name } \| null, look_missing, matches, revision }`: `matches` is §4.3 for `photo_id` (false for `null`), `revision` is the first 12 hex digits of the file's SHA-256 (the thumbnail key and the page's cache key). `DamagedPreset` is `{ file, reason }` |
| `preset_inspect(source)` → `{ changed: PresetGroup[], older_export, look }` | For the Save dialog: the groups to tick (§4.1), whether the source is an export that has no `ym:AutoFields` (§12), the look's name. `source` is `{ kind: "photo", photo_id }` or `{ kind: "file", path }` |
| `preset_save(name, groups, source, replace)` → `PresetEntry` | Makes the preset from `source` (§4.1) and writes it. Without `replace`, a name that exists is refused with "A preset called Warm evening already exists." (the page asks the question first) |
| `preset_update(name, photo_id)` → `PresetEntry` | Saves the photo's values again for the preset's own groups, same name |
| `preset_rename(name, new_name)` → `PresetEntry` | Refuses a name that exists, case aside from its own |
| `preset_apply(name, photo_ids)` → `PresetApplyReport` | §4.2. Emits `settings_changed` and queues a thumbnail for each changed photo exactly as `paste_settings` does (`AppApi::paste_settings`) |
| `preset_remove(name)` → `null` | To the Trash. A damaged file is removed by its file name: `preset_remove_file(file)` |
| `preset_export(name)` → `{ path } \| null` | The shell opens the system save panel (suggested name `Warm evening.ympreset`) and writes the file; `null` when the person cancels |
| `presets_import_dialog()` → `ImportResult \| null` and `presets_import(paths, replace)` → `ImportResult` | The first opens the system open panel (`.ympreset`, several); the second is for dropped files and for Replace after the question. `ImportResult` is `{ added: string[], replaced: string[], exists: { name, path }[], refused: { file, reason }[] }`. A name that exists is **not** replaced unless `replace` |
| `preset_reveal(name)` | Show in Finder: reveals the file |
| `preset_thumbnail(photo_id, name)` → address | The tile: the photo with the preset applied, §9.6. Answers the address of a PNG on the `ym-frame` scheme. Rejects with scope `presets` when the preset is gone |
| Event `presets_changed` (no payload) | After any change by the service, and when a `presets_list` found a different list (a preset made by `ym`) |
| `set_menu_state` | gains `save_preset` (a developed current photo) and `apply_preset` (the same); both are off while a dialog is open, like the settings items |
| `menu_action` | gains `save_preset`, `import_presets`, `new_preset_from_export` and `apply_preset`, whose event carries `preset` (the name): the Apply Preset submenu is built by the shell from `presets_list` and rebuilt on `presets_changed` (at most 30 names, the rest are on the card). `MenuActionEvent` becomes `{ action, preset? }` |

`preset_thumbnail` shares the look tiles' machinery (`docs/11` *Look tiles*): the core's job is `Core::spawn_preset_tile`, built like `film_tiles::spawn_film_tile` (the photo's settings with the preset's patch applied, rendered by `Renderer::look_thumbnail_png` through the settings' own look, at most 240 px on the long edge, sRGB, `Priority::Thumbnail`, superseded by a move to another photo). The shell keeps the PNG in the look tiles' in-memory store (key `preset:<revision>`, the same 256 MiB budget, forgotten with the photo).

## 9. The interface

The pictures are `design/presets/` (dark glass, at 1440 × 900 and 2×, the real tokens). **The mock-ups win on layout and size, this section wins on copy and behaviour, `docs/11` wins where both are silent.** The words below are the words of the mock-ups.

### 9.1 The Presets card

On the **Look tab**, **after the Look card, as its last card, in the place the Tone card had** (`docs/11` *Tool panel*). The tab's cards are **Automatic · Look · Presets** (§9.10). `src/screens/tools/PresetsCard.tsx`, used by `LookTab.tsx`.

- **Header.** "Presets", the count in the quiet type of a look section's count, and the link button **Save…** at the right.
- **Apply to.** The card's own `SegmentedControl`: **This photo / Picked · N / All · N**. It **starts on This photo and is not remembered** (every start begins on This photo), and it is separate from the Look card's. A click on a tile applies the preset to that scope; the scope, the photo and the preset are read at the click and never again (`docs/11` *Apply to*: a move to another photo before the core has answered changes nothing). The calls reach the core in the order of the clicks.
- **Tiles.** The look tiles' size, picture, name line, ring, hover and focus (`ym-look-tile`, three to a row at 400 px). The picture is the **current photo with the preset applied**. **The name is under it**; there is no number badge (keys 1 to 9 stay with the looks). The ring is §4.3. The tooltip names the groups: "Look, Light, Colour", and for a missing look "Look: Film warm isn't imported."
- **How many.** The card shows **6 tiles** (two rows) and, when there are more, the link button **Show all 14** under them; it then shows every preset and the link reads **Show fewer**. The count is folded for the running app only (like the look libraries' folding). *The proposal said "at most 12 tiles until Show all 14", which with fourteen presets would show twelve and make the link pointless in the first picture; the mock-up shows six, and which number is right is for the owner (§17, decision 1).*
- **Damaged files** are listed under the tiles, one quiet row each: the file name, the reason (§6) and the link **Delete** (to the Trash). They do not count in the header.
- **Where it sits.** The panel scrolls (`.ym-tools__scroll`) and the Look card's list of looks can be long, so the Presets card is at the end of the scroll. `design/presets/01-look-tab-presets.jpg` shows the panel scrolled to its end: the foot of the Look card, then the whole Presets card (header, Apply to, six tiles, Show all 14).

### 9.2 The empty state

With no preset the card has the header "Presets" (no count, no Save… link) and one body: **"Set a photo the way you like it, then save it as a preset."** and the button **Save preset…**. The button is disabled while the photo is not developed (title "Wait for the photo to develop").

### 9.3 The Save preset dialog

`Dialog` (glasscn), 480 px wide, opened by **Save…**, **Save preset…**, ⌥⌘S and Looks › Save Preset…; the focus starts in the Name field.

- Title **"Save preset"**.
- **Name** (`Input`), placeholder "Name". Trimmed; 1 to 64 characters; the dialog says "Give the preset a name." when empty and "Use 64 characters or fewer." when longer, under the field, and Save stays off.
- **Eight boxes**, one row each: a checkbox, the label and one quiet line (copy in the table), in the order of §2. **The groups the photo changed are ticked** (§4.1); the others are not. A glasscn checkbox is used (`src/components/glass/checkbox.tsx`, added from the registry in P-04; the registry's component is the control, none is hand-written. The mock-ups draw it).

  | Label | Line |
  |---|---|
  | Look | The look and how strongly it is applied. |
  | Lens | The lens and its amounts. |
  | Light | Exposure, contrast, brightness, highlights, shadows, white and black. |
  | Colour | White balance, colour balance, saturation and colour warp. |
  | Detail | Sharpening, noise reduction and output sharpening. |
  | Curve | The luma, red, green and blue curves. |
  | RAW | Decode options and the range. |
  | Nodes | The other nodes, their links and masks. |

- The line **"Auto stays Auto, so each photo keeps its own exposure and white balance."** under the boxes.
- **Cancel** and **Save** (primary). Save is off without a name or without a group (title "Tick at least one group"). Return in the Name field saves; Esc cancels.
- **When the name exists** the dialog's body is replaced by the question **"Replace Warm evening?"** with the line "A preset with this name is already saved. Replacing it can't be undone." and the buttons **Cancel** (back to the form, the focus on Name) and **Replace** (primary). The page finds out from `presets_list`, not from the refusal.
- **From an exported photo** (§12) the same dialog opens with the Name empty; for an export without `ym:AutoFields` a quiet line sits under the boxes: **"This photo was exported by an older Yellowmoon, so its automatic values are saved as fixed numbers. Untick Light, Colour and Detail to leave them out."**
- Saved: the dialog closes and the notice bar says "Saved the preset Warm evening." ("Replaced the preset Warm evening." after a replace).

### 9.4 A tile's menu

Right-click a tile (Control-click, the Menu key or Shift+F10 on a focused tile): glasscn's `ContextMenu` (`src/components/glass/context-menu.tsx`), named "Warm evening actions". Items:

| Item | Does | Off when |
|---|---|---|
| Apply to Picked Photos | applies to the picked photos, whatever the card's Apply to says | no photo is picked ("No photos are picked.") |
| Update from This Photo | asks "Update Warm evening?" with the line "It keeps its groups, Look, Light and Colour, and takes their values from DSC04204." and **Cancel** / **Update**; then `preset_update` | the photo is not developed |
| Rename… | the Save dialog's Name field alone (title "Rename preset", **Cancel** / **Rename**); a name that exists is refused under the field: "A preset called Cool morning already exists." | |
| Export… | `preset_export` (the system save panel) | |
| Show in Finder | `preset_reveal` | |
| Delete | `preset_remove`: to the Trash, no question (it can be taken back); notice "Moved the preset Warm evening to the Trash." | |

### 9.5 Notices (the bar at the foot, `docs/11` *Add photos, Copy settings and Paste*)

| When | Text |
|---|---|
| Applied to one photo | "Applied Warm evening to DSC04204." |
| Applied to several | "Applied Warm evening to 5 photos." |
| Some skipped | "Applied Warm evening to 4 photos. 1 photo was skipped because it isn't developed." (3 photos: "…3 photos were skipped because they aren't developed.") |
| None applied | "Nothing was applied: 2 photos aren't developed." |
| A missing look | "Warm evening uses the look Film warm, which isn't imported. Everything else was applied." (in place of the first sentence; the skip sentence follows it when there is one) |
| Nothing picked | "No photos are picked." |
| Saved, replaced, updated | "Saved the preset Warm evening." / "Replaced the preset Warm evening." / "Updated Warm evening from DSC04204." |
| Renamed | "Renamed the preset Warm evening to Cool morning." |
| Deleted | "Moved the preset Warm evening to the Trash." |
| Exported | "Exported the preset Warm evening." |
| Added | "Added the preset Warm evening." / "Added 3 presets." |
| An import with a refusal | "Added 2 presets. Couldn't add Old.ympreset: This preset was made by a newer version of Yellowmoon. Update to use it." (the first refusal is named, the others counted: "…and 1 more file.") |
| An import that found a name that exists | the question "Replace Warm evening?" (the same dialog) for each, then "Replaced the preset Warm evening." |
| Cannot write | "Yellowmoon couldn't save the preset to {folder}. Check that you can save there and that the disk has free space, then try again. Nothing was changed." |
| Cannot read the file to import | "Yellowmoon couldn't read {file}. Check that it is still there, then try again." |

### 9.6 Thumbnails

They reuse the look tiles' machinery and its numbers, not a copy of the code (`state/useLookThumbnail.ts`, `state/requestLimiter.ts`, `createTileWatch`): **requested when a tile scrolls into view, and again 400 ms after the photo's other settings stop changing** (`THUMBNAIL_SETTLE_MS`); **only the tiles in view ask**; two requests at once (`LOOK_THUMBNAIL_CONCURRENCY`), newest first; a request that has not started when the settings change or the person moves to another photo is dropped and rejects with `Superseded`, which is not a failure; a tile that failed asks again, at most twice; the shell keeps 256 MiB of tile PNGs (`LOOK_TILE_BUDGET_BYTES`, one budget for looks, film stocks and presets) and answers 404 for the least recently used, on which the page forgets the address and asks again. The page's key for a tile is the photo, the preset's `revision` and the photo's settings. A preset that is saved again has a new `revision` and its tile is made again.

### 9.7 Dropping a preset

A `.ympreset` file dropped on **Start or Photos** (the whole window accepts drops: `useWindowDrop`) is added: `presets_import(paths, false)` and the notice **"Added the preset Warm evening."** Other files in the same drop are handled as before (ARW files are opened). On Start, which has no notice bar, the line sits under the drop zone's text. A drop of presets only does not start a session and does not say "No ARW files in what you dropped." (`docs/11` *Start*). The Export panel takes the drop too, as it takes photos.

### 9.8 The Looks menu

`src-tauri/src/menu.rs` (`docs/11` *Menus*) gains, after the nine Apply Look items and a separator, **these five items**, then a separator before Show Looks Folder:

- **Save Preset…** ⌥⌘S (`CmdOrCtrl+Alt+S`)
- **Import Presets…**
- **Apply Preset** ▸ (a submenu of the saved presets by name, a checkmark on the ones that match; "No Presets" greyed when none)
- **New Preset from Exported Photo…**
- (a separator)
- **Show Presets Folder** (the shell's own, like Show Looks Folder: it creates the folder if it is missing and reveals it)

Keys 1 to 9 stay with the looks. **Why ⌥⌘S is a menu key, and why it is not ⇧⌘S.** The rule of *Menus* is that a key a text box needs must not be a native key equivalent (⇧⌘C and ⇧⌘V are the webview's because other apps use them in text). ⌥⌘S is not a text-editing command: with ⌘ held the text box types nothing (⌥ alone types "ß", which is why ⌥⇧⌘V needs `code: "KeyV"` in the webview, and why a menu key that includes ⌘ is safe), and no key handler of the webview uses it. So it is a menu equivalent, like ⌘E and ⌘W. **The first draft took ⇧⌘S; the albums of `docs/31` (the owner, 2026-11-16) use ⌘S for File › Save and ⇧⌘S for Save As…, so Save Preset moved to ⌥⌘S.** The item must be **off while a dialog is open** (`set_menu_state`, so the Save dialog's Name field cannot open a second dialog) and while no developed photo is current. The menu test that keeps the text keys out of the menu (`shortcuts_parse_and_do_not_take_the_text_keys`) counts ten key equivalents instead of seven (⌘S, ⇧⌘S and ⌥⌘S are new). Nothing here is a single-letter key, so "Single-key shortcuts" off strands nothing (`docs/28` A-P11).

### 9.9 Errors and what the person sees

Errors say what went wrong and what to do (`docs/11` *Copy rules*). The table of §9.5 holds the sentences; a core error is shown in the error bar with its own sentence. The sentences are made once, in `crates/ym-core/src/presets/` and `src/state/presets.ts`, and a test (§14) pins each.

### 9.10 The Tone card goes (P-05)

**The owner, 2026-11-16: "Presets should be at the end after tone on Look panel. In fact, remove tone section from Look and replace with Presets".** The Look tab's cards become **Automatic · Look · Presets**. The **Tone card** (Temperature, Contrast and Saturation, with its AUTO tag and Reset; `ToneCard` in `src/screens/tools/LookTab.tsx`) is removed from the Look tab. **Its controls stay where they already live**, and nothing is lost: Temperature in Colour › White balance (Kelvin; Tint is there too), Contrast in Light › Exposure, Saturation in Colour › Colour balance. They are the same settings (`white_balance`, `contrast`, `saturation`), so no setting, patch, file or CLI flag changes.

The removal is part of **P-05**, not of this task: until P-05 the app still has the Tone card and `docs/11`, `docs/01`, `docs/19`, `docs/23` and `docs/16` still describe it. **P-05's acceptance criteria include:**

- The component and what only it used: `ToneCard` and `revealToneSlider` (the Automatic card's Contrast and Saturation rows opened the Tone card's sliders). **Those two rows now open the Light tab's Contrast (`Contrast +15, Auto. Open Light`) and the Colour tab's Saturation (`Saturation −15, Auto. Open Colour`)**, as the Exposure and White balance rows already open their own tabs, and `docs/19` section 10 says so.
- Its tests: `src/screens/ToneCard.test.tsx` and the Tone parts of `LookTab.test.tsx`, `AppleAuto.test.tsx`, `AutoProviders.test.tsx` and `LensCard.test.tsx`; the e2e `e2e/tone.spec.ts` and the Tone cases of `e2e/photos.spec.ts`, `e2e/selection-frame.spec.ts`, `e2e/apple-auto.spec.ts` and `e2e/a11y.spec.ts` (the names "Open Tone"); the axe, focus-order, contrast and glass-audit states that show the card (`docs/11` *Dark glass interface*, `e2e/glass.spec.ts`, `e2e/focus-order.spec.ts` lists); and the screenshot references of `design/screenshots-c2/` that show it (`photos-look*.png`, taken again with the Presets card and looked at by the owner before they are committed).
- **Every doc reference to "Look › Tone"**, changed when the code goes: `docs/01` (the settings table: "also on Look › Tone" for `contrast`, `white_balance` and `saturation`), `docs/11` (the Photos tab table, *Tone card*, *Tool cards*, *Apple Auto* and its table of the Automatic card's arrows, *Sliders and wheels*), `docs/19` (section 10), `docs/23` §2 (the table of where every control lives, the three "Look › Tone" rows), §3.2 (the `white_balance` row) and §6, `docs/16` (the line "The Look tab has a Tone card", which becomes a decision of its own), `docs/09` (the table's "the Tone card" in the pixels row) and `docs/00` if it names the card.
- The Look tab's blurb and the tab table of `docs/11` read the new cards: **Look · Automatic, Look, Presets**.

## 10. Accessibility (`docs/28`)

- **The tiles** are buttons in a group named "Presets" like the look tiles: the accessible name is "Warm evening preset", `aria-pressed` is the ring (applying it would change nothing), the description is the groups ("Look, Light and Colour") and, for a missing look, "The look Film warm isn't imported." Return and Space apply. Each is a Tab stop; the focus ring is `--focus`, as on the look tiles. The context menu opens from the Menu key and Shift+F10, and its name is "Warm evening actions".
- **The card** is a region named "Presets" (a `section` with `aria-labelledby`, as the others). The count and Show all are in the heading's row; "Show all 14" has `aria-expanded`.
- **The Save dialog** is a `role="dialog"` named "Save preset"; the focus starts in the Name field and returns to what had it. The boxes are a group named "Save these" with one checkbox each, named by its label and described by its line; **Save's disabled reason is its description**. The question is a dialog named "Replace Warm evening?" with the focus on Cancel (Replace is not the default).
- **Announcements** (the polite region of the Photos screen, `announcementFor`): the notices of §9.5 are in the notice bar, which is a status; applying says nothing more. No key is added, so the single-key switch has nothing to turn off.
- Contrast and Reduce Transparency: the card and the dialogs are the app's own glass (`docs/28` A-05, A-07); `e2e/axe.spec.ts`, `contrast.spec.ts` and `focus-order.spec.ts` take the new states in (§14).

## 11. The command line (P-03)

```
ym presets list [--json]
ym presets save <name> --from <export.jpg | settings.json> [--groups look,light,color,…] [--replace]
ym presets import <file.ympreset> [--replace]
ym presets export <name> <file.ympreset>
ym presets remove <name>
ym render … --preset <name or file> [--preset …]
```

(`--replace` on `save` is an addition to the proposal: a name that exists needs a way to be written over, and `import` has it.) **`docs/10` gets these rows in P-03**, with the code, because `crates/ym-cli/tests/docs.rs` fails on a flag `ym` does not have; this section is their spec until then.

- **`list`** prints NAME, GROUPS and LOOK, one line each, sorted as the app does; a damaged file is a line after the table (`file: reason`). `--json`: `{ "folder": "…", "presets": [{ "name", "file", "groups", "look": { "id", "name" } | null, "look_missing" }], "damaged": [{ "file", "reason" }] }`.
- **`save`** makes a preset with `Preset::from_photo` from a **settings file** (the node-first form of `--settings`, read over the default graph) or, from P-06, **an exported photo** (§12). `--groups` takes ids; without it the groups that differ from the default settings are saved (§4.1, with the effective default look); when none differ the command exits 2 with "Nothing in this file differs from the default settings. Choose the groups with --groups." A name that exists exits 2 ("A preset called Warm evening already exists. Use --replace to write over it.") unless `--replace`.
- **`import`** reads a `.ympreset` file (§5) into the folder; a name that exists exits 2 unless `--replace`; a file the core refuses exits 2 with its sentence. **`export`** writes the file atomically and **overwrites** an existing destination (like `cp`). **`remove`** moves the file to the Trash.
- **`--preset <name or file>`**, repeatable, on `render`. **Names resolve like `--look`:** the name matches ignoring case, exactly first and then as a prefix; **no match exits 3, several matches exit 2 and list them**. **A path to a `.ympreset` file works without importing it** (an argument with a `/` or ending `.ympreset` is a path; a path that is not a file exits 3). A preset that cannot be read exits 2 before any photo is opened, as a bad `--settings` does.
- **The layering is:** the photo's defaults, then `--settings` (wholesale, as today), then each `--preset` in order (each is §4.2's patch, applied with `DevelopSettings::apply_patch`), then the flags. **A flag always has the last word**: `--preset warm --look neutral` renders Neutral with the rest of Warm. `--preset` changes no preference and the Export panel's command preview (`cli_preview`) does not mention presets (it already says "N photos have adjustments this command doesn't include.").
- **A missing look** is a warning on stderr with the app's sentence, the render goes on, and the exit code is 0 (the look is not a requested look: `--look` is). `render --json` adds **`presets: [{ "name", "file", "missing_look" }]`** to each photo's record, in the order given, with `file` null for a name from the folder.
- **Parity.** `ym render --preset p` and the app's `preset_apply(p)` of the same photo give the same bytes (`crates/ym-cli/tests/parity.rs`).
- **Exit codes** are `docs/10`'s: 0, 1 (some photos failed), 2 usage, 3 a preset or look is missing.

## 12. New Preset from Exported Photo (P-06)

Looks › New Preset from Exported Photo… (and `ym presets save --from export.jpg`) reads the settings out of the XMP of a JPEG, TIFF, PNG or DNG that Yellowmoon exported, then opens the Save dialog (in `ym`: saves). The reader is `ym_export::xmp::read_ym(path)`: the JPEG's `APP1` segment, the TIFF and DNG's tag 700, the PNG's `XML:com.adobe.xmp` iTXt chunk (the places the writers put the packet: `docs/09`); it is our own code and reads only the `ym:` properties it needs (`ym:Settings`, `ym:AutoFields`, `ym:PipelineVersion` and `xmp:CreatorTool`). A file with no `ym:Settings` (not an export, or a JPEG whose settings were too long to record) gives "This file has no Yellowmoon settings. Choose a photo that Yellowmoon exported."

**The problem.** `ym:Settings` is written *pinned* (`ResolvedSettings::pinned_settings`, `docs/09`): the Input's exposure, sharpening and noise reduction and Main's contrast, saturation, highlights, shadows, black and shadows and highlights wheels, **when they were Auto, are written as the numbers they resolved to**, so the file reproduces its pixels. A preset made from such a file would hold fixed numbers where the photo had Auto, and give every photo it is applied to that photo's exposure. A white balance that came from a picked white is written as Auto with the pick's point, whose light is gone.

**The decision.** **P-06 adds one XMP property, `ym:AutoFields`,** to every export from then on, written beside `ym:Settings` (and left out when `ym:Settings` is):

```
ym:AutoFields = {"input":{"exposure":true,"light_source":true,"noise_reduction":true,"sharpen":true,"white_balance":true},
                 "main":{"balance_highlights":true,"balance_shadows":true,"black":true,"contrast":true,"highlights":true,"saturation":true,"shadows":true},
                 "picked_white":{"kelvin":5300,"tint":4}}
```

- `input` and `main` are `AutoFlags` (`docs/01`: which Auto fields were still Auto when the photo was exported), compact JSON with sorted keys (as `ym:AutoAdvisor`).
- `picked_white` is present only when the white balance came from a picked white (the `copied_from` condition) and holds the Kelvin and tint it resolved to.
- The core writes it from the `ResolvedSettings` it already holds in `export_photo` (`ym_core::export`, which fills `YmInfo`): `YmInfo::auto_fields: Option<String>`, a thirteenth property after `ym:Settings`.

**Reading it.** A preset made from a file with `ym:AutoFields` **restores Auto** for each field flagged `true` (in `input` and `main`) in a group that is saved, and a `picked_white` becomes the manual Kelvin and tint, by the rule of `copied_from`. The result equals what `from_photo` makes of the photo that was exported: **Auto stays Auto** (§4.1), and the pre-ticking compares that with the default settings.

**An older export** (no `ym:AutoFields`) cannot say which numbers were Auto. Its values stay fixed, as written; its white balance is read as written (Auto, a picked white lost). The Save dialog **says it in one line** (§9.3), and **the pre-ticking ignores the fields that `pinned_settings` may have pinned** (they would differ from the default on every photo): a group is ticked only when a field that is never pinned differs. The person can tick Light, Colour or Detail anyway.

**Why this and not something simpler.** The alternatives were (a) to always fix the values (wrong: the preset would carry one photo's exposure), (b) to write Auto as Auto in `ym:Settings` (breaks `ym:Settings`' promise that a file's settings reproduce its pixels through `--settings`, and the golden and XMP tests that rest on it), and (c) to read the photo's Auto estimates from the file (they are not in it). A separate property leaves `ym:Settings` as it is, adds about 300 bytes, and is read only here.

**Effects on the export code (P-06 changes them).** `docs/09`'s table gains a row (one line); `crates/ym-export/src/xmp.rs` writes it in its fixed place and its unit test lists thirteen properties; `crates/ym-export/tests/structure.rs` and `ym-core`'s export tests that read the properties are updated; **no pixel changes** (the golden check compares pixels, not XMP: `golden check` 61 of 61 unchanged) and an export is still the same bytes for the same input, settings and version (rule 8). `PIPELINE_VERSION` does not change. A JPEG whose packet would not fit drops `ym:AutoFields` with `ym:Settings`.

## 13. Persistence and privacy

- **What persists, and where.** `presets/<name>.ympreset` and `presets/.lock` in the application folder, beside `looks/` (`docs/01` *Persistence*). Nothing else: no index, no thumbnails on disk, no preference (the card's Apply to and folding are not remembered).
- **A preset is about no photo.** It has no path, no file name, no camera, no date and no location. It does hold what the person typed or chose: its name, the look's name and id, and **the names the person gave their nodes** ("Skin", "Sky"), which travel with the file when it is shared; the Export… item and this section are where that is said. A mask's own numbers (positions, feather) are in the file; the pictures a mask was made from are not.
- **No network.** Presets are files. Nothing is sent, fetched or checked (hard rule 7).
- **Originals are never touched** (rule 5). A preset write goes to `presets/` only.
- **The owner's decision**, 2026-11-16, "Go with your recommendations on presets", is the reason rule 4 changed; it is quoted in `docs/01` and `docs/16`.

## 14. Tests

Per task; each is named so that a failing one says what broke.

- **P-02 (core).** `crates/ym-core/tests/presets.rs`: every field of `DevelopSettings` is in exactly one group, plus the three never saved and `pos`; a preset round trip per group (save from a photo, apply to a default photo, the group's fields equal and every other field untouched); **Auto stays Auto** (a photo with Auto exposure and a preset of Light: the target's exposure is Auto, and a manual exposure is carried); a picked white is the manual Kelvin and tint, the same numbers as `copied_from`; Object clicks and Brush strokes are not saved (the row is there, empty), Subject, People, Background, Depth, Linear, Radial and the ranges are kept; the Input's crop, orientation and white pick are never written and never touched; **with `nodes` the other nodes are replaced and without it they stay**; two presets stack and the later one wins on a shared group; `matches` is true after an apply and false after any change to a field of a group (and true with a missing look when the rest matches); a missing look applies the rest and names the look; pre-ticking per group with Apple Auto's values on the photo; the report and the undo pair equal Paste's (`before`/`after` restore the photo exactly); a patch that fails on one photo changes none. `tests/presets_file.rs`: the file bytes are the same for the same settings; version 2, an unknown top-level field and an unknown settings field are refused with the sentence; an older preset (fields missing) reads with defaults; a file that is not a preset, too large, or of the flat form of 0.1 is "This file isn't a Yellowmoon preset."; the schema against what the core writes and against refusals. `tests/presets_store.rs`: atomic writes (a failing rename leaves the old file whole), the lock excludes a second writer, names (trim, 64, case-insensitive uniqueness, Unicode, the safe file name and its collisions), the sorted list, a damaged file listed and never fatal, rename and replace, import with and without `replace`, Delete moves to a fake Trash, `YELLOWMOON_HOME`, and **a full open, edit and export cycle writes no file in `presets/`** (the `tests/core.rs` tree comparison). The wire types' field names against `src/api/types.ts`.
- **P-03 (CLI).** `crates/ym-cli/tests/presets.rs`: each subcommand, `--json`, the exit codes (2 on a name that exists, 3 on a missing preset, 2 on several matches with the list), a path to a file without importing, **the layering** (`--settings`, then presets in order, then flags: `--preset a --preset b` equals the stacked apply, a flag beats both), a missing look warns and exits 0, `render --json`'s `presets`; `tests/parity.rs`: **app and `ym` byte for byte** for a preset with and without `nodes`; `tests/docs.rs` with the new rows of `docs/10` (and `ym presets --help` snapshots).
- **P-04 (app).** `src-tauri/src/api/tests.rs` (every command, the events, `preset_apply` queues thumbnails and emits `settings_changed` once a photo, a gone look, `preset_thumbnail` and its budget), `commands.rs` (the IPC names), `src/api.mock.test.ts` (the mock repeats the rules); `src/state/presets.test.ts` (pure helpers, notices, the 6-tile rule), `src/state/store.presets.test.ts` (the scope is fixed at the click, order of calls, undo per photo, the Apply to resets), `src/screens/tools/PresetsCard.test.tsx` (the card, the empty state, ring, tooltip, damaged rows, Show all), `SavePresetDialog.test.tsx` (pre-ticking, name rules, the question, the disabled reason), `e2e/presets.spec.ts` (save, apply to picked, update, rename, delete, the notice), the screenshot states of §15 added to `design/screenshots-c2/`.
- **P-05 (menu, drops, accessibility).** `menu.rs` tests (the five items, ⌥⌘S is a new key equivalent and is not a text key, ten equivalents with the albums' ⌘S and ⇧⌘S, the state items, the Apply Preset submenu from a fake list), `src/state/menu.test.ts`, `src/state/shortcuts.test.ts` (nothing new), a drop of a `.ympreset` with and without ARW files on Start and Photos, `e2e/axe.spec.ts`, `focus-order.spec.ts`, `contrast.spec.ts` and `glass.spec.ts` with the new states, the import's question for a name that exists; **the Tone card's removal** (§9.10): no `Tone` region on the Look tab, the Automatic card's Contrast and Saturation arrows open Light and Colour, the e2e and Vitest cases of the card gone, and a test that fails while a doc still says "Look › Tone".
- **P-06 (export).** `ym_export::xmp` unit tests (thirteen properties in order, the escaping, left out with `ym:Settings`), `read_ym` on a JPEG, a TIFF, a PNG and a DNG from the writers (round trip: what `build_xmp` wrote is what `read_ym` reads, including a file with XMP from another tool before ours), `tests/structure.rs`, a preset made from an export equals the preset made from the photo that was exported (for Auto exposure, Apple Auto's values, a picked white and a manual photo), an older export (no `ym:AutoFields`) says so and ignores the pinned fields in the pre-tick, the golden check 61 of 61 unchanged.

## 15. Tasks

**P-01 · Spec, rule change, mock-ups** (done 2026-11-16). `docs/30`, `AGENTS.md` rule 4, `docs/00`, `docs/01` *Persistence*, `docs/09`, `docs/11` (two paragraphs), `docs/15`, `docs/16`, `docs/STATUS.md`, `design/presets/`. *Done when:* the documents agree, every link resolves, `cargo test -p ym-cli --test docs` passes, and the lead has read it.

**P-02 · The core** (P-01) — `ym_core::presets` (§7): the groups, `Preset` and its file with the schema, `PresetStore` and the service, `from_photo`, `patch_for`, `matches`, `changed_groups`, the missing look, `Core::preset_*` and the events, `MaskRow::emptied_for_preset`; `docs/01` and `docs/23` §3.8 get a pointer. No interface. *Done when:* the tests of §14 for P-02 pass, `cargo clippy` and `cargo deny check` are clean, and a preset applied through the core equals the same patch pasted.

**P-03 · The command line** (P-02) — `ym presets …` and `render --preset` (§11), `docs/10`'s rows, the help snapshots, `render --json`. *Done when:* the CLI tests and the parity test pass, and `docs.rs` reads the new rows.

**P-04 · The shell, the IPC and the card** (P-02; P-H1 for the look of it) — the Presets card goes after the Look card (the Tone card still follows it until P-05 removes it, §9.10), the commands and the event of §8, the mock, `PresetsCard`, `SavePresetDialog`, the tile menu, the thumbnails, the store. *Done when:* the tests of §14 pass, the glass audit finds no hand-written control, and the eight states of `design/presets/` are matched by `design/screenshots-c2/` pictures taken from the app (looked at by the owner before they are committed).

**P-05 · Menu, drops, files, the Tone card's removal and accessibility** (P-04) — the Looks menu (§9.8), the Apply Preset submenu, Import Presets… and the drop, Export…, Show Presets Folder, the question for an existing name, `docs/28`'s rows and the axe, focus and contrast states. **The Tone card is removed from the Look tab and the Presets card is its last card** (§9.10: the component, its tests, the e2e, a11y, glass-audit and screenshot cases that show it, and every doc reference to Look › Tone; the Automatic card's Contrast and Saturation arrows open Light and Colour). *Done when:* the tests of §14 pass and `docs/28`'s checklist has the preset lines.

**P-06 · New Preset from Exported Photo** (P-04) — `ym:AutoFields`, `read_ym`, `preset_inspect` from a file, the menu item, the dialog's line, `ym presets save --from export.jpg`, `docs/09`. *Done when:* the tests of §14 pass and the golden check is unchanged.

**[HUMAN] P-H1 · Review the mock-ups and the rule change** (`design/presets/`, hard rule 4 as reworded, and §17 decision 1 on how many tiles show). Nothing of P-04 starts before the owner has looked. **[HUMAN] P-H2 · Try the build** of P-04 to P-06 on a Mac: save, apply to picked photos, update, rename, delete, drop a file, the menu, the key; look at the dialogs over a bright photo.

**Gate P:** the tests of §14 pass; `golden check` unchanged; `cargo deny check`; the app and `ym` byte for byte on the preset parity cases; a full session writes nothing to `presets/` unless a preset changes; the owner's approval of the last test build.

## 16. Later (not designed; each is its own task when the owner says so)

- **Phase 2 · P-07, relative values.** A preset may hold exposure as an offset in EV and warmth as an offset in mired over the photo's Auto, as `--exposure-offset` and Apple Auto's white nudge already work, so "a stop darker" and "warmer" travel to photos with other automatic values. It needs a format version 2 and a field that says relative.
- **Phase 2 · P-08, new photos open with a preset.** The Default look preference becomes "New photos open with" a look or a preset. A change to `preferences.json` (`docs/01`) and to `Session::open`'s defaults.
- **Phase 3 · P-09, node presets.** Save Node as Preset… on a node and, in the Add node menu, "From a preset": one node's settings as a preset of its own kind.
- **Phase 3 · P-10, presets that carry instructions for the Auto grading provider** (`docs/22`): a preset names text for Apple Intelligence, Claude Code or Antigravity to grade by.
- **[HUMAN] P-H3 · The owner's starter presets: not wanted.** The owner, 2026-11-16: "No default presets are needed for now." No preset ships with the app, there is no `data/presets/`, and the Presets card starts empty (§9.2) on every new install.

## 17. Decisions made while writing the spec

The simplest reading that meets the acceptance criteria was taken wherever the proposal left a detail open.

1. **Six tiles, then "Show all N".** The proposal said "at most 12 tiles until Show all 14" and asked for six tiles with "Show all 14" in the first picture. Twelve tiles at three to a row are four rows, a card as tall as the panel's visible part; six are two rows. §9.1 shows six; the owner may choose twelve (a one-number change) at P-H1.
2. **Presets are named, and the name is the identity** (case-insensitive). No id, no separate display name. Rename is a command.
3. **`ym:AutoFields`** (§12) rather than changing `ym:Settings`. It carries `AutoFlags` and a `picked_white`. **An older export** keeps its fixed numbers, says so, and does not pre-tick the groups whose only difference is a field `pinned_settings` may have pinned.
4. **⌥⌘S is a native menu key** (§9.8), with the reasons; the item is off while a dialog is open. (⇧⌘S went to the albums' Save As…, `docs/31`.)
5. **Checkboxes** in the Save dialog come from glasscn's registry (§9.3). If the registry has none, P-04 asks before it writes one by hand, as the glass rule requires.
6. **`ym presets save` has `--replace`** and `export` overwrites its destination.
7. **Names resolve exactly, then by prefix**, so that "several matches exit 2" has a meaning (§11).
8. **Looks are matched by id.** A look that is gone or changed counts as missing (the same test as an apply of a look).
9. **Update from This Photo asks first**; Delete does not (the Trash keeps it). Rename refuses an existing name with a line under the field; Save asks "Replace …?".
10. **Brush strokes are not saved, Object clicks are not saved** (§3), and the Find kinds are kept. The difference from Copy settings is on purpose.
11. **`pos` goes with Nodes**, and a preset never clears a Pick white point (a paste does not either).
12. **The import's name conflicts** come back to the page, which asks the question; nothing is replaced behind the person's back.
13. **The Apply Preset submenu** shows at most 30 names; the card has the rest.
14. **No cap on the number of presets**; a file above 4 MiB is refused.
15. **Replace cannot be undone** (the old file is not kept); the question says so.
16. **Presets replace the Tone card** (the owner, 2026-11-16, after seeing the mock-ups: "Presets should be at the end after tone on Look panel. In fact, remove tone section from Look and replace with Presets"). The Look tab is Automatic · Look · Presets (§9.10). It supersedes the place of the first proposal (between Automatic and Look). The Tone card's three controls were already on Light, Colour and Colour; nothing moves. P-05 removes the code, the tests and the doc references; until then the app and `docs/11` still have the card.
17. **The webview's readings (P-04, P-05, 2026-11-16).** Where the spec was silent the simplest reading was taken:
    - **The list is sorted by name** (§6), also in the card: with fourteen presets the first six tiles are the first six by name.
    - **Show all N / Show fewer** appears only with more than six presets.
    - **The tooltip** is the groups, "Look, Light, Colour"; for a missing look a second line follows: "Look: Film warm isn't imported."
    - **Apply Preset in the Looks menu** applies to the card's Apply to (Look's own Apply to is separate), as the keys 1 to 9 use Look's.
    - **The import's refusals:** the first is named with its reason, the others counted: "Couldn't add Old.ympreset: <reason> Couldn't add 2 more files."
    - **`ImportResult`** is the wire name of §8 (§7's `ImportOutcome` is the core's own type). `PresetsListing` is the answer of `presets_list`, `PresetInspect` of `preset_inspect`, `PresetSource` its argument.
    - **`preset_choose_export()` → `string | null`** is added to §8: the system open panel for New Preset from Exported Photo…, because the menu event carries no path. P-06 reads the file with `preset_inspect({ kind: "file", path })`.
    - **Update's question** puts the focus on Cancel, as Replace's; its line names the groups "Look, Light and Colour".
    - **Rename** starts with the current name selected; Save… in the header is off while the photo is not developed.
    - **The Tone card's removal** is done in the webview (the component, its Vitest and e2e cases); the Look tab's doc references (§9.10's list) are the docs task of P-05 and are not yet changed.

