/**
 * The words and the small rules of presets (`docs/30-presets.md` sections 9 and 10): the group labels and lines, the name rules, the tile's
 * tooltip and description, the six-tile rule, and every sentence of the notice bar. Pure functions, so a test pins each sentence.
 */
import type { ImportResult, PresetEntry, PresetGroup } from "../api/types";

/** The groups in the order of the dialog, the file and the help. */
export const PRESET_GROUP_ORDER: readonly PresetGroup[] = ["look", "lens", "light", "color", "detail", "curve", "raw", "nodes"];

/** The label of each group (British spelling, `docs/30` section 9.3). */
export const GROUP_LABELS: Readonly<Record<PresetGroup, string>> = {
  look: "Look",
  lens: "Lens",
  light: "Light",
  color: "Colour",
  detail: "Detail",
  curve: "Curve",
  raw: "RAW",
  nodes: "Nodes",
};

/** The quiet line under each box of the Save dialog. */
export const GROUP_LINES: Readonly<Record<PresetGroup, string>> = {
  look: "The look and how strongly it is applied.",
  lens: "The lens and its amounts.",
  light: "Exposure, contrast, brightness, highlights, shadows, white and black.",
  color: "White balance, colour balance, saturation and colour warp.",
  detail: "Sharpening, noise reduction and output sharpening.",
  curve: "The luma, red, green and blue curves.",
  raw: "Decode options and the range.",
  nodes: "The other nodes, their links and masks.",
};

/** The line under the boxes. */
export const AUTO_STAYS_AUTO = "Auto stays Auto, so each photo keeps its own exposure and white balance.";

/** The line for an export made by an older Yellowmoon (`docs/30` section 9.3). */
export const OLDER_EXPORT_LINE =
  "This photo was exported by an older Yellowmoon, so its automatic values are saved as fixed numbers. Untick Light, Colour and Detail to leave them out.";

/** The longest preset name. */
export const MAX_NAME_CHARS = 64;

/** How many tiles the card shows before "Show all" (`docs/30` section 9.1, decision 1). */
export const TILES_SHOWN = 6;

/** Why a name cannot be used, or null: trimmed, 1 to 64 characters (`docs/30` section 9.3). */
export function nameProblem(name: string): string | null {
  const trimmed = name.trim();
  if (trimmed === "") return "Give the preset a name.";
  if ([...trimmed].length > MAX_NAME_CHARS) return "Use 64 characters or fewer.";
  return null;
}

/** Whether a preset of this name is in the list (names are one name ignoring case). */
export function nameExists(presets: readonly PresetEntry[], name: string, except?: string): boolean {
  const key = name.trim().toLowerCase();
  return presets.some((p) => p.name.toLowerCase() === key && p.name !== except);
}

/** The refusal under the Rename field. */
export const renameClash = (name: string): string => `A preset called ${name} already exists.`;

/** The groups as a list: "Look, Light, Colour" (the tooltip), or "Look, Light and Colour" with `and` (the description). */
export function groupsText(groups: readonly PresetGroup[], and = false): string {
  const labels = PRESET_GROUP_ORDER.filter((g) => groups.includes(g)).map((g) => GROUP_LABELS[g]);
  if (!and || labels.length < 2) return labels.join(", ");
  return `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`;
}

/** The sentence about a look that is not in the library. */
export const missingLookLine = (look: string): string => `The look ${look} isn't imported.`;

/** The tile's tooltip: the groups, and for a missing look "Look: Film warm isn't imported." (`docs/30` section 9.1). */
export function tileTooltip(entry: PresetEntry): string {
  const groups = groupsText(entry.groups);
  return entry.look_missing && entry.look ? `${groups}\nLook: ${entry.look.name} isn't imported.` : groups;
}

/** The tile's accessible description: the groups, and the missing look (`docs/30` section 10). */
export function tileDescription(entry: PresetEntry): string {
  const groups = groupsText(entry.groups, true);
  return entry.look_missing && entry.look ? `${groups}. The look ${entry.look.name} isn't imported.` : groups;
}

/** The tiles the card draws: six, or all of them. */
export function visiblePresets<T>(presets: readonly T[], showAll: boolean): readonly T[] {
  return showAll ? presets : presets.slice(0, TILES_SHOWN);
}

const photosOf = (n: number): string => `${n} ${n === 1 ? "photo" : "photos"}`;

/**
 * What the notice bar says after Apply (`docs/30` section 9.5). `only` is the name of the photo when it was one photo ("DSC04204").
 * `applied` counts the photos that now have the settings; `skipped` the ones left out because they are not developed.
 */
export function appliedNotice(name: string, applied: number, skipped: number, only: string | null, missingLook: string | null): string {
  if (applied === 0) return `Nothing was applied: ${photosOf(skipped)} ${skipped === 1 ? "isn't" : "aren't"} developed.`;
  const done =
    missingLook !== null
      ? `${name} uses the look ${missingLook}, which isn't imported. Everything else was applied.`
      : applied === 1 && only !== null
        ? `Applied ${name} to ${only}.`
        : `Applied ${name} to ${photosOf(applied)}.`;
  if (skipped === 0) return done;
  return `${done} ${photosOf(skipped)} ${skipped === 1 ? "was" : "were"} skipped because ${skipped === 1 ? "it isn't" : "they aren't"} developed.`;
}

/** "Saved the preset Warm evening." or, after a replace, "Replaced the preset Warm evening." */
export const savedNotice = (name: string, replaced: boolean): string => `${replaced ? "Replaced" : "Saved"} the preset ${name}.`;
/** After Update from This Photo. */
export const updatedNotice = (name: string, photo: string): string => `Updated ${name} from ${photo}.`;
/** After Rename. */
export const renamedNotice = (name: string, to: string): string => `Renamed the preset ${name} to ${to}.`;
/** After Delete. */
export const deletedNotice = (name: string): string => `Moved the preset ${name} to the Trash.`;
/** After Export. */
export const exportedNotice = (name: string): string => `Exported the preset ${name}.`;
/** With no photo picked. */
export const NO_PICKED_NOTICE = "No photos are picked.";

/**
 * What the notice bar says after an import (`docs/30` section 9.5): "Added the preset Warm evening.", "Added 3 presets.", and the first
 * refusal named with the others counted. `null` when nothing was added, replaced or refused (a cancelled panel, or only names that exist:
 * the page asks the question for those).
 */
export function importNotice(result: ImportResult): string | null {
  const done = [...result.added.map((name) => ({ name, replaced: false })), ...result.replaced.map((name) => ({ name, replaced: true }))];
  const parts: string[] = [];
  if (done.length === 1 && done[0]) parts.push(`${done[0].replaced ? "Replaced" : "Added"} the preset ${done[0].name}.`);
  else if (done.length > 1) parts.push(`Added ${done.length} presets.`);
  const [first, ...rest] = result.refused;
  if (first) {
    parts.push(`Couldn't add ${first.file}: ${first.reason}`);
    if (rest.length > 0) parts.push(`Couldn't add ${rest.length} more ${rest.length === 1 ? "file" : "files"}.`);
  }
  return parts.length === 0 ? null : parts.join(" ");
}

/** True for the path of a preset file. */
export const isPresetPath = (path: string): boolean => /\.ympreset$/i.test(path);

/** True for the path of an album file. */
export const isAlbumPath = (path: string): boolean => /\.ymalbum$/i.test(path);

/** Said when an album is chosen or dropped with photos (`docs/31` section 4.8). */
export const ALBUM_AND_PHOTOS = "Open an album on its own, or open photos. Yellowmoon can't open both in one go.";
/** Said when two albums are chosen. */
export const TWO_ALBUMS = "Open one album at a time.";

/** The question over an unsaved album's close or an open (`docs/31` section 6). */
export const albumQuestion = (name: string): string => `Save changes to the album ${name}?`;
/** After File › Save. */
export const albumSavedNotice = (name: string): string => `Saved the album ${name}.`;
