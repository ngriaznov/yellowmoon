/**
 * The look library as the screens list it: one section per library (`docs/08-looks.md` §Look model, owner's decision of 2026-10-16
 * in `docs/11-ui.md` §Look tab), the order the tiles and the keys 1–9 follow, and the names that tell two looks of the same title apart.
 *
 * The words here mirror `LookLibrary::label` and `LookLibrary::qualifier` in `crates/ym-looks/src/library/model.rs` (the CLI prints
 * those); `looks.test.ts` pins the strings.
 */
import type { Look, LookFormat, LookLibrary } from "../api/types";

/** One library: its section heading and the short word that tells its looks apart from a look of the same name elsewhere. */
export interface LookLibraryInfo {
  id: LookLibrary;
  /** The section heading: "ARRI looks · LogC3". */
  label: string;
  /** "LogC3"; null for the built-in looks, whose names need none. */
  qualifier: string | null;
}

/** Every library, in display order: the order of the sections, of the tile numbers and of the keys 1–9. */
export const LOOK_LIBRARIES: readonly LookLibraryInfo[] = [
  { id: "built_in", label: "Built in", qualifier: null },
  { id: "arri_logc4", label: "ARRI looks · LogC4", qualifier: "LogC4" },
  { id: "arri_logc3", label: "ARRI looks · LogC3", qualifier: "LogC3" },
  { id: "arri_logc3_to_rec709", label: "ARRI looks · LogC3 to Rec.709", qualifier: "LogC3 to Rec.709" },
  { id: "arri_alf2", label: "ARRI looks · LogC3 to Rec.709 · ALF2", qualifier: "ALF2" },
  { id: "user", label: "Your LUTs", qualifier: "Your LUTs" },
];

/** The libraries that are ARRI's own. The Look tab's call to action shows while the library holds none of their looks. */
export const ARRI_LIBRARIES: readonly LookLibrary[] = ["arri_logc4", "arri_logc3", "arri_logc3_to_rec709", "arri_alf2"];

const BY_ID: ReadonlyMap<LookLibrary, LookLibraryInfo> = new Map(LOOK_LIBRARIES.map((l) => [l.id, l]));

/** A library's heading. */
export function libraryLabel(library: LookLibrary): string {
  return BY_ID.get(library)?.label ?? library;
}

/** A library's short word; null for the built-in looks. */
export function libraryQualifier(library: LookLibrary): string | null {
  return BY_ID.get(library)?.qualifier ?? null;
}

/** Whether a look is one of ARRI's. */
export function isArriLook(look: Look): boolean {
  return ARRI_LIBRARIES.includes(look.library);
}

/** The looks in the order the tiles are drawn: library by library in display order, each in the core's order. This is also the order of the keys 1–9. */
export function looksInDisplayOrder(looks: readonly Look[]): Look[] {
  return LOOK_LIBRARIES.flatMap((l) => looks.filter((look) => look.library === l.id));
}

/** The looks of each library that has any, in display order. */
export function looksByLibrary(looks: readonly Look[]): Array<LookLibraryInfo & { looks: Look[] }> {
  return LOOK_LIBRARIES.map((l) => ({ ...l, looks: looks.filter((look) => look.library === l.id) })).filter((l) => l.looks.length > 0);
}

const nameKey = (name: string): string => name.trim().toLowerCase();

/**
 * A function that names looks for places where a name stands alone (the look chip, the default look choices): the look's name, plus
 * " · <qualifier>" ("Film · LogC3") when another look in a different library has the same name (case and outer spaces ignored).
 * Built-in looks have no qualifier, so a built-in "Film" stays "Film" while the ARRI one next to it says which library it is from.
 * Tiles inside a section keep the plain name: the section heading says the library. Build it once per list; it answers in constant time.
 */
export function lookLabeller(allLooks: readonly Look[]): (look: Look) => string {
  const libraries = new Map<string, Set<LookLibrary>>();
  for (const look of allLooks) {
    const key = nameKey(look.name);
    const seen = libraries.get(key);
    if (seen) seen.add(look.library);
    else libraries.set(key, new Set([look.library]));
  }
  return (look) => {
    const others = libraries.get(nameKey(look.name));
    const ambiguous = others !== undefined && [...others].some((library) => library !== look.library);
    const qualifier = ambiguous ? libraryQualifier(look.library) : null;
    return qualifier ? `${look.name} · ${qualifier}` : look.name;
  };
}

/** The look's name, qualified when another library has a look of the same name (see {@link lookLabeller}). */
export function lookLabel(look: Look, allLooks: readonly Look[]): string {
  return lookLabeller(allLooks)(look);
}

const FORMAT_TEXT: Record<LookFormat, string> = {
  cube: "3D LUT",
  alf4: "ALF4 look file",
  alf4c: "ALF4C look file",
  aml: "ALF2 look file",
};

/** What a look file is, for a tooltip: "3D LUT, 65 points", "ALF4 look file, 33 points". Null for a built-in look. */
export function lookKindText(look: Look): string | null {
  if (!look.format) return null;
  const kind = FORMAT_TEXT[look.format];
  return look.lut_size ? `${kind}, ${look.lut_size} points` : kind;
}

/** The tooltip of an imported look's tile: its section and what it is ("ARRI looks · LogC4 · 3D LUT, 65 points"). Undefined for a built-in look. */
export function lookTooltip(look: Look): string | undefined {
  const kind = lookKindText(look);
  return kind ? `${libraryLabel(look.library)} · ${kind}` : undefined;
}
