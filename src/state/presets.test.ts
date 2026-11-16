import { describe, expect, it } from "vitest";
import type { PresetEntry } from "../api/types";
import {
  appliedNotice,
  deletedNotice,
  exportedNotice,
  groupsText,
  importNotice,
  isAlbumPath,
  isPresetPath,
  nameExists,
  nameProblem,
  renamedNotice,
  renameClash,
  savedNotice,
  tileDescription,
  tileTooltip,
  TILES_SHOWN,
  updatedNotice,
  visiblePresets,
} from "./presets";

const entry = (over: Partial<PresetEntry> = {}): PresetEntry => ({
  name: "Warm evening",
  groups: ["look", "light", "color"],
  look: { id: "arri:3000:film-warm", name: "Film warm" },
  look_missing: false,
  matches: false,
  revision: "0123456789ab",
  ...over,
});

describe("the name rules (docs/30 section 9.3)", () => {
  it("asks for a name, trimmed, of 1 to 64 characters", () => {
    expect(nameProblem("")).toBe("Give the preset a name.");
    expect(nameProblem("   ")).toBe("Give the preset a name.");
    expect(nameProblem("Warm evening")).toBeNull();
    expect(nameProblem("  x  ")).toBeNull();
    expect(nameProblem("a".repeat(64))).toBeNull();
    expect(nameProblem("a".repeat(65))).toBe("Use 64 characters or fewer.");
    // Characters, not UTF-16 units.
    expect(nameProblem("é".repeat(64))).toBeNull();
    expect(nameProblem("🌙".repeat(64))).toBeNull();
    expect(nameProblem("🌙".repeat(65))).toBe("Use 64 characters or fewer.");
  });

  it("is one name ignoring case, and a preset may keep its own name", () => {
    const list = [entry(), entry({ name: "Cool morning" })];
    expect(nameExists(list, "warm EVENING")).toBe(true);
    expect(nameExists(list, "  Cool morning ")).toBe(true);
    expect(nameExists(list, "Night")).toBe(false);
    expect(nameExists(list, "warm evening", "Warm evening")).toBe(false);
    expect(renameClash("Cool morning")).toBe("A preset called Cool morning already exists.");
  });
});

describe("the groups as words", () => {
  it("lists them in the order of the groups, for the tooltip and for the description", () => {
    expect(groupsText(["color", "look", "light"])).toBe("Look, Light, Colour");
    expect(groupsText(["color", "look", "light"], true)).toBe("Look, Light and Colour");
    expect(groupsText(["raw"], true)).toBe("RAW");
    expect(groupsText(["nodes", "detail"], true)).toBe("Detail and Nodes");
  });

  it("says a missing look in the tooltip and in the description", () => {
    expect(tileTooltip(entry())).toBe("Look, Light, Colour");
    expect(tileDescription(entry())).toBe("Look, Light and Colour");
    const missing = entry({ look_missing: true });
    expect(tileTooltip(missing)).toBe("Look, Light, Colour\nLook: Film warm isn't imported.");
    expect(tileDescription(missing)).toBe("Look, Light and Colour. The look Film warm isn't imported.");
  });
});

describe("the card shows six tiles and then all of them", () => {
  it("cuts the list at six unless Show all is on", () => {
    const list = Array.from({ length: 14 }, (_, i) => i);
    expect(TILES_SHOWN).toBe(6);
    expect(visiblePresets(list, false)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(visiblePresets(list, true)).toHaveLength(14);
    expect(visiblePresets([1, 2], false)).toEqual([1, 2]);
  });
});

describe("the notices (docs/30 section 9.5)", () => {
  it("applied to one photo, to several, with some skipped, to none", () => {
    expect(appliedNotice("Warm evening", 1, 0, "DSC04204", null)).toBe("Applied Warm evening to DSC04204.");
    expect(appliedNotice("Warm evening", 5, 0, null, null)).toBe("Applied Warm evening to 5 photos.");
    expect(appliedNotice("Warm evening", 4, 1, null, null)).toBe("Applied Warm evening to 4 photos. 1 photo was skipped because it isn't developed.");
    expect(appliedNotice("Warm evening", 2, 3, null, null)).toBe("Applied Warm evening to 2 photos. 3 photos were skipped because they aren't developed.");
    expect(appliedNotice("Warm evening", 0, 2, null, null)).toBe("Nothing was applied: 2 photos aren't developed.");
    expect(appliedNotice("Warm evening", 0, 1, null, null)).toBe("Nothing was applied: 1 photo isn't developed.");
  });

  it("a missing look takes the place of the first sentence, and the skip sentence follows", () => {
    expect(appliedNotice("Warm evening", 1, 0, "DSC04204", "Film warm")).toBe("Warm evening uses the look Film warm, which isn't imported. Everything else was applied.");
    expect(appliedNotice("Warm evening", 3, 1, null, "Film warm")).toBe(
      "Warm evening uses the look Film warm, which isn't imported. Everything else was applied. 1 photo was skipped because it isn't developed.",
    );
  });

  it("saved, replaced, updated, renamed, deleted, exported", () => {
    expect(savedNotice("Warm evening", false)).toBe("Saved the preset Warm evening.");
    expect(savedNotice("Warm evening", true)).toBe("Replaced the preset Warm evening.");
    expect(updatedNotice("Warm evening", "DSC04204")).toBe("Updated Warm evening from DSC04204.");
    expect(renamedNotice("Warm evening", "Cool morning")).toBe("Renamed the preset Warm evening to Cool morning.");
    expect(deletedNotice("Warm evening")).toBe("Moved the preset Warm evening to the Trash.");
    expect(exportedNotice("Warm evening")).toBe("Exported the preset Warm evening.");
  });

  it("imports: one, several, with refusals named and counted, and nothing to say", () => {
    const none = { added: [], replaced: [], exists: [], refused: [] };
    expect(importNotice(none)).toBeNull();
    expect(importNotice({ ...none, added: ["Warm evening"] })).toBe("Added the preset Warm evening.");
    expect(importNotice({ ...none, added: ["a", "b", "c"] })).toBe("Added 3 presets.");
    expect(importNotice({ ...none, replaced: ["Warm evening"] })).toBe("Replaced the preset Warm evening.");
    const newer = { file: "Old.ympreset", reason: "This preset was made by a newer version of Yellowmoon. Update to use it." };
    expect(importNotice({ ...none, added: ["a", "b"], refused: [newer] })).toBe(
      "Added 2 presets. Couldn't add Old.ympreset: This preset was made by a newer version of Yellowmoon. Update to use it.",
    );
    expect(importNotice({ ...none, refused: [newer, newer, newer] })).toBe(
      "Couldn't add Old.ympreset: This preset was made by a newer version of Yellowmoon. Update to use it. Couldn't add 2 more files.",
    );
    expect(importNotice({ ...none, refused: [newer, newer] })).toContain("Couldn't add 1 more file.");
    // A name that exists has no notice: the page asks the question.
    expect(importNotice({ ...none, exists: [{ name: "Warm evening", path: "/x/Warm evening.ympreset" }] })).toBeNull();
  });

  it("knows the files of its own", () => {
    expect(isPresetPath("/a/Warm evening.ympreset")).toBe(true);
    expect(isPresetPath("/a/Warm evening.YMPRESET")).toBe(true);
    expect(isPresetPath("/a/x.ymalbum")).toBe(false);
    expect(isAlbumPath("/a/Lisbon.ymalbum")).toBe(true);
    expect(isAlbumPath("/a/Lisbon.ARW")).toBe(false);
  });
});
