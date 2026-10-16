import { describe, expect, it } from "vitest";
import type { Look, LookFormat, LookLibrary } from "../api/types";
import {
  ARRI_LIBRARIES,
  isArriLook,
  libraryLabel,
  libraryQualifier,
  LOOK_LIBRARIES,
  lookKindText,
  lookLabel,
  lookLabeller,
  looksByLibrary,
  looksInDisplayOrder,
  lookTooltip,
} from "./looks";

/** A look of a library, with the kind of file and table size the library usually has. */
function look(id: string, name: string, library: LookLibrary, format: LookFormat | null = library === "built_in" ? null : "cube", lutSize: number | null = format ? 33 : null): Look {
  const arri = library.startsWith("arri");
  return {
    id,
    name,
    group: library === "built_in" ? { kind: "built_in" } : arri ? { kind: "arri", no: 3000, name: "Film" } : { kind: "user" },
    input: library === "arri_logc3" ? "logc3_ei800" : library === "arri_logc3_to_rec709" || library === "arri_alf2" ? "logc3_to_rec709" : "logc4",
    library,
    format,
    lut_size: lutSize,
    is_default: false,
    missing: false,
  };
}

describe("the libraries", () => {
  it("are six, in display order, with the words the CLI prints", () => {
    expect(LOOK_LIBRARIES.map((l) => [l.id, l.label, l.qualifier])).toEqual([
      ["built_in", "Built in", null],
      ["arri_logc4", "ARRI looks · LogC4", "LogC4"],
      ["arri_logc3", "ARRI looks · LogC3", "LogC3"],
      ["arri_logc3_to_rec709", "ARRI looks · LogC3 to Rec.709", "LogC3 to Rec.709"],
      ["arri_alf2", "ARRI looks · LogC3 to Rec.709 · ALF2", "ALF2"],
      ["user", "Your LUTs", "Your LUTs"],
    ]);
  });

  it("have a label each, and no two share one", () => {
    expect(new Set(LOOK_LIBRARIES.map((l) => l.label)).size).toBe(LOOK_LIBRARIES.length);
    expect(libraryLabel("arri_alf2")).toBe("ARRI looks · LogC3 to Rec.709 · ALF2");
    expect(libraryQualifier("arri_logc3")).toBe("LogC3");
    expect(libraryQualifier("built_in")).toBeNull();
  });

  it("call four of them ARRI's", () => {
    expect(ARRI_LIBRARIES).toEqual(["arri_logc4", "arri_logc3", "arri_logc3_to_rec709", "arri_alf2"]);
    expect(isArriLook(look("a", "A", "arri_alf2", "aml"))).toBe(true);
    expect(isArriLook(look("b", "B", "user"))).toBe(false);
    expect(isArriLook(look("c", "C", "built_in"))).toBe(false);
  });
});

describe("looksInDisplayOrder and looksByLibrary", () => {
  const mixed = [
    look("user:a", "A", "user"),
    look("arri:3000:b-alf2", "B", "arri_alf2", "aml"),
    look("builtin:n", "Neutral", "built_in"),
    look("arri:3000:c", "C", "arri_logc4", "cube", 65),
    look("arri:3000:d-logc3", "D", "arri_logc3"),
    look("arri:3000:e", "E", "arri_logc4", "cube", 65),
    look("arri:3000:f-rec709", "F", "arri_logc3_to_rec709"),
  ];

  it("puts the libraries in order and keeps the core's order inside each", () => {
    expect(looksInDisplayOrder(mixed).map((l) => l.id)).toEqual([
      "builtin:n",
      "arri:3000:c",
      "arri:3000:e",
      "arri:3000:d-logc3",
      "arri:3000:f-rec709",
      "arri:3000:b-alf2",
      "user:a",
    ]);
  });

  it("lists only the libraries that have looks", () => {
    expect(looksByLibrary(mixed).map((s) => [s.id, s.looks.length])).toEqual([
      ["built_in", 1],
      ["arri_logc4", 2],
      ["arri_logc3", 1],
      ["arri_logc3_to_rec709", 1],
      ["arri_alf2", 1],
      ["user", 1],
    ]);
    expect(looksByLibrary([look("builtin:n", "Neutral", "built_in")]).map((s) => s.id)).toEqual(["built_in"]);
    expect(looksByLibrary([])).toEqual([]);
  });
});

describe("lookLabel", () => {
  const all = [
    look("builtin:neutral", "Neutral", "built_in"),
    look("builtin:film", "Film", "built_in"),
    look("arri:3000:film", "Film", "arri_logc4", "cube", 65),
    look("arri:3000:film-logc3", "Film", "arri_logc3"),
    look("arri:3000:film-logc3-rec709", "Film", "arri_logc3_to_rec709"),
    look("arri:3000:film-low", "Film", "arri_alf2", "aml"),
    look("user:film", "  film ", "user"),
    look("arri:3000:warm", "Warm", "arri_logc4", "cube", 65),
    look("arri:3000:warm-2", "Warm", "arri_logc4", "cube", 65),
    look("arri:3000:only-logc3", "Only here", "arri_logc3"),
  ];
  const byId = (id: string): Look => all.find((l) => l.id === id)!;

  it("is the plain name when no other library has the name", () => {
    expect(lookLabel(byId("builtin:neutral"), all)).toBe("Neutral");
    expect(lookLabel(byId("arri:3000:only-logc3"), all)).toBe("Only here");
  });

  it("adds the library's short word when another library has a look of the same name", () => {
    expect(lookLabel(byId("arri:3000:film"), all)).toBe("Film · LogC4");
    expect(lookLabel(byId("arri:3000:film-logc3"), all)).toBe("Film · LogC3");
    expect(lookLabel(byId("arri:3000:film-logc3-rec709"), all)).toBe("Film · LogC3 to Rec.709");
    expect(lookLabel(byId("arri:3000:film-low"), all)).toBe("Film · ALF2");
  });

  it("ignores case and outer spaces when it compares names, and keeps the name as it is written", () => {
    expect(lookLabel(byId("user:film"), all)).toBe("  film  · Your LUTs");
  });

  it("gives a built-in look no word, so its name stays plain", () => {
    expect(lookLabel(byId("builtin:film"), all)).toBe("Film");
  });

  it("leaves a name that only repeats inside one library alone", () => {
    expect(lookLabel(byId("arri:3000:warm"), all)).toBe("Warm");
    expect(lookLabel(byId("arri:3000:warm-2"), all)).toBe("Warm");
  });

  it("is the plain name for a look the list does not hold", () => {
    expect(lookLabel(look("arri:3000:x", "Film", "arri_logc3"), [])).toBe("Film");
  });

  it("is built once for a list by lookLabeller, with the same answers", () => {
    const label = lookLabeller(all);
    expect(all.map(label)).toEqual(all.map((l) => lookLabel(l, all)));
  });
});

describe("what a look is", () => {
  it("says it for every kind of file, with the table size", () => {
    expect(lookKindText(look("a", "A", "arri_logc4", "cube", 65))).toBe("3D LUT, 65 points");
    expect(lookKindText(look("b", "B", "arri_logc4", "alf4", 33))).toBe("ALF4 look file, 33 points");
    expect(lookKindText(look("c", "C", "user", "alf4c", 33))).toBe("ALF4C look file, 33 points");
    expect(lookKindText(look("d", "D", "arri_alf2", "aml", 33))).toBe("ALF2 look file, 33 points");
  });

  it("leaves the size out when it is not known, and says nothing for a built-in look", () => {
    expect(lookKindText(look("a", "A", "user", "cube", null))).toBe("3D LUT");
    expect(lookKindText(look("n", "Neutral", "built_in"))).toBeNull();
    expect(lookTooltip(look("n", "Neutral", "built_in"))).toBeUndefined();
  });

  it("is the tooltip of a tile: its section, then what it is", () => {
    expect(lookTooltip(look("a", "A", "arri_logc4", "cube", 65))).toBe("ARRI looks · LogC4 · 3D LUT, 65 points");
    expect(lookTooltip(look("b", "B", "arri_logc4", "alf4", 33))).toBe("ARRI looks · LogC4 · ALF4 look file, 33 points");
    expect(lookTooltip(look("c", "C", "arri_alf2", "aml", 33))).toBe("ARRI looks · LogC3 to Rec.709 · ALF2 · ALF2 look file, 33 points");
    expect(lookTooltip(look("d", "D", "user", "alf4c", 33))).toBe("Your LUTs · ALF4C look file, 33 points");
  });
});
