import { describe, expect, it } from "vitest";
import {
  DEFAULT_TEMPLATE,
  expandNameTemplate,
  FILE_EXTENSION,
  MAX_STEM_BYTES,
  MAX_TEMPLATE_LENGTH,
  previewFileName,
  sanitizeStem,
  slugify,
  templateProblem,
  templateWarning,
} from "./template";

const parts = { name: "DSC04204", look: "film-warm", date: "2026-10-04", seq: 7, camera: "sony-a1" };

describe("expandNameTemplate", () => {
  it("fills the five tokens and pads {seq} to three digits", () => {
    expect(expandNameTemplate("{name}_{look}", parts)).toBe("DSC04204_film-warm");
    expect(expandNameTemplate("{date}_{camera}_{seq}", parts)).toBe("2026-10-04_sony-a1_007");
  });

  it("leaves what is not a token as typed", () => {
    expect(expandNameTemplate("{name}-{other}", parts)).toBe("DSC04204-{other}");
  });
});

describe("previewFileName", () => {
  it("adds the extension of the format", () => {
    expect(previewFileName(DEFAULT_TEMPLATE, parts, "jpeg")).toBe("DSC04204_film-warm.jpg");
    expect(previewFileName(DEFAULT_TEMPLATE, parts, "tiff")).toBe("DSC04204_film-warm.tif");
    expect(previewFileName(DEFAULT_TEMPLATE, parts, "png")).toBe("DSC04204_film-warm.png");
    expect(previewFileName(DEFAULT_TEMPLATE, parts, "dng")).toBe("DSC04204_film-warm.dng");
    expect(Object.keys(FILE_EXTENSION)).toEqual(["jpeg", "tiff", "png", "dng"]);
  });
});

describe("sanitizeStem", () => {
  // The cases of `stems_are_made_safe` in crates/ym-export/src/naming.rs: the panel promises the name the core writes.
  it("turns what a file name can't hold into underscores", () => {
    expect(sanitizeStem('a/b\\c:d*e?f"g<h>i|j')).toBe("a_b_c_d_e_f_g_h_i_j");
    expect(sanitizeStem("tab\there")).toBe("tab_here");
    expect(sanitizeStem("bell\u0007\u0085\u007f")).toBe("bell___");
  });

  it("drops leading dots and spaces and trailing dots and spaces", () => {
    expect(sanitizeStem("..hidden")).toBe("hidden");
    expect(sanitizeStem("  spaced out.  ")).toBe("spaced out");
    expect(sanitizeStem("\u00a0\u3000x. .")).toBe("x");
  });

  it("falls back to 'export' when nothing is left", () => {
    expect(sanitizeStem("")).toBe("export");
    expect(sanitizeStem("...")).toBe("export");
    expect(sanitizeStem(" . ")).toBe("export");
  });

  it("keeps at most 200 bytes, never cuts a character in two, and trims what the cut leaves at the end", () => {
    const long = sanitizeStem("é".repeat(300));
    expect(new TextEncoder().encode(long).length).toBeLessThanOrEqual(MAX_STEM_BYTES);
    expect(long).toBe("é".repeat(100));
    expect(sanitizeStem("😀".repeat(80))).toBe("😀".repeat(50));
    expect(sanitizeStem(`${"a".repeat(199)} b`)).toBe("a".repeat(199));
  });

  it("leaves a safe name alone", () => {
    expect(sanitizeStem("DSC04204_film-warm (2) é")).toBe("DSC04204_film-warm (2) é");
  });
});

describe("previewFileName and the core's sanitising", () => {
  it("shows the name the core writes for characters a file name can't hold", () => {
    expect(previewFileName("{name}*final", parts, "jpeg")).toBe("DSC04204_final.jpg");
    expect(previewFileName("{name}?{look}|{seq}", parts, "tiff")).toBe("DSC04204_film-warm_007.tif");
    expect(templateProblem("{name}*final")).toBeNull();
  });

  it("drops a trailing dot or space, so the extension follows the name", () => {
    expect(previewFileName("{name}.", parts, "png")).toBe("DSC04204.png");
    expect(previewFileName("{name} ", parts, "jpeg")).toBe("DSC04204.jpg");
  });

  it("cuts a very long expanded name at 200 bytes", () => {
    const stem = previewFileName(`${"é".repeat(100)}{name}`, parts, "jpeg").replace(/\.jpg$/, "");
    expect(new TextEncoder().encode(stem).length).toBeLessThanOrEqual(MAX_STEM_BYTES);
  });
});

describe("slugify", () => {
  it("makes lower case words joined by hyphens", () => {
    expect(slugify("Sony A1")).toBe("sony-a1");
    expect(slugify("  Film warm!  ")).toBe("film-warm");
    expect(slugify("ILCE-7RM4")).toBe("ilce-7rm4");
  });
});

describe("templateProblem", () => {
  it("accepts the default and any mix of tokens and text", () => {
    expect(templateProblem(DEFAULT_TEMPLATE)).toBeNull();
    expect(templateProblem("{date} {name} ({look}) {seq} {camera}")).toBeNull();
    expect(templateProblem("holiday-{seq}")).toBeNull();
  });

  it("asks for a name when the box is empty", () => {
    expect(templateProblem("")).toBe("Type a file name, for example {name}_{look}.");
    expect(templateProblem("   ")).toContain("Type a file name");
  });

  it("refuses path separators and a leading dot", () => {
    expect(templateProblem("a/b")).toBe("A file name can't contain / \\ or :.");
    expect(templateProblem("a\\b")).toContain("can't contain");
    expect(templateProblem("a:b")).toContain("can't contain");
    expect(templateProblem(".hidden")).toBe("A file name can't start with a dot.");
  });

  it("names the token that is not one, and unbalanced braces", () => {
    expect(templateProblem("{name}_{lookk}")).toBe("{lookk} is not a token. Use {name}, {look}, {date}, {seq} or {camera}.");
    expect(templateProblem("{name")).toBe("Close every { with a }.");
    expect(templateProblem("name}")).toBe("A } needs a { before it.");
    expect(templateProblem("{{name}}")).toBe("Close every { with a } before the next {.");
  });

  it("limits the length", () => {
    expect(templateProblem("a".repeat(MAX_TEMPLATE_LENGTH))).toBeNull();
    expect(templateProblem("a".repeat(MAX_TEMPLATE_LENGTH + 1))).toBe(`Use at most ${MAX_TEMPLATE_LENGTH} characters.`);
  });

  it("says nothing with copy that shouts or apologises", () => {
    for (const bad of ["", "a/b", "{x}", "{name", "name}", ".x", "a".repeat(300)]) {
      const message = templateProblem(bad) ?? "";
      expect(message).not.toMatch(/!|oops|sorry/i);
    }
  });
});

describe("templateWarning", () => {
  it("warns when several photos would share one name", () => {
    expect(templateWarning("{look}", 3)).toBe("Every photo would get the same name. Add {name} or {seq}.");
    expect(templateWarning("{look}", 1)).toBeNull();
    expect(templateWarning("{look}_{seq}", 3)).toBeNull();
    expect(templateWarning("{name}", 3)).toBeNull();
  });
});
