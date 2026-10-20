import { describe, expect, it } from "vitest";
import type { PhotoSummary, SessionState } from "../api/types";
import { exportModel, FALLBACK_DESTINATION } from "./model";
import { buildJob, defaultExportOptions, defaultScope, fullLongEdge, isExportable, photosInScope, sameOptions } from "./options";

function photo(id: number, patch: Partial<PhotoSummary> = {}): PhotoSummary {
  return {
    id,
    file_name: `DSC0420${id}.ARW`,
    camera: { body: "Sony A1", model: "ILCE-1", model_id: 384, profile: "A7R IV", profile_id: "sony-ilce-7rm4", profile_version: "1.0.0", match_kind: { kind: "default" } },
    iso: 400,
    shutter_s: 0.008,
    aperture_f: 2.8,
    captured_at: `2026-10-04T10:0${id}:00`,
    width_px: 8640,
    height_px: 5760,
    status: { kind: "ready" },
    picked: false,
    reduced: false,
    apple_pending: false,
    ...patch,
  };
}

function session(photos: PhotoSummary[], current = 0): SessionState {
  return {
    folder_name: "Lisbon",
    photos,
    current_index: current,
    picked: photos.filter((p) => p.picked).map((p) => p.id),
    default_export_folder: "~/Pictures/Lisbon export",
  };
}

const THREE = session([photo(1), photo(2, { picked: true }), photo(3, { picked: true }), photo(4, { status: { kind: "skipped", reason: "Can't read this." } })], 0);

describe("scopes", () => {
  it("covers the current photo, the picked ones, or every photo that can be exported", () => {
    expect(photosInScope(THREE, "this").map((p) => p.id)).toEqual([1]);
    expect(photosInScope(THREE, "picked").map((p) => p.id)).toEqual([2, 3]);
    expect(photosInScope(THREE, "all").map((p) => p.id)).toEqual([1, 2, 3]);
  });

  it("leaves out a file that can't be opened, even when it is the current photo", () => {
    const s = { ...THREE, current_index: 3 };
    expect(photosInScope(s, "this")).toEqual([]);
    expect(isExportable(photo(9, { status: { kind: "developing", stage: "decode" } }))).toBe(true);
  });

  it("is empty without a session", () => {
    expect(photosInScope(null, "all")).toEqual([]);
  });

  it("opens on Picked when anything is picked, else This photo", () => {
    expect(defaultScope(THREE)).toBe("picked");
    expect(defaultScope(session([photo(1), photo(2)]))).toBe("this");
    expect(defaultScope(null)).toBe("this");
  });
});

describe("full size", () => {
  it("is the longest side of the biggest photo", () => {
    expect(fullLongEdge([photo(1), photo(2, { width_px: 5760, height_px: 8640 }), photo(3, { width_px: 4000, height_px: 3000 })])).toBe(8640);
    expect(fullLongEdge([])).toBeNull();
  });
});

describe("options", () => {
  it("start as docs/09 says", () => {
    expect(defaultExportOptions()).toEqual({
      format: "jpeg",
      jpeg_quality: 92,
      bit_depth: 16,
      tiff_compression: "lzw",
      color_space: "srgb",
      size: { kind: "full" },
      destination: null,
      name_template: "{name}_{look}",
      if_exists: "keep_both",
    });
  });

  it("compares by value, and builds the job from a scope", () => {
    expect(sameOptions(defaultExportOptions(), defaultExportOptions())).toBe(true);
    expect(sameOptions(defaultExportOptions(), { ...defaultExportOptions(), jpeg_quality: 80 })).toBe(false);
    expect(buildJob("picked", defaultExportOptions())).toEqual({ scope: "picked", options: defaultExportOptions() });
  });
});

describe("exportModel", () => {
  const input = { session: THREE, settings: {}, looks: null, preferences: null };

  it("counts and names the photos in the job", () => {
    const m = exportModel(input, "picked", defaultExportOptions());
    expect(m).toMatchObject({ count: 2, nPicked: 2, nAll: 3, fullEdge: 8640, destination: "~/Pictures/Lisbon export" });
    expect(m.names).toEqual(["DSC04202_neutral.jpg", "DSC04203_neutral.jpg"]);
    expect(m.example).toBe("DSC04202_neutral.jpg");
    expect([...m.included]).toEqual([2, 3]);
    expect(m.problem).toBeNull();
  });

  it("uses the extension of the format and the template the person typed", () => {
    const m = exportModel(input, "all", { ...defaultExportOptions(), format: "tiff", name_template: "{date}_{seq}_{camera}" });
    expect(m.names).toEqual(["2026-10-04_001_sony-a1.tif", "2026-10-04_002_sony-a1.tif", "2026-10-04_003_sony-a1.tif"]);
  });

  it("names a photo with no capture time 'undated', as the core does", () => {
    const undated = session([photo(1, { captured_at: "" }), photo(2)]);
    const m = exportModel({ ...input, session: undated }, "all", { ...defaultExportOptions(), name_template: "{name}_{date}" });
    expect(m.names).toEqual(["DSC04201_undated.jpg", "DSC04202_2026-10-04.jpg"]);
  });

  it("names the look a photo has, else the default look", () => {
    const withLook = {
      ...input,
      settings: { 2: { settings: { look: "arri:3000:film-warm" } } } as never,
      looks: [{ id: "builtin:signature", name: "Signature", group: { kind: "built_in" }, input: "logc4", is_default: true, missing: false }] as never,
    };
    const m = exportModel(withLook, "picked", defaultExportOptions());
    expect(m.names).toEqual(["DSC04202_film-warm.jpg", "DSC04203_signature.jpg"]);
  });

  it("shows the chosen folder, else the session's, else the fallback", () => {
    expect(exportModel(input, "this", { ...defaultExportOptions(), destination: "~/Exports" }).destination).toBe("~/Exports");
    expect(exportModel({ ...input, session: { ...THREE, default_export_folder: null } }, "this", defaultExportOptions()).destination).toBe(FALLBACK_DESTINATION);
  });

  it("reports a template that can't be used and gives no names", () => {
    const m = exportModel(input, "picked", { ...defaultExportOptions(), name_template: "{nope}" });
    expect(m.problem).toContain("{nope} is not a token");
    expect(m.names).toEqual([]);
    expect(m.example).toBe("");
  });

  it("warns about one name for many photos", () => {
    expect(exportModel(input, "picked", { ...defaultExportOptions(), name_template: "{look}" }).warning).toBe("Every photo would get the same name. Add {name} or {seq}.");
  });

  it("is empty for an empty scope", () => {
    const none = session([photo(1)]);
    const m = exportModel({ ...input, session: none }, "picked", defaultExportOptions());
    expect(m).toMatchObject({ count: 0, names: [], nPicked: 0, nAll: 1 });
    expect(m.fullEdge).toBe(8640);
  });
});
