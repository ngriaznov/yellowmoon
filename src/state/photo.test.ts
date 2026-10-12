import { describe, expect, it } from "vitest";
import type { PhotoStatus, PhotoSummary } from "../api/types";
import { bytesToUrl, cameraLine, captureDate, developFraction, folderSubline, photoName, shutterLabel } from "./photo";

const photo = (over: Partial<PhotoSummary> = {}): PhotoSummary => ({
  id: 4,
  file_name: "DSC04204.ARW",
  camera: { body: "Sony A1", model: "ILCE-1", model_id: 384, profile: "A7R IV", profile_id: "sony-ilce-7rm4", profile_version: "1.0.0", match_kind: { kind: "default" } },
  iso: 400,
  shutter_s: 1 / 125,
  aperture_f: 2.8,
  captured_at: "2026-10-04T15:20:00",
  width_px: 8640,
  height_px: 5760,
  status: { kind: "ready" },
  picked: false,
  reduced: false,
  ...over,
});

describe("photo helpers", () => {
  it("drops the extension from a file name", () => {
    expect(photoName("DSC04204.ARW")).toBe("DSC04204");
    expect(photoName("a.b.arw")).toBe("a.b");
    expect(photoName("noext")).toBe("noext");
  });

  it("writes shutter speeds the way a camera does", () => {
    expect(shutterLabel(1 / 125)).toBe("1/125");
    expect(shutterLabel(0.5)).toBe("1/2");
    expect(shutterLabel(1)).toBe("1 s");
    expect(shutterLabel(2.5)).toBe("2.5 s");
    expect(shutterLabel(30)).toBe("30 s");
    expect(shutterLabel(0)).toBe("");
  });

  it("builds the camera line of the photo cell", () => {
    expect(cameraLine(photo())).toBe("Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8");
    expect(cameraLine(photo({ aperture_f: 4 }))).toContain("f/4");
    // The Export screen's caption has no camera body, as in its mockup.
    expect(cameraLine(photo(), { body: false })).toBe("DSC04204 · ISO 400 · 1/125 · f/2.8");
  });

  it("reads the capture date and builds the folder sub-line", () => {
    expect(captureDate(photo())).toBe("2026-10-04");
    expect(folderSubline({ photos: [photo(), photo({ id: 5 })] })).toBe("2026-10-04 · 2 photos");
    expect(folderSubline({ photos: [photo()] })).toBe("2026-10-04 · 1 photo");
    expect(folderSubline({ photos: [] })).toBe("0 photos");
  });

  it("weighs the stages of development for the progress bar", () => {
    const f = (status: PhotoStatus) => developFraction(status);
    expect(f({ kind: "waiting" })).toBe(0);
    expect(f({ kind: "developing", stage: "decode" })).toBeLessThan(f({ kind: "developing", stage: "estimate" }));
    expect(f({ kind: "developing", stage: "thumbnail" })).toBeLessThan(1);
    expect(f({ kind: "ready" })).toBe(1);
    expect(f({ kind: "skipped", reason: "x" })).toBe(1);
  });

  it("turns PNG bytes into an image URL", () => {
    const url = bytesToUrl([137, 80, 78, 71]);
    expect(url).toMatch(/^(data:image\/png;base64,|blob:)/);
  });
});
