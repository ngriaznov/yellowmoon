import { describe, expect, it } from "vitest";
import type { PhotoStatus, PhotoSummary, SessionState } from "../api/types";
import { bytesToUrl, cameraLine, captureDate, developFraction, folderSubline, mergeSession, photoName, shutterLabel } from "./photo";

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

  it("takes the folder's date from the first photo that has one, and has no separator without any", () => {
    // The core sends "" for a file that could not be probed or has no capture time.
    const undated = photo({ id: 1, captured_at: "", status: { kind: "skipped", reason: "Can't read this." } });
    expect(captureDate(undated)).toBe("");
    expect(folderSubline({ photos: [undated, photo({ id: 2 }), photo({ id: 3 })] })).toBe("2026-10-04 · 3 photos");
    expect(folderSubline({ photos: [undated, photo({ id: 2, captured_at: "" })] })).toBe("2 photos");
    expect(folderSubline({ photos: [undated] })).toBe("1 photo");
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

describe("mergeSession", () => {
  const session = (photos: PhotoSummary[], over: Partial<SessionState> = {}): SessionState => ({
    folder_name: "Lisbon",
    photos,
    current_index: 0,
    picked: photos.filter((p) => p.picked).map((p) => p.id),
    default_export_folder: "Lisbon export",
    ...over,
  });

  it("keeps the page's own photos and current photo, and appends the photos that are new", () => {
    const page = session([photo({ id: 1 }), photo({ id: 2, picked: true }), photo({ id: 3 })], { current_index: 2 });
    const snapshot = session(
      [photo({ id: 1, status: { kind: "developing", stage: "decode" } }), photo({ id: 2 }), photo({ id: 3 }), photo({ id: 4, status: { kind: "waiting" } })],
      { current_index: 0, folder_name: "Lisbon and Porto", default_export_folder: "Lisbon and Porto export", opened: { added: 1, already_open: 0 } },
    );
    const merged = mergeSession(page, snapshot);
    expect(merged.photos.map((p) => p.id)).toEqual([1, 2, 3, 4]);
    expect(merged.photos[0]?.status).toEqual({ kind: "ready" });
    expect(merged.photos[3]?.status).toEqual({ kind: "waiting" });
    expect(merged.current_index).toBe(2);
    expect(merged.picked).toEqual([2]);
    expect(merged.folder_name).toBe("Lisbon and Porto");
    expect(merged.default_export_folder).toBe("Lisbon and Porto export");
    expect(merged.opened).toEqual({ added: 1, already_open: 0 });
  });

  it("takes the snapshot's current photo when the page had none, and adds nothing for photos it already has", () => {
    const page = session([], { current_index: null });
    const snapshot = session([photo({ id: 1 })], { current_index: 0 });
    expect(mergeSession(page, snapshot).current_index).toBe(0);
    const same = mergeSession(snapshot, snapshot);
    expect(same.photos).toHaveLength(1);
  });
});
