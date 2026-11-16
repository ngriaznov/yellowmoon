/**
 * Presets in the store (`docs/30` sections 4, 8 and 9): the scope and the preset are fixed at the click, the calls reach the core in the
 * order of the clicks, each photo has its own undo step, the card's "Apply to" starts on This photo, and every notice is said.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { PresetApplyReport } from "../api/types";
import { createStore } from "./store";
import type { AppStore } from "./store";
import { inputOf, mainGradeOf, mainGradePatch, outputPatch } from "./settingsGraph";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  localStorage.clear();
  sessionStorage.clear();
});

async function onPhotos(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, presets: 14, ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => {
    const photos = store.getState().session?.photos ?? [];
    expect(photos).toHaveLength(12);
    expect(photos.every((p) => p.status.kind === "ready" || p.status.kind === "skipped")).toBe(true);
  });
  await store.openPhotos();
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  await store.refreshPresets();
  return { api, store };
}

const photoIds = (store: AppStore): number[] => (store.getState().session?.photos ?? []).map((p) => p.id);
const contrastOf = async (api: MockApi, id: number) => mainGradeOf((await api.getSettings(id)).settings).contrast;

describe("the preset list", () => {
  it("is read at the start and asked for again on presets_changed", async () => {
    const { api, store } = await onPhotos();
    expect(store.getState().presets?.presets).toHaveLength(14);
    const first = photoIds(store)[0]!;
    await api.presetSave("Extra", ["light"], { kind: "photo", photo_id: first }, false);
    await vi.waitFor(() => expect(store.getState().presets?.presets).toHaveLength(15));
  });

  it("starts the card's Apply to on This photo and folded, in every new store (nothing is remembered)", async () => {
    const { store } = await onPhotos();
    expect(store.getState().presetScope).toBe("this");
    expect(store.getState().presetsShowAll).toBe(false);
    store.setPresetScope("picked");
    store.setPresetsShowAll(true);
    stops.splice(0).forEach((stop) => stop());
    const again = createStore(createMockApi({ speed: Number.POSITIVE_INFINITY }), 0);
    expect(again.getState().presetScope).toBe("this");
    expect(again.getState().presetsShowAll).toBe(false);
    // It is separate from the Look card's.
    store.setApplyScope("all");
    expect(store.getState().presetScope).toBe("picked");
  });
});

describe("applying a preset", () => {
  it("to this photo: one call, the notice names the photo, one undo step", async () => {
    const { api, store } = await onPhotos();
    const spy = vi.spyOn(api, "presetApply");
    const id = store.currentPhoto()!.id;
    await store.applyPreset("Warm evening");
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith("Warm evening", [id]);
    expect(store.getState().notice).toBe(`Applied Warm evening to ${store.currentPhoto()!.file_name.replace(/\.ARW$/i, "")}.`);
    expect(await contrastOf(api, id)).toEqual({ kind: "manual", value: 12 });
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(await contrastOf(api, id)).toEqual({ kind: "auto" });
  });

  it("fixes the scope and the photos at the click: a pick made while it is on its way changes nothing", async () => {
    const { api, store } = await onPhotos();
    const ids = photoIds(store);
    await store.setPicked(ids[1]!, true);
    await store.setPicked(ids[2]!, true);
    store.setPresetScope("picked");
    const spy = vi.spyOn(api, "presetApply");
    const done = store.applyPreset("Warm evening");
    // The person changes the picks and the scope before the core has answered.
    store.setPresetScope("this");
    await store.setPicked(ids[3]!, true);
    await store.setPicked(ids[1]!, false);
    await done;
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]).toEqual(["Warm evening", [ids[1], ids[2]]]);
    expect(store.getState().notice).toBe("Applied Warm evening to 2 photos.");
  });

  it("reaches the core in the order of the clicks, whatever the first one takes", async () => {
    const { api, store } = await onPhotos();
    const real = api.presetApply.bind(api);
    const seen: string[] = [];
    vi.spyOn(api, "presetApply").mockImplementation(async (name, photoIds) => {
      seen.push(`start ${name}`);
      if (name === "Warm evening") await new Promise((resolve) => setTimeout(resolve, 30));
      const report = await real(name, photoIds);
      seen.push(`end ${name}`);
      return report;
    });
    const first = store.applyPreset("Warm evening");
    const second = store.applyPreset("Cool morning");
    await Promise.all([first, second]);
    expect(seen).toEqual(["start Warm evening", "end Warm evening", "start Cool morning", "end Cool morning"]);
  });

  it("gives each photo its own undo step", async () => {
    const { api, store } = await onPhotos();
    const ids = photoIds(store);
    await store.setPicked(ids[0]!, true);
    await store.setPicked(ids[1]!, true);
    store.setPresetScope("picked");
    await store.applyPreset("Warm evening");
    expect(await contrastOf(api, ids[0]!)).toEqual({ kind: "manual", value: 12 });
    expect(await contrastOf(api, ids[1]!)).toEqual({ kind: "manual", value: 12 });
    // The photo on screen is the first: undo takes the preset back from it alone.
    await store.undo();
    expect(await contrastOf(api, ids[0]!)).toEqual({ kind: "auto" });
    expect(await contrastOf(api, ids[1]!)).toEqual({ kind: "manual", value: 12 });
    await store.setCurrent(ids[1]!);
    await vi.waitFor(() => expect(store.getState().canUndo).toBe(true));
    await store.undo();
    expect(await contrastOf(api, ids[1]!)).toEqual({ kind: "auto" });
  });

  it("says that nobody is picked, and applies to all of them", async () => {
    const { api, store } = await onPhotos();
    const spy = vi.spyOn(api, "presetApply");
    store.setPresetScope("picked");
    await store.applyPreset("Warm evening");
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().notice).toBe("No photos are picked.");
    await store.applyPreset("Warm evening", "all");
    expect(spy.mock.calls[0]?.[1]).toHaveLength(12);
    expect(store.getState().notice).toBe("Applied Warm evening to 12 photos.");
  });

  it("counts the photos that are not developed and the missing look", async () => {
    const { api, store } = await onPhotos({ skipped: { DSC04207: "This file can't be opened." }, missingLooks: ["film-warm"] });
    await store.applyPreset("Warm evening", "all");
    expect(store.getState().notice).toBe(
      "Warm evening uses the look Film warm, which isn't imported. Everything else was applied. 1 photo was skipped because it isn't developed.",
    );
    void api;
  });

  it("shows the error of the core when the preset is gone, and reads the list again", async () => {
    const { api, store } = await onPhotos();
    await api.presetRemove("Warm evening");
    await store.applyPreset("Warm evening");
    expect(store.getState().error?.scope).toBe("presets");
    await vi.waitFor(() => expect(store.getState().presets?.presets.some((p) => p.name === "Warm evening")).toBe(false));
  });

  it("rings the tile of what the photo has, and not after another change", async () => {
    const { store } = await onPhotos();
    await store.applyPreset("Warm evening");
    await store.refreshPresets();
    expect(store.getState().presets?.presets.find((p) => p.name === "Warm evening")?.matches).toBe(true);
    await store.update(mainGradePatch({ brightness: 9 }));
    await store.refreshPresets();
    expect(store.getState().presets?.presets.find((p) => p.name === "Warm evening")?.matches).toBe(false);
  });
});

describe("saving, renaming, updating, deleting", () => {
  it("opens the Save dialog with the groups the core says, then saves and says it", async () => {
    const { store } = await onPhotos({ presets: 0 });
    await store.update(outputPatch({ look: "builtin:signature" }));
    await store.openSavePreset();
    const dialog = store.getState().presetDialog;
    expect(store.getState().overlay).toBe("preset");
    expect(dialog).toMatchObject({ kind: "save", changed: ["look"], olderExport: false });
    if (dialog?.kind !== "save") throw new Error("no save dialog");
    expect(await store.savePreset("  Warm evening ", ["look"], dialog.source, false)).toBe(true);
    expect(store.getState().overlay).toBeNull();
    expect(store.getState().notice).toBe("Saved the preset Warm evening.");
    expect(store.getState().presets?.presets.map((p) => p.name)).toEqual(["Warm evening"]);
  });

  it("says Replaced after a replace and does not save without a name or a group", async () => {
    const { store } = await onPhotos({ presets: 1 });
    const source = { kind: "photo", photo_id: store.currentPhoto()!.id } as const;
    expect(await store.savePreset("", ["look"], source, false)).toBe(false);
    expect(await store.savePreset("Name", [], source, false)).toBe(false);
    expect(await store.savePreset("Warm evening", ["light"], source, true)).toBe(true);
    expect(store.getState().notice).toBe("Replaced the preset Warm evening.");
  });

  it("opens the dialog for an exported photo with the name empty, and the older export's line is the core's say-so", async () => {
    const { api, store } = await onPhotos();
    api.queueChoice("exported", "/x/older export.jpg");
    await store.openSavePresetFromExport();
    expect(store.getState().presetDialog).toMatchObject({ kind: "save", source: { kind: "file", path: "/x/older export.jpg" }, olderExport: true });
  });

  it("renames, refusing a name that exists under the field, and says it", async () => {
    const { store } = await onPhotos();
    store.openRenamePreset("Warm evening");
    expect(store.getState().presetDialog).toEqual({ kind: "rename", name: "Warm evening" });
    expect(await store.renamePreset("Warm evening", "cool MORNING")).toBe("A preset called cool MORNING already exists.");
    expect(store.getState().overlay).toBe("preset");
    expect(await store.renamePreset("Warm evening", "")).toBe("Give the preset a name.");
    expect(await store.renamePreset("Warm evening", "Late sun")).toBeNull();
    expect(store.getState().notice).toBe("Renamed the preset Warm evening to Late sun.");
    expect(store.getState().overlay).toBeNull();
  });

  it("asks before it updates, then updates from the photo on screen", async () => {
    const { api, store } = await onPhotos();
    const spy = vi.spyOn(api, "presetUpdate");
    store.openUpdatePreset("Warm evening");
    const dialog = store.getState().presetDialog;
    expect(dialog).toMatchObject({ kind: "update", name: "Warm evening", groups: ["look", "light", "color"] });
    expect(spy).not.toHaveBeenCalled();
    if (dialog?.kind !== "update") throw new Error("no update question");
    await store.updatePreset(dialog.name, dialog.photoId, dialog.photo);
    expect(spy).toHaveBeenCalledWith("Warm evening", dialog.photoId);
    expect(store.getState().notice).toBe(`Updated Warm evening from ${dialog.photo}.`);
  });

  it("deletes with no question and says where it went; exports and reveals", async () => {
    const { api, store } = await onPhotos();
    await store.removePreset("Warm evening");
    expect(store.getState().notice).toBe("Moved the preset Warm evening to the Trash.");
    expect(store.getState().presets?.presets.some((p) => p.name === "Warm evening")).toBe(false);
    await store.exportPreset("Cool morning");
    expect(store.getState().notice).toBe("Exported the preset Cool morning.");
    api.queueChoice("save", null);
    store.dismissNotice();
    await store.exportPreset("Cool morning");
    expect(store.getState().notice).toBeNull();
    await store.revealPreset("Cool morning");
    expect(api.system.revealed.at(-1)).toContain("Cool morning.ympreset");
  });
});

describe("importing and dropping presets", () => {
  it("adds a dropped preset on Start with the line under the drop zone, and starts no session", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openPaths(["/x/Warm evening.ympreset"]);
    expect(store.getState().session).toBeNull();
    expect(store.getState().dropNotice).toBe("Added the preset Warm evening.");
    expect(store.getState().notice).toBeNull();
  });

  it("says it in the notice bar on Photos, and a name that exists is asked about, one by one", async () => {
    const { api, store } = await onPhotos();
    await store.openPaths(["/x/Brand new.ympreset", "/x/Warm evening.ympreset", "/x/Cool morning.ympreset"]);
    expect(store.getState().notice).toBe("Added the preset Brand new.");
    expect(store.getState().presetDialog).toMatchObject({ kind: "replace-import" });
    expect(store.getState().overlay).toBe("preset");
    const spy = vi.spyOn(api, "presetsImport");
    await store.answerImportReplace(true);
    expect(spy).toHaveBeenCalledWith(["/x/Warm evening.ympreset"], true);
    expect(store.getState().presetDialog).toMatchObject({ kind: "replace-import", queue: [{ name: "Cool morning" }] });
    await store.answerImportReplace(false);
    expect(store.getState().overlay).toBeNull();
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("opens the ARW files of the same drop, and a drop of presets only never says there are no ARW files", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openPaths(["/x/a.ympreset", "/x/DSC04201.ARW"]);
    expect(store.getState().session?.photos.length).toBeGreaterThan(0);
    expect(store.getState().dropNotice).not.toBe("No ARW files in what you dropped.");
  });

  it("Looks › Import Presets… with a cancelled panel does nothing", async () => {
    const { api, store } = await onPhotos();
    await store.importPresetsDialog();
    expect(store.getState().notice).toBeNull();
    api.queueChoice("import", ["/x/Another.ympreset", "/x/Newer old.ympreset"]);
    await store.importPresetsDialog();
    expect(store.getState().notice).toBe("Added the preset Another. Couldn't add Newer old.ympreset: This preset was made by a newer version of Yellowmoon. Update to use it.");
  });
});

describe("the native menu and presets", () => {
  it("Save Preset and Apply Preset follow a developed photo and are off under a dialog", async () => {
    const { api, store } = await onPhotos();
    await vi.waitFor(() => expect(api.menuState).toMatchObject({ save_preset: true, apply_preset: true, save: true, save_as: true }));
    store.openOverlay("about");
    await vi.waitFor(() => expect(api.menuState).toMatchObject({ save_preset: false, apply_preset: false, save: false, save_as: false }));
    store.closeOverlay();
    await store.closeSession();
    await vi.waitFor(() => expect(api.menuState).toMatchObject({ save_preset: false, apply_preset: false, save: false, save_as: false }));
  });
});

describe("the report type", () => {
  it("is the paste report with the name of a missing look", async () => {
    const { api, store } = await onPhotos({ missingLooks: ["film-warm"] });
    const report: PresetApplyReport = await api.presetApply("Warm evening", [store.currentPhoto()!.id]);
    expect(report.missing_look).toBe("Film warm");
    expect(inputOf((await api.getSettings(store.currentPhoto()!.id)).settings).exposure).toBeDefined();
  });
});
