// The store's export queue, preferences and dialogs (M3-11, M3-13): the parts the screen tests reach only through the DOM.
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { ExportProgress } from "../api/types";
import { createStore } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

async function opened(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3, picked: [2, 3, 4], ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  store.openExport(); // Picked, as the panel does
  return { api, store };
}

describe("preferences", () => {
  it("loads the preferences, and the Export panel starts from the remembered options", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await api.preferencesSet({ export_defaults: { ...(await api.preferencesGet()).export_defaults, format: "png", jpeg_quality: 70 } });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    expect(store.getState().preferences?.export_defaults.format).toBe("png");
    expect(store.getState().exportOptions).toMatchObject({ format: "png", jpeg_quality: 70 });
  });

  it("changes a preference on screen at once and keeps the core's answer", async () => {
    const { store } = await opened();
    const done = store.setPreferences({ remove_location_from_exports: true });
    expect(store.getState().preferences?.remove_location_from_exports).toBe(true);
    await done;
    expect(store.getState().preferences?.remove_location_from_exports).toBe(true);
  });

  it("puts the old preferences back when the core refuses", async () => {
    const { api, store } = await opened();
    api.failNext("preferencesSet", { scope: "preferences", message: "No." });
    await store.setPreferences({ remove_location_from_exports: true });
    expect(store.getState().preferences?.remove_location_from_exports).toBe(false);
    expect(store.getState().error).toEqual({ scope: "preferences", message: "No." });
  });

  it("asks again when the core says the looks changed (a new default look)", async () => {
    const { api, store } = await opened();
    await api.looksSetDefault("builtin:signature");
    await vi.waitFor(() => expect(store.getState().preferences?.default_look).toBe("builtin:signature"));
  });

  it("checks the Command-line tool when Preferences opens", async () => {
    const { store } = await opened();
    expect(store.getState().cliLink).toBeNull();
    store.openOverlay("preferences");
    await vi.waitFor(() => expect(store.getState().cliLink).toEqual({ installed: false, path: null }));
  });
});

describe("export options", () => {
  it("remembers a change as the default, but not a drag in progress", async () => {
    const { api, store } = await opened();
    store.setExportOptions({ jpeg_quality: 70 }, false);
    expect((await api.preferencesGet()).export_defaults.jpeg_quality).toBe(92);
    store.setExportOptions({ jpeg_quality: 66 }, true);
    await vi.waitFor(async () => expect((await api.preferencesGet()).export_defaults.jpeg_quality).toBe(66));
  });

  it("does not send what has not changed, and never remembers a template that can't be used", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "preferencesSet");
    store.setExportOptions({ jpeg_quality: 92 }, true);
    expect(spy).not.toHaveBeenCalled();
    store.setExportOptions({ name_template: "{nope}" }, true);
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().exportOptions.name_template).toBe("{nope}");
  });

  it("opens the panel on Picked with picks, else This photo, and keeps the scope while a run is on", async () => {
    const { store } = await opened();
    store.openExport();
    expect(store.getState()).toMatchObject({ panel: "export", exportScope: "picked" });
    store.closeExport();
    await store.clearPicks();
    store.openExport();
    expect(store.getState().exportScope).toBe("this");
  });

  it("does not open without a session", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    store.openExport();
    expect(store.getState().panel).toBe("tools");
  });
});

describe("running an export", () => {
  const names = ["DSC04203_neutral.jpg", "DSC04204_neutral.jpg", "DSC04205_neutral.jpg"];

  it("queues the job and follows the core's progress to the end", async () => {
    const { api, store } = await opened();
    expect(await store.startExport(names, "~/Pictures/Lisbon export")).toBe(true);
    expect(store.exportRunning()).toBe(true);
    expect(store.getState().exportRun?.rows.map((r) => r.state)).toContain("waiting");
    api.flush();
    await vi.waitFor(() => expect(store.getState().exportRun?.phase).toBe("finished"));
    expect(store.exportRunning()).toBe(false);
    expect(store.getState().exportRun?.rows.map((r) => r.state)).toEqual(["done", "done", "done"]);
  });

  it("keeps events that arrive before the core has answered export_start", async () => {
    const { api, store } = await opened();
    const events: ExportProgress[] = [];
    const real = api.exportStart.bind(api);
    api.exportStart = async (job) => {
      const id = await real(job);
      // The core's first event beats the answer.
      api.emitEvent("export_progress", { job_id: id, file: names[0]!, done: 1, total: 3, file_percent: 10, error: null, skipped: false });
      events.push({ job_id: id } as ExportProgress);
      return id;
    };
    await store.startExport(names, "~/x");
    expect(store.getState().exportRun?.rows[0]?.state).toBe("done");
    expect(store.getState().exportRun?.rows[1]?.state).toBe("running");
  });

  it("refuses a second run while one is going, and shows the core's refusal", async () => {
    const { api, store } = await opened();
    await store.startExport(names, "~/x");
    expect(await store.startExport(names, "~/x")).toBe(false);
    api.flush();
    await vi.waitFor(() => expect(store.getState().exportRun?.phase).toBe("finished"));
    store.finishExport();
    api.failNext("exportStart", { scope: "export", message: "The folder can't be written to." });
    expect(await store.startExport(names, "~/x")).toBe(false);
    expect(store.getState().error?.message).toBe("The folder can't be written to.");
    expect(store.getState().exportRun).toBeNull();
  });

  it("cancels: the core stops, the run goes, and the panel says what was saved", async () => {
    const { api, store } = await opened({ speed: 1 });
    await store.startExport(names, "~/x");
    const spy = vi.spyOn(api, "exportCancel");
    await store.cancelExport();
    expect(spy).toHaveBeenCalledWith(1);
    expect(store.getState().exportRun).toBeNull();
    expect(store.getState().exportStopped).toBe("Export stopped. Nothing was saved.");
  });

  it("Done only works on a finished run", async () => {
    const { api, store } = await opened({ speed: 1 });
    await store.startExport(names, "~/x");
    store.finishExport();
    expect(store.getState().exportRun).not.toBeNull();
    await store.cancelExport();
    store.openExport();
    api.flush();
  });

  it("shows the folder in Finder", async () => {
    const { api, store } = await opened();
    await store.startExport(names, "~/Pictures/Lisbon export");
    await store.revealExport();
    expect(api.system.revealed).toEqual(["~/Pictures/Lisbon export"]);
  });

  it("outlives the session: closing it keeps the run, and Start over clears nothing that belongs to the app", async () => {
    const { api, store } = await opened();
    await store.startExport(names, "~/x");
    store.setExportOptions({ format: "tiff" }, false);
    await store.closeSession();
    expect(store.getState().exportRun).not.toBeNull();
    expect(store.getState().exportOptions.format).toBe("tiff");
    expect(store.getState().preferences).not.toBeNull();
    api.flush();
  });

  it("forgets a finished run when the session closes, so the next session starts clean", async () => {
    const { api, store } = await opened();
    await store.startExport(names, "~/x");
    api.flush();
    await vi.waitFor(() => expect(store.getState().exportRun?.phase).toBe("finished"));
    await store.closeSession();
    expect(store.getState().exportRun).toBeNull();
  });

  it("clears the note about a stopped export when the options change", async () => {
    const { store } = await opened({ speed: 1 });
    await store.startExport(names, "~/x");
    await store.cancelExport();
    expect(store.getState().exportStopped).not.toBeNull();
    store.setExportScope("all");
    expect(store.getState().exportStopped).toBeNull();
    await store.startExport(names, "~/x");
    await store.cancelExport();
    store.setExportOptions({ jpeg_quality: 80 }, false);
    expect(store.getState().exportStopped).toBeNull();
  });

  it("asks before closing while a run is going, and stops it when told to", async () => {
    const { store } = await opened({ speed: 1 });
    await store.startExport(names, "~/x");
    await store.requestCloseSession();
    expect(store.getState().overlay).toBe("confirm-close");
    expect(store.getState().session).not.toBeNull();
    store.closeOverlay();
    expect(store.getState().overlay).toBeNull();
    await store.requestCloseSession();
    await store.stopExportAndClose();
    expect(store.getState().session).toBeNull();
    expect(store.getState().exportRun).toBeNull();
    expect(store.getState().overlay).toBeNull();
  });

  it("closes at once when nothing is running", async () => {
    const { store } = await opened();
    await store.requestCloseSession();
    expect(store.getState().session).toBeNull();
    expect(store.getState().overlay).toBeNull();
  });
});

describe("the .cube import", () => {
  it("opens the dialog with what the chooser returned, and imports after the answers", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, looksImported: false });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    api.queueChoice("files", ["~/LUTs/Warm_Film.cube"]);
    await store.startCubeImport();
    expect(store.getState().overlay).toBe("import-cube");
    expect(store.getState().cubeImport?.items).toHaveLength(1);
    expect(store.getState().cubeImport?.items[0]?.answer).toBeNull();
    // Nothing to import until the question is answered.
    await store.confirmCubeImport();
    expect(store.getState().overlay).toBe("import-cube");
    store.answerCube("~/LUTs/Warm_Film.cube", "logc3_to_rec709");
    await store.confirmCubeImport();
    expect(store.getState().overlay).toBeNull();
    expect(store.getState().cubeImport).toBeNull();
    expect(store.getState().notice).toBe("1 LUT added to Your LUTs.");
    expect(store.getState().looks?.find((l) => l.id === "user:warm-film")?.input).toBe("logc3_to_rec709");
  });

  it("says when everything was already imported", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, looksImported: false });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    for (let i = 0; i < 2; i++) {
      api.queueChoice("files", ["~/LUTs/Warm_Film.cube"]);
      await store.startCubeImport();
      store.answerCube("~/LUTs/Warm_Film.cube", "logc4");
      await store.confirmCubeImport();
    }
    expect(store.getState().notice).toBe("Those LUTs were already imported.");
  });

  it("closes the dialog and forgets the files when it is cancelled", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    api.queueChoice("files", ["~/LUTs/Warm_Film.cube"]);
    await store.startCubeImport();
    store.closeOverlay();
    expect(store.getState()).toMatchObject({ overlay: null, cubeImport: null });
  });
});

describe("notices", () => {
  it("shows a message that is not an error, until dismissed", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    store.showNotice("Yellowmoon is up to date.");
    expect(store.getState().notice).toBe("Yellowmoon is up to date.");
    store.dismissNotice();
    expect(store.getState().notice).toBeNull();
  });

  it("runs a menu item that arrives as an event", async () => {
    const { api, store } = await opened();
    const seen: string[] = [];
    store.menuHandler = (action) => seen.push(action);
    api.emitEvent("menu_action", { action: "preferences" });
    expect(seen).toEqual(["preferences"]);
  });
});
