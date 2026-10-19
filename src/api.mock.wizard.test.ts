// The mock's side of the ARRI wizard, the export extras and Preferences: synthetic packages, recognition, the simulated
// import, the Downloads watcher, the system commands, display LUTs, the estimate's extra fields and the preferences store.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockApi, FAILURE_ZIPS, LOOK_FILE_ZIPS, PACKAGE_FACTS, PACKAGE_ZIPS } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { AppError, DownloadFound, ExportJob, ExportOptions, ExportProgress, LooksImportProgress } from "./api/types";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW"];
const downloads = (name: string): string => `~/Downloads/${name}`;

async function rejection(promise: Promise<unknown>): Promise<AppError> {
  try {
    await promise;
  } catch (error) {
    return error as AppError;
  }
  throw new Error("expected the command to reject");
}

function job(scope: ExportJob["scope"], options: Partial<ExportOptions> = {}): ExportJob {
  return {
    scope,
    options: {
      format: "jpeg",
      jpeg_quality: 92,
      bit_depth: 16,
      tiff_compression: "lzw",
      color_space: "srgb",
      size: { kind: "full" },
      destination: null,
      name_template: "{name}_{look}",
      if_exists: "keep_both",
      ...options,
    },
  };
}

let api: MockApi;
beforeEach(() => {
  vi.useFakeTimers();
  api = createMockApi();
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

describe("recognising synthetic packages", () => {
  it("knows the five packages by their zips, with the counts of docs/08", async () => {
    const found = await Promise.all(Object.values(PACKAGE_ZIPS).map((name) => api.looksRecognise(downloads(name))));
    expect(found.map((f) => f.recognition.kind)).toEqual(["package", "package", "package", "package", "package"]);
    const ll4 = found[0]!;
    expect(ll4).toMatchObject({
      name: "ARRI_LookLibrary_LogC4.zip",
      source: "zip",
      size_bytes: 181_000_000,
      recognition: { package_id: "look_library_logc4", title: "Look Library · LogC4", what: "Look Library · LogC4 · log to log", contents: "87 .cube · 65-point", file_count: 87 },
    });
    expect(found.map((f) => (f.recognition.kind === "package" ? f.recognition.file_count : 0))).toEqual([87, 22, 87, 87, 48]);
  });

  it("finds a name by its last part only, ignoring case, so a dropped name works as well as a full path", async () => {
    expect((await api.looksRecognise("ARRI_LOOKLIBRARY_LOGC4.ZIP")).recognition.kind).toBe("package");
    expect((await api.looksRecognise("/Users/me/Downloads/ARRI_LookLibrary_LogC4.zip")).recognition.kind).toBe("package");
  });

  it("knows an unpacked folder", async () => {
    expect(await api.looksRecognise("~/Downloads/ARRI_LookLibrary_LogC4")).toMatchObject({ source: "folder", recognition: { kind: "package", package_id: "look_library_logc4" } });
  });

  it("explains a damaged look file in plain words", async () => {
    const found = await api.looksRecognise(downloads(FAILURE_ZIPS.damagedLookFile));
    expect(found).toMatchObject({ source: "file" });
    expect(found.recognition).toEqual({
      kind: "unusable",
      package_id: null,
      reason: "Damaged_Look.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again.",
    });
  });

  it("knows ARRI's look-file downloads as the looks of their package", async () => {
    const alf4 = await api.looksRecognise(downloads(LOOK_FILE_ZIPS.alf4Library));
    expect(alf4).toMatchObject({
      source: "zip",
      recognition: { kind: "package", package_id: "look_library_logc4", title: "Look Library · LogC4", contents: "87 .alf4 · 33-point", file_count: 87 },
    });
    const aml = await api.looksRecognise(downloads(LOOK_FILE_ZIPS.amlLibrary));
    expect(aml).toMatchObject({
      recognition: { kind: "package", package_id: "look_library_logc3_to_rec709", title: "Look Library · LogC3 to Rec.709", contents: "261 .aml · 33-point", file_count: 261 },
    });
  });

  it("reads a person's own look files without a question: each says what it expects", async () => {
    expect(await api.looksRecognise(downloads(FAILURE_ZIPS.ownAlf4))).toMatchObject({
      source: "file",
      recognition: { kind: "look_files", what: "ARRI Look Files · LogC4", contents: "1 .alf4 · 33-point", file_count: 1, input: "logc4" },
    });
    expect(await api.looksRecognise(downloads(FAILURE_ZIPS.ownLookFolder))).toMatchObject({
      source: "folder",
      recognition: { kind: "look_files", file_count: 3, input: "logc4" },
    });
    const alf4c = (await api.looksRecognise(downloads(FAILURE_ZIPS.ownAlf4c))).recognition;
    expect(alf4c.kind === "look_files" && alf4c.note).toContain("its own display transforms are not used");
    // A name the mock does not know, with a look file's extension, is a missing file, not a question.
    expect((await api.looksRecognise(downloads("Gone.aml"))).recognition.kind).toBe("missing");
  });

  it("explains an incomplete package and a zip with no LUTs", async () => {
    const incomplete = (await api.looksRecognise(downloads(FAILURE_ZIPS.incomplete))).recognition;
    expect(incomplete).toMatchObject({ kind: "unusable", package_id: "look_library_logc4" });
    expect(incomplete.kind === "unusable" && incomplete.reason).toContain("should have 87 files, but this zip has 79");
    const none = (await api.looksRecognise(downloads(FAILURE_ZIPS.noLuts))).recognition;
    expect(none.kind === "unusable" && none.reason).toContain("no .cube, .alf4, .alf4c or .aml files");
  });

  it("asks the generic question for ARRI cubes of another camera and for a loose .cube", async () => {
    expect((await api.looksRecognise(downloads(FAILURE_ZIPS.otherCamera))).recognition).toMatchObject({ kind: "needs_answer", file_count: 24, lut_size: 33 });
    expect(await api.looksRecognise(downloads(FAILURE_ZIPS.ownCube))).toMatchObject({ source: "file", recognition: { kind: "needs_answer", file_count: 1 } });
  });

  it("says a file is missing when it is not known, and after it is taken away", async () => {
    expect((await api.looksRecognise(downloads("gone.zip"))).recognition).toMatchObject({ kind: "missing", reason: "That file is not there any more. Download it again, or choose it again." });
    api.removeSource(PACKAGE_ZIPS.look_library_logc4);
    expect((await api.looksRecognise(downloads(PACKAGE_ZIPS.look_library_logc4))).recognition.kind).toBe("missing");
    api.addSource("Mine.cube", { source: "file", size_bytes: 10, recognition: { kind: "needs_answer", file_count: 1, lut_size: 17, suggestion: "logc4", note: "x" } });
    expect((await api.looksRecognise("~/Mine.cube")).recognition).toMatchObject({ kind: "needs_answer", suggestion: "logc4" });
  });
});

describe("importing", () => {
  it("copies about thirty files a second at speed 1 and reports each step", async () => {
    api.reset({ looksImported: false });
    const events: LooksImportProgress[] = [];
    api.on("looks_import_progress", (e) => events.push(e));
    const importing = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc4));
    await vi.advanceTimersByTimeAsync(150);
    expect(events[0]).toMatchObject({ path: downloads("ARRI_LookLibrary_LogC4.zip"), done: 3, total: 87 });
    await vi.advanceTimersByTimeAsync(3000);
    const result = await importing;
    expect(events.at(-1)).toMatchObject({ done: 87, total: 87 });
    expect(result.added).toHaveLength(87);
    expect(result.added[0]).toMatchObject({ id: "arri:3000:film-warm", name: "Film warm", input: "logc4" });
    expect(result.display_luts_added).toBe(0);
  });

  it("names the seven looks of the mockup first and groups the library", async () => {
    api.reset({ looksImported: false });
    const importing = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc4));
    await vi.advanceTimersByTimeAsync(4000);
    await importing;
    const arri = (await api.looksList()).filter((l) => l.group.kind === "arri");
    expect(arri.slice(0, 7).map((l) => l.name)).toEqual(["Film warm", "Film cool", "Bleach bypass", "Teal and orange", "Soft print", "Night blue", "Monochrome"]);
    expect(arri).toHaveLength(87);
    expect(new Set(arri.map((l) => l.id)).size).toBe(87);
  });

  it("holds at the file count it is asked to until flush() finishes it", async () => {
    api.reset({ looksImported: false, importHoldAt: 40 });
    const events: LooksImportProgress[] = [];
    api.on("looks_import_progress", (e) => events.push(e));
    const importing = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc4));
    await vi.advanceTimersByTimeAsync(10_000);
    expect(events.at(-1)).toMatchObject({ done: 40, total: 87 });
    api.flush();
    expect((await importing).added).toHaveLength(87);
    expect(events.at(-1)).toMatchObject({ done: 87, total: 87 });
  });

  it("adds ARRI's display LUTs, which the wizard offers as the display rendering", async () => {
    expect(await api.looksDisplayLuts()).toEqual([]);
    const importing = api.looksImport(downloads(PACKAGE_ZIPS.display_luts_logc4));
    await vi.advanceTimersByTimeAsync(4000);
    const result = await importing;
    expect(result).toMatchObject({ added: [], display_luts_added: 22 });
    const luts = await api.looksDisplayLuts();
    expect(luts).toHaveLength(22);
    expect(luts[0]).toEqual({ id: "arri-display:logc4:rec709-g24", name: "Rec.709 · Gamma 2.4", input: "logc4", target: "rec709-g24" });
    await api.preferencesSet({ display_rendering: { kind: "arri_display", id: luts[0]!.id } });
    expect((await api.preferencesGet()).display_rendering).toEqual({ kind: "arri_display", id: "arri-display:logc4:rec709-g24" });
  });

  it("refuses a display rendering that is not imported", async () => {
    const error = await rejection(api.preferencesSet({ display_rendering: { kind: "arri_display", id: "arri-display:logc4:nope" } }));
    expect(error.scope).toBe("preferences");
  });

  it("gives LogC3 looks their own ids and inputs", async () => {
    api.reset({ looksImported: false });
    const a = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc3));
    const b = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc3_to_rec709));
    await vi.advanceTimersByTimeAsync(4000);
    await Promise.all([a, b]);
    const arri = (await api.looksList()).filter((l) => l.group.kind === "arri");
    expect(arri).toHaveLength(174);
    expect(new Set(arri.map((l) => l.input))).toEqual(new Set(["logc3_ei800", "logc3_to_rec709"]));
  });

  it("refuses what cannot be imported, with the explanation", async () => {
    const alf = await rejection(api.looksImport(downloads(FAILURE_ZIPS.damagedLookFile)));
    expect(alf).toEqual({ scope: "looks", message: "Damaged_Look.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again." });
    const gone = await rejection(api.looksImport(downloads("gone.zip")));
    expect(gone.message).toBe("That file is not there any more. Download it again, or choose it again.");
    const unanswered = await rejection(api.looksImport(downloads(FAILURE_ZIPS.ownCube)));
    expect(unanswered.message).toContain("Say what the LUT expects as input");
  });

  it("adds a person's own look file to Your LUTs with the input the file says, without being asked", async () => {
    api.reset({ looksImported: false });
    const importing = api.looksImport(downloads(FAILURE_ZIPS.ownAlf4));
    await vi.advanceTimersByTimeAsync(4000);
    const result = await importing;
    expect(result.added).toHaveLength(1);
    expect(result.added[0]).toMatchObject({ id: "user:warm-pull", name: "Warm Pull", group: { kind: "user" }, input: "logc4" });
    // The answer to the generic question does not change what a look file says.
    const again = api.looksImport(downloads(FAILURE_ZIPS.ownAlf4), "logc3_ei800");
    await vi.advanceTimersByTimeAsync(4000);
    expect((await again).added).toEqual([]);
    expect((await api.looksList()).find((l) => l.id === "user:warm-pull")?.input).toBe("logc4");
  });

  it("adds a loose .cube to Your LUTs under the name of the file", async () => {
    const importing = api.looksImport(downloads(FAILURE_ZIPS.ownCube), "logc3_ei800");
    await vi.advanceTimersByTimeAsync(500);
    const result = await importing;
    expect(result.added).toEqual([
      {
        id: "user:warm-film",
        name: "Warm Film",
        group: { kind: "user" },
        input: "logc3_ei800",
        library: "user",
        format: "cube",
        lut_size: 33,
        is_default: false,
        missing: false,
      },
    ]);
  });

  it("imports the same package twice without duplicating anything", async () => {
    const first = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc4));
    await vi.advanceTimersByTimeAsync(4000);
    await first;
    const total = (await api.looksList()).length;
    const again = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc4));
    await vi.advanceTimersByTimeAsync(4000);
    expect(await again).toMatchObject({ added: [], unchanged: 87 });
    expect(await api.looksList()).toHaveLength(total);
  });

  it("falls back to Neutral when the default look is no longer in the library", async () => {
    await api.looksSetDefault("arri:3000:film-cool");
    api.reset({ looksImported: false });
    expect((await api.preferencesGet()).default_look).toBe("builtin:neutral");
  });
});

describe("watching Downloads", () => {
  it("sends a download_found for a zip that lands, only while it is watching", async () => {
    const found: DownloadFound[] = [];
    api.on("download_found", (e) => found.push(e));
    expect(api.simulateDownload(PACKAGE_ZIPS.look_library_logc4)).toBe(false);
    expect(await api.looksWatchDownloads(true)).toEqual({ watching: true, problem: null });
    expect(api.simulateDownload(PACKAGE_ZIPS.look_library_logc4)).toBe(true);
    expect(found).toEqual([{ path: downloads("ARRI_LookLibrary_LogC4.zip"), file_name: "ARRI_LookLibrary_LogC4.zip", size_bytes: 181_000_000, package_hint: "look_library_logc4" }]);
    expect(await api.looksWatchDownloads(false)).toEqual({ watching: false, problem: null });
    expect(api.simulateDownload(PACKAGE_ZIPS.look_library_logc4)).toBe(false);
  });

  it("delivers the zips named in the options one after the other", async () => {
    api.reset({ downloads: [PACKAGE_ZIPS.look_library_logc4, PACKAGE_ZIPS.display_luts_logc4] });
    const found: string[] = [];
    api.on("download_found", (e) => found.push(e.file_name));
    await api.looksWatchDownloads(true);
    await vi.advanceTimersByTimeAsync(2600);
    expect(found).toEqual(["ARRI_LookLibrary_LogC4.zip"]);
    await vi.advanceTimersByTimeAsync(2600);
    expect(found).toEqual(["ARRI_LookLibrary_LogC4.zip", "ARRI_DisplayLUTs_LogC4.zip"]);
    await api.looksWatchDownloads(false);
  });

  it("says why macOS did not allow it", async () => {
    api.reset({ downloadsDenied: true });
    const status = await api.looksWatchDownloads(true);
    expect(status.watching).toBe(false);
    expect(status.problem).toContain("Downloads folder");
    expect(status.problem).toContain("choose the file yourself");
  });
});

describe("the system", () => {
  it("answers the chooser with what was queued, then with its default", async () => {
    expect(await api.chooseFolder(null)).toBe("~/Pictures/Chosen export");
    api.queueChoice("folder", "~/Exports");
    api.queueChoice("folder", null);
    expect(await api.chooseFolder("~/x")).toBe("~/Exports");
    expect(await api.chooseFolder("~/x")).toBeNull();
    expect(await api.chooseLookFiles()).toEqual([downloads("ARRI_LookLibrary_LogC4.zip")]);
    api.queueChoice("files", [downloads("a.zip"), downloads("b.zip")]);
    api.queueChoice("files", null);
    expect(await api.chooseLookFiles()).toHaveLength(2);
    expect(await api.chooseLookFiles()).toBeNull();
  });

  it("records pages opened, folders revealed and files trashed, and a trashed zip is gone", async () => {
    await api.openUrl("https://www.arri.com/en/learn/camera-systems/image-science/look-files");
    await api.revealInFinder("~/Pictures/Lisbon export");
    await api.trashPath(downloads(PACKAGE_ZIPS.look_library_logc4));
    expect(api.system).toEqual({
      opened: ["https://www.arri.com/en/learn/camera-systems/image-science/look-files"],
      revealed: ["~/Pictures/Lisbon export"],
      trashed: [downloads("ARRI_LookLibrary_LogC4.zip")],
      restarts: 0,
    });
    expect((await api.looksRecognise(downloads(PACKAGE_ZIPS.look_library_logc4))).recognition.kind).toBe("missing");
  });

  it("tracks the command-line link and reports the version", async () => {
    expect(await api.cliLinkStatus()).toEqual({ installed: false, path: null });
    expect(await api.installCliLink()).toEqual({ path: "/usr/local/bin/ym" });
    expect(await api.cliLinkStatus()).toEqual({ installed: true, path: "/usr/local/bin/ym" });
    expect(await api.appInfo()).toEqual({ version: expect.stringMatching(/^\d+\.\d+\.\d+/), pipeline_version: "1.0.0" });
  });
});

describe("the estimate's extra fields", () => {
  beforeEach(async () => {
    api.reset({ picked: [2, 3], current: 3 });
    await api.openPaths(ARW);
  });

  it("counts photos with adjustments beyond the look", async () => {
    expect((await api.exportEstimate(job("picked"))).adjusted_photos).toBe(0);
    // The current photo is DSC04204 (id 4), and it is picked with DSC04203 (id 3).
    await api.updateSettings(4, { contrast: manual(12) });
    await api.updateSettings(3, { look: "builtin:signature", look_intensity: 60 });
    expect((await api.exportEstimate(job("picked"))).adjusted_photos).toBe(1);
    expect((await api.exportEstimate(job("this"))).adjusted_photos).toBe(1);
    expect((await api.exportEstimate(job("all"))).adjusted_photos).toBe(1);
    await api.setCurrent(3);
    expect((await api.exportEstimate(job("this"))).adjusted_photos).toBe(0);
  });

  it("says a DNG is approximate for a look with its own display rendering, and still starts one", async () => {
    const importing = api.looksImport(downloads(PACKAGE_ZIPS.look_library_logc3_to_rec709));
    await vi.advanceTimersByTimeAsync(4000);
    await importing;
    expect((await api.exportEstimate(job("this"))).dng_approximate).toBe(false);
    await api.applyLook("this", "arri:3000:film-warm-logc3-rec709", 100);
    expect((await api.exportEstimate(job("this"))).dng_approximate).toBe(true);
    expect((await api.exportEstimate(job("picked"))).dng_approximate).toBe(true);
    await api.setCurrent(1);
    expect((await api.exportEstimate(job("this"))).dng_approximate).toBe(false);
    // At an intensity of 0 the picture is our own rendering.
    await api.setCurrent(4);
    await api.applyLook("this", "arri:3000:film-warm-logc3-rec709", 0);
    expect((await api.exportEstimate(job("this"))).dng_approximate).toBe(false);
    await api.applyLook("this", "arri:3000:film-warm-logc3-rec709", 100);
    expect(await api.exportStart(job("this", { format: "dng" }))).toBe(1);
    expect(await api.exportStart(job("this", { format: "tiff" }))).toBe(2);
  });
});

describe("export outcomes", () => {
  async function runJob(j: ExportJob, options = {}): Promise<ExportProgress[]> {
    api.reset({ picked: [2, 3, 4], current: 3, ...options });
    await api.openPaths(ARW);
    const events: ExportProgress[] = [];
    api.on("export_progress", (e) => events.push(e));
    await api.exportStart(j);
    await vi.advanceTimersByTimeAsync(6000);
    return events;
  }

  it("holds at a progress until flush() carries it on, as the running screenshot needs", async () => {
    const events = await runJob(job("picked"), { exportHoldAt: 1.45 });
    expect(events.at(-1)).toMatchObject({ file: "DSC04204_neutral.jpg", done: 1, total: 3, file_percent: 45 });
    api.flush();
    expect(events.at(-1)).toMatchObject({ done: 3, total: 3, file_percent: 100 });
  });

  it("keeps both by numbering a file that already exists", async () => {
    const events = await runJob(job("picked"), { existingExports: ["DSC04204_neutral.jpg"] });
    expect([...new Set(events.map((e) => e.file))]).toEqual(["DSC04203_neutral.jpg", "DSC04204_neutral 2.jpg", "DSC04205_neutral.jpg"]);
  });

  it("replaces by keeping the name, and skips by reporting the file once as skipped", async () => {
    const replace = await runJob(job("picked", { if_exists: "replace" }), { existingExports: ["DSC04204_neutral.jpg"] });
    expect(replace.some((e) => e.file === "DSC04204_neutral.jpg")).toBe(true);
    expect(replace.some((e) => e.skipped)).toBe(false);
    const skip = await runJob(job("picked", { if_exists: "skip" }), { existingExports: ["DSC04204_neutral.jpg"] });
    const skipped = skip.filter((e) => e.skipped);
    expect(skipped).toHaveLength(1);
    expect(skipped[0]).toMatchObject({ file: "DSC04204_neutral.jpg", done: 2, total: 3, error: null });
  });

  it("reports a failing photo once with its reason, and carries on", async () => {
    const events = await runJob(job("picked"), { exportFailures: { DSC04204: "The disk is full. Free some space, then export again." } });
    const failed = events.filter((e) => e.error);
    expect(failed).toHaveLength(1);
    expect(failed[0]).toMatchObject({ file: "DSC04204_neutral.jpg", done: 2, total: 3, error: "The disk is full. Free some space, then export again." });
    expect(events.at(-1)).toMatchObject({ done: 3, total: 3, file_percent: 100, error: null });
  });

  it("leaves a file that can't be developed out of All", async () => {
    api.reset({ skipped: { DSC04207: "Can't read this." } });
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(6000);
    expect((await api.exportEstimate(job("all"))).bytes).toBe(Math.round(24.9 * 11 * 1e6));
  });
});

describe("preferences", () => {
  it("keeps everything Preferences shows, and merges patches", async () => {
    const prefs = await api.preferencesSet({ remove_location_from_exports: true, watch_downloads_during_import: true, check_for_updates_automatically: false });
    expect(prefs).toMatchObject({ remove_location_from_exports: true, watch_downloads_during_import: true, check_for_updates_automatically: false, default_look: "builtin:neutral" });
    const defaults = { ...prefs.export_defaults, format: "tiff" as const, destination: "~/Exports" };
    expect((await api.preferencesSet({ export_defaults: defaults })).export_defaults).toMatchObject({ format: "tiff", destination: "~/Exports" });
    expect((await api.preferencesGet()).remove_location_from_exports).toBe(true);
  });

  it("survives a reload when asked to keep them", async () => {
    const store = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    });
    try {
      const first = createMockApi({ persistPreferences: true, looksImported: false });
      await first.preferencesSet({ remove_location_from_exports: true });
      const importing = first.looksImport(downloads(PACKAGE_ZIPS.look_library_logc4));
      await vi.advanceTimersByTimeAsync(4000);
      await importing;
      await first.looksSetDefault("arri:3000:film-cool");
      await first.installCliLink();
      first.reset({ persistPreferences: true, looksImported: false });
      // reset() clears what was kept, as "restore everything" says.
      expect((await first.preferencesGet()).remove_location_from_exports).toBe(false);
      await first.preferencesSet({ remove_location_from_exports: true });
      const again = createMockApi({ persistPreferences: true, looksImported: false });
      expect((await again.preferencesGet()).remove_location_from_exports).toBe(true);
      expect(store.size).toBe(1);
      again.reset();
      expect(store.size).toBe(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("starts from the defaults when nothing is kept", async () => {
    expect((await createMockApi().preferencesGet()).remove_location_from_exports).toBe(false);
  });
});

describe("package facts", () => {
  it("match the mockup's choose step", () => {
    expect(PACKAGE_FACTS.look_library_logc4).toMatchObject({ fileCount: 87, megabytes: 181, contents: "87 .cube · 65-point" });
    expect(PACKAGE_FACTS.display_luts_logc4).toMatchObject({ fileCount: 22, megabytes: 32 });
    expect(PACKAGE_FACTS.display_luts_logc3).toMatchObject({ fileCount: 48, megabytes: 46 });
  });
});
